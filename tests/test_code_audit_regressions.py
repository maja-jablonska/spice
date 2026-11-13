"""Regression tests for the 2026-11 code audit (follow-up to the physics audit).

* Grid interpolators did not clamp a query that lies off the grid on two or
  more axes: at (-1, -1) on a 3x3 grid 20% of the weight went to the interior
  node (1, 1), because the product of two negative-side weights is positive.
* ``UserSpectrumInterpolator.solar_parameters`` raised ValueError (it passed
  the generic ``v_micro`` key to ``to_parameters``).
* ``LazyZarrInterpolator`` raised KeyError b'teff' for stores whose
  ``param_names`` are byte strings.
* A pulsation mode with ``l >= max_pulsation_mode`` only warned and then did
  nothing.
* ``add_spherical_harmonic_spots`` silently ignored a half-specified tilt.
"""
import itertools
import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import zarr

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_pulsation, add_pulsations, evaluate_pulsations
from spice.models.spots import add_spherical_harmonic_spots
from spice.spectrum.lazy_zarr_interpolator import GridIndex, LazyZarrInterpolator, SparseGridIndex
from spice.spectrum.solar_parameters import LOGG, TEFF, VMICRO
from spice.spectrum.user_spectrum_interpolator import UserSpectrumInterpolator


# --- off-grid clamping --------------------------------------------------------

def _grid_3x3_df():
    rows = list(itertools.product([0., 1., 2.], [0., 1., 2.]))
    return pd.DataFrame({"a": [r[0] for r in rows], "b": [r[1] for r in rows],
                         "row_idx": np.arange(len(rows), dtype=np.int32)})


@pytest.mark.parametrize("index_cls", [SparseGridIndex, GridIndex])
@pytest.mark.parametrize("query, node", [
    ((-1., -1.), (0., 0.)),     # off the low corner on both axes
    ((3., 3.), (2., 2.)),       # off the high corner on both axes
    ((-1., 3.), (0., 2.)),      # mixed corners
])
def test_off_grid_query_on_two_axes_clamps_to_the_corner_node(index_cls, query, node):
    df = _grid_3x3_df()
    idx = index_cls.from_dataframe(df, ["a", "b"])
    rows, weights = idx.lookup(jnp.array(query))
    rows, weights = np.asarray(rows), np.asarray(weights)
    expected_row = int(df[(df.a == node[0]) & (df.b == node[1])].row_idx.iloc[0])
    total = {}
    for r, w in zip(rows, weights):
        if r >= 0 and w > 0:
            total[int(r)] = total.get(int(r), 0.) + float(w)
    assert total == pytest.approx({expected_row: 1.0})


@pytest.mark.parametrize("index_cls", [SparseGridIndex, GridIndex])
def test_off_grid_on_one_axis_clamps_and_interpolates_the_other(index_cls):
    df = _grid_3x3_df()
    idx = index_cls.from_dataframe(df, ["a", "b"])
    rows, weights = idx.lookup(jnp.array([-1., 0.5]))
    got = {int(r): float(w) for r, w in zip(np.asarray(rows), np.asarray(weights)) if r >= 0 and w > 0}
    row = lambda a, b: int(df[(df.a == a) & (df.b == b)].row_idx.iloc[0])  # noqa: E731
    assert got == pytest.approx({row(0., 0.): 0.5, row(0., 1.): 0.5})


# --- UserSpectrumInterpolator.solar_parameters --------------------------------

@pytest.fixture
def small_user_grid(tmp_path):
    params, spectra = [], []
    wl = np.linspace(4990., 5010., 11)
    for t, g, m in itertools.product([5000., 6000.], [4.0, 4.5], [-0.5, 0.0]):
        params.append([t, g, m])
        spectra.append(1e-3 * t + g + m + 0 * wl)
    path = tmp_path / "grid.pkl"
    with open(path, "wb") as f:
        pickle.dump({"parameters": np.array(params), "intensities": np.array(spectra),
                     "log10_wavelengths": np.log10(wl)}, f)
    return str(path)


def test_user_interpolator_solar_parameters_default_names(small_user_grid):
    emu = UserSpectrumInterpolator(small_user_grid, has_continuum=False)
    np.testing.assert_allclose(np.asarray(emu.solar_parameters), [TEFF, LOGG, 0.0])


def test_user_interpolator_solar_parameters_follow_own_names(small_user_grid):
    emu = UserSpectrumInterpolator(small_user_grid, parameter_names=["logg", "Teff", "feh"],
                                   has_continuum=False)
    np.testing.assert_allclose(np.asarray(emu.solar_parameters), [LOGG, TEFF, 0.0])


def test_user_interpolator_solar_parameters_with_vmicro_column(tmp_path):
    params, spectra = [], []
    wl = np.linspace(4990., 5010., 5)
    for t, v in itertools.product([5000., 6000.], [1.0, 3.0]):
        params.append([t, v])
        spectra.append(1e-3 * t + v + 0 * wl)
    path = tmp_path / "grid_v.pkl"
    with open(path, "wb") as f:
        pickle.dump({"parameters": np.array(params), "intensities": np.array(spectra),
                     "log10_wavelengths": np.log10(wl)}, f)
    emu = UserSpectrumInterpolator(str(path), parameter_names=["teff", "vmicro"], has_continuum=False)
    np.testing.assert_allclose(np.asarray(emu.solar_parameters), [TEFF, VMICRO])


# --- byte-string param_names in zarr stores -----------------------------------

def test_lazy_zarr_interpolator_decodes_byte_string_param_names(tmp_path):
    store = tmp_path / "grid.zarr"
    group = zarr.open_group(str(store), mode="w")
    wl = np.array([5000., 5001., 5002.], dtype=np.float32)
    group.create_array("wavelength", data=wl)
    rows = list(itertools.product([5000., 6000.], [4.0, 4.5]))
    flux = np.array([[1e-3 * t + g] * len(wl) for t, g in rows], dtype=np.float32)
    group.create_array("flux", data=flux)
    group.create_array("continuum", data=flux + 1.)
    group.create_array("param_names", data=np.asarray(["teff", "logg"], dtype="S"))
    pd.DataFrame({"teff": [r[0] for r in rows], "logg": [r[1] for r in rows],
                  "row_idx": np.arange(len(rows), dtype=np.int32)}).to_parquet(f"{store}/index.parquet")

    interp = LazyZarrInterpolator(str(store))
    assert interp.stellar_parameter_names == ["teff", "logg"]
    assert all(type(n) is str for n in interp.stellar_parameter_names)
    np.testing.assert_allclose(np.asarray(interp.solar_parameters), [TEFF, LOGG])


# --- pulsation degree beyond max_pulsation_mode -------------------------------

def _pulsation_mesh(max_mode=5):
    return IcosphereModel.construct(100, 1., 1., [5777.], ["teff"], max_pulsation_mode=max_mode)


def test_add_pulsation_rejects_degree_beyond_max_pulsation_mode():
    m = _pulsation_mesh()
    with pytest.raises(ValueError, match="max_pulsation_mode"):
        add_pulsation(m, 0, 5, 1.0, jnp.array([[0.1, 0.0]]))


def test_add_pulsations_rejects_degree_beyond_max_pulsation_mode():
    m = _pulsation_mesh()
    with pytest.raises(ValueError, match="max_pulsation_mode"):
        add_pulsations(m, jnp.array([0, 0]), jnp.array([1, 5]), jnp.array([1.0, 1.0]),
                       jnp.array([[[0.1, 0.0]], [[0.1, 0.0]]]))


def test_degree_five_works_with_a_larger_max_pulsation_mode():
    m = add_pulsation(_pulsation_mesh(max_mode=6), 0, 5, 1.0, jnp.array([[0.1, 0.0]]))
    offsets = np.asarray(evaluate_pulsations(m, 0.0).center_pulsation_offsets)
    assert np.max(np.abs(offsets)) > 1e-3


# --- half-specified spot tilt -------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"tilt_axes": jnp.array([[1., 0., 0.]])},
    {"tilt_angles": jnp.array([30.])},
])
def test_spherical_harmonic_spots_reject_half_specified_tilt(kwargs):
    m = IcosphereModel.construct(100, 1., 1., [5777.], ["teff"])
    with pytest.raises(ValueError, match="tilt_axes and tilt_angles"):
        add_spherical_harmonic_spots(m, jnp.array([0]), jnp.array([1]), jnp.array([100.]),
                                     jnp.array([0]), **kwargs)
