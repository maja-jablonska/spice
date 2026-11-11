"""Regression tests for the 2026-11 grid/interpolator audit.

* SparseGridIndex encoded each node as one int64 key; without jax_enable_x64
  JAX narrows that to int32, so on grids of more than 2**31 cells the keys
  wrapped, were no longer sorted, and most nodes were silently not found.
* UserSpectrumInterpolator used inverse-distance weighting over every grid row
  (Teff 5250 on a linear grid gave 25.71 instead of 23.85); it now interpolates
  multilinearly, and ``to_parameters(dict)`` follows the grid's column order.
* Solar defaults only matched the exact keys teff/logg/v_micro, so marcs_logg,
  vmicro, ... defaulted to 0.0.
"""
import itertools
import pickle
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from spice.spectrum.lazy_zarr_interpolator import SparseGridIndex
from spice.spectrum.solar_parameters import LOGG, TEFF, VMICRO, solar_value
from spice.spectrum.user_spectrum_interpolator import UserSpectrumInterpolator


# --- sparse index -------------------------------------------------------------

def _big_sparse_df(n_axes=11, n_per_axis=8, n_rows=400, seed=0):
    rng = np.random.default_rng(seed)
    axes = [np.linspace(0., 1., n_per_axis) + i for i in range(n_axes)]
    idx = np.unique(rng.integers(0, n_per_axis, size=(n_rows, n_axes)), axis=0)
    data = {f"p{i}": axes[i][idx[:, i]] for i in range(n_axes)}
    data["row_idx"] = np.arange(len(idx), dtype=np.int32)
    return pd.DataFrame(data), [f"p{i}" for i in range(n_axes)]


def test_sparse_index_keys_fit_int32_columns_and_find_every_node():
    df, cols = _big_sparse_df()            # 8**11 = 8.6e9 cells > 2**31
    idx = SparseGridIndex.from_dataframe(df, cols)
    assert idx.keys.dtype == jnp.int32 and idx.keys.shape[1] >= 2
    rows, weights = idx.batched_lookup(jnp.asarray(df[cols].to_numpy()))
    best = np.asarray(rows)[np.arange(len(df)), np.argmax(np.asarray(weights), axis=1)]
    np.testing.assert_array_equal(best, df.row_idx.to_numpy())
    np.testing.assert_allclose(np.asarray(weights).max(axis=1), 1.0)


def test_sparse_index_works_without_x64():
    # Fresh interpreter: the test session enables x64 globally in conftest.
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src')!r})
        sys.path.insert(0, {str(__import__('pathlib').Path(__file__).resolve().parent)!r})
        import jax; assert not jax.config.jax_enable_x64
        import numpy as np, jax.numpy as jnp
        from test_grid_audit_regressions import _big_sparse_df
        from spice.spectrum.lazy_zarr_interpolator import SparseGridIndex
        df, cols = _big_sparse_df()
        idx = SparseGridIndex.from_dataframe(df, cols)
        rows, w = idx.batched_lookup(jnp.asarray(df[cols].to_numpy(), dtype=jnp.float32))
        missing = int((np.asarray(w).sum(axis=1) == 0).sum())
        print(missing)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "0"


# --- user interpolator --------------------------------------------------------

def _linear(teff, logg, mh, mu, wl):
    return 1e-3 * teff + 3 * logg - 7 * mh + 11 * mu + 0.01 * (wl - 5000.)


@pytest.fixture
def linear_grid(tmp_path):
    teffs, loggs, mhs, mus = [5000., 5500., 6000.], [3.5, 4.0, 4.5], [-0.5, 0.0], [0.2, 0.6, 1.0]
    wl = np.linspace(4990., 5010., 21)
    params, spectra = [], []
    for t, g, m, u in itertools.product(teffs, loggs, mhs, mus):
        params.append([t, g, m, u])
        spectra.append(_linear(t, g, m, u, wl))
    path = tmp_path / "grid.pkl"
    with open(path, "wb") as f:
        pickle.dump({"parameters": np.array(params), "intensities": np.array(spectra),
                     "log10_wavelengths": np.log10(wl)}, f)
    return str(path)


@pytest.mark.parametrize("teff,logg,mh,mu", [
    (5500., 4.0, 0.0, 0.6),      # node
    (5250., 4.0, 0.0, 0.6),      # off-node in Teff (IDW gave 25.71 vs 23.85)
    (5000., 3.75, 0.0, 0.6),     # off-node in log g
    (5730., 4.31, -0.21, 0.83),  # off-node in everything
])
def test_user_interpolator_is_exact_for_linear_grids(linear_grid, teff, logg, mh, mu):
    emu = UserSpectrumInterpolator(linear_grid, has_continuum=False)
    wl = jnp.array([4995.5, 5003.25])
    got = np.asarray(emu.intensity(jnp.log10(wl), mu, jnp.array([teff, logg, mh])))[:, 0]
    np.testing.assert_allclose(got, _linear(teff, logg, mh, mu, np.asarray(wl)), rtol=1e-6)


def test_user_interpolator_clamps_outside_the_grid(linear_grid):
    emu = UserSpectrumInterpolator(linear_grid, has_continuum=False)
    wl = jnp.log10(jnp.array([5000.]))
    out = np.asarray(emu.intensity(wl, 0.6, jnp.array([9000., 4.0, 0.0])))[0, 0]
    edge = np.asarray(emu.intensity(wl, 0.6, jnp.array([6000., 4.0, 0.0])))[0, 0]
    assert out == pytest.approx(edge)


def test_to_parameters_follows_the_grid_column_order(linear_grid):
    emu = UserSpectrumInterpolator(linear_grid, parameter_names=["logg", "teff", "feh"],
                                   has_continuum=False)
    np.testing.assert_allclose(np.asarray(emu.to_parameters({"teff": 5000., "logg": 4.0, "feh": -0.5})),
                               [4.0, 5000., -0.5])
    np.testing.assert_allclose(np.asarray(emu.to_parameters({"Teff": 5000., "m/h": -0.5})),
                               [4.44, 5000., -0.5])


# --- solar defaults -----------------------------------------------------------

@pytest.mark.parametrize("name,value", [
    ("teff", TEFF), ("Teff", TEFF), ("marcs_teff", TEFF), ("logg", LOGG), ("marcs_logg", LOGG),
    ("v_micro", VMICRO), ("vmicro", VMICRO), ("marcs_vmicro", VMICRO),
    ("feh", 0.0), ("marcs_fe_h", 0.0), ("c", 0.0),
])
def test_solar_value_recognises_emulator_input_names(name, value):
    assert solar_value(name) == value
