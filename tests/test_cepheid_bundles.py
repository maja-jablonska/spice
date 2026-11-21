"""spice.cepheid_bundles: build/evaluate round trip, unit handling and pickling."""
import jax.numpy as jnp
import numpy as np
import pytest

from spice.cepheid_bundles import (CepheidBundle, LineSpectra, apply_phase_params,
                                   build_bundle, load_pickle, save_pickle,
                                   simulate_line_spectra)
from spice.models import IcosphereModel
from spice.models.mesh_transform import add_rotation, evaluate_rotation
from spice.spectrum import Blackbody

P_DAYS = 2.0
TIMES = [0.0, 0.5, 1.0]          # days
AMP = 0.02                       # fraction of the radius


@pytest.fixture(scope="module")
def bb():
    return Blackbody()


@pytest.fixture(scope="module")
def bundle(bb):
    return build_bundle(bb, bb.solar_parameters,
                        fourier_params=jnp.array([[AMP, 0.0]]), period=P_DAYS,
                        timeseries=TIMES, n_mesh=100, radius=2.0, mass=1.0,
                        rotation_velocity=20.0, max_fourier_order=1)


def _mean_radius(m):
    return float(np.mean(np.linalg.norm(np.asarray(m.d_centers) + np.asarray(m.center_pulsation_offsets), axis=1)))


def test_bundle_structure(bundle):
    assert isinstance(bundle, CepheidBundle)
    assert len(bundle.snapshots) == len(TIMES)
    assert bundle.template is not None and bundle.pulsating is not None


def test_pulsation_follows_the_fourier_series_in_days(bundle):
    # Radius displacement A cos(2 pi t / P): +A at t = 0, -A at t = P/2 (= 1 day).
    r0 = _mean_radius(bundle.template)
    r_t0 = _mean_radius(bundle.snapshots[0])
    r_half = _mean_radius(bundle.snapshots[2])
    np.testing.assert_allclose((r_t0 - r0) / r0, AMP, rtol=0.05)
    np.testing.assert_allclose((r_half - r0) / r0, -AMP, rtol=0.05)


def test_rotation_time_is_converted_from_days_to_seconds(bb):
    # With zero pulsation amplitude a snapshot at t days must equal the plain
    # rotation evaluated at t * 86400 s.
    b = build_bundle(bb, bb.solar_parameters, fourier_params=jnp.array([[0.0, 0.0]]),
                     period=P_DAYS, timeseries=[1.0], n_mesh=100, radius=2.0, mass=1.0,
                     rotation_velocity=20.0, max_fourier_order=1)
    base = IcosphereModel.construct(100, 2.0, 1.0, bb.solar_parameters, bb.parameter_names,
                                    max_fourier_order=1)
    ref = evaluate_rotation(add_rotation(base, 20.0), 86400.0)
    np.testing.assert_allclose(np.asarray(b.snapshots[0].d_centers), np.asarray(ref.d_centers),
                               atol=1e-9)
    # And it really moved (the conversion is not a no-op).
    assert np.max(np.abs(np.asarray(ref.d_centers) - np.asarray(base.d_centers))) > 1e-2


def test_apply_phase_params_sets_values_per_snapshot(bundle):
    teff = {0.0: 5000.0, 0.5: 5500.0, 1.0: 6000.0}
    out = apply_phase_params(bundle, TIMES, [(0, lambda ph: teff[ph])])
    for snap, t in zip(out.snapshots, TIMES):
        np.testing.assert_allclose(np.asarray(snap.parameters)[:, 0], teff[t])
    # The input bundle is unchanged.
    np.testing.assert_allclose(np.asarray(bundle.snapshots[0].parameters)[:, 0], 5777.0)


def test_apply_phase_params_needs_snapshots():
    with pytest.raises(ValueError):
        apply_phase_params(CepheidBundle(), [0.0], [(0, lambda ph: 1.0)])


def test_simulate_line_spectra(bundle, bb):
    out = simulate_line_spectra(bundle.snapshots, bundle.template, bb.intensity,
                                [5000.0], line_width=1.0, steps=20)
    (ls,) = out.values()
    assert isinstance(ls, LineSpectra) and list(out) == [5000.0]
    np.testing.assert_allclose(ls.wavelengths[[0, -1]], [4999.0, 5001.0])
    assert len(ls.spectra) == len(TIMES)
    assert all(s.shape == (20, 2) for s in ls.spectra) and ls.template.shape == (20, 2)
    # A bigger star at t = 0 (radius +2 %) is brighter than at t = P/2 (-2 %).
    assert ls.spectra[0][:, 0].mean() > ls.spectra[2][:, 0].mean()


def test_pickle_round_trip(bundle, tmp_path):
    path = tmp_path / "bundle.pkl"
    save_pickle(bundle, str(path))
    assert path.exists() and not list(tmp_path.glob("*.tmp.*"))
    back = load_pickle(str(path))
    assert isinstance(back, CepheidBundle)
    np.testing.assert_allclose(np.asarray(back.snapshots[1].d_centers),
                               np.asarray(bundle.snapshots[1].d_centers))
