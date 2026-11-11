"""Regression tests for the 2026-11 audit: eclipse finder and toy line profiles.

* solve_kepler_E diverged for e >~ 0.92 (eclipses came back NaN);
* eclipse_timestamps_kepler dropped eclipses within one sample of periastron
  (omega = 90 deg) and defaulted to los [0,0,1], swapping primary/secondary;
* find_binary_eclipses missed an eclipse straddling the start of its window;
* the pseudo-Voigt Gaussian part had FWHM F/2, and the "1-sigma" thermal width
  was the Doppler parameter b = sqrt(2) sigma.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from spice.models import IcosphereModel
from spice.models.binary import Binary, add_orbit
from spice.models.eclipse_utils import find_binary_eclipses
from spice.models.orbit_utils import eclipse_timestamps_kepler, solve_kepler_E
from spice.spectrum.line_profile import _pseudo_voigt_shape
from spice.spectrum.physical_line_emulator import thermal_width

DAY = 1 / 365.25


def _conjunction_time(P, e, Tperi, omega, primary=True):
    """Analytic time when body 1 is behind (primary) / in front (secondary)."""
    nu = np.pi / 2 - omega + (0.0 if primary else np.pi)
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    M = E - e * np.sin(E)
    return (Tperi + M / (2 * np.pi) * P) % P


@pytest.mark.parametrize("e", [0.0, 0.5, 0.9, 0.92, 0.95, 0.99])
def test_kepler_solver_converges_for_all_eccentricities(e):
    M = jnp.linspace(-4 * np.pi, 4 * np.pi, 20001)
    E = solve_kepler_E(M, e)
    Mwrap = (M + np.pi) % (2 * np.pi) - np.pi
    assert float(jnp.max(jnp.abs(E - e * jnp.sin(E) - Mwrap))) < 1e-10


@pytest.mark.parametrize("e,omega_deg,inc_deg", [
    (0.0, 0.0, 89.5), (0.0, 90.0, 89.5), (0.2, 90.0, 89.0), (0.3, 60.0, 88.0),
    (0.95, 250.0, 88.0),
])
def test_eclipse_times_match_conjunctions(e, omega_deg, inc_deg):
    P = 3.0 * DAY
    omega, inc = np.deg2rad(omega_deg), np.deg2rad(inc_deg)
    out = eclipse_timestamps_kepler(1.0, 0.8, P, e, 0.0, inc, omega, 0.0, 1.0, 0.8)
    t_primary, t_secondary = float(out[0]) % P, float(out[5]) % P
    for got, primary in [(t_primary, True), (t_secondary, False)]:
        expected = _conjunction_time(P, e, 0.0, omega, primary)
        # Minimum sky separation differs slightly from conjunction for e > 0, i < 90.
        diff = min(abs(got - expected), P - abs(got - expected))
        assert diff < 2e-3 * P, (primary, got / DAY, expected / DAY)


def test_eclipse_default_los_is_the_shared_convention():
    args = (1.0, 0.8, 3.0 * DAY, 0.3, 0.0, np.deg2rad(88.), np.deg2rad(60.), 0.4, 1.0, 0.8)
    default = np.asarray(eclipse_timestamps_kepler(*args))
    explicit = np.asarray(eclipse_timestamps_kepler(*args, los_vector=jnp.array([0., 0., -1.])))
    np.testing.assert_array_equal(default, explicit)


def test_find_binary_eclipses_reports_an_eclipse_at_the_window_start():
    bb = IcosphereModel.construct(200, 1.0, 1.0, jnp.array([5000.]), ["teff"])
    b = Binary.from_bodies(bb, bb._replace(radius=0.8))
    P = 3.0 * DAY
    b = add_orbit(b, P=P, ecc=0.0, T=0.0, i=np.deg2rad(90.), omega=np.deg2rad(90.), Omega=0.,
                  mean_anomaly=0., reference_time=0., vgamma=0., orbit_resolution_points=50)
    eclipses = find_binary_eclipses(b)
    mids = sorted(float(ecl["mid"]) / P for ecl in eclipses)
    assert len(mids) == 2
    np.testing.assert_allclose(mids, [0.0, 0.5], atol=1e-3)


def _fwhm(x, y):
    half = y.max() / 2
    above = x[y >= half]
    return above.max() - above.min()


def test_pseudo_voigt_reduces_to_gaussian_and_lorentzian_widths():
    x = jnp.linspace(-2., 2., 400001)
    sigma = 0.1
    gauss_like = _pseudo_voigt_shape(x, sigma, 1e-9)
    assert _fwhm(np.asarray(x), np.asarray(gauss_like)) == pytest.approx(2.3548 * sigma, rel=1e-3)
    gamma = 0.05
    voigt = _pseudo_voigt_shape(x, sigma, gamma)
    f_g, f_l = 2.3548 * sigma, 2 * gamma
    olivero = 0.5346 * f_l + np.sqrt(0.2166 * f_l ** 2 + f_g ** 2)
    assert _fwhm(np.asarray(x), np.asarray(voigt)) == pytest.approx(olivero, rel=0.02)


def test_thermal_width_is_one_sigma():
    k, m_p, c = 1.380649e-16, 1.67262192e-24, 2.99792458e10
    expected = 5500. * np.sqrt(k * 5777. / (56 * m_p)) / c
    assert float(thermal_width(5500., 5777., 56.)) == pytest.approx(expected, rel=1e-4)
