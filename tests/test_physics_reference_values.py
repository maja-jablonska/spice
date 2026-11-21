"""Physics checked against independent reference values.

Each test compares SPICE with a closed-form result (or astropy) rather than with
SPICE itself: limb-darkening laws and their flux normalisation, the Planck
function and the disc-integrated flux of a sphere, eccentric Keplerian radial
velocities, and the longitude/rotation convention of spots.
"""
import importlib

import numpy as np
import jax.numpy as jnp
import pytest

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_rotation, evaluate_rotation
from spice.models.spots import add_spot
from spice.models.utils import lat_to_theta, lon_to_phi
from spice.spectrum import Blackbody, simulate_observed_flux
# ``spice.spectrum.limb_darkening`` is also the name of a lazily exported
# function, so import the modules explicitly.
ld = importlib.import_module("spice.spectrum.limb_darkening")
fld = importlib.import_module("spice.spectrum.flux_limb_darkening")

MU = np.linspace(0.0, 1.0, 2001)[1:]          # avoid the mu = 0 clip in limb_darkening
SOLAR_RADIUS_CM = 6.957e10
TEN_PC_CM = 10 * 3.0856775814913673e18


# ---------------------------------------------------------------- limb darkening

INTENSITY_LAWS = {
    "linear": ([0.6], lambda mu, a: 1 - a[0] * (1 - mu)),
    "quadratic": ([0.4, 0.25], lambda mu, a: 1 - a[0] * (1 - mu) - a[1] * (1 - mu) ** 2),
    "square_root": ([0.3, 0.45], lambda mu, a: 1 - a[0] * (1 - mu) - a[1] * (1 - np.sqrt(mu))),
    "logarithmic": ([0.55, 0.2], lambda mu, a: 1 - a[0] * (1 - mu) - a[1] * mu * np.log(mu)),
    "nonlinear": ([0.5, -0.2, 0.6, -0.25],
                  lambda mu, a: 1 - sum(a[k] * (1 - mu ** ((k + 1) / 2)) for k in range(4))),
}


@pytest.mark.parametrize("law", sorted(INTENSITY_LAWS))
def test_intensity_limb_darkening_laws_match_their_formulas(law):
    coeffs, formula = INTENSITY_LAWS[law]
    got = np.asarray(ld.limb_darkening(jnp.asarray(MU), ld.get_limb_darkening_law_id(law),
                                       jnp.asarray(coeffs, dtype=float)))
    np.testing.assert_allclose(got, formula(MU, coeffs), rtol=0, atol=1e-12)
    assert got[-1] == pytest.approx(1.0)                      # f(1) = 1 for every law


FLUX_LAWS = {
    "linear": ([0.6, 0, 0, 0], lambda mu, c: 1 - c[0] * (1 - mu)),
    "quadratic": ([0.4, 0.25, 0, 0], lambda mu, c: 1 - c[0] * (1 - mu) - c[1] * (1 - mu) ** 2),
    "nonlinear_4": ([0.5, -0.2, 0.6, -0.25],
                    lambda mu, c: 1 - (c[0] * (1 - np.sqrt(mu)) + c[1] * (1 - mu)
                                       + c[2] * (1 - mu ** 1.5) + c[3] * (1 - mu ** 2))),
}


def _disc_integral(f_mu):
    """2 pi * integral_0^1 f(mu) mu dmu on the MU grid (plus the mu = 0 end)."""
    mu = np.concatenate([[0.0], MU])
    f = np.concatenate([[f_mu[0]], f_mu])        # f is finite at mu = 0 for all laws here
    return 2 * np.pi * np.trapezoid(f * mu, mu)


@pytest.mark.parametrize("law", sorted(FLUX_LAWS))
def test_flux_limb_darkening_laws_and_normalisation(law):
    coeffs, formula = FLUX_LAWS[law]
    law_id = fld.get_limb_darkening_law_id(law)
    c = jnp.asarray(coeffs, dtype=float)
    f = np.asarray(fld.limb_darkening(jnp.asarray(MU), law_id, c))
    np.testing.assert_allclose(f, formula(MU, coeffs), rtol=0, atol=1e-12)
    # The analytic norm is 2 pi * int f mu dmu ...
    assert float(fld.limb_darkening_norm(law_id, c)) == pytest.approx(
        _disc_integral(formula(MU, coeffs)), rel=1e-6)
    # ... so converting a flux to I(mu) and integrating back returns the flux.
    flux = 3.7
    intensity = np.asarray(fld.apply_flux_limb_darkening(flux, jnp.asarray(MU), law, coeffs))
    assert _disc_integral(intensity) == pytest.approx(flux, rel=1e-6)


# ---------------------------------------------------------------- blackbody

def _planck_erg_s_cm2_A_sr(wavelength_A, teff):
    from astropy import units as u, constants as const
    from astropy.modeling.physical_models import BlackBody
    b_nu = BlackBody(temperature=teff * u.K)(wavelength_A * u.AA)
    return (b_nu * const.c / (wavelength_A * u.AA) ** 2).to(
        u.erg / u.s / u.cm ** 2 / u.AA / u.sr).value


WAVELENGTHS_A = np.array([3000.0, 5500.0, 10000.0, 20000.0])


@pytest.mark.parametrize("teff", [3500.0, 5777.0, 12000.0])
def test_blackbody_intensity_is_the_planck_function(teff):
    got = np.asarray(Blackbody().intensity(jnp.log10(WAVELENGTHS_A), 1.0, jnp.array([teff])))
    np.testing.assert_allclose(got[:, 0], _planck_erg_s_cm2_A_sr(WAVELENGTHS_A, teff), rtol=1e-10)
    np.testing.assert_array_equal(got[:, 0], got[:, 1])


def test_blackbody_flux_is_the_emergent_surface_flux():
    # F = 2 pi int_0^1 I mu dmu: pi B without limb darkening, pi B (1 - a/3) for linear.
    planck = _planck_erg_s_cm2_A_sr(WAVELENGTHS_A, 5777.0)
    log_wl = jnp.log10(WAVELENGTHS_A)
    got = np.asarray(Blackbody().flux(log_wl, jnp.array([5777.0])))
    np.testing.assert_allclose(got[:, 0], np.pi * planck, rtol=1e-10)
    np.testing.assert_array_equal(got[:, 0], got[:, 1])
    darkened = np.asarray(Blackbody(ld_law="linear", ld_coeffs=[0.6]).flux(log_wl, jnp.array([5777.0])))
    np.testing.assert_allclose(darkened[:, 0], np.pi * planck * (1 - 0.6 / 3), rtol=1e-6)


@pytest.fixture(scope="module")
def solar_sphere():
    bb = Blackbody()
    return IcosphereModel.construct(1000, 1.0, 1.0, bb.to_parameters(), bb.parameter_names)


def test_disc_integrated_flux_is_pi_B_R2_over_d2(solar_sphere):
    # 1280 faces: the flat facets are good to ~7e-4 (second order in the facet size).
    planck = _planck_erg_s_cm2_A_sr(WAVELENGTHS_A, 5777.0)
    expected = np.pi * planck * SOLAR_RADIUS_CM ** 2 / TEN_PC_CM ** 2
    flux = np.asarray(simulate_observed_flux(Blackbody().intensity, solar_sphere,
                                             jnp.log10(WAVELENGTHS_A)))[:, 0]
    np.testing.assert_allclose(flux, expected, rtol=2e-3)
    # With linear limb darkening f = 1 - a(1 - mu) the flux drops by (1 - a/3).
    darkened = Blackbody(ld_law="linear", ld_coeffs=[0.6])
    flux_ld = np.asarray(simulate_observed_flux(darkened.intensity, solar_sphere,
                                                jnp.log10(WAVELENGTHS_A)))[:, 0]
    np.testing.assert_allclose(flux_ld, expected * (1 - 0.6 / 3), rtol=2e-3)


# ---------------------------------------------------------------- eccentric orbits

@pytest.mark.parametrize("ecc,omega", [(0.3, 0.7), (0.5, 2.0), (0.7, 4.0)])
def test_eccentric_orbit_radial_velocity_and_velocity_field(ecc, omega):
    """Body 1's radial velocity is K1 [cos(nu + omega) + e cos(omega)] (positive =
    receding along the default line of sight [0, 0, -1]), and its velocity is the
    time derivative of its position."""
    from spice.constants import G_SI
    from spice.models.orbit_utils import get_orbit_jax

    m1, m2, period_yr, inc, node = 1.5, 0.8, 10 / 365.25, np.deg2rad(80.0), 0.3
    year_s, msun_kg = 365.25 * 86400.0, 1.98840987e30
    t = np.linspace(0.0, period_yr, 2001)
    orbit = get_orbit_jax(jnp.asarray(t), m1, m2, period_yr, ecc, 0.0, inc, omega, node, 0.0, 0.0)
    pos1_m, vel1_kms = np.asarray(orbit[2]), np.asarray(orbit[3])

    mean_anomaly = 2 * np.pi * t / period_yr
    ecc_anomaly = mean_anomaly.copy()
    for _ in range(50):
        ecc_anomaly -= (ecc_anomaly - ecc * np.sin(ecc_anomaly) - mean_anomaly) / (1 - ecc * np.cos(ecc_anomaly))
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(ecc_anomaly / 2),
                                  np.sqrt(1 - ecc) * np.cos(ecc_anomaly / 2))
    period_s = period_yr * year_s
    a_m = (G_SI * (m1 + m2) * msun_kg * period_s ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    a1_m = a_m * m2 / (m1 + m2)
    k1_kms = 2 * np.pi * a1_m * np.sin(inc) / (period_s * np.sqrt(1 - ecc ** 2)) / 1e3

    rv = vel1_kms @ np.array([0.0, 0.0, -1.0])
    expected_rv = k1_kms * (np.cos(true_anomaly + omega) + ecc * np.cos(omega))
    np.testing.assert_allclose(rv, expected_rv, rtol=0, atol=1e-6 * k1_kms)
    # Orbit size: r1 = a1 (1 - e cos E).
    np.testing.assert_allclose(np.linalg.norm(pos1_m, axis=1),
                               a1_m * (1 - ecc * np.cos(ecc_anomaly)), rtol=1e-8)
    # Velocity is d(position)/dt, radial term included (positions in m, velocities in km/s).
    dpos_dt_kms = np.gradient(pos1_m, t * year_s, axis=0)[2:-2] / 1e3
    np.testing.assert_allclose(vel1_kms[2:-2], dpos_dt_kms, rtol=0, atol=2e-3 * k1_kms)


# ---------------------------------------------------------------- spot longitude

def _spot_centroid(mesh):
    weight = np.asarray(mesh.parameters[:, 0]) - 5700.0
    direction = (weight[:, None] * np.asarray(mesh.centers)).sum(0) / weight.sum()
    direction /= np.linalg.norm(direction)
    v_los = float((weight * np.asarray(mesh.los_velocities)).sum() / weight.sum())
    return direction, v_los


def test_spot_longitude_and_rotation_direction():
    """Longitude -90 deg faces the default observer (+z); the star spins toward
    decreasing longitude, so a spot at longitude 0 approaches (blueshift) and
    reaches disc centre a quarter of a rotation later."""
    v_eq = 20.0                                          # km/s, default axis +y (equator-on)
    star = add_rotation(IcosphereModel.construct(1000, 1.0, 1.0, [5700.0, 0.0], ["teff", "abun"]), v_eq)
    period_s = 2 * np.pi * 695700.0 / v_eq

    def spotted(lon_deg):
        return add_spot(star, spot_center_theta=lat_to_theta(0.0), spot_center_phi=lon_to_phi(lon_deg),
                        spot_radius=15.0, parameter_delta=1000.0, parameter_index=0)

    facing, v_facing = _spot_centroid(spotted(-90.0))
    np.testing.assert_allclose(facing, [0.0, 0.0, 1.0], atol=1e-3)
    assert abs(v_facing) < 0.05 * v_eq

    limb = spotted(0.0)
    direction, v_limb = _spot_centroid(limb)
    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-3)
    assert v_limb < -0.9 * v_eq                          # approaching limb: blueshifted

    later, v_later = _spot_centroid(evaluate_rotation(limb, 0.25 * period_s))
    np.testing.assert_allclose(later, [0.0, 0.0, 1.0], atol=1e-3)
    assert abs(v_later) < 0.05 * v_eq
    _, v_receding = _spot_centroid(evaluate_rotation(limb, 0.5 * period_s))
    assert v_receding > 0.9 * v_eq                       # then receding: redshifted
