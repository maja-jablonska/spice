"""Small helpers with no other tests: unit conversions, Fourier fitting, parameter utils."""
import jax.numpy as jnp
import numpy as np
import pytest

from spice.constants import DAY_TO_S, G_SI, SOLAR_MASS_KG, SOLAR_RAD_KM, SOLAR_RAD_M
from spice.models import IcosphereModel
from spice.models.utils import (horizontal_to_radial_ratio, lat_to_theta, lon_to_phi,
                                period_to_velocity, phi_to_lon, theta_to_lat,
                                velocity_to_period)
from spice.spectrum import Blackbody


def test_velocity_and_period_are_inverse_and_in_seconds():
    # The Sun at 2 km/s: P = 2 pi R / v ~ 25.3 days.
    p = float(velocity_to_period(2.0, 1.0))
    np.testing.assert_allclose(p, 2 * np.pi * SOLAR_RAD_KM / 2.0, rtol=1e-12)
    np.testing.assert_allclose(p / DAY_TO_S, 25.30, rtol=1e-3)
    np.testing.assert_allclose(float(period_to_velocity(p, 1.0)), 2.0, rtol=1e-12)
    # Scales linearly with radius.
    np.testing.assert_allclose(float(velocity_to_period(2.0, 3.0)), 3 * p, rtol=1e-12)


@pytest.mark.parametrize("lat", [-90.0, -30.0, 0.0, 45.0, 90.0])
def test_latitude_theta_round_trip(lat):
    theta = float(lat_to_theta(lat))
    assert 0.0 <= theta <= np.pi + 1e-12
    np.testing.assert_allclose(float(theta_to_lat(theta)), lat, atol=1e-12)


def test_latitude_conventions():
    np.testing.assert_allclose(float(lat_to_theta(90.0)), 0.0, atol=1e-12)        # north pole
    np.testing.assert_allclose(float(lat_to_theta(0.0)), np.pi / 2, atol=1e-12)    # equator
    np.testing.assert_allclose(float(lat_to_theta(-90.0)), np.pi, atol=1e-12)


@pytest.mark.parametrize("lon", [-180.0, -90.0, 0.0, 37.5, 180.0])
def test_longitude_phi_round_trip(lon):
    np.testing.assert_allclose(float(lon_to_phi(lon)), np.deg2rad(lon), atol=1e-12)
    np.testing.assert_allclose(float(phi_to_lon(lon_to_phi(lon))), lon, atol=1e-12)


def test_horizontal_to_radial_ratio_value():
    # K = G M / (omega^2 R^3); for the Sun and a 5-minute p mode K << 1.
    period_days = 5.0 / 1440.0
    omega = 2 * np.pi / (period_days * DAY_TO_S)
    expected = G_SI * SOLAR_MASS_KG / (omega ** 2 * SOLAR_RAD_M ** 3)
    k = float(horizontal_to_radial_ratio(1.0, 1.0, period_days))
    np.testing.assert_allclose(k, expected, rtol=1e-12)
    assert 1e-4 < k < 1e-2
    # K ~ P^2: a 100x longer period gives 1e4 x larger K.
    np.testing.assert_allclose(float(horizontal_to_radial_ratio(1.0, 1.0, 100 * period_days)),
                               1e4 * k, rtol=1e-10)


def test_fit_fourier_series_recovers_known_coefficients():
    from spice.utils.fourier_fit import fit_fourier_series
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    # A1 = 0.3 with phase 0.4, A2 = 0.1 with phase -1.2, mean 5
    y = 5.0 + 0.3 * np.cos(2 * np.pi * x - 0.4) + 0.1 * np.cos(4 * np.pi * x + 1.2)
    out = fit_fourier_series(y, x, n_terms=2)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 0], [0.3, 0.1], atol=1e-8)
    np.testing.assert_allclose(out[:, 1], [0.4, -1.2], atol=1e-7)


@pytest.fixture(scope="module")
def mesh():
    bb = Blackbody()
    return IcosphereModel.construct(100, 1.0, 1.0, bb.solar_parameters, bb.parameter_names)


def test_modify_mesh_parameter_from_array(mesh):
    from spice.utils.parameters import modify_mesh_parameter_from_array
    n = mesh.parameters.shape[0]
    vals = jnp.linspace(4000.0, 6000.0, n)
    out = modify_mesh_parameter_from_array(mesh, 0, vals)
    np.testing.assert_allclose(np.asarray(out.parameters)[:, 0], np.asarray(vals))
    np.testing.assert_allclose(np.asarray(mesh.parameters)[:, 0], 5777.0)   # input untouched


def test_modify_mesh_parameter_with_a_distribution(mesh):
    from spice.utils.parameters import modify_mesh_parameter

    def by_height(m, t_pole, t_eq):
        z = m.d_centers[:, 2] / m.radius
        return t_eq + (t_pole - t_eq) * z ** 2

    out = modify_mesh_parameter(mesh, 0, by_height, 7000.0, 5000.0)
    z = np.asarray(mesh.d_centers)[:, 2]
    teff = np.asarray(out.parameters)[:, 0]
    assert teff[np.argmax(np.abs(z))] > teff[np.argmin(np.abs(z))]
    np.testing.assert_allclose(teff, 5000.0 + 2000.0 * (z / 1.0) ** 2, rtol=1e-10)


def test_constant_luminosity_temperature(mesh):
    from spice.utils.parameters import constant_luminosity_temperature
    r_now = float(np.mean(np.asarray(mesh.radii)))
    # A reference radius 4x the current one: T must double (T ~ R^-1/2).
    t = float(constant_luminosity_temperature(mesh, 4.0 * r_now, 5000.0))
    np.testing.assert_allclose(t, 10000.0, rtol=1e-10)
