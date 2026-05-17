"""Regression tests for :func:`spice.models.orbit_utils.get_orbit_jax`."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from spice.models.orbit_utils import get_orbit_jax


def test_vgamma_is_kilometers_per_second():
    """``vgamma`` is in km/s; a +1 km/s offset lowers primary :math:`v_z` by 1 km/s."""
    m1 = m2 = 1.0
    P = 1.0
    time = jnp.array([0.0])
    ecc = T = 0.0
    i = omega = Omega = mean_anomaly = reference_time = 0.0

    orb0 = get_orbit_jax(
        time, m1, m2, P, ecc, T, i, omega, Omega,
        mean_anomaly, reference_time, vgamma=0.0,
    )
    orb1 = get_orbit_jax(
        time, m1, m2, P, ecc, T, i, omega, Omega,
        mean_anomaly, reference_time, vgamma=1.0,
    )
    dvz = float(orb1[3, 0, 2] - orb0[3, 0, 2])
    assert dvz == pytest.approx(-1.0, rel=0.0, abs=1e-9)


def test_mean_anomaly_reference_time_and_periastron():
    """M(t) = mean_anomaly + n*(t - ref) + n*(ref - T) == mean_anomaly + n*(t - T)."""
    m1 = m2 = 1.0
    P = 1.0
    time = jnp.array([0.0])
    ecc = 0.0
    i, omega, Omega = jnp.pi / 2, 0.0, 0.0
    reference_time = 0.1
    T = 0.2
    mean_anomaly = 0.3

    n = 2 * jnp.pi / P
    # M(t) = mean_anomaly + n*(t - ref) + n*(ref - T) = mean_anomaly + n*(t - T).
    # With ref=T=0 in the shortcut: M = ma' + n*t  =>  ma' = mean_anomaly - n*T.
    equivalent_ma0 = mean_anomaly - n * T

    full = get_orbit_jax(
        time, m1, m2, P, ecc, T, i, omega, Omega,
        mean_anomaly, reference_time, vgamma=0.0,
    )
    shortcut = get_orbit_jax(
        time, m1, m2, P, ecc, 0.0, i, omega, Omega,
        equivalent_ma0, 0.0, vgamma=0.0,
    )
    assert jnp.allclose(full[2], shortcut[2], rtol=0.0, atol=1e-9)
