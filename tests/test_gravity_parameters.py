"""Automatically computed log g must reach the atmosphere after transforms."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spice.models import IcosphereModel
from spice.models.mesh_transform import (add_rotation, evaluate_rotation, add_pulsation,
                                         evaluate_pulsations, reset_pulsations)
from spice.spectrum import simulate_observed_flux


def mesh():
    return IcosphereModel.construct(100, 1., 1., [5777., 0.], ['teff', 'logg'])


def test_rotation_updates_gravity_used_by_synthesis():
    spinning = add_rotation(mesh(), 200.)
    wavelength = jnp.linspace(3.6, 3.7, 11)
    def intensity(x, mu, parameters):
        return jnp.full((len(x), 2), parameters[1])
    actual = simulate_observed_flux(intensity, spinning, wavelength)
    radii_m = jnp.linalg.norm(spinning.centers - spinning.center, axis=1) * 695700000.
    g_mks = 274.20011165737316 * (695700000. / radii_m)**2
    g_mks -= jnp.sum((spinning.rotation_velocities * 1000.)**2, axis=1) / radii_m
    logg = jnp.log10(g_mks * 100.)
    expected = jnp.sum(spinning.visible_cast_areas * logg) * 5.08326693599739e-16 / 100.
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(spinning.parameters[:, 1], logg, rtol=1e-12)


@pytest.mark.parametrize("speed", [0., 100.])
def test_gravity_update_is_jittable_and_differentiable(speed):
    base = mesh()
    def mean_gravity(speed):
        return add_rotation(base, speed).parameters[:, 1].mean()
    value = jax.jit(mean_gravity)(speed)
    assert float(value) <= float(base.parameters[:, 1].mean())
    grad = float(jax.grad(mean_gravity)(speed))
    finite_difference = float((mean_gravity(speed + 0.01) - mean_gravity(speed - 0.01)) / 0.02)
    assert grad <= 0
    assert grad == pytest.approx(finite_difference, rel=1e-6)


def test_explicit_logg_override_survives_rotation_and_pulsations():
    with pytest.warns(UserWarning, match='override_log_g'):
        original = IcosphereModel.construct(100, 1., 1., [5777., 3.5], ['teff', 'logg'],
                                            override_log_g=True)
    rotating = evaluate_rotation(add_rotation(original, 200.), 100.)
    pulsating = add_pulsation(rotating, 0, 0, 1., jnp.array([[0.05, 0.]]))
    evolved = evaluate_pulsations(pulsating, 0.)
    for state in (rotating, evolved, reset_pulsations(evolved)):
        np.testing.assert_array_equal(state.parameters[:, 1], 3.5)
        assert bool(state.override_log_g)


def test_pulsation_and_reset_refresh_computed_gravity():
    original = mesh()
    pulsating = add_pulsation(original, 0, 0, 1., jnp.array([[0.05, 0.]]))
    evolved = evaluate_pulsations(pulsating, 0.)
    assert not np.allclose(evolved.log_gs, original.log_gs)
    np.testing.assert_allclose(evolved.parameters[:, 1], evolved.log_gs)
    restored = reset_pulsations(evolved)
    np.testing.assert_allclose(restored.parameters[:, 1], original.parameters[:, 1])


def test_legacy_positional_mesh_construction_defaults_to_computed_gravity():
    original = mesh()
    rebuilt = type(original)(*original[:-1])
    assert rebuilt.override_log_g is False
    np.testing.assert_allclose(add_rotation(rebuilt, 100.).parameters[:, 1],
                               add_rotation(original, 100.).parameters[:, 1])
