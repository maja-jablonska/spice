"""Pulsations must preserve reference geometry and its area units."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spice.models import IcosphereModel
from spice.models.mesh_transform import (add_pulsation, calculate_pulsations,
                                        evaluate_pulsations, reset_pulsations)
from spice.spectrum import simulate_monochromatic_luminosity


def make_mesh(radius=1., max_mode=5):
    return IcosphereModel.construct(100, radius, 1., [5777., 0.], ['teff', 'logg'],
                                    max_pulsation_mode=max_mode)


def constant_flux(wavelengths, parameters):
    return jnp.ones((len(wavelengths), 2))


@pytest.mark.parametrize('radius', [0.5, 1., 2.])
@pytest.mark.parametrize('explicit_mode', [False, True])
def test_zero_pulsations_preserve_geometry_gravity_and_luminosity(radius, explicit_mode):
    base = make_mesh(radius)
    model = add_pulsation(base, 0, 0, 1., jnp.array([[0., 0.]])) if explicit_mode else base
    result = evaluate_pulsations(model, 0.1)
    for field in ('vertices', 'centers', 'areas', 'parameters', 'pulsation_velocities'):
        if field == 'parameters':
            np.testing.assert_allclose(result.parameters, base.parameters, rtol=0., atol=1e-14)
        else:
            np.testing.assert_array_equal(getattr(result, field), getattr(base, field))
    wavelengths = jnp.linspace(3.6, 3.7, 7)
    expected = simulate_monochromatic_luminosity(constant_flux, base, wavelengths)
    actual = simulate_monochromatic_luminosity(constant_flux, result, wavelengths)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.asarray(actual) > 0)


def triangle_areas(vertices, faces):
    a, b, c = np.asarray(vertices)[np.asarray(faces, dtype=int)].transpose(1, 0, 2)
    return np.linalg.norm(np.cross(b-a, c-a), axis=1) / 2


@pytest.mark.parametrize('radius', [0.5, 2.])
def test_combined_modes_match_displaced_triangle_geometry(radius):
    base = make_mesh(radius)
    first = add_pulsation(base, 0, 0, 1., jnp.array([[0.06, 0.]]))
    combined = add_pulsation(first, 1, 2, 1.5,
                            jnp.array([[[0.04, 0.]], [[0.02, 0.3]], [[0.01, 0.2]]]))
    result = evaluate_pulsations(combined, 0.1)
    before = triangle_areas(base.vertices, base.faces)
    after = triangle_areas(result.vertices, result.faces)
    np.testing.assert_allclose(result.areas - base.areas, (after-before)/radius**2,
                               atol=1e-14, rtol=1e-11)
    faces = np.asarray(base.faces, dtype=int)
    centroid_shift = (np.asarray(result.vertices)[faces].mean(axis=1)
                      - np.asarray(base.vertices)[faces].mean(axis=1))
    np.testing.assert_allclose(result.centers-base.centers, centroid_shift, atol=1e-14)
    assert np.all(np.asarray(result.areas) > 0)


def test_unused_mode_slots_do_not_change_the_physical_result():
    outputs = []
    for maximum in (3, 5):
        model = add_pulsation(make_mesh(2., maximum), 1, 2, 1., jnp.array([[0.05, 0.]]))
        outputs.append(evaluate_pulsations(model, 0.1))
    for field in ('vertices', 'centers', 'areas', 'parameters'):
        np.testing.assert_allclose(getattr(outputs[0], field), getattr(outputs[1], field), atol=1e-13)


def test_single_mode_helper_preserves_zero_displacement_contract():
    base = make_mesh(2.)
    offsets = calculate_pulsations(base, jnp.array([0, 0]), jnp.zeros(3), jnp.zeros(3),
                                   base.radius, base.rotation_axis, jnp.array(0.))
    assert len(offsets) == 4
    for offset in offsets:
        np.testing.assert_array_equal(offset, 0.)


def test_reset_removes_pulsation_velocities_and_is_stable_on_reevaluation():
    base = make_mesh(2.)
    model = add_pulsation(base, 0, 0, 1., jnp.array([[0.05, 0.]]))
    moving = evaluate_pulsations(model, 0.2)
    assert np.any(np.abs(np.asarray(moving.pulsation_velocities)) > 0.1)
    restored = reset_pulsations(moving)
    for result in (restored, evaluate_pulsations(restored, 0.3)):
        for field in ('vertices', 'centers', 'areas', 'parameters', 'velocities'):
            np.testing.assert_allclose(getattr(result, field), getattr(base, field), rtol=0., atol=1e-14)


def test_pulsation_area_is_jittable_and_differentiable_in_amplitude():
    base = make_mesh(2.)
    def area(amplitude):
        model = add_pulsation(base, 0, 0, 1., jnp.array([[amplitude, 0.]]))
        return evaluate_pulsations(model, 0.1).areas.sum()
    compiled = jax.jit(area)
    gradient = float(jax.grad(compiled)(0.03))
    finite_difference = float((compiled(0.03001)-compiled(0.02999))/0.00002)
    assert np.isfinite(gradient) and gradient > 0
    assert gradient == pytest.approx(finite_difference, rel=1e-7)
