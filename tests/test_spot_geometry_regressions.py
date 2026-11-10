"""Angular caps and intrinsic surface maps must respect their coordinates."""
import jax.numpy as jnp
import numpy as np
import pytest
from spice.models import IcosphereModel
from spice.models.mesh_transform import transform
from spice.models.spots import (add_spot, add_spots, add_spherical_harmonic_spot,
                                add_spherical_harmonic_spots, generate_spherical_spot)


def mesh():
    return IcosphereModel.construct(100, 1., 1., [5777.], ['teff'])


def test_batch_default_smoothness_matches_individual_spots():
    base = mesh()
    thetas, phis = jnp.array([0.4, 1.2]), jnp.array([0.3, 0.8])
    radii, deltas, indices = jnp.array([30., 70.]), jnp.array([-100., -200.]), jnp.array([0, 0])
    expected = base
    for theta, phi, radius, delta, index in zip(thetas, phis, radii, deltas, indices):
        expected = add_spot(expected, theta, phi, radius, delta, index)
    actual = add_spots(base, thetas, phis, radii, deltas, indices)
    np.testing.assert_allclose(actual.parameters, expected.parameters, atol=1e-12)
    assert np.any(np.asarray(actual.parameters) < 5700.)


@pytest.mark.parametrize('angle', [30., 90., 120., 150.])
@pytest.mark.parametrize('radius', [0.5, 2.])
def test_angular_spot_boundary(angle, radius):
    theta = np.deg2rad(np.array([angle-20., angle, angle+20.]))
    points = radius * np.column_stack([np.sin(theta), np.zeros(3), np.cos(theta)])
    fraction = generate_spherical_spot(jnp.asarray(points), radius, 0., 0., angle, 1., 0.)
    assert float(fraction[0]) > 0.99
    assert float(fraction[1]) == pytest.approx(0.5, abs=1e-12)
    assert float(fraction[2]) < 0.01


def test_large_spot_does_not_cover_the_antipode():
    result = generate_spherical_spot(jnp.array([[0., 0., -1.]]), 1., 0., 0., 120., -100., 0.)
    assert abs(float(result[0])) < 1e-8


@pytest.mark.parametrize('angle,expected', [(0., 0.), (180., -100.), (400., -100.)])
def test_empty_and_full_spot_caps(angle, expected):
    points = jnp.array([[0., 0., 1.], [1., 0., 0.], [0., 0., -1.]])
    result = generate_spherical_spot(points, 1., 0., 0., angle, -100., 0.)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('tilted', [False, True])
@pytest.mark.parametrize('batch', [False, True])
def test_harmonic_spots_commute_with_translation(tilted, batch):
    base = mesh()
    translation = jnp.array([10., 2., -3.])
    def spot(model):
        if batch:
            kwargs = dict(tilt_axes=jnp.array([[0., 1., 0.]]), tilt_angles=jnp.array([30.])) if tilted else {}
            return add_spherical_harmonic_spots(model, jnp.array([0]), jnp.array([1]),
                                                jnp.array([-100.]), jnp.array([0]), **kwargs)
        kwargs = dict(tilt_axis=jnp.array([0., 1., 0.]), tilt_angle=30.) if tilted else {}
        return add_spherical_harmonic_spot(model, 0, 1, -100., 0, **kwargs)
    expected = transform(spot(base), translation)
    actual = spot(transform(base, translation))
    np.testing.assert_array_equal(actual.parameters, expected.parameters)
    np.testing.assert_array_equal(actual.vertices, expected.vertices)
