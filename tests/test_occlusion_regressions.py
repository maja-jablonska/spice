"""Occlusion must depend on geometry, not triangle scale or storage order."""
import jax.numpy as jnp
import numpy as np
import pytest
from spice.models import IcosphereModel
from spice.models.mesh_view import resolve_occlusion


def body(radius, center):
    return IcosphereModel.construct(100, radius, 1., [5777.], ['teff'])._replace(
        center=jnp.asarray(center, dtype=float))


@pytest.mark.parametrize('radius', [10., 20.])
def test_large_triangles_fully_hide_a_small_star(radius):
    target = body(1., [0., 0., 0.])
    foreground = body(radius, [0., 0., 30.])
    hidden = resolve_occlusion(target, foreground, n_neighbors=80)
    assert float(hidden.visible_cast_areas.sum()) == pytest.approx(0., abs=1e-12)


def test_occlusion_is_invariant_to_foreground_face_order():
    target = body(1., [0., 0., 0.])
    foreground = body(1., [0.8, 0., 5.])
    permutation = jnp.arange(foreground.faces.shape[0] - 1, -1, -1)
    fields = ('faces', 'd_centers', 'base_areas', 'parameters', 'rotation_velocities',
              'center_pulsation_offsets', 'area_pulsation_offsets', 'pulsation_velocities',
              'axis_radii', 'occluded_areas')
    reordered = foreground._replace(**{name: getattr(foreground, name)[permutation]
                                       for name in fields})
    expected = resolve_occlusion(target, foreground, n_neighbors=100)
    actual = resolve_occlusion(target, reordered, n_neighbors=100)
    np.testing.assert_allclose(actual.occluded_areas, expected.occluded_areas, atol=1e-12)
    blocked = float(actual.occluded_areas.sum() / target.visible_cast_areas.sum())
    # Independent overlap of two unit circles at separation 0.8; allow mesh error.
    analytic = (2 * np.arccos(0.4) - 0.4 * np.sqrt(4 - 0.8**2)) / np.pi
    assert blocked == pytest.approx(analytic, abs=0.005)


def test_rejected_neighbours_cannot_hide_a_disjoint_star():
    target = body(1., [0., 0., 0.])
    foreground = body(1., [5., 0., 5.])
    actual = resolve_occlusion(target, foreground, n_neighbors=100)
    np.testing.assert_array_equal(actual.occluded_areas, 0.)
