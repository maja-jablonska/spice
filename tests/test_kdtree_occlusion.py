"""Regression coverage for the legacy KD-tree occlusion entry points."""

import jax.numpy as jnp
import numpy as np
import pytest

from spice.models.mesh_transform import transform
from spice.models.mesh_view_kdtree import get_optimal_search_radius, resolve_occlusion
from tests.test_models.utils import default_icosphere


@pytest.fixture(scope="module")
def spheres():
    # Identical triangulation, scaled radii: the central transit blocks exactly
    # one quarter of the larger mesh's projected area. Positions use R_sun.
    return default_icosphere(), default_icosphere(radius=0.5, mass=0.5)


@pytest.mark.parametrize("n_grid", [12, 20])
def test_search_radius_accepts_explicit_grid_size(spheres, n_grid):
    primary, secondary = spheres
    count = get_optimal_search_radius(primary, secondary, n_grid=n_grid)
    assert isinstance(count, int) and count > 0
    assert count == get_optimal_search_radius(secondary, primary, n_grid=n_grid)


@pytest.mark.parametrize("api", ["automatic", "positional", "search_radius", "n_neighbors"])
def test_legacy_occlusion_api_resolves_central_transit(spheres, api):
    primary, secondary = spheres
    foreground = transform(secondary, jnp.array([0.0, 0.0, 3.0]))
    if api == "automatic":
        result = resolve_occlusion(primary, foreground)
    else:
        count = get_optimal_search_radius(primary, foreground, n_grid=20)
        result = (resolve_occlusion(primary, foreground, count) if api == "positional"
                  else resolve_occlusion(primary, foreground, **{api: count}))

    initial_area = float(jnp.sum(primary.visible_cast_areas))
    remaining_area = float(jnp.sum(result.visible_cast_areas))
    assert 1.0 - remaining_area / initial_area == pytest.approx(0.25, abs=1e-6)
    assert np.isfinite(result.occluded_areas).all()
    assert np.all(np.asarray(result.visible_cast_areas) >= -1e-12)
    np.testing.assert_array_equal(primary.occluded_areas, 0.0)
    np.testing.assert_array_equal(result.center, primary.center)
    np.testing.assert_array_equal(result.parameters, primary.parameters)


@pytest.mark.parametrize("position", [(0.0, 0.0, -3.0), (3.0, 0.0, 3.0)])
def test_background_or_disjoint_sphere_does_not_occlude(spheres, position):
    primary, secondary = spheres
    result = resolve_occlusion(primary, transform(secondary, jnp.array(position)),
                               n_neighbors=64)
    np.testing.assert_allclose(result.occluded_areas, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.visible_cast_areas, primary.visible_cast_areas,
                               atol=1e-12)


def test_larger_foreground_sphere_totally_eclipses_smaller_sphere(spheres):
    primary, secondary = spheres
    foreground = transform(primary, jnp.array([0.0, 0.0, 3.0]))
    result = resolve_occlusion(secondary, foreground, n_neighbors=64)
    np.testing.assert_allclose(result.visible_cast_areas, 0.0, atol=1e-10)
    visible = np.asarray(secondary.mus) > 0
    np.testing.assert_allclose(result.occluded_areas[visible], secondary.cast_areas[visible],
                               atol=1e-10)
    reverse = resolve_occlusion(foreground, secondary, n_neighbors=64)
    np.testing.assert_allclose(reverse.occluded_areas, 0.0, atol=1e-12)
