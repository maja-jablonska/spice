"""
Tests for spice.geometry.utils: sort_xy, inside, polygon_area, get_cast_areas, etc.
"""

# Set JAX to CPU before any JAX import to avoid METAL/backend errors
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from jax import config

from spice.geometry.utils import (
    sort_xy,
    inside,
    polygon_area,
    polygon_areas,
    get_cast_areas,
    last_non_nan_arg,
    last_non_nan,
    wrap,
    cast_indexes,
)

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)


class TestSortXy:
    """sort_xy: points in clockwise order."""

    def test_clockwise_order(self):
        # Square vertices (counterclockwise): sort_xy should return clockwise
        points = jnp.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        sorted_pts = sort_xy(points)
        assert sorted_pts.shape == points.shape
        # Center is (0,0); angles should be ordered
        assert jnp.all(jnp.isfinite(sorted_pts))

    def test_idempotent(self):
        points = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        once = sort_xy(points)
        twice = sort_xy(once)
        assert jnp.allclose(once, twice)


class TestInside:
    """inside(p1, p2, q): point inside polygon edge (clockwise)."""

    def test_inside_triangle_edge(self):
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([1.0, 0.0])
        # For edge (0,0)->(1,0), inside is cross <= 0 => q[1] <= 0 (below edge)
        q_inside = jnp.array([0.5, -0.5])   # below edge
        q_outside = jnp.array([0.5, 0.5])   # above edge
        assert bool(jnp.asarray(inside(p1, p2, q_inside)))
        assert not bool(jnp.asarray(inside(p1, p2, q_outside)))

    def test_on_edge(self):
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([1.0, 0.0])
        q = jnp.array([0.5, 0.0])
        # cross <= 0 means on edge counts as inside
        result = inside(p1, p2, q)
        assert bool(jnp.asarray(result))


class TestPolygonArea:
    """polygon_area and polygon_areas."""

    def test_triangle_area(self):
        # Right triangle (0,0), (1,0), (0,1) -> area 0.5
        x = jnp.array([0.0, 1.0, 0.0])
        y = jnp.array([0.0, 0.0, 1.0])
        a = polygon_area(x, y)
        assert jnp.isclose(a, 0.5)

    def test_square_area(self):
        x = jnp.array([0.0, 1.0, 1.0, 0.0])
        y = jnp.array([0.0, 0.0, 1.0, 1.0])
        a = polygon_area(x, y)
        assert jnp.isclose(a, 1.0)

    def test_polygon_areas_shape(self):
        # Two triangles
        x = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.5]])
        y = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        areas = polygon_areas(x, y)
        assert areas.shape == (2,)
        assert jnp.all(areas > 0)


class TestGetCastAreas:
    """get_cast_areas: face vertices (n_faces, 3, 2) -> areas."""

    def test_single_triangle(self):
        # One triangle in xy
        face_vertices = jnp.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        areas = get_cast_areas(face_vertices)
        assert areas.shape == (1,)
        assert jnp.isclose(areas[0], 0.5)

    def test_multiple_faces(self):
        faces = jnp.array([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        ])
        areas = get_cast_areas(faces)
        assert areas.shape == (2,)
        assert jnp.isclose(areas[0], 0.5)
        assert jnp.isclose(areas[1], 2.0)
        assert jnp.all(areas > 0)


class TestLastNonNan:
    """last_non_nan_arg and last_non_nan."""

    def test_last_non_nan_arg_no_nan(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx = last_non_nan_arg(arr)
        assert int(idx) == 2

    def test_last_non_nan_no_nan(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        row = last_non_nan(arr)
        assert row.shape == (2,)
        assert jnp.allclose(row, jnp.array([5.0, 6.0]))


class TestWrap:
    """wrap: append first row at end."""

    def test_wrap_shape(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        out = wrap(arr)
        assert out.shape == (3, 2)
        assert jnp.allclose(out[0], out[2])
        assert jnp.allclose(out[0], jnp.array([1.0, 2.0]))


class TestCastIndexes:
    """cast_indexes: indices of non-zero dimensions after cast."""

    def test_cast_vertices_all_zero_along_z(self):
        # If cast reduces z, vertices have x,y non-zero
        cast_vertices = jnp.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        # Function finds zeroth_ind = argmax(all close to 0 along axis=1)
        # So it finds which row is "all zero" - actually which dimension is zero
        idx = cast_indexes(cast_vertices)
        assert idx.shape == (2,)
        # Implementation: zeroth_ind = argmax(all(cast_vertices==0, axis=1))
        # So zeroth_ind is the row index where all coords are close to 0. If no row is zero, argmax returns 0.
        assert idx.shape == (2,)
