"""Backward-compatible KD-tree occlusion helpers.

This module keeps the older `mesh_view_kdtree` import path working while the
implementation lives in `spice.models.mesh_view`.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from .mesh_model import MeshModel
from .mesh_view import (
    construct_points_in_circles,
    construct_triangle_to_gridpts,
    find_triangle_counts,
    resolve_occlusion as _resolve_occlusion,
)

DEFAULT_N_NEIGHBOURS = 32
MIN_N_NEIGHBOURS = 8
MAX_N_NEIGHBOURS = 64
N_NEIGHBOUR_SCALE = 1.5


def _estimate_neighbours(m_occluded: MeshModel, m_occluder: MeshModel, n_grid: int = 50) -> int:
    triangle_to_gridpts, _, grid_points = construct_triangle_to_gridpts(m_occluded, n_grid=n_grid)
    circle_radius = jnp.max(m_occluder.cast_vertex_bounding_circle_radii)
    points_in_circles = construct_points_in_circles(grid_points, circle_radius)
    triangle_counts = find_triangle_counts(points_in_circles, triangle_to_gridpts)

    max_count = float(np.max(np.asarray(triangle_counts)))
    if max_count <= 0.0:
        return DEFAULT_N_NEIGHBOURS
    return int(np.clip(N_NEIGHBOUR_SCALE * max_count, MIN_N_NEIGHBOURS, MAX_N_NEIGHBOURS))


def get_optimal_search_radius(m1: MeshModel, m2: MeshModel, n_grid: int = 50) -> int:
    """Return a neighbor count compatible with the legacy search radius API."""
    n12 = _estimate_neighbours(m1, m2, n_grid=n_grid)
    n21 = _estimate_neighbours(m2, m1, n_grid=n_grid)
    return int(max(n12, n21))


def resolve_occlusion(
    m_occluded: MeshModel,
    m_occluder: MeshModel,
    search_radius: Optional[int] = None,
    *,
    n_neighbors: Optional[int] = None,
) -> MeshModel:
    """Resolve occlusion using either legacy `search_radius` or `n_neighbors`."""
    resolved_n_neighbors = n_neighbors if n_neighbors is not None else search_radius
    if resolved_n_neighbors is None:
        resolved_n_neighbors = get_optimal_search_radius(m_occluded, m_occluder)
    return _resolve_occlusion(m_occluded, m_occluder, n_neighbors=int(resolved_n_neighbors))

