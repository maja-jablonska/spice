from typing import Tuple, Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_model import MeshModel
from spice.geometry import clip, polygon_area
from spice.geometry.utils import append_value_to_last_nan
from functools import partial
from jax.tree_util import register_pytree_node_class
import jaxkd as jk
from jaxtyping import Array, Float


@jax.jit
def point_in_triangle(pt, tri_verts):  # tri_verts: (3,2)
    a, b, c = tri_verts[0], tri_verts[1], tri_verts[2]
    v0 = c - a
    v1 = b - a
    v2 = pt - a
    dot00 = jnp.dot(v0, v0)
    dot01 = jnp.dot(v0, v1)
    dot02 = jnp.dot(v0, v2)
    dot11 = jnp.dot(v1, v1)
    dot12 = jnp.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    denom = jnp.where(denom == 0.0, jnp.finfo(tri_verts.dtype).eps, denom)
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return (u >= 0) & (v >= 0) & (u + v <= 1)

@jax.jit
def construct_triangle_to_gridpts(mesh, n_grid: int = 50):
    # DO NOT boolean-index with mesh.mus inside jit
    faces   = jnp.asarray(mesh.faces).astype(jnp.int32)        # (T, 3)
    visible = (jnp.asarray(mesh.mus) > 0)                      # (T,)
    verts2d = jnp.asarray(mesh.cast_vertices).astype(jnp.float32)  # (V, 2)

    # Grid over bounds
    x_min, y_min = jnp.min(verts2d, axis=0)
    x_max, y_max = jnp.max(verts2d, axis=0)
    x_grid = jnp.linspace(x_min, x_max, n_grid)
    y_grid = jnp.linspace(y_min, y_max, n_grid)
    xx, yy = jnp.meshgrid(x_grid, y_grid, indexing="xy")
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)   # (P, 2)

    # Triangle vertices for all faces
    tri_verts = verts2d[faces]  # (T, 3, 2)

    # Vectorized point-in-triangle for ALL faces, then mask by 'visible'
    vmap_pt  = jax.vmap(lambda pt, tri: point_in_triangle(pt, tri), in_axes=(0, None))
    vmap_tri = jax.vmap(lambda tri: vmap_pt(grid_points, tri),      in_axes=(0,))
    mask_TP  = vmap_tri(tri_verts)                                  # (T, P)
    mask_TP  = jnp.where(visible[:, None], mask_TP, False)          # zero-out invisible faces

    # For each grid point pick the first visible triangle (or -1)
    first_true = jnp.argmax(mask_TP, axis=0)            # (P,)
    any_true   = jnp.any(mask_TP, axis=0)               # (P,)
    grid_tri_indices = jnp.where(any_true, first_true, -1).astype(jnp.int32)

    return mask_TP, grid_tri_indices, grid_points


@jax.jit
def construct_points_in_circles(grid_points, circle_radius=0.25):
    # grid_points: (N, 2)
    centers = grid_points[:, None, :]   # (N, 1, 2)
    pts     = grid_points[None, :, :]   # (1, N, 2)
    dists2  = jnp.sum((centers - pts) ** 2, axis=-1)  # (N, N)
    mask    = dists2 <= circle_radius**2              # (N, N) bool
    return mask


@jax.jit
def find_triangle_counts(points_in_circles_mask, triangle_to_gridpts_mask):
    # triangle_to_gridpts_mask: (T, P)
    # points_in_circles_mask:   (C, P)

    # For each (t, c), check if there exists a grid point p that is in both:
    # (T, 1, P) & (1, C, P) -> (T, C, P) -> any over P -> (T, C)
    intersects_TC = jnp.any(
        triangle_to_gridpts_mask[:, None, :] & points_in_circles_mask[None, :, :],
        axis=2
    )  # bool (T, C)

    # Count triangles per circle: sum over T
    counts_C = jnp.sum(intersects_TC, axis=0).astype(jnp.int32)  # (C,)
    return counts_C


@register_pytree_node_class
class Grid:
    def __init__(self,
                 x: ArrayLike,
                 y: ArrayLike,
                 x_span: float,
                 y_span: float,
                 grid: ArrayLike,
                 n_cells: int,
                 n_centers_m1: int,
                 n_centers_m2: int,
                 n_points_m1: int,
                 n_points_m2: int):
        self.x = x
        self.y = y
        self.x_span = x_span
        self.y_span = y_span
        self.grid = grid
        self.n_cells = n_cells
        self.n_centers_m1 = n_centers_m1
        self.n_centers_m2 = n_centers_m2
        self.n_points_m1 = n_points_m1
        self.n_points_m2 = n_points_m2

    @classmethod
    def construct(cls, m1: MeshModel, m2: MeshModel, n_cells: int):
        vs1, vs2 = m1.cast_vertices[m1.faces.astype(int)], m2.cast_vertices[m2.faces.astype(int)]

        x_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, 0], vs2[:, 0]])),
                               jnp.max(jnp.concatenate([vs1[:, 0], vs2[:, 0]])), n_cells)
        y_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, 1], vs2[:, 1]])),
                               jnp.max(jnp.concatenate([vs1[:, 1], vs2[:, 1]])), n_cells)

        nx, ny = jnp.meshgrid(x_range, y_range)
        x, y = jnp.meshgrid(x_range, y_range, sparse=True)
        x, y = x.flatten(), y.flatten()
        return cls(
            x=x,
            y=y,
            x_span=x[1] - x[0],
            y_span=y[1] - y[0],
            grid=jnp.vstack([nx.ravel(), ny.ravel()]).T,
            n_cells=n_cells,
            n_centers_m1=m1.cast_centers.shape[0],
            n_centers_m2=m2.cast_centers.shape[0],
            n_points_m1=jnp.ceil(4 * m1.cast_centers.shape[0] / (n_cells * n_cells)).astype(int).item(),
            n_points_m2=jnp.ceil(4 * m2.cast_centers.shape[0] / (n_cells * n_cells)).astype(int).item()
        )

    def __hash__(self) -> int:
        return self.n_cells

    def __eq__(self, __value: object) -> bool:
        return self.__hash__() == __value.__hash__()

    def __repr__(self) -> str:
        return "Grid(n_cells={})".format(self.n_cells)

    def tree_flatten(self):
        return (self.x, self.y, self.x_span, self.y_span, self.grid, self.n_cells,
                self.n_centers_m1, self.n_centers_m2, self.n_points_m1, self.n_points_m2), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    
@jax.jit
def get_grid_spans(m1, m2, n_cells_array):
    """Calculate grid cell spans for different grid sizes.
    
    For each number of cells in n_cells_array, calculates the span (width/height) of grid cells
    that would cover the projected area of both meshes. Returns the minimum of x and y spans
    to ensure square grid cells.
    
    Args:
        m1 (MeshModel): First mesh model with cast_vertices and faces
        m2 (MeshModel): Second mesh model with cast_vertices and faces 
        n_cells_array (ArrayLike): Array of different grid cell counts to try
        
    Returns:
        ArrayLike: Array of grid cell spans corresponding to each n_cells value
    """
    vs1, vs2 = m1.cast_vertices[m1.faces.astype(int)], m2.cast_vertices[m2.faces.astype(int)]

    x_min = jnp.min(jnp.concatenate([vs1[:, 0], vs2[:, 0]]))
    x_max = jnp.max(jnp.concatenate([vs1[:, 0], vs2[:, 0]]))
    y_min = jnp.min(jnp.concatenate([vs1[:, 1], vs2[:, 1]]))
    y_max = jnp.max(jnp.concatenate([vs1[:, 1], vs2[:, 1]]))
    
    x_spans = (x_max - x_min) / n_cells_array
    y_spans = (y_max - y_min) / n_cells_array
    
    return jnp.minimum(x_spans, y_spans)


@jax.jit
def get_mesh_view(mesh: MeshModel, los_vector: Float[Array, "3"]) -> MeshModel:
    """Cast 3D vectors of centers and center velocities to the line-of-sight

    Args:
        mesh (MeshModel): Properties to be cast (n, 3)
        los_vector (Float[Array, "3"]): LOS vector (3,)

    Returns:
        MeshModel: mesh with updated los_vector, mus, and los_velocities
    """
    return mesh._replace(
        los_vector=los_vector
    )


@jax.jit
def _visible_area(vertices1: Float[Array, "n1 3"], vertices2: Float[Array, "n2 3"]) -> Float[Array, ""]:
    """
    Compute the visible area between two polygons (vertices1 and vertices2).
    The function expects vertices1 and vertices2 to be (N, 3) arrays representing
    the 3D coordinates of the polygon vertices. The output is a scalar area.
    """
    clipped = clip(vertices1, vertices2)  # (n_clipped, 3)
    
    # Create mask for valid vertices (1=valid, 0=NaN)
    mask = ~jnp.any(jnp.isnan(clipped), axis=1)  # (n_clipped,)
    
    # Replace NaN values with zeros for safe calculations
    clipped_safe = jnp.where(jnp.isnan(clipped), 0.0, clipped)  # (n_clipped, 3)
    
    # Get coordinates and handle wrap-around
    x = clipped_safe[:, 0]  # (n_clipped,)
    y = clipped_safe[:, 1]  # (n_clipped,)
    next_idx = jnp.roll(jnp.arange(x.shape[0]), -1)  # (n_clipped,)
    
    # Calculate contribution for each edge pair, masked by validity
    terms = (x * y[next_idx] - x[next_idx] * y) * mask * mask[next_idx]  # (n_clipped,)
    
    return 0.5 * jnp.abs(jnp.sum(terms))  # scalar


def visible_area(vertices1: Float[Array, "n1 3"], vertices2: Float[Array, "n2 3"]) -> Float[Array, ""]:
    """
    Compute the visible area between two polygons (vertices1 and vertices2).
    The function expects vertices1 and vertices2 to be (N, 3) arrays representing
    the 3D coordinates of the polygon vertices. The output is a scalar area.
    """
    nan1 = jnp.any(jnp.isnan(vertices1))
    nan2 = jnp.any(jnp.isnan(vertices2))
    return jax.lax.cond(
        nan1 | nan2,
        lambda _: 0.0,
        lambda _: _visible_area(vertices1, vertices2),
        operand=None
    )


# total_visible_area: (vertices1: (n1, 3), vertices2s: (n_faces2, n2, 3)) -> (n_faces2,)
# total visible area considering multiple occluders
total_visible_area = jax.jit(jax.vmap(visible_area, in_axes=(None, 0)))

# visibility_areas: (vertices1s: (n_faces1, n1, 3), vertices2s: (n_faces2, n2, 3)) -> (n_faces1, n_faces2)
v_total_visible_area = jax.jit(jax.vmap(total_visible_area, in_axes=(0, 0)))

@partial(jax.jit, static_argnums=(2,))
def _resolve_occlusion(m_occluded: MeshModel, m_occluder: MeshModel, n_neighbors: int):
    # Use all mesh elements for m_occluded and all neighbours for them from m_occluder

    # Masks for visible faces (not used for filtering, but for later masking)
    occluder_visible_mask = m_occluder.mus > 0
    occluded_visible_mask = m_occluded.mus > 0

    # Use all centers and faces (no masking)
    m_occluded_centers = m_occluded.cast_centers  # (n_faces1, 3)
    m_occluded_faces = m_occluded.faces.astype(int)  # (n_faces1, 3)
    m_occluder_centers = m_occluder.cast_centers  # (n_faces2, 3)
    m_occluder_faces = m_occluder.faces.astype(int)  # (n_faces2, 3)

    # Query n_neighbors nearest occluder faces for each occluded face
    neighbours, _ = jk.build_and_query(m_occluder_centers, m_occluded_centers, k=32)  # (n_faces1, n_neighbors)
    
    # Calculate distances between m_occluder centers and m_occluded center neighbours
    # m_occluded_centers: (n_faces1, 3)
    # m_occluder_centers: (n_faces2, 3)
    # neighbours: (n_faces1, n_neighbors) -- indices into m_occluder_centers
    # Gather the coordinates of the neighbour occluder centers for each occluded face
    neighbour_occluder_centers = m_occluder_centers[neighbours]  # (n_faces1, n_neighbors, 3)
    # Expand m_occluded_centers to (n_faces1, n_neighbors, 3) for broadcasting
    occluded_centers_expanded = jnp.expand_dims(m_occluded_centers, axis=1)  # (n_faces1, 1, 3)
    # Compute distances
    distances = jnp.linalg.norm(occluded_centers_expanded - neighbour_occluder_centers, axis=-1)  # (n_faces1, n_neighbors)
    # Take only distances smaller than radii
    # m_occluded.radii: (n_faces1,)
    # distances: (n_faces1, n_neighbors)
    # We want a mask: (n_faces1, n_neighbors)
    radii_expanded = jnp.expand_dims(m_occluded.radii, axis=1)  # (n_faces1, 1)
    neighbour_mask = distances < radii_expanded  # (n_faces1, n_neighbors)

    # Gather the vertices for each occluded face (all faces)
    occluded_vertices = m_occluded.cast_vertices[m_occluded_faces.astype(int)]  # (n_faces1, 3, 3)

    # Gather the vertices for each neighbour occluder face for each occluded face
    # neighbours: (n_faces1, n_neighbors)
    # m_occluder_faces[neighbours]: (n_faces1, n_neighbors, 3)
    neighbours = jnp.where(neighbour_mask, neighbours, jnp.ones_like(neighbours) * jnp.nan)
    occluder_vertices = m_occluder.cast_vertices[m_occluder_faces[neighbours.astype(int)]]  # (n_faces1, n_neighbors, 3, 3)

    # Instead of flattening and calling visible_area directly (which is not batched and causes shape errors),
    # use vmap to vectorize visible_area over the neighbor axis for each occluded face.
    # v_total_visible_area: (n_faces1, n_neighbors, 3, 3), (n_faces1, n_neighbors, 3, 3) -> (n_faces1, n_neighbors)
    occlusions = v_total_visible_area(occluded_vertices, occluder_vertices)  # (n_faces1, n_neighbors)

    # Sum occlusions from all neighbours for each occluded face
    total_occlusion = jnp.sum(occlusions, axis=1)  # (n_faces1,)

    # Only keep occlusion for visible faces, and clip to the face area
    clipped_occlusions = jnp.clip(total_occlusion, 0., jnp.where(occluded_visible_mask, m_occluded.cast_areas, 0.0))
    total_occlusion = jnp.where(occluded_visible_mask, clipped_occlusions, 0.0)
    return total_occlusion


@partial(jax.jit, static_argnums=(2,))
def resolve_occlusion(m_occluded: MeshModel, m_occluder: MeshModel, n_neighbors: int) -> MeshModel:
    """Calculate the occlusion of m_occluded by m_occluder

    Args:
        m_occluded (MeshModel): occluded mesh model
        m_occluder (MeshModel): occluding mesh model

    Returns:
        MeshModel: m1 with updated visible areas
    """
    o = _resolve_occlusion(m_occluded, m_occluder, n_neighbors)
    return m_occluded._replace(
        occluded_areas=o
    )
