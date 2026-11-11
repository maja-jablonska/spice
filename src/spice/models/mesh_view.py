import jax
import jax.numpy as jnp

from spice.models.utils import cast_to_los
from .mesh_model import MeshModel
from spice.geometry import clip
from functools import partial
import jaxkd as jk
from jaxtyping import Array, Float
from spice.utils.dtypes import float_dtype as _float_dtype

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

@partial(jax.jit, static_argnames=("n_grid",))
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
    # intersects_TC[t, c] = OR_p (ttg[t, p] & pic[c, p])
    # Materializing the (T, C, P) intermediate via broadcasting OOMs at typical
    # (T, P, C) ~ (5k, 2.5k, 2.5k) sizes (~30 GB of bool). The integer matmul
    # (T, P) @ (P, C) -> (T, C) computes the same overlap count in one pass.
    overlaps_TC = (
        triangle_to_gridpts_mask.astype(jnp.int32)
        @ points_in_circles_mask.astype(jnp.int32).T
    )
    counts_C = jnp.sum(overlaps_TC > 0, axis=0).astype(jnp.int32)  # (C,)
    return counts_C


@jax.jit
def get_mesh_view(mesh: MeshModel, los_vector: Float[Array, "3"]) -> MeshModel:
    """Set the observer direction while preserving the physical rotation axis.

    Projected geometry and line-of-sight velocities follow the new view.
    For an equator-on default axis at a custom view, call this before
    ``add_rotation`` without an explicit axis. Viewing along an existing
    rotation axis gives zero projected rotational velocity.

    Args:
        mesh (MeshModel): Properties to be cast (n, 3)
        los_vector (Float[Array, "3"]): LOS vector (3,)

    Returns:
        MeshModel: mesh with updated los_vector, mus, and los_velocities
    """
    # The observer direction must not change the physical rotation axis or
    # velocity field, even when that axis equals the construction default.
    # To choose an equator-on default, set the LOS before calling add_rotation.
    return mesh._replace(los_vector=los_vector)


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
    
    # Get coordinates about the mean valid vertex (the area is translation
    # invariant; this keeps float32 from cancelling large absolute products)
    n_valid = jnp.maximum(jnp.sum(mask), 1)
    x = clipped_safe[:, 0] - jnp.sum(clipped_safe[:, 0] * mask) / n_valid  # (n_clipped,)
    y = clipped_safe[:, 1] - jnp.sum(clipped_safe[:, 1] * mask) / n_valid  # (n_clipped,)
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
    occluded_visible_mask = m_occluded.mus > 0

    # A closed surface need not have half its *faces* on the visible side:
    # adaptive triangulations and imported meshes can refine one side more.
    # Keep every target face so no visible triangle is silently omitted.
    m_occluded_centers = m_occluded.cast_centers  # (n_faces1, 3)
    m_occluded_faces = m_occluded.faces.astype(int)  # (n_faces1, 3)
    m_occluder_centers = m_occluder.cast_centers  # (n_faces2, 3)
    m_occluder_faces = m_occluder.faces.astype(int)  # (n_faces2, 3)

    # Query n_neighbors nearest occluder faces for each occluded face.
    # Only the occluder's visible hemisphere can hide anything, and its near and
    # far sides project onto the same sky positions -- so push the back-facing
    # faces far away in *query* space rather than discarding them afterwards.
    # Filtering after the query would silently waste ~half of every neighbour
    # slot (measured: back-facing faces are ~46% of the k nearest), which starves
    # the search and under-counts occlusion by 2-3%.  Displacing them keeps the
    # shapes static, so this stays jit-safe.
    occluder_front = m_occluder.mus > 0  # (n_faces2,)
    far_away = jnp.max(jnp.abs(m_occluder_centers)) * 1e3 + 1e3
    m_occluder_query_centers = jnp.where(
        occluder_front[:, None], m_occluder_centers, far_away
    )
    neighbours, _ = jk.build_and_query(
        m_occluder_query_centers, m_occluded_centers,
        k=min(n_neighbors, m_occluder_faces.shape[0]),
    )
    valid_indices = (neighbours >= 0) & (neighbours < m_occluder_faces.shape[0])
    neighbours = jnp.clip(neighbours, 0, m_occluder_faces.shape[0] - 1)

    distances = jnp.linalg.norm(
        m_occluded_centers[:, None, :] - m_occluder_centers[neighbours], axis=-1)
    # Each circle contains its entire projected triangle. Overlapping triangles
    # therefore cannot have centres farther apart than the sum of their radii.
    target_radii = m_occluded.cast_vertex_bounding_circle_radii[:, None]
    occluder_radii = m_occluder.cast_vertex_bounding_circle_radii[neighbours]
    neighbour_mask = (valid_indices & occluder_front[neighbours]
                      & (distances <= target_radii + occluder_radii))

    occluded_vertices = m_occluded.cast_vertices[m_occluded_faces]
    occluder_vertices = m_occluder.cast_vertices[m_occluder_faces[neighbours]]
    occlusions = v_total_visible_area(occluded_vertices, occluder_vertices)
    # Keep validity separate from the integer indices. NaN-to-integer casts
    # can select a real triangle, introducing face-order-dependent overlaps.
    occlusions = jnp.where(neighbour_mask, occlusions, 0.0)

    # Sum occlusions from all neighbours for each occluded face
    total_occlusion = jnp.sum(occlusions, axis=1)  # (n_faces1,)

    # Back-facing targets cannot contribute; keep the occlusion of every
    # visible face, bounded by its own projected area.
    return jnp.where(occluded_visible_mask,
                     jnp.clip(total_occlusion, 0., m_occluded.cast_areas), 0.)


@partial(jax.jit, static_argnums=(2,))
def _resolve_occlusion_from_occluder(m_occluded: MeshModel, m_occluder: MeshModel, n_neighbors: int):
    """Same result as :func:`_resolve_occlusion`, searched from the occluder side.

    For every front-facing occluder face, query the ``n_neighbors`` nearest
    visible occluded faces, clip each pair, and scatter-add the overlap onto the
    occluded face. Polygon overlap is symmetric, so this sums exactly the same
    pairs; it is the right direction when occluder faces are much smaller than
    occluded faces (a finely meshed small body transiting a coarse star), where
    one occluded face can overlap thousands of occluder faces but one occluder
    face overlaps only a handful of occluded faces.
    """
    occluded_visible_mask = m_occluded.mus > 0
    occluder_front = m_occluder.mus > 0

    n_occluded = m_occluded.faces.shape[0]
    m_occluded_centers = m_occluded.cast_centers
    m_occluded_faces = m_occluded.faces.astype(int)
    m_occluder_centers = m_occluder.cast_centers
    m_occluder_faces = m_occluder.faces.astype(int)

    # Back-facing occluded faces cannot be hidden; push them out of the query.
    far_away = jnp.max(jnp.abs(m_occluded_centers)) * 1e3 + 1e3
    m_occluded_query_centers = jnp.where(
        occluded_visible_mask[:, None], m_occluded_centers, far_away
    )
    neighbours, _ = jk.build_and_query(
        m_occluded_query_centers, m_occluder_centers,
        k=min(n_neighbors, n_occluded),
    )  # (n_occluder, k) indices of occluded faces
    valid_indices = (neighbours >= 0) & (neighbours < n_occluded)
    neighbours = jnp.clip(neighbours, 0, n_occluded - 1)

    distances = jnp.linalg.norm(
        m_occluder_centers[:, None, :] - m_occluded_centers[neighbours], axis=-1)
    occluder_radii = m_occluder.cast_vertex_bounding_circle_radii[:, None]
    target_radii = m_occluded.cast_vertex_bounding_circle_radii[neighbours]
    pair_mask = (valid_indices & occluder_front[:, None]
                 & occluded_visible_mask[neighbours]
                 & (distances <= target_radii + occluder_radii))

    occluder_vertices = m_occluder.cast_vertices[m_occluder_faces]              # (n_occluder, 3, 3)
    occluded_vertices = m_occluded.cast_vertices[m_occluded_faces[neighbours]]  # (n_occluder, k, 3, 3)
    overlaps = v_total_visible_area(occluder_vertices, occluded_vertices)       # (n_occluder, k)
    overlaps = jnp.where(pair_mask, overlaps, 0.0)

    total_occlusion = jnp.zeros(n_occluded, dtype=overlaps.dtype).at[neighbours.ravel()].add(
        overlaps.ravel())
    return jnp.where(occluded_visible_mask,
                     jnp.clip(total_occlusion, 0., m_occluded.cast_areas), 0.)


@partial(jax.jit, static_argnames=("n_neighbors", "from_occluder"))
def resolve_occlusion(m_occluded: MeshModel, m_occluder: MeshModel, n_neighbors: int,
                      from_occluder: bool = False) -> MeshModel:
    """Calculate the occlusion of m_occluded by m_occluder

    Args:
        m_occluded (MeshModel): occluded mesh model
        m_occluder (MeshModel): occluding mesh model
        n_neighbors (int): neighbours searched per face. With
            ``from_occluder=False`` (default) this is the number of occluder faces
            considered for each occluded face; with ``from_occluder=True`` it is
            the number of occluded faces considered for each occluder face.
        from_occluder (bool): search from the occluder side. Use it when the
            occluder's faces are much smaller than the occluded body's (e.g.
            transits); :meth:`spice.models.binary.Binary.from_bodies` picks the
            cheaper direction automatically.

    Returns:
        MeshModel: m1 with updated visible areas
    """
    m_occluder_center_z = cast_to_los(m_occluder.center, m_occluder.los_vector)
    m_occluded_center_z = cast_to_los(m_occluded.center, m_occluded.los_vector)
    resolve = _resolve_occlusion_from_occluder if from_occluder else _resolve_occlusion
    o = jax.lax.cond(jnp.all(m_occluder_center_z>m_occluded_center_z),
                     lambda: resolve(m_occluded, m_occluder, n_neighbors),
                     lambda: jnp.zeros_like(m_occluded.occluded_areas, dtype=_float_dtype()))
    return m_occluded._replace(
        occluded_areas=o
    )
