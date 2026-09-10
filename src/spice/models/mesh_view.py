import jax
import jax.numpy as jnp

from spice.models.utils import cast_to_los
from .mesh_model import MeshModel, _default_rotation_axis
from .mesh_transform import _add_rotation, _axis_perpendicular_to_los, _is_phoebe_model
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
    """Cast 3D vectors of centers and center velocities to the line-of-sight

    Args:
        mesh (MeshModel): Properties to be cast (n, 3)
        los_vector (Float[Array, "3"]): LOS vector (3,)

    Returns:
        MeshModel: mesh with updated los_vector, mus, and los_velocities
    """
    mesh = mesh._replace(los_vector=los_vector)

    # PHOEBE models are read-only in SPICE and set their own rotation axis from
    # the orbital inclination -- which at i = 90 deg is exactly the module
    # default, so the check below would misread it as unset. Never touch them.
    if _is_phoebe_model(mesh):
        return mesh

    # Re-derive a rotation axis that had been left at its default. The axis is
    # chosen when add_rotation runs, so a caller that sets the LOS *afterwards*
    # (as tz_fornacis_spectra.py does: add_rotation -> get_mesh_view) would keep
    # an axis derived from the construction-time LOS. If that ends up parallel
    # to the new LOS the star is silently viewed pole-on and all rotational
    # broadening disappears. Only meshes still carrying a default axis are
    # touched, so an explicitly chosen axis (including a deliberate pole-on
    # geometry) is preserved.
    is_default_axis = jnp.allclose(mesh.rotation_axis, _default_rotation_axis())
    parallel_to_los = jnp.abs(jnp.dot(
        mesh.rotation_axis / jnp.linalg.norm(mesh.rotation_axis),
        los_vector / jnp.linalg.norm(los_vector),
    )) > 0.99
    return jax.lax.cond(
        is_default_axis & parallel_to_los,
        lambda m: _replace_rotation_axis(m, _axis_perpendicular_to_los(los_vector)),
        lambda m: m,
        mesh,
    )


def _replace_rotation_axis(mesh: MeshModel, axis: Float[Array, "3"]) -> MeshModel:
    """Swap the rotation axis, rebuilding the derived matrices and velocities."""
    return _add_rotation(mesh, mesh.rotation_velocity, axis)


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
    occluded_visible_mask = m_occluded.mus > 0

    # Only front-facing elements can be occluded, and the polygon clipping below
    # is by far the most expensive part of a binary evaluation (n_faces x k
    # Sutherland-Hodgman clips). Gather the front-facing half into a fixed-size
    # buffer so the clipping runs on ~0.6 n instead of n. The budget is derived
    # from the static face count, so shapes stay jit-friendly; exactly half of a
    # closed surface faces the observer, so 0.6 n cannot truncate a real
    # visibility set. Results are scattered back to the full-length array.
    n_faces1 = m_occluded.cast_centers.shape[0]
    budget = int(0.6 * n_faces1) + 1
    front_idx = jnp.argsort(-m_occluded.mus)[:budget]  # front-facing first

    m_occluded_centers = m_occluded.cast_centers[front_idx]  # (budget, 3)
    m_occluded_faces = m_occluded.faces.astype(int)[front_idx]  # (budget, 3)
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
    neighbours, _ = jk.build_and_query(m_occluder_query_centers, m_occluded_centers, k=n_neighbors)  # (n_faces1, n_neighbors)
    
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
    radii_expanded = jnp.expand_dims(m_occluded.radii[front_idx], axis=1)  # (budget, 1)
    neighbour_mask = distances < radii_expanded  # (budget, n_neighbors)

    # Gather the vertices for each occluded face (all faces)
    occluded_vertices = m_occluded.cast_vertices[m_occluded_faces.astype(int)]  # (budget, 3, 3)

    # Gather the vertices for each neighbour occluder face for each occluded face
    # neighbours: (n_faces1, n_neighbors)
    # m_occluder_faces[neighbours]: (n_faces1, n_neighbors, 3)
    neighbours = jnp.where(neighbour_mask, neighbours, jnp.ones_like(neighbours) * jnp.nan)
    occluder_vertices = m_occluder.cast_vertices[m_occluder_faces[neighbours.astype(int)]]  # (budget, n_neighbors, 3, 3)

    # Instead of flattening and calling visible_area directly (which is not batched and causes shape errors),
    # use vmap to vectorize visible_area over the neighbor axis for each occluded face.
    # v_total_visible_area: (n_faces1, n_neighbors, 3, 3), (n_faces1, n_neighbors, 3, 3) -> (n_faces1, n_neighbors)
    occlusions = v_total_visible_area(occluded_vertices, occluder_vertices)  # (n_faces1, n_neighbors)

    # Safety net: the query above already displaces back-facing occluder faces,
    # but if a mesh has too few front-facing ones some could still be returned.
    # Only the occluder's visible hemisphere can hide anything -- its near and far
    # sides project onto the same sky positions, so counting both double-counts
    # every overlap (measured: back-facing neighbours contributed exactly 50% of
    # the summed area). The `jnp.clip` below then saturated the doubled sum at the
    # face area instead of erroring, which hid the problem while over-deepening
    # eclipses by +5.3% / +2.6% / +1.2% of the blocked flux fraction at 1280 /
    # 5120 / 20480 faces (TZ For geometry), versus 0.2-0.4% once masked.
    occluder_mus = m_occluder.mus[neighbours.astype(int)]  # (n_faces1, n_neighbors)
    occlusions = jnp.where(occluder_mus > 0, occlusions, 0.0)

    # Sum occlusions from all neighbours for each occluded face
    total_occlusion = jnp.sum(occlusions, axis=1)  # (n_faces1,)

    # Only keep occlusion for visible faces, and clip to the face area
    # Clip against each gathered face's own area, then scatter back to the full
    # per-face array (faces outside the front-facing budget contribute nothing).
    cast_sel = m_occluded.cast_areas[front_idx]
    visible_sel = occluded_visible_mask[front_idx]
    clipped_occlusions = jnp.clip(total_occlusion, 0., jnp.where(visible_sel, cast_sel, 0.0))
    clipped_occlusions = jnp.where(visible_sel, clipped_occlusions, 0.0)
    total_occlusion = jnp.zeros_like(m_occluded.cast_areas).at[front_idx].set(clipped_occlusions)
    return total_occlusion


@partial(jax.jit, static_argnames=("n_neighbors",))
def resolve_occlusion(m_occluded: MeshModel, m_occluder: MeshModel, n_neighbors: int) -> MeshModel:
    """Calculate the occlusion of m_occluded by m_occluder

    Args:
        m_occluded (MeshModel): occluded mesh model
        m_occluder (MeshModel): occluding mesh model

    Returns:
        MeshModel: m1 with updated visible areas
    """
    m_occluder_center_z = cast_to_los(m_occluder.center, m_occluder.los_vector)
    m_occluded_center_z = cast_to_los(m_occluded.center, m_occluded.los_vector)
    o = jax.lax.cond(jnp.all(m_occluder_center_z>m_occluded_center_z),
                     lambda: _resolve_occlusion(m_occluded, m_occluder, n_neighbors),
                     lambda: jnp.zeros_like(m_occluded.occluded_areas, dtype=_float_dtype()))
    return m_occluded._replace(
        occluded_areas=o
    )
