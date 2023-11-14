import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .utils import cast_to_los, cast_normalized_to_los, cast_to_normal_plane
from .mesh_model import MeshModel
from geometry import clip, polygon_area
from geometry.utils import get_cast_areas, cast_indexes, polygon_areas

@jax.jit
def get_mesh_view(mesh: MeshModel, los_vector: ArrayLike) -> MeshModel:
    """Cast 3D vectors of centers and center velocities to the line-of-sight

    Args:
        mesh (MeshModel): Properties to be casted (n, 3)
        los_vector (ArrayLike): LOS vector (3,)

    Returns:
        MeshModel: mesh with updated los_vector, mus, and los_velocities
    """
    cast_vertices=cast_to_normal_plane(mesh.vertices, los_vector)
    return mesh._replace(
        los_vector = los_vector,
        los_z=cast_to_los(mesh.center+mesh.d_centers, los_vector),
        cast_vertices = cast_vertices,
        cast_centers = cast_to_normal_plane(mesh.centers, los_vector),
        mus = cast_normalized_to_los(mesh.d_centers, los_vector),
        los_velocities = cast_to_los(mesh.velocities, los_vector),
        cast_areas=get_cast_areas(cast_vertices[mesh.faces.astype(int)])
    )

@jax.jit
def visible_area(vertices1: ArrayLike, vertices2: ArrayLike) -> ArrayLike:
    clipped = jnp.nan_to_num(clip(vertices1, vertices2))
    return polygon_area(clipped[:, 0], clipped[:, 1])

total_visible_area = jax.jit(jax.vmap(visible_area, in_axes=(None, 0)))
visibility_areas = jax.jit(jax.vmap(total_visible_area, in_axes=(0, None)))


import datetime

@jax.jit
def resolve_occlusion(mesh1: MeshModel, mesh2: MeshModel) -> ArrayLike:
    face_vertices1 = mesh1.cast_vertices[mesh1.faces.astype(int)]
    face_vertices2 = mesh2.cast_vertices[mesh2.faces.astype(int)]
    nonzero_indexer = cast_indexes(mesh1.cast_vertices)
    cast_vertices1 = face_vertices1[:, :, nonzero_indexer]
    cast_vertices2 = face_vertices2[:, :, nonzero_indexer]
    occluded = jnp.sum(jnp.where(mesh2.mus>0, visibility_areas(cast_vertices1, cast_vertices2), 0.), axis=1)
    return mesh1._replace(
            cast_areas=jnp.where(mesh1.mus>0, mesh1.cast_areas-occluded, 0)
        )
