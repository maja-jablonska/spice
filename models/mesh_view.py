import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .utils import cast_to_los, cast_normalized_to_los, cast_to_normal_plane
from .mesh_model import MeshModel

@jax.jit
def get_mesh_view(mesh: MeshModel, los_vector: ArrayLike) -> MeshModel:
    """Cast 3D vectors of centers and center velocities to the line-of-sight

    Args:
        mesh (MeshModel): Properties to be casted (n, 3)
        los_vector (ArrayLike): LOS vector (3,)

    Returns:
        MeshModel: mesh with updated los_vector, mus, and los_velocities
    """
    return mesh._replace(
        los_vector = los_vector,
        los_z=cast_to_los(mesh.center+mesh.d_centers, los_vector),
        cast_vertices = cast_to_normal_plane(mesh.vertices, los_vector),
        cast_centers = cast_to_normal_plane(mesh.centers, los_vector),
        mus = cast_normalized_to_los(mesh.d_centers, los_vector),
        los_velocities = cast_to_los(mesh.velocities, los_vector)
    )
