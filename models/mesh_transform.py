import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_model import MeshModel, DEFAULT_ROTATION_AXIS
from .utils import (cast_to_los, cast_normalized_to_los,
                    cast_to_normal_plane,
                    rotation_matrix, rotation_matrix_prim,
                    evaluate_rotation_matrix, evaluate_rotation_matrix_prim,
                    calculate_axis_radii)
from geometry.utils import get_cast_areas


@jax.jit
def transform(mesh: MeshModel, vector: ArrayLike) -> MeshModel:
    """Transform the mesh by a vector

    Args:
        mesh (MeshModel): Mesh to transform
        vector (ArrayLike): transform vector

    Returns:
        MeshModel: Mesh with transformed center and vertices coordinates
    """    
    cast_vector = cast_to_normal_plane(vector, mesh.los_vector)
    cast_vertices = cast_to_normal_plane(cast_vector+mesh.d_vertices, mesh.los_vector)
    return mesh._replace(center=vector,
                         los_z=cast_to_los(vector+mesh.d_centers, mesh.los_vector),
                         cast_vertices=cast_vertices,
                         cast_centers=cast_to_normal_plane(cast_vector+mesh.d_centers, mesh.los_vector),
                         cast_areas=get_cast_areas(cast_vertices[mesh.faces.astype(int)]))


@jax.jit
def add_rotation(mesh: MeshModel,
                 rotation_velocity: ArrayLike,
                 rotation_axis: ArrayLike = DEFAULT_ROTATION_AXIS) -> MeshModel:
    """Add a rigid rotation to the mesh model

    Args:
        mesh (MeshModel): mesh to add the rotation to
        rotation_velocity (ArrayLike): rotation velocity in km/s
        rotation_axis (ArrayLike, optional): rootation axis vector. Defaults to [0., 0., 1.].

    Returns:
        MeshModel: mesh with rotation added
    """
    rot_matrix = rotation_matrix(rotation_axis)
    rot_matrix_grad = rotation_matrix_prim(rotation_axis)
    
    return mesh._replace(rotation_axis = rotation_axis,
                         rotation_matrix = rot_matrix,
                         rotation_matrix_prim = rot_matrix_grad,
                         rotation_velocity = rotation_velocity)


@jax.jit
def evaluate_rotation(mesh: MeshModel, t: ArrayLike) -> MeshModel:
    """Evaluate the effects of a rotation at a given timestep t

    Args:
        mesh (MeshModel): mesh model of a body with a rigid rotation
        t (ArrayLike): timestep t

    Returns:
        MeshModel: mesh model with updated parameters
    """
    rotation_velocity_cm = mesh.rotation_velocity*1e5
    theta = (rotation_velocity_cm*t)/mesh.radius # cm
    t_rotation_matrix = evaluate_rotation_matrix(mesh.rotation_matrix, theta) # cm
    t_rotation_matrix_prim = evaluate_rotation_matrix_prim(mesh.rotation_matrix_prim, theta) # cm
    rotated_vertices = jnp.matmul(mesh.d_vertices, t_rotation_matrix) # cm
    rotated_centers = jnp.matmul(mesh.d_centers, t_rotation_matrix) # cm
    rotated_centers_vel = rotation_velocity_cm*jnp.matmul(mesh.d_centers/mesh.radius, t_rotation_matrix_prim) # cm

    new_axis_radii = calculate_axis_radii(rotated_centers, mesh.rotation_axis)
    cast_vertices = cast_to_normal_plane(mesh.center+rotated_vertices, mesh.los_vector)
    return mesh._replace(d_vertices = rotated_vertices,
                         d_centers = rotated_centers,
                         rotation_velocities = rotated_centers_vel*1e-5, # back to km/s
                         los_z=cast_to_los(mesh.center+rotated_centers, mesh.los_vector),
                         cast_vertices=cast_vertices,
                         cast_centers=cast_to_normal_plane(mesh.center+rotated_centers, mesh.los_vector),
                         mus = cast_normalized_to_los(rotated_centers, mesh.los_vector),
                         los_velocities = cast_to_los(rotated_centers_vel, mesh.los_vector)*1e-5,
                         axis_radii = new_axis_radii,
                         cast_areas=get_cast_areas(cast_vertices[mesh.faces.astype(int)]))


def evaluate_body_orbit(m: MeshModel, orbital_velocity: float) -> MeshModel:
    """Evaluate the effects of an orbit on a mesh model

    Args:
        m (MeshModel): mesh model of an orbiting body
        orbital_velocity (float): orbital velocity in km/s

    Returns:
        MeshModel: mesh model with updated parameters
    """
    m = m._replace(orbital_velocity=orbital_velocity)
    return m._replace(los_velocities=cast_to_los(m.velocities, m.los_vector))
