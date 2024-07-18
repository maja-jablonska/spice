import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_generation import face_center
from .mesh_model import MeshModel, DEFAULT_ROTATION_AXIS, create_harmonics_params
from spice.models.phoebe_model import PhoebeModel
from .utils import (cast_to_los,
                    cast_to_normal_plane, mesh_polar_vertices,
                    rotation_matrix, rotation_matrix_prim,
                    evaluate_rotation_matrix, evaluate_rotation_matrix_prim,
                    calculate_axis_radii,
                    evaluate_many_fouriers_for_value,
                    evaluate_many_fouriers_prim_for_value,
                    spherical_harmonic)
from spice.geometry.utils import get_cast_areas


@jax.jit
def _transform(mesh: MeshModel, vector: ArrayLike) -> MeshModel:
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
    

def transform(mesh: MeshModel, vector: ArrayLike) -> MeshModel:
    """Transform the mesh by a vector

    Args:
        mesh (MeshModel): Mesh to transform
        vector (ArrayLike): transform vector

    Returns:
        MeshModel: Mesh with transformed center and vertices coordinates
    """   
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE - the position is already evaluated in the PHOEBE model.")
    return _transform(mesh, vector)


@jax.jit
def _add_rotation(mesh: MeshModel,
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
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    return _add_rotation(mesh, rotation_velocity, rotation_axis)

@jax.jit
def _evaluate_rotation(mesh: MeshModel, t: ArrayLike) -> MeshModel:
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
    return mesh._replace(d_vertices = rotated_vertices,
                         d_centers = rotated_centers,
                         rotation_velocities = rotated_centers_vel*1e-5, # back to km/s
                         axis_radii = new_axis_radii)
    

def evaluate_rotation(mesh: MeshModel, t: ArrayLike) -> MeshModel:
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    return _evaluate_rotation(mesh, t)


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


def add_pulsation(m: MeshModel, spherical_harmonics_parameters: ArrayLike,
                  fourier_series_static_parameters: ArrayLike, fourier_series_parameters: ArrayLike) -> MeshModel:
    ind = spherical_harmonics_parameters[0] + m.max_pulsation_mode*spherical_harmonics_parameters[1]
    return m._replace(
        fourier_series_static_parameters=m.fourier_series_static_parameters.at[ind].set(fourier_series_static_parameters),
        fourier_series_parameters=m.fourier_series_parameters.at[ind].set(fourier_series_parameters)
    )
    
    
def reset_pulsations(m: MeshModel) -> MeshModel:
    return m._replace(
        vertices_pulsation_offsets=jnp.zeros_like(m.vertices_pulsation_offsets),
        center_pulsation_offsets=jnp.zeros_like(m.center_pulsation_offsets),
        area_pulsation_offsets=jnp.zeros_like(m.area_pulsation_offsets),
        fourier_series_static_parameters=jnp.nan*jnp.ones_like(m.fourier_series_static_parameters),
        fourier_series_parameters=jnp.nan*jnp.ones_like(m.fourier_series_parameters)
    )
    
@jax.jit
def calculate_pulsations(m: MeshModel, harmonic_parameters: ArrayLike, magnitude: float, magnitude_prim: float, radius: float):
    polar_vertices = mesh_polar_vertices(m.d_vertices)
    polar_centers = mesh_polar_vertices(m.d_centers)
    vertex_harmonic_mags = spherical_harmonic(harmonic_parameters[0], harmonic_parameters[1], polar_vertices)[:, jnp.newaxis]
    center_harmonic_mags = spherical_harmonic(harmonic_parameters[0], harmonic_parameters[1], polar_centers)[:, jnp.newaxis]
    direction_vectors = m.d_vertices/jnp.linalg.norm(m.d_vertices, axis=1).reshape((-1, 1))
    center_direction_vectors = m.d_centers/jnp.linalg.norm(m.d_centers, axis=1).reshape((-1, 1))
    new_vertices = (radius*magnitude*vertex_harmonic_mags*direction_vectors)
    new_areas, new_centers = jax.jit(jax.vmap(face_center, in_axes=(None, 0)))(new_vertices+m.d_vertices, m.faces.astype(jnp.int32))
    return jnp.nan_to_num(new_vertices), jnp.nan_to_num(new_areas-m.base_areas), jnp.nan_to_num(new_centers-m.d_centers), (radius*magnitude_prim*center_harmonic_mags*center_direction_vectors)

def evaluate_pulsations(m: MeshModel, t: ArrayLike):
    fourier_static_params = jnp.nan_to_num(m.fourier_series_static_parameters)
    fourier_params = jnp.nan_to_num(m.fourier_series_parameters)
    pulsation_magnitudes = jnp.nan_to_num(
        evaluate_many_fouriers_for_value(
            fourier_static_params[:, 0],
            fourier_static_params[:, 1],
            fourier_params[:, :, 0],
            fourier_params[:, :, 1],
            t)
        )
    pulsation_velocity_magnitudes = jnp.nan_to_num(
        evaluate_many_fouriers_prim_for_value(
            fourier_static_params[:, 0],
            fourier_static_params[:, 1],
            fourier_params[:, :, 0],
            fourier_params[:, :, 1],
            t)
        )
    harmonic_params = create_harmonics_params(m.max_pulsation_mode)
    vert_offsets, area_offsets, center_offsets, pulsation_velocities = jax.vmap(calculate_pulsations, in_axes=(None, 0, 0, 0, None))(m, harmonic_params, pulsation_magnitudes, pulsation_velocity_magnitudes, m.radius)
    return m._replace(
        vertices_pulsation_offsets=jnp.sum(vert_offsets, axis=0),
        center_pulsation_offsets=jnp.sum(center_offsets, axis=0),
        area_pulsation_offsets=jnp.sum(area_offsets, axis=0),
        pulsation_velocities=jnp.sum(pulsation_velocities, axis=0)*1e-5 # cm/s to km/s
    )
