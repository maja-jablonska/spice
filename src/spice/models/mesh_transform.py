from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_generation import face_center
from .mesh_model import MeshModel, DEFAULT_ROTATION_AXIS, create_harmonics_params
from spice.models.phoebe_model import PhoebeModel
from .utils import (cast_to_normal_plane, mesh_polar_vertices,
                    rotation_matrix, rotation_matrix_prim,
                    evaluate_rotation_matrix, evaluate_rotation_matrix_prim,
                    calculate_axis_radii,
                    evaluate_many_fouriers_for_value,
                    evaluate_many_fouriers_prim_for_value,
                    spherical_harmonic)


def _is_arraylike(x):
    return hasattr(x, '__array__') or hasattr(x, '__array_interface__')


@jax.jit
def _transform(mesh: MeshModel, vector: ArrayLike) -> MeshModel:
    cast_vector = cast_to_normal_plane(vector, mesh.los_vector)
    cast_vertices = cast_to_normal_plane(cast_vector + mesh.d_vertices, mesh.los_vector)
    return mesh._replace(center=vector)


def transform(mesh: MeshModel, vector: ArrayLike) -> MeshModel:
    """
        Transform the position of a mesh model based on a given vector.

        This function applies a transformation to the mesh model's position by updating its center
        with the provided vector. It checks if the mesh model is an instance of PhoebeModel and raises
        an error if so, as PHOEBE models are considered read-only within SPICE.

        Args:
            mesh (MeshModel): The mesh model to be transformed.
            vector (ArrayLike): The vector by which the mesh's position is to be updated.

        Returns:
            MeshModel: The transformed mesh model with its position updated.

        Raises:
            ValueError: If the mesh model is an instance of PhoebeModel, indicating that it is read-only.
        """
    if isinstance(mesh, PhoebeModel):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - the position is already evaluated in the PHOEBE model.")
    return _transform(mesh, vector)


@jax.jit
def _add_rotation(mesh: MeshModel,
                  rotation_velocity: ArrayLike,
                  rotation_axis: ArrayLike = DEFAULT_ROTATION_AXIS) -> MeshModel:
    rot_matrix = rotation_matrix(rotation_axis)
    rot_matrix_grad = rotation_matrix_prim(rotation_axis)

    return mesh._replace(rotation_axis=rotation_axis,
                         rotation_matrix=rot_matrix,
                         rotation_matrix_prim=rot_matrix_grad,
                         rotation_velocity=rotation_velocity)


def add_rotation(mesh: MeshModel,
                 rotation_velocity: ArrayLike,
                 rotation_axis: ArrayLike = DEFAULT_ROTATION_AXIS) -> MeshModel:
    """
    Adds rigid rotation to a mesh model based on specified velocity and axis.

    This function updates the mesh model with a new rotation velocity and axis. It checks if the mesh model
    is an instance of PhoebeModel and raises an error if so, indicating that PHOEBE models are read-only within
    SPICE and their rotation parameters cannot be modified.

    Args:
        mesh (MeshModel): The mesh model to add rotation to.
        rotation_velocity (ArrayLike): The velocity of the rotation.
        rotation_axis (ArrayLike): The axis of the rotation. Defaults to the global DEFAULT_ROTATION_AXIS.

    Returns:
        MeshModel: The mesh model with updated rotation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    return _add_rotation(mesh, rotation_velocity, rotation_axis)


@jax.jit
def _evaluate_rotation(mesh: MeshModel, t: ArrayLike) -> MeshModel:
    rotation_velocity_cm = mesh.rotation_velocity * 1e5
    theta = (rotation_velocity_cm * t) / mesh.radius  # cm
    t_rotation_matrix = evaluate_rotation_matrix(mesh.rotation_matrix, theta)  # cm
    t_rotation_matrix_prim = evaluate_rotation_matrix_prim(mesh.rotation_matrix_prim, theta)  # cm
    rotated_vertices = jnp.matmul(mesh.d_vertices, t_rotation_matrix)  # cm
    rotated_centers = jnp.matmul(mesh.d_centers, t_rotation_matrix)  # cm
    rotated_centers_vel = rotation_velocity_cm * jnp.matmul(mesh.d_centers / mesh.radius, t_rotation_matrix_prim)  # cm

    new_axis_radii = calculate_axis_radii(rotated_centers, mesh.rotation_axis)
    return mesh._replace(d_vertices=rotated_vertices,
                         d_centers=rotated_centers,
                         rotation_velocities=rotated_centers_vel * 1e-5,  # back to km/s
                         axis_radii=new_axis_radii)


def evaluate_rotation(mesh: MeshModel, t: ArrayLike) -> MeshModel:
    """
    Evaluate the rotation of a mesh model over a specified time.

    This function determines the rotation of a mesh model at a given time `t`. It checks if the mesh model
    is an instance of PhoebeModel and raises an error if so, indicating that PHOEBE models are considered
    read-only within the SPICE framework and their rotation cannot be evaluated or modified. For models
    that are not read-only, it delegates the rotation evaluation to the `_evaluate_rotation` function.

    Args:
        mesh (MeshModel): The mesh model to evaluate rotation for.
        t (ArrayLike): The time at which to evaluate the rotation.

    Returns:
        MeshModel: The mesh model with updated rotation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    return _evaluate_rotation(mesh, t)


def evaluate_body_orbit(m: MeshModel, orbital_velocity: float) -> MeshModel:
    """Evaluate the effects of an orbit on a mesh model

    Args:
        m (MeshModel): Mesh model of an orbiting body
        orbital_velocity (float): Orbital velocity in km/s

    Returns:
        MeshModel: Mesh model with updated parameters
    """
    return m._replace(orbital_velocity=orbital_velocity)


@partial(jax.jit, static_argnums=(4,))
def _add_pulsation(m: MeshModel, spherical_harmonics_parameters: ArrayLike,
                   pulsation_periods: ArrayLike, fourier_series_parameters: ArrayLike,
                   total_pad_len: int) -> MeshModel:
    harmonic_ind = spherical_harmonics_parameters[0] + m.max_pulsation_mode * spherical_harmonics_parameters[1]
    return m._replace(
        pulsation_periods=m.pulsation_periods.at[harmonic_ind].set(pulsation_periods),
        fourier_series_parameters=m.fourier_series_parameters.at[harmonic_ind].set(
            jnp.pad(fourier_series_parameters, ((0, total_pad_len), (0, 0)))
        )
    )


def add_pulsation(m: MeshModel, spherical_harmonics_parameters: ArrayLike,
                  period: float, fourier_series_parameters: ArrayLike) -> MeshModel:
    """
    Adds pulsation effects to a mesh model using spherical harmonics and Fourier series parameters.

    This function updates the mesh model's pulsation parameters based on the provided spherical harmonics
    and Fourier series parameters. It is designed to simulate pulsation effects on the mesh model, such as
    those caused by stellar oscillations or other periodic changes in shape or size.

    Args:
        m (MeshModel): The mesh model to add pulsation effects to.
        spherical_harmonics_parameters (ArrayLike): Parameters for the spherical harmonics, typically including
            the degree (l) and order (m) of the harmonics.
        period (float): Pulsation period
        fourier_series_parameters (ArrayLike): Dynamic parameters for the Fourier series that define the
            time-varying aspect of the pulsation.

    Returns:
        MeshModel: The mesh model with updated pulsation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only within SPICE.
    """
    if isinstance(m, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE.")
    if _is_arraylike(period):
        period = period[0]
    return _add_pulsation(m, spherical_harmonics_parameters, period,
                          fourier_series_parameters, m.max_fourier_order-fourier_series_parameters.shape[0])


@jax.jit
def _reset_pulsations(m: MeshModel) -> MeshModel:
    return m._replace(
        vertices_pulsation_offsets=jnp.zeros_like(m.vertices_pulsation_offsets),
        center_pulsation_offsets=jnp.zeros_like(m.center_pulsation_offsets),
        area_pulsation_offsets=jnp.zeros_like(m.area_pulsation_offsets),
        pulsation_periods=jnp.nan * jnp.ones_like(m.pulsation_periods),
        fourier_series_parameters=jnp.nan * jnp.ones_like(m.fourier_series_parameters)
    )


def reset_pulsations(m: MeshModel) -> MeshModel:
    """
    Resets the pulsation parameters of a mesh model to non-pulsating model values.

    This function resets the pulsation-related offsets and parameters of the mesh model to their default
    state, effectively removing any pulsation effects previously added. It checks if the mesh model is an
    instance of PhoebeModel and raises an error if so, as PHOEBE models are considered read-only within SPICE
    and their pulsation parameters cannot be reset.

    Args:
        m (MeshModel): The mesh model to reset pulsation parameters for.

    Returns:
        MeshModel: The mesh model with its pulsation parameters reset to default values.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating that it is read-only.
    """
    if isinstance(m, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE.")

    return _reset_pulsations(m)


@jax.jit
def calculate_pulsations(m: MeshModel, harmonic_parameters: ArrayLike, magnitude: float, magnitude_prim: float,
                         radius: float):
    polar_vertices = mesh_polar_vertices(m.d_vertices)
    polar_centers = mesh_polar_vertices(m.d_centers)
    vertex_harmonic_mags = spherical_harmonic(harmonic_parameters[0], harmonic_parameters[1], polar_vertices)[:,
                           jnp.newaxis]
    center_harmonic_mags = spherical_harmonic(harmonic_parameters[0], harmonic_parameters[1], polar_centers)[:,
                           jnp.newaxis]
    direction_vectors = m.d_vertices / jnp.linalg.norm(m.d_vertices, axis=1).reshape((-1, 1))
    center_direction_vectors = m.d_centers / jnp.linalg.norm(m.d_centers, axis=1).reshape((-1, 1))
    new_vertices = (radius * magnitude * vertex_harmonic_mags * direction_vectors)
    new_areas, new_centers = jax.jit(jax.vmap(face_center, in_axes=(None, 0)))(new_vertices + m.d_vertices,
                                                                               m.faces.astype(jnp.int32))
    return jnp.nan_to_num(new_vertices), jnp.nan_to_num(new_areas - m.base_areas), jnp.nan_to_num(
        new_centers - m.d_centers), (radius * magnitude_prim * center_harmonic_mags * center_direction_vectors)


def evaluate_pulsations(m: MeshModel, t: ArrayLike):
    """
    Evaluates and updates the mesh model with pulsation effects at a specific time.

    This function calculates the pulsation effects on a mesh model at a given time `t` by using
    Fourier series parameters for both static and dynamic components of the pulsation. It first checks if the
    mesh model is an instance of PhoebeModel, raising a ValueError if so, to enforce the read-only status of
    these models within the SPICE framework. The function then computes the pulsation magnitudes and velocities
    using Fourier series evaluations. These values are used along with spherical harmonics parameters to
    calculate offsets for vertices, areas, and centers of the mesh model, simulating the pulsation effects.
    Finally, the mesh model is updated with these calculated offsets and velocities.

    Args:
        m (MeshModel): The mesh model to evaluate pulsations for.
        t (ArrayLike): The time at which to evaluate the pulsations.

    Returns:
        MeshModel: The mesh model updated with pulsation effects.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only.
    """
    if isinstance(m, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE.")

    fourier_params = jnp.nan_to_num(m.fourier_series_parameters)
    pulsation_magnitudes = jnp.nan_to_num(
        evaluate_many_fouriers_for_value(
            m.pulsation_periods,
            fourier_params[:, :, 0],
            fourier_params[:, :, 1],
            t)
    )
    pulsation_velocity_magnitudes = jnp.nan_to_num(
        evaluate_many_fouriers_prim_for_value(
            m.pulsation_periods,
            fourier_params[:, :, 0],
            fourier_params[:, :, 1],
            t)
    )
    harmonic_params = create_harmonics_params(m.max_pulsation_mode)
    vert_offsets, area_offsets, center_offsets, pulsation_velocities = jax.vmap(calculate_pulsations,
                                                                                in_axes=(None, 0, 0, 0, None))(m,
                                                                                                               harmonic_params,
                                                                                                               pulsation_magnitudes,
                                                                                                               pulsation_velocity_magnitudes,
                                                                                                               m.radius)
    return m._replace(
        vertices_pulsation_offsets=jnp.sum(vert_offsets, axis=0),
        center_pulsation_offsets=jnp.sum(center_offsets, axis=0),
        area_pulsation_offsets=jnp.sum(area_offsets, axis=0),
        pulsation_velocities=jnp.sum(pulsation_velocities, axis=0) * 1e-5  # cm/s to km/s
    )
