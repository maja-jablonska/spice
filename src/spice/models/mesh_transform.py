from functools import partial
from typing import List, Union
import warnings

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from .mesh_generation import face_center
from .mesh_model import MeshModel, DEFAULT_ROTATION_AXIS, create_harmonics_params
from spice.models.phoebe_model import PhoebeModel
from .utils import (ModelList, rotation_matrix, rotation_matrix_prim,
                    evaluate_rotation_matrix, evaluate_rotation_matrix_prim,
                    calculate_axis_radii,
                    evaluate_many_fouriers_for_value,
                    evaluate_many_fouriers_prim_for_value,
                    spherical_harmonic_with_tilt)

from jaxtyping import Array, Float


def _is_arraylike(x):
    return hasattr(x, '__array__') or hasattr(x, '__array_interface__')


@jax.jit
def _transform(mesh: MeshModel, vector: ArrayLike) -> MeshModel:
    return mesh._replace(center=vector)


def transform(mesh: MeshModel, vector: Float[Array, "3"]) -> MeshModel:
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
def _update_parameter(mesh: MeshModel, parameter_index: int, parameter_values: Float[Array, "n_mesh_elements n_parameters"]) -> MeshModel:
    return mesh._replace(parameters=mesh.parameters.at[:, parameter_index].set(parameter_values))


def update_parameter(mesh: MeshModel, parameter: Union[str, int, ArrayLike], parameter_values: Float[Array, "n_mesh_elements n_parameters"], parameter_names: List[str] = None) -> MeshModel:
    """
    Update a specific parameter or set of parameters in the mesh model.

    This function allows updating one or multiple parameters of the mesh model. It can handle
    parameter specification by name (string), index (integer), or an array-like of indices.

    Args:
        mesh (MeshModel): The mesh model to be updated.
        parameter (Union[str, int, ArrayLike]): The parameter(s) to update. Can be:
            - A string representing the parameter name.
            - An integer representing the parameter index.
            - An array-like of integers representing multiple parameter indices.
        parameter_values (Float[Array, "n_mesh_elements n_parameters"]): The new value(s) for the specified parameter(s).
            Should match the shape of the parameter specification.
        parameter_names (List[str]): A list of parameter names used for the model while constructing the mesh.

    Returns:
        MeshModel: The updated mesh model with the new parameter value(s).

    Raises:
        ValueError: If the specified parameter name is not found in the mesh model's parameter list.

    Note:
        If the mesh is an instance of PhoebeModel, this function will raise a ValueError as PHOEBE models
        are considered read-only in SPICE.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - parameters cannot be updated.")

    if isinstance(mesh, PhoebeModel):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - parameters cannot be updated.")
    if isinstance(parameter, str):
        if parameter_names is None:
            raise ValueError(
                "A list of parameters used for the model while constructing the mesh must be provided to update parameters by name.")
        if parameter not in mesh.parameter_names:
            raise ValueError(
                f"Parameter {parameter} not found in mesh model. Model contains parameters: {mesh.parameter_names}")
        parameter_index = parameter_names.index(parameter)
    else:
        parameter_index = parameter
    return _update_parameter(mesh, parameter_index, parameter_values)


@jax.jit
def _update_parameters(mesh: MeshModel, parameter_indices: ArrayLike, parameter_values: ArrayLike) -> MeshModel:
    def update_single_parameter(params, idx_val):
        idx, val = idx_val
        return params.at[:, idx].set(val)

    updated_parameters = jax.lax.fori_loop(
        0, len(parameter_indices),
        lambda i, params: update_single_parameter(
            params, (parameter_indices[i], parameter_values[i])),
        mesh.parameters
    )
    return mesh._replace(parameters=updated_parameters)


def update_parameters(mesh: MeshModel, parameters: Union[List[str], List[int]], parameter_values: Float[Array, "n_mesh_elements n_parameters"], parameter_names: List[str] = None) -> MeshModel:
    """
    Update multiple parameters in the mesh model simultaneously.

    This function allows updating multiple parameters of the mesh model at once. It can handle
    parameter specification by names (list of strings) or indices (list of integers).

    Args:
        mesh (MeshModel): The mesh model to be updated.
        parameters (Union[List[str], List[int]]): The parameters to update. Can be:
            - A list of strings representing parameter names.
            - A list of integers representing parameter indices.
        parameter_values (Float[Array, "n_mesh_elements n_parameters"]): The new values for the specified parameters.
            Should be an array-like with the same length as the parameters list.
        parameter_names (List[str]): A list of parameter names used for the model while constructing the mesh.

    Returns:
        MeshModel: The updated mesh model with the new parameter values.

    Raises:
        ValueError: If any specified parameter name is not found in the mesh model's parameter list,
                    or if the mesh is an instance of PhoebeModel.

    Note:
        This function is more efficient than calling update_parameter multiple times when
        updating several parameters at once.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - parameters cannot be updated.")

    if not isinstance(parameters, (list, tuple, np.ndarray, jnp.ndarray)):
        raise ValueError(
            "Parameters must be a list, tuple, or array-like object.")

    if len(parameters) != len(parameter_values):
        raise ValueError(
            "The number of parameters must match the number of parameter values.")

    if isinstance(parameters[0], str):
        if parameter_names is None:
            raise ValueError(
                "A list of parameters used for the model while constructing the mesh must be provided to update parameters by name.")
        parameter_indices = []
        for param in parameters:
            if param not in parameter_names:
                raise ValueError(
                    f"Parameter {param} not found in mesh model. Model contains parameters: {mesh.parameter_names}")
            parameter_indices.append(parameter_names.index(param))
    else:
        parameter_indices = parameters

    return _update_parameters(mesh, jnp.array(parameter_indices), jnp.array(parameter_values))


@jax.jit
def _add_rotation(mesh: MeshModel,
                  rotation_velocity: float,
                  rotation_axis: Float[Array, "3"] = DEFAULT_ROTATION_AXIS) -> MeshModel:
    rot_matrix = rotation_matrix(rotation_axis)
    rot_matrix_grad = rotation_matrix_prim(rotation_axis)

    return mesh._replace(rotation_axis=rotation_axis,
                         rotation_matrix=rot_matrix,
                         rotation_matrix_prim=rot_matrix_grad,
                         rotation_velocity=rotation_velocity)


def add_rotation(mesh: MeshModel,
                 rotation_velocity: float,
                 rotation_axis: Float[Array, "3"] = DEFAULT_ROTATION_AXIS) -> MeshModel:
    """
    Adds rigid rotation to a mesh model based on specified velocity and axis.

    This function updates the mesh model with a new rotation velocity and axis. It checks if the mesh model
    is an instance of PhoebeModel and raises an error if so, indicating that PHOEBE models are read-only within
    SPICE and their rotation parameters cannot be modified.

    Args:
        mesh (MeshModel): The mesh model to add rotation to.
        rotation_velocity (float): The velocity of the rotation.
        rotation_axis (Float[Array, "3"]): The axis of the rotation. Defaults to the global [0., 0., 1.].

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
    rotation_velocity_km_s = mesh.rotation_velocity
    theta = (rotation_velocity_km_s * t) / (mesh.radius * 695700.0)
    
    t_rotation_matrix = evaluate_rotation_matrix(mesh.rotation_matrix, theta)
    
    t_rotation_matrix_prim = evaluate_rotation_matrix_prim(mesh.rotation_matrix_prim, theta)
    
    rotated_vertices = jnp.matmul(mesh.d_vertices, t_rotation_matrix)
    
    rotated_centers = jnp.matmul(mesh.d_centers, t_rotation_matrix)
    
    rotated_centers_vel = rotation_velocity_km_s * jnp.matmul(mesh.d_centers/mesh.radius, t_rotation_matrix_prim)

    new_axis_radii = calculate_axis_radii(rotated_centers, mesh.rotation_axis)
    return mesh._replace(d_vertices=rotated_vertices,
                         d_centers=rotated_centers,
                         rotation_velocities=rotated_centers_vel,  # back to km/s
                         axis_radii=new_axis_radii)


def evaluate_rotation(mesh: MeshModel, t: float) -> MeshModel:
    """
    Evaluate the rotation of a mesh model over a specified time.

    This function determines the rotation of a mesh model at a given time `t`. It checks if the mesh model
    is an instance of PhoebeModel and raises an error if so, indicating that PHOEBE models are considered
    read-only within the SPICE framework and their rotation cannot be evaluated or modified. For models
    that are not read-only, it delegates the rotation evaluation to the `_evaluate_rotation` function.

    Args:
        mesh (MeshModel): The mesh model to evaluate rotation for.
        t (float): The time at which to evaluate the rotation. [seconds]

    Returns:
        MeshModel: The mesh model with updated rotation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    return _evaluate_rotation(mesh, t)


_evaluate_rotation_v = jax.vmap(_evaluate_rotation, in_axes=(None, 0))


def evaluate_rotation_at_times(mesh: MeshModel, times: ArrayLike) -> ArrayLike:
    result_bodies = _evaluate_rotation_v(mesh, times)
    # Convert the  results to lists for each component
    body_list = jax.tree_util.tree_map(lambda x: list(x), result_bodies)
    return ModelList(jax.tree_util.tree_transpose(
        outer_treedef=jax.tree_util.tree_structure(mesh),
        inner_treedef=jax.tree_util.tree_structure([0 for _ in range(len(times))]),
        pytree_to_transpose=body_list
    ))


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
                   total_pad_len: int,
                   pulsation_axes: ArrayLike, pulsation_angles: ArrayLike) -> MeshModel:
    harmonic_ind = spherical_harmonics_parameters[0] + \
        m.max_pulsation_mode * spherical_harmonics_parameters[1]
    return m._replace(
        pulsation_periods=m.pulsation_periods.at[harmonic_ind].set(
            pulsation_periods),
        fourier_series_parameters=m.fourier_series_parameters.at[harmonic_ind].set(
            jnp.pad(fourier_series_parameters, ((0, total_pad_len), (0, 0)))
        ),
        pulsation_axes=m.pulsation_axes.at[harmonic_ind].set(pulsation_axes),
        pulsation_angles=m.pulsation_angles.at[harmonic_ind].set(
            pulsation_angles)
    )


def add_pulsation(m: MeshModel, m_order: Float, n_degree: Float,
                  period: Float, fourier_series_parameters: Float[Array, "n_terms 2"],
                  pulsation_axes: Float[Array, "3"] = None, pulsation_angles: Float = None) -> MeshModel:
    """
    Adds pulsation effects to a mesh model using spherical harmonics and Fourier series parameters.

    This function updates the mesh model's pulsation parameters based on the provided spherical harmonics
    and Fourier series parameters. It is designed to simulate pulsation effects on the mesh model, such as
    those caused by stellar oscillations or other periodic changes in shape or size.

    Args:
        m (MeshModel): The mesh model to add pulsation effects to.
        m_order (Float): The order (m) of the spherical harmonics.
        n_degree (Float): The degree (n) of the spherical harmonics.
        period (Float): Pulsation period in seconds.
        fourier_series_parameters (Float[Array, "n_terms 2"]): Dynamic parameters for the Fourier series that define the
            time-varying aspect of the pulsation. Shape should be (N, 2) where N is the number of terms,
            and each row contains [amplitude, phase] for that term.
        pulsation_axes (Float[Array, "3"]): Axes of the pulsation. Defaults to the rotation axis of the mesh model.
        pulsation_angles (Float): Angles of the pulsation. Defaults to zero.

    Returns:
        MeshModel: The mesh model with updated pulsation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only within SPICE.
    """
    if isinstance(m, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE.")
    if _is_arraylike(period):
        period = period[0]

    if pulsation_axes is None:
        pulsation_axes = m.rotation_axis
    if pulsation_angles is None:
        pulsation_angles = 0.

    harmonic_ind = m_order + m.max_pulsation_mode * n_degree
    if harmonic_ind >= len(m.pulsation_periods):
        warnings.warn("Pulsation mode is too high for the mesh model - the mesh model has" +
                      "been initialized with a maximum pulsation mode of {}".format(m.max_pulsation_mode) +
                      "This pulsation will have no effect.")

    return _add_pulsation(m, jnp.array([m_order, n_degree]), period, fourier_series_parameters,
                          int(m.max_fourier_order -
                              fourier_series_parameters.shape[0]),
                          pulsation_axes, pulsation_angles)


@partial(jax.jit, static_argnums=(4,))
def _add_pulsations(m: MeshModel,
                    pulsation_periods: ArrayLike,
                    fourier_series_parameters: ArrayLike,
                    harmonic_indices: ArrayLike,
                    total_pad_len: int,
                    pulsation_axes: ArrayLike,
                    pulsation_angles: ArrayLike) -> MeshModel:

    def update_pulsation(carry, inputs):
        m, harmonic_ind, period, fourier_params, pulsation_axes, pulsation_angles = carry, *inputs
        
        # Better handling of shape for fourier_params
        # Check if fourier_params is already 2D with shape (n, 2)
        if len(fourier_params.shape) == 2 and fourier_params.shape[1] == 2:
            reshape_params = fourier_params
        else:
            # Safely reshape to (-1, 2), ensuring we have pairs of values
            reshape_params = fourier_params.reshape((-1, 2))
            
        # Use the total_pad_len parameter that was passed to the function
        # This avoids dynamic computation inside the jitted function
        padded_fourier_params = jnp.pad(reshape_params, ((0, total_pad_len), (0, 0)))

        new_periods = m.pulsation_periods.at[harmonic_ind].set(period)
        new_fourier_params = m.fourier_series_parameters.at[harmonic_ind].set(
            padded_fourier_params)
        new_axes = m.pulsation_axes.at[harmonic_ind].set(pulsation_axes)
        new_angles = m.pulsation_angles.at[harmonic_ind].set(pulsation_angles)

        return m._replace(pulsation_periods=new_periods,
                          fourier_series_parameters=new_fourier_params,
                          pulsation_axes=new_axes,
                          pulsation_angles=new_angles), None

    updated_m, _ = jax.lax.scan(update_pulsation, m,
                                (harmonic_indices,
                                 pulsation_periods, fourier_series_parameters,
                                 pulsation_axes, pulsation_angles))

    return updated_m


def add_pulsations(m: MeshModel, m_orders: Float[Array, "n_pulsations"], n_degrees: Float[Array, "n_pulsations"],
                   periods: Float[Array, "n_pulsations"], fourier_series_parameters: Float[Array, "n_pulsations n_terms 2"],
                   pulsation_axes: Float[Array, "n_pulsations 3"] = None, pulsation_angles: Float[Array, "n_pulsations"] = None) -> MeshModel:
    """
    Adds multiple pulsation effects to a mesh model using spherical harmonics and Fourier series parameters.

    This function updates the mesh model's pulsation parameters based on the provided arrays of spherical harmonics
    and Fourier series parameters. It uses JAX's vmap capabilities for efficient computation.

    Args:
        m (MeshModel): The mesh model to add pulsation effects to.
        m_orders (Float[Array, "n_pulsations"]): Array of orders (m) of the spherical harmonics.
        n_degrees (Float[Array, "n_pulsations"]): Array of degrees (n) of the spherical harmonics.
        periods (Float[Array, "n_pulsations"]): Array of pulsation periods in seconds.
        fourier_series_parameters (Float[Array, "n_pulsations n_terms 2"]): Array of dynamic parameters for the Fourier series that define the
            time-varying aspect of the pulsations. Shape should be (K, N, 2) where K is the number of pulsations,
            N is the number of terms for each pulsation, and each inner array contains [amplitude, phase] for that term.
        pulsation_axes (Float[Array, "n_pulsations 3"]): Array of pulsation axes. Defaults to the rotation axis of the mesh model.
        pulsation_angles (Float[Array, "n_pulsations"]): Array of pulsation angles. Defaults to zero.

    Returns:
        MeshModel: The mesh model with updated pulsation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only within SPICE.
        ValueError: If the input arrays have inconsistent lengths.
    """
    if isinstance(m, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE.")

    if not (len(m_orders) == len(n_degrees) == len(periods) == len(fourier_series_parameters)):
        raise ValueError("Input arrays must have consistent lengths.")

    if pulsation_axes is None:
        pulsation_axes = m.rotation_axis.reshape(
            (1, 3)).repeat(len(m_orders), axis=0)
    if pulsation_angles is None:
        pulsation_angles = jnp.zeros_like(m_orders)

    harmonic_indices = m_orders + m.max_pulsation_mode * n_degrees
    
    # Fix inconsistency: fourier_series_parameters shape is (n_pulsations, n_terms, 2)
    # We need the second dimension (n_terms) for padding calculation
    total_pad_len = int(m.max_fourier_order -
                        fourier_series_parameters.shape[1] + 1)

    return _add_pulsations(m, periods,
                           fourier_series_parameters, harmonic_indices, total_pad_len,
                           pulsation_axes, pulsation_angles)


@jax.jit
def _reset_pulsations(m: MeshModel) -> MeshModel:
    return m._replace(
        vertices_pulsation_offsets=jnp.zeros_like(
            m.vertices_pulsation_offsets),
        center_pulsation_offsets=jnp.zeros_like(m.center_pulsation_offsets),
        area_pulsation_offsets=jnp.zeros_like(m.area_pulsation_offsets),
        pulsation_periods=jnp.nan * jnp.ones_like(m.pulsation_periods),
        fourier_series_parameters=jnp.nan *
        jnp.ones_like(m.fourier_series_parameters),
        pulsation_axes=DEFAULT_ROTATION_AXIS.reshape(
            (1, 3)).repeat(m.pulsation_periods.shape[0], axis=0),
        pulsation_angles=jnp.zeros_like(m.pulsation_periods)
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
                         radius: float, pulsation_axis: ArrayLike, pulsation_angle: ArrayLike):
    # Ensure pulsation_angle is treated as scalar if it's a single value in an array
    # Use jnp.where to handle the scalar extraction in a jax-friendly way
    scalar_pulsation_angle = jnp.where(
        jnp.size(pulsation_angle) == 1,
        pulsation_angle.reshape(()),  # Reshape to scalar
        pulsation_angle  # Keep as is if not a single-element array
    )
    
    vertex_harmonic_mags = spherical_harmonic_with_tilt(harmonic_parameters[0], harmonic_parameters[1], m.d_vertices,
                                                        pulsation_axis, scalar_pulsation_angle)[:,
                                                                                         jnp.newaxis]
    center_harmonic_mags = spherical_harmonic_with_tilt(harmonic_parameters[0], harmonic_parameters[1], m.d_centers,
                                                        pulsation_axis, scalar_pulsation_angle)[:,
                                                                                         jnp.newaxis]
    direction_vectors = m.d_vertices / \
        jnp.linalg.norm(m.d_vertices, axis=1).reshape((-1, 1))
    center_direction_vectors = m.d_centers / \
        jnp.linalg.norm(m.d_centers, axis=1).reshape((-1, 1))
    new_vertices = (radius * magnitude *
                    vertex_harmonic_mags * direction_vectors)
    new_areas, new_centers = jax.jit(jax.vmap(face_center, in_axes=(None, 0)))(new_vertices + m.d_vertices,
                                                                               m.faces.astype(jnp.int32))
    return jnp.nan_to_num(new_vertices), jnp.nan_to_num(new_areas - m.base_areas), jnp.nan_to_num(
        new_centers - m.d_centers), jnp.nan_to_num(radius * magnitude_prim * center_harmonic_mags * center_direction_vectors)


def calculate_numerical_pulsation_velocity(m: MeshModel, t: float, dt: float = 1e-6):
    """
    Calculates pulsation velocity using numerical differentiation.
    
    This function approximates the derivative of the pulsation displacement with respect to time
    using a forward difference method: [r(t+dt) - r(t)]/dt
    
    Args:
        m (MeshModel): The mesh model to calculate pulsation velocities for
        t (float): The current time in days
        dt (float, optional): The time step for numerical differentiation in days. Defaults to 1e-6.
        
    Returns:
        tuple: (pulsation_magnitudes, pulsation_velocity_magnitudes) where pulsation_velocity_magnitudes
               are calculated numerically
    """
    # Apply nan_to_num to inputs to prevent NaN propagation
    t = jnp.nan_to_num(t)
    
    # Safe extraction of Fourier parameters
    fourier_params = jnp.nan_to_num(m.fourier_series_parameters)
    
    # Calculate pulsation magnitudes at time t
    pulsation_magnitudes_t = evaluate_many_fouriers_for_value(
        jnp.nan_to_num(m.pulsation_periods),
        fourier_params[:, :, 0],
        fourier_params[:, :, 1],
        t
    )
    
    # Calculate pulsation magnitudes at time t+dt
    pulsation_magnitudes_t_plus_dt = evaluate_many_fouriers_for_value(
        jnp.nan_to_num(m.pulsation_periods),
        fourier_params[:, :, 0],
        fourier_params[:, :, 1],
        t + dt
    )
    
    # Calculate numerical derivative
    pulsation_velocity_magnitudes = (pulsation_magnitudes_t_plus_dt - pulsation_magnitudes_t) / dt
    
    return pulsation_magnitudes_t, pulsation_velocity_magnitudes


def evaluate_pulsations(m: MeshModel, t: float, use_numerical_derivative: bool = False, dt: float = 1e-6):
    """
    Evaluates and updates the mesh model with pulsation effects at a specific time.

    This function calculates the pulsation effects on a mesh model at a given time `t` by using
    Fourier series parameters for both static and dynamic components of the pulsation. It first checks if the
    mesh model is an instance of PhoebeModel, raising a ValueError if so, to enforce the read-only status of
    these models within the SPICE framework. The function then computes the pulsation magnitudes and velocities
    using either analytical or numerical derivatives of Fourier series evaluations. These values are used 
    along with spherical harmonics parameters to calculate offsets for vertices, areas, and centers of the mesh model, 
    simulating the pulsation effects. Finally, the mesh model is updated with these calculated offsets and velocities.

    Args:
        m (MeshModel): The mesh model to evaluate pulsations for.
        t (float): The time at which to evaluate the pulsations, in days.
        use_numerical_derivative (bool, optional): Whether to use numerical differentiation for velocity calculation.
            Defaults to False (use analytical derivative).
        dt (float, optional): Time step for numerical differentiation when use_numerical_derivative is True.
            Defaults to 1e-6 days.

    Returns:
        MeshModel: The mesh model updated with pulsation effects.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only.
    """
    if isinstance(m, PhoebeModel):
        raise ValueError("PHOEBE models are read-only in SPICE.")

    # Calculate pulsation magnitudes and velocities
    if use_numerical_derivative:
        # Use numerical differentiation
        pulsation_magnitudes, pulsation_velocity_magnitudes = calculate_numerical_pulsation_velocity(m, t, dt)
    else:
        # Use analytical differentiation (original method)
        # Apply nan_to_num to inputs to prevent NaN propagation
        t = jnp.nan_to_num(t)
        
        # Safe extraction of Fourier parameters
        fourier_params = jnp.nan_to_num(m.fourier_series_parameters)
        
        # Safely compute pulsation magnitudes
        pulsation_magnitudes = evaluate_many_fouriers_for_value(
            jnp.nan_to_num(m.pulsation_periods),
            fourier_params[:, :, 0],
            fourier_params[:, :, 1],
            t
        )
        
        # Safely compute pulsation velocity magnitudes
        pulsation_velocity_magnitudes = evaluate_many_fouriers_prim_for_value(
            jnp.nan_to_num(m.pulsation_periods),
            fourier_params[:, :, 0],
            fourier_params[:, :, 1],
            t
        )
    
    # Create harmonic parameters
    harmonic_params = create_harmonics_params(m.max_pulsation_mode)
    
    # Apply vmap to calculate pulsations for each harmonic
    vert_offsets, area_offsets, center_offsets, pulsation_velocities = jax.vmap(calculate_pulsations,
                                                                              in_axes=(None, 0, 0, 0, None, 0, 0))(m,
                                                                                                                   harmonic_params,
                                                                                                                   pulsation_magnitudes,
                                                                                                                   pulsation_velocity_magnitudes,
                                                                                                                   m.radius,
                                                                                                                   m.pulsation_axes,
                                                                                                                   m.pulsation_angles)
    # Use nan_to_num for safer summation
    return m._replace(
        vertices_pulsation_offsets=jnp.nan_to_num(jnp.sum(vert_offsets, axis=0)),
        center_pulsation_offsets=jnp.nan_to_num(jnp.sum(center_offsets, axis=0)),
        area_pulsation_offsets=jnp.nan_to_num(jnp.sum(area_offsets, axis=0)),
        pulsation_velocities=jnp.nan_to_num(
            jnp.sum(pulsation_velocities, axis=0) * 8.052083333333332  # solRad/day to km/s
        )
    )
