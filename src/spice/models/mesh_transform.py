from functools import partial
from importlib import import_module
from typing import List, Union
import warnings

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from .mesh_generation import face_center
from .mesh_model import MeshModel, _default_rotation_axis, create_harmonics_params
try:
    from spice.models.phoebe_model import PhoebeModel
except Exception:
    class PhoebeModel:
        """Placeholder when PHOEBE is not installed or broken."""
        pass
from .utils import (ModelList, rotation_matrix, rotation_matrix_prim,
                    evaluate_rotation_matrix,
                    calculate_axis_radii,
                    evaluate_many_fouriers_for_value,
                    evaluate_many_fouriers_prim_for_value,
                    spherical_harmonic_with_tilt,
                    spherical_harmonic_gradient_with_tilt,
                    spherical_harmonic_toroidal_with_tilt)

from jaxtyping import Array, Float

_PHOEBE_MODEL_CLASS = None
_PHOEBE_MODEL_RESOLVED = False


def _is_arraylike(x):
    return hasattr(x, '__array__') or hasattr(x, '__array_interface__')


def _get_phoebe_model_class():
    """Lazily resolve PhoebeModel class from optional PHOEBE integration."""
    global _PHOEBE_MODEL_CLASS, _PHOEBE_MODEL_RESOLVED

    if not _PHOEBE_MODEL_RESOLVED:
        try:
            module = import_module(".phoebe_model", "spice.models")
            _PHOEBE_MODEL_CLASS = module.PhoebeModel
        except ImportError:
            _PHOEBE_MODEL_CLASS = None
        _PHOEBE_MODEL_RESOLVED = True

    return _PHOEBE_MODEL_CLASS


def _is_phoebe_model(mesh: MeshModel) -> bool:
    # Fast path: avoid importing optional PHOEBE integration for non-PHOEBE meshes.
    if type(mesh).__name__ != "PhoebeModel":
        return False

    phoebe_model_class = _get_phoebe_model_class()
    return phoebe_model_class is not None and isinstance(mesh, phoebe_model_class)


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
    if _is_phoebe_model(mesh):
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
    if _is_phoebe_model(mesh):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - parameters cannot be updated.")

    if _is_phoebe_model(mesh):
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
    if _is_phoebe_model(mesh):
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
                  rotation_axis: Float[Array, "3"] = None) -> MeshModel:
    if rotation_axis is None:
        rotation_axis = _default_rotation_axis()
    rot_matrix = rotation_matrix(rotation_axis)
    rot_matrix_grad = rotation_matrix_prim(rotation_axis)

    return mesh._replace(rotation_axis=rotation_axis,
                         rotation_matrix=rot_matrix,
                         rotation_matrix_prim=rot_matrix_grad,
                         rotation_velocity=rotation_velocity)


def add_rotation(mesh: MeshModel,
                 rotation_velocity: float,
                 rotation_axis: Float[Array, "3"] = None) -> MeshModel:
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
    if _is_phoebe_model(mesh):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    if rotation_axis is None:
        rotation_axis = _default_rotation_axis()
    return _add_rotation(mesh, rotation_velocity, rotation_axis)


@jax.jit
def _evaluate_rotation(mesh: MeshModel, t: ArrayLike) -> MeshModel:
    # Preserve user rotation direction; LOS sign convention is handled when
    # projecting velocities to the line of sight.
    rotation_velocity_km_s = mesh.rotation_velocity
    theta = (rotation_velocity_km_s * t) / (mesh.radius * 695700.0)

    t_rotation_matrix = evaluate_rotation_matrix(mesh.rotation_matrix, theta)

    rotated_vertices = jnp.matmul(mesh.d_vertices, t_rotation_matrix)
    rotated_centers = jnp.matmul(mesh.d_centers, t_rotation_matrix)

    axis_hat = mesh.rotation_axis / jnp.linalg.norm(mesh.rotation_axis)

    vertex_norms = jnp.linalg.norm(rotated_vertices, axis=1, keepdims=True)
    center_norms = jnp.linalg.norm(rotated_centers, axis=1, keepdims=True)

    surface_vertices = rotated_vertices * jnp.where(vertex_norms > 0.0, mesh.radius / vertex_norms, 1.0)
    surface_centers = rotated_centers * jnp.where(center_norms > 0.0, mesh.radius / center_norms, 1.0)

    rotated_vertices_vel = (rotation_velocity_km_s / mesh.radius) * jnp.cross(surface_vertices, axis_hat)
    rotated_centers_vel = (rotation_velocity_km_s / mesh.radius) * jnp.cross(surface_centers, axis_hat)

    face_indices = mesh.faces.astype(jnp.int32)
    # Represent each triangle by the mean of its three vertex velocities and
    # its center velocity to reduce single-point sampling bias.
    averaged_face_velocities = (jnp.sum(rotated_vertices_vel[face_indices], axis=1) + rotated_centers_vel) / 4.0

    new_axis_radii = calculate_axis_radii(rotated_centers, mesh.rotation_axis)
    return mesh._replace(d_vertices=rotated_vertices,
                         d_centers=rotated_centers,
                         rotation_velocities=averaged_face_velocities,
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
    if _is_phoebe_model(mesh):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    return _evaluate_rotation(mesh, t)


_evaluate_rotation_v = jax.vmap(_evaluate_rotation, in_axes=(None, 0))


def evaluate_rotation_at_times(mesh: MeshModel, times: ArrayLike) -> ArrayLike:
    from spice.utils import log
    with log.timed(
        f"Evaluating rotation at {len(times)} time steps",
        "Rotation evaluated in {elapsed:.1f} s",
    ):
        result_bodies = _evaluate_rotation_v(mesh, times)
        # Convert the  results to lists for each component
        body_list = jax.tree_util.tree_map(lambda x: list(x), result_bodies)
        result = ModelList(jax.tree_util.tree_transpose(
            outer_treedef=jax.tree_util.tree_structure(mesh),
            inner_treedef=jax.tree_util.tree_structure([0 for _ in range(len(times))]),
            pytree_to_transpose=body_list
        ))
    return result


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
    # fourier_series_parameters: (3, n_terms, 2) -> pad to (3, max_fourier_order, 2)
    padded = jnp.pad(fourier_series_parameters, ((0, 0), (0, total_pad_len), (0, 0)))
    return m._replace(
        pulsation_periods=m.pulsation_periods.at[harmonic_ind].set(
            pulsation_periods),
        fourier_series_parameters=m.fourier_series_parameters.at[harmonic_ind].set(padded),
        pulsation_axes=m.pulsation_axes.at[harmonic_ind].set(pulsation_axes),
        pulsation_angles=m.pulsation_angles.at[harmonic_ind].set(
            pulsation_angles),
    )


def add_pulsation(m: MeshModel, m_order: Float, l_degree: Float,
                  period: Float, fourier_series_parameters: Float[Array, "3 n_terms 2"],
                  pulsation_axes: Float[Array, "3"] = None, pulsation_angles: Float = None) -> MeshModel:
    """
    Adds pulsation effects to a mesh model using vector spherical harmonics.

    The displacement is decomposed into three orthogonal components:
        xi = a_r(t) * R_lm + a_s(t) * S_lm + a_t(t) * T_lm

    where R_lm = Y_l^m * r_hat (radial), S_lm = nabla_tangential(Y_l^m) (spheroidal),
    and T_lm = r_hat x nabla_tangential(Y_l^m) (toroidal). Each component has its own
    Fourier series amplitude.

    Args:
        m (MeshModel): The mesh model to add pulsation effects to.
        m_order (Float): The order (m) of the spherical harmonics.
        l_degree (Float): The degree (l) of the spherical harmonics.
        period (Float): Pulsation period in seconds.
        fourier_series_parameters (Float[Array, "3 n_terms 2"]): Fourier series parameters
            for each VSH component. Shape (3, N, 2) where the first axis indexes
            [radial, spheroidal, toroidal] and each (N, 2) block contains
            [amplitude, phase] pairs. For a purely radial mode, set the spheroidal
            and toroidal rows to zeros.
        pulsation_axes (Float[Array, "3"]): Axes of the pulsation. Defaults to the rotation axis of the mesh model.
        pulsation_angles (Float): Angles of the pulsation. Defaults to zero.

    Returns:
        MeshModel: The mesh model with updated pulsation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only within SPICE.
    """
    if _is_phoebe_model(m):
        raise ValueError("PHOEBE models are read-only in SPICE.")
    if _is_arraylike(period):
        period = period[0]

    fourier_series_parameters = jnp.asarray(fourier_series_parameters)
    if fourier_series_parameters.ndim == 2:
        # Legacy (n_terms, 2) -> purely radial: wrap into (3, n_terms, 2)
        fourier_series_parameters = jnp.stack([
            fourier_series_parameters,
            jnp.zeros_like(fourier_series_parameters),
            jnp.zeros_like(fourier_series_parameters),
        ])

    if pulsation_axes is None:
        pulsation_axes = m.rotation_axis
    if pulsation_angles is None:
        pulsation_angles = 0.

    harmonic_ind = m_order + m.max_pulsation_mode * l_degree
    if harmonic_ind >= len(m.pulsation_periods):
        warnings.warn("Pulsation mode is too high for the mesh model - the mesh model has" +
                      "been initialized with a maximum pulsation mode of {}".format(m.max_pulsation_mode) +
                      "This pulsation will have no effect.")

    # n_terms is along axis 1 of the (3, n_terms, 2) array
    total_pad_len = int(m.max_fourier_order - fourier_series_parameters.shape[1])
    return _add_pulsation(m, jnp.array([m_order, l_degree]), period, fourier_series_parameters,
                          total_pad_len,
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

        # fourier_params: (3, n_terms, 2) -> pad to (3, max_fourier_order, 2)
        padded_fourier_params = jnp.pad(fourier_params, ((0, 0), (0, total_pad_len), (0, 0)))

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


def add_pulsations(m: MeshModel, m_orders: Float[Array, "n_pulsations"], l_degrees: Float[Array, "n_pulsations"],
                   periods: Float[Array, "n_pulsations"],
                   fourier_series_parameters: Float[Array, "n_pulsations 3 n_terms 2"],
                   pulsation_axes: Float[Array, "n_pulsations 3"] = None,
                   pulsation_angles: Float[Array, "n_pulsations"] = None) -> MeshModel:
    """
    Adds multiple pulsation effects to a mesh model using vector spherical harmonics.

    The displacement for each mode is decomposed into three orthogonal components:
        xi = a_r(t) * R_lm + a_s(t) * S_lm + a_t(t) * T_lm

    Args:
        m (MeshModel): The mesh model to add pulsation effects to.
        m_orders (Float[Array, "n_pulsations"]): Array of orders (m) of the spherical harmonics.
        l_degrees (Float[Array, "n_pulsations"]): Array of degrees (l) of the spherical harmonics.
        periods (Float[Array, "n_pulsations"]): Array of pulsation periods in seconds.
        fourier_series_parameters (Float[Array, "n_pulsations 3 n_terms 2"]): Fourier parameters
            for each pulsation and VSH component. Shape (K, 3, N, 2) where K is the number of
            pulsations, 3 indexes [radial, spheroidal, toroidal], N is the number of Fourier
            terms, and each innermost pair is [amplitude, phase].
        pulsation_axes (Float[Array, "n_pulsations 3"]): Array of pulsation axes. Defaults to the rotation axis.
        pulsation_angles (Float[Array, "n_pulsations"]): Array of pulsation angles. Defaults to zero.

    Returns:
        MeshModel: The mesh model with updated pulsation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only within SPICE.
        ValueError: If the input arrays have inconsistent lengths.
    """
    if _is_phoebe_model(m):
        raise ValueError("PHOEBE models are read-only in SPICE.")

    if not (len(m_orders) == len(l_degrees) == len(periods) == len(fourier_series_parameters)):
        raise ValueError("Input arrays must have consistent lengths.")

    fourier_series_parameters = jnp.asarray(fourier_series_parameters)
    if fourier_series_parameters.ndim == 3:
        # Legacy (n_pulsations, n_terms, 2) -> purely radial
        n_puls = fourier_series_parameters.shape[0]
        n_terms = fourier_series_parameters.shape[1]
        fourier_series_parameters = jnp.concatenate([
            fourier_series_parameters[:, jnp.newaxis, :, :],
            jnp.zeros((n_puls, 2, n_terms, 2)),
        ], axis=1)

    if pulsation_axes is None:
        pulsation_axes = m.rotation_axis.reshape(
            (1, 3)).repeat(len(m_orders), axis=0)
    if pulsation_angles is None:
        pulsation_angles = jnp.zeros_like(m_orders)

    harmonic_indices = m_orders + m.max_pulsation_mode * l_degrees

    # n_terms is along axis 2 of the (n_pulsations, 3, n_terms, 2) array
    total_pad_len = int(m.max_fourier_order -
                        fourier_series_parameters.shape[2])

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
        pulsation_axes=_default_rotation_axis().reshape(
            (1, 3)).repeat(m.pulsation_periods.shape[0], axis=0),
        pulsation_angles=jnp.zeros_like(m.pulsation_periods),
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
    if _is_phoebe_model(m):
        raise ValueError("PHOEBE models are read-only in SPICE.")

    return _reset_pulsations(m)


@jax.jit
def calculate_pulsations(m: MeshModel, harmonic_parameters: ArrayLike,
                         magnitudes: Float[Array, "3"],
                         magnitudes_prim: Float[Array, "3"],
                         radius: float, pulsation_axis: ArrayLike, pulsation_angle: ArrayLike):
    """Calculate pulsation displacement using vector spherical harmonics.

    The displacement is decomposed as:
        xi = a_r * R_lm + a_s * S_lm + a_t * T_lm

    where R_lm = Y_l^m * r_hat, S_lm = nabla_tangential(Y_l^m),
    T_lm = r_hat x nabla_tangential(Y_l^m).

    Args:
        m: Mesh model.
        harmonic_parameters: [m_order, l_degree].
        magnitudes: (3,) vector of [radial, spheroidal, toroidal] Fourier amplitudes at time t.
        magnitudes_prim: (3,) vector of time derivatives of the amplitudes.
        radius: Stellar radius.
        pulsation_axis: Tilt axis.
        pulsation_angle: Tilt angle in degrees.
    """
    scalar_pulsation_angle = jnp.where(
        jnp.size(pulsation_angle) == 1,
        pulsation_angle.reshape(()),
        pulsation_angle
    )

    m_ord = harmonic_parameters[0]
    l_deg = harmonic_parameters[1]

    # --- Radial basis: R_lm = Y_l^m * r_hat ---
    vertex_harmonic = spherical_harmonic_with_tilt(
        m_ord, l_deg, m.d_vertices, pulsation_axis, scalar_pulsation_angle)[:, jnp.newaxis]
    center_harmonic = spherical_harmonic_with_tilt(
        m_ord, l_deg, m.d_centers, pulsation_axis, scalar_pulsation_angle)[:, jnp.newaxis]

    # Normalize harmonics to (-1, 1) preserving sign
    vertex_harmonic = vertex_harmonic / (jnp.max(jnp.abs(vertex_harmonic)) + 1e-8)
    center_harmonic = center_harmonic / (jnp.max(jnp.abs(center_harmonic)) + 1e-8)

    vertex_r_hat = m.d_vertices / jnp.linalg.norm(m.d_vertices, axis=1, keepdims=True)
    center_r_hat = m.d_centers / jnp.linalg.norm(m.d_centers, axis=1, keepdims=True)

    vertex_R = vertex_harmonic * vertex_r_hat
    center_R = center_harmonic * center_r_hat

    # --- Spheroidal basis: S_lm = nabla_tangential(Y_l^m) ---
    vertex_S = spherical_harmonic_gradient_with_tilt(
        m_ord, l_deg, m.d_vertices, pulsation_axis, scalar_pulsation_angle)
    center_S = spherical_harmonic_gradient_with_tilt(
        m_ord, l_deg, m.d_centers, pulsation_axis, scalar_pulsation_angle)

    vertex_S = vertex_S / (jnp.max(jnp.linalg.norm(vertex_S, axis=1)) + 1e-8)
    center_S = center_S / (jnp.max(jnp.linalg.norm(center_S, axis=1)) + 1e-8)

    # --- Toroidal basis: T_lm = r_hat x nabla_tangential(Y_l^m) ---
    vertex_T = spherical_harmonic_toroidal_with_tilt(
        m_ord, l_deg, m.d_vertices, pulsation_axis, scalar_pulsation_angle)
    center_T = spherical_harmonic_toroidal_with_tilt(
        m_ord, l_deg, m.d_centers, pulsation_axis, scalar_pulsation_angle)

    vertex_T = vertex_T / (jnp.max(jnp.linalg.norm(vertex_T, axis=1)) + 1e-8)
    center_T = center_T / (jnp.max(jnp.linalg.norm(center_T, axis=1)) + 1e-8)

    # --- Total displacement: sum of three VSH components ---
    a_r, a_s, a_t = magnitudes[0], magnitudes[1], magnitudes[2]
    new_vertices = radius * (a_r * vertex_R + a_s * vertex_S + a_t * vertex_T)

    new_areas, new_centers = jax.jit(jax.vmap(face_center, in_axes=(None, 0)))(
        new_vertices + m.d_vertices, m.faces.astype(jnp.int32))

    # --- Pulsation velocity: same decomposition with time derivatives ---
    da_r, da_s, da_t = magnitudes_prim[0], magnitudes_prim[1], magnitudes_prim[2]
    total_pulsation_velocity = radius * (da_r * center_R + da_s * center_S + da_t * center_T)

    return (jnp.nan_to_num(new_vertices),
            jnp.nan_to_num(new_areas - m.base_areas),
            jnp.nan_to_num(new_centers - m.d_centers),
            jnp.nan_to_num(total_pulsation_velocity))


def calculate_numerical_pulsation_velocity(m: MeshModel, t: float, dt: float = 1e-6):
    """
    Calculates pulsation velocity using numerical differentiation.

    Returns magnitudes and velocity magnitudes with shape (n_modes, 3) for the
    three VSH components [radial, spheroidal, toroidal].

    Args:
        m (MeshModel): The mesh model to calculate pulsation velocities for
        t (float): The current time in days
        dt (float, optional): The time step for numerical differentiation in days. Defaults to 1e-6.

    Returns:
        tuple: (pulsation_magnitudes, pulsation_velocity_magnitudes) each of shape (n_modes, 3)
    """
    t = jnp.nan_to_num(t)
    fourier_params = jnp.nan_to_num(m.fourier_series_parameters)
    periods = jnp.nan_to_num(m.pulsation_periods)

    # fourier_params: (n_modes, 3, n_fourier, 2)
    # Reshape to evaluate all 3*n_modes Fourier series at once
    n_modes = fourier_params.shape[0]
    flat_params = fourier_params.reshape(n_modes * 3, fourier_params.shape[2], 2)
    flat_periods = jnp.repeat(periods, 3)

    mags_t = evaluate_many_fouriers_for_value(
        flat_periods, flat_params[:, :, 0], flat_params[:, :, 1], t)
    mags_t_dt = evaluate_many_fouriers_for_value(
        flat_periods, flat_params[:, :, 0], flat_params[:, :, 1], t + dt)

    vel_mags = (mags_t_dt - mags_t) / dt

    return mags_t.reshape(n_modes, 3), vel_mags.reshape(n_modes, 3)


def evaluate_pulsations(m: MeshModel, t: float, use_numerical_derivative: bool = False, dt: float = 1e-6):
    """
    Evaluates and updates the mesh model with pulsation effects at a specific time.

    Uses vector spherical harmonics with three independent amplitude components
    (radial, spheroidal, toroidal) per mode, each driven by its own Fourier series.

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
    if _is_phoebe_model(m):
        raise ValueError("PHOEBE models are read-only in SPICE.")

    from spice.utils import log
    with log.timed(
        f"Evaluating pulsations at t={t}",
        "Pulsations evaluated in {elapsed:.1f} s",
    ):
        if use_numerical_derivative:
            # Numerical derivative: returns (n_modes, 3) directly
            pulsation_magnitudes, pulsation_velocity_magnitudes = calculate_numerical_pulsation_velocity(m, t, dt)
        else:
            # Analytical Fourier evaluation for all 3 VSH components per mode
            t = jnp.nan_to_num(t)
            fourier_params = jnp.nan_to_num(m.fourier_series_parameters)  # (n_modes, 3, n_fourier, 2)
            periods = jnp.nan_to_num(m.pulsation_periods)  # (n_modes,)

            n_modes = fourier_params.shape[0]
            # Flatten (n_modes, 3) -> (n_modes*3) to use the vmapped evaluator
            flat_params = fourier_params.reshape(n_modes * 3, fourier_params.shape[2], 2)
            flat_periods = jnp.repeat(periods, 3)

            flat_magnitudes = evaluate_many_fouriers_for_value(
                flat_periods, flat_params[:, :, 0], flat_params[:, :, 1], t)
            flat_velocity_magnitudes = evaluate_many_fouriers_prim_for_value(
                flat_periods, flat_params[:, :, 0], flat_params[:, :, 1], t)

            # Reshape back to (n_modes, 3)
            pulsation_magnitudes = flat_magnitudes.reshape(n_modes, 3)
            pulsation_velocity_magnitudes = flat_velocity_magnitudes.reshape(n_modes, 3)

        harmonic_params = create_harmonics_params(m.max_pulsation_mode)

        # vmap over modes; magnitudes and magnitudes_prim are now (n_modes, 3)
        vert_offsets, area_offsets, center_offsets, pulsation_velocities = jax.vmap(
            calculate_pulsations,
            in_axes=(None, 0, 0, 0, None, 0, 0)
        )(m, harmonic_params,
          pulsation_magnitudes,
          pulsation_velocity_magnitudes,
          m.radius,
          m.pulsation_axes,
          m.pulsation_angles)

        result = m._replace(
            vertices_pulsation_offsets=jnp.nan_to_num(jnp.sum(vert_offsets, axis=0)),
            center_pulsation_offsets=jnp.nan_to_num(jnp.sum(center_offsets, axis=0)),
            area_pulsation_offsets=jnp.nan_to_num(jnp.sum(area_offsets, axis=0)),
            pulsation_velocities=jnp.nan_to_num(
                jnp.sum(pulsation_velocities, axis=0) * 8.052083333333332  # solRad/day to km/s
            )
        )
    return result
