from functools import partial
from importlib import import_module
from typing import List, Optional, Union
import warnings

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from .mesh_generation import face_center
from .mesh_model import MeshModel, _default_rotation_axis
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
                    spherical_harmonic_toroidal_with_tilt,
                    vector_spherical_harmonic_bases_complex,
                    MAX_SPHERICAL_HARMONIC_DEGREE)

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

    if isinstance(parameter, str):
        if parameter_names is None:
            raise ValueError(
                "A list of parameters used for the model while constructing the mesh must be provided to update parameters by name.")
        if parameter not in parameter_names:
            raise ValueError(
                f"Parameter {parameter} not found in mesh model. Model contains parameters: {parameter_names}")
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
                    f"Parameter {param} not found in mesh model. Model contains parameters: {parameter_names}")
            parameter_indices.append(parameter_names.index(param))
    else:
        parameter_indices = parameters

    return _update_parameters(mesh, jnp.array(parameter_indices), jnp.array(parameter_values))


@jax.jit
def _axis_perpendicular_to_los(los_vector: Float[Array, "3"]) -> Float[Array, "3"]:
    """A unit rotation axis guaranteed perpendicular to ``los_vector``.

    Takes the module default when that is already perpendicular (the usual
    case), and otherwise rejects the LOS component from it -- falling back to a
    second seed axis if the default happens to be parallel to the LOS.

    A rotation axis parallel to the line of sight projects to *zero* radial
    velocity, so the star is seen pole-on and rotational broadening vanishes
    silently. That is easy to hit whenever a caller overrides one of the two
    defaults but not the other: `tz_fornacis_spectra.py` hardcodes
    LOS = [0, 0, -1] while inheriting the default axis, which on a checkout
    whose default axis is [0, 0, 1] gave exactly 0 km/s of broadening for a
    38 km/s star.
    """
    los = los_vector / jnp.linalg.norm(los_vector)
    primary = _default_rotation_axis()
    # If the default is (nearly) parallel to the LOS, seed from another axis.
    seed = jnp.where(
        jnp.abs(jnp.dot(primary, los)) > 0.99,
        jnp.array([1.0, 0.0, 0.0], dtype=primary.dtype),
        primary,
    )
    seed = jnp.where(
        jnp.abs(jnp.dot(seed, los)) > 0.99,
        jnp.array([0.0, 1.0, 0.0], dtype=primary.dtype),
        seed,
    )
    perp = seed - jnp.dot(seed, los) * los          # reject the LOS component
    return perp / jnp.linalg.norm(perp)


def _add_rotation(mesh: MeshModel,
                  rotation_velocity: float,
                  rotation_axis: Float[Array, "3"] = None) -> MeshModel:
    if rotation_axis is None:
        rotation_axis = _axis_perpendicular_to_los(mesh.los_vector)
    rot_matrix = rotation_matrix(rotation_axis)
    rot_matrix_grad = rotation_matrix_prim(rotation_axis)

    mesh = mesh._replace(rotation_axis=rotation_axis,
                         rotation_matrix=rot_matrix,
                         rotation_matrix_prim=rot_matrix_grad,
                         rotation_velocity=rotation_velocity)

    # Populate the per-element velocity field straight away. It used to be filled
    # only by evaluate_rotation(), so a mesh with add_rotation() applied looked
    # configured but produced NO rotational broadening. binary.py never calls
    # evaluate_rotation, so every binary synthesis silently lost v sin i --
    # measured on a 38 km/s star, line FWHM stayed at 16.19 km/s (the
    # non-rotating value) instead of broadening to 64.20 km/s.
    # Evaluating at t = 0 is a no-op for the geometry: theta = v*0/R = 0 and
    # evaluate_rotation_matrix(R, 0) is the identity, so this only fills
    # rotation_velocities and callers may still call evaluate_rotation(mesh, t)
    # afterwards at any time.
    return _evaluate_rotation(mesh, 0.0)


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
        rotation_velocity (float): Equatorial rotation speed [km/s].
        rotation_axis (Float[Array, "3"]): The rotation axis. If omitted, choose an axis
            perpendicular to the mesh's current line of sight (equator-on). Changing
            the view afterwards preserves this axis.

    Note:
        The surface moves with ``v = (v_eq / R) * (r x a)`` for unit axis ``a``,
        i.e. *clockwise* seen from the tip of ``a``: the angular momentum points
        along ``-rotation_axis``. Flip the axis to spin the other way (e.g. to
        align the spin with an orbit's angular momentum).

    Returns:
        MeshModel: The mesh model with updated rotation parameters.

    Raises:
        ValueError: If the mesh model is an instance of PhoebeModel, indicating it is read-only.
    """
    if _is_phoebe_model(mesh):
        raise ValueError(
            "PHOEBE models are read-only in SPICE - the rotation is already evaluated in the PHOEBE model.")
    if rotation_axis is None:
        # Derive the default from the mesh's own line of sight rather than a
        # module constant, so it can never come out parallel to it. A parallel
        # axis projects to zero radial velocity: the star is seen pole-on and
        # rotational broadening vanishes silently. That is exactly what happened
        # to tz_fornacis_spectra.py, which hardcodes LOS = [0, 0, -1] while
        # inheriting a default axis of [0, 0, 1] on the geometry-provenance
        # branch -- 0 km/s of broadening for a 38 km/s star.
        rotation_axis = _axis_perpendicular_to_los(mesh.los_vector)
    return _add_rotation(mesh, rotation_velocity, rotation_axis)


def _refresh_computed_log_g(mesh: MeshModel) -> MeshModel:
    """Keep automatically managed atmosphere gravity aligned with the geometry."""
    if mesh.log_g_index is None:
        return mesh
    log_g = jax.lax.cond(
        mesh.override_log_g,
        lambda _: mesh.parameters[:, mesh.log_g_index],
        lambda _: mesh.log_gs,
        operand=None,
    )
    return mesh._replace(parameters=mesh.parameters.at[:, mesh.log_g_index].set(log_g))


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

    def project_to_surface(points):
        squared_norms = jnp.sum(points**2, axis=1, keepdims=True)
        # Icosphere arrays include zero padding. Avoid evaluating sqrt(0) or
        # dividing by zero there, including in the backward pass.
        norms = jnp.sqrt(jnp.where(squared_norms > 0.0, squared_norms, 1.0))
        return points * (mesh.radius / norms)

    surface_vertices = project_to_surface(rotated_vertices)
    surface_centers = project_to_surface(rotated_centers)

    rotated_vertices_vel = (rotation_velocity_km_s / mesh.radius) * jnp.cross(surface_vertices, axis_hat)
    rotated_centers_vel = (rotation_velocity_km_s / mesh.radius) * jnp.cross(surface_centers, axis_hat)

    face_indices = mesh.faces.astype(jnp.int32)
    # Represent each triangle by the mean of its three vertex velocities and
    # its center velocity to reduce single-point sampling bias.
    averaged_face_velocities = (jnp.sum(rotated_vertices_vel[face_indices], axis=1) + rotated_centers_vel) / 4.0

    new_axis_radii = calculate_axis_radii(rotated_centers, mesh.rotation_axis)
    # Pulsation displacements and velocities are vectors attached to the
    # surface: turn them with it. Leaving them in the old directions made
    # evaluate_rotation(evaluate_pulsations(m, t), t) push vertices sideways
    # (a 90 deg turn left the pulsation RV at 0.01 instead of 0.72 km/s).
    rotated = mesh._replace(d_vertices=rotated_vertices,
                            d_centers=rotated_centers,
                            vertices_pulsation_offsets=jnp.matmul(
                                mesh.vertices_pulsation_offsets, t_rotation_matrix),
                            center_pulsation_offsets=jnp.matmul(
                                mesh.center_pulsation_offsets, t_rotation_matrix),
                            pulsation_velocities=jnp.matmul(
                                mesh.pulsation_velocities, t_rotation_matrix),
                            rotation_velocities=averaged_face_velocities,
                            axis_radii=new_axis_radii)
    return _refresh_computed_log_g(rotated)


def evaluate_rotation(mesh: MeshModel, t: float) -> MeshModel:
    """
    Evaluate the rotation of a mesh model over a specified time.

    This function determines the rotation of a mesh model at a given time `t`. It checks if the mesh model
    is an instance of PhoebeModel and raises an error if so, indicating that PHOEBE models are considered
    read-only within the SPICE framework and their rotation cannot be evaluated or modified. For models
    that are not read-only, it delegates the rotation evaluation to the `_evaluate_rotation` function.

    Args:
        mesh (MeshModel): The mesh model to evaluate rotation for.
        t (float): Time to rotate by [seconds]: the surface turns by
            ``rotation_velocity * t / R`` from its *current* orientation.

    Note:
        This is incremental: ``evaluate_rotation(evaluate_rotation(m, t1), t2)`` is
        the state at ``t1 + t2``. Always start from the un-rotated mesh to get the
        state at an absolute time. Pulsation displacements and velocities turn with
        the surface.

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
    """Vectorized :func:`evaluate_rotation` over an array of times (in seconds)."""
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


def _validate_modes(m_orders, l_degrees) -> None:
    """Reject (m, l) pairs the real-harmonic basis cannot represent.

    Negative m used to land in another mode's slot (``m + max_mode * l``) and
    silently do nothing; l above MAX_SPHERICAL_HARMONIC_DEGREE is evaluated
    incorrectly by the fixed-``n_max`` harmonic routine.
    """
    try:
        ms = np.atleast_1d(np.asarray(m_orders, dtype=float))
        ls = np.atleast_1d(np.asarray(l_degrees, dtype=float))
    except Exception:  # traced values: cannot check here
        return
    if np.any(ms < 0):
        raise ValueError(
            "Negative azimuthal order m is not supported: modes use real spherical "
            "harmonics (standing waves), so m and -m give the same pattern up to an "
            "azimuthal rotation. Use m >= 0 (and pulsation_angles about the rotation "
            "axis to change its azimuth).")
    if np.any(ms > ls):
        raise ValueError("Azimuthal order m must satisfy m <= l.")
    if np.any(ls > MAX_SPHERICAL_HARMONIC_DEGREE):
        raise ValueError(
            f"Degree l > {MAX_SPHERICAL_HARMONIC_DEGREE} is not supported by the "
            "spherical-harmonic evaluator.")


_TRAVELLING_SENSE = {None: 0., "standing": 0., "prograde": -1., "retrograde": 1.}


def _travelling_sense(travelling, m_order) -> float:
    """Map 'prograde'/'retrograde'/None to the azimuthal sense used internally.

    SPICE stars spin clockwise about ``rotation_axis`` (angular momentum along
    -axis), i.e. toward decreasing longitude, so a prograde pattern moves toward
    decreasing longitude (-1) and a retrograde one toward increasing (+1).
    """
    if travelling not in _TRAVELLING_SENSE:
        raise ValueError(f"travelling must be None, 'standing', 'prograde' or 'retrograde'; got {travelling!r}")
    sense = _TRAVELLING_SENSE[travelling]
    if sense != 0. and float(np.asarray(m_order)) == 0.:
        raise ValueError("An m = 0 mode is axisymmetric and cannot travel in azimuth.")
    return sense


def _set_travelling(m: MeshModel, harmonic_indices, senses) -> MeshModel:
    current = _pulsation_travelling(m)
    return m._replace(pulsation_travelling=current.at[jnp.asarray(harmonic_indices)].set(
        jnp.asarray(senses, dtype=current.dtype)))


def add_pulsation(m: MeshModel, m_order: Float, l_degree: Float,
                  period: Float, fourier_series_parameters: Float[Array, "3 n_terms 2"],
                  pulsation_axes: Float[Array, "3"] = None, pulsation_angles: Float = None,
                  travelling: Optional[str] = None) -> MeshModel:
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
        period (Float): Pulsation period in days (the same unit as the ``t``
            passed to ``evaluate_pulsations``; the pulsation-velocity
            conversion to km/s assumes solRad/day).
        fourier_series_parameters (Float[Array, "3 n_terms 2"]): Fourier series parameters
            for each VSH component. Shape (3, N, 2) where the first axis indexes
            [radial, spheroidal, toroidal] and each (N, 2) block contains
            [amplitude, phase] pairs. For a purely radial mode, set the spheroidal
            and toroidal rows to zeros.
        pulsation_axes (Float[Array, "3"]): Axis to tilt the mode about. The
            mode's symmetry axis is the mesh's ``rotation_axis`` (read when the
            pulsations are evaluated), tilted by ``pulsation_angles`` degrees about
            this axis. Defaults to the rotation axis, i.e. no tilt.
        pulsation_angles (Float): Tilt angle in degrees. Defaults to zero.
        travelling (str, optional): ``None`` (default) for a standing mode,
            ``"prograde"`` or ``"retrograde"`` for a mode travelling in azimuth with
            or against the star's rotation. A travelling mode is
            ``Re[Y_l^m e^{-i s w t}]`` per Fourier harmonic: its pattern keeps a
            constant shape and amplitude (the peak displacement) and moves at
            ``w / m`` in longitude about the (tilted) mode axis. Requires ``m > 0``.

    Note:
        ``m`` must satisfy ``0 <= m <= l <= 10``. Standing m != 0 modes have fixed
        nodal lines (real harmonics); use ``travelling`` for moving ones.

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

    _validate_modes(m_order, l_degree)
    harmonic_ind = m_order + m.max_pulsation_mode * l_degree
    if harmonic_ind >= len(m.pulsation_periods):
        warnings.warn("Pulsation mode is too high for the mesh model - the mesh model has" +
                      "been initialized with a maximum pulsation mode of {}".format(m.max_pulsation_mode) +
                      "This pulsation will have no effect.")

    sense = _travelling_sense(travelling, m_order)

    # n_terms is along axis 1 of the (3, n_terms, 2) array
    total_pad_len = int(m.max_fourier_order - fourier_series_parameters.shape[1])
    m = _add_pulsation(m, jnp.array([m_order, l_degree]), period, fourier_series_parameters,
                       total_pad_len,
                       pulsation_axes, pulsation_angles)
    return _set_travelling(m, harmonic_ind, sense)


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
                   pulsation_angles: Float[Array, "n_pulsations"] = None,
                   travelling=None) -> MeshModel:
    """
    Adds multiple pulsation effects to a mesh model using vector spherical harmonics.

    The displacement for each mode is decomposed into three orthogonal components:
        xi = a_r(t) * R_lm + a_s(t) * S_lm + a_t(t) * T_lm

    Args:
        m (MeshModel): The mesh model to add pulsation effects to.
        m_orders (Float[Array, "n_pulsations"]): Array of orders (m) of the spherical harmonics.
        l_degrees (Float[Array, "n_pulsations"]): Array of degrees (l) of the spherical harmonics.
        periods (Float[Array, "n_pulsations"]): Array of pulsation periods in days
            (the same unit as the ``t`` passed to ``evaluate_pulsations``; the
            pulsation-velocity conversion to km/s assumes solRad/day).
        fourier_series_parameters (Float[Array, "n_pulsations 3 n_terms 2"]): Fourier parameters
            for each pulsation and VSH component. Shape (K, 3, N, 2) where K is the number of
            pulsations, 3 indexes [radial, spheroidal, toroidal], N is the number of Fourier
            terms, and each innermost pair is [amplitude, phase].
        pulsation_axes (Float[Array, "n_pulsations 3"]): Tilt axes; each mode's
            symmetry axis is the mesh's rotation axis tilted about its tilt axis by
            its angle (see :func:`add_pulsation`). Defaults to the rotation axis (no tilt).
        pulsation_angles (Float[Array, "n_pulsations"]): Tilt angles in degrees. Defaults to zero.
        travelling: ``None`` (all standing), one of ``"prograde"`` / ``"retrograde"``
            for all modes, or a sequence with one entry (or ``None``) per mode. See
            :func:`add_pulsation`.

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

    _validate_modes(m_orders, l_degrees)
    harmonic_indices = m_orders + m.max_pulsation_mode * l_degrees

    # n_terms is along axis 2 of the (n_pulsations, 3, n_terms, 2) array
    total_pad_len = int(m.max_fourier_order -
                        fourier_series_parameters.shape[2])

    if travelling is None or isinstance(travelling, str):
        travelling = [travelling] * len(m_orders)
    if len(travelling) != len(m_orders):
        raise ValueError("travelling must have one entry per pulsation.")
    senses = [_travelling_sense(tr, mo) for tr, mo in zip(travelling, np.asarray(m_orders))]

    m = _add_pulsations(m, periods,
                        fourier_series_parameters, harmonic_indices, total_pad_len,
                        pulsation_axes, pulsation_angles)
    return _set_travelling(m, harmonic_indices.astype(int), senses)


@jax.jit
def _reset_pulsations(m: MeshModel) -> MeshModel:
    reset = m._replace(
        vertices_pulsation_offsets=jnp.zeros_like(
            m.vertices_pulsation_offsets),
        center_pulsation_offsets=jnp.zeros_like(m.center_pulsation_offsets),
        area_pulsation_offsets=jnp.zeros_like(m.area_pulsation_offsets),
        pulsation_velocities=jnp.zeros_like(m.pulsation_velocities),
        pulsation_periods=jnp.nan * jnp.ones_like(m.pulsation_periods),
        fourier_series_parameters=jnp.nan *
        jnp.ones_like(m.fourier_series_parameters),
        pulsation_axes=_default_rotation_axis().reshape(
            (1, 3)).repeat(m.pulsation_periods.shape[0], axis=0),
        pulsation_angles=jnp.zeros_like(m.pulsation_periods),
        pulsation_travelling=jnp.zeros_like(m.pulsation_periods),
    )
    return _refresh_computed_log_g(reset)


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
def _pulsation_displacements(m: MeshModel, harmonic_parameters: ArrayLike,
                         magnitudes: Float[Array, "3"],
                         magnitudes_prim: Float[Array, "3"],
                         radius: float, pulsation_axis: ArrayLike, pulsation_angle: ArrayLike,
                         quadrature: Float[Array, "3"] = None,
                         quadrature_prim: Float[Array, "3"] = None,
                         travelling: ArrayLike = 0.):
    """Calculate pulsation displacement using vector spherical harmonics.

    The displacement is decomposed as:
        xi = a_r * R_lm + a_s * S_lm + a_t * T_lm

    where R_lm = Y_l^m * r_hat, S_lm = nabla_tangential(Y_l^m),
    T_lm = r_hat x nabla_tangential(Y_l^m).

    A standing mode (``travelling = 0``) uses the real part of each basis. A
    travelling mode (``travelling = +-1``) is ``Re[B e^{-i s w t}]`` per Fourier
    harmonic, i.e. ``a(t) Re(B) + s b(t) Im(B)`` with ``b`` the quadrature
    series (every phase shifted by pi/2): the pattern ``cos(m phi - s w t)``
    keeps its shape and moves in azimuth (``s = +1`` toward increasing phi about
    the rotation axis). Travelling bases are normalised by the peak of
    ``|B|`` so the amplitude is the constant peak displacement.

    Args:
        m: Mesh model.
        harmonic_parameters: [m_order, l_degree].
        magnitudes: (3,) vector of [radial, spheroidal, toroidal] Fourier amplitudes at time t.
        magnitudes_prim: (3,) vector of time derivatives of the amplitudes.
        radius: Stellar radius.
        pulsation_axis: Tilt axis.
        pulsation_angle: Tilt angle in degrees.
        quadrature, quadrature_prim: (3,) quadrature amplitudes and their time
            derivatives (only used for travelling modes).
        travelling: 0 (standing), +1 or -1 (sense of azimuthal motion).

    The untilted mode is symmetric about ``m.rotation_axis``; it used to be
    symmetric about +z regardless of the rotation axis, so on a default
    (+y axis) star non-radial modes were perpendicular to the spin.
    """
    scalar_pulsation_angle = jnp.where(
        jnp.size(pulsation_angle) == 1,
        pulsation_angle.reshape(()),
        pulsation_angle
    )
    if quadrature is None:
        quadrature = jnp.zeros_like(magnitudes)
    if quadrature_prim is None:
        quadrature_prim = jnp.zeros_like(magnitudes_prim)
    sense = jnp.asarray(travelling).reshape(())
    is_travelling = sense != 0.

    m_ord = harmonic_parameters[0]
    l_deg = harmonic_parameters[1]

    # IcosphereModel pads d_vertices with interior filler vertices at the origin
    # that no face references. The spherical-harmonic basis (especially its
    # gradient) is singular at r=0, so those padded values dominate a plain
    # `jnp.max(...)` and squash the surface amplitudes. Mask them out before
    # normalizing so `a_r`, `a_s`, `a_t` map to the peak surface displacement.
    surface_vertex = jnp.linalg.norm(m.d_vertices, axis=1) > 0.5 * radius

    vY, vS, vT = vector_spherical_harmonic_bases_complex(
        m_ord, l_deg, m.d_vertices, pulsation_axis, scalar_pulsation_angle,
        frame_axis=m.rotation_axis)
    cY, cS, cT = vector_spherical_harmonic_bases_complex(
        m_ord, l_deg, m.d_centers, pulsation_axis, scalar_pulsation_angle,
        frame_axis=m.rotation_axis)

    def peak(values, mask=None):
        values = jnp.where(mask, values, 0.0) if mask is not None else values
        return jnp.max(values) + 1e-8

    def scalar_norm(z, mask=None):
        return jnp.where(is_travelling, peak(jnp.abs(z), mask), peak(jnp.abs(z.real), mask))

    def vector_norm(z, mask=None):
        full = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=1))
        real = jnp.linalg.norm(z.real, axis=1)
        return jnp.where(is_travelling, peak(full, mask), peak(real, mask))

    vertex_norms = jnp.linalg.norm(m.d_vertices, axis=1, keepdims=True)
    vertex_r_hat = m.d_vertices / jnp.where(vertex_norms > 0., vertex_norms, 1.)
    center_r_hat = m.d_centers / jnp.linalg.norm(m.d_centers, axis=1, keepdims=True)

    # Bases normalised to a peak of 1 (radial: Y r_hat; spheroidal; toroidal)
    vR = (vY / scalar_norm(vY, surface_vertex))[:, None] * vertex_r_hat
    cR = (cY / scalar_norm(cY))[:, None] * center_r_hat
    vS = vS / vector_norm(vS, surface_vertex)
    cS = cS / vector_norm(cS)
    vT = vT / vector_norm(vT, surface_vertex)
    cT = cT / vector_norm(cT)

    def combine(R, S, T, a, b):
        standing = a[0] * R.real + a[1] * S.real + a[2] * T.real
        moving = b[0] * R.imag + b[1] * S.imag + b[2] * T.imag
        return standing + sense * moving

    new_vertices = radius * combine(vR, vS, vT, magnitudes, quadrature)
    total_pulsation_velocity = radius * combine(cR, cS, cT, magnitudes_prim, quadrature_prim)

    return jnp.nan_to_num(new_vertices), jnp.nan_to_num(total_pulsation_velocity)


@jax.jit
def _pulsation_geometry_offsets(m: MeshModel, vertex_offsets: ArrayLike):
    """Measure the combined deformation relative to the reference triangles."""
    faces = m.faces.astype(jnp.int32)
    areas_for_vertices = jax.vmap(face_center, in_axes=(None, 0))
    reference_areas, _ = areas_for_vertices(m.d_vertices, faces)
    displaced_areas, _ = areas_for_vertices(m.d_vertices + vertex_offsets, faces)
    # Stored areas exclude the reference radius squared. Compare like units,
    # and subtract the same reference geometry so zero displacement is a no-op.
    area_offsets = (displaced_areas - reference_areas) / m.radius**2
    # Reference centers are projected onto the stellar radius at construction.
    # Move them by the centroid displacement, without replacing that reference.
    center_offsets = jnp.mean(vertex_offsets[faces], axis=1)
    return area_offsets, center_offsets


@jax.jit
def calculate_pulsations(m: MeshModel, harmonic_parameters: ArrayLike,
                         magnitudes: Float[Array, "3"],
                         magnitudes_prim: Float[Array, "3"],
                         radius: float, pulsation_axis: ArrayLike, pulsation_angle: ArrayLike):
    """Return vertex, normalized-area, center, and velocity offsets for one mode.

    Areas and centers are measured relative to the undeformed mesh. For multiple
    modes, use :func:`evaluate_pulsations` so geometry is computed after summing
    vertex displacements; triangle areas are not additive across modes.
    """
    vertices, velocities = _pulsation_displacements(
        m, harmonic_parameters, magnitudes, magnitudes_prim,
        radius, pulsation_axis, pulsation_angle)
    areas, centers = _pulsation_geometry_offsets(m, vertices)
    return vertices, areas, centers, velocities


def _pulsation_travelling(m: MeshModel) -> ArrayLike:
    """Per-mode travelling sense (0 standing, +-1 travelling); None means all standing."""
    sense = getattr(m, "pulsation_travelling", None)
    if sense is None:
        return jnp.zeros(m.pulsation_periods.shape[0], dtype=m.pulsation_periods.dtype)
    return jnp.asarray(sense, dtype=m.pulsation_periods.dtype)


def _quadrature_parameters(fourier_params: ArrayLike) -> ArrayLike:
    """Same series with every phase shifted by pi/2: A cos(k w t - phi - pi/2) = A sin(k w t - phi)."""
    return fourier_params.at[..., 1].add(jnp.pi / 2)


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
        quadrature_model = m._replace(
            fourier_series_parameters=_quadrature_parameters(m.fourier_series_parameters))
        if use_numerical_derivative:
            # Numerical derivative: returns (n_modes, 3) directly
            pulsation_magnitudes, pulsation_velocity_magnitudes = calculate_numerical_pulsation_velocity(m, t, dt)
            quadrature_magnitudes, quadrature_velocity_magnitudes = calculate_numerical_pulsation_velocity(
                quadrature_model, t, dt)
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

            # Quadrature series for travelling modes (unused by standing ones)
            flat_quad = _quadrature_parameters(flat_params)
            quadrature_magnitudes = evaluate_many_fouriers_for_value(
                flat_periods, flat_quad[:, :, 0], flat_quad[:, :, 1], t).reshape(n_modes, 3)
            quadrature_velocity_magnitudes = evaluate_many_fouriers_prim_for_value(
                flat_periods, flat_quad[:, :, 0], flat_quad[:, :, 1], t).reshape(n_modes, 3)

        harmonic_params = m.spherical_harmonics_parameters

        # vmap over modes; magnitudes and magnitudes_prim are now (n_modes, 3)
        vert_offsets, pulsation_velocities = jax.vmap(
            _pulsation_displacements,
            in_axes=(None, 0, 0, 0, None, 0, 0, 0, 0, 0)
        )(m, harmonic_params,
          pulsation_magnitudes,
          pulsation_velocity_magnitudes,
          m.radius,
          m.pulsation_axes,
          m.pulsation_angles,
          quadrature_magnitudes,
          quadrature_velocity_magnitudes,
          _pulsation_travelling(m))

        combined_vertices = jnp.nan_to_num(jnp.sum(vert_offsets, axis=0))
        area_offsets, center_offsets = _pulsation_geometry_offsets(m, combined_vertices)
        result = m._replace(
            vertices_pulsation_offsets=combined_vertices,
            center_pulsation_offsets=center_offsets,
            area_pulsation_offsets=area_offsets,
            pulsation_velocities=jnp.nan_to_num(
                jnp.sum(pulsation_velocities, axis=0) * 8.052083333333332  # solRad/day to km/s
            )
        )
    return _refresh_computed_log_g(result)
