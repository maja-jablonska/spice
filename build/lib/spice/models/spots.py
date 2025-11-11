from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from spice.models.phoebe_model import PhoebeModel
from .mesh_model import MeshModel
from .utils import spherical_harmonic, spherical_harmonic_with_tilt

from jaxtyping import Array, Int, Float


def _is_arraylike(x):
    return hasattr(x, '__array__') or hasattr(x, '__array_interface__')


def __spherical_to_cartesian(theta: float, phi: float, radius: float) -> tuple[Array, Array, Array]:
    x = jnp.sin(theta) * jnp.cos(phi) * radius
    y = jnp.sin(theta) * jnp.sin(phi) * radius
    z = jnp.cos(theta) * radius
    return x, y, z


def __cartesian_distance(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> Array:
    return jnp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


v_cartesian_distance = jax.jit(jax.vmap(__cartesian_distance, in_axes=(0, 0, 0, None, None, None)))


def sigmoid(x: float, smoothness: float) -> Array:
    # Apply the sigmoid function
    return 1 / (1 + jnp.exp(-smoothness * x))


@jax.jit
def generate_spherical_spot(d_centers: Float[Array, "n_mesh_elements 3"],
                            mesh_radius: float,
                            spot_center_theta: float,
                            spot_center_phi: float,
                            spot_radius: float,
                            parameter_diff: float,
                            smoothness: float = 1.0) -> Float[Array, "n_mesh_elements 3"]:
    """
    Generate a differential parameter array based on the location of points on a sphere relative to a specified
    spherical spot.

    This function calculates the angular distance of each point (defined by `thetas` and `phis`) from a specified
    spot center (`spot_center_theta`, `spot_center_phi`). It then determines the differential parameter value for
    each point based on whether it lies within the spherical spot, applying a smoothness factor to interpolate values
    at the spot's edge.

    Args:
        Float[Array, "n_mesh_elements 3"] (Array): Cartesian coordinates of the centers of the mesh elements
        mesh_radius (float): Radius of the mesh
        spot_center_theta (float): Theta value (inclination) of the spot center.
        spot_center_phi (float): Phi value (azimuth) of the spot center.
        spot_radius (float): Angular radius of the spot.
        parameter_diff (float): The differential parameter value to be applied within the spot.
        smoothness (float, optional): Factor controlling the smoothness of the spot's edge. Defaults to 1.0.

    Returns: ArrayLike: An array of differential parameter values for each point, interpolated based on spot
    membership and edge smoothness.
    """
    
    smoothness = jnp.clip(smoothness, 0.0, 1.0)

    spot_center_x, spot_center_y, spot_center_z = __spherical_to_cartesian(spot_center_theta, spot_center_phi,
                                                                           mesh_radius)

    # Calculate distance from the spot center
    distances = v_cartesian_distance(d_centers[:, 0], d_centers[:, 1], d_centers[:, 2],
                                     spot_center_x, spot_center_y, spot_center_z)

    # Calculate the difference from the spot edge
    distance_from_edge = jnp.deg2rad(spot_radius) * mesh_radius - distances

    # Apply smoothness to interpolate values at the spot's edge
    # Use a small fraction of mesh radius for scaling, but ensure it's not too small
    scale_factor = jnp.maximum(0.01 * mesh_radius, 1e-6)
    # Clip distance_from_edge to prevent numerical issues with very large negative values
    normalized_distance = jnp.clip(distance_from_edge / scale_factor, -100.0, 100.0)
    
    # Invert smoothness parameter so 0.0 is sharp (high sigmoid slope) and 1.0 is smooth (low sigmoid slope)
    # Map smoothness from [0.0, 1.0] to [10.0, 0.01] for sigmoid steepness
    adjusted_smoothness = (1.0 - smoothness) + 0.01 * smoothness
    differences = sigmoid(normalized_distance, adjusted_smoothness) * parameter_diff

    return differences


generate_spherical_spots = jax.jit(jax.vmap(generate_spherical_spot, in_axes=(None, None,
                                                                              0, 0, 0, 0, 0)))


@jax.jit
def _add_spherical_harmonic_spot(mesh: MeshModel,
                                 m: int, n: int,
                                 param_delta: float,
                                 param_index: int) -> MeshModel:
    spot_parameters = spherical_harmonic(m, n, mesh.centers) * param_delta
    return mesh._replace(parameters=mesh.parameters.at[:, param_index].set(
        mesh.parameters[:, param_index] + spot_parameters))
    
    
@jax.jit
def _add_spherical_harmonic_spot_with_tilt(mesh: MeshModel,
                                 m: int, n: int,
                                 param_delta: float,
                                 param_index: int,
                                 tilt_axis: ArrayLike = jnp.array([0., 0., 1.]),
                                 tilt_angle: float = 0.) -> MeshModel:
    spot_parameters = spherical_harmonic_with_tilt(m, n, mesh.centers, tilt_axis, tilt_angle) * param_delta
    return mesh._replace(parameters=mesh.parameters.at[:, param_index].set(
        mesh.parameters[:, param_index] + spot_parameters))


def add_spherical_harmonic_spot(mesh: MeshModel,
                                m_order: Union[Int, Float],
                                n_degree: Union[Int, Float],
                                param_delta: Float,
                                param_index: Float,
                                tilt_axis: Float[Array, "3"] = None,
                                tilt_angle: Float = None) -> MeshModel:
    """
    Add a spherical harmonic variation to a parameter of the mesh model, effectively creating a spot-like feature.

    This function applies a spherical harmonic function to modify a specific parameter of the mesh model,
    creating a pattern that can represent various surface features like spots or temperature variations.
    The spherical harmonic function Y_n^m(θ,φ) is used to create the variation pattern on the stellar surface.

    Args:
        mesh (MeshModel): The mesh model to which the spherical harmonic variation will be applied.
            Must not be a PhoebeModel as those are read-only.
        m_order (Union[Int, Float]): The order (m) of the spherical harmonic. Must be less than or equal 
            to n_degree. Determines the number of longitudinal nodes in the pattern.
        n_degree (Union[Int, Float]): The degree (n) of the spherical harmonic. Must be greater than or equal
            to m_order. Determines the total number of nodes in the pattern.
        param_delta (Float): The maximum amplitude of the parameter variation. Represents the difference 
            between the peak of the variation and the background value.
        param_index (Float): The index of the parameter in the mesh model's parameter array that will be 
            modified by this variation.
        tilt_axis (Float[Array, "3"], optional): The axis around which to tilt the spherical harmonic pattern.
            Defaults to None (no tilt).
        tilt_angle (Float, optional): The angle in degrees to tilt the spherical harmonic pattern around
            tilt_axis. Only used if tilt_axis is provided. Defaults to None (no tilt).

    Returns:
        MeshModel: A new mesh model with the specified parameter modified according to the spherical
            harmonic variation.

    Raises:
        ValueError: If mesh is a PhoebeModel, or if m_order > n_degree.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only.")
    if isinstance(m_order, int) and isinstance(n_degree, int) and m_order > n_degree:
        raise ValueError("m must be lesser or equal to n.")
    elif _is_arraylike(m_order) and _is_arraylike(n_degree):
        m_order = int(m_order.item())
        n_degree = int(n_degree.item())
        if jnp.any(jnp.greater(m_order, n_degree)):
            raise ValueError("m must be lesser or equal to n.")
        
    if tilt_axis is not None:
        tilt_angle = 0.0 if tilt_angle is None else tilt_angle
        return _add_spherical_harmonic_spot_with_tilt(mesh, m_order, n_degree, param_delta, param_index, tilt_axis, tilt_angle)
    return _add_spherical_harmonic_spot(mesh, m_order, n_degree, param_delta, param_index)


@jax.jit
def _add_spot(mesh: MeshModel,
              spot_center_theta: float,
              spot_center_phi: float,
              spot_radius: float,
              parameter_diff: float,
              parameter_index: int,
              smoothness: float = 1.0) -> MeshModel:
    spot_parameters = generate_spherical_spot(mesh.d_centers, mesh.radius,
                                              spot_center_theta, spot_center_phi, spot_radius, parameter_diff,
                                              smoothness)
    return mesh._replace(parameters=mesh.parameters.at[:, parameter_index].set(
        mesh.parameters[:, parameter_index] + spot_parameters)
    )


def add_spot(mesh: MeshModel,
             spot_center_theta: float,
             spot_center_phi: float,
             spot_radius: float,
             parameter_delta: float,
             parameter_index: int,
             smoothness: float = 0.0) -> MeshModel:
    """
    Add a spot to a mesh model based on spherical coordinates and smoothness parameters.

    This function applies a modification to the mesh model's parameters to simulate the presence of a spot. The spot
    is defined by its center (in spherical coordinates), its radius, and a differential parameter that quantifies the
    change induced by the spot. The smoothness parameter allows for a gradual transition at the spot's edges.

    Args:
        mesh (MeshModel): The mesh model to which the spot will be added. Must not be a PHOEBE model, as they are read-only.
        spot_center_theta (float): The theta (inclination) coordinate of the spot's center, in radians.
        spot_center_phi (float): The phi (azimuthal) coordinate of the spot's center, in radians.
        spot_radius (float): The angular radius of the spot, in degrees.
        parameter_delta (float): The difference in the parameter value to be applied within the spot.
        parameter_index (int): The index of the parameter in the mesh model that will be modified by the spot.
        smoothness (float, optional): A parameter controlling the smoothness of the spot's edge. Higher values result in smoother transitions. Defaults to 0.0.

    Returns:
        MeshModel: The modified mesh model with the spot applied.

    Raises:
        ValueError: If the input mesh model is a PHOEBE model, as modifications are not supported.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only.")
    else:
        return _add_spot(mesh, spot_center_theta, spot_center_phi, spot_radius,
                         parameter_delta, parameter_index, smoothness)


def add_spots(mesh: MeshModel,
              spot_center_thetas: Float[Array, "n_spots"],
              spot_center_phis: Float[Array, "n_spots"],
              spot_radii: Float[Array, "n_spots"],
              parameter_deltas: Float[Array, "n_spots"],
              parameter_indices: Int[Array, "n_spots"],
              smoothness: Float[Array, "n_spots"] = None) -> MeshModel:
    """
    Add multiple spots to a mesh model based on spherical coordinates and smoothness parameters.

    This function applies modifications to the mesh model's parameters to simulate the presence of multiple spots.
    Each spot is defined by its center (in spherical coordinates), its radius, and a differential parameter that
    quantifies the change induced by the spot. The smoothness parameter allows for a gradual transition at the spot's edges.

    Args:
        mesh (MeshModel): The mesh model to which the spots will be added. Must not be a PHOEBE model, as they are read-only.
        spot_center_thetas (Float[Array, "n_spots"]): An array of theta (inclination) coordinates of the spots' centers, in radians.
        spot_center_phis (Float[Array, "n_spots"]): An array of phi (azimuthal) coordinates of the spots' centers, in radians. 
        spot_radii (Float[Array, "n_spots"]): An array of angular radii of the spots, in degrees.
        parameter_deltas (Float[Array, "n_spots"]): An array of differences in the parameter values to be applied within the spots.
        parameter_indices (Int[Array, "n_spots"]): An array of indices of the parameters in the mesh model that will be modified by the spots.
        smoothness (Float[Array, "n_spots"], optional): An array of factors controlling the smoothness of the spots' edges. Higher values result in smoother transitions. Defaults to None.

    Returns:
        MeshModel: The modified mesh model with the spots applied.

    Raises:
        ValueError: If the input mesh model is a PHOEBE model, as modifications are not supported.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only.")
    else:
        if parameter_indices is None:
            parameter_indices = jnp.ones_like(parameter_deltas)
        
        def scan_fn(carry, x):
            mesh, theta, phi, radius, delta, index, smooth = carry, *x
            return _add_spot(mesh, theta, phi, radius, delta, index, smooth), None

        initial_carry = mesh
        xs = (spot_center_thetas, spot_center_phis, spot_radii,
              parameter_deltas, parameter_indices, smoothness)
        
        final_mesh, _ = jax.lax.scan(scan_fn, initial_carry, xs)
        return final_mesh


@jax.jit
def _add_spherical_harmonic_spots(mesh: MeshModel,
                                  m: ArrayLike, n: ArrayLike,
                                  param_deltas: ArrayLike,
                                  param_indices: ArrayLike) -> MeshModel:
    def scan(carry, params):
        return _add_spherical_harmonic_spot(
            carry, m=params[0].astype(int), n=params[1].astype(int),
            param_delta=params[2], param_index=params[3].astype(int)
        ), params

    updated_mesh, _ = jax.lax.scan(scan, mesh, jnp.vstack([m, n, param_deltas, param_indices]).T)
    return updated_mesh


@jax.jit
def _add_spherical_harmonic_spots_with_tilt(mesh: MeshModel,
                                  m: ArrayLike, n: ArrayLike,
                                  param_deltas: ArrayLike,
                                  param_indices: ArrayLike,
                                  tilt_axes: ArrayLike,
                                  tilt_angles: ArrayLike) -> MeshModel:
    def scan(carry, params):
        # Extract tilt axis from params[4:7] since it's 3 components
        tilt_axis = params[4:7]
        return _add_spherical_harmonic_spot_with_tilt(
            carry, m=params[0].astype(int), n=params[1].astype(int),
            param_delta=params[2], param_index=params[3].astype(int),
            tilt_axis=tilt_axis, tilt_angle=params[7]
        ), params

    # Reshape tilt_axes to flatten the last dimension
    tilt_axes_flat = tilt_axes.reshape(-1)
    
    # Stack parameters, handling the flattened tilt axes
    params = jnp.vstack([
        m, n, param_deltas, param_indices,
        tilt_axes_flat[0::3],  # x components
        tilt_axes_flat[1::3],  # y components 
        tilt_axes_flat[2::3],  # z components
        tilt_angles
    ]).T

    updated_mesh, _ = jax.lax.scan(scan, mesh, params)
    return updated_mesh


def add_spherical_harmonic_spots(mesh: MeshModel,
                                 m_orders: Float[Array, "n_orders"],
                                 n_degrees: Float[Array, "n_orders"],
                                 param_deltas: Float[Array, "n_orders"],
                                 param_indices: Float[Array, "n_orders"],
                                 tilt_axes: Optional[Float[Array, "n_orders 3"]] = None,
                                 tilt_angles: Optional[Float[Array, "n_orders"]] = None) -> MeshModel:
    """
    Add multiple spherical harmonic spots to a mesh model.

    This function iteratively applies spherical harmonic modifications to a mesh model based on the provided parameters.
    Each spot is defined by spherical harmonic indices (m_order, n_degree), a parameter delta indicating the modification strength,
    and a parameter index specifying which parameter of the mesh model is modified. The function checks if the mesh model
    is not a PHOEBE model, as they are read-only, and raises an error if it is.

    Args:
        mesh (MeshModel): The mesh model to which the spots will be added.
        m_orders (Float[Array, "n_orders"]): An array of m indices (orders) for the spherical harmonics.
        n_degrees (Float[Array, "n_orders"]): An array of n indices (degrees) for the spherical harmonics.
        param_deltas (Float[Array, "n_orders"]): An array of deltas specifying the strength of the modification for each spot.
        param_indices (Float[Array, "n_orders"]): An array of parameter indices specifying which parameter of the mesh is modified by each spot.
        tilt_axes (Optional[Float[Array, "n_orders 3"]], optional): An array of tilt axes for each spot. Shape should be (n_orders, 3). Defaults to None (no tilt).
        tilt_angles (Optional[Float[Array, "n_orders"]], optional): An array of tilt angles in degrees for each spot. Defaults to None (no tilt).

    Returns:
        MeshModel: The modified mesh model with all spherical harmonic spots applied.

    Raises:
        ValueError: If the input mesh model is a PHOEBE model, as modifications are not supported.
        ValueError: If tilt_axes is provided without tilt_angles or vice versa.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only.")
    if (tilt_axes is not None) and (tilt_angles is not None):
        return _add_spherical_harmonic_spots_with_tilt(mesh, m_orders, n_degrees, param_deltas, param_indices, tilt_axes, tilt_angles)
    else:
        return _add_spherical_harmonic_spots(mesh, m_orders, n_degrees, param_deltas, param_indices)
