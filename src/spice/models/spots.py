from typing import Union

import jax
import jax.numpy as jnp
from PIL.ImageOps import scale
from jax import Array
from jax.typing import ArrayLike
from spice.models.phoebe_model import PhoebeModel
from .mesh_model import MeshModel
from .utils import mesh_polar_vertices, spherical_harmonic, spherical_harmonic_with_tilt


def _is_arraylike(x):
    return hasattr(x, '__array__') or hasattr(x, '__array_interface__')


def __cos_law_distance(theta1, phi1, theta2, phi2):
    return jnp.arccos(jnp.sin(theta1) * jnp.sin(theta2) + jnp.cos(theta1) * jnp.cos(theta2) * jnp.cos(phi1 - phi2))


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
def generate_spherical_spot(d_centers: ArrayLike,
                            mesh_radius: float,
                            spot_center_theta: float,
                            spot_center_phi: float,
                            spot_radius: float,
                            parameter_diff: float,
                            smoothness: float = 1.0):
    """
    Generate a differential parameter array based on the location of points on a sphere relative to a specified
    spherical spot.

    This function calculates the angular distance of each point (defined by `thetas` and `phis`) from a specified
    spot center (`spot_center_theta`, `spot_center_phi`). It then determines the differential parameter value for
    each point based on whether it lies within the spherical spot, applying a smoothness factor to interpolate values
    at the spot's edge.

    Args:
        d_centers (Array): Cartesian coordinates of the centers of the mesh elements
        mesh_radius (float): Radius of the mesh
        spot_center_theta (float): Theta value (inclination) of the spot center.
        spot_center_phi (float): Phi value (azimuth) of the spot center.
        spot_radius (float): Angular radius of the spot.
        parameter_diff (float): The differential parameter value to be applied within the spot.
        smoothness (float, optional): Factor controlling the smoothness of the spot's edge. Defaults to 1.0.

    Returns: ArrayLike: An array of differential parameter values for each point, interpolated based on spot
    membership and edge smoothness.
    """

    spot_center_x, spot_center_y, spot_center_z = __spherical_to_cartesian(spot_center_theta, spot_center_phi,
                                                                           mesh_radius)

    # Calculate distance from the spot center
    distances = v_cartesian_distance(d_centers[:, 0], d_centers[:, 1], d_centers[:, 2],
                                     spot_center_x, spot_center_y, spot_center_z)

    # Calculate the difference from the spot edge
    distance_from_edge = jnp.deg2rad(spot_radius) * mesh_radius - distances

    # Apply smoothness to interpolate values at the spot's edge
    differences = sigmoid(distance_from_edge / (0.01 * mesh_radius), smoothness) * parameter_diff

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
                                 tilt_degree: float = 0.) -> MeshModel:
    spot_parameters = spherical_harmonic_with_tilt(m, n, mesh.centers, tilt_axis, tilt_degree) * param_delta
    return mesh._replace(parameters=mesh.parameters.at[:, param_index].set(
        mesh.parameters[:, param_index] + spot_parameters))


def add_spherical_harmonic_spot(mesh: MeshModel,
                                m_order: Union[int, ArrayLike],
                                n_degree: Union[int, ArrayLike],
                                param_delta: Union[float, ArrayLike],
                                param_index: Union[int, ArrayLike],
                                tilt_axis: ArrayLike = None,
                                tilt_degree: float = None) -> MeshModel:
    """
    Add a spherical harmonic variation to a parameter of the mesh model, effectively creating a spot-like feature.

    This function applies a spherical harmonic function to modify a specific parameter of the mesh model,
    creating a pattern that can represent various surface features like spots or temperature variations.

    Args:
        mesh (MeshModel): The mesh model to which the spherical harmonic variation will be applied.
        m_order (Union[int, ArrayLike]): The order (m) of the spherical harmonic. Determines the number of
                                         longitudinal nodes in the pattern.
        n_degree (Union[int, ArrayLike]): The degree (n) of the spherical harmonic. Determines the total
                                          number of nodes in the pattern.
        param_delta (Union[float, ArrayLike]): The maximum amplitude of the parameter variation. Represents
                                               the difference between the peak of the variation and the
                                               background value.
        param_index (Union[int, ArrayLike]): The index of the parameter in the mesh model's parameter array
                                             that will be modified by this variation.
        tilt_axis (ArrayLike, optional): The axis of tilt for the harmonic series. Defaults to None (no tilt).
        tilt_degree (float, optional): The degree of tilt for the harmonic series. Defaults to None (no tilt).

    Returns:
        MeshModel: A new mesh model with the specified parameter modified according to the spherical
                   harmonic variation.

    Note:
        The spherical harmonic function Y_n^m(θ,φ) is used to create the variation pattern on the stellar surface.
        The combination of m_order and n_degree determines the complexity and shape of the resulting pattern.
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
        return _add_spherical_harmonic_spot_with_tilt(mesh, m_order, n_degree, param_delta, param_index, tilt_axis, tilt_degree)
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
             smoothness: float = 1.0) -> MeshModel:
    """
    Add a spot to a mesh model based on spherical coordinates and smoothness parameters.

    This function applies a modification to the mesh model's parameters to simulate the presence of a spot. The spot
    is defined by its center (in spherical coordinates), its radius, and a differential parameter that quantifies the
    change induced by the spot. The smoothness parameter allows for a gradual transition at the spot's edges.

    Args:
        mesh (MeshModel): The mesh model to which the spot will be added. Must not be a PHOEBE model, as they are read-only.
        spot_center_theta (float): The theta (inclination) coordinate of the spot's center, in radians.
        spot_center_phi (float): The phi (azimuthal) coordinate of the spot's center, in radians.
        spot_radius (float): The angular radius of the spot, in radians.
        parameter_delta (float): The difference in the parameter value to be applied within the spot.
        parameter_index (int): The index of the parameter in the mesh model that will be modified by the spot.
        smoothness (float, optional): A parameter controlling the smoothness of the spot's edge. Higher values result in smoother transitions. Defaults to 1.0.

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
              spot_center_thetas: ArrayLike,
              spot_center_phis: ArrayLike,
              spot_radii: ArrayLike,
              parameter_deltas: ArrayLike,
              parameter_indices: ArrayLike,
              smoothness: ArrayLike = None) -> MeshModel:
    """
    Add multiple spots to a mesh model based on spherical coordinates and smoothness parameters.

    This function applies modifications to the mesh model's parameters to simulate the presence of multiple spots.
    Each spot is defined by its center (in spherical coordinates), its radius, and a differential parameter that
    quantifies the change induced by the spot. The smoothness parameter allows for a gradual transition at the spot's edges.

    Args:
        mesh (MeshModel): The mesh model to which the spots will be added. Must not be a PHOEBE model, as they are read-only.
        spot_center_thetas (ArrayLike): An array of theta (inclination) coordinates of the spots' centers, in radians.
        spot_center_phis (ArrayLike): An array of phi (azimuthal) coordinates of the spots' centers, in radians.
        spot_radii (ArrayLike): An array of angular radii of the spots, in degrees.
        parameter_deltas (ArrayLike): An array of differences in the parameter values to be applied within the spots.
        parameter_indices (ArrayLike): An array of indices of the parameters in the mesh model that will be modified by the spots.
        smoothness (ArrayLike, optional): An array of factors controlling the smoothness of the spots' edges. Defaults to None.

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
        return _add_spherical_harmonic_spot_with_tilt(
            carry, m=params[0].astype(int), n=params[1].astype(int),
            param_delta=params[2], param_index=params[3].astype(int),
            tilt_axis=params[4], tilt_angle=params[5]
        ), params

    updated_mesh, _ = jax.lax.scan(scan, mesh, jnp.vstack([m, n, param_deltas, param_indices, tilt_axes, tilt_angles]).T)
    return updated_mesh


def add_spherical_harmonic_spots(mesh: MeshModel,
                                 m_orders: ArrayLike, n_degrees: ArrayLike,
                                 param_deltas: ArrayLike,
                                 param_indices: ArrayLike,
                                 tilt_axes: ArrayLike = None,
                                 tilt_angles: ArrayLike = None) -> MeshModel:
    """
    Add multiple spherical harmonic spots to a mesh model.

    This function iteratively applies spherical harmonic modifications to a mesh model based on the provided parameters.
    Each spot is defined by spherical harmonic indices (m, n), a parameter delta indicating the modification strength,
    and a parameter index specifying which parameter of the mesh model is modified. The function checks if the mesh model
    is not a PHOEBE model, as they are read-only, and raises an error if it is.

    Args:
        mesh (MeshModel): The mesh model to which the spots will be added.
        m (ArrayLike): An array of m indices for the spherical harmonics.
        n (ArrayLike): An array of n indices for the spherical harmonics.
        param_deltas (ArrayLike): An array of deltas specifying the strength of the modification for each spot.
        param_indices (ArrayLike): An array of parameter indices specifying which parameter of the mesh is modified by each spot.
        tilt_axes (ArrayLike, optional): An array of tilt axes for each spot. Defaults to None (no tilt).
        tilt_angles (ArrayLike, optional): An array of tilt angles for each spot. Defaults to None (no tilt).

    Returns:
        MeshModel: The modified mesh model with all spherical harmonic spots applied.

    Raises:
        ValueError: If the input mesh model is a PHOEBE model, as modifications are not supported.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only.")
    return _add_spherical_harmonic_spots(mesh, m_orders, n_degrees, param_deltas, param_indices)
