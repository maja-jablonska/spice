from typing import Union

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from spice.models.phoebe_model import PhoebeModel
from .mesh_model import MeshModel
from .utils import mesh_polar_vertices, spherical_harmonic


def _is_arraylike(x):
    return hasattr(x, '__array__') or hasattr(x, '__array_interface__')


def __cos_law_distance(theta1, phi1, theta2, phi2):
    return jnp.arccos(jnp.sin(theta1) * jnp.sin(theta2) + jnp.cos(theta1) * jnp.cos(theta2) * jnp.cos(phi1 - phi2))


v_cos_law_distance = jax.jit(jax.vmap(__cos_law_distance, in_axes=(0, 0, None, None)))


def sigmoid(x, smoothness):
    return 1 / (1 + jnp.exp(-smoothness * x))


@jax.jit
def generate_spherical_spot(thetas: ArrayLike,
                            phis: ArrayLike,
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
        thetas (ArrayLike): Array of theta values (inclination) for points on the sphere.
        phis (ArrayLike): Array of phi values (azimuth) for points on the sphere.
        spot_center_theta (float): Theta value (inclination) of the spot center.
        spot_center_phi (float): Phi value (azimuth) of the spot center.
        spot_radius (float): Angular radius of the spot.
        parameter_diff (float): The differential parameter value to be applied within the spot.
        smoothness (float, optional): Factor controlling the smoothness of the spot's edge. Defaults to 1.0.

    Returns: ArrayLike: An array of differential parameter values for each point, interpolated based on spot
    membership and edge smoothness.
    """

    # Calculate distance from the spot center
    distance = v_cos_law_distance(thetas, phis, spot_center_theta, spot_center_phi)

    # Calculate the difference from the spot edge
    distance_from_edge = spot_radius - distance

    # Apply smoothness to interpolate values at the spot's edge
    differences = sigmoid(distance_from_edge, smoothness) * parameter_diff

    return differences


generate_spherical_spots = jax.jit(jax.vmap(generate_spherical_spot, in_axes=(None, None,
                                                                              0, 0, 0, 0, 0)))


@jax.jit
def _add_spherical_harmonic_spot(mesh: MeshModel,
                                 m: int, n: int,
                                 param_delta: float,
                                 param_index: int) -> MeshModel:
    center_polar_coords = mesh_polar_vertices(mesh.centers)
    spot_parameters = spherical_harmonic(m, n, center_polar_coords) * param_delta
    return mesh._replace(parameters=mesh.parameters.at[:, param_index].set(
        mesh.parameters[:, param_index] + spot_parameters))


def add_spherical_harmonic_spot(mesh: MeshModel,
                                m: Union[int, ArrayLike],
                                n: Union[int, ArrayLike],
                                param_delta: Union[float, ArrayLike],
                                param_index: Union[int, ArrayLike]) -> MeshModel:
    """Add a spot/parameter variation as a spherical harmonic to the mesh model.

    Args:
        mesh (MeshModel): mesh model
        m (Union[int, ArrayLike]): m index of the spherical harmonic
        n (Union[int, ArrayLike]): n index of the spherical harmonic
        param_delta (Union[float, ArrayLike]): difference in the parameter value between the spot and the background
        param_index (Union[int, ArrayLike]): index of the parameter in the parameters array

    Returns:
        MeshModel: mesh with the spot added
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only.")
    if isinstance(m, int) and isinstance(n, int) and m > n:
        raise ValueError("m must be lesser or equal to n.")
    elif _is_arraylike(m) and _is_arraylike(n):
        m = m.astype(int)
        n = n.astype(int)
        if jnp.any(jnp.greater(m, n)):
            raise ValueError("m must be lesser or equal to n.")

    return _add_spherical_harmonic_spot(mesh, m, n, param_delta, param_index)


@jax.jit
def _add_spot(mesh: MeshModel,
              spot_center_theta: float,
              spot_center_phi: float,
              spot_radius: float,
              parameter_diff: float,
              parameter_index: int,
              smoothness: float = 1.0) -> MeshModel:
    center_polar_coords = mesh_polar_vertices(mesh.centers)
    spot_parameters = generate_spherical_spot(center_polar_coords[:, 0], center_polar_coords[:, 1],
                                              spot_center_theta, spot_center_phi, spot_radius, parameter_diff,
                                              smoothness)
    return mesh._replace(parameters=mesh.parameters.at[:, parameter_index].set(
        mesh.parameters[:, parameter_index] + spot_parameters)
    )


def add_spot(mesh: MeshModel,
             spot_center_theta: float,
             spot_center_phi: float,
             spot_radius: float,
             parameter_diff: float,
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
        parameter_diff (float): The difference in the parameter value to be applied within the spot.
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
                         parameter_diff, parameter_index, smoothness)


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


def add_spherical_harmonic_spots(mesh: MeshModel,
                                 m: ArrayLike, n: ArrayLike,
                                 param_deltas: ArrayLike,
                                 param_indices: ArrayLike) -> MeshModel:
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

    Returns:
        MeshModel: The modified mesh model with all spherical harmonic spots applied.

    Raises:
        ValueError: If the input mesh model is a PHOEBE model, as modifications are not supported.
    """
    if isinstance(mesh, PhoebeModel):
        raise ValueError("PHOEBE models are read-only.")

    if isinstance(m, int) and isinstance(n, int) and m > n:
        raise ValueError("m must be lesser or equal to n.")
    elif _is_arraylike(m) and _is_arraylike(n):
        m = m.astype(int)
        n = n.astype(int)
        if jnp.any(jnp.greater(m, n)):
            raise ValueError("m must be lesser or equal to n.")
    return _add_spherical_harmonic_spots(mesh, m, n, param_deltas, param_indices)
