import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from typing import Tuple

from jaxtyping import Array, Float


import jax
from typing import List, Any


class ModelList:
    def __init__(self, items: List[Any]):
        self.models = items

# Register the class as a pytree
def _my_list_container_flatten(container):
    # Return a tuple of (children, auxiliary_data)
    # children: the arrays/values that JAX should transform
    # auxiliary_data: metadata that won't be transformed
    children = container.models
    aux_data = None  # No additional metadata needed
    return (children, aux_data)

def _my_list_container_unflatten(aux_data, children):
    # Reconstruct the original object from transformed children
    return ModelList(children)

jax.tree_util.register_pytree_node(
    ModelList,
    _my_list_container_flatten,
    _my_list_container_unflatten
)


def vertex_to_polar(v: Float[Array, "3"]) -> Float[Array, "2"]:
    v += 1e-5
    r = jnp.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-5
    return jnp.nan_to_num(
        jnp.array([
            jnp.arctan2(v[2], r),
            jnp.arctan2(v[1], v[0])
        ])
    )


def lat_to_theta(lat_deg: float) -> float:
    """
    Convert latitude in degrees to theta (inclination) in radians.
    
    Args:
        lat_deg (float): Latitude in degrees (-90 to 90)
        
    Returns:
        float: Theta (inclination) in radians (0 to pi)
    """
    return jnp.deg2rad(90 - lat_deg)  # Convert latitude to inclination angle (0 to pi)


def lon_to_phi(lon_deg: float) -> float:
    """
    Convert longitude in degrees to phi (azimuth) in radians.
    
    Args:
        lon_deg (float): Longitude in degrees (-180 to 180)
        
    Returns:
        float: Phi (azimuth) in radians (-pi to pi)
    """
    return jnp.deg2rad(lon_deg)  # Convert longitude to azimuth angle (-pi to pi)


def theta_to_lat(theta: float) -> float:
    """
    Convert theta (inclination) in radians to latitude in degrees.
    
    Args:
        theta (float): Inclination angle in radians (0 to pi)
        
    Returns:
        float: Latitude in degrees (-90 to 90)
    """
    return 90 - jnp.rad2deg(theta)  # Convert inclination to latitude


def phi_to_lon(phi: float) -> float:
    """
    Convert phi (azimuth) in radians to longitude in degrees.
    
    Args:
        phi (float): Azimuth angle in radians (-pi to pi)
        
    Returns:
        float: Longitude in degrees (-180 to 180)
    """
    return jnp.rad2deg(phi)  # Convert azimuth to longitude


def velocity_to_period(velocity: float, radius: float) -> float:
    """
    Convert rotational velocity to rotation period.
    
    Args:
        velocity (float): Rotational velocity at the equator in km/s.
        radius (float): Radius of the star in solar radii.
    
    Returns:
        float: Rotation period in seconds.
    """
    radius_km = radius * 6.957e5  # Convert solar radii to km
    circumference = 2 * jnp.pi * radius_km  # Circumference in km
    period_seconds = circumference / velocity  # Period in seconds
    return period_seconds


def period_to_velocity(period: float, radius: float) -> float:
    """
    Convert rotation period to rotational velocity.
    
    Args:
        period (float): Rotation period in seconds.
        radius (float): Radius of the star in solar radii.
    
    Returns:
        float: Rotational velocity in km/s.
    """
    radius_km = radius * 6.957e5  # Convert solar radii to km
    circumference = 2 * jnp.pi * radius_km  # Circumference in km
    velocity = circumference / period  # Velocity in km/s
    return velocity


def inclination_to_los_axis(inclination: float) -> Float[Array, "3"]:
    """
    Convert inclination to tilt angle.
    
    Args:
        inclination (float): Inclination angle in degrees (0 to 90)
    
    Returns:
        float: LOS axis
    """
    incl_rad = jnp.deg2rad(inclination)
    return jnp.array([0., jnp.sin(incl_rad), -jnp.cos(incl_rad)])



@jax.jit
def mesh_polar_vertices(vertices: Float[Array, "n_vertices 3"]) -> Float[Array, "n_vertices 2"]:
    return (jax.vmap(vertex_to_polar, in_axes=0)(vertices))

@jax.jit
def spherical_harmonic(m, n, coordinates):
    polar_coordinates = mesh_polar_vertices(coordinates)
    m_array = (m * jnp.ones_like(polar_coordinates[:, 0])).astype(int)
    n_array = (n * jnp.ones_like(polar_coordinates[:, 1])).astype(int)
    return jax.scipy.special.sph_harm(m_array,
                                      n_array,
                                      polar_coordinates[:, 0],  # azimuthal angle (phi)
                                      polar_coordinates[:, 1],  # polar angle (theta)
                                      n_max=10).real


@jax.jit
def spherical_harmonic_with_tilt(m, n, coordinates, tilt_axis=jnp.array([0., 0., 1.]), tilt_angle=0.):
    # Normalize tilt axis
    norm = jnp.linalg.norm(tilt_axis)
    # Add epsilon to prevent division by zero
    norm = jnp.where(norm > 1e-10, norm, 1e-10)
    tilt_axis = tilt_axis / norm
    
    r_matrix = evaluate_rotation_matrix(rotation_matrix(tilt_axis), jnp.deg2rad(tilt_angle))
    rotated_coords = jnp.matmul(coordinates, r_matrix)

    return spherical_harmonic(m, n, rotated_coords)


@jax.jit
def evaluate_fourier_for_value(P: float, d: Float[Array, "1 n_orders"], phi: Float[Array, "1 n_orders"], timestamp: float) -> float:
    """
    Evaluate a Fourier series for a specific timestamp.
    
    Args:
        P (float): period P in seconds
        d (Float[Array, "1 n_orders"]): amplitudes for each order
        phi (Float[Array, "1 n_orders"]): phases for each order
        timestamp (float): time at which to evaluate the series
        
    Returns:
        float: The computed Fourier series value
    """
    # Make sure we don't have NaN values in the input
    d = jnp.nan_to_num(d)
    phi = jnp.nan_to_num(phi)
    
    def compute_fourier(P, d, phi, timestamp):
        n = jnp.arange(1, d.shape[0] + 1)
        return jnp.sum(d * jnp.cos(2 * jnp.pi * n / P * timestamp - phi))
    
    def return_zero():
        return 0.0
    
    # Skip computation if period is NaN or 0
    return jax.lax.cond(
        jnp.isnan(P) | (P == 0),
        return_zero,
        lambda: compute_fourier(P, d, phi, timestamp)
    )


@jax.jit
def evaluate_fourier_prim_for_value(P: float, d: Float[Array, "1 n_orders"], phi: Float[Array, "1 n_orders"], timestamp: float) -> float:
    """
    Evaluate the derivative of a Fourier series for a specific timestamp.
    
    Args:
        P (float): period P in seconds
        d (Float[Array, "1 n_orders"]): amplitudes for each order
        phi (Float[Array, "1 n_orders"]): phases for each order
        timestamp (float): time at which to evaluate the derivative
        
    Returns:
        float: The computed derivative value
    """
    # Make sure we don't have NaN values in the input
    d = jnp.nan_to_num(d)
    phi = jnp.nan_to_num(phi)
    
    def compute_fourier_prim(P, d, phi, timestamp):
        n = jnp.arange(1, d.shape[0] + 1)
        return -2 * jnp.pi * jnp.sum(d * n / P * jnp.sin(2 * jnp.pi * n / P * timestamp - phi))
    
    def return_zero():
        return 0.0
    
    # Skip computation if period is NaN or 0
    return jax.lax.cond(
        jnp.isnan(P) | (P == 0),
        return_zero,
        lambda: compute_fourier_prim(P, d, phi, timestamp)
    )


evaluate_many_fouriers_for_value = jax.jit(jax.vmap(evaluate_fourier_for_value, in_axes=(0, 0, 0, None)))
evaluate_many_fouriers_prim_for_value = jax.jit(jax.vmap(evaluate_fourier_prim_for_value, in_axes=(0, 0, 0, None)))


@jax.jit
def rotation_matrix(rotation_axis: Float[Array, "3"]) -> Float[Array, "3 3"]:
    a_norm = rotation_axis / jnp.linalg.norm(rotation_axis)
    a_hat = jnp.array([[0., -a_norm[2], a_norm[1]],
                       [a_norm[2], 0., -a_norm[0]],
                       [-a_norm[1], a_norm[0], 0.]])
    return a_hat


@jax.jit
def evaluate_rotation_matrix(rotation_matrix: Float[Array, "3 3"], theta: float) -> Float[Array, "3 3"]:
    return jnp.eye(3) + jnp.sin(theta) * rotation_matrix + (1 - jnp.cos(theta)) * jnp.matmul(rotation_matrix,
                                                                                             rotation_matrix)


@jax.jit
def rotation_matrix_prim(rotation_axis: Float[Array, "3"]) -> Float[Array, "3 3"]:
    a_norm = rotation_axis / jnp.linalg.norm(rotation_axis)
    a_hat = jnp.array([[0., -a_norm[2], a_norm[1]],
                       [a_norm[2], 0., -a_norm[0]],
                       [-a_norm[1], a_norm[0], 0.]])
    return a_hat


@jax.jit
def evaluate_rotation_matrix_prim(rotation_matrix_grad: Float[Array, "3 3"], theta: float) -> Float[Array, "3 3"]:
    return jnp.cos(theta) * rotation_matrix_grad + jnp.sin(theta) * jnp.matmul(rotation_matrix_grad,
                                                                               rotation_matrix_grad)


@jax.jit
def cast_to_los(vectors: Float[Array, "batch 3"], los_vector: Float[Array, "3"]) -> Float[Array, "batch"]:
    """Cast 3D vectors to the line-of-sight

    Args:
        vectors (Float[Array, "batch 3"]): Properties to be casted (n, 3)
        los_vector (Float[Array, "3"]): LOS vector (3,)

    Returns:
        ArrayLike: Casted vectors (n, 1)
    """
    return -1. * jnp.dot(vectors, los_vector)


@jax.jit
def cast_to_normal_plane(vectors: Float[Array, "batch 3"], los_vector: Float[Array, "3"]) -> Float[Array, "batch 2"]:
    """Cast 3D vectors to a 2D plane determined by a normal vector

    Args:
        vectors (Float[Array, "batch 3"]): Properties to be casted (n, 3)
        los_vector (Float[Array, "3"]): LOS vector (3,)

    Returns:
        ArrayLike: Casted vectors (n, 2)
    """
    """Cast 3D vectors to a 2D plane determined by the line-of-sight vector"""
    # Calculate the normal vector from the line-of-sight vector
    n = los_vector / jnp.linalg.norm(los_vector)
    
    # Create two orthogonal vectors in the plane perpendicular to the los_vector
    v1 = jnp.array([n[1], -n[0], 0])
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = jnp.cross(n, v1)
    
    # Project the vectors onto the plane
    x = jnp.dot(vectors, v1)
    y = jnp.dot(vectors, v2)
    
    return jnp.column_stack((x, y))


@jax.jit
def cast_normalized_to_los(vectors: Float[Array, "batch 3"], los_vector: Float[Array, "3"]) -> Float[Array, "batch"]:
    """Cast 3D vectors to the line-of-sight

    Args:
        vectors (Float[Array, "batch 3"]): Properties to be casted (n, 3)
        los_vector (Float[Array, "3"]): LOS vector (3,)

    Returns:
        ArrayLike: Casted vectors (n, 1)
    """
    last_axis = len(vectors.shape) - 1
    return -1. * jnp.dot(vectors / (jnp.linalg.norm(vectors, axis=last_axis, keepdims=True) + 1e-10),
                         los_vector)


@jax.jit
def calculate_axis_radii(centers: Float[Array, "n_mesh_elements 3"], axis: Float[Array, "3"]) -> Float[Array, "n_mesh_elements"]:
    norm_axis = len(centers.shape) - 1
    return jnp.linalg.norm(jnp.cross(axis, -centers),
                           axis=norm_axis) / jnp.linalg.norm(axis)


### UTILITY FUNCTIONS FOR FOURIER SERIES

@jax.jit
def cos_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[float, ArrayLike]:
    return P, jnp.array([[max_amplitude, 0.]])


@jax.jit
def sin_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[float, ArrayLike]:
    return P, jnp.array([[max_amplitude, -jnp.pi / 2]])


@jax.jit
def sin2_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[float, ArrayLike]:
    return 0.5 * P, jnp.array([[0.5 * max_amplitude, 0.]])


@jax.jit
def sinh_as_fourier_coords(P: float, max_amplitude: float, n: int) -> Tuple[float, ArrayLike]:
    return P, jnp.array(
        [[2 * jnp.sinh(jnp.pi) / jnp.pi * jnp.power(-1, n0 + 1) * n0 / (jnp.power(n0, 2) + 1) * max_amplitude, 0.] for
         n0 in range(1, n + 1)])
