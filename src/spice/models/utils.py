from __future__ import annotations

import time as _time
import sys as _sys

_jax_already_loaded = "jax" in _sys.modules
_t0 = _time.perf_counter()
import jax.numpy as jnp
import jax
if not _jax_already_loaded:
    from spice.utils import log as _log
    _log.info(f"JAX loaded in {_time.perf_counter() - _t0:.1f} s")
from jax.typing import ArrayLike
from typing import List, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Array, Float

from spice.utils.dtypes import float_dtype as _float_dtype
from spice.constants import SOLAR_MASS_KG, SOLAR_RAD_M, SOLAR_RAD_KM, G_SI, DAY_TO_S


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
    """Cartesian (x, y, z) -> (theta, phi) with theta = colatitude in [0, pi]
    and phi = azimuth in [-pi, pi].

    Exact everywhere, including on the poles (where phi is set to 0), and
    gradient-safe there: the double-``where`` keeps ``sqrt`` and ``arctan2``
    away from their singular point in the backward pass. An earlier version
    shifted every point by 1e-5 instead, which at a pole vertex was 10% of the
    1e-4 finite-difference step of the tangential gradient and inflated the
    m != 0 horizontal pulsation basis by ~10%.
    """
    xy2 = v[0] ** 2 + v[1] ** 2
    off_axis = xy2 > 0.
    xy = jnp.where(off_axis, jnp.sqrt(jnp.where(off_axis, xy2, 1.)), 0.)
    phi = jnp.where(off_axis,
                    jnp.arctan2(v[1], jnp.where(off_axis, v[0], 1.)),
                    0.)
    return jnp.nan_to_num(jnp.array([jnp.arctan2(xy, v[2]), phi]))


# Highest spherical-harmonic degree `spherical_harmonic` evaluates correctly
# (`jax.scipy.special.sph_harm_y` is called with this static n_max).
MAX_SPHERICAL_HARMONIC_DEGREE = 10


@jax.jit
def rotation_axis_frame(axis: Float[Array, "3"]) -> Float[Array, "3 3"]:
    """Orthonormal frame whose third column is ``axis``.

    Returns the matrix ``B = [e1 | e2 | a]`` of the minimal rotation taking +z
    onto the unit vector ``a``, so ``coords @ B`` expresses Cartesian points in
    a frame where the rotation axis is the pole. Latitudes, longitudes and
    spherical harmonics are measured in this frame. For ``axis = +z`` it is
    exactly the identity, so results for stars with an explicit z rotation
    axis are unchanged; for ``-z`` it is ``diag(1, -1, -1)``.
    """
    a = axis / jnp.linalg.norm(axis)
    z = jnp.array([0., 0., 1.], dtype=a.dtype)
    k = jnp.cross(z, a)
    s2 = jnp.dot(k, k)
    c = jnp.dot(z, a)
    K = jnp.array([[0., -k[2], k[1]],
                   [k[2], 0., -k[0]],
                   [-k[1], k[0], 0.]], dtype=a.dtype)
    general = jnp.eye(3, dtype=a.dtype) + K + (K @ K) * ((1. - c) / jnp.where(s2 > 0., s2, 1.))
    aligned = jnp.where(c > 0., jnp.eye(3, dtype=a.dtype),
                        jnp.diag(jnp.array([1., -1., -1.], dtype=a.dtype)))
    return jnp.where(s2 > 1e-24, general, aligned)


def _tilted_frame(tilt_axis, tilt_angle, frame_axis):
    """Matrix M with ``coords @ M`` = coordinates in the (tilted) pattern frame.

    The pattern's pole is ``frame_axis`` (default +z, i.e. no change), tilted by
    ``tilt_angle`` degrees about ``tilt_axis``.
    """
    if tilt_axis is None:
        tilt_axis = jnp.array([0., 0., 1.])
    norm = jnp.linalg.norm(tilt_axis)
    # Add epsilon to prevent division by zero
    norm = jnp.where(norm > 1e-10, norm, 1e-10)
    tilt_axis = tilt_axis / norm
    r_matrix = evaluate_rotation_matrix(rotation_matrix(tilt_axis), jnp.deg2rad(tilt_angle))
    if frame_axis is not None:
        r_matrix = jnp.matmul(r_matrix, rotation_axis_frame(frame_axis))
    return r_matrix


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


def horizontal_to_radial_ratio(mass: float, radius: float, pulsation_period: float) -> float:
    """
    Compute the horizontal-to-radial displacement ratio K from stellar parameters.

    From the outer mechanical boundary condition (Smeyers & Tassoul 1987):
        K = GM / (omega^2 * R^3)

    For p-modes K << 1 (radial dominates); for g-modes K >> 1 (horizontal dominates).

    Args:
        mass (float): Stellar mass in solar masses.
        radius (float): Stellar radius in solar radii.
        pulsation_period (float): Pulsation period in days.

    Returns:
        float: The ratio K = xi_h / xi_r, suitable for use as `horizontal_ratio`
            in `add_pulsation()`.
    """
    G = G_SI            # m^3 kg^-1 s^-2
    M = mass * SOLAR_MASS_KG  # kg
    R = radius * SOLAR_RAD_M # m
    omega = 2 * jnp.pi / (pulsation_period * DAY_TO_S)  # rad/s
    return G * M / (omega**2 * R**3)


def velocity_to_period(velocity: float, radius: float) -> float:
    """
    Convert rotational velocity to rotation period.
    
    Args:
        velocity (float): Rotational velocity at the equator in km/s.
        radius (float): Radius of the star in solar radii.
    
    Returns:
        float: Rotation period in seconds.
    """
    radius_km = radius * SOLAR_RAD_KM  # Convert solar radii to km
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
    radius_km = radius * SOLAR_RAD_KM  # Convert solar radii to km
    circumference = 2 * jnp.pi * radius_km  # Circumference in km
    velocity = circumference / period  # Velocity in km/s
    return velocity


def inclination_to_los_axis(inclination: float,
                            rotation_axis: Float[Array, "3"] = None) -> Float[Array, "3"]:
    """
    Line-of-sight vector (observer -> star) at inclination ``i`` to ``rotation_axis``.

    In the rotation-axis frame (:func:`rotation_axis_frame`, the same frame
    spot latitudes/longitudes and pulsation modes are measured in) the vector is
    ``(0, sin i, -cos i)``, so the angle between the LOS and the rotation axis
    is exactly ``i`` for any axis. Pass the mesh's axis, e.g.
    ``inclination_to_los_axis(60., mesh.rotation_axis)``.

    Args:
        inclination (float): Inclination in degrees (0 = pole-on, 90 = equator-on).
        rotation_axis (Float[Array, "3"], optional): The star's rotation axis.
            If omitted, ``[0, 0, 1]`` is assumed and a warning is emitted, because
            the default axis of ``add_rotation`` is not z: using the z-based
            vector on a default-axis star silently gives the wrong inclination
            (30 deg instead of 60 deg).

    Returns:
        Float[Array, "3"]: LOS vector pointing from the observer towards the star.
    """
    if rotation_axis is None:
        import warnings
        warnings.warn(
            "inclination_to_los_axis() without rotation_axis assumes a +z rotation "
            "axis; the default add_rotation axis is +y. Pass rotation_axis="
            "mesh.rotation_axis (or the axis you give add_rotation).",
            UserWarning, stacklevel=2)
        rotation_axis = jnp.array([0., 0., 1.])
    incl_rad = jnp.deg2rad(inclination)
    los_frame = jnp.array([0., jnp.sin(incl_rad), -jnp.cos(incl_rad)])
    return rotation_axis_frame(jnp.asarray(rotation_axis, dtype=los_frame.dtype)) @ los_frame



@jax.jit
def mesh_polar_vertices(vertices: Float[Array, "n_vertices 3"]) -> Float[Array, "n_vertices 2"]:
    return (jax.vmap(vertex_to_polar, in_axes=0)(vertices))

@jax.jit
def spherical_harmonic(m, n, coordinates):
    polar_coordinates = mesh_polar_vertices(coordinates)
    m_array = (m * jnp.ones_like(polar_coordinates[:, 0])).astype(int)
    n_array = (n * jnp.ones_like(polar_coordinates[:, 1])).astype(int)
    return jax.scipy.special.sph_harm_y(n_array,
                                        m_array,
                                        polar_coordinates[:, 0],  # theta (colatitude)
                                        polar_coordinates[:, 1],  # phi (azimuth)
                                        n_max=MAX_SPHERICAL_HARMONIC_DEGREE).real


@jax.jit
def spherical_harmonic_with_tilt(m, n, coordinates, tilt_axis=None, tilt_angle=0., frame_axis=None):
    """Y_l^m with its pole on ``frame_axis`` (default +z), tilted by
    ``tilt_angle`` degrees about ``tilt_axis``."""
    r_matrix = _tilted_frame(tilt_axis, tilt_angle, frame_axis)
    rotated_coords = jnp.matmul(coordinates, r_matrix)

    return spherical_harmonic(m, n, rotated_coords)


@jax.jit
def spherical_harmonic_tangential_gradient(m, n, coordinates):
    """Compute the tangential gradient of Y_l^m on the sphere as 3D Cartesian vectors.

    Uses finite differences in the Cartesian frame to compute the full gradient
    of Y_l^m, then projects out the radial component to obtain the tangential part.
    This avoids coordinate singularities at the poles.

    Args:
        m: Azimuthal order of the spherical harmonic.
        n: Degree of the spherical harmonic.
        coordinates: (n_vertices, 3) Cartesian coordinates.

    Returns:
        (n_vertices, 3) tangential gradient vectors in Cartesian coordinates.
    """
    eps = 1e-4

    # Compute Cartesian gradient via central finite differences along x, y, z
    Y0 = spherical_harmonic(m, n, coordinates)  # (n,)

    dx = jnp.array([eps, 0.0, 0.0])
    dy = jnp.array([0.0, eps, 0.0])
    dz = jnp.array([0.0, 0.0, eps])

    dYdx = (spherical_harmonic(m, n, coordinates + dx) -
            spherical_harmonic(m, n, coordinates - dx)) / (2.0 * eps)
    dYdy = (spherical_harmonic(m, n, coordinates + dy) -
            spherical_harmonic(m, n, coordinates - dy)) / (2.0 * eps)
    dYdz = (spherical_harmonic(m, n, coordinates + dz) -
            spherical_harmonic(m, n, coordinates - dz)) / (2.0 * eps)

    grad_cart = jnp.stack([dYdx, dYdy, dYdz], axis=-1)  # (n, 3)

    # Project out the radial component: tangential = grad - (grad . r_hat) r_hat
    r_hat = coordinates / (jnp.linalg.norm(coordinates, axis=1, keepdims=True) + 1e-10)
    radial_component = jnp.sum(grad_cart * r_hat, axis=1, keepdims=True) * r_hat

    return grad_cart - radial_component


@jax.jit
def spherical_harmonic_gradient_with_tilt(m, n, coordinates,
                                          tilt_axis=None,
                                          tilt_angle=0.,
                                          frame_axis=None):
    """Compute the tangential gradient of Y_l^m with a tilted pulsation axis.

    Rotates coordinates into the tilted frame, computes the tangential gradient
    there, and rotates the resulting vectors back.

    Args:
        m: Azimuthal order.
        n: Degree.
        coordinates: (n_vertices, 3) Cartesian coordinates.
        tilt_axis: Rotation axis for the tilt.
        tilt_angle: Tilt angle in degrees.
        frame_axis: Pole of the harmonic before the tilt (default +z).

    Returns:
        (n_vertices, 3) tangential gradient vectors in the original frame.
    """
    r_matrix = _tilted_frame(tilt_axis, tilt_angle, frame_axis)
    rotated_coords = jnp.matmul(coordinates, r_matrix)

    rotated_grad = spherical_harmonic_tangential_gradient(m, n, rotated_coords)

    # Rotate gradient vectors back to the original frame
    return jnp.matmul(rotated_grad, r_matrix.T)


@jax.jit
def spherical_harmonic_toroidal(m, n, coordinates):
    """Compute the toroidal vector spherical harmonic T_l^m = r_hat x nabla_tangential(Y_l^m).

    This is the third basis vector of the vector spherical harmonic decomposition,
    orthogonal to both the radial (R_lm) and spheroidal (S_lm) components.

    Args:
        m: Azimuthal order of the spherical harmonic.
        n: Degree of the spherical harmonic.
        coordinates: (n_vertices, 3) Cartesian coordinates.

    Returns:
        (n_vertices, 3) toroidal vectors in Cartesian coordinates.
    """
    r_hat = coordinates / (jnp.linalg.norm(coordinates, axis=1, keepdims=True) + 1e-10)
    tangential_grad = spherical_harmonic_tangential_gradient(m, n, coordinates)
    return jnp.cross(r_hat, tangential_grad)


@jax.jit
def spherical_harmonic_toroidal_with_tilt(m, n, coordinates,
                                          tilt_axis=None,
                                          tilt_angle=0.,
                                          frame_axis=None):
    """Compute the toroidal VSH T_l^m with a tilted pulsation axis.

    Rotates coordinates into the tilted frame, computes the toroidal component
    there, and rotates the resulting vectors back.

    Args:
        m: Azimuthal order.
        n: Degree.
        coordinates: (n_vertices, 3) Cartesian coordinates.
        tilt_axis: Rotation axis for the tilt.
        tilt_angle: Tilt angle in degrees.

    Returns:
        (n_vertices, 3) toroidal vectors in the original frame.
    """
    r_matrix = _tilted_frame(tilt_axis, tilt_angle, frame_axis)
    rotated_coords = jnp.matmul(coordinates, r_matrix)

    rotated_toroidal = spherical_harmonic_toroidal(m, n, rotated_coords)

    return jnp.matmul(rotated_toroidal, r_matrix.T)


@jax.jit
def evaluate_fourier_for_value(P: float, d: Float[Array, "1 n_orders"], phi: Float[Array, "1 n_orders"], timestamp: float, d0: float = 0.0) -> float:
    """
    Evaluate a Fourier series for a specific timestamp.
    
    Args:
        P (float): period P in seconds
        d (Float[Array, "1 n_orders"]): amplitudes for each order
        phi (Float[Array, "1 n_orders"]): phases for each order
        timestamp (float): time at which to evaluate the series
        d0 (float): DC component (constant term) of the Fourier series
        
    Returns:
        float: The computed Fourier series value
    """
    # Make sure we don't have NaN values in the input
    d = jnp.nan_to_num(d)
    phi = jnp.nan_to_num(phi)
    d0 = jnp.nan_to_num(d0)
    
    def compute_fourier(P, d, phi, timestamp, d0):
        n = jnp.arange(1, d.shape[0] + 1)
        return d0 + jnp.sum(d * jnp.cos(2 * jnp.pi * n / P * timestamp - phi))
    
    def return_zero():
        return 0.0
    
    # Skip computation if period is NaN or 0
    return jax.lax.cond(
        jnp.isnan(P) | (P == 0),
        return_zero,
        lambda: compute_fourier(P, d, phi, timestamp, d0)
    )


@jax.jit
def evaluate_fourier_prim_for_value(P: float, d: Float[Array, "1 n_orders"], phi: Float[Array, "1 n_orders"], timestamp: float, d0: float = 0.0) -> float:
    """
    Evaluate the derivative of a Fourier series for a specific timestamp.
    
    Args:
        P (float): period P in seconds
        d (Float[Array, "1 n_orders"]): amplitudes for each order
        phi (Float[Array, "1 n_orders"]): phases for each order in radians
        timestamp (float): time at which to evaluate the derivative
        d0 (float): DC component (constant term) of the Fourier series (does not affect derivative)
        
    Returns:
        float: The computed derivative value
    """
    # Make sure we don't have NaN values in the input
    d = jnp.nan_to_num(d)
    phi = jnp.nan_to_num(phi)
    
    def compute_fourier_prim(P, d, phi, timestamp, d0):
        n = jnp.arange(1, d.shape[0] + 1)
        # This is the derivative of the Fourier series with respect to time
        # Original: d0 + d * cos(2π * n/P * t - phi)
        # Derivative: -d * (2π * n/P) * sin(2π * n/P * t - phi) (d0 term disappears)
        return jnp.sum(-d * (2 * jnp.pi * n / P) * jnp.sin(2 * jnp.pi * n / P * timestamp - phi))
    
    def return_zero():
        return 0.0
    
    # Skip computation if period is NaN or 0
    return jax.lax.cond(
        jnp.isnan(P) | (P == 0),
        return_zero,
        lambda: compute_fourier_prim(P, d, phi, timestamp, d0)
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
def cast_to_normal_plane(vectors: jnp.ndarray, los_vector: jnp.ndarray) -> jnp.ndarray:
    """
    Project 3D vectors onto the plane perpendicular to los_vector.
    
    Args:
        vectors: (..., 3) array of 3D vectors to project (batch or single).
        los_vector: (3,) line-of-sight vector defining the plane normal.
        
    Returns:
        (..., 2) array of 2D coordinates in an orthonormal basis of the plane.
    """
    eps = 1e-12

    # Normalize LOS safely
    n = los_vector / (jnp.linalg.norm(los_vector) + eps)  # (3,)

    # Choose a reference axis not parallel to n
    ez = jnp.array([0.0, 0.0, 1.0], dtype=_float_dtype())
    ex = jnp.array([1.0, 0.0, 0.0], dtype=_float_dtype())

    # If |n·ez| < 0.99 use ez, else fall back to ex
    use_ez = jnp.abs(jnp.dot(n, ez)) < 0.99
    ref = jax.lax.cond(use_ez, lambda _: ez, lambda _: ex, operand=None)

    # Build orthonormal basis in the plane
    v1 = jnp.cross(n, ref)
    v1 = v1 / (jnp.linalg.norm(v1) + eps)
    v2 = jnp.cross(n, v1)  # already orthogonal; norm ~1

    # Project
    # Support both (batch,3) and (3,) inputs:
    x = jnp.einsum('...i,i->...', vectors, v1)  # (...,)
    y = jnp.einsum('...i,i->...', vectors, v2)  # (...,)

    return jnp.stack([x, y], axis=-1)


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

# The Fourier evaluators use A cos(2 pi k t / P - phi) per [A, phi] pair.

@jax.jit
def cos_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[float, ArrayLike]:
    """(P, [[A, phi]]) for ``A cos(2 pi t / P)``."""
    return P, jnp.array([[max_amplitude, 0.]])


@jax.jit
def sin_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[float, ArrayLike]:
    """(P, [[A, phi]]) for ``A sin(2 pi t / P) = A cos(2 pi t / P - pi/2)``.

    The phase used to be -pi/2, which gives ``-A sin``.
    """
    return P, jnp.array([[max_amplitude, jnp.pi / 2]])


def sin2_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[float, ArrayLike]:
    """Oscillating part of ``A sin^2(2 pi t / P) = A/2 - A/2 cos(4 pi t / P)``.

    Deprecated: the constant ``A/2`` cannot be represented by these Fourier
    parameters (they have no constant term), so only ``-A/2 cos(4 pi t / P)``,
    i.e. period ``P/2``, amplitude ``A/2``, phase ``pi``, is returned. (Before
    2026-11 the phase was 0, giving the opposite sign.)
    """
    import warnings
    warnings.warn("sin2_as_fourier_coords is deprecated and drops the constant A/2 term; "
                  "it will be removed in a future release.", DeprecationWarning, stacklevel=2)
    return 0.5 * P, jnp.array([[0.5 * max_amplitude, jnp.pi]])
