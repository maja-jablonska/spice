
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from typing import Tuple
import warnings

from .mesh_generation import face_center
from spice.utils import ExperimentalWarning

def vertex_to_polar(v: ArrayLike) -> ArrayLike:
    v += 1e-5
    r = jnp.sqrt(v[0]**2+v[1]**2+v[2]**2)+1e-5
    return jnp.nan_to_num(
        jnp.array([
            jnp.arctan2(v[2], r),
            jnp.arctan2(v[1], v[0])
        ])
    )
    
@jax.jit
def spherical_harmonic(m, n, polar_coordinates):
    m_array = (m*jnp.ones_like(polar_coordinates[:, 0])).astype(int)
    n_array = (n*jnp.ones_like(polar_coordinates[:, 1])).astype(int)
    return jax.scipy.special.sph_harm(m_array,
                                      n_array,
                                      polar_coordinates[:, 0],
                                      polar_coordinates[:, 1],
                                      n_max=10).real
    
def evaluate_fourier_for_value(d0: float, P: float, d: ArrayLike, phi: ArrayLike, timestamp: float) -> ArrayLike:
    """
    Args:
        d0 (float): amplitude D_0
        P (float): period P
        d (ArrayLike): amplitudes [1, n]
        phi (ArrayLike): phases [1, n]
        timestamp (float): timestamps

    Returns:
        ArrayLike: values
    """
    n = jnp.arange(1, d.shape[0]+1)
    return d0 + jnp.sum(d*jnp.cos(n/P*timestamp-phi))

def evaluate_fourier_prim_for_value(d0: float, P: float, d: ArrayLike, phi: ArrayLike, timestamp: float) -> ArrayLike:
    """
    Args:
        d0 (float): amplitude D_0
        P (float): period P
        d (ArrayLike): amplitudes [1, n]
        phi (ArrayLike): phases [1, n]
        timestamp (float): timestamps

    Returns:
        ArrayLike: values
    """
    n = jnp.arange(1, d.shape[0]+1)
    return jnp.sum(-d*n/P*jnp.sin(n/P*timestamp-phi))

evaluate_many_fouriers_for_value = jax.jit(jax.vmap(evaluate_fourier_for_value, in_axes=(0, 0, 0, 0, None)))
evaluate_many_fouriers_prim_for_value = jax.jit(jax.vmap(evaluate_fourier_prim_for_value, in_axes=(0, 0, 0, 0, None)))

@jax.jit
def mesh_polar_vertices(vertices: ArrayLike) -> ArrayLike:
    return (jax.vmap(vertex_to_polar, in_axes=0)(vertices))


@jax.jit
def rotation_matrix(rotation_axis: ArrayLike) -> ArrayLike:
    a_norm = rotation_axis/jnp.linalg.norm(rotation_axis)
    a_hat = jnp.array([[0., -a_norm[2], a_norm[1]],
                      [a_norm[2], 0., -a_norm[0]],
                      [-a_norm[1], a_norm[0], 0.]])
    return a_hat
    

@jax.jit
def evaluate_rotation_matrix(rotation_matrix: ArrayLike, theta: ArrayLike) -> ArrayLike:
    return jnp.eye(3) + jnp.sin(theta)*rotation_matrix + (1-jnp.cos(theta))*jnp.matmul(rotation_matrix, rotation_matrix)


@jax.jit
def rotation_matrix_prim(rotation_axis: ArrayLike) -> ArrayLike:
    a_norm = rotation_axis/jnp.linalg.norm(rotation_axis)
    a_hat = jnp.array([[0., -a_norm[2], a_norm[1]],
                      [a_norm[2], 0., -a_norm[0]],
                      [-a_norm[1], a_norm[0], 0.]])
    return a_hat


@jax.jit
def evaluate_rotation_matrix_prim(rotation_matrix_grad: ArrayLike, theta: ArrayLike) -> ArrayLike:
    return jnp.cos(theta)*rotation_matrix_grad + jnp.sin(theta)*jnp.matmul(rotation_matrix_grad, rotation_matrix_grad)


@jax.jit
def cast_to_los(vectors: ArrayLike, los_vector: ArrayLike) -> ArrayLike:
    """Cast 3D vectors to the line-of-sight

    Args:
        vectors (ArrayLike): Properties to be casted (n, 3)
        los_vector (ArrayLike): LOS vector (3,)

    Returns:
        ArrayLike: Casted vectors (n, 1)
    """
    return -1.*jnp.dot(vectors, los_vector)


@jax.jit
def cast_to_normal_plane(vectors: ArrayLike, normal_vector: ArrayLike) -> ArrayLike:
    """Cast 3D vectors to a 2D plane determined by a normal vector

    Args:
        vectors (ArrayLike): Properties to be casted (n, 3)
        normal_vector (ArrayLike): Normal vector (3,)

    Returns:
        ArrayLike: Casted vectors (n, 3), where one of the axes contains only zeroes
    """
    n_normal_vector = normal_vector/jnp.linalg.norm(normal_vector)
    cast_onto_n = jnp.dot(vectors, n_normal_vector).reshape((-1, 1))
    return vectors-cast_onto_n*n_normal_vector


@jax.jit
def cast_normalized_to_los(vectors: ArrayLike, los_vector: ArrayLike) -> ArrayLike:
    """Cast 3D vectors to the line-of-sight

    Args:
        vectors (ArrayLike): Properties to be casted (n, 3)
        los_vector (ArrayLike): LOS vector (3,)

    Returns:
        ArrayLike: Casted vectors (n, 1)
    """
    last_axis = len(vectors.shape)-1
    return -1.*jnp.dot(vectors/(jnp.linalg.norm(vectors, axis=last_axis, keepdims=True)+1e-10),
                       los_vector)


@jax.jit
def calculate_axis_radii(centers: ArrayLike, axis: ArrayLike) -> ArrayLike:
    norm_axis = len(centers.shape)-1
    return jnp.linalg.norm(jnp.cross(axis, -centers),
                           axis=norm_axis)/jnp.linalg.norm(axis)

### UTILITY FUNCTIONS FOR FOURIER SERIES

@jax.jit
def cos_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[ArrayLike, ArrayLike]:
    return jnp.array([0., P]), jnp.array([[max_amplitude, 0.]])

@jax.jit
def sin_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[ArrayLike, ArrayLike]:
    return jnp.array([0., P]), jnp.array([[max_amplitude, -jnp.pi/2]])

@jax.jit
def sin2_as_fourier_coords(P: float, max_amplitude: float) -> Tuple[ArrayLike, ArrayLike]:
    return jnp.array([0.5, 0.5*P]), jnp.array([[0.5*max_amplitude, 0.]])

@jax.jit
def sinh_as_fourier_coords(P: float, max_amplitude: float, n: int) -> Tuple[ArrayLike, ArrayLike]:
    return jnp.array([0., P]), jnp.array([[2*jnp.sinh(jnp.pi)/jnp.pi*jnp.power(-1, n0+1)*n0/(jnp.power(n0, 2)+1)*max_amplitude, 0.] for n0 in range(1, n+1)])
