
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from typing import Tuple
import warnings

from .mesh_generation import face_center
from utils import ExperimentalWarning

@jax.jit
def apply_pulsation(verts: ArrayLike,
                    faces: ArrayLike,
                    magnitude: float) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Apply the pulsation 

    Args:
        verts (ArrayLike): _description_
        faces (ArrayLike): _description_
        magnitude (float): _description_

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: _description_
    """
    warnings.warn("This feature is experimental - use with caution.", ExperimentalWarning)
    direction_vectors = verts/jnp.linalg.norm(verts, axis=1).reshape((-1, 1))
    verts = verts + magnitude*direction_vectors
    areas, centers = jax.jit(jax.vmap(face_center, in_axes=(None, 0)))(verts, faces.astype(jnp.int32))
    mus = jnp.dot(centers/jnp.linalg.norm(centers, axis=1).reshape((-1, 1)), jnp.array([0, 0, 1]))
    return verts, faces, areas, centers, mus

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
