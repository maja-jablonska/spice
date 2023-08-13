
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from typing import Tuple
from functools import partial

from .mesh_generation import icosphere, face_center

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
def spherical_harmonic(m: float, n: float, polar_coordinates: ArrayLike) -> ArrayLike:
    m_array = (m*jnp.ones_like(polar_coordinates[:, 0])).astype(int)
    n_array = (n*jnp.ones_like(polar_coordinates[:, 1])).astype(int)
    return jax.scipy.special.sph_harm(m_array,
                                      n_array,
                                      polar_coordinates[:, 0],
                                      polar_coordinates[:, 1],
                                      n_max=10)


@jax.jit
def apply_spherical_harm_pulsation(verts: ArrayLike,
                                   centers: ArrayLike,
                                   faces: ArrayLike,
                                   areas: ArrayLike,
                                   magnitude: float,
                                   m: float, n: float) -> Tuple[ArrayLike,
                                                                ArrayLike,
                                                                ArrayLike,
                                                                ArrayLike]:
    #checkify.check(m<=n, "m has to be lesser or equal n")
    vert_axis = len(verts.shape)-1
    
    direction_vectors = verts/jnp.linalg.norm(verts, axis=vert_axis, keepdims=True)
    
    polar_coordinates = jnp.nan_to_num(mesh_polar_vertices(verts))
    polar_coordinates_centers = jnp.nan_to_num(mesh_polar_vertices(centers))
    
    sph_ham = spherical_harmonic(m, n, polar_coordinates).real
    sph_ham_centers = spherical_harmonic(m, n, polar_coordinates_centers).real
    
    magnitudes = magnitude*sph_ham
    
    vert_offsets = magnitudes.reshape((-1, 1))*direction_vectors
    
    new_areas, new_centers = jax.jit(jax.vmap(face_center, in_axes=(None, 0)))(verts+vert_offsets, faces.astype(jnp.int32))
    
    return vert_offsets, new_centers-centers, new_areas-areas, sph_ham_centers[:, jnp.newaxis]


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
