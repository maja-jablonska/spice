from typing import Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_model import MeshModel, MAX_PULSATION_MODES, DEFAULT_ROTATION_AXIS
from .spots import v_spot
from .utils import (apply_spherical_harm_pulsation, cast_to_los,
                    rotation_matrix, rotation_matrix_prim,
                    evaluate_rotation_matrix, evaluate_rotation_matrix_prim,
                    calculate_axis_radii)
import warnings
from functools import partial


@jax.jit
def add_pulsation(mesh: MeshModel,
                  m: int, n: int, t_0: float, period: float, amplitude: float) -> MeshModel:
    """Generate a mesh with a pulsation applied

    Args:
        mesh (MeshModel): Mesh to add a pulsation to
        m (int): m-mode
        n (int): n-mode
        t_0 (float): time of phase = 0
        period (float): pulsation period in days
        amplitude (float): amplitude in stellar radii

    Returns:
        MeshModel: Mesh with the pulsation added
    """

    def do_nothing():
        warnings.warn("Pulsation modes exceeded (max: 20). No pulsation applied")
        return mesh

    def update_mesh():
        nonzero_ind = jnp.count_nonzero(mesh.amplitudes)
        return mesh._replace(pulsation_ms = mesh.pulsation_ms.at[nonzero_ind].set(m),
                            pulsation_ns = mesh.pulsation_ns.at[nonzero_ind].set(n),
                            t_zeros = mesh.t_zeros.at[nonzero_ind].set(t_0),
                            pulsation_periods = mesh.pulsation_periods.at[nonzero_ind].set(period),
                            amplitudes = mesh.amplitudes.at[nonzero_ind].set(amplitude))

    return jax.lax.cond(mesh.pulsation_modes() >= MAX_PULSATION_MODES,
                        do_nothing,
                        update_mesh)


@jax.jit
def add_rotation(mesh: MeshModel,
                 rotation_velocity: ArrayLike,
                 rotation_axis: ArrayLike = DEFAULT_ROTATION_AXIS):
    rot_matrix = rotation_matrix(rotation_axis)
    rot_matrix_grad = rotation_matrix_prim(rotation_axis)
    return mesh._replace(rotation_axis = rotation_axis,
                         rotation_matrix = rot_matrix,
                         rotation_matrix_prim = rot_matrix_grad,
                         rotation_velocity = rotation_velocity)


@jax.jit
def evaluate_rotation(mesh: MeshModel, t: ArrayLike):
    theta = (mesh.rotation_velocity*t)/mesh.radius # cm
    t_rotation_matrix = evaluate_rotation_matrix(mesh.rotation_matrix, theta) # cm
    t_rotation_matrix_prim = evaluate_rotation_matrix_prim(mesh.rotation_matrix_prim, theta) # cm
    rotated_vertices = jnp.matmul(mesh.vertices, t_rotation_matrix) # cm
    rotated_centers = jnp.matmul(mesh.centers, t_rotation_matrix) # cm
    rotated_centers_vel = mesh.rotation_velocity*jnp.matmul(mesh.centers/mesh.radius, t_rotation_matrix_prim) # cm

    new_axis_radii = calculate_axis_radii(rotated_centers, mesh.rotation_axis)
    return mesh._replace(vertices = rotated_vertices,
                         centers = rotated_centers,
                         velocities = rotated_centers_vel,
                         los_velocities = cast_to_los(rotated_centers_vel, mesh.los_vector),
                         axis_radii = new_axis_radii)


@jax.jit
def __evaluate_one_pulsation(mesh: MeshModel,
                             m: ArrayLike, n: ArrayLike,
                             amplification: ArrayLike, velocity: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    (vert_offsets, center_offsets,
     area_offsets, sph_ham_values) = apply_spherical_harm_pulsation(mesh.vertices,
                                                                    mesh.centers,
                                                                    mesh.faces,
                                                                    mesh.areas,
                                                                    amplification,
                                                                    m, n)
    return (jnp.nan_to_num(vert_offsets),
            jnp.nan_to_num(center_offsets),
            jnp.nan_to_num(area_offsets),
            jnp.nan_to_num((mesh.centers+center_offsets)*velocity*sph_ham_values*mesh.radius))


__vec_evaluate_pulsations = jax.jit(jax.vmap(__evaluate_one_pulsation, in_axes=(None, 0, 0, 0, 0)))


@jax.jit
def evaluate_pulsation(mesh: MeshModel, t: float) -> MeshModel:
    phases = jnp.nan_to_num((t-mesh.t_zeros)/(mesh.pulsation_periods))
    amplifications = mesh.amplitudes*jnp.sin(phases)
    vel_amplifications = mesh.amplitudes*jnp.cos(phases)
    vert_offsets, center_offsets, area_offsets, velocities = __vec_evaluate_pulsations(mesh,
                                                                                       mesh.pulsation_ms,
                                                                                       mesh.pulsation_ns,
                                                                                       amplifications,
                                                                                       vel_amplifications)
    new_centers = mesh.centers + jnp.sum(center_offsets, axis=0)
    new_velocities = mesh.velocities + jnp.sum(velocities, axis=0)
    
    return mesh._replace(vertices = mesh.vertices + jnp.sum(vert_offsets, axis=0),
                         centers = new_centers,
                         areas = mesh.areas + jnp.sum(area_offsets, axis=0),
                         velocities = new_velocities,
                         mus = cast_to_los(new_centers, mesh.los_vector),
                         los_velocities = cast_to_los(new_velocities[:, jnp.newaxis], mesh.los_vector))
