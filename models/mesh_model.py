# from abc import ABCMeta, abstractmethod, abstractproperty
# from dataclasses import dataclass
# import jax
# import jax.numpy as jnp
# import jaxlib

# from .mesh_generation import vertex_to_polar, icosphere, calculate_rotation, apply_spherical_harm_pulsation
# from overrides import overrides
# from spectrum import spectrum_flash_sum, get_spectra_flash_sum
# import phoebe
# import numpy as np
# from phoebe import u
# from typing import Dict, Optional, Union
# from enum import auto, Enum
# import astropy.units as un
# from jax.numpy import interp
# from functools import partial
# from collections import namedtuple

from typing import Union
from jax.typing import ArrayLike
import jax.numpy as jnp

from abc import ABCMeta, abstractmethod
from typing import NamedTuple

from overrides import overrides

from .mesh_generation import icosphere
from .utils import calculate_axis_radii, cast_to_los


DEFAULT_LOS_VECTOR: jnp.ndarray = jnp.array([0., 1., 0.]) # from the Y direction
DEFAULT_ROTATION_AXIS = jnp.ndarray = jnp.array([0., 0., 1.]) # from the Y direction

NO_ROTATION_MATRIX = jnp.zeros((3, 3))
MAX_PULSATION_MODES = 20
NO_PULSATION_ARRAYS = jnp.zeros(MAX_PULSATION_MODES)

# vec_apply_spherical_harm_pulsation = jax.jit(jax.vmap(apply_spherical_harm_pulsation,
#                                                       in_axes = (0, 0, 0, 0, 0, None, None)))

# def __cos_law(lat1, lon1, lat2, lon2):
#     """
#     Calculate the great circle angular distance between two points
#     on the sphere (specified in radians)

#     All args must be of equal length.    

#     """
#     lat1 = jnp.pi/2 - lat1
#     lat2 = jnp.pi/2 - lat2
#     a = jnp.sin(lat1)*jnp.sin(lat2) + jnp.cos(lat1) * jnp.cos(lat2) * jnp.cos(lon2 - lon1)

#     return jnp.arccos(a)


# def __overabundance_spot_params(theta, phi, abun, abun_bg, radius, radius_factor, coord):
#     c_th, c_ph = vertex_to_polar(coord)
#     dist_rad = __cos_law(theta, phi, c_th-jnp.pi/2, c_ph-jnp.pi)
#     angles = jnp.array([0,radius*radius_factor,radius,jnp.pi])
#     abuns = jnp.array([abun, abun, abun_bg, abun_bg])
#     y = interp(dist_rad, angles, abuns, abun, abun_bg)
#     return y


# v_spot = jax.vmap(__overabundance_spot_params, in_axes=(None, None, None, None, None, None, 0))


def inclination_to_axis_vector(inclination: ArrayLike) -> jnp.array:
    if inclination.shape==(1,):
        los_vector = jnp.array([jnp.sin(jnp.deg2rad(inclination[0])),
                                jnp.cos(jnp.deg2rad(inclination[0])), 0.])
    elif inclination.shape==(3,):
        los_vector = inclination/jnp.linalg.norm(inclination)
    else:
        raise ValueError('''Inclination has to be a either a float value, 1D array 
        with inclination value, or rotation axis as a 3D array''')

        
# MeshModel = namedtuple('MeshModel', ['timestamps',
#                                      'los_vector',
#                                      'radius',
#                                      'mass',
#                                      'teffs',
#                                      'logg',
#                                      'vturb',
#                                      'abundances',
#                                      'vertices',
#                                      'centers',
#                                      'faces',
#                                      'areas',
#                                      'mus',
#                                      'velocities',
#                                      'los_velocities'])


class MeshModel(NamedTuple):
    # Stellar properties
    radius: float
    mass: float
    abs_luminosity: float

    # Mesh properties
    vertices: ArrayLike
    faces: ArrayLike
    centers: ArrayLike
    areas: ArrayLike

    parameters: ArrayLike

    # Motion properties
    velocities: ArrayLike
    
    # Rotation
    rotation_axis: ArrayLike
    rotation_matrix: ArrayLike
    rotation_matrix_prim: ArrayLike
    axis_radii: ArrayLike
    rotation_velocity: ArrayLike

    # Pulsation
    pulsation_ms: ArrayLike
    pulsation_ns: ArrayLike
    t_zeros: ArrayLike
    pulsation_periods: ArrayLike
    amplitudes: ArrayLike

    # Mesh LOS properties
    los_vector: ArrayLike
    mus: ArrayLike
    los_velocities: ArrayLike

    @abstractmethod
    def pulsation_modes(self) -> int:
        raise NotImplementedError()


class IcosphereModel(MeshModel):
    # TODO: show this instead of MeshModel initializer
    @classmethod
    def construct(cls, n_vertices: int,
                  radius: float, mass: float,
                  abs_luminosity: float,
                  parameters: ArrayLike): # What to do about parameters?
        """Construct an Icosphere.

        Args:
            n_vertices (int): Minimal number of vertices (used to calculate number of divisions)
            radius (float): Radius in solar radii
            mass (float): Mass in solar masses
            abs_luminosity (float): Absolute luminosity in solar luminosities
            parameters (ArrayLike): Array of global parameters

        Returns:
            _type_: _description_
        """
        vertices, faces, areas, centers = icosphere(n_vertices)

        return MeshModel.__new__(cls, radius, mass, abs_luminosity,
                vertices, faces, centers, areas, parameters,
                jnp.zeros_like(centers),
                DEFAULT_ROTATION_AXIS, NO_ROTATION_MATRIX, NO_ROTATION_MATRIX, calculate_axis_radii(centers, DEFAULT_ROTATION_AXIS), 0.,
                NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS,
                DEFAULT_LOS_VECTOR, cast_to_los(centers, DEFAULT_LOS_VECTOR), jnp.zeros_like(areas))
    
    @overrides
    def pulsation_modes(self) -> int:
        return jnp.count_nonzero(self.amplitudes)


# def mesh_model(n_vertices: int,
#                timestamps: jnp.ndarray,
#                radius: float = 1.0,
#                mass: float = 1.0,
#                teff: float = SOLAR_TEFF,
#                logg: float = SOLAR_LOGG,
#                vturb: float = SOLAR_VTURB,
#                abundances: jnp.ndarray = SOLAR_ABUNDANCES,
#                los_vector: jnp.ndarray = DEFAULT_LOS_VECTOR) -> MeshModel:
#     verts, faces, areas, centers, _ = icosphere(n_vertices)
#     teffs = jnp.ones_like(areas)*teff
#     abundances = jnp.repeat(abundances.reshape((1, -1)), areas.shape[0], axis=0)
#     vertices = jnp.repeat(verts[jnp.newaxis, :, :], timestamps.shape[0], axis=0)
#     faces = jnp.repeat(faces[jnp.newaxis, :, :], timestamps.shape[0], axis=0)
#     areas = jnp.repeat(areas[jnp.newaxis, :], timestamps.shape[0], axis=0)
#     centers = jnp.repeat(centers[jnp.newaxis, :, :], timestamps.shape[0], axis=0)
#     mus = -1.*jnp.dot(centers/
#                       (jnp.linalg.norm(centers, axis=2, keepdims=True)+1e-10),
#                       los_vector)
#     velocities = jnp.zeros_like(centers)
#     los_velocities = jnp.zeros_like(areas)
#     return MeshModel(timestamps=timestamps,
#                      los_vector=los_vector,
#                      radius=radius,
#                      mass=mass,
#                      teffs=teffs,
#                      logg=logg,
#                      vturb=vturb,
#                      abundances=abundances,
#                      vertices=vertices,
#                      centers=centers,
#                      faces=faces,
#                      areas=areas,
#                      mus=mus,
#                      velocities=velocities,
#                      los_velocities=los_velocities[:, :, jnp.newaxis])

# @jax.jit
# def pulsation(mesh_model: Dict[str, list],
#               m: float, n: float,
#               magnitude: float,
#               t0: float, period: float) -> MeshModel:
#     phases = (mesh_model['timestamps'][:, jnp.newaxis]-t0)/period
#     amplifications = magnitude*jnp.sin(phases)
#     velocities = magnitude*jnp.cos(phases)
#     vert_offsets, center_offsets, area_offsets, sph_ham = vec_apply_spherical_harm_pulsation(mesh_model['vertices'],
#                                                                                              mesh_model['centers'],
#                                                                                              mesh_model['faces'],
#                                                                                              mesh_model['areas'],
#                                                                                              amplifications, m, n)
#     puls_velocities = mesh_model['velocities']+mesh_model['centers']*velocities[:, jnp.newaxis]*sph_ham
#     return MeshModel(timestamps=mesh_model['timestamps'],
#                      los_vector=mesh_model['los_vector'],
#                      radius=mesh_model['radius'],
#                      mass=mesh_model['mass'],
#                      teffs=mesh_model['teffs'],
#                      logg=mesh_model['logg'],
#                      vturb=mesh_model['vturb'],
#                      abundances=mesh_model['abundances'],
#                      vertices=mesh_model['vertices']+vert_offsets,
#                      centers=mesh_model['centers']+center_offsets,
#                      faces=mesh_model['faces'],
#                      areas=mesh_model['areas']+area_offsets,
#                      mus=mesh_model['mus'],
#                      velocities=puls_velocities,
#                      los_velocities=-1.*jnp.dot(puls_velocities, mesh_model['los_vector']))

# @jax.jit
# def rotation(mesh_model: Dict[str, list],
#              period: Optional[float] = None,
#              rotation_velocity: Optional[float] = None,
#              rotation_axis: jnp.array = DEFAULT_ROTATION_AXIS) -> MeshModel:
#     if period is None and rotation_velocity is None:
#         raise ValueError('Either period or rotation_speed has to be provided.')
#     elif period is not None and rotation_velocity is not None:
#         raise ValueError('Only one of period and rotation_speed has to be provided.')
#     elif period is not None:
#         # TODO: przeliczyć radius na km
#         omega = 2*jnp.pi/period # tu na s
#         rotation_velocity = omega*mesh_model['radius']
#     else:
#         omega = rotation_velocity/mesh_model['radius']
    
#     rot_vertices_offsets, rot_center_offsets, rotation_velocities, radii = calculate_rotation(omega,
#                                                                                               rotation_axis,
#                                                                                               mesh_model['vertices'],
#                                                                                               mesh_model['centers'],
#                                                                                               mesh_model['timestamps'])
#     new_centers = mesh_model['centers']+rot_center_offsets
#     mus = -1.*jnp.dot(new_centers/
#                   (jnp.linalg.norm(new_centers, axis=2, keepdims=True)+1e-10),
#                   mesh_model['los_vector'])
#     rot_los_vels = (jnp.dot(rotation_velocities/(
#             jnp.linalg.norm(rotation_velocities, axis=2, keepdims=True)+1e-10),
#         mesh_model['los_vector'])*radii*rotation_velocity)[:, :, jnp.newaxis]
#     return MeshModel(timestamps=mesh_model['timestamps'],
#                      los_vector=mesh_model['los_vector'],
#                      radius=mesh_model['radius'],
#                      mass=mesh_model['mass'],
#                      teffs=mesh_model['teffs'],
#                      logg=mesh_model['logg'],
#                      vturb=mesh_model['vturb'],
#                      abundances=mesh_model['abundances'],
#                      vertices=mesh_model['vertices']+rot_vertices_offsets,
#                      centers=new_centers,
#                      faces=mesh_model['faces'],
#                      areas=mesh_model['areas'],
#                      mus=mus,
#                      velocities=mesh_model['velocities']+rotation_velocities,
#                      los_velocities=mesh_model['los_velocities'].reshape((len(mesh_model['timestamps']),
#                                                                        -1, 1))+rot_los_vels)

# @jax.jit
# def teff_spot(mesh_model: Dict[str, list],
#               center_theta: float, center_phi: float,
#               teff_difference: float,
#               radius: float, radius_factor: float) -> MeshModel:
#     teffs = v_spot(center_theta, center_phi, teff_difference, 0., radius, radius_factor, mesh_model['centers'][0])
#     return MeshModel(timestamps=mesh_model['timestamps'],
#                      los_vector=mesh_model['los_vector'],
#                      radius=mesh_model['radius'],
#                      mass=mesh_model['mass'],
#                      teffs=mesh_model['teffs']+teffs,
#                      logg=mesh_model['logg'],
#                      vturb=mesh_model['vturb'],
#                      abundances=mesh_model['abundances'],
#                      vertices=mesh_model['vertices'],
#                      centers=mesh_model['centers'],
#                      faces=mesh_model['faces'],
#                      areas=mesh_model['areas'],
#                      mus=mesh_model['mus'],
#                      velocities=mesh_model['velocities'],
#                      los_velocities=mesh_model['los_velocities'])

# @jax.jit
# def abundance_spot(mesh_model: Dict[str, list],
#                    center_theta: float, center_phi: float,
#                    abundance_difference: float,
#                    radius: float, radius_factor: float,
#                    abundance_indx: int) -> MeshModel:
#     abuns = v_spot(center_theta, center_phi, abundance_difference, 0., radius, radius_factor, mesh_model['centers'][0])
#     return MeshModel(timestamps=mesh_model['timestamps'],
#                      los_vector=mesh_model['los_vector'],
#                      radius=mesh_model['radius'],
#                      mass=mesh_model['mass'],
#                      teffs=mesh_model['teffs'],
#                      logg=mesh_model['logg'],
#                      vturb=mesh_model['vturb'],
#                      abundances=mesh_model['abundances'].at[:, abundance_indx].set(
#                          mesh_model['abundances'][:, abundance_indx]+abuns),
#                      vertices=mesh_model['vertices'],
#                      centers=mesh_model['centers'],
#                      faces=mesh_model['faces'],
#                      areas=mesh_model['areas'],
#                      mus=mesh_model['mus'],
#                      velocities=mesh_model['velocities'],
#                      los_velocities=mesh_model['los_velocities'])

# @partial(jax.jit, static_argnums=(1, 3))
# def model_spectrum(mesh_model: MeshModel, time_index: int,
#                    log_wavelengths: jnp.ndarray, chunk_size: int = 256):
#     mus = mesh_model['mus'][time_index]
#     mus = jnp.where(mus>0, mus, 0.)
#     _teffs = mesh_model['teffs'][:, jnp.newaxis]
#     parameters = jnp.concatenate([_teffs,
#                                   jnp.ones_like(_teffs)*mesh_model['logg'],
#                                   jnp.ones_like(_teffs)*mesh_model['vturb'],
#                                   mesh_model['abundances']],
#                                  axis=1)
#     return spectrum_flash_sum(log_wavelengths,
#                               (mus*mesh_model['areas'][time_index])[:, jnp.newaxis],
#                               mus,
#                               mesh_model['los_velocities'][time_index],
#                               parameters,
#                               chunk_size)
