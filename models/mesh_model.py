from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from .mesh_generation import icosphere, calculate_rotation, apply_spherical_harm_pulsation
from overrides import overrides
from spectrum import spectrum_flash_sum, get_spectra_flash_sum
import phoebe
import numpy as np
from phoebe import u
from typing import Optional, Union
from enum import auto, Enum
import astropy.units as un


DEFAULT_LOS_VECTOR: jnp.ndarray = jnp.array([1., 0., 0.])
    
vec_apply_spherical_harm_pulsation = jax.jit(jax.vmap(apply_spherical_harm_pulsation,
                                                      in_axes = (0, 0, 0, 0, 0, None, None)))
    

def inclination_to_axis_vector(inclination: Union[float, jnp.array]) -> jnp.array:
    if isinstance(inclination, float):
        los_vector = jnp.array([jnp.sin(jnp.deg2rad(inclination)),
                                jnp.cos(jnp.deg2rad(inclination)), 0.])
    elif isinstance(inclination, jnp.ndarray):
        if inclination.shape==(1,):
            los_vector = jnp.array([jnp.sin(jnp.deg2rad(inclination[0])),
                                    jnp.cos(jnp.deg2rad(inclination[0])), 0.])
        elif inclination.shape==(3,):
            los_vector = inclination/jnp.linalg.norm(inclination)
        else:
            raise ValueError('''Inclination has to be a either a float value, 1D array 
            with inclination value, or rotation axis as a 3D array''')
    else:
        raise ValueError('''Inclination has to be a either a float value, 1D array 
        with inclination value, or rotation axis as a 3D array''')
    return los_vector


class StarModel(metaclass=ABCMeta):
    def __init__(self, radius: float, timestamps: jnp.array):
        self.__radius = radius
        self.__timestamps = timestamps

    @property
    def radius(self):
        return self.__radius
    
    @property
    def timestamps(self) -> jnp.ndarray:
        return self.__timestamps
    
    @abstractproperty
    def areas(self) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractproperty
    def centers(self) -> jnp.ndarray:
        raise NotImplementedError

    @abstractproperty
    def rotation_velocity(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def apply_rotation(self, period: float, inclination: Union[float, jnp.ndarray]) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def apply_pulsation(self, m: float, n: float, magnitude: float, t0: float, period: float) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_mus(self) -> jnp.ndarray:
        raise NotImplementedError
        
    @abstractmethod
    def get_mus_for_time(self, time_index: int) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_los_velocities(self) -> jnp.ndarray:
        raise NotImplementedError
        
    @abstractmethod
    def get_los_velocities_for_time(self, time_index: int) -> jnp.ndarray:
        raise NotImplementedError
        
    @abstractmethod
    def model_spectrum(self,
                       time_index: int,
                       log_wavelengths: jnp.ndarray,
                       parameters: jnp.ndarray,
                       chunk_size: int) -> jnp.ndarray:
        raise NotImplementedError
        
    @abstractmethod
    def model_spectra(self,
                      log_wavelengths: jnp.ndarray,
                      parameters: jnp.ndarray,
                      chunk_size: int) -> jnp.ndarray:
        raise NotImplementedError
    

class MeshModel(StarModel):
    def __init__(self, n_vertices: int, radius: float, timestamps: jnp.ndarray):
        super().__init__(radius, timestamps)
        
        verts, faces, areas, centers, _ = icosphere(n_vertices)
        self.__verts = jnp.repeat(verts[jnp.newaxis, :, :], self.timestamps.shape[0], axis=0)
        self.__faces = jnp.repeat(faces[jnp.newaxis, :, :], self.timestamps.shape[0], axis=0)
        self.__areas = jnp.repeat(areas[jnp.newaxis, :], self.timestamps.shape[0], axis=0)
        self.__centers = jnp.repeat(centers[jnp.newaxis, :, :], self.timestamps.shape[0], axis=0)
        
        self.__vert_offsets = jnp.zeros_like(self.__verts)
        self.__face_offsets = jnp.zeros_like(self.__faces)
        self.__area_offsets = jnp.zeros_like(self.__areas)
        self.__center_offsets = jnp.zeros_like(self.__centers)

        self.__radii = jnp.zeros_like(self.__areas)
        
        self.__omega = 0.0
        self.__rotation_velocity = 0.0
        self.__rotation_axis = jnp.array([0., 1., 0.])
        self.__pulsation_velocities = jnp.zeros_like(self.__centers)
        self.__rotation_velocities = jnp.zeros_like(self.__centers)
        
        self.__los_vector = DEFAULT_LOS_VECTOR
    
    @property
    def vertices(self) -> jnp.ndarray:
        return self.__verts+self.__vert_offsets
    
    @property
    def faces(self) -> jnp.ndarray:
        return self.__faces+self.__face_offsets
    
    @faces.setter
    def faces(self, faces: jnp.ndarray):
        self.__faces = faces
    
    @property
    def areas(self) -> jnp.ndarray:
        return self.__areas+self.__area_offsets
    
    @areas.setter
    def areas(self, areas: jnp.ndarray):
        self.__areas = areas
    
    @property
    def centers(self) -> jnp.ndarray:
        return self.__centers+self.__center_offsets
    
    @centers.setter
    def centers(self, centers: jnp.ndarray):
        self.__centers = centers
        
    @property
    def rotation_axis(self) -> jnp.ndarray:
        return self.__rotation_axis
    
    @property
    def rotation_velocity(self) -> float:
        return self.__rotation_velocity
    
    @property
    def los_vector(self) -> jnp.array:
        return self.__los_vector
    
    @los_vector.setter
    def los_vector(self, new_los_vector: jnp.array):
        self.__los_vector = new_los_vector
        
    @property
    def velocities(self) -> jnp.ndarray:
        return self.__pulsation_velocities+self.__rotation_velocities
    
    @overrides
    def get_mus(self) -> jnp.ndarray:
        return -1.*jnp.dot(self.centers/
                           (jnp.linalg.norm(self.centers, axis=2, keepdims=True)+1e-10),
                           self.los_vector)
    
    @overrides
    def get_mus_for_time(self, time_index: int) -> jnp.ndarray:
        return -1.*jnp.dot(self.centers[time_index]/
                           (jnp.linalg.norm(self.centers[time_index],
                                            axis=1, keepdims=True)+1e-10),
                           self.los_vector)
    
    def reset_to_mesh(self):
        self.__vert_offsets = jnp.zeros_like(self.__verts)
        self.__face_offsets = jnp.zeros_like(self.__faces)
        self.__area_offsets = jnp.zeros_like(self.__areas)
        self.__center_offsets = jnp.zeros_like(self.__centers)
        self.__pulsation_velocities = jnp.zeros_like(self.__centers)
        self.__rotation_velocities = jnp.zeros_like(self.__centers)
    
    @overrides
    def apply_pulsation(self, m: float, n: float, magnitude: float, t0: float, period: float) -> jnp.ndarray:
        # Each pulsation is represented by:
        # m, n, max_amplification, t0 (time of phase 0), period
        
        phases = 2*jnp.pi*(self.timestamps[:, jnp.newaxis]-t0)/period
        amplifications = magnitude*jnp.sin(phases)
        velocities = magnitude*jnp.cos(phases)
        
        # return vert_offsets, new_centers-centers, new_areas-areas
        
        # Shape
        # (timesteps, n_pulsations, n_vertices, 3)
        vert_offsets, center_offsets, area_offsets, sph_ham = vec_apply_spherical_harm_pulsation(self.vertices,
                                                                                    self.centers,
                                                                                    self.faces,
                                                                                    self.areas,
                                                                                    amplifications,
                                                                                    m, n)
        self.__center_offsets = self.__center_offsets+center_offsets
        self.__vert_offsets = self.__vert_offsets+vert_offsets
        self.__area_offsets = self.__area_offsets+area_offsets
        self.__pulsation_velocities = self.__pulsation_velocities + velocities[:, jnp.newaxis]*sph_ham

        return jnp.vstack([self.centers, self.velocities])

    @overrides
    def apply_rotation(self, period: float,
                       inclination: Union[float, jnp.ndarray] = jnp.array([0., 1., 0.])) -> jnp.ndarray:
        los_vector = inclination_to_axis_vector(inclination)
        self.__omega = 2*jnp.pi/period
        self.__rotation_velocity = self.__omega*self.radius

        self.__los_vector = inclination_to_axis_vector(90.-inclination)
        self.__center_offsets, self.__rotation_velocities, self.__radii = calculate_rotation(self.__omega,
                                                                                             self.__rotation_axis,
                                                                                             self.__centers,
                                                                                             self.timestamps)
        return jnp.vstack([self.centers, self.velocities])
    
    @overrides
    def get_los_velocities(self) -> jnp.ndarray:
        los_vels = jnp.dot(self.velocities/(
            jnp.linalg.norm(self.velocities, axis=2, keepdims=True)+1e-10
        ), self.los_vector)
        return self.__rotation_velocity*los_vels*self.__radii
    
    @overrides
    def get_los_velocities_for_time(self, time_index: int) -> jnp.ndarray:
        los_vels = jnp.dot(self.velocities[time_index]/(
            jnp.linalg.norm(self.velocities[time_index], axis=1, keepdims=True)+1e-10
            ), self.los_vector)
        return self.__rotation_velocity*los_vels*self.__radii[time_index]
    
    @overrides
    def model_spectrum(self,
                       time_index: int,
                       log_wavelengths: jnp.ndarray,
                       parameters: jnp.ndarray,
                       chunk_size: int = 256) -> jnp.ndarray:
        mus = self.get_mus_for_time(time_index)
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities_for_time(time_index)
        
        return spectrum_flash_sum(log_wavelengths,
                                  (mus*self.areas[time_index])[:, jnp.newaxis],
                                  mus[:, jnp.newaxis],
                                  los_vels[:, jnp.newaxis],
                                  parameters,
                                  chunk_size)
    
    @overrides
    def model_spectra(self,
                      log_wavelengths: jnp.ndarray,
                      parameters: jnp.ndarray,
                      chunk_size: int = 256) -> jnp.ndarray:
        mus = self.get_mus()
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities()
        spectra_flash_sum = get_spectra_flash_sum(chunk_size)
        
        return spectra_flash_sum(log_wavelengths,
                                 (mus*self.areas)[:, :, jnp.newaxis],
                                 mus[:, :, jnp.newaxis],
                                 los_vels[:, :, jnp.newaxis],
                                 parameters)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def cos_angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

class Component(Enum):
    PRIMARY = auto()
    SECONDARY = auto()
    
    def __str__(self):
        return self.name.lower()

class PhoebeModel(StarModel):
    def __init__(self,
                 n_triangles: int,
                 mass: float,
                 radius: float,
                 teff: float,
                 timestamps: np.ndarray):
        self.__b: phoebe.frontend.bundle.Bundle = phoebe.Bundle()
        self.__b.add_component('star', component='primary',
                               mass=mass, requiv=radius, teff=teff)
        self.__b.set_hierarchy('star:primary')
        self.__dataset_name = 'mesh01'
        
        self.__b.add_dataset('mesh',
                             compute_times=list(timestamps),
                             dataset=self.__dataset_name)
        self.__b.set_value('columns', value=['teffs', 'vws', 'us', 'vs', 'ws', 'loggs',
                                             'visible_centroids', 'mus', 'areas'])
        self.__b.run_compute(irrad_method='none',
                             distortion_method='rotstar',
                             model='rotmodel', overwrite=True)
        self.__b.set_value('ntriangles', value=n_triangles)
        # TODO: how NOT to run compute two times?
        self.__b.run_compute(irrad_method='none',
                             distortion_method='rotstar',
                             model='rotmodel', overwrite=True)
        
        super().__init__(radius, self.__b.times)


        
    def __get_mesh_coordinates(self, time: float, component: Optional[Component] = None) -> jnp.ndarray:
        return jnp.vstack([self.get_parameter(time, 'us', component),
                           self.get_parameter(time, 'vs', component),
                           self.get_parameter(time, 'ws', component)])
    
    def __get_mesh_normals(self, time: float, component: Optional[Component] = None) -> np.ndarray:
        return self.get_parameter(time, 'uvw_normals', component)
        
    @property
    def mass(self) -> float:
        return self.b.get_parameter(qualifier='mass',
                                    component=Component.PRIMARY).value
    
    @property
    def teff(self) -> float:
        return self.b.get_parameter(qualifier='teff',
                                    component=Component.PRIMARY).value
    
    @property
    def dataset_name(self) -> str:
        return self.__dataset_name
        
    @property
    def centers(self) -> jnp.ndarray:
        return jnp.swapaxes(
            jnp.array([self.__get_mesh_coordinates(time, Component.PRIMARY)
                       for time in self.b.times]), 1, 2)

    @property
    def areas(self) -> jnp.ndarray:
        return jnp.array([self.get_parameter(time, 'areas', Component.PRIMARY)
                          for time in self.b.times])
    
    def get_areas_for_time(self, time_index: int) -> jnp.ndarray:
        return self.get_parameter(self.timestamps[time_index], 'areas', Component.PRIMARY)
    
    @property
    def rotation_velocity(self) -> float:
        per = self.b.get_parameter(qualifier='period',
                                   component=Component.PRIMARY).value
        rad = self.b.get_parameter(qualifier='requiv',
                                   component=Component.PRIMARY).value
        return (2*jnp.pi*(un.solRad*rad).to(un.km).value)/((un.day*per).to(un.s).value)
    
    @property
    def b(self):
        return self.__b
    
    @property
    def rotation_velocity(self) -> float:
        return 2*jnp.pi*(self.radius*un.SolRad).to(un.km).value*self.__frequency
    
    @overrides
    def get_mus(self) -> jnp.ndarray:
        return jnp.array([self.get_parameter(time, 'mus', Component.PRIMARY)
                          for time in self.timestamps])
    @overrides
    def get_mus_for_time(self, time_index: int) -> jnp.ndarray:
        return self.get_parameter(self.timestamps[time_index], 'mus', Component.PRIMARY)

    @overrides
    def apply_rotation(self, period: float,
                       inclination: Union[float, jnp.ndarray] = jnp.array([0., 1., 0.])) -> jnp.ndarray:
        self.__frequency = 1/period
        self.__inclination = inclination
        self.b.set_value(qualifier='incl', component='primary', value=inclination)
        self.b.set_value(qualifier='period', component='primary', value=period)
        self.b.run_compute(irrad_method='none',
                           distortion_method='rotstar',
                           model='rotmodel', overwrite=True)
        return self.get_los_velocities()

    @overrides
    def apply_pulsation(self, m: float, n: float, magnitude: float, t0: float, period: float) -> jnp.ndarray:
        return jnp.zeros(1,)
    
    @overrides
    def get_los_velocities_for_time(self, time_index: int) -> jnp.ndarray:
        return self.get_parameter(self.timestamps[time_index], 'vws', Component.PRIMARY)
    
    @overrides
    def get_los_velocities(self) -> jnp.ndarray:
        return jnp.array([self.get_parameter(time, 'vws', Component.PRIMARY) for time in self.b.times])

    def get_parameter(self, time: float, qualifier: str, component: Optional[Component] = None) -> np.array:
        if component is not None:
            return self.b.get_parameter(qualifier=qualifier,
                                        component=str(component),
                                        dataset=self.dataset_name,
                                        kind='mesh',
                                        time=time).value
        else:
            return self.b.get_parameter(qualifier=qualifier,
                            dataset=self.dataset_name,
                            kind='mesh',
                            time=time).value
    
    def get_loggs_for_time(self, time_index: int) -> np.ndarray:
        return self.get_parameter(self.timestamps[time_index], 'loggs', Component.PRIMARY)
    
    def get_teffs_for_time(self, time_index: int) -> np.ndarray:
        return self.get_parameter(self.timestamps[time_index], 'teffs', Component.PRIMARY)
    
    @overrides
    def model_spectrum(self,
                       time_index: int,
                       log_wavelengths: jnp.ndarray,
                       parameters: jnp.ndarray,
                       chunk_size: int = 256) -> jnp.ndarray:
        mus = self.get_mus_for_time(time_index)
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities_for_time(time_index)
        
        return spectrum_flash_sum(log_wavelengths,
                                  (mus*self.get_areas_for_time(time_index))[:, jnp.newaxis],
                                  mus[:, jnp.newaxis],
                                  los_vels[:, jnp.newaxis],
                                  parameters,
                                  chunk_size)
    @overrides
    def model_spectra(self,
                      log_wavelengths: jnp.ndarray,
                      parameters: jnp.ndarray,
                      chunk_size: int = 256) -> jnp.ndarray:
        mus = self.get_mus()
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities()
        spectra_flash_sum = get_spectra_flash_sum(chunk_size)
        
        return spectra_flash_sum(log_wavelengths,
                                 (mus*self.areas)[:, :, jnp.newaxis],
                                 mus[:, :, jnp.newaxis],
                                 los_vels[:, :, jnp.newaxis],
                                 parameters)
