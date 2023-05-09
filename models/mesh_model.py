from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
import jax.numpy as jnp
from .mesh_generation import icosphere, calculate_rotation, apply_spherical_harm_pulsation
from overrides import overrides
from spectrum import spectrum_flash_sum, get_spectra_flash_sum
import phoebe
import numpy as np
from phoebe import u
from typing import Optional
from enum import auto, Enum
import astropy.units as un


DEFAULT_LOS_VECTOR: jnp.ndarray = jnp.array([0., 1., 0.])


class StarModel(metaclass=ABCMeta):
    def __init__(self, radius: float, timestamps: jnp.array):
        self.__radius = radius
        self.__timestamps = timestamps

    @property
    def radius(self):
        return self.__radius

    @property
    def phases(self) -> jnp.ndarray:
        return self.__phases
    
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
    def apply_rotation(self, omega: float, rotation_axis: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def apply_pulsations(pulsations: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_mus(self) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_los_velocities(self) -> jnp.ndarray:
        raise NotImplementedError
    

class MeshModel(StarModel):
    def __init__(self, n_vertices: int, radius: float, timestamps: jnp.ndarray):
        super().__init__(radius, timestamps)
        
        verts, faces, areas, centers, _ = icosphere(n_vertices)
        self.__verts = jnp.repeat(verts[jnp.newaxis, :, :], self.phases.shape[0], axis=0)
        self.__faces = jnp.repeat(faces[jnp.newaxis, :, :], self.phases.shape[0], axis=0)
        self.__areas = jnp.repeat(areas[jnp.newaxis, :], self.phases.shape[0], axis=0)
        self.__centers = jnp.repeat(centers[jnp.newaxis, :, :], self.phases.shape[0], axis=0)

        self.__radii = jnp.zeros_like(self.__areas)
        
        self.__omega = 0.0
        self.__rotation_velocity = 0.0
        self.__rotation_axis = jnp.array([0., 1., 0.])
        self.__velocities = jnp.zeros_like(self.__centers)
    
    @property
    def vertices(self) -> jnp.ndarray:
        return self.__verts
    
    @property
    def faces(self) -> jnp.ndarray:
        return self.__faces
    
    @faces.setter
    def faces(self, faces: jnp.ndarray):
        self.__faces = faces
    
    @property
    def areas(self) -> jnp.ndarray:
        return self.__areas
    
    @areas.setter
    def areas(self, areas: jnp.ndarray):
        self.__areas = areas
    
    @property
    def centers(self) -> jnp.ndarray:
        return self.__centers
    
    @centers.setter
    def centers(self, centers: jnp.ndarray):
        self.__centers = centers
        
    @property
    def rotation_axis(self) -> jnp.ndarray:
        return self.__rotation_axis
    
    @property
    def rotation_velocity(self) -> float:
        return self.__rotation_velocity
    
    def get_mus(self, los_vector: jnp.ndarray) -> jnp.ndarray:
        return -1.*jnp.dot(self.centers/
                           (jnp.linalg.norm(self.centers, axis=2, keepdims=True)+1e-10),
                           los_vector)
    
    def get_mus_for_time(self, time_index: int, los_vector: jnp.ndarray) -> jnp.ndarray:
        return -1.*jnp.dot(self.centers[time_index]/
                           (jnp.linalg.norm(self.centers[time_index],
                                            axis=1, keepdims=True)+1e-10),
                           los_vector)
    
    @overrides
    def apply_pulsations(pulsations: jnp.ndarray) -> jnp.ndarray:
        return self.__centers

    @overrides
    def apply_rotation(self, omega: float,
                       rotation_axis: jnp.ndarray = jnp.array([0., 1., 0.])) -> jnp.ndarray:
        self.__omega = omega
        self.__rotation_velocity = omega*self.radius
        rotation_axis = rotation_axis/jnp.linalg.norm(rotation_axis)

        # TODO: change to inclination
        self.__rotation_axis = rotation_axis
        self.__centers, self.__velocities, self.__radii = calculate_rotation(self.__omega,
                                                                             self.__rotation_axis,
                                                                             self.centers,
                                                                             self.timestamps)
        return jnp.vstack([self.__centers, self.__velocities])
    
    def get_los_velocities(self,
                           los_vector: jnp.ndarray = DEFAULT_LOS_VECTOR) -> jnp.ndarray:
        los_vels = jnp.dot(self.__velocities/(
            jnp.linalg.norm(self.__velocities, axis=2, keepdims=True)+1e-10
            ), los_vector)
        return self.__rotation_velocity*los_vels*self.__radii
    
    def get_los_velocities_for_time(self,
                                     time_index: int,
                                     los_vector: jnp.ndarray = DEFAULT_LOS_VECTOR) -> jnp.ndarray:
        los_vels = jnp.dot(self.__velocities[time_index]/(
            jnp.linalg.norm(self.__velocities[time_index], axis=1, keepdims=True)+1e-10
            ), los_vector)
        return self.__rotation_velocity*los_vels*self.__radii[time_index]
    
    def model_spectrum(self, phase_index: int,
                       los_vector: jnp.ndarray,
                       log_wavelengths: jnp.ndarray,
                       chunk_size: int = 256) -> jnp.ndarray:
        mus = self.get_mus_for_phase(phase_index, los_vector)
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities_for_phase(phase_index, los_vector)
        
        return spectrum_flash_sum(log_wavelengths,
                                  (mus*self.areas[phase_index])[:, jnp.newaxis],
                                  mus[:, jnp.newaxis],
                                  los_vels[:, jnp.newaxis],
                                  0.5*jnp.ones((1, 20)),
                                  chunk_size)
    
    def model_spectra(self, los_vector: jnp.ndarray,
                      log_wavelengths: jnp.ndarray,
                      parameters: jnp.ndarray,
                      chunk_size: int = 256) -> jnp.ndarray:
        mus = self.get_mus(los_vector)
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities(los_vector)
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
        self.__b.set_value('columns', value=['teffs', 'vws', 'us', 'vs', 'ws',
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
    
    def get_areas_for_time(self, time: float) -> jnp.ndarray:
        return self.get_parameter(time, 'areas', Component.PRIMARY)

    @property
    def areas(self) -> jnp.ndarray:
        return jnp.array([self.get_parameter(time, 'areas', Component.PRIMARY)
                          for time in self.b.times])
    
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
    
    def get_mus_for_time(self, time: float) -> jnp.ndarray:
        return self.get_parameter(time, 'mus', Component.PRIMARY)
    
    @overrides
    def get_mus(self) -> jnp.ndarray:
        return jnp.array([self.get_parameter(time, 'mus', Component.PRIMARY)
                          for time in self.timestamps])
    
    @property
    def rotation_velocity(self) -> float:
        return 2*jnp.pi*(self.radius*un.SolRad).to(un.km).value*self.__frequency
    
    def apply_rotation(self, period: float, inclination: float) -> jnp.ndarray:
        self.__frequency = 1/period
        self.__inclination = inclination
        self.b.set_value(qualifier='incl', component='primary', value=inclination)
        self.b.set_value(qualifier='period', component='primary', value=period)
        self.b.run_compute(irrad_method='none',
                           distortion_method='rotstar',
                           model='rotmodel', overwrite=True)
        return self.get_los_velocities()
        
    
    @overrides
    def apply_pulsations(pulsations: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(1,)
    
    def get_los_velocities_for_time(self, time: float) -> jnp.ndarray:
        return self.get_parameter(time, 'vws', Component.PRIMARY)
    
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
    
    def get_loggs(self, time: float, component: Optional[Component] = None) -> np.ndarray:
        return self.get_parameter(time, 'loggs', component)
    
    def get_teffs(self, time: float, component: Optional[Component] = None) -> np.ndarray:
        return self.get_parameter(time, 'teffs', component)
    
    def model_spectrum(self, time: float,
                       log_wavelengths: jnp.ndarray,
                       parameters: jnp.ndarray,
                       chunk_size: int = 256) -> jnp.ndarray:
        mus = self.get_mus_for_time(time)
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities_for_time(time)
        
        return spectrum_flash_sum(log_wavelengths,
                                  (mus*self.get_areas_for_time(time))[:, jnp.newaxis],
                                  mus[:, jnp.newaxis],
                                  los_vels[:, jnp.newaxis],
                                  parameters,
                                  chunk_size)
    
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
