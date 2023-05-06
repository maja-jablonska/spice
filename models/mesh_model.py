from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
import jax.numpy as jnp
from .mesh_generation import icosphere, calculate_rotation, apply_spherical_harm_pulsation
from overrides import overrides
from spectrum import spectrum_flash_sum, get_spectra_flash_sum


class StarModel(metaclass=ABCMeta):
    def __init__(self, radius: float, phases: jnp.array, timestamps: jnp.array):
        self.__radius = radius
        self.__phases = phases
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
    def vertices(self) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractproperty
    def faces(self) -> jnp.ndarray:
        raise NotImplementedError
    
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
    def apply_rotation(self, omega: float, rotation_axis: jnp.ndarray) -> jnp.array:
        raise NotImplementedError
    
    @abstractmethod
    def apply_pulsations(pulsations: jnp.array) -> jnp.array:
        raise NotImplementedError
    
    @abstractmethod
    def get_mus(self, los_vector: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_los_velocities(self, los_vector: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_observed_los_velocities(self, los_vector: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    

class MeshModel(StarModel):
    def __init__(self, n_vertices: int, radius: float, phases: jnp.array, timestamps: jnp.array):
        super().__init__(radius, phases, timestamps)
        
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
    
    @overrides
    def get_mus(self, los_vector: jnp.ndarray) -> jnp.ndarray:
        return -1.*jnp.dot(self.centers/
                           (jnp.linalg.norm(self.centers, axis=2, keepdims=True)+1e-10),
                           los_vector)
    
    def get_mus_for_phase(self, phase_index: int, los_vector: jnp.ndarray) -> jnp.ndarray:
        return -1.*jnp.dot(self.centers[phase_index]/
                           (jnp.linalg.norm(self.centers[phase_index],
                                            axis=1, keepdims=True)+1e-10),
                           los_vector)
    
    @overrides
    def apply_pulsations(pulsations: jnp.array) -> jnp.array:
        return self.__centers

    @overrides
    def apply_rotation(self, omega: float,
                       rotation_axis: jnp.ndarray = jnp.array([0., 1., 0.])) -> jnp.array:
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
    
    @overrides
    def get_los_velocities(self,
                           los_vector: jnp.ndarray = jnp.array([1., 0., 0.])) -> jnp.ndarray:
        los_vels = jnp.dot(self.__velocities/(
            jnp.linalg.norm(self.__velocities, axis=2, keepdims=True)+1e-10
            ), los_vector)
        return self.__rotation_velocity*los_vels*self.__radii
    
    def get_los_velocities_for_phase(self,
                                     phase_index: int,
                                     los_vector: jnp.ndarray = jnp.array([1., 0., 0.])) -> jnp.ndarray:
        los_vels = jnp.dot(self.__velocities[phase_index]/(
            jnp.linalg.norm(self.__velocities[phase_index], axis=1, keepdims=True)+1e-10
            ), los_vector)
        return self.__rotation_velocity*los_vels*self.__radii[phase_index]
    
    @overrides
    def get_observed_los_velocities(self, los_vector: jnp.ndarray) -> jnp.ndarray:
        mus = self.get_mus(los_vector)
        mu_mask = mus>0
        return self.get_los_velocities(los_vector)[mu_mask]
    
    def model_spectrum(self, phase_index: int,
                       los_vector: jnp.ndarray,
                       log_wavelengths: jnp.array,
                       chunk_size: int = 256) -> jnp.array:
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
                      log_wavelengths: jnp.array,
                      chunk_size: int = 256) -> jnp.array:
        mus = self.get_mus(los_vector)
        mus = jnp.where(mus>0, mus, 0.)
        los_vels = self.get_los_velocities(los_vector)
        spectra_flash_sum = get_spectra_flash_sum(chunk_size)
        
        return spectra_flash_sum(log_wavelengths,
                                 (mus*self.areas)[:, :, jnp.newaxis],
                                 mus[:, :, jnp.newaxis],
                                 los_vels[:, :, jnp.newaxis],
                                 jnp.repeat(0.5*jnp.ones((1, 20)), mus.shape[0], axis=0))
