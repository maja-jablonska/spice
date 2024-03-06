import numpy as np
import jax.numpy as jnp
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod, abstractproperty
from functools import wraps
import hashlib
from typing import Tuple, TypeVar
from overrides import overrides
from .mesh_generation import icosphere


# array type: np.ndarray, jnp.array
T = TypeVar("T")


# https://forum.kavli.tudelft.nl/t/caching-of-python-functions-with-array-input/59/6
def array_cache(function):
    cache = {}
    
    @wraps(function)
    def wrapped(array):
        return cache.get(hashlib.md5(array.data.tobytes()).digest(), function(array))
    
    return wrapped


class TimeseriesMesh(ABC):
    @abstractproperty
    def times(self) -> T:
        return NotImplemented
    
    @abstractproperty
    def phases(self) -> T:
        return NotImplemented
    
    def vertices(self, index: int) -> T:
        return NotImplemented

    def faces(self, index: int) -> T:
        return NotImplemented

    def face_centers(self, index: int) -> T:
        return NotImplemented
    
    def face_areas(self, index: int) -> T:
        return NotImplemented

    def vertex_mus(self, index: int, los: np.ndarray) -> T:
        return NotImplemented
    
    def center_mus(self, index: int, los: np.ndarray) -> T:
        return NotImplemented


class NumpyMesh(TimeseriesMesh):
    def __init__(self,
                 times: np.ndarray,
                 phases: np.ndarray,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 face_centers: np.ndarray,
                 face_areas: np.ndarray) -> None:
        self.__times = times
        self.__phases = phases
        self.__vertices = vertices
        self.__faces = faces
        self.__face_centers = face_centers
        self.__face_areas = face_areas
        
    @property
    @overrides
    def times(self) -> np.ndarray:
        return self.__times
    
    @property
    @overrides
    def phases(self) -> np.ndarray:
        return self.__phases

    @overrides
    def vertices(self, index: int) -> np.ndarray:
        return self.__vertices[index]

    @overrides
    def faces(self, index: int) -> np.ndarray:
        return self.__faces[index]

    @overrides
    def face_centers(self, index: int) -> np.ndarray:
        return self.__face_centers[index]
    
    @overrides
    def face_areas(self, index: int) -> T:
        return self.__face_areas[index]

    @array_cache
    def vertex_mus(self, index: int, los: np.ndarray) -> np.ndarray:
        """Return the mu vectors

        Args:
            los (np.ndarray): line-of-sight vector

        Returns:
            np.ndarray: mu values for stored vertices
        """
        return np.dot(self.vertices(index)/np.linalg.norm(self.vertices(index), axis=1, keepdims=True), los)


    @array_cache
    def center_mus(self, index: int, los: np.ndarray) -> np.ndarray:
        """Return the mu vectors

        Args:
            los (np.ndarray): line-of-sight vector

        Returns:
            np.ndarray: mu values for stored vertices
        """
        return np.dot(self.face_centers(index)/np.linalg.norm(self.face_centers(index), axis=1, keepdims=True), los)


class JAXMesh(TimeseriesMesh):
    def __init__(self,
                 times: jnp.ndarray,
                 phases: jnp.ndarray,
                 vertices: jnp.ndarray,
                 faces: jnp.ndarray,
                 face_centers: jnp.ndarray,
                 face_areas: jnp.ndarray) -> None:
        self.__vertices = vertices
        self.__faces = faces
        self.__face_centers = face_centers
        self.__face_areas = face_areas
        
    @classmethod
    def icosphere(cls, points: int, times: jnp.ndarray, phases: jnp.ndarray):
        # return verts, faces, areas, centers, verts_mask
        # vertices, f, a, c, m, vm = icosphere(500)
        vertices, faces, areas, centers, vertex_mask = icosphere(points)
        return cls(times, phases,
                   vertices[vertex_mask.astype(bool)][jnp.newaxis, :, :],
                   faces[jnp.newaxis, :, :],
                   centers[jnp.newaxis, :, :],
                   areas[jnp.newaxis, :])
    
    @property
    @overrides
    def times(self) -> jnp.ndarray:
        return self.__times
    
    @overrides
    def phases(self) -> jnp.ndarray:
        return self.__phases

    @overrides
    def vertices(self, index: int) -> jnp.ndarray:
        return self.__vertices[index]

    @overrides
    def faces(self, index: int) -> jnp.ndarray:
        return self.__faces[index]

    @overrides
    def face_centers(self, index: int) -> jnp.ndarray:
        return self.__face_centers[index]
    
    @overrides
    def face_areas(self, index: int) -> jnp.ndarray:
        return self.__face_areas[index]

    def vertex_mus(self, index: int, los: jnp.ndarray) -> jnp.ndarray:
        """Return the mu vectors

        Args:
            los (np.ndarray): line-of-sight vector

        Returns:
            np.ndarray: mu values for stored vertices
        """
        return jnp.dot(self.vertices(index)/jnp.linalg.norm(self.vertices(index), axis=1, keepdims=True), los)


    def center_mus(self, index: int, los: jnp.ndarray) -> jnp.ndarray:
        """Return the mu vectors

        Args:
            los (np.ndarray): line-of-sight vector

        Returns:
            np.ndarray: mu values for stored vertices
        """
        return jnp.dot(self.face_centers(index)/jnp.linalg.norm(self.face_centers(index), axis=1, keepdims=True), los)


class Model(ABC):
    @abstractproperty
    def mesh(self) -> TimeseriesMesh:
        return NotImplemented


class HarmonicPulsation:
    def __init__(self, period: float, amplitude: float, n: int, m: int):
        self.__period = period
        self.__amplitude = amplitude
        self.__n = n
        self.__m = m

    
class IcosphereModel(Model):
    def __init__(self, points: int, times: jnp.ndarray, phases: jnp.ndarray) -> None:
        self.__mesh = JAXMesh.icosphere(points, times, phases)

    @property
    @overrides
    def mesh(self) -> TimeseriesMesh:
        return self.__mesh
