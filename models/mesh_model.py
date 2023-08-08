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

        if len(parameters.shape) == 1:
            parameters = jnp.repeat(parameters[jnp.newaxis, :], repeats = areas.shape[0], axis = 0)

        return MeshModel.__new__(cls, radius, mass, abs_luminosity,
                vertices, faces, centers, areas, parameters,
                jnp.zeros_like(centers),
                DEFAULT_ROTATION_AXIS, NO_ROTATION_MATRIX, NO_ROTATION_MATRIX, calculate_axis_radii(centers, DEFAULT_ROTATION_AXIS), 0.,
                NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS, NO_PULSATION_ARRAYS,
                DEFAULT_LOS_VECTOR, cast_to_los(centers, DEFAULT_LOS_VECTOR), jnp.zeros_like(areas))
    
    @overrides
    def pulsation_modes(self) -> int:
        return jnp.count_nonzero(self.amplitudes)
