from typing import Union
from jax import jit
from jax.typing import ArrayLike
import jax.numpy as jnp

from abc import ABCMeta, abstractmethod
from typing import NamedTuple

from overrides import overrides

from .mesh_generation import icosphere
from .utils import calculate_axis_radii, cast_to_los, cast_to_normal_plane


DEFAULT_LOS_VECTOR: jnp.ndarray = jnp.array([0., 1., 0.]) # from the Y direction
DEFAULT_ROTATION_AXIS = jnp.ndarray = jnp.array([0., 0., 1.]) # from the Y direction

NO_ROTATION_MATRIX = jnp.zeros((3, 3))

class MeshModel(NamedTuple):
    # Stellar properties
    center: ArrayLike
    radius: float
    mass: float
    abs_luminosity: float
    log_g: ArrayLike

    # Mesh properties
    # vertices and centers in the reference frame of centered on the center vector
    d_vertices: ArrayLike
    faces: ArrayLike
    d_centers: ArrayLike
    areas: ArrayLike

    parameters: ArrayLike

    # Motion properties per-triangle
    rotation_velocities: ArrayLike
    
    # Rotation
    rotation_axis: ArrayLike
    rotation_matrix: ArrayLike
    rotation_matrix_prim: ArrayLike
    axis_radii: ArrayLike
    rotation_velocity: ArrayLike

    orbital_velocity: float

    # Mesh LOS properties
    los_vector: ArrayLike
    los_z: ArrayLike
    cast_vertices: ArrayLike
    cast_centers: ArrayLike
    
    mus: ArrayLike
    los_velocities: ArrayLike

    @property
    def vertices(self) -> ArrayLike:
        if len(self.d_vertices.shape)==2:
            return self.d_vertices + self.center
        else:
            return self.d_vertices + self.center.reshape((self.d_vertices.shape[0], *([1]*(len(self.d_vertices.shape)-2)), self.d_vertices.shape[-1]))
    
    @property
    def centers(self) -> ArrayLike:
        if len(self.d_centers.shape) == 2:
            return self.d_centers + self.center
        else:
            return self.d_centers + self.center.reshape((self.d_centers.shape[0], *([1]*(len(self.d_centers.shape)-2)), self.d_centers.shape[-1]))

    @property
    def velocities(self) -> jnp.float64:
        return self.rotation_velocities + self.orbital_velocity


    @abstractmethod
    def pulsation_modes(self) -> int:
        raise NotImplementedError()


class IcosphereModel(MeshModel):
    # TODO: show this instead of MeshModel initializer
    @classmethod
    def construct(cls, n_vertices: int,
                  radius: float,
                  mass: float,
                  abs_luminosity: float,
                  parameters: ArrayLike): # What to do about parameters?
        """Construct an Icosphere.

        Args:
            n_vertices (int): Minimal number of vertices (used to calculate number of divisions)
            radius (float): Radius in cm
            mass (float): Mass in kg
            abs_luminosity (float): Absolute luminosity in solar luminosities
            parameters (ArrayLike): Array of global parameters

        Returns:
            _type_: _description_
        """
        vertices, faces, areas, centers = icosphere(n_vertices)
        sphere_area = 4*jnp.pi*jnp.power(radius, 2)
        log_g = jnp.log(6.6743e-11*mass/jnp.power(radius, 2)/9.80665)

        if len(parameters.shape) == 1:
            parameters = jnp.repeat(parameters[jnp.newaxis, :], repeats = areas.shape[0], axis = 0)

        return MeshModel.__new__(cls, 0., radius, mass, abs_luminosity, log_g*jnp.ones_like(areas),
                d_vertices=vertices*radius, faces=faces, d_centers=centers*radius, areas=areas*sphere_area/jnp.sum(areas), parameters=parameters,
                rotation_velocities=jnp.zeros_like(centers),
                rotation_axis=DEFAULT_ROTATION_AXIS,
                rotation_matrix=NO_ROTATION_MATRIX,
                rotation_matrix_prim=NO_ROTATION_MATRIX,
                axis_radii=calculate_axis_radii(centers, DEFAULT_ROTATION_AXIS),
                rotation_velocity=0.,
                orbital_velocity=0.,
                los_vector=DEFAULT_LOS_VECTOR,
                los_z=cast_to_los(centers*radius, DEFAULT_LOS_VECTOR),
                cast_vertices=cast_to_normal_plane(vertices*radius, DEFAULT_LOS_VECTOR),
                cast_centers=cast_to_normal_plane(centers*radius, DEFAULT_LOS_VECTOR),
                mus=cast_to_los(centers, DEFAULT_LOS_VECTOR),
                los_velocities=jnp.zeros_like(areas))
