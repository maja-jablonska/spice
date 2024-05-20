from jax.typing import ArrayLike
import jax.numpy as jnp

from abc import abstractmethod
from typing import NamedTuple, Union
from collections import namedtuple

from .mesh_generation import icosphere
from .model import Model
from .utils import calculate_axis_radii, cast_to_los, cast_to_normal_plane, cast_normalized_to_los
from spice.geometry.utils import get_cast_areas


DEFAULT_LOS_VECTOR: jnp.ndarray = jnp.array([0., 1., 0.]) # from the Y direction
DEFAULT_ROTATION_AXIS = jnp.ndarray = jnp.array([0., 0., 1.]) # from the Y direction

NO_ROTATION_MATRIX = jnp.zeros((3, 3))

MeshModelNamedTuple = namedtuple("MeshModel", ["center", "radius", "mass", "abs_luminosity", "log_g", "d_vertices", "faces", "d_centers", "areas", "parameters", "rotation_velocities", "rotation_axis", "rotation_matrix", "rotation_matrix_prim", "axis_radii", "rotation_velocity", "orbital_velocity", "los_vector"])

class MeshModel(Model, MeshModelNamedTuple):
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
    def velocities(self) -> ArrayLike:
        return self.rotation_velocities + self.orbital_velocity
    
    @property
    def mus(self) -> ArrayLike:
        return cast_normalized_to_los(self.d_centers, self.los_vector)
    
    @property
    def los_velocities(self) -> ArrayLike:
        return cast_to_los(self.velocities, self.los_vector)
    
    @property
    def los_z(self) -> ArrayLike:
        return cast_to_los(self.centers, self.los_vector)
    
    @property
    def cast_vertices(self) -> ArrayLike:
        return cast_to_normal_plane(self.vertices, self.los_vector)
    
    @property
    def cast_centers(self) -> ArrayLike:
        return cast_to_normal_plane(self.centers, self.los_vector)

    @property
    def cast_areas(self) -> ArrayLike:
        return get_cast_areas(self.cast_vertices[self.faces.astype(int)])        

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
                  parameters: Union[float, ArrayLike]) -> "IcosphereModel": # What to do about parameters?
        """Construct an Icosphere.

        Args:
            n_vertices (int): Minimal number of vertices (used to calculate number of divisions)
            radius (float): Radius in cm
            mass (float): Mass in kg
            abs_luminosity (float): Absolute luminosity in solar luminosities
            parameters (ArrayLike): Array of global parameters

        Returns:
            IcosphereModel:
        """
        vertices, faces, areas, centers = icosphere(n_vertices)
        sphere_area = 4*jnp.pi*jnp.power(radius, 2)
        log_g = jnp.log(6.6743e-11*mass/jnp.power(radius, 2)/9.80665)

        parameters = jnp.atleast_1d(parameters)
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
                los_vector=DEFAULT_LOS_VECTOR)
