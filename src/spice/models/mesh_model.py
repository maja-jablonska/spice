from jax.typing import ArrayLike
import jax.numpy as jnp

from abc import abstractmethod
from typing import NamedTuple, Union

from .mesh_generation import icosphere
from .utils import calculate_axis_radii, cast_to_los, cast_to_normal_plane
from spice.geometry.utils import get_cast_areas


DEFAULT_LOS_VECTOR: jnp.ndarray = jnp.array([0., 1., 0.]) # from the Y direction
DEFAULT_ROTATION_AXIS = jnp.ndarray = jnp.array([0., 0., 1.]) # from the Y direction

NO_ROTATION_MATRIX = jnp.zeros((3, 3))

DEFAULT_MAX_PULSATION_MODE_PARAMETER = 3
DEFAULT_FOURIER_ORDER = 5

def create_harmonics_params(n: int):
    x, y = jnp.meshgrid(jnp.arange(0, n), jnp.arange(0, n))
    return jnp.vstack([x.ravel(), y.ravel()]).T

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
    base_areas: ArrayLike

    parameters: ArrayLike

    # Motion properties per-triangle
    rotation_velocities: ArrayLike
    
    # Pulsations
    vertices_pulsation_offsets: ArrayLike
    center_pulsation_offsets: ArrayLike
    area_pulsation_offsets: ArrayLike
    
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
    cast_areas: ArrayLike
    
    mus: ArrayLike
    los_velocities: ArrayLike
    # Pulsation properties
    max_pulsation_mode: int
    max_fourier_order: int
    
    spherical_harmonics_parameters: ArrayLike
    fourier_series_static_parameters: ArrayLike
    fourier_series_parameters: ArrayLike
    
    @property
    def areas(self) -> ArrayLike:
        return self.base_areas + self.area_pulsation_offsets

    @property
    def vertices(self) -> ArrayLike:
        if len(self.d_vertices.shape)==2:
            return self.d_vertices + self.center + self.vertices_pulsation_offsets
        else:
            return self.d_vertices + self.center.reshape((self.d_vertices.shape[0], *([1]*(len(self.d_vertices.shape)-2)), self.d_vertices.shape[-1])) + self.vertices_pulsation_offsets
    
    @property
    def centers(self) -> ArrayLike:
        if len(self.d_centers.shape) == 2:
            return self.d_centers + self.center + self.center_pulsation_offsets
        else:
            return self.d_centers + self.center.reshape((self.d_centers.shape[0], *([1]*(len(self.d_centers.shape)-2)), self.d_centers.shape[-1])) + self.center_pulsation_offsets

    @property
    def velocities(self) -> jnp.float64:
        return self.rotation_velocities + self.orbital_velocity


class IcosphereModel(MeshModel):
    # TODO: show this instead of MeshModel initializer
    @classmethod
    def construct(cls, n_vertices: int,
                  radius: float,
                  mass: float,
                  abs_luminosity: float,
                  parameters: Union[float, ArrayLike],
                  max_pulsation_mode: int = DEFAULT_MAX_PULSATION_MODE_PARAMETER,
                  max_fourier_order: int = DEFAULT_FOURIER_ORDER) -> "IcosphereModel": # What to do about parameters?
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
        
        cast_vertices = cast_to_normal_plane(vertices*radius, DEFAULT_LOS_VECTOR)
        cast_centers = cast_to_normal_plane(centers*radius, DEFAULT_LOS_VECTOR)
        harmonics_params = create_harmonics_params(max_pulsation_mode)

        return MeshModel.__new__(cls, 0., radius, mass, abs_luminosity, log_g*jnp.ones_like(areas),
                d_vertices=vertices*radius, faces=faces, d_centers=centers*radius, base_areas=areas*sphere_area/jnp.sum(areas), parameters=parameters,
                rotation_velocities=jnp.zeros_like(centers),
                vertices_pulsation_offsets=jnp.zeros_like(vertices),
                center_pulsation_offsets=jnp.zeros_like(centers),
                area_pulsation_offsets=jnp.zeros_like(areas),
                rotation_axis=DEFAULT_ROTATION_AXIS,
                rotation_matrix=NO_ROTATION_MATRIX,
                rotation_matrix_prim=NO_ROTATION_MATRIX,
                axis_radii=calculate_axis_radii(centers, DEFAULT_ROTATION_AXIS),
                rotation_velocity=0.,
                orbital_velocity=0.,
                los_vector=DEFAULT_LOS_VECTOR,
                los_z=cast_to_los(centers*radius, DEFAULT_LOS_VECTOR),
                cast_vertices=cast_vertices,
                cast_centers=cast_centers,
                cast_areas=get_cast_areas(cast_vertices[faces.astype(int)]),
                mus=cast_to_los(centers, DEFAULT_LOS_VECTOR),
                los_velocities=jnp.zeros_like(areas),
                max_pulsation_mode=max_pulsation_mode,
                max_fourier_order=max_fourier_order,
                spherical_harmonics_parameters=harmonics_params,
                fourier_series_static_parameters=jnp.nan*jnp.ones((harmonics_params.shape[0], 2)), # D_0 (amplitude), period
                fourier_series_parameters=jnp.nan*jnp.ones((harmonics_params.shape[0], max_fourier_order, 2))) # D_n, phi_n
