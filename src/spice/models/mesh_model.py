from jax.typing import ArrayLike
import jax.numpy as jnp

from abc import abstractmethod
from typing import List, NamedTuple, Optional, Union
from collections import namedtuple
import warnings

from .mesh_generation import icosphere
from .model import Model
from .utils import calculate_axis_radii, cast_to_los, cast_to_normal_plane, cast_normalized_to_los
from spice.geometry.utils import get_cast_areas


LOG_G_NAMES: List[str] = ['logg', 'loggs', 'log_g', 'log_gs', 'log g', 'log gs',
                         'surface gravity', 'surface gravities', 'surface_gravity', 'surface_gravities']

DEFAULT_LOS_VECTOR: jnp.ndarray = jnp.array([0., 1., 0.]) # from the Y direction
DEFAULT_ROTATION_AXIS = jnp.ndarray = jnp.array([0., 0., 1.]) # from the Y direction

NO_ROTATION_MATRIX = jnp.zeros((3, 3))

DEFAULT_MAX_PULSATION_MODE_PARAMETER = 3
DEFAULT_FOURIER_ORDER = 5

def create_harmonics_params(n: int):
    x, y = jnp.meshgrid(jnp.arange(0, n), jnp.arange(0, n))
    return jnp.vstack([x.ravel(), y.ravel()]).T


def calculate_log_gs(mass: float, d_centers: ArrayLike):
    return jnp.log(6.6743e-11*mass/jnp.power(jnp.linalg.norm(d_centers, axis=1)*1e-2, 2)/9.80665)

MeshModelNamedTuple = namedtuple("MeshModel",
                                 ["center", "radius", "mass", "abs_luminosity",
                                  "d_vertices", "faces", "d_centers",
                                  "base_areas", "parameters", "log_g_index", "rotation_velocities",
                                  "vertices_pulsation_offsets", "center_pulsation_offsets", "area_pulsation_offsets", "pulsation_velocities",
                                  "rotation_axis", "rotation_matrix", "rotation_matrix_prim",
                                  "axis_radii", "rotation_velocity", "orbital_velocity", "los_vector",
                                  "max_pulsation_mode", "max_fourier_order", "spherical_harmonics_parameters",
                                  "fourier_series_static_parameters", "fourier_series_parameters"])

class MeshModel(Model, MeshModelNamedTuple):
    # Stellar properties
    center: ArrayLike
    radius: float
    mass: float
    abs_luminosity: float

    # Mesh properties
    # vertices and centers in the reference frame of centered on the center vector
    d_vertices: ArrayLike
    faces: ArrayLike
    d_centers: ArrayLike
    base_areas: ArrayLike

    parameters: ArrayLike
    log_g_index: Optional[int]

    # Motion properties per-triangle
    rotation_velocities: ArrayLike
    
    # Pulsations
    vertices_pulsation_offsets: ArrayLike
    center_pulsation_offsets: ArrayLike
    area_pulsation_offsets: ArrayLike
    
    # Per center
    pulsation_velocities: ArrayLike
    
    # Rotation
    rotation_axis: ArrayLike
    rotation_matrix: ArrayLike
    rotation_matrix_prim: ArrayLike
    axis_radii: ArrayLike
    rotation_velocity: ArrayLike

    orbital_velocity: float

    # Mesh LOS properties
    los_vector: ArrayLike
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
    def log_gs(self) -> ArrayLike:
        return calculate_log_gs(self.mass, self.centers-self.center)

    @property
    def vertices(self) -> ArrayLike:
        if len(self.d_vertices.shape)==2:
            return self.d_vertices + self.center + self.vertices_pulsation_offsets
        else:
            return self.d_vertices + self.center.reshape((self.d_vertices.shape[0], *([1]*(len(self.d_vertices.shape)-2)), self.d_vertices.shape[-1]))
        
    @property
    def mesh_elements(self) -> ArrayLike:
        return self.vertices[self.faces.astype(int)]
    
    @property
    def centers(self) -> ArrayLike:
        if len(self.d_centers.shape) == 2:
            return self.d_centers + self.center + self.center_pulsation_offsets
        else:
            return self.d_centers + self.center.reshape((self.d_centers.shape[0], *([1]*(len(self.d_centers.shape)-2)), self.d_centers.shape[-1])) + self.center_pulsation_offsets

    @property
    def velocities(self) -> jnp.float64:
        return self.rotation_velocities + self.orbital_velocity + self.pulsation_velocities
    
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


class IcosphereModel(MeshModel):
    # TODO: show this instead of MeshModel initializer
    @classmethod
    def construct(cls, n_vertices: int,
                  radius: float,
                  mass: float,
                  abs_luminosity: float,
                  parameters: Union[float, ArrayLike],
                  parameter_names: List[str],
                  max_pulsation_mode: int = DEFAULT_MAX_PULSATION_MODE_PARAMETER,
                  max_fourier_order: int = DEFAULT_FOURIER_ORDER,
                  override_log_g: bool = True,
                  log_g_index: Optional[int] = None) -> "IcosphereModel":
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

        parameters = jnp.atleast_1d(parameters)
        if len(parameters.shape) == 1:
            parameters = jnp.repeat(parameters[jnp.newaxis, :], repeats = areas.shape[0], axis = 0)
        if override_log_g:
            if any([pn in parameter_names for pn in LOG_G_NAMES]):
                log_g_index = [i for i, pn in enumerate(parameter_names) if pn in LOG_G_NAMES][0]
                parameters = parameters.at[:, log_g_index].set(calculate_log_gs(mass, centers*radius))
            elif log_g_index and isinstance(log_g_index, int):
                parameters = parameters.at[:, log_g_index].set(calculate_log_gs(mass, centers*radius))
            else:
                warnings.warn(f"If override_log_g is True, either parameter_names must include one of [" + ",".join(LOG_G_NAMES) + "], or log_g_index must be passed for log g to be used in the spectrum emulator.")
        
        harmonics_params = create_harmonics_params(max_pulsation_mode)

        return MeshModel.__new__(cls, 0., radius, mass, abs_luminosity,
                d_vertices=vertices*radius, faces=faces, d_centers=centers*radius, base_areas=areas*sphere_area/jnp.sum(areas),
                parameters=parameters,
                log_g_index = log_g_index,
                rotation_velocities=jnp.zeros_like(centers),
                vertices_pulsation_offsets=jnp.zeros_like(vertices),
                center_pulsation_offsets=jnp.zeros_like(centers),
                area_pulsation_offsets=jnp.zeros_like(areas),
                pulsation_velocities=jnp.zeros_like(centers),
                rotation_axis=DEFAULT_ROTATION_AXIS,
                rotation_matrix=NO_ROTATION_MATRIX,
                rotation_matrix_prim=NO_ROTATION_MATRIX,
                axis_radii=calculate_axis_radii(centers, DEFAULT_ROTATION_AXIS),
                rotation_velocity=0.,
                orbital_velocity=0.,
                los_vector=DEFAULT_LOS_VECTOR,
                max_pulsation_mode=max_pulsation_mode,
                max_fourier_order=max_fourier_order,
                spherical_harmonics_parameters=harmonics_params,
                fourier_series_static_parameters=jnp.nan*jnp.ones((harmonics_params.shape[0], 2)), # D_0 (amplitude), period
                fourier_series_parameters=jnp.nan*jnp.ones((harmonics_params.shape[0], max_fourier_order, 2))) # D_n, phi_n
