import warnings
import jax.numpy as jnp
from functools import partial
import jax
from jax.typing import ArrayLike
from typing import List, NamedTuple, Optional, Tuple, Dict

from spice.models.phoebe_model import DAY_TO_S, PhoebeModel
from spice.models.phoebe_utils import Component, PhoebeConfig
from .model import Model
from .mesh_transform import transform, evaluate_body_orbit
import astropy.units as u
from .orbit_utils import get_orbit_jax
from .mesh_view import get_grid_spans, resolve_occlusion, Grid
from collections import namedtuple
from jax import tree_util


YEAR_TO_SECONDS = u.year.to(u.s)
DAY_TO_YEAR = 0.0027378507871321013
SOLAR_MASS_KG = 1.988409870698051e+30
SOLAR_RAD_CM = 6.957e10
SOLAR_RAD_M = 6.957e8

class Binary(NamedTuple):
    body1: Model
    body2: Model

    # Orbital elements
    P: float
    ecc: float
    T: float
    i: float
    omega: float
    Omega: float
    
    mean_anomaly: float
    reference_time: float
    vgamma: float

    evaluated_times: ArrayLike

    body1_centers: ArrayLike
    body2_centers: ArrayLike
    body1_velocities: ArrayLike
    body2_velocities: ArrayLike

    @classmethod
    def from_bodies(cls, body1: Model, body2: Model) -> "Binary":
        """Construct a Binary object from two mesh models.

          Args:
              body1 (Model):
              body2 (Model):

          Returns:
              Binary: a binary consisting of body1 and body2
          """
        return cls(body1, body2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   jnp.zeros_like(body1.centers), jnp.zeros_like(body2.centers),
                   jnp.zeros_like(body1.velocities), jnp.zeros_like(body2.velocities))


@tree_util.register_pytree_node_class
class PhoebeBinary(namedtuple("PhoebeBinary",
                              ["body1", "body2", "P", "ecc", "T", "i", "omega", "Omega",
                               "mean_anomaly", "reference_time", "vgamma",
                               "evaluated_times", "body1_centers", "body2_centers", "body1_velocities",
                               "body2_velocities", "phoebe_config", "parameter_labels", "parameter_values"])):
    body1: Model
    body2: Model

    # Orbital elements
    P: float
    ecc: float
    T: float
    i: float
    omega: float
    Omega: float
    mean_anomaly: float
    reference_time: float
    vgamma: float

    evaluated_times: ArrayLike

    body1_centers: ArrayLike
    body2_centers: ArrayLike
    body1_velocities: ArrayLike
    body2_velocities: ArrayLike
    phoebe_config: PhoebeConfig
    parameter_labels: List[str]
    parameter_values: Dict[str, float]

    @classmethod
    def construct(cls, phoebe_config: PhoebeConfig,
                  parameter_labels: List[str] = None,
                  parameter_values: Dict[str, float] = None) -> "PhoebeBinary":
        """
        Construct a PhoebeBinary object from a PhoebeConfig.

        This method initializes a PhoebeBinary object using the provided PhoebeConfig and optional parameter labels and values.
        It constructs the primary and secondary body models and retrieves various orbital elements and properties from the config.

        Args:
            phoebe_config (PhoebeConfig): The configuration object containing the parameters and settings for the binary system.
            parameter_labels (List[str], optional): A list of parameter labels to be used for the construction of the models. Defaults to None.
            parameter_values (Dict[str, float], optional): A dictionary of parameter values corresponding to the labels. Defaults to None.

        Returns:
            PhoebeBinary: A new instance of PhoebeBinary initialized with the provided configuration and parameters.
        """
        body1 = PhoebeModel.construct(phoebe_config, phoebe_config.times[0], parameter_labels, parameter_values, Component.PRIMARY)
        body2 = PhoebeModel.construct(phoebe_config, phoebe_config.times[0], parameter_labels, parameter_values, Component.SECONDARY)
        return PhoebeBinary.__new__(cls, body1=body1, body2=body2, evaluated_times=phoebe_config.times,
                                    P=phoebe_config.get_quantity('period', component='binary') * DAY_TO_YEAR,
                                    ecc=phoebe_config.get_quantity('ecc', component='binary'),
                                    T=phoebe_config.get_quantity('t0_perpass', component='binary') * DAY_TO_YEAR,
                                    i=phoebe_config.get_quantity('incl', component='binary') * 0.017453292519943295,
                                    omega=phoebe_config.get_quantity('per0', component='binary')*0.017453292519943295,
                                    Omega=phoebe_config.get_quantity('long_an', component='binary')*0.017453292519943295,
                                    mean_anomaly=phoebe_config.get_quantity('mean_anom', component='binary')*0.017453292519943295,
                                    reference_time=phoebe_config.get_quantity('t0_ref', component='binary')*DAY_TO_YEAR,
                                    vgamma=phoebe_config.b.get_parameter('vgamma').value,
                                    body1_centers=phoebe_config.get_all_orbit_centers(str(Component.PRIMARY)),
                                    body2_centers=phoebe_config.get_all_orbit_centers(str(Component.SECONDARY)),
                                    body1_velocities=phoebe_config.get_all_orbit_velocities(str(Component.PRIMARY)),
                                    body2_velocities=phoebe_config.get_all_orbit_velocities(str(Component.SECONDARY)),
                                    phoebe_config=phoebe_config,
                                    parameter_labels=parameter_labels,
                                    parameter_values=parameter_values
                                    )
        
    def tree_flatten(self):
        children = (self.body1, self.body2, self.P, self.ecc, self.T, self.i, self.omega, self.Omega,
                    self.evaluated_times, self.body1_centers, self.body2_centers, self.body1_velocities,
                    self.body2_velocities, self.phoebe_config, self.parameter_labels, self.parameter_values)
        aux_data = {}
        return children, aux_data
    
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
        


@partial(jax.jit, static_argnums=(10,))
def _add_orbit(binary: Binary, P: float, ecc: float,
               T: float, i: float, omega: float, Omega: float, mean_anomaly: float,
               reference_time: float, vgamma: float, orbit_resolution_points: int) -> Binary:
    orbit_resolution_times = jnp.linspace(0, P, orbit_resolution_points)
    orbit = get_orbit_jax(orbit_resolution_times, binary.body1.mass, binary.body2.mass,
                          P, ecc, T, i, omega, Omega, mean_anomaly, reference_time, vgamma)
    return binary._replace(P=P, ecc=ecc, T=T, i=i, omega=omega, Omega=Omega, mean_anomaly=mean_anomaly, reference_time=reference_time,
                           vgamma=vgamma,
                           evaluated_times=orbit_resolution_times,
                           body1_centers=(orbit[0, :, :]+orbit[2, :, :])/SOLAR_RAD_M, body2_centers=(orbit[0, :, :]+orbit[4, :, :])/SOLAR_RAD_M,
                           body1_velocities=(orbit[1, :, :]+orbit[3, :, :]), body2_velocities=(orbit[1, :, :]+orbit[5, :, :]))


def add_orbit(binary: Binary,
              P: float,
              ecc: float,
              T: float,
              i: float,
              omega: float,
              Omega: float,
              mean_anomaly: float,
              reference_time: float,
              vgamma: float,
              orbit_resolution_points: int) -> Binary:
    """
    Add orbit information to the binary object.

    This function calculates the orbit of the binary system and updates the Binary object
    with the new orbital information.

    Args:
        binary (Binary): The binary object to update.
        P (float): Orbital period in years.
        ecc (float): Eccentricity of the orbit.
        T (float): Time of periastron passage in years.
        i (float): Inclination of the orbit in radians.
        omega (float): Argument of periastron in radians.
        Omega (float): Longitude of the ascending node in radians.
        mean_anomaly (float): Mean anomaly at reference time in radians.
        reference_time (float): Reference time in years.
        orbit_resolution_points (int): Number of points to use for orbit resolution.

    Returns:
        Binary: Updated binary object with new orbital information.
    """
    if isinstance(binary, PhoebeBinary):
        raise ValueError("PhoebeBinary objects are read-only - the orbit information is already added.")
    else:
        return _add_orbit(binary, P, ecc, T, i, omega, Omega, mean_anomaly, reference_time, vgamma, orbit_resolution_points)


@jax.jit
def _interpolate_orbit(binary: Binary, time: ArrayLike) -> Tuple[Model, Model]:
    """
    Interpolate the orbit positions and velocities at a given time.
    
    This function interpolates over each spatial coordinate separately.
    """
    # Helper function: Interpolate a 2D array of shape (n_eval, dim) along the n_eval axis.
    # It returns an array of shape (dim,) representing the interpolated values.
    def interp_for_array(arr):
        # arr has shape (n_eval, dim). We want to interpolate along the first axis.
        # Transpose so that each coordinate becomes a 1D array of length n_eval.
        # Then vmap over each coordinate.
        return jnp.transpose(
            jax.vmap(lambda col: jnp.interp(time, binary.evaluated_times, col, period=binary.P))(
                arr.T
            )
        )
    
    # Interpolate the centers and velocities for each body.
    body1_center = interp_for_array(binary.body1_centers)
    body2_center = interp_for_array(binary.body2_centers)
    body1_velocity = interp_for_array(binary.body1_velocities)
    body2_velocity = interp_for_array(binary.body2_velocities)
    
    # Apply the interpolated shifts to the models.
    body1 = evaluate_body_orbit(
        transform(binary.body1, binary.body1.center + body1_center),
        body1_velocity
    )
    body2 = evaluate_body_orbit(
        transform(binary.body2, binary.body2.center + body2_center),
        body2_velocity
    )
    
    return body1, body2



@partial(jax.jit, static_argnums=(2,))
def _evaluate_orbit(binary: Binary, time: ArrayLike, grid: Grid) -> Tuple[Model, Model]:
    body1, body2 = _interpolate_orbit(binary, time)

    return jax.lax.cond(jnp.mean(body1.los_z) > jnp.mean(body2.los_z),
                        lambda: (body1, resolve_occlusion(body2, body1, grid)),
                        lambda: (resolve_occlusion(body1, body2, grid), body2))


@partial(jax.jit, static_argnums=(2,))
def v_evaluate_orbit(binary: Binary, times: ArrayLike, grid: Grid) -> Tuple[Model, Model]:
    result_body1, result_body2 = jax.vmap(_evaluate_orbit, in_axes=(None, 0, None))(binary, times, grid)
    return jax.tree.transpose(
        outer_treedef=jax.tree.structure(binary.body1),
        inner_treedef=jax.tree.structure([0 for t in times]),
        pytree_to_transpose=jax.tree.map(lambda x: list(x), result_body1)
    ), jax.tree.transpose(
        outer_treedef=jax.tree.structure(binary.body2),
        inner_treedef=jax.tree.structure([0 for t in times]),
        pytree_to_transpose=jax.tree.map(lambda x: list(x), result_body2)
    )


def get_optimal_grid_size(m1, m2, min_cells=5, max_cells=30):
    """
    Calculate optimal grid size based on mesh cast areas and grid cell sizes.
    
    The function finds a grid size where the grid cell area is smaller than the 
    average cast area of mesh faces, but not so small that it creates unnecessary 
    computational overhead.
    
    Args:
        m1 (MeshModel): First mesh model
        m2 (MeshModel): Second mesh model 
        min_cells (int): Minimum number of grid cells to try
        max_cells (int): Maximum number of grid cells to try
        
    Returns:
        int: Optimal number of grid cells
    """
    # Get visible faces only
    mus1 = m1.mus > 0
    mus2 = m2.mus > 0
    
    # Calculate target area as 1.5x max of visible cast areas
    # This provides some margin to ensure grid cells are appropriately sized
    target_area = 1.5 * jnp.maximum(
        jnp.max(m1.cast_areas[mus1]),
        jnp.max(m2.cast_areas[mus2])
    )
    
    # Calculate grid spans for range of cell counts
    n_cells_array = jnp.arange(min_cells, max_cells)
    spans = get_grid_spans(m1, m2, n_cells_array)
    
    cell_areas = spans**2
    valid_sizes = jnp.where(cell_areas >= target_area)[0]
    
    # Return first valid size, or max_cells if none found
    if valid_sizes.size > 0:
        return n_cells_array[valid_sizes[-1]].item()
    return min_cells


def evaluate_orbit(binary: Binary, time: ArrayLike, n_cells: Optional[int] = None) -> Tuple[Model, Model]:
    """
    Evaluates the orbit of binary components at a specific time.

    This function determines the positions and velocities of the binary components at a given time. If the binary
    is a PhoebeBinary, it constructs the models for the primary and secondary components based on the PhoebeConfig
    and the specified time. Otherwise, it calculates the orbit by interpolating positions and velocities for the
    given time and resolving any occlusions between the components.

    Args:
        binary (Binary): The binary system to evaluate. Can be a general Binary or a PhoebeBinary.
        time (ArrayLike): The time at which to evaluate the orbit. Can be a single value or an array of values.
        n_cells (int, optional): The number of cells to use for occlusion grid construction. The grid makes the computation faster. Defaults to 20.

    Returns:
        Tuple[Model, Model]: A tuple containing the evaluated models for the primary and secondary components of the binary.

    Note:
        If `binary` is an instance of PhoebeBinary, the function directly uses the PhoebeConfig to construct the models
        without further orbit evaluation. For general Binary instances, it constructs an occlusion grid and evaluates
        the orbit using interpolation and occlusion resolution.
    """
    if isinstance(binary, PhoebeBinary):
        return (PhoebeModel.construct(binary.phoebe_config, time, binary.parameter_labels, binary.parameter_values, Component.PRIMARY),
                PhoebeModel.construct(binary.phoebe_config, time, binary.parameter_labels, binary.parameter_values, Component.SECONDARY))
    else:
        optimal_size = get_optimal_grid_size(binary.body1, binary.body2)
        if n_cells is None:
            n_cells = int(optimal_size)
        else:
            span_size = get_grid_spans(binary.body1, binary.body2, jnp.array([n_cells]))
            if span_size < 1.5 * jnp.maximum(
                jnp.max(binary.body1.cast_areas[binary.body1.mus>0]),
                jnp.max(binary.body2.cast_areas[binary.body2.mus>0])
            ):
                warnings.warn(f"Grid size {n_cells} is too small for the given cast areas. Optimal size is {optimal_size}.")
            
        if len(binary.evaluated_times) == 0:
            raise ValueError(
                "Binary object does not have orbit information. Please add orbit information using add_orbit.")
        grid = Grid.construct(binary.body1, binary.body2, n_cells)
        return _evaluate_orbit(binary, time, grid)


def evaluate_orbit_at_times(binary: Binary, times: ArrayLike, n_cells: Optional[int] = None) -> Tuple[Model, Model]:
    """
    Evaluates the orbit of binary components at multiple specific times.

    This function leverages vectorized computation to evaluate the positions and velocities of the binary components
    at an array of given times. It constructs a grid for occlusion detection and resolution, then iteratively evaluates
    the orbit for each time point specified in the `times` array. This is particularly useful for simulations or
    animations where the orbit needs to be computed at several discrete time points.

    Args:
        binary (Binary): The binary system to evaluate. Can be a general Binary or a PhoebeBinary.
        times (ArrayLike): An array of times at which to evaluate the orbit. Each time should correspond to a specific
            point in the orbit where the positions and velocities of the binary components are desired.
        n_cells (int, optional): The number of cells to use for the occlusion grid construction. This grid is used to
            detect and resolve occlusions between the binary components. Defaults to an optimal size.

    Returns:
        Tuple[Model, Model]: A tuple containing arrays of evaluated models for the primary and secondary components of
            the binary at each time point in `times`. Each model in the tuple provides the positions and velocities of
            the binary components at the corresponding time.

    Note:
        This function is optimized for performance by using JAX's vectorized map (`vmap`) and just-in-time (`jit`)
        compilation features, enabling efficient computation over arrays of times.
    """
    
    if isinstance(binary, PhoebeBinary):
        return [PhoebeModel.construct(binary.phoebe_config, t, binary.parameter_labels, binary.parameter_values, Component.PRIMARY) for t in times], \
               [PhoebeModel.construct(binary.phoebe_config, t, binary.parameter_labels, binary.parameter_values, Component.SECONDARY) for t in times]
    else:
        optimal_size = get_optimal_grid_size(binary.body1, binary.body2)
        print("Optimal grid size: ", optimal_size)
        if n_cells is None:
            n_cells = int(optimal_size)
        else:
            print("n_cells: ", n_cells)
            span_size = get_grid_spans(binary.body1, binary.body2, jnp.array([n_cells]))
            if span_size < 1.5 * jnp.maximum(
                jnp.max(binary.body1.cast_areas[binary.body1.mus>0]),
                jnp.max(binary.body2.cast_areas[binary.body2.mus>0])
            ):
                warnings.warn(f"Grid size {n_cells} is too small for the given cast areas. Optimal size is {optimal_size}.")
        grid = Grid.construct(binary.body1, binary.body2, n_cells)
    return v_evaluate_orbit(binary, times, grid)
