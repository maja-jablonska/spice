import warnings
import jax.numpy as jnp
from functools import partial
import jax
from jax.typing import ArrayLike
from typing import List, NamedTuple, Optional, Tuple, Dict, Union, Any
import numpy as np
from jaxtyping import Float, Array
from collections import namedtuple
from jax import tree_util

from spice.models.model import Model
from spice.models.mesh_transform import transform, evaluate_body_orbit
from spice.models.orbit_utils import get_orbit_jax
from spice.models.mesh_view_kdtree import get_optimal_search_radius, resolve_occlusion

# Make PHOEBE-related imports optional
try:
    from spice.models.phoebe_model import DAY_TO_S, PhoebeModel
    from spice.models.phoebe_utils import Component, PhoebeConfig
    PHOEBE_AVAILABLE = True
except ImportError:
    PHOEBE_AVAILABLE = False

YEAR_TO_SECONDS = 3.154e7
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
    def __new__(cls, *args, **kwargs):
        if not PHOEBE_AVAILABLE:
            raise ImportError("PHOEBE is not installed. Please install it with 'pip install stellar-spice[phoebe]'")
        return super().__new__(cls, *args, **kwargs)

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



@jax.jit
def _evaluate_orbit(binary: Binary, time: ArrayLike, search_radius_factor: float) -> Tuple[Model, Model]:
    """Evaluate the orbit at a specific time.

    Args:
        binary (Binary): Binary to evaluate
        time (ArrayLike): Time at which to evaluate the orbit
        search_radius_factor (float): Search radius factor for KD-tree occlusion detection

    Returns:
        Tuple[Model, Model]: Updated primary and secondary models at the specified time
    """
    body1, body2 = _interpolate_orbit(binary, time)

    # Resolve occlusion using KD-trees
    body1 = resolve_occlusion(body1, body2, search_radius_factor)
    body2 = resolve_occlusion(body2, body1, search_radius_factor)
    
    return body1, body2

@jax.jit
def v_evaluate_orbit(binary: Binary, times: ArrayLike, search_radius_factor: float) -> Tuple[List[Model], List[Model]]:
    """Evaluate the orbit at multiple times.

    Args:
        binary (Binary): Binary to evaluate
        times (ArrayLike): Times at which to evaluate the orbit
        search_radius_factor (float): Search radius factor for KD-tree occlusion detection

    Returns:
        Tuple[List[Model], List[Model]]: Lists of updated primary and secondary models at the specified times,
            where each element corresponds to a model at a specific time
    """
    # Map the evaluation function over each time point
    result_body1, result_body2 = jax.vmap(
        lambda t: _evaluate_orbit(binary, t, search_radius_factor)
    )(times)
    
    # The issue is that jax.lax.map doesn't properly handle the nested structure
    # when resolving occlusions. Using jax.vmap instead ensures proper vectorization.
    
    # Convert the results to lists for each component
    body1_list = jax.tree_util.tree_map(lambda x: list(x), result_body1)
    body2_list = jax.tree_util.tree_map(lambda x: list(x), result_body2)
    
    # Transpose the tree structure to get a list of models
    # The API changed in newer JAX versions - tree_transpose now takes trees as positional args
    return jax.tree_util.tree_transpose(
        outer_treedef=jax.tree_util.tree_structure(binary.body1),
        inner_treedef=jax.tree_util.tree_structure([0 for _ in range(len(times))]),
        pytree_to_transpose=body1_list
    ), jax.tree_util.tree_transpose(
        outer_treedef=jax.tree_util.tree_structure(binary.body2),
        inner_treedef=jax.tree_util.tree_structure([0 for _ in range(len(times))]),
        pytree_to_transpose=body2_list
    )


def get_optimal_kdtree_params(m1, m2, min_radius_factor=1.5, max_radius_factor=5.0):
    """Determine optimal parameters for KD-tree based occlusion detection.
    
    Args:
        m1 (MeshModel): First mesh model
        m2 (MeshModel): Second mesh model
        min_radius_factor (float): Minimum radius factor to consider
        max_radius_factor (float): Maximum radius factor to consider
        
    Returns:
        float: Optimal search radius factor for KD-tree based occlusion detection
    """
    # Get visible areas of both meshes
    m1_visible_areas = m1.cast_areas[m1.mus > 0]
    m2_visible_areas = m2.cast_areas[m2.mus > 0]
    
    # Default to 2.0 if no visible areas
    if m1_visible_areas.shape[0] == 0 or m2_visible_areas.shape[0] == 0:
        return 2.0
    
    radius_factor = get_optimal_search_radius(m1, m2, min_radius_factor, max_radius_factor)
    
    print("Optimal KD-tree search radius factor:", radius_factor)
    return radius_factor


def evaluate_orbit(binary: Binary, time: ArrayLike, search_radius_factor: Optional[float] = None) -> Tuple[Model, Model]:
    """
    Evaluates the orbit of binary components at a specific time.

    This function computes the positions and velocities of the binary components at the specified time.
    It then resolves occlusions between the components using KD-tree based spatial search.

    Args:
        binary (Binary): The binary system to evaluate. Can be a general Binary or a PhoebeBinary.
        time (ArrayLike): The specific time at which to evaluate the orbit. This should be a scalar value
            representing a point in time where the positions and velocities of the binary components are desired.
        search_radius_factor (float, optional): The search radius factor to use for KD-tree based occlusion detection.
            This parameter controls how far to search for potential occluders around each triangle.
            Defaults to an optimal value determined based on the mesh properties.

    Returns:
        Tuple[Model, Model]: A tuple containing the updated models for the primary and secondary components of
            the binary at the specified time. Each model provides the positions and velocities of the
            corresponding binary component at the specified time, with occlusions resolved.

    Note:
        This function is optimized for performance by using JAX's just-in-time (`jit`) compilation feature,
        enabling efficient computation of the orbit evaluation.
    """
    if isinstance(binary, PhoebeBinary):
        return PhoebeModel.construct(binary.phoebe_config, time, binary.parameter_labels, binary.parameter_values, Component.PRIMARY), \
               PhoebeModel.construct(binary.phoebe_config, time, binary.parameter_labels, binary.parameter_values, Component.SECONDARY)
    else:
        if search_radius_factor is None:
            search_radius_factor = get_optimal_search_radius(binary.body1, binary.body2)
            print("Using search radius factor:", search_radius_factor)
        
        return _evaluate_orbit(binary, time, search_radius_factor)


def evaluate_orbit_at_times(binary: Binary, times: ArrayLike, search_radius_factor: Optional[float] = None) -> Tuple[List[Model], List[Model]]:
    """
    Evaluates the orbit of binary components at multiple specific times.

    This function leverages vectorized computation to evaluate the positions and velocities of the binary components
    at an array of given times. It uses KD-tree based spatial search for occlusion detection and resolution, then iteratively evaluates
    the orbit for each time point specified in the `times` array. This is particularly useful for simulations or
    animations where the orbit needs to be computed at several discrete time points.

    Args:
        binary (Binary): The binary system to evaluate. Can be a general Binary or a PhoebeBinary.
        times (ArrayLike): An array of times at which to evaluate the orbit. Each time should correspond to a specific
            point in the orbit where the positions and velocities of the binary components are desired.
        search_radius_factor (float, optional): The search radius factor to use for KD-tree based occlusion detection.
            This parameter controls how far to search for potential occluders around each triangle.
            Defaults to an optimal value determined based on the mesh properties.

    Returns:
        Tuple[List[Model], List[Model]]: A tuple containing lists of evaluated models for the primary and secondary 
            components of the binary at each time point in `times`. Each model in the lists provides the positions 
            and velocities of the binary components at the corresponding time.

    Note:
        This function is optimized for performance by using JAX's vectorized map (`vmap`) and just-in-time (`jit`)
        compilation features, enabling efficient computation over arrays of times.
    """
    
    if isinstance(binary, PhoebeBinary):
        return [PhoebeModel.construct(binary.phoebe_config, t, binary.parameter_labels, binary.parameter_values, Component.PRIMARY) for t in times], \
               [PhoebeModel.construct(binary.phoebe_config, t, binary.parameter_labels, binary.parameter_values, Component.SECONDARY) for t in times]
    else:
        if search_radius_factor is None:
            search_radius_factor = get_optimal_search_radius(binary.body1, binary.body2)
            print("Using search radius factor for KD-tree:", search_radius_factor)
        
        # For proper occlusion detection, we need to evaluate each time point individually
        # rather than using vectorized operations that might not properly handle the occlusion logic
        if len(times) <= 10:  # For small number of times, use individual evaluations for better accuracy
            return [evaluate_orbit(binary, t, search_radius_factor)[0] for t in times], \
                   [evaluate_orbit(binary, t, search_radius_factor)[1] for t in times]
        else:
            # For larger arrays, use vectorized version but with caution about occlusion accuracy
            return v_evaluate_orbit(binary, times, search_radius_factor)
