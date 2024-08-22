import jax.numpy as jnp
from functools import partial
import jax
from jax.typing import ArrayLike
from typing import List, NamedTuple, Tuple, Dict

from spice.models.phoebe_model import DAY_TO_S, PhoebeModel
from spice.models.phoebe_utils import Component, PhoebeConfig
from .model import Model
from .mesh_transform import transform, evaluate_body_orbit
import astropy.units as u
from .orbit_utils import get_orbit_jax
from .mesh_view import resolve_occlusion, Grid
from collections import namedtuple

YEAR_TO_SECONDS = u.year.to(u.s)
DAY_TO_YEAR = 0.0027378507871321013


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
        return cls(body1, body2, 1., 0., 0., 0., 0., 0., 0.,
                   jnp.zeros_like(body1.centers), jnp.zeros_like(body2.centers),
                   jnp.zeros_like(body1.velocities), jnp.zeros_like(body2.velocities))


class PhoebeBinary(namedtuple("PhoebeBinary",
                              ["body1", "body2", "P", "ecc", "T", "i", "omega", "Omega",
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
                                    T=phoebe_config.get_quantity('t0_ref', component='binary') * DAY_TO_YEAR,
                                    i=phoebe_config.get_quantity('incl', component='binary'),
                                    omega=phoebe_config.get_quantity('per0', component='binary'),
                                    Omega=phoebe_config.get_quantity('long_an', component='binary'),
                                    body1_centers=phoebe_config.get_all_orbit_centers(str(Component.PRIMARY)),
                                    body2_centers=phoebe_config.get_all_orbit_centers(str(Component.SECONDARY)),
                                    body1_velocities=phoebe_config.get_all_orbit_velocities(str(Component.PRIMARY)),
                                    body2_velocities=phoebe_config.get_all_orbit_velocities(str(Component.SECONDARY)),
                                    phoebe_config=phoebe_config,
                                    parameter_labels=parameter_labels,
                                    parameter_values=parameter_values
                                    )


@partial(jax.jit, static_argnums=(7,))
def _add_orbit(binary: Binary, P: float, ecc: float,
               T: float, i: float, omega: float, Omega: float, orbit_resolution_points: int) -> Binary:
    orbit_resolution_times = jnp.linspace(0, P, orbit_resolution_points)
    orbit = get_orbit_jax(orbit_resolution_times, binary.body1.mass, binary.body2.mass, P,
                          ecc, T, i, omega, Omega)
    return binary._replace(P=P, ecc=ecc, T=T, i=i, omega=omega, Omega=Omega,
                           evaluated_times=orbit_resolution_times,
                           body1_centers=orbit[2, :, :], body2_centers=orbit[4, :, :],
                           body1_velocities=orbit[3, :, :], body2_velocities=orbit[5, :, :])


def add_orbit(binary: Binary,
              P: float,
              ecc: float,
              T: float,
              i: float,
              omega: float,
              Omega: float,
              orbit_resolution_points: int) -> Binary:
    """Add orbit information to the binary object

      Args:
          binary (Binary):
          P (float): orbit period [years]
          ecc (float): orbit eccentrity
          T (float): [defined time units]
          i (float): inclination [rad]
          omega (float): []
          Omega (float): []
          orbit_resolution_points (int): number of times to resolve the orbit at

      Returns:
          Binary: object with calculated orbit property values
      """
    if isinstance(binary, PhoebeBinary):
        raise ValueError("PhoebeBinary objects are read-only - the orbit information is already added.")
    else:
        return _add_orbit(binary, P, ecc, T, i, omega, Omega, orbit_resolution_points)


@jax.jit
def _interpolate_orbit(binary: Binary, time: ArrayLike) -> Tuple[Model, Model]:
    interpolate_orbit = jax.jit(
        jax.vmap(lambda x: jnp.interp(time, binary.evaluated_times, x, period=binary.P), in_axes=(0,)))
    body1_center = interpolate_orbit(binary.body1_centers)
    body2_center = interpolate_orbit(binary.body2_centers)
    body1_velocity = interpolate_orbit(binary.body1_velocities)
    body2_velocity = interpolate_orbit(binary.body2_velocities)
    body1 = evaluate_body_orbit(transform(binary.body1, binary.body1.center + body1_center), body1_velocity)
    body2 = evaluate_body_orbit(transform(binary.body2, binary.body2.center + body2_center), body2_velocity)

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


def evaluate_orbit(binary: Binary, time: ArrayLike, n_cells: int = 20) -> Tuple[Model, Model]:
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
        if len(binary.evaluated_times) == 0:
            raise ValueError(
                "Binary object does not have orbit information. Please add orbit information using add_orbit.")
        grid = Grid.construct(binary.body1, binary.body2, n_cells)
        return _evaluate_orbit(binary, time, grid)


def evaluate_orbit_at_times(binary: Binary, times: ArrayLike, n_cells: int = 20) -> Tuple[Model, Model]:
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
            detect and resolve occlusions between the binary components. Defaults to 20.

    Returns:
        Tuple[Model, Model]: A tuple containing arrays of evaluated models for the primary and secondary components of
            the binary at each time point in `times`. Each model in the tuple provides the positions and velocities of
            the binary components at the corresponding time.

    Note:
        This function is optimized for performance by using JAX's vectorized map (`vmap`) and just-in-time (`jit`)
        compilation features, enabling efficient computation over arrays of times.
    """
    grid = Grid.construct(binary.body1, binary.body2, n_cells)
    return v_evaluate_orbit(binary, times, grid)
