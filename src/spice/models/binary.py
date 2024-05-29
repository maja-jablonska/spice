import jax.numpy as jnp
from functools import partial
import jax
from jax.typing import ArrayLike
from typing import NamedTuple, Tuple
from .mesh_model import MeshModel
from .mesh_transform import transform, evaluate_body_orbit
import astropy.units as u
from .orbit_utils import get_orbit_jax
from .mesh_view import resolve_occlusion, Grid


YEAR_TO_SECONDS = (u.year).to(u.s)


class Binary(NamedTuple):
    body1: MeshModel
    body2: MeshModel

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
    def from_bodies(cls, body1: MeshModel, body2: MeshModel) -> "Binary":
          """Construct a Binary object from two mesh models.

          Args:
              body1 (MeshModel):
              body2 (MeshModel):

          Returns:
              Binary: a binary consisting of body1 and body2
          """
          return cls(body1, body2, 1., 0., 0., 0., 0., 0., 0., jnp.zeros_like(body1.centers), jnp.zeros_like(body2.centers), jnp.zeros_like(body1.velocities), jnp.zeros_like(body2.velocities))



@partial(jax.jit, static_argnums=(7,))
def add_orbit(binary: Binary, P: float, ecc: float,
              T: float, i: float, omega: float, Omega: float, orbit_resolution_points: int) -> Binary:
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

      orbit_resolution_times = jnp.linspace(0, P, orbit_resolution_points)
      orbit = get_orbit_jax(orbit_resolution_times, binary.body1.mass, binary.body2.mass, P,
                        ecc, T, i, omega, Omega)
      return binary._replace(P=P, ecc=ecc, T=T, i=i, omega=omega, Omega=Omega,
                              evaluated_times=orbit_resolution_times,
                              body1_centers=orbit[2, :, :], body2_centers=orbit[4, :, :],
                              body1_velocities=orbit[3, :, :], body2_velocities=orbit[5, :, :])


@partial(jax.jit, static_argnums=(2,))
def _evaluate_orbit(binary: Binary, time: ArrayLike, grid: Grid) -> Tuple[MeshModel, MeshModel]:
      interpolate_orbit = jax.jit(jax.vmap(lambda x: jnp.interp(time, binary.evaluated_times, x, period=binary.P), in_axes=(0,)))
      body1_center = interpolate_orbit(binary.body1_centers)
      body2_center = interpolate_orbit(binary.body2_centers)
      body1_velocity = interpolate_orbit(binary.body1_velocities)
      body2_velocity = interpolate_orbit(binary.body2_velocities)

      body1 = evaluate_body_orbit(transform(binary.body1, binary.body1.center+body1_center), body1_velocity)
      body2 = evaluate_body_orbit(transform(binary.body2, binary.body2.center+body2_center), body2_velocity)
      
      return jax.lax.cond(jnp.mean(body1.los_z)>jnp.mean(body2.los_z),
                          lambda: (body1, resolve_occlusion(body2, body1, grid)),
                          lambda: (resolve_occlusion(body1, body2, grid), body2))
      
      
v_evaluate_orbit = jax.jit(jax.vmap(_evaluate_orbit, in_axes=(None, 0, None)), static_argnums=(2,))


def evaluate_orbit(binary: Binary, time: ArrayLike, n_cells: int = 20) -> Tuple[MeshModel, MeshModel]:
      """_summary_

      Args:
          binary (Binary): a binary object
          time (ArrayLike): time to resolve the orbit at
          n_cells (int, optional): number of grid cells to divide the calculation of occlusion to. Defaults to 20.

      Returns:
          Tuple[MeshModel, MeshModel]: (body1, body2)
      """
      grid = Grid.construct(binary.body1, binary.body2, n_cells)
      return _evaluate_orbit(binary, time, grid)


def evaluate_orbit_at_times(binary: Binary, times: ArrayLike, n_cells: int = 20) -> Tuple[MeshModel, MeshModel]:
      """_summary_

      Args:
          binary (Binary): a binary object
          times (ArrayLike): times to resolve the orbit at
          n_cells (int, optional): number of grid cells to divide the calculation of occlusion to. Defaults to 20.

      Returns:
          Tuple[MeshModel, MeshModel]: (body1, body2)
      """
      grid = Grid.construct(binary.body1, binary.body2, n_cells)
      return v_evaluate_orbit(binary, times, grid)
