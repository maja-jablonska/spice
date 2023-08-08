import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_model import MeshModel
from .utils import vertex_to_polar


def __cos_law(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle angular distance between two points
    on the sphere (specified in radians)

    All args must be of equal length.    

    """
    lat1 = jnp.pi/2 - lat1
    lat2 = jnp.pi/2 - lat2
    a = jnp.sin(lat1)*jnp.sin(lat2) + jnp.cos(lat1) * jnp.cos(lat2) * jnp.cos(lon2 - lon1)

    return jnp.arccos(a)


def __overabundance_spot_params(theta, phi, abun, abun_bg, radius, radius_factor, coord):
    c_th, c_ph = vertex_to_polar(coord)
    dist_rad = __cos_law(theta, phi, c_th-jnp.pi/2, c_ph-jnp.pi)
    angles = jnp.array([0,radius*radius_factor,radius,jnp.pi])
    abuns = jnp.array([abun, abun, abun_bg, abun_bg])
    y = jnp.interp(dist_rad, angles, abuns, abun, abun_bg)
    return y


v_spot = jax.vmap(__overabundance_spot_params, in_axes=(None, None, None, None, None, None, 0))


@jax.jit
def add_spot(mesh: MeshModel,
             theta: float,
             phi: float,
             param_delta: float,
             radius: float,
             radius_factor: float,
             param_index: int) -> MeshModel:
    spot_parameters = v_spot(theta, phi, param_delta, 0., radius, radius_factor, mesh.centers)
    return mesh._replace(parameters = mesh.parameters.at[:, param_index].set(
        mesh.parameters[:, param_index]+spot_parameters))


def add_spots(mesh: MeshModel,
              thetas: ArrayLike,
              phis: ArrayLike,
              param_deltas: ArrayLike,
              radii: ArrayLike,
              radius_factors: ArrayLike,
              param_indices: ArrayLike) -> MeshModel:
    def scan(carry, params):
        return add_spot(carry, *params[:-1], params[-1].astype(int)), params
    
    updated_mesh, _ = jax.lax.scan(scan, mesh, jnp.vstack([thetas, phis, param_deltas, radii, radius_factors, param_indices]).T)
    return updated_mesh
