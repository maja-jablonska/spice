import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_model import MeshModel
from .utils import mesh_polar_vertices, spherical_harmonic, vertex_to_polar


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
    """Add a spot to the mesh model.

    Args:
        mesh (MeshModel): mesh model
        theta (float): theta coordinate of the spot center
        phi (float): phi coordinate of the spot center
        param_delta (float): difference in the parameter value between the spot and the background
        radius (float): spot radius in radians
        radius_factor (float): factor of the difference gradient on the spot edge
        param_index (int): index of the parameter in the parameters array

    Returns:
        MeshModel: mesh with the spot added
    """
    spot_parameters = v_spot(theta, phi, param_delta, 0., radius, radius_factor, mesh.centers)
    return mesh._replace(parameters = mesh.parameters.at[:, param_index].set(
        mesh.parameters[:, param_index]+spot_parameters))
    
    
@jax.jit
def add_spherical_harmonic_spot(mesh: MeshModel,
                                m: int, n: int,
                                param_delta: float,
                                param_index: int) -> MeshModel:
    """Add a spot/parameter variation as a spherical harmonic to the mesh model.

    Args:
        mesh (MeshModel): mesh model
        m (int): m index of the spherical harmonic
        n (int): n index of the spherical harmonic
        param_delta (float): difference in the parameter value between the spot and the background
        param_index (int): index of the parameter in the parameters array

    Returns:
        MeshModel: mesh with the spot added
    """
    center_polar_coords = mesh_polar_vertices(mesh.centers)
    spot_parameters = spherical_harmonic(m, n, center_polar_coords) * param_delta
    return mesh._replace(parameters = mesh.parameters.at[:, param_index].set(
        mesh.parameters[:, param_index]+spot_parameters[:, jnp.newaxis]))


def add_spots(mesh: MeshModel,
              thetas: ArrayLike,
              phis: ArrayLike,
              param_deltas: ArrayLike,
              radii: ArrayLike,
              radius_factors: ArrayLike,
              param_indices: ArrayLike) -> MeshModel:
    """Add multiple spots to the mesh model

    Args:
        mesh (MeshModel): mesh model
        thetas (ArrayLike): theta coordinates of the spots centers
        phis (ArrayLike): phi coordinates of the spots centers
        param_deltas (ArrayLike): difference in the parameter valuea between the spota and the background
        radii (ArrayLike): spot radii in radians
        radius_factors (ArrayLike): factors of the difference gradients on the spot edges
        param_indices (ArrayLike): indices of the parameters in the parameters array

    Returns:
        MeshModel: mesh with the spots added
    """
    def scan(carry, params):
        return add_spot(carry, *params[:-1], params[-1].astype(int)), params
    
    updated_mesh, _ = jax.lax.scan(scan, mesh, jnp.vstack([thetas, phis, param_deltas, radii, radius_factors, param_indices]).T)
    return updated_mesh


@jax.jit
def add_spherical_harmonic_spots(mesh: MeshModel,
                                 m: ArrayLike, n: ArrayLike,
                                 param_deltas: ArrayLike,
                                 param_indices: ArrayLike) -> MeshModel:
    """Add a spot/parameter variation as a spherical harmonic to the mesh model.

    Args:
        mesh (MeshModel): mesh model
        m (ArrayLike): m indices of the spherical harmonic
        n (ArrayLike): n indices of the spherical harmonic
        param_deltas (ArrayLike): difference in the parameter values between the spot and the background
        param_indices (ArrayLike): indices of the parameter in the parameters array

    Returns:
        MeshModel: mesh with the spots added
    """
    def scan(carry, params):
        return add_spherical_harmonic_spot(
            carry, params[0].astype(int), params[1].astype(int), params[2], params[3].astype(int)
            ), params
        
    updated_mesh, _ = jax.lax.scan(scan, mesh, jnp.vstack([m, n, param_deltas, param_indices]).T)
    return updated_mesh
