import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from typing import Callable, List
from models import MeshModel
import math
from functools import partial
from .utils import ERG_S_TO_W, SPHERE_STERADIAN


DEFAULT_CHUNK_SIZE: int = 1024
C: float = 299792.458 # km/s

apply_vrad = lambda x, vrad: x*(vrad/C + 1)
# Docelowo: vrad w log
apply_vrad_log = lambda x, vrad: x+jnp.log10(vrad/C + 1)
v_apply_vrad = jax.jit(jax.vmap(apply_vrad, in_axes=(None, 0)))

# n_wavelengths, n_vertices
v_apply_vrad_log = jax.jit(jax.vmap(apply_vrad_log, in_axes=(None, 0), out_axes=0))


@partial(jax.jit, static_argnums=(0, 6))
def spectrum_flash_sum(intensity_fn,
                       log_wavelengths,
                       areas,
                       mus,
                       vrads,
                       parameters,
                       chunk_size: int = 256):
    '''
        Each surface element has a vector of parameters (mu, LOS velocity, etc)
        Some of these parameters are the flux model's input
    '''
    
    # Just the 1D case for now
    n_areas = areas.shape[0]
    n_parameters = parameters.shape[-1]

    v_intensity = jax.vmap(intensity_fn, in_axes=(0, 0, 0))

    @partial(jax.checkpoint, prevent_cse=False)
    def chunk_scanner(carries, _):
        chunk_idx, atmo_sum, chunk_sum = carries
        
        k_chunk_sizes = min(chunk_size, n_areas)

        # (CHUNK_SIZE, 1)
        a_chunk = lax.dynamic_slice(areas,
                                    (chunk_idx,),
                                    (k_chunk_sizes,))
        
        # (CHUNK_SIZE, 1)
        m_chunk = lax.dynamic_slice(mus,
                                    (chunk_idx,),
                                    (k_chunk_sizes,))
        
        # (CHUNK_SIZE, 1)
        vrad_chunk = lax.dynamic_slice(vrads,
                                        (chunk_idx,),
                                        (k_chunk_sizes,))
        # (CHUNK_SIZE, n_parameters)
        p_chunk = lax.dynamic_slice(parameters,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_parameters))
    
        # Shape: (CHUNK_SIZE, log_wavelengths)
        shifted_log_wavelengths = v_apply_vrad_log(log_wavelengths, vrad_chunk)
        
        # atmosphere_mul is the spectrum simulated for the corresponding wavelengths and optionally given parameters of mu, logg, and T.
        # It is then multiplied by the observed area to scale the contributions of spectra chunks
        
        # Shape: (n_vertices, 2, n_wavelengths)
        # Areas should be rescaled by mus
        # 2 corresponds to the two components: continuum and full spectrum with lines
        # n_wavelengths, n_verices, 2 (continuum+spectrum), 1
        
        # shifted_log_wavelengths (CHUNK_SIZE, n_wavelengths)
        # m_chunk (CHUNK_SIZE)
        # p_chunk (CHUNK_SIZE, n_parameters)
        v_in = v_intensity(shifted_log_wavelengths, # (n,)
                            m_chunk[:, jnp.newaxis],
                            p_chunk)
        atmosphere_mul = jnp.multiply(
                (m_chunk*a_chunk)[:, jnp.newaxis, jnp.newaxis], # Czemy nie 2D? Broadcastowanie?
                v_in)
        
        
        new_atmo_sum = atmo_sum + jnp.sum(atmosphere_mul, axis=0)#/jnp.sum(a_chunk, axis=0)
        new_chunk_sum = chunk_sum + jnp.sum(m_chunk*a_chunk, axis=0)
        
        return (chunk_idx + k_chunk_sizes, new_atmo_sum, new_chunk_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out, areas), _ = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((log_wavelengths.shape[-1], 2)), jnp.zeros(1,)),
        xs=None,
        length=math.ceil(n_areas/chunk_size))
    return (out/areas).reshape(-1, 2)


@partial(jax.jit, static_argnums=(0, 3))
def simulate_spectrum(intensity_fn: Callable[[float, float, ArrayLike], ArrayLike],
                      m: MeshModel,
                      log_wavelengths: ArrayLike,
                      chunk_size: int = DEFAULT_CHUNK_SIZE):
    return spectrum_flash_sum(intensity_fn,
                              log_wavelengths,
                              m.areas,
                              jnp.where(m.mus>0, m.mus, 0.),
                              m.los_velocities,
                              m.parameters,
                              chunk_size)


@jax.jit
def luminosity(spectrum: ArrayLike, wavelengths: ArrayLike, mesh: MeshModel) -> ArrayLike:
    """Calculate total luminosity output

    Args:
        spectrum (ArrayLike): spectrum in erg/s/cm^2/A
        wavelengths (ArrayLike): wavelengths in A
        mesh (MeshModel): mesh model

    Returns:
        ArrayLike: luminosity in erg/s
    """
    half_surface_area = 2*jnp.pi*jnp.power(mesh.radius, 2)
    luminosity = jnp.trapz(jnp.nan_to_num(spectrum[:, 0]),
                           wavelengths)*SPHERE_STERADIAN/2*half_surface_area
    return luminosity


@jax.jit
def filter_responses(wavelengths: ArrayLike, sample_wavelengths: ArrayLike, sample_responses: ArrayLike) -> ArrayLike:
    return jnp.interp(wavelengths, sample_wavelengths, sample_responses)


# TODO: debug
@jax.jit
def passband_luminosity(spectrum: ArrayLike, filter_responses: ArrayLike,
                        wavelengths: ArrayLike, mesh: MeshModel) -> ArrayLike:
    half_surface_area = 2*jnp.pi*jnp.power(mesh.radius, 2)
    luminosity = jnp.trapz(jnp.nan_to_num(spectrum[:, 0])*filter_responses,
                           wavelengths)*SPHERE_STERADIAN/2*half_surface_area
    return luminosity


@jax.jit
def absolute_bol_luminosity(luminosity: ArrayLike) -> ArrayLike:
    """Calculate bolometric absolute luminosity

    Args:
        luminosity (ArrayLike): total bolometric luminosity in erg/s

    Returns:
        ArrayLike: total luminosity in magnitude
    """
    return -2.5*jnp.log10(luminosity*ERG_S_TO_W)+71.1974


class BaseSpectrum:
    @staticmethod
    def get_label_names() -> List[str]:
        return []
    
    @staticmethod
    def is_in_bounds(parameters: ArrayLike) -> bool:
        return True
    
    @staticmethod
    def get_default_parameters() -> ArrayLike:
        return jnp.array([])
    
    @staticmethod
    def to_parameters() -> ArrayLike:
        return jnp.array([])
    
    @staticmethod
    def flux_method() -> Callable[..., ArrayLike]:
        return lambda x: x
