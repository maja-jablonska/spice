import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from jax.scipy.integrate import trapezoid
from typing import Callable, List
from spice.models import MeshModel
import math
from functools import partial
from spice.spectrum.utils import ERG_S_TO_W, intensity_wavelengths_to_hz, H_CONST_ERG_S, JY_TO_ERG
from spice.spectrum.filter import Filter


DEFAULT_CHUNK_SIZE: int = 1024
C: float = 299792.458 # km/s

apply_vrad = lambda x, vrad: x*(vrad/C + 1)
# Docelowo: vrad w log
apply_vrad_log = lambda x, vrad: x+jnp.log10(vrad/C + 1)
v_apply_vrad = jax.jit(jax.vmap(apply_vrad, in_axes=(None, 0)))

# n_wavelengths, n_vertices
v_apply_vrad_log = jax.jit(jax.vmap(apply_vrad_log, in_axes=(None, 0), out_axes=0))


@partial(jax.jit, static_argnums=(0, 6))
def __spectrum_flash_sum(intensity_fn,
                        log_wavelengths,
                        areas,
                        mus,
                        vrads,
                        parameters,
                        chunk_size: int):
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
        chunk_idx, atmo_sum = carries
        
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
                (m_chunk*a_chunk)[:, jnp.newaxis, jnp.newaxis],
                v_in)
        
        
        new_atmo_sum = atmo_sum + jnp.sum(atmosphere_mul, axis=0)
        
        return (chunk_idx + k_chunk_sizes, new_atmo_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out), _ = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((log_wavelengths.shape[-1], 2))),
        xs=None,
        length=math.ceil(n_areas/chunk_size))
    return out


@partial(jax.jit, static_argnums=(1,))
def _adjust_dim(x: ArrayLike, chunk_size: int) -> ArrayLike:
    return jnp.concatenate([x, jnp.zeros((chunk_size-x.shape[0]%chunk_size, *x.shape[1:]) )], axis=0)


@partial(jax.jit, static_argnums=(0, 3))
def simulate_spectrum(intensity_fn: Callable[[ArrayLike, float, ArrayLike], ArrayLike],
                      m: MeshModel,
                      log_wavelengths: ArrayLike,
                      chunk_size: int = DEFAULT_CHUNK_SIZE):
    
    return __spectrum_flash_sum(intensity_fn,
                                log_wavelengths,
                                _adjust_dim(jnp.where(m.mus>0, m.cast_areas, 0.), chunk_size),
                                _adjust_dim(jnp.where(m.mus>0, m.mus, 0.), chunk_size),
                                _adjust_dim(m.los_velocities, chunk_size),
                                _adjust_dim(m.parameters, chunk_size),
                                chunk_size)
    

@partial(jax.jit, static_argnums=(0, 5))
def __flux_flash_sum(flux_fn,
                    log_wavelengths,
                    areas,
                    vrads,
                    parameters,
                    chunk_size: int):
    '''
        Each surface element has a vector of parameters (mu, LOS velocity, etc)
        Some of these parameters are the flux model's input
    '''
    
    # Just the 1D case for now
    n_areas = areas.shape[0]
    n_parameters = parameters.shape[-1]

    v_flux = jax.vmap(flux_fn, in_axes=(0, 0))

    @partial(jax.checkpoint, prevent_cse=False)
    def chunk_scanner(carries, _):
        chunk_idx, atmo_sum = carries
        
        k_chunk_sizes = min(chunk_size, n_areas)

        # (CHUNK_SIZE, 1)
        a_chunk = lax.dynamic_slice(areas,
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
        v_in = v_flux(shifted_log_wavelengths, # (n,)
                      p_chunk)
        atmosphere_mul = jnp.multiply(
                a_chunk[:, jnp.newaxis, jnp.newaxis], # Czemy nie 2D? Broadcastowanie?
                v_in)
        
        
        new_atmo_sum = atmo_sum + jnp.sum(atmosphere_mul, axis=0)
        
        return (chunk_idx + k_chunk_sizes, new_atmo_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out), _ = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((log_wavelengths.shape[-1], 2))),
        xs=None,
        length=math.ceil(n_areas/chunk_size))
    return out

@partial(jax.jit, static_argnums=(0, 3))
def simulate_total_flux(flux_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
                        m: MeshModel,
                        log_wavelengths: ArrayLike,
                        chunk_size: int = DEFAULT_CHUNK_SIZE):
    return __flux_flash_sum(flux_fn,
                            log_wavelengths,
                            _adjust_dim(m.areas, chunk_size),
                            _adjust_dim(m.los_velocities, chunk_size),
                            _adjust_dim(m.parameters, chunk_size),
                            chunk_size)

@partial(jax.jit, static_argnums=(0, 3))
def luminosity(flux_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
               model: MeshModel,
               wavelengths: ArrayLike,
               chunk_size: int = DEFAULT_CHUNK_SIZE) -> ArrayLike:
    """Calculate the bolometric luminsity of the model.

    Args:
        flux_fn (Callable[[ArrayLike, ArrayLike], ArrayLike]):
        model (MeshModel):
        wavelengths (ArrayLike): wavelengths [Angstrom]
        chunk_size (int, optional): size of the chunk in the GPU memory. Defaults to 1024.

    Returns:
        ArrayLike: _description_
    """
    flux = simulate_total_flux(flux_fn, model, jnp.log10(wavelengths), chunk_size)
    return trapezoid(y=flux[:, 0], x=wavelengths*1e-8)


@jax.jit
def filter_responses(wavelengths: ArrayLike, sample_wavelengths: ArrayLike, sample_responses: ArrayLike) -> ArrayLike:
    return jnp.interp(wavelengths, sample_wavelengths, sample_responses)


@partial(jax.jit, static_argnums=(0,))
def AB_passband_luminosity(filter: Filter,
                           wavelengths: ArrayLike,
                           intensity: ArrayLike,
                           distance: float = 3.085677581491367e+19) -> ArrayLike:
    """Return the passband luminosity in a given filter.

    Args:
        filter (Filter):
        wavelengths (ArrayLike): wavelengths [Angstrom]
        intensity (ArrayLike): intensity [erg/s/cm^3]
        distance (float, optional): distance in cm. Defaults to 3.08e+19 (10 parsecs in cm)

    Returns:
        ArrayLike: passband luminosity [mag]
    """
    vws_hz, intensity_hz = intensity_wavelengths_to_hz(wavelengths, intensity)
    transmission_responses = filter.filter_responses_for_frequencies(vws_hz)
    return -2.5*jnp.log10(trapezoid(x=vws_hz, y=intensity_hz*JY_TO_ERG*transmission_responses/jnp.power(distance, 2)/(H_CONST_ERG_S*vws_hz))/
                          trapezoid(x=vws_hz, y=filter.ab_zeropoint*transmission_responses/(H_CONST_ERG_S*vws_hz)))


@jax.jit
def absolute_bol_luminosity(luminosity: ArrayLike) -> ArrayLike:
    """Calculate bolometric absolute luminosity

    Args:
        luminosity (ArrayLike): total bolometric luminosity [erg/s]

    Returns:
        ArrayLike: absolute luminosity [mag]
    """
    return -2.5*jnp.log10(luminosity*ERG_S_TO_W)+71.1974
