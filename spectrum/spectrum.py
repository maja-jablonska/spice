import jax
import jax.numpy as jnp
from jax import lax
import math
from functools import partial
from .spectrum_transformer import flux


DEFAULT_CHUNK_SIZE: int = 1024
C: float = 299792.458 #km/s

apply_vrad = lambda x, vrad: x*(vrad/C + 1)
# Docelowo: vrad w log
apply_vrad_log = lambda x, vrad: x+jnp.log10(vrad/C + 1)
v_apply_vrad = jax.jit(jax.vmap(apply_vrad, in_axes=(None, 0)))
v_apply_vrad_log = jax.jit(jax.vmap(apply_vrad_log, in_axes=(None, 0)))

atmosphere_flux = jax.jit(jax.vmap(flux, in_axes=(0, 0, None)))

def spectrum_flash_sum(log_wavelengths,
                       areas,
                       mus,
                       vrads,
                       parameters,
                       chunk_size: int = DEFAULT_CHUNK_SIZE):
    # Z każdym elementem powierzchni przekazujemy
    # wektor jego wartości (mu, przyspieszenie, itd.)
    # Część wartości będzie przekazywana do modelu
    
    # Just the 1D case for now
    n_areas, n_samples = areas.shape
    mus_flattened = mus.reshape(areas.shape)
    vrads_flattened = vrads.reshape(areas.shape)
    points = log_wavelengths.shape[0]

    @partial(jax.checkpoint, prevent_cse=False)
    def chunk_scanner(carries, _):
        chunk_idx, atmo_sum, chunk_sum = carries
        k_chunk_sizes = min(chunk_size, n_areas)

        # (CHUNK_SIZE, 1)
        a_chunk = lax.dynamic_slice(areas,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        
        # (CHUNK_SIZE, 1)
        m_chunk = lax.dynamic_slice(mus_flattened,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        
        # (CHUNK_SIZE, 1)
        vrad_chunk = lax.dynamic_slice(vrads_flattened,
                                        (chunk_idx, 0),
                                        (k_chunk_sizes, n_samples))
        
        # Shape: (CHUNK_SIZE, n_wavelengths)
        shifted_log_wavelengths = v_apply_vrad_log(log_wavelengths, vrad_chunk)
        
        # atmosphere_mul is the spectrum simulated for the corresponding wavelengths and optionally given parameters of mu, logg, and T.
        # It is then multiplied by the observed area to scale the contributions of spectra chunks
        
        # Shape: (n_vertices, 2, n_wavelengths)
        # Areas should be rescaled by mus
        
        # 2 corresponds to the two components: continuum and full spectrum with lines
        atmosphere_mul = jnp.multiply(
            a_chunk.reshape((-1, 1, 1)), # Czemy nie 2D? Broadcastowanie?
            atmosphere_flux(shifted_log_wavelengths, # (n,)
                            m_chunk,
                            parameters))
        
        new_atmo_sum = atmo_sum + jnp.sum(atmosphere_mul, axis=0)#/jnp.sum(a_chunk, axis=0)
        new_chunk_sum = chunk_sum + jnp.sum(a_chunk, axis=0)
        
        return (chunk_idx + k_chunk_sizes, new_atmo_sum, new_chunk_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out, areas), lse = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((2, log_wavelengths.shape[-1])), jnp.zeros(1,)),
        xs=None,
        length=math.ceil(n_areas/chunk_size))
    return out/areas


def get_spectra_flash_sum(CHUNK_SIZE: int = 256):
    return jax.jit(
        jax.vmap(lambda a, b, c, d, e: spectrum_flash_sum(a, b, c, d, e, CHUNK_SIZE),
                 in_axes=(None, 0, 0, 0, 0)))
