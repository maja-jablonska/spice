import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from typing import Callable
from models import MeshModel
import math
from functools import partial
#from .spectrum_transformer import flux


DEFAULT_CHUNK_SIZE: int = 1024
C: float = 299792.458 #km/s

apply_vrad = lambda x, vrad: x*(vrad/C + 1)
# Docelowo: vrad w log
apply_vrad_log = lambda x, vrad: x+jnp.log10(vrad/C + 1)
v_apply_vrad = jax.jit(jax.vmap(apply_vrad, in_axes=(None, 0)))

# n_wavelengths, n_vertices
v_apply_vrad_log = jax.jit(jax.vmap(apply_vrad_log, in_axes=(None, 0), out_axes=1))


@partial(jax.jit, static_argnums=(0, 6))
def spectrum_flash_sum(intensity_fn,
                       log_wavelengths,
                       areas,
                       mus,
                       vrads,
                       parameters,
                       chunk_size: int = 256):
    # Z każdym elementem powierzchni przekazujemy
    # wektor jego wartości (mu, przyspieszenie, itd.)
    # Część wartości będzie przekazywana do modelu
    
    # Just the 1D case for now
    n_areas = areas.shape[0]
    mus_flattened = mus.flatten()
    vrads_flattened = vrads.flatten()

    v_intensity = jax.jit(jax.vmap(lambda wv, m, p:
                                   jax.vmap(intensity_fn,
                                            in_axes=(0, 0, 0))(wv, m, p),
                                    in_axes=(0, None, None)))

    @partial(jax.checkpoint, prevent_cse=False)
    def chunk_scanner(carries, _):
        chunk_idx, atmo_sum, chunk_sum = carries
        k_chunk_sizes = min(chunk_size, n_areas)

        # (CHUNK_SIZE, 1)
        a_chunk = lax.dynamic_slice(areas,
                                    (chunk_idx,),
                                    (k_chunk_sizes,))
        
        # (CHUNK_SIZE, 1)
        m_chunk = lax.dynamic_slice(mus_flattened,
                                    (chunk_idx,),
                                    (k_chunk_sizes,))
        
        # (CHUNK_SIZE, 1)
        vrad_chunk = lax.dynamic_slice(vrads_flattened,
                                        (chunk_idx,),
                                        (k_chunk_sizes,))
        # (CHUNK_SIZE, 20)
        p_chunk = lax.dynamic_slice(parameters,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, 20))
    
        # Shape: (CHUNK_SIZE, n_wavelengths)
        shifted_log_wavelengths = v_apply_vrad_log(log_wavelengths, vrad_chunk)
        
        # atmosphere_mul is the spectrum simulated for the corresponding wavelengths and optionally given parameters of mu, logg, and T.
        # It is then multiplied by the observed area to scale the contributions of spectra chunks
        
        # Shape: (n_vertices, 2, n_wavelengths)
        # Areas should be rescaled by mus
        # 2 corresponds to the two components: continuum and full spectrum with lines
        # n_wavelengths, n_verices, 2 (continuum+spectrum), 1
        atmosphere_mul = jnp.multiply(
                m_chunk[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]*a_chunk[jnp.newaxis, :, jnp.newaxis, jnp.newaxis], # Czemy nie 2D? Broadcastowanie?
                v_intensity(shifted_log_wavelengths, # (n,)
                            m_chunk,
                            p_chunk))
        
        new_atmo_sum = atmo_sum + jnp.sum(atmosphere_mul, axis=1)#/jnp.sum(a_chunk, axis=0)
        new_chunk_sum = chunk_sum + jnp.sum(m_chunk*a_chunk, axis=0)
        
        return (chunk_idx + k_chunk_sizes, new_atmo_sum, new_chunk_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out, areas), lse = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((log_wavelengths.shape[-1], 2, 1)), jnp.zeros(1,)),
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
                              m.mus,
                              m.los_velocities,
                              m.parameters,
                              chunk_size)