import jax
import jax.numpy as jnp
from jax import lax
import math
from functools import partial
from spectrum_transformer import flux


DEFAULT_CHUNK_SIZE: int = 1024
C: float = 299792.458 #km/s

apply_vrad = lambda x, vrad: x*(vrad/C + 1)
v_apply_vrad = jax.jit(jax.vmap(apply_vrad, in_axes=(None, 0)))
atmosphere_flux = jax.jit(jax.vmap(flux, in_axes=(None, 0, None)))


@jax.jit
def interp(x, xp, fp):
    i = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, xp.shape[0] - 1)

    fp_i0 = fp[i]
    fp_i1 = fp[i-1]
    xp_i0 = xp[i]
    xp_i1 = xp[i-1]
    
    df = fp_i0 - fp_i1
    dx = xp_i0 - xp_i1
    delta = x - xp_i1

    f = jnp.where((dx == 0), 
                 fp[i], 
                 fp[i-1] + (delta / dx) * df
                )

    f = jnp.where(x < xp[:1], fp[:1], f)
    f = jnp.where(x > xp[-1:], fp[-1:], f)
    return f

v_interp = jax.jit(jax.vmap(interp, in_axes=(None, 0, 0)))


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
    points = log_wavelengths.shape[0]

    @partial(jax.checkpoint, prevent_cse=False)
    def chunk_scanner(carries, _):
        chunk_idx, atmo_sum, areas_sum = carries
        k_chunk_sizes = min(chunk_size, n_areas)

        a_chunk = lax.dynamic_slice(areas,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        m_chunk = lax.dynamic_slice(mus_flattened,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        vrad_chunk = lax.dynamic_slice(vrads,
                                        (chunk_idx, 0),
                                        (k_chunk_sizes, n_samples))

        
        # atmosphere_mul is the spectrum simulated for the corresponding wavelengths and optionally given parameters of mu, logg, and T.
        # It is then multiplied by the observed area to scale the contributions of spectra chunks
        # Shape: (n_vertices, 2, n_wavelengths)
        # 2 corresponds to the two components: continuum and full spectrum with lines
        atmosphere_mul = jnp.multiply(
            a_chunk.reshape((-1, 1, 1)),
            atmosphere_flux(log_wavelengths,
                            m_chunk,
                            parameters))

        # (observed) radial velocity correction for both continuum and spectrum correction
        # Shape: (n_vertices, 2, n_wavelengths)
        vrad_atmosphere = jax.vmap(
            lambda a: v_interp(
                log_wavelengths,
                v_apply_vrad(
                    log_wavelengths,
                    vrad_chunk),
                a), in_axes=(1,))(atmosphere_mul)
        
        # Sum the atmosphere contributions and normalize by all areas' sum
        new_atmo_sum = atmo_sum + jnp.sum(vrad_atmosphere, axis=1)
        new_areas_sum = areas_sum + jnp.sum(a_chunk, axis=0)
        
        return (chunk_idx + k_chunk_sizes, new_atmo_sum, new_areas_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out, areas_sum), lse = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((2, log_wavelengths.shape[-1])), jnp.zeros(1,)),
        xs=None,
        length=math.ceil(n_areas/chunk_size))
    return out/areas_sum

