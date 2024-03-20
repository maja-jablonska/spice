from typing import Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike
except ImportError:
    raise ImportError("Please install JAX to use SPICE.")


ERG_S_TO_W = 1e-7
SPHERE_STERADIAN = 12.56
ZERO_POINT_LUM_W = 3.0128*10**(28)
C_CENTIMETERS = 29979245800.
JY_TO_ERG = 1e-23
H_CONST_ERG_S = 6.62607015*10**(-27)


@jax.jit
def wavelengths_to_frequencies(wavelengths: ArrayLike) -> ArrayLike:
    return C_CENTIMETERS*1e8/wavelengths


@jax.jit
def intensity_wavelengths_to_hz(wavelengths: ArrayLike, intensity: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    vws_hz = wavelengths_to_frequencies(wavelengths)
    fv_Jy = (intensity*1e-4*3.34*wavelengths**2)
    
    vws_mask = jnp.argsort(vws_hz)
    fv_Jy = fv_Jy[vws_mask]
    vws_hz = jnp.sort(vws_hz)
    
    return vws_hz, fv_Jy


@jax.jit
def intensity_Jy_to_erg(intensity: ArrayLike) -> ArrayLike:
    return intensity*JY_TO_ERG
