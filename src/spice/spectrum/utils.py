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

def linear_interpolation_1d(x: jnp.ndarray, 
                          y: jnp.ndarray, 
                          x_query: jnp.ndarray) -> jnp.ndarray:
    """
    1D linear interpolation using JAX.
    
    Args:
        x: Array of x coordinates (must be sorted)
        y: Array of y coordinates 
        x_query: Points to interpolate at
        
    Returns:
        Interpolated values at x_query points
    """
    # Find indices of closest points
    idx = jnp.searchsorted(x, x_query) - 1
    idx = jnp.clip(idx, 0, len(x)-2)
    
    # Get surrounding points
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    
    # Linear interpolation
    t = (x_query - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

@jax.jit
def linear_multivariate_interpolation(points: jnp.ndarray,
                                      values: jnp.ndarray,
                                      query_points: jnp.ndarray) -> jnp.ndarray:
    """
    Multivariate interpolation using JAX.
    
    Args:
        points: Array of points (N x D)
        values: Array of values at points (N,)
        query_points: Points to interpolate at (M x D)
        
    Returns:
        Interpolated values at query points (M,)
    """
    # Compute weights using inverse distance weighting
    distances = jnp.sqrt(jnp.sum((points[:, None, :] - query_points[None, :, :]) ** 2, axis=-1))
    weights = 1.0 / (distances + 1e-10)
    weights = weights / jnp.sum(weights, axis=0, keepdims=True)
    
    # Compute weighted average
    return jnp.sum(weights * values[:, None], axis=0)


@jax.jit
def nearest_multivariate_interpolation(points: jnp.ndarray,
                                       values: jnp.ndarray,
                                       query_points: jnp.ndarray) -> jnp.ndarray:
    """
    Multivariate interpolation using JAX.
    
    Args:
        points: Array of points (N x D)
        values: Array of values at points (N,)
        query_points: Points to interpolate at (M x D)
        
    Returns:
        Interpolated values at query points (M,)
    """
    # Compute pairwise distances
    distances = jnp.sum((points[:, None, :] - query_points[None, :, :]) ** 2, axis=-1)
    # Find nearest neighbor
    nearest_idx = jnp.argmin(distances, axis=0)
    return values[nearest_idx]
