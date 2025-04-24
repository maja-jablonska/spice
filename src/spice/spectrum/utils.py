from typing import Tuple
import warnings

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
def _apply_spectral_resolution_log(
    log_wavelengths: ArrayLike,
    spectrum: ArrayLike,
    resolution: float,
    max_sigma: float = 4,
):
    """Implementation of spectral resolution application that works with jit."""
    delta_log_wave = (log_wavelengths[-1] - log_wavelengths[0]) / (log_wavelengths.shape[0] - 1)
    sigma = 1 / (2 * jnp.sqrt(2 * jnp.log(2)) * jnp.log(10) * resolution * delta_log_wave)
    
    # Use a fixed kernel size instead of dynamic size
    max_kernel_size = 100  # Fixed size for kernel
    x_positions = jnp.arange(-max_kernel_size, max_kernel_size+1)
    
    # Calculate Gaussian values for all positions
    gaussian_values = jnp.exp(-0.5 * (x_positions / sigma) ** 2)
    
    # Apply a window based on max_sigma
    window_size = jnp.int32(max_sigma * sigma)
    valid_mask = (x_positions >= -window_size) & (x_positions <= window_size)
    gaussian_values = gaussian_values * valid_mask
    
    # Normalize the kernel
    gaussian_kernel = gaussian_values / jnp.sum(gaussian_values)
    
    # Convert to appropriate types
    gaussian_kernel = jnp.asarray(gaussian_kernel)
    spectrum = jnp.asarray(spectrum)
    
    # Convolve spectrum with Gaussian kernel
    degraded_spectrum = 1.0 - jax.scipy.signal.convolve(1.0 - spectrum, gaussian_kernel, mode='same')
    
    return degraded_spectrum


def apply_spectral_resolution(
    log_wavelengths: ArrayLike,
    spectrum: ArrayLike,
    resolution: float,
    max_sigma: float = 4
):
    """
    Applies a spectral resolution to a spectrum by convolving it with a Gaussian kernel.
    
    Parameters:
    - log_wavelengths (jnp.ndarray): 1D array of log10(wavelengths), uniformly spaced in log-space.
    - spectrum (jnp.ndarray): 1D array of the spectrum to be degraded.
    - resolution (float): The spectral resolution R = λ/Δλ to apply.
    - max_sigma (float, optional): Maximum sigma for the Gaussian kernel in units of standard deviations. Default is 4.
    
    Returns:
    - degraded_spectrum (jnp.ndarray): The spectrum with the applied resolution.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")

    diffs_log = jnp.diff(log_wavelengths)
    
    if not jnp.allclose(diffs_log, diffs_log[0]):
        warnings.warn("Wavelengths are not uniformly spaced in log-space. This may lead to incorrect results.")
    
    return _apply_spectral_resolution_log(log_wavelengths, spectrum, resolution, max_sigma)


def scale_all_abundances_by_metallicity(tpayne, metallicity):
    """
    Scale all abundance parameters by the given metallicity value.
    
    Args:
        tpayne: TransformerPayne instance
        parameters: Current parameters array
        metallicity: [Fe/H] value to scale abundances by
        
    Returns:
        Updated parameters array with all abundances scaled
    """
    # Get indices of abundance parameters
    abundance_indices = [i for i, is_abundance in 
                         enumerate(tpayne.model_definition.abundance_parameters[:-1])
                         if is_abundance]
    
    # Get the corresponding parameter names
    abundance_elements = [tpayne.stellar_parameter_names[i] for i in abundance_indices]
    
    # Use the existing method to set all abundance parameters
    return tpayne.set_group_of_abundances_relative_to_solar(tpayne.solar_parameters, metallicity, abundance_elements)


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
