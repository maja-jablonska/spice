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

dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


SQRT_2PI = jnp.sqrt(2.0 * jnp.pi)
LOG2 = jnp.log(2.0)
LN10 = jnp.log(10.0)

@jax.jit
def apply_spectral_resolution(log_wavelength, flux, R, window_size=4.0):
    """
    Apply a single Gaussian spectral resolution to one spectrum (JIT-compatible).

    Parameters
    ----------
    log_wavelength : (N,)
        Wavelength grid in log10(λ). Must be uniformly spaced.
    flux : (N,)
        Input flux array (e.g. continuum-normalized spectrum).
    R : float
        Resolving power (λ / Δλ_FWHM).
    window_size : float
        Number of Gaussian σ on either side of the kernel (controls truncation).

    Returns
    -------
    flux_lowres : (N,)
        Spectrum convolved to the target resolving power.
    """
    N = flux.shape[0]
    delta_log = (log_wavelength[-1] - log_wavelength[0]) / (N - 1)

    # Gaussian sigma in log10(λ) space
    sigma_log = 1.0 / (2.0 * jnp.sqrt(2.0 * LOG2) * R * LN10)

    # Define kernel width (static cap for JIT safety)
    max_half_width = 2046
    est_half_width = jnp.floor(window_size * sigma_log / delta_log).astype(int)
    half_width = jnp.minimum(est_half_width, max_half_width)

    grid = jnp.arange(-max_half_width, max_half_width + 1)
    x = grid * delta_log
    mask = (jnp.abs(grid) <= half_width).astype(flux.dtype)
    kernel = jnp.exp(-0.5 * (x / sigma_log) ** 2) * mask
    kernel /= jnp.sum(kernel)

    # Reshape correctly: (kernel_width, in_channels=1, out_channels=1)
    kernel_reshaped = kernel[:, None, None]
    flux_reshaped = flux[None, :, None]

    # Reflect padding for edges
    pad = max_half_width
    flux_padded = jnp.pad(flux_reshaped, ((0, 0), (pad, pad), (0, 0)), mode="mean")

    # Perform 1D convolution
    flux_conv = jax.lax.conv_general_dilated(
        lhs=flux_padded,      # (N, W, C)
        rhs=kernel_reshaped,  # (kernel, inC, outC)
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        precision=jax.lax.Precision.HIGHEST
    )
    
    return flux_conv[0, :, 0]


@jax.jit
def apply_variable_resolution(
    log_wavelength: jnp.ndarray,
    flux: jnp.ndarray,
    R_array: jnp.ndarray,
    segment_size: int = 512,
):
    """
    Apply wavelength-dependent spectral resolution using local convolutions (JIT-friendly).

    Parameters
    ----------
    log_wavelength : (N,)
        Wavelength grid in log10(λ).
    flux : (N,)
        Input flux array.
    R_array : (N,)
        Resolving power as a function of wavelength (one per pixel).
    segment_size : int
        Segment width where R is locally constant.

    Returns
    -------
    flux_lowres : (N,)
        Spectrum convolved with variable resolution.
    """
    N = flux.shape[0]
    num_segments = N // segment_size + (N % segment_size > 0)

    # Indices of segments (JAX-friendly)
    segment_idx = jnp.arange(num_segments)

    def process_segment(i, _):
        i0 = i * segment_size
        i1 = jnp.minimum(i0 + segment_size, N)
        lam_seg = log_wavelength[i0:i1]
        flux_seg = flux[i0:i1]
        R_seg = jnp.mean(R_array[i0:i1])
        seg_flux = apply_spectral_resolution(lam_seg, flux_seg, R_seg)
        return None, seg_flux

    _, seg_results = jax.lax.scan(process_segment, None, segment_idx)
    return jnp.concatenate(seg_results)



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
