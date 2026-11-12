from functools import partial
from typing import Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike
except ImportError:
    raise ImportError("Please install JAX to use SPICE.")


from spice.constants import C_CM_S as C_CENTIMETERS, H_ERG_S as H_CONST_ERG_S
from spice.utils.dtypes import float_dtype as _dtype

ERG_S_TO_W = 1e-7
SPHERE_STERADIAN = 4.0 * 3.141592653589793  # full sphere [sr]
ZERO_POINT_LUM_W = 3.0128*10**(28)
JY_TO_ERG = 1e-23


def _sqrt_2pi():
    return jnp.sqrt(2.0 * jnp.pi)


def _log2():
    return jnp.log(2.0)


def _ln10():
    return jnp.log(10.0)

# Default static kernel half-width (pixels). Public wrappers widen it to the
# next power-of-two step when concrete inputs need more, so low-R/finely
# sampled spectra are no longer silently truncated.
DEFAULT_MAX_HALF_WIDTH = 2046


def _is_concrete(*values):
    return not any(isinstance(v, jax.core.Tracer) for v in values)


def _kernel_half_width(log_wavelength, R_min, window_size):
    """Pixels needed for ``window_size`` sigma at resolving power ``R_min`` (concrete inputs)."""
    import numpy as np
    log_wavelength = np.asarray(log_wavelength, dtype=float)
    delta_log = (log_wavelength[-1] - log_wavelength[0]) / (log_wavelength.shape[0] - 1)
    sigma_log = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)) * float(R_min) * np.log(10.0))
    return int(np.floor(window_size * sigma_log / delta_log))


def _static_cap(needed):
    """Smallest cap >= max(needed, default) on a 2**k - 2 ladder (few recompiles)."""
    if needed <= DEFAULT_MAX_HALF_WIDTH:
        return DEFAULT_MAX_HALF_WIDTH
    return (1 << (int(needed) + 2 - 1).bit_length()) - 2


def _resolve_cap(max_half_width, log_wavelength, R_min, window_size):
    if max_half_width is not None:
        if not isinstance(max_half_width, int) or max_half_width <= 0:
            raise ValueError("max_half_width must be a positive static integer")
        return max_half_width
    if _is_concrete(log_wavelength, R_min):
        return _static_cap(_kernel_half_width(log_wavelength, R_min, window_size))
    return DEFAULT_MAX_HALF_WIDTH


def _warn_truncated(needed, cap):
    import warnings
    needed = int(needed)
    if needed > cap:
        warnings.warn(
            f"Spectral-resolution kernel needs {needed} pixels per side but is capped at {cap}; "
            "the broadening will be too narrow. Pass a larger static max_half_width "
            "(or call outside jax.jit so the cap is chosen automatically).",
            RuntimeWarning, stacklevel=2)


def _check_truncation(needed, cap):
    # Works both eagerly and under jit (callback runs when the value is known).
    jax.debug.callback(_warn_truncated, needed, cap)


def apply_spectral_resolution(log_wavelength, flux, R, window_size=4.0, max_half_width=None):
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
    max_half_width : int, optional
        Static kernel half-width in pixels. By default it is 2046, widened
        automatically when ``log_wavelength`` and ``R`` are concrete and the
        kernel needs more. Under an outer ``jax.jit`` with traced ``R`` the
        default is used; a ``RuntimeWarning`` is raised if that truncates the
        kernel, in which case pass a larger value explicitly.

    Returns
    -------
    flux_lowres : (N,)
        Spectrum convolved to the target resolving power.
    """
    cap = _resolve_cap(max_half_width, log_wavelength, R, window_size)
    return _apply_spectral_resolution(log_wavelength, flux, R, window_size, cap)


@partial(jax.jit, static_argnames=("max_half_width",))
def _apply_spectral_resolution(log_wavelength, flux, R, window_size, max_half_width):
    N = flux.shape[0]
    delta_log = (log_wavelength[-1] - log_wavelength[0]) / (N - 1)

    # Gaussian sigma in log10(λ) space
    sigma_log = 1.0 / (2.0 * jnp.sqrt(2.0 * _log2()) * R * _ln10())

    est_half_width = jnp.floor(window_size * sigma_log / delta_log).astype(int)
    _check_truncation(est_half_width, max_half_width)
    half_width = jnp.minimum(est_half_width, max_half_width)

    grid = jnp.arange(-max_half_width, max_half_width + 1)
    x = grid * delta_log
    mask = (jnp.abs(grid) <= half_width).astype(flux.dtype)
    kernel = jnp.exp(-0.5 * (x / sigma_log) ** 2) * mask
    kernel /= jnp.sum(kernel)

    # Reshape correctly: (kernel_width, in_channels=1, out_channels=1)
    kernel_reshaped = kernel[:, None, None]
    flux_reshaped = flux[None, :, None]

    # Pad the edges with the global mean (not a reflection): within ~window_size
    # sigma of either end a sloped spectrum is biased toward that mean.
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


def apply_variable_resolution(
    log_wavelength: jnp.ndarray,
    flux: jnp.ndarray,
    R_array: jnp.ndarray,
    segment_size: int = 512,
    max_half_width: int = None,
):
    """Convolve a uniform log10 wavelength grid with segment-mean resolving power.

    All inputs have shape (N,), with N >= 2. ``segment_size`` is a positive
    static integer. Each segment includes neighbouring spectrum samples out
    to the Gaussian kernel's half-width; only the spectrum's outer edges
    are padded with the global mean, matching :func:`apply_spectral_resolution`.
    The final, shorter segment uses only its real R samples in the mean.
    ``max_half_width`` behaves as in :func:`apply_spectral_resolution`
    (chosen from the lowest R when inputs are concrete, else 2046 with a
    warning on truncation).
    """
    if (flux.ndim != 1 or log_wavelength.shape != flux.shape
            or R_array.shape != flux.shape or flux.shape[0] < 2):
        raise ValueError("wavelength, flux, and R_array must have matching 1-D shapes with at least two samples")
    if not isinstance(segment_size, int) or segment_size <= 0:
        raise ValueError("segment_size must be a positive static integer")
    R_min = jnp.min(R_array) if _is_concrete(R_array) else R_array
    cap = _resolve_cap(max_half_width, log_wavelength, R_min, 4.0)
    return _apply_variable_resolution(log_wavelength, flux, R_array, segment_size, cap)


@partial(jax.jit, static_argnames=("segment_size", "max_half_width"))
def _apply_variable_resolution(log_wavelength, flux, R_array, segment_size, max_half_width):
    n = flux.shape[0]
    n_segments = (n + segment_size - 1) // segment_size
    tail = n_segments * segment_size - n
    pad = max_half_width
    delta_log = (log_wavelength[-1] - log_wavelength[0]) / (n - 1)
    padded_flux = jnp.pad(flux, (pad, pad + tail), mode="mean")
    padded_R = jnp.pad(R_array, (0, tail))
    grid = jnp.arange(-pad, pad + 1)

    def process_segment(_, i):
        start = i * segment_size
        values = jax.lax.dynamic_slice_in_dim(padded_flux, start, segment_size + 2 * pad)
        r_values = jax.lax.dynamic_slice_in_dim(padded_R, start, segment_size)
        r_mean = jnp.sum(r_values) / jnp.minimum(segment_size, n - start)
        sigma_log = 1.0 / (2.0 * jnp.sqrt(2.0 * _log2()) * r_mean * _ln10())
        needed = jnp.floor(4.0 * sigma_log / delta_log).astype(int)
        half_width = jnp.minimum(needed, pad)
        weights = jnp.exp(-0.5 * (grid * delta_log / sigma_log)**2)
        weights = jnp.where(jnp.abs(grid) <= half_width, weights, 0.0)
        weights = (weights / jnp.sum(weights)).astype(flux.dtype)
        result = jax.lax.conv_general_dilated(
            values[None, :, None], weights[:, None, None], (1,), "VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
            precision=jax.lax.Precision.HIGHEST,
        )
        return None, (result[0, :, 0], needed)

    _, (segments, needed) = jax.lax.scan(process_segment, None, jnp.arange(n_segments))
    _check_truncation(jnp.max(needed), max_half_width)
    return segments.reshape(-1)[:n]



def scale_all_abundances_by_metallicity(tpayne, metallicity):
    """
    Scale all abundance parameters by the given metallicity value.
    
    Args:
        tpayne: TransformerPayne instance
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
    """Convert (wavelength [Angstrom], f_lambda) to (frequency [Hz], f_nu [Jy]).

    ``intensity`` must be per **centimetre** of wavelength (erg/s/cm^2/cm), not per
    Angstrom as elsewhere in SPICE: multiply a per-Angstrom flux by 1e8 first. The
    constant 3.34 approximates 1e4 / c[Angstrom/s] * 1e23 (3.33564), a 0.13% error.
    """
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
    Multivariate interpolation using inverse-distance weighting (JAX).

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
    Nearest-neighbour multivariate interpolation using JAX.

    For each query point the value of the closest point (by squared Euclidean
    distance) is returned.

    Args:
        points: Array of points (N x D)
        values: Array of values at points (N,)
        query_points: Points to interpolate at (M x D)

    Returns:
        Value of the nearest point for each query point (M,)
    """
    # Compute pairwise distances
    distances = jnp.sum((points[:, None, :] - query_points[None, :, :]) ** 2, axis=-1)
    # Find nearest neighbor
    nearest_idx = jnp.argmin(distances, axis=0)
    return values[nearest_idx]
