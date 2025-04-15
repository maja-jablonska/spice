import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from jax.scipy.integrate import trapezoid
from typing import Callable
from spice.models import MeshModel
import math
from functools import partial
from spice.spectrum.utils import ERG_S_TO_W
from spice.spectrum.filter import Filter

from jaxtyping import Array, Float

DEFAULT_CHUNK_SIZE: int = 1024
C: float = 299792.458  # km/s
SOL_RAD_CM = 69570000000.0

apply_vrad = lambda x, vrad: x * (vrad / C + 1)
apply_vrad_log = lambda x, vrad: x + jnp.log10(vrad / C + 1)
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
                         chunk_size: int,
                         disable_doppler_shift: bool = False):
    '''
    Each surface element has a vector of parameters (mu, LOS velocity, etc)
    Some of these parameters are the flux model's input
    '''

    # Just the 1D case for now
    n_areas = areas.shape[0]
    n_parameters = parameters.shape[-1]

    v_intensity = jax.vmap(intensity_fn, in_axes=(0, 0, 0))
    
    n = math.ceil(n_areas / chunk_size)

    @partial(jax.checkpoint, prevent_cse=False)
    def chunk_scanner(carries, x):
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
        shifted_log_wavelengths = jax.lax.cond(disable_doppler_shift,
                                               lambda lv, _: jnp.repeat(lv[jnp.newaxis, :], chunk_size, axis=0),
                                               v_apply_vrad_log,
                                               log_wavelengths, vrad_chunk)

        # atmosphere_mul is the spectrum simulated for the corresponding wavelengths and optionally given parameters of mu, logg, and T.
        # It is then multiplied by the observed area to scale the contributions of spectra chunks

        # Shape: (n_vertices, 2, n_wavelengths)
        # Areas should be rescaled by mus
        # 2 corresponds to the two components: continuum and full spectrum with lines
        # n_wavelengths, n_verices, 2 (continuum+spectrum), 1

        # shifted_log_wavelengths (CHUNK_SIZE, n_wavelengths)
        # m_chunk (CHUNK_SIZE)
        # p_chunk (CHUNK_SIZE, n_parameters)
        v_in = v_intensity(shifted_log_wavelengths,  # (n,)
                           m_chunk[:, jnp.newaxis],
                           p_chunk)
        atmosphere_mul = jnp.multiply(
            (a_chunk)[:, jnp.newaxis, jnp.newaxis],
            v_in)

        new_atmo_sum = atmo_sum + jnp.sum(atmosphere_mul, axis=0)

        return (chunk_idx + k_chunk_sizes, new_atmo_sum), chunk_idx+k_chunk_sizes

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out), _ = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((log_wavelengths.shape[-1], 2))),
        xs=None,
        length=n)
    return out


@partial(jax.jit, static_argnums=(0, 6, 7))
def __spectrum_flash_sum_with_padding(intensity_fn,
                         log_wavelengths,
                         areas,
                         mus,
                         vrads,
                         parameters,
                         chunk_size: int,
                         wavelengths_chunk_size: int,
                         disable_doppler_shift: bool = False):
    n_padding = wavelengths_chunk_size - (log_wavelengths.shape[0] % wavelengths_chunk_size)
    log_wavelengths_padded = jnp.pad(log_wavelengths, (0, n_padding), mode='constant', constant_values=0)
    # Reshape padded wavelengths into chunks
    wavelength_chunks = log_wavelengths_padded.reshape(-1, wavelengths_chunk_size).T
    
    def scan_fn(carry, chunk):
        result = __spectrum_flash_sum(intensity_fn,
                                    chunk,
                                    areas,
                                    mus,
                                    vrads,
                                    parameters,
                                    chunk_size,
                                    disable_doppler_shift)
        return carry, result

    _, results = lax.scan(scan_fn,
                         None,
                         wavelength_chunks)
    return results.reshape(-1, 2, order='F')


@partial(jax.jit, static_argnums=(1,))
def _adjust_dim(x: ArrayLike, chunk_size: int) -> ArrayLike:
    return jnp.concatenate([x, jnp.zeros((chunk_size - x.shape[0] % chunk_size, *x.shape[1:]))], axis=0)


# TODO: think: change to simulate_obseved_monochromatic_flux?
@partial(jax.jit, static_argnums=(0, 4, 5))
def simulate_observed_flux(intensity_fn: Callable[[Float[Array, "n_wavelengths"], float, Float[Array, "n_mesh_elements n_parameters"]], Float[Array, "n_wavelengths 2"]],
                           m: MeshModel,
                           log_wavelengths: Float[Array, "n_wavelengths"],
                           distance: float = 10.0,
                           chunk_size: int = DEFAULT_CHUNK_SIZE,
                           wavelengths_chunk_size: int = DEFAULT_CHUNK_SIZE,
                           disable_doppler_shift: bool = False) -> Float[Array, "n_wavelengths 2"]:
    """Simulate the observed flux from a mesh model.

    This function calculates the observed flux from a mesh model by combining the intensity function,
    mesh geometry, and radiative transfer effects. It accounts for visible areas, viewing angles (mu),
    line-of-sight velocities, and distance effects.

    Args:
        intensity_fn (Callable): Function that computes intensity given wavelengths, mu and parameters
        m (MeshModel): The mesh model containing geometry and physical parameters
        log_wavelengths (Float[Array, "n_wavelengths"]): Log of wavelength points to evaluate
        distance (float, optional): Distance to object in parsecs. Defaults to 10.0.
        chunk_size (int, optional): Size of chunks for parallel processing. Defaults to 1024.
        wavelengths_chunk_size (int, optional): Chunk size for wavelength array. Defaults to 1024.
        disable_doppler_shift (bool, optional): Whether to disable Doppler shift calculations. Defaults to False.

    Returns:
        Float[Array, "n_wavelengths 2"]: Array containing the computed flux at each wavelength point.
        The second dimension contains [flux, flux_error].
        Units are erg/s/cm^2/Å.
    """
    
    return jnp.nan_to_num(__spectrum_flash_sum_with_padding(intensity_fn,
                                               log_wavelengths,
                                               _adjust_dim(m.visible_cast_areas, chunk_size),
                                               _adjust_dim(jnp.where(m.mus > 0, m.mus, 0.), chunk_size),
                                               _adjust_dim(m.los_velocities, chunk_size),
                                               _adjust_dim(m.parameters, chunk_size),
                                               chunk_size,
                                               wavelengths_chunk_size,
                                               disable_doppler_shift) * jnp.power(m.radius,
                                                                                  2) * 5.08326693599739e-16 / (
                                      distance ** 2))[:len(log_wavelengths), :]


@partial(jax.jit, static_argnums=(0, 5))
def __flux_flash_sum(flux_fn,
                     log_wavelengths,
                     areas,
                     vrads,
                     parameters,
                     chunk_size: int,
                     disable_doppler_shift: bool = False):

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
        shifted_log_wavelengths = jax.lax.cond(disable_doppler_shift,
                                               lambda lv, _: jnp.repeat(lv[jnp.newaxis, :], chunk_size, axis=0),
                                               v_apply_vrad_log,
                                               log_wavelengths, vrad_chunk)

        # atmosphere_mul is the spectrum simulated for the corresponding wavelengths and optionally given parameters of mu, logg, and T.
        # It is then multiplied by the observed area to scale the contributions of spectra chunks

        # Shape: (n_vertices, 2, n_wavelengths)
        # Areas should be rescaled by mus
        # 2 corresponds to the two components: continuum and full spectrum with lines
        # n_wavelengths, n_verices, 2 (continuum+spectrum), 1

        # shifted_log_wavelengths (CHUNK_SIZE, n_wavelengths)
        # m_chunk (CHUNK_SIZE)
        # p_chunk (CHUNK_SIZE, n_parameters)
        v_in = v_flux(shifted_log_wavelengths,  # (n,)
                      p_chunk)

        atmosphere_mul = jnp.multiply(
            a_chunk[:, jnp.newaxis, jnp.newaxis],  # Czemy nie 2D? Broadcastowanie?
            v_in)

        new_atmo_sum = atmo_sum + jnp.sum(atmosphere_mul, axis=0)

        return (chunk_idx + k_chunk_sizes, new_atmo_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out), _ = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((log_wavelengths.shape[-1], 2))),
        xs=None,
        length=math.ceil(n_areas / chunk_size))
    return out


@partial(jax.jit, static_argnums=(0, 5, 6))
def __flux_flash_sum_with_padding(intensity_fn,
                         log_wavelengths,
                         areas,
                         vrads,
                         parameters,
                         chunk_size: int,
                         wavelengths_chunk_size: int,
                         disable_doppler_shift: bool = False):
    n_padding = wavelengths_chunk_size - (log_wavelengths.shape[0] % wavelengths_chunk_size)
    log_wavelengths_padded = jnp.pad(log_wavelengths, (0, n_padding), mode='constant', constant_values=0)
    # Reshape padded wavelengths into chunks
    wavelength_chunks = log_wavelengths_padded.reshape(-1, wavelengths_chunk_size).T
    
    def scan_fn(carry, chunk):
        result = __flux_flash_sum(intensity_fn,
                                    chunk,
                                    areas,
                                    vrads,
                                    parameters,
                                    chunk_size,
                                    disable_doppler_shift)
        return carry, result

    _, results = lax.scan(scan_fn,
                         None,
                         wavelength_chunks)
    return results.reshape(-1, 2, order='F')


@partial(jax.jit, static_argnums=(0, 3, 4))
def simulate_monochromatic_luminosity(flux_fn: Callable[[Float[Array, "n_wavelengths"], Float[Array, "n_mesh_elements n_parameters"]], Float[Array, "n_wavelengths 2"]],
                                      m: MeshModel,
                                      log_wavelengths: Float[Array, "n_wavelengths"],
                                      chunk_size: int = DEFAULT_CHUNK_SIZE,
                                      wavelengths_chunk_size: int = DEFAULT_CHUNK_SIZE,
                                      disable_doppler_shift: bool = False) -> Float[Array, "n_wavelengths 2"]:
    """Simulate the monochromatic luminosity from a mesh model.

    This function calculates the monochromatic luminosity from a mesh model by combining the flux function,
    mesh geometry, and radiative transfer effects. It accounts for visible areas, line-of-sight velocities,
    and radius effects.

    Args:
        flux_fn (Callable[[Float[Array, "n_wavelengths"], Float[Array, "n_mesh_elements n_parameters"]], Float[Array, "n_wavelengths 2"]]): 
            Function that computes flux given wavelengths and parameters
        m (MeshModel): The mesh model containing geometry and physical parameters
        log_wavelengths (Float[Array, "n_wavelengths"]): Log of wavelength points to evaluate
        chunk_size (int, optional): Size of chunks for parallel processing. Defaults to 1024.
        wavelengths_chunk_size (int, optional): Chunk size for wavelength array. Defaults to 1024.
        disable_doppler_shift (bool, optional): Whether to disable Doppler shift calculations. Defaults to False.

    Returns:
        Float[Array, "n_wavelengths 2"]: Array containing the computed luminosity at each wavelength point.
        The second dimension contains [luminosity, luminosity_error].
        Units are erg/s/Å.
    """
    return jnp.nan_to_num(__flux_flash_sum_with_padding(flux_fn,
                                           log_wavelengths,
                                           _adjust_dim(m.areas, chunk_size),
                                           _adjust_dim(m.los_velocities, chunk_size),
                                           _adjust_dim(m.parameters, chunk_size),
                                           chunk_size,
                                           wavelengths_chunk_size,
                                           disable_doppler_shift) * jnp.power(m.radius, 2) * 4.8399849e+21)[:len(log_wavelengths), :]


@partial(jax.jit, static_argnums=(0, 3, 4))
def luminosity(flux_fn: Callable[[Float[Array, "n_wavelengths"], Float[Array, "n_mesh_elements n_parameters"]], Float[Array, "n_wavelengths 2"]],
               model: MeshModel,
               wavelengths: Float[Array, "n_wavelengths"],
               chunk_size: int = DEFAULT_CHUNK_SIZE,
               wavelengths_chunk_size: int = DEFAULT_CHUNK_SIZE) -> float:
    """Calculate the bolometric luminosity of the model.

    This function computes the total bolometric luminosity by integrating the monochromatic luminosity
    over all wavelengths. It uses the trapezoidal rule for numerical integration.

    Args:
        flux_fn (Callable[[Float[Array, "n_wavelengths"], Float[Array, "n_mesh_elements n_parameters"]], Float[Array, "n_wavelengths 2"]]):
            Function that computes flux given wavelengths and parameters
        model (MeshModel): The mesh model containing geometry and physical parameters
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points to evaluate [Angstrom]
        chunk_size (int, optional): Size of chunks for parallel processing. Defaults to 1024.
        wavelengths_chunk_size (int, optional): Chunk size for wavelength array. Defaults to 1024.

    Returns:
        float: Total bolometric luminosity [erg/s]
    """
    flux = jnp.nan_to_num(simulate_monochromatic_luminosity(flux_fn, model, jnp.log10(wavelengths), chunk_size, wavelengths_chunk_size))
    return trapezoid(y=flux[:, 0], x=wavelengths * 1e-8)


@jax.jit
def filter_responses(wavelengths: ArrayLike, sample_wavelengths: ArrayLike, sample_responses: ArrayLike) -> ArrayLike:
    return jnp.interp(wavelengths, sample_wavelengths, sample_responses)


# TODO: rename to observed passband luminosity? how to name it so that it's consistent???
# Let's rename it to obseved_passband_flux
# later let's have AB_observed_passband_magnitude

# If we set the transmission curve to be 1.0 for all wavelengths, we get the bolometric flux (NOT MONochormatic anymore)

@partial(jax.jit, static_argnums=(0,))
def __AB_passband_luminosity_photonic(filter: Filter,
                                      wavelengths: Float[Array, "n_wavelengths"],
                                      observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the AB magnitude in a given filter passband.

    This function computes the AB magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the AB magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: AB magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)
    
    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths * 1e-8, y=wavelengths * 1e-8 * observed_flux * transmission_responses) /
            (trapezoid(x=wavelengths * 1e-8, y=transmission_responses / (wavelengths * 1e-8)))
        ) + filter.AB_zeropoint
    
    
@partial(jax.jit, static_argnums=(0,))
def __AB_passband_luminosity_photonic_gaia(filter: Filter,
                                           wavelengths: Float[Array, "n_wavelengths"],
                                           observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the AB magnitude in a given filter passband.

    This function computes the AB magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the AB magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: AB magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)
    # Montegriffo (2023)
    # P_A/(10^9 h*c) * int(lambda * f_lambda * T_zeta dlambda)
    
    nm_lambda = wavelengths/10.
    watt_flux = 1e-10*observed_flux

    return -2.5 * jnp.log10(3663830098681164.5* 
            trapezoid(x=nm_lambda, y=nm_lambda * watt_flux * transmission_responses)
    ) + filter.AB_zeropoint

@partial(jax.jit, static_argnums=(0,))
def __AB_passband_luminosity_non_photonic(filter: Filter,
                                          wavelengths: Float[Array, "n_wavelengths"],
                                          observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the AB magnitude in a given filter passband.

    This function computes the AB magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the AB magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: AB magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)

    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths, y=observed_flux / 1e8 * transmission_responses) /
            trapezoid(x=wavelengths, y=transmission_responses)
    ) + filter.AB_zeropoint
    
    
@partial(jax.jit, static_argnums=(0,))
def __AB_passband_luminosity_non_photonic_panstarrs_ps1(filter: Filter,
                                                        wavelengths: Float[Array, "n_wavelengths"],
                                                        observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the AB magnitude in a given filter passband for PanSTARRS PS1.

    This function computes the AB magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the AB magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: AB magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)

    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths, y=observed_flux * transmission_responses) /
            trapezoid(x=wavelengths, y=transmission_responses / ((wavelengths*1e-8)**2))
    ) + filter.AB_zeropoint

def AB_passband_luminosity(filter: Filter,
                           wavelengths: Float[Array, "n_wavelengths"],
                           observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the AB magnitude in a given filter passband.

    This function computes the AB magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the AB magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: AB magnitude in the filter passband [mag]
    """
    if 'gaia' in filter.name.lower():
        return __AB_passband_luminosity_photonic_gaia(filter, wavelengths, observed_flux)
    elif 'panstarrs' in filter.name.lower():
        return __AB_passband_luminosity_non_photonic_panstarrs_ps1(filter, wavelengths, observed_flux)
    elif filter.non_photonic:
        return __AB_passband_luminosity_non_photonic(filter, wavelengths, observed_flux)
    else:
        return __AB_passband_luminosity_photonic(filter, wavelengths, observed_flux)


@partial(jax.jit, static_argnums=(0,))
def __ST_passband_luminosity_non_photonic(filter: Filter,
                                          wavelengths: Float[Array, "n_wavelengths"],
                                          observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the ST magnitude in a given filter passband.

    This function computes the ST magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the ST magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: ST magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)
    
    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths, y=observed_flux / 1e8 * transmission_responses) /
            trapezoid(x=wavelengths, y=transmission_responses)
        ) + filter.ST_zeropoint
    
    
@partial(jax.jit, static_argnums=(0,))
def __ST_passband_luminosity_non_photonic_panstarrs_ps1(filter: Filter,
                                                        wavelengths: Float[Array, "n_wavelengths"],
                                                        observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the ST magnitude in a given filter passband for PanSTARRS PS1.

    This function computes the ST magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the AB magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: ST magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)

    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths, y=observed_flux * transmission_responses) /
            trapezoid(x=wavelengths, y=transmission_responses / ((wavelengths*1e-8)**2))
    ) + filter.ST_zeropoint

    
@partial(jax.jit, static_argnums=(0,))
def __ST_passband_luminosity_photonic(filter: Filter,
                                      wavelengths: Float[Array, "n_wavelengths"],
                                      observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the ST magnitude in a given filter passband.

    This function computes the ST magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the ST magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: ST magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)
    
    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths, y=wavelengths * observed_flux / 1e8 * transmission_responses) /
            trapezoid(x=wavelengths, y=wavelengths * transmission_responses)
        ) + filter.ST_zeropoint
    
    
def ST_passband_luminosity(filter: Filter,
                           wavelengths: Float[Array, "n_wavelengths"],
                           observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the ST magnitude in a given filter passband.

    This function computes the ST magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the ST magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: ST magnitude in the filter passband [mag]
    """
    if 'gaia' in filter.name.lower():
        raise ValueError("Gaia filters are not supported for ST magnitude calculations.")
    elif 'panstarrs' in filter.name.lower():
        return __ST_passband_luminosity_non_photonic_panstarrs_ps1(filter, wavelengths, observed_flux)
    elif filter.non_photonic:
        return __ST_passband_luminosity_non_photonic(filter, wavelengths, observed_flux)
    else:
        return __ST_passband_luminosity_photonic(filter, wavelengths, observed_flux)


@partial(jax.jit, static_argnums=(0,))
def __Vega_passband_luminosity_photonic_gaia(filter: Filter,
                                             wavelengths: Float[Array, "n_wavelengths"],
                                             observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the Vega magnitude in a given filter passband.

    This function computes the Vega magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the Vega magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: AB magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)
    # Montegriffo (2023)
    # P_A/(10^9 h*c) * int(lambda * f_lambda * T_zeta dlambda)
    
    nm_lambda = wavelengths/10.
    watt_flux = 1e-10*observed_flux

    return -2.5 * jnp.log10(3663830098681164.5* 
            trapezoid(x=nm_lambda, y=nm_lambda * watt_flux * transmission_responses)
    ) + filter.Vega_zeropoint

    
@partial(jax.jit, static_argnums=(0,))
def __Vega_passband_luminosity_non_photonic(filter: Filter,
                                            wavelengths: Float[Array, "n_wavelengths"],
                                            observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the Vega magnitude in a given filter passband.

    This function computes the Vega magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the Vega magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^3]

    Returns:
        float: Vega magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)
    
    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths, y=observed_flux / 1e8 * transmission_responses) /
            trapezoid(x=wavelengths, y=transmission_responses)
        ) + filter.Vega_zeropoint

@partial(jax.jit, static_argnums=(0,))
def __Vega_passband_luminosity_photonic(filter: Filter,
                                        wavelengths: Float[Array, "n_wavelengths"],
                                        observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the Vega magnitude in a given filter passband.

    This function computes the Vega magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the Vega magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^3]

    Returns:
        float: Vega magnitude in the filter passband [mag]
    """
    transmission_responses = filter.filter_responses_for_wavelengths(wavelengths)
    
    return -2.5 * jnp.log10(
            trapezoid(x=wavelengths, y=wavelengths * observed_flux / 1e8 * transmission_responses) /
            trapezoid(x=wavelengths, y=wavelengths * transmission_responses)
        ) + filter.Vega_zeropoint


def Vega_passband_luminosity(filter: Filter,
                             wavelengths: Float[Array, "n_wavelengths"],
                             observed_flux: Float[Array, "n_wavelengths 2"]) -> float:
    """Calculate the Vega magnitude in a given filter passband.

    This function computes the Vega magnitude by integrating the observed flux weighted by
    the filter transmission function and comparing to the Vega magnitude zero point.

    Args:
        filter (Filter): Filter object containing the transmission curve
        wavelengths (Float[Array, "n_wavelengths"]): Wavelength points [Angstrom]
        observed_flux (Float[Array, "n_wavelengths 2"]): Observed flux at each wavelength point
            [erg/s/cm^2/Å]

    Returns:
        float: AB magnitude in the filter passband [mag]
    """
    if filter.Vega_zeropoint is None:
        raise ValueError("Vega zeropoint is not set for this filter.")
    
    if 'gaia' in filter.name.lower():
        return __Vega_passband_luminosity_photonic_gaia(filter, wavelengths, observed_flux)
    elif filter.non_photonic:
        return __Vega_passband_luminosity_non_photonic(filter, wavelengths, observed_flux)
    else:
        return __Vega_passband_luminosity_photonic(filter, wavelengths, observed_flux)


@jax.jit
def absolute_bol_luminosity(luminosity: float) -> float:
    """Calculate the absolute bolometric magnitude from a given luminosity.

    This function converts a bolometric luminosity in erg/s to an absolute bolometric
    magnitude using the standard formula M_bol = -2.5 * log10(L) + M_bol,⊙, where
    M_bol,⊙ = 71.1974 is the zero point calibrated to the Sun's bolometric magnitude.

    Args:
        luminosity (float): Total bolometric luminosity [erg/s]

    Returns:
        float: Absolute bolometric magnitude [mag]
    """
    return -2.5 * jnp.log10(luminosity * ERG_S_TO_W) + 71.1974
