import jax.numpy as jnp
import jax

def get_vsini_kernel(vsini, delta_log_wave, N, limb_darkening_coeff=0.6):
    constant_c = 299792.458 # km/s
    v_sini_units_delta_log_wave = 1 / jnp.log(10) * (vsini / constant_c) / delta_log_wave
    x = jnp.linspace(-N, N, 2*N+1)
    x_safe = jnp.where(jnp.abs(x) < v_sini_units_delta_log_wave, x, 0.0)
    ker = (\
                2*(1-limb_darkening_coeff)*jnp.sqrt(1-(x_safe/v_sini_units_delta_log_wave)**2)\
                 +(jnp.pi*limb_darkening_coeff/2*(1-(x_safe/v_sini_units_delta_log_wave)**2))\
            )\
            /(jnp.pi*v_sini_units_delta_log_wave*(1-limb_darkening_coeff/3))
    ker = jnp.where(jnp.abs(x) < v_sini_units_delta_log_wave, ker, 0.0)
    return ker/jnp.sum(ker)


def add_vsini_broadening(log_wave, flux, vsini, limb_darkening_coeff=0.6):
    constant_c = 299792.458
    diff_log_wave = jnp.diff(log_wave)
    delta_log_wave = diff_log_wave[0]
    assert jnp.any(jnp.isclose(diff_log_wave, diff_log_wave)), "For vsini broadening log_wave must be sampled equidistantly!"

    vsini_units_delta_log_wave = 1 / jnp.log(10) * (vsini / constant_c) / delta_log_wave
    kernel_length = int(1.1 * vsini_units_delta_log_wave  + 1)

    ker_vsini = get_vsini_kernel(vsini, delta_log_wave, kernel_length, limb_darkening_coeff)
    convolved_flux = 1.0 - jax.scipy.signal.fftconvolve(1.0 - flux, ker_vsini, mode='same')

    return convolved_flux


    
