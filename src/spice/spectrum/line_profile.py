import jax
import jax.numpy as jnp

KIND_MAP = {"gaussian": 0, "lorentzian": 1, "voigt": 2}


def get_line_profile_id(kind: str) -> int:
    return KIND_MAP[kind]


def humlicek_wofz(z: jnp.ndarray) -> jnp.ndarray:
    """Approximation to the Faddeeva function (JAX-compatible)."""
    t = jnp.abs(z)
    a = jnp.where(t < 15, 1.0 / (z * z + 0.5), 1.0 / (z * z + 1.5))
    return jnp.exp(-z * z) * (1.0 + 1.128379167 * a)


def _safe_voigt_profile(delta_lambda: jnp.ndarray,
                        sigma: float,
                        gamma: float) -> jnp.ndarray:
    """Return a normalized Voigt profile without dividing by ~0 peaks."""

    # Guard against exactly zero widths so the denominator stays finite even if
    # callers try extreme combinations of microturbulent or thermal widths.
    sigma = jnp.maximum(sigma, 1e-12)

    z = (delta_lambda + 1j * gamma) / (sigma * jnp.sqrt(2.0))
    V = jnp.real(humlicek_wofz(z))

    peak = jnp.max(jnp.abs(V))
    normalized = jnp.where(peak > 1e-12, V / peak, jnp.zeros_like(V))
    return normalized


@jax.jit
def line_profile(
    wavelengths: jnp.ndarray,
    center: float,
    sigma: float,
    depth: float,
    kind_id: int = 0,  # 0=gaussian, 1=lorentzian, 2=voigt
    gamma: float = 0.05,
) -> jnp.ndarray:
    """
    Normalized absorption line profile.
    Args:
        wavelengths (jnp.ndarray): Wavelengths in Angstroms
        center (float): Line center in Angstroms
        sigma (float): Line width in Angstroms
        depth (float): Line depth (0 = no line, 1 = complete absorption)
        kind_id (int): Line profile kind (0 = gaussian, 1 = lorentzian, 2 = voigt)
        gamma (float): Lorentzian width in Angstroms

    Returns:
        jnp.ndarray: Line profile
    """
    Δλ = wavelengths - center

    def gaussian(_):
        return jnp.exp(-0.5 * (Δλ / sigma) ** 2)

    def lorentzian(_):
        return 1.0 / (1.0 + (Δλ / gamma) ** 2)

    def voigt(_):
        return _safe_voigt_profile(Δλ, sigma, gamma)

    profile = jax.lax.switch(kind_id, [gaussian, lorentzian, voigt], None)
    profile = jnp.clip(profile, 0.0, 1.0)
    clipped_depth = jnp.clip(depth, 0.0, 1.0)
    return 1.0 - clipped_depth * profile
