import jax
import jax.numpy as jnp

LAW_IDS = {"linear": 0, "quadratic": 1, "square_root": 2, "logarithmic": 3, "nonlinear": 4}

def get_limb_darkening_law_id(law: str) -> int:
    return LAW_IDS[law]

import jax
import jax.numpy as jnp

@jax.jit
def limb_darkening(mu, law_id: int, coeffs):
    """
    JIT-friendly limb darkening implementation.
    """
    mu = jnp.clip(mu, 1e-6, 1.0)
    # Pad coeffs to length 4 for JIT shape consistency
    coeffs = jnp.pad(coeffs, (0, 4 - coeffs.shape[0]))

    a, b = coeffs[0], coeffs[1]

    def linear():
        return 1.0 - a * (1.0 - mu)

    def quadratic():
        return 1.0 - a * (1.0 - mu) - b * (1.0 - mu) ** 2

    def square_root():
        return 1.0 - a * (1.0 - mu) - b * (1.0 - jnp.sqrt(mu))

    def logarithmic():
        return 1.0 - a * (1.0 - mu) - b * mu * jnp.log(mu)

    def nonlinear():
        # Claret 4-param: I/I(1) = 1 - Σ a_i * (1 - μ^(i/2))
        idx = jnp.arange(4)
        mu_exp = mu[..., None] ** ((idx + 1) / 2.0)
        term = coeffs[:4] * (1.0 - mu_exp)
        return 1.0 - jnp.sum(term, axis=-1)

    return jax.lax.switch(
        law_id, [linear, quadratic, square_root, logarithmic, nonlinear]
    )
