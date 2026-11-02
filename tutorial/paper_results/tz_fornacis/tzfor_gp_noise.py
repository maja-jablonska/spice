"""Correlated-noise likelihood for the TZ For spectra (Starfish-like global covariance, Czekala et al. 2015).

The residuals r = obs - model of each epoch are modelled as a Gaussian process on the uniform log-wavelength pixel grid:

    K_e = diag(sigma^2) + a_e^2 M32(d; l1) [+ b^2 M32(d; l2)],     M32(d; l) = (1 + sqrt3 d / l) exp(-sqrt3 d / l)

with d the velocity separation of two pixels [km/s], a_e a per-epoch amplitude, l1 a shared correlation length and an
optional second, long kernel (b, l2) for the ~2 A tail. The loss term is

    -2 ln L = sum_e w_e [ r_e^T K_e^-1 r_e + ln det K_e - sum_good ln sigma^2 ]

(the constant sum ln sigma^2 is subtracted so that the term equals the plain chi2 for a -> 0). A dense K over 156k pixels per
epoch is impossible, so each epoch is cut into contiguous blocks of B pixels and the covariance between blocks is dropped:
correct when B >> the correlation length (measured L ~ 33 px, tail ~160 px; B = 1024). Masked pixels are decoupled
exactly (their rows/columns are the identity and their residual is 0), so they contribute nothing: this is the marginal
likelihood of the good pixels. Batched float64 Cholesky on the GPU, differentiable in r and the hyperparameters.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

SQRT3 = math.sqrt(3.0)


def matern32(d, ell):
    x = SQRT3 * d / ell
    return (1.0 + x) * jnp.exp(-x)


class BlockGP:
    """Blocked GP negative log likelihood (x2) for residuals on an (n_ep, n) uniform grid with pixel step dv [km/s]."""

    def __init__(self, good, sigma, dv_kms, block=1024):
        good = np.asarray(good, bool); sigma = np.asarray(sigma, float)
        self.n_ep, self.n = good.shape; self.B = int(block); self.nb = -(-self.n // self.B); pad = self.nb * self.B - self.n
        self.pad = pad
        g = np.pad(good, ((0, 0), (0, pad)), constant_values=False)
        s2 = np.pad(np.where(good, sigma, 1.0) ** 2, ((0, 0), (0, pad)), constant_values=1.0)
        self.g = jnp.asarray(g.reshape(self.n_ep, self.nb, self.B).astype(float))
        self.s2 = jnp.asarray(s2.reshape(self.n_ep, self.nb, self.B))
        self.ln_s2_good = jnp.asarray(np.sum(np.where(g, np.log(np.pad(np.where(good, sigma, 1.0) ** 2, ((0, 0), (0, pad)), constant_values=1.0)), 0.0), axis=1))
        idx = np.arange(self.B); self.D = jnp.asarray(np.abs(idx[:, None] - idx[None, :]) * float(dv_kms))   # (B, B) km/s
        self.med_sigma = np.array([np.median(sigma[e][good[e]]) if good[e].any() else 0.01 for e in range(self.n_ep)])

    def kernel(self, ell1, b=None, ell2=None):
        """Unit-amplitude short kernel and the (already scaled) long kernel, (B, B) each."""
        T1 = matern32(self.D, ell1)
        T2 = None if b is None else b ** 2 * matern32(self.D, ell2)
        return T1, T2

    def nll(self, resid, ln_a, ln_l1, ep_w, ln_b=None, ln_l2=None):
        """-2 ln L summed over epochs with weights ep_w; resid (n_ep, n) = obs - model (masked entries ignored)."""
        r = jnp.pad(resid, ((0, 0), (0, self.pad))).reshape(self.n_ep, self.nb, self.B) * self.g
        T1, T2 = self.kernel(jnp.exp(ln_l1), None if ln_b is None else jnp.exp(ln_b), None if ln_l2 is None else jnp.exp(ln_l2))
        eye = jnp.eye(self.B)

        @jax.checkpoint
        def one(args):
            re, ge, s2e, lae = args
            Kc = jnp.exp(2.0 * lae) * T1 if T2 is None else jnp.exp(2.0 * lae) * T1 + T2
            K = ge[:, :, None] * ge[:, None, :] * Kc[None] + eye[None] * (ge * s2e + (1.0 - ge))[:, :, None]
            L = jnp.linalg.cholesky(K)
            al = jax.scipy.linalg.solve_triangular(L, re[..., None], lower=True)[..., 0]
            return jnp.sum(al ** 2) + 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)))

        per = jax.lax.map(one, (r, self.g, self.s2, ln_a))
        return jnp.sum(ep_w * (per - self.ln_s2_good))


def dense_nll(resid, good, sigma, dv, a, ell1, b=None, ell2=None):
    """Reference: exact dense GP for one epoch (small n only), same conventions."""
    m = np.asarray(good, bool); r = np.asarray(resid)[m]; s = np.asarray(sigma)[m]; idx = np.where(m)[0]
    d = np.abs(idx[:, None] - idx[None, :]) * dv
    K = a ** 2 * np.asarray(matern32(d, ell1)) + np.diag(s ** 2)
    if b is not None: K = K + b ** 2 * np.asarray(matern32(d, ell2))
    sign, ld = np.linalg.slogdet(K)
    return float(r @ np.linalg.solve(K, r) + ld - np.sum(np.log(s ** 2)))
