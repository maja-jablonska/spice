"""Gradient-based inference for TZ For atmospheric parameters.

SPICE's synthesis is JAX all the way down, so the *atmospheric* parameters are
differentiable end to end (verified: autodiff d/dTeff agrees with finite
differences to 0.03%). PHOEBE is not JAX, so the geometry is not -- but this
data does not constrain the radii anyway, so holding the mesh fixed and
differentiating with respect to Teff / log g / [Fe/H] is the right split.

The economy that makes it tractable: a surface element enters the intensity
only through its ``mu``. Across each star Teff varies by 3 K (primary) and 16 K
(secondary) and log g by <0.02 dex, so evaluating the emulator on a small
mu-grid and interpolating onto the elements replaces ~1500 network calls per
epoch with ~32, while staying differentiable. ``disc_flux_exact`` is kept so the
surrogate can be checked against the real thing rather than assumed.
"""
import numpy as np
import jax
import jax.numpy as jnp

C_KMS = 299792.458


def make_emulator(bundle="RozanskiT/TPayne-spice-harps"):
    from spice.spectrum.aemu_spectrum_emulator import (
        IntensityPretrainedAemuSpectrumEmulator)
    return IntensityPretrainedAemuSpectrumEmulator(bundle)


def disc_flux_exact(emu, log_wl, model):
    """Per-element synthesis: the reference the surrogate must reproduce."""
    from spice.spectrum.spectrum import simulate_observed_flux
    return simulate_observed_flux(emu.intensity, model, log_wl,
                                  disable_doppler_shift=True)[:, 0]


def mu_weights(mus, areas, n_mu=32):
    """Bin the fixed mesh into area-weighted mu nodes. Geometry only, done once.

    The disc integral is sum_i A_i I(lambda, mu_i); with the mesh fixed, the
    A_i and mu_i never change, so binning them collapses the integral to
    ``W @ I(mu_nodes)`` -- one emulator call per node and no per-element work.
    """
    mus = np.asarray(mus); areas = np.asarray(areas)
    vis = mus > 0
    w = np.where(vis, np.clip(areas, 0.0, None), 0.0)
    nodes = np.linspace(0.0, 1.0, n_mu)
    idx = np.clip(np.searchsorted(nodes, mus) - 1, 0, n_mu - 2)
    frac = (mus - nodes[idx]) / (nodes[idx + 1] - nodes[idx])
    W = np.zeros(n_mu)
    np.add.at(W, idx, w * (1 - frac) * vis)      # linear split between nodes
    np.add.at(W, idx + 1, w * frac * vis)        # conserves total area exactly
    return jnp.asarray(nodes), jnp.asarray(W)


def disc_flux_binned(emu, log_wl, mu_nodes, W, params_row):
    """Disc-integrated flux from binned geometry; differentiable in params_row."""
    I = jax.vmap(lambda m: emu.intensity(log_wl, m, params_row)[:, 0])(mu_nodes)
    return W @ I


def passband_mag(flux, wavelengths, response):
    """Energy-weighted passband magnitude, up to a constant zero point."""
    num = jnp.trapezoid(flux * response, wavelengths)
    den = jnp.trapezoid(response, wavelengths)
    return -2.5 * jnp.log10(num / den)


# ---------------------------------------------------------------------------
# Full photometric objective
# ---------------------------------------------------------------------------

def precompute_geometry(payload, n_mu=32):
    """Area-weighted mu bins per epoch per star. Fixed: no parameters involved.

    ``I(lambda, mu; p)`` does not depend on the epoch -- only the weights do --
    so one set of n_mu emulator calls per star serves every epoch, and each
    epoch reduces to a matrix product. This is what makes a gradient step cost
    the same as a single-epoch forward pass.
    """
    W1, W2 = [], []
    for m1, m2 in payload["models"]:
        nodes, w1 = mu_weights(m1.d_mus, m1.d_cast_areas, n_mu)
        _, w2 = mu_weights(m2.d_mus, m2.d_cast_areas, n_mu)
        W1.append(w1); W2.append(w2)
    return nodes, jnp.stack(W1), jnp.stack(W2)


def mean_params(model):
    """Area-weighted mean parameter row (0.06% vs per-element; see validation)."""
    p = np.asarray(model.parameters)
    mu = np.asarray(model.d_mus); a = np.asarray(model.d_cast_areas)
    w = np.where(mu > 0, np.clip(a, 0.0, None), 0.0)
    return jnp.asarray((p * w[:, None]).sum(0) / w.sum())


def make_photometric_objective(emu, payload, obs_phase, obs_mag, responses,
                               wavelengths, idx_teff, idx_feh, n_mu=32,
                               weights=None):
    """Returns loss(theta) with theta = [dTeff1, dTeff2, dfeh, dphi].

    Offsets are *relative to the payload's own parameters*, so theta = 0 is the
    model as exported. Per-band magnitude zero points are removed by median
    subtraction, which is exact and keeps them out of the parameter vector.
    """
    nodes, W1, W2 = precompute_geometry(payload, n_mu)
    p1_0, p2_0 = mean_params(payload["models"][0][0]), mean_params(payload["models"][0][1])
    log_wl = jnp.log10(wavelengths)
    P, T_P = 75.66647, 2452599.29040
    model_phase = jnp.asarray(((np.asarray(payload["times"]) - T_P) % P) / P)
    order = jnp.argsort(model_phase); mph = model_phase[order]
    bands = list(responses)
    if weights is None:
        weights = {b: 1.0 for b in bands}

    def forward(theta):
        p1 = p1_0.at[idx_teff].add(theta[0]).at[idx_feh].add(theta[2])
        p2 = p2_0.at[idx_teff].add(theta[1]).at[idx_feh].add(theta[2])
        I1 = jax.vmap(lambda m: emu.intensity(log_wl, m, p1)[:, 0])(nodes)
        I2 = jax.vmap(lambda m: emu.intensity(log_wl, m, p2)[:, 0])(nodes)
        flux = W1 @ I1 + W2 @ I2                     # (n_epoch, n_wavelength)
        return {b: jax.vmap(lambda f: passband_mag(f, wavelengths, r))(flux)
                for b, r in responses.items()}

    def loss(theta):
        mags = forward(theta)
        tot = 0.0
        for b in bands:
            mm = mags[b][order]; mm = mm - jnp.median(mm)
            mi = jnp.interp((obs_phase - theta[3]) % 1.0, mph, mm, period=1.0)
            r = obs_mag[b] - mi
            r = r - jnp.median(r)
            tot = tot + jnp.sum(r ** 2) / weights[b] ** 2
        return tot / sum(len(obs_mag[b]) for b in bands)

    return loss, forward


# ---------------------------------------------------------------------------
# Spectroscopy: (mu, v_los) binning
# ---------------------------------------------------------------------------

def mu_v_weights(mus, areas, vlos, n_mu=16, n_v=24, v_pad=2.0):
    """Area weights on a joint (mu, v_los) grid. Geometry only, computed once.

    Photometry can bin on mu alone, but a spectrum cannot: rotational
    broadening comes from the *spread* of line-of-sight velocity across the
    disc, which mu-binning discards. Binning jointly keeps it, while the
    emulator is still called only n_mu times -- the Doppler shifts act on the
    resulting spectra, which is cheap.
    """
    mus = np.asarray(mus); areas = np.asarray(areas); vlos = np.asarray(vlos)
    vis = mus > 0
    w = np.where(vis, np.clip(areas, 0.0, None), 0.0)
    mu_nodes = np.linspace(0.0, 1.0, n_mu)
    vmin, vmax = vlos[vis].min() - v_pad, vlos[vis].max() + v_pad
    v_nodes = np.linspace(vmin, vmax, n_v)
    W = np.zeros((n_mu, n_v))
    im = np.clip(np.searchsorted(mu_nodes, mus) - 1, 0, n_mu - 2)
    fm = (mus - mu_nodes[im]) / (mu_nodes[im + 1] - mu_nodes[im])
    iv = np.clip(np.searchsorted(v_nodes, vlos) - 1, 0, n_v - 2)
    fv = (vlos - v_nodes[iv]) / (v_nodes[iv + 1] - v_nodes[iv])
    for a, b, c_, d, ww in ((0, 0, 1 - fm, 1 - fv, w), (1, 0, fm, 1 - fv, w),
                            (0, 1, 1 - fm, fv, w), (1, 1, fm, fv, w)):
        np.add.at(W, (im + a, iv + b), ww * c_ * d * vis)
    return jnp.asarray(mu_nodes), jnp.asarray(v_nodes), jnp.asarray(W)


def disc_spectrum_binned(emu, wavelengths, mu_nodes, v_nodes, W, params_row):
    """Disc-integrated (flux, continuum), Doppler-broadened, differentiable."""
    log_wl = jnp.log10(wavelengths)
    out = jax.vmap(lambda m: emu.intensity(log_wl, m, params_row))(mu_nodes)
    I_line, I_cont = out[:, :, 0], out[:, :, 1]

    def shift(spec, v):                       # observed lambda -> rest lambda
        return jnp.interp(wavelengths, wavelengths * (1.0 + v / C_KMS), spec)

    def one_mu(line_i, cont_i, w_row):
        sl = jax.vmap(lambda v: shift(line_i, v))(v_nodes)
        sc = jax.vmap(lambda v: shift(cont_i, v))(v_nodes)
        return w_row @ sl, w_row @ sc

    fl, fc = jax.vmap(one_mu)(I_line, I_cont, W)
    return jnp.sum(fl, axis=0), jnp.sum(fc, axis=0)
