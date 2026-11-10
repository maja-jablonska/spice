"""Shared δ Cephei pulsation-template helpers and emulator parameter lists.

Single source of truth for the small helpers that were copy-pasted between
``build_cepheid_grid.py`` and the cepheid notebooks (``cepheid_build_vmicro.ipynb``,
``cepheid_fit_base_radius.ipynb``). All copies were verified functionally
identical; this is the canonical ``build_cepheid_grid.py`` version.
"""
import numpy as np

# Emulator channel names + solar reference parameters.
INTENSITY_PARAMS = ["teff", "logg", "[Fe/H]", "vmicro", "[a/Fe]",
                    "[C/Fe]", "[N/Fe]", "[O/Fe]", "[r/Fe]", "[s/Fe]", "mu"]
FLUX_PARAMS      = ["teff", "logg", "[Fe/H]", "vmicro", "[a/Fe]",
                    "[C/Fe]", "[N/Fe]", "[O/Fe]", "[r/Fe]", "[s/Fe]"]

SOLAR_INTENSITY = np.array([5777, 4.44, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
SOLAR_FLUX      = np.array([5777, 4.44, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def fourier_series(x, a0, *coeffs):
    n_terms = len(coeffs) // 2
    out = a0
    for i in range(n_terms):
        a_i, b_i = coeffs[2 * i], coeffs[2 * i + 1]
        out = out + a_i * np.cos(2 * np.pi * (i + 1) * x) + b_i * np.sin(2 * np.pi * (i + 1) * x)
    return out


def xi_micro_deltaCep(phase):
    """ξ (km/s) for δ Cep, fit to Luck 2018 (AJ 156, 171) table 3.

    Phase convention: GCVS ephemeris (Samus et al. 2017), phase 0 ≈ maximum
    light; minimum radius near phase ~0.7-0.8. Range 2.5–4.0 km/s, RMS fit
    residual 0.13 km/s.
    """
    phi = np.asarray(phase) % 1.0
    c = [3.046, 0.182, -0.507, -0.182, -0.061, -0.091, 0.040]
    val = c[0]
    for k in range(1, 4):
        val += c[2 * k - 1] * np.cos(2 * np.pi * k * phi) + c[2 * k] * np.sin(2 * np.pi * k * phi)
    return val


def radius_template_mean_normalised(phase, radius, n_terms=10):
    """(3, n_terms, 2) VSH amplitude/phase matrix for the l=0 radius curve R(phase)/<R>.

    Same packing as build_cepheid_grid.fit_radius_template, but normalised by the
    mean radius, so a mesh built with base radius <R> follows R(phase) exactly.
    """
    o = np.argsort(phase); phase, radius = np.asarray(phase)[o], np.asarray(radius)[o] / np.mean(radius)
    cols = [np.ones_like(phase)]
    for i in range(1, n_terms + 1):
        cols += [np.cos(2 * np.pi * i * phase), np.sin(2 * np.pi * i * phase)]
    coef, *_ = np.linalg.lstsq(np.stack(cols, 1), radius, rcond=None)
    radial = np.zeros((n_terms, 2))
    for i in range(n_terms):
        a_i, b_i = coef[1 + 2 * i], coef[2 + 2 * i]
        radial[i] = [np.hypot(a_i, b_i), np.arctan2(b_i, a_i)]
    return np.stack([radial, np.zeros_like(radial), np.zeros_like(radial)])


G_CGS = 6.674e-8
MSUN_G = 1.98847e33
RSUN_CM = 6.957e10


def spips_mass(pulsation_data):
    """Mass (M_sun) implied by the SPIPS MODEL table: its logg column is purely
    static, log10(G M / R^2), so M is constant to 1e-5 (4.726 for delta_cep.fits)."""
    R = np.asarray(pulsation_data["R"], dtype=float) * RSUN_CM
    logg = np.asarray(pulsation_data["logg"], dtype=float)
    return float(np.median(10 ** (logg + 2 * np.log10(R)) / G_CGS / MSUN_G))


def effective_logg(pulsation_data, period_days, mass=None):
    """Dynamic log g_eff(phase) = log10(G M / R^2 + d^2R/dt^2) from a SPIPS MODEL table.

    SPIPS ``Vpuls`` is -dR/dt (radial-velocity sign), so this equals
    G M / R^2 - dVpuls/dt. The acceleration is the derivative of the sorted,
    periodic R(phase) curve. ``mass`` defaults to :func:`spips_mass`, so the
    static part reproduces the table's own logg exactly.
    Returns (phase_sorted, logg_eff, logg_static).
    """
    phase = np.asarray(pulsation_data["PHASE"], dtype=float)
    order = np.argsort(phase)
    phase = phase[order]
    R = np.asarray(pulsation_data["R"], dtype=float)[order] * RSUN_CM
    mass = spips_mass(pulsation_data) if mass is None else mass
    g_static = G_CGS * mass * MSUN_G / R ** 2
    # periodic central differences in time
    t = phase * period_days * 86400.0
    P = period_days * 86400.0
    tp = np.concatenate([t[-1:] - P, t, t[:1] + P])
    Rp = np.concatenate([R[-1:], R, R[:1]])
    v = np.gradient(Rp, tp)
    a = np.gradient(v, tp)[1:-1]
    return phase, np.log10(g_static + a), np.log10(g_static)
