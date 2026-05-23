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
    phi = phase % 1.0
    c = [3.046, 0.182, -0.507, -0.182, -0.061, -0.091, 0.040]
    val = c[0]
    for k in range(1, 4):
        val += c[2 * k - 1] * np.cos(2 * np.pi * k * phi) + c[2 * k] * np.sin(2 * np.pi * k * phi)
    return val
