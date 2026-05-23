"""Shared TZ Fornacis literature constants (single source of truth).

These parameters were previously copy-pasted across ``tz_fornacis_spectra.py``,
``tz_fornacis_spectra_test.py`` and ``verify_gamma_fix.py``. Keeping them here
guarantees the synthesis script, its fast narrow-band test, and the orbit-level
gamma check can never drift apart.

Kept deliberately dependency-light (stdlib + numpy only — no astropy, no jax) so
the lightweight scripts can ``from tzfor_constants import *`` without pulling in
heavy or optional dependencies (the test script in particular inlined these to
dodge an astropy import that breaks on some local environments).

Values: Andersen et al. 1991 (A&A 246, 99) and Gallenne et al. 2016.
"""
import os
from pathlib import Path

import numpy as np

# Speed of light [km/s].
C_KMS = 299_792.458

# --- Masses, radii, temperatures, gravities ---
PRIMARY_MASS = 2.057       # Msun
PRIMARY_RADIUS = 8.28      # Rsun
PRIMARY_TEFF = 4930        # K
PRIMARY_LOGG = 2.91        # cgs
SECONDARY_MASS = 1.958     # Msun
SECONDARY_RADIUS = 3.94    # Rsun
SECONDARY_TEFF = 6650      # K
SECONDARY_LOGG = 3.35      # cgs

# --- Orbit ---
PERIOD_DAYS = 75.66647     # Gallenne et al. 2016
PERIOD_YR = PERIOD_DAYS / 365.25  # Julian year (matches astropy day->year exactly)
ECC = 0.0
INCL_DEG = 85.68
PER0_DEG = 65.99           # argument of periastron
LONG_AN_DEG = 269.0        # longitude of ascending node
DISTANCE_PC = 182.8        # pc (Gaia)

# --- Atmosphere (per body), beyond teff/log g ---
# [Fe/H] from Andersen et al. 1991; vmic ~1.5 km/s is the evolved-star standard.
PRIMARY_FEH = -0.30
SECONDARY_FEH = -0.30
PRIMARY_VMIC = 1.5
SECONDARY_VMIC = 1.5
PRIMARY_AFE = 0.0
SECONDARY_AFE = 0.0

# --- Broadening (km/s) ---
# Primary assumed tidally synchronized (v_eq ~ 5.5 km/s); secondary vsini ~ 38
# km/s is NOT synchronized (Andersen+1991). Macroturbulence is added post-emulator.
PRIMARY_VROT_KMS = 5.5
SECONDARY_VROT_KMS = 38.0
PRIMARY_VMACRO_KMS = 5.0
SECONDARY_VMACRO_KMS = 6.0

# --- Ephemeris / systemic velocity ---
T_P_HJD = 2_452_599.29040
GAMMA1_KMS = 17.99
GAMMA2_KMS = 18.35
GAMMA_KMS = (PRIMARY_MASS * GAMMA1_KMS + SECONDARY_MASS * GAMMA2_KMS) / (
    PRIMARY_MASS + SECONDARY_MASS
)
_FIRST_BJD = float(os.environ.get("TZ_FOR_FIRST_BJD", "2454887.5473123"))
# T_P_HJD is the time of the DEEPER minimum, treated here as the M=0 epoch.
# Downstream consumers map SPICE phase to Clausen's photometric ephemeris by
# subtracting M_AT_T_P_RAD / (2*pi); see tz_fornacis_clausen_b_compare.ipynb.
_MEAN_ANOMALY_T0 = (
    ((_FIRST_BJD - T_P_HJD) % PERIOD_DAYS) / PERIOD_DAYS * (2.0 * np.pi)
)
M_AT_T_P_RAD = float((1.5 * np.pi - np.deg2rad(PER0_DEG)) % (2.0 * np.pi))


def _repo_root(here: Path) -> Path:
    """Walk up from ``here`` to the directory containing ``src/spice``."""
    for p in [here, *here.parents]:
        if (p / "src" / "spice").is_dir():
            return p
    raise RuntimeError("Could not find spice repo root (src/spice)")


__all__ = [
    "C_KMS",
    "PRIMARY_MASS", "PRIMARY_RADIUS", "PRIMARY_TEFF", "PRIMARY_LOGG",
    "SECONDARY_MASS", "SECONDARY_RADIUS", "SECONDARY_TEFF", "SECONDARY_LOGG",
    "PERIOD_DAYS", "PERIOD_YR", "ECC", "INCL_DEG", "PER0_DEG", "LONG_AN_DEG",
    "DISTANCE_PC",
    "PRIMARY_FEH", "SECONDARY_FEH", "PRIMARY_VMIC", "SECONDARY_VMIC",
    "PRIMARY_AFE", "SECONDARY_AFE",
    "PRIMARY_VROT_KMS", "SECONDARY_VROT_KMS",
    "PRIMARY_VMACRO_KMS", "SECONDARY_VMACRO_KMS",
    "T_P_HJD", "GAMMA1_KMS", "GAMMA2_KMS", "GAMMA_KMS", "M_AT_T_P_RAD",
    "_FIRST_BJD", "_MEAN_ANOMALY_T0",
    "_repo_root",
]
