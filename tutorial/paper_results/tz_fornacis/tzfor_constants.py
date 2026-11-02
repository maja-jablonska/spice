"""Shared TZ Fornacis literature constants (single source of truth).

These parameters were previously copy-pasted across ``tz_fornacis_spectra.py``,
``tz_fornacis_spectra_test.py`` and ``verify_gamma_fix.py``. Keeping them here
guarantees the synthesis script, its fast narrow-band test, and the orbit-level
gamma check can never drift apart.

Kept deliberately dependency-light (stdlib + numpy only — no astropy, no jax) so
the lightweight scripts can ``from tzfor_constants import *`` without pulling in
heavy or optional dependencies (the test script in particular inlined these to
dodge an astropy import that breaks on some local environments).

Values: Gallenne et al. 2016 (A&A 586, A35) for the orbit, masses, radii and the
temperatures (their Table 2, Fe I equivalent widths on the disentangled spectra:
4930 +- 30 K and 6650 +- 200 K); rotation, metallicity and the inclination agree
with Andersen et al. 1991 (A&A 246, 99). Note that Gallenne et al. themselves call
their secondary temperature unimproved and adopt 6350 +- 70 K from the H-band flux
ratio, in line with Andersen's light-curve solution log Teff = 3.699 +- 0.009 and
3.803 +- 0.007 (5000 +- 104 K and 6353 +- 102 K). The joint SPICE fit recovers the
Andersen/adopted value (see ANDERSEN_* below).
"""
import os
from pathlib import Path

import numpy as np

# Speed of light [km/s].
C_KMS = 299_792.458

def _env_float(name: str, default: float) -> float:
    """Literature default, overridable via environment for fit experiments.

    Same pattern as TZ_FOR_FIRST_BJD / TZ_FOR_ORBIT_RESOLUTION below: defaults
    are the published values, so behaviour is unchanged unless a variable is set.
    Used to re-run the synthesis at fitted parameters without editing this file.
    """
    return float(os.environ.get(name, default))


# --- Masses, radii, temperatures, gravities ---
PRIMARY_MASS = 2.057       # Msun
PRIMARY_RADIUS = _env_float("TZ_FOR_PRIMARY_RADIUS", 8.28)      # Rsun
PRIMARY_TEFF = _env_float("TZ_FOR_PRIMARY_TEFF", 4930)          # K
PRIMARY_LOGG = _env_float("TZ_FOR_PRIMARY_LOGG", 2.91)          # cgs
SECONDARY_MASS = 1.958     # Msun
SECONDARY_RADIUS = _env_float("TZ_FOR_SECONDARY_RADIUS", 3.94)  # Rsun
SECONDARY_TEFF = _env_float("TZ_FOR_SECONDARY_TEFF", 6650)      # K
# Literature temperatures other than the starting values above (see the module docstring).
GALLENNE_PRIMARY_TEFF_ERR, GALLENNE_SECONDARY_TEFF_ERR = 30.0, 200.0        # K, Gallenne+2016 Table 2
GALLENNE_ADOPTED_SECONDARY_TEFF, GALLENNE_ADOPTED_SECONDARY_TEFF_ERR = 6350.0, 70.0  # K, from the H-band flux ratio
ANDERSEN_PRIMARY_TEFF, ANDERSEN_PRIMARY_TEFF_ERR = 5000.0, 104.0            # K, log Teff 3.699 +- 0.009
ANDERSEN_SECONDARY_TEFF, ANDERSEN_SECONDARY_TEFF_ERR = 6353.0, 102.0        # K, log Teff 3.803 +- 0.007
SECONDARY_LOGG = _env_float("TZ_FOR_SECONDARY_LOGG", 3.35)      # cgs, Gallenne Table 2 (fixed from Andersen); NB their own
# M2 / R2 give log g 3.54, which is what the meshes carry (per element, from the Roche potential)

# --- Orbit ---
PERIOD_DAYS = 75.66647     # Gallenne et al. 2016
PERIOD_YR = PERIOD_DAYS / 365.25  # Julian year (matches astropy day->year exactly)
ECC = 0.0
INCL_DEG = _env_float("TZ_FOR_INCL_DEG", 85.68)
# NB Gallenne et al. 2016 Table 4 give omega = 269.93 and Omega = 65.99 deg; the two are swapped here (per0 carries their
# Omega). Harmless for TZ For: e = 0.00002, so per0 only moves the periastron reference, which M_AT_T_P_RAD below absorbs
# (eclipses land on the observed phases), and long_an only rotates the orbit on the sky. Kept so that existing meshes,
# built with these values, stay consistent.
PER0_DEG = 65.99           # argument of periastron (= Gallenne's Omega, see above)
LONG_AN_DEG = 269.0        # longitude of ascending node (~ Gallenne's omega 269.93, see above)
DISTANCE_PC = 182.8        # pc (Gaia); Gallenne et al. 2016 orbital parallax: 185.9 +- 1.9 pc (no effect on normalised spectra or differential light curves)

# --- Atmosphere (per body), beyond teff/log g ---
# [Fe/H] -0.30: OUR earlier fitted value, NOT a literature value (Andersen et al. 1991: Fe/H +0.1 +- 0.15;
# Gallenne et al. 2016: +0.02 / -0.05). vmic 1.5 km/s assumed (Gallenne: 0.98 / 3.10).
PRIMARY_FEH = -0.30
SECONDARY_FEH = -0.30
PRIMARY_VMIC = 1.5
SECONDARY_VMIC = 1.5
PRIMARY_AFE = 0.0
SECONDARY_AFE = 0.0

# --- Broadening (km/s) ---
# Primary assumed tidally synchronized (v_eq ~ 5.5 km/s); secondary vsini ~ 38
# km/s is NOT synchronized (Andersen+1991). Macroturbulence is added post-emulator.
# Equatorial rotation used to build the PHOEBE Roche meshes (sets syncpar, hence the rotational flattening).
# Gallenne+2016 v sin i 45.7 at i = 85.68 deg -> 45.83 km/s for the secondary (TZ_FOR_SECONDARY_VROT_KMS=45.83).
PRIMARY_VROT_KMS = _env_float("TZ_FOR_PRIMARY_VROT_KMS", 5.5)
SECONDARY_VROT_KMS = _env_float("TZ_FOR_SECONDARY_VROT_KMS", 38.0)
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
