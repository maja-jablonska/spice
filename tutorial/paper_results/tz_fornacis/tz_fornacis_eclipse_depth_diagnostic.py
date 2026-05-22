"""Print TZ For eclipse-depth diagnostics (SPICE blackbody vs Clausen b).

Usage::

    JAX_PLATFORMS=cpu PYTHONPATH=src python tz_fornacis_eclipse_depth_diagnostic.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

PERIOD = 75.66647
T0_PHOT = 2_445_183.92451
HJD_OFF = 2_440_000.0
SHIFT = 0.1825  # from Clausen correlation in clausen_b_compare notebook

R1, R2 = 8.28, 3.94
T1, T2 = 4930.0, 6650.0
F1 = R1**2 * T1**4
F2 = R2**2 * T2**4
F1_FRAC = F1 / (F1 + F2)
F2_FRAC = F2 / (F1 + F2)


def _dm_from_flux_loss(frac_lost: float) -> float:
    return float(-2.5 * np.log10(1.0 - frac_lost))


def _clausen():
    df = pd.read_csv(HERE / "tzfor_lightcurve.csv")
    df = df[df["HJD"] > 4500.0].copy()
    hjd = df["HJD"].to_numpy() + HJD_OFF
    ph = ((hjd - T0_PHOT) % PERIOD) / PERIOD
    dm = df["b"].to_numpy() - np.median(df["b"])
    return ph, dm


def _spice_bb():
    pkl = Path(os.environ.get("TZ_FOR_BB_PICKLE", REPO / "tz_fornacis_b_blackbody.pkl"))
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    times = np.asarray(d["times"])
    orig = np.asarray(d["time_origin"])
    dm = np.asarray(d["b_mag_ab"]) - np.median(d["b_mag_ab"])
    bp = d["binary_params"]
    py = float(d["period_yr"])
    m0 = float(bp["mean_anomaly_at_t0_rad"])
    ph_spec = ((times / py + m0 / (2 * np.pi)) % 1.0) * PERIOD
    ph_phot = (
        (float(bp["t_p_hjd"]) + (times / py + m0 / (2 * np.pi)) % 1.0 * PERIOD - T0_PHOT)
        % PERIOD
    ) / PERIOD
    return times, orig, dm, ph_spec, ph_phot, bp


def _geometry_at(t_yr, bp):
    import jax.numpy as jnp
    import tz_fornacis_spectra as tzs
    from spice.models.binary import evaluate_orbit_at_times
    from spice.spectrum.blackbody import Blackbody

    tzs.N_MESH = 300
    bb = Blackbody()
    binary, *_ = tzs._build_binary_and_eclipses(bb)
    los = np.asarray(tzs.LOS_VECTOR)
    m1, m2 = evaluate_orbit_at_times(binary, jnp.asarray([float(t_yr)]))
    m1, m2 = m1[0], m2[0]
    c1, c2 = np.asarray(m1.center), np.asarray(m2.center)
    r_rel = c2 - c1
    rho = float(np.sqrt(max(0.0, np.sum(r_rel**2) - np.dot(r_rel, los) ** 2)))
    mus1, mus2 = np.asarray(m1.mus) > 0, np.asarray(m2.mus) > 0
    vf1 = float(np.asarray(m1.visible_cast_areas)[mus1].sum() / np.asarray(m1.cast_areas)[mus1].sum())
    vf2 = float(np.asarray(m2.visible_cast_areas)[mus2].sum() / np.asarray(m2.cast_areas)[mus2].sum())
    return rho, vf1, vf2


def main():
    ph_c, dm_c = _clausen()
    times, orig, dm, ph_spec, ph_phot, bp = _spice_bb()
    ph_al = (ph_phot + SHIFT) % 1.0

    print("=" * 72)
    print("TZ Fornacis eclipse depth diagnostic")
    print("=" * 72)
    print(f"Blackbody flux fractions (no limb darkening): primary={F1_FRAC:.3f}  secondary={F2_FRAC:.3f}")
    print(f"Full secondary occulted: Δm ≈ {_dm_from_flux_loss(F2_FRAC):.3f}")
    print(f"Primary visible fraction for Δm=0.20: {1 - (1 - 10**(-0.2/2.5))/F1_FRAC:.3f}")

    windows = [
        ("Clausen primary (φ<0.15)", ph_c < 0.15),
        ("Clausen secondary (0.4<φ<0.6)", (ph_c > 0.4) & (ph_c < 0.6)),
        ("SPICE all", np.ones(len(dm), bool)),
        ("SPICE primary_eclipse tag", orig == "primary_eclipse"),
        ("SPICE secondary_eclipse tag", orig == "secondary_eclipse"),
    ]
    print("\n--- Depth summary ---")
    for name, mask in windows:
        if not mask.any():
            continue
        if name.startswith("Clausen"):
            sub_ph, sub_dm = ph_c[mask], dm_c[mask]
        else:
            sub_ph, sub_dm = ph_al[mask], dm[mask]
        j = int(np.argmax(sub_dm))
        print(f"{name:32s}  max Δm={sub_dm[j]:.3f}  at phot.φ={sub_ph[j]:.3f}")

    i_deep = int(np.argmax(dm))
    i_sec = int(np.where(orig == "secondary_eclipse")[0][len(np.where(orig == "secondary_eclipse")[0]) // 2])
    print("\n--- Geometry (occlusion) ---")
    for label, i in [("Deepest SPICE (primary_eclipse tag)", i_deep),
                     ("Mid secondary_eclipse tag", i_sec)]:
        rho, vf1, vf2 = _geometry_at(times[i], bp)
        pred = _dm_from_flux_loss(F2_FRAC * (1 - vf2)) + _dm_from_flux_loss(F1_FRAC * (1 - vf1))
        # approximate independent occlusions on flux (upper bound-ish)
        print(f"{label}:")
        print(f"  spec φ={ph_spec[i]:.1f} d  phot φ={ph_phot[i]:.2f}  aligned φ={ph_al[i]:.3f}")
        print(f"  ρ={rho:.2f} R☉ (R1+R2={R1+R2:.2f})  vf_primary={vf1:.3f}  vf_secondary={vf2:.3f}")
        print(f"  measured Δm={dm[i]:.3f}  (secondary-only loss ≈ {_dm_from_flux_loss(F2_FRAC*(1-vf2)):.3f})")

    print("\n--- Interpretation ---")
    print("• Clausen primary minimum (Δb≈0.2) matches ~30% loss of primary (K giant) light:")
    print("  expected when the cooler primary is partially eclipsed (vf_primary ≈ 0.71).")
    print("• SPICE deepest point (Δm≈0.83) is total secondary occultation (vf_secondary→0),")
    print("  not the shallow primary eclipse Clausen sees at φ≈0 after phase alignment.")
    print("• The ~0.13–0.18 mag feature on SPICE secondary_eclipse samples is the right")
    print("  depth scale for a partial primary eclipse; compare that window to Clausen φ≈0,")
    print("  not the 0.83 mag spike.")
    print("• eclipse_timestamps_kepler labels (primary/secondary) refer to which star is")
    print("  in front at minimum separation; verify against visible fractions at sampled times.")


if __name__ == "__main__":
    main()
