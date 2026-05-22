"""Verify TZ For occlusion geometry vs Clausen Stromgren b.

Scans all SPICE blackbody sample times for visible fractions (vf_primary,
vf_secondary), classifies eclipse type, and compares phase/depth to Clausen
photometry — especially partial primary occultation (vf_primary ~ 0.7).

Usage::

    JAX_PLATFORMS=cpu PYTHONPATH=src python tz_fornacis_geometry_check.py
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
R1, R2 = 8.28, 3.94
T1, T2 = 4930.0, 6650.0
F1_FRAC = (R1**2 * T1**4) / (R1**2 * T1**4 + R2**2 * T2**4)
F2_FRAC = 1.0 - F1_FRAC


def _dm_flux_loss(frac_lost: float) -> float:
    return float(-2.5 * np.log10(max(1e-12, 1.0 - frac_lost)))


def _clausen():
    df = pd.read_csv(HERE / "tzfor_lightcurve.csv")
    df = df[df["HJD"] > 4500.0].copy()
    hjd = df["HJD"].to_numpy() + HJD_OFF
    ph = ((hjd - T0_PHOT) % PERIOD) / PERIOD
    dm = df["b"].to_numpy() - np.median(df["b"])
    return ph, dm


def _load_spice_bb():
    pkl = Path(os.environ.get("TZ_FOR_BB_PICKLE", REPO / "tz_fornacis_b_blackbody.pkl"))
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    times = np.asarray(d["times"], dtype=float)
    orig = np.asarray(d["time_origin"])
    dm = np.asarray(d["b_mag_ab"], dtype=float) - np.median(d["b_mag_ab"])
    bp = d["binary_params"]
    py = float(d["period_yr"])
    m0 = float(bp["mean_anomaly_at_t0_rad"])
    ph_spec = ((times / py + m0 / (2 * np.pi)) % 1.0) * PERIOD
    ph_phot = (
        (float(bp["t_p_hjd"]) + (times / py + m0 / (2 * np.pi)) % 1.0 * PERIOD - T0_PHOT)
        % PERIOD
    ) / PERIOD
    return times, orig, dm, ph_spec, ph_phot, bp


def _visible_fraction(mesh) -> float:
    mus = np.asarray(mesh.mus) > 0
    vis = np.asarray(mesh.visible_cast_areas)[mus]
    cast = np.asarray(mesh.cast_areas)[mus]
    return float(vis.sum() / cast.sum())


def _geometry_all(times_yr, bp):
    import jax.numpy as jnp
    import tz_fornacis_spectra as tzs
    from spice.models.binary import evaluate_orbit_at_times
    from spice.spectrum.blackbody import Blackbody

    # Coarse mesh + moderate orbit grid — geometry only, not spectra.
    tzs.N_MESH = int(os.environ.get("TZ_FOR_GEO_N_MESH", "80"))
    tzs.ORBIT_RESOLUTION_POINTS = int(os.environ.get("TZ_FOR_GEO_ORBIT_RES", "500"))

    los = np.asarray(tzs.LOS_VECTOR, dtype=float)
    los = los / np.linalg.norm(los)

    binary, *_ = tzs._build_binary_and_eclipses(Blackbody())
    times_yr = np.asarray(times_yr, dtype=float)
    chunk = int(os.environ.get("TZ_FOR_GEO_CHUNK", "50"))
    rho, vf1, vf2, zsign = [], [], [], []

    for start in range(0, len(times_yr), chunk):
        t_chunk = times_yr[start : start + chunk]
        m1_chunk, m2_chunk = evaluate_orbit_at_times(binary, jnp.asarray(t_chunk))
        for m1, m2 in zip(m1_chunk, m2_chunk, strict=True):
            c1, c2 = np.asarray(m1.center), np.asarray(m2.center)
            r_rel = c2 - c1
            rho.append(float(np.sqrt(max(0.0, np.sum(r_rel**2) - np.dot(r_rel, los) ** 2))))
            zsign.append(float(np.sign(np.dot(r_rel, los))))
            vf1.append(_visible_fraction(m1))
            vf2.append(_visible_fraction(m2))

    return np.asarray(rho), np.asarray(vf1), np.asarray(vf2), np.asarray(zsign)


def _predict_dm(vf1, vf2):
    loss = F1_FRAC * (1.0 - vf1) + F2_FRAC * (1.0 - vf2)
    return np.array([_dm_flux_loss(float(x)) for x in loss])


def _best_shift(ph_spice, dm_spice, ph_obs, dm_obs, n=400):
    order_o = np.argsort(ph_obs)
    ph_o, dm_o = ph_obs[order_o], dm_obs[order_o]
    best_r, best_s = -2.0, 0.0
    for shift in np.linspace(0.0, 1.0, n, endpoint=False):
        ph_try = (ph_spice + shift) % 1.0
        order_t = np.argsort(ph_try)
        dm_i = np.interp(ph_o, ph_try[order_t], dm_spice[order_t], period=1.0)
        r = np.corrcoef(dm_i, dm_o)[0, 1]
        if r > best_r:
            best_r, best_s = r, shift
    return best_s, best_r


def _depth_at_phase(ph, dm, center, half_width=0.08):
    mask = np.abs(ph - center) < half_width
    if not mask.any():
        return np.nan, np.nan
    j = int(np.argmax(dm[mask]))
    idx = np.where(mask)[0][j]
    return float(dm[idx]), float(ph[idx])


def main():
    ph_c, dm_c = _clausen()
    times, orig, dm, ph_spec, ph_phot, bp = _load_spice_bb()
    print("Computing occlusion at all sample times …")
    rho, vf1, vf2, zsign = _geometry_all(times, bp)
    dm_pred = _predict_dm(vf1, vf2)

    mask_total_sec = (vf2 < 0.02) & (vf1 > 0.98)
    mask_part_pri = (vf1 > 0.60) & (vf1 < 0.88) & (vf2 > 0.90)
    mask_part_sec = (vf2 > 0.60) & (vf2 < 0.88) & (vf1 > 0.90)
    mask_out = (vf1 > 0.995) & (vf2 > 0.995)

    print("=" * 72)
    print("TZ For geometry vs Clausen — full grid check")
    print("=" * 72)
    print(f"N samples = {len(times)}  (F2 flux frac = {F2_FRAC:.3f})")
    print(
        "Classification counts:",
        f"total secondary occultation {mask_total_sec.sum()},",
        f"partial primary {mask_part_pri.sum()},",
        f"partial secondary {mask_part_sec.sum()},",
        f"out of eclipse {mask_out.sum()}",
    )

    # Clausen reference depths
    d_c_pri, ph_c_pri = _depth_at_phase(ph_c, dm_c, 0.0)
    d_c_sec, ph_c_sec = _depth_at_phase(ph_c, dm_c, 0.5)
    print(f"\nClausen (phot T0): primary window φ≈0  max Δb = {d_c_pri:.3f}")
    print(f"Clausen:           secondary window φ≈0.5 max Δb = {d_c_sec:.3f}")

    def _report(label, mask):
        if not mask.any():
            print(f"\n{label}: (no samples)")
            return
        j = int(np.argmax(dm[mask]))
        idx = np.where(mask)[0][j]
        print(f"\n{label} (n={mask.sum()}):")
        print(
            f"  deepest measured Δm={dm[idx]:.3f}  predicted={dm_pred[idx]:.3f}"
            f"  at phot φ={ph_phot[idx]:.3f}  spec φ={ph_spec[idx]:.1f} d"
        )
        print(
            f"  ρ={rho[idx]:.2f} R☉  vf1={vf1[idx]:.3f}  vf2={vf2[idx]:.3f}"
            f"  zsign={zsign[idx]:+.0f}  tag={orig[idx]}"
        )

    _report("SPICE — total secondary occultation (vf2→0)", mask_total_sec)
    _report("SPICE — partial primary occultation (0.6<vf1<0.88)", mask_part_pri)
    _report("SPICE — partial secondary occultation", mask_part_sec)

    # Phase shifts
    shift_all, r_all = _best_shift(ph_phot, dm, ph_c, dm_c)
    shift_part, r_part = _best_shift(ph_phot[mask_part_pri], dm[mask_part_pri], ph_c, dm_c)
    shift_total, r_total = _best_shift(
        ph_phot[mask_total_sec], dm[mask_total_sec], ph_c, dm_c
    )

    print("\n--- Phase alignment (correlation vs Clausen) ---")
    print(f"All SPICE samples:              shift={shift_all:.4f}  r={r_all:.4f}")
    print(f"Partial-primary subset only:    shift={shift_part:.4f}  r={r_part:.4f}")
    print(f"Total-secondary subset only:    shift={shift_total:.4f}  r={r_total:.4f}")

    for name, shift in [
        ("global (all)", shift_all),
        ("partial-primary anchor", shift_part),
        ("total-secondary anchor", shift_total),
        ("fixed 0.1825", 0.1825),
        ("half period 0.5", 0.5),
        ("half period + global", (shift_all + 0.5) % 1.0),
    ]:
        ph_al = (ph_phot + shift) % 1.0
        d_s_pri, ph_s_pri = _depth_at_phase(ph_al, dm, 0.0)
        d_s_sec, ph_s_sec = _depth_at_phase(ph_al, dm, 0.5)
        # depth using only geometrically classified points near φ=0
        near0 = mask_part_pri & (np.abs(ph_al - 0.0) < 0.12)
        near05 = mask_total_sec & (np.abs(ph_al - 0.5) < 0.12)
        extra = ""
        if near0.any():
            j = int(np.argmax(dm[near0]))
            extra += f"  | partial-primary@φ≈0: Δm={dm[near0][j]:.3f} vf1={vf1[near0][j]:.2f}"
        if near05.any():
            j = int(np.argmax(dm[near05]))
            extra += f"  | total-sec@φ≈0.5: Δm={dm[near05][j]:.3f}"
        print(
            f"  {name:28s}  @φ≈0 Δm={d_s_pri:.3f}  @φ≈0.5 Δm={d_s_sec:.3f}{extra}"
        )

    # Where does partial-primary minimum land in phot phase (unshifted)?
    if mask_part_pri.any():
        j = int(np.argmax(dm[mask_part_pri]))
        idx = np.where(mask_part_pri)[0][j]
        print("\n--- Partial-primary eclipse location (unshifted phot phase) ---")
        print(
            f"  Deepest partial-primary: phot φ={ph_phot[idx]:.3f}"
            f"  (Clausen primary is φ=0 by definition)"
        )
        print(
            f"  Offset to align partial-primary → Clausen T0:"
            f"  shift ≈ {(1.0 - ph_phot[idx]) % 1.0:.4f}"
        )

    # Predicted vs measured correlation on eclipse samples
    ecl = ~mask_out
    if ecl.sum() > 3:
        r_pred = np.corrcoef(dm[ecl], dm_pred[ecl])[0, 1]
        print(f"\nMeasured vs blackbody-predicted Δm (eclipse samples): r = {r_pred:.4f}")

    print("\n--- Verdict ---")
    if mask_part_pri.any() and mask_total_sec.any():
        j_deep = int(np.argmax(dm[mask_total_sec]))
        j_part = int(np.argmax(dm[mask_part_pri]))
        i_deep = np.where(mask_total_sec)[0][j_deep]
        i_part = np.where(mask_part_pri)[0][j_part]
        align_deep_at0 = abs((ph_phot[i_deep] + shift_all) % 1.0) < 0.12
        part_near0 = abs(ph_phot[i_part]) < 0.12 or abs(ph_phot[i_part] - 1.0) < 0.12
        if align_deep_at0 and not part_near0:
            print(
                "• Global phase shift places the DEEP (total secondary) eclipse at Clausen φ≈0,"
            )
            print(
                "  while the PARTIAL primary eclipse sits near phot φ≈"
                f"{ph_phot[i_part]:.2f} — geometry is consistent, labels/epoch are not."
            )
        elif part_near0:
            print("• Partial-primary eclipse is near photometric φ=0 without extra shift.")
        else:
            print("• Neither eclipse type sits at Clausen φ=0 with the best global shift alone.")
            print("  Try shift ≈", f"{(1.0 - ph_phot[i_part]) % 1.0:.4f}", "to anchor partial-primary.")


if __name__ == "__main__":
    main()
