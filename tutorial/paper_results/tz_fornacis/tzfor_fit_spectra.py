"""Fit TZ For atmospheric parameters to HARPS, using the precomputed grid library.

The library (``--mode grid``, job 178152241) holds each component synthesised
once at a reference out-of-eclipse epoch over a (dTeff, dlogg, feh) grid. Here
we blend pairs of them and compare to the SuppNet-normalised spectra.

Three things this respects that a naive fit would not:

* the blended spectrum is ``(F1+F2)/(C1+C2)``, not the mean of two normalised
  spectra -- each star's lines are diluted by the other's continuum, and that
  dilution *is* the spectroscopic light ratio;
* orbital motion is applied as a rigid Doppler shift of each component relative
  to the reference epoch, which is what makes one synthesis per grid point
  enough;
* eclipse epochs are excluded. The library reuses one epoch's disc geometry, so
  its rotational broadening is only valid while the disc is unocculted.

[Fe/H] is shared between the two stars; everything else is free per star.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tzfor_constants as K  # noqa: E402

C_KMS = 299792.458
R_HARPS = 115000.0
DEEP_PHASE, SHALLOW_PHASE, ECLIPSE_HALF = 0.158, 0.658, 0.013


def component_rv(model):
    """Visible-area-weighted line-of-sight velocity of one component."""
    v = np.asarray(model.los_velocities)
    w = np.clip(np.asarray(model.d_cast_areas), 0.0, None)
    return float(np.sum(v * w) / np.sum(w))


def doppler_shift(wl, flux, dv):
    """Rigid shift by dv km/s (positive = redshift), resampled onto wl."""
    if abs(dv) < 1e-9:
        return flux
    return np.interp(wl, wl * (1.0 + dv / C_KMS), flux)


def broaden(y, lam0, dlam, vmacro):
    fwhm = float(np.hypot(C_KMS / R_HARPS, vmacro))
    return gaussian_filter1d(y, lam0 * fwhm / C_KMS / 2.3548 / dlam, mode="nearest")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_grid_n1500.pkl"))
    ap.add_argument("--meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1500_spectra_meshes.pkl"))
    ap.add_argument("--observed", default=str(HERE / "harps_normalized.npz"))
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "spectra_fit.pkl"))
    args = ap.parse_args()

    G = pickle.load(open(args.grid, "rb"))
    M = pickle.load(open(args.meshes, "rb"))
    O = np.load(args.observed, allow_pickle=True)
    lib, wl = G["library"], np.asarray(G["wavelengths"])
    ref = G["ref_epoch"]

    times = np.asarray(M["times"])
    phase = ((times - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
    rv1 = np.array([component_rv(p[0]) for p in M["models"]])
    rv2 = np.array([component_rv(p[1]) for p in M["models"]])

    in_ecl = ((np.minimum(np.abs(phase - DEEP_PHASE), 1 - np.abs(phase - DEEP_PHASE)) < ECLIPSE_HALF) |
              (np.minimum(np.abs(phase - SHALLOW_PHASE), 1 - np.abs(phase - SHALLOW_PHASE)) < ECLIPSE_HALF))
    print(f"{len(times)} epochs; excluding {in_ecl.sum()} in eclipse "
          f"(phases {np.round(phase[in_ecl],3)}) — the library's disc geometry "
          f"is only valid unocculted")

    files = list(M["harps_files"]); obs_files = [str(x) for x in O["files"]]
    windows = G["windows"]
    obs = {}
    for i, fn in enumerate(files):
        j = obs_files.index(fn)
        ow, onf = np.asarray(O["wavelengths"][j]), np.asarray(O["normed_flux"][j])
        ok = np.isfinite(ow) & np.isfinite(onf)
        obs[i] = np.interp(wl, ow[ok], onf[ok], left=np.nan, right=np.nan)

    sel = {}
    for lo, hi, name in windows:
        s = (wl >= lo) & (wl <= hi)
        sel[name] = (s, 0.5 * (wl[s][0] + wl[s][-1]), float(wl[s][1] - wl[s][0]))

    keys1 = [k for k in lib if k[0] == "primary"]
    keys2 = [k for k in lib if k[0] == "secondary"]
    print(f"library: {len(keys1)} primary x {len(keys2)} secondary entries")

    rows = []
    for k1 in keys1:
        f1r, c1r = lib[k1]
        for k2 in keys2:
            if k2[3] != k1[3]:          # [Fe/H] is shared
                continue
            f2r, c2r = lib[k2]
            tot, npts = 0.0, 0
            for i in range(len(times)):
                if in_ecl[i]:
                    continue
                d1, d2 = rv1[i] - rv1[ref], rv2[i] - rv2[ref]
                f1 = doppler_shift(wl, f1r, d1); c1 = doppler_shift(wl, c1r, d1)
                f2 = doppler_shift(wl, f2r, d2); c2 = doppler_shift(wl, c2r, d2)
                for name, (s, lam0, dlam) in sel.items():
                    a1 = broaden(f1[s], lam0, dlam, K.PRIMARY_VMACRO_KMS)
                    b1 = broaden(c1[s], lam0, dlam, K.PRIMARY_VMACRO_KMS)
                    a2 = broaden(f2[s], lam0, dlam, K.SECONDARY_VMACRO_KMS)
                    b2 = broaden(c2[s], lam0, dlam, K.SECONDARY_VMACRO_KMS)
                    mod = (a1 + a2) / (b1 + b2)
                    o = obs[i][s]
                    g = np.isfinite(o) & np.isfinite(mod)
                    tot += float(np.sum((mod[g] - o[g]) ** 2)); npts += int(g.sum())
            rows.append(dict(dT1=k1[1], dg1=k1[2], dT2=k2[1], dg2=k2[2],
                             feh=k1[3], chi2=tot, rms=float(np.sqrt(tot / npts))))
    rows.sort(key=lambda r: r["chi2"])
    with open(args.out, "wb") as f:
        pickle.dump({"rows": rows, "ref_epoch": ref}, f)

    base_T1, base_g1 = K.PRIMARY_TEFF, 2.9157
    base_T2, base_g2 = K.SECONDARY_TEFF, 3.5389
    print(f"\n{len(rows)} parameter combinations\n")
    print(f"{'rank':<5}{'Teff1':>8}{'logg1':>8}{'Teff2':>8}{'logg2':>8}{'[Fe/H]':>8}{'rms':>9}")
    for n, r in enumerate(rows[:8]):
        print(f"{n+1:<5}{base_T1+r['dT1']:>8.0f}{base_g1+r['dg1']:>8.3f}"
              f"{base_T2+r['dT2']:>8.0f}{base_g2+r['dg2']:>8.3f}{r['feh']:>8.2f}{r['rms']:>9.5f}")
    nom = [r for r in rows if r["dT1"]==0 and r["dg1"]==0 and r["dT2"]==0
           and r["dg2"]==0 and abs(r["feh"]-K.PRIMARY_FEH) < 1e-9]
    if nom:
        print(f"\nliterature/PHOEBE nominal: rms {nom[0]['rms']:.5f} "
              f"(rank {rows.index(nom[0])+1}/{len(rows)})")
    print(f"\nedge check — best at grid bounds?")
    b = rows[0]
    for key, grid in (("dT1", G["dteffs"]), ("dg1", G["dloggs"]),
                      ("dT2", G["dteffs"]), ("dg2", G["dloggs"]), ("feh", G["fehs"])):
        if np.isclose(b[key], min(grid)) or np.isclose(b[key], max(grid)):
            print(f"  {key}={b[key]:+g} AT EDGE of [{min(grid):g}, {max(grid):g}]")


if __name__ == "__main__":
    main()
