"""Fit TZ For radii and inclination to Clausen's uvby photometry.

Geometry is a PHOEBE-only problem -- the atmospheres barely move an eclipse
*shape* -- so a full (R1, R2, incl) grid costs seconds per model instead of the
~70 min the aemu hybrid needs. Run the emulator once at the answer, not inside
the loop.

Structure of the fit:

* expensive loop is over (R1, R2, incl) only;
* for each geometry the light curve is computed once on a dense phase grid,
  then third light, the magnitude zero point and a phase offset are solved
  post-hoc (they are cheap and near-linear);
* the phase offset is fitted rather than assumed, which sidesteps the
  T_P/t0_supconj convention entirely -- the observed deep minimum sits 0.158 in
  phase from T_P_HJD and I would rather measure that than argue about it.

The eclipses are PARTIAL for TZ For, so depth alone cannot separate inclination
from the radii (measured: 0.5 deg of inclination moves the deep eclipse
0.11-0.14 mag, a 5% radius change only 0.014-0.039). The degeneracy is broken by
eclipse *shape*, which is why this fits all 1169 points rather than summary
depths. The output therefore reports the confidence region, not just a best fit.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tzfor_constants as K  # noqa: E402


def observed(bands=("b", "y")):
    df = pd.read_csv(HERE / "tzfor_lightcurve.csv")
    hjd = df["HJD"].values + 2450000.0        # file stores HJD-2450000
    phase = ((hjd - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
    return phase, {b: df[b].values for b in bands}


def model_curve(R1, R2, incl, n_phase, ntriangles):
    """Dense-phase PHOEBE light curves in b and y for one geometry."""
    import tzfor_phoebe_aemu_photometry as M
    b = M.build_bundle(ntriangles, "roche")
    b.set_value("requiv@primary@component", R1)
    b.set_value("requiv@secondary@component", R2)
    b.set_value_all("incl@binary", incl)
    t0 = b.get_value("t0_supconj@binary@component")
    ts = t0 + np.linspace(0.0, K.PERIOD_DAYS, n_phase, endpoint=False)
    for band, ds in (("Stromgren:b", "lcb"), ("Stromgren:y", "lcy")):
        b.add_dataset("lc", compute_times=ts, passband=band, dataset=ds)
        b.set_value_all("pblum_mode", dataset=ds, value="absolute")
    b.run_compute(irrad_method="horvat", ltte=False)
    ph = ((ts - t0) % K.PERIOD_DAYS) / K.PERIOD_DAYS
    order = np.argsort(ph)
    out = {}
    for key, ds in (("b", "lcb"), ("y", "lcy")):
        out[key] = np.asarray(b.get_value(f"fluxes@{ds}@model"))[order]
    return ph[order], out


def chi2_for(ph_m, flux_m, ph_o, mag_o, l3_grid, dphi_grid):
    """Best chi2 over third light, zero point and phase offset, per band.

    Third light dilutes: F -> F + L3*<F>, so it shrinks every eclipse by the
    same factor -- which is exactly why it is degenerate with the depth scale
    and must be marginalised rather than fixed.
    """
    best = None
    for dphi in dphi_grid:
        interp = {k: np.interp((ph_o - dphi) % 1.0, ph_m, v, period=1.0)
                  for k, v in flux_m.items()}
        for l3 in l3_grid:
            tot = 0.0
            for k, f in interp.items():
                fl = f + l3 * np.median(f)
                m = -2.5 * np.log10(fl / np.median(fl))
                resid = mag_o[k] - m
                m_adj = m + np.median(resid)          # magnitude zero point
                tot += float(np.sum((mag_o[k] - m_adj) ** 2))
            if best is None or tot < best[0]:
                best = (tot, float(l3), float(dphi))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-r1", type=int, default=7)
    ap.add_argument("--n-r2", type=int, default=5)
    ap.add_argument("--n-incl", type=int, default=7)
    ap.add_argument("--dr1", type=float, default=0.6, help="+/- Rsun on R1")
    ap.add_argument("--dr2", type=float, default=0.4, help="+/- Rsun on R2")
    ap.add_argument("--dincl", type=float, default=0.6, help="+/- deg")
    ap.add_argument("--n-phase", type=int, default=600)
    ap.add_argument("--ntriangles", type=int, default=800)
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "radii_fit.pkl"))
    args = ap.parse_args()

    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "1"
    import phoebe
    phoebe.logger(clevel="ERROR")

    ph_o, mag_o = observed()
    print(f"{len(ph_o)} Clausen points in b and y", flush=True)

    R1s = K.PRIMARY_RADIUS + np.linspace(-args.dr1, args.dr1, args.n_r1)
    R2s = K.SECONDARY_RADIUS + np.linspace(-args.dr2, args.dr2, args.n_r2)
    incls = K.INCL_DEG + np.linspace(-args.dincl, args.dincl, args.n_incl)
    l3_grid = np.linspace(0.0, 0.30, 16)
    # The model's phase zero is t0_supconj = T_P - P/2, so its deep eclipse sits
    # at model phase 0.5 while the observed one is 0.158 from T_P: the offset
    # needed is ~0.658, NOT the 0.158 measured against T_P. Getting this wrong
    # once cost a whole 245-model grid, so the search brackets 0.658 generously
    # and the result is checked against the window edges below.
    dphi_grid = np.linspace(0.60, 0.72, 61)

    print(f"grid: {len(R1s)} R1 x {len(R2s)} R2 x {len(incls)} incl "
          f"= {len(R1s)*len(R2s)*len(incls)} PHOEBE models", flush=True)
    rows, t_start, k = [], time.time(), 0
    for R1 in R1s:
        for R2 in R2s:
            for inc in incls:
                t0 = time.time()
                ph_m, flux_m = model_curve(R1, R2, inc, args.n_phase, args.ntriangles)
                c2, l3, dphi = chi2_for(ph_m, flux_m, ph_o, mag_o, l3_grid, dphi_grid)
                rows.append(dict(R1=float(R1), R2=float(R2), incl=float(inc),
                                 chi2=c2, l3=l3, dphi=dphi))
                k += 1
                print(f"  [{k}/{len(R1s)*len(R2s)*len(incls)}] R1={R1:.3f} "
                      f"R2={R2:.3f} i={inc:.3f}  chi2={c2:.4f} L3={l3:.3f} "
                      f"dphi={dphi:.4f}  ({time.time()-t0:.1f}s)", flush=True)
                with open(args.out, "wb") as f:      # checkpoint every model
                    pickle.dump({"rows": rows, "args": vars(args)}, f)
    print(f"\ntotal {time.time()-t_start:.0f} s", flush=True)
    # Guard against exactly the failure that wasted the first run: a parameter
    # pinned to its grid edge means the optimum is outside the box, so the
    # "best fit" is the bound talking, not the data.
    best = min(rows, key=lambda r: r["chi2"])
    edges = []
    for key, grid in (("R1", R1s), ("R2", R2s), ("incl", incls),
                      ("l3", l3_grid), ("dphi", dphi_grid)):
        if np.isclose(best[key], grid[0]) or np.isclose(best[key], grid[-1]):
            edges.append(f"{key}={best[key]:.4g} at edge of "
                         f"[{grid[0]:.4g}, {grid[-1]:.4g}]")
    if edges:
        print("\n*** WARNING: best fit is railed against a grid bound —"
              " NOT a measurement ***")
        for e in edges:
            print(f"    {e}")
    rms = np.sqrt(best["chi2"] / (2 * len(ph_o)))
    print(f"\nrms {rms:.4f} mag   (Clausen scatter ~0.0041 mag)")
    print(f"BEST: R1={best['R1']:.3f} R2={best['R2']:.3f} incl={best['incl']:.3f} "
          f"L3={best['l3']:.3f} dphi={best['dphi']:.4f} chi2={best['chi2']:.4f}")
    print(f"literature: R1={K.PRIMARY_RADIUS} R2={K.SECONDARY_RADIUS} "
          f"incl={K.INCL_DEG}")


if __name__ == "__main__":
    main()
