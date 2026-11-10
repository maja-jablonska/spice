"""Per-phase atmosphere parameters for the delta Cep Turbospectrum intensity run.

Writes one row per (variant, phase) for `dcep_ts_intensities.py` (run on Gadi):
Teff(phase) from the SPIPS MODEL table, log g either the table's static
G M / R^2 or the dynamic g_eff = G M / R^2 + d^2R/dt^2 (`effective_logg`, same
SPIPS mass 4.726 M_sun), and microturbulence either the Luck (2018) curve
`xi_micro_deltaCep` or a constant. Phases are SPIPS phases (MJD0 48304.7362 =
maximum light); the Luck curve is on the GCVS maximum-light ephemeris, which
agrees to ~0.01 in phase.

Variants:
  geff_xiphi    dynamic g_eff, xi(phase)      (baseline)
  geff_xi3      dynamic g_eff, xi = 3 km/s    (xi(phase) control)
  static_xiphi  static logg,   xi(phase)      (g_eff control)

  paper         (--paper) the paper's literature model instead of the three variants above: Teff from the
                Trahin et al. (2021) curve (Fourier order 4), log g and xi from second-order Fourier series through
                Luck (2018), [Fe/H] on the grid's scale (-0.14), radius = SPIPS R(phase) scaled to 279.5 pc;
                the same curves the literature model uses for the paper's photometric and spectroscopic comparison

Usage: python dcep_ts_phase_table.py [--n-phase 40] [--out dcep_ts_phases.csv] [--paper]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from cepheid_phase import effective_logg, spips_mass, xi_micro_deltaCep

HERE = Path(__file__).resolve().parent
FEH = 0.06
XI_CONST = 3.0


def fourier(x, y, order):
    """Least-squares Fourier series (the basis used throughout the delta Cep model)."""
    A = lambda q: np.stack([np.ones_like(q)] + [f(2 * np.pi * k * q) for k in range(1, order + 1) for f in (np.cos, np.sin)], 1)
    c = np.linalg.lstsq(A(np.asarray(x, float)), np.asarray(y, float), rcond=None)[0]
    return lambda q: float((A(np.atleast_1d(np.asarray(q, float))) @ c)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", type=Path, default=HERE / "delta_cep.fits")
    ap.add_argument("--n-phase", type=int, default=40)
    ap.add_argument("--out", type=Path, default=HERE / "dcep_ts_phases.csv")
    ap.add_argument("--paper", action="store_true", help="write only the 'paper' variant (literature model of the paper)")
    ap.add_argument("--feh", type=float, default=-0.14, help="[Fe/H] of the 'paper' variant")
    ap.add_argument("--distance-scale", type=float, default=279.5 / 274.0, help="radius scale of the 'paper' variant")
    args = ap.parse_args()

    with fits.open(args.fits) as h:
        model = h["MODEL"].data
        period = float(h[0].header["HIERARCH PARAM PERIOD"])
    ph, logg_eff, logg_static = effective_logg(model, period)
    order = np.argsort(model["PHASE"])
    teff_tab = np.asarray(model["Teff"], dtype=float)[order]

    def periodic(y):
        x = np.concatenate([ph - 1, ph, ph + 1])
        return interp1d(x, np.tile(y, 3), kind="cubic")

    R_tab = np.asarray(model["R"], dtype=float)[order]
    teff_i, geff_i, gstat_i, R_i = periodic(teff_tab), periodic(logg_eff), periodic(logg_static), periodic(R_tab)
    phases = np.arange(args.n_phase) / args.n_phase
    variants = {
        "geff_xiphi": (geff_i, lambda p: xi_micro_deltaCep(p)),
        "geff_xi3": (geff_i, lambda p: XI_CONST),
        "static_xiphi": (gstat_i, lambda p: xi_micro_deltaCep(p)),
    }
    rows = []
    if args.paper:
        lit = HERE / "literature"
        tt = np.loadtxt(lit / "dcep_trahin2021_teff_curve.txt"); lk = np.loadtxt(lit / "dcep_luck2018_table6.txt")
        f_t, f_g, f_x = fourier(tt[:, 0], tt[:, 1], 4), fourier(lk[:, 0], lk[:, 2], 2), fourier(lk[:, 0], lk[:, 3], 2)
        variants = {"paper": None}
        for k, p in enumerate(phases):
            rows.append(dict(variant="paper", index=k, phase=f"{p:.5f}", teff=int(round(float(f_t(p)))),
                             logg=f"{float(f_g(p)):.3f}", feh=f"{args.feh:.2f}", vmicro=f"{float(f_x(p)):.3f}",
                             radius_rsun=f"{float(R_i(p)) * args.distance_scale:.4f}"))
        variants_iter = {}
    else:
        variants_iter = variants
    for variant, (logg_i, xi) in variants_iter.items():
        for k, p in enumerate(phases):
            rows.append(dict(variant=variant, index=k, phase=f"{p:.5f}", teff=int(round(float(teff_i(p)))),
                             logg=f"{float(logg_i(p)):.3f}", feh=f"{FEH:.2f}", vmicro=f"{float(xi(p)):.3f}",
                             radius_rsun=f"{float(R_i(p)):.4f}"))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"SPIPS mass {spips_mass(model):.3f} M_sun, P {period} d; wrote {len(rows)} rows to {args.out}")
    for variant in variants:
        sel = [r for r in rows if r["variant"] == variant]
        lg = np.array([float(r["logg"]) for r in sel]); xi = np.array([float(r["vmicro"]) for r in sel])
        te = np.array([r["teff"] for r in sel])
        print(f"  {variant:13s} Teff {te.min()}-{te.max()}  logg {lg.min():.3f}-{lg.max():.3f}  xi {xi.min():.2f}-{xi.max():.2f}")


if __name__ == "__main__":
    main()
