"""Isolated lines for per-line p-factors, picked from the synthetic delta Cep spectra themselves.

A line qualifies if, in the rest-frame disc-integrated spectrum (`rest_flux / rest_cont`
of a synth_*.npz from `dcep_ts_synthesis.py`) at EVERY phase:
  - its core is deeper than --min-depth;
  - its extent [a, b] (contiguous pixels below --cont, default 0.98, around the core) holds a
    single local minimum (no partly resolved blend in the profile);
  - the flux stays above --cont over --shift + --gap on both sides of [a, b], so the
    line's own pulsation shift (|v| <= max |vpuls| / p_min) never brings a neighbour into the window.
The window is the union over phases of [a - shift, b + shift]. Blends unresolved inside the
core are invisible here; check the survivors against the line list (`--vald` prints the
lines within +-0.3 A when run where the Turbospectrum line lists live).

Writes CSV (name, lam0, lo, hi, depth_min, depth_max) for `dcep_ts_pfactor.py --lines`.

Usage: python dcep_ts_line_select.py OUT/synth_geff_xiphi_atm.npz [--out dcep_ts_lines.csv] [--plot fig.png]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import argrelmin

HERE = Path(__file__).resolve().parent
C_KMS = 299792.458


def line_extent(f, i, cont):
    """Contiguous pixel range around index i with f < cont."""
    a = i
    while a > 0 and f[a - 1] < cont:
        a -= 1
    b = i
    while b < len(f) - 1 and f[b + 1] < cont:
        b += 1
    return a, b


def candidates(wl, f, min_depth, order):
    """Indices of local minima deeper than min_depth."""
    idx = argrelmin(f, order=order)[0]
    return idx[f[idx] < 1 - min_depth]


def check(wl, f, lam0, cont, shift_A, gap_A, order, track_A):
    """(lo, hi, depth) of the line nearest lam0 in spectrum f if isolated, else None."""
    j = np.argmin(np.abs(wl - lam0))
    lo_s = max(0, j - int(track_A / (wl[1] - wl[0])))
    i = lo_s + np.argmin(f[lo_s:j + int(track_A / (wl[1] - wl[0])) + 1])
    a, b = line_extent(f, i, cont)
    if a == 0 or b == len(f) - 1:
        return None
    if len(argrelmin(f[a:b + 1], order=order)[0]) > 1:
        return None
    pad = shift_A + gap_A
    side = (wl >= wl[a] - pad) & (wl < wl[a]) | (wl > wl[b]) & (wl <= wl[b] + pad)
    if wl[a] - pad < wl[0] or wl[b] + pad > wl[-1] or np.any(f[side] < cont):
        return None
    return wl[a] - shift_A, wl[b] + shift_A, 1 - f[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("synth", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "dcep_ts_lines.csv")
    ap.add_argument("--min-depth", type=float, default=0.2)
    ap.add_argument("--cont", type=float, default=0.98)
    ap.add_argument("--p-min", type=float, default=1.2, help="lowest p expected (sets the shift margin)")
    ap.add_argument("--gap", type=float, default=0.05, help="extra clear margin beyond the shift (A)")
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.synth, allow_pickle=False)
    wl = d["wl"]; rest = d["rest_flux"] / d["rest_cont"]
    vmax = float(np.max(np.abs(d["vpuls"]))) / args.p_min
    order = max(2, int(round(0.03 / (wl[1] - wl[0]))))
    seeds = candidates(wl, rest[0], args.min_depth, order)
    print(f"max line shift {vmax:.1f} km/s; {len(seeds)} minima deeper than {args.min_depth} at phase 0")

    rows = []
    for s in seeds:
        lam0 = wl[s]
        shift_A = lam0 * vmax / C_KMS
        res = [check(wl, f, lam0, args.cont, shift_A, args.gap, order, track_A=0.1) for f in rest]
        if any(r is None for r in res):
            continue
        lo = min(r[0] for r in res); hi = max(r[1] for r in res); dep = [r[2] for r in res]
        rows.append(dict(name=f"{lam0:.3f}", lam0=f"{lam0:.3f}", lo=f"{lo:.3f}", hi=f"{hi:.3f}",
                         depth_min=f"{min(dep):.3f}", depth_max=f"{max(dep):.3f}"))
        print(f"  {lam0:9.3f}  window {lo:.3f}-{hi:.3f} ({hi - lo:.2f} A)  depth {min(dep):.2f}-{max(dep):.2f}")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "lam0", "lo", "hi", "depth_min", "depth_max"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} isolated lines -> {args.out}")

    if args.plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(rows); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
        for ax, r in zip(axes.flat, rows):
            lo, hi = float(r["lo"]), float(r["hi"])
            m = (wl > lo - 0.5) & (wl < hi + 0.5)
            for k in range(0, len(rest), max(1, len(rest) // 4)):
                ax.plot(wl[m], rest[k][m], lw=0.7)
            ax.axvspan(lo, hi, color="0.9", zorder=0); ax.set_ylim(0, 1.05)
            ax.set_title(r["name"], fontsize=8); ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
