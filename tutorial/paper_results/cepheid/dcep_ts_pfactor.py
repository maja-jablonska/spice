"""Projection factor per RV estimator for the SPICE + Turbospectrum delta Cep spectra.

Reads synth_*.npz from `dcep_ts_synthesis.py` and measures the disc-integrated
velocity of every phase with the four estimators of `cepheid_rv.py` (Gaussian
core, bisector, CCF, CCF against a template broadened to the epoch's wedge
width) in each window, in two modes:

  rest    against the same phase's rest-frame twin (same intensities, pulsation
          velocity zeroed): absolute velocity, no template mismatch from the
          phase-dependent line strengths; p = sum(rv vpuls) / sum(rv^2)
  phase0  against the phase-0 spectrum, truth zeroed at phase 0 (what an
          observer with one template does); p from `slope_p`

Truth is the mesh's own -dR/dt on the LOS axis (`vpuls`), so SPIPS's p never
enters. Velocities are in SPICE's LOS sign (receding positive); the CCF sign is
calibrated once per window by an injected +10 km/s shift of the rest spectrum.

The fixed +-1.5 A Fe I windows (LINES) are all blended in these spectra; pass
--lines with isolated-line windows from `dcep_ts_line_select.py` instead.

Usage: python dcep_ts_pfactor.py OUT/synth_*.npz [--lines lines.csv] [--csv table.csv] [--plot fig.png]
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cepheid_rv import C_KMS, broaden_template_gauss, gauss_core_center, line_bisector, rv_fft, slope_p  # noqa: E402

LINES = {"Fe I 5005.712": 5005.712, "Fe I 5012.068": 5012.068, "Fe I 5018.027": 5018.027}
HALF_WIDTH = 1.5                         # A, per-line window (as dcep_pfactor_overlay.py)
WINDOWS = {"Mg b 5165-5190": (5165.0, 5190.0)}
METHODS = ("Gauss core", "Bisector", "CCF", "broad CCF")
# single-line CCF: the template is the line alone (flux 1 outside its window), the observed spectrum
# runs PAD_A beyond the window so the shifted line stays inside, lags limited to LINE_VMAX
PAD_A = 0.6
LINE_VMAX = 40.0


def shift(wl, f, v):
    """Spectrum Doppler-shifted by v km/s (receding positive), linear interpolation, edges held."""
    return np.interp(wl / (1 + v / C_KMS), wl, f)


def ccf_sign(wl, f, inner=None):
    t, vmax = (f, 150.0) if inner is None else (np.where(inner, f, 1.0), LINE_VMAX)
    v = rv_fft(t, shift(wl, f, 10.0), wl, velocity_max=vmax)
    # the test only fixes the CCF sign; a masked single-line window whose neighbour leaks in recovers a 10 km/s shift
    # to a few per cent (Fe I 5005.7: 10.57), which is a property of that window, not a sign error
    assert abs(abs(v) - 10.0) < 3.0, f"CCF self-test returned {v:.2f} for a 10 km/s shift"
    if abs(abs(v) - 10.0) >= 0.5:
        print(f"WARNING: CCF self-test returned {v:.2f} km/s for a 10 km/s shift (window {wl[0]:.1f}-{wl[-1]:.1f} A)", flush=True)
    return float(np.sign(v))


def measure(wl, spec, ref, vbroad, lam0=None, inner=None):
    """Velocity of spec relative to ref with the four estimators (km/s, receding positive).

    For a single line (lam0 given) `inner` marks its window inside the padded range `wl`:
    Gauss core and bisector see only the window, the CCF template is masked to it."""
    out = dict.fromkeys(METHODS, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vmax = 150.0
        if lam0 is not None:
            wi, si, ri = wl[inner], spec[inner], ref[inner]
            c, c0 = gauss_core_center(wi, si, lam0), gauss_core_center(wi, ri, lam0)
            out["Gauss core"] = (c - c0) / c0 * C_KMS
            b, b0 = line_bisector(wi, si, lam0), line_bisector(wi, ri, lam0)
            out["Bisector"] = (b - b0) / b0 * C_KMS
            vmax = LINE_VMAX
        mask = (lambda t: t) if inner is None else (lambda t: np.where(inner, t, 1.0))
        try:
            out["CCF"] = rv_fft(mask(ref), spec, wl, velocity_max=vmax)
            out["broad CCF"] = rv_fft(mask(broaden_template_gauss(wl, ref, vbroad)), spec, wl, velocity_max=vmax)
        except Exception:
            pass
    return out


def read_lines(path):
    """{name: (lo, hi, lam0)} from a `dcep_ts_line_select.py` CSV."""
    with open(path) as fh:
        return {f"line {r['name']}": (float(r["lo"]), float(r["hi"]), float(r["lam0"])) for r in csv.DictReader(fh)}


def analyse(path, lines=None):
    d = np.load(path, allow_pickle=False)
    wl = d["wl"]; norm = d["flux"] / d["cont"]; rest = d["rest_flux"] / d["rest_cont"]
    vpuls = d["vpuls"]; n = len(vpuls)
    rows = []
    if lines is None:
        lines = {name: (lam0 - HALF_WIDTH, lam0 + HALF_WIDTH, lam0) for name, lam0 in LINES.items()}
    regions = [(name, (lo, hi), lam0) for name, (lo, hi, lam0) in lines.items()]
    regions += [(name, lim, None) for name, lim in WINDOWS.items()]
    regions += [("full window", (wl[0] + 2, wl[-1] - 2), None)]
    for name, (lo, hi), lam0 in regions:
        m = (wl >= lo) & (wl <= hi)
        if m.sum() < 20:
            continue
        inner = None
        if lam0 is not None:
            m = (wl >= lo - PAD_A) & (wl <= hi + PAD_A)
            inner = (wl[m] >= lo) & (wl[m] <= hi)
        wlm = wl[m]
        sgn = ccf_sign(wlm, rest[0][m], inner)
        for mode in ("rest", "phase0"):
            rv = {k: np.full(n, np.nan) for k in METHODS}
            for k in range(n):
                ref = rest[k][m] if mode == "rest" else norm[0][m]
                vb = vpuls[k] if mode == "rest" else vpuls[k] - vpuls[0]
                if mode == "phase0" and k == 0:
                    for meth in METHODS: rv[meth][0] = 0.0
                    continue
                r = measure(wlm, norm[k][m], ref, vb, lam0, inner)
                for meth in METHODS:
                    rv[meth][k] = r[meth] * (sgn if meth in ("CCF", "broad CCF") else 1.0)
            truth = vpuls if mode == "rest" else vpuls - vpuls[0]
            for meth in METHODS:
                v = rv[meth]
                g = np.isfinite(v) & (np.abs(v) > 1e-3)
                if mode == "rest":
                    p = float(np.sum(v[g] * truth[g]) / np.sum(v[g] ** 2)) if g.sum() >= 5 else np.nan
                else:
                    p = slope_p(v, truth)
                resid = float(np.sqrt(np.nanmean((p * v - truth) ** 2))) if np.isfinite(p) else np.nan
                rows.append(dict(variant=str(d["variant"]), ld=str(d["ld"]), window=name, mode=mode, method=meth,
                                 p=p, rms_kms=resid, n=int(g.sum()), rv=v, truth=truth, phase=d["phase"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--lines", type=Path, default=None,
                    help="per-line windows from dcep_ts_line_select.py (replaces the fixed +-1.5 A Fe I windows)")
    args = ap.parse_args()
    lines = read_lines(args.lines) if args.lines else None
    rows = [r for f in args.files for r in analyse(f, lines)]
    key = lambda r: (r["variant"], r["ld"], r["window"], r["mode"])
    last = None
    for r in sorted(rows, key=key):
        if key(r) != last:
            ps = {x["method"]: x["p"] for x in rows if key(x) == key(r)}
            print(f"{r['variant']:13s} {r['ld']:7s} {r['window']:15s} {r['mode']:6s} "
                  + "  ".join(f"{m} {ps.get(m, np.nan):.3f}" for m in METHODS))
            last = key(r)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["variant", "ld", "window", "mode", "method", "p", "rms_kms", "n"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print("wrote", args.csv)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        combos = sorted({(r["variant"], r["ld"]) for r in rows})
        windows = list(dict.fromkeys(r["window"] for r in rows))
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 + 0.35 * len(windows) * len(combos)), sharex=True)
        for ax, mode in zip(axes, ("rest", "phase0")):
            y = 0; ticks = []
            for combo in combos:
                for win in windows:
                    for j, meth in enumerate(METHODS):
                        sel = [r for r in rows if (r["variant"], r["ld"]) == combo and r["window"] == win
                               and r["mode"] == mode and r["method"] == meth]
                        if sel and np.isfinite(sel[0]["p"]):
                            ax.plot(sel[0]["p"], y, "o", color=f"C{j}", label=meth if y == 0 else None)
                    ticks.append((y, f"{combo[0]} {combo[1]} | {win}")); y += 1
            ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=7)
            ax.axvline(1.5, color="0.6", lw=0.8, ls=":"); ax.set_title(f"mode: {mode}"); ax.set_xlabel("p")
            ax.invert_yaxis()
        axes[0].legend(fontsize=7, loc="best")
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
