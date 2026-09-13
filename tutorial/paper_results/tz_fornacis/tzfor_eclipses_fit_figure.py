"""Zoom on the two eclipses for the joint-fit light-curve figure of notebook section 7.

Same model and data as ``tzfor_lightcurves_fit`` (the 12-window joint fit, ``paper_models.npz``
from ``tzfor_paper_models.py``), shown only within ±0.03 in phase of each eclipse so the ingress,
egress and bottom can be judged against Clausen's points, with the residuals underneath.

``--dump`` accepts any dump with ``lc_*`` arrays: the 12-window ones store the model on its own
phase frame plus ``dphi``; the full-range ones (``fr_model_*.npz``) store per-band arrays and
``dphi`` inside ``theta``. Both are handled. Pure plotting.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))

ECLIPSES = [("deep eclipse (secondary occulted)", 0.1585), ("shallow eclipse (secondary transits)", 0.6585)]


def load_curves(path):
    """Per band: model phase (T_P frame), model mag, observed phase, observed mag, residual [mag]."""
    d = np.load(path, allow_pickle=True); out = {}
    if "lc_model_b" in d.files:                                   # tzfor_paper_models.py layout
        dphi = float(d["dphi"])
        for b in ("b", "y"):
            out[b] = ((d["lc_model_phase"] + dphi) % 1.0, d[f"lc_model_{b}"], d["obs_phase"], d[f"lc_obs_{b}"], d[f"lc_resid_{b}"])
        label = "12-window joint fit"
    else:                                                          # tzfor_fullrange_fit.py --dump-model layout
        th, pn = d["theta"], list(d["pnames"]); dphi = float(th[pn.index("dphi")])
        for b in ("b", "y"):
            out[b] = ((d[f"lc_{b}_model_phase"] + dphi) % 1.0, d[f"lc_{b}_model_mag"], (d[f"lc_{b}_obs_phase"] + dphi) % 1.0,
                      d[f"lc_{b}_obs_mag"], d[f"lc_{b}_obs_mag"] - d[f"lc_{b}_model_at_obs"])
        label = "full-range joint fit" + (" (exact per-element)" if "synthesis" in d.files and "exact" in str(d["synthesis"]) else "")
    return out, label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=str(HERE / "tzfor_aemu_out" / "paper_models.npz"))
    ap.add_argument("--out", default="tzfor_eclipses_fit"); ap.add_argument("--out-dir", default=str(HERE.parent / "paper_plots" / "tz_fornacis"))
    ap.add_argument("--half-width", type=float, default=0.03, help="phase half-width of each zoom")
    args = ap.parse_args()
    curves, label = load_curves(args.dump)
    near = lambda ph, c: np.abs(((ph - c + 0.5) % 1.0) - 0.5) < args.half_width

    fig, axes = plt.subplots(4, 2, figsize=(12.5, 10.5), gridspec_kw=dict(height_ratios=[3, 1.1, 3, 1.1]))
    for r, b in enumerate(("b", "y")):
        mp, mm, op, om, res = curves[b]
        lev = float(np.min(mm)); mm, om = mm - lev, om - lev            # brightest model point = out of eclipse = zero
        for j, (title, centre) in enumerate(ECLIPSES):
            top, bot = axes[2 * r, j], axes[2 * r + 1, j]
            so, sm = near(op, centre), near(mp, centre); o = np.argsort(mp[sm])
            top.plot(op[so], om[so], ".", color="0.35", ms=4, zorder=1, label=f"Clausen {b}")
            top.plot(mp[sm][o], mm[sm][o], "-", color="C3", lw=1.6, zorder=2, label=f"{label} model")
            top.invert_yaxis(); top.grid(alpha=0.3); top.set_ylabel(f"Δ{b} [mag]")
            top.set_title(f"{title} — Strömgren {b}", fontsize=10); top.tick_params(labelbottom=False)
            if r == 0 and j == 0: top.legend(fontsize=9, loc="lower left")
            oc = 1000 * res[so]
            bot.plot(op[so], oc, ".", color="0.35", ms=4); bot.axhline(0, color="C3", lw=1)
            bot.set_ylim(-22, 22); bot.grid(alpha=0.3); bot.set_ylabel("O−C [mmag]")
            bot.text(0.02, 0.8, f"rms {np.sqrt(np.mean(oc ** 2)):.1f} mmag, {so.sum()} points", transform=bot.transAxes, fontsize=9)
            if r == 1: bot.set_xlabel("phase from $T_P$")
            top.set_xlim(centre - args.half_width, centre + args.half_width); bot.set_xlim(top.get_xlim())
    fig.suptitle(f"TZ For eclipses: Clausen photometry against the {label} model", y=0.995)
    fig.tight_layout()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(out / f"{args.out}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig); print(f"saved {out}/{args.out}.png and .pdf ({label})")


if __name__ == "__main__":
    main()
