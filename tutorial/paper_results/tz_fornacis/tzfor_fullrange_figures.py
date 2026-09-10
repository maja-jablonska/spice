"""Paper figures for the full-range (4450-6750 A) TZ For joint fit.

Reads the model dump written by ``tzfor_fullrange_fit.py --dump-model`` (observed and
model spectra per epoch, per-star contributions, mask, and the Stromgren b, y light-curve
model) and draws the same two figures as for the 12-window fit:

* ``tzfor_spectra_fit_fullrange``: three 40 A windows at the epoch nearest phase 0.25,
  HARPS vs the model, the primary and secondary contributions, residuals, masked pixels shaded;
* ``tzfor_lightcurves_fit_fullrange``: Clausen b, y vs the model with O-C panels.

Pure plotting: no synthesis, no GPU.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tzfor_constants as K

WINDOWS = [(5160.0, 5200.0), (4900.0, 4940.0), (6400.0, 6440.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=str(HERE / "tzfor_aemu_out" / "fr_model_maskall_hp.npz"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "paper_plots" / "tz_fornacis"))
    ap.add_argument("--suffix", default="fullrange"); ap.add_argument("--phase", type=float, default=0.25)
    ap.add_argument("--label", default="full-range masked joint fit"); ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_fullrange_log.npz"))
    args = ap.parse_args()
    d = np.load(args.dump, allow_pickle=True); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    th = d["theta"]; pn = list(d["pnames"]); T1, T2, feh = (float(th[pn.index(k)]) for k in ("Teff1", "Teff2", "feh"))
    wl = 10.0 ** d["logwl"]; obs, mod, prim, sec, good = d["obs"], d["model"], d["primary"], d["secondary"], d["good"]
    phase = ((d["times"] - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
    e = int(np.argmin(np.abs(phase - args.phase)))
    if "masked" not in d.files:                                        # older dumps dropped the masked pixels: restore them from the HARPS file
        H = np.load(args.harps); obs = np.asarray(H["obs"], float)[:obs.shape[0]]
    usable = np.isfinite(obs[e]); masked = usable & ~good[e]          # the mask is the only reason a usable pixel is not fitted

    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(15, 12))
    for ax, (w0, w1) in zip(axes, WINDOWS):
        sel = (wl >= w0) & (wl <= w1)
        res = obs[e] - mod[e]
        ax.plot(wl[sel], obs[e][sel], color="k", lw=0.8, label="HARPS")
        ax.plot(wl[sel], mod[e][sel], color="C3", lw=0.9, label=f"SPICE joint-fit model ({args.label})")
        ax.plot(wl[sel], prim[e][sel], color="C0", lw=0.7, alpha=0.8, label="primary contribution")
        ax.plot(wl[sel], sec[e][sel], color="C1", lw=0.7, alpha=0.8, label="secondary contribution")
        ax.plot(wl[sel], res[sel] + 0.25, color="0.4", lw=0.7, label="residual + 0.25")
        m = masked & sel
        if m.any():
            edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.axvspan(wl[a], wl[min(b, wl.size - 1)], color="C3", alpha=0.12, lw=0)
        r_all = np.sqrt(np.nanmean(res[sel & usable] ** 2)); r_fit = np.sqrt(np.nanmean(res[sel & good[e]] ** 2))
        note = f"rms {100 * r_all:.1f}% all, {100 * r_fit:.1f}% unmasked; {100 * m.sum() / max((sel & usable).sum(), 1):.0f}% masked (shaded)" if masked.any() else f"rms {100 * r_all:.1f}% (no mask)"
        ax.text(0.01, 0.04, note, transform=ax.transAxes, fontsize=9)
        ax.set_xlim(w0, w1); ax.set_ylim(0.1, 1.07); ax.set_ylabel("normalised flux"); ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", ncol=5, fontsize=8); axes[-1].set_xlabel("wavelength [Å]")
    axes[0].set_title(f"TZ For at phase {phase[e]:.2f}: {args.label} (Teff1 {T1:.0f} K, Teff2 {T2:.0f} K, [Fe/H] {feh:+.2f}) vs HARPS")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"tzfor_spectra_fit_{args.suffix}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), gridspec_kw=dict(height_ratios=[3, 1]), sharex="col")
    for j, b in enumerate(("b", "y")):
        dphi = float(th[pn.index("dphi")])                             # model phases count from conjunction; observations from T_P
        mp, mm = (d[f"lc_{b}_model_phase"] + dphi) % 1.0, d[f"lc_{b}_model_mag"]; op, om, mo = (d[f"lc_{b}_obs_phase"] + dphi) % 1.0, d[f"lc_{b}_obs_mag"], d[f"lc_{b}_model_at_obs"]
        o = np.argsort(mp); ax = axes[0, j]
        ax.plot(op, om, ".", color="0.35", ms=3, label=f"Clausen {b}")
        ax.plot(mp[o], mm[o], color="C3", lw=1.2, label=f"{args.label} model")
        ax.invert_yaxis(); ax.set_ylabel(f"Δ{b} [mag]"); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
        rax = axes[1, j]; oc = 1000 * (om - mo)
        rax.plot(op, oc, ".", color="0.35", ms=3); rax.axhline(0, color="C3", lw=1); rax.set_ylim(-25, 25)
        rax.text(0.02, 0.85, f"rms {np.sqrt(np.mean(oc ** 2)):.1f} mmag", transform=rax.transAxes, fontsize=9)
        rax.set_xlabel("phase from $T_P$"); rax.set_ylabel("O−C [mmag]"); rax.grid(alpha=0.3)
    fig.suptitle(f"TZ For Strömgren b, y: Clausen photometry vs the {args.label} model"); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"tzfor_lightcurves_fit_{args.suffix}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"figures: {out}/tzfor_spectra_fit_{args.suffix}.* and tzfor_lightcurves_fit_{args.suffix}.* (epoch {e}, phase {phase[e]:.3f})")


if __name__ == "__main__":
    main()
