"""Figures F1 and F2 of the paper (Appendix F): the released emulator TPayne-spice-harps-interp v1.2.0 against
Turbospectrum over H-beta, at mu = 0.1, 0.4, 0.7 and 1, for two parameter sets not in the training grid:

  F1  emulator_hbeta_v120.pdf       8000 K, log g 4.0, [Fe/H] 0 (the referee's test point)
  F2  emulator_hbeta_v120_7750.pdf  7750 K, log g 4.25, [Fe/H] 0 (between MARCS gravity nodes; interpolated atmosphere)

Inputs, results/interp/hbeta/ (from the Hugging Face dataset mjablonska/spice-paper-data;
run tutorial/paper_results/fetch_paper_data.py --only hbeta):
  offnode_ts.npz              Turbospectrum normalised intensities on interpolated MARCS atmospheres (offnode_synth.py)
  offnode_emu_sw_pen1z.npz    the v1.2.0 emulator at the same parameters and mu (intensity and continuum channels)

    python make_paper_hbeta_figures.py [--out figures]

Prints the residual statistics quoted in Appendix F (per cent of the continuum).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INTERP = os.path.join(HERE, "results", "interp", "hbeta")
EMU_FILE = "offnode_emu_sw_pen1z.npz"
MODEL = "emulator v1.2.0"
R_CONV = 100000.0
CASES = (("node_8000_4.0", "8000 K, log g 4.0, [Fe/H] 0", "emulator_hbeta_v120.pdf"),
         ("mid_7750_4.25", "7750 K, log g 4.25, [Fe/H] 0", "emulator_hbeta_v120_7750.pdf"))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))



def gauss_resolution(y, wl, R):
    """Gaussian instrumental profile of resolving power R on a uniform 0.01 A grid (edge-padded)."""
    s = wl.mean() / R / 2.3548 / 0.01; h = int(np.ceil(5 * s)); x = np.arange(-h, h + 1)
    k = np.exp(-0.5 * (x / s) ** 2)
    return np.convolve(np.pad(y, (h, h), mode="edge"), k / k.sum(), mode="valid")


def hbeta_figure(d, out_pdf, zoom=(4880.0, 4885.0), r_conv=100000.0, emu_file=EMU_FILE, model=MODEL,
                 name="node_8000_4.0", star="8000 K, log g 4.0, [Fe/H] 0"):
    """Turbospectrum vs the emulator for one parameter set.
    Left: four mu over the H-beta profile, residuals below. Right: a few-angstrom zoom of one wing at mu = 1, showing
    that the metal lines on the wing are in Turbospectrum too, and the residual raw and after convolving both spectra
    to R = r_conv (it is not removed by any realistic instrumental profile)."""
    ts = np.load(os.path.join(d, "offnode_ts.npz")); em = np.load(os.path.join(d, emu_file))
    wl = ts[f"{name}/wl"]; mus = ts[f"{name}/mus"]; m = (wl >= 4800) & (wl <= 4920); w = wl[m]
    wing = ((w > 4830) & (w < 4850)) | ((w > 4875) & (w < 4895)); z = (w >= zoom[0]) & (w <= zoom[1])
    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[3, 1.4], hspace=0.08, wspace=0.16)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    b1 = fig.add_subplot(gs[0, 1]); b2 = fig.add_subplot(gs[1, 1], sharex=b1)
    out = {}
    for k, mu in enumerate(mus):
        tN = ts[f"{name}/ts_N"][k][m].astype(float); eN = (em[f"{name}/emu_I"][k] / em[f"{name}/emu_C"][k])[m].astype(float)
        rc = gauss_resolution(eN, w, r_conv) - gauss_resolution(tN, w, r_conv)
        off = 0.25 * k
        a1.plot(w, eN + off, lw=0.7, label=f"$\\mu$ = {mu:.1f}"); a1.plot(w, tN + off, color="k", lw=0.4, alpha=0.8)
        a2.plot(w, eN - tN, lw=0.6)
        out[f"mu{mu:.2f}"] = dict(rms_4800_4920=rms(eN - tN), rms_wings=rms((eN - tN)[wing]), max_abs=float(np.abs(eN - tN).max()),
                                  rms_wings_R100k=rms(rc[wing]), max_abs_wings_R100k=float(np.abs(rc[wing]).max()))
        if np.isclose(mu, 1.0):
            b1.plot(w[z], tN[z], color="k", lw=1.3, label="Turbospectrum")
            b1.plot(w[z], eN[z], color="tab:red", lw=1.0, ls="--", label="emulator")
            b2.plot(w[z], (eN - tN)[z], color="0.55", lw=0.8, label="raw")
            b2.plot(w[z], rc[z], color="tab:red", lw=1.3, label=f"both at R = {r_conv / 1000:.0f}k")
            out["zoom_mu1"] = dict(window=list(zoom), rms_raw=rms((eN - tN)[z]), rms_R100k=rms(rc[z]),
                                   ts_line_structure_rms=float(np.std(tN[z])))
    a1.axvspan(*zoom, color="0.85", zorder=0)
    a1.set_ylabel("normalised intensity (+offset)"); a1.legend(fontsize=8, loc="lower left")
    a1.set_title(star + ": Turbospectrum (black) vs " + model + " (colour)", fontsize=10, loc="left")
    a2.axhline(0, color="k", lw=0.5); a2.set_ylim(-0.04, 0.04); a2.set_ylabel("emu - TS"); a2.set_xlabel("wavelength [$\\AA$]")
    plt.setp(a1.get_xticklabels(), visible=False)
    b1.set_title(f"zoom, $\\mu$ = 1 (shaded band at left)", fontsize=10, loc="left"); b1.legend(fontsize=8, loc="lower left")
    b2.axhline(0, color="k", lw=0.5); b2.set_ylim(-0.04, 0.04); b2.legend(fontsize=8, loc="lower left", ncol=2)
    b2.set_xlabel("wavelength [$\\AA$]"); plt.setp(b1.get_xticklabels(), visible=False)
    fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for name, star, out in CASES:
        ref = hbeta_figure(INTERP, os.path.join(a.out, out), emu_file=EMU_FILE, model=MODEL, r_conv=R_CONV, name=name, star=star)
        print(f"{star} -> {out}")
        for k, v in ref.items():
            print("  ", k, {q: (round(100 * x, 3) if not isinstance(x, list) else x) for q, x in v.items()})


if __name__ == "__main__":
    main()
