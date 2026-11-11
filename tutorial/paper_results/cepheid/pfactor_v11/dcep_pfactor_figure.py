"""The paper's p-factor figure (cepheid_p_factor.pdf) from the 2026-11-03 synthesis of the paper's delta Cep model.

Recovered radial velocity against the on-mesh pulsation velocity on Fe I 5018.027 (+-1.5 A window), all relative to
phase 0 (the estimators measure every phase against the phase-0 spectrum, as an observer with one template does), for
the three syntheses of dcep_ts_synthesis.py:

    mu = 1, no LD     Turbospectrum's disc-centre spectrum on a uniform disc (--ld lin:0): p must approach 3/2
    Turbospectrum     Turbospectrum I(lambda, mu) at every phase (--ld atm)
    Transformer       the TPayne emulator v1.1.0 at the same inputs (--emulator)

The uniform disc is measured with the CCF against the broadened template, the two limb-darkened syntheses with the
plain CCF (as in the paper's caption). Velocities and p from dcep_ts_pfactor.analyse (mode 'phase0').

    python dcep_pfactor_figure.py [--dir out] [--out ../../paper_plots/cepheid/cepheid_p_factor.pdf]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dcep_ts_pfactor import analyse  # noqa: E402

LINE = "Fe I 5018.027"
SOURCES = [("$\\mu$=1, no LD", "synth_paper_lin0.npz", "broad CCF", "#151515", "-"),
           ("TURBOSPECTRUM", "synth_paper_atm.npz", "CCF", "#2a78d6", "-"),
           ("Emulator", "synth_paper_tpayne.npz", "CCF", "#c2255c", ":")]
TRUTH_COLOUR = "#eb6834"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=HERE / "out")
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "paper_plots" / "cepheid" / "cepheid_p_factor.pdf")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(4.4, 3.3))   # printed at column width: keep text >= 6.5 pt
    truth_drawn = False
    for label, fname, method, colour, ls in SOURCES:
        rows = [r for r in analyse(a.dir / fname) if r["window"] == LINE and r["mode"] == "phase0" and r["method"] == method]
        assert len(rows) == 1, (fname, len(rows))
        r = rows[0]; o = np.argsort(r["phase"])
        if not truth_drawn:
            ax.plot(r["phase"][o], r["truth"][o], color=TRUTH_COLOUR, lw=2.4, solid_capstyle="round",
                    label=r"$V_{\rm puls} - V_{\rm puls}(\varphi{=}0)$ truth")
            truth_drawn = True
        ax.plot(r["phase"][o], r["rv"][o], color=colour, ls=ls, lw=1.8 if ls != "-" else 1.2,
                label=f"{label} — {method}: $p$ = {r['p']:.3f}")
        print(f"{label:16s} {method:9s} p = {r['p']:.3f} (rms {r['rms_kms']:.2f} km/s, {r['n']} phases)")
    ax.set_xlabel("phase", fontsize=10); ax.set_ylabel(r"velocity relative to phase 0 [km$\,$s$^{-1}$]", fontsize=10)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(top=1.65 * ax.get_ylim()[1])   # headroom so the legend sits above the curves
    ax.legend(loc="upper left", fontsize=8.5, frameon=False); ax.tick_params(labelsize=9)
    fig.tight_layout(); fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
