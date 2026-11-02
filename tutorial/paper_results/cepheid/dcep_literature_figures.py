"""The three delta Cep figures of the paper, from the literature models (dcep_joint_fit.py --exact-all at --maxiter 0).

    dcep_photometry.pdf    Tycho V_T against the exact per-element model (no zero point adjusted), residuals beneath
    dcep_spectra.pdf       HARPS-N spectra at three phases in the 4480-4520 A window against the model, residuals
    dcep_teff_vmacro.pdf   the two literature Teff curves (Trahin 2021; Fourier series through Luck 2018) with the
                           Andrievsky et al. (2005) points and the HARPS-N epochs; the Luck broadening used, below

    python dcep_literature_figures.py --trahin TRAHIN.pkl [--luck LUCK.pkl] [--out DIR]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LIT = HERE / "literature"
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 150})
C = {"data": "0.3", "model": "C3", "luck": "C0", "andr": "k", "resid": "0.35"}


def fourier_fit(x, y, order):
    A = [np.ones_like(x)] + [f(2 * np.pi * k * x) for k in range(1, order + 1) for f in (np.cos, np.sin)]
    c = np.linalg.lstsq(np.stack(A, 1), y, rcond=None)[0]
    return lambda p: c[0] + sum(c[2 * k - 1] * np.cos(2 * np.pi * k * p) + c[2 * k] * np.sin(2 * np.pi * k * p) for k in range(1, order + 1))


def vt_model(r):
    o = r["obs"]["VT"]; pm = np.asarray(r["phases_model"]); me = np.asarray(r["mags_exact"]["VT"])
    ph, mag, err = (np.asarray(o[k], float) for k in ("phase", "mag", "err"))
    return pm, me, ph % 1.0, mag, err, mag - np.interp(ph % 1.0, pm, me, period=1.0)


def fig_photometry(r, out):
    pm, me, ph, mag, err, res = vt_model(r)
    o = np.argsort(pm); pp = np.r_[pm[o], pm[o][0] + 1]; mm = np.r_[me[o], me[o][0]]
    fig, ax = plt.subplots(2, 1, figsize=(4.6, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.05))
    for sh in (0, 1):
        ax[0].errorbar(ph + sh, mag, err, fmt="o", ms=3, color=C["data"], elinewidth=0.6, lw=0, label="Tycho $V_T$" if sh == 0 else None)
        ax[0].plot(pp + sh, mm, color=C["model"], lw=1.4, label="SPICE, literature parameters" if sh == 0 else None)
        ax[1].errorbar(ph + sh, 1e3 * res, 1e3 * err, fmt="o", ms=3, color=C["resid"], elinewidth=0.6, lw=0)
    ax[0].invert_yaxis(); ax[0].set_ylabel("$V_T$ [mag]"); ax[0].legend(fontsize=8, loc="lower left")
    wm = np.sum(res / err ** 2) / np.sum(1 / err ** 2)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].axhline(1e3 * wm, color=C["model"], lw=0.8, ls="--")
    ax[1].set_ylabel("O$-$C [mmag]"); ax[1].set_xlabel("pulsation phase"); ax[1].set_xlim(0, 2)
    ax[0].set_title(f"mean O$-$C {1e3 * wm:+.0f} mmag; about it $\\chi^2/N$ = {np.mean(((res - wm) / err) ** 2):.2f}", fontsize=8.5)
    fig.savefig(out / "dcep_photometry.pdf", bbox_inches="tight"); fig.savefig(out / "dcep_photometry.png", bbox_inches="tight"); plt.close(fig)


def fig_spectra(r, out, window=(4480.0, 4520.0), n_show=3):
    key = next(k for k in r["example_spectra"] if abs(float(k[0]) - window[0]) < 1e-6)
    ex = r["example_spectra"][key]; wl = np.asarray(ex["wl"]); phs = np.asarray(ex["phases"])
    pick = np.argsort(phs)[np.linspace(0, len(phs) - 1, n_show).round().astype(int)]
    pm, me, ph, mag, err, _ = vt_model(r)
    o = np.argsort(pm)
    fig = plt.figure(figsize=(11, 4.4)); gs = fig.add_gridspec(2, n_show + 1, height_ratios=[3, 1.1], width_ratios=[1.0] + [2] * n_show, hspace=0.06, wspace=0.42)
    axl = fig.add_subplot(gs[:, 0]); axl.plot(np.r_[pm[o], pm[o] + 1], np.r_[me[o], me[o]], color=C["model"], lw=1.2)
    axl.invert_yaxis(); axl.set_xlim(0, 1); axl.set_xlabel("phase"); axl.set_ylabel("$V_T$ (model) [mag]")
    for j, i in enumerate(pick):
        axl.axvline(phs[i] % 1.0, color=f"C{j}", lw=1, ls=":")
        a = fig.add_subplot(gs[0, j + 1]); b = fig.add_subplot(gs[1, j + 1], sharex=a)
        a.plot(wl, ex["obs"][i], color=C["data"], lw=0.8, label="HARPS-N"); a.plot(wl, ex["model"][i], color=C["model"], lw=0.9, label="SPICE")
        b.plot(wl, ex["obs"][i] - ex["model"][i], color=C["resid"], lw=0.6); b.axhline(0, color="k", lw=0.5)
        a.set_title(f"$\\varphi$ = {phs[i] % 1.0:.2f}", color=f"C{j}", fontsize=9); a.tick_params(labelbottom=False)
        b.set_xlabel("wavelength [Å]"); b.set_ylim(-0.25, 0.25)
        if j == 0: a.set_ylabel("normalised flux"); b.set_ylabel("O$-$C"); a.legend(fontsize=7.5, loc="lower left")
    fig.savefig(out / "dcep_spectra.pdf", bbox_inches="tight"); fig.savefig(out / "dcep_spectra.png", bbox_inches="tight"); plt.close(fig)


def fig_teff(r, out):
    p = np.linspace(0, 1, 400)
    tt = np.loadtxt(LIT / "dcep_trahin2021_teff_curve.txt"); lk = np.loadtxt(LIT / "dcep_luck2018_table6.txt")
    f_t = fourier_fit(tt[:, 0], tt[:, 1], 4); f_l = fourier_fit(lk[:, 0], lk[:, 1], 4)
    to = r["teff_obs"]; ps = np.asarray(r["phases_spec"]) % 1.0
    fig, ax = plt.subplots(2, 1, figsize=(4.6, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
    ax[0].plot(p, f_t(p), color=C["model"], lw=1.4, label="Trahin et al. 2021")
    ax[0].plot(p, f_l(p), color=C["luck"], lw=1.2, ls="--", label="Luck 2018 (Fourier, order 4)")
    ax[0].plot(lk[:, 0] % 1.0, lk[:, 1], "o", ms=3.5, color=C["luck"], mfc="none")
    ax[0].errorbar(np.asarray(to["phase"]) % 1.0, to["teff"], to["err"], fmt="s", ms=3, color=C["andr"], elinewidth=0.6, lw=0, label="Andrievsky et al. 2005")
    ax[0].plot(ps, np.full_like(ps, ax[0].get_ylim()[0] + 15), "|", color="0.5", ms=6)
    ax[0].set_ylabel("$T_{\\rm eff}$ [K]"); ax[0].legend(fontsize=7.5, loc="upper right")
    ax[1].plot(np.linspace(0, 1, len(r["vmacro_curve"])), r["vmacro_curve"], color="0.25", lw=1.2)
    ax[1].plot(ps, r["vmacro_epochs"], ".", ms=2.5, color="0.55")
    ax[1].set_ylabel("broadening FWHM\n[km s$^{-1}$]"); ax[1].set_xlabel("pulsation phase"); ax[1].set_xlim(0, 1)
    fig.savefig(out / "dcep_teff_vmacro.pdf", bbox_inches="tight"); fig.savefig(out / "dcep_teff_vmacro.png", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trahin", required=True); ap.add_argument("--luck", default=None)
    ap.add_argument("--out", default=str(HERE.parent / "paper_plots" / "cepheid"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = pickle.load(open(a.trahin, "rb"))
    fig_photometry(r, out); fig_spectra(r, out); fig_teff(r, out)
    print("wrote", ", ".join(str(out / f) for f in ("dcep_photometry.pdf", "dcep_spectra.pdf", "dcep_teff_vmacro.pdf")))


if __name__ == "__main__":
    main()
