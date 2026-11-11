"""The three delta Cep figures of the paper, from the literature-model run (exact per-element synthesis, nothing fitted; dcep_out/*.pkl):
Trahin et al. (2021) Teff curve, Luck (2018) log g / microturbulence / broadening / abundances with [Fe/H] on the grid's
scale (-0.14 = +0.08 - 0.22), all 152 HARPS-N spectra with the Nardetto et al. (2024) per-epoch systemic velocities.

    d_cephei_profiles.pdf  radius offset (SPIPS, scaled to the model distance), Teff (Trahin, with Luck 2018, the SPIPS
                           model of Merand et al. 2015 and the Andrievsky et al. 2005 line-depth-ratio points for comparison), log g and microturbulence
                           (second-order Fourier series through Luck 2018, as in the model run;
                           the 19 points are not drawn) against pulsation phase
    dcep_photometry.pdf    Johnson V (four sources, per-source offsets constrained to zero mean as in the run, so the
                           level is the model's) and Tycho V_T (no offset) against the exact per-element model, O-C beneath
    dcep_spectra.pdf       V model with three spectrum phases marked (left); HARPS-N against the model in the
                           5440-5480 A window at those phases, residuals beneath each spectrum (right)

    python dcep_literature_figures.py --result dcep_out/dcep_v11V_allFe_trahin_p3.pkl [--out DIR]

Needs the run's inputs locally: delta_cep_model_curves.csv (SPIPS curves), harpsn_windows_log.npz (observed spectra),
literature/ (Trahin and Luck tables).
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
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
PHASE_COLOURS = ("#2a8a7a", "#d68a2e", "#8e4ba8")       # the three spectrum phases
SPEC_WINDOW = (5440.0, 5480.0)                          # HARPS-N window of the spectra figure
SPEC_RANGE = (5457.0, 5477.0)                           # plotted part of it
SPEC_PHASES = (0.08, 0.43, 0.90)                        # target phases; the nearest HARPS-N epochs are shown
D_SPIPS_KPC = 0.274                                     # distance of the SPIPS radius curve (delta_cep.fits)


def fourier_fit(x, y, order):
    """Least-squares Fourier series of the given order (same basis as the model run)."""
    A = lambda p: np.stack([np.ones_like(p)] + [f(2 * np.pi * k * p) for k in range(1, order + 1) for f in (np.cos, np.sin)], 1)
    c = np.linalg.lstsq(A(np.asarray(x, float)), np.asarray(y, float), rcond=None)[0]
    return lambda p: A(np.atleast_1d(np.asarray(p, float))) @ c


SRC_COLOURS = {"Berdnikov+ 2008": "#3b5b92", "Engle+ 2014": "#2a8a7a", "Kiss+ 1998": "#d68a2e", "Moffett+ 1984": "#c0504d"}


def band_data(r, band="VT"):
    """Model curve and observations of one band. Magnitudes from several sources get the run's zero-mean per-source
    offsets (error-weighted mean O-C per source, minus the mean over sources) removed, so the level stays the model's."""
    o = r["obs"][band]; pm = np.asarray(r["phases_model"]) % 1.0; me = np.asarray(r["mags_exact"][band])
    k = np.argsort(pm); pm, me = pm[k], me[k]
    ph, mag, err = (np.asarray(o[x], float) for x in ("phase", "mag", "err"))
    model_at = lambda p: np.interp(np.asarray(p) % 1.0, pm, me, period=1.0)
    src = np.asarray(o["src"]) if "src" in o else np.full(mag.size, band)
    names = sorted(set(src.tolist()))
    if len(names) > 1:
        res = mag - model_at(ph); w = 1.0 / err ** 2
        off = {s: np.sum((res * w)[src == s]) / np.sum(w[src == s]) for s in names}
        mean = np.mean(list(off.values()))
        mag = mag - np.array([off[s] - mean for s in src])
    return pm, me, ph % 1.0, mag, err, model_at, src


def vt_data(r):
    return band_data(r, "VT")[:6]


def closed(pm, me):
    return np.r_[pm, pm[0] + 1.0], np.r_[me, me[0]]


def fig_profiles(r, out):
    a = r["args"]; scale = float(a["distance_kpc"]) / D_SPIPS_KPC
    import pandas as pd
    m = pd.read_csv(HERE / "delta_cep_model_curves.csv"); m = m.sort_values("PHASE")
    p = np.linspace(0, 1, 401)
    R = np.interp(p, m["PHASE"], m["R"] * scale, period=1.0)
    tt = np.loadtxt(LIT / "dcep_trahin2021_teff_curve.txt"); lk = np.loadtxt(LIT / "dcep_luck2018_table6.txt")
    teff = fourier_fit(tt[:, 0], tt[:, 1], int(a["fourier_order"]))(p)
    logg = fourier_fit(lk[:, 0], lk[:, 2], 2)(p)
    vmic = fourier_fit(lk[:, 0], lk[:, 3], 2)(p)
    # line broadening actually used by the model: Luck's Gaussian width (1/e half-width, x1.665 -> FWHM) with his
    # R = 60 000 instrumental FWHM (5 km/s) removed in quadrature, as a Fourier series through his 19 points
    vb_curve = np.asarray(r["vmacro_curve"], float); vb_ph = np.linspace(0, 1, vb_curve.size)
    vb_pts = np.c_[lk[:, 0], np.sqrt((lk[:, 4] * float(a["vmacro_curve_scale"])) ** 2
                                     - (299792.458 / float(a["vmacro_instr_r"])) ** 2)]
    fig = plt.figure(figsize=(9.5, 5.4)); gs = fig.add_gridspec(2, 6)   # printed at text width: text >= 6.5 pt
    a00 = fig.add_subplot(gs[0, :3]); a01 = fig.add_subplot(gs[0, 3:], sharex=a00)
    b0 = fig.add_subplot(gs[1, :2], sharex=a00); b1 = fig.add_subplot(gs[1, 2:4], sharex=a00); b2 = fig.add_subplot(gs[1, 4:], sharex=a00)
    ax = np.array([[a00, a01], [b0, b1]])
    panels = [(a00, R - R.mean(), "Radius offset", r"Radius offset [$R_\odot$]", None),
              (a01, teff, "Effective Temperature", r"$T_{\rm eff}$ [K]", None),
              (b0, logg, "Surface Gravity", r"$\log g$", None),
              (b1, vmic, "Microturbulent Velocity", r"$v_{\rm mic}$ [km/s]", None),
              (b2, np.interp(p, vb_ph, vb_curve), "Line Broadening", r"Gaussian FWHM [km/s]", vb_pts)]
    for a_, y, title, ylab, pts in panels:
        a_.plot(p, y, color="k", lw=1.5)
        if pts is not None:
            a_.plot(pts[:, 0] % 1.0, pts[:, 1], "o", ms=3.5, mfc="none", mec="0.45", mew=0.8)
        a_.set_title(title, fontsize=11); a_.set_ylabel(ylab, fontsize=10); a_.tick_params(labelsize=9)
        a_.spines["top"].set_visible(True); a_.spines["right"].set_visible(True)
    # all the temperature curves on the Teff panel: the adopted one (black, drawn above), Luck's Fourier series,
    # the SPIPS model of Merand et al. (2015) and the Andrievsky et al. (2005) points
    at = ax[0, 1]; at.lines[0].set_label("Trahin et al. 2021 (adopted)")
    f_l = fourier_fit(lk[:, 0], lk[:, 1], 4)
    at.plot(p, f_l(p), color="#2a78d6", lw=1.3, ls="--", label="Luck 2018, Fourier series")
    sp_ph = np.asarray(r["spips_phase"]) % 1.0; sp_t = np.asarray(r["spips_teff"]); o = np.argsort(sp_ph)
    at.plot(sp_ph[o], sp_t[o], color="#d68a2e", lw=1.2, ls=":", label="SPIPS model (Mérand et al. 2015)")
    # independent line-depth-ratio temperatures of Andrievsky et al. (2005), as points (referee R2: show them)
    an = r["teff_obs"]; a_ph, a_t, a_e = (np.asarray(an[k], float) for k in ("phase", "teff", "err"))
    at.errorbar(a_ph % 1.0, a_t, a_e, fmt="s", ms=3.5, color="#c0504d", ecolor="#c0504d", elinewidth=0.7, lw=0,
                label="Andrievsky et al. 2005")
    at.legend(fontsize=8, loc="upper center", frameon=False, ncol=1, handlelength=1.6, borderaxespad=0.3); at.set_ylim(5380, 7250)
    for a_ in (b0, b1, b2):
        a_.set_xlabel("Phase", fontsize=10)
    for a_ in (a00, a01):
        a_.tick_params(labelbottom=False)
    fig.tight_layout()
    save(fig, out, "d_cephei_profiles")


def fig_photometry(r, out):
    bands = [b for b in ("V", "VT") if b in r["obs"]]
    fig = plt.figure(figsize=(5.4, 4.6 * len(bands)))
    gs = fig.add_gridspec(2 * len(bands), 1, height_ratios=[3, 1] * len(bands), hspace=0.08)
    stats = {}
    for j, band in enumerate(bands):
        pm, me, ph, mag, err, model_at, src = band_data(r, band)
        res = mag - model_at(ph); chi2 = np.mean((res / err) ** 2); stats[band] = (chi2, res.size)
        a0 = fig.add_subplot(gs[2 * j]); a1 = fig.add_subplot(gs[2 * j + 1], sharex=a0)
        x, y = closed(pm, me)
        lab = "V" if band == "V" else "V_T"
        a0.plot(x, y, color="k", lw=1.6, label=f"literature model ($\\chi^2/N$ {chi2:.2f})", zorder=3)
        for s in sorted(set(src.tolist())):
            k = src == s; c = SRC_COLOURS.get(s, "0.35")
            name = s if band == "V" else r"Tycho $V_T$ (ESA 1997)"
            a0.errorbar(ph[k], mag[k], err[k], fmt="o", ms=3.0, color=c, ecolor=c, elinewidth=0.6, alpha=0.8, label=name)
            a1.errorbar(ph[k], 1e3 * res[k], 1e3 * err[k], fmt="o", ms=3.0, color=c, ecolor=c, elinewidth=0.6, alpha=0.8)
        a0.invert_yaxis(); a0.set_ylabel(f"${lab}$ [mag]"); a0.legend(fontsize=7.5, loc="lower left", frameon=False)
        a0.tick_params(labelbottom=False)
        a1.axhline(0, color="0.75", lw=0.8); a1.set_ylabel("O - C [mmag]"); a1.set_xlim(-0.03, 1.03)
        lim = min(1.15 * np.max(np.abs(1e3 * res) + 1e3 * err), 200.0); a1.set_ylim(-lim, lim)
        if j == len(bands) - 1:
            a1.set_xlabel("pulsation phase")
        else:
            a1.tick_params(labelbottom=False)
    save(fig, out, "dcep_photometry")
    return stats


def fig_spectra(r, out, harpsn):
    H = np.load(harpsn, allow_pickle=True)
    wins = [tuple(w) for w in H["windows"]]; wi = next(i for i, w in enumerate(wins) if abs(w[0] - SPEC_WINDOW[0]) < 1)
    wl = 10.0 ** np.asarray(H[f"logwl_{wi}"]); obs = np.asarray(H[f"obs_{wi}"], float)
    key = next(k for k in r["exact_all"]["spectra"] if abs(float(k[0]) - SPEC_WINDOW[0]) < 1)
    mod = np.asarray(r["exact_all"]["spectra"][key]["model"], float)
    assert mod.shape == obs.shape, (mod.shape, obs.shape)
    ps = np.asarray(r["phases_spec"]) % 1.0
    dist = lambda a_, b_: np.abs((a_ - b_ + 0.5) % 1.0 - 0.5)
    pick = [int(np.argmin(dist(ps, t))) for t in SPEC_PHASES]
    band = "V" if "V" in r["obs"] else "VT"
    lab = "V" if band == "V" else "V_T"
    pm, me, ph, mag, err, model_at, _ = band_data(r, band)
    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.9], wspace=0.14, hspace=0.12)
    axl = fig.add_subplot(gs[:, 0])
    x, y = closed(pm, me)
    axl.errorbar(ph, mag, err, fmt="o", ms=2.5, color="0.6", ecolor="0.75", elinewidth=0.6, alpha=0.8, label=f"observed ${lab}$", zorder=1)
    axl.plot(x, y, color="k", lw=1.6, label=f"literature model ${lab}$", zorder=2)
    for j, i in enumerate(pick):
        axl.plot(ps[i], model_at(ps[i]), "o", ms=10, color=PHASE_COLOURS[j], mec="k", mew=0.6, zorder=4,
                 label=f"spectrum at phase {ps[i]:.2f}")
    axl.invert_yaxis(); axl.set_xlabel("pulsation phase"); axl.set_ylabel(f"${lab}$ [mag]")
    h, l = axl.get_legend_handles_labels(); order = [0, 1, 2, 3, 4]
    axl.legend([h[k] for k in order], [l[k] for k in order], fontsize=7.5, loc="lower left", frameon=False)
    sel = (wl >= SPEC_RANGE[0]) & (wl <= SPEC_RANGE[1])
    for j, i in enumerate(pick):
        a_ = fig.add_subplot(gs[j, 1]); c = PHASE_COLOURS[j]
        o_, m_ = obs[i], mod[i]; g = np.isfinite(o_) & sel
        rms = np.sqrt(np.mean((o_ - m_)[g] ** 2))
        a_.plot(wl[sel], o_[sel], color="k", lw=0.9, label="HARPS-N")
        a_.plot(wl[sel], m_[sel], color=c, lw=1.1, label=f"literature model, phase {ps[i]:.2f}")
        a_.plot(wl[sel], (o_ - m_)[sel], color=c, lw=0.8)
        a_.axhline(0, color="0.8", lw=0.6)
        # headroom above the continuum so the legend sits clear of the spectra
        a_.set_ylim(-0.25, 1.42); a_.set_xlim(*SPEC_RANGE); a_.set_ylabel("normalised flux"); a_.set_yticks([0, 0.5, 1.0])
        a_.legend(fontsize=7, loc="upper left", ncol=2, frameon=False, borderaxespad=0.2)
        a_.text(0.995, 0.03, f"residual, rms {100 * rms:.1f}%", transform=a_.transAxes, ha="right", va="bottom", fontsize=7.5)
        a_.spines["top"].set_visible(True); a_.spines["right"].set_visible(True)
        if j < 2:
            a_.tick_params(labelbottom=False)
        else:
            a_.set_xlabel("wavelength [Å]")
    save(fig, out, "dcep_spectra")
    return [(float(ps[i]), int(i)) for i in pick]


def fig_teff_curves(r, out):
    """Every delta Cep temperature curve in play: Trahin et al. (2021), the Fourier series through Luck (2018) with his
    19 points, the SPIPS model of Merand et al. (2015) carried by delta_cep.fits, and the line-depth-ratio temperatures
    of Andrievsky et al. (2005); residuals against the adopted (Trahin) curve beneath."""
    p = np.linspace(0, 1, 401)
    tt = np.loadtxt(LIT / "dcep_trahin2021_teff_curve.txt"); lk = np.loadtxt(LIT / "dcep_luck2018_table6.txt")
    f_t = fourier_fit(tt[:, 0], tt[:, 1], int(r["args"]["fourier_order"])); f_l = fourier_fit(lk[:, 0], lk[:, 1], 4)
    sp_ph = np.asarray(r["spips_phase"]) % 1.0; sp_t = np.asarray(r["spips_teff"]); o = np.argsort(sp_ph)
    sp_ph, sp_t = sp_ph[o], sp_t[o]
    an = r["teff_obs"]; a_ph, a_t, a_e = (np.asarray(an[k], float) for k in ("phase", "teff", "err")); a_ph = a_ph % 1.0
    fig, ax = plt.subplots(2, 1, figsize=(5.4, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[2.4, 1], hspace=0.07))
    ax[0].plot(p, f_t(p), color="k", lw=1.6, label="Trahin et al. 2021 (adopted)")
    ax[0].plot(p, f_l(p), color="#2a78d6", lw=1.3, ls="--", label="Luck 2018, Fourier series")
    ax[0].plot(lk[:, 0] % 1.0, lk[:, 1], "o", ms=4, mfc="none", mec="#2a78d6", mew=0.9, label="Luck 2018")
    ax[0].plot(sp_ph, sp_t, color="#d68a2e", lw=1.2, ls=":", label="SPIPS model (Mérand et al. 2015)")
    ax[0].errorbar(a_ph, a_t, a_e, fmt="s", ms=3.5, color="#c0504d", ecolor="#c0504d", elinewidth=0.7, lw=0,
                   label="Andrievsky et al. 2005")
    ax[0].set_ylabel(r"$T_{\rm eff}$ [K]"); ax[0].set_ylim(5380, 7000)
    ax[0].legend(fontsize=7.5, loc="upper center", frameon=False, ncol=2)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].plot(p, f_l(p) - f_t(p), color="#2a78d6", lw=1.1, ls="--")
    ax[1].plot(lk[:, 0] % 1.0, lk[:, 1] - f_t(lk[:, 0] % 1.0), "o", ms=4, mfc="none", mec="#2a78d6", mew=0.9)
    ax[1].plot(sp_ph, sp_t - f_t(sp_ph), color="#d68a2e", lw=1.1, ls=":")
    ax[1].errorbar(a_ph, a_t - f_t(a_ph), a_e, fmt="s", ms=3.5, color="#c0504d", ecolor="#c0504d", elinewidth=0.7, lw=0)
    ax[1].set_ylabel("$-$ Trahin [K]"); ax[1].set_xlabel("pulsation phase"); ax[1].set_xlim(0, 1)
    for a_ in ax:
        a_.spines["top"].set_visible(True); a_.spines["right"].set_visible(True)
    d_a = a_t - f_t(a_ph); d_l = a_t - f_l(a_ph)
    print(f"Andrievsky - Trahin: mean {d_a.mean():+.0f} K, rms {d_a.std():.0f} K; Andrievsky - Luck curve: mean {d_l.mean():+.0f} K, rms {d_l.std():.0f} K; "
          f"SPIPS (Merand) - Trahin: mean {(sp_t - f_t(sp_ph)).mean():+.0f} K, rms {(sp_t - f_t(sp_ph)).std():.0f} K; {len(a_t)} Andrievsky points")
    save(fig, out, "dcep_teff_curves")


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote", out / f"{stem}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--result", default=str(HERE / "dcep_out" / "dcep_v12V13_own_trahin_p3.pkl"))
    ap.add_argument("--harpsn", default=str(HERE / "harpsn_windows_log.npz"))
    ap.add_argument("--out", default=str(HERE.parent / "paper_plots" / "cepheid"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = pickle.load(open(a.result, "rb"))
    print(f"model: Teff curve {Path(r['args']['teff_curve']).name}, [Fe/H] {r['args']['feh']:+.2f}, {len(r['phases_spec'])} spectra")
    fig_profiles(r, out)
    for b, (chi2, n) in fig_photometry(r, out).items():
        print(f"{b}: chi2/N {chi2:.2f} on {n} points (per-source offsets removed, zero mean)")
    print("spectra figure epochs (phase, index):", fig_spectra(r, out, a.harpsn))


if __name__ == "__main__":
    main()
