"""Three paper figures summarising the TZ For free-geometry result.

* ``tzfor_teff_treatments``: Teff1 vs Teff2 for every joint fit in the chain (12-window and
  full-range, plain / masked / line-strength-corrected, fixed and free geometry), with the
  block-jackknife errors, the sampler posteriors and Andersen's literature values.
* ``tzfor_eclipse_zoom``: the two eclipses in Stromgren b and y, Clausen's photometry against
  the fixed-geometry and free-geometry models, with residual panels.
* ``tzfor_geometry_degeneracy``: R2 against Teff2, the jackknife blocks that measure the error,
  and the line of constant secondary luminosity that explains why the two trade off.

Reads only the result pickles and model dumps in ``tzfor_aemu_out/``. Pure plotting, no GPU.
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tzfor_constants as K

OUT_DIR = HERE.parent / "paper_plots" / "tz_fornacis"
AEMU = HERE / "tzfor_aemu_out"


def load(name):
    p = AEMU / name
    return pickle.load(open(p, "rb")) if p.exists() else None


# tzfor_kernel_fit.py stores no pnames: its theta starts (Teff1, Teff2, [Fe/H], ...)
_FALLBACK = ["Teff1", "Teff2", "feh"]


def _names(d):
    return list(d["pnames"]) if "pnames" in d else _FALLBACK


def val(d, key, default=np.nan):
    nm = _names(d)
    return float(np.asarray(d["theta"])[nm.index(key)]) if key in nm else default


def err(d, key):
    nm = _names(d)
    if d.get("jackknife_err") is None or key not in nm: return np.nan
    return float(np.asarray(d["jackknife_err"])[nm.index(key)])


def save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig); print(f"  {OUT_DIR}/{stem}.png")


# ---------------------------------------------------------------- figure 1
def fig_treatments():
    rows = [  # file, label, marker, colour
        ("kernel_fit_result_mask5.pkl", "12 windows, 5% mask", "s", "C0"),
        ("fullrange_pass1_hp.pkl", "full range, no mask", "o", "C7"),
        ("fullrange_maskall_hp.pkl", "full range, 5% mask", "o", "C0"),
        ("fullrange_even_deltaodd.pkl", "full range, δ map (held-out epochs)", "o", "C2"),
        ("fullrange_maskall_geo.pkl", "free geometry, 5% mask", "D", "C3"),
        ("fullrange_deltaodd_geo.pkl", "free geometry, δ map", "D", "C1"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    jk = load("fullrange_maskall_geoJ.pkl")
    for f, label, m, c in rows:
        d = load(f)
        if d is None: continue
        T1, T2 = val(d, "Teff1"), val(d, "Teff2")
        e1, e2 = err(d, "Teff1"), err(d, "Teff2")
        if f == "fullrange_maskall_geo.pkl" and jk is not None:      # errors come from the separate jackknife run
            e1, e2 = err(jk, "Teff1"), err(jk, "Teff2")
        free = "free geometry" in label
        ax.errorbar(T1, T2, xerr=None if np.isnan(e1) else e1, yerr=None if np.isnan(e2) else e2,
                    fmt=m, color=c, ms=9 if free else 7, mfc=c if free else "none", mew=1.6, capsize=3, lw=1.4, label=label, zorder=3)
    for f, nm, c in (("nuts_full_pass1.npz", "posterior, no mask", "C7"), ("nuts_full_maskall.npz", "posterior, 5% mask", "C0"),
                     ("nuts_full_deltaodd.npz", "posterior, δ map", "C2")):
        p = AEMU / f
        if not p.exists(): continue
        s = np.load(p); sq = lambda a: a[:, 0] if a.ndim >= 3 and a.shape[1] == 1 else a
        ax.plot(sq(s["Teff1"]).mean(), sq(s["Teff2"]).mean(), "+", color=c, ms=11, mew=2, zorder=4)
    ax.errorbar(K.PRIMARY_TEFF, K.SECONDARY_TEFF, xerr=100, yerr=100, fmt="*", ms=17, color="k", capsize=3, label="Andersen et al. (1991)", zorder=5)
    ax.set_xlabel("$T_{\\rm eff,1}$ [K]  (K giant)"); ax.set_ylabel("$T_{\\rm eff,2}$ [K]  (secondary)")
    ax.set_title("TZ For: every joint solution, with block-jackknife errors")
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.text(0.98, 0.03, "crosses: NUTS posterior means (widths ≈ 1 K)", transform=ax.transAxes, ha="right", fontsize=8, color="0.35")
    save(fig, "tzfor_teff_treatments")


# ---------------------------------------------------------------- figure 2
def fig_eclipses():
    """Both eclipses, both bands, data against the fixed- and free-geometry models.

    Each dump stores its model magnitudes relative to the median over *its own* model phases,
    and the two runs use different phase sets (56 PHOEBE-mesh times vs 51 eclipse phases), so
    every curve is put back on its own out-of-eclipse level before plotting.
    """
    geo = np.load(AEMU / "fr_model_maskall_geo.npz", allow_pickle=True)
    fix = np.load(AEMU / "fr_model_maskall_hp.npz", allow_pickle=True)
    dphi = lambda d: float(d["theta"][list(d["pnames"]).index("dphi")])
    def curves(d, b):
        lev = float(np.min(d[f"lc_{b}_model_mag"]))                       # brightest model point = out of eclipse
        return ((d[f"lc_{b}_model_phase"] + dphi(d)) % 1.0, d[f"lc_{b}_model_mag"] - lev,
                (d[f"lc_{b}_obs_phase"] + dphi(d)) % 1.0, d[f"lc_{b}_obs_mag"] - lev,
                1000.0 * (d[f"lc_{b}_obs_mag"] - d[f"lc_{b}_model_at_obs"]))
    near = lambda ph, c, w=0.028: np.abs(((ph - c + 0.5) % 1.0) - 0.5) < w
    eclipses = [("deep eclipse (secondary occulted)", 0.1585), ("shallow eclipse (secondary transits)", 0.6585)]
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 11), gridspec_kw=dict(height_ratios=[3, 1, 3, 1]))
    for r, b in enumerate(("b", "y")):
        for j, (title, centre) in enumerate(eclipses):
            top, bot = axes[2 * r, j], axes[2 * r + 1, j]
            mp_f, mm_f, op, om, res_f = curves(fix, b); mp_g, mm_g, _, _, res_g = curves(geo, b)
            sel = near(op, centre)
            top.plot(op[sel], om[sel], ".", color="0.35", ms=4, zorder=1, label=f"Clausen {b}")
            for mp, mm, c, name in ((mp_f, mm_f, "C0", "fixed (PHOEBE) geometry"), (mp_g, mm_g, "C3", "free geometry")):
                s2 = near(mp, centre); o = np.argsort(mp[s2])
                top.plot(mp[s2][o], mm[s2][o], "-", color=c, lw=1.6, zorder=2, label=name)
            top.invert_yaxis(); top.grid(alpha=0.3); top.set_ylabel(f"Δ{b} [mag]")
            top.set_title(f"{title} — Strömgren {b}", fontsize=10)
            if r == 0 and j == 0: top.legend(fontsize=8.5, loc="lower left")
            for res, c, name in ((res_f, "C0", "fixed"), (res_g, "C3", "free")):
                bot.plot(op[sel], res[sel], ".", color=c, ms=4, alpha=0.75,
                         label=f"{name}: rms {np.sqrt(np.mean(res[sel] ** 2)):.1f} mmag")
            bot.axhline(0, color="0.5", lw=0.8); bot.grid(alpha=0.3); bot.set_ylabel("O−C [mmag]")
            bot.legend(fontsize=7.5, ncol=2, loc="upper center", framealpha=0.9)
            bot.set_ylim(-22, 22)
            if r == 1: bot.set_xlabel("phase from $T_P$")
    R1, R2 = (float(geo["theta"][list(geo["pnames"]).index(k)]) for k in ("R1", "R2"))
    fig.suptitle(f"TZ For eclipses: fixed geometry (R₁ {K.PRIMARY_RADIUS}, R₂ {K.SECONDARY_RADIUS} R☉) "
                 f"vs free geometry (R₁ {R1:.2f}, R₂ {R2:.2f} R☉) — both fit the photometry", y=0.995)
    fig.tight_layout(); save(fig, "tzfor_eclipse_zoom")


# ---------------------------------------------------------------- figure 3
def fig_degeneracy():
    jk = load("fullrange_maskall_geoJ.pkl"); geo = load("fullrange_maskall_geo.pkl"); dlt = load("fullrange_deltaodd_geo.pkl")
    if jk is None or jk.get("jackknife") is None: print("  (no geometry jackknife yet)"); return
    A = np.asarray(jk["jackknife"]); pn = list(jk["pnames"]); iR2, iT2, iR1 = pn.index("R2"), pn.index("Teff2"), pn.index("R1")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    ax.plot(A[:, iR2], A[:, iT2], "o", color="C3", mfc="none", ms=7, label="leave-one-block-out refits")
    ax.errorbar(val(geo, "R2"), val(geo, "Teff2"), xerr=err(jk, "R2"), yerr=err(jk, "Teff2"), fmt="D", color="C3", ms=10, capsize=3, lw=1.5, label="free-geometry fit", zorder=4)
    if dlt is not None:
        ax.plot(val(dlt, "R2"), val(dlt, "Teff2"), "D", color="C1", ms=9, label="free geometry, δ map", zorder=4)
    ax.plot(K.SECONDARY_RADIUS, K.SECONDARY_TEFF, "*", color="k", ms=17, label="Andersen et al. (1991)", zorder=5)
    R2f, T2f = val(geo, "R2"), val(geo, "Teff2")
    r = np.linspace(3.85, 4.20, 80); ax.plot(r, T2f * np.sqrt(R2f / r), color="0.6", lw=1.2, ls="--", zorder=1, label="constant secondary luminosity")
    ax.set_xlabel("$R_2$ [$R_\\odot$]"); ax.set_ylabel("$T_{\\rm eff,2}$ [K]"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
    ax.set_title("The secondary's light, split between size and temperature")
    ax = axes[1]
    ax.plot(A[:, iR1], A[:, iR2], "o", color="C3", mfc="none", ms=7)
    ax.errorbar(val(geo, "R1"), val(geo, "R2"), xerr=err(jk, "R1"), yerr=err(jk, "R2"), fmt="D", color="C3", ms=10, capsize=3, lw=1.5, zorder=4, label="free-geometry fit")
    if dlt is not None: ax.plot(val(dlt, "R1"), val(dlt, "R2"), "D", color="C1", ms=9, zorder=4, label="δ map")
    ax.plot(K.PRIMARY_RADIUS, K.SECONDARY_RADIUS, "*", color="k", ms=17, zorder=5, label="Andersen et al. (1991)")
    x = np.linspace(7.9, 8.45, 50)
    ax.plot(x, (K.PRIMARY_RADIUS + K.SECONDARY_RADIUS) - x, color="0.6", ls="--", lw=1.2, zorder=1, label="constant $R_1+R_2$ (eclipse duration)")
    ax.set_xlabel("$R_1$ [$R_\\odot$]"); ax.set_ylabel("$R_2$ [$R_\\odot$]"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
    ax.set_title("Radii: the sum is fixed, the ratio is not")
    fig.tight_layout(); save(fig, "tzfor_geometry_degeneracy")


# ---------------------------------------------------------------- figure 4
def fig_ratio_constraint():
    """What the photometry alone says about R2/R1, against what the spectra want.

    PHOEBE's chi2 (Roche meshes, ck2004) scanned over the radius ratio at fixed R1+R2 -- the direction the
    eclipse duration leaves free -- with Teff2 re-optimised at every ratio, since a smaller secondary is
    compensated by a hotter one at fixed eclipse depth.
    """
    f = AEMU / "phoebe_geometry_scan.pkl"
    if not f.exists(): print("  (no ratio scan yet)"); return
    d = pickle.load(open(f, "rb"))
    if "ratio_scan" not in d: print("  (ratio scan incomplete)"); return
    A = np.asarray(d["ratio_scan"], float); N = d["N"]
    q, T2, c = A[:, 0], A[:, 3], A[:, 4]
    if q.size < 4: print(f"  (ratio scan has only {q.size} points)"); return
    c0 = c.min()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    ax = axes[0]
    ax.plot(q, c - c0, "o-", color="C0", ms=5, lw=1.4, label="PHOEBE photometry (Roche, ck2004)")
    q_lit = K.SECONDARY_RADIUS / K.PRIMARY_RADIUS; q_spice, q_delta = 4.084 / 8.118, 4.045 / 8.087
    for x, col, lab in ((q_lit, "k", "Andersen et al. (1991)"), (q_spice, "C3", "SPICE free geometry, 5% mask"), (q_delta, "C1", "SPICE free geometry, δ map")):
        ax.axvline(x, color=col, ls="--", lw=1.3, label=lab)
    for lev, txt in ((1.0, "1σ"), (9.0, "3σ")):
        ax.axhline(lev, color="0.7", lw=0.8, ls=":"); ax.text(q.min(), lev, f" Δχ²={lev:.0f} ({txt})", va="bottom", fontsize=7.5, color="0.45")
    ax.set_yscale("symlog", linthresh=1.0); ax.set_ylim(bottom=0)
    ax.text(0.5, 0.02, "wiggle below 0.465 is Nelder-Mead noise (~4 in χ²)", transform=ax.transAxes, ha="center", fontsize=7.5, color="0.45")
    ax.set_xlabel("$R_2/R_1$   (at fixed $R_1+R_2$ = 12.22 $R_\odot$)"); ax.set_ylabel("Δχ² (photometry)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper center")
    ax.set_title("The photometry's constraint on the radius ratio (inclination fixed)")
    ax = axes[1]
    ax.plot(q, T2, "o-", color="C2", ms=5, lw=1.4, label="PHOEBE's best $T_{\\rm eff,2}$ at each ratio")
    for x, col in ((q_lit, "k"), (q_spice, "C3"), (q_delta, "C1")): ax.axvline(x, color=col, ls="--", lw=1.3)
    ax.plot(q_spice, 6275, "D", color="C3", ms=9, label="SPICE joint fit (spectra + photometry)")
    ax.plot(q_delta, 6211, "D", color="C1", ms=9)
    ax.set_xlabel("$R_2/R_1$"); ax.set_ylabel("$T_{\\rm eff,2}$ [K]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Size and temperature trade off at fixed eclipse depth")
    fig.tight_layout(); save(fig, "tzfor_ratio_constraint")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=["treatments", "eclipses", "degeneracy", "ratio"], default=None)
    a = ap.parse_args()
    which = a.only or ["treatments", "eclipses", "degeneracy", "ratio"]
    print("figures:")
    if "treatments" in which: fig_treatments()
    if "eclipses" in which: fig_eclipses()
    if "degeneracy" in which: fig_degeneracy()
    if "ratio" in which: fig_ratio_constraint()
