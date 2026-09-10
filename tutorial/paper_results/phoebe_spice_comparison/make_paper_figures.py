"""Paper figures for the SPICE-vs-PHOEBE atmosphere comparison.

Consumes the pickles written by ``spice_vs_phoebe_atmospheres.py``
(``atmosphere_out/intensity_grid.pkl``, ``atmosphere_out/binary_light_curves.pkl``)
and writes one PDF per figure into ``atmosphere_out/figures/``.

    python make_paper_figures.py [--out DIR] [--results DIR]

Encoding, used identically in every panel so the reader learns it once:

* **hue = atmosphere** -- MARCS/aemu, ck2004, phoenix, blackbody. Colour follows
  the physics, not the code, so the same model is the same colour everywhere.
* **linestyle = code** -- SPICE solid, PHOEBE dashed. Blackbody is the only model
  both codes have, and it is the only place both linestyles appear for one hue;
  identity is therefore never carried by colour alone.

The palette is the Okabe-Ito-derived set
``#0072B2 / #D55E00 / #009E73 / #8C6D1F``, which passes all six checks of the
dataviz validator against a light surface (lightness band, chroma floor, CVD
separation, normal-vision floor, contrast >= 3:1) with no warnings. The
diverging map used for the response heatmaps is built from the same blue and
vermillion with a neutral -- not hued -- midpoint, so zero reads as "no
difference" rather than as a colour.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# --- encoding -------------------------------------------------------------
ATM_COLOR = {
    "aemu": "#0072B2",       # SPICE's MARCS intensity bundle
    "spice_blackbody": "#0072B2",
    "ck2004": "#D55E00",
    "phoenix": "#009E73",
    "blackbody": "#8C6D1F",
}
ATM_LABEL = {
    "aemu": "SPICE (MARCS emulator)",
    "spice_blackbody": "SPICE (Blackbody)",
    "ck2004": "PHOEBE ck2004",
    "phoenix": "PHOEBE phoenix",
    "blackbody": "blackbody",
}
CODE_STYLE = {"spice": "-", "phoebe": "--"}

# SPICE and ck2004 agree so closely that a thin blue line disappears completely
# under the vermillion one -- which is the result, but an invisible curve is not
# a readable figure. SPICE is therefore drawn as a thick, slightly transparent
# band *underneath*, and each PHOEBE atmosphere gets its own dash pattern laid
# on top: where they agree the reader sees dashes sitting inside the blue band,
# which reads correctly as agreement rather than as a missing series.
ATM_DASH = {"ck2004": (5.0, 1.6), "phoenix": (1.8, 1.6), "blackbody": (7.0, 2.2)}
SPICE_LW = 2.6
SPICE_ALPHA = 0.55


def phoebe_kw(atm, **kw):
    kw.setdefault("color", ATM_COLOR[atm])
    kw.setdefault("lw", 1.2)
    kw["dashes"] = ATM_DASH[atm]
    return kw


def spice_kw(**kw):
    kw.setdefault("color", ATM_COLOR[SPICE_ID])
    kw.setdefault("lw", SPICE_LW)
    kw.setdefault("alpha", SPICE_ALPHA)
    kw.setdefault("solid_capstyle", "round")
    kw.setdefault("zorder", 2)
    return kw

BANDS = ["Johnson:B", "Johnson:V", "Stromgren:b", "Stromgren:y"]
BAND_LABEL = {"Johnson:B": "Johnson $B$", "Johnson:V": "Johnson $V$",
              "Stromgren:b": "Strömgren $b$", "Stromgren:y": "Strömgren $y$"}
# Populated per suite from the results payload; the module-level value is only
# the fallback for an old pickle that predates the suite split.
PHOEBE_ATMS = ["ck2004", "phoenix", "blackbody"]
SPICE_ID = "aemu"
SPICE_KEY = "aemu"


def set_suite(atms, spice_kind="aemu"):
    """Point every figure at one suite's atmosphere list.

    The two suites are separate comparisons -- 'atmospheres' (aemu vs
    ck2004/phoenix) and 'blackbody' (Blackbody vs blackbody) -- and a figure
    must never mix them, so the list is set once per suite rather than
    hardcoded.
    """
    global PHOEBE_ATMS, SPICE_ID, SPICE_KEY
    PHOEBE_ATMS = list(atms)
    # SPICE_KEY indexes the binary results dicts; SPICE_ID selects the label and
    # colour. They differ because the results key for the control suite is
    # "blackbody", which would otherwise collide with PHOEBE's blackbody entry.
    SPICE_KEY = spice_kind
    SPICE_ID = "aemu" if spice_kind == "aemu" else "spice_blackbody"


def spice_key_of(binary):
    """Which SPICE emulator this suite ran, read off the results."""
    for r in binary.values():
        if "spice_kind" in r:
            return r["spice_kind"]
        if r.get("spice_dmag"):
            return next(iter(r["spice_dmag"]))[0]
    return "aemu"

# Neutral-midpoint diverging map from the palette's two poles. A hue at the
# midpoint would make "no difference" look like a value.
DIVERGING = LinearSegmentedColormap.from_list(
    "spice_div", ["#0072B2", "#8FC7E3", "#F2F2F0", "#F0A882", "#D55E00"])



# Formats every figure is written in. The batch script wants PDF for the paper;
# the notebook adds "png" so it can display results inline without needing a PDF
# renderer (pymupdf is not in every kernel).
SAVE_FORMATS = ("pdf",)


def _save(fig, out, stem):
    """Write ``fig`` as ``stem`` in each of SAVE_FORMATS; return the pdf name."""
    for ext in SAVE_FORMATS:
        fig.savefig(Path(out) / f"{stem}.{ext}")
    return f"{stem}.pdf"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
        # Frame style matched to the existing paper figures
        # (single_eclipse_system_residuals.pdf, blackbody_comparison.pdf):
        # fully boxed axes, no grid.
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": False,
    })


# --- data access ----------------------------------------------------------

def index_records(grid):
    """``{(band, teff, logg, feh): record}``."""
    return {(r["band"], r["teff"], r["logg"], r["feh"]): r for r in grid["records"]}


def response(records, band, atm, teff, logg, feh, reference):
    """``log10 I(P)/I(P_ref)`` for SPICE minus the same for one PHOEBE atmosphere.

    Differencing against a shared reference removes the absolute normalisation,
    which the two codes do not share (PHOEBE's ``Inorm`` is a passband-integrated
    W/m^3, SPICE's is erg/s/cm^2/A over the same curve). What survives is the
    *shape* of the parameter dependence, which is the comparable quantity.
    """
    key = f"phoebe_{atm}_Inorm"
    r = records.get((band, teff, logg, feh))
    ref = records.get((band, *reference))
    if r is None or ref is None or key not in r or key not in ref:
        return np.nan
    vals = (r["spice_Inorm"], ref["spice_Inorm"], r[key], ref[key])
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return np.nan
    return float(np.log10(r["spice_Inorm"] / ref["spice_Inorm"])
                 - np.log10(r[key] / ref[key]))


def legend_handles(atms, codes=("spice", "phoebe")):
    """Two-part legend: hue = atmosphere, linestyle = code."""
    handles = []
    for a in atms:
        if a == SPICE_ID:
            handles.append(Line2D([], [], label=ATM_LABEL[a], **spice_kw()))
        else:
            handles.append(Line2D([], [], label=ATM_LABEL[a], **phoebe_kw(a)))
    return handles


# --- Figure 1: limb-darkening profiles ------------------------------------

def fig_limb_darkening(grid, out):
    records = index_records(grid)
    mus = np.asarray(grid["mus"])
    teffs = [4500.0, 5500.0, 6500.0]
    logg, feh, band = 4.5, 0.0, "Johnson:V"
    # The residual row deliberately omits blackbody: it is not limb darkened at
    # all, so its residual is just 1 - L(mu) (up to -0.75 here) and plotting it
    # compresses the ck2004/phoenix curves -- the ones actually being compared --
    # into the zero line. Its flat profile is already visible in the top row.
    resid_atms = ["ck2004", "phoenix"]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2), sharex=True,
                             sharey="row", constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for j, teff in enumerate(teffs):
        r = records.get((band, teff, logg, feh))
        top, bot = axes[0, j], axes[1, j]
        if r is None:
            top.set_visible(False); bot.set_visible(False)
            continue
        top.plot(mus, r["spice_profile"], **spice_kw())
        for atm in PHOEBE_ATMS:
            prof = r.get(f"phoebe_{atm}_profile")
            if prof is None or not np.all(np.isfinite(prof)):
                continue
            top.plot(mus, prof, **phoebe_kw(atm, zorder=4))
            if atm in resid_atms:
                bot.plot(mus, r["spice_profile"] - prof,
                         **phoebe_kw(atm, zorder=4))
        bot.axhline(0.0, color="0.6", lw=0.6, zorder=1)
        top.set_title(f"$T_\\mathrm{{eff}} = {teff:.0f}$ K")
        bot.set_xlabel("$\\mu$")
        if j == 0:
            top.set_ylabel("$I(\\mu)\\,/\\,I(1)$")
            bot.set_ylabel("SPICE $-$ PHOEBE")
    fig.legend(handles=legend_handles([SPICE_ID] + PHOEBE_ATMS),
               loc="outside lower center", ncol=4)
    fig.suptitle(f"Limb darkening in {BAND_LABEL[band]} "
                 f"($\\log g = {logg}$, [M/H] $= {feh:+.1f}$); "
                 "residual row excludes the un-limb-darkened blackbody",
                 fontsize=9)
    _save(fig, out, "fig_limb_darkening")
    plt.close(fig)
    return "fig_limb_darkening.pdf"


# --- Figure 2: ldint and the linear coefficient ---------------------------

def fig_ldint(grid, out):
    records = index_records(grid)
    teffs = sorted(grid["axes"]["teffs"])
    logg, feh = 4.5, 0.0

    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.0), sharex=True,
                             sharey="row", constrained_layout=True)
    for j, band in enumerate(BANDS):
        a_top, a_bot = axes[0, j], axes[1, j]
        for atm in [SPICE_ID] + PHOEBE_ATMS:
            key = "spice_ldint" if atm == SPICE_ID else f"phoebe_{atm}_ldint"
            ukey = key.replace("_ldint", "_ld_linear")
            y = [records.get((band, t, logg, feh), {}).get(key, np.nan)
                 for t in teffs]
            u = [records.get((band, t, logg, feh), {}).get(ukey, np.nan)
                 for t in teffs]
            if not np.any(np.isfinite(np.asarray(y, dtype=float))):
                continue
            kw = spice_kw() if atm == SPICE_ID else phoebe_kw(atm, zorder=4)
            a_top.plot(teffs, y, **kw)
            a_bot.plot(teffs, u, **kw)
        a_top.set_title(BAND_LABEL[band])
        a_bot.set_xlabel("$T_\\mathrm{eff}$ [K]")
        a_bot.tick_params(axis="x", labelrotation=45)
    axes[0, 0].set_ylabel("$\\mathrm{ldint}$")
    axes[1, 0].set_ylabel("linear $u$")
    fig.legend(handles=legend_handles([SPICE_ID] + PHOEBE_ATMS),
               loc="outside lower center", ncol=4)
    fig.suptitle("Intensity-to-flux conversion and limb-darkening strength "
                 f"($\\log g = {logg}$, [M/H] $= {feh:+.1f}$); "
                 "a uniform disc has ldint $=1$, $u=0$", fontsize=9)
    _save(fig, out, "fig_ldint")
    plt.close(fig)
    return "fig_ldint.pdf"


# --- Figure 3: response along each parameter axis -------------------------

def fig_response_axes(grid, out):
    records = index_records(grid)
    reference = tuple(grid["reference"])
    axes_def = [
        ("teffs", 0, "$T_\\mathrm{eff}$ [K]"),
        ("loggs", 1, "$\\log g$"),
        ("fehs", 2, "[M/H]"),
    ]
    fig, axarr = plt.subplots(4, 3, figsize=(7.1, 7.4), sharey=True,
                              constrained_layout=True)
    for i, band in enumerate(BANDS):
        for j, (axis_key, idx, xlabel) in enumerate(axes_def):
            ax = axarr[i, j]
            xs = sorted(grid["axes"][axis_key])
            for atm in PHOEBE_ATMS:
                ys = []
                for x in xs:
                    p = list(reference)
                    p[idx] = x
                    ys.append(response(records, band, atm, *p, reference))
                if np.all(~np.isfinite(np.asarray(ys, dtype=float))):
                    continue
                ax.plot(xs, ys, **phoebe_kw(atm, marker="o", ms=2.5))
            ax.axhline(0.0, color="0.6", lw=0.6, zorder=1)
            if i == 0:
                ax.set_title(xlabel.split(" [")[0], fontsize=8.5)
            if i == len(BANDS) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(f"{BAND_LABEL[band]}\n$\\Delta \\log_{{10}} I$ [dex]")
            if axis_key == "teffs":
                ax.tick_params(axis="x", labelrotation=45)
    fig.legend(handles=[Line2D([], [], label=ATM_LABEL[a], **phoebe_kw(a))
                        for a in PHOEBE_ATMS],
               loc="outside lower center", ncol=3)
    fig.suptitle("Intensity response, SPICE $-$ PHOEBE: "
                 "$\\log_{10} I(P)/I(P_\\mathrm{ref})$ differenced between the codes\n"
                 f"(one axis varied, the others held at "
                 f"$T_\\mathrm{{eff}}={reference[0]:.0f}$ K, "
                 f"$\\log g={reference[1]}$, [M/H]$={reference[2]:+.1f}$)",
                 fontsize=9)
    _save(fig, out, "fig_response_axes")
    plt.close(fig)
    return "fig_response_axes.pdf"


# --- Figure 4: response maps in the (Teff, log g) plane -------------------

def fig_response_maps(grid, out):
    records = index_records(grid)
    reference = tuple(grid["reference"])
    teffs = sorted(grid["axes"]["teffs"])
    loggs = sorted(grid["axes"]["loggs"])
    feh = 0.0

    # One symmetric scale for every panel so panels are comparable at a glance;
    # a per-panel scale would make a small difference look like a large one.
    allv = [response(records, b, a, t, g, feh, reference)
            for b in BANDS for a in PHOEBE_ATMS for t in teffs for g in loggs]
    allv = np.asarray(allv, dtype=float)
    vmax = float(np.nanmax(np.abs(allv))) if np.any(np.isfinite(allv)) else 1.0

    # No sharex/sharey: the "not tabulated" panels would otherwise inherit tick
    # locators from their siblings and print every Teff label on top of itself.
    fig, axarr = plt.subplots(len(BANDS), len(PHOEBE_ATMS),
                              figsize=(2.4 * len(PHOEBE_ATMS) + 1.4, 7.6),
                              constrained_layout=True, squeeze=False)
    # squeeze=False keeps this 2-D even for the control suite, which has a
    # single atmosphere and would otherwise come back as a 1-D array.
    im = None
    for i, band in enumerate(BANDS):
        for j, atm in enumerate(PHOEBE_ATMS):
            ax = axarr[i, j]
            Z = np.array([[response(records, band, atm, t, g, feh, reference)
                           for t in teffs] for g in loggs])
            if np.all(~np.isfinite(Z)):
                ax.text(0.5, 0.5, "not tabulated", ha="center", va="center",
                        transform=ax.transAxes, color="0.45", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                continue
            im = ax.imshow(Z, origin="lower", aspect="auto", cmap=DIVERGING,
                           vmin=-vmax, vmax=vmax,
                           extent=[-0.5, len(teffs) - 0.5, -0.5, len(loggs) - 0.5])
            ax.set_xticks(range(len(teffs)))
            ax.set_xticklabels([f"{t:.0f}" for t in teffs], rotation=45)
            ax.set_yticks(range(len(loggs)))
            ax.set_yticklabels([f"{g:g}" for g in loggs])
            ax.grid(False)
            # Direct value labels: the map shows the pattern, the numbers make it
            # readable in print and for a colour-blind or greyscale reader.
            for a_i in range(Z.shape[0]):
                for b_i in range(Z.shape[1]):
                    if np.isfinite(Z[a_i, b_i]):
                        ax.text(b_i, a_i, f"{Z[a_i, b_i]:+.02f}", ha="center",
                                va="center", fontsize=5.2, color="0.15")
            if i == 0:
                ax.set_title(ATM_LABEL[atm])
            if j == 0:
                ax.set_ylabel(f"{BAND_LABEL[band]}\n$\\log g$")
    # Declutter: x labels only on the lowest populated panel of each column,
    # y labels only in the first column. Repeating them in every panel triples
    # the ink for no added information.
    for j in range(len(PHOEBE_ATMS)):
        populated = [i for i in range(len(BANDS)) if axarr[i, j].images]
        for i in populated[:-1]:
            axarr[i, j].set_xticklabels([])
        if populated:
            axarr[populated[-1], j].set_xlabel("$T_\\mathrm{eff}$ [K]")
        for i in populated:
            if j != 0:
                axarr[i, j].set_yticklabels([])
    if im is not None:
        cb = fig.colorbar(im, ax=axarr, location="right", shrink=0.6, pad=0.02)
        cb.set_label("$\\Delta \\log_{10} I$  (SPICE $-$ PHOEBE) [dex]")
    fig.suptitle("Intensity response difference across the "
                 f"$T_\\mathrm{{eff}}$-$\\log g$ plane at [M/H] $= {feh:+.1f}$",
                 fontsize=9)
    _save(fig, out, "fig_response_maps")
    plt.close(fig)
    return "fig_response_maps.pdf"


# --- Figure 5: light ratios from the binary runs --------------------------

def fig_light_ratios(binary, out):
    systems = [k for k in ("twins", "cool-hot", "metal-poor") if k in binary]
    fig, axarr = plt.subplots(1, len(systems), figsize=(7.1, 2.9),
                              constrained_layout=True, sharey=False)
    axarr = np.atleast_1d(axarr)
    y = np.arange(len(BANDS))
    for k, name in enumerate(systems):
        ax = axarr[k]
        r = binary[name]
        for atm in PHOEBE_ATMS:
            xs = [r["phoebe_ratio"].get((atm, b), np.nan) for b in BANDS]
            ax.scatter(xs, y + 0.13, s=22, facecolors="none",
                       edgecolors=ATM_COLOR[atm], linewidths=1.2,
                       label=ATM_LABEL[atm], zorder=3)
        xs = [r["spice_ratio"].get(SPICE_KEY, {}).get(b, np.nan) for b in BANDS]
        ax.scatter(xs, y - 0.13, s=26, color=ATM_COLOR[SPICE_ID], marker="D",
                   label=ATM_LABEL[SPICE_ID], zorder=4)
        for yy in y:
            ax.axhline(yy, color="0.9", lw=6, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([BAND_LABEL[b] for b in BANDS] if k == 0 else [])
        ax.set_xlabel("$L_2/L_1$")
        # The twins are identical by construction, so every ratio is exactly 1
        # and an auto-scaled axis renders float noise as "1e-12 + 10e-1".
        # Enforce a minimum span so a degenerate panel reads as "all at 1.0".
        finite = [c.get_offsets()[:, 0] for c in ax.collections]
        finite = np.concatenate(finite) if finite else np.array([np.nan])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            mid, span = 0.5 * (lo + hi), max(hi - lo, 0.10)
            ax.set_xlim(mid - 0.75 * span, mid + 0.75 * span)
        s = r["system"]
        ax.set_title(f"{name}\n$T_\\mathrm{{eff}} = {s['teff1']:.0f}/{s['teff2']:.0f}$ K, "
                     f"[M/H] $= {s['feh']:+.2f}$", fontsize=8)
        ax.grid(axis="y", visible=False)
    # Marker-shaped handles: this figure draws points, not lines, and a legend
    # whose swatches do not match the marks is worse than none.
    marks = [Line2D([], [], ls="none", marker="D", ms=5,
                    color=ATM_COLOR[SPICE_ID], label=ATM_LABEL[SPICE_ID])]
    marks += [Line2D([], [], ls="none", marker="o", ms=5.5, mfc="none",
                     mec=ATM_COLOR[a], mew=1.2, label=ATM_LABEL[a])
              for a in PHOEBE_ATMS]
    fig.legend(handles=marks, loc="outside lower center", ncol=4)
    fig.suptitle("Out-of-eclipse light ratio: hollow = PHOEBE, filled diamond = SPICE.\n"
                 "A light-ratio error moves the two eclipse depths in opposite directions.",
                 fontsize=9)
    _save(fig, out, "fig_light_ratios")
    plt.close(fig)
    return "fig_light_ratios.pdf"


# --- Figure 6: binary light curves ----------------------------------------

def fig_binary_lightcurves(binary, out, band="Stromgren:y"):
    systems = [k for k in ("twins", "cool-hot", "metal-poor") if k in binary]
    fig = plt.figure(figsize=(7.4, 4.8), constrained_layout=True)
    # One subfigure per system. Grouping this way lets each system carry its own
    # title and x label without the two colliding with the per-panel titles, and
    # keeps the eclipse pair visually bound together.
    subfigs = fig.subfigures(1, len(systems), wspace=0.02)
    subfigs = np.atleast_1d(subfigs)
    for k, name in enumerate(systems):
        r = binary[name]
        sub = subfigs[k]
        axarr = sub.subplots(2, 2, gridspec_kw={"height_ratios": [2.4, 1]})
        times = np.asarray(r["times"])
        ref, n = r["baseline_index"], r["n_per_eclipse"]
        blocks = [("primary", slice(0, ref)), ("secondary", slice(ref + 1, ref + 1 + n))]
        for e, (label, sl) in enumerate(blocks):
            a_top, a_bot = axarr[0, e], axarr[1, e]
            t = (times[sl] - times[sl].mean()) * 24.0  # hours from mid-eclipse
            spice = r["spice_dmag"].get((SPICE_KEY, band))
            for atm in PHOEBE_ATMS:
                d = r["phoebe_dmag"].get((atm, band))
                if d is None:
                    continue
                a_top.plot(t, d[sl], **phoebe_kw(atm, zorder=4))
                if spice is not None:
                    a_bot.plot(t, (spice[sl] - d[sl]) * 1e3,
                               **phoebe_kw(atm, zorder=4))
            if spice is not None:
                a_top.plot(t, spice[sl], **spice_kw())
            a_bot.axhline(0.0, color="0.6", lw=0.6, zorder=1)
            a_top.invert_yaxis()
            a_top.set_title(label, fontsize=7)
            a_top.tick_params(labelsize=6)
            a_bot.tick_params(labelsize=6)
            if e == 0 and k == 0:
                a_top.set_ylabel(f"$\\Delta m$ [mag]", fontsize=7)
                a_bot.set_ylabel("SPICE $-$ PHOEBE\n[mmag]", fontsize=7)
        sy = r["system"]
        sub.suptitle(f"{name}\n$T_\\mathrm{{eff}} = {sy['teff1']:.0f}/{sy['teff2']:.0f}$ K, "
                     f"[M/H] $= {sy['feh']:+.2f}$", fontsize=7.5)
        sub.supxlabel("hours from mid-eclipse", fontsize=7)
    fig.legend(handles=legend_handles([SPICE_ID] + PHOEBE_ATMS),
               loc="outside lower center", ncol=4)
    fig.suptitle(f"Eclipse light curves in {BAND_LABEL[band]} "
                 "(SPICE solid band, PHOEBE dashed; residual below each)",
                 fontsize=9)
    _save(fig, out, "fig_binary_lightcurves")
    plt.close(fig)
    return "fig_binary_lightcurves.pdf"


# --- Figure 7: eclipse-depth residual summary -----------------------------

def fig_depth_residuals(binary, out):
    systems = [k for k in ("twins", "cool-hot", "metal-poor") if k in binary]
    fig, axarr = plt.subplots(1, len(systems), figsize=(7.1, 2.9),
                              constrained_layout=True, sharex=True)
    axarr = np.atleast_1d(axarr)
    rows = [(b, e) for b in BANDS for e in ("primary", "secondary")]
    y = np.arange(len(rows))
    for k, name in enumerate(systems):
        ax = axarr[k]
        r = binary[name]
        ref, n = r["baseline_index"], r["n_per_eclipse"]
        sl_of = {"primary": slice(0, ref), "secondary": slice(ref + 1, ref + 1 + n)}
        for atm, off in zip(PHOEBE_ATMS, (0.22, 0.0, -0.22)):
            xs = []
            for band, ecl in rows:
                s = r["spice_dmag"].get((SPICE_KEY, band))
                p = r["phoebe_dmag"].get((atm, band))
                if s is None or p is None:
                    xs.append(np.nan); continue
                d = (s[sl_of[ecl]] - p[sl_of[ecl]]) * 1e3
                xs.append(float(np.sqrt(np.nanmean(d ** 2))))
            ax.scatter(xs, y + off, s=20, color=ATM_COLOR[atm],
                       label=ATM_LABEL[atm], zorder=3)
        ax.set_xscale("log")
        ax.axvline(1.0, color="0.6", lw=0.6, ls=":", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{BAND_LABEL[b]} {e[:4]}." for b, e in rows]
                           if k == 0 else [])
        ax.set_xlabel("depth residual rms [mmag]")
        ax.set_title(name, fontsize=8)
        ax.grid(axis="y", visible=False)
    # One shared range so the three systems are directly comparable; sharex on a
    # log axis set after the fact does not take.
    lims = [ax.get_xlim() for ax in axarr]
    ax_lo = min(l[0] for l in lims)
    ax_hi = max(l[1] for l in lims)
    for ax in axarr:
        ax.set_xlim(ax_lo, ax_hi)
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=5,
                               color=ATM_COLOR[a], label=ATM_LABEL[a])
                        for a in PHOEBE_ATMS],
               loc="outside lower center", ncol=3)
    fig.suptitle("Eclipse-depth residual, SPICE (MARCS emulator) $-$ PHOEBE.\n"
                 "Dotted line marks 1 mmag; note the logarithmic axis.", fontsize=9)
    _save(fig, out, "fig_depth_residuals")
    plt.close(fig)
    return "fig_depth_residuals.pdf"


# --- Figure 8: absolute eclipse depths ------------------------------------

def fig_eclipse_depths(binary, out):
    """The depths themselves, as a companion to the residual figure.

    ``fig_depth_residuals`` shows how far apart the codes are; this shows what
    they actually predict, which is what an observer measures.

    Faceted by eclipse, with an independent x scale per panel. The two eclipses
    of a detached pair differ by ~5x in depth (0.03 vs 0.18 mag for cool-hot),
    so putting them on one axis compresses the primary into a stripe and hides
    the very differences the figure exists to show -- blackbody sits 24% off
    ck2004 there, which is invisible on a shared scale.
    """
    systems = [k for k in ("twins", "cool-hot", "metal-poor") if k in binary]
    fig, axarr = plt.subplots(2, len(systems), figsize=(7.4, 4.2),
                              constrained_layout=True)
    axarr = np.atleast_2d(axarr)
    y = np.arange(len(BANDS))
    for k, name in enumerate(systems):
        r = binary[name]
        ref, n = r["baseline_index"], r["n_per_eclipse"]
        sl_of = {"primary": slice(0, ref),
                 "secondary": slice(ref + 1, ref + 1 + n)}
        for e, ecl in enumerate(("primary", "secondary")):
            ax = axarr[e, k]

            def depths(source, key):
                out_ = []
                for band in BANDS:
                    d = source.get((key, band))
                    out_.append(np.nan if d is None
                                else float(np.nanmax(d[sl_of[ecl]])))
                return out_

            for yy in y:
                ax.axhline(yy, color="0.92", lw=9, zorder=0)
            for atm, off in zip(PHOEBE_ATMS, (0.22, 0.0, -0.22)):
                ax.scatter(depths(r["phoebe_dmag"], atm), y + off, s=20,
                           facecolors="none", edgecolors=ATM_COLOR[atm],
                           linewidths=1.2, zorder=3)
            ax.scatter(depths(r["spice_dmag"], SPICE_KEY), y, s=26,
                       color=ATM_COLOR[SPICE_ID], marker="D", zorder=4)
            ax.set_yticks(y)
            ax.set_yticklabels([BAND_LABEL[b] for b in BANDS] if k == 0 else [])
            ax.set_xlabel(f"{ecl} depth [mag]")
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="x", labelrotation=30, labelsize=6.5)
            if e == 0:
                sy = r["system"]
                ax.set_title(f"{name}\n"
                             f"$T_\\mathrm{{eff}} = {sy['teff1']:.0f}/{sy['teff2']:.0f}$ K, "
                             f"[M/H] $= {sy['feh']:+.2f}$", fontsize=7.5)
    marks = [Line2D([], [], ls="none", marker="D", ms=5,
                    color=ATM_COLOR[SPICE_ID], label=ATM_LABEL[SPICE_ID])]
    marks += [Line2D([], [], ls="none", marker="o", ms=5.5, mfc="none",
                     mec=ATM_COLOR[a], mew=1.2, label=ATM_LABEL[a])
              for a in PHOEBE_ATMS]
    fig.legend(handles=marks, loc="outside lower center", ncol=4)
    fig.suptitle("Eclipse depths predicted by each code and atmosphere "
                 "(filled diamond = SPICE, hollow = PHOEBE);\n"
                 "each panel has its own scale", fontsize=9)
    _save(fig, out, "fig_eclipse_depths")
    plt.close(fig)
    return "fig_eclipse_depths.pdf"


# --- Figure 9: stacked eclipse residuals (single_eclipse_system style) ----

def fig_eclipse_residual_stack(binary, out, band="Stromgren:y"):
    """One figure per system, in the layout of ``single_eclipse_system_residuals``.

    Two columns (primary / secondary eclipse) over three stacked rows:

    1. the light curves themselves, zeroed at the out-of-eclipse baseline;
    2. the absolute residual SPICE $-$ PHOEBE in mag;
    3. the same residual as a fraction of that eclipse's own depth, which is
       what actually matters for a depth measurement -- a 2 mmag residual is
       negligible on a 0.5 mag eclipse and 6% of a 0.03 mag one.

    Row 3 replaces the reference figure's visible-fraction residual: geometry is
    identical between the two codes here by construction (PHOEBE is run with
    ``distortion_method='sphere'`` to match SPICE's icosphere), so a visibility
    diagnostic would be flat by design and carry no information.
    """
    written = []
    for name, r in binary.items():
        times = np.asarray(r["times"])
        ref, n = r["baseline_index"], r["n_per_eclipse"]
        blocks = [("Primary eclipse", slice(0, ref)),
                  ("Secondary eclipse", slice(ref + 1, ref + 1 + n))]
        spice = r["spice_dmag"].get((SPICE_KEY, band))
        if spice is None:
            continue

        fig, axarr = plt.subplots(3, 2, figsize=(7.4, 5.4), sharex="col",
                                  constrained_layout=True,
                                  gridspec_kw={"height_ratios": [2.6, 1, 1]})
        for e, (title, sl) in enumerate(blocks):
            a_lc, a_res, a_frac = axarr[0, e], axarr[1, e], axarr[2, e]
            t = times[sl]
            depth = float(np.nanmax(spice[sl]))
            for atm in PHOEBE_ATMS:
                d = r["phoebe_dmag"].get((atm, band))
                if d is None:
                    continue
                a_lc.plot(t, d[sl], **phoebe_kw(atm, marker="s", ms=3, zorder=4))
                a_res.plot(t, spice[sl] - d[sl],
                           **phoebe_kw(atm, marker="o", ms=3, zorder=4))
                if depth > 0:
                    a_frac.plot(t, (spice[sl] - d[sl]) / depth * 100.0,
                                **phoebe_kw(atm, marker="o", ms=3, zorder=4))
            a_lc.plot(t, spice[sl], **spice_kw(marker="o", ms=3.4,
                                               markerfacecolor=ATM_COLOR[SPICE_ID],
                                               alpha=1.0, lw=1.6))
            for a in (a_res, a_frac):
                a.axhline(0.0, color="0.5", lw=0.6, zorder=1)
            a_lc.set_title(title)
            a_frac.set_xlabel("Time [days]")
            if e == 0:
                a_lc.set_ylabel("$\\Delta$ mag (zeroed at $t_0$)")
                a_res.set_ylabel("SPICE $-$ PHOEBE\n[mag]")
                a_frac.set_ylabel("residual /\ndepth [%]")
        fig.legend(handles=legend_handles([SPICE_ID] + PHOEBE_ATMS),
                   loc="outside lower center", ncol=4)
        sy = r["system"]
        fig.suptitle(f"{name}: $T_\\mathrm{{eff}} = {sy['teff1']:.0f}/{sy['teff2']:.0f}$ K, "
                     f"[M/H] $= {sy['feh']:+.2f}$, {BAND_LABEL[band]} "
                     f"({r['n_mesh']} elements)", fontsize=9)
        fname = _save(fig, out, f"fig_eclipse_residuals_{name}")
        plt.close(fig)
        written.append(fname)
    return written


# --- Figure 10: median/max sweep (blackbody_comparison style) -------------

def fig_parameter_sweep(grid, out):
    """Median and maximum deviation per parameter, in the layout of
    ``blackbody_comparison``.

    Rows are passbands, columns the three grid axes. At each value of the swept
    axis the statistic is taken over *all* combinations of the other two, so a
    point summarises a whole slice rather than one hand-picked cut -- the median
    is the typical disagreement, the maximum the worst case anywhere in that
    slice. Plotted as ``|Delta log10 I|`` on a log scale, matching the reference
    figure's ``|SPICE - PHOEBE|``.
    """
    records = index_records(grid)
    reference = tuple(grid["reference"])
    axes_def = [("teffs", 0, "$T_\\mathrm{eff}$ [K]"),
                ("loggs", 1, "$\\log g$"),
                ("fehs", 2, "[M/H]")]
    all_axes = {k: sorted(v) for k, v in grid["axes"].items()}

    fig, axarr = plt.subplots(len(BANDS), len(axes_def), figsize=(7.4, 7.6),
                              sharey=True, constrained_layout=True)
    for i, band in enumerate(BANDS):
        for j, (axis_key, idx, xlabel) in enumerate(axes_def):
            ax = axarr[i, j]
            xs = all_axes[axis_key]
            others = [(k2, i2) for k2, i2 in
                      (("teffs", 0), ("loggs", 1), ("fehs", 2)) if i2 != idx]
            for atm in PHOEBE_ATMS:
                med, mx = [], []
                for x in xs:
                    vals = []
                    for a in all_axes[others[0][0]]:
                        for b in all_axes[others[1][0]]:
                            p = [None, None, None]
                            p[idx] = x
                            p[others[0][1]] = a
                            p[others[1][1]] = b
                            v = response(records, band, atm, *p, reference)
                            if np.isfinite(v):
                                vals.append(abs(v))
                    med.append(np.median(vals) if vals else np.nan)
                    mx.append(np.max(vals) if vals else np.nan)
                if not np.any(np.isfinite(np.asarray(med, dtype=float))):
                    continue
                ax.plot(xs, med, color=ATM_COLOR[atm], ls="-", marker="o",
                        ms=3.2, lw=1.3, zorder=3)
                ax.plot(xs, mx, color=ATM_COLOR[atm], ls="--", marker="s",
                        ms=3.2, lw=1.1, alpha=0.85, zorder=3)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(xlabel.split(" [")[0], fontsize=8.5)
            if i == len(BANDS) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(f"{BAND_LABEL[band]}\n"
                              "$|\\Delta \\log_{10} I|$ [dex]")
            if axis_key == "teffs":
                ax.tick_params(axis="x", labelrotation=45)
    handles = [Line2D([], [], color=ATM_COLOR[a], lw=1.4, label=ATM_LABEL[a])
               for a in PHOEBE_ATMS]
    handles += [Line2D([], [], color="0.35", ls="-", marker="o", ms=3.5,
                       label="median over the other two axes"),
                Line2D([], [], color="0.35", ls="--", marker="s", ms=3.5,
                       label="maximum")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5)
    fig.suptitle("Deviation of SPICE from each PHOEBE atmosphere, summarised "
                 "over the grid\n(statistic taken across all combinations of "
                 "the two axes not shown)", fontsize=9)
    _save(fig, out, "fig_parameter_sweep")
    plt.close(fig)
    return "fig_parameter_sweep.pdf"


# --- Figure 11: mesh convergence ------------------------------------------

def load_mesh_sweep(results, suite):
    """``{n_mesh: binary_results}`` for every ``mesh<N>/`` under a suite."""
    out = {}
    root = results / suite
    if not root.exists():
        return out
    for d in sorted(root.glob("mesh*")):
        pkl = d / "binary_light_curves.pkl"
        if not pkl.exists():
            continue
        try:
            n = int(d.name[4:])
        except ValueError:
            continue
        with open(pkl, "rb") as f:
            out[n] = pickle.load(f)
    return out


def fig_mesh_convergence(sweep, out, suite):
    """Depth residual against mesh resolution.

    In the blackbody suite the two codes run identical physics, so the residual
    is *entirely* numerical and must fall as the meshes are refined; that fall is
    the calibration of what the atmospheres suite can resolve. Plotted per band
    and eclipse, log-log, with the realized PHOEBE triangle count annotated --
    ``ntriangles`` is only a target for its marching-triangles algorithm (1280
    requested comes back as 1458), whereas SPICE's icosphere is exact.
    """
    meshes = sorted(sweep)
    if len(meshes) < 2:
        return []
    systems = [k for k in ("twins", "cool-hot", "metal-poor")
               if k in sweep[meshes[0]]]
    spice_kind = spice_key_of(sweep[meshes[0]])

    fig, axarr = plt.subplots(1, len(systems), figsize=(7.4, 3.2),
                              constrained_layout=True, sharey=True, squeeze=False)
    axarr = axarr[0]
    for k, name in enumerate(systems):
        ax = axarr[k]
        for band in BANDS:
            for ecl, marker in (("primary", "o"), ("secondary", "s")):
                xs, ys = [], []
                for n in meshes:
                    r = sweep[n].get(name)
                    if r is None:
                        continue
                    ref, npe = r["baseline_index"], r["n_per_eclipse"]
                    sl = (slice(0, ref) if ecl == "primary"
                          else slice(ref + 1, ref + 1 + npe))
                    sp = r["spice_dmag"].get((spice_kind, band))
                    ph = r["phoebe_dmag"].get((PHOEBE_ATMS[0], band))
                    if sp is None or ph is None:
                        continue
                    d = (sp[sl] - ph[sl]) * 1e3
                    xs.append(n)
                    ys.append(float(np.sqrt(np.nanmean(d ** 2))))
                if len(xs) < 2:
                    continue
                ax.plot(xs, ys, marker=marker, ms=3.5, lw=1.1,
                        color=ATM_COLOR[PHOEBE_ATMS[0]],
                        alpha=0.85 if ecl == "primary" else 0.45,
                        ls="-" if ecl == "primary" else "--")
        # A 1/N guide line anchored to the coarsest point, for slope reference.
        anchor = None
        for band in BANDS:
            r = sweep[meshes[0]].get(name)
            if r is None:
                continue
            ref, npe = r["baseline_index"], r["n_per_eclipse"]
            sp = r["spice_dmag"].get((spice_kind, band))
            ph = r["phoebe_dmag"].get((PHOEBE_ATMS[0], band))
            if sp is None or ph is None:
                continue
            d = (sp[slice(0, ref)] - ph[slice(0, ref)]) * 1e3
            anchor = float(np.sqrt(np.nanmean(d ** 2)))
            break
        if anchor:
            g = np.array(meshes, dtype=float)
            ax.plot(g, anchor * (g[0] / g), color="0.5", lw=0.8, ls=":",
                    zorder=1)
            ax.annotate("$\\propto 1/N$", xy=(g[-1], anchor * g[0] / g[-1]),
                        xytext=(-2, 4), textcoords="offset points",
                        fontsize=6, color="0.4", ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(meshes)
        ax.set_xticklabels([str(m) for m in meshes])
        ax.set_xlabel("SPICE mesh elements $N$")
        ax.set_title(name, fontsize=8)
        if k == 0:
            ax.set_ylabel("depth residual rms [mmag]")
    handles = [Line2D([], [], color=ATM_COLOR[PHOEBE_ATMS[0]], marker="o",
                      ls="-", ms=4, label="primary eclipse"),
               Line2D([], [], color=ATM_COLOR[PHOEBE_ATMS[0]], marker="s",
                      ls="--", ms=4, alpha=0.5, label="secondary eclipse")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    realized = sweep[meshes[0]][systems[0]].get("n_mesh_phoebe")
    note = ""
    if realized:
        note = ("; PHOEBE realizes more than requested "
                f"({meshes[0]} -> {realized['primary']})")
    fig.suptitle(f"Mesh convergence, {suite} suite: one line per passband{note}",
                 fontsize=9)
    _save(fig, out, "fig_mesh_convergence")
    plt.close(fig)
    return ["fig_mesh_convergence.pdf"]


# --- Figure 12: ingress/egress shape ---------------------------------------

def fig_ingress_shape(binary, out, band="Stromgren:y", suffix=""):
    """Eclipse *shape* through ingress and egress, with depth divided out.

    Depth is set by the light ratio; **shape** is set by limb darkening -- how
    fast the flux falls as the occulter covers successive annuli of the disc. So
    normalising each code's curve by its own maximum depth removes the light
    ratio and leaves a near-pure LD comparison, which is a far sharper
    atmosphere diagnostic than depth alone.

    Top row: normalised light curve L(t)/L_max through the eclipse.
    Bottom row: shape residual (SPICE - PHOEBE) in percent of depth. A pure
    light-ratio error cancels here; what survives is the LD profile difference.

    Needs dense sampling -- with only a handful of points per eclipse the curve
    is a polygon and its "shape" is an artefact of where the samples landed.
    """
    systems = [k for k in ("twins", "cool-hot", "metal-poor") if k in binary]
    if not systems:
        return []
    fig, axarr = plt.subplots(2, len(systems), figsize=(7.4, 4.4),
                              constrained_layout=True, squeeze=False,
                              gridspec_kw={"height_ratios": [2.2, 1]})
    spice_kind = spice_key_of(binary)
    for k, name in enumerate(systems):
        r = binary[name]
        ref, n = r["baseline_index"], r["n_per_eclipse"]
        sl = slice(0, ref)                       # primary eclipse
        t = np.asarray(r["times"])[sl]
        t = (t - t.mean()) * 24.0                # hours from mid-eclipse
        a_top, a_bot = axarr[0, k], axarr[1, k]
        sp = r["spice_dmag"].get((spice_kind, band))
        if sp is None:
            continue
        sp_d = float(np.nanmax(sp[sl]))
        a_top.plot(t, sp[sl] / sp_d, **spice_kw())
        for atm in PHOEBE_ATMS:
            ph = r["phoebe_dmag"].get((atm, band))
            if ph is None:
                continue
            ph_d = float(np.nanmax(ph[sl]))
            a_top.plot(t, ph[sl] / ph_d, **phoebe_kw(atm, zorder=4))
            a_bot.plot(t, (sp[sl] / sp_d - ph[sl] / ph_d) * 100.0,
                       **phoebe_kw(atm, zorder=4))
        a_bot.axhline(0.0, color="0.5", lw=0.6, zorder=1)
        a_top.invert_yaxis()
        a_top.set_title(f"{name}  ({len(t)} pts)", fontsize=8)
        a_bot.set_xlabel("hours from mid-eclipse")
        if k == 0:
            a_top.set_ylabel(f"$\\Delta m / \\Delta m_\\mathrm{{max}}$")
            a_bot.set_ylabel("shape residual\n[% of depth]")
    fig.legend(handles=legend_handles([SPICE_ID] + PHOEBE_ATMS),
               loc="outside lower center", ncol=4)
    fig.suptitle(f"Eclipse shape in {BAND_LABEL[band]}, depth normalised out "
                 "(isolates limb darkening from the light ratio)", fontsize=9)
    fname = _save(fig, out, f"fig_ingress_shape{suffix}")
    plt.close(fig)
    return [fname]


def shape_residual_rms(r, spice_kind, atm, band):
    """RMS of the depth-normalised shape residual over the primary eclipse [%].

    Dividing each curve by its own maximum depth removes the light ratio, so
    what is left is limb darkening. Returns NaN when either curve is missing.
    """
    ref = r["baseline_index"]
    sl = slice(0, ref)
    sp = r["spice_dmag"].get((spice_kind, band))
    ph = r["phoebe_dmag"].get((atm, band))
    if sp is None or ph is None:
        return np.nan
    sp_d, ph_d = float(np.nanmax(sp[sl])), float(np.nanmax(ph[sl]))
    if not (sp_d > 0 and ph_d > 0):
        return np.nan
    d = (sp[sl] / sp_d - ph[sl] / ph_d) * 100.0
    return float(np.sqrt(np.nanmean(d ** 2)))


def fig_shape_summary(dense, out, control=None):
    """Shape-residual rms per passband -- the quantitative form of the shape figure.

    Limb darkening is wavelength dependent (cooler continuum opacity toward the
    blue steepens the profile), so the band-to-band pattern is itself a
    diagnostic rather than a repeat of the same number four times.

    When the blackbody control is supplied its residual is drawn as a shaded
    floor: identical physics on both sides, so anything at or below that level
    is geometry rather than atmosphere.
    """
    systems = [k for k in ("cool-hot", "metal-poor", "twins") if k in dense]
    if not systems:
        return []
    spice_kind = spice_key_of(dense)
    fig, axarr = plt.subplots(1, len(systems), figsize=(7.1, 3.0),
                              constrained_layout=True, squeeze=False, sharex=True)
    axarr = axarr[0]
    y = np.arange(len(BANDS))
    for k, name in enumerate(systems):
        ax = axarr[k]
        r = dense[name]
        for yy in y:
            ax.axhline(yy, color="0.92", lw=8, zorder=0)
        if control is not None and name in control:
            c_kind = spice_key_of(control)
            c_atms = control[name].get("atms") or ["blackbody"]
            floor = [shape_residual_rms(control[name], c_kind, c_atms[0], b)
                     for b in BANDS]
            ax.scatter(floor, y, marker="|", s=110, color="0.35", zorder=5,
                       label="blackbody control (numerical floor)")
        for atm, off in zip(PHOEBE_ATMS, (0.17, -0.17)):
            xs = [shape_residual_rms(r, spice_kind, atm, b) for b in BANDS]
            ax.scatter(xs, y + off, s=26, color=ATM_COLOR[atm],
                       label=ATM_LABEL[atm], zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([BAND_LABEL[b] for b in BANDS] if k == 0 else [])
        ax.set_xlabel("shape residual rms [% of depth]")
        ax.set_title(name, fontsize=8)
        ax.grid(axis="y", visible=False)
    handles, labels = axarr[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="outside lower center", ncol=3)
    fig.suptitle("Eclipse-shape disagreement per passband, depth normalised out\n"
                 "(shape is set by limb darkening; the light ratio cancels)",
                 fontsize=9)
    _save(fig, out, "fig_shape_summary")
    plt.close(fig)
    return ["fig_shape_summary.pdf"]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=here / "atmosphere_out")
    parser.add_argument("--suites", nargs="+", default=None,
                        help="default: every suite directory present")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <results>/<suite>/figures")
    args = parser.parse_args()

    setup_style()
    suites = args.suites or sorted(
        d.name for d in args.results.iterdir()
        if d.is_dir() and (d / "intensity_grid.pkl").exists()
           or d.is_dir() and (d / "binary_light_curves.pkl").exists())
    if not suites:
        print(f"no suite directories with results under {args.results}")
        return

    for suite in suites:
        res = args.results / suite
        out = args.out / suite if args.out else res / "figures"
        out.mkdir(parents=True, exist_ok=True)
        written = []

        grid_path = res / "intensity_grid.pkl"
        binary_path = res / "binary_light_curves.pkl"
        grid = binary = None
        if grid_path.exists():
            with open(grid_path, "rb") as f:
                grid = pickle.load(f)
        if binary_path.exists():
            with open(binary_path, "rb") as f:
                binary = pickle.load(f)

        # Fix this suite's identity before drawing anything, so no figure can
        # mix the two comparisons.
        atms = (grid or {}).get("phoebe_atms")
        if atms is None and binary is not None:
            atms = next(iter(binary.values())).get("atms")
        spice_kind = spice_key_of(binary) if binary else (
            "blackbody" if suite == "blackbody" else "aemu")
        set_suite(atms or PHOEBE_ATMS, spice_kind)
        print(f"\n[{suite}] SPICE={spice_kind}  PHOEBE={PHOEBE_ATMS}")

        if grid is not None:
            written += [fig_limb_darkening(grid, out), fig_ldint(grid, out),
                        fig_response_axes(grid, out), fig_response_maps(grid, out),
                        fig_parameter_sweep(grid, out)]
        if binary is not None:
            written += [fig_light_ratios(binary, out),
                        fig_binary_lightcurves(binary, out),
                        fig_eclipse_depths(binary, out),
                        fig_depth_residuals(binary, out)]
            written += fig_eclipse_residual_stack(binary, out)

        dense_path = res / "dense" / "binary_light_curves.pkl"
        if dense_path.exists():
            with open(dense_path, "rb") as f:
                dense = pickle.load(f)
            written += fig_ingress_shape(dense, out)
            control = None
            ctrl_path = args.results / "blackbody" / "dense" / "binary_light_curves.pkl"
            if ctrl_path.exists():
                with open(ctrl_path, "rb") as f:
                    control = pickle.load(f)
            written += fig_shape_summary(dense, out, control)

        sweep = load_mesh_sweep(args.results, suite)
        if len(sweep) >= 2:
            written += fig_mesh_convergence(sweep, out, suite)

        for name in written:
            print(f"  wrote {out / name}")


if __name__ == "__main__":
    main()
