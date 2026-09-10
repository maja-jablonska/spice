"""The TZ For line-formation grid, drawn from the *fitted* spectra.

A remake of the schematic Fe-line grid (`tz_fornacis_fe_line_spice_grid.pdf`) using the joint
fit's own output instead of a demonstration model: three orbital phases across, and the primary's
contribution, the secondary's contribution and the blended spectrum down, with the observed HARPS
spectrum over the blend and the residual beneath it.

The 21 HARPS epochs miss both eclipses (the nearest sits at phase 0.149 against an eclipse centred
on 0.159), so the columns cannot be "out of eclipse / primary eclipse / secondary eclipse" as in
the schematic. They are chosen instead to walk the radial-velocity cycle, which is what actually
separates the two stars in a spectrum: maximum separation with the giant redshifted, the blended
configuration where the two line systems overlap, and maximum separation the other way. The orbit
sketch along the top is drawn from the fitted geometry, with each star's disc sized by its fitted
radius, coloured by its fitted temperature and limb-darkened.

Reads a model dump from ``tzfor_fullrange_fit.py --dump-model``; no synthesis, no GPU.
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tzfor_constants as K

DEFAULT_DUMP = HERE / "tzfor_aemu_out" / "fr_model_maskall_hp.npz"
OUT_DIR = HERE.parent / "paper_plots" / "tz_fornacis"
C_KMS = 299792.458


def blackbody_colour(teff):
    """A rough visual colour for a star of this temperature, for the orbit sketch only."""
    anchors = [(3500, (0.95, 0.55, 0.25)), (4900, (1.00, 0.78, 0.45)), (6300, (1.00, 0.95, 0.85)), (8000, (0.80, 0.87, 1.00))]
    t = float(np.clip(teff, anchors[0][0], anchors[-1][0]))
    for (t0, c0), (t1, c1) in zip(anchors, anchors[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(a + f * (b - a) for a, b in zip(c0, c1))
    return anchors[-1][1]


def draw_star(ax, x, y, radius, teff, u=0.6, n=160, zorder=2):
    """A limb-darkened disc at (x, y) in data units."""
    g = np.linspace(-1, 1, n); X, Y = np.meshgrid(g, g); r2 = X ** 2 + Y ** 2
    mu = np.sqrt(np.clip(1 - r2, 0, 1)); img = np.where(r2 <= 1, 1 - u * (1 - mu), np.nan)
    base = blackbody_colour(teff)
    cmap = LinearSegmentedColormap.from_list("star", [(0.12 * base[0], 0.10 * base[1], 0.16), base])
    ax.imshow(img, extent=(x - radius, x + radius, y - radius, y + radius), origin="lower",
              cmap=cmap, norm=Normalize(0.0, 1.0), zorder=zorder, interpolation="bilinear")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=str(DEFAULT_DUMP))
    ap.add_argument("--window", type=float, nargs=2, default=(5029.0, 5035.0), help="wavelength range [A]")
    ap.add_argument("--epochs", type=int, nargs="*", default=None, help="epoch indices; default picks the RV cycle automatically")
    ap.add_argument("--rest-line", type=float, default=5031.9, help="a rest wavelength to mark (dotted)")
    ap.add_argument("--radius-exaggeration", type=float, default=1.5, help="draw the discs this many times larger than scale; capped so they never appear to touch when the real stars do not")
    ap.add_argument("--out", default="tzfor_line_grid_fitted"); ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    th, pn = d["theta"], list(d["pnames"])
    getp = lambda k, dflt=np.nan: float(th[pn.index(k)]) if k in pn else dflt
    T1, T2 = getp("Teff1"), getp("Teff2")
    R1, R2 = getp("R1", K.PRIMARY_RADIUS), getp("R2", K.SECONDARY_RADIUS)
    incl = getp("incl", K.INCL_DEG)
    wl = 10.0 ** d["logwl"]; rv = d["rv"]
    phase = ((d["times"] - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
    sep = rv[1] - rv[0]

    # ---- the orbit, taken from the fitted radial velocities themselves ----
    # The mesh ephemeris and the photometric one differ by a third of a cycle (the fit absorbs that in
    # dphi), so the sketch must not assume where the eclipses fall: solve the circular-orbit RV curve
    # RV = gamma + K cos(2 pi (phase - phase_max)) instead, which is exact for these 21 epochs.
    M = np.column_stack([np.ones_like(phase), np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)])
    coef1 = np.linalg.lstsq(M, rv[0], rcond=None)[0]
    gamma, K1 = float(coef1[0]), float(np.hypot(coef1[1], coef1[2]))
    ph_max1 = float(np.arctan2(coef1[2], coef1[1]) / (2 * np.pi)) % 1.0     # phase of the giant's maximum recession
    rms_fit = float(np.std(rv[0] - M @ coef1))
    # star 2's orbital angle: RV2 - gamma ∝ cos(theta2) and theta2 = 2 pi (phase - ph_max1) + pi.
    # Sky: x = a2 cos(theta), y = a2 cos(i) sin(theta), depth z = a2 sin(i) sin(theta) (observer at +z).
    theta2 = lambda p_: 2 * math.pi * (p_ - ph_max1) + math.pi
    conj = [(ph_max1 + 0.25) % 1.0, (ph_max1 + 0.75) % 1.0]
    deep = min(conj, key=lambda p_: math.sin(theta2(p_)))                    # secondary behind the giant = deeper minimum
    print(f"RV curve: gamma {gamma:+.2f}, K1 {K1:.2f} km/s, residual rms {rms_fit:.3f} km/s; "
          f"conjunctions at phase {conj[0]:.3f} and {conj[1]:.3f} (secondary occulted at {deep:.3f})")

    # ---- three phases along the radial-velocity cycle ----
    if args.epochs:
        epochs = list(args.epochs)
    else:
        epochs = [int(np.argmin(sep)), int(np.argmin(np.abs(sep))), int(np.argmax(sep))]
    def describe(e):
        th2 = theta2(phase[e]); front = math.sin(th2) > 0
        if abs(sep[e]) < 0.35 * (K1 + np.hypot(*np.linalg.lstsq(M, rv[1], rcond=None)[0][1:])):
            return "near conjunction — line systems blended\n" + ("secondary in front" if front else "secondary behind")
        return ("maximum separation — giant receding" if sep[e] < 0 else "maximum separation — giant approaching") + \
               "\n" + ("secondary in front" if front else "secondary behind")
    titles = [describe(e) for e in epochs]

    w0, w1 = args.window; sel = (wl >= w0) & (wl <= w1)
    # semi-major axis of the relative orbit from Kepler's third law with the dynamical masses
    G, Msun, yr, Rsun = 6.67430e-11, 1.98847e30, 3.15576e7, 6.957e8
    a = (G * (K.PRIMARY_MASS + K.SECONDARY_MASS) * Msun * (K.PERIOD_YR * yr) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / Rsun
    q = K.SECONDARY_MASS / K.PRIMARY_MASS            # the barycentre splits the orbit by mass, not by radius
    # Exaggerating the discs makes the configuration readable, but must never fake a contact: cap the
    # factor at the closest projected approach among the phases actually drawn.
    EX = args.radius_exaggeration
    fig = plt.figure(figsize=(13.5, 9.6))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.05, 1.0, 1.0, 1.45], hspace=0.17, wspace=0.07)

    def sky_sep(p_):
        th = theta2(p_); ci_ = math.cos(math.radians(incl))
        return a * math.hypot(math.cos(th), ci_ * math.sin(th))
    closest = min(sky_sep(phase[e]) for e in epochs)
    if EX * (R1 + R2) > closest:
        EX = max(1.0, 0.95 * closest / (R1 + R2))
        print(f"radius exaggeration capped at {EX:.2f} (closest projected separation {closest:.1f} R_sun vs R1+R2 {R1 + R2:.1f})")

    for col, (e, title) in enumerate(zip(epochs, titles)):
        ph = phase[e]
        # --- orbit sketch: a circular orbit seen at the fitted inclination ---
        ax = fig.add_subplot(gs[0, col]); ax.set_aspect("equal"); ax.axis("off")
        th2 = theta2(ph); ci = math.cos(math.radians(incl))
        a2, a1 = a / (1 + q), a * q / (1 + q)          # barycentric semi-major axes (q = M2/M1)
        x2, y2, z2 = a2 * math.cos(th2), a2 * ci * math.sin(th2), math.sin(th2)
        x1, y1, z1 = -a1 * math.cos(th2), -a1 * ci * math.sin(th2), -math.sin(th2)
        pairs = [(x1, y1, EX * R1, T1, z1), (x2, y2, EX * R2, T2, z2)]
        for x, y, r, t, z in sorted(pairs, key=lambda p_: p_[4]):     # the more distant star is drawn first
            draw_star(ax, x, y, r, t, zorder=2 + (z > 0))
        ax.annotate("", xy=(a * 0.82, -a * 0.13), xytext=(a * 0.82, a * 0.13), arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.6))
        ax.text(a * 0.82, a * 0.17, "to observer", color="C3", ha="center", fontsize=7.5)
        ax.set_xlim(-a * 1.0, a * 1.0); ax.set_ylim(-a * 0.32, a * 0.32)
        ax.set_title(f"{title}\n$\\varphi$ = {ph:.3f},  $\\Delta v$ = {sep[e]:+.0f} km s$^{{-1}}$", fontsize=9.5, pad=4)
        if col == 0:
            ax.text(0.0, -0.02, f"separations to scale, discs ×{EX:g}", transform=ax.transAxes, fontsize=7, color="0.45")

        # --- the two contributions and the blend ---
        rows = [("primary", f"primary (K giant, {T1:.0f} K)", "C0", rv[0, e]),
                ("secondary", f"secondary ({T2:.0f} K)", "C1", rv[1, e])]
        for r, (key, label, colour, v) in enumerate(rows):
            ax = fig.add_subplot(gs[1 + r, col])
            ax.plot(wl[sel], d[key][e][sel], color=colour, lw=1.1)
            ax.axvline(args.rest_line * (1 + v / C_KMS), color=colour, ls=":", lw=1.0)
            ax.set_xlim(w0, w1); ax.grid(alpha=0.25)
            if col: ax.set_xlim(w0 + 1e-6, w1)
            ax.tick_params(labelbottom=False)
            if col == 0: ax.set_ylabel(label.replace(" (", "\n("), fontsize=9)
            else: ax.tick_params(labelleft=False)
            ax.text(0.02, 0.06, f"RV {v:+.1f} km s$^{{-1}}$", transform=ax.transAxes, fontsize=8, color=colour)

        ax = fig.add_subplot(gs[3, col])
        good = d["good"][e] & sel; obs = np.where(np.isfinite(d["obs"][e]), d["obs"][e], np.nan)
        ax.plot(wl[sel], obs[sel], color="0.35", lw=0.9, label="HARPS")
        ax.plot(wl[sel], d["model"][e][sel], color="C3", lw=1.2, label="joint-fit model")
        res = obs[sel] - d["model"][e][sel]
        ax.plot(wl[sel], res + 0.45, color="0.55", lw=0.7, label="residual + 0.45")
        ax.axhline(0.45, color="0.8", lw=0.6)
        for v, colour in ((rv[0, e], "C0"), (rv[1, e], "C1")):    # where each star puts the marked line
            ax.axvline(args.rest_line * (1 + v / C_KMS), color=colour, ls=":", lw=1.0)
        ax.set_xlim(w0, w1); ax.set_ylim(0.35, 1.06); ax.grid(alpha=0.25)
        ax.set_xlabel("wavelength [Å]")
        ax.set_xticks(np.arange(math.ceil(w0), math.floor(w1) + 0.1, 1.0))
        if col: ax.set_xlim(w0 + 1e-6, w1)                         # keep the shared boundary tick off the neighbour
        rms = np.sqrt(np.nanmean((obs[good] - d["model"][e][good]) ** 2))
        ax.text(0.02, 0.06, f"rms {100 * rms:.1f}%", transform=ax.transAxes, fontsize=8)
        if col == 0:
            ax.set_ylabel("blended spectrum\n(normalised flux)", fontsize=9); ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
        else:
            ax.tick_params(labelleft=False)

    # common y limits per row, so the columns are comparable
    axes = fig.get_axes()
    for r in range(1, 3):
        row = [axes[i] for i in range(len(axes)) if i % 4 == r]
        lo = min(a_.get_ylim()[0] for a_ in row); hi = max(a_.get_ylim()[1] for a_ in row)
        for a_ in row: a_.set_ylim(lo, hi)

    geo_note = "geometry free" if "R1" in pn else "geometry fixed at the literature values"
    fig.suptitle(f"TZ For: how the blended spectrum is built, at the joint fit  "
                 f"($T_1$ {T1:.0f} K, $T_2$ {T2:.0f} K, $R_1$ {R1:.2f}, $R_2$ {R2:.2f} $R_\\odot$, {geo_note})", y=0.975, fontsize=12)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.out}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig); print(f"epochs {epochs} at phases {np.round(phase[epochs], 3)}")
    print(f"saved {out}/{args.out}.png and .pdf")


if __name__ == "__main__":
    main()
