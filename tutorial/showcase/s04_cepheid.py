"""Showcase 4 - delta Cephei breathing.

The SPIPS model of delta Cep (radius, Teff, dynamic log g and Luck-2018 xi over one
5.37 d cycle) drives a radially pulsating SPICE mesh. Left: the disc at true relative
size, coloured by Teff. Middle: an Fe I line, its asymmetry and shift through the
cycle. Right: radius, Teff and the line's centroid velocity traced out.
No rotation and no macroturbulence, so all the line motion is pulsation.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "paper_results" / "cepheid"))
import common as C  # noqa: E402

CSV = HERE.parent / "paper_results" / "cepheid" / "delta_cep_model_curves.csv"
PERIOD = 5.366265401100268   # days (cepheid_grid.csv)
FEH = 0.08                   # Luck 2018
LINE = 6151.617              # Fe I, clean in F/G supergiants
WL = (6150.0, 6153.3)
MASS = 4.726                 # SPIPS-implied (cepheid_phase.spips_mass)


def compute(args, out):
    C.setup_jax()
    import jax.numpy as jnp
    import pandas as pd
    from spice.models import IcosphereModel
    from spice.models.mesh_transform import add_pulsation, evaluate_pulsations
    from cepheid_phase import radius_template_mean_normalised, xi_micro_deltaCep, effective_logg

    emu = C.load_emulator()
    n_frames = 8 if args.quick else 80
    n_vert = 800 if args.quick else 5000
    wl, lw = C.wl_grid(*WL, 200 if args.quick else 660)
    tab = pd.read_csv(CSV)
    ph_t, R_t, T_t = tab["PHASE"].values, tab["R"].values, tab["Teff"].values
    ph_g, logg_g, _ = effective_logg({"PHASE": ph_t, "R": R_t, "logg": tab["logg"].values}, PERIOD, MASS)
    fourier = radius_template_mean_normalised(ph_t, R_t)
    R_mean = float(R_t.mean())
    names = emu.stellar_parameter_names
    p0 = C.stellar_params(emu, T_t.mean(), 1.9, FEH, 3.0)
    base = IcosphereModel.construct(n_vert, R_mean, MASS, p0, names, max_fourier_order=10)
    puls = add_pulsation(base, m_order=0, l_degree=0, period=PERIOD, fourier_series_parameters=fourier)

    phases = np.arange(n_frames) / n_frames
    i_t, i_g, i_x = names.index("teff"), names.index("logg"), names.index("vmicro")
    flux, snaps, radius, teff, vcen = [], [], [], [], []
    for k, ph in enumerate(phases):
        m = evaluate_pulsations(puls, ph * PERIOD)
        T = float(np.interp(ph, ph_t, T_t, period=1.0))
        g = float(np.interp(ph, ph_g, logg_g, period=1.0))
        P = m.parameters.at[:, i_t].set(T).at[:, i_g].set(g).at[:, i_x].set(float(xi_micro_deltaCep(ph)))
        m = m._replace(parameters=P)
        f, c = C.synth(emu, m, lw)
        nf = f / c
        vel = (wl / LINE - 1) * C.C_KMS
        depth = np.clip(1 - nf, 0, None)
        core = np.abs(vel - vel[np.argmin(nf)]) < 25
        vcen.append(float(np.sum(vel[core] * depth[core]) / np.sum(depth[core])))
        flux.append(nf)
        radius.append(np.sqrt(float(np.sum(np.asarray(m.visible_cast_areas))) / np.pi))
        teff.append(T)
        snaps.append(C.snapshot(m, {"teff": np.asarray(m.parameters)[:, i_t], "vlos": m.los_velocities}))
        print(f"phase {ph:.3f}: R {radius[-1]:.2f} Teff {T:.0f} v {vcen[-1]:+.1f}", flush=True)
    np.savez_compressed(out / "data.npz", wl=wl, flux=np.array(flux), phase=phases, radius=np.array(radius),
                        teff=np.array(teff), vcen=np.array(vcen), R_mean=R_mean, **C.pack_snapshots(snaps))


def render(out, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    snaps = C.unpack_snapshots(d)
    wl, flux, ph = d["wl"], d["flux"], d["phase"]
    R, T, v = d["radius"], d["teff"], d["vcen"]
    vel = (wl / LINE - 1) * C.C_KMS
    norm = Normalize(T.min() - 100, T.max() + 100)
    lim = R.max() * 1.08
    mean_prof = flux.mean(axis=0)

    def frame(k):
        fig = plt.figure(figsize=(13, 5.2))
        ax0 = fig.add_axes([0.0, 0.08, 0.30, 0.84])
        ax1 = fig.add_axes([0.35, 0.15, 0.30, 0.75])
        axr = [fig.add_axes([0.73, 0.68 - 0.26 * j, 0.25, 0.22]) for j in range(3)]
        C.draw_disk(ax0, snaps[k], "teff", "inferno", norm)
        C.disk_axes(ax0, lim=lim)
        ax0.text(0.5, 0.0, f"phase {ph[k]:.2f}   R = {R[k]:.1f} R$_\\odot$", transform=ax0.transAxes,
                 ha="center", color=t["muted"])
        ax1.plot(vel, mean_prof, color=t["obs"], lw=1.0, label="cycle mean")
        ax1.plot(vel, flux[k], color=t["accent"], lw=2.0, label="this phase")
        ax1.axvline(v[k], color=t["accent2"], lw=1.2, ls="--", label="line centroid")
        ax1.set_xlim(vel[0], vel[-1]); ax1.set_xlabel("Velocity [km/s]"); ax1.set_ylabel("Normalised flux")
        ax1.set_title("Fe I 6151.6 Å", loc="left", color=t["muted"]); ax1.legend(loc="lower left")
        for ax, y, lab, col in zip(axr, (R, T, v), ("R [R$_\\odot$]", "T$_\\mathrm{eff}$ [K]", "RV [km/s]"),
                                   (t["accent4"], t["accent"], t["accent2"])):
            yy, xx = np.append(y, y[0]), np.append(ph, 1.0)
            ax.plot(xx, yy, color=t["grid"], lw=1.4)
            ax.plot(xx[:k + 1], yy[:k + 1], color=col, lw=2)
            ax.plot([ph[k]], [y[k]], "o", color=col, ms=6)
            ax.set_xlim(0, 1); ax.set_ylabel(lab, fontsize=11); ax.tick_params(labelsize=9)
            if ax is not axr[-1]:
                ax.set_xticklabels([])
        axr[-1].set_xlabel("Pulsation phase", fontsize=11)
        return fig

    C.save_animation(frame, len(ph), out / f"cepheid_{theme}", fps=16, still_index=0)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
