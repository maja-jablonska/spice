"""Showcase 11 - solar-like oscillations.

A Sun (v_eq = 2 km/s, equator-on) carrying a handful of non-radial p modes
(l = 0..5, periods 4.5-5.8 min). Left: the surface velocity pattern beating as the
modes drift in and out of phase. Middle: the Fe I 6173 A line minus its time
average. Right: the disc-integrated RV -- low-degree modes survive disc integration,
high-degree ones mostly cancel, which is why Sun-as-a-star data see only l <= 3.
Caveat: the local surface velocities (~0.6 km/s peak) are solar-like, but the Sun spreads
that power over millions of modes; here it sits in seven low-degree ones, so the
disc-integrated RV (~100 m/s) is roughly 50x the Sun's few m/s.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LINE = 6173.334
WL = (6172.35, 6174.35)
MODES = [  # (l, m, period [min], relative amplitude, phase)
    (0, 0, 5.40, 0.6, 0.0),
    (1, 1, 5.10, 0.8, 1.0),
    (2, 0, 4.85, 0.7, 2.1),
    (2, 2, 5.65, 0.6, 0.4),
    (3, 1, 4.60, 0.5, 2.8),
    (4, 3, 5.25, 0.6, 1.7),
    (5, 4, 4.95, 0.7, 0.9),
]
VPEAK = 0.6     # km/s, target peak surface velocity


def build(base, scale):
    import jax.numpy as jnp
    from spice.models.mesh_transform import add_pulsation
    m = base
    for l, mm, per, amp, phi in MODES:
        fs = jnp.array([[amp * scale, phi]])
        horiz = fs.at[:, 0].multiply(0.3) if l > 0 else jnp.zeros_like(fs)    # some horizontal motion
        fs = jnp.stack([fs, horiz, jnp.zeros_like(fs)])
        m = add_pulsation(m, m_order=mm, l_degree=l, period=per / 1440.0, fourier_series_parameters=fs)
    return m


def compute(args, out):
    C.setup_jax()
    from spice.models import IcosphereModel
    from spice.models.mesh_transform import add_rotation, evaluate_pulsations
    from spice.models.mesh_view import get_mesh_view
    from spice.models.utils import inclination_to_los_axis
    import jax.numpy as jnp

    emu = C.load_emulator()
    n_vert = 800 if args.quick else 20000
    p = C.stellar_params(emu, 5777, 4.44, 0.0, 1.0)
    base = IcosphereModel.construct(n_vert, 1.0, 1.0, p, emu.stellar_parameter_names, max_pulsation_mode=8)
    base = get_mesh_view(base, inclination_to_los_axis(90.0))
    base = add_rotation(base, 2.0, rotation_axis=jnp.array(C.Z_AXIS))
    # calibrate the amplitude unit: peak |v_puls| over a few probe times -> VPEAK
    probe = build(base, 1e-5)
    vmax = max(float(np.abs(np.asarray(evaluate_pulsations(probe, t).pulsation_velocities)).max())
               for t in np.linspace(0, 10 / 1440.0, 7))
    scale = 1e-5 * VPEAK / vmax
    puls = build(base, scale)
    print(f"amplitude scale {scale:.3e} (peak v at 1e-5 was {vmax:.4f} km/s)", flush=True)

    n_fr = 10 if args.quick else 150
    t_min = np.linspace(0, 45, n_fr)                 # 45 minutes, 18 s cadence
    wl, lw = C.wl_grid(*WL, 160 if args.quick else 500)
    flux, rv, snaps = [], [], []
    for k, tm in enumerate(t_min):
        m = evaluate_pulsations(puls, tm / 1440.0)
        f, c = C.synth(emu, m, lw)
        flux.append(f / c)
        vlos = np.asarray(m.los_velocities)
        vis = np.asarray(m.visible_cast_areas)
        rv.append(float(np.sum(vlos * vis) / np.sum(vis)))
        vpuls = np.asarray(m.pulsation_velocities) @ np.asarray(m.los_vector)   # = pulsation part of los_velocities
        snaps.append(C.snapshot(m, {"vpuls": vpuls}))
        if k % 25 == 0:
            print(f"t = {tm:.1f} min", flush=True)
    np.savez_compressed(out / "data.npz", wl=wl, flux=np.array(flux), rv=np.array(rv), t_min=t_min,
                        **C.pack_snapshots(snaps))


def render(out, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    snaps = C.unpack_snapshots(d)
    wl, flux, rv, tm = d["wl"], d["flux"], d["rv"], d["t_min"]
    vel = (wl / LINE - 1) * C.C_KMS
    resid = flux - flux.mean(axis=0)
    rlim = np.abs(resid).max() * 1.1
    vall = np.concatenate([s["vpuls"] for s in snaps])
    vl = np.percentile(np.abs(vall), 99)
    norm = Normalize(-vl, vl)
    rvm = 1e3 * (rv - rv.mean())

    def frame(k):
        fig = plt.figure(figsize=(13, 5))
        ax0 = fig.add_axes([0.0, 0.06, 0.32, 0.86])
        ax1 = fig.add_axes([0.37, 0.15, 0.27, 0.73])
        ax2 = fig.add_axes([0.72, 0.15, 0.26, 0.73])
        C.draw_disk(ax0, snaps[k], "vpuls", "RdBu_r", norm, shade=True)
        C.disk_axes(ax0)
        ax0.text(0.5, 0.0, f"t = {tm[k]:4.1f} min", transform=ax0.transAxes, ha="center", color=t["muted"])
        ax1.axhline(0, color=t["grid"], lw=1)
        ax1.plot(vel, resid[k], color=t["accent"], lw=2)
        ax1.set_ylim(-rlim, rlim); ax1.set_xlim(vel[0], vel[-1])
        ax1.set_xlabel("Velocity [km/s]"); ax1.set_ylabel("Flux − time average")
        ax1.set_title("Fe I 6173 Å", loc="left", color=t["muted"])
        ax2.plot(tm, rvm, color=t["grid"], lw=1.2)
        ax2.plot(tm[:k + 1], rvm[:k + 1], color=t["accent2"], lw=2)
        ax2.plot([tm[k]], [rvm[k]], "o", color=t["accent2"], ms=7)
        ax2.set_xlim(tm[0], tm[-1]); ax2.set_xlabel("Time [min]"); ax2.set_ylabel("Disc-integrated RV [m/s]")
        fig.text(0.37, 0.95, "7 low-degree modes carry all the power: disc-integrated RV ~50× the Sun's", color=t["muted"], fontsize=11)
        return fig

    C.save_animation(frame, len(tm), out / f"solar_oscillations_{theme}", fps=15, still_index=len(tm) // 3)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
