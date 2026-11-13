"""Showcase 1 - "Spin it, see the lines".

A Sun-like star spun up from 0 to 100 km/s and back. Left: the visible disc coloured
by line-of-sight velocity (the star visibly turning, mesh drawn at the true rotation
phase). Right: the Mg b triplet from the v1.2.0 emulator broadening in step, with the
non-rotating spectrum as a ghost.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

WL = (5164.0, 5188.0)
VMAX = 100.0


def v_schedule(n):
    """0 -> VMAX (eased), hold, back to 0; returns per-frame v_eq."""
    up = n * 2 // 5
    hold = n - 2 * up
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, up))
    return np.concatenate([VMAX * s, np.full(hold, VMAX), VMAX * s[::-1]])


def compute(args, out):
    C.setup_jax()
    import jax.numpy as jnp
    from spice.models.mesh_transform import add_rotation, evaluate_rotation

    emu = C.load_emulator()
    n_frames = 12 if args.quick else 90
    n_vert = 500 if args.quick else 5000
    wl, lw = C.wl_grid(*WL, 400 if args.quick else 2400)
    params = C.stellar_params(emu, 5777, 4.44, 0.0, 1.0)
    base = C.make_star(params, emu.stellar_parameter_names, n_vertices=n_vert, inclination=90.0)
    vs = v_schedule(n_frames)

    # unique speeds only (the schedule is symmetric) -> spectra
    uniq, inv = np.unique(np.round(vs, 6), return_inverse=True)
    spec = {}
    for v in uniq:
        m = add_rotation(base, float(v), rotation_axis=jnp.array(C.Z_AXIS)) if v > 0 else base
        f, c = C.synth(emu, m, lw)
        spec[v] = f / c
        print(f"v={v:6.1f} done", flush=True)
    flux = np.stack([spec[uniq[i]] for i in inv])

    # rotation phase: integrate the angle so the spin speeds up with v
    ang = np.cumsum(vs) / VMAX * (2 * np.pi / 30)       # one turn per ~30 frames at full speed
    snaps = []
    spin = add_rotation(base, VMAX, rotation_axis=jnp.array(C.Z_AXIS))
    period = C.rotation_period_s(1.0, VMAX)
    for k in range(n_frames):
        m = evaluate_rotation(spin, ang[k] / (2 * np.pi) * period)
        vlos = np.asarray(m.los_velocities) * vs[k] / VMAX
        snaps.append(C.snapshot(m, {"vlos": vlos}))
    np.savez_compressed(out / "data.npz", wl=wl, flux=flux, v=vs, **C.pack_snapshots(snaps))


def render(out, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    snaps = C.unpack_snapshots(d)
    wl, flux, vs = d["wl"], d["flux"], d["v"]
    norm = Normalize(-VMAX, VMAX)
    cmap = "RdBu_r"

    def frame(k):
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_axes([0.02, 0.08, 0.36, 0.84])
        ax1 = fig.add_axes([0.46, 0.17, 0.51, 0.72])
        C.draw_disk(ax0, snaps[k], "vlos", cmap, norm)
        C.disk_axes(ax0)
        ax0.text(0.5, -0.02, f"$v_\\mathrm{{eq}}\\,\\sin i$ = {vs[k]:5.1f} km/s", transform=ax0.transAxes,
                 ha="center", va="top", fontsize=17, color=t["fg"])
        ax1.plot(wl, flux[0], color=t["obs"], lw=1.0, alpha=0.6, label="not rotating")
        ax1.plot(wl, flux[k], color=t["accent"], lw=2.0, label="rotating")
        ax1.set_xlim(wl[0], wl[-1]); ax1.set_ylim(0.0, 1.08)
        ax1.set_xlabel("Wavelength [Å]"); ax1.set_ylabel("Normalised flux")
        ax1.set_title("Mg b triplet, $T_\\mathrm{eff}$ 5777 K, log g 4.44", loc="left", color=t["muted"])
        ax1.legend(loc="lower left")
        return fig

    C.save_animation(frame, len(vs), out / f"rotation_{theme}", fps=20, still_index=len(vs) // 2)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
