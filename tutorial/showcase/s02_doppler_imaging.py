"""Showcase 2 - Doppler imaging in reverse.

Two cool spots on a rotating K-ish dwarf (i = 60 deg, v_eq = 35 km/s). As the star
turns, each spot removes light from one velocity bin of the line profile, so a
"bump" travels across the Fe I 6173 A line. Panels: the disc (Teff map), the line
profile against the spot-free profile, and the residual dynamic spectrum filling in
phase by phase -- the forward version of a Doppler-imaging plot.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LINE = 6173.334          # Fe I (the SDO/HMI line), isolated in a solar-type spectrum
WL = (6171.55, 6174.85)
INCL, VEQ, RADIUS = 60.0, 35.0, 1.0
TEFF = 5200.0
SPOTS = [  # (latitude deg, longitude deg, radius deg, delta T)
    (40.0, 0.0, 22.0, -1500.0),
    (-15.0, 150.0, 16.0, -1300.0),
]


def compute(args, out):
    C.setup_jax()
    import jax.numpy as jnp
    from spice.models.mesh_transform import evaluate_rotation
    from spice.models.spots import add_spot
    from spice.models.utils import lat_to_theta, lon_to_phi

    emu = C.load_emulator()
    n_frames = 10 if args.quick else 72
    n_vert = 800 if args.quick else 10000
    wl, lw = C.wl_grid(*WL, 200 if args.quick else 700)
    p = C.stellar_params(emu, TEFF, 4.5, 0.0, 1.0)
    clean = C.make_star(p, emu.stellar_parameter_names, n_vertices=n_vert, radius=RADIUS,
                        inclination=INCL, vrot=VEQ)
    spotted = clean
    for lat, lon, r, dT in SPOTS:
        spotted = add_spot(spotted, lat_to_theta(lat), lon_to_phi(lon), r, dT, 0, smoothness=0.3)

    f0, c0 = C.synth(emu, clean, lw)
    ref = f0 / c0
    period = C.rotation_period_s(RADIUS, VEQ)
    phases = np.arange(n_frames) / n_frames
    flux, snaps = [], []
    for k, ph in enumerate(phases):
        m = evaluate_rotation(spotted, ph * period)
        f, c = C.synth(emu, m, lw)
        flux.append(f / c)          # continuum-normalised, as an observer would
        snaps.append(C.snapshot(m, {"teff": np.asarray(m.parameters)[:, 0]}))
        print(f"phase {ph:.3f} done", flush=True)
    np.savez_compressed(out / "data.npz", wl=wl, ref=ref, flux=np.array(flux), phase=phases,
                        **C.pack_snapshots(snaps))


def render(out, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    snaps = C.unpack_snapshots(d)
    wl, ref, flux, ph = d["wl"], d["ref"], d["flux"], d["phase"]
    vel = (wl / LINE - 1) * C.C_KMS
    resid = flux - ref
    lim = np.percentile(np.abs(resid), 99.5)
    norm = Normalize(TEFF - 1600, TEFF + 100)
    vsini = VEQ * np.sin(np.deg2rad(INCL))

    def frame(k):
        fig = plt.figure(figsize=(13, 5.2))
        ax0 = fig.add_axes([0.0, 0.06, 0.33, 0.86])
        ax1 = fig.add_axes([0.38, 0.15, 0.30, 0.75])
        ax2 = fig.add_axes([0.74, 0.15, 0.22, 0.75])
        C.draw_disk(ax0, snaps[k], "teff", "inferno", norm)
        C.disk_axes(ax0)
        ax0.text(0.5, 0.0, f"rotation phase {ph[k]:.2f}", transform=ax0.transAxes, ha="center",
                 color=t["muted"])
        ax1.plot(vel, ref, color=t["obs"], lw=1.2, label="no spots")
        ax1.plot(vel, flux[k], color=t["accent"], lw=2.0, label="spotted")
        for s in (-1, 1):
            ax1.axvline(s * vsini, color=t["grid"], lw=1, zorder=0)
        ax1.set_xlim(vel[0], vel[-1])
        ax1.set_xlabel("Velocity [km/s]"); ax1.set_ylabel("Normalised flux")
        ax1.set_title("Fe I 6173 Å", loc="left", color=t["muted"])
        ax1.legend(loc="lower left")
        img = np.full_like(resid, np.nan)
        img[:k + 1] = resid[:k + 1]
        ax2.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[vel[0], vel[-1], 0, 1], interpolation="nearest")
        ax2.set_facecolor(t["panel"])
        ax2.set_xlabel("Velocity [km/s]"); ax2.set_ylabel("Phase")
        ax2.set_title("spotted − clean", loc="left", color=t["muted"])
        return fig

    C.save_animation(frame, len(ph), out / f"doppler_imaging_{theme}", fps=15, still_index=len(ph) - 1)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
