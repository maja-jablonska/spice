"""Showcase 3 - an eclipsing binary: sky view, light curve and composite spectrum.

F + K dwarf pair, P = 2.5 d, i = 87 deg, synchronised aligned spins. Top: the two
meshes on the sky coloured by line-of-sight velocity (orbit + spin), with the
occlusion resolved per element. Bottom left: the Stromgren y light curve tracing out.
Bottom right: the composite Mg b spectrum, the two line sets splitting and merging
(the cooler component's lines are the deeper, narrower set).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

P_DAYS = 2.5
INCL_DEG = 87.0
STARS = [dict(teff=6400.0, logg=4.14, radius=1.6, mass=1.30),
         dict(teff=4900.0, logg=4.46, radius=0.9, mass=0.85)]
WL_LINES = (5164.0, 5190.0)
WL_PHOT = (5050.0, 5950.0)


def build(emu, n_vert):
    import jax.numpy as jnp
    from spice.models import IcosphereModel, Binary
    from spice.models.binary import add_orbit
    from spice.models.mesh_view import get_mesh_view
    los = jnp.array([0.0, 0.0, -1.0])
    bodies = []
    for s in STARS:
        p = C.stellar_params(emu, s["teff"], s["logg"], 0.0, 1.2)
        bodies.append(get_mesh_view(IcosphereModel.construct(n_vert, s["radius"], s["mass"], p,
                                                             emu.stellar_parameter_names), los))
    P = P_DAYS / 365.25
    orbit = dict(P=P, ecc=0.0, T=0.0, i=np.deg2rad(INCL_DEG), omega=0.0, Omega=0.0, mean_anomaly=0.0,
                 reference_time=0.0, vgamma=0.0, orbit_resolution_points=400)
    b = add_orbit(Binary.from_bodies(*bodies), **orbit)
    h = C.orbit_normal(b)
    spun = [C.add_aligned_rotation(m, 2 * np.pi * s["radius"] * 695700.0 / (P_DAYS * 86400.0), h)
            for m, s in zip(bodies, STARS)]
    b = add_orbit(Binary.from_bodies(*spun), **orbit)
    return b, P, h


def compute(args, out):
    C.setup_jax()
    import jax.numpy as jnp
    from spice.models.binary import evaluate_orbit_at_times
    from spice.spectrum import AB_passband_luminosity
    from spice.spectrum.filter import Stromgreny

    emu = C.load_emulator()
    n_vert = 500 if args.quick else 4000
    n_frames = 12 if args.quick else 160
    n_lc = 20 if args.quick else 480
    b, P, h = build(emu, n_vert)
    wl, lw = C.wl_grid(*WL_LINES, 300 if args.quick else 1600)
    wp, lwp = C.wl_grid(*WL_PHOT, 300 if args.quick else 1800)
    filt = Stromgreny()

    def phot(m1, m2):
        f = C.synth(emu, m1, lwp)[0] + C.synth(emu, m2, lwp)[0]
        return float(AB_passband_luminosity(filt, wp, f))

    # dense light curve (photometry only)
    t_lc = np.linspace(0, P, n_lc, endpoint=False)
    p1s, p2s = C.orbit_at_times(b, t_lc)
    lc = np.array([phot(m1, m2) for m1, m2 in zip(p1s, p2s)])
    print("light curve done", flush=True)

    # animation frames: spectra + sky snapshots (+ the frame's own magnitude)
    t_fr = np.linspace(0, P, n_frames, endpoint=False) + 0.25 * P / n_frames
    f1s, f2s = C.orbit_at_times(b, t_fr)
    spec1, spec2, cont1, cont2, snaps, mag = [], [], [], [], [], []
    for k, (m1, m2) in enumerate(zip(f1s, f2s)):
        a1, c1 = C.synth(emu, m1, lw); a2, c2 = C.synth(emu, m2, lw)
        spec1.append(a1); spec2.append(a2); cont1.append(c1); cont2.append(c2)
        mag.append(phot(m1, m2))
        s1 = C.snapshot(m1, {"vlos": m1.los_velocities}, up=h)
        s2 = C.snapshot(m2, {"vlos": m2.los_velocities}, up=h)
        snaps += [s1, s2]
        if k % 10 == 0:
            print(f"frame {k}/{n_frames}", flush=True)
    np.savez_compressed(out / "data.npz", wl=wl, t_lc=t_lc / P, lc=lc, t_fr=t_fr / P, mag=np.array(mag),
                        spec1=np.array(spec1), spec2=np.array(spec2), cont1=np.array(cont1),
                        cont2=np.array(cont2), h=h, **C.pack_snapshots(snaps))


def render(out, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    snaps = C.unpack_snapshots(d)
    wl, ph_lc, lc, ph, mag = d["wl"], d["t_lc"], d["lc"], d["t_fr"], d["mag"]
    s1, s2, c1, c2 = d["spec1"], d["spec2"], d["cont1"], d["cont2"]
    comp = (s1 + s2) / (c1 + c2)
    for sn in snaps:   # spin only: remove each star's own (area-weighted) orbital velocity
        if len(sn["vlos"]):
            sn["vspin"] = sn["vlos"] - np.average(sn["vlos"], weights=sn["area"])
    vmax = np.percentile(np.abs(np.concatenate([s["vspin"] for s in snaps if "vspin" in s])), 99)
    norm = Normalize(-vmax, vmax)
    allpoly = np.concatenate([s["poly"] for s in snaps])
    xlim = np.abs(allpoly[..., 0]).max() * 1.05
    ylim = max(np.abs(allpoly[..., 1]).max() * 1.3, 2.2)

    def frame(k):
        fig = plt.figure(figsize=(12, 7.2))
        ax0 = fig.add_axes([0.03, 0.56, 0.94, 0.42])
        ax1 = fig.add_axes([0.08, 0.09, 0.36, 0.38])
        ax2 = fig.add_axes([0.53, 0.09, 0.44, 0.38])
        C.draw_disk(ax0, [sn for sn in (snaps[2 * k], snaps[2 * k + 1]) if "vspin" in sn], "vspin", "RdBu_r", norm)
        ax0.set_xlim(-xlim, xlim); ax0.set_ylim(-ylim, ylim); ax0.set_aspect("equal"); ax0.axis("off")
        ax0.text(0.0, 0.92, f"orbital phase {ph[k]:.3f}", transform=ax0.transAxes, color=t["muted"])
        ax0.text(1.0, 0.92, f"colour: spin velocity, ±{vmax:.0f} km/s", transform=ax0.transAxes, ha="right", color=t["muted"], fontsize=11)
        ax1.plot(ph_lc, lc, color=t["grid"], lw=1.5)
        sel = ph_lc <= ph[k]
        ax1.plot(ph_lc[sel], lc[sel], color=t["accent2"], lw=2)
        ax1.plot([ph[k]], [mag[k]], "o", color=t["accent"], ms=7)
        ax1.set_ylim(lc.max() + 0.05 * np.ptp(lc), lc.min() - 0.05 * np.ptp(lc))
        ax1.set_xlim(0, 1); ax1.set_xlabel("Orbital phase"); ax1.set_ylabel("Strömgren y [AB mag, 10 pc]")
        ax2.plot(wl, comp[0], color=t["obs"], lw=1.0, alpha=0.6, label=f"phase {ph[0]:.2f}")
        ax2.plot(wl, comp[k], color=t["accent"], lw=1.8, label="now")
        ax2.legend(loc="lower left", fontsize=11)
        ax2.set_xlim(wl[0], wl[-1]); ax2.set_ylim(0.15, 1.05)
        ax2.set_xlabel("Wavelength [Å]"); ax2.set_ylabel("Normalised flux")
        ax2.set_title("Mg b, F + K dwarf", loc="left", color=t["muted"])
        return fig

    C.save_animation(frame, len(ph), out / f"eclipse_{theme}", fps=20, still_index=0)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
