"""Showcase 10 - a planetary transit and the Rossiter-McLaughlin effect.

A hot Jupiter (R = 0.13 R_sun, P = 3 d, b ~ 0.3) crosses a rotating F dwarf
(v_eq = 30 km/s), once on an aligned orbit and once with a projected obliquity of
60 deg. The planet hides first the approaching and then the receding limb (or, when
misaligned, mostly one of them), so the measured radial velocity jumps around the
Keplerian curve and a "Doppler shadow" crosses the line profile. Planet flux is
ignored (night side); the star's spectrum is synthesised with per-element occlusion.
RVs come from a template fit (shift of the unocculted spectrum) over the Mg b window.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

STAR = dict(teff=6300.0, logg=4.3, radius=1.3, mass=1.25, veq=30.0)
PLANET = dict(radius=0.13, mass=0.0012)
P_DAYS, INCL_DEG = 3.0, 88.0
LAMBDAS = [0.0, 60.0]
WL = (5164.0, 5188.0)
LINES = [5167.322, 5172.684, 5183.604]   # Mg I b


def rodrigues(v, axis, ang):
    k = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * np.dot(k, v) * (1 - np.cos(ang))


def template_rv(wl, f, tmpl, vgrid):
    chi = np.array([np.sum((f - np.interp(wl / (1 + v / C.C_KMS), wl, tmpl)) ** 2) for v in vgrid])
    j = int(np.clip(np.argmin(chi), 1, len(vgrid) - 2))
    a, b, c = chi[j - 1], chi[j], chi[j + 1]
    return vgrid[j] + 0.5 * (a - c) / (a - 2 * b + c) * (vgrid[1] - vgrid[0])


def lsd_profile(wl, resid, vel):
    """Mean residual in velocity space over the Mg b lines (a poor man's LSD)."""
    return np.mean([np.interp(lam * (1 + vel / C.C_KMS), wl, resid) for lam in LINES], axis=0)


def run(emu, lam_deg, args):
    import jax.numpy as jnp
    from spice.models import IcosphereModel, Binary, find_binary_eclipses
    from spice.models.binary import add_orbit, evaluate_orbit_at_times, evaluate_orbit
    from spice.models.mesh_view import get_mesh_view
    los = np.array([0.0, 0.0, -1.0])
    n_vert = 600 if args.quick else 8000
    p = C.stellar_params(emu, STAR["teff"], STAR["logg"], 0.0, 1.3)
    star = get_mesh_view(IcosphereModel.construct(n_vert, STAR["radius"], STAR["mass"], p,
                                                  emu.stellar_parameter_names), jnp.asarray(los))
    planet = get_mesh_view(IcosphereModel.construct(200 if args.quick else 600, PLANET["radius"], PLANET["mass"], p,
                                                    emu.stellar_parameter_names), jnp.asarray(los))
    P = P_DAYS / 365.25
    kw = dict(P=P, ecc=0.0, T=0.0, i=np.deg2rad(INCL_DEG), omega=0.0, Omega=0.0, mean_anomaly=0.0,
              reference_time=0.0, vgamma=0.0, orbit_resolution_points=600)
    h = C.orbit_normal(add_orbit(Binary.from_bodies(star, planet), **kw))
    spin = rodrigues(h, los, np.deg2rad(lam_deg))
    star_s = C.add_aligned_rotation(star, STAR["veq"], spin)
    b = add_orbit(Binary.from_bodies(star_s, planet), **kw)
    # the transit = the eclipse with the planet in front
    win = None
    for e in find_binary_eclipses(b):
        m1, m2 = evaluate_orbit(b, e["mid"])
        if float(np.asarray(m2.center) @ los) < float(np.asarray(m1.center) @ los):
            win = (float(e["T1"]), float(e["T4"]))
    assert win is not None, "no transit found"
    dur = win[1] - win[0]
    n_fr = 10 if args.quick else 100
    times = np.linspace(win[0] - 0.35 * dur, win[1] + 0.35 * dur, n_fr)
    s_list, p_list = C.orbit_at_times(b, times)
    wl, lw = C.wl_grid(*WL, 400 if args.quick else 2400)
    ft, ct = C.synth(emu, star_s, lw)           # unocculted, at rest: the template
    tmpl = ft / ct
    vgrid = np.linspace(-1.5, 1.5, 61)
    vel = np.linspace(-60, 60, 241)
    rv, shadow, snaps = [], [], []
    for k, (ms, mp) in enumerate(zip(s_list, p_list)):
        f, c = C.synth(emu, ms, lw)
        nf = f / c
        rv.append(template_rv(wl, nf, tmpl, vgrid))
        shadow.append(lsd_profile(wl, nf - tmpl, vel))
        snaps.append(C.snapshot(ms, {"vlos": ms.los_velocities}, up=h))
        sp = C.snapshot(mp, {"vlos": np.zeros(len(np.asarray(mp.mus)))}, up=h)
        snaps.append(sp)
        if k % 20 == 0:
            print(f"lambda {lam_deg}: frame {k}/{n_fr}", flush=True)
    t_mid = 0.5 * (win[0] + win[1])
    return dict(t_hours=(times - t_mid) * 365.25 * 24, rv=np.array(rv), shadow=np.array(shadow), vel=vel,
                contact_hours=(np.array(win) - t_mid) * 365.25 * 24, snaps=snaps)


def compute(args, out):
    C.setup_jax()
    emu = C.load_emulator()
    save = {}
    allsnaps = []
    for j, lam in enumerate(LAMBDAS):
        r = run(emu, lam, args)
        allsnaps += r.pop("snaps")
        for k, v in r.items():
            save[f"l{j}_{k}"] = v
    np.savez_compressed(out / "data.npz", lambdas=np.array(LAMBDAS), **save, **C.pack_snapshots(allsnaps))


def render(out, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    lams = d["lambdas"]
    snaps = C.unpack_snapshots(d)
    n_fr = len(d["l0_t_hours"])
    norm = Normalize(-STAR["veq"], STAR["veq"])
    sh_lim = np.percentile(np.abs(np.concatenate([d[f"l{j}_shadow"] for j in range(len(lams))])), 99.5)
    cols = [t["accent2"], t["accent"]]

    def frame(k):
        fig = plt.figure(figsize=(13, 8))
        axrv = fig.add_axes([0.08, 0.08, 0.88, 0.25])
        for j, lam in enumerate(lams):
            x0 = 0.03 + 0.5 * j
            axd = fig.add_axes([x0, 0.43, 0.2, 0.5])
            axs = fig.add_axes([x0 + 0.25, 0.45, 0.2, 0.45])
            i0 = 2 * (j * n_fr + k)
            st, pl = snaps[i0], snaps[i0 + 1]
            C.draw_disk(axd, st, "vlos", "RdBu_r", norm)
            if len(pl["poly"]):
                from matplotlib.collections import PolyCollection
                front = pl["depth"].mean() < st["depth"].mean() if len(st["depth"]) else True
                if front:
                    axd.add_collection(PolyCollection(pl["poly"], facecolors="#000000", edgecolors="#000000", lw=0.3))
            cx = st["poly"][..., 0].mean(); cy = st["poly"][..., 1].mean()
            C.disk_axes(axd, lim=STAR["radius"] * 1.12, center=(cx, cy))
            axd.set_title(f"projected obliquity λ = {lam:.0f}°", color=t["fg"], fontsize=13)
            th = d[f"l{j}_t_hours"]; sh = d[f"l{j}_shadow"]; vel = d[f"l{j}_vel"]
            img = np.full_like(sh, np.nan); img[:k + 1] = sh[:k + 1]
            axs.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-sh_lim, vmax=sh_lim,
                       extent=[vel[0], vel[-1], th[0], th[-1]], interpolation="nearest")
            axs.set_facecolor(t["panel"])
            for s in (-1, 1):
                axs.axvline(s * STAR["veq"], color=t["muted"], lw=0.8, ls=":")
            axs.set_xlabel("Velocity [km/s]", fontsize=11); axs.set_ylabel("Hours from mid-transit", fontsize=11)
            axs.set_title("Doppler shadow", loc="left", color=t["muted"], fontsize=11)
            rv = 1e3 * (d[f"l{j}_rv"])
            axrv.plot(th, rv, color=t["grid"], lw=1.2)
            axrv.plot(th[:k + 1], rv[:k + 1], color=cols[j], lw=2.2, label=f"λ = {lam:.0f}°")
            axrv.plot([th[k]], [rv[k]], "o", color=cols[j], ms=7)
        for c in d["l0_contact_hours"]:
            axrv.axvline(c, color=t["muted"], lw=0.8, ls="--")
        axrv.set_xlim(d["l0_t_hours"][0], d["l0_t_hours"][-1])
        axrv.set_xlabel("Hours from mid-transit"); axrv.set_ylabel("Radial velocity [m/s]")
        axrv.legend(loc="upper right", ncol=2)
        return fig

    C.save_animation(frame, n_fr, out / f"rossiter_mclaughlin_{theme}", fps=15, still_index=n_fr // 2)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
