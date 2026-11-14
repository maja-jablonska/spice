"""Shared helpers for the SPICE showcase animations (website / talks).

Every showcase script splits into ``compute`` (JAX, heavy, writes ``data.npz``)
and ``render`` (matplotlib only, reads the npz, writes GIF + still PNG in a dark
and a light theme), so the figures can be restyled without re-running synthesis.

Geometry convention used throughout: the rotation axis is +z and the line of sight
comes from :func:`spice.models.utils.inclination_to_los_axis`, so ``inclination``
really is the angle between the spin axis and the line of sight and spot latitudes
from ``lat_to_theta`` are true latitudes (see the 2026-10 Kochukhov geometry trap:
mixing the default y axis with these helpers silently gives the wrong inclination).
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np

HF_BUNDLE = "mjablonska/TPayne-spice-harps-interp"   # v1.2.0 on main
EMULATOR_PARAMS = ["teff", "logg", "feh", "vmicro", "a", "c", "n", "o", "r", "s"]
Z_AXIS = (0.0, 0.0, 1.0)
C_KMS = 299792.458


# --------------------------------------------------------------------------- CLI
def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--quick", action="store_true", help="tiny mesh / few frames smoke test")
    p.add_argument("--render-only", action="store_true", help="re-render from an existing data.npz")
    p.add_argument("--theme", default="both", choices=["dark", "light", "both"])
    return p


def themes(arg: str):
    return ["dark", "light"] if arg == "both" else [arg]


def setup_jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


# --------------------------------------------------------------------------- emulator
def load_emulator():
    """TPayne-spice-harps-interp v1.2.0 (11 inputs incl. mu). Gadi jobs point
    ``SPICE_HARPS_BUNDLE`` at the released bundle directory; otherwise the HF id."""
    from spice.spectrum import IntensityPretrainedAemuSpectrumEmulator
    name = os.environ.get("SPICE_HARPS_BUNDLE", HF_BUNDLE)
    emu = IntensityPretrainedAemuSpectrumEmulator(name)
    names = list(emu.stellar_parameter_names)
    if names != EMULATOR_PARAMS:
        raise RuntimeError(f"unexpected emulator contract {names}; showcase expects {EMULATOR_PARAMS} "
                           "(the old RozanskiT/TPayne-spice-harps bundle is pre-C/N/O-fix, do not use it)")
    print(f"emulator: {name}", flush=True)
    return emu


def stellar_params(emu, teff, logg, feh=0.0, vmicro=1.5, **abund):
    d = dict(teff=float(teff), logg=float(logg), feh=float(feh), vmicro=float(vmicro))
    d.update({k: float(v) for k, v in abund.items()})
    return emu.to_parameters(d)


# --------------------------------------------------------------------------- geometry
def make_star(params, names, n_vertices=5000, radius=1.0, mass=1.0, inclination=90.0,
              vrot=0.0, los=None):
    """Icosphere star, spin axis +z, seen at ``inclination`` (deg), equatorial
    velocity ``vrot`` km/s. Returns the mesh with the view set."""
    import jax.numpy as jnp
    from spice.models import IcosphereModel
    from spice.models.mesh_transform import add_rotation
    from spice.models.mesh_view import get_mesh_view
    from spice.models.utils import inclination_to_los_axis

    m = IcosphereModel.construct(n_vertices, radius, mass, params, names)
    los = inclination_to_los_axis(inclination) if los is None else jnp.asarray(los)
    m = get_mesh_view(m, los)
    if vrot:
        m = add_rotation(m, vrot, rotation_axis=jnp.array(Z_AXIS))
    return m


def rotation_period_s(radius_rsun, vrot_kms):
    return 2 * np.pi * radius_rsun * 695700.0 / vrot_kms


def wl_grid(w0, w1, n):
    w = np.linspace(w0, w1, n)
    return w, np.log10(w)


def synth(emu, mesh, log_wl, chunk_size=1024, **kw):
    from spice.spectrum import simulate_observed_flux
    f = np.asarray(simulate_observed_flux(emu.intensity, mesh, log_wl, chunk_size=chunk_size, **kw))
    return f[:, 0], f[:, 1]


def spin_vector(mesh):
    """Unit angular-velocity vector actually realised by the mesh velocity field
    (<r x v> = 2/3 R^2 omega for rigid rotation) -- used to check prograde spin
    instead of trusting the axis-sign convention."""
    r = np.asarray(mesh.d_centers)
    v = np.asarray(mesh.rotation_velocities)
    w = np.cross(r, v).mean(axis=0)
    return w / np.linalg.norm(w)


def orbit_normal(binary):
    """Unit orbital angular-momentum direction of body2 about body1."""
    r = np.asarray(binary.body2_centers) - np.asarray(binary.body1_centers)
    v = np.asarray(binary.body2_velocities) - np.asarray(binary.body1_velocities)
    h = np.cross(r, v).mean(axis=0)
    return h / np.linalg.norm(h)


def orbit_at_times(binary, times, batch=8):
    """evaluate_orbit_at_times in small batches: the vmapped occlusion allocates
    (n_times x n_elements x n_neighbours x ...) at once and runs out of GPU memory
    for long time series on fine meshes."""
    import jax.numpy as jnp
    from spice.models.binary import evaluate_orbit_at_times
    times = np.asarray(times)
    m1s, m2s = [], []
    for i in range(0, len(times), batch):
        a, b = evaluate_orbit_at_times(binary, jnp.asarray(times[i:i + batch]))
        m1s += list(a); m2s += list(b)
    return m1s, m2s


def add_aligned_rotation(mesh, vrot, normal):
    """Rigid rotation with angular velocity along ``normal`` (prograde, aligned)."""
    import jax.numpy as jnp
    from spice.models.mesh_transform import add_rotation
    for sgn in (1.0, -1.0):
        m = add_rotation(mesh, vrot, rotation_axis=jnp.asarray(sgn * np.asarray(normal)))
        if np.dot(spin_vector(m), normal) > 0.99:
            return m
    raise RuntimeError("could not realise a prograde spin along the orbit normal")


# --------------------------------------------------------------------------- mesh snapshots
def _sky_basis(los, up=Z_AXIS):
    n = np.asarray(los, float); n = n / np.linalg.norm(n)
    u = np.asarray(up, float); u = u - u.dot(n) * n
    if np.linalg.norm(u) < 1e-6:
        u = np.array([0.0, 1.0, 0.0]) - n[1] * n
    u = u / np.linalg.norm(u)
    r = np.cross(n, u)
    return n, r, u


def snapshot(mesh, fields: dict, up=Z_AXIS, los=None):
    """Sky-plane polygons of the visible half of ``mesh`` + per-face fields.

    ``fields`` maps a name to a per-face array (or a callable taking the mesh).
    Returns a dict of numpy arrays: ``poly`` (n_vis, 3, 2) in R_sun, ``depth``,
    ``mu``, ``area`` (visible cast area) and one entry per field.
    Faces with zero visible area (back side or fully occluded) are dropped.
    """
    los = np.asarray(mesh.los_vector if los is None else los)
    n, r, u = _sky_basis(los, up)
    tri = np.asarray(mesh.mesh_elements)                  # (F, 3, 3)
    vis = np.asarray(mesh.visible_cast_areas)
    keep = vis > 0
    poly = np.stack([tri[keep] @ r, tri[keep] @ u], axis=-1)
    out = dict(poly=poly.astype(np.float32),
               depth=(np.asarray(mesh.centers)[keep] @ n).astype(np.float32),
               mu=np.asarray(mesh.mus)[keep].astype(np.float32),
               area=vis[keep].astype(np.float32))
    for k, v in fields.items():
        val = v(mesh) if callable(v) else v
        out[k] = np.asarray(val)[keep].astype(np.float32)
    return out


def pack_snapshots(snaps, prefix="snap"):
    """Ragged list of snapshot dicts -> flat npz-able arrays with offsets."""
    keys = snaps[0].keys()
    off = np.cumsum([0] + [len(s["mu"]) for s in snaps])
    out = {f"{prefix}_offsets": off}
    for k in keys:
        out[f"{prefix}_{k}"] = np.concatenate([s[k] for s in snaps])
    return out


def unpack_snapshots(d, prefix="snap"):
    off = d[f"{prefix}_offsets"]
    keys = [k[len(prefix) + 1:] for k in d.files if k.startswith(prefix + "_") and k != f"{prefix}_offsets"]
    return [{k: d[f"{prefix}_{k}"][off[i]:off[i + 1]] for k in keys} for i in range(len(off) - 1)]


# --------------------------------------------------------------------------- style
THEMES = {  # dataviz reference palette: slots 1-3 (blue, orange, aqua) validate all-pairs in both modes
    "dark": dict(bg="#1a1a19", panel="#1a1a19", fg="#f0efec", muted="#c3c2b7", grid="#3a3a37",
                 accent="#d95926", accent2="#3987e5", accent4="#199e70", accent3="#e66767",
                 line="#f0efec", obs="#8d8c86"),
    "light": dict(bg="#fcfcfb", panel="#fcfcfb", fg="#0b0b0b", muted="#52514e", grid="#e2e1dc",
                  accent="#eb6834", accent2="#2a78d6", accent4="#1baf7a", accent3="#e34948",
                  line="#0b0b0b", obs="#a3a29d"),
}


def apply_theme(theme: str):
    import matplotlib as mpl
    t = THEMES[theme]
    mpl.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["panel"], "savefig.facecolor": t["bg"],
        "axes.edgecolor": t["muted"], "axes.labelcolor": t["fg"], "text.color": t["fg"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.grid": False,
        "grid.color": t["grid"], "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    })
    return t


def draw_disk(ax, snaps, field, cmap, norm, shade=True, edge=True, alpha=1.0):
    """Painter's-algorithm draw of one or more snapshots (far faces first).

    ``shade`` darkens towards the limb (cheap linear limb darkening for looks only;
    the spectra themselves use the emulator's own mu dependence)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    if isinstance(snaps, dict):
        snaps = [snaps]
    poly = np.concatenate([s["poly"] for s in snaps])
    depth = np.concatenate([s["depth"] for s in snaps])
    mu = np.concatenate([s["mu"] for s in snaps])
    val = np.concatenate([s[field] for s in snaps])
    order = np.argsort(-depth)                          # far (large depth) first
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    rgba = cm(norm(val[order]))
    if shade:
        k = (0.35 + 0.65 * np.clip(mu[order], 0, 1))[:, None]
        rgba[:, :3] = rgba[:, :3] * k
    rgba[:, 3] = alpha
    pc = PolyCollection(poly[order], facecolors=rgba, edgecolors=rgba if edge else "none",
                        linewidths=0.3, antialiased=True)
    ax.add_collection(pc)
    return pc


def disk_axes(ax, lim=1.15, center=(0.0, 0.0)):
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------------------------------- output
def fig_to_image(fig, dpi):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_animation(frames_fn, n_frames, path_stem: Path, fps=20, dpi=100, still_index=None):
    """``frames_fn(i) -> matplotlib Figure``; writes ``<stem>.gif`` and ``<stem>.png``
    (the still at ``still_index``, default middle frame, at 2x dpi)."""
    import matplotlib.pyplot as plt
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    still_index = n_frames // 2 if still_index is None else still_index
    imgs = []
    for i in range(n_frames):
        fig = frames_fn(i)
        imgs.append(fig_to_image(fig, dpi))
        if i == still_index:
            fig.savefig(path_stem.with_suffix(".png"), dpi=2 * dpi)
        plt.close(fig)
    # One shared adaptive palette keeps colours stable across frames (no flicker).
    pal = imgs[len(imgs) // 2].quantize(colors=255, method=0, dither=0)
    q = [im.quantize(palette=pal, dither=0) for im in imgs]
    q[0].save(path_stem.with_suffix(".gif"), save_all=True, append_images=q[1:],
              duration=int(1000 / fps), loop=0, optimize=True, disposal=1)
    print(f"wrote {path_stem}.gif ({n_frames} frames) and .png", flush=True)


def save_still(fig, path_stem: Path, dpi=200):
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(path_stem.with_suffix(".pdf"))
    print(f"wrote {path_stem}.png/.pdf", flush=True)


def outdir(args) -> Path:
    p = Path(args.out)
    p.mkdir(parents=True, exist_ok=True)
    return p
