"""Render a handful of pulsation animations that contrast different VSH
components and (l, m) modes.

Outputs (written next to this file):
  - pulsation_vsh_components.gif   Radial vs Spheroidal vs Toroidal at (l=2, m=1),
                                   mollweide projection
  - pulsation_lm_modes.gif         Four radial modes: zonal/tesseral/sectoral,
                                   mollweide projection
  - pulsation_observer_disk.gif    Tilted-inclination observer view of a
                                   sectoral (l=m=3) mode on the star's disk
  - pulsation_3d_vsh.gif           Same VSH-basis comparison rendered as a
                                   deforming 3D mesh
  - pulsation_3d_lm_modes.gif      Same (l, m) grid rendered as deforming 3D
                                   meshes

Run from the repo root with the spice src path on PYTHONPATH, e.g.::

    PYTHONPATH=src python tutorial/pulsation_animations.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

# Ensure the in-tree package is importable without an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# spice.__init__ also does this, but we set it before the first jax import in
# case this module is ever run as a plain script.
if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_pulsation, evaluate_pulsations
from spice.plots import animate_observed_disk, compute_pulsation_scalar
from spice.plots.plot_mesh import _mesh_center, smart_save


OUT_DIR = Path(__file__).resolve().parent
PERIOD = 0.1  # days
N_FRAMES = 48
FPS = 15
N_VERTICES = 1000  # low enough for quick JIT, high enough for smooth maps

# 3D renders are slower per frame; drop the mesh resolution + frame count.
N_FRAMES_3D = 36
FPS_3D = 12
N_VERTICES_3D = 642


def vsh_fourier(amp_r: float = 0.0, amp_s: float = 0.0, amp_t: float = 0.0,
                phase: float = 0.0) -> jnp.ndarray:
    """Single-term Fourier array for the three VSH components."""
    return jnp.array([
        [[amp_r, phase]],
        [[amp_s, phase]],
        [[amp_t, phase]],
    ])


def build_base_mesh(n_vertices: int = N_VERTICES) -> "MeshModel":
    return IcosphereModel.construct(
        n_vertices=n_vertices,
        radius=1.0,
        mass=1.0,
        parameters=jnp.array([5700.0, 4.4]),
        parameter_names=["teff", "log_g"],
        max_pulsation_mode=5,
        max_fourier_order=3,
    )


def evaluate_sequence(m_puls, phases: np.ndarray) -> List["MeshModel"]:
    return [evaluate_pulsations(m_puls, t=float(ph) * PERIOD) for ph in phases]


# ---------------------------------------------------------------------------
# Shared multi-panel mollweide animation
# ---------------------------------------------------------------------------

def _lonlat(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return phi, 0.5 * np.pi - theta


def animate_mollweide_panels(panel_frames: Sequence[Sequence["MeshModel"]],
                             panel_titles: Sequence[str],
                             field: str,
                             *,
                             phases: np.ndarray,
                             suptitle: str,
                             save_path: Path,
                             fps: int = FPS,
                             marker_size: float = 16.0) -> None:
    """Render N mesh sequences as side-by-side mollweide maps, shared cbar.

    Each inner sequence is one panel's phase frames; the frame index is
    shared across panels.
    """
    n_panels = len(panel_frames)
    assert n_panels == len(panel_titles)
    n_frames = len(panel_frames[0])
    for fs in panel_frames:
        assert len(fs) == n_frames

    # Precompute (lon, lat, values) per panel per frame, and a shared norm.
    coords: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    values: List[List[np.ndarray]] = []
    label = ""
    for seq in panel_frames:
        c, v = [], []
        for m in seq:
            theta, phi, vals, label = compute_pulsation_scalar(m, field=field)
            c.append(_lonlat(np.asarray(theta), np.asarray(phi)))
            v.append(np.asarray(vals))
        coords.append(c)
        values.append(v)

    stack = np.concatenate([np.ravel(v) for panel in values for v in panel])
    amp = float(np.nanmax(np.abs(stack)))
    if not np.isfinite(amp) or amp == 0.0:
        amp = 1.0
    norm = mpl.colors.Normalize(vmin=-amp, vmax=amp)
    cmap_name = "cmr.redshift"
    try:
        mpl.colormaps[cmap_name]
    except Exception:
        cmap_name = "RdBu_r"

    fig = plt.figure(figsize=(4.2 * n_panels, 3.6), dpi=120)
    axes, scatters = [], []
    for i, title in enumerate(panel_titles):
        ax = fig.add_subplot(1, n_panels, i + 1, projection="mollweide")
        ax.grid(True, linestyle=":", linewidth=0.4, color="grey")
        lon, lat = coords[i][0]
        sc = ax.scatter(lon, lat, c=values[i][0], cmap=cmap_name, norm=norm,
                        s=marker_size, edgecolors="none")
        ax.set_title(title, fontsize=11)
        axes.append(ax)
        scatters.append(sc)

    mappable = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal",
                        shrink=0.5, pad=0.06, aspect=40)
    cbar.set_label(label)
    suptitle_text = fig.suptitle(f"{suptitle}  —  phase = {phases[0]:.2f}",
                                 y=0.995, fontsize=13)

    def update(frame_idx: int):
        for i, sc in enumerate(scatters):
            lon, lat = coords[i][frame_idx]
            sc.set_offsets(np.column_stack([lon, lat]))
            sc.set_array(values[i][frame_idx])
        suptitle_text.set_text(f"{suptitle}  —  phase = {phases[frame_idx]:.2f}")
        return scatters + [suptitle_text]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // max(fps, 1), blit=False)
    smart_save(anim, str(save_path), fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Radial vs spheroidal vs toroidal
# ---------------------------------------------------------------------------

def render_vsh_components(base) -> None:
    phases = np.linspace(0.0, 1.0, N_FRAMES, endpoint=False)
    kwargs = dict(m_order=1, l_degree=2, period=PERIOD)

    m_radial = add_pulsation(
        base, fourier_series_parameters=vsh_fourier(amp_r=0.06), **kwargs,
    )
    m_spheroidal = add_pulsation(
        base, fourier_series_parameters=vsh_fourier(amp_s=0.06), **kwargs,
    )
    m_toroidal = add_pulsation(
        base, fourier_series_parameters=vsh_fourier(amp_t=0.06), **kwargs,
    )

    panels = [
        evaluate_sequence(m_radial, phases),
        evaluate_sequence(m_spheroidal, phases),
        evaluate_sequence(m_toroidal, phases),
    ]
    titles = [
        r"Radial  $R_\ell^m$",
        r"Spheroidal  $S_\ell^m$",
        r"Toroidal  $T_\ell^m$",
    ]

    animate_mollweide_panels(
        panels, titles, field="v_los",
        phases=phases,
        suptitle=r"VSH basis at $(\ell=2,\,m=1)$ — LOS velocity",
        save_path=OUT_DIR / "pulsation_vsh_components.gif",
    )


# ---------------------------------------------------------------------------
# 2. Zonal / tesseral / sectoral grid
# ---------------------------------------------------------------------------

def render_lm_modes(base) -> None:
    phases = np.linspace(0.0, 1.0, N_FRAMES, endpoint=False)

    cases = [
        ((1, 0), r"Dipole  $(\ell=1,\,m=0)$"),
        ((2, 0), r"Zonal  $(\ell=2,\,m=0)$"),
        ((3, 2), r"Tesseral  $(\ell=3,\,m=2)$"),
        ((3, 3), r"Sectoral  $(\ell=3,\,m=3)$"),
    ]

    panels: List[List["MeshModel"]] = []
    titles: List[str] = []
    for (l, m), title in cases:
        m_mode = add_pulsation(
            base, m_order=m, l_degree=l, period=PERIOD,
            fourier_series_parameters=vsh_fourier(amp_r=0.05, amp_s=0.02),
        )
        panels.append(evaluate_sequence(m_mode, phases))
        titles.append(title)

    animate_mollweide_panels(
        panels, titles, field="v_los",
        phases=phases,
        suptitle=r"$(\ell,\,m)$ radial modes — LOS velocity",
        save_path=OUT_DIR / "pulsation_lm_modes.gif",
    )


# ---------------------------------------------------------------------------
# Shared multi-panel 3D mesh animation
# ---------------------------------------------------------------------------

def animate_3d_panels(panel_frames: Sequence[Sequence["MeshModel"]],
                      panel_titles: Sequence[str],
                      field: str,
                      *,
                      phases: np.ndarray,
                      suptitle: str,
                      save_path: Path,
                      fps: int = FPS_3D,
                      view_angles: Tuple[float, float] = (20.0, 30.0)) -> None:
    """Render N mesh sequences as side-by-side 3D meshes, shared cbar.

    The ``Poly3DCollection`` for each panel is created once and updated
    in-place each frame so the mesh visibly deforms with pulsation.
    """
    n_panels = len(panel_frames)
    assert n_panels == len(panel_titles)
    n_frames = len(panel_frames[0])
    for fs in panel_frames:
        assert len(fs) == n_frames

    panels_tris: List[List[np.ndarray]] = []
    panels_vals: List[List[np.ndarray]] = []
    label = ""
    for seq in panel_frames:
        tris, vals = [], []
        for m in seq:
            _, _, v, label = compute_pulsation_scalar(m, field=field)
            tris.append(np.asarray(m.mesh_elements))
            vals.append(np.asarray(v))
        panels_tris.append(tris)
        panels_vals.append(vals)

    stack = np.concatenate([np.ravel(v) for panel in panels_vals for v in panel])
    amp = float(np.nanmax(np.abs(stack)))
    if not np.isfinite(amp) or amp == 0.0:
        amp = 1.0
    norm = mpl.colors.Normalize(vmin=-amp, vmax=amp)
    cmap_name = "cmr.redshift"
    try:
        cmap_obj = mpl.colormaps[cmap_name]
    except Exception:
        cmap_name = "RdBu_r"
        cmap_obj = mpl.colormaps[cmap_name]

    fig = plt.figure(figsize=(3.6 * n_panels, 4.0), dpi=120)
    axes: List[plt.Axes] = []
    collections: List[Poly3DCollection] = []

    ref = panel_frames[0][0]
    center = _mesh_center(ref)
    # Pulsations push vertices out by ~amp * radius; 1.2x gives a tight frame
    # without clipping at peak phase.
    axes_lim = 1.2 * float(ref.radius)

    for i, title in enumerate(panel_titles):
        ax = fig.add_subplot(1, n_panels, i + 1, projection="3d")
        tri0 = panels_tris[i][0]
        face_colors = cmap_obj(norm(panels_vals[i][0]))
        poly = Poly3DCollection(tri0, facecolors=face_colors,
                                edgecolors="none", linewidths=0, shade=False)
        ax.add_collection3d(poly)
        ax.set_xlim(center[0] - axes_lim, center[0] + axes_lim)
        ax.set_ylim(center[1] - axes_lim, center[1] + axes_lim)
        ax.set_zlim(center[2] - axes_lim, center[2] + axes_lim)
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, y=0.98)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        axes.append(ax)
        collections.append(poly)

    mappable = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal",
                        shrink=0.5, pad=0.04, aspect=40)
    cbar.set_label(label)
    suptitle_text = fig.suptitle(f"{suptitle}  —  phase = {phases[0]:.2f}",
                                 y=0.995, fontsize=13)

    def update(frame_idx: int):
        for i, poly in enumerate(collections):
            poly.set_verts(panels_tris[i][frame_idx])
            poly.set_facecolors(cmap_obj(norm(panels_vals[i][frame_idx])))
        suptitle_text.set_text(
            f"{suptitle}  —  phase = {phases[frame_idx]:.2f}"
        )
        return collections + [suptitle_text]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // max(fps, 1), blit=False)
    smart_save(anim, str(save_path), fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. 3D VSH-basis comparison
# ---------------------------------------------------------------------------

def render_3d_vsh() -> None:
    base = build_base_mesh(N_VERTICES_3D)
    phases = np.linspace(0.0, 1.0, N_FRAMES_3D, endpoint=False)
    kwargs = dict(m_order=1, l_degree=2, period=PERIOD)

    # Larger amplitude than the 2D panels so the mesh deformation is obvious.
    m_radial = add_pulsation(
        base, fourier_series_parameters=vsh_fourier(amp_r=0.10), **kwargs,
    )
    m_spheroidal = add_pulsation(
        base, fourier_series_parameters=vsh_fourier(amp_s=0.10), **kwargs,
    )
    m_toroidal = add_pulsation(
        base, fourier_series_parameters=vsh_fourier(amp_t=0.10), **kwargs,
    )

    panels = [
        evaluate_sequence(m_radial, phases),
        evaluate_sequence(m_spheroidal, phases),
        evaluate_sequence(m_toroidal, phases),
    ]
    titles = [
        r"Radial  $R_\ell^m$",
        r"Spheroidal  $S_\ell^m$",
        r"Toroidal  $T_\ell^m$",
    ]
    animate_3d_panels(
        panels, titles, field="v_los",
        phases=phases,
        suptitle=r"VSH basis at $(\ell=2,\,m=1)$ — 3D mesh",
        save_path=OUT_DIR / "pulsation_3d_vsh.gif",
        view_angles=(20.0, 30.0),
    )


# ---------------------------------------------------------------------------
# 4. 3D (l, m) grid
# ---------------------------------------------------------------------------

def render_3d_lm_modes() -> None:
    base = build_base_mesh(N_VERTICES_3D)
    phases = np.linspace(0.0, 1.0, N_FRAMES_3D, endpoint=False)

    cases = [
        ((1, 0), r"Dipole  $(\ell=1,\,m=0)$"),
        ((2, 0), r"Zonal  $(\ell=2,\,m=0)$"),
        ((3, 2), r"Tesseral  $(\ell=3,\,m=2)$"),
        ((3, 3), r"Sectoral  $(\ell=3,\,m=3)$"),
    ]

    panels: List[List["MeshModel"]] = []
    titles: List[str] = []
    for (l, m), title in cases:
        m_mode = add_pulsation(
            base, m_order=m, l_degree=l, period=PERIOD,
            fourier_series_parameters=vsh_fourier(amp_r=0.08, amp_s=0.02),
        )
        panels.append(evaluate_sequence(m_mode, phases))
        titles.append(title)

    animate_3d_panels(
        panels, titles, field="v_los",
        phases=phases,
        suptitle=r"$(\ell,\,m)$ radial modes — 3D mesh",
        save_path=OUT_DIR / "pulsation_3d_lm_modes.gif",
        view_angles=(22.0, 35.0),
    )


# ---------------------------------------------------------------------------
# 5. Observer-disk view of a sectoral mode at a tilted inclination
# ---------------------------------------------------------------------------

def render_observer_disk() -> None:
    # Tilt the line-of-sight so the equatorial drift of a sectoral mode
    # crosses the visible disk rather than sitting edge-on at the equator.
    inc_deg = 65.0
    inc = np.deg2rad(inc_deg)
    los = jnp.array([np.sin(inc), 0.0, -np.cos(inc)], dtype=jnp.float64)

    base = IcosphereModel.construct(
        n_vertices=N_VERTICES,
        radius=1.0,
        mass=1.0,
        parameters=jnp.array([5700.0, 4.4]),
        parameter_names=["teff", "log_g"],
        max_pulsation_mode=5,
        max_fourier_order=3,
    )
    base = base._replace(los_vector=los / jnp.linalg.norm(los))

    m_sectoral = add_pulsation(
        base, m_order=3, l_degree=3, period=PERIOD,
        fourier_series_parameters=vsh_fourier(amp_r=0.05, amp_s=0.02),
    )

    phases = np.linspace(0.0, 1.0, N_FRAMES, endpoint=False)
    frames = [evaluate_pulsations(m_sectoral, t=float(ph) * PERIOD)
              for ph in phases]

    animate_observed_disk(
        frames, field="v_los",
        timestamps=phases, timestamp_label="phase",
        draw_limb=True, draw_rotation_axis=True,
        fps=FPS,
        save_path=str(OUT_DIR / "pulsation_observer_disk.gif"),
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    base = build_base_mesh()

    print("Rendering VSH-component comparison (mollweide)...")
    render_vsh_components(base)

    print("Rendering (l, m) mode grid (mollweide)...")
    render_lm_modes(base)

    print("Rendering tilted observer-disk view...")
    render_observer_disk()

    print("Rendering VSH-component comparison (3D mesh)...")
    render_3d_vsh()

    print("Rendering (l, m) mode grid (3D mesh)...")
    render_3d_lm_modes()

    print("Done. Animations in:", OUT_DIR)


if __name__ == "__main__":
    main()
