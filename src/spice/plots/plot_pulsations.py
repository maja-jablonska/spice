"""Plotting helpers for stellar pulsation velocity fields.

These functions operate on a :class:`spice.models.MeshModel` whose
``pulsation_velocities`` have already been populated (typically by
``spice.models.mesh_transform.evaluate_pulsations(mesh, t)``).  They project
those per-element Cartesian vectors into a local spherical basis and render
the resulting scalars directly on the mesh — no grid resampling and no
derivative-based diagnostic fields.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_mesh import (
    DEFAULT_CMAP,
    _mesh_center,
    _set_axes_centered,
    _single_mesh_arrow_vectors,
    smart_save,
)

if TYPE_CHECKING:
    from spice.models import MeshModel


# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------

PULSATION_FIELDS: Tuple[str, ...] = (
    "v_r",
    "v_theta",
    "v_phi",
    "v_h",
    "speed",
    "v_los",
)

DEFAULT_FIELD_LABELS = {
    "v_r": r"$v_r$ [km/s]",
    "v_theta": r"$v_\theta$ [km/s]",
    "v_phi": r"$v_\phi$ [km/s]",
    "v_h": r"$|v_h|$ [km/s]",
    "speed": r"$|v|$ [km/s]",
    "v_los": r"$v_\mathrm{LOS}$ [km/s]",
}

_DIVERGING_FIELDS = frozenset({"v_r", "v_theta", "v_phi", "v_los"})

_DEFAULT_DIVERGING_CMAP = "cmr.redshift"
_DEFAULT_MAGNITUDE_CMAP = DEFAULT_CMAP


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _orthonormal_basis_from_axis(axis: np.ndarray) -> np.ndarray:
    """Right-handed 3x3 matrix whose third column is ``axis``."""
    z = _unit(_as_np(axis))
    seed = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = _unit(np.cross(seed, z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _spherical_components(positions: np.ndarray,
                          velocities: np.ndarray,
                          up_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """Return per-element ``(theta, phi, v_r, v_theta, v_phi)``.

    Colatitude ``theta`` is measured from ``up_axis``; ``theta_hat`` points
    from the pole toward the equator and ``phi_hat`` points eastward.
    """
    R = _orthonormal_basis_from_axis(up_axis)
    pos_local = positions @ R
    vel_local = velocities @ R

    r = np.linalg.norm(pos_local, axis=1)
    r_safe = np.where(r > 1e-12, r, 1e-12)
    x, y, z = pos_local[:, 0], pos_local[:, 1], pos_local[:, 2]

    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    r_hat = np.stack([sin_t * cos_p, sin_t * sin_p, cos_t], axis=1)
    theta_hat = np.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], axis=1)
    phi_hat = np.stack([-sin_p, cos_p, np.zeros_like(phi)], axis=1)

    v_r = np.sum(vel_local * r_hat, axis=1)
    v_theta = np.sum(vel_local * theta_hat, axis=1)
    v_phi = np.sum(vel_local * phi_hat, axis=1)

    return theta, phi, v_r, v_theta, v_phi


def _resolve_up_axis(mesh: "MeshModel",
                     up_axis: Optional[np.ndarray] = None) -> np.ndarray:
    if up_axis is not None:
        return _unit(_as_np(up_axis))
    return _unit(_as_np(mesh.rotation_axis))


# ---------------------------------------------------------------------------
# Per-element scalar computation
# ---------------------------------------------------------------------------

def compute_pulsation_scalar(mesh: "MeshModel",
                             field: str = "speed",
                             up_axis: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return ``(theta, phi, values, label)`` evaluated at each mesh cell.

    Parameters
    ----------
    mesh : MeshModel
        Mesh with ``pulsation_velocities`` already populated.
    field : str
        One of :data:`PULSATION_FIELDS`.
    up_axis : array-like, shape (3,), optional
        Axis to use as the "pole" when defining the spherical frame.
        Defaults to the mesh rotation axis.

    Returns
    -------
    theta, phi : ndarray, shape (n_cells,)
        Spherical coordinates of each mesh-cell centre, in radians.
    values : ndarray, shape (n_cells,)
        The requested scalar evaluated at each cell.
    label : str
        Suggested colorbar label.
    """
    if field not in PULSATION_FIELDS:
        raise ValueError(
            f"Unknown pulsation field {field!r}; expected one of {PULSATION_FIELDS}"
        )

    positions = _as_np(mesh.d_centers)
    velocities = _as_np(mesh.pulsation_velocities)
    axis = _resolve_up_axis(mesh, up_axis)

    theta, phi, v_r, v_theta, v_phi = _spherical_components(positions, velocities, axis)

    if field == "v_r":
        values = v_r
    elif field == "v_theta":
        values = v_theta
    elif field == "v_phi":
        values = v_phi
    elif field == "v_h":
        values = np.sqrt(v_theta ** 2 + v_phi ** 2)
    elif field == "speed":
        values = np.sqrt(v_r ** 2 + v_theta ** 2 + v_phi ** 2)
    else:  # v_los
        los = _unit(_as_np(mesh.los_vector))
        values = velocities @ los

    return theta, phi, values, DEFAULT_FIELD_LABELS[field]


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

_FLAT_PROJECTIONS = ("mollweide", "hammer", "aitoff", "lambert")


def _projection_kwargs(projection: str) -> dict:
    proj = projection.lower()
    if proj in _FLAT_PROJECTIONS:
        return {"projection": proj}
    if proj in ("rect", "rectangular", "equirectangular", "plate_carree", "none"):
        return {}
    raise ValueError(
        f"Unknown projection {projection!r}; expected one of "
        f"{_FLAT_PROJECTIONS + ('rect',)}"
    )


def _norm_for_field(field: str, values: np.ndarray,
                    vmin: Optional[float], vmax: Optional[float]) -> mpl.colors.Normalize:
    if vmin is None and vmax is None and field in _DIVERGING_FIELDS:
        amp = float(np.nanmax(np.abs(values)))
        if amp == 0.0 or not np.isfinite(amp):
            amp = 1.0
        return mpl.colors.Normalize(vmin=-amp, vmax=amp)
    lo = np.nanmin(values) if vmin is None else vmin
    hi = np.nanmax(values) if vmax is None else vmax
    if lo == hi:
        hi = lo + 1e-9
    return mpl.colors.Normalize(vmin=lo, vmax=hi)


def _cmap_for_field(field: str, cmap: Optional[str]) -> str:
    if cmap is not None:
        return cmap
    if field in _DIVERGING_FIELDS:
        return _DEFAULT_DIVERGING_CMAP
    return _DEFAULT_MAGNITUDE_CMAP


def _lonlat_for_projection(theta: np.ndarray, phi: np.ndarray,
                           projection: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert spherical ``(theta, phi)`` to the coordinates used by ``ax``."""
    lat = 0.5 * np.pi - theta
    if projection.lower() in _FLAT_PROJECTIONS:
        return phi, lat
    return np.rad2deg(phi), np.rad2deg(lat)


def _setup_rect_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)


def _overlay_quiver(ax: plt.Axes, theta: np.ndarray, phi: np.ndarray,
                    v_theta: np.ndarray, v_phi: np.ndarray,
                    projection: str, *, stride: int, color: str) -> None:
    """Sparse horizontal-velocity arrows at the mesh-cell positions."""
    s = max(int(stride), 1)
    idx = np.arange(0, theta.size, s)
    # Latitude increases opposite to colatitude, so the upward (north) arrow
    # component is ``-v_theta``.
    x, y = _lonlat_for_projection(theta[idx], phi[idx], projection)
    U = v_phi[idx]
    V = -v_theta[idx]
    ax.quiver(x, y, U, V, color=color, pivot="middle", width=0.0025)


# ---------------------------------------------------------------------------
# Public: 2D surface map
# ---------------------------------------------------------------------------

def plot_pulsation_map(mesh: "MeshModel",
                       field: str = "speed",
                       projection: str = "mollweide",
                       *,
                       up_axis: Optional[np.ndarray] = None,
                       cmap: Optional[str] = None,
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       marker_size: float = 12.0,
                       overlay_quiver: bool = False,
                       quiver_stride: int = 12,
                       quiver_color: str = "white",
                       title: Optional[str] = None,
                       axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                       colorbar: bool = True
                       ) -> Tuple[plt.Figure, plt.Axes]:
    """Render a per-element pulsation scalar as a flat surface map.

    Parameters
    ----------
    mesh : MeshModel
        Mesh with pulsation velocities already evaluated.
    field : str
        One of :data:`PULSATION_FIELDS`.
    projection : str
        ``'mollweide'``, ``'hammer'``, ``'aitoff'``, ``'lambert'`` or
        ``'rect'`` for a plain equirectangular axis.
    marker_size : float
        Size of the per-cell scatter points.
    overlay_quiver : bool
        If ``True``, overlay sparse horizontal-velocity arrows sampled from
        the mesh cells.
    """
    theta, phi, values, label = compute_pulsation_scalar(
        mesh, field=field, up_axis=up_axis
    )
    cmap_name = _cmap_for_field(field, cmap)
    norm = _norm_for_field(field, values, vmin, vmax)

    if axes is None:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111, **_projection_kwargs(projection))
    else:
        fig, ax = axes

    x, y = _lonlat_for_projection(theta, phi, projection)
    sc = ax.scatter(x, y, c=values, cmap=cmap_name, norm=norm,
                    s=marker_size, edgecolors="none")

    if projection.lower() in _FLAT_PROJECTIONS:
        ax.grid(True, linestyle=":", linewidth=0.5, color="grey")
    else:
        _setup_rect_axes(ax)

    if overlay_quiver:
        _, _, _, v_theta, v_phi = _spherical_components(
            _as_np(mesh.d_centers),
            _as_np(mesh.pulsation_velocities),
            _resolve_up_axis(mesh, up_axis),
        )
        _overlay_quiver(ax, theta, phi, v_theta, v_phi,
                        projection, stride=quiver_stride, color=quiver_color)

    ax.set_title(title or label)

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08,
                            orientation="horizontal")
        cbar.set_label(label)

    return fig, ax


def plot_pulsation_components(mesh: "MeshModel",
                              fields: Sequence[str] = ("v_r", "v_theta",
                                                       "v_phi", "speed"),
                              projection: str = "mollweide",
                              *,
                              up_axis: Optional[np.ndarray] = None,
                              figsize: Optional[Tuple[float, float]] = None,
                              cmap_overrides: Optional[dict] = None,
                              marker_size: float = 12.0,
                              suptitle: Optional[str] = None
                              ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot several pulsation scalars side-by-side on matching projections."""
    n = len(fields)
    if n == 0:
        raise ValueError("Provide at least one field to plot")

    n_cols = 2 if n >= 2 else 1
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (6.5 * n_cols, 3.5 * n_rows)

    fig = plt.figure(figsize=figsize)
    axes: List[plt.Axes] = []
    cmap_overrides = cmap_overrides or {}

    for i, field in enumerate(fields):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, **_projection_kwargs(projection))
        plot_pulsation_map(
            mesh, field=field, projection=projection, up_axis=up_axis,
            cmap=cmap_overrides.get(field), axes=(fig, ax),
            marker_size=marker_size,
        )
        axes.append(ax)

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Public: cross-section cuts
# ---------------------------------------------------------------------------

def plot_pulsation_cross_section(mesh: "MeshModel",
                                 slice: str = "longitude",
                                 fixed_deg: float = 0.0,
                                 fields: Sequence[str] = ("v_r", "v_theta", "v_phi"),
                                 *,
                                 tolerance_deg: float = 5.0,
                                 up_axis: Optional[np.ndarray] = None,
                                 axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                                 title: Optional[str] = None
                                 ) -> Tuple[plt.Figure, plt.Axes]:
    """1-D cut through the per-element pulsation scalars along a narrow band.

    Parameters
    ----------
    slice : {'longitude', 'latitude'}
        ``'longitude'``: keep cells near a fixed longitude, plot field(lat).
        ``'latitude'``: keep cells near a fixed latitude, plot field(lon).
    fixed_deg : float
        Longitude or latitude at which to take the cut, in degrees.
    tolerance_deg : float
        Half-width of the band (in degrees) used to select cells.
    fields : sequence of str
        Which scalar fields to overlay on the same axis.
    """
    if axes is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig, ax = axes

    for field in fields:
        theta, phi, values, label = compute_pulsation_scalar(
            mesh, field=field, up_axis=up_axis
        )
        lat = 0.5 * np.pi - theta
        if slice == "longitude":
            phi_target = np.deg2rad(fixed_deg)
            # smallest signed angular difference, handles wrap
            dphi = np.arctan2(np.sin(phi - phi_target), np.cos(phi - phi_target))
            mask = np.abs(dphi) < np.deg2rad(tolerance_deg)
            order = np.argsort(lat[mask])
            ax.plot(np.rad2deg(lat[mask])[order], values[mask][order],
                    "o-", label=label, markersize=3)
            ax.set_xlabel("latitude [deg]")
        elif slice == "latitude":
            lat_target = np.deg2rad(fixed_deg)
            mask = np.abs(lat - lat_target) < np.deg2rad(tolerance_deg)
            order = np.argsort(phi[mask])
            ax.plot(np.rad2deg(phi[mask])[order], values[mask][order],
                    "o-", label=label, markersize=3)
            ax.set_xlabel("longitude [deg]")
        else:
            raise ValueError(
                f"slice must be 'longitude' or 'latitude', got {slice!r}"
            )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_ylabel("velocity [km/s]")
    cut_name = "phi" if slice == "longitude" else "lat"
    ax.set_title(
        title
        or f"Pulsation cross section at {cut_name} = {fixed_deg:g} deg "
           f"(±{tolerance_deg:g} deg)"
    )
    ax.legend(loc="best", fontsize=9)
    return fig, ax


# ---------------------------------------------------------------------------
# Public: 3D sphere with scalar colour + sparse arrows
# ---------------------------------------------------------------------------

def plot_pulsation_3D_sparse(mesh: "MeshModel",
                             scalar: str = "speed",
                             arrow_stride: int = 10,
                             *,
                             up_axis: Optional[np.ndarray] = None,
                             cmap: Optional[str] = None,
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             arrow_scale: float = 0.35,
                             arrow_color: str = "white",
                             axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                             title: Optional[str] = None,
                             draw_los_vector: bool = True,
                             draw_rotation_axis: bool = True,
                             colorbar: bool = True
                             ) -> Tuple[plt.Figure, plt.Axes]:
    """3D mesh coloured by a per-element scalar with a sparse quiver overlay.

    Faces come from ``mesh.mesh_elements`` (the actual pulsation-deformed
    triangles); arrows are the raw per-cell ``pulsation_velocities``
    subsampled by ``arrow_stride``.
    """
    positions = _as_np(mesh.d_centers)
    velocities = _as_np(mesh.pulsation_velocities)
    center = _mesh_center(mesh)
    radius = float(mesh.radius)

    _, _, values, label = compute_pulsation_scalar(
        mesh, field=scalar, up_axis=up_axis
    )
    cmap_name = _cmap_for_field(scalar, cmap)
    norm = _norm_for_field(scalar, values, vmin, vmax)

    triangles = _as_np(mesh.mesh_elements)  # (n_faces, 3, 3)

    if axes is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = axes

    face_colors = mpl.colormaps[cmap_name](norm(values))
    poly = Poly3DCollection(triangles, facecolors=face_colors,
                            edgecolors="none", linewidths=0, shade=False)
    ax.add_collection3d(poly)

    n_cells = positions.shape[0]
    if arrow_stride > 1:
        idx = np.arange(0, n_cells, int(arrow_stride))
    else:
        idx = np.arange(n_cells)
    if idx.size > 0:
        start = center + radius * positions[idx] / (
            np.linalg.norm(positions[idx], axis=1, keepdims=True) + 1e-12
        )
        vmax_mag = float(np.max(np.linalg.norm(velocities[idx], axis=1)) + 1e-12)
        arrows = velocities[idx] * (arrow_scale * radius / vmax_mag)
        ax.quiver(start[:, 0], start[:, 1], start[:, 2],
                  arrows[:, 0], arrows[:, 1], arrows[:, 2],
                  color=arrow_color, linewidth=1.2)

    axes_lim = 1.5 * radius
    _set_axes_centered(ax, center, axes_lim)
    ax.set_xlabel(r"$X\,[R_\odot]$")
    ax.set_ylabel(r"$Y\,[R_\odot]$")
    ax.set_zlabel(r"$Z\,[R_\odot]$")

    los_start, los_vec, rot_start, rot_vec = _single_mesh_arrow_vectors(mesh)
    if draw_los_vector:
        ax.quiver(*los_start, *los_vec, color="red", linewidth=2.5,
                  label="LOS vector")
    if draw_rotation_axis:
        ax.quiver(*rot_start, *rot_vec, color="black", linewidth=2.5,
                  label="Rotation axis")
    if draw_los_vector or draw_rotation_axis:
        ax.legend(loc="upper left")

    if colorbar:
        mappable = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.12)
        cbar.set_label(label)
    ax.set_title(title or label)
    return fig, ax


# ---------------------------------------------------------------------------
# Public: zoom onto a small surface patch
# ---------------------------------------------------------------------------

def plot_pulsation_patch_zoom(mesh: "MeshModel",
                              field: str = "v_r",
                              *,
                              center_lonlat_deg: Tuple[float, float] = (0.0, 0.0),
                              angular_radius_deg: float = 20.0,
                              up_axis: Optional[np.ndarray] = None,
                              cmap: Optional[str] = None,
                              vmin: Optional[float] = None,
                              vmax: Optional[float] = None,
                              symmetric: Optional[bool] = None,
                              show_arrows: bool = True,
                              arrow_component: str = "horizontal",
                              arrow_density: float = 0.6,
                              arrow_length_frac: float = 0.07,
                              arrow_color: str = "black",
                              arrow_width: float = 0.0035,
                              arrow_headwidth: float = 3.0,
                              arrow_headlength: float = 4.0,
                              show_edges: bool = False,
                              seed: int = 0,
                              figsize: Tuple[float, float] = (7.0, 6.5),
                              axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                              title: Optional[str] = None,
                              colorbar: bool = True
                              ) -> Tuple[plt.Figure, plt.Axes]:
    """Zoom onto a surface patch as a 2D orthographic projection.

    Cells whose centre direction lies within ``angular_radius_deg`` of the
    ``(lon, lat)`` target (in the ``up_axis`` frame) are projected
    orthographically onto the tangent plane at the patch centre, with the
    in-plane basis aligned to ``(east, north)``. Going 2D avoids 3D quiver's
    foreshortening and bulky arrowheads, and lets face colour carry magnitude
    without fighting with perspective cues.

    Arrows are drawn at a uniform display length so their geometry encodes
    only direction; use the face colour for magnitude. The default set is the
    **horizontal** velocity component — on a spheroidal/toroidal mode the flow
    pattern reads immediately (converging/diverging nodes, rotation sense).
    Set ``arrow_component='full'`` to include the radial component in the
    projected vectors.

    Parameters
    ----------
    center_lonlat_deg : (lon, lat)
        Patch centre in degrees, in the ``up_axis`` frame (longitude east of
        the zero-meridian, latitude from the equator).
    angular_radius_deg : float
        Half-angle of the cone used to select cells. Orthographic projection
        is reasonable up to ~30°; beyond that the edges distort noticeably.
    arrow_component : {'horizontal', 'full'}
        Which part of the 3D velocity to project onto the tangent plane.
    arrow_density : float
        Fraction of visible cells to draw arrows for (default 0.6). Random
        subsample seeded with ``seed`` for reproducibility.
    arrow_length_frac : float
        Uniform arrow length as a fraction of the plot half-width.
    arrow_width, arrow_headwidth, arrow_headlength : float
        Passed through to :meth:`~matplotlib.axes.Axes.quiver` for shaft and
        head sizing. Defaults give thin shafts with modest heads.
    show_edges : bool
        If ``True``, draw faint cell boundaries. Off by default so the mode
        pattern is read from the smooth colour gradient rather than the
        triangulation.
    """
    if arrow_component not in ("horizontal", "full"):
        raise ValueError(
            f"arrow_component must be 'horizontal' or 'full', got {arrow_component!r}"
        )
    if not (0.0 < arrow_density <= 1.0):
        raise ValueError(
            f"arrow_density must be in (0, 1], got {arrow_density!r}"
        )

    positions = _as_np(mesh.d_centers)
    velocities = _as_np(mesh.pulsation_velocities)
    triangles = _as_np(mesh.mesh_elements)
    radius = float(mesh.radius)
    sphere_center = _mesh_center(mesh)

    up = _resolve_up_axis(mesh, up_axis)
    basis = _orthonormal_basis_from_axis(up)

    lon_deg, lat_deg = center_lonlat_deg
    phi_t = np.deg2rad(lon_deg)
    theta_t = 0.5 * np.pi - np.deg2rad(lat_deg)
    n_local = np.array([
        np.sin(theta_t) * np.cos(phi_t),
        np.sin(theta_t) * np.sin(phi_t),
        np.cos(theta_t),
    ])
    n_world = basis @ n_local
    n_world = n_world / (np.linalg.norm(n_world) + 1e-12)

    # Tangent basis at the patch centre.  ``north`` is the up-axis component
    # lying in the tangent plane; ``east`` completes a right-handed frame with
    # the outward normal.  At a pole the up-axis is degenerate, so fall back
    # to the local-x direction (perpendicular to up by construction).
    up_unit = up / (np.linalg.norm(up) + 1e-12)
    proj_up = up_unit - (up_unit @ n_world) * n_world
    if np.linalg.norm(proj_up) < 1e-6:
        alt = basis[:, 0]
        proj_up = alt - (alt @ n_world) * n_world
    north = proj_up / (np.linalg.norm(proj_up) + 1e-12)
    east = np.cross(north, n_world)

    rel = positions - sphere_center
    dirs = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-12)
    cos_cut = float(np.cos(np.deg2rad(angular_radius_deg)))
    mask = (dirs @ n_world) >= cos_cut
    if not mask.any():
        raise ValueError(
            f"No cells inside the {angular_radius_deg:g}° patch at "
            f"(lon, lat) = ({lon_deg:g}, {lat_deg:g}) deg"
        )

    _, _, values, label = compute_pulsation_scalar(
        mesh, field=field, up_axis=up_axis
    )
    values_patch = values[mask]

    norm, _ = _shared_norm([values_patch], field, vmin, vmax, symmetric)
    cmap_name = _cmap_for_field(field, cmap)
    cmap_obj = mpl.colormaps[cmap_name]

    tri_rel = triangles[mask] - sphere_center              # (n_patch, 3, 3)
    tri_2d = np.stack([tri_rel @ east, tri_rel @ north],
                      axis=-1)                              # (n_patch, 3, 2)

    if axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = axes

    face_colors = cmap_obj(norm(values_patch))
    edge_kw = dict(edgecolors="0.4", linewidths=0.2) if show_edges \
        else dict(edgecolors="none", linewidths=0)
    poly = PolyCollection(tri_2d, facecolors=face_colors,
                          antialiaseds=True, **edge_kw)
    ax.add_collection(poly)

    if show_arrows:
        visible_idx = np.nonzero(mask)[0]
        n_arrows = max(int(round(visible_idx.size * float(arrow_density))), 1)
        if n_arrows < visible_idx.size:
            rng = np.random.default_rng(seed)
            sel = rng.choice(visible_idx, size=n_arrows, replace=False)
        else:
            sel = visible_idx

        pos_rel = positions[sel] - sphere_center
        x_arr = pos_rel @ east
        y_arr = pos_rel @ north

        v_sel = velocities[sel]
        if arrow_component == "horizontal":
            r_hat_sel = pos_rel / (
                np.linalg.norm(pos_rel, axis=1, keepdims=True) + 1e-12
            )
            v_sel = v_sel - np.sum(v_sel * r_hat_sel, axis=1, keepdims=True) * r_hat_sel
        vx = v_sel @ east
        vy = v_sel @ north

        speed = np.sqrt(vx ** 2 + vy ** 2)
        speed_safe = np.where(speed > 1e-12, speed, 1e-12)
        patch_half = radius * np.sin(np.deg2rad(angular_radius_deg))
        arrow_len = float(arrow_length_frac) * patch_half
        ux = (vx / speed_safe) * arrow_len
        uy = (vy / speed_safe) * arrow_len
        dead = speed <= 1e-12
        ux[dead] = 0.0
        uy[dead] = 0.0

        ax.quiver(x_arr, y_arr, ux, uy,
                  color=arrow_color,
                  angles="xy", scale_units="xy", scale=1.0,
                  width=arrow_width,
                  headwidth=arrow_headwidth,
                  headlength=arrow_headlength,
                  headaxislength=arrow_headlength * 0.9,
                  linewidth=0.0,
                  zorder=3)

    lim = radius * np.sin(np.deg2rad(angular_radius_deg)) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"east $[R_\odot]$")
    ax.set_ylabel(r"north $[R_\odot]$")

    if colorbar:
        mappable = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.04)
        cbar.set_label(label)

    default_title = (
        f"{label} — patch at (lon, lat) = ({lon_deg:g}°, {lat_deg:g}°), "
        f"r = {angular_radius_deg:g}°"
    )
    ax.set_title(title or default_title)
    return fig, ax


# ---------------------------------------------------------------------------
# Public: disk overview + patch zoom side-by-side
# ---------------------------------------------------------------------------

def plot_pulsation_disk_with_patch_zoom(
    mesh: "MeshModel",
    field: str = "v_r",
    *,
    center_lonlat_deg: Tuple[float, float] = (0.0, 0.0),
    angular_radius_deg: float = 20.0,
    up_axis: Optional[np.ndarray] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric: Optional[bool] = None,
    # disk panel
    disk_marker_size: float = 14.0,
    disk_draw_limb: bool = True,
    disk_draw_rotation_axis: bool = False,
    # patch marker overlaid on the disk
    patch_marker_shape: str = "square",
    patch_marker_color: str = "black",
    patch_marker_linewidth: float = 1.5,
    # zoom panel (forwarded to plot_pulsation_patch_zoom)
    show_arrows: bool = True,
    arrow_component: str = "horizontal",
    arrow_density: float = 0.6,
    arrow_length_frac: float = 0.07,
    arrow_color: str = "black",
    arrow_width: float = 0.0035,
    arrow_headwidth: float = 3.0,
    arrow_headlength: float = 4.0,
    show_edges: bool = False,
    seed: int = 0,
    # layout
    figsize: Tuple[float, float] = (12.5, 5.5),
    width_ratios: Tuple[float, float] = (1.0, 1.0),
    disk_title: Optional[str] = None,
    zoom_title: Optional[str] = None,
    suptitle: Optional[str] = None,
    colorbar: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Observer disk and patch zoom side-by-side, patch footprint marked on the disk.

    The left axis is an orthographic projection along ``mesh.los_vector`` of
    the visible hemisphere (``mesh.mus > 0``), coloured by ``field``. A
    square (``patch_marker_shape='square'``) or circle (``'circle'``) drawn
    at the projected patch centre shows where the right-hand zoom is taken;
    side length / diameter is ``2 R sin(angular_radius_deg)``, which
    bounds the patch on the sphere (it slightly overestimates near the
    limb because of foreshortening).

    Both panels share the same colour norm, computed from the visible-disk
    values, so the colour of a cell reads consistently across the two views.
    The zoom panel is rendered by :func:`plot_pulsation_patch_zoom` — every
    arrow/edge keyword on that function is exposed here with the same name.
    """
    if patch_marker_shape not in ("square", "circle"):
        raise ValueError(
            f"patch_marker_shape must be 'square' or 'circle', "
            f"got {patch_marker_shape!r}"
        )

    radius = float(mesh.radius)

    up = _resolve_up_axis(mesh, up_axis)
    basis = _orthonormal_basis_from_axis(up)
    lon_deg, lat_deg = center_lonlat_deg
    phi_t = np.deg2rad(lon_deg)
    theta_t = 0.5 * np.pi - np.deg2rad(lat_deg)
    n_local = np.array([
        np.sin(theta_t) * np.cos(phi_t),
        np.sin(theta_t) * np.sin(phi_t),
        np.cos(theta_t),
    ])
    n_world = basis @ n_local
    n_world = n_world / (np.linalg.norm(n_world) + 1e-12)

    los = _unit(_as_np(mesh.los_vector))
    _, v1, v2 = _disk_basis(los)
    d_centers = _as_np(mesh.d_centers)
    offsets = _as_np(mesh.center_pulsation_offsets)
    pos = d_centers + offsets
    disk_xy = np.stack([pos @ v1, pos @ v2], axis=1)

    _, _, values, label = compute_pulsation_scalar(
        mesh, field=field, up_axis=up_axis
    )
    mu = _as_np(mesh.mus)
    visible = mu > 0

    norm, _ = _shared_norm([values[visible]], field, vmin, vmax, symmetric)
    cmap_name = _cmap_for_field(field, cmap)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios), wspace=0.25)
    ax_disk = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    # --- disk panel -------------------------------------------------------
    ax_disk.set_aspect("equal", adjustable="box")
    lim = 1.15 * radius
    ax_disk.set_xlim(-lim, lim)
    ax_disk.set_ylim(-lim, lim)
    ax_disk.set_xlabel(r"disk $x\,[R_\odot]$")
    ax_disk.set_ylabel(r"disk $y\,[R_\odot]$")

    if disk_draw_limb:
        ax_disk.add_patch(plt.Circle(
            (0, 0), radius, fill=False, edgecolor="grey",
            linewidth=0.8, linestyle="--", zorder=0,
        ))

    if disk_draw_rotation_axis:
        rot = _unit(_as_np(mesh.rotation_axis))
        rx, ry = float(rot @ v1), float(rot @ v2)
        ax_disk.plot([-1.1 * radius * rx, 1.1 * radius * rx],
                     [-1.1 * radius * ry, 1.1 * radius * ry],
                     color="black", linewidth=0.8, linestyle=":", zorder=1)

    ax_disk.scatter(disk_xy[visible, 0], disk_xy[visible, 1],
                    c=values[visible], cmap=cmap_name, norm=norm,
                    s=disk_marker_size, edgecolors="none", zorder=2)

    patch_xy = np.array([float(n_world @ v1), float(n_world @ v2)]) * radius
    side = 2.0 * radius * float(np.sin(np.deg2rad(angular_radius_deg)))
    # `mesh.mus = -n · los_vector` in this codebase, so a cell is on the
    # visible hemisphere when its outward direction has `n · los < 0`.
    patch_visible = float(n_world @ los) < 0.0

    marker_ls = "-" if patch_visible else ":"
    if patch_marker_shape == "square":
        marker = plt.Rectangle(
            (patch_xy[0] - side / 2, patch_xy[1] - side / 2),
            side, side, fill=False,
            edgecolor=patch_marker_color, linewidth=patch_marker_linewidth,
            linestyle=marker_ls, zorder=4,
        )
    else:
        marker = plt.Circle(
            (patch_xy[0], patch_xy[1]), side / 2, fill=False,
            edgecolor=patch_marker_color, linewidth=patch_marker_linewidth,
            linestyle=marker_ls, zorder=4,
        )
    ax_disk.add_patch(marker)

    if not patch_visible:
        ax_disk.text(patch_xy[0], patch_xy[1], "behind",
                     ha="center", va="center", fontsize=8,
                     color=patch_marker_color, zorder=5)

    ax_disk.set_title(disk_title or f"{label} — visible disk")

    # --- zoom panel -------------------------------------------------------
    plot_pulsation_patch_zoom(
        mesh, field=field,
        center_lonlat_deg=center_lonlat_deg,
        angular_radius_deg=angular_radius_deg,
        up_axis=up_axis,
        cmap=cmap,
        vmin=float(norm.vmin), vmax=float(norm.vmax),
        symmetric=symmetric,
        show_arrows=show_arrows,
        arrow_component=arrow_component,
        arrow_density=arrow_density,
        arrow_length_frac=arrow_length_frac,
        arrow_color=arrow_color,
        arrow_width=arrow_width,
        arrow_headwidth=arrow_headwidth,
        arrow_headlength=arrow_headlength,
        show_edges=show_edges,
        seed=seed,
        axes=(fig, ax_zoom),
        title=zoom_title,
        colorbar=False,
    )

    if colorbar:
        mappable = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=[ax_disk, ax_zoom],
                            orientation="horizontal", shrink=0.6, pad=0.12)
        cbar.set_label(label)

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig, (ax_disk, ax_zoom)


# ---------------------------------------------------------------------------
# Public: phase-grid small multiples
# ---------------------------------------------------------------------------

def _shared_norm(values_list: Sequence[np.ndarray],
                 field: str,
                 vmin: Optional[float],
                 vmax: Optional[float],
                 symmetric: Optional[bool]) -> Tuple[mpl.colors.Normalize, bool]:
    if symmetric is None:
        symmetric = field in _DIVERGING_FIELDS
    if vmin is None and vmax is None:
        stacked = np.concatenate([np.ravel(v) for v in values_list]) \
            if values_list else np.zeros(0)
        if symmetric and stacked.size:
            amp = float(np.nanmax(np.abs(stacked)))
            if amp == 0.0 or not np.isfinite(amp):
                amp = 1.0
            vmin, vmax = -amp, amp
        elif stacked.size:
            vmin = float(np.nanmin(stacked))
            vmax = float(np.nanmax(stacked))
            if vmin == vmax:
                vmax = vmin + 1e-9
        else:
            vmin, vmax = -1.0, 1.0
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax), symmetric


def plot_pulsation_phase_grid(meshes: Sequence["MeshModel"],
                              field: str = "v_r",
                              *,
                              phases: Optional[Sequence[float]] = None,
                              phase_label: str = r"$\phi$",
                              n_cols: Optional[int] = None,
                              up_axis: Optional[np.ndarray] = None,
                              cmap: Optional[str] = None,
                              vmin: Optional[float] = None,
                              vmax: Optional[float] = None,
                              symmetric: Optional[bool] = None,
                              view_angles: Optional[Tuple[float, float]] = None,
                              figsize: Optional[Tuple[float, float]] = None,
                              draw_rotation_axis: bool = False,
                              suptitle: Optional[str] = None,
                              colorbar: bool = True
                              ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Tile the mesh at several phases of one pulsation cycle.

    Intended for 6–8 equally spaced phases. Viewing geometry and the colour
    scale are locked across panels so azimuthal drift (sectoral modes) and
    nodal-pattern evolution (tesseral modes) are easy to read.

    Parameters
    ----------
    meshes : sequence of MeshModel
        One mesh per phase, each with ``pulsation_velocities`` already
        evaluated.
    field : str
        Scalar from :data:`PULSATION_FIELDS` used to colour the faces.
    phases : sequence of float, optional
        Phase value for each mesh (e.g. ``np.linspace(0, 1, n, endpoint=False)``).
        Used for panel labels; if omitted, panels are labelled by index.
    n_cols : int, optional
        Columns in the grid. Defaults to 4 for ``n >= 7``, 3 for ``n in 5..6``,
        otherwise ``n``.
    view_angles : (elev, azim), optional
        Applied to every panel. Omit to use matplotlib's default orientation.
    draw_rotation_axis : bool
        Overlay the rotation axis on each panel.
    """
    n = len(meshes)
    if n == 0:
        raise ValueError("`meshes` must contain at least one mesh")

    values_list: List[np.ndarray] = []
    label = DEFAULT_FIELD_LABELS.get(field, field)
    for m in meshes:
        _, _, values, label = compute_pulsation_scalar(
            m, field=field, up_axis=up_axis
        )
        values_list.append(values)

    norm, _ = _shared_norm(values_list, field, vmin, vmax, symmetric)
    cmap_name = _cmap_for_field(field, cmap)
    cmap_obj = mpl.colormaps[cmap_name]

    if n_cols is None:
        if n <= 4:
            n_cols = n
        elif n <= 6:
            n_cols = 3
        else:
            n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (3.5 * n_cols, 3.5 * n_rows + (0.8 if colorbar else 0.0))

    fig = plt.figure(figsize=figsize)
    axes_list: List[plt.Axes] = []

    ref_mesh = meshes[0]
    center = _mesh_center(ref_mesh)
    axes_lim = 1.5 * float(ref_mesh.radius)

    for i, (m, values) in enumerate(zip(meshes, values_list)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        triangles = _as_np(m.mesh_elements)
        face_colors = cmap_obj(norm(values))
        poly = Poly3DCollection(triangles, facecolors=face_colors,
                                edgecolors="none", linewidths=0, shade=False)
        ax.add_collection3d(poly)

        _set_axes_centered(ax, center, axes_lim)
        if view_angles is not None:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])

        if draw_rotation_axis:
            _, _, rot_start, rot_vec = _single_mesh_arrow_vectors(m)
            ax.quiver(*rot_start, *rot_vec, color="black", linewidth=1.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        if phases is not None:
            panel_title = f"{phase_label} = {float(phases[i]):.3f}"
        else:
            panel_title = f"frame {i + 1}/{n}"
        ax.set_title(panel_title, fontsize=10)
        axes_list.append(ax)

    if colorbar:
        mappable = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=axes_list, shrink=0.7,
                            orientation="horizontal", pad=0.04)
        cbar.set_label(label)

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig, axes_list


# ---------------------------------------------------------------------------
# Public: comet overlay
# ---------------------------------------------------------------------------

def plot_pulsation_comet(meshes: Sequence["MeshModel"],
                         field: str = "v_r",
                         *,
                         up_axis: Optional[np.ndarray] = None,
                         cmap: Optional[str] = None,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None,
                         symmetric: Optional[bool] = None,
                         alpha_range: Tuple[float, float] = (0.08, 0.4),
                         alpha_schedule: Optional[Sequence[float]] = None,
                         view_angles: Optional[Tuple[float, float]] = None,
                         figsize: Tuple[float, float] = (8.0, 8.0),
                         axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                         title: Optional[str] = None,
                         colorbar: bool = True,
                         draw_rotation_axis: bool = False
                         ) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay a fading trail of preceding-phase meshes behind the current mesh.

    ``meshes[-1]`` is rendered solid; earlier meshes fade back linearly between
    ``alpha_range``. Readable up to about :math:`\\ell = 3`; above that the
    overlay gets busy and :func:`plot_pulsation_phase_grid` is the better tool.
    """
    n = len(meshes)
    if n == 0:
        raise ValueError("`meshes` must contain at least one mesh")

    values_list: List[np.ndarray] = []
    label = DEFAULT_FIELD_LABELS.get(field, field)
    for m in meshes:
        _, _, values, label = compute_pulsation_scalar(
            m, field=field, up_axis=up_axis
        )
        values_list.append(values)

    norm, _ = _shared_norm(values_list, field, vmin, vmax, symmetric)
    cmap_name = _cmap_for_field(field, cmap)
    cmap_obj = mpl.colormaps[cmap_name]

    if alpha_schedule is None:
        if n == 1:
            alphas = [1.0]
        else:
            a_lo, a_hi = alpha_range
            trail = np.linspace(a_lo, a_hi, n - 1)
            alphas = list(trail) + [1.0]
    else:
        if len(alpha_schedule) != n:
            raise ValueError(
                f"alpha_schedule length {len(alpha_schedule)} != n meshes {n}"
            )
        alphas = [float(a) for a in alpha_schedule]

    if axes is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = axes

    for m, values, alpha in zip(meshes, values_list, alphas):
        triangles = _as_np(m.mesh_elements)
        face_colors = cmap_obj(norm(values))
        face_colors[:, 3] = float(alpha)
        poly = Poly3DCollection(triangles, facecolors=face_colors,
                                edgecolors="none", linewidths=0, shade=False)
        ax.add_collection3d(poly)

    ref_mesh = meshes[-1]
    center = _mesh_center(ref_mesh)
    axes_lim = 1.5 * float(ref_mesh.radius)
    _set_axes_centered(ax, center, axes_lim)
    ax.set_xlabel(r"$X\,[R_\odot]$")
    ax.set_ylabel(r"$Y\,[R_\odot]$")
    ax.set_zlabel(r"$Z\,[R_\odot]$")
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])

    if draw_rotation_axis:
        _, _, rot_start, rot_vec = _single_mesh_arrow_vectors(ref_mesh)
        ax.quiver(*rot_start, *rot_vec, color="black", linewidth=2.0,
                  label="Rotation axis")
        ax.legend(loc="upper left")

    if colorbar:
        mappable = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.12)
        cbar.set_label(label)
    ax.set_title(title or label)
    return fig, ax


# ---------------------------------------------------------------------------
# Public: phase animation
# ---------------------------------------------------------------------------

def animate_pulsation_phase(meshes: Sequence["MeshModel"],
                            field: str = "v_los",
                            projection: str = "mollweide",
                            *,
                            timestamps: Optional[Sequence[float]] = None,
                            timestamp_label: str = "t",
                            up_axis: Optional[np.ndarray] = None,
                            cmap: Optional[str] = None,
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            symmetric: Optional[bool] = None,
                            marker_size: float = 12.0,
                            fps: int = 20,
                            save_path: Optional[str] = None
                            ) -> FuncAnimation:
    """Animate a per-element pulsation scalar over phase.

    The colour scale is fixed across all frames.  Cell positions are updated
    every frame so deformed meshes (with nonzero pulsation offsets) animate
    correctly.
    """
    if len(meshes) == 0:
        raise ValueError("`meshes` must contain at least one mesh")

    label = DEFAULT_FIELD_LABELS[field]
    frame_coords: List[Tuple[np.ndarray, np.ndarray]] = []
    frame_values: List[np.ndarray] = []
    for m in meshes:
        theta, phi, values, label = compute_pulsation_scalar(
            m, field=field, up_axis=up_axis
        )
        frame_coords.append((theta, phi))
        frame_values.append(values)
    stack = np.stack(frame_values, axis=0)

    if symmetric is None:
        symmetric = field in _DIVERGING_FIELDS
    if vmin is None and vmax is None:
        if symmetric:
            amp = float(np.nanmax(np.abs(stack)))
            if amp == 0.0 or not np.isfinite(amp):
                amp = 1.0
            vmin, vmax = -amp, amp
        else:
            vmin = float(np.nanmin(stack))
            vmax = float(np.nanmax(stack))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_name = _cmap_for_field(field, cmap)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, **_projection_kwargs(projection))

    theta0, phi0 = frame_coords[0]
    x0, y0 = _lonlat_for_projection(theta0, phi0, projection)
    sc = ax.scatter(x0, y0, c=stack[0], cmap=cmap_name, norm=norm,
                    s=marker_size, edgecolors="none")

    if projection.lower() in _FLAT_PROJECTIONS:
        ax.grid(True, linestyle=":", linewidth=0.5, color="grey")
    else:
        _setup_rect_axes(ax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08,
                        orientation="horizontal")
    cbar.set_label(label)
    title = ax.set_title(label)

    def _update(frame_idx: int):
        theta_f, phi_f = frame_coords[frame_idx]
        xf, yf = _lonlat_for_projection(theta_f, phi_f, projection)
        sc.set_offsets(np.column_stack([xf, yf]))
        sc.set_array(stack[frame_idx])
        if timestamps is not None:
            title.set_text(f"{label}  ({timestamp_label}={timestamps[frame_idx]:.3f})")
        else:
            title.set_text(f"{label}  (frame {frame_idx + 1}/{len(meshes)})")
        return (sc, title)

    anim = FuncAnimation(fig, _update, frames=len(meshes),
                         interval=1000 // max(fps, 1), blit=False)

    if save_path is not None:
        smart_save(anim, save_path, fps=fps)

    return anim


# ---------------------------------------------------------------------------
# Public: observed-disk animation (LOS-projected)
# ---------------------------------------------------------------------------

def _disk_basis(los_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n_hat, v1, v2)`` where ``(v1, v2)`` span the plane perpendicular to LOS."""
    n = _unit(_as_np(los_vector))
    seed = np.array([0.0, 0.0, 1.0]) if abs(np.dot(n, [0.0, 0.0, 1.0])) < 0.99 \
        else np.array([1.0, 0.0, 0.0])
    v1 = _unit(np.cross(n, seed))
    v2 = np.cross(n, v1)
    return n, v1, v2


def _project_to_disk(mesh: "MeshModel",
                     v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Project cell centres into the LOS-normal plane, re-centred on the star."""
    d_centers = _as_np(mesh.d_centers)
    offsets = _as_np(mesh.center_pulsation_offsets)
    pos = d_centers + offsets
    return np.stack([pos @ v1, pos @ v2], axis=1)


ValueFn = Callable[["MeshModel"], Tuple[np.ndarray, str]]


def animate_observed_disk(meshes: Sequence["MeshModel"],
                          field: str = "v_los",
                          *,
                          value_fn: Optional[ValueFn] = None,
                          up_axis: Optional[np.ndarray] = None,
                          cmap: Optional[str] = None,
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          symmetric: Optional[bool] = None,
                          marker_size: float = 14.0,
                          timestamps: Optional[Sequence[float]] = None,
                          timestamp_label: str = "t",
                          draw_limb: bool = True,
                          draw_rotation_axis: bool = False,
                          fps: int = 20,
                          figsize: Tuple[float, float] = (6.5, 6.5),
                          save_path: Optional[str] = None
                          ) -> FuncAnimation:
    """Animate the star as seen by the observer: disk projected along ``mesh.los_vector``.

    Each frame projects the visible hemisphere (``mesh.mus > 0``) onto the plane
    perpendicular to ``mesh.los_vector`` and colours cells by a scalar — by
    default the pulsation LOS velocity. Hidden cells are dropped per frame. The
    colour scale is fixed across the sequence.

    Parameters
    ----------
    meshes : sequence of MeshModel
        One mesh per frame, each with pulsations already evaluated.
    field : str
        Pulsation field from :data:`PULSATION_FIELDS` used for colour. Ignored
        if ``value_fn`` is given.
    value_fn : callable, optional
        ``value_fn(mesh) -> (values, label)``; escape hatch for scalars outside
        the pulsation field registry (e.g. ``mesh.mus`` or
        ``mesh.los_velocities``).
    up_axis : array-like, optional
        Passed through to :func:`compute_pulsation_scalar`.
    symmetric : bool, optional
        Force a diverging zero-centred colour scale. Defaults to ``True`` for
        diverging fields and when ``value_fn`` is used with signed values.
    draw_limb : bool
        If ``True``, draw a reference circle at ``mesh.radius`` (the first
        mesh's radius is used).
    draw_rotation_axis : bool
        If ``True``, overlay the projected rotation axis as a dashed line.
    save_path : str, optional
        If given, persist the animation via :func:`smart_save`.
    """
    if len(meshes) == 0:
        raise ValueError("`meshes` must contain at least one mesh")

    # --- frame data --------------------------------------------------------
    los_ref = _as_np(meshes[0].los_vector)
    _, v1, v2 = _disk_basis(los_ref)

    frame_xy: List[np.ndarray] = []
    frame_values: List[np.ndarray] = []
    frame_mask: List[np.ndarray] = []
    label = DEFAULT_FIELD_LABELS.get(field, field)

    for m in meshes:
        if value_fn is not None:
            values, label = value_fn(m)
            values = _as_np(values)
        else:
            _, _, values, label = compute_pulsation_scalar(
                m, field=field, up_axis=up_axis
            )
        mu = _as_np(m.mus)
        mask = mu > 0
        frame_mask.append(mask)
        frame_values.append(values)
        frame_xy.append(_project_to_disk(m, v1, v2))

    visible_values = np.concatenate(
        [v[mk] for v, mk in zip(frame_values, frame_mask)]
    ) if frame_values else np.zeros(0)

    if symmetric is None:
        symmetric = (value_fn is None and field in _DIVERGING_FIELDS)
    if vmin is None and vmax is None:
        if symmetric and visible_values.size:
            amp = float(np.nanmax(np.abs(visible_values)))
            if amp == 0.0 or not np.isfinite(amp):
                amp = 1.0
            vmin, vmax = -amp, amp
        elif visible_values.size:
            vmin = float(np.nanmin(visible_values))
            vmax = float(np.nanmax(visible_values))
            if vmin == vmax:
                vmax = vmin + 1e-9
        else:
            vmin, vmax = -1.0, 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_name = _cmap_for_field(field if value_fn is None else "speed", cmap) \
        if not symmetric else _cmap_for_field("v_los", cmap)

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    radius = float(meshes[0].radius)
    lim = 1.15 * radius
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"disk $x\,[R_\odot]$")
    ax.set_ylabel(r"disk $y\,[R_\odot]$")

    if draw_limb:
        ax.add_patch(plt.Circle((0, 0), radius, fill=False,
                                edgecolor="grey", linewidth=0.8,
                                linestyle="--", zorder=0))

    if draw_rotation_axis:
        rot = _unit(_as_np(meshes[0].rotation_axis))
        rx, ry = float(rot @ v1), float(rot @ v2)
        ax.plot([-1.1 * radius * rx, 1.1 * radius * rx],
                [-1.1 * radius * ry, 1.1 * radius * ry],
                color="black", linewidth=0.8, linestyle=":", zorder=1)

    mask0 = frame_mask[0]
    xy0 = frame_xy[0][mask0]
    vals0 = frame_values[0][mask0]
    sc = ax.scatter(xy0[:, 0], xy0[:, 1], c=vals0, cmap=cmap_name, norm=norm,
                    s=marker_size, edgecolors="none", zorder=2)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label(label)
    title = ax.set_title(label)

    def _update(frame_idx: int):
        mask = frame_mask[frame_idx]
        xy = frame_xy[frame_idx][mask]
        sc.set_offsets(xy)
        sc.set_array(frame_values[frame_idx][mask])
        if timestamps is not None:
            title.set_text(f"{label}  ({timestamp_label}={timestamps[frame_idx]:.3f})")
        else:
            title.set_text(f"{label}  (frame {frame_idx + 1}/{len(meshes)})")
        return (sc, title)

    anim = FuncAnimation(fig, _update, frames=len(meshes),
                         interval=1000 // max(fps, 1), blit=False)

    if save_path is not None:
        smart_save(anim, save_path, fps=fps)

    return anim
