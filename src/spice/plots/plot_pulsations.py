"""Reusable plotting helpers for stellar pulsation velocity fields.

These functions operate on a :class:`spice.models.MeshModel` whose pulsation
state has already been evaluated (typically by
``spice.models.mesh_transform.evaluate_pulsations(mesh, t)``).  They produce
physically meaningful scalar diagnostics of the surface velocity field and
render them via flat surface projections, streamlines, cross-section cuts,
sparse-arrow overlays, or phase animations.

Design goals
------------
* **Basis-aware.**  Everything is computed in the local spherical frame
  ``(r_hat, theta_hat, phi_hat)`` defined relative to a user-chosen "up"
  axis (default: the mesh rotation axis), so nodal patterns of
  :math:`Y_\\ell^m` modes line up as expected regardless of how the star is
  tilted in the simulation frame.
* **Scalar first.**  Horizontal divergence and radial vorticity are computed
  from the resampled :math:`v_\\theta, v_\\phi` grid with the correct sphere
  metric.  They cleanly separate spheroidal modes (divergence-dominated) from
  toroidal modes (curl-dominated).
* **No new dependencies.**  Unstructured mesh samples are resampled onto a
  regular ``(\\theta, \\phi)`` grid via geodesic inverse-distance weighting in
  pure NumPy.  Only NumPy and Matplotlib are required at runtime.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

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

#: Built-in scalar field names understood by :func:`compute_pulsation_field`
#: and the plotting functions that wrap it.
PULSATION_FIELDS: Tuple[str, ...] = (
    "v_r",
    "v_theta",
    "v_phi",
    "v_h",
    "speed",
    "v_los",
    "div_h",
    "curl_r",
)

DEFAULT_FIELD_LABELS = {
    "v_r": r"$v_r$ [km/s]",
    "v_theta": r"$v_\theta$ [km/s]",
    "v_phi": r"$v_\phi$ [km/s]",
    "v_h": r"$|v_h|$ [km/s]",
    "speed": r"$|v|$ [km/s]",
    "v_los": r"$v_\mathrm{LOS}$ [km/s]",
    "div_h": r"$\nabla_h \cdot v_h$ [(km/s)/R]",
    "curl_r": r"$(\nabla \times v)_r$ [(km/s)/R]",
}

#: Fields for which a diverging colormap centred on zero is the right default.
_DIVERGING_FIELDS = frozenset({"v_r", "v_theta", "v_phi", "v_los", "div_h", "curl_r"})

_DEFAULT_DIVERGING_CMAP = "cmr.redshift"
_DEFAULT_MAGNITUDE_CMAP = DEFAULT_CMAP  # cmr.bubblegum from plot_mesh


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    """Convert a jax / torch / numpy-ish array to a numpy array."""
    return np.asarray(x, dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    """Safely normalise a 1-D vector (returns zeros if the norm is ~0)."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _orthonormal_basis_from_axis(axis: np.ndarray) -> np.ndarray:
    """Build a right-handed 3x3 rotation matrix whose third column is ``axis``.

    The returned matrix ``R`` maps simulation-frame Cartesian vectors to a
    local "axis-up" frame via ``v_local = R.T @ v_sim``.  Concretely,
    ``R[:, 2] = axis``, while the first two columns span the plane
    perpendicular to it.
    """
    z = _unit(_as_np(axis))
    # pick a seed direction not parallel to z
    seed = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = _unit(np.cross(seed, z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # columns = basis vectors


def _spherical_components(positions: np.ndarray,
                          velocities: np.ndarray,
                          up_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """Project per-element Cartesian velocities into the local spherical frame.

    Parameters
    ----------
    positions : ndarray, shape (n, 3)
        Position vectors (relative to the sphere centre).
    velocities : ndarray, shape (n, 3)
        Velocity vectors in the same Cartesian frame as ``positions``.
    up_axis : ndarray, shape (3,)
        Axis to use as the "pole".  Colatitude ``theta`` is measured from
        this axis.

    Returns
    -------
    theta : ndarray, shape (n,)
        Colatitude in radians, in [0, pi].
    phi : ndarray, shape (n,)
        Azimuth in radians, in (-pi, pi].
    v_r, v_theta, v_phi : ndarray, shape (n,)
        Velocity components expressed in the local spherical basis
        ``(r_hat, theta_hat, phi_hat)``.  ``theta_hat`` points from the pole
        toward the equator; ``phi_hat`` points eastward.
    """
    R = _orthonormal_basis_from_axis(up_axis)
    pos_local = positions @ R        # (n, 3)
    vel_local = velocities @ R

    r = np.linalg.norm(pos_local, axis=1)
    r_safe = np.where(r > 1e-12, r, 1e-12)
    x, y, z = pos_local[:, 0], pos_local[:, 1], pos_local[:, 2]

    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))  # 0 at pole
    phi = np.arctan2(y, x)

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    # Local spherical basis vectors in the rotated frame
    r_hat = np.stack([sin_t * cos_p, sin_t * sin_p, cos_t], axis=1)
    theta_hat = np.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], axis=1)
    phi_hat = np.stack([-sin_p, cos_p, np.zeros_like(phi)], axis=1)

    v_r = np.sum(vel_local * r_hat, axis=1)
    v_theta = np.sum(vel_local * theta_hat, axis=1)
    v_phi = np.sum(vel_local * phi_hat, axis=1)

    return theta, phi, v_r, v_theta, v_phi


# ---------------------------------------------------------------------------
# Resampling onto a regular (theta, phi) grid
# ---------------------------------------------------------------------------

def _make_theta_phi_grid(n_theta: int, n_phi: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return open 1-D grids ``theta`` and ``phi``.

    ``theta`` avoids the exact poles (where divergence/curl formulas diverge)
    and ``phi`` spans ``(-pi, pi]``.
    """
    # Stay a half-cell away from the poles so metric factors stay finite.
    theta = np.linspace(0.0, np.pi, n_theta + 2)[1:-1]
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    return theta, phi


def _idw_resample(theta_s: np.ndarray, phi_s: np.ndarray, values: np.ndarray,
                  theta_grid: np.ndarray, phi_grid: np.ndarray,
                  k: int = 8, power: float = 2.0) -> np.ndarray:
    """Resample scalar samples on the sphere onto a regular grid.

    Uses inverse-distance weighting over the ``k`` nearest mesh samples in
    *geodesic* (great-circle) distance.  Vectorised in NumPy; no extra deps.

    Parameters
    ----------
    theta_s, phi_s : ndarray, shape (n,)
        Sample spherical coordinates (colatitude, azimuth).
    values : ndarray, shape (n,) or (n, k_channels)
        Scalar (or multi-channel) values at each sample.
    theta_grid, phi_grid : ndarray
        1-D target grids.  The output has shape
        ``(len(theta_grid), len(phi_grid))`` — or ``(..., k_channels)`` for
        multi-channel inputs.
    k : int
        Number of nearest neighbours to use.
    power : float
        Inverse-distance exponent.  Larger values favour the closest sample.
    """
    values = np.asarray(values)
    single_channel = values.ndim == 1
    if single_channel:
        values = values[:, None]

    # Cartesian unit vectors for samples and grid (dot product = cos(angle))
    sx = np.sin(theta_s) * np.cos(phi_s)
    sy = np.sin(theta_s) * np.sin(phi_s)
    sz = np.cos(theta_s)
    samples_xyz = np.stack([sx, sy, sz], axis=1)       # (n, 3)

    T, P = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    gx = np.sin(T) * np.cos(P)
    gy = np.sin(T) * np.sin(P)
    gz = np.cos(T)
    grid_xyz = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (m, 3)

    # Dot products -> cosine of great-circle angle; angle = arccos(clip).
    # For IDW we only need a monotonic "distance", so 1 - cos is enough and
    # is safer numerically than arccos near zero.
    dots = grid_xyz @ samples_xyz.T                     # (m, n)
    dist2 = np.clip(1.0 - dots, 0.0, 2.0)               # ~theta^2/2 for small
    k = min(k, dist2.shape[1])
    nn_idx = np.argpartition(dist2, k - 1, axis=1)[:, :k]

    row_idx = np.arange(dist2.shape[0])[:, None]
    nn_d = dist2[row_idx, nn_idx]                        # (m, k)
    nn_v = values[nn_idx]                                # (m, k, c)

    eps = 1e-12
    weights = 1.0 / np.power(nn_d + eps, 0.5 * power)    # d in 1-cos ~ theta^2
    wsum = weights.sum(axis=1, keepdims=True)
    out = np.einsum("mk,mkc->mc", weights, nn_v) / wsum  # (m, c)

    grid_shape = (theta_grid.size, phi_grid.size, values.shape[1])
    out = out.reshape(grid_shape)
    if single_channel:
        out = out[..., 0]
    return out


# ---------------------------------------------------------------------------
# Differential operators on a regular (theta, phi) grid
# ---------------------------------------------------------------------------

def _horizontal_divergence(v_theta: np.ndarray, v_phi: np.ndarray,
                           theta: np.ndarray, phi: np.ndarray,
                           radius: float = 1.0) -> np.ndarray:
    """Horizontal divergence on the sphere.

    .. math::

        \\nabla_h \\cdot v_h = \\frac{1}{r\\sin\\theta}
            \\left[ \\partial_\\theta(\\sin\\theta\\, v_\\theta) +
                    \\partial_\\phi v_\\phi \\right].

    ``v_theta`` and ``v_phi`` must be shape ``(len(theta), len(phi))`` and the
    phi axis is assumed periodic.
    """
    sin_t = np.sin(theta)[:, None]
    # sin(theta) * v_theta differentiated w.r.t. theta
    sinvt = sin_t * v_theta
    dsinvt_dtheta = np.gradient(sinvt, theta, axis=0)

    # v_phi differentiated w.r.t. phi (periodic — wrap for the ends)
    v_phi_wrapped = np.concatenate([v_phi[:, -1:], v_phi, v_phi[:, :1]], axis=1)
    phi_step = phi[1] - phi[0]
    dvp_dphi = (v_phi_wrapped[:, 2:] - v_phi_wrapped[:, :-2]) / (2.0 * phi_step)

    return (dsinvt_dtheta + dvp_dphi) / (radius * sin_t)


def _radial_vorticity(v_theta: np.ndarray, v_phi: np.ndarray,
                      theta: np.ndarray, phi: np.ndarray,
                      radius: float = 1.0) -> np.ndarray:
    """Radial component of the curl.

    .. math::

        (\\nabla \\times v)_r = \\frac{1}{r\\sin\\theta}
            \\left[ \\partial_\\theta(\\sin\\theta\\, v_\\phi) -
                    \\partial_\\phi v_\\theta \\right].
    """
    sin_t = np.sin(theta)[:, None]
    sinvp = sin_t * v_phi
    dsinvp_dtheta = np.gradient(sinvp, theta, axis=0)

    v_theta_wrapped = np.concatenate([v_theta[:, -1:], v_theta, v_theta[:, :1]], axis=1)
    phi_step = phi[1] - phi[0]
    dvt_dphi = (v_theta_wrapped[:, 2:] - v_theta_wrapped[:, :-2]) / (2.0 * phi_step)

    return (dsinvp_dtheta - dvt_dphi) / (radius * sin_t)


# ---------------------------------------------------------------------------
# Top-level field computation
# ---------------------------------------------------------------------------

def _resolve_up_axis(mesh: "MeshModel",
                     up_axis: Optional[np.ndarray] = None) -> np.ndarray:
    if up_axis is not None:
        return _unit(_as_np(up_axis))
    return _unit(_as_np(mesh.rotation_axis))


def compute_pulsation_field(mesh: "MeshModel",
                            field: str = "div_h",
                            n_theta: int = 121,
                            n_phi: int = 241,
                            up_axis: Optional[np.ndarray] = None,
                            k_neighbors: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Compute a pulsation scalar field on a regular ``(theta, phi)`` grid.

    Parameters
    ----------
    mesh : MeshModel
        A mesh with ``pulsation_velocities`` already populated (call
        ``evaluate_pulsations`` first).
    field : str
        One of :data:`PULSATION_FIELDS`.
    n_theta, n_phi : int
        Grid resolution.  Defaults give ~1.5 deg cells which is plenty for
        low-order modes.
    up_axis : array-like, shape (3,), optional
        Direction used as the "pole" when defining the spherical frame.
        Defaults to the mesh rotation axis.
    k_neighbors : int
        Number of mesh samples used in the inverse-distance resampling.

    Returns
    -------
    theta_grid, phi_grid : ndarray
        1-D coordinate arrays (radians).
    values : ndarray, shape (n_theta, n_phi)
        The requested scalar field on the grid.
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

    theta_s, phi_s, v_r, v_theta, v_phi = _spherical_components(
        positions, velocities, axis
    )
    theta_grid, phi_grid = _make_theta_phi_grid(n_theta, n_phi)

    # For derivative-based fields we need gridded v_theta and v_phi first.
    if field in ("div_h", "curl_r"):
        stacked = np.stack([v_theta, v_phi], axis=1)           # (n, 2)
        gridded = _idw_resample(theta_s, phi_s, stacked,
                                theta_grid, phi_grid, k=k_neighbors)
        v_theta_grid = gridded[..., 0]
        v_phi_grid = gridded[..., 1]
        if field == "div_h":
            values = _horizontal_divergence(v_theta_grid, v_phi_grid,
                                            theta_grid, phi_grid, radius=1.0)
        else:
            values = _radial_vorticity(v_theta_grid, v_phi_grid,
                                       theta_grid, phi_grid, radius=1.0)
        return theta_grid, phi_grid, values, DEFAULT_FIELD_LABELS[field]

    if field == "v_r":
        scalar = v_r
    elif field == "v_theta":
        scalar = v_theta
    elif field == "v_phi":
        scalar = v_phi
    elif field == "v_h":
        scalar = np.sqrt(v_theta ** 2 + v_phi ** 2)
    elif field == "speed":
        scalar = np.sqrt(v_r ** 2 + v_theta ** 2 + v_phi ** 2)
    elif field == "v_los":
        los = _unit(_as_np(mesh.los_vector))
        scalar = velocities @ los
    else:  # pragma: no cover — guarded above
        raise ValueError(field)

    values = _idw_resample(theta_s, phi_s, scalar,
                           theta_grid, phi_grid, k=k_neighbors)
    return theta_grid, phi_grid, values, DEFAULT_FIELD_LABELS[field]


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _projection_kwargs(projection: str) -> dict:
    proj = projection.lower()
    if proj in ("mollweide", "hammer", "aitoff", "lambert"):
        return {"projection": proj}
    if proj in ("rect", "rectangular", "equirectangular", "plate_carree", "none"):
        return {}
    raise ValueError(
        f"Unknown projection {projection!r}; expected 'mollweide', 'hammer', "
        f"'aitoff', 'lambert' or 'rect'"
    )


def _pcolormesh_on_projection(ax: plt.Axes, theta_grid: np.ndarray,
                              phi_grid: np.ndarray, values: np.ndarray,
                              projection: str, cmap: str,
                              norm: mpl.colors.Normalize) -> mpl.collections.QuadMesh:
    """Plot a scalar field on a projection axis.

    Converts (theta, phi) → (longitude, latitude) for the built-in
    matplotlib sphere projections, or leaves it in degrees on a rectangular
    axis for the ``'rect'`` mode.
    """
    proj = projection.lower()
    # Matplotlib's projections expect x=longitude in radians in (-pi, pi]
    # and y=latitude in radians in (-pi/2, pi/2].
    if proj in ("mollweide", "hammer", "aitoff", "lambert"):
        lat = 0.5 * np.pi - theta_grid
        lon = phi_grid
        Lon, Lat = np.meshgrid(lon, lat)
        return ax.pcolormesh(Lon, Lat, values, cmap=cmap, norm=norm,
                             shading="auto")
    # rectangular plot in degrees
    lat = np.rad2deg(0.5 * np.pi - theta_grid)
    lon = np.rad2deg(phi_grid)
    Lon, Lat = np.meshgrid(lon, lat)
    mesh = ax.pcolormesh(Lon, Lat, values, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    return mesh


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


# ---------------------------------------------------------------------------
# Public: 2D surface maps
# ---------------------------------------------------------------------------

def plot_pulsation_map(mesh: "MeshModel",
                       field: str = "div_h",
                       projection: str = "mollweide",
                       *,
                       n_theta: int = 121,
                       n_phi: int = 241,
                       up_axis: Optional[np.ndarray] = None,
                       cmap: Optional[str] = None,
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       overlay_quiver: bool = False,
                       quiver_stride: int = 12,
                       quiver_color: str = "white",
                       title: Optional[str] = None,
                       axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                       update_colorbar: bool = True
                       ) -> Tuple[plt.Figure, plt.Axes]:
    """Render a pulsation diagnostic as a flat surface map.

    Handles technique #1 (scalar diagnostics), #2 (Mollweide/rect surface
    maps), #5 (basis-aligned components — pick ``field='v_theta'`` etc.) and
    #7 (dense scalar + optional sparse quiver overlay) from the user's
    wishlist.

    Parameters
    ----------
    mesh : MeshModel
        Mesh with pulsations already evaluated.
    field : str
        One of :data:`PULSATION_FIELDS`.  ``'div_h'`` is the best default for
        spheroidal modes; ``'curl_r'`` for toroidal.
    projection : str
        ``'mollweide'``, ``'hammer'``, ``'aitoff'``, ``'lambert'`` or
        ``'rect'`` for a plain equirectangular axis.
    overlay_quiver : bool
        If ``True``, overlay the horizontal velocity field as sparse arrows
        on top of the scalar map.  Uses every ``quiver_stride``-th grid node.
    quiver_stride : int
        Stride used when subsampling grid nodes for the overlay.
    axes : (Figure, Axes), optional
        Reuse an existing figure/axis.  Pass ``None`` to create fresh ones.

    Returns
    -------
    fig, ax : tuple
    """
    theta_grid, phi_grid, values, label = compute_pulsation_field(
        mesh, field=field, n_theta=n_theta, n_phi=n_phi, up_axis=up_axis
    )
    cmap_name = _cmap_for_field(field, cmap)
    norm = _norm_for_field(field, values, vmin, vmax)

    if axes is None:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111, **_projection_kwargs(projection))
    else:
        fig, ax = axes

    pcm = _pcolormesh_on_projection(ax, theta_grid, phi_grid, values,
                                    projection, cmap_name, norm)

    if overlay_quiver:
        # Recompute the horizontal components on the same grid for arrows.
        _, _, v_th_grid, v_ph_grid = _horizontal_components_on_grid(
            mesh, theta_grid, phi_grid, up_axis=up_axis
        )
        _overlay_quiver(ax, theta_grid, phi_grid, v_th_grid, v_ph_grid,
                        projection, stride=quiver_stride, color=quiver_color)

    ax.set_title(title or label)
    if projection.lower() in ("mollweide", "hammer", "aitoff", "lambert"):
        ax.grid(True, linestyle=":", linewidth=0.5, color="grey")

    if update_colorbar:
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.75, pad=0.08,
                            orientation="horizontal")
        cbar.set_label(label)

    return fig, ax


def _horizontal_components_on_grid(mesh: "MeshModel",
                                   theta_grid: np.ndarray,
                                   phi_grid: np.ndarray,
                                   *,
                                   up_axis: Optional[np.ndarray] = None,
                                   k_neighbors: int = 8
                                   ) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """Return (theta_grid, phi_grid, v_theta_grid, v_phi_grid) for ``mesh``.

    Exposed so callers that already have the scalar field computed can reuse
    the grid for overlays without recomputing it from scratch.
    """
    positions = _as_np(mesh.d_centers)
    velocities = _as_np(mesh.pulsation_velocities)
    axis = _resolve_up_axis(mesh, up_axis)
    theta_s, phi_s, _, v_theta, v_phi = _spherical_components(
        positions, velocities, axis
    )
    stacked = np.stack([v_theta, v_phi], axis=1)
    gridded = _idw_resample(theta_s, phi_s, stacked,
                            theta_grid, phi_grid, k=k_neighbors)
    return theta_grid, phi_grid, gridded[..., 0], gridded[..., 1]


def _overlay_quiver(ax: plt.Axes, theta_grid: np.ndarray, phi_grid: np.ndarray,
                    v_theta: np.ndarray, v_phi: np.ndarray,
                    projection: str, *, stride: int, color: str) -> None:
    s = max(int(stride), 1)
    T = theta_grid[::s]
    P = phi_grid[::s]
    VT = v_theta[::s, ::s]
    VP = v_phi[::s, ::s]
    # On a sphere (theta = colatitude), increasing theta means moving toward
    # the equator and then the south pole.  Latitude decreases with theta,
    # so the north-pointing (upward) arrow component is ``-v_theta``.
    Lon, Lat = np.meshgrid(P, 0.5 * np.pi - T)
    proj = projection.lower()
    if proj in ("mollweide", "hammer", "aitoff", "lambert"):
        ax.quiver(Lon, Lat, VP, -VT, color=color,
                  scale_units="xy", angles="xy", pivot="middle", width=0.0025)
    else:
        ax.quiver(np.rad2deg(Lon), np.rad2deg(Lat), VP, -VT, color=color,
                  scale_units="xy", angles="xy", pivot="middle", width=0.0025)


def plot_pulsation_components(mesh: "MeshModel",
                              fields: Sequence[str] = ("div_h", "curl_r",
                                                       "v_r", "v_h"),
                              projection: str = "mollweide",
                              *,
                              n_theta: int = 121,
                              n_phi: int = 241,
                              up_axis: Optional[np.ndarray] = None,
                              figsize: Optional[Tuple[float, float]] = None,
                              cmap_overrides: Optional[dict] = None,
                              suptitle: Optional[str] = None
                              ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot several pulsation diagnostics side-by-side.

    Handy for comparing, e.g., horizontal divergence and radial vorticity on
    the same figure so the spheroidal/toroidal split is obvious at a glance.

    Parameters
    ----------
    fields : sequence of str
        Which fields to plot.  Any value from :data:`PULSATION_FIELDS`.
    """
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
            mesh, field=field, projection=projection,
            n_theta=n_theta, n_phi=n_phi, up_axis=up_axis,
            cmap=cmap_overrides.get(field), axes=(fig, ax),
        )
        axes.append(ax)

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Public: streamlines
# ---------------------------------------------------------------------------

def plot_pulsation_streamlines(mesh: "MeshModel",
                               background: Optional[str] = "div_h",
                               *,
                               n_theta: int = 121,
                               n_phi: int = 241,
                               up_axis: Optional[np.ndarray] = None,
                               density: float = 1.5,
                               linewidth: Union[float, str] = 1.0,
                               stream_color: Optional[str] = None,
                               cmap: Optional[str] = None,
                               title: Optional[str] = None,
                               axes: Optional[Tuple[plt.Figure, plt.Axes]] = None
                               ) -> Tuple[plt.Figure, plt.Axes]:
    """Plot surface streamlines of the horizontal pulsation field.

    Implements technique #3 (streamlines instead of arrows).  Uses
    matplotlib's :func:`~matplotlib.pyplot.streamplot` on an equirectangular
    (longitude × latitude) grid; this shows toroidal "swirling belts" and
    spheroidal source/sink patterns very clearly.

    Notes
    -----
    Streamlines are computed in unweighted equirectangular coordinates, so
    lines near the poles look more closely packed than they would on a true
    sphere.  That is expected and does not affect the qualitative flow
    topology.

    Parameters
    ----------
    background : str or None
        If given, a scalar field drawn underneath the streamlines.  ``None``
        draws streamlines on a blank axis.
    density : float
        Passed through to :func:`streamplot`.
    linewidth : float or ``'speed'``
        Either a constant line width, or ``'speed'`` to scale with local
        horizontal speed.
    stream_color : str or None
        Constant color for the streamlines.  If ``None`` and ``linewidth``
        is ``'speed'``, lines are also coloured by local speed.
    """
    theta_grid, phi_grid = _make_theta_phi_grid(n_theta, n_phi)
    _, _, v_theta_grid, v_phi_grid = _horizontal_components_on_grid(
        mesh, theta_grid, phi_grid, up_axis=up_axis
    )

    if axes is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig, ax = axes

    # Draw optional background field
    if background is not None:
        _, _, bg_values, bg_label = compute_pulsation_field(
            mesh, field=background, n_theta=n_theta, n_phi=n_phi,
            up_axis=up_axis
        )
        norm = _norm_for_field(background, bg_values, None, None)
        pcm = _pcolormesh_on_projection(
            ax, theta_grid, phi_grid, bg_values,
            "rect", _cmap_for_field(background, cmap), norm
        )
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.75, pad=0.08,
                            orientation="horizontal")
        cbar.set_label(bg_label)
    else:
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

    # Latitude increases upward; v_theta (colatitude component) flips sign.
    lat_deg = np.rad2deg(0.5 * np.pi - theta_grid)
    lon_deg = np.rad2deg(phi_grid)
    # streamplot requires strictly increasing 1-D coordinates
    order = np.argsort(lat_deg)
    lat_sorted = lat_deg[order]
    U = v_phi_grid[order, :]
    V = (-v_theta_grid)[order, :]

    speed = np.sqrt(U ** 2 + V ** 2)
    if linewidth == "speed":
        lw = 0.25 + 2.0 * (speed / (np.nanmax(speed) + 1e-12))
    else:
        lw = float(linewidth)

    if stream_color is None:
        color_arg = speed
        stream_cmap = "viridis"
    else:
        color_arg = stream_color
        stream_cmap = None

    ax.streamplot(
        lon_deg, lat_sorted, U, V,
        density=density, linewidth=lw, color=color_arg, cmap=stream_cmap,
    )

    ax.set_title(title or "Horizontal pulsation streamlines")
    return fig, ax


# ---------------------------------------------------------------------------
# Public: cross-section cuts
# ---------------------------------------------------------------------------

def plot_pulsation_cross_section(mesh: "MeshModel",
                                 slice: str = "longitude",
                                 fixed_deg: float = 0.0,
                                 fields: Sequence[str] = ("v_r", "v_theta", "v_phi"),
                                 *,
                                 n_theta: int = 361,
                                 n_phi: int = 721,
                                 up_axis: Optional[np.ndarray] = None,
                                 axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                                 title: Optional[str] = None
                                 ) -> Tuple[plt.Figure, plt.Axes]:
    """1-D cuts through the pulsation field.

    Implements technique #8 (cross-sections / latitude slices).

    Parameters
    ----------
    slice : {'longitude', 'latitude'}
        ``'longitude'``: hold phi fixed, plot field(theta).  Good for seeing
        meridional (ell-dependent) structure.
        ``'latitude'``: hold theta fixed (via latitude), plot field(phi).
        Good for seeing m-dependent azimuthal structure.
    fixed_deg : float
        Longitude (for ``slice='longitude'``) or latitude (for
        ``slice='latitude'``) at which to take the cut, in degrees.
    fields : sequence of str
        Which scalar fields to plot on the same axis.
    """
    theta_grid, phi_grid = _make_theta_phi_grid(n_theta, n_phi)

    if axes is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig, ax = axes

    for field in fields:
        _, _, values, label = compute_pulsation_field(
            mesh, field=field, n_theta=n_theta, n_phi=n_phi, up_axis=up_axis
        )
        if slice == "longitude":
            # Pick the column closest to requested phi.
            phi_target = np.deg2rad(fixed_deg)
            idx = int(np.argmin(np.abs(phi_grid - phi_target)))
            lat_deg = np.rad2deg(0.5 * np.pi - theta_grid)
            ax.plot(lat_deg, values[:, idx], label=label)
            ax.set_xlabel("latitude [deg]")
        elif slice == "latitude":
            theta_target = np.deg2rad(90.0 - fixed_deg)
            idx = int(np.argmin(np.abs(theta_grid - theta_target)))
            lon_deg = np.rad2deg(phi_grid)
            ax.plot(lon_deg, values[idx, :], label=label)
            ax.set_xlabel("longitude [deg]")
        else:
            raise ValueError(
                f"slice must be 'longitude' or 'latitude', got {slice!r}"
            )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_ylabel("velocity / diagnostic")
    cut_name = "phi" if slice == "longitude" else "lat"
    ax.set_title(title or f"Pulsation cross section at {cut_name} = {fixed_deg:g} deg")
    ax.legend(loc="best", fontsize=9)
    return fig, ax


# ---------------------------------------------------------------------------
# Public: 3D sphere with scalar colour + sparse arrows
# ---------------------------------------------------------------------------

def plot_pulsation_3D_sparse(mesh: "MeshModel",
                             scalar: str = "div_h",
                             arrow_stride: int = 10,
                             *,
                             n_theta: int = 121,
                             n_phi: int = 241,
                             up_axis: Optional[np.ndarray] = None,
                             cmap: Optional[str] = None,
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             arrow_scale: float = 0.35,
                             arrow_color: str = "white",
                             axes: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                             title: Optional[str] = None,
                             draw_los_vector: bool = True,
                             draw_rotation_axis: bool = True
                             ) -> Tuple[plt.Figure, plt.Axes]:
    """3D sphere coloured by a scalar field with a sparse quiver overlay.

    Implements technique #7 (dense scalar + sparse arrows) while keeping the
    3D geometry that some audiences prefer over a flat map.  Colours come
    from ``scalar`` (computed on a regular grid), arrows are the raw
    per-element ``pulsation_velocities`` subsampled by ``arrow_stride``.
    """
    positions = _as_np(mesh.d_centers)
    velocities = _as_np(mesh.pulsation_velocities)
    center = _mesh_center(mesh)
    radius = float(mesh.radius)

    theta_grid, phi_grid, scalar_grid, label = compute_pulsation_field(
        mesh, field=scalar, n_theta=n_theta, n_phi=n_phi, up_axis=up_axis
    )
    cmap_name = _cmap_for_field(scalar, cmap)
    norm = _norm_for_field(scalar, scalar_grid, vmin, vmax)

    # Build a rotated (theta, phi) -> sim-frame Cartesian grid so the colours
    # wrap correctly even when the pulsation frame is tilted.
    R = _orthonormal_basis_from_axis(_resolve_up_axis(mesh, up_axis))
    T, P = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    sx = np.sin(T) * np.cos(P)
    sy = np.sin(T) * np.sin(P)
    sz = np.cos(T)
    local_xyz = np.stack([sx, sy, sz], axis=-1)         # (nt, np, 3)
    sim_xyz = local_xyz @ R.T                            # back to sim frame
    Xs = center[0] + radius * sim_xyz[..., 0]
    Ys = center[1] + radius * sim_xyz[..., 1]
    Zs = center[2] + radius * sim_xyz[..., 2]

    if axes is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = axes

    face_colors = mpl.colormaps[cmap_name](norm(scalar_grid))
    ax.plot_surface(Xs, Ys, Zs, facecolors=face_colors,
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)

    # Sparse arrow overlay
    n_cells = positions.shape[0]
    if arrow_stride > 1:
        idx = np.arange(0, n_cells, int(arrow_stride))
    else:
        idx = np.arange(n_cells)
    if idx.size > 0:
        # Arrow starts at the stellar surface (not the mesh centre), arrows
        # are rescaled so the longest one has length ``arrow_scale * radius``.
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
                            n_theta: int = 121,
                            n_phi: int = 241,
                            up_axis: Optional[np.ndarray] = None,
                            cmap: Optional[str] = None,
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            symmetric: Optional[bool] = None,
                            fps: int = 20,
                            save_path: Optional[str] = None
                            ) -> FuncAnimation:
    """Animate a pulsation diagnostic over phase.

    Implements technique #6 (animate instead of static snapshot).  Pass a
    sequence of meshes that have each been evaluated at a different phase
    (e.g. ``[evaluate_pulsations(base, t) for t in t_values]``).

    The colour scale is fixed across all frames.  By default, for diverging
    fields it is set to ``±max |values|`` across the whole sequence; for
    magnitude fields it uses the global ``[min, max]``.  Override with
    ``vmin``/``vmax`` or ``symmetric=True``.

    Parameters
    ----------
    meshes : sequence of MeshModel
        One evaluated mesh per frame.
    field : str
        Any entry in :data:`PULSATION_FIELDS`.
    timestamps : sequence of float, optional
        If provided, used in the frame title.
    save_path : str, optional
        If set, the animation is saved to this path via ``smart_save``.
    """
    if len(meshes) == 0:
        raise ValueError("`meshes` must contain at least one mesh")

    # Precompute every frame so the colour scale is consistent and the
    # animation is responsive at playback.
    frames = []
    label = DEFAULT_FIELD_LABELS[field]
    for m in meshes:
        theta_grid, phi_grid, values, label = compute_pulsation_field(
            m, field=field, n_theta=n_theta, n_phi=n_phi, up_axis=up_axis
        )
        frames.append(values)
    stack = np.stack(frames, axis=0)

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
    pcm = _pcolormesh_on_projection(ax, theta_grid, phi_grid, stack[0],
                                    projection, cmap_name, norm)
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.75, pad=0.08,
                        orientation="horizontal")
    cbar.set_label(label)
    title = ax.set_title(label)
    if projection.lower() in ("mollweide", "hammer", "aitoff", "lambert"):
        ax.grid(True, linestyle=":", linewidth=0.5, color="grey")

    def _update(frame_idx: int):
        # pcolormesh stores a 2D grid as a flat colour array; use set_array
        # with the raveled values at the face centres.
        pcm.set_array(stack[frame_idx].ravel())
        if timestamps is not None:
            title.set_text(f"{label}  ({timestamp_label}={timestamps[frame_idx]:.3f})")
        else:
            title.set_text(f"{label}  (frame {frame_idx + 1}/{len(meshes)})")
        return (pcm, title)

    anim = FuncAnimation(fig, _update, frames=len(meshes),
                         interval=1000 // max(fps, 1), blit=False)

    if save_path is not None:
        smart_save(anim, save_path, fps=fps)

    return anim
