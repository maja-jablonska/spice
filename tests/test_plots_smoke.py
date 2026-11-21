"""Smoke tests for spice.plots on tiny meshes (matplotlib Agg backend).

They check that each plotting entry point builds a figure with the expected
artists (and that animations write a file), not the pixels themselves.
"""
import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spice.models import IcosphereModel
from spice.models.mesh_transform import (add_pulsation, add_rotation,
                                         evaluate_pulsations, evaluate_rotation)
from spice.spectrum import Blackbody


@pytest.fixture(scope="module")
def star():
    bb = Blackbody()
    m = IcosphereModel.construct(100, 1.0, 1.0, bb.solar_parameters, bb.parameter_names)
    return evaluate_rotation(add_rotation(m, 10.0), 0.0)


@pytest.fixture(scope="module")
def pulsating():
    bb = Blackbody()
    m = IcosphereModel.construct(100, 1.0, 1.0, bb.solar_parameters, bb.parameter_names)
    # Radial mode (l = m = 0), amplitude 1 % of the radius, period 1 day.
    mp = add_pulsation(m, 0, 0, 1.0, jnp.array([[0.01, 0.0]]))
    return [evaluate_pulsations(mp, t) for t in (0.0, 0.25, 0.5, 0.75)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _n_faces(mesh):
    return int(np.asarray(mesh.faces).shape[0])


def test_plot_3D_draws_every_face(star):
    from spice.plots import plot_3D
    fig, ax = plot_3D(star)
    polys = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    assert len(polys) == 1
    assert len(polys[0].get_facecolor()) == _n_faces(star)
    # The colorbar spans the coloured property (mus by default).
    cbar = fig.axes[-1]
    lo, hi = sorted(cbar.get_ylim())
    np.testing.assert_allclose([lo, hi], [float(np.min(star.mus)), float(np.max(star.mus))],
                               atol=1e-6)


def test_plot_3D_points_mode_scatters_centres(star):
    from spice.plots import plot_3D
    fig, ax = plot_3D(star, property="los_velocities", mode="POINTS",
                      draw_los_vector=False, draw_rotation_axis=False)
    sc = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(sc) == 1 and len(sc[0].get_array()) == _n_faces(star)


def test_plot_3D_rejects_an_unknown_mode(star):
    from spice.plots import plot_3D
    with pytest.raises(ValueError):
        plot_3D(star, mode="WIREFRAME")


def test_plot_2D_returns_axes_and_colorbar(star):
    from spice.plots import plot_2D
    fig, ax, cbar_ax = plot_2D(star, property=0)     # parameter index 0 = Teff
    assert cbar_ax is not None and len(ax.collections) >= 1
    arr = ax.collections[0].get_array()
    np.testing.assert_allclose(arr, 5777.0, rtol=1e-6)


def test_plot_3D_binary_draws_both_meshes(star):
    from spice.plots import plot_3D_binary
    from spice.models.mesh_transform import transform
    other = transform(star, jnp.array([3.0, 0.0, 0.0]))
    fig, ax = plot_3D_binary(star, other)
    polys = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    assert sum(len(p.get_facecolor()) for p in polys) == 2 * _n_faces(star)


def test_plot_3D_sequence_makes_one_panel_per_mesh(star):
    from spice.plots import plot_3D_sequence
    fig, axes, cbar_ax = plot_3D_sequence([star, star, star], timestamps=[0.0, 1.0, 2.0])
    used = [a for a in np.ravel(axes) if a.collections]
    assert len(used) == 3


def test_plot_3D_mesh_and_spectrum(star):
    from spice.plots import plot_3D_mesh_and_spectrum
    wl = np.linspace(4000.0, 7000.0, 50)
    spec = np.linspace(1.0, 2.0, 50)
    fig, mesh_ax, spec_ax = plot_3D_mesh_and_spectrum(star, spec, wl, timestamp=0.5,
                                                      timestamp_label="d")
    (line,) = spec_ax.get_lines()
    np.testing.assert_allclose(line.get_xdata(), wl)
    np.testing.assert_allclose(line.get_ydata(), spec)


def test_animate_single_star_writes_a_gif(star, tmp_path):
    from spice.plots.plot_mesh import animate_single_star
    out = tmp_path / "star.gif"
    ret = animate_single_star([star, star], str(out), timestamps=[0.0, 1.0],
                              figure_size=(3, 3))
    assert ret == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:3] == b"GIF"


def test_radial_mode_has_only_a_radial_velocity(pulsating):
    from spice.plots import compute_pulsation_scalar
    mesh = pulsating[1]          # t = P/4: maximum radial velocity
    _, _, v_r, _ = compute_pulsation_scalar(mesh, "v_r")
    _, _, v_h, _ = compute_pulsation_scalar(mesh, "v_h")
    _, _, speed, _ = compute_pulsation_scalar(mesh, "speed")
    assert np.max(np.abs(v_r)) > 0
    np.testing.assert_allclose(np.abs(v_r), speed, rtol=1e-6)
    assert np.max(np.abs(v_h)) < 1e-6 * np.max(speed)
    # A radial mode moves every element by the same amount.
    np.testing.assert_allclose(v_r, v_r[0], rtol=1e-3)


def test_unknown_pulsation_field_raises(pulsating):
    from spice.plots import compute_pulsation_scalar
    with pytest.raises(ValueError):
        compute_pulsation_scalar(pulsating[0], "v_banana")


def test_plot_pulsation_map_scatters_every_element(pulsating):
    from spice.plots import plot_pulsation_map
    fig, ax = plot_pulsation_map(pulsating[1], "v_r", overlay_quiver=False)
    sc = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(sc[0].get_array()) == _n_faces(pulsating[0])


def test_plot_pulsation_components_one_axis_per_field(pulsating):
    from spice.plots import plot_pulsation_components
    fig, axes = plot_pulsation_components(pulsating[0], fields=("v_r", "speed"))
    assert len(axes) == 2


def test_plot_pulsation_cross_section_plots_each_field(pulsating):
    from spice.plots import plot_pulsation_cross_section
    fig, ax = plot_pulsation_cross_section(pulsating[0], fields=("v_r", "v_theta"),
                                           tolerance_deg=30.0)
    labels = [l.get_label() for l in ax.get_lines() if not l.get_label().startswith("_")]
    assert len(labels) == 2


def test_plot_pulsation_phase_grid_one_panel_per_phase(pulsating):
    from spice.plots import plot_pulsation_phase_grid
    fig, axes = plot_pulsation_phase_grid(pulsating, "v_r", phases=[0, 0.25, 0.5, 0.75])
    assert len([a for a in axes if a.collections]) == 4


def test_animate_pulsation_phase_writes_a_gif(pulsating, tmp_path):
    from spice.plots import animate_pulsation_phase
    out = tmp_path / "puls.gif"
    anim = animate_pulsation_phase(pulsating, "v_r", timestamps=[0, 0.25, 0.5, 0.75],
                                   save_path=str(out), fps=4)
    assert anim is not None
    assert out.exists() and out.read_bytes()[:3] == b"GIF"
