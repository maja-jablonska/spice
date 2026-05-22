"""Edge-profile figure for the spot smoothness parameter.

Replaces the old ``smoothness.png`` panel: the sigmoid edge profile that
`spice.models.spots.generate_spherical_spot` applies as a function of the
``smoothness`` argument. Mirrors the formula in ``spots.py`` exactly so the
plot stays in sync with the implementation.
"""
from __future__ import annotations

import cmasher  # noqa: F401  registers cmr.* colormaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCALE = 0.01  # 0.01 * mesh_radius, with mesh_radius = 1


def edge_profile(distance, smoothness):
    """Sigmoid profile from spots.generate_spherical_spot."""
    adjusted = (1.0 - smoothness) + 0.01 * smoothness
    normalized = np.clip(distance / SCALE, -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-normalized * adjusted))


def transition_width(smoothness, low=0.1, high=0.9):
    """Arc-length over which the profile rises from `low` to `high`."""
    adjusted = (1.0 - smoothness) + 0.01 * smoothness
    return SCALE * (np.log(high / (1 - high)) - np.log(low / (1 - low))) / adjusted


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#2b2b2b",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#1a1a1a",
    "ytick.color": "#1a1a1a",
    "xtick.direction": "out",
    "ytick.direction": "out",
})

smoothness_values = np.array([0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95])
cmap = mpl.colormaps["cmr.bubblegum"]
colors = cmap(np.linspace(0.0, 1.0, len(smoothness_values)))

distances = np.linspace(-0.2, 0.2, 2000)

fig, (ax, ax_w) = plt.subplots(
    1, 2, figsize=(11.5, 5.0),
    gridspec_kw=dict(width_ratios=[3.1, 1.0], wspace=0.32),
)

# ----- Left panel: edge profile family ---------------------------------------
lines = []
for s, c in zip(smoothness_values, colors):
    line, = ax.plot(distances, edge_profile(distances, s),
                    color=c, lw=2.4, label=f"{s:.2f}")
    lines.append(line)

# Spot-edge marker.
ax.axvline(0.0, color="#3a3a3a", lw=0.9, ls=(0, (2, 2)))
ax.text(0.0, 1.018, "spot boundary",
        ha="center", va="bottom", fontsize=8.8, color="#3a3a3a",
        transform=ax.get_xaxis_transform())

# Inside / outside tags.
tag_kw = dict(fontsize=10, fontweight="medium",
              bbox=dict(boxstyle="round,pad=0.35", fc="white",
                        ec="#bbbbbb", lw=0.6, alpha=0.92))
ax.text(-0.155, 0.92, "outside the spot", color="#1f4f8c", ha="center", **tag_kw)
ax.text(0.155, 0.08, "inside the spot", color="#a8324a", ha="center", **tag_kw)

ax.set_xlim(-0.2, 0.2)
ax.set_ylim(0.0, 1.02)
ax.set_xticks(np.arange(-0.20, 0.21, 0.05))
ax.set_yticks(np.arange(0.0, 1.01, 0.2))
ax.set_xlabel(r"signed distance from spot edge  $d$  [rad·R]", fontsize=11.5)
ax.set_ylabel(r"normalized parameter delta  $\Delta / \Delta_{\max}$", fontsize=11.5)
ax.tick_params(labelsize=10)
ax.grid(False)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# ----- Right panel: edge thickness vs smoothness -----------------------------
s_dense = np.linspace(0.0, 0.99, 400)
ax_w.plot(s_dense, transition_width(s_dense), color="#3a3a3a", lw=1.6, zorder=2)
ax_w.scatter(smoothness_values, transition_width(smoothness_values),
             c=colors, s=48, zorder=3, edgecolor="white", linewidths=0.8)
ax_w.set_yscale("log")
ax_w.set_xlim(-0.03, 1.0)
ax_w.set_ylim(transition_width(0.0), transition_width(0.99))

# Right axis: arc length in degrees (R = 1 stellar radius).
ax_w_deg = ax_w.twinx()
ax_w_deg.set_yscale("log")
ax_w_deg.set_ylim(transition_width(0.0) * 180 / np.pi,
                  transition_width(0.99) * 180 / np.pi)
ax_w_deg.set_ylabel("equivalent arc  [deg]", fontsize=10.5)

ax_w.set_xlabel("smoothness", fontsize=10.5)
ax_w.set_ylabel("10–90% edge width  [rad·R]", fontsize=10.5)
ax_w.tick_params(labelsize=9)
ax_w_deg.tick_params(labelsize=9)
ax_w.grid(True, which="both", color="#dddddd", lw=0.5, alpha=0.7)
ax_w.set_axisbelow(True)
ax_w.spines["top"].set_visible(False)
ax_w_deg.spines["top"].set_visible(False)
ax_w.set_title("edge thickness vs. smoothness", fontsize=10.5, pad=7)

# ----- Top legend ------------------------------------------------------------
leg = fig.legend(
    handles=lines,
    labels=[f"{s:.2f}" for s in smoothness_values],
    title="smoothness",
    loc="upper center",
    bbox_to_anchor=(0.40, 1.005),
    ncol=len(smoothness_values),
    frameon=False,
    fontsize=10,
    title_fontsize=10.5,
    handlelength=2.0,
    columnspacing=1.6,
)

fig.subplots_adjust(left=0.055, right=0.93, top=0.84, bottom=0.13)

out_paths = [
    "/Users/mjablons/Downloads/smoothness_improved.pdf",
    "/Users/mjablons/code/spice/tutorial/paper_results/paper_plots/spot_smoothness_profile.pdf",
]
for p in out_paths:
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"wrote {p}")
