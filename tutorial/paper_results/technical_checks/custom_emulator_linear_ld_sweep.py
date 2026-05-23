"""Validation sweep for ``custom_emulator_linear_ld.ipynb``.

Runs SPICE's ``simulate_observed_flux`` with the stored-spectrum + linear
limb-darkening emulator from the tutorial across a grid of stellar radii,
distances, and icosphere vertex counts, and compares against the analytic
hemisphere integral

    F_obs(lambda) = pi (1 - a/3) I(mu=1, lambda) (R / d)^2.

For each (R, d, N_vertices) point the script records:

  * median |residual|,
  * max |residual|,
  * max signed residual,
  * min signed residual,
  * (sum(visible_cast_areas) - pi R^2) / (pi R^2)  -- the geometric error
    that ties the residual to the mesh.

Two artifacts are saved next to the script (filenames overridable via CLI):
  * a CSV table with one row per (N, R, d) point,
  * a PNG figure with rows = N_vertices and columns = (max-signed, min-signed)
    residual, plotted on a shared diverging color scale (R on the y-axis,
    d on the x-axis).

Run:
    python custom_emulator_linear_ld_sweep.py
    python custom_emulator_linear_ld_sweep.py --n-vertices 1000 5000 20000 \
        --radii 0.5 1.0 2.0 5.0 --distances 10 50 100 500
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Force CPU + x64 before importing JAX. macOS Metal is broken for this workload
# (forced by spice/__init__.py too) and float32 misplaces narrow line cores
# during the wavelength <-> log10 round trip in `intensity`, which dominates
# the residual unless x64 is on. Matches tests/conftest.py.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Allow running directly from the tutorial/ folder without a pip install,
# matching the shim used in custom_emulator_linear_ld.ipynb.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spice.models import IcosphereModel
from spice.spectrum import simulate_observed_flux


# Same constant SPICE bakes into _simulate_observed_flux_impl (spectrum.py).
R_SUN_OVER_PC_SQ = 5.08326693599739e-16


class StoredSpectrumEmulator:
    """SpectrumEmulator that returns a tabulated spectrum with linear LD.

    Mirrors the class defined in ``custom_emulator_linear_ld.ipynb``. The
    wavelength / flux / continuum tables are captured as ``jnp.ndarray``
    closures so the JIT-compiled ``intensity`` path is just two ``jnp.interp``
    calls plus a linear LD factor.
    """

    def __init__(
        self,
        wavelengths: np.ndarray,
        flux: np.ndarray,
        continuum: np.ndarray,
        ld_a: float = 0.6,
    ):
        self._wavelengths = jnp.asarray(wavelengths)
        self._flux = jnp.asarray(flux)
        self._continuum = jnp.asarray(continuum)
        self.ld_a = float(ld_a)

    @property
    def parameter_names(self) -> List[str]:
        # MeshModel.parameters must be 2-D; a single dummy slot keeps that
        # invariant even when the emulator has no per-element knobs.
        return ["dummy"]

    @property
    def stellar_parameter_names(self) -> List[str]:
        return self.parameter_names

    @property
    def solar_parameters(self) -> jnp.ndarray:
        return jnp.array([0.0])

    def to_parameters(
        self, parameter_values: Optional[Dict[str, Any]] = None
    ) -> jnp.ndarray:
        return self.solar_parameters

    def intensity(
        self,
        log_wavelengths: jnp.ndarray,
        mu: float,
        parameters: jnp.ndarray,
    ) -> jnp.ndarray:
        wavelengths = jnp.power(10.0, log_wavelengths)
        flux_interp = jnp.interp(wavelengths, self._wavelengths, self._flux)
        continuum_interp = jnp.interp(
            wavelengths, self._wavelengths, self._continuum
        )
        mu_safe = jnp.clip(mu, 0.0, 1.0)
        ld_factor = 1.0 - self.ld_a * (1.0 - mu_safe)
        return jnp.stack(
            [flux_interp * ld_factor, continuum_interp * ld_factor], axis=-1
        )

    def flux(
        self, log_wavelengths: jnp.ndarray, parameters: jnp.ndarray
    ) -> jnp.ndarray:
        return self.intensity(log_wavelengths, 1.0, parameters)


def load_reference_spectrum(path: Path):
    with open(path, "rb") as f:
        spectrum_data = pickle.load(f)
    wavelengths = np.asarray(spectrum_data["wavelength"], dtype=np.float64)
    flux = np.asarray(spectrum_data["flux"], dtype=np.float64)
    continuum = np.asarray(spectrum_data["continuum"], dtype=np.float64)
    return wavelengths, flux, continuum


def run_one_config(
    emulator: StoredSpectrumEmulator,
    ref_wavelengths: np.ndarray,
    ref_flux: np.ndarray,
    n_vertices: int,
    radius: float,
    distance: float,
    mass: float = 1.0,
) -> Dict[str, float]:
    """Build the mesh, run simulate_observed_flux, return residual statistics."""

    mesh = IcosphereModel.construct(
        n_vertices,
        radius,
        mass,
        emulator.to_parameters(),
        emulator.parameter_names,
        override_log_g=False,
    )

    sim_log_wavelengths = jnp.log10(jnp.asarray(ref_wavelengths))
    observed = simulate_observed_flux(
        emulator.intensity,
        mesh,
        sim_log_wavelengths,
        distance=distance,
        disable_doppler_shift=True,
    )
    observed_flux = np.asarray(observed[:, 0])

    geometric_factor = (
        np.pi
        * (1.0 - emulator.ld_a / 3.0)
        * radius**2
        * R_SUN_OVER_PC_SQ
        / distance**2
    )
    expected_flux = ref_flux * geometric_factor

    residual = (observed_flux - expected_flux) / expected_flux
    cast_area_sum = float(jnp.sum(mesh.visible_cast_areas))
    cast_area_error = (cast_area_sum - np.pi * radius**2) / (np.pi * radius**2)

    return {
        "n_vertices": n_vertices,
        "n_faces": int(mesh.faces.shape[0]),
        "radius_rsun": radius,
        "distance_pc": distance,
        "median_abs_pct": float(np.median(np.abs(residual)) * 100.0),
        "max_abs_pct": float(np.max(np.abs(residual)) * 100.0),
        "max_signed_pct": float(np.max(residual) * 100.0),
        "min_signed_pct": float(np.min(residual) * 100.0),
        "cast_area_error_pct": float(cast_area_error * 100.0),
    }


def build_table(
    emulator: StoredSpectrumEmulator,
    ref_wavelengths: np.ndarray,
    ref_flux: np.ndarray,
    n_vertices_list: Sequence[int],
    radii: Sequence[float],
    distances: Sequence[float],
) -> pd.DataFrame:
    rows = []
    total = len(n_vertices_list) * len(radii) * len(distances)
    i = 0
    for n_vertices in n_vertices_list:
        for radius in radii:
            for distance in distances:
                i += 1
                print(
                    f"[{i:3d}/{total}] N={n_vertices:>6d}  "
                    f"R={radius:5.2f} Rsun  d={distance:7.2f} pc",
                    flush=True,
                )
                row = run_one_config(
                    emulator,
                    ref_wavelengths,
                    ref_flux,
                    n_vertices=n_vertices,
                    radius=radius,
                    distance=distance,
                )
                rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmaps(
    df: pd.DataFrame,
    n_vertices_list: Sequence[int],
    radii: Sequence[float],
    distances: Sequence[float],
    output_path: Path,
) -> None:
    """Two-column heatmap grid: max-signed and min-signed residual per mesh size.

    All panels share a symmetric diverging color scale so visual magnitudes are
    directly comparable across mesh sizes.
    """
    n_rows = len(n_vertices_list)
    metrics = ("max_signed_pct", "min_signed_pct")
    metric_titles = ("max signed residual [%]", "min signed residual [%]")

    grids = {}
    for n_vertices in n_vertices_list:
        sub = df[df["n_vertices"] == n_vertices]
        for metric in metrics:
            grid = (
                sub.pivot(
                    index="radius_rsun", columns="distance_pc", values=metric
                )
                .reindex(index=radii, columns=distances)
                .to_numpy()
            )
            grids[(n_vertices, metric)] = grid

    abs_max = max(np.abs(g).max() for g in grids.values())
    abs_max = max(abs_max, 1e-12)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(9.5, 3.0 * n_rows + 1.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    last_im = None
    for row, n_vertices in enumerate(n_vertices_list):
        for col, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[row, col]
            grid = grids[(n_vertices, metric)]
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-abs_max,
                vmax=abs_max,
                extent=(
                    -0.5,
                    len(distances) - 0.5,
                    -0.5,
                    len(radii) - 0.5,
                ),
            )
            last_im = im
            ax.set_xticks(range(len(distances)))
            ax.set_xticklabels([f"{d:g}" for d in distances])
            ax.set_yticks(range(len(radii)))
            ax.set_yticklabels([f"{r:g}" for r in radii])

            for i_r in range(len(radii)):
                for i_d in range(len(distances)):
                    val = grid[i_r, i_d]
                    ax.text(
                        i_d,
                        i_r,
                        f"{val:.3g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black" if abs(val) < 0.5 * abs_max else "white",
                    )

            if row == 0:
                ax.set_title(title)
            if col == 0:
                n_faces = int(df.loc[df["n_vertices"] == n_vertices, "n_faces"].iloc[0])
                ax.set_ylabel(
                    f"N={n_vertices}\n({n_faces} faces)\n\nradius [R$_\\odot$]"
                )
            if row == n_rows - 1:
                ax.set_xlabel("distance [pc]")

    fig.suptitle(
        "simulate_observed_flux residual vs analytic $\\pi(1-a/3)(R/d)^2$\n"
        "stored-spectrum emulator, linear LD (a = 0.6)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.84])
    fig.colorbar(last_im, cax=cbar_ax, label="signed residual [%]")
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved heatmap to {output_path}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_spectrum = here.parent / "data" / "test_spectrum.pkl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-vertices",
        type=int,
        nargs="+",
        default=[1000, 5000, 20000],
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0],
        help="Stellar radii in solar units.",
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=[10.0, 50.0, 100.0, 500.0],
        help="Distances in parsecs.",
    )
    parser.add_argument("--ld-a", type=float, default=0.6)
    parser.add_argument(
        "--spectrum-path",
        type=Path,
        default=default_spectrum,
        help="Path to test_spectrum.pkl (defaults to ../data/test_spectrum.pkl).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=here / "custom_emulator_linear_ld_sweep.csv",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=here / "custom_emulator_linear_ld_sweep.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading reference spectrum from {args.spectrum_path}")
    ref_wavelengths, ref_flux, ref_continuum = load_reference_spectrum(
        args.spectrum_path
    )
    print(
        f"  wavelength range: {ref_wavelengths.min():.1f} - "
        f"{ref_wavelengths.max():.1f} A  ({ref_wavelengths.size} pts)"
    )

    emulator = StoredSpectrumEmulator(
        ref_wavelengths, ref_flux, ref_continuum, ld_a=args.ld_a
    )

    print(
        f"\nSweep: {len(args.n_vertices)} mesh sizes "
        f"x {len(args.radii)} radii x {len(args.distances)} distances "
        f"= {len(args.n_vertices) * len(args.radii) * len(args.distances)} runs\n"
    )

    df = build_table(
        emulator,
        ref_wavelengths,
        ref_flux,
        n_vertices_list=args.n_vertices,
        radii=args.radii,
        distances=args.distances,
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n=== Residual table ===")
    print(df.to_string(index=False))

    df.to_csv(args.csv_out, index=False)
    print(f"\nSaved table to {args.csv_out}")

    plot_heatmaps(
        df,
        n_vertices_list=list(args.n_vertices),
        radii=list(args.radii),
        distances=list(args.distances),
        output_path=args.png_out,
    )


if __name__ == "__main__":
    main()
