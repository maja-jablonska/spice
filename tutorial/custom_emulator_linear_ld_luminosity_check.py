"""Analytic-check sweep for ``simulate_monochromatic_luminosity``.

Companion to ``custom_emulator_linear_ld_sweep.py``. Builds the same
``StoredSpectrumEmulator`` (tabulated spectrum + linear LD) and asks whether
the radius scaling of ``simulate_monochromatic_luminosity`` matches the
analytic expectation, mirroring the test that surfaced the R^4 bug in
``simulate_observed_flux``.

What the function actually computes
-----------------------------------
``simulate_monochromatic_luminosity`` integrates::

    Sigma_i  m.areas[i] * flux_fn(log_wl, parameters[i])

then multiplies the result by ``m.radius**2 * 4.8399849e+21``. The constant is
``R_sun_cm**2``. The integrated array is ``m.areas`` (i.e. ``base_areas`` plus
pulsation offsets) which -- unlike ``visible_cast_areas`` -- is *unit-sphere*
normalised: empirically ``sum(m.areas) ~ 4*pi`` for every radius. So the
``radius**2 * R_sun_cm**2`` prefactor is the unit-sphere -> physical-cm^2
conversion, not a double-count.

Analytic expectation for our emulator
-------------------------------------
``StoredSpectrumEmulator.flux(log_wl, params) = intensity(log_wl, mu=1, ...)``
returns the disk-centre intensity ``I(1, lambda)``. With uniform parameters
across the surface, the unit-sphere integral collapses to
``4*pi * I(1, lambda)`` (modulo a small O(1e-3) icosphere-discretisation
overhead in ``sum(m.areas)``). After SPICE's prefactor::

    L_SPICE(lambda) = sum(m.areas) * I(1, lambda) * R**2 * R_sun_cm**2
                    ~ 4*pi * I(1, lambda) * R**2 * R_sun_cm**2

This is *not* the textbook monochromatic luminosity (that would carry an
extra ``pi * (1 - a/3)`` because flux = pi * I for limb-darkened sources),
but our emulator is intentionally returning intensity, so the analytic for
this experiment is the bare ``4*pi * R^2 * R_sun_cm^2 * I(1)``. What we
care about is the *radius scaling*: the ratio between the SPICE output and
this analytic should be 1 +/- (icosphere area error) for every R.

If a ``radius**N`` bug analogous to the observed-flux one were lurking,
``L_SPICE / L_analytic`` would have a clean ``R^(N-2)`` signature.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

from spice.models import IcosphereModel
from spice.spectrum import simulate_monochromatic_luminosity


R_SUN_CM_SQ = 4.8399849e21  # matches the constant in spectrum.py:399


class StoredSpectrumEmulator:
    """SpectrumEmulator that returns a tabulated spectrum with linear LD."""

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
    mass: float = 1.0,
) -> Dict[str, float]:
    mesh = IcosphereModel.construct(
        n_vertices,
        radius,
        mass,
        emulator.to_parameters(),
        emulator.parameter_names,
        override_log_g=False,
    )

    sim_log_wavelengths = jnp.log10(jnp.asarray(ref_wavelengths))
    L_spice = simulate_monochromatic_luminosity(
        emulator.flux,
        mesh,
        sim_log_wavelengths,
        disable_doppler_shift=True,
    )
    L_spice_line = np.asarray(L_spice[:, 0])

    sum_areas = float(jnp.sum(mesh.areas))
    # Analytic L for this emulator: sum(m.areas) * I(1, lambda) * R^2 * R_sun_cm^2.
    # We use the empirical sum_areas (instead of the perfect 4*pi) so the
    # ~0.1% icosphere area-discretisation overhead doesn't pollute the radius
    # scaling check.
    L_analytic = sum_areas * ref_flux * (radius ** 2) * R_SUN_CM_SQ

    residual = (L_spice_line - L_analytic) / L_analytic

    # Also report ratio to R=1 so a residual radius scaling would jump out.
    return {
        "n_vertices": n_vertices,
        "n_faces": int(mesh.faces.shape[0]),
        "radius_rsun": radius,
        "sum_areas": sum_areas,
        "sum_areas_over_4pi": sum_areas / (4.0 * np.pi),
        "median_residual_pct": float(np.median(residual) * 100.0),
        "max_abs_residual_pct": float(np.max(np.abs(residual)) * 100.0),
        "L_at_5500A": float(np.interp(5500.0, ref_wavelengths, L_spice_line)),
    }


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
    )
    parser.add_argument("--ld-a", type=float, default=0.6)
    parser.add_argument(
        "--spectrum-path",
        type=Path,
        default=default_spectrum,
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=here / "custom_emulator_linear_ld_luminosity_check.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading reference spectrum from {args.spectrum_path}")
    ref_wavelengths, ref_flux, ref_continuum = load_reference_spectrum(
        args.spectrum_path
    )
    emulator = StoredSpectrumEmulator(
        ref_wavelengths, ref_flux, ref_continuum, ld_a=args.ld_a
    )

    print(
        f"\nSweep: {len(args.n_vertices)} mesh sizes x {len(args.radii)} radii "
        f"= {len(args.n_vertices) * len(args.radii)} runs\n"
    )

    rows = []
    total = len(args.n_vertices) * len(args.radii)
    i = 0
    for n_vertices in args.n_vertices:
        for radius in args.radii:
            i += 1
            print(
                f"[{i:3d}/{total}] N={n_vertices:>6d}  R={radius:5.2f} Rsun",
                flush=True,
            )
            rows.append(
                run_one_config(
                    emulator,
                    ref_wavelengths,
                    ref_flux,
                    n_vertices=n_vertices,
                    radius=radius,
                )
            )

    df = pd.DataFrame(rows)

    # Cross-radius diagnostic: ratio of L(R) / L(R=1) at the same N.
    # If radius scaling is exactly R^2, this column equals R^2 exactly.
    # An anomalous radius exponent N would show up as R^N here.
    df["L_ratio_to_R1"] = np.nan
    for n_vertices in args.n_vertices:
        sub_mask = df["n_vertices"] == n_vertices
        if not (df.loc[sub_mask, "radius_rsun"] == 1.0).any():
            continue
        L_ref = float(
            df.loc[sub_mask & (df["radius_rsun"] == 1.0), "L_at_5500A"].iloc[0]
        )
        df.loc[sub_mask, "L_ratio_to_R1"] = df.loc[sub_mask, "L_at_5500A"] / L_ref

    df["expected_R_squared"] = df["radius_rsun"] ** 2
    df["L_ratio_minus_R2_pct"] = (
        (df["L_ratio_to_R1"] - df["expected_R_squared"])
        / df["expected_R_squared"]
        * 100.0
    )

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n=== Results ===")
    print(df.to_string(index=False))

    df.to_csv(args.csv_out, index=False)
    print(f"\nSaved table to {args.csv_out}")

    print(
        "\nVerdict heuristic:\n"
        "  - 'L_ratio_to_R1' column should equal R^2 exactly if the radius\n"
        "    scaling is correct. The 'L_ratio_minus_R2_pct' column reports\n"
        "    the deviation in percent.\n"
        "  - 'median_residual_pct' should be ~0 if our analytic\n"
        "    sum(m.areas) * I(1) * R^2 * R_sun_cm^2 matches SPICE.\n"
    )


if __name__ == "__main__":
    main()
