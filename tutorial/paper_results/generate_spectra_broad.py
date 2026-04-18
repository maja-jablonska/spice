#!/usr/bin/env python3
"""Generate the broad-band synthetic spectra for Delta Cephei.

This command-line utility reproduces the workflow from `cepheid.ipynb` to
evaluate the pulsating stellar mesh and compute the continuum and flux spectra
using the TransformerPayne emulator. The resulting spectra are stored in a
pickle file for later analysis.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
from tqdm import tqdm

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_pulsation, evaluate_pulsations
from spice.spectrum import simulate_observed_flux
from spice.utils.parameters import modify_mesh_parameter_from_array
from transformer_payne import TransformerPayne

jax.config.update("jax_enable_x64", True)


FOURIER_SERIES_PARAMETERS = np.array(
    [
        [5.08410023e-02, 2.52778534e00],
        [1.07786710e-02, 1.89320108e00],
        [3.64884277e-03, 1.24266937e00],
        [1.44494986e-03, 5.48410078e-01],
        [6.17318686e-04, -2.43612157e-01],
        [2.88661671e-04, -1.15420866e00],
        [1.52119818e-04, -2.10861958e00],
        [8.50754672e-05, -3.00762780e00],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute spectra_broad following tutorial/paper_results/cepheid.ipynb"
    )
    parser.add_argument(
        "--fits",
        type=Path,
        default=Path("delta_cep.fits"),
        help="Path to the SPIPS3 results FITS file (default: delta_cep.fits).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("delta_cephei_spectra_broad.pkl"),
        help="Destination pickle file for the spectra (default: delta_cephei_spectra_broad.pkl).",
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=100,
        help="Number of phases to evaluate in the pulsation time series (default: 100).",
    )
    parser.add_argument(
        "--wavelength-min",
        type=float,
        default=3000.0,
        help="Minimum wavelength in Angstroms for the synthesized spectra (default: 3000 Å).",
    )
    parser.add_argument(
        "--wavelength-max",
        type=float,
        default=9000.0,
        help="Maximum wavelength in Angstroms for the synthesized spectra (default: 9000 Å).",
    )
    parser.add_argument(
        "--wavelength-samples",
        type=int,
        default=40000,
        help="Number of wavelength samples to evaluate (default: 40000).",
    )
    return parser.parse_args()


def load_spips_data(fits_path: Path) -> tuple[Table, Table]:
    pulsation_data = Table.read(fits_path, format="fits", hdu=2)
    model_data = Table.read(fits_path, format="fits", hdu=1)
    return pulsation_data, model_data


def build_interpolators(pulsation_data: Table) -> tuple[interp1d, interp1d]:
    phase_data = np.array(pulsation_data["PHASE"])
    teff_data = np.array(pulsation_data["Teff"])
    logg_data = np.array(pulsation_data["logg"])

    sort_idx = np.argsort(phase_data)
    sorted_phase = phase_data[sort_idx]
    sorted_teff = teff_data[sort_idx]
    sorted_logg = logg_data[sort_idx]

    teff_interp = interp1d(
        sorted_phase,
        sorted_teff,
        kind="cubic",
        bounds_error=False,
        fill_value=(sorted_teff[-1], sorted_teff[0]),
        assume_sorted=True,
    )
    logg_interp = interp1d(
        sorted_phase,
        sorted_logg,
        kind="cubic",
        bounds_error=False,
        fill_value=(sorted_logg[-1], sorted_logg[0]),
        assume_sorted=True,
    )
    return teff_interp, logg_interp


def build_mesh_model(tp: TransformerPayne, pulsation_data: Table, period: float) -> tuple:
    mesh = IcosphereModel.construct(
        20000,
        43.06,
        4.8,
        tp.to_parameters(
            {
                "logteff": np.log10(float(pulsation_data["Teff"][0])),
                "logg": float(pulsation_data["logg"][0]),
                "vmic": 2.0,
                "Fe": 0.08,
            }
        ),
        tp.stellar_parameter_names,
        max_fourier_order=10,
    )
    mesh_with_pulsation = add_pulsation(
        mesh,
        m_order=0,
        l_degree=0,
        period=period,
        fourier_series_parameters=FOURIER_SERIES_PARAMETERS,
    )
    return mesh_with_pulsation


def evaluate_mesh_timeseries(mesh_with_pulsation, times: np.ndarray) -> list:
    evaluated = []
    for t in tqdm(times, desc="Evaluating pulsation states"):
        evaluated.append(evaluate_pulsations(mesh_with_pulsation, float(t)))
    return evaluated


def apply_teff_logg_updates(
    mesh_states: list,
    phases: np.ndarray,
    teff_interp: interp1d,
    logg_interp: interp1d,
) -> list:
    updated_states = []
    for mesh_state, phase in zip(mesh_states, phases):
        teff = float(teff_interp(phase % 1.0))
        logg = float(logg_interp(phase % 1.0))
        mesh_teff = modify_mesh_parameter_from_array(mesh_state, 0, jnp.log10(teff))
        mesh_logg = modify_mesh_parameter_from_array(mesh_teff, 1, logg)
        updated_states.append(mesh_logg)
    return updated_states


def simulate_spectra(tp: TransformerPayne, mesh_states: list, wavelengths: jnp.ndarray) -> np.ndarray:
    spectra = []
    log_wavelengths = jnp.log10(wavelengths)
    for mesh_state in tqdm(mesh_states, desc="Simulating spectra"):
        flux = simulate_observed_flux(tp.intensity, mesh_state, log_wavelengths)
        spectra.append(np.asarray(flux))
    return np.stack(spectra, axis=0)


def main() -> None:
    args = parse_args()

    pulsation_data, model_data = load_spips_data(args.fits)
    period = float(model_data["PERIOD"][0])

    tp = TransformerPayne.download()
    mesh_with_pulsation = build_mesh_model(tp, pulsation_data, period)

    timeseries = np.linspace(0.0, period, args.num_phases)
    mesh_states = evaluate_mesh_timeseries(mesh_with_pulsation, timeseries)

    teff_interp, logg_interp = build_interpolators(pulsation_data)
    phases = (timeseries % period) / period
    mesh_states = apply_teff_logg_updates(mesh_states, phases, teff_interp, logg_interp)

    wavelengths = jnp.linspace(args.wavelength_min, args.wavelength_max, args.wavelength_samples)
    spectra_broad = simulate_spectra(tp, mesh_states, wavelengths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        pickle.dump(
            {
                "wavelengths": np.asarray(wavelengths),
                "spectra_broad": spectra_broad,
            },
            fh,
        )

    print(f"Wrote spectra_broad with shape {spectra_broad.shape} to {args.output}")


if __name__ == "__main__":
    main()

