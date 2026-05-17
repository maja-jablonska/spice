"""Generate TZ Fornacis synthetic lightcurves on GPU.

Runs two synthesis sweeps from the same binary configuration and pretrained
intensity emulator and writes one pickle per sweep:

* ``general``: ``num_times`` samples uniformly spaced across the full orbital
  period — the standard out-of-eclipse lightcurve.
* ``eclipse``: ``num_eclipse_times`` samples split evenly between the primary
  and secondary eclipse windows predicted by
  :func:`spice.models.orbit_utils.eclipse_timestamps_kepler` (same Keplerian
  convention and line-of-sight ``[0, 0, -1]`` as ``check_phoebe_eclipses.py``).

GPU
---
The script does not pin a JAX platform; on a Linux GPU node with ``jax[cuda]``
installed JAX auto-selects the local CUDA device, and on macOS / CPU-only nodes
SPICE's ``__init__`` keeps it on CPU (slower but still works). The PBS wrapper
``tz_fornacis_lightcurves.pbs`` sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` /
``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` and unsets ``JAX_PLATFORMS`` so the
process can use the full GPU device. Print ``jax.devices()`` at startup so the
log shows whether the GPU was actually picked up.

Resumability
------------
Each per-mode pickle is written only if missing, so resubmitting after a
walltime hit reuses the already-finished sweep.

Usage
-----
::

    # both sweeps (default)
    python tz_fornacis_lightcurves.py --output-dir data

    # eclipse-only, higher resolution
    python tz_fornacis_lightcurves.py --mode eclipse \\
        --num-eclipse-times 200 --num-wavelengths 100000

The pickle filenames match the convention already used by
``tz_fornacis_lightcurve_plot.ipynb`` (``tz_fornacis_data_<mode>_<NW>.pkl``).
"""
import os
import pickle
import time
from pathlib import Path

# JAX env vars are best set BEFORE importing jax; this is a no-op when the PBS
# wrapper already exported them.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import click
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import astropy.units as u
from tqdm import tqdm


# TZ Fornacis literature parameters used throughout the notebook
# (tz_fornacis_spectra.ipynb). Kept in one place so the eclipse-time
# computation and the binary model can't drift apart.
PRIMARY_MASS = 2.057       # Msun
PRIMARY_RADIUS = 8.28      # Rsun
PRIMARY_TEFF = 4930        # K
PRIMARY_LOGG = 2.91        # cgs
SECONDARY_MASS = 1.958     # Msun
SECONDARY_RADIUS = 3.94    # Rsun
SECONDARY_TEFF = 6650      # K
SECONDARY_LOGG = 3.35      # cgs
PERIOD_DAYS = 75.66647     # Gallenne et al. 2016 (A&A 586, A35), Table 4
ECC = 0.0
INCL_DEG = 85.68
PER0_DEG = 65.99           # argument of periastron
LONG_AN_DEG = 269.0        # longitude of ascending node
DISTANCE_PC = 182.8        # pc (Gaia); must match tz_fornacis_spectra.py
N_MESH = 500               # icosphere subdivisions per body

# --- Systemic velocity & spectroscopic ephemeris (Gallenne et al. 2016, Table 4)
GAMMA1_KMS = 17.99         # primary systemic velocity
GAMMA2_KMS = 18.35         # secondary systemic velocity
GAMMA_KMS = (PRIMARY_MASS * GAMMA1_KMS + SECONDARY_MASS * GAMMA2_KMS) \
            / (PRIMARY_MASS + SECONDARY_MASS)                       # ~+18.16
T_P_HJD = 2452599.29040    # HJD of spectroscopic conjunction (Tp)

# Mean anomaly at SPICE time t=0 so the first synthetic sample matches the same
# orbital phase as the first HARPS exposure (sorted ADP ingestion order in
# `harps_read.ipynb`).  Override the reference BJD with TZ_FOR_FIRST_BJD if your
# FITS set starts on a different night.
_FIRST_BJD = float(os.environ.get("TZ_FOR_FIRST_BJD", "2454887.5473123"))
_MEAN_ANOMALY_T0 = (
    ((_FIRST_BJD - T_P_HJD) % PERIOD_DAYS) / PERIOD_DAYS * (2.0 * np.pi)
)


def _build_binary_and_eclipses(em):
    """Build the TZ For binary and predict (primary, secondary) eclipse windows.

    Returns
    -------
    binary : Binary
    period_yr : float
    primary_window_yr : (float, float)   # t1, t4 in years
    secondary_window_yr : (float, float)
    """
    from spice.models.binary import Binary, add_orbit
    from spice.models.mesh_model import IcosphereModel
    from spice.models.orbit_utils import eclipse_timestamps_kepler

    period_yr = (PERIOD_DAYS * u.d).to(u.year).value
    incl_rad = jnp.deg2rad(INCL_DEG)
    per0_rad = jnp.deg2rad(PER0_DEG)
    long_an_rad = jnp.deg2rad(LONG_AN_DEG)

    body1 = IcosphereModel.construct(
        N_MESH, PRIMARY_RADIUS, PRIMARY_MASS,
        em.to_parameters(dict(teff=PRIMARY_TEFF, logg=PRIMARY_LOGG)),
        em.stellar_parameter_names,
        override_log_g=False,
    )
    body2 = IcosphereModel.construct(
        N_MESH, SECONDARY_RADIUS, SECONDARY_MASS,
        em.to_parameters(dict(teff=SECONDARY_TEFF, logg=SECONDARY_LOGG)),
        em.stellar_parameter_names,
        override_log_g=False,
    )
    binary = Binary.from_bodies(body1, body2)
    # add_orbit(..., mean_anomaly, reference_time, vgamma, orbit_resolution_points)
    # ``vgamma`` / ``mean_anomaly`` / ``T`` follow ``get_orbit_jax`` (km/s; rad;
    # periastron time in years).  Phase zero of the time grid is matched to
    # ``_FIRST_BJD`` via ``_MEAN_ANOMALY_T0`` and Gallenne's ``T_P_HJD``.
    binary = add_orbit(
        binary, period_yr, ECC, 0.,
        incl_rad, per0_rad, long_an_rad,
        float(_MEAN_ANOMALY_T0), 0., float(GAMMA_KMS), 15,
    )

    (_, t1_p, _, _, t4_p,
     _, t1_s, _, _, t4_s) = eclipse_timestamps_kepler(
        PRIMARY_MASS, SECONDARY_MASS,
        period_yr, ECC, 0.,
        incl_rad, per0_rad, long_an_rad,
        PRIMARY_RADIUS, SECONDARY_RADIUS,
        pad=1.1,
        los_vector=jnp.array([0., 0., -1.]),
    )
    primary_window = (float(t1_p), float(t4_p))
    secondary_window = (float(t1_s), float(t4_s))
    if not (np.all(np.isfinite(primary_window)) and np.all(np.isfinite(secondary_window))):
        raise RuntimeError(
            "eclipse_timestamps_kepler returned NaN: impact parameter exceeds "
            "(R1+R2)*pad at the requested geometry — no eclipse to sample."
        )
    return binary, period_yr, primary_window, secondary_window


def _evaluate_orbit_chunked(binary, times, chunk_size):
    """Evaluate the orbit in fixed-size chunks (re-uses one JIT specialisation).

    The orbit eval JIT specialises on the input array shape; passing a single
    flat ``times`` array re-compiles every time ``len(times)`` changes. The
    notebook chunks via ``times.reshape((10, n//10))`` for the same reason. We
    keep that pattern but make the chunk size explicit so eclipse runs (with
    far fewer points) don't pay a recompile per call.
    """
    from spice.models.binary import evaluate_orbit_at_times

    times = jnp.asarray(times)
    n = times.shape[0]
    pb1, pb2 = [], []
    for i in tqdm(range(0, n, chunk_size), desc="orbit"):
        chunk = times[i:i + chunk_size]
        r1, r2 = evaluate_orbit_at_times(binary, chunk)
        pb1.extend(list(r1))
        pb2.extend(list(r2))
    return pb1, pb2


def _synthesize_spectra(em, vws, pb1, pb2):
    """Run simulate_observed_flux per time-point for each body."""
    from spice.spectrum import simulate_observed_flux

    log_vws = jnp.log10(vws)
    spectra_body1 = [
        simulate_observed_flux(em.intensity, _pb, log_vws, distance=DISTANCE_PC)
        for _pb in tqdm(pb1, desc="primary")
    ]
    spectra_body2 = [
        simulate_observed_flux(em.intensity, _pb, log_vws, distance=DISTANCE_PC)
        for _pb in tqdm(pb2, desc="secondary")
    ]
    return np.asarray(spectra_body1), np.asarray(spectra_body2)


def _build_sample_times(mode, period_yr, primary_window, secondary_window,
                       num_times, num_eclipse_times):
    """Return list of (label, times_array) sweeps to run for ``mode``."""
    runs = []
    if mode in ("general", "both"):
        times_general = jnp.linspace(0., period_yr, num_times)
        runs.append(("general", times_general))
    if mode in ("eclipse", "both"):
        half = num_eclipse_times // 2
        t_primary = jnp.linspace(primary_window[0], primary_window[1], half)
        t_secondary = jnp.linspace(secondary_window[0], secondary_window[1],
                                   num_eclipse_times - half)
        runs.append(("eclipses", jnp.concatenate([t_primary, t_secondary])))
    return runs


def _binary_params_dict():
    """Snapshot of the literature parameters; saved alongside each pickle."""
    return {
        "primary_mass": PRIMARY_MASS, "secondary_mass": SECONDARY_MASS,
        "primary_radius": PRIMARY_RADIUS, "secondary_radius": SECONDARY_RADIUS,
        "primary_teff": PRIMARY_TEFF, "primary_logg": PRIMARY_LOGG,
        "secondary_teff": SECONDARY_TEFF, "secondary_logg": SECONDARY_LOGG,
        "inclination_deg": INCL_DEG, "per0_deg": PER0_DEG,
        "long_an_deg": LONG_AN_DEG, "ecc": ECC,
        "period_days": PERIOD_DAYS, "n_mesh": N_MESH,
        "gamma1_kms": GAMMA1_KMS, "gamma2_kms": GAMMA2_KMS,
        "gamma_kms": GAMMA_KMS, "t_p_hjd": T_P_HJD,
        "first_bjd_phase_ref": _FIRST_BJD,
        "mean_anomaly_at_t0_rad": float(_MEAN_ANOMALY_T0),
        "reference_time_yr": 0.0,
        "gamma_applied_in_spice": True,
        "t_p_applied_in_spice": True,
        "distance_pc": DISTANCE_PC,
    }


@click.command()
@click.option("--mode", type=click.Choice(["general", "eclipse", "both"]),
              default="both", show_default=True,
              help="Which sweeps to run. 'both' writes two pickles.")
@click.option("--num-times", default=150, show_default=True,
              help="# uniform time samples for the general lightcurve.")
@click.option("--num-eclipse-times", default=80, show_default=True,
              help="Total # samples for the eclipse sweep (split evenly between "
                   "primary and secondary windows).")
@click.option("--num-wavelengths", default=40000, show_default=True)
@click.option("--wl-min", default=3000., show_default=True, help="Wavelength min [Angstrom].")
@click.option("--wl-max", default=10000., show_default=True, help="Wavelength max [Angstrom].")
@click.option("--orbit-chunk", default=15, show_default=True,
              help="Times per evaluate_orbit_at_times batch (controls JIT cache reuse).")
@click.option("--output-dir", default="data", show_default=True)
@click.option("--model", default="RozanskiT/TPayne-spice-harps", show_default=True,
              help="Hugging Face repo id or local bundle dir for the intensity emulator.")
@click.option("--force/--no-force", default=False, show_default=True,
              help="Overwrite existing per-mode pickles instead of skipping.")
def main(mode, num_times, num_eclipse_times, num_wavelengths, wl_min, wl_max,
         orbit_chunk, output_dir, model, force):
    """Synthesize TZ For general + eclipse lightcurves, writing one pickle per mode."""
    print(f"[jax] backend={jax.default_backend()}  devices={jax.devices()}")

    from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
    print(f"Loading intensity emulator: {model}")
    em = IntensityPretrainedAemuSpectrumEmulator(model)

    binary, period_yr, primary_window, secondary_window = _build_binary_and_eclipses(em)
    yr_to_day = (1 * u.year).to(u.day).value
    print(f"Distance [pc]:               {DISTANCE_PC}")
    print(f"Period [days]:               {period_yr * yr_to_day:.4f}")
    print(f"Primary eclipse [days]:      {primary_window[0]*yr_to_day:.4f} -> {primary_window[1]*yr_to_day:.4f}")
    print(f"Secondary eclipse [days]:    {secondary_window[0]*yr_to_day:.4f} -> {secondary_window[1]*yr_to_day:.4f}")

    vws = jnp.linspace(wl_min, wl_max, num_wavelengths)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _build_sample_times(mode, period_yr, primary_window, secondary_window,
                              num_times, num_eclipse_times)

    for label, times in runs:
        out_pkl = output_dir / f"tz_fornacis_data_{label}_{num_wavelengths}.pkl"
        if out_pkl.exists() and not force:
            print(f"[skip] {out_pkl} already exists (use --force to overwrite)")
            continue

        print(f"\n=== {label}: {len(times)} time points ===")

        t0 = time.time()
        pb1, pb2 = _evaluate_orbit_chunked(binary, times, orbit_chunk)
        print(f"orbit eval: {time.time() - t0:.1f}s")

        t0 = time.time()
        spectra_body1, spectra_body2 = _synthesize_spectra(em, vws, pb1, pb2)
        print(f"synthesis:  {time.time() - t0:.1f}s")

        # Write atomically: temp file + rename so a kill mid-write can't leave
        # a half-pickle that the plot notebook would silently load.
        tmp_pkl = out_pkl.with_suffix(out_pkl.suffix + ".tmp")
        with open(tmp_pkl, "wb") as f:
            pickle.dump({
                "mode": label,
                "spectra_body1": spectra_body1,
                "spectra_body2": spectra_body2,
                "mesh_body1": pb1,
                "mesh_body2": pb2,
                "wavelengths": np.asarray(vws),
                "times": np.asarray(times),
                "period_yr": period_yr,
                "primary_eclipse_window_yr": primary_window,
                "secondary_eclipse_window_yr": secondary_window,
                "binary_params": _binary_params_dict(),
                "model": model,
            }, f)
        os.replace(tmp_pkl, out_pkl)
        print(f"\u2713 saved {out_pkl}")


if __name__ == "__main__":
    main()
