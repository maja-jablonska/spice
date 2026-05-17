"""Generate TZ Fornacis synthetic spectra (astro_emulators_toolkit) on a merged time grid.

Single sweep, single output pickle. The time array is the union of:

  * ``num-times`` equally-spaced samples across the full orbital period
    (the out-of-eclipse baseline), and
  * ``num-eclipse-times`` samples split evenly between the primary and
    secondary eclipse windows predicted by
    :func:`spice.models.orbit_utils.eclipse_timestamps_kepler`
    (line-of-sight ``[0, 0, -1]``, matching ``check_phoebe_eclipses.py``).

The merged array is sorted; each time-point carries a ``time_origin`` tag
(``'general'`` / ``'primary_eclipse'`` / ``'secondary_eclipse'``) so
downstream notebooks can recover the eclipse-only or general-only subset
without a second pass.

Requires the optional SPICE extra: ``pip install "stellar-spice[aemu]"``.

NOTE on R^2 bug
---------------
``simulate_observed_flux`` previously multiplied each component's spectrum
by an extra factor of R^2 (the spurious ``m.radius**2`` in
``_simulate_observed_flux_impl``), biasing primary-vs-secondary contributions
by ``(R_primary / R_secondary)^2`` ≈ 4.42x for TZ For. Fixed in commit
8a7e2652. Pickles produced before that commit are biased; re-run this
script and discard them.

NOTE on parameter constants
---------------------------
The TZ For literature constants below MUST match those in the sibling
``tz_fornacis_lightcurves.py`` so the two scripts stay comparable. A
runtime check at the top of ``main`` verifies this and aborts on drift.

Usage
-----
::

    python tz_fornacis_spectra.py --output-dir data --force
    python tz_fornacis_spectra.py --wl-min 4800 --wl-max 6800 --num-wavelengths 20000
"""
import os
import pickle
import sys
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


# TZ Fornacis literature parameters. Must match tz_fornacis_lightcurves.py.
PRIMARY_MASS = 2.057       # Msun
PRIMARY_RADIUS = 8.28      # Rsun
PRIMARY_TEFF = 4930        # K
PRIMARY_LOGG = 2.91        # cgs
SECONDARY_MASS = 1.958     # Msun
SECONDARY_RADIUS = 3.94    # Rsun
SECONDARY_TEFF = 6650      # K
SECONDARY_LOGG = 3.35      # cgs
PERIOD_DAYS = 75.66647     # Gallenne et al. 2016 — must match tz_fornacis_lightcurves.py
ECC = 0.0
INCL_DEG = 85.68
PER0_DEG = 65.99           # argument of periastron
LONG_AN_DEG = 269.0        # longitude of ascending node
DISTANCE_PC = 182.8        # pc (Gaia); must match tz_fornacis_lightcurves.py
WL_MIN = 3000.0            # Angstrom; default wavelength grid lower edge
WL_MAX = 10000.0           # Angstrom; default wavelength grid upper edge
N_MESH = 5000              # icosphere subdivisions per body (spectra use finer mesh)

# Ephemeris / vgamma — must match tz_fornacis_lightcurves.py
T_P_HJD = 2452599.29040
GAMMA1_KMS = 17.99
GAMMA2_KMS = 18.35
GAMMA_KMS = (PRIMARY_MASS * GAMMA1_KMS + SECONDARY_MASS * GAMMA2_KMS) / (
    PRIMARY_MASS + SECONDARY_MASS
)
_FIRST_BJD = float(os.environ.get("TZ_FOR_FIRST_BJD", "2454887.5473123"))
_MEAN_ANOMALY_T0 = (
    ((_FIRST_BJD - T_P_HJD) % PERIOD_DAYS) / PERIOD_DAYS * (2.0 * np.pi)
)


def _check_constants_in_sync():
    """Abort if literature constants drifted from tz_fornacis_lightcurves.py."""
    sibling = Path(__file__).resolve().parent / "tz_fornacis_lightcurves.py"
    if not sibling.is_file():
        return
    sys.path.insert(0, str(sibling.parent))
    try:
        import tz_fornacis_lightcurves as tlc
    finally:
        if str(sibling.parent) in sys.path:
            sys.path.remove(str(sibling.parent))
    here = {
        "PRIMARY_MASS": PRIMARY_MASS, "PRIMARY_RADIUS": PRIMARY_RADIUS,
        "PRIMARY_TEFF": PRIMARY_TEFF, "PRIMARY_LOGG": PRIMARY_LOGG,
        "SECONDARY_MASS": SECONDARY_MASS, "SECONDARY_RADIUS": SECONDARY_RADIUS,
        "SECONDARY_TEFF": SECONDARY_TEFF, "SECONDARY_LOGG": SECONDARY_LOGG,
        "PERIOD_DAYS": PERIOD_DAYS, "ECC": ECC,
        "INCL_DEG": INCL_DEG, "PER0_DEG": PER0_DEG, "LONG_AN_DEG": LONG_AN_DEG,
        "GAMMA_KMS": GAMMA_KMS, "T_P_HJD": T_P_HJD,
        "DISTANCE_PC": DISTANCE_PC,
    }
    drifted = []
    for name, value in here.items():
        sibling_value = getattr(tlc, name, None)
        if sibling_value is None:
            continue
        if sibling_value != value:
            drifted.append((name, value, sibling_value))
    if drifted:
        msg_lines = ["TZ For constants drifted from tz_fornacis_lightcurves.py:"]
        for name, here_v, there_v in drifted:
            msg_lines.append(f"  {name}: spectra={here_v}  vs  lightcurves={there_v}")
        raise RuntimeError("\n".join(msg_lines))
    if not np.isclose(_MEAN_ANOMALY_T0, tlc._MEAN_ANOMALY_T0, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"_MEAN_ANOMALY_T0 mismatch: spectra={_MEAN_ANOMALY_T0} "
            f"lightcurves={tlc._MEAN_ANOMALY_T0}"
        )


def _build_binary_and_eclipses(em):
    """Build the TZ For binary and predict eclipse windows (aemu parameter convention)."""
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


def _build_merged_times(period_yr, primary_window, secondary_window,
                        num_times, num_eclipse_times):
    """Union of uniform + dense-eclipse time samples, sorted, with origin tags."""
    half = num_eclipse_times // 2
    t_general = np.linspace(0., period_yr, num_times)
    t_primary = np.linspace(primary_window[0], primary_window[1], half)
    t_secondary = np.linspace(secondary_window[0], secondary_window[1],
                              num_eclipse_times - half)

    times = np.concatenate([t_general, t_primary, t_secondary])
    origins = np.concatenate([
        np.full(t_general.shape, 'general', dtype=object),
        np.full(t_primary.shape, 'primary_eclipse', dtype=object),
        np.full(t_secondary.shape, 'secondary_eclipse', dtype=object),
    ])
    sort_idx = np.argsort(times, kind='stable')
    return times[sort_idx], origins[sort_idx]


def _evaluate_orbit_chunked(binary, times, chunk_size):
    """Re-uses one JIT specialisation by chunking a flat times array."""
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


def _binary_params_dict(*, wl_min: float, wl_max: float):
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
        "wl_min_A": wl_min,
        "wl_max_A": wl_max,
    }


@click.command()
@click.option("--num-times", default=150, show_default=True,
              help="Equally-spaced time samples across the orbital period.")
@click.option("--num-eclipse-times", default=80, show_default=True,
              help="Total samples within eclipse windows (split evenly between "
                   "primary and secondary).")
@click.option("--num-wavelengths", default=40000, show_default=True,
              help="Wavelength grid resolution.")
@click.option("--wl-min", default=WL_MIN, show_default=True,
              help="Wavelength grid minimum [Angstrom].")
@click.option("--wl-max", default=WL_MAX, show_default=True,
              help="Wavelength grid maximum [Angstrom].")
@click.option("--orbit-chunk", default=15, show_default=True,
              help="Times per evaluate_orbit_at_times batch (controls JIT cache reuse).")
@click.option("--output-dir", default="data", show_default=True,
              help="Directory for the output pickle.")
@click.option("--model", default="RozanskiT/TPayne-spice-harps", show_default=True,
              help="Hugging Face repo id or local bundle dir (astro_emulators_toolkit).")
@click.option("--force/--no-force", default=False, show_default=True,
              help="Overwrite existing pickle instead of skipping.")
def generate_tz_fornacis_spectra(num_times, num_eclipse_times, num_wavelengths,
                                  wl_min, wl_max, orbit_chunk, output_dir,
                                  model, force):
    """Synthesize TZ Fornacis spectra on a merged general+eclipse time grid."""
    if wl_min >= wl_max:
        raise click.BadParameter("wl-max must be greater than wl-min", param_hint="wl-max")

    _check_constants_in_sync()
    print(f"[jax] backend={jax.default_backend()}  devices={jax.devices()}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = output_dir / f"tz_fornacis_data_{num_wavelengths}.pkl"
    if out_pkl.exists() and not force:
        print(f"[skip] {out_pkl} already exists (use --force to overwrite)")
        return

    from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
    print(f"Loading intensity emulator: {model}")
    em = IntensityPretrainedAemuSpectrumEmulator(model)

    print("Building TZ For binary...")
    print(f"Distance [pc]:               {DISTANCE_PC}")
    binary, period_yr, primary_window, secondary_window = _build_binary_and_eclipses(em)
    yr_to_day = (1 * u.year).to(u.day).value
    print(f"Period [days]:               {period_yr * yr_to_day:.4f}")
    print(f"Primary eclipse [days]:      "
          f"{primary_window[0]*yr_to_day:.4f} -> {primary_window[1]*yr_to_day:.4f}")
    print(f"Secondary eclipse [days]:    "
          f"{secondary_window[0]*yr_to_day:.4f} -> {secondary_window[1]*yr_to_day:.4f}")

    times_yr, time_origin = _build_merged_times(
        period_yr, primary_window, secondary_window,
        num_times, num_eclipse_times,
    )
    n_general = int(np.sum(time_origin == 'general'))
    n_p_ecl = int(np.sum(time_origin == 'primary_eclipse'))
    n_s_ecl = int(np.sum(time_origin == 'secondary_eclipse'))
    print(f"Merged grid: {len(times_yr)} times "
          f"(general={n_general}, primary_eclipse={n_p_ecl}, secondary_eclipse={n_s_ecl})")
    print(f"Wavelength grid [A]:         {wl_min:.1f} -> {wl_max:.1f}  ({num_wavelengths} points)")

    vws = jnp.linspace(wl_min, wl_max, num_wavelengths)

    t0 = time.time()
    pb1, pb2 = _evaluate_orbit_chunked(binary, times_yr, orbit_chunk)
    print(f"orbit eval: {time.time() - t0:.1f}s")

    t0 = time.time()
    spectra_body1, spectra_body2 = _synthesize_spectra(em, vws, pb1, pb2)
    print(f"synthesis:  {time.time() - t0:.1f}s")

    tmp_pkl = out_pkl.with_suffix(out_pkl.suffix + ".tmp")
    with open(tmp_pkl, "wb") as f:
        pickle.dump({
            "mode": "merged",
            "spectra_body1": spectra_body1,
            "spectra_body2": spectra_body2,
            "mesh_body1": pb1,
            "mesh_body2": pb2,
            "wavelengths": np.asarray(vws),
            "wl_min": wl_min,
            "wl_max": wl_max,
            "times": times_yr,
            "time_origin": time_origin,
            "period_yr": period_yr,
            "primary_eclipse_window_yr": primary_window,
            "secondary_eclipse_window_yr": secondary_window,
            "binary_params": _binary_params_dict(wl_min=wl_min, wl_max=wl_max),
            "model": model,
        }, f)
    os.replace(tmp_pkl, out_pkl)
    print(f"\u2713 saved {out_pkl}")


if __name__ == "__main__":
    generate_tz_fornacis_spectra()
