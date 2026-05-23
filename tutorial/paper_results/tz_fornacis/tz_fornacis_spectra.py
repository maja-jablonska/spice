"""Generate TZ Fornacis synthetic spectra (astro_emulators_toolkit) on a merged time grid.

Single sweep, single output pickle. The time array is the union of:

  * ``num-times`` equally-spaced samples across the full orbital period
    (the out-of-eclipse baseline), and
  * ``num-eclipse-times`` samples split evenly between the primary and
    secondary eclipse windows predicted by
    :func:`spice.models.orbit_utils.eclipse_timestamps_kepler`
    (same line-of-sight as synthesis/occlusion: ``[0, 1, 0]``).

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


# TZ Fornacis literature parameters live in the shared constants module so the
# spectra / lightcurve / test scripts stay in lock-step. Only values specific to
# *this* script (wavelength window, mesh resolution, orbit precompute density,
# line of sight) are defined locally below.
from tzfor_constants import *  # noqa: F401,F403  (literature constants + _repo_root)

# Wavelength grid: defaults trimmed to a HARPS metal-band window with fine
# sampling (Δλ ~ 0.01 Å → native R ≳ 5e5 at 5500 Å) so that downstream R = 80k
# convolution is honest for both HARPS and SPICE. Pass --wl-min/--wl-max to
# widen for paper figures that need bluer / redder coverage.
WL_MIN = 4800.0            # Angstrom; default wavelength grid lower edge
WL_MAX = 6800.0            # Angstrom; default wavelength grid upper edge
N_MESH = 5000              # icosphere subdivisions per body (spectra use finer mesh)

# Keplerian orbit precomputation for add_orbit / linear interpolation at evaluate_orbit.
# 15 points smears eclipse geometry (~2% of P fits in one interpolation segment);
# use a dense grid (see check_phoebe_eclipses.py). Override via TZ_FOR_ORBIT_RESOLUTION.
ORBIT_RESOLUTION_POINTS = int(os.environ.get("TZ_FOR_ORBIT_RESOLUTION", "5000"))

# Line of sight for eclipse timing, visibility (mus), occlusion, and flux synthesis.
# Use the standard astronomical convention LOS = [0, 0, -1] — this is what
# ``orbit_utils.get_orbit_jax`` / ``relative_pos_vel_at`` produce naturally
# (orbit normal along +z at i=0; i is measured against +z), and what
# ``tutorial/paper_results/phoebe_spice_comparison/single_eclipse_system.ipynb``
# passes to ``eclipse_timestamps_kepler`` and ``get_mesh_view``. Earlier
# versions used LOS = [0, 1, 0] (the mesh_model default), which silently
# reinterpreted ``i`` as a tilt against +y and yielded a bogus 2 R☉ impact
# parameter for TZ For (effective inclination 89° instead of 85.68°); the
# blackbody primary minimum then bottomed out at Δb ≈ 0.83 mag instead of
# the literature Δb ≈ 0.2 mag.
LOS_VECTOR = jnp.array([0., 0., -1.])

# add_orbit(..., T=0, mean_anomaly=M0, reference_time=0); eclipse_timestamps_kepler(..., Tperi=0).
# Same mean anomaly requires M0 + n*t_spice = n*t_kepler  =>  t_spice = t_kepler - M0/(2π)*P.
ORBIT_TPERI_YR = 0.0
ORBIT_REFERENCE_TIME_YR = 0.0


def _kepler_eclipse_times_to_spice(times_kepler_yr, period_yr, mean_anomaly_rad):
    """Map eclipse_timestamps_kepler outputs (years since Tperi) to SPICE orbit times."""
    shift = -(float(mean_anomaly_rad) / (2.0 * np.pi)) * float(period_yr)
    t = np.asarray(times_kepler_yr, dtype=float) + shift
    return t


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
        "ORBIT_RESOLUTION_POINTS": ORBIT_RESOLUTION_POINTS,
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
    from spice.models.mesh_transform import add_rotation
    from spice.models.mesh_view import get_mesh_view
    from spice.models.orbit_utils import eclipse_timestamps_kepler

    period_yr = (PERIOD_DAYS * u.d).to(u.year).value
    incl_rad = jnp.deg2rad(INCL_DEG)
    per0_rad = jnp.deg2rad(PER0_DEG)
    long_an_rad = jnp.deg2rad(LONG_AN_DEG)

    primary_params = dict(
        teff=PRIMARY_TEFF, logg=PRIMARY_LOGG,
        feh=PRIMARY_FEH, vmic=PRIMARY_VMIC, afe=PRIMARY_AFE,
    )
    secondary_params = dict(
        teff=SECONDARY_TEFF, logg=SECONDARY_LOGG,
        feh=SECONDARY_FEH, vmic=SECONDARY_VMIC, afe=SECONDARY_AFE,
    )
    body1 = IcosphereModel.construct(
        N_MESH, PRIMARY_RADIUS, PRIMARY_MASS,
        em.to_parameters(primary_params),
        em.stellar_parameter_names,
        override_log_g=False,
    )
    body2 = IcosphereModel.construct(
        N_MESH, SECONDARY_RADIUS, SECONDARY_MASS,
        em.to_parameters(secondary_params),
        em.stellar_parameter_names,
        override_log_g=False,
    )
    # Axial rotation: gives each component its rotational broadening kernel via
    # mesh-element-level Doppler. The primary is tidally synchronized to the
    # orbit; the secondary spins much faster than synchronous (Andersen+1991).
    body1 = add_rotation(body1, rotation_velocity=PRIMARY_VROT_KMS)
    body2 = add_rotation(body2, rotation_velocity=SECONDARY_VROT_KMS)
    body1 = get_mesh_view(body1, LOS_VECTOR)
    body2 = get_mesh_view(body2, LOS_VECTOR)
    binary = Binary.from_bodies(body1, body2)
    binary = add_orbit(
        binary, period_yr, ECC, 0.,
        incl_rad, per0_rad, long_an_rad,
        float(_MEAN_ANOMALY_T0), 0., float(GAMMA_KMS), ORBIT_RESOLUTION_POINTS,
    )

    (_, t1_p, _, _, t4_p,
     _, t1_s, _, _, t4_s) = eclipse_timestamps_kepler(
        PRIMARY_MASS, SECONDARY_MASS,
        period_yr, ECC, 0.,
        incl_rad, per0_rad, long_an_rad,
        PRIMARY_RADIUS, SECONDARY_RADIUS,
        pad=1.1,
        los_vector=LOS_VECTOR,
    )
    primary_kepler_yr = (float(t1_p), float(t4_p))
    secondary_kepler_yr = (float(t1_s), float(t4_s))
    if not (
        np.all(np.isfinite(primary_kepler_yr))
        and np.all(np.isfinite(secondary_kepler_yr))
    ):
        raise RuntimeError(
            "eclipse_timestamps_kepler returned NaN: impact parameter exceeds "
            "(R1+R2)*pad at the requested geometry — no eclipse to sample."
        )
    primary_window = tuple(_kepler_eclipse_times_to_spice(
        primary_kepler_yr, period_yr, _MEAN_ANOMALY_T0,
    ))
    secondary_window = tuple(_kepler_eclipse_times_to_spice(
        secondary_kepler_yr, period_yr, _MEAN_ANOMALY_T0,
    ))
    return (
        binary, period_yr,
        primary_window, secondary_window,
        primary_kepler_yr, secondary_kepler_yr,
    )


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


def _apply_macroturbulence(spectra, log_vws, vmacro_kms):
    """Convolve each (n_wavelengths, 2) per-time spectrum with a Gaussian
    macroturbulent kernel of FWHM ``vmacro_kms``.

    Implementation: ``apply_spectral_resolution`` is a Gaussian convolution on
    a uniform log10(λ) grid with FWHM in velocity units = c / R, so we set
    ``R = c / vmacro_kms`` to get the desired macroturbulent kernel. Applied
    to both line and continuum channels so the s[:, 0] / s[:, 1] ratio used
    downstream stays well-defined (continuum is already smooth so the
    convolution barely touches it).
    """
    from spice.spectrum import apply_spectral_resolution

    if vmacro_kms <= 0.0:
        return spectra
    R_macro = C_KMS / vmacro_kms
    # Pin the 1-D Gaussian convolution to CPU: it routes through
    # jax.lax.conv_general_dilated, which dispatches to cuDNN on GPU, and some
    # Gadi cuDNN 9 installs are missing libcudnn_engines_runtime_compiled.so.9
    # (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED). The op is tiny — there's no
    # GPU win to fight for — and forcing it to CPU avoids cuDNN entirely.
    cpu = jax.devices("cpu")[0]
    log_vws_cpu = jax.device_put(log_vws, cpu)
    out = []
    for s in spectra:
        s0_cpu = jax.device_put(jnp.asarray(s[:, 0]), cpu)
        s1_cpu = jax.device_put(jnp.asarray(s[:, 1]), cpu)
        with jax.default_device(cpu):
            line = apply_spectral_resolution(log_vws_cpu, s0_cpu, R_macro)
            cont = apply_spectral_resolution(log_vws_cpu, s1_cpu, R_macro)
        out.append(np.stack([np.asarray(line), np.asarray(cont)], axis=-1))
    return np.asarray(out)


def _synthesize_spectra(em, vws, pb1, pb2):
    """Run simulate_observed_flux per time-point for each body, then apply
    per-component macroturbulent broadening."""
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
    print(f"Applying macroturbulence: primary={PRIMARY_VMACRO_KMS} km/s, "
          f"secondary={SECONDARY_VMACRO_KMS} km/s")
    spectra_body1 = _apply_macroturbulence(spectra_body1, log_vws, PRIMARY_VMACRO_KMS)
    spectra_body2 = _apply_macroturbulence(spectra_body2, log_vws, SECONDARY_VMACRO_KMS)
    return np.asarray(spectra_body1), np.asarray(spectra_body2)


def _binary_params_dict(*, wl_min: float, wl_max: float):
    """Snapshot of the literature parameters; saved alongside each pickle."""
    return {
        "primary_mass": PRIMARY_MASS, "secondary_mass": SECONDARY_MASS,
        "primary_radius": PRIMARY_RADIUS, "secondary_radius": SECONDARY_RADIUS,
        "primary_teff": PRIMARY_TEFF, "primary_logg": PRIMARY_LOGG,
        "secondary_teff": SECONDARY_TEFF, "secondary_logg": SECONDARY_LOGG,
        "primary_feh": PRIMARY_FEH, "secondary_feh": SECONDARY_FEH,
        "primary_vmic_kms": PRIMARY_VMIC, "secondary_vmic_kms": SECONDARY_VMIC,
        "primary_afe": PRIMARY_AFE, "secondary_afe": SECONDARY_AFE,
        "primary_vrot_kms": PRIMARY_VROT_KMS, "secondary_vrot_kms": SECONDARY_VROT_KMS,
        "primary_vmacro_kms": PRIMARY_VMACRO_KMS,
        "secondary_vmacro_kms": SECONDARY_VMACRO_KMS,
        "inclination_deg": INCL_DEG, "per0_deg": PER0_DEG,
        "long_an_deg": LONG_AN_DEG, "ecc": ECC,
        "period_days": PERIOD_DAYS, "n_mesh": N_MESH,
        "gamma1_kms": GAMMA1_KMS, "gamma2_kms": GAMMA2_KMS,
        "gamma_kms": GAMMA_KMS, "t_p_hjd": T_P_HJD,
        "first_bjd_phase_ref": _FIRST_BJD,
        "mean_anomaly_at_t0_rad": float(_MEAN_ANOMALY_T0),
        # M at HJD = t_p_hjd; subtract m_at_t_p_rad/(2π) from phase_spice to map
        # SPICE phase onto Clausen photometric phase (see notebook).
        "m_at_t_p_rad": M_AT_T_P_RAD,
        "reference_time_yr": 0.0,
        "orbit_resolution_points": ORBIT_RESOLUTION_POINTS,
        "los_vector": [float(x) for x in LOS_VECTOR],
        # γ reaches the spectrum via orbit_utils.get_orbit_jax applying it once
        # on the barycenter along +LOS_VECTOR; +γ ⇒ +los_velocity ⇒ redshift,
        # consistent with PHOEBE's vgamma > 0 = receding convention.
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
@click.option("--num-wavelengths", default=200000, show_default=True,
              help="Wavelength grid resolution. Default gives Δλ ≈ 0.01 Å across "
                   "4800–6800 Å so a downstream R = 80k convolution is honest "
                   "for both HARPS and SPICE.")
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
    (binary, period_yr, primary_window, secondary_window,
     primary_kepler_yr, secondary_kepler_yr) = _build_binary_and_eclipses(em)
    yr_to_day = (1 * u.year).to(u.day).value
    print(f"Period [days]:               {period_yr * yr_to_day:.4f}")
    print(f"Primary eclipse (SPICE t) [days]:   "
          f"{primary_window[0]*yr_to_day:.4f} -> {primary_window[1]*yr_to_day:.4f}")
    print(f"Secondary eclipse (SPICE t) [days]: "
          f"{secondary_window[0]*yr_to_day:.4f} -> {secondary_window[1]*yr_to_day:.4f}")
    print(f"  (Kepler Tperi=0 windows [days]: primary "
          f"{primary_kepler_yr[0]*yr_to_day:.2f}–{primary_kepler_yr[1]*yr_to_day:.2f}, "
          f"secondary {secondary_kepler_yr[0]*yr_to_day:.2f}–{secondary_kepler_yr[1]*yr_to_day:.2f})")

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
            "primary_eclipse_window_kepler_yr": primary_kepler_yr,
            "secondary_eclipse_window_kepler_yr": secondary_kepler_yr,
            "binary_params": _binary_params_dict(wl_min=wl_min, wl_max=wl_max),
            "model": model,
        }, f)
    os.replace(tmp_pkl, out_pkl)
    print(f"\u2713 saved {out_pkl}")


if __name__ == "__main__":
    generate_tz_fornacis_spectra()
