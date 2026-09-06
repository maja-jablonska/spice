"""TZ For photometry: PHOEBE geometry + SPICE synthesis with the aemu emulator.

The blackbody comparison (``phoebe_spice_comparison/tzfor_phoebe_vs_spice.py``)
showed the two codes agree on *geometry* to ~0.2% once passbands are matched,
and that SPICE's spherical mesh costs 0.6-0.7% in primary eclipse depth against
PHOEBE's Roche surface. This script keeps PHOEBE's geometry and replaces the
blackbody with real MARCS atmospheres:

* **PHOEBE** builds the mesh -- Roche distortion, gravity darkening,
  irradiation -- and owns the eclipses. ``PhoebeConfig.get_projected_areas``
  returns ``areas * mus * visibilities``, so PHOEBE's partial-triangle
  visibility *is* the occlusion; SPICE never computes any.
* **SPICE** does the radiative transfer, calling the intensity bundle
  ``RozanskiT/TPayne-spice-harps`` per surface element at that element's own
  teff, log g and mu. Because it is a true intensity bundle (``mu`` is an input
  channel), limb darkening comes out of MARCS rather than an assumed law.

Only Stromgren b and y are produced: the bundle is a HARPS-range emulator, so
there is no bolometric option -- and b/y are the Clausen bands anyway.

Requires ``JAX_ENABLE_X64=1`` (the transformer_payne core refuses to build
otherwise).
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DAYS_TO_YR = 0.0027378507871321013
BUNDLE = "RozanskiT/TPayne-spice-harps"

# Beyond teff / log g, which PHOEBE supplies per element, the bundle needs these
# fixed per star. Andersen+1991 metallicity; vmicro is the evolved-star standard.
ABUNDANCE_DEFAULTS = {"feh": -0.30, "vmicro": 1.5,
                      "a": 0.0, "c": 0.0, "n": 0.0, "o": 0.0, "r": 0.0, "s": 0.0}

MESH_COLUMNS = ["teffs", "loggs", "mus", "visibilities", "areas",
                "us", "vs", "ws", "vus", "vvs", "vws"]

# Spectral windows for the HARPS comparison. The full 3781-6914 A range at
# HARPS sampling is far too expensive per surface element, so use two compact
# diagnostic regions at ~0.02 A (Nyquist for R = 115000). The emulator is
# wavelength-conditioned, so a non-contiguous grid is fine.
# The bundle's training grid is 3780-6910 A (source_log10_wavelength). Stromgren
# u (3150-3850) is 90% outside it and the emulator still returns confident finite
# values there, so hybrid photometry is restricted to v/b/y. v clips a 30 A edge.
HYBRID_PHOT_BANDS = ("Stromgren:v", "Stromgren:b", "Stromgren:y")
HYBRID_PHOT_RANGE = (3780.0, 5900.0)

SPECTRAL_WINDOWS = [
    (5160.0, 5200.0, "Mg b triplet + Fe I"),
    (6540.0, 6580.0, "H-alpha"),
]


def phoebe_filter(phoebe_name, non_photonic=True):
    """A SPICE ``Filter`` built from PHOEBE's own transmission table.

    Using PHOEBE's curves on both sides removes passband definition as a free
    variable. Two traps this avoids:

    * ``ptf_table['wl']`` is in **metres**, and PHOEBE's bolometric passband is
      named in nanometres -- ``Bolometric:900-40000`` really spans 9000-400000 A.
    * ``non_photonic=True`` selects the energy-weighted branch of
      ``AB_passband_luminosity``, matching PHOEBE's default
      ``intens_weighting='energy'``; the photon-counting default adds a factor
      of lambda to the integrand.

    The tables come out of FITS big-endian ('>f8'), which JAX rejects.
    """
    import jax.numpy as jnp
    from phoebe.atmospheres import passbands as pbs
    from spice.spectrum.filter import Filter

    pb = pbs.get_passband(phoebe_name)
    wl_a = np.asarray(pb.ptf_table["wl"]).astype(np.float64) * 1e10  # m -> A
    tr = np.asarray(pb.ptf_table["fl"]).astype(np.float64)
    return Filter(jnp.array([wl_a, tr]), name=f"PHOEBE {phoebe_name}",
                  non_photonic=non_photonic)


def harps_epochs(data_dir):
    """(BJD, filename) for every readable HARPS spectrum.

    One file in the archive is 0 bytes; skip it rather than aborting the run.
    """
    import glob
    from astropy.io import fits

    good, skipped = [], []
    for path in sorted(glob.glob(str(Path(data_dir) / "ADP*.fits"))):
        try:
            with fits.open(path) as h:
                mjd = h[0].header.get("MJD-OBS")
            if mjd is None:
                raise ValueError("no MJD-OBS")
            good.append((float(mjd) + 2400000.5, Path(path).name))
        except Exception as exc:
            skipped.append((Path(path).name, f"{type(exc).__name__}: {exc}"))
    for name, why in skipped:
        print(f"  [skip] {name}: {why}")
    return sorted(good)


def spectral_grid(n_per_window):
    """Concatenated wavelength grid over SPECTRAL_WINDOWS."""
    import jax.numpy as jnp
    return jnp.concatenate([
        jnp.linspace(lo, hi, n_per_window) for lo, hi, _ in SPECTRAL_WINDOWS])


def build_bundle(n_mesh, distortion="roche", gravity_darkening=True, irradiation=True):
    """TZ For in PHOEBE with the physics SPICE cannot supply itself."""
    import phoebe
    from astropy import units as u
    import tzfor_constants as K

    phoebe.logger(clevel="ERROR")
    b = phoebe.default_binary()
    b.flip_constraint("mass@primary", solve_for="sma")

    b.set_value("period@binary@component", K.PERIOD_DAYS)
    b.set_value("q@binary@component", K.SECONDARY_MASS / K.PRIMARY_MASS)
    b.set_value("ecc@binary@component", K.ECC)
    b.set_value("mass@primary@component", K.PRIMARY_MASS)
    b.set_value_all("incl@binary", K.INCL_DEG)
    b.set_value("per0@binary@component", K.PER0_DEG)
    b.set_value("long_an@binary@component", K.LONG_AN_DEG)
    b.set_value("vgamma", K.GAMMA_KMS)
    # Anchor the ephemeris. PHOEBE's default t0_supconj is 0.0 (BJD zero), which
    # is harmless when the sampling times are derived from the bundle itself
    # (the photometry path does that), but puts the model at arbitrary orbital
    # phases the moment real observation BJDs are used -- a constant 0.291 in
    # phase, 22 d, for TZ For. T_P_HJD is the time of the deeper minimum, which
    # for a circular orbit is a conjunction.
    # T_P_HJD is the DEEPER minimum; PHOEBE's t0_supconj is the shallower one
    # (superior conjunction of the primary), half a period away. Measured: with
    # t0_supconj = T_P the deeper eclipse lands at phase 0.5000, with
    # T_P - P/2 it lands at 0.9987. Anchoring at T_P would put the model half a
    # period out, which for a near-equal-light SB2 swaps the two stars'
    # velocities and still looks plausible.
    b.set_value("t0_supconj@binary@component", K.T_P_HJD - K.PERIOD_DAYS / 2.0)
    b.set_value("requiv@primary@component", K.PRIMARY_RADIUS)
    b.set_value("requiv@secondary@component", K.SECONDARY_RADIUS)
    b.set_value("teff@primary@component", K.PRIMARY_TEFF)
    b.set_value("teff@secondary@component", K.SECONDARY_TEFF)
    b.set_value("distance", K.DISTANCE_PC * u.pc)
    b.set_value_all("distortion_method", distortion)
    b.set_value_all("ntriangles", n_mesh)

    # Both stars are cooler than ~7000 K, so both have convective envelopes:
    # Lucy's beta = 0.32. Setting 0.0 reproduces the blackbody-comparison setup.
    # Rotation. PHOEBE's default syncpar=1 (tidally locked) is right for the
    # primary but gives the secondary 2.6 km/s against Andersen's 38 km/s --
    # a 14x under-broadening that would pass unnoticed in a synthetic spectrum.
    # syncpar = v_rot / v_synchronous, with v_sync = 2 pi R / P.
    for comp, radius, vrot in (("primary", K.PRIMARY_RADIUS, K.PRIMARY_VROT_KMS),
                               ("secondary", K.SECONDARY_RADIUS, K.SECONDARY_VROT_KMS)):
        v_sync = 2 * np.pi * radius * 6.957e5 / (K.PERIOD_DAYS * 86400.0)
        b.set_value(f"syncpar@{comp}@component", vrot / v_sync)

    b.set_value_all("gravb_bol", 0.32 if gravity_darkening else 0.0)
    b.set_value_all("irrad_method", "horvat" if irradiation else "none")
    return b


def _sample_times(b, n_per_eclipse):
    """Both eclipse windows plus a quadrature baseline, sorted.

    Sorted because PHOEBE reorders ``compute_times`` internally; see the note in
    ``phoebe_spice_comparison/tzfor_phoebe_vs_spice.py``.
    """
    import jax.numpy as jnp
    from spice.models.orbit_utils import eclipse_timestamps_kepler

    _, t1_p, _, _, t4_p, _, t1_s, _, _, t4_s = eclipse_timestamps_kepler(
        b.get_parameter("mass@primary@component").value,
        b.get_parameter("mass@secondary@component").value,
        b.get_parameter("period@binary@component").value * DAYS_TO_YR,
        b.get_parameter("ecc@binary@component").value,
        b.get_parameter("t0_perpass@binary@component").value * DAYS_TO_YR,
        jnp.deg2rad(b.get_parameter("incl@binary@component").value),
        b.get_parameter("per0@binary@component").value * 0.017453292519943295,
        b.get_parameter("long_an@binary@component").value * 0.017453292519943295,
        b.get_parameter("requiv@primary@component").value,
        b.get_parameter("requiv@secondary@component").value,
        pad=1.15, los_vector=jnp.array([0.0, 0.0, -1.0]),
    )
    edges = [float(t) / DAYS_TO_YR for t in (t1_p, t4_p, t1_s, t4_s)]
    period = b.get_parameter("period@binary@component").value
    baseline = 0.5 * (edges[0] + edges[1]) + 0.25 * period
    times = np.sort(np.concatenate([
        [baseline],
        np.linspace(edges[0], edges[1], n_per_eclipse),
        np.linspace(edges[2], edges[3], n_per_eclipse),
    ]))
    return times, edges, baseline, int(np.argmin(np.abs(times - baseline)))


def export_meshes(b, times, parameter_names, out_path):
    """Build the PHOEBE meshes and pickle them. Runs on the PHOEBE side only.

    PHOEBE is not installed in the Gadi env (and installing it there risks the
    jax/flax stack), but it is also the cheap half -- seconds for a whole light
    curve, against minutes per epoch for the emulator. So build the meshes
    locally and ship them.

    ``PhoebeModel`` is a plain namedtuple of arrays and its PHOEBE guard lives
    in ``construct``, not ``__new__``, so the pickle rehydrates on a host with
    no PHOEBE at all.
    """
    import pickle as _pickle
    from spice.models import PhoebeModel
    from spice.models.phoebe_utils import Component, PhoebeConfig

    config = PhoebeConfig(b, mesh_dataset_name="mesh01")
    fixed = {k: v for k, v in ABUNDANCE_DEFAULTS.items() if k in parameter_names}

    pairs, teff_spread = [], []
    for i, t in enumerate(times):
        models = []
        for component in (Component.PRIMARY, Component.SECONDARY):
            m = PhoebeModel.construct(config, float(t), parameter_names=parameter_names,
                                      parameter_values=fixed, component=component)
            models.append(m)
            if i == 0:
                # (n_elements, n_parameters): one row per surface element.
                pars = np.asarray(m.parameters)
                assert pars.shape[1] == len(parameter_names), (
                    f"parameter array is {pars.shape}, expected "
                    f"(n_elements, {len(parameter_names)})")
                teffs = pars[:, parameter_names.index("marcs_teff")]
                loggs = pars[:, parameter_names.index("marcs_logg")]
                teff_spread.append((str(component), float(teffs.min()), float(teffs.max()),
                                    float(loggs.min()), float(loggs.max())))
        pairs.append(tuple(models))
    payload = {"times": np.asarray(times), "models": pairs,
               "parameter_names": list(parameter_names), "fixed": fixed,
               "teff_spread": teff_spread}
    with open(out_path, "wb") as f:
        _pickle.dump(payload, f, protocol=4)
    return payload


def synthesize_from_payload(payload, emu, wavelengths, filters, verbose=True):
    """Emulator synthesis only -- no PHOEBE import anywhere in this path."""
    import jax.numpy as jnp
    from spice.spectrum.spectrum import AB_passband_luminosity, simulate_observed_flux

    log_wavelengths = jnp.log10(wavelengths)
    times = payload["times"]
    mags = {k: [] for k in filters}
    for i, (m1, m2) in enumerate(payload["models"]):
        t0 = time.time()
        total = None
        for model in (m1, m2):
            spec = simulate_observed_flux(emu.intensity, model, log_wavelengths,
                                          disable_doppler_shift=True)
            total = spec[:, 0] if total is None else total + spec[:, 0]
        for key, filt in filters.items():
            mags[key].append(float(AB_passband_luminosity(filt, wavelengths, total)))
        if verbose:
            print(f"  epoch {i+1}/{len(times)}  t={times[i]:10.4f} d  "
                  f"{time.time()-t0:7.1f} s", flush=True)
    return {k: np.array(v) for k, v in mags.items()}


def synthesize_spectra(payload, emu, wavelengths, verbose=True):
    """Doppler-shifted spectra per epoch, kept split by component.

    Two deliberate differences from the photometry path:

    * ``disable_doppler_shift=False`` -- the point of a spectrum. Shifts come
      from ``PhoebeModel.los_velocities``, which carries orbital motion,
      rotation (at the syncpar set in ``build_bundle``) and vgamma together.
    * components are stored separately, so the spectroscopic light ratio and
      each star's line depths stay recoverable afterwards.

    The emulator's two output channels are (flux, continuum), so the normalized
    model spectrum is just ``flux / continuum`` -- no continuum fitting needed
    on the model side.
    """
    from spice.spectrum.spectrum import simulate_observed_flux

    import jax.numpy as jnp
    log_wavelengths = jnp.log10(wavelengths)
    times = payload["times"]
    out = {"primary": [], "secondary": [],
           "primary_continuum": [], "secondary_continuum": []}
    for i, (m1, m2) in enumerate(payload["models"]):
        t0 = time.time()
        for model, key in ((m1, "primary"), (m2, "secondary")):
            spec = simulate_observed_flux(emu.intensity, model, log_wavelengths,
                                          disable_doppler_shift=False)
            out[key].append(np.asarray(spec[:, 0]))
            out[key + "_continuum"].append(np.asarray(spec[:, 1]))
        if verbose:
            print(f"  epoch {i+1}/{len(times)}  BJD={times[i]:14.5f}  "
                  f"{time.time()-t0:7.1f} s", flush=True)
    return {k: np.asarray(v) for k, v in out.items()}


def synthesize_grid(payload, emu, wavelengths, dteffs, dloggs, fehs,
                    ref_epoch=None, verbose=True):
    """Per-component spectra over a (dTeff, dlogg, feh) grid, for fitting.

    Economies that make an all-free-parameter study affordable:

    * The grid is generated by *shifting* the per-element teff/logg on the
      existing PHOEBE meshes, so PHOEBE never re-runs and the gravity-darkened
      structure and rotation are preserved -- only the star's mean moves.
    * Each component is synthesised once, at a reference out-of-eclipse epoch.
      Orbital motion is a rigid shift of the whole component spectrum, applied
      afterwards at fit time; out of eclipse the projected disc geometry is
      effectively phase-independent, so the rotational broadening carried in
      the mesh velocities is reusable. (Eclipse epochs would break this and are
      excluded from the fit.)

    Returns per-component flux and continuum for every grid point.
    """
    import jax.numpy as jnp
    from spice.spectrum.spectrum import simulate_observed_flux

    names = payload["parameter_names"]
    it, ig, ife = (names.index("marcs_teff"), names.index("marcs_logg"),
                   names.index("feh"))
    log_wavelengths = jnp.log10(wavelengths)
    if ref_epoch is None:
        ref_epoch = 0

    out = {}
    total = 2 * len(dteffs) * len(dloggs) * len(fehs)
    k = 0
    for comp_idx, comp in ((0, "primary"), (1, "secondary")):
        base = payload["models"][ref_epoch][comp_idx]
        base_par = np.asarray(base.parameters)
        for dT in dteffs:
            for dg in dloggs:
                for feh in fehs:
                    par = base_par.copy()
                    par[:, it] += dT
                    par[:, ig] += dg
                    par[:, ife] = feh
                    model = base._replace(parameters=jnp.asarray(par))
                    t0 = time.time()
                    spec = simulate_observed_flux(emu.intensity, model,
                                                  log_wavelengths,
                                                  disable_doppler_shift=False)
                    out[(comp, float(dT), float(dg), float(feh))] = (
                        np.asarray(spec[:, 0]), np.asarray(spec[:, 1]))
                    k += 1
                    if verbose:
                        print(f"  [{k}/{total}] {comp:<9} dT{dT:+6.0f} "
                              f"dlogg{dg:+5.2f} feh{feh:+5.2f}  "
                              f"{time.time()-t0:6.1f} s", flush=True)
    return out


def lightcurve_times(b, n_ecl, n_out):
    """Phases covering both eclipses densely plus an out-of-eclipse baseline.

    The previous hybrid photometry run sampled only the two eclipse windows and
    was compared against PHOEBE's own light curve, never against data. To test
    the chain end to end we need the out-of-eclipse level too, since that is
    what sets the magnitude zero point when fitting Clausen.
    """
    import jax.numpy as jnp
    from spice.models.orbit_utils import eclipse_timestamps_kepler
    _, t1p, _, _, t4p, _, t1s, _, _, t4s = eclipse_timestamps_kepler(
        b.get_parameter("mass@primary@component").value,
        b.get_parameter("mass@secondary@component").value,
        b.get_parameter("period@binary@component").value * DAYS_TO_YR,
        b.get_parameter("ecc@binary@component").value,
        b.get_parameter("t0_perpass@binary@component").value * DAYS_TO_YR,
        jnp.deg2rad(b.get_parameter("incl@binary@component").value),
        b.get_parameter("per0@binary@component").value * 0.017453292519943295,
        b.get_parameter("long_an@binary@component").value * 0.017453292519943295,
        b.get_parameter("requiv@primary@component").value,
        b.get_parameter("requiv@secondary@component").value,
        pad=1.25, los_vector=jnp.array([0.0, 0.0, -1.0]))
    e = [float(x) / DAYS_TO_YR for x in (t1p, t4p, t1s, t4s)]
    P = b.get_parameter("period@binary@component").value
    t0 = b.get_value("t0_supconj@binary@component")
    out = t0 + np.linspace(0.05, 0.45, n_out) * P      # clear of both eclipses
    return np.sort(np.concatenate([np.linspace(e[0], e[1], n_ecl),
                                   np.linspace(e[2], e[3], n_ecl), out]))


def report_teff_spread(teff_spread):
    """Guard the reason for importing a PHOEBE mesh at all.

    If teff arrives constant, the import collapsed to a uniform surface and
    gravity darkening / irradiation were silently discarded.
    """
    print("\nPHOEBE per-element values imported into SPICE:")
    for comp, tlo, thi, glo, ghi in teff_spread:
        flag = "   <-- CONSTANT: gravity darkening was LOST" if thi - tlo < 1e-6 else ""
        print(f"  {comp:<10} teff {tlo:8.2f}..{thi:8.2f} K (spread {thi-tlo:6.2f})"
              f"   logg {glo:5.3f}..{ghi:5.3f}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="all",
                    choices=["export", "synth", "all",
                             "export-spectra", "synth-spectra", "grid",
                             "export-lc", "synth-lc"],
                    help="'export'/'synth' are the photometry pair; "
                         "'export-spectra'/'synth-spectra' are the same split "
                         "for the HARPS comparison (meshes at the observed "
                         "epochs, then Doppler-shifted spectra). The 'export' "
                         "half needs PHOEBE; the 'synth' half needs aemu only "
                         "and is what goes to Gadi.")
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"),
                    help="HARPS ADP*.fits archive, for --mode export-spectra")
    ap.add_argument("--teff2", type=float, default=None,
                    help="override Teff2 (export-lc): the hybrid has only ever "
                         "run at the 6650 default and never against real data")
    ap.add_argument("--n-out", type=int, default=12,
                    help="out-of-eclipse epochs, needed to set the zero point")
    ap.add_argument("--dteff", default="-300,-150,0,150,300",
                    help="grid mode: Teff offsets [K] applied per star")
    ap.add_argument("--dlogg", default="-0.3,0,0.3",
                    help="grid mode: log g offsets [dex] applied per star")
    ap.add_argument("--feh", default="-0.5,-0.3,-0.1",
                    help="grid mode: [Fe/H] values (shared by both stars)")
    ap.add_argument("--n-per-window", type=int, default=2000,
                    help="samples per spectral window (~0.02 A over 40 A)")
    ap.add_argument("--meshes", default=None,
                    help="payload path: written by --mode export, read by --mode synth")
    ap.add_argument("--n-mesh", type=int, default=1000)
    ap.add_argument("--n-per-eclipse", type=int, default=15)
    ap.add_argument("--n-wavelengths", type=int, default=1500,
                    help="samples across 4300-5900 A (both Stromgren bands)")
    ap.add_argument("--distortion", default="roche",
                    choices=["roche", "sphere", "rotstar"])
    ap.add_argument("--no-gravity-darkening", action="store_true")
    ap.add_argument("--no-irradiation", action="store_true")
    ap.add_argument("--output", default=str(Path(__file__).parent / "tzfor_aemu_out"))
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "1"
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    stem = f"tzfor_aemu_{args.distortion}_n{args.n_mesh}"
    if "spectra" in args.mode:
        suffix = "_spectra_meshes.pkl"
    elif args.mode in ("export-lc", "synth-lc"):
        suffix = f"_lc{'' if args.teff2 is None else int(args.teff2)}_meshes.pkl"
    else:
        suffix = "_meshes.pkl"
    meshes = Path(args.meshes) if args.meshes else out / f"{stem}{suffix}"

    # ---- hybrid light curve vs the OBSERVED photometry ---------------------
    if args.mode == "export-lc":
        from spice.spectrum.aemu_spectrum_emulator import (
            IntensityPretrainedAemuSpectrumEmulator)
        names = IntensityPretrainedAemuSpectrumEmulator(BUNDLE).stellar_parameter_names
        b = build_bundle(args.n_mesh, args.distortion,
                         gravity_darkening=not args.no_gravity_darkening,
                         irradiation=not args.no_irradiation)
        if args.teff2 is not None:
            b.set_value("teff@secondary@component", float(args.teff2))
        T2 = b.get_value("teff@secondary@component")
        times = lightcurve_times(b, args.n_per_eclipse, args.n_out)
        b.add_dataset("mesh", compute_times=times, columns=MESH_COLUMNS, dataset="mesh01")
        print(f"Teff2={T2:.0f} K, {len(times)} epochs "
              f"({args.n_per_eclipse} per eclipse + {args.n_out} out)", flush=True)
        t0 = time.time()
        b.run_compute(irrad_method="none" if args.no_irradiation else "horvat",
                      ltte=False)
        print(f"  PHOEBE {time.time()-t0:.1f} s", flush=True)
        payload = export_meshes(b, times, names, meshes)
        payload["teff2"] = float(T2); payload["args"] = vars(args)
        # PHOEBE is not installed on the GPU nodes, so carry its transmission
        # curves in the payload rather than importing phoebe there.
        from phoebe.atmospheres import passbands as pbs
        payload["passbands"] = {
            n: (np.asarray(pbs.get_passband(n).ptf_table["wl"]).astype(np.float64)*1e10,
                np.asarray(pbs.get_passband(n).ptf_table["fl"]).astype(np.float64))
            for n in HYBRID_PHOT_BANDS}
        with open(meshes, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        report_teff_spread(payload["teff_spread"])
        print(f"\nwrote {meshes} ({meshes.stat().st_size/1e6:.1f} MB)")
        return

    if args.mode == "synth-lc":
        import jax.numpy as jnp
        from spice.spectrum.aemu_spectrum_emulator import (
            IntensityPretrainedAemuSpectrumEmulator)
        from spice.spectrum.spectrum import AB_passband_luminosity, simulate_observed_flux
        with open(meshes, "rb") as f:
            payload = pickle.load(f)
        report_teff_spread(payload["teff_spread"])
        emu = IntensityPretrainedAemuSpectrumEmulator(BUNDLE)
        from spice.spectrum.filter import Filter
        # Rebuilt from the curves carried in the payload; non_photonic=True to
        # match PHOEBE's energy weighting.
        filters = {n.split(":")[1]: Filter(jnp.array([w, f]), name=f"PHOEBE {n}",
                                           non_photonic=True)
                   for n, (w, f) in payload["passbands"].items()}
        lo, hi = HYBRID_PHOT_RANGE
        wavelengths = jnp.linspace(lo, hi, args.n_wavelengths)
        log_wl = jnp.log10(wavelengths)
        print(f"\nsynthesizing {len(payload['times'])} epochs, "
              f"{args.n_wavelengths} wavelengths, bands {list(filters)}", flush=True)
        mags = {k: [] for k in filters}
        t0 = time.time()
        for i, (m1, m2) in enumerate(payload["models"]):
            te = time.time()
            s1 = simulate_observed_flux(emu.intensity, m1, log_wl, disable_doppler_shift=True)
            s2 = simulate_observed_flux(emu.intensity, m2, log_wl, disable_doppler_shift=True)
            tot = s1[:, 0] + s2[:, 0]
            for k, f in filters.items():
                mags[k].append(float(AB_passband_luminosity(f, wavelengths, tot)))
            print(f"  epoch {i+1}/{len(payload['times'])} {time.time()-te:6.1f} s", flush=True)
        print(f"  done in {time.time()-t0:.1f} s", flush=True)
        res = out / f"tzfor_hybrid_lc_T{payload['teff2']:.0f}.pkl"
        with open(res, "wb") as f:
            pickle.dump({"times": payload["times"], "teff2": payload["teff2"],
                         "mags": {k: np.array(v) for k, v in mags.items()},
                         "args": vars(args)}, f, protocol=4)
        print(f"\nwrote {res}")
        return

    # ---- spectra: PHOEBE meshes at the observed HARPS epochs ---------------
    if args.mode == "export-spectra":
        from spice.spectrum.aemu_spectrum_emulator import (
            IntensityPretrainedAemuSpectrumEmulator)
        names = IntensityPretrainedAemuSpectrumEmulator(BUNDLE).stellar_parameter_names

        epochs = harps_epochs(args.data_dir)
        times = np.array([e[0] for e in epochs])
        print(f"{len(times)} HARPS epochs, phases "
              f"{((times - 2452599.29040) % 75.66647 / 75.66647).min():.3f}"
              f"..{((times - 2452599.29040) % 75.66647 / 75.66647).max():.3f}")

        b = build_bundle(args.n_mesh, args.distortion,
                         gravity_darkening=not args.no_gravity_darkening,
                         irradiation=not args.no_irradiation)
        for comp in ("primary", "secondary"):
            print(f"  syncpar@{comp} = "
                  f"{b.get_value(f'syncpar@{comp}@component'):.3f}")
        b.add_dataset("mesh", compute_times=times, columns=MESH_COLUMNS,
                      dataset="mesh01")
        t0 = time.time()
        b.run_compute(irrad_method="none" if args.no_irradiation else "horvat",
                      ltte=False)
        print(f"  PHOEBE done in {time.time()-t0:.1f} s", flush=True)

        payload = export_meshes(b, times, names, meshes)
        payload["harps_files"] = [e[1] for e in epochs]
        payload["args"] = vars(args)
        with open(meshes, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        report_teff_spread(payload["teff_spread"])
        print(f"\nwrote {meshes}  ({meshes.stat().st_size/1e6:.1f} MB)")
        return

    if args.mode == "grid":
        import jax.numpy as jnp
        from spice.spectrum.aemu_spectrum_emulator import (
            IntensityPretrainedAemuSpectrumEmulator)
        with open(meshes, "rb") as f:
            payload = pickle.load(f)
        report_teff_spread(payload["teff_spread"])
        emu = IntensityPretrainedAemuSpectrumEmulator(BUNDLE)
        wavelengths = spectral_grid(args.n_per_window)
        parse = lambda s: [float(x) for x in s.split(",")]
        dteffs, dloggs, fehs = parse(args.dteff), parse(args.dlogg), parse(args.feh)

        # Reference epoch: the one furthest from either eclipse, so the disc is
        # unocculted and the geometry is representative of all fitted epochs.
        times = np.asarray(payload["times"])
        ph = ((times - 2452599.29040) % 75.66647) / 75.66647
        ref = int(np.argmin(np.abs(np.minimum(np.abs(ph - 0.25),
                                              np.abs(ph - 0.75)) )))
        print(f"\nreference epoch {ref}: phase {ph[ref]:.3f} "
              f"({payload['harps_files'][ref]})")
        print(f"grid: {len(dteffs)} dTeff x {len(dloggs)} dlogg x {len(fehs)} feh "
              f"x 2 components = {2*len(dteffs)*len(dloggs)*len(fehs)} syntheses",
              flush=True)
        t0 = time.time()
        lib = synthesize_grid(payload, emu, wavelengths, dteffs, dloggs, fehs,
                              ref_epoch=ref)
        print(f"  done in {time.time()-t0:.1f} s", flush=True)
        res = out / f"tzfor_aemu_grid_n{args.n_mesh}.pkl"
        with open(res, "wb") as f:
            pickle.dump({"library": lib, "wavelengths": np.asarray(wavelengths),
                         "windows": SPECTRAL_WINDOWS, "dteffs": dteffs,
                         "dloggs": dloggs, "fehs": fehs, "ref_epoch": ref,
                         "ref_phase": float(ph[ref]),
                         "ref_file": payload["harps_files"][ref],
                         "times": times, "harps_files": payload["harps_files"],
                         "parameter_names": payload["parameter_names"],
                         "args": vars(args)}, f, protocol=4)
        print(f"\nwrote {res}")
        return

    if args.mode == "synth-spectra":
        import jax.numpy as jnp
        from spice.spectrum.aemu_spectrum_emulator import (
            IntensityPretrainedAemuSpectrumEmulator)
        with open(meshes, "rb") as f:
            payload = pickle.load(f)
        report_teff_spread(payload["teff_spread"])
        emu = IntensityPretrainedAemuSpectrumEmulator(BUNDLE)
        wavelengths = spectral_grid(args.n_per_window)
        for lo, hi, name in SPECTRAL_WINDOWS:
            print(f"  window {lo:.0f}-{hi:.0f} A  ({name})")
        print(f"\nsynthesizing {len(payload['times'])} epochs, "
              f"{wavelengths.shape[0]} wavelengths, Doppler ON ...", flush=True)
        t0 = time.time()
        spectra = synthesize_spectra(payload, emu, wavelengths)
        print(f"  done in {time.time()-t0:.1f} s", flush=True)
        res = out / f"tzfor_aemu_spectra_n{args.n_mesh}.pkl"
        with open(res, "wb") as f:
            pickle.dump({"times": payload["times"],
                         "harps_files": payload.get("harps_files"),
                         "wavelengths": np.asarray(wavelengths),
                         "windows": SPECTRAL_WINDOWS,
                         "spectra": spectra,
                         "teff_spread": payload["teff_spread"],
                         "args": vars(args)}, f, protocol=4)
        print(f"\nwrote {res}")
        return

    # ---- PHOEBE side -------------------------------------------------------
    if args.mode in ("export", "all"):
        # Deferred: naming the bundle's parameters is all we need from aemu here,
        # and the export host may not have a GPU.
        from spice.spectrum.aemu_spectrum_emulator import (
            IntensityPretrainedAemuSpectrumEmulator)
        names = IntensityPretrainedAemuSpectrumEmulator(BUNDLE).stellar_parameter_names

        b = build_bundle(args.n_mesh, args.distortion,
                         gravity_darkening=not args.no_gravity_darkening,
                         irradiation=not args.no_irradiation)
        times, edges, baseline, ref = _sample_times(b, args.n_per_eclipse)
        b.add_dataset("mesh", compute_times=times, columns=MESH_COLUMNS, dataset="mesh01")
        b.add_dataset("lc", compute_times=times, passband="Stromgren:b", dataset="lc_b")
        b.add_dataset("lc", compute_times=times, passband="Stromgren:y", dataset="lc_y")
        for ds in ("lc_b", "lc_y"):
            b.set_value_all("pblum_mode", dataset=ds, value="absolute")
        print(f"PHOEBE: {len(times)} epochs, ntriangles={args.n_mesh}, "
              f"{args.distortion} distortion ...", flush=True)
        t0 = time.time()
        b.run_compute(irrad_method="none" if args.no_irradiation else "horvat",
                      ltte=False)
        print(f"  done in {time.time()-t0:.1f} s", flush=True)

        payload = export_meshes(b, times, names, meshes)
        payload["phoebe_mag"] = {
            "b": -2.5 * np.log10(np.asarray(b.get_value("fluxes@lc_b@model"))),
            "y": -2.5 * np.log10(np.asarray(b.get_value("fluxes@lc_y@model"))),
        }
        payload["edges"], payload["baseline_index"] = edges, ref
        payload["args"] = vars(args)
        with open(meshes, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        report_teff_spread(payload["teff_spread"])
        print(f"\nwrote {meshes}  ({meshes.stat().st_size/1e6:.1f} MB)")
        if args.mode == "export":
            return

    # ---- emulator side (Gadi) ---------------------------------------------
    import jax.numpy as jnp
    from spice.spectrum.aemu_spectrum_emulator import (
        IntensityPretrainedAemuSpectrumEmulator)
    from spice.spectrum.filter import Stromgrenb, Stromgreny

    with open(meshes, "rb") as f:
        payload = pickle.load(f)
    report_teff_spread(payload["teff_spread"])

    emu = IntensityPretrainedAemuSpectrumEmulator(BUNDLE)
    filters = {"b": Stromgrenb(), "y": Stromgreny()}
    wavelengths = jnp.linspace(4300.0, 5900.0, args.n_wavelengths)

    print(f"\nsynthesizing {len(payload['times'])} epochs with {BUNDLE} "
          f"({args.n_wavelengths} wavelengths) ...", flush=True)
    t0 = time.time()
    spice_mag = synthesize_from_payload(payload, emu, wavelengths, filters)
    print(f"  done in {time.time()-t0:.1f} s", flush=True)

    ref = payload["baseline_index"]
    n = payload["args"]["n_per_eclipse"]
    prim, sec = slice(0, ref), slice(ref + 1, ref + 1 + n)
    phoebe_mag = payload["phoebe_mag"]
    print(f"\n{'band':<5}{'eclipse':<12}{'PHOEBE':>12}{'SPICE+aemu':>13}"
          f"{'diff':>10}{'rel':>9}")
    rows = []
    for band in ("b", "y"):
        for label, sl in (("primary", prim), ("secondary", sec)):
            dp = float(np.max(phoebe_mag[band][sl] - phoebe_mag[band][ref]))
            ds = float(np.max(spice_mag[band][sl] - spice_mag[band][ref]))
            rows.append((band, label, dp, ds))
            print(f"{band:<5}{label:<12}{dp:>12.5f}{ds:>13.5f}"
                  f"{ds-dp:>+10.5f}{(ds-dp)/dp*100:>+8.2f}%")

    with open(out / f"{stem}_result.pkl", "wb") as f:
        pickle.dump({"times": payload["times"], "edges": payload["edges"],
                     "baseline_index": ref, "phoebe_mag": phoebe_mag,
                     "spice_mag": spice_mag, "depths": rows,
                     "teff_spread": payload["teff_spread"],
                     "args": vars(args)}, f)
    print(f"\nwrote {out / (stem + '_result.pkl')}")


if __name__ == "__main__":
    main()
