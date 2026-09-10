"""PHOEBE vs SPICE for TZ Fornacis specifically.

``check_phoebe_eclipses.py`` sweeps a synthetic grid of geometries; this script
pins the comparison to the actual TZ For system (Andersen et al. 1991 /
Gallenne et al. 2016, via ``tzfor_constants``) so the two codes can be compared
on the configuration the paper actually uses.

Both codes are configured to isolate *geometry and Planck weighting* -- the
things SPICE is responsible for -- from atmosphere physics neither code shares:

* PHOEBE: ``atm='blackbody'``, ``ld_func='linear'`` with ``ld_coeffs=[0]``
  (uniform disk, matching SPICE's ``Blackbody`` emulator, which has no limb
  darkening), ``irrad_method='none'``, ``gravb_bol=0``, and
  ``distortion_method='sphere'`` so PHOEBE's Roche surface matches SPICE's
  spherical icosphere. ``--distortion roche`` relaxes the last one to measure
  what the sphericity assumption costs for this (well-detached) system.
* Both meshes get the same element count.

Every orbital element is *read back out of the PHOEBE bundle* after the TZ For
values are set, so the two codes cannot be fed different orbits.

Three comparisons are reported:

1. light curves through both eclipses in Bolometric / Stromgren b / Stromgren y,
   differenced against the out-of-eclipse baseline (so the absolute AB
   zero-point convention drops out -- see the photonic-branch unit caveat in
   ``AB_passband_luminosity``);
2. eclipse depths per band, which encode the light ratio as well as the
   geometry;
3. radial velocities over the full orbit -- flux-weighted PHOEBE ``rv`` dataset
   against SPICE's ``los_velocities``. This exercises the PHOEBE line-of-sight
   convention directly (see ``tests/test_phoebe_velocity_convention.py``).
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tz_fornacis"))

DAYS_TO_YR = 0.0027378507871321013
DEG_TO_RAD = 0.017453292519943295

_PHOEBE_INSTALL_HINT = (
    "PHOEBE is required for this script but is not installed. Install via the "
    'optional extra declared in pyproject.toml: `pip install "stellar-spice[phoebe]"`.'
)


def _import_phoebe():
    try:
        import phoebe
    except ImportError as exc:
        raise ValueError(_PHOEBE_INSTALL_HINT) from exc
    return phoebe


def build_bundle(distortion="sphere", n_mesh=5120):
    """A PHOEBE bundle holding the TZ For literature parameters."""
    phoebe = _import_phoebe()
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

    b.set_value("requiv@primary@component", K.PRIMARY_RADIUS)
    b.set_value("requiv@secondary@component", K.SECONDARY_RADIUS)
    b.set_value("teff@primary@component", K.PRIMARY_TEFF)
    b.set_value("teff@secondary@component", K.SECONDARY_TEFF)

    b.set_value("distance", K.DISTANCE_PC * u.pc)
    b.set_value_all("distortion_method", distortion)
    b.set_value_all("ntriangles", n_mesh)
    return b


def phoebe_filter(phoebe_name, non_photonic=True):
    """Build a SPICE ``Filter`` from PHOEBE's own transmission table.

    Using PHOEBE's curves for both codes removes passband definition as a free
    variable, so a residual difference can only come from the geometry or the
    radiative transfer. Two traps this avoids:

    * PHOEBE stores ``ptf_table['wl']`` in **metres**, and its bolometric
      passband is named in nanometres -- ``Bolometric:900-40000`` really spans
      9000-400000 A (0.9-40 um), not 900-40000 A. Comparing SPICE's stock
      ``Bolometric`` (1-30000 A) against it charges a ~20% light-ratio error to
      the geometry for a 4930/6650 K pair.
    * ``non_photonic=True`` selects the energy-weighted branch of
      ``AB_passband_luminosity``, matching PHOEBE's default
      ``intens_weighting='energy'``; the default (photon-counting) branch adds
      a factor of lambda to the integrand.
    """
    import jax.numpy as jnp
    from phoebe.atmospheres import passbands as pbs
    from spice.spectrum.filter import Filter

    pb = pbs.get_passband(phoebe_name)
    # PHOEBE's tables come out of FITS as big-endian ('>f8'), which JAX rejects;
    # astype(float) also gives native byte order.
    wl_angstrom = np.asarray(pb.ptf_table["wl"]).astype(np.float64) * 1e10  # m -> A
    transmission = np.asarray(pb.ptf_table["fl"]).astype(np.float64)
    return Filter(
        jnp.array([wl_angstrom, transmission]),
        name=f"PHOEBE {phoebe_name}",
        non_photonic=non_photonic,
    )


def _configure_atmosphere(b):
    """Uniform-disk blackbody, no irradiation, no gravity darkening."""
    b.set_value_all("ld_mode", "manual")
    b.set_value_all("ld_func", "linear")
    b.set_value_all("ld_coeffs", [0.0])
    b.set_value_all("ld_mode_bol", "manual")
    b.set_value_all("ld_func_bol", "linear")
    b.set_value_all("ld_coeffs_bol", [0.0])
    b.set_value_all("atm", "blackbody")
    b.set_value_all("irrad_method", "none")
    b.set_value_all("gravb_bol", 0.0)
    # phoebe_filter() builds the SPICE side with non_photonic=True; pin the
    # PHOEBE side to match rather than trusting the default.
    b.set_value_all("intens_weighting", "energy")


def spice_binary(b, n_mesh):
    """The SPICE counterpart, with every element taken from the bundle."""
    import jax.numpy as jnp
    from spice.models.binary import Binary, add_orbit
    from spice.models.mesh_model import IcosphereModel
    from spice.models.mesh_view import get_mesh_view
    from spice.spectrum.blackbody import Blackbody

    bb = Blackbody()

    def body(mass, radius, teff):
        return get_mesh_view(
            IcosphereModel.construct(
                n_mesh, radius, mass, jnp.array([teff, 0.0]), ["teff", "abun"]
            ),
            jnp.array([0.0, 0.0, -1.0]),
        )

    body1 = body(
        b.get_parameter("mass@primary@component").value,
        b.get_parameter("requiv@primary@component").value,
        b.get_parameter("teff@primary@component").value,
    )
    body2 = body(
        b.get_parameter("mass@secondary@component").value,
        b.get_parameter("requiv@secondary@component").value,
        b.get_parameter("teff@secondary@component").value,
    )
    binary = Binary.from_bodies(body1, body2)
    binary = add_orbit(
        binary,
        P=b.get_parameter("period@binary@component").value * DAYS_TO_YR,
        ecc=b.get_parameter("ecc@binary@component").value,
        T=0.0,
        i=jnp.deg2rad(b.get_parameter("incl@binary@component").value),
        omega=b.get_parameter("per0@binary@component").value * DEG_TO_RAD,
        Omega=b.get_parameter("long_an@binary@component").value * DEG_TO_RAD,
        vgamma=b.get_parameter("vgamma").value,
        reference_time=b.get_parameter("t0_ref@binary@component").value * DAYS_TO_YR,
        mean_anomaly=b.get_parameter("mean_anom@binary@component").value * DEG_TO_RAD,
        # The eclipse window is ~3% of the 75.7 d period, so the position grid
        # add_orbit interpolates on has to be dense or ingress/egress are
        # effectively extrapolated (same reasoning as check_phoebe_eclipses).
        orbit_resolution_points=20000,
    )
    return binary, bb


def eclipse_windows(b):
    """Eclipse contact times (days) from SPICE's own Kepler solver."""
    import jax.numpy as jnp
    from spice.models.orbit_utils import eclipse_timestamps_kepler

    _, t1_p, _, _, t4_p, _, t1_s, _, _, t4_s = eclipse_timestamps_kepler(
        b.get_parameter("mass@primary@component").value,
        b.get_parameter("mass@secondary@component").value,
        b.get_parameter("period@binary@component").value * DAYS_TO_YR,
        b.get_parameter("ecc@binary@component").value,
        b.get_parameter("t0_perpass@binary@component").value * DAYS_TO_YR,
        jnp.deg2rad(b.get_parameter("incl@binary@component").value),
        b.get_parameter("per0@binary@component").value * DEG_TO_RAD,
        b.get_parameter("long_an@binary@component").value * DEG_TO_RAD,
        b.get_parameter("requiv@primary@component").value,
        b.get_parameter("requiv@secondary@component").value,
        pad=1.15,
        los_vector=jnp.array([0.0, 0.0, -1.0]),
    )
    return tuple(float(t) / DAYS_TO_YR for t in (t1_p, t4_p, t1_s, t4_s))


def _sample_times(b, n_per_eclipse):
    """Both eclipse windows plus a quadrature baseline for normalisation.

    The array is returned **sorted**: PHOEBE sorts ``compute_times`` internally
    and returns its model arrays in that order, whereas SPICE's
    ``evaluate_orbit_at_times`` preserves the caller's order. Handing an
    unsorted array to both silently mis-pairs the two light curves (an
    interleaved baseline shifts one eclipse block by exactly one sample).
    ``light_curves`` additionally asserts PHOEBE echoed the times back
    unpermuted, so this can't regress silently.
    """
    t1_p, t4_p, t1_s, t4_s = eclipse_windows(b)
    if not all(np.isfinite([t1_p, t4_p, t1_s, t4_s])):
        raise RuntimeError(f"no eclipse found for TZ For geometry: {(t1_p, t4_p, t1_s, t4_s)}")
    period = b.get_parameter("period@binary@component").value
    # Quadrature relative to the primary eclipse centre: a quarter period away
    # is guaranteed outside both windows for this well-detached system.
    baseline = 0.5 * (t1_p + t4_p) + 0.25 * period
    primary = np.linspace(t1_p, t4_p, n_per_eclipse)
    secondary = np.linspace(t1_s, t4_s, n_per_eclipse)
    times = np.sort(np.concatenate([[baseline], primary, secondary]))
    baseline_index = int(np.argmin(np.abs(times - baseline)))
    return times, (t1_p, t4_p, t1_s, t4_s), baseline, baseline_index


def light_curves(b, n_mesh, n_per_eclipse, n_wavelengths):
    """Bolometric + Stromgren b/y light curves from both codes."""
    import jax.numpy as jnp
    from spice.models.binary import evaluate_orbit_at_times
    from spice.spectrum.spectrum import AB_passband_luminosity, simulate_observed_flux

    times, edges, baseline, baseline_index = _sample_times(b, n_per_eclipse)

    # Every passband comes from PHOEBE's own table -- see phoebe_filter().
    filters = {
        "bol": phoebe_filter("Bolometric:900-40000"),
        "b": phoebe_filter("Stromgren:b"),
        "y": phoebe_filter("Stromgren:y"),
    }
    b.add_dataset("orb", compute_times=times, dataset="orb01")
    b.add_dataset("lc", compute_times=times,
                  passband="Bolometric:900-40000", dataset="lc_bol")
    b.add_dataset("lc", compute_times=times, passband="Stromgren:b", dataset="lc_b")
    b.add_dataset("lc", compute_times=times, passband="Stromgren:y", dataset="lc_y")
    for ds in ("lc_bol", "lc_b", "lc_y"):
        b.set_value_all("pblum_mode", dataset=ds, value="absolute")
    _configure_atmosphere(b)

    t0 = time.time()
    b.run_compute(irrad_method="none", ltte=False)
    phoebe_seconds = time.time() - t0
    for ds in ("lc_bol", "lc_b", "lc_y"):
        echoed = np.asarray(b.get_value(f"times@{ds}@model"))
        if not np.allclose(echoed, times, rtol=0, atol=1e-9):
            raise RuntimeError(
                f"PHOEBE reordered the times for {ds}; the light curves would be "
                f"mis-paired with SPICE's (first mismatch at index "
                f"{int(np.argmax(~np.isclose(echoed, times)))})"
            )
    phoebe_flux = {
        "bol": np.asarray(b.get_value("fluxes@lc_bol@model")),
        "b": np.asarray(b.get_value("fluxes@lc_b@model")),
        "y": np.asarray(b.get_value("fluxes@lc_y@model")),
    }

    binary, bb = spice_binary(b, n_mesh)
    pb1, pb2 = evaluate_orbit_at_times(binary, times * DAYS_TO_YR)

    # One log-spaced grid serves all three passbands, spanning the union of
    # their supports (the bolometric band reaches 40 um). Log spacing keeps the
    # resolution constant in R: at the default n_wavelengths that is ~10 A at
    # 4700 A, i.e. ~70 samples across the 700 A-wide Stromgren curves.
    # NB: despite its name this property returns the full 2xN curve; row 0 is wl.
    supports = [np.asarray(f.transmission_curve_wavelengths)[0] for f in filters.values()]
    lo = min(float(s.min()) for s in supports)
    hi = max(float(s.max()) for s in supports)
    wavelengths = jnp.logspace(np.log10(lo), np.log10(hi), n_wavelengths)
    log_wavelengths = jnp.log10(wavelengths)

    t0 = time.time()
    spice_mag = {k: [] for k in filters}
    spice_ratio = None
    for i, (_pb1, _pb2) in enumerate(zip(pb1, pb2)):
        spec1 = simulate_observed_flux(bb.intensity, _pb1, log_wavelengths,
                                       disable_doppler_shift=True)
        spec2 = simulate_observed_flux(bb.intensity, _pb2, log_wavelengths,
                                       disable_doppler_shift=True)
        total = spec1[:, 0] + spec2[:, 0]
        for key, filt in filters.items():
            spice_mag[key].append(float(AB_passband_luminosity(filt, wavelengths, total)))
        if i == baseline_index:
            # Out of eclipse: the light ratio each code predicts. Eclipse depth
            # is essentially (blocked area) x (that body's share of the light),
            # so a light-ratio error shows up as depth errors of *opposite*
            # sign in the two eclipses -- the signature that distinguishes it
            # from a geometry error, which biases both the same way.
            spice_ratio = {
                key: float(np.trapezoid(
                        np.asarray(spec2[:, 0]) * np.asarray(
                            filt.filter_responses_for_wavelengths(wavelengths)),
                        np.asarray(wavelengths))
                    / np.trapezoid(
                        np.asarray(spec1[:, 0]) * np.asarray(
                            filt.filter_responses_for_wavelengths(wavelengths)),
                        np.asarray(wavelengths)))
                for key, filt in filters.items()
            }
    spice_seconds = time.time() - t0
    spice_mag = {k: np.array(v) for k, v in spice_mag.items()}

    # PHOEBE's equivalent: passband luminosities per component.
    phoebe_ratio = {}
    for key, ds in (("bol", "lc_bol"), ("b", "lc_b"), ("y", "lc_y")):
        pbl = b.compute_pblums(dataset=ds)
        l1 = float(pbl[f"pblum@primary@{ds}"].value)
        l2 = float(pbl[f"pblum@secondary@{ds}"].value)
        phoebe_ratio[key] = l2 / l1

    # Differential magnitudes against the baseline sample (index 0), so the AB
    # zero-point convention and any absolute-scale difference cancel.
    ref = baseline_index
    phoebe_mag = {k: -2.5 * np.log10(v / v[ref]) for k, v in phoebe_flux.items()}
    spice_dmag = {k: v - v[ref] for k, v in spice_mag.items()}

    return {
        "times": times,
        "edges": edges,
        "baseline_time": baseline,
        "baseline_index": baseline_index,
        "phoebe_flux": phoebe_flux,
        "phoebe_dmag": phoebe_mag,
        "spice_mag": spice_mag,
        "spice_dmag": spice_dmag,
        "spice_ratio": spice_ratio,
        "phoebe_ratio": phoebe_ratio,
        "n_per_eclipse": n_per_eclipse,
        "phoebe_seconds": phoebe_seconds,
        "spice_seconds": spice_seconds,
    }


def radial_velocities(b, n_mesh, n_phases):
    """Flux-weighted RVs over a full orbit, PHOEBE's rv dataset vs SPICE.

    SPICE has no rv dataset, so its RV is the projected-area-weighted mean of
    ``los_velocities`` -- the same quantity PHOEBE's ``rv`` reduces to for a
    uniform disk, up to the per-element intensity weight (identical here, since
    both bodies are isothermal blackbodies).
    """
    import numpy as _np
    from spice.models.binary import evaluate_orbit_at_times

    period = b.get_parameter("period@binary@component").value
    t0_supconj = b.get_parameter("t0_supconj@binary@component").value
    times = t0_supconj + _np.linspace(0.0, period, n_phases, endpoint=False)

    b.add_dataset("rv", compute_times=times, dataset="rv_full")
    _configure_atmosphere(b)
    b.run_compute(irrad_method="none", ltte=False)

    phoebe_rv = {
        "primary": _np.asarray(b.get_value("rvs@primary@rv_full@model")),
        "secondary": _np.asarray(b.get_value("rvs@secondary@rv_full@model")),
    }

    binary, _ = spice_binary(b, n_mesh)
    pb1, pb2 = evaluate_orbit_at_times(binary, times * DAYS_TO_YR)

    def weighted_rv(bodies):
        out = []
        for body in bodies:
            v = _np.asarray(body.los_velocities)
            # visible_cast_areas is the mu-projected area minus the occluded
            # part, zeroed on the far hemisphere: the visible-light weight for
            # a uniform disk.
            w = _np.clip(_np.asarray(body.visible_cast_areas), 0.0, None)
            out.append(float(_np.sum(v * w) / _np.sum(w)))
        return _np.array(out)

    return {
        "times": times,
        "phases": (times - t0_supconj) / period,
        "phoebe_rv": phoebe_rv,
        "spice_rv": {"primary": weighted_rv(pb1), "secondary": weighted_rv(pb2)},
    }


def _depth(dmag, edges_slice):
    """Maximum depth (mag) inside one eclipse window."""
    return float(np.nanmax(dmag[edges_slice]))


def report(lc, rv, meta):
    n = lc["n_per_eclipse"]
    ref = lc["baseline_index"]
    # Sorted layout: the whole primary window precedes the quadrature baseline,
    # which precedes the whole secondary window.
    prim = slice(0, ref)
    sec = slice(ref + 1, ref + 1 + n)
    assert ref == n, f"unexpected baseline position {ref} (expected {n})"

    lines = []
    add = lines.append
    add("=" * 74)
    add(f"TZ Fornacis: PHOEBE vs SPICE   ({meta['distortion']} distortion, "
        f"{meta['n_mesh']} elements)")
    add("=" * 74)
    add(f"  sma = {meta['sma']:.4f} Rsun,  requiv = "
        f"{meta['requiv1']:.3f} / {meta['requiv2']:.3f} Rsun,  "
        f"incl = {meta['incl']:.3f} deg")
    add(f"  PHOEBE {lc['phoebe_seconds']:.1f} s,  SPICE {lc['spice_seconds']:.1f} s "
        f"for {len(lc['times'])} epochs")
    add("")
    add("Eclipse depths [mag]")
    add(f"  {'band':<6}{'eclipse':<12}{'PHOEBE':>10}{'SPICE':>10}{'diff':>10}{'rel':>9}")
    for band in ("bol", "b", "y"):
        for label, sl in (("primary", prim), ("secondary", sec)):
            dp = _depth(lc["phoebe_dmag"][band], sl)
            ds = _depth(lc["spice_dmag"][band], sl)
            rel = (ds - dp) / dp * 100 if dp else float("nan")
            add(f"  {band:<6}{label:<12}{dp:>10.5f}{ds:>10.5f}"
                f"{ds - dp:>+10.5f}{rel:>+8.2f}%")
    add("")
    add("Out-of-eclipse light ratio L2/L1 (sets the eclipse depths)")
    add(f"  {'band':<6}{'PHOEBE':>12}{'SPICE':>12}{'rel diff':>11}")
    for band in ("bol", "b", "y"):
        pr = lc["phoebe_ratio"].get(band, float("nan"))
        sr = lc["spice_ratio"][band]
        add(f"  {band:<6}{pr:>12.5f}{sr:>12.5f}{(sr - pr) / pr * 100:>+10.2f}%")
    add("")
    add("Light-curve residual (SPICE - PHOEBE) [mmag]")
    add(f"  {'band':<6}{'eclipse':<12}{'rms':>10}{'max|.|':>10}")
    for band in ("bol", "b", "y"):
        for label, sl in (("primary", prim), ("secondary", sec)):
            r = (lc["spice_dmag"][band][sl] - lc["phoebe_dmag"][band][sl]) * 1e3
            add(f"  {band:<6}{label:<12}{np.sqrt(np.mean(r ** 2)):>10.3f}"
                f"{np.max(np.abs(r)):>10.3f}")
    add("")
    add("Colour of the residual: b - y depth difference [mmag]")
    for label, sl in (("primary", prim), ("secondary", sec)):
        db = (_depth(lc["spice_dmag"]["b"], sl) - _depth(lc["phoebe_dmag"]["b"], sl)) * 1e3
        dy = (_depth(lc["spice_dmag"]["y"], sl) - _depth(lc["phoebe_dmag"]["y"], sl)) * 1e3
        add(f"  {label:<12}b {db:>+8.3f}   y {dy:>+8.3f}   b-y {db - dy:>+8.3f}")
    add("")
    add("Radial velocities over one orbit [km/s]")
    add(f"  {'component':<12}{'PHOEBE K':>10}{'SPICE K':>10}{'rms diff':>11}{'max diff':>11}")
    for comp in ("primary", "secondary"):
        p_rv = rv["phoebe_rv"][comp]
        s_rv = rv["spice_rv"][comp]
        kp = 0.5 * (np.nanmax(p_rv) - np.nanmin(p_rv))
        ks = 0.5 * (np.nanmax(s_rv) - np.nanmin(s_rv))
        d = s_rv - p_rv
        add(f"  {comp:<12}{kp:>10.4f}{ks:>10.4f}"
            f"{np.sqrt(np.nanmean(d ** 2)):>11.4f}{np.nanmax(np.abs(d)):>11.4f}")
    add("=" * 74)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-mesh", type=int, default=5120,
                        help="mesh elements for both codes (SPICE snaps to an "
                             "icosphere subdivision level: 1280 / 5120 / 20480)")
    parser.add_argument("--n-per-eclipse", type=int, default=40)
    parser.add_argument("--n-wavelengths", type=int, default=1500)
    parser.add_argument("--n-rv-phases", type=int, default=60)
    parser.add_argument("--distortion", default="sphere",
                        choices=["sphere", "roche", "rotstar"],
                        help="PHOEBE distortion method; 'sphere' matches SPICE's "
                             "icosphere, 'roche' measures what sphericity costs")
    parser.add_argument("--skip-rv", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).parent / "tzfor_out"))
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # Match the precision the test-suite runs at: in float32 the Planck function
    # overflows at the 900 A end of the bolometric grid.
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    b = build_bundle(distortion=args.distortion, n_mesh=args.n_mesh)
    meta = {
        "distortion": args.distortion,
        "n_mesh": args.n_mesh,
        "sma": b.get_parameter("sma@binary@component").value,
        "requiv1": b.get_parameter("requiv@primary@component").value,
        "requiv2": b.get_parameter("requiv@secondary@component").value,
        "incl": b.get_parameter("incl@binary@component").value,
    }

    lc = light_curves(b, args.n_mesh, args.n_per_eclipse, args.n_wavelengths)
    rv = ({"phoebe_rv": {"primary": np.array([np.nan]), "secondary": np.array([np.nan])},
           "spice_rv": {"primary": np.array([np.nan]), "secondary": np.array([np.nan])},
           "times": np.array([]), "phases": np.array([])}
          if args.skip_rv else
          radial_velocities(build_bundle(args.distortion, args.n_mesh),
                            args.n_mesh, args.n_rv_phases))

    text = report(lc, rv, meta)
    print(text)

    stem = f"tzfor_{args.distortion}_n{args.n_mesh}"
    (out / f"{stem}.txt").write_text(text + "\n")
    with open(out / f"{stem}.pkl", "wb") as f:
        pickle.dump({"lc": lc, "rv": rv, "meta": meta}, f)
    print(f"\nwrote {out / (stem + '.pkl')}")


if __name__ == "__main__":
    main()
