"""End-to-end binary light curves for one comparison suite.

Layer 3 of ``spice_vs_phoebe_atmospheres.py`` (invoked as ``--mode binary``).
The intensity grid in that script predicts, from single-star quantities alone,
how much the light ratio and hence the eclipse depths should move; this module
checks that the predicted differences are the ones that actually appear once
both codes build a mesh, occlude it and integrate over an orbit.

**Two suites, run separately, never mixed.**

``--suite atmospheres``
    SPICE's MARCS aemu bundle against PHOEBE ``ck2004`` and ``phoenix``. Both
    sides carry their own limb darkening -- mu is an input channel of the
    bundle, and PHOEBE uses ``ld_mode='interp'`` to read its tabulated I(mu) --
    so the residual is radiative transfer.

``--suite blackbody``
    SPICE's ``Blackbody`` emulator against PHOEBE ``atm='blackbody'``, both as
    uniform discs. This is the control: the physics is *identical* on the two
    sides, so whatever is left is geometry, meshing and passband integration,
    and it sets the floor below which the atmospheres suite cannot resolve
    anything.

Comparing the MARCS emulator against PHOEBE's blackbody would charge the
difference between two different atmospheres to the two codes, which is why the
suites are kept apart rather than run as a cross-product.

Settings common to both suites, all chosen so that only the intended difference
survives:

    distortion_method = 'sphere'   PHOEBE's surface matches SPICE's icosphere
    irrad_method      = 'none'     no reflection/irradiation in either code
    gravb_bol         = 0.0        SPICE's icosphere has no gravity darkening
    ld_*_bol          = linear, [0]  bolometric LD only feeds irradiation
    intens_weighting  = 'energy'   matches non_photonic=True on the SPICE side
    pblum_mode        = 'absolute' no renormalisation between the codes

Cost warning: the aemu light curve calls the intensity bundle once per surface
element per epoch and is by far the slowest thing here. The blackbody suite is
~100x faster.
"""
import time

import numpy as np

DAYS_TO_YR = 0.0027378507871321013
DEG_TO_RAD = 0.017453292519943295

# Representative systems, spanning the axes the intensity grid says matter.
# Masses/radii are chosen so both components are well inside their Roche lobes
# and the system eclipses at the quoted inclination.
SYSTEMS = {
    # Equal twins: the light ratio is 1 by symmetry, so any depth difference is
    # limb darkening and geometry alone -- the control.
    "twins": dict(m1=1.0, q=1.0, r1=1.0, r2=1.0, teff1=5800.0, teff2=5800.0,
                  feh=0.0, period=5.0, incl=89.0),
    # Large colour contrast (TZ For-like): where the atmospheres disagree most,
    # because line blanketing acts asymmetrically on a cool and a warm star.
    "cool-hot": dict(m1=2.06, q=0.952, r1=8.28, r2=3.94, teff1=4930.0,
                     teff2=6650.0, feh=-0.30, period=75.7, incl=85.7),
    # The same geometry at low metallicity: line blanketing is much weaker, so
    # the blackbody should be a *better* approximation here than above.
    "metal-poor": dict(m1=2.06, q=0.952, r1=8.28, r2=3.94, teff1=4930.0,
                       teff2=6650.0, feh=-2.00, period=75.7, incl=85.7),
}

BANDS = ["Johnson:B", "Johnson:V", "Stromgren:b", "Stromgren:y"]


def build_bundle(system, n_mesh, distortion="sphere"):
    """A PHOEBE bundle for one entry of :data:`SYSTEMS`."""
    import phoebe

    phoebe.logger(clevel="ERROR")
    b = phoebe.default_binary()
    b.flip_constraint("mass@primary", solve_for="sma")

    b.set_value("period@binary@component", system["period"])
    b.set_value("q@binary@component", system["q"])
    b.set_value("ecc@binary@component", 0.0)
    b.set_value("mass@primary@component", system["m1"])
    b.set_value_all("incl@binary", system["incl"])
    b.set_value("requiv@primary@component", system["r1"])
    b.set_value("requiv@secondary@component", system["r2"])
    b.set_value("teff@primary@component", system["teff1"])
    b.set_value("teff@secondary@component", system["teff2"])
    b.set_value_all("abun", system["feh"])
    b.set_value_all("distortion_method", distortion)
    b.set_value_all("ntriangles", n_mesh)
    return b


def configure_atmosphere(b, atm):
    """Pin PHOEBE to one atmosphere, with the limb darkening that belongs to it.

    ``blackbody`` gets a uniform disc (``ld_mode='manual'``,
    ``ld_func='linear'``, ``ld_coeffs=[0]``) because that is what SPICE's
    ``Blackbody`` emulator is -- it has no mu dependence at all.

    ``ck2004`` / ``phoenix`` get ``ld_mode='interp'``, which reads PHOEBE's
    tabulated ``I(mu)`` directly and is the analogue of the aemu bundle's native
    mu channel. The alternative, ``ld_mode='lookup'``, fits a parametric law to
    the same table, which would charge the fit residual to the atmosphere
    comparison.

    Two naming traps in this one setting:

    * "interp" is a value of **ld_mode**, not of ``ld_func`` (whose choices are
      the parametric laws: linear, logarithmic, quadratic, square_root, power).
      ``ld_func='interp'`` raises ValueError.
    * under ``ld_mode='interp'`` there is no ``ldatm`` parameter at all -- the
      limb darkening comes from ``atm`` itself, so PHOEBE removes it and
      ``set_value_all('ldatm', ...)`` fails with "no parameters found". It is
      only meaningful under 'lookup', where the LD table may differ from the
      intensity table.

    Irradiation and gravity darkening are off in every case: neither is part of
    what is being compared, and SPICE's icosphere has no gravity darkening to
    match.
    """
    b.set_value_all("atm", atm)
    if atm == "blackbody":
        # The control suite. A uniform disc: ld_coeffs=[0] in the linear law
        # means L(mu) = 1, exactly what SPICE's Blackbody emulator is (no mu
        # dependence at all). With identical physics on both sides, any residual
        # is geometry, meshing or passband integration.
        b.set_value_all("ld_mode", "manual")
        b.set_value_all("ld_func", "linear")
        b.set_value_all("ld_coeffs", [0.0])
    else:
        b.set_value_all("ld_mode", "interp")
    # Bolometric LD only feeds irradiation, which is off, but PHOEBE still
    # validates it, so keep it manual and trivial.
    b.set_value_all("ld_mode_bol", "manual")
    b.set_value_all("ld_func_bol", "linear")
    b.set_value_all("ld_coeffs_bol", [0.0])
    b.set_value_all("irrad_method", "none")
    b.set_value_all("gravb_bol", 0.0)
    # Match the energy-weighted branch used on the SPICE side.
    b.set_value_all("intens_weighting", "energy")


def realized_triangles(system, n_mesh, distortion="sphere"):
    """How many triangles PHOEBE actually builds for a given ``ntriangles``.

    ``ntriangles`` is a *target* for PHOEBE's marching-triangles algorithm, not
    an exact count: a request of 1280 comes back as 1458. SPICE's icosphere, by
    contrast, is exact at its subdivision levels (1280 / 5120 / 20480 = 20*4^n).
    Recording the realized number keeps the two codes' resolutions honestly
    labelled instead of both being called "1280".
    """
    import phoebe
    phoebe.logger(clevel="ERROR")
    b = build_bundle(system, n_mesh, distortion)
    # ld_mode lives on lc datasets, so configure_atmosphere needs one to exist.
    # PHOEBE rejects a leading underscore in a dataset label ("first character
    # of label is a forbidden character"), so these cannot be named _lc / _m.
    b.add_dataset("lc", compute_times=[0.0], passband="Johnson:V", dataset="meshprobe")
    b.add_dataset("mesh", compute_times=[0.0], columns=["areas"], dataset="meshprobemesh")
    configure_atmosphere(b, "blackbody")
    b.run_compute(irrad_method="none", ltte=False)
    return {c: int(np.asarray(b.get_value("areas", dataset="meshprobemesh",
                                          component=c, context="model")).size)
            for c in ("primary", "secondary")}


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


def sample_times(b, n_per_eclipse):
    """Both eclipse windows plus a quadrature baseline, **sorted**.

    PHOEBE sorts ``compute_times`` internally and returns its model arrays in
    that order, whereas SPICE's ``evaluate_orbit_at_times`` preserves the
    caller's order; handing an unsorted array to both silently mis-pairs the two
    light curves. ``run_binary`` additionally asserts PHOEBE echoed the times
    back unpermuted.
    """
    t1_p, t4_p, t1_s, t4_s = eclipse_windows(b)
    if not all(np.isfinite([t1_p, t4_p, t1_s, t4_s])):
        raise RuntimeError(
            f"no eclipse at this geometry: {(t1_p, t4_p, t1_s, t4_s)}")
    period = b.get_parameter("period@binary@component").value
    baseline = 0.5 * (t1_p + t4_p) + 0.25 * period
    times = np.sort(np.concatenate([
        [baseline],
        np.linspace(t1_p, t4_p, n_per_eclipse),
        np.linspace(t1_s, t4_s, n_per_eclipse),
    ]))
    return times, (t1_p, t4_p, t1_s, t4_s), int(np.argmin(np.abs(times - baseline)))


def spice_binary(b, n_mesh, emu, feh, vmicro, blackbody=False, n_neighbours=None):
    """SPICE's own icosphere binary, with every element read out of the bundle.

    ``log_g_index`` is passed explicitly: ``IcosphereModel.construct`` only
    auto-computes log g when a parameter name is in ``LOG_G_NAMES``, and the
    bundle's name is ``marcs_logg``, which is not (the prefix-stripping
    canonicalisation lives in ``phoebe_model``, not here). Without it SPICE
    would use the constant we passed in rather than the value implied by the
    same mass and radius PHOEBE is using.
    """
    import jax.numpy as jnp
    from spice.models.binary import Binary, add_orbit
    from spice.models.mesh_model import IcosphereModel
    from spice.models.mesh_view import get_mesh_view

    if blackbody:
        from spice.spectrum.blackbody import Blackbody
        emulator = Blackbody()
        # Blackbody's contract is a single parameter, ['Teff'] -- it has no
        # abundance and no log g. (Other scripts pass ['teff', 'abun'] with a
        # two-element array, which construct accepts because it only checks that
        # the lengths agree, but the second column is then dead weight.)
        names = list(emulator.parameter_names)
        log_g_index = None

        def parameters(teff):
            return jnp.array([teff])
    else:
        emulator = emu
        names = list(emulator.stellar_parameter_names)
        log_g_index = names.index("marcs_logg")

        def parameters(teff):
            return emulator.to_parameters({
                "marcs_teff": float(teff), "marcs_logg": 4.0,
                "feh": float(feh), "vmicro": float(vmicro)})

    def body(mass, radius, teff):
        return get_mesh_view(
            IcosphereModel.construct(n_mesh, radius, mass, parameters(teff),
                                     names, log_g_index=log_g_index),
            jnp.array([0.0, 0.0, -1.0]),
        )

    body1 = body(b.get_parameter("mass@primary@component").value,
                 b.get_parameter("requiv@primary@component").value,
                 b.get_parameter("teff@primary@component").value)
    body2 = body(b.get_parameter("mass@secondary@component").value,
                 b.get_parameter("requiv@secondary@component").value,
                 b.get_parameter("teff@secondary@component").value)
    # ``n_neighbours`` is how many candidate occluder faces the KD-tree search
    # considers per occluded face. Binary.from_bodies picks it automatically as
    # clip(1.5*max(triangle_counts), MIN_N_NEIGHBOURS, 64), but that estimate
    # *falls* as the mesh refines (49 -> 31 -> 24 for these systems at
    # 1280/5120/20480) and pins to the floor exactly when a finer mesh needs
    # more neighbours, not fewer. The search is then starved and occlusion is
    # under-counted -- measured at mid primary eclipse for cool-hot, the
    # occluded area is 0.26% low at N=1280, 0.67% at 5120 and 0.97% at 20480,
    # which makes the eclipse progressively too shallow and defeats a mesh
    # convergence test. Passing an explicit value (>=48 converges here) avoids
    # it without changing library behaviour.
    binary = Binary.from_bodies(body1, body2,
                                n_neighbours1=n_neighbours,
                                n_neighbours2=n_neighbours)
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
        # The eclipse window is a small fraction of the period, and add_orbit
        # interpolates positions off a uniform grid; too coarse a grid
        # effectively extrapolates through ingress and egress.
        orbit_resolution_points=20000,
    )
    return binary, emulator


def spice_light_curve(binary, emulator, times, filters, wavelengths,
                      chunk_size, verbose=True):
    """Differential magnitudes per band, plus the out-of-eclipse light ratio."""
    import jax.numpy as jnp
    from spice.models.binary import evaluate_orbit_at_times
    from spice.spectrum.spectrum import AB_passband_luminosity, simulate_observed_flux

    log_wavelengths = jnp.log10(wavelengths)
    pb1, pb2 = evaluate_orbit_at_times(binary, times * DAYS_TO_YR)

    mags = {k: [] for k in filters}
    per_body = []
    for i, (_pb1, _pb2) in enumerate(zip(pb1, pb2)):
        t0 = time.time()
        # ``wavelengths_chunk_size`` is left at its default, which since the
        # 2026-09 change means "one chunk, no padding" for any grid up to 1024
        # points. ``chunk_size`` is still passed explicitly so the CLI can tune
        # it: it sets how wide the emulator graph is vmapped and inlined, which
        # is compile time rather than runtime, and the sweet spot depends on the
        # bundle. Results are invariant to both (machine precision).
        spec1 = simulate_observed_flux(emulator.intensity, _pb1, log_wavelengths,
                                       chunk_size=chunk_size,
                                       disable_doppler_shift=True)
        spec2 = simulate_observed_flux(emulator.intensity, _pb2, log_wavelengths,
                                       chunk_size=chunk_size,
                                       disable_doppler_shift=True)
        total = spec1[:, 0] + spec2[:, 0]
        for key, filt in filters.items():
            mags[key].append(float(AB_passband_luminosity(filt, wavelengths, total)))
        per_body.append((np.asarray(spec1[:, 0]), np.asarray(spec2[:, 0])))
        if verbose:
            print(f"      epoch {i + 1:3d}/{len(times)}  {time.time() - t0:6.1f} s",
                  flush=True)
    return {k: np.array(v) for k, v in mags.items()}, per_body


def light_ratio(per_body, index, filters, wavelengths):
    """``L2/L1`` in each band at the given (out-of-eclipse) epoch."""
    import jax.numpy as jnp
    spec1, spec2 = per_body[index]
    out = {}
    for key, filt in filters.items():
        r = np.asarray(filt.filter_responses_for_wavelengths(jnp.asarray(wavelengths)))
        out[key] = float(np.trapezoid(spec2 * r, wavelengths)
                         / np.trapezoid(spec1 * r, wavelengths))
    return out


def run_binary(args):
    """``--mode binary`` entry point, called from spice_vs_phoebe_atmospheres."""
    import jax.numpy as jnp
    from spice_vs_phoebe_atmospheres import (
        EMULATOR_RANGE_A, load_emulator, load_passbands,
    )

    systems = args.systems or ["cool-hot"]
    suite = args.suite
    spice_kind = "blackbody" if suite == "blackbody" else "aemu"
    bands = load_passbands(BANDS, atms=args.atms)
    filters = {k: v["filter"] for k, v in bands.items()}
    lo = max(EMULATOR_RANGE_A[0], min(b["wl"].min() for b in bands.values()))
    hi = min(EMULATOR_RANGE_A[1], max(b["wl"].max() for b in bands.values()))
    wavelengths = jnp.asarray(np.linspace(lo, hi, args.n_wavelengths))

    emu = load_emulator(args.bundle) if spice_kind == "aemu" else None

    results = {}
    for name in systems:
        system = SYSTEMS[name]
        print(f"\n=== {name}: {system} ===", flush=True)
        b0 = build_bundle(system, args.n_mesh, args.distortion)
        realized = realized_triangles(system, args.n_mesh, args.distortion)
        print(f"  mesh: SPICE {args.n_mesh} exact, PHOEBE {args.n_mesh} requested "
              f"-> {realized['primary']}/{realized['secondary']} realized",
              flush=True)
        times, edges, ref = sample_times(b0, args.n_per_eclipse)
        print(f"  {len(times)} epochs, baseline at index {ref}", flush=True)

        phoebe_dmag, phoebe_ratio = {}, {}
        for atm in args.atms:
            b = build_bundle(system, args.n_mesh, args.distortion)
            for band in BANDS:
                b.add_dataset("lc", compute_times=times, passband=band,
                              dataset=_ds(band))
                b.set_value_all("pblum_mode", dataset=_ds(band), value="absolute")
            configure_atmosphere(b, atm)
            t0 = time.time()
            b.run_compute(irrad_method="none", ltte=False)
            print(f"  PHOEBE {atm:<10} {time.time() - t0:6.1f} s", flush=True)
            for band in BANDS:
                echoed = np.asarray(b.get_value(f"times@{_ds(band)}@model"))
                if not np.allclose(echoed, times, rtol=0, atol=1e-9):
                    raise RuntimeError(
                        f"PHOEBE reordered the times for {band}; the light "
                        "curves would be mis-paired with SPICE's")
                flux = np.asarray(b.get_value(f"fluxes@{_ds(band)}@model"))
                phoebe_dmag[(atm, band)] = -2.5 * np.log10(flux / flux[ref])
                pbl = b.compute_pblums(dataset=_ds(band))
                phoebe_ratio[(atm, band)] = (
                    float(pbl[f"pblum@secondary@{_ds(band)}"].value)
                    / float(pbl[f"pblum@primary@{_ds(band)}"].value))

        # Exactly one SPICE run per suite. Running both emulators and
        # cross-comparing them against every PHOEBE atmosphere is what produced
        # the meaningless "MARCS emulator vs blackbody" numbers; each suite now
        # pairs its own SPICE side with its own PHOEBE side and nothing else.
        spice_dmag, spice_ratio = {}, {}
        for label, blackbody in ((spice_kind, spice_kind == "blackbody"),):
            binary, emulator = spice_binary(b0, args.n_mesh, emu, system["feh"],
                                            args.vmicro, blackbody=blackbody,
                                            n_neighbours=getattr(args, "n_neighbours", None))
            print(f"  SPICE  {label}", flush=True)
            t0 = time.time()
            mags, per_body = spice_light_curve(
                binary, emulator, times, filters, wavelengths,
                args.chunk_size, verbose=args.verbose)
            print(f"  SPICE  {label:<10} {time.time() - t0:6.1f} s", flush=True)
            for band in BANDS:
                spice_dmag[(label, band)] = mags[band] - mags[band][ref]
            spice_ratio[label] = light_ratio(per_body, ref, filters, wavelengths)

        results[name] = {
            "system": system, "times": times, "edges": edges,
            "baseline_index": ref, "n_per_eclipse": args.n_per_eclipse,
            "phoebe_dmag": phoebe_dmag, "phoebe_ratio": phoebe_ratio,
            "spice_dmag": spice_dmag, "spice_ratio": spice_ratio,
            "n_mesh": args.n_mesh, "distortion": args.distortion,
            "suite": suite, "spice_kind": spice_kind, "atms": list(args.atms),
            "n_mesh_phoebe": realized,
            "n_neighbours": getattr(args, "n_neighbours", None),
        }
    return results


def _ds(band):
    return "lc_" + band.replace(":", "_").lower()


def report_binary(results, atms):
    """Text report for one suite.

    The SPICE column is whichever emulator this suite ran -- read off the
    results rather than assumed -- so the blackbody control reports its own
    Blackbody run instead of an absent 'aemu' key.
    """
    lines = []
    add = lines.append
    for name, r in results.items():
        n = r["n_per_eclipse"]
        ref = r["baseline_index"]
        assert ref == n, f"unexpected baseline position {ref} (expected {n})"
        prim, sec = slice(0, ref), slice(ref + 1, ref + 1 + n)
        spice_kind = r.get("spice_kind", "aemu")
        spice_col = {"aemu": "SPICE aemu", "blackbody": "SPICE bb"}[spice_kind]
        run_atms = [a for a in (r.get("atms") or atms)]

        add("=" * 92)
        add(f"[{r.get('suite', '?')}] {name}: {r['system']}")
        add(f"{r['n_mesh']} mesh elements, PHOEBE distortion "
            f"'{r['distortion']}', {len(r['times'])} epochs")
        add("=" * 92)
        add("")
        add("Eclipse depths [mag]")
        add(f"  {'band':<14}{'eclipse':<11}"
            + "".join(f"{'PH ' + a:>13}" for a in run_atms)
            + f"{spice_col:>13}")
        for band in BANDS:
            for label, sl in (("primary", prim), ("secondary", sec)):
                row = f"  {band:<14}{label:<11}"
                for a in run_atms:
                    d = r["phoebe_dmag"].get((a, band))
                    row += f"{np.nanmax(d[sl]) if d is not None else np.nan:>13.5f}"
                d = r["spice_dmag"].get((spice_kind, band))
                row += f"{np.nanmax(d[sl]) if d is not None else np.nan:>13.5f}"
                add(row)
        add("")
        add("Out-of-eclipse light ratio L2/L1")
        add(f"  {'band':<14}" + "".join(f"{'PH ' + a:>13}" for a in run_atms)
            + f"{spice_col:>13}")
        for band in BANDS:
            row = f"  {band:<14}"
            for a in run_atms:
                row += f"{r['phoebe_ratio'].get((a, band), np.nan):>13.5f}"
            row += f"{r['spice_ratio'].get(spice_kind, {}).get(band, np.nan):>13.5f}"
            add(row)
        add("")
        add(f"Depth residual {spice_col} - PHOEBE [mmag]  (rms over the window)")
        add(f"  {'band':<14}{'eclipse':<11}" + "".join(f"{a:>13}" for a in run_atms))
        for band in BANDS:
            for label, sl in (("primary", prim), ("secondary", sec)):
                row = f"  {band:<14}{label:<11}"
                sp = r["spice_dmag"].get((spice_kind, band))
                for a in run_atms:
                    ph = r["phoebe_dmag"].get((a, band))
                    if sp is None or ph is None:
                        row += f"{np.nan:>13.3f}"
                    else:
                        d = (sp[sl] - ph[sl]) * 1e3
                        row += f"{np.sqrt(np.nanmean(d ** 2)):>13.3f}"
                add(row)
        add("=" * 92)
    return "\n".join(lines)
