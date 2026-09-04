"""SPICE's aemu emulator against PHOEBE's model atmospheres, as a function of
stellar parameters.

``tzfor_phoebe_vs_spice.py`` pins *both* codes to a blackbody so that only the
geometry can differ. This script does the opposite: it holds the geometry fixed
(or removes it entirely) and varies the radiative transfer, so the residual is
attributable to the atmospheres.

Three modes, cheapest first. Each one explains the next:

``--mode intensity`` (no orbit, seconds per grid point)
    Passband-integrated specific intensity ``I(mu)`` for a single star, from
    SPICE's ``RozanskiT/TPayne-spice-harps`` intensity bundle (MARCS) and from
    PHOEBE's own tables (``blackbody`` / ``ck2004`` / ``phoenix``) via
    ``Passband.Imu``, over a grid in Teff, log g and [M/H]. Reports

    * the **response** ``log10 I(P) / I(P_ref)`` -- how each code's normal
      intensity moves with the parameters. Absolute normalisation cancels, so
      this is directly comparable even though the two codes carry different
      units. It is also the quantity that sets an eclipsing binary's light
      ratio, hence its eclipse depths.
    * the **limb-darkening profile** ``I(mu)/I(1)`` and its integral
      ``ldint = int_0^1 mu L(mu) dmu``, which converts intensity to flux and
      therefore scales eclipse depth directly.

``--mode pairs`` (post-processing of the intensity grid)
    Turns the response map into the observable: for a list of (primary,
    secondary) parameter pairs, the light ratio ``L2/L1`` each code predicts per
    passband, and the disagreement between them. A light-ratio error shows up as
    eclipse-depth errors of *opposite sign* in the two eclipses, which is what
    distinguishes it from a geometry error.

``--mode binary`` (full light curves, minutes per configuration)
    The end-to-end check at a handful of representative systems: SPICE's own
    icosphere mesh + emulator against PHOEBE's mesh + each atmosphere, through
    both eclipses. Confirms that the depth differences predicted by the
    intensity grid are the ones that actually appear.

Passbands come from PHOEBE's own transmission tables for *both* codes (see
``tzfor_phoebe_vs_spice.phoebe_filter``), so the passband definition is never a
free variable. Only bands that fall inside the emulator's 3781-6910 A range are
used:

    Johnson:B, Johnson:V   blackbody + ck2004 + phoenix
    Stromgren:b, Stromgren:y   blackbody + ck2004

Requires ``jax_enable_x64`` (set here) and both optional extras:
``pip install "stellar-spice[phoebe,aemu]"``.
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

BUNDLE = "RozanskiT/TPayne-spice-harps"

# The emulator's wavelength support (log10 A domain of its reference scaling).
EMULATOR_RANGE_A = (3781.0, 6910.0)

# Passbands entirely or almost entirely inside EMULATOR_RANGE_A. The fraction
# of each transmission curve that falls outside is checked at load time and
# reported, since anything the emulator cannot reach is silently dropped from
# the SPICE integral but *is* present in PHOEBE's tabulated value.
PASSBANDS = ["Johnson:B", "Johnson:V", "Stromgren:b", "Stromgren:y"]

# Parameter grid axes. The intersection of the three sources:
#   aemu TPayne-spice-harps  teff 3000-8000, logg -0.5-5.5, feh -5.0-1.0
#   PHOEBE ck2004            Teff 3500-50000, logg 0.0-5.0, abun -2.5-0.5
#   PHOEBE phoenix           Teff 2300-12000, logg 0.0-6.0, abun -4.0-1.0
DEFAULT_TEFFS = [4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0]
DEFAULT_LOGGS = [1.5, 2.5, 3.5, 4.5]
DEFAULT_FEHS = [-2.0, -1.0, -0.5, 0.0, 0.5]

# ck2004 is computed at a fixed 2 km/s microturbulence; match it on the SPICE
# side rather than leaving the bundle at its own default.
DEFAULT_VMICRO = 2.0

# Reference point the response is measured against. Chosen inside every table.
REFERENCE_PARAMS = (5500.0, 4.5, 0.0)

_PHOEBE_HINT = (
    "PHOEBE is required for this script but is not installed. Install via the "
    'optional extra declared in pyproject.toml: `pip install "stellar-spice[phoebe]"`.'
)


def _import_phoebe():
    try:
        import phoebe
    except ImportError as exc:
        raise ValueError(_PHOEBE_HINT) from exc
    return phoebe


# --------------------------------------------------------------------------
# Passbands
# --------------------------------------------------------------------------

def load_passbands(names=PASSBANDS):
    """PHOEBE ``Passband`` objects plus the SPICE ``Filter`` built from the same
    transmission table.

    Both codes must integrate over an identical curve or the comparison charges
    a passband-definition difference to the atmospheres. Two traps this avoids
    are documented in ``tzfor_phoebe_vs_spice.phoebe_filter``: PHOEBE stores
    ``ptf_table['wl']`` in **metres**, and its tables come out of FITS
    big-endian (``>f8``), which JAX rejects.
    """
    _import_phoebe()
    import jax.numpy as jnp
    from phoebe.atmospheres import passbands as pbs
    from spice.spectrum.filter import Filter

    out = {}
    for name in names:
        pb = pbs.get_passband(name)
        wl = np.asarray(pb.ptf_table["wl"]).astype(np.float64) * 1e10  # m -> A
        fl = np.asarray(pb.ptf_table["fl"]).astype(np.float64)
        inside = (wl >= EMULATOR_RANGE_A[0]) & (wl <= EMULATOR_RANGE_A[1])
        covered = float(np.trapezoid(fl[inside], wl[inside]) / np.trapezoid(fl, wl))
        atms = sorted({c.split(":")[0] for c in pb.content})
        out[name] = {
            "passband": pb,
            "filter": Filter(jnp.array([wl, fl]), name=f"PHOEBE {name}",
                             non_photonic=True),
            "wl": wl,
            "fl": fl,
            "covered_fraction": covered,
            "atms": [a for a in ("blackbody", "ck2004", "phoenix") if a in atms],
        }
    return out


# --------------------------------------------------------------------------
# SPICE side
# --------------------------------------------------------------------------

def load_emulator(bundle=BUNDLE):
    from spice.spectrum.aemu_spectrum_emulator import (
        IntensityPretrainedAemuSpectrumEmulator,
    )
    return IntensityPretrainedAemuSpectrumEmulator(bundle)


def _spice_parameters(emu, teff, logg, feh, vmicro):
    """Bundle parameter vector for one star.

    The Aug-2026 retrain renamed everything (``teff`` -> ``marcs_teff``), and
    ``to_parameters`` fills *unknown* names with 0.0 rather than raising -- so a
    dict keyed on the old names yields a flat, meaningless spectrum. Assert the
    names we set are actually in the contract.
    """
    values = {"marcs_teff": float(teff), "marcs_logg": float(logg),
              "feh": float(feh), "vmicro": float(vmicro)}
    missing = [k for k in values if k not in emu.stellar_parameter_names]
    if missing:
        raise ValueError(
            f"bundle {BUNDLE} does not expose {missing}; its parameters are "
            f"{emu.stellar_parameter_names}. Fetch the current revision."
        )
    return emu.to_parameters(values)


def spice_intensity_grid(emu, log_wavelengths, params, mus):
    """``(n_mu, n_wavelength)`` specific intensity from the emulator.

    ``mu`` is an input channel of this bundle, so the limb darkening comes out
    of MARCS itself rather than from an assumed law. Channel 0 is the line
    spectrum (channel 1 is the continuum).
    """
    import jax
    import jax.numpy as jnp

    per_mu = jax.vmap(emu.intensity, in_axes=(None, 0, None))(
        log_wavelengths, jnp.asarray(mus), params
    )
    return np.asarray(per_mu[..., 0])


def passband_integrate(intensities, wavelengths, response):
    """Energy-weighted passband integral ``int I(lambda) T(lambda) dlambda``.

    Energy weighting (not photon counting) matches PHOEBE's
    ``photon_weighted=False`` and the ``non_photonic=True`` branch of
    ``AB_passband_luminosity``; the photon branch adds a factor of lambda to the
    integrand and would bias the two codes against each other by a
    colour-dependent amount.
    """
    return np.trapezoid(intensities * response[None, :], wavelengths, axis=-1)


# --------------------------------------------------------------------------
# PHOEBE side
# --------------------------------------------------------------------------

def phoebe_intensity(pb, atm, teff, logg, abun, mus):
    """``I(mu)`` over ``mus`` for one atmosphere, or NaN outside the table.

    ``ld_func='interp'`` reads the tabulated ``I(mu)`` directly instead of
    fitting a parametric law to it, which is the closest analogue to the
    emulator's native mu channel.

    Two PHOEBE API traps:

    * ``Imu`` assigns into its arguments, so scalars raise ``TypeError:
      'float' object does not support item assignment``. Every parameter has to
      be passed as an array of the same length as ``mu``.
    * ``atm='blackbody'`` has no tabulated ``I(mu)`` at all, so it rejects
      ``ld_func='interp'`` outright. PHOEBE's blackbody is not limb darkened,
      so the profile is flat by construction and only ``Inorm`` is needed --
      which is exactly what SPICE's own ``Blackbody`` emulator does too. That
      flatness is itself one of the results.
    """
    mus = np.asarray(mus, dtype=float)
    n = len(mus)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if atm == "blackbody":
                i_norm = float(np.asarray(pb.Inorm(
                    Teff=np.array([float(teff)]), logg=np.array([float(logg)]),
                    abun=np.array([float(abun)]), atm="blackbody",
                    ld_func="linear", ld_coeffs=[0.0],
                    photon_weighted=False)).ravel()[0])
                return np.full(n, i_norm)
            return np.asarray(pb.Imu(
                Teff=np.full(n, float(teff)), logg=np.full(n, float(logg)),
                abun=np.full(n, float(abun)), mu=mus, atm=atm, ldatm=atm,
                ld_func="interp", photon_weighted=False), dtype=float).ravel()
    except Exception:
        # Off the table edge. NaN rather than aborting: the grid deliberately
        # spans the union of three different table footprints.
        return np.full(n, np.nan)


def phoebe_ldint(pb, atm, teff, logg, abun):
    """PHOEBE's tabulated ``ldint``, or NaN where it is not defined.

    ``blackbody`` has no limb darkening in PHOEBE, so its ``ldint`` is that of a
    uniform disc (1.0 in this convention) by definition rather than by table
    lookup.
    """
    if atm == "blackbody":
        return 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.asarray(pb.ldint(
                Teff=np.array([float(teff)]), logg=np.array([float(logg)]),
                abun=np.array([float(abun)]), ldatm=atm, ld_func="interp",
                photon_weighted=False)).ravel()[0])
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# Limb darkening summary
# --------------------------------------------------------------------------

def ld_integral(mus, profile, n_quad=64):
    """``ldint = 2 int_0^1 mu L(mu) dmu`` for a normalised profile ``L``.

    The factor 2 is PHOEBE's convention, verified against ``Passband.ldint``:
    for Johnson:V / ck2004 at 5500 K, log g 4.5, [M/H] 0 the tabulated profile
    integrates to 0.38750 here and PHOEBE reports 0.77514. Normalising this way
    puts a uniform disc at exactly 1.0, so the number reads as "flux relative to
    an unlimb-darkened star of the same normal intensity".

    ``ldint`` converts normal intensity to flux. An eclipse depth is
    (blocked area) x (that star's share of the light), and ``ldint`` sets the
    second factor, so a disagreement here shows up directly in the depths.
    """
    finite = np.isfinite(profile)
    if finite.sum() < 3:
        return np.nan
    roots, weights = np.polynomial.legendre.leggauss(n_quad)
    roots = (roots + 1) / 2
    weights = weights / 2
    interp = np.interp(roots, mus[finite], profile[finite])
    return float(2.0 * np.sum(weights * roots * interp))


def fit_linear_ld(mus, profile):
    """Least-squares linear coefficient ``u`` in ``L(mu) = 1 - u(1 - mu)``.

    Reported as a single interpretable number next to the full profile, so the
    parameter dependence can be read off a table rather than a stack of curves.
    """
    finite = np.isfinite(profile)
    if finite.sum() < 3:
        return np.nan
    x = 1.0 - mus[finite]
    y = 1.0 - profile[finite]
    return float(np.dot(x, y) / np.dot(x, x))


# --------------------------------------------------------------------------
# Mode: intensity
# --------------------------------------------------------------------------

def run_intensity(args, extra_points=()):
    """The Teff x log g x [M/H] cartesian grid, plus any ``extra_points``.

    ``pairs`` mode needs the exact parameters of the stars it is asked about,
    and those are real stellar values that will not sit on a round cartesian
    grid. Appending them keeps the grid itself coarse (it is only there to map
    the trend) without forcing the caller to snap a system to grid nodes.
    """
    import jax.numpy as jnp

    bands = load_passbands()
    emu = load_emulator(args.bundle)

    print("Passbands (emulator covers 3781-6910 A):")
    for name, b in bands.items():
        print(f"  {name:<14} {b['wl'].min():7.1f}-{b['wl'].max():7.1f} A   "
              f"in-range {b['covered_fraction'] * 100:6.2f}%   atms={b['atms']}")
    print()

    mus = np.asarray(args.mus)
    # One wavelength grid for every band: the emulator is wavelength-conditioned
    # so a single forward pass per (parameters, mu) serves all four passbands.
    lo = max(EMULATOR_RANGE_A[0], min(b["wl"].min() for b in bands.values()))
    hi = min(EMULATOR_RANGE_A[1], max(b["wl"].max() for b in bands.values()))
    wavelengths = np.linspace(lo, hi, args.n_wavelengths)
    log_wavelengths = jnp.asarray(np.log10(wavelengths))
    responses = {
        name: np.asarray(
            b["filter"].filter_responses_for_wavelengths(jnp.asarray(wavelengths)))
        for name, b in bands.items()
    }

    grid = list(itertools.product(args.teffs, args.loggs, args.fehs))
    for point in extra_points:
        if tuple(point) not in grid:
            grid.append(tuple(float(x) for x in point))
    print(f"{len(grid)} grid points ({len(extra_points)} requested explicitly) "
          f"x {len(mus)} mu x {args.n_wavelengths} wavelengths\n")

    records = []
    t_start = time.time()
    for n, (teff, logg, feh) in enumerate(grid):
        t0 = time.time()
        params = _spice_parameters(emu, teff, logg, feh, args.vmicro)
        spice_i = spice_intensity_grid(emu, log_wavelengths, params, mus)

        for band, b in bands.items():
            spice_pb = passband_integrate(spice_i, wavelengths, responses[band])
            # Normal intensity is the mu = 1 end of the profile.
            i_norm_spice = float(spice_pb[-1])
            profile_spice = spice_pb / i_norm_spice
            rec = {
                "teff": teff, "logg": logg, "feh": feh, "band": band,
                "mus": mus,
                "spice_Inorm": i_norm_spice,
                "spice_profile": profile_spice,
                "spice_ldint": ld_integral(mus, profile_spice),
                "spice_ld_linear": fit_linear_ld(mus, profile_spice),
            }
            for atm in b["atms"]:
                ph = phoebe_intensity(b["passband"], atm, teff, logg, feh, mus)
                i_norm_ph = float(ph[-1])
                profile_ph = ph / i_norm_ph
                rec[f"phoebe_{atm}_Inorm"] = i_norm_ph
                rec[f"phoebe_{atm}_profile"] = profile_ph
                rec[f"phoebe_{atm}_ldint"] = ld_integral(mus, profile_ph)
                rec[f"phoebe_{atm}_ld_linear"] = fit_linear_ld(mus, profile_ph)
                # PHOEBE's own ldint on the same profile, as a check that
                # ld_integral reproduces its convention (and that the mu grid is
                # dense enough). Not used in any comparison, only reported.
                rec[f"phoebe_{atm}_ldint_native"] = phoebe_ldint(
                    b["passband"], atm, teff, logg, feh)
            records.append(rec)

        if args.verbose:
            print(f"  [{n + 1:3d}/{len(grid)}] teff={teff:6.0f} logg={logg:4.2f} "
                  f"feh={feh:+5.2f}  {time.time() - t0:5.2f} s", flush=True)
    print(f"\nintensity grid done in {time.time() - t_start:.1f} s")

    result = {
        "records": records,
        "mus": mus,
        "wavelengths": wavelengths,
        "bands": {k: {kk: vv for kk, vv in v.items()
                      if kk in ("covered_fraction", "atms")}
                  for k, v in bands.items()},
        "axes": {"teffs": args.teffs, "loggs": args.loggs, "fehs": args.fehs},
        "vmicro": args.vmicro,
        "bundle": args.bundle,
        "reference": REFERENCE_PARAMS,
    }
    return result


def _response_map(records, band, atm, reference):
    """``log10 I(P) / I(P_ref)`` for both codes on one (band, atm).

    Differencing against a common reference removes the absolute normalisation,
    which the two codes do not share (PHOEBE's ``Inorm`` is a passband-integrated
    W/m^3, SPICE's is erg/s/cm^2/A integrated over the same curve). What survives
    is the shape of the parameter dependence, which is what the comparison is
    about.
    """
    key = f"phoebe_{atm}_Inorm"
    ref = next((r for r in records
                if r["band"] == band
                and (r["teff"], r["logg"], r["feh"]) == tuple(reference)), None)
    if ref is None or key not in ref:
        return []
    if not np.isfinite(ref[key]) or ref[key] <= 0:
        return []
    rows = []
    for r in records:
        if r["band"] != band or key not in r:
            continue
        if not (np.isfinite(r[key]) and r[key] > 0):
            continue
        rows.append({
            "teff": r["teff"], "logg": r["logg"], "feh": r["feh"],
            "spice": np.log10(r["spice_Inorm"] / ref["spice_Inorm"]),
            "phoebe": np.log10(r[key] / ref[key]),
        })
    return rows


def report_intensity(result):
    records = result["records"]
    reference = result["reference"]
    lines = []
    add = lines.append

    add("=" * 88)
    add("SPICE (aemu MARCS intensity bundle) vs PHOEBE model atmospheres")
    add(f"bundle {result['bundle']}, vmicro = {result['vmicro']} km/s")
    add(f"response measured against Teff/logg/[M/H] = {reference}")
    add("=" * 88)
    add("")

    add("Intensity response  log10 I(P) / I(P_ref)   [dex]")
    add("A perfect match is 0.000 in the last column at every grid point; a")
    add("systematic trend in it is an atmosphere difference that will appear as")
    add("a light-ratio (hence eclipse-depth) error for any pair spanning it.")
    add("")
    for band in result["bands"]:
        for atm in result["bands"][band]["atms"]:
            rows = _response_map(records, band, atm, reference)
            if not rows:
                continue
            d = np.array([r["spice"] - r["phoebe"] for r in rows])
            add(f"  {band} vs {atm}:  n={len(rows):3d}  "
                f"rms={np.sqrt(np.mean(d ** 2)):.4f}  "
                f"max|.|={np.max(np.abs(d)):.4f} dex  "
                f"({(10 ** np.max(np.abs(d)) - 1) * 100:+.1f}% in flux)")
    add("")

    # Per-axis breakdown: hold two axes at the reference value and walk the
    # third, so the dependence is legible rather than averaged away.
    for axis, idx in (("Teff", 0), ("logg", 1), ("[M/H]", 2)):
        add(f"Response difference SPICE - PHOEBE along {axis} "
            f"(other axes at reference) [dex]")
        for band in result["bands"]:
            for atm in result["bands"][band]["atms"]:
                rows = _response_map(records, band, atm, reference)
                keys = ("teff", "logg", "feh")
                sel = [r for r in rows
                       if all(r[k] == reference[i] for i, k in enumerate(keys)
                              if i != idx)]
                if len(sel) < 2:
                    continue
                sel.sort(key=lambda r: r[keys[idx]])
                vals = "  ".join(
                    f"{r[keys[idx]]:g}:{r['spice'] - r['phoebe']:+.3f}" for r in sel)
                add(f"  {band:<14}{atm:<11}{vals}")
        add("")

    add("Limb darkening: ldint = int_0^1 mu L(mu) dmu, and the linear "
        "coefficient u")
    add("ldint converts normal intensity to flux, so a difference here scales")
    add("eclipse depth directly. PHOEBE's blackbody is not limb darkened at all")
    add("(ldint = 1, u = 0) -- the gap to it is what a blackbody run costs.")
    add("")
    add(f"  {'band':<14}{'Teff':>6}{'logg':>6}{'[M/H]':>7}"
        f"{'SPICE':>9}{'ck2004':>9}{'phoenix':>9}"
        f"{'u_SPICE':>9}{'u_ck':>8}{'u_ph':>8}")
    for r in records:
        if r["logg"] != reference[1] or r["feh"] != reference[2]:
            continue
        add(f"  {r['band']:<14}{r['teff']:>6.0f}{r['logg']:>6.2f}{r['feh']:>7.2f}"
            f"{r['spice_ldint']:>9.4f}"
            f"{r.get('phoebe_ck2004_ldint', np.nan):>9.4f}"
            f"{r.get('phoebe_phoenix_ldint', np.nan):>9.4f}"
            f"{r['spice_ld_linear']:>9.4f}"
            f"{r.get('phoebe_ck2004_ld_linear', np.nan):>8.4f}"
            f"{r.get('phoebe_phoenix_ld_linear', np.nan):>8.4f}")
    add("")
    add(f"  {'band':<14}{'Teff':>6}{'logg':>6}{'[M/H]':>7}"
        f"{'SPICE':>9}{'ck2004':>9}{'phoenix':>9}"
        f"{'u_SPICE':>9}{'u_ck':>8}{'u_ph':>8}")
    for r in records:
        if r["teff"] != reference[0] or r["feh"] != reference[2]:
            continue
        add(f"  {r['band']:<14}{r['teff']:>6.0f}{r['logg']:>6.2f}{r['feh']:>7.2f}"
            f"{r['spice_ldint']:>9.4f}"
            f"{r.get('phoebe_ck2004_ldint', np.nan):>9.4f}"
            f"{r.get('phoebe_phoenix_ldint', np.nan):>9.4f}"
            f"{r['spice_ld_linear']:>9.4f}"
            f"{r.get('phoebe_ck2004_ld_linear', np.nan):>8.4f}"
            f"{r.get('phoebe_phoenix_ld_linear', np.nan):>8.4f}")
    add("=" * 88)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Mode: pairs
# --------------------------------------------------------------------------

def report_pairs(result, pairs):
    """Light ratio ``L2/L1`` per code, for pairs drawn from the intensity grid.

    ``L`` here is ``Inorm * ldint`` -- normal intensity times the intensity-to-
    flux conversion -- i.e. the per-unit-area passband luminosity. The ratio of
    two stars' ``L`` is what fixes the two eclipse depths, and unlike ``Inorm``
    itself it is dimensionless, so the codes' different normalisations cancel
    exactly.
    """
    records = {(r["band"], r["teff"], r["logg"], r["feh"]): r
               for r in result["records"]}
    lines = []
    add = lines.append
    add("=" * 88)
    add("Light ratio L2/L1 = (Inorm x ldint)_2 / (Inorm x ldint)_1")
    add("A light-ratio error moves the two eclipse depths in OPPOSITE "
        "directions,")
    add("which is what distinguishes it from a geometry error (same sign in "
        "both).")
    add("=" * 88)

    for p1, p2 in pairs:
        add("")
        add(f"  primary   Teff={p1[0]:.0f} logg={p1[1]:.2f} [M/H]={p1[2]:+.2f}")
        add(f"  secondary Teff={p2[0]:.0f} logg={p2[1]:.2f} [M/H]={p2[2]:+.2f}")
        add(f"  {'band':<14}{'atm':<11}{'SPICE':>10}{'PHOEBE':>10}"
            f"{'rel diff':>11}")
        for band in result["bands"]:
            r1 = records.get((band, *p1))
            r2 = records.get((band, *p2))
            if r1 is None or r2 is None:
                add(f"  {band:<14}(not in grid)")
                continue
            spice_ratio = ((r2["spice_Inorm"] * r2["spice_ldint"])
                           / (r1["spice_Inorm"] * r1["spice_ldint"]))
            for atm in result["bands"][band]["atms"]:
                k_i, k_l = f"phoebe_{atm}_Inorm", f"phoebe_{atm}_ldint"
                if k_i not in r1 or k_i not in r2:
                    continue
                # PHOEBE's blackbody is not limb darkened, so its ldint is the
                # uniform-disc 0.5 and cancels in the ratio; keep the same
                # formula for every atmosphere rather than special-casing it.
                num = r2[k_i] * r2[k_l]
                den = r1[k_i] * r1[k_l]
                if not (np.isfinite(num) and np.isfinite(den) and den != 0):
                    continue
                ph_ratio = num / den
                add(f"  {band:<14}{atm:<11}{spice_ratio:>10.5f}"
                    f"{ph_ratio:>10.5f}"
                    f"{(spice_ratio - ph_ratio) / ph_ratio * 100:>+10.2f}%")
    add("=" * 88)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_triplet(text):
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected Teff,logg,feh (three numbers), got {text!r}")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="intensity",
                        choices=["intensity", "pairs", "binary"])
    parser.add_argument("--bundle", default=BUNDLE)
    parser.add_argument("--teffs", type=float, nargs="+", default=DEFAULT_TEFFS)
    parser.add_argument("--loggs", type=float, nargs="+", default=DEFAULT_LOGGS)
    parser.add_argument("--fehs", type=float, nargs="+", default=DEFAULT_FEHS)
    parser.add_argument("--vmicro", type=float, default=DEFAULT_VMICRO,
                        help="microturbulence for the emulator; ck2004 is "
                             "tabulated at 2 km/s")
    parser.add_argument("--mus", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                 0.9, 0.95, 1.0],
                        help="must end at 1.0 (the normal direction)")
    parser.add_argument("--n-wavelengths", type=int, default=400)
    parser.add_argument("--pair", type=_parse_triplet, nargs="+", default=None,
                        help="pairs mode: an even number of Teff,logg,feh "
                             "triplets, consumed two at a time")
    parser.add_argument("--load", type=Path, default=None,
                        help="pairs mode: reuse a pickle from an earlier "
                             "intensity run instead of recomputing")

    binary = parser.add_argument_group("binary mode")
    binary.add_argument("--systems", nargs="+", default=None,
                        help="entries of binary_atmosphere_comparison.SYSTEMS "
                             "(default: cool-hot)")
    binary.add_argument("--atms", nargs="+",
                        default=["blackbody", "ck2004", "phoenix"],
                        help="PHOEBE atmospheres to run")
    binary.add_argument("--n-mesh", type=int, default=1280,
                        help="elements for both codes (SPICE snaps to an "
                             "icosphere subdivision: 1280 / 5120 / 20480)")
    binary.add_argument("--n-per-eclipse", type=int, default=8)
    binary.add_argument("--distortion", default="sphere",
                        choices=["sphere", "roche", "rotstar"],
                        help="PHOEBE distortion method; 'sphere' matches "
                             "SPICE's icosphere so only the atmospheres differ")
    # Mirrors spice.spectrum.spectrum.DEFAULT_CHUNK_SIZE, inlined rather than
    # imported: importing spice here would pull in jax before main() sets
    # jax_enable_x64, which every other import in this script is deferred past.
    binary.add_argument("--chunk-size", type=int, default=256,
                        help="surface elements per jit chunk in "
                             "simulate_observed_flux. Sets how wide the "
                             "emulator graph is vmapped and inlined, so it "
                             "costs compile time rather than runtime; results "
                             "are invariant to it.")
    binary.add_argument("--include-spice-blackbody", action="store_true",
                        default=True,
                        help="also run SPICE's own Blackbody emulator, giving "
                             "the in-code atmosphere cost to compare against "
                             "PHOEBE's")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "atmosphere_out")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.mus[-1] != 1.0:
        parser.error("--mus must end at 1.0; the profile is normalised there")

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "binary":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from binary_atmosphere_comparison import report_binary, run_binary

        results = run_binary(args)
        with open(args.output / "binary_light_curves.pkl", "wb") as f:
            pickle.dump(results, f)
        text = report_binary(results, args.atms)
        print()
        print(text)
        (args.output / "binary_light_curves.txt").write_text(text + "\n")
        print(f"\nwrote {args.output}")
        return

    if args.mode == "pairs" and (args.pair is None or len(args.pair) % 2):
        parser.error("--pair needs an even number of Teff,logg,feh triplets")

    if args.load is not None:
        with open(args.load, "rb") as f:
            result = pickle.load(f)
    else:
        result = run_intensity(args, extra_points=args.pair or ())
        with open(args.output / "intensity_grid.pkl", "wb") as f:
            pickle.dump(result, f)

    text = report_intensity(result)
    print(text)
    (args.output / "intensity_grid.txt").write_text(text + "\n")

    if args.mode == "pairs":
        pairs = list(zip(args.pair[0::2], args.pair[1::2]))
        text2 = report_pairs(result, pairs)
        print()
        print(text2)
        (args.output / "light_ratios.txt").write_text(text2 + "\n")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
