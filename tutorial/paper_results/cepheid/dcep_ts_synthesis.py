"""Disc-integrate the per-phase Turbospectrum intensities of delta Cep with SPICE.

Input: OUT/<variant>.npz from `dcep_ts_intensities.py` (I and continuum-normalised
N on (phase, mu, wl)). For each phase the mesh follows the SPIPS radius curve
(mean-normalised l=0 template, base radius <R>, so the pulsation velocity is the
SPIPS one), and every surface element takes that phase's TS intensity at its own
mu, Doppler-shifted by its LOS velocity. A rest-frame twin of every snapshot
(pulsation velocities zeroed, same intensities) is synthesised as the per-phase
zero-velocity template, so estimators can be run either absolutely (against the
rest twin) or observationally (against phase 0).

Limb darkening (--ld, repeatable):
  atm         the TS mu-dependence (linear in mu between the 12 TS mu points)
  lin:<u>     I(mu=1, lambda) * (1 - u (1 - mu)) for lines and continuum alike
              (lin:0 = uniform disc, analytical p = 1.5)

Truth stored per phase (km/s, SPICE LOS sign: receding positive):
  vpuls       -dR/dt of the mesh (area-weighted radial velocity over the sphere,
              expansion -> negative), i.e. the pulsation velocity on the LOS axis
  vlos_geom   projected-area-weighted mean LOS pulsation velocity of the visible
              disc (no intensity weighting); vpuls / vlos_geom = 1.5 exactly

Output: OUT/synth_<variant>_<ld>.npz with wl, phase, flux, cont, rest_flux,
rest_cont (nphase, nwl), vpuls, vlos_geom, radius (projected-area), ld, variant.

--emulator BUNDLE replaces the TS intensities by the TPayne emulator evaluated at the SAME per-phase inputs (Teff, log g,
[Fe/H], microturbulence from the TS table; atmosphere [Fe/H] = [Fe/H] as in the interpolated TS atmospheres; abundance
offsets 0 = solar-scaled as in the TS run), on the same mesh: output synth_<variant>_tpayne.npz. Comparing the two isolates
the emulator (plus its training-grid synthesis) from everything else in the model.

Usage: python dcep_ts_synthesis.py --ts-dir DIR --variant geff_xiphi [--ld atm --ld lin:0]
           [--wl 5000 5200] [--n-mesh 5000] [--max-phases N] [--emulator BUNDLE] [--no-rest]
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from astropy.io import fits  # noqa: E402

from spice.cepheid_bundles import build_bundle  # noqa: E402
from spice.models.utils import cast_to_los  # noqa: E402
from spice.spectrum import simulate_observed_flux  # noqa: E402
from cepheid_phase import radius_template_mean_normalised  # noqa: E402

MASS = 4.726   # SPIPS-implied (cepheid_phase.spips_mass); only enters log g, which the TS tables fix


def mu_table_intensity(lw_tab, mus, I, C):
    """intensity_fn for SPICE: linear in mu between table points, linear in log wavelength."""
    mus = jnp.asarray(mus); I = jnp.asarray(I); C = jnp.asarray(C); lw_tab = jnp.asarray(lw_tab)
    n = mus.shape[0]

    def fn(log_wl, mu, parameters):
        m = jnp.clip(jnp.ravel(mu)[0], mus[0], mus[-1])   # mu arrives as shape (1,)
        j = jnp.clip(jnp.searchsorted(mus, m) - 1, 0, n - 2)
        w = (m - mus[j]) / (mus[j + 1] - mus[j])
        Ii = (1 - w) * I[j] + w * I[j + 1]
        Ci = (1 - w) * C[j] + w * C[j + 1]
        return jnp.stack([jnp.interp(log_wl, lw_tab, Ii), jnp.interp(log_wl, lw_tab, Ci)], axis=-1)
    return fn


def linear_ld_intensity(lw_tab, I1, C1, u):
    """intensity_fn: the mu = 1 spectrum scaled by a linear limb-darkening law."""
    I1 = jnp.asarray(I1); C1 = jnp.asarray(C1); lw_tab = jnp.asarray(lw_tab)

    def fn(log_wl, mu, parameters):
        s = 1.0 - u * (1.0 - jnp.clip(jnp.ravel(mu)[0], 0.0, 1.0))
        return jnp.stack([s * jnp.interp(log_wl, lw_tab, I1), s * jnp.interp(log_wl, lw_tab, C1)], axis=-1)
    return fn


def emu_row(emu, P_tab, k):
    """Emulator inputs at TS phase k: the TS table's Teff, log g, [Fe/H], vmicro; marcs_fe_h = [Fe/H] (the TS atmospheres
    are interpolated in [Fe/H]); every abundance offset 0 (solar-scaled, as in the TS run)."""
    names = list(emu.stellar_parameter_names)
    val = {"marcs_teff": P_tab["teff"][k], "teff": P_tab["teff"][k], "marcs_logg": P_tab["logg"][k], "logg": P_tab["logg"][k],
           "feh": P_tab["feh"][k], "marcs_fe_h": P_tab["feh"][k], "vmicro": P_tab["vmicro"][k]}
    return jnp.asarray([float(val.get(n, 0.0)) for n in names])


def emulator_intensity(emu, row):
    """intensity_fn for SPICE: the emulator at one fixed parameter row for every element (mu arrives as shape (1,))."""
    def fn(log_wl, mu, parameters):
        return emu.intensity(log_wl, jnp.ravel(mu)[0], row)
    return fn


def truth_velocities(m):
    centers = np.asarray(m.centers)
    normals = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    radial = np.sum(np.asarray(m.pulsation_velocities) * normals, axis=1)
    vpuls = -float(np.average(radial, weights=np.asarray(m.areas)))
    vlos = np.asarray(-cast_to_los(m.pulsation_velocities, m.los_vector))
    vis = np.asarray(m.visible_cast_areas)
    vlos_geom = float(np.sum(vlos * vis) / np.sum(vis))
    return vpuls, vlos_geom


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-dir", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ld", action="append", default=None)
    ap.add_argument("--wl", type=float, nargs=2, default=(5000.0, 5200.0))
    ap.add_argument("--step", type=float, default=0.01, help="output sampling [A]")
    ap.add_argument("--n-mesh", type=int, default=5000)
    ap.add_argument("--vrot", type=float, default=0.0)
    ap.add_argument("--radius-scale", type=float, default=1.0,
                    help="scale of the SPIPS radius curve (mean radius and pulsation amplitude), e.g. 279.5/274.0 for the "
                         "paper's model at the Trahin et al. (2021) distance")
    ap.add_argument("--max-phases", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--fits", type=Path, default=HERE / "delta_cep.fits")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--emulator", default=None, help="TPayne bundle: synthesise with the emulator at the TS table's inputs")
    ap.add_argument("--no-rest", action="store_true", help="skip the rest-frame twins (only needed for the p-factor work)")
    args = ap.parse_args()
    lds = args.ld or ["atm"]
    if args.emulator:
        lds = ["tpayne"]
    out_dir = args.out or args.ts_dir

    ts = np.load(args.ts_dir / f"{args.variant}.npz")
    ok = np.all(np.isfinite(ts["I"]), axis=(1, 2))
    P_tab = {k: np.asarray(ts[k], float)[ok] for k in ("teff", "logg", "feh", "vmicro")}
    phases = ts["phase"][ok]; I_all = ts["I"][ok].astype(np.float64); N_all = ts["N"][ok].astype(np.float64)
    if args.max_phases:
        phases, I_all, N_all = phases[:args.max_phases], I_all[:args.max_phases], N_all[:args.max_phases]
    C_all = I_all / N_all
    mus = np.asarray(ts["mus"], float); lw_tab = np.log10(np.asarray(ts["wl"], float))
    print(f"{args.variant}: {len(phases)}/{len(ts['phase'])} phases, mu {mus.min()}-{mus.max()} ({len(mus)}), "
          f"table {ts['wl'][0]:.1f}-{ts['wl'][-1]:.1f} A", flush=True)
    assert mus[-1] == 1.0

    with fits.open(args.fits, ignore_missing_simple=True) as h:
        period = float(h[0].header["HIERARCH PARAM PERIOD"])
        model = h["MODEL"].data
    R_sp = np.asarray(model["R"], float); ph_sp = np.asarray(model["PHASE"], float)
    fourier = radius_template_mean_normalised(ph_sp, R_sp)
    R_sp = R_sp * args.radius_scale
    radius = float(R_sp.mean())
    dummy = SimpleNamespace(stellar_parameter_names=["teff", "logg"])
    t = time.time()
    bundle = build_bundle(dummy, np.array([6000.0, 1.9]), fourier_params=fourier, period=period,
                          timeseries=jnp.asarray(phases * period), n_mesh=args.n_mesh, radius=radius,
                          mass=MASS, rotation_velocity=args.vrot, desc="  mesh")
    snaps = bundle.snapshots
    rest = [m._replace(pulsation_velocities=jnp.zeros_like(m.pulsation_velocities)) for m in snaps]
    truth = np.array([truth_velocities(m) for m in snaps])
    rad = np.array([math.sqrt(float(jnp.sum(m.visible_cast_areas)) / math.pi) for m in snaps])
    R_at = np.interp(phases, np.sort(ph_sp), R_sp[np.argsort(ph_sp)], period=1.0)
    p_geom = np.sum(truth[:, 0] * truth[:, 1]) / np.sum(truth[:, 1] ** 2)
    print(f"mesh {args.n_mesh} vertices, {len(snaps)} snapshots in {time.time() - t:.0f}s; projected radius vs SPIPS "
          f"rms {np.sqrt(np.mean((rad - R_at) ** 2)):.3f} R_sun; vpuls {truth[:, 0].min():.1f}..{truth[:, 0].max():.1f} km/s; "
          f"p_geom = vpuls / vlos_geom = {p_geom:.4f}", flush=True)
    assert abs(p_geom - 1.5) < 0.02, "truth sign/weighting convention broken"

    emu = None
    if args.emulator:
        from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
        emu = IntensityPretrainedAemuSpectrumEmulator(args.emulator)
        print(f"emulator {args.emulator}: inputs {emu.stellar_parameter_names}", flush=True)
        missing = {"teff", "logg", "feh", "vmicro"} - set(emu.stellar_parameter_names)
        assert not missing, f"emulator lacks inputs {missing}: emu_row would set them to 0"
    wl = np.arange(args.wl[0], args.wl[1] + args.step / 2, args.step)
    lw = jnp.log10(jnp.asarray(wl))
    for ld in lds:
        t = time.time()
        flux, rest_flux = np.zeros((len(phases), len(wl), 2)), np.zeros((len(phases), len(wl), 2))
        for k in range(len(phases)):
            if ld == "atm":
                fn = mu_table_intensity(lw_tab, mus, I_all[k], C_all[k])
            elif ld == "tpayne":
                fn = emulator_intensity(emu, emu_row(emu, P_tab, k))
            elif ld.startswith("lin:"):
                fn = linear_ld_intensity(lw_tab, I_all[k, -1], C_all[k, -1], float(ld[4:]))
            else:
                raise ValueError(f"unknown --ld {ld!r}")
            flux[k] = np.asarray(simulate_observed_flux(fn, snaps[k], lw, chunk_size=args.chunk_size))
            if not args.no_rest:
                rest_flux[k] = np.asarray(simulate_observed_flux(fn, rest[k], lw, chunk_size=args.chunk_size))
            if k == 0:
                print(f"  [{ld}] first phase {time.time() - t:.0f}s (incl. compile)", flush=True)
        path = out_dir / f"synth_{args.variant}_{ld.replace(':', '')}.npz"
        np.savez(path, wl=wl, phase=phases, flux=flux[..., 0], cont=flux[..., 1],
                 rest_flux=rest_flux[..., 0], rest_cont=rest_flux[..., 1],
                 vpuls=truth[:, 0], vlos_geom=truth[:, 1], radius=rad, ld=ld, variant=args.variant,
                 n_mesh=args.n_mesh, vrot=args.vrot, radius_scale=args.radius_scale)
        print(f"  [{ld}] {len(phases)} phases in {time.time() - t:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
