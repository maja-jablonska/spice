"""Compare PHOEBE and SPICE bolometric eclipses.

Single-run (backwards-compatible):
    python check_phoebe_eclipses.py 90.0 5.0 --q 1.0 --ecc 0.0 \
        --primary_mass 1.0 --n_times 10 --output_path lc_eclipse

Grid mode (multiprocessing across a CSV of parameter sets):
    python check_phoebe_eclipses.py --grid-config grid.csv \
        --n-workers 16 --output_path lc_eclipse

CSV columns (header required): inclination,period[,q,ecc,primary_mass,n_times].
Missing optional columns fall back to the corresponding CLI flags.

Each worker builds one PHOEBE default_binary() at startup and reuses it for
every task — only parameter values and dataset times are updated between
runs. Outputs are written one pickle per parameter set; existing files are
skipped, so the script is restartable across SLURM/job-array invocations.
"""
import os
# JAX env vars must be set before jax is imported
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

import multiprocessing as mp
# Spawn is required for JAX safety in worker subprocesses
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

import argparse
import csv
import gc
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from astropy import units as u
from tqdm import tqdm

# Heavy imports deferred to worker init so the parent process stays light
# and so re-imports in spawned workers happen in a clean interpreter.

DAYS_TO_YR = 0.0027378507871321013
DEG_TO_RAD = 0.017453292519943295

# Per-worker singletons: built once in init_worker(), reused across all tasks
_WORKER = {}


def _build_phoebe_bundle():
    """Build a default_binary and apply the constraint flip exactly once."""
    import phoebe
    b = phoebe.default_binary()
    b.flip_constraint('mass@primary', solve_for='sma')
    return b


def _ensure_datasets(b, times):
    """Add (first call) or update (subsequent calls) the mesh/orb/lc datasets.

    The static dataset configuration (ld_*, atm, irrad_method, …) is applied
    only on first add — those values don't change across grid points so we
    don't need to re-set them every iteration.
    """
    from phoebe.parameters.dataset import _mesh_columns
    times = np.asarray(times)
    if 'mesh01' not in b.datasets:
        b.add_dataset('mesh', times=times, columns=_mesh_columns, dataset='mesh01')
        b.add_dataset('orb', compute_times=times, dataset='orb01')
        b.add_dataset('lc', compute_times=times,
                      passband='Bolometric:900-40000', dataset='lc_bolometric')
        b.set_value_all('pblum_mode', dataset='lc_bolometric', value='absolute')
        b.set_value('distance', 10 * u.pc)
        b.set_value_all('ld_mode', 'manual')
        b.set_value_all('ld_func', 'linear')
        b.set_value_all('ld_coeffs', [0.])
        b.set_value_all('ld_mode_bol', 'manual')
        b.set_value_all('ld_func_bol', 'linear')
        b.set_value_all('ld_coeffs_bol', [0.])
        b.set_value_all('atm', 'blackbody')
        b.set_value_all('irrad_method', 'none')
        b.set_value_all('gravb_bol', 0.0)
    else:
        b.set_value_all('times', dataset='mesh01', value=times)
        b.set_value_all('compute_times', dataset='orb01', value=times)
        b.set_value_all('compute_times', dataset='lc_bolometric', value=times)


def _default_icosphere(mass, bb):
    from spice.models.binary import Binary  # noqa: F401 - touch to warm import
    from spice.models.mesh_model import IcosphereModel
    from spice.models.mesh_view import get_mesh_view
    return get_mesh_view(
        IcosphereModel.construct(1000, 1., mass, bb.solar_parameters, bb.parameter_names),
        jnp.array([0., 0., -1.]),
    )


def init_worker():
    """One-time initialization for each multiprocessing worker.

    Builds the PHOEBE bundle, the Blackbody emulator, and the Bolometric
    filter — all of which are reused for every task this worker handles.
    """
    import phoebe
    from spice.spectrum.blackbody import Blackbody
    from spice.spectrum.filter import Bolometric

    phoebe.multiprocessing_set_nprocs(1)
    _WORKER['bundle'] = _build_phoebe_bundle()
    _WORKER['bb'] = Blackbody()
    _WORKER['bol'] = Bolometric()


def run_one(inclination, period, q, ecc, n_times, primary_mass, output_path):
    """Run a single grid point, reusing the worker's PHOEBE bundle."""
    from spice.models.binary import Binary, add_orbit, evaluate_orbit_at_times
    from spice.models.orbit_utils import eclipse_timestamps_kepler
    from spice.spectrum.spectrum import AB_passband_luminosity, simulate_observed_flux

    if 'bundle' not in _WORKER:
        # Allows calling run_one directly without going through Pool
        init_worker()
    b = _WORKER['bundle']
    bb = _WORKER['bb']
    bol = _WORKER['bol']

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_pkl = output_path / (
        f'eclipses_incl_{inclination}_period_{period}_q_{q}_ecc_{ecc}'
        f'_primary_mass_{primary_mass}.pkl'
    )
    if out_pkl.exists():
        print(f"[skip] {out_pkl.name}")
        return str(out_pkl)

    t_start = time.time()

    b.set_value('period@binary@component', period)
    b.set_value('q@binary@component', q)
    b.set_value('ecc@binary@component', ecc)
    b.set_value('mass@primary@component', primary_mass)
    b.set_value_all('incl@binary', inclination)

    _, t1_p, _, _, t4_p, _, t1_s, _, _, t4_s = eclipse_timestamps_kepler(
        b.get_parameter('mass@primary@component').value,
        b.get_parameter('mass@secondary@component').value,
        b.get_parameter('period@binary@component').value * DAYS_TO_YR,
        b.get_parameter('ecc@binary@component').value,
        b.get_parameter('t0_perpass@binary@component').value * DAYS_TO_YR,
        jnp.deg2rad(b.get_parameter('incl@binary@component').value),
        b.get_parameter('per0@binary@component').value * DEG_TO_RAD,
        b.get_parameter('long_an@binary@component').value * DAYS_TO_YR,
        b.get_parameter('requiv@primary@component').value,
        b.get_parameter('requiv@secondary@component').value,
        pad=1.1,
        los_vector=jnp.array([0., 0., -1.]),
    )

    half = int(n_times / 2)
    times = np.concatenate([
        np.linspace(float(t1_p) / DAYS_TO_YR, float(t4_p) / DAYS_TO_YR, half),
        np.linspace(float(t1_s) / DAYS_TO_YR, float(t4_s) / DAYS_TO_YR, half),
    ])
    times_yr = times * DAYS_TO_YR

    _ensure_datasets(b, times)
    b.compute_pblums(pbflux=True, set_value=True)
    b.compute_ld_coeffs(ld_mode='manual', ld_func='linear', ld_coeffs=[0.])
    b.run_compute(irrad_method='none', coordinates='uvw', ltte=False, ntriangles=20000)

    fluxes_phoebe = np.asarray(b.get_parameter('fluxes@lc_bolometric@model').value)

    body1 = _default_icosphere(b.get_parameter('mass@primary@component').value, bb)
    body2 = _default_icosphere(b.get_parameter('mass@secondary@component').value, bb)
    binary = Binary.from_bodies(body1, body2)
    binary = add_orbit(
        binary,
        P=b.get_parameter('period@binary@component').value * DAYS_TO_YR,
        ecc=b.get_parameter('ecc@binary@component').value,
        T=b.get_parameter('t0_perpass@binary@component').value * DAYS_TO_YR,
        i=jnp.deg2rad(b.get_parameter('incl@binary@component').value),
        omega=b.get_parameter('per0@binary@component').value * DEG_TO_RAD,
        Omega=b.get_parameter('long_an@binary@component').value * DAYS_TO_YR,
        vgamma=b.get_parameter('vgamma').value,
        reference_time=b.get_parameter('t0_ref@binary@component').value * DAYS_TO_YR,
        mean_anomaly=b.get_parameter('mean_anom@binary@component').value * DEG_TO_RAD,
        orbit_resolution_points=len(times_yr),
    )

    pb1, pb2 = evaluate_orbit_at_times(binary, times_yr)

    vws = jnp.linspace(900, 40000, 1000)
    log_vws = jnp.log10(vws)
    bol_lum = []
    for _pb1, _pb2 in zip(pb1, pb2):
        spec1 = simulate_observed_flux(bb.intensity, _pb1, log_vws, disable_doppler_shift=True)
        spec2 = simulate_observed_flux(bb.intensity, _pb2, log_vws, disable_doppler_shift=True)
        bol_lum.append(AB_passband_luminosity(bol, vws, spec1[:, 0] + spec2[:, 0]))
    bol_lum = np.array(bol_lum)

    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'phoebe_binary': b,
            'spice_body1': pb1,
            'spice_body2': pb2,
            'fluxes_phoebe': fluxes_phoebe,
            'bol_lum': bol_lum - bol_lum[0],
            'times': times,
            'n_times': n_times,
            'sma': b.get_parameter('sma@binary@component').value,
            'primary_mass': primary_mass,
            'secondary_mass': b.get_parameter('mass@secondary@component').value,
            'inclination': inclination,
            'period': period,
            'q': q,
            'ecc': ecc,
        }, f)

    # Bound JAX cache + Python heap growth across many tasks per worker
    gc.collect()
    jax.clear_caches()

    print(f"[done] {out_pkl.name} in {time.time() - t_start:.0f}s")
    return str(out_pkl)


def _process_task(task):
    try:
        return run_one(**task)
    except Exception as exc:
        print(f"[fail] {task}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def _read_grid_csv(path, defaults):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'inclination': float(r['inclination']),
                'period': float(r['period']),
                'q': float(r.get('q') or defaults['q']),
                'ecc': float(r.get('ecc') or defaults['ecc']),
                'primary_mass': float(r.get('primary_mass') or defaults['primary_mass']),
                'n_times': int(r.get('n_times') or defaults['n_times']),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inclination', type=float, nargs='?', default=None,
                        help='Inclination in degrees (single-run mode).')
    parser.add_argument('period', type=float, nargs='?', default=None,
                        help='Period in days (single-run mode).')
    parser.add_argument('--q', type=float, default=1.0, help='Mass ratio')
    parser.add_argument('--ecc', type=float, default=0.0, help='Eccentricity')
    parser.add_argument('--primary_mass', type=float, default=1.0, help='Primary mass [Msun]')
    parser.add_argument('--n_times', type=int, default=10,
                        help='Time points per run (split evenly between primary/secondary eclipse).')
    parser.add_argument('--output_path', type=str, default='lc_eclipse',
                        help='Output directory for per-run pickles.')
    parser.add_argument('--grid-config', type=Path, default=None,
                        help='CSV grid file. When set, runs all rows in parallel.')
    parser.add_argument('--n-workers', type=int, default=os.cpu_count() or 1,
                        help='Number of parallel workers (grid mode).')
    args = parser.parse_args()

    defaults = {'q': args.q, 'ecc': args.ecc, 'primary_mass': args.primary_mass,
                'n_times': args.n_times}

    if args.grid_config is not None:
        tasks = _read_grid_csv(args.grid_config, defaults)
        for t in tasks:
            t['output_path'] = args.output_path
        n_workers = max(1, args.n_workers)
        print(f"Running {len(tasks)} tasks across {n_workers} worker(s)")
        if n_workers == 1:
            init_worker()
            for t in tqdm(tasks):
                _process_task(t)
        else:
            with mp.Pool(n_workers, initializer=init_worker, maxtasksperchild=None) as pool:
                for _ in tqdm(pool.imap_unordered(_process_task, tasks), total=len(tasks)):
                    pass
        return

    if args.inclination is None or args.period is None:
        parser.error('inclination and period are required unless --grid-config is given')

    init_worker()
    run_one(args.inclination, args.period, args.q, args.ecc,
            args.n_times, args.primary_mass, args.output_path)


if __name__ == '__main__':
    main()
