#!/usr/bin/env python
"""Time ``simulate_observed_flux`` for lazy vs eager grid loading.

Writes a synthetic zarr grid at each requested ``(N_nodes, N_lambda)`` shape,
builds ``FluxLazyZarrInterpolator`` in ``in_memory=False`` (lazy) and
``in_memory=True`` (eager) modes, and times one ``simulate_observed_flux`` call
per mode. With ``--with-linear`` it also times the in-memory multilinear
``UserSpectrumInterpolator`` on the same grid (same linear limb darkening,
coefficient 0.6), written to ``grid_synthesis_time_sweep_linear.csv``.
Reports mean +/- std (s) averaged over x32 and x64.
"""
import argparse
import pickle
import csv
import gc
import os
import shutil
import sys
import tempfile
import time
import warnings
from collections import defaultdict

import numpy as np

if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(os.path.join(_SRC, "spice")):
    sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import jax
import jax.numpy as jnp

from spice.spectrum.lazy_zarr_interpolator import FluxLazyZarrInterpolator
from spice.spectrum.user_spectrum_interpolator import UserSpectrumInterpolator
from spice.models import IcosphereModel
from spice.spectrum.spectrum import simulate_observed_flux

from benchmark_grid_loading import write_synthetic_grid, N_LOGG

SHAPES = [(64, 8000), (256, 8000), (1024, 2000), (1024, 8000),
          (1024, 32000), (4096, 8000)]
N_MESH_VERTICES = 300
DISTANCE_PC = 10.0
CHUNK_SIZE = 64
WL_CHUNK_SIZE = 1024
N_WARMUP = 1
N_RUNS = 3


def time_synthesis(interp, mesh, log_w):
    def call():
        out = simulate_observed_flux(
            interp.intensity, mesh, log_w,
            distance=DISTANCE_PC,
            chunk_size=CHUNK_SIZE,
            wavelengths_chunk_size=WL_CHUNK_SIZE,
        )
        out.block_until_ready()
        return out

    t0 = time.perf_counter()
    call()
    compile_s = time.perf_counter() - t0
    for _ in range(max(0, N_WARMUP - 1)):
        call()
    runs = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        call()
        runs.append(time.perf_counter() - t0)
    return float(np.mean(runs)), float(np.std(runs)), float(compile_s)


def write_user_grid(zarr_store, pkl_path):
    """Copy a synthetic zarr grid into the pickle layout of UserSpectrumInterpolator."""
    import zarr
    g = zarr.open(zarr_store, mode="r")
    log_wl = np.log10(np.asarray(g["wavelength"][:], dtype=np.float64))
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "parameters": np.asarray(g["params"][:]),
            "intensities": np.asarray(g["flux"][:]),
            "log10_wavelengths": log_wl,
            "continuum_intensities": np.asarray(g["continuum"][:]),
            "continuum_wavelengths": log_wl,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


def measure_point(n_nodes, n_wave, tmp_root, sink, precision, with_linear=False):
    n_teff = max(2, round(n_nodes / N_LOGG))
    axis_lengths = [n_teff, N_LOGG]
    store = os.path.join(tmp_root, f"synth_{n_teff}x{N_LOGG}_{n_wave}.zarr")
    actual_n_nodes, param_names, _ = write_synthetic_grid(store, axis_lengths, n_wave)
    try:
        params = jnp.asarray([5500.0, 4.25], dtype=jnp.float64 if precision == "x64" else jnp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = IcosphereModel.construct(
                N_MESH_VERTICES, 1.0, 1.0, params, list(param_names),
                override_log_g=False,
            )
        log_w = jnp.log10(jnp.linspace(4500.0, 7000.0, n_wave))
        modes = ["lazy", "eager"] + (["linear"] if with_linear else [])
        for mode in modes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if mode == "linear":
                    pkl = os.path.join(tmp_root, "user_grid.pkl")
                    write_user_grid(store, pkl)
                    interp = UserSpectrumInterpolator(
                        pkl, parameter_names=list(param_names), has_continuum=True,
                        has_mu=False, limb_darkening_law="linear",
                    )
                    os.remove(pkl)
                else:
                    interp = FluxLazyZarrInterpolator(
                        store, params=list(param_names), in_memory=(mode == "eager"),
                    )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_s, std_s, compile_s = time_synthesis(interp, mesh, log_w)
            print(f"  [{precision}] ({actual_n_nodes:>4d}, {n_wave:>5d}) {mode:<5s}: "
                  f"{mean_s*1e3:8.1f} +/- {std_s*1e3:6.1f} ms "
                  f"(compile {compile_s:.2f} s)")
            sink.append({
                "precision": precision,
                "n_nodes": actual_n_nodes, "n_wavelengths": n_wave,
                "variant": mode,
                "mean_s": mean_s, "std_s": std_s, "compile_s": compile_s,
            })
            del interp
            gc.collect()
    finally:
        shutil.rmtree(store, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-linear", action="store_true",
                    help="also time the in-memory multilinear UserSpectrumInterpolator")
    args = ap.parse_args()
    rows = []
    tmp_root = tempfile.mkdtemp(prefix="spice_synth_time_")
    try:
        for precision in ("x32", "x64"):
            jax.config.update("jax_enable_x64", precision == "x64")
            print(f"\n########## precision: {precision} ##########")
            for n_nodes, n_wave in SHAPES:
                measure_point(n_nodes, n_wave, tmp_root, rows, precision, args.with_linear)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    csv_name = "grid_synthesis_time_sweep_linear.csv" if args.with_linear else "grid_synthesis_time_sweep.csv"
    csv_path = os.path.join(_HERE, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "precision", "n_nodes", "n_wavelengths", "variant",
            "mean_s", "std_s", "compile_s",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Print averaged table.
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["n_nodes"], r["n_wavelengths"])][r["variant"]].append(r["mean_s"])
    variants = ["lazy", "eager"] + (["linear"] if args.with_linear else [])
    print(f"\n{'(N_nodes, N_lam)':<22}" + "".join(f"{v + ' (s)':>12}" for v in variants))
    for n_nodes, n_wave in SHAPES:
        cell = agg.get((n_nodes, n_wave), {})
        if not cell:
            continue
        means = [sum(cell[v]) / len(cell[v]) for v in variants]
        print(f"({n_nodes:>5d}, {n_wave:>6d})    " + "".join(f"{m:>12.3f}" for m in means))


if __name__ == "__main__":
    main()
