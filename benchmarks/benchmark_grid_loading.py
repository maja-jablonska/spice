#!/usr/bin/env python
"""Benchmark how long it takes to *load* a linear (zarr) spectral grid.

This isolates ``LazyZarrInterpolator.__init__`` -- i.e. the cost paid once,
up front, when a ``FluxLazyZarrInterpolator`` / ``IntensityLazyZarrInterpolator``
is constructed:

* opening the zarr store,
* reading ``index.parquet`` and building the (sparse) grid index,
* device-putting the static ``wavelength`` array, and
* (when ``in_memory=True``) device-putting the whole ``flux`` / ``continuum``
  row-wise arrays.

It does NOT time spectrum synthesis -- for that see ``benchmark_spice_components.py``.

Three sweeps are run, each timed for both ``in_memory=True`` (the jit-friendly
in-memory accumulation path) and ``in_memory=False`` (lazy: row-wise arrays stay
on the zarr store):

1. ``nodes``       -- scaling with the number of grid nodes (here ``n_Teff * n_logg``),
   wavelength sampling and parameter dimensionality held fixed. Drives the
   grid-index build and, for the in-memory path, the size of the row-wise arrays.
2. ``wavelengths`` -- scaling with the per-spectrum wavelength sampling, node
   count held fixed. The grid index is unchanged; only the row-wise array size
   (in-memory path) and the ``wavelength`` array grow.
3. ``dims``        -- scaling with the number of parameter axes (1-D, 2-D, 3-D,
   ...), with the total node count and wavelength sampling held roughly fixed
   (each axis gets ``round(n_nodes ** (1/d))`` points). Isolates how the
   ``SparseGridIndex`` build (mixed-radix key encoding + ``argsort``) and the
   ``2**d``-corner bookkeeping scale with dimensionality independently of grid size.

For each point we report the total grid size on disk-equivalent terms
(uncompressed ``flux`` + ``continuum`` bytes) alongside the load time, so the
result can also be read as "load time vs. grid size".

Results stream to a CSV as they are taken; unless ``--no-plot`` is given a
3-panel plot is written at the end.

Examples
--------
    python benchmark_grid_loading.py
    python benchmark_grid_loading.py --quick
    python benchmark_grid_loading.py --node-counts 64 256 1024 4096 16384 \\
        --n-wavelengths 8000 --n-runs 5
    python benchmark_grid_loading.py --dims 1 2 3 4 5 --nodes-for-dim-sweep 4096
    python benchmark_grid_loading.py --in-memory true   # only the in-memory path
"""

import argparse
import csv
import gc
import os
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

# SPICE pins JAX to the CPU backend on macOS; mirror benchmark_spice_components.py
# and do it before ``import jax`` so an experimental jax-metal plugin can't grab
# the backend first. On Linux this is left alone (a GPU node picks up CUDA).
if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(os.path.join(_SRC, "spice")):
    sys.path.insert(0, _SRC)

from spice.spectrum.lazy_zarr_interpolator import FluxLazyZarrInterpolator


# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
WAVELENGTH_MIN = 4500.0
WAVELENGTH_MAX = 7000.0

# Parameter-axis names used (in order) for synthetic grids. The first axis is
# always treated as "Teff" for the purpose of filling the spectra; the rest only
# matter as extra grid dimensions. The dims sweep can use up to len(AXIS_SPECS).
AXIS_SPECS = [
    ("teff", 4500.0, 7000.0),
    ("logg", 3.5, 5.0),
    ("feh", -2.0, 0.5),
    ("alpha", -0.4, 0.8),
    ("vmic", 0.5, 4.0),
    ("vsini", 0.0, 30.0),
]

# ``nodes`` sweep: number of grid nodes (= n_Teff * N_LOGG). n_Teff is derived
# as round(node_count / N_LOGG).
N_LOGG = 4
DEFAULT_NODE_COUNTS = (16, 64, 256, 1024, 4096)
QUICK_NODE_COUNTS = (16, 64, 256)
N_WAVELENGTHS_FOR_NODE_SWEEP = 4000

# ``wavelengths`` sweep.
DEFAULT_N_WAVELENGTHS = (1000, 4000, 16000, 64000)
QUICK_N_WAVELENGTHS = (1000, 4000, 16000)
NODES_FOR_WL_SWEEP = 256

# ``dims`` sweep.
DEFAULT_DIMS = (1, 2, 3, 4)
QUICK_DIMS = (1, 2, 3)
NODES_FOR_DIM_SWEEP = 4096
N_WAVELENGTHS_FOR_DIM_SWEEP = 4000

N_TIMING_RUNS = 5
N_WARMUP_RUNS = 1
BYTES_PER_VALUE = 4  # float32 zarr arrays


# --------------------------------------------------------------------------- #
# Synthetic grid                                                               #
# --------------------------------------------------------------------------- #
def _planck_like(wavelengths_angstrom, teff):
    """Cheap Planck-shaped curve, only used to fill the grid with non-trivial
    numbers (the loader never looks at the values)."""
    h = 6.62607015e-27
    c = 2.99792458e10
    k = 1.380649e-16
    w_cm = np.asarray(wavelengths_angstrom) * 1e-8
    return (2 * h * c ** 2 / w_cm ** 5 / (np.exp(h * c / (w_cm * k * teff)) - 1.0)) * 1e-8


def axis_lengths_for_dims(n_dims, n_nodes):
    """Distribute ``~n_nodes`` grid points over ``n_dims`` axes as evenly as
    possible (each axis gets ``round(n_nodes ** (1/d))`` points, with the first
    axis padded to absorb the rounding error so the product stays close to
    ``n_nodes``)."""
    if n_dims > len(AXIS_SPECS):
        raise ValueError(f"dims sweep supports at most {len(AXIS_SPECS)} axes; got {n_dims}")
    per_axis = max(2, round(n_nodes ** (1.0 / n_dims)))
    lengths = [per_axis] * n_dims
    # Nudge the first axis so the product is as close to n_nodes as we can get
    # while keeping every axis length >= 2.
    rest = int(np.prod(lengths[1:])) if n_dims > 1 else 1
    lengths[0] = max(2, round(n_nodes / rest))
    return lengths


def write_synthetic_grid(store_path, axis_lengths, n_wave):
    """Write a ``(prod(axis_lengths), n_wave)`` zarr grid plus its ``index.parquet``.

    ``axis_lengths`` gives the number of distinct points along each parameter
    axis (axes are taken from ``AXIS_SPECS`` in order). Returns
    ``(n_nodes, param_names, grid_bytes)`` where ``grid_bytes`` is the
    uncompressed size of the row-wise arrays (``flux`` + ``continuum``).
    """
    import pandas as pd
    import zarr

    n_dims = len(axis_lengths)
    specs = AXIS_SPECS[:n_dims]
    param_names = [s[0] for s in specs]
    axes = [np.linspace(lo, hi, n) for (_, lo, hi), n in zip(specs, axis_lengths)]
    wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n_wave).astype(np.float32)

    # Cartesian product of the axes -> one row per grid node.
    mesh = np.meshgrid(*axes, indexing="ij")
    cols = [m.reshape(-1) for m in mesh]
    n_rows = cols[0].shape[0]

    # Build spectra from the first axis (treated as Teff); broadcast over the
    # rest so generation stays cheap even for many nodes.
    teff_values = axes[0]
    cont_by_teff = {float(t): _planck_like(wavelengths, t).astype(np.float32)
                    for t in teff_values}
    dips = np.ones_like(wavelengths)
    for centre, width, depth in [(4861.0, 6.0, 0.35), (5183.0, 5.0, 0.28),
                                 (5890.0, 3.5, 0.50), (6563.0, 9.0, 0.40)]:
        dips = dips * (1.0 - depth * np.exp(-0.5 * ((wavelengths - centre) / width) ** 2))

    flux = np.empty((n_rows, n_wave), dtype=np.float32)
    continuum = np.empty((n_rows, n_wave), dtype=np.float32)
    for i in range(n_rows):
        cont = cont_by_teff[float(cols[0][i])].copy()
        # A tiny multiplicative perturbation from the remaining axes so rows
        # are not identical (not that the loader cares).
        for j in range(1, n_dims):
            cont = cont * np.float32(1.0 + 1e-3 * (j + cols[j][i] / (abs(cols[j][i]) + 1.0)))
        continuum[i, :] = cont
        flux[i, :] = cont * dips

    chunk_rows = min(8, n_rows)
    group = zarr.open_group(str(store_path), mode="w")
    group.create_array("wavelength", data=wavelengths)
    group.create_array("flux", data=flux, chunks=(chunk_rows, n_wave))
    group.create_array("continuum", data=continuum, chunks=(chunk_rows, n_wave))
    group.create_array("params", data=np.stack(cols, axis=1).astype(np.float32))

    df_data = {name: col for name, col in zip(param_names, cols)}
    df_data["row_idx"] = np.arange(n_rows, dtype=np.int64)
    pd.DataFrame(df_data).to_parquet(os.path.join(str(store_path), "index.parquet"))

    return n_rows, param_names, flux.nbytes + continuum.nbytes


# --------------------------------------------------------------------------- #
# Timing                                                                       #
# --------------------------------------------------------------------------- #
def time_load(store_path, param_names, in_memory, n_runs, n_warmup):
    """Construct ``FluxLazyZarrInterpolator(store_path, in_memory=...)`` repeatedly.

    Returns ``(mean_s, std_s, first_s)``. ``first_s`` is the wall time of the
    first construction (which, for the very first call of the whole process,
    also pays JAX/zarr/pandas import + a cold OS page cache); the steady-state
    ``mean_s`` is taken over ``n_runs`` later constructions.
    """
    def build():
        interp = FluxLazyZarrInterpolator(store_path, params=list(param_names),
                                          in_memory=in_memory)
        # Touch the device arrays so any lazy device_put is realised.
        for v in interp.rowwise_data.values():
            if hasattr(v, "block_until_ready"):
                v.block_until_ready()
        for v in interp.static_data.values():
            if hasattr(v, "block_until_ready"):
                v.block_until_ready()
        if hasattr(interp.grid_index, "keys"):
            interp.grid_index.keys.block_until_ready()
        return interp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        interp = build()
        first_s = time.perf_counter() - t0
        del interp
        gc.collect()

        for _ in range(max(0, n_warmup - 1)):
            interp = build()
            del interp
            gc.collect()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            interp = build()
            times.append(time.perf_counter() - t0)
            del interp
            gc.collect()

    return float(np.mean(times)), float(np.std(times)), float(first_s)


# --------------------------------------------------------------------------- #
# CSV sink                                                                     #
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["sweep", "in_memory", "n_dims", "axis_lengths", "n_nodes",
              "n_wavelengths", "grid_bytes", "grid_mb",
              "mean_load_s", "std_load_s", "first_load_s"]


class CsvSink:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()

    def add(self, row):
        self.rows.append(row)
        self._writer.writerow(row)
        self._fh.flush()
        return row

    def close(self):
        self._fh.close()
        print(f"\nWrote {len(self.rows)} rows to {self.path}")


# --------------------------------------------------------------------------- #
# Sweeps                                                                       #
# --------------------------------------------------------------------------- #
def _measure(sweep, axis_lengths, n_wave, in_memory_modes, n_runs, n_warmup,
             tmpdir, sink):
    n_dims = len(axis_lengths)
    tag = "x".join(str(n) for n in axis_lengths)
    store_path = os.path.join(tmpdir, f"grid_{sweep}_{tag}_{n_wave}.zarr")
    n_nodes, param_names, grid_bytes = write_synthetic_grid(store_path, axis_lengths, n_wave)
    grid_mb = grid_bytes / 1e6
    print(f"  grid {n_nodes:>7d} nodes ({n_dims}-D: {tag}) x {n_wave:>7d} wl  "
          f"({grid_mb:8.1f} MB on disk-eq):")
    try:
        for in_memory in in_memory_modes:
            mean_s, std_s, first_s = time_load(store_path, param_names, in_memory,
                                               n_runs, n_warmup)
            mode = "in_memory" if in_memory else "lazy     "
            print(f"    {mode}  {mean_s*1e3:9.2f} +/- {std_s*1e3:7.2f} ms"
                  f"   (first {first_s*1e3:9.1f} ms)")
            sink.add({
                "sweep": sweep,
                "in_memory": bool(in_memory),
                "n_dims": n_dims,
                "axis_lengths": tag,
                "n_nodes": n_nodes,
                "n_wavelengths": n_wave,
                "grid_bytes": grid_bytes,
                "grid_mb": grid_mb,
                "mean_load_s": mean_s,
                "std_load_s": std_s,
                "first_load_s": first_s,
            })
    finally:
        shutil.rmtree(store_path, ignore_errors=True)


def sweep_nodes(cfg, sink, tmpdir):
    print(f"\n--- sweep: scaling with #grid nodes "
          f"(2-D, n_logg={N_LOGG}, n_wl={cfg.n_wl_for_node_sweep}) ---")
    for n_nodes in cfg.node_counts:
        n_teff = max(2, round(n_nodes / N_LOGG))
        _measure("nodes", [n_teff, N_LOGG], cfg.n_wl_for_node_sweep,
                 cfg.in_memory_modes, cfg.n_runs, cfg.n_warmup, tmpdir, sink)


def sweep_wavelengths(cfg, sink, tmpdir):
    n_teff = max(2, round(cfg.nodes_for_wl_sweep / N_LOGG))
    print(f"\n--- sweep: scaling with #wavelengths "
          f"(2-D, {n_teff * N_LOGG} nodes) ---")
    for n_wave in cfg.n_wavelengths:
        _measure("wavelengths", [n_teff, N_LOGG], n_wave,
                 cfg.in_memory_modes, cfg.n_runs, cfg.n_warmup, tmpdir, sink)


def sweep_dims(cfg, sink, tmpdir):
    print(f"\n--- sweep: scaling with #parameter axes "
          f"(~{cfg.nodes_for_dim_sweep} nodes, n_wl={cfg.n_wl_for_dim_sweep}) ---")
    for n_dims in cfg.dims:
        axis_lengths = axis_lengths_for_dims(n_dims, cfg.nodes_for_dim_sweep)
        _measure("dims", axis_lengths, cfg.n_wl_for_dim_sweep,
                 cfg.in_memory_modes, cfg.n_runs, cfg.n_warmup, tmpdir, sink)


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def _plot_panel(ax, rows, sweep, in_memory_modes, xkey, xlabel, title, logx=True):
    for im in in_memory_modes:
        pts = sorted((r for r in rows if r["sweep"] == sweep and r["in_memory"] == im),
                     key=lambda r: r[xkey])
        if pts:
            ax.errorbar([r[xkey] for r in pts],
                        [r["mean_load_s"] * 1e3 for r in pts],
                        yerr=[r["std_load_s"] * 1e3 for r in pts],
                        marker="o", capsize=3, label=f"in_memory={im}")
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("grid load time [ms]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def make_plots(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(skipping plot: {exc})")
        return

    sweeps_present = {r["sweep"] for r in rows}
    panels = [s for s in ("nodes", "wavelengths", "dims") if s in sweeps_present]
    if not panels:
        return
    modes = sorted({r["in_memory"] for r in rows})

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5),
                             squeeze=False)
    axes = axes[0]
    for ax, sweep in zip(axes, panels):
        if sweep == "nodes":
            _plot_panel(ax, rows, "nodes", modes, "n_nodes",
                        "number of grid nodes", "Load time vs #nodes")
        elif sweep == "wavelengths":
            _plot_panel(ax, rows, "wavelengths", modes, "grid_mb",
                        "grid size (flux + continuum, uncompressed) [MB]",
                        "Load time vs grid size (#wavelengths sweep)")
        elif sweep == "dims":
            _plot_panel(ax, rows, "dims", modes, "n_dims",
                        "number of parameter axes",
                        "Load time vs #parameter axes", logx=False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _parse_in_memory(value):
    if value in ("true", "1", "yes"):
        return (True,)
    if value in ("false", "0", "no"):
        return (False,)
    if value == "both":
        return (False, True)
    raise argparse.ArgumentTypeError(f"expected true/false/both, got {value!r}")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="smaller sweeps for a fast smoke test")
    p.add_argument("--node-counts", type=int, nargs="+", default=None,
                   help=f"grid-node counts for the 'nodes' sweep (default {DEFAULT_NODE_COUNTS})")
    p.add_argument("--n-wavelengths", type=int, nargs="+", default=None,
                   help=f"wavelength counts for the 'wavelengths' sweep (default {DEFAULT_N_WAVELENGTHS})")
    p.add_argument("--dims", type=int, nargs="+", default=None,
                   help=f"parameter-axis counts for the 'dims' sweep "
                        f"(default {DEFAULT_DIMS}; max {len(AXIS_SPECS)})")
    p.add_argument("--n-wl-for-node-sweep", type=int, default=N_WAVELENGTHS_FOR_NODE_SWEEP,
                   help="fixed wavelength count used during the 'nodes' sweep")
    p.add_argument("--nodes-for-wl-sweep", type=int, default=NODES_FOR_WL_SWEEP,
                   help="fixed node count used during the 'wavelengths' sweep")
    p.add_argument("--nodes-for-dim-sweep", type=int, default=NODES_FOR_DIM_SWEEP,
                   help="approx. fixed node count used during the 'dims' sweep")
    p.add_argument("--n-wl-for-dim-sweep", type=int, default=N_WAVELENGTHS_FOR_DIM_SWEEP,
                   help="fixed wavelength count used during the 'dims' sweep")
    p.add_argument("--in-memory", type=_parse_in_memory, default=(False, True),
                   metavar="{true,false,both}",
                   help="which FluxLazyZarrInterpolator(in_memory=...) modes to time (default both)")
    p.add_argument("--sweeps", nargs="+", choices=("nodes", "wavelengths", "dims"),
                   default=("nodes", "wavelengths", "dims"), help="which sweeps to run")
    p.add_argument("--n-runs", type=int, default=N_TIMING_RUNS,
                   help="number of timed constructions averaged per point")
    p.add_argument("--n-warmup", type=int, default=N_WARMUP_RUNS,
                   help="number of warm-up constructions before timing")
    p.add_argument("--csv", default=os.path.join(_HERE, "grid_loading_benchmark.csv"),
                   help="path to the streamed results CSV")
    p.add_argument("--plot", default=os.path.join(_HERE, "grid_loading_benchmark.png"),
                   help="path to the output plot")
    p.add_argument("--no-plot", action="store_true", help="skip the plot")
    p.add_argument("--keep-grids", action="store_true",
                   help="don't delete the temporary zarr grids (debugging)")
    return p


class _Cfg:
    pass


def main(argv=None):
    args = build_parser().parse_args(argv)

    cfg = _Cfg()
    if args.quick:
        cfg.node_counts = args.node_counts or QUICK_NODE_COUNTS
        cfg.n_wavelengths = args.n_wavelengths or QUICK_N_WAVELENGTHS
        cfg.dims = args.dims or QUICK_DIMS
        cfg.n_runs = min(args.n_runs, 3)
    else:
        cfg.node_counts = args.node_counts or DEFAULT_NODE_COUNTS
        cfg.n_wavelengths = args.n_wavelengths or DEFAULT_N_WAVELENGTHS
        cfg.dims = args.dims or DEFAULT_DIMS
        cfg.n_runs = args.n_runs
    cfg.n_warmup = args.n_warmup
    cfg.n_wl_for_node_sweep = args.n_wl_for_node_sweep
    cfg.nodes_for_wl_sweep = args.nodes_for_wl_sweep
    cfg.nodes_for_dim_sweep = args.nodes_for_dim_sweep
    cfg.n_wl_for_dim_sweep = args.n_wl_for_dim_sweep
    cfg.in_memory_modes = args.in_memory

    print("Grid-loading benchmark")
    print(f"  in_memory modes  : {cfg.in_memory_modes}")
    print(f"  node counts      : {list(cfg.node_counts)}")
    print(f"  wavelength counts: {list(cfg.n_wavelengths)}")
    print(f"  dims             : {list(cfg.dims)}")
    print(f"  n_runs={cfg.n_runs}  n_warmup={cfg.n_warmup}")

    sink = CsvSink(args.csv)
    tmp_root = tempfile.mkdtemp(prefix="spice_grid_load_bench_")
    try:
        if "nodes" in args.sweeps:
            sweep_nodes(cfg, sink, tmp_root)
        if "wavelengths" in args.sweeps:
            sweep_wavelengths(cfg, sink, tmp_root)
        if "dims" in args.sweeps:
            sweep_dims(cfg, sink, tmp_root)
    finally:
        sink.close()
        if not args.keep_grids:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            print(f"(kept temporary grids in {tmp_root})")

    if not args.no_plot and sink.rows:
        make_plots(sink.rows, args.plot)


if __name__ == "__main__":
    main()
