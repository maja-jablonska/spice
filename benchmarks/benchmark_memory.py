#!/usr/bin/env python
"""Benchmark the *memory* footprint of SPICE intensity sources.

Two things are measured for each source:

* **resident bytes held by the object** -- the sum of ``.nbytes`` over every
  on-device (JAX) array reachable from the constructed emulator/interpolator.
  This is exact and reproducible; it is the headline number.
* **process RSS after construction** -- ``psutil`` resident-set size of the whole
  process once the object is built. Includes the interpreter, JAX, zarr, etc.,
  so it is only useful as a coarse sanity check / absolute floor; the *delta*
  between sources is dominated by allocator slack and JAX's buffer pools and is
  not reported.

With ``--with-synthesis`` an extra column ``synthesis_peak_increase_mb`` is
recorded: the rise in ``ru_maxrss`` (the kernel's high-water mark for the
process) caused by running one ``simulate_observed_flux`` call -- i.e. the
transient working set of synthesis (chunk buffers, the JIT'd executable,
device buffer pools, and -- in the lazy grid path -- the zarr reads), on top of
everything already allocated. It is monotonic, so only the *first* call that hits
a given peak shows it.

Sources covered:

* ``static``       -- a fixed precomputed 1-D spectrum (the ``static`` benchmark
  source); essentially zero.
* ``gaussian``     -- ``GaussianLineEmulator``; a handful of small arrays.
* ``grid``         -- ``FluxLazyZarrInterpolator`` over synthetic zarr grids of
  growing size, for both ``in_memory=False`` (row-wise arrays stay lazy on the
  store) and ``in_memory=True`` (the whole flux/continuum array is device-put).
  The headline number is compared against the closed-form
  ``2 * n_nodes * n_wavelengths * 4 B`` (float32 flux + continuum).
* ``aemu_*``       -- pretrained ``astro_emulators_toolkit`` bundles, if the
  optional extra is installed (``pip install "stellar-spice[aemu]"``); their
  footprint is the network weights. Skipped with a note otherwise.

Results stream to a CSV; unless ``--no-plot`` is given a plot of "resident bytes
vs grid size" is written at the end.

Examples
--------
    python benchmark_memory.py
    python benchmark_memory.py --quick
    python benchmark_memory.py --sources grid --node-counts 256 1024 4096 16384 \\
        --n-wavelengths 8000 --with-synthesis
    python benchmark_memory.py --sources aemu_harps aemu_small_random
    python benchmark_memory.py --precision x32   # x32 only (default: both x32 and x64)
"""

import argparse
import csv
import gc
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(os.path.join(_SRC, "spice")):
    sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)  # so we can import the sibling grid-loading helpers

import jax
import jax.numpy as jnp

from spice.spectrum.lazy_zarr_interpolator import FluxLazyZarrInterpolator
from spice.spectrum.gaussian_line_emulator import GaussianLineEmulator

from benchmark_grid_loading import write_synthetic_grid  # noqa: E402
from _bench_common import _planck_like, _resolve_precisions, _parse_in_memory

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None


# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
WAVELENGTH_MIN, WAVELENGTH_MAX = 4500.0, 7000.0
N_LOGG = 4
DEFAULT_NODE_COUNTS = (64, 256, 1024, 4096)
QUICK_NODE_COUNTS = (64, 256, 1024)
DEFAULT_N_WAVELENGTHS = 8000
DEFAULT_WL_FOR_SWEEP = (2000, 8000, 32000)
QUICK_WL_FOR_SWEEP = (2000, 8000)
NODES_FOR_WL_SWEEP = 1024

AEMU_BUNDLES = {
    "aemu_harps": "RozanskiT/TPayne-spice-harps",
    "aemu_small_random": "/Users/mjablons/code/spice/data/new_fe_bundle",
}

# Small mesh / wavelength grid used only by --with-synthesis.
SYNTH_MESH_VERTICES = 300
SYNTH_N_WAVELENGTHS = 2000
SYNTH_CHUNK = 256
SYNTH_WL_CHUNK = 1024
SYNTH_DISTANCE_PC = 10.0


# --------------------------------------------------------------------------- #
# Memory probes                                                                #
# --------------------------------------------------------------------------- #
def rss_bytes():
    if _PROC is not None:
        return int(_PROC.memory_info().rss)
    return maxrss_bytes()


def maxrss_bytes():
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return int(ru if sys.platform == "darwin" else ru * 1024)


def _array_nbytes(x):
    nb = getattr(x, "nbytes", None)
    if nb is not None:
        return int(nb)
    try:
        return int(np.asarray(x).nbytes)
    except Exception:
        return 0


def _is_device_array(x):
    return isinstance(x, jax.Array)


def resident_device_bytes(obj, _seen=None, _depth=0):
    """Sum ``.nbytes`` over every ``jax.Array`` reachable from ``obj`` (dicts,
    lists/tuples, and ``__dict__`` attributes), de-duplicating by id."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or _depth > 6:
        return 0
    _seen.add(id(obj))

    if _is_device_array(obj):
        return _array_nbytes(obj)
    total = 0
    if isinstance(obj, dict):
        for v in obj.values():
            total += resident_device_bytes(v, _seen, _depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            total += resident_device_bytes(v, _seen, _depth + 1)
    else:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                total += resident_device_bytes(v, _seen, _depth + 1)
    return total


def zarr_handle_bytes(obj):
    """Bytes that *would* be resident if the lazy zarr arrays were materialised
    (i.e. the row-wise flux/continuum still on the store). Reported so the lazy
    vs in-memory trade-off is explicit."""
    total = 0
    rowwise = getattr(obj, "rowwise_data", {}) or {}
    for v in rowwise.values():
        if not _is_device_array(v) and hasattr(v, "shape") and hasattr(v, "dtype"):
            total += int(np.prod(v.shape)) * np.dtype(v.dtype).itemsize
    return total


# --------------------------------------------------------------------------- #
# Source construction                                                          #
# --------------------------------------------------------------------------- #
def build_static_spectrum():
    """A fixed 1-D reference spectrum resampled per call (the ``static`` source)."""
    ref_w = np.linspace(WAVELENGTH_MIN * 0.5, WAVELENGTH_MAX * 1.5, 3000)
    cont = _planck_like(ref_w, 5777.0)
    flux = cont * (1.0 - 0.4 * np.exp(-0.5 * ((ref_w - 5890.0) / 4.0) ** 2))

    class StaticSpectrum:
        stellar_parameter_names = ["Teff"]

        def __init__(self):
            self._ref_logw = jnp.log10(jnp.asarray(ref_w))
            self._ref_flux = jnp.asarray(flux)
            self._ref_cont = jnp.asarray(cont)

        @property
        def solar_parameters(self):
            return jnp.array([5777.0])

        def to_parameters(self, parameter_values=None):
            return self.solar_parameters

        def intensity(self, log_wavelengths, mu, parameters):  # noqa: ARG002
            f = jnp.interp(log_wavelengths, self._ref_logw, self._ref_flux)
            c = jnp.interp(log_wavelengths, self._ref_logw, self._ref_cont)
            return jnp.stack([f, c], axis=-1)

    return StaticSpectrum()


def build_gaussian():
    centres = [4861.0, 5183.0, 5270.0, 5500.0, 5890.0, 6563.0]
    return GaussianLineEmulator(
        line_centers=centres,
        line_widths=[0.3 + 0.05 * i for i in range(len(centres))],
        line_depths=[0.45, 0.30, 0.25, 0.50, 0.55, 0.40],
    )


def build_aemu(hf_name):
    from spice.spectrum.aemu_spectrum_emulator import (
        IntensityPretrainedAemuSpectrumEmulator,
    )
    return IntensityPretrainedAemuSpectrumEmulator(hf_name)


def aemu_weight_bytes(emulator_obj):
    """Sum ``.nbytes`` of the flax parameter pytree of an aemu emulator."""
    try:
        params = emulator_obj.emulator.params
        leaves = jax.tree_util.tree_leaves(params)
        return int(sum(_array_nbytes(x) for x in leaves))
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Synthesis probe (optional)                                                   #
# --------------------------------------------------------------------------- #
def _run_one_synthesis(source):
    from spice.models import IcosphereModel
    from spice.spectrum.spectrum import simulate_observed_flux

    lo = getattr(source, "min_stellar_parameters", None)
    hi = getattr(source, "max_stellar_parameters", None)
    if lo is not None and hi is not None:
        params = jnp.asarray(0.5 * (np.asarray(lo, float) + np.asarray(hi, float)))
    else:
        params = jnp.asarray(source.solar_parameters)
    names = list(source.stellar_parameter_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = IcosphereModel.construct(SYNTH_MESH_VERTICES, 1.0, 1.0, params, names,
                                        override_log_g=False)
        log_w = jnp.log10(jnp.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, SYNTH_N_WAVELENGTHS))
        out = simulate_observed_flux(source.intensity, mesh, log_w,
                                     distance=SYNTH_DISTANCE_PC,
                                     chunk_size=SYNTH_CHUNK,
                                     wavelengths_chunk_size=SYNTH_WL_CHUNK)
        out.block_until_ready()
    return out


# --------------------------------------------------------------------------- #
# CSV                                                                          #
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["precision", "source", "variant", "n_nodes", "n_wavelengths",
              "resident_device_mb", "predicted_data_mb", "lazy_store_mb",
              "rss_after_construct_mb", "synthesis_peak_increase_mb"]


class CsvSink:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.precision = None  # set per pass; injected into every row
        self._fh = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._w.writeheader()
        self._fh.flush()

    def add(self, row):
        full = {k: row.get(k, "") for k in CSV_FIELDS}
        full["precision"] = self.precision
        self.rows.append(full)
        self._w.writerow(full)
        self._fh.flush()
        return full

    def close(self):
        self._fh.close()
        print(f"\nWrote {len(self.rows)} rows to {self.path}")


def _mb(x):
    return x / 1e6


def _rss_suffix(res):
    s = f"  RSS={res['rss_after_construct_mb']:.1f} MB"
    if "synthesis_peak_increase_mb" in res:
        s += f"  synth peak +{res['synthesis_peak_increase_mb']:.1f} MB"
    return s


# --------------------------------------------------------------------------- #
# Measurement core                                                             #
# --------------------------------------------------------------------------- #
def _measure_object(build_fn, *, with_synthesis):
    """Construct via ``build_fn`` (no args), return a dict of measurements and
    the constructed object (caller decides whether to keep it)."""
    gc.collect()
    obj = build_fn()
    # Make sure any lazily-deferred device_put / jit setup is realised.
    for attr in ("rowwise_data", "static_data"):
        for v in (getattr(obj, attr, {}) or {}).values():
            if hasattr(v, "block_until_ready"):
                v.block_until_ready()
    gi = getattr(obj, "grid_index", None)
    if gi is not None and hasattr(gi, "keys"):
        gi.keys.block_until_ready()
    gc.collect()

    res = {
        "resident_device_mb": _mb(resident_device_bytes(obj)),
        "lazy_store_mb": _mb(zarr_handle_bytes(obj)),
        "rss_after_construct_mb": _mb(rss_bytes()),
    }
    if with_synthesis:
        gc.collect()
        peak_before = maxrss_bytes()
        out = _run_one_synthesis(obj)
        del out
        gc.collect()
        res["synthesis_peak_increase_mb"] = _mb(max(0, maxrss_bytes() - peak_before))
    return res, obj


# --------------------------------------------------------------------------- #
# Sweeps                                                                       #
# --------------------------------------------------------------------------- #
def sweep_grid(cfg, sink, tmp_root):
    print("\n--- source: grid (FluxLazyZarrInterpolator) ---")
    # (a) scale #nodes at fixed #wavelengths
    print(f"  [nodes sweep, n_wl={cfg.n_wl_for_node_sweep}]")
    for n_nodes in cfg.node_counts:
        n_teff = max(2, round(n_nodes / N_LOGG))
        _grid_point(cfg, sink, tmp_root, [n_teff, N_LOGG], cfg.n_wl_for_node_sweep)
    # (b) scale #wavelengths at fixed #nodes
    n_teff = max(2, round(cfg.nodes_for_wl_sweep / N_LOGG))
    print(f"  [wavelengths sweep, {n_teff * N_LOGG} nodes]")
    for n_wave in cfg.n_wavelengths_sweep:
        _grid_point(cfg, sink, tmp_root, [n_teff, N_LOGG], n_wave)


def _grid_point(cfg, sink, tmp_root, axis_lengths, n_wave):
    tag = "x".join(str(n) for n in axis_lengths)
    store = os.path.join(tmp_root, f"mem_{tag}_{n_wave}.zarr")
    n_nodes, param_names, _ = write_synthetic_grid(store, axis_lengths, n_wave)
    predicted = 2 * n_nodes * n_wave * 4  # float32 flux + continuum
    print(f"    {n_nodes:>7d} nodes x {n_wave:>7d} wl  "
          f"(data ~ {_mb(predicted):8.1f} MB):")
    try:
        for in_memory in cfg.in_memory_modes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res, obj = _measure_object(
                    lambda: FluxLazyZarrInterpolator(store, params=list(param_names),
                                                     in_memory=in_memory),
                    with_synthesis=cfg.with_synthesis)
            variant = "in_memory" if in_memory else "lazy"
            print(f"      {variant:<10s} resident={res['resident_device_mb']:8.2f} MB"
                  f"  (lazy-store={res['lazy_store_mb']:8.2f} MB)"
                  + _rss_suffix(res))
            sink.add({
                "source": "grid", "variant": variant,
                "n_nodes": n_nodes, "n_wavelengths": n_wave,
                "predicted_data_mb": _mb(predicted),
                **res,
            })
            del obj
            gc.collect()
    finally:
        shutil.rmtree(store, ignore_errors=True)


def measure_simple_sources(cfg, sink):
    builders = []
    if "static" in cfg.sources:
        builders.append(("static", "-", build_static_spectrum))
    if "gaussian" in cfg.sources:
        builders.append(("gaussian", "-", build_gaussian))
    for name, variant, fn in builders:
        print(f"\n--- source: {name} ---")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, obj = _measure_object(fn, with_synthesis=cfg.with_synthesis)
        print(f"  resident={res['resident_device_mb']:.3f} MB" + _rss_suffix(res))
        sink.add({"source": name, "variant": variant, **res})
        del obj
        gc.collect()


def measure_aemu_sources(cfg, sink):
    wanted = [s for s in cfg.sources if s in AEMU_BUNDLES]
    if not wanted:
        return
    try:
        from spice.spectrum.aemu_spectrum_emulator import (  # noqa: F401
            IntensityPretrainedAemuSpectrumEmulator,
        )
    except (ImportError, ValueError) as exc:
        print(f"\n(skipping aemu sources {wanted}: astro_emulators_toolkit "
              f"not available -- {exc})")
        return
    for s in wanted:
        hf = AEMU_BUNDLES[s]
        print(f"\n--- source: {s}  (aemu bundle '{hf}') ---")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, obj = _measure_object(lambda: build_aemu(hf),
                                       with_synthesis=cfg.with_synthesis)
        wbytes = aemu_weight_bytes(obj)
        # The frozen-apply closure also holds the params; resident_device_bytes
        # may or may not have reached them, so report the weight pytree size
        # explicitly via predicted_data_mb.
        print(f"  weights={_mb(wbytes):.2f} MB"
              f"  resident(reached)={res['resident_device_mb']:.2f} MB"
              + _rss_suffix(res))
        sink.add({"source": s, "variant": "-",
                  "predicted_data_mb": _mb(wbytes), **res})
        del obj
        gc.collect()


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def make_plot(rows, path):
    grid_rows = [r for r in rows if r["source"] == "grid" and r["n_nodes"]]
    if not grid_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping plot: {exc})")
        return

    def _data_mb(r):
        return float(r["predicted_data_mb"])

    fig, ax = plt.subplots(figsize=(8, 6))
    # Reference y = x line (resident == raw data size); under the in_memory points.
    xs = sorted({_data_mb(r) for r in grid_rows})
    if xs:
        ax.plot(xs, xs, ls="--", color="grey", alpha=0.6, zorder=1,
                label="resident = raw data")
    # Scatter (markers only): the node and wavelength sweeps can land on the same
    # raw-data size, so connecting lines would be misleading.
    precs = sorted({r.get("precision") for r in grid_rows}, key=lambda p: (p is None, p))
    for variant, label, marker in (("in_memory", "in_memory=True (resident)", "o"),
                                   ("lazy", "in_memory=False (resident)", "s")):
        for prec in precs:
            pts = [r for r in grid_rows if r["variant"] == variant
                   and r.get("precision") == prec]
            if not pts:
                continue
            lbl = label + (f" [{prec}]" if len(precs) > 1 else "")
            ax.plot([_data_mb(r) for r in pts],
                    [float(r["resident_device_mb"]) for r in pts],
                    marker=marker, linestyle="", markersize=8, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("raw grid data size (flux + continuum, float32) [MB]")
    ax.set_ylabel("resident device bytes held by interpolator [MB]")
    ax.set_title("Grid memory footprint vs grid size")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
ALL_SOURCES = ("static", "gaussian", "grid") + tuple(AEMU_BUNDLES)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="smaller sweeps")
    p.add_argument("--sources", nargs="+", default=("static", "gaussian", "grid"),
                   choices=ALL_SOURCES, help="which sources to measure")
    p.add_argument("--node-counts", type=int, nargs="+", default=None,
                   help=f"grid-node counts (default {DEFAULT_NODE_COUNTS})")
    p.add_argument("--n-wavelengths", type=int, nargs="+", default=None,
                   help=f"wavelength counts for the grid wavelengths sweep "
                        f"(default {DEFAULT_WL_FOR_SWEEP})")
    p.add_argument("--n-wl-for-node-sweep", type=int, default=DEFAULT_N_WAVELENGTHS,
                   help="fixed wavelength count during the grid node sweep")
    p.add_argument("--nodes-for-wl-sweep", type=int, default=NODES_FOR_WL_SWEEP,
                   help="fixed node count during the grid wavelength sweep")
    p.add_argument("--in-memory", type=_parse_in_memory, default=(False, True),
                   metavar="{true,false,both}", help="grid in_memory modes (default both)")
    p.add_argument("--precision", choices=("x32", "x64", "both"), default="both",
                   help="JAX float precision; 'both' measures every source in x32 and "
                        "x64 (aemu_* sources are x64-only either way; default both)")
    p.add_argument("--with-synthesis", action="store_true",
                   help="also run one simulate_observed_flux call and report RSS afterwards")
    p.add_argument("--csv", default=os.path.join(_HERE, "memory_benchmark.csv"))
    p.add_argument("--plot", default=os.path.join(_HERE, "memory_benchmark.png"))
    p.add_argument("--no-plot", action="store_true")
    return p


class _Cfg:
    pass


def main(argv=None):
    args = build_parser().parse_args(argv)
    if psutil is None:
        print("(psutil not available -- RSS deltas fall back to ru_maxrss, which "
              "is a monotonic high-water mark and so less meaningful)")

    cfg = _Cfg()
    cfg.sources = tuple(args.sources)
    if args.quick:
        cfg.node_counts = args.node_counts or QUICK_NODE_COUNTS
        cfg.n_wavelengths_sweep = args.n_wavelengths or QUICK_WL_FOR_SWEEP
    else:
        cfg.node_counts = args.node_counts or DEFAULT_NODE_COUNTS
        cfg.n_wavelengths_sweep = args.n_wavelengths or DEFAULT_WL_FOR_SWEEP
    cfg.n_wl_for_node_sweep = args.n_wl_for_node_sweep
    cfg.nodes_for_wl_sweep = args.nodes_for_wl_sweep
    cfg.in_memory_modes = args.in_memory
    cfg.with_synthesis = args.with_synthesis
    precisions = _resolve_precisions(args.precision)

    print("Memory benchmark")
    print(f"  precision       : {', '.join(precisions)}")
    print(f"  sources         : {list(cfg.sources)}")
    if "grid" in cfg.sources:
        print(f"  grid in_memory  : {cfg.in_memory_modes}")
        print(f"  grid node counts: {list(cfg.node_counts)}")
        print(f"  grid wl counts  : {list(cfg.n_wavelengths_sweep)}")
    print(f"  with_synthesis  : {cfg.with_synthesis}")

    aemu_requested = [s for s in cfg.sources if s in AEMU_BUNDLES]

    sink = CsvSink(args.csv)
    tmp_root = tempfile.mkdtemp(prefix="spice_mem_bench_")
    try:
        for precision in precisions:
            jax.config.update("jax_enable_x64", precision == "x64")
            sink.precision = precision
            print(f"\n########## precision: {precision} "
                  f"(jax_enable_x64={jax.config.read('jax_enable_x64')}) ##########")
            measure_simple_sources(cfg, sink)
            if "grid" in cfg.sources:
                sweep_grid(cfg, sink, tmp_root)
            if precision == "x64":
                measure_aemu_sources(cfg, sink)
            elif aemu_requested:
                print(f"\n(skipping aemu sources {aemu_requested} at x32 -- "
                      "aemu runs x64 only)")
    finally:
        sink.close()
        shutil.rmtree(tmp_root, ignore_errors=True)

    if not args.no_plot and sink.rows:
        make_plot(sink.rows, args.plot)


if __name__ == "__main__":
    main()
