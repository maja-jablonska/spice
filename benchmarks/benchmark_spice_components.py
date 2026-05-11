#!/usr/bin/env python
"""Benchmark spectrum synthesis in SPICE.

Three intensity sources of increasing cost are compared, all driven through
``spice.spectrum.simulate_observed_flux``:

* ``static``       – a fixed, precomputed spectrum that is merely interpolated
  onto the requested wavelengths and is independent of stellar parameters and
  ``mu``.  This isolates the cost of SPICE's own machinery: mesh geometry,
  Doppler shifting, area/visibility weighting, the wavelength + mesh-element
  chunking, and the ``lax.scan`` accumulation.
* ``gaussian``     – ``GaussianLineEmulator``: a small parametric emulator
  (Planck continuum plus a handful of Gaussian absorption lines).
* ``linear_grid``  – ``FluxLazyZarrInterpolator`` over a tiny synthetic zarr
  grid: multilinear interpolation in (Teff, logg), linear wavelength
  resampling, and a linear limb-darkening law.  A stand-in for a real
  precomputed atmosphere grid.

Three sweeps are run for each source:

1. ``wavelengths`` – scaling with the number of wavelength points (fixed mesh,
   fixed chunk sizes).
2. ``mesh``        – scaling with mesh size / number of surface elements (fixed
   wavelengths, fixed chunk sizes).
3. ``chunks``      – a 2-D sweep over the mesh-element chunk size and the
   wavelength chunk size (fixed mesh, fixed wavelengths).

Results are written to a CSV; unless ``--no-plot`` is given they are also
plotted.

Examples
--------
    python benchmark_spice_components.py
    python benchmark_spice_components.py --quick
    python benchmark_spice_components.py --sources static linear_grid \\
        --mesh-resolutions 300 1000 5000 --n-runs 7
"""

import argparse
import csv
import math
import os
import sys
import tempfile
import time
import warnings
from contextlib import ExitStack

import numpy as np

# SPICE pins JAX to the CPU backend on macOS (the Metal backend is slow/broken
# for this workload), but only once ``spice`` itself is imported -- do it first
# so an experimental jax-metal plugin can't grab the backend during
# ``import jax``.  On Linux this is left alone, so a GPU node picks up CUDA
# automatically; set ``JAX_PLATFORMS`` yourself to override either way.
if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Prefer the in-repo source tree when running from a checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(os.path.join(_SRC, "spice")):
    sys.path.insert(0, _SRC)

import jax
import jax.numpy as jnp

from spice.models import IcosphereModel
from spice.spectrum.spectrum import simulate_observed_flux
from spice.spectrum.gaussian_line_emulator import GaussianLineEmulator
from spice.spectrum.lazy_zarr_interpolator import FluxLazyZarrInterpolator


# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
SOURCES = ("static", "gaussian", "linear_grid")

# Wavelength window used for every sweep (Angstrom).  Wide enough that the
# Gaussian lines fall inside it and the synthetic grid brackets it.
WAVELENGTH_MIN = 4500.0
WAVELENGTH_MAX = 6500.0

# Sweep 1: scaling with the number of wavelengths.
N_WAVELENGTHS_SWEEP = (200, 1000, 4000, 16000)
FIXED_MESH_FOR_WL_SWEEP = 2000          # n_vertices

# Sweep 2: scaling with mesh size.
MESH_RESOLUTIONS = (300, 1000, 5000, 20000)   # n_vertices
FIXED_N_WAVELENGTHS_FOR_MESH_SWEEP = 2000

# Sweep 3: chunk-size grid.  Values are clamped to the array length, so e.g.
# 100000 effectively means "no chunking".
MESH_CHUNK_SIZES = (256, 1024, 4096, 100000)
WAVELENGTH_CHUNK_SIZES = (256, 1024, 4096, 100000)
FIXED_MESH_FOR_CHUNK_SWEEP = 5000
FIXED_N_WAVELENGTHS_FOR_CHUNK_SWEEP = 4000

# Chunk sizes used by the (non-chunk) scaling sweeps.
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_WAVELENGTHS_CHUNK_SIZE = 1024

# Synthetic grid for the ``linear_grid`` source.
GRID_TEFF_AXIS = (4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0)
GRID_LOGG_AXIS = (3.5, 4.0, 4.5, 5.0)
GRID_N_WAVELENGTHS = 4000     # native sampling of the synthetic grid

N_TIMING_RUNS = 5
N_WARMUP_RUNS = 1
DISTANCE_PC = 10.0


# --------------------------------------------------------------------------- #
# Intensity sources                                                            #
# --------------------------------------------------------------------------- #
class StaticSpectrum:
    """A fixed precomputed spectrum, resampled onto the requested wavelengths.

    Independent of the stellar parameters and ``mu``, so a benchmark using this
    source measures essentially only the SPICE integration machinery.  It is a
    class (not a closure) so that ``self.intensity`` is a stable, hashable
    static argument to ``simulate_observed_flux``'s jit cache.
    """

    # Nominal parameter list; the values are never read by ``intensity``.
    stellar_parameter_names = ["Teff"]

    def __init__(self, ref_wavelengths, ref_flux, ref_continuum):
        self._ref_logw = jnp.log10(jnp.asarray(ref_wavelengths))
        self._ref_flux = jnp.asarray(ref_flux)
        self._ref_cont = jnp.asarray(ref_continuum)

    @property
    def solar_parameters(self):
        return jnp.array([5777.0])

    def to_parameters(self, parameter_values=None):
        return self.solar_parameters

    def intensity(self, log_wavelengths, mu, parameters):  # noqa: ARG002
        flux = jnp.interp(log_wavelengths, self._ref_logw, self._ref_flux)
        cont = jnp.interp(log_wavelengths, self._ref_logw, self._ref_cont)
        return jnp.stack([flux, cont], axis=-1)


def _planck_like(wavelengths_angstrom, teff):
    """A cheap Planck-shaped curve in erg/s/cm^2/A units (only used to fill the
    synthetic grid / static spectrum with something non-trivial)."""
    h = 6.62607015e-27
    c = 2.99792458e10
    k = 1.380649e-16
    w_cm = np.asarray(wavelengths_angstrom) * 1e-8
    return (2 * h * c ** 2 / w_cm ** 5 / (np.exp(h * c / (w_cm * k * teff)) - 1.0)) * 1e-8


def build_static_spectrum():
    ref_w = np.linspace(WAVELENGTH_MIN * 0.5, WAVELENGTH_MAX * 1.5, 3000)
    cont = _planck_like(ref_w, 5777.0)
    # A few broad absorption dips so flux != continuum.
    flux = cont.copy()
    for centre, width, depth in [(4861.0, 8.0, 0.4), (5183.0, 6.0, 0.3),
                                 (5890.0, 4.0, 0.5), (6563.0, 10.0, 0.45)]:
        flux *= 1.0 - depth * np.exp(-0.5 * ((ref_w - centre) / width) ** 2)
    return StaticSpectrum(ref_w, flux, cont)


def build_gaussian_emulator():
    # A handful of lines spread across the benchmarking window.
    centres = [4861.0, 5183.0, 5270.0, 5500.0, 5890.0, 6563.0]
    return GaussianLineEmulator(
        line_centers=centres,
        line_widths=[0.3 + 0.05 * i for i in range(len(centres))],
        line_depths=[0.45, 0.30, 0.25, 0.50, 0.55, 0.40],
    )


def _write_synthetic_grid(store_path):
    """Write a tiny (Teff, logg) zarr grid plus its ``index.parquet``."""
    import pandas as pd
    import zarr

    wavelengths = np.linspace(WAVELENGTH_MIN * 0.5, WAVELENGTH_MAX * 1.5,
                              GRID_N_WAVELENGTHS).astype(np.float32)
    rows = [(float(t), float(g)) for t in GRID_TEFF_AXIS for g in GRID_LOGG_AXIS]
    n_rows = len(rows)
    n_wave = wavelengths.shape[0]

    flux = np.empty((n_rows, n_wave), dtype=np.float32)
    continuum = np.empty((n_rows, n_wave), dtype=np.float32)
    for i, (teff, logg) in enumerate(rows):
        cont = _planck_like(wavelengths, teff) * (1.0 + 0.02 * (logg - 4.5))
        spec = cont.copy()
        for centre, width, depth in [(4861.0, 6.0, 0.35), (5183.0, 5.0, 0.28),
                                     (5890.0, 3.5, 0.5), (6563.0, 9.0, 0.4)]:
            spec *= 1.0 - depth * np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)
        flux[i, :] = spec.astype(np.float32)
        continuum[i, :] = cont.astype(np.float32)

    group = zarr.open_group(str(store_path), mode="w")
    group.create_array("wavelength", data=wavelengths)
    group.create_array("flux", data=flux, chunks=(min(8, n_rows), n_wave))
    group.create_array("continuum", data=continuum, chunks=(min(8, n_rows), n_wave))
    group.create_array("params", data=np.array(rows, dtype=np.float32))
    # ``param_names`` is the only thing LazyZarrInterpolator could still want
    # from the store, but we pass ``params=["teff", "logg"]`` explicitly so it
    # is never read; skip it (zarr v3 won't store an object-dtype string array).

    df = pd.DataFrame({
        "teff": [r[0] for r in rows],
        "logg": [r[1] for r in rows],
        "row_idx": np.arange(n_rows, dtype=np.int64),
    })
    df.to_parquet(os.path.join(str(store_path), "index.parquet"))


def build_linear_grid(stack: ExitStack):
    tmpdir = stack.enter_context(tempfile.TemporaryDirectory(prefix="spice_bench_grid_"))
    store_path = os.path.join(tmpdir, "grid.zarr")
    _write_synthetic_grid(store_path)
    return FluxLazyZarrInterpolator(
        store_path,
        params=["teff", "logg"],
        in_memory=True,          # jit-friendly accumulation path
    )


def make_source(name, stack: ExitStack):
    if name == "static":
        return build_static_spectrum()
    if name == "gaussian":
        return build_gaussian_emulator()
    if name == "linear_grid":
        return build_linear_grid(stack)
    raise ValueError(f"unknown source {name!r}")


# --------------------------------------------------------------------------- #
# Mesh + wavelength helpers                                                    #
# --------------------------------------------------------------------------- #
def make_mesh(source, n_vertices):
    param_names = list(source.stellar_parameter_names)
    params = jnp.asarray(source.solar_parameters)
    # ``override_log_g`` only matters when 'logg' is one of the parameters
    # (the synthetic grid). Suppress its warning for the single-parameter
    # emulators by disabling it there.
    override_log_g = any(
        pn.lower() in ("logg", "log_g", "loggs", "log_gs", "log g", "log gs")
        for pn in param_names
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IcosphereModel.construct(
            n_vertices, 1.0, 1.0, params, param_names,
            override_log_g=override_log_g,
        )


def n_mesh_elements(mesh):
    return int(mesh.parameters.shape[0])


def make_log_wavelengths(n_wavelengths):
    return jnp.log10(jnp.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n_wavelengths))


# --------------------------------------------------------------------------- #
# Timing                                                                       #
# --------------------------------------------------------------------------- #
def time_run(source, mesh, log_wavelengths, chunk_size, wavelengths_chunk_size,
             n_runs, n_warmup, disable_doppler_shift):
    """Return (mean_s, std_s, compile_s) for one configuration."""
    def call():
        out = simulate_observed_flux(
            source.intensity, mesh, log_wavelengths,
            distance=DISTANCE_PC,
            chunk_size=chunk_size,
            wavelengths_chunk_size=wavelengths_chunk_size,
            disable_doppler_shift=disable_doppler_shift,
        )
        out.block_until_ready()
        return out

    t0 = time.perf_counter()
    first = call()
    compile_s = time.perf_counter() - t0
    if not bool(jnp.all(jnp.isfinite(first))):
        warnings.warn("non-finite flux encountered; result may be unreliable")

    for _ in range(max(0, n_warmup - 1)):
        call()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times)), float(compile_s)


# --------------------------------------------------------------------------- #
# Sweeps                                                                       #
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["sweep", "source", "n_vertices", "n_mesh_elements", "n_wavelengths",
              "chunk_size", "wavelengths_chunk_size", "mean_time_s", "std_time_s",
              "compile_time_s"]


class CsvSink:
    """Streams benchmark rows to a CSV file, flushing after every row so a
    long run that dies partway still leaves all completed measurements on disk.
    Also keeps the rows in memory for plotting / the end-of-run summary."""

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


def _row(sweep, source_name, n_vertices, n_mesh, n_wl, cs, wcs, mean_s, std_s, compile_s):
    return {
        "sweep": sweep,
        "source": source_name,
        "n_vertices": n_vertices,
        "n_mesh_elements": n_mesh,
        "n_wavelengths": n_wl,
        "chunk_size": cs,
        "wavelengths_chunk_size": wcs,
        "mean_time_s": mean_s,
        "std_time_s": std_s,
        "compile_time_s": compile_s,
    }


def sweep_wavelengths(source_name, source, cfg, sink):
    print(f"\n--- [{source_name}] sweep: scaling with #wavelengths "
          f"(mesh={cfg.fixed_mesh_for_wl} vertices) ---")
    mesh = make_mesh(source, cfg.fixed_mesh_for_wl)
    n_mesh = n_mesh_elements(mesh)
    for n_wl in cfg.n_wavelengths_sweep:
        log_w = make_log_wavelengths(n_wl)
        cs = min(cfg.chunk_size, n_mesh)
        wcs = min(cfg.wavelengths_chunk_size, n_wl)
        mean_s, std_s, comp_s = time_run(
            source, mesh, log_w, cs, wcs,
            cfg.n_runs, cfg.n_warmup, cfg.disable_doppler_shift)
        print(f"  n_wl={n_wl:>7d}  cs={cs:>6d} wcs={wcs:>6d}  "
              f"{mean_s*1e3:9.2f} ± {std_s*1e3:7.2f} ms   (compile {comp_s:6.2f} s)")
        sink.add(_row("wavelengths", source_name, cfg.fixed_mesh_for_wl, n_mesh,
                      n_wl, cs, wcs, mean_s, std_s, comp_s))


def sweep_mesh(source_name, source, cfg, sink):
    print(f"\n--- [{source_name}] sweep: scaling with mesh size "
          f"(n_wl={cfg.fixed_n_wl_for_mesh}) ---")
    log_w = make_log_wavelengths(cfg.fixed_n_wl_for_mesh)
    for n_vertices in cfg.mesh_resolutions:
        mesh = make_mesh(source, n_vertices)
        n_mesh = n_mesh_elements(mesh)
        cs = min(cfg.chunk_size, n_mesh)
        wcs = min(cfg.wavelengths_chunk_size, cfg.fixed_n_wl_for_mesh)
        mean_s, std_s, comp_s = time_run(
            source, mesh, log_w, cs, wcs,
            cfg.n_runs, cfg.n_warmup, cfg.disable_doppler_shift)
        print(f"  n_vtx={n_vertices:>7d} (n_mesh={n_mesh:>7d})  cs={cs:>6d} wcs={wcs:>6d}  "
              f"{mean_s*1e3:9.2f} ± {std_s*1e3:7.2f} ms   (compile {comp_s:6.2f} s)")
        sink.add(_row("mesh", source_name, n_vertices, n_mesh,
                      cfg.fixed_n_wl_for_mesh, cs, wcs, mean_s, std_s, comp_s))


def sweep_chunks(source_name, source, cfg, sink):
    print(f"\n--- [{source_name}] sweep: chunk sizes "
          f"(mesh={cfg.fixed_mesh_for_chunks} vertices, n_wl={cfg.fixed_n_wl_for_chunks}) ---")
    mesh = make_mesh(source, cfg.fixed_mesh_for_chunks)
    n_mesh = n_mesh_elements(mesh)
    n_wl = cfg.fixed_n_wl_for_chunks
    log_w = make_log_wavelengths(n_wl)

    # De-duplicate after clamping (e.g. several "no chunking" values collapse).
    mesh_chunks = sorted({min(c, n_mesh) for c in cfg.mesh_chunk_sizes})
    wl_chunks = sorted({min(c, n_wl) for c in cfg.wavelength_chunk_sizes})

    for cs in mesh_chunks:
        for wcs in wl_chunks:
            mean_s, std_s, comp_s = time_run(
                source, mesh, log_w, cs, wcs,
                cfg.n_runs, cfg.n_warmup, cfg.disable_doppler_shift)
            n_mc = math.ceil(n_mesh / cs)
            n_wc = math.ceil(n_wl / wcs)
            print(f"  cs={cs:>6d} ({n_mc:>3d} mesh-chunks)  wcs={wcs:>6d} ({n_wc:>3d} wl-chunks)  "
                  f"{mean_s*1e3:9.2f} ± {std_s*1e3:7.2f} ms   (compile {comp_s:6.2f} s)")
            sink.add(_row("chunks", source_name, cfg.fixed_mesh_for_chunks, n_mesh,
                          n_wl, cs, wcs, mean_s, std_s, comp_s))


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #


def make_plots(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(skipping plots: {exc})")
        return

    sources = sorted({r["source"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: scaling with #wavelengths.
    ax = axes[0]
    for s in sources:
        pts = sorted((r for r in rows if r["sweep"] == "wavelengths" and r["source"] == s),
                     key=lambda r: r["n_wavelengths"])
        if pts:
            ax.errorbar([r["n_wavelengths"] for r in pts],
                        [r["mean_time_s"] * 1e3 for r in pts],
                        yerr=[r["std_time_s"] * 1e3 for r in pts],
                        marker="o", capsize=3, label=s)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of wavelengths"); ax.set_ylabel("time per call [ms]")
    ax.set_title("Scaling with #wavelengths"); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: scaling with mesh size.
    ax = axes[1]
    for s in sources:
        pts = sorted((r for r in rows if r["sweep"] == "mesh" and r["source"] == s),
                     key=lambda r: r["n_mesh_elements"])
        if pts:
            ax.errorbar([r["n_mesh_elements"] for r in pts],
                        [r["mean_time_s"] * 1e3 for r in pts],
                        yerr=[r["std_time_s"] * 1e3 for r in pts],
                        marker="o", capsize=3, label=s)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of surface elements"); ax.set_ylabel("time per call [ms]")
    ax.set_title("Scaling with mesh size"); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: chunk-size heat-strip (one line per mesh-chunk value).
    ax = axes[2]
    for s in sources:
        cpts = [r for r in rows if r["sweep"] == "chunks" and r["source"] == s]
        if not cpts:
            continue
        for cs in sorted({r["chunk_size"] for r in cpts}):
            pts = sorted((r for r in cpts if r["chunk_size"] == cs),
                         key=lambda r: r["wavelengths_chunk_size"])
            ax.plot([r["wavelengths_chunk_size"] for r in pts],
                    [r["mean_time_s"] * 1e3 for r in pts],
                    marker="o", label=f"{s} cs={cs}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wavelength chunk size"); ax.set_ylabel("time per call [ms]")
    ax.set_title("Effect of chunk sizes"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
class Config:
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES),
                   help="which intensity sources to benchmark")
    p.add_argument("--sweeps", nargs="+", choices=("wavelengths", "mesh", "chunks"),
                   default=["wavelengths", "mesh", "chunks"], help="which sweeps to run")
    p.add_argument("--n-runs", type=int, default=N_TIMING_RUNS,
                   help="timed runs per configuration")
    p.add_argument("--n-warmup", type=int, default=N_WARMUP_RUNS,
                   help="warmup runs (>=1; the first triggers compilation)")
    p.add_argument("--n-wavelengths-sweep", nargs="+", type=int, default=list(N_WAVELENGTHS_SWEEP))
    p.add_argument("--mesh-resolutions", nargs="+", type=int, default=list(MESH_RESOLUTIONS))
    p.add_argument("--mesh-chunk-sizes", nargs="+", type=int, default=list(MESH_CHUNK_SIZES))
    p.add_argument("--wavelength-chunk-sizes", nargs="+", type=int, default=list(WAVELENGTH_CHUNK_SIZES))
    p.add_argument("--fixed-mesh-for-wl", type=int, default=FIXED_MESH_FOR_WL_SWEEP)
    p.add_argument("--fixed-n-wl-for-mesh", type=int, default=FIXED_N_WAVELENGTHS_FOR_MESH_SWEEP)
    p.add_argument("--fixed-mesh-for-chunks", type=int, default=FIXED_MESH_FOR_CHUNK_SWEEP)
    p.add_argument("--fixed-n-wl-for-chunks", type=int, default=FIXED_N_WAVELENGTHS_FOR_CHUNK_SWEEP)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="mesh-element chunk size used by the (non-chunk) scaling sweeps")
    p.add_argument("--wavelengths-chunk-size", type=int, default=DEFAULT_WAVELENGTHS_CHUNK_SIZE,
                   help="wavelength chunk size used by the (non-chunk) scaling sweeps")
    p.add_argument("--disable-doppler-shift", action="store_true",
                   help="skip the per-element Doppler shift in simulate_observed_flux")
    p.add_argument("--no-x64", action="store_true",
                   help="do not enable float64 (SPICE's tests run with x64 on)")
    p.add_argument("--output-prefix", default="spice_components_benchmark")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="tiny sizes / 2 runs — for smoke-testing the script")
    return p.parse_args(argv)


def config_from_args(args):
    cfg = Config()
    cfg.n_runs = args.n_runs
    cfg.n_warmup = max(1, args.n_warmup)
    cfg.n_wavelengths_sweep = args.n_wavelengths_sweep
    cfg.mesh_resolutions = args.mesh_resolutions
    cfg.mesh_chunk_sizes = args.mesh_chunk_sizes
    cfg.wavelength_chunk_sizes = args.wavelength_chunk_sizes
    cfg.fixed_mesh_for_wl = args.fixed_mesh_for_wl
    cfg.fixed_n_wl_for_mesh = args.fixed_n_wl_for_mesh
    cfg.fixed_mesh_for_chunks = args.fixed_mesh_for_chunks
    cfg.fixed_n_wl_for_chunks = args.fixed_n_wl_for_chunks
    cfg.chunk_size = args.chunk_size
    cfg.wavelengths_chunk_size = args.wavelengths_chunk_size
    cfg.disable_doppler_shift = args.disable_doppler_shift

    if args.quick:
        cfg.n_runs = 2
        cfg.n_warmup = 1
        cfg.n_wavelengths_sweep = [50, 200]
        cfg.mesh_resolutions = [42, 162]
        cfg.mesh_chunk_sizes = [64, 100000]
        cfg.wavelength_chunk_sizes = [64, 100000]
        cfg.fixed_mesh_for_wl = 80
        cfg.fixed_n_wl_for_mesh = 100
        cfg.fixed_mesh_for_chunks = 162
        cfg.fixed_n_wl_for_chunks = 200
        cfg.chunk_size = 1024
        cfg.wavelengths_chunk_size = 1024
    return cfg


def main(argv=None):
    args = parse_args(argv)
    if not args.no_x64:
        jax.config.update("jax_enable_x64", True)

    cfg = config_from_args(args)

    print("SPICE spectrum-synthesis benchmark")
    print("=" * 70)
    print(f"jax {jax.__version__} | backend {jax.default_backend()} | "
          f"x64 {'on' if jax.config.read('jax_enable_x64') else 'off'}")
    print(f"sources : {', '.join(args.sources)}")
    print(f"sweeps  : {', '.join(args.sweeps)}")
    print(f"wavelength window : {WAVELENGTH_MIN:.0f}-{WAVELENGTH_MAX:.0f} A")
    print(f"timed runs / config : {cfg.n_runs}  (warmup {cfg.n_warmup})")
    print(f"doppler shift : {'disabled' if cfg.disable_doppler_shift else 'enabled'}")
    print("=" * 70)

    csv_path = f"{args.output_prefix}.csv"
    sink = CsvSink(csv_path)
    print(f"streaming results to : {csv_path}")
    try:
        with ExitStack() as stack:
            sources = {name: make_source(name, stack) for name in args.sources}
            for name in args.sources:
                src = sources[name]
                if "wavelengths" in args.sweeps:
                    sweep_wavelengths(name, src, cfg, sink)
                if "mesh" in args.sweeps:
                    sweep_mesh(name, src, cfg, sink)
                if "chunks" in args.sweeps:
                    sweep_chunks(name, src, cfg, sink)
    finally:
        sink.close()
    all_rows = sink.rows

    if not args.no_plot and all_rows:
        make_plots(all_rows, f"{args.output_prefix}.png")

    # Compact summary.
    print("\nSummary (mean ms per call):")
    print("-" * 70)
    for sweep in ("wavelengths", "mesh", "chunks"):
        sweep_rows = [r for r in all_rows if r["sweep"] == sweep]
        if not sweep_rows:
            continue
        print(f"[{sweep}]")
        for r in sweep_rows:
            print(f"  {r['source']:<12} n_mesh={r['n_mesh_elements']:>7d} "
                  f"n_wl={r['n_wavelengths']:>7d} cs={r['chunk_size']:>6d} "
                  f"wcs={r['wavelengths_chunk_size']:>6d}  "
                  f"{r['mean_time_s']*1e3:9.2f} ms")
    print("\nDone.")


if __name__ == "__main__":
    main()
