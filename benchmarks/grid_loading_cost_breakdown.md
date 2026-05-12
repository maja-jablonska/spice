# Cost breakdown of loading a linear (zarr) spectral grid

What `LazyZarrInterpolator.__init__` (and its `FluxLazyZarrInterpolator` /
`IntensityLazyZarrInterpolator` subclasses) actually spends time on — i.e.
everything `benchmarks/benchmark_grid_loading.py` measures. Line references are
to `src/spice/spectrum/lazy_zarr_interpolator.py`.

## Steps, in order, with what each scales with

### 1. Module imports — one-time, per process
`import zarr` / `import pandas` / `from tqdm.auto import tqdm` (≈ lines 494–510),
plus JAX itself if not already imported. The **first** interpolator built in a
process pays ≈ 0.1–1 s for these; every later construction pays ≈ 0. This is
why the benchmark reports a separate `first_load_s` and does a warm-up
construction before timing.

### 2. Opening the store + reading metadata — negligible
`zarr.open(zarr_path, mode='r')`, `self.store['flux'].shape[0]` (≈ lines 528–531).
Reads a few small JSON files (`.zgroup` / `.zarray`). Sub-millisecond, independent
of grid size.

### 3. Reading `index.parquet` — scales with the number of grid nodes
`pd.read_parquet(index_path)` (≈ line 540). Cost ∝ (number of grid nodes) ×
(number of parameter columns). A few ms for ~10³ nodes; tens of ms for 10⁵–10⁶
nodes.

### 4. Building the grid index — scales with #nodes (log-linear), weakly with #dims
`SparseGridIndex.from_dataframe` (≈ lines 229–267):

- `np.unique` per axis — O(N log N)
- `np.searchsorted` per axis — O(N log A)
- mixed-radix key encoding `indices @ strides` — O(N·d)
- **`np.argsort(keys)` — O(N log N); the dominant op within this step**
- `jax.device_put` of `keys`, `values`, `strides`, `axes` + `block_until_ready`
  — small integer arrays of length N, cheap

So this is ∝ #nodes (log-linear), essentially independent of the number of
wavelengths, and only weakly dependent on the number of parameter axes. At a
fixed node count it barely moves with dimensionality, because it is small next to
step 6.

> If you pass `sparse=False`, `GridIndex.from_dataframe` (≈ lines 349–365)
> instead allocates a **dense `∏(axis lengths)` int32 array** — combinatorial in
> the number of axes / axis lengths, easily GB or an OOM. That is why
> `sparse=True` is the default; there is a warning above 1 GB (≈ lines 547–555).

### 5. `min` / `max` stellar parameters — negligible
A handful of `.min()` / `.max()` calls on the axis arrays (≈ lines 564–566).

### 6. Loading the arrays — the dominant cost when `in_memory=True`
Loop over `['wavelength', 'flux', 'continuum']` (≈ lines 604–620):

- **`wavelength`** (static, length = `n_wave`): `jax.device_put(np.asarray(arr[:]))`
  — reads + decompresses the whole wavelength vector and copies it to device.
  ∝ #wavelengths, but tiny (`n_wave × 4 B`).
- **`flux` + `continuum`** (row-wise, shape `(N, n_wave)`):
  - **`in_memory=True`** (or `'auto'` under the 2 GB threshold):
    `np.asarray(arr[:])` **reads and decompresses the entire array from the zarr
    store**, then `jax.device_put` does a host→device copy. Cost ≈
    `N × n_wave × 4 B × 2 arrays` of disk I/O + blosc/zstd decompression CPU + a
    large host→device transfer. **This dominates `in_memory=True` load time** and
    scales with the **total grid size = #nodes × #wavelengths**.
  - **`in_memory=False`**: stores only the zarr array *handle*
    (`self.rowwise_data[k] = arr`) — **no spectrum data is read at construction
    time**. The cost is not removed, it is *deferred*: every
    `simulate_observed_flux` call then performs on-the-fly zarr reads /
    decompression of the rows it needs.

### Also affecting the numbers
- **OS page cache / filesystem**: the first read from a cold cache (or a
  network / Lustre filesystem, e.g. on an HPC node) is much slower than a warm
  re-read — this is the gap between `first_load_s` and `mean_load_s`.
- **Compression**: zarr arrays are normally compressed, so step 6's "read"
  includes decompression CPU, not just raw bytes off disk — chunk size and codec
  matter.

## Summary table

| Step | Scales with | Typical weight |
|---|---|---|
| zarr / pandas / JAX import | once per process | 0.1–1 s on the first construction, then 0 |
| `zarr.open` + array metadata | — | sub-ms |
| read `index.parquet` | #nodes × #param-columns | ms → tens of ms |
| build `SparseGridIndex` (argsort dominates) | #nodes (log-linear), weak in #dims | ms → tens of ms |
| dense `GridIndex` (`sparse=False` only) | ∏(axis lengths) | can be GB / OOM — avoid |
| `device_put(wavelength)` | #wavelengths | sub-ms |
| `device_put(flux) + device_put(continuum)`, `in_memory=True` | **#nodes × #wavelengths** (read + decompress + H2D copy) | **dominant** |
| storing array handles, `in_memory=False` | — | ≈ 0 (cost deferred to synthesis time) |

## In one sentence

With `in_memory=True`, grid load time ≈ (total uncompressed grid bytes) ÷
(read + decompress + host→device throughput), and everything else is a few-ms
constant. With `in_memory=False`, loading is just "parse the parquet + build the
sparse index + `device_put` a small wavelength array" — flat and cheap — and the
per-spectrum data cost is paid during synthesis instead.

## Empirical illustration

From a `benchmarks/benchmark_grid_loading.py` run on a CPU backend (synthetic
grids; numbers are indicative, not a hardware benchmark):

- `in_memory=False`: ≈ 5 ms to construct, independent of grid size, node count,
  and number of parameter axes.
- `in_memory=True`: ≈ 20 ms for an ≈ 8 MB grid, ≈ 210 ms for an ≈ 131 MB grid —
  i.e. roughly linear in the uncompressed grid size at a few ms per MB — and
  essentially flat across 1-D → 3-D parameter spaces at a fixed node count,
  confirming that the index build (step 4) is negligible next to the array
  transfer (step 6) in this regime.

Regenerate with:

```bash
python benchmarks/benchmark_grid_loading.py            # full sweeps
python benchmarks/benchmark_grid_loading.py --quick    # fast smoke test
```
