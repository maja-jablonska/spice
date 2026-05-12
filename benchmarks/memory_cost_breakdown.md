# Memory-footprint breakdown: spectral grids and emulators

Companion to `grid_loading_cost_breakdown.md` / `emulator_cost_breakdown.md` and
to `benchmarks/benchmark_memory.py`. Two kinds of memory matter:

* **resident / persistent** — what a constructed interpolator or emulator keeps
  alive for its whole lifetime (mostly device arrays);
* **transient / peak** — what one `simulate_observed_flux` call additionally
  allocates while it runs (chunk buffers, gathered rows, the JIT'd executable,
  XLA scratch, and — in the lazy grid path — zarr decompression buffers).

All sizes below are for the default float32 path. With `jax.config.update("jax_enable_x64", True)`
(which the test suite turns on) every floating-point array doubles.

On a **CPU** backend there is a single memory space, so process RSS covers both
device and host buffers. On a **GPU**, the device arrays listed under "resident"
live in **VRAM**; this is why `LazyZarrInterpolator` has an
`in_memory_threshold_bytes` (default 2 GB) guard before it will eagerly pull a
grid onto the device.

---

## 1. Spectral grid — `LazyZarrInterpolator` / `FluxLazyZarrInterpolator` / `IntensityLazyZarrInterpolator`

Notation: `S` = number of grid nodes (rows in the zarr arrays), `n_λ` = number of
wavelength samples per spectrum, `d` = number of parameter axes, `|a_i|` = number
of distinct values on axis `i` (so `∏|a_i| ≥ S`, with equality iff the grid is a
full hyper-rectangle).

### Resident (held by the constructed object)

| Component | Where | Size | Notes |
|---|---|---|---|
| `wavelength` | `static_data` | `n_λ · 4 B` | one shared 1-D vector |
| sparse grid index `keys` | `grid_index` (`sparse=True`, default) | `S · 8 B` (int64 flat keys) | |
| sparse grid index `values` | `grid_index` | `S · 4 B` (int32 row indices) | |
| sparse grid index `axes` (+ host copies) | `grid_index` | `Σ|a_i| · 8 B` ×2 | tiny |
| `flux`, `continuum` — **`in_memory=True`** | `rowwise_data` | **`2 · S · n_λ · 4 B`** | **the dominant term** |
| `flux`, `continuum` — **`in_memory=False`** | `rowwise_data` | a few KB | only zarr array *handles*; data stays on the store (disk) |
| `min/max_stellar_parameters`, `store` handle | — | ≈ 0 | |

So:

* **`in_memory=True`**: resident ≈ **`2 · S · n_λ · 4 B`** + `S · 12 B` (index) +
  `n_λ · 4 B`. The flux/continuum arrays dwarf everything else once `S·n_λ` is
  non-trivial (e.g. `S = 10⁴`, `n_λ = 2×10⁴` → ≈ 1.6 GB). The benchmark confirms
  the resident bytes equal the closed-form `2·S·n_λ·4 B` to the byte.
* **`in_memory=False`**: resident ≈ `S · 12 B + n_λ · 4 B` — i.e. essentially
  just the sparse index and the wavelength vector (a few tens of KB up to ~12 MB
  for a million-node grid). The spectrum data is never resident; it is read from
  the zarr store on demand.

> **`sparse=False`** swaps the sparse index for a dense `GridIndex`: a full
> `∏|a_i|` int32 array (plus a host copy) — combinatorial in `d`, easily GB or an
> OOM for high-dimensional grids. Hence `sparse=True` is the default and there is
> a >1 GB warning. The flux/continuum accounting above is unchanged.

### Transient (peak during one `simulate_observed_flux` call)

Driven by the chunk sizes (`chunk_size` over mesh elements, `wavelengths_chunk_size`
over wavelengths, and `accumulate_chunk_size` inside the in-memory accumulator):

* Doppler-shifted wavelength grids: ≈ `chunk_size · n_λ · 4 B` per mesh chunk
  (one log-λ grid per surface element in the chunk).
* **`in_memory=True`** accumulation (`_accumulate_in_memory`): per batch of
  `accumulate_chunk_size` queries (default 64) it gathers the `2^d` bracketing
  rows for each query → peak intermediate ≈ **`accumulate_chunk_size · 2^d · n_λ · 4 B`**
  (×2 for flux+continuum). Note the `2^d` — a 5-D grid is 32× a 1-D grid here, so
  lower `accumulate_chunk_size` if `d` is large and `n_λ` big. (`accumulate_chunk_size=None`
  falls back to a fully sequential `lax.map`: `2^d · n_λ` per query, lowest peak,
  slowest.)
* **`in_memory=False`**: each query reads its `2^d` bracketing rows from the zarr
  store, decompressing the chunks that contain them → working set ≈ `2^d ·` (zarr
  chunk bytes) plus the decompressor's scratch, per query; no large persistent
  spike, but synthesis is I/O-bound rather than compute-bound.
* Output accumulator: `n_λ · 2 · 4 B` — small.
* The XLA executable + buffer pools: tens of MB, independent of grid size, paid
  once (first call) and then reused.

**Rule of thumb:** with `in_memory=True`, total ≈ (`2·S·n_λ·4 B` resident) +
(`accumulate_chunk_size·2^d·n_λ·4 B·2` transient). With `in_memory=False`,
resident is negligible and the peak is set by zarr chunking + XLA buffers — you
trade RAM for per-call I/O.

---

## 2. Static spectrum (`StaticSpectrum` — the `static` benchmark source)

* **Resident:** three 1-D arrays of length `n_ref` (log-λ, flux, continuum) —
  `3 · n_ref · dtype` (≈ 72 KB for `n_ref = 3000` in float64, half that in
  float32). The benchmark reads ≈ 0.04 MB.
* **Transient per call:** two `jnp.interp` temporaries of length `n_λ` plus the
  `(n_λ, 2)` output — a few `n_λ·4 B`. Independent of stellar parameters and `μ`.

Effectively a rounding error; this source exists to expose the *shared* synthesis
machinery's footprint (mesh arrays, chunk buffers, XLA), not its own.

## 3. Parametric emulators (Gaussian lines, blackbody)

* **Resident:** a handful of length-`n_lines` arrays (centres/widths/depths), i.e.
  bytes. The benchmark reads ≈ 0 MB.
* **Transient per call:** the Planck continuum and the running absorption product,
  each length `n_λ`, plus one `exp` temporary per line → ≈ `(2 + n_lines) · n_λ · 4 B`.
  `flux(...)` evaluates `intensity` at `mus_number` (default 20) μ-nodes via
  `jax.vmap`, so the activations are ≈ `mus_number ·` that — still small.

## 4. aemu emulators (`astro_emulators_toolkit` pretrained bundles)

* **Resident:** the flax parameter pytree — the **network weights** — plus the
  (small) `model_state`, the `graphdef` (structure only, negligible), and
  `bundle_extras` (for a fixed-grid flux bundle, the native wavelength grid,
  `n_native · 4 B`). For an MLP of hidden width `H` and `L` layers this is
  ≈ `L · H² · 4 B` plus the input layer (`n_input_features · H · 4 B`) and the
  output layer:
  * **wavelength-conditioned intensity bundle** (e.g. `TPayne-spice-*`): output
    is 2 channels → output layer ≈ `H · 2 · 4 B`, so the footprint is just the
    hidden stack — typically a few MB to a few tens of MB;
  * **fixed-grid flux bundle**: output layer ≈ `H · n_native · 4 B`, which can
    dominate (a whole spectrum of weights per hidden unit).

  `benchmark_memory.py` reports this directly as the sum of `.nbytes` over
  `emulator.params` (the `aemu_*` sources are skipped unless the extra is
  installed and the bundle has been fetched).
* **Transient per call:** the forward-pass activations. Under
  `simulate_observed_flux` the network is evaluated for a whole mesh chunk at
  once, and (for the wavelength-conditioned bundles) across the `n_λ` wavelength
  positions, so peak ≈ **`chunk_size · n_λ · H · 4 B`** per layer (XLA fuses
  much of this, so it is an upper bound), `× #layers`, `× mus_number` for the
  disc-integrated `flux` (a `vmap` over μ). Plus the XLA executable, which for a
  network is itself larger than for the analytic sources. None of this scales
  with a "grid size" — it scales with the architecture and the chunk/μ counts.

**Contrast with the grid:** a zarr grid in `in_memory=True` mode pays its memory
*up front* as a flat data array (`∝ S·n_λ`), then each evaluation is a cheap
gather+blend whose transient is `accumulate_chunk_size·2^d·n_λ`. An aemu emulator
keeps only the weights resident (`∝` architecture, not data), but every
evaluation materialises a stack of dense activations (`∝ chunk_size·n_λ·H`). Pick
`in_memory=False` for the grid when the data array would not fit in (V)RAM — you
then pay per-call zarr I/O instead.

---

## Summary table

| Source | Resident (lifetime) | Transient per `simulate_observed_flux` call |
|---|---|---|
| Static spectrum | `3 · n_ref · dtype` (≈ tens of KB) | 2× `jnp.interp` over `n_λ` + `(n_λ,2)` output |
| Gaussian / blackbody | a few `n_lines`-length arrays (bytes) | ≈ `(2 + n_lines) · n_λ · 4 B`, ×`mus_number` for flux |
| Grid, `in_memory=False` | `S·12 B` (sparse index) + `n_λ·4 B` — negligible | zarr read+decompress of `2^d` bracketing rows per query (I/O-bound) + XLA buffers |
| Grid, `in_memory=True` | **`2·S·n_λ·4 B`** + sparse index + `n_λ·4 B` | `accumulate_chunk_size · 2^d · n_λ · 4 B` ×2 + Doppler grids `chunk_size·n_λ·4 B` + XLA |
| Grid, `sparse=False` | + dense index `∏|a_i|·4 B` (combinatorial — can OOM) | as above |
| aemu intensity bundle | network weights ≈ `L·H²·4 B` (+ small `model_state`) | activations ≈ `chunk_size·n_λ·H·4 B` ×`#layers` ×`mus_number` + XLA |
| aemu fixed-grid flux bundle | weights, dominated by `H·n_native·4 B` output layer (+ native λ grid) | one forward pass `≈ chunk_size·H·4 B` ×`#layers` + `jnp.interp` to `n_λ` + XLA |
| *(all sources)* | — | the JIT'd `simulate_observed_flux` executable + XLA buffer pools (tens of MB, once, then reused) + the mesh arrays (`∝ #surface elements`) |

## Reproducing

```bash
python benchmarks/benchmark_memory.py                 # static + gaussian + grid sweeps
python benchmarks/benchmark_memory.py --quick
python benchmarks/benchmark_memory.py --with-synthesis # also probe the per-call peak (ru_maxrss rise)
python benchmarks/benchmark_memory.py --sources aemu_harps aemu_small_random   # needs the [aemu] extra + a fetched bundle
```

The headline column is `resident_device_mb` (exact: sum of `.nbytes` over the
object's device arrays). `predicted_data_mb` is the closed-form `2·S·n_λ·4 B`
for the grid, or the weight-pytree size for an aemu bundle. `rss_after_construct_mb`
is the whole-process RSS (coarse — includes the interpreter, JAX, zarr) and
`synthesis_peak_increase_mb` is the rise in `ru_maxrss` caused by one synthesis
call (monotonic, so only the first call to reach a given peak shows it). Do not
read fine structure into the RSS columns; the device-bytes column is the one to
quote.
