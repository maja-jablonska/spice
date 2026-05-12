# Cost breakdown: static spectrum and aemu (astro_emulators_toolkit) emulators

Companion to `grid_loading_cost_breakdown.md`. For each intensity source there
are two separate questions:

* **(A) construction / loading** — the one-off cost of building the emulator
  object;
* **(B) evaluation** — the cost of one `intensity(...)` / `flux(...)` call, which
  `simulate_observed_flux` invokes per surface element (vmapped and chunked).

Line references are to files under `src/spice/spectrum/`.

## 0. Shared machinery (common to every intensity source)

`simulate_observed_flux` (`spectrum.py`) does the same thing regardless of the
source: `jax.vmap` the `intensity_fn` over mesh elements, chunk over mesh
elements (`chunk_size`) under `jax.checkpoint`, optionally chunk over wavelengths
(`wavelengths_chunk_size`), Doppler-shift the wavelength grid per element, weight
by projected area / visibility, and accumulate with `lax.scan`. The **first**
call also traces and JIT-compiles this whole pipeline (with the source's
evaluation code inlined) — a one-time cost of order seconds that grows with the
chunk counts and the source's complexity; later calls hit the compiled
executable.

Everything below is *on top of* this shared cost. The `static` source in
`benchmark_spice_components.py` exists precisely to measure the shared cost in
isolation.

## 1. Static spectrum (`StaticSpectrum` — the `static` benchmark source)

### (A) Construction
`jnp.log10(jnp.asarray(ref_wavelengths))`, `jnp.asarray(ref_flux)`,
`jnp.asarray(ref_continuum)` — three small host→device copies of a 1-D reference
spectrum, O(n_ref). Sub-millisecond. **There is nothing to "load"** — no files,
no network, no weights.

### (B) Evaluation
`intensity(log_wavelengths, mu, parameters)` is two
`jnp.interp(log_wavelengths, ref_logw, ref_*)` calls plus a `jnp.stack` — O(n_ref
+ n_wavelengths). It does **not** depend on the stellar parameters or `mu`: it
returns the same curve resampled onto the requested grid. So the marginal
per-call cost over the shared machinery is essentially just the two interpolations
(∝ #wavelengths) — which is why this source is the "floor" in the component
benchmark.

**Takeaway:** static ≈ zero load cost, ≈ zero marginal evaluation cost on top of
the shared mesh / Doppler / scan machinery.

## 2. Parametric emulators (Gaussian lines, blackbody) — brief aside

* **(A) Construction:** store a few `jnp.array`s (line centres / widths / depths,
  or nothing for a pure blackbody). Trivial.
* **(B) Evaluation:** a Planck continuum (one `exp`, elementwise over
  #wavelengths) plus one `exp` per line × #lines and a product — ∝ #wavelengths ×
  (1 + #lines). `flux(...)` runs `mus_number` (default 20) intensity evaluations
  via `jax.vmap` over `mu` and a Gauss–Legendre weighted sum → ≈20×, even though
  the blackbody continuum is `mu`-independent. Still cheap; no load cost.

## 3. aemu emulators (`PretrainedAemuSpectrumEmulator` and subclasses)

These wrap `aemu.Emulator` from the optional `astro_emulators_toolkit`
(`pip install "stellar-spice[aemu]"`).

### (A) Construction / loading — this is the expensive part

1. **`import astro_emulators_toolkit`** — module-level in
   `aemu_spectrum_emulator.py` (line ~40), so it runs the first time anything in
   that module is touched (`spice.spectrum` lazy-imports it). It pulls in
   `flax.nnx` and JAX internals: one-time per process, hundreds of ms to a
   couple of seconds. (Raises a clear error if the extra is missing or there's a
   flax/JAX version mismatch.)
2. **`_load_emulator(name)`** (lines ~43–57):
   * `name` is a local bundle directory → `aemu.Emulator.from_bundle(name)`:
     reads the bundle files (flax `graphdef`, `params`, `spec`/metadata,
     `bundle_extras`) from disk and reconstructs the model. Disk-I/O +
     deserialization bound, ∝ serialized weight size.
   * `name` is a Hugging Face repo id → `aemu.Emulator.from_pretrained(name)`:
     **on first use this downloads the whole bundle from the HF Hub over the
     network** into `~/.cache/huggingface/hub`, then does the same load. First
     call: network-bound — seconds to minutes depending on bundle size and
     connection; every later call (any process on the machine) is served from the
     local cache, i.e. just the disk load. This is why the `aemu_*` sources are
     not in the component benchmark's default `--sources` set.

   The weight size — hence this step's cost — is set by the network architecture
   (number / width of hidden layers and the output dimension). For a
   **fixed-grid flux bundle** the final layer is `hidden × n_native_wavelengths`,
   which dominates and can be large (a full spectrum of outputs per hidden unit).
   For a **wavelength-conditioned intensity bundle** the wavelength is an input
   feature and the output is only 2 channels, so the net is comparatively compact.
3. **`AemuSpectrumEmulator.__init__`** (lines ~156–183):
   * `_affine_reference_scaling(emulator, "inputs")` — reads small metadata dicts
     (min/max trees). ≈ free.
   * `_build_frozen_apply(emulator)` → `emulator._ensure_task()` +
     `_make_frozen_apply_runtime(graphdef, params, model_state, post_fn, jit=False)`
     — builds a closure that merges the flax `graphdef` with `params` /
     `model_state` into a plain callable usable under `jax.jit` / `vmap`. With
     `jit=False` there is no compilation here; the cost is pytree bookkeeping plus
     getting the parameter arrays onto the device. ∝ weight size.
4. **Subclass extras:**
   * `IntensityPretrainedAemuSpectrumEmulator` (lines ~389–410): resolves the
     *output* reference scaling, the `mu` channel index, the output leaf name —
     all small metadata, ≈ free; raises if the bundle lacks input+output scaling
     or a `mu` channel.
   * `FluxPretrainedAemuSpectrumEmulator` (lines ~304–318): infers the native
     wavelength grid from `bundle_extras['wavelength_angstrom']` (or numeric
     channel names) → `jnp.asarray` of a length-`n_native` vector. Tiny.

**So aemu load cost ≈ (one-time `import astro_emulators_toolkit`) + (first-time
HF download, if `name` is a repo id) + (deserialize + `device_put` the network
weights). The first two are paid once per process / once per machine; the third
scales with the model size.**

### (B) Evaluation — the runtime cost

`IntensityPretrainedAemuSpectrumEmulator.intensity(log_wavelengths, mu, parameters)`
(called per mesh element by `simulate_observed_flux`, vmapped & chunked):

1. `_splice_mu` — concatenate `mu` into the parameter vector at the bundle's
   channel position. Trivial.
2. `_intensity_at_mu` (lines ~423–445):
   * assemble `{"parameters": (1, n_p), "wavelengths": (1, n_w)}` and
     `aemu.normalize_tree` — affine min-max scaling, elementwise, cheap;
   * **`self._apply(x_scaled)` — the forward pass through the neural network.**
     This dominates the per-call cost. It scales with the network architecture
     (≈ hidden_width² × #layers) and with the input shape — since these bundles
     are wavelength-conditioned, the `wavelengths` array of length `n_w` is part
     of the input, so the work grows with the number of requested wavelengths
     (the net is effectively evaluated across the `n_w` wavelength positions,
     ≈ `n_w × hidden_width² × #layers` MACs per call);
   * `aemu.denormalize_tree` — affine, cheap; `10**y` — elementwise.
3. `flux(...)` (disc-integrated, lines ~460–482) = `mus_number` (default 20)
   `intensity` evaluations via `jax.vmap` over `mu` plus a Gauss–Legendre
   weighted sum → ≈20× the `intensity` cost. (`IntensityLazyZarrInterpolator` and
   `GaussianLineEmulator.flux` use the same 20-point quadrature.)
4. `FluxPretrainedAemuSpectrumEmulator` variant (lines ~320–338): one network
   evaluation producing the full `n_native` spectrum (input is `parameters` only,
   no wavelength conditioning), then `jnp.interp` to resample onto the requested
   wavelengths — cost ∝ model size + #wavelengths, with no wavelength dependence
   inside the network; the disc-integrated flux comes out directly.

Plus, as in §0, the first `simulate_observed_flux` call JIT-compiles the whole
pipeline with `_apply` inlined into the mesh-chunk scan — a one-time cost of order
seconds.

**Takeaway:** two distinct, both-real costs — **construction** dominated by the
one-time toolkit import and (on first use) the HF download, thereafter ∝ weight
size; **evaluation** a dense neural-network forward pass per mesh element, ∝
network size × #wavelengths, ×≈20 for disc-integrated flux, plus a one-time JIT
compile. Contrast the zarr grid: there the *load* moves the whole data array onto
the device up front (∝ grid bytes) and each evaluation is then a cheap
multilinear gather+blend; with aemu the load is "weights only" but every
evaluation pays a dense matmul.

## Where the time goes — summary table

| Source | Load / construct cost | Per-evaluation cost (per mesh element) | Disc-flux multiplier |
|---|---|---|---|
| Static spectrum | ≈ 0 (a few small `device_put`s of a 1-D reference curve) | 2× `jnp.interp` over #wavelengths; independent of params and μ | n/a (μ-independent) |
| Gaussian lines / blackbody | ≈ 0 (a few `jnp.array`s) | Planck `exp` + one `exp` per line, elementwise over #wavelengths | ×`mus_number` (≈ 20) via vmap over μ |
| zarr grid, `in_memory=False` | parse `index.parquet` + build sparse grid index (∝ #nodes) | zarr read + decompress of the bracketing rows, then multilinear blend | ×Gauss–Legendre μ-quadrature inside the interpolator |
| zarr grid, `in_memory=True` | the above **+ `device_put` of the entire flux/continuum array (∝ #nodes × #wavelengths)** | multilinear gather + blend on device (cheap) | ″ |
| aemu intensity bundle | `import astro_emulators_toolkit` (once/process) + HF download on first use + deserialize/`device_put` weights (∝ model size) | one NN forward pass, ∝ network size × #wavelengths | ×`mus_number` (≈ 20) via vmap over μ |
| aemu fixed-grid flux bundle | ″ (+ load the native λ grid) | one NN forward pass (∝ model size, incl. the `hidden × n_native` last layer) + `jnp.interp` to #wavelengths | computed directly as flux |

And for **every** source: the first `simulate_observed_flux` call JIT-compiles
the full pipeline (mesh-element chunking, Doppler shift, area weighting,
`lax.scan` accumulation) — a one-time cost shared by all, and exactly what the
`static` benchmark source is designed to isolate.

## Reproducing the numbers

Per-evaluation / synthesis timings: `benchmarks/benchmark_spice_components.py`
(`--sources static gaussian linear_grid aemu_harps aemu_small_random`). Grid
*load* timings: `benchmarks/benchmark_grid_loading.py`. There is no dedicated
load-time benchmark for the aemu bundles because the dominant term — the
Hugging Face download — is network/cache dependent rather than a property of
the code.
