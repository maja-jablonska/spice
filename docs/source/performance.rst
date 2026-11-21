Performance and GPU
===================

SPICE's cost is set by three numbers: the surface elements of the mesh, the
wavelengths of the grid, and the cost of one emulator call. This page covers
what JAX compiles and when, how to set the chunk sizes, precision, GPU set-up,
and how to batch time series.

Compilation
-----------

The first call of :func:`~spice.spectrum.simulate_observed_flux` (and of the
orbit and luminosity functions) traces and compiles the computation; later
calls with the same array shapes and static arguments reuse the compiled
code. A new compilation is triggered by:

- a different number of mesh elements or wavelengths,
- a different ``chunk_size`` or ``wavelengths_chunk_size`` (static arguments),
- a different intensity function or limb-darkening configuration,
- a change of precision (``jax_enable_x64``).

Keep the mesh resolution, wavelength grid and chunk sizes fixed inside loops
and fits.

Chunk sizes
-----------

``simulate_observed_flux`` evaluates the emulator over ``chunk_size`` surface
elements at a time (default 256) and ``wavelengths_chunk_size`` wavelengths at a
time (default ``min(n_wavelengths, 1024)``), accumulating the chunks with
``jax.checkpoint`` so memory stays bounded. Smaller chunks use less memory;
larger chunks widen the vectorised emulator call. Measured on a laptop CPU
(Apple M2, 1280-element mesh; first call includes compilation, then the
median of five warm calls):

.. list-table::
   :header-rows: 1

   * - ``chunk_size``
     - Blackbody, 1000 wavelengths, float32
     - Blackbody, 1000 wavelengths, float64
     - paper emulator (aemu v1.2.0), 200 wavelengths, float64
   * - 64
     - 0.23 s, then 5.4 ms
     - 0.23 s, then 10.0 ms
     - 2.8 s, then 2.30 s
   * - 256 (default)
     - 0.19 s, then 4.4 ms
     - 0.20 s, then 10.2 ms
     - 3.0 s, then 3.26 s
   * - 1024
     - 0.19 s, then 6.5 ms
     - 0.18 s, then 14.4 ms
     - 4.9 s, then 5.55 s

For the analytic emulators the default is close to the best choice. For the
neural emulator on a CPU, *smaller* chunks are faster: 2.3 s at
``chunk_size=64`` against 5.6 s at 1024 for 1280 elements × 200 wavelengths
(about 9 µs per element and wavelength at the best setting). On a GPU, larger
chunks keep the device busy; raise ``chunk_size`` until memory or compile time
limits it, and time a few values for your mesh and grid.

Precision
---------

SPICE follows JAX's precision setting (see :doc:`troubleshooting`). The
TransformerPayne aemu bundles need 64-bit mode (``JAX_ENABLE_X64=1``) for
their wavelength handling; the analytic emulators work in either. On a CPU,
float64 costs little; on consumer GPUs float64 throughput is a small fraction
of float32, while data-centre GPUs (V100, A100, H100) run it at half rate.

On the laptop CPU above, float64 roughly doubles the warm Blackbody
synthesis time (4.4 ms → 10.2 ms at the default chunk size).

GPU
---

SPICE runs on a GPU whenever JAX sees one. Install a CUDA-enabled JAX
(``pip install -U "jax[cuda12]"``; see the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_),
then check with:

.. code-block:: python

    import jax
    print(jax.devices())      # [CudaDevice(id=0)] when the GPU is used

On macOS SPICE sets ``JAX_PLATFORMS=cpu`` unless you set it yourself, because
the Metal backend is slow for this workload.

By default JAX reserves 75 % of the GPU memory on first use. To share a GPU or
to see the real usage, set before importing JAX:

.. code-block:: bash

    export XLA_PYTHON_CLIENT_PREALLOCATE=false     # allocate on demand
    # or cap the reservation:
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

(see `GPU memory allocation <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>`_).
Out-of-memory errors in synthesis are fixed by smaller ``chunk_size`` or
``wavelengths_chunk_size``; in gradients, by forward-mode differentiation
(see :doc:`gradients`).

The emulator of the SPICE paper (``mjablonska/TPayne-spice-harps-interp``
v1.2.0) on data-centre GPUs, measured for the paper with x64,
``chunk_size=1024`` and ``wavelengths_chunk_size=min(n_wavelengths, 1024)``
(mean time per warm ``simulate_observed_flux`` call):

.. list-table::
   :header-rows: 1

   * - Configuration
     - V100-SXM2-32GB
     - A100-PCIE-40GB
   * - 5,120 elements, 200 wavelengths
     - 0.30 s
     - 0.10 s
   * - 5,120 elements, 1,000 wavelengths
     - 1.52 s
     - 0.48 s
   * - 5,120 elements, 4,000 wavelengths
     - 6.17 s
     - 1.95 s
   * - 1,280 elements, 1,024 wavelengths
     - 0.52 s
     - 0.16 s
   * - 20,480 elements, 1,024 wavelengths
     - 5.15 s
     - 1.62 s

The cost grows linearly with the number of elements times wavelengths: the
emulator is evaluated once per element and wavelength. For the analytic
emulators the same synthesis takes milliseconds.

Time series
-----------

Evaluate all epochs first and reuse one wavelength grid:
:func:`~spice.models.mesh_transform.evaluate_rotation_at_times`,
:func:`~spice.models.binary.evaluate_orbit_at_times` and
:func:`~spice.models.binary.evaluate_orbit_at_times_stacked` vectorise over
times. The synthesis itself can be a Python loop over the meshes (one
compilation, reused for every epoch) or a :func:`jax.vmap` over time, which
compiles one larger batched computation:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from spice.models.mesh_transform import evaluate_rotation

    times = jnp.linspace(0., 86400., 20)   # seconds
    spectra = jax.vmap(
        lambda t: simulate_observed_flux(emu.intensity, evaluate_rotation(star, t), log_wl)
    )(times)                                # (20, n_wavelengths, 2)

Both give identical spectra. On the laptop CPU (Blackbody, 1280 elements,
1000 wavelengths, 20 epochs), the loop is faster: 0.10 s against 0.39 s warm in
float32, 0.23 s against 0.44 s in float64, because the CPU is already busy
inside each call. The batched form can pay off on a GPU, where a single call
does not fill the device; it needs memory for all epochs at once.

Mesh resolution
---------------

The number of faces sets both cost and accuracy. The refinement levels and
their face counts are listed in :doc:`mesh`; how many are needed depends on
the problem:

- rotational broadening and line-profile variations: :doc:`spectral_synthesis`;
- luminosities and magnitudes: "Luminosity and mesh convergence" in
  :doc:`synthetic_photometry`;
- eclipses: "Occlusion and mesh resolution" in :doc:`binaries`.
