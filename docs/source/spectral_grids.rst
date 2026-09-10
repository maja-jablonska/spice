Spectral Grids
==============

For realistic spectra, SPICE interpolates precomputed synthetic-spectra grids
stored as `zarr <https://zarr.dev>`_ arrays. The interpolators load lazily,
so grids much larger than memory can be queried efficiently, and lookups are
JAX-compatible (jit/vmap-able) for use inside
:func:`~spice.spectrum.simulate_observed_flux`.

Install the required dependencies with the ``[grid]`` extra:

.. code-block:: bash

    pip install "stellar-spice[grid]"

.. note::

   Every code snippet on this page has a matching section in the companion
   notebook
   `tutorial/docs_examples/spectral_grids_examples.ipynb <https://github.com/maja-jablonska/spice/blob/main/tutorial/docs_examples/spectral_grids_examples.ipynb>`_,
   which builds a tiny synthetic grid so it runs without downloading data.

Grid layout
-----------

A grid is a zarr store with (at least) these arrays:

- ``wavelength`` — shared wavelength sampling, shape ``(L,)``
- ``flux`` and ``continuum`` — one row per grid node, shape ``(N, L)``
- ``params`` — node parameters, shape ``(N, P)``
- ``param_names`` — the ``P`` parameter labels (e.g. ``teff``, ``logg``, ``mu``)

plus an ``index.parquet`` file that maps parameter combinations to row
indices. Generate it once per grid:

.. code-block:: bash

    python -m spice.spectrum.generate_zarr_index /path/to/grid.zarr
    # add --exclude-mu for grids without a mu axis

Loading and querying
--------------------

.. code-block:: python

    import jax.numpy as jnp
    from spice.spectrum import FluxLazyZarrInterpolator

    interp = FluxLazyZarrInterpolator("/path/to/grid.zarr", params=["teff", "logg"])

    # Multilinear interpolation across all axes at arbitrary points.
    # The result has shape (n_wavelengths, 2): [flux, continuum].
    log_wavelengths = jnp.log10(jnp.linspace(5000., 5010., 200))
    spec = interp.flux(log_wavelengths, interp.to_parameters({"teff": 5777., "logg": 4.44}))

Key constructor options:

``sparse`` (default True)
    Uses a hash-based :class:`~spice.spectrum.SparseGridIndex` that supports
    irregular (non-Cartesian-product) grids. ``sparse=False`` builds a dense
    N-d index array — faster lookups, but memory grows with the product of
    axis lengths.

``in_memory`` (default False)
    ``True`` moves the flux/continuum rows onto the JAX device up front;
    ``"auto"`` does so only if they fit under ``in_memory_threshold_bytes``
    (2 GiB by default). Keep the default for grids larger than memory.

``accumulate_chunk_size`` (default 64)
    Queries processed in parallel per chunk on the in-memory path; raise it
    to better utilize a GPU at the cost of peak memory.

Intensity and flux variants
---------------------------

Two subclasses adapt a grid to the two synthesis interfaces:

:class:`~spice.spectrum.IntensityLazyZarrInterpolator`
    For grids with a ``mu`` axis (specific intensities). Its ``intensity``
    method plugs directly into
    :func:`~spice.spectrum.simulate_observed_flux`; ``flux`` integrates over
    ``mu`` with Gauss–Legendre quadrature.

:class:`~spice.spectrum.FluxLazyZarrInterpolator`
    For flux-only grids. Its ``intensity`` derives angle dependence from a
    flux-conserving limb-darkening law (``ld_law``/``ld_coeffs`` keywords,
    linear by default).

.. code-block:: python

    from spice.spectrum import IntensityLazyZarrInterpolator, simulate_observed_flux

    interp = IntensityLazyZarrInterpolator("/path/to/grid.zarr",
                                           params=["teff", "logg", "mu"])
    flux = simulate_observed_flux(interp.intensity, mesh, log_wavelengths)

Mesh parameters must be ordered like the interpolator's stellar parameters —
construct meshes with ``parameter_names=interp.stellar_parameter_names`` (or
``interp.parameter_names`` without ``mu``) and use ``interp.to_parameters``
for single queries.

Geometry provenance
-------------------

Model-atmosphere grids often mix radiative-transfer geometries — MARCS-style
grids compute giants (low ``logg``) in spherical symmetry and dwarfs
plane-parallel. The index can record this per node:

.. code-block:: bash

    # Preferred: the store carries a `geometry` array written by the producer
    python -m spice.spectrum.generate_zarr_index /path/to/grid.zarr

    # Fallback heuristic: label nodes with logg < 3 as spherical
    python -m spice.spectrum.generate_zarr_index /path/to/grid.zarr --geometry-from-logg 3.0

The label is provenance only — it is never an interpolation axis — and is
exposed as ``interp.geometry`` (a per-row array of ``"spherical"`` /
``"plane_parallel"``, or ``None`` for indices that predate the feature) so
you can detect queries that straddle the geometry boundary, where
interpolation mixes the two treatments.
