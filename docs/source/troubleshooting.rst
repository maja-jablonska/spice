Troubleshooting
===============

Installation conflicts
----------------------

The current package requires Python >=3.11, JAX/JAXlib >=0.10, and
NumPy >=2.0,<3. A NumPy 1.26 pin inherited from an older environment cannot
satisfy those requirements. Follow :doc:`installation` to create a fresh
environment, install the checkout, and run ``python -m pip check``.

Precision and wavelength grids
------------------------------

SPICE follows JAX's precision setting. Enable float64 before constructing
meshes and wavelength arrays when calculations need it:

.. code-block:: python

    import jax
    jax.config.update('jax_enable_x64', True)

Alternatively, set ``JAX_ENABLE_X64=1`` before starting Python.

A synthesis kernel requires a finite, strictly increasing grid uniform in
log10 wavelength. Use ``jnp.linspace(log10(lo), log10(hi), n)`` rather than
``jnp.log10(jnp.linspace(lo, hi, n))``. Float32 rounding is accepted, but
duplicate wavelength samples are not. If the spacing cannot be represented
in float32, enable float64 and recreate the grid; casting existing rounded
values does not recover precision. Direct ``simulate_observed_flux`` does
not impose the kernel's uniform-grid requirement.

Unexpected magnitudes or luminosities
-------------------------------------

- Synthesis takes log10 wavelengths in Angstroms; photometry and
  ``luminosity`` take linear wavelengths in Angstroms.
- Photometry expects ``spectrum[:, 0]`` in erg/s/cm²/Å. It returns a
  magnitude, despite the historical ``*_passband_luminosity`` name.
- Use a true emergent surface flux for ``luminosity``. ``Blackbody.flux``
  returns intensity at ``mu=1``; the un-darkened blackbody example in
  :doc:`synthetic_photometry` supplies the required factor of pi.
- Check wavelength coverage: ``luminosity`` integrates only the range
  supplied, and photometry requires the full passband response.
- Results affected by the earlier per-Angstrom conversion errors should
  be recomputed. Remove any manual unit corrections that compensated for
  those bugs; see :doc:`conventions`.

Missing rotational broadening
-----------------------------

Looking along the rotation axis gives zero projected rotational velocity.
``get_mesh_view`` preserves the physical axis. To request an equator-on
default for a custom line of sight, set the view first, then call
``add_rotation`` without an explicit axis. Rotation velocities are populated
immediately. See :doc:`mesh` for both equator-on and pole-on examples.

Resource exhaustion
-------------------

For direct synthesis, ``chunk_size`` defaults to 256 surface elements and
``wavelengths_chunk_size=None`` selects ``min(n_wavelengths, 1024)``.
Reducing either value can lower peak memory use. Given an existing emulator,
mesh, and linear wavelength grid:

.. code-block:: python

    import jax.numpy as jnp
    from spice.spectrum import simulate_observed_flux

    flux = simulate_observed_flux(
        emu.intensity, model, jnp.log10(wavelengths),
        chunk_size=128, wavelengths_chunk_size=256,
    )

Chunk sizes are static arguments, so changing them triggers compilation.

GPU buffer comparator differences
---------------------------------

If XLA reports a buffer-comparator mismatch during GPU matrix-multiplication
autotuning, increasing matmul precision may help:

.. code-block:: python

    import jax
    jax.config.update('jax_default_matmul_precision', 'high')

This setting controls matrix multiplication precision; it does not enable
float64 array storage. Compare with CPU results if a discrepancy remains.
