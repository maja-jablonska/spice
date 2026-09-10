Spectral Synthesis
==================

:func:`~spice.spectrum.simulate_observed_flux` is SPICE's main entry point:
it turns a mesh model plus an emulator's intensity function into the
disc-integrated spectrum an observer would record.

.. note::

   Every code snippet on this page has a matching section in the companion
   notebook
   `tutorial/docs_examples/spectral_synthesis_examples.ipynb <https://github.com/maja-jablonska/spice/blob/main/tutorial/docs_examples/spectral_synthesis_examples.ipynb>`_.

How it works
------------

For every visible mesh element, SPICE:

1. evaluates the emulator's intensity at the element's parameters and viewing
   angle ``mu``,
2. Doppler-samples the rest-frame spectrum according to the element's
   line-of-sight velocity (rotation + pulsation + orbit),
3. weights by the element's visible, projected area, and
4. sums the contributions, scaled by :math:`1/d^2` for the distance.

The loop is chunked with ``jax.checkpoint`` so memory stays bounded for large
meshes and wavelength grids, and the whole computation is JIT-compiled.

Basic usage
-----------

.. code-block:: python

    import numpy as np
    from spice.models import IcosphereModel
    from spice.spectrum import simulate_observed_flux, Blackbody

    bb = Blackbody()
    star = IcosphereModel.construct(1000, 1., 1., bb.solar_parameters, bb.parameter_names)

    wavelengths = np.linspace(4000., 7000., 2000)   # Angstroms
    flux = simulate_observed_flux(bb.intensity, star, np.log10(wavelengths))

.. important::

   The wavelength argument is ``log10`` of the wavelength in Angstroms.
   Passing linear wavelengths is the most common mistake and produces
   silently wrong results.

The result has shape ``(n_wavelengths, 2)``: the disc-integrated versions of
the emulator's two output channels. For the grid interpolators these are
``[flux, continuum]`` — so ``flux[:, 0] / flux[:, 1]`` is the normalized
spectrum; :class:`~spice.spectrum.Blackbody` duplicates its intensity into
both columns. Units are erg/s/cm²/Å at the requested distance.

Parameters that matter
----------------------

``distance`` (float, parsecs, default 10.0)
    The flux scales as :math:`(R/d)^2`. The default 10 pc makes
    ``AB_passband_luminosity`` of the result an absolute magnitude.

``disable_doppler_shift`` (bool, default False)
    Skips the per-element Doppler sampling — useful for isolating geometric
    effects or speeding up static models.

``chunk_size`` / ``wavelengths_chunk_size`` (int, default 1024)
    Trade memory for speed. Larger chunks help on GPU; reduce them if you
    hit out-of-memory errors on large meshes or dense wavelength grids.

``ld_law`` / ``ld_coeffs``
    Bind a limb-darkening law into the intensity function (e.g.
    ``ld_law="linear"``, ``ld_coeffs=[0.5]``). Only effective for intensity
    functions that accept these keywords, such as the flux-emulator-based
    implementations; :class:`~spice.spectrum.Blackbody` takes its LD
    configuration in its constructor instead.

Doppler shifts
--------------

Each element's ``los_velocity`` (negative = approaching, see
:doc:`conventions`) shifts its contribution: the rest-frame spectrum is
sampled at :math:`\lambda_{\mathrm{obs}} / (1 + v/c)`, so approaching
material imprints blueshifted features and a rotating star shows the
familiar line broadening:

.. code-block:: python

    from spice.models.mesh_transform import add_rotation, evaluate_rotation

    fast = evaluate_rotation(add_rotation(star, rotation_velocity=50.), 0.)
    flux_rot = simulate_observed_flux(bb.intensity, fast, np.log10(wavelengths))

Luminosities
------------

Three companions integrate over the whole stellar surface instead of the
visible disc:

.. code-block:: python

    from spice.spectrum import simulate_monochromatic_luminosity, luminosity, absolute_bol_luminosity

    # L_lambda(lambda): (n_wavelengths, 2), erg/s/Angstrom
    mono_lum = simulate_monochromatic_luminosity(bb.flux, star, np.log10(wavelengths))

    # Bolometric luminosity in erg/s (trapezoidal integral over wavelength)
    L = luminosity(bb.flux, star, wavelengths)

    # Absolute bolometric magnitude (IAU zero point)
    M_bol = absolute_bol_luminosity(L)

These take the emulator's ``flux`` function (no ``mu`` dependence), not
``intensity``.

.. note::

   The result is physically meaningful only when the emulator's ``flux``
   channel is a true surface flux. :meth:`Blackbody.flux
   <spice.spectrum.Blackbody.flux>` returns the :math:`\mu = 1` intensity
   (see its docstring), so blackbody "luminosities" are useful for
   resolution-convergence comparisons but are not absolute physical values —
   compare the theoretical Stefan–Boltzmann calculation in
   :doc:`synthetic_photometry`.

Performance notes
-----------------

- The first call compiles; subsequent calls with the same shapes reuse the
  compiled kernel. Keep wavelength-grid and mesh sizes fixed inside loops.
- Chunk sizes are static arguments — changing them recompiles.
- For time series, prefer evaluating meshes first (e.g. with
  ``evaluate_rotation_at_times`` / ``evaluate_orbit_at_times``) and reusing
  one wavelength grid across all synthesis calls.
