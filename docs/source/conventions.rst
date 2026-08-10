Conventions and Units
=====================

SPICE mixes astronomical conventions from several traditions. This page is
the single source of truth for signs and units; every function docstring
should agree with it.

Geometry and line of sight
--------------------------

- The default line-of-sight vector is ``[0., 1., 0.]`` (the +Y axis) for a
  single :class:`~spice.models.MeshModel`; binary-orbit utilities default to
  ``[0., 0., -1.]``, the standard astronomical convention. Pass an explicit
  ``los_vector`` when combining the two.
- ``mesh.mus`` is the cosine of the angle between an element's surface normal
  and the line of sight; only elements with positive ``mus`` contribute to
  synthesized flux.
- Occlusion handling requires each component to be cast to the line of sight
  first via :func:`~spice.models.mesh_view.get_mesh_view`.

Velocities and Doppler shifts
-----------------------------

- ``mesh.los_velocities`` is in km/s with **negative = approaching**
  (blueshift), positive = receding (redshift).
- Synthesis samples the emitter's rest-frame spectrum at
  :math:`\lambda_{\mathrm{rest}} = \lambda_{\mathrm{obs}} / (1 + v/c)`, so a
  receding surface imprints its features redward of rest.
- The systemic velocity ``vgamma`` (km/s, positive = receding, matching
  PHOEBE's ``vgamma@binary``) is applied along the orbit's line of sight.

Time and periods
----------------

.. warning::

   Rotation and pulsation use **different time units**.

- :func:`~spice.models.mesh_transform.evaluate_rotation` interprets ``t`` in
  **seconds** (rotation velocity is km/s).
- :func:`~spice.models.mesh_transform.evaluate_pulsations` and the pulsation
  ``period`` are in **days**; the pulsation-velocity conversion to km/s
  assumes solRad/day.
- Orbital elements in :func:`~spice.models.binary.add_orbit` (``P``, ``T``,
  ``reference_time``) are in **years**; PHOEBE-imported meshes carry PHOEBE's
  native **days**.

Wavelengths and spectra
-----------------------

- All synthesis entry points take ``log10`` of the wavelength in Angstroms
  (``np.log10(wavelengths)``), not the wavelength itself.
- Flux is in erg/s/cm²/Å at the requested ``distance`` in parsecs
  (default 10 pc); intensities are erg/s/cm²/Å/sr.
- Synthesis results have shape ``(n_wavelengths, 2)``: the disc-integrated
  versions of the emulator's two channels — ``[flux, continuum]`` for the
  grid interpolators; :class:`~spice.spectrum.Blackbody` duplicates its
  intensity into both.

Stellar parameters
------------------

- Mesh radius and mass are in solar units (solRad, solMass).
- Surface gravity is :math:`\log_{10} g` in cgs (cm/s²). By default, any
  parameter named like ``logg`` (see ``LOG_G_NAMES``) is computed by SPICE
  per element from mass and per-element radius. Pass ``override_log_g=True``
  to keep explicitly supplied log g values instead; a warning is issued to
  flag that SPICE's computed values are being overridden.
- Effective temperature is linear Kelvin for :class:`~spice.spectrum.Blackbody`
  and the grid interpolators, but **log10 Kelvin** (``logteff``) for
  TPayne-style aemu bundles — check your emulator's ``parameter_names``.

Spots and pulsations
--------------------

- Spot centers (``spot_center_theta``, ``spot_center_phi``) are in
  **radians**; spot radii are in **degrees**; tilt angles are in **degrees**.
- Spherical-harmonic spots and pulsation modes are indexed by order ``m`` and
  degree ``l`` with :math:`m \le l`.
- Pulsation Fourier parameters have shape ``(3, N, 2)`` per mode — the first
  axis indexes the vector-spherical-harmonic components
  ``[radial, spheroidal, toroidal]``, and each innermost pair is
  ``[amplitude, phase]``. A 2-D ``(N, 2)`` input is interpreted as purely
  radial. Radial amplitudes are fractions of the stellar radius.

Photometric systems
-------------------

- :func:`~spice.spectrum.AB_passband_luminosity`,
  :func:`~spice.spectrum.ST_passband_luminosity`, and
  :func:`~spice.spectrum.Vega_passband_luminosity` return magnitudes in
  their respective systems from the same flux array (linear wavelengths
  here, in Angstroms).
- Gaia filters are photonic and **not supported for ST magnitudes**; see
  :doc:`synthetic_photometry` for the per-filter capabilities.

Precision
---------

- SPICE follows JAX's global precision: float32 by default, float64 after
  ``jax.config.update("jax_enable_x64", True)`` (set it before constructing
  meshes). Mixed float32 wavelengths with float64 mesh state are handled
  internally by casting the velocity term to the wavelength dtype.
