Conventions and Units
=====================

SPICE mixes astronomical conventions from several traditions. This page is
the single source of truth for signs and units; every function docstring
should agree with it.

Geometry and line of sight
--------------------------

- Native :class:`~spice.models.MeshModel` objects and binary-orbit utilities
  default to line of sight ``[0., 0., -1.]`` (observer → star). The default
  rotation axis is ``[0., 1., 0.]``, giving an equator-on view.
- PHOEBE models use ``[0., 0., +1.]`` for their velocity projection because
  the imported uvw velocities have a different sign convention. Their
  viewing cosines and projected geometry come directly from PHOEBE; retain
  this adapter convention to preserve PHOEBE's radial-velocity signs.
- :func:`~spice.models.mesh_view.get_mesh_view` changes the observer direction
  while preserving the rotation axis and velocity field. To choose an
  equator-on axis for a custom LOS, call it **before** ``add_rotation`` and
  leave ``rotation_axis`` unspecified. Viewing along an existing rotation
  axis gives a pole-on star with zero projected rotational velocity.
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
  **seconds** (rotation velocity is km/s) and is incremental (it rotates the
  current geometry).
- Rotation is clockwise seen from the tip of ``rotation_axis``
  (``v = (v_eq/R) r x a``): the angular momentum points along ``-rotation_axis``.
- ``los_vector`` must be a unit vector; ``mus`` and ``los_velocities`` scale
  with its length.
- :func:`~spice.models.mesh_transform.evaluate_pulsations` and the pulsation
  ``period`` are in **days**; the pulsation-velocity conversion to km/s
  assumes solRad/day.
- Without ``jax_enable_x64`` times are float32: a JD-sized epoch in years is
  rounded to ~0.15 d. Pass orbit times relative to a reference epoch near the
  observations (SPICE warns when the rounding exceeds 1e-4 of the period), or
  enable x64.
- Orbital elements in :func:`~spice.models.binary.add_orbit` (``P``, ``T``,
  ``reference_time``) are in **years**; PHOEBE-imported meshes carry PHOEBE's
  native **days**.
- ``mean_anomaly`` in :func:`~spice.models.binary.add_orbit` is an *additive
  phase offset on top of* the periastron timing (``M(t) = mean_anomaly +
  n·(t − T)``), not "the mean anomaly at the reference epoch". Set either
  ``mean_anomaly`` or ``T``, not both — PHOEBE-consistent values for both
  double-count the periastron phase. ``PhoebeBinary`` is unaffected (it uses
  PHOEBE's own precomputed orbit).

Wavelengths and spectra
-----------------------

- ``simulate_observed_flux`` and ``simulate_monochromatic_luminosity`` take
  ``log10`` of the wavelength in Angstroms.
  ``luminosity`` and the photometry functions instead take **linear**
  wavelengths in Angstroms.
- Flux is in erg/s/cm²/Å at the requested ``distance`` in parsecs
  (default 10 pc); intensities are erg/s/cm²/Å/sr.
- Synthesis results have shape ``(n_wavelengths, 2)``: the disc-integrated
  versions of the emulator's two channels — ``[flux, continuum]`` for the
  grid interpolators; :class:`~spice.spectrum.Blackbody` duplicates its
  intensity into both.

Stellar parameters
------------------

- Mesh radius and mass are in solar units (solRad, solMass).
- Mesh construction accepts a scalar for one parameter, a list or 1-D array
  shared by all faces, or a 2-D array with one row per face. Values are
  converted to the active floating-point dtype, so integer inputs do not
  truncate computed surface gravity.
- Surface gravity is :math:`\log_{10} g` in cgs (cm/s²). By default, any
  parameter named like ``logg`` (see ``LOG_G_NAMES``) is computed by SPICE
  per element from mass and per-element radius. Pass ``override_log_g=True``
  to keep explicitly supplied log g values instead; a warning is issued to
  flag that SPICE's computed values are being overridden.
- ``mesh.log_gs`` includes the centrifugal reduction from the current
  per-element rotation speeds. ``calculate_log_gs`` accepts those speeds
  in km/s and converts them to m/s for the acceleration calculation. Rotation
  and pulsation updates copy this gravity into the atmosphere parameter column
  unless the model was constructed with ``override_log_g=True``.
- Effective temperature is linear Kelvin (``teff``) for :class:`~spice.spectrum.Blackbody`,
  the grid interpolators and the current TPayne aemu bundles; some older bundles used
  log10 Kelvin (``logteff``). Check your emulator's ``stellar_parameter_names``.
- ``to_parameters`` accepts ``teff`` and ``Teff`` interchangeably and raises
  ``ValueError`` for any name the emulator does not have, instead of silently
  leaving that parameter at its default.

Spots and pulsations
--------------------

- Spot centers (``spot_center_theta``, ``spot_center_phi``) are in
  **radians**; spot radii are in **degrees**; tilt angles are in **degrees**.
  The angular boundary is converted to a chord length before distance
  comparisons. Zero-radius spots are empty; radii of 180 degrees or more
  cover the whole surface. ``add_spots`` defaults to the same zero smoothness
  as ``add_spot``. Harmonic spot patterns use the reference surface relative
  to the star, so translating a mesh does not change its pattern.
- Spot colatitudes/longitudes, spherical-harmonic spot patterns and pulsation
  modes are all measured in the **rotation-axis frame**: the pole is the
  mesh's ``rotation_axis`` (:func:`~spice.models.utils.rotation_axis_frame`),
  read when the spot is added or the pulsations are evaluated, so add rotation
  first. For an explicit ``[0., 0., 1.]`` axis this frame is the identity.
  :func:`~spice.models.utils.inclination_to_los_axis` takes the same axis
  (``inclination_to_los_axis(i, mesh.rotation_axis)``).
- Spherical-harmonic spots and pulsation modes are indexed by order ``m`` and
  degree ``l`` with :math:`0 \le m \le l \le 10`. They are real harmonics, so
  ``m != 0`` pulsation modes are standing waves.
- ``evaluate_rotation`` turns the pulsation displacements and velocities with
  the surface, so ``evaluate_rotation(evaluate_pulsations(m, t), t)`` keeps the
  mode attached to the star (``t`` is in days for pulsations, seconds for
  rotation).
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
  their respective systems from a 1-D flux density in erg/s/cm²/Å
  (``spectrum[:, 0]``) and linear wavelengths in Angstroms. Pass the
  synthesis output directly, without converting it to flux per centimetre.
- Gaia filters are photonic and **not supported for ST magnitudes**; see
  :doc:`synthetic_photometry` for the per-filter capabilities.

Precision
---------

- SPICE follows JAX's global precision: float32 by default, float64 after
  ``jax.config.update("jax_enable_x64", True)`` (set it before constructing
  meshes). Mixed float32 wavelengths with float64 mesh state are handled
  internally by casting the velocity term to the wavelength dtype.

Updating existing calculations
------------------------------

The corrections on this branch change absolute predictions. Recompute
stored results with the corrected functions:

- The per-Angstrom photometry correction removes a +20 magnitude offset
  from affected AB, ST, and Vega paths, including Gaia. Energy-response
  weighting was corrected as well, so recomputation is preferable to
  applying a single offset to every stored magnitude.
- ``luminosity`` now integrates erg/s/Å over Angstroms. For the same input
  surface-flux spectrum and mesh, it is a factor of :math:`10^8` larger
  than the previous result; the corresponding bolometric magnitude is
  20 magnitudes smaller. Check the emulator's surface-flux convention too
  (:doc:`spectral_synthesis`).
- Changing the observer direction preserves the rotation axis. Code that
  relies on an equator-on default should set the view before adding rotation
  (:doc:`mesh`).
- Eclipses process every target face, including meshes with more than 60%
  of their faces visible. Recheck eclipse curves from uneven tessellations.
- The centrifugal gravity reduction now uses consistent velocity units;
  rebuild or re-evaluate previously saved rotating meshes. Automatically
  managed atmosphere gravity now follows rotation and pulsation changes.
