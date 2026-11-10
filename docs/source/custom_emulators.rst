Custom Emulators
================

Any spectrum model can drive SPICE's synthesis: the mesh machinery only needs
a function mapping ``(log_wavelengths, mu, parameters)`` to per-element
intensities. The :class:`~spice.spectrum.SpectrumEmulator` interface
formalizes this so your model also works with parameter helpers and the
luminosity functions.

.. note::

   Every code snippet on this page has a matching section in the companion notebook
   `tutorial/docs_examples/custom_emulators_examples.ipynb <https://github.com/maja-jablonska/spice/blob/main/tutorial/docs_examples/custom_emulators_examples.ipynb>`_.

The interface
-------------

.. code-block:: python

    from spice.spectrum import SpectrumEmulator

Two members are required:

``stellar_parameter_names`` (property)
    The ordered labels of the stellar parameters your model expects — this
    is what mesh construction matches against (and how ``logg`` overriding
    finds its column; see :doc:`conventions`).

``to_parameters(parameters)``
    Converts a ``{name: value}`` mapping (or ``None`` for solar defaults)
    into the model's parameter vector.

Two are optional capabilities, raising ``NotImplementedError`` by default:

``intensity(log_wavelengths, mu, parameters)``
    Specific intensity in erg/s/cm²/Å/sr for one surface element at
    viewing angle ``mu``.
    Must return shape ``(n_wavelengths, 2)`` — by convention
    ``[flux-like, continuum-like]`` channels; duplicate your intensity if
    you have no continuum. This is what
    :func:`~spice.spectrum.simulate_observed_flux` consumes.

``flux(log_wavelengths, parameters)``
    Emergent surface flux in erg/s/cm²/Å, shape ``(n_wavelengths, 2)``,
    used by the luminosity functions. For an azimuthally symmetric
    intensity, this is :math:`F_\lambda = 2\pi\int_0^1 I_\lambda(\mu)\mu\,d\mu`.
    Do not pass an observed flux diluted by distance or an unintegrated
    intensity as a surface-flux function.

Requirements on the implementation:

- It must be **JAX-traceable**: use ``jax.numpy``, no Python branching on
  array values — SPICE jits and vmaps it over mesh elements.
- Wavelengths arrive as **log10 of Angstroms**.
- Parameters arrive as a plain array in ``stellar_parameter_names`` order
  (one row per mesh element).

A worked example
----------------

A Gaussian absorption line on a blackbody continuum, with linear limb
darkening and a temperature-dependent line depth:

.. code-block:: python

    import jax.numpy as jnp
    from spice.spectrum import SpectrumEmulator
    from spice.spectrum.blackbody import blackbody_intensity

    class ToyLineEmulator(SpectrumEmulator):
        def __init__(self, line_center=5500.0, width=0.5, ld_coeff=0.6):
            self.line_center = line_center
            self.width = width
            self.ld_coeff = ld_coeff

        @property
        def stellar_parameter_names(self):
            return ["teff"]

        def to_parameters(self, parameters=None):
            parameters = parameters or {}
            return jnp.array([parameters.get("teff", 5777.0)])

        def intensity(self, log_wavelengths, mu, parameters):
            wavelengths = jnp.power(10.0, log_wavelengths)
            continuum = blackbody_intensity(log_wavelengths, mu, parameters)[:, 0]
            # linear limb darkening
            continuum = continuum * (1.0 - self.ld_coeff * (1.0 - mu))
            # deeper line for cooler atmospheres
            depth = jnp.clip(0.8 * 5777.0 / parameters[0], 0.0, 0.95)
            line = 1.0 - depth * jnp.exp(-0.5 * ((wavelengths - self.line_center) / self.width) ** 2)
            return jnp.stack([continuum * line, continuum], axis=-1)

Use it exactly like the built-in emulators:

.. code-block:: python

    import numpy as np
    from spice.models import IcosphereModel
    from spice.spectrum import simulate_observed_flux

    emulator = ToyLineEmulator()
    star = IcosphereModel.construct(1000, 1., 1.,
                                    emulator.to_parameters({"teff": 6000.}),
                                    emulator.stellar_parameter_names)

    wavelengths = np.linspace(5490., 5510., 800)
    flux = simulate_observed_flux(emulator.intensity, star, np.log10(wavelengths))

Because the intensity is JAX-traceable, everything composes: add rotation
and the line broadens; add a temperature spot and the line strength becomes
rotation-phase-dependent.

Built-in implementations to crib from
-------------------------------------

- :class:`~spice.spectrum.Blackbody` — minimal single-parameter emulator
  with optional limb darkening (``blackbody.py``).
- :class:`~spice.spectrum.GaussianLineEmulator` /
  :class:`~spice.spectrum.PhysicalLineEmulator` — analytic line profiles
  with configurable centers, widths, and depths.
- :class:`~spice.spectrum.UserSpectrumInterpolator` — interpolates a
  user-supplied spectrum table.
- The :doc:`grid interpolators <spectral_grids>` — the full-featured
  reference, including ``mu`` handling and lazy loading.
- The aemu wrappers (:doc:`aemu_integration`) — neural-network emulator
  bundles consumed through the same interface.

User spectrum parameter inputs
------------------------------

``UserSpectrumInterpolator.to_parameters`` accepts a dictionary or a list,
tuple, NumPy array, or JAX array in ``parameter_names`` order. Array inputs
may be a single parameter row or a two-dimensional batch of rows; pass one
row at a time to ``intensity`` (or map it with JAX). The default dictionary
keys are ``teff``, ``logg``, and ``m/h``; supported aliases such as
``metallicity`` are also accepted.

The stored parameter matrix has one column per stellar parameter, plus a
final ``mu`` column only when ``has_mu=True``. The wavelength is supplied
separately and is retained during interpolation. Inconsistent query and grid
dimensions raise ``ValueError``.

For a table without angular samples (``has_mu=False``), a dictionary passed
to ``intensity`` may also contain ``limb_darkening``. This
metadata is read before the atmosphere parameters are converted to an array.
For a linear law, a coefficient of 0 gives no darkening and a coefficient
of 1 multiplies the disk-centre intensity by ``mu``. If coefficients are
omitted, the interpolator warns and uses the law's default values.
