Synthetic Photometry
===================================

SPICE provides robust capabilities for synthetic photometry calculations. This section demonstrates how to use SPICE to generate synthetic photometry for various passbands and calculate stellar luminosities.

.. note::

   Every code snippet on this page has a matching section in the companion notebook
   `tutorial/docs_examples/synthetic_photometry_examples.ipynb <https://github.com/maja-jablonska/spice/blob/main/tutorial/docs_examples/synthetic_photometry_examples.ipynb>`_.

Magnitude systems
-----------------

Three functions turn an observed spectrum (linear wavelengths in Angstroms
plus the flux column from :func:`~spice.spectrum.simulate_observed_flux`)
into a magnitude:

- :func:`~spice.spectrum.AB_passband_luminosity` — AB system; supported by
  every filter.
- :func:`~spice.spectrum.ST_passband_luminosity` — ST system; **not
  supported for Gaia filters** (raises ``ValueError``).
- :func:`~spice.spectrum.Vega_passband_luminosity` — Vega system; requires
  the filter to carry a Vega zero point (currently Gaia G/BP/RP and
  2MASS J/H/Ks).

Photonic filters (photon-counting responses, e.g. Gaia) and energy-based
(non-photonic) responses are handled internally per filter — you never need
to convert; PanSTARRS PS1 filters likewise route through a dedicated branch.
At the default synthesis distance of 10 pc, the AB "apparent" magnitude of a
model is its absolute magnitude.

Available filters
-----------------

All filters live in ``spice.spectrum.filter``; transmission-curve data ships
with the package.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - System
     - Classes
     - Notes
   * - Johnson-Cousins
     - ``JohnsonCousinsU/B/V/R/I``
     -
   * - Hipparcos/Tycho
     - ``HipparcosHp``, ``TychoBT``, ``TychoVT``
     -
   * - Gaia
     - ``GaiaG``, ``GaiaBP``, ``GaiaRP``, ``GaiaRVS``
     - photonic; no ST magnitudes; Vega zero points for G/BP/RP
   * - SDSS
     - ``SDSSu/g/r/i/z``
     -
   * - 2MASS
     - ``TWOMASSJ/H/K``
     - Vega zero points available
   * - GALEX
     - ``GALEXFUV``, ``GALEXNUV``
     -
   * - LSST
     - ``LSSTu/g/r/i/z/y``
     -
   * - PanSTARRS PS1
     - ``PANSTARRS_PS1_g/r/i/z/y/w/open``
     - dedicated non-photonic handling
   * - Strömgren
     - ``Stromgrenu/v/b/y``
     -
   * - Bolometric
     - ``Bolometric``
     - flat response over the full wavelength range

Custom passbands can be built directly from a transmission curve via the
:class:`~spice.spectrum.Filter` base class.

Passband Luminosities
---------------------

SPICE can calculate luminosities for different photometric filters, given a synthetic spectrum:

.. code-block:: python

    from spice.spectrum.filter import JohnsonCousinsU, JohnsonCousinsB, JohnsonCousinsV, Bolometric, GaiaG
    from spice.spectrum.spectrum import AB_passband_luminosity, luminosity

    # Calculate passband luminosities
    filters = [JohnsonCousinsU(), JohnsonCousinsB(), JohnsonCousinsV(), Bolometric(), GaiaG()]
    passband_lums = [AB_passband_luminosity(f, wavelengths, flux) for f in filters]

This code snippet demonstrates how to calculate luminosities for Johnson-Cousins U, B, V, Bolometric, and Gaia G passbands.

Solar Luminosity Calculation
----------------------------

SPICE can be used to calculate theoretical stellar luminosities, such as the Sun's:

.. code-block:: python

    import astropy.units as u
    import jax.numpy as jnp

    # Calculate theoretical solar luminosity
    sigma = (5.67e-8 * u.W / (u.m**2) / (u.K**4)).to(u.erg / (u.cm**2) / (u.s) / (u.K**4))
    solar_luminosity = 0.9997011 * jnp.sum(model.areas) * (u.solRad.to(u.cm)**2) * sigma * (5772*u.K)**4

    print(f"Theoretical luminosity of the Sun: {solar_luminosity:.3e} erg/s")

This calculation uses the Stefan-Boltzmann law and the known properties of the Sun to compute its theoretical luminosity.

Blackbody Luminosity Offsets
----------------------------

SPICE includes utilities to calculate luminosity offsets for blackbody models with varying resolutions:

.. code-block:: python

    from spice.models import IcosphereModel
    from spice.spectrum import simulate_observed_flux, luminosity, absolute_bol_luminosity, Blackbody
    from spice.spectrum.filter import JohnsonCousinsB, JohnsonCousinsI, GaiaG, JohnsonCousinsV
    from spice.spectrum.spectrum import AB_passband_luminosity, ST_passband_luminosity

    def calculate_blackbody_luminosity(n_vertices):
        bb = Blackbody()
        model = IcosphereModel.construct(n_vertices, 1., 1., bb.solar_parameters, bb.parameter_names)
        
        wavelengths = jnp.linspace(1., 100000., 100000)
        flux = simulate_observed_flux(bb.intensity, model, jnp.log10(wavelengths), 10., chunk_size=1000, disable_doppler_shift=True)
        
        solar_luminosity = luminosity(bb.flux, model, wavelengths)
        
        return {
            'n_vertices': len(model.d_vertices),
            'solar_luminosity': solar_luminosity,
            'absolute_bol_luminosity': absolute_bol_luminosity(solar_luminosity),
            'AB_solar_apparent_mag_B': AB_passband_luminosity(JohnsonCousinsB(), wavelengths, flux[:, 0]),
            'AB_solar_apparent_mag_V': AB_passband_luminosity(JohnsonCousinsV(), wavelengths, flux[:, 0]),
            # Gaia filters are photonic and not supported for ST magnitudes
            'ST_solar_apparent_mag_V': ST_passband_luminosity(JohnsonCousinsV(), wavelengths, flux[:, 0]),
        }

    # Calculate for different resolutions
    results = [calculate_blackbody_luminosity(n) for n in [100, 1000, 5000, 10000]]

This example shows how to calculate luminosities and magnitudes for blackbody models with different numbers of vertices, allowing for analysis of how model resolution affects the results.

These examples demonstrate SPICE's capabilities in synthetic photometry, from basic passband luminosity calculations to more complex analyses of blackbody models at various resolutions.
