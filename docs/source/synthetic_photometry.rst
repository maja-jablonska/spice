Synthetic Photometry
===================================

SPICE provides robust capabilities for synthetic photometry calculations. This section demonstrates how to use SPICE to generate synthetic photometry for various passbands and calculate stellar luminosities.

Passband Luminosities
---------------------

SPICE can calculate luminosities for different photometric filters, given a synthetic spectrum:

.. code-block:: python

    from spice.spectrum.filter import BesselU, BesselB, BesselV, Bolometric, GaiaG
    from spice.spectrum.spectrum import AB_passband_luminosity, luminosity
    
    # Calculate passband luminosities
    filters = [BesselU(), BesselB(), BesselV(), Bolometric(), GaiaG()]
    passband_lums = [AB_passband_luminosity(f, wavelengths, flux) for f in filters]

This code snippet demonstrates how to calculate luminosities for Bessel U, B, V, Bolometric, and Gaia G passbands.

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
    from spice.spectrum import simulate_observed_flux, luminosity, absolute_bol_luminosity
    from spice.spectrum.filter import BesselB, BesselI, GaiaG, JohnsonV
    from spice.spectrum.spectrum import AB_passband_luminosity, ST_passband_luminosity
    from transformer_payne import Blackbody

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
            'AB_solar_apparent_mag_B': AB_passband_luminosity(BesselB(), wavelengths, flux[:, 0]),
            'AB_solar_apparent_mag_V': AB_passband_luminosity(JohnsonV(), wavelengths, flux[:, 0]),
            'ST_solar_apparent_mag_G': ST_passband_luminosity(GaiaG(), wavelengths, flux[:, 0]),
        }

    # Calculate for different resolutions
    results = [calculate_blackbody_luminosity(n) for n in [100, 1000, 5000, 10000]]

This example shows how to calculate luminosities and magnitudes for blackbody models with different numbers of vertices, allowing for analysis of how model resolution affects the results.

These examples demonstrate SPICE's capabilities in synthetic photometry, from basic passband luminosity calculations to more complex analyses of blackbody models at various resolutions.
