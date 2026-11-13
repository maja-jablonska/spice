Synthetic Photometry
====================

The passband functions return **magnitudes**, despite their historical
``*_passband_luminosity`` names. Supply linear wavelengths in Angstroms and
a one-dimensional flux density in erg/s/cm²/Å. The spectrum must cover the
filter's response and resolve its structure; missing wavelengths are not
extrapolated into the magnitude integral.

.. note::

   Every snippet on this page, with the values it prints, is in the companion notebook
   `tutorial/docs_examples/synthetic_photometry_examples.ipynb <https://github.com/maja-jablonska/spice/blob/main/tutorial/docs_examples/synthetic_photometry_examples.ipynb>`_.

Magnitude systems
-----------------

- :func:`~spice.spectrum.AB_passband_luminosity` uses the AB system and is
  supported by every filter.
- :func:`~spice.spectrum.ST_passband_luminosity` uses the ST system. Gaia
  filters are not supported and raise ``ValueError``.
- :func:`~spice.spectrum.Vega_passband_luminosity` requires a Vega zero point
  or an AB-to-Vega offset on the filter: Gaia G/BP/RP/RVS (Riello et al. 2021),
  2MASS J/H/Ks (Cohen et al. 2003 zero-magnitude fluxes), and the
  standard-system Johnson-Cousins, Hipparcos/Tycho and Strömgren bands, which
  follow Bessell & Murphy (2012) as ``m_AB + offset``.
- In SPICE 1.9.0 and earlier, ``AB_passband_luminosity`` for Johnson-Cousins,
  Hipparcos/Tycho and Strömgren silently included those Bessell & Murphy
  offsets (U -0.761, B +0.130, V +0.020, R -0.163, I -0.404, Hp +0.030,
  BT +0.100, VT +0.037, u +0.308, v +0.327, b +0.187, y +0.027 mag), i.e. it
  returned standard-system magnitudes. Use ``Vega_passband_luminosity`` to get
  those values now.

At 10 pc, the returned magnitude is an absolute magnitude. At another
synthesis distance it is the apparent magnitude at that distance, without
interstellar extinction unless your input spectrum includes it.

Passband magnitudes from a spectrum
-----------------------------------

.. code-block:: python

    import jax.numpy as jnp
    from spice.models import IcosphereModel
    from spice.spectrum import Blackbody, simulate_observed_flux
    from spice.spectrum import (AB_passband_luminosity, ST_passband_luminosity,
                                Vega_passband_luminosity)
    from spice.spectrum.filter import JohnsonCousinsV, GaiaG

    bb = Blackbody()
    star = IcosphereModel.construct(100, 1., 1.,
                                    bb.solar_parameters, bb.parameter_names)
    wavelengths = jnp.linspace(3000., 12000., 2001)  # Angstroms
    spectrum = simulate_observed_flux(bb.intensity, star,
                                     jnp.log10(wavelengths), distance=10.)
    flux = spectrum[:, 0]  # erg/s/cm²/Angstrom, not the full two-column array

    magnitudes = {
        'V_AB': AB_passband_luminosity(JohnsonCousinsV(), wavelengths, flux),
        'V_ST': ST_passband_luminosity(JohnsonCousinsV(), wavelengths, flux),
        'G_AB': AB_passband_luminosity(GaiaG(), wavelengths, flux),
        'G_Vega': Vega_passband_luminosity(GaiaG(), wavelengths, flux),
    }

Pass the synthesis flux directly. Do not multiply or divide it by
:math:`10^8`, and do not convert the wavelengths to centimetres. Photon
counting and energy-response weighting are applied internally, including
Gaia's count-rate calibration. PanSTARRS PS1 curves are photon-counting
capture cross-sections (Tonry et al. 2012) and use the photon-counting
equations; 2MASS uses the energy (RSR) equations.
For a fixed intrinsic spectrum, changing distance from :math:`d_1` to
:math:`d_2` changes every magnitude by :math:`5\log_{10}(d_2/d_1)`.

``GaiaRVS`` uses the packaged `ESA Gaia DR3 response curve
<https://www.cosmos.esa.int/web/gaia/dr3-passbands>`_, covering approximately
8460–8700 Å. Its published 21.317 mag zero point belongs to the Vega system;
the AB zero point is derived separately from a 3631 Jy reference spectrum.
The distributed data includes its original attribution and source links.

Wavelength coverage
-------------------

A magnitude is only as good as the wavelength grid behind it. The passband
functions integrate over the grid they are given; if that grid covers only part
of the passband, the result is the band average over the covered part, which
is biased by however much the missing flux differs from it. Spectral emulators
make this easy to hit: the HARPS intensity emulator covers 3780–6910 Å, so
91 per cent of Hipparcos :math:`H_p`, 61 per cent of Gaia :math:`G` and 0 per
cent of 2MASS are inside it, and a bolometric quantity sees 27–41 per cent of
the flux of a 4500–8000 K star.

SPICE warns (``spice.spectrum.WavelengthCoverageWarning``) when

* more than 0.5 per cent of a passband's response lies outside the wavelengths
  passed to ``AB_passband_luminosity``, ``ST_passband_luminosity`` or
  ``Vega_passband_luminosity``;
* ``simulate_observed_flux``, ``simulate_monochromatic_luminosity`` or
  ``luminosity`` asks an emulator for wavelengths outside
  the range it was trained on (``emulator.wavelength_range``; ``None`` for an
  unrestricted model such as ``Blackbody``).

``Filter.wavelength_coverage(wavelengths)`` returns the covered fraction
directly. The emulator check finds the emulator through the function you pass,
so it applies to a bound method such as ``emulator.intensity`` or
``emulator.flux``; a function of your own that calls the emulator (like
``surface_flux`` below) hides it, and no range check is made. All of these
checks need concrete arrays and are skipped inside ``jax.jit``; call the
function once outside the traced code to see them.

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
     - Vega = AB + Bessell & Murphy (2012) offset
   * - Hipparcos/Tycho
     - ``HipparcosHp``, ``TychoBT``, ``TychoVT``
     - Vega = AB + Bessell & Murphy (2012) offset
   * - Gaia
     - ``GaiaG``, ``GaiaBP``, ``GaiaRP``, ``GaiaRVS``
     - photonic; no ST magnitudes; AB and Vega zero points
   * - SDSS
     - ``SDSSu/g/r/i/z``
     -
   * - 2MASS
     - ``TWOMASSJ/H/K``
     - energy (RSR) weighting; Vega zero points from Cohen et al. (2003)
   * - GALEX
     - ``GALEXFUV``, ``GALEXNUV``
     -
   * - LSST
     - ``LSSTu/g/r/i/z/y``
     -
   * - PanSTARRS PS1
     - ``PANSTARRS_PS1_g/r/i/z/y/w/open``
     - photon counting (capture cross-sections)
   * - Strömgren
     - ``Stromgrenu/v/b/y``
     - Vega = AB + Bessell & Murphy (2012) offset
   * - Bolometric
     - ``Bolometric``
     - flat response from 1 to 30000 Å; zero outside

Custom passbands can be built directly from a transmission curve via the
:class:`~spice.spectrum.Filter` base class.

Reference spectra and custom filters
------------------------------------

Set ``non_photonic=False`` for a photon-counting response and ``True`` for
an energy response. The curve has shape ``(2, n_samples)``: wavelengths in
Angstroms followed by dimensionless transmission values. Responses are zero
outside the tabulated wavelength range.

A flat AB reference spectrum has :math:`F_\nu = 3631` Jy. Converting this
to flux per Angstrom provides a useful absolute calibration check:

.. code-block:: python

    from spice.spectrum.filter import Filter

    reference_filter = Filter(
        jnp.array([[4000., 4500., 6000., 7000.], [0., 0.8, 1., 0.]]),
        name='Reference passband', non_photonic=False,
    )
    # c in Angstrom/s; F_nu in erg/s/cm²/Hz.
    ab_reference = 3.631e-20 * 2.99792458e18 / wavelengths**2
    ab_zero = AB_passband_luminosity(reference_filter, wavelengths, ab_reference)

    st_reference = jnp.full_like(wavelengths, 10.**(-21.10 / 2.5))
    st_zero = ST_passband_luminosity(reference_filter, wavelengths, st_reference)
    print(float(ab_zero), float(st_zero))  # approximately 0, 0 mag

This check uses the default AB/ST zero points of ``Filter``. Gaia uses its
own instrument zero points and photon-count calibration. A custom Vega
filter needs a zero point calibrated consistently with its response.

Luminosity and mesh convergence
-------------------------------

Use :func:`~spice.spectrum.luminosity` for an energy luminosity in erg/s,
then :func:`~spice.spectrum.absolute_bol_luminosity` for a bolometric
magnitude. Integrating a restricted wavelength range gives only that
range's contribution. The ``Bolometric`` filter above is a finite passband;
its AB magnitude is not a bolometric magnitude.

For an un-darkened blackbody, the emergent surface flux is :math:`\pi B_\lambda`.
``Blackbody.flux`` currently returns the intensity at ``mu=1``, so define
the surface-flux function explicitly before computing an absolute luminosity:

.. code-block:: python

    import numpy as np
    from spice.spectrum import luminosity, absolute_bol_luminosity

    def surface_flux(log_wavelengths, parameters):
        return jnp.pi * bb.intensity(log_wavelengths, 1., parameters)

    lum_wavelengths = np.geomspace(100., 1.e6, 4096)  # Angstroms
    L = luminosity(surface_flux, star, lum_wavelengths)  # erg/s
    M_bol = absolute_bol_luminosity(L)

    # Stefan-Boltzmann prediction using this mesh's surface area.
    sigma_cgs = 5.670374419e-5  # erg/s/cm²/K⁴
    surface_area_cm2 = jnp.sum(star.areas) * (star.radius * 6.957e10)**2
    L_stefan_boltzmann = surface_area_cm2 * sigma_cgs * bb.solar_parameters[0]**4
    print(float(L / L_stefan_boltzmann))  # approximately 1

Increase the wavelength coverage and sampling to check the integral, and
increase mesh resolution to check the surface-area approximation. For a
spherical star, the mesh area approaches :math:`4\pi R^2`. No empirical
luminosity offset is needed. See :doc:`spectral_synthesis` for surface-flux
requirements with limb darkening and atmosphere emulators, and
:doc:`conventions` for changes affecting results from earlier versions.
