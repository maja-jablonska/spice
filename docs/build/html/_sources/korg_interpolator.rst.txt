Korg.jl Interpolator
================

The ``KorgSpectrumEmulator`` is a simple interpolator for a grid of model atmospheres generated with the `Korg.jl <https://ajwheeler.github.io/Korg.jl/stable/>`_ package.

The line list used for this grid is the `GALAH DR3 <https://www.galah-survey.org>`_ line list.

Basic Usage
----------

The interpolator can be used to generate synthetic spectra based on a grid of model atmospheres:

.. code-block:: python

    from spice.spectrum.spectrum_korg import KorgSpectrumEmulator

    k = KorgSpectrumEmulator()

The ``KorgSpectrumEmulator`` is downloaded from huggingface to a selected cache location. It's around 10 GBs, so if you want to change the location of the HuggingFace cache, make sure to set the environment variable:

.. code-block:: bash

    export HF_HOME=/your/cache/location


A small grid (~300 MB) will be downloaded by default. To switch to a larger grid (~10 GB), pass the argument of ``grid_type='regular'``.

Using the Interpolator with a Mesh Model
--------------

The interpolator can be used to generate synthetic spectra based on a mesh model, just like other spectrum models in SPICE.

.. code-block:: python

    from spice.models import IcosphereModel
    from spice.spectrum import simulate_observed_flux

    base_temp = 5700
    wavelengths = jnp.linspace(5650, 5910, 10000)
    m = IcosphereModel.construct(1000, 1., 1., k.to_parameters(dict(teff=base_temp)), k.stellar_parameter_names)
    non_rotated_spec = simulate_observed_flux(k.intensity, m, jnp.log10(wavelengths))

This code snipped will produce a synthetic spectrum for a 5700 K star, without any rotational effects.

.. image:: ../img/korg_non_rotating_spec.png
   :width: 600
   :alt: a synthetic spectrum without rotational effects

To add rotational effects, we can use the ``add_rotation`` function from the ``spice.models.mesh_transform`` module.

.. code-block:: python

    from spice.models.mesh_transform import add_rotation, evaluate_rotation

    m_r = add_rotation(m, rotation_velocity=10.)
    m_rotated = evaluate_rotation(m_r, 0.)
    rotated_spec = simulate_observed_flux(k.intensity, m_rotated, jnp.log10(wavelengths))

This code snipped will produce a synthetic spectrum for a 5700 K star, with a rotation velocity of 10 km/s.

.. image:: ../img/korg_rotating_spec.png
   :width: 1000
   :alt: a synthetic spectrum with rotational effects

Parameters
---------

The interpolator supports the following stellar parameters:

- ``teff``: Effective temperature (K)
- ``logg``: Surface gravity (log g)
- ``feh``: Metallicity [Fe/H]
- :math:`\mu`: Angle between the normal and the line of sight

The valid ranges depend on the underlying model grid being used.

For the small grid, the valid ranges are:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Valid Range
   * - ``teff``
     - 5000 - 6000 K
   * - ``logg``
     - 4.0 - 5.0
   * - ``feh``
     - -2.5 - 1.0
   * - :math:`\mu`
     - 0.0 - 1.0

The grid is sampled as follows:

.. image:: ../img/small_grid_params.png
   :width: 1000
   :alt: small grid parameter space

Some example spectra and continua are shown below:

.. image:: ../img/small_grid_spectra.png
   :width: 1000
   :alt: synthetic spectra in the small grid

.. image:: ../img/small_grid_continua.png
   :width: 1000
   :alt: synthetic continua in the small grid