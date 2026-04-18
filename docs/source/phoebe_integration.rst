PHOEBE Integration
==================

SPICE provides integration with the PHOEBE (PHysics Of Eclipsing BinariEs) library, allowing users to model binary star systems and generate synthetic spectra based on PHOEBE models.

Setting up a PHOEBE Model
-------------------------

To use PHOEBE with SPICE, you first need to create a PHOEBE model. Here's a basic example:

.. code-block:: python

    import phoebe
    from spice.models import PhoebeModel, PhoebeConfig
    from phoebe.parameters.dataset import _mesh_columns
    
    # Create a PHOEBE bundle
    b = phoebe.default_star()

    # Define some times
    times = np.linspace(0, 1, 100)
    # SPICE requires several columns, so we'll add all available mesh columns
    COLUMNS = _mesh_columns
    b.add_dataset('mesh', times=times, columns=COLUMNS, dataset='mesh01')

    # Make sure to set the coordinates to 'uvw'
    b.run_compute( coordinates='uvw')
    
    # Create a PhoebeConfig object
    p = PhoebeConfig(b)
    
    # Generate a PhoebeModel for a specific time
    time = 0.0  # time in days
    pm = PhoebeModel.construct(p, bb.parameter_names, {pn: sp for pn, sp in zip(bb.parameter_names, bb.solar_parameters)})

The `PhoebeConfig` class wraps a PHOEBE bundle and provides methods to extract relevant information for SPICE. The `PhoebeModel` class represents a snapshot of the binary system at a specific time.

There are a few requirements PHOEBE needs to be set up so that SPICE can extract the necessary information:

- The `mesh` dataset needs to be added with all the necessary columns (see `_mesh_columns`)
- The `coordinates` parameter needs to be set to `uvw`
- For some emulators, a dictionary of stellar parameters and the corresponding values needs to be provided. Some parameters are not provided by PHOEBE and need to be manually assigned to the mesh model

Generating Spectra
------------------

Once you have a PhoebeModel, you can use it with SPICE's spectral synthesis functions:

.. code-block:: python

    from spice.models import Blackbody
    from spice.spectrum import simulate_observed_flux
    import numpy as np
    
    # Create a Blackbody model
    bb = Blackbody()
    
    # Generate wavelengths
    wavelengths = np.linspace(4000, 10000, 1000)
    
    # Simulate a spectrum
    spectrum = simulate_observed_flux(bb.intensity, pm, np.log10(wavelengths))

This will generate a synthetic spectrum based on the PHOEBE model at the specified time. (Default distance is $d=10$ pc)
