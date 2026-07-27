Binaries
=======================

PHOEBE Configuration and Binary System Setup
--------------------------------------------

Here's an example of how to set up a basic PHOEBE binary system:

.. code-block:: python

    import phoebe
    import numpy as np
    from phoebe.parameters.dataset import _mesh_columns

    # Create a default binary system
    b = phoebe.default_binary()

    # Define time points (in days)
    times = np.linspace(0, 1, 100)

    # Add datasets
    b.add_dataset('mesh', compute_times=times, columns=_mesh_columns, dataset='mesh01')
    b.add_dataset('orb', compute_times=times, dataset='orb01')
    b.add_dataset('lc', compute_times=times, passband='Johnson:V', dataset='lc01')

    b.set_value('distance@system', 10)  # in solar radii

    # Make sure to set the coordinates to 'uvw'
    b.run_compute(coordinates='uvw')

This setup creates a basic binary system with PHOEBE and performs an initial computation. The system can then be wrapped with SPICE's `PhoebeBinary` class for further analysis and integration with other SPICE components.

Once the bundle has been computed, wrap it for SPICE and synthesise a spectrum:

.. code-block:: python

    from spice.models import PhoebeBinary
    from spice.models.binary import evaluate_orbit
    from spice.models.phoebe_utils import PhoebeConfig
    from spice.spectrum import simulate_observed_flux, Blackbody

    bb = Blackbody()

    # Wrap the PHOEBE meshes (read-only inside SPICE)
    config = PhoebeConfig(b, 'mesh01')
    binary = PhoebeBinary.construct(config, ['teff', 'logg', 'abun'])

    # Evaluate the components at a snapshot time, then synthesise a spectrum
    primary, secondary = evaluate_orbit(binary, config.times[0])
    wavelengths = np.logspace(3, 4, 1000)
    flux = simulate_observed_flux(bb.intensity, primary, np.log10(wavelengths))

For more detailed information on using PHOEBE with SPICE, refer to the full tutorial in the examples section.
