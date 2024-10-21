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
    b.add_dataset('mesh', times=times, columns=_mesh_columns, dataset='mesh01')
    b.add_dataset('orb', compute_times=times, dataset='orb01')
    b.add_dataset('lc', compute_times=times, passband='Johnson:V', dataset='lc01')

    b.set_value('distance@system', 10)  # in solar radii

    # Make sure to set the coordinates to 'uvw'
    b.run_compute(coordinates='uvw')

This setup creates a basic binary system with PHOEBE and performs an initial computation. The system can then be used with SPICE's `PhoebeBinary` class for further analysis and integration with other SPICE components.

For more detailed information on using PHOEBE with SPICE, refer to the full tutorial in the examples section.
