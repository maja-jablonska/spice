Binaries
========

SPICE models binary systems in two ways: **natively**, by placing two mesh
models on a Keplerian orbit, or by **importing PHOEBE meshes** for
Roche-distorted close binaries. Both produce component meshes you synthesize
with :func:`~spice.spectrum.simulate_observed_flux`.

.. note::

   Every code snippet on this page has a matching section in the companion notebook
   `tutorial/docs_examples/binaries_examples.ipynb <https://github.com/maja-jablonska/spice/blob/main/tutorial/docs_examples/binaries_examples.ipynb>`_
   (the PHOEBE sections require the ``phoebe`` extra).

Native Keplerian binaries
-------------------------

Build the two components, cast them to the line of sight (required for
occlusion handling), and combine:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from spice.models import IcosphereModel, Binary
    from spice.models.binary import add_orbit, evaluate_orbit, evaluate_orbit_at_times
    from spice.models.mesh_view import get_mesh_view
    from spice.spectrum import simulate_observed_flux, Blackbody

    bb = Blackbody()
    los = jnp.array([0.0, 1.0, 0.0])

    primary = get_mesh_view(
        IcosphereModel.construct(1000, 1.0, 1.0, bb.solar_parameters, bb.parameter_names), los)
    secondary = get_mesh_view(
        IcosphereModel.construct(1000, 0.8, 0.8, bb.solar_parameters, bb.parameter_names), los)

    binary = Binary.from_bodies(primary, secondary)

Attach the orbit — component masses come from the meshes; the elements are:

.. code-block:: python

    binary = add_orbit(
        binary,
        P=1.0,                    # orbital period [years]
        ecc=0.1,                  # eccentricity
        T=0.0,                    # time of periastron passage [years]
        i=np.pi / 3,              # inclination [rad]
        omega=0.0,                # argument of periastron [rad]
        Omega=0.0,                # longitude of the ascending node [rad]
        mean_anomaly=0.0,         # mean anomaly at the reference time [rad]
        reference_time=0.0,       # reference time [years]
        vgamma=0.0,               # systemic velocity [km/s], positive = receding
        orbit_resolution_points=50,
    )

``orbit_resolution_points`` sets how densely the orbit is precomputed for
interpolation; increase it for very eccentric orbits.

Evaluating the orbit
^^^^^^^^^^^^^^^^^^^^

:func:`~spice.models.binary.evaluate_orbit` returns the two component meshes
at one time — positioned, velocity-tagged, and with mutual occlusions
resolved (eclipsed area removed from the occluded component's visible
areas). :func:`~spice.models.binary.evaluate_orbit_at_times` maps this over
a time array:

.. code-block:: python

    times = jnp.linspace(0.0, 1.0, 100)
    primaries, secondaries = evaluate_orbit_at_times(binary, times)

    wavelengths = np.linspace(900, 40000, 1000)
    spectra = [
        simulate_observed_flux(bb.intensity, p1, np.log10(wavelengths))[:, 0]
        + simulate_observed_flux(bb.intensity, p2, np.log10(wavelengths))[:, 0]
        for p1, p2 in zip(primaries, secondaries)
    ]

Passing the summed spectra through
:func:`~spice.spectrum.AB_passband_luminosity` gives an eclipsing light
curve (see :doc:`quickstart` for the complete example), and each component's
``los_velocities`` carry the orbital radial-velocity signal, so spectral
time series show the orbital Doppler shifts (plus ``vgamma``).

Locating eclipses
^^^^^^^^^^^^^^^^^

:func:`~spice.models.find_eclipses` locates eclipse windows from sampled
sky-projected positions and velocities of the two components: it returns the
contact times (T1–T4), mid-eclipse time, and whether each event is partial,
total, or grazing. Use it to concentrate expensive synthesis time points
around the events instead of sampling the whole orbit densely.

PHOEBE binaries
---------------

For semi-detached and contact systems where Roche geometry and tidal
distortion matter, compute the meshes in PHOEBE and wrap them for SPICE
(install with ``pip install "stellar-spice[phoebe]"``):

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

PHOEBE meshes are read-only inside SPICE — geometry, rotation, and orbital
state all come from the PHOEBE computation (times in PHOEBE's native days),
and SPICE adds the spectral synthesis on top. For more detail on the wrapper
classes see :doc:`phoebe_integration`.
