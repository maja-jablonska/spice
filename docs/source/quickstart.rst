Quickstart
==========

This page walks through the two most common workflows end to end: a single
rotating star and an eclipsing binary. Both run with the core install only.

A single rotating star
----------------------

.. code-block:: python

    import numpy as np
    from spice.models import IcosphereModel
    from spice.models.mesh_transform import add_rotation
    from spice.spectrum import simulate_observed_flux, Blackbody

    bb = Blackbody()

    # A solar-like star: ~1000 mesh vertices, 1 solar radius, 1 solar mass
    star = IcosphereModel.construct(1000, 1., 1., bb.solar_parameters, bb.parameter_names)

    # Solid-body rotation with a 2 km/s equatorial velocity
    star = add_rotation(star, rotation_velocity=2.0)

    # simulate_observed_flux expects *log10* wavelengths (in Angstroms) and
    # returns an (n_wavelengths, 2) array; column 0 is the flux in
    # erg/s/cm^2/Angstrom at the default distance of 10 pc.
    wavelengths = np.logspace(3, 4, 1000)  # 1,000-10,000 Angstroms
    flux = simulate_observed_flux(bb.intensity, star, np.log10(wavelengths))

Three things to remember (all covered in detail in :doc:`conventions` and
:doc:`spectral_synthesis`):

- wavelengths are passed as ``log10`` of Angstroms;
- the result has two columns — flux and the emulator's second channel
  (continuum for grid interpolators);
- the mesh, not the emulator, carries all geometry: rotation, pulsations,
  spots, and orbits are applied to the :class:`~spice.models.MeshModel`.

An eclipsing binary with a light curve
--------------------------------------

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from spice.models import IcosphereModel, Binary
    from spice.models.binary import add_orbit, evaluate_orbit_at_times
    from spice.models.mesh_view import get_mesh_view
    from spice.spectrum import simulate_observed_flux, Blackbody, AB_passband_luminosity
    from spice.spectrum.filter import GaiaG

    bb = Blackbody()
    los = jnp.array([0.0, 1.0, 0.0])  # line of sight

    # Cast each component to the line of sight so occlusions can be resolved
    primary = get_mesh_view(
        IcosphereModel.construct(1000, 1.0, 1.0, bb.solar_parameters, bb.parameter_names), los)
    secondary = get_mesh_view(
        IcosphereModel.construct(1000, 0.8, 0.8, bb.solar_parameters, bb.parameter_names), los)

    binary = Binary.from_bodies(primary, secondary)

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
        vgamma=0.0,               # systemic velocity [km/s]
        orbit_resolution_points=50,
    )

    # Evaluate the orbit across phases (eclipses/occlusions resolved internally)
    times = jnp.linspace(0.0, 1.0, 100)
    primaries, secondaries = evaluate_orbit_at_times(binary, times)

    # Combined Gaia G-band light curve
    wavelengths = np.linspace(900, 40000, 1000)
    gaia_g = GaiaG()
    light_curve = [
        AB_passband_luminosity(
            gaia_g,
            wavelengths,
            simulate_observed_flux(bb.intensity, p1, np.log10(wavelengths))[:, 0]
            + simulate_observed_flux(bb.intensity, p2, np.log10(wavelengths))[:, 0],
        )
        for p1, p2 in zip(primaries, secondaries)
    ]

See :doc:`binaries` for the full binary-modelling guide (including PHOEBE
import) and :doc:`synthetic_photometry` for the available filters and
magnitude systems.

Where to go next
----------------

- :doc:`mesh` — building meshes; rotation, pulsations, and spots
- :doc:`spectral_synthesis` — how flux synthesis works and its knobs
- :doc:`spectral_grids` — realistic spectra from precomputed atmosphere grids
- :doc:`custom_emulators` — plugging in your own spectrum model
