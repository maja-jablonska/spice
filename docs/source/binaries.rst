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
    los = jnp.array([0.0, 0.0, -1.0])  # line of sight (observer -> star)

    primary = get_mesh_view(
        IcosphereModel.construct(1000, 1.0, 1.0, bb.solar_parameters, bb.parameter_names), los)
    secondary = get_mesh_view(
        IcosphereModel.construct(1000, 0.8, 0.8, bb.solar_parameters, bb.parameter_names), los)

    binary = Binary.from_bodies(primary, secondary)

Attach the orbit — component masses come from the meshes; the elements are:

.. code-block:: python

    binary = add_orbit(
        binary,
        P=0.01,                   # orbital period [years] (~3.65 days)
        ecc=0.0,                  # eccentricity
        T=0.0,                    # time of periastron passage [years]
        i=np.pi / 2,              # inclination [rad], edge-on so eclipses occur
        omega=0.0,                # argument of periastron [rad]
        Omega=0.0,                # longitude of the ascending node [rad]
        mean_anomaly=0.0,         # additive phase offset from periastron timing [rad]
        reference_time=0.0,       # reference time [years]
        vgamma=0.0,               # systemic velocity [km/s], positive = receding
        orbit_resolution_points=20,
    )

``orbit_resolution_points`` sets how densely the orbit is precomputed for
interpolation (positions and velocities are interpolated linearly between
the precomputed points); increase it for very eccentric orbits or when you
need the orbit more accurately than this short example does.

.. warning::

   ``mean_anomaly`` is an *additive phase offset on top of* the periastron
   timing — the orbit solver uses ``M(t) = mean_anomaly + n·(t − T)``. Set
   either ``mean_anomaly`` or ``T``, not both: passing PHOEBE-style values
   for both (where the mean anomaly at the reference epoch already encodes
   the periastron time) double-counts the periastron phase and shifts every
   orbital event. This does not affect ``PhoebeBinary``, which uses PHOEBE's
   own precomputed orbit throughout.

Evaluating the orbit
^^^^^^^^^^^^^^^^^^^^

:func:`~spice.models.binary.evaluate_orbit` returns the two component meshes
at one time — positioned, velocity-tagged, and with mutual occlusions
resolved (eclipsed area removed from the occluded component's visible
areas). :func:`~spice.models.binary.evaluate_orbit_at_times` maps this over
a time array:

.. code-block:: python

    # Each time step resolves occlusions between two ~1000-vertex meshes;
    # on a laptop CPU these 20 steps take several minutes.
    times = jnp.linspace(0.0, 0.01, 20)   # one orbital period
    primaries, secondaries = evaluate_orbit_at_times(binary, times)

    wavelengths = np.linspace(900, 40000, 500)
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

The cost is dominated by the occlusion step, so it grows with the number of
time steps and the mesh resolution. Sample densely only where it matters,
for example inside the eclipse windows (see `Locating eclipses`_ below).

Occlusion and mesh resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At every time step the component that is farther along the line of sight
is the occluded one. Each of its front-facing faces (``mu > 0``) is clipped
against the overlapping front-facing faces of the nearer component in the
sky plane, and the overlap is subtracted from its visible area. No face is
dropped: an uneven tessellation can have more than half of its faces visible
without losing any of them from the calculation.

Candidate face pairs come from a nearest-neighbour search, whose size ``k``
and direction :meth:`~spice.models.Binary.from_bodies` picks once, from the
meshes as passed. This is why the components must be cast to the line of
sight with ``get_mesh_view`` first. ``k`` is estimated from the projected face
sizes and face densities (clipped to 24-256). By default each occluded face
searches its ``k`` nearest occluder faces. When that would need more than 256
neighbours, as for a small, finely meshed body in front of coarse faces (a
planetary transit), the search runs from the occluder side instead
(``from_occluder``): each occluder face searches its nearest occluded faces.
Both directions sum the same overlapping pairs. Before the November 2026 fix
the occluded-side search was simply capped in that case, which dropped most
of the occluder faces, and transits came out about 49-62% too shallow. Redo
transit light curves computed with earlier versions.

Passing ``n_neighbours1``/``n_neighbours2`` to ``from_bodies`` overrides the
estimate and always uses the occluded-side search. If even the cheaper
direction needs more than 256 neighbours, ``from_bodies`` emits a
``RuntimeWarning`` (the eclipse may come out too shallow); give the two bodies
more similar mesh resolutions. Mesh resolution still limits eclipse
accuracy, so check convergence for grazing events, very different component
sizes, and locally refined meshes.

Locating eclipses
^^^^^^^^^^^^^^^^^

:func:`~spice.models.find_binary_eclipses` locates eclipse windows directly
from a ``Binary``: it samples a low-resolution orbit itself from the
binary's stored orbital elements, so no manual orbit evaluation is needed
(for a ``PhoebeBinary`` see `Locating eclipses in a PHOEBE binary`_). It
returns the contact times (T1–T4), the mid-eclipse time, and the eclipse
``kind``. Use it to concentrate expensive synthesis time points around the
events instead of sampling the whole orbit densely:

.. code-block:: python

    from spice.models import find_binary_eclipses

    eclipses = find_binary_eclipses(binary)   # scans one period by default
    for e in eclipses:
        print(e['kind'], e['T1'], e['mid'], e['T4'])   # times in years

    # Concentrate synthesis points inside the first eclipse window
    dense = jnp.linspace(eclipses[0]['T1'], eclipses[0]['T4'], 10)
    primaries, secondaries = evaluate_orbit_at_times(binary, dense)

``kind`` is one of:

``'partial'``
    The discs overlap, but the smaller one never lies entirely inside the
    larger one. ``T2`` and ``T3`` are ``None``.
``'total'`` / ``'annular'``
    The smaller disc lies entirely inside the larger one between the inner
    contacts ``T2`` and ``T3``. The event is called ``'total'`` when
    ``body1`` is at least as large as ``body2`` and ``'annular'`` otherwise.
``'grazing'``
    The minimum separation reaches the inner-contact limit ``|R1 - R2|``, but
    the inner contacts cannot be resolved (totality is numerically
    borderline). ``T2`` and ``T3`` are ``None``.

The label reflects only the relative radii of ``body1`` and ``body2``, not
which star is in front. Both events of one system therefore get the same
label. For the edge-on system above this finds two eclipses per period, at
phases 0.25 and 0.75, both ``'total'``, although the phase-0.25 event is
physically a transit of the smaller secondary across the primary. To tell
which component is in front, compare the components' positions along the
line of sight at ``e['mid']`` (the companion notebook shows how).

The sky plane defaults to the plane perpendicular to ``body1``'s line of
sight — the same vector the occlusion resolution uses — so the finder agrees
with what the synthesis will actually eclipse; pass ``los_vector=`` to
override. The lower-level :func:`~spice.models.find_eclipses` remains available
for orbits sampled by other means: it takes sky-projected positions and
velocities of the two components explicitly.

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

    b.set_value('distance@system', 10)  # in PHOEBE's default distance unit (m)

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

Locating eclipses in a PHOEBE binary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~spice.models.find_binary_eclipses` also accepts a ``PhoebeBinary``.
In that case it does not solve a Kepler orbit. It scans PHOEBE's own
precomputed orbit: the ``us``/``vs`` positions and ``vus``/``vvs`` velocities
of the ``orb`` dataset in the ``uvw`` frame (``u`` and ``v`` span the sky
plane, ``w`` is the line of sight). The bundle therefore needs an ``orb``
dataset computed at the same times as the mesh, with
``run_compute(coordinates='uvw')``, as in the example above. ``PhoebeConfig``
uses ``'orb01'`` automatically, or the dataset passed as
``orbit_dataset_name``. The component radii are PHOEBE's ``requiv``, so the
contact times treat Roche-distorted components as spheres of equal volume.

Times are in days (the PHOEBE clock). ``n_points``, ``times`` and
``los_vector`` are ignored. When the computed window spans at least one
orbital period, the finder pads it periodically (the samples shifted by one
period on either side), so an eclipse cut by either end of the window is still
found. It returns every eclipse whose mid time lies in ``[t[0], t[-1])``; the
contact times of an edge eclipse can fall just outside the computed times.
Padding assumes a strictly periodic orbit, with no apsidal motion or period
change. A window shorter than one period cannot be padded: the finder emits
a ``RuntimeWarning``, and eclipses cut by its ends are missed. For the bundle
above, sampled over exactly one 1-day period, it finds both eclipses, centred
at 0 and 0.5 d (both ``'partial'``):

.. code-block:: python

    from spice.models import find_binary_eclipses

    eclipses = find_binary_eclipses(binary)   # the PhoebeBinary from above
    for e in eclipses:
        print(e['kind'], e['T1'], e['mid'], e['T4'])   # times in days
