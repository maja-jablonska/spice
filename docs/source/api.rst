SPICE: API
===================================

Models
------

MeshModel
~~~~~~~~~
.. autoclass:: spice.models.mesh_model.MeshModel
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: count, index

   .. py:attribute:: center
      :type: Float[Array, "3"]

      Center position vector of the mesh model.

   .. py:attribute:: radius
      :type: float

      Radius in solar radii.

   .. py:attribute:: mass
      :type: float 

      Mass in solar masses.

   .. py:attribute:: d_vertices
      :type: Float[Array, "n_vertices 3"]

      Vertices positions relative to center.

   .. py:attribute:: faces
      :type: Float[Array, "n_faces 3"]

      Triangle face indices.

   .. py:attribute:: d_centers
      :type: Float[Array, "n_mesh_elements 3"]

      Triangle centers relative to center.

   .. py:attribute:: base_areas
      :type: Float[Array, "n_mesh_elements"]

      Base surface areas of triangles.

   .. py:attribute:: parameters
      :type: Float[Array, "n_mesh_elements n_parameters"]

      Physical parameters for each mesh element.

   .. py:attribute:: rotation_velocities
      :type: Float[Array, "n_mesh_elements 3"]

      Rotation velocities in km/s.

   .. py:attribute:: vertices_pulsation_offsets
      :type: Float[Array, "n_vertices 3"]

      Pulsation offsets for vertices in km/s.

   .. py:attribute:: center_pulsation_offsets
      :type: Float[Array, "n_mesh_elements 3"]

      Pulsation offsets for centers in km/s.

   .. py:attribute:: area_pulsation_offsets
      :type: Float[Array, "n_mesh_elements"]

      Pulsation offsets for areas.

   .. py:attribute:: pulsation_velocities
      :type: Float[Array, "n_mesh_elements 3"]

      Pulsation velocities in km/s.

   .. py:attribute:: rotation_axis
      :type: Float[Array, "3"]

      Rotation axis vector.

   .. py:attribute:: rotation_matrix
      :type: Float[Array, "3 3"]

      Rotation transformation matrix.

   .. py:attribute:: rotation_matrix_prim
      :type: Float[Array, "3 3"]

      Primary rotation transformation matrix.

   .. py:attribute:: axis_radii
      :type: Float[Array, "n_mesh_elements"]

      Radii from rotation axis.

   .. py:attribute:: occluded_areas
      :type: Float[Array, "n_mesh_elements"]

      Areas of occluded mesh elements.

   .. py:attribute:: los_vector
      :type: Float[Array, "3"]

      Line-of-sight vector.

   .. py:attribute:: spherical_harmonics_parameters
      :type: Float[Array, "n_puls_orders 2"]

      Parameters for spherical harmonics.

   .. py:attribute:: pulsation_periods
      :type: Float[Array, "n_puls_orders"]

      Periods of pulsation modes.

   .. py:attribute:: fourier_series_parameters
      :type: Float[Array, "n_puls_orders 3 n_fourier_orders 2"]

      Fourier series parameters for pulsations. The second axis indexes the
      vector-spherical-harmonic components [radial, spheroidal, toroidal];
      each innermost pair is [amplitude, phase].

   .. py:attribute:: pulsation_axes
      :type: Float[Array, "n_puls_orders 3"]

      Axes of pulsation modes.

   .. py:attribute:: pulsation_angles
      :type: Float[Array, "n_puls_orders"]

      Angles of pulsation modes.

IcosphereModel
~~~~~~~~~~~~~~
.. autoclass:: spice.models.mesh_model.IcosphereModel
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: count, index, construct

   .. method:: construct(n_vertices: int, radius: float, mass: float, parameters: Union[float, Float[Array, "n_mesh_elements n_parameters"]], parameter_names: List[str], max_pulsation_mode: int = 5, max_fourier_order: int = 5, override_log_g: bool = True, log_g_index: Optional[int] = None) -> IcosphereModel
      
      Constructs an IcosphereModel with specified stellar and mesh properties.

      This method generates an icosphere mesh and initializes the model with given parameters, including
      stellar properties (mass, radius) and mesh properties (vertices, faces, areas, centers). 
      It also handles the calculation of surface gravity (log g) values if required.

      :param n_vertices: Number of vertices for the icosphere mesh
      :param radius: Radius of the icosphere in solar radii
      :param mass: Mass of the stellar object in solar masses
      :param parameters: Parameters for the model, can be a single value or an array
      :param parameter_names: Names of the parameters, used for identifying log g parameter
      :param max_pulsation_mode: Maximum pulsation mode for the model, defaults to 5
      :param max_fourier_order: Maximum order of Fourier series for pulsation calculation, defaults to 5
      :param override_log_g: Whether to override the log g values based on model's mass and centers, defaults to True
      :param log_g_index: Index of the log g parameter in parameters array. Required if override_log_g is True and specific log g parameter name not in parameter_names
      :return: An instance of IcosphereModel initialized with the specified properties

Constants
~~~~~~~~~
.. data:: spice.models.mesh_model.LOG_G_NAMES
   :type: List[str]
   :value: ['logg', 'loggs', 'log_g', 'log_gs', 'log g', 'log gs', 'surface gravity', 'surface gravities', 'surface_gravity', 'surface_gravities']

   List of valid parameter names for surface gravity. If the parameter name is not in this list, the surface gravity will be calculated using the mass and center positions.

The default line-of-sight vector is ``[0., 1., 0.]`` (the +Y direction) and the
default rotation axis is ``[0., 0., 1.]`` (the Z axis); both are created
internally with the dtype matching the active precision setting.

.. data:: spice.models.mesh_model.DEFAULT_MAX_PULSATION_MODE_PARAMETER
   :type: int
   :value: 5

   Default maximum pulsation mode.

.. data:: spice.models.mesh_model.DEFAULT_FOURIER_ORDER  
   :type: int
   :value: 5

   Default Fourier order for pulsations.


.. autofunction:: spice.models.mesh_model.calculate_log_gs


Mesh Transformations
--------------------

All transformations are functional: they return a new model and never mutate
in place. See :doc:`conventions` for units and sign conventions.

.. autofunction:: spice.models.mesh_transform.transform

.. autofunction:: spice.models.mesh_transform.update_parameter

.. autofunction:: spice.models.mesh_transform.update_parameters

Rotation
~~~~~~~~

.. autofunction:: spice.models.mesh_transform.add_rotation

.. autofunction:: spice.models.mesh_transform.evaluate_rotation

.. autofunction:: spice.models.mesh_transform.evaluate_rotation_at_times

.. autofunction:: spice.models.mesh_transform.evaluate_body_orbit

Pulsations
~~~~~~~~~~

.. autofunction:: spice.models.mesh_transform.add_pulsation

.. autofunction:: spice.models.mesh_transform.add_pulsations

.. autofunction:: spice.models.mesh_transform.evaluate_pulsations

.. autofunction:: spice.models.mesh_transform.reset_pulsations

Spots
~~~~~

.. autofunction:: spice.models.spots.add_spot

.. autofunction:: spice.models.spots.add_spots

.. autofunction:: spice.models.spots.add_spherical_harmonic_spot

.. autofunction:: spice.models.spots.add_spherical_harmonic_spots


Mesh View and Occlusion
-----------------------

.. autofunction:: spice.models.mesh_view.get_mesh_view

.. autofunction:: spice.models.mesh_view.visible_area

.. autofunction:: spice.models.mesh_view.resolve_occlusion


Binaries and Orbits
-------------------

.. autoclass:: spice.models.binary.Binary
   :members: from_bodies

.. autofunction:: spice.models.binary.add_orbit

.. autofunction:: spice.models.binary.evaluate_orbit

.. autofunction:: spice.models.binary.evaluate_orbit_at_times

.. autofunction:: spice.models.binary.evaluate_orbit_at_times_stacked

.. autofunction:: spice.models.eclipse_utils.find_eclipses

.. autofunction:: spice.models.orbit_utils.get_orbit_jax


PHOEBE Integration
------------------

Available with the ``phoebe`` extra; see :doc:`phoebe_integration`.

.. autoclass:: spice.models.binary.PhoebeBinary
   :members: construct

.. autoclass:: spice.models.phoebe_model.PhoebeModel
   :members: construct

.. autoclass:: spice.models.phoebe_utils.PhoebeConfig
   :members:


Spectral Synthesis
------------------

.. autofunction:: spice.spectrum.spectrum.simulate_observed_flux

.. autofunction:: spice.spectrum.spectrum.simulate_monochromatic_luminosity

.. autofunction:: spice.spectrum.spectrum.luminosity

.. autofunction:: spice.spectrum.spectrum.absolute_bol_luminosity

.. autofunction:: spice.spectrum.utils.apply_spectral_resolution


Emulators
---------

.. autoclass:: spice.spectrum.spectrum_emulator.SpectrumEmulator
   :members:

.. autoclass:: spice.spectrum.blackbody.Blackbody
   :members:

.. autoclass:: spice.spectrum.gaussian_line_emulator.GaussianLineEmulator
   :members:

.. autoclass:: spice.spectrum.physical_line_emulator.PhysicalLineEmulator
   :members:

.. autoclass:: spice.spectrum.user_spectrum_interpolator.UserSpectrumInterpolator
   :members:


Grid Interpolation
------------------

Available with the ``grid`` extra; see :doc:`spectral_grids`.

.. autoclass:: spice.spectrum.lazy_zarr_interpolator.LazyZarrInterpolator
   :members: to_parameters, get_weighted_batch, is_in_bounds

.. autoclass:: spice.spectrum.lazy_zarr_interpolator.IntensityLazyZarrInterpolator
   :members: intensity, flux

.. autoclass:: spice.spectrum.lazy_zarr_interpolator.FluxLazyZarrInterpolator
   :members: intensity, flux

.. autoclass:: spice.spectrum.lazy_zarr_interpolator.GridIndex

.. autoclass:: spice.spectrum.lazy_zarr_interpolator.SparseGridIndex

.. autofunction:: spice.spectrum.generate_zarr_index.write_index_parquet

.. autofunction:: spice.spectrum.generate_zarr_index.build_index_frame_from_zarr

.. autofunction:: spice.spectrum.generate_zarr_index.geometry_labels_from_logg


Synthetic Photometry
--------------------

The available passbands are listed in :doc:`synthetic_photometry`.

.. autoclass:: spice.spectrum.filter.Filter
   :members:

.. autofunction:: spice.spectrum.spectrum.AB_passband_luminosity

.. autofunction:: spice.spectrum.spectrum.ST_passband_luminosity

.. autofunction:: spice.spectrum.spectrum.Vega_passband_luminosity


Plotting
--------

.. autofunction:: spice.plots.plot_mesh.plot_2D

.. autofunction:: spice.plots.plot_mesh.plot_3D

.. autofunction:: spice.plots.plot_mesh.plot_3D_binary

.. autofunction:: spice.plots.plot_mesh.plot_3D_sequence

.. autofunction:: spice.plots.plot_mesh.plot_3D_mesh_and_spectrum

Pulsation visualization helpers (scalar projections of
``mesh.pulsation_velocities``):

.. automodule:: spice.plots.plot_pulsations
   :members: compute_pulsation_scalar, plot_pulsation_map, plot_pulsation_components, plot_pulsation_cross_section, plot_pulsation_phase_grid, animate_pulsation_phase
