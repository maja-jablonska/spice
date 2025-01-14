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
      :type: Float[Array, "n_puls_orders n_fourier_orders 2"]

      Fourier series parameters for pulsations.

   .. py:attribute:: pulsation_axes
      :type: Float[Array, "n_puls_orders 3"]

      Axes of pulsation modes.

   .. py:attribute:: pulsation_angles
      :type: Float[Array, "n_puls_orders"]

      Angles of pulsation modes.

IcosphereModel
~~~~~~~~~~~~~
.. autoclass:: spice.models.mesh_model.IcosphereModel
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: count, index, construct

   .. method:: construct(n_vertices: int, radius: float, mass: float, parameters: Union[float, Float[Array, "n_mesh_elements n_parameters"]], parameter_names: List[str], max_pulsation_mode: int = 3, max_fourier_order: int = 5, override_log_g: bool = True, log_g_index: Optional[int] = None) -> IcosphereModel
      
      Constructs an IcosphereModel with specified stellar and mesh properties.

      This method generates an icosphere mesh and initializes the model with given parameters, including
      stellar properties (mass, radius) and mesh properties (vertices, faces, areas, centers). 
      It also handles the calculation of surface gravity (log g) values if required.

      :param n_vertices: Number of vertices for the icosphere mesh
      :param radius: Radius of the icosphere in solar radii
      :param mass: Mass of the stellar object in solar masses
      :param parameters: Parameters for the model, can be a single value or an array
      :param parameter_names: Names of the parameters, used for identifying log g parameter
      :param max_pulsation_mode: Maximum pulsation mode for the model, defaults to 3
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

.. data:: spice.models.mesh_model.DEFAULT_LOS_VECTOR
   :type: jnp.ndarray
   :value: [0., 1., 0.]

   Default line-of-sight vector (from Y direction).

.. data:: spice.models.mesh_model.DEFAULT_ROTATION_AXIS
   :type: jnp.ndarray 
   :value: [0., 0., 1.]

   Default rotation axis (Z axis).

.. data:: spice.models.mesh_model.DEFAULT_MAX_PULSATION_MODE_PARAMETER
   :type: int
   :value: 3

   Default maximum pulsation mode.

.. data:: spice.models.mesh_model.DEFAULT_FOURIER_ORDER  
   :type: int
   :value: 5

   Default Fourier order for pulsations.

Helper Functions
~~~~~~~~~~~~~~

.. function:: calculate_log_gs(mass: float, d_centers: ArrayLike, rot_velocities: ArrayLike = 0.0)

   Calculates surface gravity (log g) values for mesh elements based on mass, center positions and rotation velocities.

   The surface gravity is calculated using:

   .. math::

      \log g = \log \left(\frac{GM}{R^2} - \frac{v_{rot}^2}{R}\right) - \log(9.80665)

   where:
   - G is the gravitational constant (in solar units)
   - M is the mass in solar masses 
   - R is the radius at each mesh point in solar radii
   - v_rot is the rotation velocity in km/s
   - 9.80665 converts from solar surface gravity units to cgs units (cm/s^2)

   :param mass: Mass of the star in solar masses
   :param d_centers: Center positions of mesh elements relative to star center
   :param rot_velocities: Rotation velocities of mesh elements in km/s, defaults to 0.0
   :return: Array of log g values for each mesh element


Mesh Transformations
------------------

Transform Functions
~~~~~~~~~~~~~~~~~

.. function:: transform(mesh: MeshModel, vector: Float[Array, "3"]) -> MeshModel

   Transform the position of a mesh model based on a given vector.

   This function applies a transformation to the mesh model's position by updating its center
   with the provided vector. PHOEBE models are considered read-only within SPICE.

   :param mesh: The mesh model to be transformed
   :param vector: The vector by which the mesh's position is to be updated
   :return: The transformed mesh model with its position updated
   :raises ValueError: If the mesh model is an instance of PhoebeModel

.. function:: update_parameter(mesh: MeshModel, parameter: Union[str, int, ArrayLike], parameter_values: Float[Array, "n_mesh_elements n_parameters"], parameter_names: List[str] = None) -> MeshModel

   Update a specific parameter or set of parameters in the mesh model.

   This function allows updating one or multiple parameters of the mesh model. It can handle
   parameter specification by name (string), index (integer), or an array-like of indices.

   :param mesh: The mesh model to be updated
   :param parameter: The parameter(s) to update - can be name, index or array of indices
   :param parameter_values: The new value(s) for the specified parameter(s)
   :param parameter_names: List of parameter names used for the model
   :return: The updated mesh model
   :raises ValueError: If parameter name not found or mesh is PhoebeModel

.. function:: update_parameters(mesh: MeshModel, parameters: Union[List[str], List[int]], parameter_values: Float[Array, "n_mesh_elements n_parameters"], parameter_names: List[str] = None) -> MeshModel

   Update multiple parameters in the mesh model simultaneously.

   More efficient than calling update_parameter multiple times when updating several parameters at once.

   :param mesh: The mesh model to be updated
   :param parameters: List of parameter names or indices to update
   :param parameter_values: New values for the specified parameters
   :param parameter_names: List of parameter names used for the model
   :return: The updated mesh model
   :raises ValueError: If parameter names not found or mesh is PhoebeModel

Rotation Functions
~~~~~~~~~~~~~~~~

.. function:: evaluate_rotation(mesh: MeshModel, t: float) -> MeshModel

   Evaluate the rotation of a mesh model at a specific time.

   Updates the mesh model's rotation parameters based on the given time.

   :param mesh: The mesh model to evaluate rotation for
   :param t: The time at which to evaluate the rotation (seconds)
   :return: The mesh model with updated rotation parameters
   :raises ValueError: If mesh is PhoebeModel

.. function:: evaluate_body_orbit(m: MeshModel, orbital_velocity: float) -> MeshModel

   Evaluate the effects of an orbit on a mesh model.

   :param m: Mesh model of an orbiting body
   :param orbital_velocity: Orbital velocity in km/s
   :return: Mesh model with updated parameters

Pulsation Functions
~~~~~~~~~~~~~~~~~

.. function:: add_pulsations(m: MeshModel, m_orders: Float[Array, "n_pulsations"], n_degrees: Float[Array, "n_pulsations"], periods: Float[Array, "n_pulsations"], fourier_series_parameters: Float[Array, "n_pulsations n_terms 2"], pulsation_axes: Float[Array, "n_pulsations 3"] = None, pulsation_angles: Float[Array, "n_pulsations"] = None) -> MeshModel

   Adds multiple pulsation effects to a mesh model using spherical harmonics and Fourier series parameters.

   :param m: The mesh model to add pulsation effects to
   :param m_orders: Array of orders (m) of the spherical harmonics
   :param n_degrees: Array of degrees (n) of the spherical harmonics  
   :param periods: Array of pulsation periods in seconds
   :param fourier_series_parameters: Array of dynamic parameters for the Fourier series
   :param pulsation_axes: Array of pulsation axes (defaults to rotation axis)
   :param pulsation_angles: Array of pulsation angles (defaults to zero)
   :return: The mesh model with updated pulsation parameters
   :raises ValueError: If mesh is PhoebeModel or input arrays have inconsistent lengths

.. function:: reset_pulsations(m: MeshModel) -> MeshModel

   Resets the pulsation parameters of a mesh model to non-pulsating model values.

   :param m: The mesh model to reset pulsation parameters for
   :return: The mesh model with pulsation parameters reset
   :raises ValueError: If mesh is PhoebeModel

.. function:: evaluate_pulsations(m: MeshModel, t: float) -> MeshModel

   Evaluates and updates the mesh model with pulsation effects at a specific time.

   Calculates pulsation effects using Fourier series parameters for both static and dynamic components.
   Updates the mesh with calculated offsets and velocities.

   :param m: The mesh model to evaluate pulsations for
   :param t: The time at which to evaluate the pulsations
   :return: The mesh model updated with pulsation effects
   :raises ValueError: If mesh is PhoebeModel


Mesh View
------------------

Mesh View Functions
~~~~~~~~~~~~~~~~~

.. function:: get_grid_spans(m1: MeshModel, m2: MeshModel, n_cells_array: ArrayLike) -> ArrayLike

   Calculate grid cell spans for different grid sizes.

   For each number of cells in n_cells_array, calculates the span (width/height) of grid cells
   that would cover the projected area of both meshes. Returns the minimum of x and y spans
   to ensure square grid cells.

   :param m1: First mesh model with cast_vertices and faces
   :param m2: Second mesh model with cast_vertices and faces
   :param n_cells_array: Array of different grid cell counts to try
   :return: Array of grid cell spans corresponding to each n_cells value

.. function:: get_mesh_view(mesh: MeshModel, los_vector: Float[Array, "3"]) -> MeshModel

   Cast 3D vectors of centers and center velocities to the line-of-sight.

   :param mesh: Properties to be cast (n, 3)
   :param los_vector: LOS vector (3,)
   :return: mesh with updated los_vector, mus, and los_velocities

.. function:: visible_area(vertices1: Float[Array, "n_vertices 3"], vertices2: Float[Array, "n_vertices 3"]) -> Float[Array, "n_vertices"]

   Calculate visible area between two sets of vertices.

   :param vertices1: First set of vertices
   :param vertices2: Second set of vertices
   :return: Area of visible region

.. function:: resolve_occlusion(m1: MeshModel, m2: MeshModel, grid: Grid) -> MeshModel

   Calculate the occlusion of m1 by m2.

   :param m1: occluded mesh model
   :param m2: occluding mesh model  
   :param grid: grid for calculation optimization
   :return: m1 with updated visible areas


Spots
------------------

Spot Functions
~~~~~~~~~~~~~~~~~

.. function:: add_spot(mesh: MeshModel, spot_center_theta: float, spot_center_phi: float, spot_radius: float, parameter_delta: float, parameter_index: int, smoothness: float = 1.0) -> MeshModel

   Add a spot to a mesh model based on spherical coordinates and smoothness parameters.

   This function applies a modification to the mesh model's parameters to simulate the presence of a spot. The spot
   is defined by its center (in spherical coordinates), its radius, and a differential parameter that quantifies the
   change induced by the spot. The smoothness parameter allows for a gradual transition at the spot's edges.

   :param mesh: The mesh model to which the spot will be added
   :param spot_center_theta: The theta (inclination) coordinate of the spot's center, in radians
   :param spot_center_phi: The phi (azimuthal) coordinate of the spot's center, in radians 
   :param spot_radius: The angular radius of the spot, in radians
   :param parameter_delta: The difference in the parameter value to be applied within the spot
   :param parameter_index: The index of the parameter in the mesh model that will be modified
   :param smoothness: Factor controlling the smoothness of the spot's edge, defaults to 1.0
   :return: The modified mesh model with the spot applied
   :raises ValueError: If mesh is a PhoebeModel

.. function:: add_spots(mesh: MeshModel, spot_center_thetas: Float[Array, "n_spots"], spot_center_phis: Float[Array, "n_spots"], spot_radii: Float[Array, "n_spots"], parameter_deltas: Float[Array, "n_spots"], parameter_indices: Int[Array, "n_spots"], smoothness: Float[Array, "n_spots"] = None) -> MeshModel

   Add multiple spots to a mesh model based on spherical coordinates and smoothness parameters.

   :param mesh: The mesh model to which the spots will be added
   :param spot_center_thetas: Array of theta coordinates of spot centers, in radians
   :param spot_center_phis: Array of phi coordinates of spot centers, in radians
   :param spot_radii: Array of angular radii of spots, in radians
   :param parameter_deltas: Array of parameter value differences for each spot
   :param parameter_indices: Array of parameter indices to modify for each spot
   :param smoothness: Array of edge smoothness factors for each spot
   :return: The modified mesh model with all spots applied
   :raises ValueError: If mesh is a PhoebeModel

.. function:: add_spherical_harmonic_spot(mesh: MeshModel, m_order: Union[Int, Float], n_degree: Union[Int, Float], param_delta: Float, param_index: Float, tilt_axis: Float[Array, "3"] = None, tilt_degree: Float = None) -> MeshModel

   Add a spherical harmonic variation to a parameter of the mesh model.

   Creates a spot-like feature using spherical harmonic function Y_n^m(θ,φ) to modify surface parameters.

   :param mesh: The mesh model to modify
   :param m_order: Order (m) of spherical harmonic, must be ≤ n_degree
   :param n_degree: Degree (n) of spherical harmonic
   :param param_delta: Maximum amplitude of parameter variation
   :param param_index: Index of parameter to modify
   :param tilt_axis: Optional axis for tilting the pattern
   :param tilt_degree: Optional tilt angle in degrees
   :return: Modified mesh model with spherical harmonic variation
   :raises ValueError: If m_order > n_degree or mesh is PhoebeModel

.. function:: add_spherical_harmonic_spots(mesh: MeshModel, m_orders: Float[Array, "n_orders"], n_degrees: Float[Array, "n_orders"], param_deltas: Float[Array, "n_orders"], param_indices: Float[Array, "n_orders"], tilt_axes: Optional[Float[Array, "n_orders 3"]] = None, tilt_angles: Optional[Float[Array, "n_orders"]] = None) -> MeshModel

   Add multiple spherical harmonic spots to a mesh model.

   :param mesh: The mesh model to modify
   :param m_orders: Array of m indices for spherical harmonics
   :param n_degrees: Array of n indices for spherical harmonics
   :param param_deltas: Array of modification strengths
   :param param_indices: Array of parameter indices to modify
   :param tilt_axes: Optional array of tilt axes for each spot
   :param tilt_angles: Optional array of tilt angles in radians
   :return: Modified mesh model with all harmonic spots
   :raises ValueError: If mesh is PhoebeModel or tilt parameters mismatched

