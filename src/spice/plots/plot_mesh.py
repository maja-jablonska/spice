import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import art3d
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from spice.models import MeshModel
from jax.typing import ArrayLike
import cmasher as cmr

PLOT_MODES = ['MESH', 'POINTS']
COLORMAP_PROPERTIES = ['mus', 'los_velocities', 'cast_areas', 'visible_cast_areas', 'log_gs']
DEFAULT_PROPERTY = 'mus'

DEFAULT_PLOT_PROPERTY_LABELS = {
    'mus': r'$\mu$',
    'los_velocities': 'LOS velocity [km/s]',
    'cast_areas': 'cast area [cm$^2$]',
    'visible_cast_areas': 'visible cast area [cm$^2$]',
    'log_gs': 'log g'
}

DEFAULT_PROPERTY_CMAPS = {
    'los_velocities': 'cmr.redshift_r',
}

DEFAULT_CMAP = 'cmr.bubblegum'


def _evaluate_to_be_mapped_property(mesh: MeshModel,
                                    property: Union[str, int] = DEFAULT_PROPERTY,
                                    property_label: Optional[str] = None) -> Tuple[ArrayLike, str]:
    """
    Evaluate the property to be mapped in visualization.
    
    Parameters
    ----------
    mesh : MeshModel
        The mesh model object
    property : Union[str, int], default: 'mus'
        Property name or index to visualize
    property_label : Optional[str], default: None
        Custom label for the property. If None, a default label is used
        
    Returns
    -------
    Tuple[ArrayLike, str]
        A tuple containing the property values and the property label
    
    Raises
    ------
    ValueError
        If the property is invalid or not found
    """
    if isinstance(property, str):
        if property in DEFAULT_PLOT_PROPERTY_LABELS:
            to_be_mapped = getattr(mesh, property)
            if property_label is None:
                property_label = DEFAULT_PLOT_PROPERTY_LABELS[property]
        elif hasattr(mesh, property):
            to_be_mapped = getattr(mesh, property)
            if property_label is None:
                property_label = property
        else:
            raise ValueError(f'Invalid property {property} - must be an attribute of MeshModel or one of {list(DEFAULT_PLOT_PROPERTY_LABELS.keys())}')
    elif isinstance(property, int):
        if property > mesh.parameters.shape[-1]-1:
            raise ValueError(f'Invalid property index {property} - must be smaller than {mesh.parameters.shape[-1]}')
        to_be_mapped = mesh.parameters[:, property]
        if property_label is None:
            property_label = ''
    else:
        raise ValueError(f"Property must be either of type str or int")
    
    return to_be_mapped, property_label


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

def smart_save(anim, filename, fps=20):
    ext = filename.split('.')[-1].lower()

    if ext == 'gif':
        # 1st choice – Pillow (always available, rock-solid)
        try:
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer)
            return
        except Exception as e:
            print(f'PillowWriter failed: {e}, falling back to ffmpeg…')

        # 2nd choice – ffmpeg with RGB fallback
        extra = ['-vf',
                 'format=rgb24,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                 '-pix_fmt', 'rgb24']
        writer = FFMpegWriter(fps=fps, codec='gif', extra_args=extra)
        anim.save(filename, writer=writer)

    else:  # mp4/webm/apng/…
        if FFMpegWriter.isAvailable():
            writer = FFMpegWriter(fps=fps, codec='libx264')
            anim.save(filename, writer=writer)
        else:
            anim.save(filename, fps=fps)     # Pillow fallback


def plot_3D(mesh: MeshModel,
            property: Union[str, int] = DEFAULT_PROPERTY,
            axes: Optional[Tuple[plt.figure, plt.axes]] = None,
            cmap: Optional[str] = None,
            property_label: Optional[str] = None,
            mode: str = 'MESH',
            update_colorbar: bool = True,
            draw_los_vector: bool = True,
            draw_rotation_axis: bool = True):
    """
    Create a 3D visualization of a mesh model.
    
    Parameters
    ----------
    mesh : MeshModel
        The mesh model to visualize
    property : Union[str, int], default: 'mus'
        Property to color the mesh by (either attribute name or parameter index)
    axes : Optional[Tuple[plt.figure, plt.axes]], default: None
        Custom figure and axes to plot on. If None, creates new ones
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
    mode : str, default: 'MESH'
        Visualization mode - 'MESH' (triangular mesh) or 'POINTS' (scatter)
    update_colorbar : bool, default: True
        Whether to add/update the colorbar
    draw_los_vector : bool, default: True
        Whether to draw the line-of-sight vector
    draw_rotation_axis : bool, default: True
        Whether to draw the rotation axis
        
    Returns
    -------
    Tuple[plt.figure, plt.axes]
        Figure and axes with the plot
    
    Raises
    ------
    ValueError
        If the mode is invalid
    """
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of {PLOT_MODES}. Got {mode.upper()}')
    mode = mode.upper()
    
    # Get the property data and label
    to_be_mapped, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)
    
    # Set up the colormap
    if cmap is None:
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)

    # Create or get figure and axes
    if axes is None:
        fig = plt.figure(figsize=(10, 12))
        spec = fig.add_gridspec(10, 12)
        plot_ax = fig.add_subplot(spec[:, :11], projection='3d')
    else:
        try:
            fig, plot_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or a tuple (figure, axes) for the figure and plot axis")
    
    # Set up axes limits and labels
    axes_lim = 1.5*mesh.radius
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)

    # Draw vectors if requested
    if draw_los_vector:
        normalized_los_vector = mesh.los_vector/np.linalg.norm(mesh.los_vector)
        plot_ax.quiver(*(-2.0*mesh.radius*normalized_los_vector), *(mesh.radius*normalized_los_vector),
                   color='red', linewidth=3., label='LOS vector')
    
    if draw_rotation_axis:
        # Choose arrow color based on background color
        if plt.style.available and plt.rcParams['axes.facecolor'] == 'black':
            arrow_color = 'white'
        else:
            arrow_color = 'black'
        
        normalized_rotation_axis = mesh.rotation_axis/np.linalg.norm(mesh.rotation_axis)
        plot_ax.quiver(*(0.75*mesh.radius*normalized_rotation_axis), *(mesh.radius*normalized_rotation_axis),
                    color=arrow_color, linewidth=3., label='Rotation axis')
        
    # Add legend if vectors were drawn
    if draw_los_vector or draw_rotation_axis:
        plot_ax.legend()

    # Set up colormap
    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Visualize the data according to the mode
    if mode == 'MESH':
        vs2 = mesh.mesh_elements
        face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
        p = art3d.Poly3DCollection(vs2, facecolors=face_colors, edgecolor="black", linewidths=0.01)
        plot_ax.add_collection(p)
        mappable.set_array([])
    else:  # mode == 'POINTS'
        p = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                            c=to_be_mapped, cmap=cmap, norm=norm)
        
    # Add colorbar
    if update_colorbar:
        cbar = fig.colorbar(mappable, shrink=0.45, pad=0.125, ax=plot_ax)
        cbar.set_label(cbar_label, fontsize=12)
        
    return fig, plot_ax


def plot_3D_binary(mesh1: MeshModel,
                   mesh2: MeshModel,
                   property: Union[str, int] = DEFAULT_PROPERTY,
                   axes: Optional[Tuple[plt.figure, plt.axes]] = None,
                   cmap: Optional[str] = None,
                   property_label: Optional[str] = None,
                   mode: str = 'MESH',
                   update_colorbar: bool = True,
                   scale_radius: float = 1.0,
                   draw_los_vector: bool = True,
                   draw_rotation_axes: bool = True):
    """
    Create a 3D visualization of a binary system with two mesh models.
    
    Parameters
    ----------
    mesh1 : MeshModel
        First mesh model in the binary system
    mesh2 : MeshModel
        Second mesh model in the binary system
    property : Union[str, int], default: 'mus'
        Property to color the meshes by (attribute name or parameter index)
    axes : Optional[Tuple[plt.figure, plt.axes]], default: None
        Custom figure and axes to plot on. If None, creates new ones
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
    mode : str, default: 'MESH'
        Visualization mode - 'MESH' (triangular mesh) or 'POINTS' (scatter)
    update_colorbar : bool, default: True
        Whether to add/update the colorbar
    scale_radius : float, default: 1.0
        Scale factor for both stellar radii
    draw_los_vector : bool, default: True
        Whether to draw the line-of-sight vector
    draw_rotation_axes : bool, default: True
        Whether to draw the rotation axes of both stars
        
    Returns
    -------
    Tuple[plt.figure, plt.axes]
        Figure and axes with the plot
    
    Raises
    ------
    ValueError
        If the mode is invalid
    """
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of {PLOT_MODES}. Got {mode.upper()}')
    mode = mode.upper()
    
    # Get property data and label
    to_be_mapped1, cbar_label = _evaluate_to_be_mapped_property(mesh1, property, property_label)
    to_be_mapped2, _ = _evaluate_to_be_mapped_property(mesh2, property, property_label)
    to_be_mapped = np.concatenate([to_be_mapped1, to_be_mapped2])
    
    # Set up colormap
    if cmap is None:
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)

    # Create or get figure and axes
    if axes is None:
        fig = plt.figure(figsize=(10, 12))
        spec = fig.add_gridspec(10, 12)
        plot_ax = fig.add_subplot(spec[:, :11], projection='3d')
    else:
        try:
            fig, plot_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or a tuple (figure, axes) for the figure and plot axis")
    
    # Set up axes limits and labels
    axes_lim = np.max(np.abs((mesh1.radius+mesh1.center)-(mesh2.center-mesh2.radius)))
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)

    # Draw vectors if requested
    if draw_los_vector:
        normalized_los_vector = mesh1.los_vector/np.linalg.norm(mesh1.los_vector)
        plot_ax.quiver(*(-1.5*axes_lim*normalized_los_vector), *((axes_lim*normalized_los_vector)/2),
                color='red', linewidth=3., label='LOS vector')
        
    if draw_rotation_axes:
        normalized_rotation_axis1 = mesh1.rotation_axis/np.linalg.norm(mesh1.rotation_axis)
        normalized_rotation_axis2 = mesh2.rotation_axis/np.linalg.norm(mesh2.rotation_axis)
        
        plot_ax.quiver(*(mesh1.center+normalized_rotation_axis1*mesh1.radius*scale_radius), 
                       *(mesh1.radius*normalized_rotation_axis1*scale_radius),
                       color='black', linewidth=3., label='Rotation axis of mesh1')
        plot_ax.quiver(*(mesh2.center+normalized_rotation_axis2*mesh2.radius*scale_radius), 
                       *(mesh2.radius*normalized_rotation_axis2*scale_radius),
                       color='blue', linewidth=3., label='Rotation axis of mesh2')
    
    # Add legend if vectors were drawn
    if draw_los_vector or draw_rotation_axes:
        plot_ax.legend()

    # Set up colormap normalization
    norm = mpl.colors.Normalize(vmin=np.nanmin(to_be_mapped), vmax=np.nanmax(to_be_mapped))
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Visualize the data according to the mode
    if mode == 'MESH':
        # First mesh
        vs2_1 = mesh1.center+(mesh1.mesh_elements-mesh1.center)*scale_radius
        face_colors1 = mpl.colormaps[cmap](norm(to_be_mapped1))
        p1 = art3d.Poly3DCollection(vs2_1, facecolors=face_colors1, edgecolor="black", linewidths=0.01)
        
        # Second mesh
        vs2_2 = mesh2.center+(mesh2.mesh_elements-mesh2.center)*scale_radius
        face_colors2 = mpl.colormaps[cmap](norm(to_be_mapped2))
        p2 = art3d.Poly3DCollection(vs2_2, facecolors=face_colors2, edgecolor="black", linewidths=0.01)
        
        plot_ax.add_collection(p1)
        plot_ax.add_collection(p2)
        mappable.set_array([])
    else:  # mode == 'POINTS'
        # First mesh as points
        centers1 = mesh1.center+(mesh1.centers-mesh1.center)*scale_radius
        p1 = plot_ax.scatter(centers1[:, 0], centers1[:, 1], centers1[:, 2],
                             c=to_be_mapped1, cmap=cmap, norm=norm)
        
        # Second mesh as points
        centers2 = mesh2.center+(mesh2.centers-mesh2.center)*scale_radius
        p2 = plot_ax.scatter(centers2[:, 0], centers2[:, 1], centers2[:, 2],
                             c=to_be_mapped2, cmap=cmap, norm=norm)
        
    # Add colorbar
    if update_colorbar:
        cbar = fig.colorbar(mappable, shrink=0.45, pad=0.125, ax=plot_ax)
        cbar.set_label(cbar_label, fontsize=12)

    return fig, plot_ax


def plot_3D_sequence(meshes: List[MeshModel],
                     property: Union[str, int] = DEFAULT_PROPERTY,
                     timestamps: Optional[ArrayLike] = None,
                     axes: Optional[Tuple[plt.figure, List[plt.axes], plt.axes]] = None,
                     cmap: Optional[str] = None,
                     property_label: Optional[str] = None,
                     timestamp_label: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 10),
                     mode: str = 'MESH',
                     draw_los_vector: bool = True,
                     draw_rotation_axes: bool = True,
                     tight_layout: bool = True):
    """
    Create a visualization of a sequence of mesh models in a grid layout.
    
    Parameters
    ----------
    meshes : List[MeshModel]
        List of mesh models to visualize in sequence
    property : Union[str, int], default: 'mus'
        Property to color the meshes by (attribute name or parameter index)
    timestamps : Optional[ArrayLike], default: None
        List of timestamps corresponding to each mesh
    axes : Optional[Tuple[plt.figure, List[plt.axes], plt.axes]], default: None
        Custom figure, plot axes list, and colorbar axis to plot on. If None, creates new ones
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
    timestamp_label : Optional[str], default: None
        Label to accompany timestamps (e.g., "hours", "days")
    figsize : Tuple[int, int], default: (12, 10)
        Figure size if creating a new figure
    mode : str, default: 'MESH'
        Visualization mode - 'MESH' (triangular mesh) or 'POINTS' (scatter)
    draw_los_vector : bool, default: True
        Whether to draw the line-of-sight vector
    draw_rotation_axes : bool, default: True
        Whether to draw the rotation axes of both stars
    tight_layout : bool, default: True
        Whether to call plt.tight_layout()
        
    Returns
    -------
    Tuple[plt.figure, List[plt.axes], plt.axes]
        Figure, list of plot axes, and colorbar axis
    
    Raises
    ------
    ValueError
        If the mode is invalid
    """
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of {PLOT_MODES}. Got {mode.upper()}')
    mode = mode.upper()

    # Setup timestamp label
    timestamp_label = timestamp_label or ''
    
    # Get property data
    _, cbar_label = _evaluate_to_be_mapped_property(meshes[0], property, property_label)
    to_be_mapped_arrays = [_evaluate_to_be_mapped_property(mesh, property, property_label)[0] for mesh in meshes]
    to_be_mapped_arrays_concatenated = np.concatenate(to_be_mapped_arrays)

    # Set up the colormap
    if cmap is None:
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)

    # Determine axis limits and subplot layout
    axes_lim = 1.5*max([mesh.radius for mesh in meshes])
    num_plots = len(meshes)

    # Create or get figure and axes
    if axes is None:
        # Determine the number of rows and columns for the subplots
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))

        # Create a figure with gridspec
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(10*rows, cols + 1, width_ratios=[*([1]*cols), 0.05])
        plot_axes = [fig.add_subplot(gs[10*(i // cols):10*((i // cols)+1), i % cols], projection='3d') 
                     for i in range(num_plots)]

        # Add the colorbar in the last column with more space to the right
        cax_limit = max(rows, 5)
        # Add a small horizontal padding to move the colorbar more to the right
        gs_right = cols + 0.2  # Add padding to position
        cbar_ax = fig.add_subplot(gs[cax_limit:-cax_limit, -1])
        cbar_ax.set_position([cbar_ax.get_position().x0 + 0.02, 
                             cbar_ax.get_position().y0,
                             cbar_ax.get_position().width,
                             cbar_ax.get_position().height])
    else:
        try:
            fig, plot_axes, cbar_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or a tuple (figure, list_of_axes, colorbar_axis)")
    
    # Set up common axis properties
    for plot_ax in plot_axes:
        plot_ax.set_xlim3d(-axes_lim, axes_lim)
        plot_ax.set_ylim3d(-axes_lim, axes_lim)
        plot_ax.set_zlim3d(-axes_lim, axes_lim)
        plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=10)
        plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=10)
        plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=10)

    # Set up color normalization
    norm = mpl.colors.Normalize(vmin=to_be_mapped_arrays_concatenated.min(), 
                               vmax=to_be_mapped_arrays_concatenated.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot each mesh
    for plot_ax, (to_be_mapped, (i, mesh)) in zip(plot_axes, zip(to_be_mapped_arrays, enumerate(meshes))):
        # Draw los vector and rotation axis
        normalized_los_vector = mesh.los_vector/np.linalg.norm(mesh.los_vector)
        normalized_rotation_axis = mesh.rotation_axis/np.linalg.norm(mesh.rotation_axis)

        if draw_los_vector:
            plot_ax.quiver(*(-2.0*mesh.radius*normalized_los_vector), *(mesh.radius*normalized_los_vector),
                        color='red', linewidth=3., label='LOS vector')
        if draw_rotation_axes:
            plot_ax.quiver(*(0.75*mesh.radius*normalized_rotation_axis), *(mesh.radius*normalized_rotation_axis),
                        color='black', linewidth=3., label='Rotation axis')

        # Visualize according to mode
        if mode == 'MESH':
            vs2 = mesh.vertices[mesh.faces.astype(int)]
            face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
            p = art3d.Poly3DCollection(vs2, facecolors=face_colors, edgecolor="black", linewidth=0.1)
            plot_ax.add_collection(p)
        else:  # mode == 'POINTS'
            p = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                                c=to_be_mapped, cmap=cmap, norm=norm)
            
        # Add timestamp if available
        if timestamps is not None:
            plot_ax.set_title(f"Time: {timestamps[i]:.2f} {timestamp_label}", pad=1.0)
            
    # Add legend to the last subplot
    if draw_los_vector or draw_rotation_axes:
        plot_axes[int(cols-1)].legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    
    # Set up colorbar
    mappable.set_array([])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    if tight_layout:
        plt.tight_layout()

    return fig, plot_axes, cbar_ax


def plot_2D(mesh: MeshModel,
            property: Union[str, int] = DEFAULT_PROPERTY,
            axes: Optional[Tuple[plt.figure, plt.axes, plt.axes]] = None,
            cmap: Optional[str] = None,
            x_index: int = 0,
            y_index: int = 1,
            property_label: Optional[str] = None):
    """
    Create a 2D visualization of a mesh model.
    
    Parameters
    ----------
    mesh : MeshModel
        The mesh model to visualize
    property : Union[str, int], default: 'mus'
        Property to color the mesh by (attribute name or parameter index)
    axes : Optional[Tuple[plt.figure, plt.axes, plt.axes]], default: None
        Custom figure, plot axis, and colorbar axis to plot on. If None, creates new ones
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    x_index : int, default: 0
        Index for the x-axis (0=X, 1=Y, 2=Z)
    y_index : int, default: 1
        Index for the y-axis (0=X, 1=Y, 2=Z)
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
        
    Returns
    -------
    Tuple[plt.figure, plt.axes, plt.axes]
        Figure, plot axis, and colorbar axis
    
    Raises
    ------
    ValueError
        If x_index or y_index are invalid
    """
    # Set up the colormap
    if cmap is None:
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)
    
    # Validate axis indices
    if x_index == y_index:
        raise ValueError('x_index and y_index cannot be the same')
    elif x_index not in (0, 1, 2) or y_index not in (0, 1, 2):
        raise ValueError('x_index and y_index must be 0, 1, or 2 (corresponding to X, Y, Z)')
    
    # Create or get figure and axes
    if axes is None:
        fig = plt.figure(figsize=(6, 5))
        spec = fig.add_gridspec(10, 12)
        plot_ax = fig.add_subplot(spec[:, :11])
        cbar_ax = fig.add_subplot(spec[2:8, 11])
    else:
        try:
            fig, plot_ax, cbar_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or a tuple (figure, plot_axis, colorbar_axis)")
    
    # Set up axis labels
    xy_labels = ['$X [R_\\odot]$', '$Y [R_\\odot]$', '$Z [R_\\odot]$']
    
    # Get property data
    to_be_mapped, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)
    
    # Only show points with positive mu (visible from observer)
    positive_mu_mask = mesh.mus > 0

    # Create the scatter plot
    scatter = plot_ax.scatter(mesh.centers[positive_mu_mask, x_index], 
                           mesh.centers[positive_mu_mask, y_index],
                           c=to_be_mapped[positive_mu_mask], 
                           cmap=cmap)
    
    # Set labels
    plot_ax.set_xlabel(xy_labels[x_index], fontsize=14)
    plot_ax.set_ylabel(xy_labels[y_index], fontsize=14)
    
    # Add colorbar
    norm = scatter.norm
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)

    return fig, plot_ax, cbar_ax


def plot_3D_mesh_and_spectrum(mesh, spectrum, wavelengths, 
                            property=DEFAULT_PROPERTY, 
                            cmap=None, 
                            property_label=None,
                            figsize=(15, 5),
                            mode='MESH',
                            draw_los_vector=True,
                            draw_rotation_axis=True,
                            linewidth=0.1,
                            axes_lim=None,
                            timestamp=None,
                            timestamp_label=None) -> Tuple[plt.figure, plt.axes, plt.axes]:
    """
    Create an animation showing both a 3D mesh model and its corresponding spectrum over time.
    
    Parameters
    ----------
    meshes : List[MeshModel]
        List of mesh models to animate
    spectra : ndarray
        Array of spectra corresponding to each mesh, shape (n_frames, n_wavelengths)
    wavelengths : ndarray
        Wavelength array for the spectra
    property : Union[str, int], default: DEFAULT_PROPERTY
        Property to color the mesh by (attribute name or parameter index)
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
    filename : str, default: 'mesh_and_spectra_animation.mp4'
        Output filename for the animation
    figsize : Tuple[int, int], default: (12, 8)
        Figure size
    mode : str, default: 'MESH'
        Visualization mode - 'MESH' (triangular mesh) or 'POINTS' (scatter)
    draw_los_vector : bool, default: True
        Whether to draw the line-of-sight vector
    draw_rotation_axis : bool, default: True
        Whether to draw the rotation axis
    linewidth : float, default: 0.1
        Line width for mesh edges
    axes_lim : Optional[float], default: None
        Limit for all axes. If None, calculated from mesh radius
    timestamps : Optional[ArrayLike], default: None
        List of timestamps corresponding to each mesh
    timestamp_label : Optional[str], default: None
        Label to accompany timestamps (e.g., "hours", "days")
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the mesh and spectrum plots
    mesh_ax : matplotlib.axes.Axes
        The axis containing the mesh plot
    spec_ax : matplotlib.axes.Axes
        The axis containing the spectrum plot
    
    Raises
    ------
    ValueError
        If the mode is invalid or if spectra and meshes have different lengths
    """
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of {PLOT_MODES}. Got {mode.upper()}')
    mode = mode.upper()
    
    # Setup timestamp label
    timestamp_label = timestamp_label or ''
    
    to_be_mapped, label = _evaluate_to_be_mapped_property(mesh, property, property_label)
    cbar_label = label
    
    # Set up the colormap
    if cmap is None:
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)
    
    # Determine axis limits
    if axes_lim is None:
        axes_lim = 1.5 * mesh.radius
    
    # Create figure with two subplots: 3D mesh and spectrum
    fig = plt.figure(figsize=figsize)
    # Adjust GridSpec to make the colorbar closer to the mesh plot
    gs = plt.GridSpec(4, 4, width_ratios=[1, 0.05, 0.05, 2])

    # 3D mesh subplot
    mesh_ax = fig.add_subplot(gs[:, 0], projection='3d')
    mesh_ax.set_xlim3d(-axes_lim, axes_lim)
    mesh_ax.set_ylim3d(-axes_lim, axes_lim)
    mesh_ax.set_zlim3d(-axes_lim, axes_lim)
    mesh_ax.set_xlabel('$X [R_\\odot]$', fontsize=10)
    mesh_ax.set_ylabel('$Y [R_\\odot]$', fontsize=10)
    mesh_ax.set_zlabel('$Z [R_\\odot]$', fontsize=10)
    
    # Spectrum subplot
    spec_ax = fig.add_subplot(gs[1:3, 3])
    spec_ax.set_xlabel('Wavelength [$\AA$]', fontsize=10)
    spec_ax.set_ylabel('Flux [erg/s/cm$^3$]', fontsize=10)
    
    # Set up color normalization for the mesh
    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), 
                               vmax=to_be_mapped.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Add colorbar for the mesh in the middle column
    # Reduced spacing between mesh and colorbar
    cbar_ax = fig.add_subplot(gs[1:3, 1])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    # Find min/max for spectrum y-axis
    spec_min = np.min(spectrum)
    spec_max = np.max(spectrum)
    spec_padding = 0.05 * (spec_max - spec_min)
    spec_ax.set_ylim(spec_min - spec_padding, spec_max + spec_padding)
    
        
    # Reset axis properties
    mesh_ax.set_xlim3d(-axes_lim, axes_lim)
    mesh_ax.set_ylim3d(-axes_lim, axes_lim)
    mesh_ax.set_zlim3d(-axes_lim, axes_lim)
    mesh_ax.set_xlabel('$X [R_\\odot]$', fontsize=10)
    mesh_ax.set_ylabel('$Y [R_\\odot]$', fontsize=10)
    mesh_ax.set_zlabel('$Z [R_\\odot]$', fontsize=10)
    
    spec_ax.plot(wavelengths, spectrum, color='black')
        
    # Draw los vector and rotation axis
    if draw_los_vector:
        normalized_los_vector = mesh.los_vector/np.linalg.norm(mesh.los_vector)
        mesh_ax.quiver(*(-2.0*mesh.radius*normalized_los_vector), *(mesh.radius*normalized_los_vector),
                    color='red', linewidth=3., label='LOS vector')
    
    if draw_rotation_axis:
        normalized_rotation_axis = mesh.rotation_axis/np.linalg.norm(mesh.rotation_axis)
        mesh_ax.quiver(*(0.75*mesh.radius*normalized_rotation_axis), *(mesh.radius*normalized_rotation_axis),
                    color='black', linewidth=3., label='Rotation axis')
    
    # Visualize mesh according to mode
    if mode == 'MESH':
        vs2 = mesh.vertices[mesh.faces.astype(int)]
        face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
        mesh_collection = art3d.Poly3DCollection(vs2, facecolors=face_colors, 
                                                edgecolor="black", linewidths=linewidth)
        mesh_ax.add_collection(mesh_collection)
    else:  # mode == 'POINTS'
        mesh_collection = mesh_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                                        c=to_be_mapped, cmap=cmap, norm=norm)
        
        # Add timestamp if available
    if timestamp is not None:
        title = f"Time: {timestamp:.2f} {timestamp_label}"
        fig.suptitle(title, y=0.85)
    if draw_los_vector and draw_rotation_axis:
        mesh_ax.legend(loc='upper right', fontsize=10)
        
    plt.tight_layout()
    return fig, mesh_ax, spec_ax


def animate_single_star(meshes: List[MeshModel],
                        filename: str,
                        property: Union[str, int] = DEFAULT_PROPERTY,
                        timestamps: Optional[ArrayLike] = None,
                        cmap: Optional[str] = None,
                        property_label: Optional[str] = None,
                        timestamp_label: Optional[str] = 'days',
                        mode: str = 'MESH',
                        skip_frames: int = 1,
                        linewidth: float = 0.01,
                        figure_size: Tuple[int, int] = (10, 10),
                        draw_los_vector: bool = True,
                        draw_rotation_axis: bool = True,
                        view_angles: Optional[Tuple[float, float]] = None,
                        fixed_norm: bool = True):
    """
    Create an animation of a single star model over time (e.g., pulsating star).
    
    Parameters
    ----------
    meshes : List[MeshModel]
        List of mesh models representing the star at different time steps
    filename : str
        Output filename for the animation
    property : Union[str, int], default: 'mus'
        Property to color the mesh by (attribute name or parameter index)
    timestamps : Optional[ArrayLike], default: None
        List of timestamps corresponding to each mesh
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
    timestamp_label : Optional[str], default: "days""
        Label to accompany timestamps (e.g., "hours", "days")
    mode : str, default: 'MESH'
        Visualization mode - 'MESH' (triangular mesh) or 'POINTS' (scatter)
    skip_frames : int, default: 1
        Only use every nth frame for performance
    linewidth : float, default: 0.01
        Line width for mesh edges
    figure_size : Tuple[int, int], default: (10, 10)
        Size of the figure in inches
    draw_los_vector : bool, default: True
        Whether to draw the line-of-sight vector
    draw_rotation_axis : bool, default: True
        Whether to draw the rotation axis
    view_angles : Optional[Tuple[float, float]], default: None
        Tuple of (elevation, azimuth) viewing angles. If None, uses default view
    fixed_norm : bool, default: True
        Whether to keep color normalization fixed across all frames
        
    Returns
    -------
    str
        The path to the saved animation file
    """
    from matplotlib.animation import FuncAnimation
    
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of {PLOT_MODES}. Got {mode.upper()}')
    mode = mode.upper()
    
    # Filter meshes and timestamps using skip_frames
    meshes = meshes[::skip_frames]
    if timestamps is not None:
        timestamps = timestamps[::skip_frames]
    
    # Initial setup
    fig = plt.figure(figsize=figure_size)
    spec = fig.add_gridspec(10, 12)
    plot_ax = fig.add_subplot(spec[:, :10], projection='3d')
    cbar_ax = fig.add_subplot(spec[2:8, 11])
    
    # Set up colormap
    if cmap is None:
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)
    
    # Get property data
    mesh = meshes[0]
    
    # If using fixed color normalization across frames, compute min/max across all meshes
    if fixed_norm:
        all_property_values = []
        for mesh in meshes:
            prop_values, _ = _evaluate_to_be_mapped_property(mesh, property, property_label)
            all_property_values.append(prop_values)
        all_property_values = np.concatenate(all_property_values)
        vmin, vmax = np.nanmin(all_property_values), np.nanmax(all_property_values)
    else:
        # Just get the property label from the first mesh
        prop_values, _ = _evaluate_to_be_mapped_property(mesh, property, property_label)
        vmin, vmax = np.nanmin(prop_values), np.nanmax(prop_values)
    
    _, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)
    
    # Setup axis limits
    axes_lim = 1.5 * mesh.radius
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)
    
    # Set custom view angle if provided
    if view_angles is not None:
        elev, azim = view_angles
        plot_ax.view_init(elev=elev, azim=azim)
    
    # Set up colormap normalization
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Set up colorbar
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    # Title for timestamps
    title = fig.suptitle("", fontsize=14, y=0.98)
    
    # Define the update function for animation
    def update(frame):
        # Clear previous frame elements
        plot_ax.clear()
        
        # Reset axis properties after clearing
        plot_ax.set_xlim3d(-axes_lim, axes_lim)
        plot_ax.set_ylim3d(-axes_lim, axes_lim)
        plot_ax.set_zlim3d(-axes_lim, axes_lim)
        plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
        plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
        plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)
        
        # Set custom view angle if provided
        if view_angles is not None:
            elev, azim = view_angles
            plot_ax.view_init(elev=elev, azim=azim)
        
        # Get current mesh
        mesh = meshes[frame]
        
        # Update timestamp if available
        if timestamps is not None:
            ts_str = f"Time: {timestamps[frame]:.2f} {timestamp_label or ''}"
            title.set_text(ts_str)
        
        # Get property data for current mesh
        to_be_mapped, _ = _evaluate_to_be_mapped_property(mesh, property, property_label)
        
        # Draw vectors
        if draw_los_vector:
            normalized_los_vector = mesh.los_vector / np.linalg.norm(mesh.los_vector)
            los_vector = plot_ax.quiver(*(-2.0 * mesh.radius * normalized_los_vector), 
                                      *(mesh.radius * normalized_los_vector),
                                      color='red', linewidth=3., label='LOS vector')
        
        if draw_rotation_axis:
            normalized_rotation_axis = mesh.rotation_axis / np.linalg.norm(mesh.rotation_axis)
            rotation_axis = plot_ax.quiver(*(0.75 * mesh.radius * normalized_rotation_axis), 
                                         *(mesh.radius * normalized_rotation_axis),
                                         color='black', linewidth=3., label='Rotation axis')
        
        # Visualize the mesh
        if mode == 'MESH':
            vs2 = mesh.vertices[mesh.faces.astype(int)]
            face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
            mesh_collection = art3d.Poly3DCollection(vs2, facecolors=face_colors, 
                                                   edgecolor="black", linewidths=linewidth)
            plot_ax.add_collection(mesh_collection)
        else:  # mode == 'POINTS'
            mesh_collection = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                                           c=to_be_mapped, cmap=cmap, norm=norm)
        
        # Only show legend in the first frame
        if frame == 0 and (draw_los_vector or draw_rotation_axis):
            plot_ax.legend(loc='upper right', fontsize=12)
        
        # Add padding between plot and colorbar
        plt.subplots_adjust(right=0.85)
        
        # Return empty list since we're not using blit=True
        return []
    
    # Try to save the animation, catching any errors
    try:
        anim = FuncAnimation(fig, update, frames=len(meshes), blit=False)
        smart_save(anim, filename, fps=20)
    except Exception as e:
        import os
        import time
        import warnings
        
        # Check if the file exists and when it was created
        if os.path.exists(filename):
            # Check both creation and modification time to handle overwrites
            file_creation_time = os.path.getctime(filename)
            file_modification_time = os.path.getmtime(filename)
            current_time = time.time()
            # Use the most recent timestamp (creation or modification)
            most_recent_time = max(file_creation_time, file_modification_time)
            # If file was created or modified in the last 10 seconds, assume it's from this run
            if current_time - most_recent_time < 10:
                warnings.warn(f"Animation save warning: {str(e)}")
            else:
                print(f'File {filename} is older than 10 seconds (meaning it exists but is not from this run), re-raising error')
                # File exists but is older, re-raise the error
                raise
        else:
            print(f'File {filename} does not exist, re-raising error')
            # File doesn't exist, re-raise the error
            raise
    
    return filename


def animate_mesh_and_spectra(meshes, spectra, wavelengths, 
                             property=DEFAULT_PROPERTY, 
                             cmap=None, 
                             property_label=None,
                             filename='mesh_and_spectra_animation.mp4',
                             figsize=(15, 5),
                             mode='MESH',
                             draw_los_vector=True,
                             draw_rotation_axis=True,
                             linewidth=0.1,
                             axes_lim=None,
                             timestamps=None,
                             timestamp_label=None):
    """
    Create an animation showing both a 3D mesh model and its corresponding spectrum over time.
    
    Parameters
    ----------
    meshes : List[MeshModel]
        List of mesh models to animate
    spectra : ndarray
        Array of spectra corresponding to each mesh, shape (n_frames, n_wavelengths)
    wavelengths : ndarray
        Wavelength array for the spectra
    property : Union[str, int], default: DEFAULT_PROPERTY
        Property to color the mesh by (attribute name or parameter index)
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
    filename : str, default: 'mesh_and_spectra_animation.mp4'
        Output filename for the animation
    figsize : Tuple[int, int], default: (12, 8)
        Figure size
    mode : str, default: 'MESH'
        Visualization mode - 'MESH' (triangular mesh) or 'POINTS' (scatter)
    draw_los_vector : bool, default: True
        Whether to draw the line-of-sight vector
    draw_rotation_axis : bool, default: True
        Whether to draw the rotation axis
    linewidth : float, default: 0.1
        Line width for mesh edges
    axes_lim : Optional[float], default: None
        Limit for all axes. If None, calculated from mesh radius
    timestamps : Optional[ArrayLike], default: None
        List of timestamps corresponding to each mesh
    timestamp_label : Optional[str], default: None
        Label to accompany timestamps (e.g., "hours", "days")
        
    Returns
    -------
    str
        Path to the saved animation file
    
    Raises
    ------
    ValueError
        If the mode is invalid or if spectra and meshes have different lengths
    """
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of {PLOT_MODES}. Got {mode.upper()}')
    mode = mode.upper()
    
    if len(meshes) != len(spectra):
        raise ValueError(f"Number of meshes ({len(meshes)}) must match number of spectra ({len(spectra)})")
    
    # Setup timestamp label
    timestamp_label = timestamp_label or ''
    
    # Get property data for coloring the mesh
    to_be_mapped_arrays = []
    for mesh in meshes:
        to_be_mapped, label = _evaluate_to_be_mapped_property(mesh, property, property_label)
        to_be_mapped_arrays.append(to_be_mapped)
    
    cbar_label = label
    to_be_mapped_arrays_concatenated = np.concatenate(to_be_mapped_arrays)
    
    # Set up the colormap
    if cmap is None:
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)
    
    # Determine axis limits
    if axes_lim is None:
        axes_lim = 1.5 * max([mesh.radius for mesh in meshes])
    
    # Create figure with two subplots: 3D mesh and spectrum
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(4, 4, width_ratios=[1, 0.05, 0.05, 2])

    # 3D mesh subplot
    mesh_ax = fig.add_subplot(gs[:, 0], projection='3d')
    mesh_ax.set_xlim3d(-axes_lim, axes_lim)
    mesh_ax.set_ylim3d(-axes_lim, axes_lim)
    mesh_ax.set_zlim3d(-axes_lim, axes_lim)
    mesh_ax.set_xlabel('$X [R_\\odot]$', fontsize=10)
    mesh_ax.set_ylabel('$Y [R_\\odot]$', fontsize=10)
    mesh_ax.set_zlabel('$Z [R_\\odot]$', fontsize=10)
    
    # Spectrum subplot
    spec_ax = fig.add_subplot(gs[1:3, 3])
    spec_ax.set_xlabel('Wavelength [$\AA$]', fontsize=10)
    spec_ax.set_ylabel('Flux [erg/s/cm$^3$]', fontsize=10)
    
    # Set up color normalization for the mesh
    norm = mpl.colors.Normalize(vmin=to_be_mapped_arrays_concatenated.min(), 
                               vmax=to_be_mapped_arrays_concatenated.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Add colorbar for the mesh
    cbar_ax = fig.add_subplot(gs[1:3, 1])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    # Find min/max for spectrum y-axis
    spec_min = np.min(spectra)
    spec_max = np.max(spectra)
    spec_padding = 0.05 * (spec_max - spec_min)
    spec_ax.set_ylim(spec_min - spec_padding, spec_max + spec_padding)
    
    # Initial spectrum line
    spectrum_line, = spec_ax.plot(wavelengths, spectra[0], color='black')
    plt.tight_layout()
    
    # Animation update function
    def update(frame):
        mesh_ax.clear()
        
        # Reset axis properties
        mesh_ax.set_xlim3d(-axes_lim, axes_lim)
        mesh_ax.set_ylim3d(-axes_lim, axes_lim)
        mesh_ax.set_zlim3d(-axes_lim, axes_lim)
        mesh_ax.set_xlabel('$X [R_\\odot]$', fontsize=10)
        mesh_ax.set_ylabel('$Y [R_\\odot]$', fontsize=10)
        mesh_ax.set_zlabel('$Z [R_\\odot]$', fontsize=10)
        
        # Get current mesh and property data
        mesh = meshes[frame]
        to_be_mapped = to_be_mapped_arrays[frame]
        
        # Draw los vector and rotation axis
        if draw_los_vector:
            normalized_los_vector = mesh.los_vector/np.linalg.norm(mesh.los_vector)
            mesh_ax.quiver(*(-2.0*mesh.radius*normalized_los_vector), *(mesh.radius*normalized_los_vector),
                        color='red', linewidth=3., label='LOS vector')
        
        if draw_rotation_axis:
            normalized_rotation_axis = mesh.rotation_axis/np.linalg.norm(mesh.rotation_axis)
            mesh_ax.quiver(*(0.75*mesh.radius*normalized_rotation_axis), *(mesh.radius*normalized_rotation_axis),
                        color='black', linewidth=3., label='Rotation axis')
        
        # Visualize mesh according to mode
        if mode == 'MESH':
            vs2 = mesh.vertices[mesh.faces.astype(int)]
            face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
            mesh_collection = art3d.Poly3DCollection(vs2, facecolors=face_colors, 
                                                   edgecolor="black", linewidths=linewidth)
            mesh_ax.add_collection(mesh_collection)
        else:  # mode == 'POINTS'
            mesh_collection = mesh_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                                           c=to_be_mapped, cmap=cmap, norm=norm)
        
        # Update spectrum
        spectrum_line.set_ydata(spectra[frame])
        
        # Add timestamp if available
        if timestamps is not None:
            title = f"Time: {timestamps[frame]:.2f} {timestamp_label}"
            fig.suptitle(title, y=0.85)
        
        # Only show legend in the first frame
        if frame == 0 and (draw_los_vector or draw_rotation_axis):
            mesh_ax.legend(loc='upper right', fontsize=10)
        
        return [mesh_collection, spectrum_line]
    
    # Create and save animation
    try:
        anim = FuncAnimation(fig, update, frames=len(meshes), blit=False)
        smart_save(anim, filename, fps=20)
    except Exception as e:
        import os
        import time
        import warnings
        
        # Check if the file exists and when it was created
        if os.path.exists(filename):
            # Check both creation and modification time to handle overwrites
            file_creation_time = os.path.getctime(filename)
            file_modification_time = os.path.getmtime(filename)
            current_time = time.time()
            # Use the most recent timestamp (creation or modification)
            most_recent_time = max(file_creation_time, file_modification_time)
            # If file was created or modified in the last 10 seconds, assume it's from this run
            if current_time - most_recent_time < 10:
                warnings.warn(f"Animation save warning: {str(e)}")
            else:
                print(f'File {filename} is older than 10 seconds (meaning it exists but is not from this run), re-raising error')
                # File exists but is older, re-raise the error
                raise
        else:
            print(f'File {filename} does not exist, re-raising error')
            # File doesn't exist, re-raise the error
            raise
    
    plt.close(fig)
    return filename


def animate_binary(binary1_meshes, binary2_meshes, filename, 
                   property='los_velocities', 
                   timestamps=None, 
                   cmap=None,
                   property_label=None,
                   timestamp_label='days',
                   mode='MESH',
                   skip_frames=1,
                   linewidth=0.01,
                   figure_size=(10, 10),
                   draw_los_vector=True,
                   draw_rotation_axes=True,
                   view_angles=None,
                   fixed_norm=True,
                   scale_radius=1.0):
    """
    Create an animation of a binary system over time.
    
    Parameters
    ----------
    binary1_meshes : List[MeshModel]
        List of mesh models for the first star at different time steps
    binary2_meshes : List[MeshModel]
        List of mesh models for the second star at different time steps
    filename : str
        Output filename for the animation
    property : Union[str, int], default: 'los_velocities'
        Property to color the meshes by (attribute name or parameter index)
    timestamps : Optional[ArrayLike], default: None
        List of timestamps corresponding to each mesh
    cmap : Optional[str], default: None
        Matplotlib colormap name. If None, uses defaults based on property
    property_label : Optional[str], default: None
        Custom label for the colorbar. If None, uses default for the property
    timestamp_label : Optional[str], default: "days"
        Label to accompany timestamps (e.g., "hours", "days")
    mode : str, default: 'MESH'
        Visualization mode - 'MESH' (triangular mesh) or 'POINTS' (scatter)
    skip_frames : int, default: 1
        Only use every nth frame for performance
    linewidth : float, default: 0.01
        Line width for mesh edges
    figure_size : Tuple[int, int], default: (10, 10)
        Size of the figure in inches
    draw_los_vector : bool, default: True
        Whether to draw the line-of-sight vector
    draw_rotation_axes : bool, default: True
        Whether to draw the rotation axes of both stars
    view_angles : Optional[Tuple[float, float]], default: None
        Tuple of (elevation, azimuth) viewing angles. If None, uses default view
    fixed_norm : bool, default: True
        Whether to keep color normalization fixed across all frames
    scale_radius : float, default: 1.0
        Scale factor for both stellar radii
        
    Returns
    -------
    str
        The path to the saved animation file
    """
    if mode.upper() not in ['MESH', 'POINTS']:
        raise ValueError(f'Mode must be one of ["MESH", "POINTS"]. Got {mode.upper()}')
    mode = mode.upper()
    
    # Filter meshes and timestamps using skip_frames
    binary1_meshes = binary1_meshes[::skip_frames]
    binary2_meshes = binary2_meshes[::skip_frames]
    if timestamps is not None:
        timestamps = timestamps[::skip_frames]
    
    # Initial setup
    fig = plt.figure(figsize=figure_size)
    spec = fig.add_gridspec(10, 12)
    plot_ax = fig.add_subplot(spec[:, :10], projection='3d')
    cbar_ax = fig.add_subplot(spec[2:8, 11])
    
    # Get first meshes for initial setup
    mesh1 = binary1_meshes[0]
    mesh2 = binary2_meshes[0]
    
    # If using fixed color normalization across frames, compute min/max across all meshes
    if fixed_norm:
        all_property_values = []
        for i in range(len(binary1_meshes)):
            # Get property values for both stars
            mesh1, mesh2 = binary1_meshes[i], binary2_meshes[i]
            
            # Get property data
            to_be_mapped1, _ = _evaluate_to_be_mapped_property(mesh1, property, property_label)
            to_be_mapped2, _ = _evaluate_to_be_mapped_property(mesh2, property, property_label)
            
            all_property_values.append(np.concatenate([to_be_mapped1, to_be_mapped2]))
        
        all_property_values = np.concatenate(all_property_values)
        vmin, vmax = np.nanmin(all_property_values), np.nanmax(all_property_values)
    else:
        # Just get the property values from the first meshes
        to_be_mapped1, _ = _evaluate_to_be_mapped_property(mesh1, property, property_label)
        to_be_mapped2, _ = _evaluate_to_be_mapped_property(mesh2, property, property_label)
        to_be_mapped = np.concatenate([to_be_mapped1, to_be_mapped2])
        vmin, vmax = np.nanmin(to_be_mapped), np.nanmax(to_be_mapped)
    
    # Get property label
    _, cbar_label = _evaluate_to_be_mapped_property(mesh1, property, property_label)
    
    # Set up axes limits
    axes_lim = np.max(np.abs((mesh1.radius+mesh1.center)-(mesh2.center-mesh2.radius))) * scale_radius * 1.2
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)
    
    # Set custom view angle if provided
    if view_angles is not None:
        elev, azim = view_angles
        plot_ax.view_init(elev=elev, azim=azim)
    
    # Set up colormap
    if cmap is None:
        from spice.plots.plot_mesh import DEFAULT_PROPERTY_CMAPS, DEFAULT_CMAP
        cmap = DEFAULT_PROPERTY_CMAPS.get(property, DEFAULT_CMAP)
    
    # Set up colormap normalization
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Set up colorbar
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)
    
    # Title for timestamps
    title = fig.suptitle("", fontsize=14, y=0.98)
    
    # Define the update function for animation
    def update(frame):
        # Clear previous frame elements
        plot_ax.clear()
        
        # Reset axis properties after clearing
        plot_ax.set_xlim3d(-axes_lim, axes_lim)
        plot_ax.set_ylim3d(-axes_lim, axes_lim)
        plot_ax.set_zlim3d(-axes_lim, axes_lim)
        plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
        plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
        plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)
        
        # Set custom view angle if provided
        if view_angles is not None:
            elev, azim = view_angles
            plot_ax.view_init(elev=elev, azim=azim)
        
        # Get current meshes
        mesh1 = binary1_meshes[frame]
        mesh2 = binary2_meshes[frame]
        
        # Update timestamp if available
        if timestamps is not None:
            ts_str = f"Time: {timestamps[frame]:.2f} {timestamp_label or ''}"
            title.set_text(ts_str)
        
        # Get property data for current meshes
        to_be_mapped1, _ = _evaluate_to_be_mapped_property(mesh1, property, property_label)
        to_be_mapped2, _ = _evaluate_to_be_mapped_property(mesh2, property, property_label)
        
        # Draw vectors
        if draw_los_vector:
            normalized_los_vector = mesh1.los_vector / np.linalg.norm(mesh1.los_vector)
            plot_ax.quiver(*(-1.5*axes_lim*normalized_los_vector), *((axes_lim*normalized_los_vector)/2),
                    color='red', linewidth=3., label='LOS vector')
        
        if draw_rotation_axes:
            normalized_rotation_axis1 = mesh1.rotation_axis / np.linalg.norm(mesh1.rotation_axis)
            normalized_rotation_axis2 = mesh2.rotation_axis / np.linalg.norm(mesh2.rotation_axis)
            
            plot_ax.quiver(*(mesh1.center+normalized_rotation_axis1*mesh1.radius*scale_radius), 
                           *(mesh1.radius*normalized_rotation_axis1*scale_radius),
                           color='black', linewidth=3., label='Rotation axis of mesh1')
            plot_ax.quiver(*(mesh2.center+normalized_rotation_axis2*mesh2.radius*scale_radius), 
                           *(mesh2.radius*normalized_rotation_axis2*scale_radius),
                           color='blue', linewidth=3., label='Rotation axis of mesh2')
        
        # Visualize the meshes
        if mode == 'MESH':
            # First mesh
            vs2_1 = mesh1.center+(mesh1.mesh_elements-mesh1.center)*scale_radius
            face_colors1 = mpl.colormaps[cmap](norm(to_be_mapped1))
            p1 = art3d.Poly3DCollection(vs2_1, facecolors=face_colors1, edgecolor="black", linewidths=linewidth)
            
            # Second mesh
            vs2_2 = mesh2.center+(mesh2.mesh_elements-mesh2.center)*scale_radius
            face_colors2 = mpl.colormaps[cmap](norm(to_be_mapped2))
            p2 = art3d.Poly3DCollection(vs2_2, facecolors=face_colors2, edgecolor="black", linewidths=linewidth)
            
            plot_ax.add_collection(p1)
            plot_ax.add_collection(p2)
        else:  # mode == 'POINTS'
            # First mesh as points
            centers1 = mesh1.center+(mesh1.centers-mesh1.center)*scale_radius
            plot_ax.scatter(centers1[:, 0], centers1[:, 1], centers1[:, 2],
                           c=to_be_mapped1, cmap=cmap, norm=norm)
            
            # Second mesh as points
            centers2 = mesh2.center+(mesh2.centers-mesh2.center)*scale_radius
            plot_ax.scatter(centers2[:, 0], centers2[:, 1], centers2[:, 2],
                           c=to_be_mapped2, cmap=cmap, norm=norm)
        
        # Only show legend in the first frame
        if draw_los_vector or draw_rotation_axes:
            plot_ax.legend(loc='upper right', fontsize=12)
        
        # Add padding between plot and colorbar
        plt.subplots_adjust(right=0.85)
        
        # Return empty list since we're not using blit=True
        return []
    anim = FuncAnimation(fig, update, frames=len(binary1_meshes), blit=False)
    smart_save(anim, filename, fps=20)
    
    plt.close(fig)
    return filename

# Helper function to evaluate property to be mapped
def _evaluate_to_be_mapped_property(mesh, property, property_label=None):
    """Helper function to get property data and label for visualization."""
    if isinstance(property, str):
        # If property is a string, try to get it as an attribute
        if hasattr(mesh, property):
            to_be_mapped = getattr(mesh, property)
            label = property_label or property
        else:
            raise ValueError(f"Mesh does not have attribute '{property}'")
    elif isinstance(property, int):
        # If property is an integer, assume it's a parameter index
        to_be_mapped = mesh.parameters[:, property]
        label = property_label or f"Parameter {property}"
    else:
        raise ValueError(f"Property must be a string or integer, got {type(property)}")
    
    return to_be_mapped, label