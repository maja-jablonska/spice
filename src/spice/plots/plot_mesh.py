import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import art3d
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from spice.models import MeshModel
from jax.typing import ArrayLike

PLOT_MODES = ['MESH', 'POINTS']
COLORMAP_PROPERTIES = ['mus', 'los_velocities', 'cast_areas', 'log_gs']
DEFAULT_PROPERTY = 'mus'

DEFAULT_PLOT_PROPERTY_LABELS = {
    'mus': r'$\mu$',
    'los_velocities': 'LOS velocity [km/s]',
    'cast_areas': 'cast area [km$^2$]',
    'log_gs': 'log g'
}


def _evaluate_to_be_mapped_property(mesh: MeshModel,
                                    property: Union[str, int] = DEFAULT_PROPERTY,
                                    property_label: Optional[str] = None) -> Tuple[ArrayLike, str]:

    if type(property) is str:
        if property not in COLORMAP_PROPERTIES:
            raise ValueError(f'Invalid property {property} - must be one of ({",".join(COLORMAP_PROPERTIES)})')
        else:
            to_be_mapped = getattr(mesh, property)
        if property_label is None:
            property_label = DEFAULT_PLOT_PROPERTY_LABELS.get(property, '')
    elif type(property) is int:
        if property > mesh.parameters.shape[-1]-1:
            raise ValueError(f'Invalid property index {property} - must be smaller than {mesh.parameters.shape[-1]}')
        else:
            to_be_mapped = mesh.parameters[:, property]
        if property_label is None:
            property_label = ''
    else:
        raise ValueError(f"Property must be either of type str or int")
    
    return to_be_mapped, property_label


def plot_3D(mesh: MeshModel,
            property: Union[str, int] = DEFAULT_PROPERTY,
            axes: Optional[Tuple[plt.figure, plt.axes]] = None,
            cmap: str = 'turbo',
            property_label: Optional[str] = None,
            mode: str = 'MESH',
            update_colorbar: bool = True,
            draw_los_vector: bool = True,
            draw_rotation_axis: bool = True):
    
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of ["MESH", "POINTS"]. Got {mode.upper()}')
    mode = mode.upper()
    
    to_be_mapped, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)

    if axes is None:
        fig = plt.figure(figsize=(10, 12))
        spec = fig.add_gridspec(10, 12)
        plot_ax = fig.add_subplot(spec[:, :11], projection='3d')
        plot_ax.view_init(elev=30, azim=60)
    else:
        try:
            fig, plot_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or (plt.figure, plt.axes, plt.axes) for the plot axis and colorbar axis")
    axes_lim = 1.5*mesh.radius
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)

    if draw_los_vector:
        normalized_los_vector = mesh.los_vector/np.linalg.norm(mesh.los_vector)
        plot_ax.quiver(*(-2.0*mesh.radius*normalized_los_vector), *(mesh.radius*normalized_los_vector),
                   color='red', linewidth=3., label='LOS vector')
    
    if draw_rotation_axis:
        normalized_rotation_axis = mesh.rotation_axis/np.linalg.norm(mesh.rotation_axis)
        plot_ax.quiver(*(0.75*mesh.radius*normalized_rotation_axis), *(mesh.radius*normalized_rotation_axis),
                    color='black', linewidth=3., label='Rotation axis')
        
    if draw_los_vector or draw_rotation_axis:
        plot_ax.legend()

    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    if mode == 'MESH':
        vs2 = mesh.mesh_elements
        face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
        p = art3d.Poly3DCollection(vs2, facecolors=face_colors, edgecolor="black")
        plot_ax.add_collection(p)
        mappable.set_array([])
    else:
        p = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                            c=to_be_mapped, cmap=cmap, norm=norm)
        
    if update_colorbar:
        cbar = fig.colorbar(mappable, shrink=0.45, pad=0.125, ax=plot_ax)
        cbar.set_label(cbar_label, fontsize=12)

    return fig, plot_ax


def plot_3D_binary(mesh1: MeshModel,
                   mesh2: MeshModel,
                   property: Union[str, int] = DEFAULT_PROPERTY,
                   axes: Optional[Tuple[plt.figure, plt.axes]] = None,
                   cmap: str = 'turbo',
                   property_label: Optional[str] = None,
                   mode: str = 'MESH',
                   update_colorbar: bool = True,
                   scale_radius: float = 1.0,
                   draw_los_vector: bool = True,
                   draw_rotation_axes: bool = True):
    
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of ["MESH", "POINTS"]. Got {mode.upper()}')
    mode = mode.upper()
    
    to_be_mapped1, cbar_label = _evaluate_to_be_mapped_property(mesh1, property, property_label)
    to_be_mapped2, _ = _evaluate_to_be_mapped_property(mesh2, property, property_label)
    to_be_mapped = np.concatenate([to_be_mapped1, to_be_mapped2])

    if axes is None:
        fig = plt.figure(figsize=(10, 12))
        spec = fig.add_gridspec(10, 12)
        plot_ax = fig.add_subplot(spec[:, :11], projection='3d')
        plot_ax.view_init(elev=30, azim=60)
    else:
        try:
            fig, plot_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or (plt.figure, plt.axes, plt.axes) for the plot axis and colorbar axis")
    
    axes_lim = np.max(np.abs((mesh1.radius+mesh1.center)-(mesh2.center-mesh2.radius)))
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)

    if draw_los_vector:
        normalized_los_vector = mesh1.los_vector/np.linalg.norm(mesh1.los_vector)
        plot_ax.quiver(*(-1.5*axes_lim*normalized_los_vector), *((axes_lim*normalized_los_vector)/2),
                color='red', linewidth=3., label='LOS vector')
        
    if draw_rotation_axes:
        normalized_rotation_axis1 = mesh1.rotation_axis/np.linalg.norm(mesh1.rotation_axis)
        normalized_rotation_axis2 = mesh2.rotation_axis/np.linalg.norm(mesh2.rotation_axis)
        
        plot_ax.quiver(*(mesh1.center+normalized_rotation_axis1*mesh1.radius*scale_radius), *(mesh1.radius*normalized_rotation_axis1*np.sqrt(scale_radius)),
                    color='black', linewidth=3., label='Rotation axis of mesh1')
        plot_ax.quiver(*(mesh2.center+normalized_rotation_axis2*mesh2.radius*scale_radius), *(mesh2.radius*normalized_rotation_axis2*np.sqrt(scale_radius)),
                    color='blue', linewidth=3., label='Rotation axis of mesh2')
    if draw_los_vector or draw_rotation_axes:
        plot_ax.legend()

    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    if mode == 'MESH':
        vs2_1 = mesh1.center+(mesh1.mesh_elements-mesh1.center)*scale_radius
        face_colors1 = mpl.colormaps[cmap](norm(to_be_mapped1))
        p1 = art3d.Poly3DCollection(vs2_1, facecolors=face_colors1, edgecolor="black", linewidths=0.01)
        
        vs2_2 = mesh2.center+(mesh2.mesh_elements-mesh2.center)*scale_radius
        face_colors2 = mpl.colormaps[cmap](norm(to_be_mapped2))
        p2 = art3d.Poly3DCollection(vs2_2, facecolors=face_colors2, edgecolor="black", linewidths=0.01)
        
        plot_ax.add_collection(p1)
        plot_ax.add_collection(p2)
        mappable.set_array([])
    else:
        centers1 = mesh1.center+(mesh1.centers-mesh1.center)*scale_radius
        p1 = plot_ax.scatter(centers1[:, 0], centers1[:, 1], centers1[:, 2],
                             c=to_be_mapped1, cmap=cmap, norm=norm)
        centers2 = mesh2.center+(mesh2.centers-mesh2.center)*scale_radius
        p2 = plot_ax.scatter(centers2[:, 0], centers2[:, 1], centers2[:, 2],
                             c=to_be_mapped2, cmap=cmap, norm=norm)
        
    if update_colorbar:
        cbar = fig.colorbar(mappable, shrink=0.45, pad=0.125, ax=plot_ax)
        cbar.set_label(cbar_label, fontsize=12)

    return fig, plot_ax


def plot_3D_sequence(meshes: List[MeshModel],
                     property: Union[str, int] = DEFAULT_PROPERTY,
                     timestamps: Optional[ArrayLike] = None,
                     axes: Optional[Tuple[plt.figure, List[plt.axes], plt.axes]] = None,
                     cmap: str = 'turbo',
                     property_label: Optional[str] = None,
                     timestamp_label: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 10),
                     mode: str = 'MESH'):
    
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of ["MESH", "POINTS"]. Got {mode.upper()}')
    mode = mode.upper()

    timestamp_label = timestamp_label or ''
    
    _, cbar_label = _evaluate_to_be_mapped_property(meshes[0], property, property_label)
    to_be_mapped_arrays = [_evaluate_to_be_mapped_property(mesh, property, property_label)[0] for mesh in meshes]
    to_be_mapped_arrays_concatenated = np.concatenate(to_be_mapped_arrays)

    axes_lim = 1.5*max([mesh.radius for mesh in meshes])
    num_plots = len(meshes)

    if axes is None:
        # Determine the number of rows and columns for the subplots
        cols = int(num_plots**0.5)
        rows = num_plots // cols
        rows += num_plots % cols

        # Create a figure with gridspec
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(10*rows, cols + 1, width_ratios=[*([1]*cols), 0.05])
        plot_axes = [fig.add_subplot(gs[10*(i // cols):10*((i // cols)+1), i % cols], projection='3d') for i in range(num_plots)]

        # Add the colorbar in the last column
        cax_limit = max(rows, 5)
        cbar_ax = fig.add_subplot(gs[cax_limit:-cax_limit, -1])
    else:
        try:
            fig, plot_axes, cbar_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or (plt.figure, plt.axes, plt.axes) for the plot axis and colorbar axis")
    
    for plot_ax in plot_axes:
        plot_ax.set_xlim3d(-axes_lim, axes_lim)
        plot_ax.set_ylim3d(-axes_lim, axes_lim)
        plot_ax.set_zlim3d(-axes_lim, axes_lim)
        plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=10)
        plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=10)
        plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=10)

    norm = mpl.colors.Normalize(vmin=to_be_mapped_arrays_concatenated.min(), vmax=to_be_mapped_arrays_concatenated.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    for plot_ax, (to_be_mapped, (i, mesh)) in zip(plot_axes, zip(to_be_mapped_arrays, enumerate(meshes))):
        normalized_los_vector = mesh.los_vector/np.linalg.norm(mesh.los_vector)
        normalized_rotation_axis = mesh.rotation_axis/np.linalg.norm(mesh.rotation_axis)

        plot_ax.quiver(*(-2.0*mesh.radius*normalized_los_vector), *(mesh.radius*normalized_los_vector),
                    color='red', linewidth=3., label='LOS vector')
        plot_ax.quiver(*(0.75*mesh.radius*normalized_rotation_axis), *(mesh.radius*normalized_rotation_axis),
                    color='black', linewidth=3., label='Rotation axis')

        if mode == 'MESH':
            vs2 = mesh.vertices[mesh.faces.astype(int)]
            face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
            p = art3d.Poly3DCollection(vs2, facecolors=face_colors, edgecolor="black")
            plot_ax.add_collection(p)
            if timestamps is not None:
                plot_ax.set_title("Time: {ts:.2f} {timestamp_label}".format(ts=timestamps[i], timestamp_label=timestamp_label), pad=1.0)
        else:
            p = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                                c=to_be_mapped, cmap=cmap, norm=norm)
            
    plot_axes[int(cols-1)].legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    mappable.set_array([])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)
    fig.tight_layout()

    return fig, plot_axes, cbar_ax


def animate_mesh_and_spectra(meshes: List[MeshModel],
                             timestamps: ArrayLike,
                             wavelengths: ArrayLike,
                             spectra: ArrayLike,
                             filename: str,
                             property: Union[str, int] = DEFAULT_PROPERTY,
                             cmap: str = 'turbo',
                             property_label: Optional[str] = None,
                             timestamp_label: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 10),
                             mode: str = 'MESH'):
    try:
        from celluloid import Camera
    except ImportError:
        raise ImportError("celluloid is not installed. Please install celluloid to use pre-defined animation functions.")

    fig = plt.figure(figsize=(24, 10))
    spec = fig.add_gridspec(10, 24)
    plot_ax = fig.add_subplot(spec[:, :10], projection='3d')
    cax = fig.add_subplot(spec[2:8, 10])
    spectrum_ax = fig.add_subplot(spec[2:8, 13:])
    camera = Camera(fig)

    mesh = meshes[0]
    to_be_mapped, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)
    axes_lim = 1.5*mesh.radius
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)

    normalized_los_vector = mesh.los_vector/np.linalg.norm(mesh.los_vector)
    normalized_rotation_axis = mesh.rotation_axis/np.linalg.norm(mesh.rotation_axis)


    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.tight_layout()


    for i, (mesh, s) in enumerate(zip(meshes, spectra)):
        norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        if mode == 'MESH':
            vs2 = mesh.vertices[mesh.faces.astype(int)]
            face_colors = mpl.colormaps[cmap](norm(to_be_mapped))
            p = art3d.Poly3DCollection(vs2, facecolors=face_colors, edgecolor="black", zorder=0)
            
            plot_ax.add_collection(p)
            plot_ax.quiver(*(-2.0*mesh.radius*normalized_los_vector), *(mesh.radius*normalized_los_vector),
                    color='red', linewidth=3., label='LOS vector', zorder=2)
            plot_ax.quiver(*(0.75*mesh.radius*normalized_rotation_axis), *(mesh.radius*normalized_rotation_axis),
                            color='black', linewidth=3., label='Rotation axis', zorder=2)
            if i == 0:
                plot_ax.legend(fontsize=14)
                
            if timestamps is not None:
                spectrum_ax.text(0.4, 1.1, "Time: {ts:.2f} {timestamp_label}".format(ts=timestamps[i], timestamp_label=timestamp_label),
                                 fontsize=24,
                                 transform=spectrum_ax.transAxes)
        else:
            p = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                                c=to_be_mapped, cmap=cmap, norm=norm)
            
        cbar = fig.colorbar(mappable, shrink=0.45, pad=0.125, cax=cax)
        cbar.set_label(cbar_label, fontsize=12)
        spectrum_ax.plot(wavelengths, s[:, 0], color='black')
        spectrum_ax.set_ylabel('intensity [erg/s/cm$^2$]', fontsize=14)
        spectrum_ax.set_xlabel('wavelength [$\\AA$]', fontsize=14)
        
        camera.snap()
        
    animation = camera.animate()
    animation.save(filename)
    
    return animation


def animate_binary(meshes1: MeshModel,
                   meshes2: MeshModel,
                   filename: str,
                   property: Union[str, int] = DEFAULT_PROPERTY,
                   cmap: str = 'turbo',
                   property_label: Optional[str] = None,
                   mode: str = 'MESH',
                   scale_radius: float = 1.0,
                   draw_los_vector: bool = True,
                   draw_rotation_axes: bool = True):
    
    if mode.upper() not in PLOT_MODES:
        raise ValueError(f'Mode must be one of ["MESH", "POINTS"]. Got {mode.upper()}')
    mode = mode.upper()
    
    try:
        from celluloid import Camera
    except ImportError:
        raise ImportError("celluloid is not installed. Please install celluloid to use pre-defined animation functions.")
    
    mesh1 = meshes1[0]
    _, cbar_label = _evaluate_to_be_mapped_property(mesh1, property, property_label)
    to_be_mapped1s = np.concatenate([_evaluate_to_be_mapped_property(mesh1, property, property_label)[0] for mesh1 in meshes1])
    to_be_mapped2s = np.concatenate([_evaluate_to_be_mapped_property(mesh2, property, property_label)[0] for mesh2 in meshes2])
    
    to_be_mapped = np.concatenate([to_be_mapped1s, to_be_mapped2s])

    fig = plt.figure(figsize=(10, 12))
    spec = fig.add_gridspec(10, 12)
    plot_ax = fig.add_subplot(spec[:, :11], projection='3d')
    cax = fig.add_subplot(spec[2:8, 11])
    plot_ax.view_init(elev=30, azim=60)
    camera = Camera(fig)
    
    center_differences = [np.abs((mesh1.radius+mesh1.center)-(mesh2.center-mesh1.radius)) for mesh1, mesh2 in zip(meshes1, meshes2)]
    
    axes_lim = 1.05*np.max(center_differences)
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\\odot]$', fontsize=14)

    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, (mesh1, mesh2) in enumerate(zip(meshes1, meshes2)):
        to_be_mapped1, _ = _evaluate_to_be_mapped_property(mesh1, property, property_label)
        to_be_mapped2, _ = _evaluate_to_be_mapped_property(mesh2, property, property_label)
        if mode == 'MESH':
            
            if draw_los_vector:
                normalized_los_vector = mesh1.los_vector/np.linalg.norm(mesh1.los_vector)
                plot_ax.quiver(*(-1.5*axes_lim*normalized_los_vector), *((axes_lim*normalized_los_vector)/2),
                        color='red', linewidth=3., label='LOS vector')
                
            if draw_rotation_axes:
                normalized_rotation_axis1 = mesh1.rotation_axis/np.linalg.norm(mesh1.rotation_axis)
                normalized_rotation_axis2 = mesh2.rotation_axis/np.linalg.norm(mesh2.rotation_axis)
                
                plot_ax.quiver(*(mesh1.center+normalized_rotation_axis1*mesh1.radius*scale_radius), *(mesh1.radius*normalized_rotation_axis1*np.sqrt(scale_radius)),
                            color='black', linewidth=3., label='Rotation axis of mesh1')
                plot_ax.quiver(*(mesh2.center+normalized_rotation_axis2*mesh2.radius*scale_radius), *(mesh2.radius*normalized_rotation_axis2*np.sqrt(scale_radius)),
                            color='blue', linewidth=3., label='Rotation axis of mesh2')
            if i==0 and (draw_los_vector or draw_rotation_axes):
                plot_ax.legend()
            
            vs2_1 = mesh1.center+(mesh1.mesh_elements-mesh1.center)*scale_radius
            face_colors1 = mpl.colormaps[cmap](norm(to_be_mapped1))
            p1 = art3d.Poly3DCollection(vs2_1, facecolors=face_colors1, edgecolor="black", linewidths=0.01)
            
            vs2_2 = mesh2.center+(mesh2.mesh_elements-mesh2.center)*scale_radius
            face_colors2 = mpl.colormaps[cmap](norm(to_be_mapped2))
            p2 = art3d.Poly3DCollection(vs2_2, facecolors=face_colors2, edgecolor="black", linewidths=0.01)
            
            plot_ax.add_collection(p1)
            plot_ax.add_collection(p2)
            mappable.set_array([])
        else:
            centers1 = mesh1.center+(mesh1.centers-mesh1.center)*scale_radius
            p1 = plot_ax.scatter(centers1[:, 0], centers1[:, 1], centers1[:, 2],
                                c=to_be_mapped1, cmap=cmap, norm=norm)
            centers2 = mesh2.center+(mesh2.centers-mesh2.center)*scale_radius
            p2 = plot_ax.scatter(centers2[:, 0], centers2[:, 1], centers2[:, 2],
                                c=to_be_mapped2, cmap=cmap, norm=norm)
            
        cbar = fig.colorbar(mappable, shrink=0.45, pad=0.125, cax=cax)
        cbar.set_label(cbar_label, fontsize=12)
            
        camera.snap()
        
    animation = camera.animate()
    animation.save(filename)

    return animation



def plot_2D(mesh: MeshModel,
            property: Union[str, int] = DEFAULT_PROPERTY,
            axes: Optional[Tuple[plt.figure, plt.axes, plt.axes]] = None,
            cmap: str = 'turbo',
            x_index: int = 0,
            y_index: int = 1,
            property_label: Optional[str] = None):
    if x_index == y_index:
        raise ValueError('x_index and y_index cannot be the same index!')
    elif x_index >= 3 or y_index >= 3:
        raise ValueError('x_index and y_index must be 0, 1, or 2!')
    
    if axes is None:
        fig = plt.figure(figsize=(6, 5))
        spec = fig.add_gridspec(10, 12)
        plot_ax = fig.add_subplot(spec[:, :11])
        cbar_ax = fig.add_subplot(spec[2:8, 11])
    else:
        try:
            fig, plot_ax, cbar_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or (plt.figure, plt.axes, plt.axes) for the plot axis and colorbar axis")
    
    xy_labels = ['$X [R_\\odot]$', '$Y [R_\\odot]$', '$Z [R_\\odot]$']
    
    to_be_mapped, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)
    positive_mu_mask = mesh.mus>0

    plot_ax.scatter(mesh.centers[positive_mu_mask, x_index], mesh.centers[positive_mu_mask, y_index],
                c=to_be_mapped[positive_mu_mask], cmap=cmap)
    plot_ax.set_xlabel(xy_labels[x_index], fontsize=14)
    plot_ax.set_ylabel(xy_labels[y_index], fontsize=14)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)

    return fig, plot_ax, cbar_ax


def plot_3D_mesh_and_spectrum(mesh: MeshModel,
                              wavelengths: ArrayLike,
                              spectrum: ArrayLike,
                              mesh_plot_kwargs: Optional[Dict[str, Any]] = None):
    mesh_plot_kwargs = mesh_plot_kwargs or {}
    
    fig = plt.figure(figsize=(24, 10))
    spec = fig.add_gridspec(10, 24)
    plot_ax = fig.add_subplot(spec[:, :10], projection='3d')
    plot_ax.view_init(elev=30, azim=-60)

    spectrum_ax = fig.add_subplot(spec[3:7, 11:-1])
    spectrum_ax.set_xlabel('wavelength [$\\AA$]', fontsize=13)
    spectrum_ax.set_ylabel('intensity [erg/s/cm$^2$]', fontsize=13)

    spectrum_ax.plot(wavelengths, spectrum, color='black')
    return *plot_3D(mesh, axes=(fig, plot_ax), **mesh_plot_kwargs), spectrum_ax
