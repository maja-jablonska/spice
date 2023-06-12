import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional, Tuple
import numpy as np
from models import MeshModel


COLORMAP_PROPERTIES = ['mus', 'los_velocities']


def plot_3D(mesh: MeshModel,
            property: str = 'mus',
            axes: Optional[Tuple[plt.figure, plt.axes, plt.axes]] = None,
            cmap: str = 'turbo'):
    if property not in COLORMAP_PROPERTIES:
        raise ValueError(f'Invalid property {property} - must be one of ({",".join(COLORMAP_PROPERTIES)})')
    
    if axes is None:
        fig = plt.figure(figsize=(10, 12))
        spec = fig.add_gridspec(10, 12)
        plot_ax = fig.add_subplot(spec[:, :11], projection='3d')
        plot_ax.view_init(elev=30, azim=-60)
        cbar_ax = fig.add_subplot(spec[2:8, 11])
    else:
        try:
            fig, plot_ax, cbar_ax = axes
        except ValueError:
            raise ValueError("Pass either no axes or (plt.figure, plt.axes, plt.axes) for the plot axis and colorbar axis")
    axes_lim = 1.5*mesh.radius
    plot_ax.set_xlim3d(-axes_lim, axes_lim)
    plot_ax.set_ylim3d(-axes_lim, axes_lim)
    plot_ax.set_zlim3d(-axes_lim, axes_lim)
    plot_ax.set_xlabel('$X [R_\odot]$', fontsize=14)
    plot_ax.set_ylabel('$Y [R_\odot]$', fontsize=14)
    plot_ax.set_zlabel('$Z [R_\odot]$', fontsize=14)

    plot_ax.quiver(*(-2.0*mesh.radius*mesh.los_vector), *mesh.los_vector,
                   color='red', linewidth=3., label='LOS vector')
    plot_ax.quiver(*(0.75*mesh.radius*mesh.rotation_axis), *mesh.rotation_axis,
                   color='black', linewidth=3., label='Rotation axis')
    plot_ax.legend()

    to_be_mapped = getattr(mesh, property)
    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())

    p = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                        c=to_be_mapped, cmap=cmap, norm=norm)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(r'$\mu$')

# def plot_velocities_for_phase(mesh: MeshModel,
#                               phase_index: int,
#                               los_vector: np.ndarray,
#                               cmap: str = 'turbo',
#                               axes: Optional[Tuple[plt.axes, plt.axes]] = None):
#     los_vels = np.nan_to_num(np.array(mesh.get_los_velocities(los_vector)))
#     norm = mpl.colors.Normalize(vmin=los_vels.min(), vmax=los_vels.max())
    
#     if axes is None:
#         fig = plt.figure(figsize=(10, 15))
#         spec = fig.add_gridspec(12, 12)
#         ax = fig.add_subplot(spec[:, :11], projection='3d')
#         cbar_ax = fig.add_subplot(spec[3:9, 11])
#     else:
#         ax, cbar_ax = axes
        
#     axes_lim = 1.5*mesh.radius
#     ax.set_xlim3d(-axes_lim, axes_lim)
#     ax.set_ylim3d(-axes_lim, axes_lim)
#     ax.set_ylim3d(-axes_lim, axes_lim)

#     ax.quiver(*(-1.5*mesh.radius*los_vector), *los_vector, color='red', linewidth=3.)
#     ax.quiver(0., 0., 0., *mesh.rotation_axis, color='black', linewidth=3.)
#     centers = mesh.centers[phase_index]
    
#     p = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
#                    c=los_vels[phase_index], cmap=cmap, norm=norm)
    
#     cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
#     cbar.set_label('LOS velocity')
    
#     return ax
