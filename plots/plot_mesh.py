import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional, Tuple, Union
import numpy as np
from models import MeshModel
from jax.typing import ArrayLike

COLORMAP_PROPERTIES = ['mus', 'los_velocities']
DEFAULT_PROPERTY = 'mus'

DEFAULT_PLOT_PROPERTY_LABELS = {
    'mus': r'$\mu$',
    'los_velocities': 'LOS velocity [km/s]'
}

def _evaluate_to_be_mapped_property(mesh: MeshModel,
                                    property: Union[str, int] = DEFAULT_PROPERTY,
                                    property_label: Optional[str] = None) -> ArrayLike:
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
            axes: Optional[Tuple[plt.figure, plt.axes, plt.axes]] = None,
            cmap: str = 'turbo',
            property_label: Optional[str] = None):
    to_be_mapped, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)

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

    norm = mpl.colors.Normalize(vmin=to_be_mapped.min(), vmax=to_be_mapped.max())

    p = plot_ax.scatter(mesh.centers[:, 0], mesh.centers[:, 1], mesh.centers[:, 2],
                        c=to_be_mapped, cmap=cmap, norm=norm)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=12)


def plot_2D(mesh: MeshModel,
            property: Union[str, int] = DEFAULT_PROPERTY,
            cmap: str = 'turbo',
            x_index: int = 0,
            y_index: int = 1,
            property_label: Optional[str] = None):
    if x_index == y_index:
        raise ValueError('x_index and y_index cannot be the same index!')
    elif x_index >= 3 or y_index >= 3:
        raise ValueError('x_index and y_index must be 0, 1, or 2!')
    
    xy_labels = ['$X [R_\odot]$', '$Y [R_\odot]$', '$Z [R_\odot]$']
    
    to_be_mapped, cbar_label = _evaluate_to_be_mapped_property(mesh, property, property_label)
    positive_mu_mask = mesh.mus>0

    plt.scatter(mesh.centers[positive_mu_mask, x_index], mesh.centers[positive_mu_mask, y_index],
                c=to_be_mapped[positive_mu_mask], cmap=cmap)
    plt.gca().set_xlabel(xy_labels[x_index], fontsize=14)
    plt.gca().set_ylabel(xy_labels[y_index], fontsize=14)
    
    cbar = plt.colorbar()
    cbar.set_label(cbar_label, fontsize=12)