import matplotlib.pyplot as plt
import matplotlib as mpl
from models import MeshModel
from typing import Optional, Tuple
import numpy as np


def plot_velocities_for_phase(mesh: MeshModel,
                              phase_index: int,
                              los_vector: np.ndarray,
                              cmap: str = 'turbo',
                              axes: Optional[Tuple[plt.axes, plt.axes]] = None):
    los_vels = np.nan_to_num(np.array(mesh.get_los_velocities(los_vector)))
    norm = mpl.colors.Normalize(vmin=los_vels.min(), vmax=los_vels.max())
    
    if axes is None:
        fig = plt.figure(figsize=(10, 15))
        spec = fig.add_gridspec(12, 12)
        ax = fig.add_subplot(spec[:, :11], projection='3d')
        cbar_ax = fig.add_subplot(spec[3:9, 11])
    else:
        ax, cbar_ax = axes
        
    axes_lim = 1.5*mesh.radius
    ax.set_xlim3d(-axes_lim, axes_lim)
    ax.set_ylim3d(-axes_lim, axes_lim)
    ax.set_ylim3d(-axes_lim, axes_lim)

    ax.quiver(*(-1.5*mesh.radius*los_vector), *los_vector, color='red', linewidth=3.)
    ax.quiver(0., 0., 0., *mesh.rotation_axis, color='black', linewidth=3.)
    centers = mesh.centers[phase_index]
    
    p = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   c=los_vels[phase_index], cmap=cmap, norm=norm)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('LOS velocity')
    
    return ax
