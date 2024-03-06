import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .model import Model
from typing import List, Optional


def model_scatter3d(model: Model,
                    index: int,
                    fig: Optional[plt.Figure] = None,
                    los: Optional[List[float]] = None,
                    cmap: Optional[str] = None) -> plt.Figure:
    if fig is None:
        fig = plt.figure(figsize=(10, 15))

    spec = fig.add_gridspec(10, 12)
    
    if los is not None:
        scatter_ax = fig.add_subplot(spec[:, :11], projection='3d')
        cbar_ax = fig.add_subplot(spec[3:6, 11])
    else:
        scatter_ax = fig.add_subplot(spec[:, :], projection='3d')
    
    scatter_ax.set_xlim3d(-1.5, 1.5)
    scatter_ax.set_ylim3d(-1.5, 1.5)
    scatter_ax.set_ylim3d(-1.5, 1.5)
    scatter_ax.set_xlabel('x [radii]')
    scatter_ax.set_ylabel('y [radii]')
    scatter_ax.set_zlabel('z [radii]')
    
    centers = model.mesh.face_centers(index)
    
    if los is not None:
        # Plot a line-of-sight arrow
        
        # Normalize
        norm_los = los/np.linalg.norm(los)
        plot_los = -1.*norm_los
        scatter_ax.quiver(*(np.clip(2.*norm_los, a_min=-2., a_max=2.)), *plot_los, color='red', linewidth=3.)
        
        mus = model.mesh.center_mus(index, los)
        cmap = cmap if cmap is not None else "turbo"
        
        p = scatter_ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=mus, cmap=cmap);
        cbar = fig.colorbar(p, cax=cbar_ax)
        cbar.set_label('$\mu$', fontsize=14)
        
    else:
        scatter_ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='black');
    
    return fig
