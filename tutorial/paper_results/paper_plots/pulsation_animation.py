#!/usr/bin/env python3
"""
Pulsation Stacking Animation

This script creates an animation showing:
1. A moving dot tracing the current amplitude on the function plot
2. Emergent radial velocities on spherical harmonics by multiplying field values by the current amplitude
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    import cmasher as cmr
    cmap_default = "cmr.guppy"
except ImportError:
    cmr = None
    cmap_default = "viridis"
from matplotlib import animation
import matplotlib.patches as patches

# ---------- math: associated Legendre & real Y_lm (Condon–Shortley) ----------

def _double_factorial_odd(m: int):
    if m == 0:
        return 1.0
    m_f = float(m)
    return jnp.exp(gammaln(2.0*m_f + 1.0) - m_f*jnp.log(2.0) - gammaln(m_f + 1.0))

def _P_lm(l: int, m: int, x):
    mabs = int(abs(m))
    Pmm = ((-1.0)**mabs) * _double_factorial_odd(mabs) * (1.0 - x*x)**(0.5*mabs)
    if l == mabs:
        return Pmm
    Pm1m = x*(2*mabs + 1.0)*Pmm
    if l == mabs + 1:
        return Pm1m
    P_lm_prev2 = Pmm
    P_lm_prev1 = Pm1m
    for ell in range(mabs + 2, l + 1):
        ell_f = float(ell)
        P_lm_curr = ((2.0*ell_f - 1.0)*x*P_lm_prev1 - (ell_f + mabs - 1.0)*P_lm_prev2) / (ell_f - mabs)
        P_lm_prev2, P_lm_prev1 = P_lm_prev1, P_lm_curr
    return P_lm_prev1

def real_sph_harm(l: int, m: int, theta, phi):
    mabs = abs(m)
    x = jnp.cos(theta)
    Plm = _P_lm(l, mabs, x)
    l_f = float(l)
    m_f = float(mabs)
    Nlm = jnp.sqrt((2.0*l_f + 1.0)/(4.0*jnp.pi) *
                   jnp.exp(gammaln(l_f - m_f + 1.0) - gammaln(l_f + m_f + 1.0)))
    base = Nlm * Plm
    if m > 0:
        return jnp.sqrt(2.0) * base * jnp.cos(m * phi)
    elif m < 0:
        return jnp.sqrt(2.0) * base * jnp.sin(mabs * phi)
    else:
        return base

# ---------- rendering helpers ----------

def sphere_xyz(theta, phi, r=1.0):
    X = r * jnp.sin(theta) * jnp.cos(phi)
    Y = r * jnp.sin(theta) * jnp.sin(phi)
    Z = r * jnp.cos(theta)
    return X, Y, Z

def plot_field_on_sphere(ax, field, theta, phi, cmap="cmr.redshift",
                         vmin=None, vmax=None, add_colorbar=False, norm=None, sm=None):
    X = jnp.sin(theta) * jnp.cos(phi)
    Y = jnp.sin(theta) * jnp.sin(phi)
    Z = jnp.cos(theta)

    field = np.array(field)
    # Use the provided norm and sm for consistent color mapping
    if norm is None:
        norm = Normalize(vmin=np.nanmin(field), vmax=np.nanmax(field))
    if sm is None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

    ax.plot_surface(np.array(X), np.array(Y), np.array(Z),
                    rstride=1, cstride=1,
                    facecolors=cm.get_cmap(cmap)(norm(field)),
                    linewidth=0, antialiased=True, shade=False)

    # no grid/labels
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    # Only add colorbar if requested (handled outside for global colorbar)
    if add_colorbar and sm is not None:
        cb = ax.figure.colorbar(sm, ax=ax, pad=0.04, fraction=0.05, shrink=0.9)
        cb.set_label("Field value (shared colormap)")

def rescale_to_unit(field):
    a = jnp.max(jnp.abs(field))
    return jnp.where(a > 0, field / a, field)

def main():
    # ---------- choose components ----------
    components = [
        (1, 0, 1.0),
        (3, 3, 0.8),
        (3, 1, -0.6),
    ]

    # ---------- grid ----------
    n_th, n_ph = 300, 600
    theta = jnp.linspace(0.0, jnp.pi, n_th)
    phi   = jnp.linspace(0.0, 2.0*jnp.pi, n_ph)
    TH, PH = jnp.meshgrid(theta, phi, indexing="ij")

    # Calculate individual fields
    fields = []
    for (l, m, a) in components:
        Y = real_sph_harm(l, m, TH, PH)
        fields.append(a * Y)

    sum_field = jnp.sum(jnp.stack(fields, axis=0), axis=0)
    sum_rescaled = rescale_to_unit(sum_field)

    # Create time series for animation
    T = np.linspace(0, 2*np.pi, 100)  # One full cycle
    amplitude_function = np.cos(T)  # Simple cosine amplitude variation

    print(f"Components: {components}")
    print(f"Time points: {len(T)}")
    print(f"Amplitude range: [{amplitude_function.min():.3f}, {amplitude_function.max():.3f}]")

    # ---------- create animation figure ----------
    fig = plt.figure(figsize=(16, 8))

    # Left side: amplitude function with moving dot
    ax_amplitude = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
    ax_amplitude.plot(T, amplitude_function, 'b-', linewidth=2, label='Amplitude Function')
    ax_amplitude.set_xlabel('Time')
    ax_amplitude.set_ylabel('Amplitude')
    ax_amplitude.set_title('Current Amplitude')
    ax_amplitude.grid(True, alpha=0.3)
    ax_amplitude.legend()
    ax_amplitude.set_xlim(0, 2*np.pi)
    ax_amplitude.set_ylim(-1.2, 1.2)

    # Add moving dot
    dot, = ax_amplitude.plot([], [], 'ro', markersize=10, label='Current Position')

    # Right side: spherical harmonics with emergent radial velocities
    ax_sphere = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2, projection='3d')
    ax_sphere.set_title('Emergent Radial Velocities')

    # Compute global vmin/vmax for consistent color mapping
    all_fields = fields + [sum_field, sum_rescaled]
    global_vmin = float(np.nanmin([np.nanmin(np.array(f)) for f in all_fields]))
    global_vmax = float(np.nanmax([np.nanmax(np.array(f)) for f in all_fields]))
    norm = Normalize(vmin=global_vmin, vmax=global_vmax)
    cmap = cmap_default
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Initial sphere plot
    plot_field_on_sphere(ax_sphere, sum_field, TH, PH, 
                         cmap=cmap, norm=norm, sm=sm, add_colorbar=False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Radial Velocity")

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    # Animation function
    def animate(frame):
        # Update moving dot position
        current_time = T[frame]
        current_amplitude = amplitude_function[frame]
        dot.set_data([current_time], [current_amplitude])
        
        # Clear previous sphere
        ax_sphere.clear()
        ax_sphere.set_title(f'Emergent Radial Velocities (t={current_time:.2f}, A={current_amplitude:.2f})')
        
        # Calculate emergent radial velocities by multiplying field by current amplitude
        emergent_field = sum_field * current_amplitude
        
        # Plot the emergent field on sphere
        plot_field_on_sphere(ax_sphere, emergent_field, TH, PH, 
                             cmap=cmap, norm=norm, sm=sm, add_colorbar=False)
        
        return [dot]

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(T), 
                                  interval=100, blit=False, repeat=True)

    plt.show()

    # Save animation as GIF
    try:
        from matplotlib.animation import PillowWriter
        
        # Save as GIF
        writer = PillowWriter(fps=10)
        ani.save('pulsation_animation.gif', writer=writer, dpi=100)
        print("Animation saved as 'pulsation_animation.gif'")
        
    except ImportError:
        print("Pillow not available. Install with: pip install Pillow")
        print("Animation can still be viewed interactively in the notebook")

    # Alternative: Save as MP4 if available
    try:
        from matplotlib.animation import FFMpegWriter
        
        # Save as MP4
        writer = FFMpegWriter(fps=10, metadata=dict(artist='Pulsation Animation'))
        ani.save('pulsation_animation.mp4', writer=writer, dpi=100)
        print("Animation saved as 'pulsation_animation.mp4'")
        
    except ImportError:
        print("FFmpeg not available. Install with: pip install ffmpeg-python")
        print("Or use the GIF version above")

if __name__ == "__main__":
    main()
