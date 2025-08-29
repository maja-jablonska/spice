# Pulsation Stacking Animation

This directory contains an animation that visualizes pulsation stacking with emergent radial velocities on spherical harmonics.

## Files

- `pulsation_animation.py` - Python script version
- `pulsation_animation.ipynb` - Jupyter notebook version (interactive)

## What the Animation Shows

The animation demonstrates:

1. **Left Panel**: A moving red dot traces the current amplitude along a cosine function over time
2. **Right Panel**: A 3D sphere showing emergent radial velocities calculated by multiplying the spherical harmonic field values by the current amplitude

## Key Features

- **Real-time synchronization**: The moving dot position corresponds exactly to the current amplitude value
- **Dynamic spherical harmonics**: The sphere's color pattern changes in real-time as the amplitude varies
- **Emergent radial velocities**: Shows how the base spherical harmonic pattern is modulated by the time-varying amplitude
- **Consistent color mapping**: Uses a shared colormap across all frames for visual consistency

## How to Run

### Option 1: Python Script

```bash
python pulsation_animation.py
```

### Option 2: Jupyter Notebook

```bash
jupyter notebook pulsation_animation.ipynb
```

## Dependencies

Required packages:

- `jax`
- `numpy`
- `matplotlib`
- `cmasher` (for colormaps)

Optional packages for saving:

- `Pillow` (for GIF export)
- `ffmpeg-python` (for MP4 export)

## Customization

You can modify the animation by changing:

- **Components**: Edit the `components` list to use different spherical harmonic modes
- **Time function**: Change `amplitude_function` to use different amplitude variations
- **Animation speed**: Adjust the `interval` parameter in `FuncAnimation`
- **Resolution**: Modify `n_th` and `n_ph` for different sphere resolutions

## Output

The animation will display interactively and can be saved as:

- GIF file (`pulsation_animation.gif`)
- MP4 file (`pulsation_animation.mp4`) if FFmpeg is available

## Physics Interpretation

This animation visualizes how stellar pulsations create time-varying radial velocity patterns on the stellar surface. The spherical harmonics represent the spatial structure of the pulsation modes, while the amplitude function shows how these modes are excited over time. The emergent radial velocities show the actual observable velocity field at any given moment.

