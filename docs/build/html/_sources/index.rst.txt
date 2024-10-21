SPICE: SPectra Integration Compiled Engine
===================================

SPICE (SPectra Integration Compiled Engine) is a Python library designed for simulating synthetic **spectra of inhomogeneous stellar surfaces**. SPICE offers the following capabilities:

The core principle of SPICE is numerical integration over a stellar surface. Here's how it works:

1. The stellar surface is divided into many small elements using tessellation.
2. For each surface element, a synthetic spectrum is calculated.
3. These individual spectra are then summed to produce the final, integrated spectrum of the entire stellar surface.

This approach allows SPICE to accurately model stars with inhomogeneous surfaces, including features like spots, temperature variations, or chemical abundance patterns.

1. **Stellar Surface Modeling**:
   - Create icosphere models of stellar surfaces with varying resolutions
   - Add spots and other surface features to the stellar models
   - Implement rotation and pulsations for stellar mesh models

2. **Spectral Synthesis**:
   - Generate synthetic spectra using various models (e.g., Blackbody, TransformerPayne)
   - Simulate observed flux for different stellar configurations with varying resolutions
   - Account for Doppler shifts in spectral calculations

3. **Synthetic Photometry**:
   - Calculate monochromatic and bolometric luminosities
   - Compute magnitudes in various photometric systems (e.g., AB, ST)
   - Support for multiple standard filters (e.g., Johnson, Bessel, Gaia)

4. **Binary System Modeling**:
   - Simulate synthetic spectra and photometric for binary star systems
   - Account for eclipsing

5. **Data Visualization**:
   - 3D plotting of stellar surface models
   - Spectrum visualization tools

6. **Performance Optimization**:
   - Utilize JAX for efficient computations
   - Support for GPU acceleration

7. **Interoperability**:
   - Integration with other astronomical tools and libraries (e.g., PHOEBE, Synphot)


The underlying spectrum model is configurable - we provide a machine-learning based spectrum emulator, `Transformer Payne <https://github.com/RozanskiT/transformer_payne>`_

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   examples
   mesh
   binaries
   phoebe_integration
   api
   troubleshooting