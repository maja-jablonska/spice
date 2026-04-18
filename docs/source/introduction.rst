Introduction
============

SPICE (SPectra Integration Compiled Engine) is a Python library for simulating synthetic spectra of stars with inhomogeneous surfaces. It uses numerical integration over tessellated stellar surfaces to generate accurate synthetic spectra that account for surface features, rotation, pulsations, and binary interactions.

Core Concepts
------------

Surface Integration
^^^^^^^^^^^^^^^^^
The fundamental approach in SPICE is to:

1. Divide the stellar surface into small elements using icosphere tessellation
2. Calculate synthetic spectra for each surface element 
3. Sum the individual spectra to produce the integrated spectrum

This enables accurate modeling of stars with:

- Surface spots and temperature variations
- Chemical abundance patterns
- Rotation and pulsation effects
- Binary interactions and eclipses

Key Components
-------------

Mesh Models
^^^^^^^^^^
SPICE uses triangular mesh models to represent stellar surfaces. The ``IcosphereModel`` class provides:

- Configurable resolution through vertex count
- Surface element properties (areas, centers, normals)
- Support for spots and other surface features
- Rotation and pulsation capabilities

Binary Systems
^^^^^^^^^^^^^
The ``Binary`` class enables modeling of binary star systems with:

- Keplerian orbital motion
- Mutual eclipses and occultations 
- Combined spectra calculation
- Integration with PHOEBE for binary parameters

Spectral Synthesis
^^^^^^^^^^^^^^^^
SPICE supports multiple spectrum models:

- Simple blackbody radiation
- Machine learning based TransformerPayne emulator
- Custom model integration

Performance
----------
SPICE leverages JAX for:

- Just-in-time compilation
- Automatic differentiation
- GPU acceleration
- Vectorized operations

This enables efficient computation of synthetic spectra even for high-resolution surface meshes and complex binary configurations.

Getting Started
-------------
The following sections will guide you through:

- Creating and manipulating mesh models
- Setting up binary systems
- Generating synthetic spectra
- Visualizing results
