SPICE: SPectra Integration Compiled Engine
===================================

A library for synthetic spectra of inhomogenous stellar surfaces.

The principle of SPICE is a numerical integration of a stellar surface with various features - for every element of a tesselated
stellar surface, a synthetic spectrum is calculated. The result spectrum is the sum of individual elements' spectra.

The underlying spectrum model is configurable - we provide a machine-learning based spectrum emulator, `Transformer Payne <https://github.com/RozanskiT/transformer_payne>`_

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   examples
   api