Installation
============

SPICE is published on PyPI as ``stellar-spice`` and installs as the ``spice``
package:

.. code-block:: bash

    pip install stellar-spice

The core install covers mesh modelling, blackbody and analytic line-profile
synthesis, native binaries, synthetic photometry, and plotting.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Extra
     - Enables
     - Installs
   * - ``[grid]``
     - Zarr-backed model-atmosphere grid interpolation
       (:doc:`spectral_grids`)
     - ``zarr``, ``pandas``, ``polars``, ``pyarrow``
   * - ``[phoebe]``
     - Importing Roche-lobe geometry and tidally distorted binary meshes
       from PHOEBE (:doc:`phoebe_integration`)
     - ``phoebe``
   * - ``[huggingface]``
     - Downloading pretrained emulator bundles from Hugging Face
     - ``huggingface-hub``
   * - ``[aemu]``
     - Neural-network spectrum emulators loaded as bundles
       (:doc:`aemu_integration`)
     - ``astro-emulators-toolkit``
   * - ``[dev]``
     - Running the test suite
     - ``pytest``, ``pytest-datadir``, ``coverage``

Extras combine freely, e.g.:

.. code-block:: bash

    pip install "stellar-spice[grid,phoebe]"

Verifying the install
---------------------

.. code-block:: python

    from spice.models import IcosphereModel
    from spice.spectrum import Blackbody

    bb = Blackbody()
    star = IcosphereModel.construct(100, 1., 1., bb.solar_parameters, bb.parameter_names)
    print(star.d_vertices.shape)

Platform notes
--------------

- SPICE is built on `JAX <https://jax.readthedocs.io>`_; a GPU is used
  automatically when a CUDA-enabled ``jaxlib`` is installed.
- On macOS, SPICE forces ``JAX_PLATFORMS=cpu`` unless you set the variable
  yourself — the experimental Metal backend is slower than CPU for this
  workload and numerically unreliable.
- Double precision is highly encouraged via
  ``jax.config.update("jax_enable_x64", True)``; see :doc:`conventions`.
