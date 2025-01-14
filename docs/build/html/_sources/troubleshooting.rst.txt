Troubleshooting
===================================

Resource Exhaustion
------------------------

If you run into resource exhaustion while synthesizing spectra, try changing the ```chunk_size```. The default chunk size is 1024.

.. code-block:: python

    t = TransformerPayne.download()
    
    flux = simulate_observed_flux(t.intensity, model, jnp.log10(wavelengths), chunk_size=128)


Buffer Comparator Difference
------------------------------

If you run into a similiar error:

.. code-block:: python

    E0112 21:13:14.359553  650420 buffer_comparator.cc:157] Difference at 41: 0, expected 4.13377
2025-01-12 21:13:14.359561: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1175] Results do not match the reference. This is likely a bug/unexpected loss of precision.

Usually, this can be fixed by updating the JAX config:

.. code-block:: python

    import jax
    
    jax.config.update("jax_default_matmul_precision", "high")

