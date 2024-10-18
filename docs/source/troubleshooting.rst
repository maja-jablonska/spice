Troubleshooting
===================================

Resource Exhaustion
------------------------

If you run into resource exhaustion while synthesizing spectra, try changing the ```chunk_size```. The default chunk size is 1024.

.. code-block:: python

    t = TransformerPayne.download()
    
    flux = simulate_observed_flux(t.intensity, model, jnp.log10(wavelengths), chunk_size=128)
