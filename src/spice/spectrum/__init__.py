def __getattr__(name):
    _lazy_imports = {
        # spectrum.py
        "simulate_observed_flux": ".spectrum",
        "simulate_monochromatic_luminosity": ".spectrum",
        "luminosity": ".spectrum",
        "AB_passband_luminosity": ".spectrum",
        "absolute_bol_luminosity": ".spectrum",
        # utils.py
        "ERG_S_TO_W": ".utils",
        "SPHERE_STERADIAN": ".utils",
        "ZERO_POINT_LUM_W": ".utils",
        "apply_spectral_resolution": ".utils",
        # spectrum_emulator.py
        "SpectrumEmulator": ".spectrum_emulator",
        # gaussian_line_emulator.py
        "GaussianLineEmulator": ".gaussian_line_emulator",
        # limb_darkening.py
        "limb_darkening": ".limb_darkening",
        "get_limb_darkening_law_id": ".limb_darkening",
        # line_profile.py
        "get_line_profile_id": ".line_profile",
        "line_profile": ".line_profile",
        # physical_line_emulator.py
        "PhysicalLineEmulator": ".physical_line_emulator",
        # lazy_zarr_interpolator.py
        "GridIndex": ".lazy_zarr_interpolator",
        "SparseGridIndex": ".lazy_zarr_interpolator",
        "LazyZarrInterpolator": ".lazy_zarr_interpolator",
        "IntensityLazyZarrInterpolator": ".lazy_zarr_interpolator",
        "FluxLazyZarrInterpolator": ".lazy_zarr_interpolator",
    }

    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
