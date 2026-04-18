def __getattr__(name):
    _lazy_imports = {
        "plot_3D": ".plot_mesh",
        "plot_3D_sequence": ".plot_mesh",
        "plot_3D_binary": ".plot_mesh",
        "plot_2D": ".plot_mesh",
        "plot_3D_mesh_and_spectrum": ".plot_mesh",
        # plot_pulsations.py — scalar diagnostics and flat surface maps
        # of the pulsation velocity field
        "compute_pulsation_field": ".plot_pulsations",
        "plot_pulsation_map": ".plot_pulsations",
        "plot_pulsation_components": ".plot_pulsations",
        "plot_pulsation_streamlines": ".plot_pulsations",
        "plot_pulsation_cross_section": ".plot_pulsations",
        "plot_pulsation_3D_sparse": ".plot_pulsations",
        "animate_pulsation_phase": ".plot_pulsations",
        "PULSATION_FIELDS": ".plot_pulsations",
    }

    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
