def __getattr__(name):
    _lazy_imports = {
        "plot_3D": ".plot_mesh",
        "plot_3D_sequence": ".plot_mesh",
        "plot_3D_binary": ".plot_mesh",
        "plot_2D": ".plot_mesh",
        "plot_3D_mesh_and_spectrum": ".plot_mesh",
        # plot_pulsations.py — per-element scalar projections of
        # mesh.pulsation_velocities
        "compute_pulsation_scalar": ".plot_pulsations",
        "plot_pulsation_map": ".plot_pulsations",
        "plot_pulsation_components": ".plot_pulsations",
        "plot_pulsation_cross_section": ".plot_pulsations",
        "plot_pulsation_3D_sparse": ".plot_pulsations",
        "plot_pulsation_patch_zoom": ".plot_pulsations",
        "plot_pulsation_disk_with_patch_zoom": ".plot_pulsations",
        "plot_pulsation_phase_grid": ".plot_pulsations",
        "plot_pulsation_comet": ".plot_pulsations",
        "animate_pulsation_phase": ".plot_pulsations",
        "animate_observed_disk": ".plot_pulsations",
        "PULSATION_FIELDS": ".plot_pulsations",
    }

    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
