def __getattr__(name):
    _lazy_imports = {
        # model.py
        "Model": ".model",
        # mesh_generation.py
        "icosphere": ".mesh_generation",
        # mesh_model.py
        "IcosphereModel": ".mesh_model",
        "MeshModel": ".mesh_model",
        # spots.py
        "add_spherical_harmonic_spot": ".spots",
        "add_spherical_harmonic_spots": ".spots",
        # binary.py
        "Binary": ".binary",
        # utils.py
        "lat_to_theta": ".utils",
        "lon_to_phi": ".utils",
        "theta_to_lat": ".utils",
        "phi_to_lon": ".utils",
        "horizontal_to_radial_ratio": ".utils",
        # eclipse_utils.py
        "find_eclipses": ".eclipse_utils",
    }

    # PHOEBE-related imports
    _phoebe_imports = {
        "PhoebeModel": ".phoebe_model",
        "PhoebeBinary": ".binary",
    }

    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)

    if name in _phoebe_imports:
        try:
            import importlib
            module = importlib.import_module(_phoebe_imports[name], __package__)
            return getattr(module, name)
        except Exception:
            raise AttributeError(
                f"{name} requires PHOEBE to be installed"
            )

    if name == "PHOEBE_AVAILABLE":
        try:
            import importlib
            importlib.import_module(".phoebe_model", __package__)
            return True
        except Exception:
            return False

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
