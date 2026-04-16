"""Public exports for :mod:`spice.models`.

This package intentionally uses lazy imports so lightweight use-cases do not pay
for optional/heavy dependencies (PHOEBE, socket stack, occlusion helpers, etc.)
until those symbols are actually requested.
"""

from importlib import import_module

PHOEBE_AVAILABLE = False

_EXPORTS = {
    "Model": (".model", "Model"),
    "icosphere": (".mesh_generation", "icosphere"),
    "IcosphereModel": (".mesh_model", "IcosphereModel"),
    "MeshModel": (".mesh_model", "MeshModel"),
    "add_spherical_harmonic_spot": (".spots", "add_spherical_harmonic_spot"),
    "add_spherical_harmonic_spots": (".spots", "add_spherical_harmonic_spots"),
    "Binary": (".binary", "Binary"),
    "find_eclipses": (".eclipse_utils", "find_eclipses"),
    "lat_to_theta": (".utils", "lat_to_theta"),
    "lon_to_phi": (".utils", "lon_to_phi"),
    "theta_to_lat": (".utils", "theta_to_lat"),
    "phi_to_lon": (".utils", "phi_to_lon"),
}

__all__ = [*list(_EXPORTS.keys()), "PhoebeModel", "PhoebeBinary", "PHOEBE_AVAILABLE"]


def __getattr__(name: str):
    global PHOEBE_AVAILABLE

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)

    if name in {"PhoebeModel", "PhoebeBinary"}:
        try:
            phoebe_model = import_module(".phoebe_model", __name__)
            binary = import_module(".binary", __name__)
            PHOEBE_AVAILABLE = True
            return getattr(phoebe_model if name == "PhoebeModel" else binary, name)
        except ImportError as exc:
            PHOEBE_AVAILABLE = False
            raise AttributeError(
                f"{name} is unavailable because optional PHOEBE dependencies could not be imported"
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
