from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.setdefault("PHOEBE_ENABLE_ONLINE_PASSBANDS", "FALSE")

import multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

import argparse
import gc
import importlib.metadata
import json
import logging
import sys
import time
import traceback
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import phoebe
from astropy import units as u
from phoebe.parameters.dataset import _mesh_columns
from tqdm import tqdm
from transformer_payne import Blackbody

try:
    import zarr
except ImportError as exc:
    zarr = None
    ZARR_IMPORT_ERROR = exc
else:
    ZARR_IMPORT_ERROR = None

try:
    from numcodecs import Blosc
except ImportError:
    Blosc = None

try:
    import psutil
except ImportError:
    psutil = None

from spice.models.binary import Binary, add_orbit, evaluate_orbit_at_times
from spice.models.mesh_model import IcosphereModel
from spice.models.mesh_view import get_mesh_view
from spice.models.orbit_utils import eclipse_timestamps_kepler
from spice.models.phoebe_utils import Component, PhoebeConfig
from spice.spectrum.filter import Bolometric
from spice.spectrum.spectrum import AB_passband_luminosity, simulate_observed_flux


phoebe.multiprocessing_set_nprocs(1)
LOGGER = logging.getLogger(__name__)

DAY_TO_YEAR = 0.0027378507871321013
DEG_TO_RAD = 0.017453292519943295
LOS_VECTOR = jnp.array([0.0, 0.0, -1.0])

MESH_DATASET_NAME = "mesh01"
ORBIT_DATASET_NAME = "orb01"
LC_DATASET_NAME = "lc_bolometric"

PARAMETER_ORDER = [
    "inclination_deg",
    "period_days",
    "q",
    "ecc",
    "primary_mass_msun",
]

PHOEBE_SCALAR_PARAMETER_SPECS = {
    "inclination_deg": ("incl@binary@component", 1.0),
    "period_days": ("period@binary@component", 1.0),
    "q": ("q@binary@component", 1.0),
    "ecc": ("ecc@binary@component", 1.0),
    "primary_mass_msun": ("mass@primary@component", 1.0),
    "secondary_mass_msun": ("mass@secondary@component", 1.0),
    "sma_rsun": ("sma@binary@component", 1.0),
    "primary_requiv_rsun": ("requiv@primary@component", 1.0),
    "secondary_requiv_rsun": ("requiv@secondary@component", 1.0),
    "t0_perpass_days": ("t0_perpass@binary@component", 1.0),
    "t0_ref_days": ("t0_ref@binary@component", 1.0),
    "mean_anomaly_deg": ("mean_anom@binary@component", 1.0),
    "omega_deg": ("per0@binary@component", 1.0),
    "long_an_deg": ("long_an@binary@component", 1.0),
    "vgamma_kms": ("vgamma", 1.0),
    "distance_pc": ("distance", 1.0),
}


def require_zarr() -> None:
    if zarr is None:
        raise ImportError(
            "This script requires `zarr`. Install it in the same environment as PHOEBE/SPICE."
        ) from ZARR_IMPORT_ERROR


def get_memory_usage() -> float:
    if psutil is None:
        return float("nan")
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(stage_name: str, start_time: float | None = None) -> float:
    current_time = time.time()
    memory_mb = get_memory_usage()
    if start_time is not None:
        elapsed = current_time - start_time
        LOGGER.info("[MEMORY] %s: %.1f MB (elapsed: %.1fs)", stage_name, memory_mb, elapsed)
    else:
        LOGGER.info("[MEMORY] %s: %.1f MB", stage_name, memory_mb)
    return current_time


def force_garbage_collection() -> None:
    gc.collect()
    jax.clear_caches()
    try:
        from jax.lib import xla_bridge

        xla_bridge.get_backend().defragment()
    except Exception:
        pass
    gc.collect()


def safe_distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def to_attr_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): to_attr_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_attr_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def default_log_path_for_output(output_path: Path) -> Path:
    return Path(f"{output_path}.log")


def default_failure_dir_for_output(output_path: Path) -> Path:
    return Path(f"{output_path}.failures")


def setup_logging(log_path: Path, log_level: str, overwrite: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper())
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w" if overwrite else "a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings.simplefilter("default")
    LOGGER.info("Logging configured")
    LOGGER.info("Log file: %s", log_path)


def case_label(case: Mapping[str, float]) -> str:
    return (
        f"case_{int(case['case_index']):06d} "
        f"(incl={case['inclination_deg']}, "
        f"period={case['period_days']}, "
        f"q={case['q']}, "
        f"ecc={case['ecc']}, "
        f"primary_mass={case['primary_mass_msun']})"
    )


def extract_bundle_failure_snapshot(
    bundle: phoebe.Bundle | None,
    use_legacy_long_an_conversion: bool,
) -> Dict[str, Any]:
    if bundle is None:
        return {}

    snapshot: Dict[str, Any] = {}
    try:
        snapshot["phoebe_scalars"] = extract_phoebe_scalar_parameters(
            bundle,
            use_legacy_long_an_conversion=use_legacy_long_an_conversion,
        )
    except Exception as exc:
        snapshot["phoebe_scalars_error"] = str(exc)

    try:
        snapshot["datasets"] = list(bundle.datasets)
    except Exception as exc:
        snapshot["datasets_error"] = str(exc)

    for qualifier, label in (
        (f"times@{MESH_DATASET_NAME}", "mesh_times_days"),
        (f"compute_times@{ORBIT_DATASET_NAME}", "orbit_compute_times_days"),
        (f"compute_times@{LC_DATASET_NAME}", "lc_compute_times_days"),
    ):
        try:
            snapshot[label] = np.asarray(bundle.get_parameter(qualifier).value).tolist()
        except Exception:
            continue

    return snapshot


def dump_failed_case_file(
    failure_dir: Path,
    case: Mapping[str, float],
    stage_name: str,
    error_message: str,
    traceback_text: str,
    bundle: phoebe.Bundle | None,
    use_legacy_long_an_conversion: bool,
    times_days: np.ndarray | None,
    eclipse_metadata: Mapping[str, Any] | None,
    phoebe_scalars: Mapping[str, Any] | None,
) -> Path:
    failure_dir.mkdir(parents=True, exist_ok=True)
    case_id = f"case_{int(case['case_index']):06d}"
    failure_path = failure_dir / f"{case_id}.json"

    payload: Dict[str, Any] = {
        "case": {key: to_attr_value(value) for key, value in case.items()},
        "failed_stage": stage_name,
        "error_message": error_message,
        "traceback": traceback_text,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "times_days": None if times_days is None else np.asarray(times_days).tolist(),
        "eclipse_metadata": None if eclipse_metadata is None else to_attr_value(dict(eclipse_metadata)),
        "phoebe_scalars": None if phoebe_scalars is None else to_attr_value(dict(phoebe_scalars)),
        "bundle_snapshot": extract_bundle_failure_snapshot(
            bundle,
            use_legacy_long_an_conversion=use_legacy_long_an_conversion,
        ),
    }
    payload["phoebe_error"] = error_message if "phoebe" in error_message.lower() else None

    failure_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return failure_path


def default_compressor():
    if zarr is not None and hasattr(zarr, "codecs") and hasattr(zarr.codecs, "BloscCodec"):
        return zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    if Blosc is None:
        return None
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def default_chunks(array: np.ndarray) -> Tuple[int, ...] | None:
    if array.ndim == 0:
        return None
    if array.ndim == 1:
        return (min(array.shape[0], 4096),)
    if array.ndim == 2:
        return (min(array.shape[0], 256), array.shape[1])
    if array.ndim == 3:
        return (1, min(array.shape[1], 2048), array.shape[2])
    if array.ndim == 4:
        return (1, min(array.shape[1], 512), array.shape[2], array.shape[3])
    return (1, *array.shape[1:])


def write_array(group: "zarr.Group", name: str, data: Any, compressor=None) -> None:
    array = np.asarray(data)
    kwargs: Dict[str, Any] = {"data": array, "overwrite": True}
    chunks = default_chunks(array)
    if chunks is not None:
        kwargs["chunks"] = chunks

    create_array = getattr(group, "create_array", None)
    if create_array is not None:
        if compressor is not None and array.dtype.kind not in {"U", "S", "O"}:
            if zarr is not None and hasattr(zarr, "codecs") and hasattr(zarr.codecs, "BloscCodec") and isinstance(
                compressor, zarr.codecs.BloscCodec
            ):
                kwargs["compressors"] = (compressor,)
            else:
                kwargs["compressor"] = compressor
        try:
            create_array(name, **kwargs)
            return
        except TypeError:
            kwargs.pop("compressor", None)
            kwargs.pop("compressors", None)
            create_array(name, **kwargs)
            return

    legacy_kwargs: Dict[str, Any] = {"data": array, "overwrite": True}
    if chunks is not None:
        legacy_kwargs["chunks"] = chunks
    if compressor is not None and array.dtype.kind not in {"U", "S", "O"}:
        legacy_kwargs["compressor"] = compressor
    try:
        group.create_dataset(name, **legacy_kwargs)
    except TypeError:
        legacy_kwargs.pop("compressor", None)
        group.create_dataset(name, **legacy_kwargs)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.nansum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(np.nansum(values * weights) / weight_sum)


def safe_parameter_value(bundle: phoebe.Bundle, qualifier: str) -> float:
    value = bundle.get_parameter(qualifier).value
    if hasattr(value, "to_value"):
        value = value.to_value()
    return float(np.asarray(value))


def get_long_an_for_spice(
    bundle: phoebe.Bundle, use_legacy_long_an_conversion: bool
) -> float:
    long_an_deg = safe_parameter_value(bundle, "long_an@binary@component")
    if use_legacy_long_an_conversion:
        return long_an_deg * DAY_TO_YEAR
    return long_an_deg * DEG_TO_RAD


def build_case_grid(args: argparse.Namespace) -> List[Dict[str, float]]:
    cases: List[Dict[str, float]] = []
    for case_index, values in enumerate(
        product(
            args.inclinations,
            args.periods,
            args.qs,
            args.eccs,
            args.primary_masses,
        )
    ):
        inclination, period, q, ecc, primary_mass = values
        cases.append(
            {
                "case_index": case_index,
                "inclination_deg": float(inclination),
                "period_days": float(period),
                "q": float(q),
                "ecc": float(ecc),
                "primary_mass_msun": float(primary_mass),
            }
        )
    return cases


def create_reusable_phoebe_bundle(
    initial_case: Mapping[str, float],
    n_times: int,
) -> phoebe.Bundle:
    placeholder_times = np.linspace(0.0, float(initial_case["period_days"]), n_times, dtype=np.float64)

    bundle = phoebe.default_binary()
    bundle.flip_constraint("mass@primary", solve_for="sma")

    bundle.add_dataset(
        "mesh",
        times=placeholder_times,
        columns=_mesh_columns,
        dataset=MESH_DATASET_NAME,
    )
    bundle.add_dataset("orb", compute_times=placeholder_times, dataset=ORBIT_DATASET_NAME)
    bundle.add_dataset(
        "lc",
        compute_times=placeholder_times,
        passband="Bolometric:900-40000",
        dataset=LC_DATASET_NAME,
    )

    bundle.set_value_all("pblum_mode", dataset=LC_DATASET_NAME, value="absolute")
    bundle.set_value_all("gravb_bol", 0.0)
    bundle.set_value("distance", 10 * u.pc)
    bundle.set_value_all("ld_mode", "manual")
    bundle.set_value_all("ld_func", "linear")
    bundle.set_value_all("ld_coeffs", [0.0])
    bundle.set_value_all("ld_mode_bol", "manual")
    bundle.set_value_all("ld_func_bol", "linear")
    bundle.set_value_all("ld_coeffs_bol", [0.0])
    bundle.set_value_all("atm", "blackbody")
    bundle.set_value_all("irrad_method", "none")

    apply_case_parameters(bundle, initial_case)
    return bundle


def apply_case_parameters(bundle: phoebe.Bundle, case: Mapping[str, float]) -> None:
    bundle.set_value("period@binary@component", case["period_days"])
    bundle.set_value("q@binary@component", case["q"])
    bundle.set_value("ecc@binary@component", case["ecc"])
    bundle.set_value("mass@primary@component", case["primary_mass_msun"])
    bundle.set_value_all("incl@binary", case["inclination_deg"])


def update_bundle_times(bundle: phoebe.Bundle, times_days: np.ndarray) -> None:
    bundle.set_value(f"times@{MESH_DATASET_NAME}", times_days)
    bundle.set_value(f"compute_times@{ORBIT_DATASET_NAME}", times_days)
    bundle.set_value(f"compute_times@{LC_DATASET_NAME}", times_days)


def compute_case_times(
    bundle: phoebe.Bundle,
    n_times: int,
    use_legacy_long_an_conversion: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    primary_mass = safe_parameter_value(bundle, "mass@primary@component")
    secondary_mass = safe_parameter_value(bundle, "mass@secondary@component")
    period_years = safe_parameter_value(bundle, "period@binary@component") * DAY_TO_YEAR
    ecc = safe_parameter_value(bundle, "ecc@binary@component")
    t0_perpass_years = safe_parameter_value(bundle, "t0_perpass@binary@component") * DAY_TO_YEAR
    inclination_rad = safe_parameter_value(bundle, "incl@binary@component") * DEG_TO_RAD
    omega_rad = safe_parameter_value(bundle, "per0@binary@component") * DEG_TO_RAD
    long_an = get_long_an_for_spice(bundle, use_legacy_long_an_conversion)
    primary_radius = safe_parameter_value(bundle, "requiv@primary@component")
    secondary_radius = safe_parameter_value(bundle, "requiv@secondary@component")

    _, t1_p, _, _, t4_p, _, t1_s, _, _, t4_s = eclipse_timestamps_kepler(
        primary_mass,
        secondary_mass,
        period_years,
        ecc,
        t0_perpass_years,
        inclination_rad,
        omega_rad,
        long_an,
        primary_radius,
        secondary_radius,
        pad=1.1,
        los_vector=LOS_VECTOR,
    )

    eclipse_years = {
        "primary_t1_years": float(np.asarray(t1_p)),
        "primary_t4_years": float(np.asarray(t4_p)),
        "secondary_t1_years": float(np.asarray(t1_s)),
        "secondary_t4_years": float(np.asarray(t4_s)),
    }
    if not np.all(np.isfinite(list(eclipse_years.values()))):
        raise RuntimeError(f"Non-finite eclipse timestamps: {eclipse_years}")
    if eclipse_years["primary_t4_years"] < eclipse_years["primary_t1_years"]:
        raise RuntimeError(f"Primary eclipse window is inverted: {eclipse_years}")
    if eclipse_years["secondary_t4_years"] < eclipse_years["secondary_t1_years"]:
        raise RuntimeError(f"Secondary eclipse window is inverted: {eclipse_years}")

    n_primary = n_times // 2
    n_secondary = n_times - n_primary
    if n_primary <= 0 or n_secondary <= 0:
        raise ValueError("n_times must be at least 2 so both eclipse windows are sampled.")

    primary_times_days = np.linspace(
        eclipse_years["primary_t1_years"] / DAY_TO_YEAR,
        eclipse_years["primary_t4_years"] / DAY_TO_YEAR,
        n_primary,
        dtype=np.float64,
    )
    secondary_times_days = np.linspace(
        eclipse_years["secondary_t1_years"] / DAY_TO_YEAR,
        eclipse_years["secondary_t4_years"] / DAY_TO_YEAR,
        n_secondary,
        dtype=np.float64,
    )
    times_days = np.concatenate([primary_times_days, secondary_times_days])
    times_years = times_days * DAY_TO_YEAR

    eclipse_days = {
        key.replace("_years", "_days"): value / DAY_TO_YEAR for key, value in eclipse_years.items()
    }
    metadata = {**eclipse_years, **eclipse_days}
    metadata["n_primary_times"] = n_primary
    metadata["n_secondary_times"] = n_secondary
    LOGGER.debug("Computed eclipse metadata: %s", metadata)
    return times_days, times_years, metadata


def build_default_icosphere(bb: Blackbody, mass: float, mesh_points: int):
    return get_mesh_view(
        IcosphereModel.construct(
            mesh_points,
            1.0,
            mass,
            bb.solar_parameters,
            bb.parameter_names,
        ),
        LOS_VECTOR,
    )


def build_spice_binary(
    bundle: phoebe.Bundle,
    bb: Blackbody,
    times_years: np.ndarray,
    mesh_points: int,
    use_legacy_long_an_conversion: bool,
) -> Tuple[Binary, Sequence[Any], Sequence[Any]]:
    body1 = build_default_icosphere(
        bb, safe_parameter_value(bundle, "mass@primary@component"), mesh_points
    )
    body2 = build_default_icosphere(
        bb, safe_parameter_value(bundle, "mass@secondary@component"), mesh_points
    )
    binary = Binary.from_bodies(body1, body2)
    binary = add_orbit(
        binary,
        P=safe_parameter_value(bundle, "period@binary@component") * DAY_TO_YEAR,
        ecc=safe_parameter_value(bundle, "ecc@binary@component"),
        T=safe_parameter_value(bundle, "t0_perpass@binary@component") * DAY_TO_YEAR,
        i=safe_parameter_value(bundle, "incl@binary@component") * DEG_TO_RAD,
        omega=safe_parameter_value(bundle, "per0@binary@component") * DEG_TO_RAD,
        Omega=get_long_an_for_spice(bundle, use_legacy_long_an_conversion),
        vgamma=safe_parameter_value(bundle, "vgamma"),
        reference_time=safe_parameter_value(bundle, "t0_ref@binary@component") * DAY_TO_YEAR,
        mean_anomaly=safe_parameter_value(bundle, "mean_anom@binary@component") * DEG_TO_RAD,
        orbit_resolution_points=len(times_years),
    )
    primary_models, secondary_models = evaluate_orbit_at_times(binary, times_years)
    return binary, primary_models, secondary_models


def compute_spice_lightcurve_products(
    primary_models: Sequence[Any],
    secondary_models: Sequence[Any],
    bb: Blackbody,
    wavelengths_angstrom: jnp.ndarray,
) -> Dict[str, np.ndarray]:
    bolometric_filter = Bolometric()
    log_wavelengths = jnp.log10(wavelengths_angstrom)
    n_times = len(primary_models)
    n_wavelengths = int(wavelengths_angstrom.shape[0])

    primary_spectra = np.empty((n_times, n_wavelengths), dtype=np.float32)
    secondary_spectra = np.empty((n_times, n_wavelengths), dtype=np.float32)
    total_spectra = np.empty((n_times, n_wavelengths), dtype=np.float32)
    bolometric_luminosity = np.empty(n_times, dtype=np.float64)

    for index, (primary_model, secondary_model) in enumerate(
        tqdm(
            zip(primary_models, secondary_models),
            total=n_times,
            desc="SPICE spectra",
            leave=False,
        )
    ):
        primary_flux = np.asarray(
            simulate_observed_flux(
                bb.intensity,
                primary_model,
                log_wavelengths,
                disable_doppler_shift=True,
            )
        )[:, 0]
        secondary_flux = np.asarray(
            simulate_observed_flux(
                bb.intensity,
                secondary_model,
                log_wavelengths,
                disable_doppler_shift=True,
            )
        )[:, 0]
        total_flux = primary_flux + secondary_flux

        primary_spectra[index] = primary_flux.astype(np.float32, copy=False)
        secondary_spectra[index] = secondary_flux.astype(np.float32, copy=False)
        total_spectra[index] = total_flux.astype(np.float32, copy=False)
        bolometric_luminosity[index] = float(
            np.asarray(AB_passband_luminosity(bolometric_filter, wavelengths_angstrom, total_flux))
        )

        if (index + 1) % 5 == 0:
            force_garbage_collection()

    return {
        "primary_spectra": primary_spectra,
        "secondary_spectra": secondary_spectra,
        "total_spectra": total_spectra,
        "bolometric_luminosity": bolometric_luminosity,
        "bolometric_luminosity_delta": bolometric_luminosity - bolometric_luminosity[0],
    }


def collect_spice_component_data(models: Sequence[Any]) -> Dict[str, Any]:
    first = models[0]

    static = {
        "radius_rsun": np.asarray(first.radius),
        "mass_msun": np.asarray(first.mass),
        "faces": np.asarray(first.faces),
        "d_vertices_rsun": np.asarray(first.d_vertices),
        "d_centers_rsun": np.asarray(first.d_centers),
        "base_areas_rsun2": np.asarray(first.base_areas),
        "parameters": np.asarray(first.parameters),
        "log_g_index": np.asarray(-1 if first.log_g_index is None else first.log_g_index),
        "rotation_axis": np.asarray(first.rotation_axis),
        "rotation_matrix": np.asarray(first.rotation_matrix),
        "rotation_matrix_prim": np.asarray(first.rotation_matrix_prim),
        "axis_radii_rsun": np.asarray(first.axis_radii),
        "los_vector": np.asarray(first.los_vector),
        "spherical_harmonics_parameters": np.asarray(first.spherical_harmonics_parameters),
        "pulsation_periods": np.asarray(first.pulsation_periods),
        "fourier_series_parameters": np.asarray(first.fourier_series_parameters),
        "pulsation_axes": np.asarray(first.pulsation_axes),
        "pulsation_angles": np.asarray(first.pulsation_angles),
    }

    dynamic = {
        "body_center_rsun": np.stack([np.asarray(model.center) for model in models], axis=0),
        "body_orbital_velocity_kms": np.stack(
            [np.asarray(model.orbital_velocity) for model in models], axis=0
        ),
        "mesh_centers_rsun": np.stack([np.asarray(model.centers) for model in models], axis=0),
        "mesh_velocities_kms": np.stack(
            [np.asarray(model.velocities) for model in models], axis=0
        ),
        "mus": np.stack([np.asarray(model.mus) for model in models], axis=0),
        "los_velocities_kms": np.stack(
            [np.asarray(model.los_velocities) for model in models], axis=0
        ),
        "los_z_rsun": np.stack([np.asarray(model.los_z) for model in models], axis=0),
        "cast_centers_rsun": np.stack(
            [np.asarray(model.cast_centers) for model in models], axis=0
        ),
        "cast_areas_rsun2": np.stack([np.asarray(model.cast_areas) for model in models], axis=0),
        "visible_cast_areas_rsun2": np.stack(
            [np.asarray(model.visible_cast_areas) for model in models], axis=0
        ),
        "occluded_areas_rsun2": np.stack(
            [np.asarray(model.occluded_areas) for model in models], axis=0
        ),
    }
    return {"static": static, "dynamic": dynamic}


def collect_phoebe_orbit_and_summary_data(
    bundle: phoebe.Bundle,
    times_days: np.ndarray,
    store_phoebe_mesh: bool,
) -> Dict[str, Any]:
    phoebe_config = PhoebeConfig(bundle, MESH_DATASET_NAME, ORBIT_DATASET_NAME)
    result: Dict[str, Any] = {"orbit": {}, "summary": {}, "mesh": {}}

    for label, component in (("primary", Component.PRIMARY), ("secondary", Component.SECONDARY)):
        orbit_centers = np.asarray(phoebe_config.get_all_orbit_centers(component))
        orbit_velocities = np.asarray(phoebe_config.get_all_orbit_velocities(component))
        result["orbit"][label] = {
            "centers_uvw_rsun": orbit_centers,
            "velocities_uvw_kms": orbit_velocities,
        }

        visible_element_count = np.empty(len(times_days), dtype=np.int32)
        projected_area_sum = np.empty(len(times_days), dtype=np.float64)
        mu_mean = np.empty(len(times_days), dtype=np.float64)
        mu_max = np.empty(len(times_days), dtype=np.float64)
        radial_velocity_mean = np.empty(len(times_days), dtype=np.float64)
        teff_area_weighted_mean = np.empty(len(times_days), dtype=np.float64)
        logg_mean = np.empty(len(times_days), dtype=np.float64)

        projected_centers = []
        center_velocities = []
        mus = []
        projected_areas = []
        teffs = []
        loggs = []

        for time_index, time_day in enumerate(times_days):
            mesh_centers = np.asarray(phoebe_config.get_mesh_projected_centers(time_day, component))
            mesh_center_velocities = np.asarray(
                phoebe_config.get_center_velocities(time_day, component)
            )
            mesh_mus = np.asarray(phoebe_config.get_mus(time_day, component))
            mesh_projected_areas = np.asarray(
                phoebe_config.get_projected_areas(time_day, component)
            )
            mesh_teffs = np.asarray(phoebe_config.get_teffs(time_day, component))
            mesh_loggs = np.asarray(phoebe_config.get_loggs(time_day, component))
            mesh_radial_velocities = np.asarray(
                phoebe_config.get_radial_velocities(time_day, component)
            )

            weights = np.clip(mesh_projected_areas, a_min=0.0, a_max=None)
            visible_element_count[time_index] = int(np.count_nonzero(mesh_mus > 0.0))
            projected_area_sum[time_index] = float(np.nansum(mesh_projected_areas))
            mu_mean[time_index] = float(np.nanmean(mesh_mus))
            mu_max[time_index] = float(np.nanmax(mesh_mus))
            radial_velocity_mean[time_index] = float(np.nanmean(mesh_radial_velocities))
            teff_area_weighted_mean[time_index] = weighted_mean(mesh_teffs, weights)
            logg_mean[time_index] = float(np.nanmean(mesh_loggs))

            if store_phoebe_mesh:
                projected_centers.append(mesh_centers)
                center_velocities.append(mesh_center_velocities)
                mus.append(mesh_mus)
                projected_areas.append(mesh_projected_areas)
                teffs.append(mesh_teffs)
                loggs.append(mesh_loggs)

        result["summary"][label] = {
            "visible_element_count": visible_element_count,
            "projected_area_sum_rsun2": projected_area_sum,
            "mu_mean": mu_mean,
            "mu_max": mu_max,
            "radial_velocity_mean_kms": radial_velocity_mean,
            "teff_area_weighted_mean_k": teff_area_weighted_mean,
            "logg_mean": logg_mean,
        }

        if store_phoebe_mesh:
            result["mesh"][label] = {
                "projected_centers_uvw_rsun": np.stack(projected_centers, axis=0),
                "center_velocities_uvw_kms": np.stack(center_velocities, axis=0),
                "mus": np.stack(mus, axis=0),
                "projected_areas_rsun2": np.stack(projected_areas, axis=0),
                "teffs_k": np.stack(teffs, axis=0),
                "loggs": np.stack(loggs, axis=0),
            }

    return result


def extract_phoebe_scalar_parameters(
    bundle: phoebe.Bundle, use_legacy_long_an_conversion: bool
) -> Dict[str, float]:
    values: Dict[str, Any] = {}
    missing_parameters: List[str] = []
    for name, (qualifier, factor) in PHOEBE_SCALAR_PARAMETER_SPECS.items():
        try:
            values[name] = safe_parameter_value(bundle, qualifier) * factor
        except Exception:
            missing_parameters.append(name)
    values["long_an_rad_for_spice"] = get_long_an_for_spice(
        bundle, use_legacy_long_an_conversion
    )
    values["long_an_conversion_mode"] = (
        "legacy_day_to_year" if use_legacy_long_an_conversion else "degrees_to_radians"
    )
    if missing_parameters:
        values["missing_parameters"] = missing_parameters
    return values


def collect_case_results(
    bundle: phoebe.Bundle,
    bb: Blackbody,
    times_days: np.ndarray,
    times_years: np.ndarray,
    wavelengths_angstrom: jnp.ndarray,
    mesh_points: int,
    ntriangles: int,
    store_phoebe_mesh: bool,
    use_legacy_long_an_conversion: bool,
) -> Dict[str, Any]:
    LOGGER.info("Running PHOEBE compute (ntriangles=%d)", ntriangles)
    log_memory_usage("Before PHOEBE compute")
    try:
        bundle.compute_pblums(pbflux=True, set_value=True)
        bundle.compute_ld_coeffs(ld_mode="manual", ld_func="linear", ld_coeffs=[0.0])
        bundle.run_compute(
            irrad_method="none",
            coordinates="uvw",
            ltte=False,
            ntriangles=ntriangles,
            overwrite=True,
        )
    except Exception as exc:
        LOGGER.exception("PHOEBE run_compute failed: %s", exc)
        raise RuntimeError(f"phoebe_run_compute failed: {exc}") from exc
    log_memory_usage("After PHOEBE compute")
    force_garbage_collection()

    LOGGER.info("Collecting PHOEBE lightcurve, orbit, and summary outputs")
    try:
        phoebe_fluxes = np.asarray(bundle.get_parameter(f"fluxes@{LC_DATASET_NAME}@model").value)
        phoebe_results = collect_phoebe_orbit_and_summary_data(
            bundle, times_days, store_phoebe_mesh=store_phoebe_mesh
        )
    except Exception as exc:
        LOGGER.exception("PHOEBE output collection failed: %s", exc)
        raise RuntimeError(f"phoebe_output_collection failed: {exc}") from exc

    LOGGER.info("Building SPICE binary and evaluating orbit")
    binary, spice_primary_models, spice_secondary_models = build_spice_binary(
        bundle,
        bb,
        times_years,
        mesh_points=mesh_points,
        use_legacy_long_an_conversion=use_legacy_long_an_conversion,
    )
    log_memory_usage("After SPICE orbit evaluation")
    force_garbage_collection()

    LOGGER.info("Computing SPICE spectra and bolometric luminosity")
    spice_lightcurve = compute_spice_lightcurve_products(
        spice_primary_models,
        spice_secondary_models,
        bb,
        wavelengths_angstrom,
    )
    log_memory_usage("After SPICE bolometric products")
    force_garbage_collection()

    comparison = {
        "phoebe_flux_delta": phoebe_fluxes - phoebe_fluxes[0],
        "spice_bolometric_delta": spice_lightcurve["bolometric_luminosity_delta"],
    }
    comparison["delta_residual"] = (
        comparison["spice_bolometric_delta"] - comparison["phoebe_flux_delta"]
    )
    LOGGER.info(
        "Computed comparison residuals: max_abs=%.6e mean_abs=%.6e",
        float(np.nanmax(np.abs(comparison["delta_residual"]))),
        float(np.nanmean(np.abs(comparison["delta_residual"]))),
    )

    return {
        "phoebe": {
            "fluxes_bolometric": phoebe_fluxes,
            "fluxes_bolometric_delta": phoebe_fluxes - phoebe_fluxes[0],
            "orbit": phoebe_results["orbit"],
            "summary": phoebe_results["summary"],
            "mesh": phoebe_results["mesh"],
        },
        "spice": {
            "binary_orbit": {
                "evaluated_times_years": np.asarray(binary.evaluated_times),
                "body1_centers_rsun": np.asarray(binary.body1_centers),
                "body2_centers_rsun": np.asarray(binary.body2_centers),
                "body1_velocities_kms": np.asarray(binary.body1_velocities),
                "body2_velocities_kms": np.asarray(binary.body2_velocities),
                "n_neighbours1": np.asarray(binary.n_neighbours1),
                "n_neighbours2": np.asarray(binary.n_neighbours2),
            },
            "primary_component": collect_spice_component_data(spice_primary_models),
            "secondary_component": collect_spice_component_data(spice_secondary_models),
            "lightcurve": spice_lightcurve,
        },
        "comparison": comparison,
    }


def write_nested_arrays(group: "zarr.Group", data: Mapping[str, Any], compressor) -> None:
    for key, value in data.items():
        if isinstance(value, Mapping):
            subgroup = group.require_group(key)
            write_nested_arrays(subgroup, value, compressor)
        else:
            write_array(group, key, value, compressor=compressor)


def initialise_store(
    output_path: Path,
    args: argparse.Namespace,
    cases: Sequence[Mapping[str, float]],
    wavelengths_angstrom: np.ndarray,
    bb: Blackbody,
):
    require_zarr()
    mode = "w" if args.overwrite else "a"
    root = zarr.open_group(str(output_path), mode=mode)
    compressor = default_compressor()

    if not args.overwrite and "format_name" in root.attrs:
        existing_grid_shape = list(root.attrs.get("grid_shape", []))
        requested_grid_shape = [
            len(args.inclinations),
            len(args.periods),
            len(args.qs),
            len(args.eccs),
            len(args.primary_masses),
        ]
        if root.attrs.get("format_name") != "spice_phoebe_eclipses_grid":
            raise ValueError(
                f"Existing store at {output_path} is not a spice_phoebe_eclipses_grid store. "
                "Use --overwrite or choose a new --output-path."
            )
        if existing_grid_shape and existing_grid_shape != requested_grid_shape:
            raise ValueError(
                f"Existing store grid_shape={existing_grid_shape} does not match the requested "
                f"grid_shape={requested_grid_shape}. Use --overwrite or choose a new --output-path."
            )
        if int(root.attrs.get("n_times", args.n_times)) != args.n_times:
            raise ValueError(
                f"Existing store n_times={root.attrs.get('n_times')} does not match "
                f"requested n_times={args.n_times}. Use --overwrite or choose a new --output-path."
            )
        if int(root.attrs.get("mesh_points", args.mesh_points)) != args.mesh_points:
            raise ValueError(
                f"Existing store mesh_points={root.attrs.get('mesh_points')} does not match "
                f"requested mesh_points={args.mesh_points}. Use --overwrite or choose a new --output-path."
            )
        if int(root.attrs.get("ntriangles", args.ntriangles)) != args.ntriangles:
            raise ValueError(
                f"Existing store ntriangles={root.attrs.get('ntriangles')} does not match "
                f"requested ntriangles={args.ntriangles}. Use --overwrite or choose a new --output-path."
            )
        if bool(root.attrs.get("store_phoebe_mesh", args.store_phoebe_mesh)) != bool(
            args.store_phoebe_mesh
        ):
            raise ValueError(
                "Existing store store_phoebe_mesh setting does not match the requested one. "
                "Use --overwrite or choose a new --output-path."
            )

    root.attrs["format_name"] = "spice_phoebe_eclipses_grid"
    root.attrs["format_version"] = 1
    root.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    root.attrs["source_script"] = "tutorial/paper_results/spice_phoebe_comparision/check_phoebe_eclipses_grid.py"
    root.attrs["reference_script"] = "tutorial/paper_results/spice_phoebe_comparision/check_phoebe_eclipses.py"
    root.attrs["command"] = " ".join(sys.argv)
    root.attrs["grid_parameter_order"] = PARAMETER_ORDER
    root.attrs["grid_shape"] = [
        len(args.inclinations),
        len(args.periods),
        len(args.qs),
        len(args.eccs),
        len(args.primary_masses),
    ]
    root.attrs["n_cases"] = len(cases)
    root.attrs["n_times"] = args.n_times
    root.attrs["mesh_points"] = args.mesh_points
    root.attrs["ntriangles"] = args.ntriangles
    root.attrs["store_phoebe_mesh"] = args.store_phoebe_mesh
    root.attrs["use_legacy_long_an_conversion"] = args.use_legacy_long_an_conversion
    root.attrs["notes"] = (
        "The store keeps rich per-case outputs in /cases/case_xxxxxx. "
        "PHOEBE full mesh dumps are opt-in because they can be large."
    )
    root.attrs["software_versions"] = to_attr_value(
        {
            "jax": getattr(jax, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "phoebe": getattr(phoebe, "__version__", None),
            "zarr": getattr(zarr, "__version__", None) if zarr is not None else None,
            "numcodecs": safe_distribution_version("numcodecs"),
            "stellar-spice": safe_distribution_version("stellar-spice"),
            "transformer-payne": safe_distribution_version("transformer-payne"),
        }
    )

    common_group = root.require_group("common")
    common_group.attrs["blackbody_parameter_names"] = list(bb.parameter_names)
    common_group.attrs["mesh_dataset_name"] = MESH_DATASET_NAME
    common_group.attrs["orbit_dataset_name"] = ORBIT_DATASET_NAME
    common_group.attrs["lc_dataset_name"] = LC_DATASET_NAME
    common_group.attrs["units"] = to_attr_value(
        {
            "times_days": "days",
            "times_years": "years",
            "wavelengths_angstrom": "angstrom",
            "centers_rsun": "solar_radii",
            "velocities_kms": "km/s",
            "areas_rsun2": "solar_radii^2",
            "masses_msun": "solar_masses",
            "teffs_k": "K",
        }
    )
    write_array(common_group, "wavelengths_angstrom", wavelengths_angstrom, compressor=compressor)
    write_array(common_group, "los_vector", np.asarray(LOS_VECTOR), compressor=compressor)
    write_array(
        common_group,
        "blackbody_solar_parameters",
        np.asarray(bb.solar_parameters),
        compressor=compressor,
    )

    axes_group = root.require_group("axes")
    write_array(
        axes_group,
        "inclination_deg",
        np.asarray(args.inclinations, dtype=np.float64),
        compressor=compressor,
    )
    write_array(
        axes_group,
        "period_days",
        np.asarray(args.periods, dtype=np.float64),
        compressor=compressor,
    )
    write_array(axes_group, "q", np.asarray(args.qs, dtype=np.float64), compressor=compressor)
    write_array(
        axes_group, "ecc", np.asarray(args.eccs, dtype=np.float64), compressor=compressor
    )
    write_array(
        axes_group,
        "primary_mass_msun",
        np.asarray(args.primary_masses, dtype=np.float64),
        compressor=compressor,
    )

    manifest_group = root.require_group("manifest")
    write_array(
        manifest_group,
        "parameter_grid",
        np.asarray([[case[name] for name in PARAMETER_ORDER] for case in cases], dtype=np.float64),
        compressor=compressor,
    )
    write_array(
        manifest_group,
        "case_indices",
        np.asarray([case["case_index"] for case in cases], dtype=np.int64),
        compressor=compressor,
    )
    return root, compressor


def write_case_to_store(
    root: "zarr.Group",
    compressor,
    case: Mapping[str, float],
    times_days: np.ndarray,
    times_years: np.ndarray,
    eclipse_metadata: Mapping[str, float],
    phoebe_scalars: Mapping[str, Any],
    results: Mapping[str, Any],
    elapsed_seconds: float,
    peak_memory_mb: float,
) -> None:
    cases_group = root.require_group("cases")
    case_id = f"case_{int(case['case_index']):06d}"
    case_group = cases_group.require_group(case_id)

    case_group.attrs["status"] = "success"
    case_group.attrs["case_index"] = int(case["case_index"])
    case_group.attrs["elapsed_seconds"] = float(elapsed_seconds)
    case_group.attrs["peak_memory_mb"] = float(peak_memory_mb)
    for name in PARAMETER_ORDER:
        case_group.attrs[name] = float(case[name])
    for key, value in eclipse_metadata.items():
        case_group.attrs[key] = to_attr_value(value)
    for key, value in phoebe_scalars.items():
        case_group.attrs[key] = to_attr_value(value)

    time_group = case_group.require_group("time")
    write_array(time_group, "times_days", times_days, compressor=compressor)
    write_array(time_group, "times_years", times_years, compressor=compressor)

    write_nested_arrays(case_group.require_group("phoebe"), results["phoebe"], compressor)
    write_nested_arrays(case_group.require_group("spice"), results["spice"], compressor)
    write_nested_arrays(case_group.require_group("comparison"), results["comparison"], compressor)

    comparison_group = case_group["comparison"]
    comparison_group.attrs["max_abs_delta_residual"] = float(
        np.nanmax(np.abs(np.asarray(results["comparison"]["delta_residual"])))
    )
    comparison_group.attrs["mean_abs_delta_residual"] = float(
        np.nanmean(np.abs(np.asarray(results["comparison"]["delta_residual"])))
    )


def mark_case_failed(
    root: "zarr.Group",
    case: Mapping[str, float],
    elapsed_seconds: float,
    stage_name: str,
    error_message: str,
    traceback_text: str,
) -> None:
    cases_group = root.require_group("cases")
    case_id = f"case_{int(case['case_index']):06d}"
    case_group = cases_group.require_group(case_id)
    case_group.attrs["status"] = "failed"
    case_group.attrs["case_index"] = int(case["case_index"])
    case_group.attrs["elapsed_seconds"] = float(elapsed_seconds)
    case_group.attrs["failed_stage"] = stage_name
    case_group.attrs["error_message"] = error_message
    case_group.attrs["traceback"] = traceback_text
    for name in PARAMETER_ORDER:
        case_group.attrs[name] = float(case[name])


def case_completed(root: "zarr.Group", case_index: int) -> bool:
    case_id = f"case_{case_index:06d}"
    try:
        cases_group = root["cases"]
        case_group = cases_group[case_id]
    except KeyError:
        return False
    return case_group.attrs.get("status") == "success"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PHOEBE/SPICE eclipse comparisons over a parameter grid while "
            "reusing a single PHOEBE bundle and writing results to a Zarr store."
        )
    )
    parser.add_argument(
        "--inclinations",
        nargs="+",
        type=float,
        default=[90.0],
        help="Inclination values in degrees.",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        type=float,
        default=[1.0],
        help="Orbital period values in days.",
    )
    parser.add_argument(
        "--qs",
        nargs="+",
        type=float,
        default=[1.0],
        help="Mass-ratio values.",
    )
    parser.add_argument(
        "--eccs",
        nargs="+",
        type=float,
        default=[0.0],
        help="Eccentricity values.",
    )
    parser.add_argument(
        "--primary-masses",
        nargs="+",
        type=float,
        default=[1.0],
        help="Primary-mass values in solar masses.",
    )
    parser.add_argument(
        "--n-times",
        type=int,
        default=10,
        help="Number of time samples per case. Half are placed in each eclipse window.",
    )
    parser.add_argument(
        "--mesh-points",
        type=int,
        default=1000,
        help="Number of mesh vertices for each SPICE body.",
    )
    parser.add_argument(
        "--ntriangles",
        type=int,
        default=20000,
        help="Number of PHOEBE triangles used in run_compute.",
    )
    parser.add_argument(
        "--wavelength-min",
        type=float,
        default=900.0,
        help="Minimum wavelength in angstrom.",
    )
    parser.add_argument(
        "--wavelength-max",
        type=float,
        default=40000.0,
        help="Maximum wavelength in angstrom.",
    )
    parser.add_argument(
        "--n-wavelengths",
        type=int,
        default=1000,
        help="Number of wavelength samples for the SPICE bolometric calculation.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="lc_eclipse_grid.zarr",
        help="Path to the output Zarr store.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing Zarr store.",
    )
    parser.add_argument(
        "--store-phoebe-mesh",
        action="store_true",
        help="Store per-element PHOEBE mesh arrays for every time and component.",
    )
    parser.add_argument(
        "--use-legacy-long-an-conversion",
        action="store_true",
        help=(
            "Preserve the original script's long_an*DAY_TO_YEAR conversion for the "
            "SPICE orbit path. By default this script uses radians, which matches "
            "the SPICE orbit utilities."
        ),
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to a log file. Defaults to <output-path>.log.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console and file log level.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_times < 2:
        raise ValueError("--n-times must be at least 2.")
    if args.mesh_points <= 0:
        raise ValueError("--mesh-points must be positive.")
    if args.ntriangles <= 0:
        raise ValueError("--ntriangles must be positive.")
    if args.n_wavelengths <= 1:
        raise ValueError("--n-wavelengths must be greater than 1.")
    if args.wavelength_max <= args.wavelength_min:
        raise ValueError("--wavelength-max must be greater than --wavelength-min.")
    if any(value <= 0.0 for value in args.periods):
        raise ValueError("All period values must be positive.")
    if any(value <= 0.0 for value in args.primary_masses):
        raise ValueError("All primary-mass values must be positive.")
    if any(value <= 0.0 for value in args.qs):
        raise ValueError("All q values must be positive.")
    if any(value < 0.0 or value >= 1.0 for value in args.eccs):
        raise ValueError("All eccentricity values must satisfy 0 <= ecc < 1.")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    log_path = (
        Path(args.log_path).expanduser().resolve()
        if args.log_path is not None
        else default_log_path_for_output(output_path)
    )
    failure_dir = default_failure_dir_for_output(output_path)
    setup_logging(log_path=log_path, log_level=args.log_level, overwrite=args.overwrite)
    LOGGER.info("Command: %s", " ".join(sys.argv))
    LOGGER.info("Output path: %s", output_path)
    LOGGER.info("Failure dump directory: %s", failure_dir)

    validate_args(args)
    require_zarr()

    cases = build_case_grid(args)
    if len(cases) == 0:
        raise RuntimeError("The parameter grid is empty.")

    LOGGER.info("Preparing %d parameter combinations", len(cases))
    start_time = time.time()
    log_memory_usage("Script start")

    bb = Blackbody()
    wavelengths_angstrom = jnp.linspace(
        args.wavelength_min, args.wavelength_max, args.n_wavelengths
    )
    LOGGER.info(
        "Wavelength grid configured: min=%.1f A, max=%.1f A, n=%d",
        args.wavelength_min,
        args.wavelength_max,
        args.n_wavelengths,
    )

    root, compressor = initialise_store(
        output_path=output_path,
        args=args,
        cases=cases,
        wavelengths_angstrom=np.asarray(wavelengths_angstrom),
        bb=bb,
    )
    LOGGER.info("Zarr store initialised at %s", output_path)
    bundle = create_reusable_phoebe_bundle(cases[0], n_times=args.n_times)
    LOGGER.info("Reusable PHOEBE bundle created from first case template")

    completed = 0
    skipped = 0
    failed = 0

    for case in tqdm(cases, desc="Grid cases"):
        label = case_label(case)
        if (not args.overwrite) and case_completed(root, int(case["case_index"])):
            skipped += 1
            LOGGER.info("Skipping already completed %s", label)
            continue

        LOGGER.info("Starting %s", label)
        case_start = time.time()
        peak_memory_mb = get_memory_usage()
        current_stage = "initialise_case"
        times_days: np.ndarray | None = None
        eclipse_metadata: Dict[str, Any] | None = None
        phoebe_scalars: Dict[str, Any] | None = None

        try:
            current_stage = "apply_case_parameters"
            LOGGER.info("%s | applying PHOEBE parameters", label)
            apply_case_parameters(bundle, case)

            current_stage = "compute_case_times"
            LOGGER.info("%s | computing eclipse timestamps and time grid", label)
            times_days, times_years, eclipse_metadata = compute_case_times(
                bundle,
                args.n_times,
                use_legacy_long_an_conversion=args.use_legacy_long_an_conversion,
            )
            LOGGER.info(
                "%s | prepared %d times: primary %.6f-%.6f d, secondary %.6f-%.6f d",
                label,
                len(times_days),
                eclipse_metadata["primary_t1_days"],
                eclipse_metadata["primary_t4_days"],
                eclipse_metadata["secondary_t1_days"],
                eclipse_metadata["secondary_t4_days"],
            )

            current_stage = "update_bundle_times"
            LOGGER.info("%s | updating PHOEBE dataset time arrays", label)
            update_bundle_times(bundle, times_days)

            current_stage = "extract_phoebe_scalar_parameters"
            LOGGER.info("%s | extracting PHOEBE scalar parameters", label)
            phoebe_scalars = extract_phoebe_scalar_parameters(
                bundle,
                use_legacy_long_an_conversion=args.use_legacy_long_an_conversion,
            )

            current_stage = "collect_case_results"
            LOGGER.info("%s | running PHOEBE and SPICE computations", label)
            results = collect_case_results(
                bundle=bundle,
                bb=bb,
                times_days=times_days,
                times_years=times_years,
                wavelengths_angstrom=wavelengths_angstrom,
                mesh_points=args.mesh_points,
                ntriangles=args.ntriangles,
                store_phoebe_mesh=args.store_phoebe_mesh,
                use_legacy_long_an_conversion=args.use_legacy_long_an_conversion,
            )
            peak_memory_mb = max(peak_memory_mb, get_memory_usage())

            current_stage = "write_case_to_store"
            LOGGER.info("%s | writing case results to Zarr store", label)
            write_case_to_store(
                root=root,
                compressor=compressor,
                case=case,
                times_days=times_days,
                times_years=times_years,
                eclipse_metadata=eclipse_metadata,
                phoebe_scalars=phoebe_scalars,
                results=results,
                elapsed_seconds=time.time() - case_start,
                peak_memory_mb=peak_memory_mb,
            )
            completed += 1
            LOGGER.info(
                "%s | completed in %.2fs (peak_memory=%.1f MB)",
                label,
                time.time() - case_start,
                peak_memory_mb,
            )
        except Exception as exc:
            failed += 1
            traceback_text = traceback.format_exc()
            mark_case_failed(
                root=root,
                case=case,
                elapsed_seconds=time.time() - case_start,
                stage_name=current_stage,
                error_message=f"{current_stage}: {exc}",
                traceback_text=traceback_text,
            )
            failure_path = dump_failed_case_file(
                failure_dir=failure_dir,
                case=case,
                stage_name=current_stage,
                error_message=f"{current_stage}: {exc}",
                traceback_text=traceback_text,
                bundle=bundle,
                use_legacy_long_an_conversion=args.use_legacy_long_an_conversion,
                times_days=times_days,
                eclipse_metadata=eclipse_metadata,
                phoebe_scalars=phoebe_scalars,
            )
            LOGGER.error("%s | wrote failure dump to %s", label, failure_path)
            LOGGER.exception("%s | failed during %s: %s", label, current_stage, exc)
        finally:
            force_garbage_collection()
            LOGGER.debug("%s | garbage collection completed", label)

    root.attrs["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    root.attrs["completed_cases"] = int(completed)
    root.attrs["skipped_cases"] = int(skipped)
    root.attrs["failed_cases"] = int(failed)
    root.attrs["total_elapsed_seconds"] = float(time.time() - start_time)

    LOGGER.info(
        "Finished grid sweep: completed=%d, skipped=%d, failed=%d, output=%s",
        completed,
        skipped,
        failed,
        output_path,
    )
    log_memory_usage("Script completion", start_time)


if __name__ == "__main__":
    main()
