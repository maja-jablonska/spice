import jax
jax.config.update("jax_enable_x64", True)
import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


LOGGER = logging.getLogger(__name__)
DEFAULT_MESH_SIZES = [1000, 5000, 10000]
DEFAULT_OUTPUT = Path(__file__).with_name("rotation_velocity_sweep_results.txt")
DEFAULT_CSV_OUTPUT = Path(__file__).with_name("rotation_velocity_sweep_results.csv")
_ROTATION_DEPENDENCIES: dict[str, object] | None = None


@dataclass
class SweepResult:
    radius: float
    rotation_velocity: float
    sample_count: int
    max_error: float
    mean_error: float
    rms_error: float
    max_speed_error: float
    max_radius_based_speed_error: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SPICE mesh rotation velocities against the theoretical rigid-"
            "rotation velocities computed from each point's latitude and longitude."
        )
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 5.0],
        help="Stellar radii to test.",
    )
    parser.add_argument(
        "--rotation-velocities",
        nargs="+",
        type=float,
        default=[10.0, 25.0, 50.0, 100.0, 200.0],
        help="Equatorial rotation velocities in km/s to test.",
    )
    parser.add_argument(
        "--mesh-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_MESH_SIZES,
        help="Icosphere mesh sizes passed to IcosphereModel.construct.",
    )
    parser.add_argument(
        "--mesh-size",
        type=int,
        default=None,
        help="Deprecated single mesh size override. If provided, only this mesh size is used.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Stellar mass in solar masses.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="Evaluation time in seconds.",
    )
    parser.add_argument(
        "--use-selection-mask",
        action="store_true",
        help="Use the same subset of points highlighted in rotation.ipynb.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Print the top-k worst cases by max vector error.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Text file to write the sweep tables to.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="CSV file to write the sweep results to.",
    )
    return parser.parse_args(argv)


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_mesh_sizes(args: argparse.Namespace) -> list[int]:
    if args.mesh_size is not None:
        LOGGER.warning("--mesh-size is deprecated; use --mesh-sizes instead.")
        return [args.mesh_size]
    return list(args.mesh_sizes)


def get_rotation_dependencies() -> dict[str, object]:
    global _ROTATION_DEPENDENCIES
    if _ROTATION_DEPENDENCIES is None:

        import jax.numpy as jnp
        from transformer_payne import Blackbody

        from spice.models import IcosphereModel
        from spice.models.mesh_transform import add_rotation, evaluate_rotation
        from spice.models.mesh_view import get_mesh_view

        _ROTATION_DEPENDENCIES = {
            "jnp": jnp,
            "Blackbody": Blackbody,
            "IcosphereModel": IcosphereModel,
            "add_rotation": add_rotation,
            "evaluate_rotation": evaluate_rotation,
            "get_mesh_view": get_mesh_view,
            "los_vector": jnp.array([0.0, 1.0, 0.0]),
        }
    return _ROTATION_DEPENDENCIES


def select_indices(centers: np.ndarray, use_selection_mask: bool) -> np.ndarray:
    if not use_selection_mask:
        return np.arange(len(centers))

    valid_indices = np.where(
        (centers[:, 0] > -0.15)
        & (centers[:, 1] < 0.25)
        & (centers[:, 2] > 0.0)
        & (centers[:, 2] < 0.75)
    )[0]
    return valid_indices[::3]


def build_model(mesh_size: int, radius: float, mass: float, rotation_velocity: float, time: float):
    LOGGER.debug(
        "Building model: mesh_size=%s radius=%s mass=%s rotation_velocity=%s time=%s",
        mesh_size,
        radius,
        mass,
        rotation_velocity,
        time,
    )
    deps = get_rotation_dependencies()
    bb = deps["Blackbody"]()
    model = deps["IcosphereModel"].construct(
        mesh_size,
        radius,
        mass,
        bb.to_parameters(),
        bb.parameter_names,
    )
    model = deps["add_rotation"](model, rotation_velocity)
    rotated = deps["evaluate_rotation"](model, time)
    return deps["get_mesh_view"](rotated, deps["los_vector"])


def calculate_theoretical_velocities(centers: np.ndarray, equatorial_speed: float) -> np.ndarray:
    element_radii = np.linalg.norm(centers, axis=1)
    scale = np.divide(
        equatorial_speed,
        element_radii,
        out=np.zeros_like(element_radii),
        where=element_radii > 0.0,
    )
    return np.column_stack(
        [
            centers[:, 1] * scale,
            -centers[:, 0] * scale,
            np.zeros_like(scale),
        ]
    )


def calculate_radius_based_theoretical_velocities(
    centers: np.ndarray,
    element_radii: np.ndarray,
    equatorial_speed: float,
) -> np.ndarray:
    cylindrical_radius = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
    theoretical_speed = equatorial_speed * cylindrical_radius / element_radii
    scale = np.divide(
        theoretical_speed,
        cylindrical_radius,
        out=np.zeros_like(theoretical_speed),
        where=cylindrical_radius > 0.0,
    )

    return np.column_stack(
        [
            centers[:, 1] * scale,
            -centers[:, 0] * scale,
            np.zeros_like(theoretical_speed),
        ]
    )


def calculate_weighted_error_statistics(
    errors: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    positive_weights = np.clip(weights, a_min=0.0, a_max=None)
    if positive_weights.sum() <= 0.0:
        mean_error = float(np.mean(errors))
        rms_error = float(np.sqrt(np.mean(errors**2)))
        return mean_error, rms_error

    mean_error = float(np.average(errors, weights=positive_weights))
    rms_error = float(np.sqrt(np.average(errors**2, weights=positive_weights)))
    return mean_error, rms_error


def measure_difference(
    mesh_size: int,
    radius: float,
    mass: float,
    rotation_velocity: float,
    time: float,
    use_selection_mask: bool,
) -> SweepResult:
    LOGGER.debug(
        "Measuring difference for radius=%.3f rotation_velocity=%.3f",
        radius,
        rotation_velocity,
    )
    mesh_view = build_model(mesh_size, radius, mass, rotation_velocity, time)
    centers = np.asarray(mesh_view.centers)
    radii = np.asarray(mesh_view.radii)
    areas = np.asarray(mesh_view.areas)
    indices = select_indices(centers, use_selection_mask)

    selected_centers = centers[indices]
    selected_radii = radii[indices]
    selected_areas = areas[indices]
    mesh_velocities = np.asarray(mesh_view.velocities)[indices]
    theoretical_velocities = calculate_theoretical_velocities(
        selected_centers,
        equatorial_speed=float(rotation_velocity),
    )
    radius_based_theoretical_velocities = calculate_radius_based_theoretical_velocities(
        selected_centers,
        element_radii=selected_radii,
        equatorial_speed=float(rotation_velocity),
    )

    velocity_difference = mesh_velocities - theoretical_velocities
    vector_error = np.linalg.norm(velocity_difference, axis=1)
    weighted_mean_error, weighted_rms_error = calculate_weighted_error_statistics(
        vector_error,
        selected_areas,
    )
    speed_error = np.abs(
        np.linalg.norm(mesh_velocities, axis=1)
        - np.linalg.norm(theoretical_velocities, axis=1)
    )
    radius_based_speed_error = np.abs(
        np.linalg.norm(mesh_velocities, axis=1)
        - np.linalg.norm(radius_based_theoretical_velocities, axis=1)
    )

    return SweepResult(
        radius=radius,
        rotation_velocity=rotation_velocity,
        sample_count=len(indices),
        max_error=float(np.max(vector_error)),
        mean_error=weighted_mean_error,
        rms_error=weighted_rms_error,
        max_speed_error=float(np.max(speed_error)),
        max_radius_based_speed_error=float(np.max(radius_based_speed_error)),
    )


def format_results_table(results: list[SweepResult], top_k: int) -> str:
    lines = [
        "radius | rotation_velocity | samples | max_error | mean_error(area-weighted) | rms_error(area-weighted) | max_speed_error | max_radius_based_speed_error"
    ]
    for result in results:
        lines.append(
            f"{result.radius:6.2f} | "
            f"{result.rotation_velocity:17.2f} | "
            f"{result.sample_count:7d} | "
            f"{result.max_error:9.6e} | "
            f"{result.mean_error:10.6e} | "
            f"{result.rms_error:9.6e} | "
            f"{result.max_speed_error:15.6e} | "
            f"{result.max_radius_based_speed_error:26.6e}"
        )

    lines.append("")
    lines.append(f"Top {min(top_k, len(results))} cases by max vector error:")
    for result in sorted(results, key=lambda item: item.max_error, reverse=True)[:top_k]:
        lines.append(
            f"radius={result.radius:.2f}, "
            f"rotation_velocity={result.rotation_velocity:.2f}, "
            f"max_error={result.max_error:.6e}, "
            f"mean_error={result.mean_error:.6e}, "
            f"max_radius_based_speed_error={result.max_radius_based_speed_error:.6e}"
        )
    return "\n".join(lines)


def format_report(results_by_mesh: dict[int, list[SweepResult]], top_k: int) -> str:
    sections = []
    for mesh_size, results in results_by_mesh.items():
        section_lines = [
            f"mesh_size={mesh_size}",
            format_results_table(results, top_k=top_k),
        ]
        sections.append("\n".join(section_lines))
    return "\n\n".join(sections) + "\n"


def write_report(report_text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def flatten_results(results_by_mesh: dict[int, list[SweepResult]]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for mesh_size, results in results_by_mesh.items():
        for result in results:
            rows.append(
                {
                    "mesh_size": mesh_size,
                    "radius": result.radius,
                    "rotation_velocity": result.rotation_velocity,
                    "sample_count": result.sample_count,
                    "max_error": result.max_error,
                    "mean_error": result.mean_error,
                    "rms_error": result.rms_error,
                    "max_speed_error": result.max_speed_error,
                    "max_radius_based_speed_error": result.max_radius_based_speed_error,
                }
            )
    return rows


def write_csv_results(results_by_mesh: dict[int, list[SweepResult]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mesh_size",
        "radius",
        "rotation_velocity",
        "sample_count",
        "max_error",
        "mean_error",
        "rms_error",
        "max_speed_error",
        "max_radius_based_speed_error",
    ]
    rows = flatten_results(results_by_mesh)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    mesh_sizes = resolve_mesh_sizes(args)

    combinations = [
        (mesh_size, radius, rotation_velocity)
        for mesh_size in mesh_sizes
        for radius in args.radii
        for rotation_velocity in args.rotation_velocities
    ]
    LOGGER.info(
        "Starting sweep across %d mesh sizes and %d radius/velocity combinations (%d total cases).",
        len(mesh_sizes),
        len(args.radii) * len(args.rotation_velocities),
        len(combinations),
    )
    results_by_mesh = {mesh_size: [] for mesh_size in mesh_sizes}
    iterator = combinations
    if not args.no_progress:
        iterator = tqdm(combinations, desc="Measuring velocity differences", unit="case")

    for mesh_size, radius, rotation_velocity in iterator:
        LOGGER.info(
            "Running case mesh_size=%s radius=%.3f rotation_velocity=%.3f",
            mesh_size,
            radius,
            rotation_velocity,
        )
        results_by_mesh[mesh_size].append(
            measure_difference(
                mesh_size=mesh_size,
                radius=radius,
                mass=args.mass,
                rotation_velocity=rotation_velocity,
                time=args.time,
                use_selection_mask=args.use_selection_mask,
            )
        )

    report_text = format_report(results_by_mesh, top_k=args.top_k)
    output_path = write_report(report_text, args.output)
    csv_output_path = write_csv_results(results_by_mesh, args.csv_output)
    LOGGER.info(
        "Completed sweep. Writing text summary to %s and CSV results to %s",
        output_path.resolve(),
        csv_output_path.resolve(),
    )
    print(report_text, end="")


if __name__ == "__main__":
    main()
