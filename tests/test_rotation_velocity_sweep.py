import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tutorial"
    / "paper_results"
    / "paper_plots"
    / "rotation_velocity_sweep.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("rotation_velocity_sweep", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_multiple_mesh_sizes():
    module = load_module()

    args = module.parse_args([])

    assert args.mesh_sizes == [1000, 5000, 10000]
    assert args.mesh_size is None
    assert args.csv_output.name == "rotation_velocity_sweep_results.csv"


def test_resolve_mesh_sizes_prefers_deprecated_single_override():
    module = load_module()

    args = module.parse_args(["--mesh-size", "2000"])

    assert module.resolve_mesh_sizes(args) == [2000]


def test_write_report_includes_one_table_per_mesh_size(tmp_path):
    module = load_module()

    results_by_mesh = {
        1000: [
            module.SweepResult(
                radius=1.0,
                rotation_velocity=10.0,
                sample_count=1280,
                max_error=1.0,
                mean_error=0.5,
                rms_error=0.75,
                max_speed_error=1.0,
                max_radius_based_speed_error=1.0,
            )
        ],
        5000: [
            module.SweepResult(
                radius=2.0,
                rotation_velocity=25.0,
                sample_count=5120,
                max_error=2.0,
                mean_error=1.0,
                rms_error=1.25,
                max_speed_error=2.0,
                max_radius_based_speed_error=2.0,
            )
        ],
    }

    report_text = module.format_report(results_by_mesh, top_k=1)
    output_path = tmp_path / "rotation_velocity_sweep_results.txt"
    module.write_report(report_text, output_path)

    written = output_path.read_text(encoding="utf-8")
    assert written == report_text
    assert "mesh_size=1000" in written
    assert "mesh_size=5000" in written
    assert written.count("radius | rotation_velocity | samples") == 2


def test_write_csv_results_exports_flat_rows(tmp_path):
    module = load_module()

    results_by_mesh = {
        1000: [
            module.SweepResult(
                radius=1.0,
                rotation_velocity=10.0,
                sample_count=1280,
                max_error=1.0,
                mean_error=0.5,
                rms_error=0.75,
                max_speed_error=1.0,
                max_radius_based_speed_error=1.0,
            )
        ],
        5000: [
            module.SweepResult(
                radius=2.0,
                rotation_velocity=25.0,
                sample_count=5120,
                max_error=2.0,
                mean_error=1.0,
                rms_error=1.25,
                max_speed_error=2.0,
                max_radius_based_speed_error=2.0,
            )
        ],
    }

    output_path = tmp_path / "rotation_velocity_sweep_results.csv"
    module.write_csv_results(results_by_mesh, output_path)

    written = output_path.read_text(encoding="utf-8").splitlines()
    assert written[0] == (
        "mesh_size,radius,rotation_velocity,sample_count,max_error,mean_error,"
        "rms_error,max_speed_error,max_radius_based_speed_error"
    )
    assert "1000,1.0,10.0,1280,1.0,0.5,0.75,1.0,1.0" in written
    assert "5000,2.0,25.0,5120,2.0,1.0,1.25,2.0,2.0" in written


def test_calculate_theoretical_velocities_uses_direct_cartesian_rotation():
    module = load_module()

    centers = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        ]
    )

    velocities = module.calculate_theoretical_velocities(centers, equatorial_speed=10.0)

    assert np.allclose(
        velocities,
        np.array(
            [
                [0.0, -10.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, -10.0 / np.sqrt(2.0), 0.0],
            ]
        ),
    )


def test_calculate_weighted_error_statistics_uses_face_area_weights():
    module = load_module()

    mean_error, rms_error = module.calculate_weighted_error_statistics(
        np.array([1.0, 3.0]),
        np.array([1.0, 3.0]),
    )

    assert np.isclose(mean_error, 2.5)
    assert np.isclose(rms_error, np.sqrt(7.0))
