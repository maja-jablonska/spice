"""Binary construction and public orbit evaluation, including optional PHOEBE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.phoebe_helpers import import_phoebe_or_skip

from spice.constants import G_SI, SOLAR_MASS_KG, SOLAR_RAD_M
from spice.models.binary import (
    Binary,
    PhoebeBinary,
    add_orbit,
    evaluate_orbit,
    evaluate_orbit_at_times,
    evaluate_orbit_at_times_stacked,
)
from tests.test_models.utils import default_icosphere


@pytest.fixture(scope="module")
def binary():
    return Binary.from_bodies(
        default_icosphere(), default_icosphere(radius=0.5, mass=0.5)
    )


@pytest.fixture(scope="module")
def orbit_kwargs():
    # A detached, circular, edge-on binary. Distances are in solar radii;
    # native orbit times and periods are in years.
    return dict(P=10.0 / 365.25, ecc=0.0, T=0.0, i=np.pi / 2,
                omega=0.0, Omega=0.0, mean_anomaly=0.0,
                reference_time=0.0, vgamma=0.0, orbit_resolution_points=257)


@pytest.fixture(scope="module")
def orbit(binary, orbit_kwargs):
    return add_orbit(binary, **orbit_kwargs)


def assert_models_equal(actual, expected):
    assert type(actual) is type(expected)
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual),
                                          jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-10, atol=1e-10)


def test_from_bodies_preserves_models_and_neighbor_overrides(binary):
    assert isinstance(binary, Binary)
    assert binary.body1.radius == 1.0
    assert binary.body2.radius == 0.5
    assert isinstance(binary.n_neighbours1, int) and binary.n_neighbours1 > 0
    assert isinstance(binary.n_neighbours2, int) and binary.n_neighbours2 > 0

    explicit = Binary.from_bodies(binary.body1, binary.body2,
                                 n_neighbours1=40, n_neighbours2=48)
    assert explicit.body1 is binary.body1
    assert explicit.body2 is binary.body2
    assert (explicit.n_neighbours1, explicit.n_neighbours2) == (40, 48)


def test_add_orbit_preserves_original(binary, orbit, orbit_kwargs):
    assert binary.P == 1.0
    np.testing.assert_array_equal(binary.body1_centers, 0.0)
    np.testing.assert_array_equal(binary.body2_centers, 0.0)
    np.testing.assert_array_equal(binary.body1.occluded_areas, 0.0)
    np.testing.assert_array_equal(binary.body2.occluded_areas, 0.0)
    assert_models_equal(orbit.body1, binary.body1)
    assert_models_equal(orbit.body2, binary.body2)
    for name in ("P", "ecc", "T", "i", "omega", "Omega", "mean_anomaly",
                 "reference_time", "vgamma"):
        assert float(getattr(orbit, name)) == pytest.approx(orbit_kwargs[name])
    assert orbit.evaluated_times.shape == (257,)
    np.testing.assert_allclose(orbit.evaluated_times[jnp.array([0, -1])], [0.0, orbit.P])


@pytest.mark.parametrize("phase", [0.0, 0.125, 0.25, 0.5, 0.75])
def test_evaluate_orbit_matches_circular_kepler_solution(orbit, phase):
    primary, secondary = evaluate_orbit(orbit, phase * orbit.P)
    period_seconds = float(orbit.P) * 365.25 * 86400.0
    mass1, mass2 = float(primary.mass), float(secondary.mass)
    separation_m = (G_SI * (mass1 + mass2) * SOLAR_MASS_KG
                    * period_seconds**2 / (4 * np.pi**2))**(1 / 3)
    angle = 2 * np.pi * phase
    direction = np.array([np.cos(angle), 0.0, -np.sin(angle)])
    tangent = np.array([-np.sin(angle), 0.0, -np.cos(angle)])

    # Allow the small difference between the rounded shared G constant and
    # the more precise value in the orbit solver, but catch unit/sign errors.
    for model, fraction in ((primary, mass2 / (mass1 + mass2)),
                            (secondary, -mass1 / (mass1 + mass2))):
        np.testing.assert_allclose(
            model.center, fraction * separation_m / SOLAR_RAD_M * direction,
            rtol=2e-5, atol=1e-10,
        )
        np.testing.assert_allclose(
            model.orbital_velocity,
            fraction * 2 * np.pi * separation_m / period_seconds / 1000 * tangent,
            rtol=2e-5, atol=1e-10,
        )
    np.testing.assert_allclose(mass1 * primary.center + mass2 * secondary.center,
                               0.0, atol=1e-10)
    np.testing.assert_allclose(
        mass1 * primary.orbital_velocity + mass2 * secondary.orbital_velocity,
        0.0, atol=1e-10,
    )


@pytest.mark.parametrize("phase, blocked_fractions", [
    (0.0, (0.0, 0.0)),
    (0.25, (0.25, 0.0)),  # smaller secondary is in front
    (0.5, (0.0, 0.0)),
    (0.75, (0.0, 1.0)),  # secondary is completely behind primary
])
def test_evaluate_orbit_resolves_eclipses(orbit, phase, blocked_fractions):
    models = evaluate_orbit(orbit, phase * orbit.P)
    for original, evaluated, expected in zip(
            (orbit.body1, orbit.body2), models, blocked_fractions):
        initial_area = float(jnp.sum(original.visible_cast_areas))
        blocked = 1.0 - float(jnp.sum(evaluated.visible_cast_areas)) / initial_area
        # These scaled, aligned meshes have the same projected outline, so
        # central transit depth is the squared radius ratio, even at low resolution.
        assert blocked == pytest.approx(expected, abs=1e-6)


def test_evaluate_orbit_is_periodic(orbit):
    reference = evaluate_orbit(orbit, 0.137 * orbit.P)
    for offset in (-1, 1, 2):
        result = evaluate_orbit(orbit, (0.137 + offset) * orbit.P)
        for actual, expected in zip(result, reference):
            assert_models_equal(actual, expected)


@pytest.mark.parametrize("stacked", [False, True], ids=["lists", "stacked"])
def test_evaluate_orbit_at_times_matches_scalar(orbit, stacked):
    # Include a non-grid time, both eclipses, and an out-of-period time.
    times = jnp.array([0.137, 0.25, 0.75, 1.125]) * orbit.P
    evaluate = evaluate_orbit_at_times_stacked if stacked else evaluate_orbit_at_times
    primary, secondary = evaluate(orbit, times)
    if stacked:
        for model in (primary, secondary):
            assert all(leaf.shape[0] == len(times)
                       for leaf in jax.tree_util.tree_leaves(model))
    else:
        assert isinstance(primary, list) and isinstance(secondary, list)
        assert len(primary) == len(secondary) == len(times)
    for index, time in enumerate(times):
        for actual, expected in zip((primary, secondary), evaluate_orbit(orbit, time)):
            model = (jax.tree_util.tree_map(lambda leaf: leaf[index], actual)
                     if stacked else actual[index])
            assert_models_equal(model, expected)


@pytest.fixture(scope="module")
def phoebe_binary():
    # Build a small real bundle instead of loading a missing/version-sensitive
    # pickle. Only these integration tests need the optional PHOEBE extra.
    phoebe = import_phoebe_or_skip()
    from spice.models.phoebe_utils import PhoebeConfig

    phoebe.logger(clevel="ERROR")
    bundle = phoebe.default_binary()
    bundle.set_value("period@binary", 2.0)
    bundle.set_value("incl@binary", 80.0)
    bundle.set_value("q", 0.7)
    bundle.set_value("sma@binary", 10.0)
    bundle.set_value_all("ntriangles", 300)
    times = [0.5, 1.5]
    bundle.add_dataset("orb", times=times, dataset="orb01")
    bundle.add_dataset(
        "mesh", times=times, dataset="mesh01",
        columns=["vus", "vvs", "vws", "us", "vs", "ws", "mus",
                 "visibilities", "areas", "teffs", "loggs"],
    )
    bundle.run_compute(irrad_method="none", ltte=False)
    return PhoebeBinary.construct(PhoebeConfig(bundle), ["teff", "logg", "abun"])


def test_phoebe_binary_construct_imports_orbit_and_parameters(phoebe_binary):
    binary = phoebe_binary
    assert isinstance(binary, PhoebeBinary)
    assert binary.P == pytest.approx(2.0 / 365.25)
    assert binary.i == pytest.approx(np.deg2rad(80.0))
    assert binary.body2.mass / binary.body1.mass == pytest.approx(0.7)
    np.testing.assert_array_equal(binary.evaluated_times, [0.5, 1.5])
    separation = np.linalg.norm(binary.body1_centers - binary.body2_centers, axis=1)
    np.testing.assert_allclose(separation, 10.0, rtol=1e-10)
    for model, centers, velocities in (
            (binary.body1, binary.body1_centers, binary.body1_velocities),
            (binary.body2, binary.body2_centers, binary.body2_velocities)):
        assert centers.shape == velocities.shape == (2, 3)
        np.testing.assert_allclose(model.center, centers[0])
        np.testing.assert_allclose(model.orbital_velocity, velocities[0])
        assert model.parameters.shape == (len(model.mus), 3)
        assert np.isfinite(model.parameters).all()
        assert np.all(model.parameters[:, 0] > 0.0)
        np.testing.assert_allclose(model.parameters[:, 1], model.log_gs)
        np.testing.assert_allclose(model.parameters[:, 2], 0.0)


def test_phoebe_evaluate_orbit_at_times_matches_scalar(phoebe_binary):
    # PHOEBE evaluation uses its dataset times in days, unlike native binaries.
    times = phoebe_binary.evaluated_times
    primary, secondary = evaluate_orbit_at_times(phoebe_binary, times)
    assert len(primary) == len(secondary) == len(times)
    for index, time in enumerate(times):
        for actual, expected, centers, velocities in zip(
                (primary[index], secondary[index]), evaluate_orbit(phoebe_binary, time),
                (phoebe_binary.body1_centers, phoebe_binary.body2_centers),
                (phoebe_binary.body1_velocities, phoebe_binary.body2_velocities)):
            assert_models_equal(actual, expected)
            assert actual.time == time
            np.testing.assert_allclose(actual.center, centers[index])
            np.testing.assert_allclose(actual.orbital_velocity, velocities[index])
    assert not np.allclose(primary[0].center, primary[1].center)


def test_add_orbit_rejects_phoebe_binary(phoebe_binary, orbit_kwargs):
    with pytest.raises(ValueError, match="read-only"):
        add_orbit(phoebe_binary, **orbit_kwargs)


def test_stacked_evaluation_rejects_phoebe_binary(phoebe_binary):
    with pytest.raises(NotImplementedError, match="use evaluate_orbit_at_times instead"):
        evaluate_orbit_at_times_stacked(phoebe_binary, phoebe_binary.evaluated_times)
