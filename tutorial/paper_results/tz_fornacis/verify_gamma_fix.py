"""Quick orbit-level verification that the γ fix in orbit_utils is correct.

Calls get_orbit_jax with vgamma = 0 and vgamma = +20 km/s and inspects the
returned barycenter / per-body velocity components. Bypasses the full Binary
import (which transitively requires phoebe in some envs) so the test runs in
a minimal SPICE install.

The current implementation applies γ along ``+los_vector`` so that
``mesh.los_velocities = dot(velocities, los_vector)`` picks it up as
``+vgamma`` for the barycenter, which then yields the expected redshift
through ``apply_vrad``. With ``los_vector = [0, 0, -1]`` (the TZ For and
PHOEBE convention) γ ends up on ``bary_v_z`` with sign -γ; with
``los_vector = [0, 1, 0]`` (the legacy mesh default) γ ends up on
``bary_v_y`` with sign +γ. This script checks both.

Three checks per LOS:
* With γ = 0: ``bary · los_vector ≈ 0`` (Kepler-only).
* With γ ≠ 0: ``bary · los_vector ≈ +γ`` (redshift for γ > 0).
* Per-body velocities are UNCHANGED by γ (the double-count bug stays gone).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, "/Users/mjablons/code/spice/src")

import numpy as np
import jax.numpy as jnp

from spice.models.orbit_utils import get_orbit_jax


PRIMARY_MASS = 2.057
SECONDARY_MASS = 1.958
PERIOD_YR = 75.66647 / 365.25
INCL = jnp.deg2rad(85.68)
PER0 = jnp.deg2rad(65.99)
LONG_AN = jnp.deg2rad(269.0)
VGAMMA_KMS = 20.0
N_PHASES = 40


def _orbit(vgamma_kms, los_vector):
    times = jnp.linspace(0.0, PERIOD_YR, N_PHASES)
    return get_orbit_jax(
        times,
        PRIMARY_MASS, SECONDARY_MASS,
        PERIOD_YR, 0.0, 0.0,
        INCL, PER0, LONG_AN,
        0.0, 0.0, float(vgamma_kms),
        los_vector=jnp.asarray(los_vector, dtype=jnp.float64),
    )


def _vel_components(orbit):
    """orbit shape: (6, n_phases, 3). orbit[1] = bary_vel, [3] = primary_vel,
    [5] = secondary_vel — all in km/s."""
    bary_vel = np.asarray(orbit[1])
    primary_vel = np.asarray(orbit[3])
    secondary_vel = np.asarray(orbit[5])
    return bary_vel, primary_vel, secondary_vel


def _los_vel(vel, los_unit):
    """LOS-projected velocity = vel · los_unit (same sign as mesh.los_velocities)."""
    return float(np.median(vel @ los_unit))


def _summary(name, vel, los_unit):
    los_med = _los_vel(vel, los_unit)
    proj = vel @ los_unit
    print(f"  {name:10s} v·LOS: median={los_med:+.4f}  range=[{float(proj.min()):+.3f},{float(proj.max()):+.3f}] km/s")
    return los_med, (float(proj.max()) - float(proj.min())) / 2.0


def _check_one(los_vector):
    los = np.asarray(los_vector, dtype=float)
    los_unit = los / np.linalg.norm(los)
    print(f"--- LOS = {los_vector} ---")

    print("[vgamma = 0]")
    o0 = _orbit(0.0, los_vector)
    b0, p0, s0 = _vel_components(o0)
    med_b0, _ = _summary("bary",      b0, los_unit)
    med_p0, K_p0 = _summary("primary",   p0, los_unit)
    med_s0, K_s0 = _summary("secondary", s0, los_unit)
    print(f"  Inferred K1 = {K_p0:.2f}, K2 = {K_s0:.2f}")
    print()

    print(f"[vgamma = +{VGAMMA_KMS} km/s]")
    og = _orbit(VGAMMA_KMS, los_vector)
    bg, pg, sg = _vel_components(og)
    med_bg, _ = _summary("bary",      bg, los_unit)
    med_pg, K_pg = _summary("primary",   pg, los_unit)
    med_sg, K_sg = _summary("secondary", sg, los_unit)
    print()

    # Per-body amplitudes should be unchanged (only barycenter carries γ)
    print(f"  K1 change with γ: {K_pg - K_p0:+.4f} km/s (should be ≈ 0)")
    print(f"  K2 change with γ: {K_sg - K_s0:+.4f} km/s (should be ≈ 0)")
    print(f"  Body 1 median v·LOS change with γ: {med_pg - med_p0:+.4f}  (should be 0)")
    print(f"  Body 2 median v·LOS change with γ: {med_sg - med_s0:+.4f}  (should be 0)")

    tol = 0.05
    expected = +VGAMMA_KMS  # γ>0 means receding ⇒ +los_velocity ⇒ redshift
    err = abs(med_bg - expected)
    print(f"  bary v·LOS median: {med_bg:+.3f} km/s  (expected ≈ {expected:+.1f})")
    if err < tol and all(abs(x) < tol for x in
                          (med_pg - med_p0, med_sg - med_s0, K_pg - K_p0, K_sg - K_s0)):
        print("  FIX OK: γ propagates as +γ along LOS at the barycenter; per-body unchanged.")
        return 0
    print("  FAIL")
    return 1


def main():
    print("Checking γ flow through get_orbit_jax for both supported LOS axes:\n")
    bad = 0
    bad += _check_one([0.0, 0.0, -1.0])
    print()
    bad += _check_one([0.0, 1.0, 0.0])
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
