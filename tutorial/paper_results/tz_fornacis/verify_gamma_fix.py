"""Quick orbit-level verification that the γ fix in orbit_utils is correct.

Calls get_orbit_jax with vgamma = 0 and vgamma = +20 km/s and inspects the
returned barycenter / per-body velocity components. Bypasses the full Binary
import (which transitively requires phoebe in some envs) so the test runs in
a minimal SPICE install.

Three checks:
* With γ = 0: bary_vy ≈ 0 (Kepler-only) and per-body vy oscillates around 0.
* With γ ≠ 0: bary_vy ≈ -γ (γ now lives on the barycenter, along -y).
* Per-body velocities are UNCHANGED by γ (the double-count bug is gone).
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


def _orbit(vgamma_kms):
    times = jnp.linspace(0.0, PERIOD_YR, N_PHASES)
    return get_orbit_jax(
        times,
        PRIMARY_MASS, SECONDARY_MASS,
        PERIOD_YR, 0.0, 0.0,
        INCL, PER0, LONG_AN,
        0.0, 0.0, float(vgamma_kms),
    )


def _vel_components(orbit):
    """orbit shape: (6, n_phases, 3). orbit[1] = bary_vel, [3] = primary_vel,
    [5] = secondary_vel — all in km/s."""
    bary_vel = np.asarray(orbit[1])
    primary_vel = np.asarray(orbit[3])
    secondary_vel = np.asarray(orbit[5])
    return bary_vel, primary_vel, secondary_vel


def _summary(name, vel):
    med_y = float(np.median(vel[:, 1]))
    min_y, max_y = float(vel[:, 1].min()), float(vel[:, 1].max())
    print(f"  {name:10s} v.y: median={med_y:+.4f}  range=[{min_y:+.3f},{max_y:+.3f}] km/s")
    return med_y, (max_y - min_y) / 2.0


def main():
    print("Checking γ flow through get_orbit_jax (post-fix):")
    print()

    print("[vgamma = 0]")
    o0 = _orbit(0.0)
    b0, p0, s0 = _vel_components(o0)
    med_b0, _ = _summary("bary", b0)
    med_p0, K_p0 = _summary("primary", p0)
    med_s0, K_s0 = _summary("secondary", s0)
    print(f"  Inferred K1 = {K_p0:.2f}, K2 = {K_s0:.2f} (literature: ~38.7 / ~39.6)")
    print()

    print(f"[vgamma = +{VGAMMA_KMS} km/s]")
    og = _orbit(VGAMMA_KMS)
    bg, pg, sg = _vel_components(og)
    med_bg, _ = _summary("bary", bg)
    med_pg, K_pg = _summary("primary", pg)
    med_sg, K_sg = _summary("secondary", sg)
    print()

    # Per-body amplitudes should be unchanged (only barycenter carries γ)
    print(f"K1 change with γ: {K_pg - K_p0:+.4f} km/s (should be ≈ 0)")
    print(f"K2 change with γ: {K_sg - K_s0:+.4f} km/s (should be ≈ 0)")
    print(f"Body 1 median v.y change with γ: {med_pg - med_p0:+.4f} (should be 0; per-body is γ-free)")
    print(f"Body 2 median v.y change with γ: {med_sg - med_s0:+.4f} (should be 0; per-body is γ-free)")
    print()

    print(f"Expected bary v.y with γ = +{VGAMMA_KMS}: {-VGAMMA_KMS:+.1f} km/s")
    print(f"Observed bary v.y median: {med_bg:+.3f} km/s")
    tol = 0.05
    err = abs(med_bg - (-VGAMMA_KMS))
    if err < tol:
        # And per-body should be untouched
        per_body_ok = (abs(med_pg - med_p0) < tol and abs(med_sg - med_s0) < tol
                       and abs(K_pg - K_p0) < tol and abs(K_sg - K_s0) < tol)
        if per_body_ok:
            print("\nFIX OK: γ lives on the barycenter along -y exactly once; "
                  "per-body Kepler motion is unchanged.")
            print("Downstream _add_orbit will form body1_vel = bary_vel + primary_vel, "
                  f"giving body 1's total y-component median ≈ {med_bg + med_p0:+.3f} km/s.")
            return 0
        else:
            print("\nPER-BODY VELOCITY UNEXPECTEDLY CHANGED with γ — partial fix.")
            return 1
    elif abs(med_bg - (-2 * VGAMMA_KMS)) < tol:
        print("\nSTILL DOUBLE-COUNTED on the barycenter (would have given -2γ).")
        return 1
    elif abs(med_bg) < tol:
        print("\nγ DROPPED — barycenter is at 0 instead of -γ. Fix didn't apply.")
        return 1
    else:
        print(f"\nUNEXPECTED bary v.y deviation: {med_bg - (-VGAMMA_KMS):+.3f} km/s.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
