"""Showcase 8 (compute) - spectra grid for the interactive "star builder" page.

Mg b spectra (v1.2.0 emulator) on a grid of Teff x v_eq x inclination x spot
latitude x rotation phase, plus the geometry the browser needs to draw the same
star: unit-sphere vertices/faces, the per-face spot weight SPICE assigned, the
rotation matrices SPICE actually realised at each phase and the sky basis per
inclination. s08_build_html.py turns data.npz into a self-contained HTML page.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TEFFS = [4500.0, 5250.0, 6000.0, 6750.0, 7500.0]
VEQS = [0.0, 10.0, 25.0, 50.0, 100.0]
INCLS = [30.0, 60.0, 90.0]
SPOT_LATS = [None, 0.0, 45.0, 70.0]
SPOT_R, SPOT_DT_FRAC = 25.0, -0.25        # spot radius [deg], delta T as a fraction of Teff
N_PHASE = 12
WL = (5164.0, 5188.0)
LOGG = 4.4


def compute(args, out):
    C.setup_jax()
    import jax.numpy as jnp
    from spice.models.mesh_transform import add_rotation, evaluate_rotation
    from spice.models.spots import add_spot
    from spice.models.utils import lat_to_theta, lon_to_phi

    emu = C.load_emulator()
    quick = args.quick
    teffs, veqs, incls = (TEFFS[::4], VEQS[::4], INCLS[::2]) if quick else (TEFFS, VEQS, INCLS)
    n_vert = 400 if quick else 4000
    n_phase = 3 if quick else N_PHASE
    wl, lw = C.wl_grid(*WL, 120 if quick else 640)
    names = emu.stellar_parameter_names
    phases = np.arange(n_phase) / n_phase
    # flux[teff, veq, incl, spot, phase, wl]; unspotted rows repeat the phase-0 spectrum
    flux = np.zeros((len(teffs), len(veqs), len(incls), len(SPOT_LATS), n_phase, len(wl)), np.float32)
    weights = None
    for (a, T), (b, v), (c, inc) in itertools.product(enumerate(teffs), enumerate(veqs), enumerate(incls)):
        p = C.stellar_params(emu, T, LOGG, 0.0, 1.5)
        base = C.make_star(p, names, n_vertices=n_vert, inclination=inc)
        if weights is None:
            weights = []
            for lat in SPOT_LATS[1:]:
                s = add_spot(base, lat_to_theta(lat), lon_to_phi(0.0), SPOT_R, -1.0, 0, smoothness=0.2)
                weights.append(np.asarray(base.parameters)[:, 0] - np.asarray(s.parameters)[:, 0])
            weights = np.array(weights, np.float32)
        for d, lat in enumerate(SPOT_LATS):
            m0 = base if lat is None else add_spot(base, lat_to_theta(lat), lon_to_phi(0.0), SPOT_R,
                                                 SPOT_DT_FRAC * T, 0, smoothness=0.2)
            if v > 0:
                m0 = add_rotation(m0, v, rotation_axis=jnp.array(C.Z_AXIS))
            for e, ph in enumerate(phases):
                if lat is None and e > 0:
                    flux[a, b, c, d, e] = flux[a, b, c, d, 0]
                    continue
                if v > 0:
                    m = evaluate_rotation(m0, ph * C.rotation_period_s(1.0, v))
                else:   # turn the geometry, keep the velocity field at zero
                    spin = add_rotation(m0, 1.0, rotation_axis=jnp.array(C.Z_AXIS))
                    m = evaluate_rotation(spin, ph * C.rotation_period_s(1.0, 1.0))
                    m = m._replace(rotation_velocities=jnp.zeros_like(m.rotation_velocities))
                f, cont = C.synth(emu, m, lw)
                flux[a, b, c, d, e] = f / cont
        print(f"Teff {T:.0f} v {v:.0f} i {inc:.0f} done", flush=True)

    # geometry for the browser: unit-sphere mesh, realised rotation matrices, sky bases
    p = C.stellar_params(emu, 5800.0, LOGG, 0.0, 1.5)
    ref = add_rotation(C.make_star(p, names, n_vertices=n_vert, inclination=90.0), 10.0,
                       rotation_axis=jnp.array(C.Z_AXIS))
    v0 = np.asarray(ref.d_vertices)
    rots = []
    for ph in phases:
        vt = np.asarray(evaluate_rotation(ref, ph * C.rotation_period_s(1.0, 10.0)).d_vertices)
        good = np.linalg.norm(v0, axis=1) > 0
        Rm, *_ = np.linalg.lstsq(v0[good], vt[good], rcond=None)   # vt = v0 @ Rm
        rots.append(Rm)
    from spice.models.utils import inclination_to_los_axis
    bases = np.array([np.stack(C._sky_basis(np.asarray(inclination_to_los_axis(i)))) for i in incls])
    np.savez_compressed(out / "data.npz", wl=wl, flux=flux, teffs=teffs, veqs=veqs, incls=incls,
                        spot_lats=np.array([np.nan if s is None else s for s in SPOT_LATS]), phases=phases,
                        vertices=v0.astype(np.float32), faces=np.asarray(ref.faces).astype(np.int32),
                        spot_weights=weights, rotations=np.array(rots), bases=bases,
                        spot_dt_frac=SPOT_DT_FRAC, logg=LOGG)
    print("wrote", out / "data.npz", flux.shape, flush=True)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    compute(a, C.outdir(a))
