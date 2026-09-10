"""Feasibility: are eclipse light curves differentiable in the geometry with SPICE's own binary?

Builds TZ For from two SPICE icospheres on a Keplerian orbit (spherical stars,
SPICE occlusion), synthesises the b-band flux near the deep eclipse through
SynthesisKernel + aemu, and asks JAX for d(eclipse depth)/d(R2) and
d(eclipse depth)/d(inclination). Compares with finite differences. If the
gradients are finite and agree, free geometry is a matter of plumbing; if the
occlusion is not differentiable, this says so. GPU job, minutes.
"""
import math, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, jax, jax.numpy as jnp
import tzfor_grad_inference as GI, tzfor_constants as K
from spice.models import IcosphereModel
from spice.models.binary import Binary, add_orbit, evaluate_orbit
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, kernel_flux_multi

print("devices:", jax.devices(), flush=True)
emu = GI.make_emulator()
names = ["marcs_teff", "marcs_logg", "feh", "vmicro", "a", "c", "n", "o", "r", "s"]
row = lambda T, g: jnp.array([T, g, -0.23, 1.5, 0, 0, 0, 0, 0, 0.])
M1, M2 = getattr(K, "PRIMARY_MASS", 2.057), getattr(K, "SECONDARY_MASS", 1.958)
R1, R2_0, INCL0 = K.PRIMARY_RADIUS, K.SECONDARY_RADIUS, K.INCL_DEG
P_yr = K.PERIOD_DAYS / 365.25
N_VERT = int(os.environ.get("N_VERT", 800))
lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), 4000)

def flux_b(R2, incl_deg, phases):
    """b-band-ish flux (4300-5900 A mean) of the blend at the given orbital phases."""
    b1 = IcosphereModel.construct(N_VERT, R1, M1, row(4889., 2.915), names)
    b2 = IcosphereModel.construct(N_VERT, R2, M2, row(6416., 3.534), names)
    binary = Binary.from_bodies(b1, b2)
    binary = add_orbit(binary, P_yr, 0.0, 0.0, jnp.deg2rad(incl_deg), 0.0, 0.0, 0.0, 0.0, 0.0, 200)
    out = []
    for ph in phases:
        m1, m2 = evaluate_orbit(binary, ph * P_yr)
        k1 = build_synthesis_kernel(m1, lw_ph, n_mu=16, oversample=1, half_width=8)
        k2 = build_synthesis_kernel(m2, lw_ph, n_mu=16, oversample=1, half_width=8)
        f = kernel_flux_multi(emu.intensity, [k1], row(4889., 2.915))[0, :, 0] + kernel_flux_multi(emu.intensity, [k2], row(6416., 3.534))[0, :, 0]
        out.append(jnp.mean(f))
    return jnp.stack(out)

phases = jnp.array([0.25, 0.5])              # out of eclipse, and conjunction (which eclipse depends on the orbit convention)
try:
    t = time.time(); f0 = jax.block_until_ready(flux_b(R2_0, INCL0, phases)); print(f"forward ok in {time.time()-t:.0f}s: flux ratio conj/quadrature = {float(f0[1]/f0[0]):.4f}", flush=True)
    depth = lambda R2, incl: 1.0 - flux_b(R2, incl, phases)[1] / flux_b(R2, incl, phases)[0]
    # no outer jit: icosphere() needs a concrete vertex count
    t = time.time(); g = jax.grad(depth, argnums=(0, 1))(R2_0, INCL0); jax.block_until_ready(g); print(f"grad ok in {time.time()-t:.0f}s: d depth/dR2 = {float(g[0]):+.5f} per Rsun, d depth/d incl = {float(g[1]):+.5f} per deg", flush=True)
    dj = depth
    h = 0.02; fd_r = (dj(R2_0 + h, INCL0) - dj(R2_0 - h, INCL0)) / (2 * h)
    h = 0.05; fd_i = (dj(R2_0, INCL0 + h) - dj(R2_0, INCL0 - h)) / (2 * h)
    print(f"finite differences: d depth/dR2 = {float(fd_r):+.5f}, d depth/d incl = {float(fd_i):+.5f}", flush=True)
    print(f"depth at the reference geometry: {float(dj(R2_0, INCL0)):.4f}", flush=True)
    ok = np.isfinite(float(g[0])) and np.isfinite(float(g[1]))
    print("VERDICT:", "gradients finite;" if ok else "gradients NOT finite;", f"agreement with FD: R2 {abs(float(g[0]) / float(fd_r) - 1) * 100 if fd_r != 0 else float('nan'):.1f}%  incl {abs(float(g[1]) / float(fd_i) - 1) * 100 if fd_i != 0 else float('nan'):.1f}%", flush=True)
except Exception as e:  # noqa: BLE001
    print("VERDICT: geometry gradient FAILED:", type(e).__name__, str(e)[:400]); traceback.print_exc()
print("===== DONE =====")
