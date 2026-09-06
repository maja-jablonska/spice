"""Disc integration as a convolution: fast, differentiable synthesis for a fixed mesh.

``simulate_observed_flux`` evaluates the intensity function once per surface
element at that element's Doppler-shifted wavelengths -- exact, but for a
neural emulator it is ~10^3 network calls per epoch, which makes gradient-based
inference impractical.

On a grid that is uniform in ``log10(lambda)`` a Doppler shift is a pure
translation, and the intensity of every element depends on the geometry only
through ``mu``. Interpolating ``I(lambda, mu)`` linearly between a handful of
``mu`` nodes therefore turns the disc integral

    F(x) = sum_i A_i I(x - s_i, mu_i),      s_i = log10(1 + v_i / c)

into ``n_mu`` one-dimensional convolutions of the emulator output at the nodes
with per-node *velocity kernels* ``K_mu`` -- the area-weighted distribution of
sub-pixel shifts of the elements that map onto that node. The kernels carry all
of the geometry (rotation, pulsation, orbital motion, eclipses: an occluded
element simply has zero area) and are built once per epoch; the emulator then
runs ``n_mu`` times per star instead of once per element, for every epoch.

Two approximations, both controllable and both validated against
``simulate_observed_flux`` in ``tests/test_synthesis_kernel.py``:

* linear interpolation in ``mu`` between ``n_mu`` nodes (0.08% at 32 nodes for
  the aemu MARCS emulator);
* linear interpolation of the intensity at each element's fractional pixel
  shift. This is why the kernel works on an internal grid ``oversample`` times
  finer than the caller's: at 4x the worst-case flux error against the exact
  path was 0.28% (0.04% rms) and 0.06% in a normalised spectrum. The caller's
  grid is a subset of the fine grid, so the returned flux is never interpolated.

Building the kernel is itself differentiable in the element velocities (the
sub-pixel split is linear in the shift), so velocity-field parameters can be
fitted too if the kernel is rebuilt inside the traced function with a static
``half_width``.

Limitation: one parameter row per kernel. Elements are assumed to share the
atmospheric parameters (use an area-weighted mean; 0.06% for a gravity-darkened
TZ For component). Non-uniform surfaces need a (mu, parameter) node grid --
the same construction with one more axis -- and are not implemented yet.
"""
import math
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from spice.constants import C_KM_S
from spice.models import MeshModel

# (R_sun / pc)^2: the same solid-angle dilution simulate_observed_flux applies.
_RSUN_PER_PC_SQ = 5.08326693599739e-16


class SynthesisKernel(NamedTuple):
    """Geometry of one mesh at one epoch, reduced to per-``mu``-node velocity kernels.

    Attributes:
        log_wavelengths: the caller's grid, uniform in log10(Angstrom), shape (n_w,).
        mu_nodes: ``mu`` interpolation nodes on [0, 1], shape (n_mu,).
        kernels: area-weighted shift kernels on the fine grid, shape (n_mu, 2*half_width+1)
            [R_sun^2]. ``kernels.sum()`` is the total visible projected area.
        oversample: fine-grid pixels per caller pixel.
        half_width: kernel half width in fine pixels.
    """
    log_wavelengths: Float[Array, "n_w"]
    mu_nodes: Float[Array, "n_mu"]
    kernels: Float[Array, "n_mu k"]
    oversample: int
    half_width: int

    @property
    def fine_log_wavelengths(self) -> Float[Array, "n_fine"]:
        """The internal grid: caller grid oversampled and padded by ``half_width`` on each side."""
        lw = self.log_wavelengths
        n = lw.shape[0]
        dfine = (lw[-1] - lw[0]) / ((n - 1) * self.oversample)
        n_core = (n - 1) * self.oversample + 1
        idx = jnp.arange(-self.half_width, n_core + self.half_width)
        return lw[0] + idx * dfine


def _check_uniform_log_grid(log_wavelengths: np.ndarray) -> float:
    if log_wavelengths.ndim != 1 or log_wavelengths.shape[0] < 2:
        raise ValueError("log_wavelengths must be a 1-D array with at least two points")
    d = np.diff(log_wavelengths)
    dlog = float((log_wavelengths[-1] - log_wavelengths[0]) / (log_wavelengths.shape[0] - 1))
    if dlog <= 0 or np.max(np.abs(d - dlog)) > 1e-3 * dlog:
        raise ValueError(
            "SynthesisKernel needs a grid uniform in log10(wavelength): build it with "
            "jnp.linspace(log10(lo), log10(hi), n). A grid uniform in wavelength is not "
            "(a Doppler shift is a translation only in log wavelength).")
    return dlog


def build_synthesis_kernel(mesh: MeshModel,
                           log_wavelengths: ArrayLike,
                           n_mu: int = 32,
                           oversample: int = 4,
                           v_pad_kms: float = 2.0,
                           half_width: Optional[int] = None) -> SynthesisKernel:
    """Reduce a mesh's visible geometry to per-``mu``-node velocity kernels.

    Args:
        mesh: any ``MeshModel`` (SPICE or PHOEBE) at one epoch. Only
            ``visible_cast_areas``, ``mus`` and ``los_velocities`` are used, so
            occlusion is honoured (occluded elements have zero visible area).
        log_wavelengths: caller's grid, uniform in log10(Angstrom).
        n_mu: number of ``mu`` interpolation nodes.
        oversample: fine-grid pixels per caller pixel (4 is the validated default).
        v_pad_kms: margin added to the velocity range when sizing the kernel.
        half_width: kernel half width in fine pixels. Computed from the mesh's
            velocity range when ``None`` (needs a concrete mesh); pass it
            explicitly to build the kernel under ``jit``/``grad``.

    Returns:
        SynthesisKernel
    """
    lw_np = np.asarray(log_wavelengths, dtype=float)
    dlog = _check_uniform_log_grid(lw_np)
    dfine = dlog / oversample

    areas = jnp.asarray(mesh.visible_cast_areas)
    mus = jnp.asarray(mesh.mus)
    vlos = jnp.asarray(mesh.los_velocities)
    visible = areas > 0
    w = jnp.where(visible, areas, 0.0)

    if half_width is None:
        vmax = float(np.max(np.abs(np.asarray(vlos)[np.asarray(visible)]), initial=0.0)) + v_pad_kms
        half_width = int(math.ceil(math.log10(1.0 + vmax / C_KM_S) / dfine)) + 1
    k = 2 * half_width + 1

    # Observed spectrum of element i is the rest spectrum shifted by +s_i in
    # log10(lambda) (positive v_los = receding = redshift), matching
    # apply_vrad_log in spectrum.py which samples the rest spectrum at x - s.
    s = jnp.log10(1.0 + vlos / C_KM_S) / dfine            # fractional fine pixels
    j = jnp.floor(s)
    f = s - j
    j = j.astype(jnp.int32) + half_width                   # kernel index of the floor shift

    nodes = jnp.linspace(0.0, 1.0, n_mu)
    dmu = 1.0 / (n_mu - 1)
    im = jnp.clip(jnp.floor(mus / dmu), 0, n_mu - 2).astype(jnp.int32)
    fm = jnp.clip((mus - nodes[im]) / dmu, 0.0, 1.0)

    kernels = jnp.zeros((n_mu, k), dtype=w.dtype)
    for dm, wm in ((0, 1.0 - fm), (1, fm)):
        for dj, wj in ((0, 1.0 - f), (1, f)):
            kernels = kernels.at[im + dm, jnp.clip(j + dj, 0, k - 1)].add(w * wm * wj)
    return SynthesisKernel(jnp.asarray(lw_np), nodes, kernels, int(oversample), int(half_width))


def kernel_flux(intensity_fn: Callable[[Float[Array, "n_w"], float, Float[Array, "n_p"]], Float[Array, "n_w 2"]],
                kernel: SynthesisKernel,
                parameters: Float[Array, "n_p"],
                distance: float = 10.0) -> Float[Array, "n_w 2"]:
    """Observed flux for one parameter row through a prebuilt kernel.

    Same units and distance scaling as ``simulate_observed_flux``
    (erg/s/cm^2/Angstrom at ``distance`` pc), both output channels of
    ``intensity_fn`` convolved separately. Differentiable in ``parameters``;
    the emulator is evaluated once per ``mu`` node on the fine grid with
    per-node rematerialisation and an FFT convolution, so memory does not grow
    with ``n_mu`` or with the kernel width.
    """
    x_fine = kernel.fine_log_wavelengths
    params = jnp.asarray(parameters)
    n_fine = x_fine.shape[0]
    k = kernel.kernels.shape[1]
    n_core = n_fine - k + 1
    # Linear convolution via FFT: a direct convolution's backward pass
    # materialises an (n_fine x k) intermediate per mu node, which is 37 GB for
    # the full HARPS range. The mu sum is accumulated in the frequency domain
    # one node at a time, so memory stays at a single node's worth.
    n_fft = 1 << int(math.ceil(math.log2(n_fine + k - 1)))
    kern_f = jnp.fft.rfft(kernel.kernels, n=n_fft, axis=1)         # (n_mu, n_fft//2+1)

    @jax.checkpoint
    def node(carry, inputs):
        mu, kf = inputs
        spec = intensity_fn(x_fine, mu, params)                     # (n_fine, 2)
        return carry + jnp.fft.rfft(spec, n=n_fft, axis=0) * kf[:, None], None

    acc0 = jnp.zeros((n_fft // 2 + 1, 2), dtype=jnp.result_type(kern_f.dtype, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32))
    acc, _ = jax.lax.scan(node, acc0, (kernel.mu_nodes, kern_f))
    full = jnp.fft.irfft(acc, n=n_fft, axis=0)
    fine = full[k - 1:k - 1 + n_core]                               # the 'valid' segment
    flux = fine[::kernel.oversample]                                # caller grid is a subset: no interpolation
    return flux * (_RSUN_PER_PC_SQ / distance ** 2)
