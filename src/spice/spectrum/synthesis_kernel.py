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

Non-uniform surfaces (gravity darkening, irradiation) are handled with one
more node axis: give ``build_synthesis_kernel`` a per-element *coordinate* --
for gravity darkening the local log g, since the temperature is then a function
of g alone -- and a few coordinate nodes. Each element is split linearly between
its two neighbouring nodes, the kernels become ``(n_mu, n_c, k)``, and
``kernel_flux`` takes one parameter row per coordinate node (see
``gravity_darkened_rows``). The emulator then runs ``n_mu * n_c`` times per
star, and because its output does not depend on the epoch,
``kernel_flux_multi`` evaluates it once and reuses it for every epoch.
"""
import math
from typing import Callable, NamedTuple, Optional, Sequence

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
    """Geometry of one mesh at one epoch, reduced to per-node velocity kernels.

    Attributes:
        log_wavelengths: the caller's grid, uniform in log10(Angstrom), shape (n_w,).
        mu_nodes: ``mu`` interpolation nodes on [0, 1], shape (n_mu,).
        kernels: area-weighted shift kernels on the fine grid, shape
            (n_mu, n_c, 2*half_width+1) [R_sun^2]. ``kernels.sum()`` is the total
            visible projected area. ``n_c`` is 1 for a uniform surface.
        oversample: fine-grid pixels per caller pixel.
        half_width: kernel half width in fine pixels.
        coord_nodes: surface-coordinate nodes (e.g. log g), shape (n_c,); a single
            zero for a uniform surface.
    """
    log_wavelengths: Float[Array, "n_w"]
    mu_nodes: Float[Array, "n_mu"]
    kernels: Float[Array, "n_mu n_c k"]
    oversample: int
    half_width: int
    coord_nodes: Float[Array, "n_c"]

    @property
    def fine_log_wavelengths(self) -> Float[Array, "n_fine"]:
        """The internal grid: caller grid oversampled and padded by ``half_width`` on each side."""
        return _fine_grid(self.log_wavelengths, self.oversample, self.half_width)


def _fine_grid(lw, oversample, half_width):
    n = lw.shape[0]
    dfine = (lw[-1] - lw[0]) / ((n - 1) * oversample)
    n_core = (n - 1) * oversample + 1
    idx = jnp.arange(-half_width, n_core + half_width)
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


def _linear_split(values, nodes):
    """Index of the lower node and the fractional weight of the upper one, clipped to the range."""
    n = nodes.shape[0]
    if n == 1:
        return jnp.zeros(values.shape, jnp.int32), jnp.zeros_like(values)
    i = jnp.clip(jnp.searchsorted(nodes, values, side="right") - 1, 0, n - 2)
    f = jnp.clip((values - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0, 1.0)
    return i.astype(jnp.int32), f


def build_synthesis_kernel(mesh: MeshModel,
                           log_wavelengths: ArrayLike,
                           n_mu: int = 32,
                           oversample: int = 4,
                           v_pad_kms: float = 2.0,
                           half_width: Optional[int] = None,
                           element_coordinate: Optional[ArrayLike] = None,
                           coordinate_nodes: Optional[ArrayLike] = None) -> SynthesisKernel:
    """Reduce a mesh's visible geometry to per-node velocity kernels.

    Args:
        mesh: any ``MeshModel`` (SPICE or PHOEBE) at one epoch. Only
            ``visible_cast_areas``, ``mus`` and ``los_velocities`` are used, so
            occlusion is honoured (occluded elements have zero visible area).
        log_wavelengths: caller's grid, uniform in log10(Angstrom).
        n_mu: number of ``mu`` interpolation nodes.
        oversample: fine-grid pixels per caller pixel (2 suffices at HARPS
            sampling; 4 is the conservative default).
        v_pad_kms: margin added to the velocity range when sizing the kernel.
        half_width: kernel half width in fine pixels. Computed from the mesh's
            velocity range when ``None`` (needs a concrete mesh); pass it
            explicitly to build the kernel under ``jit``/``grad``.
        element_coordinate: optional per-element surface coordinate on which
            the atmospheric parameters depend (log g for gravity darkening).
        coordinate_nodes: sorted nodes of that coordinate, one parameter row
            each in ``kernel_flux``. Elements outside the range are clipped.

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
    im, fm = _linear_split(mus, nodes)

    if element_coordinate is None:
        if coordinate_nodes is not None:
            raise ValueError("coordinate_nodes given without element_coordinate")
        cnodes = jnp.zeros((1,), dtype=w.dtype)
        ic, fc = jnp.zeros(mus.shape, jnp.int32), jnp.zeros_like(mus)
    else:
        if coordinate_nodes is None:
            raise ValueError("element_coordinate given without coordinate_nodes")
        cnodes = jnp.asarray(coordinate_nodes, dtype=w.dtype).reshape(-1)
        ic, fc = _linear_split(jnp.asarray(element_coordinate, dtype=w.dtype), cnodes)
    n_c = cnodes.shape[0]

    kernels = jnp.zeros((n_mu, n_c, k), dtype=w.dtype)
    for dm, wm in ((0, 1.0 - fm), (1, fm)):
        for dc, wc in ((0, 1.0 - fc), (1, fc)):
            for dj, wj in ((0, 1.0 - f), (1, f)):
                kernels = kernels.at[im + dm, jnp.minimum(ic + dc, n_c - 1), jnp.clip(j + dj, 0, k - 1)].add(w * wm * wc * wj)
    return SynthesisKernel(jnp.asarray(lw_np), nodes, kernels, int(oversample), int(half_width), cnodes)


def gravity_darkened_rows(base_row: ArrayLike, idx_teff: int, idx_logg: int,
                          logg_nodes: ArrayLike, teff_ref: ArrayLike, logg_ref: ArrayLike,
                          beta: ArrayLike) -> Float[Array, "n_c n_p"]:
    """Parameter rows at log g nodes under ``Teff = Teff_ref (g / g_ref)^beta``.

    Differentiable in ``teff_ref``, ``logg_ref`` and ``beta``, so the gravity
    darkening exponent can be fitted. ``beta`` here is the exponent on g
    (von Zeipel: 0.25; convective envelopes: ~0.08), i.e. the bolometric
    ``gravb_bol`` of PHOEBE divided by 4.
    """
    base = jnp.asarray(base_row).reshape(-1)
    nodes = jnp.asarray(logg_nodes).reshape(-1)
    rows = jnp.broadcast_to(base, (nodes.shape[0], base.shape[0]))
    teff = teff_ref * jnp.power(10.0, beta * (nodes - logg_ref))
    return rows.at[:, idx_teff].set(teff).at[:, idx_logg].set(nodes)


def _stack_kernels(kernels: Sequence[SynthesisKernel]):
    """Common fine grid for several epochs: pad every kernel (centred) to the widest one."""
    k0 = kernels[0]
    for kk in kernels[1:]:
        if (kk.oversample != k0.oversample or kk.log_wavelengths.shape != k0.log_wavelengths.shape
                or kk.mu_nodes.shape != k0.mu_nodes.shape or kk.coord_nodes.shape != k0.coord_nodes.shape):
            raise ValueError("kernel_flux_multi needs kernels built on the same grid and nodes")
    hw = max(int(kk.half_width) for kk in kernels)
    stacked = jnp.stack([jnp.pad(kk.kernels, ((0, 0), (0, 0), (hw - kk.half_width, hw - kk.half_width)))
                         for kk in kernels])                       # (n_epoch, n_mu, n_c, 2*hw+1)
    return stacked, hw


def kernel_flux_multi(intensity_fn: Callable[[Float[Array, "n_w"], float, Float[Array, "n_p"]], Float[Array, "n_w 2"]],
                      kernels: Sequence[SynthesisKernel],
                      parameters: ArrayLike,
                      distance: float = 10.0,
                      wavelength_chunk_size: int = 32768) -> Float[Array, "n_epoch n_w 2"]:
    """Observed flux for every epoch's kernel with the emulator evaluated once.

    The emulator output at the (mu, coordinate) nodes does not depend on the
    epoch, so each node is synthesised once, transformed, and multiplied by
    every epoch's kernel spectrum; per-epoch cost is FFT products only.

    Args:
        intensity_fn: ``(log_wavelengths, mu, parameter_row) -> (n_w, 2)``.
        kernels: one ``SynthesisKernel`` per epoch, same grid and nodes.
        parameters: one row ``(n_p,)`` shared by all coordinate nodes, or one
            row per coordinate node ``(n_c, n_p)`` (see ``gravity_darkened_rows``).
        distance: parsec, as in ``simulate_observed_flux``.
        wavelength_chunk_size: emulator evaluation chunk on the fine grid; each
            chunk is rematerialised, which caps the network's backward memory.

    Returns:
        ``(n_epoch, n_w, 2)`` in erg/s/cm^2/Angstrom at ``distance``.
    """
    k0 = kernels[0]
    stacked, hw = _stack_kernels(kernels)
    n_epoch, n_mu, n_c, k = stacked.shape
    x_fine = _fine_grid(k0.log_wavelengths, k0.oversample, hw)
    n_fine = x_fine.shape[0]
    n_core = n_fine - k + 1
    rows = jnp.asarray(parameters)
    if rows.ndim == 1:
        rows = jnp.broadcast_to(rows, (n_c, rows.shape[0]))
    if rows.shape[0] != n_c:
        raise ValueError(f"parameters has {rows.shape[0]} rows for {n_c} coordinate nodes")

    # Linear convolution via FFT, mu/coordinate sum accumulated in the
    # frequency domain one node at a time: a direct convolution's backward pass
    # materialises an (n_fine x k) intermediate per node (37 GB for the full
    # HARPS range), and the emulator's backward pass keeps every layer's
    # activations for the points it saw -- hence the wavelength chunks too.
    n_fft = 1 << int(math.ceil(math.log2(n_fine + k - 1)))
    chunk = int(min(wavelength_chunk_size, n_fine))
    n_chunks = -(-n_fine // chunk)
    x_chunks = jnp.pad(x_fine, (0, n_chunks * chunk - n_fine), mode="edge").reshape(n_chunks, chunk)

    mu_flat = jnp.repeat(k0.mu_nodes, n_c)                          # (n_mu*n_c,)
    row_flat = jnp.tile(rows, (n_mu, 1))                            # (n_mu*n_c, n_p)
    kern_flat = jnp.transpose(stacked, (1, 2, 0, 3)).reshape(n_mu * n_c, n_epoch, k)

    @jax.checkpoint
    def node(carry, inputs):
        mu, row, kern = inputs
        spec = jax.lax.map(jax.checkpoint(lambda xc: intensity_fn(xc, mu, row)), x_chunks)
        spec = spec.reshape(n_chunks * chunk, -1)[:n_fine]         # (n_fine, 2)
        spec_f = jnp.fft.rfft(spec, n=n_fft, axis=0)                # (F, 2)
        kern_f = jnp.fft.rfft(kern, n=n_fft, axis=1)                # (n_epoch, F)
        return carry + kern_f[:, :, None] * spec_f[None, :, :], None

    real_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    acc0 = jnp.zeros((n_epoch, n_fft // 2 + 1, 2), dtype=jnp.result_type(real_dtype, jnp.complex64))
    acc, _ = jax.lax.scan(node, acc0, (mu_flat, row_flat, kern_flat))
    full = jnp.fft.irfft(acc, n=n_fft, axis=1)
    fine = full[:, k - 1:k - 1 + n_core]                            # the 'valid' segment
    flux = fine[:, ::k0.oversample]                                 # caller grid is a subset: no interpolation
    return flux * (_RSUN_PER_PC_SQ / distance ** 2)


def kernel_flux(intensity_fn: Callable[[Float[Array, "n_w"], float, Float[Array, "n_p"]], Float[Array, "n_w 2"]],
                kernel: SynthesisKernel,
                parameters: ArrayLike,
                distance: float = 10.0,
                wavelength_chunk_size: int = 32768) -> Float[Array, "n_w 2"]:
    """Observed flux for one epoch; see ``kernel_flux_multi``.

    Same units and distance scaling as ``simulate_observed_flux``
    (erg/s/cm^2/Angstrom at ``distance`` pc), both output channels of
    ``intensity_fn`` convolved separately. Differentiable in ``parameters``.
    """
    return kernel_flux_multi(intensity_fn, [kernel], parameters, distance, wavelength_chunk_size)[0]
