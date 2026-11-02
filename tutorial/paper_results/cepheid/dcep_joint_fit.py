"""delta Cep joint photometry + HARPS-N spectroscopy fit through SynthesisKernel.

The delta Cep analogue of tz_fornacis/tzfor_kernel_fit.py, built on
dcep_kernel_fit.py (photometry only). Geometry is the SPIPS radius curve on a
5000-vertex icosphere with 12 km/s rotation; the mesh is evaluated at the
photometric model phases AND at the phase of every HARPS-N exposure, so each
spectroscopic epoch has its own pulsation velocity field and radius. The
per-epoch SynthesisKernels carry that geometry; the emulator is evaluated once
per (Teff node, mu node) and every epoch interpolates linearly between the
Teff nodes at its own Teff(phase), exactly as for the photometry.

    theta = [Teff Fourier coefficients (T0, a_k, b_k),
             E(B-V), [Fe/H], dphi, vmacro, vscale]

vmacro: one Gaussian macroturbulence (FWHM, km/s) combined with the HARPS-N
resolution (115 000), free by default -- Cepheid line widths are dominated by
it. vscale: multiplies the pulsation velocity field (NOT the rotation), i.e.
the ratio of the projection factor SPICE realises to the p = 1.2712 SPIPS used
to build the radius curve; fixed at 1 unless --free-vscale (the kernels are
then rebuilt inside the gradient, which SynthesisKernel supports with a
static half width). The systemic velocity of each observing season is
measured by cross-correlating the starting model with the Mg b window and is
then fixed (delta Cep is a binary: 2015 and 2019 differ by ~1 km/s).

Loss: chi2_phot (V, B-V from delta_cep.fits, optional per-source zero points)
+ spec_weight * chi2_spec (per epoch and 40 A window, sigma = 1/S_N of the
covering order in quadrature with --spec-floor, closed-form linear continuum
correction per epoch and window). The spectroscopic surrogate is checked
against exact per-element synthesis at the start and at the best fit.

Data preparation: dcep_harpsn_prepare.py (TNG archive -> SuppNet -> windows).
"""
import argparse, json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")
if sys.platform == "darwin" and "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "src"
for p in (str(SRC), str(HERE), str(HERE.parent), str(HERE.parent / "tz_fornacis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import jax
import jax.numpy as jnp
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import minimize
from types import SimpleNamespace

import emulator_params as ep
from cepheid_bundles import build_bundle, apply_phase_params
from build_cepheid_grid import make_phase_modifiers, _phase_modifiers_for
from dcep_kernel_fit import (BANDS, alambda_over_av, fourier_eval, fourier_fit, radius_template_mean_normalised,
                             periodic_interp, band_mag, load_spips, photometry_from_table, teff_observations, check_band_coverage)
from tzfor_compare_spectra import velocity_offset
from spice.models.utils import cast_to_los
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, kernel_flux_multi

C_KMS = 299792.458
R_HARPSN = 115000.0
PC_PER_KPC = 1000.0


class ElementMesh(NamedTuple):
    """The four arrays simulate_observed_flux reads from a mesh (a pytree, so it can be traced)."""
    visible_cast_areas: jnp.ndarray
    mus: jnp.ndarray
    los_velocities: jnp.ndarray
    parameters: jnp.ndarray


def spips_phase(mjd, mjd0, period, period1_s_per_yr):
    """SPIPS phase convention: linear ephemeris with a period change (reproduces the DATA table to 0.001)."""
    x = np.asarray(mjd, float) - mjd0
    pdot = period1_s_per_yr / (365.25 * 86400.0)
    return (x / period - 0.5 * pdot * x ** 2 / period ** 2) % 1.0


def los_parts(mesh):
    """Per-element LOS velocity split into rotation and pulsation (km/s, receding positive)."""
    rot = -cast_to_los(mesh.rotation_velocities, mesh.los_vector)
    puls = -cast_to_los(mesh.pulsation_velocities, mesh.los_vector)
    return jnp.asarray(mesh.mus), jnp.asarray(mesh.visible_cast_areas), jnp.asarray(rot), jnp.asarray(puls)


def kernel_from_parts(parts, lw, n_mu, oversample, half_width, dv, vscale):
    mus, areas, rot, puls = parts
    m = SimpleNamespace(mus=mus, visible_cast_areas=areas, los_velocities=rot + vscale * puls + dv)
    return build_synthesis_kernel(m, lw, n_mu, oversample, half_width=half_width)


def gauss_taps(sigma_px, half):
    x = jnp.arange(-half, half + 1, dtype=jnp.float64)
    k = jnp.exp(-0.5 * (x / sigma_px) ** 2)
    return k / jnp.sum(k)


def conv_valid(y, k):
    """jnp.convolve(row, k, mode="valid") along the last axis of y, as a sum of shifted slices. XLA lowers jnp.convolve to
    a cuDNN convolution, which fails on V100s (gpuvolta) with cuDNN 9.27 ("unknown cudnn status: 5003"); this form is
    plain elementwise arithmetic on every device and agrees with jnp.convolve to 5e-16 in float64."""
    n = y.shape[-1] - k.shape[0] + 1
    return sum(k[k.shape[0] - 1 - j] * y[..., j:j + n] for j in range(k.shape[0]))


def map_epochs(fn, theta, xs, devices=None):
    """lax.map of the rematerialised fn(theta, x) over the leading (epoch) axis of the arrays xs. With several devices the
    epochs are split between them by shard_map (padded to a multiple of the device count with copies of the last epoch,
    dropped again on return); theta is replicated."""
    n = xs[0].shape[0]; nd = len(devices) if devices else 1
    if nd <= 1:
        return jax.lax.map(jax.checkpoint(lambda x: fn(theta, x)), xs)
    n_pad = -(-n // nd) * nd
    xs = tuple(jnp.concatenate([a, jnp.broadcast_to(a[-1:], (n_pad - n,) + a.shape[1:])]) for a in xs)
    def local(th, *xl):
        return jax.lax.map(jax.checkpoint(lambda x: fn(th, x)), xl)
    shard_map = getattr(jax, "shard_map", None)
    if shard_map is None:                                                    # older JAX
        from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec as P
    specs = dict(mesh=Mesh(np.asarray(devices), ("dev",)), in_specs=(P(),) + tuple(P("dev") for _ in xs), out_specs=P("dev"))
    try:
        sharded = shard_map(local, check_vma=False, **specs)
    except TypeError:
        sharded = shard_map(local, check_rep=False, **specs)
    return jax.tree.map(lambda a: a[:n], sharded(theta, *xs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=str(HERE / "delta_cep.fits"))
    ap.add_argument("--harpsn", default=str(HERE / "harpsn_windows_log.npz"))
    ap.add_argument("--out", default=str(HERE / "dcep_fit_out" / "dcep_joint_fit_result.pkl"))
    ap.add_argument("--bundle", default="RozanskiT/TPayne-spice-harps")
    ap.add_argument("--n-mesh", type=int, default=5000)
    ap.add_argument("--n-phases", type=int, default=40)
    ap.add_argument("--n-wl", type=int, default=16000)
    ap.add_argument("--wl-range", type=float, nargs=2, default=(3780.0, 6910.0))
    ap.add_argument("--n-mu", type=int, default=32)
    ap.add_argument("--n-teff", type=int, default=16)
    ap.add_argument("--teff-pad", type=float, default=200.0)
    ap.add_argument("--fourier-order", type=int, default=4)
    ap.add_argument("--bands", nargs="+", default=["VT"],
                    help="fitted bands; each passband must lie entirely inside --wl-range (only Tycho VT does for HARPS: Johnson V "
                         "leaks 0.2%%, B 0.6%%, B_T 4-5%%, Hp 9-10%% of a 6000 K star's flux)")
    ap.add_argument("--ext-law", default="ccm89", choices=["ccm89", "f99", "g23"],
                    help="extinction law: ccm89 (Cardelli et al. 1989), f99 (Fitzpatrick 1999, the law of the Trahin et al. 2021 "
                         "E(B-V)) or g23 (Gordon et al. 2023); f99/g23 need the dust_extinction package")
    ap.add_argument("--ext-table", default=None, help="A_lambda / A_V table (wavelength [A], ratio; '# R_V x' header) instead of "
                    "CCM89, e.g. Fitzpatrick 1999 (the law Trahin et al. 2021 define E(B-V) with)")
    ap.add_argument("--allow-partial-bands", action="store_true",
                    help="accept bands that extend beyond --wl-range (truncated synthesis; for reproducing older runs only)")
    ap.add_argument("--predict-bands", nargs="+", default=[],
                    help="bands synthesised and reported (own zero point, held-out chi2) but given no weight in the fit; for "
                         "passbands that extend beyond the emulator's wavelength range (Hp, B_T), whose magnitudes the "
                         "truncated synthesis cannot reproduce exactly")
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--mass", type=float, default=5.26)
    ap.add_argument("--vrot", type=float, default=12.0)
    ap.add_argument("--vmicro", type=float, default=2.0)
    ap.add_argument("--marcs-fe-h-mode", default="feh", choices=["feh", "snap"], help="atmosphere [Fe/H] input of 12-input bundles: "
                    "'feh' = the synthesis [Fe/H]; 'snap' = its nearest MARCS node, as in the training grid (line formation uses [Fe/H] either way)")
    ap.add_argument("--marcs-fe-h", type=float, default=None, help="fix the atmosphere [Fe/H] input to this value (normally a MARCS "
                    "node, e.g. the other node bracketing [Fe/H]); overrides --marcs-fe-h-mode. Line formation still uses [Fe/H]")
    ap.add_argument("--vmicro-curve", default=None, help="vmicro(phase) from a table with phase in column 0 and vmicro in column 3 "
                    "(e.g. literature/dcep_luck2018_table6.txt): order-2 Fourier fit, applied to every mesh element")
    ap.add_argument("--logg-curve", default=None, help="log g(phase) from a table with phase in column 0 and log g in column 2 "
                    "(e.g. literature/dcep_luck2018_table6.txt, spectroscopic ionisation-balance gravities): order-2 Fourier fit, replaces "
                    "the SPIPS gravity in every mesh and in both exact paths")
    ap.add_argument("--vmacro-curve", default=None, help="starting vmacro(phase): table with phase in column 0 and a Gaussian broadening "
                    "velocity in column 4 (Luck 2018 Vb), times --vmacro-curve-scale, Fourier-fitted to --vmacro-order")
    ap.add_argument("--vmacro-instr-r", type=float, default=0.0, help="resolving power of the spectra the --vmacro-curve "
                    "broadening was measured on (Luck 2018 total broadening includes his instrument: Sandiford R 60000 for delta Cep); "
                    "its Gaussian FWHM c/R is removed in quadrature from the table FWHM (0 = keep the table as is)")
    ap.add_argument("--vmacro-curve-scale", type=float, default=1.665, help="table velocity -> FWHM (1.665 = 2 sqrt(ln 2) for a 1/e half-width)")
    ap.add_argument("--feh", type=float, default=None)
    ap.add_argument("--free-feh", action="store_true")
    ap.add_argument("--ebv0", type=float, default=0.092,
                    help="starting / fixed E(B-V): Trahin et al. 2021 (A&A 656, A102) SPIPS value for delta Cep, 0.092 +- 0.021")
    ap.add_argument("--fix-ebv", action="store_true")
    ap.add_argument("--teff-curve", default=None, help="start (and, with --fix-teff, fixed) Teff(phase) from a table with phase in column 0 "
                    "and Teff in column 1 (e.g. literature/dcep_luck2018_table6.txt, excitation-balance temperatures), Fourier order "
                    "--fourier-order, instead of the SPIPS curve")
    ap.add_argument("--fix-teff", action="store_true", help="hold the whole Teff(phase) curve at the SPIPS Fourier fit")
    ap.add_argument("--fix-teff-shape", action="store_true", help="hold the SPIPS Teff(phase) shape, fit only the mean T0")
    ap.add_argument("--teff-offset", type=float, default=0.0,
                    help="add this many K to the SPIPS Teff curve used as the start (and as the fixed curve with --fix-teff); "
                         "learning the template on a shifted curve tests whether the template carries the temperature scale")
    ap.add_argument("--fix-dphi", action="store_true", help="pin the radius-curve phase offset at 0")
    ap.add_argument("--distance-kpc", type=float, default=None)
    ap.add_argument("--per-source-zp", action="store_true", help="alias for --zp-mode source")
    ap.add_argument("--zp-mode", choices=("none", "source", "zero-mean"), default=None,
                    help="magnitude zero points: 'source' = free offset per source (level lost); 'zero-mean' = per-source offsets "
                         "constrained to zero mean (level kept, so distance x reddening is measured); colours never get offsets")
    ap.add_argument("--free-distance", action="store_true", help="fit the distance (flux scale) with a parallax prior")
    ap.add_argument("--plx-prior", type=float, nargs=2, default=(3.66, 0.15), metavar=("MAS", "ERR"),
                    help="parallax prior for --free-distance (default HST FGS 3.66 +- 0.15; Gaia companion ~3.36 +- 0.05)")
    ap.add_argument("--holdout-source", default=None, help="photometric source excluded from the fit and reported as a prediction test")
    ap.add_argument("--err-floor", type=float, default=0.0)
    ap.add_argument("--bv-sys", type=float, default=0.0,
                    help="extra B-V model error added in quadrature to the B-V sigma only [mag] (hydrostatic-colour systematic)")
    ap.add_argument("--phot-mode", choices=("colour", "BV"), default="colour",
                    help="'colour': fit V and B-V as published; 'BV': rebuild B = (B-V) + V from the same-night V of the same source and fit "
                         "B and V as independent magnitudes (removes the Cov(V, B-V) = -sigma_V^2 the colour mode ignores); B shares V's per-source zero point")
    ap.add_argument("--free-bsys", action="store_true",
                    help="fit an extra scatter s added in quadrature to the B (BV mode) or B-V (colour mode) errors, with the Gaussian log-normalisation term")
    ap.add_argument("--gravity", choices=("static", "dynamic"), default="static",
                    help="log g(phase): SPIPS static GM/R^2, or effective GM/R^2 + d2R/dt2 (cepheid_phase.effective_logg)")
    ap.add_argument("--phot-err", type=float, default=None, help="replace the catalogue photometric errors by this value [mag]")
    ap.add_argument("--init-from", default=None, help="result pickle whose physical parameters (matched by label) initialise this run")
    ap.add_argument("--free-ramp", action="store_true", help="fit a scale factor on the pulsation amplitude of the radius curve (photometric area only)")
    # spectroscopy
    ap.add_argument("--windows", type=float, nargs="+", default=None, help="window START wavelengths to keep")
    ap.add_argument("--drop-windows", type=float, nargs="+", default=[5560.0])
    ap.add_argument("--season", type=int, nargs="*", default=None, help="keep only these observing years")
    ap.add_argument("--no-continuum-fix", action="store_true", help="compare the model with the normalised spectra as they are "
                    "(no per-window linear continuum correction): a pure forward model")
    ap.add_argument("--n-spec-epochs", type=int, default=None, help="smoke tests: every k-th epoch so that N remain")
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--spec-floor", type=float, default=0.005)
    ap.add_argument("--spec-weight", type=float, default=1.0)
    ap.add_argument("--vmacro0", type=float, default=10.0, help="starting macroturbulence FWHM [km/s]")
    ap.add_argument("--fix-vmacro", action="store_true")
    ap.add_argument("--vmacro-order", type=int, default=0,
                    help="Fourier order of the macroturbulence phase curve vmacro(phase) = v0 + sum c_k cos + s_k sin; 0 = constant")
    ap.add_argument("--free-vscale", action="store_true")
    ap.add_argument("--vscale0", type=float, default=None, help="value (start, or with --maxiter 0 the fixed value) of the "
                    "pulsation-velocity scale, p = p_SPIPS x vscale; applied after --init-from (profile scans of p)")
    ap.add_argument("--dv-sys", type=float, nargs="*", default=None, help="systemic velocity per season [km/s] (else measured by CCF)")
    ap.add_argument("--dv-mode", choices=("season", "epoch"), default="epoch",
                    help="'epoch': each exposure gets its own CCF velocity offset (absorbs the SPIPS RV-spline shape error, "
                         "~2 km/s on the steep branch); 'season': one offset per season (use with --free-vscale)")
    ap.add_argument("--mask", type=float, nargs="+", default=[], help="wavelength ranges excluded from the spectra (lo hi ...)")
    ap.add_argument("--line-mask", default=None,
                    help="text file of REST wavelengths [A] (first column; '#' comments): only pixels within --line-mask-hw of a listed "
                         "line, after shifting it by each epoch's line velocity (dv_epoch + <v_los,puls>), enter the spectral chi2")
    ap.add_argument("--line-mask-hw", type=float, default=15.0, help="half-width [km/s] of each --line-mask line")
    ap.add_argument("--cno0", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("C", "N", "O"),
                    help="[C/Fe] [N/Fe] [O/Fe] (emulator c, n, o; Magg+2022 scale): fixed values, or the start with --free-cno")
    ap.add_argument("--free-cno", action="store_true", help="fit [C/Fe], [N/Fe], [O/Fe] (post-dredge-up supergiant: C down, N up)")
    ap.add_argument("--resume-pass1", default=None,
                    help="checkpoint <out>_pass1.pkl written by an earlier run with the same setup: skip pass 1 and run only pass 2 "
                         "(--sigma-from-residuals) from its solution and per-window sigma, e.g. after a walltime kill in pass 2")
    ap.add_argument("--free-depth-scale", action="store_true",
                    help="free global line-depth scale s: normalised model -> 1 - s (1 - model) (diagnostic of a depth deficit)")
    ap.add_argument("--line-template", action="store_true",
                    help="fit a multiplicative rest-frame correction template per window (shared by all epochs) that absorbs the emulator's line-list errors")
    ap.add_argument("--template-ridge", type=float, default=1.0, help="ridge prior weight on the template: penalty = ridge * sum (t / 0.05)^2")
    ap.add_argument("--template-from", default=None, help="result pickle whose fitted template initialises this run's template")
    ap.add_argument("--fix-template", action="store_true", help="keep the template at its initial values (use with --template-from)")
    ap.add_argument("--robust-delta", type=float, default=0.0, help="pseudo-Huber transition in sigma units (0 = plain chi2)")
    ap.add_argument("--error-model", default=None, help="JSON from dcep_error_model.py: per-window and per-band sigma scales "
                    "(residual chi2/N x correlation length [x epoch correlation]) measured at a previous best fit; scales sigma only")
    ap.add_argument("--sigma-from-residuals", action="store_true",
                    help="second pass: replace the spectral sigma of every (epoch, window) by the residual rms of the first pass")
    ap.add_argument("--holdout-windows", type=float, nargs="*", default=[], help="window START wavelengths modelled but excluded from the fit (prediction test)")
    ap.add_argument("--holdout-epochs", type=int, default=0, help="hold out every k-th spectroscopic epoch from the fit and report its chi2 (validates the template)")
    ap.add_argument("--phot-only", action="store_true")
    ap.add_argument("--spec-only", action="store_true")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--surrogate-tol", type=float, default=0.01)
    ap.add_argument("--jackknife", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--dv-from", default=None,
                    help="restore the per-epoch velocity offsets (dv_epoch, dv_season) from a previous result pickle "
                         "instead of re-measuring them: with --init-from this replays that fit bit-exactly")
    ap.add_argument("--save-all-spectra", action="store_true",
                    help="store the continuum-matched best-fit model of EVERY epoch in every kept window (result['all_spectra']), with "
                         "obs, good, sigma and the per-epoch line velocity dv_epoch + <v_los,puls> (input of dcep_line_list.py)")
    ap.add_argument("--exact-examples", action="store_true",
                    help="after the fit, synthesise the example epochs of every window per element (the emulator "
                         "called for each face) instead of through the kernel surrogate; stored as example_spectra_exact")
    ap.add_argument("--exact-photometry", action="store_true",
                    help="after the fit, synthesise the photometric model phases per element and store the band "
                         "magnitudes as mags_exact (the surrogate's are mags_fit)")
    ap.add_argument("--exact-all", action="store_true",
                    help="after the fit, synthesise EVERY spectroscopic epoch in every kept window and every photometric model "
                         "phase per element (no kernel quadrature) and score them with the fit's own chi2; stored as exact_all. "
                         "With --maxiter 0 this is a pure forward model at the starting parameters")
    ap.add_argument("--check-grad", action="store_true", help="compare autodiff with central finite differences at the start")
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--exact-loss", action="store_true",
                    help="optimise the EXACT per-element model: every spectroscopic epoch and photometric phase synthesised element by element "
                         "(simulate_observed_flux, each face at its own Teff(phase), mu and Doppler shift) inside the gradient -- no kernels and "
                         "no Teff-node interpolation; the per-season CCF velocities are measured against the exact start model too")
    ap.add_argument("--free-vrot-scale", action="store_true", help="fit a scale on the rotational LOS velocities (--exact-loss only)")
    ap.add_argument("--vrot-scale-max", type=float, default=2.0, help="upper bound of the rotation scale (the Luck-curve fits railed at 2.0)")
    ap.add_argument("--exact-phot-wl", type=int, default=16000, help="wavelength points of the exact photometric synthesis (over --wl-range); 4000 left "
                    "B and B_T off by ~5 mmag rms (Doppler-shifted lines crossing the sample points); the start prints a 2x-sampling check")
    ap.add_argument("--exact-chunk", type=int, default=256, help="surface elements per chunk in simulate_observed_flux")
    ap.add_argument("--cheap-block-pca", default=None, help="multi-fidelity --exact-loss fit: a block_pca_emulator.py output directory; "
                    "the block-PCA MLP (block_pca_spice.BlockPCAIntensityEmulator) is the cheap model, corrected to TPayne's value and "
                    "gradient at every outer iterate (first-order consistent trust-region loop; converges to TPayne's optimum)")
    ap.add_argument("--cheap-model", default="mlp_4x2048", help="model subdirectory of --cheap-block-pca")
    ap.add_argument("--cheap-student", default=None, help="with --cheap-block-pca: a 23_distill_basis_student.py run directory; the "
                    "basis student (block_pca_spice.BasisStudentIntensityEmulator, ~4x closer to TPayne) makes the cheap SPECTRA, "
                    "block-PCA keeps the cheap photometry (the student would have to rebuild its whole 456k-pixel grid there)")
    ap.add_argument("--cheap-margin", type=float, default=80.0, help="Doppler margin (km/s) of the student's cached window rows")
    ap.add_argument("--cheap-chunk", type=int, default=64, help="surface elements per chunk in the cheap loss (each element rebuilds the "
                    "whole 313k-pixel block-PCA grid)")
    ap.add_argument("--mf-outer", type=int, default=30, help="outer iterations (one TPayne value+gradient each)")
    ap.add_argument("--mf-inner", type=int, default=30, help="L-BFGS-B iterations on the corrected cheap loss per outer iteration")
    ap.add_argument("--mf-radius", type=float, default=1.0, help="initial trust radius (max |step| in the fit's scaled units)")
    ap.add_argument("--mf-min-radius", type=float, default=1e-3)
    ap.add_argument("--gpus", type=int, default=1, help="--exact-loss: split the epochs and photometric phases over this many devices (jax.devices()[:n])")
    args = ap.parse_args()
    EM = None
    if args.error_model:
        EM = json.load(open(args.error_model))
        print(f"error model {args.error_model}: spectral sigma x " + ", ".join(f"{k} {v:.1f}" for k, v in EM["spec"].items())
              + "; photometric sigma x " + ", ".join(f"{k} {v:.2f}" for k, v in EM["phot"].items()), flush=True)
    if args.cheap_block_pca and not args.exact_loss:
        raise SystemExit("--cheap-block-pca needs --exact-loss (the cheap model replaces TPayne in the exact per-element synthesis)")
    if args.cheap_student and not args.cheap_block_pca:
        raise SystemExit("--cheap-student needs --cheap-block-pca (block-PCA makes the cheap photometry)")
    print("devices:", jax.devices(), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ SPIPS inputs
    data, model, zps, meta = load_spips(args.fits)
    hdr = fits.getheader(args.fits, 0)
    period1 = float(hdr.get("PARAM PERIOD1", 0.0))
    period = meta["period"]; d_pc = (args.distance_kpc or meta["d_kpc"]) * PC_PER_KPC
    feh0 = args.feh if args.feh is not None else meta["metal"]
    args.bands = list(dict.fromkeys(list(args.bands) + list(args.predict_bands)))
    check_band_coverage(args.bands, args.wl_range, args.allow_partial_bands)
    obs = photometry_from_table(data, set(args.bands))
    teff_obs = teff_observations(data)
    if args.phot_mode == "BV":
        # every SPIPS B-V point has a V point from the same source within minutes: pair them
        V, C = obs["V"], obs.pop("B-V"); pick = []
        for i in range(C["mag"].size):
            same = np.where(V["src"] == C["src"][i])[0]
            dph = np.abs(V["phase"][same] - C["phase"][i]); dph = np.minimum(dph, 1.0 - dph)
            j = same[np.argmin(dph)]
            if dph.min() > 2e-3:      # 0.01 d; Engle+ 2014 took B and V up to ~4 min apart
                raise SystemExit(f"--phot-mode BV: no same-night V for B-V point {i} ({C['src'][i]}, phase {C['phase'][i]:.4f})")
            pick.append(j)
        pick = np.asarray(pick)
        # B and V are measured separately: B gets the paired V's error (the catalogue B-V errors equal the V errors, so
        # sigma_B^2 = sigma_BV^2 - sigma_V^2 would vanish); with --phot-err both are replaced anyway
        obs["B"] = dict(phase=C["phase"], mag=C["mag"] + V["mag"][pick], err=V["err"][pick].copy(), src=C["src"])
        print(f"phot-mode BV: {pick.size} B points rebuilt from B-V + same-night V", flush=True)
    N_ph = sum(v["mag"].size for k, v in obs.items() if k not in args.predict_bands)
    print(f"SPIPS: P {period:.6f} d, dP/dt {period1} s/yr, MJD0 {meta['mjd0']:.4f}, d {d_pc:.0f} pc, [Fe/H] {feh0:+.2f}, "
          f"p {meta['pfactor']:.4f}; vmicro " + (f"from {args.vmicro_curve}" if args.vmicro_curve else f"fixed {args.vmicro:.1f}"), flush=True)
    for k, v in obs.items():
        print(f"  {k:4s}: {v['mag'].size:4d} points, sources {sorted(set(v['src']))}")

    mph = np.asarray(model["PHASE"], float); mteff = np.asarray(model["Teff"], float)
    o = np.argsort(mph); mph, mteff = mph[o], mteff[o]
    K = args.fourier_order
    coef0, fit0 = fourier_fit(mph, mteff, K)
    if args.teff_curve:
        tab = np.loadtxt(args.teff_curve)
        coef0, _ = fourier_fit(tab[:, 0], tab[:, 1], K)
        g = np.asarray(fourier_eval(jnp.asarray(np.linspace(0, 1, 200)), jnp.asarray(coef0)))
        print(f"Teff(phase) from {args.teff_curve}: {g.min():.0f}-{g.max():.0f} K, mean {g.mean():.0f} (SPIPS {mteff.min():.0f}-{mteff.max():.0f}, "
              f"mean {mteff.mean():.0f}); fit rms {np.std(tab[:, 1] - np.asarray(fourier_eval(jnp.asarray(tab[:, 0]), jnp.asarray(coef0)))):.0f} K", flush=True)
    coef0[0] += args.teff_offset
    logg_mean = float(np.mean(model["logg"])); vmic_mean = args.vmicro
    logg_curve = None
    if args.logg_curve:
        tab = np.loadtxt(args.logg_curve)
        c_lg, _ = fourier_fit(tab[:, 0], tab[:, 2], 2)
        logg_curve = lambda p, c=c_lg: float(fourier_eval(jnp.asarray(p), jnp.asarray(c)))
        g = np.array([logg_curve(p) for p in np.linspace(0, 1, 200)])
        print(f"log g(phase) from {args.logg_curve}: {g.min():.2f}-{g.max():.2f}, mean {g.mean():.2f} "
              f"(SPIPS mean {logg_mean:.2f}; table values {tab[:, 2].min():.2f}-{tab[:, 2].max():.2f}, fit rms {np.std(tab[:, 2] - [logg_curve(p) for p in tab[:, 0]]):.2f})", flush=True)
        logg_mean = float(g.mean())
    vmic_at = lambda p: vmic_mean                       # vmicro(phase) [km/s]; constant unless --vmicro-curve
    if args.vmicro_curve:
        tab = np.loadtxt(args.vmicro_curve)
        c_vm, _ = fourier_fit(tab[:, 0], tab[:, 3], 2)
        vmic_at = lambda p, c=c_vm: float(fourier_eval(jnp.asarray(p), jnp.asarray(c)))
        g = np.array([vmic_at(p) for p in np.linspace(0, 1, 200)]); vmic_mean = float(g.mean())
        print(f"vmicro(phase) from {args.vmicro_curve}: {g.min():.2f}-{g.max():.2f} km/s, mean {vmic_mean:.2f}", flush=True)

    # ------------------------------------------------------------------ HARPS-N data
    H = np.load(args.harpsn, allow_pickle=True); H = {k: H[k] for k in H.files}
    all_windows = [tuple(w) for w in H["windows"]]
    keep = [i for i, w in enumerate(all_windows)
            if (args.windows is None or any(abs(w[0] - s) < 1 for s in args.windows))
            and not any(abs(w[0] - s) < 1 for s in args.drop_windows)]
    sel = np.ones(H["times"].size, bool)
    if args.season:
        sel &= np.isin(H["season"], args.season)
    idx = np.where(sel)[0]
    if args.n_spec_epochs and args.n_spec_epochs < idx.size:
        idx = idx[np.linspace(0, idx.size - 1, args.n_spec_epochs).round().astype(int)]
    times = np.asarray(H["times"])[idx]; season = np.asarray(H["season"])[idx]
    seasons = sorted(set(season.tolist())); sidx = np.array([seasons.index(s) for s in season])
    phases_spec = spips_phase(times, meta["mjd0"], period, period1)
    dlog = float(H["dlog"])
    print(f"HARPS-N: {idx.size} epochs (seasons {seasons}), phases {phases_spec.min():.3f}-{phases_spec.max():.3f}, "
          f"windows {[all_windows[i] for i in keep]}", flush=True)
    obs_sp, good_sp, sig_sp, xw, lw_obs = {}, {}, {}, {}, {}
    N_sp = 0
    for wi in keep:
        ob = np.array(H[f"obs_{wi}"], float)[idx]
        wl_w = 10.0 ** np.asarray(H[f"logwl_{wi}"])
        for lo, hi in zip(args.mask[::2], args.mask[1::2]):
            ob[:, (wl_w >= lo) & (wl_w <= hi)] = np.nan
        g = np.isfinite(ob)
        obs_sp[wi] = jnp.asarray(np.where(g, ob, 1.0)); good_sp[wi] = jnp.asarray(g)
        sig_sp[wi] = jnp.asarray(np.hypot(np.asarray(H["sigma_win"])[idx, wi], args.spec_floor))[:, None]
        if EM is not None:
            sig_sp[wi] = sig_sp[wi] * EM["spec"][f"{H['windows'][wi][0]:.0f}"]
        xw[wi] = jnp.linspace(-1.0, 1.0, ob.shape[1]); lw_obs[wi] = jnp.asarray(H[f"logwl_{wi}"]); N_sp += int(g.sum())

    # ------------------------------------------------------------------ emulator + meshes
    from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
    emu = IntensityPretrainedAemuSpectrumEmulator(args.bundle)
    iT, iG, iF = ep.parameter_index(emu, "teff"), ep.parameter_index(emu, "logg"), ep.parameter_index(emu, "feh")
    iVM = ep.parameter_index(emu, "vmicro")
    # 12-input bundles also take the atmosphere metallicity marcs_fe_h: set_ab writes it from [Fe/H] on every update
    # (--marcs-fe-h-mode: 'feh' copies it, 'snap' takes the nearest MARCS node as in the training grid), so a free
    # [Fe/H] never leaves the structure at the start value. Line formation always uses the continuous [Fe/H].
    from spice.spectrum.marcs_nodes import resolve_marcs_fe_h
    iMF = list(emu.stellar_parameter_names).index("marcs_fe_h") if "marcs_fe_h" in emu.stellar_parameter_names else None
    print(f"marcs_fe_h: {'not an input' if iMF is None else ('mode ' + repr(args.marcs_fe_h_mode) if args.marcs_fe_h is None else f'FIXED at {args.marcs_fe_h:+.2f}')}", flush=True)
    base_row = jnp.asarray(ep.to_parameters(emu, dict(teff=float(mteff[0]), logg=logg_mean, feh=feh0, vmicro=vmic_mean)))
    # abundances enter the emulator as one vector ab = ([Fe/H], [C/Fe], [N/Fe], [O/Fe]); [Fe/H] also sets marcs_fe_h (iF above)
    iCNO = np.array([ep.parameter_index(emu, k) for k in ("c_fe", "n_fe", "o_fe")])
    base_row = base_row.at[iCNO].set(jnp.asarray(args.cno0, dtype=base_row.dtype))
    ab0 = jnp.asarray([feh0, *args.cno0])

    def set_ab(x, ab):
        """Write [Fe/H] (and marcs_fe_h) and C, N, O into a row or an element matrix (last axis = parameters)."""
        x = x.at[..., iF].set(ab[0]).at[..., iCNO].set(ab[1:])
        if iMF is None:
            return x
        return x.at[..., iMF].set(resolve_marcs_fe_h(ab[0], args.marcs_fe_h_mode) if args.marcs_fe_h is None else args.marcs_fe_h)

    _r = set_ab(base_row, ab0)
    print(f"emulator inputs at the start: feh (line formation) = {float(_r[iF]):+.3f}"
          + ("" if iMF is None else f", marcs_fe_h (atmosphere structure) = {float(_r[iMF]):+.3f}"), flush=True)

    with fits.open(args.fits, ignore_missing_simple=True) as h:
        pulsation_data = Table(h[2].data)
    R_sp = np.asarray(pulsation_data["R"], float); ph_sp = np.asarray(pulsation_data["PHASE"], float)
    R_SCALE_D = args.distance_kpc / meta["d_kpc"] if args.distance_kpc else 1.0      # radius and pulsation-velocity amplitude scale
    if args.distance_kpc and abs(args.distance_kpc - meta["d_kpc"]) > 1e-6:
        # SPIPS radii are angular diameters (interferometry) x its distance: keep the angular diameter, scale R with d
        R_sp = R_sp * args.distance_kpc / meta["d_kpc"]
        print(f"distance {1e3 * args.distance_kpc:.1f} pc instead of the SPIPS {1e3 * meta['d_kpc']:.1f} pc: SPIPS radii scaled by "
              f"{args.distance_kpc / meta['d_kpc']:.4f} (same angular diameter), <R> = {R_sp.mean():.2f} R_sun", flush=True)
    fourier = radius_template_mean_normalised(ph_sp, R_sp)
    teff_at, logg_at, _ = make_phase_modifiers(pulsation_data, gravity=args.gravity, period_days=period)
    if logg_curve is not None:
        logg_at = logg_curve                      # literature gravity curve instead of SPIPS (--gravity is then unused)
    if args.teff_curve:
        teff_at = lambda p, c=coef0.copy(): float(fourier_eval(jnp.asarray(p), jnp.asarray(c)))   # start meshes on the literature curve too
    radius = args.radius if args.radius is not None else float(R_sp.mean())
    phases_model = np.linspace(0.0, 1.0, args.n_phases, endpoint=False)
    t = time.time()
    b_ph = build_bundle(emu, np.asarray(base_row), fourier_params=fourier, period=period, timeseries=jnp.asarray(phases_model * period),
                        n_mesh=args.n_mesh, radius=radius, mass=args.mass, rotation_velocity=args.vrot,
                        param_names_attr="stellar_parameter_names", desc="  photometric epochs")
    b_sp = build_bundle(emu, np.asarray(base_row), fourier_params=fourier, period=period, timeseries=jnp.asarray(phases_spec * period),
                        n_mesh=args.n_mesh, radius=radius, mass=args.mass, rotation_velocity=args.vrot,
                        param_names_attr="stellar_parameter_names", desc="  spectroscopic epochs")
    b_sp = apply_phase_params(b_sp, phases_spec, _phase_modifiers_for(emu, teff_at, logg_at, vmic_at))
    snaps_ph, snaps_sp = b_ph.snapshots, b_sp.snapshots
    print(f"built {len(snaps_ph)} + {len(snaps_sp)} snapshots ({args.n_mesh} vertices) in {time.time() - t:.0f}s", flush=True)
    rad = np.array([math.sqrt(float(jnp.sum(m.visible_cast_areas)) / math.pi) for m in snaps_ph])
    R_at = np.interp(phases_model, ph_sp[np.argsort(ph_sp)], R_sp[np.argsort(ph_sp)], period=1.0)
    print(f"  projected-area radius vs SPIPS at the model phases: rms {np.sqrt(np.mean((rad - R_at) ** 2)):.3f} R_sun")
    vpuls_model = np.array([float(np.average(np.asarray(-cast_to_los(m.pulsation_velocities, m.los_vector)), weights=np.asarray(m.visible_cast_areas)))
                            for m in snaps_sp])   # area-weighted LOS pulsation velocity per spectroscopic epoch (diagnostic)

    # ------------------------------------------------------------------ photometry pieces (as dcep_kernel_fit)
    lw = jnp.linspace(math.log10(args.wl_range[0]), math.log10(args.wl_range[1]), args.n_wl)
    wl = 10.0 ** np.asarray(lw); wl_j = jnp.asarray(wl)
    ph_kernels = [build_synthesis_kernel(m, lw, args.n_mu, 1) for m in snaps_ph]
    from spice.spectrum import filter as spfilter
    resp, zp = {}, {}
    for b in args.bands:
        cls, zkey = BANDS[b]
        resp[b] = jnp.asarray(getattr(spfilter, cls)().filter_responses_for_wavelengths(wl_j)); zp[b] = zps[zkey]
    alam_av = jnp.asarray(alambda_over_av(wl, meta["r_v"], args.ext_table, args.ext_law))
    print(f"extinction: {args.ext_table or args.ext_law}, R_V {meta['r_v']}; A_lambda/A_V at 4400 / 5500 / 6500 A = "
          + " / ".join(f"{float(np.interp(x, wl, np.asarray(alam_av))):.3f}" for x in (4400.0, 5500.0, 6500.0)), flush=True)
    d0_kpc = d_pc / PC_PER_KPC
    zp_mode = args.zp_mode or ("source" if args.per_source_zp else "none")
    T_lo, T_hi = float(mteff.min() - args.teff_pad), float(mteff.max() + args.teff_pad)
    T_nodes = jnp.linspace(T_lo, T_hi, args.n_teff)
    logg_ph = jnp.asarray([logg_at(p) for p in phases_model]) - logg_mean
    VM_PH = jnp.asarray([vmic_at(p) for p in phases_model])          # vmicro of the photometric phases (exact paths)
    logg_sp = jnp.asarray([logg_at(p) for p in phases_spec]) - logg_mean
    DG = 0.05
    ph_model = jnp.asarray(phases_model)
    obs_j = {}
    for k, v in obs.items():
        srcs = sorted(set(v["src"]))
        # PBS -v cannot pass spaces inside EXTRA_ARGS: accept 'Kiss+_1998' for 'Kiss+ 1998'
        hold = (args.holdout_source or "").replace("_", " ")
        infit = np.array([s != hold for s in v["src"]], float) if hold else np.ones(v["src"].size)
        if k in args.predict_bands:
            infit = np.zeros(v["src"].size)
        if args.holdout_source and infit.sum() == v["src"].size:
            print(f"WARNING: --holdout-source {args.holdout_source!r} matches no {k} points; sources are {srcs}", flush=True)
        err = np.full_like(v["err"], args.phot_err) if args.phot_err else v["err"]
        em_s = EM["phot"].get(k, 1.0) if EM is not None else 1.0
        obs_j[k] = dict(phase=jnp.asarray(v["phase"]), mag=jnp.asarray(v["mag"]), sig=em_s * jnp.asarray(np.hypot(np.hypot(err, args.err_floor), args.bv_sys if k in ("B-V", "B") else 0.0)),
                        src=jnp.asarray([srcs.index(s) for s in v["src"]]), n_src=len(srcs), srcs=srcs, infit=jnp.asarray(infit))

    def phot_nodes(ab):
        def one(T):
            row = set_ab(base_row.at[iT].set(T), ab)
            f0 = kernel_flux_multi(emu.intensity, ph_kernels, row, wavelength_chunk_size=args.chunk)[..., 0]
            fg = kernel_flux_multi(emu.intensity, ph_kernels, row.at[iG].add(DG), wavelength_chunk_size=args.chunk)[..., 0]
            return f0 + (fg - f0) * (logg_ph / DG)[:, None]
        return jax.lax.map(one, T_nodes)

    def node_weights(teff_e):
        """Catmull-Rom (C1) interpolation across the uniform Teff nodes: 4 indices and 4 weights per epoch.

        Linear interpolation gave a piecewise-quadratic chi2 whose gradient jumps at every node
        crossing; with 3.5M spectral pixels L-BFGS-B stalled on those kinks after ~15 iterations.
        """
        n = args.n_teff
        i = jnp.clip(jnp.searchsorted(T_nodes, teff_e, side="right") - 1, 0, n - 2)
        t = jnp.clip((teff_e - T_nodes[i]) / (T_nodes[i + 1] - T_nodes[i]), 0.0, 1.0)
        t2, t3 = t * t, t * t * t
        w = jnp.stack([-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2], -1)
        idx = jnp.stack([jnp.clip(i - 1, 0, n - 1), i, i + 1, jnp.clip(i + 2, 0, n - 1)], -1)
        return idx, w

    def node_interp(F_nodes, idx, w, e_idx):
        """F_nodes[k, e] weighted over the 4 stencil nodes; F_nodes (n_T, n_ep, ...)."""
        out = 0.0
        for k in range(4):
            out = out + jnp.expand_dims(w[:, k], tuple(range(1, F_nodes.ndim - 1))) * F_nodes[idx[:, k], e_idx]
        return out

    n_coef = 2 * K + 1
    KV = args.vmacro_order; n_vmac = 2 * KV + 1
    I_EBV, I_FEH, I_DPHI, I_VMAC, I_VSC = n_coef, n_coef + 1, n_coef + 2, n_coef + 3, n_coef + 3 + n_vmac
    I_DIST = I_VSC + 1
    I_RAMP = I_DIST + 1
    I_BSYS = I_RAMP + 1
    I_DEPTH = I_BSYS + 1
    I_CNO = I_DEPTH + 1                       # [C/Fe], [N/Fe], [O/Fe]
    I_VROT = I_CNO + 3                        # scale on the rotational LOS velocities (exact loss only)

    def ab_of(theta):
        return jnp.concatenate([theta[I_FEH][None], theta[I_CNO:I_CNO + 3]])
    BSYS_KEY = "B" if args.phot_mode == "BV" else "B-V"
    rad_j = jnp.asarray(rad); rad_mean = float(np.mean(rad))
    VMAC_MIN = 0.3

    def vmacro_at(theta, phase):
        """Macroturbulence FWHM [km/s] at the given (model-frame) phases: a Fourier curve, floored at VMAC_MIN."""
        return jnp.broadcast_to(jnp.maximum(fourier_eval(phase, theta[I_VMAC:I_VMAC + n_vmac]), VMAC_MIN), jnp.shape(phase))

    def model_mags(theta, F_nodes):
        teff_e = fourier_eval(ph_model, theta[:n_coef]); idx, w = node_weights(teff_e)
        flux = node_interp(F_nodes, idx, w, jnp.arange(args.n_phases))
        dist_scale = (10.0 / (theta[I_DIST] * PC_PER_KPC)) ** 2
        # radius-curve amplitude scale: projected area of a sphere goes as R^2, so scale the flux per epoch
        area_scale = ((rad_mean + theta[I_RAMP] * (rad_j - rad_mean)) / rad_j) ** 2
        flux = flux * dist_scale * area_scale[:, None] * jnp.power(10.0, -0.4 * meta["r_v"] * theta[I_EBV] * alam_av)[None, :]
        return {b: jax.vmap(lambda fl: band_mag(fl, wl_j, resp[b], zp[b]))(flux) for b in args.bands}, teff_e

    def chi2_phot(theta, F_nodes):
        return chi2_phot_mags(theta, *model_mags(theta, F_nodes))

    def chi2_phot_mags(theta, mags, teff_e):
        wall =jnp.sum(jnp.maximum(T_nodes[0] - teff_e, 0.0) ** 2 + jnp.maximum(teff_e - T_nodes[-1], 0.0) ** 2) / 50.0 ** 2
        tot = wall; parts = {}; off_of = {}
        for key, o in sorted(obs_j.items(), key=lambda kv: kv[0] == "B"):     # V before B: B reuses V's zero points
            curve = (mags[key.split("-")[0]] - mags[key.split("-")[1]]) if "-" in key else mags[key]
            pred = periodic_interp((o["phase"] - theta[I_DPHI]) % 1.0, ph_model, curve)
            r = o["mag"] - pred
            # per-source zero points apply to MAGNITUDES only: an offset per source in a COLOUR would absorb
            # E(B-V) entirely; 'zero-mean' keeps the mean level so that distance x reddening stays measurable
            if zp_mode != "none" and "-" not in key:
                if key == "B" and "V" in off_of:
                    # same per-source offset as V, so that B - V carries no per-source zero point (as in colour mode)
                    offV, srcsV = off_of["V"]
                    off = jnp.asarray([offV[srcsV.index(s)] for s in o["srcs"]])
                else:
                    w = 1.0 / o["sig"] ** 2
                    off = jax.ops.segment_sum(r * w, o["src"], num_segments=o["n_src"]) / jax.ops.segment_sum(w, o["src"], num_segments=o["n_src"])
                    if zp_mode == "zero-mean":
                        off = off - jnp.mean(off)
                off_of[key] = (off, o["srcs"])
                r = r - off[o["src"]]
            if args.free_bsys and key == BSYS_KEY:
                # extra scatter s: sigma^2 -> sigma^2 + s^2, with the Gaussian normalisation (zero at s = 0) so s cannot grow for free
                v_eff = o["sig"] ** 2 + theta[I_BSYS] ** 2
                c2 = r ** 2 / v_eff + jnp.log(v_eff / o["sig"] ** 2)
            else:
                c2 = (r / o["sig"]) ** 2
            c = jnp.sum(c2 * o["infit"]); parts[key] = c; tot = tot + c
            if args.holdout_source or args.predict_bands:
                parts[key + " holdout"] = jnp.sum(c2 * (1.0 - o["infit"]))
        if args.free_distance:
            plx, plx_err = args.plx_prior
            tot = tot + ((1.0 / theta[I_DIST] - plx) / plx_err) ** 2
        return tot, parts

    # ------------------------------------------------------------------ spectroscopy pieces
    parts_sp = [los_parts(m) for m in snaps_sp]
    vmax = max(float(jnp.max(jnp.abs(r + p))) for _, _, r, p in parts_sp)
    hw = {wi: int(math.ceil(math.log10(1.0 + (1.5 * vmax + 40.0) / C_KMS) / (dlog / args.oversample))) + 1 for wi in keep}
    print(f"max |v_los| {vmax:.1f} km/s; kernel half widths {sorted(set(hw.values()))} fine px", flush=True)

    def build_spec_kernels(dv_e, vscale):
        """dv_e: per-epoch LOS velocity offset (n_ep,), km/s."""
        return {wi: [kernel_from_parts(parts_sp[e], lw_obs[wi], args.n_mu, args.oversample, hw[wi], dv_e[e], vscale)
                     for e in range(len(snaps_sp))] for wi in keep}

    def spec_nodes(ab, kernels):
        out = {}
        for wi in keep:
            def one(T, kk=kernels[wi]):
                row = set_ab(base_row.at[iT].set(T), ab)
                f0 = kernel_flux_multi(emu.intensity, kk, row, wavelength_chunk_size=args.chunk)          # (n_ep, n_px, 2)
                fg = kernel_flux_multi(emu.intensity, kk, row.at[iG].add(DG), wavelength_chunk_size=args.chunk)
                return f0 + (fg - f0) * (logg_sp / DG)[:, None, None]
            out[wi] = jax.lax.map(one, T_nodes)                                                        # (n_T, n_ep, n_px, 2)
        return out

    VMAC_MAX = 40.0
    half_br = int(math.ceil(4.0 * math.hypot(C_KMS / R_HARPSN, VMAC_MAX) / (C_KMS * math.log(10.0) * dlog) / 2.3548))

    def broaden(y, vmacro):                     # y (..., n): Gaussian in log-lambda pixels, edge padded; one width
        sigma_px = jnp.hypot(C_KMS / R_HARPSN, vmacro) / (C_KMS * math.log(10.0) * dlog) / 2.3548
        k = gauss_taps(sigma_px, half_br)
        yp = jnp.pad(y, [(0, 0)] * (y.ndim - 1) + [(half_br, half_br)], mode="edge")
        return conv_valid(yp, k)

    def broaden_epochs(sp, vmac_e):             # sp (n_ep, n_px, 2), vmac_e (n_ep,): a different width per epoch
        sigma_px = jnp.hypot(C_KMS / R_HARPSN, vmac_e) / (C_KMS * math.log(10.0) * dlog) / 2.3548
        taps = jax.vmap(lambda s_: gauss_taps(s_, half_br))(sigma_px)                        # (n_ep, k)
        def one(row, k):                                                                     # row (n_px, 2)
            yp = jnp.pad(row.T, [(0, 0), (half_br, half_br)], mode="edge")
            return conv_valid(yp, k)                                                         # (2, n_px)
        return jax.vmap(one)(sp, taps)                                                       # (n_ep, 2, n_px)

    def continuum_fix(m, ob, gd, x):
        if args.no_continuum_fix:
            return m
        w = jnp.where(gd, 1.0, 0.0)
        A = jnp.stack([m, m * x], 1) * w[:, None]
        coef = jnp.linalg.solve(A.T @ A + 1e-12 * jnp.eye(2), A.T @ (ob * w))
        return m * (coef[0] + coef[1] * x)

    e_sp = jnp.arange(len(snaps_sp)); ph_spec = jnp.asarray(phases_spec)

    def model_spectra(theta, S_nodes, wi):
        """Normalised, broadened model spectra (n_ep, n_px) for one window (before the continuum fix)."""
        teff_e = fourier_eval(ph_spec - theta[I_DPHI], theta[:n_coef]); idx, w = node_weights(teff_e)
        sp = node_interp(S_nodes[wi], idx, w, e_sp)                                           # (n_ep, n_px, 2)
        br = broaden_epochs(sp, vmacro_at(theta, ph_spec - theta[I_DPHI]))                    # (n_ep, 2, n_px)
        return 1.0 - theta[I_DEPTH] * (1.0 - br[:, 0] / br[:, 1])

    # rest-frame line-correction template: c_e(lw) = 1 + t_w(lw - shift_e), shift_e = the epoch's model line velocity
    TAU_T = 0.05
    n_px = {wi: int(lw_obs[wi].size) for wi in keep}
    t_off = {}; _o = 0
    for wi in keep:
        t_off[wi] = _o; _o += n_px[wi]
    n_tmpl = _o if args.line_template else 0
    tmpl_shift = {"val": jnp.zeros(len(snaps_sp))}          # log10 shift per epoch, set after the velocity measurement

    def template_of(x_t, wi):
        return x_t[t_off[wi]:t_off[wi] + n_px[wi]] * TAU_T

    def apply_template(mod, x_t, wi):
        if not args.line_template:
            return mod
        t_w = template_of(x_t, wi); lw = lw_obs[wi]
        return mod * (1.0 + jax.vmap(lambda sh: jnp.interp(lw - sh, lw, t_w))(tmpl_shift["val"]))

    def rho(r):
        if args.robust_delta > 0:
            d = args.robust_delta
            return d * d * (jnp.sqrt(1.0 + (r / d) ** 2) - 1.0) * 2.0      # ~ r^2 for |r| << d, linear beyond
        return r ** 2

    ep_in = np.ones(len(snaps_sp)); 
    if args.holdout_epochs:
        ep_in[::args.holdout_epochs] = 0.0
        print(f"holding out {int((ep_in == 0).sum())} of {ep_in.size} spectroscopic epochs (every {args.holdout_epochs}th) from the fit", flush=True)
    ep_in_j = jnp.asarray(ep_in)

    def chi2_spec(theta, S_nodes, active, x_t=None):
        tot = 0.0; parts = {}
        for wi in keep:
            mod = model_spectra(theta, S_nodes, wi)
            mod = apply_template(mod, x_t, wi)
            mod = jax.vmap(lambda m, o, g: continuum_fix(m, o, g, xw[wi]))(mod, obs_sp[wi], good_sp[wi])
            r = jnp.where(good_sp[wi], (obs_sp[wi] - mod) / sig_sp[wi], 0.0)
            per_ep = jnp.sum(rho(r), axis=1)
            c = jnp.sum(per_ep * ep_in_j); parts[wi] = c; tot = tot + active[wi] * c
            if args.holdout_epochs:
                parts[-(wi + 1)] = jnp.sum(per_ep * (1.0 - ep_in_j))          # negative key = held-out epochs of window wi (pytree keys must sort)
        if args.line_template:
            tot = tot + args.template_ridge * jnp.sum((x_t[:n_tmpl] * TAU_T / TAU_T) ** 2)
        return tot, parts

    # ------------------------------------------------------------------ parameters
    theta0 = np.concatenate([coef0, [args.ebv0, feh0, 0.0, args.vmacro0], np.zeros(2 * KV), [1.0, d0_kpc, 1.0, 0.01, 1.0], list(args.cno0), [1.0]])
    S = np.concatenate([[100.0], np.full(2 * K, 50.0), [0.02, 0.1, 0.002, 1.0], np.full(2 * KV, 1.0), [0.02, 0.01, 0.02, 0.005, 0.02], [0.1, 0.1, 0.1], [0.05]])
    lo = np.concatenate([[T_lo], coef0[1:] - 400.0, [0.0, -1.0, -0.1, 0.5], np.full(2 * KV, -20.0), [0.7, 0.15, 0.6, 0.0, 0.5], [-1.0, -1.0, -1.0], [0.2]])
    hi = np.concatenate([[T_hi], coef0[1:] + 400.0, [0.4, 0.5, 0.1, VMAC_MAX], np.full(2 * KV, 20.0), [1.4, 0.5, 1.4, 0.15, 2.0], [1.0, 1.0, 1.0], [2.0]])
    hi[I_VROT] = args.vrot_scale_max
    if args.vmacro_curve:
        tab = np.loadtxt(args.vmacro_curve)
        fwhm_tab = args.vmacro_curve_scale * tab[:, 4]
        if args.vmacro_instr_r > 0:
            fi = C_KMS / args.vmacro_instr_r
            if np.any(fwhm_tab <= fi): raise SystemExit(f"--vmacro-instr-r: table FWHM below the instrumental {fi:.2f} km/s")
            print(f"vmacro curve: removing the R = {args.vmacro_instr_r:.0f} instrumental FWHM {fi:.2f} km/s in quadrature "
                  f"(table FWHM {fwhm_tab.min():.1f}-{fwhm_tab.max():.1f} -> {np.sqrt(fwhm_tab ** 2 - fi ** 2).min():.1f}-"
                  f"{np.sqrt(fwhm_tab ** 2 - fi ** 2).max():.1f} km/s)", flush=True)
            fwhm_tab = np.sqrt(fwhm_tab ** 2 - fi ** 2)
        c_vm, _ = fourier_fit(tab[:, 0], fwhm_tab, KV)
        theta0[I_VMAC:I_VMAC + n_vmac] = c_vm
        hi[I_VMAC] = max(hi[I_VMAC], c_vm[0] + 10.0)
        g = np.asarray(fourier_eval(jnp.linspace(0, 1, 200), jnp.asarray(c_vm)))
        print(f"starting vmacro(phase) from {args.vmacro_curve} x {args.vmacro_curve_scale}: FWHM {g.min():.1f}-{g.max():.1f} km/s, "
              f"order {KV}, max at phase {np.linspace(0, 1, 200)[np.argmax(g)]:.2f}", flush=True)
    fixed = np.zeros(theta0.size, bool)
    fixed[I_DIST] = not args.free_distance
    fixed[I_RAMP] = not args.free_ramp
    fixed[I_BSYS] = not args.free_bsys
    fixed[I_DEPTH] = not args.free_depth_scale
    fixed[I_CNO:I_CNO + 3] = not args.free_cno
    fixed[I_DPHI] = args.fix_dphi
    fixed[I_EBV] = args.fix_ebv; fixed[I_FEH] = not args.free_feh
    if args.fix_teff:
        fixed[:n_coef] = True
    elif args.fix_teff_shape:
        fixed[1:n_coef] = True
    fixed[I_VMAC:I_VMAC + n_vmac] = args.fix_vmacro or args.phot_only
    fixed[I_VSC] = not args.free_vscale
    if args.free_vrot_scale and not args.exact_loss:
        raise SystemExit("--free-vrot-scale needs --exact-loss (the kernels carry the rotation)")
    fixed[I_VROT] = not args.free_vrot_scale
    if args.spec_only:
        fixed[I_EBV] = True
    free_idx = np.where(~fixed)[0]
    labels = (["T0"] + [f"{'a' if j % 2 == 0 else 'b'}{j // 2 + 1}" for j in range(2 * K)] + ["E(B-V)", "[Fe/H]", "dphi", "vmacro"]
              + [f"{'vc' if j % 2 == 0 else 'vs'}{j // 2 + 1}" for j in range(2 * KV)] + ["vscale", "d_kpc", "ramp", "s_" + BSYS_KEY, "depth_scale", "[C/Fe]", "[N/Fe]", "[O/Fe]", "vrot_scale"])
    if args.init_from:
        prev = pickle.load(open(args.init_from, "rb")); pl = list(prev["labels"]); n_init = 0
        for i, name in enumerate(labels):
            if name in pl:
                theta0[i] = float(prev["theta"][pl.index(name)]); n_init += 1
        lo[1:n_coef] = np.minimum(lo[1:n_coef], theta0[1:n_coef] - 200.0); hi[1:n_coef] = np.maximum(hi[1:n_coef], theta0[1:n_coef] + 200.0)
        print(f"initialised {n_init} of {len(labels)} parameters from {args.init_from}: T0 {theta0[0]:.0f}, E(B-V) {theta0[I_EBV]:.3f}, dphi {theta0[I_DPHI]:+.4f}", flush=True)
    if args.vscale0 is not None:
        theta0[I_VSC] = args.vscale0
        print(f"vscale set to {args.vscale0:.4f} (p = {meta['pfactor'] * args.vscale0:.4f})", flush=True)
    print("free:", [labels[i] for i in free_idx], flush=True)
    theta0_j, S_j, fi_j = jnp.asarray(theta0), jnp.asarray(S), jnp.asarray(free_idx)

    n_free = free_idx.size
    print(f"parameter vector: {n_free} physical + {n_tmpl} template values" + (" (holdout windows: %s)" % args.holdout_windows if args.holdout_windows else ""), flush=True)

    def unpack(x):
        return theta0_j.at[fi_j].set(theta0_j[fi_j] + x[:n_free] * S_j[fi_j])

    def tmpl_of(x):
        return x[n_free:] if args.line_template else None

    # ------------------------------------------------------------------ exact per-element model (--exact-loss)
    if args.exact_loss:
        from spice.spectrum.spectrum import simulate_observed_flux
        if args.line_template:
            raise SystemExit("--exact-loss: no --line-template")
        st = lambda ms, f: jnp.stack([jnp.asarray(f(m)) for m in ms])
        EX_SP = (jnp.stack([q[0] for q in parts_sp]), jnp.stack([q[1] for q in parts_sp]), jnp.stack([q[2] for q in parts_sp]),
                 jnp.stack([q[3] for q in parts_sp]), st(snaps_sp, lambda m: m.parameters))                  # mus, areas, rot, puls, params
        EX_PH = (st(snaps_ph, lambda m: m.visible_cast_areas), st(snaps_ph, lambda m: m.mus), st(snaps_ph, lambda m: m.los_velocities), st(snaps_ph, lambda m: m.parameters))
        def phot_grid(n):
            lw_ = jnp.linspace(math.log10(args.wl_range[0]), math.log10(args.wl_range[1]), n); wl_ = 10.0 ** lw_
            return (lw_, wl_, {b: jnp.asarray(getattr(spfilter, BANDS[b][0])().filter_responses_for_wavelengths(wl_)) for b in args.bands},
                    jnp.asarray(alambda_over_av(np.asarray(wl_), meta["r_v"], args.ext_table, args.ext_law)))
        PG = {"g": phot_grid(args.exact_phot_wl)}         # the grid model_mags_exact uses; swapped once for the 2x check
        DEVS = list(jax.devices())[:args.gpus] if args.gpus > 1 else None
        DV = {"e": jnp.zeros(len(snaps_sp))}          # LOS offsets of the exact spectra; the CCF velocities once measured

        # swapped to the cheap model (and its cond/checkpoint-free element scan) while the cheap loss is traced (--cheap-block-pca)
        EMU = {"f": emu.intensity, "chunk": args.exact_chunk, "sim": simulate_observed_flux}

        def exact_flux(a, mu, v, p, lw_, spec=False):
            f = EMU.get("fs", EMU["f"]) if spec else EMU["f"]
            return EMU["sim"](f, ElementMesh(a, mu, v, p), lw_, chunk_size=EMU["chunk"])

        def exact_model_spectra(theta, wis):
            """Normalised, broadened exact spectra {wi: (n_ep, n_px)} of the windows wis (before the continuum fix), one synthesis per epoch."""
            lw_cat = jnp.concatenate([lw_obs[wi] for wi in wis])
            teff_e = fourier_eval(ph_spec - theta[I_DPHI], theta[:n_coef])
            def one(th, xs):
                mu, a, rot, puls, p, T, lg, dv = xs
                p = set_ab(p.at[:, iT].set(T), ab_of(th)).at[:, iG].set(logg_mean + lg)
                return exact_flux(a, mu, th[I_VROT] * rot + th[I_VSC] * puls + dv, p, lw_cat, spec=True)
            raw = map_epochs(one, theta, (*EX_SP, teff_e, logg_sp, DV["e"]), DEVS)                          # (n_ep, n_cat, 2)
            vmac_e = vmacro_at(theta, ph_spec - theta[I_DPHI]); out = {}; o = 0
            for wi in wis:
                br = broaden_epochs(raw[:, o:o + n_px[wi]], vmac_e); o += n_px[wi]
                out[wi] = 1.0 - theta[I_DEPTH] * (1.0 - br[:, 0] / br[:, 1])
            return out

        def model_mags_exact(theta):
            lw_x, wl_x, resp_x, alam_x = PG["g"]
            teff_e = fourier_eval(ph_model, theta[:n_coef])
            def one(th, xs):
                a, mu, v, p, T, lg, vm = xs
                return exact_flux(a, mu, v, set_ab(p.at[:, iT].set(T), ab_of(th)).at[:, iG].set(logg_mean + lg).at[:, iVM].set(vm), lw_x)[:, 0]
            flux = map_epochs(one, theta, (*EX_PH, teff_e, logg_ph, VM_PH), DEVS)
            dist_scale = (10.0 / (theta[I_DIST] * PC_PER_KPC)) ** 2
            area_scale = ((rad_mean + theta[I_RAMP] * (rad_j - rad_mean)) / rad_j) ** 2
            flux = flux * dist_scale * area_scale[:, None] * jnp.power(10.0, -0.4 * meta["r_v"] * theta[I_EBV] * alam_x)[None, :]
            return {b: jax.vmap(lambda fl: band_mag(fl, wl_x, resp_x[b], zp[b]))(flux) for b in args.bands}, teff_e

        def chi2_spec_exact(theta, active):
            tot = 0.0; parts = {}; mods = exact_model_spectra(theta, tuple(keep))
            for wi in keep:
                mod = jax.vmap(lambda m, o, g: continuum_fix(m, o, g, xw[wi]))(mods[wi], obs_sp[wi], good_sp[wi])
                r = jnp.where(good_sp[wi], (obs_sp[wi] - mod) / sig_sp[wi], 0.0)
                c = jnp.sum(jnp.sum(rho(r), axis=1) * ep_in_j); parts[wi] = c; tot = tot + active[wi] * c
            return tot, parts

        t = time.time(); m_ex = jax.jit(lambda th: model_mags_exact(th)[0])(theta0_j)
        m_k, _ = jax.jit(lambda th: model_mags(th, jax.jit(phot_nodes)(ab0)))(theta0_j)
        g1 = PG["g"]; PG["g"] = phot_grid(2 * args.exact_phot_wl); m_2x = jax.jit(lambda th: model_mags_exact(th)[0])(theta0_j); PG["g"] = g1
        dd = lambda a, b_: f"{1e3 * float(jnp.mean(a - b_)):+.2f}+-{1e3 * float(jnp.std(a - b_)):.2f}"
        print(f"exact photometry ({args.exact_phot_wl} wl) at the start: vs 2x sampling " + " ".join(f"{b} {dd(m_ex[b], m_2x[b])}" for b in args.bands)
              + f" mmag; vs kernel ({args.n_wl} wl) " + " ".join(f"{b} {dd(m_ex[b], m_k[b])}" for b in args.bands) + f" mmag ({time.time() - t:.0f}s incl. compile)", flush=True)
        # everything downstream (CCF, loss, report, example spectra, result) now sees the exact model
        model_spectra = lambda theta, S_nodes, wi: exact_model_spectra(theta, (wi,))[wi]
        model_mags = lambda theta, F_nodes: model_mags_exact(theta)
        print(f"EXACT LOSS ({len(DEVS) if DEVS else 1} device(s)): {len(snaps_sp)} epochs x {len(keep)} windows ({sum(n_px[wi] for wi in keep)} px), {len(snaps_ph)} photometric phases x {args.exact_phot_wl} wl; "
              f"{int(EX_SP[0].shape[1])} elements per mesh", flush=True)

    # ------------------------------------------------------------------ systemic velocity per season (CCF vs start model)
    t = time.time()
    F0 = None if args.exact_loss else jax.jit(phot_nodes)(ab0); jax.block_until_ready(F0)
    print(f"photometric node fluxes: {time.time() - t:.0f}s", flush=True)
    dv_season = np.zeros(len(seasons))
    K0 = build_spec_kernels(np.zeros(len(snaps_sp)), 1.0)
    wi_mg = next((wi for wi in keep if abs(all_windows[wi][0] - 5160.0) < 1), keep[0])
    t = time.time()
    S0 = None if args.exact_loss else jax.jit(lambda ab: spec_nodes(ab, K0))(ab0)
    if S0 is not None:
        jax.block_until_ready(S0[keep[0]])
        print(f"spectroscopic node spectra ({len(keep)} windows x {len(snaps_sp)} epochs x {args.n_teff} Teff x {args.n_mu} mu): {time.time() - t:.0f}s", flush=True)

    def ccf_epochs(theta, S_nodes, wi):
        mod = np.asarray(jax.jit(lambda th: model_spectra(th, S_nodes, wi))(theta)); ob = np.asarray(obs_sp[wi]); gd = np.asarray(good_sp[wi])
        wl_w = 10.0 ** np.asarray(lw_obs[wi]); dvs = []
        for e in range(mod.shape[0]):
            o = np.where(gd[e], ob[e], np.nan)
            dv, cc, rail = velocity_offset(wl_w, mod[e], o, max_shift_kms=60.0)
            dvs.append(dv if not rail else np.nan)
        return np.asarray(dvs)

    dv0 = ccf_epochs(theta0_j, S0, wi_mg)
    good_dv = np.isfinite(dv0)
    if args.dv_sys is not None:
        assert len(args.dv_sys) == len(seasons), f"--dv-sys needs {len(seasons)} values (seasons {seasons})"
        dv_season = np.asarray(args.dv_sys, float)
    for s, yr in enumerate(seasons):
        m = (sidx == s) & np.isfinite(dv0)
        med = np.median(dv0[m]); m &= np.abs(dv0 - med) < 10.0          # spurious CCF peaks (one epoch sat at +45 km/s)
        good_dv &= ~((sidx == s) & ~m)
        if args.dv_sys is None:
            dv_season[s] = np.median(dv0[m])
        print(f"  season {yr}: systemic velocity from the CCF of {m.sum()} epochs: {dv_season[s]:+.2f} km/s "
              f"(scatter {np.std(dv0[m] - dv_season[s]):.2f}, {int(((sidx == s) & ~m).sum())} outliers dropped); "
              f"a phase-dependent residual here means a pulsation velocity / p-factor mismatch", flush=True)
    # velocity residual vs LOS pulsation velocity: slope != 0 <=> the model's pulsation velocity amplitude is off
    m = good_dv
    b, a = np.polyfit(vpuls_model[m], dv0[m] - dv_season[sidx[m]], 1)
    print(f"  residual dv vs model <v_los,puls>: slope {b:+.3f} (0 = amplitude right; -0.1 = model 10% too large), rms {np.std(dv0[m] - dv_season[sidx[m]] - (a + b * vpuls_model[m])):.2f} km/s", flush=True)
    resid_dv = np.where(good_dv, dv0 - dv_season[sidx], 0.0)
    steep = np.abs(resid_dv) > 1.0
    print(f"  epochs whose CCF velocity differs from the SPIPS pulsation curve by >1 km/s: {int(steep.sum())} of {steep.size} "
          f"(phases {np.round(np.sort(phases_spec[steep]), 3).tolist()[:12]}{' ...' if steep.sum() > 12 else ''})", flush=True)
    dv_epoch = dv_season[sidx] + (resid_dv if args.dv_mode == "epoch" else 0.0)
    print(f"  dv mode '{args.dv_mode}': kernels shifted by {'per-epoch CCF velocities' if args.dv_mode == 'epoch' else 'the season medians only'}", flush=True)
    if args.dv_from:
        prev = pickle.load(open(args.dv_from, "rb"))
        assert prev["dv_epoch"].size == dv_epoch.size, f"--dv-from has {prev['dv_epoch'].size} epochs, this run {dv_epoch.size}"
        print(f"  per-epoch velocities restored from {args.dv_from}: rms difference to the re-measured ones "
              f"{np.nanstd(prev['dv_epoch'] - dv_epoch):.3f} km/s", flush=True)
        dv_epoch = np.asarray(prev["dv_epoch"], float); dv_season = np.asarray(prev["dv_season"], float)
    K_fixed = build_spec_kernels(dv_epoch, 1.0)
    dv_e_j = jnp.asarray(dv_epoch)
    if args.exact_loss:
        DV["e"] = dv_e_j
    if args.line_mask:
        # keep only pixels near listed rest wavelengths, each shifted by its epoch's line velocity (the template's
        # CCF shifts agree with dv_epoch + <v_los,puls> to ~1 km/s, well inside the half-width)
        lines_rest = np.loadtxt(args.line_mask, comments="#", ndmin=1, usecols=0)      # first column; others may be text (element, flags)
        v_ep = dv_epoch + vpuls_model; N_sp = 0
        for wi in keep:
            wl_rest = 10.0 ** np.asarray(lw_obs[wi])[None, :] / (1.0 + v_ep[:, None] / C_KMS)
            near = np.zeros(wl_rest.shape, bool)
            for l0 in lines_rest[(lines_rest > wl_rest.min() - 1.0) & (lines_rest < wl_rest.max() + 1.0)]:
                near |= np.abs(wl_rest / l0 - 1.0) * C_KMS < args.line_mask_hw
            g = np.asarray(good_sp[wi]) & near
            good_sp[wi] = jnp.asarray(g); N_sp += int(g.sum())
            print(f"  line mask {all_windows[wi][0]:.0f}: {int(near[0].sum())} of {near.shape[1]} pixels kept at epoch 0", flush=True)
        print(f"line mask {args.line_mask}: {lines_rest.size} lines, +-{args.line_mask_hw} km/s, {N_sp} spectral pixels in the fit", flush=True)

    emulator_in_graph = args.free_feh or args.free_cno or args.free_vscale
    if args.exact_loss:
        def nodes_for(theta):
            return None, None
    elif emulator_in_graph:
        def nodes_for(theta):
            kk = build_spec_kernels(dv_e_j, theta[I_VSC]) if args.free_vscale else K_fixed
            return phot_nodes(ab_of(theta)) if (args.free_feh or args.free_cno) else F0, spec_nodes(ab_of(theta), kk)
    else:
        S_fixed = jax.jit(lambda ab: spec_nodes(ab, K_fixed))(ab0); jax.block_until_ready(S_fixed[keep[0]])
        def nodes_for(theta):
            return F0, S_fixed

    active_all = {wi: (0.0 if any(abs(all_windows[wi][0] - s) < 1 for s in args.holdout_windows) else 1.0) for wi in keep}

    def total(x, active):
        th = unpack(x); F, Sn = nodes_for(th)
        c_ph, p_ph = (0.0, {}) if args.spec_only else chi2_phot(th, F)
        if args.exact_loss:
            c_sp, p_sp = (0.0, {}) if args.phot_only else chi2_spec_exact(th, active)
        else:
            c_sp, p_sp = (0.0, {}) if args.phot_only else chi2_spec(th, Sn, active, tmpl_of(x))
        return c_ph + args.spec_weight * c_sp, (c_ph, c_sp, p_ph, p_sp)

    def make_vg():
        return jax.jit(jax.value_and_grad(total, has_aux=True))
    vg = make_vg()
    x0_all = np.zeros(n_free + n_tmpl)
    if args.line_template and args.template_from:
        prev = pickle.load(open(args.template_from, "rb"))["template"]
        for wi in keep:
            t_prev = prev.get(all_windows[wi])
            if t_prev is not None and t_prev.size == n_px[wi]:
                x0_all[n_free + t_off[wi]:n_free + t_off[wi] + n_px[wi]] = np.asarray(t_prev) / TAU_T
            else:
                print(f"WARNING: no matching template for window {all_windows[wi]} in {args.template_from}", flush=True)
        print(f"template initialised from {args.template_from} (rms {np.std(x0_all[n_free:]) * TAU_T:.4f})" + (", held fixed" if args.fix_template else ""), flush=True)

    # rest-frame shifts for the template: CCF of each epoch's model against a zero-velocity model of the same Teff
    if args.line_template:
        K_rest = {wi_mg: [kernel_from_parts((parts_sp[0][0], parts_sp[0][1], 0.0 * parts_sp[0][2], 0.0 * parts_sp[0][3]),
                                            lw_obs[wi_mg], args.n_mu, args.oversample, hw[wi_mg], 0.0, 0.0)]}
        def rest_nodes(ab):
            def one(T):
                row = set_ab(base_row.at[iT].set(T), ab)
                return kernel_flux_multi(emu.intensity, K_rest[wi_mg], row, wavelength_chunk_size=args.chunk)   # (1, n_px, 2)
            return jax.lax.map(one, T_nodes)
        R0 = jax.jit(rest_nodes)(ab0)
        def rest_spectra(theta):
            teff_e = fourier_eval(ph_spec - theta[I_DPHI], theta[:n_coef]); idx, w = node_weights(teff_e)
            sp = node_interp(R0, idx, w, jnp.zeros_like(e_sp))
            br = broaden_epochs(sp, vmacro_at(theta, ph_spec - theta[I_DPHI]))
            return br[:, 0] / br[:, 1]
        _, S_tmp = nodes_for(theta0_j)
        mod_e = np.asarray(jax.jit(lambda th: model_spectra(th, S_tmp, wi_mg))(theta0_j))
        rest_e = np.asarray(jax.jit(rest_spectra)(theta0_j))
        wl_mg = 10.0 ** np.asarray(lw_obs[wi_mg]); v_line = np.zeros(len(snaps_sp))
        for e in range(len(snaps_sp)):
            dv, _, rail = velocity_offset(wl_mg, rest_e[e], mod_e[e], max_shift_kms=80.0)
            v_line[e] = dv if not rail else dv_epoch[e] + vpuls_model[e]
        if args.dv_from:
            # a replay: the shifts were measured against THAT run's starting model, and the CCF step is 1 km/s,
            # so re-measuring against a different starting model moves them by a step for many epochs
            prev_shift = pickle.load(open(args.dv_from, "rb")).get("template_shift_kms")
            if prev_shift is not None and prev_shift.size == v_line.size:
                print(f"template rest-frame shifts restored from {args.dv_from} (rms difference to re-measured "
                      f"{np.std(prev_shift - v_line):.2f} km/s)", flush=True)
                v_line = np.asarray(prev_shift, float)
        tmpl_shift["val"] = jnp.asarray(np.log10(1.0 + v_line / C_KMS))
        print(f"template rest-frame shifts: model line velocities {v_line.min():+.1f}..{v_line.max():+.1f} km/s "
              f"(vs dv_epoch + <v_los,puls>: rms diff {np.std(v_line - dv_epoch - vpuls_model):.2f} km/s)", flush=True)
        vg = make_vg()

    # ------------------------------------------------------------------ exact-synthesis check of the spectroscopic surrogate
    from spice.spectrum.spectrum import simulate_observed_flux
    e0 = len(snaps_sp) // 2; wi0 = keep[0]

    def exact_window(theta, e, wi):
        m = snaps_sp[e]; p = jnp.asarray(m.parameters)
        T_e = fourier_eval(ph_spec[e] - theta[I_DPHI], theta[:n_coef])
        p = set_ab(p.at[:, iT].set(T_e), ab_of(theta)).at[:, iG].set(logg_mean + logg_sp[e])
        los_unit = m.los_vector / jnp.linalg.norm(m.los_vector)
        # los_velocities = -cast_to_los(velocities, los): adding +dv * los_unit to the velocity moves the LOS velocity by +dv
        puls = m.pulsation_velocities * theta[I_VSC] + dv_e_j[e] * los_unit[None, :]
        mm = m._replace(parameters=p, pulsation_velocities=puls)
        # sign check: LOS velocity must have moved by exactly +dv
        d = jnp.asarray(mm.los_velocities) - (parts_sp[e][2] + theta[I_VSC] * parts_sp[e][3])
        s = simulate_observed_flux(emu.intensity, mm, lw_obs[wi], chunk_size=256)
        br = broaden(s.T, vmacro_at(theta, ph_spec[e] - theta[I_DPHI]))
        return 1.0 - theta[I_DEPTH] * (1.0 - br[0] / br[1]), d

    def check(theta, label):
        _, Sn = nodes_for(theta)
        surr = np.asarray(jax.jit(lambda th: model_spectra(th, Sn, wi0))(theta))[e0]
        t = time.time(); ex, d = jax.jit(lambda th: exact_window(th, e0, wi0))(theta); ex = np.asarray(ex); d = np.asarray(d)
        dv_expect = float(dv_epoch[e0])
        print(f"{label}: epoch {e0} (phase {phases_spec[e0]:.3f}) window {all_windows[wi0][0]:.0f}: surrogate vs exact max {np.abs(surr - ex).max():.5f} "
              f"rms {np.sqrt(np.mean((surr - ex) ** 2)):.6f}; LOS shift check {d.mean():+.3f} km/s (expected {dv_expect:+.3f}); exact took {time.time() - t:.0f}s", flush=True)
        return float(np.abs(surr - ex).max())
    if not args.skip_validation and not args.phot_only:
        if check(theta0_j, "start") > args.surrogate_tol:
            print("ABORT: spectroscopic surrogate outside tolerance"); sys.exit(2)

    # ------------------------------------------------------------------ optimisation
    t = time.time(); (v0, aux0), g0 = vg(jnp.asarray(x0_all), active_all); jax.block_until_ready(g0)
    print(f"first value+grad (compile) {time.time() - t:.0f}s; ", end="", flush=True)
    if args.exact_loss:
        print("(no separate warm timing in --exact-loss: one evaluation is the cost of the fit's first step)", flush=True)
    else:
        t = time.time(); jax.block_until_ready(vg(jnp.asarray(x0_all), active_all)[1]); print(f"warm {time.time() - t:.2f}s", flush=True)

    def report(aux, tag):
        c_ph, c_sp, p_ph, p_sp = aux
        def n_of(k):
            base = k.replace(" holdout", ""); inf = np.asarray(obs_j[base]["infit"])
            return max(int((1.0 - inf).sum()) if k.endswith("holdout") else int(inf.sum()), 1)
        s = f"{tag}: chi2_phot/N {float(c_ph) / max(N_ph, 1):.3f} (" + " ".join(f"{k} {float(v) / n_of(k):.2f}" for k, v in p_ph.items()) + ")"
        n_in = max(float(ep_in.sum()), 1.0); n_out = max(float((1 - ep_in).sum()), 1.0)
        s += f"  chi2_spec/N {float(c_sp) / max(N_sp * n_in / ep_in.size, 1):.3f} ("
        s += " ".join(f"{all_windows[wi][0]:.0f}:{float(v) / max(int(np.asarray(good_sp[wi]).sum()) * n_in / ep_in.size, 1):.1f}{'*' if active_all.get(wi, 1.0) == 0.0 else ''}"
                      for wi, v in p_sp.items() if wi >= 0) + ")"
        if args.holdout_epochs:
            hold = [float(v) / max(int(np.asarray(good_sp[-k - 1]).sum()) * n_out / ep_in.size, 1) for k, v in p_sp.items() if k < 0]
            s += f"  HELD-OUT epochs chi2/N {np.mean(hold):.3f} (per window {' '.join(f'{h:.1f}' for h in hold)})"
        print(s, flush=True)
    report(aux0, "start")
    if args.check_grad:
        fval = jax.jit(lambda x: total(x, active_all)[0]); h = 1e-3
        for j in list(range(n_free)) + ([n_free + 5, n_free + n_tmpl // 2] if n_tmpl else []):
            dx = jnp.zeros(n_free + n_tmpl).at[j].set(h)
            fd = (float(fval(dx)) - float(fval(-dx))) / (2 * h)
            name = labels[free_idx[j]] if j < n_free else f"tmpl[{j - n_free}]"
            print(f"  grad check {name:9s}: autodiff {float(g0[j]):+.6e}  finite diff {fd:+.6e}  ratio {float(g0[j]) / fd if fd else float('nan'):.4f}", flush=True)

    def cheap_vg():
        """value_and_grad of the same loss with the block-PCA emulator (and, with --cheap-student, the basis student for the
        spectra) in every exact synthesis (traced once, then cached)."""
        if "vg" not in CHEAP:
            from block_pca_spice import BlockPCAIntensityEmulator, BasisStudentIntensityEmulator, simulate_observed_flux_scan
            emu_c = BlockPCAIntensityEmulator(args.bundle, args.cheap_block_pca, args.cheap_model)
            what = f"block-PCA {args.cheap_model}"
            if args.cheap_student:
                wins = [(float(10.0 ** jnp.min(lw_obs[wi])), float(10.0 ** jnp.max(lw_obs[wi]))) for wi in sorted(lw_obs)]
                emu_s = BasisStudentIntensityEmulator(args.bundle, args.cheap_student, wins, margin_kms=args.cheap_margin)
                vmax = float(jnp.max(jnp.abs(EX_SP[2]))) * 1.5 + float(jnp.max(jnp.abs(EX_SP[3]))) * 1.5 + float(jnp.max(jnp.abs(DV["e"])))
                print(f"basis student {args.cheap_student}: {len(wins)} windows, {emu_s._S} cached rows each, margin {args.cheap_margin} km/s "
                      f"(rough max |v| incl. 1.5x rotation/pulsation scale and season offsets: {vmax:.0f} km/s)", flush=True)
                if vmax > args.cheap_margin:
                    print("  WARNING: velocities may exceed the cached margin (requests clamp to the window edge)", flush=True)
                EMU["fs"] = emu_s.intensity; what += " photometry + basis student spectra"
            vg_c = make_vg(); EMU.update(f=emu_c.intensity, chunk=args.cheap_chunk, sim=simulate_observed_flux_scan)
            try:
                t = time.time(); jax.block_until_ready(vg_c(jnp.asarray(CHEAP["x"]), CHEAP["active"]))
                print(f"cheap ({what}) loss compiled and evaluated in {time.time() - t:.0f}s", flush=True)
            finally:
                EMU.update(f=emu.intensity, chunk=args.exact_chunk, sim=simulate_observed_flux); EMU.pop("fs", None)
            CHEAP["vg"] = vg_c
        return CHEAP["vg"]
    CHEAP = {}

    def multi_fidelity(f_exact, x0, active, bounds, f0):
        """Trust-region loop: L-BFGS-B on cheap(x) + [exact - cheap](x_k) + [grad exact - grad cheap](x_k).(x - x_k) inside
        |x - x_k| <= radius; accept a step only if the exact (TPayne) chi2 drops. Values are chi2 / f0 as in fit()."""
        CHEAP.update(x=np.asarray(x0, float), active=active); vg_c = cheap_vg()
        def f_cheap(x):
            (v, _), g = vg_c(jnp.asarray(x), active); return float(v) / f0, np.asarray(g, float) / f0
        lo_b, hi_b = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        x = np.asarray(x0, float); fe, ge = f_exact(x); rad = args.mf_radius; k = 0; msg = f"outer iteration limit ({args.mf_outer})"
        vc0, _ = f_cheap(x)
        print(f"multi-fidelity start: chi2 TPayne {fe * f0:.6g}, block-PCA {vc0 * f0:.6g} (offset {(fe - vc0) * f0:+.4g})", flush=True)
        for k in range(1, args.mf_outer + 1):
            t = time.time(); vc, gc = f_cheap(x); a, b = fe - vc, ge - gc
            def f_sur(y, a=a, b=b, xk=x):
                v, g = f_cheap(y); return v + a + float(b @ (y - xk)), g + b
            box = list(zip(np.maximum(lo_b, x - rad), np.minimum(hi_b, x + rad)))
            rr = minimize(f_sur, x, jac=True, method="L-BFGS-B", bounds=box, options=dict(maxiter=args.mf_inner, ftol=1e-13, gtol=1e-10, maxcor=20))
            pred = fe - float(rr.fun)
            if pred <= 1e-12 * max(fe, 1e-30):
                msg = "converged: the corrected cheap model predicts no further decrease"; break
            fn, gn = f_exact(rr.x); ratio = (fe - fn) / pred; step = float(np.max(np.abs(rr.x - x)))
            print(f"  MF outer {k}: chi2 {fe * f0:.7g} -> {fn * f0:.7g} (predicted {(fe - pred) * f0:.7g}, ratio {ratio:.2f}), max step {step:.3g}, "
                  f"radius {rad:.3g}, {rr.nit} cheap it / {rr.nfev} cheap evals, {time.time() - t:.0f}s", flush=True)
            if ratio < 0.25:
                rad = 0.25 * min(rad, step)
            elif ratio > 0.75 and step > 0.9 * rad:
                rad = 2.0 * rad
            if fn < fe:
                x, fe, ge = np.asarray(rr.x, float), fn, gn
            if rad < args.mf_min_radius:
                msg = f"trust radius below {args.mf_min_radius}"; break
        return SimpleNamespace(x=x, nit=k, message=f"multi-fidelity: {msg}")

    def fit(x0, active, maxiter, label):
        hist = []
        # L-BFGS-B's stopping test is RELATIVE to f: with chi2 ~ 1e7 the default ftol quits after ~10 steps,
        # so optimise f / f(x0) and demand a 1e-11 relative change.
        f0 = max(float(vg(jnp.asarray(x0), active)[0][0]), 1.0)
        prog = (args.out[:-4] if args.out.endswith(".pkl") else args.out) + "_progress.pkl"
        def f(x):
            te = time.time(); (v, aux_), g = vg(jnp.asarray(x), active)
            if args.exact_loss:
                th_ = np.asarray(unpack(jnp.asarray(x)))
                if not hist or float(v) < min(hist):      # best point so far: an --init-from warm start if the job dies
                    with open(prog, "wb") as fh:
                        pickle.dump(dict(theta=th_, labels=list(labels), chi2=float(v), n_eval=len(hist) + 1, label=label), fh, protocol=4)
                print(f"  eval {len(hist) + 1}: chi2 {float(v):.6g} (phot {float(aux_[0]):.1f} spec {float(aux_[1]):.1f}) {time.time() - te:.0f}s | "
                      + " ".join(f"{labels[i]} {th_[i]:.4g}" for i in free_idx), flush=True)
            hist.append(float(v)); return float(v) / f0, np.asarray(g, float) / f0
        bounds = [((lo[i] - theta0[i]) / S[i], (hi[i] - theta0[i]) / S[i]) for i in free_idx] + (
            [(float(v), float(v)) for v in np.asarray(x0)[n_free:]] if args.fix_template else [(-10.0, 10.0)] * n_tmpl)
        t = time.time()
        if maxiter == 0:
            # evaluation only (a replay of a stored solution): L-BFGS-B still takes one line-search step at maxiter=0
            r = SimpleNamespace(x=np.asarray(x0, float), nit=0, message="no optimisation (maxiter 0)")
        elif args.cheap_block_pca:
            r = multi_fidelity(f, x0, active, bounds, f0)
        else:
            r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter, ftol=1e-11, gtol=1e-8, maxfun=4 * maxiter, maxcor=20))
        (_, aux), _ = vg(jnp.asarray(r.x), active)
        th = np.asarray(unpack(jnp.asarray(r.x)))
        print(f"{label}: {r.nit} it, {len(hist)} eval, {time.time() - t:.0f}s, {r.message[:45]}", flush=True)
        report(aux, "  end")
        vm = np.asarray(vmacro_at(jnp.asarray(th), jnp.linspace(0, 1, 200)))
        print(f"  T0 {th[0]:.0f}  E(B-V) {th[I_EBV]:.4f}  [Fe/H] {th[I_FEH]:+.3f}"
              + (f"  [C/Fe] {th[I_CNO]:+.3f} [N/Fe] {th[I_CNO + 1]:+.3f} [O/Fe] {th[I_CNO + 2]:+.3f}" if args.free_cno or any(args.cno0) else "")
              + f"  dphi {th[I_DPHI]:+.4f}  vmacro {th[I_VMAC]:.2f} km/s"
              + (f" (curve {vm.min():.1f}-{vm.max():.1f}, max at phase {np.linspace(0, 1, 200)[np.argmax(vm)]:.2f})" if KV else "")
              + f"  vscale {th[I_VSC]:.4f} (p_SPICE = {meta['pfactor'] * th[I_VSC] * R_SCALE_D:.3f}: SPIPS p x vscale x the radius-curve distance scale {R_SCALE_D:.4f})"
              + (f"  vrot {args.vrot * th[I_VROT]:.2f} km/s" if args.free_vrot_scale else "")
              + (f"  d {th[I_DIST] * PC_PER_KPC:.1f} pc (plx {1.0 / th[I_DIST]:.3f} mas)" if args.free_distance else "")
              + (f"  radius amplitude x{th[I_RAMP]:.3f}" if args.free_ramp else "")
              + (f"  s_{BSYS_KEY} {th[I_BSYS]:.4f} mag" if args.free_bsys else "")
              + (f"  line-depth scale {th[I_DEPTH]:.4f}" if args.free_depth_scale else "")
              + (f"  template rms {float(jnp.std(jnp.asarray(r.x)[n_free:]) * TAU_T):.4f}" if n_tmpl else ""), flush=True)
        return r, th, aux

    def residual_sigma(x, label):
        """Per-(epoch, window) residual rms of the current model, floored at the pixel noise + spec floor."""
        th = unpack(jnp.asarray(x)); _, Sn = nodes_for(th); x_t = tmpl_of(jnp.asarray(x)); new = {}
        for wi in keep:
            mod = jax.jit(lambda t_, xt: apply_template(model_spectra(t_, Sn, wi), xt, wi))(th, x_t)
            mod = jax.vmap(lambda m, o, g: continuum_fix(m, o, g, xw[wi]))(mod, obs_sp[wi], good_sp[wi])
            res = np.where(np.asarray(good_sp[wi]), np.asarray(obs_sp[wi] - mod), np.nan)
            rms = np.nanstd(res, axis=1)
            new[wi] = jnp.asarray(np.maximum(rms, np.asarray(sig_sp[wi])[:, 0] * 0.0 + args.spec_floor))[:, None]
        print(f"{label}: per-window residual rms median " + " ".join(f"{all_windows[wi][0]:.0f}:{float(np.median(np.asarray(new[wi]))):.4f}" for wi in keep), flush=True)
        return new

    ckpt_path = (args.out[:-4] if args.out.endswith(".pkl") else args.out) + "_pass1.pkl"
    if args.resume_pass1:
        ck = pickle.load(open(args.resume_pass1, "rb"))
        if list(ck["labels"]) != list(labels) or not np.array_equal(np.asarray(ck["free_idx"]), free_idx) or len(ck["x"]) != n_free + n_tmpl:
            raise SystemExit(f"--resume-pass1 {args.resume_pass1}: free parameters / template size differ from this run's setup")
        if not (args.sigma_from_residuals and not args.phot_only):
            raise SystemExit("--resume-pass1 only makes sense with --sigma-from-residuals (it skips to pass 2)")
        x1 = np.asarray(ck["x"], float)
        for wi in keep:
            sig_sp[wi] = jnp.asarray(ck["sigma_pass1"][all_windows[wi]])
        print(f"resumed from {args.resume_pass1}: pass-1 solution and sigma restored, running pass 2 only", flush=True)
        vg = make_vg()
        r, th, aux = fit(x1, active_all, args.maxiter, "pass 2 (sigma = pass-1 residual rms)")
    else:
        r, th, aux = fit(x0_all, active_all, args.maxiter, "all data")
        if args.sigma_from_residuals and not args.phot_only:
            new_sig = residual_sigma(r.x, "pass 1")
            # checkpoint: pass 2 can be rerun from here with --resume-pass1 if the job dies later (walltime)
            with open(ckpt_path, "wb") as fh:
                pickle.dump(dict(stage="pass1", x=np.asarray(r.x), theta=np.asarray(th), labels=list(labels), free_idx=np.asarray(free_idx),
                                 sigma_pass1={all_windows[wi]: np.asarray(new_sig[wi]) for wi in keep}, args=vars(args)), fh, protocol=4)
            print(f"pass-1 checkpoint saved: {ckpt_path}", flush=True)
            for wi in keep:
                sig_sp[wi] = new_sig[wi]
            vg = make_vg()
            r, th, aux = fit(np.asarray(r.x), active_all, args.maxiter, "pass 2 (sigma = pass-1 residual rms)")
    x_best = np.asarray(r.x)
    teff_curve = np.asarray(fourier_eval(jnp.asarray(mph), jnp.asarray(th[:n_coef])))
    print(f"  fitted Teff curve {teff_curve.min():.0f}-{teff_curve.max():.0f} K vs SPIPS {mteff.min():.0f}-{mteff.max():.0f}; rms(fit - SPIPS) {np.sqrt(np.mean((teff_curve - mteff) ** 2)):.0f} K")
    if teff_obs is not None:
        pred = np.asarray(fourier_eval(jnp.asarray((teff_obs["phase"] - th[I_DPHI]) % 1.0), jnp.asarray(th[:n_coef])))
        print(f"  vs Andrievsky+2005 Teff: mean {np.mean(pred - teff_obs['teff']):+.0f} K, rms {np.sqrt(np.mean((pred - teff_obs['teff']) ** 2)):.0f} K")
    F_b, S_b = nodes_for(jnp.asarray(th))
    mags_fit, teff_e = model_mags(jnp.asarray(th), F_b)
    dv_best = None if args.phot_only else ccf_epochs(jnp.asarray(th), S_b, wi_mg)
    if dv_best is not None:
        m = np.isfinite(dv_best); b, a = np.polyfit(vpuls_model[m], dv_best[m], 1)
        print(f"  best fit: residual CCF velocity per epoch (Mg b): mean {np.nanmean(dv_best):+.2f}, rms {np.nanstd(dv_best):.2f} km/s; "
              f"slope vs <v_los,puls> {b:+.3f}", flush=True)
    result = dict(theta=th, theta0=theta0, free_idx=free_idx, labels=labels, S=S, x=np.asarray(r.x), chi2_phot=float(aux[0]), chi2_spec=float(aux[1]),
                  chi2_phot_parts={k: float(v) for k, v in aux[2].items()}, chi2_spec_parts={(all_windows[k] if k >= 0 else ("holdout", all_windows[-k - 1])): float(v) for k, v in aux[3].items()},
                  N_ph=N_ph, N_sp=N_sp, args=vars(args), meta=meta, windows=[all_windows[i] for i in keep], seasons=seasons, dv_season=dv_season, dv_epoch=dv_epoch,
                  dv_start=dv0, dv_best=dv_best, vpuls_model=vpuls_model, phases_spec=phases_spec, times_spec=times, season_spec=season,
                  phases_model=phases_model, mags_fit={b: np.asarray(v) for b, v in mags_fit.items()},
                  mags_start={b: np.asarray(v) for b, v in model_mags(theta0_j, F0)[0].items()}, teff_model_epochs=np.asarray(teff_e),
                  spips_phase=mph, spips_teff=mteff, teff_obs=teff_obs, obs=obs, T_nodes=np.asarray(T_nodes), fourier_order=K,
                  vmacro_order=KV, vmacro_curve=np.asarray(vmacro_at(jnp.asarray(th), jnp.linspace(0, 1, 200))),
                  zp_mode=zp_mode, holdout_windows=args.holdout_windows, holdout_source=args.holdout_source,
                  template={all_windows[wi]: np.asarray(template_of(jnp.asarray(x_best[n_free:]), wi)) for wi in keep} if n_tmpl else None,
                  template_shift_kms=(np.asarray(C_KMS * (10.0 ** np.asarray(tmpl_shift["val"]) - 1.0)) if n_tmpl else None),
                  sigma_spec={all_windows[wi]: np.asarray(sig_sp[wi])[:, 0] for wi in keep},
                  vmacro_epochs=np.asarray(vmacro_at(jnp.asarray(th), ph_spec - th[I_DPHI])))
    # a few model spectra for plotting: first kept window, 6 epochs spread in phase
    ex_ep = np.linspace(0, len(snaps_sp) - 1, min(6, len(snaps_sp))).round().astype(int)
    result["example_spectra"] = {}
    for wi in keep[:3]:
        mod = np.asarray(jax.jit(lambda t_, xt: apply_template(model_spectra(t_, S_b, wi), xt, wi))(jnp.asarray(th), tmpl_of(jnp.asarray(x_best))))
        modc = np.asarray(jax.vmap(lambda m_, o_, g_: continuum_fix(m_, o_, g_, xw[wi]))(jnp.asarray(mod), obs_sp[wi], good_sp[wi]))
        result["example_spectra"][all_windows[wi]] = dict(wl=10.0 ** np.asarray(lw_obs[wi]), epochs=ex_ep, phases=phases_spec[ex_ep],
                                                          obs=np.where(np.asarray(good_sp[wi])[ex_ep], np.asarray(obs_sp[wi])[ex_ep], np.nan), model=modc[ex_ep])
    if args.save_all_spectra and not args.phot_only:
        result["all_spectra"] = {}; t0 = time.time()
        for wi in keep:
            mod = np.asarray(jax.jit(lambda t_, xt: apply_template(model_spectra(t_, S_b, wi), xt, wi))(jnp.asarray(th), tmpl_of(jnp.asarray(x_best))))
            modc = np.asarray(jax.vmap(lambda m_, o_, g_: continuum_fix(m_, o_, g_, xw[wi]))(jnp.asarray(mod), obs_sp[wi], good_sp[wi]))
            result["all_spectra"][all_windows[wi]] = dict(wl=10.0 ** np.asarray(lw_obs[wi]), obs=np.asarray(obs_sp[wi]), good=np.asarray(good_sp[wi]),
                                                          model=modc, sigma=np.asarray(sig_sp[wi])[:, 0])
        result["v_line_epoch"] = dv_epoch + vpuls_model
        print(f"all spectra saved: {len(keep)} windows x {len(snaps_sp)} epochs ({time.time() - t0:.0f}s)", flush=True)
    if args.exact_examples and not args.phot_only:
        # the same epochs, every kept window, synthesised per element -- the emulator called for each face at its
        # own mu and Doppler-shifted individually -- instead of through the kernel surrogate. Template, broadening
        # and continuum treatment are identical, so the two are directly comparable pixel by pixel.
        result["example_spectra_exact"] = {}
        x_t = tmpl_of(jnp.asarray(x_best)); th_j = jnp.asarray(th); t0 = time.time()
        for wi in keep:
            lw_w = lw_obs[wi]                       # NOT `lw`: that is the photometric grid, used again below
            def one_exact(th_, e):
                m_ = exact_window(th_, e, wi)[0]
                if args.line_template:
                    m_ = m_ * (1.0 + jnp.interp(lw_w - tmpl_shift["val"][e], lw_w, template_of(x_t, wi)))
                return continuum_fix(m_, obs_sp[wi][e], good_sp[wi][e], xw[wi])
            sur = np.asarray(jax.jit(lambda t_, xt: apply_template(model_spectra(t_, S_b, wi), xt, wi))(th_j, x_t))
            sur = np.asarray(jax.vmap(lambda m_, o_, g_: continuum_fix(m_, o_, g_, xw[wi]))(jnp.asarray(sur), obs_sp[wi], good_sp[wi]))
            ex = np.stack([np.asarray(jax.jit(lambda t_, e=int(e): one_exact(t_, e))(th_j)) for e in ex_ep])
            gd = np.asarray(good_sp[wi])[ex_ep]
            dev = np.where(gd, ex - sur[ex_ep], 0.0)
            result["example_spectra_exact"][all_windows[wi]] = dict(
                wl=10.0 ** np.asarray(lw_w), epochs=ex_ep, phases=phases_spec[ex_ep],
                obs=np.where(gd, np.asarray(obs_sp[wi])[ex_ep], np.nan), model=ex, model_surrogate=sur[ex_ep],
                max_dev=np.abs(dev).max(axis=1), rms_dev=np.sqrt(np.mean(dev ** 2, axis=1)))
            print(f"  exact synthesis window {all_windows[wi][0]:.0f}: {len(ex_ep)} epochs, surrogate - exact max "
                  f"{np.abs(dev).max():.5f} rms {np.sqrt(np.mean(dev ** 2)):.6f} ({time.time() - t0:.0f}s so far)", flush=True)

    if args.exact_all and not args.phot_only:
        # every epoch of every kept window per element, through the same broadening, template, continuum fix, sigma and
        # robust loss as chi2_spec: the spectroscopic chi2 of the exact model with no kernel quadrature anywhere
        th_j = jnp.asarray(th); x_t = tmpl_of(jnp.asarray(x_best)); t0 = time.time()
        teff_sp = fourier_eval(ph_spec - th_j[I_DPHI], th_j[:n_coef]); vmac_sp = vmacro_at(th_j, ph_spec - th_j[I_DPHI])

        @jax.jit
        def exact_epoch(m, teff, lgg, dv, vmac, lw_w):
            p = set_ab(jnp.asarray(m.parameters).at[:, iT].set(teff), ab_of(th_j)).at[:, iG].set(logg_mean + lgg)
            los_unit = m.los_vector / jnp.linalg.norm(m.los_vector)
            mm = m._replace(parameters=p, pulsation_velocities=m.pulsation_velocities * th_j[I_VSC] + dv * los_unit[None, :])
            br = broaden(simulate_observed_flux(emu.intensity, mm, lw_w, chunk_size=256).T, vmac)
            return 1.0 - th_j[I_DEPTH] * (1.0 - br[0] / br[1])
        ex_all = {}; c_sp_ex = 0.0; p_sp_ex = {}
        for wi in keep:
            mod = jnp.stack([exact_epoch(m, teff_sp[e], logg_sp[e], dv_e_j[e], vmac_sp[e], lw_obs[wi]) for e, m in enumerate(snaps_sp)])
            mod = jax.vmap(lambda m_, o_, g_: continuum_fix(m_, o_, g_, xw[wi]))(apply_template(mod, x_t, wi), obs_sp[wi], good_sp[wi])
            sur = apply_template(model_spectra(th_j, S_b, wi), x_t, wi)
            sur = jax.vmap(lambda m_, o_, g_: continuum_fix(m_, o_, g_, xw[wi]))(sur, obs_sp[wi], good_sp[wi])
            r = jnp.where(good_sp[wi], (obs_sp[wi] - mod) / sig_sp[wi], 0.0)
            c = float(jnp.sum(jnp.sum(rho(r), axis=1) * ep_in_j)); p_sp_ex[all_windows[wi]] = c; c_sp_ex += active_all[wi] * c
            dev = np.where(np.asarray(good_sp[wi]), np.asarray(mod - sur), 0.0)
            ex_all[all_windows[wi]] = dict(model=np.asarray(mod, np.float32), model_surrogate=np.asarray(sur, np.float32),
                                           max_dev=np.abs(dev).max(axis=1), rms_dev=np.sqrt(np.mean(dev ** 2, axis=1)))
            print(f"  exact-all window {all_windows[wi][0]:.0f}: {len(snaps_sp)} epochs, chi2/N {c / max(int(np.asarray(good_sp[wi]).sum()), 1):.2f} "
                  f"(surrogate {float(aux[3][wi]) / max(int(np.asarray(good_sp[wi]).sum()), 1):.2f}), exact - surrogate max {np.abs(dev).max():.5f} "
                  f"rms {np.sqrt(np.mean(dev ** 2)):.6f} ({time.time() - t0:.0f}s so far)", flush=True)
        result["exact_all"] = dict(spectra=ex_all, chi2_spec=c_sp_ex, chi2_spec_parts=p_sp_ex, teff_epochs=np.asarray(teff_sp))
        print(f"exact-all spectra: chi2_spec/N {c_sp_ex / max(N_sp, 1):.3f} (surrogate {float(aux[1]) / max(N_sp, 1):.3f})", flush=True)
    if args.exact_all and not args.spec_only:
        args.exact_photometry = True          # the photometric half of --exact-all is the block below, scored after it

    if args.exact_photometry and not args.spec_only:
        # every photometric model phase synthesised per element at the fitted parameters, then scaled, reddened
        # and band-integrated exactly as model_mags does for the surrogate
        th_j = jnp.asarray(th); teff_ph = np.asarray(fourier_eval(ph_model, th_j[:n_coef]))
        dist_scale = (10.0 / (th[I_DIST] * PC_PER_KPC)) ** 2
        area_scale = ((rad_mean + th[I_RAMP] * (rad - rad_mean)) / rad) ** 2
        red = np.asarray(jnp.power(10.0, -0.4 * meta["r_v"] * th[I_EBV] * alam_av))
        t0 = time.time(); flux_ex = []
        for e, m in enumerate(snaps_ph):
            p_ = jnp.asarray(m.parameters)
            p_ = set_ab(p_.at[:, iT].set(teff_ph[e]), ab_of(jnp.asarray(th))).at[:, iG].set(logg_mean + float(logg_ph[e])).at[:, iVM].set(float(VM_PH[e]))
            f_ = simulate_observed_flux(emu.intensity, m._replace(parameters=p_), lw, chunk_size=256)[:, 0]
            flux_ex.append(np.asarray(f_) * dist_scale * area_scale[e] * red)
        flux_ex = np.stack(flux_ex)
        result["mags_exact"] = {b: np.array([float(band_mag(jnp.asarray(fl), wl_j, resp[b], zp[b])) for fl in flux_ex])
                                for b in args.bands}
        for b in args.bands:
            d_ = 1e3 * (np.asarray(mags_fit[b]) - result["mags_exact"][b])
            print(f"  exact photometry {b}: surrogate - exact mean {d_.mean():+.2f} mmag, rms {d_.std():.2f}, max |d| {np.abs(d_).max():.2f} mmag", flush=True)
        print(f"  exact photometry: {len(snaps_ph)} phases in {time.time() - t0:.0f}s", flush=True)
        if args.exact_all:
            c_ph_ex, p_ph_ex = chi2_phot_mags(th_j, {b: jnp.asarray(v) for b, v in result["mags_exact"].items()}, teff_ph)
            result.setdefault("exact_all", {}).update(chi2_phot=float(c_ph_ex), chi2_phot_parts={k: float(v) for k, v in p_ph_ex.items()})
            print(f"exact-all photometry: chi2_phot/N {float(c_ph_ex) / max(N_ph, 1):.3f} (" + " ".join(f"{k} {float(v):.1f}" for k, v in p_ph_ex.items())
                  + f"; surrogate {float(aux[0]) / max(N_ph, 1):.3f})", flush=True)

    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)

    try:
        if args.no_hessian or emulator_in_graph or n_tmpl:
            raise RuntimeError("skipped (emulator inside the graph or template parameters present)")
        Hm = np.asarray(jax.hessian(lambda x: total(x, active_all)[0])(jnp.asarray(r.x))); ev = np.linalg.eigvalsh(Hm)
        print(f"Hessian eigenvalues (scaled units): {np.array2string(ev, precision=3)}", flush=True)
        result["hessian"] = Hm
        if np.all(ev > 0):
            cov = 2.0 * np.linalg.inv(Hm); sg = np.sqrt(np.diag(cov)) * S[free_idx]
            Cc = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
            print("  formal 1-sigma: " + "  ".join(f"{labels[i]} {s:.3g}" for i, s in zip(free_idx, sg)))
            result.update(cov=cov, sigma=dict(zip([labels[i] for i in free_idx], sg)), corr=Cc)
    except Exception as exc:  # noqa: BLE001
        print(f"Hessian failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    if not args.skip_validation and not args.phot_only:
        check(jnp.asarray(th), "best fit")

    if args.jackknife and len(keep) > 1 and not args.phot_only:
        print("\n=== leave-one-window-out jackknife (warm start) ===", flush=True)
        jk = {}
        for wi in keep:
            active = {w: (0.0 if w == wi else 1.0) for w in keep}
            _, thj, _ = fit(np.asarray(r.x), active, 60, f"drop {all_windows[wi][0]:.0f}-{all_windows[wi][1]:.0f}")
            jk[all_windows[wi]] = thj
        A = np.array(list(jk.values())); n = A.shape[0]
        jk_err = np.sqrt((n - 1) / n * np.sum((A - A.mean(0)) ** 2, axis=0))
        print("jackknife error: " + "  ".join(f"{labels[i]} {jk_err[i]:.3g}" for i in free_idx), flush=True)
        result["jackknife"] = jk; result["jackknife_err"] = jk_err
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
