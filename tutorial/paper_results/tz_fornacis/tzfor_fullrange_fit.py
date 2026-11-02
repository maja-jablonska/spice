"""TZ For joint fit over the FULL HARPS range through SynthesisKernel (GPU).

One log-uniform grid 4450-6750 A at the HARPS pixel (harps_fullrange_log.npz;
Balmer cores, Na D, tellurics and the detector gap pre-masked), Clausen b, y
photometry, PHOEBE geometry fixed, gravity darkening fixed (beta unconstrained
on this system). Compared with tzfor_kernel_fit.py:

  * spline continuum per epoch (cubic B-splines, knots every --knot-spacing A)
    instead of a polynomial per 40 A window;
  * --free-vmacro: macroturbulence per star through the broadening kernel;
  * --free-vsini: a scale on the rotational part of each mesh's velocity field
    (los = v_orb + s * (los - v_orb)), kernels rebuilt inside the loss;
  * --delta: a per-pixel line-strength correction map of the PRIMARY in its
    rest frame, multiplying the emulator's line channel before the kernels so
    it Doppler-shifts with the star; Gaussian prior (--delta-sigma) plus a
    second-difference smoothness penalty; fitted by alternating with theta;
  * --derive-mask-out: after the fit, stack the residual in the primary's rest
    frame over --mask-epochs and write a per-epoch mask (threshold, dilation);
    --mask-in applies such a mask; --fit-epochs restricts the fit to odd/even
    epochs for the circularity test;
  * --jackknife-blocks N: leave-one-contiguous-block-out refits.
"""
import argparse, math, os, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent))   # parent: block_pca_spice
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.interpolate import BSpline
import tzfor_grad_inference as GI, tzfor_constants as K
from tzfor_kernel_fit import load_photometry, C_KMS, R_HARPS
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, gravity_darkened_rows, kernel_flux_multi
from spice.models import IcosphereModel
from spice.models.binary import Binary, add_orbit, _evaluate_orbit


class Geom(NamedTuple):
    """The three arrays build_synthesis_kernel reads from a mesh."""
    visible_cast_areas: jnp.ndarray
    mus: jnp.ndarray
    los_velocities: jnp.ndarray


class ElementMesh(NamedTuple):
    """The four arrays simulate_observed_flux reads from a mesh (a pytree, so it can be traced)."""
    visible_cast_areas: jnp.ndarray
    mus: jnp.ndarray
    los_velocities: jnp.ndarray
    parameters: jnp.ndarray


def epoch_subset(n, which):
    idx = np.arange(n)
    return {"all": idx, "odd": idx[1::2], "even": idx[::2]}[which]


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
    ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_fullrange_log.npz"))
    ap.add_argument("--spec-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1500_spectra_meshes.pkl"))
    ap.add_argument("--lc-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1200_lc6650_meshes.pkl"))
    ap.add_argument("--photometry", default=str(HERE / "tzfor_lightcurve.csv"))
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "fullrange_fit.pkl"))
    ap.add_argument("--result-in", default=None, help="warm start (theta, delta) from a previous result")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="start values by name, applied after --result-in (e.g. Teff1=4930 Teff2=6350 feh=-0.30)")
    ap.add_argument("--n-mu", type=int, default=16); ap.add_argument("--n-g", type=int, default=2); ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--mu-interp", default="linear", choices=["linear", "cubic"], help="kernel interpolation between mu nodes (cubic: error ~h^4, same emulator cost)")
    ap.add_argument("--mu-spacing", default="uniform", choices=["uniform", "sqrt"], help="mu nodes uniform in mu or in sqrt(mu) (denser at the limb)")
    ap.add_argument("--coord-interp", default="linear", choices=["linear", "cubic"], help="interpolation between the log g nodes (cubic needs --n-g >= 4)")
    ap.add_argument("--shift-interp", default="linear", choices=["linear", "cubic"], help="sub-pixel Doppler-shift interpolation (cubic: 2e-7 at 2x oversampling)")
    ap.add_argument("--beta", type=float, default=0.079, help="gravity-darkening exponent on g, fixed (unconstrained on TZ For)")
    ap.add_argument("--vmacro0", type=float, nargs=2, default=(5.0, 6.0)); ap.add_argument("--free-vmacro", action="store_true")
    ap.add_argument("--free-vsini", action="store_true")
    ap.add_argument("--cno1", type=float, nargs=3, default=None, metavar=("C", "N", "O"),
                    help="primary's [C/Fe] [N/Fe] [O/Fe] (emulator c, n, o; Magg+2022 scale), fixed; default: the mesh values (scaled solar)")
    ap.add_argument("--free-cno", action="store_true", help="fit the primary's c, n, o (starting from --cno1 or the mesh values)")
    ap.add_argument("--free-geometry", action="store_true", help="free R1, R2, inclination: light curves from SPICE's own binary (icospheres + occlusion) inside the loss; spectral kernels rescaled by R^2")
    ap.add_argument("--geo-n-vert", type=int, default=1000); ap.add_argument("--geo-mean-anom", type=float, default=0.0, help="mean anomaly [rad] that puts the deep eclipse at SPICE phase 0 (from tzfor_spice_lc_feasibility.py)")
    ap.add_argument("--geo-eclipse-halfwidth", type=float, default=0.025); ap.add_argument("--geo-n-eclipse", type=int, default=25, help="model phases per eclipse (spheres: the out-of-eclipse flux is constant, one quadrature phase covers it)")
    ap.add_argument("--dv-sys", type=float, default=0.25)
    ap.add_argument("--gamma-components", action="store_true", help="give each star its own systemic velocity, gamma1 / gamma2 of "
                    "Gallenne et al. 2016 (tzfor_constants GAMMA1_KMS / GAMMA2_KMS), relative to the mass-weighted GAMMA_KMS of the meshes: each "
                    "star's spectrum is synthesised on its own wavelength grid shifted by (gamma_s - GAMMA_KMS); kernel path and exact dump",
                    )
    ap.add_argument("--knot-spacing", type=float, default=40.0, help="continuum spline knot spacing [A]")
    ap.add_argument("--mesh-teff", action="store_true", help="exact dump: use the per-element temperatures carried by the meshes (PHOEBE's "
                    "own Teff map: gravity darkening on the Roche/rotationally flattened surfaces, plus any irradiation heating) instead of "
                    "Teff1/Teff2 x the gravity-darkening law; a literal port of the PHOEBE model (the in-fit kernel path keeps the law)")
    ap.add_argument("--no-continuum-fix", action="store_true", help="compare the model with the SuppNet-normalised spectra as they are "
                    "(no per-epoch continuum correction): a pure forward model, nothing adjusted to the spectra")
    ap.add_argument("--phot-ephemeris", type=float, nargs=2, default=None, metavar=("T0_HJD", "P_D"),
                    help="phase the photometry with this published ephemeris (phase 0 = the deeper minimum), e.g. Andersen et al. 1991 "
                         "Eq. 2: 2445032.6091 75.66675, instead of the spectroscopic T_P / P of Gallenne et al. 2016; keep dphi = 0")
    ap.add_argument("--defect-in", default=None, help="model dump made with --exact-dump at the start point: the fixed difference exact - kernel (spectra and light curves) is added to the kernel model inside the loss, so the fit converges on the exact model's residuals (defect correction / corrected surrogate)")
    ap.add_argument("--defect-ref", default=None, help="the kernel-synthesis model dump at the same parameters as --defect-in")
    ap.add_argument("--phot-clip", type=float, default=0.0, help="mmag: drop photometric points with |O-C| above this at the start point (0 = keep all)")
    ap.add_argument("--phot-nightly-sigma", type=float, default=0.0, help="mmag: profile a zero point per night with this Gaussian prior, in closed form inside the loss (0 = off)")
    ap.add_argument("--spec-floor", type=float, default=0.005); ap.add_argument("--spec-scale", type=float, default=1.0, help="multiply the spectral chi2 (e.g. 1/inflation^2 from NUTS) to test the photometry/spectroscopy weighting")
    ap.add_argument("--mask-in", default=None); ap.add_argument("--derive-mask-out", default=None)
    ap.add_argument("--mask-threshold", type=float, default=0.05); ap.add_argument("--mask-nsigma", type=float, default=4.0); ap.add_argument("--mask-dilate", type=int, default=2)
    ap.add_argument("--mask-epochs", default="all", choices=["all", "odd", "even"]); ap.add_argument("--fit-epochs", default="all", choices=["all", "odd", "even"])
    ap.add_argument("--delta", action="store_true"); ap.add_argument("--delta-sigma", type=float, default=0.03); ap.add_argument("--delta-smooth", type=float, default=1.0)
    ap.add_argument("--delta-alternations", type=int, default=3); ap.add_argument("--delta-maxiter", type=int, default=60)
    ap.add_argument("--delta-mode", default="joint", choices=["joint", "fit-only", "fixed"],
                    help="joint: alternate theta and delta (degenerate: delta can mimic the primary's line profiles); "
                         "fit-only: fit delta once at the warm-start theta on --fit-epochs and stop; fixed: use delta from --result-in unchanged")
    ap.add_argument("--delta-in", default=None, help="result pkl whose delta map is used (delta-mode fixed)")
    ap.add_argument("--jackknife-blocks", type=int, default=0)
    ap.add_argument("--dump-model", default=None, help="npz with the continuum-matched model spectra, observations, mask and light-curve model at the fitted theta (for figures)")
    ap.add_argument("--exact-dump", action="store_true", help="build the --dump-model spectra and light curves with exact per-element synthesis (simulate_observed_flux) instead of the kernel surrogate; use with --maxiter 0")
    ap.add_argument("--noise-check", action="store_true", help="measure the float32 roundoff of the objective and compare AD with finite differences at the start point")
    ap.add_argument("--stall-retries", type=int, default=2, help="L-BFGS-B restarts from a perturbed point when a fit stalls in its first line search")
    ap.add_argument("--maxiter", type=int, default=150); ap.add_argument("--n-phot-wl", type=int, default=16000); ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--gpus", type=int, default=1, help="shard the kernel node loop over this many devices (jax.devices()[:n])")
    ap.add_argument("--validation-out", default=None, help="npz with the epoch-0 exact and kernel spectra of the start-point validation (for diagnosing the difference)")
    ap.add_argument("--n-epochs", type=int, default=None); ap.add_argument("--n-lc-epochs", type=int, default=None); ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--exact-loss", action="store_true", help="optimise the EXACT per-element model (simulate_observed_flux for every epoch, star and light-curve phase "
                    "inside the gradient): no kernel quadrature anywhere; needs --exact-windows")
    ap.add_argument("--exact-windows", type=float, nargs="*", default=None, metavar="LO HI", help="keep only these wavelength ranges [A] of the HARPS grid (pairs lo hi)")
    ap.add_argument("--exact-phot-wl", type=int, default=16000, help="wavelength points of the exact light-curve synthesis over 4300-5900 A; only those "
                    "under the b or y response are synthesised (4000 aliased Doppler-shifted lines into 10 mmag of fake light curve)")
    ap.add_argument("--clip-at", nargs="*", default=[], metavar="NAME=VALUE", help="evaluate --phot-clip at these parameter values instead of the start "
                    "point (keeps the clipped set, hence the objective, identical across warm-started stages)")
    ap.add_argument("--exact-chunk", type=int, default=256, help="surface elements per chunk in simulate_observed_flux")
    ap.add_argument("--marcs-fe-h-mode", default="feh", choices=["feh", "snap"], help="atmosphere [Fe/H] input of 12-input bundles: "
                    "'feh' = the synthesis [Fe/H]; 'snap' = its nearest MARCS node, as in the training grid (line formation uses [Fe/H] either way)")
    ap.add_argument("--feh-secondary", type=float, default=None, help="fixed [Fe/H] of the secondary (line formation; its atmosphere "
                    "node follows --marcs-fe-h-mode); the fitted / set 'feh' then applies to the primary only (e.g. Gallenne+2016: +0.02 / -0.05)")
    ap.add_argument("--free-feh-secondary", action="store_true", help="fit the secondary's [Fe/H] as its own parameter 'feh2' "
                    "(start: --feh-secondary, else the primary's start value); 'feh' then applies to the primary only")
    ap.add_argument("--vmicro", type=float, nargs=2, default=None, metavar=("V1", "V2"),
                    help="microturbulence [km/s] of primary and secondary, replacing the mesh values (e.g. Gallenne+2016: 0.98 3.10)")
    ap.add_argument("--gp-noise", action="store_true", help="correlated spectral noise (tzfor_gp_noise.BlockGP): Matern-3/2 GP per epoch "
                    "with free amplitudes gp_lna<e> and a shared length gp_lnl1 [km/s], fitted jointly; the loss uses -2 ln L instead of the plain spectral chi2")
    ap.add_argument("--gp-long", action="store_true", help="with --gp-noise: add a second, long Matern-3/2 kernel (gp_lnb, gp_lnl2), shared by all epochs")
    ap.add_argument("--gp-block", type=int, default=1024, help="GP block length [pixels]; covariance between blocks is dropped")
    ap.add_argument("--phot-noise-scale", action="store_true", help="fit an error scale per band (phot_lnf_b, phot_lnf_y): sigma -> f sigma, with the 2 N ln f term")
    ap.add_argument("--free-vmicro", action="store_true", help="fit the primary's microturbulence as parameter 'vt1' (start: --vmicro V1, "
                    "else 1.0; bounds 0.3-4 km/s); the secondary keeps --vmicro V2 or the mesh value")
    ap.add_argument("--vsini", type=float, nargs=2, default=None, metavar=("V1", "V2"),
                    help="fixed projected rotation [km/s] of primary and secondary: the mesh rotation field is scaled to it about the "
                         "orbital mean (as --free-vsini does), e.g. Gallenne+2016: 6.1 45.7")
    ap.add_argument("--marcs-fe-h", type=float, default=None, help="fix the atmosphere [Fe/H] input to this value (normally a MARCS "
                    "node, e.g. the other node bracketing [Fe/H]); overrides --marcs-fe-h-mode. Line formation still uses [Fe/H]")
    ap.add_argument("--cheap-block-pca", default=None, help="multi-fidelity --exact-loss fit: a block_pca_emulator.py output directory; the "
                    "block-PCA MLP (block_pca_spice.BlockPCAIntensityEmulator) is the cheap model, corrected to TPayne's value and gradient "
                    "at every outer iterate (first-order consistent trust-region loop; converges to TPayne's optimum)")
    ap.add_argument("--cheap-model", default="mlp_4x2048", help="model subdirectory of --cheap-block-pca")
    ap.add_argument("--cheap-chunk", type=int, default=64, help="surface elements per chunk in the cheap loss")
    ap.add_argument("--mf-outer", type=int, default=30, help="outer iterations (one TPayne value+gradient each)")
    ap.add_argument("--mf-inner", type=int, default=30, help="L-BFGS-B iterations on the corrected cheap loss per outer iteration")
    ap.add_argument("--mf-radius", type=float, default=16.0, help="initial trust radius (max |step| in scaled units)")
    ap.add_argument("--mf-min-radius", type=float, default=1e-3)
    ap.add_argument("--mock-ts", default=None, help="injection-recovery: replace the spectra and light curves by the exact model built from the "
                    "Turbospectrum intensity tables in this tzfor_ts_tables.py npz (no emulator), at --mock-truth; the fit then recovers them with the emulator. Needs --exact-loss")
    ap.add_argument("--mock-truth", nargs="*", default=[], metavar="NAME=VALUE", help="truth parameters of the mock (default: the start point)")
    ap.add_argument("--mock-noise-seed", type=int, default=-1, help="add Gaussian noise at the data's sigma with this seed (default -1: noise-free mock, the fit then measures the bias)")
    ap.add_argument("--mock-chunk", type=int, default=16, help="surface elements per chunk while the mock is synthesised from the tables")
    ap.add_argument("--mock-out", default=None, help="npz with the mock spectra and light curves and the emulator model at the truth")
    args = ap.parse_args()
    if args.cheap_block_pca and not args.exact_loss:
        raise SystemExit("--cheap-block-pca needs --exact-loss (the cheap model replaces TPayne in the exact per-element synthesis)")
    print("devices:", jax.devices(), flush=True)

    emu = GI.make_emulator(marcs_fe_h_mode=args.marcs_fe_h_mode, marcs_fe_h=args.marcs_fe_h)
    SP = pickle.load(open(args.spec_meshes, "rb")); LC = pickle.load(open(args.lc_meshes, "rb"))
    H = np.load(args.harps, allow_pickle=True); H = {k: H[k] for k in H.files}
    names = SP["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
    iCNO = [names.index(k) for k in ("c", "n", "o")]
    assert np.allclose(H["times"], SP["times"])
    lw_full = np.asarray(H["logwl"], float); pix = None
    if args.exact_windows:
        wins = np.asarray(args.exact_windows, float).reshape(-1, 2); wl_full = 10.0 ** lw_full
        pix = np.where(np.any((wl_full[:, None] >= wins[None, :, 0]) & (wl_full[:, None] <= wins[None, :, 1]), axis=1))[0]
        H["obs"] = H["obs"][:, pix]; H["logwl"] = lw_full[pix]
        print(f"exact windows {wins.tolist()}: {pix.size} of {lw_full.size} pixels", flush=True)
    if args.exact_loss and pix is None: raise SystemExit("--exact-loss needs --exact-windows")
    if args.n_epochs:
        SP["models"] = SP["models"][:args.n_epochs]; SP["times"] = SP["times"][:args.n_epochs]; H["obs"] = H["obs"][:args.n_epochs]; H["sigma_blocks"] = H["sigma_blocks"][:args.n_epochs]
    if args.n_lc_epochs:
        LC["models"] = LC["models"][:args.n_lc_epochs]; LC["times"] = LC["times"][:args.n_lc_epochs]
    n_ep, n_lc = len(SP["models"]), len(LC["models"])
    lw_obs = np.asarray(H["logwl"], float); n = lw_obs.size; wl = 10.0 ** lw_obs; dlog = float(H["dlog"])
    shift_sys = math.log10(1.0 + args.dv_sys / C_KMS); lw_model = jnp.asarray(lw_obs - shift_sys)
    DGAM = [K.GAMMA1_KMS - K.GAMMA_KMS, K.GAMMA2_KMS - K.GAMMA_KMS] if args.gamma_components else [0.0, 0.0]
    if args.gamma_components and args.exact_loss:
        raise SystemExit("--gamma-components: kernel path and exact dump only (not --exact-loss)")
    lw_star = [lw_model - math.log10(1.0 + DGAM[s] / C_KMS) for s in (0, 1)]      # per-star model grid (= lw_model without the flag)
    if args.gamma_components:
        print(f"per-star systemic velocities: gamma1 {K.GAMMA1_KMS:.2f}, gamma2 {K.GAMMA2_KMS:.2f} km/s; offsets from the meshes' "
              f"{K.GAMMA_KMS:.3f}: {DGAM[0]:+.3f} / {DGAM[1]:+.3f} km/s", flush=True)

    # ---- per-star geometry: base rows, log g nodes, per-epoch element arrays ----
    def star_setup(models, s):
        m0 = models[0][s]; vis = np.asarray(m0.mus) > 0; a = np.asarray(m0.visible_cast_areas)[vis]; p = np.asarray(m0.parameters)[vis]
        allg = np.concatenate([np.asarray(m.parameters)[np.asarray(m.mus) > 0, iG] for m in (pair[s] for pair in models)])
        return (jnp.asarray(np.average(p, axis=0, weights=a)), float(np.average(p[:, iG], weights=a)), jnp.asarray(np.linspace(allg.min() - 0.005, allg.max() + 0.005, args.n_g)))
    base = [None, None]; gref = [0.0, 0.0]; gn = [None, None]
    for s in (0, 1): base[s], gref[s], gn[s] = star_setup(SP["models"], s)
    base_lc = [star_setup(LC["models"], s)[0] for s in (0, 1)]
    T0 = [float(base[0][iT]), float(base[1][iT])]; feh0 = float(base[0][iF])

    def elems(m):
        a = np.asarray(m.visible_cast_areas); mu = np.asarray(m.mus); v = np.asarray(m.los_velocities); g = np.asarray(m.parameters)[:, iG]
        vis = (mu > 0) & (a > 0); v_orb = float(np.average(v[vis], weights=a[vis]))
        return jnp.asarray(a), jnp.asarray(mu), jnp.asarray(v), jnp.asarray(g), v_orb
    E = [[elems(pair[s]) for pair in SP["models"]] for s in (0, 1)]
    rv = np.array([[E[s][e][4] + args.dv_sys + DGAM[s] for e in range(n_ep)] for s in (0, 1)])          # component RVs per epoch
    print(f"start: Teff1 {T0[0]:.1f}  Teff2 {T0[1]:.1f}  [Fe/H] {feh0:+.2f}  | RV1 {rv[0].min():+.0f}..{rv[0].max():+.0f}  RV2 {rv[1].min():+.0f}..{rv[1].max():+.0f} km/s", flush=True)

    # kernel half width sized for a vsini scale up to 2.5
    dfine = dlog / args.oversample
    KOPT = dict(mu_interp=args.mu_interp, mu_spacing=args.mu_spacing, coord_interp=args.coord_interp, shift_interp=args.shift_interp)
    DEV = list(jax.devices())[:args.gpus] if args.gpus > 1 else None
    print(f"devices: {len(jax.devices())} visible ({jax.devices()[0].platform}), kernel node loop sharded over {len(DEV) if DEV else 1}", flush=True)
    print(f"kernel quadrature: mu {args.mu_interp}/{args.mu_spacing} x{args.n_mu}, log g {args.coord_interp} x{args.n_g}, shift {args.shift_interp} at {args.oversample}x", flush=True)
    VS_FIX = [1.0, 1.0]                       # rotation-field scales when --vsini fixes the projected rotation
    if args.vsini is not None:
        vsm = [max(float(np.max(np.abs(np.asarray(E[s][e][2]) - E[s][e][4])[(np.asarray(E[s][e][1]) > 0) & (np.asarray(E[s][e][0]) > 0)]))
                   for e in range(n_ep)) for s in (0, 1)]
        VS_FIX = [args.vsini[s] / vsm[s] for s in (0, 1)]
        print(f"v sin i fixed at {args.vsini[0]:.2f} / {args.vsini[1]:.2f} km/s: meshes have {vsm[0]:.2f} / {vsm[1]:.2f}, "
              f"rotation scales {VS_FIX[0]:.4f} / {VS_FIX[1]:.4f}", flush=True)
    HW = [int(math.ceil(math.log10(1.0 + (max(abs(E[s][e][4]) for e in range(n_ep)) + 2.5 * max(float(jnp.max(jnp.abs(E[s][e][2] - E[s][e][4]))) for e in range(n_ep)) + 3.0) / C_KMS) / dfine)) + 1 for s in (0, 1)]

    R_0 = [K.PRIMARY_RADIUS, K.SECONDARY_RADIUS]
    def kernels_for(s, vscale, rscale=1.0):
        """Per-epoch spectral kernels of star s; ``rscale`` = R/R_0 scales the projected areas (light ratio) of the well-detached, near-spherical star."""
        ks = []
        for e in range(n_ep):
            a, mu, v, g, v_orb = E[s][e]
            los = v_orb + vscale * (v - v_orb)
            ks.append(build_synthesis_kernel(Geom(a * rscale ** 2, mu, los), lw_star[s], args.n_mu, args.oversample, half_width=HW[s], element_coordinate=g, coordinate_nodes=gn[s], **KOPT))
        return ks
    fixed_kernels = None if (args.free_vsini or args.free_geometry or args.exact_loss) else [kernels_for(0, VS_FIX[0]), kernels_for(1, VS_FIX[1])]
    lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), args.n_phot_wl); wl_ph = 10.0 ** lw_ph
    ph_k = [[build_synthesis_kernel(pair[s], lw_ph, args.n_mu, 1, element_coordinate=jnp.asarray(pair[s].parameters)[:, iG], coordinate_nodes=gn[s], **KOPT) for pair in LC["models"]] for s in (0, 1)]
    resp = {nm.split(":")[1]: jnp.interp(wl_ph, jnp.asarray(LC["passbands"][nm][0]), jnp.asarray(LC["passbands"][nm][1]), left=0., right=0.) for nm in ("Stromgren:b", "Stromgren:y")}
    print(f"kernels: half widths {HW} px, {'rebuilt in-loss (free vsini)' if args.free_vsini else 'prebuilt'}", flush=True)

    # ---- data, masks, noise, continuum basis ----
    obs_np = np.array(H["obs"], float); good_np = np.isfinite(obs_np)
    if args.mask_in:
        m = np.asarray(np.load(args.mask_in, allow_pickle=True)["mask"], bool)[:n_ep]; m = m if pix is None else m[:, pix]; good_np &= ~m
        print(f"mask-in: dropping {100 * m[np.isfinite(obs_np)].mean():.1f}% of usable pixels", flush=True)
    fit_e = epoch_subset(n_ep, args.fit_epochs); ep_w = np.zeros(n_ep); ep_w[fit_e] = 1.0
    edges = np.asarray(H["block_edges"]); blk = np.clip(np.searchsorted(edges, wl) - 1, 0, edges.size - 2)
    sig_np = np.hypot(np.nan_to_num(H["sigma_blocks"][:n_ep, blk], nan=0.01), args.spec_floor)
    obs = jnp.asarray(np.where(good_np, obs_np, 1.0)); good = jnp.asarray(good_np); sig = jnp.asarray(sig_np); ep_w = jnp.asarray(ep_w); blk_j = jnp.asarray(blk)
    N_sp = int(good_np[fit_e].sum())
    kn = np.arange(lw_obs[0], lw_obs[-1], math.log10(1 + args.knot_spacing / 5500.0)); interior = kn[(kn > lw_obs[0] + 1e-9) & (kn < lw_obs[-1] - 1e-9)]
    t_kn = np.concatenate([[lw_obs[0]] * 4, interior, [lw_obs[-1]] * 4])      # clamped cubic knot vector: k+1 copies at each end
    Bmat = BSpline.design_matrix(lw_obs, t_kn, 3).toarray()
    Bmat = jnp.asarray(Bmat[:, (Bmat > 1e-3).sum(0) >= 30]); nB = Bmat.shape[1]      # windows: drop splines with (almost) no pixels under them
    print(f"continuum: {nB} cubic B-spline coefficients per epoch ({args.knot_spacing} A knots); fitting {len(fit_e)} epochs, {N_sp} pixels", flush=True)
    ph_o, mag_o = load_photometry(args.photometry); sig_ph = {"b": 0.0041, "y": 0.0035}; N_ph = sum(len(v) for v in mag_o.values())
    import pandas as pd
    _hjd = pd.read_csv(args.photometry)["HJD"].values; _nights, night_idx = np.unique(np.floor(_hjd), return_inverse=True); n_nights = int(_nights.size); night_idx = jnp.asarray(night_idx)
    if args.phot_ephemeris is not None:
        T0e, Pe = args.phot_ephemeris
        ph_o = ((np.asarray(_hjd, float) + 2440000.0 - T0e) % Pe) / Pe        # Clausen HJD are truncated by 2 440 000
        print(f"photometry phased with the published ephemeris HJD {T0e} + {Pe} E (phase 0 = deeper minimum)", flush=True)
    ph_w = {b: jnp.ones(len(v)) for b, v in mag_o.items()}       # photometric point weights (0 for clipped points); updated in place by --phot-clip
    sig_night = args.phot_nightly_sigma / 1000.0
    print(f"photometry: {N_ph} points on {n_nights} nights; nightly zero points {'profiled, prior %.1f mmag' % args.phot_nightly_sigma if sig_night > 0 else 'off'}", flush=True)

    def band_residuals(mags, dphi, b, nightly=True, sig_b=None):
        """(model curve, model at the observed phases, observed mags on the model's zero point, residuals, nightly offsets).
        A global zero point (median) is always removed; with --phot-nightly-sigma a per-night offset is then profiled in
        closed form, z_k = sum_i w_i r_i / sigma^2 / (sum_i w_i / sigma^2 + 1 / sigma_night^2), its Gaussian prior being
        the ridge term, so nights with only in-eclipse points cannot absorb a genuine depth error."""
        mi = jnp.interp((ph_o - dphi) % 1.0, mph, mags, period=1.0); res = mag_o[b] - mi; zp = jnp.median(res); res = res - zp
        if nightly and sig_night > 0:
            w = ph_w[b] / (sig_ph[b] if sig_b is None else sig_b) ** 2
            z = jax.ops.segment_sum(w * res, night_idx, n_nights) / (jax.ops.segment_sum(w, night_idx, n_nights) + 1.0 / sig_night ** 2)
            res = res - z[night_idx]
        else:
            z = jnp.zeros(n_nights)
        return mags, mi, mag_o[b] - zp - z[night_idx], res, z
    model_phase = ((np.asarray(LC["times"]) - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS; order = np.argsort(model_phase); mph = jnp.asarray(model_phase[order])
    if args.free_geometry:
        # SPICE binary clock: deep eclipse at phase 0 (--geo-mean-anom), shallow at 0.5; spheres -> flat outside eclipses
        hw, ne = args.geo_eclipse_halfwidth, args.geo_n_eclipse
        geo_phases = np.unique(np.concatenate([np.linspace(-hw, hw, ne) % 1.0, 0.5 + np.linspace(-hw, hw, ne), [0.25]])); mph = jnp.asarray(geo_phases)
        print(f"free geometry: {geo_phases.size} model phases, {args.geo_n_vert} vertices per star, mean anomaly {args.geo_mean_anom:.4f}", flush=True)

    DEF_SPEC = jnp.zeros_like(obs); DEF_LC = {b: jnp.zeros(mph.shape[0]) for b in ("b", "y")}
    if args.defect_in:
        if not args.defect_ref: raise SystemExit("--defect-in needs --defect-ref")
        if args.free_geometry: raise SystemExit("--defect-in is for the fixed-geometry fit (its light-curve phases are the mesh phases)")
        De = np.load(args.defect_in, allow_pickle=True); Dk = np.load(args.defect_ref, allow_pickle=True)
        if not np.allclose(np.asarray(De["theta"]), np.asarray(Dk["theta"])): raise SystemExit("--defect-in and --defect-ref were made at different parameters")
        ds = np.asarray(De["model"])[:n_ep] - np.asarray(Dk["model"])[:n_ep]; ds = np.where(np.isfinite(ds), ds, 0.0)
        DEF_SPEC = jnp.asarray(ds)
        for b in DEF_LC:
            if not np.allclose(np.asarray(De[f"lc_{b}_model_phase"]), np.asarray(mph)): raise SystemExit("defect light-curve phases differ from the model phases")
            DEF_LC[b] = jnp.asarray(np.asarray(De[f"lc_{b}_model_mag"]) - np.asarray(Dk[f"lc_{b}_model_mag"]))
        print(f"defect correction ({De['synthesis']} - {Dk['synthesis']} at theta {np.round(np.asarray(De['theta'])[:3], 1)}): spectra max {np.abs(ds[good_np]).max():.2e} "
              f"rms {np.sqrt((ds[good_np] ** 2).mean()):.2e}; light curves " + ", ".join(f"{b} {1000 * float(jnp.abs(v).max()):.3f} mmag" for b, v in DEF_LC.items()), flush=True)

    HW_PH = 24   # orbital + rotational velocities reach ~100 km/s = 17 px on the photometric grid

    @jax.checkpoint
    def spice_phase_kernels(bn, t):
        """Both stars' kernel arrays at one time, checkpointed and run through lax.scan: an unrolled loop let XLA keep all
        phases' occlusion intermediates alive at once (47 GB), the scan holds one phase at a time."""
        m1, m2 = _evaluate_orbit(bn, t, n_neighbors1=NN1, n_neighbors2=NN2)
        return build_synthesis_kernel(m1, lw_ph, args.n_mu, 1, half_width=HW_PH, **KOPT).kernels, build_synthesis_kernel(m2, lw_ph, args.n_mu, 1, half_width=HW_PH, **KOPT).kernels

    def spice_kernels_all(bn):
        """Lists (per phase) of SynthesisKernel for both stars, from a scan over the model phases."""
        _, (K1, K2) = jax.lax.scan(lambda c, t: (c, spice_phase_kernels(bn, t)), None, jnp.asarray(geo_phases) * K.PERIOD_YR)
        return [geo_tmpl[0]._replace(kernels=K1[i]) for i in range(len(geo_phases))], [geo_tmpl[1]._replace(kernels=K2[i]) for i in range(len(geo_phases))]

    def spice_bodies(R1, R2, rows_plain):
        return (IcosphereModel.construct(args.geo_n_vert, R1, K.PRIMARY_MASS, rows_plain[0], names),
                IcosphereModel.construct(args.geo_n_vert, R2, K.SECONDARY_MASS, rows_plain[1], names))
    if args.free_geometry:
        # neighbour counts of the occlusion search: from_bodies needs concrete geometry, so fix them here (1.5x margin for larger radii)
        _b1, _b2 = spice_bodies(K.PRIMARY_RADIUS, K.SECONDARY_RADIUS, base_lc); _ref = Binary.from_bodies(_b1, _b2)
        NN1, NN2 = min(64, int(math.ceil(1.5 * int(_ref.n_neighbours1)))), min(64, int(math.ceil(1.5 * int(_ref.n_neighbours2)))); print(f"occlusion neighbour counts: {NN1}, {NN2}", flush=True)
        _m1, _m2 = _evaluate_orbit(add_orbit(_ref, K.PERIOD_YR, 0.0, 0.0, jnp.deg2rad(K.INCL_DEG), 0.0, 0.0, args.geo_mean_anom, 0.0, 0.0, 400), 0.25 * K.PERIOD_YR, n_neighbors1=NN1, n_neighbors2=NN2)
        geo_tmpl = (build_synthesis_kernel(_m1, lw_ph, args.n_mu, 1, half_width=HW_PH, **KOPT), build_synthesis_kernel(_m2, lw_ph, args.n_mu, 1, half_width=HW_PH, **KOPT))   # static fields template

    def spice_binary(R1, R2, inc_deg, rows_plain):
        b1, b2 = spice_bodies(R1, R2, rows_plain)
        bn = Binary(b1, b2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., jnp.zeros_like(b1.centers), jnp.zeros_like(b2.centers), jnp.zeros_like(b1.velocities), jnp.zeros_like(b2.velocities), NN1, NN2)
        return add_orbit(bn, K.PERIOD_YR, 0.0, 0.0, jnp.deg2rad(inc_deg), 0.0, 0.0, args.geo_mean_anom, 0.0, 0.0, 400)

    # ---- parameter vector ----
    pnames = ["Teff1", "Teff2", "feh", "dphi"]; theta0 = [T0[0], T0[1], feh0, 0.0]; S = [100., 100., 0.1, 0.001]; lo = [3800., 3800., -1.5, -0.02]; hi = [6900., 6900., 0.5, 0.02]   # dphi: Clausen minima vs Gallenne T_P (~ -0.0004 cycles)
    if args.free_vmacro: pnames += ["vmac1", "vmac2"]; theta0 += list(args.vmacro0); S += [1., 1.]; lo += [0.5, 0.5]; hi += [20., 20.]
    if args.free_vsini: pnames += ["vsini_scale1", "vsini_scale2"]; theta0 += [1.0, 1.0]; S += [0.1, 0.1]; lo += [0.5, 0.5]; hi += [2.0, 2.0]
    if args.free_geometry: pnames += ["R1", "R2", "incl"]; theta0 += [K.PRIMARY_RADIUS, K.SECONDARY_RADIUS, K.INCL_DEG]; S += [0.1, 0.05, 0.1]; lo += [6.0, 2.5, 78.0]; hi += [11.0, 6.0, 90.0]
    cno0 = list(args.cno1) if args.cno1 is not None else [float(base[0][i]) for i in iCNO]
    if args.free_feh_secondary:
        pnames += ["feh2"]; theta0 += [args.feh_secondary if args.feh_secondary is not None else feh0]; S += [0.1]; lo += [-1.5]; hi += [0.5]
    if args.gp_noise:
        from tzfor_gp_noise import BlockGP
        if args.exact_loss or args.delta or args.cheap_block_pca or args.spec_scale != 1.0:
            raise SystemExit("--gp-noise: kernel path only (no --exact-loss, --delta, --cheap-block-pca) and --spec-scale 1")
        GP = BlockGP(good_np, sig_np, dlog * math.log(10.0) * C_KMS, block=args.gp_block)
        a0 = np.sqrt(8.0) * GP.med_sigma                    # chi2/N ~ 9 at the unit-weight optimum: correlated part ~ sqrt(8) sigma
        pnames += [f"gp_lna{e}" for e in range(n_ep)] + ["gp_lnl1"]; theta0 += list(np.log(a0)) + [math.log(6.0)]
        S += [0.1] * (n_ep + 1); lo += [math.log(1e-4)] * n_ep + [math.log(0.3)]; hi += [math.log(0.3)] * n_ep + [math.log(40.0)]
        if args.gp_long:
            pnames += ["gp_lnb", "gp_lnl2"]; theta0 += [math.log(float(np.median(a0)) / 3.0), math.log(80.0)]; S += [0.1, 0.1]
            lo += [math.log(1e-4), math.log(40.0)]; hi += [math.log(0.3), math.log(2000.0)]
        print(f"GP noise: block {args.gp_block} px ({GP.nb} blocks/epoch), pixel {dlog * math.log(10.0) * C_KMS:.3f} km/s, start a_e {np.round(a0, 4).tolist()}, l1 6 km/s"
              + (", long kernel b %.4f l2 80 km/s" % (float(np.median(a0)) / 3.0) if args.gp_long else ""), flush=True)
    if args.phot_noise_scale:
        pnames += ["phot_lnf_b", "phot_lnf_y"]; theta0 += [0.0, 0.0]; S += [0.05, 0.05]; lo += [math.log(0.3)] * 2; hi += [math.log(5.0)] * 2
    if args.free_vmicro:
        pnames += ["vt1"]; theta0 += [args.vmicro[0] if args.vmicro is not None else 1.0]; S += [0.2]; lo += [0.3]; hi += [4.0]
    if args.free_cno: pnames += ["c1", "n1", "o1"]; theta0 += cno0; S += [0.1, 0.1, 0.1]; lo += [-1.0, -1.0, -1.0]; hi += [1.0, 1.0, 1.0]
    print(f"primary C/N/O: {'free, start ' if args.free_cno else 'fixed '}{cno0}", flush=True)
    theta0 = np.array(theta0); S = np.array(S); n_par = len(pnames)
    delta0 = np.zeros(n)
    if args.result_in:
        Rin = pickle.load(open(args.result_in, "rb"))
        for i, nm in enumerate(pnames):
            if nm in Rin["pnames"]: theta0[i] = Rin["theta"][Rin["pnames"].index(nm)]
        if Rin.get("delta") is not None and args.delta: delta0 = np.asarray(Rin["delta"])
        print("warm start from", args.result_in, flush=True)
    P = {nm: i for i, nm in enumerate(pnames)}
    def PSIG(theta, b):
        """Photometric sigma of band b: catalogue value, times the fitted scale with --phot-noise-scale (None = catalogue)."""
        return sig_ph[b] * jnp.exp(theta[P[f"phot_lnf_{b}"]]) if args.phot_noise_scale else None
    for kv in args.set:
        nm, val = kv.split("=")
        assert nm in P, f"--set {nm}: not a parameter of this run ({pnames})"
        theta0[P[nm]] = float(val)
    if args.set:
        print("start values set: " + "  ".join(f"{nm} {theta0[i]:.4g}" for i, nm in enumerate(pnames)), flush=True)

    iVM = names.index("vmicro")
    def cno(x, s, theta):
        """Set the primary's c, n, o on a row or an element matrix (last axis = parameters); the secondary keeps the mesh values.
        Also applies --feh-secondary, --vmicro and --free-vmicro: every path builds a star's rows as cno(row with [Fe/H] set, s, theta)."""
        if s == 1 and args.free_feh_secondary:
            x = x.at[..., iF].set(theta[P["feh2"]])
        elif s == 1 and args.feh_secondary is not None:
            x = x.at[..., iF].set(args.feh_secondary)
        if s == 0 and args.free_vmicro:
            x = x.at[..., iVM].set(theta[P["vt1"]])
        elif args.vmicro is not None:
            x = x.at[..., iVM].set(args.vmicro[s])
        if s != 0 or (args.cno1 is None and not args.free_cno):
            return x
        v = [theta[P[k]] for k in ("c1", "n1", "o1")] if args.free_cno else args.cno1
        for i, vi in zip(iCNO, v):
            x = x.at[..., i].set(vi)
        return x

    # ---- model ----
    HALF_BR = 60
    xb = jnp.arange(-HALF_BR, HALF_BR + 1)
    def broaden(y, vmacro):
        sigma_pix = math.hypot(C_KMS / R_HARPS, 1.0) * 0 + jnp.hypot(C_KMS / R_HARPS, vmacro) / (C_KMS * math.log(10.0) * dlog) / 2.3548
        k = jnp.exp(-0.5 * (xb / sigma_pix) ** 2); k = k / k.sum()
        yp = jnp.pad(y, [(0, 0)] * (y.ndim - 1) + [(HALF_BR, HALF_BR)], mode="edge")
        return conv_valid(yp, k)

    def intensity_with_delta(delta):
        def fn(x, mu, prow):
            out = emu.intensity(x, mu, prow)
            return out.at[:, 0].multiply(1.0 + jnp.interp(x, lw_model, delta))
        return fn

    def rows(theta):
        r = []
        for s in (0, 1):
            b = cno(base[s].at[iF].set(theta[P["feh"]]), s, theta); r.append(gravity_darkened_rows(b, iT, iG, gn[s], theta[P[f"Teff{s+1}"]], gref[s], args.beta))
        return r

    def spectra_parts(theta, delta):
        """Broadened line and continuum channels of each star: (F1, F2, C) with C the summed continuum."""
        r = rows(theta)
        vs = [theta[P["vsini_scale1"]], theta[P["vsini_scale2"]]] if args.free_vsini else VS_FIX
        rs = [theta[P["R1"]] / R_0[0], theta[P["R2"]] / R_0[1]] if args.free_geometry else [1.0, 1.0]
        ks = fixed_kernels if fixed_kernels is not None else [kernels_for(0, vs[0], rs[0]), kernels_for(1, vs[1], rs[1])]
        vm = [theta[P["vmac1"]], theta[P["vmac2"]]] if args.free_vmacro else list(args.vmacro0)
        s1 = kernel_flux_multi(intensity_with_delta(delta), ks[0], r[0], wavelength_chunk_size=args.chunk, devices=DEV)
        s2 = kernel_flux_multi(emu.intensity, ks[1], r[1], wavelength_chunk_size=args.chunk, devices=DEV)
        F1, F2 = broaden(s1[..., 0], vm[0]), broaden(s2[..., 0], vm[1]); C = broaden(s1[..., 1], vm[0]) + broaden(s2[..., 1], vm[1])
        return F1, F2, C

    def spectra(theta, delta):
        F1, F2, C = spectra_parts(theta, delta)
        return (F1 + F2) / C                                                      # (n_ep, n)

    def continuum_fix(model, ob, gd):
        if args.no_continuum_fix:
            return model
        w = jnp.where(gd, 1.0, 0.0); A = model[:, None] * Bmat * w[:, None]
        coef = jnp.linalg.solve(A.T @ A + 1e-9 * jnp.eye(nB), A.T @ (ob * w)); return model * (Bmat @ coef)

    def spec_model(theta, delta):
        return jax.vmap(continuum_fix)(spectra(theta, delta), obs, good) + DEF_SPEC

    def chi2_spec(theta, delta, block_w, mod=None):
        mod = spec_model(theta, delta) if mod is None else mod
        r = jnp.where(good, (obs - mod) / sig, 0.0)
        per_epoch = jnp.sum(block_w[blk_j][None, :] * r ** 2, axis=1)      # block weights mapped to pixels
        return jnp.sum(ep_w * per_epoch)                                     # two-level sum: float32 roundoff ~1e-6 of chi2, not 1e-5

    def phot_model(theta):
        """Per band: (model curve on the model phases, model at the observed phases, observed mags on the model's zero point, residuals)."""
        if args.free_geometry:
            # spheres: one emulator row per star (gravity darkening is a <20 K intra-star effect on TZ For), kernels from SPICE's occlusion
            rp = [cno(base_lc[s].at[iT].set(theta[P[f"Teff{s+1}"]]).at[iF].set(theta[P["feh"]]), s, theta) for s in (0, 1)]
            bn = spice_binary(theta[P["R1"]], theta[P["R2"]], theta[P["incl"]], rp); ks = spice_kernels_all(bn)
            flux = kernel_flux_multi(emu.intensity, ks[0], rp[0], devices=DEV)[..., 0] + kernel_flux_multi(emu.intensity, ks[1], rp[1], devices=DEV)[..., 0]
        else:
            r = []
            for s in (0, 1):
                b = cno(base_lc[s].at[iF].set(theta[P["feh"]]), s, theta); r.append(gravity_darkened_rows(b, iT, iG, gn[s], theta[P[f"Teff{s+1}"]], gref[s], args.beta))
            flux = kernel_flux_multi(emu.intensity, ph_k[0], r[0], devices=DEV)[..., 0] + kernel_flux_multi(emu.intensity, ph_k[1], r[1], devices=DEV)[..., 0]
        out = {}
        for b, rr in resp.items():
            mags = jax.vmap(lambda f: GI.passband_mag(f, wl_ph, rr))(flux); mags = (mags if args.free_geometry else mags[order] + DEF_LC[b]); mags = mags - jnp.median(mags)
            out[b] = band_residuals(mags, theta[P["dphi"]], b, sig_b=PSIG(theta, b))
        return out

    def chi2_phot(theta, pm=None):
        """Plain photometric chi2 (catalogue sigmas; the nightly offsets profiled with the fitted sigmas if --phot-noise-scale)."""
        return sum(jnp.sum(ph_w[b] * v[3] ** 2) / sig_ph[b] ** 2 + (jnp.sum((v[4] / sig_night) ** 2) if sig_night > 0 else 0.0)
                   for b, v in (phot_model(theta) if pm is None else pm).items())

    def delta_prior(delta):
        return jnp.sum((delta / args.delta_sigma) ** 2) + args.delta_smooth * jnp.sum((jnp.diff(delta, 2) / args.delta_sigma) ** 2)

    if args.exact_loss:
        # ---- exact per-element model inside the loss: every surface element of every epoch synthesised individually ----
        # Each element gets its own gravity-darkened Teff and its own Doppler shift (the vsini scale acts on the element
        # velocities about the orbital mean, as in the exact dump); macroturbulence and the continuum fit are unchanged.
        # The windows are synthesised with HALF_BR padding pixels on each side so that the broadening sees real spectrum.
        from spice.spectrum.spectrum import simulate_observed_flux
        if args.delta or args.defect_in: raise SystemExit("--exact-loss: no --delta or --defect-in")
        runs = np.split(pix, np.where(np.diff(pix) > 1)[0] + 1); seg = [r.size for r in runs]
        lw_ex = jnp.asarray(np.concatenate([lw_full[r[0]] + dlog * np.arange(-HALF_BR, r.size + HALF_BR) for r in runs]) - shift_sys)

        def stack_star(models, s):
            ms = [pair[s] for pair in models]; st = lambda f: jnp.stack([jnp.asarray(f(m)) for m in ms])
            A, MU, V = st(lambda m: m.visible_cast_areas), st(lambda m: m.mus), st(lambda m: m.los_velocities)
            vis = (np.asarray(MU) > 0) & (np.asarray(A) > 0)
            VO = jnp.asarray([np.average(np.asarray(V)[i][vis[i]], weights=np.asarray(A)[i][vis[i]]) for i in range(len(ms))])
            return A, MU, V, VO, st(lambda m: m.parameters)
        EXS = [stack_star(SP["models"], s) for s in (0, 1)]; EXL = [stack_star(LC["models"], s) for s in (0, 1)]

        def rscale(theta, s):
            return theta[P[f"R{s + 1}"]] / R_0[s] if args.free_geometry else 1.0

        def elem_params(p, s, theta):
            """Element rows: gravity-darkened Teff, [Fe/H], CNO; under --free-geometry log g = log g(mesh) - 2 log10(R / R_0)
            (fixed mass), which the kernel path's R^2 area scaling leaves out."""
            teff = theta[P[f"Teff{s + 1}"]] * jnp.power(10.0, args.beta * (p[:, iG] - gref[s]))
            p = cno(p.at[:, iT].set(teff).at[:, iF].set(theta[P["feh"]]), s, theta)
            return p.at[:, iG].add(-2.0 * jnp.log10(rscale(theta, s))) if args.free_geometry else p

        # swapped to the cheap model (and its cond/checkpoint-free element scan) while the cheap loss is traced (--cheap-block-pca)
        EMU = {"f": emu.intensity, "chunk": args.exact_chunk, "sim": simulate_observed_flux}

        def exact_flux(a, mu, v, p, lw):
            return EMU["sim"](EMU["f"], ElementMesh(a, mu, v, p), lw, chunk_size=EMU["chunk"])

        def broaden_segments(y, vmacro):
            """(n_ep, padded windows) -> (n_ep, n): the broadening of `broaden`, applied per window on its padded segment."""
            sigma_pix = jnp.hypot(C_KMS / R_HARPS, vmacro) / (C_KMS * math.log(10.0) * dlog) / 2.3548
            k = jnp.exp(-0.5 * (xb / sigma_pix) ** 2); k = k / k.sum(); out = []; o = 0
            for L in seg:
                out.append(conv_valid(y[:, o:o + L + 2 * HALF_BR], k)); o += L + 2 * HALF_BR
            return jnp.concatenate(out, axis=1)

        def spectra_parts_exact(theta, delta=None):
            vm = [theta[P["vmac1"]], theta[P["vmac2"]]] if args.free_vmacro else list(args.vmacro0)
            out = []
            for s in (0, 1):
                def one(th, xs, s=s):
                    a, mu, v, vo, p = xs
                    vs_ = th[P[f"vsini_scale{s + 1}"]] if args.free_vsini else VS_FIX[s]
                    return exact_flux(a * rscale(th, s) ** 2, mu, vo + vs_ * (v - vo), elem_params(p, s, th), lw_ex)
                out.append(map_epochs(one, theta, EXS[s], DEV))                               # (n_ep, n_ex, 2)
            F1, F2 = broaden_segments(out[0][..., 0], vm[0]), broaden_segments(out[1][..., 0], vm[1])
            C = broaden_segments(out[0][..., 1], vm[0]) + broaden_segments(out[1][..., 1], vm[1])
            return F1, F2, C

        def phot_grid(n):
            """n log-uniform points over 4300-5900 A, keeping only those under the b or y response plus one zero-response
            neighbour on each side, so the trapezoid across the gaps adds exactly nothing."""
            lw_ = np.linspace(math.log10(4300.0), math.log10(5900.0), n); wl_ = 10.0 ** lw_
            rs = {nm.split(":")[1]: np.interp(wl_, np.asarray(LC["passbands"][nm][0]), np.asarray(LC["passbands"][nm][1]), left=0., right=0.) for nm in ("Stromgren:b", "Stromgren:y")}
            k = sum(rs.values()) > 0; k = k | np.roll(k, 1) | np.roll(k, -1)
            return jnp.asarray(lw_[k]), jnp.asarray(wl_[k]), {b: jnp.asarray(r[k]) for b, r in rs.items()}

        def make_phot_exact(lw_phx, wl_phx, resp_x):
            def phot_model_exact(theta):
                if args.free_geometry:
                    # SPICE's own binary (icospheres + occlusion) rebuilt from R1, R2, incl at every model phase, every
                    # element synthesised individually; spheres carry one row per star (as in the kernel free-geometry
                    # path) with log g following the radius
                    def one(th, xs):
                        rp = [cno(base_lc[s].at[iT].set(th[P[f"Teff{s+1}"]]).at[iF].set(th[P["feh"]]), s, th) for s in (0, 1)]
                        rp = [r_.at[iG].add(-2.0 * jnp.log10(rscale(th, s))) for s, r_ in enumerate(rp)]
                        m1, m2 = _evaluate_orbit(spice_binary(th[P["R1"]], th[P["R2"]], th[P["incl"]], rp), xs[0], n_neighbors1=NN1, n_neighbors2=NN2)
                        return (EMU["sim"](EMU["f"], m1, lw_phx, chunk_size=EMU["chunk"])[:, 0]
                                + EMU["sim"](EMU["f"], m2, lw_phx, chunk_size=EMU["chunk"])[:, 0])
                    flux = map_epochs(one, theta, (jnp.asarray(geo_phases) * K.PERIOD_YR,), DEV)
                else:
                    flux = 0.0
                    for s in (0, 1):
                        def one(th, xs, s=s):
                            a, mu, v, vo, p = xs
                            return exact_flux(a, mu, v, elem_params(p, s, th), lw_phx)[:, 0]
                        flux = flux + map_epochs(one, theta, EXL[s], DEV)
                out = {}
                for b, rr in resp_x.items():
                    mags = jax.vmap(lambda f: GI.passband_mag(f, wl_phx, rr))(flux)
                    mags = (mags if args.free_geometry else mags[order]); mags = mags - jnp.median(mags)
                    out[b] = band_residuals(mags, theta[P["dphi"]], b, sig_b=PSIG(theta, b))
                return out
            return phot_model_exact
        gx = phot_grid(args.exact_phot_wl); phot_model_exact = make_phot_exact(*gx)

        # sampling convergence of the exact light curves (Doppler-shifted lines crossing the sample points), and the kernel path
        t = time.time(); pe = jax.jit(phot_model_exact)(jnp.asarray(theta0)); pk = jax.jit(phot_model)(jnp.asarray(theta0))
        p2 = jax.jit(make_phot_exact(*phot_grid(2 * args.exact_phot_wl)))(jnp.asarray(theta0))
        print(f"exact light curves on {int(gx[0].size)} of {args.exact_phot_wl} wl (b/y support): vs 2x sampling "
              + " ".join(f"{b} max {1000 * float(jnp.max(jnp.abs(pe[b][0] - p2[b][0]))):.3f}" for b in pe) + " mmag; vs kernel (" + str(args.n_phot_wl) + " wl) "
              + " ".join(f"{b} max {1000 * float(jnp.max(jnp.abs(pe[b][0] - pk[b][0]))):.3f}" for b in pe) + f" mmag ({time.time() - t:.0f}s incl. compile)", flush=True)
        spectra_parts, phot_model = spectra_parts_exact, phot_model_exact       # the loss, clipping and dump below now use the exact model
        print(f"EXACT LOSS ({len(DEV) if DEV else 1} device(s)): {n_ep} epochs x {len(seg)} windows ({int(lw_ex.size)} synthesised px per star), "
              + (f"{len(geo_phases)} SPICE-binary light-curve phases (free R1, R2, incl)" if args.free_geometry else f"{n_lc} light-curve phases") + f" x {args.exact_phot_wl} wl", flush=True)
        if args.free_geometry:
            # the spectra keep the PHOEBE meshes (areas x (R/R_0)^2): an epoch whose occultation would change with the radii is flagged here
            A2 = np.asarray(EXS[1][0]).sum(1); A1 = np.asarray(EXS[0][0]).sum(1)
            print(f"spectral epochs: visible-area ratio to the median, primary {(A1 / np.median(A1)).min():.3f}-{(A1 / np.median(A1)).max():.3f}, "
                  f"secondary {(A2 / np.median(A2)).min():.3f}-{(A2 / np.median(A2)).max():.3f} (well below 1 = in eclipse: occultation not rescaled)", flush=True)

    mock_info = None
    if args.mock_ts:
        # ---- injection-recovery: the data become the exact model built from Turbospectrum intensities at --mock-truth ----
        # Same geometry, broadening, epochs, light-curve phases, masks and noise model as the real fit; only the intensities
        # differ (Turbospectrum on interpolated MARCS atmospheres instead of the emulator), so the recovered parameters
        # minus the truth are the emulator's bias on this data set.
        from ts_table_intensity import TSTableIntensity
        if not args.exact_loss: raise SystemExit("--mock-ts needs --exact-loss (the mock is the exact per-element model)")
        ts = TSTableIntensity(args.mock_ts, iT, iG); th_true = theta0.copy()
        for kv in args.mock_truth:
            nm, val = kv.split("="); assert nm in P, f"--mock-truth {nm}: not a parameter of this run ({pnames})"; th_true[P[nm]] = float(val)
        print(f"MOCK from {args.mock_ts} (TS Teff {ts.truth_teff.tolist()}, [Fe/H] {ts.truth_feh:+.2f}) at truth "
              + " ".join(f"{nm} {v:.5g}" for nm, v in zip(pnames, th_true)), flush=True)
        if abs(th_true[P["Teff1"]] - ts.truth_teff[0]) > 1 or abs(th_true[P["Teff2"]] - ts.truth_teff[1]) > 1 or abs(th_true[P["feh"]] - ts.truth_feh) > 1e-3:
            raise SystemExit("--mock-truth Teff1/Teff2/feh must match the TS tables (they were synthesised at these values only)")
        thj = jnp.asarray(th_true)
        ts.check_rows([np.concatenate([np.asarray(elem_params(jnp.asarray(EXS[s][4][e]), s, thj)) for e in range(n_ep)]) for s in (0, 1)])
        if not args.free_geometry:
            ts.check_rows([np.concatenate([np.asarray(elem_params(jnp.asarray(EXL[s][4][e]), s, thj)) for e in range(n_lc)]) for s in (0, 1)])
        t = time.time(); EMU.update(f=ts.intensity, chunk=args.mock_chunk)
        try:
            # fresh wrappers: jax.jit caches traces by function, and phot_model was already traced with the emulator above
            F1m, F2m, Cm = jax.jit(lambda t_: spectra_parts(t_))(thj); pm_mock = jax.jit(lambda t_: phot_model(t_))(thj); jax.block_until_ready(Cm)
        finally:
            EMU.update(f=emu.intensity, chunk=args.exact_chunk)
        mock = np.asarray((F1m + F2m) / Cm); print(f"mock synthesised from the TS tables in {time.time() - t:.0f}s", flush=True)
        # the emulator at the same truth, before any fitting: the mismatch the fit has to absorb
        F1e, F2e, Ce = jax.jit(lambda t_: spectra_parts(t_))(thj); pm_emu = jax.jit(lambda t_: phot_model(t_))(thj); emu_mod = np.asarray((F1e + F2e) / Ce)
        d = (emu_mod - mock)[good_np]
        if not np.any(d) or not any(float(jnp.max(jnp.abs(pm_emu[b][0] - pm_mock[b][0]))) > 0 for b in pm_mock):
            raise SystemExit("mock identical to the emulator model: a cached trace was reused")
        lr = np.asarray(F2m / F1m)[good_np] / np.asarray(F2e / F1e)[good_np] - 1.0
        print(f"emulator vs TS at the truth (fitted pixels): rms {d.std():.5f} mean {d.mean():+.5f} max {np.abs(d).max():.4f}; "
              f"spectroscopic light ratio F2/F1 emulator/TS - 1: mean {lr.mean():+.4f}", flush=True)
        print("emulator vs TS light curves at the truth: " + " ".join(
            f"{b} rms {1000 * float(jnp.std(pm_emu[b][0] - pm_mock[b][0])):.2f} max {1000 * float(jnp.max(jnp.abs(pm_emu[b][0] - pm_mock[b][0]))):.2f} mmag" for b in pm_mock), flush=True)
        rng = np.random.default_rng(args.mock_noise_seed) if args.mock_noise_seed >= 0 else None
        obs_np = np.where(np.isfinite(obs_np), mock, np.nan)                    # the data's NaN pattern (pre-masked pixels) is kept
        if rng is not None: obs_np = obs_np + rng.normal(0.0, 1.0, obs_np.shape) * sig_np
        obs = jnp.asarray(np.where(good_np, obs_np, 1.0))
        for b in mag_o:
            mag_o[b] = np.asarray(pm_mock[b][1]) + (rng.normal(0.0, sig_ph[b], np.shape(mag_o[b])) if rng is not None else 0.0)
        mock_info = dict(ts=args.mock_ts, truth=th_true, noise_seed=args.mock_noise_seed, emu_vs_ts_rms=float(d.std()), emu_vs_ts_mean=float(d.mean()),
                         lc_rms_mmag={b: 1000 * float(jnp.std(pm_emu[b][0] - pm_mock[b][0])) for b in pm_mock})
        if args.mock_out:
            np.savez(args.mock_out, logwl=lw_obs, mock=mock, emu_at_truth=emu_mod, good=good_np, sigma=sig_np, truth=th_true, pnames=np.array(pnames),
                     F1_ts=np.asarray(F1m), F2_ts=np.asarray(F2m), C_ts=np.asarray(Cm), F1_emu=np.asarray(F1e), F2_emu=np.asarray(F2e), C_emu=np.asarray(Ce),
                     **{f"lc_{b}_{k}": np.asarray(v[0]) for k, pmx in (("ts", pm_mock), ("emu", pm_emu)) for b, v in pmx.items()},
                     **{f"lc_{b}_obs_mock": np.asarray(mag_o[b]) for b in mag_o}, model_phase=np.asarray(mph))
            print(f"mock saved to {args.mock_out}", flush=True)

    if args.phot_clip > 0:
        # static outlier mask from the plain residuals (global zero point only) at the start point, or at --clip-at
        th_clip = theta0.copy()
        for kv in args.clip_at:
            nm, val = kv.split("="); assert nm in P, f"--clip-at {nm}: not a parameter of this run ({pnames})"; th_clip[P[nm]] = float(val)
        pm0 = jax.jit(phot_model)(jnp.asarray(th_clip))
        for b in ph_w:
            mags0 = pm0[b][0]; res0 = np.asarray(band_residuals(mags0, th_clip[P["dphi"]], b, nightly=False)[3])
            keep = np.abs(res0) <= args.phot_clip / 1000.0; ph_w[b] = jnp.asarray(keep.astype(float))
            print(f"phot clip {b}: dropping {int((~keep).sum())} of {keep.size} points with |O-C| > {args.phot_clip:.0f} mmag at the start point", flush=True)
        N_ph = int(sum(float(jnp.sum(w)) for w in ph_w.values()))

    def total(x, dlt, block_w):
        th = jnp.asarray(theta0) + x * jnp.asarray(S)
        if args.gp_noise or args.phot_noise_scale:
            pm = phot_model(th); c_ph = chi2_phot(th, pm)
            if args.phot_noise_scale:     # -2 ln L with sigma_b -> f_b sigma_b (nightly prior unchanged)
                l_ph = sum(jnp.sum(ph_w[b] * v[3] ** 2) / PSIG(th, b) ** 2 + 2.0 * jnp.sum(ph_w[b]) * th[P[f"phot_lnf_{b}"]]
                           + (jnp.sum((v[4] / sig_night) ** 2) if sig_night > 0 else 0.0) for b, v in pm.items())
            else:
                l_ph = c_ph
            if args.gp_noise:
                mod = spec_model(th, dlt); c_sp = chi2_spec(th, dlt, block_w, mod)
                l_sp = GP.nll(jnp.where(good, obs - mod, 0.0), jnp.stack([th[P[f"gp_lna{e}"]] for e in range(n_ep)]), th[P["gp_lnl1"]], ep_w,
                              th[P["gp_lnb"]] if args.gp_long else None, th[P["gp_lnl2"]] if args.gp_long else None)
            else:
                c_sp = chi2_spec(th, dlt, block_w); l_sp = args.spec_scale * c_sp
            return l_ph + l_sp + (delta_prior(dlt) if args.delta else 0.0), (c_ph, c_sp)
        c_ph = chi2_phot(th); c_sp = chi2_spec(th, dlt, block_w)
        return c_ph + args.spec_scale * c_sp + (delta_prior(dlt) if args.delta else 0.0), (c_ph, c_sp)
    vg_theta = jax.jit(jax.value_and_grad(total, argnums=0, has_aux=True))

    CHEAP = {}
    def cheap_vg(x, dlt, block_w):
        """value_and_grad of the same loss with the block-PCA emulator in every exact synthesis (traced once, then cached)."""
        if "vg" not in CHEAP:
            from block_pca_spice import BlockPCAIntensityEmulator, simulate_observed_flux_scan
            bundle = os.environ.get("SPICE_HARPS_BUNDLE")
            if not bundle: raise SystemExit("--cheap-block-pca needs SPICE_HARPS_BUNDLE (a local v2b bundle directory)")
            emu_c = BlockPCAIntensityEmulator(bundle, args.cheap_block_pca, args.cheap_model)
            vg_c = jax.jit(jax.value_and_grad(total, argnums=0, has_aux=True))
            # emu takes release-1 rows and inserts marcs_fe_h (snap / fixed) itself: give the cheap model the same full rows
            f_c = (lambda lw, mu, prow: emu_c.intensity(lw, mu, emu._full(prow))) if hasattr(emu, "_full") else emu_c.intensity
            EMU.update(f=f_c, chunk=args.cheap_chunk, sim=simulate_observed_flux_scan)
            try:
                t = time.time(); jax.block_until_ready(vg_c(jnp.asarray(x), dlt, block_w))
                print(f"cheap (block-PCA {args.cheap_model}) loss compiled and evaluated in {time.time() - t:.0f}s", flush=True)
            finally:
                EMU.update(f=emu.intensity, chunk=args.exact_chunk, sim=simulate_observed_flux)
            CHEAP["vg"] = vg_c
        return CHEAP["vg"]

    def multi_fidelity(f, x0, bounds, dlt, block_w):
        """Trust-region loop: L-BFGS-B on cheap(x) + [exact - cheap](x_k) + [grad exact - grad cheap](x_k).(x - x_k) inside
        |x - x_k| <= radius; accept a step only if the exact (TPayne) chi2 drops. Values are scaled by the start chi2."""
        x = np.asarray(x0, float); v0, g0_ = f(x); f0 = max(abs(v0), 1.0); fe, ge = v0 / f0, np.asarray(g0_) / f0
        f_exact = lambda y: (lambda v_, g_: (v_ / f0, np.asarray(g_) / f0))(*f(y))
        vg_c = cheap_vg(x, dlt, block_w)
        def f_cheap(y):
            (v_, _), g_ = vg_c(jnp.asarray(y), dlt, block_w); return float(v_) / f0, np.asarray(g_, float) / f0
        lo_b, hi_b = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        rad = args.mf_radius; k = 0; msg = f"outer iteration limit ({args.mf_outer})"
        vc0, _ = f_cheap(x)
        print(f"multi-fidelity start: chi2 TPayne {fe * f0:.6g}, block-PCA {vc0 * f0:.6g} (offset {(fe - vc0) * f0:+.4g})", flush=True)
        for k in range(1, args.mf_outer + 1):
            t = time.time(); vc, gc = f_cheap(x); a, b = fe - vc, ge - gc
            def f_sur(y, a=a, b=b, xk=x):
                v_, g_ = f_cheap(y); return v_ + a + float(b @ (y - xk)), g_ + b
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
        return SimpleNamespace(x=x, nit=k, fun=fe * f0, message=f"multi-fidelity: {msg}")
    vg_delta = jax.jit(jax.value_and_grad(total, argnums=1, has_aux=True))
    blocks_all = jnp.ones(edges.size - 1)

    # ---- validation against exact per-element synthesis (epoch 0, theta0, vsini 1) ----
    if not args.skip_validation and not args.exact_loss:
        from spice.spectrum.spectrum import simulate_observed_flux
        th = jnp.asarray(theta0); r = rows(th); m1, m2 = SP["models"][0]; vm = list(args.vmacro0)
        def exact(mm, rr, s):
            p = jnp.asarray(mm.parameters); teff = th[P[f"Teff{s+1}"]] * jnp.power(10.0, args.beta * (p[:, iG] - gref[s]))
            return simulate_observed_flux(emu.intensity, mm._replace(parameters=cno(p.at[:, iT].set(teff).at[:, iF].set(th[P["feh"]]), s, th)), lw_star[s], chunk_size=256)
        t = time.time(); e1 = exact(m1, r[0], 0); e2 = exact(m2, r[1], 1)
        ex = np.asarray((broaden(e1[None, :, 0], vm[0]) + broaden(e2[None, :, 0], vm[1])) / (broaden(e1[None, :, 1], vm[0]) + broaden(e2[None, :, 1], vm[1])))[0]
        su = np.asarray(jax.jit(lambda t_: spectra(t_, jnp.zeros(n))[0])(th)); g0 = good_np[0]
        d = np.abs(su - ex)[g0]; print(f"kernel vs exact (epoch 0, full range, {n} px): max {d.max():.5f} rms {np.sqrt((d ** 2).mean()):.6f}  ({time.time() - t:.0f}s)", flush=True)
        if args.validation_out:
            e1n, e2n = np.asarray(e1), np.asarray(e2)
            kk = fixed_kernels or [kernels_for(0, VS_FIX[0]), kernels_for(1, VS_FIX[1])]
            k1 = np.asarray(kernel_flux_multi(emu.intensity, kk[0][:1], r[0], wavelength_chunk_size=args.chunk, devices=DEV))[0]
            k2 = np.asarray(kernel_flux_multi(emu.intensity, kk[1][:1], r[1], wavelength_chunk_size=args.chunk, devices=DEV))[0]
            np.savez(args.validation_out, logwl=np.asarray(lw_model), exact=ex, kernel=su, good=g0, n_mu=args.n_mu, oversample=args.oversample,
                     exact_line1=e1n[:, 0], exact_cont1=e1n[:, 1], exact_line2=e2n[:, 0], exact_cont2=e2n[:, 1],
                     kernel_line1=k1[:, 0], kernel_cont1=k1[:, 1], kernel_line2=k2[:, 0], kernel_cont2=k2[:, 1], **KOPT)
            print(f"validation spectra saved to {args.validation_out}", flush=True)
        if d.max() > 0.01: print("ABORT: kernel model outside tolerance"); sys.exit(2)

    # ---- optimisation ----
    x0 = np.zeros(n_par); dlt = jnp.asarray(delta0)
    t = time.time(); (v, aux), g = vg_theta(jnp.asarray(x0), dlt, blocks_all); jax.block_until_ready(g); print(f"first value+grad (compile) {time.time() - t:.0f}s", flush=True)
    if not args.exact_loss:      # in --exact-loss one evaluation costs as much as a fit step: no separate warm timing
        t = time.time(); jax.block_until_ready(vg_theta(jnp.asarray(x0), dlt, blocks_all)[1]); print(f"warm value+grad {time.time() - t:.1f}s", flush=True)
    print(f"start: chi2_phot/N {float(aux[0]) / N_ph:.3f}  chi2_spec/N {float(aux[1]) / N_sp:.3f}", flush=True)
    bounds = list(zip((np.array(lo) - theta0) / S, (np.array(hi) - theta0) / S))
    if args.noise_check:
        fval = lambda x_: float(vg_theta(jnp.asarray(x_), dlt, blocks_all)[0][0])
        rep = [fval(x0) for _ in range(3)]; print(f"noise check: 3 repeat evaluations {rep} -> spread {max(rep) - min(rep):.3g} (chi2 total {rep[0]:.6g})", flush=True)
        for i, nm in enumerate(pnames):
            for h in (1e-3, 1e-2, 1e-1):
                xp = x0.copy(); xp[i] += h; xm = x0.copy(); xm[i] -= h; fp, fm = fval(xp), fval(xm)
                print(f"  {nm:12s} h={h:<5g} f(+h)-f(-h) = {fp - fm:+11.4g}   FD grad {(fp - fm) / (2 * h):+12.6g}   AD grad {float(g[i]):+12.6g}", flush=True)

    def fit_theta(x_start, dlt, block_w, maxiter, label):
        hist = []
        prog = (args.out[:-4] if args.out.endswith(".pkl") else args.out) + "_progress.pkl"
        def f(x):
            te = time.time(); (v, aux_), g = vg_theta(jnp.asarray(x), dlt, block_w)
            if args.exact_loss:
                th_ = theta0 + np.asarray(x) * S
                if not hist or float(v) < min(hist):      # best point so far: a --result-in warm start if the job dies
                    pickle.dump(dict(pnames=pnames, theta=th_, chi2=float(v), n_eval=len(hist) + 1, label=label), open(prog, "wb"), protocol=4)
                print(f"  eval {len(hist) + 1}: chi2 {float(v):.6g} (phot/N {float(aux_[0]) / N_ph:.3f} spec/N {float(aux_[1]) / N_sp:.3f}) {time.time() - te:.0f}s | "
                      + " ".join(f"{nm} {val:.5g}" for nm, val in zip(pnames, th_)), flush=True)
            hist.append(float(v)); return float(v), np.asarray(g, float)
        # L-BFGS-B stalls ("ABNORMAL", 0-1 iterations) when the start is already within the float32 roundoff of the
        # objective (~10 units of chi2 out of 1e7): the achievable decrease along the first steepest-descent line is below
        # the noise. Restart from a perturbed point (0.3 scaled units: 30 K, 0.03 dex, 0.3 km/s) so the fit re-enters the
        # basin with real decreases and settles to the noise-limited optimum. Keep the best point seen.
        t = time.time(); x_s = np.asarray(x_start, float); best = None; rng = np.random.default_rng(0); tries = 0; nit = 0
        lo_b = np.array([b[0] for b in bounds]); hi_b = np.array([b[1] for b in bounds])
        if maxiter == 0:      # evaluation only: L-BFGS-B takes one line-search step even at maxiter=0
            best = SimpleNamespace(x=x_s, nit=0, message="no optimisation (maxiter 0)")
        elif args.cheap_block_pca:
            best = multi_fidelity(f, x_s, bounds, dlt, block_w); nit = best.nit
        while maxiter > 0 and not args.cheap_block_pca:
            r = minimize(f, x_s, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter)); nit += r.nit
            if best is None or r.fun < best.fun: best = r
            stalled = r.nit <= 1 and "ABNORMAL" in r.message
            if not stalled or tries >= args.stall_retries: break
            tries += 1; x_s = np.clip(np.asarray(best.x) + rng.normal(0.0, 0.3, size=x_s.size), lo_b, hi_b)
            print(f"  {label}: stalled ({r.message[:30].strip()}) -> retry {tries} from a perturbed start", flush=True)
        r = best; r.nit = nit
        (_, aux), _ = vg_theta(jnp.asarray(r.x), dlt, block_w); th = theta0 + np.asarray(r.x) * S
        print(f"{label}: {r.nit} it, {len(hist)} eval, {time.time() - t:.0f}s, {r.message[:30]} | chi2_phot/N {float(aux[0]) / N_ph:.3f} chi2_spec/N {float(aux[1]) / N_sp:.3f} | "
              + " ".join(f"{nm} {val:.4g}" for nm, val in zip(pnames, th)), flush=True)
        return np.asarray(r.x), th, aux

    def fit_delta(x, dlt, maxiter):
        def f(d):
            (v, _), g = vg_delta(jnp.asarray(x), jnp.asarray(d), blocks_all); return float(v), np.asarray(g, float)
        t = time.time(); r = minimize(f, np.asarray(dlt), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
        d = np.asarray(r.x); print(f"delta fit: {r.nit} it, {time.time() - t:.0f}s, rms delta {100 * d.std():.2f}% (prior {100 * args.delta_sigma:.0f}%)", flush=True)
        return jnp.asarray(d)

    if args.delta and args.delta_mode == "fixed":
        dlt = jnp.asarray(pickle.load(open(args.delta_in, "rb"))["delta"]); print(f"delta held fixed from {args.delta_in} (rms {100 * float(jnp.std(dlt)):.2f}%)", flush=True)
    if args.delta and args.delta_mode == "fit-only":
        dlt = fit_delta(x0, dlt, args.delta_maxiter); th = theta0.copy(); x = x0
        (_, aux), _ = vg_theta(jnp.asarray(x0), dlt, blocks_all); th_nodelta = th.copy()
        print("delta fitted at the warm-start theta; theta not refitted (use delta-mode fixed on the other epoch subset)", flush=True)
    else:
        x, th, aux = fit_theta(x0, dlt, blocks_all, args.maxiter, "pass 1" + (" (delta fixed)" if args.delta and args.delta_mode == "fixed" else " (delta = 0)"))
        th_nodelta = th.copy()
        if args.delta and args.delta_mode == "joint":
            for it in range(args.delta_alternations):
                dlt = fit_delta(x, dlt, args.delta_maxiter)
                x, th, aux = fit_theta(x, dlt, blocks_all, 40, f"alternation {it + 1}: theta | delta")
            print("delta effect on theta: " + " ".join(f"{nm} {b - a:+.4g}" for nm, a, b in zip(pnames, th_nodelta, th)), flush=True)

    result = dict(pnames=pnames, theta=th, theta_nodelta=th_nodelta, x=x, S=S, theta0=theta0, chi2_phot=float(aux[0]), chi2_spec=float(aux[1]), N_ph=N_ph, N_sp=N_sp,
                  phot_clip=args.phot_clip, phot_nightly_sigma=args.phot_nightly_sigma, phot_weights={b: np.asarray(w) for b, w in ph_w.items()},
                  defect_in=args.defect_in, defect_ref=args.defect_ref,
                  kernel=dict(n_mu=args.n_mu, n_g=args.n_g, oversample=args.oversample, **KOPT),
                  delta=(np.asarray(dlt) if args.delta else None), args=vars(args), exact_loss=args.exact_loss, exact_windows=args.exact_windows, rv=rv, fit_epochs=fit_e, beta=args.beta, delta_mode=args.delta_mode,
                  mock=mock_info)
    pickle.dump(result, open(args.out, "wb"), protocol=4)

    # ---- model dump for figures (optional) ----
    if args.dump_model and args.exact_dump and not args.exact_loss:
        # ---- the same model, evaluated element by element instead of through the kernels ----
        # Every element gets its own Teff (gravity darkening), the fitted vsini scale is applied to the element
        # velocities about the orbital mean (exactly what the surrogate does to its kernels), and the result goes
        # through the same macroturbulent broadening and continuum fit. This is the path the surrogate was
        # validated against; here it produces the figures themselves.
        from spice.spectrum.spectrum import simulate_observed_flux
        assert not args.free_geometry, "exact dump is for the PHOEBE-mesh geometry"
        thj = jnp.asarray(th)
        vs = [th[P["vsini_scale1"]], th[P["vsini_scale2"]]] if args.free_vsini else VS_FIX
        vm = [th[P["vmac1"]], th[P["vmac2"]]] if args.free_vmacro else list(args.vmacro0)
        ifn = [intensity_with_delta(dlt), emu.intensity]

        def exact_star(mm, s, lw, teff, feh_, scale):
            p = jnp.asarray(mm.parameters)
            t_el = p[:, iT] if args.mesh_teff else teff * jnp.power(10.0, args.beta * (p[:, iG] - gref[s]))
            cv = jnp.asarray(mm.center_velocities); a = np.asarray(mm.visible_cast_areas); mu = np.asarray(mm.mus)
            vis = (mu > 0) & (a > 0); cvm = jnp.asarray(np.average(np.asarray(cv)[vis], axis=0, weights=a[vis]))
            mm2 = mm._replace(parameters=cno(p.at[:, iT].set(t_el).at[:, iF].set(feh_), s, thj), center_velocities=cvm + scale * (cv - cvm))
            return simulate_observed_flux(ifn[s], mm2, lw, chunk_size=256)                       # (n, 2): line, continuum

        if args.mesh_teff:
            for s_ in (0, 1):
                tt_ = np.concatenate([np.asarray(m[s_].parameters)[:, iT][np.asarray(m[s_].mus) > 0] for m in SP["models"]])
                print(f"  --mesh-teff: star {s_ + 1} uses the meshes' own Teff map, visible elements {tt_.min():.1f}-{tt_.max():.1f} K "
                      f"(mean {tt_.mean():.1f}); Teff{s_ + 1} = {th[P[f'Teff{s_ + 1}']]:.0f} only enters the in-fit kernel path", flush=True)
        t0_ = time.time(); F1, F2, C = [], [], []
        for e_ in range(n_ep):
            m1_, m2_ = SP["models"][e_]
            o1 = exact_star(m1_, 0, lw_star[0], thj[P["Teff1"]], thj[P["feh"]], vs[0]); o2 = exact_star(m2_, 1, lw_star[1], thj[P["Teff2"]], thj[P["feh"]], vs[1])
            F1.append(broaden(o1[None, :, 0], vm[0])[0]); F2.append(broaden(o2[None, :, 0], vm[1])[0])
            C.append(broaden(o1[None, :, 1], vm[0])[0] + broaden(o2[None, :, 1], vm[1])[0])
            print(f"  exact spectra: epoch {e_ + 1}/{n_ep} ({time.time() - t0_:.0f}s)", flush=True)
        F1, F2, C = jnp.stack(F1), jnp.stack(F2), jnp.stack(C)
        raw = (F1 + F2) / C; fixed = jax.vmap(continuum_fix)(raw, obs, good); corr = fixed / raw
        mod, prim, sec = (np.asarray(a_) for a_ in (fixed, F1 / C * corr, F2 / C * corr))
        # kernel surrogate at the same theta, for the record
        F1k, F2k, Ck = spectra_parts(thj, dlt); mk = np.asarray(jax.vmap(continuum_fix)((F1k + F2k) / Ck, obs, good))
        dk = np.abs(mod - mk)[good_np]; print(f"exact vs kernel over all fitted pixels: max {dk.max():.5f} rms {np.sqrt((dk ** 2).mean()):.6f}", flush=True)

        # light curves, element by element on the PHOEBE light-curve meshes
        t0_ = time.time(); flux = []
        for j_, pair in enumerate(LC["models"]):
            f_ = 0.0
            for s in (0, 1):
                p_ = jnp.asarray(pair[s].parameters)
                t_ = p_[:, iT] if args.mesh_teff else thj[P[f"Teff{s + 1}"]] * jnp.power(10.0, args.beta * (p_[:, iG] - gref[s]))
                o_ = simulate_observed_flux(emu.intensity, pair[s]._replace(parameters=cno(p_.at[:, iT].set(t_).at[:, iF].set(thj[P["feh"]]), s, thj)), lw_ph, chunk_size=256)
                f_ = f_ + o_[:, 0]
            flux.append(f_)
        flux = jnp.stack(flux); print(f"  exact light curves: {len(LC['models'])} phases ({time.time() - t0_:.0f}s)", flush=True)
        pm = {}
        for b, rr in resp.items():
            mags = jax.vmap(lambda f: GI.passband_mag(f, wl_ph, rr))(flux)[order]; mags = mags - jnp.median(mags)
            pm[b] = band_residuals(mags, thj[P["dphi"]], b, sig_b=PSIG(thj, b))
        pk = jax.jit(phot_model)(thj)
        for b in pm: print(f"  exact vs kernel light curve {b}: max {1000 * float(jnp.max(jnp.abs(pm[b][0] - pk[b][0]))):.3f} mmag", flush=True)
        # the fit's own chi2 (unit block weights), scored on the exact model
        r_ = jnp.where(good, (obs - jnp.asarray(mod)) / sig, 0.0); c_sp_ex = float(jnp.sum(ep_w * jnp.sum(r_ ** 2, axis=1)))
        c_ph_ex = float(sum(jnp.sum(ph_w[b] * v[3] ** 2) / sig_ph[b] ** 2 + (jnp.sum((v[4] / sig_night) ** 2) if sig_night > 0 else 0.0) for b, v in pm.items()))
        print(f"exact model: chi2_phot/N {c_ph_ex / N_ph:.3f} (" + " ".join(f"{b} rms {1000 * float(jnp.sqrt(jnp.sum(ph_w[b] * v[3] ** 2) / jnp.sum(ph_w[b]))):.2f} mmag" for b, v in pm.items())
              + f")  chi2_spec/N {c_sp_ex / N_sp:.3f}   [kernel at the same theta: {float(aux[0]) / N_ph:.3f} / {float(aux[1]) / N_sp:.3f}]", flush=True)
        lc = dict(chi2_phot_exact=c_ph_ex, chi2_spec_exact=c_sp_ex, N_ph=N_ph, N_sp=N_sp)
    elif args.dump_model:
        def _parts(t_, d_):
            F1, F2, C = spectra_parts(t_, d_); raw = (F1 + F2) / C; fixed = jax.vmap(continuum_fix)(raw, obs, good); corr = fixed / raw
            return fixed + DEF_SPEC, F1 / C * corr, F2 / C * corr
        mod, prim, sec = (np.asarray(a) for a in jax.jit(_parts)(jnp.asarray(th), dlt))
        pm = jax.jit(phot_model)(jnp.asarray(th)); lc = {}
    if args.dump_model:
        for b, v in pm.items():
            mags, mi, mo, res, z = v
            lc[f"lc_{b}_model_phase"] = np.asarray(mph); lc[f"lc_{b}_model_mag"] = np.asarray(mags); lc[f"lc_{b}_obs_phase"] = np.asarray((ph_o - th[P["dphi"]]) % 1.0)
            lc[f"lc_{b}_obs_mag"] = np.asarray(mo); lc[f"lc_{b}_model_at_obs"] = np.asarray(mi); lc[f"lc_{b}_sigma"] = sig_ph[b]
            lc[f"lc_{b}_weight"] = np.asarray(ph_w[b]); lc[f"lc_{b}_night_offset"] = np.asarray(z)[np.asarray(night_idx)]
        np.savez(args.dump_model, logwl=lw_obs, obs=np.where(np.isfinite(obs_np), obs_np, np.nan), model=mod, primary=prim, secondary=sec, good=good_np, masked=np.isfinite(obs_np) & ~good_np, sigma=sig_np, rv=rv, times=np.asarray(SP["times"]),
                 fit_epochs=fit_e, theta=th, pnames=np.array(pnames), delta=(np.asarray(dlt) if args.delta else np.zeros(0)), block_edges=edges,
                 synthesis=("exact per-element" if (args.exact_dump or args.exact_loss) else ("kernel + exact defect correction" if args.defect_in else "kernel surrogate")), **lc)
        print(f"model dump ({'exact per-element' if (args.exact_dump or args.exact_loss) else 'kernel surrogate'}): {mod.shape} spectra + {len(pm)} light curves -> {args.dump_model}", flush=True)

    # ---- residual stack -> learned mask (optional) ----
    if args.derive_mask_out:
        mod = np.asarray(jax.jit(lambda t_, d_: jax.vmap(continuum_fix)(spectra(t_, d_), obs, good))(jnp.asarray(th), dlt))
        res = np.where(good_np, obs_np - mod, np.nan); me = epoch_subset(n_ep, args.mask_epochs)
        def stack(rvs):
            Sx = np.full((len(me), n), np.nan)
            for j, e in enumerate(me):
                Sx[j] = np.interp(lw_obs + math.log10(1.0 + rvs[e] / C_KMS), lw_obs, res[e], left=np.nan, right=np.nan)
            med = np.nanmedian(Sx, 0); mad = 1.4826 * np.nanmedian(np.abs(Sx - med), 0); cnt = np.isfinite(Sx).sum(0)
            bad = (np.abs(med) > args.mask_threshold) & (np.abs(med) > args.mask_nsigma * mad / np.sqrt(np.maximum(cnt, 1)))
            B = np.zeros(n, bool)
            for i in np.where(bad)[0]: B[max(0, i - args.mask_dilate):i + args.mask_dilate + 1] = True
            return B
        B1, B2 = stack(rv[0]), stack(rv[1]); mask = np.zeros((n_ep, n), bool)
        for e in range(n_ep):
            for B, rvs in ((B1, rv[0]), (B2, rv[1])):
                mask[e] |= np.interp(lw_obs - math.log10(1.0 + rvs[e] / C_KMS), lw_obs, B.astype(float), left=0, right=0) > 0.5
        rb = np.sqrt(np.nanmean(res ** 2)); ra = np.sqrt(np.nanmean(np.where(mask, np.nan, res) ** 2))
        np.savez(args.derive_mask_out, mask=mask, threshold=args.mask_threshold, mask_epochs=me, rv=rv)
        print(f"mask derived from {args.mask_epochs} epochs: primary-frame bad px {B1.sum()}, secondary {B2.sum()}; masks {100 * mask[good_np].mean():.1f}% of usable pixels; residual rms {100 * rb:.2f}% -> {100 * ra:.2f}%; saved {args.derive_mask_out}", flush=True)

    # ---- block jackknife ----
    if args.jackknife_blocks:
        nb = edges.size - 1; per = nb // args.jackknife_blocks; jk = []
        for j in range(args.jackknife_blocks):
            bw = np.ones(nb); bw[j * per:(j + 1) * per] = 0.0
            _, thj, _ = fit_theta(x, dlt, jnp.asarray(bw), 40, f"drop block {j + 1}/{args.jackknife_blocks} ({edges[j * per]:.0f}-{edges[min((j + 1) * per, nb)]:.0f} A)")
            jk.append(thj)
        A = np.array(jk); m = A.shape[0]; err = np.sqrt((m - 1) / m * np.sum((A - A.mean(0)) ** 2, 0))
        print("jackknife error: " + " ".join(f"{nm} {e_:.3g}" for nm, e_ in zip(pnames, err)), flush=True)
        result["jackknife"] = A; result["jackknife_err"] = err
        pickle.dump(result, open(args.out, "wb"), protocol=4)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
