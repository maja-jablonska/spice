"""delta Cep: gradient fit of the phase-dependent temperature curve through SynthesisKernel.

The delta Cep analogue of tz_fornacis/tzfor_kernel_fit.py. Geometry is held
fixed (SPIPS radius curve turned into the l=0 pulsation template, 12 km/s
rotation, SPIPS distance) and the atmospheric side is differentiable:

    theta = [Teff Fourier coefficients (T0, a_k, b_k for k=1..K),
             E(B-V), [Fe/H], dphi]

Photometry: Johnson V and B-V from delta_cep.fits (SPIPS data table, all
sources), optionally Hipparcos Hp and Tycho B_T/V_T. Synthetic magnitudes use
the Vega zero points stored by SPIPS in the FITS header (ZP_WM2UM), the
SPIPS distance, and a CCM89 (R_V = 3.1) reddening applied to the spectrum
before the passband integral -- so E(B-V) is a physical parameter, not a
zero-point nuisance. NOTE the SPIPS run stored in delta_cep.fits has
E(B-V) = 0 frozen and its MODEL column sits ~0.25 mag brighter than the V
data; that is A_V for E(B-V) ~ 0.08, which this fit recovers.

Cost model. A Cepheid's Teff changes every epoch, so the TZ For trick of one
emulator evaluation per (mu, log g) node shared by all epochs does not apply
directly. Instead the emulator is evaluated once per (Teff node, mu node) on a
grid of Teff nodes spanning the pulsation range, and each epoch's spectrum is
the linear interpolation between the two neighbouring Teff nodes at that
epoch's Teff(phase). Everything downstream of the emulator (interpolation
weights, reddening, passbands, phase interpolation) is cheap, so with [Fe/H]
frozen the node spectra are computed ONCE and the fit runs in seconds on a
laptop; --free-feh puts the emulator back inside the differentiated graph
(GPU territory, as for TZ For).

Both the kernel path and the Teff-node interpolation are checked against exact
per-element synthesis (simulate_observed_flux) at the start unless
--skip-validation is given. The approximations of holding log g and vmicro at
their phase means are quantified in the same check.

Spectroscopy: the per-epoch kernels already carry the pulsation + rotation
velocity field, so a HARPS-N term (delta Cep is at +58 deg: HARPS never saw
it; Nardetto et al. 2017 have 103 HARPS-N spectra) plugs in exactly as the
HARPS term of tzfor_kernel_fit.py. No observed delta Cep spectra are in this
repository yet, so this script is photometry-only.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
if sys.platform == "darwin" and "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "src"
for p in (str(SRC), str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import jax
import jax.numpy as jnp
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import minimize

import emulator_params as ep
from cepheid_bundles import build_bundle, apply_phase_params
from build_cepheid_grid import make_phase_modifiers, _phase_modifiers_for
from cepheid_phase import radius_template_mean_normalised  # noqa: F401  (re-exported for dcep_joint_fit)
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, kernel_flux_multi

C_KMS = 299792.458
PC_PER_KPC = 1000.0

# SPIPS data-table band name -> (spice filter class name, header ZP key)
BANDS = {
    "V":  ("JohnsonCousinsV", "V_GCPD_Johnson"),
    "B":  ("JohnsonCousinsB", "B_GCPD_Johnson"),
    "Hp": ("HipparcosHp",     "Hp_MvB_Hipparcos"),
    "BT": ("TychoBT",         "B_MvB_TYCHO"),
    "VT": ("TychoVT",         "V_MvB_TYCHO"),
}
OBS_BAND = {"V_GCPD_Johnson": "V", "Hp_MvB_Hipparcos": "Hp", "B_MvB_TYCHO": "BT", "V_MvB_TYCHO": "VT"}
OBS_COLOR = {"B_GCPD_Johnson-V_GCPD_Johnson": ("B", "V")}


# ---------------------------------------------------------------------------
# Small physics helpers
# ---------------------------------------------------------------------------

def alambda_over_av(wl_angstrom, r_v=3.1, table=None, law="ccm89"):
    """A_lambda / A_V for the law 'ccm89' (built in), 'f99' (Fitzpatrick 1999) or 'g23' (Gordon et al. 2023, the most
    recent R_V-dependent Milky Way average; both via the dust_extinction package), or interpolated from a table
    (columns wavelength [A], A_lambda / A_V; a '# R_V <value>' header line must match r_v)."""
    if table is None:
        if law == "ccm89":
            return ccm89_alambda_over_av(wl_angstrom, r_v)
        import astropy.units as u
        from dust_extinction.parameter_averages import F99, G23
        model = {"f99": F99, "g23": G23}[law](Rv=r_v)
        return np.asarray(model(np.asarray(wl_angstrom, float) * u.AA), float)
    rv_tab = None
    for line in open(table):
        if line.startswith("#") and "R_V" in line:
            rv_tab = float(line.split("R_V")[1].split()[0])
    if rv_tab is None or abs(rv_tab - r_v) > 1e-6:
        raise SystemExit(f"extinction table {table}: R_V {rv_tab} does not match the SPIPS R_V {r_v}")
    t = np.loadtxt(table)
    wl = np.asarray(wl_angstrom, float)
    if wl.min() < t[0, 0] or wl.max() > t[-1, 0]:
        raise SystemExit(f"extinction table {table} covers {t[0, 0]:.0f}-{t[-1, 0]:.0f} A, not {wl.min():.0f}-{wl.max():.0f} A")
    return np.interp(wl, t[:, 0], t[:, 1])


def ccm89_alambda_over_av(wl_angstrom, r_v=3.1):
    """Cardelli, Clayton & Mathis (1989) A_lambda / A_V, optical + near-IR branches."""
    x = 1e4 / np.asarray(wl_angstrom, float)          # 1/micron
    a = np.empty_like(x); b = np.empty_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61; b[ir] = -0.527 * x[ir] ** 1.61
    y = x[~ir] - 1.82
    a[~ir] = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4
              + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7)
    b[~ir] = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4
              - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7)
    return a + b / r_v


def fourier_eval(phase, coef):
    """coef = [T0, a1, b1, a2, b2, ...]; phase in cycles."""
    out = coef[0]
    k = (coef.shape[0] - 1) // 2
    for i in range(k):
        w = 2.0 * jnp.pi * (i + 1) * phase
        out = out + coef[1 + 2 * i] * jnp.cos(w) + coef[2 + 2 * i] * jnp.sin(w)
    return out


def fourier_fit(phase, y, k):
    """Least-squares Fourier coefficients of a periodic curve."""
    cols = [np.ones_like(phase)]
    for i in range(1, k + 1):
        cols += [np.cos(2 * np.pi * i * phase), np.sin(2 * np.pi * i * phase)]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, A @ coef


def periodic_interp(x, xp, fp):
    """Linear interpolation on a periodic grid xp in [0, 1) (sorted), x in [0, 1)."""
    n = xp.shape[0]
    xp_ext = jnp.concatenate([xp, xp[:1] + 1.0]); fp_ext = jnp.concatenate([fp, fp[:1]])
    return jnp.interp(x, xp_ext, fp_ext)


def band_mag(flux_erg_cm2_A, wl_A, response, zp_erg_cm2_A):
    """Vega magnitude, energy-weighted band average against the SPIPS zero point."""
    num = jnp.trapezoid(flux_erg_cm2_A * response, wl_A)
    den = jnp.trapezoid(response, wl_A)
    return -2.5 * jnp.log10(num / den / zp_erg_cm2_A)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_spips(fits_path):
    with fits.open(fits_path, ignore_missing_simple=True) as h:
        hdr = h[0].header
        data = Table(h[1].data)
        model = Table(h[2].data)
    zps = {}
    for key in hdr:
        if key.startswith("MODEL ZP_WM2UM "):
            zps[key.split("MODEL ZP_WM2UM ")[1]] = float(hdr[key]) * 0.1   # W/m^2/um -> erg/s/cm^2/A
    meta = dict(period=float(hdr["PARAM PERIOD"]), mjd0=float(hdr["PARAM MJD0"]),
                d_kpc=float(hdr["PARAM d_kpc"]), ebv_spips=float(hdr["PARAM E(B-V)"]),
                metal=float(hdr["PARAM METAL"]), pfactor=float(hdr["PARAM P-FACTOR"]),
                r_v=float(hdr.get("MODEL Rv", 3.1)))
    return data, model, zps, meta


def check_band_coverage(bands, wl_range, allow):
    """Refuse passbands with response outside the synthesised range (the truncated magnitudes are biased)."""
    from spice.spectrum import filter as spfilter
    wl = np.linspace(1000.0, 30000.0, 290001)
    for b in bands:
        r = np.asarray(getattr(spfilter, BANDS[b][0])().filter_responses_for_wavelengths(wl), float)
        on = wl[r > 1e-4 * r.max()]
        if on.min() < wl_range[0] or on.max() > wl_range[1]:
            msg = f"band {b} spans {on.min():.0f}-{on.max():.0f} A, outside the synthesised {wl_range[0]:.0f}-{wl_range[1]:.0f} A"
            if not allow:
                raise SystemExit(msg + " (use bands fully inside the range, e.g. VT, or --allow-partial-bands)")
            print("WARNING:", msg, flush=True)


def photometry_from_table(data, bands):
    """Observed magnitudes / colours per band key, with per-source labels."""
    obs = {}
    for row in data:
        kind_src = str(row["OBS"]).split("|")[0]
        kind = kind_src.split(";")[0].strip()
        src = kind_src.split(";")[1].strip() if ";" in kind_src else ""
        if "|" not in str(row["OBS"]) or row["MJD"] is None or not np.isfinite(row["MJD"]):
            continue
        band = str(row["OBS"]).split("|")[1].strip()
        if kind == "mag" and band in OBS_BAND and OBS_BAND[band] in bands:
            key = OBS_BAND[band]
        elif kind == "color" and band in OBS_COLOR and all(b in bands for b in OBS_COLOR[band]):
            key = "-".join(OBS_COLOR[band])
        else:
            continue
        obs.setdefault(key, []).append((float(row["PHASE"]), float(row["MEAS"]), float(row["ERR"]), src))
    out = {}
    for key, rows in obs.items():
        ph, m, e, src = zip(*rows)
        out[key] = dict(phase=np.asarray(ph), mag=np.asarray(m), err=np.asarray(e), src=np.asarray(src))
    return out


def teff_observations(data):
    rows = [(float(r["PHASE"]), float(r["MEAS"]), float(r["ERR"])) for r in data
            if str(r["OBS"]).split(";")[0].strip() == "teff"]
    if not rows:
        return None
    ph, t, e = map(np.asarray, zip(*rows))
    return dict(phase=ph, teff=t, err=e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=str(HERE / "delta_cep.fits"))
    ap.add_argument("--out", default=str(HERE / "dcep_fit_out" / "dcep_kernel_fit_result.pkl"))
    ap.add_argument("--bundle", default="RozanskiT/TPayne-spice-harps")
    ap.add_argument("--n-mesh", type=int, default=5000)
    ap.add_argument("--n-phases", type=int, default=40, help="model epochs, evenly spaced in phase")
    ap.add_argument("--n-wl", type=int, default=16000, help="log-uniform wavelength samples across the passbands")
    ap.add_argument("--wl-range", type=float, nargs=2, default=(3780.0, 6910.0),
                    help="the TPayne-spice-harps bundle extrapolates nonsense outside 3780-6910 A; passbands are truncated and renormalised")
    ap.add_argument("--n-mu", type=int, default=32)
    ap.add_argument("--n-teff", type=int, default=16, help="Teff interpolation nodes across the pulsation range")
    ap.add_argument("--teff-pad", type=float, default=200.0, help="node range margin beyond the SPIPS Teff curve [K]")
    ap.add_argument("--fourier-order", type=int, default=4, help="harmonics in Teff(phase)")
    ap.add_argument("--bands", nargs="+", default=["VT"],
                    help="subset of V B Hp BT VT (B enters via B-V); each passband must lie inside --wl-range (only VT does for HARPS)")
    ap.add_argument("--allow-partial-bands", action="store_true",
                    help="accept bands that extend beyond --wl-range (truncated synthesis; for reproducing older runs only)")
    ap.add_argument("--radius", type=float, default=None, help="mean radius [R_sun]; default: SPIPS mean R (harmonic template is mean-normalised)")
    ap.add_argument("--mass", type=float, default=5.26)
    ap.add_argument("--vrot", type=float, default=12.0, help="rotation velocity [km/s]")
    ap.add_argument("--feh", type=float, default=None, help="starting [Fe/H]; default: SPIPS METAL")
    ap.add_argument("--vmicro", type=float, default=2.0, help="fixed microturbulence [km/s]; keep it on a MARCS node (1, 2, 5), see node_fluxes")
    ap.add_argument("--free-feh", action="store_true", help="fit [Fe/H] (emulator inside the gradient; slow on CPU)")
    ap.add_argument("--ebv0", type=float, default=0.092, help="starting E(B-V) (Trahin et al. 2021 SPIPS value for delta Cep, 0.092 +- 0.021)")
    ap.add_argument("--fix-ebv", action="store_true")
    ap.add_argument("--fix-teff-shape", action="store_true", help="fit only the mean Teff, keep the SPIPS shape")
    ap.add_argument("--distance-kpc", type=float, default=None, help="default: SPIPS d_kpc")
    ap.add_argument("--err-floor", type=float, default=0.0, help="added in quadrature to the photometric errors [mag]")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--check-grad", action="store_true", help="compare autodiff with central finite differences at the start")
    ap.add_argument("--seed-from", default=None, help="warm start theta from a previous result pickle")
    args = ap.parse_args()
    print("devices:", jax.devices(), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ---- SPIPS inputs ----
    data, model, zps, meta = load_spips(args.fits)
    period = meta["period"]; d_pc = (args.distance_kpc or meta["d_kpc"]) * PC_PER_KPC
    feh0 = args.feh if args.feh is not None else meta["metal"]
    print(f"SPIPS: P {period:.6f} d, d {d_pc:.0f} pc, [Fe/H] {feh0:+.2f}, p {meta['pfactor']:.4f}, "
          f"E(B-V) frozen at {meta['ebv_spips']:.3f} in the stored run; vmicro fixed at {args.vmicro:.1f} km/s", flush=True)
    check_band_coverage(args.bands, args.wl_range, args.allow_partial_bands)
    obs = photometry_from_table(data, set(args.bands))
    for k, v in obs.items():
        print(f"  {k:4s}: {v['mag'].size:4d} points, median err {np.median(v['err']):.3f}, sources {sorted(set(v['src']))}")
    teff_obs = teff_observations(data)
    N_ph = sum(v["mag"].size for v in obs.values())

    # ---- Teff(phase) start: Fourier fit of the SPIPS curve ----
    mph = np.asarray(model["PHASE"], float); mteff = np.asarray(model["Teff"], float)
    o = np.argsort(mph); mph, mteff = mph[o], mteff[o]
    K = args.fourier_order
    coef0, fit0 = fourier_fit(mph, mteff, K)
    print(f"Teff(phase) Fourier order {K}: rms vs SPIPS spline {np.sqrt(np.mean((fit0 - mteff) ** 2)):.1f} K, "
          f"range {mteff.min():.0f}-{mteff.max():.0f} K", flush=True)
    logg_mean = float(np.mean(model["logg"]))
    vmic_mean = args.vmicro

    # ---- emulator + mesh snapshots ----
    from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
    emu = IntensityPretrainedAemuSpectrumEmulator(args.bundle)
    names = list(emu.stellar_parameter_names)
    iT, iG, iF = ep.parameter_index(emu, "teff"), ep.parameter_index(emu, "logg"), ep.parameter_index(emu, "feh")
    iV = ep.parameter_index(emu, "vmicro")
    base_row = jnp.asarray(ep.to_parameters(emu, dict(teff=float(mteff[0]), logg=logg_mean, feh=feh0, vmicro=vmic_mean)))
    vmicro_fixed = lambda p: vmic_mean
    print("emulator parameters:", names, "| base row", np.asarray(base_row).round(3), flush=True)

    with fits.open(args.fits, ignore_missing_simple=True) as h:
        pulsation_data = Table(h[2].data)
    # build_cepheid_grid.fit_radius_template normalises by R(phase=0), which scales the
    # pulsation amplitude by R(0)/<R> (4% for delta Cep). Normalise by the mean instead so
    # base radius = <R> reproduces the SPIPS curve exactly.
    R_sp = np.asarray(pulsation_data["R"], float); ph_sp = np.asarray(pulsation_data["PHASE"], float)
    fourier = radius_template_mean_normalised(ph_sp, R_sp)
    teff_at, logg_at, vmicro_at = make_phase_modifiers(pulsation_data)
    phases_model = np.linspace(0.0, 1.0, args.n_phases, endpoint=False)
    t = time.time()
    radius = args.radius if args.radius is not None else float(R_sp.mean())
    bundle = build_bundle(emu, np.asarray(base_row), fourier_params=fourier, period=period,
                          timeseries=jnp.asarray(phases_model * period), n_mesh=args.n_mesh,
                          radius=radius, mass=args.mass, rotation_velocity=args.vrot,
                          param_names_attr="stellar_parameter_names", desc="  evaluating mesh")
    # snapshots carry the SPIPS per-phase Teff/logg/vmicro: used only by the exact-synthesis checks
    bundle = apply_phase_params(bundle, phases_model, _phase_modifiers_for(emu, teff_at, logg_at, vmicro_fixed))
    snaps = bundle.snapshots
    print(f"built {len(snaps)} snapshots ({args.n_mesh} vertices) in {time.time() - t:.0f}s", flush=True)
    # projected-area radius is what the flux sees (MeshModel.radii is mesh-resolution dependent)
    rad = np.array([math.sqrt(float(jnp.sum(m.visible_cast_areas)) / math.pi) for m in snaps])
    R_sp_at = np.interp(phases_model, mph, np.asarray(model["R"], float)[o], period=1.0)
    print(f"  projected-area radius over phase {rad.min():.2f}-{rad.max():.2f} R_sun, mean {rad.mean():.3f} "
          f"(SPIPS {R_sp.min():.2f}-{R_sp.max():.2f}, mean {R_sp.mean():.3f}); rms(model - SPIPS at the model phases) {np.sqrt(np.mean((rad - R_sp_at) ** 2)):.3f} R_sun", flush=True)

    # ---- kernels ----
    lw = jnp.linspace(math.log10(args.wl_range[0]), math.log10(args.wl_range[1]), args.n_wl)
    wl = 10.0 ** np.asarray(lw)
    t = time.time()
    kernels = [build_synthesis_kernel(m, lw, args.n_mu, 1) for m in snaps]
    jax.block_until_ready(kernels[-1].kernels)
    print(f"built {len(kernels)} kernels in {time.time() - t:.0f}s (half width {max(k.half_width for k in kernels)} px)", flush=True)

    # ---- passbands, zero points, reddening ----
    from spice.spectrum import filter as spfilter
    resp, zp = {}, {}
    for b in args.bands:
        cls, zkey = BANDS[b]
        resp[b] = jnp.asarray(getattr(spfilter, cls)().filter_responses_for_wavelengths(jnp.asarray(wl)))
        zp[b] = zps[zkey]
        cover = float(jnp.trapezoid(resp[b], jnp.asarray(wl)))
        full = getattr(spfilter, cls)().transmission_curve_wavelengths
        full = float(np.trapezoid(np.asarray(full[1]), np.asarray(full[0])))
        print(f"  band {b:3s}: ZP {zp[b]:.4e} erg/s/cm^2/A, integrated response {cover:.0f} A "
              f"({100 * (1 - cover / full):.1f}% of the passband lies outside the emulator range and is dropped)")
    alam_av = jnp.asarray(ccm89_alambda_over_av(wl, meta["r_v"]))
    dist_scale = (10.0 / d_pc) ** 2
    wl_j = jnp.asarray(wl)

    # ---- Teff nodes ----
    T_lo, T_hi = float(mteff.min() - args.teff_pad), float(mteff.max() + args.teff_pad)
    T_nodes = jnp.linspace(T_lo, T_hi, args.n_teff)
    print(f"Teff nodes: {args.n_teff} from {T_lo:.0f} to {T_hi:.0f} K (spacing {(T_hi - T_lo) / (args.n_teff - 1):.0f} K)", flush=True)

    # log g follows the SPIPS phase curve: per-epoch offsets from the mean, applied as a
    # secant correction dF = [F(g_mean + dg) - F(g_mean)] * (g_e - g_mean) / dg (two
    # emulator passes instead of one; the whole log g effect is ~2 mmag in B, so linear
    # is plenty). vmicro is NOT varied with phase: the TPayne-spice-harps bundle
    # (3df764e) is only trustworthy at its MARCS vmicro nodes -- between 1 and 5 km/s
    # its broadband flux drifts by ~23 mmag/(km/s) in V and jumps by 35-40 mmag at
    # ~1.35 and ~3.45 km/s (the node midpoints), which is the same grid-memorisation
    # failure the pre-August bundle had in Teff. Hold it at a node (--vmicro, default 2).
    logg_e = jnp.asarray([logg_at(p) for p in phases_model]) - logg_mean
    DG = 0.05

    def node_fluxes(feh):
        """(n_T, n_epoch, n_w) flux at 10 pc for every Teff node with the per-epoch log g / vmicro
        corrections folded in; emulator runs 3 * n_T * n_mu times."""
        def one(T):
            row = base_row.at[iT].set(T).at[iF].set(feh)
            f0 = kernel_flux_multi(emu.intensity, kernels, row, wavelength_chunk_size=args.chunk)[..., 0]
            fg = kernel_flux_multi(emu.intensity, kernels, row.at[iG].add(DG), wavelength_chunk_size=args.chunk)[..., 0]
            return f0 + (fg - f0) * (logg_e / DG)[:, None]
        return jax.lax.map(one, T_nodes)

    # ---- parameter vector ----
    # theta = [coef (2K+1), ebv, feh, dphi]
    n_coef = 2 * K + 1
    theta0 = np.concatenate([coef0, [args.ebv0, feh0, 0.0]])
    S = np.concatenate([[100.0], np.full(2 * K, 50.0), [0.02, 0.1, 0.002]])
    lo = np.concatenate([[T_lo], coef0[1:] - 400.0, [0.0, -1.0, -0.1]])
    hi = np.concatenate([[T_hi], coef0[1:] + 400.0, [0.4, 0.5, 0.1]])
    if args.seed_from:
        prev = pickle.load(open(args.seed_from, "rb"))
        theta0 = np.asarray(prev["theta"]); print("warm start from", args.seed_from)
    fixed = np.zeros(theta0.size, bool)
    if args.fix_ebv: fixed[n_coef] = True
    if not args.free_feh: fixed[n_coef + 1] = True
    if args.fix_teff_shape: fixed[1:n_coef] = True
    free_idx = np.where(~fixed)[0]
    print(f"free parameters: {free_idx.size} of {theta0.size} "
          f"({'Teff0' if not args.fix_teff_shape else 'Teff0 only'}{'+shape' if not args.fix_teff_shape else ''}, "
          f"E(B-V) {'free' if not args.fix_ebv else 'fixed'}, [Fe/H] {'free' if args.free_feh else 'fixed'}, dphi free)", flush=True)

    ph_model = jnp.asarray(phases_model)
    obs_j = {k: dict(phase=jnp.asarray(v["phase"]), mag=jnp.asarray(v["mag"]),
                     sig=jnp.asarray(np.hypot(v["err"], args.err_floor))) for k, v in obs.items()}

    def model_mags(theta, F_nodes):
        """Per-band model magnitudes at the model epochs, (n_epoch,) each."""
        coef = theta[:n_coef]; ebv = theta[n_coef]
        teff_e = fourier_eval(ph_model, coef)                                    # (n_epoch,)
        i = jnp.clip(jnp.searchsorted(T_nodes, teff_e, side="right") - 1, 0, args.n_teff - 2)
        f = jnp.clip((teff_e - T_nodes[i]) / (T_nodes[i + 1] - T_nodes[i]), 0.0, 1.0)   # constant beyond the node range
        e_idx = jnp.arange(args.n_phases)
        flux = (1.0 - f)[:, None] * F_nodes[i, e_idx] + f[:, None] * F_nodes[i + 1, e_idx]   # (n_epoch, n_w)
        flux = flux * dist_scale * jnp.power(10.0, -0.4 * meta["r_v"] * ebv * alam_av)[None, :]
        return {b: jax.vmap(lambda fl: band_mag(fl, wl_j, resp[b], zp[b]))(flux) for b in args.bands}, teff_e

    def chi2(theta, F_nodes):
        mags, teff_e = model_mags(theta, F_nodes)
        dphi = theta[n_coef + 2]
        # keep Teff(phase) inside the node range (the flux is frozen beyond it): soft wall, 50 K scale
        wall = jnp.sum(jnp.maximum(T_nodes[0] - teff_e, 0.0) ** 2 + jnp.maximum(teff_e - T_nodes[-1], 0.0) ** 2) / 50.0 ** 2
        tot = wall; parts = {}
        for key, o in obs_j.items():
            if "-" in key:
                b1, b2 = key.split("-"); curve = mags[b1] - mags[b2]
            else:
                curve = mags[key]
            pred = periodic_interp((o["phase"] - dphi) % 1.0, ph_model, curve)
            c = jnp.sum(((o["mag"] - pred) / o["sig"]) ** 2)
            parts[key] = c; tot = tot + c
        return tot, parts

    # ---- node fluxes: once if [Fe/H] is frozen ----
    t = time.time()
    F0 = jax.jit(node_fluxes)(jnp.asarray(feh0)); jax.block_until_ready(F0)
    print(f"node fluxes (2 x {args.n_teff} Teff x {args.n_mu} mu nodes x {args.n_wl} px x {args.n_phases} epochs): "
          f"{time.time() - t:.0f}s incl. compile", flush=True)

    # ---- validation against exact per-element synthesis ----
    if not args.skip_validation:
        from spice.spectrum.spectrum import simulate_observed_flux
        e = args.n_phases // 3
        m = snaps[e]; p_mesh = np.asarray(m.parameters)
        T_e = float(np.mean(p_mesh[:, iT]))
        row_e = base_row.at[iT].set(T_e).at[iF].set(feh0)
        t = time.time()
        exact_row = simulate_observed_flux(emu.intensity, m._replace(parameters=jnp.broadcast_to(row_e, p_mesh.shape)), lw, chunk_size=256)[:, 0]
        exact_mesh = simulate_observed_flux(emu.intensity, m, lw, chunk_size=256)[:, 0]
        kern_row = kernel_flux_multi(emu.intensity, [kernels[e]], row_e, wavelength_chunk_size=args.chunk)[0, :, 0]
        i = int(np.clip(np.searchsorted(np.asarray(T_nodes), T_e, side="right") - 1, 0, args.n_teff - 2))
        f = (T_e - float(T_nodes[i])) / float(T_nodes[i + 1] - T_nodes[i])
        interp_row = (1 - f) * F0[i, e] + f * F0[i + 1, e]
        def mags_of(fl):
            fl = jnp.asarray(fl) * dist_scale
            return {b: float(band_mag(fl, wl_j, resp[b], zp[b])) for b in args.bands}
        mk, mx, mi, mm = mags_of(kern_row), mags_of(exact_row), mags_of(interp_row), mags_of(exact_mesh)
        rel = lambda a, b: float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1)))
        print(f"validation, epoch {e} (Teff {T_e:.0f} K, logg {np.mean(p_mesh[:, iG]):.3f} vs mean {logg_mean:.3f}, "
              f"vmicro fixed {vmic_mean:.1f}), exact synthesis {time.time() - t:.0f}s:")
        print(f"  kernel vs exact, same row (mean logg/vmicro):   max rel flux/px {rel(kern_row, exact_row):.2e}; "
              + "  ".join(f"d{b} {1e3 * (mk[b] - mx[b]):+.2f} mmag" for b in args.bands))
        print(f"  mean logg vs per-phase logg (both exact):         "
              + "  ".join(f"d{b} {1e3 * (mx[b] - mm[b]):+.2f} mmag" for b in args.bands))
        print(f"  FULL SURROGATE (Teff-node interp + secant logg) vs exact per-phase mesh:  "
              + "  ".join(f"d{b} {1e3 * (mi[b] - mm[b]):+.2f} mmag" for b in args.bands)
              + f"   (per-pixel max rel {rel(interp_row, exact_mesh):.2e}, passband-weighted rel "
              + f"{float(np.abs(np.trapezoid((np.asarray(interp_row) - np.asarray(exact_mesh)) * np.asarray(resp[args.bands[0]]), wl) / np.trapezoid(np.asarray(exact_mesh) * np.asarray(resp[args.bands[0]]), wl))):.2e} in {args.bands[0]})", flush=True)

    # ---- objective ----
    theta0_j = jnp.asarray(theta0); S_j = jnp.asarray(S)
    def unpack(x_free):
        full = theta0_j.at[jnp.asarray(free_idx)].set(theta0_j[jnp.asarray(free_idx)] + x_free * S_j[jnp.asarray(free_idx)])
        return full

    if args.free_feh:
        def total(x):
            th = unpack(x); F = node_fluxes(th[n_coef + 1]); c, parts = chi2(th, F); return c, parts
    else:
        def total(x):
            th = unpack(x); c, parts = chi2(th, F0); return c, parts
    vg = jax.jit(jax.value_and_grad(total, has_aux=True))
    t = time.time(); (v0, parts0), g0 = vg(jnp.zeros(free_idx.size)); jax.block_until_ready(g0)
    print(f"first value+grad (compile) {time.time() - t:.1f}s; warm: ", end="", flush=True)
    t = time.time(); jax.block_until_ready(vg(jnp.zeros(free_idx.size))[1]); print(f"{time.time() - t:.2f}s", flush=True)
    print("start: chi2/N {:.3f}  ".format(float(v0) / N_ph) + "  ".join(f"{k} {float(c) / obs[k]['mag'].size:.2f}" for k, c in parts0.items()), flush=True)
    if args.check_grad:
        fval = jax.jit(lambda x: total(x)[0]); h = 1e-3
        for j in range(free_idx.size):
            dx = jnp.zeros(free_idx.size).at[j].set(h)
            fd = (float(fval(dx)) - float(fval(-dx))) / (2 * h)
            print(f"  grad check param {free_idx[j]:2d}: autodiff {float(g0[j]):+.5e}  finite diff {fd:+.5e}")
        gn = np.asarray(g0) / np.linalg.norm(np.asarray(g0))
        for a in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
            print(f"  line search: f(x0 - {a:.0e} g_hat) - f(x0) = {float(fval(-a * jnp.asarray(gn))) - float(v0):+.4e}   (|g| {np.linalg.norm(np.asarray(g0)):.3e})")

    hist = []
    def f(x):
        (v, _), g = vg(jnp.asarray(x)); hist.append(float(v))
        if args.check_grad:
            print(f"    eval {len(hist):3d}: f {float(v):.6e} |g| {float(jnp.linalg.norm(g)):.3e} nan(f) {bool(jnp.isnan(v))} nan(g) {bool(jnp.isnan(g).any())} |x| {np.linalg.norm(x):.3e}", flush=True)
        return float(v), np.asarray(g, float)
    bounds = [((lo[i] - theta0[i]) / S[i], (hi[i] - theta0[i]) / S[i]) for i in free_idx]
    t = time.time()
    r = minimize(f, np.zeros(free_idx.size), jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=args.maxiter))
    th = np.asarray(unpack(jnp.asarray(r.x)))
    (v1, parts1), _ = vg(jnp.asarray(r.x))
    print(f"\noptimisation: {r.nit} it, {len(hist)} eval, {time.time() - t:.0f}s, {r.message}")
    print("end:   chi2/N {:.3f}  ".format(float(v1) / N_ph) + "  ".join(f"{k} {float(c) / obs[k]['mag'].size:.2f}" for k, c in parts1.items()))
    coef = th[:n_coef]
    teff_curve = np.asarray(fourier_eval(jnp.asarray(mph), jnp.asarray(coef)))
    print(f"  Teff0 = {coef[0]:.0f} K (SPIPS Fourier mean {coef0[0]:.0f}); fitted curve {teff_curve.min():.0f}-{teff_curve.max():.0f} K "
          f"vs SPIPS {mteff.min():.0f}-{mteff.max():.0f}; rms(fit - SPIPS) {np.sqrt(np.mean((teff_curve - mteff) ** 2)):.0f} K")
    print(f"  E(B-V) = {th[n_coef]:.4f}   [Fe/H] = {th[n_coef + 1]:+.3f}   dphi = {th[n_coef + 2]:+.4f}")
    if teff_obs is not None:
        pred = np.asarray(fourier_eval(jnp.asarray((teff_obs['phase'] - th[n_coef + 2]) % 1.0), jnp.asarray(coef)))
        spips_at = np.interp(teff_obs["phase"], mph, mteff, period=1.0)
        print(f"  Andrievsky+2005 Teff ({teff_obs['teff'].size} pts): fit - obs mean {np.mean(pred - teff_obs['teff']):+.0f} K, "
              f"rms {np.sqrt(np.mean((pred - teff_obs['teff']) ** 2)):.0f} K  |  SPIPS - obs mean {np.mean(spips_at - teff_obs['teff']):+.0f}, "
              f"rms {np.sqrt(np.mean((spips_at - teff_obs['teff']) ** 2)):.0f} K")

    mags_fit, teff_e = model_mags(jnp.asarray(th), F0 if not args.free_feh else jax.jit(node_fluxes)(jnp.asarray(th[n_coef + 1])))
    mags_start, _ = model_mags(theta0_j, F0)
    result = dict(theta=th, theta0=theta0, free_idx=free_idx, S=S, chi2=float(v1), chi2_parts={k: float(c) for k, c in parts1.items()},
                  N_ph=N_ph, hist=hist, args=vars(args), meta=meta, zps=zp, fourier_order=K,
                  phases_model=phases_model, mags_fit={b: np.asarray(v) for b, v in mags_fit.items()},
                  mags_start={b: np.asarray(v) for b, v in mags_start.items()}, teff_model_epochs=np.asarray(teff_e),
                  spips_phase=mph, spips_teff=mteff, teff_obs=teff_obs, obs=obs, T_nodes=np.asarray(T_nodes))
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)

    # ---- errors: Hessian of chi2 in scaled units ----
    try:
        Hm = np.asarray(jax.hessian(lambda x: total(x)[0])(jnp.asarray(r.x))); ev = np.linalg.eigvalsh(Hm)
        print(f"Hessian eigenvalues (scaled units): {np.array2string(ev, precision=3)}")
        result["hessian"] = Hm
        if np.all(ev > 0):
            cov = 2.0 * np.linalg.inv(Hm); sg = np.sqrt(np.diag(cov)) * S[free_idx]
            Cc = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
            labels = ["T0"] + [f"{'a' if j % 2 == 0 else 'b'}{j // 2 + 1}" for j in range(2 * K)] + ["E(B-V)", "[Fe/H]", "dphi"]
            print("  formal 1-sigma (chi2+1): " + "  ".join(f"{labels[i]} {s:.3g}" for i, s in zip(free_idx, sg)))
            names_free = [labels[i] for i in free_idx]
            if "E(B-V)" in names_free:
                j = names_free.index("E(B-V)")
                print("  corr(E(B-V), .): " + "  ".join(f"{n} {Cc[j, i]:+.2f}" for i, n in enumerate(names_free) if i != j))
            result.update(cov=cov, sigma=dict(zip(names_free, sg)), corr=Cc, labels=names_free)
    except Exception as exc:  # noqa: BLE001
        print(f"Hessian failed: {type(exc).__name__}: {str(exc)[:200]}")
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
