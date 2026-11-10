"""Turbospectrum specific intensities for delta Cep, one synthesis per (variant, phase).

Runs on Gadi. Reads the table written by `dcep_ts_phase_table.py`, interpolates a
MARCS t02 atmosphere (interpol_modeles) to each phase's (Teff, log g, [Fe/H]) and
synthesises I(lambda, mu) with the production HARPS-grid physics (fixed C/N/O
interface, solar-scaled abundances at [Fe/H]) at that phase's microturbulence.
The atmosphere stays on the t02 MARCS models for every xi, so xi enters only the
line synthesis (no MARCS turbulence-node steps).

Where the t02 store has holes (hot, low-g corners) the bracket on that axis is
widened instead of switching to another turbulence label; `points.json` records
the corners used.

Reuses the off-node driver in /scratch/y89/mj8805/emu_check (`offnode_synth.py`:
interpolate, build_config, parse_intensity, _run_one).

Output per variant: OUT/<variant>.npz with wl (nwl,), mus (nmu,), phase/teff/logg/
feh/vmicro (nphase,), I (nphase, nmu, nwl) absolute intensity and N
(nphase, nmu, nwl) continuum-normalised intensity (float32); missing phases are NaN.

--spherical runs bsyn with spherical transfer (the production driver hard-codes
'SPHERICAL: F'; the interpolated MARCS models carry the radius column of their
1 M_sun parent models, so the sphericity is that of a 1 M_sun star at this log g,
R ~ 19 R_sun, vs delta Cep's ~43 R_sun). --radius-from-table shifts the radius
column to SPIPS R(phase) at tau_Ross = 1 (validated against a native 5 M_sun MARCS
model by dcep_ts_sph_check.py). In spherical mode the TS mu points are chosen per
model so that the output is on the photospheric mu grid (see spherical_mu_points),
plus 4 rays through the ring outside tau_Ross = 1.

Usage: python dcep_ts_intensities.py --table dcep_ts_phases.csv --out DIR
           [--workers N] [--variants a,b] [--max-rows N] [--lam 4950 5250] [--spherical]
"""
import argparse, csv, itertools, json, os, sys, time
from multiprocessing import Pool

import numpy as np

TS_SCRIPTS = "/scratch/mk27/mj8805/Turbospectrum_NLTE/scripts"
PP_BLOCK = """'SPHERICAL:'  'F'
  30
  {float(config.resolution):.2f}
  15
  {turb_val:.2f}
"""
# TS defaults for spherical transfer: mihal (core rays), taum, ncore, diflog (input.f reads all four)
SPH_BLOCK = """'SPHERICAL:'  'T'
  30
  300.00
  15
  1.30
"""


def _install_spherical_driver():
    """Register a copy of the production run_turbospectrum with 'SPHERICAL: T' under the same module
    name, before offnode_synth imports it, so forked workers and pickled configs resolve to it.
    The production file in mk27 is not touched."""
    import importlib.util
    if TS_SCRIPTS not in sys.path:
        sys.path.insert(0, TS_SCRIPTS)
    path = os.path.join(TS_SCRIPTS, "run_turbospectrum.py")
    src = open(path).read()
    assert src.count(PP_BLOCK) == 1, "production bsyn SPHERICAL block changed; update PP_BLOCK"
    spec = importlib.util.spec_from_loader("run_turbospectrum", loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec); mod.__file__ = path
    sys.modules["run_turbospectrum"] = mod
    exec(compile(src.replace(PP_BLOCK, SPH_BLOCK), path, "exec"), mod.__dict__)


SPHERICAL = "--spherical" in sys.argv
if SPHERICAL:
    _install_spherical_driver()
sys.path.insert(0, "/scratch/y89/mj8805/emu_check")
import offnode_synth as OS  # noqa: E402  (also imports run_turbospectrum as OS.RT)

RT = OS.RT
MUS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LABEL = "02"


def _pairs(nodes, x, widen):
    """Bracketing node pairs for x, adjacent first, then widened by up to `widen` nodes per side."""
    nodes = sorted(nodes)
    lo, hi = OS.bracket(nodes, x)
    i, j = nodes.index(lo), nodes.index(hi)
    out = []
    for di, dj in sorted(itertools.product(range(widen + 1), repeat=2), key=lambda d: (sum(d), max(d))):
        if i - di >= 0 and j + dj < len(nodes):
            out.append((nodes[i - di], nodes[j + dj]))
    return out


def find_brackets_t02(teff, logg, feh, widen=2):
    """8 spherical t02 corners (t, g, z order, as interpol_modeles expects) with every file present."""
    for geom in ("p", "s"):
        cands = itertools.product(_pairs(OS.TEFF_NODES, teff, widen), _pairs(OS.LOGG_NODES, logg, widen),
                                  _pairs(OS.FEH_NODES, feh, widen))
        def width(c):
            (t1, t2), (g1, g2), (z1, z2) = c
            return (t2 - t1) / 250 + (g2 - g1) / 0.5 + (z2 - z1) / 0.25
        for (t1, t2), (g1, g2), (z1, z2) in sorted(cands, key=width):
            paths = [os.path.join(OS.STORE, OS.marcs_file(geom, t, g, z, LABEL))
                     for t in (t1, t2) for g in (g1, g2) for z in (z1, z2)]
            if all(os.path.exists(p) for p in paths):
                return paths, geom, (t1, t2, g1, g2, z1, z2)
    raise FileNotFoundError(f"no complete t02 bracket for {teff}/{logg}/{feh}")


RSUN_CM = 6.957e10


def _interp_layers(lines):
    """(n, first layer line index) of an interpol_modeles output ('sphINTERPOL' header: name, n, lambda_ref, logg, ...)."""
    return int(lines[0].split()[1]), 1


def model_radius(path):
    """Radius [cm] at tau_Ross = 1 of an interpol_modeles model (columns: lgtau5000 T lgPe lgPg xi r lgtauRoss)."""
    lines = open(path).read().splitlines()
    n, i0 = _interp_layers(lines)
    cols = np.array([[float(x.replace("D", "E")) for x in l.split()[:7]] for l in lines[i0:i0 + n]])
    return float(np.interp(0.0, cols[:, 6], cols[:, 5]))


def set_model_radius(path, R_cm):
    """Shift the radius column so r(tau_Ross = 1) = R_cm, keeping r - r(tau_Ross = 1) (the geometric depth
    scale, set by the pressure scale height at this g and T, not by the mass). Returns the old radius."""
    lines = open(path).read().splitlines()
    n, i0 = _interp_layers(lines)
    R_old = model_radius(path)
    for k in range(i0, i0 + n):
        tok = lines[k].split()
        r = float(tok[5].replace("D", "E")) - R_old + R_cm
        lines[k] = " ".join([f"{float(tok[0]):8.4f}", f"{float(tok[1]):8.2f}", f"{float(tok[2]):8.4f}",
                             f"{float(tok[3]):8.4f}", f"{float(tok[4]):8.4f}", f"{r:13.6E}", f"{float(tok[6]):8.4f}"]
                            + tok[7:])
    open(path, "w").write("\n".join(lines) + "\n")
    return R_old


RING_FRACS = (0.05, 0.15, 0.35, 0.7)


def spherical_mu_points(path, mus_target=MUS, ring_fracs=RING_FRACS):
    """Outer-boundary mu values that Turbospectrum's spherical intensities need for a photospheric mesh.

    In spherical transfer TS measures mu at the model's outer radius r_out, while the SPICE mesh is the
    tau_Ross = 1 sphere (radius R). A ray at impact parameter b = R sqrt(1 - mu_s^2) leaves the
    atmosphere at mu_out = sqrt(1 - (b / r_out)^2), so the photospheric limb (mu_s = 0) sits at
    mu_out = sqrt(1 - (R / r_out)^2), not at 0. Also returns mu_out for rays through the ring
    R < b < r_out (b = R + f (r_out - R)), which the mesh does not cover.
    Returns (mu_disc, mu_ring, ring_b_over_R, r_out / R).
    """
    lines = open(path).read().splitlines()
    n, i0 = _interp_layers(lines)
    r = np.array([float(l.split()[5].replace("D", "E")) for l in lines[i0:i0 + n]])
    R, r_out = model_radius(path), float(r.max())
    mus_target = np.asarray(mus_target, float)
    mu_disc = np.sqrt(1.0 - (R / r_out) ** 2 * (1.0 - mus_target ** 2))
    b = R + np.asarray(ring_fracs) * (r_out - R)
    mu_ring = np.sqrt(1.0 - (b / r_out) ** 2)
    return mu_disc, mu_ring, b / R, r_out / R


def _pick_mu(mus_file, wanted, tol=2e-4):
    idx = np.array([int(np.argmin(np.abs(mus_file - m))) for m in wanted])
    assert np.all(np.abs(mus_file[idx] - wanted) < tol), (mus_file, wanted)
    return idx


def prepare(row, out_root, radius_from_table=False):
    name = f"{row['variant']}_p{int(row['index']):02d}"
    root = os.path.join(out_root, "runs", name)
    atm_dir = os.path.join(root, "atm"); os.makedirs(atm_dir, exist_ok=True)
    teff, logg, feh = int(row["teff"]), float(row["logg"]), float(row["feh"])
    target = os.path.join(atm_dir, RT.get_model_filename(teff, logg, feh, LABEL))
    paths, geom, corners = find_brackets_t02(teff, logg, feh)
    ok, log = OS.interpolate(paths, teff, logg, feh, target, root)
    if not ok:
        raise RuntimeError(f"interpolation failed for {name}: {log}")
    info = dict(name=name, geom=geom, corners=corners, brackets=[os.path.basename(p) for p in paths])
    if radius_from_table:
        R_new = float(row["radius_rsun"]) * RSUN_CM
        info["radius_model_cm"] = set_model_radius(target, R_new)
        info["radius_set_cm"] = R_new
    info["model_path"] = target
    return name, root, atm_dir, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--variants", default="", help="comma-separated subset")
    ap.add_argument("--max-rows", type=int, default=0, help="per variant (smoke tests)")
    ap.add_argument("--lam", type=float, nargs=2, default=(4950.0, 5250.0))
    ap.add_argument("--spherical", action="store_true",
                    help="spherical transfer in bsyn (patched driver); outputs are named <variant>_sph")
    ap.add_argument("--radius-from-table", action="store_true",
                    help="shift each model's radius column to the table's radius_rsun (SPIPS R(phase)) at "
                         "tau_Ross = 1; outputs are named <variant>_sphR")
    args = ap.parse_args()
    tag = ("_sphR" if args.radius_from_table else "_sph") if args.spherical else ""
    os.makedirs(args.out, exist_ok=True)

    rows = list(csv.DictReader(open(args.table)))
    wanted = [v for v in args.variants.split(",") if v]
    if wanted:
        rows = [r for r in rows if r["variant"] in wanted]
    if args.max_rows:
        rows = [r for r in rows if int(r["index"]) < args.max_rows]
    mu_file = RT._write_mu_points_file(MUS, os.path.join(args.out, "tmp"), "mus12")

    jobs, meta = [], {}
    for row in rows:
        try:
            name, root, atm_dir, info = prepare(row, args.out, args.radius_from_table)
        except Exception as e:
            print(f"[SKIP] {row['variant']} {row['index']}: {e}", flush=True); continue
        row_mu_file = mu_file
        if args.spherical:
            mu_disc, mu_ring, b_R, rout_R = spherical_mu_points(info["model_path"])
            mu_disc, mu_ring = np.round(mu_disc, 5), np.round(mu_ring, 5)
            info.update(mu_disc=mu_disc.tolist(), mu_ring=mu_ring.tolist(), ring_b_over_R=b_R.tolist(),
                        r_out_over_R=rout_R)
            row_mu_file = RT._write_mu_points_file(sorted(set(mu_disc.tolist() + mu_ring.tolist())),
                                                   os.path.join(root, "tmp"), "mus_sph")
            print(f"  {name}: r_out/R = {rout_R:.4f}, photospheric limb at mu_out = {mu_disc[0]:.4f}", flush=True)
        config = OS.build_config(root, atm_dir, row_mu_file, lam=tuple(args.lam))
        params = dict(teff=int(row["teff"]), logg=float(row["logg"]), feh=float(row["feh"]),
                      turbvel=float(row["vmicro"]), t_value=LABEL, **OS.ABUND0)
        jobs.append((name, params, config)); meta[name] = dict(row=row, info=info)
        print(f"[prepared] {name} {info['geom']} corners {info['corners']}", flush=True)
    json.dump(meta, open(os.path.join(args.out, "points.json"), "w"), indent=1, default=str)

    t0 = time.time()
    with Pool(max(1, min(args.workers, len(jobs)))) as pool:
        results = pool.map(OS._run_one, jobs, chunksize=1)
    print(f"synthesis wall {time.time() - t0:.0f}s for {len(jobs)} runs", flush=True)

    by_variant, failed = {}, len(rows) - len(jobs)
    for (name, params, config), res in zip(jobs, results):
        meta[name]["status"] = res.get("status"); meta[name]["elapsed_s"] = res.get("elapsed_s")
        path = res.get("output_path")
        if not (path and os.path.exists(path)):
            print(f"[FAILED] {name}: {str(res.get('message', ''))[:300]}", flush=True); failed += 1; continue
        wl, mus, I_abs, I_norm = OS.parse_intensity(path)
        extra = {}
        if args.spherical:
            info = meta[name]["info"]
            i_d, i_r = _pick_mu(mus, np.array(info["mu_disc"])), _pick_mu(mus, np.array(info["mu_ring"]))
            extra = dict(ring_I=I_abs[i_r], ring_N=I_norm[i_r], mu_out=mus[i_d], ring_b=np.array(info["ring_b_over_R"]),
                         r_out=info["r_out_over_R"])
            I_abs, I_norm, mus = I_abs[i_d], I_norm[i_d], np.asarray(MUS, float)   # now on the photospheric mu_s grid
        by_variant.setdefault(meta[name]["row"]["variant"], []).append((meta[name]["row"], wl, mus, I_abs, I_norm, extra))
    json.dump(meta, open(os.path.join(args.out, "points.json"), "w"), indent=1, default=str)

    for variant, items in by_variant.items():
        allrows = [r for r in rows if r["variant"] == variant]
        n = len(allrows); wl, mus = items[0][1], items[0][2]
        I = np.full((n, len(mus), len(wl)), np.nan, np.float32); N = np.full_like(I, np.nan)
        ext = {}
        if args.spherical:
            nr = len(RING_FRACS)
            ext = dict(ring_I=np.full((n, nr, len(wl)), np.nan, np.float32), ring_N=np.full((n, nr, len(wl)), np.nan, np.float32),
                       mu_out=np.full((n, len(mus)), np.nan), ring_b=np.full((n, nr), np.nan), r_out=np.full(n, np.nan))
        for row, w, m, Ia, In, extra in items:
            assert np.array_equal(w, wl) and np.array_equal(m, mus)
            k = [int(r["index"]) for r in allrows].index(int(row["index"]))
            I[k], N[k] = Ia, In
            for key, v in extra.items():
                ext[key][k] = v
        col = lambda c, t=float: np.array([t(r[c]) for r in allrows])
        if "radius_rsun" in allrows[0]:
            ext["radius_rsun"] = col("radius_rsun")
        # spherical: mus is the photospheric mu_s grid (what SPICE's mesh uses); mu_out is TS's boundary mu,
        # ring_* are the rays outside the tau_Ross = 1 sphere at impact parameters ring_b (units of R)
        np.savez(os.path.join(args.out, f"{variant}{tag}.npz"), wl=wl, mus=mus, I=I, N=N,
                 phase=col("phase"), teff=col("teff"), logg=col("logg"), feh=col("feh"), vmicro=col("vmicro"), **ext)
        print(f"wrote {variant}{tag}.npz: {len(items)}/{n} phases, {len(mus)} mu, {len(wl)} px", flush=True)
    if failed:
        print(f"{failed} of {len(rows)} rows failed or were skipped", flush=True)
        sys.exit(1)   # so a dependent job (afterok) does not start on a partial table
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
