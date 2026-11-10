"""Off-node emulator check: synthesize Turbospectrum intensity spectra on MARCS
atmospheres *interpolated* (interpol_modeles) to parameters between the MARCS
nodes, with the production HARPS-grid physics (same line lists, molecules,
abundance handling, 3780-6910 A at 0.01 A), at mu = 0.1, 0.4, 0.7, 1.0, and
compare with the released TPayne-spice-harps emulator evaluated at the same
(Teff, log g, [Fe/H], vmicro, abundances, mu).

Usage: python offnode_synth.py --out DIR [--smoke] [--workers N] [--only NAME,...]
"""
import argparse, dataclasses, json, os, shutil, sys, time, traceback
import numpy as np

TS_ROOT = "/scratch/mk27/mj8805/Turbospectrum_NLTE"
STORE = f"{TS_ROOT}/input_files/model_atmospheres/1D/marcs_standard_comp/marcs_standard_comp"
PROD_CFG = "/scratch/y89/mj8805/emu_check/prod_config/input_config.json"
VAL_ZARR = "/scratch/y89/tr0953/maja_intensities_models/repacked_harps_1024/harps_final_repacked_validation_000000.zarr"
MUS = [0.1, 0.4, 0.7, 1.0]
sys.path.insert(0, f"{TS_ROOT}/scripts")
import run_turbospectrum as RT  # noqa: E402

TEFF_NODES = list(range(2500, 4000, 100)) + list(range(4000, 8001, 250))
LOGG_NODES = [x / 2 for x in range(-1, 12)]  # -0.5 .. 5.5
FEH_NODES = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]

# name: teff, logg, feh, vmicro, abundances(dict), kind; optional atm_feh: [Fe/H] of the interpolated atmosphere
# (default feh). Set it to the nearest MARCS node to mimic the training grid: structure at the node, line synthesis
# (babsma/bsyn metallicity) at the continuous feh.
POINTS = {
    # controls
    "ctrl_node_6500_3.5":   dict(teff=6500, logg=3.5, feh=-0.25, vmicro=1.5, kind="node"),
    "ctrl_valrow":          dict(kind="valrow"),  # replica of a held-out validation row, real MARCS file
    # science points (TZ For stars at the fitted parameters; log g from M, R)
    "tzfor1_4896_2.91":     dict(teff=4896, logg=2.91, feh=-0.18, vmicro=1.5, kind="interp"),
    "tzfor2_6396_3.54":     dict(teff=6396, logg=3.54, feh=-0.18, vmicro=1.5, kind="interp"),
    # single-axis off-node tests
    "mid_teff_6375_4.0":    dict(teff=6375, logg=4.0, feh=0.0, vmicro=2.0, kind="interp"),
    "mid_logg_6500_3.75":   dict(teff=6500, logg=3.75, feh=0.0, vmicro=2.0, kind="interp"),
    # midpoints across the domain (all three axes off-node)
    "mid_4375_2.25":        dict(teff=4375, logg=2.25, feh=-0.62, vmicro=1.5, kind="interp"),
    "mid_5125_4.75":        dict(teff=5125, logg=4.75, feh=-0.12, vmicro=1.0, kind="interp"),
    "mid_5875_1.75":        dict(teff=5875, logg=1.75, feh=0.06, vmicro=2.5, kind="interp"),
    "mid_6625_4.25":        dict(teff=6625, logg=4.25, feh=0.12, vmicro=2.0, kind="interp"),
    "mid_7125_4.25":        dict(teff=7125, logg=4.25, feh=0.12, vmicro=2.0, kind="interp"),
    "mid_7625_4.25":        dict(teff=7625, logg=4.25, feh=0.12, vmicro=2.0, kind="interp"),
}
ABUND0 = dict(a=0.0, c=0.0, n=0.0, o=0.0, r=0.0, s=0.0)


def bracket(values, x):
    values = sorted(values)
    if x <= values[0]: return values[0], values[1]
    if x >= values[-1]: return values[-2], values[-1]
    for i in range(len(values) - 1):
        if values[i] <= x < values[i + 1]: return values[i], values[i + 1]
    raise ValueError


def marcs_file(geom, teff, logg, feh, label):
    alpha = RT.standard_composition_alpha(feh)
    mass = "0.0" if geom == "p" else "1.0"
    return f"{geom}{int(teff)}_g{logg:+.1f}_m{mass}_t{label}_st_z{feh:+.2f}_a{alpha:+.2f}_c+0.00_n+0.00_o{alpha:+.2f}_r+0.00_s+0.00.mod"


def find_brackets(teff, logg, feh, label, store_labels):
    """8 corner models (t,g,z order as in RT.find_bracketing_models); try plane-parallel first, then spherical, then other t-labels."""
    t1, t2 = bracket(TEFF_NODES, teff); g1, g2 = bracket(LOGG_NODES, logg); z1, z2 = bracket(FEH_NODES, feh)
    tried = []
    labels = [label] + [l for l in store_labels if l != label]
    for lab in labels:
        for geom in ("p", "s"):
            paths = []
            for t in (t1, t2):
                for g in (g1, g2):
                    for z in (z1, z2):
                        paths.append(os.path.join(STORE, marcs_file(geom, t, g, z, lab)))
            missing = [p for p in paths if not os.path.exists(p)]
            if not missing:
                return paths, geom, lab, (t1, t2, g1, g2, z1, z2)
            tried.append((lab, geom, [os.path.basename(m) for m in missing]))
    raise FileNotFoundError(f"no complete bracket for {teff}/{logg}/{feh}: {tried}")


def interpolate(paths, teff, logg, feh, out_path, tmp_dir):
    """Mirror RT.ModelInterpolator.interpolate with an explicit bracket list."""
    import subprocess
    alt = os.path.join(tmp_dir, os.path.basename(out_path) + ".alt")
    s = "".join(f"'{p}'\n" for p in paths) + f"'{out_path}'\n'{alt}'\n{teff}\n{logg}\n{feh}\n.false.\n.false.\n''\n"
    pr = subprocess.run([f"{TS_ROOT}/interpolator/interpol_modeles_linux"], input=s, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=TS_ROOT)
    ok = pr.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    return ok, pr.stdout[-2000:]


def build_config(out_root, atm_dir, mu_file, lam=(3780.0, 6910.0), flux_mode=False):
    cfg = json.load(open(PROD_CFG))
    cfg = RT._normalize_config_dict(cfg, default_project_root=TS_ROOT)
    accepted = {f.name for f in dataclasses.fields(RT.TurbospectrumConfig)}
    cfg = {k: v for k, v in cfg.items() if k in accepted}
    cfg.update(project_root=TS_ROOT, model_atmosphere_path=atm_dir,
               linelist_path=f"{TS_ROOT}/input_files/linelists",
               babsma_path=f"{TS_ROOT}/exec-gf/babsma_lu", bsyn_path=f"{TS_ROOT}/exec-gf/bsyn_lu",
               interpol_path=f"{TS_ROOT}/interpolator/interpol_modeles_linux",
               output_dir=f"{out_root}/spectra", log_dir=f"{out_root}/logs", tmp_dir=f"{out_root}/tmp",
               model_opac_dir=os.path.relpath(f"{out_root}/opac", TS_ROOT),
               lambda_min=float(lam[0]), lambda_max=float(lam[1]), lambda_step=0.01,
               output_mode="Flux" if flux_mode else "Intensity", mu_points_file="" if flux_mode else mu_file, force=True)
    config = RT.TurbospectrumConfig(**cfg)
    for d in (config.output_dir, config.log_dir, config.tmp_dir, f"{out_root}/opac"):
        os.makedirs(d, exist_ok=True)
    RT.ensure_directories(config)
    config.linelist_file_path = RT.create_linelist_file(config)
    return config


def valrow_params():
    import zarr
    z = zarr.open(VAL_ZARR, mode="r"); names = list(z["param_names"][:]); p = z["params"][:]
    i = {n: k for k, n in enumerate(names)}
    sel = np.where((np.abs(p[:, i["teff"]] - 5796) < 0.6) & (np.abs(p[:, i["logg"]] - 3.599) < 0.002))[0]
    assert len(sel) == 1, sel
    r = int(sel[0]); row = {n: float(p[r, k]) for k, n in enumerate(names)}
    return r, row


def parse_flux(path):
    arr = np.loadtxt(path, comments="#")  # wavelength, normalised flux, absolute flux
    return arr[:, 0], np.asarray([1.0]), arr[:, 2][None, :], arr[:, 1][None, :]


def parse_intensity(path):
    mus = None
    with open(path) as fh:
        for _ in range(20):
            line = fh.readline()
            if not line.lstrip().startswith("#"): break
            if "mu-points" in line:
                mus = [float(x) for x in line.split("mu-points")[-1].split()]
    arr = np.loadtxt(path, comments="#")
    nmu = len(mus)
    I_abs = np.stack([arr[:, 3 + 2 * k] for k in range(nmu)]); I_norm = np.stack([arr[:, 4 + 2 * k] for k in range(nmu)])
    return arr[:, 0], np.asarray(mus), I_abs, I_norm


def emulate(points, wl, out_npz):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, "/scratch/y89/mj8805/spice_docsexp/src")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
    em = IntensityPretrainedAemuSpectrumEmulator("RozanskiT/TPayne-spice-harps")
    names = em.stellar_parameter_names
    f = jax.jit(lambda lw, mu, p: em.intensity(lw, mu, p))
    lw = jnp.log10(jnp.asarray(wl))
    out = {}
    for name, pt in points.items():
        d = dict(marcs_teff=pt["teff"], marcs_logg=pt["logg"], feh=pt["feh"], vmicro=pt["vmicro"], **pt["abund"])
        p = jnp.asarray([d[n] for n in names])
        t = time.time()
        arr = np.stack([np.asarray(f(lw, mu, p)) for mu in pt["mus"]]).astype(np.float32)  # (nmu, nwl, 2)
        out[f"{name}/emu_I"] = arr[..., 0]; out[f"{name}/emu_C"] = arr[..., 1]
        print(f"emulated {name} in {time.time()-t:.1f}s", flush=True)
    np.savez_compressed(out_npz, **out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--only", default="")
    ap.add_argument("--lam", type=float, nargs=2, default=(3780.0, 6910.0), help="wavelength range (smoke tests)")
    ap.add_argument("--flux", action="store_true", help="disc-integrated Flux mode instead of Intensity at MUS")
    ap.add_argument("--cno-fix", action="store_true", help="omit the c/n/o INDIVIDUAL ABUNDANCES lines (production passes them as absolute values, i.e. a bug)")
    args = ap.parse_args()
    out_root = args.out; os.makedirs(out_root, exist_ok=True)
    store_labels = RT._scan_available_turb_labels(STORE); print("store t-labels:", store_labels, flush=True)
    mu_file = RT._write_mu_points_file(MUS, f"{out_root}/tmp", "mus4")
    only = [s for s in args.only.split(",") if s]
    prepared = {}
    for name, pt in POINTS.items():
        if only and name not in only: continue
        try:
            if pt["kind"] == "valrow_fixed":  # replica with explicit snapped-atmosphere tuple
                pt = dict(pt)
            elif pt["kind"] == "valrow":
                r, row = valrow_params()
                pt = dict(teff=int(round(row["teff"])), logg=row["logg"], feh=row["feh"], vmicro=row["vmicro"], kind="valrow",
                          abund={k: row[k] for k in "acnors"}, mus=[row["mu"]], val_index=r,
                          marcs=(row["marcs_teff"], row["marcs_logg"], row["marcs_fe_h"], row["marcs_turb"]))
            else:
                pt = dict(pt, abund={**ABUND0, **pt.get("abund", {})}, mus=list(MUS))
            label = RT._nearest_turb_label(pt["vmicro"], store_labels)
            atm_dir = f"{out_root}/atm/{name}"; os.makedirs(atm_dir, exist_ok=True)
            target = os.path.join(atm_dir, RT.get_model_filename(int(pt["teff"]), pt["logg"], pt["feh"], label))
            if pt["kind"] in ("node", "valrow", "valrow_fixed"):
                # exact MARCS file (nearest-neighbour as in production), copied under the name the driver expects
                if pt["kind"] == "node":
                    src = os.path.join(STORE, marcs_file("p", pt["teff"], pt["logg"], pt["feh"], label))
                    if not os.path.exists(src): src = os.path.join(STORE, marcs_file("s", pt["teff"], pt["logg"], pt["feh"], label))
                else:
                    mt, mg, mz, mturb = pt["marcs"]
                    lab = f"{int(round(mturb)):02d}"
                    src = os.path.join(STORE, marcs_file("p", mt, mg, mz, lab))
                    if not os.path.exists(src): src = os.path.join(STORE, marcs_file("s", mt, mg, mz, lab))
                assert os.path.exists(src), src
                shutil.copyfile(src, target); info = dict(source=os.path.basename(src), geom=os.path.basename(src)[0])
            else:
                atm_feh = float(pt.get("atm_feh", pt["feh"]))  # the synthesis below always uses the continuous pt["feh"]
                paths, geom, lab, corners = find_brackets(pt["teff"], pt["logg"], atm_feh, label, store_labels)
                ok, log = interpolate(paths, pt["teff"], pt["logg"], atm_feh, target, f"{out_root}/tmp")
                if not ok: raise RuntimeError(f"interpolation failed for {name}: {log}")
                info = dict(geom=geom, label=lab, corners=corners, brackets=[os.path.basename(p) for p in paths], atm_feh=atm_feh)
            with open(target) as fh: first = fh.readline().strip()
            info["first_line"] = first; info["model_file"] = os.path.basename(target)
            pt["info"] = info; pt["atm_dir"] = atm_dir; pt["label"] = label
            prepared[name] = pt
            print(f"[prepared] {name}: {info}", flush=True)
        except Exception as e:
            print(f"[SKIP] {name}: {e}", flush=True); traceback.print_exc()
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in prepared.items()}, open(f"{out_root}/points.json", "w"), indent=1, default=str)
    if args.smoke:
        print("smoke done", flush=True); return

    # ---- synthesis (one process per point) ----
    jobs = []
    for name, pt in prepared.items():
        config = build_config(out_root, pt["atm_dir"], mu_file, lam=tuple(args.lam), flux_mode=args.flux)
        if pt["kind"] in ("valrow", "valrow_fixed"):
            llp = config.linelist_file_path
            config = dataclasses.replace(config, mu_points_file=RT._write_mu_points_file(pt["mus"], f"{out_root}/tmp", f"mus_{name}"))
            config.linelist_file_path = llp  # dataclasses.replace drops the dynamically added attribute
        ab = dict(pt["abund"])
        if args.cno_fix:
            for k in ("c", "n", "o"): ab.pop(k, None)
        params = dict(teff=int(pt["teff"]), logg=float(pt["logg"]), feh=float(pt["feh"]), turbvel=float(pt["vmicro"]),
                      t_value=pt["label"], **ab)
        jobs.append((name, params, config))
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_run_one, jobs)
    print(f"synthesis wall {time.time()-t0:.0f}s", flush=True)
    out = {}; wl_ref = None
    for (name, params, config), res in zip(jobs, results):
        print(name, res.get("status"), res.get("message", "")[:200], flush=True)
        if res.get("status") not in ("success", "ok", "done") and not (res.get("output_path") and os.path.exists(res.get("output_path", ""))):
            continue
        wl, mus, I_abs, I_norm = (parse_flux if args.flux else parse_intensity)(res["output_path"])
        if wl_ref is None: wl_ref = wl
        out[f"{name}/wl"] = wl.astype(np.float64); out[f"{name}/mus"] = mus
        out[f"{name}/ts_I"] = I_abs.astype(np.float32); out[f"{name}/ts_N"] = I_norm.astype(np.float32)
        prepared[name]["mus_out"] = mus.tolist()
    # validation-row reference spectrum from the zarr
    if "ctrl_valrow" in prepared and "ctrl_valrow/wl" in out:
        import zarr
        z = zarr.open(VAL_ZARR, mode="r"); r = prepared["ctrl_valrow"]["val_index"]
        out["ctrl_valrow/zarr_wl"] = z["wavelength"][:]; out["ctrl_valrow/zarr_flux"] = z["flux"][r, :]
        out["ctrl_valrow/zarr_cont"] = z["continuum"][r, :]; out["ctrl_valrow/zarr_cont_wl"] = z["continuum_wavelength"][:]
    np.savez_compressed(f"{out_root}/offnode_ts.npz", **out)
    json.dump(prepared, open(f"{out_root}/points.json", "w"), indent=1, default=str)
    # ---- emulator at the same parameters ----
    if wl_ref is None or args.flux:
        print("no spectra produced or flux mode; skipping emulator step", flush=True); return
    pts = {n: dict(teff=p["teff"], logg=p["logg"], feh=p["feh"], vmicro=p["vmicro"], abund=p["abund"], mus=p.get("mus_out", p["mus"]))
           for n, p in prepared.items() if f"{n}/wl" in out}
    emulate(pts, wl_ref, f"{out_root}/offnode_emu.npz")
    print("ALL DONE", flush=True)


def _run_one(job):
    name, params, config = job
    t = time.time()
    try:
        res = RT.run_single_synthesis((params, config))
    except Exception as e:
        res = dict(status="error", message=f"{e}\n{traceback.format_exc()}")
    res = dict(res); res["elapsed_s"] = time.time() - t
    print(f"[{name}] {res.get('status')} {res.get('elapsed_s'):.0f}s {str(res.get('message',''))[:300]}", flush=True)
    return res


if __name__ == "__main__":
    main()
