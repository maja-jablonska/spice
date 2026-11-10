"""Select Gaia FGK benchmark stars that resemble the two TZ For components and fetch their HARPS spectra.

Reference values: data/fgk.txt (the GBS v3 compilation on Gadi, /scratch/mk27/mj8805/Turbospectrum_NLTE/fgk.txt):
Teff from the angular diameter and bolometric flux, log g from mass and radius, spectroscopic [Fe/H].
The selection is fixed before any fit and does not look at the spectra:

  giants  (TZ For primary, 4817-4930 K, log g 2.91, [Fe/H] -0.3):  4700 <= Teff <= 5100, 2.3 <= log g <= 3.1
  F stars (TZ For secondary, 6160-6350 K, log g 3.54):             5900 <= Teff <= 6700, 3.7 <= log g <= 4.4
  both: -0.7 <= [Fe/H] <= +0.4, sigma(Teff) <= 70 K, no flag (no SB2, no indirect angular diameter),
        a HARPS row in the table with S/N >= 200.

For each star the three highest-S/N HARPS 1D spectra (ESO Phase 3, ADP s1d) are downloaded from the ESO
archive (TAP + dataportal), found by a 72-arcsec cone around the SIMBAD position (proper motions up to
~1.3 arcsec/yr over the HARPS era) and checked against the target name.

Runs locally (astro env: requests, astropy). Writes data/targets.csv and data/harps/<star>/<dp_id>.fits.
Usage: python gbs_fetch.py [--n-files 3] [--dry-run]
"""
import argparse, csv, io, os, re, sys, time

import requests

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
SIMBAD = ["https://simbad.cfa.harvard.edu/simbad/sim-tap/sync", "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"]   # CfA mirror first: CDS was unreachable 2026-11-01
ESO_TAP = "https://archive.eso.org/tap_obs/sync"
ESO_TAP_BASE = "https://archive.eso.org/tap_obs"   # async jobs: the sync endpoint answers 400/502 when busy
SNR_CAP = float(os.environ.get("GBS_SNR_CAP", 580.0))   # the 2014 HARPS reprocessing reports S/N ~600-620 for saturated or corrupt exposures (filled cores, negative flux)
ESO_FILE = "https://dataportal.eso.org/dataPortal/file/{}"

SEL = {"giant": dict(teff=(4700, 5100), logg=(2.3, 3.1)), "fstar": dict(teff=(5900, 6700), logg=(3.7, 4.4))}
FEH = (-0.7, 0.4); E_TEFF_MAX = 70.0; SNR_MIN = 200.0
# --extended (added 2026-11-01, after the first calibration): more F/G dwarfs for the Teff systematic and metal-poor stars to test
# whether the [Fe/H] offset is a constant zero-point. Looser: sigma(Teff) <= 100 K, S/N >= 150, the 'Indirect theta' flag allowed
# for the metal-poor group (kept in the 'flag' column). Metal-poor stars get [alpha/Fe] = GBS [Mg/H] - [Fe/H] in the fit.
SEL_EXT = {"dwarf": dict(teff=(5700, 6700), logg=(3.7, 4.6), feh=(-0.7, 0.4), flags=("-",)),
           "metalpoor": dict(teff=(4400, 6700), logg=(-1.0, 5.0), feh=(-3.5, -0.7), flags=("-", "Indirect theta"))}
E_TEFF_MAX_EXT = 100.0; SNR_MIN_EXT = 150.0


def tap(url, query, timeout=300, tries=5):
    """Synchronous TAP query; the ESO service answers 400 or times out when it is busy, so retry with backoff."""
    for k in range(tries):
        try:
            r = requests.get(url, params=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=query), timeout=timeout)
            r.raise_for_status()
            return list(csv.DictReader(io.StringIO(r.text)))
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            print(f"    TAP retry {k + 1} after: {str(e)[:120]}", flush=True); time.sleep(20 * (k + 1))


def select_extended(path, exclude):
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["origin"] != "HARPS" or r["star_alt1"] in exclude or r["star_alt1"] == "-":
            continue
        try:
            teff, eteff, logg, feh, snr = float(r["Teff"]), float(r["eTeff"]), float(r["logg"]), float(r["[Fe/H]"]), float(r["snr"])
        except ValueError:
            continue
        kind = next((k for k, s in SEL_EXT.items() if s["teff"][0] <= teff <= s["teff"][1] and s["logg"][0] <= logg <= s["logg"][1]
                     and s["feh"][0] <= feh <= s["feh"][1] and r["flag"].strip() in s["flags"]), None)
        if kind is None or eteff > E_TEFF_MAX_EXT or snr < SNR_MIN_EXT:
            continue
        star = r["star_alt1"]
        alpha = float(r["[Mg/H]"]) - feh if kind == "metalpoor" and r["[Mg/H]"] not in ("", "0.0") else 0.0
        if star not in out or snr > out[star]["snr_table"]:
            out[star] = dict(star=star, hip=r["star"], kind=kind, teff=teff, eteff=eteff, logg=logg, elogg=float(r["elogg"]), feh=feh,
                             efeh=float(r["e[Fe/H]"]), vmic=float(r["vmic"]), vmac=float(r["vmac"]), vsini=float(r["vsini"]), snr_table=snr,
                             flag=r["flag"].strip(), alpha=round(alpha, 3))
    return list(out.values())


def select_wide(path, exclude):
    """--wide: every GBS star in the method's range, whether or not the table lists a HARPS spectrum (the archive has HARPS
    spectra of many stars the GBS took from other instruments): 4400 <= Teff <= 6700, sigma(Teff) <= 100, not SB2.
    kind: 'metalpoor' below [Fe/H] -0.7 (alpha = GBS [Mg/H] - [Fe/H]), else 'giant_w' (log g < 3.5) or 'dwarf_w'."""
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        star = r["star_alt1"] if r["star_alt1"] != "-" else r["star"]
        if star in exclude or star in out or star == "Sun" or r["flag"].strip() == "SB2":
            continue
        try:
            teff, eteff, logg, feh = float(r["Teff"]), float(r["eTeff"]), float(r["logg"]), float(r["[Fe/H]"])
        except ValueError:
            continue
        if not (4400 <= teff <= 6700) or eteff > E_TEFF_MAX_EXT:
            continue
        kind = "metalpoor" if feh < -0.7 else ("giant_w" if logg < 3.5 else "dwarf_w")
        alpha = float(r["[Mg/H]"]) - feh if kind == "metalpoor" and r["[Mg/H]"] not in ("", "0.0") else 0.0
        out[star] = dict(star=star, hip=r["star"], kind=kind, teff=teff, eteff=eteff, logg=logg, elogg=float(r["elogg"]), feh=feh,
                         efeh=float(r["e[Fe/H]"]), vmic=float(r["vmic"]), vmac=float(r["vmac"]), vsini=float(r["vsini"]), snr_table="",
                         flag=r["flag"].strip(), alpha=round(alpha, 3))
    return list(out.values())


def select(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    out = {}
    for r in rows:
        if r["origin"] != "HARPS":
            continue
        try:
            teff, eteff, logg, feh, snr = float(r["Teff"]), float(r["eTeff"]), float(r["logg"]), float(r["[Fe/H]"]), float(r["snr"])
        except ValueError:
            continue
        kind = next((k for k, s in SEL.items() if s["teff"][0] <= teff <= s["teff"][1] and s["logg"][0] <= logg <= s["logg"][1]), None)
        if kind is None or not (FEH[0] <= feh <= FEH[1]) or eteff > E_TEFF_MAX or snr < SNR_MIN or r["flag"].strip() != "-":
            continue
        star = r["star_alt1"]
        out[star] = dict(star=star, hip=r["star"], kind=kind, teff=teff, eteff=eteff, logg=logg, elogg=float(r["elogg"]), feh=feh,
                         efeh=float(r["e[Fe/H]"]), vmic=float(r["vmic"]), vmac=float(r["vmac"]), vsini=float(r["vsini"]), snr_table=snr)
    return list(out.values())


def tap_async(base, query, tries=3):
    """Asynchronous TAP job (pyvo); far more reliable than the ESO sync endpoint."""
    import pyvo
    for k in range(tries):
        try:
            t = pyvo.dal.TAPService(base).run_async(query, timeout=600).to_table()
            return [{c: (str(v) if c != "snr" else float(v)) for c, v in zip(t.colnames, row)} for row in t]
        except Exception as e:
            if k == tries - 1:
                raise requests.RequestException(str(e))
            print(f"    async TAP retry {k + 1}: {str(e)[:100]}", flush=True); time.sleep(30)


def simbad_name(star):
    """'HD168723' -> 'HD 168723'; other names (Procyon) unchanged."""
    m = re.fullmatch(r"(HD|HIP)(\d+)", star)
    return f"{m.group(1)} {m.group(2)}" if m else star


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-files", type=int, default=3); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extended", action="store_true", help="the extended dwarf + metal-poor selection -> data/targets_ext.csv")
    ap.add_argument("--wide", action="store_true", help="all GBS stars in range, HARPS row or not -> data/targets_wide.csv")
    ap.add_argument("--min-snr", type=float, default=0.0, help="drop archive spectra below this S/N (--wide uses 100)")
    args = ap.parse_args()
    targets = select(os.path.join(DATA, "fgk.txt"))
    if args.extended or args.wide:
        ext = select_extended(os.path.join(DATA, "fgk.txt"), {t["star"] for t in targets})
        targets = ext if args.extended else select_wide(os.path.join(DATA, "fgk.txt"), {t["star"] for t in targets} | {t["star"] for t in ext})
        if args.wide:
            args.min_snr = max(args.min_snr, 100.0)
    tag = "_ext" if args.extended else "_wide" if args.wide else ""
    print(f"{len(targets)} targets: " + ", ".join(f"{t['star']} ({t['kind']}, {t['teff']:.0f}+-{t['eteff']:.0f} K, {t['logg']:.2f}, {t['feh']:+.2f})" for t in targets), flush=True)

    for t in targets:
        nm = simbad_name(t["star"])
        q_pos = f"SELECT b.ra, b.dec FROM basic AS b JOIN ident AS i ON b.oid = i.oidref WHERE i.id = '{nm}'"
        res = None
        for url in SIMBAD:
            try:
                res = tap(url, q_pos, timeout=60, tries=2); break
            except requests.RequestException as e:
                print(f"    SIMBAD {url.split('/')[2]} failed: {str(e)[:80]}", flush=True)
        if res is None:
            print(f"  {t['star']}: SIMBAD query failed; rerun to retry", flush=True); t["n_files"] = 0; continue
        if not res:
            print(f"  {t['star']}: not in SIMBAD as '{nm}'", flush=True); t["n_files"] = 0; continue
        ra, dec = float(res[0]["ra"]), float(res[0]["dec"]); t["ra"], t["dec"] = ra, dec
        if dec > 35.0:                                           # HARPS (La Silla, -29 deg): nothing useful north of +35
            print(f"  {t['star']}: dec {dec:+.1f}, not observable with HARPS, skipped", flush=True); t["n_files"] = 0; continue
        # no TOP / ORDER BY: on a field with no HARPS data the ESO server scans the whole table and answers 502 after ~90 s;
        # all rows are fetched and sorted by S/N here instead
        q = (f"SELECT dp_id, target_name, snr FROM ivoa.ObsCore WHERE obs_collection='HARPS' AND dataproduct_type='spectrum' "
             f"AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, 0.02))=1")
        have = sorted(f[:-5] for f in os.listdir(os.path.join(DATA, "harps", t["star"]))) if os.path.isdir(os.path.join(DATA, "harps", t["star"])) else []
        if len(have) >= args.n_files and not args.dry_run:      # already fetched: no archive query
            t["n_files"] = len(have); t["dp_ids"] = ";".join(have); print(f"  {t['star']}: {len(have)} spectra on disk", flush=True); continue
        try:
            try:                                                 # sync answers in seconds when ESO is healthy; async jobs die when it is not
                rows = tap(ESO_TAP_BASE + "/sync", q, timeout=60, tries=6)
            except requests.RequestException:
                rows = tap_async(ESO_TAP_BASE, q, tries=1)
        except requests.RequestException as e:
            print(f"  {t['star']}: archive query failed ({str(e)[:80]}); rerun to retry", flush=True); t["n_files"] = len(have); continue
        rows = sorted((r for r in rows if r["snr"] not in ("", "--", "nan") and args.min_snr <= float(r["snr"]) < SNR_CAP), key=lambda r: -float(r["snr"]))
        key = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
        names = {key(r["target_name"]) for r in rows}
        rows = rows[:args.n_files]
        t["n_files"] = len(rows); t["eso_targets"] = ";".join(sorted({r["target_name"] for r in rows})); t["dp_ids"] = ";".join(r["dp_id"] for r in rows)
        t["snr_files"] = ";".join(f"{float(r['snr']):.0f}" for r in rows)
        print(f"  {t['star']}: {len(rows)} HARPS spectra, S/N {t['snr_files']}, ESO names {sorted(names)}", flush=True)
        if args.dry_run:
            continue
        d = os.path.join(DATA, "harps", t["star"]); os.makedirs(d, exist_ok=True)
        for r in rows:
            path = os.path.join(d, r["dp_id"] + ".fits")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            for attempt in range(3):
                try:
                    with requests.get(ESO_FILE.format(r["dp_id"]), timeout=300, stream=True) as resp:
                        resp.raise_for_status()
                        with open(path + ".part", "wb") as fh:
                            for chunk in resp.iter_content(1 << 20):
                                fh.write(chunk)
                    os.replace(path + ".part", path); break
                except requests.RequestException as e:
                    print(f"    {r['dp_id']}: {e} (attempt {attempt + 1})", flush=True); time.sleep(5)
    cols = ["star", "hip", "kind", "teff", "eteff", "logg", "elogg", "feh", "efeh", "vmic", "vmac", "vsini", "snr_table", "ra", "dec", "n_files", "snr_files", "eso_targets", "dp_ids"]
    if tag:
        cols += ["flag", "alpha"]
    with open(os.path.join(DATA, f"targets{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(targets)
    print(f"wrote {os.path.join(DATA, f'targets{tag}.csv')}", flush=True)


if __name__ == "__main__":
    main()
