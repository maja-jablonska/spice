"""Download the data behind the SPICE paper from Hugging Face to the paths the paper scripts expect.

The files live in the public dataset https://huggingface.co/datasets/mjablonska/spice-paper-data (revision v1.1):

    kochukhov2017_fe_lte/   LTE Fe grid of the spotted-star test (Sect. 3.1) and its synthesised profiles
    dcep/                   delta Cep literature-model runs and the synthetic p-factor spectra (Sect. 3.2)
    hbeta/                  Turbospectrum and emulator spectra over H-beta (Appendix F, Figs. F1-F2)

    python tutorial/paper_results/fetch_paper_data.py [--only kochukhov dcep hbeta] [--revision v1.1]

Needs huggingface_hub (pip install "stellar-spice[huggingface]"). Files already present are skipped.
"""
import argparse
import os
import shutil
from pathlib import Path

REPO_ID = "mjablonska/spice-paper-data"
REVISION = "v1.1"
ROOT = Path(__file__).resolve().parents[2]          # repository root
PR = ROOT / "tutorial" / "paper_results"

# dataset path (file or directory prefix) -> local destination
FILES = {
    "kochukhov": {
        "kochukhov2017_fe_lte/kochukhov2017_fe_lte.zarr": ROOT / "data" / "kochukhov2017_fe_lte.zarr",
        "kochukhov2017_fe_lte/kochukhov2017_fe_lte_spectra.pkl": PR / "spots" / "kochukhov2017_fe_lte_spectra.pkl",
    },
    "dcep": {
        **{f"dcep/literature_models/dcep_v12x150_{a}_{b}_p3.pkl": PR / "cepheid" / "dcep_out" / f"dcep_v12x150_{a}_{b}_p3.pkl"
           for a in ("own", "all") for b in ("trahin", "luck")},
        "dcep/literature_models/dcep_v12x150_own_trahin_p3_vmacpc.pkl": PR / "cepheid" / "dcep_out" / "dcep_v12x150_own_trahin_p3_vmacpc.pkl",
        # synth_paper_atm / lin0 are reached through the symlinks in pfactor_v11/out_v120_m009/
        "dcep/pfactor_synthesis/synth_paper_atm.npz": PR / "cepheid" / "pfactor_v11" / "out_m009" / "synth_paper_atm.npz",
        "dcep/pfactor_synthesis/synth_paper_lin0.npz": PR / "cepheid" / "pfactor_v11" / "out_m009" / "synth_paper_lin0.npz",
        "dcep/pfactor_synthesis/synth_paper_tpayne.npz": PR / "cepheid" / "pfactor_v11" / "out_v120_m009" / "synth_paper_tpayne.npz",
    },
    "hbeta": {
        f"hbeta/{f}": PR / "emulator_validation" / "results" / "interp" / "hbeta" / f
        for f in ("offnode_ts.npz", "offnode_emu_sw_pen1z.npz")
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FILES), default=sorted(FILES))
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()
    from huggingface_hub import list_repo_files, snapshot_download

    remote = list_repo_files(REPO_ID, repo_type="dataset", revision=a.revision)
    staging = ROOT / "data" / ".paper_data_download"   # data/ is git-ignored
    for group in a.only:
        for src, dst in FILES[group].items():
            if dst.exists():
                print(f"present  {dst.relative_to(ROOT)}")
                continue
            members = [f for f in remote if f == src or f.startswith(src + "/")]
            if not members:
                raise SystemExit(f"{src} not found in {REPO_ID}@{a.revision}")
            snapshot_download(REPO_ID, repo_type="dataset", revision=a.revision, allow_patterns=members,
                              local_dir=staging)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / src, dst)                   # same filesystem: a move, not a copy
            print(f"fetched  {dst.relative_to(ROOT)}")
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
