#!/usr/bin/env python
"""Merge per-Cepheid bundle/spectra pickles from one build directory into another.

Use when the linear-LD sweep was written to ``cepheid_ld_sweep/`` separately from
the main grid in ``cepheid_grid/``. After merging, ``cepheid_analysis.ipynb`` can
load a single directory with ``with_ld``, ``harps``, ``iron_line``, and
``flux_linear_*`` variants together.

Example (on Gadi)::

    python merge_cepheid_pickles.py \\
        --dst /home/100/mj8805/scr_y89/spice/cepheid_grid \\
        --src /scratch/y89/$USER/cepheid_ld_sweep \\
        --names cep_DeltaCep cep_DeltaCep_norot
"""
from __future__ import annotations

import argparse
import pickle
import sys
import os
from pathlib import Path

# Add project src to sys.path
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spice.cepheid_bundles import save_pickle, load_pickle


def _merge_dict_pickle(dst: Path, src: Path) -> None:
    if not src.exists():
        return
    incoming = load_pickle(str(src))
    if not isinstance(incoming, dict):
        raise TypeError(f"{src} is not a dict pickle")
    if dst.exists():
        base = load_pickle(str(dst))
        if not isinstance(base, dict):
            raise TypeError(f"{dst} is not a dict pickle")
        base.update(incoming)
        merged = base
    else:
        merged = incoming
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(merged, str(dst))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dst", type=Path, required=True,
                   help="Destination directory (e.g. cepheid_grid/).")
    p.add_argument("--src", type=Path, required=True,
                   help="Source directory with extra variants to merge in.")
    p.add_argument("--names", nargs="+", required=True, help="Cepheid base names.")
    args = p.parse_args()

    for name in args.names:
        for suffix in ("bundles", "spectra"):
            _merge_dict_pickle(
                args.dst / f"{name}_{suffix}.pkl",
                args.src / f"{name}_{suffix}.pkl",
            )
        print(f"merged {name} → {args.dst.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
