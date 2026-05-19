#!/usr/bin/env python
"""Stitch phase-chunked Cepheid pickles into a single combined pickle.

When ``run_cepheid_harps_rotating.pbs`` is submitted with
``PHASE_CHUNK_N>1`` each job writes to ``<OUT_DIR>/chunk_<i>/`` and produces

    <name>_bundles.pkl   {variant: CepheidBundle}    (snapshots = i-th slice)
    <name>_spectra.pkl   {variant: {line_center: LineSpectra}}  (same slicing)

This script discovers all ``chunk_*`` subdirectories of ``<OUT_DIR>``,
concatenates the per-snapshot lists in chunk order, and writes a merged
pair ``<OUT_DIR>/<name>_bundles.pkl`` / ``<OUT_DIR>/<name>_spectra.pkl``
that ``cepheid_analysis.ipynb`` can load directly.

Usage
-----
    python tutorial/paper_results/merge_phase_chunks.py <OUT_DIR>
    python tutorial/paper_results/merge_phase_chunks.py <OUT_DIR> --names cep_DeltaCep
    python tutorial/paper_results/merge_phase_chunks.py <OUT_DIR> --dry-run

Assumptions
-----------
* All chunk_* directories share the same wavelength grid and variant set.
  The script bails with a clear error if they don't.
* The ``template`` field of each ``LineSpectra`` is the t=0 snapshot and
  is identical across chunks; we keep chunk 0's copy.
* The ``CepheidBundle.template`` and ``CepheidBundle.pulsating`` fields
  are likewise chunk-invariant; chunk 0's copies are kept.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from cepheid_bundles import CepheidBundle, LineSpectra, load_pickle, save_pickle  # noqa: E402


CHUNK_RE = re.compile(r"^chunk_(\d+)$")


def _ordered_chunk_dirs(out_dir: Path) -> list[Path]:
    pairs = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        m = CHUNK_RE.match(child.name)
        if m:
            pairs.append((int(m.group(1)), child))
    pairs.sort()
    return [p for _, p in pairs]


def _discover_names(chunk_dirs: Iterable[Path]) -> list[str]:
    names: set[str] = set()
    for d in chunk_dirs:
        for f in d.glob("*_bundles.pkl"):
            names.add(f.name.removesuffix("_bundles.pkl"))
    return sorted(names)


def _merge_bundles(chunk_payloads: list[dict]) -> dict:
    keys = set(chunk_payloads[0].keys())
    for i, p in enumerate(chunk_payloads[1:], start=1):
        if set(p.keys()) != keys:
            raise SystemExit(
                f"chunk {i} bundle keys {sorted(p.keys())} != chunk 0 {sorted(keys)}"
            )
    merged: dict = {}
    for variant in sorted(keys):
        first: CepheidBundle = chunk_payloads[0][variant]
        snapshots = []
        for p in chunk_payloads:
            snapshots.extend(p[variant].snapshots or [])
        merged[variant] = CepheidBundle(
            pulsating=first.pulsating,
            snapshots=snapshots,
            template=first.template,
        )
    return merged


def _merge_spectra(chunk_payloads: list[dict]) -> dict:
    keys = set(chunk_payloads[0].keys())
    for i, p in enumerate(chunk_payloads[1:], start=1):
        if set(p.keys()) != keys:
            raise SystemExit(
                f"chunk {i} spectrum variants {sorted(p.keys())} != chunk 0 {sorted(keys)}"
            )
    merged: dict = {}
    for variant in sorted(keys):
        line_keys = set(chunk_payloads[0][variant].keys())
        for i, p in enumerate(chunk_payloads[1:], start=1):
            if set(p[variant].keys()) != line_keys:
                raise SystemExit(
                    f"chunk {i} variant {variant!r} line centers differ from chunk 0"
                )
        per_variant: dict = {}
        for lc in sorted(line_keys):
            first: LineSpectra = chunk_payloads[0][variant][lc]
            wl = np.asarray(first.wavelengths)
            for i, p in enumerate(chunk_payloads[1:], start=1):
                this_wl = np.asarray(p[variant][lc].wavelengths)
                if this_wl.shape != wl.shape or not np.allclose(this_wl, wl):
                    raise SystemExit(
                        f"chunk {i} wavelength grid for variant={variant!r} "
                        f"line={lc} disagrees with chunk 0"
                    )
            spectra = []
            for p in chunk_payloads:
                spectra.extend(p[variant][lc].spectra)
            per_variant[lc] = LineSpectra(
                wavelengths=wl,
                spectra=spectra,
                template=first.template,
            )
        merged[variant] = per_variant
    return merged


def _merge_one(name: str, chunk_dirs: list[Path], out_dir: Path, dry_run: bool) -> None:
    bundle_chunks: list[dict] = []
    spectra_chunks: list[dict] = []
    for d in chunk_dirs:
        bp = d / f"{name}_bundles.pkl"
        sp = d / f"{name}_spectra.pkl"
        if not bp.exists() or not sp.exists():
            print(f"  skip chunk {d.name}: missing {bp.name} or {sp.name}")
            return
        bundle_chunks.append(load_pickle(str(bp)))
        spectra_chunks.append(load_pickle(str(sp)))

    merged_bundles = _merge_bundles(bundle_chunks)
    merged_spectra = _merge_spectra(spectra_chunks)

    n_snap = len(next(iter(merged_bundles.values())).snapshots or [])
    per_variant = next(iter(merged_spectra.values()))
    n_spec = len(next(iter(per_variant.values())).spectra)
    print(
        f"  {name}: {len(bundle_chunks)} chunk(s) → "
        f"{n_snap} snapshots, {n_spec} spectra per line "
        f"({len(merged_bundles)} bundle / {len(merged_spectra)} spectrum variants)"
    )

    if dry_run:
        return

    out_b = out_dir / f"{name}_bundles.pkl"
    out_s = out_dir / f"{name}_spectra.pkl"
    save_pickle(merged_bundles, str(out_b))
    save_pickle(merged_spectra, str(out_s))
    print(f"  → wrote {out_b.name} + {out_s.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path,
                   help="Directory containing chunk_0/, chunk_1/, ... subdirectories.")
    p.add_argument("--names", nargs="+", default=None,
                   help="Subset of Cepheid names to merge (default: all discovered).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be merged without writing pickles.")
    args = p.parse_args()

    out_dir: Path = args.out_dir.resolve()
    if not out_dir.is_dir():
        raise SystemExit(f"{out_dir} is not a directory")

    chunk_dirs = _ordered_chunk_dirs(out_dir)
    if not chunk_dirs:
        raise SystemExit(f"no chunk_* subdirectories under {out_dir}")
    print(f"Found {len(chunk_dirs)} chunk(s): {[d.name for d in chunk_dirs]}")

    names = args.names or _discover_names(chunk_dirs)
    if not names:
        raise SystemExit("no <name>_bundles.pkl files found in any chunk directory")
    print(f"Merging {len(names)} Cepheid(s): {names}")

    for name in names:
        _merge_one(name, chunk_dirs, out_dir, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
