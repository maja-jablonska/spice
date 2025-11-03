#!/usr/bin/env python3
"""
Scan a grid of expected eclipse calculation files and report the ones that are
missing or unreadable (.pkl can't be opened/loaded).

Filename pattern example:
eclipses_incl_85.0_period_0.25_q_0.5_ecc_0.0_primary_mass_5.0.pkl
"""

from __future__ import annotations
import argparse
import itertools as it
import pickle
from pathlib import Path
from decimal import Decimal, InvalidOperation

# --------------------------- config ---------------------------------

# Grids (kept as strings to preserve exact decimal forms when possible)
INCLINATIONS = ["80.", "85.", "90."]               # one decimal place
PERIODS      = ["0.1", "0.25", "0.5", "1.", "3.", "5.", "7.", "10.", "15.", "20.", "50."]
QS           = ["0.5", "0.6", "0.7", "0.8", "0.9", "1."]
ECCS         = ["0.", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
PRIMARY_MASS = ["1.", "3.", "5.", "7.", "9."]      # one decimal place

# --------------------------- helpers --------------------------------

def normalize_num_str(x: str) -> str:
    """
    Convert a numeric string to a normalized representation *matching your examples*:
      - Remove unnecessary trailing zeros, but ensure at least one decimal for integers
      - Preserve exact strings like '0.25' and produce '1.0' from '1.' or '1'
    """
    try:
        d = Decimal(x)
    except InvalidOperation:
        # If it's already a quirky literal like "1.", fall back to manual handling
        x = x.strip()
        if x.endswith("."):
            return x[:-1] + ".0"
        return x

    # Remove trailing zeros/exponent
    s = format(d.normalize(), 'f')  # plain decimal form
    if '.' not in s:
        s += '.0'
    return s

def fname(incl: str, period: str, q: str, ecc: str, mass: str, prefix: str) -> str:
    """
    Build the expected filename using the normalized string rules.
    """
    incl_s   = normalize_num_str(incl)
    period_s = normalize_num_str(period)
    q_s      = normalize_num_str(q)
    ecc_s    = normalize_num_str(ecc)
    mass_s   = normalize_num_str(mass)
    return f"{prefix}_incl_{incl_s}_period_{period_s}_q_{q_s}_ecc_{ecc_s}_primary_mass_{mass_s}.pkl"

def is_pickle_readable(path: Path) -> bool:
    """
    Try loading the pickle to ensure it's actually readable.
    """
    try:
        with path.open("rb") as f:
            # Load only enough to confirm unpickling works; full load is fine here.
            _ = pickle.load(f)
        return True
    except Exception:
        return False

# --------------------------- main -----------------------------------

def main():
    p = argparse.ArgumentParser(description="Identify missing or unreadable eclipse calculation files.")
    p.add_argument(
        "-d", "--directory", type=Path, default=Path("."),
        help="Directory to scan (default: current directory)."
    )
    p.add_argument(
        "--prefix", default="eclipses",
        help="Filename prefix before parameter fields (default: 'eclipses')."
    )
    p.add_argument(
        "--list-unreadable", action="store_true",
        help="Also list files that exist but are unreadable/corrupted."
    )
    p.add_argument(
        "--write-missing", type=Path, default=None,
        help="Optionally write missing filenames to a text file."
    )
    p.add_argument(
        "--write-unreadable", type=Path, default=None,
        help="Optionally write unreadable filenames to a text file."
    )

    args = p.parse_args()
    root: Path = args.directory

    combos = list(it.product(INCLINATIONS, PERIODS, QS, ECCS, PRIMARY_MASS))
    total = len(combos)

    missing: list[str] = []
    unreadable: list[str] = []

    for incl, period, q, ecc, mass in combos:
        filename = fname(incl, period, q, ecc, mass, prefix=args.prefix)
        path = root / filename

        if not path.exists():
            missing.append(str(path))
        else:
            if not is_pickle_readable(path):
                unreadable.append(str(path))

    ok = total - len(missing) - len(unreadable)

    # ---------------------- reporting ----------------------
    print("=== Eclipse Grid Audit ===")
    print(f"Directory        : {root.resolve()}")
    print(f"Filename prefix  : {args.prefix}")
    print(f"Grid size        : {total}")
    print(f"Readable files   : {ok}")
    print(f"Missing files    : {len(missing)}")
    print(f"Unreadable files : {len(unreadable)}")

    if missing:
        print("\n-- Missing files --")
        for fp in missing:
            print(fp)

    if args.list_unreadable and unreadable:
        print("\n-- Unreadable/Corrupted files --")
        for fp in unreadable:
            print(fp)

    if args.write_missing:
        args.write_missing.write_text("\n".join(missing) + ("\n" if missing else ""))
        print(f"\nWrote missing list to: {args.write_missing.resolve()}")

    if args.write_unreadable:
        args.write_unreadable.write_text("\n".join(unreadable) + ("\n" if unreadable else ""))
        print(f"Wrote unreadable list to: {args.write_unreadable.resolve()}")

if __name__ == "__main__":
    main()
