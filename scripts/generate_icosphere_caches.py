"""Pre-generate icosphere caches shipped with the package.

Writes ``src/spice/icosphere_cache/icosphere_{N}_compat.pickle`` for each
requested subdivision level. The package loader looks for these via
``pkgutil.get_data('spice', 'icosphere_cache/icosphere_{N}_compat.pickle')``
so anything written here will be picked up after a re-install.

Usage:
    python scripts/generate_icosphere_caches.py            # subdivs 0..6
    python scripts/generate_icosphere_caches.py --max 5    # subdivs 0..5
    python scripts/generate_icosphere_caches.py --force    # overwrite existing
"""
import argparse
import os
import pickle
import sys
import time

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH   = os.path.join(REPO_ROOT, "src")
CACHE_DIR  = os.path.join(SRC_PATH, "spice", "icosphere_cache")
sys.path.insert(0, SRC_PATH)

from spice.models.mesh_generation import _icosphere  # noqa: E402


def faces_for(subdiv: int) -> int:
    return 5 * 4 ** (subdiv + 1)


def vertices_for(subdiv: int) -> int:
    return 4 * (5 * 4 ** subdiv - 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=6,
                        help="Maximum subdivision level to generate (inclusive).")
    parser.add_argument("--min", type=int, default=0,
                        help="Minimum subdivision level to generate (inclusive).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite caches that already exist.")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache dir: {CACHE_DIR}")

    for subdiv in range(args.min, args.max + 1):
        out_path = os.path.join(CACHE_DIR, f"icosphere_{subdiv}_compat.pickle")
        rel_path = os.path.relpath(out_path, REPO_ROOT)

        if os.path.exists(out_path) and not args.force:
            print(f"  [skip]  subdiv={subdiv}  ({rel_path} exists; use --force to rebuild)")
            continue

        n_verts = vertices_for(subdiv)
        n_faces = faces_for(subdiv)
        print(f"  [build] subdiv={subdiv}  ({n_verts} verts, {n_faces} faces) ...", flush=True)
        t0 = time.perf_counter()
        verts, faces, areas, centers = _icosphere(subdiv)
        dt = time.perf_counter() - t0

        with open(out_path, "wb") as fh:
            pickle.dump((verts, faces, areas, centers), fh)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  [write] subdiv={subdiv}  -> {rel_path}  ({size_kb:.1f} KB, {dt:.1f} s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
