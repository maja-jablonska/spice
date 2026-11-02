"""Convert a block_pca_emulator.py MLP checkpoint (params.pkl, an nnx.State of Net.layers[i].{kernel,bias}) to the
weights.npz (W0, b0, W1, b1, ...) that block_pca_spice.BlockPCAIntensityEmulator loads.

    python bp_params_to_npz.py RUN_DIR [--check REF_weights.npz]
"""
import pickle
import sys
from pathlib import Path

import jax
import numpy as np


def convert(run):
    st = pickle.load(open(Path(run) / "params.pkl", "rb"))
    W, b = {}, {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(st)[0]:
        keys = [getattr(k, "key", getattr(k, "idx", getattr(k, "name", None))) for k in path]
        idx = [k for k in keys if isinstance(k, int)]
        name = [k for k in keys if isinstance(k, str) and k in ("kernel", "bias")]
        if len(idx) != 1 or len(name) != 1:
            raise ValueError(f"unexpected leaf path {keys}")
        (W if name[0] == "kernel" else b)[idx[0]] = np.asarray(leaf)
    assert sorted(W) == sorted(b) == list(range(len(W))), (sorted(W), sorted(b))
    out = {}
    for i in range(len(W)):
        out[f"W{i}"], out[f"b{i}"] = W[i], b[i]
        assert W[i].shape[1] == b[i].shape[0]
        if i:
            assert W[i - 1].shape[1] == W[i].shape[0]
    return out


def main():
    run = sys.argv[1]
    out = convert(run)
    if "--check" in sys.argv:
        ref = np.load(sys.argv[sys.argv.index("--check") + 1])
        assert sorted(ref.files) == sorted(out), (ref.files, list(out))
        for k in out:
            assert ref[k].shape == out[k].shape and np.array_equal(ref[k], out[k]), k
        print(f"CHECK OK: {run} reproduces {sys.argv[sys.argv.index('--check') + 1]} exactly ({len(out) // 2} layers)")
        return
    np.savez(Path(run) / "weights.npz", **out)
    print(f"wrote {Path(run) / 'weights.npz'}: " + ", ".join(f"W{i} {out[f'W{i}'].shape}" for i in range(len(out) // 2)))


if __name__ == "__main__":
    main()
