"""SuppNet normalisation via an .npz handoff, to keep zarr out of suppnet-env.

``harps_suppnet_normalize.py`` imports zarr at module level, but suppnet-env
holds TensorFlow 2.15 pinned to ``numpy<2`` and has no zarr/numcodecs.
Installing them there risks the same kind of breakage that took out the astro
env earlier, so instead:

    export  (astro env)        zarr  -> spectra.npz
    normalize (suppnet-env)    npz   -> normalized.npz
    ingest  (astro env)        npz   -> back into the zarr store

The normalisation itself is the *same* call sequence as the established
script's ``suppnet_normalize_row``: divide by the median, ``nn.normalize``,
``get_smoothed_continuum``, then flux / smoothed continuum.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def cmd_export(args):
    import zarr
    root = zarr.open_group(str(HERE / args.zarr), mode="r")
    np.savez_compressed(
        args.npz,
        wavelengths=root["wavelengths"][:],
        fluxes=root["fluxes"][:],
        files=np.array(list(root.attrs["raw_source_files"])),
    )
    print(f"exported {root['wavelengths'].shape[0]} spectra -> {args.npz}")


def cmd_normalize(args):
    repo = Path(os.environ.get("SUPPNET_REPO", Path.home() / "code/suppnet")).expanduser()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # SUPPNet's published weights were written with the pre-2.11 Keras
    # optimizer, and TF >= 2.11 refuses to restore those slots into the new
    # one ("trying to restore a checkpoint from a legacy Keras optimizer").
    # suppnet/SUPPNet.py does `from tensorflow.keras.optimizers import Nadam`
    # at import time and compiles with it, so swap in the legacy class *before*
    # importing suppnet. The optimizer is irrelevant here -- this is inference
    # only -- and this leaves both the env and the SUPPNet checkout untouched.
    # Patch the name in the module that *uses* it: suppnet/SUPPNet.py did
    # `from tensorflow.keras.optimizers import Nadam` at import time, and
    # tf.keras is a lazy module, so rebinding tf.keras.optimizers.Nadam does
    # not reach the already-bound symbol.
    import tensorflow as tf
    import suppnet.SUPPNet as _S
    _S.Nadam = tf.keras.optimizers.legacy.Nadam

    from suppnet.NN_utility import get_smoothed_continuum, get_suppnet  # noqa: E402

    d = np.load(args.npz, allow_pickle=True)
    waves, fluxes, files = d["wavelengths"], d["fluxes"], d["files"]
    # Same construction as harps_suppnet_normalize.py: norm_only=False so
    # normalize() returns (continuum, continuum_err, segmentation, seg_err).
    nn = get_suppnet(resampling_step=args.resampling_step, step_size=256,
                     norm_only=False, which_weights=args.weights)

    out_norm, out_cont, out_wave = [], [], []
    for i in range(waves.shape[0]):
        w, f = waves[i], fluxes[i]
        m = np.isfinite(w) & np.isfinite(f)
        w, f = w[m], f[m]
        med = np.nanmedian(f)
        if not np.isfinite(med) or med == 0:
            raise ValueError(f"row {i} ({files[i]}): non-finite or zero median")
        f = f / med
        cont, cont_err, _seg, _seg_err = nn.normalize(w, f)
        cont_smo = get_smoothed_continuum(w, cont, cont_err)
        out_wave.append(w); out_cont.append(cont_smo); out_norm.append(f / cont_smo)
        print(f"  [{i+1}/{waves.shape[0]}] {files[i]}  n={w.size}", flush=True)

    maxlen = max(x.size for x in out_wave)
    pad = lambda a: np.array([np.pad(x, (0, maxlen - x.size),
                                     constant_values=np.nan) for x in a])
    np.savez_compressed(args.out, wavelengths=pad(out_wave),
                        normed_flux=pad(out_norm), continuum=pad(out_cont),
                        files=files)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.set_defaults(func=cmd_export)
    e.add_argument("--zarr", default="harps_data.zarr")
    e.add_argument("--npz", default="harps_raw.npz")
    n = sub.add_parser("normalize"); n.set_defaults(func=cmd_normalize)
    n.add_argument("--npz", default="harps_raw.npz")
    n.add_argument("--out", default="harps_normalized.npz")
    n.add_argument("--weights", default="active", choices=("active", "synth", "emission"))
    n.add_argument("--resampling-step", type=float, default=0.05)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
