"""Absolute level and shape of the delta Cep V_T residuals of a delta Cep literature-model result (exact photometry).

    python dcep_vt_level.py RESULT.pkl [...]
"""
import pickle
import sys

import numpy as np

for f in sys.argv[1:]:
    r = pickle.load(open(f, "rb"))
    o = r["obs"]["VT"]; pm = np.asarray(r["phases_model"]); me = np.asarray(r["mags_exact"]["VT"])
    ph, mag, sig = (np.asarray(o[k], float) for k in ("phase", "mag", "err"))
    res = mag - np.interp(ph % 1.0, pm, me, period=1.0)
    wm = np.sum(res / sig ** 2) / np.sum(1 / sig ** 2)
    print(f"{f}: V_T n {res.size}, O-C mean {1e3 * res.mean():+.1f} mmag (weighted {1e3 * wm:+.1f}), rms {1e3 * res.std():.1f}, "
          f"median sigma {1e3 * np.median(sig):.1f}; chi2/N {np.mean((res / sig) ** 2):.2f}, about the weighted mean {np.mean(((res - wm) / sig) ** 2):.2f}")
