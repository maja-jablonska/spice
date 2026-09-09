"""Tabulate the full-range (4450-6750 A) TZ For joint-fit chain: one row per result pickle in tzfor_aemu_out/.

Prints a markdown table (Teff1, Teff2, [Fe/H], vmac, vsini scales, chi2/N for photometry and spectra, and the
block-jackknife errors where a fit has them) plus the treatment systematic (spread of Teff2 across the
full-epoch treatments). Runs on the pickles only: no synthesis, no GPU needed.
"""
import pickle, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; OUT = HERE / "tzfor_aemu_out"
ROWS = [  # (file, label)
    ("fullrange_pass1.pkl", "all epochs, no mask (pass 1)"),
    ("fullrange_pass1_hp.pkl", "  same, full float32 matmul precision + jackknife"),
    ("fullrange_pass1_lowvsini.pkl", "  same, vsini_1 bound 0.1 (no better point found)"),
    ("fullrange_sync.pkl", "  same, synchronous rotation (vsini scale = 1)"),
    ("fullrange_odd_plain.pkl", "odd epochs, no mask"),
    ("fullrange_even_plain.pkl", "even epochs, no mask"),
    ("fullrange_even_maskodd.pkl", "even epochs, mask from odd (circularity test)"),
    ("fullrange_maskall.pkl", "all epochs, 5% mask (all epochs)"),
    ("fullrange_maskall_v2.pkl", "  same, stall-safe optimiser + jackknife"),
    ("fullrange_maskall_hp.pkl", "  same, full float32 matmul precision + jackknife"),
    ("fullrange_maskall_sync.pkl", "  same, synchronous rotation"),
    ("fullrange_delta_odd.pkl", "delta map derived on odd epochs (theta held)"),
    ("fullrange_even_deltaodd.pkl", "even epochs, delta from odd held fixed"),
]

def load(f):
    p = OUT / f
    return pickle.load(open(p, "rb")) if p.exists() else None

def cell(d, nm, fmt, err_fmt=None):
    if nm not in d["pnames"]: return "-"
    i = d["pnames"].index(nm); v = float(np.asarray(d["theta"])[i]); s = fmt % v
    if err_fmt and d.get("jackknife_err") is not None:
        e = float(np.asarray(d["jackknife_err"])[i]); s += " ± " + err_fmt % e
    return s

def main():
    hdr = ["fit", "Teff1 [K]", "Teff2 [K]", "[Fe/H]", "vmac1", "vmac2", "vsini scale 1", "vsini scale 2", "chi2_ph/N", "chi2_sp/N", "N_sp"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]; teff2 = {}
    for f, label in ROWS:
        d = load(f)
        if d is None: lines.append(f"| {label} | (pending: {f}) |" + " |" * (len(hdr) - 2)); continue
        row = [label, cell(d, "Teff1", "%.0f", "%.0f"), cell(d, "Teff2", "%.0f", "%.0f"), cell(d, "feh", "%.3f", "%.3f"), cell(d, "vmac1", "%.2f"), cell(d, "vmac2", "%.2f"),
               cell(d, "vsini_scale1", "%.2f"), cell(d, "vsini_scale2", "%.2f"), "%.3f" % (d["chi2_phot"] / d["N_ph"]), "%.2f" % (d["chi2_spec"] / d["N_sp"]), f"{d['N_sp']:,}"]
        lines.append("| " + " | ".join(row) + " |")
        if len(d["fit_epochs"]) == 21 and d["args"].get("maxiter", 1) > 0: teff2[label] = (float(d["theta"][0]), float(d["theta"][1]), float(d["theta"][2]))
    print("\n".join(lines))
    if len(teff2) > 1:
        A = np.array(list(teff2.values()))
        print(f"\nTreatment spread over {len(teff2)} full-epoch fits: Teff1 {A[:, 0].min():.0f}-{A[:, 0].max():.0f} K (std {A[:, 0].std():.0f}), "
              f"Teff2 {A[:, 1].min():.0f}-{A[:, 1].max():.0f} K (std {A[:, 1].std():.0f}), [Fe/H] {A[:, 2].min():.3f}..{A[:, 2].max():.3f} (std {A[:, 2].std():.3f})")
    for f in ("fullrange_pass1.pkl", "fullrange_pass1_hp.pkl", "fullrange_maskall.pkl", "fullrange_maskall_v2.pkl", "fullrange_maskall_hp.pkl"):
        d = load(f)
        if d is None or d.get("jackknife") is None: continue
        J = np.asarray(d["jackknife"]); th = np.asarray(d["theta"]); same = [j for j in range(J.shape[0]) if np.allclose(J[j], th, rtol=0, atol=1e-6)]
        print(f"{f}: jackknife blocks {J.shape[0]}, blocks identical to the full fit (stalled refits): {same}")

if __name__ == "__main__":
    main()
