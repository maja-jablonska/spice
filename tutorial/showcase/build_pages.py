"""Build the two interactive showcase pages from the Gadi outputs (runs locally, no JAX).

    python build_pages.py --star OUT/s08_star_builder_grid/data.npz \
                          --phot OUT/s09_photometry_grid/data.npz --out DIR

Writes for each page a full HTML document for the docs website (``<name>.html``) and
a body-only fragment for claude.ai artifact previews (``<name>_artifact.html``).
All data is embedded (base64 typed arrays); the browser draws the star from the same
mesh, spot weights and realised rotation matrices SPICE used, and every spectrum shown
is a precomputed SPICE synthesis (nearest grid point, no interpolation).
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a).astype(dtype)).tobytes()).decode()


def star_payload(path):
    d = np.load(path)
    flux = d["flux"]
    q = np.clip(np.round(flux / 1.2 * 65535), 0, 65535)
    w = d["spot_weights"]
    return dict(
        wl=b64(d["wl"], "<f4"), flux=b64(q, "<u2"), dims=list(flux.shape),
        teffs=d["teffs"].tolist(), veqs=d["veqs"].tolist(), incls=d["incls"].tolist(),
        spotLats=[None if np.isnan(x) else float(x) for x in d["spot_lats"]], phases=d["phases"].tolist(),
        verts=b64(d["vertices"], "<f4"), faces=b64(d["faces"], "<i4"),
        spotW=b64(np.round(np.clip(w, 0, 1) * 255), "u1"), rots=b64(d["rotations"], "<f4"),
        bases=b64(d["bases"], "<f4"), spotDtFrac=float(d["spot_dt_frac"]), logg=float(d["logg"]))


def phot_payload(path):
    d = np.load(path)
    return dict(
        wl=b64(d["wl"], "<f4"), flux=b64(d["flux"], "<f4"), n=int(len(d["wl"])),
        stars=[dict(name=str(n), teff=float(p[0]), logg=float(p[1]), radius=float(p[2]))
               for n, p in zip(d["star_names"], d["star_params"])],
        filters=[dict(name=str(n), zp=float(z)) for n, z in zip(d["filter_names"], d["filter_zps"])],
        curves=b64(d["filter_curves"], "<f4"), mags=d["mags"].tolist())


def write(name, title, template, payload, out):
    html = (HERE / "pages" / template).read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    frag = html
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + html + "\n</body>\n</html>\n")
    (out / f"{name}.html").write_text(full)
    (out / f"{name}_artifact.html").write_text(frag)
    print(f"wrote {out / name}.html ({len(full) / 1e6:.2f} MB) and _artifact.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--star"); ap.add_argument("--phot"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.star:
        write("star_builder", "Star Builder", "star_builder.html", star_payload(a.star), out)
    if a.phot:
        write("passband_lab", "Passband Lab", "passband_lab.html", phot_payload(a.phot), out)
