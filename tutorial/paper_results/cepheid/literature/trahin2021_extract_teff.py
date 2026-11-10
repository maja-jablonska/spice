"""Recover the SPIPS Teff(phase) model curve of delta Cep from the vector Fig. B.6 of Trahin et al. 2021 (page 19).

Walks the SVG tree composing nested group / path transforms into page coordinates. The left column of the lower
figure (delta Cep) holds panels a (velocity), c (Teff) and b (angular diameter), sharing the phase axis. Calibration
uses the tick marks: phase ticks under panel b (labels 0.0 ... 1.0, left to right) and the Teff ticks left of panel c
(labels 5.5, 6.0, 6.5, 7.0 kK, bottom to top). The model is the dark-green stroke inside panel c.

    python extract_teff.py p19.svg OUT_PREFIX
"""
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

NS = "{http://www.w3.org/2000/svg}"
MODEL = "rgb(9.999084%, 50%, 25%)"
BLACK = "rgb(0%, 0%, 0%)"
num = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"


def mat(s):
    if not s:
        return np.eye(3)
    m = re.match(r"matrix\(([^)]+)\)", s)
    if m:
        a, b, c, d, e, f = [float(x) for x in re.split(r"[ ,]+", m.group(1).strip())]
        return np.array([[a, c, e], [b, d, f], [0, 0, 1.0]])
    m = re.match(r"translate\(([^)]+)\)", s)
    if m:
        v = [float(x) for x in re.split(r"[ ,]+", m.group(1).strip())] + [0.0]
        return np.array([[1, 0, v[0]], [0, 1, v[1]], [0, 0, 1.0]])
    raise ValueError(s)


def walk(el, M, out):
    M = M @ mat(el.get("transform"))
    if el.tag == NS + "path" and el.get("stroke") and el.get("d"):
        pts = []
        for cmd, args in re.findall(r"([MLC])([^MLCZ]*)", el.get("d")):
            xy = np.array([float(x) for x in re.findall(num, args)]).reshape(-1, 2)
            pts.extend(xy[-1:] if cmd == "C" else xy)
        if pts:
            P = np.c_[np.array(pts), np.ones(len(pts))] @ M.T
            out.append((el.get("stroke"), P[:, 0], P[:, 1]))
    if el.tag == NS + "use":
        ref = el.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
        if ref in IDS and IDS[ref].tag != NS + "symbol" and not ref.startswith("glyph"):
            walk(IDS[ref], M, out)
    for ch in el:
        if ch.tag != NS + "defs":
            walk(ch, M, out)


root = ET.parse(sys.argv[1]).getroot()
IDS = {e.get("id"): e for e in root.iter() if e.get("id")}
allp = []
walk(root, np.eye(3), allp)
lower = lambda X, Y: X.min() < 290 and X.max() < 330 and Y.min() > 395         # delta Cep figure, left column (page points)
model = [(X, Y) for s, X, Y in allp if s == MODEL and lower(X, Y)]
print("model strokes:", [(len(X), round(X.min(), 1), round(X.max(), 1), round(Y.min(), 1), round(Y.max(), 1)) for X, Y in model])
ticks = [(X, Y) for s, X, Y in allp if s == BLACK and len(X) == 2 and lower(X, Y)]
vt = sorted({(round(X[0], 3), round(float(Y.max()), 2)) for X, Y in ticks if abs(X[0] - X[1]) < 1e-3 and 1.0 < abs(Y[0] - Y[1]) < 4.0})
ht = sorted({(round(float(X.min()), 2), round(Y[0], 3)) for X, Y in ticks if abs(Y[0] - Y[1]) < 1e-3 and 1.0 < abs(X[0] - X[1]) < 4.0})
print("vertical ticks (x, ymax):", vt)
print("horizontal ticks (x0, y):", ht)
np.savez(sys.argv[2] + "_raw.npz", model=np.array([np.concatenate([X for X, _ in model]), np.concatenate([Y for _, Y in model])]),
         vt=np.array(vt), ht=np.array(ht))
