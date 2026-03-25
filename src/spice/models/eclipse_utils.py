import numpy as np

# ---------- Utilities ----------
def _hermite_coeffs(p0, p1, v0, v1, h):
    # Returns coefficients for cubic Hermite: p(s)=a*s^3+b*s^2+c*s+d, s in [0,1]
    d = p0
    c = h*v0
    b = 3*(p1 - p0) - h*(2*v0 + v1)
    a = 2*(p0 - p1) + h*(v0 + v1)
    return a, b, c, d

def _hermite_eval(a, b, c, d, s):
    # position and derivative wrt t
    p = ((a*s + b)*s + c)*s + d                   # position
    dp_ds = (3*a*s + 2*b)*s + c                   # derivative wrt s
    return p, dp_ds

def _bracket_sign_changes(fvals):
    # Return list of indices i where f[i]*f[i+1] <= 0 and not both zero
    idx = []
    for i in range(len(fvals)-1):
        a, b = fvals[i], fvals[i+1]
        if a == 0:  # treat exact sample zeros as tiny sign flip
            a = np.copysign(1e-30, b if b!=0 else 1.0)
        if b == 0:
            b = np.copysign(1e-30, a if a!=0 else 1.0)
        if a*b < 0:
            idx.append(i)
    return idx

def _brentq(f, a, b, fa=None, fb=None, tol=1e-9, maxiter=100):
    # Robust 1D root finder without SciPy (a,b must bracket)
    if fa is None: fa = f(a)
    if fb is None: fb = f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0:
        return None
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb == 0: return b
        if fa == 0: return a
        if np.abs(fc) < np.abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2*np.finfo(float).eps*np.abs(b) + 0.5*tol
        xm = 0.5*(c - b)
        if np.abs(xm) <= tol1:
            return b
        if np.abs(e) >= tol1 and np.abs(fa) > np.abs(fb):
            s = fb/fa
            if a == c:
                p = 2*xm*s
                q = 1 - s
            else:
                q = fa/fc; r = fb/fc
                p = s*(2*xm*q*(q - r) - (b - a)*(r - 1))
                q = (q - 1)*(r - 1)*(s - 1)
            if p > 0: q = -q
            p = np.abs(p)
            cond1 = (2*p < 3*xm*q - np.abs(tol1*q))
            cond2 = (p < np.abs(0.5*e*q))
            if cond1 and cond2:
                e = d; d = p/q
            else:
                d = xm; e = d
        else:
            d = xm; e = d
        a, fa = b, fb
        if np.abs(d) > tol1:
            b += d
        else:
            b += np.copysign(tol1, xm)
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b

# ---------- Main builder ----------
class RelativeSkyMotion:
    def __init__(self, t, pos1, vel1, pos2, vel2):
        t = np.asarray(t)
        assert np.all(np.diff(t) > 0), "t must be strictly increasing"
        self.t = t
        self.N = len(t)
        self.h = np.diff(t)

        r1 = np.asarray(pos1); v1 = np.asarray(vel1)
        r2 = np.asarray(pos2); v2 = np.asarray(vel2)
        self.r = r2 - r1
        self.v = v2 - v1

        # Precompute Hermite coeffs per segment, for x and y independently
        self.coeffs = []
        for i in range(self.N-1):
            h = self.h[i]
            a_x, b_x, c_x, d_x = _hermite_coeffs(self.r[i,0], self.r[i+1,0], self.v[i,0], self.v[i+1,0], h)
            a_y, b_y, c_y, d_y = _hermite_coeffs(self.r[i,1], self.r[i+1,1], self.v[i,1], self.v[i+1,1], h)
            self.coeffs.append(((a_x,b_x,c_x,d_x),(a_y,b_y,c_y,d_y)))

    def rv(self, tq):
        # piecewise evaluation of r(t) and v(t)
        tq = np.atleast_1d(tq)
        r = np.zeros((len(tq),2))
        v = np.zeros((len(tq),2))
        for k, tval in enumerate(tq):
            if tval <= self.t[0]:
                i = 0; s = 0.0
            elif tval >= self.t[-1]:
                i = self.N-2; s = 1.0
            else:
                i = np.searchsorted(self.t, tval) - 1
                h = self.h[i]; s = (tval - self.t[i])/h
            (ax,bx,cx,dx),(ay,by,cy,dy) = self.coeffs[i]
            px, dpx_ds = _hermite_eval(ax,bx,cx,dx,s)
            py, dpy_ds = _hermite_eval(ay,by,cy,dy,s)
            r[k,0], r[k,1] = px, py
            v[k,0], v[k,1] = dpx_ds/self.h[i], dpy_ds/self.h[i]  # chain rule: ds/dt = 1/h
        return r, v

    def rho(self, tq):
        r, _ = self.rv(tq)
        return np.linalg.norm(r, axis=1)

    def drho_dt(self, tq):
        r, v = self.rv(tq)
        rho = np.linalg.norm(r, axis=1)
        # handle rho ~ 0 safely
        dot = np.einsum('ij,ij->i', r, v)
        out = np.zeros_like(rho)
        mask = rho > 0
        out[mask] = dot[mask]/rho[mask]
        return out

# ---------- Eclipse finder ----------
def find_eclipses(t, pos1, vel1, pos2, vel2, R1, R2,
                  tol=1e-6, sample_factor=5, want_total=True):
    """
    Returns list of eclipses; each is a dict with keys:
    {'seg': i, 'T1':..., 'T2':..., 'mid':..., 'T3':..., 'T4':..., 'kind': 'partial'|'total'|'grazing'}
    """
    motion = RelativeSkyMotion(t, pos1, vel1, pos2, vel2)
    Rsum = float(R1 + R2)
    Rdiff = float(abs(R1 - R2))

    # Coarse scan on a refined grid to bracket events robustly
    ti = []
    for i in range(len(t)-1):
        n = max(3, sample_factor)  # per segment
        ti.append(np.linspace(t[i], t[i+1], n, endpoint=(i==len(t)-2)))
    Tgrid = np.unique(np.concatenate(ti))
    rho = motion.rho(Tgrid)

    # Bracket potential T1/T4 (outer limb)
    gout = rho - Rsum
    br_out = _bracket_sign_changes(gout)

    eclipses = []
    used = set()

    def root_on(func, a, b):
        return _brentq(func, a, b, tol=tol)

    for idx in br_out:
        a, b = Tgrid[idx], Tgrid[idx+1]
        # Find T1
        T1 = root_on(lambda x: motion.rho([x])[0] - Rsum, a, b)
        if T1 is None: 
            continue

        # Now march forward to find T2 (inner limb) or mid if no totality
        # First bracket the interior of eclipse until we cross outer limb again
        # Expand a small window to ensure we capture T2..T4
        # Use local dense grid
        local = np.linspace(T1, min(T1 + (b-a)*50, t[-1]), 64)
        rho_local = motion.rho(local)

        # Determine if totality is possible (min rho < Rdiff)
        min_idx = np.argmin(rho_local)
        rho_min = rho_local[min_idx]
        t_min_guess = local[min_idx]

        # T4: next crossing of gout back to zero after T1
        # Find bracket for T4 by scanning forward
        T4 = None
        for j in range(idx+1, len(Tgrid)-1):
            if gout[j]*gout[j+1] < 0 and Tgrid[j] > T1:
                T4 = root_on(lambda x: motion.rho([x])[0] - Rsum, Tgrid[j], Tgrid[j+1])
                break

        # If we didn't find T4 in coarse grid, try to extend search within the last segment
        if T4 is None:
            # Try a bounded search toward the end
            for i2 in range(len(t)-1):
                a2, b2 = t[i2], t[i2+1]
                # skip intervals earlier than T1
                if b2 <= T1: 
                    continue
                # check sign change in this raw segment's ends
                g_a2 = motion.rho([a2])[0] - Rsum
                g_b2 = motion.rho([b2])[0] - Rsum
                if g_a2*g_b2 < 0:
                    T4 = root_on(lambda x: motion.rho([x])[0] - Rsum, a2, b2)
                    break

        # If no T4, treat as incomplete bracket (skip)
        if T4 is None:
            continue

        # Mid-eclipse: where r·v = 0 between T1 and T4
        # Option B (einsum; robust if shapes are (1,2))
        mid = root_on(lambda x: float(np.einsum('ij,ij->', *motion.rv([x]))), T1, T4)

        if mid is None:
            mid = t_min_guess

        # T2/T3 if (potentially) total
        T2 = T3 = None
        kind = 'partial'
        if want_total and rho_min < Rdiff + 1e-12:
            # Find inner contacts where rho = |R1-R2|
            # bracket around mid for inner crossings
            # Coarse brackets inside [T1,T4]
            local2 = np.linspace(T1, T4, 64)
            gin = motion.rho(local2) - Rdiff
            br_in = _bracket_sign_changes(gin)
            # Expect two roots
            roots = []
            for k in br_in:
                rtk = root_on(lambda x: motion.rho([x])[0] - Rdiff, local2[k], local2[k+1])
                if rtk is not None:
                    roots.append(rtk)
            if len(roots) >= 2:
                roots.sort()
                T2, T3 = roots[0], roots[-1]
                kind = 'total' if R1 >= R2 else 'annular'  # purely geometric naming
            else:
                # Numerically grazing totality; keep partial but note grazing
                kind = 'grazing'

        # Deduplicate if an earlier pass already recorded this eclipse (overlapping scan windows)
        key = (round(T1, 9), round(T4, 9))
        if key in used:
            continue
        used.add(key)

        eclipses.append(dict(seg=idx,
                             T1=T1, T2=T2, mid=mid, T3=T3, T4=T4,
                             kind=kind))

    return eclipses
