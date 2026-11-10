import numpy as np

# ---------- Utilities ----------
def _sky_plane_basis(los_vector):
    # Orthonormal basis (e1, e2) spanning the plane perpendicular to the LOS,
    # returned as a (3, 2) projection matrix.
    los = np.asarray(los_vector, dtype=float)
    los = los / np.linalg.norm(los)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, los)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, los) * los
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(los, e1)
    return np.stack([e1, e2], axis=1)
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
    import time as _time
    from spice.utils import log as _spice_log
    if _spice_log.is_verbose():
        _spice_log.info("Finding eclipses...")
    _t0 = _time.perf_counter()
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
        # Only ingress crossings (rho decreasing through Rsum) open an eclipse;
        # egress crossings would otherwise pair with the next ingress and record
        # a bogus "eclipse" spanning the out-of-eclipse gap.
        if gout[idx] < gout[idx+1]:
            continue
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

        # If no T4 (or a degenerate window-edge bracket that resolved back to
        # T1 within the root tolerance), treat as incomplete and skip
        if T4 is None or T4 - T1 <= tol:
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

    _done = f"Found {len(eclipses)} eclipse(s) in {_time.perf_counter() - _t0:.1f} s"
    if _spice_log.is_verbose():
        _spice_log.substitute(_done)
    else:
        _spice_log.info(_done)
    return eclipses


def find_binary_eclipses(binary, n_points=100, times=None, los_vector=None,
                         tol=1e-6, sample_factor=5, want_total=True):
    """Locate eclipses directly from a :class:`Binary` or :class:`PhoebeBinary`.

    Convenience wrapper around :func:`find_eclipses`: it samples a low-resolution
    Keplerian orbit itself from the binary's stored orbital elements and component
    masses, projects the component positions and velocities onto the sky plane
    (perpendicular to the line of sight), and searches that sampling for eclipses
    using the component radii. The Hermite interpolation and root refinement in
    :func:`find_eclipses` make a coarse sampling sufficient — the default 100
    points per period locate contact times to high precision.

    Args:
        binary: A ``Binary`` (after :func:`~spice.models.binary.add_orbit`) or
            ``PhoebeBinary``. Orbital elements, masses, and radii are read from it.
        n_points (int, optional): Number of coarse orbit samples over one period.
            Keplerian ``Binary`` only; ignored if ``times`` is given. Defaults to 100.
        times (array_like, optional): Explicit scan window in **years** (the
            ``add_orbit`` clock), strictly increasing. Keplerian ``Binary`` only;
            defaults to one full period ``[0, P]``.
        los_vector (array_like, optional): Line-of-sight vector defining the sky
            plane. Keplerian ``Binary`` only; defaults to ``binary.body1.los_vector``
            — the same vector used by ``add_orbit`` and the occlusion resolution,
            so the finder agrees with what the synthesis will actually eclipse.
        tol, sample_factor, want_total: Passed through to :func:`find_eclipses`.

    Returns:
        List of eclipse dicts (keys ``T1``, ``T2``, ``mid``, ``T3``, ``T4``,
        ``kind``) as documented in :func:`find_eclipses`. Times are in years for
        a ``Binary`` and in **days** for a ``PhoebeBinary`` (matching the PHOEBE
        clock of ``evaluated_times``).

    Note:
        For a ``PhoebeBinary`` the finder scans PHOEBE's own precomputed uvw
        orbit samples (``evaluated_times`` / ``body*_centers`` / ``body*_velocities``)
        rather than re-solving the Kepler orbit from elements — u/v span the sky
        plane and w is the line of sight, so the events found are exactly the ones
        the PHOEBE meshes produce, and eclipses are only found within the
        precomputed time window.
    """
    import jax.numpy as jnp
    from spice.constants import SOLAR_RAD_M
    from spice.models.binary import PhoebeBinary
    from spice.models.orbit_utils import get_orbit_jax, Constants

    R1 = float(binary.body1.radius)
    R2 = float(binary.body2.radius)

    if isinstance(binary, PhoebeBinary):
        # PHOEBE uvw frame: u,v are the sky plane, w the line of sight. Positions
        # are in solar radii, velocities in km/s, times in days — convert the
        # velocities to solRad/day so the Hermite interpolation is consistent.
        kms_to_solrad_per_day = 86400.0e3 / SOLAR_RAD_M
        t = np.asarray(binary.evaluated_times, dtype=float)
        pos1 = np.asarray(binary.body1_centers)[:, :2]
        pos2 = np.asarray(binary.body2_centers)[:, :2]
        vel1 = np.asarray(binary.body1_velocities)[:, :2] * kms_to_solrad_per_day
        vel2 = np.asarray(binary.body2_velocities)[:, :2] * kms_to_solrad_per_day
        return find_eclipses(t, pos1, vel1, pos2, vel2, R1, R2,
                             tol=tol, sample_factor=sample_factor, want_total=want_total)

    if times is None:
        times = np.linspace(0.0, float(binary.P), int(n_points))
    else:
        times = np.asarray(times, dtype=float)

    if los_vector is None:
        los_vector = binary.body1.los_vector

    # vgamma only shifts the barycenter; the relative sky motion is unaffected,
    # so it is left out of the coarse orbit.
    orbit = get_orbit_jax(jnp.asarray(times), binary.body1.mass, binary.body2.mass,
                          binary.P, binary.ecc, binary.T, binary.i, binary.omega,
                          binary.Omega, binary.mean_anomaly, binary.reference_time,
                          vgamma=0.0, los_vector=jnp.asarray(los_vector))

    # get_orbit_jax returns positions in metres and velocities in km/s;
    # convert to solar radii and solar radii per year to match the radii and
    # the year-valued time grid.
    kms_to_solrad_per_year = 1.0e3 * Constants.yr / SOLAR_RAD_M
    basis = _sky_plane_basis(los_vector)
    pos1 = np.asarray(orbit[2]) / SOLAR_RAD_M @ basis
    vel1 = np.asarray(orbit[3]) * kms_to_solrad_per_year @ basis
    pos2 = np.asarray(orbit[4]) / SOLAR_RAD_M @ basis
    vel2 = np.asarray(orbit[5]) * kms_to_solrad_per_year @ basis

    return find_eclipses(times, pos1, vel1, pos2, vel2, R1, R2,
                         tol=tol, sample_factor=sample_factor, want_total=want_total)
