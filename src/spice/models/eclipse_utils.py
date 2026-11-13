import warnings

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
                  tol=1e-6, sample_factor=5, want_total=True,
                  depth1=None, depth2=None):
    """Find eclipses in a sampled two-body orbit.

    Args:
        t: Sample times, strictly increasing.
        pos1, vel1, pos2, vel2: Sky-plane positions ``(n, 2)`` and velocities of the
            two components, in units consistent with ``t`` and the radii.
        R1, R2: Component radii.
        tol, sample_factor, want_total: Root tolerance, coarse samples per segment,
            and whether to resolve the inner contacts T2/T3.
        depth1, depth2 (optional): Each component's distance from the observer
            along the line of sight at the sample times (larger = farther). When
            given, each eclipse records which component is in front and is named
            after it; without them ``front`` is ``None`` and a full overlap is named
            from the radii alone, as if ``body1`` were in front.

    Returns:
        List of dicts with keys ``seg``, ``T1``, ``T2``, ``mid``, ``T3``, ``T4``,
        ``front`` and ``kind``. ``front`` is 1 or 2, the component nearer the
        observer at mid-eclipse. ``kind`` is ``'partial'``, ``'grazing'`` (the inner
        contacts cannot be resolved), ``'total'`` (the front component is at least as
        large as the back one and hides it completely between T2 and T3) or
        ``'annular'`` (a smaller front component lies entirely inside the back
        one's disc between T2 and T3, e.g. a transit).
    """
    t0 = _log_eclipse_search_start()
    eclipses = _find_eclipses_raw(t, pos1, vel1, pos2, vel2, R1, R2,
                                  tol=tol, sample_factor=sample_factor, want_total=want_total,
                                  depth1=depth1, depth2=depth2)
    _log_eclipse_search_done(eclipses, t0)
    return eclipses


def _log_eclipse_search_start():
    import time as _time
    from spice.utils import log as _spice_log
    if _spice_log.is_verbose():
        _spice_log.info("Finding eclipses...")
    return _time.perf_counter()


def _log_eclipse_search_done(eclipses, t0):
    import time as _time
    from spice.utils import log as _spice_log
    done = f"Found {len(eclipses)} eclipse(s) in {_time.perf_counter() - t0:.1f} s"
    if _spice_log.is_verbose():
        _spice_log.substitute(done)
    else:
        _spice_log.info(done)


def _find_eclipses_raw(t, pos1, vel1, pos2, vel2, R1, R2,
                       tol=1e-6, sample_factor=5, want_total=True,
                       depth1=None, depth2=None):
    """:func:`find_eclipses` without the progress log (callers log after filtering)."""
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

        # Minimum separation over the whole eclipse (decides whether totality is
        # possible). A fixed-length window after T1 stopped short of mid-eclipse
        # whenever the eclipse spanned more than ~10 orbit samples.
        local = np.linspace(T1, T4, 129)
        rho_local = motion.rho(local)
        min_idx = np.argmin(rho_local)
        rho_min = rho_local[min_idx]
        t_min_guess = local[min_idx]

        # Mid-eclipse: where r·v = 0 between T1 and T4
        # Option B (einsum; robust if shapes are (1,2))
        mid = root_on(lambda x: float(np.einsum('ij,ij->', *motion.rv([x]))), T1, T4)

        if mid is None:
            mid = t_min_guess
        # Include the separation at mid-eclipse itself, so an instantaneous full
        # overlap (equal radii, zero impact parameter) is seen regardless of the grid.
        rho_min = min(rho_min, float(motion.rho([mid])[0]))

        # Which component is nearer the observer at mid-eclipse
        front = None
        if depth1 is not None and depth2 is not None:
            d1 = np.interp(mid, t, np.asarray(depth1, dtype=float))
            d2 = np.interp(mid, t, np.asarray(depth2, dtype=float))
            front = 1 if d1 < d2 else 2
        R_front, R_back = (R2, R1) if front == 2 else (R1, R2)

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
                kind = 'total' if R_front >= R_back else 'annular'
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
                             front=front, kind=kind))

    return eclipses


def _periodic_pad(t, arrays, period, pad):
    """Extend a sampled periodic orbit by ``pad`` on both sides.

    Samples shifted by -``period`` that fall before ``t[0]`` and samples shifted by
    +``period`` that fall after ``t[-1]`` (within ``pad``) are added, so the result
    stays strictly increasing. ``t`` must span at least one period.
    """
    t = np.asarray(t, dtype=float)
    t_lo, t_hi = t[0], t[-1]
    before = (t - period < t_lo) & (t - period >= t_lo - pad)
    after = (t + period > t_hi) & (t + period <= t_hi + pad)
    t_new = np.concatenate([t[before] - period, t, t[after] + period])
    arrays_new = tuple(np.concatenate([a[before], a, a[after]]) for a in arrays)
    return t_new, arrays_new


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
            defaults to one full period: every eclipse whose mid time lies in ``[0, P)``,
            including one that straddles ``t = 0``.
        los_vector (array_like, optional): Line-of-sight vector defining the sky
            plane. Keplerian ``Binary`` only; defaults to ``binary.body1.los_vector``
            — the same vector used by ``add_orbit`` and the occlusion resolution,
            so the finder agrees with what the synthesis will actually eclipse.
        tol, sample_factor, want_total: Passed through to :func:`find_eclipses`.

    Returns:
        List of eclipse dicts (keys ``T1``, ``T2``, ``mid``, ``T3``, ``T4``,
        ``front``, ``kind``) as documented in :func:`find_eclipses`. ``front`` (1 or
        2) is the component nearer the observer at mid-eclipse, and ``kind`` is named
        after it: ``'total'`` when the front star hides the other completely,
        ``'annular'`` when a smaller front star crosses the other's disc. Times are in years for
        a ``Binary`` and in **days** for a ``PhoebeBinary`` (matching the PHOEBE
        clock of ``evaluated_times``).

    Note:
        For a ``PhoebeBinary`` the finder scans PHOEBE's own precomputed uvw
        orbit samples (``evaluated_times`` / ``body*_centers`` / ``body*_velocities``)
        rather than re-solving the Kepler orbit from elements — u/v span the sky
        plane and w is the line of sight, so the events found are exactly the ones
        the PHOEBE meshes produce. If the precomputed window spans at least one
        orbital period, it is padded periodically (samples shifted by +-P), so an
        eclipse straddling either end of the window is still found; every eclipse
        whose mid time lies in ``[t[0], t[-1])`` is returned, and its contact times
        may fall just outside the window. A window shorter than one period cannot
        be padded: a ``RuntimeWarning`` is emitted and eclipses cut by its ends are
        missed. Periodic padding assumes a strictly periodic orbit (no apsidal
        motion or period change across the window).
    """
    import jax.numpy as jnp
    from spice.constants import SOLAR_RAD_M
    from spice.models.binary import PhoebeBinary, DAY_TO_YEAR
    from spice.models.orbit_utils import get_orbit_jax, Constants

    t0 = _log_eclipse_search_start()
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
        # PHOEBE's w axis points toward the observer: distance from the observer is -w.
        depth1 = -np.asarray(binary.body1_centers)[:, 2]
        depth2 = -np.asarray(binary.body2_centers)[:, 2]
        P_days = float(binary.P) / DAY_TO_YEAR
        t_lo, t_hi = float(t[0]), float(t[-1])
        if t_hi - t_lo >= P_days * (1.0 - 1e-9):
            # The window covers a full period: pad it on both sides with the
            # samples shifted by one period, so an eclipse straddling either edge
            # is bracketed by its own ingress and egress.
            t, (pos1, vel1, pos2, vel2, depth1, depth2) = _periodic_pad(
                t, (pos1, vel1, pos2, vel2, depth1, depth2), P_days, 0.25 * P_days)
            eclipses = _find_eclipses_raw(t, pos1, vel1, pos2, vel2, R1, R2,
                                     tol=tol, sample_factor=sample_factor, want_total=want_total,
                                     depth1=depth1, depth2=depth2)
            eps = 1e-6 * P_days
            # Half-open window, like the [0, P) window of a Keplerian Binary: an
            # eclipse at the start of a one-period window is not reported again
            # at its end.
            eclipses = [e for e in eclipses if t_lo - eps <= e["mid"] < t_hi - eps]
            _log_eclipse_search_done(eclipses, t0)
            return eclipses
        warnings.warn(
            f"The PHOEBE orbit covers {t_hi - t_lo:.4g} d, less than one period "
            f"({P_days:.4g} d), so it cannot be padded periodically: eclipses whose "
            "ingress or egress falls outside the computed times are not found. "
            "Compute the orb dataset over at least one full period.",
            RuntimeWarning, stacklevel=2)
        eclipses = _find_eclipses_raw(t, pos1, vel1, pos2, vel2, R1, R2,
                                      tol=tol, sample_factor=sample_factor, want_total=want_total,
                                      depth1=depth1, depth2=depth2)
        _log_eclipse_search_done(eclipses, t0)
        return eclipses

    default_window = times is None
    if default_window:
        # Pad the period by a quarter on each side so an eclipse straddling
        # t = 0 (or t = P) is bracketed by its ingress; only eclipses whose mid
        # time falls in [0, P) are returned below, so each is reported once.
        P = float(binary.P)
        times = np.linspace(-0.25 * P, 1.25 * P, int(1.5 * n_points))
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
    # Distance from the observer along the line of sight (los points observer -> star).
    los_hat = np.asarray(los_vector, dtype=float)
    los_hat = los_hat / np.linalg.norm(los_hat)
    depth1 = np.asarray(orbit[2]) / SOLAR_RAD_M @ los_hat
    depth2 = np.asarray(orbit[4]) / SOLAR_RAD_M @ los_hat

    eclipses = _find_eclipses_raw(times, pos1, vel1, pos2, vel2, R1, R2,
                                  tol=tol, sample_factor=sample_factor, want_total=want_total,
                                  depth1=depth1, depth2=depth2)
    if default_window:
        # The same eclipse can be found at both ends (mid ~ 0 and ~ P); keep the
        # copy inside [0, P) with a round-off tolerance.
        eps = 1e-6 * P
        eclipses = [ecl for ecl in eclipses if -eps <= ecl["mid"] < P - eps]
    _log_eclipse_search_done(eclipses, t0)
    return eclipses
