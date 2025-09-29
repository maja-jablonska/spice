import jax
import jax.numpy as jnp
from jax import jit, lax


SOLAR_MASS_KG = 1.988409870698051e+30
SOLAR_RAD_M = 6.957e8


class c:# Solar mas
    Msun = 1.989e30 # kg
    # Astronomical unit
    AU = 1.496e11 # m
    # Gravitational constant
    G = 6.674e-11 # m^3 kg^-1 s^-2
    # Year in seconds
    yr = 365.25*24*3600 # s
    # Day
    day = 24*3600 # s
    # Earth mass
    Mearth = 5.972e24 # kg
    # Parsec
    pc = 3.086e16 # m
    # Solar radius
    Rsun = 6.957e8 # m
    # light year
    ly = 9.461e15 # m
    # mas
    mas = 4.848e-9 # rad

# Function to calculate the orbital period of a binary system

@jit
def true_anomaly(M, ecc, itermax=8):
    """
    Calculate true and eccentric anomaly in Kepler orbits using JAX.
    
    Parameters:
    -----------
    M : array_like
        Mean anomaly (phase of the star)
    ecc : array_like
        Eccentricity
    itermax : int
        Maximum number of iterations
        
    Returns:
    --------
    F : array_like
        Eccentric anomaly (E)
    true_an : array_like
        True anomaly (theta)
    
    Description:
    ------------
    Solves Kepler's equation: E - e*sin(E) = M
    where E is the eccentric anomaly and M is the mean anomaly.
    
    The relationship between eccentric anomaly (E) and true anomaly (theta) is:
    tan(theta/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    """
    # Initial approximation
    Fn = M + ecc * jnp.sin(M) + ecc**2 / 2.0 * jnp.sin(2 * M)
    
    # Define a single iteration of Newton-Raphson
    def iteration_step(F, M, ecc):
        Mn = F - ecc * jnp.sin(F)
        Fn = F + (M - Mn) / (1.0 - ecc * jnp.cos(F))
        return Fn
    
    # Iterative solving using JAX's fori_loop for better performance
    # This allows JAX to optimize the loop execution
    def body_fun(i, F):
        return iteration_step(F, M, ecc)
    
    F = jax.lax.fori_loop(0, itermax, body_fun, Fn)
    
    # Calculate true anomaly from eccentric anomaly
    # Handle the safe computation of true anomaly to avoid division by zero
    tan_term = jnp.tan(F / 2.0)
    sqrt_term = jnp.sqrt((1.0 + ecc) / (1.0 - ecc))
    true_an = 2.0 * jnp.arctan(sqrt_term * tan_term)
    
    return F, true_an

def rotate_matrix_OX(angle):
    return jnp.array([[1, 0, 0],
                     [0, jnp.cos(angle), -jnp.sin(angle)],
                     [0, jnp.sin(angle), jnp.cos(angle)]])

def rotate_matrix_OZ(angle):
    return jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                     [jnp.sin(angle), jnp.cos(angle), 0],
                     [0, 0, 1]])

def get_transform_matrix(elements):
    i, omega, Omega = elements
    # rotate the orbit
    M_Omega = rotate_matrix_OZ(Omega)
    M_omega = rotate_matrix_OZ(omega)
    M_i = rotate_matrix_OX(i)
    return M_Omega @ M_i @ M_omega

def transform_orbit(x, v, elements):
    M = get_transform_matrix(elements)
    M = M.at[2,:].multiply(-1.0) # TODO: check if this is correct, this might be a result of some incorrect rotation

    return M @ x, M @ v


def get_orbit_jax(time, m1, m2, P, ecc, T, i, omega, Omega, mean_anomaly, reference_time, vgamma=0.):
    """
    Compute a Keplerian orbit for a binary system using JAX.
    Modified to match convention in dynamics() function.
    
    Parameters:
      time  : JAX array of times at which to evaluate the orbit.
      m1    : Mass of body 1 (in kg).
      m2    : Mass of body 2 (in kg).
      P     : Orbital period (in seconds, or consistent time units).
      ecc   : Orbital eccentricity.
      T     : Time of periastron passage (in same units as time).
      i     : Inclination (radians).
      omega : Argument of periastron (radians).
      Omega : Longitude of ascending node (radians).
      mean_anomaly : Mean anomaly at reference time (radians).
      reference_time : Reference time (in same units as time).
    
    Returns:
      An array containing positions and velocities with shape (6, len(time), 3).
      orbit[0]: barycenter positions
      orbit[1]: barycenter velocities (km/s)
      orbit[2]: displacement of body 1
      orbit[3]: velocity of body 1 (km/s)
      orbit[4]: displacement of body 2
      orbit[5]: velocity of body 2 (km/s)
    """
    # Gravitational constant in SI units [m^3/(kg s^2)]
    G = 6.67430e-11
    
    time = time*c.yr
    reference_time = reference_time*c.yr
    # Total mass.
    mass_ratio = m2/m1
    total_mass = m1 + m2
    m1_ = m1
    m2_ = m2

    m1 = m1*c.Msun
    m2 = m2*c.Msun
    P = P*c.yr
    T = T*c.yr

    # Compute semimajor axis of the relative orbit from Kepler's third law:
    #   P^2 = (4π²/G(M₁+M₂)) * a³   =>   a = [G (m1+m2) P²/(4π²)]^(1/3)
    a = (G * total_mass * c.Msun * P**2 / (4 * jnp.pi**2))**(1/3)

    # Mean motion.
    n = 2 * jnp.pi / P

    # Mean anomaly.
    M_anom = n * time + mean_anomaly 

    # Solve Kepler's equation for the eccentric anomaly E.
    E, theta = true_anomaly(M_anom, ecc)
    
    # Calculate the semi-major axes for each body
    a1 = a * mass_ratio / (1 + mass_ratio)  # Primary's semi-major axis
    a2 = a / (1 + mass_ratio)               # Secondary's semi-major axis
    
    # Calculate the radius for each component
    r1 = a1 * (1 - ecc * jnp.cos(E))  # Primary's radius from barycenter
    r2 = a2 * (1 - ecc * jnp.cos(E))  # Secondary's radius from barycenter
    
    # Compute parameters for velocity calculation
    l1 = r1 * (1 + ecc * jnp.cos(theta))
    l2 = r2 * (1 + ecc * jnp.cos(theta))
    L1 = 2 * jnp.pi * a1**2 / P * jnp.sqrt(1 - ecc**2)
    L2 = 2 * jnp.pi * a2**2 / P * jnp.sqrt(1 - ecc**2)
    
    # Compute radial and angular velocities
    rdot1 = L1 / l1 * ecc * jnp.sin(theta)
    rdot2 = L2 / l2 * ecc * jnp.sin(theta)
    thetadot1 = L1 / r1**2
    thetadot2 = L2 / r2**2
    
    # Common trigonometric calculations
    sin_longan = jnp.sin(Omega)
    cos_longan = jnp.cos(Omega)
    cos_incl = jnp.cos(i)
    sin_incl = jnp.sin(-i)  # Note the negative inclination
    
    # --- PRIMARY COMPONENT CALCULATIONS ---
    
    # Adjust for argument of periastron
    theta_primary = theta + omega
    
    # Precompute trigonometric functions
    sin_theta_primary = jnp.sin(theta_primary)
    cos_theta_primary = jnp.cos(theta_primary)
    
    # Convert to Cartesian coordinates
    x_primary = r1 * (cos_longan * cos_theta_primary - sin_longan * sin_theta_primary * cos_incl)
    y_primary = r1 * (sin_longan * cos_theta_primary + cos_longan * sin_theta_primary * cos_incl)
    z_primary = r1 * (sin_theta_primary * sin_incl)
    
    # Calculate velocity components
    vx_primary_ = cos_theta_primary * rdot1 - sin_theta_primary * r1 * thetadot1
    vy_primary_ = sin_theta_primary * rdot1 + cos_theta_primary * r1 * thetadot1
    
    vx_primary = cos_longan * vx_primary_ - sin_longan * vy_primary_ * cos_incl
    vy_primary = sin_longan * vx_primary_ + cos_longan * vy_primary_ * cos_incl
    vz_primary = sin_incl * vy_primary_
    
    # # Apply systemic velocity correction
    vz_primary = vz_primary - vgamma
    z_primary = z_primary - vgamma * (time - reference_time)
    
    # --- SECONDARY COMPONENT CALCULATIONS ---
    
    # Adjust for secondary component (half an orbit away) and argument of periastron
    theta_secondary = theta + jnp.pi + omega
    
    # Precompute trigonometric functions
    sin_theta_secondary = jnp.sin(theta_secondary)
    cos_theta_secondary = jnp.cos(theta_secondary)
    
    # Convert to Cartesian coordinates
    x_secondary = r2 * (cos_longan * cos_theta_secondary - sin_longan * sin_theta_secondary * cos_incl)
    y_secondary = r2 * (sin_longan * cos_theta_secondary + cos_longan * sin_theta_secondary * cos_incl)
    z_secondary = r2 * (sin_theta_secondary * sin_incl)
    
    # Calculate velocity components
    vx_secondary_ = cos_theta_secondary * rdot2 - sin_theta_secondary * r2 * thetadot2
    vy_secondary_ = sin_theta_secondary * rdot2 + cos_theta_secondary * r2 * thetadot2
    
    vx_secondary = cos_longan * vx_secondary_ - sin_longan * vy_secondary_ * cos_incl
    vy_secondary = sin_longan * vx_secondary_ + cos_longan * vy_secondary_ * cos_incl
    vz_secondary = sin_incl * vy_secondary_
    
    # # Apply systemic velocity correction
    vz_secondary = vz_secondary - vgamma
    z_secondary = z_secondary - vgamma * (time - reference_time)
    
    # --- BARYCENTER CALCULATIONS ---
    
    # Calculate barycenter positions
    bary_x = (m1_ * x_primary + m2_ * x_secondary) / total_mass
    bary_y = (m1_ * y_primary + m2_ * y_secondary) / total_mass
    bary_z = (m1_ * z_primary + m2_ * z_secondary) / total_mass
    
    # Calculate barycenter velocities
    bary_vx = (m1_ * vx_primary + m2_ * vx_secondary) / total_mass
    bary_vy = (m1_ * vy_primary + m2_ * vy_secondary) / total_mass
    bary_vz = (m1_ * vz_primary + m2_ * vz_secondary) / total_mass
    
    # Organize position and velocity tuples
    primary_pos = jnp.stack([x_primary, y_primary, z_primary], axis=1)
    primary_vel = jnp.stack([vx_primary, vy_primary, vz_primary], axis=1)
    
    secondary_pos = jnp.stack([x_secondary, y_secondary, z_secondary], axis=1)
    secondary_vel = jnp.stack([vx_secondary, vy_secondary, vz_secondary], axis=1)
    
    barycenter_pos = jnp.stack([bary_x, bary_y, bary_z], axis=1)
    barycenter_vel = jnp.stack([bary_vx, bary_vy, bary_vz], axis=1)
    
    orbit = jnp.stack([
        barycenter_pos,                      # orbit[2]: displacement of body 1
        barycenter_vel/1e3,               # orbit[1]: barycenter velocities (km/s)
        primary_pos,                       # orbit[2]: displacement of body 1
        primary_vel/1e3,                   # orbit[3]: velocity of body 1 (km/s)
        secondary_pos,                       # orbit[4]: displacement of body 2
        secondary_vel/1e3                   # orbit[5]: velocity of body 2 (km/s)
    ], axis=0)
    
    return orbit

# ----------------------------------------------------------------------------
# JAX-friendly eclipse timestamps from Keplerian elements
# ----------------------------------------------------------------------------

# --------------------------
# Utilities: Kepler + rotations
# --------------------------
@jit
def solve_kepler_E(M, e, maxit=30, tol=1e-12):
    """Solve Kepler's equation M = E - e sin E (JAX, vectorized)."""
    # Wrap M near 0 for stability
    Mwrap = (M + jnp.pi) % (2*jnp.pi) - jnp.pi
    # Good initial guess (Danby-ish)
    E0 = Mwrap + e*jnp.sin(Mwrap)/(1.0 - jnp.sin(Mwrap + e) + jnp.sin(Mwrap))

    def body_fun(k, E):
        f  = E - e*jnp.sin(E) - Mwrap
        fp = 1.0 - e*jnp.cos(E)
        dE = -f/fp
        E  = E + dE
        return E

    E = lax.fori_loop(0, maxit, body_fun, E0)
    return E

@jit
def true_anomaly_from_E(E, e):
    """Return true anomaly nu from E and e."""
    s = jnp.sqrt((1+e)/(1-e))
    t = jnp.tan(E/2.0) * s
    nu = 2.0*jnp.arctan(t)
    # wrap to [0, 2pi)
    nu = (nu + 2*jnp.pi) % (2*jnp.pi)
    return nu

@jit
def r_of_nu(a, e, nu):
    return a*(1 - e*e) / (1 + e*jnp.cos(nu))

@jit
def rot_x(theta, v):
    c, s = jnp.cos(theta), jnp.sin(theta)
    x, y, z = v
    return jnp.array([x, c*y - s*z, s*y + c*z])

@jit
def rot_z(theta, v):
    c, s = jnp.cos(theta), jnp.sin(theta)
    x, y, z = v
    return jnp.array([c*x - s*y, s*x + c*y, z])

# --------------------------
# Relative orbit sampler (observer frame)
# --------------------------
@jit
def relative_pos_vel_at(t, params):
    """
    Relative position/velocity (secondary - primary) in observer frame at time t.
    params: (m1, m2, P, e, Tperi, inc, omega, Omega, G)
    All SI; angles in radians.
    Returns: r_rel (3,), v_rel (3,)
    """
    m1, m2, P, e, Tperi, inc, omega, Omega, G = params
    mtot = m1 + m2
    n = 2*jnp.pi / P

    # Mean anomaly and eccentric anomaly
    M = n*(t - Tperi)
    E = solve_kepler_E(M, e)
    nu = true_anomaly_from_E(E, e)

    # Semi-major axis of relative orbit
    a = (G*mtot*P*P/(4*jnp.pi*jnp.pi))**(1/3)

    # Radius and time derivatives (two-body planar)
    r = r_of_nu(a, e, nu)
    # Convenient factors
    h = jnp.sqrt(G*mtot*a*(1 - e*e))           # specific angular momentum
    rdot   = (G*mtot/h) * e*jnp.sin(nu)        # radial speed
    nudot  = h/(r*r)                            # angular rate

    # Orbital-plane position/velocity (x' = r cos nu, y' = r sin nu)
    xop = r*jnp.cos(nu)
    yop = r*jnp.sin(nu)
    vxop = rdot*jnp.cos(nu) - r*nudot*jnp.sin(nu)
    vyop = rdot*jnp.sin(nu) + r*nudot*jnp.cos(nu)

    # Rotate by argument of periastron, inclination, node to observer frame
    r1 = rot_z(omega, jnp.array([xop, yop, 0.0]))
    r2 = rot_x(inc,   r1)
    r3 = rot_z(Omega, r2)

    v1 = rot_z(omega, jnp.array([vxop, vyop, 0.0]))
    v2 = rot_x(inc,   v1)
    v3 = rot_z(Omega, v2)

    return r3, v3

# --------------------------
# Parabolic refinement (on rho^2)
# --------------------------
@jit
def parabola_vertex_t(t0, t1, t2, y0, y1, y2):
    # Fit y = a t^2 + b t + c from 3 points and return vertex time
    A = jnp.array([[t0*t0, t0, 1.0],
                   [t1*t1, t1, 1.0],
                   [t2*t2, t2, 1.0]])
    Y = jnp.array([y0, y1, y2])
    # Solve (no conditionals; JAX does LU under the hood)
    coeff = jnp.linalg.solve(A, Y)
    a, b, c = coeff[0], coeff[1], coeff[2]
    # If a <= 0, vertex becomes t1 (middle sample); this is smooth enough for scheduling
    t_v = jnp.where(a > 0, -b/(2*a), t1)
    return t_v

# --------------------------
# Eclipse timestamps (approx) from elements
# --------------------------
@jit
def eclipse_timestamps_kepler(m1, m2, P, e, Tperi, inc, omega, Omega,
                              R1, R2, G=6.67430e-11, nscan=4096, pad=1.02,
                              los_vector=None):
    """
    Estimates eclipse timestamps from Keplerian elements.

    Inputs are in SI units, except for:
     - P: period in years
     - Tperi: time of periastron passage in years
     - angles (inc, omega, Omega) are in radians
     - los_vector: line-of-sight vector (defaults to [0,0,1])

    Returns:
      t_mid_primary, t1_p, t2_p, t3_p, t4_p,
      t_mid_secondary, t1_s, t2_s, t3_s, t4_s
    All returned times are in years.
    Any missing event is NaN.
    """
    if los_vector is None:
        los_vector = jnp.array([0.0, 0.0, 1.0])
    los_vector = los_vector / jnp.linalg.norm(los_vector)

    m1 = m1 * SOLAR_MASS_KG
    m2 = m2 * SOLAR_MASS_KG
    R1 = R1 * SOLAR_RAD_M
    R2 = R2 * SOLAR_RAD_M
    # Convert time inputs from years to seconds for SI calculations
    P_sec = P * c.yr
    Tperi_sec = Tperi * c.yr

    # Sample one period uniformly
    ts = Tperi_sec + jnp.linspace(0.0, P_sec, nscan, endpoint=False)
    params = (m1, m2, P_sec, e, Tperi_sec, inc, omega, Omega, G)

    # Vectorized positions/velocities
    r_rel, v_rel = jax.vmap(relative_pos_vel_at, in_axes=(0, None))(ts, params)

    # Projected separation on the plane of the sky
    r_rel_dot_los = jnp.dot(r_rel, los_vector)
    r_rel_sq = jnp.sum(r_rel**2, axis=-1)
    rho_sq = r_rel_sq - r_rel_dot_los**2
    rho = jnp.sqrt(jnp.maximum(0.0, rho_sq)) # ensure non-negative due to precision

    zsign = jnp.sign(r_rel_dot_los)  # <0: secondary in front (primary eclipse)

    # Identify local minima (exclude endpoints)
    # A point k is a minimum if rho[k] <= rho[k-1] and rho[k] <= rho[k+1]
    rho_prev = jnp.roll(rho, 1)
    rho_next = jnp.roll(rho, -1)
    is_min = (rho <= rho_prev) & (rho <= rho_next)
    # avoid rolled comparisons at the edges
    is_min = is_min.at[0].set(False).at[-1].set(False)

    # Split candidates by front/back
    is_front = (zsign < 0) & is_min
    is_back  = (zsign > 0) & is_min

    big = 1e300
    rho_front = jnp.where(is_front, rho, big)
    rho_back  = jnp.where(is_back,  rho, big)

    # Best minima indices (argmin over masked arrays)
    i_front = jnp.argmin(rho_front)
    i_back  = jnp.argmin(rho_back)

    # Guard: if no candidate exists, set an index with NaN later
    has_front = jnp.isfinite(jnp.where(i_front < nscan, rho_front[i_front], big)) & (rho_front[i_front] < big)
    has_back  = jnp.isfinite(jnp.where(i_back  < nscan, rho_back[i_back],  big)) & (rho_back[i_back]  < big)

    # Parabolic refinement (rho^2) around the chosen minima
    def refine(i):
        i0 = jnp.clip(i, 1, nscan-2)
        t0, t1, t2 = ts[i0-1], ts[i0], ts[i0+1]
        y0, y1, y2 = rho[i0-1]**2, rho[i0]**2, rho[i0+1]**2
        t_mid = parabola_vertex_t(t0, t1, t2, y0, y1, y2)
        # Evaluate r,v at refined time
        r_m, v_m = relative_pos_vel_at(t_mid, params)
        
        # Projected separation and velocity on the plane of the sky
        r_m_dot_los = jnp.dot(r_m, los_vector)
        rho_m_sq = jnp.sum(r_m**2) - r_m_dot_los**2
        rho_m = jnp.sqrt(jnp.maximum(0.0, rho_m_sq))

        v_m_dot_los = jnp.dot(v_m, los_vector)
        v_m_sq = jnp.sum(v_m**2)
        vperp_sq = v_m_sq - v_m_dot_los**2
        vperp = jnp.sqrt(jnp.maximum(0.0, vperp_sq))
        
        return t_mid, rho_m, vperp

    tmid_f, b_f, vperp_f = refine(i_front)
    tmid_b, b_b, vperp_b = refine(i_back)

    # Durations from chord geometry (clip if no overlap)
    def durations(b, vperp):
        # Outer & inner chord lengths
        L14 = 2.0*jnp.sqrt(jnp.maximum(0.0, (R1+R2)**2 - b*b))
        L23 = 2.0*jnp.sqrt(jnp.maximum(0.0, (jnp.abs(R1-R2))**2 - b*b))
        T14 = L14 / jnp.maximum(vperp, 1e-30)
        T23 = L23 / jnp.maximum(vperp, 1e-30)
        return T14, T23

    T14_f, T23_f = durations(b_f, vperp_f)
    T14_b, T23_b = durations(b_b, vperp_b)

    # Accept eclipses only if b < (R1+R2)*pad
    ok_f = has_front & (b_f < (R1 + R2)*pad)
    ok_b = has_back  & (b_b < (R1 + R2)*pad)

    def times_from(tmid, T14, T23):
        t1 = tmid - 0.5*T14
        t4 = tmid + 0.5*T14
        # If T23 == 0 (grazing/none), return NaN for inner contacts
        t2 = jnp.where(T23 > 0, tmid - 0.5*T23, jnp.nan)
        t3 = jnp.where(T23 > 0, tmid + 0.5*T23, jnp.nan)
        return t1, t2, t3, t4

    t1f, t2f, t3f, t4f = times_from(tmid_f, T14_f, T23_f)
    t1b, t2b, t3b, t4b = times_from(tmid_b, T14_b, T23_b)

    # Mask outputs if not ok
    def mask_block(ok, tmid, t1, t2, t3, t4):
        nan = jnp.nan
        return (jnp.where(ok, tmid, nan),
                jnp.where(ok, t1,   nan),
                jnp.where(ok & jnp.isfinite(t2), t2, nan),
                jnp.where(ok & jnp.isfinite(t3), t3, nan),
                jnp.where(ok, t4,   nan))

    tmid_p, t1_p, t2_p, t3_p, t4_p = mask_block(ok_f, tmid_f, t1f, t2f, t3f, t4f)
    tmid_s, t1_s, t2_s, t3_s, t4_s = mask_block(ok_b, tmid_b, t1b, t2b, t3b, t4b)

    return (tmid_p / c.yr, t1_p / c.yr, t2_p / c.yr, t3_p / c.yr, t4_p / c.yr,
            tmid_s / c.yr, t1_s / c.yr, t2_s / c.yr, t3_s / c.yr, t4_s / c.yr)
