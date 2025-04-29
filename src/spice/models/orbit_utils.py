import jax
import jax.numpy as jnp
from jax import jit


SOLAR_MASS_KG = 1.988409870698051e+30


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