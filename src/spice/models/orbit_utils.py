import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial


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

@partial(jax.jit, static_argnums=(2,))
def solve_kepler_jax(M_t, e, iterations = 10):
    E = M_t
    # Manually unroll the loop, 3 iterations
    # this is not enough for high eccentricities
    # TODO: improve this!!!
    def kepler_scan(E, x):
        E = E - (E - e * jnp.sin(E) - M_t) / (1 - e * jnp.cos(E))
        return E, x
    
    E, _ = jax.lax.scan(kepler_scan, E, jnp.linspace(0, 1, iterations))
    return E

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

# def get_orbit_jax(time, m1, m2, P, ecc, T, i, omega, Omega) -> ArrayLike:
#     M1 = m1
#     M2 = m2
#     period_seconds = P*c.yr
#     a = (period_seconds**2 * c.G * (M1 + M2) * SOLAR_MASS_KG / (4 * jnp.pi**2))**(1/3)*100 # m -> cm
#     M_t = 2*jnp.pi / P * (time - T) # mean anomaly
#     E = solve_kepler_jax(M_t, ecc) # eccentric anomaly
#     nu = 2 * jnp.arctan2(jnp.sqrt((1+ecc)/(1-ecc)) * jnp.sin(E/2), jnp.cos(E/2)) # true anomaly
#     r = a * (1 - ecc * jnp.cos(E)) # distance
#     vec_x = jnp.array([r* jnp.cos(nu), r* jnp.sin(nu), jnp.zeros_like(r)])
#     C = jnp.sqrt(c.G*(M1+M2)*SOLAR_MASS_KG/(a*(1-ecc**2)))
#     vec_v = jnp.array([-C * jnp.sin(nu), C * (ecc + jnp.cos(nu)), jnp.zeros_like(r)])
#     # now get orbits for the bodies 1 and 2
#     vec_v1 = -vec_v * M2 / (M1 + M2)
#     vec_v2 = vec_v * M1 / (M1 + M2)
#     vec_x1 = -vec_x * M2 / (M1 + M2)
#     vec_x2 = vec_x * M1 / (M1 + M2)
#     # visual orbit:
#     # Positive x is North, positive y is West, (positive z is towards the observer ?)
#     vec_x_obs, vec_v_obs = transform_orbit(vec_x, vec_v, (i, omega, Omega))
#     vec_x1_obs, vec_v1_obs = transform_orbit(vec_x1, vec_v1, (i, omega, Omega))
#     vec_x2_obs, vec_v2_obs = transform_orbit(vec_x2, vec_v2, (i, omega, Omega))
#     # In this code: the negative y is East and negative x is North
#     return jnp.stack((vec_x_obs, vec_v_obs, vec_x1_obs, vec_v1_obs, vec_x2_obs, vec_v2_obs), axis=0)

import jax
import jax.numpy as jnp

@jax.jit
def _solve_kepler(M, ecc, max_iter=8):
    """
    Solve Kepler's equation M = E - ecc * sin(E) for the eccentric anomaly E.
    Uses a fixed number of Newton iterations (vectorized over time).
    """
    # Initial guess: E0 = M.
    E0 = M

    def body_fun(_, E):
        # Newton's method update.
        f = E - ecc * jnp.sin(E) - M
        fprime = 1 - ecc * jnp.cos(E)
        return E - f / fprime

    E = jax.lax.fori_loop(0, max_iter, body_fun, E0)
    return E


def get_orbit_jax(time, m1, m2, P, ecc, T, i, omega, Omega):
    """
    Compute a Keplerian orbit for a binary system using JAX.
    
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
    
    Returns:
      pos1 : Positions of body 1 relative to the barycenter; an array of shape (N, 3).
      pos2 : Positions of body 2 relative to the barycenter; an array of shape (N, 3).
      vel1 : Velocities of body 1 relative to the barycenter; an array of shape (N, 3).
      vel2 : Velocities of body 2 relative to the barycenter; an array of shape (N, 3).
    """
    # Gravitational constant in SI units [m^3/(kg s^2)]
    G = 6.67430e-11
    
    time = time*c.yr

    m1 = m1*c.Msun
    m2 = m2*c.Msun
    P = P*c.yr
    T = T*c.yr

    # Total mass.
    M_tot = m1 + m2

    # Compute semimajor axis of the relative orbit from Kepler's third law:
    #   P^2 = (4π²/G(M₁+M₂)) * a³   =>   a = [G (m1+m2) P²/(4π²)]^(1/3)
    a = (G * M_tot * P**2 / (4 * jnp.pi**2))**(1/3)
    #jax.debug.print("a: {a}", a=a)

    # Mean motion.
    n = 2 * jnp.pi / P

    # Mean anomaly.
    M_anom = n * (time - T) - (jnp.pi/2)

    # Solve Kepler's equation for the eccentric anomaly E.
    E = _solve_kepler(M_anom, ecc)

    # --- Compute positions in the orbital plane ---
    # Using the parametric form of an ellipse:
    #   x_orb = a (cos E - ecc)
    #   y_orb = a sqrt(1 - ecc²) sin E
    x_orb = a * (jnp.cos(E) - ecc)
    y_orb = a * jnp.sqrt(1 - ecc**2) * jnp.sin(E)
    # The orbit lies in the x-y plane.
    r_orb = jnp.stack([x_orb, y_orb, jnp.zeros_like(x_orb)], axis=-1)

    # --- Compute velocities in the orbital plane ---
    # First, compute dE/dt = n / (1 - ecc*cos E)
    dE_dt = n / (1 - ecc * jnp.cos(E))
    # Then, differentiate the parametric equations:
    vx_orb = -a * jnp.sin(E) * dE_dt
    vy_orb = a * jnp.sqrt(1 - ecc**2) * jnp.cos(E) * dE_dt
    v_orb = jnp.stack([vx_orb, vy_orb, jnp.zeros_like(vx_orb)], axis=-1)

    # --- Rotate from the orbital plane to inertial coordinates ---
    # The rotation is performed by R = Rz(Omega) @ Rx(i) @ Rz(omega)
    cos_omega, sin_omega = jnp.cos(omega), jnp.sin(omega)
    cos_Omega, sin_Omega = jnp.cos(Omega), jnp.sin(Omega)
    cos_i, sin_i = jnp.cos(i), jnp.sin(i)

    Rz_omega = jnp.array([[ cos_omega, -sin_omega, 0],
                          [ sin_omega,  cos_omega, 0],
                          [       0.0,        0.0, 1]])
    Rx_i = jnp.array([[1,      0.0,     0.0],
                      [0,   cos_i,   -sin_i],
                      [0,   sin_i,    cos_i]])
    Rz_Omega = jnp.array([[ cos_Omega, -sin_Omega, 0],
                          [ sin_Omega,  cos_Omega, 0],
                          [       0.0,        0.0, 1]])
    
    # Full rotation matrix.
    R = Rz_Omega @ Rx_i @ Rz_omega

    # Apply the rotation.
    # Here r_orb and v_orb are arrays of shape (N, 3). We can apply the rotation
    # by a matrix multiplication on the right (using the transpose of R).
    r_inertial = jnp.dot(r_orb, R.T)
    v_inertial = jnp.dot(v_orb, R.T)

    # --- Compute the individual orbits relative to the barycenter ---
    # The reduced (relative) orbit is split using the mass ratio.
    # Positions:
    pos1 = - (m2 / M_tot) * r_inertial  # Body 1
    pos2 =   (m1 / M_tot) * r_inertial  # Body 2
    # Velocities:
    vel1 = - (m2 / M_tot) * v_inertial
    vel2 =   (m1 / M_tot) * v_inertial

    #return pos1, pos2, vel1, vel2
    
    # ---------------------------
    # 8. Assemble the Output
    # ---------------------------
    # Stack the arrays into a single output of shape (6, len(time), 3):
    orbit = jnp.stack([
        r_inertial * jnp.array([1, 1, -1]),  # orbit[0]: barycenter positions
        v_inertial/1e3 * jnp.array([1, 1, -1]),  # orbit[1]: barycenter velocities
        pos1 * jnp.array([1, 1, -1]),              # orbit[2]: displacement of body 1
        vel1/1e3 * jnp.array([1, 1, -1]),              # orbit[3]: velocity of body 1 [km/s]
        pos2 * jnp.array([1, 1, -1]),              # orbit[4]: displacement of body 2
        vel2/1e3 * jnp.array([1, 1, -1])               # orbit[5]: velocity of body 2 [km/s]
    ], axis=0)
    
    return orbit
