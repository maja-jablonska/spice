import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial


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

def get_orbit_jax(time, m1, m2, P, ecc, T, i, omega, Omega) -> ArrayLike:
    M1 = m1
    M2 = m2
    period_seconds = P*c.yr
    a = (period_seconds**2 * c.G * (M1 + M2) / (4 * jnp.pi**2))**(1/3)*100 # m -> cm
    M_t = 2*jnp.pi / P * (time - T) # mean anomaly
    E = solve_kepler_jax(M_t, ecc) # eccentric anomaly
    nu = 2 * jnp.arctan2(jnp.sqrt((1+ecc)/(1-ecc)) * jnp.sin(E/2), jnp.cos(E/2)) # true anomaly
    r = a * (1 - ecc * jnp.cos(E)) # distance
    vec_x = jnp.array([r* jnp.cos(nu), r* jnp.sin(nu), jnp.zeros_like(r)])
    C = jnp.sqrt(c.G*(M1+M2)/(a*(1-ecc**2)))
    vec_v = jnp.array([-C * jnp.sin(nu), C * (ecc + jnp.cos(nu)), jnp.zeros_like(r)])
    # now get orbits for the bodies 1 and 2
    vec_v1 = -vec_v * M2 / (M1 + M2)
    vec_v2 = vec_v * M1 / (M1 + M2)
    vec_x1 = -vec_x * M2 / (M1 + M2)
    vec_x2 = vec_x * M1 / (M1 + M2)
    # visual orbit:
    # Positive x is North, positive y is West, (positive z is towards the observer ?)
    vec_x_obs, vec_v_obs = transform_orbit(vec_x, vec_v, (i, omega, Omega))
    vec_x1_obs, vec_v1_obs = transform_orbit(vec_x1, vec_v1, (i, omega, Omega))
    vec_x2_obs, vec_v2_obs = transform_orbit(vec_x2, vec_v2, (i, omega, Omega))
    # In this code: the negative y is East and negative x is North
    return jnp.stack((vec_x_obs, vec_v_obs, vec_x1_obs, vec_v1_obs, vec_x2_obs, vec_v2_obs), axis=0)
