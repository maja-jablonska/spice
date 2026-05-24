"""Physical and astronomical constants shared across SPICE.

Collected here so that the same quantity is not redefined — and allowed to
drift — in several modules. Speed of light, Planck's constant and Boltzmann's
constant are exact by the 2019 SI definitions; the solar mass/radius follow the
IAU 2015 nominal values.

Multiple unit variants are provided (rather than computed on the fly) so that
existing call sites can switch to the shared name without any change in value.
"""

# -- Speed of light ---------------------------------------------------------
C_KM_S = 299792.458          # km / s
C_CM_S = 2.99792458e10       # cm / s

# -- Planck constant --------------------------------------------------------
H_ERG_S = 6.62607015e-27     # erg s

# -- Boltzmann constant -----------------------------------------------------
K_B_ERG_K = 1.380649e-16     # erg / K

# -- Proton mass ------------------------------------------------------------
M_P_G = 1.6726219e-24        # g

# -- Newtonian gravitational constant --------------------------------------
G_SI = 6.674e-11             # m^3 kg^-1 s^-2

# -- Solar mass -------------------------------------------------------------
SOLAR_MASS_KG = 1.988409870698051e30   # kg

# -- Solar radius -----------------------------------------------------------
SOLAR_RAD_M = 6.957e8        # m
SOLAR_RAD_CM = 6.957e10      # cm
SOLAR_RAD_KM = 6.957e5       # km

# -- Time -------------------------------------------------------------------
DAY_TO_S = 86400.0           # s
