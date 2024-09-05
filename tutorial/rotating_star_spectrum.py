from spice.models import IcosphereModel
from spice.models.mesh_transform import add_rotation, evaluate_rotation
from spice.models.spots import add_spots
from spice.plots import plot_3D, plot_3D_mesh_and_spectrum
from spice.spectrum import simulate_observed_flux
import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pickle
from transformer_payne import TransformerPayne

# from jax import config
# config.update("jax_enable_x64", True)

print("WARNING: This script should be run on a GPU with significant memory (>40GB) or the chunk sizes should be adjusted!")
print("    Calculation still is supposed to take ~2h10min on single A100 40GB with default settings")

min_number_of_triangles = 1500
no_times = 100
rotation_velocity_on_equator = 100 # km/s
chunk_size = 32

tp = TransformerPayne.download()

base_logtemp = np.log10(4000)
splot_logtemp = np.log10(8000)

m = IcosphereModel.construct(min_number_of_triangles, u.solRad.to(u.cm), 1., 1., tp.to_parameters(dict(logteff=base_logtemp)))

m = add_spots(mesh=m,
              thetas=jnp.array([0.5*jnp.pi]),
              phis=jnp.array([0.]),
              param_deltas=jnp.array([splot_logtemp - base_logtemp]),
              radii=jnp.array([jnp.pi/4]),
              radius_factors=jnp.array([1.0]),
              param_indices=jnp.array([0]))

TIMESTAMPS = np.linspace(0, 3600*12.14, no_times)

mt = add_rotation(m, rotation_velocity_on_equator, jnp.array([0., 0., 1.]))
mts = [evaluate_rotation(mt, t) for t in TIMESTAMPS]

vws = jnp.concatenate([jnp.linspace(3000., 3999.9, 1000),
                        jnp.linspace(4000., 5000., 20000),
                        jnp.linspace(5000.1, 11000, 1000)])

# for a single spectrum:
%time spec = simulate_observed_flux(tp.intensity, mts[0], jnp.log10(vws), chunk_size=chunk_size).block_until_ready() # Wall time: 30.7 s on A100, 40GB for 1280 triangles (1min 24s for 5120)
%time spec = simulate_observed_flux(tp.intensity, mts[0], jnp.log10(vws), chunk_size=chunk_size).block_until_ready() # Wall time: 18.7 s on A100, 40GB for 1280 triangles (1min 17s for 5120)

# Compute spectra:
# This should take about 15min on A100, 40GB (50 times, 1280 traingles) 
# Or: 2h10min for 100 times and 5120 triangles
%time specs = [simulate_observed_flux(tp.intensity, mt0, jnp.log10(vws), chunk_size=chunk_size) for mt0 in mts]

# Now compute photometric time series
from spice.spectrum.filter import BesselU, BesselB, BesselV
from spice.spectrum.spectrum import AB_passband_luminosity

bessel_B = BesselB()
bessel_U = BesselU()
bessel_V = BesselV()

d = u.AU.to(u.cm)
# TODO: remove this "/1e30" that fixes the problem of overflow in float32 by setting the precision to double.
U_phot = np.array([AB_passband_luminosity(bessel_U, vws, flux[:, 0]/1e30, distance=d) for flux in specs])
B_phot = np.array([AB_passband_luminosity(bessel_B, vws, flux[:, 0]/1e30, distance=d) for flux in specs])
V_phot = np.array([AB_passband_luminosity(bessel_V, vws, flux[:, 0]/1e30, distance=d) for flux in specs])

# Define a dictionary to hold all the data
data = {
    "timestamps": TIMESTAMPS,
    "spectra": np.array(specs),
    "U_phot": U_phot,
    "B_phot": B_phot,
    "V_phot": V_phot
}

# Specify the filename
filename = 'data/stellar_data.pkl'

# Write the data to a file
with open(filename, 'wb') as file:
    pickle.dump(data, file)

print("Data successfully saved to:", filename)
