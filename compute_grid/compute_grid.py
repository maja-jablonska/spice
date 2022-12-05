import numpy as np
import h5py

from vidmapy.kurucz.atlas import Atlas
from vidmapy.kurucz.synthe import Synthe
from vidmapy.kurucz.parameters import Parameters
import multiprocessing as mp

def compute_spectrum(p: Parameters):

    atlas_worker= Atlas()
    model= atlas_worker.get_model(p)

    synthe_worker= Synthe()
    spectrum = synthe_worker.get_spectrum(model, parameters=p, quiet=True)


if __main__ == '__name__':
    p= Parameters(wave_min=3000, 
                  wave_max=10000, 
                  resolution=300000, 
                  metallicity=0.0,
                  teff=10000,
                  logg=3.5,
                  metallicity=0.0,
                  microturbulence=2.
                 )

    