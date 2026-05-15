from spice.models import IcosphereModel

def default_icosphere(radius: float = 1.0, mass: float = 1.0):
    return IcosphereModel.construct(100, radius, mass,
                                    [5700, 0.], ['teff', 'abun'])
