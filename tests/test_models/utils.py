from spice.models import IcosphereModel


SOLAR_RAD_CM = 69570000000.0
SOLAR_MASS_KG = 1.988409870698051e+30


def default_icosphere():
    return IcosphereModel.construct(100, SOLAR_RAD_CM, SOLAR_MASS_KG,
                                    [5700, 0.], ['teff', 'abun'])
