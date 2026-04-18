from spice.models import IcosphereModel

def default_icosphere():
    return IcosphereModel.construct(100, 1., 1.,
                                    [5700, 0.], ['teff', 'abun'])
