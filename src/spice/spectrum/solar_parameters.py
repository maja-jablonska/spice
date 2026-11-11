TEFF = 5777.0
LOGG = 4.44
VMICRO = 2.0

SOLAR_PARAMETERS = {
    "teff": TEFF,
    "logg": LOGG,
    "v_micro": VMICRO,
}

# Emulator inputs use many spellings for the same quantity (``teff``/``Teff``,
# ``vmicro``/``v_micro``) and atmosphere-grid prefixes (``marcs_teff``). Matching
# only the exact keys above silently gave every other spelling a solar default
# of 0.0 -- e.g. log g = 0 and vmicro = 0 for the HARPS bundle's marcs_* inputs.
_ATMOSPHERE_PREFIXES = ("marcs_", "atm_", "atmosphere_")
_ALIASES = {
    "teff": TEFF, "t_eff": TEFF, "effective_temperature": TEFF,
    "logg": LOGG, "log_g": LOGG, "surface_gravity": LOGG,
    "v_micro": VMICRO, "vmicro": VMICRO, "vmic": VMICRO, "v_mic": VMICRO,
    "xi": VMICRO, "microturbulence": VMICRO, "vturb": VMICRO,
}


def solar_value(name: str, default: float = 0.0) -> float:
    """Solar value for an emulator input name, or ``default`` (0.0) if unknown.

    Case-insensitive, ignores atmosphere-grid prefixes such as ``marcs_``, and
    knows the usual aliases. Metallicities and relative abundances are 0 for the
    Sun, which is what ``default`` gives them.
    """
    key = str(name).lower()
    for prefix in _ATMOSPHERE_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return _ALIASES.get(key, default)
