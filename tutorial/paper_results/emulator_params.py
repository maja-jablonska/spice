"""Map canonical stellar-parameter names onto whatever an emulator calls them.

Aemu bundles carry their own parameter contract, and it changes between
revisions. ``RozanskiT/TPayne-spice-harps`` is the cautionary example: the
May 2026 revision (``bcc8a6e``) exposed

    ['teff', 'logg', '[Fe/H]', 'vmicro', '[a/Fe]', '[C/Fe]', ...]

while the August 2026 retrain (``3df764e``, the one that fixed the 250 K Teff
quantization) exposes

    ['marcs_teff', 'marcs_logg', 'feh', 'vmicro', 'a', 'c', 'n', 'o', 'r', 's']

``SpectrumEmulator.to_parameters`` fills names it does not recognize with
``0.0`` and only *warns* that the resulting vector is out of domain, so a
script written against the old contract keeps running and quietly synthesizes
nonsense (Teff = 0 K). The zarr interpolators use yet another set of names, and
``TPayne-spice-small-random`` wants ``logteff`` — log10 of the temperature, not
the temperature.

Resolve names through this module instead of hardcoding them: it accepts any of
the spellings below, maps them to the emulator's own, applies the log10
transform where a bundle wants it, and raises instead of silently zero-filling
when a parameter the caller considers essential has no counterpart.

Usage::

    import emulator_params as ep

    params = ep.to_parameters(em, dict(teff=6562.0, logg=1.883, feh=0.06))
    i_teff = ep.parameter_index(em, "teff")        # None if the bundle lacks it
    teff_fn = ep.value_transform(em, "teff")       # identity, or log10
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Canonical name -> every bundle spelling we have seen for it. The canonical
# name itself is always accepted; order matters only for readability.
CANONICAL_ALIASES: Dict[str, tuple] = {
    "teff":   ("teff", "marcs_teff", "Teff", "T_eff", "logteff", "log_teff"),
    "logg":   ("logg", "marcs_logg", "log_g", "loggs", "surface_gravity"),
    "feh":    ("feh", "[Fe/H]", "fe_h", "marcs_fe_h", "m_h", "[M/H]", "mh"),
    "vmicro": ("vmicro", "vmic", "v_micro", "marcs_turb", "turbvel", "microturbulence"),
    "a_fe":   ("a_fe", "[a/Fe]", "a", "afe", "alpha", "marcs_a_fe", "[alpha/Fe]"),
    "c_fe":   ("c_fe", "[C/Fe]", "c", "cfe", "marcs_c_fe"),
    "n_fe":   ("n_fe", "[N/Fe]", "n", "nfe", "marcs_n_fe"),
    "o_fe":   ("o_fe", "[O/Fe]", "o", "ofe", "marcs_o_fe"),
    "r_fe":   ("r_fe", "[r/Fe]", "r", "rfe", "marcs_r_fe"),
    "s_fe":   ("s_fe", "[s/Fe]", "s", "sfe", "marcs_s_fe"),
}

# Bundle parameter names that hold log10 of the canonical quantity.
LOG10_PARAMETER_NAMES = frozenset({"logteff", "log_teff"})

# Parameters whose absence makes a synthesis meaningless rather than merely
# approximate. Callers can override per call site.
DEFAULT_REQUIRED = ("teff", "logg")

_ALIAS_TO_CANONICAL: Dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in CANONICAL_ALIASES.items()
    for alias in aliases
}


def canonical_name(name: str) -> str:
    """Canonical name for ``name``.

    Raises:
        KeyError: if ``name`` is not a spelling this module knows. Better to
            fail here than to hand an unknown key to ``to_parameters`` and have
            it silently become 0.0.
    """
    try:
        return _ALIAS_TO_CANONICAL[str(name).lower()]
    except KeyError:
        raise KeyError(
            f"unknown stellar parameter name {name!r}; known names: "
            + ", ".join(sorted(_ALIAS_TO_CANONICAL))
        ) from None


def emulator_parameter_names(emulator) -> List[str]:
    """``emulator.stellar_parameter_names`` as a plain list."""
    return list(getattr(emulator, "stellar_parameter_names", []) or [])


def resolve_names(emulator) -> Dict[str, str]:
    """Map canonical name -> the name this emulator actually uses.

    Only parameters the emulator carries appear in the result.
    """
    resolved: Dict[str, str] = {}
    for bundle_name in emulator_parameter_names(emulator):
        canonical = _ALIAS_TO_CANONICAL.get(str(bundle_name).lower())
        # First spelling wins: a bundle carrying both `teff` and `logteff`
        # would be ambiguous, and we have never seen one.
        if canonical is not None and canonical not in resolved:
            resolved[canonical] = bundle_name
    return resolved


def parameter_index(emulator, name: str) -> Optional[int]:
    """Index of ``name`` (any spelling) in the emulator's parameter vector.

    Returns ``None`` when the emulator does not carry that parameter.
    """
    canonical = canonical_name(name)
    bundle_name = resolve_names(emulator).get(canonical)
    if bundle_name is None:
        return None
    return emulator_parameter_names(emulator).index(bundle_name)


def value_transform(emulator, name: str) -> Callable[[float], float]:
    """Transform taking a canonical value to what this emulator expects.

    Identity for every bundle we ship against except the ``logteff`` ones,
    where it is log10.
    """
    canonical = canonical_name(name)
    bundle_name = resolve_names(emulator).get(canonical)
    if bundle_name is not None and str(bundle_name).lower() in LOG10_PARAMETER_NAMES:
        return lambda v: math.log10(float(v))
    return lambda v: float(v)


def bundle_values(emulator, values: Dict[str, float],
                  required: Sequence[str] = DEFAULT_REQUIRED) -> Dict[str, float]:
    """Rekey ``values`` (any spellings) to the emulator's own parameter names.

    Parameters the emulator does not carry are dropped — that is a legitimate
    difference between bundles (not every one has vmicro). Parameters listed in
    ``required`` must resolve, or this raises.
    """
    resolved = resolve_names(emulator)
    out: Dict[str, float] = {}
    for name, value in values.items():
        canonical = canonical_name(name)
        bundle_name = resolved.get(canonical)
        if bundle_name is None:
            continue
        out[bundle_name] = value_transform(emulator, canonical)(value)

    missing = [name for name in required if canonical_name(name) not in resolved]
    if missing:
        raise ValueError(
            f"emulator does not expose {missing} under any known name — its "
            f"parameters are {emulator_parameter_names(emulator)}. Add the "
            "bundle's spelling to CANONICAL_ALIASES in emulator_params.py; "
            "without it to_parameters would silently substitute 0.0."
        )
    return out


def to_parameters(emulator, values: Dict[str, float],
                  required: Sequence[str] = DEFAULT_REQUIRED) -> np.ndarray:
    """``emulator.to_parameters`` with the names resolved first."""
    return emulator.to_parameters(bundle_values(emulator, values, required=required))


def phase_modifiers(emulator, modifiers: Iterable,
                    required: Sequence[str] = DEFAULT_REQUIRED) -> List[tuple]:
    """Build ``[(parameter_index, fn), ...]`` for phase-dependent parameters.

    ``modifiers`` is an iterable of ``(canonical_name, fn)`` where ``fn`` maps a
    phase to the canonical-units value. The returned callables include the
    emulator's own transform, so a ``logteff`` bundle gets log10(Teff(phase)).
    Names the emulator lacks are dropped; names in ``required`` must resolve.
    """
    out: List[tuple] = []
    seen: List[str] = []
    for name, fn in modifiers:
        canonical = canonical_name(name)
        seen.append(canonical)
        index = parameter_index(emulator, canonical)
        if index is None:
            continue
        transform = value_transform(emulator, canonical)
        out.append((index, (lambda f, t: lambda phase: t(f(phase)))(fn, transform)))

    resolved = resolve_names(emulator)
    missing = [name for name in required
               if canonical_name(name) in seen and canonical_name(name) not in resolved]
    if missing:
        raise ValueError(
            f"phase-dependent {missing} cannot be applied: the emulator exposes "
            f"{emulator_parameter_names(emulator)}. Synthesizing with these held "
            "constant would silently change the experiment."
        )
    return out
