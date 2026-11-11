"""Shared helper for tests that need the optional PHOEBE extra."""
import pytest


def import_phoebe_or_skip():
    """Import PHOEBE, skipping the test if it is absent *or* broken.

    ``pytest.importorskip`` only catches ImportError, but a PHOEBE that is
    installed against incompatible astropy/NumPy fails with AttributeError and
    would otherwise error out the whole module.
    """
    try:
        import phoebe
    except ImportError:
        pytest.skip("PHOEBE is an optional extra", allow_module_level=True)
    except Exception as e:  # installed but broken
        pytest.skip(f"PHOEBE is installed but fails to import: {e!r}",
                    allow_module_level=True)
    return phoebe
