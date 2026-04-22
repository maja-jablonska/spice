"""Lightweight logging helpers for spice.

Two goals:

1. Messages like ``[spice] ...`` cooperate with any active ``tqdm`` progress
   bar so the bar is not redrawn on its own line for every log line.
2. "Timed" operations produce a single, clean completion line, so the
   output reads like::

       [spice] IcosphereModel constructed in 0.3 s
       [spice] Pulsations evaluated in 0.7 s

   instead of alternating ``Evaluating...`` / ``Evaluated in X s`` pairs.

By default the "starting" half of a :func:`timed` block is **suppressed**.
Set ``SPICE_LOG_VERBOSE=1`` (or pass ``verbose=True`` explicitly) to bring
the start line back. In that mode the helper also tries to substitute the
start line in place with the completion line using ANSI escape codes; this
works reliably in real terminals and falls back to a plain new-line write
in environments where cursor-movement escapes aren't honoured (e.g.
Jupyter, pipes to files).

Use::

    from spice.utils import log

    log.info("Doing something")

    with log.timed("Constructing model", "Model constructed in {elapsed:.1f} s"):
        ...  # work

The helpers are intentionally dependency-free at import time (``tqdm`` is
only imported lazily) so this module is safe to import from anywhere.
"""

from __future__ import annotations

import os
import sys
import time as _time
from contextlib import contextmanager

_PREFIX = "[spice]"

_ANSI_ERASE_PREV_LINE = "\x1b[1A\x1b[2K\r"


def is_verbose() -> bool:
    """Return ``True`` if verbose logging is enabled via ``SPICE_LOG_VERBOSE``."""
    return os.environ.get("SPICE_LOG_VERBOSE", "0").lower() not in ("", "0", "false", "no")


def _env_verbose() -> bool:  # backwards-compatible alias
    return is_verbose()


def _active_tqdm_instances():
    """Return active tqdm bar instances (empty tuple if tqdm unavailable/idle).

    We look at ``tqdm.std.tqdm._instances`` (the shared WeakSet) so that bars
    created via any import path — ``from tqdm import tqdm``, ``tqdm.auto``,
    ``tqdm.notebook``, ``tqdm.std`` — are all visible here.
    """
    try:
        from tqdm.std import tqdm as _tqdm_std
    except Exception:
        return ()
    try:
        return tuple(_tqdm_std._instances)
    except Exception:
        return ()


def _tqdm_write_target():
    """Return ``(tqdm_cls, file)`` to use for cooperative writes, or ``None``.

    We always write to the same stream the active bar is using so that
    ``tqdm.write``'s clear/refresh bookkeeping stays on one output stream.
    In Jupyter, stdout and stderr are rendered as distinct streams, so
    writing log messages to stdout while the bar lives on stderr makes the
    bar be re-drawn as a fresh line every time (the symptom this fixes).
    """
    instances = _active_tqdm_instances()
    if not instances:
        return None
    try:
        from tqdm.std import tqdm as _tqdm_std
    except Exception:
        return None
    bar = instances[0]
    fp = getattr(bar, "fp", None) or sys.stderr
    return _tqdm_std, fp


def _supports_ansi() -> bool:
    """Heuristic: is the current stdout capable of handling ANSI cursor escapes?

    Note: we specifically look for a TTY. Jupyter kernels capture stdout/stderr
    as separate streams and typically do *not* honour cursor-movement escapes,
    so we treat them as "no ANSI substitution" even though colour codes do work.
    """
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _write_line(msg: str) -> None:
    """Write a single log line, playing nicely with any active tqdm bars.

    Routes the write to the same file as the active bar so the bar's
    ``\\r``-based self-updating output isn't interleaved onto a fresh line
    for every log message (which is what happens if log text is written to
    stdout while the bar lives on stderr).
    """
    target = _tqdm_write_target()
    if target is not None:
        tqdm_cls, fp = target
        tqdm_cls.write(msg, file=fp)
    else:
        print(msg, flush=True)


def _overwrite_prev_line(msg: str) -> None:
    """Erase the previously printed log line and write ``msg`` in its place.

    Falls back to :func:`_write_line` (a plain new-line write) in environments
    where ANSI cursor escapes aren't reliable (non-TTY, Jupyter, piped output).
    """
    if not _supports_ansi():
        _write_line(msg)
        return

    target = _tqdm_write_target()
    if target is not None:
        tqdm_cls, fp = target
        try:
            tqdm_cls.write(f"{_ANSI_ERASE_PREV_LINE}{msg}", file=fp)
            return
        except Exception:
            pass

    sys.stdout.write(_ANSI_ERASE_PREV_LINE)
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def info(msg: str) -> None:
    """Log a single ``[spice] ...`` message."""
    _write_line(f"{_PREFIX} {msg}")


def substitute(msg: str) -> None:
    """Log ``[spice] <msg>`` overwriting the previously printed line in place.

    Useful when the final message depends on data computed during the operation
    and can't be expressed as a static :func:`timed` template. Falls back to a
    plain new-line write when ANSI cursor escapes aren't available.
    """
    _overwrite_prev_line(f"{_PREFIX} {msg}")


@contextmanager
def timed(
    start_msg: str,
    done_template: str | None = None,
    inplace: bool = True,
    verbose: bool | None = None,
):
    """Log a timed operation.

    On exit, prints ``[spice] <done_template.format(elapsed=<seconds>)>``.

    By default the starting message is **not** printed, which yields a clean
    single-line completion log in any environment. Set the environment
    variable ``SPICE_LOG_VERBOSE=1`` or pass ``verbose=True`` to also emit
    ``[spice] <start_msg>...`` on enter, in which case the completion line
    will additionally try to overwrite the start line in place (controlled by
    ``inplace``).

    Args:
        start_msg: Description of the operation (e.g. ``"Constructing
            IcosphereModel"``). Used both as the start-line message (when
            verbose) and, when ``done_template`` is ``None``, as the basis
            for the default completion template.
        done_template: Template for the completion message. Must accept a
            ``{elapsed}`` placeholder. Defaults to
            ``f"{start_msg} done in {{elapsed:.1f}} s"``.
        inplace: When ``True`` (default) and a start line was emitted, try
            to overwrite it in place with the completion line. Set to
            ``False`` when the block emits other output (e.g. a nested
            ``tqdm`` bar) between the start and the completion.
        verbose: Override the environment default for this call. When
            ``None`` (default), the ``SPICE_LOG_VERBOSE`` env var decides.
    """
    show_start = _env_verbose() if verbose is None else bool(verbose)

    if show_start:
        _write_line(f"{_PREFIX} {start_msg}...")

    t0 = _time.perf_counter()
    try:
        yield
    finally:
        elapsed = _time.perf_counter() - t0
        template = done_template if done_template is not None else (
            start_msg + " done in {elapsed:.1f} s"
        )
        done_msg = template.format(elapsed=elapsed)
        line = f"{_PREFIX} {done_msg}"

        if show_start and inplace:
            _overwrite_prev_line(line)
        else:
            _write_line(line)


__all__ = ["info", "is_verbose", "substitute", "timed"]
