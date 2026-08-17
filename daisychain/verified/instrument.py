"""Invocation counters for the verified units.

Turn it on and every verified unit records how many times its neural forward
actually ran (and how many scalar ops it produced). This is the evidence that a
training/inference pass genuinely computed *through* the verified GUDA logic --
not around it.
"""
from collections import Counter

COUNTS = Counter()
_ENABLED = False


def enable():
    global _ENABLED
    _ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


def reset():
    COUNTS.clear()


def bump(key: str, n: int = 1):
    if _ENABLED:
        COUNTS[key] += int(n)


def report() -> dict:
    return dict(COUNTS)


def enabled() -> bool:
    """Whether counting is on.

    Callers need this to tell "the units did not run" apart from "nobody was
    counting". A zero count from a disabled instrument is not evidence about the
    units; reporting it as one is how a silent no-op passes for a result.
    """
    return _ENABLED


def require(**minimums: int) -> dict:
    """Assert each counter reached a minimum, and return the counts.

    The counters are the evidence that compute went THROUGH the verified units
    rather than around them -- but nothing forces anyone to look. A run where the
    units never fired otherwise produces the same silence as a run where they
    fired and the result was unremarkable.

        instrument.enable()
        ... train ...
        instrument.require(**{"NeuralMul4.forward_calls": 1})

    Raises RuntimeError if counting is disabled (a zero here would be an artifact
    of the probe, not a fact about the run) or if any counter fell short.
    """
    if not _ENABLED:
        raise RuntimeError(
            "instrument.require() called while counting is disabled -- enable() "
            "first, or the zero counts say nothing about whether the units ran")
    short = {k: (COUNTS.get(k, 0), n) for k, n in minimums.items()
             if COUNTS.get(k, 0) < n}
    if short:
        raise AssertionError(
            "verified units under-invoked: "
            + ", ".join("%s=%d (want >=%d)" % (k, got, want)
                        for k, (got, want) in sorted(short.items())))
    return dict(COUNTS)
