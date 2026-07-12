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
