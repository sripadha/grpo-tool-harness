"""Second version of `paired_family_bootstrap`, written from the contract in the docstring only.

It was written by another model without reading the main version (`accessdesk/eval/bootstrap.py`).
Used by tests only.

Choices where the contract was silent when this version was written (since the comparison of the two
versions, 2026-11-17, the contract says that other types than the stated ones are outside it):
- a, b may be lists or tuples; they are turned into arrays with np.asarray. Bool and integer values are
  accepted and turned into floats; strings, objects and complex numbers raise ValueError.
- NaN or inf values and values outside [0, 1] are not checked; NaN propagates into the result.
- n_draws must be an integer (Python int or numpy integer, not bool); anything else raises ValueError.
- seed is passed to np.random.default_rng as it is.
- level that is NaN fails "strictly between 0 and 1" and raises ValueError.
"""
from __future__ import annotations

import numbers

import numpy as np


def _as_values(x, name: str) -> np.ndarray:
    """x as a 1-D float array of length at least 1, or ValueError."""
    try:
        arr = np.asarray(x)
    except (ValueError, TypeError) as exc:  # ragged nested lists, for example
        raise ValueError(f"{name} cannot be read as an array: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {arr.ndim} dimensions")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must hold at least one value")
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must hold real numbers, got dtype {arr.dtype}")
    return arr.astype(np.float64)


def paired_family_bootstrap(a, b, n_draws: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Interval of the difference "b minus a" between two models, by resampling families.

    Input
        a, b      1-D arrays of the same length n, with one value per family: the family's pass rate under
                  model a and under model b. Position f is the same family in both arrays.
        n_draws   number of bootstrap draws.
        seed      seed of the random generator.
        level     the level of the interval, between 0 and 1 (0.95 means a 95% interval).

    What one draw is
        Draw n family indices with replacement (each index from 0 to n - 1, equally likely). Use the SAME
        indices `idx` for both models and take mean(b[idx]) - mean(a[idx]). This is what "paired" means: a
        family that is drawn twice counts twice for both models, and a family that is not drawn is missing
        for both. Wordings and samples inside a family are never resampled; they are already inside the
        family's value.

    The draw procedure, fixed so that two versions written to this contract give the same numbers
        rng   = np.random.default_rng(seed)                 the only source of randomness
        idx   = rng.integers(0, n, size=(n_draws, n))       one call; row d holds the n indices of draw d
        stats = for each row of idx: mean(b[row]) - mean(a[row])          n_draws numbers
        lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])     numpy's default method
                                                                                 (linear interpolation)

    Output: {"diff": .., "lo": .., "hi": ..}, three plain Python floats
        diff   mean(b) - mean(a) on the data as it is (no resampling).
        lo     the (1 - level) / 2 quantile of the n_draws differences (the 2.5th percentile for 0.95).
        hi     the 1 - (1 - level) / 2 quantile (the 97.5th percentile for 0.95).
        This is the percentile interval. Nothing else is applied (no bias correction, no t factor).

    Raises ValueError if a and b are not 1-D arrays of the same length of at least 1, if n_draws < 1, or if
    level is not strictly between 0 and 1.
    a and b hold real numbers (floats, ints or bools; a list is read as an array), n_draws and seed are
    Python ints, and level is a float. Arguments of other types are outside this contract: what the call
    does then is not defined.
    """
    a_arr = _as_values(a, "a")
    b_arr = _as_values(b, "b")
    if a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(f"a and b must have the same length, got {a_arr.shape[0]} and {b_arr.shape[0]}")
    if isinstance(n_draws, bool) or not isinstance(n_draws, numbers.Integral):
        raise ValueError(f"n_draws must be an integer, got {n_draws!r}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    if isinstance(level, bool) or not isinstance(level, numbers.Real):
        raise ValueError(f"level must be a number, got {level!r}")
    if not (0.0 < level < 1.0):  # also False for NaN
        raise ValueError(f"level must be strictly between 0 and 1, got {level}")

    n = a_arr.shape[0]
    n_draws = int(n_draws)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))

    # One draw per row, computed row by row so that each statistic is literally mean(b[row]) - mean(a[row]).
    stats = np.empty(n_draws, dtype=np.float64)
    for d in range(n_draws):
        row = idx[d]
        stats[d] = np.mean(b_arr[row]) - np.mean(a_arr[row])

    q_lo = (1 - level) / 2
    q_hi = 1 - (1 - level) / 2
    lo, hi = np.quantile(stats, [q_lo, q_hi])

    diff = np.mean(b_arr) - np.mean(a_arr)
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi)}
