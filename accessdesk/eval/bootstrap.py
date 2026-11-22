"""The paired bootstrap by family: the interval of every comparison in the results table.

This is the main version of `paired_family_bootstrap`. The rest of the code reaches it through
`accessdesk.impl.get("paired_family_bootstrap")`. An independent second version, written from the docstring
below only, is in `accessdesk/_second/bootstrap.py`; tests compare the two.
Tests: `.venv/bin/python -m pytest tests/core/test_paired_family_bootstrap.py
tests/core/test_differential_bootstrap.py`
"""
from __future__ import annotations

import numpy as np


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
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError(f"a and b must be 1-D arrays of one length, at least 1; got {a.shape} and {b.shape}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level}")
    n = a.size

    # All draws at once: a table of family indices with one row per draw and n indices in each row.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))

    # a[idx] has the shape of idx and holds a's value for each drawn family; the mean of a row is the mean of
    # one draw. The same table idx is used for b: that is the pairing.
    stats = b[idx].mean(axis=1) - a[idx].mean(axis=1)

    # The percentile interval: cut the same share off both ends of the n_draws differences.
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [tail, 1.0 - tail])

    return {"diff": float(b.mean() - a.mean()), "lo": float(lo), "hi": float(hi)}
