"""Hand-made tests for the SECOND version of paired_family_bootstrap (accessdesk/_second/bootstrap.py).

Expected values are worked out by hand from the contract; the reasoning is in the comments.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from accessdesk._second.bootstrap import paired_family_bootstrap as pfb


def test_constant_gap_gives_a_zero_width_interval():
    # b = a + 0.25 for every family, so every draw's mean(b[idx]) - mean(a[idx]) is 0.25 (up to rounding):
    # diff = lo = hi = 0.25.
    a = np.array([0.0, 0.25, 0.5, 0.75])
    b = a + 0.25
    out = pfb(a, b, n_draws=500, seed=3)
    assert out["diff"] == pytest.approx(0.25)
    assert out["lo"] == pytest.approx(0.25)
    assert out["hi"] == pytest.approx(0.25)


def test_pairing_same_values_gives_exact_zero():
    # a == b, with very different families (0 and 1). Paired: the same idx for both, so each draw is exactly
    # 0. An unpaired version (separate indices) would give draws of -1, -0.5, 0, 0.5, 1 and a wide interval.
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 1.0])
    out = pfb(a, b, n_draws=2000, seed=0)
    assert out == {"diff": 0.0, "lo": 0.0, "hi": 0.0}


def test_single_family():
    # n = 1: every draw picks family 0, so every draw is 0.75 - 0.25 = 0.5; lo = hi = diff = 0.5.
    out = pfb([0.25], [0.75], n_draws=10, seed=1)
    assert out == {"diff": 0.5, "lo": 0.5, "hi": 0.5}


def test_known_draw_distribution_gives_the_end_points():
    # a = [0, 0], b = [0, 1]. A draw is (number of times family 1 is drawn) / 2: 0 with probability 1/4,
    # 0.5 with 1/2, 1 with 1/4. With 10,000 draws about 2,500 are 0 and 2,500 are 1, so the sorted draws at
    # positions 249 and 250 (2.5th percentile: h = 9999 * 0.025 = 249.975) are both 0 and those at
    # 9748 and 9749 (h = 9999 * 0.975 = 9749.025) are both 1. diff = 0.5 - 0 = 0.5.
    out = pfb([0.0, 0.0], [0.0, 1.0], n_draws=10_000, seed=0)
    assert out == {"diff": 0.5, "lo": 0.0, "hi": 1.0}


def test_draws_follow_the_fixed_procedure_and_linear_interpolation():
    # Two draws, level 0.5, so q = 0.25 and 0.75. With two sorted draws s0 <= s1, numpy's linear method
    # puts quantile q at h = (2 - 1) * q = q, so lo = s0 + 0.25 * (s1 - s0) and hi = s0 + 0.75 * (s1 - s0).
    # The two draws are rebuilt here from the fixed procedure with plain Python sums.
    a = [0.0, 0.5, 1.0]
    b = [1.0, 0.5, 0.0]
    seed = 11
    idx = np.random.default_rng(seed).integers(0, 3, size=(2, 3))
    draws = []
    for row in idx.tolist():
        draws.append(sum(b[i] for i in row) / 3 - sum(a[i] for i in row) / 3)
    s0, s1 = sorted(draws)
    out = pfb(a, b, n_draws=2, seed=seed, level=0.5)
    # diff: mean(b) - mean(a) = 0.5 - 0.5 = 0.
    assert out["diff"] == 0.0
    assert out["lo"] == pytest.approx(s0 + 0.25 * (s1 - s0), abs=1e-12)
    assert out["hi"] == pytest.approx(s0 + 0.75 * (s1 - s0), abs=1e-12)


def test_one_draw_gives_that_draw_for_both_ends():
    # n_draws = 1: any quantile of one number is that number. a = [0, 0, 0], b = [1, 0, 0]: the draw is
    # (number of 0s in the row) / 3, computed here from the same rng.
    row = np.random.default_rng(5).integers(0, 3, size=(1, 3))[0].tolist()
    expected = row.count(0) / 3
    out = pfb([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], n_draws=1, seed=5)
    assert out["lo"] == pytest.approx(expected)
    assert out["hi"] == pytest.approx(expected)
    assert out["diff"] == pytest.approx(1 / 3)


def test_swapping_models_flips_the_interval():
    # Same idx for both calls; each draw changes sign, so lo(b, a) = -hi(a, b) and hi(b, a) = -lo(a, b).
    rng = np.random.default_rng(42)
    a = rng.random(9)
    b = rng.random(9)
    fwd = pfb(a, b, n_draws=999, seed=7)
    back = pfb(b, a, n_draws=999, seed=7)
    assert back["diff"] == pytest.approx(-fwd["diff"])
    assert back["lo"] == pytest.approx(-fwd["hi"])
    assert back["hi"] == pytest.approx(-fwd["lo"])
    assert fwd["lo"] < fwd["hi"]


def test_wider_level_gives_a_wider_interval():
    # Same draws (same seed); the 0.99 quantiles lie outside the 0.80 quantiles.
    rng = np.random.default_rng(1)
    a = rng.random(20)
    b = rng.random(20)
    narrow = pfb(a, b, n_draws=2000, seed=2, level=0.80)
    wide = pfb(a, b, n_draws=2000, seed=2, level=0.99)
    assert wide["lo"] <= narrow["lo"] <= narrow["hi"] <= wide["hi"]
    assert wide["diff"] == narrow["diff"]


def test_same_seed_same_result_and_plain_floats():
    a = [0.1, 0.4, 0.9, 0.3]
    b = [0.2, 0.4, 0.6, 0.8]
    first = pfb(a, b, n_draws=300, seed=9)
    second = pfb(np.array(a), np.array(b), n_draws=300, seed=9)
    assert first == second
    assert set(first) == {"diff", "lo", "hi"}
    for v in first.values():
        assert type(v) is float
    # mean(b) - mean(a) = 0.5 - 0.425 = 0.075.
    assert first["diff"] == pytest.approx(0.075)


def test_different_seed_changes_the_interval():
    rng = np.random.default_rng(3)
    a = rng.random(15)
    b = rng.random(15)
    one = pfb(a, b, n_draws=200, seed=0)
    two = pfb(a, b, n_draws=200, seed=1)
    assert one["diff"] == two["diff"]
    assert (one["lo"], one["hi"]) != (two["lo"], two["hi"])


def test_bool_and_int_inputs_are_read_as_numbers():
    # Pass/fail as 0/1: a = [1, 0, 0, 1] mean 0.5, b = [1, 1, 1, 0] mean 0.75, diff = 0.25.
    out = pfb(np.array([True, False, False, True]), [1, 1, 1, 0], n_draws=50, seed=0)
    assert out["diff"] == pytest.approx(0.25)


def test_inputs_are_not_changed():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.3, 0.2, 0.1])
    a_copy, b_copy = a.copy(), b.copy()
    pfb(a, b, n_draws=20, seed=0)
    assert np.array_equal(a, a_copy) and np.array_equal(b, b_copy)


@pytest.mark.parametrize(
    "a, b",
    [
        ([], []),                                  # length 0
        ([0.1, 0.2], [0.1]),                       # different lengths
        ([[0.1, 0.2]], [[0.1, 0.2]]),              # 2-D
        (0.5, 0.5),                                # 0-D
        (["x", "y"], ["x", "y"]),                  # not numbers
    ],
)
def test_bad_arrays_raise(a, b):
    with pytest.raises(ValueError):
        pfb(a, b, n_draws=10)


@pytest.mark.parametrize("n_draws", [0, -3, 2.5, True])
def test_bad_n_draws_raise(n_draws):
    with pytest.raises(ValueError):
        pfb([0.1, 0.2], [0.3, 0.4], n_draws=n_draws)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_bad_level_raise(level):
    with pytest.raises(ValueError):
        pfb([0.1, 0.2], [0.3, 0.4], n_draws=10, level=level)


def test_numpy_integer_n_draws_is_accepted():
    out = pfb([0.1, 0.2], [0.3, 0.4], n_draws=np.int64(5), seed=0)
    assert out["diff"] == pytest.approx(0.2)
