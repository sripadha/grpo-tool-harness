"""`paired_family_bootstrap`: hand-made cases, coverage, and why the family is the unit. Run on the main
version and on the second version.

Every expected number is worked out by hand in a comment, from the contract in the function's docstring
(accessdesk/eval/bootstrap.py). Each test takes the function under test as the argument
`paired_family_bootstrap`: the main version, and the second version of accessdesk/_second once it is written
(until then those tests are skipped).

A bootstrap is random, so most cases are chosen such that the answer does not depend on the seed: either every
draw gives the same number, or the limits are far from any value that a seed could move them to. The contract
also fixes the draw procedure; one test repeats that procedure step by step and asks for the same numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import VERSIONS, function_under_test


@pytest.fixture(params=VERSIONS)
def paired_family_bootstrap(request):
    """The function under test: the main version, then the second version."""
    return function_under_test("paired_family_bootstrap", request.param)


def test_diff_is_mean_b_minus_mean_a(paired_family_bootstrap):
    # mean(a) = (0 + 1) / 2 = 0.5, mean(b) = (1 + 1) / 2 = 1.0.
    # diff = 1.0 - 0.5 = +0.5 (b minus a, not a minus b).
    result = paired_family_bootstrap(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 200, 0, 0.95)
    assert result["diff"] == pytest.approx(0.5, abs=1e-12)


def test_result_has_three_floats_in_order(paired_family_bootstrap):
    a = np.array([0.0, 0.5, 1.0, 0.25])
    b = np.array([0.5, 0.5, 0.75, 1.0])
    result = paired_family_bootstrap(a, b, 500, 3, 0.95)
    assert sorted(result) == ["diff", "hi", "lo"]
    assert all(isinstance(result[key], float) for key in result)     # not arrays: the table is saved as JSON
    assert result["lo"] <= result["diff"] <= result["hi"]


def test_same_gain_in_every_family_gives_an_interval_of_width_zero(paired_family_bootstrap):
    # The families are very different from each other (0.0 to 0.75), but model b is 0.25 above model a in
    # every family. With the same indices for both models, every draw is exactly 0.25: lo = hi = 0.25.
    # (Resampling a and b with different indices would give a wide interval here: that is the pairing bug.)
    a = np.array([0.0, 0.25, 0.5, 0.75, 0.0, 0.75])
    b = a + 0.25
    result = paired_family_bootstrap(a, b, 1000, 0, 0.95)
    assert result["diff"] == pytest.approx(0.25, abs=1e-12)
    assert result["lo"] == pytest.approx(0.25, abs=1e-12)
    assert result["hi"] == pytest.approx(0.25, abs=1e-12)


def test_two_families_the_draws_are_0_half_or_1(paired_family_bootstrap):
    # Per-family differences b - a = 0 and 1. A draw takes two families with replacement:
    # (0, 0) -> 0, (0, 1) or (1, 0) -> 0.5, (1, 1) -> 1, with chances 1/4, 1/2, 1/4.
    a = np.array([0.5, 0.0])
    b = np.array([0.5, 1.0])
    # Level 0.95: the 2.5th percentile is 0 (a quarter of the draws are 0) and the 97.5th is 1.
    result = paired_family_bootstrap(a, b, 4000, 0, 0.95)
    assert result["diff"] == pytest.approx(0.5, abs=1e-12)
    assert result["lo"] == pytest.approx(0.0, abs=1e-12)
    assert result["hi"] == pytest.approx(1.0, abs=1e-12)
    # Level 0.4: the 30th and the 70th percentile both fall on 0.5 (the draws from the 25th to the 75th
    # percentile are 0.5).
    result = paired_family_bootstrap(a, b, 4000, 0, 0.4)
    assert result["lo"] == pytest.approx(0.5, abs=1e-12)
    assert result["hi"] == pytest.approx(0.5, abs=1e-12)


def test_hundred_families_half_improve(paired_family_bootstrap):
    # 100 families. Model a fails all. Model b passes 50 of them fully.
    # Per-family difference: 50 ones, 50 zeros.
    # A draw is (number of ones among 100 draws) / 100, and that number is Binomial(100, 0.5):
    # P(X <= 38) = 0.0105, P(X <= 39) = 0.0176, P(X <= 40) = 0.0284, P(X <= 41) = 0.0443.
    # So the 2.5th percentile is 40 / 100 = 0.40, and by symmetry the 97.5th is 0.60.
    # (Normal approximation: 0.5 -/+ 1.96 * sqrt(0.25 / 100) = 0.402 and 0.598.)
    # With 10,000 draws the limit can land one step beside it, so 0.39 to 0.41 and 0.59 to 0.61 are accepted.
    a = np.zeros(100)
    b = np.array([1.0] * 50 + [0.0] * 50)
    result = paired_family_bootstrap(a, b, 10_000, 0, 0.95)
    assert result["diff"] == pytest.approx(0.5, abs=1e-12)
    assert result["lo"] == pytest.approx(0.40, abs=0.0101)
    assert result["hi"] == pytest.approx(0.60, abs=0.0101)


def test_sign_follows_b_minus_a(paired_family_bootstrap):
    # Model b is worse than model a in every family, by 0.5 or by 1.0: every draw is negative.
    a = np.array([1.0, 1.0, 0.5, 1.0])
    b = np.array([0.0, 0.5, 0.0, 0.0])
    result = paired_family_bootstrap(a, b, 1000, 0, 0.95)
    # diff = mean(b) - mean(a) = 0.125 - 0.875 = -0.75.
    assert result["diff"] == pytest.approx(-0.75, abs=1e-12)
    assert -1.0 <= result["lo"] <= result["hi"] <= -0.5


def test_same_seed_same_result_and_another_seed_another_draw(paired_family_bootstrap):
    a = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.0, 0.125, 0.875, 0.375])
    b = np.array([0.5, 0.25, 1.0, 0.5, 1.0, 0.625, 0.0, 0.75, 0.125, 1.0, 0.25])
    first = paired_family_bootstrap(a, b, 2000, 7, 0.95)
    again = paired_family_bootstrap(a, b, 2000, 7, 0.95)
    assert first == again
    # 20 other seeds: the point estimate never moves; the limits come from other draws, so not all 20 seeds
    # can give the same pair of limits as seed 7.
    others = [paired_family_bootstrap(a, b, 2000, seed, 0.95) for seed in range(100, 120)]
    assert all(other["diff"] == first["diff"] for other in others)
    assert any((other["lo"], other["hi"]) != (first["lo"], first["hi"]) for other in others)


def test_higher_level_gives_a_wider_interval(paired_family_bootstrap):
    a = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.0, 0.125, 0.875, 0.375])
    b = np.array([0.5, 0.25, 1.0, 0.5, 1.0, 0.625, 0.0, 0.75, 0.125, 1.0, 0.25])
    narrow = paired_family_bootstrap(a, b, 4000, 0, 0.5)
    wide = paired_family_bootstrap(a, b, 4000, 0, 0.99)
    assert wide["lo"] < narrow["lo"] < narrow["hi"] < wide["hi"]


def test_inputs_are_checked(paired_family_bootstrap):
    two, three = np.array([0.0, 1.0]), np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        paired_family_bootstrap(two, three, 100, 0, 0.95)                    # the lengths differ
    with pytest.raises(ValueError):
        paired_family_bootstrap(two, two, 100, 0, 95)                        # the level is 0.95, not 95
    with pytest.raises(ValueError):
        paired_family_bootstrap(two, two, 100, 0, 1.0)                       # strictly below 1
    with pytest.raises(ValueError):
        paired_family_bootstrap(two, two, 100, 0, 0.0)                       # strictly above 0
    with pytest.raises(ValueError):
        paired_family_bootstrap(np.array([]), np.array([]), 100, 0, 0.95)    # no family


def test_n_draws_must_be_at_least_1(paired_family_bootstrap):
    two = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        paired_family_bootstrap(two, two, 0, 0, 0.95)
    with pytest.raises(ValueError):
        paired_family_bootstrap(np.zeros((2, 2)), np.zeros((2, 2)), 100, 0, 0.95)      # not 1-D
    one_draw = paired_family_bootstrap(two, np.array([1.0, 1.0]), 1, 0, 0.95)
    assert one_draw["lo"] == one_draw["hi"]                         # one draw: both limits are that draw


def test_the_draws_follow_the_procedure_of_the_contract(paired_family_bootstrap):
    # The contract fixes the draws: one generator from the seed, one call for all indices (one row per draw),
    # the statistic of each row, and numpy's default quantile. Written out here as a plain loop.
    a = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25])
    b = np.array([0.5, 0.25, 1.0, 0.5, 1.0, 0.625, 0.0])
    rng = np.random.default_rng(11)
    idx = rng.integers(0, 7, size=(300, 7))
    draws = sorted(float(np.mean(b[row]) - np.mean(a[row])) for row in idx)
    result = paired_family_bootstrap(a, b, 300, 11, 0.9)
    # 5th and 95th percentile of 300 sorted draws, linear interpolation: positions 0.05 * 299 = 14.95 and
    # 0.95 * 299 = 284.05.
    lo = draws[14] + 0.95 * (draws[15] - draws[14])
    hi = draws[284] + 0.05 * (draws[285] - draws[284])
    assert result["lo"] == pytest.approx(lo, abs=1e-12)
    assert result["hi"] == pytest.approx(hi, abs=1e-12)
    assert result["diff"] == pytest.approx(float(np.mean(b) - np.mean(a)), abs=1e-12)
    # Another seed gives other limits here, so the test above does depend on the seed's draws.
    other = paired_family_bootstrap(a, b, 300, 12, 0.9)
    assert (other["lo"], other["hi"]) != (result["lo"], result["hi"])


def test_the_result_holds_python_floats(paired_family_bootstrap):
    result = paired_family_bootstrap(np.array([0.0, 0.5, 1.0]), np.array([0.5, 0.5, 0.75]), 50, 1, 0.9)
    assert [type(result[key]) for key in ("diff", "lo", "hi")] == [float, float, float]     # not numpy types


def test_lists_are_accepted_and_the_inputs_are_not_changed(paired_family_bootstrap):
    a, b = [0.0, 0.25, 1.0, 0.5], [0.5, 0.25, 1.0, 0.75]
    from_lists = paired_family_bootstrap(a, b, 200, 4, 0.95)
    assert (a, b) == ([0.0, 0.25, 1.0, 0.5], [0.5, 0.25, 1.0, 0.75])
    assert from_lists == paired_family_bootstrap(np.array(a), np.array(b), 200, 4, 0.95)


# ---------------------------------------------------------------------------
# Coverage: on data with a known true difference, the interval contains it about as often as it says
# ---------------------------------------------------------------------------
TRUE_DIFF = 0.10


def coverage(bootstrap_function, level: float, n_repeats: int = 500, n_families: int = 40) -> float:
    """Share of repeated data sets whose interval contains the true difference.

    Each family has its own difficulty (shared by both models, standard deviation 0.3) and its own gain
    (mean TRUE_DIFF, standard deviation 0.15). The shared difficulty is much larger than the gain, as in
    the real task, where some families are easy for every model."""
    rng = np.random.default_rng(20261116)
    hits = 0
    for repeat in range(n_repeats):
        difficulty = rng.normal(0.0, 0.3, n_families)
        a = difficulty + rng.normal(0.0, 0.05, n_families)
        b = difficulty + TRUE_DIFF + rng.normal(0.0, 0.15, n_families)
        result = bootstrap_function(a, b, 800, repeat, level)
        hits += result["lo"] <= TRUE_DIFF <= result["hi"]
    return hits / n_repeats


def unpaired_bootstrap(a, b, n_draws, seed, level):
    """The pairing bug, for comparison: different families are drawn for the two models."""
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, a.size, size=(n_draws, a.size))
    idx_b = rng.integers(0, b.size, size=(n_draws, b.size))
    draws = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return {"diff": float(b.mean() - a.mean()), "lo": float(lo), "hi": float(hi)}


def test_coverage_is_about_the_stated_level(paired_family_bootstrap):
    # 500 repeats: the share found has a standard error of about 0.01 at 95% and 0.018 at 80%.
    # The percentile interval is known to be slightly too narrow at 40 families (about 94% for a nominal 95%).
    assert 0.90 <= coverage(paired_family_bootstrap, 0.95) <= 0.975
    assert 0.72 <= coverage(paired_family_bootstrap, 0.80) <= 0.86


def test_the_coverage_test_can_tell_an_unpaired_bootstrap():
    # Without pairing, the shared difficulty of the families enters the interval: it becomes far too wide and
    # contains the true difference (almost) every time. The coverage test above would reject it.
    assert coverage(unpaired_bootstrap, 0.80, n_repeats=200) > 0.97


# ---------------------------------------------------------------------------
# The family is the unit: resampling episodes gives an interval that is too narrow
# ---------------------------------------------------------------------------
def episode_bootstrap_width(passes_a: np.ndarray, passes_b: np.ndarray, n_draws: int, seed: int) -> float:
    """Width of a 95% interval that resamples single episodes (the wrong unit), each model on its own."""
    rng = np.random.default_rng(seed)
    flat_a, flat_b = passes_a.ravel(), passes_b.ravel()
    idx_a = rng.integers(0, flat_a.size, size=(n_draws, flat_a.size))
    idx_b = rng.integers(0, flat_b.size, size=(n_draws, flat_b.size))
    draws = flat_b[idx_b].mean(axis=1) - flat_a[idx_a].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(hi - lo)


def test_family_interval_is_clearly_wider_than_an_episode_interval_when_the_gain_differs_by_family(
        paired_family_bootstrap):
    # 40 families, 32 episodes each (4 wordings x 8 samples). Model a passes 10 of 32 in every family.
    # Model b passes 29 of 32 in the first 20 families (gain 19/32 = 0.594) and 10 of 32 in the other 20
    # (gain 0).
    episodes = np.arange(32)
    passes_a = np.tile((episodes < 10).astype(float), (40, 1))
    passes_b = np.vstack([np.tile((episodes < 29).astype(float), (20, 1)),
                          np.tile((episodes < 10).astype(float), (20, 1))])
    rate_a, rate_b = passes_a.mean(axis=1), passes_b.mean(axis=1)

    by_family = paired_family_bootstrap(rate_a, rate_b, 2000, 0, 0.95)
    # Same estimate either way: mean gain = (20 * 19/32 + 20 * 0) / 40 = 19/64 = 0.296875.
    assert by_family["diff"] == pytest.approx(19 / 64, abs=1e-12)
    assert float(passes_b.mean() - passes_a.mean()) == pytest.approx(19 / 64, abs=1e-12)

    # By family: the per-family gains are 0.594 or 0, standard deviation 0.297; standard error
    # 0.297 / sqrt(40) = 0.047; width of a 95% interval about 3.92 * 0.047 = 0.184.
    family_width = by_family["hi"] - by_family["lo"]
    assert family_width == pytest.approx(0.184, abs=0.02)
    # By episode: pooled pass rates 0.3125 and 0.609 over 1280 episodes each; standard error
    # sqrt((0.3125 * 0.6875 + 0.609 * 0.391) / 1280) = 0.0188; width about 3.92 * 0.0188 = 0.074.
    episode_width = episode_bootstrap_width(passes_a, passes_b, 400, 0)
    assert episode_width == pytest.approx(0.074, abs=0.01)
    assert family_width > 2.0 * episode_width


def test_family_and_episode_intervals_agree_when_every_family_has_the_same_gain(paired_family_bootstrap):
    # The control for the test above: when all families are alike (10 of 32 against 20 of 32 in every
    # family) the family interval is not wider; it has width 0, because there is nothing between families.
    # The episode interval still shows sampling noise. So the extra width above comes from the families.
    episodes = np.arange(32)
    passes_a = np.tile((episodes < 10).astype(float), (40, 1))
    passes_b = np.tile((episodes < 20).astype(float), (40, 1))
    by_family = paired_family_bootstrap(passes_a.mean(axis=1), passes_b.mean(axis=1), 2000, 0, 0.95)
    assert by_family["hi"] - by_family["lo"] == pytest.approx(0.0, abs=1e-12)
    assert episode_bootstrap_width(passes_a, passes_b, 400, 0) > 0.05
