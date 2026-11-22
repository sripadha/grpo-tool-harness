"""`group_advantages`: hand-made cases, run on the main version and on the second version.

Every expected number is worked out by hand in a comment, from the contract in the function's docstring
(accessdesk/train/grpo_math.py). Each test takes the function under test as the argument `group_advantages`:
the main version, and the second version of accessdesk/_second once it is written (until then those tests
are skipped).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import VERSIONS, function_under_test

NAN = float("nan")


@pytest.fixture(params=VERSIONS)
def group_advantages(request):
    """The function under test: the main version, then the second version."""
    return function_under_test("group_advantages", request.param)


def test_reward_minus_group_mean(group_advantages):
    # One group, rewards 1.0 and 0.0. Mean 0.5. Advantages 1.0 - 0.5 = +0.5 and 0.0 - 0.5 = -0.5.
    advantages, keep = group_advantages(np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(advantages, [[0.5, -0.5]], atol=1e-12)
    assert keep.tolist() == [True]


def test_tied_group_gives_zeros_and_is_dropped(group_advantages):
    # Three groups, each with four equal rewards: all pass, all fail, all pass with the same penalty.
    # max - min = 0 <= 0, so each is dropped: advantages 0, keep False.
    rewards = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.95, 0.95, 0.95, 0.95]])
    advantages, keep = group_advantages(rewards)
    assert advantages.tolist() == [[0.0] * 4] * 3
    assert keep.tolist() == [False, False, False]


def test_small_reward_differences_are_kept_at_tol_0_and_dropped_at_tol_01(group_advantages):
    # Rewards 1, 1, 0.9, 0.9. Mean 0.95. Advantages +0.05, +0.05, -0.05, -0.05.
    rewards = np.array([[1.0, 1.0, 0.9, 0.9]])
    advantages, keep = group_advantages(rewards, tie_tol=0.0)
    np.testing.assert_allclose(advantages, [[0.05, 0.05, -0.05, -0.05]], atol=1e-12)
    assert keep.tolist() == [True]
    # max - min = 0.1 (0.09999999999999998 in floating point), which is <= 0.1: dropped.
    advantages, keep = group_advantages(rewards, tie_tol=0.1)
    assert advantages.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert keep.tolist() == [False]


def test_default_tie_tol_is_zero(group_advantages):
    # Rewards 1.0 and 0.99 differ by 0.01 > 0: kept by default. Mean 0.995, advantages +0.005 and -0.005.
    advantages, keep = group_advantages(np.array([[1.0, 0.99]]))
    np.testing.assert_allclose(advantages, [[0.005, -0.005]], atol=1e-12)
    assert keep.tolist() == [True]


def test_no_division_by_the_standard_deviation(group_advantages):
    # Rewards 1, 0, 0, 0. Mean 0.25. Advantages +0.75, -0.25, -0.25, -0.25.
    # (Divided by the standard deviation 0.433 they would be +1.73, -0.58, -0.58, -0.58: wrong.)
    advantages, keep = group_advantages(np.array([[1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(advantages, [[0.75, -0.25, -0.25, -0.25]], atol=1e-12)
    # Rewards 0.2 and 0.1. Mean 0.15. Advantages +0.05 and -0.05: the size follows the rewards.
    advantages, keep = group_advantages(np.array([[0.2, 0.1]]))
    np.testing.assert_allclose(advantages, [[0.05, -0.05]], atol=1e-12)
    assert keep.tolist() == [True]


def test_each_group_uses_its_own_mean(group_advantages):
    # Group 0: 1, 0 -> mean 0.5 -> +0.5, -0.5, kept. Group 1: 1, 1 -> a tie -> zeros, dropped.
    # Group 2: 0, 0.95 -> mean 0.475 -> -0.475, +0.475, kept.
    # (With one mean over all six rewards, 0.6583, every number would be different.)
    advantages, keep = group_advantages(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.95]]))
    np.testing.assert_allclose(advantages, [[0.5, -0.5], [0.0, 0.0], [-0.475, 0.475]], atol=1e-12)
    assert keep.tolist() == [True, False, True]


def test_nan_sample_is_left_out_of_the_mean_and_gets_zero(group_advantages):
    # Rewards 1, NaN, 0, 0: three of four samples are valid, so the group stays.
    # Mean of the valid rewards = (1 + 0 + 0) / 3 = 1/3.
    # Advantages: 1 - 1/3 = +2/3; the NaN sample 0; 0 - 1/3 = -1/3 twice.
    # (With NaN counted as a reward of 0 the mean would be 1/4 and the first advantage 0.75: wrong.)
    advantages, keep = group_advantages(np.array([[1.0, NAN, 0.0, 0.0]]))
    np.testing.assert_allclose(advantages, [[2 / 3, 0.0, -1 / 3, -1 / 3]], atol=1e-12)
    assert keep.tolist() == [True]


def test_exactly_half_valid_is_kept_and_fewer_than_half_is_dropped(group_advantages):
    rewards = np.array([
        [1.0, 0.0, NAN, NAN],   # 2 of 4 valid: exactly half, kept. Mean 0.5 -> +0.5, -0.5, 0, 0.
        [1.0, NAN, NAN, NAN],   # 1 of 4 valid: fewer than half, dropped.
        [NAN, NAN, NAN, NAN],   # nothing valid: dropped.
        [1.0, 0.0, 0.0, NAN],   # 3 of 4 valid: kept. Mean 1/3 -> +2/3, -1/3, -1/3, 0.
    ])
    advantages, keep = group_advantages(rewards)
    expected = [[0.5, -0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [2 / 3, -1 / 3, -1 / 3, 0.0]]
    np.testing.assert_allclose(advantages, expected, atol=1e-12)
    assert keep.tolist() == [True, False, False, True]


def test_five_samples_need_three_valid(group_advantages):
    # Group size 5: half is 2.5. Two valid samples are fewer than half: dropped. Three are enough.
    rewards = np.array([[1.0, 0.0, NAN, NAN, NAN], [1.0, 0.0, 0.0, NAN, NAN]])
    advantages, keep = group_advantages(rewards)
    # Second group: mean 1/3 -> +2/3, -1/3, -1/3, 0, 0.
    np.testing.assert_allclose(advantages, [[0.0] * 5, [2 / 3, -1 / 3, -1 / 3, 0.0, 0.0]], atol=1e-12)
    assert keep.tolist() == [False, True]


def test_tie_among_the_valid_samples_is_dropped(group_advantages):
    # Rewards 1, NaN, 1, 1: the three valid rewards are equal, so this is a tie.
    advantages, keep = group_advantages(np.array([[1.0, NAN, 1.0, 1.0]]))
    assert advantages.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert keep.tolist() == [False]


def test_output_never_contains_nan(group_advantages):
    rewards = np.array([[1.0, NAN, 0.0, 0.0], [NAN, NAN, NAN, 1.0], [NAN, NAN, NAN, NAN]])
    advantages, keep = group_advantages(rewards)
    assert not np.isnan(advantages).any()


def test_advantages_of_a_kept_group_add_up_to_zero(group_advantages):
    # Eight samples with mixed rewards and one lost sample. Sum of (reward - mean) over the valid ones is 0.
    rewards = np.array([[1.0, 0.95, 0.0, 0.88, 1.0, 0.0, NAN, 0.81],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    advantages, keep = group_advantages(rewards)
    assert keep.tolist() == [True, True]
    np.testing.assert_allclose(advantages.sum(axis=1), [0.0, 0.0], atol=1e-12)
    # Second group: mean 1/8. The one pass gets +7/8, each fail -1/8.
    np.testing.assert_allclose(advantages[1], [-0.125] * 7 + [0.875], atol=1e-12)


def test_shapes_types_and_input_unchanged(group_advantages):
    rewards = np.array([[1.0, 0.0, NAN], [0.0, 0.0, 0.0]])
    before = rewards.copy()
    advantages, keep = group_advantages(rewards)
    assert advantages.shape == (2, 3)
    assert keep.shape == (2,)
    assert keep.dtype == bool
    np.testing.assert_array_equal(rewards, before)     # NaN equals NaN in this comparison


def test_rewards_must_have_two_dimensions(group_advantages):
    with pytest.raises(ValueError):
        group_advantages(np.array([1.0, 0.0]))


def test_rewards_need_at_least_one_group_and_one_sample(group_advantages):
    with pytest.raises(ValueError):
        group_advantages(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        group_advantages(np.zeros((3, 0)))
    with pytest.raises(ValueError):
        group_advantages(np.zeros((2, 2, 2)))


def test_eight_samples_with_penalties(group_advantages):
    # A group as GRPO sees it: 8 samples, rewards in hundredths / 100.
    # 1.00, 0.95, 0.00, 0.00, 1.00, 0.88, 0.00, 0.00. Sum = 3.83, mean = 0.47875.
    rewards = np.array([[1.00, 0.95, 0.00, 0.00, 1.00, 0.88, 0.00, 0.00]])
    advantages, keep = group_advantages(rewards)
    expected = [0.52125, 0.47125, -0.47875, -0.47875, 0.52125, 0.40125, -0.47875, -0.47875]
    np.testing.assert_allclose(advantages, [expected], atol=1e-12)
    assert keep.tolist() == [True]


def test_pass_only_group_with_different_penalties_is_kept_at_tol_0_only(group_advantages):
    # Every sample passed, with rewards 1.00 and 0.95: equal for pass-or-fail, not equal as rewards.
    rewards = np.array([[1.00, 0.95, 1.00, 1.00]])
    advantages, keep = group_advantages(rewards, tie_tol=0.0)
    # Mean 0.9875: +0.0125, -0.0375, +0.0125, +0.0125.
    np.testing.assert_allclose(advantages, [[0.0125, -0.0375, 0.0125, 0.0125]], atol=1e-12)
    assert keep.tolist() == [True]
    # tie_tol 0.25 is above the largest gap between two passes (1.00 - 0.77) and far below a pass against a
    # fail, so it drops this group and keeps a group with a pass and a fail.
    rewards = np.array([[1.00, 0.95, 1.00, 1.00], [1.00, 0.0, 0.77, 0.0]])
    advantages, keep = group_advantages(rewards, tie_tol=0.25)
    assert keep.tolist() == [False, True]
    assert advantages[0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_negative_rewards_of_failing_episodes(group_advantages):
    # With cfg.penalise_malformed_on_fail on, a fail is 0, -0.05, -0.10 or -0.15.
    # Group 0: every sample failed, with 0, 1, 3 and 0 malformed calls: 0, -0.05, -0.15, 0.
    #   The rewards differ, so the group is kept. Mean = -0.20 / 4 = -0.05.
    #   Advantages: +0.05, 0.00, -0.10, +0.05. The fail without a malformed call is above the mean.
    # Group 1: one pass (1.00) and three fails (-0.15, 0, -0.05). Mean = 0.80 / 4 = 0.20.
    #   Advantages: +0.80, -0.35, -0.20, -0.25.
    # Group 2: every sample failed with the same penalty: a tie, dropped.
    rewards = np.array([[0.0, -0.05, -0.15, 0.0], [1.0, -0.15, 0.0, -0.05], [-0.05, -0.05, -0.05, -0.05]])
    advantages, keep = group_advantages(rewards)
    expected = [[0.05, 0.0, -0.10, 0.05], [0.80, -0.35, -0.20, -0.25], [0.0, 0.0, 0.0, 0.0]]
    np.testing.assert_allclose(advantages, expected, atol=1e-12)
    assert keep.tolist() == [True, True, False]
    # tie_tol 0.2 is above the largest gap between two fails (0.15): group 0 becomes a tie, group 1 stays.
    advantages, keep = group_advantages(rewards, tie_tol=0.2)
    assert keep.tolist() == [False, True, False]


# --- a case that the comparison of the two versions decided (2026-11-17) ------------------------------------
INF = float("inf")


@pytest.mark.parametrize("rewards", [
    [[INF, 0.0]],                       # the main version gave [[nan, -inf]] with keep True
    [[-INF, 0.0, 1.0]],                 # the main version gave [[nan, inf, inf]]
    [[INF, INF]],                       # inf - inf is NaN, which is not <= 0: kept, with NaN advantages
    [[INF, NAN]],                       # exactly half valid
    [[INF, NAN, NAN]],                  # a group that rule 2 drops: the infinite reward is still refused
    [[1.0, 0.0], [0.5, -INF]],          # one good group and one with an infinite reward
], ids=["inf", "minus_inf", "two_inf", "inf_and_nan", "inf_in_a_dropped_group", "inf_in_the_second_group"])
def test_an_infinite_reward_is_refused(group_advantages, rewards):
    # The contract says that the advantages never contain NaN, and it names NaN as the only mark of a lost
    # sample. The main version let an infinite reward through and returned NaN advantages (inf minus a mean
    # of inf). The second version raised ValueError. Decided: ValueError, in whatever group the value is.
    with pytest.raises(ValueError):
        group_advantages(np.array(rewards))
    with pytest.raises(ValueError):
        group_advantages(np.array(rewards), tie_tol=INF)     # also when the tolerance would drop the group


def test_very_large_finite_rewards_are_not_refused(group_advantages):
    # Only an infinite value is refused. 1e300 and -1e300: mean 0, advantages +1e300 and -1e300.
    advantages, keep = group_advantages(np.array([[1e300, -1e300]]))
    assert advantages.tolist() == [[1e300, -1e300]] and keep.tolist() == [True]
