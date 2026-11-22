"""Hand-made cases for the SECOND versions of group_advantages and policy_loss.

Every expected value is worked out by hand from the contract, in the comments. These tests do not use the
main version.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from accessdesk._second.grpo_math import group_advantages, policy_loss

NAN = float("nan")


# ---------------------------------------------------------------------------
# group_advantages
# ---------------------------------------------------------------------------

def test_one_mixed_group():
    # valid rewards 1.0, 0.0, 0.0, 1.0 -> mean 0.5 -> advantages +0.5, -0.5, -0.5, +0.5
    adv, keep = group_advantages(np.array([[1.0, 0.0, 0.0, 1.0]]))
    assert keep.tolist() == [True]
    assert adv.tolist() == [[0.5, -0.5, -0.5, 0.5]]


def test_uneven_group_sums_to_zero():
    # 0.9, 0.0, 0.0: mean 0.3 -> 0.6, -0.3, -0.3 (sum 0)
    adv, keep = group_advantages(np.array([[0.9, 0.0, 0.0]]))
    assert keep.tolist() == [True]
    assert np.allclose(adv, [[0.6, -0.3, -0.3]], atol=1e-12)
    assert abs(adv.sum()) < 1e-12


def test_no_division_by_std():
    # 0.2 and 0.0: mean 0.1 -> +0.1, -0.1. With std scaling it would be +1, -1.
    adv, _ = group_advantages(np.array([[0.2, 0.0]]))
    assert np.allclose(adv, [[0.1, -0.1]], atol=1e-12)


def test_all_equal_group_is_dropped():
    # every valid reward is 1.0: max - min = 0 <= 0 -> dropped, all zeros
    adv, keep = group_advantages(np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]))
    assert keep.tolist() == [False, True]
    assert adv[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    # second row: mean 0.75 -> 0.25, -0.75, 0.25, 0.25
    assert adv[1].tolist() == [0.25, -0.75, 0.25, 0.25]


def test_nan_sample_gets_zero_and_mean_uses_valid_only():
    # valid: 1.0, 0.0, 0.5 -> mean 0.5 -> 0.5, -0.5, 0.0; the NaN sample gets 0.0
    rewards = np.array([[1.0, NAN, 0.0, 0.5]])
    adv, keep = group_advantages(rewards)
    assert keep.tolist() == [True]
    assert adv.tolist() == [[0.5, 0.0, -0.5, 0.0]]
    assert not np.isnan(adv).any()


def test_exactly_half_valid_stays():
    # 2 of 4 valid (exactly half): stays. Valid 1.0, 0.0 -> mean 0.5 -> 0.5, -0.5
    adv, keep = group_advantages(np.array([[1.0, NAN, NAN, 0.0]]))
    assert keep.tolist() == [True]
    assert adv.tolist() == [[0.5, 0.0, 0.0, -0.5]]


def test_fewer_than_half_valid_is_dropped():
    # 1 of 4 valid: fewer than half -> dropped, even though the other rule would not drop it
    adv, keep = group_advantages(np.array([[1.0, NAN, NAN, NAN]]))
    assert keep.tolist() == [False]
    assert adv.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    # odd size: 3 samples, 1 valid -> 1 < 1.5 -> dropped
    adv, keep = group_advantages(np.array([[1.0, 0.0, NAN], [1.0, NAN, NAN]]), tie_tol=-1.0)
    assert keep.tolist() == [True, False]
    # row 0: 1.0, 0.0 -> mean 0.5 -> 0.5, -0.5, 0.0
    assert adv[0].tolist() == [0.5, -0.5, 0.0]


def test_all_nan_group_is_dropped():
    adv, keep = group_advantages(np.array([[NAN, NAN], [0.0, 1.0]]))
    assert keep.tolist() == [False, True]
    assert adv[0].tolist() == [0.0, 0.0]
    assert adv[1].tolist() == [-0.5, 0.5]


def test_tie_tol_boundary():
    # 0.9 and 0.85: distance 0.05 (in floats 0.050000000000000044)
    rewards = np.array([[0.9, 0.85]])
    _, keep = group_advantages(rewards, tie_tol=0.04)       # 0.05 > 0.04 -> kept
    assert keep.tolist() == [True]
    _, keep = group_advantages(rewards, tie_tol=0.055)      # 0.05 <= 0.055 -> dropped
    assert keep.tolist() == [False]
    # the contract's own example: 1.0 - 0.9 = 0.09999999999999998 <= 0.1 -> dropped
    _, keep = group_advantages(np.array([[1.0, 0.9]]), tie_tol=0.1)
    assert keep.tolist() == [False]


def test_tie_tol_uses_valid_rewards_only():
    # valid: 1.0, 0.98, 1.0 (spread 0.02); NaN ignored. tie_tol 0.03 -> dropped; 0.01 -> kept
    rewards = np.array([[1.0, 0.98, NAN, 1.0]])
    _, keep = group_advantages(rewards, tie_tol=0.03)
    assert keep.tolist() == [False]
    adv, keep = group_advantages(rewards, tie_tol=0.01)
    assert keep.tolist() == [True]
    # mean (1.0 + 0.98 + 1.0) / 3 = 0.99333.. -> 0.00667, -0.01333, 0, 0.00667
    assert np.allclose(adv, [[0.02 / 3, -0.04 / 3, 0.0, 0.02 / 3]], atol=1e-12)


def test_negative_rewards():
    # penalised fail -0.15 and a pass 0.85: mean 0.35 -> -0.5, 0.5
    adv, keep = group_advantages(np.array([[-0.15, 0.85]]))
    assert keep.tolist() == [True]
    assert np.allclose(adv, [[-0.5, 0.5]], atol=1e-12)


def test_output_types_and_input_unchanged():
    rewards = np.array([[1.0, NAN, 0.0, 0.0]])
    before = rewards.copy()
    adv, keep = group_advantages(rewards)
    assert np.array_equal(rewards, before, equal_nan=True)
    assert adv.shape == rewards.shape and adv.dtype == np.float64
    assert keep.shape == (1,) and keep.dtype == np.bool_


@pytest.mark.parametrize("bad", [np.array([1.0, 0.0]), np.zeros((1, 2, 2)), np.zeros((0, 4)), np.zeros((3, 0))])
def test_group_advantages_bad_shapes(bad):
    with pytest.raises(ValueError):
        group_advantages(bad)


def test_group_advantages_infinite_reward_raises():
    # my choice where the contract is silent (see the module docstring)
    with pytest.raises(ValueError):
        group_advantages(np.array([[math.inf, 0.0]]))


# ---------------------------------------------------------------------------
# policy_loss
# ---------------------------------------------------------------------------

def test_loss_by_hand():
    # episode 0: mask 1,1,0 on logprobs -1, -2, -5 -> -3; advantage 0.5 -> -1.5
    # episode 1: mask 0,1,1 on logprobs -4, -1, -1 -> -2; advantage -0.5 -> +1.0
    # sum = -0.5; loss = -(1/2) * -0.5 = 0.25
    logprobs = np.array([[-1.0, -2.0, -5.0], [-4.0, -1.0, -1.0]])
    mask = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    advantages = np.array([0.5, -0.5])
    assert policy_loss(logprobs, mask, advantages, 2.0) == pytest.approx(0.25, abs=1e-15)


def test_loss_sign():
    # positive advantage: higher logprob -> lower loss
    mask = np.ones((1, 2))
    adv = np.array([1.0])
    low = policy_loss(np.array([[-3.0, -3.0]]), mask, adv, 1.0)    # -(1 * -6) = 6
    high = policy_loss(np.array([[-1.0, -1.0]]), mask, adv, 1.0)   # -(1 * -2) = 2
    assert low == 6.0 and high == 2.0 and high < low
    # negative advantage: the other way
    assert policy_loss(np.array([[-1.0, -1.0]]), mask, -adv, 1.0) == -2.0


def test_zero_advantages_give_zero():
    loss = policy_loss(np.array([[-1.0, -7.0]]), np.ones((1, 2)), np.zeros(1), 3.0)
    assert loss == 0.0


def test_masked_token_does_not_matter():
    mask = np.array([[1.0, 0.0]])
    a = policy_loss(np.array([[-1.0, -0.1]]), mask, np.array([2.0]), 4.0)
    b = policy_loss(np.array([[-1.0, -900.0]]), mask, np.array([2.0]), 4.0)
    # -(1/4) * 2 * -1 = 0.5
    assert a == b == 0.5


def test_sum_not_mean():
    # adding an episode with advantage 0 or a masked token changes nothing
    base = policy_loss(np.array([[-2.0]]), np.ones((1, 1)), np.array([1.0]), 1.0)   # 2.0
    more_eps = policy_loss(np.array([[-2.0], [-9.0]]), np.ones((2, 1)), np.array([1.0, 0.0]), 1.0)
    more_tok = policy_loss(np.array([[-2.0, -9.0]]), np.array([[1.0, 0.0]]), np.array([1.0]), 1.0)
    assert base == more_eps == more_tok == 2.0


def test_norm_const_only_divides():
    args = (np.array([[-1.0, -3.0]]), np.ones((1, 2)), np.array([0.25]))
    # -(1/1) * 0.25 * -4 = 1.0 ; with norm_const 8 -> 0.125
    assert policy_loss(*args, 1.0) == 1.0
    assert policy_loss(*args, 8.0) == 0.125


def test_integer_and_bool_mask():
    lp = np.array([[-1.0, -2.0]])
    adv = np.array([1.0])
    # mask 1, 0 -> -(1 * -1) = 1.0
    assert policy_loss(lp, np.array([[1, 0]]), adv, 1.0) == 1.0
    assert policy_loss(lp, np.array([[True, False]]), adv, 1.0) == 1.0


@pytest.mark.parametrize("norm_const", [0.0, -1.0, float("nan")])
def test_bad_norm_const(norm_const):
    with pytest.raises(ValueError):
        policy_loss(np.zeros((1, 2)), np.ones((1, 2)), np.zeros(1), norm_const)


@pytest.mark.parametrize(
    "logprobs, mask, advantages",
    [
        (np.zeros(2), np.ones(2), np.zeros(1)),                  # logprobs 1-D
        (np.zeros((1, 2)), np.ones((1, 3)), np.zeros(1)),        # mask shape differs
        (np.zeros((2, 2)), np.ones((2, 2)), np.zeros(3)),        # advantages length differs
        (np.zeros((2, 2)), np.ones((2, 2)), np.zeros((2, 1))),   # advantages 2-D
        (np.zeros((1, 2, 1)), np.ones((1, 2, 1)), np.zeros(1)),  # logprobs 3-D
    ],
)
def test_bad_shapes(logprobs, mask, advantages):
    with pytest.raises(ValueError):
        policy_loss(logprobs, mask, advantages, 1.0)


def test_pipeline_end_to_end():
    # group [1.0, 0.0] -> advantages 0.5, -0.5; one token each, logprob -1 for both
    # loss = -(1/1) * (0.5 * -1 + -0.5 * -1) = 0
    adv, keep = group_advantages(np.array([[1.0, 0.0]]))
    loss = policy_loss(np.array([[-1.0], [-1.0]]), np.ones((2, 1)), adv.reshape(-1), 1.0)
    assert keep.tolist() == [True] and loss == 0.0
