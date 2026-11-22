"""`policy_loss`: hand-made cases, run on the main version and on the second version.

Every expected number is worked out by hand in a comment, from the contract in the function's docstring
(accessdesk/train/grpo_math.py). Each test takes the function under test as the argument `policy_loss`: the
main version, and the second version of accessdesk/_second once it is written (until then those tests are
skipped).

The inputs are numpy arrays (there is no torch on this machine). The loss is a sum of products, so a change of
one log-probability shows the gradient exactly; a separate test on the rented GPU checks the torch gradient.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import VERSIONS, function_under_test

# Two episodes of three tokens. Used by several tests.
LOGPROBS = np.array([[-1.0, -2.0, -3.0],
                     [-0.5, -0.5, -4.0]])
MASK = np.array([[0.0, 1.0, 1.0],
                 [1.0, 1.0, 0.0]])
ADVANTAGES = np.array([0.5, -0.5])
# Sampled-token sums: episode 0: -2 - 3 = -5. Episode 1: -0.5 - 0.5 = -1.
# Sum of advantage * token sum: 0.5 * (-5) + (-0.5) * (-1) = -2.5 + 0.5 = -2.0.
# With norm_const 4: loss = -(1 / 4) * (-2.0) = 0.5.
LOSS_AT_NORM_4 = 0.5


@pytest.fixture(params=VERSIONS)
def policy_loss(request):
    """The function under test: the main version, then the second version."""
    return function_under_test("policy_loss", request.param)


def test_hand_computed_value(policy_loss):
    assert float(policy_loss(LOGPROBS, MASK, ADVANTAGES, 4.0)) == pytest.approx(LOSS_AT_NORM_4, abs=1e-12)


def test_result_is_a_scalar(policy_loss):
    assert np.ndim(policy_loss(LOGPROBS, MASK, ADVANTAGES, 4.0)) == 0


def test_positive_advantage_and_higher_logprob_give_lower_loss(policy_loss):
    # One episode, advantage +1, two sampled tokens, norm_const 1.
    # Log-probabilities -2, -2: loss = -(1 * (-4)) = 4. Log-probabilities -1, -1 (more likely): loss = 2.
    mask = np.ones((1, 2))
    low = policy_loss(np.array([[-2.0, -2.0]]), mask, np.array([1.0]), 1.0)
    high = policy_loss(np.array([[-1.0, -1.0]]), mask, np.array([1.0]), 1.0)
    assert float(low) == pytest.approx(4.0)
    assert float(high) == pytest.approx(2.0)
    assert high < low


def test_negative_advantage_and_higher_logprob_give_higher_loss(policy_loss):
    # The same episode with advantage -1: losses -4 and -2. Making a bad episode more likely costs.
    mask = np.ones((1, 2))
    low = policy_loss(np.array([[-2.0, -2.0]]), mask, np.array([-1.0]), 1.0)
    high = policy_loss(np.array([[-1.0, -1.0]]), mask, np.array([-1.0]), 1.0)
    assert float(low) == pytest.approx(-4.0)
    assert float(high) == pytest.approx(-2.0)
    assert high > low


def test_zero_advantages_give_exactly_zero(policy_loss):
    loss = policy_loss(LOGPROBS, MASK, np.zeros(2), 4.0)
    assert float(loss) == 0.0


def test_masked_tokens_do_not_count(policy_loss):
    # Change only the log-probabilities where the mask is 0 (token 0 of episode 0, token 2 of episode 1).
    changed = LOGPROBS.copy()
    changed[0, 0] = -100.0
    changed[1, 2] = -0.001
    assert float(policy_loss(changed, MASK, ADVANTAGES, 4.0)) == pytest.approx(LOSS_AT_NORM_4, abs=1e-12)


def test_each_sampled_token_moves_the_loss_by_minus_advantage_over_norm(policy_loss):
    # Raise one sampled log-probability by 1. The loss must change by -advantage / norm_const:
    # episode 0 (advantage +0.5), token 1: 0.5 - 0.5 / 4 = 0.375.
    changed = LOGPROBS.copy()
    changed[0, 1] += 1.0
    assert float(policy_loss(changed, MASK, ADVANTAGES, 4.0)) == pytest.approx(0.375, abs=1e-12)
    # episode 1 (advantage -0.5), token 0: 0.5 + 0.5 / 4 = 0.625.
    changed = LOGPROBS.copy()
    changed[1, 0] += 1.0
    assert float(policy_loss(changed, MASK, ADVANTAGES, 4.0)) == pytest.approx(0.625, abs=1e-12)


def test_norm_const_only_divides(policy_loss):
    # norm_const 1: loss = 2.0. norm_const 4: 0.5. norm_const 8: 0.25.
    assert float(policy_loss(LOGPROBS, MASK, ADVANTAGES, 1.0)) == pytest.approx(2.0, abs=1e-12)
    assert float(policy_loss(LOGPROBS, MASK, ADVANTAGES, 8.0)) == pytest.approx(0.25, abs=1e-12)


def test_loss_is_a_sum_and_not_a_mean(policy_loss):
    # Add a third episode with advantage 0, and a fourth masked token to every episode. Nothing in the formula
    # changes, so the loss stays 0.5. (A mean over episodes or over tokens would change it.)
    logprobs = np.array([[-1.0, -2.0, -3.0, -9.0],
                         [-0.5, -0.5, -4.0, -9.0],
                         [-7.0, -7.0, -7.0, -9.0]])
    mask = np.array([[0.0, 1.0, 1.0, 0.0],
                     [1.0, 1.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0, 0.0]])
    advantages = np.array([0.5, -0.5, 0.0])
    assert float(policy_loss(logprobs, mask, advantages, 4.0)) == pytest.approx(LOSS_AT_NORM_4, abs=1e-12)


def test_longer_episode_counts_more(policy_loss):
    # Two episodes with the same advantage +1 and log-probability -1 per token; one has 1 sampled token, the
    # other 3. Token sums -1 and -3. loss = -(1 * (-1) + 1 * (-3)) / 2 = 2.0.
    # (Per-episode means would give 1.0.)
    logprobs = -np.ones((2, 3))
    mask = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert float(policy_loss(logprobs, mask, np.array([1.0, 1.0]), 2.0)) == pytest.approx(2.0, abs=1e-12)


def test_wrong_shapes_and_norm_const_are_refused(policy_loss):
    with pytest.raises(ValueError):
        policy_loss(LOGPROBS, MASK, ADVANTAGES, 0.0)
    with pytest.raises(ValueError):
        policy_loss(LOGPROBS, MASK[:, :2], ADVANTAGES, 4.0)
    # Advantages of shape [2, 1] would silently broadcast to [2, 2] in a product: this must be an error.
    with pytest.raises(ValueError):
        policy_loss(LOGPROBS, MASK, ADVANTAGES.reshape(2, 1), 4.0)


def test_inputs_are_not_changed(policy_loss):
    logprobs, mask, advantages = LOGPROBS.copy(), MASK.copy(), ADVANTAGES.copy()
    policy_loss(logprobs, mask, advantages, 4.0)
    np.testing.assert_array_equal(logprobs, LOGPROBS)
    np.testing.assert_array_equal(mask, MASK)
    np.testing.assert_array_equal(advantages, ADVANTAGES)


def test_integer_masks_are_accepted(policy_loss):
    # The loss mask is a list of 0 and 1; as an integer array it must give the same number as a float array.
    loss = policy_loss(LOGPROBS, MASK.astype(int), ADVANTAGES, 4.0)
    assert float(loss) == pytest.approx(LOSS_AT_NORM_4, abs=1e-12)


# --- a case that the comparison of the two versions decided (2026-11-17) ------------------------------------
@pytest.mark.parametrize("which", ["logprobs", "mask", "advantages", "all three"])
def test_plain_lists_are_refused_with_a_value_error(policy_loss, which):
    # The contract names arrays or tensors of a stated shape, and ValueError for a wrong shape. A plain list
    # has no `.shape`: the main version raised AttributeError, the second version ValueError. Decided:
    # ValueError. (A list is not converted, because a numpy conversion would cut the gradient on the GPU.)
    logprobs, mask, advantages = LOGPROBS, MASK, ADVANTAGES
    if which in ("logprobs", "all three"):
        logprobs = LOGPROBS.tolist()
    if which in ("mask", "all three"):
        mask = MASK.tolist()
    if which in ("advantages", "all three"):
        advantages = ADVANTAGES.tolist()
    with pytest.raises(ValueError):
        policy_loss(logprobs, mask, advantages, 4.0)
