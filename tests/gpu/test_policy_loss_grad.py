"""The gradient of `policy_loss` with torch tensors.

The hand-made tests in tests/core/test_policy_loss.py use numpy arrays, so they cannot see a
gradient. On the GPU the log-probabilities are torch tensors that carry the gradient, and what the optimiser
gets is

    d loss / d logprobs[i, t] = -advantages[i] * mask[i, t] / norm_const

These tests check exactly that, on small tensors whose expected gradient is worked out by hand in a comment.
The function under test is `impl.get("policy_loss")`: the version that every run uses. Which version that
is (`impl.which`) is printed (run pytest with -s). No model is needed.
"""
from __future__ import annotations

import pytest

from accessdesk import impl


@pytest.fixture
def policy_loss():
    print(f"\npolicy_loss under test: the {impl.which('policy_loss')} version")
    return impl.get("policy_loss")


def _case():
    """Two episodes of three tokens (the case of tests/core/test_policy_loss.py)."""
    import torch

    logprobs = torch.tensor([[-1.0, -2.0, -3.0],
                             [-0.5, -0.5, -4.0]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[0.0, 1.0, 1.0],
                         [1.0, 1.0, 0.0]], dtype=torch.float64)
    advantages = torch.tensor([0.5, -0.5], dtype=torch.float64)
    return logprobs, mask, advantages


def test_value_and_gradient_by_hand(policy_loss):
    import torch

    logprobs, mask, advantages = _case()
    loss = policy_loss(logprobs, mask, advantages, 4.0)
    assert loss.dim() == 0 and loss.requires_grad
    # Sampled-token sums: -5 and -1. loss = -(1/4) * (0.5 * (-5) + (-0.5) * (-1)) = -(1/4) * (-2) = 0.5.
    assert loss.item() == pytest.approx(0.5, abs=1e-12)
    loss.backward()
    # -advantage * mask / 4:  episode 0: -0.5 * [0, 1, 1] / 4;  episode 1: +0.5 * [1, 1, 0] / 4.
    expected = torch.tensor([[0.0, -0.125, -0.125],
                             [0.125, 0.125, 0.0]], dtype=torch.float64)
    assert torch.allclose(logprobs.grad, expected, atol=1e-15)


def test_gradient_formula_on_unequal_numbers(policy_loss):
    """The same formula on values that have no symmetry, so a swapped axis or a wrong sign cannot cancel."""
    import torch

    logprobs = torch.tensor([[-0.3, -1.7, -0.2, -2.9],
                             [-4.1, -0.6, -0.8, -1.3],
                             [-0.9, -0.4, -3.3, -0.1]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [1.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
    advantages = torch.tensor([0.75, -0.25, -0.5], dtype=torch.float64)
    policy_loss(logprobs, mask, advantages, 7.0).backward()
    expected = -(advantages[:, None] * mask) / 7.0
    assert torch.allclose(logprobs.grad, expected, atol=1e-15)
    # One entry by hand: episode 2, token 1: -(-0.5) * 1 / 7 = 1/14.
    assert logprobs.grad[2, 1].item() == pytest.approx(1 / 14, abs=1e-15)


def test_zero_advantages_give_exactly_zero_loss_and_gradient(policy_loss):
    import torch

    logprobs, mask, _ = _case()
    loss = policy_loss(logprobs, mask, torch.zeros(2, dtype=torch.float64), 4.0)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(logprobs.grad).item() == 0


def test_a_masked_token_gets_no_gradient_and_changes_nothing(policy_loss):
    import torch

    logprobs, mask, advantages = _case()
    policy_loss(logprobs, mask, advantages, 4.0).backward()
    assert logprobs.grad[0, 0].item() == 0.0 and logprobs.grad[1, 2].item() == 0.0
    changed = logprobs.detach().clone()
    changed[0, 0], changed[1, 2] = -50.0, -70.0             # the two masked entries
    assert policy_loss(changed, mask, advantages, 4.0).item() == pytest.approx(0.5, abs=1e-12)


def test_sign_a_gradient_step_raises_the_logprob_of_a_good_episode(policy_loss):
    """Gradient descent moves against the gradient. For a positive advantage the gradient of a sampled token
    is negative, so the step makes its log-probability larger; for a negative advantage, smaller."""
    logprobs, mask, advantages = _case()
    policy_loss(logprobs, mask, advantages, 4.0).backward()
    step = -1.0 * logprobs.grad                              # learning rate 1
    assert step[0, 1].item() > 0 and step[0, 2].item() > 0   # advantage +0.5
    assert step[1, 0].item() < 0 and step[1, 1].item() < 0   # advantage -0.5


def test_the_normaliser_only_divides(policy_loss):
    """Twice the normaliser: half the gradient. More episodes or tokens in the batch: no change for the
    others."""
    import torch

    logprobs, mask, advantages = _case()
    policy_loss(logprobs, mask, advantages, 4.0).backward()
    at_4 = logprobs.grad.clone()

    logprobs_b, _, _ = _case()
    policy_loss(logprobs_b, mask, advantages, 8.0).backward()
    assert torch.allclose(logprobs_b.grad, at_4 / 2, atol=1e-15)

    # A third episode with advantage 0 and one more column of masked tokens: the old entries keep their
    # gradient.
    wider = torch.tensor([[-1.0, -2.0, -3.0, -9.0],
                          [-0.5, -0.5, -4.0, -9.0],
                          [-6.0, -6.0, -6.0, -6.0]], dtype=torch.float64, requires_grad=True)
    wider_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0],
                               [1.0, 1.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    wider_advantages = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float64)
    loss = policy_loss(wider, wider_mask, wider_advantages, 4.0)
    loss.backward()
    assert loss.item() == pytest.approx(0.5, abs=1e-12)
    assert torch.allclose(wider.grad[:2, :3], at_4, atol=1e-15)
    assert torch.count_nonzero(wider.grad[2]).item() == 0
    assert torch.count_nonzero(wider.grad[:, 3]).item() == 0


def test_float32_tensors_as_in_training(policy_loss):
    """The training loop gives float32 log-probabilities, a float32 mask and float32 advantages."""
    import torch

    logprobs = torch.tensor([[-1.0, -2.0], [-0.5, -0.25]], dtype=torch.float32, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    advantages = torch.tensor([0.25, -0.25], dtype=torch.float32)
    loss = policy_loss(logprobs, mask, advantages, 2.0)
    # -(1/2) * (0.25 * (-3) + (-0.25) * (-0.5)) = -(1/2) * (-0.625) = 0.3125
    assert loss.dtype == torch.float32 and loss.item() == pytest.approx(0.3125, abs=1e-7)
    loss.backward()
    expected = torch.tensor([[-0.125, -0.125], [0.125, 0.0]], dtype=torch.float32)
    assert torch.allclose(logprobs.grad, expected, atol=1e-7)


def test_wrong_inputs_raise(policy_loss):
    import torch

    logprobs, mask, advantages = _case()
    with pytest.raises(ValueError):
        policy_loss(logprobs, mask, advantages, 0.0)
    with pytest.raises(ValueError):
        policy_loss(logprobs, mask[:, :2], advantages, 4.0)
    with pytest.raises(ValueError):
        policy_loss(logprobs, mask, torch.zeros(3, dtype=torch.float64), 4.0)
