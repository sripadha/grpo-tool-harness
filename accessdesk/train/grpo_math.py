"""The two pieces of GRPO arithmetic: group advantages and the policy loss.

These are the main versions of `group_advantages` and `policy_loss`. The rest of the code reaches them through
`accessdesk.impl.get(name)`. Independent second versions, written from the docstrings below only, are in
`accessdesk/_second/grpo_math.py`; tests compare the two.
Tests: `.venv/bin/python -m pytest tests/core/test_group_advantages.py tests/core/test_policy_loss.py
tests/core/test_differential_grpo_math.py`

This file must not import torch: `policy_loss` receives numpy arrays in the tests on a laptop and torch
tensors on the GPU, and is written with operations that both accept.
"""
from __future__ import annotations

import numpy as np


def group_advantages(rewards, tie_tol: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Advantage of each sample against the mean of its own group, and which groups to keep.

    Input
        rewards   float array of shape [n_groups, group_size]. One row is one group: the samples of one
                  scenario in one training step. A value is the episode's reward divided by 100: 0.0 for a
                  fail and 0.77 to 1.0 for a pass with the default config (a fail can go down to -0.15 when
                  cfg.penalise_malformed_on_fail is on). NaN marks a sample that was lost to a harness error
                  ("invalid").
        tie_tol   a group whose valid rewards all lie within this distance of each other is a tie.

    Output: (advantages, keep)
        advantages   float array, the same shape as rewards. It never contains NaN.
        keep         bool array of shape [n_groups].

    Rules, for each group on its own
        1. A sample is valid if its reward is not NaN.
        2. The group is dropped if fewer than half of its samples are valid. Exactly half is enough to stay.
        3. The group is dropped if (max - min) of its valid rewards <= tie_tol. With the default tie_tol of
           0.0 this drops exactly the groups in which every valid reward is the same number.
        4. A dropped group has keep False and advantage 0.0 for every sample.
        5. A kept group has keep True. A valid sample gets its reward minus the mean of the group's VALID
           rewards. An invalid sample gets 0.0.

    There is no division by the standard deviation and no other scaling: an advantage is a difference of two
    rewards on the 0-to-1 scale. So the advantages of a kept group add up to 0.

    The comparison in rule 3 is made on floating-point numbers. For example 1.0 - 0.9 is 0.09999999999999998
    in floating point, which is <= 0.1. Where the exact behaviour at a boundary matters, choose a tie_tol that
    lies between two possible reward distances (rewards move in steps of 0.01).

    Raises ValueError if rewards does not have exactly two dimensions, if it has no groups or no samples,
    or if a reward is +infinity or -infinity, in whatever group it is (only NaN marks an invalid sample, and
    an infinite reward would put NaN into the advantages). The input array is not changed.
    """
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError(f"rewards must be [n_groups, group_size], both at least 1; got {rewards.shape}")
    if np.isinf(rewards).any():
        raise ValueError("a reward is infinite; only NaN may mark a sample that was lost")
    n_groups, group_size = rewards.shape

    # Start from "every group dropped": all advantages 0, keep False. The loop fills in the kept groups.
    advantages = np.zeros((n_groups, group_size), dtype=float)
    keep = np.zeros(n_groups, dtype=bool)

    for g in range(n_groups):
        # Rule 1: the valid samples of this group, and their rewards.
        valid = ~np.isnan(rewards[g])
        valid_rewards = rewards[g][valid]

        # Rule 2: fewer than half valid -> dropped.
        # (2 * count < size says "count < size / 2" without a fraction.)
        if 2 * valid_rewards.size < group_size:
            continue

        # Rule 3: all valid rewards (almost) equal -> a tie, nothing to learn from this group -> dropped.
        if valid_rewards.max() - valid_rewards.min() <= tie_tol:
            continue

        # Rule 5: a kept group. Each valid sample gets its reward minus the group's mean; the others stay 0.
        advantages[g][valid] = valid_rewards - valid_rewards.mean()
        keep[g] = True

    return advantages, keep


def policy_loss(logprobs, mask, advantages, norm_const: float):
    """The plain policy-gradient loss with fixed advantages.

    Input
        logprobs     [n_episodes, n_tokens]: the log-probability that the model gives to each token of each
                     episode. On the GPU this is a torch tensor that carries the gradient.
        mask         [n_episodes, n_tokens]: the weight of each token in the loss, 1 for a token that the
                     model sampled and 0 for every other token (prompt, tool results, padding).
        advantages   [n_episodes]: one fixed number per episode (from group_advantages). No gradient.
        norm_const   a fixed positive number. It is the same for every step, whatever the number of kept
                     groups, episodes or tokens.

    Output: the scalar

        loss = -(1 / norm_const) * sum over episodes i of ( advantages[i] * sum over tokens t of
                                                             mask[i, t] * logprobs[i, t] )

    What follows from this formula
        - Sign: for an episode with a positive advantage, a higher log-probability of its sampled tokens gives
          a lower loss. For a negative advantage it gives a higher loss.
        - If every advantage is 0, the loss is exactly 0 (and so is its gradient).
        - A token with mask 0 does not change the loss, whatever its log-probability is. Its log-probability
          must still be a finite number, because 0 * (-infinity) is NaN.
        - norm_const only divides. The loss is a sum and not a mean: an extra episode with advantage 0 or an
          extra masked token changes nothing.

    Use only operations that numpy arrays and torch tensors share: `*`, `.sum()` (give the axis as a plain
    positional number, for example `.sum(-1)`), unary minus, and division by a Python number. Do not convert
    the inputs with numpy functions: that would cut the gradient on the GPU.

    Raises ValueError if norm_const is not greater than 0, if logprobs, mask or advantages has no `.shape`
    (a plain Python list, for example), or if the shapes are not as stated above.
    """
    # The checks. A wrong shape would not fail by itself: the product below would stretch it to fit.
    if not norm_const > 0:
        raise ValueError(f"norm_const must be greater than 0, got {norm_const}")
    if not all(hasattr(value, "shape") for value in (logprobs, mask, advantages)):
        raise ValueError("logprobs, mask and advantages must be arrays or tensors, not plain lists")
    if len(logprobs.shape) != 2 or tuple(mask.shape) != tuple(logprobs.shape):
        raise ValueError(f"logprobs and mask must both be [n_episodes, n_tokens]; "
                         f"got {tuple(logprobs.shape)} and {tuple(mask.shape)}")
    if tuple(advantages.shape) != (logprobs.shape[0],):
        raise ValueError(f"advantages must be [n_episodes] = [{logprobs.shape[0]}]; "
                         f"got {tuple(advantages.shape)}")

    # Inner sum: for each episode, the log-probabilities of its sampled tokens added up. Shape [n_episodes].
    episode_logprob = (mask * logprobs).sum(-1)

    # Outer sum: each episode weighted by its advantage. One number.
    weighted_sum = (advantages * episode_logprob).sum()

    # Minus, because an optimiser makes the loss smaller and we want good episodes to become more likely.
    return -(weighted_sum / norm_const)
