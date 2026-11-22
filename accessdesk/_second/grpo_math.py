"""Second versions of `group_advantages` and `policy_loss`.

Written by another model from the contract in the function's docstring only, without reading the main version.
Used by tests only (the differential test compares them with the main version).

Choices where the contract was silent when this version was written. The comparison of the two versions
(2026-11-17) put two of them into the contract: an infinite reward is a ValueError, and so is an input of
policy_loss without a `.shape`.
- group_advantages takes anything np.asarray can turn into a float array (lists too); it works on a copy.
- An infinite reward raises ValueError (only NaN marks an invalid sample, and inf would give NaN advantages).
- "Fewer than half valid" means 2 * n_valid < group_size, so with 3 samples 2 must be valid.
- tie_tol is used literally: a negative (or NaN) tie_tol never finds a tie.
- policy_loss: an input without a .shape (for example a list) raises ValueError; zero episodes or zero tokens
  give a loss of 0. Non-finite log-probabilities are not checked (that would need numpy functions).
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
    table = np.array(rewards, dtype=float)    # a copy: the caller's array is never changed
    if table.ndim != 2:
        raise ValueError(f"rewards must have two dimensions, got shape {table.shape}")
    n_groups, group_size = table.shape
    if n_groups == 0 or group_size == 0:
        raise ValueError(f"rewards has no groups or no samples: shape {table.shape}")
    if np.isinf(table).any():
        raise ValueError("rewards contains an infinite value; only NaN may mark an invalid sample")

    advantages = np.zeros((n_groups, group_size), dtype=float)
    keep = np.zeros(n_groups, dtype=bool)
    for g in range(n_groups):
        valid_positions = [j for j in range(group_size) if not np.isnan(table[g, j])]
        if 2 * len(valid_positions) < group_size:          # rule 2: fewer than half are valid
            continue
        values = [float(table[g, j]) for j in valid_positions]
        if max(values) - min(values) <= tie_tol:            # rule 3: a tie
            continue
        group_mean = sum(values) / len(values)
        keep[g] = True
        for j in valid_positions:                           # rule 5; invalid samples stay at 0.0
            advantages[g, j] = table[g, j] - group_mean
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
    if not norm_const > 0:                       # also catches NaN
        raise ValueError(f"norm_const must be greater than 0, got {norm_const!r}")
    for name, value in (("logprobs", logprobs), ("mask", mask), ("advantages", advantages)):
        if not hasattr(value, "shape"):
            raise ValueError(f"{name} must be an array or tensor with a shape, got {type(value).__name__}")
    lp_shape, mask_shape, adv_shape = tuple(logprobs.shape), tuple(mask.shape), tuple(advantages.shape)
    if len(lp_shape) != 2:
        raise ValueError(f"logprobs must be [n_episodes, n_tokens], got shape {lp_shape}")
    if mask_shape != lp_shape:
        raise ValueError(f"mask shape {mask_shape} differs from logprobs shape {lp_shape}")
    if adv_shape != (lp_shape[0],):
        raise ValueError(f"advantages must have shape ({lp_shape[0]},), got {adv_shape}")

    # Per episode: the masked sum of its log-probabilities; then weight each episode by its advantage.
    per_episode = (mask * logprobs).sum(-1)
    weighted_total = (advantages * per_episode).sum()
    return -weighted_total / norm_const
