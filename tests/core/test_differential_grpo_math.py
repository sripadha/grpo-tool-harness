"""The second versions of `group_advantages` and `policy_loss` against the main versions on generated inputs.

The two comparisons at the top are skipped while the second version (accessdesk/_second/grpo_math.py) is not
written. So that a skipped test is not an untested test, the rest of the file runs today: the comparison
passes for the main version against itself, and it fails for each of several wrong versions.

Further down, added by the comparison of the two versions: edge inputs.
  - group_advantages: every group of 1 to 4 samples over six reward values (NaN among them) with eight
    tolerances; every such group with the tolerance set exactly at its reward distance, one float below it
    and one float above it; all groups of one size as one batch; and inputs that must be refused (wrong
    shapes, an infinite reward in every position).
  - policy_loss: masks of zeros, of ones, of ints and of bools, advantages that are all 0, sizes down to one
    episode with one token, and about 120 inputs that must be refused (shapes that a product would stretch
    without an error, a norm_const that is not above 0, plain lists).
For both functions the right result is also worked out a third way (exact fractions; plain loops), and each
version is compared with that, not only with the other version: the second policy_loss computes its number
with the same two lines as the main one, so the two agreeing says little by itself.
"""
from __future__ import annotations

import functools
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from accessdesk import impl
from accessdesk.train import grpo_math as main
from tests.conftest import VERSIONS, function_under_test, skip_unless_second_is_written

# Rewards as the harness makes them, in steps of 0.01: a pass is between 0.77 and 1.00; a fail is 0, or 0 down
# to -0.15 when the switch for penalties on failing episodes is on.
REWARD_VALUES = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.95, 0.88, 0.81, 0.77, -0.05, -0.15])
# Tolerances that lie between two possible reward distances (multiples of 0.01), plus exact equality. So a
# rounding error of the last bit can never decide whether a group is a tie.
TIE_TOLS = (0.0, 0.0, 0.025, 0.075, 0.125, 0.5)


def advantage_inputs(n_cases: int = 400):
    """Generated (rewards, tie_tol) pairs: small groups, many ties, lost samples."""
    rng = np.random.default_rng(1)
    for _ in range(n_cases):
        n_groups = int(rng.integers(1, 7))
        group_size = int(rng.integers(1, 10))
        rewards = rng.choice(REWARD_VALUES, size=(n_groups, group_size))
        constant_rows = rng.random(n_groups) < 0.3            # whole groups with one reward: ties
        rewards[constant_rows] = rewards[constant_rows][:, :1]
        rewards[rng.random((n_groups, group_size)) < 0.2] = np.nan
        yield rewards, float(rng.choice(TIE_TOLS))


def compare_group_advantages(other_function, main_function) -> None:
    """Assert that the two functions agree on every generated input."""
    for rewards, tie_tol in advantage_inputs():
        expected_advantages, expected_keep = main_function(rewards.copy(), tie_tol)
        advantages, keep = other_function(rewards.copy(), tie_tol)
        message = f"rewards={rewards.tolist()} tie_tol={tie_tol}"
        assert np.asarray(keep).tolist() == expected_keep.tolist(), message
        np.testing.assert_allclose(np.asarray(advantages), expected_advantages, atol=1e-12, err_msg=message)


def loss_inputs(n_cases: int = 300):
    """Generated (logprobs, mask, advantages, norm_const) tuples."""
    rng = np.random.default_rng(2)
    for _ in range(n_cases):
        n_episodes = int(rng.integers(1, 9))
        n_tokens = int(rng.integers(1, 13))
        logprobs = -rng.exponential(1.0, size=(n_episodes, n_tokens))
        mask = (rng.random((n_episodes, n_tokens)) < 0.5).astype(float)
        advantages = rng.choice([0.0, 0.0, 0.5, -0.5, 0.875, -0.125, 0.0125], size=n_episodes)
        yield logprobs, mask, advantages, float(rng.choice([1.0, 8.0, 128.0, 12800.0]))


def compare_policy_loss(other_function, main_function) -> None:
    """Assert that the two functions agree on every generated input."""
    for logprobs, mask, advantages, norm_const in loss_inputs():
        expected = float(main_function(logprobs, mask, advantages, norm_const))
        found = float(other_function(logprobs.copy(), mask.copy(), advantages.copy(), norm_const))
        message = f"logprobs={logprobs.tolist()} mask={mask.tolist()} advantages={advantages.tolist()}"
        assert found == pytest.approx(expected, rel=1e-9, abs=1e-12), f"{message} norm_const={norm_const}"


# ---------------------------------------------------------------------------
# The second versions (skipped until they are written)
# ---------------------------------------------------------------------------
def test_the_second_group_advantages_matches_the_main_version():
    skip_unless_second_is_written("group_advantages")
    compare_group_advantages(impl.second_version("group_advantages"), main.group_advantages)


def test_the_second_policy_loss_matches_the_main_version():
    skip_unless_second_is_written("policy_loss")
    compare_policy_loss(impl.second_version("policy_loss"), main.policy_loss)


# ---------------------------------------------------------------------------
# The comparison itself: it passes for a correct version and fails for wrong ones
# ---------------------------------------------------------------------------
def test_generated_advantage_inputs_cover_the_cases_that_matter():
    cases = list(advantage_inputs())
    n_with_nan = sum(1 for rewards, _ in cases if np.isnan(rewards).any())
    kept = [main.group_advantages(rewards, tie_tol)[1] for rewards, tie_tol in cases]
    n_kept = sum(int(keep.sum()) for keep in kept)
    n_dropped = sum(int((~keep).sum()) for keep in kept)
    assert len(cases) == 400
    assert n_with_nan > 100
    assert n_kept > 200 and n_dropped > 200
    n_with_negative = sum(1 for rewards, _ in cases if (rewards[~np.isnan(rewards)] < 0).any())
    assert n_with_negative > 100


def test_comparisons_pass_for_the_main_version_against_itself():
    compare_group_advantages(main.group_advantages, main.group_advantages)
    compare_policy_loss(main.policy_loss, main.policy_loss)


def _advantages_divided_by_std(rewards, tie_tol=0.0):
    advantages, keep = main.group_advantages(rewards, tie_tol)
    for g in np.flatnonzero(keep):
        advantages[g] = advantages[g] / rewards[g][~np.isnan(rewards[g])].std()
    return advantages, keep


def _advantages_with_nan_as_zero_reward(rewards, tie_tol=0.0):
    return main.group_advantages(np.nan_to_num(rewards, nan=0.0), tie_tol)


def _advantages_with_one_mean_for_the_batch(rewards, tie_tol=0.0):
    advantages, keep = main.group_advantages(rewards, tie_tol)
    valid = ~np.isnan(rewards)
    for g in np.flatnonzero(keep):
        advantages[g][valid[g]] = rewards[g][valid[g]] - rewards[valid].mean()
    return advantages, keep


def _advantages_strictly_below_tol(rewards, tie_tol=0.0):
    """Drops a group only if max - min < tie_tol: at tie_tol 0 nothing is ever a tie."""
    advantages, keep = main.group_advantages(rewards, tie_tol)
    for g in range(rewards.shape[0]):
        values = rewards[g][~np.isnan(rewards[g])]
        if 2 * values.size >= rewards.shape[1] and values.max() - values.min() == tie_tol:
            keep[g] = True
    return advantages, keep


def _advantages_half_valid_is_not_enough(rewards, tie_tol=0.0):
    advantages, keep = main.group_advantages(rewards, tie_tol)
    for g in range(rewards.shape[0]):
        if 2 * int((~np.isnan(rewards[g])).sum()) == rewards.shape[1]:
            advantages[g] = 0.0
            keep[g] = False
    return advantages, keep


def _advantages_keep_always_true(rewards, tie_tol=0.0):
    advantages, keep = main.group_advantages(rewards, tie_tol)
    return advantages, np.ones_like(keep)


WRONG_ADVANTAGES = [_advantages_divided_by_std, _advantages_with_nan_as_zero_reward,
                    _advantages_with_one_mean_for_the_batch, _advantages_strictly_below_tol,
                    _advantages_half_valid_is_not_enough, _advantages_keep_always_true]


@pytest.mark.parametrize("wrong", WRONG_ADVANTAGES, ids=lambda function: function.__name__.lstrip("_"))
def test_comparison_fails_for_a_wrong_group_advantages(wrong):
    with pytest.raises(AssertionError):
        compare_group_advantages(wrong, main.group_advantages)


def _loss_as_mean_over_episodes(logprobs, mask, advantages, norm_const):
    return -(advantages * (mask * logprobs).sum(-1)).mean() / norm_const


def _loss_as_mean_over_tokens(logprobs, mask, advantages, norm_const):
    return -(advantages * (mask * logprobs).sum(-1) / np.maximum(mask.sum(-1), 1.0)).sum() / norm_const


def _loss_without_mask(logprobs, mask, advantages, norm_const):
    return -(advantages * logprobs.sum(-1)).sum() / norm_const


def _loss_with_wrong_sign(logprobs, mask, advantages, norm_const):
    return (advantages * (mask * logprobs).sum(-1)).sum() / norm_const


def _loss_without_norm_const(logprobs, mask, advantages, norm_const):
    return -(advantages * (mask * logprobs).sum(-1)).sum()


WRONG_LOSSES = [_loss_as_mean_over_episodes, _loss_as_mean_over_tokens, _loss_without_mask,
                _loss_with_wrong_sign, _loss_without_norm_const]


@pytest.mark.parametrize("wrong", WRONG_LOSSES, ids=lambda function: function.__name__.lstrip("_"))
def test_comparison_fails_for_a_wrong_policy_loss(wrong):
    with pytest.raises(AssertionError):
        compare_policy_loss(wrong, main.policy_loss)


# ---------------------------------------------------------------------------
# Edge inputs for group_advantages
# ---------------------------------------------------------------------------
NAN = float("nan")
INF = float("inf")
EDGE_REWARDS = (NAN, 0.0, 1.0, 0.95, 0.77, -0.15)
# 0.05, 0.18, 0.23, 1.0 and 1.15 are distances between two of the rewards above, as typed. In floating
# point 1.0 - 0.95 is a little above 0.05 and 0.95 - 0.77 a little below 0.18: tolerances at a boundary.
EDGE_TOLS = (0.0, 0.05, 0.18, 0.23, 0.95, 1.0, 1.15, -1.0)
AdvantageCase = tuple[list[list[float]], float]            # (the rows of rewards, tie_tol)


def edge_groups() -> list[list[float]]:
    """Every group of 1, 2, 3 or 4 samples over EDGE_REWARDS: 6 + 36 + 216 + 1296 groups."""
    return [list(row) for size in (1, 2, 3, 4) for row in itertools.product(EDGE_REWARDS, repeat=size)]


def distance(group: list[float]) -> float | None:
    """max - min of the group's valid rewards in floating point, or None if they are all equal or absent."""
    valid = [reward for reward in group if not math.isnan(reward)]
    return max(valid) - min(valid) if len(set(valid)) > 1 else None


@functools.lru_cache(maxsize=None)
def advantage_edge_cases() -> list[AdvantageCase]:
    """Single groups with every tolerance; single groups with the tolerance at, just below and just above
    their own reward distance; and all groups of one size as one batch, in a mixed order."""
    groups = edge_groups()
    cases: list[AdvantageCase] = [([group], tie_tol) for group in groups for tie_tol in EDGE_TOLS]
    for group in groups:
        gap = distance(group)
        if gap is not None:
            below, above = math.nextafter(gap, 0.0), math.nextafter(gap, INF)
            cases += [([group], gap), ([group], below), ([group], above)]
    rng = np.random.default_rng(4)
    for size in (1, 2, 3, 4):
        batch = [group for group in groups if len(group) == size]
        for tie_tol in EDGE_TOLS:
            cases.append(([batch[i] for i in rng.permutation(len(batch))], tie_tol))
    return cases


def by_fractions(rewards, tie_tol: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The contract's rules a third way: group by group in plain Python, the mean and the differences in
    exact fractions (a float is an exact fraction), rounded to a float once at the end. The comparison of
    rule 3 is made on floats, as the contract says."""
    advantages, keep = [], []
    for group in np.asarray(rewards, dtype=float).tolist():
        valid = [reward for reward in group if not math.isnan(reward)]
        if 2 * len(valid) < len(group) or max(valid) - min(valid) <= tie_tol:
            advantages.append([0.0] * len(group))
            keep.append(False)
            continue
        mean = sum(Fraction(reward) for reward in valid) / len(valid)
        advantages.append([0.0 if math.isnan(reward) else float(Fraction(reward) - mean) for reward in group])
        keep.append(True)
    return np.array(advantages), np.array(keep)


def compare_group_advantages_on(cases: list[AdvantageCase], other_function, main_function) -> None:
    """Assert that the two functions agree on every case, and that the result has the stated form."""
    for rows, tie_tol in cases:
        rewards = np.array(rows)
        expected_advantages, expected_keep = main_function(rewards.copy(), tie_tol)
        advantages, keep = other_function(rewards.copy(), tie_tol)
        message = f"rewards={rows} tie_tol={tie_tol!r}"
        assert np.asarray(keep).dtype == bool and np.asarray(keep).shape == (len(rows),), message
        assert np.asarray(advantages).shape == rewards.shape, message
        assert np.asarray(keep).tolist() == expected_keep.tolist(), message
        # A NaN in either result fails here too: NaN <= 1e-12 is False.
        assert np.abs(np.asarray(advantages) - expected_advantages).max() <= 1e-12, message


def refused_advantage_inputs() -> list[tuple]:
    """(rewards, tie_tol) that the contract refuses: wrong shapes, and an infinite reward in every position
    of every group of up to 3 samples (also in a group that rule 2 or rule 3 would drop)."""
    shapes = [np.zeros(3), np.zeros((2, 2, 2)), np.array(1.0), np.zeros((0, 3)), np.zeros((3, 0)),
              np.zeros((0, 0)), [], [[]], [1.0, 0.0]]
    inputs = [(rewards, 0.0) for rewards in shapes]
    for size in (1, 2, 3):
        for others in itertools.product(EDGE_REWARDS, repeat=size - 1):
            for position in range(size):
                for infinite in (INF, -INF):
                    group = list(others[:position]) + [infinite] + list(others[position:])
                    inputs.append((np.array([group]), 0.0))
                    inputs.append((np.array([[1.0, 0.0, 0.5][:size], group]), INF))
    return inputs


def is_refused(function, *args) -> bool:
    """True if the call raises ValueError; any other error is not caught."""
    try:
        function(*args)
    except ValueError:
        return True
    return False


def test_the_second_group_advantages_matches_the_main_version_on_the_edge_cases():
    skip_unless_second_is_written("group_advantages")
    compare_group_advantages_on(advantage_edge_cases(), impl.second_version("group_advantages"),
                                main.group_advantages)


@pytest.mark.parametrize("version", VERSIONS)
def test_each_group_advantages_gives_the_result_found_with_fractions(version):
    function = function_under_test("group_advantages", version)
    compare_group_advantages_on(advantage_edge_cases(), function, by_fractions)
    generated = [(rewards.tolist(), tie_tol) for rewards, tie_tol in advantage_inputs()]
    compare_group_advantages_on(generated, function, by_fractions)


@pytest.mark.parametrize("version", VERSIONS)
def test_each_group_advantages_refuses_wrong_shapes_and_infinite_rewards(version):
    function = function_under_test("group_advantages", version)
    inputs = refused_advantage_inputs()
    assert len(inputs) == 9 + 2 * 2 * (1 + 2 * 6 + 3 * 36)
    accepted = [(np.asarray(rewards).tolist(), tie_tol) for rewards, tie_tol in inputs
                if not is_refused(function, rewards, tie_tol)]
    assert accepted == []


def test_the_advantage_edge_cases_hold_what_they_claim():
    cases = advantage_edge_cases()
    n_groups, n_with_a_distance = 1554, sum(1 for group in edge_groups() if distance(group) is not None)
    assert len(edge_groups()) == n_groups
    assert len(cases) == n_groups * len(EDGE_TOLS) + 3 * n_with_a_distance + 4 * len(EDGE_TOLS)
    assert n_with_a_distance > 1000
    # At its own distance a group is a tie, one float below it is kept, one float above it is a tie. This
    # holds when at least half of the samples are valid; the answers come from the rule, not from a version.
    at_the_boundary = cases[n_groups * len(EDGE_TOLS):n_groups * len(EDGE_TOLS) + 3 * n_with_a_distance]
    kept = [bool(by_fractions(np.array(rows), tie_tol)[1][0]) for rows, tie_tol in at_the_boundary]
    enough_valid = [2 * sum(1 for reward in rows[0] if not math.isnan(reward)) >= len(rows[0])
                    for rows, _ in at_the_boundary]
    assert kept == [enough and position % 3 == 1 for position, enough in enumerate(enough_valid)]
    assert sum(kept) > 800
    # Groups of size 1 and 2, one valid sample, no valid sample, ties, and the rounding at a boundary.
    singles = {(tuple(rows[0]), tie_tol): bool(by_fractions(np.array(rows), tie_tol)[1][0])
               for rows, tie_tol in cases[:n_groups * len(EDGE_TOLS)]}
    assert singles[((1.0,), 0.0)] is False and singles[((1.0,), -1.0)] is True       # one sample: a tie at 0
    assert singles[((1.0, NAN), 0.0)] is False and singles[((1.0, NAN), -1.0)] is True
    assert singles[((1.0, NAN, NAN), -1.0)] is False                                 # fewer than half valid
    assert singles[((1.0, 0.95), 0.05)] is True         # 1.0 - 0.95 is 0.050000000000000044 in floats: kept
    assert singles[((0.95, 0.77), 0.18)] is False       # 0.95 - 0.77 is 0.17999999999999994: a tie
    assert sum(singles.values()) > 4000 and sum(not kept for kept in singles.values()) > 4000


def _advantages_infinite_rewards_allowed(rewards, tie_tol=0.0):
    """Wrong since the decision of 2026-11-17 (this is what the main version did before): an infinite reward
    is not refused."""
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2 or rewards.size == 0:
        raise ValueError("shape")
    finite = np.where(np.isinf(rewards), 0.0, rewards)
    return main.group_advantages(finite, tie_tol)


def _advantages_rounded_distance(rewards, tie_tol=0.0):
    """Wrong on purpose: the distance of rule 3 is rounded to 9 places before it is compared."""
    advantages, keep = main.group_advantages(rewards, tie_tol)
    for g in range(rewards.shape[0]):
        values = rewards[g][~np.isnan(rewards[g])]
        if 2 * values.size >= rewards.shape[1]:
            keep[g] = not round(float(values.max() - values.min()), 9) <= tie_tol
            advantages[g] = 0.0
            if keep[g]:
                advantages[g][~np.isnan(rewards[g])] = values - values.mean()
    return advantages, keep


def _advantages_one_valid_sample_kept(rewards, tie_tol=0.0):
    """Wrong on purpose: a group with a single valid sample is kept (with advantage 0) when half is valid."""
    advantages, keep = main.group_advantages(rewards, tie_tol)
    for g in range(rewards.shape[0]):
        if int((~np.isnan(rewards[g])).sum()) == 1 and rewards.shape[1] <= 2:
            keep[g] = True
    return advantages, keep


EDGE_WRONG_ADVANTAGES = WRONG_ADVANTAGES + [_advantages_rounded_distance, _advantages_one_valid_sample_kept]


@pytest.mark.parametrize("wrong", EDGE_WRONG_ADVANTAGES, ids=lambda function: function.__name__.lstrip("_"))
def test_the_edge_comparison_fails_for_a_wrong_group_advantages(wrong):
    with np.errstate(all="ignore"):               # a wrong version may divide 0 by 0; that gives NaN
        with pytest.raises(AssertionError):
            compare_group_advantages_on(advantage_edge_cases(), wrong, main.group_advantages)
        with pytest.raises(AssertionError):
            compare_group_advantages_on(advantage_edge_cases(), wrong, by_fractions)


def test_the_refusal_check_fails_for_a_version_that_lets_infinite_rewards_through():
    accepted = [1 for rewards, tie_tol in refused_advantage_inputs()
                if not is_refused(_advantages_infinite_rewards_allowed, rewards, tie_tol)]
    assert len(accepted) == 2 * 2 * (1 + 2 * 6 + 3 * 36)


# ---------------------------------------------------------------------------
# Edge inputs for policy_loss
# ---------------------------------------------------------------------------
MASK_KINDS = ("random", "zeros", "ones", "one_token", "ints", "bools")
ADVANTAGE_KINDS = ("random", "zeros", "one_not_zero", "plus_and_minus")


@functools.lru_cache(maxsize=None)
def loss_edge_inputs() -> list[tuple]:
    """(logprobs, mask, advantages, norm_const, mask kind, advantage kind) for every combination of four
    numbers of episodes, four numbers of tokens, six kinds of mask, four kinds of advantages and three
    norm_consts."""
    rng = np.random.default_rng(6)
    inputs = []
    for n_episodes, n_tokens in itertools.product((1, 2, 3, 8), (1, 2, 5, 40)):
        for mask_kind, advantage_kind in itertools.product(MASK_KINDS, ADVANTAGE_KINDS):
            for norm_const in (1.0, 96.0, 0.5):
                logprobs = -rng.exponential(2.0, size=(n_episodes, n_tokens))
                mask = (rng.random((n_episodes, n_tokens)) < 0.5).astype(float)
                if mask_kind == "zeros":
                    mask = np.zeros((n_episodes, n_tokens))
                elif mask_kind == "ones":
                    mask = np.ones((n_episodes, n_tokens))
                elif mask_kind == "one_token":
                    mask = np.zeros((n_episodes, n_tokens))
                    mask[int(rng.integers(n_episodes)), int(rng.integers(n_tokens))] = 1.0
                elif mask_kind == "ints":
                    mask = mask.astype(int)
                elif mask_kind == "bools":
                    mask = mask.astype(bool)
                advantages = rng.choice([0.5, -0.5, 0.23, -0.77, 0.0125, 1.15], size=n_episodes)
                if advantage_kind == "zeros":
                    advantages = np.zeros(n_episodes)
                elif advantage_kind == "one_not_zero":
                    advantages = np.zeros(n_episodes)
                    advantages[int(rng.integers(n_episodes))] = -0.77
                elif advantage_kind == "plus_and_minus":
                    advantages = advantages - advantages.mean()        # as a kept group gives them
                inputs.append((logprobs, mask, advantages, norm_const, mask_kind, advantage_kind))
    return inputs


def by_loops(logprobs, mask, advantages, norm_const: float) -> float:
    """The contract's formula a third way: two plain loops over Python floats, added up with math.fsum."""
    terms = [float(advantages[i]) * float(mask[i][t]) * float(logprobs[i][t])
             for i in range(len(logprobs)) for t in range(len(logprobs[i]))]
    return -math.fsum(terms) / norm_const


def compare_policy_loss_on_the_edge_inputs(other_function, main_function) -> None:
    """Assert that the two functions agree on every edge input, and that a loss that must be 0 is 0."""
    for logprobs, mask, advantages, norm_const, mask_kind, advantage_kind in loss_edge_inputs():
        expected = float(main_function(logprobs, mask, advantages, norm_const))
        found = float(other_function(logprobs.copy(), mask.copy(), advantages.copy(), norm_const))
        message = (f"mask {mask_kind}, advantages {advantage_kind}, norm_const {norm_const}: "
                   f"logprobs={logprobs.tolist()} mask={mask.tolist()} advantages={advantages.tolist()}")
        assert found == pytest.approx(expected, rel=1e-9, abs=1e-12), message
        if mask_kind == "zeros" or advantage_kind == "zeros":
            assert found == 0.0, message                              # exactly 0, not only close to it


def refused_loss_inputs() -> list[tuple]:
    """(logprobs, mask, advantages, norm_const) that the contract refuses. Many of the shapes would give a
    number without an error if they were not checked: a product stretches them to fit."""
    z = np.zeros
    inputs: list[tuple] = []
    for n, t in ((1, 1), (2, 3), (3, 3), (4, 2)):
        inputs += [(z((n, t)), z((n, t)), z(n), norm_const)
                   for norm_const in (0.0, -1.0, NAN, -INF, -0.0, 0, -3)]
        inputs += [(z(shape), z(shape), z(n), 1.0) for shape in ((t,), (n, t, 1), (1, n, t), ())]
        mask_shapes = {(n, t + 1), (n + 1, t), (t, n), (n,), (t,), (n, t, 1), (1, n, t), (n * t,), (1, t),
                       (n, 1), (1, 1), ()} - {(n, t)}
        inputs += [(z((n, t)), z(shape), z(n), 1.0) for shape in sorted(mask_shapes)]
        advantage_shapes = {(n, 1), (1, n), (), (n + 1,), (n - 1,), (t,), (n, t), (1,), (0,)} - {(n,)}
        inputs += [(z((n, t)), z((n, t)), z(shape), 1.0) for shape in sorted(advantage_shapes)]
        as_lists = [z((n, t)).tolist(), z((n, t)).tolist(), z(n).tolist()]
        for position in range(3):
            arrays = [z((n, t)), z((n, t)), z(n)]
            arrays[position] = as_lists[position]
            inputs.append((*arrays, 1.0))
        inputs.append((*as_lists, 1.0))
    return inputs


def test_the_second_policy_loss_matches_the_main_version_on_the_edge_inputs():
    skip_unless_second_is_written("policy_loss")
    compare_policy_loss_on_the_edge_inputs(impl.second_version("policy_loss"), main.policy_loss)


@pytest.mark.parametrize("version", VERSIONS)
def test_each_policy_loss_gives_the_number_found_with_plain_loops(version):
    function = function_under_test("policy_loss", version)
    compare_policy_loss_on_the_edge_inputs(function, by_loops)
    compare_policy_loss(function, by_loops)                      # and on the 300 random inputs above


@pytest.mark.parametrize("version", VERSIONS)
def test_each_policy_loss_refuses_wrong_shapes_norm_consts_and_lists(version):
    function = function_under_test("policy_loss", version)
    inputs = refused_loss_inputs()
    assert len(inputs) > 110
    accepted = [(np.shape(logprobs), np.shape(mask), np.shape(advantages), norm_const, type(mask).__name__)
                for logprobs, mask, advantages, norm_const in inputs
                if not is_refused(function, logprobs, mask, advantages, norm_const)]
    assert accepted == []


def test_the_loss_edge_inputs_hold_what_they_claim():
    inputs = loss_edge_inputs()
    assert len(inputs) == 16 * 6 * 4 * 3
    losses = [by_loops(*case[:4]) for case in inputs]
    must_be_zero = [case[4] == "zeros" or case[5] == "zeros" for case in inputs]
    assert all(loss == 0.0 for loss, zero in zip(losses, must_be_zero) if zero)
    assert sum(must_be_zero) == 16 * 3 * (4 + 6 - 1)
    assert sum(1 for loss in losses if loss > 0.01) > 150
    assert sum(1 for loss in losses if loss < -0.01) > 150
    assert {case[1].dtype.kind for case in inputs} == {"f", "i", "b"}


def _loss_shapes_not_checked(logprobs, mask, advantages, norm_const):
    """Wrong on purpose: only norm_const is checked; numpy stretches a wrong shape or raises its own error."""
    if not norm_const > 0:
        raise ValueError("norm_const")
    try:
        per_episode = (np.asarray(mask) * np.asarray(logprobs)).sum(-1)
        return -(np.asarray(advantages) * per_episode).sum() / norm_const
    except Exception as error:                # an error of numpy, turned into the one the contract names
        raise ValueError("numpy refused the shapes") from error


def _loss_norm_const_not_checked(logprobs, mask, advantages, norm_const):
    """Wrong on purpose: a norm_const of 0 or below is used (numpy divides by 0 with a warning only)."""
    with np.errstate(all="ignore"):
        return main.policy_loss(logprobs, mask, advantages, 1.0) / np.float64(norm_const)


def _loss_lists_converted(logprobs, mask, advantages, norm_const):
    """Wrong since the decision of 2026-11-17: plain lists are turned into arrays and accepted."""
    return main.policy_loss(np.asarray(logprobs), np.asarray(mask), np.asarray(advantages), norm_const)


@pytest.mark.parametrize("wrong, n_accepted", [(_loss_shapes_not_checked, 20),
                                               (_loss_norm_const_not_checked, 20),
                                               (_loss_lists_converted, 16)],
                         ids=lambda value: value.__name__.lstrip("_") if callable(value) else str(value))
def test_the_refusal_check_fails_for_a_wrong_policy_loss(wrong, n_accepted):
    accepted = [1 for args in refused_loss_inputs() if not is_refused(wrong, *args)]
    assert len(accepted) >= n_accepted


@pytest.mark.parametrize("wrong", WRONG_LOSSES, ids=lambda function: function.__name__.lstrip("_"))
def test_the_edge_comparison_fails_for_a_wrong_policy_loss(wrong):
    with pytest.raises(AssertionError):
        compare_policy_loss_on_the_edge_inputs(wrong, main.policy_loss)
    with pytest.raises(AssertionError):
        compare_policy_loss_on_the_edge_inputs(wrong, by_loops)
