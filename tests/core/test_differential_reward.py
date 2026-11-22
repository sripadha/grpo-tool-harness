"""The second version of reward() against the main version, on a grid of episodes and several configs.

The comparison is skipped while the second version (accessdesk/_second/reward.py) is not written.
The grid and the comparison itself are tested here in any case, with wrong functions.

Two sets of inputs:
  - the grid: 2,912 episodes under 7 configs. The two versions must give the same int.
  - the edge grid: every pair of counts from 0 to 14 (so every pair that an episode can reach, and the
    pairs with more malformed calls than turns, which no episode reaches), with budgets from -1 to 9, a pass
    and a fail, under 8 sets of reward numbers with the switch off and on; and negative counts, which both
    versions must refuse. Here the right number is also worked out a third way, by counting one malformed
    call and one turn at a time, so a mistake that the two versions share would show.
"""
from __future__ import annotations

import dataclasses
import functools

import pytest

from accessdesk import impl
from accessdesk.config import DEFAULT, RunConfig
from accessdesk.env.reward import reward as main_reward
from tests.conftest import VERSIONS, function_under_test, skip_unless_second_is_written

# In every config the two caps together stay below the pass reward, as in the default config.
CONFIGS = [
    DEFAULT,
    dataclasses.replace(DEFAULT, penalise_malformed_on_fail=True),
    dataclasses.replace(DEFAULT, turn_limit=12),
    RunConfig(pass_reward=50, malformed_penalty=7, malformed_cap=10, turn_penalty=3, turn_cap=4),
    RunConfig(pass_reward=100, malformed_penalty=0, malformed_cap=0, turn_penalty=0, turn_cap=0),
    RunConfig(pass_reward=100, malformed_penalty=20, malformed_cap=30, turn_penalty=1, turn_cap=50),
    RunConfig(pass_reward=50, malformed_penalty=7, malformed_cap=10, turn_penalty=3, turn_cap=4,
              penalise_malformed_on_fail=True),
]


def grid() -> list[tuple]:
    return [(passed, n_malformed, n_turns, turn_budget, cfg)
            for cfg in CONFIGS
            for passed in (True, False)
            for turn_budget in (4, 5)
            for n_turns in range(1, 14)
            for n_malformed in range(0, 8)]


def disagreements(other, main=main_reward) -> list[tuple]:
    """The episodes of the grid on which `other` does not give the int of the main version."""
    wrong = []
    for args in grid():
        got, wanted = other(*args), main(*args)
        if got != wanted or type(got) is not int:
            wrong.append((args, got, wanted))
    return wrong


def test_the_second_reward_agrees_with_the_main_version():
    skip_unless_second_is_written("reward")
    wrong = disagreements(impl.second_version("reward"))
    assert wrong == [], f"{len(wrong)} of {len(grid())} episodes differ; the first one: {wrong[0]}"


# --- the comparison itself, tested today ---------------------------------------------------------
def test_the_grid_is_large_and_holds_every_kind_of_episode():
    episodes = grid()
    assert len(episodes) == 7 * 2 * 2 * 13 * 8
    values = {main_reward(*args) for args in episodes}
    assert {0, 100, 95, 98, 75, 77, 79} <= values
    assert {-5, -10, -15, -7} <= values                       # fails with the switch on
    assert min(values) == -15


def test_the_main_version_agrees_with_itself():
    assert disagreements(main_reward) == []


def no_caps(passed, n_malformed, n_turns, turn_budget, cfg):
    if not passed:
        return 0
    turn_part = cfg.turn_penalty * max(0, n_turns - turn_budget)
    return cfg.pass_reward - cfg.malformed_penalty * n_malformed - turn_part


def default_numbers_only(passed, n_malformed, n_turns, turn_budget, cfg):
    return main_reward(passed, n_malformed, n_turns, turn_budget, DEFAULT)


def both_penalties_on_fail(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong on purpose: a fail is a pass minus the pass reward, so it loses the turn penalty too."""
    return main_reward(True, n_malformed, n_turns, turn_budget, cfg) - (0 if passed else cfg.pass_reward)


def switch_ignored(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong on purpose: a fail is 0 with the switch on as well."""
    return main_reward(passed, n_malformed, n_turns, turn_budget, cfg) if passed else 0


def switch_always_on(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong on purpose: a fail loses the malformed-call penalty with the switch off as well."""
    return main_reward(passed, n_malformed, n_turns, turn_budget,
                       dataclasses.replace(cfg, penalise_malformed_on_fail=True))


def fail_penalty_without_cap(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong on purpose: with the switch on, a fail loses the penalty of every malformed call."""
    if passed or not cfg.penalise_malformed_on_fail:
        return main_reward(passed, n_malformed, n_turns, turn_budget, cfg)
    return -cfg.malformed_penalty * n_malformed


def switch_changes_a_pass(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong on purpose: with the switch on, a pass loses the malformed-call penalty twice."""
    value = main_reward(passed, n_malformed, n_turns, turn_budget, cfg)
    if passed and cfg.penalise_malformed_on_fail:
        value -= min(cfg.malformed_cap, cfg.malformed_penalty * n_malformed)
    return value


def bonus_for_short_episodes(passed, n_malformed, n_turns, turn_budget, cfg):
    return main_reward(passed, n_malformed, n_turns, turn_budget, cfg) + (
        cfg.turn_penalty * max(0, turn_budget - n_turns) if passed else 0)


def as_a_fraction(passed, n_malformed, n_turns, turn_budget, cfg):
    return main_reward(passed, n_malformed, n_turns, turn_budget, cfg) / 100


WRONG_FUNCTIONS = [no_caps, default_numbers_only, both_penalties_on_fail, switch_ignored, switch_always_on,
                   fail_penalty_without_cap, switch_changes_a_pass, bonus_for_short_episodes, as_a_fraction]


@pytest.mark.parametrize("wrong_function", WRONG_FUNCTIONS, ids=[f.__name__ for f in WRONG_FUNCTIONS])
def test_the_comparison_catches_wrong_functions(wrong_function):
    assert len(disagreements(wrong_function)) >= 20


# ---------------------------------------------------------------------------
# The edge grid: every pair of counts, more configs, and the number worked out a third way
# ---------------------------------------------------------------------------
# Sets of reward numbers: (pass_reward, malformed_penalty, malformed_cap, turn_penalty, turn_cap).
EDGE_NUMBERS = [
    (100, 5, 15, 2, 10),                # the default
    (100, 4, 15, 3, 10),                # a cap that is not a multiple of the penalty: 4, 8, 12, 15
    (100, 20, 7, 9, 5),                 # a cap below one penalty: the first malformed call costs the cap
    (100, 0, 15, 0, 10),                # penalties of 0: nothing is ever lost
    (100, 5, 0, 2, 0),                  # caps of 0: nothing is ever lost
    (20, 10, 15, 5, 10),                # the caps together above the pass reward: a pass can be below 0
    (1, 1, 1, 1, 1),                    # the smallest numbers
    (10_000, 333, 1_000_000, 77, 1_000_000),      # caps that no episode reaches
]
EDGE_CONFIGS = [RunConfig(pass_reward=numbers[0], malformed_penalty=numbers[1], malformed_cap=numbers[2],
                          turn_penalty=numbers[3], turn_cap=numbers[4], penalise_malformed_on_fail=switch)
                for numbers in EDGE_NUMBERS for switch in (False, True)]
EDGE_COUNTS = range(0, 15)
EDGE_BUDGETS = range(-1, 10)
NEGATIVE_COUNTS = [(-1, 4), (-1, 0), (-7, 3), (0, -1), (3, -1), (2, -9), (-1, -1), (-5, -2)]


@functools.lru_cache(maxsize=None)
def edge_grid() -> list[tuple]:
    """Every (passed, n_malformed, n_turns, turn_budget, cfg) of the edge grid; the negative counts last."""
    episodes = [(passed, n_malformed, n_turns, turn_budget, cfg)
                for cfg in EDGE_CONFIGS
                for passed in (True, False)
                for turn_budget in EDGE_BUDGETS
                for n_turns in EDGE_COUNTS
                for n_malformed in EDGE_COUNTS]
    refused = [(passed, n_malformed, n_turns, turn_budget, cfg)
               for cfg in EDGE_CONFIGS
               for passed in (True, False)
               for turn_budget in (0, 4)
               for n_malformed, n_turns in NEGATIVE_COUNTS]
    return episodes + refused


def by_counting(passed, n_malformed, n_turns, turn_budget, cfg):
    """The contract's number worked out without min(), max() or a product: one malformed call and one turn
    over the budget at a time, each one costing its penalty until the cap is reached. It is the same number
    as long as no penalty and no cap is negative (true for every config of this file)."""
    if n_malformed < 0 or n_turns < 0:
        raise ValueError("a negative count")
    malformed_loss = 0
    for _ in range(n_malformed):
        malformed_loss += cfg.malformed_penalty
        if malformed_loss >= cfg.malformed_cap:
            malformed_loss = cfg.malformed_cap
            break
    if not passed:
        return -malformed_loss if cfg.penalise_malformed_on_fail else 0
    turn_loss = 0
    for _ in range(turn_budget + 1, n_turns + 1):               # the turns after the last turn of the budget
        turn_loss += cfg.turn_penalty
        if turn_loss >= cfg.turn_cap:
            turn_loss = cfg.turn_cap
            break
    return cfg.pass_reward - malformed_loss - turn_loss


def result_of(function, *args) -> object:
    """What a call gives: the reward, or the name of the error that it raised."""
    try:
        return function(*args)
    except Exception as error:                      # any error is a result here, so that it can be compared
        return type(error).__name__


@functools.lru_cache(maxsize=None)
def edge_results(function) -> list[object]:
    """The result of `function` for every episode of the edge grid (kept, because it is asked for often)."""
    return [result_of(function, *args) for args in edge_grid()]


def edge_disagreements(other, main=main_reward) -> list[tuple]:
    """The episodes of the edge grid on which `other` does not give the result of `main`: the same int, or
    ValueError where `main` raises ValueError."""
    wrong = []
    for args, wanted in zip(edge_grid(), edge_results(main)):
        got = result_of(other, *args)
        if got != wanted or type(got) is not type(wanted) or not (type(got) is int or got == "ValueError"):
            wrong.append((args, got, wanted))
    return wrong


def test_the_second_reward_agrees_with_the_main_version_on_the_edge_grid():
    skip_unless_second_is_written("reward")
    wrong = edge_disagreements(impl.second_version("reward"))
    assert wrong == [], f"{len(wrong)} of {len(edge_grid())} episodes differ; the first one: {wrong[0]}"


@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_gives_the_number_found_by_counting(version):
    wrong = edge_disagreements(function_under_test("reward", version), main=by_counting)
    assert wrong == [], f"{len(wrong)} of {len(edge_grid())} episodes differ; the first one: {wrong[0]}"


def test_the_edge_grid_holds_what_it_claims():
    episodes = edge_grid()
    n_refused = 16 * 2 * 2 * 8
    assert len(episodes) == 16 * 2 * 11 * 15 * 15 + n_refused
    results = [result_of(main_reward, *args) for args in episodes]
    assert results[-n_refused:] == ["ValueError"] * n_refused            # every negative count is refused
    assert all(type(result) is int for result in results[:-n_refused])   # and nothing else is
    # Every pair of counts that an episode of up to 14 turns can reach, and the pairs that none can reach.
    pairs = {(n_malformed, n_turns) for _, n_malformed, n_turns, _, _ in episodes[:-n_refused]}
    assert {(m, t) for t in range(1, 15) for m in range(0, t + 1)} <= pairs
    assert sum(1 for m, t in pairs if m > t) == 105
    # The cap that is not a multiple of the penalty, the cap below one penalty, a pass below 0.
    by_config: dict[tuple, set[int]] = {}
    for (passed, _, _, _, cfg), result in zip(episodes[:-n_refused], results):
        key = (cfg.pass_reward, cfg.malformed_penalty, cfg.penalise_malformed_on_fail, passed)
        by_config.setdefault(key, set()).add(result)
    assert by_config[(100, 4, True, False)] == {0, -4, -8, -12, -15}
    assert by_config[(100, 20, True, False)] == {0, -7}
    assert by_config[(100, 20, False, False)] == {0}
    assert min(by_config[(20, 10, False, True)]) == 20 - 15 - 10
    assert by_config[(100, 0, False, True)] == {100} and by_config[(1, 1, True, True)] == {1, 0, -1}


def negative_counts_allowed(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong since the decision of 2026-11-17 (this is what the main version did before): the formula is
    used for a negative count as well, so -1 malformed calls are a bonus."""
    malformed_part = min(cfg.malformed_cap, cfg.malformed_penalty * n_malformed)
    if not passed:
        return -malformed_part if cfg.penalise_malformed_on_fail else 0
    turn_part = min(cfg.turn_cap, cfg.turn_penalty * max(0, n_turns - turn_budget))
    return cfg.pass_reward - malformed_part - turn_part


def more_malformed_than_turns_refused(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong since the decision of 2026-11-17 (the first draft of the second version did this)."""
    if n_malformed > n_turns:
        raise ValueError("more malformed calls than turns")
    return main_reward(passed, n_malformed, n_turns, turn_budget, cfg)


def budget_below_one_refused(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong since the decision of 2026-11-17: the turn budget is not checked."""
    if turn_budget < 1:
        raise ValueError("no budget")
    return main_reward(passed, n_malformed, n_turns, turn_budget, cfg)


def cap_reached_one_call_late(passed, n_malformed, n_turns, turn_budget, cfg):
    """Wrong on purpose: the penalty may pass the cap once before it is cut (wrong when the cap is not a
    multiple of the penalty)."""
    loss = cfg.malformed_penalty * n_malformed
    over = loss - cfg.malformed_cap
    if 0 < over < cfg.malformed_penalty:
        return main_reward(passed, n_malformed, n_turns, turn_budget, cfg) - (
            over if passed or cfg.penalise_malformed_on_fail else 0)
    return main_reward(passed, n_malformed, n_turns, turn_budget, cfg)


EDGE_WRONG_FUNCTIONS = WRONG_FUNCTIONS + [negative_counts_allowed, more_malformed_than_turns_refused,
                                          budget_below_one_refused, cap_reached_one_call_late]


@pytest.mark.parametrize("wrong_function", EDGE_WRONG_FUNCTIONS,
                         ids=[function.__name__ for function in EDGE_WRONG_FUNCTIONS])
def test_the_edge_comparison_catches_wrong_functions(wrong_function):
    assert len(edge_disagreements(wrong_function)) >= 20
    assert len(edge_disagreements(wrong_function, main=by_counting)) >= 20
