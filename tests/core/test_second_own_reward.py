"""Hand-made tests for the second version of reward(), with expected values worked out from the contract."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from accessdesk._second.reward import reward
from accessdesk.config import DEFAULT, RunConfig

SWITCH_ON = dataclasses.replace(DEFAULT, penalise_malformed_on_fail=True)


# ---- pass, default config (100, -5 per malformed up to 15, -2 per turn over budget up to 10) ----

@pytest.mark.parametrize(
    "n_malformed, n_turns, turn_budget, expected",
    [
        (0, 4, 4, 100),   # clean, exactly on budget: 100 - 0 - 0
        (0, 2, 4, 100),   # under budget earns nothing extra: max(0, 2-4)=0 -> 100
        (0, 5, 4, 98),    # one turn over: 100 - 2*1
        (0, 9, 4, 90),    # five over: 100 - min(10, 10) = 90
        (0, 12, 4, 90),   # eight over: 100 - min(10, 16) = 90 (turn cap)
        (1, 4, 4, 95),    # one malformed: 100 - 5
        (2, 4, 4, 90),    # 100 - 10
        (3, 4, 4, 85),    # 100 - 15 (exactly at the cap)
        (4, 6, 4, 81),    # 100 - min(15, 20)=15 - min(10, 2*2)=4 -> 81
        (2, 7, 4, 84),    # 100 - 10 - 6 = 84
        (6, 20, 4, 75),   # both caps: 100 - 15 - 10 = 75 (lowest pass with defaults)
    ],
)
def test_pass_default(n_malformed: int, n_turns: int, turn_budget: int, expected: int) -> None:
    got = reward(True, n_malformed, n_turns, turn_budget, DEFAULT)
    assert got == expected
    assert type(got) is int


@pytest.mark.parametrize("n_malformed, n_turns, turn_budget, expected", [(0, 4, 4, 100), (3, 9, 4, 75)])
def test_pass_is_the_same_with_the_switch_on(n_malformed: int, n_turns: int, turn_budget: int,
                                             expected: int) -> None:
    # "Pass ... the same with the switch on and off"
    assert reward(True, n_malformed, n_turns, turn_budget, SWITCH_ON) == expected
    assert reward(True, n_malformed, n_turns, turn_budget, DEFAULT) == expected


# ---- fail ----

@pytest.mark.parametrize("n_malformed, n_turns", [(0, 1), (0, 8), (3, 8), (8, 8)])
def test_fail_switch_off_is_zero(n_malformed: int, n_turns: int) -> None:
    got = reward(False, n_malformed, n_turns, 3, DEFAULT)
    assert got == 0
    assert type(got) is int


@pytest.mark.parametrize(
    "n_malformed, n_turns, expected",
    [
        (0, 3, 0),     # -min(15, 0) = 0
        (1, 3, -5),    # -min(15, 5)
        (2, 3, -10),   # -min(15, 10)
        (3, 3, -15),   # -min(15, 15)
        (7, 8, -15),   # -min(15, 35) = -15
    ],
)
def test_fail_switch_on_loses_only_the_malformed_penalty(n_malformed: int, n_turns: int, expected: int) -> None:
    assert reward(False, n_malformed, n_turns, 3, SWITCH_ON) == expected


def test_fail_never_gets_the_turn_penalty() -> None:
    # 0 malformed, far over budget: the turn penalty never applies to a fail -> 0 with the switch on and off
    assert reward(False, 0, 20, 3, SWITCH_ON) == 0
    assert reward(False, 0, 20, 3, DEFAULT) == 0
    # 1 malformed, far over budget, switch on: -5 only, not -5 - 10
    assert reward(False, 1, 20, 3, SWITCH_ON) == -5


# ---- every number is read from cfg ----

def test_numbers_come_from_the_config() -> None:
    cfg = RunConfig(pass_reward=50, malformed_penalty=7, malformed_cap=20, turn_penalty=3, turn_cap=4,
                    penalise_malformed_on_fail=True)
    # pass: 50 - min(20, 7*2)=14 - min(4, 3*1)=3 -> 33
    assert reward(True, 2, 6, 5, cfg) == 33
    # pass: 50 - min(20, 21)=20 - min(4, 3*5)=4 -> 26
    assert reward(True, 3, 10, 5, cfg) == 26
    # fail with switch on: -min(20, 14) = -14
    assert reward(False, 2, 6, 5, cfg) == -14
    # fail with switch on: -min(20, 28) = -20
    assert reward(False, 4, 6, 5, cfg) == -20


def test_zero_penalties_and_caps() -> None:
    cfg = RunConfig(malformed_penalty=0, turn_penalty=0)
    assert reward(True, 5, 30, 3, cfg) == 100     # 100 - 0 - 0
    cfg = RunConfig(malformed_cap=0, turn_cap=0, penalise_malformed_on_fail=True)
    assert reward(True, 5, 30, 3, cfg) == 100     # caps of 0 -> min(0, x) = 0
    assert reward(False, 5, 30, 3, cfg) == 0      # -min(0, 25) = 0


def test_caps_reaching_pass_reward_follow_the_formula() -> None:
    # The contract only promises pass > fail while the caps sum below pass_reward; outside that the formula holds.
    cfg = RunConfig(pass_reward=20, malformed_cap=15, turn_cap=10)
    assert reward(True, 3, 10, 4, cfg) == -5      # 20 - 15 - 10


def test_default_range_holds_on_a_grid() -> None:
    # Contract: rewards lie in [0, 100] with the switch off and in [-15, 100] with it on; every pass > every fail.
    for cfg, low in ((DEFAULT, 0), (SWITCH_ON, -15)):
        passes, fails = [], []
        for n_turns in range(0, 12):
            for n_malformed in range(0, n_turns + 1):
                for budget in range(1, 8):
                    passes.append(reward(True, n_malformed, n_turns, budget, cfg))
                    fails.append(reward(False, n_malformed, n_turns, budget, cfg))
        assert min(passes + fails) >= low
        assert max(passes + fails) <= 100
        assert min(passes) > max(fails)
        assert max(fails) <= 0


# ---- choices where the contract is silent ----

def test_numpy_inputs_give_a_python_int() -> None:
    got = reward(np.bool_(True), np.int64(1), np.int64(5), np.int32(4), DEFAULT)   # 100 - 5 - 2 = 93
    assert got == 93
    assert type(got) is int
    assert reward(np.bool_(False), np.int64(1), np.int64(5), np.int32(4), SWITCH_ON) == -5


@pytest.mark.parametrize("bad", ["True", None, 0.5, [True]])
def test_passed_must_be_a_truth_value(bad: object) -> None:
    with pytest.raises(TypeError):
        reward(bad, 0, 3, 3, DEFAULT)  # type: ignore[arg-type]


def test_float_counts_are_refused() -> None:
    with pytest.raises(TypeError):
        reward(True, 1.0, 3, 3, DEFAULT)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reward(True, 0, 3, 2.5, DEFAULT)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_malformed, n_turns", [(-1, 3), (0, -1)])
def test_impossible_counts_are_refused(n_malformed: int, n_turns: int) -> None:
    for passed in (True, False):
        with pytest.raises(ValueError):
            reward(passed, n_malformed, n_turns, 3, SWITCH_ON)


def test_more_malformed_than_turns_follows_the_formula() -> None:
    # impossible in a real episode, but not refused: pass 100 - min(15, 10) - min(10, 2*max(0, 1-3)) = 90
    assert reward(True, 2, 1, 3, DEFAULT) == 90
    assert reward(False, 2, 1, 3, SWITCH_ON) == -10   # -min(15, 10)


def test_turn_budget_is_not_range_checked() -> None:
    # budget 0: 3 turns over -> 100 - 6 = 94
    assert reward(True, 0, 3, 0, DEFAULT) == 94
