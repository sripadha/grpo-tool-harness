"""reward(): hand-made cases and properties, run on the main version and on the second version.

Every expected number below was worked out by hand from the contract in the function's docstring
(accessdesk/env/reward.py), not by running any implementation. Each test takes the function under test as
the argument `reward`: the main version, and the second version of accessdesk/_second once it is written
(until then those tests are skipped).
"""
from __future__ import annotations

import dataclasses

import pytest

from accessdesk.config import RunConfig
from tests.conftest import VERSIONS, function_under_test

DEFAULT = RunConfig()     # pass 100; 5 per malformed call, at most 15; 2 per turn over the budget, at most 10
SWITCH_ON = dataclasses.replace(DEFAULT, penalise_malformed_on_fail=True)
# Other numbers everywhere, to show that the function reads the config and holds no number of its own.
OTHER = RunConfig(pass_reward=50, malformed_penalty=7, malformed_cap=10, turn_penalty=3, turn_cap=4)
OTHER_SWITCH_ON = dataclasses.replace(OTHER, penalise_malformed_on_fail=True)
NO_PENALTY = RunConfig(malformed_penalty=0, malformed_cap=0, turn_penalty=0, turn_cap=0)


@pytest.fixture(params=VERSIONS)
def reward(request):
    """The function under test: the main version, then the second version."""
    return function_under_test("reward", request.param)


CASES = [
    # (label, passed, n_malformed, n_turns, turn_budget, cfg, the right reward)
    # --- fail, the switch off (the default): always 0 ---
    ("fail, clean and short", False, 0, 3, 4, DEFAULT, 0),
    ("fail, switch off, 1 malformed call", False, 1, 5, 5, DEFAULT, 0),
    ("fail, switch off, 2 malformed calls and one turn over", False, 2, 6, 5, DEFAULT, 0),
    ("fail, switch off, 3 malformed calls", False, 3, 8, 5, DEFAULT, 0),
    ("fail, switch off, 5 malformed calls", False, 5, 8, 5, DEFAULT, 0),
    ("fail at the turn limit, all malformed", False, 8, 8, 4, DEFAULT, 0),
    ("fail under other numbers", False, 1, 9, 4, OTHER, 0),
    # --- fail, the switch on: minus 5 per malformed call, at most 15; turns cost a fail nothing ---
    ("fail, switch on, no malformed call: 0", False, 0, 4, 5, SWITCH_ON, 0),
    ("fail, switch on, 1 malformed call: -5", False, 1, 5, 5, SWITCH_ON, -5),
    ("fail, switch on, 2 malformed calls: -10", False, 2, 6, 5, SWITCH_ON, -10),
    ("fail, switch on, 3 malformed calls: -15", False, 3, 8, 5, SWITCH_ON, -15),
    ("fail, switch on, 5 malformed calls: the cap of 15 holds", False, 5, 8, 5, SWITCH_ON, -15),
    ("fail, switch on, all 8 turns malformed: the cap of 15 holds", False, 8, 8, 4, SWITCH_ON, -15),
    ("fail, switch on, four turns over the budget and no malformed call: 0", False, 0, 8, 4, SWITCH_ON, 0),
    ("fail, switch on, other numbers, 1 malformed call: -7", False, 1, 9, 4, OTHER_SWITCH_ON, -7),
    ("fail, switch on, other numbers, 2 malformed calls: the cap of 10 holds", False, 2, 6, 4,
     OTHER_SWITCH_ON, -10),
    # --- pass, no penalty ---
    ("pass on the shortest path (budget is path + 1)", True, 0, 4, 5, DEFAULT, 100),
    ("pass with one extra read, exactly on the budget", True, 0, 5, 5, DEFAULT, 100),
    ("pass in fewer turns than the budget earns nothing extra", True, 0, 3, 5, DEFAULT, 100),
    ("pass with the switch on is not changed by the switch", True, 0, 4, 4, SWITCH_ON, 100),
    # --- pass, malformed calls only (turns kept inside the budget) ---
    ("one malformed call: 100 - 5", True, 1, 5, 5, DEFAULT, 95),
    ("two malformed calls: 100 - 10", True, 2, 5, 5, DEFAULT, 90),
    ("three malformed calls: 100 - 15", True, 3, 5, 5, DEFAULT, 85),
    ("four malformed calls: the cap of 15 holds", True, 4, 5, 5, DEFAULT, 85),
    ("six malformed calls: the cap of 15 holds", True, 6, 4, 4, DEFAULT, 85),
    # --- pass, extra turns only ---
    ("one turn over the budget: 100 - 2", True, 0, 6, 5, DEFAULT, 98),
    ("two turns over: 100 - 4", True, 0, 6, 4, DEFAULT, 96),
    ("four turns over (the most with 8 turns and a budget of 4): 100 - 8", True, 0, 8, 4, DEFAULT, 92),
    ("five turns over: 100 - 10", True, 0, 10, 5, DEFAULT, 90),
    ("six turns over: the cap of 10 holds", True, 0, 11, 5, DEFAULT, 90),
    ("twenty turns over: the cap of 10 holds", True, 0, 24, 4, DEFAULT, 90),
    # --- pass, both penalties ---
    ("one malformed call that also puts the episode one turn over: 100 - 5 - 2", True, 1, 6, 5, DEFAULT, 93),
    ("two malformed and three over: 100 - 10 - 6", True, 2, 8, 5, DEFAULT, 84),
    ("lowest pass with a budget of 5 and 8 turns: 100 - 15 - 6", True, 3, 8, 5, DEFAULT, 79),
    ("lowest pass with a budget of 4 and 8 turns: 100 - 15 - 8", True, 4, 8, 4, DEFAULT, 77),
    ("both caps at once: 100 - 15 - 10", True, 9, 20, 5, DEFAULT, 75),
    # --- pass, the switch on: the same numbers as with the switch off ---
    ("pass, switch on, one malformed and one over: 100 - 5 - 2, as with the switch off", True, 1, 6, 5,
     SWITCH_ON, 93),
    ("pass, switch on, lowest pass with a budget of 4: 100 - 15 - 8, as with the switch off", True, 4, 8, 4,
     SWITCH_ON, 77),
    # --- the numbers come from the config ---
    ("other numbers, clean: 50", True, 0, 4, 4, OTHER, 50),
    ("other numbers, one malformed and one over: 50 - 7 - 3", True, 1, 5, 4, OTHER, 40),
    ("other numbers, both caps: 50 - 10 - 4", True, 2, 6, 4, OTHER, 36),
    ("other numbers, one over only: 50 - 3", True, 0, 5, 4, OTHER, 47),
    ("other numbers, one malformed only: 50 - 7", True, 1, 4, 4, OTHER, 43),
    ("other numbers, switch on, a pass is not changed: 50 - 7 - 3", True, 1, 5, 4, OTHER_SWITCH_ON, 40),
    ("penalties set to zero", True, 5, 8, 4, NO_PENALTY, 100),
]


@pytest.mark.parametrize("label, passed, n_malformed, n_turns, turn_budget, cfg, answer", CASES,
                         ids=[case[0] for case in CASES])
def test_reward_hand_made(reward, label, passed, n_malformed, n_turns, turn_budget, cfg, answer):
    result = reward(passed, n_malformed, n_turns, turn_budget, cfg)
    assert result == answer and type(result) is int


def test_the_hand_made_cases_reach_every_branch():
    fails = [case for case in CASES if not case[1]]
    passes = [case for case in CASES if case[1]]
    assert len(fails) >= 12 and len(passes) >= 20
    switch_on = [case for case in fails if case[5].penalise_malformed_on_fail]
    switch_off = [case for case in fails if not case[5].penalise_malformed_on_fail]
    for n_malformed in (0, 1, 3, 5):                    # the same fails with the switch off and on
        assert any(case[2] == n_malformed for case in switch_on)
        assert any(case[2] == n_malformed for case in switch_off)
    assert {case[6] for case in switch_off} == {0}
    assert {0, -5, -10, -15, -7} <= {case[6] for case in switch_on}
    assert any(c[2] * c[5].malformed_penalty > c[5].malformed_cap for c in switch_on)   # the cap bites a fail
    assert any(case[5].penalise_malformed_on_fail and case[2] > 0 for case in passes)   # a pass, switch on
    assert any(case[5] != RunConfig() and case[6] not in (0, 100) for case in passes)   # config is read
    assert any(c[2] * c[5].malformed_penalty > c[5].malformed_cap for c in passes)      # malformed cap bites
    assert any((c[3] - c[4]) * c[5].turn_penalty > c[5].turn_cap for c in passes)       # turn cap bites
    assert any(c[3] < c[4] for c in passes)                                             # under the budget


def test_reward_is_a_whole_number(reward):
    for passed, n_malformed, n_turns in ((True, 0, 4), (True, 1, 7), (False, 2, 8)):
        assert type(reward(passed, n_malformed, n_turns, 5, DEFAULT)) is int
        assert type(reward(passed, n_malformed, n_turns, 5, SWITCH_ON)) is int


def test_reward_table_for_the_default_config_typed_by_hand(reward):
    # rows: turns over the budget 0..6; columns: malformed calls 0..4
    table = [
        [100, 95, 90, 85, 85],
        [98, 93, 88, 83, 83],
        [96, 91, 86, 81, 81],
        [94, 89, 84, 79, 79],
        [92, 87, 82, 77, 77],
        [90, 85, 80, 75, 75],
        [90, 85, 80, 75, 75],
    ]
    fail_with_the_switch_on = [0, -5, -10, -15, -15]      # by malformed calls 0..4, whatever the turns are
    for over, row in enumerate(table):
        for n_malformed, wanted in enumerate(row):
            for budget in (4, 5):
                assert reward(True, n_malformed, budget + over, budget, DEFAULT) == wanted
                assert reward(True, n_malformed, budget + over, budget, SWITCH_ON) == wanted
                assert reward(False, n_malformed, budget + over, budget, DEFAULT) == 0
                assert reward(False, n_malformed, budget + over, budget, SWITCH_ON) == \
                    fail_with_the_switch_on[n_malformed]


def test_the_lowest_pass_that_an_episode_can_reach_is_77_or_79(reward):
    """PLAN 5.2: with the 8-turn limit, 77 for the three-turn outcomes and 79 for the four-turn outcomes."""
    for shortest_path, lowest in ((3, 77), (4, 79)):
        budget = shortest_path + 1
        reachable = [reward(True, n_malformed, n_turns, budget, DEFAULT)
                     for n_turns in range(shortest_path, DEFAULT.turn_limit + 1)
                     for n_malformed in range(0, n_turns - shortest_path + 1)]
        assert min(reachable) == lowest and max(reachable) == 100
        assert all(value > 0 for value in reachable)


@pytest.mark.parametrize("switch", [False, True], ids=["switch off", "switch on"])
def test_every_pass_is_above_every_fail(reward, switch):
    """Over every (n_malformed, n_turns) that an episode of the harness can end with, for both budgets.

    An episode has 1 to cfg.turn_limit turns. A fail can have any number of malformed messages, up to all
    of its turns. A pass was closed by a well-formed call, so at most all turns but one are malformed
    (this is more than a real pass can have, which also needs its reads and its write).
    """
    cfg = dataclasses.replace(DEFAULT, penalise_malformed_on_fail=switch)
    turns = range(1, cfg.turn_limit + 1)
    passes = [reward(True, n_malformed, n_turns, budget, cfg)
              for budget in (4, 5) for n_turns in turns for n_malformed in range(0, n_turns)]
    fails = [reward(False, n_malformed, n_turns, budget, cfg)
             for budget in (4, 5) for n_turns in turns for n_malformed in range(0, n_turns + 1)]
    assert len(passes) == 2 * 36 and len(fails) == 2 * 44
    assert min(passes) > max(fails)
    assert (min(passes), max(passes)) == (77, 100)
    assert (min(fails), max(fails)) == ((-15, 0) if switch else (0, 0))


def test_more_malformed_calls_or_more_turns_never_raise_the_reward(reward):
    for cfg in (DEFAULT, SWITCH_ON):
        for passed in (True, False):
            for budget in (4, 5):
                for n_turns in range(1, 12):
                    for n_malformed in range(0, 8):
                        here = reward(passed, n_malformed, n_turns, budget, cfg)
                        assert reward(passed, n_malformed + 1, n_turns, budget, cfg) <= here
                        assert reward(passed, n_malformed, n_turns + 1, budget, cfg) <= here


def test_the_switch_changes_the_fails_and_nothing_else(reward):
    for n_malformed in range(0, 6):
        for n_turns in range(1, 10):
            for budget in (4, 5):
                args = (n_malformed, n_turns, budget)
                assert reward(True, *args, SWITCH_ON) == reward(True, *args, DEFAULT)
                assert reward(False, *args, DEFAULT) == 0
                assert reward(False, *args, SWITCH_ON) == -min(15, 5 * n_malformed)


@pytest.mark.parametrize("field, value, wanted", [
    ("pass_reward", 60, 60 - 5 - 2), ("malformed_penalty", 1, 100 - 1 - 2), ("turn_penalty", 9, 100 - 5 - 9),
    ("malformed_cap", 3, 100 - 3 - 2), ("turn_cap", 1, 100 - 5 - 1),
])
def test_each_reward_number_is_read_from_the_config(reward, field, value, wanted):
    cfg = dataclasses.replace(DEFAULT, **{field: value})
    assert reward(True, 1, 6, 5, cfg) == wanted          # one malformed call, one turn over the budget


@pytest.mark.parametrize("field, value, wanted", [
    ("malformed_penalty", 1, -2), ("malformed_cap", 3, -3), ("pass_reward", 60, -10),
    ("turn_penalty", 9, -10), ("turn_cap", 1, -10),
])
def test_the_penalty_of_a_fail_reads_only_the_two_malformed_numbers(reward, field, value, wanted):
    cfg = dataclasses.replace(SWITCH_ON, **{field: value})
    assert reward(False, 2, 7, 5, cfg) == wanted         # two malformed calls, two turns over the budget


# --- cases that the comparison of the two versions decided (2026-11-17) -------------------------------------
def test_more_malformed_calls_than_turns_is_not_refused(reward):
    # No real episode has more malformed messages than messages, and the contract did not say what to do
    # with such counts. The first draft of the second version raised ValueError; the main version used the
    # formula. Decided: the two counts are not compared with each other, the formula applies as written.
    #   2 malformed calls, 1 turn, budget 3: 100 - min(15, 5 * 2) - min(10, 2 * max(0, 1 - 3)) = 100 - 10 - 0
    assert reward(True, 2, 1, 3, DEFAULT) == 90
    assert reward(False, 2, 1, 3, DEFAULT) == 0
    assert reward(False, 2, 1, 3, SWITCH_ON) == -10
    #   9 malformed calls, 0 turns: the cap of 15.
    assert reward(True, 9, 0, 3, DEFAULT) == 85
    assert reward(False, 9, 0, 3, SWITCH_ON) == -15


@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize("cfg", [DEFAULT, SWITCH_ON, OTHER_SWITCH_ON], ids=["default", "switch_on", "other"])
@pytest.mark.parametrize("n_malformed, n_turns", [(-1, 4), (-3, 0), (0, -1), (2, -5), (-1, -1)])
def test_a_negative_count_is_refused(reward, passed, cfg, n_malformed, n_turns):
    # The contract did not say it. The main version used the formula: with -1 malformed calls a pass was
    # 100 - min(15, -5) = 105, above the pass reward, and a fail with the switch on was +5, although "a fail
    # is never above 0". The second version raised ValueError. Decided: ValueError, for a pass and for a fail.
    with pytest.raises(ValueError):
        reward(passed, n_malformed, n_turns, 4, cfg)


def test_the_turn_budget_is_not_checked(reward):
    # Decided with the cases above: only the two counts are checked. A budget of 0 or below is used as it is.
    #   budget 0, 3 turns: 3 turns over, 2 * 3 = 6.       budget -2, 3 turns: 5 over, 2 * 5 = 10 = the cap.
    assert reward(True, 0, 3, 0, DEFAULT) == 94
    assert reward(True, 0, 3, -2, DEFAULT) == 90
    assert reward(True, 0, 0, 0, DEFAULT) == 100          # zero counts are not negative: allowed
    assert reward(False, 0, 0, 0, SWITCH_ON) == 0
