"""The reward of one episode, computed after the episode has ended.

This is the main version of `reward`. The rest of the code reaches it through `accessdesk.impl.get("reward")`.
An independent second version, written from the docstring below only, is in `accessdesk/_second/reward.py`;
tests compare the two.
Tests: `.venv/bin/python -m pytest tests/core/test_reward.py tests/core/test_differential_reward.py`
"""
from __future__ import annotations

from accessdesk.config import RunConfig


def reward(passed: bool, n_malformed: int, n_turns: int, turn_budget: int, cfg: RunConfig) -> int:
    """The reward in whole hundredths (an int; 100 means 1.00).

    Arguments:
        passed       the verifier's answer for the final state
        n_malformed  the number of assistant messages that were not one valid tool call
        n_turns      the number of assistant messages of the episode (malformed ones included)
        turn_budget  the shortest correct path for this family plus one turn
        cfg          the run config; every number below is read from it, none is written in this function

    Fail (passed is False):
        cfg.penalise_malformed_on_fail is False (the default):  0
        cfg.penalise_malformed_on_fail is True:   -min(cfg.malformed_cap, cfg.malformed_penalty * n_malformed)
        With the switch off, penalties apply to passing episodes only, so no penalty can make giving up early
        look better than trying. With the switch on, a failing episode loses the malformed-call penalty (and
        only that one: the turn penalty never applies to a fail). A fail is never above 0.

    Pass (passed is True), the same with the switch on and off:
        cfg.pass_reward
        - min(cfg.malformed_cap, cfg.malformed_penalty * n_malformed)
        - min(cfg.turn_cap, cfg.turn_penalty * max(0, n_turns - turn_budget))

    With the default config: a pass is 100, minus 5 per malformed call (at most 15), minus 2 per turn beyond
    the budget (at most 10). Turns below the budget earn nothing extra. A fail is 0; with the switch on it is
    0, -5, -10 or -15. So rewards lie between 0 and 100 with the switch off and between -15 and 100 with it
    on, and every pass is above every fail as long as the two caps together are below cfg.pass_reward.

    Raises ValueError if n_malformed or n_turns is negative: no episode has such a count, and a negative
    n_malformed would turn the penalty into a bonus. Nothing else is checked: n_malformed may be above
    n_turns (no real episode has that either), turn_budget may be any whole number, and the config numbers
    are used as they are. In all these cases the formulas above apply as written. (The ranges stated above
    hold for config numbers that are not negative.)
    passed is a bool and the three counts are Python ints. Arguments of other types are outside this
    contract: what the call does then is not defined.
    """
    # A count below zero is a mistake of the caller. It is refused, because the formulas below would turn
    # a negative number of malformed calls into a bonus.
    if n_malformed < 0 or n_turns < 0:
        raise ValueError(f"a count cannot be negative: n_malformed={n_malformed}, n_turns={n_turns}")

    # The penalty for malformed calls: a fixed amount for each one, but never more than the cap.
    malformed_part = min(cfg.malformed_cap, cfg.malformed_penalty * n_malformed)

    if not passed:
        # A fail earns nothing. With the switch on it also loses the malformed-call penalty.
        if cfg.penalise_malformed_on_fail:
            return -malformed_part
        return 0

    # The penalty for turns: only the turns beyond the budget count, and again never more than the cap.
    turns_over_budget = max(0, n_turns - turn_budget)
    turn_part = min(cfg.turn_cap, cfg.turn_penalty * turns_over_budget)

    # A pass: the full reward minus the two penalties.
    return cfg.pass_reward - malformed_part - turn_part
