"""Second version of `reward`, written from the contract in the docstring only.

It was written by another model without reading the main version. Used by tests only.

Choices where the contract was silent when this version was written. The comparison of the two versions
(2026-11-17) put the rules about negative counts, about `n_malformed > n_turns`, about `turn_budget` and
about the config numbers into the contract; the checks of the types are outside it.
- `passed` must equal True or False (a Python bool, a numpy bool, or 0/1); anything else raises TypeError.
- The three counts must be integers (Python ints or numpy integers; floats raise TypeError).
- `n_malformed` and `n_turns` must be >= 0, otherwise ValueError (checked on a fail too, although a fail does
  not use `n_turns`).
- `n_malformed > n_turns` cannot happen in a real episode (malformed messages are counted inside
  `n_turns`), but it is not refused: the formula is applied as written. (A first draft refused it; the
  differential test feeds such inputs, and refusing them would stop the two versions from being compared
  on them.)
- `turn_budget` is not range-checked: the formula is used as written.
- The config numbers are used as written, even if they are negative or the caps reach `pass_reward`.
- Config numbers are not converted; with the int fields of RunConfig the result is a Python int.
"""
from __future__ import annotations

import operator

from accessdesk.config import RunConfig


def _whole(value: object, name: str) -> int:
    """`value` as a Python int; TypeError if it is not an integer (bool counts as 0/1)."""
    try:
        return int(operator.index(value))
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


def _capped_loss(per_item: int, count: int, cap: int) -> int:
    """per_item * count, but never more than cap (the contract's min(cap, per_item * count))."""
    uncapped = per_item * count
    if uncapped > cap:
        return cap
    return uncapped


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
    if isinstance(passed, str) or passed not in (True, False):
        raise TypeError(f"passed must be True or False, got {passed!r}")
    malformed = _whole(n_malformed, "n_malformed")
    turns = _whole(n_turns, "n_turns")
    budget = _whole(turn_budget, "turn_budget")
    if malformed < 0:
        raise ValueError(f"n_malformed must be >= 0, got {malformed}")
    if turns < 0:
        raise ValueError(f"n_turns must be >= 0, got {turns}")

    malformed_loss = _capped_loss(cfg.malformed_penalty, malformed, cfg.malformed_cap)

    if passed == False:  # noqa: E712  (accepts numpy bools and 0 as well)
        if cfg.penalise_malformed_on_fail:
            return 0 - malformed_loss
        return 0

    turns_over_budget = turns - budget
    if turns_over_budget < 0:
        turns_over_budget = 0
    turn_loss = _capped_loss(cfg.turn_penalty, turns_over_budget, cfg.turn_cap)

    return cfg.pass_reward - malformed_loss - turn_loss
