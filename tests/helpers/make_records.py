"""Builders of hand-made episode records, for tests of the metrics and tables only.

A record made here has every key of the episode record of BUILD_SPEC section 8, in the same order, and its
counters agree with each other (a pass is closed, a malformed message is a turn, an episode that never closed
used the whole turn limit). The tool calls inside `turns` are stand-ins: no metric reads them.
"""
from __future__ import annotations

from accessdesk.config import DEFAULT

SPLIT_OF_PREFIX = {"tr": "train", "dv": "dev", "te": "test", "ca": "calib"}
SHORTEST_PATH = {"grant": 4, "approval": 4, "already": 3, "inactive": 3}   # turns of the correct solution
# Reward of a pass -> the number of malformed messages that gives it under the default config. A malformed
# message costs 5 points and also uses a turn; the budget is the shortest path plus one turn, and each turn
# over the budget costs 2 points. So: 1 malformed = 100 - 5 = 95; 2 malformed = 100 - 10 - 2 (one turn over)
# = 88; 3 malformed = 100 - 15 - 4 (two turns over) = 81.
MALFORMED_OF_REWARD = {100: 0, 95: 1, 88: 2, 81: 3}
# Reward of a fail -> the number of malformed messages that gives it. 0: any fail under the default config.
# Below 0: a fail under a config with `penalise_malformed_on_fail` on, 5 points per malformed message, at
# most 15. (The records still name the default config's hash: no metric reads the config.)
MALFORMED_OF_FAIL = {0: 0, -5: 1, -10: 2, -15: 3}
CONFIG_SHA256 = DEFAULT.sha256()


def _turn(index: int, malformed: bool, closes: bool) -> dict:
    """One turn record in the format of section 8."""
    if malformed:
        return {"turn": index, "assistant_text": "(hand-made malformed message)", "truncated": False,
                "n_tokens": None, "parse_ok": False, "parse_error": "no_call", "tool_key": None, "args": None,
                "tool_error": None, "tool_response": '{"error": "hand-made"}'}
    tool_key = "close_request" if closes else "lookup_person"
    args = {"resolution": "r", "person_id": "p", "role_id": "q"} if closes else {"name": "n"}
    return {"turn": index, "assistant_text": "(hand-made call)", "truncated": False, "n_tokens": None,
            "parse_ok": True, "parse_error": None, "tool_key": tool_key, "args": args,
            "tool_error": None, "tool_response": '{"result": "hand-made"}'}


def make_record(family_id: str = "dv-000", style: int = 1, sample: int = 0, *, passed: bool = True,
                reward: int | None = None, outcome: str = "grant", closed: bool = True,
                n_turns: int | None = None, n_malformed: int = 0, harness_error: str | None = None,
                arm: str = "base", run: str = "hand-made", ruleset: str = "default", toolset: str = "five",
                prompt: str = "final") -> dict:
    """One episode record.

    Defaults: a clean pass (reward 100) in the shortest number of turns. `reward` defaults to 100 for a pass
    and 0 for a fail; a fail may be given a reward below 0 (the switch for penalties on failing episodes).
    `n_turns` defaults to the shortest path plus the malformed messages for a closed episode
    and to the turn limit for an episode that never closed. Malformed messages come first in `turns`.
    Raises ValueError for a combination that the harness cannot produce. The size of a pass's reward is not
    checked against the penalties here: a caller that gives `reward` also gives the matching `n_malformed`."""
    if reward is None:
        reward = 100 if passed else 0
    if n_turns is None:
        n_turns = SHORTEST_PATH[outcome] + n_malformed if closed else DEFAULT.turn_limit
    if passed and not closed:
        raise ValueError("a pass needs a closed request")
    if passed != (reward > 0):
        raise ValueError("a pass has a reward above 0 and a fail has a reward of 0 or below")
    if closed and n_malformed >= n_turns:
        raise ValueError("a closed episode ends with a well-formed call")
    if not closed and n_turns != DEFAULT.turn_limit:
        raise ValueError("an episode that never closed used the whole turn limit")
    if n_turns > DEFAULT.turn_limit or n_malformed > n_turns:
        raise ValueError("too many turns or too many malformed messages")
    turns = [_turn(i, malformed=i < n_malformed, closes=closed and i == n_turns - 1) for i in range(n_turns)]
    request = {"resolution": "r", "person_id": "p", "role_id": "q"} if closed else \
        {"resolution": None, "person_id": None, "role_id": None}
    return {
        "schema": 1, "run": run, "arm": arm, "split": SPLIT_OF_PREFIX[family_id[:2]], "family_id": family_id,
        "style": style, "scenario_id": f"{family_id}/s{style}", "outcome": outcome, "ruleset": ruleset,
        "toolset": toolset, "prompt": prompt, "sample": sample,
        "seed": {"run_seed": 0, "pass": 0, "chunk": 0, "pos": sample},
        "turns": turns, "n_turns": n_turns, "n_malformed": n_malformed, "n_tool_errors": 0, "closed": closed,
        "ended_by": "close" if closed else "turn_limit",
        "final_state": {"access": [], "approvals": [], "grants": [], "request": request},
        "passed": passed, "reward": reward, "turn_budget": SHORTEST_PATH[outcome] + 1,
        "impl": {"verify": "main", "reward": "main"},
        "text_sha256": "0" * 64, "text_is_placeholder": True, "config_sha256": CONFIG_SHA256,
        "settings": DEFAULT.compared_settings(), "harness_error": harness_error,
    }


def scenario_records(family_id: str, style: int, rewards: list[int], *, outcome: str = "grant",
                     arm: str = "base") -> list[dict]:
    """The samples of one scenario, one record per reward, with sample index 0, 1, 2, ...

    A reward of 100, 95, 88 or 81 is a pass with 0, 1, 2 or 3 malformed messages (see MALFORMED_OF_REWARD).
    A reward of 0 is a fail that closed the request wrongly. A reward of -5, -10 or -15 is such a fail after
    1, 2 or 3 malformed messages, as the switch for penalties on failing episodes scores it."""
    records = []
    for sample, reward in enumerate(rewards):
        if reward in MALFORMED_OF_FAIL:
            records.append(make_record(family_id, style, sample, passed=False, reward=reward, outcome=outcome,
                                       n_malformed=MALFORMED_OF_FAIL[reward], arm=arm))
        elif reward in MALFORMED_OF_REWARD:
            records.append(make_record(family_id, style, sample, passed=True, reward=reward, outcome=outcome,
                                       n_malformed=MALFORMED_OF_REWARD[reward], arm=arm))
        else:
            raise ValueError(f"reward {reward} is not one of {sorted(MALFORMED_OF_FAIL)}, "
                             f"{sorted(MALFORMED_OF_REWARD)}")
    return records
