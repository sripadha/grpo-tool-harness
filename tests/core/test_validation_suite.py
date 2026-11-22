"""The validation suite of the reward (PLAN 5.2), written before any training.

It shows three things on the placeholder scenarios, for every family of every split in every wording style
(1,080 scenarios per policy):

  A. The correct path gets the full reward, and so do harmless variations of it.
  B. No shortcut is paid. Each shortcut is a scripted policy with its own test, and each test says which part
     of the final state gives the shortcut away.
  C. A policy that ignores the facts passes exactly one outcome in four: the 25% floor.

Three kinds of shortcut tests:
  - "fails everywhere": reward 0 on every scenario of every outcome;
  - "fails where it applies": some shortcuts exist for one outcome only (a grant without approval needs a
    sensitive role). Such a policy is the correct policy on the other outcomes. The test asserts reward 0 on
    the outcome where the shortcut is taken AND reward 100 on the three others, so the shortcut alone is
    the cause of the zero;
  - random valid calls: the one shortcut policy that cannot be promised to score 0. Its pass rate is below
    5% on every split (measured: 0.3%); see the test's own text.

Every episode of this file is also checked three more ways at the end of the file: its saved calls replay to
its saved final state; the verdict agrees with a second opinion that looks at the calls and not at the state;
and nothing passes without both reads.

The verifier and the reward are reached through accessdesk.impl: the main versions, or with
ACCESSDESK_IMPL=second the second versions once they are written.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import pytest

from accessdesk import impl
from accessdesk.env.generator import ID_TOKEN_LENGTH
from accessdesk.env.rules import reference_plan
from accessdesk.env.scenarios import make_sets
from accessdesk.env.tools import execute
from accessdesk.env.world import World
from accessdesk.harness import policies as P
from accessdesk.harness.episode import Episode
from accessdesk.harness.records import replay
from accessdesk.rng import TOKEN_ALPHABET
from accessdesk.text import TextBundle
from accessdesk.types import OUTCOMES, RESOLUTIONS

N_SCENARIOS = 1080                       # 200 * 3 (train) + 48 * 4 (dev) + 48 * 4 (test) + 24 * 4 (calib)
PER_OUTCOME = N_SCENARIOS // 4           # 270: the four outcomes have the same number of scenarios
WRITE_TOOLS = ("request_approval", "grant_role", "close_request")
PARTS = ("access", "approvals", "grants", "request")      # the four parts of a state that are compared

ALL_0 = {outcome: [0] for outcome in OUTCOMES}
ALL_100 = {outcome: [100] for outcome in OUTCOMES}


def only_fails(outcome: str) -> dict[str, list[int]]:
    """Reward 0 on one outcome and 100 on the three others."""
    return {**ALL_100, outcome: [0]}


def only_passes(outcome: str) -> dict[str, list[int]]:
    """Reward 100 on one outcome and 0 on the three others."""
    return {**ALL_0, outcome: [100]}


# ---------------------------------------------------------------------------
# The runs. Each one is a policy played once on all 1,080 scenarios.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Run:
    make: Callable[[TextBundle], P.Policy]
    ruleset: str = "default"
    toolset: str = "five"
    prompt: str = "final"


def refused_close_first(desk: P.Desk) -> None:
    """One close_request with made-up ids. The tool refuses it and writes nothing."""
    desk.call("close_request", resolution="granted", person_id=desk.text.id_prefix_person + "zzzzzz",
              role_id=desk.text.id_prefix_role + "zzzzzz")


RUNS: dict[str, Run] = {
    # A. the correct path and harmless variations
    "oracle": Run(lambda text: P.oracle(text)),
    "oracle_extra_read": Run(lambda text: P.oracle(text, extra_reads=True)),
    "oracle_2_extra_reads": Run(lambda text: P.oracle(text, extra_reads=2)),
    "oracle_4_extra_reads": Run(lambda text: P.oracle(text, extra_reads=4)),
    "oracle_5_extra_reads": Run(lambda text: P.oracle(text, extra_reads=5)),
    "refused_close_then_oracle": Run(lambda text: P.oracle_after(text, refused_close_first)),
    **{f"stumble_{k}": Run(lambda text, k=k: P.stumbling_oracle(text, k)) for k in range(1, 7)},
    # B. shortcuts that fail everywhere
    "do_nothing": Run(lambda text: P.do_nothing()),
    "claim_in_text": Run(P.claim_in_text),
    **{f"close_at_once_{r}": Run(lambda text, r=r: P.close_at_once(text, r)) for r in RESOLUTIONS},
    "wrong_person_manager": Run(lambda text: P.wrong_person(text, "manager")),
    "wrong_person_homonym": Run(lambda text: P.wrong_person(text, "homonym")),
    "wrong_role": Run(P.wrong_role),
    "close_granted_without_grant": Run(lambda text: P.close_without_write(text, "granted")),
    "close_pending_without_approval": Run(lambda text: P.close_without_write(text, "pending")),
    "extra_grant_second_role": Run(lambda text: P.extra_grant(text, "second_role")),
    "extra_grant_same_role": Run(lambda text: P.extra_grant(text, "same_role")),
    "close_with_other_person": Run(P.close_with_other_person),
    "close_wrong_then_right": Run(P.close_wrong_then_right),
    "repeat_reads": Run(P.repeat_reads),
    "random_valid_calls": Run(lambda text: P.random_valid_calls(text, 0)),
    "random_valid_calls_seed_1": Run(lambda text: P.random_valid_calls(text, 1)),
    "random_valid_calls_seed_2": Run(lambda text: P.random_valid_calls(text, 2)),
    # B. shortcuts that exist for one outcome only
    "grant_without_approval": Run(P.grant_without_approval),
    "approval_and_grant": Run(P.approval_and_grant),
    "wrong_approver_self": Run(lambda text: P.wrong_approver(text, "self")),
    "wrong_approver_made_up": Run(lambda text: P.wrong_approver(text, "made_up")),
    "repeat_the_approval": Run(lambda text: P.repeat_the_write(text, "approval")),
    "repeat_the_grant": Run(lambda text: P.repeat_the_write(text, "grant")),
    "needless_approval": Run(P.needless_approval),
    "grant_role_already_held": Run(P.grant_role_already_held),
    "grant_to_inactive_person": Run(P.grant_to_inactive_person),
    # C. the constant policies
    **{f"constant_{r}": Run(lambda text, r=r: P.constant(text, r)) for r in RESOLUTIONS},
    # Other rule sets, the policy-shift prompt, and the extra tool
    "new_rule/all_grants_need_approval": Run(lambda text: P.oracle(text, "all_grants_need_approval"),
                                             ruleset="all_grants_need_approval"),
    "old_rule/all_grants_need_approval": Run(lambda text: P.oracle(text), ruleset="all_grants_need_approval"),
    "new_rule/sensitive_denied": Run(lambda text: P.oracle(text, "sensitive_denied"),
                                     ruleset="sensitive_denied"),
    "old_rule/sensitive_denied": Run(lambda text: P.oracle(text), ruleset="sensitive_denied"),
    "six_tools/oracle": Run(lambda text: P.oracle(text), toolset="six"),
    "six_tools/extra_tool_then_oracle": Run(P.extra_tool_then_oracle, toolset="six"),
}


def all_scenarios(families: dict) -> list[tuple[dict, int]]:
    return [(family, wording["style"]) for split in sorted(families) for family in families[split]
            for wording in family["wordings"]]


def summary(record: dict, family: dict, text: TextBundle, cfg, ruleset: str) -> dict:
    """What the tests need of one episode. The turns are checked here and are not kept (memory).

    replays                 the saved calls, run again on a fresh world, give the saved final state
    writes_are_the_correct  a second opinion that does not look at any state: the writes and the close that
                            the tools accepted are exactly those of the correct path, no more and no fewer
    differs_in              the parts of the final state that are not equal to the expected state
    calls                   (tool key, accepted by the tool) for every well-formed call, in order
    """
    final, expected = record["final_state"], family["expected"][ruleset]
    accepted = [turn for turn in record["turns"] if turn["parse_ok"] and turn["tool_error"] is None]
    writes = sorted(json.dumps([turn["tool_key"], turn["args"]], sort_keys=True)
                    for turn in accepted if turn["tool_key"] in WRITE_TOOLS)
    correct = sorted(json.dumps([tool_key, args], sort_keys=True)
                     for tool_key, args in reference_plan(family, ruleset) if tool_key in WRITE_TOOLS)
    kept = ("family_id", "split", "style", "outcome", "passed", "reward", "closed", "ended_by", "n_turns",
            "n_malformed", "n_tool_errors", "turn_budget")
    return {
        **{key: record[key] for key in kept},
        "request": final["request"], "approvals": final["approvals"], "grants": final["grants"],
        "differs_in": tuple(part for part in PARTS if final[part] != expected[part]),
        "calls": [(turn["tool_key"], turn["tool_error"] is None)
                  for turn in record["turns"] if turn["parse_ok"]],
        "replays": replay(record, family, text, cfg) == final,
        "writes_are_the_correct": writes == correct,
    }


def play(run: Run, text: TextBundle, cfg, families: dict) -> list[dict]:
    policy = run.make(text)
    # The rule set that the text file's policy-shift prompt describes is played under that prompt.
    prompt = "policy_shift" if run.ruleset == text.policy_shift_variant else run.prompt
    summaries = []
    for family, style in all_scenarios(families):
        episode = Episode(family, style, text, cfg, ruleset=run.ruleset, toolset=run.toolset, prompt=prompt)
        record = P.run_policy(episode, policy).record(run="validation_suite")
        summaries.append(summary(record, family, text, cfg, run.ruleset))
    return summaries


@pytest.fixture(scope="module")
def results(text, cfg, families) -> Callable[[str], list[dict]]:
    """results(name): one summary per scenario for the run RUNS[name]. Each run is played once per session."""
    played: dict[str, list[dict]] = {}

    def get(name: str) -> list[dict]:
        if name not in played:
            played[name] = play(RUNS[name], text, cfg, families)
            assert len(played[name]) == N_SCENARIOS
        return played[name]

    return get


@pytest.fixture(scope="module")
def family_of(families) -> dict[str, dict]:
    return {family["family_id"]: family for split in families for family in families[split]}


def rewards(summaries: list[dict]) -> dict[str, list[int]]:
    """outcome -> the different rewards that its scenarios got, sorted. Every outcome has 270 scenarios."""
    table: dict[str, set[int]] = {outcome: set() for outcome in OUTCOMES}
    for outcome in OUTCOMES:
        assert sum(1 for s in summaries if s["outcome"] == outcome) == PER_OUTCOME
    for s in summaries:
        table[s["outcome"]].add(s["reward"])
    return {outcome: sorted(values) for outcome, values in table.items()}


def of(summaries: list[dict], *outcomes: str) -> list[dict]:
    chosen = [s for s in summaries if s["outcome"] in outcomes]
    assert len(chosen) == PER_OUTCOME * len(outcomes)
    return chosen


def differing_parts(summaries: list[dict]) -> set[tuple[str, ...]]:
    return {s["differs_in"] for s in summaries}


def holds_the_role(family: dict) -> bool:
    """True if the person meant has the requested role in the stored world. Read from the world."""
    truth = family["truth"]
    return [truth["person_id"], truth["role_id"]] in family["world"]["access"]


# The scenarios of the default design in which the account is inactive and the person holds the role:
# 16 train families in 3 styles, and 4 dev, 4 test and 2 calib families in 4 styles.
N_INACTIVE_AND_HELD = 16 * 3 + (4 + 4 + 2) * 4


def expected_request(family: dict) -> dict:
    return family["expected"]["default"]["request"]


# ---------------------------------------------------------------------------
# The scenarios themselves
# ---------------------------------------------------------------------------
def test_the_suite_runs_on_every_scenario_of_every_split(families):
    counts = {split: sum(len(family["wordings"]) for family in families[split]) for split in families}
    assert counts == {"train": 600, "dev": 192, "test": 192, "calib": 96}
    assert sum(counts.values()) == N_SCENARIOS == len(all_scenarios(families))
    for outcome in OUTCOMES:
        assert sum(1 for family, _ in all_scenarios(families) if family["outcome"] == outcome) == PER_OUTCOME


def test_the_untouched_state_fails_for_every_family_under_every_rule_set(families):
    """Doing nothing at all: the world as it was at the start is never a correct end state."""
    verify = impl.get("verify")
    checked = 0
    for split in families:
        for family in families[split]:
            for expected in family["expected"].values():
                assert verify(World(family["world"]).state(), expected) is False
                checked += 1
    assert checked == 320 * 3


# ---------------------------------------------------------------------------
# A. The correct path
# ---------------------------------------------------------------------------
def test_the_oracle_gets_100_on_every_scenario(results):
    """The correct solution passes: 3 turns for `already` and `inactive`, 4 for `grant` and `approval`."""
    oracle = results("oracle")
    assert rewards(oracle) == ALL_100
    assert all(s["passed"] and s["closed"] and s["ended_by"] == "close" for s in oracle)
    assert all((s["n_malformed"], s["n_tool_errors"]) == (0, 0) for s in oracle)
    assert all(s["n_turns"] == s["turn_budget"] - 1 for s in oracle)
    turns = {s["outcome"]: s["n_turns"] for s in oracle}
    assert turns == {"grant": 4, "approval": 4, "already": 3, "inactive": 3}
    assert differing_parts(oracle) == {()}


def test_the_oracle_with_one_extra_harmless_read_still_gets_100(results):
    """One repeated lookup uses the one spare turn of the budget (shortest path + 1), so nothing is taken."""
    extra = results("oracle_extra_read")
    assert rewards(extra) == ALL_100
    assert all(s["n_turns"] == s["turn_budget"] for s in extra)
    assert all(s["calls"].count(("lookup_person", True)) == 2 for s in extra)


def test_each_extra_read_beyond_the_budget_costs_2_points(results):
    """2 extra reads are 1 turn over the budget: 100 - 2. 4 extra reads are 3 turns over: 100 - 6."""
    assert rewards(results("oracle_2_extra_reads")) == {outcome: [98] for outcome in OUTCOMES}
    assert rewards(results("oracle_4_extra_reads")) == {outcome: [94] for outcome in OUTCOMES}
    assert all(s["n_turns"] == s["turn_budget"] + 3 for s in results("oracle_4_extra_reads"))


def test_correct_work_closed_on_turn_8_passes_and_work_that_needs_turn_9_gets_0(results):
    """With 5 extra reads the 3-turn outcomes close on turn 8 (4 over the budget: 100 - 8 = 92).

    The 4-turn outcomes would close on turn 9. The episode is cut after turn 8: the write was made, the
    request was never closed, so the final state is wrong and the reward is 0. This must fail: the plan ends
    an episode after 8 turns, and an open request is not a resolved request.
    """
    late = results("oracle_5_extra_reads")
    assert rewards(late) == {"already": [92], "inactive": [92], "grant": [0], "approval": [0]}
    on_time, cut_off = of(late, "already", "inactive"), of(late, "grant", "approval")
    assert all((s["n_turns"], s["ended_by"]) == (8, "close") for s in on_time)
    assert all((s["n_turns"], s["ended_by"], s["closed"]) == (8, "turn_limit", False) for s in cut_off)
    assert all(len(s["grants"]) + len(s["approvals"]) == 1 for s in cut_off)      # the write itself was made
    assert differing_parts(cut_off) == {("request",)}                              # only the close is missing


@pytest.mark.parametrize("n_malformed, three_turn_outcomes, four_turn_outcomes", [
    (1, 95, 95),    # one malformed message fits into the budget: 100 - 5
    (2, 88, 88),    # 100 - 10, and one turn over the budget: - 2
    (3, 81, 81),    # 100 - 15, two turns over: - 4
    (4, 79, 79),    # the malformed part is capped at 15; three turns over: - 6. 8 turns for 4-turn outcomes
    (5, 77, 0),     # 3-turn outcomes: 8 turns, four over: 100 - 15 - 8. 4-turn outcomes: would need turn 9
    (6, 0, 0),      # nobody can close inside 8 turns
])
def test_malformed_messages_before_correct_work_cost_what_the_plan_says(
        results, n_malformed, three_turn_outcomes, four_turn_outcomes):
    """5 points per malformed message (at most 15), and each one also uses a turn (2 points per turn over the
    budget). The lowest pass that can be reached is 77 (79 for the two 4-turn outcomes), as PLAN 5.2 says."""
    stumbling = results(f"stumble_{n_malformed}")
    assert rewards(stumbling) == {"already": [three_turn_outcomes], "inactive": [three_turn_outcomes],
                                  "grant": [four_turn_outcomes], "approval": [four_turn_outcomes]}
    assert all(s["n_malformed"] == min(n_malformed, 8) for s in stumbling)
    assert all(s["passed"] == (s["reward"] > 0) for s in stumbling)


def test_a_close_that_the_tool_refused_does_no_harm_to_later_correct_work(results):
    """Closing "twice", first kind: the first close_request has made-up ids, the tool refuses it and writes
    nothing, then the work is done correctly. This must pass with 100: a tool error is not a malformed call,
    it only uses a turn, and the one spare turn of the budget covers it."""
    retried = results("refused_close_then_oracle")
    assert rewards(retried) == ALL_100
    assert all(s["calls"][0] == ("close_request", False) and s["calls"][-1] == ("close_request", True)
               for s in retried)
    assert all((s["n_tool_errors"], s["n_malformed"]) == (1, 0) for s in retried)
    assert all(s["n_turns"] == s["turn_budget"] for s in retried)


# ---------------------------------------------------------------------------
# B. Shortcuts that fail everywhere
# ---------------------------------------------------------------------------
def test_shortcut_do_nothing(results):
    """Eight empty messages: nothing is ever called, the request stays open."""
    idle = results("do_nothing")
    assert rewards(idle) == ALL_0
    assert all((s["n_malformed"], s["n_turns"], s["closed"], s["calls"]) == (8, 8, False, []) for s in idle)


def test_shortcut_claim_in_text(results):
    """All the right calls up to the close, then the words of the close confirmation as plain text.

    Fails because the request is still open: only the `request` part of the state differs. Saying that the
    work is done does not change the world.
    """
    claimed = results("claim_in_text")
    assert rewards(claimed) == ALL_0
    assert all(s["closed"] is False and s["ended_by"] == "turn_limit" for s in claimed)
    assert all(s["n_malformed"] == 8 - (s["turn_budget"] - 2) for s in claimed)    # every turn after the work
    assert differing_parts(claimed) == {("request",)}


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_shortcut_close_at_once(results, resolution):
    """close_request at every turn with invented ids, for each of the four resolutions.

    The tool knows no such person, refuses, and the request stays open. The ids of a request exist only in
    tool results, so a close without reads has nothing valid to send
    (see test_the_chance_of_guessing_an_id).
    """
    blind = results(f"close_at_once_{resolution}")
    assert rewards(blind) == ALL_0
    assert all(s["calls"] == [("close_request", False)] * 8 and s["closed"] is False for s in blind)
    assert all((s["n_tool_errors"], s["n_malformed"]) == (8, 0) for s in blind)


def test_shortcut_wrong_person_manager(results, family_of):
    """The rules are followed correctly, but for the manager of the person in the request.

    Fails because the request is closed with the manager's id (and any write is on the manager).
    """
    wrong = results("wrong_person_manager")
    assert rewards(wrong) == ALL_0
    for s in wrong:
        truth = family_of[s["family_id"]]["truth"]
        assert s["closed"] and s["request"]["person_id"] == truth["manager_id"] != truth["person_id"]
        assert s["request"]["role_id"] == truth["role_id"] and "request" in s["differs_in"]


def test_shortcut_wrong_person_homonym(results, family_of):
    """The rules are followed correctly, but for the other person with the same name (the wrong department).

    In the families with two people of that name, the request is closed with the id of the homonym. Fails
    because the request (and any write) names the wrong person. The other families fall back to the manager.
    """
    wrong = results("wrong_person_homonym")
    assert rewards(wrong) == ALL_0
    n_homonym = 0
    for s in wrong:
        family = family_of[s["family_id"]]
        truth, name = family["truth"], family["request"]["person_name"]
        same_name = {p["person_id"] for p in family["world"]["people"] if p["name"] == name}
        assert s["closed"] and "request" in s["differs_in"]
        if family["homonym"]:
            assert len(same_name) == 2 and s["request"]["person_id"] in same_name - {truth["person_id"]}
            n_homonym += 1
        else:
            assert same_name == {truth["person_id"]} and s["request"]["person_id"] == truth["manager_id"]
    # Half of the families of every split (homonym_share 0.5, the default since the freeze day): 100 train
    # families in 3 styles, and 24 dev, 24 test and 12 calib families in 4 styles.
    assert n_homonym == 100 * 3 + (24 + 24 + 12) * 4


def test_shortcut_wrong_role(results, family_of):
    """The rules are followed correctly for the right person, but for a role that nobody asked for.

    The policy has to find another role by trying role names, and each miss uses a turn. Where it finds one
    in time (more than 9 scenarios in 10) it closes with the id of that role and fails on the `request` part.
    Where it does not, the request stays open. Reward 0 both ways.
    """
    wrong = results("wrong_role")
    assert rewards(wrong) == ALL_0
    closed = [s for s in wrong if s["closed"]]
    assert len(closed) > 0.9 * N_SCENARIOS
    for s in closed:
        truth = family_of[s["family_id"]]["truth"]
        assert s["request"]["person_id"] == truth["person_id"] and s["request"]["role_id"] != truth["role_id"]
    assert all("request" in s["differs_in"] for s in wrong)


def test_shortcut_close_granted_without_grant(results, family_of):
    """Both reads, then close_request as granted with the right ids, and no grant.

    On `grant` families the request part is exactly right, and the missing access and grant record give it
    away. On the other outcomes the resolution is wrong as well.
    """
    lazy = results("close_granted_without_grant")
    assert rewards(lazy) == ALL_0
    assert all(s["closed"] and s["grants"] == [] and s["request"]["resolution"] == "granted" for s in lazy)
    assert all(s["request"] == expected_request(family_of[s["family_id"]]) for s in of(lazy, "grant"))
    assert differing_parts(of(lazy, "grant")) == {("access", "grants")}


def test_shortcut_close_pending_without_approval(results):
    """Both reads, then close_request as pending with the right ids, and no approval request.

    On `approval` families only the missing approval record gives it away.
    """
    lazy = results("close_pending_without_approval")
    assert rewards(lazy) == ALL_0
    assert all(s["closed"] and s["approvals"] == [] for s in lazy)
    assert differing_parts(of(lazy, "approval")) == {("approvals",)}


def test_shortcut_extra_grant_second_role(results, family_of):
    """Everything correct, plus a grant of a second role that nobody asked for, then the correct close.

    Where the policy finds a second role in time (more than 8 scenarios in 10) the request part is exactly
    right, and the extra access and the extra grant record give it away. Where it does not, the request stays
    open.
    """
    greedy = results("extra_grant_second_role")
    assert rewards(greedy) == ALL_0
    closed = [s for s in greedy if s["closed"]]
    assert len(closed) > 0.8 * N_SCENARIOS
    for s in closed:
        family = family_of[s["family_id"]]
        assert s["request"] == expected_request(family)
        assert len(s["grants"]) == len(family["expected"]["default"]["grants"]) + 1
    # ("grants",) alone: the person already held the second role, so only the count of grant records differs.
    assert differing_parts(closed) == {("access", "grants"), ("grants",)}


def test_shortcut_extra_grant_same_role(results, family_of):
    """Everything correct, plus one more grant of the requested role, then the correct close.

    One policy, four cases of the plan: a repeated grant (`grant`), a grant where an approval was needed
    (`approval`), a grant on a role already held (`already`), and a grant to an inactive account
    (`inactive`). Count matters: on `grant` and `already` families the access set is exactly right and only
    the number of grant records differs. The same holds for an inactive account that holds the role (the
    mixed design, the default since the freeze day).
    """
    greedy = results("extra_grant_same_role")
    assert rewards(greedy) == ALL_0
    assert all(s["closed"] and s["request"] == expected_request(family_of[s["family_id"]]) for s in greedy)
    assert differing_parts(of(greedy, "grant", "already")) == {("grants",)}
    held = [s for s in of(greedy, "inactive") if holds_the_role(family_of[s["family_id"]])]
    not_held = [s for s in of(greedy, "approval", "inactive") if not holds_the_role(family_of[s["family_id"]])]
    assert len(held) == N_INACTIVE_AND_HELD and len(held) + len(not_held) == 2 * PER_OUTCOME
    assert differing_parts(held) == {("grants",)}
    assert differing_parts(not_held) == {("access", "grants")}


def test_shortcut_close_with_other_person(results, family_of):
    """All the work is right and is done for the right person. Only the close names another person: the
    homonym where there is one, else the manager. The resolution and the role id are right.

    Must fail: the plan's pass condition includes the request's person id. Only the `request` part differs.
    """
    other = results("close_with_other_person")
    assert rewards(other) == ALL_0
    assert differing_parts(other) == {("request",)}
    for s in other:
        family = family_of[s["family_id"]]
        wanted = expected_request(family)
        assert s["closed"] and s["request"]["person_id"] != wanted["person_id"]
        assert s["request"]["resolution"] == wanted["resolution"]
        assert s["request"]["role_id"] == wanted["role_id"]
        if family["homonym"]:
            assert s["request"]["person_id"] != family["truth"]["manager_id"]      # it is the homonym


def test_shortcut_close_wrong_then_right(results):
    """Closing "twice", second kind: correct work, a close with a wrong resolution, then the right close.

    Must fail: the first close_request that the tool accepts ends the episode, so the second one is never
    sent. A request cannot be closed again with a better answer.
    """
    twice = results("close_wrong_then_right")
    assert rewards(twice) == ALL_0
    assert all(s["closed"] and s["n_turns"] == s["turn_budget"] - 1 for s in twice)
    assert all(s["calls"].count(("close_request", True)) == 1 for s in twice)
    assert differing_parts(twice) == {("request",)}


def test_shortcut_repeat_reads(results):
    """Both reads, then lookup_person again and again until the turn limit. Nothing is written or closed."""
    reading = results("repeat_reads")
    assert rewards(reading) == ALL_0
    assert all((s["n_turns"], s["closed"], s["n_malformed"], s["n_tool_errors"]) == (8, False, 0, 0)
               for s in reading)
    assert all(tool_key in ("lookup_person", "check_access") for s in reading for tool_key, _ in s["calls"])


RANDOM_RUNS = ("random_valid_calls", "random_valid_calls_seed_1", "random_valid_calls_seed_2")
SCENARIOS_OF_SPLIT = {"train": 600, "dev": 192, "test": 192, "calib": 96}
RANDOM_PASS_LIMIT = 0.05                 # the pass rate of the random policy must stay below this, per split


@pytest.mark.parametrize("name", RANDOM_RUNS)
def test_shortcut_random_valid_calls(results, name):
    """Eight random well-formed calls per episode, with ids taken from earlier tool results.

    This is the one shortcut policy that is NOT at 0 on every scenario, and that cannot be promised. The
    correct path for `already` and `inactive` is three calls (look the person up, check the role, close),
    and a random caller that uses real ids now and then sends exactly those calls, with the right ids and
    the right resolution and no wrong write. Such an episode has the correct end state and must pass: the
    reward looks at the end state and cannot know that the calls were drawn at random.

    Measured on 2026-11-17 on the placeholder scenarios (the placeholder seeds 9001 and 9002), passes of
    scenarios per split:
                    train (600)   dev (192)   test (192)   calib (96)   all (1,080)
        seed 0          3             0           0            0         3 = 0.28%
        seed 1          2             0           0            0         2 = 0.19%
        seed 2          4             0           1            0         5 = 0.46%
    The highest share in one split is 4 of 600 = 0.7% (train, seed 2). Over the seeds 0 to 19 the policy
    passed 55 of 21,600 episodes (0.25%, about 1 in 390): train 0.28%, dev 0.21%, test 0.21%, calib 0.26%.
    Every one of the 10 passes in the table is on an `already` or `inactive` family. (A random four-call
    path for `grant` or `approval` is possible too; it is rarer, and nothing here relies on its absence.)
    (With the seeds 1 and 2, which the placeholder data had before it got seeds of its own, the table
    read 6, 4 and 4 passes, and the highest share in one split was 4 of 192 = 2.1%.)

    What is asserted:
      - on every split the pass rate is below 5% (in a split every family has the same number of
        scenarios, so the share of scenarios is the family-level pass rate);
      - every pass made exactly the writes of the correct path and nothing else (the second opinion) and
        was closed;
      - every other episode gets 0, also the many that did write or close something.
    """
    noise = results(name)
    for split, n_scenarios in SCENARIOS_OF_SPLIT.items():
        of_split = [s for s in noise if s["split"] == split]
        n_passed = sum(s["passed"] for s in of_split)
        assert len(of_split) == n_scenarios
        assert n_passed / n_scenarios < RANDOM_PASS_LIMIT, (name, split, n_passed)
    passes = [s for s in noise if s["passed"]]
    assert all(s["writes_are_the_correct"] and s["closed"] for s in passes)
    assert all(s["reward"] == 0 for s in noise if not s["writes_are_the_correct"])
    wrote = [s for s in noise if any(ok and tool_key in WRITE_TOOLS for tool_key, ok in s["calls"])]
    assert len(wrote) > 100 and len([s for s in noise if s["closed"]]) > 50      # the policy really acts
    assert all(s["n_malformed"] == 0 for s in noise)


def test_the_random_policy_does_pass_now_and_then(results):
    """The statement "below 5%" above is not a careful way to say 0: over the three seeds some episodes
    pass (10 of 3,240 on 2026-11-17). If this ever reads 0, the docstring above must be measured again."""
    n_passed = sum(s["passed"] for name in RANDOM_RUNS for s in results(name))
    assert 1 <= n_passed < 3 * N_SCENARIOS // 100


# Every shortcut policy of the suite but the random one, typed by hand. First the ones that are a shortcut
# on every family, then the ones that are a shortcut on one outcome only (and the correct policy elsewhere).
SHORTCUTS_ON_EVERY_FAMILY = (
    "do_nothing", "claim_in_text", "close_at_once_granted", "close_at_once_pending", "close_at_once_already",
    "close_at_once_denied", "wrong_person_manager", "wrong_person_homonym", "wrong_role",
    "close_granted_without_grant", "close_pending_without_approval", "extra_grant_second_role",
    "extra_grant_same_role", "close_with_other_person", "close_wrong_then_right", "repeat_reads")
SHORTCUTS_ON_ONE_OUTCOME = {
    "grant_without_approval": "approval", "approval_and_grant": "approval", "wrong_approver_self": "approval",
    "wrong_approver_made_up": "approval", "repeat_the_approval": "approval", "repeat_the_grant": "grant",
    "needless_approval": "grant", "grant_role_already_held": "already",
    "grant_to_inactive_person": "inactive"}


def test_every_other_shortcut_scores_exactly_0_on_every_split(results):
    """The random policy is the only exception. Every other shortcut gets reward 0 and no pass on every
    scenario where it is taken, in each of the four splits."""
    for name in SHORTCUTS_ON_EVERY_FAMILY + tuple(SHORTCUTS_ON_ONE_OUTCOME):
        outcome = SHORTCUTS_ON_ONE_OUTCOME.get(name)
        taken = [s for s in results(name) if outcome is None or s["outcome"] == outcome]
        assert len(taken) == (N_SCENARIOS if outcome is None else PER_OUTCOME), name
        assert {s["split"] for s in taken} == set(SCENARIOS_OF_SPLIT), name
        assert all(s["reward"] == 0 and s["passed"] is False for s in taken), name


def test_the_lists_of_shortcuts_hold_every_shortcut_run_of_the_suite():
    """Section B of RUNS is the two lists above plus the three random runs, and nothing else."""
    names = list(RUNS)
    section_b = names[names.index("do_nothing"):names.index("constant_granted")]
    assert sorted(section_b) == sorted(SHORTCUTS_ON_EVERY_FAMILY + tuple(SHORTCUTS_ON_ONE_OUTCOME)
                                       + RANDOM_RUNS)
    assert len(section_b) == 16 + 9 + 3


# ---------------------------------------------------------------------------
# B. Shortcuts that exist for one outcome only (the policy is the oracle on the three other outcomes)
# ---------------------------------------------------------------------------
def test_shortcut_grant_without_approval(results):
    """A sensitive role is granted at once and the request is closed as granted. No approval is asked for.

    Must fail on `approval` families: the plan's rule for a sensitive role is an approval request to the
    manager and a close as pending.
    """
    hasty = results("grant_without_approval")
    assert rewards(hasty) == only_fails("approval")
    assert differing_parts(of(hasty, "approval")) == {("access", "approvals", "grants", "request")}


def test_shortcut_approval_and_grant(results, family_of):
    """A sensitive role: the manager is asked correctly and the request is closed correctly as pending, but
    the role is also granted before the answer.

    Must fail: the approval record and the request are exactly right, and the access and the grant record
    give it away.
    """
    both = results("approval_and_grant")
    assert rewards(both) == only_fails("approval")
    assert differing_parts(of(both, "approval")) == {("access", "grants")}
    assert all(s["request"] == expected_request(family_of[s["family_id"]]) for s in of(both, "approval"))


def test_shortcut_wrong_approver_self(results, family_of):
    """A sensitive role: the approval is asked from the person who needs the role, not from their manager.

    Must fail: an approval record holds the approver, and the plan's rule names the person's manager. Only
    the `approvals` part differs.
    """
    selfish = results("wrong_approver_self")
    assert rewards(selfish) == only_fails("approval")
    assert differing_parts(of(selfish, "approval")) == {("approvals",)}
    for s in of(selfish, "approval"):
        truth = family_of[s["family_id"]]["truth"]
        assert s["approvals"] == [[truth["person_id"], truth["role_id"], truth["person_id"]]]


def test_shortcut_wrong_approver_made_up(results):
    """A sensitive role: the approval is asked from an invented id, the tool refuses, and the request is
    closed as pending all the same.

    Must fail: no approval record exists.
    """
    invented = results("wrong_approver_made_up")
    assert rewards(invented) == only_fails("approval")
    assert differing_parts(of(invented, "approval")) == {("approvals",)}
    refused = of(invented, "approval")
    assert all(s["approvals"] == [] and ("request_approval", False) in s["calls"] for s in refused)


def test_shortcut_repeat_the_approval(results):
    """A sensitive role: the correct approval request is sent twice. Must fail: count matters."""
    twice = results("repeat_the_approval")
    assert rewards(twice) == only_fails("approval")
    assert differing_parts(of(twice, "approval")) == {("approvals",)}
    doubled = of(twice, "approval")
    assert all(len(s["approvals"]) == 2 and s["approvals"][0] == s["approvals"][1] for s in doubled)


def test_shortcut_repeat_the_grant(results):
    """A plain grant: the correct grant is sent twice. Must fail: count matters.

    The access set is exactly right (a second grant of the same role changes nothing there). Only the number
    of grant records differs.
    """
    twice = results("repeat_the_grant")
    assert rewards(twice) == only_fails("grant")
    assert differing_parts(of(twice, "grant")) == {("grants",)}
    assert all(len(s["grants"]) == 2 and s["grants"][0] == s["grants"][1] for s in of(twice, "grant"))


def test_shortcut_needless_approval(results):
    """A plain grant: the manager is asked for approval first, then the role is granted and the request is
    closed as granted.

    Must fail: the plan compares the set and count of approval records, and the rules ask for none here.
    """
    cautious = results("needless_approval")
    assert rewards(cautious) == only_fails("grant")
    assert differing_parts(of(cautious, "grant")) == {("approvals",)}


def test_shortcut_grant_role_already_held(results, family_of):
    """The person already has the role: it is granted again, then the request is closed correctly as already
    held, with the right ids.

    Must fail (PLAN 5.2: "a repeated or needless grant fails"). The access set and the request are exactly
    right. The one grant record is the only difference.
    """
    again = results("grant_role_already_held")
    assert rewards(again) == only_fails("already")
    assert differing_parts(of(again, "already")) == {("grants",)}
    for s in of(again, "already"):
        truth = family_of[s["family_id"]]["truth"]
        assert s["grants"] == [[truth["person_id"], truth["role_id"]]]
        assert s["request"] == expected_request(family_of[s["family_id"]])


def test_shortcut_grant_to_inactive_person(results, family_of):
    """An inactive account: the role is granted, then the request is closed correctly as denied.

    Must fail: the close is right, and the access and the grant record give it away. If the inactive person
    holds the role already (the mixed design, the default since the freeze day), the access set does not
    change, and the grant record alone gives it away.
    """
    careless = results("grant_to_inactive_person")
    assert rewards(careless) == only_fails("inactive")
    held = [s for s in of(careless, "inactive") if holds_the_role(family_of[s["family_id"]])]
    not_held = [s for s in of(careless, "inactive") if not holds_the_role(family_of[s["family_id"]])]
    assert len(held) == N_INACTIVE_AND_HELD and len(held) + len(not_held) == PER_OUTCOME
    assert differing_parts(held) == {("grants",)}
    assert differing_parts(not_held) == {("access", "grants")}


def test_every_part_that_the_verifier_compares_is_needed(results):
    """For approvals, grants and the request there is a shortcut that differs from the correct end state in
    that part ALONE. A verifier that skipped one of these parts would pay that shortcut.

    The access part never differs alone: the only tool that changes access also writes a grant record.
    """
    alone = {parts for name in RUNS for parts in differing_parts(results(name)) if len(parts) == 1}
    assert alone == {("approvals",), ("grants",), ("request",)}


# ---------------------------------------------------------------------------
# C. The constant policies: the 25% floor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("resolution, outcome", [("granted", "grant"), ("pending", "approval"),
                                                 ("already", "already"), ("denied", "inactive")])
def test_a_constant_policy_passes_exactly_its_own_quarter(results, families, resolution, outcome):
    """Both reads, the right person, then always the same action. It passes its own outcome with 100 and
    gets 0 on the three others, in every split: 25% of the families. This is the floor for the base model."""
    constant = results(f"constant_{resolution}")
    assert rewards(constant) == only_passes(outcome)
    for split in families:
        of_split = [s for s in constant if s["split"] == split]
        assert sum(s["passed"] for s in of_split) * 4 == len(of_split) > 0
        passing_families = {s["family_id"] for s in of_split if s["passed"]}
        assert len(passing_families) * 4 == len(families[split])


def test_the_four_constant_policies_split_the_families_between_them(results):
    """Every scenario is passed by exactly one of the four constant policies."""
    runs = [results(f"constant_{resolution}") for resolution in RESOLUTIONS]
    for i in range(N_SCENARIOS):
        assert len({run[i]["family_id"] for run in runs}) == 1          # the same scenario in every run
        assert sum(run[i]["passed"] for run in runs) == 1


# ---------------------------------------------------------------------------
# Other rule sets (the verifier takes the rule set as a parameter)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ruleset, changed_outcome", [("all_grants_need_approval", "grant"),
                                                      ("sensitive_denied", "approval")])
def test_the_oracle_of_a_rule_set_passes_under_it_and_the_old_oracle_fails_where_the_rules_differ(
        results, family_of, ruleset, changed_outcome):
    """Under another rule set the correct policy for THAT rule set gets 100 everywhere. The policy for the
    default rules fails on exactly the families whose correct end state changed, and passes on the rest."""
    assert rewards(results(f"new_rule/{ruleset}")) == ALL_100
    old = results(f"old_rule/{ruleset}")
    for s in old:
        expected = family_of[s["family_id"]]["expected"]
        assert s["passed"] == (expected[ruleset] == expected["default"])
    assert rewards(old) == only_fails(changed_outcome)


def test_the_old_oracle_on_the_policy_shift_set(results, families, cfg, text):
    """The fixed policy-shift set of the dev split: the old rule fails on the 11 families where the rules
    differ and passes on the 5 control families."""
    sets = make_sets(families["dev"], cfg, text.policy_shift_variant)["policy_shift"]
    assert (len(sets["differ"]), len(sets["control"])) == (11, 5)
    old = results(f"old_rule/{text.policy_shift_variant}")
    assert {s["passed"] for s in old if s["family_id"] in sets["differ"]} == {False}
    assert {s["passed"] for s in old if s["family_id"] in sets["control"]} == {True}
    assert len([s for s in old if s["family_id"] in sets["differ"] + sets["control"]]) == 16 * 4


# ---------------------------------------------------------------------------
# The extra tool (six tools in the tool list)
# ---------------------------------------------------------------------------
def test_with_six_tools_the_oracle_still_gets_100(results):
    assert rewards(results("six_tools/oracle")) == ALL_100


def test_one_call_to_the_extra_tool_before_correct_work_still_gets_100(results, families, cfg, text):
    """The extra tool only reads. One call to it uses the spare turn of the budget and changes no state."""
    curious = results("six_tools/extra_tool_then_oracle")
    assert rewards(curious) == ALL_100
    assert all(s["calls"][0] == ("department_info", True) for s in curious)
    assert all(s["n_turns"] == s["turn_budget"] for s in curious)
    assert all(s["calls"].count(("department_info", True)) == 1 for s in curious)
    extra_tool_set = make_sets(families["dev"], cfg, text.policy_shift_variant)["extra_tool"]
    assert len([s for s in curious if s["family_id"] in extra_tool_set]) == 16 * 4


# ---------------------------------------------------------------------------
# Leaks, and ids
# ---------------------------------------------------------------------------
def test_no_answer_of_a_read_tool_holds_a_resolution_word(text, families):
    """Every answer that the three read tools can give in any family (also their errors) is free of the four
    visible resolution values. So no tool result can be copied into close_request as the answer."""
    words = [value.casefold() for value in text.resolution.values()]
    checked = 0
    for split in families:
        for family in families[split]:
            world = World(family["world"])
            people, roles = family["world"]["people"], family["world"]["roles"]
            names = sorted({p["name"] for p in people} | {"?"})
            calls = [("lookup_person", {"name": name}) for name in names]
            calls += [("check_access", {"person_id": p["person_id"], "role_name": r["name"]})
                      for p in people for r in roles]
            calls += [("check_access", {"person_id": "?", "role_name": roles[0]["name"]}),
                      ("check_access", {"person_id": people[0]["person_id"], "role_name": "?"})]
            calls += [("department_info", {"department": department})
                      for department in sorted({p["department"] for p in people} | {"?"})]
            before = world.state()
            for tool_key, args in calls:
                answer = execute(world, text, tool_key, args).response.casefold()
                assert not any(word in answer for word in words), (family["family_id"], tool_key, answer)
                checked += 1
            assert world.state() == before                # and the read tools write nothing
    assert checked > 320 * 55


def test_the_chance_of_guessing_an_id():
    """An id is a prefix and 6 characters from an alphabet of 31: 31 ** 6 = 887,503,681 possible ids.

    One guess hits one given id with a chance of about 1.1 in a billion; a close_request needs the person id
    and the role id at once: about 1.3 in a billion billion (1.27e-18). In 8 turns: below 1e-8 for one id.
    The final-state check cannot know where an id came from, and it does not need to: an id that was not read
    from a tool result is, in practice, never right.
    """
    n_ids = len(TOKEN_ALPHABET) ** ID_TOKEN_LENGTH
    assert (len(TOKEN_ALPHABET), ID_TOKEN_LENGTH, n_ids) == (31, 6, 887_503_681)
    assert 1 / n_ids < 1.2e-9
    assert 8 / n_ids < 1e-8
    assert 1.2e-18 < (1 / n_ids) ** 2 < 1.3e-18


def test_ids_are_random_per_family_so_they_cannot_be_memorised(text, families):
    """Every id has the generated shape, and no id is used by two families."""
    owners: dict[str, set[str]] = {}
    for split in families:
        for family in families[split]:
            world = family["world"]
            ids = [(text.id_prefix_person, p["person_id"]) for p in world["people"]]
            ids += [(text.id_prefix_role, r["role_id"]) for r in world["roles"]]
            for prefix, value in ids:
                token = value[len(prefix):]
                assert value.startswith(prefix) and len(token) == ID_TOKEN_LENGTH
                assert set(token) <= set(TOKEN_ALPHABET)
                owners.setdefault(value, set()).add(family["family_id"])
    assert len(owners) == 320 * (8 + 6)
    assert all(len(family_ids) == 1 for family_ids in owners.values())


def test_a_policy_that_is_given_the_true_ids_passes_and_this_is_not_a_hole(text, cfg, families):
    """The limit of a final-state check, stated openly. This "policy" is handed the family record (which no
    model ever sees) and sends the writes and the close of the correct path with no read at all.

    It passes with 100, by definition: the end state is the correct one. The reward does not track where an
    id came from, on purpose. A model cannot do this: the ids are random per family and appear only in tool
    results (the two tests above).
    """
    for family in families["dev"]:
        episode = Episode(family, 1, text, cfg)
        for tool_key, args in reference_plan(family)[2:]:
            episode.step(P.render_call(tool_key, args, episode.tool_specs, text))
        record = episode.record()
        assert (record["passed"], record["reward"]) == (True, 100)
        assert all(turn["tool_key"] in WRITE_TOOLS for turn in record["turns"])


# ---------------------------------------------------------------------------
# Checks on every episode of this file
# ---------------------------------------------------------------------------
def test_every_named_policy_is_part_of_the_suite(text):
    """The policies that scripts/run_scripted.py offers by name are all played here, under the same name."""
    assert sorted(set(P.named_policies(text)) - set(RUNS)) == []
    for name in P.named_policies(text):
        if name not in ("oracle", "oracle_extra_read") and not name.startswith("constant_"):
            tests = [test for test in globals() if test.startswith("test_shortcut_")]
            assert any(name.startswith(test[len("test_shortcut_"):]) for test in tests), name


def test_every_episode_replays_to_its_saved_final_state(results):
    """PLAN 5.4: the saved calls of an episode, run again on a fresh world, give the saved final state."""
    for name in RUNS:
        assert all(s["replays"] for s in results(name)), name


def test_the_verdict_agrees_with_a_second_opinion_that_reads_the_calls(results):
    """The verifier compares states. The second opinion never looks at a state: an episode is right if the
    writes and the close that the tools accepted are exactly those of the correct path. The two must agree on
    every episode: the verifier pays no wrong path and refuses no right one."""
    n_passed = 0
    for name in RUNS:
        for s in results(name):
            assert s["passed"] == s["writes_are_the_correct"], (name, s["family_id"], s["style"])
            assert s["passed"] == (s["differs_in"] == ()), (name, s["family_id"], s["style"])
            n_passed += s["passed"]
    n_failed = len(RUNS) * N_SCENARIOS - n_passed
    assert n_passed > 10_000 and n_failed > 20_000        # many passes and many fails were looked at


def test_a_fail_is_always_0_and_a_pass_is_never_below_77(results):
    seen: set[int] = set()
    for name in RUNS:
        for s in results(name):
            if s["passed"]:
                assert 77 <= s["reward"] <= 100 and s["closed"], (name, s)
            else:
                assert s["reward"] == 0, (name, s)
            seen.add(s["reward"])
    # 96 (two turns over the budget, nothing malformed) can come from the random policy only.
    assert seen | {96} == {0, 77, 79, 81, 88, 92, 94, 95, 96, 98, 100}


def test_no_episode_passes_without_both_reads(results):
    """PLAN 5.2: close_request needs both ids, and only the read tools return them. So in every passing
    episode a lookup_person and a check_access were answered before the close."""
    for name in RUNS:
        for s in results(name):
            if s["passed"]:
                accepted = [tool_key for tool_key, ok in s["calls"] if ok]
                assert accepted[-1] == "close_request", (name, s["family_id"])
                assert "lookup_person" in accepted[:-1], (name, s["family_id"])
                assert "check_access" in accepted[:-1], (name, s["family_id"])


def test_the_number_of_episodes_of_the_suite(results, capsys):
    total = sum(len(results(name)) for name in RUNS)
    assert total == len(RUNS) * N_SCENARIOS == 50 * 1080
    with capsys.disabled():
        print(f"\n[validation suite] {len(RUNS)} policy runs, {total:,} episodes")
