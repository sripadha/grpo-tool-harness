"""The validation suite on the mixed design, and the three findings on the design of the scenarios.

Calibration round 2 measures two designs, so both must hold what the reward promises:
  - the plain design (the default of accessdesk/config.py until the freeze day, 2026-11-18): the outcome
    fixes the three rule facts, and no two of them are true together;
  - the mixed design (cfg.mix_facts, here with a homonym in half of the families): a fact that only a later
    rule would look at may be true as well (accessdesk/env/generator.py, MIXED_KINDS). It is the final
    design, and the default of the config since the freeze day.
Both designs are built here with their two values set by hand (the fixtures `plain` and `mixed`), so the
tests of this file do not depend on which of the two is the default.

What is shown here, on families built in memory from the placeholder text (all four splits, every style):
  1. The rule order. A policy that knows the rules and tries them in another order passes every family of
     the plain design: that design cannot see the order. On the mixed design it fails on exactly the
     families whose kind makes the order matter; the kinds and the numbers are typed by hand below.
  2. The mixed design keeps what the reward promises: the oracle gets 100 everywhere, and each constant
     policy passes exactly its own quarter.
  3. The expected states of the mixed design, derived a second time: a truth table typed by hand, read with
     the facts of the stored world. Nothing of accessdesk/env/rules.py is called.
  4. Reading the department. A policy that takes the first match of a name passes a family without a
     homonym and half of the families with one, so its pass rate falls as the share of homonyms rises.
  5. The policy-shift controls. The constant policy that fits the changed rule passes every "differ"
     family and fails on every control.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from accessdesk.env.generator import SPLITS, make_all_splits, make_split
from accessdesk.env.scenarios import make_sets
from accessdesk.env.tools import execute
from accessdesk.env.world import World
from accessdesk.harness import policies as P
from accessdesk.harness.episode import Episode
from accessdesk.types import OUTCOMES
from tests.core.test_validation_suite import RUNS, SHORTCUTS_ON_EVERY_FAMILY, SHORTCUTS_ON_ONE_OUTCOME

SIZES = {"train": 200, "dev": 48, "test": 48, "calib": 24}
N_SCENARIOS = 1080                       # 200 * 3 + 48 * 4 + 48 * 4 + 24 * 4

# ---------------------------------------------------------------------------
# Typed by hand: the rules as a truth table, and the kinds of family of the mixed design
# ---------------------------------------------------------------------------
# (person active, person holds the role, role sensitive) -> the correct resolution under the rule sets
# (default, all_grants_need_approval, sensitive_denied). An inactive person is always denied; an active
# person who holds the role always gets "already"; only then does the role's sensitivity count.
RESOLUTION_BY_HAND = {
    (False, False, False): ("denied", "denied", "denied"),
    (False, False, True): ("denied", "denied", "denied"),
    (False, True, False): ("denied", "denied", "denied"),
    (False, True, True): ("denied", "denied", "denied"),
    (True, True, False): ("already", "already", "already"),
    (True, True, True): ("already", "already", "already"),
    (True, False, True): ("pending", "pending", "denied"),
    (True, False, False): ("granted", "pending", "granted"),
}
RULESET_COLUMN = {"default": 0, "all_grants_need_approval": 1, "sensitive_denied": 2}
OUTCOME_OF_RESOLUTION = {"denied": "inactive", "already": "already", "pending": "approval",
                         "granted": "grant"}

# The kinds of the mixed design: (outcome, facts) -> families per split of 200, 48, 48 and 24 families
# (50, 12, 12 and 6 of each outcome). Inactive: 1/3, 1/3, 1/6, 1/6; of 50 that is 16.67, 16.67, 8.33, 8.33,
# and the two units left over go to the two largest remainders: 17, 17, 8, 8. Already: 1/2, 1/2.
KIND_COUNTS = {
    ("grant", (True, False, False)): (50, 12, 12, 6),
    ("approval", (True, False, True)): (50, 12, 12, 6),
    ("already", (True, True, False)): (25, 6, 6, 3),
    ("already", (True, True, True)): (25, 6, 6, 3),             # active, role held AND sensitive
    ("inactive", (False, False, False)): (17, 4, 4, 2),
    ("inactive", (False, False, True)): (17, 4, 4, 2),          # inactive AND role sensitive
    ("inactive", (False, True, False)): (8, 2, 2, 1),           # inactive AND role held
    ("inactive", (False, True, True)): (8, 2, 2, 1),            # inactive AND role held AND sensitive
}
HELD_SENSITIVE = ("already", (True, True, True))
INACTIVE_SENSITIVE = ("inactive", (False, False, True))
INACTIVE_HELD = ("inactive", (False, True, False))
INACTIVE_HELD_SENSITIVE = ("inactive", (False, True, True))

# The five wrong orders of the three cases -> the kinds on which such a policy is wrong. A kind is lost
# when a case that applies is tried before the case that the desk tries first.
I, H, S = "inactive", "already", "sensitive"
WRONG_ORDERS = {
    # reverse: sensitive, then already held, then inactive. Wrong wherever two facts are true.
    (S, H, I): {HELD_SENSITIVE, INACTIVE_SENSITIVE, INACTIVE_HELD, INACTIVE_HELD_SENSITIVE},
    # only "sensitive" and "already held" change places: wrong where the role is held and sensitive and
    # the person is active (asks for approval; the desk says "already").
    (I, S, H): {HELD_SENSITIVE},
    # only "inactive" and "already held" change places: wrong where an inactive person holds the role
    # (says "already"; the desk denies).
    (H, I, S): {INACTIVE_HELD, INACTIVE_HELD_SENSITIVE},
    # "inactive" last: wrong for every inactive person with a second fact.
    (H, S, I): {INACTIVE_SENSITIVE, INACTIVE_HELD, INACTIVE_HELD_SENSITIVE},
    # "sensitive" first: wrong wherever the role is sensitive and another fact is true.
    (S, I, H): {HELD_SENSITIVE, INACTIVE_SENSITIVE, INACTIVE_HELD_SENSITIVE},
}
# Families on which each wrong order fails, per split (train, dev, test, calib): sums of KIND_COUNTS.
WRONG_ORDER_FAILS = {
    (S, H, I): (58, 14, 14, 7),          # 25 + 17 + 8 + 8;  6 + 4 + 2 + 2;  3 + 2 + 1 + 1
    (I, S, H): (25, 6, 6, 3),
    (H, I, S): (16, 4, 4, 2),
    (H, S, I): (33, 8, 8, 4),
    (S, I, H): (50, 12, 12, 6),
}


def facts_of(family: dict) -> tuple[bool, bool, bool]:
    """(active, holds the role, sensitive), read from the stored world. The person is found by the name and
    the department of the request and the role by its name: the `truth` ids are not used."""
    world, request = family["world"], family["request"]
    (person,) = [p for p in world["people"]
                 if (p["name"], p["department"]) == (request["person_name"], request["department"])]
    (role,) = [r for r in world["roles"] if r["name"] == request["role_name"]]
    return (person["status"] == "active", [person["person_id"], role["role_id"]] in world["access"],
            role["sensitive"])


def kind_of(family: dict) -> tuple[str, tuple[bool, bool, bool]]:
    return family["outcome"], facts_of(family)


def rules_that_apply(family: dict) -> int:
    """How many of the three cases of the rules are true: inactive, role held, role sensitive."""
    active, holds_role, sensitive = facts_of(family)
    return (not active) + holds_role + sensitive


def expected_by_hand(family: dict, ruleset: str) -> dict:
    """The correct end state, from RESOLUTION_BY_HAND and the stored world alone."""
    world, request = family["world"], family["request"]
    (person,) = [p for p in world["people"]
                 if (p["name"], p["department"]) == (request["person_name"], request["department"])]
    (role,) = [r for r in world["roles"] if r["name"] == request["role_name"]]
    pair = [person["person_id"], role["role_id"]]
    resolution = RESOLUTION_BY_HAND[facts_of(family)][RULESET_COLUMN[ruleset]]
    return {
        "access": sorted(world["access"] + ([pair] if resolution == "granted" else [])),
        "approvals": [pair + [person["manager_id"]]] if resolution == "pending" else [],
        "grants": [pair] if resolution == "granted" else [],
        "request": {"resolution": resolution, "person_id": pair[0], "role_id": pair[1]},
    }


def person_meant_is_the_first_match(family: dict) -> bool:
    matches = [p for p in family["world"]["people"] if p["name"] == family["request"]["person_name"]]
    return matches[0]["person_id"] == family["truth"]["person_id"]


# ---------------------------------------------------------------------------
# The families and the runs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mixed_cfg(cfg):
    """The design of the folder data/scenarios_calib2_mixed, at the default split sizes."""
    return dataclasses.replace(cfg, mix_facts=True, homonym_share=0.5)


@pytest.fixture(scope="module")
def mixed(text, pools, mixed_cfg) -> dict[str, list[dict]]:
    return make_all_splits(text, pools, mixed_cfg)


@pytest.fixture(scope="module")
def plain_cfg(cfg):
    """The design before the freeze day, with its two values set by hand: the outcome fixes the three rule
    facts, and one family in five has a homonym."""
    return dataclasses.replace(cfg, mix_facts=False, homonym_share=0.2)


@pytest.fixture(scope="module")
def plain(text, pools, plain_cfg) -> dict[str, list[dict]]:
    return make_all_splits(text, pools, plain_cfg)


def play(policy: P.Policy, families: list[dict], text, cfg, ruleset: str = "default") -> list[dict]:
    """One episode per scenario of the families: what the tests need of each record."""
    prompt = "policy_shift" if ruleset == text.policy_shift_variant else "final"
    played = []
    for family in families:
        for wording in family["wordings"]:
            episode = Episode(family, wording["style"], text, cfg, ruleset=ruleset, prompt=prompt)
            record = P.run_policy(episode, policy).record(run="mixed_design")
            played.append({key: record[key] for key in ("family_id", "split", "style", "outcome", "passed",
                                                        "reward", "closed", "n_turns", "turn_budget",
                                                        "n_malformed", "n_tool_errors")}
                          | {"resolution": record["final_state"]["request"]["resolution"]})
    return played


def in_all_splits(families: dict[str, list[dict]]) -> list[dict]:
    return [family for split in SPLITS for family in families[split]]


def first_match(text) -> P.Policy:
    """Follows the rules correctly for the FIRST person that the lookup returns. It never reads the
    department, so with two people of the requested name it is right only if the one meant comes first."""
    def script(desk: P.Desk) -> None:
        person = P._find_people(desk)[0]
        check = P._check(desk, person)
        action = P._action(desk, person, check)
        P._write(desk, action, person, check["role_id"])
        P._close(desk, action, person, check["role_id"])

    return P.scripted(text, script)


# ---------------------------------------------------------------------------
# The mixed design itself
# ---------------------------------------------------------------------------
def test_the_truth_table_names_every_combination_of_the_three_facts_once():
    assert sorted(RESOLUTION_BY_HAND) == sorted((a, h, s) for a in (False, True) for h in (False, True)
                                                for s in (False, True))
    assert set(OUTCOME_OF_RESOLUTION.values()) == set(OUTCOMES)
    # The kinds of the table are the eight combinations too, each under the outcome that the default
    # rules give it, and every outcome has a quarter of each split.
    assert sorted(facts for _, facts in KIND_COUNTS) == sorted(RESOLUTION_BY_HAND)
    for (outcome, facts) in KIND_COUNTS:
        assert OUTCOME_OF_RESOLUTION[RESOLUTION_BY_HAND[facts][0]] == outcome
    for outcome in OUTCOMES:
        per_split = [counts for (o, _), counts in KIND_COUNTS.items() if o == outcome]
        assert [sum(column) for column in zip(*per_split)] == [50, 12, 12, 6]


def test_the_kinds_of_the_mixed_design_have_their_exact_quotas_in_every_split(mixed):
    """Read from the stored worlds: every split holds exactly the hand-typed number of each kind."""
    assert {split: len(mixed[split]) for split in SPLITS} == SIZES
    for column, split in enumerate(SPLITS):
        counts = Counter(kind_of(family) for family in mixed[split])
        assert counts == {kind: per_split[column] for kind, per_split in KIND_COUNTS.items()}, split
        outcomes = Counter(family["outcome"] for family in mixed[split])
        assert outcomes == {outcome: SIZES[split] // 4 for outcome in OUTCOMES}, split
    # Two or three rule facts are true together in 93 of the 320 families; in the plain design in none.
    several = [family for family in in_all_splits(mixed) if rules_that_apply(family) > 1]
    assert len(several) == 58 + 14 + 14 + 7 == 93


def test_the_plain_design_never_has_two_rule_facts_together(plain):
    kinds = {("grant", (True, False, False)), ("approval", (True, False, True)),
             ("already", (True, True, False)), ("inactive", (False, False, False))}
    assert {kind_of(family) for family in in_all_splits(plain)} == kinds
    assert not any("facts" in family for family in in_all_splits(plain))


def test_a_mixed_family_records_its_facts_and_the_world_holds_them(mixed):
    for family in in_all_splits(mixed):
        assert list(family) == ["family_id", "split", "outcome", "homonym", "facts", "world_seed", "world",
                                "request", "truth", "expected", "turn_budget", "wordings"]
        assert list(family["facts"]) == ["active", "holds_role", "sensitive"]
        assert tuple(family["facts"].values()) == facts_of(family), family["family_id"]
        assert all(type(value) is bool for value in family["facts"].values())


def test_the_tools_show_the_facts_of_a_mixed_family_and_the_facts_key_is_never_read(mixed, text, cfg):
    """The lookup and the access check return the three facts. The "facts" key of the record is a label for
    people: an episode of the same family without the key is the same episode, message for message."""
    keys = text.result_keys
    for family in in_all_splits(mixed):
        world, request, truth = World(family["world"]), family["request"], family["truth"]
        lookup = json.loads(execute(world, text, "lookup_person", {"name": request["person_name"]}).response)
        (meant,) = [m for m in lookup[keys["matches"]] if m[keys["department"]] == request["department"]]
        check = json.loads(execute(world, text, "check_access", {"person_id": truth["person_id"],
                                                               "role_name": request["role_name"]}).response)
        active, holds_role, sensitive = facts_of(family)
        assert meant[keys["person_id"]] == truth["person_id"]
        assert meant[keys["status"]] == text.status["active" if active else "inactive"]
        assert check == {keys["role_id"]: truth["role_id"], keys["sensitive"]: sensitive,
                         keys["has_access"]: holds_role}
    for kind in (INACTIVE_HELD_SENSITIVE, HELD_SENSITIVE):
        family = next(f for f in mixed["dev"] if kind_of(f) == kind)
        without_key = {key: value for key, value in family.items() if key != "facts"}
        records = [P.run_policy(Episode(f, 1, text, cfg), P.oracle(text)).record()
                   for f in (family, without_key)]
        assert records[0] == records[1] and records[0]["passed"]
        assert [m["role"] for m in Episode(family, 1, text, cfg).messages] == ["system", "user"]
        assert Episode(family, 1, text, cfg).messages == Episode(without_key, 1, text, cfg).messages


# ---------------------------------------------------------------------------
# 3. The expected states, derived a second time
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("design", ["mixed", "plain"])
def test_expected_states_equal_the_truth_table_typed_by_hand(design, mixed, plain):
    """For every family and each of the three rule sets: the stored expected state (rules.expected_state)
    equals the state built here from the truth table and the stored world. The outcome label is the first
    rule that applies under the default rules, and the turn budget is the shortest path plus one."""
    built = mixed if design == "mixed" else plain
    seen = set()
    for family in in_all_splits(built):
        assert list(family["expected"]) == list(RULESET_COLUMN)
        for ruleset in RULESET_COLUMN:
            by_hand = expected_by_hand(family, ruleset)
            assert family["expected"][ruleset] == by_hand, (family["family_id"], ruleset)
            writes = by_hand["request"]["resolution"] in ("granted", "pending")
            assert family["turn_budget"][ruleset] == (5 if writes else 4)
        default_resolution = family["expected"]["default"]["request"]["resolution"]
        assert family["outcome"] == OUTCOME_OF_RESOLUTION[default_resolution], family["family_id"]
        seen.add(facts_of(family))
    assert len(seen) == (8 if design == "mixed" else 4)        # every row of the table was needed


def test_the_hand_check_of_expected_states_can_fail(mixed):
    """A family whose stored state follows another order of the rules is caught: take an inactive person
    with a sensitive role and store the state "approval asked" for it."""
    family = next(f for f in mixed["dev"] if kind_of(f) == INACTIVE_SENSITIVE)
    truth = family["truth"]
    wrong = {**family["expected"]["default"],
             "approvals": [[truth["person_id"], truth["role_id"], truth["manager_id"]]],
             "request": {**family["expected"]["default"]["request"], "resolution": "pending"}}
    assert wrong != expected_by_hand(family, "default") == family["expected"]["default"]


# ---------------------------------------------------------------------------
# 2. The oracle and the constant policies on the mixed design
# ---------------------------------------------------------------------------
def test_the_oracle_gets_100_on_every_scenario_of_the_mixed_design(mixed, text, cfg):
    played = play(P.oracle(text), in_all_splits(mixed), text, cfg)
    assert len(played) == N_SCENARIOS
    assert all(s["passed"] and s["reward"] == 100 and s["closed"] for s in played)
    assert all((s["n_malformed"], s["n_tool_errors"]) == (0, 0) for s in played)
    assert all(s["n_turns"] == s["turn_budget"] - 1 for s in played)
    assert {s["outcome"]: s["n_turns"] for s in played} == {"grant": 4, "approval": 4, "already": 3,
                                                            "inactive": 3}


@pytest.mark.parametrize("ruleset", ["all_grants_need_approval", "sensitive_denied"])
def test_the_oracle_of_another_rule_set_gets_100_on_the_mixed_design(mixed, text, cfg, ruleset):
    played = play(P.oracle(text, ruleset), in_all_splits(mixed), text, cfg, ruleset)
    assert len(played) == N_SCENARIOS and all(s["passed"] and s["reward"] == 100 for s in played)


@pytest.mark.parametrize("resolution, outcome", [("granted", "grant"), ("pending", "approval"),
                                                 ("already", "already"), ("denied", "inactive")])
def test_a_constant_policy_passes_exactly_a_quarter_of_the_mixed_design(mixed, text, cfg, resolution,
                                                                       outcome):
    """Whatever facts are mixed in, the outcomes are a quarter each, and a policy that always acts for one
    resolution passes its own outcome and nothing else, in every split: still the 25% floor."""
    played = play(P.constant(text, resolution), in_all_splits(mixed), text, cfg)
    assert len(played) == N_SCENARIOS
    assert all(s["passed"] == (s["outcome"] == outcome) for s in played)
    assert all(s["reward"] == (100 if s["passed"] else 0) for s in played)
    for split in SPLITS:
        passing = {s["family_id"] for s in played if s["split"] == split and s["passed"]}
        assert len(passing) * 4 == SIZES[split], split


def test_no_shortcut_of_the_validation_suite_is_paid_on_the_mixed_design(mixed, text, cfg):
    """The 25 shortcut policies of tests/core/test_validation_suite.py (all but the random one), played on
    every scenario of the mixed design. A shortcut that is taken on every family gets 0 everywhere. A
    shortcut that exists for one outcome only gets 0 on that outcome and, being the oracle elsewhere, 100
    on the three others: so a second true fact opens no way around the final-state check."""
    assert len(SHORTCUTS_ON_EVERY_FAMILY) + len(SHORTCUTS_ON_ONE_OUTCOME) == 16 + 9
    for name in SHORTCUTS_ON_EVERY_FAMILY:
        played = play(RUNS[name].make(text), in_all_splits(mixed), text, cfg)
        assert len(played) == N_SCENARIOS
        assert all(s["reward"] == 0 and s["passed"] is False for s in played), name
    for name, outcome in SHORTCUTS_ON_ONE_OUTCOME.items():
        played = play(RUNS[name].make(text), in_all_splits(mixed), text, cfg)
        assert len(played) == N_SCENARIOS
        assert all(s["reward"] == (0 if s["outcome"] == outcome else 100) for s in played), name
        assert all(s["passed"] == (s["outcome"] != outcome) for s in played), name


# ---------------------------------------------------------------------------
# 1. The order of the rules
# ---------------------------------------------------------------------------
def test_the_wrong_orders_are_the_five_other_orders_and_the_reverse_is_one_of_them(text):
    assert len(WRONG_ORDERS) == 5 and (I, H, S) not in WRONG_ORDERS and P.RULE_ORDER == (I, H, S)
    assert all(sorted(order) == sorted(P.RULE_ORDER) for order in WRONG_ORDERS)
    assert tuple(reversed(P.RULE_ORDER)) == (S, H, I)
    with pytest.raises(ValueError):
        P.rule_order(text, (I, H))
    with pytest.raises(ValueError):
        P.rule_order(text, (I, H, H))
    for order, kinds in WRONG_ORDERS.items():
        fails = tuple(sum(column) for column in zip(*(KIND_COUNTS[kind] for kind in kinds)))
        assert fails == WRONG_ORDER_FAILS[order], order


@pytest.mark.parametrize("order", list(WRONG_ORDERS))
def test_a_wrong_rule_order_passes_every_family_of_the_plain_design(plain, text, plain_cfg, order):
    """The finding: in the plain design no two rule facts are true together, so the first case that
    applies is the only case that applies, and the order of the cases is never tested."""
    played = play(P.rule_order(text, order), in_all_splits(plain), text, plain_cfg)
    assert len(played) == N_SCENARIOS
    assert all(s["passed"] and s["reward"] == 100 for s in played)


@pytest.mark.parametrize("order", list(WRONG_ORDERS))
def test_a_wrong_rule_order_fails_exactly_where_the_mixed_design_makes_the_order_matter(
        mixed, text, cfg, order):
    played = play(P.rule_order(text, order), in_all_splits(mixed), text, cfg)
    assert len(played) == N_SCENARIOS
    family_of = {family["family_id"]: family for family in in_all_splits(mixed)}
    for s in played:
        wrong_here = kind_of(family_of[s["family_id"]]) in WRONG_ORDERS[order]
        assert s["passed"] == (not wrong_here), (order, s["family_id"], s["style"])
        assert s["reward"] == (0 if wrong_here else 100) and s["closed"]
    for column, split in enumerate(SPLITS):
        failing = {s["family_id"] for s in played if s["split"] == split and not s["passed"]}
        assert len(failing) == WRONG_ORDER_FAILS[order][column], (order, split)


def test_the_reverse_order_fails_93_families_of_the_mixed_design(mixed, text, cfg):
    """Sensitive first, then already held, then inactive, else grant: wrong in 58 + 14 + 14 + 7 = 93 of the
    320 families (314 of the 1,080 scenarios), which is every family with two rule facts. What it does
    there: it asks for approval where the desk denies or says "already", or says "already" where the desk
    denies."""
    played = play(P.reverse_order(text), in_all_splits(mixed), text, cfg)
    failed = [s for s in played if not s["passed"]]
    assert len({s["family_id"] for s in failed}) == 93 and len(failed) == 58 * 3 + 35 * 4 == 314
    family_of = {family["family_id"]: family for family in in_all_splits(mixed)}
    did = Counter((kind_of(family_of[s["family_id"]]), s["resolution"]) for s in failed)
    assert set(did) == {(HELD_SENSITIVE, "pending"), (INACTIVE_SENSITIVE, "pending"),
                        (INACTIVE_HELD_SENSITIVE, "pending"), (INACTIVE_HELD, "already")}
    assert play(P.reverse_order(text), mixed["dev"], text, cfg) == \
        play(P.rule_order(text, (S, H, I)), mixed["dev"], text, cfg)


def test_the_right_order_is_the_oracle(mixed, text, cfg):
    by_order = play(P.rule_order(text, P.RULE_ORDER), mixed["dev"], text, cfg)
    assert by_order == play(P.oracle(text), mixed["dev"], text, cfg)
    assert all(s["passed"] for s in by_order) and len(by_order) == 192


# ---------------------------------------------------------------------------
# 4. Reading the department: the share of same-name families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("share, n_homonym, n_passed", [
    (0.2, 8, 44),       # 40 families without a homonym + 4 of the 8 with one: 91.7%
    (0.5, 24, 36),      # 24 + 12: 75%
    (1.0, 48, 24),      # 0 + 24: 50%
])
def test_a_policy_that_never_reads_the_department_passes_less_as_the_share_of_homonyms_rises(
        text, pools, cfg, share, n_homonym, n_passed):
    """Dev split, 48 families. "Take the first match" is right in every family without a homonym and in
    exactly half of the families with one (the place of the person meant is balanced). The oracle, which
    reads the department, passes everything at every share."""
    dev = make_split("dev", text, pools, dataclasses.replace(cfg, homonym_share=share))
    assert sum(family["homonym"] for family in dev) == n_homonym
    family_of = {family["family_id"]: family for family in dev}
    played = play(first_match(text), dev, text, cfg)
    for s in played:
        family = family_of[s["family_id"]]
        assert s["passed"] == (not family["homonym"] or person_meant_is_the_first_match(family))
    assert len({s["family_id"] for s in played if s["passed"]}) == n_passed
    assert all(s["passed"] and s["reward"] == 100 for s in play(P.oracle(text), dev, text, cfg))
    # The policy "the other person of that name" fails in every family, at every share.
    assert not any(s["passed"] for s in play(P.wrong_person(text, "homonym"), dev, text, cfg))


def test_the_first_match_policy_on_the_mixed_design_with_half_homonyms(mixed, text, cfg):
    """With a homonym in half of the families the policy passes three families in four, in every split."""
    played = play(first_match(text), in_all_splits(mixed), text, cfg)
    for split in SPLITS:
        passing = {s["family_id"] for s in played if s["split"] == split and s["passed"]}
        assert len(passing) * 4 == 3 * SIZES[split], split


# ---------------------------------------------------------------------------
# 5. The policy-shift controls
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("design", ["plain", "mixed"])
@pytest.mark.parametrize("variant, constant, differing, controls", [
    ("all_grants_need_approval", "pending", "grant", {"already": 3, "inactive": 2}),
    ("sensitive_denied", "denied", "approval", {"grant": 3, "already": 2}),
])
def test_the_constant_policy_of_the_changed_rule_passes_the_differ_families_and_no_control(
        design, variant, constant, differing, controls, plain, mixed, text, cfg):
    """all_grants_need_approval: "always ask the manager for approval and close as pending" is right on
    every family where the new rule changes the answer, without reading one fact. The controls must tell
    it from a model that follows the new rule, so it must fail on all 5 of them. Before the repair two of
    the five controls were approval families, on which it passes. The same for sensitive_denied and
    "always close as denied"."""
    dev = (plain if design == "plain" else mixed)["dev"]
    shift = make_sets(dev, cfg, variant)["policy_shift"]
    assert (len(shift["differ"]), len(shift["control"])) == (11, 5)
    by_id = {family["family_id"]: family for family in dev}
    assert {by_id[i]["outcome"] for i in shift["differ"]} == {differing}
    assert Counter(by_id[i]["outcome"] for i in shift["control"]) == controls
    chosen = [by_id[i] for i in shift["differ"] + shift["control"]]

    played = play(P.constant(text, constant), chosen, text, cfg, variant)
    assert len(played) == 16 * 4
    assert {s["passed"] for s in played if s["family_id"] in shift["differ"]} == {True}
    assert {s["passed"] for s in played if s["family_id"] in shift["control"]} == {False}
    # The policy for the new rule passes all 16; the policy for the old rule passes the controls only.
    assert all(s["passed"] for s in play(P.oracle(text, variant), chosen, text, cfg, variant))
    old = play(P.oracle(text), chosen, text, cfg, variant)
    assert {s["passed"] for s in old if s["family_id"] in shift["differ"]} == {False}
    assert {s["passed"] for s in old if s["family_id"] in shift["control"]} == {True}
