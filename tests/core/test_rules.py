"""accessdesk/env/rules.py: the decision table, the shortest correct plan and the expected state."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from accessdesk import impl
from accessdesk.env.rules import RULESETS, decide, expected_state, reference_plan, target_facts, turn_budget
from accessdesk.env.tools import execute
from accessdesk.env.world import World

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# decide: the whole table, typed by hand
# ---------------------------------------------------------------------------
# (active, has_access, sensitive) -> action under (default, all_grants_need_approval, sensitive_denied)
TABLE = {
    (True, False, False): ("grant", "approval", "grant"),
    (True, False, True): ("approval", "approval", "deny"),
    (True, True, False): ("already", "already", "already"),
    (True, True, True): ("already", "already", "already"),      # "has access" is checked before "sensitive"
    (False, False, False): ("deny", "deny", "deny"),
    (False, False, True): ("deny", "deny", "deny"),              # "inactive" is checked first
    (False, True, False): ("deny", "deny", "deny"),
    (False, True, True): ("deny", "deny", "deny"),
}


def test_rulesets():
    assert RULESETS == ("default", "all_grants_need_approval", "sensitive_denied")


@pytest.mark.parametrize("facts", list(TABLE))
def test_decide_matches_the_table(facts):
    assert tuple(decide(*facts, ruleset=ruleset) for ruleset in RULESETS) == TABLE[facts]
    assert decide(*facts) == TABLE[facts][0]             # the default rule set is the default argument


def test_decide_refuses_an_unknown_rule_set():
    with pytest.raises(ValueError):
        decide(True, False, False, "anything_goes")
    with pytest.raises(ValueError):
        decide(False, False, False, "")                  # also when the first rule would already apply


# ---------------------------------------------------------------------------
# Four families written by hand (no generator), one per outcome
# ---------------------------------------------------------------------------
WORLD = {
    "people": [
        {"person_id": "p1", "name": "Ada Stone", "department": "Finance", "status": "active",
         "manager_id": "p3"},
        {"person_id": "p2", "name": "Cy Lake", "department": "Sales", "status": "inactive",
         "manager_id": "p3"},
        {"person_id": "p3", "name": "Bo Reed", "department": "Finance", "status": "active",
         "manager_id": None},
    ],
    "roles": [
        {"role_id": "r1", "name": "Wiki Editor", "sensitive": False},
        {"role_id": "r2", "name": "Payroll Editor", "sensitive": True},
        {"role_id": "r3", "name": "Help Desk Agent", "sensitive": False},
    ],
    "access": [["p1", "r1"], ["p3", "r2"]],
}
ACCESS = [["p1", "r1"], ["p3", "r2"]]


def family(person_id: str, role_id: str) -> dict:
    person = next(p for p in WORLD["people"] if p["person_id"] == person_id)
    role = next(r for r in WORLD["roles"] if r["role_id"] == role_id)
    return {"world": copy.deepcopy(WORLD),
            "request": {"person_name": person["name"], "department": person["department"],
                        "role_name": role["name"]},
            "truth": {"person_id": person_id, "role_id": role_id, "manager_id": person["manager_id"]}}


def state(access: list, approvals: list, grants: list, resolution: str, person_id: str, role_id: str) -> dict:
    return {"access": access, "approvals": approvals, "grants": grants,
            "request": {"resolution": resolution, "person_id": person_id, "role_id": role_id}}


GRANT = family("p1", "r3")        # active, lacks the role, role not sensitive
APPROVAL = family("p1", "r2")     # active, lacks the role, role sensitive
ALREADY = family("p1", "r1")      # active, has the role
INACTIVE = family("p2", "r3")     # inactive

GRANTED_STATE = state([["p1", "r1"], ["p1", "r3"], ["p3", "r2"]], [], [["p1", "r3"]], "granted", "p1", "r3")
PENDING_R3 = state(ACCESS, [["p1", "r3", "p3"]], [], "pending", "p1", "r3")
PENDING_R2 = state(ACCESS, [["p1", "r2", "p3"]], [], "pending", "p1", "r2")
DENIED_R2 = state(ACCESS, [], [], "denied", "p1", "r2")
ALREADY_STATE = state(ACCESS, [], [], "already", "p1", "r1")
DENIED_P2 = state(ACCESS, [], [], "denied", "p2", "r3")

# (family, rule set) -> the expected state, written by hand
EXPECTED = [
    ("grant", GRANT, "default", GRANTED_STATE),
    ("grant", GRANT, "all_grants_need_approval", PENDING_R3),
    ("grant", GRANT, "sensitive_denied", GRANTED_STATE),
    ("approval", APPROVAL, "default", PENDING_R2),
    ("approval", APPROVAL, "all_grants_need_approval", PENDING_R2),
    ("approval", APPROVAL, "sensitive_denied", DENIED_R2),
    ("already", ALREADY, "default", ALREADY_STATE),
    ("already", ALREADY, "all_grants_need_approval", ALREADY_STATE),
    ("already", ALREADY, "sensitive_denied", ALREADY_STATE),
    ("inactive", INACTIVE, "default", DENIED_P2),
    ("inactive", INACTIVE, "all_grants_need_approval", DENIED_P2),
    ("inactive", INACTIVE, "sensitive_denied", DENIED_P2),
]
EXPECTED_IDS = [f"{label}/{ruleset}" for label, _, ruleset, _ in EXPECTED]


def test_target_facts():
    assert target_facts(GRANT) == (True, False, False)
    assert target_facts(APPROVAL) == (True, False, True)
    assert target_facts(ALREADY) == (True, True, False)
    assert target_facts(INACTIVE) == (False, False, False)
    assert target_facts(family("p3", "r2")) == (True, True, True)


@pytest.mark.parametrize("label, fam, ruleset, wanted", EXPECTED, ids=EXPECTED_IDS)
def test_expected_state_of_hand_made_families(label, fam, ruleset, wanted):
    assert expected_state(fam, ruleset) == wanted


def test_expected_state_defaults_to_the_default_rule_set():
    assert expected_state(GRANT) == GRANTED_STATE
    assert reference_plan(GRANT) == reference_plan(GRANT, "default")
    assert turn_budget(GRANT) == turn_budget(GRANT, "default")


def test_rule_order_on_mixed_facts_that_the_generator_never_makes():
    inactive_holder = family("p2", "r1")
    inactive_holder["world"]["access"].append(["p2", "r1"])           # inactive AND already holds the role
    inactive_holder["world"]["roles"][0]["sensitive"] = True          # ... AND the role is sensitive
    for ruleset in RULESETS:
        assert expected_state(inactive_holder, ruleset) == state(
            [["p1", "r1"], ["p2", "r1"], ["p3", "r2"]], [], [], "denied", "p2", "r1")
    sensitive_holder = family("p3", "r2")                             # active, holds a sensitive role
    for ruleset in RULESETS:
        assert expected_state(sensitive_holder, ruleset) == state(ACCESS, [], [], "already", "p3", "r2")


def test_expected_state_lists_are_sorted_even_if_the_world_lists_are_not():
    fam = family("p1", "r3")
    fam["world"]["access"] = [["p3", "r2"], ["p1", "r1"]]
    assert expected_state(fam)["access"] == [["p1", "r1"], ["p1", "r3"], ["p3", "r2"]]


@pytest.mark.parametrize("label, fam, ruleset, wanted", EXPECTED, ids=EXPECTED_IDS)
def test_expected_state_does_not_change_or_share_the_family(label, fam, ruleset, wanted):
    before = copy.deepcopy(fam)
    result = expected_state(fam, ruleset)
    result["access"].append(["x", "y"])
    result["access"][0][0] = "changed"
    result["request"]["resolution"] = "changed"
    assert fam == before
    assert expected_state(fam, ruleset) == wanted


LOOKUP = ("lookup_person", {"name": "Ada Stone"})


def test_the_shortest_plan_of_hand_made_families():
    ids_r3 = {"person_id": "p1", "role_id": "r3"}
    assert reference_plan(GRANT) == [
        LOOKUP,
        ("check_access", {"person_id": "p1", "role_name": "Help Desk Agent"}),
        ("grant_role", ids_r3),
        ("close_request", {"resolution": "granted", **ids_r3}),
    ]
    assert reference_plan(GRANT, "all_grants_need_approval") == [
        LOOKUP,
        ("check_access", {"person_id": "p1", "role_name": "Help Desk Agent"}),
        ("request_approval", {**ids_r3, "approver_id": "p3"}),
        ("close_request", {"resolution": "pending", **ids_r3}),
    ]
    assert reference_plan(APPROVAL) == [
        LOOKUP,
        ("check_access", {"person_id": "p1", "role_name": "Payroll Editor"}),
        ("request_approval", {"person_id": "p1", "role_id": "r2", "approver_id": "p3"}),
        ("close_request", {"resolution": "pending", "person_id": "p1", "role_id": "r2"}),
    ]
    assert reference_plan(APPROVAL, "sensitive_denied") == [
        LOOKUP,
        ("check_access", {"person_id": "p1", "role_name": "Payroll Editor"}),
        ("close_request", {"resolution": "denied", "person_id": "p1", "role_id": "r2"}),
    ]
    assert reference_plan(ALREADY) == [
        LOOKUP,
        ("check_access", {"person_id": "p1", "role_name": "Wiki Editor"}),
        ("close_request", {"resolution": "already", "person_id": "p1", "role_id": "r1"}),
    ]
    assert reference_plan(INACTIVE) == [
        ("lookup_person", {"name": "Cy Lake"}),
        ("check_access", {"person_id": "p2", "role_name": "Help Desk Agent"}),
        ("close_request", {"resolution": "denied", "person_id": "p2", "role_id": "r3"}),
    ]


def test_the_shortest_plan_keeps_the_parameter_order_of_each_tool():
    from accessdesk.text import TOOL_PARAMS
    for _, fam, ruleset, _ in EXPECTED:
        for tool_key, args in reference_plan(fam, ruleset):
            assert tuple(args) == TOOL_PARAMS[tool_key]


def test_turn_budget_is_the_shortest_path_plus_one():
    assert [turn_budget(f) for f in (GRANT, APPROVAL, ALREADY, INACTIVE)] == [5, 5, 4, 4]
    assert turn_budget(GRANT, "all_grants_need_approval") == 5
    assert turn_budget(APPROVAL, "sensitive_denied") == 4
    for _, fam, ruleset, _ in EXPECTED:
        assert turn_budget(fam, ruleset) == len(reference_plan(fam, ruleset)) + 1


@pytest.mark.parametrize("label, fam, ruleset, wanted", EXPECTED, ids=EXPECTED_IDS)
def test_running_the_shortest_plan_through_the_tools_gives_the_expected_state(text, label, fam, ruleset,
                                                                              wanted):
    world = World(fam["world"])
    outcomes = [execute(world, text, tool_key, args) for tool_key, args in reference_plan(fam, ruleset)]
    assert [outcome.error for outcome in outcomes] == [None] * len(outcomes)
    assert [outcome.closes for outcome in outcomes] == [False] * (len(outcomes) - 1) + [True]
    assert world.state() == wanted


def test_a_family_whose_truth_is_not_in_its_world_is_refused():
    for key in ("person_id", "role_id"):
        broken = family("p1", "r3")
        broken["truth"][key] = "nowhere"
        with pytest.raises(ValueError):
            expected_state(broken)


# ---------------------------------------------------------------------------
# expected_state is independent of the tools
# ---------------------------------------------------------------------------
def test_rules_module_does_not_import_the_tools_or_the_world():
    tree = ast.parse((ROOT / "accessdesk/env/rules.py").read_text(encoding="utf-8"))
    imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in imported for alias in node.names}
    modules = {node.module for node in imported if isinstance(node, ast.ImportFrom)}
    assert names | modules == {"annotations", "__future__"}


def test_expected_state_works_when_the_tools_cannot_run(monkeypatch):
    def broken(*args, **kwargs):
        raise AssertionError("expected_state must not run a tool")
    monkeypatch.setattr("accessdesk.env.tools.execute", broken)
    monkeypatch.setattr("accessdesk.env.world.World.state", broken)
    assert expected_state(GRANT) == GRANTED_STATE


# ---------------------------------------------------------------------------
# Every generated family, every rule set
# ---------------------------------------------------------------------------
# outcome -> action under (default, all_grants_need_approval, sensitive_denied), typed by hand
ACTION_OF_OUTCOME = {
    "grant": ("grant", "approval", "grant"),
    "approval": ("approval", "approval", "deny"),
    "already": ("already", "already", "already"),
    "inactive": ("deny", "deny", "deny"),
}
RESOLUTION = {"grant": "granted", "approval": "pending", "already": "already", "deny": "denied"}
BUDGET = {"grant": 5, "approval": 5, "already": 4, "deny": 4}


def verify(final_state: dict, expected_state_: dict) -> bool:
    """verify() of the version in use: the main version, or the second one with ACCESSDESK_IMPL=second."""
    return impl.get("verify")(final_state, expected_state_)


def all_families(families: dict) -> list[dict]:
    return [fam for split in ("train", "dev", "test", "calib") for fam in families[split]]


def test_stored_expected_state_has_the_right_shape_for_every_family_and_rule_set(families):
    checked = 0
    for fam in all_families(families):
        person, role, manager = fam["truth"]["person_id"], fam["truth"]["role_id"], fam["truth"]["manager_id"]
        initial = sorted(list(pair) for pair in fam["world"]["access"])
        assert sorted(fam["expected"]) == sorted(RULESETS) == sorted(fam["turn_budget"])
        for ruleset, action in zip(RULESETS, ACTION_OF_OUTCOME[fam["outcome"]]):
            expected = fam["expected"][ruleset]
            assert list(expected) == ["access", "approvals", "grants", "request"]
            assert expected["request"] == {"resolution": RESOLUTION[action], "person_id": person,
                                           "role_id": role}
            if action == "grant":
                assert [person, role] not in initial
                assert expected["access"] == sorted(initial + [[person, role]])
                assert expected["grants"] == [[person, role]]
            else:
                assert expected["access"] == initial
                assert expected["grants"] == []
            assert expected["approvals"] == ([[person, role, manager]] if action == "approval" else [])
            assert fam["turn_budget"][ruleset] == BUDGET[action]
            assert expected == expected_state(fam, ruleset)
            checked += 1
    assert checked == 320 * 3


def test_expected_states_of_two_rule_sets_differ_exactly_when_the_actions_differ(families):
    for fam in all_families(families):
        actions = dict(zip(RULESETS, ACTION_OF_OUTCOME[fam["outcome"]]))
        for a in RULESETS:
            for b in RULESETS:
                assert (fam["expected"][a] == fam["expected"][b]) == (actions[a] == actions[b])


def test_the_shortest_plan_through_the_tools_gives_the_expected_state_for_every_family(families, text):
    """The independent check of expected_state: it is built without the tools, the plan runs through them."""
    checked = 0
    for fam in all_families(families):
        for ruleset in RULESETS:
            world = World(fam["world"])
            plan = reference_plan(fam, ruleset)
            outcomes = [execute(world, text, tool_key, args) for tool_key, args in plan]
            assert all(outcome.error is None for outcome in outcomes)
            assert world.closed and outcomes[-1].closes
            assert world.state() == fam["expected"][ruleset]
            assert verify(world.state(), fam["expected"][ruleset]) is True
            checked += 1
    assert checked == 320 * 3


def test_an_untouched_world_matches_no_expected_state(families):
    for fam in all_families(families):
        untouched = World(fam["world"]).state()
        for ruleset in RULESETS:
            assert verify(untouched, fam["expected"][ruleset]) is False

