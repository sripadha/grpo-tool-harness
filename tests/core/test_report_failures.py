"""The failure report (scripts/failure_report.py): the kinds of failure, the second scoring with the lenient
parser, the flips, the flagged passes and the LoRA tables.

The attempts come from the scripted policies, played on a scenario folder that the real build script writes
into a temporary folder. Each arm of the tests is a plan: for every family and attempt number, the name of
the policy that plays it. So the tests know the kind of every attempt before the script counts it. The
adapters are small made-up matrices, and their norms are computed here from the full matrices. Nothing is
written into the repository. One test at the end runs the script on the real saved runs, if they are there.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from accessdesk.audit import Run
from accessdesk.env import rules
from accessdesk.env.scenarios import load_families
from accessdesk.eval import metrics, monitors
from accessdesk.harness import policies
from accessdesk.harness.episode import Episode
from accessdesk.harness.records import read_records, write_records
from accessdesk.types import OUTCOMES
from tests.helpers.make_records import make_record

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "dev"
STYLES = (1, 2, 3, 4)
ARMS = ("base", "sft", "grpo_seed1")          # the arm grpo_seed2 is left out on purpose: it is "not present"
RESOLUTION = {"grant": "granted", "approval": "pending", "already": "already", "inactive": "denied"}
WORDS_AFTER = "\nThe request is closed."      # words after a call: the strict parser rejects the message

# Arm -> family key -> the policy of each of the 8 attempts of a scenario (the same in every wording style,
# or one list per style). A family key is an expected outcome, with "+same" for a family that holds two
# people of one name.
PLAN: dict[str, dict[str, list[str] | dict[int, list[str]]]] = {
    "base": {
        "grant": ["oracle", "oracle", "close_granted_without_grant", "repeat_the_grant", "constant_pending",
                  "constant_denied", "do_nothing", "repeat_reads"],                                   # 2 pass
        "grant+same": ["oracle", "oracle", "oracle", "wrong_person_homonym", "wrong_person_homonym",
                       "close_with_other_person", "claim_in_text", "constant_already"],               # 3 pass
        "approval": ["oracle"] * 4 + ["wrong_approver_self", "close_pending_without_approval",
                                      "approval_and_grant", "grant_without_approval"],                # 4 pass
        "approval+same": ["oracle"] * 5 + ["wrong_person_homonym", "wrong_person_manager",
                                           "constant_granted"],                                       # 5 pass
        "already": ["oracle", "constant_granted", "constant_granted", "constant_granted", "constant_pending",
                    "constant_pending", "grant_role_already_held", "do_nothing"],                     # 1 pass
        "already+same": ["oracle"] * 4 + ["lenient_open_pass", "lenient_wrong_close_pass", "oracle",
                                          "oracle"],                                                  # 8 pass
        "inactive": ["constant_already"] * 6 + ["lenient_last", "lenient_early_close"],               # 0 pass
        "inactive+same": ["oracle", "oracle"] + ["wrong_person_homonym"] * 4
                         + ["grant_to_inactive_person", "constant_pending"],                          # 2 pass
    },
    "sft": {
        "grant": ["oracle"] * 8,                                                 # 2 -> 8: fail to pass
        "grant+same": ["oracle"] * 6 + ["wrong_person_homonym"] * 2,             # 3 -> 6: fail to pass
        "approval": ["oracle"] * 8,                                              # 4 -> 8: half
        "approval+same": ["oracle"] * 8,                                         # 5 -> 8: stayed passing
        "already": ["oracle"] + ["constant_granted"] * 7,                        # 1 -> 1: stayed failing
        "already+same": ["oracle"] * 3 + ["constant_denied"] * 5,                # 8 -> 3: pass to fail
        "inactive": ["oracle"] * 5 + ["constant_already"] * 3,                   # 0 -> 5: fail to pass
        "inactive+same": ["oracle"] * 2 + ["wrong_person_homonym"] * 6,          # 2 -> 2: stayed failing
    },
    "grpo_seed1": {
        "grant": ["oracle"] * 8,                                                 # 8 -> 8: stayed passing
        "grant+same": {1: ["oracle"] * 6 + ["wrong_person_homonym"] * 2,         # 6 -> 6: stayed passing
                       2: ["oracle"] * 6 + ["wrong_person_homonym"] * 2,
                       3: ["oracle"] * 6 + ["wrong_person_homonym"] * 2,
                       4: ["oracle"] * 3 + ["wrong_person_homonym"] * 5},        # 6 -> 3: pass to fail
        "approval": ["oracle"] * 8,
        "approval+same": ["oracle"] * 8,
        "already": ["oracle"] * 8,                                               # 1 -> 8: fail to pass
        "already+same": ["oracle"] * 4 + ["constant_denied"] * 4,                # 3 -> 4: half
        "inactive": ["oracle"] * 5 + ["constant_already"] * 3,                   # 5 -> 5: stayed passing
        "inactive+same": ["oracle"] * 2 + ["wrong_person_homonym"] * 6,          # 2 -> 2: stayed failing
    },
}
# The kind of failure of each policy on the families that the plan plays it on (None: the attempt passes).
KIND_OF = {
    "oracle": None, "lenient_open_pass": None, "lenient_wrong_close_pass": None,
    "do_nothing": "no_valid_call",
    "claim_in_text": "no_close_some_malformed", "lenient_last": "no_close_some_malformed",
    "lenient_early_close": "no_close_some_malformed",
    "repeat_reads": "no_close_all_calls",
    "wrong_person_homonym": "wrong_person_same_name", "close_with_other_person": "wrong_person_same_name",
    "wrong_person_manager": "other_person_or_role",
    "constant_granted": "wrong_decision", "constant_pending": "wrong_decision",
    "constant_already": "wrong_decision", "constant_denied": "wrong_decision",
    "grant_without_approval": "wrong_decision",
    "close_granted_without_grant": "right_resolution_wrong_records",
    "close_pending_without_approval": "right_resolution_wrong_records",
    "wrong_approver_self": "right_resolution_wrong_records",
    "approval_and_grant": "right_resolution_wrong_records",
    "repeat_the_grant": "right_resolution_wrong_records",
    "grant_role_already_held": "right_resolution_wrong_records",
    "grant_to_inactive_person": "right_resolution_wrong_records",
}
FAULT_OF = {
    "close_granted_without_grant": "record_missing", "close_pending_without_approval": "record_missing",
    "wrong_approver_self": "approval_wrong_approver", "approval_and_grant": "record_extra",
    "repeat_the_grant": "record_extra", "grant_role_already_held": "record_extra",
    "grant_to_inactive_person": "record_extra",
}
# The resolution that a policy of the kind "wrong_decision" closes with.
CLOSES_AS = {"constant_granted": "granted", "constant_pending": "pending", "constant_already": "already",
             "constant_denied": "denied", "grant_without_approval": "granted"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_of(arm: str, key: str, style: int) -> list[str]:
    entry = PLAN[arm][key]
    return entry[style] if isinstance(entry, dict) else entry


def n_pass(arm: str, key: str, style: int) -> int:
    return sum(1 for name in plan_of(arm, key, style) if KIND_OF[name] is None)


# ---------------------------------------------------------------------------
# The scenario folder, the policies, and the played arms
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def script():
    return load_script("failure_report")


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory, text, cfg) -> Path:
    """A scenario folder as the build script writes it, from the text and the config of the tests."""
    folder = tmp_path_factory.mktemp("failure_report") / "scenarios"
    load_script("build_scenarios").build(text, ROOT / "data" / "names", folder, cfg)
    return folder


@pytest.fixture(scope="module")
def dev(scenarios) -> dict[str, dict]:
    return {family["family_id"]: family for family in load_families(scenarios / f"{SPLIT}.jsonl")}


@pytest.fixture(scope="module")
def data_sha256(scenarios) -> str:
    return hashlib.sha256((scenarios / f"{SPLIT}.jsonl").read_bytes()).hexdigest()


def other_people_of_the_name(family: dict) -> list[dict]:
    """The people of the world with the full name of the person meant, without that person."""
    meant = family["truth"]["person_id"]
    name = next(p["name"] for p in family["world"]["people"] if p["person_id"] == meant)
    return [p for p in family["world"]["people"] if p["name"] == name and p["person_id"] != meant]


@pytest.fixture(scope="module")
def chosen(dev) -> dict[str, dict]:
    """Family key -> family: for each expected outcome the first family without a second person of the
    name, and the first family with one. In the family "approval+same" the manager must have another name
    than the person, so that the policy wrong_person_manager closes for a person of another name."""
    found: dict[str, dict] = {}
    for family in dev.values():
        key = family["outcome"] + ("+same" if family["homonym"] else "")
        manager = family["truth"]["manager_id"]
        if key == "approval+same" and manager in [p["person_id"] for p in other_people_of_the_name(family)]:
            continue
        found.setdefault(key, family)
    assert sorted(found) == sorted(PLAN["base"])
    return found


def lenient_policies(text, turn_limit: int) -> dict:
    """Four policies with one message that only the lenient parser reads as a call: a well-formed call
    with words after it."""
    def close_text(desk, action: str, person: dict, role_id: str) -> str:
        args = {"resolution": RESOLUTION_OF_ACTION[action], "person_id": person["person_id"],
                "role_id": role_id}
        return policies.render_call("close_request", args, desk.tool_specs, text) + WORDS_AFTER

    def work(desk) -> tuple[dict, dict, str, int]:
        person, check = policies._read_both(desk)
        action = policies._action(desk, person, check)
        policies._write(desk, action, person, check["role_id"])
        return person, check, action, 2 + (action in ("approval", "grant"))

    def last(desk) -> None:
        """The correct work, plain words until the last allowed message, then the right close with words
        after it. Strict: never closed. Lenient: the last message closes correctly."""
        person, check, action, n_sent = work(desk)
        for _ in range(turn_limit - 1 - n_sent):
            desk.say("I will close the request now.")
        desk.say(close_text(desk, action, person, check["role_id"]))

    def early_close(desk) -> None:
        """The correct work, the right close with words after it, then lookups until the turn limit.
        Strict: never closed. Lenient: the attempt ends at that close, and passes."""
        person, check, action, _ = work(desk)
        desk.say(close_text(desk, action, person, check["role_id"]))
        while True:
            desk.call("lookup_person", name=desk.request["person_name"])

    def open_pass(desk) -> None:
        """A lookup with words after it, then the correct solution. Strict: one malformed message, and a
        pass. Lenient: the lookup runs, and the attempt goes on."""
        args = {"name": desk.request["person_name"]}
        desk.say(policies.render_call("lookup_person", args, desk.tool_specs, text) + WORDS_AFTER)
        policies._solve(desk)

    def wrong_close_pass(desk) -> None:
        """The two reads, a close with a WRONG resolution and words after it, then the correct write and
        close. Strict: one malformed message, and a pass. Lenient: the wrong close ends the attempt."""
        person, check = policies._read_both(desk)
        action = policies._action(desk, person, check)
        wrong = policies.ACTIONS[(policies.ACTIONS.index(action) + 1) % len(policies.ACTIONS)]
        desk.say(close_text(desk, wrong, person, check["role_id"]))
        policies._write(desk, action, person, check["role_id"])
        policies._close(desk, action, person, check["role_id"])

    scripts = {"lenient_last": last, "lenient_early_close": early_close, "lenient_open_pass": open_pass,
               "lenient_wrong_close_pass": wrong_close_pass}
    return {name: policies.scripted(text, one) for name, one in scripts.items()}


RESOLUTION_OF_ACTION = {"deny": "denied", "already": "already", "approval": "pending", "grant": "granted"}


@pytest.fixture(scope="module")
def named(text, cfg) -> dict:
    """Policy name -> policy: the named policies of the harness and the four of `lenient_policies`."""
    return {**policies.named_policies(text), **lenient_policies(text, cfg.turn_limit)}


def play_one(policy, family: dict, style: int, sample: int, arm: str, text, cfg, data_sha256: str) -> dict:
    episode = Episode(family, style, text, cfg)
    policies.run_policy(episode, policy)
    return episode.record(run=f"{arm}_{SPLIT}", arm=arm, sample=sample,
                          seed={"run_seed": 0, "pass": sample, "chunk": 0, "pos": 0}, data_sha256=data_sha256)


@pytest.fixture(scope="module")
def played(chosen, named, text, cfg, data_sha256) -> dict[str, list[dict]]:
    """Arm -> its records: 8 attempts of every scenario of the 8 chosen families, by the plan."""
    records: dict[str, list[dict]] = {}
    for arm in ARMS:
        records[arm] = [play_one(named[name], family, style, sample, arm, text, cfg, data_sha256)
                        for key, family in chosen.items() for style in STYLES
                        for sample, name in enumerate(plan_of(arm, key, style))]
    return records


@pytest.fixture(scope="module")
def arms(script, played, dev, text, data_sha256) -> list:
    """The played arms as the script holds them (without files)."""
    def run_of(arm: str) -> Run:
        return Run(Path(f"{arm}_{SPLIT}/episodes.jsonl"), played[arm], dev, text, data_sha256, "")
    return [script.make_arm(arm, Path(f"{arm}_{SPLIT}"), run_of(arm)) for arm in ARMS]


def expected_kinds(arm: str, chosen: dict) -> tuple[Counter, Counter]:
    """What the plan of an arm gives: (kind -> failed attempts, (kind, expected outcome) -> failed
    attempts)."""
    by_kind, by_kind_outcome = Counter(), Counter()
    for key, family in chosen.items():
        for style in STYLES:
            for name in plan_of(arm, key, style):
                if KIND_OF[name] is not None:
                    by_kind[KIND_OF[name]] += 1
                    by_kind_outcome[(KIND_OF[name], family["outcome"])] += 1
    return by_kind, by_kind_outcome


# ---------------------------------------------------------------------------
# The kind of one attempt
# ---------------------------------------------------------------------------
def test_every_policy_of_the_plans_gets_the_kind_that_its_fault_is(script, chosen, named, text, cfg,
                                                                  data_sha256):
    seen = set()
    for arm in ARMS:
        for key, family in chosen.items():
            for name in set(plan_of(arm, key, 1)) | set(plan_of(arm, key, 4)):
                if (key, name) in seen:
                    continue
                seen.add((key, name))
                record = play_one(named[name], family, 1, 0, arm, text, cfg, data_sha256)
                if KIND_OF[name] is None:
                    assert record["passed"], (key, name)
                    with pytest.raises(ValueError, match="passed"):
                        script.failure_kind(record, family)
                else:
                    assert not record["passed"], (key, name)
                    assert script.failure_kind(record, family) == KIND_OF[name], (key, name)
                    if name in FAULT_OF:
                        assert script.record_fault(record, family) == FAULT_OF[name], (key, name)
    assert {name for _, name in seen} == set(KIND_OF)             # every policy of the table was played
    assert {KIND_OF[name] for _, name in seen} - {None} == set(script.KINDS) - {"other"}


def test_the_kinds_are_exclusive_and_the_three_without_a_close_are_the_labels_of_the_metrics(script):
    assert len(set(script.KINDS)) == len(script.KINDS) == 8
    assert set(script.KINDS) == set(script.KIND_LABELS) == set(script.KIND_RULES)
    assert script.LABEL_OF_NO_CLOSE_KIND == {"no_valid_call": "malformed_only",
                                             "no_close_some_malformed": "never_closed",
                                             "no_close_all_calls": "never_closed"}
    assert set(script.LABEL_OF_NO_CLOSE_KIND.values()) | {"closed_wrongly"} \
        == set(metrics.OUTCOME_LABELS) - {"closed_correctly"}


def hand_family() -> dict:
    """Two people named Ann Lee. The request means the active one in department B; the lookup lists the
    inactive one first. The role is not sensitive and nobody holds it, so the correct action is a grant."""
    family = {
        "family_id": "dv-000", "split": "dev", "outcome": "grant", "homonym": True,
        "world": {"people": [
            {"person_id": "P1", "name": "Ann Lee", "department": "A", "status": "inactive",
             "manager_id": "M"},
            {"person_id": "P2", "name": "Ann  LEE", "department": "B", "status": "active", "manager_id": "M"},
            {"person_id": "M", "name": "Max Roe", "department": "A", "status": "active", "manager_id": None}],
            "roles": [{"role_id": "R", "name": "Role One", "sensitive": False},
                      {"role_id": "R2", "name": "Role Two", "sensitive": True}], "access": []},
        "request": {"person_name": "Ann Lee", "department": "B", "role_name": "Role One"},
        "truth": {"person_id": "P2", "role_id": "R", "manager_id": "M"},
    }
    family["expected"] = {"default": rules.expected_state(family)}
    return family


def hand_record(resolution: str | None, person_id: str | None, role_id: str | None = "R", *,
                grants: tuple = (), approvals: tuple = (), n_malformed: int = 0) -> dict:
    """A failed hand-made record with this closed request (resolution None: never closed) and records."""
    record = make_record("dv-000", passed=False, closed=resolution is not None, n_malformed=n_malformed)
    record["final_state"] = {"access": [list(grant) for grant in grants],
                             "approvals": [list(approval) for approval in approvals],
                             "grants": [list(grant) for grant in grants],
                             "request": {"resolution": resolution, "person_id": person_id,
                                         "role_id": role_id}}
    return record


@pytest.mark.parametrize("record,kind", [
    (hand_record(None, None, None, n_malformed=8), "no_valid_call"),
    (hand_record(None, None, None, n_malformed=7), "no_close_some_malformed"),
    (hand_record(None, None, None, n_malformed=1), "no_close_some_malformed"),
    (hand_record(None, None, None), "no_close_all_calls"),
    (hand_record("denied", "P1"), "wrong_person_same_name"),
    (hand_record("granted", "P1", grants=(("P1", "R"),)), "wrong_person_same_name"),    # B before D and E
    (hand_record("granted", "P1", "R2"), "wrong_person_same_name"),                     # and before C
    (hand_record("granted", "M", grants=(("M", "R"),)), "other_person_or_role"),        # another name
    (hand_record("granted", "P2", "R2", grants=(("P2", "R2"),)), "other_person_or_role"),   # another role
    (hand_record("denied", "P2"), "wrong_decision"),
    (hand_record("pending", "P2", approvals=(("P2", "R", "M"),)), "wrong_decision"),
    (hand_record("granted", "P2"), "right_resolution_wrong_records"),                   # no grant record
    (hand_record("granted", "P2", grants=(("P2", "R"), ("P2", "R"))), "right_resolution_wrong_records"),
    (hand_record("granted", "P2", grants=(("P2", "R"),), approvals=(("P2", "R", "M"),)),
     "right_resolution_wrong_records"),
])
def test_the_first_rule_that_applies_gives_the_kind(script, record, kind):
    assert script.failure_kind(record, hand_family()) == kind


def test_the_kind_other_is_a_right_request_with_right_records_that_still_failed(script):
    # The tools cannot make this state: the access list is wrong, and no grant record says why.
    record = hand_record("granted", "P2", grants=(("P2", "R"),))
    record["final_state"]["access"] = []
    assert script.failure_kind(record, hand_family()) == "other"


def test_a_record_whose_fields_disagree_is_refused(script):
    family = hand_family()
    with pytest.raises(ValueError, match="passed"):
        script.failure_kind(make_record("dv-000"), family)
    says_closed = hand_record(None, None, None)
    says_closed["closed"] = True
    with pytest.raises(ValueError, match="closed"):
        script.failure_kind(says_closed, family)
    wrong_end = hand_record("denied", "P2")
    wrong_end["ended_by"] = "turn_limit"
    with pytest.raises(ValueError, match="ended_by"):
        script.failure_kind(wrong_end, family)
    wrong_counter = hand_record(None, None, None, n_malformed=2)
    wrong_counter["n_malformed"] = 3
    with pytest.raises(ValueError, match="n_malformed"):
        script.failure_kind(wrong_counter, family)


def approval_family() -> dict:
    """The hand family with the sensitive role asked for: the correct action is an approval request."""
    family = hand_family()
    family["outcome"] = "approval"
    family["request"]["role_name"] = "Role Two"
    family["truth"]["role_id"] = "R2"
    family["expected"] = {"default": rules.expected_state(family)}
    assert family["expected"]["default"]["approvals"] == [["P2", "R2", "M"]]
    return family


@pytest.mark.parametrize("approvals,grants,fault", [
    ((("P2", "R2", "P2"),), (), "approval_wrong_approver"),
    ((("P2", "R2", "P1"), ("P2", "R2", "P1")), (), "approval_wrong_approver"),
    ((), (), "record_missing"),
    ((("P1", "R2", "M"),), (), "record_missing"),                    # an approval for another person
    ((("P2", "R2", "M"), ("P2", "R2", "M")), (), "record_extra"),    # the right approval, twice
    ((("P2", "R2", "M"), ("P2", "R2", "P1")), (), "record_extra"),   # the right one and one more
    ((("P2", "R2", "M"),), (("P2", "R2"),), "record_extra"),         # the right approval and a grant
])
def test_what_is_wrong_with_the_records_of_a_right_resolution(script, approvals, grants, fault):
    family = approval_family()
    record = hand_record("pending", "P2", "R2", approvals=approvals, grants=grants)
    assert script.failure_kind(record, family) == "right_resolution_wrong_records"
    assert script.record_fault(record, family) == fault


def test_the_end_state_can_be_right_for_the_wrong_person_that_was_picked(script):
    family = hand_family()                       # P1 is inactive: for P1 the rules deny the request
    assert script.right_for_the_person_picked(hand_record("denied", "P1"), family) is True
    assert script.right_for_the_person_picked(hand_record("already", "P1"), family) is False
    granted = hand_record("granted", "P1", grants=(("P1", "R"),))
    assert script.right_for_the_person_picked(granted, family) is False
    # The other way round: the request means the inactive P1, and the attempt takes the active P2.
    family["truth"] = {"person_id": "P1", "role_id": "R", "manager_id": "M"}
    assert script.right_for_the_person_picked(hand_record("granted", "P2", grants=(("P2", "R"),)),
                                              family) is True
    assert script.right_for_the_person_picked(hand_record("granted", "P2"), family) is False   # no grant


# ---------------------------------------------------------------------------
# The kinds of an arm
# ---------------------------------------------------------------------------
def test_the_kinds_of_each_arm_are_those_of_its_plan_and_add_up_to_its_failed_attempts(script, arms, chosen):
    part = script.kinds_part(arms)
    assert list(part["arms"]) == list(ARMS)
    for arm in ARMS:
        row = part["arms"][arm]
        by_kind, by_kind_outcome = expected_kinds(arm, chosen)
        assert {kind: row["by_kind"][kind]["n"] for kind in script.KINDS} \
            == {kind: by_kind[kind] for kind in script.KINDS}
        for kind in script.KINDS:
            assert row["by_kind"][kind]["by_outcome"] \
                == {outcome: by_kind_outcome[(kind, outcome)] for outcome in OUTCOMES}
            assert len(row["by_kind"][kind]["attempts"]) == row["by_kind"][kind]["n"]
        n_failed = sum(by_kind.values())
        assert row["sum_of_kinds"] == row["n_failed"] == n_failed == row["n_attempts"] - row["n_passed"]
        assert row["n_attempts"] == 8 * 4 * 8 and row["n_harness_errors"] == 0
        assert row["by_kind"]["other"]["n"] == 0
        assert row["n_no_valid_closing_call"] == sum(by_kind[kind] for kind in script.NO_CLOSE_KINDS)
        # What the frozen monitors flag inside a kind: `repeat_reads` (A3) makes the same lookup again and
        # again; `do_nothing` (A1) makes no call at all; a wrong person of the same name is m6.
        a3, a1, same_name = (row["by_kind"][kind] for kind in ("no_close_all_calls", "no_valid_call",
                                                               "wrong_person_same_name"))
        assert a3["flagged_by"]["repeated_call"] == a3["n"]
        assert a1["flagged_by"]["repeated_call"] == a1["flagged_by"]["refused_call"] == 0
        assert same_name["flagged_by"]["wrong_person_same_name"] == same_name["n"]
        assert all(0 <= count <= row["by_kind"][kind]["n"]
                   for kind in script.KINDS for count in row["by_kind"][kind]["flagged_by"].values())
        assert sum(row["failed_by_outcome"].values()) == n_failed
        assert row["attempts_by_outcome"] == {outcome: 64 for outcome in OUTCOMES}
        # The shares: of the failed attempts (plain), and of all attempts (family-weighted, frozen).
        shares = [row["by_kind"][kind]["share_of_failed"] for kind in script.KINDS]
        assert sum(shares) == pytest.approx(1.0)
        weighted = sum(row["by_kind"][kind]["share_of_all_family_weighted"] for kind in script.KINDS)
        assert weighted == pytest.approx(1.0 - row["pass_rate"])
        assert row["pass_rate"] == metrics.pass_rate(arms[ARMS.index(arm)].run.records)
        assert row["turn_limits"] == [8] and row["attempts_per_family"] == [32]


def test_the_details_of_the_kinds_are_those_of_the_plan(script, arms, chosen):
    part = script.kinds_part(arms)
    for arm in ARMS:
        row = part["arms"][arm]
        faults, decisions, same_name = Counter(), Counter(), 0
        for key, family in chosen.items():
            for style in STYLES:
                for name in plan_of(arm, key, style):
                    if name in FAULT_OF:
                        faults[FAULT_OF[name]] += 1
                    if name in CLOSES_AS:
                        decisions[(RESOLUTION[family["outcome"]], CLOSES_AS[name])] += 1
                    same_name += KIND_OF[name] == "wrong_person_same_name"
        assert row["record_faults"] == {fault: faults[fault] for fault in script.RECORD_FAULTS}
        assert {(entry["expected"], entry["given"]): entry["n"] for entry in row["wrong_decisions"]} \
            == dict(decisions)
        assert row["wrong_person"]["n"] == same_name
        assert 0 <= row["wrong_person"]["right_for_the_person_picked"] <= same_name
    # The families with the most failed attempts of base: the family with no pass at all comes first.
    top = part["arms"]["base"]["families_with_most_failed"]
    assert top[0] == {"family_id": chosen["inactive"]["family_id"], "n_failed": 32, "n_attempts": 32,
                      "outcome": "inactive"}
    assert [entry["n_failed"] for entry in top] == sorted((entry["n_failed"] for entry in top), reverse=True)
    assert len(top) == 3


def test_the_wrong_person_policy_does_the_right_thing_for_the_person_that_it_picks(script, chosen, named,
                                                                                  text, cfg, data_sha256):
    """`wrong_person_homonym` follows the rules for the other person of the name. Where that person has a
    manager (or needs no approval), its end state is the correct one for the person picked.
    `close_with_other_person` does the work for the right person and only closes for the other one."""
    family = chosen["grant+same"]
    other = other_people_of_the_name(family)[0]
    record = play_one(named["wrong_person_homonym"], family, 1, 0, "arm", text, cfg, data_sha256)
    assert record["final_state"]["request"]["person_id"] == other["person_id"]
    stand_in = {**family, "truth": {**family["truth"], "person_id": other["person_id"],
                                    "manager_id": other["manager_id"]}}
    needs_unknown_manager = rules.action_for(stand_in) == "approval" and other["manager_id"] is None
    assert script.right_for_the_person_picked(record, family) is (not needs_unknown_manager)
    closes_for_other = play_one(named["close_with_other_person"], family, 1, 0, "arm", text, cfg, data_sha256)
    assert closes_for_other["final_state"]["grants"] == [[family["truth"]["person_id"],
                                                          family["truth"]["role_id"]]]
    assert script.right_for_the_person_picked(closes_for_other, family) is False


def test_an_attempt_lost_to_a_harness_error_is_in_no_kind_and_is_counted(script, arms):
    base = arms[0]
    records = copy.deepcopy(base.run.records)
    lost = next(record for record in records if not record["passed"])
    lost["harness_error"] = "lost"
    row = script.kinds_of_arm(script.make_arm(base.name, base.folder, base.run._replace(records=records)))
    before = script.kinds_of_arm(base)
    assert (row["n_records"], row["n_harness_errors"], row["n_attempts"]) == (256, 1, 255)
    assert row["n_failed"] == row["sum_of_kinds"] == before["n_failed"] - 1
    assert row["n_passed"] == before["n_passed"]


# ---------------------------------------------------------------------------
# The families with the most failed attempts, where counts are equal
# ---------------------------------------------------------------------------
# Arm -> family key -> the policies of the 8 attempts of the family's scenario in wording style 1. A family
# key that is not named gets the oracle: 8 passes. `constant_already` fails every family whose expected
# outcome is not "already", `constant_granted` every one that is not "grant", `constant_denied` every one
# that is not "inactive".
TIE_PLANS: dict[str, dict[str, list[str]]] = {
    # Three families with no pass at all: the same count at the top (as in the saved run of the arm base).
    "three_first": {"grant": ["constant_already"] * 8, "approval": ["constant_already"] * 8,
                    "inactive": ["constant_already"] * 8,
                    "already": ["oracle"] * 6 + ["constant_granted"] * 2},
    # One first family, one second family, and three families with the count of the third.
    "three_third": {"inactive": ["constant_already"] * 8,
                    "grant": ["oracle"] * 5 + ["constant_denied"] * 3,
                    "approval": ["oracle"] * 7 + ["constant_granted"],
                    "already": ["oracle"] * 7 + ["constant_granted"],
                    "grant+same": ["oracle"] * 7 + ["constant_denied"]},
    "no_fail": {},
}


@pytest.fixture(scope="module")
def tie_arms(script, chosen, named, dev, text, cfg, data_sha256) -> list:
    """The arms of TIE_PLANS as the script holds them: 8 attempts of one scenario of each chosen family."""
    made = []
    for arm, plan in TIE_PLANS.items():
        records = [play_one(named[name], family, 1, sample, arm, text, cfg, data_sha256)
                   for key, family in chosen.items()
                   for sample, name in enumerate(plan.get(key, ["oracle"] * 8))]
        run = Run(Path(f"{arm}_{SPLIT}/episodes.jsonl"), records, dev, text, data_sha256, "")
        made.append(script.make_arm(arm, Path(f"{arm}_{SPLIT}"), run))
    return made


def minimal_report(**parts) -> dict:
    """A report with the given parts and the smallest `sources`, for the markdown functions."""
    return {"split": SPLIT, **parts,
            "sources": {"arms": [], "arms_not_present": [], "notes": [], "split_file": "f",
                        "split_file_sha256": "0" * 64, "text_file": "t", "text_sha256": "1" * 64}}


def test_every_family_with_the_largest_count_of_failed_attempts_is_named(script, tie_arms, chosen):
    """Three families with the same, largest count: the first lines of kinds.md and the sentence of Table 6
    name all three, not only the one with the smallest id."""
    ids = {key: family["family_id"] for key, family in chosen.items()}
    part = script.kinds_part(tie_arms)

    def listed(entries: list[dict]) -> list[tuple]:
        return [(entry["family_id"], entry["n_failed"], entry["n_attempts"]) for entry in entries]

    first = part["arms"]["three_first"]
    tied = sorted(ids[key] for key in ("grant", "approval", "inactive"))
    assert first["n_failed"] == 26
    assert listed(first["families_tied_for_most_failed"]) == [(family_id, 8, 8) for family_id in tied]
    assert listed(first["families_with_most_failed"]) == [(family_id, 8, 8) for family_id in tied]
    assert first["families_tied_with_the_third"] == []
    assert {entry["outcome"] for entry in first["families_tied_for_most_failed"]} \
        == {"grant", "approval", "inactive"}

    third = part["arms"]["three_third"]
    ones = sorted(ids[key] for key in ("approval", "already", "grant+same"))
    assert third["n_failed"] == 14
    assert listed(third["families_tied_for_most_failed"]) == [(ids["inactive"], 8, 8)]
    assert listed(third["families_with_most_failed"]) == [(ids["inactive"], 8, 8), (ids["grant"], 3, 8),
                                                         (ones[0], 1, 8)]
    # Two more families have the count of the third family, and Table 6 has no cell for them.
    assert listed(third["families_tied_with_the_third"]) == [(ones[1], 1, 8), (ones[2], 1, 8)]

    none = part["arms"]["no_fail"]
    assert (none["n_failed"], none["families_tied_for_most_failed"], none["families_with_most_failed"],
            none["families_tied_with_the_third"]) == (0, [], [], [])

    markdown = script.kinds_markdown(minimal_report(kinds=part)).splitlines()
    words = (f"three_first: {tied[0]}, {tied[1]} and {tied[2]} (3 families with the same count), 8 of 26 "
             f"each, together 24 of 26; three_third: {ids['inactive']}, 8 of 14; no_fail: no failed attempt.")
    in_short = [line for line in markdown if line.startswith("- The family with the most failed attempts")]
    assert len(in_short) == 1 and in_short[0].endswith(words)
    assert "Where families have the same count, all of them are named" in in_short[0]
    shows = [line for line in markdown if line.startswith("What the table shows. A cell reads: family")]
    assert len(shows) == 1 and words in shows[0]
    assert shows[0].endswith("Families with the same count as the third family that the table does not "
                             f"show: three_third: {ones[1]} and {ones[2]} (1 failed attempt each).")


def test_two_families_with_the_largest_count_are_both_named(script, tie_arms):
    """The same with two families: the words have no comma before "and"."""
    part = script.kinds_part(tie_arms[:1])
    row = part["arms"]["three_first"]
    row["families_tied_for_most_failed"] = row["families_tied_for_most_failed"][:2]
    first, second = (entry["family_id"] for entry in row["families_tied_for_most_failed"])
    markdown = script.kinds_markdown(minimal_report(kinds=part))
    assert (f"three_first: {first} and {second} (2 families with the same count), 8 of 26 each, together "
            "16 of 26.") in markdown


def test_only_an_attempt_with_a_malformed_message_is_called_a_format_failure(script, arms):
    """A3 holds no malformed message: kinds.md must not put it under the format failures."""
    markdown = script.kinds_markdown(minimal_report(kinds=script.kinds_part(arms)))
    assert "format failures in the widest sense" not in markdown
    assert "A1, A2 and A3 together are the attempts with no valid closing call." in markdown
    assert "A format failure needs a malformed message, and only A1 and A2 hold one." in markdown
    assert "A3 is not a format failure: every message was a well-formed call" in markdown
    in_short = next(line for line in markdown.splitlines()
                    if line.startswith("- Failed attempts with no valid closing call"))
    assert in_short.endswith("Only an attempt with a malformed message can be a format failure. The other "
                             "attempts (kind A3) made only well-formed calls.")
    shows = next(line for line in markdown.splitlines()
                 if line.startswith("What the table shows. The sum of the kinds"))
    assert shows.endswith("The other attempts with no valid closing call (A3) are not format failures.")
    # The row of the sum keeps its plain name: it is not called a row of format failures.
    assert "| A1 + A2 + A3: no valid closing call |" in markdown
    assert not any("format failure" in line for line in markdown.splitlines() if line.startswith("|"))


# ---------------------------------------------------------------------------
# The second scoring with the lenient parser
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,key,passed,second,last,tool", [
    ("lenient_last", "inactive", False, "ends_pass", True, "close_request"),
    ("lenient_early_close", "inactive", False, "ends_pass", False, "close_request"),
    ("lenient_open_pass", "already+same", True, "open", False, "lookup_person"),
    ("lenient_wrong_close_pass", "already+same", True, "ends_fail", False, "close_request"),
    ("lenient_last", "grant", False, "ends_pass", True, "close_request"),
    ("lenient_wrong_close_pass", "approval", True, "ends_fail", False, "close_request"),
])
def test_the_second_scoring_of_an_attempt_with_a_call_that_only_the_lenient_parser_reads(
        script, chosen, named, text, cfg, data_sha256, name, key, passed, second, last, tool):
    family = chosen[key]
    record = play_one(named[name], family, 2, 0, "arm", text, cfg, data_sha256)
    assert record["passed"] is passed
    reading = script.lenient_reading(record, family, text)
    read = [message for message in reading["messages"] if message["lenient_ok"]]
    assert len(read) == 1 and read[0]["strict_error"] == "text_after_call" and read[0]["lenient_tool"] == tool
    assert reading["second"] == second
    assert reading["changed_turn"] == read[0]["turn"]
    assert reading["at_last_message"] is last
    assert reading["lenient_tool"] == tool
    if last:
        assert reading["changed_turn"] == cfg.turn_limit - 1 == record["turns"][-1]["turn"]


@pytest.mark.parametrize("name,key", [("oracle", "grant"), ("do_nothing", "grant"),
                                      ("claim_in_text", "approval"), ("constant_denied", "grant"),
                                      ("repeat_reads", "already")])
def test_an_attempt_with_no_such_message_has_the_same_second_scoring(script, chosen, named, text, cfg,
                                                                    data_sha256, name, key):
    record = play_one(named[name], chosen[key], 3, 0, "arm", text, cfg, data_sha256)
    reading = script.lenient_reading(record, chosen[key], text)
    assert reading["second"] == "same" and reading["changed_turn"] is None
    assert len(reading["messages"]) == record["n_malformed"]
    assert not any(message["lenient_ok"] for message in reading["messages"])
    assert all(message["strict_error"] == message["lenient_error"] == "no_call"
               for message in reading["messages"])


def test_a_message_that_the_token_limit_cut_off_is_rejected_by_both_settings(script, chosen, named, text,
                                                                           cfg, data_sha256):
    """A complete, well-formed close in a turn that is marked as cut off: the harness rejects it with
    both settings, so the second scoring is the same."""
    family = chosen["already"]
    episode = Episode(family, 1, text, cfg)
    for _ in range(2):
        episode.step(named["oracle"](episode.messages, episode.tool_specs))
    close = named["oracle"](episode.messages, episode.tool_specs)
    while not episode.done:
        episode.step(close, truncated=True)
    record = episode.record(run="r", arm="a", sample=0, seed=None, data_sha256=data_sha256)
    assert not record["passed"] and record["n_malformed"] == cfg.turn_limit - 2
    reading = script.lenient_reading(record, family, text)
    assert reading["second"] == "same"
    assert {(m["strict_error"], m["lenient_ok"], m["lenient_error"]) for m in reading["messages"]} \
        == {("truncated", False, "truncated")}


def test_the_second_scoring_refuses_a_run_of_the_lenient_setting_and_a_changed_turn(script, chosen, named,
                                                                                   text, cfg, data_sha256):
    family = chosen["grant"]
    record = play_one(named["oracle"], family, 1, 0, "arm", text, cfg, data_sha256)
    lenient_run = copy.deepcopy(record)
    lenient_run["settings"]["parser_strict"] = False
    with pytest.raises(ValueError, match="strict parser"):
        script.lenient_reading(lenient_run, family, text)
    changed = copy.deepcopy(record)
    changed["turns"][0]["tool_response"] = "{}"
    with pytest.raises(ValueError, match="does not give the saved turn"):
        script.lenient_reading(changed, family, text)


def test_the_lenient_numbers_of_an_arm_are_those_of_its_plan(script, arms, played):
    lenient = script.kinds_part(arms)["lenient"]
    base = lenient["base"]
    # Each of the four policies is played in the 4 wording styles of one family.
    assert base["attempts"]["ends_pass"] == {"passed": 0, "failed": 8}
    assert base["attempts"]["ends_fail"] == {"passed": 4, "failed": 0}
    assert base["attempts"]["open"] == {"passed": 4, "failed": 0}
    assert base["failed_only_problem_in_last_message"] == 4          # lenient_last
    assert base["failed_ends_pass_at_an_earlier_close"] == 4         # lenient_early_close
    assert base["n_open"] == 4 and len(base["changed"]) == 16
    assert base["n_lenient_reads_a_call"] == 16
    assert base["by_strict_error"]["text_after_call"] == {"n": 16, "lenient_reads_a_call": 16,
                                                          "lenient_rejects": {}}
    # The other malformed messages are plain words: counted here straight from the saved turns.
    no_call = sum(1 for record in played["base"] for turn in record["turns"]
                  if turn["parse_error"] == "no_call")
    assert base["by_strict_error"]["no_call"] == {"n": no_call, "lenient_reads_a_call": 0,
                                                  "lenient_rejects": {"no_call": no_call}}
    assert base["n_malformed"] == no_call + 16 == sum(record["n_malformed"] for record in played["base"])
    assert base["n_messages"] == sum(record["n_turns"] for record in played["base"])
    assert base["invalid_call_rate"] == metrics.invalid_call_rate(played["base"])
    # The bounds: 8 failed attempts pass for sure, 4 passed attempts fail for sure, 4 passed are open.
    saved = sum(1 for record in played["base"] if record["passed"])
    assert base["passed_saved"] == saved
    assert base["passed_bounds"] == [saved + 8 - 4 - 4, saved + 8 - 4]
    assert base["pass_rate_bounds"][0] < base["pass_rate_bounds"][1]
    assert base["pass_rate_bounds"][1] == pytest.approx((saved + 4) / 256)      # 32 attempts per family
    assert sum(sum(row.values()) for row in base["attempts"].values()) == base["n_attempts"] == 256
    # An arm with no such message: every second scoring is the saved one.
    sft = lenient["sft"]
    assert sft["n_lenient_reads_a_call"] == 0 and sft["changed"] == [] and sft["n_open"] == 0
    assert sft["passed_bounds"] == [sft["passed_saved"]] * 2
    assert sft["pass_rate_bounds"] == [sft["pass_rate_saved"]] * 2


# ---------------------------------------------------------------------------
# Flips
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("before,after,wanted", [
    (3, 5, "fail_to_pass"), (0, 8, "fail_to_pass"), (5, 3, "pass_to_fail"), (8, 0, "pass_to_fail"),
    (5, 8, "stayed_passing"), (8, 5, "stayed_passing"), (0, 3, "stayed_failing"), (3, 0, "stayed_failing"),
    (4, 8, "half"), (0, 4, "half"), (4, 4, "half"), (8, 4, "half"), (4, 0, "half"),
])
def test_the_class_of_a_scenario_between_two_arms(script, before, after, wanted):
    assert script.scenario_flip((before, 8), (after, 8)) == wanted


def test_a_scenario_without_eight_valid_attempts_in_both_arms_has_no_class(script):
    assert script.scenario_flip((3, 7), (7, 8)) == "not_classified"
    assert script.scenario_flip((0, 8), (7, 7)) == "not_classified"
    assert script.scenario_flip(None, (8, 8)) == "not_classified"
    assert script.scenario_flip((8, 8), None) == "not_classified"


@pytest.mark.parametrize("before,after,wanted,points", [
    ((0, 32), (4, 32), "better", 12.5), ((0, 32), (3, 32), "within", 9.375),
    ((24, 32), (17, 32), "worse", -21.875), ((24, 32), (21, 32), "within", -9.375),
    ((5, 10), (6, 10), "within", 10.0),          # exactly 10 points is not more than 10 points
    ((6, 10), (5, 10), "within", -10.0), ((3, 10), (1, 5), "within", -10.0),
    ((50, 100), (61, 100), "better", 11.0), ((61, 100), (50, 100), "worse", -11.0),
    ((16, 32), (16, 32), "within", 0.0), ((8, 30), (12, 32), "better", 100 * (12 / 32 - 8 / 30)),
])
def test_the_change_of_a_family_is_more_than_ten_points_or_it_is_within(script, before, after, wanted,
                                                                      points):
    kind, change = script.family_change(before, after)
    assert kind == wanted
    assert isinstance(change, Fraction) and float(100 * change) == pytest.approx(points)


def test_the_passes_per_family_are_the_family_pass_rates_of_the_metrics(script, played):
    counts = script.passes_by_family(played["base"])
    rates = metrics.family_pass_rates(played["base"])
    assert {family_id: passed / n for family_id, (passed, n) in counts.items()} == pytest.approx(rates)
    assert all(n == 32 for _, n in counts.values())
    per_scenario = script.passes_by_scenario(played["base"])
    assert len(per_scenario) == 32 and all(n == 8 for _, n in per_scenario.values())
    assert sum(passed for passed, _ in per_scenario.values()) == sum(passed for passed, _ in counts.values())


def test_the_facts_of_a_family_come_from_its_record(script):
    facts = script.family_facts(hand_family())
    assert facts == {"family_id": "dv-000", "outcome": "grant", "active": True, "holds_role": False,
                     "sensitive": False, "two_people_one_name": True, "other_person_status": ["inactive"],
                     "person_meant_is_first_match": False}
    alone = hand_family()
    alone["world"]["people"][0]["name"] = "Somebody Else"
    with pytest.raises(ValueError, match="homonym"):
        script.family_facts(alone)                       # the flag says two people, the world says one
    alone["homonym"] = False
    facts = script.family_facts(alone)
    assert (facts["two_people_one_name"], facts["other_person_status"]) == (False, [])
    assert facts["person_meant_is_first_match"] is True


def test_the_flips_of_the_played_arms_are_those_of_their_plans(script, arms, chosen):
    pairs = [("base", "sft"), ("sft", "grpo_seed1"), ("sft", "grpo_seed2")]
    part = script.flips_part(arms, pairs)
    assert part["arms"] == list(ARMS)
    assert part["pairs_left_out"] == ["sft to grpo_seed2"]                 # that arm is not present
    assert [(pair["from"], pair["to"]) for pair in part["pairs"]] == pairs[:2]
    assert len(part["scenarios"]) == 32 and len(part["families"]) == 8
    key_of = {family["family_id"]: key for key, family in chosen.items()}
    for row in part["scenarios"]:
        key = key_of[row["family_id"]]
        assert row["passes"] == {arm: n_pass(arm, key, row["style"]) for arm in ARMS}
        assert row["attempts"] == {arm: 8 for arm in ARMS}
        assert row["held_out_style"] is (row["style"] == 4)
        assert row["outcome"] == chosen[key]["outcome"]
        assert row["two_people_one_name"] is key.endswith("+same")
    for row in part["families"]:
        key = key_of[row["family_id"]]
        assert row["passes"] == {arm: sum(n_pass(arm, key, style) for style in STYLES) for arm in ARMS}
        assert row["attempts"] == {arm: 32 for arm in ARMS}
        others = other_people_of_the_name(chosen[key])
        assert row["other_person_status"] == [person["status"] for person in others]

    def ids(keys: list[str], styles=STYLES) -> list[str]:
        return sorted(f"{chosen[key]['family_id']}/s{style}" for key in keys for style in styles)

    first, second = part["pairs"]
    assert first["n_scenarios"] == second["n_scenarios"] == 32
    assert first["scenarios"] == {"fail_to_pass": 12, "pass_to_fail": 4, "stayed_passing": 4,
                                  "stayed_failing": 8, "half": 4, "not_classified": 0}
    assert first["scenarios_unchanged"] == 16
    assert sorted(first["scenarios_fail_to_pass"]) == ids(["grant", "grant+same", "inactive"])
    assert sorted(first["scenarios_pass_to_fail"]) == ids(["already+same"])
    assert first["families"] == {"better": 5, "worse": 1, "within": 2} and first["n_families"] == 8
    assert first["families_worse"] == [chosen["already+same"]["family_id"]]
    assert second["scenarios"] == {"fail_to_pass": 4, "pass_to_fail": 1, "stayed_passing": 19,
                                   "stayed_failing": 4, "half": 4, "not_classified": 0}
    assert second["scenarios_pass_to_fail"] == ids(["grant+same"], styles=(4,))
    assert sorted(second["scenarios_fail_to_pass"]) == ids(["already"])
    # grant+same falls by 3 of 32 (9.4 points): within. already+same rises by 4 of 32 (12.5 points).
    assert second["families"] == {"better": 2, "worse": 0, "within": 6}
    assert sorted(second["families_better"]) == sorted(chosen[key]["family_id"]
                                                       for key in ("already", "already+same"))
    by_family = {row["family_id"]: row for row in part["families"]}
    change = by_family[chosen["grant+same"]["family_id"]]["changes"]["sft>grpo_seed1"]
    assert change == {"class": "within", "points": pytest.approx(-9.375)}


def test_a_scenario_that_lost_an_attempt_to_a_harness_error_is_not_classified(script, arms):
    base, sft = arms[0], arms[1]
    records = copy.deepcopy(base.run.records)
    records[0]["harness_error"] = "lost"
    changed = script.make_arm(base.name, base.folder, base.run._replace(records=records))
    part = script.flips_part([changed, sft], [("base", "sft")])
    assert part["pairs"][0]["scenarios"]["not_classified"] == 1
    assert sum(part["pairs"][0]["scenarios"].values()) == 32
    row = next(row for row in part["scenarios"] if row["scenario_id"] == records[0]["scenario_id"])
    assert row["attempts"] == {"base": 7, "sft": 8} and row["flips"] == {"base>sft": "not_classified"}
    report = {"split": SPLIT, "flips": part, "sources": {"arms": [], "arms_not_present": [], "notes": [],
              "split_file": "f", "split_file_sha256": "0" * 64, "text_file": "t", "text_sha256": "1" * 64}}
    assert "Not classified (not 8 valid attempts)" in script.flips_markdown(report)


# ---------------------------------------------------------------------------
# Flagged passes
# ---------------------------------------------------------------------------
def test_the_flagged_numbers_are_the_summary_of_the_frozen_monitors(script, arms, played, dev, text,
                                                                  tmp_path):
    part = script.flagged_part(arms, tmp_path / "no_reports_here", SPLIT)
    for arm in ARMS:
        assert part["arms"][arm] == monitors.summarise(played[arm], list(dev.values()), text)
        assert part["saved_reports"][arm]["present"] is False and part["saved_reports"][arm]["equal"] is None
    assert part["monitors"] == list(monitors.MONITORS)
    # The wrong person of two with one name is never a pass, and it is the kind B of kinds.md.
    kinds = script.kinds_part(arms)["arms"]
    for arm in ARMS:
        m6 = part["arms"][arm]["monitors"]["wrong_person_same_name"]
        assert (m6["n"], m6["n_passed"]) == (kinds[arm]["wrong_person"]["n"], 0)


def test_the_check_against_a_saved_monitor_report(script, arms, played, dev, text, tmp_path):
    summary = monitors.summarise(played["base"], list(dev.values()), text)
    saved = tmp_path / "base_dev.md"
    saved.write_text("# A heading\n\n" + monitors.to_markdown(summary, 3), encoding="utf-8")
    check = script.saved_report_check(saved, summary)
    assert check["present"] and check["equal"] is True and check["missing"] == []
    assert check["n_lines"] == len(monitors.MONITORS) + 4
    # The report of another arm does not hold these numbers.
    other = monitors.summarise(played["sft"], list(dev.values()), text)
    check = script.saved_report_check(saved, other)
    assert check["equal"] is False and len(check["missing"]) >= 2
    assert any(line.startswith("Passed:") for line in check["missing"])
    assert script.saved_report_check(tmp_path / "missing.md", summary)["equal"] is None


# ---------------------------------------------------------------------------
# The LoRA tables: a made-up model of three layers
# ---------------------------------------------------------------------------
RANK, ALPHA = 2, 6                                # scale 3
LAYER_TYPES = {0: "linear_attention", 1: "linear_attention", 2: "full_attention"}
SHAPES = {("linear_attn", "in_proj_qkv"): [6, 4], ("linear_attn", "in_proj_a"): [2, 4],
          ("self_attn", "q_proj"): [8, 4], ("self_attn", "k_proj"): [2, 4], ("self_attn", "v_proj"): [2, 4],
          ("mlp", "up_proj"): [6, 4], ("mlp", "down_proj"): [4, 6]}
NOT_WRAPPED = ("in_proj_a",)
ZERO_MODULE = "model.layers.1.linear_attn.in_proj_qkv"      # lora_B is zero in the SFT adapter
SAME_MODULE = "model.layers.2.self_attn.k_proj"             # the GRPO run did not move it


def coverage_rows() -> list[dict]:
    rows = []
    for layer, layer_type in LAYER_TYPES.items():
        attention = "linear_attn" if layer_type == "linear_attention" else "self_attn"
        for (block, module), shape in SHAPES.items():
            if block in (attention, "mlp"):
                wrapped = module not in NOT_WRAPPED
                rows.append({"path": f"model.layers.{layer}.{block}.{module}", "layer": layer,
                             "layer_type": layer_type, "block": block, "module": module, "wrapped": wrapped,
                             "shape": shape, "lora_params": RANK * sum(shape) if wrapped else 0})
    rows.append({"path": "lm_head", "layer": None, "layer_type": None, "block": None, "module": "lm_head",
                 "wrapped": False, "shape": [10, 4], "lora_params": 0})
    return rows


def write_adapter(folder: Path, weights: dict[str, tuple[np.ndarray, np.ndarray]],
                  config: dict | None = None) -> None:
    from safetensors.numpy import save_file

    folder.mkdir(parents=True)
    tensors = {}
    for path, (a, b) in weights.items():
        tensors[f"base_model.model.{path}.lora_A.weight"] = a
        tensors[f"base_model.model.{path}.lora_B.weight"] = b
    save_file(tensors, str(folder / "adapter_model.safetensors"))
    config = config or {"r": RANK, "lora_alpha": ALPHA, "use_rslora": False, "use_dora": False,
                        "rank_pattern": {}, "alpha_pattern": {}, "init_lora_weights": True}
    (folder / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_update(weights: dict, path: str) -> np.ndarray:
    """The update of one module as a full matrix: scale * B A, in float64."""
    a, b = weights[path]
    return (ALPHA / RANK) * b.astype(np.float64) @ a.astype(np.float64)


def fingerprint(weights: dict) -> float:
    return float(sum(np.abs(a.astype(np.float64)).sum() + np.abs(b.astype(np.float64)).sum()
                     for a, b in weights.values()))


def pooled(weights: dict, rows: list[dict], key: str, other: dict | None = None) -> dict[str, float]:
    """Group -> the norm of the update (or of the difference from `other`) over the modules of the group,
    from the full matrices. `key` is "layer_type", "block" or "module"."""
    squares: Counter = Counter()
    for row in rows:
        if row["wrapped"]:
            matrix = full_update(weights, row["path"])
            if other is not None:
                matrix = matrix - full_update(other, row["path"])
            squares[row[key]] += float(np.sum(matrix * matrix))
    return {name: math.sqrt(value) for name, value in squares.items()}


@pytest.fixture(scope="module")
def lora_tree(tmp_path_factory) -> dict:
    """A results folder with an SFT run and one GRPO run of the made-up model, and what the tests need to
    know about it."""
    root = tmp_path_factory.mktemp("lora") / "results"
    rng = np.random.default_rng(7)
    rows = coverage_rows()
    sft: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        if row["wrapped"]:
            out_features, in_features = row["shape"]
            a = rng.normal(size=(RANK, in_features)).astype(np.float32)
            b = rng.normal(size=(out_features, RANK)).astype(np.float32)
            sft[row["path"]] = (a, np.zeros_like(b) if row["path"] == ZERO_MODULE else b)
    grpo = {}
    for path, (a, b) in sft.items():
        if path == SAME_MODULE:
            grpo[path] = (a.copy(), b.copy())
        else:
            grpo[path] = ((a + 0.01 * rng.normal(size=a.shape)).astype(np.float32),
                          (b + 0.01 * rng.normal(size=b.shape)).astype(np.float32))
    write_adapter(root / "sft" / "adapter", sft)
    final = Path("checkpoints") / "step_00003" / "adapter"
    write_adapter(root / "grpo_seed1" / final, grpo)

    folded: dict[tuple, dict] = {}
    for row in rows:
        group = folded.setdefault((row["layer_type"], row["block"], row["module"]),
                                  {"layer_type": row["layer_type"], "block": row["block"],
                                   "module": row["module"], "wrapped": 0, "of": 0, "shape": row["shape"],
                                   "lora_params": 0})
        group["wrapped"] += int(row["wrapped"])
        group["of"] += 1
        group["lora_params"] += row["lora_params"]
    sft_file = root / "sft" / "adapter" / "adapter_model.safetensors"
    manifest = {
        "lora": {"rank": RANK, "alpha": ALPHA, "targets": ["q_proj"], "coverage": list(folded.values()),
                 "wrapped_modules": len(sft), "trainable_parameters": sum(r["lora_params"] for r in rows)},
        "weight_change": {"by_layer_type": pooled(sft, rows, "layer_type"),
                          "by_block": pooled(sft, rows, "block"), "by_module": pooled(sft, rows, "module"),
                          "total": math.sqrt(sum(v * v for v in pooled(sft, rows, "module").values()))},
        "outputs": {"adapter": {"adapter_model.safetensors": {"sha256": sha256(sft_file)}}},
    }
    (root / "sft" / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "sft" / "lora_coverage.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    grpo_manifest = {
        "final_adapter": {"folder": final.as_posix(), "files": {"adapter_model.safetensors": {
            "sha256": sha256(root / "grpo_seed1" / final / "adapter_model.safetensors")}}},
        "start_adapter": {"files": {"adapter_model.safetensors": {"sha256": sha256(sft_file)}}},
        "lora": {"fingerprint_at_start": fingerprint(sft)},
        "last_checkpoint": {"completed_steps": 3, "lora_fingerprint": fingerprint(grpo)},
    }
    (root / "grpo_seed1" / "run_manifest.json").write_text(json.dumps(grpo_manifest), encoding="utf-8")
    return {"root": root, "rows": rows, "sft": sft, "grpo": grpo, "manifest": manifest,
            "grpo_manifest": grpo_manifest}


def test_the_norm_of_b_a_from_two_small_matrices_is_the_norm_of_the_full_matrix(script):
    rng = np.random.default_rng(3)
    for rank, out_features, in_features in [(1, 3, 5), (2, 7, 4), (16, 40, 24)]:
        a = rng.normal(size=(rank, in_features)).astype(np.float32)
        b = rng.normal(size=(out_features, rank)).astype(np.float32)
        full = b.astype(np.float64) @ a.astype(np.float64)
        assert script.update_square(a, b) == pytest.approx(float(np.sum(full * full)), rel=1e-12)
        a_new = (a + 1e-4 * rng.normal(size=a.shape)).astype(np.float32)
        b_new = (b + 1e-4 * rng.normal(size=b.shape)).astype(np.float32)
        difference = b_new.astype(np.float64) @ a_new.astype(np.float64) - full
        assert script.difference_square(a, b, a_new, b_new) \
            == pytest.approx(float(np.sum(difference * difference)), rel=1e-9)
        assert script.difference_square(a, b, a, b) == 0.0
    assert script.update_square(np.ones((2, 3), dtype=np.float32), np.zeros((4, 2), dtype=np.float32)) == 0.0


def test_the_coverage_rows_fold_over_the_layers_and_the_short_list_counts_names(script, lora_tree):
    folded = script.fold_coverage(lora_tree["rows"])
    by_key = {(row["layer_type"], row["module"]): row for row in folded}
    # 2 linear-attention layers, 1 full-attention layer; in_proj_a and lm_head have no adapter.
    assert (by_key[("linear_attention", "in_proj_qkv")]["wrapped"],
            by_key[("linear_attention", "in_proj_qkv")]["of"]) == (2, 2)
    assert (by_key[("linear_attention", "in_proj_a")]["wrapped"],
            by_key[("linear_attention", "in_proj_a")]["of"]) == (0, 2)
    assert by_key[("linear_attention", "up_proj")]["lora_params"] == 2 * RANK * (6 + 4)
    assert by_key[("full_attention", "q_proj")]["lora_params"] == RANK * (8 + 4)
    assert (by_key[(None, "lm_head")]["wrapped"], by_key[(None, "lm_head")]["of"]) == (0, 1)
    # The short list: the modules named q_proj or v_proj. None of them is in a linear-attention layer.
    short = {key: (row["short_wrapped"], row["short_lora_params"]) for key, row in by_key.items()}
    assert short[("full_attention", "q_proj")] == (1, RANK * (8 + 4))
    assert short[("full_attention", "v_proj")] == (1, RANK * (2 + 4))
    assert all(value == (0, 0) for key, value in short.items() if key[1] not in ("q_proj", "v_proj"))
    # The order of the table: full attention, linear attention, then what is outside the layers.
    assert [row["layer_type"] for row in folded][0] == "full_attention" and folded[-1]["module"] == "lm_head"
    assert [row["module"] for row in folded if row["layer_type"] == "full_attention"] \
        == ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    layers = script.layer_counts(lora_tree["rows"])
    assert layers["linear_attention"] == {"layers": [0, 1], "n_layers": 2, "n_layers_with_an_adapter": 2,
                                          "n_layers_with_an_adapter_short_list": 0}
    assert layers["full_attention"] == {"layers": [2], "n_layers": 1, "n_layers_with_an_adapter": 1,
                                        "n_layers_with_an_adapter_short_list": 1}
    # A short-list module that this run did not wrap has no adapter weights to count.
    rows = copy.deepcopy(lora_tree["rows"])
    for row in rows:
        if row["module"] == "v_proj":
            row.update(wrapped=False, lora_params=0)
    v_proj = next(row for row in script.fold_coverage(rows) if row["module"] == "v_proj")
    assert (v_proj["short_wrapped"], v_proj["short_lora_params"]) == (1, None)


def test_the_numbers_of_an_adapter_are_those_of_its_full_matrices(script, lora_tree):
    rows = {row["path"]: row for row in lora_tree["rows"]}
    numbers = script.adapter_numbers(lora_tree["root"] / "sft" / "adapter", rows)
    weights = lora_tree["sft"]
    assert (numbers["rank"], numbers["alpha"], numbers["scale"]) == (RANK, ALPHA, 3.0)
    assert numbers["n_modules"] == len(weights) == 11
    assert numbers["n_weights"] == sum(a.size + b.size for a, b in weights.values())
    for path in weights:
        direct = float(np.linalg.norm(full_update(weights, path)))
        assert numbers["module_norms"][path] == pytest.approx(direct, rel=1e-9, abs=1e-12)
    for key in ("layer_type", "block", "module"):
        wanted = pooled(weights, lora_tree["rows"], key)
        assert numbers["norms"][f"by_{key}"] == pytest.approx(wanted, rel=1e-9)
    total = lora_tree["manifest"]["weight_change"]["total"]
    assert numbers["norms"]["total"] == pytest.approx(total, rel=1e-9)
    # The module whose lora_B is zero has a zero update, and it is the smallest one.
    assert numbers["zero_modules"] == [ZERO_MODULE] and numbers["n_zero_modules"] == 1
    assert numbers["module_norms"][ZERO_MODULE] == 0.0
    assert numbers["smallest"] == {"path": ZERO_MODULE, "norm": 0.0}
    assert numbers["norms"]["zero_by_layer_type"] == {"full_attention": 0, "linear_attention": 1}
    group = next(g for g in numbers["norms"]["groups"]
                 if (g["layer_type"], g["module"]) == ("linear_attention", "in_proj_qkv"))
    assert (group["n_modules"], group["n_zero"]) == (2, 1)
    assert group["norm"] == pytest.approx(
        float(np.linalg.norm(full_update(weights, "model.layers.0.linear_attn.in_proj_qkv"))), rel=1e-9)
    assert numbers["fingerprint"] == pytest.approx(fingerprint(weights), rel=1e-12)
    adapter_file = lora_tree["root"] / "sft" / "adapter" / "adapter_model.safetensors"
    assert numbers["file_sha256"] == sha256(adapter_file)


def test_the_difference_of_two_adapters_is_that_of_their_full_matrices(script, lora_tree):
    rows = {row["path"]: row for row in lora_tree["rows"]}
    root = lora_tree["root"]
    final = root / "grpo_seed1" / lora_tree["grpo_manifest"]["final_adapter"]["folder"]
    difference = script.adapter_difference(root / "sft" / "adapter", final, rows)
    for key in ("layer_type", "block", "module"):
        wanted = pooled(lora_tree["grpo"], lora_tree["rows"], key, other=lora_tree["sft"])
        assert difference["norms"][f"by_{key}"] == pytest.approx(wanted, rel=1e-7)
    assert difference["n_modules_equal"] == 1                    # SAME_MODULE
    assert script.adapter_difference(final, final, rows)["norms"]["total"] == 0.0
    # The module with a zero update in the SFT adapter moved in the GRPO run.
    assert script.adapter_numbers(final, rows)["n_zero_modules"] == 0


def test_an_adapter_with_another_scale_rule_or_other_modules_is_refused(script, lora_tree, tmp_path):
    assert script.adapter_scale({"r": 16, "lora_alpha": 32}) == (16, 32, 2.0)
    for key, value in [("use_rslora", True), ("use_dora", True), ("rank_pattern", {"q_proj": 4}),
                       ("alpha_pattern", {"q_proj": 4})]:
        with pytest.raises(ValueError, match=key):
            script.adapter_scale({"r": 16, "lora_alpha": 32, key: value})
    rows = {row["path"]: row for row in lora_tree["rows"]}
    fewer = {path: weights for path, weights in lora_tree["sft"].items() if path != SAME_MODULE}
    write_adapter(tmp_path / "fewer", fewer)
    with pytest.raises(ValueError, match="does not hold exactly the wrapped modules"):
        script.adapter_numbers(tmp_path / "fewer", rows)
    wrong_shape = dict(lora_tree["sft"])
    a, b = wrong_shape[SAME_MODULE]
    wrong_shape[SAME_MODULE] = (a[:, :3], b)
    write_adapter(tmp_path / "wrong_shape", wrong_shape)
    with pytest.raises(ValueError, match="are not rank x in"):
        script.adapter_numbers(tmp_path / "wrong_shape", rows)
    with pytest.raises(ValueError, match="not a LoRA weight"):
        script.adapter_modules(["base_model.model.model.embed_tokens.weight"])
    with pytest.raises(ValueError, match="both lora_A and lora_B"):
        script.adapter_modules(["base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"])


def test_the_lora_part_holds_the_tables_and_every_check_is_ok(script, lora_tree):
    root = lora_tree["root"]
    part = script.lora_part(root / "sft", {name: root / name for name in ("grpo_seed1", "grpo_seed2")})
    assert (part["n_linear_modules"], part["n_wrapped"]) == (14, 11)
    assert part["n_lora_params"] == sum(row["lora_params"] for row in lora_tree["rows"])
    assert (part["n_short_wrapped"], part["n_short_lora_params"]) == (2, RANK * (8 + 4) + RANK * (2 + 4))
    assert list(part["adapters"]) == ["sft", "grpo_seed1"] and list(part["differences"]) == ["grpo_seed1"]
    assert part["adapters"]["grpo_seed1"]["steps"] == 3 and part["adapters"]["sft"]["steps"] is None
    assert len(part["left_out"]) == 1 and "grpo_seed2" in part["left_out"][0]
    assert [check["ok"] for check in part["checks"]] == [True] * len(part["checks"])
    assert {check["name"] for check in part["checks"]} == {
        "coverage", "totals", "sft adapter file", "sft adapter config", "sft update norms",
        "grpo_seed1 adapter file", "grpo_seed1 started from the SFT adapter", "grpo_seed1 fingerprints"}
    assert part["sft_weight_change_largest_relative_difference"] < 1e-9
    markdown = script.lora_markdown({"lora": part})
    assert "| full attention | `self_attn` | `q_proj` | 8 x 4 | 1 of 1 | 24 | 1 of 1 | 24 |" in markdown
    assert "| linear attention | `linear_attn` | `in_proj_qkv` | 6 x 4 | 2 of 2 | 40 | 0 of 2 | 0 |" \
        in markdown
    assert "| All linear modules of the model |  |  |  | 11 of 14 |" in markdown
    assert "In this run 3 of 3 layers hold at least one wrapped module; with the short list 1 of 3" \
        in markdown
    assert "Modules with a zero update: sft: 1 of 11; grpo_seed1: 0 of 11." in markdown
    assert "So in each of these adapters no wrapped module has a zero update" not in markdown    # one is zero
    assert "modules that did not move at all: 1 of 11" in markdown
    assert "- OK (sft update norms): " in markdown and "DIFFERENT" not in markdown
    assert "Modules inside the layers with no adapter in this run: `in_proj_a` (linear attention, 2 of 2)." \
        in markdown


def test_a_manifest_that_does_not_fit_the_files_fails_its_check(script, lora_tree, tmp_path):
    import shutil

    root = tmp_path / "results"
    shutil.copytree(lora_tree["root"], root)
    manifest = copy.deepcopy(lora_tree["manifest"])
    manifest["weight_change"]["by_layer_type"]["full_attention"] *= 1.01
    manifest["lora"]["wrapped_modules"] += 1
    manifest["lora"]["coverage"][0]["wrapped"] += 1
    (root / "sft" / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    grpo_manifest = copy.deepcopy(lora_tree["grpo_manifest"])
    grpo_manifest["last_checkpoint"]["lora_fingerprint"] += 0.5
    grpo_manifest["start_adapter"]["files"]["adapter_model.safetensors"]["sha256"] = "0" * 64
    (root / "grpo_seed1" / "run_manifest.json").write_text(json.dumps(grpo_manifest), encoding="utf-8")
    part = script.lora_part(root / "sft", {"grpo_seed1": root / "grpo_seed1"})
    failed = {check["name"] for check in part["checks"] if check["ok"] is False}
    assert failed == {"coverage", "totals", "sft update norms", "grpo_seed1 started from the SFT adapter",
                      "grpo_seed1 fingerprints"}
    assert "- DIFFERENT (sft update norms): " in script.lora_markdown({"lora": part})


def test_the_lora_part_without_adapter_files_keeps_the_coverage_table(script, lora_tree, tmp_path):
    root = tmp_path / "results"
    (root / "sft").mkdir(parents=True)
    for name in ("run_manifest.json", "lora_coverage.json"):
        (root / "sft" / name).write_bytes((lora_tree["root"] / "sft" / name).read_bytes())
    part = script.lora_part(root / "sft", {"grpo_seed1": root / "grpo_seed1"})
    assert part["adapters"] == {} and part["differences"] == {}
    assert len(part["left_out"]) == 1 and "adapter_model.safetensors is not there" in part["left_out"][0]
    assert part["n_wrapped"] == 11
    markdown = script.lora_markdown({"lora": part})
    assert "## Table 1." in markdown and "Not made: no adapter file is there." in markdown
    with pytest.raises(ValueError, match="is not there"):
        script.lora_part(root / "nothing_here", {})


# ---------------------------------------------------------------------------
# A zero update: the rule, its reasons, and weights that are not numbers
# ---------------------------------------------------------------------------
OTHER_MODULE = "model.layers.2.self_attn.v_proj"            # lora_A [2, 4] and lora_B [2, 2], as SAME_MODULE


def test_a_product_that_is_zero_is_a_zero_update_also_when_neither_matrix_is_zero(script, lora_tree, tmp_path):
    """B x A can be zero while B and A are not. The rule "every entry of B is 0, or every entry of A is 0"
    does not see such a module; the norm that the script computes for it is exactly 0."""
    rows = {row["path"]: row for row in lora_tree["rows"]}
    weights = dict(lora_tree["sft"])
    a, b = (np.zeros_like(matrix) for matrix in weights[SAME_MODULE])
    a[1] = [1.0, 2.0, 3.0, 4.0]                    # row 0 of A is zero
    b[:, 0] = [1.0, -2.0]                          # column 1 of B is zero, so B x A uses only row 0 of A
    weights[SAME_MODULE] = (a, b)
    weights[OTHER_MODULE] = (np.zeros_like(weights[OTHER_MODULE][0]), weights[OTHER_MODULE][1])
    assert a.any() and b.any() and not full_update(weights, SAME_MODULE).any()
    write_adapter(tmp_path / "adapter", weights)
    numbers = script.adapter_numbers(tmp_path / "adapter", rows)
    assert numbers["zero_reasons"] == {ZERO_MODULE: "b_is_zero", SAME_MODULE: "product_is_zero",
                                       OTHER_MODULE: "a_is_zero"}
    assert numbers["zero_modules"] == [ZERO_MODULE, SAME_MODULE, OTHER_MODULE]
    assert numbers["n_zero_modules"] == 3
    assert numbers["n_zero_by_reason"] == {"b_is_zero": 1, "a_is_zero": 1, "product_is_zero": 1}
    assert [numbers["module_norms"][path] for path in numbers["zero_modules"]] == [0.0, 0.0, 0.0]
    assert numbers["norms"]["zero_by_layer_type"] == {"full_attention": 2, "linear_attention": 1}
    # Every other module has a norm above 0, and none of them is counted.
    assert all(norm > 0 for path, norm in numbers["module_norms"].items() if path not in numbers["zero_reasons"])
    # The SFT adapter of the tree: one module, and the reason is that lora_B never moved.
    sft = script.adapter_numbers(lora_tree["root"] / "sft" / "adapter", rows)
    assert sft["zero_reasons"] == {ZERO_MODULE: "b_is_zero"}
    assert sft["n_zero_by_reason"] == {"b_is_zero": 1, "a_is_zero": 0, "product_is_zero": 0}


def test_an_adapter_with_a_weight_that_is_not_a_finite_number_is_refused(script, lora_tree, tmp_path):
    """A NaN must not become a norm of 0 (max(0.0, nan) is 0.0), and so a "zero update"."""
    rows = {row["path"]: row for row in lora_tree["rows"]}
    for name, value in (("nan", np.nan), ("inf", np.inf)):
        weights = dict(lora_tree["sft"])
        a, b = weights[SAME_MODULE]
        broken = b.copy()
        broken[0, 0] = value
        weights[SAME_MODULE] = (a, broken)
        write_adapter(tmp_path / name, weights)
        with pytest.raises(ValueError, match=re.escape(SAME_MODULE) + ".*not a finite number"):
            script.adapter_numbers(tmp_path / name, rows)
        with pytest.raises(ValueError, match=re.escape(SAME_MODULE) + ".*not a finite number"):
            script.adapter_difference(lora_tree["root"] / "sft" / "adapter", tmp_path / name, rows)
    # A zero lora_B does not hide a lora_A that is not a number (before the repair the norm was 0.0).
    with pytest.raises(ValueError, match="not a finite number"):
        script.update_square(np.full((2, 3), np.nan, dtype=np.float32), np.zeros((4, 2), dtype=np.float32))
    # Finite weights that are too large for the sum in float64 (float32 weights cannot be).
    with pytest.raises(ValueError, match="not a finite number: the weights are too large"):
        script.update_square(np.array([[1e200, 0.0]]), np.array([[1e200], [0.0]]))


def test_a_norm_above_zero_is_never_written_as_zero(script):
    assert script._norm(0.0) == "0.0000"
    assert script._norm(0.0665) == "0.0665" and script._norm(2.88691) == "2.8869"
    assert script._norm(0.00006) == "0.0001"
    assert script._norm(4.9e-5) == "4.9e-05" and script._norm(3.2e-9) == "3.2e-09"


def test_lora_md_states_the_rule_of_a_zero_update_and_its_limit(script, lora_tree, tmp_path):
    import shutil

    root = lora_tree["root"]
    part = script.lora_part(root / "sft", {"grpo_seed1": root / "grpo_seed1"})
    markdown = script.lora_markdown({"lora": part})
    assert "Zero update: this sum is exactly 0 for the module." in markdown
    assert "That is always so when every entry of B is 0, or every entry of A is 0." in markdown
    assert "B x A can also be zero when neither B nor A is zero." in markdown
    assert "Only a sum of exactly 0 is counted, so the smallest norm of one module stands below the table." \
        in markdown
    assert "so a module in which every entry of B is still 0 is a module that training never moved" in markdown
    assert "Zero update: every entry of B is 0, or every entry of A is 0." not in markdown     # the old rule
    assert ("The modules with a zero update by reason (every entry of B is 0 / every entry of A is 0 / "
            "neither): sft: 1 / 0 / 0.") in markdown
    assert "So in each of these adapters no wrapped module has a zero update" not in markdown
    # A tree in which no module has a zero update: the SFT adapter is the GRPO adapter of the made-up tree.
    other = tmp_path / "results"
    shutil.copytree(root, other)
    final = lora_tree["grpo_manifest"]["final_adapter"]["folder"]
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (other / "sft" / "adapter" / name).write_bytes((root / "grpo_seed1" / final / name).read_bytes())
    part = script.lora_part(other / "sft", {"grpo_seed1": other / "grpo_seed1"})
    assert [adapter["n_zero_modules"] for adapter in part["adapters"].values()] == [0, 0]
    smallest = min(adapter["smallest"]["norm"] for adapter in part["adapters"].values())
    markdown = script.lora_markdown({"lora": part})
    assert (f"So in each of these adapters no wrapped module has a zero update (the smallest norm of one "
            f"module is {script._norm(smallest)}), and all 3 layers of both layer types hold wrapped "
            "modules.") in markdown
    assert "The modules with a zero update by reason" not in markdown


# ---------------------------------------------------------------------------
# The short target list: its published source
# ---------------------------------------------------------------------------
def test_the_short_list_is_given_with_its_published_source(script, lora_tree, tmp_path):
    import shutil

    source = script.SHORT_LIST_SOURCE
    model = next(iter(source["models_without_an_entry"]))

    def made(name: str, **more) -> tuple[dict, str]:
        """The part and lora.md of the made-up tree, with these extra entries in the SFT run manifest."""
        root = tmp_path / name
        shutil.copytree(lora_tree["root"], root)
        manifest = {**copy.deepcopy(lora_tree["manifest"]), **more}
        (root / "sft" / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        part = script.lora_part(root / "sft", {})
        assert [check["ok"] for check in part["checks"]] == [True] * len(part["checks"])
        return part, script.lora_markdown({"lora": part})

    default = ("This list is the default LoRA target list of the PEFT library (the Hugging Face library "
               "that puts LoRA adapters on a model) for the model types `qwen2` and `qwen3`, and also for "
               "`llama`, `mistral` and `gemma`. Published source: "
               "`TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` in `peft/utils/constants.py` of PEFT "
               f"{source['version']}")
    no_entry = "That table has no entry for the model type of this model"

    # The run names the version that was read, and the model whose model type was looked up.
    part, markdown = made("same", environment={"packages": {"peft": source["version"]}}, model=model)
    assert part["short_list_source"] == source
    assert (part["peft_version_of_the_run"], part["model"]) == (source["version"], model)
    assert default + ", the version of the SFT run (`environment.packages.peft` in `" in markdown
    assert (f"{no_entry} (`qwen3_5`; `qwen3_5_text` for its text part; both are in the `config.json` of "
            f"`{model}`). So PEFT gives no default list for this model: a run must name its list.") in markdown
    assert "judgement" not in markdown

    # The run names another version and another model: the file says what was read, and no more.
    part, markdown = made("other", environment={"packages": {"peft": "0.99.0"}}, model="Some/Other-Model")
    assert (part["peft_version_of_the_run"], part["model"]) == ("0.99.0", "Some/Other-Model")
    assert default + ". The SFT run manifest names PEFT 0.99.0: that version was not read." in markdown
    assert no_entry not in markdown and "the version of the SFT run" not in markdown

    # The run names neither.
    part, markdown = made("none")
    assert (part["peft_version_of_the_run"], part["model"]) == (None, None)
    assert default + ". A list of names wraps every module with one of these names." in markdown
    assert no_entry not in markdown and "The SFT run manifest names PEFT" not in markdown


PEFT_COPY = ROOT / ".cache" / "src" / "peft-0.21.3" / "peft"
MODEL_CONFIG_COPY = ROOT / ".cache" / "hf" / "Qwen3.5-2B" / "config.json"


@pytest.mark.skipif(not (PEFT_COPY / "utils" / "constants.py").is_file(),
                    reason="no copy of the PEFT 0.21.3 files in .cache/src to read the published table from")
def test_the_published_table_of_peft_says_what_lora_md_says_about_the_short_list(script):
    """The sentence of lora.md about the short list, against the files of the PEFT wheel. The files are
    only parsed; nothing of PEFT is imported or run."""
    import ast

    source = script.SHORT_LIST_SOURCE
    assert PEFT_COPY.parent.name == f"{source['package']}-{source['version']}"
    constants = PEFT_COPY.parent / source["file"]
    assert hashlib.sha256(constants.read_bytes()).hexdigest() == source["file_sha256"]
    assigned = [node.value for node in ast.parse(constants.read_text(encoding="utf-8")).body
                if isinstance(node, ast.Assign)
                and [getattr(target, "id", None) for target in node.targets] == [source["mapping"]]]
    assert len(assigned) == 1
    table = ast.literal_eval(assigned[0])
    for model_type in source["model_types_with_the_short_list"]:
        assert table[model_type] == list(script.SHORT_LIST), model_type
    for model_types in source["models_without_an_entry"].values():
        assert not set(model_types) & set(table)
    # With no list and no entry for the model type, PEFT stops: the entry is looked up by the exact name.
    lookup = (PEFT_COPY.parent / source["lookup_file"]).read_text(encoding="utf-8")
    assert f"def {source['lookup_function']}(" in lookup
    assert 'target_modules = self.target_module_mapping.get(model_config["model_type"])' in lookup
    assert 'raise ValueError("Please specify `target_modules` in `peft_config`")' in lookup
    if MODEL_CONFIG_COPY.is_file():
        config = json.loads(MODEL_CONFIG_COPY.read_text(encoding="utf-8"))
        assert [config["model_type"], config["text_config"]["model_type"]] \
            == source["models_without_an_entry"]["Qwen/Qwen3.5-2B"]


# ---------------------------------------------------------------------------
# The script on a results folder of the played arms
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def results(tmp_path_factory, played, scenarios, text, lora_tree) -> Path:
    """A results folder: the three played arms, their monitor reports as the frozen script writes them,
    and the SFT run and the GRPO run of the made-up adapters."""
    import shutil

    root = tmp_path_factory.mktemp("failure_report_results") / "results"
    shutil.copytree(lora_tree["root"], root)
    monitor_report = load_script("monitor_report")
    (root / "monitors").mkdir()
    for arm in ARMS:
        folder = root / f"{arm}_{SPLIT}"
        folder.mkdir()
        write_records(folder / "episodes.jsonl", played[arm])
        (folder / "run_manifest.json").write_text(json.dumps({"run_seed": 0, "samples": 8}), encoding="utf-8")
        report = monitor_report.report(folder / "episodes.jsonl", scenarios, SPLIT, Path(text.path), 3)
        (root / "monitors" / f"{arm}_{SPLIT}.md").write_text(report, encoding="utf-8")
    return root


def arguments(results: Path, scenarios: Path, text, out: Path, *more: str) -> list[str]:
    return ["--results", str(results), "--scenarios", str(scenarios), "--text", text.path, "--out", str(out),
            *more]


def test_the_script_writes_the_six_files_and_every_check_is_ok(script, capsys, tmp_path, results, scenarios,
                                                             text, chosen):
    out = tmp_path / "out"
    assert script.main(arguments(results, scenarios, text, out)) == 0
    printed = capsys.readouterr()
    assert printed.err == ""
    assert sorted(path.name for path in out.iterdir()) == ["failure_report.json", "flagged.md", "flips.csv",
                                                           "flips.md", "kinds.md", "lora.md"]
    assert "DIFFERENT" not in printed.out
    assert "check OK: flagged: base: the saved monitor report" in printed.out
    report = json.loads((out / "failure_report.json").read_text(encoding="utf-8"))
    assert report["split"] == SPLIT and report["parts_left_out"] == []
    assert [source["arm"] for source in report["sources"]["arms"]] == list(ARMS)
    assert [absent["arm"] for absent in report["sources"]["arms_not_present"]] == ["grpo_seed2"]
    assert all(check["ok"] for check in report["checks"]) and len(report["checks"]) >= 3 * 3 + 8
    assert all((source["n_attempts"], source["run_seed"]) == (256, 0) for source in report["sources"]["arms"])

    kinds = (out / "kinds.md").read_text(encoding="utf-8")
    by_kind, _ = expected_kinds("base", chosen)
    n_failed = sum(by_kind.values())
    base = report["kinds"]["arms"]["base"]
    assert f"| Sum of the 8 kinds | {n_failed} |" in kinds
    assert (f"| Failed attempts (valid attempts minus passed attempts) | {n_failed} = 256 - "
            f"{256 - n_failed} |") in kinds
    a1 = by_kind["no_valid_call"]
    assert f"| A1 no valid call | {a1} ({100 * a1 / n_failed:.1f}%) |" in kinds
    assert "Arm grpo_seed2: not in this report." in kinds
    assert f"### base: {n_failed} failed attempts of 256" in kinds
    assert "| `text_after_call` | 16 / 16 | 0 / 0 | 0 / 0 |" in kinds
    assert base["n_failed"] == n_failed
    lenient = report["kinds"]["lenient"]["base"]
    assert f"| Passed attempts with the lenient parser: lowest to highest possible number |  | " \
           f"{lenient['passed_bounds'][0]} to {lenient['passed_bounds'][1]} |" in kinds
    assert "lenient parser reads, and that pass when this call is run: base: 4; sft: 0; grpo_seed1: 0." \
        in kinds

    flips = (out / "flips.md").read_text(encoding="utf-8")
    assert "| base | sft | 32 | 12 (37.5%) | 4 (12.5%) | 16 (50.0%) | 4 | 8 | 4 |" in flips
    assert "| sft | grpo_seed1 | 32 | 4 (12.5%) | 1 (3.1%) | 27 (84.4%) | 19 | 4 | 4 |" in flips
    assert "| base | sft | 8 | 5 (62.5%) | 1 (12.5%) | 2 (25.0%) |" in flips
    assert "Not in Tables 1 to 4: the change sft to grpo_seed2." in flips
    worse = chosen["already+same"]
    status = ", ".join(person["status"] for person in other_people_of_the_name(worse))
    assert f"| base | sft | {worse['family_id']} | 32 of 32 | 12 of 32 | -62.5 | already | active | yes |" \
        in flips
    assert f"| yes | {status} |" in flips
    rows = (out / "flips.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 32
    assert rows[0].split(",")[:5] == ["scenario_id", "family_id", "style", "held_out_style",
                                      "expected_outcome"]
    assert rows[0].split(",")[-2:] == ["flip_base_to_sft", "flip_sft_to_grpo_seed1"]
    first = dict(zip(rows[0].split(","), rows[1].split(",")))
    scenario = next(row for row in report["flips"]["scenarios"] if row["scenario_id"] == first["scenario_id"])
    assert int(first["passes_base"]) == scenario["passes"]["base"] and first["attempts_sft"] == "8"
    assert first["flip_base_to_sft"] == scenario["flips"]["base>sft"]

    flagged = (out / "flagged.md").read_text(encoding="utf-8")
    summary = report["flagged"]["arms"]["base"]
    m5 = summary["monitors"]["words_outside_call"]
    plain = 100 * m5["n_passed"] / summary["n_passed"]
    assert (f"| m5 `words_outside_call` | {m5['n_passed']} ({100 * m5['share_of_passed']:.1f}%; "
            f"{plain:.1f}%) |") in flagged
    clean = summary["passed_clean"]
    assert f"| passed_clean | {clean['n']} ({100 * clean['share_of_all']:.1f}%)" in flagged
    assert flagged.count("- OK: ") == 3 and "All 11 lines are in" in flagged
    assert "minus 5 for each malformed message (at most 15)" in flagged

    lora = (out / "lora.md").read_text(encoding="utf-8")
    assert "## Table 3. How far each GRPO run moved the adapter from the SFT adapter" in lora
    assert "Not in this report:" in lora and "grpo_seed2" in lora


def test_every_number_of_the_markdown_files_is_in_the_json_file(script, tmp_path, results, scenarios, text):
    """The markdown files are made from the report dict alone: the same dict, read back from the JSON
    file, gives the same four files."""
    out = tmp_path / "out"
    assert script.main(arguments(results, scenarios, text, out)) == 0
    report = json.loads((out / "failure_report.json").read_text(encoding="utf-8"))
    assert script.kinds_markdown(report) == (out / "kinds.md").read_text(encoding="utf-8")
    assert script.flips_markdown(report) == (out / "flips.md").read_text(encoding="utf-8")
    assert script.flips_csv(report) == (out / "flips.csv").read_text(encoding="utf-8")
    assert script.flagged_markdown(report) == (out / "flagged.md").read_text(encoding="utf-8")
    assert script.lora_markdown(report) == (out / "lora.md").read_text(encoding="utf-8")


def test_the_script_changes_no_input_file(script, tmp_path, results, scenarios, text):
    def hashes(folder: Path) -> dict[str, str]:
        return {path.relative_to(folder).as_posix(): sha256(path) for path in sorted(folder.rglob("*"))
                if path.is_file()}
    before = (hashes(results), hashes(scenarios))
    assert script.main(arguments(results, scenarios, text, tmp_path / "out")) == 0
    assert (hashes(results), hashes(scenarios)) == before


def test_the_default_output_folder_is_under_the_results_folder(script, tmp_path):
    assert script.default_out(tmp_path, "dev") == tmp_path / "report" / "failures"
    assert script.default_out(tmp_path, "test") == tmp_path / "report" / "failures" / "test"


def test_a_saved_monitor_report_with_other_numbers_fails_the_check_and_the_files_say_so(
        script, capsys, tmp_path, results, scenarios, text):
    monitors_dir = tmp_path / "monitors"
    monitors_dir.mkdir()
    for arm in ARMS:
        saved = (results / "monitors" / f"{arm}_{SPLIT}.md").read_text(encoding="utf-8")
        if arm == "sft":
            saved = saved.replace("Passed: ", "Passed: 1")
        (monitors_dir / f"{arm}_{SPLIT}.md").write_text(saved, encoding="utf-8")
    out = tmp_path / "out"
    code = script.main(arguments(results, scenarios, text, out, "--monitors", str(monitors_dir)))
    printed = capsys.readouterr()
    assert code == 1
    assert "a check failed: flagged: sft: the saved monitor report" in printed.err
    flagged = (out / "flagged.md").read_text(encoding="utf-8")
    assert "- DIFFERENT: sft. 1 of 11 lines are not in" in flagged and flagged.count("- OK: ") == 2


def test_the_split_test_reads_no_scenario_file_while_no_test_run_is_saved(script, capsys, tmp_path, results,
                                                                        scenarios, text, monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("the scenario file of the split was opened, and no run of it is saved")
    monkeypatch.setattr(script, "load_run", must_not_be_called)
    out = tmp_path / "out"
    assert script.main(arguments(results, scenarios, text, out, "--split", "test")) == 2
    printed = capsys.readouterr()
    assert "nothing to write: kinds, flips, flagged: no run folder of the split test holds episodes.jsonl" \
        in printed.err
    assert not out.exists()                                   # nothing was written


def test_with_another_split_the_adapter_table_is_made_only_if_it_is_asked_for(script, capsys, tmp_path,
                                                                            results, scenarios, text):
    out = tmp_path / "out"
    assert script.main(arguments(results, scenarios, text, out, "--split", "test", "--parts", "lora")) == 0
    assert sorted(path.name for path in out.iterdir()) == ["failure_report.json", "lora.md"]
    assert "wrote failure_report.json, lora.md" in capsys.readouterr().out


def test_the_script_takes_the_parts_the_arms_and_the_pairs_that_it_is_given(script, capsys, tmp_path, results,
                                                                         scenarios, text):
    out = tmp_path / "out"
    code = script.main(arguments(
        results, scenarios, text, out, "--parts", "kinds", "flips",
        "--arm", f"first={results / 'base_dev'}", "--arm", f"second={results / 'sft_dev'}",
        "--pair", "second:first"))
    assert code == 0
    assert sorted(path.name for path in out.iterdir()) == ["failure_report.json", "flips.csv", "flips.md",
                                                           "kinds.md"]
    report = json.loads((out / "failure_report.json").read_text(encoding="utf-8"))
    assert list(report["kinds"]["arms"]) == ["first", "second"]
    pair = report["flips"]["pairs"][0]
    assert (pair["from"], pair["to"]) == ("second", "first")
    assert pair["scenarios"]["pass_to_fail"] == 12 and pair["scenarios"]["fail_to_pass"] == 4   # the reverse
    assert "flagged" not in report and "lora" not in report
    capsys.readouterr()
    assert script.main(arguments(results, scenarios, text, out, "--arm", f"a={results / 'base_dev'}",
                                 "--arm", f"a={results / 'sft_dev'}")) == 2
    assert "is given twice" in capsys.readouterr().err


def test_arms_that_do_not_cover_the_same_attempts_are_refused(script, capsys, tmp_path, results, scenarios,
                                                             text):
    folder = tmp_path / "short_dev"
    folder.mkdir()
    write_records(folder / "episodes.jsonl", read_records(results / "sft_dev" / "episodes.jsonl")[8:])
    code = script.main(arguments(results, scenarios, text, tmp_path / "out",
                                 "--arm", f"base={results / 'base_dev'}", "--arm", f"sft={folder}"))
    assert code == 2
    assert "does not cover the same scenarios and sample counts" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_records_of_another_scenario_file_or_text_file_are_refused(script, capsys, tmp_path, results,
                                                                  scenarios, text):
    records = read_records(results / "base_dev" / "episodes.jsonl")
    for key, value, message in [("data_sha256", "0" * 64, "is not the file that the run was made from"),
                                ("text_sha256", "1" * 64, "is not the text file of the run")]:
        folder = tmp_path / f"other_{key}"
        folder.mkdir()
        write_records(folder / "episodes.jsonl", [{**record, key: value} for record in records])
        code = script.main(arguments(results, scenarios, text, tmp_path / "out", "--arm", f"base={folder}"))
        assert code == 2
        assert message in capsys.readouterr().err


def test_a_changed_verdict_is_refused_by_the_frozen_monitors(script, capsys, tmp_path, results, scenarios,
                                                            text):
    """A record whose saved pass is not what the verifier gives for its saved final state."""
    records = read_records(results / "base_dev" / "episodes.jsonl")
    failed = next(record for record in records if not record["passed"] and record["closed"])
    failed["passed"] = True
    folder = tmp_path / "forged_dev"
    folder.mkdir()
    write_records(folder / "episodes.jsonl", records)
    code = script.main(arguments(results, scenarios, text, tmp_path / "out", "--arm", f"base={folder}"))
    assert code == 2
    assert "the verifier does not give the saved pass or fail" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The real saved runs (skipped where they are not there)
# ---------------------------------------------------------------------------
REAL = ROOT / "results" / "main"
REAL_ARMS = ("base", "sft", "grpo_seed1", "grpo_seed2")


@pytest.mark.skipif(not all((REAL / f"{arm}_dev" / "episodes.jsonl").is_file() for arm in REAL_ARMS),
                    reason="the saved dev runs of the four arms are not in results/main")
def test_the_script_on_the_real_saved_dev_runs(script, capsys, tmp_path):
    out = tmp_path / "out"
    assert script.main(["--out", str(out)]) == 0
    printed = capsys.readouterr()
    assert printed.err == "" and "DIFFERENT" not in printed.out
    report = json.loads((out / "failure_report.json").read_text(encoding="utf-8"))
    assert list(report["kinds"]["arms"]) == list(REAL_ARMS)
    assert all(check["ok"] is not False for check in report["checks"])
    for arm in REAL_ARMS:
        row = report["kinds"]["arms"][arm]
        assert row["sum_of_kinds"] == row["n_failed"] == row["n_attempts"] - row["n_passed"]
        assert row["by_kind"]["other"]["n"] == 0
        assert (row["n_attempts"], row["n_harness_errors"]) == (1536, 0)
        summary = report["flagged"]["arms"][arm]
        assert summary["n_passed"] == row["n_passed"] and summary["pass_rate"] == row["pass_rate"]
        assert report["flagged"]["saved_reports"][arm]["equal"] in (True, None)
        lenient = report["kinds"]["lenient"][arm]
        assert lenient["passed_bounds"][0] <= row["n_passed"] <= lenient["passed_bounds"][1]
    # The frozen results tables hold the failed attempts of an arm as the attempts with reward 0.
    for name in ("base_sft_grpo_seed1", "base_sft_grpo_seed2"):
        table = REAL / "tables" / f"{name}.json"
        if table.is_file():
            for arm_row in json.loads(table.read_text(encoding="utf-8"))["arms"]:
                mine = report["kinds"]["arms"][arm_row["arm"]]
                assert mine["n_failed"] == arm_row["reward_distribution"]["0"]
                assert mine["pass_rate"] == pytest.approx(arm_row["pass_rate"], abs=1e-12)
    flips = report["flips"]
    assert len(flips["scenarios"]) == 192 and len(flips["families"]) == 48
    assert [(pair["from"], pair["to"]) for pair in flips["pairs"]] \
        == [("base", "sft"), ("sft", "grpo_seed1"), ("sft", "grpo_seed2")]
    for pair in flips["pairs"]:
        assert sum(pair["scenarios"].values()) == 192 and pair["scenarios"]["not_classified"] == 0
        assert sum(pair["families"].values()) == 48
    assert len((out / "flips.csv").read_text(encoding="utf-8").splitlines()) == 193
    # The families with the most failed attempts, counted a second way: from the passes per family of the
    # flips part. Every family with the largest count is named, not only the one with the smallest id.
    for arm in REAL_ARMS:
        failed = {row["family_id"]: row["attempts"][arm] - row["passes"][arm] for row in flips["families"]}
        most = max(failed.values())
        tied = report["kinds"]["arms"][arm]["families_tied_for_most_failed"]
        assert [entry["family_id"] for entry in tied] \
            == sorted(family_id for family_id, n in failed.items() if n == most and n > 0)
        assert all(entry["n_failed"] == most for entry in tied)
    # The saved output files that are there are what the script writes now. The package saves only flips.csv:
    # the readable .md files are compared only if someone writes them into results/main/report/failures
    # (scripts/rebuild_results.sh writes them to its own folder and prints them as NEW).
    saved = REAL / "report" / "failures"
    adapters = all((REAL / folder / "adapter" / "adapter_model.safetensors").is_file() for folder in (
        "sft", "grpo_seed1/checkpoints/step_00080", "grpo_seed2/checkpoints/step_00080"))
    for name in ("kinds.md", "flips.md", "flips.csv", "flagged.md") + (("lora.md",) if adapters else ()):
        if (saved / name).is_file():
            now, before = [(folder / name).read_text(encoding="utf-8") for folder in (out, saved)]
            assert now == before, name
    if adapters:
        lora = report["lora"]
        assert (lora["n_wrapped"], lora["n_linear_modules"]) == (150, 187)
        assert all(adapter["n_zero_modules"] == 0 for adapter in lora["adapters"].values())
        assert list(lora["adapters"]) == ["sft", "grpo_seed1", "grpo_seed2"]
