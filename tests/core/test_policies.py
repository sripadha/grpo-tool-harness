"""accessdesk/harness/policies.py: the tools of the scripted policies, and what a policy may see.

What each policy scores on the scenarios is tested in test_validation_suite.py.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import re

import pytest

from accessdesk.env.generator import make_split
from accessdesk.env.rules import RULESETS, decide
from accessdesk.harness import policies
from accessdesk.harness.episode import Episode
from accessdesk.harness.parser import parse_assistant_message
from accessdesk.harness.policies import (Desk, named_policies, oracle, oracle_action, random_valid_calls,
                                         render_call, request_reader, run_policy, scripted)
from accessdesk.harness.render import render_tool_call
from accessdesk.rng import TOKEN_ALPHABET
from accessdesk.types import RESOLUTIONS
from tests.conftest import MARKER_PATTERN
from tests.core.test_episode import EXPECTED_DENIED, EXPECTED_PENDING, FAMILY


def hand_family(text) -> dict:
    """The hand-made family of test_episode.py, with wordings from the templates of the text file.

    The role r1 (not requested) gets the second role name of the text file, so that a policy that tries the
    role names in order misses once and then finds it.
    """
    family = copy.deepcopy(FAMILY)
    assert family["request"]["role_name"] not in text.roles[:2]
    family["world"]["roles"][0]["name"] = text.roles[1]
    family["wordings"] = [{"style": style, "template": 0,
                           "text": text.request_templates[style][0].format(**family["request"])}
                          for style in (1, 2, 3, 4)]
    return family


def call(text, tool_key: str, **args: str) -> str:
    """The assistant text of one call, built without policies.py. A resolution is given as a visible value."""
    spec = text.tools[tool_key]
    return render_tool_call(spec.name, {param.name: args[param.key] for param in spec.params})


def all_scenarios(families: dict) -> list[tuple[dict, dict]]:
    return [(family, wording) for split in sorted(families) for family in families[split]
            for wording in family["wordings"]]


# ---------------------------------------------------------------------------
# The rules of the oracle
# ---------------------------------------------------------------------------
def test_the_oracles_rules_typed_by_hand():
    #        active has    sens     default     all need approval  sensitive denied
    rows = [(False, False, False, "deny",     "deny",            "deny"),
            (False, False, True,  "deny",     "deny",            "deny"),
            (False, True,  False, "deny",     "deny",            "deny"),
            (False, True,  True,  "deny",     "deny",            "deny"),
            (True,  True,  False, "already",  "already",         "already"),
            (True,  True,  True,  "already",  "already",         "already"),
            (True,  False, False, "grant",    "approval",        "grant"),
            (True,  False, True,  "approval", "approval",        "deny")]
    assert len({row[:3] for row in rows}) == 8
    for active, has_access, sensitive, *actions in rows:
        got = [oracle_action(active, has_access, sensitive, ruleset)
               for ruleset in ("default", "all_grants_need_approval", "sensitive_denied")]
        assert got == actions, (active, has_access, sensitive)
    assert oracle_action(True, False, True) == "approval"       # the default rule set is the default


def test_the_oracles_rules_agree_with_the_rules_of_the_environment_on_every_input():
    """Two statements of the same rules, written apart: policies.oracle_action and rules.decide."""
    assert "decide" not in vars(policies)       # the oracle does not borrow the environment's function
    cases = list(itertools.product(RULESETS, (False, True), (False, True), (False, True)))
    assert len(cases) == 24
    for ruleset, active, has_access, sensitive in cases:
        assert oracle_action(active, has_access, sensitive, ruleset) == \
            decide(active, has_access, sensitive, ruleset)
    with pytest.raises(ValueError, match="unknown rule set"):
        oracle_action(True, False, False, "none")


# ---------------------------------------------------------------------------
# Reading the request
# ---------------------------------------------------------------------------
def test_the_request_reader_recovers_the_request_of_every_scenario(text, families):
    read = request_reader(text)
    scenarios = all_scenarios(families)
    assert len(scenarios) == 1080
    for family, wording in scenarios:
        assert read(wording["text"]) == family["request"], wording["text"]


def test_the_request_reader_refuses_a_sentence_it_cannot_read(text):
    read = request_reader(text)
    for sentence in ("", "Please help.", text.request_templates[1][0]):
        with pytest.raises(ValueError, match="0 readings"):
            read(sentence)
    known = text.request_templates[1][0].format(person_name="Ada Stone", department=text.departments[0],
                                                role_name=text.roles[0])
    assert read(known) == {"person_name": "Ada Stone", "department": text.departments[0],
                           "role_name": text.roles[0]}
    unknown_role = text.request_templates[1][0].format(
        person_name="Ada Stone", department=text.departments[0], role_name="Moon Base Keeper")
    with pytest.raises(ValueError, match="0 readings"):
        read(unknown_role)


def test_the_request_reader_refuses_a_sentence_with_two_readings(text):
    two = ("{person_name} in {department}: {role_name}", "{person_name} {department}: {role_name}")
    templates = {style: two for style in (1, 2, 3, 4)}
    read = request_reader(dataclasses.replace(text, request_templates=templates))
    with pytest.raises(ValueError, match="2 readings"):
        read(f"Ada Stone in {text.departments[0]}: {text.roles[0]}")


# ---------------------------------------------------------------------------
# Writing a call
# ---------------------------------------------------------------------------
def test_render_call_writes_visible_names_and_the_visible_resolution(text):
    specs = text.tool_specs("five")
    args = {"role_id": "r2", "resolution": "pending", "person_id": "p1"}
    got = render_call("close_request", args, specs, text)
    assert got == call(text, "close_request", resolution=text.resolution["pending"], person_id="p1",
                       role_id="r2")
    parsed = parse_assistant_message(got, specs)
    assert parsed.ok and parsed.args == {"resolution": text.resolution["pending"], "person_id": "p1",
                                         "role_id": "r2"}
    assert render_call("lookup_person", {"name": "Ada Stone"}, specs, text) == \
        call(text, "lookup_person", name="Ada Stone")


def test_render_call_refuses_a_tool_outside_the_tool_list_and_wrong_arguments(text):
    five, six = text.tool_specs("five"), text.tool_specs("six")
    assert parse_assistant_message(render_call("department_info", {"department": "Sales"}, six, text), six).ok
    with pytest.raises(ValueError, match="no tool"):
        render_call("department_info", {"department": "Sales"}, five, text)
    with pytest.raises(ValueError, match="takes"):
        render_call("lookup_person", {"full_name": "Ada Stone"}, five, text)
    with pytest.raises(ValueError, match="takes"):
        render_call("grant_role", {"person_id": "p1"}, five, text)


# ---------------------------------------------------------------------------
# A policy sees the messages and nothing else
# ---------------------------------------------------------------------------
def conversation(text, family: dict, assistant_and_tool: list[tuple[str, dict]]) -> list[dict]:
    """Messages written by hand: system, the request in style 1, then (assistant text, tool answer) pairs."""
    messages = [{"role": "system", "content": text.system_prompt},
                {"role": "user", "content": family["wordings"][0]["text"]}]
    for assistant_text, answer in assistant_and_tool:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "tool", "content": json.dumps(answer)})
    return messages


def person_answer(text, person_id: str, department: str, status: str, manager_id: str | None) -> dict:
    keys = text.result_keys
    return {keys["person_id"]: person_id, keys["name"]: "Ada Stone", keys["department"]: department,
            keys["status"]: text.status[status], keys["manager_id"]: manager_id}


def check_answer(text, role_id: str, sensitive: bool, has_access: bool) -> dict:
    keys = text.result_keys
    return {keys["role_id"]: role_id, keys["sensitive"]: sensitive, keys["has_access"]: has_access}


def test_the_oracle_walks_the_correct_path_on_messages_written_by_hand(text):
    """No Episode and no world here: the policy gets a list of messages and must answer from it alone."""
    family, specs, policy = hand_family(text), text.tool_specs("five"), oracle(text)
    lookup = call(text, "lookup_person", name="Ada Stone")
    check = call(text, "check_access", person_id="p1", role_name="Payroll Editor")
    approval = call(text, "request_approval", person_id="p1", role_id="r2", approver_id="p3")
    close = call(text, "close_request", resolution=text.resolution["pending"], person_id="p1", role_id="r2")
    people = {text.result_keys["matches"]: [person_answer(text, "p2", "Sales", "inactive", "p3"),
                                            person_answer(text, "p1", "Finance", "active", "p3")]}
    confirmation = {text.result_keys["result"]: text.confirmation["approval_requested"]}
    steps = [(lookup, people), (check, check_answer(text, "r2", True, False)), (approval, confirmation)]
    assert policy(conversation(text, family, []), specs) == lookup
    assert policy(conversation(text, family, steps[:1]), specs) == check       # p1: the one in Finance
    assert policy(conversation(text, family, steps[:2]), specs) == approval
    assert policy(conversation(text, family, steps[:3]), specs) == close


@pytest.mark.parametrize("status, sensitive, has_access, ruleset, next_call", [
    ("active", False, False, "default", ("grant_role", {})),
    ("active", True, False, "default", ("request_approval", {"approver_id": "p3"})),
    ("active", False, True, "default", ("close_request", {"resolution": "already"})),
    ("active", True, True, "default", ("close_request", {"resolution": "already"})),
    ("inactive", False, False, "default", ("close_request", {"resolution": "denied"})),
    ("inactive", True, True, "default", ("close_request", {"resolution": "denied"})),
    ("active", False, False, "all_grants_need_approval", ("request_approval", {"approver_id": "p3"})),
    ("active", True, False, "sensitive_denied", ("close_request", {"resolution": "denied"})),
])
def test_the_oracle_decides_from_the_tool_answers_in_the_messages(text, status, sensitive, has_access,
                                                                  ruleset, next_call):
    """The answers below are made up and contradict the family record. The oracle must follow the answers."""
    family, specs = hand_family(text), text.tool_specs("five")
    steps = [(call(text, "lookup_person", name="Ada Stone"),
              {text.result_keys["matches"]: [person_answer(text, "p1", "Finance", status, "p3")]}),
             (call(text, "check_access", person_id="p1", role_name="Payroll Editor"),
              check_answer(text, "r2", sensitive, has_access))]
    tool_key, extra = next_call
    if "resolution" in extra:
        extra = {"resolution": text.resolution[extra["resolution"]]}
    wanted = call(text, tool_key, **{"person_id": "p1", "role_id": "r2", **extra})
    assert oracle(text, ruleset)(conversation(text, family, steps), specs) == wanted


def test_a_policy_is_a_function_of_the_messages_and_does_not_change_them(text, cfg):
    family = hand_family(text)
    for name, policy in named_policies(text).items():
        episode = Episode(family, 2, text, cfg)
        while not episode.done:
            before = copy.deepcopy(episode.messages)
            first = policy(episode.messages, episode.tool_specs)
            second = policy(copy.deepcopy(before), list(episode.tool_specs))
            assert first == second and isinstance(first, str), name
            assert episode.messages == before, name
            episode.step(first)


def test_a_policy_cannot_be_given_a_family_record(text):
    """The only way into a policy is (messages, tool_specs)."""
    policy = oracle(text)
    with pytest.raises(TypeError):
        policy([], [], hand_family(text))


def test_the_oracle_solves_the_hand_made_family_under_each_rule_set(text, cfg):
    family = hand_family(text)
    for ruleset, expected, n_turns in (("default", EXPECTED_PENDING, 4), ("all_grants_need_approval",
                                       EXPECTED_PENDING, 4), ("sensitive_denied", EXPECTED_DENIED, 3)):
        episode = run_policy(Episode(family, 3, text, cfg, ruleset=ruleset), oracle(text, ruleset))
        record = episode.record()
        assert episode.done and record["final_state"] == expected
        assert (record["passed"], record["reward"], record["n_turns"]) == (True, 100, n_turns)


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------
def test_desk_gives_answers_with_internal_keys(text):
    family = hand_family(text)
    lookup = call(text, "lookup_person", name="Ada Stone")
    answer = {text.result_keys["matches"]: [person_answer(text, "p1", "Finance", "active", None)]}
    request = request_reader(text)(family["wordings"][0]["text"])
    desk = Desk(conversation(text, family, [(lookup, answer)]), text.tool_specs("five"), text, request)
    assert desk.request == family["request"]
    people = desk.call("lookup_person", name="Ada Stone")["matches"]
    assert people == [{"person_id": "p1", "name": "Ada Stone", "department": "Finance",
                       "status": text.status["active"], "manager_id": None}]
    assert desk.is_active(people[0]) is True
    assert desk.is_active({"status": text.status["inactive"]}) is False


def test_a_script_that_does_not_repeat_itself_is_refused(text, cfg):
    names = iter(["Ada Stone", "Bo Reed", "Cy Wood"])
    policy = scripted(text, lambda desk: [desk.call("lookup_person", name=next(names)) for _ in range(9)])
    episode = Episode(hand_family(text), 1, text, cfg)
    episode.step(policy(episode.messages, episode.tool_specs))        # sends the first name
    with pytest.raises(RuntimeError, match="does not repeat"):
        policy(episode.messages, episode.tool_specs)                  # now starts with the second name


def test_a_script_that_ends_before_the_episode_is_refused(text, cfg):
    policy = scripted(text, lambda desk: desk.call("lookup_person", name="Ada Stone"))
    episode = Episode(hand_family(text), 1, text, cfg)
    episode.step(policy(episode.messages, episode.tool_specs))
    with pytest.raises(RuntimeError, match="script has ended"):
        policy(episode.messages, episode.tool_specs)


def test_run_policy_plays_to_the_end_and_returns_the_episode(text, cfg):
    episode = Episode(hand_family(text), 1, text, cfg)
    assert run_policy(episode, policies.do_nothing()) is episode
    assert episode.done and len(episode.turns) == cfg.turn_limit and episode.n_malformed == cfg.turn_limit


# ---------------------------------------------------------------------------
# Single policies, on the hand-made family (p1 is meant; p2 has the same name; p3 is the manager of both)
# ---------------------------------------------------------------------------
def record_of(text, cfg, policy, **kwargs) -> dict:
    return run_policy(Episode(hand_family(text), 1, text, cfg, **kwargs), policy).record()


def test_wrong_person_takes_the_homonym_or_the_manager(text, cfg):
    homonym = record_of(text, cfg, policies.wrong_person(text, "homonym"))
    manager = record_of(text, cfg, policies.wrong_person(text, "manager"))
    assert homonym["final_state"]["request"] == {"resolution": "denied", "person_id": "p2", "role_id": "r2"}
    assert manager["final_state"]["request"] == {"resolution": "already", "person_id": "p3", "role_id": "r2"}
    assert homonym["turns"][1]["args"]["person_id"] == "p2"          # the access check was for that person
    assert manager["turns"][1]["args"]["person_id"] == "p3"
    with pytest.raises(ValueError):
        policies.wrong_person(text, "nobody")


def test_wrong_role_and_extra_grant_find_the_other_role_by_asking(text, cfg):
    wrong = record_of(text, cfg, policies.wrong_role(text))
    assert wrong["final_state"]["request"] == {"resolution": "already", "person_id": "p1", "role_id": "r1"}
    assert (wrong["n_tool_errors"], wrong["n_malformed"]) == (1, 0)        # the miss is a tool error
    extra = record_of(text, cfg, policies.extra_grant(text, "second_role"))
    assert extra["final_state"]["grants"] == [["p1", "r1"]]
    assert extra["final_state"]["approvals"] == [["p1", "r2", "p3"]]
    same = record_of(text, cfg, policies.extra_grant(text, "same_role"))
    assert same["final_state"]["grants"] == [["p1", "r2"]]


def test_the_constant_policies_act_for_their_resolution_whatever_the_facts(text, cfg):
    states = {key: record_of(text, cfg, policies.constant(text, key))["final_state"] for key in RESOLUTIONS}
    assert {key: state["request"]["resolution"] for key, state in states.items()} == \
        {key: key for key in RESOLUTIONS}
    assert all(state["request"]["person_id"] == "p1" and state["request"]["role_id"] == "r2"
               for state in states.values())
    assert states["granted"]["grants"] == [["p1", "r2"]] and states["granted"]["approvals"] == []
    assert states["pending"] == EXPECTED_PENDING
    assert states["already"]["grants"] == [] and states["denied"]["approvals"] == []


def test_close_at_once_uses_ids_of_the_right_shape_that_no_world_has(text, cfg, families):
    record = record_of(text, cfg, policies.close_at_once(text, "granted"))
    args = record["turns"][0]["args"]
    token = f"[{TOKEN_ALPHABET}]{{6}}"
    assert re.fullmatch(re.escape(text.id_prefix_person) + token, args["person_id"])
    assert re.fullmatch(re.escape(text.id_prefix_role) + token, args["role_id"])
    assert args["person_id"][len(text.id_prefix_person):] != args["role_id"][len(text.id_prefix_role):]
    known = {person["person_id"] for split in families for family in families[split]
             for person in family["world"]["people"]}
    known |= {role["role_id"] for split in families for family in families[split]
              for role in family["world"]["roles"]}
    assert len(known) > 3000 and args["person_id"] not in known and args["role_id"] not in known
    other_seed = record_of(text, cfg, policies.close_at_once(text, "granted", seed=1))["turns"][0]["args"]
    assert other_seed["person_id"] != args["person_id"]


def test_the_extra_tool_policy_needs_the_six_tool_list(text, cfg):
    record = record_of(text, cfg, policies.extra_tool_then_oracle(text), toolset="six")
    assert [turn["tool_key"] for turn in record["turns"]] == [
        "department_info", "lookup_person", "check_access", "request_approval", "close_request"]
    assert record["turns"][0]["tool_error"] is None and record["reward"] == 100
    with pytest.raises(ValueError, match="no tool"):
        record_of(text, cfg, policies.extra_tool_then_oracle(text), toolset="five")


def test_the_stumbling_oracle_sends_empty_messages_first(text, cfg):
    record = record_of(text, cfg, policies.stumbling_oracle(text, 2))
    assert [turn["parse_error"] for turn in record["turns"]] == ["no_call", "no_call", None, None, None, None]
    assert [turn["assistant_text"] for turn in record["turns"][:2]] == ["", ""]


def test_claim_in_text_says_the_confirmation_of_the_close_tool(text, cfg):
    record = record_of(text, cfg, policies.claim_in_text(text))
    said = [turn["assistant_text"] for turn in record["turns"][3:]]
    assert said == [text.confirmation["request_closed"]] * 5
    assert record["final_state"]["approvals"] == [["p1", "r2", "p3"]] and record["closed"] is False


# ---------------------------------------------------------------------------
# Random valid calls
# ---------------------------------------------------------------------------
def random_records(text, cfg, families: list[dict], seed: int, toolset: str = "five") -> list[dict]:
    policy = random_valid_calls(text, seed)
    return [run_policy(Episode(family, wording["style"], text, cfg, toolset=toolset), policy).record()
            for family in families for wording in family["wordings"]]


def test_random_valid_calls_are_all_well_formed_and_use_every_tool(text, cfg, families):
    for toolset, n_tools in (("five", 5), ("six", 6)):
        records = random_records(text, cfg, families["dev"], seed=0, toolset=toolset)
        turns = [turn for record in records for turn in record["turns"]]
        assert len(turns) > 1000 and all(turn["parse_ok"] for turn in turns)
        assert len({turn["tool_key"] for turn in turns}) == n_tools
        served = {turn["tool_key"] for turn in turns if turn["tool_error"] is None}
        assert len(served) == n_tools              # every tool was called at least once without an error
        assert any(turn["tool_error"] is not None for turn in turns)


def test_random_valid_calls_depend_on_the_seed_and_on_nothing_else(text, cfg, families):
    first = random_records(text, cfg, families["calib"], seed=0)
    again = random_records(text, cfg, families["calib"], seed=0)
    other = random_records(text, cfg, families["calib"], seed=1)
    assert json.dumps(first) == json.dumps(again)
    # The tools that are chosen differ, not only the made-up ids (which depend on the seed as well).
    assert [r["turns"][0]["tool_key"] for r in first] != [r["turns"][0]["tool_key"] for r in other]
    openings = {record["turns"][0]["tool_key"] for record in first}
    assert len(openings) >= 4                             # scenarios do not all start with the same call


# ---------------------------------------------------------------------------
# Rule 6: no model-visible string is written in the harness code
# ---------------------------------------------------------------------------
def test_with_a_marker_bundle_every_visible_string_of_an_episode_is_a_marker(marker_text, pools, cfg):
    """In marker_text every visible string is a unique marker. So anything else in a call or a tool answer
    would be a string that the code wrote itself."""
    marker = re.compile(MARKER_PATTERN)
    generated_id = re.compile(f"{MARKER_PATTERN}[{TOKEN_ALPHABET}]{{6}}")
    first_names, last_names = pools["calib"]
    full_names = {f"{first} {last}" for first in first_names for last in last_names}

    error_text = re.compile(f"{MARKER_PATTERN} (.*)")      # the marker bundle's error texts: "marker {value}"

    def allowed(value: object) -> bool:
        if not isinstance(value, str) or marker.fullmatch(value) or generated_id.fullmatch(value):
            return True
        error = error_text.fullmatch(value)
        return value in full_names or bool(error and (error.group(1) == "" or allowed(error.group(1))))

    def strings(value: object) -> list:
        if isinstance(value, dict):
            return [s for k, v in value.items() for s in [k] + strings(v)]
        return [s for item in value for s in strings(item)] if isinstance(value, list) else [value]

    checked = 0
    chosen = {"oracle": oracle(marker_text), "stumbling": policies.stumbling_oracle(marker_text, 1),
              "claim": policies.claim_in_text(marker_text), "random": random_valid_calls(marker_text, 0),
              "made-up": policies.close_at_once(marker_text, "granted")}
    for family in make_split("calib", marker_text, pools, cfg):
        for name, policy in chosen.items():
            episode = run_policy(Episode(family, 4, marker_text, cfg, toolset="six"), policy)
            assert allowed(episode.system_prompt)
            if name in ("oracle", "stumbling"):
                assert episode.record()["passed"], (name, family["family_id"])
            for turn in episode.turns:
                for value in strings(json.loads(turn["tool_response"])):
                    assert allowed(value), (name, value)
                    checked += 1
                if turn["parse_ok"]:
                    call_text = turn["assistant_text"]
                    names = re.findall(r"<(?:function|parameter)=([^>]*)>", call_text)
                    assert names and all(marker.fullmatch(visible) for visible in names), call_text
                    parsed = parse_assistant_message(call_text, episode.tool_specs)
                    assert all(allowed(value) for value in parsed.args.values()), call_text
                    checked += len(names) + len(parsed.args)
                else:
                    assert allowed(turn["assistant_text"]) or turn["assistant_text"] == ""
    assert checked > 3000
