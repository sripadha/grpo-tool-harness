"""accessdesk/harness/episode.py, on one family written by hand (no generator, no scripted policy)."""
from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from accessdesk import impl
from accessdesk.config import RunConfig
from accessdesk.harness.episode import META_KEYS, PROMPTS, SCHEMA, Episode
from accessdesk.harness.render import render_tool_call
from tests.helpers.make_records import make_record

# Two people share a name: p1 in Finance (active) and p2 in Sales (inactive). p3 manages both.
# The request: the sensitive role r2 for p1. Under the default rules the desk asks p3 for approval.
INITIAL_ACCESS = [["p1", "r1"], ["p3", "r2"]]
NO_REQUEST = {"resolution": None, "person_id": None, "role_id": None}
EXPECTED_PENDING = {"access": INITIAL_ACCESS, "approvals": [["p1", "r2", "p3"]], "grants": [],
                    "request": {"resolution": "pending", "person_id": "p1", "role_id": "r2"}}
EXPECTED_DENIED = {"access": INITIAL_ACCESS, "approvals": [], "grants": [],
                   "request": {"resolution": "denied", "person_id": "p1", "role_id": "r2"}}
FAMILY = {
    "family_id": "hand-1", "split": "dev", "outcome": "approval", "homonym": True, "world_seed": 0,
    "world": {
        "people": [
            {"person_id": "p1", "name": "Ada Stone", "department": "Finance", "status": "active",
             "manager_id": "p3"},
            {"person_id": "p2", "name": "Ada Stone", "department": "Sales", "status": "inactive",
             "manager_id": "p3"},
            {"person_id": "p3", "name": "Bo Reed", "department": "Finance", "status": "active",
             "manager_id": None},
        ],
        "roles": [{"role_id": "r1", "name": "Wiki Editor", "sensitive": False},
                  {"role_id": "r2", "name": "Payroll Editor", "sensitive": True}],
        "access": INITIAL_ACCESS,
    },
    "request": {"person_name": "Ada Stone", "department": "Finance", "role_name": "Payroll Editor"},
    "truth": {"person_id": "p1", "role_id": "r2", "manager_id": "p3"},
    "expected": {"default": EXPECTED_PENDING, "all_grants_need_approval": EXPECTED_PENDING,
                 "sensitive_denied": EXPECTED_DENIED},
    "turn_budget": {"default": 5, "all_grants_need_approval": 5, "sensitive_denied": 4},
    "wordings": [{"style": 1, "template": 0, "text": "Ada Stone of Finance needs Payroll Editor."},
                 {"style": 4, "template": 2, "text": "Finance asks for Payroll Editor for Ada Stone."}],
}
FAMILY_AT_START = copy.deepcopy(FAMILY)

RECORD_KEYS = ["schema", "run", "arm", "split", "family_id", "style", "scenario_id", "outcome", "ruleset",
               "toolset", "prompt", "sample", "seed", "turns", "n_turns", "n_malformed", "n_tool_errors",
               "closed", "ended_by", "final_state", "passed", "reward", "turn_budget", "impl", "text_sha256",
               "text_is_placeholder", "config_sha256", "settings", "harness_error"]
TURN_KEYS = ["turn", "assistant_text", "truncated", "n_tokens", "parse_ok", "parse_error", "tool_key", "args",
             "tool_error", "tool_response"]


def call(text, tool_key: str, **args: str) -> str:
    """The assistant text of one call. `args` use internal parameter keys; values are written as given."""
    spec = text.tools[tool_key]
    return render_tool_call(spec.name, {param.name: args[param.key] for param in spec.params})


def correct_calls(text) -> list[str]:
    """The correct path for FAMILY under the default rules, typed by hand."""
    return [
        call(text, "lookup_person", name="Ada Stone"),
        call(text, "check_access", person_id="p1", role_name="Payroll Editor"),
        call(text, "request_approval", person_id="p1", role_id="r2", approver_id="p3"),
        call(text, "close_request", resolution=text.resolution["pending"], person_id="p1", role_id="r2"),
    ]


def play(episode: Episode, messages: list[str]) -> Episode:
    for message in messages:
        episode.step(message)
    return episode


@pytest.fixture
def episode(text, cfg) -> Episode:
    return Episode(FAMILY, 1, text, cfg)


@pytest.fixture(autouse=True)
def the_family_is_never_changed():
    yield
    assert FAMILY == FAMILY_AT_START


# ---------------------------------------------------------------------------
# A new episode
# ---------------------------------------------------------------------------
def test_a_new_episode_holds_the_prompt_the_request_and_the_five_tools(episode, text):
    assert episode.system_prompt == text.system_prompt
    assert episode.request_text == "Ada Stone of Finance needs Payroll Editor."
    assert episode.messages == [{"role": "system", "content": text.system_prompt},
                                {"role": "user", "content": "Ada Stone of Finance needs Payroll Editor."}]
    assert episode.tool_specs == text.tool_specs("five") and len(episode.tool_specs) == 5
    assert episode.turns == [] and episode.done is False
    assert episode.world.state() == {"access": INITIAL_ACCESS, "approvals": [], "grants": [],
                                     "request": NO_REQUEST}


def test_the_style_chooses_the_wording(text, cfg):
    assert Episode(FAMILY, 4, text, cfg).request_text == "Finance asks for Payroll Editor for Ada Stone."
    for missing in (2, 3, 0, "1"):
        with pytest.raises(ValueError, match="no wording in style"):
            Episode(FAMILY, missing, text, cfg)


def test_toolset_six_adds_the_extra_tool_last(text, cfg):
    specs = Episode(FAMILY, 1, text, cfg, toolset="six").tool_specs
    five = [spec.key for spec in text.tool_specs("five")]
    assert [spec.key for spec in specs] == five + ["department_info"] and len(specs) == 6
    with pytest.raises(ValueError, match="toolset"):
        Episode(FAMILY, 1, text, cfg, toolset="seven")


def test_each_prompt_name_gives_its_own_system_prompt(text, cfg):
    variant = text.policy_shift_variant
    shifted = Episode(FAMILY, 1, text, cfg, prompt="policy_shift", ruleset=variant)
    prompts = {"final": Episode(FAMILY, 1, text, cfg, prompt="final").system_prompt,
               "plain": Episode(FAMILY, 1, text, cfg, prompt="plain").system_prompt,
               "policy_shift": shifted.system_prompt}
    assert prompts == {"final": text.system_prompt, "plain": text.plain_system_prompt,
                       "policy_shift": text.policy_shift_system_prompt}
    assert len(set(prompts.values())) == 3 and set(prompts) == set(PROMPTS)
    for name, prompt in prompts.items():
        ruleset = variant if name == "policy_shift" else "default"
        assert Episode(FAMILY, 1, text, cfg, prompt=name, ruleset=ruleset).messages[0]["content"] == prompt


def test_the_policy_shift_prompt_is_refused_with_any_other_rule_set(text, cfg):
    assert text.policy_shift_variant == "all_grants_need_approval"
    for ruleset in ("default", "sensitive_denied"):
        with pytest.raises(ValueError, match="policy-shift prompt"):
            Episode(FAMILY, 1, text, cfg, prompt="policy_shift", ruleset=ruleset)


def test_bad_arguments_are_refused(text, cfg):
    with pytest.raises(ValueError, match="unknown rule set"):
        Episode(FAMILY, 1, text, cfg, ruleset="strict")
    with pytest.raises(ValueError, match="unknown prompt"):
        Episode(FAMILY, 1, text, cfg, prompt="system")
    without_plain = dataclasses.replace(text, plain_system_prompt="")
    with pytest.raises(ValueError, match="no plain system prompt"):
        Episode(FAMILY, 1, without_plain, cfg, prompt="plain")
    assert Episode(FAMILY, 1, without_plain, cfg).system_prompt == text.system_prompt


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------
def test_a_valid_call_is_executed_and_its_turn_record_is_complete(episode, text):
    message = call(text, "lookup_person", name="Ada Stone")
    turn = episode.step(message, n_tokens=31)
    keys = text.result_keys
    people = [{keys["person_id"]: person_id, keys["name"]: "Ada Stone", keys["department"]: department,
               keys["status"]: text.status[status], keys["manager_id"]: "p3"}
              for person_id, department, status in (("p1", "Finance", "active"), ("p2", "Sales", "inactive"))]
    assert turn == {"turn": 0, "assistant_text": message, "truncated": False, "n_tokens": 31,
                    "parse_ok": True, "parse_error": None, "tool_key": "lookup_person",
                    "args": {"name": "Ada Stone"}, "tool_error": None,
                    "tool_response": json.dumps({keys["matches"]: people}, ensure_ascii=False)}
    assert list(turn) == TURN_KEYS
    assert episode.turns == [turn] and episode.done is False
    assert episode.messages[2:] == [{"role": "assistant", "content": message},
                                    {"role": "tool", "content": turn["tool_response"]}]


def test_the_correct_path_ends_at_the_close_and_passes(episode, text):
    turns = [episode.step(message) for message in correct_calls(text)]
    assert [turn["tool_key"] for turn in turns] == ["lookup_person", "check_access", "request_approval",
                                                    "close_request"]
    assert [turn["turn"] for turn in turns] == [0, 1, 2, 3]
    assert episode.done is True and episode.world.closed is True
    assert len(episode.messages) == 2 + 2 * 4
    confirmation = {text.result_keys["result"]: text.confirmation["request_closed"]}
    assert json.loads(episode.messages[-1]["content"]) == confirmation
    record = episode.record()
    assert record["final_state"] == EXPECTED_PENDING
    assert (record["passed"], record["reward"]) == (True, 100)
    assert (record["ended_by"], record["closed"]) == ("close", True)
    assert (record["n_turns"], record["n_malformed"], record["n_tool_errors"]) == (4, 0, 0)


def test_the_saved_arguments_hold_the_internal_resolution_key(episode, text):
    assert text.resolution["pending"] != "pending"       # visible value and internal key differ
    turn = play(episode, correct_calls(text)).turns[-1]
    assert turn["args"] == {"resolution": "pending", "person_id": "p1", "role_id": "r2"}
    assert text.resolution["pending"] in turn["assistant_text"]


def test_text_before_the_call_is_allowed_and_the_message_is_kept_as_written(episode, text):
    message = "\n First I look the person up.\n" + call(text, "lookup_person", name="Ada Stone") + " \n"
    turn = episode.step(message)
    assert turn["parse_ok"] is True and turn["assistant_text"] == message       # white space included
    assert episode.messages[2] == {"role": "assistant", "content": message}


@pytest.mark.parametrize("message, code", [
    ("", "no_call"),
    ("The request is closed.", "no_call"),
    ("<tool_call>\n<function=no_such_tool>\n</function>\n</tool_call>", "unknown_tool"),
])
def test_a_malformed_message_uses_a_turn_and_runs_nothing(episode, text, message, code):
    turn = episode.step(message)
    assert turn == {"turn": 0, "assistant_text": message, "truncated": False, "n_tokens": None,
                    "parse_ok": False, "parse_error": code, "tool_key": None, "args": None,
                    "tool_error": None,
                    "tool_response": json.dumps({text.result_keys["error"]: text.errors["malformed_call"]})}
    assert (episode.n_malformed, episode.n_tool_errors, len(episode.turns)) == (1, 0, 1)
    assert episode.messages[-1] == {"role": "tool", "content": turn["tool_response"]}
    assert episode.world.state()["request"] == NO_REQUEST and episode.done is False


def test_each_kind_of_malformed_message_is_counted(episode, text):
    lookup = call(text, "lookup_person", name="Ada Stone")
    two_calls = lookup + "\n" + lookup
    close = text.tools["close_request"]
    resolution, person_id, role_id = (param.name for param in close.params)
    bad_resolution = render_tool_call(close.name, {resolution: "pending", person_id: "p1", role_id: "r2"})
    missing_argument = render_tool_call(close.name, {resolution: text.resolution["pending"]})
    messages = [two_calls, lookup + " done", bad_resolution, missing_argument]
    codes = [episode.step(message)["parse_error"] for message in messages]
    assert codes == ["multiple_calls", "text_after_call", "bad_value", "missing_argument"]
    assert episode.n_malformed == 4 and episode.world.closed is False


def test_a_truncated_turn_is_malformed_even_if_its_text_is_a_valid_call(episode, text):
    turn = episode.step(call(text, "lookup_person", name="Ada Stone"), truncated=True, n_tokens=160)
    assert (turn["parse_ok"], turn["parse_error"], turn["truncated"], turn["n_tokens"]) == \
        (False, "truncated", True, 160)
    assert episode.n_malformed == 1


def test_a_tool_error_uses_a_turn_writes_nothing_and_is_not_malformed(episode, text):
    turn = episode.step(call(text, "grant_role", person_id="p9", role_id="r2"))
    assert (turn["parse_ok"], turn["tool_error"]) == (True, "unknown_person_id")
    error = {text.result_keys["error"]: text.errors["unknown_person_id"].format(value="p9")}
    assert json.loads(turn["tool_response"]) == error
    assert (episode.n_malformed, episode.n_tool_errors, len(episode.turns)) == (0, 1, 1)
    assert episode.world.state() == {"access": INITIAL_ACCESS, "approvals": [], "grants": [],
                                     "request": NO_REQUEST}


def test_a_close_that_the_tool_refuses_does_not_end_the_episode(episode, text):
    turn = episode.step(call(text, "close_request", resolution=text.resolution["pending"], person_id="p1",
                             role_id="r9"))
    assert turn["tool_error"] == "unknown_role_id"
    assert episode.done is False and episode.world.closed is False


def test_wrong_but_valid_actions_are_executed_and_show_in_the_final_state(episode, text):
    play(episode, [call(text, "grant_role", person_id="p2", role_id="r2"),
                   call(text, "request_approval", person_id="p1", role_id="r2", approver_id="p1"),
                   call(text, "close_request", resolution=text.resolution["granted"], person_id="p2",
                        role_id="r1")])
    record = episode.record()
    assert record["final_state"] == {
        "access": [["p1", "r1"], ["p2", "r2"], ["p3", "r2"]], "approvals": [["p1", "r2", "p1"]],
        "grants": [["p2", "r2"]], "request": {"resolution": "granted", "person_id": "p2", "role_id": "r1"}}
    assert (record["passed"], record["reward"]) == (False, 0)
    assert (record["closed"], record["n_tool_errors"]) == (True, 0)


def test_the_episode_ends_at_the_turn_limit(episode, text, cfg):
    assert cfg.turn_limit == 8
    for i in range(8):
        assert episode.done is False
        episode.step(call(text, "lookup_person", name="Ada Stone"))
    assert episode.done is True
    record = episode.record()
    assert (record["n_turns"], record["ended_by"], record["closed"]) == (8, "turn_limit", False)
    assert (record["passed"], record["reward"]) == (False, 0)
    assert record["final_state"]["request"] == NO_REQUEST


def test_a_close_on_the_last_allowed_turn_still_counts(episode, text):
    reads = [call(text, "lookup_person", name="Ada Stone")] * 4
    play(episode, reads + correct_calls(text))
    record = episode.record()
    assert (record["n_turns"], record["ended_by"], record["passed"]) == (8, "close", True)
    assert record["reward"] == 100 - 2 * (8 - 5)       # 3 turns over the budget of 5


def test_work_that_would_need_a_ninth_turn_is_cut_off_and_fails(episode, text):
    reads = [call(text, "lookup_person", name="Ada Stone")] * 5
    play(episode, reads + correct_calls(text)[:3])      # the close would be turn 9
    assert episode.done is True
    record = episode.record()
    assert record["final_state"]["approvals"] == [["p1", "r2", "p3"]]      # the work was done ...
    assert (record["ended_by"], record["passed"], record["reward"]) == ("turn_limit", False, 0)   # not closed


def test_step_raises_after_the_end_and_changes_nothing(episode, text):
    play(episode, correct_calls(text))
    before = (copy.deepcopy(episode.turns), copy.deepcopy(episode.messages), episode.world.state())
    with pytest.raises(RuntimeError, match="over"):
        episode.step(call(text, "grant_role", person_id="p1", role_id="r2"))
    assert (episode.turns, episode.messages, episode.world.state()) == before


def test_the_config_sets_the_turn_limit_and_the_strictness(text):
    short = Episode(FAMILY, 1, text, RunConfig(turn_limit=2))
    play(short, ["", ""])
    assert short.done is True
    after_call = call(text, "lookup_person", name="Ada Stone") + "\nNow I wait."
    strict = Episode(FAMILY, 1, text, RunConfig(parser_strict=True)).step(after_call)
    lenient = Episode(FAMILY, 1, text, RunConfig(parser_strict=False)).step(after_call)
    assert (strict["parse_ok"], strict["parse_error"]) == (False, "text_after_call")
    assert (lenient["parse_ok"], lenient["tool_key"]) == (True, "lookup_person")


def test_two_episodes_of_one_family_share_no_state(text, cfg):
    first, second = Episode(FAMILY, 1, text, cfg), Episode(FAMILY, 1, text, cfg)
    play(first, correct_calls(text))
    assert second.turns == [] and len(second.messages) == 2 and second.done is False
    assert second.world.state()["approvals"] == [] and second.world.closed is False


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
def test_there_is_no_record_before_the_end(episode, text):
    with pytest.raises(RuntimeError, match="not done"):
        episode.record()
    play(episode, correct_calls(text)[:3])
    with pytest.raises(RuntimeError, match="not done"):
        episode.record()


def test_the_whole_record_of_the_correct_path(text, cfg, monkeypatch):
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    episode = play(Episode(FAMILY, 4, text, cfg), correct_calls(text))
    seed = {"run_seed": 7, "pass": 1, "chunk": 2, "pos": 3}
    record = episode.record(run="r1", arm="base", sample=5, seed=seed)
    assert record == {
        "schema": 1, "run": "r1", "arm": "base", "split": "dev", "family_id": "hand-1", "style": 4,
        "scenario_id": "hand-1/s4", "outcome": "approval", "ruleset": "default", "toolset": "five",
        "prompt": "final", "sample": 5, "seed": seed,
        "turns": episode.turns, "n_turns": 4, "n_malformed": 0, "n_tool_errors": 0, "closed": True,
        "ended_by": "close", "final_state": EXPECTED_PENDING, "passed": True, "reward": 100, "turn_budget": 5,
        "impl": {"verify": "main", "reward": "main"},
        "text_sha256": text.sha256, "text_is_placeholder": True, "config_sha256": cfg.sha256(),
        # 260: the token cap of accessdesk/config.py since the freeze day (it was 160 before).
        "settings": {"turn_limit": 8, "max_new_tokens": 260, "parser_strict": True, "temperature": 1.0,
                     "top_k": 0, "top_p": 1.0, "pass_reward": 100, "malformed_penalty": 5,
                     "malformed_cap": 15, "turn_penalty": 2, "turn_cap": 10,
                     "penalise_malformed_on_fail": False},
        "harness_error": None,
    }
    assert SCHEMA == 1


def test_the_record_carries_the_hash_of_its_scenario_file_only_if_it_is_given(episode, text):
    play(episode, correct_calls(text))
    data_hash = "0123456789abcdef" * 4
    with_hash = episode.record(run="r1", data_sha256=data_hash)
    assert list(with_hash) == RECORD_KEYS + ["data_sha256"]
    assert with_hash["data_sha256"] == data_hash
    without = episode.record(run="r1")
    assert list(without) == RECORD_KEYS and "data_sha256" not in without
    assert {key: with_hash[key] for key in RECORD_KEYS} == without
    assert list(episode.record(data_sha256=None)) == RECORD_KEYS          # None is "not given"


@pytest.mark.parametrize("value", ["abc", "0123456789ABCDEF" * 4, "g" * 64, "0" * 63, "0" * 65, 5,
                                   "data/scenarios/dev.jsonl", b"0" * 64, ""])
def test_a_data_hash_that_is_not_a_sha256_is_refused(episode, text, value):
    play(episode, correct_calls(text))
    with pytest.raises(TypeError, match="data_sha256"):
        episode.record(data_sha256=value)


def test_the_record_names_the_text_file_and_the_config_it_was_made_with(text):
    other_text = dataclasses.replace(text, sha256="f" * 64, is_placeholder=False)
    other_cfg = RunConfig(turn_limit=6)
    record = play(Episode(FAMILY, 1, other_text, other_cfg), correct_calls(text)).record()
    assert (record["text_sha256"], record["text_is_placeholder"]) == ("f" * 64, False)
    assert record["config_sha256"] == other_cfg.sha256() != RunConfig().sha256()


def test_the_record_has_the_keys_of_the_specification_in_order(episode, text):
    record = play(episode, correct_calls(text)).record()
    assert list(record) == RECORD_KEYS
    assert list(record) == list(make_record())           # the hand-made records of the metric tests agree
    assert all(list(turn) == TURN_KEYS for turn in record["turns"])
    assert list(record["turns"][0]) == list(make_record()["turns"][0])


def test_meta_fields_have_defaults_and_unknown_ones_are_refused(episode, text):
    play(episode, correct_calls(text))
    record = episode.record()
    assert META_KEYS == ("run", "arm", "sample", "seed")
    assert [record[key] for key in META_KEYS] == [None, None, 0, None]
    assert episode.record(arm="x")["arm"] == "x"
    with pytest.raises(TypeError, match="smaple"):
        episode.record(smaple=1)


def test_the_record_is_plain_json_and_the_same_for_the_same_messages(text, cfg):
    messages = ["", call(text, "grant_role", person_id="p9", role_id="r2")] + correct_calls(text)
    lines = [json.dumps(play(Episode(FAMILY, 1, text, cfg), messages).record(run="a"), ensure_ascii=False)
             for _ in range(2)]
    assert lines[0] == lines[1]
    assert json.loads(lines[0])["turns"][1]["tool_error"] == "unknown_person_id"


def test_the_verifier_gets_the_expected_state_of_the_episodes_rule_set(text, cfg):
    pending = correct_calls(text)
    denied = pending[:2] + [call(text, "close_request", resolution=text.resolution["denied"], person_id="p1",
                                 role_id="r2")]
    results = {(ruleset, name): play(Episode(FAMILY, 1, text, cfg, ruleset=ruleset), messages).record()
               for ruleset in ("default", "all_grants_need_approval", "sensitive_denied")
               for name, messages in (("pending", pending), ("denied", denied))}
    assert {key: record["passed"] for key, record in results.items()} == {
        ("default", "pending"): True, ("default", "denied"): False,
        ("all_grants_need_approval", "pending"): True, ("all_grants_need_approval", "denied"): False,
        ("sensitive_denied", "pending"): False, ("sensitive_denied", "denied"): True}
    assert results[("sensitive_denied", "denied")]["turn_budget"] == 4
    assert results[("sensitive_denied", "denied")]["ruleset"] == "sensitive_denied"


@pytest.mark.parametrize("n_malformed, n_tool_errors, reward", [
    (0, 0, 100),     # 4 turns, budget 5
    (1, 0, 95),      # 5 turns: one malformed message, still inside the budget
    (2, 0, 88),      # 6 turns: 2 * 5 for the malformed messages, 2 for one turn over the budget
    (3, 0, 81),      # 7 turns: 15 (the cap) and 4
    (4, 0, 79),      # 8 turns: 15 (the cap) and 6
    (0, 1, 100),     # a tool error costs its turn only: 5 turns, inside the budget
    (0, 2, 98),      # 6 turns: one over the budget
    (1, 1, 93),      # 6 turns: 5 and 2
])
def test_the_reward_counts_malformed_messages_and_turns_over_the_budget(text, cfg, n_malformed, n_tool_errors,
                                                                        reward):
    tool_error = call(text, "check_access", person_id="p9", role_name="Payroll Editor")
    episode = Episode(FAMILY, 1, text, cfg)
    play(episode, [""] * n_malformed + [tool_error] * n_tool_errors + correct_calls(text))
    record = episode.record()
    assert (record["n_malformed"], record["n_tool_errors"], record["n_turns"]) == \
        (n_malformed, n_tool_errors, 4 + n_malformed + n_tool_errors)
    assert (record["passed"], record["reward"]) == (True, reward)


@pytest.mark.parametrize("n_malformed, with_the_switch_on", [(0, 0), (1, -5), (2, -10), (3, -15), (5, -15)])
def test_a_failed_episode_scores_zero_or_with_the_switch_on_minus_its_malformed_penalty(
        text, n_malformed, with_the_switch_on):
    # The episode closes the request with the wrong resolution after `n_malformed` empty messages.
    wrong_close = call(text, "close_request", resolution=text.resolution["denied"], person_id="p1",
                       role_id="r2")
    for switch, wanted in ((False, 0), (True, with_the_switch_on)):
        episode = Episode(FAMILY, 1, text, RunConfig(penalise_malformed_on_fail=switch))
        record = play(episode, [""] * n_malformed + [wrong_close]).record()
        assert (record["passed"], record["closed"], record["n_malformed"]) == (False, True, n_malformed)
        assert record["reward"] == wanted and type(record["reward"]) is int


def test_an_episode_of_malformed_messages_only_scores_zero_or_minus_the_cap(text):
    for switch, wanted in ((False, 0), (True, -15)):
        episode = Episode(FAMILY, 1, text, RunConfig(penalise_malformed_on_fail=switch))
        record = play(episode, [""] * 8).record()
        assert (record["passed"], record["ended_by"], record["reward"]) == (False, "turn_limit", wanted)


def test_the_switch_does_not_change_the_reward_of_a_pass(text):
    for switch in (False, True):
        episode = Episode(FAMILY, 1, text, RunConfig(penalise_malformed_on_fail=switch))
        record = play(episode, [""] + correct_calls(text)).record()
        assert (record["passed"], record["reward"]) == (True, 95)


def test_verify_and_reward_are_reached_through_impl(episode, text, cfg, monkeypatch):
    play(episode, [""] + correct_calls(text))
    calls = []

    def fake_get(name):
        def verify(final_state, expected_state):
            calls.append(("verify", final_state, expected_state))
            return False

        def reward(passed, n_malformed, n_turns, turn_budget, config):
            calls.append(("reward", passed, n_malformed, n_turns, turn_budget, config))
            return 42

        return {"verify": verify, "reward": reward}[name]

    monkeypatch.setattr(impl, "get", fake_get)
    monkeypatch.setattr(impl, "which", lambda name: "second")
    record = episode.record()
    assert (record["passed"], record["reward"]) == (False, 42)
    assert record["impl"] == {"verify": "second", "reward": "second"}
    assert calls == [("verify", EXPECTED_PENDING, FAMILY["expected"]["default"]),
                     ("reward", False, 1, 5, 5, cfg)]


@pytest.mark.parametrize("verdict", [1, 0, None, "yes"])
def test_a_verifier_that_does_not_return_a_bool_is_refused(episode, text, monkeypatch, verdict):
    play(episode, correct_calls(text))
    monkeypatch.setattr(impl, "get", lambda name: (lambda *args: verdict))
    with pytest.raises(TypeError, match="verify"):
        episode.record()


@pytest.mark.parametrize("value", [100.0, True, -1, -15, 101, None])
def test_a_reward_that_is_not_an_int_from_0_to_100_is_refused(episode, text, monkeypatch, value):
    play(episode, correct_calls(text))
    functions = {"verify": lambda *args: True, "reward": lambda *args: value}
    monkeypatch.setattr(impl, "get", lambda name: functions[name])
    with pytest.raises(TypeError, match="reward"):
        episode.record()


@pytest.mark.parametrize("value, accepted", [(0, True), (-1, True), (-15, True), (100, True), (-16, False),
                                             (101, False), (-15.0, False), (False, False)])
def test_with_the_switch_on_a_reward_down_to_minus_the_cap_is_accepted(text, monkeypatch, value, accepted):
    episode = Episode(FAMILY, 1, text, RunConfig(penalise_malformed_on_fail=True))
    play(episode, [""] * 8)
    functions = {"verify": lambda *args: False, "reward": lambda *args: value}
    monkeypatch.setattr(impl, "get", lambda name: functions[name])
    if accepted:
        assert episode.record()["reward"] == value
    else:
        with pytest.raises(TypeError, match="reward"):
            episode.record()


def test_the_lowest_accepted_reward_follows_the_cap_of_the_config(text, monkeypatch):
    episode = Episode(FAMILY, 1, text, RunConfig(penalise_malformed_on_fail=True, malformed_cap=7))
    play(episode, [""] * 8)
    functions = {"verify": lambda *args: False, "reward": lambda *args: -8}
    monkeypatch.setattr(impl, "get", lambda name: functions[name])
    with pytest.raises(TypeError, match="from -7 to 100"):
        episode.record()


# ---------------------------------------------------------------------------
# The config in the record (findings of 2026-11-17)
# ---------------------------------------------------------------------------
def test_the_config_hash_of_a_record_is_the_hash_of_its_own_config_whatever_was_hashed_before(text):
    """Two configs that compare equal can have different hashes: JSON writes 1 and 1.0 differently. A hash
    kept per config (compared with ==) gave the second one the hash of the first, so a record could
    disagree with its run manifest."""
    typed = RunConfig(temperature=1, top_p=1)             # whole numbers where the default has 1.0
    default = RunConfig()
    assert typed == default and typed.sha256() != default.sha256()
    for order in ((typed, default), (default, typed)):
        for cfg in order:
            record = play(Episode(FAMILY, 1, text, cfg), correct_calls(text)).record()
            assert record["config_sha256"] == cfg.sha256()


def test_the_record_carries_the_settings_that_every_arm_of_a_table_must_share(text):
    cfg = RunConfig(turn_limit=6, max_new_tokens=96, parser_strict=False, temperature=0.7, top_k=20,
                    top_p=0.9, chunk_size=7, malformed_penalty=4, data_seed=99)
    record = play(Episode(FAMILY, 1, text, cfg), correct_calls(text)).record()
    assert record["settings"] == {
        "turn_limit": 6, "max_new_tokens": 96, "parser_strict": False, "temperature": 0.7, "top_k": 20,
        "top_p": 0.9, "pass_reward": 100, "malformed_penalty": 4, "malformed_cap": 15, "turn_penalty": 2,
        "turn_cap": 10, "penalise_malformed_on_fail": False}
    assert record["settings"] == cfg.compared_settings()
    assert json.loads(json.dumps(record["settings"])) == record["settings"]
