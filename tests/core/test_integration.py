"""Cross-module tests: the pieces of the repository used together, the way a real run uses them.

Every other test file checks one module, often with hand-made inputs. Here real episodes (the Episode class,
the scripted policies, the real parser and tools, the generated families) are followed through the rest of the
code:

    record files and replay            the saved calls give the saved final state again
    the record format                  the keys of the build specification, typed by hand here
    metrics and the results table      1.0 for the correct policy, 0.25 for a constant one, by family
    the token buffer and the loss mask the buffer of a real record is the chat template's own render
    the rollout loop of the GPU code   with a scripted stand-in for the model (no torch is needed)
    the text bundle                    with a bundle of markers, a whole conversation holds only markers

The tests that need the tokenizer files carry the marker `tokenizer` and are skipped without them.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import re
import unicodedata

import pytest

from accessdesk import impl
from accessdesk.config import RunConfig
from accessdesk.env.generator import TRAIN_STYLES, make_all_splits, make_split
from accessdesk.eval import metrics, tables
from accessdesk.harness import policies
from accessdesk.harness.episode import Episode
from accessdesk.harness.hf_backend import SampledTurn, rollout
from accessdesk.harness.policies import named_policies, oracle, run_policy
from accessdesk.harness.records import read_records, replay, write_records
from accessdesk.harness.tokens import TokenLayout, buffer_from_record
from accessdesk.rng import TOKEN_ALPHABET, Rng
from accessdesk.text import STYLES, check_text, load_text, model_text_path, placeholder_path
from accessdesk.train.forward import loss_positions
from accessdesk.types import OUTCOMES, RESOLUTIONS, Segment
from tests.conftest import MARKER_PATTERN

# The policies whose dev episodes the tests below share. The last three are shortcuts of three kinds: calls
# that the tools refuse, a write too many, and messages that are no tool call.
SHORTCUTS = ("random_valid_calls", "extra_grant_second_role", "claim_in_text")
CONSTANTS = {"constant_granted": "grant", "constant_pending": "approval", "constant_already": "already",
             "constant_denied": "inactive"}                   # policy name -> the one outcome it is right for
PLAYED = ("oracle", *CONSTANTS, *SHORTCUTS)

# The record format of BUILD_SPEC section 8, typed by hand (not taken from the code).
RECORD_KEYS = ["schema", "run", "arm", "split", "family_id", "style", "scenario_id", "outcome", "ruleset",
               "toolset", "prompt", "sample", "seed", "turns", "n_turns", "n_malformed", "n_tool_errors",
               "closed", "ended_by", "final_state", "passed", "reward", "turn_budget", "impl", "text_sha256",
               "text_is_placeholder", "config_sha256", "settings", "harness_error"]
TURN_KEYS = ["turn", "assistant_text", "truncated", "n_tokens", "parse_ok", "parse_error", "tool_key", "args",
             "tool_error", "tool_response"]
STATE_KEYS = ["access", "approvals", "grants", "request"]
ARGS_OF_TOOL = {"lookup_person": ["name"], "check_access": ["person_id", "role_name"],
                "request_approval": ["person_id", "role_id", "approver_id"],
                "grant_role": ["person_id", "role_id"],
                "close_request": ["resolution", "person_id", "role_id"], "department_info": ["department"]}


def play_split(families: list[dict], policy, name: str, text, cfg) -> list[dict]:
    """One record per scenario (every family in every one of its wordings), played with a scripted policy."""
    records = []
    for family in families:
        for wording in family["wordings"]:
            episode = run_policy(Episode(family, wording["style"], text, cfg), policy)
            seed = {"run_seed": 0, "pass": 0, "chunk": 0, "pos": len(records)}
            records.append(episode.record(run="integration", arm=name, sample=0, seed=seed))
    return records


def first_of_each_outcome(families: list[dict]) -> dict[str, dict]:
    return {outcome: next(f for f in families if f["outcome"] == outcome) for outcome in OUTCOMES}


@pytest.fixture(scope="module")
def dev_records(text, cfg, families) -> dict[str, list[dict]]:
    """policy name -> its 192 dev records (48 families in 4 wordings), for the policies in PLAYED."""
    named = named_policies(text, seed=0)
    return {name: play_split(families["dev"], named[name], name, text, cfg) for name in PLAYED}


@pytest.fixture(scope="module")
def dev_by_id(families) -> dict[str, dict]:
    return {family["family_id"]: family for family in families["dev"]}


# ---------------------------------------------------------------------------
# The record format, on real records
# ---------------------------------------------------------------------------
def test_the_shared_episodes_are_of_the_kinds_that_the_tests_need(dev_records):
    assert all(len(records) == 192 for records in dev_records.values())
    assert all(r["passed"] and r["reward"] == 100 for r in dev_records["oracle"])
    refused = dev_records["random_valid_calls"]
    assert sum(r["n_tool_errors"] for r in refused) > 500         # calls that a tool answered with an error
    assert any(r["ended_by"] == "turn_limit" for r in refused) and any(r["closed"] for r in refused)
    too_much = dev_records["extra_grant_second_role"]
    assert not any(r["passed"] for r in too_much) and sum(r["closed"] for r in too_much) > 100
    assert sum(len(r["final_state"]["grants"]) for r in too_much) > 150
    talk = dev_records["claim_in_text"]
    assert all(not r["closed"] and r["n_malformed"] >= 4 and r["reward"] == 0 for r in talk)


@pytest.mark.parametrize("name", PLAYED)
def test_real_records_have_the_format_of_the_specification(name, dev_records, text, cfg):
    for record in dev_records[name]:
        assert list(record) == RECORD_KEYS
        assert record["schema"] == 1 and record["split"] == "dev"
        assert (record["run"], record["arm"], record["sample"]) == ("integration", name, 0)
        assert list(record["seed"]) == ["run_seed", "pass", "chunk", "pos"]
        assert record["scenario_id"] == f"{record['family_id']}/s{record['style']}"
        assert (record["ruleset"], record["toolset"], record["prompt"]) == ("default", "five", "final")
        assert record["n_turns"] == len(record["turns"]) <= cfg.turn_limit
        assert record["n_malformed"] == sum(not turn["parse_ok"] for turn in record["turns"])
        assert record["n_tool_errors"] == sum(turn["tool_error"] is not None for turn in record["turns"])
        assert record["ended_by"] == ("close" if record["closed"] else "turn_limit")
        assert record["closed"] or record["n_turns"] == cfg.turn_limit
        assert list(record["final_state"]) == STATE_KEYS
        assert list(record["final_state"]["request"]) == ["resolution", "person_id", "role_id"]
        assert type(record["passed"]) is bool and type(record["reward"]) is int
        # The default config: a fail is exactly 0 (the switch for penalties on failing episodes is off).
        assert (record["reward"] > 0) == record["passed"] and 0 <= record["reward"] <= 100
        assert not record["passed"] or record["closed"]
        assert list(record["impl"]) == ["verify", "reward"]
        assert set(record["impl"].values()) == {impl.mode()}
        assert (record["text_sha256"], record["text_is_placeholder"]) == (text.sha256, text.is_placeholder)
        assert record["config_sha256"] == cfg.sha256() and record["harness_error"] is None
        for index, turn in enumerate(record["turns"]):
            assert list(turn) == TURN_KEYS and turn["turn"] == index
            assert (turn["truncated"], turn["n_tokens"]) == (False, None)       # no sampler: nothing to count
            if turn["parse_ok"]:
                assert turn["parse_error"] is None and list(turn["args"]) == ARGS_OF_TOOL[turn["tool_key"]]
            else:
                assert turn["parse_error"] is not None
                assert (turn["tool_key"], turn["args"], turn["tool_error"]) == (None, None, None)
            assert isinstance(json.loads(turn["tool_response"]), dict)


def test_a_saved_close_holds_the_internal_resolution_key_and_the_message_the_visible_value(dev_records, text):
    """The specification does not say which of the two `args` holds. The code saves the call as it was run."""
    closes = [turn for record in dev_records["oracle"] for turn in record["turns"]
              if turn["tool_key"] == "close_request"]
    assert len(closes) == 192
    for turn in closes:
        assert turn["args"]["resolution"] in RESOLUTIONS
        assert f"\n{text.resolution[turn['args']['resolution']]}\n" in turn["assistant_text"]
    pending = [turn for turn in closes if turn["args"]["resolution"] == "pending"]
    assert len(pending) == 48
    if text.resolution["pending"] != "pending":           # only then can the two be told apart
        assert all("\npending\n" not in turn["assistant_text"] for turn in pending)


# ---------------------------------------------------------------------------
# Record files and replay
# ---------------------------------------------------------------------------
def test_real_records_come_back_from_disk_as_they_were(dev_records, tmp_path):
    records = [record for name in PLAYED for record in dev_records[name]]
    path = tmp_path / "episodes.jsonl"
    write_records(path, records)
    saved = read_records(path)
    assert saved == records                                     # a tuple anywhere would come back as a list
    assert [list(r) for r in saved] == [list(r) for r in records]
    assert [[list(t) for t in r["turns"]] for r in saved] == [[list(t) for t in r["turns"]] for r in records]
    write_records(tmp_path / "again.jsonl", saved)
    assert (tmp_path / "again.jsonl").read_bytes() == path.read_bytes()
    assert len(path.read_bytes().splitlines()) == len(records) == 192 * len(PLAYED)


def test_both_readers_of_a_record_file_survive_odd_characters_in_a_message(text, cfg, families, tmp_path):
    """A model can write any character. Line breaks of every kind inside a message must not cut a record in
    two, for the reader of the harness and for the one of the results table alike."""
    odd = ["line separator \u2028 and paragraph separator \u2029", "next line \x85", "cr \r lf \n crlf \r\n",
           "nul \x00 and bell \x07", "\ufeffbyte order mark", 'tab \t quote " backslash \\ <|im_end|>',
           "accent e\u0301 and a face \U0001F600", ""]
    episode = Episode(families["dev"][0], 1, text, cfg)
    for message in odd:
        episode.step(message)
    record = episode.record(run="integration", arm="odd")
    assert record["n_malformed"] == 8 and [turn["assistant_text"] for turn in record["turns"]] == odd
    path = tmp_path / "episodes.jsonl"
    write_records(path, [record, record])
    assert path.read_bytes().count(b"\n") == 2
    assert read_records(path) == [record, record]
    assert tables._read_jsonl(str(path)) == [record, record]


@pytest.mark.parametrize("name", ("oracle", *SHORTCUTS))
def test_saved_records_replay_to_their_saved_final_state(name, dev_records, dev_by_id, text, cfg, tmp_path):
    path = tmp_path / "episodes.jsonl"
    write_records(path, dev_records[name])
    saved = read_records(path)
    assert len(saved) == 192
    for record in saved:
        family = dev_by_id[record["family_id"]]
        state = replay(record, family, text, cfg)
        assert state == record["final_state"], record["scenario_id"]
        # The saved verdict follows from the replayed state alone.
        assert impl.main_version("verify")(state, family["expected"]["default"]) is record["passed"]


def test_a_record_that_was_changed_does_not_replay(dev_records, dev_by_id, text, cfg):
    record = next(r for r in dev_records["oracle"] if r["outcome"] == "grant")
    family = dev_by_id[record["family_id"]]
    assert [turn["tool_key"] for turn in record["turns"]] == ["lookup_person", "check_access", "grant_role",
                                                              "close_request"]
    without_grant = copy.deepcopy(record)
    del without_grant["turns"][2]
    assert replay(without_grant, family, text, cfg) != record["final_state"]
    other_answer = copy.deepcopy(record)
    other_answer["turns"][0]["tool_response"] = "{}"
    with pytest.raises(ValueError):
        replay(other_answer, family, text, cfg)
    another_family = next(f for f in dev_by_id.values() if f["family_id"] != record["family_id"])
    with pytest.raises(ValueError):
        replay(record, another_family, text, cfg)


def test_a_second_episode_on_a_family_starts_from_the_same_world(text, cfg, families):
    family = first_of_each_outcome(families["dev"])["grant"]
    before = json.dumps(family, ensure_ascii=False)
    policy = named_policies(text)["extra_grant_same_role"]               # it writes two grants, then closes
    first = Episode(family, 1, text, cfg)
    start = first.world.state()
    run_policy(first, policy)
    assert len(first.world.state()["grants"]) == 2 and first.world.state() != start
    second = Episode(family, 1, text, cfg)
    assert second.world.state() == start and second.turns == [] and not second.done
    assert second.messages == first.messages[:2]
    assert run_policy(second, policy).record() == first.record()
    assert json.dumps(family, ensure_ascii=False) == before


NEEDED_TO_PLAY = ("family_id", "world", "wordings")


def test_an_episode_is_played_without_the_answer_fields_of_the_family(text, cfg, families):
    """step() and the policies must not read the truth, the expected states, the outcome or the request
    fields. So a family record without them plays exactly the same; only record() needs them."""
    named = named_policies(text)
    chosen = list(first_of_each_outcome(families["dev"]).values())
    chosen.append(next(f for f in families["dev"] if f["homonym"]))
    for family in chosen:
        bare = {key: copy.deepcopy(family[key]) for key in NEEDED_TO_PLAY}
        for name in ("oracle", "random_valid_calls", "wrong_person_homonym"):
            full = run_policy(Episode(family, 4, text, cfg), named[name])
            blind = run_policy(Episode(bare, 4, text, cfg), named[name])
            assert blind.messages == full.messages and blind.turns == full.turns
            assert blind.world.state() == full.world.state()
            with pytest.raises(KeyError):
                blind.record()


def test_the_record_names_the_version_that_was_used(monkeypatch, text, cfg, families):
    episode = run_policy(Episode(families["dev"][0], 1, text, cfg), oracle(text))
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    assert episode.record()["impl"] == {"verify": "main", "reward": "main"}
    monkeypatch.setenv("ACCESSDESK_IMPL", "second")
    if impl.second_is_written("verify") and impl.second_is_written("reward"):
        record = episode.record()
        assert record["impl"] == {"verify": "second", "reward": "second"}
        assert (record["passed"], record["reward"]) == (True, 100)
    else:
        # Mode "second" never falls back to the main version: a second version that is not written stops
        # the run at its first use.
        with pytest.raises(NotImplementedError, match="the second version is not written yet"):
            episode.record()


@pytest.mark.parametrize("switch", [False, True], ids=["switch off", "switch on"])
def test_real_fails_score_zero_or_with_the_switch_on_minus_their_malformed_penalty(switch, text, families):
    """Whole episodes under a config with the switch for penalties on failing episodes: the record is
    accepted, a pass keeps its reward, and a fail loses 5 per malformed message, at most 15."""
    cfg = RunConfig(penalise_malformed_on_fail=switch)
    named = named_policies(text, seed=0)
    by_policy = {name: play_split(families["dev"][:16], named[name], name, text, cfg)
                 for name in ("oracle", "claim_in_text", "random_valid_calls", "constant_granted")}
    records = [record for name in by_policy for record in by_policy[name]]
    fails = [record for record in records if not record["passed"]]
    passes = [record for record in records if record["passed"]]
    assert len(passes) >= 64 and len(fails) >= 100
    assert all(record["reward"] == 100 for record in passes)
    for record in fails:
        wanted = -min(15, 5 * record["n_malformed"]) if switch else 0
        assert record["reward"] == wanted and type(record["reward"]) is int
    assert any(record["n_malformed"] == 0 for record in fails)          # fails that stay at 0 with the switch
    # The metrics take such records: a reward below 0 is a fail like any other.
    talk = by_policy["claim_in_text"]                                   # 8 messages, none of them a call
    assert metrics.pass_rate(talk) == 0.0
    assert metrics.reward_distribution(talk) == {(-15 if switch else 0): 64}
    assert metrics.zero_variance_share(talk + [{**record, "sample": 1} for record in talk]) == {
        "n_scenarios": 64, "equal_reward": 1.0, "equal_reward_all_pass": 0.0, "equal_reward_all_fail": 1.0,
        "equal_passfail": 1.0}


# ---------------------------------------------------------------------------
# Metrics and the results table, on real records
# ---------------------------------------------------------------------------
def test_the_oracle_is_at_1_and_each_constant_policy_at_one_quarter(dev_records):
    assert metrics.pass_rate(dev_records["oracle"]) == 1.0
    assert metrics.pass_rate_by(dev_records["oracle"], "outcome") == {outcome: 1.0 for outcome in OUTCOMES}
    assert metrics.reward_distribution(dev_records["oracle"]) == {100: 192}
    for name, right_for in CONSTANTS.items():
        records = dev_records[name]
        assert metrics.pass_rate(records) == 0.25
        by_outcome = metrics.pass_rate_by(records, "outcome")
        assert by_outcome == {outcome: float(outcome == right_for) for outcome in OUTCOMES}
        rates = metrics.family_pass_rates(records)
        assert len(rates) == 48 and sorted(rates.values()) == [0.0] * 36 + [1.0] * 12
        assert metrics.pass_rate_by(records, "seen_held_out") == {"seen": 0.25, "held_out": 0.25}
        assert metrics.reward_distribution(records) == {0: 144, 100: 48}
        assert metrics.outcome_labels(records) == {"closed_correctly": 0.25, "closed_wrongly": 0.75,
                                                   "never_closed": 0.0, "malformed_only": 0.0}
    assert metrics.pass_rate(dev_records["claim_in_text"]) == 0.0
    assert metrics.outcome_labels(dev_records["claim_in_text"])["never_closed"] == 1.0


def test_the_family_is_the_unit_of_the_pass_rate(dev_records):
    """Keep one wording of every family that passes and all four of every family that fails. A rate by
    episode then moves to 12 of 156; the rate by family stays at 12 of 48."""
    records = dev_records["constant_granted"]
    uneven = [r for r in records if not r["passed"] or r["style"] == 1]
    assert len(uneven) == 12 + 36 * 4
    assert sum(r["passed"] for r in uneven) / len(uneven) == 12 / 156
    assert metrics.pass_rate(uneven) == 0.25
    assert metrics.pass_rate(list(reversed(uneven))) == 0.25             # the order of the records is nothing


def test_a_harness_error_is_counted_and_left_out_of_every_rate(dev_records):
    records = copy.deepcopy(dev_records["constant_granted"])
    failed = next(r for r in records if not r["passed"])
    passed = next(r for r in records if r["passed"])
    failed["harness_error"] = passed["harness_error"] = "made-up failure"
    assert metrics.harness_error_count(records) == 2 and len(metrics.valid_records(records)) == 190
    assert metrics.pass_rate(records) == 0.25                 # each family still has three valid wordings
    lost_family = [r for r in records if r["family_id"] != passed["family_id"]]
    assert metrics.pass_rate(lost_family) == 11 / 47


def test_the_results_table_from_record_files(dev_records, tmp_path):
    for name in ("constant_granted", "oracle"):
        write_records(tmp_path / f"{name}.jsonl", dev_records[name])
    tables.main(["--arm", f"constant={tmp_path / 'constant_granted.jsonl'}",
                 "--arm", f"oracle={tmp_path / 'oracle.jsonl'}", "--baseline", "constant",
                 "--out-json", str(tmp_path / "table.json"), "--out-md", str(tmp_path / "table.md"),
                 "--draws", "2000"])
    table = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert (table["n_families"], table["n_scenarios"], table["n_episodes_per_arm"]) == (48, 192, 192)
    constant, correct = table["arms"]
    assert (constant["arm"], constant["pass_rate"], constant["n_harness_errors"]) == ("constant", 0.25, 0)
    assert (correct["arm"], correct["pass_rate"], correct["pass_rate_lo"], correct["pass_rate_hi"]) == \
        ("oracle", 1.0, 1.0, 1.0)
    assert constant["pass_rate_lo"] < 0.25 < constant["pass_rate_hi"]
    assert constant["pass_rate_by_outcome"] == {"grant": 1.0, "approval": 0.0, "already": 0.0,
                                                "inactive": 0.0}
    (pair,) = table["pairs"]
    assert (pair["a"], pair["b"], pair["n_families"], pair["diff"]) == ("constant", "oracle", 48, 0.75)
    assert 0.5 < pair["lo"] < 0.75 < pair["hi"] < 1.0 and 0.5 < pair["t_lo"] < 0.75 < pair["t_hi"] < 1.0
    assert "| oracle minus constant | 48 | +75.0 |" in (tmp_path / "table.md").read_text(encoding="utf-8")


def test_the_table_refuses_arms_that_were_not_run_on_the_same_scenarios(dev_records):
    arms = {"constant": dev_records["constant_granted"], "oracle": dev_records["oracle"]}
    assert tables.compare_arms(arms, "constant", n_draws=100)["pairs"][0]["diff"] == 0.75
    with pytest.raises(ValueError, match="same scenarios"):
        tables.compare_arms({**arms, "oracle": arms["oracle"][:-1]}, "constant", n_draws=100)


def test_the_table_refuses_arms_that_were_made_with_different_text_files(dev_records):
    """Scenario files built from two text files have the same family ids, so the coverage check alone
    would let a comparison of two different environments through."""
    other_text = [{**record, "text_sha256": "0" * 64} for record in dev_records["oracle"]]
    arms = {"constant": dev_records["constant_granted"], "oracle": other_text}
    with pytest.raises(ValueError, match="same text file"):
        tables.compare_arms(arms, "constant", n_draws=100)
    one_stray = dev_records["oracle"][:-1] + other_text[-1:]
    with pytest.raises(ValueError, match="same text file"):
        tables.compare_arms({**arms, "oracle": one_stray}, "constant", n_draws=100)


# ---------------------------------------------------------------------------
# Tokens: the buffer of a real record against the model's own chat template
# ---------------------------------------------------------------------------
def visible_call(turn: dict, text) -> dict:
    """The saved call of a turn as the chat template takes it: visible tool name and visible arguments."""
    spec = text.tools[turn["tool_key"]]
    arguments = {param.name: text.resolution[turn["args"][param.key]] if param.key == "resolution"
                 else turn["args"][param.key] for param in spec.params}
    return {"type": "function", "function": {"name": spec.name, "arguments": arguments}}


def template_ids(tokenizer, record: dict, system_prompt: str, request_text: str, text) -> list[int]:
    """The ids of the chat template's render of a whole episode, with the calls given as structured data
    (not as the assistant's text). The tool result of the last turn is not part of it."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": request_text}]
    for turn in record["turns"]:
        messages.append({"role": "assistant", "content": "", "tool_calls": [visible_call(turn, text)]})
        messages.append({"role": "tool", "content": turn["tool_response"]})
    schemas = [spec.schema() for spec in text.tool_specs(record["toolset"])]
    encoded = tokenizer.apply_chat_template(messages[:-1], tools=schemas, add_generation_prompt=False,
                                            enable_thinking=False, tokenize=True, return_dict=True)
    return list(encoded["input_ids"])


def oracle_record(family: dict, style: int, text, cfg, toolset: str) -> tuple[dict, Episode]:
    """The correct episode of a scenario; with six tools it first calls the extra tool once."""
    policy = oracle(text) if toolset == "five" else policies.extra_tool_then_oracle(text)
    episode = run_policy(Episode(family, style, text, cfg, toolset=toolset), policy)
    return episode.record(), episode


@pytest.mark.tokenizer
@pytest.mark.parametrize("toolset", ("five", "six"))
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_the_buffer_of_a_real_oracle_record_is_the_chat_templates_render(tokenizer, text, cfg, families,
                                                                          outcome, toolset):
    family = first_of_each_outcome(families["dev"])[outcome]
    layout = TokenLayout(tokenizer, text.tool_specs(toolset))
    line_break = tokenizer.encode("\n", add_special_tokens=False)   # the template ends with one more "\n"
    for style in (1, 4):
        record, episode = oracle_record(family, style, text, cfg, toolset)
        assert record["passed"] and record["toolset"] == toolset
        assert (record["turns"][0]["tool_key"] == "department_info") == (toolset == "six")
        buffer = buffer_from_record(layout, record, family, text)
        wanted = template_ids(tokenizer, record, episode.system_prompt, episode.request_text, text)
        assert buffer.ids + line_break == wanted
        # The buffer needs the family for its wording only.
        bare = {"family_id": family["family_id"], "wordings": family["wordings"]}
        assert buffer_from_record(layout, record, bare, text).ids == buffer.ids
        # The comparison can fail: another tool result, or another wording, gives other ids.
        changed = copy.deepcopy(record)
        changed["turns"][0]["tool_response"] = changed["turns"][1]["tool_response"]
        assert buffer_from_record(layout, changed, family, text).ids != buffer.ids
        assert buffer_from_record(layout, {**record, "style": 2}, family, text).ids != buffer.ids


@pytest.mark.tokenizer
@pytest.mark.parametrize("toolset", ("five", "six"))
def test_the_loss_mask_of_a_real_record_covers_the_assistants_messages_and_nothing_else(tokenizer, text, cfg,
                                                                                        families, toolset):
    layout = TokenLayout(tokenizer, text.tool_specs(toolset))
    end_of_turn = tokenizer.decode([layout.im_end_id])
    for family in first_of_each_outcome(families["dev"]).values():
        record, _ = oracle_record(family, 2, text, cfg, toolset)
        buffer = buffer_from_record(layout, record, family, text)
        mask = impl.get("build_loss_mask")(len(buffer.ids), buffer.segments)
        assert len(mask) == len(buffer.ids)
        sampled = [segment for segment in buffer.segments if segment.kind == "sampled"]
        assert [segment.turn for segment in sampled] == list(range(record["n_turns"]))
        for segment, turn in zip(sampled, record["turns"]):
            written = tokenizer.decode(buffer.ids[segment.start:segment.end], skip_special_tokens=False)
            assert written == turn["assistant_text"] + end_of_turn
            assert mask[segment.start:segment.end] == [1] * (segment.end - segment.start)
        assert sum(mask) == sum(segment.end - segment.start for segment in sampled)
        with_weight_0 = tokenizer.decode([i for i, weight in zip(buffer.ids, mask) if weight == 0],
                                         skip_special_tokens=False)
        assert text.system_prompt in with_weight_0 and record["turns"][0]["tool_response"] in with_weight_0
        assert all(turn["assistant_text"] not in with_weight_0 for turn in record["turns"])


@pytest.mark.tokenizer
def test_the_training_positions_of_two_real_episodes_point_at_the_assistants_tokens(tokenizer, text, cfg,
                                                                                    families):
    """accessdesk.train.forward.loss_positions, fed with real buffers and their masks: in the left-padded
    batch, the column after each returned position holds the target, and the targets of an episode are the
    assistant's messages. (The arithmetic alone; no torch and no model.)"""
    layout = TokenLayout(tokenizer, text.tool_specs("five"))
    by_outcome = first_of_each_outcome(families["dev"])
    pairs = [(oracle_record(by_outcome[outcome], 3, text, cfg, "five")[0], by_outcome[outcome])
             for outcome in ("grant", "inactive")]                   # 4 turns and 3 turns: unequal lengths
    buffers = [buffer_from_record(layout, record, family, text) for record, family in pairs]
    seqs = [buffer.ids for buffer in buffers]
    weights = [impl.get("build_loss_mask")(len(buffer.ids), buffer.segments) for buffer in buffers]
    assert len(seqs[0]) != len(seqs[1])
    rows, columns, targets, counts = loss_positions(seqs, weights)
    width = max(len(seq) for seq in seqs)
    padded = [[layout.pad_id] * (width - len(seq)) + seq for seq in seqs]
    assert counts == [sum(w) for w in weights] and len(targets) == sum(counts)
    assert all(padded[row][column + 1] == target for row, column, target in zip(rows, columns, targets))
    end_of_turn = tokenizer.decode([layout.im_end_id])
    start = 0
    for (record, _), count in zip(pairs, counts):
        trained_on = tokenizer.decode(targets[start:start + count], skip_special_tokens=False)
        assert trained_on == "".join(turn["assistant_text"] + end_of_turn for turn in record["turns"])
        start += count


# ---------------------------------------------------------------------------
# The rollout loop of the GPU backend, with a scripted stand-in for the model
# ---------------------------------------------------------------------------
class ScriptedSampler:
    """Stands in for the model in hf_backend.rollout: it answers a prompt with the next message of a script,
    as token ids. A prompt is recognised by its first prompt (the system turn, the tools, the request). The
    turn number is read from the prompt: every finished turn has added two end-of-turn tokens, the
    assistant's and the tool result's."""

    def __init__(self, tokenizer, cfg, layout: TokenLayout) -> None:
        self.tokenizer, self.cfg = tokenizer, cfg
        self.stop_id, self.pad_id = layout.im_end_id, layout.pad_id
        self.scripts: dict[tuple[int, ...], list[str]] = {}

    def add_script(self, first_prompt: list[int], texts: list[str]) -> None:
        self.scripts[tuple(first_prompt)] = texts

    def sample(self, prompts: list[list[int]], seed: int) -> list[SampledTurn]:
        return [self._turn(prompt) for prompt in prompts]

    def _turn(self, prompt: list[int]) -> SampledTurn:
        (first,) = [key for key in self.scripts if tuple(prompt[:len(key)]) == key]
        turn = prompt[len(first):].count(self.stop_id) // 2
        ids = self.tokenizer.encode(self.scripts[first][turn], add_special_tokens=False) + [self.stop_id]
        return SampledTurn(ids=ids, logprobs=[-1.0] * len(ids), truncated=False)


def scripted_rollout(tokenizer, text, cfg, scenarios: list[tuple[dict, int]],
                     fail_once_at: int | None = None, more_meta: dict | None = None):
    """Roll out `scenarios` with accessdesk.harness.hf_backend.rollout and a sampler that answers each prompt
    with the next message of the scripted oracle, as token ids. Returns (records of the rollout, the records
    of the same scenarios played without tokens, the layout). The scenarios must differ from each other.
    `fail_once_at`: the episode at this position raises in its first step() call."""
    layout = TokenLayout(tokenizer, text.tool_specs("five"))
    sampler = ScriptedSampler(tokenizer, cfg, layout)
    episodes, played = [], []
    for position, (family, style) in enumerate(scenarios):
        scripted = run_policy(Episode(family, style, text, cfg), oracle(text))
        played.append(scripted.record(run="integration", arm="scripted", sample=0))
        episode = Episode(family, style, text, cfg)
        sampler.add_script(layout.first_prompt(episode.system_prompt, episode.request_text),
                           [turn["assistant_text"] for turn in scripted.turns])
        if position == fail_once_at:
            _fail_in_the_first_step(episode)
        episodes.append(episode)
    records = rollout(sampler, {"five": layout}, episodes, run_seed=5, pass_index=0,
                      meta={"run": "integration", "arm": "scripted", **(more_meta or {})})
    return records, played, layout


def _fail_in_the_first_step(episode: Episode) -> None:
    real_step = episode.step

    def step(*args, **kwargs):
        episode.step = real_step
        raise RuntimeError("made-up failure")

    episode.step = step


@pytest.mark.tokenizer
def test_the_rollout_loop_gives_the_scripted_records_and_the_templates_tokens(tokenizer, text, cfg, families):
    cfg = dataclasses.replace(cfg, chunk_size=3)                                   # two chunks: 3 + 2
    by_outcome = first_of_each_outcome(families["dev"])
    scenarios = [(by_outcome[outcome], style) for outcome, style in zip(OUTCOMES, (1, 2, 3, 4))]
    scenarios.append((next(f for f in families["dev"] if f["homonym"] and f not in by_outcome.values()), 4))
    records, played, layout = scripted_rollout(tokenizer, text, cfg, scenarios)
    line_break = tokenizer.encode("\n", add_special_tokens=False)
    assert len(records) == 5
    for position, (record, scripted, (family, style)) in enumerate(zip(records, played, scenarios)):
        tokens = record["tokens"]
        assert list(record) == RECORD_KEYS + ["tokens"] and list(tokens) == ["ids", "segments", "logprobs"]
        assert record["seed"] == {"run_seed": 5, "pass": 0, "chunk": position // 3, "pos": position % 3}
        # Apart from the seed and the token counts, the record is the one of the episode played as text.
        same = [key for key in RECORD_KEYS if key not in ("seed", "turns")]
        assert {key: record[key] for key in same} == {key: scripted[key] for key in same}
        assert [{**turn, "n_tokens": None} for turn in record["turns"]] == scripted["turns"]
        assert record["passed"] and record["reward"] == 100 and record["harness_error"] is None
        # Its tokens are the buffer that is built from the text, and the chat template's render.
        buffer = buffer_from_record(layout, scripted, family, text)
        assert tokens["ids"] == buffer.ids
        assert [Segment(*segment) for segment in tokens["segments"]] == buffer.segments
        episode = Episode(family, style, text, cfg)
        wanted = template_ids(tokenizer, scripted, episode.system_prompt, episode.request_text, text)
        assert tokens["ids"] + line_break == wanted
        assert len(tokens["logprobs"]) == sum(turn["n_tokens"] for turn in record["turns"])
        assert replay(record, family, text, cfg) == record["final_state"]


@pytest.mark.tokenizer
def test_a_harness_error_of_the_rollout_is_a_whole_record_that_no_number_counts(tokenizer, text, cfg,
                                                                                families):
    scenarios = [(family, 1) for family in families["dev"][:4]]
    records, played, layout = scripted_rollout(tokenizer, text, cfg, scenarios, fail_once_at=1)
    assert [record["harness_error"] is None for record in records] == [True, False, True, True]
    broken = records[1]
    assert "made-up failure" in broken["harness_error"]
    assert list(broken) == RECORD_KEYS + ["tokens"] and broken["scenario_id"] == played[1]["scenario_id"]
    assert (broken["ruleset"], broken["toolset"], broken["prompt"], broken["sample"]) == \
        ("default", "five", "final", 0)
    # Left out of the rates, counted as an error, refused as training data, kept in the coverage of an arm.
    assert metrics.harness_error_count(records) == 1 and len(metrics.valid_records(records)) == 3
    assert metrics.pass_rate(records) == 1.0 and len(metrics.family_pass_rates(records)) == 3
    with pytest.raises(ValueError):
        buffer_from_record(layout, broken, scenarios[1][0], text)
    coverage = tables.scenario_sample_counts("scripted", records)
    assert coverage == {(family["family_id"], 1): 1 for family, _ in scenarios}


@pytest.mark.tokenizer
def test_the_rollout_writes_the_hash_of_the_scenario_file_into_every_record(tokenizer, text, cfg, families):
    """The records of the model runs carried no `data_sha256` (the calibration runs have none), so the
    results table could not refuse two arms that were run on different scenario files."""
    data_hash = "0123456789abcdef" * 4
    scenarios = [(family, 1) for family in families["dev"][:3]]
    records, _, _ = scripted_rollout(tokenizer, text, cfg, scenarios, fail_once_at=1,
                                     more_meta={"data_sha256": data_hash})
    assert [record["harness_error"] is None for record in records] == [True, False, True]
    for record in records:                                # the harness error carries it too
        assert list(record) == RECORD_KEYS + ["data_sha256", "tokens"]
        assert record["data_sha256"] == data_hash
    # Two such arms on different files are refused by the table; before, both were accepted with a note.
    other = [{**record, "data_sha256": "f" * 64, "arm": "other"} for record in records]
    with pytest.raises(ValueError, match="same scenario file"):
        tables.compare_arms({"scripted": records, "other": other}, "scripted", n_draws=50)
    # Without the key in `meta` the records have no such field, as before.
    plain, _, _ = scripted_rollout(tokenizer, text, cfg, scenarios[:1])
    assert list(plain[0]) == RECORD_KEYS + ["tokens"]


class NoSampler:
    """A sampler that must never be asked for a turn: it only has what rollout() checks before it starts."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg, self.stop_id, self.pad_id = cfg, 1, 2

    def sample(self, prompts: list[list[int]], seed: int) -> list:
        raise AssertionError("rollout() sampled although an episode has another config")


class NoLayout:
    im_end_id, pad_id = 1, 2


@pytest.mark.parametrize("change", [{"max_new_tokens": 64}, {"turn_limit": 4}, {"parser_strict": False},
                                    {"temperature": 0.5}, {"top_k": 20}, {"top_p": 0.9},
                                    {"malformed_penalty": 0}, {"chunk_size": 4}])
def test_rollout_refuses_an_episode_that_was_built_with_another_config_than_the_sampler(text, cfg, families,
                                                                                        change):
    """PLAN 5.4: one config object holds the token cap, the turn limit, the parser mode and the sampling
    settings. The sampler reads the token cap and the sampling settings, the episode the turn limit, the
    parser mode and the reward numbers: rollout() must not start when the two were given different
    configs. (No tokenizer is needed: the check comes before the first prompt.)"""
    family = families["dev"][0]
    other = dataclasses.replace(cfg, **change)
    episodes = [Episode(family, 1, text, cfg), Episode(family, 2, text, other)]
    with pytest.raises(ValueError, match="another config than the sampler"):
        rollout(NoSampler(cfg), {"five": NoLayout()}, episodes, run_seed=0, pass_index=0, meta={})
    with pytest.raises(ValueError, match="another config than the sampler"):
        rollout(NoSampler(other), {"five": NoLayout()}, episodes[:1], run_seed=0, pass_index=0, meta={})
    assert not any(episode.turns for episode in episodes)


def test_rollout_refuses_a_meta_key_that_it_does_not_know(text, cfg, families):
    episodes = [Episode(families["dev"][0], 1, text, cfg)]
    with pytest.raises(TypeError, match="data_sha256 and first_sample"):
        rollout(NoSampler(cfg), {"five": NoLayout()}, episodes, run_seed=0, pass_index=0,
                meta={"model": "some/model"})


# ---------------------------------------------------------------------------
# The text bundle: what a conversation holds, and what the tokenizer does to it
# ---------------------------------------------------------------------------
MARKER = re.compile(MARKER_PATTERN)
GENERATED_ID = re.compile(f"{MARKER_PATTERN}[{TOKEN_ALPHABET}]{{6}}")     # a marker (the prefix) + 6 letters
# The chat format of a tool call, typed by hand from chat_template.jinja.
CALL_SYNTAX = re.compile(r"</?tool_call>|<function=|</function>|<parameter=|</parameter>|>|\s")
JSON_ONLY = re.compile(r'(?:[{}\[\]:," ]|true|false|null|[0-9])*')


def left_over(content: str, names: list[str]) -> str:
    """What is left of a message when every generated id, marker and person name is taken out of it."""
    content = MARKER.sub("", GENERATED_ID.sub("", content))
    for name in sorted(names, key=len, reverse=True):
        content = content.replace(name, "")
    return content


def strangers(messages: list[dict], names: list[str]) -> list[tuple[str, str]]:
    """(role, left-over text) of every message that holds more than markers, ids, names and its format."""
    found = []
    for message in messages:
        rest = left_over(message["content"], names)
        if message["role"] == "assistant":
            rest = CALL_SYNTAX.sub("", rest)
        elif message["role"] == "tool":
            rest = "" if JSON_ONLY.fullmatch(rest) else rest
        if rest.strip() != "":
            found.append((message["role"], rest))
    return found


def test_with_a_marker_bundle_a_whole_conversation_holds_only_markers_ids_names_and_json(marker_text, pools,
                                                                                         cfg):
    """In marker_text every visible string of the text file is a unique marker. If a whole conversation then
    holds anything but markers, generated ids, person names, the tags of a tool call and JSON punctuation,
    some code wrote a model-visible string itself."""
    runs = {"oracle": ("five", oracle(marker_text)),
            "extra tool, then oracle": ("six", policies.extra_tool_then_oracle(marker_text)),
            "one malformed message, then oracle": ("five", policies.stumbling_oracle(marker_text, 1)),
            "wrong role (tool errors)": ("five", policies.wrong_role(marker_text)),
            "close at once (made-up ids)": ("five", policies.close_at_once(marker_text, "denied"))}
    n_messages = 0
    seen_roles: set[str] = set()
    for family in make_split("calib", marker_text, pools, cfg):
        names = [person["name"] for person in family["world"]["people"]]
        for label, (toolset, policy) in runs.items():
            for prompt in ("final", "plain"):
                episode = run_policy(Episode(family, 4, marker_text, cfg, toolset=toolset, prompt=prompt),
                                     policy)
                assert MARKER.fullmatch(episode.messages[0]["content"]), label
                assert strangers(episode.messages, names) == [], (label, family["family_id"])
                n_messages += len(episode.messages)
                seen_roles.update(message["role"] for message in episode.messages)
        shifted = Episode(family, 1, marker_text, cfg, ruleset=marker_text.policy_shift_variant,
                          prompt="policy_shift")
        run_policy(shifted, oracle(marker_text, marker_text.policy_shift_variant))
        assert shifted.record()["passed"] and strangers(shifted.messages, names) == []
    assert n_messages > 2000 and seen_roles == {"system", "user", "assistant", "tool"}

    # The check can fail: with the placeholder text the same conversation is full of other words.
    plain = load_text(placeholder_path())
    family = make_split("calib", plain, pools, cfg)[0]
    episode = run_policy(Episode(family, 4, plain, cfg), oracle(plain))
    names = [person["name"] for person in family["world"]["people"]]
    assert {role for role, _ in strangers(episode.messages, names)} == {"system", "user", "assistant", "tool"}
    # And a single word written by code would be found.
    assert strangers([{"role": "tool", "content": '{"Zq001zQ": "ok"}'}], names) == [("tool", '{"": "ok"}')]
    clean = '{"Zq001zQ": [true, null, 3, "Zq002zQabc234"]}'
    assert strangers([{"role": "tool", "content": clean}], names) == []


def test_the_text_fixture_can_be_pointed_at_another_text_file(monkeypatch, tmp_path):
    """ACCESSDESK_TEST_TEXT (tests/conftest.py) is how these tests and the validation suite are run on the
    final text file. Without it the fixture is the placeholder."""
    from tests.conftest import ROOT, TEXT_FILE_VARIABLE, text_file

    monkeypatch.delenv(TEXT_FILE_VARIABLE, raising=False)
    assert text_file() == placeholder_path()
    monkeypatch.setenv(TEXT_FILE_VARIABLE, "text/model_text.yaml")
    assert text_file() == ROOT / "text" / "model_text.yaml"
    monkeypatch.setenv(TEXT_FILE_VARIABLE, str(tmp_path / "other.yaml"))
    assert text_file() == tmp_path / "other.yaml"
    monkeypatch.setenv(TEXT_FILE_VARIABLE, "")
    assert text_file() == placeholder_path()


def survives_the_tokenizer(tokenizer, value: str) -> bool:
    ids = tokenizer.encode(value, add_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=False) == value


def bundle_and_families(which: str, text, families, pools, cfg) -> tuple:
    """("fixture"): the text and the families of the fixtures. ("model"): text/model_text.yaml and the
    families generated from it; the test is skipped until that file is complete."""
    if which == "fixture":
        return text, families
    if check_text(model_text_path()):
        pytest.skip("text/model_text.yaml is not complete yet")
    bundle = load_text(model_text_path())
    return bundle, make_all_splits(bundle, pools, cfg)


@pytest.mark.parametrize("which", ("fixture", "model"))
def test_the_two_scripted_rows_hold_on_every_split_of_a_text_file(which, text, families, pools, cfg):
    """The correct policy gets 100 on every scenario and each constant policy passes exactly its own
    outcome, in every split. The second case starts by itself on the day text/model_text.yaml is complete,
    so a plain `pytest` then checks the final text and its wordings too. The whole validation suite
    on that text: see ACCESSDESK_TEST_TEXT in tests/conftest.py."""
    bundle, splits = bundle_and_families(which, text, families, pools, cfg)
    named = named_policies(bundle)
    assert sorted(splits) == ["calib", "dev", "test", "train"]
    for split in sorted(splits):
        correct = play_split(splits[split], named["oracle"], "oracle", bundle, cfg)
        assert len(correct) == sum(len(family["wordings"]) for family in splits[split])
        assert all(record["passed"] and record["reward"] == 100 for record in correct), split
        for name, right_for in CONSTANTS.items():
            records = play_split(splits[split], named[name], name, bundle, cfg)
            assert metrics.pass_rate(records) == 0.25, (split, name)
            by_outcome = metrics.pass_rate_by(records, "outcome")
            assert by_outcome == {outcome: float(outcome == right_for) for outcome in OUTCOMES}, (split, name)


def test_the_held_out_style_is_one_and_the_same_in_the_generator_and_in_the_metrics(families, dev_records):
    assert metrics.HELD_OUT_STYLE not in TRAIN_STYLES and STYLES == (*TRAIN_STYLES, metrics.HELD_OUT_STYLE)
    styles = {split: {wording["style"] for family in families[split] for wording in family["wordings"]}
              for split in families}
    assert styles["train"] == set(TRAIN_STYLES)
    assert all(styles[split] == set(STYLES) for split in ("dev", "test", "calib"))
    held_out = [r for r in dev_records["oracle"] if r["style"] == metrics.HELD_OUT_STYLE]
    assert len(held_out) == 48
    assert metrics.pass_rate_by(dev_records["oracle"], "style") == {style: 1.0 for style in STYLES}


@pytest.mark.tokenizer
@pytest.mark.parametrize("which", ("fixture", "model"))
def test_every_string_that_the_model_must_write_back_survives_the_tokenizer(which, tokenizer, text, families,
                                                                            pools, cfg):
    """The tokenizer normalises text (Unicode NFC). A name, role or department that it changes would reach
    the model in another spelling than the world holds, and the model's correct call would then not match.
    Checked for the text of the fixture, and for text/model_text.yaml as soon as that file is complete."""
    bundle, _ = bundle_and_families(which, text, families, pools, cfg)
    values = [name for first, last in pools.values() for name in (*first, *last)]
    values += [*bundle.departments, *bundle.roles, *bundle.resolution.values(), *bundle.status.values()]
    values += [spec.name for spec in bundle.tool_specs("six")]
    values += [param.name for spec in bundle.tool_specs("six") for param in spec.params]
    rng = Rng(0)
    values += [prefix + rng.token(6) for prefix in (bundle.id_prefix_person, bundle.id_prefix_role)
               for _ in range(200)]
    assert len(values) > 800
    changed = [value for value in values if not survives_the_tokenizer(tokenizer, value)
               or unicodedata.normalize("NFC", value) != value]
    assert changed == []
    # The check can fail: a letter followed by a combining accent comes back as one character.
    assert not survives_the_tokenizer(tokenizer, "René")
    assert survives_the_tokenizer(tokenizer, "René")
