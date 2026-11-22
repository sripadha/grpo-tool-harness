"""accessdesk/harness/records.py: the record files, and the replay of a saved episode."""
from __future__ import annotations

import copy
import json

import pytest

from accessdesk.config import RunConfig
from accessdesk.harness import policies
from accessdesk.harness.episode import Episode
from accessdesk.harness.records import read_records, replay, write_records
from tests.core.test_episode import EXPECTED_PENDING, FAMILY, call, correct_calls, play


def hand_record(text, cfg, messages: list[str] | None = None) -> dict:
    """The record of the hand-made family of test_episode.py, played with messages typed by hand."""
    if messages is None:
        messages = ["", call(text, "grant_role", person_id="p9", role_id="r2")] + correct_calls(text)
    return play(Episode(FAMILY, 1, text, cfg), messages).record(run="hand", arm="test")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def test_records_come_back_from_a_file_as_they_were_written(tmp_path, text, cfg):
    records = [hand_record(text, cfg), hand_record(text, cfg, [""] * 8)]
    path = tmp_path / "episodes.jsonl"
    write_records(path, records)
    assert read_records(path) == records
    assert [list(r) for r in read_records(path)] == [list(r) for r in records]        # the key order is kept


def test_the_file_is_one_json_object_per_line_in_utf8(tmp_path):
    records = [{"b": 1, "a": "Zoë Müller", "c": [1, {"z": None}]}, {"text": "two\nlines"}]
    path = tmp_path / "episodes.jsonl"
    write_records(path, records)
    raw = path.read_bytes()
    assert raw == ('{"b": 1, "a": "Zoë Müller", "c": [1, {"z": null}]}\n'
                   '{"text": "two\\nlines"}\n').encode("utf-8")
    assert read_records(path) == records


def test_writing_replaces_the_file_and_an_empty_list_gives_an_empty_file(tmp_path):
    path = tmp_path / "episodes.jsonl"
    write_records(path, [{"n": 1}, {"n": 2}])
    write_records(path, [{"n": 3}])
    assert read_records(path) == [{"n": 3}]
    write_records(path, [])
    assert path.read_bytes() == b"" and read_records(path) == []


def test_the_same_episodes_give_the_same_bytes(tmp_path, text, cfg, families):
    for name in ("a", "b"):
        policy = policies.random_valid_calls(text, 3)
        records = [policies.run_policy(Episode(family, 4, text, cfg), policy).record(run="x")
                   for family in families["calib"]]
        write_records(tmp_path / name, records)
    assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()
    assert len((tmp_path / "a").read_bytes().splitlines()) == 24


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def test_replay_gives_the_final_state_of_the_saved_calls(text, cfg):
    record = hand_record(text, cfg)
    assert (record["n_malformed"], record["n_tool_errors"], record["n_turns"]) == (1, 1, 6)
    assert replay(record, FAMILY, text, cfg) == EXPECTED_PENDING == record["final_state"]
    untouched = hand_record(text, cfg, [""] * 8)
    assert replay(untouched, FAMILY, text, cfg) == untouched["final_state"]
    assert replay(untouched, FAMILY, text, cfg)["request"]["resolution"] is None


def test_replay_changes_neither_the_record_nor_the_family(text, cfg):
    record = hand_record(text, cfg)
    before = (copy.deepcopy(record), copy.deepcopy(FAMILY))
    replay(record, FAMILY, text, cfg)
    assert (record, FAMILY) == before


def test_replay_uses_the_saved_calls_and_not_the_texts_or_the_saved_state(text, cfg):
    record = copy.deepcopy(hand_record(text, cfg))
    for turn in record["turns"]:
        turn["assistant_text"] = "(lost)"
    record["final_state"] = {"wrong": True}
    record["passed"], record["reward"] = False, 0
    assert replay(record, FAMILY, text, cfg) == EXPECTED_PENDING


def test_replay_works_on_a_record_that_was_read_from_a_file(tmp_path, text, cfg):
    write_records(tmp_path / "e.jsonl", [hand_record(text, cfg)])
    record = read_records(tmp_path / "e.jsonl")[0]
    assert replay(record, FAMILY, text, cfg) == record["final_state"] == EXPECTED_PENDING


def test_replay_refuses_a_record_of_another_family(text, cfg):
    other = {**copy.deepcopy(FAMILY), "family_id": "hand-2"}
    with pytest.raises(ValueError, match="family hand-1"):
        replay(hand_record(text, cfg), other, text, cfg)


def test_replay_refuses_more_turns_than_the_turn_limit(text, cfg):
    record = hand_record(text, cfg)
    assert replay(record, FAMILY, text, RunConfig(turn_limit=6)) == EXPECTED_PENDING
    with pytest.raises(ValueError, match="turn limit is 5"):
        replay(record, FAMILY, text, RunConfig(turn_limit=5))


def test_replay_refuses_a_call_after_the_close(text, cfg):
    record = copy.deepcopy(hand_record(text, cfg))
    record["turns"].append({**record["turns"][0], "turn": 6})
    with pytest.raises(ValueError, match="turn 6 comes after the request was closed"):
        replay(record, FAMILY, text, cfg)


@pytest.mark.parametrize("turn, field, value", [
    (4, "args", {"person_id": "p1", "role_id": "r2", "approver_id": "p1"}),      # other approver: same answer
    (3, "args", {"person_id": "p1", "role_name": "Wiki Editor"}),                # other role: another answer
    (2, "tool_response", '{"people": []}'),                                      # an answer that was edited
    (1, "tool_error", None),                                                     # an error that was removed
    (5, "args", {"resolution": "pending", "person_id": "p1", "role_id": "r9"}),  # the close now fails
])
def test_replay_notices_a_record_that_was_changed(text, cfg, turn, field, value):
    record = copy.deepcopy(hand_record(text, cfg))
    record["turns"][turn][field] = value
    try:
        state = replay(record, FAMILY, text, cfg)
    except ValueError as error:
        assert "differs from the saved one" in str(error)
    else:
        # A changed call with the same tool answer is only visible in the state.
        assert state != record["final_state"]
        assert (turn, field) == (4, "args")


def test_every_scripted_episode_of_the_dev_split_replays_to_its_saved_state(text, cfg, families):
    """The same check runs on all 54,000 episodes of test_validation_suite.py; this one is quick to read."""
    chosen = {"oracle": policies.oracle(text), "random": policies.random_valid_calls(text, 0),
              "wrong_role": policies.wrong_role(text), "stumbling": policies.stumbling_oracle(text, 2),
              "extra_grant": policies.extra_grant(text, "second_role")}
    states = set()
    for name, policy in chosen.items():
        for family in families["dev"]:
            for wording in family["wordings"]:
                record = policies.run_policy(Episode(family, wording["style"], text, cfg), policy).record()
                final = record["final_state"]
                assert replay(record, family, text, cfg) == final, (name, record["scenario_id"])
                states.add(json.dumps(final["request"]) + str(len(final["grants"])))
    assert len(states) > 100           # many different end states were replayed, not one
