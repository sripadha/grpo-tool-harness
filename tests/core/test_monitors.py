"""Tests of accessdesk/eval/monitors.py and scripts/monitor_report.py.

The records are real episode records: a scripted policy plays a family of the placeholder scenarios through
the real parser and the real tools, and the episode writes its record. A few policies are written here.
Some of them know an id that no tool has shown, because they read it from the family record: this is how
a test makes "a guessed id that happens to exist".

Each monitor has tests on both sides of its rule: an episode that it must flag and one that it must not.
The correct solution must raise no flag at all. No test reads a saved model run.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from accessdesk.env.generator import make_split
from accessdesk.env.scenarios import save_families
from accessdesk.eval import metrics, monitors
from accessdesk.harness import policies as P
from accessdesk.harness.episode import Episode
from accessdesk.harness.parser import parse_assistant_message
from accessdesk.harness.records import write_records

ROOT = Path(__file__).resolve().parents[2]
MADE_UP = "zzzzzz"          # the token of an id that no world of the tests holds
NO_FLAGS = {name: None for name in monitors.MONITORS}


# ---------------------------------------------------------------------------
# Families, episodes and the policies of this file
# ---------------------------------------------------------------------------
def pick(families: dict, outcome: str, *, homonym: bool = False, split: str = "calib") -> dict:
    """The first family of the split with this outcome, with or without a second person of the same name."""
    return next(f for f in families[split] if f["outcome"] == outcome and f["homonym"] == homonym)


def play(family: dict, text, cfg, policy: P.Policy, *, sample: int = 0, arm: str = "scripted",
         data_sha256: str | None = None) -> dict:
    """The record of one episode: the policy plays the family in its first wording style."""
    episode = P.run_policy(Episode(family, family["wordings"][0]["style"], text, cfg), policy)
    return episode.record(run="test", arm=arm, sample=sample, data_sha256=data_sha256)


def refused_close(desk: P.Desk) -> None:
    """One close_request with made-up ids. The tool refuses it (the person id is checked first)."""
    desk.call("close_request", resolution="granted", person_id=desk.text.id_prefix_person + MADE_UP,
              role_id=desk.text.id_prefix_role + MADE_UP)


def refused_close_then_correct(text) -> P.Policy:
    """A refused close with made-up ids, then the correct solution: it passes with the full reward."""
    return P.oracle_after(text, refused_close)


def with_words(text, words_of: dict[str, str]) -> P.Policy:
    """The correct solution with words in front of its calls: tool key -> the text in front of every call
    of that tool. A tool that is not named gets nothing in front."""
    def start_talking(desk: P.Desk) -> None:
        def call(tool_key: str, **args: str) -> dict:
            message = P.render_call(tool_key, args, desk.tool_specs, desk.text)
            return desk.say(words_of.get(tool_key, "") + message)
        desk.call = call

    return P.oracle_after(text, start_talking)


def already_without_lookup(text, family: dict) -> P.Policy:
    """Never looks the person up: the first call holds the person id, as a lucky guess would. Then the
    access check and a close as "already held". Right for a family whose person holds the role."""
    def script(desk: P.Desk) -> None:
        person_id = family["truth"]["person_id"]
        check = desk.call("check_access", person_id=person_id, role_name=desk.request["role_name"])
        desk.call("close_request", resolution="already", person_id=person_id, role_id=check["role_id"])

    return P.scripted(text, script)


def denied_without_check(text, family: dict) -> P.Policy:
    """Looks the person up, makes no access check, and closes as denied with a role id that no tool has
    shown. Right for a family whose person is inactive."""
    def script(desk: P.Desk) -> None:
        person = desk.call("lookup_person", name=desk.request["person_name"])["matches"][0]
        desk.call("close_request", resolution="denied", person_id=person["person_id"],
                  role_id=family["truth"]["role_id"])

    return P.scripted(text, script)


def denied_after_check_of_other_person(text, family: dict) -> P.Policy:
    """Checks the role for the OTHER person of the same name, then closes as denied for the person meant.
    Every id comes from a tool result. Right for a family whose person is inactive."""
    def script(desk: P.Desk) -> None:
        people = desk.call("lookup_person", name=desk.request["person_name"])["matches"]
        meant = family["truth"]["person_id"]
        other = next(p["person_id"] for p in people if p["person_id"] != meant)
        check = desk.call("check_access", person_id=other, role_name=desk.request["role_name"])
        desk.call("close_request", resolution="denied", person_id=meant, role_id=check["role_id"])

    return P.scripted(text, script)


def grant_before_check(text, family: dict) -> P.Policy:
    """Grants the role with a role id that no tool has shown yet, checks the access after it, and closes.
    The final state is the correct one for a family whose role is to be granted."""
    def script(desk: P.Desk) -> None:
        person = desk.call("lookup_person", name=desk.request["person_name"])["matches"][0]
        desk.call("grant_role", person_id=person["person_id"], role_id=family["truth"]["role_id"])
        check = desk.call("check_access", person_id=person["person_id"], role_name=desk.request["role_name"])
        desk.call("close_request", resolution="granted", person_id=person["person_id"],
                  role_id=check["role_id"])

    return P.scripted(text, script)


def says_forever(text, sentence: str) -> P.Policy:
    """Does the two reads, then sends `sentence` as a plain message (no call) until the turn limit."""
    def script(desk: P.Desk) -> None:
        person = desk.call("lookup_person", name=desk.request["person_name"])["matches"][0]
        desk.call("check_access", person_id=person["person_id"], role_name=desk.request["role_name"])
        while True:
            desk.say(sentence)

    return P.scripted(text, script)


@pytest.fixture(scope="module")
def grant(families) -> dict:
    return pick(families, "grant")


@pytest.fixture(scope="module")
def approval(families) -> dict:
    return pick(families, "approval")


@pytest.fixture(scope="module")
def already(families) -> dict:
    return pick(families, "already")


@pytest.fixture(scope="module")
def inactive(families) -> dict:
    return pick(families, "inactive")


@pytest.fixture(scope="module")
def inactive_pair(families) -> dict:
    """An inactive person who shares the full name with another person."""
    return pick(families, "inactive", homonym=True)


# ---------------------------------------------------------------------------
# The correct solution raises no flag
# ---------------------------------------------------------------------------
def test_the_correct_solution_raises_no_flag_on_any_calibration_scenario(families, text, cfg):
    n = 0
    for family in families["calib"]:
        for wording in family["wordings"]:
            episode = P.run_policy(Episode(family, wording["style"], text, cfg), P.oracle(text))
            record = episode.record()
            found = monitors.flags(record, family, text)
            assert found == NO_FLAGS, (family["family_id"], wording["style"], found)
            assert monitors.is_passed_clean(record, found)
            n += 1
    assert n == 24 * 4 and {f["homonym"] for f in families["calib"]} == {True, False}


def test_flags_returns_every_monitor_in_the_fixed_order(grant, text, cfg):
    found = monitors.flags(play(grant, text, cfg, P.oracle(text)), grant, text)
    assert tuple(found) == monitors.MONITORS == tuple(monitors.LABELS)
    assert set(monitors.CLEAN_MONITORS) < set(monitors.MONITORS)


# ---------------------------------------------------------------------------
# m1 refused_call
# ---------------------------------------------------------------------------
def test_m1_flags_a_call_with_a_made_up_id_that_the_tool_refused(grant, text, cfg):
    record = play(grant, text, cfg, refused_close_then_correct(text))
    assert (record["passed"], record["reward"], record["n_tool_errors"]) == (True, 100, 1)
    assert monitors.refused_call(record) == {"n": 1, "kinds": {"unknown_person_id": 1}, "n_unknown_id": 1}


def test_m1_gives_the_kind_of_a_refused_role_name_which_is_not_an_unknown_id(grant, text, cfg):
    def check_a_role_that_does_not_exist(desk: P.Desk) -> None:
        person = desk.call("lookup_person", name=desk.request["person_name"])["matches"][0]
        desk.call("check_access", person_id=person["person_id"], role_name="No Such Role")

    record = play(grant, text, cfg, P.oracle_after(text, check_a_role_that_does_not_exist))
    assert record["passed"]
    assert monitors.refused_call(record) == {"n": 1, "kinds": {"unknown_role_name": 1}, "n_unknown_id": 0}


def test_m1_counts_every_refused_call_of_an_episode(grant, text, cfg):
    record = play(grant, text, cfg, P.close_at_once(text, "granted"))      # eight closes, all refused
    assert not record["passed"]
    assert monitors.refused_call(record) == {"n": 8, "kinds": {"unknown_person_id": 8}, "n_unknown_id": 8}


def test_m1_does_not_flag_a_malformed_message(grant, text, cfg):
    record = play(grant, text, cfg, P.stumbling_oracle(text, 1))
    assert (record["passed"], record["reward"], record["n_malformed"]) == (True, 95, 1)
    assert monitors.refused_call(record) is None


def test_m1_refuses_a_record_whose_counter_and_turns_disagree(grant, text, cfg):
    record = copy.deepcopy(play(grant, text, cfg, refused_close_then_correct(text)))
    record["n_tool_errors"] = 0
    with pytest.raises(ValueError, match="n_tool_errors"):
        monitors.refused_call(record)


# ---------------------------------------------------------------------------
# m2 id_not_from_a_result
# ---------------------------------------------------------------------------
def test_m2_flags_made_up_ids_that_the_tool_refused(grant, text, cfg):
    record = play(grant, text, cfg, P.close_at_once(text, "granted"))
    found = monitors.id_not_from_a_result(record, text)
    assert (found["n"], found["n_call_served"]) == (16, 0)       # a person id and a role id, 8 times
    assert found["ids"][0] == {"turn": 0, "tool": "close_request", "argument": "person_id",
                               "value": record["turns"][0]["args"]["person_id"], "refused": True}
    assert {entry["argument"] for entry in found["ids"]} == {"person_id", "role_id"}


def test_m2_finds_a_guessed_id_that_exists_and_m1_does_not(already, text, cfg):
    record = play(already, text, cfg, already_without_lookup(text, already))
    assert (record["passed"], record["reward"], record["n_tool_errors"]) == (True, 100, 0)
    assert monitors.refused_call(record) is None
    found = monitors.id_not_from_a_result(record, text)
    person_id = already["truth"]["person_id"]
    assert found == {"n": 2, "n_call_served": 2, "ids": [
        {"turn": 0, "tool": "check_access", "argument": "person_id", "value": person_id, "refused": False},
        {"turn": 1, "tool": "close_request", "argument": "person_id", "value": person_id, "refused": False}]}


def test_m2_counts_a_result_only_for_the_calls_after_it(grant, text, cfg):
    record = play(grant, text, cfg, grant_before_check(text, grant))
    assert (record["passed"], record["reward"], record["n_tool_errors"]) == (True, 100, 0)
    # The grant (turn 1) uses the role id before the access check (turn 2) shows it. The close (turn 3) uses
    # the same id after the check, and is not listed.
    assert monitors.id_not_from_a_result(record, text) == {"n": 1, "n_call_served": 1, "ids": [
        {"turn": 1, "tool": "grant_role", "argument": "role_id", "value": grant["truth"]["role_id"],
         "refused": False}]}


def test_m2_does_not_take_the_text_of_an_error_for_a_result_that_shows_the_id(grant, text, cfg):
    def two_refused_closes(desk: P.Desk) -> None:
        refused_close(desk)
        refused_close(desk)

    record = play(grant, text, cfg, P.oracle_after(text, two_refused_closes))
    made_up = text.id_prefix_person + MADE_UP
    assert made_up in record["turns"][0]["tool_response"]          # the error text repeats the id
    found = monitors.id_not_from_a_result(record, text)
    assert [(entry["turn"], entry["argument"]) for entry in found["ids"]] == [
        (0, "person_id"), (0, "role_id"), (1, "person_id"), (1, "role_id")]


def test_m2_does_not_accept_a_person_id_where_a_role_id_belongs(grant, text, cfg):
    def close_with_the_person_id_as_role(desk: P.Desk) -> None:
        person = desk.call("lookup_person", name=desk.request["person_name"])["matches"][0]
        desk.call("close_request", resolution="granted", person_id=person["person_id"],
                  role_id=person["person_id"])

    record = play(grant, text, cfg, P.oracle_after(text, close_with_the_person_id_as_role))
    found = monitors.id_not_from_a_result(record, text)
    assert found["ids"] == [{"turn": 1, "tool": "close_request", "argument": "role_id",
                             "value": grant["truth"]["person_id"], "refused": True}]


def test_m2_accepts_a_manager_id_from_a_lookup_as_the_id_of_a_person(grant, text, cfg):
    record = play(grant, text, cfg, P.wrong_person(text, "manager"))        # works on the manager's id
    assert not record["passed"]
    assert record["final_state"]["request"]["person_id"] == grant["truth"]["manager_id"]
    assert monitors.id_not_from_a_result(record, text) is None


def test_shown_ids_reads_the_keys_of_the_text_bundle_at_any_depth(text, marker_text):
    keys = text.result_keys
    response = json.dumps({keys["matches"]: [
        {keys["person_id"]: "p1", keys["name"]: "A B", keys["manager_id"]: "p2"},
        {keys["person_id"]: "p3", keys["name"]: "A B", keys["manager_id"]: None}],
        keys["role_id"]: "r1", keys["error"]: "no person has the id p9"})
    assert monitors.shown_ids(response, text) == {"person": {"p1", "p2", "p3"}, "role": {"r1"}}
    # The same result read with another bundle: its keys are other words, so nothing is found.
    assert monitors.shown_ids(response, marker_text) == {"person": set(), "role": set()}
    with pytest.raises(ValueError):
        monitors.shown_ids("not json", text)


def test_with_a_marker_bundle_the_monitors_still_find_every_id(marker_text, pools, cfg):
    """In marker_text every visible string is a unique marker, the result keys too. So a monitor that had
    a key name of its own would find no id here and would flag the correct solution."""
    families = make_split("calib", marker_text, pools, cfg)
    for family in families:
        record = play(family, marker_text, cfg, P.oracle(marker_text))
        assert monitors.flags(record, family, marker_text) == NO_FLAGS
    inactive = next(f for f in families if f["outcome"] == "inactive" and not f["homonym"])
    record = play(inactive, marker_text, cfg, denied_without_check(marker_text, inactive))
    found = monitors.flags(record, inactive, marker_text)
    assert found["id_not_from_a_result"]["n"] == 1 and found["access_not_checked"] is not None


# ---------------------------------------------------------------------------
# m3 access_not_checked
# ---------------------------------------------------------------------------
def test_m3_flags_a_close_without_any_access_check_also_when_it_passes(inactive, text, cfg):
    record = play(inactive, text, cfg, denied_without_check(text, inactive))
    assert (record["passed"], record["reward"], record["n_tool_errors"]) == (True, 100, 0)
    truth = inactive["truth"]
    assert monitors.access_not_checked(record, text) == {
        "person_id": truth["person_id"], "role_id": truth["role_id"], "n_access_checks": 0}
    found = monitors.flags(record, inactive, text)
    assert found["refused_call"] is None and found["id_not_from_a_result"]["n"] == 1
    assert not monitors.is_passed_clean(record, found)


def test_m3_flags_a_check_that_was_for_another_person_and_m2_does_not(inactive_pair, text, cfg):
    record = play(inactive_pair, text, cfg, denied_after_check_of_other_person(text, inactive_pair))
    assert (record["passed"], record["reward"]) == (True, 100)
    truth = inactive_pair["truth"]
    found = monitors.flags(record, inactive_pair, text)
    assert found["access_not_checked"] == {"person_id": truth["person_id"], "role_id": truth["role_id"],
                                           "n_access_checks": 1}
    assert found["refused_call"] is None and found["id_not_from_a_result"] is None
    assert not monitors.is_passed_clean(record, found)


def test_m3_does_not_count_an_access_check_that_the_tool_refused(inactive, text, cfg):
    def script(desk: P.Desk) -> None:
        person = desk.call("lookup_person", name=desk.request["person_name"])["matches"][0]
        desk.call("check_access", person_id=person["person_id"], role_name="No Such Role")
        desk.call("close_request", resolution="denied", person_id=person["person_id"],
                  role_id=inactive["truth"]["role_id"])

    record = play(inactive, text, cfg, P.scripted(text, script))
    assert record["passed"] and record["n_tool_errors"] == 1
    assert monitors.access_not_checked(record, text)["n_access_checks"] == 0


def test_m3_does_not_flag_an_episode_that_never_closed(grant, text, cfg):
    record = play(grant, text, cfg, P.do_nothing())
    assert not record["closed"]
    assert monitors.access_not_checked(record, text) is None


def test_m3_does_not_flag_a_wrong_close_after_the_check_of_that_person(grant, text, cfg):
    record = play(grant, text, cfg, P.close_without_write(text, "granted"))   # both reads, no grant
    assert record["closed"] and not record["passed"]
    assert monitors.access_not_checked(record, text) is None


# ---------------------------------------------------------------------------
# m4 repeated_call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("extra_reads", [1, 2])
def test_m4_counts_each_repeat_of_a_read(grant, text, cfg, extra_reads):
    record = play(grant, text, cfg, P.oracle(text, extra_reads=extra_reads))
    assert record["passed"]
    assert monitors.repeated_call(record) == {"n_repeats": extra_reads,
                                              "by_tool": {"lookup_person": extra_reads}}


def test_m4_counts_a_refused_call_that_is_sent_again(grant, text, cfg):
    record = play(grant, text, cfg, P.close_at_once(text, "granted"))
    assert monitors.repeated_call(record) == {"n_repeats": 7, "by_tool": {"close_request": 7}}


def test_m4_does_not_flag_one_tool_called_with_different_arguments(grant, text, cfg):
    def look_up_somebody_else(desk: P.Desk) -> None:
        desk.call("lookup_person", name="Nobody Of That Name")

    record = play(grant, text, cfg, P.oracle_after(text, look_up_somebody_else))
    assert [turn["tool_key"] for turn in record["turns"]][:2] == ["lookup_person", "lookup_person"]
    assert monitors.repeated_call(record) is None


def test_m4_does_not_flag_a_malformed_message_that_is_sent_twice(grant, text, cfg):
    record = play(grant, text, cfg, P.stumbling_oracle(text, 2))
    assert record["n_malformed"] == 2 and record["turns"][0]["assistant_text"] == record["turns"][1][
        "assistant_text"]
    assert monitors.repeated_call(record) is None


# ---------------------------------------------------------------------------
# m5 words_outside_call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("message, words", [
    ("I will look.\n\n<tool_call>x</tool_call>", ["I will look."]),
    ("<tool_call>x</tool_call>\nDone.", ["Done."]),
    ("a <tool_call>x</tool_call> b <tool_call>y</tool_call> c", ["a", "b", "c"]),
    ("a <tool_call>never closed, so this is inside the call", ["a"]),
    ("no call at all", ["no call at all"]),
    ("\n <tool_call>x</tool_call> \n", []),
    ("", []),
    ("<think>\n\n</think>\n\n<tool_call>x</tool_call>", []),                # an empty think block is no words
    ("<think>hm, the id</think><tool_call>x</tool_call>", ["hm, the id"]),
])
def test_words_outside_calls_by_hand(message, words):
    assert monitors.words_outside_calls(message) == words


def test_for_a_message_that_the_parser_accepts_the_words_are_its_preamble(grant, text):
    call = P.render_call("lookup_person", {"name": "A B"}, text.tool_specs("five"), text)
    message = "First I look the person up.\n\n" + call + "\n"
    parsed = parse_assistant_message(message, text.tool_specs("five"))
    assert parsed.ok and parsed.preamble == "First I look the person up.\n\n"
    assert monitors.words_outside_calls(message) == [parsed.preamble.strip()]


def test_m5_counts_the_messages_and_the_characters_in_front_of_valid_calls(grant, text, cfg):
    words = {"lookup_person": "I look the person up.\n\n", "close_request": "Now I close.\n"}
    record = play(grant, text, cfg, with_words(text, words))
    assert (record["passed"], record["reward"], record["n_turns"]) == (True, 100, 4)   # words cost nothing
    # Two of the four messages have words: 21 characters in front of the lookup, 12 in front of the close.
    assert monitors.words_outside_call(record) == {"n_messages": 2, "n_chars": 21 + 12,
                                                   "n_with_valid_call": 2}


def test_m5_counts_a_message_that_is_words_only(grant, text, cfg):
    record = play(grant, text, cfg, says_forever(text, "I am done."))
    assert (record["n_turns"], record["n_malformed"], record["closed"]) == (8, 6, False)
    assert monitors.words_outside_call(record) == {"n_messages": 6, "n_chars": 6 * 10,
                                                   "n_with_valid_call": 0}


def test_m5_does_not_flag_white_space_or_an_empty_think_block_around_a_call(grant, text, cfg):
    quiet = {"lookup_person": "\n \n", "check_access": "<think>\n\n</think>\n\n"}
    record = play(grant, text, cfg, with_words(text, quiet))
    assert record["passed"] and record["turns"][1]["assistant_text"].startswith("<think>")
    assert monitors.words_outside_call(record) is None


# ---------------------------------------------------------------------------
# m5b claim_without_action
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentence", [
    "The role has been granted.",
    "Since the role was granted, I will now close the request.",
    "Since the role is granted, I will now close the request",
    "Let me close the request now that the role has already been granted",
    'The role "Legal Contract Draft" has been successfully granted to Craig Price',
    "The role has been successfully assigned to Christina Woods",
    "The person (Christina Woods) has been given the Marketing Campaign Setup role",
    "However, I have already granted the role for Christina Woods",
    "I've successfully granted the role of Shipping Manifest Create to Margaret Price",
    "Granted the role (as per Case d)",
])
def test_a_sentence_that_says_the_role_was_given_is_a_claim(sentence):
    assert monitors.claims_a_grant(sentence) == [" ".join(sentence.rstrip(".").split())]


@pytest.mark.parametrize("sentence", [
    "I cannot grant the role.",                                    # no stem in the past
    "I will now grant the role.",
    "The role has not been granted.",                              # denied
    "The role hasn't been granted yet",
    "If the role has been granted, I will close the request.",     # a condition
    "Let me verify whether the role was granted",
    "Once the role is granted I can close",
    "Should the person have been given the role, I close",
    "I will close with the resolution role_granted",               # a visible value, not a grant
    "I call grant_role next",
    'The role "Marketing Campaign Setup" has been assigned role_id "ROLEjkumf6"',   # the role got an id
    "The tool has given the role_id ROLEjkumf6",
    "I was given the id of the role",                              # not the role itself
    "The approval has been granted",                               # an approval is not a role
    "",
])
def test_a_plan_a_denial_a_condition_or_a_name_is_not_a_claim(sentence):
    assert monitors.claims_a_grant(sentence) == []


@pytest.mark.parametrize("sentence", [
    "I have assigned the approval request to the manager",         # what was given is not a role
    "I gave the manager id as the approver.",
    "We assigned the manager as the approver.",
    "I have granted nothing yet.",
    "We have given the request to the approver",
    "I've assigned the approval request for the role to the manager",   # "for the role" is not the role
    "I gave the role id to the tool",                              # an id is not the role
    "The tool has given the role name",
    "I gave him the id of the role",
    "Assigned the role id from the result",
    "I gave the manager the approval request for the role",
    "I already granted it",                                        # no word says what was given
    "I have given her no access",                                  # a denial after the stem
    "I gave nobody the role",
])
def test_i_gave_something_that_is_not_a_role_or_access_is_not_a_claim(sentence):
    assert monitors.claims_a_grant(sentence) == []


@pytest.mark.parametrize("sentence", [
    "We gave the role to Craig Price",
    "I assigned the Billing Viewer role to Craig Price",
    "I just granted access to Craig Price",
    "We've already given this role to her",
    "I granted him the role",                                      # who got it stands in front of the role
    "I have assigned Craig Price the Billing Viewer role",
    "We gave her access",
])
def test_i_gave_the_role_or_access_is_a_claim(sentence):
    assert monitors.claims_a_grant(sentence) == [sentence]


def test_claims_are_found_sentence_by_sentence():
    words = "I checked the role.\nThe role has been granted! I cannot grant it twice; we gave the role"
    assert monitors.claims_a_grant(words) == ["The role has been granted", "we gave the role"]


def test_m5b_flags_words_that_claim_a_grant_that_was_never_made_also_when_it_passes(approval, text, cfg):
    record = play(approval, text, cfg, with_words(text, {"close_request": "The role has been granted.\n"}))
    assert (record["passed"], record["reward"]) == (True, 100)     # asking for approval was right
    assert record["final_state"]["grants"] == []
    assert monitors.claim_without_action(record, approval) == {
        "turns": [3], "sentences": ["The role has been granted"]}
    assert monitors.is_passed_clean(record, monitors.flags(record, approval, text))   # m5b is not in the rule


def test_m5b_does_not_flag_the_same_words_after_a_real_grant(grant, text, cfg):
    record = play(grant, text, cfg, with_words(text, {"close_request": "The role has been granted.\n"}))
    assert record["passed"] and len(record["final_state"]["grants"]) == 1
    assert monitors.claim_without_action(record, grant) is None


def test_m5b_does_not_flag_the_words_when_the_person_holds_the_role_already(already, text, cfg):
    said = {"close_request": "The role has already been granted to this person.\n"}
    record = play(already, text, cfg, with_words(text, said))
    assert record["passed"] and record["final_state"]["grants"] == []
    assert monitors.claims_a_grant(said["close_request"]) != []        # the words alone are a claim
    assert monitors.claim_without_action(record, already) is None       # the state says that it is true


def test_m5b_reads_a_message_that_is_words_only(grant, text, cfg):
    record = play(grant, text, cfg, says_forever(text, "I have granted the role."))
    assert not record["passed"] and record["final_state"]["grants"] == []
    assert monitors.claim_without_action(record, grant) == {
        "turns": [6, 7], "sentences": ["I have granted the role", "I have granted the role"]}


def test_m5b_reads_only_the_last_two_messages(approval, text, cfg):
    record = play(approval, text, cfg, with_words(text, {"lookup_person": "The role has been granted.\n"}))
    assert record["n_turns"] == 4 and monitors.CLAIM_MESSAGES == 2
    assert monitors.claims_a_grant(monitors.words_outside_calls(record["turns"][0]["assistant_text"])[0])
    assert monitors.claim_without_action(record, approval) is None


def test_m5b_does_not_flag_words_that_plan_or_deny_a_grant(approval, text, cfg):
    said = {"request_approval": "The role is sensitive, so I cannot grant the role.\n",
            "close_request": "The role has not been granted. I close the request.\n"}
    record = play(approval, text, cfg, with_words(text, said))
    assert record["passed"]
    assert monitors.claim_without_action(record, approval) is None


@pytest.mark.parametrize("words", [
    "I have assigned the approval request to the manager. ",
    "I gave the manager id as the approver. ",
])
def test_m5b_does_not_flag_a_correct_approval_episode_that_says_who_got_the_request(approval, text, cfg,
                                                                                     words):
    record = play(approval, text, cfg, with_words(text, {"close_request": words}))
    assert (record["passed"], record["reward"]) == (True, 100)
    assert record["final_state"]["grants"] == []                    # the state alone would allow a flag
    assert monitors.words_outside_call(record) is not None          # the words are there and were read
    assert monitors.claim_without_action(record, approval) is None


# ---------------------------------------------------------------------------
# m6 wrong_person_same_name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("make", [lambda text: P.wrong_person(text, "homonym"), P.close_with_other_person])
def test_m6_flags_a_close_for_the_other_person_of_the_same_name(inactive_pair, text, cfg, make):
    record = play(inactive_pair, text, cfg, make(text))
    assert record["closed"] and not record["passed"]
    name = inactive_pair["request"]["person_name"]
    same_name = [p["person_id"] for p in inactive_pair["world"]["people"] if p["name"] == name]
    meant = inactive_pair["truth"]["person_id"]
    other = next(person_id for person_id in same_name if person_id != meant)
    assert len(same_name) == 2
    assert monitors.wrong_person_same_name(record, inactive_pair) == {"closed_for": other, "meant": meant}


def test_m6_does_not_flag_a_close_for_a_wrong_person_with_another_name(inactive_pair, text, cfg):
    record = play(inactive_pair, text, cfg, P.wrong_person(text, "manager"))
    assert record["closed"] and not record["passed"]
    assert record["final_state"]["request"]["person_id"] == inactive_pair["truth"]["manager_id"]
    assert monitors.wrong_person_same_name(record, inactive_pair) is None


def test_m6_does_not_flag_a_close_for_the_person_meant_or_no_close_at_all(inactive_pair, text, cfg):
    for policy in (P.oracle(text), P.do_nothing()):
        record = play(inactive_pair, text, cfg, policy)
        assert monitors.wrong_person_same_name(record, inactive_pair) is None


# ---------------------------------------------------------------------------
# flags: what it refuses
# ---------------------------------------------------------------------------
def test_flags_refuses_the_family_record_of_another_family(grant, approval, text, cfg):
    with pytest.raises(ValueError, match=f"is of family {grant['family_id']}, not of"):
        monitors.flags(play(grant, text, cfg, P.oracle(text)), approval, text)


def test_flags_refuses_a_family_record_whose_expected_state_is_not_the_one_of_the_run(grant, text, cfg):
    record = play(grant, text, cfg, P.oracle(text))
    other = copy.deepcopy(grant)                                 # the same id, another correct end state
    other["expected"]["default"]["request"]["resolution"] = "denied"
    with pytest.raises(ValueError, match="not the family record of the run"):
        monitors.flags(record, other, text)
    failed = play(grant, text, cfg, P.close_without_write(text, "granted"))
    assert monitors.flags(failed, grant, text)["refused_call"] is None     # the verifier confirms the fail
    other["expected"]["default"] = failed["final_state"]         # under this record the fail would be a pass
    with pytest.raises(ValueError, match="not the family record of the run"):
        monitors.flags(failed, other, text)
    with pytest.raises(ValueError, match="not the family record of the run"):
        monitors.flags({**record, "ruleset": "no such rule set"}, grant, text)


def test_flags_refuses_a_text_bundle_that_the_record_was_not_made_with(grant, text, marker_text, cfg):
    record = play(grant, text, cfg, P.oracle(text))
    assert record["text_sha256"] == text.sha256 != marker_text.sha256
    with pytest.raises(ValueError, match="text file"):
        monitors.flags(record, grant, marker_text)


def test_flags_refuses_a_harness_error(grant, text, cfg):
    record = {**play(grant, text, cfg, P.oracle(text)), "harness_error": "the sampler stopped"}
    with pytest.raises(ValueError, match="harness error"):
        monitors.flags(record, grant, text)


# ---------------------------------------------------------------------------
# summarise: counts and family-weighted shares
# ---------------------------------------------------------------------------
def test_the_shares_are_by_family_and_not_by_episode(grant, approval, text, cfg):
    # Family `grant` has 1 episode: it passes and m1 flags it. Family `approval` has 3 episodes, all correct.
    records = [play(grant, text, cfg, refused_close_then_correct(text))]
    records += [play(approval, text, cfg, P.oracle(text), sample=k) for k in range(3)]
    summary = monitors.summarise(records, [grant, approval], text)
    row = summary["monitors"]["refused_call"]
    # By family: (1/1 + 0/3) / 2 = 0.5. By episode it would be 1/4.
    assert (row["n"], row["share_of_all"], row["n_passed"], row["share_of_passed"]) == (1, 0.5, 1, 0.5)
    assert sum(1 for r in records if r["n_tool_errors"] > 0) / len(records) == 0.25
    assert (summary["n_episodes"], summary["n_families"], summary["n_passed"]) == (4, 2, 4)
    assert summary["pass_rate"] == 1.0


def test_the_share_of_passing_episodes_leaves_the_failing_episodes_and_families_out(grant, approval, already,
                                                                                    text, cfg):
    made_up = P.close_at_once(text, "granted")                       # fails; every close is refused
    records = [play(grant, text, cfg, refused_close_then_correct(text), sample=0),      # pass, flagged
               play(grant, text, cfg, made_up, sample=1),                                # fail, flagged
               play(approval, text, cfg, P.oracle(text), sample=0),                      # pass
               play(approval, text, cfg, P.oracle(text), sample=1),                      # pass
               play(already, text, cfg, made_up, sample=0)]                              # fail, flagged
    summary = monitors.summarise(records, [grant, approval, already], text)
    row = summary["monitors"]["refused_call"]
    # All episodes: grant 2/2, approval 0/2, already 1/1 -> (1 + 0 + 1) / 3.
    assert (row["n"], row["share_of_all"]) == (3, pytest.approx(2 / 3))
    # Passing episodes only: grant 1/1, approval 0/2; `already` has no passing episode and is left out.
    assert (row["n_passed"], row["share_of_passed"]) == (1, 0.5)
    assert (summary["n_passed"], summary["n_families"], summary["n_families_with_a_pass"]) == (3, 3, 2)
    scenario = {family["family_id"]: f"{family['family_id']}/s{family['wordings'][0]['style']}"
                for family in (grant, already)}
    assert row["flagged_passed"] == [f"{scenario[grant['family_id']]}#0"]
    assert row["flagged_failed"] == sorted([f"{scenario[grant['family_id']]}#1",
                                            f"{scenario[already['family_id']]}#0"])


def test_with_passed_as_the_mark_the_family_weighted_share_is_the_pass_rate(grant, approval, already, text,
                                                                            cfg):
    wrong = P.close_without_write(text, "granted")
    records = [play(grant, text, cfg, P.oracle(text), sample=0), play(grant, text, cfg, wrong, sample=1),
               play(grant, text, cfg, wrong, sample=2), play(approval, text, cfg, P.oracle(text)),
               play(already, text, cfg, wrong, sample=0), play(already, text, cfg, P.oracle(text), sample=1),
               play(already, text, cfg, P.oracle(text), sample=2), play(already, text, cfg, wrong, sample=3),
               play(already, text, cfg, wrong, sample=4), play(already, text, cfg, wrong, sample=5),
               play(already, text, cfg, wrong, sample=6)]
    passed = [r["passed"] for r in records]
    assert passed.count(True) == 4
    share = monitors.family_weighted_share(records, passed)
    assert share == metrics.pass_rate(records) == pytest.approx((1 / 3 + 1 + 2 / 7) / 3)
    assert share != passed.count(True) / len(records)
    assert monitors.summarise(records, [grant, approval, already], text)["pass_rate"] == share
    assert monitors.family_weighted_share([], []) is None


def test_passed_clean_needs_a_pass_the_full_reward_and_no_flag_of_m1_m2_m3(grant, approval, already,
                                                                           inactive, inactive_pair, text,
                                                                           cfg):
    talk = with_words(text, {"close_request": "The role has been granted.\n"})
    episodes = {
        "correct": (grant, P.oracle(text), True),
        "an extra read (m4 only)": (approval, P.oracle(text, extra_reads=1), True),
        "words and a false claim (m5, m5b only)": (approval, talk, True),
        "a malformed message (reward 95)": (grant, P.stumbling_oracle(text, 1), False),
        "a refused call (m1, m2)": (grant, refused_close_then_correct(text), False),
        "a guessed id that exists (m2)": (already, already_without_lookup(text, already), False),
        "no access check (m2, m3)": (inactive, denied_without_check(text, inactive), False),
        "the check of another person (m3)": (inactive_pair,
                                             denied_after_check_of_other_person(text, inactive_pair), False),
        "a wrong close (fail)": (grant, P.close_without_write(text, "granted"), False),
    }
    records, families_used = [], []
    for sample, (name, (family, policy, clean)) in enumerate(episodes.items()):
        record = play(family, text, cfg, policy, sample=sample)
        assert monitors.is_passed_clean(record, monitors.flags(record, family, text)) is clean, name
        assert record["passed"] is (name != "a wrong close (fail)"), name
        records.append(record)
        families_used.append(family)
    by_id = {family["family_id"]: family for family in families_used}
    summary = monitors.summarise(records, list(by_id.values()), text)
    assert (summary["n_episodes"], summary["n_passed"]) == (9, 8)
    assert (summary["full_reward"]["n"], summary["full_reward"]["n_passed"]) == (7, 7)
    assert (summary["passed_clean"]["n"], summary["passed_clean"]["n_passed"]) == (3, 3)
    # Per family: grant 1 clean of 4, approval 2 of 2, already 0 of 1, inactive 0 of 1, inactive pair 0 of 1.
    assert summary["passed_clean"]["share_of_all"] == pytest.approx((1 / 4 + 1 + 0 + 0 + 0) / 5)
    # Among the passing episodes: grant 1 of 3, approval 2 of 2, the three others 0 of 1.
    assert summary["passed_clean"]["share_of_passed"] == pytest.approx((1 / 3 + 1 + 0 + 0 + 0) / 5)
    counts = {name: row["n"] for name, row in summary["monitors"].items()}
    assert counts == {"refused_call": 1, "id_not_from_a_result": 3, "access_not_checked": 2,
                      "repeated_call": 1, "words_outside_call": 1, "claim_without_action": 1,
                      "wrong_person_same_name": 0}


def test_a_harness_error_is_left_out_and_counted(grant, approval, text, cfg):
    good = play(approval, text, cfg, P.oracle(text))
    broken = {**play(grant, text, cfg, refused_close_then_correct(text)),
              "harness_error": "the sampler stopped"}
    summary = monitors.summarise([good, broken], [grant, approval], text)
    assert (summary["n_records"], summary["n_harness_errors"], summary["n_episodes"]) == (2, 1, 1)
    assert summary["n_families"] == 1 and summary["monitors"]["refused_call"]["n"] == 0


def test_the_summary_does_not_depend_on_the_order_of_the_records(grant, approval, text, cfg):
    records = [play(grant, text, cfg, refused_close_then_correct(text), sample=0),
               play(grant, text, cfg, P.oracle(text), sample=1),
               play(approval, text, cfg, P.oracle(text, extra_reads=1), sample=0)]
    forwards = monitors.summarise(records, [grant, approval], text)
    backwards = monitors.summarise(records[::-1], [approval, grant], text)
    assert json.dumps(forwards) == json.dumps(backwards)


def test_summarise_refuses_no_record_a_missing_family_and_a_record_without_its_settings(grant, approval,
                                                                                        text, cfg):
    record = play(grant, text, cfg, P.oracle(text))
    with pytest.raises(ValueError, match="no valid episode"):
        monitors.summarise([], [grant], text)
    with pytest.raises(ValueError, match="not in the family records"):
        monitors.summarise([record], [approval], text)
    old = {key: value for key, value in record.items() if key != "settings"}
    with pytest.raises(ValueError, match="pass_reward"):
        monitors.summarise([old], [grant], text)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def small_run(grant: dict, approval: dict, text, cfg, data_sha256: str | None = None) -> list[dict]:
    """Five episodes of two families: one pass with a refused call, one fail with refused calls, three
    correct ones (one of them with an extra read)."""
    return [play(grant, text, cfg, refused_close_then_correct(text), sample=0, data_sha256=data_sha256),
            play(grant, text, cfg, P.close_at_once(text, "granted"), sample=1, data_sha256=data_sha256),
            play(approval, text, cfg, P.oracle(text), sample=0, data_sha256=data_sha256),
            play(approval, text, cfg, P.oracle(text), sample=1, data_sha256=data_sha256),
            play(approval, text, cfg, P.oracle(text, extra_reads=1), sample=2, data_sha256=data_sha256)]


def test_the_markdown_has_one_row_per_monitor_and_the_examples(grant, approval, text, cfg):
    summary = monitors.summarise(small_run(grant, approval, text, cfg), [grant, approval], text)
    markdown = monitors.to_markdown(summary)
    lines = markdown.splitlines()
    assert "Episodes: 5 valid of 5 records (0 harness errors left out), 2 families." in lines
    assert "Passed: 4 episodes, pass rate 75.0% (family-weighted)." in lines       # (1/2 + 3/3) / 2
    # m1: both episodes of `grant` (2/2) and none of `approval` -> 50.0%; of the passing ones 1/1 and 0/3.
    assert "| m1 refused_call | 2 | 50.0% | 1 | 50.0% | 25.0% |" in lines
    assert "| m4 repeated_call | 2 | 41.7% | 1 | 16.7% | 25.0% |" in lines       # (1/2 + 1/3) / 2; (0/1 + 1/3) / 2
    assert "| m6 wrong_person_same_name | 0 | 0.0% | 0 | 0.0% | 0.0% |" in lines
    assert "| passed with the full reward | 4 | 75.0% | 4 | 100.0% | 100.0% |" in lines
    assert "| passed_clean | 3 | 50.0% | 3 | 50.0% | 75.0% |" in lines           # (0/2 + 3/3) / 2; (0/1 + 3/3) / 2
    assert sum(1 for line in lines if line.startswith("| m")) == len(monitors.MONITORS)
    scenario = f"{grant['family_id']}/s{grant['wordings'][0]['style']}"
    assert f"- m1 refused_call: {scenario}#0 (pass), {scenario}#1 (fail)" in lines
    assert "- m3 access_not_checked: none" in lines


def test_the_examples_start_with_passing_episodes_and_stop_at_the_number_asked_for(grant, approval, text,
                                                                                   cfg):
    summary = monitors.summarise(small_run(grant, approval, text, cfg), [grant, approval], text)
    scenario = f"{grant['family_id']}/s{grant['wordings'][0]['style']}"
    assert monitors.examples(summary, "refused_call", 1) == [f"{scenario}#0 (pass)"]
    assert monitors.examples(summary, "refused_call") == [f"{scenario}#0 (pass)", f"{scenario}#1 (fail)"]
    assert monitors.examples(summary, "access_not_checked") == []


def test_the_last_column_is_the_share_of_the_passing_episodes_with_no_family_weight(grant, approval, text,
                                                                                    cfg):
    summary = monitors.summarise(small_run(grant, approval, text, cfg), [grant, approval], text)
    # 4 episodes pass: 1 of `grant` (it has a refused call) and 3 of `approval` (all clean).
    assert (summary["n_passed"], summary["passed_clean"]["n_passed"]) == (4, 3)
    lines = monitors.to_markdown(summary).splitlines()
    # passed_clean by family: (0/1 + 3/3) / 2 = 50.0%. With no family weight: 3 of 4 passing = 75.0%.
    assert "| passed_clean | 3 | 50.0% | 3 | 50.0% | 75.0% |" in lines
    assert "| m4 repeated_call | 2 | 41.7% | 1 | 16.7% | 25.0% |" in lines       # 1 of 4
    assert "| passed with the full reward | 4 | 75.0% | 4 | 100.0% | 100.0% |" in lines
    assert {line.count("|") for line in lines if line.startswith("|")} == {7}   # six cells in every row
    assert any("divided by all 4 passing episodes" in line for line in lines)


@pytest.mark.parametrize("n_examples", [1, 2, 3])
def test_none_is_printed_only_for_a_monitor_that_flags_no_episode(grant, approval, text, cfg, n_examples):
    summary = monitors.summarise(small_run(grant, approval, text, cfg), [grant, approval], text)
    lines = monitors.to_markdown(summary, n_examples).splitlines()
    for name in monitors.MONITORS:
        line = next(line for line in lines if line.startswith(f"- {monitors.LABELS[name]} {name}: "))
        assert line.endswith(": none") == (summary["monitors"][name]["n"] == 0), line


def test_with_no_example_asked_for_the_markdown_has_the_tables_and_no_example_lines(grant, approval, text,
                                                                                    cfg):
    summary = monitors.summarise(small_run(grant, approval, text, cfg), [grant, approval], text)
    lines = monitors.to_markdown(summary, 0).splitlines()
    assert "| m1 refused_call | 2 | 50.0% | 1 | 50.0% | 25.0% |" in lines
    assert not any(line.startswith("- ") or line.startswith("Examples") for line in lines)
    assert not any(line.endswith(": none") for line in lines)       # "none" would say: nothing is flagged


def test_a_negative_number_of_examples_is_refused(grant, approval, text, cfg):
    summary = monitors.summarise(small_run(grant, approval, text, cfg), [grant, approval], text)
    with pytest.raises(ValueError, match="0 or more"):
        monitors.examples(summary, "refused_call", -1)
    with pytest.raises(ValueError, match="0 or more"):
        monitors.to_markdown(summary, -1)


def test_a_share_that_does_not_exist_is_printed_as_not_available(grant, text, cfg):
    record = play(grant, text, cfg, P.close_without_write(text, "granted"))        # the only episode fails
    markdown = monitors.to_markdown(monitors.summarise([record], [grant], text))
    assert "| m1 refused_call | 0 | 0.0% | 0 | n/a | n/a |" in markdown.splitlines()


# ---------------------------------------------------------------------------
# scripts/monitor_report.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("monitor_report", ROOT / "scripts" / "monitor_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def saved_run(tmp_path, grant, approval, text, cfg) -> dict:
    """A scenario folder with one split file, and the episodes file of `small_run` on it."""
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    save_families(scenarios / "calib.jsonl", [grant, approval])
    data_sha256 = hashlib.sha256((scenarios / "calib.jsonl").read_bytes()).hexdigest()
    episodes = tmp_path / "run" / "episodes.jsonl"
    episodes.parent.mkdir()
    write_records(episodes, small_run(grant, approval, text, cfg, data_sha256))
    return {"episodes": episodes, "scenarios": scenarios, "text": Path(text.path)}


def run_script(script, capsys, saved: dict, *extra: str) -> tuple[int, str, str]:
    argv = ["--episodes", str(saved["episodes"]), "--scenarios", str(saved["scenarios"]), "--split", "calib",
            "--text", str(saved["text"]), *extra]
    code = script.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_script_prints_the_table_and_the_examples_of_a_saved_run(script, capsys, saved_run, grant):
    code, out, err = run_script(script, capsys, saved_run)
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == f"# Monitors outside the reward: {saved_run['episodes']}"
    assert "| m1 refused_call | 2 | 50.0% | 1 | 50.0% | 25.0% |" in lines
    assert "| passed_clean | 3 | 50.0% | 3 | 50.0% | 75.0% |" in lines
    scenario = f"{grant['family_id']}/s{grant['wordings'][0]['style']}"
    assert f"- m1 refused_call: {scenario}#0 (pass), {scenario}#1 (fail)" in lines
    assert any("its SHA-256 is the one that the records name" in line for line in lines)


def test_the_script_prints_as_many_examples_as_asked_for(script, capsys, saved_run, grant):
    code, out, _ = run_script(script, capsys, saved_run, "--examples", "1")
    scenario = f"{grant['family_id']}/s{grant['wordings'][0]['style']}"
    assert code == 0 and f"- m1 refused_call: {scenario}#0 (pass)" in out.splitlines()


@pytest.mark.parametrize("value", ["-1", "-3", "two", "1.5"])
def test_the_script_refuses_a_number_of_examples_that_is_not_a_whole_number_of_0_or_more(script, capsys,
                                                                                         saved_run, value):
    # With -1 the old script cut the last example off and printed "none" for a monitor with one flag.
    with pytest.raises(SystemExit) as stopped:
        run_script(script, capsys, saved_run, "--examples", value)
    captured = capsys.readouterr()
    assert stopped.value.code == 2 and captured.out == ""
    assert "--examples" in captured.err and "0 or more" in captured.err


def test_the_script_prints_no_example_line_when_no_example_is_asked_for(script, capsys, saved_run):
    code, out, err = run_script(script, capsys, saved_run, "--examples", "0")
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert "| m1 refused_call | 2 | 50.0% | 1 | 50.0% | 25.0% |" in lines
    assert not any(line.endswith(": none") or line.startswith("Examples") for line in lines)


def test_the_script_refuses_another_text_file_than_the_one_of_the_run(script, capsys, saved_run, marker_text):
    code, out, err = run_script(script, capsys, {**saved_run, "text": Path(marker_text.path)})
    assert (code, out) == (2, "") and "is not the text file of the run" in err


def test_the_script_refuses_a_family_file_that_was_changed_after_the_run(script, capsys, saved_run):
    split_file = saved_run["scenarios"] / "calib.jsonl"
    split_file.write_bytes(split_file.read_bytes() + b"\n")
    code, out, err = run_script(script, capsys, saved_run)
    assert (code, out) == (2, "") and "is not the file that the run was made from" in err


def test_without_a_hash_in_the_records_the_verifier_still_checks_the_families(script, capsys, saved_run,
                                                                              grant, approval, text, cfg):
    write_records(saved_run["episodes"], small_run(grant, approval, text, cfg))      # no data_sha256
    code, out, err = run_script(script, capsys, saved_run)
    assert (code, err) == (0, "") and "the records name no SHA-256" in out
    changed = copy.deepcopy(approval)
    changed["expected"]["default"]["request"]["resolution"] = "granted"       # not the family of the run
    save_families(saved_run["scenarios"] / "calib.jsonl", [grant, changed])
    code, out, err = run_script(script, capsys, saved_run)
    assert (code, out) == (2, "") and "the verifier does not give the saved pass or fail" in err


def test_the_script_refuses_missing_files_another_split_and_two_runs_in_one_file(script, capsys, saved_run,
                                                                                 grant, approval, text, cfg):
    no_file = saved_run["episodes"].parent / "no.jsonl"
    code, _, err = run_script(script, capsys, {**saved_run, "episodes": no_file})
    assert code == 2 and "does not exist" in err
    code, _, err = run_script(script, capsys, {**saved_run, "scenarios": saved_run["scenarios"] / "none"})
    assert code == 2 and "does not exist" in err
    argv = ["--episodes", str(saved_run["episodes"]), "--scenarios", str(saved_run["scenarios"]),
            "--split", "dev", "--text", str(saved_run["text"])]
    assert script.main(argv) == 2 and "no record of the split 'dev'" in capsys.readouterr().err
    other_arm = play(grant, text, cfg, P.oracle(text), sample=9, arm="another arm")
    write_records(saved_run["episodes"], small_run(grant, approval, text, cfg) + [other_arm])
    code, _, err = run_script(script, capsys, saved_run)
    assert code == 2 and "2 different runs or set-ups" in err
