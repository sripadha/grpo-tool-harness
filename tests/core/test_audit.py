"""The audit of ANALYSIS_PLAN.md section 8 (accessdesk/audit/ and its four scripts): the seeded samples,
the reading sheet and the labels form, the formula audit, and the positive control.

The attempts come from the scripted policies, played on a scenario folder that the real build script
writes into a temporary folder, and from hand-made records. Nothing is written into the repository.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from accessdesk import impl
from accessdesk.audit import Run, broken, formula, load_run, record_differences, sample, sheet
from accessdesk.env import rules
from accessdesk.env.scenarios import load_families
from accessdesk.eval import monitors
from accessdesk.harness import policies
from accessdesk.harness.episode import Episode
from accessdesk.harness.records import write_records
from accessdesk.rng import Rng
from accessdesk.text import normalise
from accessdesk.types import OUTCOMES
from tests.helpers.make_records import make_record, scenario_records

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "calib"
ARM = "arm"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory, text, cfg) -> Path:
    """A scenario folder as the build script writes it, from the text and the config of the tests."""
    folder = tmp_path_factory.mktemp("audit") / "scenarios_for_audit"
    load_script("build_scenarios").build(text, ROOT / "data" / "names", folder, cfg)
    return folder


@pytest.fixture(scope="module")
def families(scenarios) -> dict[str, dict]:
    """The families of the split that the tests play, by id. (In this file the name `families` means
    this dict, not the four splits of conftest.py.)"""
    return {family["family_id"]: family for family in load_families(scenarios / f"{SPLIT}.jsonl")}


@pytest.fixture(scope="module")
def data_sha256(scenarios) -> str:
    return hashlib.sha256((scenarios / f"{SPLIT}.jsonl").read_bytes()).hexdigest()


def play(policy, families, text, cfg, data_sha256, *, sample_number=0, outcome=None,
         ruleset="default") -> list[dict]:
    """One record per scenario of the families (only those of `outcome`, if given), by a scripted policy,
    graded under `ruleset`."""
    records = []
    for family in families.values():
        if outcome is not None and family["outcome"] != outcome:
            continue
        for wording in family["wordings"]:
            episode = Episode(family, wording["style"], text, cfg, ruleset=ruleset)
            policies.run_policy(episode, policy)
            records.append(episode.record(run="played", arm="played", sample=sample_number,
                                          seed={"run_seed": 0, "pass": sample_number, "chunk": 0, "pos": 0},
                                          data_sha256=data_sha256))
    return records


def stray_call_then_oracle(text):
    """One access check for a person id that no tool showed (the tool refuses it), then the correct
    solution. The attempt passes with the full reward: the refused call uses the one spare turn."""
    def stray(desk):
        desk.call("check_access", person_id="nobody-with-this-id", role_name=desk.request["role_name"])
    return policies.oracle_after(text, stray)


@pytest.fixture(scope="module")
def played(families, text, cfg, data_sha256) -> dict[str, list[dict]]:
    """The attempts of five policies on every scenario; each policy has its own sample number, so that
    all of them can stand in one file. `no_write` closes every request as granted and grants nothing:
    it fails everywhere, and its request is always closed."""
    chosen = {"oracle": policies.oracle(text), "wrong_role": policies.wrong_role(text),
              "stray": stray_call_then_oracle(text), "stumbling": policies.stumbling_oracle(text, 1),
              "no_write": policies.close_without_write(text, "granted")}
    return {name: play(policy, families, text, cfg, data_sha256, sample_number=number)
            for number, (name, policy) in enumerate(chosen.items())}


def run_of(records: list[dict], families, text, data_sha256) -> Run:
    """The records as one run, as `load_run` gives it for a file."""
    return Run(Path("episodes.jsonl"), records, families, text, data_sha256, "")


def hand_attempt(family_id: str, style: int = 1, sample_number: int = 0, *, arm: str = ARM,
                 outcome: str = "grant", reward: int = 100, flagged: tuple[str, ...] = ()) -> sample.Attempt:
    """A hand-made attempt: reward 100 or 95 is a pass, 0 a fail. `flagged` names the monitors that found
    something."""
    record = make_record(family_id, style, sample_number, passed=reward > 0, reward=reward, outcome=outcome,
                         n_malformed=1 if reward == 95 else 0)
    found = {name: ({"n": 1} if name in flagged else None) for name in monitors.MONITORS}
    return sample.Attempt(arm, record, {}, found)


def keys_of(attempts) -> list[tuple[str, str, int]]:
    return [attempt.key for attempt in attempts]


def many_attempts(reward: int = 100, per_outcome: int = 12, first: int = 0) -> list[sample.Attempt]:
    """`per_outcome` hand-made attempts of each outcome, all with one reward. The family numbers start
    at `first`, so two calls with another `first` (1000 apart) give attempts with other identities."""
    return [hand_attempt(f"dv-{number:03d}", 1 + number % 4, number % 3, outcome=outcome, reward=reward)
            for position, outcome in enumerate(OUTCOMES)
            for number in range(first + 100 * position, first + 100 * position + per_outcome)]


# ---------------------------------------------------------------------------
# Loading a run
# ---------------------------------------------------------------------------
def saved(tmp_path: Path, records: list[dict]) -> Path:
    episodes = tmp_path / "episodes.jsonl"
    write_records(episodes, records)
    return episodes


def test_a_run_is_loaded_with_its_families_its_text_and_the_hashes_of_its_files(
        tmp_path, scenarios, text, played, families, data_sha256):
    episodes = saved(tmp_path, played["oracle"])
    run = load_run(episodes, scenarios, SPLIT, Path(text.path))
    assert run.records == played["oracle"] and run.families == families
    assert run.text.sha256 == text.sha256 and run.data_sha256 == data_sha256
    assert run.episodes_sha256 == hashlib.sha256(episodes.read_bytes()).hexdigest()


def test_a_file_that_is_missing_or_holds_two_set_ups_is_refused(tmp_path, scenarios, text, played):
    with pytest.raises(ValueError, match="does not exist"):
        load_run(tmp_path / "missing.jsonl", scenarios, SPLIT, Path(text.path))
    two = played["oracle"][:4] + [{**record, "prompt": "plain"} for record in played["oracle"][4:8]]
    with pytest.raises(ValueError, match="2 different runs or set-ups"):
        load_run(saved(tmp_path, two), scenarios, SPLIT, Path(text.path))
    with pytest.raises(ValueError, match="has no record of the split 'dev'"):
        load_run(saved(tmp_path, played["oracle"]), scenarios, "dev", Path(text.path))


def test_records_of_another_text_file_or_another_scenario_file_are_refused(tmp_path, scenarios, text, played):
    other_text = [{**record, "text_sha256": "0" * 64} for record in played["oracle"]]
    with pytest.raises(ValueError, match="is not the text file of the run"):
        load_run(saved(tmp_path, other_text), scenarios, SPLIT, Path(text.path))
    other_data = [{**record, "data_sha256": "0" * 64} for record in played["oracle"]]
    with pytest.raises(ValueError, match="is not the file that the run was made from"):
        load_run(saved(tmp_path, other_data), scenarios, SPLIT, Path(text.path))


def test_a_split_file_that_was_changed_after_the_build_is_refused(tmp_path, scenarios, text, played):
    copied = tmp_path / "scenarios"
    copied.mkdir()
    for path in scenarios.iterdir():
        (copied / path.name).write_bytes(path.read_bytes())
    with (copied / f"{SPLIT}.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="is not the file that the build wrote"):
        load_run(saved(tmp_path, played["oracle"]), copied, SPLIT, Path(text.path))


# ---------------------------------------------------------------------------
# The seeded draw
# ---------------------------------------------------------------------------
def test_the_draw_takes_k_attempts_and_does_not_depend_on_the_order_of_the_file():
    attempts = many_attempts()
    drawn = sample.draw(attempts, 5, seed=0, kind=sample.TOP)
    assert len(drawn) == 5 and len(set(keys_of(drawn))) == 5
    assert keys_of(sample.draw(list(reversed(attempts)), 5, 0, sample.TOP)) == keys_of(drawn)
    assert keys_of(sample.draw(Rng(7).shuffle(attempts), 5, 0, sample.TOP)) == keys_of(drawn)


def test_the_draw_is_the_same_when_it_is_made_again_and_another_with_another_seed():
    attempts = many_attempts()
    first = keys_of(sample.draw(attempts, 5, 0, sample.TOP))
    assert keys_of(sample.draw(attempts, 5, 0, sample.TOP)) == first
    assert keys_of(sample.draw(attempts, 5, 1, sample.TOP)) != first
    assert keys_of(sample.draw(attempts, 5, 0, sample.FAILED)) != first


def test_the_draw_depends_on_the_identity_of_an_attempt_and_not_on_what_it_holds():
    attempts = many_attempts()
    other_content = [attempt._replace(record={**attempt.record, "n_turns": 7, "run": "another run"})
                     for attempt in attempts]
    drawn = keys_of(sample.draw(attempts, 5, 0, sample.TOP))
    assert keys_of(sample.draw(other_content, 5, 0, sample.TOP)) == drawn
    # The arm is part of the identity: under another arm name the ranks are other ranks.
    renamed = [attempt._replace(arm="other") for attempt in attempts]
    under_other_name = keys_of(sample.draw(renamed, 5, 0, sample.TOP))
    assert [(ARM, scenario, number) for _, scenario, number in under_other_name] != drawn


def test_a_drawn_attempt_stays_drawn_when_attempts_with_a_higher_rank_are_added():
    attempts = many_attempts()
    drawn = sample.draw(attempts, 5, 0, sample.TOP)
    highest = max(sample.rank(0, sample.TOP, attempt.key) for attempt in drawn)
    extra = [attempt for attempt in many_attempts(per_outcome=30)
             if attempt.key not in keys_of(attempts) and sample.rank(0, sample.TOP, attempt.key) > highest]
    assert len(extra) > 10
    assert keys_of(sample.draw(attempts + extra, 5, 0, sample.TOP)) == keys_of(drawn)


def test_fewer_attempts_than_k_are_all_taken():
    attempts = many_attempts()[:3]
    assert sorted(keys_of(sample.draw(attempts, 5, 0, sample.TOP))) == sorted(keys_of(attempts))


def test_two_attempts_with_one_identity_are_refused():
    attempts = many_attempts()
    with pytest.raises(ValueError, match="more than once"):
        sample.draw(attempts + attempts[:1], 5, 0, sample.TOP)


# ---------------------------------------------------------------------------
# The two samples
# ---------------------------------------------------------------------------
def test_the_top_reward_sample_has_five_top_reward_attempts_of_each_outcome():
    attempts = many_attempts(100) + many_attempts(95, 40, first=1000) + many_attempts(0, 50, first=2000)
    chosen = sample.top_reward_sample(attempts, seed=0)
    assert len(chosen) == 20
    assert all(item.attempt.record["reward"] == 100 and item.why == sample.DRAWN for item in chosen)
    assert all(item.kind == sample.TOP for item in chosen)
    for outcome in OUTCOMES:
        assert sum(1 for item in chosen if item.attempt.record["outcome"] == outcome) == 5


def test_every_flagged_top_reward_attempt_is_in_the_sample_and_no_other_flagged_attempt():
    flagged = [hand_attempt(f"dv-9{number:02d}", outcome="grant", flagged=("refused_call",))
               for number in range(7)]
    flagged.append(hand_attempt("dv-980", outcome="already", flagged=("access_not_checked",)))
    not_top = [hand_attempt("dv-990", reward=95, flagged=("refused_call",)),
               hand_attempt("dv-991", reward=0, flagged=("id_not_from_a_result",))]
    other_monitor = [hand_attempt(f"dv-97{number}", flagged=("repeated_call", "words_outside_call"))
                     for number in range(5)]
    attempts = many_attempts() + flagged + not_top + other_monitor
    chosen = sample.top_reward_sample(attempts, seed=0)
    in_sample = {item.attempt.key: item.why for item in chosen}
    assert all(attempt.key in in_sample for attempt in flagged)
    assert all(in_sample[attempt.key] in (sample.FLAGGED, sample.BOTH) for attempt in flagged)
    assert not any(attempt.key in in_sample for attempt in not_top)
    # m4 and m5 alone do not put an attempt into the sample; such an attempt can only be drawn.
    assert all(in_sample.get(attempt.key, sample.DRAWN) == sample.DRAWN for attempt in other_monitor)
    drawn = [why for why in in_sample.values() if why in (sample.DRAWN, sample.BOTH)]
    assert len(drawn) == 20
    assert len(chosen) == 20 + sum(1 for why in in_sample.values() if why == sample.FLAGGED)


def test_the_reverse_sample_has_three_failed_attempts_of_each_outcome():
    attempts = many_attempts(0) + [hand_attempt(f"dv-8{number:02d}", reward=100) for number in range(30)]
    chosen = sample.failed_sample(attempts, seed=0)
    assert len(chosen) == 12
    assert all(item.attempt.record["passed"] is False and item.kind == sample.FAILED for item in chosen)
    for outcome in OUTCOMES:
        assert sum(1 for item in chosen if item.attempt.record["outcome"] == outcome) == 3


def test_an_outcome_with_too_few_attempts_gives_all_it_has():
    attempts = many_attempts(0)[:2] + many_attempts(100, first=1000)[:4]        # all of the first outcome
    chosen = sample.choose(attempts, seed=0)
    assert sorted(item.kind for item in chosen) == [sample.FAILED] * 2 + [sample.TOP] * 4


def test_both_samples_do_not_depend_on_the_order_of_the_arms_and_of_the_files():
    one = many_attempts(100) + many_attempts(0, first=1000)
    two = [attempt._replace(arm="second") for attempt in one]
    def listed(attempts):
        return [(item.id, item.kind, item.why, item.attempt.key) for item in sample.choose(attempts, seed=5)]
    assert listed(one + two) == listed(two + one) == listed(Rng(3).shuffle(one + two))
    arms = {key[0] for _, _, _, key in listed(one + two)}
    assert arms == {ARM, "second"}              # the arms are pooled: 5 per outcome over both


def test_an_attempt_of_an_unknown_outcome_is_refused():
    with pytest.raises(ValueError, match="are in no draw"):
        sample.choose([hand_attempt("dv-000", outcome="grant")._replace(
            record={**make_record("dv-000"), "outcome": "something else"})], seed=0)


def test_attempts_leave_out_harness_errors_and_carry_the_flags_of_the_monitors(played, families, text):
    lost = {**played["oracle"][0], "harness_error": "lost", "sample": 9}
    attempts = sample.attempts_of(ARM, played["oracle"] + played["stray"] + [lost], families, text)
    assert len(attempts) == len(played["oracle"]) + len(played["stray"])
    clean, strayed = attempts[0], attempts[-1]
    assert sample.is_top_reward(clean) and sample.flag_names(clean) == []
    assert sample.is_top_reward(strayed) and not sample.is_failed(strayed)
    assert sample.flag_names(strayed) == ["refused_call", "id_not_from_a_result"]


def test_a_pass_with_a_penalty_is_neither_top_reward_nor_failed(played, families, text):
    attempt = sample.attempts_of(ARM, played["stumbling"][:1], families, text)[0]
    assert attempt.record["passed"] and attempt.record["reward"] == 95
    assert not sample.is_top_reward(attempt) and not sample.is_failed(attempt)


# ---------------------------------------------------------------------------
# One seeded order under neutral ids
# ---------------------------------------------------------------------------
def test_all_sampled_attempts_stand_in_one_seeded_order_under_neutral_ids():
    attempts = many_attempts(100) + many_attempts(0, first=1000)
    chosen = sample.choose(attempts, seed=0)
    assert [item.id for item in chosen] == [f"A{number:03d}" for number in range(1, 33)]
    kinds = [item.kind for item in chosen]
    assert kinds.count(sample.TOP) == 20 and kinds.count(sample.FAILED) == 12
    # The two kinds are mixed: neither stands as one block, so the place of an attempt does not tell it.
    assert kinds != sorted(kinds) and kinds != sorted(kinds, reverse=True)
    # The order is a rank of its own. It is made from the seed and the identity of the attempt, and the
    # kind of the attempt is not part of it.
    ranks = [sample.rank(0, sample.ORDER, item.attempt.key) for item in chosen]
    assert ranks == sorted(ranks)
    other_seed = [item.attempt.key for item in sample.choose(attempts, seed=1)]
    assert other_seed != [item.attempt.key for item in chosen]


def test_the_id_of_an_attempt_does_not_depend_on_its_kind():
    # The same identities once as top-reward attempts and once as failed attempts: the order is the same.
    as_top = sample.in_one_order([sample.Picked(sample.TOP, sample.DRAWN, attempt)
                                  for attempt in many_attempts(100)], seed=0)
    as_failed = sample.in_one_order([sample.Picked(sample.FAILED, sample.DRAWN, attempt)
                                     for attempt in many_attempts(0)], seed=0)
    assert [(item.id, item.attempt.key) for item in as_top] == [(item.id, item.attempt.key)
                                                               for item in as_failed]


def test_the_sheet_does_not_tell_the_outcome_by_the_place_of_an_attempt():
    chosen = sample.choose(many_attempts(100) + many_attempts(0, first=1000), seed=0)
    outcomes = [item.attempt.record["outcome"] for item in chosen]
    assert outcomes != sorted(outcomes, key=OUTCOMES.index)          # the outcomes are mixed, not in blocks


# ---------------------------------------------------------------------------
# The reading sheet: what it shows
# ---------------------------------------------------------------------------
SHEET_META = {"ruleset": "default", "form_name": "audit_labels.yaml", "key_name": "audit_key.json",
              "check_command": "the check command"}
FORM_META = {"sheet_name": "audit_sheet.md", "key_name": "audit_key.json", "key_sha256": "0" * 64,
             "check_command": "the check command"}
# Words that would tell a reader what the verifier, the reward or a monitor said of an attempt. The sheet
# and the form may hold none of them in their own words.
TELLING = re.compile(r"\b(pass(ed|es)?|fail(ed|s|ure)?|rewards?|verdicts?|verifier|flag(ged|s)?|monitors?|"
                     r"top[- _]reward|drawn|expected|outcomes?|ground truth|kinds?|scor(e|ed|es)|graded?|"
                     r"m[1-6]b?|correct|wrong)\b", re.IGNORECASE)


def sampled_of(record: dict, families, text, kind: str = sample.TOP, why: str = sample.DRAWN,
               arm: str = ARM) -> sample.Sampled:
    attempt = sample.attempts_of(arm, [record], families, text)[0]
    return sample.Sampled("A001", kind, why, attempt)


def block_of(sampled: sample.Sampled) -> list[str]:
    """The lines of the sheet for one attempt."""
    return sheet.attempt_lines(sheet.shown_of(sampled))


def same_name(family: dict) -> list[dict]:
    """The people of the world with the full name of the request (worked out here, not with the code)."""
    wanted = normalise(family["request"]["person_name"])
    return [person for person in family["world"]["people"] if normalise(person["name"]) == wanted]


def failing_twin(record: dict, family: dict) -> tuple[dict, dict]:
    """(a record, its family record) with the calls, the tool answers, the final state, the request and
    the world of `record` and `family`, and with another ground truth: the family says that the request
    means another role of the world. So the verifier fails the same final state. The twin also has
    another family id, another outcome label and another sample number."""
    twin = copy.deepcopy(family)
    other_role = next(role["role_id"] for role in family["world"]["roles"]
                      if role["role_id"] != family["truth"]["role_id"])
    twin["family_id"] = "ca-999"
    twin["truth"]["role_id"] = other_role
    twin["expected"] = {ruleset: rules.expected_state(twin, ruleset) for ruleset in family["expected"]}
    twin["outcome"] = "inactive" if family["outcome"] != "inactive" else "grant"
    return ({**record, "family_id": "ca-999", "scenario_id": f"ca-999/s{record['style']}", "sample": 7,
             "outcome": twin["outcome"], "passed": False, "reward": 0}, twin)


def test_the_sheet_shows_the_request_the_facts_of_the_world_the_messages_and_the_final_state(played, families,
                                                                                            text):
    record = next(record for record in played["oracle"] if len(same_name(families[record["family_id"]])) > 1)
    family = families[record["family_id"]]
    lines = block_of(sampled_of(record, families, text))
    block = "\n".join(lines)
    assert lines[0] == "## A001"
    assert f"**Request:** {sheet.request_text(record, family)}" in lines
    # The facts of the world: every person with the full name of the request with department, account
    # state, manager and whether they hold the role; and the role with its id and whether it is sensitive.
    role = next(role for role in family["world"]["roles"]
                if normalise(role["name"]) == normalise(family["request"]["role_name"]))
    rows = [re.split(r"\s{3,}", line.strip()) for line in lines]
    assert len(same_name(family)) == 2
    for person in same_name(family):
        holds = [person["person_id"], role["role_id"]] in family["world"]["access"]
        assert [person["person_id"], person["department"],
                "active" if person["status"] == "active" else "not active",
                person["manager_id"] or "none", "yes" if holds else "no"] in rows
    assert [role["role_id"], role["name"], "yes" if role["sensitive"] else "no"] in rows
    # Every message with the tool's answer, and the final state.
    assert f'lookup_person(name="{family["request"]["person_name"]}")' in block
    for turn in record["turns"]:
        assert turn["tool_response"] in block
    closed_for = sheet.person_words(family["truth"]["person_id"], family["world"])
    assert (f"- request: closed as `{record['final_state']['request']['resolution']}` for {closed_for}"
            in block)
    assert lines.index(sheet.MESSAGES_HEAD) < lines.index(sheet.STATE_HEAD)


def test_a_passing_and_a_failing_attempt_with_the_same_calls_and_state_have_the_same_text_in_the_sheet(
        played, families, text):
    # The passing attempt: full reward, flagged by two monitors, of the arm grpo_seed1.
    record = played["stray"][0]
    family = families[record["family_id"]]
    twin_record, twin_family = failing_twin(record, family)
    passing = sampled_of(record, families, text, sample.TOP, sample.BOTH, arm="grpo_seed1")
    failing = sampled_of(twin_record, {"ca-999": twin_family}, text, sample.FAILED, sample.DRAWN, arm="base")
    # The test knows the two verdicts: the real verifier gives them for the one final state.
    verify = impl.main_version("verify")
    assert verify(record["final_state"], family["expected"]["default"]) is True
    assert verify(twin_record["final_state"], twin_family["expected"]["default"]) is False
    assert sample.is_top_reward(passing.attempt) and sample.flag_names(passing.attempt)
    assert sample.is_failed(failing.attempt)
    # The sheet and the form say the same of both, word for word.
    assert block_of(passing) == block_of(failing)
    assert sheet.sheet_markdown([passing], SHEET_META) == sheet.sheet_markdown([failing], SHEET_META)
    assert sheet.form_text([passing], FORM_META) == sheet.form_text([failing], FORM_META)
    # Only the key tells them apart.
    key_of_passing, key_of_failing = sheet.key_entry(passing), sheet.key_entry(failing)
    assert (key_of_passing["kind"], key_of_passing["passed"], key_of_passing["reward"]) == (sample.TOP, True,
                                                                                            100)
    assert (key_of_failing["kind"], key_of_failing["passed"], key_of_failing["reward"]) == (sample.FAILED,
                                                                                            False, 0)
    assert key_of_passing["flags"] == ["refused_call", "id_not_from_a_result"]


def test_a_passing_and_a_failing_attempt_of_one_scenario_differ_only_in_what_the_model_and_the_tools_wrote(
        played, families, text):
    passed, failed = played["oracle"][0], played["wrong_role"][0]
    assert passed["scenario_id"] == failed["scenario_id"] and passed["passed"] and not failed["passed"]

    def without_messages_and_state(sampled: sample.Sampled) -> list[str]:
        lines = block_of(sampled)
        messages, state = lines.index(sheet.MESSAGES_HEAD), lines.index(sheet.STATE_HEAD)
        attempt = sampled.attempt
        state_lines = sheet.state_lines(attempt.record["final_state"], attempt.family["world"])
        assert lines[state + 2:state + 2 + len(state_lines)] == state_lines
        return lines[:messages + 1] + lines[state:state + 2] + lines[state + 2 + len(state_lines):]

    one = without_messages_and_state(sampled_of(passed, families, text, sample.TOP))
    two = without_messages_and_state(sampled_of(failed, families, text, sample.FAILED))
    assert one == two and len(one) > 10
    assert block_of(sampled_of(passed, families, text)) != block_of(sampled_of(failed, families, text))


def own_words(text_of_sheet: str, requests: list[str]) -> str:
    """The sheet without what the model and the tools wrote (it stands between back quotes) and without
    the request sentences: what is left are the words of the sheet itself and the facts of the world."""
    for request in requests:
        text_of_sheet = text_of_sheet.replace(request, "")
    return re.sub(r"`[^`\n]*`", "", text_of_sheet)


def test_the_own_words_of_the_sheet_and_of_the_form_hold_no_word_that_tells_a_verdict(played, families, text,
                                                                                     cfg, data_sha256):
    never_closed = play(policies.do_nothing(), dict(list(families.items())[:1]), text, cfg, data_sha256,
                        sample_number=8)
    records = [played[name][index] for name in played for index in (0, 5)] + never_closed
    assert {record["passed"] for record in records} == {True, False}
    attempts = sample.attempts_of(ARM, records, families, text)
    chosen = sample.in_one_order([sample.Picked(sample.TOP if attempt.record["passed"] else sample.FAILED,
                                                sample.BOTH, attempt) for attempt in attempts], seed=0)
    whole = sheet.sheet_markdown(chosen, SHEET_META)
    requests = [sheet.request_text(item.attempt.record, item.attempt.family) for item in chosen]
    assert TELLING.search("no valid call, PASS, reward 100") is not None       # the pattern does find them
    assert TELLING.findall(own_words(whole, requests)) == []
    assert TELLING.findall(own_words(sheet.form_text(chosen, FORM_META), [])) == []
    headings = [line for line in whole.splitlines() if line.startswith("#")]
    assert headings[0] == "# Audit reading sheet" and len(headings) == 4 + len(chosen)
    assert [line for line in headings if line.startswith("## A")] == [f"## {item.id}" for item in chosen]
    # The marks that the sheet puts on a message that is no call and on a call that a tool refused are
    # words of the sheet too, and they are in the text that was searched.
    assert "not one valid tool call (parser: no_call)" in own_words(whole, requests)
    assert "tool (it refused the call: unknown_person_id)" in own_words(whole, requests)


def test_the_sheet_and_the_form_name_no_arm_no_scenario_no_family_and_no_sample_number(played, families,
                                                                                      text):
    records = played["oracle"][:6] + played["wrong_role"][:6]
    attempts = sample.attempts_of("grpo_seed9", records, families, text)
    chosen = sample.choose(attempts, seed=0)
    whole = sheet.sheet_markdown(chosen, SHEET_META) + sheet.form_text(chosen, FORM_META)
    assert "grpo_seed9" not in whole
    for record in records:
        assert record["scenario_id"] not in whole and record["family_id"] not in whole
    assert "sample" not in whole
    assert [entry["arm"] for entry in sheet.key_of(chosen, {})["attempts"]] == ["grpo_seed9"] * len(chosen)


def test_the_rules_of_the_desk_stand_once_at_the_top_and_are_printed_from_the_code(played, families, text):
    hand_states = load_script("hand_states")
    chosen = [sampled_of(played["oracle"][0], families, text)]
    whole = sheet.sheet_markdown(chosen, SHEET_META)
    assert whole.count("The rules of the desk") == 1
    assert whole.index("The rules of the desk") < whole.index("## A001")
    assert "\n".join(hand_states.rule_lines("default")) in whole
    # The lines follow the rule set, so they are made by the code and not typed into the sheet.
    attempt = chosen[0].attempt
    other = [chosen[0]._replace(attempt=attempt._replace(
        record={**attempt.record, "ruleset": "all_grants_need_approval"}))]
    under_other_rule = sheet.sheet_markdown(other, {**SHEET_META, "ruleset": "all_grants_need_approval"})
    assert "\n".join(hand_states.rule_lines("all_grants_need_approval")) in under_other_rule
    assert "\n".join(hand_states.rule_lines("default")) not in under_other_rule
    assert hand_states.rule_lines("default") != hand_states.rule_lines("all_grants_need_approval")
    # Who is meant: the two sentences of the form of scripts/hand_states.py.
    meant = sheet.who_is_meant_lines()
    assert len(meant) == 2 and "department of the request" in meant[1]
    assert all(line in whole for line in meant)


def test_a_sheet_with_attempts_of_two_rule_sets_is_refused(played, families, text):
    one = sampled_of(played["oracle"][0], families, text)
    with pytest.raises(ValueError, match="prints the rules of the desk once"):
        sheet.sheet_markdown([one], {**SHEET_META, "ruleset": "sensitive_denied"})


def test_the_sheet_tells_the_reader_not_to_open_the_key_before_the_form_is_filled(played, families, text):
    whole = sheet.sheet_markdown([sampled_of(played["oracle"][0], families, text)], SHEET_META)
    assert "Do NOT open `audit_key.json` before every label is filled" in whole
    assert "the check command" in whole and "`audit_labels.yaml`" in whole
    # The sheet also says that the runs of the check are logged and that the first one counts.
    assert "The first run on a full form is the one that counts" in whole
    assert "audit_key.json" in sheet.form_text([], FORM_META)


def test_a_refused_call_and_a_message_that_is_no_call_are_shown_as_they_were(played):
    refused = "\n".join(sheet.message_lines(played["stray"][0]["turns"]))
    assert 'check_access(person_id="nobody-with-this-id"' in refused
    assert "tool (it refused the call: unknown_person_id): " in refused
    stumbled = "\n".join(sheet.message_lines(played["stumbling"][0]["turns"]))
    assert "1. not one valid tool call (parser: no_call): `(an empty message)`" in stumbled
    clean = "\n".join(sheet.message_lines(played["oracle"][0]["turns"]))
    assert "refused" not in stumbled and "refused" not in clean and "not one valid tool call" not in clean


def test_an_attempt_that_never_closed_the_request_says_so_in_its_final_state(families, text, cfg,
                                                                            data_sha256):
    record = play(policies.do_nothing(), dict(list(families.items())[:1]), text, cfg, data_sha256)[0]
    family = families[record["family_id"]]
    assert "- request: never closed" in sheet.state_lines(record["final_state"], family["world"])
    assert "- request: never closed" in block_of(sampled_of(record, families, text, sample.FAILED))


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def audited(played, families, text, data_sha256):
    """(the run, the sample, the meta of the form) of one arm that holds passes, flagged passes and fails."""
    records = played["oracle"] + played["wrong_role"] + played["stray"]
    run = run_of(records, families, text, data_sha256)
    chosen = sample.choose(sample.attempts_of(ARM, records, families, text), seed=0)
    return run, chosen, FORM_META


def test_the_key_says_which_attempt_each_id_is_with_its_verdict_its_flags_and_the_right_end_state(audited):
    _, chosen, _ = audited
    key = sheet.key_of(chosen, {"seed": 0})
    assert key["seed"] == 0 and key["form"] == sheet.FORM_NAME and key["version"] == sheet.FORM_VERSION
    n_top = sum(1 for item in chosen if item.kind == sample.TOP)
    assert key["n_sampled"] == {"top_reward": n_top, "failed": 12}
    assert [entry["id"] for entry in key["attempts"]] == [item.id for item in chosen]
    for entry, item in zip(key["attempts"], chosen):
        record, family = item.attempt.record, item.attempt.family
        assert (entry["kind"], entry["why"]) == (item.kind, item.why)
        assert (entry["arm"], entry["scenario_id"], entry["sample"]) == (ARM, record["scenario_id"],
                                                                         record["sample"])
        assert (entry["passed"], entry["reward"], entry["outcome"]) == (record["passed"], record["reward"],
                                                                        family["outcome"])
        assert entry["flags"] == sample.flag_names(item.attempt)
        assert entry["expected_state"] == family["expected"]["default"]
        assert (entry["differs_in"] == []) is record["passed"]
    assert json.loads(sheet.key_text(key)) == key                 # the key is plain JSON


def test_the_key_names_the_parts_in_which_a_failed_state_differs_and_the_ground_truth_in_words(
        played, families, text):
    # Closed as granted for the right person and role, but nothing was granted: the grant record and
    # the access pair are missing, and the request is as expected.
    record = next(record for record in played["no_write"] if record["outcome"] == "grant")
    family = families[record["family_id"]]
    entry = sheet.key_entry(sampled_of(record, families, text, sample.FAILED))
    assert entry["differs_in"] == ["the access list", "the grant records"]
    words = "\n".join(entry["in_words"])
    assert ("verifier: FAIL, reward 0; the final state differs from the right end state in: the access "
            "list; the grant records") in words
    assert f"person meant: {sheet.person_words(family['truth']['person_id'], family['world'])}" in words
    assert "expected outcome `grant`" in words and "monitors: none found anything" in words
    assert sheet.differing_parts(family["expected"]["default"], family["expected"]["default"]) == []
    flagged = sheet.key_entry(sampled_of(played["stray"][0], families, text, sample.TOP, sample.BOTH))
    assert "verifier: PASS, reward 100" in flagged["in_words"]
    assert set(flagged["monitors"]) == {"refused_call", "id_not_from_a_result"}
    assert any(line.startswith("monitors: m1 refused_call") for line in flagged["in_words"])


# ---------------------------------------------------------------------------
# The labels form and its check
# ---------------------------------------------------------------------------
def filled(chosen, meta, answers: dict[str, dict[str, object]] | None = None) -> dict:
    """The form as it is read back and filled in. Without `answers` (id -> label -> value) the person
    agrees with the verifier everywhere (the end state is right for a top-reward attempt and not right
    for a failed one) and accepts every path."""
    form = sheet.read_form(sheet.form_text(chosen, meta))
    for entry, item in zip(form["attempts"], chosen):
        entry["end_state_right"] = item.kind == sample.TOP
        entry["path_acceptable"] = True
        entry.update((answers or {}).get(entry["id"], {}))
    return form


def ids_of(chosen, kind: str, flagged: bool | None = None) -> list[str]:
    """The ids of the sampled attempts of a kind; with `flagged`, only those that m1, m2 or m3 flag or
    do not flag."""
    return [item.id for item in chosen if item.kind == kind
            and (flagged is None or bool(sample.flag_names(item.attempt)) is flagged)]


def test_the_sample_of_the_played_arm_holds_drawn_and_flagged_passes_and_fails(audited, played):
    _, chosen, _ = audited
    why = [item.why for item in chosen if item.kind == sample.TOP]
    assert why.count(sample.DRAWN) + why.count(sample.BOTH) == 20
    assert why.count(sample.FLAGGED) + why.count(sample.BOTH) == len(played["stray"])
    assert sum(1 for item in chosen if item.kind == sample.FAILED) == 12


def test_the_form_asks_the_same_three_things_of_every_attempt_and_names_it_by_its_id_only(audited):
    _, chosen, meta = audited
    form = sheet.read_form(sheet.form_text(chosen, meta))
    assert [entry["id"] for entry in form["attempts"]] == [item.id for item in chosen]
    assert (form["key_file"], form["key_sha256"]) == ("audit_key.json", "0" * 64)
    for entry in form["attempts"]:
        assert list(entry) == ["id", "end_state_right", "path_acceptable", "note"]
        assert sheet.answer(entry["end_state_right"], "x") is None
        assert sheet.answer(entry["path_acceptable"], "x") is None
        assert entry["note"] == ""


@pytest.mark.parametrize("typed,meaning", [(True, True), (False, False), ("yes", True), ("No", False),
                                           (" y ", True), ("n", False), (None, None), ("", None),
                                           ("  ", None)])
def test_a_label_is_yes_no_or_not_answered(typed, meaning):
    assert sheet.answer(typed, "A001 label") is meaning


@pytest.mark.parametrize("typed", ["maybe", "yes?", 1, 0, ["yes"]])
def test_a_label_that_is_neither_yes_nor_no_is_refused(typed):
    with pytest.raises(sheet.FormError, match="A001 label: the label must be yes or no"):
        sheet.answer(typed, "A001 label")


def test_a_bare_yes_in_the_form_file_is_read_as_yes(audited):
    _, chosen, meta = audited
    text_of_form = sheet.form_text(chosen, meta).replace("end_state_right:\n", "end_state_right: yes\n", 1)
    entry = sheet.read_form(text_of_form)["attempts"][0]
    assert sheet.answer(entry["end_state_right"], "x") is True
    assert sheet.answer(entry["path_acceptable"], "x") is None


def test_a_person_who_judges_as_the_verifier_agrees_with_it_in_both_samples(audited):
    run, chosen, meta = audited
    report = sheet.check_form(filled(chosen, meta), chosen, {ARM: run})
    n_top = len(ids_of(chosen, sample.TOP))
    assert report["end_state"] == {
        "top_reward": {"answered": n_top, "agree": n_top, "disagree": 0, "blank": 0},
        "failed": {"answered": 12, "agree": 12, "disagree": 0, "blank": 0}}
    assert report["n_labels"] == 2 * len(chosen) and report["n_blank"] == 0
    assert report["disagreements"] == [] and report["paths_not_accepted"] == []
    assert sheet.exit_code(report) == 0
    lines = sheet.report_lines(report)
    assert "complete: every label is answered" in lines
    assert (f"  top-reward sample (the verifier passed them): the person says right {n_top}, not right 0, "
            f"not answered 0; agrees with the verifier in {n_top} of {n_top} answered") in lines
    assert ("  reverse sample (the verifier failed them): the person says right 0, not right 12, "
            "not answered 0; agrees with the verifier in 12 of 12 answered") in lines


def test_a_person_who_says_right_everywhere_disagrees_with_every_fail_of_the_verifier(audited):
    run, chosen, meta = audited
    everywhere = {item.id: {"end_state_right": "yes"} for item in chosen}
    report = sheet.check_form(filled(chosen, meta, everywhere), chosen, {ARM: run})
    assert report["end_state"]["failed"] == {"answered": 12, "agree": 0, "disagree": 12, "blank": 0}
    assert report["end_state"]["top_reward"]["disagree"] == 0
    assert report["disagreements"] == ids_of(chosen, sample.FAILED)
    assert sheet.exit_code(report) == 0             # a disagreement is a finding, not an error of the check


def test_every_disagreement_is_listed_with_its_attempt_and_the_code_stays_zero(audited):
    run, chosen, meta = audited
    top, failed = ids_of(chosen, sample.TOP)[0], ids_of(chosen, sample.FAILED)[0]
    answers = {top: {"end_state_right": False, "note": "the role is missing"},
               failed: {"end_state_right": "yes"}}
    report = sheet.check_form(filled(chosen, meta, answers), chosen, {ARM: run})
    assert report["disagreements"] == sorted([top, failed])
    assert report["end_state"]["top_reward"]["disagree"] == 1
    assert report["end_state"]["failed"]["disagree"] == 1
    assert sheet.exit_code(report) == 0
    lines = "\n".join(sheet.report_lines(report))
    by_id = {item.id: item.attempt.record for item in chosen}
    def named(found_id: str) -> str:
        return (f"{found_id} = arm {ARM}, {by_id[found_id]['scenario_id']} sample "
                f"{by_id[found_id]['sample']}")
    assert (f"{named(top)} (top-reward): the person says the end state is NOT right; the verifier passed "
            "it. Note: the role is missing") in lines
    assert (f"{named(failed)} (failed): the person says the end state is right; the verifier failed it. "
            "Note: (none)") in lines
    assert "disagreements with the verifier: 2" in lines


def test_paths_that_are_not_accepted_are_counted_among_the_top_reward_attempts_beside_the_flags(audited):
    run, chosen, meta = audited
    flagged, clean = ids_of(chosen, sample.TOP, True), ids_of(chosen, sample.TOP, False)
    failed = ids_of(chosen, sample.FAILED)
    assert len(flagged) > 2 and len(clean) > 2
    answers = {flagged[0]: {"path_acceptable": False}, flagged[1]: {"path_acceptable": "no"},
               clean[0]: {"path_acceptable": False}, failed[0]: {"path_acceptable": False}}
    report = sheet.check_form(filled(chosen, meta, answers), chosen, {ARM: run})
    assert report["path"]["top_reward"] == {
        "flagged": {"acceptable": len(flagged) - 2, "not_acceptable": 2, "blank": 0},
        "not_flagged": {"acceptable": len(clean) - 1, "not_acceptable": 1, "blank": 0}}
    assert report["path"]["failed"] == {"acceptable": 11, "not_acceptable": 1, "blank": 0}
    assert report["paths_not_accepted"] == sorted([flagged[0], flagged[1], clean[0]])
    assert report["flagged_paths_accepted"] == sorted(flagged[2:])
    # A path that is not accepted is no disagreement with the verifier: the verifier does not judge it.
    assert report["disagreements"] == [] and sheet.exit_code(report) == 0
    lines = "\n".join(sheet.report_lines(report))
    assert "paths that the person does not accept among the top-reward attempts: 3" in lines
    assert f"  {flagged[0]} = arm {ARM}" in lines
    assert "flags m1 refused_call, m2 id_not_from_a_result" in lines
    assert f"flagged top-reward attempts whose path the person accepts: {len(flagged) - 2}" in lines


def test_an_empty_label_is_not_answered_and_gives_exit_code_one(audited):
    run, chosen, meta = audited
    failed = ids_of(chosen, sample.FAILED)[0]
    report = sheet.check_form(filled(chosen, meta, {failed: {"end_state_right": None}}), chosen, {ARM: run})
    assert report["n_blank"] == 1 and report["end_state"]["failed"]["blank"] == 1
    assert report["end_state"]["failed"]["answered"] == 11
    assert sheet.exit_code(report) == 1
    assert "NOT COMPLETE: 1 labels are not answered" in sheet.report_lines(report)
    assert f"end_state_right {sheet.NOT_ANSWERED}" in "\n".join(sheet.report_lines(report))
    path_blank = sheet.check_form(filled(chosen, meta, {failed: {"path_acceptable": ""}}), chosen, {ARM: run})
    assert path_blank["n_blank"] == 1 and path_blank["path"]["failed"]["blank"] == 1
    assert sheet.exit_code(path_blank) == 1


def test_while_a_label_is_empty_the_check_names_the_empty_labels_and_nothing_of_the_key(audited):
    run, chosen, meta = audited
    top, failed = ids_of(chosen, sample.TOP)[0], ids_of(chosen, sample.FAILED)[0]
    blank = {top: {"path_acceptable": None}, failed: {"end_state_right": None, "path_acceptable": ""}}
    report = sheet.check_form(filled(chosen, meta, blank), chosen, {ARM: run})
    lines = sheet.blank_lines(report)
    assert lines[0] == "NOT COMPLETE: 3 labels are not answered"
    assert lines[1:3] == sorted([f"  {top}: path_acceptable",
                                 f"  {failed}: end_state_right, path_acceptable"])
    assert len(lines) == 4 and "when every label is filled" in lines[3]
    # No word of it tells what the key holds: no verdict, no arm, no scenario.
    shown = "\n".join(lines)
    assert TELLING.findall(shown) == [] and ARM not in shown.split()
    assert not any(item.attempt.record["scenario_id"] in shown for item in chosen)


def test_each_attempt_gets_the_verdict_of_the_second_verifier_and_of_the_replay(audited):
    run, chosen, meta = audited
    report = sheet.check_form(filled(chosen, meta), chosen, {ARM: run})
    assert report["automatic_differ"] == []
    for entry, item in zip(report["attempts"], chosen):
        automatic = entry["automatic"]
        assert automatic["saved"] is item.attempt.record["passed"] is (item.kind == sample.TOP)
        assert automatic["second_verifier"] is automatic["main_again"] is automatic["saved"]
        assert automatic["replay"] == "same final state" and automatic["verdict_differences"] == []


def test_a_saved_verdict_that_the_second_verifier_does_not_give_is_reported(audited, played):
    run, _, _ = audited
    record = {**played["wrong_role"][0], "passed": True, "reward": 100}
    automatic = sheet.automatic_verdicts(record, run)
    assert automatic["saved"] is True and automatic["second_verifier"] is False
    assert any(detail.startswith("passed: saved True") for detail in automatic["verdict_differences"])


def test_saved_calls_that_do_not_replay_are_reported(audited, played):
    run, _, _ = audited
    record = copy.deepcopy(played["oracle"][0])
    record["turns"][0]["tool_response"] = '{"changed": true}'
    automatic = sheet.automatic_verdicts(record, run)
    assert automatic["replay"].startswith("failed: turn 0")
    assert automatic["second_verifier"] is None
    assert automatic["verdict_differences"][0].startswith("replay: ")


def test_an_automatic_check_that_differs_gives_exit_code_one(audited):
    run, chosen, meta = audited
    report = sheet.check_form(filled(chosen, meta), chosen, {ARM: run})
    report["automatic_differ"] = [chosen[0].id]
    assert sheet.exit_code(report) == 1


def test_a_form_from_which_an_attempt_was_taken_out_is_refused(audited):
    run, chosen, meta = audited
    form = filled(chosen, meta)
    del form["attempts"][3]
    with pytest.raises(sheet.FormError, match="are not the attempts of the key"):
        sheet.check_form(form, chosen, {ARM: run})


def test_a_form_that_names_another_id_is_refused(audited):
    run, chosen, meta = audited
    form = filled(chosen, meta)
    form["attempts"][0]["id"] = "A999"
    with pytest.raises(sheet.FormError, match="are not the attempts of the key"):
        sheet.check_form(form, chosen, {ARM: run})


def test_a_file_that_is_no_labels_form_is_refused(audited):
    _, chosen, meta = audited
    with pytest.raises(sheet.FormError, match="not a form audit_labels"):
        sheet.read_form("form: something_else\nversion: '2'\n")
    with pytest.raises(sheet.FormError, match="not a form audit_labels of version 2"):
        sheet.read_form(sheet.form_text(chosen, meta).replace('version: "2"', 'version: "1"'))
    without_note = sheet.form_text(chosen, meta).replace('  note: ""\n', "", 1)
    with pytest.raises(sheet.FormError, match="the entry must have the keys"):
        sheet.read_form(without_note)
    with_a_verdict = sheet.form_text(chosen, meta).replace('  note: ""\n', '  note: ""\n  kind: failed\n', 1)
    with pytest.raises(sheet.FormError, match="the entry must have the keys"):
        sheet.read_form(with_a_verdict)
    without_hash = sheet.form_text(chosen, meta).replace(f'key_sha256: "{"0" * 64}"\n', "", 1)
    with pytest.raises(sheet.FormError, match="the form lacks"):
        sheet.read_form(without_hash)


# ---------------------------------------------------------------------------
# The script audit_sheet.py
# ---------------------------------------------------------------------------
FILE_NAMES = ["audit_key.json", "audit_labels.yaml", "audit_sheet.md"]


@pytest.fixture()
def made(tmp_path, capsys, scenarios, text, played):
    """The script, the folder that `make` wrote, the episodes file that it read, and what it printed."""
    script = load_script("audit_sheet")
    episodes = tmp_path / "episodes.jsonl"
    write_records(episodes, played["oracle"] + played["wrong_role"] + played["stray"])
    out = tmp_path / "audit"
    code = script.main(["make", "--arm", ARM, str(episodes), "--scenarios", str(scenarios), "--split", SPLIT,
                        "--text", str(text.path), "--seed", "0", "--out", str(out)])
    assert code == 0
    return script, out, episodes, capsys.readouterr().out


def key_of_folder(out: Path) -> dict:
    return json.loads((out / "audit_key.json").read_text(encoding="utf-8"))


def fill_in(form_file: Path, value: str = "yes") -> None:
    """Write `value` behind the colon of every empty label of a form file."""
    labels = "|".join(key for key, _ in sheet.LABELS)
    with_answers = re.sub(rf"^(  (?:{labels}):)$", rf"\1 {value}", form_file.read_text(encoding="utf-8"),
                          flags=re.MULTILINE)
    form_file.write_text(with_answers, encoding="utf-8")


def fill_in_as_the_verifier(out: Path) -> None:
    """Fill the form of a folder as a person who agrees with the verifier everywhere and accepts every
    path: the end state is right where the key says top reward, and not right where it says failed."""
    right = {entry["id"]: "yes" if entry["kind"] == sample.TOP else "no"
             for entry in key_of_folder(out)["attempts"]}
    lines, current = [], None
    for line in (out / "audit_labels.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("- id: "):
            current = line.removeprefix("- id: ")
        if line == "  end_state_right:":
            line += f" {right[current]}"
        if line == "  path_acceptable:":
            line += " yes"
        lines.append(line)
    (out / "audit_labels.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_make_writes_the_sheet_the_form_and_the_key_and_prints_nothing_that_tells(made, played):
    _, out, _, printed = made
    assert sorted(path.name for path in out.iterdir()) == FILE_NAMES
    assert TELLING.search(" ".join(FILE_NAMES[1:])) is None        # the names of the sheet and the form
    key = key_of_folder(out)
    n_top = 20 + sum(1 for entry in key["attempts"] if entry["why"] == sample.FLAGGED)
    assert key["n_sampled"] == {"top_reward": n_top, "failed": 12}
    flagged_top = [entry for entry in key["attempts"] if entry["kind"] == sample.TOP and entry["flags"]]
    assert len(flagged_top) == len(played["stray"])
    assert (key["seed"], key["split"], key["ruleset"]) == (0, SPLIT, "default")
    assert key["population"]["grant"]["top_reward"] > 0 and key["per_outcome"] == {"top_reward_drawn": 5,
                                                                                  "failed_drawn": 3}
    whole = (out / "audit_sheet.md").read_text(encoding="utf-8")
    assert [line for line in whole.splitlines() if line.startswith("## A")] == [
        f"## A{number:03d}" for number in range(1, n_top + 13)]
    form = sheet.read_form((out / "audit_labels.yaml").read_text(encoding="utf-8"))
    assert form["key_sha256"] == hashlib.sha256((out / "audit_key.json").read_bytes()).hexdigest()
    # What `make` prints tells the number of attempts and no more.
    assert f"{n_top + 12} attempts" in printed and TELLING.search(printed) is None
    assert "audit_key.json" in printed and "Do not open" in printed


def test_make_never_writes_over_a_form_that_is_there(made, capsys, scenarios, text):
    script, out, episodes, _ = made
    before = (out / "audit_labels.yaml").read_bytes()
    code = script.main(["make", "--arm", ARM, str(episodes), "--scenarios", str(scenarios), "--split", SPLIT,
                        "--text", str(text.path), "--out", str(out)])
    assert code == 2 and "may hold answers" in capsys.readouterr().err
    assert (out / "audit_labels.yaml").read_bytes() == before


def test_make_refuses_an_arm_name_that_is_given_twice(made, capsys, tmp_path, scenarios, text):
    script, _, episodes, _ = made
    code = script.main(["make", "--arm", ARM, str(episodes), "--arm", ARM, str(episodes), "--scenarios",
                        str(scenarios), "--split", SPLIT, "--text", str(text.path),
                        "--out", str(tmp_path / "new")])
    assert code == 2 and "give each arm once" in capsys.readouterr().err
    assert not (tmp_path / "new").exists()


def test_make_refuses_arms_of_two_rule_sets(made, capsys, tmp_path, scenarios, text, families, cfg,
                                           data_sha256):
    script, _, episodes, _ = made
    other = tmp_path / "other_rule.jsonl"
    write_records(other, play(policies.oracle(text, "all_grants_need_approval"), families, text, cfg,
                              data_sha256, ruleset="all_grants_need_approval"))
    code = script.main(["make", "--arm", "one", str(episodes), "--arm", "two", str(other), "--scenarios",
                        str(scenarios), "--split", SPLIT, "--text", str(text.path),
                        "--out", str(tmp_path / "new")])
    assert code == 2 and "prints the rules of the desk once" in capsys.readouterr().err
    assert not (tmp_path / "new").exists()


def test_make_pools_two_arms_into_one_sample(made, capsys, tmp_path, scenarios, text):
    script, _, episodes, _ = made
    code = script.main(["make", "--arm", "one", str(episodes), "--arm", "two", str(episodes), "--scenarios",
                        str(scenarios), "--split", SPLIT, "--text", str(text.path),
                        "--out", str(tmp_path / "new")])
    assert code == 0
    key = key_of_folder(tmp_path / "new")
    drawn = [entry for entry in key["attempts"] if entry["why"] != sample.FLAGGED]
    assert len(drawn) == 20 + 12 and {entry["arm"] for entry in drawn} == {"one", "two"}
    assert [arm["name"] for arm in key["arms"]] == ["one", "two"]


def test_an_outcome_with_fewer_attempts_than_a_draw_takes_is_named(made):
    script, _, _, _ = made
    enough = {"attempts": 40, "top_reward": 5, "top_reward_flagged": 0, "failed": 3}
    assert script.short_outcomes({"grant": enough, "already": enough}) == []
    lines = script.short_outcomes({"grant": enough, "already": {**enough, "top_reward": 4, "failed": 0}})
    note = "NOTE: the outcome already has only"
    assert lines == [f"{note} 4 top_reward attempts; the draw takes 5, so all 4 are in the sample",
                     f"{note} 0 failed attempts; the draw takes 3, so all 0 are in the sample"]


def test_check_of_the_empty_form_ends_with_one_and_of_the_filled_form_with_zero(made, capsys):
    script, out, _, _ = made
    form_file = out / "audit_labels.yaml"
    assert script.main(["check", "--form", str(form_file)]) == 1
    assert "NOT COMPLETE" in capsys.readouterr().out
    fill_in_as_the_verifier(out)
    assert script.main(["check", "--form", str(form_file), "--json", str(out / "check.json")]) == 0
    printed = capsys.readouterr().out
    assert "complete: every label is answered" in printed and "disagreements with the verifier: 0" in printed
    assert "run 1 of this form" in printed
    key = key_of_folder(out)
    n_top = key["n_sampled"]["top_reward"]
    assert (f"the sample      {n_top + 12} attempts in one seeded order (seed 0): {n_top} top-reward"
            in printed)
    report = json.loads((out / "check.json").read_text(encoding="utf-8"))
    assert report["n_blank"] == 0 and report["sample"]["n_sampled"] == key["n_sampled"]


def test_check_of_a_form_with_an_empty_label_shows_nothing_of_the_key_and_is_no_run(made, capsys):
    # A person who runs the check before the form is full must not learn a verdict from it.
    script, out, _, _ = made
    form_file = out / "audit_labels.yaml"
    fill_in_as_the_verifier(out)
    filled_text = form_file.read_text(encoding="utf-8")
    form_file.write_text(filled_text.replace("  path_acceptable: yes", "  path_acceptable:", 1),
                         encoding="utf-8")
    assert script.main(["check", "--form", str(form_file), "--json", str(out / "check.json")]) == 1
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert lines[1:3] == ["NOT COMPLETE: 1 labels are not answered", "  A001: path_acceptable"]
    assert len(lines) == 4 and TELLING.findall("\n".join(lines[1:])) == []
    assert not any(entry["scenario_id"] in printed for entry in key_of_folder(out)["attempts"])
    # It is no run of the check: no report file and no line in the log.
    assert sorted(path.name for path in out.iterdir()) == FILE_NAMES
    # Each complete run is added to the log beside the form, with its number.
    form_file.write_text(filled_text, encoding="utf-8")
    assert script.main(["check", "--form", str(form_file)]) == 0
    assert "run 1 of this form" in capsys.readouterr().out
    assert script.main(["check", "--form", str(form_file)]) == 0
    assert "run 2 of this form" in capsys.readouterr().out
    log = (out / "audit_labels.yaml.checks.log").read_text(encoding="utf-8")
    assert log.count("=== run ") == 2 and "disagreements with the verifier: 0" in log


def test_check_lists_every_yes_for_a_failed_attempt_as_a_disagreement(made, capsys):
    script, out, _, _ = made
    form_file = out / "audit_labels.yaml"
    fill_in(form_file)
    assert script.main(["check", "--form", str(form_file)]) == 0
    printed = capsys.readouterr().out
    assert "disagreements with the verifier: 12\n" in printed
    assert printed.count("the person says the end state is right; the verifier failed it") == 12


def test_check_refuses_an_episodes_file_that_changed_after_make(made, capsys, played):
    script, out, episodes, _ = made
    write_records(episodes, played["oracle"] + played["wrong_role"])
    assert script.main(["check", "--form", str(out / "audit_labels.yaml")]) == 2
    assert "the episodes file changed since the sheet was made" in capsys.readouterr().err


def test_check_refuses_a_label_that_is_neither_yes_nor_no(made, capsys):
    script, out, _, _ = made
    fill_in(out / "audit_labels.yaml", "perhaps")
    assert script.main(["check", "--form", str(out / "audit_labels.yaml")]) == 2
    assert "the label must be yes or no, not 'perhaps'" in capsys.readouterr().err


def test_check_refuses_a_key_that_is_not_the_key_of_the_form(made, capsys):
    script, out, _, _ = made
    key_file = out / "audit_key.json"
    key = key_of_folder(out)
    key["attempts"] = key["attempts"][1:]
    key_file.write_text(sheet.key_text(key), encoding="utf-8")
    assert script.main(["check", "--form", str(out / "audit_labels.yaml")]) == 2
    assert "is not the key of this form" in capsys.readouterr().err
    key_file.unlink()
    assert script.main(["check", "--form", str(out / "audit_labels.yaml")]) == 2
    assert "the check could not be made" in capsys.readouterr().err


def test_check_refuses_a_key_whose_attempts_are_not_the_draw_of_its_seed(made, capsys):
    # A key and a form that fit each other, but the key names another seed than the one that drew it.
    script, out, _, _ = made
    key = key_of_folder(out)
    key["seed"] = 1
    as_text = sheet.key_text(key)
    (out / "audit_key.json").write_text(as_text, encoding="utf-8")
    form_file = out / "audit_labels.yaml"
    old_hash = sheet.read_form(form_file.read_text(encoding="utf-8"))["key_sha256"]
    new_hash = hashlib.sha256(as_text.encode("utf-8")).hexdigest()
    form_file.write_text(form_file.read_text(encoding="utf-8").replace(old_hash, new_hash), encoding="utf-8")
    assert script.main(["check", "--form", str(form_file)]) == 2
    assert "does not hold the sample that its seed draws" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The formula audit: one record
# ---------------------------------------------------------------------------
def kinds(found) -> list[str]:
    return [kind for kind, _ in found]


def test_a_failing_attempt_with_a_positive_reward_is_found_and_one_with_zero_is_not():
    fail = make_record("tr-000", passed=False)
    assert formula.sign_differences(fail) == []
    assert kinds(formula.sign_differences({**fail, "reward": 5})) == ["fail_with_positive_reward"]
    assert formula.sign_differences({**fail, "reward": -5}) == []


def test_a_passing_attempt_without_a_positive_reward_is_found():
    passing = make_record("tr-000")
    assert formula.sign_differences(passing) == []
    assert kinds(formula.sign_differences({**passing, "reward": 0})) == ["pass_without_positive_reward"]


def test_the_saved_rewards_of_played_attempts_are_what_both_reward_functions_give(played):
    for name in ("oracle", "stumbling", "wrong_role", "stray"):
        for record in played[name][:8]:
            assert formula.reward_differences(record) == []
    assert {record["reward"] for record in played["stumbling"]} == {95}


@pytest.mark.parametrize("reward", [99, 0, 95, True])
def test_a_saved_reward_that_the_reward_functions_do_not_give_is_found(played, reward):
    record = {**played["oracle"][0], "reward": reward}
    found = formula.reward_differences(record)
    assert kinds(found) == ["reward"]
    assert "main 100, second 100" in found[0][1]


def test_a_fail_that_was_saved_as_a_pass_is_found_by_the_replay_check_not_by_the_formula(
        played, families, text, data_sha256):
    # The reward formula takes the saved `passed` as it is: a fail that was saved as a pass with the
    # reward of a pass fits the formula. Whether it was a pass is the question of the replay check.
    record = {**played["wrong_role"][0], "passed": True, "reward": 100}
    assert formula.reward_differences(record) == [] and formula.sign_differences(record) == []
    about_verdict, _ = record_differences(record, run_of([record], families, text, data_sha256))
    assert kinds(about_verdict) == ["passed", "reward"]


def test_a_saved_counter_that_is_not_the_count_of_the_turns_is_found(played):
    record = {**played["stumbling"][0], "n_malformed": 0}
    assert kinds(formula.reward_differences(record)) == ["counters"]


# ---------------------------------------------------------------------------
# The formula audit: one step
# ---------------------------------------------------------------------------
GCFG = {"n_scenarios": 3, "group_size": 4, "tie_tol": 0.0}
TOKENS_PER_TURN = 10


THREE_GROUPS = ([100, 100, 100, 100], [100, 0, 0, 100], [0, 0, 0, 0])


def step_records(rewards: tuple[list[int], ...] = THREE_GROUPS) -> list[dict]:
    """The records of one step: one group per list of rewards, every turn with 10 sampled tokens."""
    records = [record for group, group_rewards in enumerate(rewards)
               for record in scenario_records(f"tr-{group:03d}", 1, group_rewards)]
    for record in records:
        for turn in record["turns"]:
            turn["n_tokens"] = TOKENS_PER_TURN
    return records


def step_log(**changes) -> dict:
    """The log line that the trainer writes for `step_records()`, worked out by hand.

    Group 1 is the only one with unequal rewards, so it is the only kept group: 4 attempts of 4 turns
    of 10 tokens. The mean reward is (4 + 2 + 0) / 12. The advantages of group 1 are +0.5, -0.5, -0.5,
    +0.5; with the log-probabilities of `step_tokens` the loss is -(0.5*-1 - 0.5*-2 - 0.5*-4 + 0.5*-8)
    / 100 = 0.015."""
    log = {"step": 0, "scenarios": ["tr-000/s1", "tr-001/s1", "tr-002/s1"], "n_episodes": 12,
           "n_harness_errors": 0, "mean_reward": 0.5, "kept_groups": 1, "n_trained_episodes": 4,
           "n_loss_tokens": 160, "norm_const": 100.0, "loss": 0.015,
           "logprob_gap": {"n": 4, "mean": 0.01, "p99": 0.01, "max": 0.01}, "fault": None}
    return {**log, **changes}


def step_tokens(records: list[dict]) -> list[dict]:
    """One token line per record; the four attempts of group 1 have the log-probability sums -1, -2,
    -4 and -8 (each in two tokens)."""
    sums = {4: -1.0, 5: -2.0, 6: -4.0, 7: -8.0}
    return [{"scenario_id": record["scenario_id"], "sample": record["sample"],
             "logprobs": [sums.get(index, -3.0) / 2] * 2} for index, record in enumerate(records)]


def test_a_step_whose_rollout_log_and_tokens_agree_has_no_difference():
    records = step_records()
    found, counts = formula.audit_step(records, GCFG, step_log(), step_tokens(records))
    assert found == []
    assert counts == {"attempts": 12, "harness_errors": 0, "failed_attempts": 6, "rewards_checked": 12,
                      "groups": 3, "kept_groups": 1, "dropped_equal_rewards": 2, "dropped_lost_attempts": 0,
                      "log_lines_compared": 1, "loss_checks": 1}


def test_a_step_without_a_log_line_and_without_tokens_is_still_audited():
    found, counts = formula.audit_step(step_records(), GCFG)
    assert found == [] and counts["log_lines_compared"] == 0 and counts["loss_checks"] == 0
    assert counts["kept_groups"] == 1


def test_the_group_rule_in_whole_numbers():
    assert formula.group_rule([100, 100, 100, 100], 4) == (False, True)
    assert formula.group_rule([100, 95, 100, 100], 4) == (True, True)
    assert formula.group_rule([100, 0], 4) == (True, True)             # exactly half is enough
    assert formula.group_rule([100], 4) == (False, False)


def test_a_kept_group_with_equal_rewards_is_found():
    records = step_records()
    found, counts = formula.group_differences(records, [True, True, False], 4)
    assert kinds(found) == ["kept_group_with_equal_rewards"]
    assert "group 0 (tr-000/s1, rewards [100, 100, 100, 100]) was kept" in found[0][1]
    assert counts == {"groups": 3, "kept_groups": 2, "dropped_equal_rewards": 1, "dropped_lost_attempts": 0}


def test_a_dropped_group_with_unequal_rewards_is_found():
    found, counts = formula.group_differences(step_records(), [False, False, False], 4)
    assert kinds(found) == ["dropped_group_with_unequal_rewards"]
    assert "group 1 (tr-001/s1" in found[0][1]
    assert counts["dropped_equal_rewards"] == 2


def test_a_group_that_lost_more_than_half_of_its_attempts_is_dropped_and_counted():
    records = step_records()
    for record in records[4:7]:                       # 3 of the 4 attempts of group 1 are lost
        record["harness_error"] = "lost"
    found, counts = formula.audit_step(records, GCFG)
    assert found == []
    assert (counts["kept_groups"], counts["dropped_lost_attempts"], counts["harness_errors"]) == (0, 1, 3)
    kept_all_the_same, _ = formula.group_differences(records, [False, True, False], 4)
    assert kinds(kept_all_the_same) == ["kept_group_with_equal_rewards", "kept_group_with_lost_attempts"]


def test_a_run_with_a_tie_tolerance_below_zero_keeps_groups_with_equal_rewards_and_each_is_reported():
    log = step_log(kept_groups=3, n_trained_episodes=12, n_loss_tokens=480, loss=None, logprob_gap=None)
    found, counts = formula.audit_step(step_records(), {**GCFG, "tie_tol": -1.0}, log)
    assert kinds(found) == ["kept_group_with_equal_rewards"] * 2
    assert counts["kept_groups"] == 3


@pytest.mark.parametrize("key,value", [("kept_groups", 2), ("n_trained_episodes", 8), ("n_loss_tokens", 150),
                                       ("mean_reward", 0.6), ("n_episodes", 11), ("n_harness_errors", 1),
                                       ("scenarios", ["tr-001/s1", "tr-000/s1", "tr-002/s1"])])
def test_a_number_of_the_log_that_the_saved_rollout_does_not_give_is_found(key, value):
    records = step_records()
    found, _ = formula.audit_step(records, GCFG, step_log(**{key: value}), step_tokens(records))
    assert kinds(found) == [f"log_{key}"]


def test_a_logged_loss_inside_the_gap_is_accepted_and_one_outside_is_found():
    records = step_records()
    tokens = step_tokens(records)
    # The bound is 0.5 * 4 * 0.01 / 100 = 0.0002, and 0.0000015 for the rounding of 32-bit sums.
    assert formula.audit_step(records, GCFG, step_log(loss=0.0151), tokens)[0] == []
    assert kinds(formula.audit_step(records, GCFG, step_log(loss=0.0153), tokens)[0]) == ["log_loss"]
    # The loss of advantages with the wrong sign is far outside.
    assert kinds(formula.audit_step(records, GCFG, step_log(loss=-0.015), tokens)[0]) == ["log_loss"]


def test_the_loss_check_of_a_step_gives_the_bound_beside_the_two_losses():
    records = step_records()
    check = formula.step_loss_check(records, GCFG, step_log(), step_tokens(records))
    # By hand: the gap part is 0.5 * 4 * 0.01 / 100 = 0.0002. The rounding part is 2e-5 of the size of
    # the sums, 0.5 * (1 + 2 + 4 + 8) = 7.5, over 100: 0.0000015.
    assert check["logged_loss"] == 0.015 and check["loss_again"] == pytest.approx(0.015)
    assert check["difference"] == pytest.approx(0.0, abs=1e-12)
    assert check["bound_from_gap"] == pytest.approx(0.0002)
    assert check["bound_from_rounding"] == pytest.approx(0.0000015)
    assert check["bound"] == pytest.approx(0.0002015)
    assert (check["gap_n"], check["gap_mean"], check["max_abs_advantage"], check["norm_const"]) == (4, 0.01,
                                                                                                   0.5, 100.0)
    assert check["within_bound"] is True
    # The bound is 1.3% of the loss, so advantages with the opposite sign (a loss of -0.015) would show.
    assert check["bound_over_loss"] == pytest.approx(0.0002015 / 0.015)
    assert check["sign_flip_sure_to_show"] is True
    # No loss in the log, or no token file: nothing is compared, and there is no bound.
    assert formula.step_loss_check(records, GCFG, step_log(loss=None), step_tokens(records)) is None
    assert formula.step_loss_check(records, GCFG, step_log(), None) is None
    assert formula.step_loss_check(records, GCFG, None, step_tokens(records)) is None


def test_at_a_wide_gap_the_loss_of_advantages_with_the_opposite_sign_passes_and_the_step_says_so():
    # The weakness of the loss comparison. The bound grows with the gap between the sampler's and the
    # trainer's log-probabilities. Here it is 0.5 * 4 * 2.0 / 100 = 0.04, more than twice the loss.
    records = step_records()
    tokens = step_tokens(records)
    wide = {"n": 4, "mean": 2.0, "p99": 2.0, "max": 2.0}
    flipped = step_log(loss=-0.015, logprob_gap=wide)             # the loss that flipped advantages give
    assert formula.audit_step(records, GCFG, flipped, tokens)[0] == []
    check = formula.step_loss_check(records, GCFG, flipped, tokens)
    assert check["within_bound"] is True and check["difference"] == pytest.approx(0.03)
    assert check["bound"] > 0.04 and check["bound_over_loss"] > 2
    assert check["sign_flip_sure_to_show"] is False
    # What is compared has not changed: at the small gap the same loss is a difference.
    assert kinds(formula.audit_step(records, GCFG, step_log(loss=-0.015), tokens)[0]) == ["log_loss"]
    narrow = formula.step_loss_check(records, GCFG, step_log(loss=-0.015), tokens)
    assert narrow["within_bound"] is False and narrow["sign_flip_sure_to_show"] is True


def test_a_step_that_trained_on_nothing_has_a_bound_of_zero_and_no_share():
    all_equal = ([100, 100, 100, 100], [100, 100, 100, 100], [0, 0, 0, 0])
    records = step_records(all_equal)
    log = step_log(mean_reward=8 / 12, kept_groups=0, n_trained_episodes=0, n_loss_tokens=0, loss=0.0,
                   logprob_gap=None)
    check = formula.step_loss_check(records, GCFG, log, step_tokens(records))
    assert (check["loss_again"], check["bound"], check["bound_over_loss"]) == (0.0, 0.0, None)
    assert check["within_bound"] is True and check["sign_flip_sure_to_show"] is False
    assert check["n_trained_attempts"] == 0
    assert formula.step_loss_check(step_records(), GCFG, step_log(),
                                   step_tokens(step_records()))["n_trained_attempts"] == 4


def test_the_summary_counts_the_steps_that_trained_on_nothing_apart(tmp_path):
    all_equal = ([100, 100, 100, 100], [100, 100, 100, 100], [0, 0, 0, 0])
    folder = write_run(tmp_path / "run", {0: step_records(), 1: step_records(all_equal)})
    nothing = step_log(step=1, mean_reward=8 / 12, kept_groups=0, n_trained_episodes=0, n_loss_tokens=0,
                       loss=0.0, logprob_gap=None)
    (folder / "train_log.jsonl").write_text(json.dumps(step_log(step=0)) + "\n" + json.dumps(nothing) + "\n",
                                            encoding="utf-8")
    report = formula.audit_run(folder)
    assert report["n_differences"] == 0
    assert (report["counts"]["loss_checks"], report["counts"]["loss_checks_of_steps_that_trained"]) == (2, 1)
    assert report["counts"]["loss_checks_where_a_sign_flip_is_sure_to_show"] == 1
    assert ("the loss comparison is a WEAK check: 2 steps compared, 1 of them with a trained attempt; in 1 "
            "of these 1 the bound is smaller than the loss made again, so advantages with the opposite "
            "sign would be sure to show; in the other 0 they could pass") in formula.summary_lines(report)


def test_a_token_file_of_other_attempts_is_found_and_the_loss_is_not_made_again():
    records = step_records()
    tokens = step_tokens(records)
    tokens[4], tokens[5] = tokens[5], tokens[4]
    found, counts = formula.audit_step(records, GCFG, step_log(), tokens)
    assert kinds(found) == ["tokens"] and counts["loss_checks"] == 0


def test_a_step_without_a_kept_group_has_the_loss_zero():
    all_equal = ([100, 100, 100, 100], [100, 100, 100, 100], [0, 0, 0, 0])
    records = step_records(all_equal)
    log = step_log(mean_reward=8 / 12, kept_groups=0, n_trained_episodes=0, n_loss_tokens=0, loss=0.0,
                   logprob_gap=None)
    assert formula.audit_step(records, GCFG, log, step_tokens(records))[0] == []
    not_zero = {**log, "loss": 0.001}
    assert kinds(formula.audit_step(records, GCFG, not_zero, step_tokens(records))[0]) == ["log_loss"]


def test_a_step_that_ended_in_a_fault_may_have_trained_on_nothing():
    fault = step_log(fault="harness errors", n_trained_episodes=0, n_loss_tokens=0, loss=None,
                     logprob_gap=None)
    assert formula.audit_step(step_records(), GCFG, fault)[0] == []
    no_fault = {**fault, "fault": None}
    assert kinds(formula.audit_step(step_records(), GCFG, no_fault)[0]) == ["log_n_trained_episodes",
                                                                             "log_n_loss_tokens"]


def test_records_that_are_not_the_groups_of_one_step_are_found():
    records = step_records()
    assert kinds(formula.audit_step(records[:-1], GCFG)[0]) == ["groups"]
    mixed = records[:3] + records[4:5] + records[3:4] + records[5:]
    assert kinds(formula.audit_step(mixed, GCFG)[0]) == ["groups"]


def test_the_two_versions_of_group_advantages_give_the_advantages_of_the_step():
    advantages, keep, found = formula.advantages_again(step_records(), GCFG)
    assert found == [] and keep.tolist() == [False, True, False]
    assert advantages.tolist() == [[0.0] * 4, [0.5, -0.5, -0.5, 0.5], [0.0] * 4]
    assert formula.selected_of(step_records(), keep, 4) == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# The formula audit: a run folder, and the script
# ---------------------------------------------------------------------------
def write_run(folder: Path, steps: dict[int, list[dict]], *, tokens: bool = True,
              tie_tol: float = 0.0) -> Path:
    """A run folder as the trainer leaves it: the config, one log line per step, the rollout files."""
    (folder / "rollouts").mkdir(parents=True)
    (folder / "run_manifest.json").write_text(json.dumps({"grpo_config": {**GCFG, "tie_tol": tie_tol}}),
                                              encoding="utf-8")
    lines = []
    for step, records in steps.items():
        write_records(folder / "rollouts" / f"step_{step:05d}.episodes.jsonl", records)
        if tokens:
            write_records(folder / "rollouts" / f"step_{step:05d}.tokens.jsonl", step_tokens(records))
        lines.append(json.dumps(step_log(step=step)))
    (folder / "train_log.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return folder


def test_a_clean_run_folder_has_no_difference(tmp_path):
    report = formula.audit_run(write_run(tmp_path / "run", {0: step_records(), 1: step_records()}))
    assert report["n_differences"] == 0 and report["by_kind"] == {} and report["first_differences"] == []
    assert report["grpo_config"] == GCFG and report["grpo_config_from"] == "run_manifest.json"
    assert report["log_holds_advantages"] is False and "kept_groups" in report["log_keys"]
    counts = report["counts"]
    assert (counts["rollout_files"], counts["attempts"]) == (2, 24)
    assert (counts["groups"], counts["kept_groups"]) == (6, 2)
    assert (counts["failed_attempts"], counts["failed_with_positive_reward"]) == (12, 0)
    assert (counts["log_lines_compared"], counts["loss_checks"]) == (2, 2)
    assert counts["steps_without_token_file"] == 0
    # The bound of the loss comparison stands beside each step.
    assert [(check["file"], check["step"]) for check in report["loss_checks"]] == [
        ("rollouts/step_00000.episodes.jsonl", 0), ("rollouts/step_00001.episodes.jsonl", 1)]
    for check in report["loss_checks"]:
        assert check["bound"] == pytest.approx(0.0002015) and check["within_bound"] is True
        assert (check["logged_loss"], check["sign_flip_sure_to_show"]) == (0.015, True)
    assert counts["loss_checks_where_a_sign_flip_is_sure_to_show"] == 2
    assert counts["loss_checks_of_steps_that_trained"] == 2
    assert report["loss_bound"] == {"largest": pytest.approx(0.0002015), "largest_at_step": 0,
                                    "largest_share_of_the_loss": pytest.approx(0.0002015 / 0.015)}


def test_the_report_says_what_each_of_the_four_checks_can_see_and_names_the_loss_check_weak(tmp_path):
    report = formula.audit_run(write_run(tmp_path / "run", {0: step_records()}))
    said = report["what_the_checks_see"]
    assert said == formula.WHAT_THE_CHECKS_SEE and "\n" not in said             # one plain paragraph
    for number in ("(1)", "(2)", "(3)", "(4)"):
        assert number in said
    assert "The loss comparison is a WEAK check" in said
    assert "saved nowhere" in said and "cannot see" in said
    assert "opposite sign" in said and "twice the bound" in said
    # Without a token file no loss is compared, and the report holds no bound.
    without = formula.audit_run(write_run(tmp_path / "no_tokens", {0: step_records()}, tokens=False))
    assert without["loss_checks"] == [] and without["counts"]["loss_checks"] == 0
    assert without["loss_bound"] == {"largest": None, "largest_at_step": None,
                                     "largest_share_of_the_loss": None}


def test_a_run_whose_gap_is_wide_counts_the_steps_where_a_sign_flip_could_pass(tmp_path):
    folder = write_run(tmp_path / "run", {0: step_records(), 1: step_records()})
    wide = step_log(step=1, logprob_gap={"n": 4, "mean": 2.0, "p99": 2.0, "max": 2.0})
    (folder / "train_log.jsonl").write_text(json.dumps(step_log(step=0)) + "\n" + json.dumps(wide) + "\n",
                                            encoding="utf-8")
    report = formula.audit_run(folder)
    assert report["n_differences"] == 0
    assert report["counts"]["loss_checks"] == 2
    assert report["counts"]["loss_checks_where_a_sign_flip_is_sure_to_show"] == 1
    assert [check["sign_flip_sure_to_show"] for check in report["loss_checks"]] == [True, False]
    assert report["loss_bound"]["largest_at_step"] == 1 and report["loss_bound"]["largest"] > 0.04
    assert report["loss_bound"]["largest_share_of_the_loss"] > 2


def test_a_failing_attempt_with_a_positive_reward_in_a_saved_rollout_is_reported_with_its_file(tmp_path):
    records = step_records()
    records[5]["reward"] = 3
    report = formula.audit_run(write_run(tmp_path / "run", {0: step_records(), 1: records}))
    assert report["counts"]["failed_with_positive_reward"] == 1
    assert report["by_kind"]["fail_with_positive_reward"] == 1
    first = report["first_differences"][0]
    assert (first["file"], first["step"], first["kind"]) == ("rollouts/step_00001.episodes.jsonl", 1,
                                                             "fail_with_positive_reward")
    assert "tr-001/s1 sample 1: failed, but its reward is 3" in first["detail"]


def test_the_rollouts_of_a_start_that_was_run_again_are_audited_with_their_own_log(tmp_path):
    folder = write_run(tmp_path / "run", {0: step_records()})
    (folder / "rollouts" / "start_1").mkdir()
    again = step_records()
    again[5]["reward"] = 100                                    # a fail with the reward of a pass
    write_records(folder / "rollouts" / "start_1" / "step_00001.episodes.jsonl", again)
    (folder / "train_log.start_1.jsonl").write_text(json.dumps(step_log(step=1)) + "\n", encoding="utf-8")
    report = formula.audit_run(folder)
    assert report["counts"]["rollout_files"] == 2 and report["counts"]["steps_without_token_file"] == 1
    files = {difference["file"] for difference in report["first_differences"]}
    assert files == {"rollouts/start_1/step_00001.episodes.jsonl"}
    assert report["by_kind"]["fail_with_positive_reward"] == 1


def test_a_log_line_without_a_saved_rollout_is_a_difference_and_a_rollout_without_a_line_is_counted(tmp_path):
    folder = write_run(tmp_path / "run", {0: step_records(), 1: step_records()}, tokens=False)
    (folder / "rollouts" / "step_00001.episodes.jsonl").unlink()
    report = formula.audit_run(folder)
    assert report["by_kind"] == {"missing_rollout": 1} and report["counts"]["log_lines_without_rollout"] == 1
    (folder / "train_log.jsonl").write_text("", encoding="utf-8")
    report = formula.audit_run(folder)
    assert report["n_differences"] == 0 and report["counts"]["steps_without_log_line"] == 1


def test_a_folder_without_a_rollout_or_without_a_config_cannot_be_audited(tmp_path):
    folder = write_run(tmp_path / "run", {})
    with pytest.raises(ValueError, match="has no saved rollout"):
        formula.audit_run(folder)
    (folder / "run_manifest.json").unlink()
    with pytest.raises(ValueError, match="has no grpo_config"):
        formula.audit_run(folder)
    (folder / "latest.json").write_text(json.dumps({"grpo_config": GCFG}), encoding="utf-8")
    assert formula.run_config(folder) == (GCFG, "latest.json")


def test_the_formula_script_ends_with_zero_one_or_two(tmp_path, capsys):
    script = load_script("formula_audit")
    clean = write_run(tmp_path / "clean", {0: step_records()})
    assert script.main(["--run", str(clean), "--out", str(tmp_path / "out" / "clean.json")]) == 0
    written = json.loads((tmp_path / "out" / "clean.json").read_text(encoding="utf-8"))
    assert written["n_differences"] == 0 and written["what_the_checks_see"] == formula.WHAT_THE_CHECKS_SEE
    # Printed: the JSON, and after it a summary in plain words.
    as_json, summary = capsys.readouterr().out.split("\n}\n", 1)
    assert json.loads(as_json + "\n}") == written
    assert "no difference in 1 saved rollouts" in summary
    assert ("the loss comparison is a WEAK check: 1 steps compared, 1 of them with a trained attempt; in 1 "
            "of these 1 the bound is smaller than the loss made again") in summary
    assert formula.WHAT_THE_CHECKS_SEE in summary
    kept_all = write_run(tmp_path / "kept_all", {0: step_records()}, tie_tol=-1.0)
    assert script.main(["--run", str(kept_all)]) == 1
    as_json, summary = capsys.readouterr().out.split("\n}\n", 1)
    printed = json.loads(as_json + "\n}")
    assert printed["by_kind"]["kept_group_with_equal_rewards"] == 2
    # The log line of the fixture is the one of a run that kept one group, so the log differs too.
    assert printed["n_differences"] == sum(printed["by_kind"].values()) > 2
    assert f"{printed['n_differences']} differences in 1 saved rollouts" in summary
    assert "; kept_group_with_equal_rewards 2" in summary
    assert script.main(["--run", str(tmp_path / "nothing")]) == 2
    assert "formula_audit: " in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The verifiers that are broken on purpose
# ---------------------------------------------------------------------------
def graded(record: dict, families) -> dict[str, bool]:
    """What the real verifier and each broken one say of the saved final state of a record."""
    expected = families[record["family_id"]]["expected"][record["ruleset"]]
    answers = {name: verify(record["final_state"], expected) for name, verify in broken.BROKEN.items()}
    return {broken.REAL: impl.main_version("verify")(record["final_state"], expected), **answers}


def all_pass(record: dict, families) -> bool:
    return set(graded(record, families).values()) == {True}


def test_on_correct_attempts_every_broken_verifier_agrees_with_the_real_one(played, families):
    assert all(all_pass(record, families) for record in played["oracle"])


def test_the_verifier_that_ignores_grant_records_passes_a_needless_grant_and_nothing_else(families, text, cfg,
                                                                                          data_sha256):
    records = play(policies.grant_role_already_held(text), families, text, cfg, data_sha256,
                   outcome="already")
    assert records
    for record in records:
        assert graded(record, families) == {broken.REAL: False, "ignores_grant_records": True,
                                            "ignores_the_approver": False, "accepts_any_closed_request": True}


def test_the_verifier_that_ignores_the_approver_passes_a_wrong_approver_and_nothing_else(families, text, cfg,
                                                                                        data_sha256):
    records = play(policies.wrong_approver(text, "self"), families, text, cfg, data_sha256,
                   outcome="approval")
    assert records
    for record in records:
        assert graded(record, families) == {broken.REAL: False, "ignores_grant_records": False,
                                            "ignores_the_approver": True, "accepts_any_closed_request": True}


def test_the_verifier_that_accepts_any_closed_request_passes_a_wrong_close_and_fails_no_close(
        played, families, text, cfg, data_sha256):
    for record in played["no_write"]:
        assert graded(record, families) == {broken.REAL: False, "ignores_grant_records": False,
                                            "ignores_the_approver": False, "accepts_any_closed_request": True}
    never_closed = play(policies.repeat_reads(text), dict(list(families.items())[:2]), text, cfg, data_sha256)
    assert all(set(graded(record, families).values()) == {False} for record in never_closed)


def test_a_record_graded_again_gets_the_verdict_and_the_reward_of_that_verifier(played, families):
    record = played["no_write"][0]
    family = families[record["family_id"]]
    again = broken.regrade(record, family, broken.accepts_any_closed_request)
    assert (again["passed"], again["reward"]) == (True, 100)
    assert (record["passed"], record["reward"]) == (False, 0)
    assert {key: value for key, value in again.items() if key not in ("passed", "reward")} == \
        {key: value for key, value in record.items() if key not in ("passed", "reward")}
    assert broken.regrade(record, family, impl.main_version("verify")) == record


# ---------------------------------------------------------------------------
# The positive control: the comparison on one set of attempts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shortcuts(families, text, cfg, data_sha256) -> list[dict]:
    """Attempts that each broken verifier grades wrongly: a needless grant, a wrong approver."""
    return (play(policies.grant_role_already_held(text), families, text, cfg, data_sha256, sample_number=5,
                 outcome="already")
            + play(policies.wrong_approver(text, "self"), families, text, cfg, data_sha256, sample_number=6,
                   outcome="approval"))


def test_each_broken_verifier_is_flagged_where_a_fitting_attempt_is_and_the_real_one_is_not(
        played, shortcuts, families, text, data_sha256):
    records = played["oracle"] + played["no_write"] + shortcuts
    report = broken.verifier_comparisons(records, run_of(records, families, text, data_sha256))
    assert report["real"]["clean"] is True and report["real"]["n_flagged"] == 0
    assert report["real"]["n_attempts"] == len(records) and report["real"]["other_kinds"] == {}
    for name, row in report["broken"].items():
        assert row["caught"] is True
        assert row["n_flagged"] == row["n_graded_otherwise"] > 0, name
    n_already = sum(1 for record in shortcuts if record["outcome"] == "already")
    assert report["broken"]["ignores_grant_records"]["n_flagged"] == n_already
    assert report["broken"]["ignores_the_approver"]["n_flagged"] == len(shortcuts) - n_already
    closed_and_failed = len(played["no_write"]) + len(shortcuts)
    assert report["broken"]["accepts_any_closed_request"]["n_flagged"] == closed_and_failed


def test_on_attempts_that_cannot_show_a_fault_a_broken_verifier_is_not_flagged(played, families, text,
                                                                              data_sha256):
    records = played["oracle"] + played["no_write"]              # no needless grant, no wrong approver
    report = broken.verifier_comparisons(records, run_of(records, families, text, data_sha256))
    assert report["real"]["clean"] is True
    assert report["broken"]["accepts_any_closed_request"]["caught"] is True
    for name in ("ignores_grant_records", "ignores_the_approver"):
        assert (report["broken"][name]["caught"], report["broken"][name]["n_graded_otherwise"]) == (False, 0)


def test_the_real_verifier_is_flagged_for_a_saved_state_that_its_calls_do_not_give(played, families, text,
                                                                                  data_sha256):
    # A saved final state without the grant record that its saved calls write: the real verifier grades
    # the saved state, the replay gives another state, so the comparison flags the real verifier too.
    position = next(index for index, record in enumerate(played["oracle"]) if record["outcome"] == "grant")
    changed = copy.deepcopy(played["oracle"])
    changed[position]["final_state"]["grants"] = []
    report = broken.verifier_comparisons(changed, run_of(changed, families, text, data_sha256))
    assert (report["real"]["clean"], report["real"]["n_flagged"]) == (False, 1)
    assert report["real"]["flagged"] == [monitors.episode_id(changed[position])]


def test_over_several_files_the_counts_are_added_up(played, shortcuts, families, text, data_sha256):
    def compared(records: list[dict]) -> dict:
        return broken.verifier_comparisons(records, run_of(records, families, text, data_sha256))
    one, two = compared(played["oracle"] + played["no_write"]), compared(shortcuts)
    together = broken.pooled([one, two])
    assert together["real"]["clean"] is True
    assert together["real"]["n_attempts"] == one["real"]["n_attempts"] + two["real"]["n_attempts"]
    for name, row in together["broken"].items():
        assert row["n_flagged"] == one["broken"][name]["n_flagged"] + two["broken"][name]["n_flagged"]
        assert row["caught"] is True
    changed = copy.deepcopy(shortcuts[:1])
    changed[0]["final_state"]["approvals"] = [["nobody", "nothing", "nobody"]]
    assert broken.pooled([one, two, compared(changed)])["real"]["clean"] is False


# ---------------------------------------------------------------------------
# The positive control: the scripted probe attempts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def oracle_run(played, families, text, data_sha256) -> Run:
    """A run of correct attempts only: no saved attempt of it can show a broken verifier."""
    return run_of(played["oracle"], families, text, data_sha256)


@pytest.fixture(scope="module")
def probes(oracle_run) -> dict:
    return broken.probe_comparisons(oracle_run)


def test_the_probe_families_are_the_first_family_of_each_outcome_in_file_order(families):
    chosen = broken.probe_families(families)
    assert [family["outcome"] for family in chosen] == list(OUTCOMES)
    in_file_order = list(families.values())
    for family in chosen:
        first = next(other for other in in_file_order if other["outcome"] == family["outcome"])
        assert family["family_id"] == first["family_id"]
    # The choice follows the order of the file, and a split without an outcome gives no probes.
    backwards = dict(reversed(list(families.items())))
    assert [family["family_id"] for family in broken.probe_families(backwards)] != [
        family["family_id"] for family in chosen]
    without_grant = {key: family for key, family in families.items() if family["outcome"] != "grant"}
    with pytest.raises(ValueError, match="no family of the expected outcome grant"):
        broken.probe_families(without_grant)


def test_the_probes_are_every_named_scripted_policy_on_each_probe_family(oracle_run, families, text):
    records = broken.probe_records(oracle_run)
    names = list(policies.named_policies(text, 0))
    assert len(names) > 25 and len(records) == 4 * len(names)
    assert [record["arm"] for record in records[::4]] == names
    wanted = [(family["family_id"], family["wordings"][0]["style"])
              for family in broken.probe_families(families)]
    for record in records:
        assert record["run"] == broken.PROBE_RUN
    assert [(record["family_id"], record["style"]) for record in records[:4]] == wanted
    # A probe is a record as the harness saves it, of the files and the set-up of the run: the audit's
    # comparison finds no difference of any kind in it.
    for record in records:
        assert record_differences(record, oracle_run) == ([], [])
    assert {record["passed"] for record in records} == {True, False}
    assert all(record["passed"] and record["reward"] == 100 for record in records[:4])       # the oracle


def test_the_probes_are_made_again_the_same_and_under_the_set_up_of_the_run(played, families, text,
                                                                           data_sha256, oracle_run):
    assert broken.probe_records(oracle_run) == broken.probe_records(oracle_run)
    # Another turn limit in the settings of the run is the turn limit of the probes.
    short = [{**record, "settings": {**record["settings"], "turn_limit": 5}} for record in played["oracle"]]
    records = broken.probe_records(run_of(short, families, text, data_sha256))
    assert {record["settings"]["turn_limit"] for record in records} == {5}
    never_closed = [record for record in records if record["arm"] == "repeat_reads"]
    assert {record["n_turns"] for record in never_closed} == {5}
    with pytest.raises(ValueError, match="no valid attempt"):
        broken.probe_records(run_of([], families, text, data_sha256))


def test_each_broken_verifier_is_flagged_on_the_probes_and_the_real_one_is_not(probes, families, text):
    assert probes["real"]["clean"] is True and probes["real"]["n_flagged"] == 0
    assert probes["real"]["n_attempts"] == 4 * len(probes["policies"])
    assert probes["policies"] == list(policies.named_policies(text, 0))
    by_outcome = {family["outcome"]: family for family in broken.probe_families(families)}
    assert [entry["family_id"] for entry in probes["families"]] == [
        by_outcome[outcome]["family_id"] for outcome in OUTCOMES]

    def probe(policy: str, outcome: str) -> str:
        family = by_outcome[outcome]
        return f"{policy} on {family['family_id']}/s{family['wordings'][0]['style']}"

    for name, row in probes["broken"].items():
        assert row["caught"] is True and row["n_flagged"] == row["n_graded_otherwise"] > 0, name
        assert len(row["flagged"]) == row["n_flagged"]                   # every flagged probe is named
    # The fitting attempts: a grant of a role that is held already, a grant made twice, an approval
    # that is asked from the wrong person, and a close with no action.
    grants = probes["broken"]["ignores_grant_records"]["flagged"]
    assert probe("grant_role_already_held", "already") in grants and probe("repeat_the_grant", "grant") in grants
    assert probe("wrong_approver_self", "approval") in probes["broken"]["ignores_the_approver"]["flagged"]
    closed = probes["broken"]["accepts_any_closed_request"]["flagged"]
    assert probe("close_granted_without_grant", "grant") in closed
    assert probe("grant_without_approval", "approval") in closed
    assert not any(found.startswith("oracle on ") for row in probes["broken"].values()
                   for found in row["flagged"])


def test_the_control_passes_with_the_probes_when_no_saved_attempt_can_show_a_fault(probes, oracle_run):
    saved = broken.verifier_comparisons(oracle_run.records, oracle_run)
    assert all(row["n_flagged"] == 0 for row in saved["broken"].values())
    report = broken.positive_control([saved], probes)
    assert report["passed"] is True
    assert report["saved"]["broken"]["ignores_grant_records"]["n_flagged"] == 0
    markdown = broken.to_markdown(report, "episodes.jsonl")
    assert "**The control PASSES.**" in markdown
    assert "| ignores_grant_records | 0 | 0 | no fitting attempt among the saved ones |" in markdown
    assert "| the real verifier | 0 | 0 | clean |" in markdown
    assert "not a failure of the control" in markdown
    # The policies of the probes are listed, and each broken verifier's flagged probes are named.
    assert ", ".join(probes["policies"]) in markdown
    assert "grant_role_already_held on " in markdown and "wrong_approver_self on " in markdown


def test_the_control_fails_when_the_real_verifier_is_flagged_on_a_saved_attempt(probes, played, families, text,
                                                                               data_sha256):
    changed = copy.deepcopy(played["oracle"])
    position = next(index for index, record in enumerate(changed) if record["outcome"] == "grant")
    changed[position]["final_state"]["grants"] = []
    saved = broken.verifier_comparisons(changed, run_of(changed, families, text, data_sha256))
    report = broken.positive_control([saved], probes)
    assert report["passed"] is False and report["probes"]["real"]["clean"] is True
    markdown = broken.to_markdown(report, "episodes.jsonl")
    assert "**The control FAILS.**" in markdown
    # The real verifier fails the changed state, and the saved verdict was a pass: 1 graded otherwise.
    assert "| the real verifier | 1 | 1 | FLAGGED |" in markdown


def test_the_control_fails_when_a_verifier_that_is_not_broken_stands_among_the_broken_ones(oracle_run):
    not_broken = {"not_broken": impl.main_version("verify"),
                  "accepts_any_closed_request": broken.accepts_any_closed_request}
    probes = broken.probe_comparisons(oracle_run, not_broken)
    assert probes["broken"]["not_broken"]["caught"] is False
    assert probes["broken"]["accepts_any_closed_request"]["caught"] is True
    saved = broken.verifier_comparisons(oracle_run.records, oracle_run, not_broken)
    report = broken.positive_control([saved], probes)
    assert report["passed"] is False
    assert "| not_broken | 0 | 0 | NOT CAUGHT |" in broken.to_markdown(report, "episodes.jsonl")
    only_broken = {"accepts_any_closed_request": broken.accepts_any_closed_request}
    assert broken.positive_control(
        [broken.verifier_comparisons(oracle_run.records, oracle_run, only_broken)],
        broken.probe_comparisons(oracle_run, only_broken))["passed"] is True


def test_many_saved_flags_do_not_stand_in_for_the_probes(played, shortcuts, families, text, data_sha256,
                                                        oracle_run):
    # The saved attempts show every fault, but the probes are made for a verifier that they do not
    # catch: the control looks at the probes, so it fails.
    records = played["oracle"] + played["no_write"] + shortcuts
    not_broken = {"not_broken": impl.main_version("verify")}
    saved = broken.verifier_comparisons(records, run_of(records, families, text, data_sha256))
    assert all(row["caught"] for row in saved["broken"].values())
    probes = {**broken.probe_comparisons(oracle_run, not_broken)}
    assert broken.control_passes(probes, saved) is False


def test_the_positive_control_script_takes_several_files_together(tmp_path, capsys, scenarios, text, played,
                                                                 shortcuts):
    script = load_script("audit_positive_control")
    write_records(tmp_path / "one.jsonl", played["oracle"] + played["no_write"])
    write_records(tmp_path / "two.jsonl", shortcuts)
    common = ["--scenarios", str(scenarios), "--split", SPLIT, "--text", str(text.path)]
    both = ["--episodes", str(tmp_path / "one.jsonl"), str(tmp_path / "two.jsonl")]
    assert script.main([*both, *common, "--out", str(tmp_path / "both")]) == 0
    printed = capsys.readouterr().out
    assert "**The control PASSES.**" in printed and printed.count("valid attempts; flagged: the real") == 2
    report = json.loads((tmp_path / "both" / "positive_control.json").read_text(encoding="utf-8"))
    assert report["passed"] is True and len(report["files"]) == 2
    assert report["probes"]["policies"] == list(policies.named_policies(text, 0))
    assert report["probes"]["real"]["n_attempts"] == 4 * len(report["probes"]["policies"])
    n_already = sum(1 for record in shortcuts if record["outcome"] == "already")
    assert report["saved"]["broken"]["ignores_grant_records"]["n_flagged"] == n_already
    assert [one["broken"]["ignores_grant_records"]["n_flagged"] for one in report["files"]] == [0, n_already]
    assert (tmp_path / "both" / "positive_control.md").read_text(encoding="utf-8") in printed
    twice = ["--episodes", str(tmp_path / "one.jsonl"), str(tmp_path / "one.jsonl")]
    assert script.main([*twice, *common, "--out", str(tmp_path / "twice")]) == 2
    assert "an episodes file is given twice" in capsys.readouterr().err


def test_the_positive_control_script_ends_with_zero_when_it_passes_and_one_when_it_fails(
        tmp_path, capsys, scenarios, text, played):
    script = load_script("audit_positive_control")
    def run_script(name: str, records: list[dict]) -> int:
        episodes = tmp_path / f"{name}.jsonl"
        write_records(episodes, records)
        return script.main(["--episodes", str(episodes), "--scenarios", str(scenarios), "--split", SPLIT,
                            "--text", str(text.path), "--out", str(tmp_path / name)])
    # Correct attempts only: no saved attempt shows a fault, and the probes carry the control.
    assert run_script("correct_only", played["oracle"]) == 0
    printed = capsys.readouterr().out
    assert "**The control PASSES.**" in printed
    report = json.loads((tmp_path / "correct_only" / "positive_control.json").read_text(encoding="utf-8"))
    assert report["passed"] is True and set(report["probes"]["broken"]) == set(broken.BROKEN)
    assert all(row["n_flagged"] == 0 for row in report["saved"]["broken"].values())
    # A saved state that its calls do not give: the real verifier is flagged, and the control fails.
    changed = copy.deepcopy(played["oracle"])
    position = next(index for index, record in enumerate(changed) if record["outcome"] == "grant")
    changed[position]["final_state"]["grants"] = []
    assert run_script("real_flagged", changed) == 1
    assert "**The control FAILS.**" in capsys.readouterr().out
    missing = ["--episodes", str(tmp_path / "missing.jsonl"), "--scenarios", str(scenarios), "--split",
               SPLIT, "--text", str(text.path), "--out", str(tmp_path / "none")]
    assert script.main(missing) == 2
    assert not (tmp_path / "none").exists()
