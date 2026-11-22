"""scripts/replay_check.py: saved episodes are replayed and graded again; a record that was changed is found.

The episodes come from the scripted policies, played on a scenario folder that the real build script
writes into a temporary folder. Nothing is written into the repository.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import shutil
from pathlib import Path

import pytest

from accessdesk.env.scenarios import load_families
from accessdesk.harness import policies
from accessdesk.harness.records import write_records

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "calib"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def replay_check():
    return load_script("replay_check")


@pytest.fixture(scope="module")
def run_scripted():
    return load_script("run_scripted")


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory, text, cfg) -> Path:
    """A scenario folder as the build script writes it, from the text and the config of the tests."""
    folder = tmp_path_factory.mktemp("replay") / "scenarios_for_replay"
    load_script("build_scenarios").build(text, ROOT / "data" / "names", folder, cfg)
    return folder


@pytest.fixture(scope="module")
def families(scenarios) -> list[dict]:
    """The families of the split that the tests play: read from the built folder, as the script reads
    them. (In this file the name `families` means this list, not the four splits of conftest.py.)"""
    return load_families(scenarios / f"{SPLIT}.jsonl")


@pytest.fixture(scope="module")
def data_sha256(scenarios) -> str:
    return hashlib.sha256((scenarios / f"{SPLIT}.jsonl").read_bytes()).hexdigest()


def play(run_scripted, policy, families, text, cfg, data_sha256, *, ruleset="default", toolset="five"):
    """One record per scenario of the families, made the way scripts/run_scripted.py makes them."""
    return run_scripted.run_split(families, policy, "scripted", text, cfg, ruleset=ruleset, toolset=toolset,
                                  seed=0, data_sha256=data_sha256)


@pytest.fixture(scope="module")
def good_records(run_scripted, families, text, cfg, data_sha256) -> list[dict]:
    """Episodes of three kinds: clean passes, passes after two malformed messages, and fails."""
    chosen = (policies.oracle(text), policies.stumbling_oracle(text, 2), policies.wrong_role(text))
    return [record for policy in chosen for record in play(run_scripted, policy, families, text, cfg,
                                                           data_sha256)]


def check(replay_check, capsys, tmp_path, records, scenarios, text, *options: str) -> tuple[int, str, str]:
    """Write the records into a file and run the script on it: (exit code, stdout, stderr)."""
    episodes = tmp_path / "episodes.jsonl"
    write_records(episodes, records)
    code = replay_check.main(["--episodes", str(episodes), "--scenarios", str(scenarios), "--split", SPLIT,
                              "--text", str(text.path), *options])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def counts(output: str) -> dict[str, int]:
    """The four counts of the printed report, by their label."""
    labels = ("records", "harness errors", "checked", "with a difference")
    return {label: int(line[len(label):].split()[0]) for line in output.splitlines()
            for label in labels if line.startswith(label + " ")}


def first(records: list[dict], wanted) -> int:
    """The index of the first record for which wanted(record) is true."""
    return next(index for index, record in enumerate(records) if wanted(record))


# ---------------------------------------------------------------------------
# Episodes that nobody changed
# ---------------------------------------------------------------------------
def test_scripted_episodes_of_every_named_policy_have_no_difference(replay_check, run_scripted, capsys,
                                                                    tmp_path, scenarios, families, text, cfg,
                                                                    data_sha256):
    records = [record for policy in policies.named_policies(text).values()
               for record in play(run_scripted, policy, families, text, cfg, data_sha256)]
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    n_scenarios = sum(len(family["wordings"]) for family in families)
    assert len(records) == len(policies.named_policies(text)) * n_scenarios > 2000
    assert code == 0
    assert counts(out) == {"records": len(records), "harness errors": 0, "checked": len(records),
                           "with a difference": 0}
    assert "no difference" in out
    # The file holds passes and fails, malformed messages and tool errors: the check saw every kind.
    assert {record["passed"] for record in records} == {True, False}
    assert any(record["n_malformed"] for record in records)
    assert any(record["n_tool_errors"] for record in records)


def test_episodes_of_another_rule_set_and_of_the_six_tool_set_have_no_difference(
        replay_check, run_scripted, capsys, tmp_path, scenarios, families, text, cfg, data_sha256):
    other = "all_grants_need_approval"
    common = (families, text, cfg, data_sha256)
    records = (play(run_scripted, policies.oracle(text, other), *common, ruleset=other)
               + play(run_scripted, policies.oracle(text), *common, ruleset=other)
               + play(run_scripted, policies.extra_tool_then_oracle(text), *common, toolset="six"))
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    assert code == 0 and counts(out)["with a difference"] == 0
    assert {record["ruleset"] for record in records} == {other, "default"}
    assert {record["toolset"] for record in records} == {"five", "six"}


def test_a_record_is_graded_with_the_reward_numbers_that_it_names(replay_check, run_scripted, capsys,
                                                                  tmp_path, scenarios, families, text, cfg,
                                                                  data_sha256):
    """Episodes made with other reward numbers and another turn limit carry them in `settings`. The check
    takes them from there: with the numbers of accessdesk/config.py these rewards would be differences."""
    other = dataclasses.replace(cfg, malformed_penalty=7, turn_penalty=3, turn_limit=7)
    records = play(run_scripted, policies.stumbling_oracle(text, 2), families, text, other, data_sha256)
    assert {record["reward"] for record in records} == {100 - 14 - 3}      # 2 malformed, 1 turn over
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    assert code == 0 and counts(out)["with a difference"] == 0
    # The same records without their settings are graded with the default numbers: every reward differs.
    bare = [{key: value for key, value in record.items() if key != "settings"} for record in records]
    code, out, _ = check(replay_check, capsys, tmp_path, bare, scenarios, text)
    assert code == 1 and counts(out)["with a difference"] == len(records)
    assert "reward: saved 83; graded again: main 88, second 88" in out


def test_episodes_of_a_run_with_another_chunk_size_have_no_difference(replay_check, run_scripted, capsys,
                                                                      tmp_path, scenarios, families, text, cfg,
                                                                      data_sha256):
    """scripts/gpu/rollout.py --chunk-size changes the config hash of a run and no setting of its records.
    Such a run is an honest run: the check of the config hash must not report it."""
    other = dataclasses.replace(cfg, chunk_size=7)
    records = play(run_scripted, policies.oracle(text), families, text, other, data_sha256)
    assert {record["config_sha256"] for record in records} == {other.sha256()} != {cfg.sha256()}
    assert all(record["settings"] == cfg.compared_settings() for record in records)
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    assert code == 0 and counts(out)["with a difference"] == 0


def test_the_check_reads_and_writes_nothing_but_its_own_output(replay_check, capsys, tmp_path, scenarios,
                                                               text, good_records):
    def snapshot() -> dict[str, str]:
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for folder in (tmp_path, scenarios) for path in sorted(folder.rglob("*")) if path.is_file()}

    write_records(tmp_path / "episodes.jsonl", good_records)
    before = snapshot()
    assert check(replay_check, capsys, tmp_path, good_records, scenarios, text)[0] == 0
    assert snapshot() == before


# ---------------------------------------------------------------------------
# Records that were changed on purpose: each change must be found
# ---------------------------------------------------------------------------
def clean_pass(record: dict) -> bool:
    return record["reward"] == 100


def stumbled_pass(record: dict) -> bool:
    return record["passed"] and record["n_malformed"] == 2


def granted(record: dict) -> bool:
    return record["passed"] and record["outcome"] == "grant" and record["n_malformed"] == 0


def asked_for_approval(record: dict) -> bool:
    return record["passed"] and record["outcome"] == "approval" and record["n_malformed"] == 0


def set_reward(record: dict) -> None:
    record["reward"] = 95


def set_failed(record: dict) -> None:
    record["passed"] = False


def drop_the_grant_from_the_state(record: dict) -> None:
    record["final_state"]["grants"] = []


def change_the_resolution_in_the_state(record: dict) -> None:
    record["final_state"]["request"]["resolution"] = "denied"


def change_a_tool_result(record: dict) -> None:
    record["turns"][1]["tool_response"] = "{}"


def ask_another_approver(record: dict) -> None:
    """The saved call names the person as the own approver. The tool answer is the same confirmation, so
    only the state of the replay shows the change."""
    args = record["turns"][2]["args"]
    assert record["turns"][2]["tool_key"] == "request_approval" and args["approver_id"] != args["person_id"]
    args["approver_id"] = args["person_id"]


def call_a_malformed_message_well_formed(record: dict) -> None:
    record["turns"][0]["parse_ok"] = True


def change_an_assistant_text(record: dict) -> None:
    record["turns"][0]["assistant_text"] += " and some words after the call"


def hide_a_malformed_message(record: dict) -> None:
    """The first malformed message is taken out and the counters are made to fit, but the reward still
    carries its penalty."""
    del record["turns"][0]
    for index, turn in enumerate(record["turns"]):
        turn["turn"] = index
    record["n_turns"] -= 1
    record["n_malformed"] -= 1


def count_one_malformed_message_less(record: dict) -> None:
    record["n_malformed"] -= 1


def say_never_closed(record: dict) -> None:
    record["closed"], record["ended_by"] = False, "turn_limit"


def change_the_outcome_label(record: dict) -> None:
    record["outcome"] = "inactive" if record["outcome"] != "inactive" else "grant"


def change_the_turn_budget(record: dict) -> None:
    record["turn_budget"] += 2


def change_the_placeholder_flag(record: dict) -> None:
    record["text_is_placeholder"] = not record["text_is_placeholder"]


def name_another_schema(record: dict) -> None:
    record["schema"] = 2


def call_a_tool_that_does_not_exist(record: dict) -> None:
    record["turns"][0]["tool_key"] = "delete_person"


def change_a_reward_number_in_the_settings(record: dict) -> None:
    record["settings"]["malformed_penalty"] = 1


def raise_the_pass_reward_and_the_reward(record: dict) -> None:
    """The reward fits the changed setting. Only the config hash of the record still tells."""
    record["settings"]["pass_reward"] = 500
    record["reward"] += 400


def name_a_setting_that_is_no_config_field(record: dict) -> None:
    record["settings"]["bonus"] = 3


def name_another_text(record: dict) -> None:
    record["text_sha256"] = "0" * 64


def name_another_data_file(record: dict) -> None:
    record["data_sha256"] = "0" * 64


def name_an_unknown_family(record: dict) -> None:
    record["family_id"] = "ca-99999"


def lose_the_turns(record: dict) -> None:
    del record["turns"]


CHANGES = [
    (set_reward, clean_pass, {"reward"}),
    (set_failed, clean_pass, {"passed"}),
    (drop_the_grant_from_the_state, granted, {"final_state"}),
    (change_the_resolution_in_the_state, clean_pass, {"final_state"}),
    (change_a_tool_result, clean_pass, {"replay"}),
    (ask_another_approver, asked_for_approval, {"final_state", "turns", "passed", "reward"}),
    (call_a_malformed_message_well_formed, stumbled_pass, {"replay"}),
    (change_an_assistant_text, clean_pass, {"turns"}),
    (hide_a_malformed_message, stumbled_pass, {"reward"}),
    (count_one_malformed_message_less, stumbled_pass, {"counters"}),
    (say_never_closed, clean_pass, {"counters"}),
    (change_the_outcome_label, clean_pass, {"labels"}),
    (change_the_turn_budget, clean_pass, {"labels"}),
    (change_the_placeholder_flag, clean_pass, {"labels"}),
    (name_another_schema, clean_pass, {"record"}),
    (call_a_tool_that_does_not_exist, clean_pass, {"replay"}),      # the tools refuse the saved call
    (change_a_reward_number_in_the_settings, stumbled_pass, {"settings", "reward"}),
    (raise_the_pass_reward_and_the_reward, clean_pass, {"settings"}),
    (name_a_setting_that_is_no_config_field, clean_pass, {"settings"}),
    (name_another_text, clean_pass, {"text"}),
    (name_another_data_file, clean_pass, {"data"}),
    (name_an_unknown_family, clean_pass, {"family"}),
    (lose_the_turns, clean_pass, {"record"}),
]


@pytest.mark.parametrize("change, wanted, kinds", CHANGES, ids=[change.__name__ for change, _, _ in CHANGES])
def test_one_changed_record_is_found_among_the_others(replay_check, capsys, tmp_path, scenarios, text,
                                                      good_records, change, wanted, kinds):
    records = copy.deepcopy(good_records)
    index = first(records, wanted)
    change(records[index])
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    assert code == 1
    assert counts(out) == {"records": len(records), "harness errors": 0, "checked": len(records),
                           "with a difference": 1}
    shown = [line for line in out.splitlines() if line.startswith("  line ")]
    assert shown and all(line.startswith(f"  line {index + 1}  ") for line in shown)
    assert {line.split("  ")[3].split(":")[0] for line in shown} == kinds


def test_the_report_names_the_saved_value_and_both_versions(replay_check, capsys, tmp_path, scenarios, text,
                                                            good_records):
    records = copy.deepcopy(good_records)
    records[first(records, clean_pass)]["reward"] = 95
    _, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    assert "reward: saved 95; graded again: main 100, second 100" in out
    assert "by kind: reward 1" in out


@pytest.mark.parametrize("name, wrong", [("reward", lambda *args: 0), ("verify", lambda *args: False)])
def test_a_second_version_that_disagrees_is_reported_though_the_main_one_agrees(
        replay_check, capsys, tmp_path, scenarios, text, good_records, monkeypatch, name, wrong):
    """Both versions grade. If only the second one gives another answer, that is a difference."""
    module = "verifier" if name == "verify" else "reward"
    monkeypatch.setattr(f"accessdesk._second.{module}.{name}", wrong)
    code, out, _ = check(replay_check, capsys, tmp_path, good_records, scenarios, text)
    n_passes = sum(record["passed"] for record in good_records)
    assert code == 1 and counts(out)["with a difference"] == n_passes > 0
    saved, main, second = ("True", "True", "False") if name == "verify" else ("100", "100", "0")
    key = "passed" if name == "verify" else "reward"
    assert f"{key}: saved {saved}; graded again: main {main}, second {second}" in out


def test_only_the_first_ten_differences_are_printed_and_all_are_counted(replay_check, capsys, tmp_path,
                                                                        scenarios, text, good_records):
    records = copy.deepcopy(good_records)
    for record in records[:12]:
        record["reward"] = 95
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    shown = [line for line in out.splitlines() if line.startswith("  line ")]
    assert code == 1 and counts(out)["with a difference"] == 12
    assert [line.split()[1] for line in shown] == [str(number) for number in range(1, 11)]
    assert "the first 10 of 12 differences" in out and "by kind: reward 12" in out


# ---------------------------------------------------------------------------
# What the report shows beside the differences
# ---------------------------------------------------------------------------
def test_the_report_shows_the_settings_of_the_records_and_how_even_the_file_is(replay_check, capsys, tmp_path,
                                                                             scenarios, text, cfg,
                                                                             good_records):
    """A record that was taken out is no difference of any record. The report still shows it: one scenario
    has fewer records than the others."""
    code, out, _ = check(replay_check, capsys, tmp_path, good_records, scenarios, text)
    n_scenarios = len({record["scenario_id"] for record in good_records})
    assert code == 0 and f"scenarios          {n_scenarios}, each with 3 record(s)" in out
    assert "settings           1 kind(s) among the checked records" in out
    assert f'"pass_reward": {cfg.pass_reward}' in out and f"  {len(good_records)} record(s): " in out
    without_one = [record for index, record in enumerate(good_records) if index != 5]
    code, out, _ = check(replay_check, capsys, tmp_path, without_one, scenarios, text)
    assert code == 0 and counts(out)["with a difference"] == 0
    assert f"scenarios          {n_scenarios}, with 2 to 3 records each: NOT EVEN" in out


# ---------------------------------------------------------------------------
# Harness errors, the rule set option, and files that do not belong together
# ---------------------------------------------------------------------------
def test_a_harness_error_record_is_counted_and_not_checked(replay_check, capsys, tmp_path, scenarios, text,
                                                           good_records):
    records = copy.deepcopy(good_records)
    records[3].update({"harness_error": "CUDA out of memory", "reward": 12345, "turns": []})
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text)
    assert code == 0
    assert counts(out) == {"records": len(records), "harness errors": 1, "checked": len(records) - 1,
                           "with a difference": 0}


def test_a_file_without_a_record_to_check_is_not_reported_as_good(replay_check, capsys, tmp_path, scenarios,
                                                                  text, good_records):
    only_errors = [{**record, "harness_error": "stopped"} for record in good_records[:3]]
    for records in ([], only_errors):
        code, out, err = check(replay_check, capsys, tmp_path, records, scenarios, text)
        assert code == 2 and "no record was checked" in err
        assert counts(out)["checked"] == 0 and counts(out)["harness errors"] == len(records)


def test_the_ruleset_option_demands_that_rule_set_of_every_record(replay_check, run_scripted, capsys,
                                                                  tmp_path, scenarios, families, text, cfg,
                                                                  data_sha256):
    other = "all_grants_need_approval"
    records = play(run_scripted, policies.oracle(text, other), families, text, cfg, data_sha256,
                   ruleset=other)
    assert check(replay_check, capsys, tmp_path, records, scenarios, text)[0] == 0
    assert check(replay_check, capsys, tmp_path, records, scenarios, text, "--ruleset", other)[0] == 0
    code, out, _ = check(replay_check, capsys, tmp_path, records, scenarios, text, "--ruleset", "default")
    assert code == 1 and counts(out)["with a difference"] == len(records)
    assert f"ruleset: the record is of the rule set {other}, not of default" in out


def test_a_text_file_or_a_split_file_that_is_not_the_one_of_the_build_stops_the_check(
        replay_check, capsys, tmp_path, scenarios, text, marker_text, good_records):
    code, _, err = check(replay_check, capsys, tmp_path, good_records, scenarios, marker_text)
    assert code == 2 and "another text file" in err

    changed = tmp_path / "changed"
    shutil.copytree(scenarios, changed)
    with (changed / f"{SPLIT}.jsonl").open("ab") as handle:
        handle.write(b"\n")
    code, _, err = check(replay_check, capsys, tmp_path, good_records, changed, text)
    assert code == 2 and "is not the file that the build wrote" in err

    (changed / "summary.json").unlink()
    code, _, err = check(replay_check, capsys, tmp_path, good_records, changed, text)
    assert code == 2 and "summary.json is missing" in err


def test_a_family_whose_expected_state_is_not_what_the_rules_give_is_found(replay_check, families, text,
                                                                           data_sha256, good_records):
    """The family file is trusted only as far as the rules agree with it. Here a copy of one family expects
    a grant that the rules do not ask for."""
    record = good_records[first(good_records, clean_pass)]
    family = copy.deepcopy(next(f for f in families if f["family_id"] == record["family_id"]))
    family["expected"]["default"]["grants"].append(["p-made-up", "r-made-up"])
    found = replay_check.check_record(record, {family["family_id"]: family}, text, data_sha256)
    assert {kind for kind, _ in found} == {"expected", "passed", "reward"}
    assert replay_check.check_record(record, {f["family_id"]: f for f in families}, text, data_sha256) == []
