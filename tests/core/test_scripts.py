"""scripts/run_scripted.py and scripts/play.py, run through their main functions (no terminal needed), and
the parts of scripts/gpu/rollout.py that need no model."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from accessdesk import freeze
from accessdesk.env.rules import reference_plan
from accessdesk.env.scenarios import load_families
from accessdesk.eval import tables
from accessdesk.harness.policies import named_policies, oracle
from accessdesk.harness.records import read_records, replay

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "data" / "scenarios_placeholder"
NO_TOKENIZER = ROOT / "no-such-folder"
TOKENIZER = ROOT / ".cache" / "hf" / "Qwen3.5-4B"
# Three answers of accessdesk.freeze.frozen_state, made up here: before the freeze, a frozen tree that is as
# its manifest says, and a frozen tree with two differences.
NOT_FROZEN = {"manifest_sha256": None, "code_sha256": "c" * 64, "matches": False,
              "differences": ["MANIFEST.json is missing: the tree is not frozen"]}
AS_FROZEN = {"manifest_sha256": "a" * 64, "code_sha256": "c" * 64, "matches": True, "differences": []}
CHANGED = {"manifest_sha256": "a" * 64, "code_sha256": "d" * 64, "matches": False,
           "differences": ["changed: accessdesk/harness/parser.py (group harness); the manifest has "
                           "111111111111, the file has 222222222222",
                           "new: accessdesk/eval/audit.py (group evaluation) matches a frozen pattern and is "
                           "not in the manifest"]}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_scripted():
    return load_script("run_scripted")


@pytest.fixture(scope="module")
def play_script():
    return load_script("play")


@pytest.fixture(autouse=True)
def tree_state(run_scripted, rollout_script, monkeypatch) -> dict:
    """The two run scripts ask `accessdesk.freeze.frozen_state` whether the repository is as frozen, and
    stop on a frozen tree that has changed. In this file they get a made-up answer, so that no test depends
    on the state of the repository on the day it runs. The answer is "not frozen", unless a test puts
    another one into `state` of the dict that this fixture returns. `roots` holds what the scripts asked
    about. One test below asks the real function about the repository itself."""
    asked: dict = {"state": NOT_FROZEN, "roots": []}

    def answer(root: Path) -> dict:
        asked["roots"].append(Path(root))
        return asked["state"]

    for script in (run_scripted, rollout_script):
        assert script.frozen_state is freeze.frozen_state and script.run_problem is freeze.run_problem
        monkeypatch.setattr(script, "frozen_state", answer)
    return asked


@pytest.fixture(scope="module")
def calib() -> dict[str, dict]:
    """One calibration family of each outcome, read from the scenario folder itself."""
    by_outcome: dict[str, dict] = {}
    for family in load_families(SCENARIOS / "calib.jsonl"):
        by_outcome.setdefault(family["outcome"], family)
    assert len(by_outcome) == 4
    return by_outcome


# ---------------------------------------------------------------------------
# run_scripted.py
# ---------------------------------------------------------------------------
def scripted(run_scripted, capsys, *argv: str) -> tuple[int, str, str]:
    code = run_scripted.main(["--scenarios", str(SCENARIOS), *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def rates(output: str) -> dict[str, str]:
    """The printed table: first word of the line -> its percentage."""
    return {line.split()[0]: line.split()[1] for line in output.splitlines() if line.startswith("  ")}


def test_the_oracle_passes_every_scenario_of_the_dev_split(run_scripted, capsys, tmp_path, text, cfg):
    out_file = tmp_path / "sub" / "oracle.jsonl"
    code, out, _ = scripted(run_scripted, capsys, "--split", "dev", "--policy", "oracle",
                            "--out", str(out_file))
    assert code == 0
    assert rates(out) == {"grant": "100.0%", "approval": "100.0%", "already": "100.0%", "inactive": "100.0%",
                          "all": "100.0%"}
    assert "(192 episodes, 48 families)" in out
    records = read_records(out_file)
    assert len(records) == 192 and all(r["passed"] and r["reward"] == 100 for r in records)
    labels = {(r["run"], r["arm"], r["split"], r["sample"]) for r in records}
    assert labels == {("scripted", "oracle", "dev", 0)}
    assert [r["seed"]["pos"] for r in records] == list(range(192))
    assert len({r["scenario_id"] for r in records}) == 192
    assert all(r["text_sha256"] == text.sha256 and r["config_sha256"] == cfg.sha256() for r in records)
    families = {family["family_id"]: family for family in load_families(SCENARIOS / "dev.jsonl")}
    assert all(replay(r, families[r["family_id"]], text, cfg) == r["final_state"] for r in records)


@pytest.mark.parametrize("policy, passing", [
    ("constant_granted", "grant"), ("constant_pending", "approval"), ("constant_already", "already"),
    ("constant_denied", "inactive")])
def test_a_constant_policy_is_reported_at_25_percent(run_scripted, capsys, tmp_path, policy, passing):
    code, out, _ = scripted(run_scripted, capsys, "--split", "test", "--policy", policy,
                            "--out", str(tmp_path / "c.jsonl"))
    wanted = {outcome: "0.0%" for outcome in ("grant", "approval", "already", "inactive")}
    assert code == 0 and rates(out) == {**wanted, passing: "100.0%", "all": "25.0%"}


def test_every_record_carries_the_hash_of_the_split_file_it_was_run_on(run_scripted, capsys, tmp_path):
    hashes = {}
    for split in ("dev", "calib"):
        code, _, _ = scripted(run_scripted, capsys, "--split", split, "--policy", "oracle",
                              "--out", str(tmp_path / f"{split}.jsonl"))
        records = read_records(tmp_path / f"{split}.jsonl")
        assert code == 0 and len(records) > 90
        assert all(list(record)[-1] == "data_sha256" for record in records)
        hashes[split] = {record["data_sha256"] for record in records}
    # The hash is taken over the bytes of the split file, and it is the one in the folder's summary.
    summary = json.loads((SCENARIOS / "summary.json").read_text(encoding="utf-8"))
    for split in ("dev", "calib"):
        of_the_file = hashlib.sha256((SCENARIOS / f"{split}.jsonl").read_bytes()).hexdigest()
        assert hashes[split] == {of_the_file} == {summary["files_sha256"][f"{split}.jsonl"]}
    assert hashes["dev"] != hashes["calib"]


def test_the_results_table_refuses_runs_on_two_different_scenario_files(run_scripted, capsys, tmp_path):
    """Two scenario folders with the same family ids, the same text and the same config, but other bytes in
    dev.jsonl: the records of the two runs look alike in every other field, and only the data hash tells."""
    other = tmp_path / "other_scenarios"
    shutil.copytree(SCENARIOS, other)
    families = load_families(other / "dev.jsonl")
    changed = [json.dumps({**family, "world_seed": family["world_seed"] + 1}, ensure_ascii=False)
               for family in families]
    (other / "dev.jsonl").write_bytes(("\n".join(changed) + "\n").encode("utf-8"))
    # The other folder is a build of its own: its summary names the hash of its own dev.jsonl.
    summary = json.loads((other / "summary.json").read_text(encoding="utf-8"))
    summary["files_sha256"]["dev.jsonl"] = hashlib.sha256((other / "dev.jsonl").read_bytes()).hexdigest()
    (other / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def run(folder: Path, policy: str, out: str) -> list[dict]:
        code = run_scripted.main(["--scenarios", str(folder), "--split", "dev", "--policy", policy,
                                  "--out", str(tmp_path / out)])
        capsys.readouterr()
        assert code == 0
        return read_records(tmp_path / out)

    here = run(SCENARIOS, "constant_granted", "here.jsonl")
    here_too = run(SCENARIOS, "oracle", "here_too.jsonl")
    there = run(other, "oracle", "there.jsonl")
    assert [r["scenario_id"] for r in here] == [r["scenario_id"] for r in there]
    assert {r["data_sha256"] for r in here} != {r["data_sha256"] for r in there}
    # The same file for both arms: a table, and no note.
    table = tables.compare_arms({"constant": here, "oracle": here_too}, "constant", n_draws=200)
    assert table["notes"] == [] and table["pairs"][0]["diff"] == 0.75
    # Another file for the second arm: refused.
    with pytest.raises(ValueError, match="same scenario file"):
        tables.compare_arms({"constant": here, "oracle": there}, "constant", n_draws=200)
    # Without the hashes the same two runs would be compared without an error: that is what the hash is for.
    bare = [{key: value for key, value in record.items() if key != "data_sha256"} for record in there]
    assert len(tables.compare_arms({"constant": here, "oracle": bare}, "constant", n_draws=200)["notes"]) == 1


def test_run_split_without_a_hash_writes_records_without_the_key(run_scripted, text, cfg):
    families = load_families(SCENARIOS / "calib.jsonl")[:2]
    policy = named_policies(text)["oracle"]
    records = run_scripted.run_split(families, policy, "oracle", text, cfg, ruleset="default",
                                     toolset="five", seed=0)
    assert len(records) == 8 and all("data_sha256" not in record for record in records)


def test_the_train_split_has_three_styles_per_family(run_scripted, capsys, tmp_path):
    code, out, _ = scripted(run_scripted, capsys, "--split", "train", "--policy", "do_nothing",
                            "--out", str(tmp_path / "n.jsonl"))
    assert code == 0 and "(600 episodes, 200 families)" in out and rates(out)["all"] == "0.0%"


def test_two_runs_write_the_same_bytes(run_scripted, capsys, tmp_path):
    for name in ("a", "b"):
        scripted(run_scripted, capsys, "--split", "calib", "--policy", "random_valid_calls", "--seed", "5",
                 "--out", str(tmp_path / name))
    assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()
    scripted(run_scripted, capsys, "--split", "calib", "--policy", "random_valid_calls", "--seed", "6",
             "--out", str(tmp_path / "c"))
    assert (tmp_path / "c").read_bytes() != (tmp_path / "a").read_bytes()


def test_another_rule_set_needs_its_own_oracle(run_scripted, capsys, tmp_path):
    common = ("--split", "dev", "--ruleset", "all_grants_need_approval", "--out", str(tmp_path / "r.jsonl"))
    _, new, _ = scripted(run_scripted, capsys, "--policy", "oracle_of_ruleset", *common)
    assert rates(new)["all"] == "100.0%"
    assert {r["ruleset"] for r in read_records(tmp_path / "r.jsonl")} == {"all_grants_need_approval"}
    _, old, _ = scripted(run_scripted, capsys, "--policy", "oracle", *common)
    assert rates(old) == {"grant": "0.0%", "approval": "100.0%", "already": "100.0%", "inactive": "100.0%",
                          "all": "75.0%"}


def test_the_extra_tool_set_can_be_run(run_scripted, capsys, tmp_path):
    code, out, _ = scripted(run_scripted, capsys, "--split", "calib", "--toolset", "six",
                            "--policy", "oracle", "--out", str(tmp_path / "six.jsonl"))
    assert code == 0 and rates(out)["all"] == "100.0%"
    assert {r["toolset"] for r in read_records(tmp_path / "six.jsonl")} == {"six"}


def test_list_prints_every_policy_name(run_scripted, capsys, text):
    assert run_scripted.main(["--list"]) == 0
    names = capsys.readouterr().out.split()
    assert names == list(named_policies(text)) + ["oracle_of_ruleset"]
    assert "oracle" in names and "constant_denied" in names and len(names) == len(set(names))


def test_missing_or_unknown_arguments_stop_the_script(run_scripted, capsys, tmp_path):
    for argv in (["--policy", "oracle"],
                 ["--scenarios", str(SCENARIOS), "--policy", "oracle"],
                 ["--scenarios", str(SCENARIOS), "--policy", "no_such_policy", "--out", str(tmp_path / "x")],
                 ["--scenarios", str(SCENARIOS), "--policy", "oracle", "--out", str(tmp_path / "x"),
                  "--split", "validation"]):
        with pytest.raises(SystemExit) as stop:
            run_scripted.main(argv)
        assert stop.value.code == 2
    capsys.readouterr()
    assert not (tmp_path / "x").exists()


def test_scenarios_of_another_text_file_or_config_are_refused(run_scripted, capsys, tmp_path):
    for field, message in (("text_sha256", "another text file"), ("config_sha256", "another config")):
        folder = tmp_path / field
        shutil.copytree(SCENARIOS, folder)
        summary = json.loads((folder / "summary.json").read_text())
        summary[field] = "0" * 64
        (folder / "summary.json").write_text(json.dumps(summary))
        code = run_scripted.main(["--scenarios", str(folder), "--policy", "oracle",
                                  "--out", str(tmp_path / "o")])
        assert code == 2 and message in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    code = run_scripted.main(["--scenarios", str(tmp_path / "empty"), "--policy", "oracle",
                              "--out", str(tmp_path / "o")])
    assert code == 2 and "summary.json is missing" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# play.py
# ---------------------------------------------------------------------------
def short_form(text, tool_key: str, args: dict[str, str]) -> str:
    """One line `tool_name key="value" ...` with the visible names, as a person would type it."""
    spec = text.tools[tool_key]
    values = {**args, "resolution": text.resolution[args["resolution"]]} if "resolution" in args else args
    return spec.name + "".join(f' {param.name}="{values[param.key]}"' for param in spec.params)


def correct_lines(text, family: dict) -> str:
    """The correct path for a family, typed in the short form, each message ended by an empty line."""
    return "".join(short_form(text, tool_key, args) + "\n\n" for tool_key, args in reference_plan(family))


def hand_play(play_script, typed: str, *argv: str, tokenizer: Path = NO_TOKENIZER) -> tuple[int, str]:
    out = io.StringIO()
    code = play_script.main(["--scenarios", str(SCENARIOS), "--tokenizer", str(tokenizer), *argv],
                            stdin=io.StringIO(typed), stdout=out)
    return code, out.getvalue()


@pytest.mark.parametrize("outcome", ["grant", "approval", "already", "inactive"])
def test_the_correct_path_typed_in_the_short_form_passes(play_script, text, calib, outcome):
    family = calib[outcome]
    code, out = hand_play(play_script, correct_lines(text, family), "--family", family["family_id"],
                          "--style", "4")
    assert code == 0
    assert "Result: PASS, reward 100 of 100" in out and "Parts that differ: none" in out
    assert "The episode is over: the request was closed." in out
    n_turns = len(reference_plan(family))
    assert f"Turns: {n_turns} (budget without a penalty: {n_turns + 1}). Malformed messages: 0." in out
    assert out.count("Parser: one valid call") == n_turns and "NOT a valid tool call" not in out
    assert f"--- Turn {n_turns} of 8" in out and f"--- Turn {n_turns + 1} of 8" not in out


def test_the_output_shows_the_request_the_tools_and_each_tool_result(play_script, text, calib):
    family = calib["approval"]
    code, out = hand_play(play_script, correct_lines(text, family), "--family", family["family_id"],
                          "--style", "2")
    request = next(w["text"] for w in family["wordings"] if w["style"] == 2)
    assert code == 0 and request in out and text.system_prompt in out
    assert all(json.dumps(spec.schema(), ensure_ascii=False) in out for spec in text.tool_specs("five"))
    assert text.tools["department_info"].name not in out
    assert "NOT in the chat format" in out
    assert out.index("End of the prompt") < out.index("--- Turn 1 of 8")
    # The expanded call and the answers of the tools are printed, in order.
    assert f"<function={text.tools['lookup_person'].name}>" in out
    answers = [line for line in out.splitlines() if line.startswith("{") and "type" not in line]
    assert len(answers) == 4 and family["truth"]["person_id"] in answers[0]
    assert json.loads(answers[1])[text.result_keys["sensitive"]] is True
    assert json.loads(answers[2]) == {text.result_keys["result"]: text.confirmation["approval_requested"]}
    # The expected state is shown at the end only.
    assert out.index("Expected state:") > out.index("The episode is over")
    assert f'approvals {json.dumps(family["expected"]["default"]["approvals"])}' in out


def test_the_expected_state_is_not_shown_before_the_end(play_script, text, calib):
    family = calib["approval"]
    lines = correct_lines(text, family).split("\n\n")
    code, out = hand_play(play_script, "\n\n".join(lines[:2]) + "\n\n", "--family", family["family_id"])
    assert code == 1 and "The input ended before the episode was over. Nothing is saved." in out
    assert "Expected state" not in out and "Result:" not in out
    assert family["truth"]["manager_id"] in out          # from the lookup answer, which a model sees too
    assert text.resolution["pending"] not in out.split("End of the prompt")[1]


def test_a_full_tool_call_typed_over_several_lines_is_sent_as_typed(play_script, text, calib):
    family = calib["inactive"]
    lookup, check, close = reference_plan(family)
    names = {key: text.tools[key] for key in ("lookup_person", "close_request")}
    full_call = (f"<tool_call>\n<function={names['lookup_person'].name}>\n"
                 f"<parameter={names['lookup_person'].params[0].name}>\n{lookup[1]['name']}\n</parameter>\n"
                 "</function>\n</tool_call>\n\n")
    typed = full_call + short_form(text, *check) + "\n\n" + short_form(text, *close) + "\n\n"
    code, out = hand_play(play_script, typed, "--family", family["family_id"])
    assert code == 0 and "Result: PASS, reward 100 of 100" in out


def test_empty_lines_before_a_message_are_skipped_and_send_nothing(play_script, text, calib):
    family = calib["already"]
    typed = "\n\n" + correct_lines(text, family).replace("\n\n", "\n\n\n  \n") + "\n\n"
    code, out = hand_play(play_script, typed, "--family", family["family_id"])
    assert code == 0 and "Result: PASS, reward 100 of 100" in out
    assert "Turns: 3 (budget without a penalty: 4). Malformed messages: 0." in out


def test_plain_text_is_a_malformed_turn_and_costs_five_points(play_script, text, calib):
    family = calib["grant"]
    typed = "I will look the person up now.\n\n" + correct_lines(text, family)
    code, out = hand_play(play_script, typed, "--family", family["family_id"])
    assert code == 0
    assert "Parser: NOT a valid tool call (no_call); this turn counts as malformed" in out
    assert text.errors["malformed_call"] in out
    assert "Turns: 5 (budget without a penalty: 5). Malformed messages: 1. Tool errors: 0." in out
    assert "Result: PASS, reward 95 of 100" in out


def test_a_wrong_decision_is_shown_as_a_fail_with_the_parts_that_differ(play_script, text, calib):
    family = calib["approval"]
    lookup, check, approval, close = reference_plan(family)
    grant = ("grant_role", {"person_id": family["truth"]["person_id"], "role_id": family["truth"]["role_id"]})
    granted = ("close_request", {**close[1], "resolution": "granted"})
    typed = "".join(short_form(text, *step) + "\n\n" for step in (lookup, check, grant, granted))
    code, out = hand_play(play_script, typed, "--family", family["family_id"])
    assert code == 0
    assert "Result: FAIL, reward 0 of 100" in out
    assert "Parts that differ: access, approvals, grants, request" in out
    pair = json.dumps([family["truth"]["person_id"], family["truth"]["role_id"]])
    assert f"pairs of the start, plus [{pair}]" in out and "pairs of the start, plus []" in out


def test_a_short_form_that_cannot_be_read_is_not_sent_and_uses_no_turn(play_script, text, calib):
    family = calib["already"]
    name = text.tools["lookup_person"].name
    no_key = f"{name} {family['request']['person_name']}\n\n"
    open_quote = f'{name} x="unclosed\n\n'
    typed = no_key + open_quote + correct_lines(text, family)
    code, out = hand_play(play_script, typed, "--family", family["family_id"])
    assert code == 0 and out.count("Not sent:") == 2 and out.count("This did not use a turn.") == 2
    assert out.count("--- Turn 1 of 8") == 3
    assert "Turns: 3 (budget without a penalty: 4). Malformed messages: 0. Tool errors: 0." in out
    assert "Result: PASS, reward 100 of 100" in out


def test_a_wrong_key_in_the_short_form_reaches_the_real_parser(play_script, text, calib):
    family = calib["already"]
    typed = f'{text.tools["lookup_person"].name} nme="x"\n\n' + correct_lines(text, family)
    code, out = hand_play(play_script, typed, "--family", family["family_id"])
    assert code == 0 and "NOT a valid tool call (unknown_argument)" in out and "reward 95 of 100" in out


def test_the_episode_stops_at_the_turn_limit(play_script, calib):
    family = calib["grant"]
    code, out = hand_play(play_script, "hello\n\n" * 9, "--family", family["family_id"])
    assert code == 0 and out.count("NOT a valid tool call") == 8
    assert "The episode is over: the turn limit was reached." in out
    assert "Turns: 8 (budget without a penalty: 5). Malformed messages: 8. Tool errors: 0." in out
    assert "Result: FAIL, reward 0 of 100" in out and "Parts that differ: access, grants, request" in out


def test_out_adds_the_record_to_the_file(play_script, text, cfg, calib, tmp_path):
    family, out_file = calib["grant"], tmp_path / "hand" / "played.jsonl"
    argv = ("--family", family["family_id"], "--style", "3", "--out", str(out_file))
    assert hand_play(play_script, correct_lines(text, family), *argv)[0] == 0
    code, out = hand_play(play_script, "no\n\n" + correct_lines(text, family), *argv)
    assert code == 0 and f"Added to {out_file} (2 episodes in the file)." in out
    records = read_records(out_file)
    assert [(r["run"], r["arm"], r["sample"], r["reward"]) for r in records] == \
        [("hand_play", "human", 0, 100), ("hand_play", "human", 1, 95)]
    assert all(r["scenario_id"] == f"{family['family_id']}/s3" for r in records)
    assert all(replay(r, family, text, cfg) == r["final_state"] for r in records)
    # An unfinished episode is not saved.
    assert hand_play(play_script, "no\n\n", *argv)[0] == 1 and len(read_records(out_file)) == 2


def test_a_test_family_is_refused_unless_it_is_allowed(play_script, text, capsys):
    family = load_families(SCENARIOS / "test.jsonl")[0]
    code, out = hand_play(play_script, correct_lines(text, family), "--family", family["family_id"])
    assert code == 2 and out == "" and "is a test family" in capsys.readouterr().err
    code, out = hand_play(play_script, correct_lines(text, family), "--family", family["family_id"],
                          "--allow-test")
    assert code == 0 and "Result: PASS" in out


def test_bad_arguments_give_a_plain_message(play_script, capsys, calib, tmp_path):
    assert hand_play(play_script, "", "--family", "ca-99")[0] == 2
    assert "no family 'ca-99'" in capsys.readouterr().err
    train = load_families(SCENARIOS / "train.jsonl")[0]
    assert hand_play(play_script, "", "--family", train["family_id"], "--style", "4")[0] == 2
    assert "no wording in style 4" in capsys.readouterr().err
    folder = tmp_path / "other"
    shutil.copytree(SCENARIOS, folder)
    summary = json.loads((folder / "summary.json").read_text())
    (folder / "summary.json").write_text(json.dumps({**summary, "text_sha256": "0" * 64}))
    out = io.StringIO()
    code = play_script.main(["--scenarios", str(folder), "--family", calib["grant"]["family_id"]],
                            stdin=io.StringIO(""), stdout=out)
    assert code == 2 and "another text file" in capsys.readouterr().err and out.getvalue() == ""
    (folder / "summary.json").unlink()
    assert hand_play(play_script, "", "--scenarios", str(folder), "--family", "ca-00")[0] == 2
    assert "summary.json is missing" in capsys.readouterr().err


def test_an_incomplete_text_file_is_reported_without_a_traceback(play_script, capsys):
    """text/model_text.yaml is empty until the candidate has written it."""
    from accessdesk.text import check_text, model_text_path
    if not check_text(model_text_path()):
        pytest.skip("the candidate's text file is complete")
    code, out = hand_play(play_script, "", "--family", "ca-00", "--text", str(model_text_path()))
    assert code == 2 and out == "" and "problem(s)" in capsys.readouterr().err


def test_the_policy_shift_prompt_brings_its_rule_set_and_six_tools_can_be_shown(play_script, text, calib):
    family = calib["grant"]            # under the policy-shift rules a plain grant needs an approval
    new_rule = reference_plan(family, text.policy_shift_variant)
    typed = "".join(short_form(text, *step) + "\n\n" for step in new_rule)
    code, out = hand_play(play_script, typed, "--family", family["family_id"], "--prompt", "policy_shift",
                          "--toolset", "six")
    assert code == 0 and f"rule set {text.policy_shift_variant}, 6 tools" in out
    assert text.policy_shift_system_prompt in out and text.tools["department_info"].name in out
    assert "Result: PASS, reward 100 of 100" in out
    old_rule = "".join(short_form(text, *step) + "\n\n" for step in reference_plan(family))
    code, out = hand_play(play_script, old_rule, "--family", family["family_id"], "--prompt", "policy_shift")
    assert code == 0 and "Result: FAIL, reward 0 of 100" in out


@pytest.mark.tokenizer
def test_with_the_tokenizer_the_prompt_is_the_chat_template_text(play_script, text, calib, tokenizer):
    family = calib["inactive"]
    code, out = hand_play(play_script, correct_lines(text, family), "--family", family["family_id"],
                          "--style", "1", tokenizer=TOKENIZER)
    request = next(w["text"] for w in family["wordings"] if w["style"] == 1)
    messages = [{"role": "system", "content": text.system_prompt}, {"role": "user", "content": request}]
    schemas = [spec.schema() for spec in text.tool_specs("five")]
    rendered = tokenizer.apply_chat_template(messages, tools=schemas, add_generation_prompt=True,
                                             enable_thinking=False, tokenize=False)
    assert code == 0 and "The prompt (exactly as the model gets it)" in out
    shown = out.split("The prompt (exactly as the model gets it)\n" + "=" * 78 + "\n")[1]
    shown = shown.split("\n" + "=" * 78 + "\nEnd of the prompt")[0]
    assert shown == rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Result: PASS, reward 100 of 100" in out


# ---------------------------------------------------------------------------
# The code hash in the run record, and the stop on a frozen tree that has changed: run_scripted.py
# ---------------------------------------------------------------------------
def scripted_manifest(out_file: Path) -> dict:
    return json.loads(Path(f"{out_file}.run_manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("state", [NOT_FROZEN, AS_FROZEN], ids=["not_frozen", "as_frozen"])
def test_a_scripted_run_writes_a_manifest_that_names_its_code_and_its_records(run_scripted, capsys, tmp_path,
                                                                              text, cfg, tree_state, state):
    tree_state["state"] = state
    out_file = tmp_path / "oracle.jsonl"
    code, out, err = scripted(run_scripted, capsys, "--split", "dev", "--policy", "oracle",
                              "--out", str(out_file))
    assert code == 0 and err == ""
    assert tree_state["roots"] == [ROOT]                         # the script asked about the repository
    manifest = scripted_manifest(out_file)
    assert f"wrote {out_file} and {out_file}.run_manifest.json" in out
    # The code: the hash and the whole answer of the freeze module, and that the option was not used.
    assert manifest["code_sha256"] == state["code_sha256"] and manifest["frozen"] == state
    assert manifest["allow_changed_tree"] is False
    # The records file is named by its hash, so the manifest belongs to this file and to no other.
    raw = out_file.read_bytes()
    assert manifest["records"] == {"path": str(out_file), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "bytes": len(raw)}
    assert (manifest["episodes"], manifest["families"]) == (192, 48)
    # What was run, and on what: the same hashes as in every record.
    assert manifest["script"] == "scripts/run_scripted.py"
    assert manifest["command"] == ["scripts/run_scripted.py", "--scenarios", str(SCENARIOS), "--split", "dev",
                                   "--policy", "oracle", "--out", str(out_file)]
    assert (manifest["policy"], manifest["split"], manifest["ruleset"], manifest["toolset"],
            manifest["seed"]) == ("oracle", "dev", "default", "five", 0)
    records = read_records(out_file)
    assert {record["data_sha256"] for record in records} == {manifest["split_file_sha256"]}
    assert {record["text_sha256"] for record in records} == {manifest["text_sha256"]} == {text.sha256}
    assert {record["config_sha256"] for record in records} == {manifest["config_sha256"]} == {cfg.sha256()}
    assert manifest["config"] == cfg.to_dict() and manifest["text_is_placeholder"] is True
    assert set(manifest["impl"]) >= {"verify", "reward"}


def test_a_scripted_run_stops_on_a_frozen_tree_that_has_changed(run_scripted, capsys, tmp_path, tree_state):
    tree_state["state"] = CHANGED
    out_file = tmp_path / "sub" / "oracle.jsonl"
    code, out, err = scripted(run_scripted, capsys, "--split", "dev", "--policy", "oracle",
                              "--out", str(out_file))
    # Exit code 2, every difference on its own line, and the two ways on. Nothing is written.
    assert code == 2 and out == ""
    assert "the tree is frozen, and it differs from MANIFEST.json:" in err
    for difference in CHANGED["differences"]:
        assert f"\n- {difference}\n" in err
    assert "scripts/freeze.py amend" in err and "--allow-changed-tree" in err
    assert not out_file.parent.exists()
    # --list runs no episode and is not stopped.
    code, out, _ = scripted(run_scripted, capsys, "--list")
    assert code == 0 and "oracle" in out.split()


def test_a_scripted_run_goes_on_on_a_changed_tree_only_with_the_option_and_says_so(run_scripted, capsys,
                                                                                    tmp_path, tree_state):
    tree_state["state"] = CHANGED
    out_file = tmp_path / "oracle.jsonl"
    code, out, err = scripted(run_scripted, capsys, "--split", "dev", "--policy", "oracle",
                              "--out", str(out_file), "--allow-changed-tree")
    assert code == 0 and err == "" and "all       100.0%" in out
    manifest = scripted_manifest(out_file)
    assert manifest["allow_changed_tree"] is True
    assert manifest["frozen"] == CHANGED and manifest["frozen"]["matches"] is False
    assert manifest["code_sha256"] == CHANGED["code_sha256"]
    assert manifest["command"][-1] == "--allow-changed-tree"
    # Before the freeze the option changes nothing but its own field.
    tree_state["state"] = NOT_FROZEN
    code, _, _ = scripted(run_scripted, capsys, "--split", "dev", "--policy", "oracle",
                          "--out", str(out_file), "--allow-changed-tree")
    manifest = scripted_manifest(out_file)
    assert code == 0 and manifest["allow_changed_tree"] is True and manifest["frozen"] == NOT_FROZEN


def test_a_scripted_run_on_the_repository_itself_follows_the_state_of_the_freeze(run_scripted, monkeypatch,
                                                                                 capsys, tmp_path):
    """No stand-in here: the real `frozen_state` reads the repository. What the script must do depends on
    the repository itself: on a frozen tree that has changed it stops; before the freeze, and on a tree
    that is as frozen, it runs and its manifest names the code hash that the freeze module gives."""
    monkeypatch.setattr(run_scripted, "frozen_state", freeze.frozen_state)
    state = freeze.frozen_state(ROOT)
    out_file = tmp_path / "oracle.jsonl"
    code, _, err = scripted(run_scripted, capsys, "--split", "calib", "--policy", "oracle",
                            "--out", str(out_file))
    if state["manifest_sha256"] is not None and not state["matches"]:
        assert code == 2 and not out_file.exists()
        assert all(f"- {difference}" in err for difference in state["differences"])
    else:
        manifest = scripted_manifest(out_file)
        assert code == 0 and manifest["frozen"] == state
        assert manifest["code_sha256"] == freeze.code_sha256(ROOT) and len(manifest["code_sha256"]) == 64


# ---------------------------------------------------------------------------
# A scenario file that was changed after the build stops a run (finding of 2026-11-17)
# ---------------------------------------------------------------------------
def edited_copy(tmp_path: Path) -> Path:
    """A copy of the scenario folder in which the expected grant record of every grant family of dev.jsonl
    is gone (the answer key was changed), and summary.json is as the build wrote it."""
    folder = tmp_path / "edited"
    shutil.copytree(SCENARIOS, folder)
    families = load_families(folder / "dev.jsonl")
    for family in families:
        if family["outcome"] == "grant":
            family["expected"]["default"]["grants"] = []
    lines = [json.dumps(family, ensure_ascii=False) + "\n" for family in families]
    (folder / "dev.jsonl").write_bytes("".join(lines).encode("utf-8"))
    return folder


def test_scripted_run_refuses_a_split_file_that_was_edited_after_the_build(run_scripted, capsys, tmp_path):
    """Before the repair this run went through and printed "grant 0.0% ... all 75.0%" for the oracle."""
    folder = edited_copy(tmp_path)
    out = tmp_path / "oracle.jsonl"
    code = run_scripted.main(["--scenarios", str(folder), "--split", "dev", "--policy", "oracle",
                              "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 2 and not out.exists()
    assert "dev.jsonl is not the file that the build wrote" in captured.err
    # The other split files of that folder are as built, and still run.
    code = run_scripted.main(["--scenarios", str(folder), "--split", "calib", "--policy", "oracle",
                              "--out", str(out)])
    assert code == 0 and "all       100.0%" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scripts/gpu/rollout.py: the parts that need no model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rollout_script():
    return load_script("gpu/rollout")


def test_rollout_script_refuses_a_file_that_was_edited_after_the_build(rollout_script, text, tmp_path):
    assert rollout_script.check_scenarios(SCENARIOS, text, ("dev.jsonl", "sets.json")) is None
    folder = edited_copy(tmp_path)
    problem = rollout_script.check_scenarios(folder, text, ("dev.jsonl",))
    assert problem is not None and "dev.jsonl is not the file that the build wrote" in problem
    assert rollout_script.check_scenarios(folder, text, ("calib.jsonl", "sets.json")) is None
    # sets.json is checked when a set is asked for.
    sets_file = folder / "sets.json"
    sets_file.write_text(sets_file.read_text(encoding="utf-8").replace("dv-00", "dv-01", 1), encoding="utf-8")
    problem = rollout_script.check_scenarios(folder, text, ("calib.jsonl", "sets.json"))
    assert problem is not None and "sets.json is not the file that the build wrote" in problem
    (folder / "test.jsonl").unlink()
    assert "is missing" in rollout_script.check_scenarios(folder, text, ("test.jsonl",))


def test_rollout_script_takes_the_families_of_a_named_set(rollout_script):
    sets = json.loads((SCENARIOS / "sets.json").read_text(encoding="utf-8"))
    wanted = {"curve_subset": sets["curve_subset"], "policy_shift_differ": sets["policy_shift"]["differ"],
              "policy_shift_control": sets["policy_shift"]["control"], "extra_tool": sets["extra_tool"]}
    assert tuple(wanted) == rollout_script.SETS
    assert [len(ids) for ids in wanted.values()] == [20, 11, 5, 16]
    families = load_families(SCENARIOS / "dev.jsonl")
    for name, ids in wanted.items():
        assert rollout_script.set_family_ids(SCENARIOS, name) == ids
        args = rollout_script.parse_args(["--model", "m", "--scenarios", str(SCENARIOS), "--split", "dev",
                                          "--out", "x", "--set", name])
        found, source = rollout_script.wanted_family_ids(args)
        assert found == ids and name in source
        chosen = rollout_script.select_families(families, found, None, source)
        assert sorted(family["family_id"] for family in chosen) == sorted(ids)
        assert chosen == [family for family in families if family in chosen]        # file order is kept
    # No list at all: every family. A hand-made file of ids still works. An unknown id stops the run.
    args = rollout_script.parse_args(["--model", "m", "--scenarios", str(SCENARIOS), "--split", "dev",
                                      "--out", "x"])
    assert rollout_script.wanted_family_ids(args) == (None, "")
    assert rollout_script.select_families(families, None, 3) == families[:3]
    with pytest.raises(SystemExit, match="not in the split"):
        rollout_script.select_families(families, ["dv-00", "te-00"], None, "the set x")


def test_rollout_script_does_not_take_a_set_and_a_file_of_ids_together(rollout_script, capsys):
    with pytest.raises(SystemExit):
        rollout_script.parse_args(["--model", "m", "--scenarios", str(SCENARIOS), "--split", "dev",
                                   "--out", "x", "--set", "curve_subset", "--families", "ids.txt"])
    assert "not allowed with argument" in capsys.readouterr().err


class FakeSampler:
    def __init__(self, model, tokenizer, cfg) -> None:
        self.cfg = cfg

    def settings(self) -> dict:
        return {}


def run_rollout_main(rollout_script, monkeypatch, tmp_path: Path, text, *argv: str,
                     loads: list | None = None) -> tuple[int, list[dict]]:
    """main() of scripts/gpu/rollout.py with the model replaced: `load_model`, the sampler and the token
    layout are stand-ins, and the stand-in for `rollout` plays the scripted oracle and writes the record
    fields from `meta`, as the real rollout() does. Returns (exit code, the `meta` of every rollout call).
    With `loads`, every call of `load_model` adds the model's name to that list."""
    from accessdesk.harness import hf_backend, tokens

    calls: list[dict] = []
    policy = oracle(text)

    def fake_load_model(name, *args, **kwargs):
        if loads is not None:
            loads.append(name)
        return None, None

    def fake_rollout(sampler, layouts, episodes, *, run_seed, pass_index, meta):
        calls.append(dict(meta))
        fields = {key: value for key, value in meta.items() if key != "first_sample"}
        records = []
        for pos, episode in enumerate(episodes):
            while not episode.done:
                episode.step(policy(episode.messages, episode.tool_specs), n_tokens=1)
            seed = {"run_seed": run_seed, "pass": pass_index, "chunk": 0, "pos": pos}
            record = episode.record(**fields, sample=meta.get("first_sample", 0), seed=seed)
            records.append({**record, "tokens": {"ids": [], "segments": [], "logprobs": []}})
        return records

    monkeypatch.setattr(hf_backend, "load_model", fake_load_model)
    monkeypatch.setattr(hf_backend, "Sampler", FakeSampler)
    monkeypatch.setattr(hf_backend, "environment_report", lambda: {})
    monkeypatch.setattr(hf_backend, "kernel_report", lambda model: {})
    monkeypatch.setattr(hf_backend, "rollout", fake_rollout)
    monkeypatch.setattr(tokens, "TokenLayout", lambda tokenizer, tool_specs: None)
    monkeypatch.setattr(rollout_script, "peak_gpu_memory_gb", lambda: None)
    monkeypatch.setattr(rollout_script, "SMOKE_DIR", tmp_path)       # the placeholder text may write here
    code = rollout_script.main(["--model", "Qwen/Qwen3.5-9B",
                                "--text", str(ROOT / "text" / "placeholder_text.yaml"),
                                "--allow-placeholder", "--scenarios", str(SCENARIOS), *argv])
    return code, calls


def test_rollout_script_gives_the_hash_of_the_split_file_to_every_record(rollout_script, monkeypatch,
                                                                         tmp_path, text, capsys):
    """Small task (a), and the finding that the model runs carried no `data_sha256`: the script passes the
    hash of the split file (the one in its manifest) in the `meta` of every rollout() call."""
    out = tmp_path / "run"
    code, calls = run_rollout_main(rollout_script, monkeypatch, tmp_path, text, "--split", "dev",
                                   "--set", "curve_subset", "--samples", "2", "--out", str(out))
    capsys.readouterr()
    assert code == 0
    of_the_file = hashlib.sha256((SCENARIOS / "dev.jsonl").read_bytes()).hexdigest()
    assert [call["data_sha256"] for call in calls] == [of_the_file, of_the_file]
    assert [call["first_sample"] for call in calls] == [0, 1]
    records = read_records(out / "episodes.jsonl")
    sets = json.loads((SCENARIOS / "sets.json").read_text(encoding="utf-8"))
    # 20 families of the curve subset, 4 wordings, 2 samples; no other family.
    assert len(records) == 20 * 4 * 2 and all(record["data_sha256"] == of_the_file for record in records)
    assert sorted({record["family_id"] for record in records}) == sets["curve_subset"]
    assert all("tokens" not in record for record in records)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True and manifest["split_file_sha256"] == of_the_file
    assert manifest["family_set"] == "curve_subset" and manifest["families"] == 20
    assert manifest["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"       # the pinned 9B


def test_rollout_script_needs_the_dev_split_for_a_set(rollout_script, monkeypatch, tmp_path, text, capsys):
    code, calls = run_rollout_main(rollout_script, monkeypatch, tmp_path, text, "--split", "calib",
                                   "--set", "extra_tool", "--out", str(tmp_path / "run"))
    assert code == 2 and calls == [] and "--set needs --split dev" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


# ---------------------------------------------------------------------------
# The code hash in the run record, and the stop on a frozen tree that has changed: rollout.py
# ---------------------------------------------------------------------------
ONE_PASS = ("--split", "dev", "--set", "curve_subset", "--styles", "1", "--samples", "1")


@pytest.mark.parametrize("state", [NOT_FROZEN, AS_FROZEN], ids=["not_frozen", "as_frozen"])
def test_the_run_manifest_of_a_rollout_names_the_code_that_made_it(rollout_script, monkeypatch, tmp_path,
                                                                   text, capsys, tree_state, state):
    tree_state["state"] = state
    out = tmp_path / "run"
    code, calls = run_rollout_main(rollout_script, monkeypatch, tmp_path, text, *ONE_PASS, "--out", str(out))
    assert code == 0 and len(calls) == 1 and capsys.readouterr().err == ""
    assert tree_state["roots"] == [ROOT]                         # the script asked about the repository
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["code_sha256"] == state["code_sha256"] and manifest["frozen"] == state
    assert manifest["allow_changed_tree"] is False


def test_a_rollout_stops_on_a_frozen_tree_that_has_changed_before_the_model_loads(rollout_script, monkeypatch,
                                                                                  tmp_path, text, capsys,
                                                                                  tree_state):
    tree_state["state"] = CHANGED
    out, loads = tmp_path / "run", []
    code, calls = run_rollout_main(rollout_script, monkeypatch, tmp_path, text, *ONE_PASS, "--out", str(out),
                                   loads=loads)
    captured = capsys.readouterr()
    # Exit code 2, every difference on its own line, and the two ways on. No model, no episode, no file.
    assert code == 2 and calls == [] and loads == [] and not out.exists()
    assert "the tree is frozen, and it differs from MANIFEST.json:" in captured.err
    for difference in CHANGED["differences"]:
        assert f"\n- {difference}\n" in captured.err
    assert "scripts/freeze.py amend" in captured.err and "--allow-changed-tree" in captured.err
    # A wrong argument is still reported first: it needs no look at the tree.
    tree_state["roots"].clear()
    code, calls = run_rollout_main(rollout_script, monkeypatch, tmp_path, text, "--split", "calib",
                                   "--set", "extra_tool", "--out", str(out))
    assert code == 2 and "--set needs --split dev" in capsys.readouterr().err and tree_state["roots"] == []


def test_a_rollout_goes_on_on_a_changed_tree_only_with_the_option_and_says_so(rollout_script, monkeypatch,
                                                                               tmp_path, text, capsys,
                                                                               tree_state):
    tree_state["state"] = CHANGED
    out, loads = tmp_path / "run", []
    code, calls = run_rollout_main(rollout_script, monkeypatch, tmp_path, text, *ONE_PASS, "--out", str(out),
                                   "--allow-changed-tree", loads=loads)
    capsys.readouterr()
    assert code == 0 and len(calls) == 1 and loads == ["Qwen/Qwen3.5-9B"]
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True and manifest["allow_changed_tree"] is True
    assert manifest["frozen"] == CHANGED and manifest["frozen"]["matches"] is False
    assert manifest["code_sha256"] == CHANGED["code_sha256"]
    assert len(read_records(out / "episodes.jsonl")) == 20


def test_the_real_freeze_module_gives_the_rollout_script_a_state_that_fits_a_manifest(rollout_script):
    """No stand-in here: the real `frozen_state` reads the repository, as the script does at its start. Its
    answer has the four keys, can be written as JSON, and names the code hash of the freeze module."""
    state = freeze.frozen_state(rollout_script.ROOT)
    assert rollout_script.ROOT == ROOT
    assert set(state) == {"manifest_sha256", "code_sha256", "matches", "differences"}
    assert state["code_sha256"] == freeze.code_sha256(ROOT) and len(state["code_sha256"]) == 64
    assert json.loads(json.dumps(state)) == state
    # Before the freeze nothing stops a run; after it, only a tree with a difference does.
    problem = freeze.run_problem(state)
    assert (problem is not None) == (state["manifest_sha256"] is not None and not state["matches"])


def test_the_9b_revision_is_pinned():
    from accessdesk.harness.hf_backend import PINNED_REVISIONS, resolve_revision

    assert PINNED_REVISIONS["Qwen/Qwen3.5-9B"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert resolve_revision("Qwen/Qwen3.5-9B", None) == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert resolve_revision("Qwen/Qwen3.5-9B", "other") == "other"
    assert all(len(revision) == 40 and set(revision) <= set("0123456789abcdef")
               for revision in PINNED_REVISIONS.values())


# ---------------------------------------------------------------------------
# How a script is started (finding of 2026-11-17)
# ---------------------------------------------------------------------------
def test_no_script_needs_its_docstring_to_start():
    """`python -OO` strips docstrings: `__doc__` is then None, and `__doc__.split(...)` stopped eight
    scripts before they did anything."""
    scripts = sorted((ROOT / "scripts").rglob("*.py"))
    assert len(scripts) >= 10
    for path in scripts:
        assert "__doc__.split" not in path.read_text(encoding="utf-8"), path
    result = subprocess.run([sys.executable, "-OO", str(ROOT / "scripts" / "run_scripted.py"), "--list"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0 and "oracle" in result.stdout.split()


# ---------------------------------------------------------------------------
# scripts/check_wordings.py: shared word 5-grams between train and the other splits (PLAN 5.3)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def check_wordings():
    return load_script("check_wordings")


def wording_family(split: str, texts: dict[int, str]) -> dict:
    """A family record with only what the script reads: the request and one wording text per style."""
    request = {"person_name": "Ada Low", "department": "Finance", "role_name": "Ledger Viewer"}
    return {"split": split, "request": request,
            "wordings": [{"style": style, "template": 0, "text": text.format(**request)}
                         for style, text in texts.items()]}


def test_masked_words_and_five_grams_by_hand(check_wordings):
    request = {"person_name": "Ada Low", "department": "Finance", "role_name": "Finance Ledger Viewer"}
    text = "Please give Ada Low of Finance the Finance Ledger Viewer role today"
    # The role name holds the department's word: the longer value is masked first.
    words = check_wordings.masked_words(text, request)
    assert words == ["please", "give", "<person>", "of", "<department>", "the", "<role>", "role", "today"]
    grams = check_wordings.n_grams(words)
    assert len(grams) == 5 and ("please", "give", "<person>", "of", "<department>") in grams
    assert ("the", "<role>", "role", "today") not in grams           # four words are not a 5-gram
    assert check_wordings.n_grams(words[:4]) == set()


def test_shared_five_grams_by_hand(check_wordings):
    seen = "please give {person_name} of {department} the {role_name} role"            # 8 words: 4 5-grams
    held_out = "{role_name} is needed by {person_name} who works in {department}"      # 9 words: 5 5-grams
    families = {"train": [wording_family("train", {1: seen})],
                "dev": [wording_family("dev", {1: seen, 4: held_out})]}
    rows = check_wordings.shared_table(families)
    assert rows == [
        {"split": "dev", "style": 1, "held_out": False, "wordings": 1, "five_grams": 4,
         "shared_same_style": 4, "shared_any_style": 4},
        {"split": "dev", "style": 4, "held_out": True, "wordings": 1, "five_grams": 5,
         "shared_same_style": 0, "shared_any_style": 0}]
    assert check_wordings.problems(rows, families["train"]) == []
    # A held-out wording that takes over the first six words of the seen one shares its first two 5-grams.
    leaky = "please give {person_name} of {department} the access called {role_name}"
    families["dev"] = [wording_family("dev", {1: seen, 4: leaky})]
    rows = check_wordings.shared_table(families)
    assert (rows[1]["five_grams"], rows[1]["shared_same_style"], rows[1]["shared_any_style"]) == (5, 0, 2)
    assert check_wordings.problems(rows, families["train"]) == [
        "dev, style 4 (held out): 2 of 5 5-grams are also in a train wording"]
    # A train family with a wording in the held-out style: said, and all its 5-grams are shared.
    families["train"].append(wording_family("train", {4: leaky}))
    rows = check_wordings.shared_table(families)
    assert (rows[1]["shared_same_style"], rows[1]["shared_any_style"]) == (5, 5)
    assert check_wordings.problems(rows, families["train"]) == [
        "train has wordings in the held-out style 4",
        "dev, style 4 (held out): 5 of 5 5-grams are also in a train wording"]


def test_the_held_out_style_of_the_scenario_files_shares_no_five_gram_with_train(check_wordings, capsys,
                                                                                 tmp_path):
    out = tmp_path / "wordings.json"
    assert check_wordings.main(["--scenarios", str(SCENARIOS), "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("split  style  held out  wordings  5-grams")
    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["n"] == 5 and table["problems"] == []
    assert [(row["split"], row["style"]) for row in table["rows"]] == \
        [(split, style) for split in ("dev", "test", "calib") for style in (1, 2, 3, 4)]
    for row in table["rows"]:
        assert row["wordings"] == 3 and row["five_grams"] > 0          # three templates per style
        if row["style"] == 4:
            assert row["held_out"] and row["shared_any_style"] == 0
        else:                                                          # the same templates as in train
            assert row["shared_same_style"] == row["five_grams"]
    # A train file that holds a held-out wording is found, and the script ends with code 1.
    folder = tmp_path / "leaky"
    shutil.copytree(SCENARIOS, folder)
    train, dev = load_families(folder / "train.jsonl"), load_families(folder / "dev.jsonl")
    train[0]["request"], train[0]["wordings"] = dev[0]["request"], dev[0]["wordings"]
    lines = [json.dumps(family, ensure_ascii=False) + "\n" for family in train]
    (folder / "train.jsonl").write_bytes("".join(lines).encode("utf-8"))
    assert check_wordings.main(["--scenarios", str(folder)]) == 1
    err = capsys.readouterr().err
    assert "PROBLEM: train has wordings in the held-out style 4" in err
    assert "PROBLEM: dev, style 4 (held out):" in err
