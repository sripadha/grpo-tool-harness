"""The rebuild command (scripts/rebuild_results.sh): what it lists, how it compares, and its exit code.

The first part uses a tiny made-up results folder in a temporary folder: no saved run of results/main/ is
read or written there. The state "the test runs exist" is tried only with made-up files (empty episodes
files and made-up run manifests) in --list mode, which runs no command. The comparison rules (P: the path
of the fresh folder, outside and inside the repository; T: an ISO-8601 time; M: the manifest hash of a run
manifest) are tried on the comparison code of the script itself.

The last part runs the command on the saved runs of results/main/, if they are there: --list must name
every table file of results/main/tables/ and every monitor report, and a run of the cheaper steps must
find every made file equal (or equal by rule). The test never runs the audit sheet step.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rebuild_results.sh"
PYTHON = sys.executable
ARMS = ("base", "sft", "grpo_seed1", "grpo_seed2")
REAL_MAIN = ROOT / "results" / "main"


def rebuild(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHON=PYTHON)
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout)


def compare_code() -> str:
    text = SCRIPT.read_text()
    start = text.index("COMPARE_PY='") + len("COMPARE_PY='")
    return text[start:text.index("\n'\n", start)]


def run_compare(made: Path, saved: Path, fresh: str, parent: str, fresh_rel: str = "",
                hide: str = "", history: str = "", present: str = "") -> subprocess.CompletedProcess:
    """The comparison code of the script. history: the file of the states of MANIFEST.json; present: the
    SHA-256 of the present MANIFEST.json (rule M)."""
    return subprocess.run([PYTHON, "-c", compare_code(), str(made), str(saved), fresh, parent, fresh_rel, hide,
                           history, present], capture_output=True, text=True)


def history_file(folder: Path, *hashes: str) -> str:
    """A made-up file of the states of MANIFEST.json, one row per hash."""
    path = folder / "manifest_history.csv"
    path.write_text("sha256,state,utc,source\n" + "".join(
        f"{value},state {number},2026-11-18T21:48:10Z,made up\n" for number, value in enumerate(hashes, 1)))
    return str(path)


# ---- the comparison rules ----

def test_equal_bytes_are_equal(tmp_path):
    (tmp_path / "a").write_text("x 1\n")
    (tmp_path / "b").write_text("x 1\n")
    done = run_compare(tmp_path / "a", tmp_path / "b", "/fresh", "results")
    assert done.returncode == 0 and done.stdout.startswith("equal  ")


def test_rule_p_replaces_only_the_fresh_folder(tmp_path):
    (tmp_path / "made").write_text('{"out": "/tmp/fresh9/main/tables/t.json", "n": 3}\n')
    (tmp_path / "saved").write_text('{"out": "results/main/tables/t.json", "n": 3}\n')
    done = run_compare(tmp_path / "made", tmp_path / "saved", "/tmp/fresh9", "results")
    assert done.returncode == 0
    assert done.stdout.startswith("equal by rule") and "1 line(s) differ only in the path" in done.stdout


def test_rule_p_does_not_hide_a_number(tmp_path):
    (tmp_path / "made").write_text('{"out": "/tmp/fresh9/main/t.json", "n": 4}\n')
    (tmp_path / "saved").write_text('{"out": "results/main/t.json", "n": 3}\n')
    done = run_compare(tmp_path / "made", tmp_path / "saved", "/tmp/fresh9", "results")
    assert done.returncode == 1 and done.stdout.startswith("DIFFERENT")
    assert '-{"out": "results/main/t.json", "n": 3}' in done.stdout


def test_rule_p_with_the_fresh_folder_outside_the_repository(tmp_path):
    """Outside the repository the scripts write the absolute path; no relative form is given."""
    (tmp_path / "made").write_text("see /tmp/fresh9/main/monitors/base_dev.md\nn 3\n")
    (tmp_path / "saved").write_text("see results/main/monitors/base_dev.md\nn 3\n")
    done = run_compare(tmp_path / "made", tmp_path / "saved", "/tmp/fresh9", "results", "")
    assert done.returncode == 0 and "1 line(s) differ only in the path" in done.stdout


def test_rule_p_with_the_fresh_folder_inside_the_repository(tmp_path):
    """Inside the repository the scripts write the path relative to the repository: rule P replaces it too,
    and the absolute form as well (the absolute path ends with the relative one)."""
    fresh = "/home/x/repo/_dev/report_code/fresh"
    (tmp_path / "made").write_text("- OK: base. All 11 lines are in `_dev/report_code/fresh/main/monitors/base_dev.md`.\n"
                                   f"file {fresh}/main/report/x.md\nn 3\n")
    (tmp_path / "saved").write_text("- OK: base. All 11 lines are in `results/main/monitors/base_dev.md`.\n"
                                    "file results/main/report/x.md\nn 3\n")
    done = run_compare(tmp_path / "made", tmp_path / "saved", fresh, "results", "_dev/report_code/fresh")
    assert done.returncode == 0, done.stdout
    assert done.stdout.startswith("equal by rule") and "2 line(s) differ only in the path" in done.stdout
    # Without the relative form (the defect of 2026-11-19) the first line differs.
    done = run_compare(tmp_path / "made", tmp_path / "saved", fresh, "results", "")
    assert done.returncode == 1 and done.stdout.startswith("DIFFERENT")
    # The relative form does not hide a changed number.
    (tmp_path / "made").write_text("- OK: base. All 12 lines are in `_dev/report_code/fresh/main/monitors/base_dev.md`.\n"
                                   f"file {fresh}/main/report/x.md\nn 3\n")
    done = run_compare(tmp_path / "made", tmp_path / "saved", fresh, "results", "_dev/report_code/fresh")
    assert done.returncode == 1 and "+- OK: base. All 12 lines" in done.stdout


def test_the_command_names_both_forms_of_the_fresh_folder_inside_the_repository(tmp_path):
    """The bash part: a fresh folder inside the repository gives rule P its relative form; one outside
    does not. The step `tables` stops at once here (its inputs are missing), so nothing is run."""
    main = tmp_path / "results" / "main"
    main.mkdir(parents=True)
    inside = ROOT / f".rebuild_test_{os.getpid()}"
    try:
        done = rebuild("--results", str(main), "--only", "tables", "--out", str(inside))
        assert f"rule P: in a made file, {inside} and {inside.name} (the same folder, relative to the " \
               f"repository) are read as {tmp_path / 'results'}" in done.stdout
    finally:
        import shutil
        shutil.rmtree(inside, ignore_errors=True)
    done = rebuild("--results", str(main), "--only", "tables", "--out", str(tmp_path / "fresh"))
    assert f"rule P: in a made file, {tmp_path / 'fresh'} is read as {tmp_path / 'results'}" in done.stdout


def manifest_text(manifest_hash: str, code: str = "c" * 64, episodes: int = 192,
                  out: str = "results/main/reference/oracle_dev.jsonl") -> str:
    return json.dumps({"command": ["scripts/run_scripted.py", "--out", out], "code_sha256": code,
                       "frozen": {"manifest_sha256": manifest_hash, "code_sha256": code, "matches": True,
                                  "differences": []},
                       "episodes": episodes, "started": "2026-11-18T21:50:00Z"}, indent=2) + "\n"


def test_rule_m_a_run_manifest_under_an_amended_manifest_with_the_same_code(tmp_path):
    made, saved = tmp_path / "x.run_manifest.json", tmp_path / "saved.run_manifest.json"
    made.write_text(manifest_text("b" * 64, out="/tmp/fresh9/main/reference/oracle_dev.jsonl").replace(
        "2026-11-18T21:50:00Z", "2026-11-19T17:00:00Z"))
    saved.write_text(manifest_text("a" * 64))
    history = history_file(tmp_path, "a" * 64, "b" * 64)
    done = run_compare(made, saved, "/tmp/fresh9", "results", history=history, present="b" * 64)
    assert done.returncode == 0, done.stdout
    assert done.stdout.startswith(f"equal by rule M  {saved}: frozen.manifest_sha256 is {'a' * 64} (state 1) in "
                                  f"the saved file and {'b' * 64} (the present MANIFEST.json) in the made one; "
                                  f"code_sha256 and frozen.code_sha256 are {'c' * 64} in both.")
    assert "only MANIFEST.json changed" in done.stdout and "1 by rule P, 1 by rule T" in done.stdout


def test_rule_m_needs_a_recorded_saved_hash_and_the_present_made_hash(tmp_path):
    """Rule M is tight: the saved hash must be a row of the history file, and the made hash the SHA-256 of
    the present MANIFEST.json. Else the file is DIFFERENT, and a line says which condition failed."""
    made, saved = tmp_path / "made.run_manifest.json", tmp_path / "saved.run_manifest.json"
    made.write_text(manifest_text("b" * 64))
    saved.write_text(manifest_text("a" * 64))
    # The saved hash is no recorded state of MANIFEST.json.
    done = run_compare(made, saved, "/fresh", "results", history=history_file(tmp_path, "e" * 64, "b" * 64),
                       present="b" * 64)
    assert done.returncode == 1 and "DIFFERENT" in done.stdout
    assert f"rule M does not apply to {saved}: the saved frozen.manifest_sha256 is not a row of" in done.stdout
    # The made hash is not the one of the present MANIFEST.json.
    done = run_compare(made, saved, "/fresh", "results", history=history_file(tmp_path, "a" * 64, "b" * 64),
                       present="f" * 64)
    assert done.returncode == 1 and "DIFFERENT" in done.stdout
    assert "the made frozen.manifest_sha256 is not the SHA-256 of the present MANIFEST.json" in done.stdout
    # No history file, or no present hash: rule M applies to no file.
    for history, present in (("", "b" * 64), (str(tmp_path / "absent.csv"), "b" * 64),
                             (history_file(tmp_path, "a" * 64, "b" * 64), "")):
        done = run_compare(made, saved, "/fresh", "results", history=history, present=present)
        assert done.returncode == 1 and "DIFFERENT" in done.stdout, (history, present)
    # Both conditions hold: equal by rule M.
    done = run_compare(made, saved, "/fresh", "results", history=history_file(tmp_path, "a" * 64, "b" * 64),
                       present="b" * 64)
    assert done.returncode == 0 and done.stdout.startswith("equal by rule M")


def test_the_history_file_holds_every_state_of_manifest_json_so_far():
    """results/manifest_history.csv: the freeze (commit a1e2dfc) and the state after amendment 1 (commit
    b6f3565), checked against git when it is there; and the present MANIFEST.json is one of its rows."""
    path = ROOT / "results" / "manifest_history.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["sha256", "state", "utc", "source"]
    assert [(row["sha256"], row["state"], row["utc"]) for row in rows[:2]] == [
        ("f1ea719eadb4e16a25af51ee9b4ec7050097de3df451469d0b001833b85a99e9", "the freeze", "2026-11-18T21:48:10Z"),
        ("f7eaa77fade932ce24c690f0fb987f62f06c5fd1d025bfed4a2f5d794a294d81", "after amendment 1",
         "2026-11-19T06:53:40Z")]
    assert len({row["sha256"] for row in rows}) == len(rows)
    present = hashlib.sha256((ROOT / "MANIFEST.json").read_bytes()).hexdigest()
    assert present in {row["sha256"] for row in rows}, "add a row for the present MANIFEST.json"
    for commit, value in (("a1e2dfc", rows[0]["sha256"]), ("b6f3565", rows[1]["sha256"])):
        shown = subprocess.run(["git", "show", f"{commit}:MANIFEST.json"], cwd=ROOT, capture_output=True)
        if shown.returncode != 0:
            pytest.skip("the git history is not in this copy")
        assert hashlib.sha256(shown.stdout).hexdigest() == value, commit


def test_the_command_says_when_the_present_manifest_is_not_in_the_history(tmp_path):
    main = tmp_path / "results" / "main"
    make_reference(main)
    history = tmp_path / "history.csv"
    history.write_text("sha256,state,utc,source\n" + "0" * 64 + ",old,2026-11-18T21:48:10Z,made up\n")
    present = hashlib.sha256((ROOT / "MANIFEST.json").read_bytes()).hexdigest()
    done = rebuild("--results", str(main), "--only", "reference", "--out", str(tmp_path / "fresh1"),
                   "--manifest-history", str(history))
    lines = [line for line in done.stdout.splitlines() if line.startswith("MANIFEST HISTORY")]
    assert lines == [f"MANIFEST HISTORY  the present MANIFEST.json (SHA-256 {present}) is not a row of {history}: "
                     "add a row for it (the main session adds one after each amendment)"]
    assert done.returncode == 1 and "1 missing input(s)" in done.stdout
    # With the row, the same command is quiet about it and ends with 0.
    history.write_text(history.read_text() + f"{present},now,2026-11-19T06:53:40Z,made up\n")
    done = rebuild("--results", str(main), "--only", "reference", "--out", str(tmp_path / "fresh2"),
                   "--manifest-history", str(history))
    assert "MANIFEST HISTORY" not in done.stdout and done.returncode == 0, done.stdout
    assert f"rule M: the present MANIFEST.json (SHA-256 {present}) is a row of {history} (2 state(s))" in done.stdout


def test_rule_m_does_not_apply_when_the_code_differs_or_another_field_differs(tmp_path):
    made, saved = tmp_path / "made.run_manifest.json", tmp_path / "saved.run_manifest.json"
    saved.write_text(manifest_text("a" * 64))
    # Another code hash: DIFFERENT, with the manifest hash among the lines shown.
    made.write_text(manifest_text("b" * 64, code="d" * 64))
    done = run_compare(made, saved, "/fresh", "results")
    assert done.returncode == 1 and done.stdout.startswith("DIFFERENT")
    assert f'+    "manifest_sha256": "{"b" * 64}",' in done.stdout
    # Only the top-level code_sha256 differs: DIFFERENT too.
    made.write_text(manifest_text("b" * 64).replace(f'"code_sha256": "{"c" * 64}"', f'"code_sha256": "{"d" * 64}"', 1))
    done = run_compare(made, saved, "/fresh", "results")
    assert done.returncode == 1 and done.stdout.startswith("DIFFERENT")
    # The same code but another field differs as well: DIFFERENT, and that field is shown.
    made.write_text(manifest_text("b" * 64, episodes=191))
    done = run_compare(made, saved, "/fresh", "results")
    assert done.returncode == 1 and '+  "episodes": 191,' in done.stdout
    # A file that is not a run manifest gets no rule M.
    other_made, other_saved = tmp_path / "made.json", tmp_path / "saved.json"
    other_made.write_text(manifest_text("b" * 64))
    other_saved.write_text(manifest_text("a" * 64))
    assert run_compare(other_made, other_saved, "/fresh", "results").returncode == 1


def test_a_hidden_file_shows_no_line_of_its_content(tmp_path):
    (tmp_path / "made").write_text("pass rate 0.5\n")
    (tmp_path / "saved").write_text("pass rate 0.4\n")
    done = run_compare(tmp_path / "made", tmp_path / "saved", "/fresh", "results", "", "hide")
    assert done.returncode == 1 and done.stdout.startswith("DIFFERENT") and "no line is shown" in done.stdout
    assert "0.5" not in done.stdout and "0.4" not in done.stdout


def test_rule_t_masks_only_a_time(tmp_path):
    (tmp_path / "made").write_text("made at 2026-11-19T16:00:01Z\nrows 5\n")
    (tmp_path / "saved").write_text("made at 2026-11-18T09:12:00Z\nrows 5\n")
    done = run_compare(tmp_path / "made", tmp_path / "saved", "/fresh", "results")
    assert done.returncode == 0 and "1 line(s) only in a time (rule T)" in done.stdout
    (tmp_path / "made").write_text("made at 2026-11-19T16:00:01Z\nrows 6\n")
    done = run_compare(tmp_path / "made", tmp_path / "saved", "/fresh", "results")
    assert done.returncode == 1 and "-rows 5" in done.stdout and "+rows 6" in done.stdout


def test_a_binary_file_is_compared_by_its_bytes(tmp_path):
    (tmp_path / "made.png").write_bytes(b"\x89PNG\xff\x00")
    (tmp_path / "saved.png").write_bytes(b"\x89PNG\xfe\x00")
    done = run_compare(tmp_path / "made.png", tmp_path / "saved.png", "/fresh", "results")
    assert done.returncode == 1 and "not a text file" in done.stdout


def test_a_missing_saved_file_is_new_and_a_missing_made_file_is_not_made(tmp_path):
    (tmp_path / "made").write_text("a\n")
    done = run_compare(tmp_path / "made", tmp_path / "none", "/fresh", "results")
    assert done.returncode == 3 and done.stdout.startswith("NEW")
    done = run_compare(tmp_path / "none", tmp_path / "made", "/fresh", "results")
    assert done.returncode == 4 and done.stdout.startswith("SAVED, NOT MADE")


# ---- the command on a tiny made-up results folder ----

def make_reference(main: Path) -> Path:
    """One reference row file, made by the frozen scripts/run_scripted.py (oracle on dev, 192 attempts)."""
    out = main / "reference" / "oracle_dev.jsonl"
    out.parent.mkdir(parents=True)
    subprocess.run([PYTHON, "scripts/run_scripted.py", "--scenarios", "data/scenarios", "--split", "dev",
                    "--text", "text/model_text.yaml", "--policy", "oracle", "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return out


def test_the_reference_step_on_a_made_up_folder_is_equal_and_finds_a_change(tmp_path):
    main = tmp_path / "results" / "main"
    rows = make_reference(main)
    done = rebuild("--results", str(main), "--only", "reference", "--out", str(tmp_path / "fresh1"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"equal  {main}/reference/oracle_dev.jsonl" in done.stdout
    # The run manifest names its own --out: it is equal by rule P (or by rule M if MANIFEST.json changed).
    assert f"equal by rule  {main}/reference/oracle_dev.jsonl.run_manifest.json" in done.stdout
    assert "the test runs do not exist" in done.stdout
    assert "summary: 1 equal, 1 equal by rule, 0 DIFFERENT" in done.stdout

    lines = rows.read_text().splitlines()
    record = json.loads(lines[0])
    record["reward"] = record["reward"] + 1
    lines[0] = json.dumps(record)
    rows.write_text("\n".join(lines) + "\n")
    done = rebuild("--results", str(main), "--only", "reference", "--out", str(tmp_path / "fresh2"))
    assert done.returncode == 1
    assert f"DIFFERENT  {main}/reference/oracle_dev.jsonl" in done.stdout


def test_a_missing_input_gives_exit_code_1(tmp_path):
    main = tmp_path / "results" / "main"
    main.mkdir(parents=True)
    done = rebuild("--results", str(main), "--only", "tables", "--out", str(tmp_path / "fresh"))
    assert done.returncode == 1
    assert f"MISSING input  {main}/base_dev/episodes.jsonl" in done.stdout


def test_wrong_options_give_exit_code_2(tmp_path):
    assert rebuild("--only", "nothing_such").returncode == 2
    assert rebuild("--results", str(tmp_path / "absent")).returncode == 2
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    assert rebuild("--out", str(tmp_path / "full"), "--only", "tables").returncode == 2


def made_up_test_state(main: Path, with_test: bool) -> None:
    for arm in ARMS:
        for split in ("dev",) + (("test",) if with_test else ()):
            run = main / f"{arm}_{split}"
            run.mkdir(parents=True)
            (run / "episodes.jsonl").write_text("")
            (run / "run_manifest.json").write_text(json.dumps({"split": split, "ruleset": "default"}))
    (main / "tables").mkdir()
    (main / "monitors").mkdir()
    for arm in ARMS:
        (main / "monitors" / f"{arm}_dev.md").write_text("made up\n")
        if with_test:
            (main / "monitors" / f"{arm}_test.md").write_text("made up\n")
    if with_test:
        for ext in ("json", "md"):
            (main / "tables" / f"test_base_sft_grpo_seed1_seed2.{ext}").write_text("made up\n")


def test_list_with_the_test_runs_absent(tmp_path):
    main = tmp_path / "results" / "main"
    made_up_test_state(main, with_test=False)
    done = rebuild("--results", str(main), "--list")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "the test runs do not exist" in done.stdout
    assert "table_test" not in done.stdout and "failures_test" not in done.stdout
    assert not re.search(r"(monitors|replay)_\w+_test", done.stdout)
    assert "test_base_sft_grpo_seed1_seed2" not in done.stdout
    assert "list only: nothing was run" in done.stdout


def test_list_with_made_up_test_runs_present(tmp_path):
    main = tmp_path / "results" / "main"
    made_up_test_state(main, with_test=True)
    done = rebuild("--results", str(main), "--list")
    out = done.stdout
    assert done.returncode == 0, out + done.stderr
    assert "the test runs exist" in out
    assert re.search(r"would run  table_test: .*--arm grpo_seed2=\S*grpo_seed2_test/episodes.jsonl "
                     r"--baseline base --pair base:sft --pair sft:grpo_seed1 --pair sft:grpo_seed2", out)
    for ext in ("json", "md"):
        assert f"would compare  {main}/tables/test_base_sft_grpo_seed1_seed2.{ext}" in out
    for arm in ARMS:
        assert f"would run  monitors_{arm}_test:" in out and "--split test" in out
        assert f"would compare  {main}/monitors/{arm}_test.md" in out
        assert re.search(rf"would run  replay_{arm}_test: .*--split test .*--ruleset default", out)
    assert "would run  failures_test:" in out


# ---- the command on the saved runs ----

needs_real = pytest.mark.skipif(
    not all((REAL_MAIN / f"{arm}_dev" / "episodes.jsonl").is_file() for arm in ARMS)
    or not (REAL_MAIN / "tables").is_dir() or not (REAL_MAIN / "monitors").is_dir(),
    reason="the saved dev runs of results/main are not here")


@needs_real
def test_list_names_every_saved_table_and_monitor_report():
    done = rebuild("--list")
    assert done.returncode == 0, done.stdout + done.stderr
    tables = sorted(p.name for p in (REAL_MAIN / "tables").iterdir() if p.is_file())
    monitors = sorted(p.name for p in (REAL_MAIN / "monitors").iterdir() if p.is_file())
    assert tables and monitors
    for name in tables:
        assert f"would compare  results/main/tables/{name}" in done.stdout
    for name in monitors:
        assert f"would compare  results/main/monitors/{name}" in done.stdout
    for name in ("figure", "training", "failures", "stretch"):
        assert f"report piece: {name}" in done.stdout


@needs_real
def test_the_cheaper_steps_on_the_saved_runs_are_equal(tmp_path):
    steps = "tables,monitors,second_desk,formula,positive_control,sft_profile,figure,failures"
    done = rebuild("--only", steps, "--out", str(tmp_path / "fresh"), timeout=900)
    lines = done.stdout.splitlines()
    assert not [x for x in lines if x.startswith(("DIFFERENT", "FAILED", "MISSING", "SAVED, NOT MADE"))], done.stdout
    assert "0 DIFFERENT" in done.stdout and "0 failed" in done.stdout
    assert done.returncode == 0, done.stdout + done.stderr
    for name in sorted(p.name for p in (REAL_MAIN / "tables").iterdir() if p.is_file()):
        assert f"equal  results/main/tables/{name}" in done.stdout
