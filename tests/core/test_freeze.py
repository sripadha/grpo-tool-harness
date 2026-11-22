"""accessdesk/freeze.py, scripts/freeze.py and scripts/approve_text.py.

Every test works on a small tree that it makes in a temporary folder: fake files in the places of the real
ones. The scripts get that folder with --root or with the environment variable, so nothing is written
into the repository. The repository itself is only read: the scripts, the placeholder text as a source of
valid values, and four tests that say so.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from accessdesk import freeze
from accessdesk.config import RunConfig
from accessdesk.env import rules
from accessdesk.env.scenarios import load_families, save_families
from accessdesk.text import load_text, placeholder_path

ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-11-18T12:00:00Z"
LATER = "2026-11-19T08:30:00Z"
APPROVED = "# Read and approved by: Test Person, 2026-11-18 (UTC)"
NOT_YET = "# Read and approved by: (not yet)"

# The fake tree. Group -> the files that must be in it (the scenario files are added by the builder).
FROZEN = {
    "scenarios": ["data/scenarios/calib.jsonl", "data/scenarios/dev.jsonl", "data/scenarios/sets.json",
                  "data/scenarios/summary.json", "data/scenarios/test.jsonl", "data/scenarios/train.jsonl"],
    "names": ["data/names/PROVENANCE.md", "data/names/first_names.txt", "data/names/last_names.txt",
              "data/names/make_name_lists.py"],
    "text": ["text/model_text.yaml"],
    "text_provenance": ["accessdesk/drafting.py", "scripts/gpu/draft_text.py",
                        "text/drafting_instructions.yaml", "text/drafting_log.jsonl"],
    "grader": ["accessdesk/_second/__init__.py", "accessdesk/_second/reward.py",
               "accessdesk/_second/verifier.py", "accessdesk/env/reward.py", "accessdesk/env/rules.py",
               "accessdesk/env/verifier.py", "accessdesk/impl.py"],
    "environment": ["accessdesk/__init__.py", "accessdesk/env/__init__.py", "accessdesk/env/generator.py",
                    "accessdesk/env/world.py", "accessdesk/rng.py", "accessdesk/text.py",
                    "accessdesk/types.py"],
    "harness": ["accessdesk/harness/__init__.py", "accessdesk/harness/episode.py",
                "accessdesk/harness/parser.py"],
    "evaluation": ["accessdesk/_second/bootstrap.py", "accessdesk/eval/__init__.py",
                   "accessdesk/eval/bootstrap.py", "accessdesk/eval/metrics.py", "accessdesk/eval/tables.py",
                   "scripts/build_scenarios.py", "scripts/check_wordings.py", "scripts/gpu/rollout.py",
                   "scripts/hand_states.py", "scripts/monitor_report.py", "scripts/pick_overfit_families.py",
                   "scripts/replay_check.py", "scripts/run_scripted.py", "scripts/sft_profile.py"],
    "config": ["accessdesk/config.py"],
    "plan": ["ANALYSIS_PLAN.md"],
    "lock": ["pyproject.toml", "requirements-gpu.txt"],
    "audit": ["accessdesk/audit/__init__.py", "accessdesk/audit/sheet.py", "scripts/audit_positive_control.py",
              "scripts/audit_sheet.py", "scripts/formula_audit.py", "scripts/second_desk_report.py"],
    "report": ["scripts/failure_report.py", "scripts/make_figure.py", "scripts/rebuild_results.sh",
               "scripts/requirements-report.txt", "scripts/stretch_numbers.py", "scripts/training_report.py"],
}
# Files of the fake tree that must be in no group.
NOT_FROZEN = ["README.md", "accessdesk/_second/grpo_math.py", "accessdesk/eval/calibration.py",
              "accessdesk/freeze.py", "accessdesk/train/grpo.py", "results/base_dev/episodes.jsonl",
              "scripts/approve_text.py", "scripts/calibration_report.py", "scripts/freeze.py",
              "scripts/gpu/train_grpo.py", "tests/core/test_verify.py", "text/placeholder_text.yaml"]
SEEDS = {"train": [1, 2, 3, 4], "dev": [11, 12], "test": [21, 22], "calib": [31, 32]}
# The calibration folder beside the final one. Its calib split holds the families of the final calib split
# (the calibration seed stays the same), and that is allowed.
CALIBRATION_SEEDS = {"train": [101, 102], "dev": [111], "test": [121], "calib": [31, 32]}
ID_PREFIX = {"train": "tr", "dev": "dv", "test": "te", "calib": "ca"}


# ---------------------------------------------------------------------------
# The fake tree
# ---------------------------------------------------------------------------
def put(root: Path, path: str, content: bytes | None = None) -> None:
    """Write one file of the tree. Without content: a line that names the file, so no two files are equal."""
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(f"the file {path}\n".encode() if content is None else content)


def sha(root: Path, path: str) -> str:
    return hashlib.sha256((root / path).read_bytes()).hexdigest()


def text_bytes(approval_line: str = APPROVED, placeholder: bool = False) -> bytes:
    """A valid text file for the tests: the values of the placeholder file of the repository, with the
    placeholder flag as asked, under a header that ends with the approval line."""
    data = yaml.safe_load(placeholder_path().read_text(encoding="utf-8"))
    data["meta"]["placeholder"] = placeholder
    header = f"# A text file made for the tests.\n#\n{approval_line}\n\n"
    return (header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)).encode("utf-8")


def family(split: str, index: int, seed: int) -> dict:
    """A family record with the fields that the freeze reads. The world and the request follow the seed."""
    return {"family_id": f"{ID_PREFIX[split]}-{index:02d}", "split": split, "world_seed": seed,
            "world": {"people": [{"person_id": f"p{seed}"}]}, "request": {"person_name": f"Person {seed}"}}


def put_scenarios(root: Path, seeds: dict[str, list[int]], folder: str = "data/scenarios") -> None:
    """Write a scenario folder as the build does: the four split files, sets.json and a summary.json that
    names the text file, the name files and the config of the tree and the hashes of the written files."""
    for split, split_seeds in seeds.items():
        lines = [json.dumps(family(split, index, seed)) + "\n" for index, seed in enumerate(split_seeds)]
        put(root, f"{folder}/{split}.jsonl", "".join(lines).encode())
    put(root, f"{folder}/sets.json", b'{"curve_subset": ["dv-00"]}\n')
    put_summary(root, folder)


def put_summary(root: Path, folder: str = "data/scenarios", **changes: object) -> None:
    """Write summary.json again for the files that are in the tree now; `changes` replace its fields."""
    config = RunConfig()
    listed = [path.name for path in sorted((root / folder).iterdir()) if path.name != "summary.json"]
    summary = {
        "text_file": "model_text.yaml", "text_sha256": sha(root, "text/model_text.yaml"),
        "text_is_placeholder": False,
        "name_files_sha256": {name: sha(root, f"data/names/{name}")
                              for name in ("first_names.txt", "last_names.txt")},
        "config": config.to_dict(), "config_sha256": config.sha256(),
        "splits": {split: {"families": len((root / folder / f"{split}.jsonl").read_bytes().splitlines())}
                   for split in SEEDS},
        "files_sha256": {name: sha(root, f"{folder}/{name}") for name in listed},
    }
    put(root, f"{folder}/summary.json", (json.dumps({**summary, **changes}, indent=2) + "\n").encode())


@pytest.fixture
def tree(tmp_path, monkeypatch) -> Path:
    """A tree that may be frozen: every required file, an approved text, final scenarios that fit it, and
    a calibration folder beside them.

    Its families and its name files are small fakes: no generator gives them. So the one check that makes
    the data again (freeze._built_again_problems) is replaced here by one that finds nothing. That check
    has its own tests, on a folder that the real build script wrote (built_tree)."""
    monkeypatch.setattr(freeze, "_built_again_problems", lambda root: [])
    return make_tree(tmp_path / "repo")


def make_tree(root: Path) -> Path:
    for path in [path for paths in FROZEN.values() for path in paths] + NOT_FROZEN:
        if not path.startswith("data/scenarios/"):
            put(root, path)
    put(root, "text/model_text.yaml", text_bytes())
    put_scenarios(root, SEEDS)
    put_scenarios(root, CALIBRATION_SEEDS, "data/scenarios_calib1")
    return root


def all_files(root: Path) -> dict[str, str]:
    """path -> hash, for every file of the tree."""
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def frozen(root: Path) -> dict:
    return freeze.build_manifest(root, signed_by="Test Person", frozen_at=NOW)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def freeze_script():
    return load_script("freeze")


@pytest.fixture(scope="module")
def approve_script():
    return load_script("approve_text")


# ---------------------------------------------------------------------------
# Which files are frozen
# ---------------------------------------------------------------------------
def test_every_file_of_a_frozen_kind_is_in_one_group_and_the_other_files_are_in_none(tree):
    files = freeze.frozen_files(tree)
    assert files == FROZEN
    listed = [path for paths in files.values() for path in paths]
    assert len(listed) == len(set(listed))                       # no file is in two groups
    assert not set(listed) & set(NOT_FROZEN)
    assert all((tree / path).is_file() for path in NOT_FROZEN)   # they are there, and still not frozen
    assert not any(path.startswith("data/scenarios_calib1/") for path in listed)


def test_a_file_that_appears_under_a_wildcard_joins_the_group(tree):
    put(tree, "accessdesk/eval/monitors.py")
    put(tree, "accessdesk/harness/tokens.py")
    put(tree, "accessdesk/env/tools.py")
    put(tree, "data/scenarios/deep/extra.json")
    put(tree, "data/names/provenance.json")
    put(tree, "uv.lock")
    files = freeze.frozen_files(tree)
    assert "accessdesk/eval/monitors.py" in files["evaluation"]
    assert "accessdesk/harness/tokens.py" in files["harness"]
    assert "accessdesk/env/tools.py" in files["environment"]
    assert "data/scenarios/deep/extra.json" in files["scenarios"]
    assert "data/names/provenance.json" in files["names"]
    assert "uv.lock" in files["lock"]


def test_compiled_python_files_are_never_frozen(tree):
    before = freeze.frozen_files(tree)
    put(tree, "accessdesk/harness/__pycache__/parser.cpython-311.pyc")
    put(tree, "data/names/__pycache__/make_name_lists.cpython-311.pyc")
    put(tree, "accessdesk/eval/old.pyc")
    assert freeze.frozen_files(tree) == before


# The scripts that compute a number of the write-up or apply a rule of the analysis plan. Typed by hand.
NUMBER_AND_RULE_SCRIPTS = {
    "scripts/gpu/rollout.py",               # every episode of a model
    "scripts/run_scripted.py",              # the two reference rows
    "scripts/replay_check.py",              # the audit: the saved calls, graded again
    "scripts/monitor_report.py",            # the flagged shares
    "scripts/hand_states.py",               # the check of the six hand-written end states
    "scripts/check_wordings.py",            # the shared 5-grams of train and the held-out wordings
    "scripts/build_scenarios.py",           # the scenarios and their expected end states
    "scripts/sft_profile.py",               # the rule on the size of a GRPO step
    "scripts/pick_overfit_families.py",     # the rule that chooses the families of the overfit check
}


def test_the_table_names_each_file_once_and_keeps_training_and_calibration_code_out():
    named = freeze.exact_paths()
    assert set(freeze.required_files()) == set(named)            # every named file must exist at the freeze
    assert not hasattr(freeze, "MAY_BE_ABSENT")                  # no named file may be absent any more
    assert {path for path, group in named.items() if group == "evaluation" and path.startswith("scripts/")} \
        == NUMBER_AND_RULE_SCRIPTS
    assert "evaluation" in freeze.CODE_GROUPS                    # so the code hash of a run covers them
    assert set(freeze.CODE_GROUPS) == {"grader", "environment", "harness", "evaluation", "config"}
    patterns = [pattern for group in freeze.FROZEN_GROUPS.values() for pattern in group]
    assert not any(pattern.startswith("accessdesk/train") for pattern in patterns)
    assert not any("calibration" in pattern for pattern in patterns)
    assert "accessdesk/eval/calibration.py" in freeze.NOT_FROZEN


# The scripts of the audit and of the second-desk report: written after the freeze. Typed by hand.
AUDIT_SCRIPTS = {
    "scripts/second_desk_report.py",        # the numbers of the second desk
    "scripts/audit_sheet.py",               # the blind reading sheet, and the check of the filled form
    "scripts/formula_audit.py",             # the formula audit of the saved GRPO rollouts
    "scripts/audit_positive_control.py",    # the positive control of the audit
}


def test_the_audit_code_is_a_frozen_group_of_its_own_and_is_not_in_the_code_hash(tree):
    named = freeze.exact_paths()
    assert {path for path, group in named.items() if group == "audit"} == AUDIT_SCRIPTS
    assert "audit" in freeze.FROZEN_GROUPS and "audit" not in freeze.CODE_GROUPS
    manifest, code_before = frozen(tree), freeze.code_sha256(tree)
    assert freeze.check_manifest(tree, manifest) == []
    # A changed audit file and a new audit module are found by the check, and the code hash stays.
    put(tree, "accessdesk/audit/sheet.py", b"changed after the amendment\n")
    put(tree, "accessdesk/audit/later/new_module.py")
    differences = freeze.check_manifest(tree, manifest)
    assert any(line.startswith("changed: accessdesk/audit/sheet.py (group audit)") for line in differences)
    assert any(line.startswith("new: accessdesk/audit/later/new_module.py (group audit)")
               for line in differences)
    assert len(differences) == 2
    assert freeze.code_sha256(tree) == code_before == manifest["code_sha256"]
    # A missing audit script is a problem before a freeze or an amendment, like every named file.
    (tree / "scripts/formula_audit.py").unlink()
    assert any("scripts/formula_audit.py" in problem for problem in freeze.tree_problems(tree))


# The report code: written after the freeze, on 2026-11-19. Typed by hand.
REPORT_FILES = {
    "scripts/make_figure.py",               # the report piece figure
    "scripts/training_report.py",           # the report piece training (and the cost table)
    "scripts/failure_report.py",            # the report piece failures
    "scripts/stretch_numbers.py",           # the report piece stretch (results/stretch.md)
    "scripts/rebuild_results.sh",           # makes every reported file again and compares it
    "scripts/requirements-report.txt",      # the package list of the figure
}


def test_the_report_code_is_a_frozen_group_of_its_own_and_is_not_in_the_code_hash(tree):
    named = freeze.exact_paths()
    assert {path for path, group in named.items() if group == "report"} == REPORT_FILES
    assert "report" in freeze.FROZEN_GROUPS and "report" not in freeze.CODE_GROUPS
    assert all(freeze.is_exact(pattern) for pattern in freeze.FROZEN_GROUPS["report"])
    manifest, code_before = frozen(tree), freeze.code_sha256(tree)
    assert freeze.check_manifest(tree, manifest) == []
    # A changed report file is found by the check, and the code hash stays.
    put(tree, "scripts/stretch_numbers.py", b"changed after the amendment\n")
    differences = freeze.check_manifest(tree, manifest)
    assert len(differences) == 1
    assert differences[0].startswith("changed: scripts/stretch_numbers.py (group report)")
    assert freeze.code_sha256(tree) == code_before == manifest["code_sha256"]
    # A missing report file is a problem before a freeze or an amendment, like every named file.
    (tree / "scripts/rebuild_results.sh").unlink()
    assert any("scripts/rebuild_results.sh" in problem for problem in freeze.tree_problems(tree))


def test_in_the_real_repository_the_named_code_files_exist_and_no_file_is_in_two_groups():
    """Reads the repository. A wrong path in FROZEN_GROUPS would freeze nothing, without a word."""
    code = [path for path in freeze.required_files() if path.endswith(".py")] + ["pyproject.toml"]
    assert len(code) > 20
    assert [path for path in code if not (ROOT / path).is_file()] == []
    files = freeze.frozen_files(ROOT)                            # raises if a file is in two groups
    assert "accessdesk/eval/calibration.py" not in files["evaluation"]
    assert not any(path.startswith("accessdesk/train/") for paths in files.values() for path in paths)
    assert {"accessdesk/env/verifier.py", "accessdesk/env/reward.py"} <= set(files["grader"])
    assert not {"accessdesk/env/verifier.py", "accessdesk/env/reward.py"} & set(files["environment"])
    assert NUMBER_AND_RULE_SCRIPTS <= set(files["evaluation"])
    assert AUDIT_SCRIPTS <= set(files["audit"])
    assert set(files["report"]) == REPORT_FILES
    assert {path for path in files["audit"] if path.startswith("accessdesk/")} == {
        f"accessdesk/audit/{name}.py" for name in ("__init__", "broken", "formula", "sample", "second_desk",
                                                    "sheet")}


# Files under scripts/ and accessdesk/ that are in no frozen group and not in the training code that every
# training run records, each with the reason. Typed by hand.
IN_NO_GROUP = {
    "accessdesk/freeze.py": "the freeze itself: it makes no number; every hash can be made again with sha256sum",
    "scripts/freeze.py": "the freeze itself: it writes, checks and amends MANIFEST.json",
    "scripts/approve_text.py": "it writes one header line of the text file; the text file is frozen",
    "accessdesk/eval/calibration.py": "calibration numbers are not results",
    "scripts/calibration_report.py": "calibration numbers are not results",
    "scripts/play.py": "a person plays one scenario in the terminal; nothing of it is reported",
    "scripts/gpu/bench.py": "speed and memory of the machine before the runs; not a result",
    "scripts/gpu/smoke.py": "trial runs under results/_smoke/; not a result",
    "scripts/gpu/dryrun.py": "trial runs under results/_smoke/; not a result",
    "scripts/remote/rent.py": "rents and stops the machine",
    "scripts/remote/setup_box.sh": "sets the machine up",
    "scripts/remote/watchdog.sh": "stops a forgotten machine",
    "scripts/make_package.sh": "builds the zip of the package from the committed files and scans it; makes no number",
    "scripts/render_writeup.py": "renders WRITEUP.md to a PDF and prints its page count and figure width; makes no number",
    # The audit code (accessdesk/audit/ and its four scripts) was in this list from the day it was written
    # until the amendment: it is in the frozen group "audit" now.
    "scripts/gpu/measure_layout_gradients.py": "written after the freeze on 2026-11-18; a measurement of the "
                                               "training side; recorded at its run",
    "scripts/gpu/chain_test.sh": "the chain of the test run, written after the freeze on 2026-11-19 (UTC); it "
                                 "starts frozen scripts and reports no number of its own (its checks read "
                                 "saved files, count records and stop the chain); the record of the run "
                                 "(results/main/test_run_record.txt) names its SHA-256",
    # The report code (scripts/make_figure.py, training_report.py, failure_report.py, stretch_numbers.py,
    # rebuild_results.sh, requirements-report.txt) was in this list from the day it was written until
    # amendment 2: it is in the frozen group "report" now.
}


def training_code_files() -> set[str]:
    """The files whose hashes every training run writes into its run manifest: `training_code` of
    scripts/gpu/sft_data.py, which train_sft.py and train_grpo.py call too."""
    folder = str(ROOT / "scripts" / "gpu")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return set(importlib.import_module("sft_data").training_code()["files"])


def test_every_file_of_scripts_and_accessdesk_is_frozen_or_recorded_at_training_or_listed_with_a_reason():
    """Reads the repository. A new script that computes a number must not stay outside without a word: a
    file under scripts/ or accessdesk/ that is in none of the three places makes this test fail."""
    tree = {path.relative_to(ROOT).as_posix() for folder in ("scripts", "accessdesk")
            for path in (ROOT / folder).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    frozen_now = {path for paths in freeze.frozen_files(ROOT).values() for path in paths}
    training = training_code_files()
    assert len(tree) > 60
    # No file is in two of the three places, and every listed file exists.
    assert not frozen_now & training and not frozen_now & set(IN_NO_GROUP) and not training & set(IN_NO_GROUP)
    assert training <= tree and set(IN_NO_GROUP) <= tree
    assert sorted(tree - frozen_now - training - set(IN_NO_GROUP)) == []
    # The training code: every file of accessdesk/train/, the two training parts of accessdesk/_second/,
    # the three training scripts and the two chain scripts.
    assert {path for path in tree if path.startswith("accessdesk/train/")} <= training
    assert training == {path for path in tree if path.startswith("accessdesk/train/")} | {
        "accessdesk/_second/grpo_math.py", "accessdesk/_second/loss_mask.py", "scripts/gpu/sft_data.py",
        "scripts/gpu/train_sft.py", "scripts/gpu/train_grpo.py", "scripts/gpu/chain_stage1.sh",
        "scripts/gpu/chain_stage2.sh"}
    # That the three training scripts write this list into their run manifests is tested with their runs,
    # in tests/core/test_train_scripts.py.


def test_the_data_fields_of_the_manifest_are_fields_of_the_config():
    assert set(freeze.SEED_FIELDS + freeze.DESIGN_FIELDS) <= set(RunConfig().to_dict())


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------
def test_the_same_tree_gives_the_same_manifest_except_for_the_time(tree, tmp_path):
    first = frozen(tree)
    again = freeze.build_manifest(tree, signed_by="Test Person", frozen_at=LATER)
    assert {**again, "frozen_at": NOW} == first and again["frozen_at"] == LATER
    assert json.dumps(first) == json.dumps({**again, "frozen_at": NOW})          # also the order of the keys
    # A second tree with the same files in another folder, written in another order: the same manifest.
    other = tmp_path / "elsewhere" / "copy"
    for path in sorted(all_files(tree), reverse=True):
        put(other, path, (tree / path).read_bytes())
    assert frozen(other) == first


def test_the_manifest_holds_the_signer_the_note_and_an_empty_history(tree):
    manifest = freeze.build_manifest(tree, signed_by="  Test Person ", frozen_at=NOW, note="first freeze")
    assert (manifest["schema"], manifest["frozen_at"], manifest["signed_by"], manifest["note"]) == \
        (1, NOW, "Test Person", "first freeze")
    assert manifest["amendments"] == []
    assert list(manifest["groups"]) == list(freeze.FROZEN_GROUPS)
    with pytest.raises(ValueError, match="name of the person"):
        freeze.build_manifest(tree, signed_by="  ", frozen_at=NOW)


def test_each_file_is_listed_with_the_hash_and_the_size_of_its_bytes(tree):
    manifest = frozen(tree)
    for group, paths in FROZEN.items():
        entry = manifest["groups"][group]
        assert [file["path"] for file in entry["files"]] == sorted(paths)
        for file in entry["files"]:
            raw = (tree / file["path"]).read_bytes()
            assert (file["sha256"], file["bytes"]) == (hashlib.sha256(raw).hexdigest(), len(raw))


def test_a_group_hash_is_what_sha256sum_gives_for_the_files_of_the_group(tree):
    if shutil.which("sha256sum") is None:
        pytest.skip("the program sha256sum is not installed")
    manifest = frozen(tree)
    for group in ("grader", "scenarios", "plan"):
        paths = [file["path"] for file in manifest["groups"][group]["files"]]
        listing = subprocess.run(["sha256sum", *paths], cwd=tree, capture_output=True, check=True).stdout
        assert manifest["groups"][group]["sha256"] == hashlib.sha256(listing).hexdigest()
    lines = "".join(f"{manifest['groups'][group]['sha256']}  {group}\n" for group in freeze.CODE_GROUPS)
    assert manifest["code_sha256"] == hashlib.sha256(lines.encode()).hexdigest()


@pytest.mark.parametrize("path, in_code", [
    ("accessdesk/env/reward.py", True), ("accessdesk/_second/verifier.py", True),
    ("accessdesk/env/world.py", True), ("accessdesk/harness/parser.py", True),
    ("accessdesk/eval/metrics.py", True), ("scripts/gpu/rollout.py", True), ("accessdesk/config.py", True),
    ("scripts/sft_profile.py", True), ("scripts/pick_overfit_families.py", True),
    ("scripts/monitor_report.py", True), ("scripts/hand_states.py", True),
    ("ANALYSIS_PLAN.md", False), ("pyproject.toml", False), ("data/names/first_names.txt", False),
    ("text/drafting_log.jsonl", False), ("accessdesk/drafting.py", False),
    ("accessdesk/train/grpo.py", False), ("accessdesk/eval/calibration.py", False),
    ("accessdesk/audit/sheet.py", False), ("scripts/audit_sheet.py", False),
    ("scripts/second_desk_report.py", False), ("scripts/stretch_numbers.py", False),
    ("scripts/rebuild_results.sh", False)])
def test_the_code_hash_changes_with_a_file_of_the_five_code_groups_and_with_no_other(tree, path, in_code):
    before = freeze.code_sha256(tree)
    assert before == frozen(tree)["code_sha256"]
    put(tree, path, (tree / path).read_bytes() + b"# one more line\n")
    assert (freeze.code_sha256(tree) != before) == in_code


def test_the_text_hash_is_the_hash_that_the_text_module_gives(tree):
    manifest = frozen(tree)
    assert manifest["text_sha256"] == load_text(tree / "text" / "model_text.yaml").sha256
    assert manifest["text_sha256"] == manifest["groups"]["text"]["files"][0]["sha256"]


def test_the_config_hash_and_values_are_those_of_the_config_module(tree):
    manifest = frozen(tree)
    assert manifest["run_config_sha256"] == RunConfig().sha256()
    assert manifest["run_config"] == RunConfig().to_dict()


def test_the_seeds_and_the_design_switches_are_those_that_the_summary_names(tree):
    """The summary of this tree names other seeds than the config module: the manifest shows the summary's."""
    config = {**RunConfig().to_dict(), "data_seed": 20261118, "calib_seed": 77, "mix_facts": True,
              "homonym_share": 0.5}
    put_summary(tree, config=config)
    data = frozen(tree)["data"]
    assert data["seeds"] == {"data_seed": 20261118, "calib_seed": 77}
    assert data["design"] == {"mix_facts": True, "homonym_share": 0.5}
    assert data["families"] == {"train": 4, "dev": 2, "test": 2, "calib": 2}
    assert data["text_sha256"] == sha(tree, "text/model_text.yaml")
    assert data["config_sha256"] == RunConfig().sha256()
    (tree / "data" / "scenarios" / "summary.json").unlink()
    assert frozen(tree)["data"]["seeds"] == {"data_seed": None, "calib_seed": None}


def test_a_folder_that_is_not_a_git_work_tree_has_no_git_state(tree):
    nothing = {"commit": None, "dirty": None, "uncommitted_frozen_files": None}
    assert frozen(tree)["git"] == nothing
    # Reads the repository: one of its folders is not taken for the repository itself.
    assert freeze.git_state(ROOT / "accessdesk", ["config.py"]) == nothing


def test_the_git_state_of_the_real_repository_names_a_commit():
    """Reads the repository with read-only git commands."""
    state = freeze.git_state(ROOT, ["pyproject.toml"])
    if state["commit"] is None:
        pytest.skip("git is not installed, or the repository is not a git work tree")
    assert re.fullmatch(r"[0-9a-f]{40}", state["commit"])
    assert isinstance(state["dirty"], bool) and isinstance(state["uncommitted_frozen_files"], list)


def test_building_a_manifest_changes_no_file(tree):
    before = all_files(tree)
    frozen(tree)
    freeze.problems_before_freeze(tree)
    freeze.check_manifest(tree, frozen(tree))
    freeze.frozen_state(tree)
    assert all_files(tree) == before


# ---------------------------------------------------------------------------
# problems_before_freeze: every refusal
# ---------------------------------------------------------------------------
def test_a_complete_tree_has_no_problem(tree):
    assert freeze.problems_before_freeze(tree, signed_by="Test Person") == []
    assert freeze.tree_problems(tree) == []


def remove(path: str):
    return lambda root: (root / path).unlink()


def only(problems: list[str], phrase: str) -> bool:
    """True if the list is exactly one problem, and it holds the phrase."""
    return len(problems) == 1 and phrase in problems[0]


@pytest.mark.parametrize("spoil, phrase", [
    (remove("ANALYSIS_PLAN.md"), "ANALYSIS_PLAN.md is missing (group plan)"),
    (remove("accessdesk/env/verifier.py"), "accessdesk/env/verifier.py is missing (group grader)"),
    (remove("accessdesk/_second/reward.py"), "accessdesk/_second/reward.py is missing (group grader)"),
    (remove("scripts/replay_check.py"), "scripts/replay_check.py is missing (group evaluation)"),
    (remove("scripts/monitor_report.py"), "scripts/monitor_report.py is missing (group evaluation)"),
    (remove("scripts/hand_states.py"), "scripts/hand_states.py is missing (group evaluation)"),
    (remove("scripts/sft_profile.py"), "scripts/sft_profile.py is missing (group evaluation)"),
    (remove("scripts/pick_overfit_families.py"),
     "scripts/pick_overfit_families.py is missing (group evaluation)"),
    (remove("text/drafting_log.jsonl"), "text/drafting_log.jsonl is missing (group text_provenance)"),
    (remove("accessdesk/config.py"), "accessdesk/config.py is missing (group config)"),
    (remove("pyproject.toml"), "pyproject.toml is missing (group lock)"),
    (lambda root: put(root, "ANALYSIS_PLAN.md", b" \n\n"), "ANALYSIS_PLAN.md is empty"),
    (lambda root: shutil.rmtree(root / "accessdesk" / "harness"), "the group harness has no file"),
], ids=lambda value: value if isinstance(value, str) else "")
def test_a_missing_or_empty_required_file_or_group_is_a_problem(tree, spoil, phrase):
    spoil(tree)
    assert only(freeze.problems_before_freeze(tree), phrase)


def test_an_empty_init_file_is_no_problem(tree):
    put(tree, "accessdesk/_second/__init__.py", b"")
    assert freeze.problems_before_freeze(tree) == []


def test_a_manifest_that_exists_already_is_a_problem(tree):
    put(tree, "MANIFEST.json", b"{}\n")
    assert only(freeze.problems_before_freeze(tree), "MANIFEST.json exists already")


def test_an_empty_name_of_the_signer_is_a_problem(tree):
    for name in ("", "   "):
        assert only(freeze.problems_before_freeze(tree, signed_by=name), "the name of the person who signs")
    assert freeze.problems_before_freeze(tree, signed_by="A") == []


def retext(root: Path, **options: object) -> None:
    """Put another text file into the tree and build the summary again, so that only the text is wrong."""
    put(root, "text/model_text.yaml", text_bytes(**options))
    put_summary(root)


@pytest.mark.parametrize("options, phrase", [
    ({"placeholder": True}, "text/model_text.yaml is a placeholder"),
    ({"approval_line": NOT_YET}, "still says '# Read and approved by: (not yet)'"),
    ({"approval_line": "# Read and approved by:   (not yet)  "}, "still says"),
    ({"approval_line": "# Read by nobody"}, "exactly once, not 0 times"),
    ({"approval_line": APPROVED + "\n" + NOT_YET}, "exactly once, not 2 times"),
    ({"approval_line": "# Read and approved by:"}, "is empty"),
    ({"approval_line": "# Read and approved by: , 2026-11-18 (UTC)"}, "is empty"),
], ids=["placeholder", "not_yet", "not_yet_with_spaces", "no_line", "two_lines", "no_name", "date_only"])
def test_a_placeholder_text_or_a_text_that_nobody_approved_is_a_problem(tree, options, phrase):
    retext(tree, **options)
    assert only(freeze.problems_before_freeze(tree), phrase)


@pytest.mark.parametrize("value", ["TODO", "-", "?", "not yet", "(not yet).", "Test Person",
                                   "Test Person, 2026-11-18", "Test Person, 2026-13-45 (UTC)",
                                   "(not yet), 2026-11-18 (UTC)"])
def test_a_word_in_the_place_of_the_approval_is_not_an_approval(tree, value):
    """Only the line that scripts/approve_text.py writes counts: a name, a date that exists, (UTC)."""
    retext(tree, approval_line=f"# Read and approved by: {value}")
    assert only(freeze.problems_before_freeze(tree), "is not an approval")


def test_a_name_with_a_comma_is_an_approval(tree):
    retext(tree, approval_line="# Read and approved by: Person, Test, 2026-11-18 (UTC)")
    assert freeze.problems_before_freeze(tree) == []


def test_an_approval_line_below_the_header_does_not_count(tree):
    put(tree, "text/model_text.yaml", text_bytes(NOT_YET) + f"\n{APPROVED}\n".encode())
    put_summary(tree)
    assert only(freeze.problems_before_freeze(tree), "still says")


def test_a_text_file_that_is_not_valid_is_a_problem(tree):
    put(tree, "text/model_text.yaml", f"{APPROVED}\nmeta:\n  placeholder: false\n".encode())
    put_summary(tree)
    assert only(freeze.problems_before_freeze(tree), "text/model_text.yaml is not a valid text file")


def test_scenarios_that_were_built_from_another_text_are_a_problem(tree):
    put(tree, "text/model_text.yaml", text_bytes() + b"# one more comment\n")     # the summary is the old one
    assert only(freeze.problems_before_freeze(tree), "data/scenarios was built from another text file")


def append_to(path: str):
    def spoil(root: Path) -> None:
        with (root / path).open("ab") as handle:
            handle.write(b"\n")
    return spoil


@pytest.mark.parametrize("spoil, phrase", [
    (append_to("data/scenarios/dev.jsonl"), "data/scenarios/dev.jsonl is not the file that the build wrote"),
    (append_to("data/scenarios/sets.json"), "data/scenarios/sets.json is not the file that the build wrote"),
    (remove("data/scenarios/test.jsonl"), "data/scenarios/test.jsonl is missing"),
    (lambda root: put(root, "data/scenarios/notes.txt"),
     "data/scenarios/notes.txt is not a file of the build"),
    (lambda root: put_summary(root, text_is_placeholder=True), "was not a placeholder"),
    (lambda root: put_summary(root, files_sha256={}), "lists no file hash"),
    (lambda root: put(root, "data/scenarios/summary.json", b"not json"), "cannot be read as a JSON object"),
    (append_to("data/names/first_names.txt"), "was built from another data/names/first_names.txt"),
    (lambda root: put_summary(root, name_files_sha256={"first_names.txt": None, "last_names.txt": None}),
     "was built without the name file"),
    (lambda root: put_summary(root, config_sha256="0" * 64), "was built with another config"),
    (lambda root: put_summary(root, config={**RunConfig().to_dict(), "data_seed": -5}),
     "was built with another config"),
], ids=["split_file_changed", "sets_changed", "split_file_missing", "file_not_of_the_build",
        "summary_says_placeholder", "no_file_hashes", "summary_unreadable", "name_file_changed",
        "made_up_names", "other_config_hash", "other_config_values"])
def test_scenarios_that_do_not_fit_their_summary_or_the_tree_are_a_problem(tree, spoil, phrase):
    spoil(tree)
    problems = freeze.problems_before_freeze(tree)
    assert problems and all(phrase in problem for problem in problems)
    assert len(problems) == (2 if "made-up" in problems[0] else 1)        # one line per name file


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_a_world_seed_of_a_calibration_folder_in_the_final_data_is_a_problem(tree, split):
    """The same seed gives the same world here, so both lines appear: the seed and the content."""
    put_scenarios(tree, {**SEEDS, split: SEEDS[split][:-1] + [CALIBRATION_SEEDS["train"][0]]})
    problems = freeze.problems_before_freeze(tree)
    assert len(problems) == 2
    assert "1 of the 8 world seeds of the train, dev and test families of data/scenarios are in " \
           "data/scenarios_calib1" in problems[0]
    assert "1 of the 8 train, dev and test families of data/scenarios have the world and the request of " \
           "a family in data/scenarios_calib1" in problems[1]


def test_a_calibration_family_under_another_seed_number_is_still_found(tree):
    """A family is its world and its request. A copy with a new seed number is the same family."""
    put_scenarios(tree, {**SEEDS, "dev": [11, 500]})
    path = tree / "data" / "scenarios" / "dev.jsonl"
    path.write_bytes(path.read_bytes().replace(b"p500", b"p111").replace(b"Person 500", b"Person 111"))
    put_summary(tree)
    problems = freeze.problems_before_freeze(tree)
    assert only(problems, "have the world and the request of a family in data/scenarios_calib1")


def test_every_used_folder_is_looked_at_and_each_one_gets_its_own_line(tree):
    put_scenarios(tree, {"train": [1, 900], "dev": [], "test": [], "calib": []}, "data/scenarios_placeholder")
    put_scenarios(tree, {"train": [2], "dev": [12], "test": [], "calib": []}, "data/scenarios_calib3_mixed")
    problems = freeze.problems_before_freeze(tree)
    assert [problem for problem in problems if "world seeds" in problem] == [
        "2 of the 8 world seeds of the train, dev and test families of data/scenarios are in "
        "data/scenarios_calib3_mixed",
        "1 of the 8 world seeds of the train, dev and test families of data/scenarios are in "
        "data/scenarios_placeholder"]
    assert len(problems) == 4 and not any("scenarios_calib1" in problem for problem in problems)


def test_the_same_family_ids_and_the_same_calibration_split_in_a_used_folder_are_no_problem(tree):
    """The generator numbers the families of every build alike, so the ids are always shared. And the calib
    split of the final folder may hold the calibration families: no result uses that split."""
    def ids(folder: str) -> set[str]:
        return {json.loads(line)["family_id"] for path in sorted((tree / folder).glob("*.jsonl"))
                for line in path.read_text().splitlines()}

    assert ids("data/scenarios_calib1") <= ids("data/scenarios")
    assert SEEDS["calib"] == CALIBRATION_SEEDS["calib"]
    assert freeze.problems_before_freeze(tree) == []


@pytest.mark.parametrize("key", ["world", "request", "world_seed"])
def test_a_family_that_cannot_be_compared_with_the_used_folders_is_a_problem(tree, key):
    """Such a family was not looked at. The answer must not be: nothing found."""
    path = tree / "data" / "scenarios" / "dev.jsonl"
    families = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del families[0][key]
    path.write_text("".join(json.dumps(record) + "\n" for record in families), encoding="utf-8")
    put_summary(tree)
    problems = freeze.problems_before_freeze(tree)
    assert sum("cannot be compared with the folders data/scenarios_*" in problem for problem in problems) == 1
    assert len(problems) == (2 if key == "world_seed" else 1)    # a missing seed is also not a family record


def test_a_world_seed_that_two_families_of_the_final_data_share_is_a_problem(tree):
    put_scenarios(tree, {**SEEDS, "test": [21, SEEDS["train"][0]]})
    assert only(freeze.problems_before_freeze(tree), "1 world seed(s) are used by more than one family")


def test_the_check_for_used_world_seeds_agrees_with_the_one_of_the_build_script(tree):
    """scripts/build_scenarios.py stops a final build that reuses a world seed. The freeze must find the
    same folders, read from the files."""
    build = load_script("build_scenarios")
    put_scenarios(tree, {"train": [3], "dev": [], "test": [], "calib": [31]}, "data/scenarios_calib2_plain")
    put_scenarios(tree, {"train": [], "dev": [], "test": [], "calib": [32, 700]},
                  "data/scenarios_placeholder")
    final = tree / "data" / "scenarios"
    families = {split: [family(split, index, seed) for index, seed in enumerate(seeds)]
                for split, seeds in SEEDS.items()}
    of_the_build = build.reused_world_seeds(families, build.earlier_dirs(final))
    of_the_freeze = [problem for problem in freeze.problems_before_freeze(tree) if "world seeds" in problem]
    assert len(of_the_build) == len(of_the_freeze) == 1
    assert of_the_build[0].startswith("1 of the 8 world seeds") and of_the_freeze[0].startswith("1 of the 8")
    assert "scenarios_calib2_plain" in of_the_build[0] and "scenarios_calib2_plain" in of_the_freeze[0]


def test_every_problem_is_listed_at_once(tree):
    (tree / "ANALYSIS_PLAN.md").unlink()
    put(tree, "MANIFEST.json", b"{}\n")
    put(tree, "text/model_text.yaml", text_bytes(NOT_YET))
    problems = freeze.problems_before_freeze(tree, signed_by="")
    assert len(problems) == 5
    for phrase in ("exists already", "who signs", "ANALYSIS_PLAN.md is missing", "still says",
                   "built from another text file"):
        assert sum(phrase in problem for problem in problems) == 1


# ---------------------------------------------------------------------------
# problems_before_freeze: scenarios that the code no longer gives
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_build(tmp_path_factory) -> Path:
    """A tree whose folder data/scenarios the real build script wrote (scripts/build_scenarios.py), from the
    text file of the tree, the two name files of the repository and the config of accessdesk/config.py.
    It is made once and no test changes it: each test gets a copy (built_tree)."""
    root = make_tree(tmp_path_factory.mktemp("real_build") / "repo")
    for name in ("first_names.txt", "last_names.txt"):
        put(root, f"data/names/{name}", (ROOT / "data" / "names" / name).read_bytes())
    shutil.rmtree(root / "data" / "scenarios")
    text = load_text(root / "text" / "model_text.yaml")
    load_script("build_scenarios").build(text, root / "data" / "names", root / "data" / "scenarios", RunConfig())
    return root


@pytest.fixture
def built_tree(real_build, tmp_path) -> Path:
    """A copy of the tree with the real build, for one test. No check of the freeze is replaced here."""
    return Path(shutil.copytree(real_build, tmp_path / "built"))


def as_an_older_build_wrote_it(root: Path, name: str, change) -> None:
    """Change the content of one file of data/scenarios and write the new file hash into summary.json. The
    folder then fits its own summary, as a folder does that was built before the code changed."""
    folder = root / "data" / "scenarios"
    if name.endswith(".jsonl"):
        families = load_families(folder / name)
        change(families)
        save_families(folder / name, families)
    else:
        content = json.loads((folder / name).read_text(encoding="utf-8"))
        change(content)
        (folder / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    summary["files_sha256"][name] = sha(root, f"data/scenarios/{name}")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def test_a_folder_that_the_build_script_wrote_has_no_problem(built_tree):
    """Every check runs on real families here. It also holds the freeze and the build script together: the
    freeze makes the families again in its own way, and that way must give what the script wrote."""
    assert freeze.problems_before_freeze(built_tree, signed_by="Test Person") == []
    assert freeze.tree_problems(built_tree) == []


def other_resolution(families: list[dict]) -> None:
    request = families[0]["expected"]["default"]["request"]
    request["resolution"] = "granted" if request["resolution"] == "denied" else "denied"


def one_more_turn(families: list[dict]) -> None:
    families[3]["turn_budget"]["default"] += 1


@pytest.mark.parametrize("name, change, phrase", [
    ("dev.jsonl", other_resolution, "1 of the 48 families differ; the first is dv-00, in: expected"),
    ("test.jsonl", one_more_turn, "1 of the 48 families differ; the first is te-03, in: turn_budget"),
    ("train.jsonl", lambda families: families[7]["wordings"].reverse(),
     "1 of the 200 families differ; the first is tr-007, in: wordings"),
    ("calib.jsonl", lambda families: families.pop(), "the file holds 23 families, the build gives 24"),
    ("sets.json", lambda sets: sets["curve_subset"].pop(), "it differs in: curve_subset"),
], ids=["expected_state", "turn_budget", "wordings", "one_family_less", "sets"])
def test_scenarios_that_the_code_no_longer_gives_are_a_problem(built_tree, name, change, phrase):
    """A fault that a review found: the freeze took a folder whose saved expected state was not the state
    that accessdesk/env/rules.py gives. The folder fitted its summary, and nothing else was asked."""
    as_an_older_build_wrote_it(built_tree, name, change)
    problems = freeze.problems_before_freeze(built_tree)
    assert only(problems, f"data/scenarios/{name} is not what the build gives now")
    assert phrase in problems[0]


def test_data_that_was_built_before_the_rules_changed_is_a_problem(built_tree, monkeypatch):
    """The rules change after the build: they deny every request now. The folder is as the build wrote it,
    and its expected states are those of the old rules. The grader would grade against them."""
    monkeypatch.setattr(rules, "decide", lambda active, has_access, sensitive, ruleset="default": "deny")
    problems = freeze.problems_before_freeze(built_tree)
    assert problems and all(problem.startswith("data/scenarios") for problem in problems)
    for split in ("train", "dev", "test", "calib"):
        assert sum(f"data/scenarios/{split}.jsonl is not what the build gives now" in problem
                   and "expected" in problem for problem in problems) == 1


def test_building_the_data_again_writes_nothing(built_tree):
    before = all_files(built_tree)
    freeze.problems_before_freeze(built_tree)
    assert all_files(built_tree) == before


# ---------------------------------------------------------------------------
# check_manifest
# ---------------------------------------------------------------------------
def test_an_untouched_tree_has_no_difference(tree):
    assert freeze.check_manifest(tree, frozen(tree)) == []


def test_a_changed_a_missing_and_a_new_file_are_each_found(tree):
    manifest = frozen(tree)
    old, gone = sha(tree, "accessdesk/env/reward.py"), sha(tree, "accessdesk/harness/parser.py")
    put(tree, "accessdesk/env/reward.py", b"def reward(): return 100\n")
    (tree / "accessdesk" / "harness" / "parser.py").unlink()
    put(tree, "accessdesk/eval/audit.py")
    new = sha(tree, "accessdesk/env/reward.py")
    assert freeze.check_manifest(tree, manifest) == [
        f"changed: accessdesk/env/reward.py (group grader); the manifest has {old[:12]}, the file has "
        f"{new[:12]}",
        f"missing: accessdesk/harness/parser.py (group harness); the manifest has {gone[:12]}",
        "new: accessdesk/eval/audit.py (group evaluation) matches a frozen pattern and is not in the "
        "manifest"]


@pytest.mark.parametrize("path", ["data/scenarios/dev.jsonl", "data/names/last_names.txt",
                                  "text/model_text.yaml", "text/drafting_instructions.yaml",
                                  "accessdesk/config.py", "ANALYSIS_PLAN.md", "requirements-gpu.txt",
                                  "accessdesk/_second/bootstrap.py", "scripts/run_scripted.py",
                                  "scripts/sft_profile.py", "scripts/pick_overfit_families.py",
                                  "scripts/monitor_report.py", "scripts/hand_states.py"])
def test_one_changed_byte_in_any_frozen_file_is_found(tree, path):
    manifest = frozen(tree)
    raw = (tree / path).read_bytes()
    put(tree, path, raw[:-1] + bytes([raw[-1] ^ 1]))
    differences = freeze.check_manifest(tree, manifest)
    assert only(differences, f"changed: {path} (group ")


@pytest.mark.parametrize("path", ["data/scenarios/extra.jsonl", "data/names/more_names.txt",
                                  "accessdesk/harness/new_parser.py", "accessdesk/env/new_tool.py",
                                  "accessdesk/eval/monitors.py", "requirements-extra.txt", "uv.lock",
                                  # A helper in any place from which frozen code could load it.
                                  "accessdesk/env/tables.yaml", "accessdesk/env/sub/helper.py",
                                  "accessdesk/harness/templates/chat.jinja", "accessdesk/eval/extra/helper.py",
                                  "accessdesk/helpers.py", "accessdesk/_second/helper.py"])
def test_a_new_file_that_matches_a_frozen_pattern_is_found(tree, path):
    manifest = frozen(tree)
    put(tree, path)
    assert only(freeze.check_manifest(tree, manifest), f"new: {path} (group ")


@pytest.mark.parametrize("path", ["accessdesk/freeze.py", "accessdesk/_second/grpo_math.py",
                                  "accessdesk/_second/loss_mask.py", "text/placeholder_text.yaml",
                                  "accessdesk/train/sft.py", "accessdesk/eval/calibration.py", "NOTES.md",
                                  "scripts/gpu/train_sft.py", "results/base_dev/run_manifest.json",
                                  "data/scenarios_calib1/dev.jsonl", "accessdesk/eval/__pycache__/x.pyc"])
def test_a_new_or_changed_file_outside_the_frozen_patterns_is_no_difference(tree, path):
    manifest = frozen(tree)
    put(tree, path, b"written after the freeze\n")
    assert freeze.check_manifest(tree, manifest) == []


def test_a_manifest_that_was_changed_by_hand_is_found(tree):
    manifest = frozen(tree)
    put(tree, "accessdesk/env/reward.py", b"def reward(): return 100\n")
    # Somebody writes the new file hash into the manifest and nothing else.
    forged = copy.deepcopy(manifest)
    entry = next(file for file in forged["groups"]["grader"]["files"]
                 if file["path"] == "accessdesk/env/reward.py")
    entry["sha256"] = sha(tree, "accessdesk/env/reward.py")
    assert only(freeze.check_manifest(tree, forged), "the hash of the group grader in the manifest is not")
    # Also the group hash is written in: the code hash still tells.
    forged["groups"]["grader"]["sha256"] = freeze.hash_groups(tree)["grader"]["sha256"]
    assert only(freeze.check_manifest(tree, forged), "code_sha256 in the manifest is not")
    forged["code_sha256"] = freeze.code_sha256(tree)
    assert freeze.check_manifest(tree, forged) == []          # now it is simply a manifest of the new tree
    # The same for the text hash and the config hash.
    for key, phrase in (("text_sha256", "text_sha256 in the manifest"),
                        ("run_config_sha256", "the RunConfig hash")):
        assert only(freeze.check_manifest(tree, {**forged, key: "0" * 64}), phrase)


def test_seeds_and_config_values_that_were_changed_by_hand_in_the_manifest_are_found(tree):
    manifest = json.loads(json.dumps(frozen(tree)))            # as the script reads it from the file
    assert freeze.check_manifest(tree, manifest) == []
    forged = copy.deepcopy(manifest)
    forged["data"]["seeds"]["data_seed"] = 12345
    assert only(freeze.check_manifest(tree, forged), "`data` in the manifest")
    forged = copy.deepcopy(manifest)
    forged["data"]["design"]["mix_facts"] = not manifest["data"]["design"]["mix_facts"]
    assert only(freeze.check_manifest(tree, forged), "`data` in the manifest")
    forged = copy.deepcopy(manifest)
    forged["run_config"]["pass_reward"] = 500                   # the hash beside it is left as it was
    assert only(freeze.check_manifest(tree, forged), "run_config in the manifest does not hold the values")


def test_a_group_that_only_the_manifest_or_only_the_table_has_is_found(tree, monkeypatch):
    manifest = frozen(tree)
    without = {group: patterns for group, patterns in freeze.FROZEN_GROUPS.items() if group != "plan"}
    monkeypatch.setattr(freeze, "FROZEN_GROUPS", without)
    differences = freeze.check_manifest(tree, manifest)
    assert "the group plan is only in the manifest" in differences
    assert sum(line.startswith("missing: ") for line in differences) == 1
    monkeypatch.setattr(freeze, "FROZEN_GROUPS", {**without, "plan": ("ANALYSIS_PLAN.md",),
                                                  "notes": ("README.md",)})
    differences = freeze.check_manifest(tree, manifest)
    assert "the group notes is only in FROZEN_GROUPS of accessdesk/freeze.py" in differences
    assert "new: README.md (group notes) matches a frozen pattern and is not in the manifest" in differences


def test_something_that_is_not_a_manifest_is_refused(tree):
    for wrong in ({}, {"schema": 2, "groups": {}}, {"schema": 1}, []):
        with pytest.raises(ValueError, match="not a manifest of schema 1"):
            freeze.check_manifest(tree, wrong)


# ---------------------------------------------------------------------------
# amend_manifest
# ---------------------------------------------------------------------------
def test_an_amendment_needs_a_reason_and_a_difference(tree):
    manifest = frozen(tree)
    with pytest.raises(ValueError, match="nothing differs"):
        freeze.amend_manifest(tree, manifest, reason="no reason at all", amended_at=LATER)
    put(tree, "accessdesk/eval/tables.py", b"fixed\n")
    for reason in ("", "  \n"):
        with pytest.raises(ValueError, match="needs a reason"):
            freeze.amend_manifest(tree, manifest, reason=reason, amended_at=LATER)


def test_an_amendment_keeps_the_first_hashes_and_adds_one_entry_to_the_history(tree):
    first = frozen(tree)
    untouched = copy.deepcopy(first)
    path = "accessdesk/eval/tables.py"
    hashes = [sha(tree, path)]
    put(tree, path, b"a first fix\n")
    hashes.append(sha(tree, path))
    second = freeze.amend_manifest(tree, first, reason=" a wrong column in the table ", amended_at=LATER)

    assert first == untouched                                              # the old manifest is not changed
    assert freeze.check_manifest(tree, second) == []                       # the new one fits the tree
    assert {key: second[key] for key in ("schema", "frozen_at", "signed_by", "note", "git")} == \
        {key: first[key] for key in ("schema", "frozen_at", "signed_by", "note", "git")}
    assert len(second["amendments"]) == 1
    entry = second["amendments"][0]
    assert (entry["amended_at"], entry["reason"]) == (LATER, "a wrong column in the table")
    assert entry["changes"] == [{"path": path, "group": "evaluation", "kind": "changed",
                                 "old_sha256": hashes[0], "new_sha256": hashes[1]}]
    assert entry["before"]["code_sha256"] == first["code_sha256"] != entry["after"]["code_sha256"]
    assert entry["after"]["code_sha256"] == second["code_sha256"] == freeze.code_sha256(tree)
    assert entry["before"]["group_sha256"]["evaluation"] == first["groups"]["evaluation"]["sha256"]
    assert entry["before"]["group_sha256"]["grader"] == entry["after"]["group_sha256"]["grader"]

    # A second amendment: the same file again, a new analysis module, and a file that is gone.
    put(tree, path, b"a second fix\n")
    hashes.append(sha(tree, path))
    put(tree, "accessdesk/eval/audit.py")
    gone = sha(tree, "requirements-gpu.txt")
    (tree / "requirements-gpu.txt").unlink()
    third = freeze.amend_manifest(tree, second, reason="audit code, written after the freeze",
                                  amended_at="2026-11-20T09:00:00Z")
    assert third["amendments"][0] == entry                                 # the earlier entry is as it was
    assert third["amendments"][1]["changes"] == [
        {"path": "accessdesk/eval/audit.py", "group": "evaluation", "kind": "new", "old_sha256": None,
         "new_sha256": sha(tree, "accessdesk/eval/audit.py")},
        {"path": path, "group": "evaluation", "kind": "changed", "old_sha256": hashes[1],
         "new_sha256": hashes[2]},
        {"path": "requirements-gpu.txt", "group": "lock", "kind": "missing", "old_sha256": gone,
         "new_sha256": None}]
    # Every hash that the file ever had under the freeze can be read from the history.
    history = [change for amendment in third["amendments"] for change in amendment["changes"]
               if change["path"] == path]
    assert [history[0]["old_sha256"]] + [change["new_sha256"] for change in history] == hashes
    assert third["amendments"][0]["before"]["code_sha256"] == first["code_sha256"]
    assert third["amendments"][1]["before"]["code_sha256"] == second["code_sha256"]
    assert freeze.check_manifest(tree, third) == [] and third["frozen_at"] == NOW


def test_an_amendment_takes_the_new_seeds_and_hashes_of_scenarios_that_were_built_again(tree):
    first = frozen(tree)
    put_scenarios(tree, {**SEEDS, "dev": [13, 14]})
    amended = freeze.amend_manifest(tree, first, reason="a wrong dev family", amended_at=LATER)
    changed = {change["path"] for change in amended["amendments"][0]["changes"]}
    assert changed == {"data/scenarios/dev.jsonl", "data/scenarios/summary.json"}
    assert amended["code_sha256"] == first["code_sha256"]                  # no code changed
    assert amended["groups"]["scenarios"]["sha256"] != first["groups"]["scenarios"]["sha256"]


def test_an_amendment_keeps_the_seeds_and_the_config_values_of_the_freeze(tree):
    """The new manifest holds the new seeds. The seeds of the freeze must still be readable in it."""
    first = frozen(tree)
    put_summary(tree, config={**RunConfig().to_dict(), "data_seed": 999})
    second = freeze.amend_manifest(tree, first, reason="the data was built again with another seed",
                                   amended_at=LATER)
    entry = second["amendments"][0]
    assert first["data"]["seeds"]["data_seed"] == RunConfig().data_seed != 999
    assert second["data"]["seeds"]["data_seed"] == 999
    assert entry["before"]["data"] == first["data"] and entry["after"]["data"] == second["data"]
    assert entry["before"]["run_config"] == first["run_config"] == RunConfig().to_dict()
    assert json.loads(json.dumps(second)) == second             # it can be written as JSON


# ---------------------------------------------------------------------------
# frozen_state, for run scripts
# ---------------------------------------------------------------------------
def test_without_a_manifest_the_state_names_the_code_and_says_not_frozen(tree):
    state = freeze.frozen_state(tree)
    assert state == {"manifest_sha256": None, "code_sha256": freeze.code_sha256(tree), "matches": False,
                     "differences": ["MANIFEST.json is missing: the tree is not frozen"]}


def test_a_frozen_tree_matches_until_a_frozen_file_changes(tree, freeze_script):
    assert freeze_script.main(["--root", str(tree), "write", "--signed-by", "Test Person"], now=NOW) == 0
    raw = (tree / "MANIFEST.json").read_bytes()
    state = freeze.frozen_state(tree)
    assert state == {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
                     "code_sha256": json.loads(raw)["code_sha256"], "matches": True, "differences": []}
    put(tree, "accessdesk/harness/parser.py", b"another parser\n")
    changed = freeze.frozen_state(tree)
    assert changed["matches"] is False and changed["manifest_sha256"] == state["manifest_sha256"]
    assert changed["code_sha256"] != state["code_sha256"]                  # the run names the code it used
    assert only(changed["differences"], "changed: accessdesk/harness/parser.py (group harness)")
    json.dumps(changed)                                                    # it can go into a run manifest


def test_a_manifest_that_cannot_be_read_never_matches(tree):
    for raw in (b"not json", b"[]\n", b'{"schema": 7, "groups": {}}\n'):
        put(tree, "MANIFEST.json", raw)
        state = freeze.frozen_state(tree)
        assert state["matches"] is False and only(state["differences"], "MANIFEST.json cannot be read")
        assert state["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
        assert "MANIFEST.json cannot be read" in freeze.run_problem(state)       # such a tree stops a run


def test_a_run_may_start_before_the_freeze_and_on_a_tree_that_is_as_frozen(tree, freeze_script):
    """`run_problem` is what scripts/gpu/rollout.py and scripts/run_scripted.py ask before they start."""
    assert freeze.run_problem(freeze.frozen_state(tree)) is None                 # there is no manifest yet
    put(tree, "accessdesk/harness/parser.py", b"a parser that changed before the freeze\n")
    assert freeze.run_problem(freeze.frozen_state(tree)) is None
    assert freeze_script.main(["--root", str(tree), "write", "--signed-by", "Test Person"], now=NOW) == 0
    assert freeze.run_problem(freeze.frozen_state(tree)) is None                 # as frozen
    # A file outside the frozen patterns may change after the freeze: training code is not frozen.
    put(tree, "accessdesk/train/grpo.py", b"training code that changed after the freeze\n")
    assert freeze.run_problem(freeze.frozen_state(tree)) is None


def test_a_run_must_not_start_on_a_frozen_tree_that_has_changed_unless_it_is_allowed(tree, freeze_script,
                                                                                    capsys):
    assert freeze_script.main(["--root", str(tree), "write", "--signed-by", "Test Person"], now=NOW) == 0
    put(tree, "accessdesk/harness/parser.py", b"another parser\n")
    put(tree, "scripts/sft_profile.py", b"another rule\n")
    (tree / "ANALYSIS_PLAN.md").unlink()
    put(tree, "accessdesk/eval/audit.py")
    state = freeze.frozen_state(tree)
    problem = freeze.run_problem(state)
    assert problem.startswith("the tree is frozen, and it differs from MANIFEST.json:\n- ")
    # Every difference of `check` is in the message, each on its own line: a changed harness file, a
    # changed rule script, a missing plan and a new analysis module.
    assert len(state["differences"]) == 4
    assert [line[2:] for line in problem.splitlines() if line.startswith("- ")] == state["differences"]
    for phrase in ("changed: accessdesk/harness/parser.py (group harness)",
                   "changed: scripts/sft_profile.py (group evaluation)",
                   "missing: ANALYSIS_PLAN.md (group plan)", "new: accessdesk/eval/audit.py (group evaluation)"):
        assert phrase in problem
    assert "scripts/freeze.py amend" in problem and "--allow-changed-tree" in problem
    # The option of the run scripts lets the run start; the state still says that the tree differs.
    assert freeze.run_problem(state, allow_changed_tree=True) is None and state["matches"] is False
    # After an amendment the tree is as its manifest says again, and a run may start without the option.
    put(tree, "ANALYSIS_PLAN.md")
    capsys.readouterr()
    assert freeze_script.main(["--root", str(tree), "amend", "--reason", "a test"], now=LATER) == 0
    assert freeze.run_problem(freeze.frozen_state(tree)) is None


# ---------------------------------------------------------------------------
# scripts/freeze.py, through main(argv)
# ---------------------------------------------------------------------------
def run(freeze_script, capsys, tree, *argv: str, now: str = NOW) -> tuple[int, str, str]:
    code = freeze_script.main(["--root", str(tree), *argv], now=now)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_write_makes_the_manifest_and_a_second_write_is_refused(tree, freeze_script, capsys):
    before = all_files(tree)
    code, out, _ = run(freeze_script, capsys, tree, "write", "--signed-by", "Test Person", "--note", "first")
    manifest = json.loads((tree / "MANIFEST.json").read_bytes())
    assert code == 0 and "MANIFEST.json" in out and manifest["code_sha256"] in out
    assert manifest == freeze.build_manifest(tree, signed_by="Test Person", frozen_at=NOW, note="first")
    assert (tree / "MANIFEST.json").read_bytes().endswith(b"}\n")
    written = all_files(tree)
    assert {path: digest for path, digest in written.items() if path != "MANIFEST.json"} == before

    code, _, err = run(freeze_script, capsys, tree, "write", "--signed-by", "Somebody Else", now=LATER)
    assert code == 1 and "MANIFEST.json exists already" in err
    assert all_files(tree) == written                                      # the first manifest stays


def test_write_refuses_with_the_whole_list_and_writes_nothing(tree, freeze_script, capsys):
    (tree / "ANALYSIS_PLAN.md").unlink()
    put(tree, "text/model_text.yaml", text_bytes(NOT_YET))
    put_summary(tree)
    before = all_files(tree)
    code, out, err = run(freeze_script, capsys, tree, "write", "--signed-by", " ")
    assert code == 1 and out == ""
    assert err.splitlines()[0] == "not frozen: 3 problem(s)"
    assert [line[:2] for line in err.splitlines()[1:]] == ["- ", "- ", "- "]
    for phrase in ("who signs the freeze is empty", "ANALYSIS_PLAN.md is missing", "still says"):
        assert phrase in err
    assert all_files(tree) == before and not (tree / "MANIFEST.json").exists()


def test_write_and_check_take_a_tree_with_a_real_build(built_tree, freeze_script, capsys):
    """The command line on real families, with no check replaced."""
    assert run(freeze_script, capsys, built_tree, "write", "--signed-by", "Test Person")[0] == 0
    assert (built_tree / "MANIFEST.json").is_file()
    assert run(freeze_script, capsys, built_tree, "check")[0] == 0


def test_write_refuses_data_that_the_code_no_longer_gives(built_tree, freeze_script, capsys):
    as_an_older_build_wrote_it(built_tree, "dev.jsonl", other_resolution)
    code, out, err = run(freeze_script, capsys, built_tree, "write", "--signed-by", "Test Person")
    assert code == 1 and "data/scenarios/dev.jsonl is not what the build gives now" in out + err
    assert not (built_tree / "MANIFEST.json").exists()


def test_amend_refuses_while_the_data_is_not_what_the_changed_code_gives(built_tree, freeze_script, capsys,
                                                                       monkeypatch):
    """After the freeze the rules change. The amendment must wait until the data is built again: else the
    new manifest would name new rules beside the expected states of the old ones."""
    assert run(freeze_script, capsys, built_tree, "write", "--signed-by", "Test Person")[0] == 0
    frozen_bytes = (built_tree / "MANIFEST.json").read_bytes()
    put(built_tree, "accessdesk/env/rules.py", b"the rules, changed after the freeze\n")
    monkeypatch.setattr(rules, "decide", lambda active, has_access, sensitive, ruleset="default": "deny")
    code, out, err = run(freeze_script, capsys, built_tree, "amend", "--reason", "a new rule", now=LATER)
    assert code == 1 and "is not what the build gives now" in out + err
    assert (built_tree / "MANIFEST.json").read_bytes() == frozen_bytes


def test_the_time_of_the_freeze_is_utc_in_iso_form(tree, freeze_script, capsys):
    earliest = datetime.now(timezone.utc).replace(microsecond=0)
    assert freeze_script.main(["--root", str(tree), "write", "--signed-by", "Test Person"]) == 0
    capsys.readouterr()
    frozen_at = json.loads((tree / "MANIFEST.json").read_bytes())["frozen_at"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", frozen_at)
    written = datetime.strptime(frozen_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert earliest <= written <= datetime.now(timezone.utc)


def test_check_ends_with_0_for_no_difference_1_for_a_difference_and_2_without_a_manifest(tree, freeze_script,
                                                                                         capsys):
    code, _, err = run(freeze_script, capsys, tree, "check")
    assert code == 2 and "is missing: the tree is not frozen" in err
    run(freeze_script, capsys, tree, "write", "--signed-by", "Test Person")
    code, out, err = run(freeze_script, capsys, tree, "check")
    assert code == 0 and "nothing changed" in out and err == ""
    put(tree, "accessdesk/env/verifier.py", b"def verify(a, b): return True\n")
    put(tree, "accessdesk/eval/audit.py")
    before = all_files(tree)
    code, out, err = run(freeze_script, capsys, tree, "check")
    assert code == 1 and out == ""
    assert err.splitlines()[0] == "2 difference(s) between the tree and MANIFEST.json"
    assert "- changed: accessdesk/env/verifier.py (group grader)" in err
    assert "- new: accessdesk/eval/audit.py (group evaluation)" in err
    assert all_files(tree) == before                                       # check writes nothing


def test_amend_records_the_change_and_then_check_is_quiet_again(tree, freeze_script, capsys):
    run(freeze_script, capsys, tree, "write", "--signed-by", "Test Person")
    first = json.loads((tree / "MANIFEST.json").read_bytes())
    put(tree, "accessdesk/eval/tables.py", b"a fix\n")
    others = {path: digest for path, digest in all_files(tree).items() if path != "MANIFEST.json"}
    code, out, _ = run(freeze_script, capsys, tree, "amend", "--reason", "a wrong column", now=LATER)
    assert code == 0 and "changed  accessdesk/eval/tables.py  (group evaluation)" in out
    amended = json.loads((tree / "MANIFEST.json").read_bytes())
    assert amended == freeze.amend_manifest(tree, first, reason="a wrong column", amended_at=LATER)
    assert (amended["frozen_at"], amended["amendments"][0]["amended_at"]) == (NOW, LATER)
    assert amended["amendments"][0]["before"]["code_sha256"] == first["code_sha256"]
    assert run(freeze_script, capsys, tree, "check")[0] == 0
    assert {path: digest for path, digest in all_files(tree).items() if path != "MANIFEST.json"} == others
    assert sorted(path.name for path in tree.iterdir() if "MANIFEST" in path.name) == ["MANIFEST.json"]


def test_amend_refuses_an_empty_reason_no_difference_a_problem_and_a_missing_manifest(tree, freeze_script,
                                                                                    capsys):
    code, _, err = run(freeze_script, capsys, tree, "amend", "--reason", "x")
    assert code == 2 and "there is nothing to amend" in err
    run(freeze_script, capsys, tree, "write", "--signed-by", "Test Person")
    frozen_bytes = (tree / "MANIFEST.json").read_bytes()

    code, _, err = run(freeze_script, capsys, tree, "amend", "--reason", "nothing happened")
    assert code == 1 and "nothing differs from the manifest" in err
    put(tree, "accessdesk/eval/tables.py", b"a fix\n")
    code, _, err = run(freeze_script, capsys, tree, "amend", "--reason", "  ")
    assert code == 2 and "needs a reason" in err
    # A split file was changed by hand: the tree has a problem, and a manifest of it would say nothing.
    append_to("data/scenarios/dev.jsonl")(tree)
    code, _, err = run(freeze_script, capsys, tree, "amend", "--reason", "a new dev file")
    assert code == 1 and "not amended: 1 problem(s)" in err and "is not the file that the build wrote" in err
    assert (tree / "MANIFEST.json").read_bytes() == frozen_bytes           # every refusal left it as it was


def test_show_prints_one_line_per_group_with_or_without_a_manifest(tree, freeze_script, capsys):
    before = all_files(tree)
    code, out, _ = run(freeze_script, capsys, tree, "show")
    assert code == 0 and "MANIFEST.json is not there" in out and all_files(tree) == before
    rows = {line.split()[0]: line.split()[1:] for line in out.splitlines()[-len(freeze.FROZEN_GROUPS):]}
    assert list(rows) == list(freeze.FROZEN_GROUPS)
    assert rows["grader"] == ["7", freeze.hash_groups(tree)["grader"]["sha256"]]
    assert rows["scenarios"][0] == "6" and rows["plan"][0] == "1"

    run(freeze_script, capsys, tree, "write", "--signed-by", "Test Person")
    put(tree, "accessdesk/eval/tables.py", b"a fix\n")
    run(freeze_script, capsys, tree, "amend", "--reason", "a wrong column", now=LATER)
    code, out, _ = run(freeze_script, capsys, tree, "show")
    manifest = json.loads((tree / "MANIFEST.json").read_bytes())
    assert code == 0 and f"frozen at {NOW}, signed by Test Person" in out
    assert f"code_sha256        {manifest['code_sha256']}" in out
    assert f"  1. {LATER}  1 file(s): a wrong column" in out
    rows = {line.split()[0]: line.split()[1:] for line in out.splitlines()[-len(freeze.FROZEN_GROUPS):]}
    assert len(FROZEN["evaluation"]) == 14                       # the files of the fake tree, counted by hand
    assert rows["evaluation"] == ["14", manifest["groups"]["evaluation"]["sha256"]]


def test_the_root_can_come_from_the_environment_and_the_option_wins(tree, tmp_path, freeze_script, capsys,
                                                                    monkeypatch):
    other = make_tree(tmp_path / "other")
    monkeypatch.setenv(freeze_script.ROOT_VARIABLE, str(tree))
    assert freeze_script.main(["write", "--signed-by", "Test Person"], now=NOW) == 0
    assert (tree / "MANIFEST.json").is_file() and not (other / "MANIFEST.json").exists()
    assert freeze_script.main(["check"]) == 0
    assert freeze_script.main(["--root", str(other), "check"]) == 2        # --root wins: no manifest there
    capsys.readouterr()
    assert freeze_script.tree_root(None) == tree
    monkeypatch.delenv(freeze_script.ROOT_VARIABLE)
    assert freeze_script.tree_root(None) == ROOT and freeze_script.tree_root(other) == other


# ---------------------------------------------------------------------------
# scripts/approve_text.py (always on a copy in a temporary folder)
# ---------------------------------------------------------------------------
@pytest.fixture
def text_file(tmp_path) -> Path:
    """A valid text file whose header still says "(not yet)"."""
    path = tmp_path / "copy" / "model_text.yaml"
    path.parent.mkdir()
    path.write_bytes(text_bytes(NOT_YET))
    return path


def approve(approve_script, capsys, path: Path, name: str, today: str = "2026-11-18") -> tuple[int, str, str]:
    code = approve_script.main(["--name", name, "--text", str(path)], today=today)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_approval_replaces_not_yet_by_the_name_and_the_date_and_changes_no_other_byte(text_file, capsys,
                                                                                    approve_script):
    old = text_file.read_bytes()
    values = load_text(text_file)
    code, out, _ = approve(approve_script, capsys, text_file, "  Test Person ")
    new = text_file.read_bytes()
    assert code == 0 and APPROVED in out
    assert new == old.replace(b"(not yet)", b"Test Person, 2026-11-18 (UTC)")
    assert old.count(b"(not yet)") == 1 and len(new) == len(old) + len(b"Test Person, 2026-11-18 (UTC)") - 9
    old_lines, new_lines = old.split(b"\n"), new.split(b"\n")
    differing = [index for index, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    assert len(old_lines) == len(new_lines) and differing == [2] and new_lines[2] == APPROVED.encode()
    after = load_text(text_file)
    assert after.sha256 != values.sha256                                   # the file has a new hash
    assert {**vars(after), "sha256": ""} == {**vars(values), "sha256": ""}  # and every value is the same
    assert [path.name for path in text_file.parent.iterdir()] == ["model_text.yaml"]


def test_after_the_approval_the_freeze_takes_the_text_as_approved(tree, approve_script, capsys):
    path = tree / "text" / "model_text.yaml"
    path.write_bytes(text_bytes(NOT_YET))
    put_summary(tree)
    assert only(freeze.problems_before_freeze(tree), "still says")
    assert approve(approve_script, capsys, path, "Test Person")[0] == 0
    # The hash of the text is new now, so the scenarios must be built again from it.
    assert only(freeze.problems_before_freeze(tree), "built from another text file")
    put_summary(tree)
    assert freeze.problems_before_freeze(tree) == []


def test_only_the_header_line_is_changed_when_the_same_line_stands_below_the_header(text_file, approve_script,
                                                                                 capsys):
    text_file.write_bytes(text_bytes(NOT_YET) + f"\n{NOT_YET}\n".encode())      # a comment after the values
    assert approve(approve_script, capsys, text_file, "Test Person")[0] == 0
    new_lines = text_file.read_bytes().split(b"\n")
    assert new_lines[2] == APPROVED.encode() and new_lines[-2] == NOT_YET.encode()
    assert text_file.read_bytes().count(b"(not yet)") == 1


def test_a_name_in_another_script_is_written_as_utf8(text_file, approve_script, capsys):
    assert approve(approve_script, capsys, text_file, "Zoë Müller-Šimić")[0] == 0
    line = "# Read and approved by: Zoë Müller-Šimić, 2026-11-18 (UTC)"
    assert line.encode("utf-8") in text_file.read_bytes()
    assert freeze.approval_problem(text_file.read_bytes()) is None


def test_the_date_of_the_approval_is_the_utc_date_of_today(text_file, approve_script, capsys):
    days = {datetime.now(timezone.utc).date().isoformat()}
    assert approve_script.main(["--name", "Test Person", "--text", str(text_file)]) == 0
    days.add(datetime.now(timezone.utc).date().isoformat())               # two days if midnight passed
    capsys.readouterr()
    value = freeze.approval_values(text_file.read_bytes())[0]
    assert value in {f"Test Person, {day} (UTC)" for day in days}


@pytest.mark.parametrize("name, phrase", [("", "the name is empty"), ("   ", "the name is empty"),
                                          ("Test\nPerson", "one line"), ("(not yet)", "must not hold"),
                                          ("Test\u2028Person", "one line"), ("Test\x85Person", "one line"),
                                          ("Test\rPerson", "one line"), ("Test\ud800", "as UTF-8")])
def test_a_name_that_cannot_be_written_is_refused(text_file, approve_script, capsys, name, phrase):
    old = text_file.read_bytes()
    code, _, err = approve(approve_script, capsys, text_file, name)
    assert code == 1 and phrase in err and "is unchanged" in err
    assert text_file.read_bytes() == old
    assert [path.name for path in text_file.parent.iterdir()] == ["model_text.yaml"]


@pytest.mark.parametrize("content, phrase", [
    (lambda: text_bytes("# no approval line here"), "exactly once, not 0 times"),
    (lambda: text_bytes(NOT_YET + "\n" + NOT_YET), "exactly once, not 2 times"),
    (lambda: text_bytes(APPROVED), "is not exactly"),
    (lambda: text_bytes(NOT_YET + " "), "is not exactly"),
    (lambda: text_bytes("# Read and approved by:  (not yet)"), "is not exactly"),
    (lambda: text_bytes(NOT_YET, placeholder=True), "is a placeholder"),
    (lambda: f"{NOT_YET}\nmeta:\n  placeholder: false\n".encode(), "is not a valid text file"),
    (lambda: text_bytes(NOT_YET).replace(b"\n", b"\r\n"), "(not yet)\\r'"),
], ids=["no_line", "two_lines", "approved_before", "space_after", "two_spaces", "placeholder", "not_valid",
        "crlf_line_ends"])
def test_a_file_without_exactly_the_open_line_is_refused_and_left_as_it_is(text_file, approve_script, capsys,
                                                                         content, phrase):
    old = content()
    text_file.write_bytes(old)
    code, out, err = approve(approve_script, capsys, text_file, "Test Person")
    assert code == 1 and out == "" and phrase in err
    assert text_file.read_bytes() == old
    assert [path.name for path in text_file.parent.iterdir()] == ["model_text.yaml"]


def test_a_second_approval_is_refused_and_keeps_the_first(text_file, approve_script, capsys):
    assert approve(approve_script, capsys, text_file, "Test Person")[0] == 0
    first = text_file.read_bytes()
    code, _, err = approve(approve_script, capsys, text_file, "Somebody Else", today="2026-11-19")
    assert code == 1 and "is not exactly" in err and text_file.read_bytes() == first
