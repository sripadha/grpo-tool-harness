"""The freeze: which files must not change after the first training run, their hashes, and the checks.

PLAN 5.3. Before any training run the project is frozen: the scenarios, the splits and sets, the grader
(verifier and reward), the parser and harness, the evaluation code, the text that the model reads, the
config and the analysis plan. Their hashes go into MANIFEST.json. After that a frozen file changes only
with a recorded reason (an amendment), and results from before and after a change are separate experiments.

What is here:
    FROZEN_GROUPS            the frozen files, as path patterns in named groups
    build_manifest           the manifest of the tree as it is now (scripts/freeze.py writes it to disk)
    problems_before_freeze   why the tree must not be frozen yet
    check_manifest           every difference between the tree and a manifest
    amend_manifest           the manifest with the new hashes and one more entry in its history
    code_sha256, frozen_state    for run scripts: the hash of the code that made a run
    run_problem              for run scripts: why a run must not start on a frozen tree that has changed

The three kinds of hash. Each one can be computed again with `sha256sum`, without this module:
    file    the SHA-256 of the file's bytes
    group   the SHA-256 of the lines "<file hash>  <path>\\n" of the group's files, in path order.
            These lines are what `sha256sum` prints, so `sha256sum PATHS | sha256sum` gives the group hash.
    code    the SHA-256 of the lines "<group hash>  <group name>\\n" for the groups of CODE_GROUPS

This module reads files as bytes. It never writes, renames or deletes a file. The only programs it starts
are read-only git commands. To ask whether data/scenarios is still what the code gives, it makes the
families and the sets again in memory (generator, rules, sets) and compares; nothing of that is written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from datetime import date
from pathlib import Path

from accessdesk.config import RunConfig
from accessdesk.env.generator import SPLITS, load_name_pools, make_all_splits
from accessdesk.env.scenarios import load_families, make_sets
from accessdesk.text import check_text, load_text

SCHEMA = 1                                    # version of the manifest
MANIFEST_NAME = "MANIFEST.json"
ROOT = Path(__file__).resolve().parent.parent

# Group name -> path patterns, relative to the repository root, with "/".
#   A pattern without a wildcard (* ? [) names one file. That file must exist at the freeze. A pattern
#   with a wildcard takes every file that it matches ("**" goes into folders); it may match nothing, and a
#   file that appears later under it is reported as new.
#   A file that is named without a wildcard belongs to that group, also where a wildcard of another group
#   matches it (the three grader files in accessdesk/env/). Otherwise a file in two groups is an error.
#   The wildcards over the code are wide on purpose: every file of accessdesk/env/, accessdesk/harness/
#   and accessdesk/eval/ (any kind, also in a folder below), every module at the top of accessdesk/ and
#   every module of accessdesk/_second/, except the files of NOT_FROZEN. So a helper that is written
#   after the freeze, in any of these places, is reported as new. A new text file beside
#   text/model_text.yaml is not caught: a run names the hash of the text that it read.
#
# Not frozen, on purpose:
#   accessdesk/train/ and the training parts of accessdesk/_second/ (grpo_math.py, loss_mask.py), the three
#       training scripts (scripts/gpu/sft_data.py, train_sft.py, train_grpo.py) and the two chain scripts:
#       training code is recorded at each training run, in the manifest of that run (`training_code` in
#       scripts/gpu/sft_data.py).
#   accessdesk/eval/calibration.py and scripts/calibration_report.py: calibration numbers are not results.
#   The tests, the placeholder text, the folders data/scenarios_* (calibration and test data), results/,
#       the rental and smoke scripts, and this module with its scripts: none of them makes a reported number.
# tests/core/test_freeze.py walks scripts/ and accessdesk/ of the repository: every file there is in a
# group, or in the training code, or in the list of that test with its reason.
FROZEN_GROUPS: dict[str, tuple[str, ...]] = {
    "scenarios": ("data/scenarios/summary.json", "data/scenarios/**/*"),
    "names": ("data/names/first_names.txt", "data/names/last_names.txt", "data/names/PROVENANCE.md",
              "data/names/**/*"),
    "text": ("text/model_text.yaml",),
    # How the text was made: the instructions, every candidate, and the program that drew and chose them
    # (the header of the text file names that program).
    "text_provenance": ("text/drafting_instructions.yaml", "text/drafting_log.jsonl",
                        "accessdesk/drafting.py", "scripts/gpu/draft_text.py"),
    "grader": ("accessdesk/env/verifier.py", "accessdesk/env/reward.py", "accessdesk/env/rules.py",
               "accessdesk/_second/__init__.py", "accessdesk/_second/verifier.py",
               "accessdesk/_second/reward.py", "accessdesk/impl.py", "accessdesk/_second/*.py"),
    "environment": ("accessdesk/__init__.py", "accessdesk/text.py", "accessdesk/types.py",
                    "accessdesk/rng.py", "accessdesk/env/**/*", "accessdesk/*.py"),
    "harness": ("accessdesk/harness/**/*",),
    # The code that computes or checks a number, the scripts that produce or check episodes, and the two
    # scripts that apply a rule of the analysis plan: scripts/sft_profile.py (the size of a GRPO step) and
    # scripts/pick_overfit_families.py (the families of the overfit check). The wildcard takes every
    # analysis module that is written after the freeze: `check` then reports it as new, and an amendment
    # gives it a hash and a date (PLAN 5.3).
    # A script that is written later and computes a number or applies a rule must be added here by hand.
    "evaluation": ("accessdesk/eval/bootstrap.py", "accessdesk/eval/metrics.py", "accessdesk/eval/tables.py",
                   "accessdesk/eval/**/*", "accessdesk/_second/bootstrap.py",
                   "scripts/gpu/rollout.py", "scripts/run_scripted.py", "scripts/build_scenarios.py",
                   "scripts/check_wordings.py", "scripts/replay_check.py", "scripts/monitor_report.py",
                   "scripts/hand_states.py", "scripts/sft_profile.py", "scripts/pick_overfit_families.py"),
    "config": ("accessdesk/config.py",),
    # PREDICTIONS.md was in this group until amendment 3 (2026-11-21). The file is not part of the package;
    # the amendment in MANIFEST.json keeps its SHA-256 at the freeze.
    "plan": ("ANALYSIS_PLAN.md",),
    "lock": ("pyproject.toml", "requirements*.txt", "*.lock"),
    # The audit of ANALYSIS_PLAN.md section 8 and the second-desk report of section 3. This code was
    # written after the freeze, on 2026-11-18, and it only reads saved attempts. An amendment gave it
    # its hashes ("gets a hash and a date there and splits nothing"). The group is NOT in CODE_GROUPS:
    # no attempt was made by this code, so code_sha256 is the same before and after, for every run.
    # A module that is written later in accessdesk/audit/ is reported as new, like one in accessdesk/eval/.
    "audit": ("scripts/second_desk_report.py", "scripts/audit_sheet.py", "scripts/formula_audit.py",
              "scripts/audit_positive_control.py", "accessdesk/audit/**/*"),
    # The report code: the four pieces of the report (figure, training, failures, stretch), the command that
    # makes every reported file again and compares it, and the package list of the figure. Written after the
    # freeze, on 2026-11-19; it only reads saved runs. Amendment 2 gave it its hashes ("gets a hash and a
    # date there and splits nothing"). Not in CODE_GROUPS, like "audit": code_sha256 stays the same.
    "report": ("scripts/make_figure.py", "scripts/training_report.py", "scripts/failure_report.py",
               "scripts/stretch_numbers.py", "scripts/rebuild_results.sh", "scripts/requirements-report.txt"),
}
# Files that a wildcard of FROZEN_GROUPS matches and that are not frozen (see "Not frozen" above).
NOT_FROZEN = ("accessdesk/eval/calibration.py", "accessdesk/freeze.py",
              "accessdesk/_second/grpo_math.py", "accessdesk/_second/loss_mask.py")
# The groups of code_sha256, in the order in which their hashes are put together. The groups "audit" and
# "report" are left out on purpose (see FROZEN_GROUPS): the hash names the code that makes and scores an
# attempt.
CODE_GROUPS = ("grader", "environment", "harness", "evaluation", "config")

TEXT_FILE = "text/model_text.yaml"
FINAL_DIR = "data/scenarios"                  # the folder of the final families
SUMMARY_FILE = FINAL_DIR + "/summary.json"
SETS_NAME = "sets.json"                       # the file of the dev sets in the folder of the families
NAMES_DIR = "data/names"
NAME_FILES = ("first_names.txt", "last_names.txt")    # the two files that the generator reads
EARLIER_DIRS_PATTERN = "data/scenarios_*"     # folders of families that were already used
FRESH_SPLITS = ("train", "dev", "test")       # the splits of the final folder that must be new
SEED_FIELDS = ("data_seed", "calib_seed")     # fields of the build config in summary.json
DESIGN_FIELDS = ("mix_facts", "homonym_share")
APPROVAL_PREFIX = "# Read and approved by:"   # a header line of the text file
NOT_APPROVED = "(not yet)"                    # what stands after it until a person has read the text
# What stands after it when a person has approved: "NAME, YYYY-MM-DD (UTC)" (scripts/approve_text.py).
APPROVED_FORM = re.compile(r"(?P<name>.*\S), (?P<day>\d{4}-\d{2}-\d{2}) \(UTC\)")
HASH_RULE = ("file: SHA-256 of the bytes. group: SHA-256 of the lines '<file sha256>  <path>\\n' of its "
             "files in path order (the output of sha256sum). code_sha256: SHA-256 of the lines "
             "'<group sha256>  <group>\\n' for the groups " + ", ".join(CODE_GROUPS) + ".")
_WILDCARDS = "*?["
_SHOWN = 12                                   # hexadecimal digits of a hash in a message


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------
def sha256_of_bytes(raw: bytes) -> str:
    """The SHA-256 of the bytes, as 64 hexadecimal digits."""
    return hashlib.sha256(raw).hexdigest()


def sha256_of_file(path: str | Path) -> str:
    """The SHA-256 of the file's bytes, as 64 hexadecimal digits."""
    return sha256_of_bytes(Path(path).read_bytes())


def sha256_of_lines(pairs: list[tuple[str, str]]) -> str:
    """The SHA-256 of the lines "<hash>  <name>\\n", one per pair (hash, name), in the given order."""
    return sha256_of_bytes("".join(f"{digest}  {name}\n" for digest, name in pairs).encode("utf-8"))


# ---------------------------------------------------------------------------
# Which files are frozen
# ---------------------------------------------------------------------------
def is_exact(pattern: str) -> bool:
    """True for a pattern that names one file: it holds none of the wildcard characters * ? [."""
    return not any(character in pattern for character in _WILDCARDS)


def exact_paths() -> dict[str, str]:
    """path -> group, for every path that FROZEN_GROUPS names without a wildcard.

    Raises ValueError for a mistake in the table: a path that two groups name, or one that is in NOT_FROZEN.
    """
    found: dict[str, str] = {}
    for group, patterns in FROZEN_GROUPS.items():
        for pattern in filter(is_exact, patterns):
            if found.get(pattern, group) != group:
                raise ValueError(f"{pattern} is named in two groups: {found[pattern]} and {group}")
            if pattern in NOT_FROZEN:
                raise ValueError(f"{pattern} is named in the group {group} and in NOT_FROZEN")
            found[pattern] = group
    return found


def required_files() -> dict[str, str]:
    """path -> group, for the files that must exist at the freeze: every path that FROZEN_GROUPS names
    without a wildcard."""
    return exact_paths()


def _is_compiled(path: str) -> bool:
    """True for a file that Python makes when code runs (.gitignore lists them). They are never frozen."""
    return "__pycache__" in path.split("/") or path.endswith(".pyc")


def frozen_files(root: str | Path) -> dict[str, list[str]]:
    """group -> the files of the tree that belong to it now: sorted paths, relative to root, with "/".

    Every group of FROZEN_GROUPS is a key, also one without a file. No path is in two lists.
    Raises ValueError if wildcards of two groups match the same file (a mistake in FROZEN_GROUPS).
    """
    root = Path(root)
    named = exact_paths()
    group_of = {path: group for path, group in named.items() if (root / path).is_file()}
    for group, patterns in FROZEN_GROUPS.items():
        for pattern in patterns:
            if is_exact(pattern):
                continue
            for match in root.glob(pattern):
                path = match.relative_to(root).as_posix()
                if not match.is_file() or path in named or path in NOT_FROZEN or _is_compiled(path):
                    continue
                if group_of.get(path, group) != group:
                    raise ValueError(f"{path} is in two groups: {group_of[path]} and {group}")
                group_of[path] = group
    return {group: sorted(path for path, owner in group_of.items() if owner == group)
            for group in FROZEN_GROUPS}


def hash_groups(root: str | Path) -> dict[str, dict]:
    """group -> {"sha256": the group hash, "files": [{"path", "sha256", "bytes"}, ...] in path order}.

    The same tree gives the same result: nothing but the file bytes and the paths goes into it.
    """
    root = Path(root)
    groups = {}
    for group, paths in frozen_files(root).items():
        files = []
        for path in paths:
            raw = (root / path).read_bytes()
            files.append({"path": path, "sha256": sha256_of_bytes(raw), "bytes": len(raw)})
        groups[group] = {"sha256": sha256_of_lines([(file["sha256"], file["path"]) for file in files]),
                         "files": files}
    return groups


def _code_sha256_of(groups: dict[str, dict]) -> str:
    """One hash over the group hashes of CODE_GROUPS, in that order."""
    return sha256_of_lines([(groups[group]["sha256"], group) for group in CODE_GROUPS])


def code_sha256(root: str | Path = ROOT) -> str:
    """One hash over the code that makes and scores an episode and computes a number: the groups grader,
    environment, harness, evaluation and config of the tree as it is now. No manifest is needed."""
    return _code_sha256_of(hash_groups(root))


# ---------------------------------------------------------------------------
# What the manifest says beside the file hashes
# ---------------------------------------------------------------------------
def _git(root: Path, *args: str) -> str | None:
    """The output of one read-only git command in root. None if git is not there or the command fails."""
    command = ["git", "--no-optional-locks", "-C", str(root), *args]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def git_state(root: str | Path, paths: list[str] | None = None) -> dict:
    """What git says about the tree, from read-only commands:

        commit                    the commit that HEAD names
        dirty                     True if `git status` lists any changed or untracked file
        uncommitted_frozen_files  the paths of `paths` (the frozen files) that git lists as changed or
                                  untracked: the commit does not hold them as they are now. None if no
                                  path is given.

    All three are None if git is not installed or root is not the top folder of a git work tree.
    """
    root = Path(root)
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top.strip()).resolve() != root.resolve():
        return {"commit": None, "dirty": None, "uncommitted_frozen_files": None}
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    listed = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths) if paths else None
    uncommitted = None if listed is None else sorted(line[3:].split(" -> ")[-1].strip('"')
                                                     for line in listed.splitlines() if line.strip())
    return {"commit": commit.strip() if commit else None,
            "dirty": None if status is None else bool(status.strip()),
            "uncommitted_frozen_files": uncommitted}


def _read_json(path: Path) -> dict | None:
    """The JSON object of a file. None if the file is missing, is not JSON or is not an object."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def data_summary(root: str | Path) -> dict:
    """What data/scenarios/summary.json says about the build of the final scenarios: the two data seeds,
    the design switches, the families per split, and the hashes of the text file and of the config that
    the build used. A value is None where the file or the key is missing."""
    summary = _read_json(Path(root) / SUMMARY_FILE) or {}
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    splits = summary.get("splits") if isinstance(summary.get("splits"), dict) else {}
    return {
        "summary_file": SUMMARY_FILE,
        "seeds": {name: config.get(name) for name in SEED_FIELDS},
        "design": {name: config.get(name) for name in DESIGN_FIELDS},
        "families": {split: counts.get("families") for split, counts in splits.items()
                     if isinstance(counts, dict)},
        "text_sha256": summary.get("text_sha256"),
        "config_sha256": summary.get("config_sha256"),
    }


def _hashes_of(root: Path, groups: dict[str, dict]) -> dict:
    """The hashes that a run or a write-up names: of the code, of the text file, and of RunConfig as the
    config module gives it (with the values, so that the manifest can be read without the code)."""
    text_file = root / TEXT_FILE
    config = RunConfig()
    return {"code_sha256": _code_sha256_of(groups),
            "text_sha256": sha256_of_file(text_file) if text_file.is_file() else None,
            "run_config_sha256": config.sha256(),
            "run_config": config.to_dict()}


def build_manifest(root: str | Path, *, signed_by: str, frozen_at: str, note: str = "") -> dict:
    """The manifest of the tree as it is now. Nothing is written.

    The same tree gives the same manifest, except for `frozen_at`. It holds:
        schema, frozen_at, signed_by, note
        git                  see git_state (all None without git)
        hash_rule            how the hashes are made, in words
        code_sha256          one hash over the groups of CODE_GROUPS
        text_sha256          the hash of text/model_text.yaml (None if it is missing)
        run_config_sha256, run_config    RunConfig().sha256() and RunConfig().to_dict() of the imported
                             config module
        data                 see data_summary
        groups               see hash_groups
        amendments           an empty list

    This function does not ask whether the tree may be frozen: see problems_before_freeze.
    Raises ValueError if `signed_by` is empty.
    """
    if not signed_by.strip():
        raise ValueError("a manifest needs the name of the person who signs it")
    root = Path(root)
    groups = hash_groups(root)
    paths = [file["path"] for group in groups.values() for file in group["files"]]
    return {"schema": SCHEMA, "frozen_at": frozen_at, "signed_by": signed_by.strip(), "note": note,
            "git": git_state(root, paths), "hash_rule": HASH_RULE, **_hashes_of(root, groups),
            "data": data_summary(root), "groups": groups, "amendments": []}


# ---------------------------------------------------------------------------
# Why the tree must not be frozen yet
# ---------------------------------------------------------------------------
def problems_before_freeze(root: str | Path, *, signed_by: str | None = None) -> list[str]:
    """The reasons why the tree must not be frozen yet, as plain sentences. Empty: it may be frozen.

    - MANIFEST.json exists already.
    - `signed_by` is given and is empty.
    - Everything that tree_problems finds.
    """
    problems = []
    if (Path(root) / MANIFEST_NAME).exists():
        problems.append(f"{MANIFEST_NAME} exists already: the tree was frozen before. A change after the "
                        "freeze is recorded with `amend`")
    if signed_by is not None and not signed_by.strip():
        problems.append("the name of the person who signs the freeze is empty")
    return problems + tree_problems(root)


def tree_problems(root: str | Path) -> list[str]:
    """Why the tree is not a state that a manifest may describe. It is asked before the freeze and before
    every amendment.

    - A required file is missing or empty, or a group has no file.
    - The text file is not valid, is a placeholder, or no person has put a name after
      "Read and approved by:" in its header.
    - data/scenarios does not fit the rest of the tree (see _scenario_problems).
    - data/scenarios fits the rest of the tree, and still it does not hold the families and the sets that
      the code gives now (see _built_again_problems).
    - A train, dev or test family of data/scenarios was already used in a folder data/scenarios_*.
    """
    root = Path(root)
    scenario_problems = _scenario_problems(root)
    return (_missing_or_empty(root) + _text_problems(root) + scenario_problems
            + ([] if scenario_problems else _built_again_problems(root)) + _used_family_problems(root))


def _missing_or_empty(root: Path) -> list[str]:
    """A required file that is missing or holds nothing but white space, and a group without any file.
    An empty __init__.py is not a problem: that is an ordinary Python file."""
    problems = []
    groups_with_a_missing_file = set()
    for path, group in required_files().items():
        file = root / path
        if not file.is_file():
            problems.append(f"{path} is missing (group {group})")
            groups_with_a_missing_file.add(group)
        elif file.name != "__init__.py" and not file.read_bytes().strip():
            problems.append(f"{path} is empty (group {group})")
    for group, paths in frozen_files(root).items():
        if not paths and group not in groups_with_a_missing_file:
            problems.append(f"the group {group} has no file")
    return problems


def header_lines(raw: bytes) -> list[str]:
    """The header of a text file: its lines from the top, up to the first line that is neither a comment
    (it starts with "#") nor blank. Line ends are not part of a line."""
    lines = []
    for line in raw.decode("utf-8", errors="replace").split("\n"):
        line = line.rstrip("\r")
        if line.strip() and not line.startswith("#"):
            break
        lines.append(line)
    return lines


def approval_values(raw: bytes) -> list[str]:
    """What stands after "# Read and approved by:" in each header line that starts with these words,
    without the white space around it. One entry per such line."""
    return [line[len(APPROVAL_PREFIX):].strip() for line in header_lines(raw)
            if line.startswith(APPROVAL_PREFIX)]


def approval_problem(raw: bytes) -> str | None:
    """Why the header does not say that a person has read and approved the text, or None if it says so.

    The header must hold a line that starts with "# Read and approved by:" exactly once, and that line
    must be the line that scripts/approve_text.py writes: "# Read and approved by: NAME, YYYY-MM-DD (UTC)"
    with a date that exists. "(not yet)" in the place of the name means that nobody has approved, and so
    does an empty name. A word without a date ("TODO", "-", "?") is not an approval.
    This cannot tell a name from another word in front of a date. It asks that the line was written on
    purpose, in the one form of the script."""
    values = approval_values(raw)
    if len(values) != 1:
        return (f"the header of {TEXT_FILE} must hold the line '{APPROVAL_PREFIX} NAME' exactly once, "
                f"not {len(values)} times")
    if values[0] == NOT_APPROVED:
        return (f"the header of {TEXT_FILE} still says '{APPROVAL_PREFIX} {NOT_APPROVED}': no person has "
                "approved the text (scripts/approve_text.py)")
    if not values[0].split(",")[0].strip():
        return f"the name after '{APPROVAL_PREFIX}' in the header of {TEXT_FILE} is empty"
    match = APPROVED_FORM.fullmatch(values[0])
    if match is None or NOT_APPROVED in match["name"] or not _is_a_day(match["day"]):
        return (f"the header line '{APPROVAL_PREFIX} {values[0]}' of {TEXT_FILE} is not an approval: it "
                f"must read '{APPROVAL_PREFIX} NAME, YYYY-MM-DD (UTC)', as scripts/approve_text.py writes it")
    return None


def _is_a_day(text: str) -> bool:
    """True if the text is a date that exists, in ISO form: 2026-11-18."""
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _text_problems(root: Path) -> list[str]:
    """The text file must be valid, must not be a placeholder (accessdesk.text: meta.placeholder), and
    its header must name the person who approved it. A missing file is reported by _missing_or_empty."""
    path = root / TEXT_FILE
    if not path.is_file():
        return []
    problems = []
    invalid = check_text(path)
    if invalid:
        problems.append(f"{TEXT_FILE} is not a valid text file: {len(invalid)} problem(s); the first: "
                        f"{invalid[0]}")
    elif load_text(path).is_placeholder:
        problems.append(f"{TEXT_FILE} is a placeholder (meta.placeholder is true)")
    approval = approval_problem(path.read_bytes())
    return problems + ([approval] if approval else [])


def _scenario_problems(root: Path) -> list[str]:
    """data/scenarios must be the folder that the build wrote from this text file, these name files and
    this config. The meaning is that of scripts/gpu/rollout.py (check_scenarios) and of
    scripts/run_scripted.py (scenarios_problem), asked for every file at once:

    - summary.json names the hash of text/model_text.yaml, and does not say "placeholder";
    - every file that summary.json lists (files_sha256) is there with that hash, and the folder holds no
      other file;
    - summary.json names the hashes of the two name files in data/names;
    - summary.json holds the config that the config module gives now: the same hash and the same values
      (scripts/run_scripted.py refuses a folder of another config, so the final data must be built from
      accessdesk/config.py as it is, without build options);
    - no world seed is used by two families of the folder.
    A missing summary.json is reported by _missing_or_empty.
    """
    folder = root / FINAL_DIR
    if not (root / SUMMARY_FILE).is_file():
        return []
    summary = _read_json(root / SUMMARY_FILE)
    if summary is None:
        return [f"{SUMMARY_FILE} cannot be read as a JSON object"]
    problems = []
    text_file = root / TEXT_FILE
    if text_file.is_file() and summary.get("text_sha256") != sha256_of_file(text_file):
        problems.append(f"{FINAL_DIR} was built from another text file: {SUMMARY_FILE} names another "
                        f"text_sha256 than the hash of {TEXT_FILE}")
    if summary.get("text_is_placeholder") is not False:
        problems.append(f"{SUMMARY_FILE} does not say that the text of the build was not a placeholder")
    problems += _listed_file_problems(folder, summary.get("files_sha256"))
    problems += _name_file_problems(root, summary.get("name_files_sha256"))
    config = RunConfig()
    if summary.get("config_sha256") != config.sha256() or summary.get("config") != config.to_dict():
        problems.append(f"{FINAL_DIR} was built with another config: {SUMMARY_FILE} does not hold the "
                        "config (hash and values) that accessdesk/config.py gives now")
    return problems + _seed_used_twice(folder)


def _listed_file_problems(folder: Path, listed: object) -> list[str]:
    """Each file that summary.json lists must be in the folder with the listed hash, and the folder must
    hold no file that summary.json does not list (summary.json itself is the one exception)."""
    if not isinstance(listed, dict) or not listed:
        return [f"{SUMMARY_FILE} lists no file hash (files_sha256)"]
    problems = []
    for name, digest in listed.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{FINAL_DIR}/{name} is missing ({SUMMARY_FILE} lists it)")
        elif sha256_of_file(path) != digest:
            problems.append(f"{FINAL_DIR}/{name} is not the file that the build wrote ({SUMMARY_FILE} "
                            "names another SHA-256)")
    known = set(listed) | {Path(SUMMARY_FILE).name}
    for path in sorted(folder.rglob("*")):
        name = path.relative_to(folder).as_posix()
        if path.is_file() and name not in known and not _is_compiled(name):
            problems.append(f"{FINAL_DIR}/{name} is not a file of the build ({SUMMARY_FILE} does not "
                            "list it)")
    return problems


def _name_file_problems(root: Path, listed: object) -> list[str]:
    """The build must have read the name files that are in data/names now. A hash of None in summary.json
    means that the build used made-up names in place of a missing file."""
    if not isinstance(listed, dict) or not listed:
        return [f"{SUMMARY_FILE} names no name file (name_files_sha256)"]
    problems = []
    for name, digest in listed.items():
        path = root / NAMES_DIR / name
        if digest is None:
            problems.append(f"{FINAL_DIR} was built without the name file {name} (made-up names)")
        elif path.is_file() and sha256_of_file(path) != digest:
            problems.append(f"{FINAL_DIR} was built from another {NAMES_DIR}/{name} ({SUMMARY_FILE} names "
                            "another SHA-256)")
    return problems


def _built_again_problems(root: Path) -> list[str]:
    """data/scenarios must hold what the code gives now, not only what an earlier build wrote.

    The folder is tied to its summary by file hashes. But the summary is a file of the build itself. After
    a change of the generator (accessdesk/env/generator.py), of the rules (accessdesk/env/rules.py) or of
    the sets (accessdesk/env/scenarios.py) an old folder still fits its summary, and its saved expected
    states are no longer the states that the frozen grader gives. Every pass graded against them is wrong.

    So the families and the sets are made again in memory, the way scripts/build_scenarios.py makes them
    (`build`; a test holds the two together), from the text file, the name files and the config of the
    tree. They are compared with the files, value by value. One line for each file that differs. Nothing
    is written.

    tree_problems asks this only when _scenario_problems finds nothing: the folder then fits this text,
    these names and this config, so a difference comes from the code. Nothing is found here if the text
    file is not valid or a name file is missing: _text_problems and _missing_or_empty report those.
    """
    folder, text_file = root / FINAL_DIR, root / TEXT_FILE
    needed = [root / SUMMARY_FILE, text_file] + [root / NAMES_DIR / name for name in NAME_FILES]
    if not all(path.is_file() for path in needed) or check_text(text_file):
        return []
    text, cfg = load_text(text_file), RunConfig()
    try:
        made = make_all_splits(text, load_name_pools(root / NAMES_DIR), cfg)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return [f"{FINAL_DIR} cannot be built again from this tree: {type(error).__name__} {error}"]
    found = {f"{split}.jsonl": _split_file_difference(folder / f"{split}.jsonl", _as_json(made[split]))
             for split in SPLITS}
    try:
        sets = _as_json(make_sets(made["dev"], cfg, text.policy_shift_variant))
        found[SETS_NAME] = _sets_file_difference(folder / SETS_NAME, sets)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        found[SETS_NAME] = f"the sets cannot be made again: {type(error).__name__} {error}"
    return [f"{FINAL_DIR}/{name} is not what the build gives now from {TEXT_FILE}, {NAMES_DIR} and "
            f"accessdesk/config.py: {difference}. The folder was built by other code: build it again "
            "(scripts/build_scenarios.py)"
            for name, difference in found.items() if difference]


def _as_json(value: object) -> object:
    """The value as it is after it was written as JSON and read again (a tuple is then a list)."""
    return json.loads(json.dumps(value))


def _split_file_difference(path: Path, made: list[dict]) -> str | None:
    """How the families of a split file differ from the families that were made again, or None."""
    try:
        saved = load_families(path)
    except (OSError, ValueError):
        return "the file is missing or cannot be read as family records"
    if len(saved) != len(made):
        return f"the file holds {len(saved)} families, the build gives {len(made)}"
    pairs = [(old, new) for old, new in zip(saved, made) if old != new]
    if not pairs:
        return None
    old, new = pairs[0]
    keys = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
    return (f"{len(pairs)} of the {len(saved)} families differ; the first is {old.get('family_id')}, in: "
            f"{', '.join(keys)}")


def _sets_file_difference(path: Path, made: dict) -> str | None:
    """How the sets file differs from the sets that were made again, or None."""
    saved = _read_json(path)
    if saved is None:
        return "the file is missing or cannot be read as a JSON object"
    keys = sorted(key for key in set(saved) | set(made) if saved.get(key) != made.get(key))
    return f"it differs in: {', '.join(keys)}" if keys else None


def _families_of(folder: Path, splits: tuple[str, ...] | None = None) -> list[dict]:
    """The family records of a folder: of the named splits that exist, or of every *.jsonl file."""
    paths = sorted(folder.glob("*.jsonl")) if splits is None else [folder / f"{split}.jsonl"
                                                                   for split in splits]
    return [family for path in paths if path.is_file() for family in load_families(path)]


def _seed_used_twice(folder: Path) -> list[str]:
    """PLAN 5.3: world seeds do not cross splits. So no world seed may be in the folder twice."""
    try:
        seeds = [family["world_seed"] for family in _families_of(folder)]
    except (ValueError, KeyError, TypeError):
        return [f"the split files of {FINAL_DIR} cannot be read as family records"]
    twice = len(seeds) - len(set(seeds))
    return [f"{twice} world seed(s) are used by more than one family of {FINAL_DIR}"] if twice else []


def _content(family: dict) -> str:
    """What makes a family the same family, whatever its id and its seed number say: its world and its
    request, as one string."""
    return json.dumps({"world": family["world"], "request": family["request"]}, sort_keys=True)


def _used_family_problems(root: Path) -> list[str]:
    """PLAN 5.3: no calibration family is in train, dev or test. One line for each folder data/scenarios_*
    (the calibration and placeholder data, which were read and rolled out during the work) that holds a
    family of the train, dev or test split of data/scenarios.

    Two families are the same if they have the same world seed (the check of scripts/build_scenarios.py,
    reused_world_seeds) or the same world and request. The family id is not compared: the generator
    numbers the families of every build in the same way (dv-00, dv-01, ...), so every build has the same
    ids. The calib split of data/scenarios is not looked at, as in the build: its families are for
    calibration only, and no result uses them.
    A family that cannot be compared is a problem too: the answer is never "nothing found" for a family
    that was not looked at.
    """
    try:
        new = _families_of(root / FINAL_DIR, FRESH_SPLITS)
        new_seeds = {family["world_seed"] for family in new}
        new_contents = {_content(family) for family in new}
    except (ValueError, KeyError, TypeError):
        return [f"the train, dev and test families of {FINAL_DIR} cannot be compared with the folders "
                f"{EARLIER_DIRS_PATTERN}: a split file cannot be read, or a family has no world seed, no "
                "world or no request"]
    problems = []
    for folder in sorted(path for path in root.glob(EARLIER_DIRS_PATTERN) if path.is_dir()):
        name = folder.relative_to(root).as_posix()
        try:
            old = _families_of(folder)
            seeds = new_seeds & {family["world_seed"] for family in old}
            contents = new_contents & {_content(family) for family in old}
        except (ValueError, KeyError, TypeError):
            problems.append(f"the split files of {name} cannot be read as family records")
            continue
        if seeds:
            problems.append(f"{len(seeds)} of the {len(new_seeds)} world seeds of the train, dev and test "
                            f"families of {FINAL_DIR} are in {name}")
        if contents:
            problems.append(f"{len(contents)} of the {len(new)} train, dev and test families of "
                            f"{FINAL_DIR} have the world and the request of a family in {name}")
    return problems


# ---------------------------------------------------------------------------
# The tree against a manifest
# ---------------------------------------------------------------------------
def _groups_of(manifest: dict) -> dict[str, dict]:
    """The groups of a manifest. Raises ValueError if the dict is not a manifest of this schema."""
    groups = manifest.get("groups") if isinstance(manifest, dict) else None
    if not isinstance(groups, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"not a manifest of schema {SCHEMA}")
    return groups


def file_changes(manifest: dict, groups_now: dict[str, dict]) -> list[dict]:
    """One entry for each file whose hash in `groups_now` is not its hash in the manifest:

        {"path", "group", "kind", "old_sha256", "new_sha256"}

    kind "changed": other bytes. "missing": in the manifest, not in the tree (new_sha256 is None).
    "new": in the tree under a frozen pattern, not in the manifest (old_sha256 is None).
    The entries are in the order of the groups, and in path order inside a group.
    """
    then = _groups_of(manifest)
    changes = []
    for group in list(then) + [group for group in groups_now if group not in then]:
        old = {file["path"]: file["sha256"] for file in then.get(group, {}).get("files", [])}
        new = {file["path"]: file["sha256"] for file in groups_now.get(group, {}).get("files", [])}
        for path in sorted(set(old) | set(new)):
            if old.get(path) == new.get(path):
                continue
            kind = "missing" if path not in new else "new" if path not in old else "changed"
            changes.append({"path": path, "group": group, "kind": kind,
                            "old_sha256": old.get(path), "new_sha256": new.get(path)})
    return changes


def _short(digest: str | None) -> str:
    """The first digits of a hash, as a message shows it."""
    return "none" if digest is None else digest[:_SHOWN]


def _change_line(change: dict) -> str:
    """One file change as a sentence."""
    where = f"{change['path']} (group {change['group']})"
    if change["kind"] == "missing":
        return f"missing: {where}; the manifest has {_short(change['old_sha256'])}"
    if change["kind"] == "new":
        return f"new: {where} matches a frozen pattern and is not in the manifest"
    return (f"changed: {where}; the manifest has {_short(change['old_sha256'])}, the file has "
            f"{_short(change['new_sha256'])}")


def check_manifest(root: str | Path, manifest: dict) -> list[str]:
    """Every difference between the tree and the manifest, as plain sentences. Empty: nothing changed.

    - A frozen file with other bytes, a frozen file that is missing, and a new file that matches a
      frozen pattern.
    - A group that only the manifest or only FROZEN_GROUPS has.
    - A hash of the manifest that is not the hash of what the manifest itself lists: a group hash,
      code_sha256 or text_sha256 (a manifest that was changed by hand).
    - Another RunConfig hash than the config module gives now, or other values under the same hash.
    - `data` (seeds, design switches, families per split) that is not what data/scenarios/summary.json
      says now.
    signed_by, frozen_at, note and git cannot be compared with anything in the tree. A run names the hash
    of the whole manifest file (frozen_state, manifest_sha256), so a later change of them shows there.
    Raises ValueError if `manifest` is not a manifest of this schema.
    """
    root = Path(root)
    then, now = _groups_of(manifest), hash_groups(root)
    changes = file_changes(manifest, now)
    differences = [_change_line(change) for change in changes]
    unsettled = {change["group"] for change in changes}       # groups whose hash must differ anyway
    for group in list(then) + [group for group in now if group not in then]:
        if group not in now or group not in then:
            where = "the manifest" if group in then else "FROZEN_GROUPS of accessdesk/freeze.py"
            differences.append(f"the group {group} is only in {where}")
            unsettled.add(group)
        elif group not in unsettled and then[group].get("sha256") != now[group]["sha256"]:
            differences.append(f"the hash of the group {group} in the manifest is not the hash of the "
                               "files that the manifest lists")
            unsettled.add(group)
    hashes = _hashes_of(root, now)
    if not unsettled & set(CODE_GROUPS) and manifest.get("code_sha256") != hashes["code_sha256"]:
        differences.append("code_sha256 in the manifest is not the hash of its code groups")
    if "text" not in unsettled and manifest.get("text_sha256") != hashes["text_sha256"]:
        differences.append(f"text_sha256 in the manifest is not the hash of {TEXT_FILE}")
    if manifest.get("run_config_sha256") != hashes["run_config_sha256"]:
        differences.append(f"the RunConfig hash changed; the manifest has "
                           f"{_short(manifest.get('run_config_sha256'))}, the config module gives "
                           f"{_short(hashes['run_config_sha256'])}")
    elif manifest.get("run_config") != hashes["run_config"]:
        differences.append("run_config in the manifest does not hold the values that the config module "
                           "gives (its hash is the same, so the values were changed by hand)")
    if "scenarios" not in unsettled and manifest.get("data") != data_summary(root):
        differences.append(f"`data` in the manifest (seeds, design switches, families per split) is not "
                           f"what {SUMMARY_FILE} says")
    return differences


def _kept_in_history(manifest: dict) -> dict:
    """What an amendment keeps of a manifest in the history: its hashes, and the values that the next
    manifest replaces (run_config, and data with the seeds and the design switches)."""
    return {"code_sha256": manifest.get("code_sha256"), "text_sha256": manifest.get("text_sha256"),
            "run_config_sha256": manifest.get("run_config_sha256"),
            "run_config": copy.deepcopy(manifest.get("run_config")),
            "data": copy.deepcopy(manifest.get("data")),
            "group_sha256": {group: entry.get("sha256") for group, entry in _groups_of(manifest).items()}}


def amend_manifest(root: str | Path, manifest: dict, *, reason: str, amended_at: str) -> dict:
    """A new manifest for the tree as it is now, with one more entry at the end of "amendments".

    schema, frozen_at, signed_by, note, git and the earlier amendments stay as they are. The hashes, the
    groups, run_config and data are those of the tree now. The new entry holds:
        amended_at, reason
        git       see git_state, at the time of the amendment
        changes   see file_changes: for each file the old and the new hash
        before, after    code_sha256, text_sha256, run_config_sha256, every group hash, and the values
                  of run_config and of data (seeds, design switches), before and after the amendment
    So the first state stays readable: `before` of the first entry holds the hashes, the seeds and the
    config values of the freeze.

    `manifest` is not changed. Raises ValueError if the reason is empty, if nothing differs from the
    manifest, or if `manifest` is not a manifest of this schema.
    """
    if not reason.strip():
        raise ValueError("an amendment needs a reason")
    root = Path(root)
    groups = hash_groups(root)
    changes = file_changes(manifest, groups)
    amended = copy.deepcopy(manifest)
    amended.update(_hashes_of(root, groups))
    amended.update({"data": data_summary(root), "groups": groups})
    before, after = _kept_in_history(manifest), _kept_in_history(amended)
    if not changes and before == after:
        raise ValueError("nothing differs from the manifest: there is nothing to amend")
    paths = [file["path"] for group in groups.values() for file in group["files"]]
    entry = {"amended_at": amended_at, "reason": reason.strip(), "git": git_state(root, paths),
             "changes": changes, "before": before, "after": after}
    amended["amendments"] = list(amended.get("amendments", [])) + [entry]
    return amended


# ---------------------------------------------------------------------------
# For run scripts
# ---------------------------------------------------------------------------
def frozen_state(root: str | Path = ROOT) -> dict:
    """What a run script writes into its run manifest, so that the run names the code that made it:

        manifest_sha256   the SHA-256 of the bytes of MANIFEST.json; None if the file is not there
        code_sha256       code_sha256(root): the hash of the code as it is now
        matches           True only if MANIFEST.json is there and check_manifest finds no difference
        differences       the list of check_manifest, or the one reason why there is no such list
    """
    path = Path(root) / MANIFEST_NAME
    state = {"manifest_sha256": None, "code_sha256": code_sha256(root), "matches": False}
    if not path.is_file():
        return {**state, "differences": [f"{MANIFEST_NAME} is missing: the tree is not frozen"]}
    raw = path.read_bytes()
    state["manifest_sha256"] = sha256_of_bytes(raw)
    try:
        differences = check_manifest(root, json.loads(raw.decode("utf-8")))
    except ValueError as error:
        return {**state, "differences": [f"{MANIFEST_NAME} cannot be read: {error}"]}
    return {**state, "matches": not differences, "differences": differences}


def run_problem(state: dict, allow_changed_tree: bool = False) -> str | None:
    """Why a run must not start on this tree, or None if it may. `state` is what frozen_state returns.

    Before the freeze (there is no MANIFEST.json) every run may start. After the freeze a run may start
    only on a tree that is as the manifest says: a run on other code or data is another experiment
    (PLAN 5.3). With `allow_changed_tree` it may start all the same. The run scripts that ask this
    (scripts/gpu/rollout.py and scripts/run_scripted.py) have the option --allow-changed-tree for that,
    and write the option and `state` into their run manifest.
    The message lists every difference, one a line.
    """
    if state["manifest_sha256"] is None or state["matches"] or allow_changed_tree:
        return None
    lines = "".join(f"\n- {line}" for line in state["differences"])
    return (f"the tree is frozen, and it differs from {MANIFEST_NAME}:{lines}\n"
            "Record a wanted change with scripts/freeze.py amend. To run all the same, pass "
            "--allow-changed-tree: the run manifest then says that the tree was not as frozen.")
