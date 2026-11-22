"""Tables about failed attempts, flips, flagged passes and the LoRA adapter. The script only reads saved runs.

    .venv/bin/python scripts/failure_report.py

With no argument it reads the dev runs of the four arms under results/main (base_dev, sft_dev,
grpo_seed1_dev, grpo_seed2_dev), the SFT run (results/main/sft) and the two GRPO runs, and it writes into
results/main/report/failures/:

    kinds.md              the failed attempts of each arm by expected outcome and by kind of failure, and what
                          a lenient parser can and cannot change
    flips.md, flips.csv   the passes of 8 of every scenario in each arm; the scenarios and the families that
                          changed between two arms
    flagged.md            the passing attempts that each monitor flags, and the two stricter pass rules
    lora.md               which modules the LoRA adapter wraps, and how far training moved each of them
    failure_report.json   every number of these files

No model and no GPU are used, no attempt is run again by a model, and no saved file is changed. torch is
never imported: the adapter files are read with safetensors and numpy.

--results DIR        the folder of the saved runs (default: results/main)
--split NAME         the split of the attempts (default: dev). The run folder of an arm is
                     <results>/<arm>_<split>. An arm whose folder is not there is left out, and the files
                     say so. The scenario file of the split is opened only if a run folder of the split
                     is there. So `--split test` reads nothing while no test run is saved.
--arm NAME=FOLDER    an arm and its run folder (it holds episodes.jsonl). Give it once per arm. Default: the
                     four arms base, sft, grpo_seed1, grpo_seed2.
--pair A:B           a change "from arm A to arm B" of flips.md. Default: base:sft, sft:grpo_seed1,
                     sft:grpo_seed2. A pair with an arm that is not there is left out.
--scenarios DIR      the scenario folder of the runs (default: data/scenarios)
--text FILE          the text file of the runs (default: text/model_text.yaml)
--monitors DIR       the saved monitor reports <arm>_<split>.md, to compare with (default:
                     <results>/monitors)
--sft-run DIR        the SFT run: run_manifest.json, lora_coverage.json, adapter/ (default: <results>/sft)
--grpo-run NAME=DIR  a GRPO run: run_manifest.json and its final adapter (default: grpo_seed1 and grpo_seed2
                     under <results>)
--parts NAMES        which tables to make: kinds, flips, flagged, lora (default: all four with the split
                     dev; with another split the first three, because the adapter table has no split)
--out DIR            the output folder (default: <results>/report/failures for the split dev, and
                     <results>/report/failures/<split> for another split)

Where the rules are. Each number is made from saved files by a rule that the output file states:
  - kinds: `failure_kind`, `record_fault`, `right_for_the_person_picked`, `lenient_reading`
  - flips: `scenario_flip`, `family_change`
  - flagged: `summarise` of accessdesk/eval/monitors.py (frozen); nothing is defined here
  - lora: `fold_coverage`, `update_square`, `difference_square`, and `adapter_numbers` for a zero update
The metrics of ANALYSIS_PLAN.md section 5 come from the frozen functions (accessdesk/eval/metrics.py and
monitors.py): the pass rate, the family pass rates, the labels of an attempt, the family-weighted shares.
One statement of lora.md does not come from a saved run: why the short target list is q_proj and v_proj. It
names its published source (SHORT_LIST_SOURCE, with what was read and when), and it holds no number.

Exit codes: 0 the files were written. 1 the files were written, and a check failed: the file that holds the
check says which one, and so does stderr. 2 nothing was written (stderr says why: a file is not the file of
the run, the arms do not cover the same attempts, or no input is there).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.audit import Run, load_run, load_script  # noqa: E402
from accessdesk.env import rules  # noqa: E402
from accessdesk.eval import metrics, monitors, tables  # noqa: E402
from accessdesk.harness.episode import Episode  # noqa: E402
from accessdesk.harness.parser import parse_assistant_message  # noqa: E402
from accessdesk.text import TextBundle, model_text_path, normalise  # noqa: E402
from accessdesk.types import OUTCOMES, PARSE_ERRORS, RESOLUTIONS  # noqa: E402

DEFAULT_SPLIT = "dev"
DEFAULT_ARMS = ("base", "sft", "grpo_seed1", "grpo_seed2")
DEFAULT_PAIRS = (("base", "sft"), ("sft", "grpo_seed1"), ("sft", "grpo_seed2"))
DEFAULT_GRPO_RUNS = ("grpo_seed1", "grpo_seed2")
PARTS = ("kinds", "flips", "flagged", "lora")
EPISODE_PARTS = ("kinds", "flips", "flagged")       # the parts that read attempts
EPISODES_FILE, MANIFEST_FILE = "episodes.jsonl", "run_manifest.json"
JSON_NAME = "failure_report.json"

# ---------------------------------------------------------------------------
# The kinds of failure. Each failed attempt has exactly one (see `failure_kind`).
# ---------------------------------------------------------------------------
NO_CLOSE_KINDS = ("no_valid_call", "no_close_some_malformed", "no_close_all_calls")
CLOSED_KINDS = ("wrong_person_same_name", "other_person_or_role", "wrong_decision",
                "right_resolution_wrong_records", "other")
KINDS = NO_CLOSE_KINDS + CLOSED_KINDS
KIND_LABELS = {
    "no_valid_call": "A1 no valid call",
    "no_close_some_malformed": "A2 no close, some messages malformed",
    "no_close_all_calls": "A3 no close, every message a well-formed call",
    "wrong_person_same_name": "B the wrong person of two with one name",
    "other_person_or_role": "C another person or another role",
    "wrong_decision": "D a wrong decision for the right person",
    "right_resolution_wrong_records": "E the right resolution with a wrong or missing record",
    "other": "F other",
}
KIND_RULES = {
    "no_valid_call": "The request was never closed (`ended_by` is `turn_limit`, and `final_state.request."
                     "resolution` is null). Every message is malformed (`n_malformed` = `n_turns`).",
    "no_close_some_malformed": "The request was never closed. At least one message is malformed, and at "
                               "least one is a well-formed call (0 < `n_malformed` < `n_turns`).",
    "no_close_all_calls": "The request was never closed. No message is malformed (`n_malformed` = 0): "
                          "every message was a well-formed call, and none of them closed the request.",
    "wrong_person_same_name": "The request was closed. The person of the closed request (`final_state."
                              "request.person_id`) is not the person meant, and has the same full name "
                              "(monitor m6 `wrong_person_same_name` of accessdesk/eval/monitors.py).",
    "other_person_or_role": "The request was closed, and B does not apply. The person id or the role id of "
                            "the closed request is not the expected one.",
    "wrong_decision": "The request was closed for the expected person and role. The resolution of the "
                      "closed request is not the expected one.",
    "right_resolution_wrong_records": "The request was closed for the expected person and role with the "
                                      "expected resolution. The approval records or the grant records of "
                                      "`final_state` are not the expected ones.",
    "other": "None of the rules above applies.",
}
# The label of accessdesk/eval/metrics.py (`episode_label`, frozen) that each kind must have. Every kind of
# a closed request has the label "closed_wrongly".
LABEL_OF_NO_CLOSE_KIND = {"no_valid_call": "malformed_only", "no_close_some_malformed": "never_closed",
                          "no_close_all_calls": "never_closed"}
# What is wrong with the records of an attempt of the kind "right_resolution_wrong_records".
RECORD_FAULTS = ("approval_wrong_approver", "record_missing", "record_extra")
RECORD_FAULT_LABELS = {
    "approval_wrong_approver": "an approval request for the wrong approver",
    "record_missing": "an expected record is missing",
    "record_extra": "a record too many",
}
RECORD_FAULT_RULES = {
    "approval_wrong_approver": "The expected approval record is not in `final_state.approvals`, and that "
                               "list holds an approval record for the same person and role with another "
                               "approver.",
    "record_missing": "The rule above does not apply, and an expected grant record or approval record is "
                      "not in `final_state` (for example: closed as granted with no grant record).",
    "record_extra": "No expected record is missing, and `final_state` holds a grant record or an approval "
                    "record that is not expected (for example: a role granted twice).",
}
# What the second scoring with the lenient parser can say about one attempt (see `lenient_reading`).
SECOND = ("same", "ends_pass", "ends_fail", "open")
# The fields of a turn that the model reads next, or that say what the harness did. If the lenient harness
# gives the same values as the saved turn, the rest of the saved conversation is still the model's.
SAME_TURN_FIELDS = ("parse_ok", "tool_key", "args", "tool_error", "tool_response")

# ---------------------------------------------------------------------------
# Flips
# ---------------------------------------------------------------------------
ATTEMPTS_PER_SCENARIO = 8       # ANALYSIS_PLAN.md section 2: 8 attempts per scenario
MOSTLY_FAILING_MAX = 3          # "mostly failing": 3 or fewer of the 8 attempts passed
MOSTLY_PASSING_MIN = 5          # "mostly passing": 5 or more of the 8 attempts passed
FAMILY_POINTS = 10              # a family changed if its pass rate moved by MORE than 10 points
FLIP_CLASSES = ("fail_to_pass", "pass_to_fail", "stayed_passing", "stayed_failing", "half",
                "not_classified")
UNCHANGED_CLASSES = ("stayed_passing", "stayed_failing", "half")
FAMILY_CLASSES = ("better", "worse", "within")

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
ADAPTER_FILE, ADAPTER_CONFIG, COVERAGE_FILE = "adapter_model.safetensors", "adapter_config.json", \
    "lora_coverage.json"
ADAPTER_KEY = re.compile(r"base_model\.model\.(?P<path>.+)\.lora_(?P<part>[AB])\.weight")
SHORT_LIST = ("q_proj", "v_proj")            # the short target list that the coverage table compares with
# Why this short list: a published table, not a saved run and not a judgement. The entries were read on
# 2026-11-19 in the files of the wheel peft-0.21.3, the PEFT version that the run manifests name. In `file`
# (its SHA-256 is `file_sha256`) the dict `mapping` gives exactly ["q_proj", "v_proj"] for each model type
# of `model_types_with_the_short_list`, and it has no key for the model types of `models_without_an_entry`
# (model name -> `model_type` and `text_config.model_type` in the config.json of that model).
# `lookup_function` in `lookup_file` takes the entry by the exact model type, and it raises ValueError if
# the run gives no list and the table has no entry. tests/core/test_report_failures.py reads these files
# again where a copy of them is in .cache/. lora.md prints the words; no number comes from here.
SHORT_LIST_SOURCE = {
    "package": "peft", "version": "0.21.3", "file": "peft/utils/constants.py",
    "file_sha256": "b840578c9626740ad01e84c31f713f85aecc0c696bdde29e4e3891f23db1e9f5",
    "mapping": "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "model_types_with_the_short_list": ["qwen2", "qwen3", "llama", "mistral", "gemma"],
    "lookup_file": "peft/tuners/tuners_utils.py", "lookup_function": "_prepare_adapter_config",
    "models_without_an_entry": {"Qwen/Qwen3.5-2B": ["qwen3_5", "qwen3_5_text"]},
}
LAYER_TYPE_LABELS = {"full_attention": "full attention", "linear_attention": "linear attention",
                     None: "outside the layers"}
LAYER_TYPE_ORDER = ("full_attention", "linear_attention", None)
BLOCK_ORDER = ("self_attn", "linear_attn", "mlp", None)
MODULE_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a",
                "in_proj_b", "gate_proj", "up_proj", "down_proj", "lm_head")
# Why the update of a module is zero (`adapter_numbers`): the first reason that applies is given.
ZERO_REASONS = ("b_is_zero", "a_is_zero", "product_is_zero")
NORM_TOLERANCE = 1e-6           # relative; the manifest's norms were computed in float32 on the GPU
FINGERPRINT_TOLERANCE = 1e-9    # relative; two sums of the same 15.6 million numbers in another order


class Arm(NamedTuple):
    """One arm: its name, its run folder, its attempts with their families, its run manifest, and the
    summary of the frozen monitors (`monitors.summarise`). Made by `make_arm`."""
    name: str
    folder: Path
    run: Run
    manifest: dict | None
    summary: dict


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    """The SHA-256 of the file's bytes, read in pieces of 1 MiB (an adapter file has about 60 MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def shown(path: Path) -> str:
    """The path as the files show it: from the repository root, with "/", if it is inside the repository."""
    return load_script("monitor_report").shown_path(Path(path))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    """0.7254 -> "72.5%"; None -> "n/a"."""
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def count_share(n: int, of: int) -> str:
    """"671 (72.5%)": a count and its share of `of`; the count alone if `of` is 0."""
    return f"{n} ({pct(n / of)})" if of else str(n)


def md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows] + [""]


def _multiset(records: list) -> Counter:
    """The records of one part of a state as a multiset: order does not matter, count does."""
    return Counter(tuple(record) for record in records)


def attempt_id(record: dict) -> str:
    """The name of one attempt inside one run: "dv-46/s3#0" is attempt 0 of the scenario dv-46/s3."""
    return monitors.episode_id(record)


# ---------------------------------------------------------------------------
# Reading the arms
# ---------------------------------------------------------------------------
def arm_folders(results: Path, split: str, named: list[tuple[str, Path]] | None) -> dict[str, Path]:
    """arm name -> run folder, in the order of the tables. Default: the four arms under `results`."""
    if not named:
        return {arm: results / f"{arm}_{split}" for arm in DEFAULT_ARMS}
    folders: dict[str, Path] = {}
    for name, folder in named:
        if name in folders:
            raise ValueError(f"--arm {name!r} is given twice")
        folders[name] = folder
    return folders


def make_arm(name: str, folder: Path, run: Run, manifest: dict | None = None) -> Arm:
    """An arm with the summary of the frozen monitors of its attempts.

    `monitors.summarise` also checks every valid record against its family record: the frozen verifier
    must give the saved pass or fail for the saved final state. So every table of this script is made
    from records whose verdict was checked. Raises ValueError where `summarise` does (no valid record, a
    record of another family file or text file, a saved verdict that the verifier does not give)."""
    summary = monitors.summarise(run.records, list(run.families.values()), run.text)
    return Arm(name, folder, run, manifest, summary)


def load_arms(folders: dict[str, Path], scenarios: Path, split: str,
              text_path: Path) -> tuple[list[Arm], list[dict], list[str]]:
    """(the arms that are present, the arms that are not, notes).

    An arm is present if its folder holds episodes.jsonl. Nothing of the scenario folder is opened while
    no arm is present. Each present arm is read with `accessdesk.audit.load_run`: the file holds one run
    and one set-up, the text file is the one that the records name, and the split file is the file that
    the build wrote, with the hash that the records name. `make_arm` then checks the verdict of every
    record. Last, the checks of the frozen table program are made for the arms together
    (accessdesk/eval/tables.py): the same text file, the same scenario file, the same settings, the same
    grader, and the same scenarios with the same number of attempts.
    Raises ValueError with the reason if one of them does not hold."""
    present, absent = [], []
    for name, folder in folders.items():
        if (folder / EPISODES_FILE).is_file():
            present.append((name, folder))
        else:
            absent.append({"arm": name, "folder": shown(folder)})
    arms = []
    for name, folder in present:
        run = load_run(folder / EPISODES_FILE, scenarios, split, text_path)
        manifest_file = folder / MANIFEST_FILE
        manifest = read_json(manifest_file) if manifest_file.is_file() else None
        arms.append(make_arm(name, folder, run, manifest))
    notes: list[str] = []
    if arms:
        records_by_arm = {arm.name: arm.run.records for arm in arms}
        tables.check_same_text(records_by_arm)
        tables.check_same_grader(records_by_arm)
        notes += tables.check_same_data(records_by_arm) + tables.check_same_settings(records_by_arm)
        tables.check_same_coverage(records_by_arm, arms[0].name)
        tables.check_arms_differ(records_by_arm)
    return arms, absent, notes


def arm_source(arm: Arm) -> dict:
    """What the files say about the input of one arm."""
    valid = metrics.valid_records(arm.run.records)
    manifest = arm.manifest or {}
    return {
        "arm": arm.name, "episodes": shown(arm.run.episodes), "episodes_sha256": arm.run.episodes_sha256,
        "n_records": len(arm.run.records), "n_harness_errors": metrics.harness_error_count(arm.run.records),
        "n_attempts": len(valid), "n_families": len({record["family_id"] for record in valid}),
        "n_scenarios": len({record["scenario_id"] for record in valid}),
        "run_seed": manifest.get("run_seed"), "attempts_per_scenario": manifest.get("samples"),
        "run_manifest": shown(arm.folder / MANIFEST_FILE) if arm.manifest is not None else None,
    }


# ---------------------------------------------------------------------------
# Kinds of failure
# ---------------------------------------------------------------------------
def is_closed(record: dict) -> bool:
    """True if the attempt closed the request: `final_state.request.resolution` is not null.

    The record says it in two more fields, `closed` and `ended_by`. Raises ValueError if they disagree:
    such a record is not what the harness writes (scripts/replay_check.py finds it too)."""
    closed = record["final_state"]["request"]["resolution"] is not None
    if record["closed"] is not closed or record["ended_by"] != ("close" if closed else "turn_limit"):
        raise ValueError(f"{attempt_id(record)}: closed is {record['closed']!r} and ended_by is "
                         f"{record['ended_by']!r}, but the final state says closed is {closed}")
    return closed


def failure_kind(record: dict, family: dict) -> str:
    """The kind of failure of one failed attempt: exactly one name of KINDS. The first rule that applies
    gives the kind.

    Read are the saved record (`final_state`, the calls in `turns`, `ended_by`, `n_malformed`) and the
    family record (the expected state of the record's rule set, and for kind B the names of the world).

    The request was never closed (no valid closing call):
        no_valid_call             every message is malformed
        no_close_some_malformed   at least one message is malformed and at least one is a well-formed call
        no_close_all_calls        no message is malformed
    The request was closed:
        wrong_person_same_name    closed for another person with the same full name as the person meant
                                  (`monitors.wrong_person_same_name`, frozen)
        other_person_or_role      not the kind before; the person id or the role id of the closed request
                                  is not the expected one
        wrong_decision            the expected person and role; the resolution is not the expected one
        right_resolution_wrong_records   the expected person, role and resolution; the approval records
                                  or the grant records are not the expected ones (compared as multisets,
                                  as the verifier does)
        other                     none of these. With these tools it cannot happen: the request and the
                                  records would be right, and the access list changes only with a grant
                                  record. The kind is kept, so that a count above 0 is seen.
    Raises ValueError for a passed attempt, and for a record whose counters are not those of its turns."""
    if record["passed"]:
        raise ValueError(f"{attempt_id(record)} passed: it has no kind of failure")
    n_malformed = sum(1 for turn in record["turns"] if not turn["parse_ok"])
    if n_malformed != record["n_malformed"] or len(record["turns"]) != record["n_turns"]:
        raise ValueError(f"{attempt_id(record)}: n_malformed or n_turns is not what the turns give")
    if not is_closed(record):
        if n_malformed == record["n_turns"]:
            return "no_valid_call"
        return "no_close_some_malformed" if n_malformed else "no_close_all_calls"
    expected, state = family["expected"][record["ruleset"]], record["final_state"]
    if monitors.wrong_person_same_name(record, family) is not None:
        return "wrong_person_same_name"
    if any(state["request"][key] != expected["request"][key] for key in ("person_id", "role_id")):
        return "other_person_or_role"
    if state["request"]["resolution"] != expected["request"]["resolution"]:
        return "wrong_decision"
    if any(_multiset(state[part]) != _multiset(expected[part]) for part in ("approvals", "grants")):
        return "right_resolution_wrong_records"
    return "other"


def record_fault(record: dict, family: dict) -> str:
    """What is wrong with the records of an attempt of the kind "right_resolution_wrong_records": exactly
    one name of RECORD_FAULTS. The first rule that applies gives it.

        approval_wrong_approver   the expected approval record is missing, and the final state holds an
                                  approval record for the same person and role with another approver
        record_missing            not the rule before; an expected grant or approval record is missing
        record_extra              no expected record is missing; the final state holds a grant or approval
                                  record that is not expected (the kind says that the records differ)"""
    expected, state = family["expected"][record["ruleset"]], record["final_state"]
    missing = {part: _multiset(expected[part]) - _multiset(state[part]) for part in ("approvals", "grants")}
    for person_id, role_id, approver_id in missing["approvals"]:
        if any((person, role) == (person_id, role_id) and approver != approver_id
               for person, role, approver in state["approvals"]):
            return "approval_wrong_approver"
    return "record_missing" if missing["approvals"] or missing["grants"] else "record_extra"


def right_for_the_person_picked(record: dict, family: dict) -> bool:
    """True if the end state of the attempt is the correct end state for the person that the attempt
    closed the request for (and the role that the request means).

    The correct end state for that person is `rules.expected_state` (frozen) of the family record in
    which the person picked stands in place of the person meant, with the picked person's own manager.
    The frozen verifier compares the saved final state with it. So True means: the attempt took the
    wrong person, and then did what the rules ask for that person."""
    picked = record["final_state"]["request"]["person_id"]
    people = {person["person_id"]: person for person in family["world"]["people"]}
    if picked not in people:
        return False
    stand_in = {**family, "truth": {"person_id": picked, "role_id": family["truth"]["role_id"],
                                    "manager_id": people[picked]["manager_id"]}}
    wanted = rules.expected_state(stand_in, record["ruleset"])
    return bool(impl.get("verify")(record["final_state"], wanted))


def kinds_of_arm(arm: Arm) -> dict:
    """The failed attempts of one arm by kind and by expected outcome, with the checks that the counts add
    up. Unit: the attempt. An attempt with a harness error is in no number.

    Three lists name families by their failed attempts (each entry: family id, failed attempts, valid
    attempts, expected outcome):
        families_with_most_failed       the three families with the most failed attempts, the smaller
                                        family id first where two counts are equal (Table 6 of kinds.md)
        families_tied_for_most_failed   every family that has the largest count: one family, or more where
                                        the largest count is shared
        families_tied_with_the_third    the families that are not among the three and have the count of
                                        the third one"""
    run = arm.run
    valid = metrics.valid_records(run.records)
    failed = [record for record in valid if not record["passed"]]
    kind_of = {}
    for record in failed:
        kind = failure_kind(record, run.families[record["family_id"]])
        wanted = LABEL_OF_NO_CLOSE_KIND.get(kind, "closed_wrongly")
        if metrics.episode_label(record) != wanted:       # a second, frozen definition must agree
            raise ValueError(f"{attempt_id(record)}: kind {kind}, but metrics.episode_label says "
                             f"{metrics.episode_label(record)}")
        kind_of[id(record)] = kind
    unknown = sorted({record["outcome"] for record in valid} - set(OUTCOMES))
    if unknown:
        raise ValueError(f"arm {arm.name}: unknown expected outcomes {unknown}")
    # The failed attempts that each frozen monitor flags, from the summary of the arm.
    flagged = {name: set(arm.summary["monitors"][name]["flagged_failed"]) for name in monitors.MONITORS}
    by_kind = {}
    for kind in KINDS:
        members = [record for record in failed if kind_of[id(record)] == kind]
        marks = [kind_of.get(id(record)) == kind for record in valid]
        by_kind[kind] = {
            "n": len(members),
            "share_of_failed": len(members) / len(failed) if failed else None,
            "share_of_all_family_weighted": monitors.family_weighted_share(valid, marks),
            "by_outcome": {outcome: sum(1 for record in members if record["outcome"] == outcome)
                           for outcome in OUTCOMES},
            "flagged_by": {name: sum(1 for record in members if attempt_id(record) in flagged[name])
                           for name in monitors.MONITORS},
            "attempts": sorted(attempt_id(record) for record in members),
        }
    n_passed = sum(1 for record in valid if record["passed"])
    sum_of_kinds = sum(row["n"] for row in by_kind.values())
    pass_rate = metrics.pass_rate(valid)
    weighted = sum(row["share_of_all_family_weighted"] for row in by_kind.values())
    if sum_of_kinds != len(failed) or len(failed) != len(valid) - n_passed \
            or not math.isclose(weighted, 1.0 - pass_rate, abs_tol=1e-9):
        raise RuntimeError(f"arm {arm.name}: the kinds do not add up to the failed attempts")
    if (len(valid), n_passed, pass_rate) != (arm.summary["n_episodes"], arm.summary["n_passed"],
                                             arm.summary["pass_rate"]):
        raise RuntimeError(f"arm {arm.name}: the attempts are not those of the summary of the monitors")

    decisions = Counter()
    faults = Counter()
    picked_right = 0
    for record in failed:
        family, kind = run.families[record["family_id"]], kind_of[id(record)]
        if kind == "wrong_decision":
            decisions[(family["expected"][record["ruleset"]]["request"]["resolution"],
                       record["final_state"]["request"]["resolution"])] += 1
        elif kind == "right_resolution_wrong_records":
            faults[record_fault(record, family)] += 1
        elif kind == "wrong_person_same_name":
            picked_right += right_for_the_person_picked(record, family)
    failed_per_family = Counter(record["family_id"] for record in failed)
    attempts_per_family = Counter(record["family_id"] for record in valid)
    # The families in falling order of their failed attempts, the smaller family id first where two counts
    # are equal. `top` is what Table 6 shows. Equal counts must stay visible, so two more lists are kept:
    # every family with the largest count (`tied_first`), and the families outside `top` that have the
    # count of its last family (`tied_third`).
    ranked = sorted(failed_per_family, key=lambda family_id: (-failed_per_family[family_id], family_id))
    top = ranked[:3]
    tied_first = [family_id for family_id in ranked
                  if failed_per_family[family_id] == failed_per_family[ranked[0]]]
    tied_third = [family_id for family_id in ranked[3:]
                  if failed_per_family[family_id] == failed_per_family[top[-1]]]

    def family_rows(family_ids: list[str]) -> list[dict]:
        return [{"family_id": family_id, "n_failed": failed_per_family[family_id],
                 "n_attempts": attempts_per_family[family_id],
                 "outcome": run.families[family_id]["outcome"]} for family_id in family_ids]

    return {
        "n_records": len(run.records), "n_harness_errors": len(run.records) - len(valid),
        "n_attempts": len(valid), "n_passed": n_passed, "n_failed": len(failed), "pass_rate": pass_rate,
        "turn_limits": sorted({(record.get("settings") or {}).get("turn_limit") for record in valid},
                              key=str),
        "attempts_per_family": sorted(set(attempts_per_family.values())),
        "attempts_by_outcome": {outcome: sum(1 for record in valid if record["outcome"] == outcome)
                                for outcome in OUTCOMES},
        "failed_by_outcome": {outcome: sum(1 for record in failed if record["outcome"] == outcome)
                              for outcome in OUTCOMES},
        "by_kind": by_kind, "sum_of_kinds": sum_of_kinds,
        "n_no_valid_closing_call": sum(by_kind[kind]["n"] for kind in NO_CLOSE_KINDS),
        "wrong_decisions": [{"expected": expected, "given": given, "n": decisions[(expected, given)]}
                            for expected in RESOLUTIONS for given in RESOLUTIONS
                            if decisions[(expected, given)]],
        "record_faults": {fault: faults[fault] for fault in RECORD_FAULTS},
        "wrong_person": {"n": by_kind["wrong_person_same_name"]["n"],
                         "right_for_the_person_picked": picked_right,
                         # What the frozen monitor m6 counts in this arm: it must be kind B, and never a pass.
                         "monitor_m6": {key: arm.summary["monitors"]["wrong_person_same_name"][key]
                                        for key in ("n", "n_passed")}},
        "families_with_most_failed": family_rows(top),
        "families_tied_for_most_failed": family_rows(tied_first),
        "families_tied_with_the_third": family_rows(tied_third),
        "n_families_with_a_failed_attempt": len(failed_per_family),
    }


# ---------------------------------------------------------------------------
# The second scoring with the lenient parser: the part that is exact
# ---------------------------------------------------------------------------
def lenient_reading(record: dict, family: dict, text: TextBundle) -> dict:
    """What the lenient parser changes in one saved attempt, as far as the saved text can tell.

    Two things are computed, both with frozen code and with nothing run again by a model.

    1. Every malformed message of the attempt is parsed again with `parse_assistant_message(strict=False)`,
       with the saved `truncated` flag, as the harness would call it with the lenient setting. `messages`
       holds one entry per malformed message: {"turn", "strict_error", "lenient_ok", "lenient_error",
       "lenient_tool"}. `lenient_ok` says that the lenient parser finds one well-formed call in it.

    2. The saved messages are given, in order, to a fresh episode of the harness with the lenient setting
       (`Episode` with `parser_strict` off; the config is otherwise the one of the record). While the
       lenient harness gives the saved tool answer, the saved next message is still the message of the
       model. At the first message that the lenient parser reads as a call and the strict parser did not,
       the lenient harness runs that call. `second` says what follows:
           same        there is no such message: with the lenient parser the conversation is the saved
                       one, and so is its pass or fail
           ends_pass   the attempt ends at that message (the call closes the request, or it was the last
                       allowed message), and the frozen verifier passes the end state
           ends_fail   the attempt ends at that message, and the verifier fails the end state
           open        the attempt does not end there. The next message of the model would answer another
                       tool result than the saved one, and no saved text holds it. The attempt cannot be
                       scored a second time.
       `changed_turn` is the turn of that message (None for "same"), `at_last_message` says that it is
       the last saved message, and `lenient_tool` is the tool that the lenient harness ran.

    Raises ValueError if the record was not made with the strict parser, or if the harness does not give
    the saved turn for a saved message (scripts/replay_check.py checks the same for every record)."""
    config = load_script("replay_check").config_of(record)
    if not config.parser_strict:
        raise ValueError(f"{attempt_id(record)} was not made with the strict parser: there is no second, "
                         "lenient scoring of it")
    episode = Episode(family, record["style"], text, dataclasses.replace(config, parser_strict=False),
                      ruleset=record["ruleset"], toolset=record["toolset"], prompt=record["prompt"])
    messages = []
    for saved in record["turns"]:
        if not saved["parse_ok"]:
            parsed = parse_assistant_message(saved["assistant_text"], episode.tool_specs, strict=False,
                                             truncated=saved["truncated"])
            messages.append({"turn": saved["turn"], "strict_error": saved["parse_error"],
                             "lenient_ok": parsed.ok, "lenient_error": parsed.error,
                             "lenient_tool": parsed.tool_key})
    reading = {"messages": messages, "second": "same", "changed_turn": None, "at_last_message": None,
               "lenient_tool": None}
    for saved in record["turns"]:
        if episode.done:
            raise ValueError(f"{attempt_id(record)}: turn {saved['turn']} comes after the end of the attempt")
        made = episode.step(saved["assistant_text"], truncated=saved["truncated"], n_tokens=saved["n_tokens"])
        if all(made[key] == saved[key] for key in SAME_TURN_FIELDS):
            continue
        if saved["parse_ok"] or not made["parse_ok"]:
            raise ValueError(f"{attempt_id(record)}, turn {saved['turn']}: the harness does not give the "
                             "saved turn for the saved message (run scripts/replay_check.py on this file)")
        reading.update(changed_turn=saved["turn"], lenient_tool=made["tool_key"],
                       at_last_message=saved["turn"] == record["turns"][-1]["turn"])
        if episode.done:
            passed = impl.get("verify")(episode.world.state(), family["expected"][record["ruleset"]])
            reading["second"] = "ends_pass" if passed else "ends_fail"
        else:
            reading["second"] = "open"
        break
    return reading


def lenient_of_arm(arm: Arm) -> dict:
    """The malformed messages of one arm by the reason of the strict parser, with what the lenient parser
    finds in them, and the second scoring of every attempt (`lenient_reading`).

    `passed_bounds`: the lowest and the highest number of passed attempts that a lenient parser can give
    for these saved conversations. Lowest: every "open" attempt fails. Highest: every "open" attempt
    passes. `pass_rate_bounds` is the frozen `metrics.pass_rate` of the same two cases."""
    run = arm.run
    valid = metrics.valid_records(run.records)
    by_code = {code: {"n": 0, "lenient_reads_a_call": 0, "lenient_rejects": Counter()}
               for code in PARSE_ERRORS}
    attempts = {second: {"passed": 0, "failed": 0} for second in SECOND}
    changed, lowest, highest = [], [], []
    only_last = earlier_close = 0
    for record in valid:
        reading = lenient_reading(record, run.families[record["family_id"]], run.text)
        for message in reading["messages"]:
            row = by_code[message["strict_error"]]
            row["n"] += 1
            if message["lenient_ok"]:
                row["lenient_reads_a_call"] += 1
            else:
                row["lenient_rejects"][message["lenient_error"]] += 1
        second, passed = reading["second"], bool(record["passed"])
        attempts[second]["passed" if passed else "failed"] += 1
        low = second == "ends_pass" or (second == "same" and passed)
        lowest.append({**record, "passed": low})
        highest.append({**record, "passed": low or second == "open"})
        if second != "same":
            changed.append({"attempt": attempt_id(record), "passed": passed, "turn": reading["changed_turn"],
                            "strict_error": record["turns"][reading["changed_turn"]]["parse_error"],
                            "lenient_tool": reading["lenient_tool"], "second": second,
                            "at_last_message": reading["at_last_message"]})
        if second == "ends_pass" and not passed:
            only_last += bool(reading["at_last_message"])
            earlier_close += not reading["at_last_message"]
    n_messages = sum(len(record["turns"]) for record in valid)
    n_malformed = sum(row["n"] for row in by_code.values())
    if n_malformed != sum(record["n_malformed"] for record in valid) \
            or not math.isclose(n_malformed / n_messages, metrics.invalid_call_rate(valid), abs_tol=1e-12):
        raise RuntimeError(f"arm {arm.name}: the malformed messages are not those of "
                           "metrics.invalid_call_rate")
    return {
        "n_attempts": len(valid), "n_messages": n_messages, "n_malformed": n_malformed,
        "invalid_call_rate": metrics.invalid_call_rate(valid),
        "n_lenient_reads_a_call": sum(row["lenient_reads_a_call"] for row in by_code.values()),
        "by_strict_error": {code: {"n": row["n"], "lenient_reads_a_call": row["lenient_reads_a_call"],
                                   "lenient_rejects": dict(sorted(row["lenient_rejects"].items()))}
                            for code, row in by_code.items()},
        "attempts": attempts,
        "failed_only_problem_in_last_message": only_last,
        "failed_ends_pass_at_an_earlier_close": earlier_close,
        "n_open": attempts["open"]["passed"] + attempts["open"]["failed"],
        "passed_saved": sum(1 for record in valid if record["passed"]),
        "passed_bounds": [sum(1 for record in lowest if record["passed"]),
                          sum(1 for record in highest if record["passed"])],
        "pass_rate_saved": metrics.pass_rate(valid),
        "pass_rate_bounds": [metrics.pass_rate(lowest), metrics.pass_rate(highest)],
        "changed": changed,
    }


def kinds_part(arms: list[Arm]) -> dict:
    return {"kinds": list(KINDS), "labels": KIND_LABELS, "rules": KIND_RULES,
            "record_fault_rules": RECORD_FAULT_RULES,
            "arms": {arm.name: kinds_of_arm(arm) for arm in arms},
            "lenient": {arm.name: lenient_of_arm(arm) for arm in arms}}


# ---------------------------------------------------------------------------
# Flips
# ---------------------------------------------------------------------------
def passes_by_scenario(records: list[dict]) -> dict[tuple[str, int], tuple[int, int]]:
    """(family id, style) -> (passed attempts, valid attempts), scenarios in sorted order."""
    counts: dict[tuple[str, int], list[int]] = {}
    for record in metrics.valid_records(records):
        entry = counts.setdefault((record["family_id"], record["style"]), [0, 0])
        entry[0] += bool(record["passed"])
        entry[1] += 1
    return {key: (counts[key][0], counts[key][1]) for key in sorted(counts)}


def passes_by_family(records: list[dict]) -> dict[str, tuple[int, int]]:
    """family id -> (passed attempts, valid attempts). The share of the two is the family pass rate of
    the frozen `metrics.family_pass_rates`; ValueError if it is not."""
    counts: dict[str, list[int]] = {}
    for record in metrics.valid_records(records):
        entry = counts.setdefault(record["family_id"], [0, 0])
        entry[0] += bool(record["passed"])
        entry[1] += 1
    rates = metrics.family_pass_rates(records)
    if set(rates) != set(counts) or any(not math.isclose(passed / n, rates[family_id], abs_tol=1e-12)
                                        for family_id, (passed, n) in counts.items()):
        raise ValueError("the passes per family are not the family pass rates of accessdesk/eval/metrics.py")
    return {family_id: (counts[family_id][0], counts[family_id][1]) for family_id in sorted(counts)}


def scenario_flip(before: tuple[int, int] | None, after: tuple[int, int] | None) -> str:
    """The class of one scenario between two arms. `before` and `after` are (passed, valid attempts).

    With 8 valid attempts in both arms:
        fail_to_pass     3 or fewer passed before, 5 or more passed after
        pass_to_fail     5 or more passed before, 3 or fewer passed after
        stayed_passing   5 or more passed in both arms
        stayed_failing   3 or fewer passed in both arms
        half             exactly 4 of 8 passed in one arm or in both
    The last three are the unchanged scenarios. `not_classified`: the scenario does not have exactly 8
    valid attempts in both arms (attempts lost to harness errors), so "of 8" has no meaning for it."""
    if before is None or after is None or before[1] != ATTEMPTS_PER_SCENARIO \
            or after[1] != ATTEMPTS_PER_SCENARIO:
        return "not_classified"
    low_before, high_before = before[0] <= MOSTLY_FAILING_MAX, before[0] >= MOSTLY_PASSING_MIN
    low_after, high_after = after[0] <= MOSTLY_FAILING_MAX, after[0] >= MOSTLY_PASSING_MIN
    if low_before and high_after:
        return "fail_to_pass"
    if high_before and low_after:
        return "pass_to_fail"
    if high_before and high_after:
        return "stayed_passing"
    if low_before and low_after:
        return "stayed_failing"
    return "half"


def family_change(before: tuple[int, int], after: tuple[int, int]) -> tuple[str, Fraction]:
    """(the class of one family between two arms, the change of its pass rate as an exact fraction).

    The pass rate of a family is passed attempts / valid attempts over all its wordings (the family pass
    rate of the plan). The change is the rate after minus the rate before.
        better   the change is more than +10 points (more than 0.10)
        worse    the change is less than -10 points
        within   the change is from -10 to +10 points, both ends included
    The comparison is made with exact fractions, so a change of exactly 10 points is "within"."""
    change = Fraction(after[0], after[1]) - Fraction(before[0], before[1])
    limit = Fraction(FAMILY_POINTS, 100)
    return ("better" if change > limit else "worse" if change < -limit else "within"), change


def family_facts(family: dict) -> dict:
    """The facts of one family that the flip tables show, all read from the family record.

    `active`, `holds_role`, `sensitive`: the three rule facts of the person meant and the role asked for
    (`rules.target_facts`, frozen). `two_people_one_name`: the world holds another person with the full
    name of the person meant (names compared as the lookup tool compares them). `other_person_status`:
    the account status of each such person. `person_meant_is_first_match`: the lookup tool lists the
    person meant before the other person (`target_is_first_match` of scripts/build_scenarios.py, frozen).
    Raises ValueError if the flag `homonym` of the record says something else than its world."""
    active, holds_role, sensitive = rules.target_facts(family)
    meant = family["truth"]["person_id"]
    names = {person["person_id"]: normalise(person["name"]) for person in family["world"]["people"]}
    others = [person for person in family["world"]["people"]
              if person["person_id"] != meant and names[person["person_id"]] == names[meant]]
    if "homonym" in family and bool(family["homonym"]) != bool(others):
        raise ValueError(f"family {family['family_id']}: the flag homonym is {family['homonym']!r}, but "
                         f"the world holds {len(others)} other people with the name of the person meant")
    return {"family_id": family["family_id"], "outcome": family["outcome"], "active": active,
            "holds_role": holds_role, "sensitive": sensitive, "two_people_one_name": bool(others),
            "other_person_status": [person["status"] for person in others],
            "person_meant_is_first_match": load_script("build_scenarios").target_is_first_match(family)}


def flips_part(arms: list[Arm], pairs: list[tuple[str, str]]) -> dict:
    """The passes of every scenario and family in each arm, and the classes of every pair of arms."""
    names = [arm.name for arm in arms]
    families = arms[0].run.families
    per_scenario = {arm.name: passes_by_scenario(arm.run.records) for arm in arms}
    per_family = {arm.name: passes_by_family(arm.run.records) for arm in arms}
    keys = sorted({key for counts in per_scenario.values() for key in counts})
    facts = {family_id: family_facts(families[family_id]) for family_id in sorted({key[0] for key in keys})}
    used = [(a, b) for a, b in pairs if a in names and b in names]
    left_out = [f"{a} to {b}" for a, b in pairs if (a, b) not in used]

    scenarios = []
    for family_id, style in keys:
        row = {"scenario_id": f"{family_id}/s{style}", "family_id": family_id, "style": style,
               "held_out_style": style == metrics.HELD_OUT_STYLE, "outcome": facts[family_id]["outcome"],
               "two_people_one_name": facts[family_id]["two_people_one_name"],
               "other_person_status": facts[family_id]["other_person_status"],
               "passes": {name: per_scenario[name].get((family_id, style), (0, 0))[0] for name in names},
               "attempts": {name: per_scenario[name].get((family_id, style), (0, 0))[1] for name in names}}
        row["flips"] = {f"{a}>{b}": scenario_flip(per_scenario[a].get((family_id, style)),
                                                  per_scenario[b].get((family_id, style))) for a, b in used}
        scenarios.append(row)
    family_rows = []
    for family_id, fact in facts.items():
        row = {**fact, "passes": {name: per_family[name].get(family_id, (0, 0))[0] for name in names},
               "attempts": {name: per_family[name].get(family_id, (0, 0))[1] for name in names},
               "changes": {}}
        for a, b in used:
            if family_id in per_family[a] and family_id in per_family[b]:
                kind, change = family_change(per_family[a][family_id], per_family[b][family_id])
                row["changes"][f"{a}>{b}"] = {"class": kind, "points": float(100 * change)}
        family_rows.append(row)

    pair_rows = []
    for a, b in used:
        key = f"{a}>{b}"
        classes = Counter(row["flips"][key] for row in scenarios)
        changed = Counter(row["changes"][key]["class"] for row in family_rows if key in row["changes"])
        pair_rows.append({
            "from": a, "to": b, "n_scenarios": len(scenarios),
            "scenarios": {name: classes[name] for name in FLIP_CLASSES},
            "scenarios_unchanged": sum(classes[name] for name in UNCHANGED_CLASSES),
            "scenarios_pass_to_fail": [row["scenario_id"] for row in scenarios
                                       if row["flips"][key] == "pass_to_fail"],
            "scenarios_fail_to_pass": [row["scenario_id"] for row in scenarios
                                       if row["flips"][key] == "fail_to_pass"],
            "n_families": sum(changed.values()),
            "families": {name: changed[name] for name in FAMILY_CLASSES},
            "families_worse": [row["family_id"] for row in family_rows
                               if row["changes"].get(key, {}).get("class") == "worse"],
            "families_better": [row["family_id"] for row in family_rows
                                if row["changes"].get(key, {}).get("class") == "better"],
        })
    return {"arms": names, "pairs": pair_rows, "pairs_left_out": left_out,
            "rule": {"attempts_per_scenario": ATTEMPTS_PER_SCENARIO, "mostly_failing_max": MOSTLY_FAILING_MAX,
                     "mostly_passing_min": MOSTLY_PASSING_MIN, "family_points": FAMILY_POINTS},
            "scenarios": scenarios, "families": family_rows}


# ---------------------------------------------------------------------------
# Flagged passes
# ---------------------------------------------------------------------------
def saved_report_check(path: Path, summary: dict) -> dict:
    """Are the numbers of `summary` those of the saved monitor report `path`?

    The frozen `monitors.to_markdown` writes the numbers of a summary as lines: the two lines of the
    sizes ("Episodes: ..." and "Passed: ..."), one table row per monitor, and the two rows of the
    stricter pass rules. Each of these lines, made again here from the saved attempts, must be in the
    saved report, character for character. `equal` is None if the saved report is not there."""
    if not path.is_file():
        return {"file": shown(path), "present": False, "equal": None, "n_lines": 0, "missing": []}
    saved = set(path.read_text(encoding="utf-8").splitlines())
    made = [line for line in monitors.to_markdown(summary, 0).splitlines()
            if line.startswith(("Episodes:", "Passed:", "| m", "| passed"))]
    missing = [line for line in made if line not in saved]
    return {"file": shown(path), "present": True, "equal": not missing, "n_lines": len(made),
            "missing": missing}


def flagged_part(arms: list[Arm], monitors_dir: Path, split: str) -> dict:
    """Per arm the summary of the frozen `monitors.summarise` (made by `make_arm`), and the check against
    the saved report."""
    part: dict = {"monitors": list(monitors.MONITORS), "labels": monitors.LABELS,
                  "clean_monitors": list(monitors.CLEAN_MONITORS), "arms": {}, "saved_reports": {}}
    for arm in arms:
        part["arms"][arm.name] = arm.summary
        part["saved_reports"][arm.name] = saved_report_check(monitors_dir / f"{arm.name}_{split}.md",
                                                            arm.summary)
    return part


# ---------------------------------------------------------------------------
# The LoRA adapter
# ---------------------------------------------------------------------------
def _group_key(row: dict) -> tuple:
    """Sort key of a (layer type, block, module) row: the order of the tables."""
    def position(value, order):
        return order.index(value) if value in order else len(order)
    return (position(row["layer_type"], LAYER_TYPE_ORDER), position(row["block"], BLOCK_ORDER),
            position(row["module"], MODULE_ORDER), str(row["module"]))


def fold_coverage(rows: list[dict]) -> list[dict]:
    """The rows of lora_coverage.json (one per linear module of the model) folded over the layers: one row
    per (layer type, block, module).

        of, wrapped        how many modules the row has, and how many of them have an adapter in this run
        lora_params        the adapter weights of the row in this run (the sum of the file's `lora_params`)
        short_wrapped      how many modules of the row the short target list would wrap: the modules whose
                           name is q_proj or v_proj (SHORT_LIST). A list of names wraps every module with
                           one of these names, so this is a count of names, not a run.
        short_lora_params  the adapter weights of those modules at the rank of this run: the file's
                           `lora_params` of the same modules (all of them are wrapped in this run too).
                           None if one of them is not wrapped here, so that no number is made up.
        shape              [out, in] of the weight of one module of the row"""
    groups: dict[tuple, dict] = {}
    for row in rows:
        key = (row["layer_type"], row["block"], row["module"])
        group = groups.setdefault(key, {"layer_type": row["layer_type"], "block": row["block"],
                                        "module": row["module"], "shape": row["shape"], "of": 0,
                                        "wrapped": 0, "lora_params": 0, "short_wrapped": 0,
                                        "short_lora_params": 0})
        group["of"] += 1
        group["wrapped"] += bool(row["wrapped"])
        group["lora_params"] += row["lora_params"]
        if row["module"] in SHORT_LIST:
            group["short_wrapped"] += 1
            if not row["wrapped"]:
                group["short_lora_params"] = None       # this run has no adapter of that module to count
            elif group["short_lora_params"] is not None:
                group["short_lora_params"] += row["lora_params"]
    return sorted(groups.values(), key=_group_key)


def layer_counts(rows: list[dict]) -> dict:
    """Per layer type: its layers, how many of them hold at least one adapter in this run, and how many
    would hold one with the short target list."""
    counts = {}
    for layer_type in sorted({row["layer_type"] for row in rows if row["layer"] is not None}):
        mine = [row for row in rows if row["layer_type"] == layer_type and row["layer"] is not None]
        counts[layer_type] = {
            "layers": sorted({row["layer"] for row in mine}),
            "n_layers": len({row["layer"] for row in mine}),
            "n_layers_with_an_adapter": len({row["layer"] for row in mine if row["wrapped"]}),
            "n_layers_with_an_adapter_short_list": len({row["layer"] for row in mine
                                                        if row["module"] in SHORT_LIST}),
        }
    return counts


def adapter_scale(config: dict) -> tuple[int, float, float]:
    """(rank, alpha, scale) of an adapter config: the scale of the update is alpha / rank.

    Raises ValueError for a config with another rule (rslora, DoRA, or a rank or alpha of its own for
    some modules): the norms of this script would then not be the norms of the update."""
    other = [key for key in ("use_rslora", "use_dora") if config.get(key)]
    other += [key for key in ("rank_pattern", "alpha_pattern") if config.get(key)]
    if other:
        raise ValueError(f"the adapter config uses {other}: the scale is not alpha / rank for every module")
    rank, alpha = config["r"], config["lora_alpha"]
    return rank, alpha, alpha / rank


def adapter_modules(keys: list[str]) -> dict[str, dict[str, str]]:
    """module path -> {"A": tensor name of lora_A, "B": tensor name of lora_B}, for the names of an adapter
    file. A module path is as in lora_coverage.json, for example "model.layers.3.self_attn.q_proj".
    Raises ValueError for a name of another form, and for a module without both tensors."""
    found: dict[str, dict[str, str]] = {}
    for key in keys:
        match = ADAPTER_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"the adapter file holds {key!r}, which is not a LoRA weight")
        found.setdefault(match.group("path"), {})[match.group("part")] = key
    broken = sorted(path for path, parts in found.items() if set(parts) != {"A", "B"})
    if broken:
        raise ValueError(f"{len(broken)} modules of the adapter file do not have both lora_A and lora_B, "
                         f"for example {broken[0]}")
    return {path: found[path] for path in sorted(found)}


def update_square(a: np.ndarray, b: np.ndarray) -> float:
    """The squared Frobenius norm of the matrix B A, without building it.

    `a` is lora_A [rank, in] and `b` is lora_B [out, rank]. The squared norm of B A is the sum of the
    entries of (B^T B) * (A A^T), entry by entry: two matrices of rank x rank. (It is trace(A^T B^T B A)
    = trace((B^T B)(A A^T)), and both factors are symmetric.) Computed in float64; never below 0.

    The sum is exactly 0.0 if every entry of B is 0 or every entry of A is 0. If B A is zero and neither
    matrix is, the sum is 0.0 or a very small number: its terms are rounded, and they need not cancel
    exactly.
    Raises ValueError if an entry of A or of B is NaN or infinite, or if the sum is not a finite number
    (entries too large for float64; float32 weights cannot be). Without this, a NaN would become the norm
    0: `max(0.0, nan)` is 0.0."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    if not (np.isfinite(a64).all() and np.isfinite(b64).all()):
        raise ValueError("a LoRA weight is not a finite number (NaN or infinity), so the update has no norm")
    with np.errstate(over="ignore", invalid="ignore"):      # an overflow is reported by the check below
        square = float(np.sum((b64.T @ b64) * (a64 @ a64.T)))
    if not math.isfinite(square):
        raise ValueError("the squared norm of the update is not a finite number: the weights are too large")
    return max(0.0, square)


def difference_square(a_old: np.ndarray, b_old: np.ndarray, a_new: np.ndarray, b_new: np.ndarray) -> float:
    """The squared Frobenius norm of B_new A_new - B_old A_old, without building either matrix.

    The difference is (B_new - B_old) A_new + B_old (A_new - A_old). That is P Q with P = [B_new - B_old,
    B_old] (out x 2 rank) and Q = [A_new; A_new - A_old] (2 rank x in), so `update_square(Q, P)` gives its
    squared norm from two matrices of 2 rank x 2 rank. Written with the small differences, so that two
    adapters that are near to each other lose no digits."""
    a_old, b_old = a_old.astype(np.float64), b_old.astype(np.float64)
    a_new, b_new = a_new.astype(np.float64), b_new.astype(np.float64)
    return update_square(np.concatenate([a_new, a_new - a_old], axis=0),
                         np.concatenate([b_new - b_old, b_old], axis=1))


def _square_of(path: str, square, *matrices: np.ndarray) -> float:
    """`square(*matrices)`, where `square` is `update_square` or `difference_square`. A ValueError of it
    (a weight that is not a finite number) is raised again with the module `path` in front."""
    try:
        return square(*matrices)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def _pooled(squares: dict[str, float], rows: dict[str, dict], zero: tuple[str, ...] = ()) -> dict:
    """Squared norms per module path, pooled: the norm of a group of modules is the square root of the
    sum of the squared norms of its modules. Returns {"groups": [...], "by_layer_type", "by_block",
    "by_module", "total"}; `groups` has one entry per (layer type, block, module), with the number of
    its modules and how many of them are in `zero` (the modules with a zero update). `zero_by_layer_type`
    counts the same per layer type.
    Raises ValueError for a module that is not a projection of a layer: the tables have no row for it."""
    groups: dict[tuple, dict] = {}
    pooled = {"by_layer_type": Counter(), "by_block": Counter(), "by_module": Counter()}
    zero_by_layer_type: Counter = Counter()
    for path, square in squares.items():
        row = rows[path]
        if row["layer_type"] is None or row["block"] is None:
            raise ValueError(f"{path} has an adapter but is not a module of a layer")
        group = groups.setdefault((row["layer_type"], row["block"], row["module"]),
                                  {"layer_type": row["layer_type"], "block": row["block"],
                                   "module": row["module"], "n_modules": 0, "n_zero": 0, "square": 0.0})
        group["n_modules"] += 1
        group["n_zero"] += path in zero
        group["square"] += square
        zero_by_layer_type[row["layer_type"]] += path in zero
        for key, name in (("by_layer_type", row["layer_type"]), ("by_block", row["block"]),
                          ("by_module", row["module"])):
            pooled[key][name] += square
    out: dict = {key: {name: math.sqrt(value) for name, value in sorted(values.items())}
                 for key, values in pooled.items()}
    kept = ("layer_type", "block", "module", "n_modules", "n_zero")
    out["groups"] = [{**{key: group[key] for key in kept}, "norm": math.sqrt(group["square"])}
                     for group in sorted(groups.values(), key=_group_key)]
    out["zero_by_layer_type"] = dict(sorted(zero_by_layer_type.items()))
    out["total"] = math.sqrt(sum(squares.values()))
    return out


def adapter_numbers(adapter_dir: Path, rows: dict[str, dict]) -> dict:
    """The numbers of one saved adapter, read with safetensors and numpy (torch is not used).

    For every wrapped module the update of the weight is scale * B A, with scale = alpha / rank of the
    adapter config. Reported:
        norms         the Frobenius norm of the update, pooled per (layer type, block, module), per layer
                      type, per block, per module name, and over all modules (`_pooled`)
        module_norms  the same norm for every single module (path -> norm)
        zero_modules  the modules with a zero update: the squared norm that `update_square` gives, times
                      the squared scale, is exactly 0. `zero_reasons` gives the reason of each one, the
                      first that applies (ZERO_REASONS): "b_is_zero" every entry of lora_B is 0 (lora_B
                      starts at zero, so training never moved this module); "a_is_zero" every entry of
                      lora_A is 0; "product_is_zero" neither, and the product is zero all the same. The
                      first two are exact. The third one is not: see `update_square`. So `smallest` is
                      reported beside the count. `n_zero_by_reason` counts the reasons.
        smallest      the module with the smallest norm
        fingerprint   the sum of the absolute values of all adapter weights (in float64): the number that
                      the GRPO run manifests keep for the adapter they started from and ended with
    `rows` maps a module path to its row of lora_coverage.json. Raises ValueError if the adapter does not
    hold exactly the wrapped modules of that file, if a tensor has another shape than rank x in, out x
    rank, or if a weight is not a finite number."""
    from safetensors import safe_open

    config = read_json(adapter_dir / ADAPTER_CONFIG)
    rank, alpha, scale = adapter_scale(config)
    squares: dict[str, float] = {}
    zero: dict[str, str] = {}
    fingerprint = 0.0
    n_weights = 0
    with safe_open(str(adapter_dir / ADAPTER_FILE), framework="np") as handle:
        modules = adapter_modules(list(handle.keys()))
        wrapped = sorted(path for path, row in rows.items() if row["wrapped"])
        if sorted(modules) != wrapped:
            only = sorted(set(modules) ^ set(wrapped))
            raise ValueError(f"{adapter_dir / ADAPTER_FILE} does not hold exactly the wrapped modules of "
                             f"the coverage file: {len(only)} differ, for example {only[0]}")
        for path, names in modules.items():
            a, b = handle.get_tensor(names["A"]), handle.get_tensor(names["B"])
            out_features, in_features = rows[path]["shape"]
            if a.shape != (rank, in_features) or b.shape != (out_features, rank):
                raise ValueError(f"{path}: lora_A {a.shape} and lora_B {b.shape} are not rank x in and "
                                 f"out x rank for rank {rank} and the weight {rows[path]['shape']}")
            squares[path] = scale * scale * _square_of(path, update_square, a, b)
            if not b.any():
                zero[path] = "b_is_zero"
            elif not a.any():
                zero[path] = "a_is_zero"
            elif squares[path] == 0.0:
                zero[path] = "product_is_zero"
            fingerprint += float(np.abs(a.astype(np.float64)).sum() + np.abs(b.astype(np.float64)).sum())
            n_weights += a.size + b.size
    smallest = min(squares, key=lambda path: (squares[path], path))
    return {"folder": shown(adapter_dir), "file_sha256": sha256_of_file(adapter_dir / ADAPTER_FILE),
            "rank": rank, "alpha": alpha, "scale": scale, "n_modules": len(squares), "n_weights": n_weights,
            "b_starts_at_zero": config.get("init_lora_weights") is True,
            "norms": _pooled(squares, rows, tuple(zero)),
            "module_norms": {path: math.sqrt(square) for path, square in squares.items()},
            "zero_modules": list(zero), "n_zero_modules": len(zero), "zero_reasons": zero,
            "n_zero_by_reason": {reason: sum(1 for found in zero.values() if found == reason)
                                 for reason in ZERO_REASONS},
            "smallest": {"path": smallest, "norm": math.sqrt(squares[smallest])},
            "fingerprint": fingerprint}


def adapter_difference(old_dir: Path, new_dir: Path, rows: dict[str, dict]) -> dict:
    """The Frobenius norm of (the update of the adapter in `new_dir`) minus (the update of the adapter in
    `old_dir`), pooled like `adapter_numbers`. `n_modules_equal` counts the modules whose lora_A and
    lora_B are the same in every entry in both adapters: the modules that did not move at all.
    Both adapters must have the same scale and the same modules, and every weight must be a finite number;
    ValueError if not."""
    from safetensors import safe_open

    scales = [adapter_scale(read_json(folder / ADAPTER_CONFIG))[2] for folder in (old_dir, new_dir)]
    if scales[0] != scales[1]:
        raise ValueError(f"the adapters {old_dir} and {new_dir} have the scales {scales}")
    squares: dict[str, float] = {}
    n_equal = 0
    with safe_open(str(old_dir / ADAPTER_FILE), framework="np") as old, \
            safe_open(str(new_dir / ADAPTER_FILE), framework="np") as new:
        old_modules, new_modules = adapter_modules(list(old.keys())), adapter_modules(list(new.keys()))
        if list(old_modules) != list(new_modules):
            raise ValueError(f"the adapters {old_dir} and {new_dir} do not hold the same modules")
        for path in old_modules:
            a_old, b_old = (old.get_tensor(old_modules[path][part]) for part in "AB")
            a_new, b_new = (new.get_tensor(new_modules[path][part]) for part in "AB")
            squares[path] = scales[0] * scales[0] * _square_of(path, difference_square, a_old, b_old,
                                                               a_new, b_new)
            n_equal += np.array_equal(a_old, a_new) and np.array_equal(b_old, b_new)
    return {"norms": _pooled(squares, rows), "n_modules_equal": int(n_equal)}


def _relative(found: float, wanted: float) -> float:
    return abs(found - wanted) / abs(wanted) if wanted else abs(found)


def _get(mapping: dict, *keys: str):
    """The value at a path of keys, or None if a key is not there."""
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _check(name: str, ok: bool | None, detail: str) -> dict:
    """One check: `ok` is True, False, or None if it could not be made."""
    return {"name": name, "ok": ok, "detail": detail}


def lora_part(sft_run: Path, grpo_runs: dict[str, Path]) -> dict:
    """The coverage table and the weight tables. Raises ValueError if the SFT run manifest or the coverage
    file is not there (the part cannot be made); an adapter file that is not there is left out, and the
    part says so (adapter files are large and are not in the repository)."""
    manifest_file, coverage_file = sft_run / MANIFEST_FILE, sft_run / COVERAGE_FILE
    for path in (manifest_file, coverage_file):
        if not path.is_file():
            raise ValueError(f"{path} is not there")
    manifest, coverage = read_json(manifest_file), read_json(coverage_file)["rows"]
    rows = {row["path"]: row for row in coverage}
    lora = manifest["lora"]
    folded = fold_coverage(coverage)
    checks = []

    def same(row: dict) -> tuple:
        return (row["layer_type"], row["block"], row["module"], row["wrapped"], row["of"], row["lora_params"])
    in_manifest = sorted((same(row) for row in lora["coverage"]), key=str)
    checks.append(_check(
        "coverage", in_manifest == sorted((same(row) for row in folded), key=str),
        f"the {len(coverage)} rows of {shown(coverage_file)}, folded over the layers, against the "
        f"{len(lora['coverage'])} rows of lora.coverage in {shown(manifest_file)}"))
    n_wrapped = sum(1 for row in coverage if row["wrapped"])
    n_params = sum(row["lora_params"] for row in coverage)
    checks.append(_check(
        "totals", (n_wrapped, n_params) == (lora["wrapped_modules"], lora["trainable_parameters"]),
        f"{n_wrapped} wrapped modules and {n_params} adapter weights in the coverage file; the manifest "
        f"says {lora['wrapped_modules']} and {lora['trainable_parameters']}"))

    part: dict = {
        "sft_run": shown(sft_run), "manifest": shown(manifest_file), "coverage_file": shown(coverage_file),
        "manifest_sha256": sha256_of_file(manifest_file), "coverage_sha256": sha256_of_file(coverage_file),
        "rank": lora["rank"], "alpha": lora["alpha"], "targets": lora["targets"],
        # What the SFT run manifest names, for the sentence about the source of the short list (None: the
        # manifest does not name it).
        "model": manifest.get("model"),
        "peft_version_of_the_run": _get(manifest, "environment", "packages", SHORT_LIST_SOURCE["package"]),
        "short_list": list(SHORT_LIST), "short_list_source": json.loads(json.dumps(SHORT_LIST_SOURCE)),
        "n_linear_modules": len(coverage), "n_wrapped": n_wrapped,
        "n_lora_params": n_params, "coverage": folded, "layers": layer_counts(coverage),
        "n_short_wrapped": sum(row["short_wrapped"] for row in folded),
        "n_short_lora_params": (None if any(row["short_lora_params"] is None for row in folded)
                                else sum(row["short_lora_params"] for row in folded)),
        "adapters": {}, "differences": {}, "left_out": [], "checks": checks,
    }

    sft_adapter = sft_run / "adapter"
    if not (sft_adapter / ADAPTER_FILE).is_file():
        part["left_out"].append(f"{shown(sft_adapter / ADAPTER_FILE)} is not there: no weight table")
        return part
    sft = adapter_numbers(sft_adapter, rows)
    part["adapters"]["sft"] = {**sft, "what": "the SFT adapter", "steps": None}
    named = _get(manifest, "outputs", "adapter", ADAPTER_FILE, "sha256")
    checks.append(_check(
        "sft adapter file", None if named is None else sft["file_sha256"] == named,
        f"the SHA-256 of {shown(sft_adapter / ADAPTER_FILE)} against outputs.adapter of the SFT manifest"))
    checks.append(_check(
        "sft adapter config", (sft["rank"], sft["alpha"]) == (lora["rank"], lora["alpha"]),
        f"rank {sft['rank']} and alpha {sft['alpha']} in {ADAPTER_CONFIG}; the manifest says rank "
        f"{lora['rank']} and alpha {lora['alpha']}"))
    saved = manifest.get("weight_change")
    if saved is None:
        checks.append(_check("sft update norms", None, "the SFT manifest has no weight_change"))
    else:
        pairs = [(saved["total"], sft["norms"]["total"])]
        pairs += [(value, sft["norms"][key].get(name, 0.0))
                  for key in ("by_layer_type", "by_block", "by_module") for name, value in saved[key].items()]
        worst = max(_relative(found, wanted) for wanted, found in pairs)
        part["sft_weight_change_largest_relative_difference"] = worst
        checks.append(_check(
            "sft update norms", worst <= NORM_TOLERANCE,
            f"the {len(pairs)} norms of weight_change in the SFT manifest (computed with torch at the end "
            "of the SFT run) against the norms made here from the adapter file: the largest relative "
            f"difference is {worst:.1e} (limit {NORM_TOLERANCE:.0e})"))

    for name, run_dir in grpo_runs.items():
        run_manifest_file = run_dir / MANIFEST_FILE
        if not run_manifest_file.is_file():
            part["left_out"].append(f"{shown(run_manifest_file)} is not there: no column for {name}")
            continue
        run_manifest = read_json(run_manifest_file)
        folder = _get(run_manifest, "final_adapter", "folder")
        if folder is None or not (run_dir / folder / ADAPTER_FILE).is_file():
            missing = run_dir / (folder or "(no final_adapter in the run manifest)") / ADAPTER_FILE
            part["left_out"].append(f"{shown(missing)} is not there: no column for {name}")
            continue
        adapter_dir = run_dir / folder
        numbers = adapter_numbers(adapter_dir, rows)
        part["adapters"][name] = {**numbers, "what": f"the final adapter of the GRPO run {name}",
                                  "steps": _get(run_manifest, "last_checkpoint", "completed_steps"),
                                  "run_manifest": shown(run_manifest_file)}
        part["differences"][name] = adapter_difference(sft_adapter, adapter_dir, rows)
        final = _get(run_manifest, "final_adapter", "files", ADAPTER_FILE, "sha256")
        start = _get(run_manifest, "start_adapter", "files", ADAPTER_FILE, "sha256")
        checks.append(_check(
            f"{name} adapter file", None if final is None else numbers["file_sha256"] == final,
            f"the SHA-256 of {shown(adapter_dir / ADAPTER_FILE)} against final_adapter of the run manifest"))
        checks.append(_check(
            f"{name} started from the SFT adapter", None if start is None else start == sft["file_sha256"],
            f"start_adapter of {shown(run_manifest_file)} against the SHA-256 of the SFT adapter file"))
        at_start = _get(run_manifest, "lora", "fingerprint_at_start")
        at_end = _get(run_manifest, "last_checkpoint", "lora_fingerprint")
        if at_start is None or at_end is None:
            checks.append(_check(f"{name} fingerprints", None, "the run manifest names no fingerprint"))
        else:
            worst = max(_relative(sft["fingerprint"], at_start), _relative(numbers["fingerprint"], at_end))
            checks.append(_check(
                f"{name} fingerprints", worst <= FINGERPRINT_TOLERANCE,
                f"the sum of the absolute adapter weights is {sft['fingerprint']:.6f} in the SFT file and "
                f"{numbers['fingerprint']:.6f} in the final file; the run manifest says {at_start:.6f} at "
                f"its start and {at_end:.6f} at its end"))
    return part


# ---------------------------------------------------------------------------
# Markdown: kinds.md
# ---------------------------------------------------------------------------
def _sources_lines(report: dict) -> list[str]:
    """The lines that say where the attempts come from; shared by kinds.md, flips.md and flagged.md."""
    lines = []
    for source in report["sources"]["arms"]:
        seed = "" if source["run_seed"] is None else f", run seed {source['run_seed']}"
        lines.append(f"- Arm {source['arm']}: `{source['episodes']}` (SHA-256 "
                     f"{source['episodes_sha256'][:12]}). {source['n_attempts']} valid attempts in "
                     f"{source['n_scenarios']} scenarios of {source['n_families']} families{seed}. "
                     f"Left out: {source['n_harness_errors']} attempts that the harness lost.")
    for absent in report["sources"]["arms_not_present"]:
        lines.append(f"- Arm {absent['arm']}: not in this report. The folder `{absent['folder']}` holds no "
                     f"`{EPISODES_FILE}`.")
    lines.append(f"- Families: `{report['sources']['split_file']}` (SHA-256 "
                 f"{report['sources']['split_file_sha256'][:12]}). The records of every arm name this "
                 "file.")
    lines.append(f"- Text file of the runs: `{report['sources']['text_file']}` (SHA-256 "
                 f"{report['sources']['text_sha256'][:12]}).")
    lines += [f"- Note: {note}" for note in report["sources"]["notes"]]
    return lines


ARM_WORDS = {
    "base": "the model before the training of this project",
    "sft": "the model after stage 1, supervised fine-tuning (SFT) on passing attempts of base",
    "grpo_seed1": "the sft model after stage 2 (GRPO, a reinforcement-learning method), training seed 1",
    "grpo_seed2": "the same stage 2 with training seed 2",
}


def _arm_words(arms: list[str]) -> list[str]:
    """The line that says what an arm is, with the arms of ANALYSIS_PLAN.md section 3 that are present."""
    known = [f"{arm}: {ARM_WORDS[arm]}" for arm in arms if arm in ARM_WORDS]
    line = "- Arm: one model under test, with all its attempts."
    return [line + (f" The arms of the plan: {'; '.join(known)}." if known else "")]


def _check_lines(checks: list[dict]) -> list[str]:
    words = {True: "OK", False: "DIFFERENT", None: "NOT MADE"}
    return [f"- {words[check['ok']]} ({check['name']}): {check['detail']}." for check in checks]


def _largest(counts: dict[str, int]) -> str | None:
    """The key with the largest count (the first one in the order of the dict if two are equal); None if
    every count is 0."""
    best = max(counts, key=lambda key: counts[key], default=None)
    return best if best is not None and counts[best] > 0 else None


def _and(names: list[str]) -> str:
    """["a"] -> "a"; ["a", "b"] -> "a and b"; ["a", "b", "c"] -> "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _most_failed_words(row: dict) -> str:
    """The family with the most failed attempts of one arm, and its part of the failed attempts of the arm:
    "dv-46, 15 of 21". Where several families have that largest count, all of them are named, so that the
    first one is not read as the only one: "dv-04, dv-22 and dv-29 (3 families with the same count), 32 of
    925 each, together 96 of 925". `row` is the arm's entry of `kinds_of_arm`."""
    tied, n_failed = row["families_tied_for_most_failed"], row["n_failed"]
    if not tied:
        return "no failed attempt"
    names = _and([entry["family_id"] for entry in tied])
    if len(tied) == 1:
        return f"{names}, {tied[0]['n_failed']} of {n_failed}"
    return (f"{names} ({len(tied)} families with the same count), {tied[0]['n_failed']} of {n_failed} each, "
            f"together {sum(entry['n_failed'] for entry in tied)} of {n_failed}")


def _kinds_in_short(part: dict) -> list[str]:
    """The first lines of kinds.md: five counts per arm, each one from a table of the file."""
    rows_of, lenient = part["arms"], part["lenient"]
    arms = list(rows_of)

    def per_arm(cell) -> str:
        return "; ".join(f"{arm}: {cell(arm)}" for arm in arms)

    def largest(arm: str) -> str:
        kind = _largest({kind: rows_of[arm]["by_kind"][kind]["n"] for kind in KINDS})
        if kind is None:
            return "no failed attempt"
        name = KIND_LABELS[kind].split(" ", 1)[1]              # the label without its letter
        return f"{name}, {count_share(rows_of[arm]['by_kind'][kind]['n'], rows_of[arm]['n_failed'])}"

    return [
        "- Failed attempts of all valid attempts (Table 1): "
        + per_arm(lambda arm: f"{rows_of[arm]['n_failed']} of {rows_of[arm]['n_attempts']}") + ".",
        f"- The kind with the most failed attempts (Table 1): {per_arm(largest)}.",
        "- Failed attempts with no valid closing call, and those of them with a malformed message "
        "(Table 1): "
        + per_arm(lambda arm: f"{rows_of[arm]['n_no_valid_closing_call']}, of them "
                              f"{sum(rows_of[arm]['by_kind'][kind]['n'] for kind in NO_CLOSE_KINDS[:2])}")
        + ". Only an attempt with a malformed message can be a format failure. The other attempts (kind "
          "A3) made only well-formed calls.",
        "- The family with the most failed attempts, and its part of the failed attempts of the arm "
        "(Table 6). Where families have the same count, all of them are named: "
        + per_arm(lambda arm: _most_failed_words(rows_of[arm])) + ".",
        "- A lenient parser (Tables 7 and 8). Malformed messages in which it finds a call: "
        + per_arm(lambda arm: f"{lenient[arm]['n_lenient_reads_a_call']} of {lenient[arm]['n_malformed']}")
        + ". Failed attempts that pass for certain with it: "
        + per_arm(lambda arm: str(lenient[arm]["attempts"]["ends_pass"]["failed"]))
        + ". Attempts whose second result cannot be known: "
        + per_arm(lambda arm: str(lenient[arm]["n_open"])) + ".",
    ]


def kinds_markdown(report: dict) -> str:
    part, split = report["kinds"], report["split"]
    arms = list(part["arms"])
    rows_of = part["arms"]
    limits = sorted({limit for arm in arms for limit in rows_of[arm]["turn_limits"]}, key=str)
    limit_words = f"{limits[0]} messages" if len(limits) == 1 else "the allowed number of messages"
    lines = [
        f"# Failed attempts by kind of failure (split {split})",
        "",
        "Made by `scripts/failure_report.py` from saved attempts. No model was run.",
        "",
        "## In short",
        "",
        *_kinds_in_short(part),
        "",
        "The words, the rules and the tables behind these lines follow.",
        "",
        "## Where the numbers come from",
        "",
        *_sources_lines(report),
        "- Unit: the attempt. Each count is a number of attempts. A share in brackets is the count "
        "divided by the failed attempts of the same arm. These shares have no family weight.",
        "",
        "## Words",
        "",
        *_arm_words(arms),
        "- Attempt: one conversation of the model about one request.",
        "- Family: one world with one correct end state. Scenario: one family in one wording style.",
        "- Harness: the program that runs an attempt. It gives the request to the model, runs each tool "
        "call, and saves the attempt.",
        "- Verifier: the program that compares the end state of an attempt with the correct end state. "
        "A failed attempt is an attempt for which the verifier says no.",
        "- Family-weighted: the share inside each family first, then the mean over the families. The "
        "pass rate of the plan is family-weighted.",
        "- Expected outcome: the correct result of a family. There are four. grant: give the role. "
        "approval: ask the manager of the person. already: the person holds the role already. inactive: "
        "refuse, because the account of the person is not active.",
        "- Close: the tool call that ends an attempt. It names a person, a role and a resolution. The "
        "resolutions are granted, pending, already and denied. They are the correct ones for the outcomes "
        "grant, approval, already and inactive, in this order.",
        "- Malformed message: a message of the model that is not exactly one valid tool call. The harness "
        "runs no tool for it.",
        "- Record: a line that a tool writes into the end state. A grant record is one served grant. An "
        "approval record is one served approval request, with its approver.",
        "",
        "## The kinds and their rules",
        "",
        "Each failed attempt has exactly one kind. The rules read the saved record of the attempt "
        "(`final_state`, the calls in `turns`, `ended_by`, `n_malformed`) and the family record. They are "
        "tried from top to bottom. The first rule that applies gives the kind.",
        "",
        *md_table(["Kind", "Rule"], [[KIND_LABELS[kind], KIND_RULES[kind]] for kind in KINDS]),
        "A1, A2 and A3 together are the attempts with no valid closing call. A format failure needs a "
        "malformed message, and only A1 and A2 hold one. A3 is not a format failure: every message was a "
        f"well-formed call, and the model used {limit_words} and did not close the request. The small "
        "table below Table 1 shows what the attempts with no valid closing call did in place of a close.",
        "",
        "## Table 1. The failed attempts of each arm by kind",
        "",
    ]
    table = []
    for kind in KINDS:
        table.append([KIND_LABELS[kind]] + [count_share(rows_of[arm]["by_kind"][kind]["n"],
                                                        rows_of[arm]["n_failed"]) for arm in arms])
        if kind == NO_CLOSE_KINDS[-1]:
            table.append(["A1 + A2 + A3: no valid closing call"]
                         + [count_share(rows_of[arm]["n_no_valid_closing_call"], rows_of[arm]["n_failed"])
                            for arm in arms])
    table.append([f"Sum of the {len(KINDS)} kinds"] + [str(rows_of[arm]["sum_of_kinds"]) for arm in arms])
    table.append(["Failed attempts (valid attempts minus passed attempts)"]
                 + [f"{rows_of[arm]['n_failed']} = {rows_of[arm]['n_attempts']} - {rows_of[arm]['n_passed']}"
                    for arm in arms])
    table.append(["Pass rate (family-weighted, `metrics.pass_rate`)"]
                 + [pct(rows_of[arm]["pass_rate"]) for arm in arms])
    lines += md_table(["Kind"] + arms, table)
    most = []
    for arm in arms:
        kind = _largest({kind: rows_of[arm]["by_kind"][kind]["n"] for kind in KINDS})
        if kind is None:
            most.append(f"{arm}: no failed attempt")
        else:
            row = rows_of[arm]["by_kind"][kind]
            most.append(f"{arm}: {KIND_LABELS[kind].split()[0]}, {row['n']} of {rows_of[arm]['n_failed']} "
                        f"({pct(row['share_of_failed'])})")
    no_close = ", ".join(f"{arm} {rows_of[arm]['n_no_valid_closing_call']} of {rows_of[arm]['n_failed']}"
                         for arm in arms)
    with_malformed = ", ".join(
        f"{arm} {sum(rows_of[arm]['by_kind'][kind]['n'] for kind in NO_CLOSE_KINDS[:2])}" for arm in arms)
    lines += [
        "What the table shows. The sum of the kinds is the number of failed attempts in every arm. The "
        f"kind with the most failed attempts: {'; '.join(most)}. Failed attempts with no valid closing "
        f"call: {no_close}. Of them with at least one malformed message (A1 + A2): {with_malformed}. "
        "The other attempts with no valid closing call (A3) are not format failures.",
        "",
        "The ids of the attempts of each kind are in `failure_report.json` "
        "(`kinds.arms.<arm>.by_kind.<kind>.attempts`). An id is the scenario id, `#`, and the number of "
        "the attempt.",
        "",
        "What the attempts with no valid closing call did. A monitor is a frozen rule that looks for one "
        "thing in a saved attempt (`accessdesk/eval/monitors.py`; see `flagged.md`). Two monitors say "
        "what these attempts did in place of a close. m4: the attempt called the same tool with the "
        "same arguments more than once. m1: the attempt made a well-formed call that the tool answered "
        "with an error (for example an id that does not exist). Each cell: attempts of the kind that the "
        "monitor flags, of all attempts of the kind.",
        "",
        *md_table(["Kind", "Monitor"] + arms, [
            [KIND_LABELS[kind], f"{monitors.LABELS[name]} `{name}`"]
            + [f"{rows_of[arm]['by_kind'][kind]['flagged_by'][name]} of {rows_of[arm]['by_kind'][kind]['n']}"
               for arm in arms]
            for kind in NO_CLOSE_KINDS for name in ("repeated_call", "refused_call")]),
        "What the table shows. Attempts of the kind A3 in which the same call was made again (m4): "
        + "; ".join(f"{arm}: {rows_of[arm]['by_kind']['no_close_all_calls']['flagged_by']['repeated_call']} "
                    f"of {rows_of[arm]['by_kind']['no_close_all_calls']['n']}" for arm in arms) + ".",
        "",
        "## Table 2. The failed attempts of each arm by kind and by expected outcome",
        "",
    ]
    worst_outcome = []
    for arm in arms:
        row = rows_of[arm]
        lines += [f"### {arm}: {row['n_failed']} failed attempts of {row['n_attempts']}", ""]
        table = [[KIND_LABELS[kind]] + [str(row["by_kind"][kind]["by_outcome"][outcome])
                                        for outcome in OUTCOMES] + [str(row["by_kind"][kind]["n"])]
                 for kind in KINDS]
        table.append([f"Sum of the {len(KINDS)} kinds"]
                     + [str(sum(row["by_kind"][kind]["by_outcome"][outcome] for kind in KINDS))
                        for outcome in OUTCOMES] + [str(row["sum_of_kinds"])])
        table.append(["Failed attempts of all attempts"]
                     + [f"{row['failed_by_outcome'][outcome]} of {row['attempts_by_outcome'][outcome]}"
                        for outcome in OUTCOMES] + [f"{row['n_failed']} of {row['n_attempts']}"])
        lines += md_table(["Kind"] + list(OUTCOMES) + ["all outcomes"], table)
        outcome = _largest(row["failed_by_outcome"])
        worst_outcome.append(f"{arm}: no failed attempt" if outcome is None else
                             f"{arm}: {outcome}, {row['failed_by_outcome'][outcome]} of "
                             f"{row['attempts_by_outcome'][outcome]} attempts")
    lines += [
        "What the tables show. In every column the sum of the kinds is the number of failed attempts. "
        f"The expected outcome with the most failed attempts: {'; '.join(worst_outcome)}.",
        "",
        "## Table 3. Wrong decisions (kind D): the expected resolution and the resolution of the close",
        "",
    ]
    pairs = sorted({(row["expected"], row["given"])
                    for arm in arms for row in rows_of[arm]["wrong_decisions"]},
                   key=lambda pair: (RESOLUTIONS.index(pair[0]), RESOLUTIONS.index(pair[1])))
    table = []
    for expected, given in pairs:
        counts = [next((row["n"] for row in rows_of[arm]["wrong_decisions"]
                        if (row["expected"], row["given"]) == (expected, given)), 0) for arm in arms]
        table.append([expected, given] + [str(count) for count in counts])
    table.append(["All of kind D", ""]
                 + [str(rows_of[arm]["by_kind"]["wrong_decision"]["n"]) for arm in arms])
    lines += md_table(["Expected resolution", "The attempt closed as"] + arms, table)
    common = []
    for arm in arms:
        decisions = rows_of[arm]["wrong_decisions"]
        if decisions:
            top = max(decisions, key=lambda row: row["n"])
            common.append(f"{arm}: expected {top['expected']}, closed as {top['given']} ({top['n']} of "
                          f"{rows_of[arm]['by_kind']['wrong_decision']['n']})")
        else:
            common.append(f"{arm}: none")
    lines += [
        "What the table shows. Each cell is a number of attempts of kind D. The most common wrong decision: "
        f"{'; '.join(common)}.",
        "",
        "## Table 4. The right resolution with a wrong or missing record (kind E): what is wrong",
        "",
        "Each attempt of kind E gets exactly one row. The first rule that applies gives the row.",
        "",
    ]
    table = [[RECORD_FAULT_LABELS[fault], RECORD_FAULT_RULES[fault]]
             + [str(rows_of[arm]["record_faults"][fault]) for arm in arms] for fault in RECORD_FAULTS]
    table.append(["All of kind E", ""]
                 + [str(rows_of[arm]["by_kind"]["right_resolution_wrong_records"]["n"]) for arm in arms])
    lines += md_table(["What is wrong", "Rule"] + arms, table)
    lines += [
        "What the table shows. The three rows add up to the attempts of kind E in every arm.",
        "",
        "## Table 5. The wrong person of two with one name (kind B)",
        "",
        "The second row counts the attempts of kind B whose end state is the correct end state for the "
        "person that the attempt picked. The correct end state for that person comes from the frozen "
        "rules (`rules.expected_state`), and the frozen verifier compares. Such an attempt took the wrong "
        "person and then did what the rules ask for that person.",
        "",
    ]
    lines += md_table([""] + arms, [
        ["Attempts of kind B"] + [str(rows_of[arm]["wrong_person"]["n"]) for arm in arms],
        ["Of them: the end state is correct for the person picked"]
        + [str(rows_of[arm]["wrong_person"]["right_for_the_person_picked"]) for arm in arms]])
    lines += [
        "What the table shows. The first row is the number of failed attempts that monitor m6 flags (see "
        "`flagged.md`).",
        "",
        "## Table 6. The families with the most failed attempts",
        "",
        "For each arm: the three families with the most failed attempts (the smaller family id first "
        "when two counts are equal).",
        "",
    ]
    table = []
    for arm in arms:
        row = rows_of[arm]
        cells = [f"{top['family_id']} ({top['outcome']}): {top['n_failed']} of {top['n_attempts']}"
                 for top in row["families_with_most_failed"]]
        table.append([arm, str(row["n_failed"]), str(row["n_families_with_a_failed_attempt"])]
                     + cells + [""] * (3 - len(cells)))
    lines += md_table(["Arm", "Failed attempts", "Families with a failed attempt", "First family",
                       "Second family", "Third family"], table)
    most_failed = "; ".join(f"{arm}: {_most_failed_words(rows_of[arm])}" for arm in arms)
    not_shown = []
    for arm in arms:
        more = rows_of[arm]["families_tied_with_the_third"]
        if more:
            count = f"{more[0]['n_failed']} failed attempt{'' if more[0]['n_failed'] == 1 else 's'}"
            not_shown.append(f"{arm}: {_and([entry['family_id'] for entry in more])} "
                             f"({count}{' each' if len(more) > 1 else ''})")
    lines += [
        "What the table shows. A cell reads: family (expected outcome): failed attempts of the attempts "
        "of that family. The family with the most failed attempts of each arm, and its part of the failed "
        f"attempts of the arm (every family with the largest count is named): {most_failed}."
        + (" Families with the same count as the third family that the table does not show: "
           f"{'; '.join(not_shown)}." if not_shown else ""),
        "",
    ]
    lines += _lenient_lines(report)
    lines += [
        "## Checks",
        "",
        "The script stops and writes nothing if one of these three does not hold:",
        "",
        "- The frozen verifier gives the saved pass or fail for the saved end state of every attempt "
        "(`monitors.summarise`).",
        "- Each kind agrees with the frozen label of the attempt (`metrics.episode_label`): A1 is "
        "`malformed_only`, A2 and A3 are `never_closed`, and B to F are `closed_wrongly`.",
        "- The malformed messages of Table 7 are those of the frozen `metrics.invalid_call_rate`.",
        "",
        "The script made these checks and wrote the result:",
        "",
        *_check_lines([check for check in report.get("checks", []) if check["part"] == "kinds"]),
    ]
    return "\n".join(lines) + "\n"


def _lenient_lines(report: dict) -> list[str]:
    lenient = report["kinds"]["lenient"]
    arms = list(lenient)
    lines = [
        "## A second scoring with a lenient parser: what is exact, and what is not possible",
        "",
        "The parser is the program that reads a message of the model as a tool call. Every run used the "
        "strict setting: a message counts only if it is exactly one valid call. The parser also has a "
        "lenient setting (`strict=False` in `accessdesk/harness/parser.py`). The lenient setting:",
        "",
        "- reads the first call of a message that holds several calls;",
        "- ignores text after the call, and a missing closing tag;",
        "- ignores a parameter that the tool does not have;",
        "- compares tool names and parameter names without regard to case;",
        "- reads a call that is written as JSON.",
        "",
        "Both settings reject a message that reached the length limit and was cut off.",
        "",
        "The limit. A full second scoring of a saved conversation is not possible. Each message of the "
        "model answers the tool result before it. If the lenient parser reads a call where the strict "
        "parser read none, the harness runs that call and gives another tool result. The next message "
        "of the model would answer that result. No saved text holds that message.",
        "",
        "What is exact:",
        "",
        "1. Each malformed message was parsed again with the lenient setting (Table 7).",
        "2. The saved messages of each attempt were given again, in order, to the frozen harness with "
        "the lenient setting. While the lenient harness gives the saved tool result, the saved next "
        "message is still the message of the model.",
        "3. If no message is read differently, the conversation is the same with both settings. So the "
        "pass or fail is the same.",
        "4. If the attempt ends at the first message that only the lenient parser reads as a call, its "
        "end state is known. This is so when that call closes the request, and when that message is the "
        "last allowed one. The frozen verifier then gives the second pass or fail.",
        "5. If the attempt goes on after that message, it cannot be scored a second time.",
        "",
        "Table 8 counts the attempts of the cases 3, 4 and 5.",
        "",
        "## Table 7. Malformed messages by the reason of the strict parser, and what the lenient parser "
        "finds",
        "",
        "Each cell: malformed messages with this reason / of them the lenient parser finds one "
        "well-formed call. Unit: the message. All attempts of the arm are counted, passed and failed.",
        "",
    ]
    codes = [code for code in PARSE_ERRORS if any(lenient[arm]["by_strict_error"][code]["n"] for arm in arms)]
    table = [[f"`{code}`"] + [f"{lenient[arm]['by_strict_error'][code]['n']} / "
                              f"{lenient[arm]['by_strict_error'][code]['lenient_reads_a_call']}"
                              for arm in arms] for code in codes]
    table.append(["All malformed messages"]
                 + [f"{lenient[arm]['n_malformed']} / {lenient[arm]['n_lenient_reads_a_call']}"
                    for arm in arms])
    table.append(["All messages of the model"] + [str(lenient[arm]["n_messages"]) for arm in arms])
    table.append(["Malformed share (`metrics.invalid_call_rate`)"]
                 + [pct(lenient[arm]["invalid_call_rate"]) for arm in arms])
    lines += md_table(["Reason of the strict parser"] + arms, table)
    reads = "; ".join(f"{arm}: {lenient[arm]['n_lenient_reads_a_call']} of {lenient[arm]['n_malformed']}"
                      for arm in arms)
    lines += [
        "The reasons: `no_call`: the message holds no call. `truncated`: the message reached the length "
        "limit and was cut off. `multiple_calls`: more than one call. `bad_format`: the call does not "
        "follow the format. "
        "`text_after_call`: text follows the call. `unknown_tool`, `unknown_argument`, "
        "`missing_argument`, `duplicate_argument`, `empty_argument`: a wrong tool name or parameter. "
        "`bad_value`: a resolution that is not one of the four.",
        "",
        f"What the table shows. Malformed messages in which the lenient parser finds one call: {reads}.",
        "",
        "## Table 8. The second scoring of every attempt",
        "",
    ]
    names = {"same": "Case 3. No message is read differently: the same pass or fail",
             "ends_pass": "Case 4. The attempt ends at the newly read call, and the end state passes",
             "ends_fail": "Case 4. The attempt ends at the newly read call, and the end state fails",
             "open": "Case 5. The attempt goes on after the newly read call: no second scoring"}
    table = []
    for second in SECOND:
        for verdict in ("failed", "passed"):
            table.append([names[second], f"{verdict} in the saved run"]
                         + [str(lenient[arm]["attempts"][second][verdict]) for arm in arms])
    table.append(["Failed attempts whose only problem is a malformed last message that the lenient parser "
                  "reads, and that pass when this call is run", "failed in the saved run"]
                 + [str(lenient[arm]["failed_only_problem_in_last_message"]) for arm in arms])
    table.append(["Passed attempts, saved run", ""] + [str(lenient[arm]["passed_saved"]) for arm in arms])
    table.append(["Passed attempts with the lenient parser: lowest to highest possible number", ""]
                 + [f"{lenient[arm]['passed_bounds'][0]} to {lenient[arm]['passed_bounds'][1]}"
                    for arm in arms])
    table.append(["Pass rate, saved run (family-weighted)", ""]
                 + [pct(lenient[arm]["pass_rate_saved"]) for arm in arms])
    table.append(["Pass rate with the lenient parser: lowest to highest possible value", ""]
                 + [" to ".join(pct(bound) for bound in lenient[arm]["pass_rate_bounds"]) for arm in arms])
    lines += md_table(["Second scoring", "Saved result"] + arms, table)
    only = "; ".join(f"{arm}: {lenient[arm]['failed_only_problem_in_last_message']}" for arm in arms)
    opened = "; ".join(f"{arm}: {lenient[arm]['n_open']} of {lenient[arm]['n_attempts']}" for arm in arms)
    held = "; ".join(f"{arm}: {lenient[arm]['n_lenient_reads_a_call']} messages in "
                     f"{len(lenient[arm]['changed'])} attempts" for arm in arms)
    lines += [
        "The lowest possible number counts every attempt of case 5 as failed. The highest counts every "
        "one of them as passed. Both pass rates come from the frozen `metrics.pass_rate`.",
        "",
        "What the table shows. Failed attempts whose only problem is a malformed last message that the "
        f"lenient parser reads, and that pass when this call is run: {only}. Attempts that cannot be "
        f"scored a second time: {opened}. For every other attempt the second pass or fail is exact. The "
        f"messages of Table 7 in which the lenient parser finds a call: {held}.",
        "",
    ]
    listed = [(arm, entry) for arm in arms for entry in lenient[arm]["changed"]]
    if listed:
        lines += ["The attempts with a newly read call (scenario id # attempt number):", ""]
        lines += md_table(
            ["Arm", "Attempt", "Saved result", "Number of the message in the attempt",
             "Reason of the strict parser",
             "Tool that the lenient parser reads", "Second scoring"],
            [[arm, entry["attempt"], "passed" if entry["passed"] else "failed",
              f"{entry['turn'] + 1}" + (" (the last one)" if entry["at_last_message"] else ""),
              f"`{entry['strict_error']}`", f"`{entry['lenient_tool']}`", names[entry["second"]]]
             for arm, entry in listed])
    return lines


# ---------------------------------------------------------------------------
# Markdown and CSV: flips
# ---------------------------------------------------------------------------
def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _other_person(row: dict) -> str:
    """The status of the other person of the name, in words."""
    if not row["two_people_one_name"]:
        return "none"
    return ", ".join(row["other_person_status"])


def flips_markdown(report: dict) -> str:
    part, split = report["flips"], report["split"]
    arms, rule = part["arms"], part["rule"]
    n, low, high, points = (rule["attempts_per_scenario"], rule["mostly_failing_max"],
                            rule["mostly_passing_min"], rule["family_points"])
    scenarios, families = part["scenarios"], part["families"]
    lines = [
        f"# Flips: scenarios and families that changed between two arms (split {split})",
        "",
        "Made by `scripts/failure_report.py` from saved attempts. No model was run.",
        "",
        "## Where the numbers come from",
        "",
        *_sources_lines(report),
        f"- All {len(scenarios)} scenarios are in `flips.csv`, one row each, with the same numbers.",
        "- Check: the passes of each family, divided by its valid attempts, are the family pass rates of "
        "the frozen `metrics.family_pass_rates`. The script stops if they are not.",
        "",
        "## Words and rules",
        "",
        *_arm_words(arms),
        "- Attempt: one conversation of the model about one request. Family: one world with one correct "
        "end state. Scenario: one family in one wording style.",
        f"- Passes of {n}: how many of the {n} attempts of a scenario passed in one arm.",
        f"- Mostly failing: {low} or fewer of the {n} attempts passed. Mostly passing: {high} or more "
        "passed.",
        f"- A scenario flips when it is mostly failing in one arm and mostly passing in the other arm. "
        f"All other scenarios are unchanged. A scenario with exactly {n // 2} passes of {n} in an arm is "
        "neither mostly failing nor mostly passing, so it is unchanged.",
        "- Family pass rate: the passed attempts of a family divided by its valid attempts, over all its "
        "wording styles (`metrics.family_pass_rates`).",
        f"- A family changed when its pass rate moved by more than {points} points. A point is one "
        "hundredth: from 50% to 62.5% is a change of +12.5 points.",
        "",
        f"## Table 1. Scenarios: passes of {n} in two arms",
        "",
    ]
    not_classified = any(pair["scenarios"]["not_classified"] for pair in part["pairs"])
    header = ["From", "To", "Scenarios", "Mostly failing to mostly passing",
              "Mostly passing to mostly failing", "Unchanged", "of the unchanged: mostly passing in both",
              "mostly failing in both", f"{n // 2} of {n} in one arm or both"]
    if not_classified:
        header.append(f"Not classified (not {n} valid attempts)")
    table = []
    for pair in part["pairs"]:
        counts, total = pair["scenarios"], pair["n_scenarios"]
        row = [pair["from"], pair["to"], str(total), count_share(counts["fail_to_pass"], total),
               count_share(counts["pass_to_fail"], total), count_share(pair["scenarios_unchanged"], total),
               str(counts["stayed_passing"]), str(counts["stayed_failing"]), str(counts["half"])]
        if not_classified:
            row.append(str(counts["not_classified"]))
        table.append(row)
    lines += md_table(header, table)
    shows = "; ".join(f"{pair['from']} to {pair['to']}: {pair['scenarios']['fail_to_pass']} up and "
                      f"{pair['scenarios']['pass_to_fail']} down of {pair['n_scenarios']}"
                      for pair in part["pairs"])
    lines += [
        "What the table shows. Each share is a share of the scenarios of the row. Scenarios that went "
        f"from mostly failing to mostly passing (up) and the reverse (down): {shows}."
        + ("" if not_classified else f" Every scenario has {n} valid attempts in every arm, so every "
                                     "scenario has a class."),
        "",
    ]
    if part["pairs_left_out"]:
        lines += [f"- Not in Tables 1 to 4: the change {name}. An arm of it is not present."
                  for name in part["pairs_left_out"]] + [""]
    lines += [
        f"## Table 2. Families: pass rate in two arms, a change of more than {points} points",
        "",
    ]
    table = [[pair["from"], pair["to"], str(pair["n_families"]),
              count_share(pair["families"]["better"], pair["n_families"]),
              count_share(pair["families"]["worse"], pair["n_families"]),
              count_share(pair["families"]["within"], pair["n_families"])] for pair in part["pairs"]]
    lines += md_table(["From", "To", "Families", f"Better by more than {points} points",
                       f"Worse by more than {points} points", f"Within {points} points"], table)
    shows = "; ".join(f"{pair['from']} to {pair['to']}: {pair['families']['better']} better and "
                      f"{pair['families']['worse']} worse of {pair['n_families']}" for pair in part["pairs"])
    lines += [
        "What the table shows. Each share is a share of the families of the row. Families that got "
        f"better and families that got worse by more than {points} points: {shows}.",
        "",
        "## Table 3. The scenarios that went from mostly passing to mostly failing",
        "",
    ]
    by_id = {row["scenario_id"]: row for row in scenarios}
    table = []
    for pair in part["pairs"]:
        for scenario_id in pair["scenarios_pass_to_fail"]:
            row = by_id[scenario_id]
            table.append([pair["from"], pair["to"], scenario_id, row["outcome"],
                          f"{row['passes'][pair['from']]} of {row['attempts'][pair['from']]}",
                          f"{row['passes'][pair['to']]} of {row['attempts'][pair['to']]}"])
    if table:
        lines += md_table(["From", "To", "Scenario", "Expected outcome", "Passes before", "Passes after"],
                          table)
        lines += [f"What the table shows. {len(table)} rows: every scenario that flipped down, in every "
                  "pair of arms. A scenario id is the family id, `/s`, and the number of the wording style.",
                  ""]
    else:
        lines += ["No scenario went from mostly passing to mostly failing in any pair of arms.", ""]
    lines += [
        f"## Table 4. The families that got worse by more than {points} points, with their facts",
        "",
        "The facts come from the family record. Person meant: the person that the request is about. "
        "Other person of the name: the account status of the second person with the same full name, or "
        "none. First match: the person that the lookup tool lists first for that name.",
        "",
    ]
    by_family = {row["family_id"]: row for row in families}
    table = []
    for pair in part["pairs"]:
        key = f"{pair['from']}>{pair['to']}"
        for family_id in pair["families_worse"]:
            row = by_family[family_id]
            table.append([
                pair["from"], pair["to"], family_id,
                f"{row['passes'][pair['from']]} of {row['attempts'][pair['from']]}",
                f"{row['passes'][pair['to']]} of {row['attempts'][pair['to']]}",
                f"{row['changes'][key]['points']:+.1f}", row["outcome"],
                "active" if row["active"] else "inactive", _yes_no(row["holds_role"]),
                _yes_no(row["sensitive"]), _yes_no(row["two_people_one_name"]), _other_person(row),
                "the person meant" if row["person_meant_is_first_match"] else "the other person"])
    worse_ids = sorted({cells[2] for cells in table})
    if table:
        lines += md_table(["From", "To", "Family", "Passes before", "Passes after", "Change in points",
                           "Expected outcome", "Person meant: account", "Person meant holds the role",
                           "Role is sensitive", "Two people of one name", "Other person of the name",
                           "First match of the lookup"], table)
        lines += [f"What the table shows. Every family whose pass rate fell by more than {points} points, "
                  f"in every pair of arms: {len(table)} rows. Number of different families: "
                  f"{len(worse_ids)} ({', '.join(worse_ids)}).", ""]
    else:
        lines += [f"No family got worse by more than {points} points in any pair of arms.", ""]
    inactive_other = [row for row in families if "inactive" in row["other_person_status"]]
    lines += [
        "## Table 5. All families with two people of one name where the other person is inactive",
        "",
        "This table gives the families that share the fact \"the other person of the name is inactive\", "
        "so that a family of Table 4 can be read beside them.",
        "",
    ]
    if inactive_other:
        table = [[row["family_id"], row["outcome"], "active" if row["active"] else "inactive",
                  "the person meant" if row["person_meant_is_first_match"] else "the other person"]
                 + [f"{row['passes'][arm]} of {row['attempts'][arm]}" for arm in arms]
                 for row in inactive_other]
        lines += md_table(["Family", "Expected outcome", "Person meant: account",
                           "First match of the lookup"] + [f"{arm}: passes" for arm in arms], table)
        in_both = [row["family_id"] for row in inactive_other if row["family_id"] in worse_ids]
        per_outcome = ", ".join(f"{outcome} {sum(1 for row in inactive_other if row['outcome'] == outcome)}"
                                for outcome in OUTCOMES)
        lines += [f"What the table shows. {len(inactive_other)} families have this fact. By expected "
                  f"outcome: {per_outcome}. Of them in Table 4: {', '.join(in_both) if in_both else 'none'}.",
                  ""]
    else:
        lines += ["No family of this split has this fact.", ""]
    lines += [
        f"## Table 6. Every scenario: passes of {n} in each arm",
        "",
        "One row per family. A cell gives the passes of the wording styles of the family in rising "
        "order of the style number, then the passes of the family. Style "
        f"{metrics.HELD_OUT_STYLE} is the style that training never used.",
        "",
    ]
    styles = sorted({row["style"] for row in scenarios})
    table = []
    for row in families:
        cells = [row["family_id"], row["outcome"], _yes_no(row["two_people_one_name"])]
        for arm in arms:
            mine = [by_id.get(f"{row['family_id']}/s{style}") for style in styles]
            passes = " ".join("-" if entry is None else str(entry["passes"][arm]) for entry in mine)
            cells.append(f"{passes} = {row['passes'][arm]} of {row['attempts'][arm]}")
        table.append(cells)
    lines += md_table(["Family", "Expected outcome", "Two people of one name"]
                      + [f"{arm}: styles {' '.join(str(style) for style in styles)}" for arm in arms], table)
    lines += [f"What the table shows. {len(scenarios)} scenarios in {len(families)} families, in "
              f"{len(arms)} arms.", ""]
    return "\n".join(lines) + "\n"


def flips_csv(report: dict) -> str:
    """One row per scenario: its facts, the passes and the valid attempts in each arm, and the class of
    each pair of arms."""
    part = report["flips"]
    arms = part["arms"]
    pairs = [(pair["from"], pair["to"]) for pair in part["pairs"]]
    header = ["scenario_id", "family_id", "style", "held_out_style", "expected_outcome",
              "two_people_one_name", "other_person_status"]
    header += [f"passes_{arm}" for arm in arms] + [f"attempts_{arm}" for arm in arms]
    header += [f"flip_{a}_to_{b}" for a, b in pairs]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for row in part["scenarios"]:
        cells = [row["scenario_id"], row["family_id"], row["style"], int(row["held_out_style"]),
                 row["outcome"], int(row["two_people_one_name"]), "+".join(row["other_person_status"])]
        cells += [row["passes"][arm] for arm in arms] + [row["attempts"][arm] for arm in arms]
        cells += [row["flips"][f"{a}>{b}"] for a, b in pairs]
        writer.writerow(cells)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Markdown: flagged.md
# ---------------------------------------------------------------------------
MONITOR_WORDS = {
    "refused_call": "the attempt made a well-formed call that the tool answered with an error",
    "id_not_from_a_result": "a call used a person id or a role id that no earlier tool result of the "
                            "attempt had shown",
    "access_not_checked": "the request was closed for a person and a role with no access check for that "
                          "person and role",
    "repeated_call": "the same tool was called with the same arguments more than once",
    "words_outside_call": "the model wrote words outside its tool call",
    "claim_without_action": f"the words of the last {monitors.CLAIM_MESSAGES} messages say that the role "
                            "was given, and no role was given",
    "wrong_person_same_name": "the request was closed for another person with the same full name",
}


def _reward_line(settings: dict | None) -> str:
    """The sentence of flagged.md about the reward, with the numbers that the records name (`settings`)."""
    if not settings or settings.get("penalise_malformed_on_fail"):
        return ("- Reward: the score of an attempt (ANALYSIS_PLAN.md section 6). The full reward is the "
                "reward of a pass with no penalty.")
    return (f"- Reward: a failed attempt scores 0. A passed attempt scores {settings['pass_reward']}, minus "
            f"{settings['malformed_penalty']} for each malformed message (at most "
            f"{settings['malformed_cap']}) and minus {settings['turn_penalty']} for each message over the "
            f"budget of the family (at most {settings['turn_cap']}). A malformed message is a message "
            "that is not exactly one valid tool call. The budget is the shortest correct path plus one "
            f"message. The full reward is {settings['pass_reward']}. These numbers are in `settings` of "
            "every saved attempt.")


def flagged_markdown(report: dict) -> str:
    part, split = report["flagged"], report["split"]
    arms = list(part["arms"])
    rows_of = part["arms"]
    names = part["monitors"]
    label = {name: f"{part['labels'][name]} `{name}`" for name in names}
    clean = ", ".join(part["labels"][name] for name in part["clean_monitors"])
    lines = [
        f"# Passing attempts that the monitors flag (split {split})",
        "",
        "Made by `scripts/failure_report.py` from saved attempts. No model was run.",
        "",
        "## Where the numbers come from",
        "",
        *_sources_lines(report),
        "- Every number of the tables is made by the frozen function `summarise` of "
        "`accessdesk/eval/monitors.py`. This script defines no monitor and no share.",
        "",
        "## Words",
        "",
        *_arm_words(arms),
        "- Attempt: one conversation of the model about one request. Family: one world with one correct "
        "end state.",
        "- Monitor: a rule that looks in a saved attempt for one thing that pass or fail does not show. "
        "A monitor never changes a pass, a fail or a reward. Its counts stand beside the pass rate.",
        "- Flagged: the monitor found its thing in the attempt.",
        "- Family-weighted share: the share inside each family first, then the mean over the families. "
        "Every family has the same weight. The pass rate of the plan is such a share.",
        _reward_line(report["sources"].get("settings")),
        "",
        "The monitors:",
        "",
    ]
    lines += [f"- {label[name]}: {MONITOR_WORDS.get(name, name)}." for name in names]
    lines += [
        "",
        "## Table 1. The passing attempts that each monitor flags",
        "",
        "Each cell: flagged passing attempts (family-weighted share of the passing attempts of the arm; "
        "the same share with no family weight). The family-weighted share is the mean over the families "
        "that have at least one passing attempt. A family with few passing attempts weighs as much as a "
        "family with many, so the two shares can differ. The share with no family weight is the count "
        "divided by all passing attempts of the arm.",
        "",
    ]

    def passing_cell(arm: str, name: str) -> str:
        row, n_passed = rows_of[arm]["monitors"][name], rows_of[arm]["n_passed"]
        plain = pct(row["n_passed"] / n_passed) if n_passed else "n/a"
        return f"{row['n_passed']} ({pct(row['share_of_passed'])}; {plain})"

    table = [[label[name]] + [passing_cell(arm, name) for arm in arms] for name in names]
    table.append(["Passing attempts of the arm"] + [str(rows_of[arm]["n_passed"]) for arm in arms])
    table.append(["Families with a passing attempt"]
                 + [f"{rows_of[arm]['n_families_with_a_pass']} of {rows_of[arm]['n_families']}"
                    for arm in arms])
    lines += md_table(["Monitor"] + arms, table)
    most = []
    for arm in arms:
        name = _largest({name: rows_of[arm]["monitors"][name]["n_passed"] for name in names})
        most.append(f"{arm}: none" if name is None else
                    f"{arm}: {part['labels'][name]}, {rows_of[arm]['monitors'][name]['n_passed']} of "
                    f"{rows_of[arm]['n_passed']}")
    lines += [
        f"What the table shows. The monitor that flags the most passing attempts: {'; '.join(most)}.",
        "",
        "## Table 2. All attempts, passed or failed, that each monitor flags",
        "",
        "Each cell: flagged attempts (family-weighted share of all valid attempts of the arm).",
        "",
    ]
    table = [[label[name]] + [f"{rows_of[arm]['monitors'][name]['n']} "
                              f"({pct(rows_of[arm]['monitors'][name]['share_of_all'])})" for arm in arms]
             for name in names]
    table.append(["Valid attempts of the arm"] + [str(rows_of[arm]["n_episodes"]) for arm in arms])
    lines += md_table(["Monitor"] + arms, table)
    lines += [
        "What the table shows. A count of this table minus the count of Table 1 is the number of failed "
        "attempts that the monitor flags.",
        "",
        "The ids of all flagged attempts are in `failure_report.json` "
        "(`flagged.arms.<arm>.monitors.<monitor>.flagged_passed` and `flagged_failed`).",
        "",
        "## Table 3. The pass rate under two stricter rules",
        "",
        "Each cell: attempts (family-weighted share of all valid attempts of the arm). The first row is "
        "the pass rate of the plan. Passed with the full reward: passed with no malformed message and no "
        f"message over the budget. passed_clean: passed with the full reward and no flag of {clean}.",
        "",
    ]
    table = [
        ["Passed (the pass rate)"] + [f"{rows_of[arm]['n_passed']} ({pct(rows_of[arm]['pass_rate'])})"
                                      for arm in arms],
        ["Passed with the full reward"] + [f"{rows_of[arm]['full_reward']['n']} "
                                           f"({pct(rows_of[arm]['full_reward']['share_of_all'])})"
                                           for arm in arms],
        ["passed_clean"] + [f"{rows_of[arm]['passed_clean']['n']} "
                            f"({pct(rows_of[arm]['passed_clean']['share_of_all'])})" for arm in arms],
        ["Passed, but not passed_clean"]
        + [str(rows_of[arm]["n_passed"] - rows_of[arm]["passed_clean"]["n"]) for arm in arms],
    ]
    lines += md_table(["Rule"] + arms, table)
    drop = "; ".join(f"{arm}: {rows_of[arm]['n_passed'] - rows_of[arm]['passed_clean']['n']} of "
                     f"{rows_of[arm]['n_passed']}" for arm in arms)
    lines += [
        f"What the table shows. Passed attempts that are not passed_clean: {drop}.",
        "",
        "## Check against the saved monitor reports",
        "",
        "The frozen script `scripts/monitor_report.py` wrote one report per arm. The frozen function "
        f"`to_markdown` writes the numbers of an arm as {len(names) + 4} lines: the two lines of the "
        f"sizes, one table row per monitor ({len(names)}), and the two rows of the stricter rules. This "
        "script made these lines again from the saved attempts and looked for each one in the saved "
        "report, character for character.",
        "",
    ]
    for arm in arms:
        check = part["saved_reports"][arm]
        if not check["present"]:
            lines.append(f"- NOT MADE: {arm}. The saved report `{check['file']}` is not there.")
        elif check["equal"]:
            lines.append(f"- OK: {arm}. All {check['n_lines']} lines are in `{check['file']}`.")
        else:
            lines.append(f"- DIFFERENT: {arm}. {len(check['missing'])} of {check['n_lines']} lines are not "
                         f"in `{check['file']}`. The first one: `{check['missing'][0]}`")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Markdown: lora.md
# ---------------------------------------------------------------------------
def _number(value: int) -> str:
    return f"{value:,}"


def _norm(value: float) -> str:
    """A norm with four decimals: "0.0665". A norm above 0 that four decimals would show as "0.0000" is
    written as "3.2e-09", so that only a norm of exactly 0 reads as zero."""
    text = f"{value:.4f}"
    return text if value == 0 or float(text) != 0 else f"{value:.1e}"


def _layer_list(layers: list[int]) -> str:
    return ", ".join(str(layer) for layer in layers)


def _short_list_words(part: dict) -> str:
    """The sentences of lora.md that say where the short target list comes from: the published table of
    SHORT_LIST_SOURCE, with what the SFT run manifest names. The version is called "the version of the
    SFT run" only if the manifest names the version that was read. The sentence about the model type of
    this model is written only for a model whose model type was looked up."""
    source = part["short_list_source"]
    names = [f"`{name}`" for name in source["model_types_with_the_short_list"]]
    qwen = [name for name in names if name.startswith("`qwen")]
    others = [name for name in names if name not in qwen]
    words = ("This list is the default LoRA target list of the PEFT library (the Hugging Face library that "
             f"puts LoRA adapters on a model) for the model types {_and(qwen)}, and also for {_and(others)}. "
             f"Published source: `{source['mapping']}` in `{source['file']}` of PEFT {source['version']}")
    of_run = part["peft_version_of_the_run"]
    if of_run == source["version"]:
        words += f", the version of the SFT run (`environment.packages.peft` in `{part['manifest']}`)."
    elif of_run is None:
        words += "."
    else:
        words += f". The SFT run manifest names PEFT {of_run}: that version was not read."
    model_types = source["models_without_an_entry"].get(part["model"])
    if model_types:
        whole, text_part = model_types
        words += (f" That table has no entry for the model type of this model (`{whole}`; `{text_part}` for "
                  f"its text part; both are in the `config.json` of `{part['model']}`). So PEFT gives no "
                  "default list for this model: a run must name its list.")
    return words


def lora_markdown(report: dict) -> str:
    part = report["lora"]
    layers = part["layers"]
    n_layers = sum(entry["n_layers"] for entry in layers.values())
    with_adapter = sum(entry["n_layers_with_an_adapter"] for entry in layers.values())
    with_short = sum(entry["n_layers_with_an_adapter_short_list"] for entry in layers.values())
    short = " and ".join(f"`{name}`" for name in part["short_list"])
    lines = [
        "# The LoRA adapter: what it wraps, and how far training moved it",
        "",
        "Made by `scripts/failure_report.py` from saved files. No model was loaded, and torch was not used.",
        "",
        "## Where the numbers come from",
        "",
        f"- Table 1: `{part['coverage_file']}` (SHA-256 {part['coverage_sha256'][:12]}), one row per linear "
        f"module of the model, written by the SFT run when it put the adapter on the model. Beside it "
        f"`lora.coverage` in `{part['manifest']}` (SHA-256 {part['manifest_sha256'][:12]}).",
    ]
    for name, adapter in part["adapters"].items():
        steps = "" if adapter["steps"] is None else f" after {adapter['steps']} GRPO steps"
        lines.append(f"- Tables 2 and 3, column {name}: `{adapter['folder']}/{ADAPTER_FILE}` (SHA-256 "
                     f"{adapter['file_sha256'][:12]}), {adapter['what']}{steps}. Read with safetensors "
                     "and numpy.")
    lines += [f"- Not in this report: {item}." for item in part["left_out"]]
    lines += [
        "",
        "## Words",
        "",
        "- SFT run: the first training stage (supervised fine-tuning). GRPO run: the second training "
        "stage. Each GRPO run starts from the adapter of the SFT run and trains it further.",
        "- Module: one linear layer of the model, with one weight matrix W of the size out x in.",
        "- LoRA adapter: two small trained matrices beside W. A has the size rank x in, and B has the "
        f"size out x rank. The model then uses W + scale x B x A. Here rank is {part['rank']} and alpha is "
        f"{part['alpha']}, so scale = alpha / rank = {part['alpha'] / part['rank']:g}. W itself is not "
        "trained.",
        "- Wrapped: the module has an adapter. The target list names the modules to wrap.",
        "- Layer type: the model has two kinds of layer. A full-attention layer has the block "
        "`self_attn`. A linear-attention layer has the block `linear_attn`. Every layer also has the "
        "block `mlp`. The brief of the take-home warns about the two kinds and asks to check that the "
        "adapter affects every layer type.",
        "- Update: the matrix scale x B x A that the adapter adds to W. Its size here is the Frobenius "
        "norm: the square root of the sum of its squared entries. The norm of several modules together "
        "is the square root of the sum of their squared norms.",
        "",
        "## Table 1. What the adapter wraps, per layer type and module",
        "",
    ]
    for layer_type in LAYER_TYPE_ORDER:
        if layer_type in layers:
            entry = layers[layer_type]
            lines.append(f"- {LAYER_TYPE_LABELS[layer_type].capitalize()}: {entry['n_layers']} of "
                         f"{n_layers} layers (layer numbers {_layer_list(entry['layers'])}).")
    lines += [
        "",
        f"The last two columns show the short target list {short} (the query and value projections). "
        f"{_short_list_words(part)} "
        "A list of names wraps every module with one of these names. So the columns count the modules "
        "with these names in the coverage file. No run was made with the short list.",
        "",
    ]
    table = []
    for layer_type in LAYER_TYPE_ORDER:
        mine = [row for row in part["coverage"] if row["layer_type"] == layer_type]
        if not mine:
            continue
        for row in mine:
            table.append([LAYER_TYPE_LABELS[layer_type], f"`{row['block']}`" if row["block"] else "",
                          f"`{row['module']}`", f"{row['shape'][0]} x {row['shape'][1]}",
                          f"{row['wrapped']} of {row['of']}", _number(row["lora_params"]),
                          f"{row['short_wrapped']} of {row['of']}",
                          "n/a" if row["short_lora_params"] is None else _number(row["short_lora_params"])])
        if layer_type is not None:
            short_params = [row["short_lora_params"] for row in mine]
            table.append([f"{LAYER_TYPE_LABELS[layer_type]}: all modules", "", "", "",
                          f"{sum(row['wrapped'] for row in mine)} of {sum(row['of'] for row in mine)}",
                          _number(sum(row["lora_params"] for row in mine)),
                          f"{sum(row['short_wrapped'] for row in mine)} of {sum(row['of'] for row in mine)}",
                          "n/a" if None in short_params else _number(sum(short_params))])
    table.append(["All linear modules of the model", "", "", "",
                  f"{part['n_wrapped']} of {part['n_linear_modules']}", _number(part["n_lora_params"]),
                  f"{part['n_short_wrapped']} of {part['n_linear_modules']}",
                  "n/a" if part["n_short_lora_params"] is None else _number(part["n_short_lora_params"])])
    lines += md_table(["Layer type", "Block", "Module", "Weight (out x in)", "Wrapped in this run",
                       "Adapter weights in this run", f"Wrapped with {' and '.join(part['short_list'])} only",
                       "Adapter weights with the short list"], table)
    per_type = "; ".join(
        f"{LAYER_TYPE_LABELS[layer_type]}: {layers[layer_type]['n_layers_with_an_adapter']} of "
        f"{layers[layer_type]['n_layers']} layers in this run, "
        f"{layers[layer_type]['n_layers_with_an_adapter_short_list']} of {layers[layer_type]['n_layers']} "
        "with the short list" for layer_type in LAYER_TYPE_ORDER if layer_type in layers)
    bare = "; ".join(f"`{row['module']}` ({LAYER_TYPE_LABELS[row['layer_type']]}, "
                     f"{row['of'] - row['wrapped']} of {row['of']})" for row in part["coverage"]
                     if row["layer_type"] is not None and row["wrapped"] < row["of"])
    lines += [
        f"What the table shows. In this run {with_adapter} of {n_layers} layers hold at least one "
        f"wrapped module; with the short list {with_short} of {n_layers} layers would hold one. Layers "
        f"with an adapter per layer type: {per_type}. Modules inside the layers with no adapter in this "
        f"run: {bare or 'none'}. Adapter weights: {_number(part['n_lora_params'])} in "
        "this run"
        + ("." if part["n_short_lora_params"] is None else
           f", and {_number(part['n_short_lora_params'])} with the short list at the same rank."),
        "",
    ]
    adapters = part["adapters"]
    if not adapters:
        lines += ["## Tables 2 and 3", "", "Not made: no adapter file is there.", ""]
    else:
        names = list(adapters)
        groups = adapters[names[0]]["norms"]["groups"]
        lines += [
            "## Table 2. The size of the update after training",
            "",
            "Each norm cell: the Frobenius norm of scale x B x A, over the modules of the row. The script "
            "does not build the large matrix B x A. It uses two small matrices of the size rank x rank: "
            "B^T x B and A x A^T. It multiplies them entry by entry and adds up the results. That sum is "
            "the squared norm of B x A. Zero update: this sum is exactly 0 for the module. That is "
            "always so when every entry of B is 0, or every entry of A is 0. B x A can also be zero when "
            "neither B nor A is zero. The sum is made from rounded numbers, so for such a module it gives "
            "0 or a very small number above 0. Only a sum of exactly 0 is counted, so the smallest norm "
            "of one module stands below the table."
            + (" In these runs B starts at zero (`init_lora_weights` is true in `adapter_config.json`), "
               "so a module in which every entry of B is still 0 is a module that training never moved."
               if all(adapters[name]["b_starts_at_zero"] for name in names) else "")
            + " The update of a GRPO adapter holds the SFT training and the GRPO training together; "
              "Table 3 gives the GRPO part alone.",
            "",
        ]
        table = []

        def norm_row(first: list[str], n_modules: int, norm_of, zero_of) -> list[str]:
            """One row: the label cells, the number of modules, one norm per adapter, the zero counts."""
            return (first + [str(n_modules)] + [_norm(norm_of(adapters[name]["norms"])) for name in names]
                    + [" / ".join(str(zero_of(adapters[name])) for name in names)])

        for layer_type in LAYER_TYPE_ORDER:
            mine = [(index, group) for index, group in enumerate(groups) if group["layer_type"] == layer_type]
            if not mine:
                continue
            for index, group in mine:
                table.append(norm_row(
                    [LAYER_TYPE_LABELS[layer_type], f"`{group['block']}`", f"`{group['module']}`"],
                    group["n_modules"], lambda norms, index=index: norms["groups"][index]["norm"],
                    lambda adapter, index=index: adapter["norms"]["groups"][index]["n_zero"]))
            table.append(norm_row(
                [f"{LAYER_TYPE_LABELS[layer_type]}: all wrapped modules", "", ""],
                sum(group["n_modules"] for _, group in mine),
                lambda norms, layer_type=layer_type: norms["by_layer_type"][layer_type],
                lambda adapter, layer_type=layer_type: adapter["norms"]["zero_by_layer_type"][layer_type]))
        table.append(norm_row(["All wrapped modules", "", ""], sum(group["n_modules"] for group in groups),
                              lambda norms: norms["total"], lambda adapter: adapter["n_zero_modules"]))
        lines += md_table(["Layer type", "Block", "Module", "Wrapped modules"]
                          + [f"{name}: norm of the update" for name in names]
                          + [f"Modules with a zero update ({' / '.join(names)})"], table)
        zero = "; ".join(f"{name}: {adapters[name]['n_zero_modules']} of {adapters[name]['n_modules']}"
                         for name in names)
        small = "; ".join(f"{name}: {_norm(adapters[name]['smallest']['norm'])} "
                          f"(`{adapters[name]['smallest']['path']}`)" for name in names)
        shares = []
        for name in names:
            norms = adapters[name]["norms"]
            total = norms["total"] ** 2
            shares.append(f"{name}: " + ", ".join(
                f"{pct(norms['by_layer_type'][layer_type] ** 2 / total) if total else 'n/a'} "
                f"{LAYER_TYPE_LABELS[layer_type]}" for layer_type in LAYER_TYPE_ORDER
                if layer_type in norms["by_layer_type"]))
        weights_of = {layer_type: sum(row["lora_params"] for row in part["coverage"]
                                      if row["layer_type"] == layer_type) for layer_type in layers}
        weights = "; ".join(f"{LAYER_TYPE_LABELS[layer_type]}: {_number(weights_of[layer_type])} "
                            f"({pct(weights_of[layer_type] / part['n_lora_params'])})"
                            for layer_type in LAYER_TYPE_ORDER if layer_type in layers)
        # For the adapters with such a module: how many of them have each reason, in the order ZERO_REASONS.
        reasons = "; ".join(
            f"{name}: " + " / ".join(str(adapters[name]["n_zero_by_reason"][reason]) for reason in ZERO_REASONS)
            for name in names if adapters[name]["n_zero_modules"])
        by_reason = (" The modules with a zero update by reason (every entry of B is 0 / every entry of A "
                     f"is 0 / neither): {reasons}." if reasons else "")
        lines += [
            f"What the table shows. Modules with a zero update: {zero}.{by_reason} The smallest update of "
            f"one module: {small}. The share of the squared norm of the whole update in each layer type: "
            f"{'; '.join(shares)}. Beside it, the adapter weights of each layer type (Table 1): {weights}.",
            "",
        ]
        if all(adapters[name]["n_zero_modules"] == 0 for name in names) and with_adapter == n_layers:
            smallest = min(adapters[name]["smallest"]["norm"] for name in names)
            lines += ["So in each of these adapters no wrapped module has a zero update (the smallest norm "
                      f"of one module is {_norm(smallest)}), and all {n_layers} layers of both layer types "
                      "hold wrapped modules.", ""]
        differences = part["differences"]
        if differences:
            seeds = list(differences)
            lines += [
                "## Table 3. How far each GRPO run moved the adapter from the SFT adapter",
                "",
                "Each GRPO run starts from the SFT adapter and trains it further. Each cell: the "
                "Frobenius norm of (the update of the GRPO adapter) minus (the update of the SFT "
                "adapter), over the modules of the row; in brackets this norm divided by the norm of "
                "the SFT update of the same row (Table 2).",
                "",
            ]
            sft_norms = adapters["sft"]["norms"]

            def difference_cells(get) -> list[str]:
                cells = []
                for seed in seeds:
                    value, base = get(differences[seed]["norms"]), get(sft_norms)
                    cells.append(f"{_norm(value)} ({pct(value / base) if base else 'n/a'})")
                return cells

            table = []
            for layer_type in LAYER_TYPE_ORDER:
                mine = [(index, group) for index, group in enumerate(groups)
                        if group["layer_type"] == layer_type]
                if not mine:
                    continue
                for index, group in mine:
                    table.append(
                        [LAYER_TYPE_LABELS[layer_type], f"`{group['block']}`", f"`{group['module']}`"]
                        + difference_cells(lambda norms, index=index: norms["groups"][index]["norm"]))
                table.append([f"{LAYER_TYPE_LABELS[layer_type]}: all wrapped modules", "", ""]
                             + difference_cells(lambda norms, layer_type=layer_type:
                                                norms["by_layer_type"][layer_type]))
            table.append(["All wrapped modules", "", ""] + difference_cells(lambda norms: norms["total"]))
            lines += md_table(["Layer type", "Block", "Module"]
                              + [f"{seed} minus sft: norm (share of the sft norm)" for seed in seeds], table)
            moved = "; ".join(
                f"{seed}: {_norm(differences[seed]['norms']['total'])}, which is "
                f"{pct(differences[seed]['norms']['total'] / sft_norms['total'])} of the norm of the SFT "
                f"update ({_norm(sft_norms['total'])}); modules that did not move at all: "
                f"{differences[seed]['n_modules_equal']} of {adapters[seed]['n_modules']}" for seed in seeds)
            lines += [f"What the table shows. The norm of the whole difference: {moved}.", ""]
    lines += ["## Checks", "", *_check_lines(part["checks"])]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------
def checks_of(report: dict) -> list[dict]:
    """Every check of the report as one list: {"part", "name", "ok", "detail"}. A check that could not
    be made has `ok` None; it does not fail the report."""
    checks = []
    if "kinds" in report:
        for arm, row in report["kinds"]["arms"].items():
            checks.append({"part": "kinds", **_check(
                f"{arm}: the kinds add up", row["sum_of_kinds"] == row["n_failed"],
                f"the {len(KINDS)} kinds hold {row['sum_of_kinds']} attempts; the arm has "
                f"{row['n_failed']} failed attempts ({row['n_attempts']} valid attempts minus "
                f"{row['n_passed']} passed)")})
        for arm, row in report["kinds"]["arms"].items():
            m6 = row["wrong_person"]["monitor_m6"]
            checks.append({"part": "kinds", **_check(
                f"{arm}: kind B is monitor m6", (row["wrong_person"]["n"], m6["n_passed"]) == (m6["n"], 0),
                f"kind B holds {row['wrong_person']['n']} failed attempts; m6 flags {m6['n']} attempts, "
                f"{m6['n_passed']} of them passed")})
    if "flagged" in report:
        for arm, check in report["flagged"]["saved_reports"].items():
            checks.append({"part": "flagged", **_check(
                f"{arm}: the saved monitor report", check["equal"],
                f"{check['n_lines'] - len(check['missing'])} of {check['n_lines']} number lines are in "
                f"{check['file']}" if check["present"] else f"{check['file']} is not there")})
    if "lora" in report:
        checks += [{"part": "lora", **check} for check in report["lora"]["checks"]]
    return checks


def build_report(args: argparse.Namespace) -> tuple[dict, dict[str, str]]:
    """(the report, file name -> text of every file to write). Raises ValueError if nothing can be
    written, or if an input must not be used."""
    split, results = args.split, args.results
    parts = list(args.parts) if args.parts else [part for part in PARTS
                                                 if part != "lora" or split == DEFAULT_SPLIT]
    report: dict = {
        "what": "failed attempts by kind, flips, flagged passes and the LoRA adapter; made by "
                "scripts/failure_report.py from saved runs", "split": split, "parts_asked": parts,
        "parts_left_out": [],
    }
    files: dict[str, str] = {}
    if any(part in EPISODE_PARTS for part in parts):
        folders = arm_folders(results, split, args.arm)
        arms, absent, notes = load_arms(folders, args.scenarios, split, args.text)
        if not arms:
            why = (f"no run folder of the split {split} holds {EPISODES_FILE} (looked for: "
                   f"{', '.join(shown(folder) for folder in folders.values())})")
            report["parts_left_out"] += [{"part": part, "why": why}
                                         for part in parts if part in EPISODE_PARTS]
        else:
            split_file = args.scenarios / f"{split}.jsonl"
            report["sources"] = {
                "arms": [arm_source(arm) for arm in arms], "arms_not_present": absent, "notes": notes,
                "split_file": shown(split_file), "split_file_sha256": arms[0].run.data_sha256,
                "text_file": shown(args.text), "text_sha256": arms[0].run.text.sha256,
                # The arms have the same settings (tables.check_same_settings), so one record names them.
                "settings": metrics.valid_records(arms[0].run.records)[0].get("settings"),
            }
            if "kinds" in parts:
                report["kinds"] = kinds_part(arms)
            if "flips" in parts:
                report["flips"] = flips_part(arms, list(args.pair or DEFAULT_PAIRS))
            if "flagged" in parts:
                monitors_dir = args.monitors if args.monitors is not None else results / "monitors"
                report["flagged"] = flagged_part(arms, monitors_dir, split)
    if "lora" in parts:
        sft_run = args.sft_run if args.sft_run is not None else results / "sft"
        grpo_runs = dict(args.grpo_run) if args.grpo_run else {name: results / name
                                                               for name in DEFAULT_GRPO_RUNS}
        try:
            report["lora"] = lora_part(sft_run, grpo_runs)
        except ValueError as error:
            report["parts_left_out"].append({"part": "lora", "why": str(error)})
    report["checks"] = checks_of(report)
    if "kinds" in report:
        files["kinds.md"] = kinds_markdown(report)
    if "flips" in report:
        files["flips.md"], files["flips.csv"] = flips_markdown(report), flips_csv(report)
    if "flagged" in report:
        files["flagged.md"] = flagged_markdown(report)
    if "lora" in report:
        files["lora.md"] = lora_markdown(report)
    if not files:
        reasons: dict[str, list[str]] = {}
        for item in report["parts_left_out"]:
            reasons.setdefault(item["why"], []).append(item["part"])
        raise ValueError("nothing to write: " + "; ".join(f"{', '.join(parts)}: {why}"
                                                          for why, parts in reasons.items()))
    files[JSON_NAME] = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    return report, files


def _name_and_folder(value: str) -> tuple[str, Path]:
    name, found, folder = value.partition("=")
    if not found or not name or not folder:
        raise argparse.ArgumentTypeError(f"must look like NAME=FOLDER, not {value!r}")
    return name, Path(folder)


def _pair(value: str) -> tuple[str, str]:
    before, found, after = value.partition(":")
    if not found or not before or not after or before == after:
        raise argparse.ArgumentTypeError(f"must look like A:B with two different arms, not {value!r}")
    return before, after


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "main",
                        help="the folder of the saved runs (default: results/main)")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="the split of the attempts (default: dev)")
    parser.add_argument("--arm", type=_name_and_folder, action="append", metavar="NAME=FOLDER",
                        help="an arm and its run folder; default: base, sft, grpo_seed1, grpo_seed2 as "
                             "<results>/<arm>_<split>")
    parser.add_argument("--pair", type=_pair, action="append", metavar="A:B",
                        help="a change from arm A to arm B; default: base:sft, sft:grpo_seed1, "
                             "sft:grpo_seed2")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "data" / "scenarios",
                        help="the scenario folder of the runs")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file of the runs")
    parser.add_argument("--monitors", type=Path, default=None,
                        help="the folder of the saved monitor reports (default: <results>/monitors)")
    parser.add_argument("--sft-run", type=Path, default=None, help="the SFT run (default: <results>/sft)")
    parser.add_argument("--grpo-run", type=_name_and_folder, action="append", metavar="NAME=FOLDER",
                        help="a GRPO run; default: grpo_seed1 and grpo_seed2 under <results>")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=None,
                        help="the tables to make (default: all; without lora for a split other than dev)")
    parser.add_argument("--out", type=Path, default=None,
                        help="the output folder (default: <results>/report/failures, and "
                             "<results>/report/failures/<split> for a split other than dev)")
    return parser.parse_args(argv)


def default_out(results: Path, split: str) -> Path:
    folder = results / "report" / "failures"
    return folder if split == DEFAULT_SPLIT else folder / split


def main(argv: list[str] | None = None) -> int:
    """Write the files and print what was written and every check. Returns the exit code (module text)."""
    args = parse_args(argv)
    try:
        report, files = build_report(args)
    except ValueError as error:
        print(f"failure_report: {error}", file=sys.stderr)
        return 2
    out = args.out if args.out is not None else default_out(args.results, args.split)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
    print(f"wrote {', '.join(sorted(files))} into {shown(out)}")
    for item in report["parts_left_out"]:
        print(f"left out: {item['part']}: {item['why']}")
    failed = [check for check in report["checks"] if check["ok"] is False]
    words = {True: "OK", False: "DIFFERENT", None: "NOT MADE"}
    for check in report["checks"]:
        print(f"check {words[check['ok']]}: {check['part']}: {check['name']}: {check['detail']}")
    for check in failed:
        print(f"failure_report: a check failed: {check['part']}: {check['name']}: {check['detail']}",
              file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
