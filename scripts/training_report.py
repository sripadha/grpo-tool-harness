"""The training report: tables about the GRPO runs, made from their logs and their saved rollouts.
Written after the freeze. It only reads saved files: no model, no GPU, no torch, and no saved run is changed.

    .venv/bin/python scripts/training_report.py

Without an option it reads the two GRPO runs of results/main and writes into results/main/report/training/:

    steps_seed1.csv, steps_seed2.csv   one row per training step of a run (steps_<NAME>.csv); summary.md
                           explains each column
    summary.md             per run: attempts, failed attempts, kept groups, trained attempts, loss tokens,
                           the pass rate in blocks of 10 steps (all, and per expected outcome), hours
    signal.md              which groups gave the learning signal: per expected outcome, per kind of family
                           (its three rule facts; two people of one name or not), and the groups that were
                           kept only for a penalty among passing attempts
    std_division.md        an offline calculation on the kept groups: the advantages as the trainer made
                           them, and as a trainer that divides by the standard deviation of the group would
    cost.md                GPU hours and USD by stage, from the spend ledger and the run manifests
    training_report.json   every number of the four .md files, and the step rows

--run NAME FOLDER    one GRPO run: a short name (letters, digits, "_") and its folder. Give it once per run.
                     Default: seed1 results/main/grpo_seed1 and seed2 results/main/grpo_seed2
--main DIR           the folder of the saved runs of stage 1 and of the checks (default results/main)
--calibration DIR    the folder of the calibration runs (default results/calibration)
--ledger FILE        the spend ledger (default results/spend_ledger.csv)
--rentals FILE       what each rental was for, written by hand from the work log (columns instance_id,
                     rented_utc, stopped_utc, purpose; optional column stopped_from: empty or "ledger" if
                     stopped_utc is the time of the stop row of the ledger, "work log" if it is the time at
                     which the work log says that the machine stopped). Default: rentals.csv beside the
                     ledger. Optional
--test-record FILE   the record of the test run (default <main>/test_run_record.txt). Optional. Only the
                     time at the start of each of its lines is read, to find the rental of the run
--scenarios DIR      the scenario folder of the runs; only its train.jsonl is read (default data/scenarios)
--freeze-manifest FILE   the manifest of the freeze; only `frozen_at` is read (default MANIFEST.json)
--out DIR            the folder for the output files (default results/main/report/training)
--block-steps N      the number of steps in one block of the pass-rate tables (default 10)

What is read
    of each GRPO run   run_manifest.json, train_log.jsonl and rollouts/step_*.episodes.jsonl
    for the families   train.jsonl of --scenarios (never the dev or the test file)
    for the cost       the ledger; `frozen_at` of the freeze manifest; run_manifest.json of the other saved
                       runs (only `seconds`, the number of attempts and whether the run is complete);
                       checks/layout_gradients/layout_gradients.json (only the seconds of its stages);
                       the rentals file (--rentals); the time at the start of each line of the test
                       record (--test-record) and of the rehearsal records (test_run_record.txt below
                       --smoke)
No token file is read, and no attempt of the dev or the test set. A test rollout (base_test, sft_test,
<run>_test) enters the cost table only if its folder exists, and then only with the seconds and the
number of attempts of its run manifest.

Where the plan defines a number, the frozen function makes it: the pass rate (`pass_rate`,
`pass_rate_by`), the attempts per reward value (`reward_distribution`) and, for the checks of the log, the
functions of accessdesk/eval/metrics.py. The advantages are made again from the saved rewards by the
function of the trainer (`group_advantages`), through accessdesk/audit/formula.py.

The script writes nothing and stops with exit code 2, with the reason on stderr, if a number could not be
trusted:
  - a run folder has no run manifest, no GRPO config, no log or no saved rollout; the steps of the log are
    not 0, 1, 2, ...; or a step of the log has no rollout file (or the reverse);
  - the frozen audit function `audit_step` finds a difference between a saved rollout and its log line,
    or between the two versions of the reward or of the advantage function;
  - a log value that this script makes again from the rollout (pass rate, equal-reward shares,
    invalid-call rate, tokens and turns per attempt) differs from the log;
  - the totals of the run manifest differ from the log, or the log is not the file that the manifest names;
  - train.jsonl is not the file that the run manifest and the records name, a record is of a family that
    the file does not hold, or its outcome is not the one of its family;
  - two runs do not have the same step size, the same `tie_tol` or the same numbers of the reward: their
    numbers do not belong in one table;
  - the ledger lacks a column, or a row of it cannot be read; or the freeze manifest holds no `frozen_at`.

Two things that the tables say and do not hide: a run that the stop rule ended is reported with its
status and its fault, and a kept group of a step that ended in a fault is named as such (that step made
no update). Every amount in USD is the ledger's estimate; the invoice is the authority.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.audit import formula, load_script  # noqa: E402
from accessdesk.config import DEFAULT as DEFAULT_CONFIG  # noqa: E402
from accessdesk.env import generator  # noqa: E402
from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402
from accessdesk.train import grpo  # noqa: E402
from accessdesk.types import OUTCOMES  # noqa: E402

SCRIPT = "scripts/training_report.py"
TRAIN_SPLIT = "train"
TRAIN_FILE = "train.jsonl"                      # the only scenario file that this script opens
DEFAULT_RUNS = (("seed1", "results/main/grpo_seed1"), ("seed2", "results/main/grpo_seed2"))
JSON_NAME = "training_report.json"
RUN_NAME = re.compile(r"[A-Za-z0-9_]+")
SHOWN = 5                                       # problems that a refusal prints in full
# The kinds of a kept group and of a dropped group, by what its valid attempts did.
KEPT_TYPES = ("pass_and_fail", "penalty_only", "all_fail")
DROPPED_TYPES = ("dropped_all_pass", "dropped_all_fail", "dropped_lost_attempts", "dropped_other")
# The division by the standard deviation. The main form: the standard deviation of the valid rewards of the
# group with n in the divisor, and nothing added. The check form: n - 1 in the divisor and 0.0001 added.
MAIN_FORM = {"ddof": 0, "eps": 0.0}
CHECK_FORM = {"ddof": 1, "eps": 1e-4}
VERSIONS = ("trainer", "divided", "divided_check")
# The values of a log line that this script makes again from the saved rollout (the frozen audit function
# compares the counts; see LOG_COMPARED of accessdesk/audit/formula.py).
LOG_FIELDS_COMPARED = ("pass_rate", "zero_variance_exact", "zero_variance_pass_fail", "invalid_call_rate",
                       "turns_per_episode", "tokens_per_episode", "tokens_per_turn")
# The shares of the frozen function `zero_variance_share` that are added up over the steps, as counts.
ZERO_VARIANCE_KEYS = ("equal_reward", "equal_reward_all_pass", "equal_reward_all_fail", "equal_passfail")
# The training code that defines the advantage. Its hashes of today are compared with those in the run
# manifest, because this code is not frozen: each run records it.
ADVANTAGE_CODE = ("accessdesk/train/grpo_math.py", "accessdesk/train/grpo.py")
# The numbers of the reward (fields of the run config). The words of the output files are made from them.
REWARD_KEYS = ("pass_reward", "malformed_penalty", "malformed_cap", "turn_penalty", "turn_cap",
               "penalise_malformed_on_fail")
# Where the version with division is published. Both were read on 2026-11-19: the paper as
# https://arxiv.org/html/2402.03300v3 ("these rewards are normalized by subtracting the group average and
# dividing by the group standard deviation"), the code as trl/trainer/grpo_trainer.py of the main branch
# of github.com/huggingface/trl ("std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)" and
# "advantages = advantages / (std_rewards + 1e-4)", the latter unless scale_rewards is "none").
DIVISION_SOURCES = (
    "The paper that introduced GRPO has this form (Shao et al., DeepSeekMath, arXiv:2402.03300, section "
    "4.1.2). One public code that has it is the `GRPOTrainer` of the Hugging Face TRL library: with its "
    "option `scale_rewards` set to `group`, it divides by the standard deviation of the group plus 0.0001.")

# The columns of steps_<name>.csv: (name, what it is and its unit, where it comes from).
STEP_COLUMNS = (
    ("step", "the number of the training step, from 0", "log"),
    ("scenarios", "the scenarios of the step, as family id / wording style, with a space between them",
     "log `scenarios`"),
    ("attempts", "attempts sampled in the step", "log `n_episodes`"),
    ("harness_errors", "attempts that the harness lost; they are in no other number",
     "log `n_harness_errors`"),
    ("failed_attempts", "valid attempts that did not pass", "rollout file"),
    ("mean_reward", "mean over the valid attempts of reward / 100 (0 to 1)", "log"),
    ("pass_rate", "pass rate of the step's attempts (the frozen function `pass_rate`; 0 to 1)", "log"),
    ("groups", "groups in the step (scenarios per step)", "run manifest"),
    ("kept_groups", "groups whose rewards are not all equal", "log"),
    ("kept_pass_and_fail", "kept groups with at least one pass and at least one fail", "rollout file"),
    ("kept_penalty_only", "kept groups in which every valid attempt passed", "rollout file"),
    ("zero_variance_exact", "share of the groups whose attempts all have exactly the same reward (0 to 1)",
     "log"),
    ("zero_variance_pass_fail", "share of the groups in which all attempts passed or all failed (0 to 1)",
     "log"),
    ("trained_attempts", "valid attempts of the kept groups", "log `n_trained_episodes`"),
    ("loss_tokens", "tokens that the model wrote in the trained attempts", "log `n_loss_tokens`"),
    ("loss", "the GRPO loss of the step (the number that the update makes smaller); 0 when no group is kept",
     "log"),
    ("grad_norm", "size of the gradient before clipping; 0 when no group is kept", "log"),
    ("logprob_gap_mean", "mean absolute difference, over the loss tokens, between the log-probability "
                         "that the sampler saved and the one of the training pass; empty when no group "
                         "is kept", "log `logprob_gap.mean`"),
    ("invalid_call_rate", "malformed assistant messages divided by all assistant messages (0 to 1)", "log"),
    ("tokens_per_attempt", "tokens that the model wrote, per valid attempt", "log `tokens_per_episode`"),
    ("tokens_per_turn", "tokens that the model wrote, per assistant message", "log"),
    ("turns_per_attempt", "assistant messages per valid attempt", "log `turns_per_episode`"),
    ("rollout_seconds", "seconds of sampling in the step", "log"),
    ("seconds", "seconds of the whole step: sampling and update", "log"),
    ("fault", "the reason if the stop rule ended the run at this step; else empty", "log"),
)

# The cost table: the folders of each stage, under --main.
STAGE_1 = (
    ("base rollouts (dev, train, second desk)", ("base_dev", "base_train", "base_shift_differ",
                                                 "base_shift_control")),
    ("SFT (choice of the data, training)", ("sft_data", "sft")),
    ("sft rollouts (dev, train, second desk)", ("sft_dev", "sft_train", "sft_shift_differ",
                                                "sft_shift_control")),
)
CHECK_RUNS = (("overfit check", "checks/overfit"), ("pilot", "checks/pilot"))
LAYOUT_FILE = "checks/layout_gradients/layout_gradients.json"
TEST_ARMS = ("base_test", "sft_test")           # and <run folder>_test of each GRPO run
LEDGER_COLUMNS = ("time_utc", "event", "instance_id", "gpu", "dph_usd", "bandwidth_est_usd", "projected_usd",
                  "actual_usd", "note")
# What the rental script writes in front of the note of a normal stop. It is cut from the note in the table.
USUAL_NOTE = re.compile(r"^[\d.]+ h x price \+ download estimate; an estimate, the invoice is the "
                        r"authority\.?\s*")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(part: float, whole: float) -> float | None:
    """part / whole, or None if the whole is 0."""
    return part / whole if whole else None


def label_of(name: str) -> str:
    """The name of a run in the tables: "seed 1" for seed1; any other name as it is."""
    match = re.fullmatch(r"seed(\d+)", name)
    return f"seed {match.group(1)}" if match else name


def parse_utc(text: str) -> datetime:
    """"2026-11-18T21:48:10Z" as a time in UTC. Raises ValueError for another form."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def hours_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600.0


def shown(path: Path) -> str:
    """The path as the report shows it (relative to the repository if it is inside it)."""
    return load_script("monitor_report").shown_path(path)


def sha256_of(path: Path) -> str:
    return load_script("monitor_report").sha256_of_file(path)


# ---------------------------------------------------------------------------
# The families of the train split
# ---------------------------------------------------------------------------
def family_kind(family: dict) -> dict:
    """What the tables of signal.md group a family by: its expected outcome, its three rule facts, the
    name of its kind (`kind_name` of the generator) and whether its world holds two people of one name.

    A family record of the mixed design holds its facts; one of the default design holds none, and then
    the facts are those of its outcome (`FACTS_OF_OUTCOME` of the generator)."""
    outcome = family["outcome"]
    if family.get("facts") is None:
        facts = generator.FACTS_OF_OUTCOME[outcome]
    else:
        facts = tuple(bool(family["facts"][key]) for key in generator.FACT_KEYS)
    return {"outcome": outcome, **dict(zip(generator.FACT_KEYS, facts)),
            "kind": generator.kind_name(outcome, facts), "same_name": bool(family["homonym"])}


def load_train_families(scenarios: Path) -> tuple[dict[str, dict], str]:
    """(family id -> the grouping of `family_kind`, the SHA-256 of train.jsonl). Only train.jsonl is read."""
    path = scenarios / TRAIN_FILE
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    families = load_families(path)
    wrong = sorted({family["split"] for family in families} - {TRAIN_SPLIT})
    if wrong:
        raise ValueError(f"{path} holds families of the split {wrong}: only the train split is read here")
    return {family["family_id"]: family_kind(family) for family in families}, sha256_of(path)


# ---------------------------------------------------------------------------
# One step of a run
# ---------------------------------------------------------------------------
def sampled_tokens(record: dict) -> int:
    """The tokens that the model wrote in one attempt: the sum of `n_tokens` over its turns."""
    return sum(turn["n_tokens"] or 0 for turn in record["turns"])


def log_field_values(records: list[dict]) -> dict:
    """The values of LOG_FIELDS_COMPARED, made again from the records of one step with the frozen functions
    of accessdesk/eval/metrics.py, as `rollout_stats` of the trainer makes them. Empty without a valid
    record."""
    valid = metrics.valid_records(records)
    if not valid:
        return {}
    zero = metrics.zero_variance_share(records)
    n_tokens = sum(sampled_tokens(record) for record in valid)
    n_turns = sum(len(record["turns"]) for record in valid)
    return {"pass_rate": metrics.pass_rate(records),
            "zero_variance_exact": zero["equal_reward"], "zero_variance_pass_fail": zero["equal_passfail"],
            "invalid_call_rate": metrics.invalid_call_rate(records),
            "turns_per_episode": metrics.turn_stats(records)["mean_turns"],
            "tokens_per_episode": n_tokens / len(valid), "tokens_per_turn": ratio(n_tokens, n_turns)}


def log_field_differences(log: dict, records: list[dict]) -> list[str]:
    """Every value of LOG_FIELDS_COMPARED that the log holds and that the saved rollout does not give."""
    again = log_field_values(records)
    found = []
    for key in LOG_FIELDS_COMPARED:
        if key not in log or key not in again:
            continue
        logged, made = log[key], again[key]
        same = logged == made if logged is None or made is None else abs(logged - made) <= formula.FLOAT_SLACK
        if not same:
            found.append(f"the log has {key} {logged!r}; the saved rollout gives {made!r}")
    return found


def source_problems(records: list[dict], families: dict[str, dict], train_sha256: str) -> list[str]:
    """Why the records of one step are not attempts on the families of train.jsonl: another split, another
    scenario file (`data_sha256`; a record without it names no file), an unknown family, or an outcome that
    is not the one of the family."""
    found = []
    splits = sorted({record["split"] for record in records} - {TRAIN_SPLIT})
    if splits:
        found.append(f"records of the split {splits}: a GRPO run trains on the train split only")
    named = {record["data_sha256"] for record in records if record.get("data_sha256")}
    if named - {train_sha256}:
        found.append(f"the records name another scenario file than {TRAIN_FILE} of --scenarios (SHA-256 "
                     f"{sorted(value[:12] for value in named - {train_sha256})})")
    for record in records:
        family = families.get(record["family_id"])
        if family is None:
            found.append(f"family {record['family_id']} is not in {TRAIN_FILE}")
            break
        if family["outcome"] != record["outcome"]:
            found.append(f"{record['scenario_id']}: the record has the outcome {record['outcome']}, its "
                         f"family has {family['outcome']}")
            break
    return found


def divided_by_std(rewards: np.ndarray, advantages: np.ndarray, ddof: int = 0,
                   eps: float = 0.0) -> np.ndarray:
    """The advantages of ONE kept group, divided by the standard deviation of the group's rewards.

    rewards      the rewards of the group on the scale of the trainer (reward / 100); NaN for an attempt
                 that the harness lost
    advantages   the advantages that the trainer's function gives for this group: reward minus the mean of
                 the valid rewards, and 0 for a lost attempt
    ddof, eps    the standard deviation is taken over the valid rewards with (n - ddof) in the divisor, and
                 eps is added to it

    Returns advantages / (standard deviation + eps), and 0 for a lost attempt. If the divisor is 0 (a group
    whose valid rewards are all equal: the trainer keeps no such group with a tie_tol of 0 or more), every
    value is 0."""
    valid = ~np.isnan(rewards)
    divided = np.zeros(advantages.shape, dtype=float)
    if valid.sum() <= ddof:
        return divided
    divisor = float(rewards[valid].std(ddof=ddof)) + eps
    if divisor > 0:
        divided[valid] = advantages[valid] / divisor
    return divided


def penalty_parts(record: dict) -> tuple[int, int]:
    """(points lost for malformed messages, points lost for turns over the budget) of a PASSED attempt.

    Each part is a difference of two values of the frozen reward function, with the config that the record
    names: the reward without that penalty, minus the reward with both penalties."""
    cfg = load_script("replay_check").config_of(record)
    reward = impl.main_version("reward")
    budget = record["turn_budget"]
    with_both = reward(True, record["n_malformed"], record["n_turns"], budget, cfg)
    malformed = reward(True, 0, record["n_turns"], budget, cfg) - with_both
    turns = reward(True, record["n_malformed"], 0, budget, cfg) - with_both
    return malformed, turns


def group_numbers(step: int, index: int, members: list[dict], family: dict, rewards: np.ndarray,
                  advantages: np.ndarray, kept: bool, group_size: int) -> dict:
    """Everything that the tables need of one group (the attempts of one scenario in one step).

    members      the records of the group, in file order
    family       the grouping of its family (`family_kind`)
    rewards      its row of `reward_matrix` of the trainer: reward / 100, NaN for a lost attempt
    advantages   its row of the advantages that `group_advantages` gives
    kept         whether `group_advantages` keeps the group

    `group_type` is one of KEPT_TYPES for a kept group and one of DROPPED_TYPES for a dropped one.
    `abs_advantage` holds, for each version of VERSIONS, the sum and the largest of the absolute advantages
    of the valid attempts; all 0 for a dropped group."""
    valid = [record for record in members if record["harness_error"] is None]
    n_passed = sum(1 for record in valid if record["passed"])
    n_failed = len(valid) - n_passed
    saved = [record["reward"] for record in valid]
    if kept:
        group_type = "penalty_only" if n_failed == 0 else "all_fail" if n_passed == 0 else "pass_and_fail"
    else:
        _, enough = formula.group_rule(saved, group_size)
        group_type = ("dropped_lost_attempts" if not enough else "dropped_all_pass" if n_failed == 0
                      else "dropped_all_fail" if n_passed == 0 else "dropped_other")
    nothing = np.zeros(advantages.shape, dtype=float)
    versions = {"trainer": advantages,
                "divided": divided_by_std(rewards, advantages, **MAIN_FORM) if kept else nothing,
                "divided_check": divided_by_std(rewards, advantages, **CHECK_FORM) if kept else nothing}
    penalised = []                              # of a penalty-only group: the parts of each lost reward
    if group_type == "penalty_only":
        for record in valid:
            malformed, turns = penalty_parts(record)
            if malformed or turns:
                penalised.append({"malformed_points": malformed, "turn_points": turns})
    return {
        "step": step, "group": index, "scenario_id": members[0]["scenario_id"],
        "family_id": members[0]["family_id"], "style": members[0]["style"], **family,
        "n_attempts": len(members), "n_valid": len(valid), "n_passed": n_passed, "n_failed": n_failed,
        "rewards": sorted(saved), "kept": bool(kept), "group_type": group_type,
        "sampled_tokens": sum(sampled_tokens(record) for record in valid),
        "abs_advantage": {version: {"sum": float(np.abs(values).sum()), "max": float(np.abs(values).max())}
                          for version, values in versions.items()},
        "zero_divisor": bool(kept and not np.any(versions["divided"])),
        "penalised": penalised,
    }


def step_row(log: dict, groups: list[dict], n_failed: int) -> dict:
    """One row of steps_<name>.csv: the columns of STEP_COLUMNS."""
    gap = log.get("logprob_gap")
    return {
        "step": log["step"], "scenarios": " ".join(log["scenarios"]),
        "attempts": log["n_episodes"], "harness_errors": log["n_harness_errors"], "failed_attempts": n_failed,
        "mean_reward": log.get("mean_reward"), "pass_rate": log.get("pass_rate"),
        "groups": len(groups), "kept_groups": log["kept_groups"],
        "kept_pass_and_fail": sum(1 for group in groups if group["group_type"] == "pass_and_fail"),
        "kept_penalty_only": sum(1 for group in groups if group["group_type"] == "penalty_only"),
        "zero_variance_exact": log.get("zero_variance_exact"),
        "zero_variance_pass_fail": log.get("zero_variance_pass_fail"),
        "trained_attempts": log["n_trained_episodes"], "loss_tokens": log["n_loss_tokens"],
        "loss": log.get("loss"), "grad_norm": log.get("grad_norm"),
        "logprob_gap_mean": None if gap is None else gap["mean"],
        "invalid_call_rate": log.get("invalid_call_rate"),
        "tokens_per_attempt": log.get("tokens_per_episode"), "tokens_per_turn": log.get("tokens_per_turn"),
        "turns_per_attempt": log.get("turns_per_episode"),
        "rollout_seconds": log.get("rollout_seconds"), "seconds": log.get("seconds"),
        "fault": log.get("fault"),
    }


# ---------------------------------------------------------------------------
# One run: reading and checking
# ---------------------------------------------------------------------------
def read_run(folder: Path, families: dict[str, dict], train_sha256: str) -> dict:
    """The saved steps of one GRPO run folder, checked. Raises ValueError with the reasons if a number of
    the run could not be trusted (see the module text).

    Returns `manifest` (the run manifest), `gcfg` (n_scenarios, group_size, tie_tol), `logs` (the log
    lines in step order), `groups` (`group_numbers` of every group), `attempts` (one small dict per
    attempt: what the frozen metric functions read, and its sampled tokens), `steps` (`step_row` of every
    step), `checks` (what was compared, in counts), `equal_reward_groups` (the counts that the frozen
    function `zero_variance_share` gives, applied to each step by itself and added up: `groups_counted`
    and one count for each key of ZERO_VARIANCE_KEYS) and `rollouts_run_again` (rollout files of steps
    that a restart ran again; they are in no table)."""
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{folder} has no run_manifest.json")
    manifest = read_json(manifest_path)
    gcfg, _ = formula.run_config(folder)
    n_scenarios, group_size = gcfg["n_scenarios"], gcfg["group_size"]
    if manifest.get("split", TRAIN_SPLIT) != TRAIN_SPLIT:
        raise ValueError(f"{folder}: the run manifest names the split {manifest['split']!r}, not train")
    if manifest.get("split_file_sha256") not in (None, train_sha256):
        raise ValueError(f"{folder}: {TRAIN_FILE} of --scenarios is not the file that the run manifest "
                         "names (split_file_sha256)")
    log_path = folder / grpo.LOG_NAME
    logs = formula.log_lines(log_path)
    rollouts = formula.saved_rollouts(folder)
    files = {step: path for path, step, log_file in rollouts if log_file == log_path}
    if not logs or not files:
        raise ValueError(f"{folder} has no {grpo.LOG_NAME} or no saved rollout "
                         "(rollouts/step_*.episodes.jsonl)")
    if sorted(logs) != list(range(len(logs))) or set(files) != set(logs):
        raise ValueError(f"{folder}: the log has the steps {sorted(logs)[:3]}... ({len(logs)} lines) and "
                         f"rollouts/ has {len(files)} files: every step from 0 on needs one of each")

    problems: list[str] = []
    groups: list[dict] = []
    attempts: list[dict] = []
    steps: list[dict] = []
    counts: Counter = Counter()
    equal: Counter = Counter({key: 0 for key in ("groups_counted", *ZERO_VARIANCE_KEYS)})
    for step in sorted(logs):
        log = logs[step]
        records = read_records(files[step])
        found = [f"{kind}: {detail}" for kind, detail in formula.audit_step(records, gcfg, log)[0]]
        found += source_problems(records, families, train_sha256)
        if not found:
            found += log_field_differences(log, records)
        if found:
            problems += [f"step {step}: {detail}" for detail in found]
            continue
        advantages, keep, _ = formula.advantages_again(records, gcfg)
        rewards = grpo.reward_matrix(records, n_scenarios, group_size)
        of_step = []
        for index in range(n_scenarios):
            members = records[index * group_size:(index + 1) * group_size]
            of_step.append(group_numbers(step, index, members, families[members[0]["family_id"]],
                                         rewards[index], advantages[index], bool(keep[index]), group_size))
        valid = metrics.valid_records(records)
        steps.append(step_row(log, of_step, sum(1 for record in valid if not record["passed"])))
        groups += of_step
        attempts += [{"step": step, "family_id": record["family_id"], "style": record["style"],
                      "sample": record["sample"], "outcome": record["outcome"], "passed": record["passed"],
                      "reward": record["reward"], "harness_error": record["harness_error"],
                      "sampled_tokens": sampled_tokens(record)} for record in records]
        zero = metrics.zero_variance_share(records)     # the frozen function, on the groups of this step
        for key in ZERO_VARIANCE_KEYS if zero["n_scenarios"] else ():
            equal[key] += round(zero[key] * zero["n_scenarios"])
        equal["groups_counted"] += zero["n_scenarios"]
        counts["steps_compared_with_the_log"] += 1
        counts["log_values_made_again"] += sum(1 for key in LOG_FIELDS_COMPARED if key in log)
        counts["records_that_name_the_scenario_file"] += sum(1 for record in records
                                                             if record.get("data_sha256"))
    if problems:
        more = "" if len(problems) <= SHOWN else f"; and {len(problems) - SHOWN} more"
        raise ValueError(f"{folder}: " + "; ".join(problems[:SHOWN]) + more)
    return {"manifest": manifest, "gcfg": gcfg, "logs": [logs[step] for step in sorted(logs)],
            "groups": groups, "attempts": attempts, "steps": steps, "checks": dict(counts),
            "equal_reward_groups": dict(equal), "rollouts_run_again": len(rollouts) - len(files)}


def manifest_problems(folder: Path, manifest: dict, totals: dict) -> tuple[list[str], dict]:
    """(where the run manifest does not fit the log and the rollouts that were read, what was compared).

    Compared: the totals of its `summary`, and the SHA-256 of the log that it names. A key that the
    manifest does not have is not compared; the second value names the keys that were
    (`summary_keys_compared`) and says whether the manifest names the log (`log_sha256_compared`) and the
    scenario file (`names_the_scenario_file`)."""
    summary = manifest.get("summary") or {}
    wanted = {"steps": totals["steps"], "groups": totals["groups"], "kept_groups": totals["kept_groups"],
              "episodes": totals["attempts"], "harness_errors": totals["harness_errors"],
              "trained_episodes": totals["trained_attempts"], "loss_tokens": totals["loss_tokens"]}
    found = [f"the run manifest has {key} {summary[key]!r}; the log gives {value!r}"
             for key, value in wanted.items() if key in summary and summary[key] != value]
    named = ((manifest.get("outputs") or {}).get("log") or {}).get("sha256")
    if named is not None and named != sha256_of(folder / grpo.LOG_NAME):
        found.append(f"{grpo.LOG_NAME} is not the file that the run manifest names (outputs.log.sha256)")
    return found, {"summary_keys_compared": [key for key in wanted if key in summary],
                   "log_sha256_compared": named is not None,
                   "names_the_scenario_file": manifest.get("split_file_sha256") is not None}


def reward_settings(manifest: dict) -> dict:
    """The numbers of the reward that the run was made with: the fields of REWARD_KEYS of the `config` in
    the run manifest; for a manifest without them, those of accessdesk/config.py."""
    config = {**DEFAULT_CONFIG.to_dict(), **(manifest.get("config") or {})}
    return {key: config[key] for key in REWARD_KEYS}


def advantage_code_state(manifest: dict) -> dict:
    """For each file of ADVANTAGE_CODE: "same" if its SHA-256 of today is the one that the run manifest
    recorded (`training_code.files`), "changed" if not, "not recorded" if the manifest names none."""
    recorded = (manifest.get("training_code") or {}).get("files") or {}
    state = {}
    for name in ADVANTAGE_CODE:
        if name not in recorded:
            state[name] = "not recorded"
        else:
            state[name] = "same" if sha256_of(ROOT / name) == recorded[name] else "changed"
    return state


# ---------------------------------------------------------------------------
# One run: the numbers of summary.md
# ---------------------------------------------------------------------------
def type_counts(groups: list[dict]) -> dict[str, int]:
    """The number of groups of each kind of KEPT_TYPES and DROPPED_TYPES."""
    found = Counter(group["group_type"] for group in groups)
    return {group_type: found.get(group_type, 0) for group_type in KEPT_TYPES + DROPPED_TYPES}


def attempt_numbers(attempts: list[dict]) -> dict:
    """Counts and means of a list of attempts (the small dicts of `read_run`): `attempts`, `valid`,
    `passed`, `failed`, `mean_reward` (mean over the valid attempts of reward / 100) and
    `sampled_tokens`, `tokens_per_attempt` (over the valid attempts)."""
    valid = metrics.valid_records(attempts)
    passed = sum(1 for attempt in valid if attempt["passed"])
    tokens = sum(attempt["sampled_tokens"] for attempt in valid)
    return {"attempts": len(attempts), "valid": len(valid), "passed": passed, "failed": len(valid) - passed,
            "mean_reward": ratio(sum(attempt["reward"] for attempt in valid), grpo.REWARD_SCALE * len(valid)),
            "sampled_tokens": tokens, "tokens_per_attempt": ratio(tokens, len(valid))}


def run_totals(read: dict) -> dict:
    """The totals of one run, from its log lines, its groups and its attempts."""
    logs, groups, zero = read["logs"], read["groups"], read["equal_reward_groups"]
    numbers = attempt_numbers(read["attempts"])
    kinds = type_counts(groups)
    kept = sum(kinds[group_type] for group_type in KEPT_TYPES)
    without = [log["step"] for log in logs if log["kept_groups"] == 0]
    with_fault = {log["step"] for log in logs if log.get("fault") is not None}
    trained = sum(log["n_trained_episodes"] for log in logs)
    loss_tokens = sum(log["n_loss_tokens"] for log in logs)
    seen: set[str] = set()
    repeated = None                 # the first step that samples a scenario of an earlier step
    for group in groups:
        if group["scenario_id"] in seen and repeated is None:
            repeated = group["step"]
        seen.add(group["scenario_id"])
    return {
        "steps": len(logs), "groups": len(groups), **numbers,
        "different_scenarios": len(seen), "first_step_with_a_scenario_of_an_earlier_step": repeated,
        "harness_errors": numbers["attempts"] - numbers["valid"],
        "failed_share_of_valid": ratio(numbers["failed"], numbers["valid"]),
        "kept_groups": kept, "kept_share_of_groups": ratio(kept, len(groups)),
        "dropped_groups": len(groups) - kept,
        "dropped_share_of_groups": ratio(len(groups) - kept, len(groups)),
        "groups_by_type": kinds,
        # The equal-reward share of the plan (the frozen function `zero_variance_share`), each step by
        # itself: counts of groups, and their shares of the groups that the function counted.
        "equal_reward_groups": {**zero, **{f"{key}_share": ratio(zero[key], zero["groups_counted"])
                                           for key in ZERO_VARIANCE_KEYS}},
        # A step that ended in a fault made no update: its groups with unequal rewards trained nothing.
        "kept_groups_in_steps_with_a_fault": sum(1 for group in groups
                                                 if group["kept"] and group["step"] in with_fault),
        "steps_without_kept_group": len(without), "steps_without_kept_group_list": without,
        "steps_without_kept_group_share_of_steps": ratio(len(without), len(logs)),
        # Of these steps: those for which the log holds the loss 0 and the gradient norm 0.
        "steps_without_kept_group_and_zero_gradient": sum(
            1 for log in logs if log["kept_groups"] == 0 and log.get("loss") == 0.0
            and log.get("grad_norm") == 0.0),
        "trained_attempts": trained, "trained_share_of_attempts": ratio(trained, numbers["attempts"]),
        "loss_tokens": loss_tokens,
        "loss_tokens_share_of_sampled_tokens": ratio(loss_tokens, numbers["sampled_tokens"]),
        "mean_reward_points": None if numbers["mean_reward"] is None
        else grpo.REWARD_SCALE * numbers["mean_reward"],
        "attempts_in_dropped_groups": sum(group["n_attempts"] for group in groups if not group["kept"]),
        "sampled_tokens_in_dropped_groups": sum(group["sampled_tokens"] for group in groups
                                                if not group["kept"]),
        "reward_values": {str(reward): count
                          for reward, count in metrics.reward_distribution(read["attempts"]).items()},
    }


def run_time(read: dict) -> dict:
    """The seconds of one run: of the whole script (run manifest) and of its steps (log)."""
    logs, manifest = read["logs"], read["manifest"]
    in_steps = sum(log["seconds"] for log in logs)
    sampling = sum(log["rollout_seconds"] for log in logs)
    without = sum(log["seconds"] for log in logs if log["kept_groups"] == 0)
    seconds = {"run_seconds": manifest.get("seconds"), "step_seconds": in_steps,
               "sampling_seconds": sampling, "update_seconds": in_steps - sampling,
               "seconds_in_steps_without_kept_group": without}
    return {**seconds, "started_utc": manifest.get("started_utc"), "ended_utc": manifest.get("ended_utc"),
            "sampling_share_of_step_seconds": ratio(sampling, in_steps),
            "share_of_step_seconds_in_steps_without_kept_group": ratio(without, in_steps),
            # The same times in hours, as the tables print them.
            "hours": {key.replace("_seconds", "").replace("seconds_", ""): None if value is None
                      else value / 3600.0 for key, value in seconds.items()}}


def largest(logs: list[dict], value) -> dict:
    """{"value", "step"} of the largest value(log) over the log lines where it is not None; both None if
    there is none."""
    found = [(value(log), log["step"]) for log in logs if value(log) is not None]
    if not found:
        return {"value": None, "step": None}
    top = max(found, key=lambda pair: (pair[0], -pair[1]))
    return {"value": top[0], "step": top[1]}


def stop_rule_numbers(read: dict, totals: dict) -> dict:
    """The numbers that the stop rule of the plan looks at (`find_fault` of the trainer), with the limits
    of the run's GRPO config, and the largest gradient norm with the clip."""
    logs = read["logs"]
    config = read["manifest"].get("grpo_config") or {}
    clip = config.get("max_grad_norm")
    return {
        "steps_with_a_fault": sum(1 for log in logs if log.get("fault") is not None),
        "harness_errors": totals["harness_errors"],
        "harness_error_rate": ratio(totals["harness_errors"], totals["attempts"]),
        "limit_harness_error_rate": config.get("max_harness_error_rate"),
        "largest_logprob_gap_mean": largest(logs, lambda log: (log.get("logprob_gap") or {}).get("mean")),
        "limit_logprob_gap_mean": config.get("max_logprob_gap"),
        "invalid_call_rate_step_0": logs[0].get("invalid_call_rate"),
        "largest_invalid_call_rate": largest(logs, lambda log: log.get("invalid_call_rate")),
        "limit_invalid_call_rate": config.get("max_invalid_call_rate"),
        "largest_grad_norm": largest(logs, lambda log: log.get("grad_norm")),
        "clip_grad_norm": clip,
        "steps_clipped": None if clip is None else sum(1 for log in logs
                                                       if log.get("grad_norm") is not None
                                                       and log["grad_norm"] > clip),
    }


def blocks_of(read: dict, block_steps: int) -> list[dict]:
    """The run in blocks of `block_steps` steps (the last block may be shorter). Per block: the counts of
    `attempt_numbers`; `pass_rate` and `pass_rate_by_outcome` by the frozen functions `pass_rate` and
    `pass_rate_by`; per outcome the passed and the valid attempts; the families and scenarios in the block;
    the groups by kind; and the steps with no kept group."""
    blocks = []
    n_steps = len(read["logs"])
    for first in range(0, n_steps, block_steps):
        last = min(first + block_steps, n_steps) - 1
        inside = lambda item: first <= item["step"] <= last           # noqa: E731
        attempts = [attempt for attempt in read["attempts"] if inside(attempt)]
        groups = [group for group in read["groups"] if inside(group)]
        logs = [log for log in read["logs"] if inside(log)]
        valid = metrics.valid_records(attempts)
        kinds = type_counts(groups)
        n_kept = sum(kinds[group_type] for group_type in KEPT_TYPES)
        numbers = attempt_numbers(attempts)
        by_outcome = metrics.pass_rate_by(attempts, "outcome") if valid else {}
        blocks.append({
            "first_step": first, "last_step": last, "steps": len(logs), **numbers,
            "failed_share_of_valid": ratio(numbers["failed"], numbers["valid"]),
            "mean_reward_points": None if numbers["mean_reward"] is None
            else grpo.REWARD_SCALE * numbers["mean_reward"],
            "pass_rate": metrics.pass_rate(attempts) if valid else None,
            "families": len({attempt["family_id"] for attempt in valid}),
            "scenarios": len({(attempt["family_id"], attempt["style"]) for attempt in valid}),
            "by_outcome": {outcome: {
                "pass_rate": by_outcome[outcome],
                "passed": sum(1 for attempt in valid if attempt["outcome"] == outcome and attempt["passed"]),
                "valid": sum(1 for attempt in valid if attempt["outcome"] == outcome)}
                for outcome in by_outcome},
            "groups": len(groups), "kept_groups": n_kept, "kept_share_of_groups": ratio(n_kept, len(groups)),
            "groups_by_type": kinds,
            "steps_without_kept_group": sum(1 for log in logs if log["kept_groups"] == 0),
        })
    return blocks


# ---------------------------------------------------------------------------
# One run: the numbers of signal.md
# ---------------------------------------------------------------------------
def tally(groups: list[dict], key, n_kept: int, n_failed: int) -> dict:
    """The groups, counted by `key(group)`. Per key: `groups` sampled, `attempts` (valid), `kept_groups`,
    one count for each kind of KEPT_TYPES, `failed_attempts`, and four shares: the key's groups as a share
    of all groups; the kept groups as a share of the key's sampled groups and of all kept groups of the
    run (`n_kept`); and the failed attempts as a share of all failed attempts of the run (`n_failed`).
    Keys in sorted order."""
    rows: dict = {}
    for group in groups:
        row = rows.setdefault(key(group), {"groups": 0, "attempts": 0, "kept_groups": 0,
                                           **{group_type: 0 for group_type in KEPT_TYPES},
                                           "failed_attempts": 0})
        row["groups"] += 1
        row["attempts"] += group["n_valid"]
        row["failed_attempts"] += group["n_failed"]
        if group["kept"]:
            row["kept_groups"] += 1
            row[group["group_type"]] += 1
    for row in rows.values():
        row["share_of_all_groups"] = ratio(row["groups"], len(groups))
        row["kept_share_of_its_groups"] = ratio(row["kept_groups"], row["groups"])
        row["share_of_all_kept_groups"] = ratio(row["kept_groups"], n_kept)
        row["share_of_all_failed_attempts"] = ratio(row["failed_attempts"], n_failed)
    return {name: rows[name] for name in sorted(rows)}


def facts_order(row: dict) -> tuple:
    """Sort key of a row by rule facts: the outcomes in the order of OUTCOMES; inside an outcome the kinds
    in the order of MIXED_KINDS of the generator."""
    facts = tuple(row[key] for key in generator.FACT_KEYS)
    kinds = [kind for kind, _ in generator.MIXED_KINDS.get(row["outcome"], ())]
    return (OUTCOMES.index(row["outcome"]) if row["outcome"] in OUTCOMES else len(OUTCOMES),
            kinds.index(facts) if facts in kinds else len(kinds), facts)


def signal_numbers(read: dict, families: dict[str, dict], totals: dict) -> dict:
    """Where the kept groups of one run came from (the content of signal.md for the run)."""
    groups = read["groups"]
    n_kept, n_failed = totals["kept_groups"], totals["failed"]
    def facts_key(item: dict) -> tuple:
        return (item["outcome"],) + tuple(item[key] for key in generator.FACT_KEYS)

    by_outcome = tally(groups, lambda group: group["outcome"], n_kept, n_failed)
    by_facts = tally(groups, facts_key, n_kept, n_failed)
    by_name = tally(groups, lambda group: group["same_name"], n_kept, n_failed)
    by_both = tally(groups, lambda group: facts_key(group) + (group["same_name"],), n_kept, n_failed)
    in_train = Counter(facts_key(family) for family in families.values())
    names_in_train = Counter(family["same_name"] for family in families.values())

    def facts_rows(tallied: dict, with_name: bool) -> list[dict]:
        rows = []
        for key, row in tallied.items():
            named = dict(zip(("outcome",) + generator.FACT_KEYS + (("same_name",) if with_name else ()), key))
            named["kind"] = generator.kind_name(named["outcome"],
                                                tuple(named[fact] for fact in generator.FACT_KEYS))
            if not with_name:
                named["train_families"] = in_train[key]
            rows.append({**named, **row})
        return sorted(rows, key=lambda row: facts_order(row) + ((row.get("same_name"),) if with_name else ()))

    penalty_only = [group for group in groups if group["group_type"] == "penalty_only"]
    penalised = [attempt for group in penalty_only for attempt in group["penalised"]]
    causes = Counter("both" if attempt["malformed_points"] and attempt["turn_points"]
                     else "malformed_message_only" if attempt["malformed_points"]
                     else "turns_over_budget_only" for attempt in penalised)
    pass_and_fail = [group for group in groups if group["group_type"] == "pass_and_fail"]
    failed_in_group = Counter(group["n_failed"] for group in pass_and_fail)
    kept_by_step: dict[int, list[str]] = {}
    for group in groups:
        if group["kept"]:
            kept_by_step.setdefault(group["step"], []).append(group["group_type"])
    only_penalty = sorted(step for step, kinds in kept_by_step.items() if set(kinds) == {"penalty_only"})
    return {
        "kept_groups": n_kept, "failed_attempts": n_failed,
        "by_outcome": [{"outcome": outcome, **by_outcome[outcome]}
                       for outcome in list(OUTCOMES) + sorted(set(by_outcome) - set(OUTCOMES))
                       if outcome in by_outcome],
        "by_facts": facts_rows(by_facts, with_name=False),
        "by_same_name": [{"same_name": flag, "train_families": names_in_train[flag], **by_name[flag]}
                         for flag in (True, False) if flag in by_name],
        "by_facts_and_same_name": facts_rows(by_both, with_name=True),
        "penalty_only": {
            "groups": len(penalty_only), "share_of_kept_groups": ratio(len(penalty_only), n_kept),
            "attempts": sum(group["n_valid"] for group in penalty_only),
            "penalised_attempts": len(penalised),
            "penalised_share_of_attempts": ratio(len(penalised),
                                                 sum(group["n_valid"] for group in penalty_only)),
            "penalised_by_cause": {cause: causes.get(cause, 0)
                                   for cause in ("malformed_message_only", "turns_over_budget_only", "both")},
            "points_lost_for_malformed_messages": sum(attempt["malformed_points"] for attempt in penalised),
            "points_lost_for_turns_over_budget": sum(attempt["turn_points"] for attempt in penalised),
        },
        "pass_and_fail": {
            "groups": len(pass_and_fail), "share_of_kept_groups": ratio(len(pass_and_fail), n_kept),
            "failed_attempts": sum(group["n_failed"] for group in pass_and_fail),
            "groups_by_number_of_failed_attempts": {str(count): failed_in_group[count]
                                                    for count in sorted(failed_in_group)},
        },
        "steps_with_a_kept_group": len(kept_by_step),
        "steps_with_a_kept_group_share_of_steps": ratio(len(kept_by_step), totals["steps"]),
        "steps_whose_kept_groups_are_all_penalty_only": len(only_penalty),
        "steps_whose_kept_groups_are_all_penalty_only_list": only_penalty,
    }


# ---------------------------------------------------------------------------
# One run: the numbers of std_division.md
# ---------------------------------------------------------------------------
def std_division_numbers(read: dict) -> dict:
    """The absolute advantages of the kept groups of one run, in the three versions of VERSIONS.

    by_type      for each kind of KEPT_TYPES: `groups`, `attempts` (valid), and per version the sum, the
                 mean per attempt and the largest absolute advantage
    total        per version: the sum over all kept groups
    penalty_only_share_of_total   per version: the sum of the penalty-only groups divided by the total
    ratio_of_means   per version: the mean of a pass-and-fail attempt divided by the mean of a
                 penalty-only attempt (None if one of the two kinds has no group)
    kept_groups_with_zero_divisor   kept groups whose standard deviation is 0 (possible only with a
                 tie_tol below 0); their divided advantages are counted as 0"""
    kept = [group for group in read["groups"] if group["kept"]]
    by_type = {}
    for group_type in KEPT_TYPES:
        chosen = [group for group in kept if group["group_type"] == group_type]
        n_attempts = sum(group["n_valid"] for group in chosen)
        sums = {version: sum(group["abs_advantage"][version]["sum"] for group in chosen)
                for version in VERSIONS}
        by_type[group_type] = {
            "groups": len(chosen), "share_of_kept_groups": ratio(len(chosen), len(kept)),
            "attempts": n_attempts, "sum_abs_advantage": sums,
            "mean_abs_advantage": {version: ratio(sums[version], n_attempts) for version in VERSIONS},
            "max_abs_advantage": {version: max((group["abs_advantage"][version]["max"] for group in chosen),
                                               default=None) for version in VERSIONS},
        }
    total = {version: sum(by_type[group_type]["sum_abs_advantage"][version] for group_type in KEPT_TYPES)
             for version in VERSIONS}
    means = {group_type: by_type[group_type]["mean_abs_advantage"] for group_type in KEPT_TYPES}
    return {
        "kept_groups": len(kept), "attempts": sum(group["n_valid"] for group in kept),
        "by_type": by_type, "total_abs_advantage": total,
        "penalty_only_share_of_total": {
            version: ratio(by_type["penalty_only"]["sum_abs_advantage"][version], total[version])
            for version in VERSIONS},
        "ratio_of_means_pass_and_fail_to_penalty_only": {
            version: (ratio(means["pass_and_fail"][version], means["penalty_only"][version])
                      if means["pass_and_fail"][version] is not None
                      and means["penalty_only"][version] is not None else None) for version in VERSIONS},
        "kept_groups_with_zero_divisor": sum(1 for group in kept if group["zero_divisor"]),
    }


def worked_examples(reads: dict[str, dict]) -> list[dict]:
    """For the kinds penalty_only and pass_and_fail: the reward pattern that most kept groups of that kind
    have (all runs together; only groups that lost no attempt), with the advantage of each of its reward
    values in the version of the trainer and in the divided one.

    The advantages of the pattern are made by the trainer's function `group_advantages` from the pattern's
    rewards / 100, and divided by `divided_by_std`. Of two patterns with the same number of groups the one
    with the lower rewards comes first. A kind without a group has no entry."""
    examples = []
    for group_type in ("penalty_only", "pass_and_fail"):
        patterns: Counter = Counter()
        per_run: dict[str, Counter] = {}
        n_groups = {}
        for name, read in reads.items():
            chosen = [group for group in read["groups"] if group["group_type"] == group_type]
            n_groups[name] = len(chosen)
            for group in chosen:
                if group["n_valid"] == group["n_attempts"]:
                    pattern = tuple(sorted(Counter(group["rewards"]).items()))
                    patterns[pattern] += 1
                    per_run.setdefault(name, Counter())[pattern] += 1
        if not patterns:
            continue
        pattern = min(patterns, key=lambda item: (-patterns[item], item))
        saved = np.array([[reward for reward, count in pattern for _ in range(count)]], dtype=float)
        rewards = saved / grpo.REWARD_SCALE
        tie_tol = next(iter(reads.values()))["gcfg"]["tie_tol"]
        advantages, _ = impl.main_version("group_advantages")(rewards, tie_tol)
        divided = divided_by_std(rewards[0], np.asarray(advantages, dtype=float)[0], **MAIN_FORM)
        position = {reward: int(np.argmax(saved[0] == reward)) for reward, _ in pattern}
        examples.append({
            "group_type": group_type,
            "rewards": [{"reward": reward, "attempts": count} for reward, count in pattern],
            "groups_with_this_pattern": {name: per_run.get(name, Counter())[pattern] for name in reads},
            "groups_of_this_kind": n_groups,
            "advantage_by_reward": [{"reward": reward, "attempts": count,
                                     "trainer": float(np.asarray(advantages)[0][position[reward]]),
                                     "divided": float(divided[position[reward]])}
                                    for reward, count in pattern],
        })
    return examples


# ---------------------------------------------------------------------------
# One run: everything
# ---------------------------------------------------------------------------
def analyse_run(name: str, folder: Path, families: dict[str, dict], train_sha256: str,
                block_steps: int) -> tuple[dict, dict]:
    """(the numbers of one run for the report, what `read_run` returned). Raises ValueError if the run
    could not be trusted."""
    read = read_run(folder, families, train_sha256)
    manifest, gcfg = read["manifest"], read["gcfg"]
    totals = run_totals(read)
    problems, compared = manifest_problems(folder, manifest, totals)
    if problems:
        raise ValueError(f"{folder}: " + "; ".join(problems))
    config = manifest.get("grpo_config") or {}
    again = manifest.get("steps_run_again") or {}
    numbers = {
        "name": name, "label": label_of(name), "folder": shown(folder),
        "set_up": {
            "status": manifest.get("status"), "complete": manifest.get("complete"),
            "seed": manifest.get("seed"), "starts": len(manifest.get("starts") or []) or None,
            "scenarios_per_step": gcfg["n_scenarios"], "attempts_per_scenario": gcfg["group_size"],
            "tie_tol": gcfg["tie_tol"], "steps_planned": config.get("steps"),
            "learning_rate": config.get("lr"),
            "reference_tokens": config.get("reference_tokens"), "norm_const": manifest.get("norm_const"),
            "start_adapter": (manifest.get("start_adapter") or {}).get("folder"),
            "train_families": manifest.get("families"), "train_scenarios": manifest.get("scenarios"),
            "wording_styles": manifest.get("styles"), "reward_scale": grpo.REWARD_SCALE,
            "reward_settings": reward_settings(manifest),
            "attempts_run_again_after_a_restart": again.get("episodes"),
            "rollout_files_of_steps_run_again": read["rollouts_run_again"],
        },
        "checks": {**read["checks"], "log_values_compared": list(LOG_FIELDS_COMPARED),
                   "counts_compared_by_the_frozen_audit": [key for key in formula.LOG_COMPARED
                                                           if key != "loss"],
                   "run_manifest": compared,
                   "advantage_code_against_the_run_manifest": advantage_code_state(manifest)},
        "totals": totals, "time": run_time(read), "stop_rule": stop_rule_numbers(read, totals),
        "blocks": blocks_of(read, block_steps),
        "signal": signal_numbers(read, families, totals),
        "std_division": std_division_numbers(read),
        "steps": read["steps"],
    }
    return numbers, read


# ---------------------------------------------------------------------------
# The cost: the ledger
# ---------------------------------------------------------------------------
def ledger_number(row: dict, key: str, line: int) -> float | None:
    """The number in one cell of the ledger, or None for an empty cell. Raises ValueError otherwise."""
    text = (row.get(key) or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"line {line} of the ledger: {key} is {text!r}, not a number") from None


def read_rentals(path: Path, frozen_at: datetime) -> dict:
    """The rentals of the spend ledger, each with its hours and its estimated USD, split at the freeze.

    The ledger holds one row with the event "rent" and one with "destroy" for each machine
    (`instance_id`). Per rental:
        rented_utc, stopped_utc    the times of the two rows; stopped_utc is None while the rental is open
        usd_per_hour               `dph_usd` of the rent row
        usd_estimate               `actual_usd` of the destroy row: the ledger's estimate. None if open
        download_estimate_usd      `bandwidth_est_usd` (of the destroy row, else of the rent row)
        hours                      (usd_estimate - download_estimate_usd) / usd_per_hour: the hours that
                                   the estimate charges
        clock_hours                stopped_utc - rented_utc
        at_projected_limit         True if usd_estimate is `projected_usd` of the rent row: the ledger
                                   took the upper limit of the rental, not a measured time
        max_minutes                `max_minutes` of the rent row: the upper limit of the rental (None if empty)
        note                       the note of the destroy row, without the usual first sentence
        phase                      "before" (stopped at or before the freeze), "after" (rented at or after
                                   it), "across" (the freeze lies inside it) or "open" (the ledger has no
                                   stop row for it, or no amount in that row)
        hours_before, usd_before, hours_after, usd_after   the hours and the USD on each side of the
                                   freeze. A rental across the freeze is split by the clock: the share of
                                   its clock time that lies before `frozen_at`
    Also returned: `before` and `after` (hours and usd on each side of the freeze, over the closed
    rentals), `total` (rentals, hours, usd of the closed rentals), `rentals_by_phase`, `open_rentals`
    (rentals without a stop row or without an amount), `rows`, `other_rows`.
    Raises ValueError if a column is missing, a time or a number cannot be read, a machine is rented or
    stopped twice, or a machine is stopped that was never rented."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in LEDGER_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no column {missing}")
        rows = list(reader)
    found: dict[str, dict] = {}
    other_rows = 0
    for line, row in enumerate(rows, start=2):
        event, instance = row["event"], row["instance_id"]
        if event not in ("rent", "destroy"):
            other_rows += 1
            continue
        try:
            when = parse_utc(row["time_utc"])
        except ValueError:
            raise ValueError(f"line {line} of the ledger: the time {row['time_utc']!r} cannot be "
                             "read") from None
        entry = found.setdefault(instance, {})
        if event in entry:
            raise ValueError(f"line {line} of the ledger: the machine {instance} has two rows {event!r}")
        if event == "destroy" and "rent" not in entry:
            raise ValueError(f"line {line} of the ledger: the machine {instance} is stopped but was not "
                             "rented")
        entry[event] = {"when": when, "row": row, "line": line}

    rentals = []
    for instance, entry in found.items():
        rent, stop = entry["rent"], entry.get("destroy")
        price = ledger_number(rent["row"], "dph_usd", rent["line"])
        rental = {"instance_id": instance, "gpu": rent["row"]["gpu"],
                  "rented_utc": rent["row"]["time_utc"], "stopped_utc": None, "usd_per_hour": price,
                  "usd_estimate": None, "download_estimate_usd": None, "hours": None, "clock_hours": None,
                  "at_projected_limit": False, "note": None, "phase": "open",
                  "max_minutes": ledger_number(rent["row"], "max_minutes", rent["line"]),
                  "hours_before": None, "usd_before": None, "hours_after": None, "usd_after": None}
        usd = None if stop is None else ledger_number(stop["row"], "actual_usd", stop["line"])
        if stop is not None:
            rental["stopped_utc"] = stop["row"]["time_utc"]
        if stop is not None and usd is not None and price:
            download = ledger_number(stop["row"], "bandwidth_est_usd", stop["line"])
            if download is None:
                download = ledger_number(rent["row"], "bandwidth_est_usd", rent["line"]) or 0.0
            projected = ledger_number(rent["row"], "projected_usd", rent["line"])
            hours = (usd - download) / price
            clock = hours_between(rent["when"], stop["when"])
            if stop["when"] <= frozen_at:
                phase, share_before = "before", 1.0
            elif rent["when"] >= frozen_at:
                phase, share_before = "after", 0.0
            else:
                phase, share_before = "across", hours_between(rent["when"], frozen_at) / clock
            rental.update({
                "usd_estimate": usd, "download_estimate_usd": download,
                "hours": hours, "clock_hours": clock,
                "at_projected_limit": projected is not None and abs(usd - projected) < 5e-5,
                "note": USUAL_NOTE.sub("", stop["row"]["note"] or "").strip() or None, "phase": phase,
                "hours_before": hours * share_before, "usd_before": usd * share_before,
                "hours_after": hours * (1.0 - share_before), "usd_after": usd * (1.0 - share_before)})
        rentals.append(rental)
    closed = [rental for rental in rentals if rental["phase"] != "open"]

    def side(key: str) -> dict:
        return {"hours": sum(rental[f"hours_{key}"] for rental in closed),
                "usd": sum(rental[f"usd_{key}"] for rental in closed)}

    return {"rentals": rentals, "rows": len(rows), "other_rows": other_rows,
            "open_rentals": [rental["instance_id"] for rental in rentals if rental["phase"] == "open"],
            "rentals_by_phase": {phase: sum(1 for rental in rentals if rental["phase"] == phase)
                                 for phase in ("before", "across", "after", "open")},
            "before": side("before"), "after": side("after"),
            "total": {"rentals": len(closed), "hours": sum(rental["hours"] for rental in closed),
                      "usd": sum(rental["usd_estimate"] for rental in closed)}}


# ---------------------------------------------------------------------------
# The cost: the run manifests
# ---------------------------------------------------------------------------
def run_cost(folder: Path) -> dict:
    """What the cost table reads of one saved run: `found` (it has a run manifest), `complete`, `seconds`,
    `attempts` (`summary.episodes`; 0 for a run that samples nothing), `attempts_run_again` (of a GRPO run
    that was restarted), `script`, and `on_frozen_tree` (its manifest says that the run started on the
    frozen tree). Nothing else of the manifest is used."""
    path = folder / "run_manifest.json"
    if not path.is_file():
        return {"folder": shown(folder), "found": False}
    manifest = read_json(path)
    frozen = manifest.get("frozen") or {}
    return {"folder": shown(folder), "found": True, "complete": manifest.get("complete") is True,
            "seconds": manifest.get("seconds"),
            "attempts": (manifest.get("summary") or {}).get("episodes") or 0,
            "attempts_run_again": (manifest.get("steps_run_again") or {}).get("episodes") or 0,
            "script": manifest.get("script") or (manifest.get("command") or [None])[0],
            "on_frozen_tree": frozen.get("matches") is True and frozen.get("manifest_sha256") is not None}


def cost_row(stage: str, part: str, folders: list[Path]) -> dict:
    """One row of the stage table: the runs of `folders` that are complete and have seconds, added up.
    `not_counted` names the folders that are missing, not complete or without seconds."""
    runs = [run_cost(folder) for folder in folders]
    counted = [run for run in runs if run["found"] and run["complete"] and run["seconds"] is not None]
    return {"stage": stage, "part": part, "runs": len(counted),
            "seconds": sum(run["seconds"] for run in counted),
            "attempts": sum(run["attempts"] + run["attempts_run_again"] for run in counted),
            "on_frozen_tree": sum(1 for run in counted if run["on_frozen_tree"]),
            "folders": [run["folder"] for run in counted],
            "not_counted": [run["folder"] for run in runs if run not in counted]}


def layout_row(path: Path) -> dict:
    """The row of the layout-gradient check: the seconds of its stages (`stages.*.seconds` of
    layout_gradients.json). It samples no attempt: it reads the attempts of the pilot."""
    row = {"stage": "Checks", "part": "layout gradients (reads saved attempts; samples none)", "runs": 0,
           "seconds": 0.0, "attempts": 0, "on_frozen_tree": 0, "folders": [], "not_counted": []}
    if not path.is_file():
        row["not_counted"].append(shown(path))
        return row
    result = read_json(path)
    stages = result.get("stages") or {}
    frozen = ((result.get("code") or {}).get("frozen")) or {}
    row.update({"runs": 1, "seconds": sum(stage.get("seconds") or 0.0 for stage in stages.values()),
                "on_frozen_tree": int(frozen.get("matches") is True
                                      and frozen.get("manifest_sha256") is not None),
                "folders": [shown(path)]})
    return row


def stage_rows(main: Path, runs: list[tuple[str, Path]], calibration: Path) -> tuple[list[dict], dict]:
    """(the rows of the stages after the freeze, the calibration row).

    Stage 1 and the checks are folders under `main`. Stage 2 is, for each GRPO run folder F: F itself, and
    beside it F_dev, F_curve_*, F_shift_differ and F_shift_control. A test rollout (base_test and sft_test
    under `main`, F_test beside F) is a row only if its folder exists. Every other folder of `main` (and of
    `main`/checks) with a run manifest is in the row "other runs", so that no saved run is left out
    silently."""
    used: set[Path] = set()

    def row(stage: str, part: str, folders: list[Path]) -> dict:
        used.update(folder.resolve() for folder in folders)
        return cost_row(stage, part, folders)

    rows = [row("Stage 1", part, [main / name for name in names]) for part, names in STAGE_1]
    rows += [row("Checks", part, [main / name]) for part, name in CHECK_RUNS]
    rows.append(layout_row(main / LAYOUT_FILE))
    tests = [main / name for name in TEST_ARMS]
    for name, folder in runs:
        stage = f"Stage 2, {label_of(name)}"
        beside = lambda suffix: folder.parent / (folder.name + suffix)           # noqa: E731
        curve = sorted(path for path in folder.parent.glob(folder.name + "_curve_*") if path.is_dir())
        rows += [row(stage, "GRPO training", [folder]),
                 row(stage, "dev rollout of the last checkpoint", [beside("_dev")]),
                 row(stage, "dev curve (one small rollout per checkpoint)", curve),
                 row(stage, "second desk", [beside("_shift_differ"), beside("_shift_control")])]
        tests.append(beside("_test"))
    present = [folder for folder in tests if folder.is_dir()]
    if present:
        rows.append(row("Test run", "test rollouts", present))
    others = [path.parent for pattern in ("*/run_manifest.json", "checks/*/run_manifest.json")
              for path in sorted(main.glob(pattern)) if path.parent.resolve() not in used]
    if others:
        rows.append(row("Other runs", "run manifests under --main that no stage above names", others))
    calibration_runs = sorted(path.parent for path in calibration.glob("*/*/run_manifest.json")) \
        if calibration.is_dir() else []
    return rows, cost_row("Before the freeze", "calibration rollouts", calibration_runs)


# Folders of --main that hold outputs of the report programs, not runs.
OUTPUT_FOLDERS = ("audit", "checks", "monitors", "reference", "report", "tables")


def folders_without_run_manifest(main: Path) -> list[str]:
    """The folders of `main` (and of `main`/checks) that hold no run manifest, are no output folder and are
    in no row of the stage table (the layout-gradient check is a row of its own)."""
    if not main.is_dir():
        return []
    found = [path for path in sorted(main.iterdir()) if path.is_dir() and path.name not in OUTPUT_FOLDERS]
    if (main / "checks").is_dir():
        found += [path for path in sorted((main / "checks").iterdir()) if path.is_dir()
                  and (main / LAYOUT_FILE).parent.resolve() != path.resolve()]
    return [path.relative_to(main).as_posix() for path in found if not (path / "run_manifest.json").is_file()]


def smoke_numbers(smoke: Path | None, frozen_at) -> dict:
    """The trial runs of `smoke` (every run_manifest.json below it): count, seconds and attempts, split by
    `started_utc` of the manifest: before the freeze, after it, or no start time. Only `complete`,
    `seconds`, `summary.episodes` and `started_utc` are read."""
    parts = {key: {"runs": 0, "seconds": 0.0, "attempts": 0} for key in ("before", "after", "no_start_time")}
    result = {"folder": None if smoke is None else shown(smoke), "found": bool(smoke and smoke.is_dir()),
              "runs": 0, "seconds": 0.0, "attempts": 0, "not_counted": 0, **parts}
    if not result["found"]:
        return result
    for path in sorted(smoke.rglob("run_manifest.json")):
        manifest = read_json(path)
        seconds = manifest.get("seconds")
        if manifest.get("complete") is not True or not isinstance(seconds, (int, float)):
            result["not_counted"] += 1
            continue
        attempts = (manifest.get("summary") or {}).get("episodes") or 0
        started = manifest.get("started_utc")
        try:
            key = "no_start_time" if not started else \
                ("before" if parse_utc(started) < frozen_at else "after")
        except (TypeError, ValueError):
            key = "no_start_time"
        for target in (result, result[key]):
            target["runs"] += 1
            target["seconds"] += seconds
            target["attempts"] += attempts
    for target in (result, *(result[key] for key in parts)):
        target["hours"] = target["seconds"] / 3600.0
    return result


RENTALS_COLUMNS = ("instance_id", "rented_utc", "stopped_utc", "purpose")
# The optional column of the rentals file: where its stop time comes from.
STOPPED_FROM_LEDGER, STOPPED_FROM_WORK_LOG = "ledger", "work log"
RECORD_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) ")


def rental_purposes(path: Path | None, rentals: list[dict]) -> dict:
    """What each rental was for: the file of purposes (written by hand from the work log), read if it is
    there. Returns {"file", "found", "purposes": {instance_id: purpose}, "stops_from_the_work_log":
    {instance_id: stopped_utc}, "problems": [...]}. A problem is a rental of the file that the ledger does
    not hold, or a time of the file that is not the time of the ledger. The one exception: a stop time with
    "work log" in the column stopped_from is the time of the work log, not of the ledger, and is kept apart.
    A rental of the ledger without a line in the file has no purpose (the table says so)."""
    result = {"file": None if path is None else shown(path), "found": bool(path and path.is_file()),
              "purposes": {}, "stops_from_the_work_log": {}, "problems": []}
    if not result["found"]:
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in RENTALS_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no column {missing}")
        rows = list(reader)
    by_id = {rental["instance_id"]: rental for rental in rentals}
    for line, row in enumerate(rows, start=2):
        instance = row["instance_id"].strip()
        if instance in result["purposes"]:
            raise ValueError(f"line {line} of {path}: the rental {instance} has a second line")
        result["purposes"][instance] = row["purpose"].strip()
        rental = by_id.get(instance)
        if rental is None:
            result["problems"].append(f"{instance} is in {result['file']} but not in the ledger")
            continue
        if row["rented_utc"].strip() != rental["rented_utc"]:
            result["problems"].append(f"{instance}: rented_utc is {row['rented_utc'].strip()} in "
                                      f"{result['file']} and {rental['rented_utc']} in the ledger")
        stopped = row["stopped_utc"].strip()
        source = (row.get("stopped_from") or "").strip() or STOPPED_FROM_LEDGER
        if source not in (STOPPED_FROM_LEDGER, STOPPED_FROM_WORK_LOG):
            raise ValueError(f"line {line} of {path}: stopped_from must be empty, {STOPPED_FROM_LEDGER!r} or "
                             f"{STOPPED_FROM_WORK_LOG!r}, not {source!r}")
        if source == STOPPED_FROM_WORK_LOG:
            if not stopped:
                raise ValueError(f"line {line} of {path}: a stop time from the work log needs a time")
            result["stops_from_the_work_log"][instance] = stopped
            continue
        if stopped and stopped != (rental["stopped_utc"] or ""):
            result["problems"].append(f"{instance}: stopped_utc is {stopped} in {result['file']} and "
                                      f"{rental['stopped_utc'] or 'none'} in the ledger")
    return result


def record_rentals(test_record: Path | None, smoke: Path | None, rentals: list[dict]) -> list[dict]:
    """The rental in which each record of the test chain lies by the clock: the test record (if it is
    there) and every test_run_record.txt below `smoke` (the rehearsals). Only the time at the start of each
    line is read. A record lies in a rental if all these times are at or after its rent row and at or before
    its stop row (or the rental is still open)."""
    records = []
    if test_record is not None and test_record.is_file():
        records.append(("test run", test_record))
    if smoke is not None and smoke.is_dir():
        records += [("rehearsal", path) for path in sorted(smoke.rglob("test_run_record.txt"))]
    found = []
    for kind, path in records:
        times = []
        with path.open(encoding="utf-8") as handle:
            for text in handle:
                match = RECORD_LINE.match(text)
                if match:
                    times.append(parse_utc(match.group(1)))
        inside = []
        if times:
            first, last = min(times), max(times)
            for rental in rentals:
                stopped = None if rental["stopped_utc"] is None else parse_utc(rental["stopped_utc"])
                if parse_utc(rental["rented_utc"]) <= first and (stopped is None or last <= stopped):
                    inside.append(rental["instance_id"])
        found.append({"kind": kind, "record": shown(path), "lines_with_a_time": len(times), "rentals": inside})
    return found


def cost_numbers(main: Path, runs: list[tuple[str, Path]], calibration: Path, ledger: Path,
                 freeze_manifest: Path, run_numbers: dict[str, dict], smoke: Path | None = None, *,
                 rentals_file: Path | None = None, test_record: Path | None = None) -> dict:
    """The content of cost.md. See `read_rentals` for the ledger and `stage_rows` for the run manifests.

    USD of a stage after the freeze = its hours x `usd_per_hour_after_freeze`, the estimated USD of the
    rented time after the freeze divided by its hours. The stage before the freeze is the rented time
    before the freeze (ledger), because the trial runs have no run manifest in the calibration folder.
    `not_in_a_run_manifest` is what is left of the rented time after the freeze."""
    if not freeze_manifest.is_file():
        raise ValueError(f"{freeze_manifest} does not exist: the cost table needs the time of the freeze")
    frozen_text = read_json(freeze_manifest).get("frozen_at")
    try:
        frozen_at = parse_utc(frozen_text)
    except (TypeError, ValueError):
        raise ValueError(f"{freeze_manifest}: frozen_at is {frozen_text!r}, not a time such as "
                         "2026-11-18T21:48:10Z") from None
    if not ledger.is_file():
        raise ValueError(f"{ledger} does not exist")
    rentals = read_rentals(ledger, frozen_at)
    rows, calibration_row = stage_rows(main, runs, calibration)
    price = ratio(rentals["after"]["usd"], rentals["after"]["hours"])
    # The prices of the rentals that were open at some time after the freeze: lowest and highest.
    prices = [rental["usd_per_hour"] for rental in rentals["rentals"]
              if rental["phase"] in ("across", "after")]
    for row in rows:
        row["hours"] = row["seconds"] / 3600.0
        row["usd"] = None if price is None else row["hours"] * price
    calibration_row["hours"] = calibration_row["seconds"] / 3600.0

    stages: list[dict] = [{
        "stage": "Calibration and trials before the freeze", "source": "ledger",
        "attempts": calibration_row["attempts"], "attempts_note": "calibration rollouts only",
        "hours": rentals["before"]["hours"], "usd": rentals["before"]["usd"]}]
    for name in dict.fromkeys(row["stage"] for row in rows):
        parts = [row for row in rows if row["stage"] == name]
        stages.append({"stage": name, "source": "run manifests",
                       "attempts": sum(row["attempts"] for row in parts), "attempts_note": None,
                       "hours": sum(row["hours"] for row in parts),
                       "usd": None if price is None else sum(row["usd"] for row in parts)})
    in_manifests = sum(row["hours"] for row in rows)
    left = {"hours": rentals["after"]["hours"] - in_manifests,
            "usd": None if price is None else rentals["after"]["usd"] - in_manifests * price}
    stages.append({"stage": "Rented after the freeze, in no run manifest of this table", "source": "ledger "
                   "minus run manifests", "attempts": None, "attempts_note": None, **left})
    stages.append({"stage": "Total", "source": "ledger", "attempts": None, "attempts_note": None,
                   "hours": rentals["total"]["hours"], "usd": rentals["total"]["usd"]})

    grpo_rows = {}
    for name, numbers in run_numbers.items():
        totals, time = numbers["totals"], numbers["time"]
        without = time["seconds_in_steps_without_kept_group"]
        grpo_rows[name] = {
            "attempts": totals["attempts"],
            "attempts_in_dropped_groups": totals["attempts_in_dropped_groups"],
            "share_of_attempts_in_dropped_groups": ratio(totals["attempts_in_dropped_groups"],
                                                         totals["attempts"]),
            "sampled_tokens": totals["sampled_tokens"],
            "sampled_tokens_in_dropped_groups": totals["sampled_tokens_in_dropped_groups"],
            "share_of_sampled_tokens_in_dropped_groups": ratio(totals["sampled_tokens_in_dropped_groups"],
                                                               totals["sampled_tokens"]),
            "dropped_groups_by_type": {group_type: totals["groups_by_type"][group_type]
                                       for group_type in DROPPED_TYPES},
            "steps": totals["steps"], "steps_without_kept_group": totals["steps_without_kept_group"],
            "steps_without_kept_group_share_of_steps": totals["steps_without_kept_group_share_of_steps"],
            "seconds_in_steps_without_kept_group": without,
            "hours_in_steps_without_kept_group": without / 3600.0,
            "share_of_step_seconds": time["share_of_step_seconds_in_steps_without_kept_group"],
            "usd_of_steps_without_kept_group": None if price is None else without / 3600.0 * price,
            "step_seconds": time["step_seconds"], "sampling_seconds": time["sampling_seconds"],
            "update_seconds": time["update_seconds"],
        }
    counted = sum(row["runs"] for row in rows)
    # The rentals that were rented after the last GRPO run had ended (`ended_utc` of the run manifests).
    ends = [numbers["time"]["ended_utc"] for numbers in run_numbers.values()]
    later = None
    test_in_table = any(row["stage"] == "Test run" for row in rows)
    # With a test run in the table, a rental after the GRPO runs may hold a run of the table (the test
    # rollouts hold no start time), so the hours that would be left cannot be given.
    if ends and all(ends) and not test_in_table:
        last_end = max(parse_utc(end) for end in ends)
        chosen = [rental for rental in rentals["rentals"]
                  if rental["phase"] == "after" and parse_utc(rental["rented_utc"]) >= last_end]
        later = {"last_grpo_run_ended_utc": last_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "rentals": [rental["instance_id"] for rental in chosen],
                 "hours": sum(rental["hours"] for rental in chosen),
                 "usd": sum(rental["usd_estimate"] for rental in chosen)}
        # What is left of the rented time after the freeze if no run of the table ran in these rentals.
        later["hours_left_for_the_other_rentals"] = left["hours"] - later["hours"]
    return {
        "frozen_at": frozen_text, "ledger": shown(ledger), "ledger_sha256": sha256_of(ledger),
        "ledger_rows": rentals["rows"], "ledger_other_rows": rentals["other_rows"],
        "rentals": rentals["rentals"], "open_rentals": rentals["open_rentals"],
        "rentals_by_phase": rentals["rentals_by_phase"],
        "rented": {"before_freeze": rentals["before"], "after_freeze": rentals["after"],
                   "total": rentals["total"]},
        "usd_per_hour_after_freeze": price,
        "usd_per_hour_of_the_rentals_after_freeze": {"lowest": min(prices, default=None),
                                                     "highest": max(prices, default=None)},
        "stages": stages, "stage_rows": rows, "calibration": calibration_row,
        "not_in_a_run_manifest": left, "rentals_after_the_last_grpo_run": later,
        "hours_in_run_manifests_after_freeze": in_manifests,
        "run_manifests_counted": counted,
        "run_manifests_on_the_frozen_tree": sum(row["on_frozen_tree"] for row in rows),
        "test_run_in_the_table": test_in_table,
        "trial_runs": smoke_numbers(smoke, frozen_at),
        "folders_without_run_manifest": folders_without_run_manifest(main),
        "grpo": grpo_rows,
        "purposes": rental_purposes(rentals_file, rentals["rentals"]),
        "records": record_rentals(test_record, smoke, rentals["rentals"]),
    }


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------
def build_report(runs: list[tuple[str, Path]], *, scenarios: Path, main: Path, calibration: Path,
                 ledger: Path, freeze_manifest: Path, block_steps: int = 10, smoke: Path | None = None,
                 rentals_file: Path | None = None, test_record: Path | None = None) -> dict:
    """Every number of the output files, as one dict (the content of training_report.json). Raises
    ValueError for everything that the module text names; then nothing must be written."""
    names = [name for name, _ in runs]
    if not runs or len(set(names)) != len(names) or not all(RUN_NAME.fullmatch(name) for name in names):
        raise ValueError(f"give each run once, with its own name of letters, digits and '_': {names}")
    if block_steps < 1:
        raise ValueError(f"--block-steps must be 1 or more, not {block_steps}")
    families, train_sha256 = load_train_families(scenarios)
    numbers, reads = {}, {}
    for name, folder in runs:
        numbers[name], reads[name] = analyse_run(name, folder, families, train_sha256, block_steps)
    first = next(iter(numbers.values()))["set_up"]
    for key in ("reward_settings", "scenarios_per_step", "attempts_per_scenario", "tie_tol"):
        other = [name for name, run in numbers.items() if run["set_up"][key] != first[key]]
        if other:
            raise ValueError(f"the runs do not have the same {key} ({other} differ from {names[0]}): their "
                             "numbers do not belong in one table")
    return {
        "what": "GRPO training report: summary, learning signal, division by the standard deviation, cost",
        "script": SCRIPT, "script_sha256": sha256_of(ROOT / SCRIPT), "block_steps": block_steps,
        "reward_settings": first["reward_settings"],
        "inputs": {
            "runs": {name: shown(folder) for name, folder in runs},
            "train_file": shown(scenarios / TRAIN_FILE), "train_file_sha256": train_sha256,
            "train_families": len(families), "main": shown(main), "calibration": shown(calibration),
            "ledger": shown(ledger), "freeze_manifest": shown(freeze_manifest),
            "rentals": None if rentals_file is None else shown(rentals_file),
            "test_record": None if test_record is None else shown(test_record),
            "trial_runs": None if smoke is None else shown(smoke)},
        "std_division_forms": {"divided": MAIN_FORM, "divided_check": CHECK_FORM,
                               "reward_scale": grpo.REWARD_SCALE},
        "worked_examples": worked_examples(reads),
        "step_columns": [{"column": name, "meaning": meaning, "source": source}
                         for name, meaning, source in STEP_COLUMNS],
        "runs": numbers,
        "cost": cost_numbers(main, runs, calibration, ledger, freeze_manifest, numbers, smoke,
                             rentals_file=rentals_file, test_record=test_record),
    }


# ---------------------------------------------------------------------------
# Words and tables
# ---------------------------------------------------------------------------
def whole(value: int | None) -> str:
    return "no value" if value is None else f"{value:,}"


def percent(value: float | None, digits: int = 1) -> str:
    return "no value" if value is None else f"{100.0 * value:.{digits}f}%"


def count_share(part: int, total: int, digits: int = 1) -> str:
    """"1,456 (14.2%)": a count and its share of `total`."""
    return f"{whole(part)} ({percent(ratio(part, total), digits)})"


def count_of(part: int, total: int, digits: int = 1) -> str:
    """"91 of 640 (14.2%)"."""
    return f"{whole(part)} of {whole(total)} ({percent(ratio(part, total), digits)})"


def hours_text(seconds: float | None) -> str:
    return "no value" if seconds is None else f"{seconds / 3600.0:.2f}"


def decimal(value: float | None, digits: int = 2) -> str:
    return "no value" if value is None else f"{value:.{digits}f}"


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def usd_text(value: float | None) -> str:
    return "no value" if value is None else f"{value:.2f}"


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def joined(parts: list[str]) -> str:
    """"A, B and C"."""
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]


def run_list(report: dict) -> str:
    """"seed 1 (`results/main/grpo_seed1`) and seed 2 (`results/main/grpo_seed2`)"."""
    return joined([f"{run['label']} (`{run['folder']}`)" for run in report["runs"].values()])


def per_run(report: dict, text) -> str:
    """"A (seed 1) and B (seed 2)" for text(run) of each run."""
    return joined([f"{text(run)} ({run['label']})" for run in report["runs"].values()])


def by_run(report: dict, text) -> str:
    """"seed 1: A; seed 2: B" for text(run) of each run."""
    return "; ".join(f"{run['label']}: {text(run)}" for run in report["runs"].values())


def fault_lines(report: dict) -> list[str]:
    """One sentence for the output files if a kept group is in a step that ended in a fault; else nothing."""
    if not any(run["totals"]["kept_groups_in_steps_with_a_fault"] for run in report["runs"].values()):
        return []
    return ["A step that ended in a fault made no update. Kept groups in such steps: "
            + by_run(report, lambda run: whole(run["totals"]["kept_groups_in_steps_with_a_fault"]))
            + ". This file counts them as kept groups, because their rewards are not all equal."]


def group_words(report: dict) -> tuple[int, int]:
    """(scenarios per step, attempts per scenario) of the runs, for the definitions."""
    set_up = next(iter(report["runs"].values()))["set_up"]
    return set_up["scenarios_per_step"], set_up["attempts_per_scenario"]


def reward_words(report: dict) -> str:
    """The reward in one definition, with the numbers of the runs' config (`reward_settings`)."""
    numbers = report["reward_settings"]
    fail = ("0 or less for a failed attempt (it loses the points for its malformed messages)"
            if numbers["penalise_malformed_on_fail"] else "0 for a failed attempt")
    return (f"Reward: {fail}. {numbers['pass_reward']} for a passed attempt, minus "
            f"{numbers['malformed_penalty']} for each malformed message (at most "
            f"{numbers['malformed_cap']}), minus {numbers['turn_penalty']} for each turn over the budget (at "
            f"most {numbers['turn_cap']}). The budget is the shortest correct path plus one turn.")


def reward_points(report: dict, mean_reward: float | None) -> str:
    """A mean reward of the log's scale (reward / 100) in reward points, with two decimals."""
    scale = report["std_division_forms"]["reward_scale"]
    return decimal(None if mean_reward is None else scale * mean_reward)


GRPO_WORDS = ("GRPO: the training method of these runs. It compares the rewards of the attempts of one "
              "group, and it makes the attempts with a reward above the mean of the group more likely.")
ATTEMPT_WORDS = ("Attempt: one conversation of the model with the tools of the desk. A turn is one message "
                 "of the model in it. A malformed message is a message that is not exactly one valid tool "
                 "call.")
OUTCOME_WORDS = ("The four expected outcomes: `grant` (the role is given), `approval` (the role is sensitive "
                 "and needs approval), `already` (the person holds the role already), `inactive` (the "
                 "person is inactive).")
TYPE_WORDS = {"pass_and_fail": "passes and fails", "penalty_only": "penalty-only",
              "all_fail": "every attempt failed, with different rewards"}
PHASE_WORDS = {"before": "before the freeze", "across": "across the freeze", "after": "after the freeze",
               "open": "no stop row, or no amount"}


def phase_words(rental: dict) -> str:
    """The words of the column "By the clock" for one rental of the ledger."""
    if rental["phase"] != "open":
        return PHASE_WORDS[rental["phase"]]
    if rental["stopped_utc"] is None:
        return "still open: not yet in the total"
    return "no amount: not in the total"


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------
def manifest_check_lines(report: dict) -> list[str]:
    """The two lines of the checks in summary.md that depend on what the run manifests and the records
    name: the totals and the hash of the log, and the hash of the scenario file. Each says only what was
    compared."""
    runs = list(report["runs"].values())
    keys = [key for key in runs[0]["checks"]["run_manifest"]["summary_keys_compared"]
            if all(key in run["checks"]["run_manifest"]["summary_keys_compared"] for run in runs)]
    totals = (f"- The totals in `run_manifest.json` ({', '.join(f'`{key}`' for key in keys)}) are the totals "
              "of the log and the rollout files." if keys else
              "- The run manifests hold no totals that could be compared with the log.")
    if all(run["checks"]["run_manifest"]["log_sha256_compared"] for run in runs):
        totals += " The log is the file that the manifest names by its SHA-256."
    else:
        totals += " At least one run manifest names no SHA-256 of its log, so that was not compared."
    train = f"`{report['inputs']['train_file']}`"
    by_manifest = all(run["checks"]["run_manifest"]["names_the_scenario_file"] for run in runs)
    by_records = all(run["checks"].get("records_that_name_the_scenario_file", 0) == run["totals"]["attempts"]
                     for run in runs)
    if by_manifest and by_records:
        scenario = (f"- {train} is the scenario file that the run manifests and all records name by its "
                    "SHA-256.")
    else:
        named = by_run(report, lambda run: (
            f"{whole(run['checks'].get('records_that_name_the_scenario_file', 0))} of "
            f"{whole(run['totals']['attempts'])} records"))
        in_manifest = by_run(report, lambda run: yes_no(
            run["checks"]["run_manifest"]["names_the_scenario_file"]))
        scenario = (f"- {train}: no record and no run manifest names another scenario file. Records that "
                    f"name it by its SHA-256: {named}. Run manifests that name it: {in_manifest}.")
    return [totals, scenario]


def summary_in_short(report: dict) -> list[str]:
    """The first lines of summary.md: a few numbers of the tables below, in sentences."""
    def block_words(item: dict) -> str:
        return f"{percent(item['pass_rate'])} (steps {item['first_step']} to {item['last_step']})"

    return [
        "- Attempts sampled: " + by_run(report, lambda run: (
            f"{whole(run['totals']['attempts'])} in {whole(run['totals']['steps'])} steps")) + ".",
        "- Failed attempts: "
        + by_run(report, lambda run: count_of(run["totals"]["failed"], run["totals"]["valid"], 2)) + ".",
        "- Kept groups: "
        + by_run(report, lambda run: count_of(run["totals"]["kept_groups"], run["totals"]["groups"]))
        + ". The other groups gave no learning signal.",
        "- Steps with no kept group: "
        + by_run(report, lambda run: count_of(run["totals"]["steps_without_kept_group"],
                                              run["totals"]["steps"])) + ".",
        "- Pass rate of the training attempts in the first and in the last block of steps: "
        + by_run(report, lambda run: f"{block_words(run['blocks'][0])} and {block_words(run['blocks'][-1])}")
        + ".",
        "- Hours of the run: " + by_run(report, lambda run: (
            f"{hours_text(run['time']['run_seconds'])}, of which sampling "
            f"{hours_text(run['time']['sampling_seconds'])}")) + ".",
    ]


def render_summary(report: dict) -> str:
    runs = list(report["runs"].values())
    n_scenarios, group_size = group_words(report)
    block = report["block_steps"]
    top = report["reward_settings"]["pass_reward"]
    names = [run["label"] for run in runs]
    out = [
        "# GRPO training: what the runs did",
        "",
        f"Made by `{SCRIPT}` from saved files only. The runs: {run_list(report)}. Of each run it reads "
        "`train_log.jsonl`, `rollouts/step_*.episodes.jsonl` and `run_manifest.json`. All attempts here are "
        "training attempts on the train split. No number in this file is from the dev set or the test set.",
        "",
        "## Words",
        "",
        f"- {GRPO_WORDS}",
        f"- {ATTEMPT_WORDS}",
        "- Scenario: one family (one world with one correct end state) in one wording style.",
        f"- Step: one round of training. The model makes {group_size} attempts at each of {n_scenarios} "
        "scenarios, and then the weights get one update.",
        f"- Group: the {group_size} attempts of one scenario in one step.",
        f"- {reward_words(report)}",
        "- Harness error: a fault of the program that runs an attempt, not of the model. Such an attempt is "
        "lost: it is counted, and it is in no other number.",
        "- Kept group: a group whose attempts do not all have the same reward. Only kept groups give a "
        "learning signal. A group with equal rewards is dropped. The share of the groups with equal "
        "rewards is the zero-variance fraction.",
        "- Group with passes and fails: a kept group with at least one passed and one failed attempt.",
        "- Penalty-only group: a kept group in which every attempt passed. Its rewards differ only by "
        "penalties.",
        "- Trained attempt: an attempt of a kept group.",
        "- Sampled token: a token that the model wrote. Loss token: a sampled token of a trained attempt.",
        "- Pass rate: the frozen function `pass_rate` of `accessdesk/eval/metrics.py`. It is the mean over "
        "families of the family's share of passed attempts.",
        "",
        "## In short",
        "",
        *summary_in_short(report),
        "",
        "## Set-up of each run",
        "",
        "Source: `run_manifest.json`. Adapter: the small set of added weights that the training changes "
        "(LoRA). The other weights of the model stay as they are.",
        "",
    ]
    set_rows = [
        ("Status of the run", lambda run: str(run["set_up"]["status"])),
        ("Training seed", lambda run: str(run["set_up"]["seed"])),
        ("Scenarios per step x attempts per scenario",
         lambda run: f"{run['set_up']['scenarios_per_step']} x {run['set_up']['attempts_per_scenario']}"),
        ("Steps planned", lambda run: whole(run["set_up"]["steps_planned"])),
        ("Learning rate", lambda run: str(run["set_up"]["learning_rate"])),
        ("Adapter at the start", lambda run: f"`{run['set_up']['start_adapter']}`"),
        ("Training scenarios (families)",
         lambda run: f"{whole(run['set_up']['train_scenarios'])} ({whole(run['set_up']['train_families'])})"),
        ("Starts of the script", lambda run: whole(run["set_up"]["starts"])),
        ("Attempts of steps that a restart ran again",
         lambda run: whole(run["set_up"]["attempts_run_again_after_a_restart"])),
    ]
    out.append(table(["Setting", *names], [[name, *[cell(run) for run in runs]] for name, cell in set_rows]))

    out += ["## Totals", ""]
    total_rows = [
        ("Steps", lambda t, m: whole(t["steps"]), "lines of `train_log.jsonl`; each has its rollout file"),
        ("Attempts sampled", lambda t, m: whole(t["attempts"]), "records in the rollout files"),
        ("Attempts lost to a harness error", lambda t, m: whole(t["harness_errors"]),
         "`harness_error` is not null; such an attempt is in no other number"),
        ("Failed attempts", lambda t, m: count_share(t["failed"], t["valid"], 2),
         "`passed` is false; share of the valid attempts"),
        (f"Mean reward, of {top}", lambda t, m: reward_points(report, t["mean_reward"]),
         "mean of `reward` over the valid attempts"),
        ("Groups sampled", lambda t, m: whole(t["groups"]), "steps x scenarios per step"),
        ("Kept groups", lambda t, m: count_of(t["kept_groups"], t["groups"]),
         "the rule of the trainer: the rewards of the group are not all equal; share of the groups sampled"),
        ("- with passes and fails", lambda t, m: whole(t["groups_by_type"]["pass_and_fail"]), ""),
        ("- penalty-only", lambda t, m: whole(t["groups_by_type"]["penalty_only"]), ""),
        ("Dropped groups", lambda t, m: count_of(t["dropped_groups"], t["groups"]),
         "the groups that are not kept"),
        ("Groups with equal rewards (zero variance)",
         lambda t, m: count_of(t["equal_reward_groups"]["equal_reward"],
                               t["equal_reward_groups"]["groups_counted"]),
         "frozen function `zero_variance_share` (`equal_reward`), each step by itself"),
        ("- every attempt passed", lambda t, m: whole(t["equal_reward_groups"]["equal_reward_all_pass"]), ""),
        ("- every attempt failed", lambda t, m: whole(t["equal_reward_groups"]["equal_reward_all_fail"]), ""),
        ("Groups in which all attempts passed or all failed",
         lambda t, m: count_of(t["equal_reward_groups"]["equal_passfail"],
                               t["equal_reward_groups"]["groups_counted"]),
         "the same function (`equal_passfail`); the penalty-only groups are among them"),
        ("Steps with no kept group", lambda t, m: count_of(t["steps_without_kept_group"], t["steps"]),
         "`kept_groups` of the log line is 0"),
        ("Trained attempts", lambda t, m: count_share(t["trained_attempts"], t["attempts"]),
         "`n_trained_episodes` of the log, added up; share of the attempts sampled"),
        ("Sampled tokens", lambda t, m: whole(t["sampled_tokens"]),
         "`n_tokens` of every turn of the valid attempts"),
        ("Loss tokens", lambda t, m: count_share(t["loss_tokens"], t["sampled_tokens"]),
         "`n_loss_tokens` of the log, added up; share of the sampled tokens"),
        ("Hours of the run", lambda t, m: hours_text(m["run_seconds"]),
         "`seconds` of `run_manifest.json`: the whole script, with the loading of the model"),
        ("- hours in the steps", lambda t, m: hours_text(m["step_seconds"]),
         "`seconds` of the log, added up"),
        ("- of these, sampling", lambda t, m: f"{hours_text(m['sampling_seconds'])} "
                                              f"({percent(m['sampling_share_of_step_seconds'])})",
         "`rollout_seconds` of the log, added up; share of the hours in the steps"),
        ("- of these, update", lambda t, m: hours_text(m["update_seconds"]),
         "`seconds` minus `rollout_seconds`"),
    ]
    if any(run["totals"]["groups_by_type"]["dropped_lost_attempts"]
           or run["totals"]["groups_by_type"]["dropped_other"] for run in runs):
        total_rows.insert(10, ("- of these, dropped because the harness lost more than half of the attempts",
                               lambda t, m: whole(t["groups_by_type"]["dropped_lost_attempts"]
                                                  + t["groups_by_type"]["dropped_other"]), ""))
    if any(run["totals"]["groups_by_type"]["all_fail"] for run in runs):
        total_rows.insert(9, ("- every attempt failed, with different rewards",
                              lambda t, m: whole(t["groups_by_type"]["all_fail"]), ""))
    out.append(table(["Number", *names, "Rule and source"],
                     [[name, *[cell(run["totals"], run["time"]) for run in runs], rule]
                      for name, cell, rule in total_rows]))
    zero = by_run(report, lambda run: (
        f"{whole(run['totals']['steps_without_kept_group_and_zero_gradient'])} of "
        f"{whole(run['totals']['steps_without_kept_group'])} steps"))
    out += ["A step with no kept group has no gradient of its own. For such steps the log holds the loss 0 "
            f"and the gradient norm 0 ({zero}).", ""]
    out += [line for sentence in fault_lines(report) for line in (sentence, "")]
    if any(run["set_up"]["tie_tol"] for run in runs):
        out += ["A run here has a `tie_tol` other than 0: rewards that lie within `tie_tol` of each other "
                "count as equal.", ""]

    out += [f"## Pass rate in blocks of {block} steps", "",
            "The pass rate is the frozen function `pass_rate` (family-weighted). The counts beside it are "
            "attempts. A family with two wordings in one block has more attempts and the same weight, so "
            "the pass rate can differ a little from the passed attempts divided by the attempts.", "",
            "Each step takes other scenarios, until all training scenarios are used (`step_scenarios` in "
            "`accessdesk/train/grpo.py`). The steps before the first repeat hold only scenarios that the "
            "run had not sampled before. Different scenarios in the groups of a run: "
            + by_run(report, lambda run: (
                f"{whole(run['totals']['different_scenarios'])} in {whole(run['totals']['groups'])} groups"
                + ("" if run["totals"]["first_step_with_a_scenario_of_an_earlier_step"] is None else
                   "; the first step with a scenario of an earlier step is step "
                   f"{run['totals']['first_step_with_a_scenario_of_an_earlier_step']}"))) + ".", ""]
    for run in runs:
        out += [f"### {run['label']}", ""]
        rows = []
        for item in run["blocks"]:
            rows.append([
                f"{item['first_step']} to {item['last_step']}", whole(item["attempts"]),
                count_share(item["failed"], item["valid"], 2), percent(item["pass_rate"]),
                reward_points(report, item["mean_reward"]), decimal(item["tokens_per_attempt"], 1),
                count_of(item["kept_groups"], item["groups"]),
                whole(item["groups_by_type"]["pass_and_fail"]), whole(item["groups_by_type"]["penalty_only"]),
                f"{whole(item['steps_without_kept_group'])} of {whole(item['steps'])}"])
        out.append(table(["Steps", "Attempts", "Failed attempts", "Pass rate", f"Mean reward, of {top}",
                          "Sampled tokens per attempt", "Kept groups", "with passes and fails",
                          "penalty-only", "Steps with no kept group"], rows))
        outcomes = [outcome for outcome in OUTCOMES
                    if any(outcome in item["by_outcome"] for item in run["blocks"])]
        rows = []
        for item in run["blocks"]:
            cells = []
            for outcome in outcomes:
                found = item["by_outcome"].get(outcome)
                cells.append("no attempt" if found is None else
                             f"{percent(found['pass_rate'])} ({whole(found['passed'])} of "
                             f"{whole(found['valid'])})")
            rows.append([f"{item['first_step']} to {item['last_step']}", *cells])
        out += [f"Pass rate per expected outcome, {run['label']} (frozen function `pass_rate_by`; in "
                "brackets: passed attempts of the attempts with that outcome):", "",
                table(["Steps", *outcomes], rows)]
    out += [OUTCOME_WORDS, ""]

    out += ["## Attempts at each reward value", "",
            "Frozen function `reward_distribution`. Unit: attempts.", ""]
    for run in runs:
        values = "; ".join(f"reward {reward}: {whole(count)}"
                           for reward, count in run["totals"]["reward_values"].items())
        out.append(f"- {run['label']}: {values}")
    out.append("")

    out += ["## Numbers of the stop rule", "",
            "The plan stops a run only for a fault. This table holds the numbers that the stop rule reads, "
            "from `train_log.jsonl`, and the limits from `run_manifest.json`.", ""]
    stop_rows = [
        ("Steps that ended in a fault", lambda s: whole(s["steps_with_a_fault"]),
         lambda s: "a fault ends the run"),
        ("Attempts lost to a harness error", lambda s: f"{whole(s['harness_errors'])} "
                                                       f"({percent(s['harness_error_rate'], 2)})",
         lambda s: f"above {percent(s['limit_harness_error_rate'], 0)} of the attempts"),
        ("Largest mean log-probability gap of a step",
         lambda s: "no value" if s["largest_logprob_gap_mean"]["value"] is None else
         f"{s['largest_logprob_gap_mean']['value']:.4f} (step {s['largest_logprob_gap_mean']['step']})",
         lambda s: f"above {s['limit_logprob_gap_mean']}"),
        ("Invalid-call rate: at step 0; the largest of a step",
         lambda s: f"{percent(s['invalid_call_rate_step_0'], 2)}; "
                   f"{percent(s['largest_invalid_call_rate']['value'], 2)} "
                   f"(step {s['largest_invalid_call_rate']['step']})",
         lambda s: f"above {percent(s['limit_invalid_call_rate'], 0)} and at least twice the value of "
                   "step 0"),
        ("Largest gradient norm before clipping; steps that were clipped",
         lambda s: "no value" if s["largest_grad_norm"]["value"] is None else
         f"{s['largest_grad_norm']['value']:.4f} (step {s['largest_grad_norm']['step']}); "
         f"{whole(s['steps_clipped'])}",
         lambda s: f"the clip is at {s['clip_grad_norm']} (not a stop rule)"),
    ]
    out.append(table(["Number", *names, "Limit"],
                     [[name, *[cell(run["stop_rule"]) for run in runs], limit(runs[0]["stop_rule"])]
                      for name, cell, limit in stop_rows]))
    out += ["- Fault: one of the conditions of the stop rule in `ANALYSIS_PLAN.md` (section 7), such as a "
            "loss that is not a finite number.",
            "- Log-probability gap: the mean absolute difference, over the loss tokens of a step, between "
            "the log-probability of a token when the model wrote it and the one that the training step "
            "computed for the same token.",
            "- Invalid-call rate: malformed messages divided by all messages of the model.",
            "- Gradient norm: the size of the change that a step asks of the weights. A norm above the clip "
            "is scaled down to the clip.", ""]

    out += ["## The step files", "",
            "One file per run: " + ", ".join(f"`steps_{run['name']}.csv`" for run in runs)
            + ". One row per step. An empty cell means that the log has no value.", "",
            table(["Column", "What it is, and its unit", "Source"],
                  [[f"`{item['column']}`", item["meaning"], item["source"]]
                   for item in report["step_columns"]])]

    out += ["## Checks before these tables were written", ""]
    compared = per_run(report, lambda run: whole(run["checks"].get("steps_compared_with_the_log", 0)))
    out += [
        f"- Steps compared with their rollout file: {compared}. In every step the frozen audit function "
        "`audit_step` (`accessdesk/audit/formula.py`) found no difference: the scenario ids, the counts of "
        "attempts and of harness errors, the mean reward, the kept groups, the trained attempts and the loss "
        "tokens of the log are those of the saved rollout; and the two versions of the reward function and "
        "of the advantage function agree.",
        "- In every step this script made these log values again from the rollout file with the frozen "
        "metric functions, and each agreed: " + ", ".join(f"`{key}`" for key in LOG_FIELDS_COMPARED) + ".",
        *manifest_check_lines(report),
        "- Not compared, read from the log as they are: the loss, the gradient norm, the log-probability "
        "gap and the seconds. The formula audit (`scripts/formula_audit.py`) compares the loss with the "
        "saved token files.",
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# signal.md
# ---------------------------------------------------------------------------
KEPT_HEADER = ["Kept groups, of the groups sampled", "Share of all kept groups", "With passes and fails",
               "Penalty-only", "Failed attempts (share of all failed attempts)"]


def kept_cells(row: dict) -> list[str]:
    """The cells that every table of signal.md shares, for one row of `tally`."""
    return [count_of(row["kept_groups"], row["groups"]), percent(row["share_of_all_kept_groups"]),
            whole(row["pass_and_fail"]), whole(row["penalty_only"]),
            f"{whole(row['failed_attempts'])} ({percent(row['share_of_all_failed_attempts'])})"]


def kind_words(row: dict) -> str:
    """The rule facts of a row in words."""
    person = "an active person" if row["active"] else "an inactive person"
    holds = "who holds the role already" if row["holds_role"] else "who does not hold the role"
    role = "a sensitive role" if row["sensitive"] else "a role that is not sensitive"
    return f"{person} {holds}, and {role}"


def in_short(report: dict) -> list[str]:
    """The first lines of signal.md: a few counts of the tables below, in sentences."""
    runs = list(report["runs"].values())
    lines = [
        "- Kept groups: "
        + by_run(report, lambda run: count_of(run["totals"]["kept_groups"], run["totals"]["groups"]))
        + ". The other groups gave no learning signal.",
        "- Penalty-only groups among the kept groups: "
        + by_run(report, lambda run: count_of(run["signal"]["penalty_only"]["groups"],
                                              run["signal"]["kept_groups"]))
        + ". In such a group no attempt failed.",
        "- Failed attempts: "
        + by_run(report, lambda run: count_of(run["totals"]["failed"], run["totals"]["valid"], 2)) + ".",
    ]
    for run in runs:
        rows = run["signal"]["by_facts"]
        if not rows or not run["totals"]["failed"]:
            continue
        most = max(rows, key=lambda row: row["failed_attempts"])
        lines.append(f"- The kind of family with the most failed attempts, {run['label']}: "
                     f"{kind_words(most)} (expected outcome `{most['outcome']}`). It has "
                     f"{whole(most['failed_attempts'])} of the {whole(run['totals']['failed'])} failed "
                     f"attempts ({percent(most['share_of_all_failed_attempts'])}) and "
                     f"{whole(most['groups'])} of the {whole(run['totals']['groups'])} groups sampled "
                     f"({percent(ratio(most['groups'], run['totals']['groups']))}).")
    named = [next((row for row in run["signal"]["by_same_name"] if row["same_name"]), None) for run in runs]
    if all(row is not None for row in named) and all(run["totals"]["failed"] for run in runs):
        lines.append(
            "- Families with two people of one name ("
            f"{whole(named[0]['train_families'])} of the {whole(report['inputs']['train_families'])} train "
            "families) gave these shares of the failed attempts: "
            + by_run(report, lambda run: count_of(
                next(row for row in run["signal"]["by_same_name"] if row["same_name"])["failed_attempts"],
                run["totals"]["failed"])) + ".")
    return lines


def render_signal(report: dict) -> str:
    runs = list(report["runs"].values())
    _, group_size = group_words(report)
    top = report["reward_settings"]["pass_reward"]
    out = [
        "# Where the learning signal came from",
        "",
        f"Made by `{SCRIPT}` from the saved training attempts (`rollouts/step_*.episodes.jsonl`) of "
        f"{run_list(report)}, and from the family records in `{report['inputs']['train_file']}`. No number "
        "in this file is from the dev set or the test set.",
        "",
        "## Words",
        "",
        f"- {GRPO_WORDS}",
        f"- {ATTEMPT_WORDS}",
        "- Family: one world with one correct end state. Scenario: one family in one wording style.",
        f"- Group: the {group_size} attempts of one scenario in one training step.",
        f"- {reward_words(report)}",
        "- Kept group: a group whose attempts do not all have the same reward. GRPO learns only from kept "
        "groups: the advantage of an attempt is its reward minus the mean reward of its group, and it is 0 "
        "for every attempt of a group with equal rewards.",
        "- Group with passes and fails: a kept group with at least one passed and one failed attempt.",
        f"- Penalty-only group: a kept group in which all {group_size} attempts passed. It was kept only "
        "because some attempts lost points for a penalty.",
        "- Rule facts of a family: the person is active or not; the person holds the requested role or not; "
        "the role is sensitive or not. The desk uses the first rule that fits: an inactive person, then a "
        "role held already, then a sensitive role, then a grant.",
        "- Two people of one name: the world of the family holds two people with the same full name.",
        "",
        "## In short",
        "",
        *in_short(report),
        "",
        "## The kept groups",
        "",
    ]
    rows = [
        ("Groups sampled", lambda run: whole(run["totals"]["groups"])),
        ("Kept groups", lambda run: count_of(run["totals"]["kept_groups"], run["totals"]["groups"])),
        ("- with passes and fails (share of the kept groups)",
         lambda run: count_share(run["signal"]["pass_and_fail"]["groups"], run["signal"]["kept_groups"])),
        ("- penalty-only: every attempt passed, and a penalty made the rewards differ (share of the kept "
         "groups)",
         lambda run: count_share(run["signal"]["penalty_only"]["groups"], run["signal"]["kept_groups"])),
        ("Failed attempts, of the attempts sampled",
         lambda run: count_of(run["totals"]["failed"], run["totals"]["valid"], 2)),
        ("Steps with a kept group",
         lambda run: count_of(run["signal"]["steps_with_a_kept_group"], run["totals"]["steps"])),
        ("- of these, steps whose kept groups were all penalty-only",
         lambda run: whole(run["signal"]["steps_whose_kept_groups_are_all_penalty_only"])),
    ]
    out.append(table(["Number", *[run["label"] for run in runs]],
                     [[name, *[cell(run) for run in runs]] for name, cell in rows]))
    out += [line for sentence in fault_lines(report) for line in (sentence, "")]

    for run in runs:
        signal = run["signal"]
        out += [f"## {run['label']}: kept groups per expected outcome", "",
                table(["Expected outcome", *KEPT_HEADER],
                      [[row["outcome"], *kept_cells(row)] for row in signal["by_outcome"]])]
        out += [f"## {run['label']}: kept groups per kind of family (rule facts)", "",
                table(["Expected outcome", "Person is active", "Person holds the role", "Role is sensitive",
                       "Families of this kind in train", *KEPT_HEADER],
                      [[row["outcome"], yes_no(row["active"]), yes_no(row["holds_role"]),
                        yes_no(row["sensitive"]), whole(row["train_families"]), *kept_cells(row)]
                       for row in signal["by_facts"]])]
        out += [f"## {run['label']}: kept groups with and without two people of one name", "",
                table(["Two people of one name", "Families of this kind in train", *KEPT_HEADER],
                      [[yes_no(row["same_name"]), whole(row["train_families"]), *kept_cells(row)]
                       for row in signal["by_same_name"]])]
    out += [OUTCOME_WORDS, "",
            "The table of the rule facts crossed with \"two people of one name\" is in "
            f"`{JSON_NAME}` (`by_facts_and_same_name`).", ""]

    out += ["## Penalty-only groups: what the penalty was", "",
            f"A penalised attempt is a passed attempt with a reward below {top}. The frozen reward function "
            "splits its lost points into the part for malformed messages and the part for turns over the "
            "budget.", ""]
    rows = [
        ("Penalty-only groups", lambda p: whole(p["groups"])),
        ("Attempts in these groups", lambda p: whole(p["attempts"])),
        ("Penalised attempts in these groups (share of the attempts in these groups)",
         lambda p: count_share(p["penalised_attempts"], p["attempts"])),
        ("- lost points for a malformed message only",
         lambda p: whole(p["penalised_by_cause"]["malformed_message_only"])),
        ("- lost points for turns over the budget only",
         lambda p: whole(p["penalised_by_cause"]["turns_over_budget_only"])),
        ("- lost points for both", lambda p: whole(p["penalised_by_cause"]["both"])),
        ("Points lost for malformed messages, all penalised attempts together",
         lambda p: whole(p["points_lost_for_malformed_messages"])),
        ("Points lost for turns over the budget, all penalised attempts together",
         lambda p: whole(p["points_lost_for_turns_over_budget"])),
    ]
    out.append(table(["Number", *[run["label"] for run in runs]],
                     [[name, *[cell(run["signal"]["penalty_only"]) for run in runs]] for name, cell in rows]))

    out += ["## Groups with passes and fails: how many attempts failed", "",
            f"Unit: groups. Each group has {group_size} attempts.", ""]
    counts = sorted({int(count) for run in runs
                     for count in run["signal"]["pass_and_fail"]["groups_by_number_of_failed_attempts"]})
    out.append(table(["Failed attempts in the group", *[run["label"] for run in runs]],
                     [[whole(count), *[whole(run["signal"]["pass_and_fail"]
                                             ["groups_by_number_of_failed_attempts"].get(str(count), 0))
                                       for run in runs]] for count in counts]
                     + [["All groups with passes and fails",
                         *[whole(run["signal"]["pass_and_fail"]["groups"]) for run in runs]]]))

    out += ["## How each number is made", "",
            "- Kept group: the trainer's function `group_advantages` keeps the group for the saved rewards "
            "(made again here through `accessdesk/audit/formula.py`). The log holds the number of kept "
            "groups of each step, and it agrees.",
            "- Expected outcome, rule facts and \"two people of one name\": the fields `outcome`, `facts` "
            f"and `homonym` of the family record in `{report['inputs']['train_file']}`.",
            "- Groups sampled: every scenario is sampled once in a pass over the training scenarios, so a "
            "kind with more families has more groups.",
            "- These tables count groups and attempts. They do not say why an attempt failed.",
            ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# std_division.md
# ---------------------------------------------------------------------------
def pattern_words(rewards: list[dict]) -> str:
    """"15 x 100, 1 x 95" for a reward pattern, the most frequent reward first."""
    ordered = sorted(rewards, key=lambda item: (-item["attempts"], -item["reward"]))
    return ", ".join(f"{item['attempts']} x {item['reward']}" for item in ordered)


def two_sentences(report: dict) -> list[str]:
    """The two plain sentences on what the division would have changed, with the numbers of the runs."""
    runs = list(report["runs"].values())
    points = report["reward_settings"]["malformed_penalty"]
    first = ("With the division, the one odd attempt of a group would get the same weight in the loss "
             f"whether it lost {points} points for a malformed message or failed.")
    usable = all(run["std_division"]["by_type"]["penalty_only"]["groups"]
                 and run["std_division"]["by_type"]["pass_and_fail"]["groups"] for run in runs)
    if not usable:
        return [first, "At least one run here has no penalty-only group or no group with passes and fails, "
                       "so the shares of the two kinds cannot be compared."]
    def share(version: str) -> str:
        return per_run(report, lambda run: percent(
            run["std_division"]["penalty_only_share_of_total"][version]))

    divided, trainer = share("divided"), share("trainer")
    return [first, f"The penalty-only groups would then hold {divided} of the total absolute advantage, not "
                   f"{trainer}."]


def render_std_division(report: dict) -> str:
    runs = list(report["runs"].values())
    _, group_size = group_words(report)
    scale = report["std_division_forms"]["reward_scale"]
    check = report["std_division_forms"]["divided_check"]
    out = [
        "# The advantage with and without division by the standard deviation",
        "",
        f"An offline calculation on the kept groups of {run_list(report)}. Nothing was trained for it. Made "
        f"by `{SCRIPT}` from the saved rewards (`rollouts/step_*.episodes.jsonl`). No number in this file "
        "is from the dev set or the test set.",
        "",
        "## Words",
        "",
        f"- {GRPO_WORDS}",
        f"- {ATTEMPT_WORDS}",
        f"- Group: the {group_size} attempts of one scenario (one task in one wording) in one training "
        "step.",
        f"- {reward_words(report)}",
        "- Loss: the number that a training step makes smaller.",
        "- Advantage: the weight of one attempt in the loss of its step. A positive advantage makes the "
        "tokens of the attempt more likely, a negative one less likely.",
        "- Kept group: a group whose rewards are not all equal. Only attempts of kept groups have an "
        "advantage other than 0.",
        "- Group with passes and fails: a kept group with at least one passed and one failed attempt.",
        "- Penalty-only group: a kept group in which every attempt passed. Its rewards differ only by "
        "penalties.",
        "- Standard deviation of a group: how far the rewards of the group lie from their mean (the root of "
        "the mean squared distance).",
        "- Absolute advantage: the advantage without its sign.",
        "",
        "## The two versions",
        "",
        "- This trainer (`group_advantages` in `accessdesk/train/grpo_math.py`): advantage = reward minus "
        "the mean reward of the group. There is no division.",
        "- With division: advantage = (reward minus the mean reward of the group) divided by the standard "
        f"deviation of the group. {DIVISION_SOURCES}",
        "",
        f"The scale: a saved reward is a whole number. The trainer divides it by {scale} (`REWARD_SCALE` in "
        f"`accessdesk/train/grpo.py`), so a reward of {scale} is 1.0 and an advantage of 0.05 is "
        f"{0.05 * scale:g} reward points. The version with division has no reward scale: its unit is one "
        "standard deviation of the group.",
        "",
    ]
    examples = report["worked_examples"]
    if examples:
        out += ["## A worked example", "",
                "For each kind of kept group: the reward pattern that most groups of these runs have.", ""]
        rows = []
        for example in examples:
            with_pattern = sum(example["groups_with_this_pattern"].values())
            of_kind = sum(example["groups_of_this_kind"].values())
            for position, item in enumerate(example["advantage_by_reward"]):
                first = position == 0
                rows.append([
                    TYPE_WORDS[example["group_type"]] if first else "",
                    pattern_words(example["rewards"]) if first else "",
                    f"{whole(with_pattern)} of {whole(of_kind)}" if first else "",
                    f"reward {item['reward']} ({whole(item['attempts'])} "
                    f"{'attempt' if item['attempts'] == 1 else 'attempts'})",
                    signed(item["trainer"]), signed(item["divided"], 3)])
        out.append(table(["Kind of kept group", f"Rewards of the {group_size} attempts",
                          "Groups with this pattern, all runs", "Attempt",
                          "Advantage, this trainer (0 to 1)",
                          "Advantage with division (standard deviations)"], rows))
        out += ["Without division, the attempt that differs gets a small advantage (without its sign) if it "
                "lost points for a penalty, and a large one if it failed. With division, the pattern alone "
                "sets the advantage: the size of the reward difference does not count.", ""]

    out += ["## Mean absolute advantage of an attempt", "",
            "Unit of a row: the attempts of the kept groups of that kind. The mean is the sum of the "
            "absolute advantages divided by the number of attempts.", ""]
    rows = []
    for run in runs:
        for group_type in KEPT_TYPES:
            item = run["std_division"]["by_type"][group_type]
            if group_type == "all_fail" and not item["groups"]:
                continue
            rows.append([run["label"], TYPE_WORDS[group_type], whole(item["groups"]), whole(item["attempts"]),
                         decimal(item["mean_abs_advantage"]["trainer"], 4),
                         decimal(item["mean_abs_advantage"]["divided"], 3)])
    out.append(table(["Run", "Kind of kept group", "Groups", "Attempts",
                      "Mean absolute advantage, this trainer (0 to 1)",
                      "Mean absolute advantage with division (standard deviations)"], rows))
    ratios = [run["std_division"]["ratio_of_means_pass_and_fail_to_penalty_only"] for run in runs]
    if all(item["trainer"] is not None and item["divided"] is not None for item in ratios):
        plain = per_run(report, lambda run: decimal(
            run["std_division"]["ratio_of_means_pass_and_fail_to_penalty_only"]["trainer"], 1))
        divided = per_run(report, lambda run: decimal(
            run["std_division"]["ratio_of_means_pass_and_fail_to_penalty_only"]["divided"], 2))
        out += ["The mean of an attempt in a group with passes and fails, divided by the mean of an attempt "
                f"in a penalty-only group: this trainer {plain}; with division {divided}.", ""]

    out += ["## Share of the total absolute advantage of a run", "",
            "The total is the sum of the absolute advantages of all attempts of all kept groups of the run.",
            ""]
    rows = []
    for run in runs:
        numbers = run["std_division"]
        penalty = numbers["by_type"]["penalty_only"]["sum_abs_advantage"]
        rows.append([run["label"], whole(numbers["kept_groups"]),
                     count_share(numbers["by_type"]["penalty_only"]["groups"], numbers["kept_groups"]),
                     decimal(numbers["total_abs_advantage"]["trainer"], 1),
                     f"{decimal(penalty['trainer'], 2)} "
                     f"({percent(numbers['penalty_only_share_of_total']['trainer'])})",
                     decimal(numbers["total_abs_advantage"]["divided"], 1),
                     f"{decimal(penalty['divided'], 1)} "
                     f"({percent(numbers['penalty_only_share_of_total']['divided'])})"])
    out.append(table(["Run", "Kept groups", "Penalty-only groups (share of the kept groups)",
                      "Total, this trainer", "of it in penalty-only groups", "Total with division",
                      "of it in penalty-only groups"], rows))

    out += ["## What the division would have changed", "", " ".join(two_sentences(report)), ""]

    out += ["## Limits", "",
            "- This is arithmetic on saved rewards. No model was trained with the division, so its effect on "
            "the pass rate is not known.",
            "- The absolute advantage is the weight of an attempt in the loss. It is not the size of the "
            "change of the weights: the loss multiplies each advantage by the log-probabilities of the "
            "tokens of the attempt, and the optimiser (AdamW, `make_optimizer` in `accessdesk/train/sft.py`) "
            "rescales the gradient.",
            "- The standard deviation here has the number of attempts in its divisor, and nothing is added "
            f"to it. Check with another form (the number of attempts minus {check['ddof']} in the divisor, "
            f"and {check['eps']} added to the standard deviation): the penalty-only groups then hold "
            + per_run(report, lambda run: percent(
                run["std_division"]["penalty_only_share_of_total"]["divided_check"]))
            + " of the total absolute advantage.",
            ]
    out += [f"- {sentence}" for sentence in fault_lines(report)]
    zero = sum(run["std_division"]["kept_groups_with_zero_divisor"] for run in runs)
    if zero:
        out.append(f"- {whole(zero)} kept groups have a standard deviation of 0 (a `tie_tol` below 0 keeps "
                   "groups with equal rewards). Their divided advantages are counted as 0.")
    states = {state for run in runs
              for state in run["checks"]["advantage_code_against_the_run_manifest"].values()}
    code = " and ".join(f"`{name}`" for name in ADVANTAGE_CODE)
    if states == {"same"}:
        out.append("- The trainer saves no advantage. This script makes the advantages again from the saved "
                   f"rewards with the trainer's own function. The files {code} have the SHA-256 that each "
                   "run recorded in its `run_manifest.json`.")
    else:
        out.append("- The trainer saves no advantage. This script makes the advantages again from the saved "
                   "rewards with the function of today. Compared with the SHA-256 that each run recorded "
                   f"in its `run_manifest.json`, the files {code} are: {', '.join(sorted(states))}.")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# cost.md
# ---------------------------------------------------------------------------
def trial_lines(cost: dict) -> list[str]:
    """The bullet on the trial runs (`--smoke`): what their run manifests hold, by start time."""
    trials = cost["trial_runs"]
    if not trials["found"]:
        return [f"- No folder of trial runs was given or found ({trials['folder']})."]
    before, after, unknown = trials["before"], trials["after"], trials["no_start_time"]
    text = (f"- The trial runs in `{trials['folder']}` have {whole(trials['runs'])} complete run manifests "
            f"with seconds. They hold {decimal(trials['hours'])} hours ({whole(trials['attempts'])} "
            "attempts). By `started_utc` of the manifest: "
            f"{whole(before['runs'])} started before the freeze ({decimal(before['hours'])} hours), "
            f"{whole(after['runs'])} started after it ({decimal(after['hours'])} hours), and "
            f"{whole(unknown['runs'])} have no start time ({decimal(unknown['hours'])} hours). This table "
            "cannot place the runs without a start time before or after the freeze, so it adds no trial run "
            "to a row.")
    if trials["not_counted"]:
        text += f" {whole(trials['not_counted'])} more run manifests are not complete or have no seconds."
    return [text]


def purpose_lines(cost: dict) -> list[str]:
    """The section "What each rental was for": the rentals with their purposes, and the records of the test
    chain placed in a rental by the clock."""
    purposes = cost["purposes"]
    out = ["## What each rental was for", ""]
    if not purposes["found"]:
        out += [f"No file of purposes was found ({purposes['file']}), so this file does not say what ran on "
                "each rental.", ""]
    else:
        from_log = purposes.get("stops_from_the_work_log") or {}
        by_id = {rental["instance_id"]: rental for rental in cost["rentals"]}

        def stop_cell(rental: dict) -> str:
            if rental["instance_id"] in from_log:
                return f"{from_log[rental['instance_id']]} (work log)"
            return rental["stopped_utc"] or "still open"

        rows = [[rental["instance_id"], rental["rented_utc"], stop_cell(rental),
                 decimal(rental["hours"]),
                 usd_text(rental["usd_estimate"]),
                 purposes["purposes"].get(rental["instance_id"], f"not in {purposes['file']}")]
                for rental in cost["rentals"]]
        out.append(table(["Rental", "Rented (UTC)", "Stopped (UTC)", "Hours", "USD (estimate)", "Purpose"],
                         rows))
        text = (f"The purposes were written by hand from the work log (`{purposes['file']}`); the times, hours "
                "and amounts are those of the ledger")
        text += (", except a stop time marked \"(work log)\": that one is the time at which the work log says "
                 "the machine stopped. A stop time without a mark is the time of the stop row of the ledger."
                 if from_log else ". A stop time is the time of the stop row of the ledger.")
        if any(purpose == "not recorded" for purpose in purposes["purposes"].values()):
            text += " \"not recorded\" means that the log does not say what ran on that machine."
        out += [text, ""]
        for instance, stopped in from_log.items():
            rental = by_id.get(instance)
            if rental is None:
                continue
            line = (f"- Rental {instance}: the stop time {stopped} is from the work log. The stop row of the "
                    f"ledger was written at {rental['stopped_utc'] or 'no time'}")
            if rental["at_projected_limit"]:
                limit = rental.get("max_minutes")
                line += (", and the ledger charges the upper limit of the rental"
                         + (f": {limit:g} minutes (`max_minutes` of its rent row)" if limit is not None else "")
                         + f", {decimal(rental['hours'])} hours")
            out.append(line + ".")
        if from_log:
            out.append("")
        out += [f"- Not as in the ledger: {problem}." for problem in purposes["problems"]]
    lines = []
    for record in cost["records"]:
        if not record["rentals"]:
            where = "in no single rental of the ledger"
        else:
            where = "in rental " + " and ".join(record["rentals"])
        what = "The test run" if record["kind"] == "test run" else "A rehearsal of the test chain"
        lines.append(f"- {what}: the times of the {whole(record['lines_with_a_time'])} lines of "
                     f"`{record['record']}` lie {where}.")
    if not any(record["kind"] == "test run" for record in cost["records"]):
        lines.append("- The record of the test run (`test_run_record.txt` in the folder of the saved runs) was "
                     "not there when this file was made, so the test run is placed here only by the purposes "
                     "above, not by the clock.")
    out += ["By the clock (the record files of the test chain; only their times are read):", "", *lines, ""]
    return out


def render_cost(report: dict) -> str:
    cost = report["cost"]
    runs = list(report["runs"].values())
    rented = cost["rented"]
    price = cost["usd_per_hour_after_freeze"]
    left = cost["not_in_a_run_manifest"]
    spread = cost["usd_per_hour_of_the_rentals_after_freeze"]
    gpus = sorted({rental["gpu"] for rental in cost["rentals"]})
    out = [
        "# Compute and cost",
        "",
        f"Made by `{SCRIPT}` from `{cost['ledger']}` and from the seconds in the run manifests.",
        "",
        "**Every amount in USD is an estimate.** The ledger makes each amount from the hours of a rental, "
        "its price per hour and an estimate for downloads. The invoice of the GPU provider is the "
        "authority. This file does not replace it.",
        "",
        "## Words",
        "",
        f"- Rental: one rented machine with one GPU ({', '.join(gpus)}), from the time it was rented to "
        "the time it was stopped.",
        f"- Ledger: `{cost['ledger']}`. The rental script writes one row when a machine is rented and one "
        "row when it is stopped.",
        "- Freeze: the time at which the scenarios, the grader, the evaluation code and the analysis plan "
        f"were fixed (`frozen_at` in `MANIFEST.json`: {cost['frozen_at']}).",
        "- Run manifest: the file `run_manifest.json` that every run writes. It holds the seconds of the "
        "run.",
        "- GPU hours: hours of one GPU.",
        "- Attempt: one conversation of the model with the tools of the desk.",
        "- Rollout: a run that only samples attempts with one model. Dev and train are two sets of tasks: "
        "train is for training, dev is for evaluation.",
        "- Calibration: rollouts before the freeze that are in no results table. They chose the text, the "
        "token cap, the design of the tasks and the model (`ANALYSIS_PLAN.md`, section 10).",
        "- Stage 1: the rollouts of the untrained model (`base`), the first training (SFT: supervised "
        "fine-tuning on the model's own passing attempts) and the rollouts of the SFT model (`sft`).",
        "- Checks: three short runs. The overfit check trains GRPO on two tasks. The pilot makes a few "
        "GRPO steps at the real settings. The layout-gradient check measures gradients on saved attempts.",
        "- Stage 2: the second training (GRPO, which learns from the rewards of groups of attempts): "
        f"{joined([run['label'] for run in runs])}, each run with its rollouts. The dev curve is one small "
        "dev rollout for each of some checkpoints of a GRPO run. The second desk is an extra evaluation "
        "with a changed approval rule.",
        "- Group: the attempts of one scenario (one task in one wording) in one GRPO step. A group whose "
        "attempts all have the same reward is dropped: it gives no learning signal.",
        "",
        "## Cost by stage",
        "",
    ]
    rows = []
    for stage in cost["stages"]:
        attempts = "" if stage["attempts"] is None else whole(stage["attempts"])
        if stage["attempts_note"]:
            attempts += f" ({stage['attempts_note']})"
        rows.append([stage["stage"], attempts, decimal(stage["hours"]), usd_text(stage["usd"]),
                     stage["source"]])
    out.append(table(["Stage", "Attempts sampled", "GPU hours", "USD (estimate)", "The hours come from"],
                     rows))
    out += ["Rows are rounded; the total is the sum before rounding.", ""]
    calibration = cost["calibration"]
    across = cost["rentals_by_phase"]["across"]
    out += [
        "How to read this table:",
        "",
        "- The ledger has one amount for each rental. It does not say which run a rented hour paid for, so "
        "the ledger alone cannot give the cost of a stage."
        + (f" {whole(across)} rental was open across the freeze. It is split by the clock."
           if across == 1 else
           f" {whole(across)} rentals were open across the freeze. They are split by the clock."
           if across else ""),
        f"- Before the freeze the row is the rented time itself: {decimal(rented['before_freeze']['hours'])} "
        f"hours. The {whole(calibration['runs'])} calibration rollouts in "
        f"`{report['inputs']['calibration']}` hold {decimal(calibration['hours'])} of these hours "
        f"({whole(calibration['attempts'])} attempts). The other hours are in no run manifest of "
        f"`{report['inputs']['calibration']}`.",
        *trial_lines(cost),
        "- After the freeze the split is by the seconds of the run manifests. USD of a stage = its hours x "
        f"{decimal(price, 4)} USD per hour. That price is the estimated USD of the rented time after the "
        f"freeze ({usd_text(rented['after_freeze']['usd'])}) divided by its hours "
        f"({decimal(rented['after_freeze']['hours'])}). The run manifests of this table hold "
        f"{decimal(cost['hours_in_run_manifests_after_freeze'])} hours together.",
        "- The rentals that were open at some time after the freeze had prices from "
        f"{decimal(spread['lowest'], 4)} to {decimal(spread['highest'], 4)} USD per hour. The ledger does "
        "not say which run used which machine, so every stage gets the one mean price. The USD of a stage "
        "can be wrong by that difference in price.",
        "- The seconds of a rollout are the seconds of its sampling loop: the loading of the model is not "
        "in them. The seconds of a training run are those of the whole script.",
        "- \"Rented after the freeze, in no run manifest of this table\" is what is left of the rented "
        "time after the freeze. It holds the loading of the models for the rollouts, the time between "
        "runs, and every run that this table does not list.",
    ]
    later = cost["rentals_after_the_last_grpo_run"]
    if later and later["rentals"]:
        n_later = len(later["rentals"])
        out.append(
            f"- The rented time after the freeze holds {whole(n_later)} "
            f"{'rental' if n_later == 1 else 'rentals'} of {decimal(later['hours'])} hours that "
            f"{'was' if n_later == 1 else 'were'} rented after the last GRPO run had ended "
            f"({later['last_grpo_run_ended_utc']}, `ended_utc` of its run manifest). If no run of this "
            f"table ran in {'it' if n_later == 1 else 'them'}, "
            f"{decimal(later['hours_left_for_the_other_rentals'])} of "
            f"the {decimal(left['hours'])} hours are left for the other rentals. The run manifest of a "
            "rollout holds no start time, so this script cannot check that by the clock.")
    if cost["test_run_in_the_table"]:
        out.append("- The test rollouts are in the table with the seconds and the attempts of their run "
                   "manifests. No result of the test set is read here.")
    else:
        out.append("- The folders of the test rollouts were not there when this table was made, so the "
                   "test run is not a row of this table.")
    if cost["folders_without_run_manifest"]:
        names = ", ".join(f"`{name}`" for name in cost["folders_without_run_manifest"])
        out.append(f"- These folders of `{report['inputs']['main']}` hold no run manifest, so they are in no "
                   f"row of this table: {names}. This script has no seconds for them.")
    if left["hours"] < 0:
        out.append("- The run manifests hold more hours than the closed rentals after the freeze. A rental "
                   "is still open, or the ledger lacks a row.")
    for rental in cost["rentals"]:
        if rental["phase"] != "open":
            continue
        if rental["stopped_utc"] is None:
            out.append(f"- Rental {rental['instance_id']} is still open: the ledger had no stop row for it when "
                       "this file was made. It is not yet in the total, and in no amount of this file.")
        else:
            out.append(f"- Rental {rental['instance_id']} has a stop row with no amount in the ledger. It is "
                       "not in the total, and in no amount of this file.")
    out += [f"- {whole(cost['run_manifests_on_the_frozen_tree'])} of the "
            f"{whole(cost['run_manifests_counted'])} runs after the freeze say in their manifest that they "
            "started on the frozen tree.", ""]

    out += ["## Detail of the stages after the freeze", "",
            "Source: `seconds` and `summary.episodes` of each `run_manifest.json`.", ""]
    rows = [[row["stage"], row["part"], whole(row["runs"]), whole(row["attempts"]), decimal(row["hours"]),
             usd_text(row["usd"])] for row in cost["stage_rows"]]
    out.append(table(["Stage", "Part", "Runs", "Attempts sampled", "GPU hours", "USD (estimate)"], rows))
    missing = [name for row in cost["stage_rows"] for name in row["not_counted"]]
    if missing:
        out += ["Not counted, because the folder is missing, the run is not complete or it has no seconds: "
                + ", ".join(f"`{name}`" for name in missing) + ".", ""]

    out += ["## The ledger, rental by rental", "",
            "Hours: (USD estimate minus the download estimate) divided by the price per hour. These are the "
            "hours that the estimate charges. Rented and Stop row: the times of the rent row and of the stop "
            "row of the ledger.", ""]
    rows = []
    for rental in cost["rentals"]:
        note = rental["note"] or ""
        if rental["at_projected_limit"]:
            note = (note + " " if note else "") + "(The amount is the upper limit of the rental, not a " \
                                                  "measured time.)"
        rows.append([rental["instance_id"], rental["rented_utc"], rental["stopped_utc"] or "still open",
                     decimal(rental["hours"]), decimal(rental["usd_per_hour"], 4),
                     usd_text(rental["usd_estimate"]), phase_words(rental), note])
    rows.append(["All closed rentals", "", "", decimal(rented["total"]["hours"]), "",
                 usd_text(rented["total"]["usd"]), "", ""])
    out.append(table(["Rental", "Rented (UTC)", "Stop row (UTC)", "Hours", "USD per hour", "USD (estimate)",
                      "By the clock", "What the ledger notes"], rows))
    out += [f"Rented time before the freeze: {decimal(rented['before_freeze']['hours'])} hours, "
            f"{usd_text(rented['before_freeze']['usd'])} USD. After the freeze: "
            f"{decimal(rented['after_freeze']['hours'])} hours, {usd_text(rented['after_freeze']['usd'])} "
            "USD. A rental across the freeze counts on both sides, by the share of its clock time on each "
            "side.", ""]
    out += purpose_lines(cost)

    out += ["## GRPO: sampled attempts that gave no learning signal", "",
            "Sources: `rollouts/step_*.episodes.jsonl` and `train_log.jsonl` of each GRPO run.", ""]
    grpo_rows = [
        ("Attempts sampled in training", lambda g: whole(g["attempts"])),
        ("Attempts in dropped groups (share of the attempts sampled)",
         lambda g: count_share(g["attempts_in_dropped_groups"], g["attempts"])),
        ("Sampled tokens in dropped groups",
         lambda g: count_of(g["sampled_tokens_in_dropped_groups"], g["sampled_tokens"])),
        ("Steps with no kept group", lambda g: count_of(g["steps_without_kept_group"], g["steps"])),
        ("GPU hours in the steps with no kept group (share of the hours in the steps)",
         lambda g: f"{hours_text(g['seconds_in_steps_without_kept_group'])} "
                   f"({percent(g['share_of_step_seconds'])})"),
        ("USD of these hours (estimate)", lambda g: usd_text(g["usd_of_steps_without_kept_group"])),
        ("GPU hours in the steps: sampling; update",
         lambda g: f"{hours_text(g['sampling_seconds'])}; {hours_text(g['update_seconds'])}"),
    ]
    out.append(table(["Number", *[run["label"] for run in runs]],
                     [[name, *[cell(cost["grpo"][run["name"]]) for run in runs]]
                      for name, cell in grpo_rows]))
    zero = by_run(report, lambda run: f"{whole(run['totals']['steps_without_kept_group_and_zero_gradient'])} "
                                      f"of {whole(run['totals']['steps_without_kept_group'])}")
    out += [
        "- A dropped group is a group whose attempts all have the same reward. It still had to be sampled: "
        "only its rewards show that they are all equal.",
        "- A step with no kept group has no gradient of its own. The log holds the loss 0 and the gradient "
        f"norm 0 for these steps ({zero}). The trainer still makes the optimiser step, so the weights can "
        "move a little from the momentum of earlier steps (`accessdesk/train/grpo.py`).",
        "- The time of a step is not split between its groups, because the attempts of a step are sampled "
        "together.",
        "",
        "## How each number is made", "",
        "- USD (estimate) of a rental: `actual_usd` of its stop row in the ledger.",
        "- GPU hours of a run: `seconds` of its run manifest divided by 3600. For the layout-gradient check: "
        "the seconds of its stages in `layout_gradients.json`.",
        "- Attempts sampled: `summary.episodes` of the run manifest. SFT and the layout-gradient check "
        "sample no attempt: they read saved attempts.",
        f"- The ledger had {whole(cost['ledger_rows'])} rows when this file was made (SHA-256 "
        f"`{cost['ledger_sha256'][:12]}`).",
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def steps_csv(rows: list[dict]) -> str:
    """The step rows of one run as CSV text, with the columns of STEP_COLUMNS. None is an empty cell."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    columns = [name for name, _, _ in STEP_COLUMNS]
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row[name] is None else row[name] for name in columns])
    return buffer.getvalue()


def output_files(report: dict) -> dict[str, str]:
    """File name -> text, for every output file."""
    files = {f"steps_{name}.csv": steps_csv(run["steps"]) for name, run in report["runs"].items()}
    files.update({"summary.md": render_summary(report), "signal.md": render_signal(report),
                  "std_division.md": render_std_division(report), "cost.md": render_cost(report),
                  JSON_NAME: json.dumps(report, indent=2, ensure_ascii=False) + "\n"})
    return files


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "FOLDER"), default=None,
                        help="a GRPO run: a short name and its folder; once per run (default: seed1 and "
                             "seed2 of results/main)")
    parser.add_argument("--main", type=Path, default=ROOT / "results" / "main",
                        help="the folder of the saved runs of stage 1 and of the checks")
    parser.add_argument("--calibration", type=Path, default=ROOT / "results" / "calibration",
                        help="the folder of the calibration runs")
    parser.add_argument("--smoke", type=Path, default=ROOT / "results" / "_smoke",
                        help="the folder of the trial runs; only seconds, summary.episodes and started_utc of "
                             "its run manifests are read")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results" / "spend_ledger.csv",
                        help="the spend ledger")
    parser.add_argument("--rentals", type=Path, default=None,
                        help="what each rental was for (default: rentals.csv beside the ledger); optional")
    parser.add_argument("--test-record", type=Path, default=None,
                        help="the record of the test run (default: <main>/test_run_record.txt); optional")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "data" / "scenarios",
                        help=f"the scenario folder of the runs; only its {TRAIN_FILE} is read")
    parser.add_argument("--freeze-manifest", type=Path, default=ROOT / "MANIFEST.json",
                        help="the manifest of the freeze; only frozen_at is read")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "main" / "report" / "training",
                        help="the folder for the output files")
    parser.add_argument("--block-steps", type=int, default=10,
                        help="steps in one block of the pass-rate tables")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the output files. Returns the exit code: 0, or 2 if nothing was written."""
    args = parse_args(argv)
    runs = [(name, Path(folder)) for name, folder in args.run] if args.run else \
        [(name, ROOT / folder) for name, folder in DEFAULT_RUNS]
    try:
        report = build_report(runs, scenarios=args.scenarios, main=args.main, calibration=args.calibration,
                              ledger=args.ledger, freeze_manifest=args.freeze_manifest,
                              block_steps=args.block_steps, smoke=args.smoke,
                              rentals_file=args.rentals if args.rentals is not None
                              else args.ledger.parent / "rentals.csv",
                              test_record=args.test_record if args.test_record is not None
                              else args.main / "test_run_record.txt")
        files = output_files(report)
    except (OSError, ValueError) as error:
        print(f"training_report: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} files into {args.out}: " + ", ".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
