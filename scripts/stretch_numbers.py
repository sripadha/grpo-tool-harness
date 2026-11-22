"""The stretch item of the brief: its numbers, its tables and its document. The experiment was NOT run.

The stretch item: add a bonus when reworded versions of the same scenario produce the same correct action,
and test whether it helps beyond a reward for correctness alone. This script runs no model and no training.
It only reads saved runs, and it writes three files:

    <out>/stretch_numbers.json   every number, with the files that it was made from
    <out>/stretch_numbers.md     the same numbers as tables; each table states its rule, unit and source
    <doc>                        the document for the reviewer (results/stretch.md); it quotes the tables

    .venv/bin/python scripts/stretch_numbers.py

--main       the folder of the saved runs (default: results/main). The runs are read under these names:
                 base_dev, sft_dev, <grpo>_dev   the attempts of each arm on dev (episodes.jsonl and
                                                 run_manifest.json)
                 <grpo>                          a GRPO training run (run_manifest.json, train_log.jsonl,
                                                 rollouts/step_*.episodes.jsonl)
                 sft_train                       attempts of the SFT model on training families
                 tables                          the tables of `python -m accessdesk.eval.tables` (optional)
                 base_test, sft_test, <grpo>_test   the attempts on the test set. They are read only if every
                                                 one of them is there and complete; else the files say that
                                                 the test set is not in them
--grpo       the name of a GRPO arm; give it once per arm (default: grpo_seed1 and grpo_seed2)
--folder     KEY=PATH: another place for one of the names above, for example sft_dev=/some/folder
--ledger     the spend ledger (default: results/spend_ledger.csv). Only the time and the hourly price of its
             "rent" lines are read, and only the lines up to the end of the last saved GRPO training run
             (`ended_utc` of its run manifest) are used, so a machine rented later does not change any output.
             Without the file, or without `ended_utc`, no cost in USD is given
--scenarios  the scenario folder (default: data/scenarios). Only dev.jsonl is read, for the facts of a family
             that got worse. No other split file is opened. Without the file these facts are left out
--plan       the frozen analysis plan (default: ANALYSIS_PLAN.md). Only its section 3 (the arms and the
             planned extras) is read: section 6 of the document says that the plan does not list this
             experiment only if that section exists, has a "Planned extras" paragraph and holds none of the
             words of PLAN_WORDS. Without the file that sentence is left out
--out        the folder for the two number files (default: <main>/report/stretch)
--doc        the document (default: <main>/../stretch.md; with the default --main this is results/stretch.md)
--draws, --seed, --level   the bootstrap (default: 10,000 draws, seed 0, 95%: ANALYSIS_PLAN.md section 4)
--bonus, --attempts-per-wording, --families-per-step, --seeds-per-arm
             the numbers of the design (default: 20 points, 4, 10, 3). They are judgements, not results

Where ANALYSIS_PLAN.md section 5 defines a metric, the frozen function computes it here (`pass_rate`,
`all_wordings_pass`, `variance_shares`, `family_bootstrap_ci` of accessdesk/eval/metrics.py; the table rows
of accessdesk/eval/tables.py; `paired_family_bootstrap` through accessdesk.impl). The advantages come from
the function that the training used (`group_advantages` through accessdesk.impl, on the reward matrix of
`reward_matrix` in accessdesk/train/grpo.py). No metric is defined a second time.

The script writes nothing and stops with exit code 2, with the reason on stderr, if
  - a run that it needs is missing, is not complete, holds another split, or holds more than one run;
  - the arms were not run on the same scenarios and settings (the checks of `tables.compare_arms`);
  - a saved table (in `tables`) gives another value than this script for the same arm and split;
  - the saved attempts of a training step do not fit its log line (scenarios, number of kept groups);
  - the scenario file is not the file that the dev runs name.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.env.scenarios import split_scenario_id  # noqa: E402
from accessdesk.eval import metrics, tables  # noqa: E402
from accessdesk.train import grpo  # noqa: E402

JSON_NAME, TABLES_NAME = "stretch_numbers.json", "stretch_numbers.md"
DOC_NAME = "stretch.md"
GRPO_ARMS = ("grpo_seed1", "grpo_seed2")
BASE, SFT = "base", "sft"
DEV, TRAIN, TEST = "dev", "train", "test"

# --- The numbers of the design. Every one is a judgement: no run chose it, and no source gives it. ---
BONUS_POINTS = 20            # the largest bonus of a passing attempt, in points (a pass is 100 points)
ATTEMPTS_PER_WORDING = 4     # attempts of each wording in one family group
FAMILIES_PER_STEP = 10       # family groups in one training step
SEEDS_PER_ARM = 3            # training runs of each arm
N_ARMS = 2                   # with and without the bonus: this one is the brief's
# --- The two examples. Made-up groups, chosen to be easy to follow. ---
EXAMPLE_PASSES, EXAMPLE_FAILS = 12, 4                 # one group of one wording
EXAMPLE_FAMILY = (("A", Fraction(1)), ("B", Fraction(1, 2)), ("C", Fraction(0)))
#                                                       one family group: (wording, its share of passes)
# --- Our rule for "the wordings of a family disagree": one style passes at least HIGH of its attempts and
#     another style at most LOW. With 8 attempts: 5 or more against 3 or fewer. ---
HIGH, LOW = Fraction(5, 8), Fraction(3, 8)
# --- Our word for the room that is left: "little" below this share of all-wordings-pass (0.05 is 5 points).
#     A judgement. ---
LITTLE_ROOM = 0.05
# --- The words that would show that section 3 of the analysis plan lists this experiment (lower case). ---
PLAN_WORDS = ("stretch", "bonus", "reworded", "rewording", "consisten", "agreement", "same correct action")
TOLERANCE = 1e-9             # two advantages, or two values of one metric, closer than this are the same
MAX_FAMILIES_IN_DOC = 3      # the document shows at most this many families in full; the table file all

# What is kept of an attempt record. `turns` and `final_state` are large and no number here reads them
# (the close of the request is kept apart, as `final_request`).
KEPT_KEYS = ("schema", "run", "arm", "split", "family_id", "style", "scenario_id", "outcome", "ruleset",
             "toolset", "prompt", "sample", "seed", "n_turns", "n_malformed", "closed", "ended_by", "passed",
             "reward", "impl", "text_sha256", "text_is_placeholder", "config_sha256", "settings",
             "harness_error", "data_sha256")
NEEDED_KEYS = ("split", "family_id", "style", "scenario_id", "sample", "passed", "reward", "harness_error")
# The code that computes the numbers here. The output names the SHA-256 of each of these files.
CODE_FILES = ("scripts/stretch_numbers.py", "accessdesk/eval/metrics.py", "accessdesk/eval/tables.py",
              "accessdesk/eval/bootstrap.py", "accessdesk/impl.py", "accessdesk/train/grpo_math.py",
              "accessdesk/train/grpo.py")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    """The SHA-256 of the file's bytes, as 64 hexadecimal digits. The file is read in pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def shown_path(path: Path) -> str:
    """The path as the files show it: relative to the repository root, with "/", if it is inside the
    repository; else as it was given."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"{shown_path(path)} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def read_attempts(path: Path) -> list[dict]:
    """The attempt records of one episodes file, in file order. The file is read line by line, and only the
    keys that the numbers need are kept. Raises ValueError if the file is missing or a record lacks a key."""
    if not path.is_file():
        raise ValueError(f"{shown_path(path)} does not exist")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            full = json.loads(line)
            missing = [key for key in NEEDED_KEYS if key not in full]
            if missing:
                raise ValueError(f"{shown_path(path)}: a record has no {missing}")
            record = {key: full[key] for key in KEPT_KEYS if key in full}
            record["final_request"] = (full.get("final_state") or {}).get("request")
            records.append(record)
    return records


def load_run(folder: Path, split: str) -> dict:
    """One saved rollout: its records, its run manifest and what the output says about it.

    Raises ValueError if a file is missing, the run manifest does not say `complete`, a record is of another
    split than `split`, or the file holds more than one run or set-up."""
    episodes, manifest_path = folder / "episodes.jsonl", folder / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ValueError(f"{shown_path(manifest_path)} does not say that the run is complete")
    records = read_attempts(episodes)
    if not records:
        raise ValueError(f"{shown_path(episodes)} has no record")
    splits = sorted({record["split"] for record in records})
    if splits != [split]:
        raise ValueError(f"{shown_path(episodes)} must hold records of the split {split!r} only; it holds "
                         f"{splits}")
    setups = sorted({tuple(str(record.get(key)) for key in ("run", "arm", "ruleset", "toolset", "prompt"))
                     for record in records})
    if len(setups) != 1:
        raise ValueError(f"{shown_path(episodes)} holds {len(setups)} runs or set-ups: {setups}")
    info = {"episodes": shown_path(episodes), "episodes_sha256": sha256_of_file(episodes),
            "run_manifest": shown_path(manifest_path), "run": records[0].get("run"),
            "arm_in_records": records[0].get("arm"), "adapter": manifest.get("adapter"),
            "run_seed": manifest.get("run_seed"), "seconds": manifest.get("seconds"),
            "gpu": (manifest.get("environment") or {}).get("gpu_name"),
            "attempts": len(records), "harness_errors": metrics.harness_error_count(records)}
    return {"records": records, "manifest": manifest, "info": info}


# ---------------------------------------------------------------------------
# Small helpers on records
# ---------------------------------------------------------------------------
def by_family(records: list[dict]) -> dict[str, list[dict]]:
    """family id -> its records (harness errors included), families in sorted order."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["family_id"], []).append(record)
    return {family_id: groups[family_id] for family_id in sorted(groups)}


def passes_by_style(records: list[dict]) -> dict[int, list[int]]:
    """style -> [passed attempts, valid attempts] of the records of ONE family, styles in rising order."""
    counts: dict[int, list[int]] = {}
    for record in metrics.valid_records(records):
        cell = counts.setdefault(record["style"], [0, 0])
        cell[0] += int(bool(record["passed"]))
        cell[1] += 1
    return {style: counts[style] for style in sorted(counts)}


def styles_disagree(by_style: dict[int, list[int]]) -> bool:
    """Our rule: one style passes at least HIGH of its valid attempts and another style at most LOW."""
    shares = [Fraction(passed, valid) for passed, valid in by_style.values() if valid]
    return len(shares) >= 2 and max(shares) >= HIGH and min(shares) <= LOW


def family_values(records: list[dict]) -> dict[str, float]:
    """family id -> the family's part of `metrics.all_wordings_pass`: the product of the pass rates of its
    wording styles. The frozen function is called on the records of one family, so its mean is over one
    family and is that product. A family with no valid record is absent."""
    values = {}
    for family_id, group in by_family(records).items():
        if metrics.valid_records(group):
            values[family_id] = metrics.all_wordings_pass(group)
    return values


def _number(value) -> float:
    return float(value)


def _same(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= TOLERANCE


# ---------------------------------------------------------------------------
# Part 2: a bonus inside a group of one wording
# ---------------------------------------------------------------------------
def one_wording_example(n_passes: int, n_fails: int, bonus_points: float, pass_reward: int = 100) -> dict:
    """A made-up group of one wording, with and without a bonus on every pass. Points throughout.

    The advantages come from the function of the training (`group_advantages`)."""
    advantages_of = impl.get("group_advantages")
    rows = {}
    for label, reward_of_pass in (("without", pass_reward), ("with", pass_reward + bonus_points)):
        rewards = np.array([[reward_of_pass] * n_passes + [0] * n_fails], dtype=float) / grpo.REWARD_SCALE
        advantages, keep = advantages_of(rewards, 0.0)
        rows[label] = {"reward_of_a_pass": _number(reward_of_pass), "reward_of_a_fail": 0.0,
                       "mean_of_the_group": _number(rewards.mean() * grpo.REWARD_SCALE),
                       "advantage_of_a_pass": _number(advantages[0][0] * grpo.REWARD_SCALE),
                       "advantage_of_a_fail": _number(advantages[0][-1] * grpo.REWARD_SCALE),
                       "kept": bool(keep[0])}
    return {"passes": n_passes, "fails": n_fails, "attempts": n_passes + n_fails,
            "bonus_points": bonus_points, **rows,
            "factor": _number(rows["with"]["advantage_of_a_pass"] / rows["without"]["advantage_of_a_pass"]),
            "factor_by_formula": _number((pass_reward + bonus_points) / pass_reward)}


def _same_order(before: np.ndarray, after: np.ndarray) -> bool:
    """True if every pair of attempts is ranked the same way by both lists of advantages."""
    first = np.sign(np.round(before[:, None] - before[None, :], 9))
    second = np.sign(np.round(after[:, None] - after[None, :], 9))
    return bool((first == second).all())


def saved_groups_with_bonus(folder: Path, bonus_points: float) -> dict:
    """What a bonus on every pass would have changed in the groups of one saved GRPO run. Nothing is trained.

    For every step of `train_log.jsonl`, the saved attempts of the step give the reward matrix of the
    training (`grpo.reward_matrix`), and `group_advantages` is run on it twice: as it is, and with
    `bonus_points` added to the reward of every passing attempt. Unit of every count: the group (the attempts
    of one scenario in one step).

    Raises ValueError if a step's file does not fit its log line, or if a kept group does not behave as the
    arithmetic says (same groups kept, same order of the attempts)."""
    manifest = read_json(folder / "run_manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError(f"{shown_path(folder / 'run_manifest.json')} does not say that the run is complete")
    config = manifest["grpo_config"]
    n_scenarios, group_size, tie_tol = config["n_scenarios"], config["group_size"], config["tie_tol"]
    pass_reward = (manifest.get("config") or {}).get("pass_reward", 100)
    factor = (pass_reward + bonus_points) / pass_reward
    advantages_of = impl.get("group_advantages")
    log_path = folder / grpo.LOG_NAME
    if not log_path.is_file():
        raise ValueError(f"{shown_path(log_path)} does not exist")
    log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not log:
        raise ValueError(f"{shown_path(log_path)} has no step")
    counts = Counter()
    lowest_pass: list[float] = []
    rollout_hashes = []
    for row in log:
        path = folder / "rollouts" / f"step_{row['step']:05d}.episodes.jsonl"
        records = read_attempts(path)
        rollout_hashes.append(f"{sha256_of_file(path)}  {path.name}\n")
        try:
            rewards = grpo.reward_matrix(records, n_scenarios, group_size)
        except ValueError as error:
            raise ValueError(f"{shown_path(path)}: {error}") from None
        scenario_ids = [records[group * group_size]["scenario_id"] for group in range(n_scenarios)]
        if scenario_ids != row["scenarios"]:
            raise ValueError(f"{shown_path(path)} does not hold the scenarios of step {row['step']} of the "
                             "log")
        passed = np.array([bool(record["passed"]) and record["harness_error"] is None for record in records],
                          dtype=float).reshape(n_scenarios, group_size)
        with_bonus = rewards + (bonus_points / grpo.REWARD_SCALE) * passed
        before, keep = advantages_of(rewards, tie_tol)
        after, keep_with_bonus = advantages_of(with_bonus, tie_tol)
        if int(np.sum(keep)) != row["kept_groups"]:
            raise ValueError(f"step {row['step']} of {shown_path(folder)}: the saved attempts give "
                             f"{int(np.sum(keep))} kept groups, the log says {row['kept_groups']}")
        families_of_step = Counter(split_scenario_id(scenario_id)[0] for scenario_id in scenario_ids)
        for group in range(n_scenarios):
            valid = ~np.isnan(rewards[group])
            is_pass = passed[group][valid] > 0
            counts["groups"] += 1
            counts["attempts"] += group_size
            pass_points = rewards[group][valid][is_pass] * grpo.REWARD_SCALE
            counts["passes"] += int(is_pass.sum())
            counts["passes_below_full_reward"] += int(np.sum(pass_points < pass_reward - TOLERANCE))
            if pass_points.size:
                lowest_pass.append(float(pass_points.min()))
            fail_points = rewards[group][valid][~is_pass] * grpo.REWARD_SCALE
            counts["fails_not_at_zero"] += int(np.sum(np.abs(fail_points) > TOLERANCE))
            counts["harness_errors"] += int((~valid).sum())
            counts["kept"] += int(keep[group])
            counts["kept_with_bonus"] += int(keep_with_bonus[group])
            counts["keep_decision_changes"] += int(keep[group] != keep_with_bonus[group])
            counts["with_a_pass_and_a_fail"] += int(is_pass.any() and (~is_pass).any())
            counts["with_another_wording_of_the_family_in_the_step"] += int(
                families_of_step[split_scenario_id(scenario_ids[group])[0]] > 1)
            if not keep[group]:
                continue
            old, new = before[group][valid] * grpo.REWARD_SCALE, after[group][valid] * grpo.REWARD_SCALE
            points = rewards[group][valid] * grpo.REWARD_SCALE
            counts["kept_order_changes"] += int(not _same_order(old, new))
            shift = new - old
            if is_pass.all() or not is_pass.any():
                # Every attempt passes (the rewards differ by penalties), or every attempt fails (possible
                # only with penalties on fails): the bonus moves every reward and the mean by one amount.
                counts["kept_one_outcome"] += 1
                counts["kept_one_outcome_unchanged"] += int(np.allclose(shift, 0.0, atol=TOLERANCE))
            elif np.allclose(points[is_pass], pass_reward) and np.allclose(points[~is_pass], 0.0):
                counts["kept_mixed_full_reward"] += 1
                counts["kept_mixed_full_reward_scaled"] += int(np.allclose(new, factor * old, atol=TOLERANCE))
            else:
                counts["kept_mixed_other"] += 1
                counts["kept_mixed_other_two_shifts"] += int(
                    np.allclose(shift[is_pass], shift[is_pass][0], atol=TOLERANCE)
                    and np.allclose(shift[~is_pass], shift[~is_pass][0], atol=TOLERANCE))
    for done, of in (("kept_one_outcome_unchanged", "kept_one_outcome"),
                     ("kept_mixed_full_reward_scaled", "kept_mixed_full_reward"),
                     ("kept_mixed_other_two_shifts", "kept_mixed_other")):
        if counts[done] != counts[of]:
            raise ValueError(f"{shown_path(folder)}: {counts[of] - counts[done]} kept groups ({of}) do not "
                             "change as the arithmetic of a bonus on every pass says")
    if counts["keep_decision_changes"] or counts["kept_order_changes"]:
        raise ValueError(f"{shown_path(folder)}: a bonus on every pass changed which groups are kept or the "
                         "order of the attempts of a kept group; the arithmetic says that it cannot")
    names = ("groups", "attempts", "harness_errors", "kept", "kept_with_bonus", "keep_decision_changes",
             "kept_order_changes", "kept_mixed_full_reward", "kept_one_outcome", "kept_mixed_other",
             "with_a_pass_and_a_fail", "with_another_wording_of_the_family_in_the_step", "passes",
             "passes_below_full_reward", "fails_not_at_zero")
    return {
        "folder": shown_path(folder), "status": manifest.get("status"), "seed": config.get("seed"),
        "steps": len(log), "scenarios_per_step": n_scenarios, "attempts_per_group": group_size,
        "attempts_per_step": n_scenarios * group_size, "learning_rate": config.get("lr"),
        "tie_tol": tie_tol, "styles": manifest.get("styles"), "pass_reward": pass_reward,
        "start_adapter": (manifest.get("start_adapter") or {}).get("folder"),
        "bonus_points": bonus_points, "factor": factor,
        **{name: int(counts[name]) for name in names},
        "lowest_pass_reward": _number(min(lowest_pass)) if lowest_pass else None,
        "seconds": manifest.get("seconds"), "seconds_in_steps": _number(sum(row["seconds"] for row in log)),
        "ended_utc": manifest.get("ended_utc"),
        "gpu": (manifest.get("environment") or {}).get("gpu_name"),
        "train_log_sha256": sha256_of_file(log_path),
        "run_manifest_sha256": sha256_of_file(folder / "run_manifest.json"),
        "rollout_files": len(rollout_hashes),
        "rollout_files_sha256": hashlib.sha256("".join(rollout_hashes).encode("utf-8")).hexdigest(),
    }


def refuses_a_group_of_two_scenarios() -> bool:
    """True if the reward matrix of the saved training code refuses a group that holds two scenarios."""
    probe = [{"scenario_id": "tr-000/s1", "harness_error": None, "reward": 100},
             {"scenario_id": "tr-000/s2", "harness_error": None, "reward": 0}]
    try:
        grpo.reward_matrix(probe, 1, 2)
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# Part 4: the family group of the design
# ---------------------------------------------------------------------------
def family_group(rewards: list, passed: list[bool], wordings: list, bonus_points: float,
                 tie_tol: float = 0.0) -> dict:
    """The advantages of ONE family group of the design, in both arms. Points throughout. Nothing is trained.

    rewards    the reward of each attempt in points; None for an attempt that the harness lost
    passed     pass or fail of each attempt
    wordings   the wording (style) of each attempt

    Arm 1: the reward as it is. Arm 2: a passing attempt also gets  bonus_points * (q - q_mean), where q is
    the share of passes among the family's valid attempts in the OTHER wordings, and q_mean is the mean of q
    over the family's passing attempts. So the bonuses of a group add up to zero. A pass with no valid attempt
    in another wording gets no bonus. The bonus is made with exact fractions, so that equal shares give a
    bonus of exactly zero. In both arms the advantage is `group_advantages` of the 12 (or so) rewards as one
    group: the reward minus the mean of the family's valid attempts. A group with equal rewards is dropped.
    """
    n = len(rewards)
    valid = [reward is not None for reward in rewards]
    shares: dict[int, Fraction] = {}
    for i in range(n):
        if not (valid[i] and passed[i]):
            continue
        others = [j for j in range(n) if valid[j] and wordings[j] != wordings[i]]
        if others:
            shares[i] = Fraction(sum(1 for j in others if passed[j]), len(others))
    mean_share = sum(shares.values(), Fraction(0)) / len(shares) if shares else None
    bonus = [float(Fraction(bonus_points) * (shares[i] - mean_share)) if i in shares else 0.0
             for i in range(n)]
    advantages_of = impl.get("group_advantages")
    arm1 = np.array([[np.nan if reward is None else reward / grpo.REWARD_SCALE for reward in rewards]])
    arm2 = np.array([[np.nan if reward is None else (reward + extra) / grpo.REWARD_SCALE
                      for reward, extra in zip(rewards, bonus)]])
    advantages1, keep1 = advantages_of(arm1, tie_tol)
    advantages2, keep2 = advantages_of(arm2, tie_tol)
    return {"valid": valid, "share_of_passes_in_the_other_wordings": {i: float(q) for i, q in shares.items()},
            "mean_share": None if mean_share is None else float(mean_share), "bonus": bonus,
            "arm1": [float(value * grpo.REWARD_SCALE) for value in advantages1[0]],
            "arm2": [float(value * grpo.REWARD_SCALE) for value in advantages2[0]],
            "kept_arm1": bool(keep1[0]), "kept_arm2": bool(keep2[0])}


def family_example(bonus_points: float, attempts_per_wording: int) -> dict:
    """The made-up family group of the document: the wordings of EXAMPLE_FAMILY, each with
    `attempts_per_wording` attempts, of which its share passes (rounded down): 4, 2 and 0 of 4."""
    n_passes_of = {wording: int(share * attempts_per_wording) for wording, share in EXAMPLE_FAMILY}
    rewards, passed, wordings = [], [], []
    for wording, n_passes in n_passes_of.items():
        for attempt in range(attempts_per_wording):
            wordings.append(wording)
            passed.append(attempt < n_passes)
            rewards.append(100 if attempt < n_passes else 0)
    group = family_group(rewards, passed, wordings, bonus_points)
    by_wording = {}
    for wording, n_passes in n_passes_of.items():
        first = wordings.index(wording)
        row = {"passes": n_passes, "attempts": attempts_per_wording}
        if n_passes:
            row.update({"share_of_passes_in_the_other_wordings":
                        group["share_of_passes_in_the_other_wordings"][first],
                        "bonus_of_a_pass": group["bonus"][first],
                        "arm1_advantage_of_a_pass": group["arm1"][first],
                        "arm2_advantage_of_a_pass": group["arm2"][first]})
        if n_passes < attempts_per_wording:
            last = first + attempts_per_wording - 1
            row.update({"arm1_advantage_of_a_fail": group["arm1"][last],
                        "arm2_advantage_of_a_fail": group["arm2"][last]})
        by_wording[wording] = row
    return {"wordings": by_wording, "mean_share": group["mean_share"], "bonus_points": bonus_points,
            "sum_of_the_bonuses": float(sum(group["bonus"])), "attempts": len(rewards)}


def design_on_saved_attempts(records: list[dict], attempts_per_wording: int, n_wordings: int,
                             bonus_points: float) -> list[dict]:
    """The arithmetic of the design on saved attempts of the SFT model on training families: one line per
    draw. Nothing is trained: this shows how often, and by how much, the bonus would change an advantage at
    the start of both arms.

    A draw: for each family, `attempts_per_wording` saved attempts of each of its wordings are one family
    group (draw 1: the lowest sample numbers; draw 2: the next ones; and so on, while every wording has
    enough). A family that does not have exactly `n_wordings` wordings is left out and counted. Unit of the
    counts: the family group."""
    families = {}
    left_out = 0
    for family_id, group in by_family(records).items():
        per_style: dict[int, list[dict]] = {}
        for record in sorted(group, key=lambda r: (r["style"], r["sample"])):
            per_style.setdefault(record["style"], []).append(record)
        if len(per_style) != n_wordings:
            left_out += 1
            continue
        families[family_id] = per_style
    if not families:
        raise ValueError(f"no family of the saved attempts has exactly {n_wordings} wordings")
    n_draws = min(len(attempts) // attempts_per_wording for per_style in families.values()
                  for attempts in per_style.values())
    if n_draws < 1:
        raise ValueError(f"a wording has fewer than {attempts_per_wording} saved attempts")
    lines = []
    for draw in range(n_draws):
        first, last = draw * attempts_per_wording, (draw + 1) * attempts_per_wording
        counts = Counter()
        bonus_sizes, advantage_sizes, largest = [], [], 0.0
        for per_style in families.values():
            chosen = [record for attempts in per_style.values() for record in attempts[first:last]]
            ok = [record["harness_error"] is None for record in chosen]
            passed = [bool(record["passed"]) and good for record, good in zip(chosen, ok)]
            rewards = [record["reward"] if good else None for record, good in zip(chosen, ok)]
            group = family_group(rewards, passed, [record["style"] for record in chosen], bonus_points)
            counts["groups"] += 1
            counts["equal_rewards"] += int(not group["kept_arm1"])
            counts["kept"] += int(group["kept_arm1"])
            counts["keep_decision_differs"] += int(group["kept_arm1"] != group["kept_arm2"])
            counts["with_a_pass_and_a_fail"] += int(any(passed) and any(good and not won
                                                                         for good, won in zip(ok, passed)))
            sizes = [abs(group["bonus"][i]) for i in range(len(chosen)) if passed[i]]
            if not sizes or max(sizes) <= TOLERANCE:
                continue
            counts["bonus_not_zero"] += 1
            counts["passes_in_those_groups"] += len(sizes)
            bonus_sizes.extend(sizes)
            largest = max(largest, max(sizes))
            advantage_sizes.extend(abs(group["arm1"][i]) for i in range(len(chosen)) if ok[i])
            valid = np.array(ok)
            counts["order_changes"] += int(not _same_order(np.array(group["arm1"])[valid],
                                                           np.array(group["arm2"])[valid]))
        lines.append({
            "draw": draw + 1, "sample_positions": [first + 1, last],
            "groups": counts["groups"], "attempts_per_group": n_wordings * attempts_per_wording,
            "groups_with_equal_rewards": counts["equal_rewards"], "groups_kept": counts["kept"],
            "groups_where_the_arms_keep_differently": counts["keep_decision_differs"],
            "groups_with_a_pass_and_a_fail": counts["with_a_pass_and_a_fail"],
            "groups_where_the_bonus_is_not_zero": counts["bonus_not_zero"],
            "groups_where_the_order_of_the_attempts_changes": counts["order_changes"],
            "passes_in_those_groups": counts["passes_in_those_groups"],
            "mean_size_of_the_bonus_points": float(np.mean(bonus_sizes)) if bonus_sizes else None,
            "largest_bonus_points": largest if bonus_sizes else None,
            "mean_size_of_the_arm1_advantage_points": (float(np.mean(advantage_sizes))
                                                       if advantage_sizes else None),
        })
    for line in lines:
        line["families_left_out"] = left_out
    return lines


# ---------------------------------------------------------------------------
# Part 3: one split (dev, or test when it is there)
# ---------------------------------------------------------------------------
def split_numbers(runs: dict[str, dict], grpo_arms: list[str], *, n_draws: int, seed: int,
                  level: float) -> dict:
    """Every number of one split. `runs`: arm name -> the result of `load_run`, with BASE first, then SFT,
    then the GRPO arms.

    The table rows are those of the frozen `tables.compare_arms`, which also checks that the arms were run on
    the same scenarios, text, data and settings (ValueError if not)."""
    arms = list(runs)
    records = {arm: runs[arm]["records"] for arm in arms}
    pairs = [(BASE, SFT)] + [(SFT, arm) for arm in grpo_arms] + [(BASE, arm) for arm in grpo_arms]
    pairs += [(first, second) for i, first in enumerate(grpo_arms) for second in grpo_arms[i + 1:]]
    table = tables.compare_arms(records, BASE, pairs, n_draws=n_draws, seed=seed, level=level)
    rows = {row["arm"]: row for row in table["arms"]}
    n_families = table["n_families"]
    bootstrap = impl.get("paired_family_bootstrap")

    families = {arm: by_family(records[arm]) for arm in arms}
    values = {arm: family_values(records[arm]) for arm in arms}
    style_counts = {arm: {family_id: passes_by_style(group) for family_id, group in families[arm].items()}
                    for arm in arms}
    outcomes = {record["family_id"]: record.get("outcome") for record in records[BASE]}
    per_arm = {}
    for arm in arms:
        row, valid = rows[arm], metrics.valid_records(records[arm])
        mean_value = float(np.mean(list(values[arm].values())))
        if not _same(mean_value, row["all_wordings_pass"]):
            raise ValueError(f"arm {arm}: the mean of the family values ({mean_value}) is not the frozen "
                             f"all_wordings_pass ({row['all_wordings_pass']})")
        rates = metrics.family_pass_rates(records[arm])
        room = {family_id: (1.0 - value) / len(values[arm]) for family_id, value in values[arm].items()
                if value < 1.0}
        of_values = metrics.family_bootstrap_ci(list(values[arm].values()), n_draws, seed, level)
        passed = sum(1 for record in valid if record["passed"])
        per_arm[arm] = {
            "attempts": len(records[arm]), "valid_attempts": len(valid),
            "harness_errors": row["n_harness_errors"], "passed_attempts": passed,
            "failed_attempts": len(valid) - passed, "share_of_passed_attempts": passed / len(valid),
            "pass_rate": row["pass_rate"], "pass_rate_lo": row["pass_rate_lo"],
            "pass_rate_hi": row["pass_rate_hi"],
            "all_wordings_pass": row["all_wordings_pass"],
            "all_wordings_pass_lo": of_values["lo"], "all_wordings_pass_hi": of_values["hi"],
            "room": 1.0 - row["all_wordings_pass"],
            "families": len(rates),
            "families_with_a_failed_attempt": sum(1 for rate in rates.values() if rate < 1.0),
            "families_below_one_on_all_wordings": len(room),
            "room_by_family": dict(sorted(room.items(), key=lambda item: (-item[1], item[0]))),
            "families_whose_wordings_disagree": [family_id for family_id, counts in style_counts[arm].items()
                                                 if styles_disagree(counts)],
            "variance_shares": row["variance_shares"],
            "smallest_detectable_effect": row["smallest_detectable_effect"],
        }

    # Families with fewer passes than under sft (as a share of the family's valid attempts).
    def share(arm: str, family_id: str) -> Fraction:
        valid = metrics.valid_records(families[arm].get(family_id, []))
        return Fraction(sum(1 for record in valid if record["passed"]), len(valid)) if valid else Fraction(-1)

    worse = {arm: [family_id for family_id in families[SFT]
                   if Fraction(0) <= share(arm, family_id) < share(SFT, family_id)] for arm in grpo_arms}
    worse_in_every = [family_id for family_id in families[SFT]
                      if grpo_arms and all(family_id in worse[arm] for arm in grpo_arms)]
    without_them = {}
    if worse_in_every:
        for arm in arms:
            rest = [record for record in records[arm] if record["family_id"] not in worse_in_every]
            valid = metrics.valid_records(rest)
            if valid:
                without_them[arm] = {"variance_shares": metrics.variance_shares(rest),
                                     "failed_attempts": sum(1 for record in valid if not record["passed"]),
                                     "valid_attempts": len(valid)}

    shown = sorted(set(worse_in_every) | {family_id for arm in arms
                                          for family_id in per_arm[arm]["families_whose_wordings_disagree"]}
                   | {family_id for arm in grpo_arms for family_id in per_arm[arm]["room_by_family"]})
    family_rows = {}
    for family_id in shown:
        family_rows[family_id] = {"outcome": outcomes.get(family_id), "arms": {}}
        for arm in arms:
            group = metrics.valid_records(families[arm].get(family_id, []))
            failed = [record for record in group if not record["passed"]]
            closes = Counter((record.get("final_request") or {}).get("resolution") or "no close"
                             for record in failed)
            family_rows[family_id]["arms"][arm] = {
                "passes_by_style": {str(style): counts
                                    for style, counts in style_counts[arm].get(family_id, {}).items()},
                "passed_attempts": sum(1 for record in group if record["passed"]),
                "valid_attempts": len(group),
                "all_wordings_pass": values[arm].get(family_id),
                "wordings_disagree": family_id in per_arm[arm]["families_whose_wordings_disagree"],
                "closes_of_the_failed_attempts": dict(sorted(closes.items(), key=lambda item: (-item[1],
                                                                                              item[0]))),
                "person_of_the_close_in_failed_attempts": dict(Counter(
                    (record.get("final_request") or {}).get("person_id") or "no close" for record in failed)),
            }

    comparisons = []
    for row in table["pairs"]:
        first, second = row["a"], row["b"]
        common = [family_id for family_id in values[first] if family_id in values[second]]
        a = np.array([values[first][family_id] for family_id in common])
        b = np.array([values[second][family_id] for family_id in common])
        on_values = bootstrap(a, b, n_draws, seed, level)
        comparisons.append({
            "a": first, "b": second, "families": row["n_families"],
            "in_the_plan": not (first in grpo_arms and second in grpo_arms),
            "pass_rate": {"diff": row["diff"], "lo": row["lo"], "hi": row["hi"],
                          "width": row["hi"] - row["lo"], "t_lo": row["t_lo"], "t_hi": row["t_hi"]},
            "all_wordings_pass": {"diff": float(on_values["diff"]), "lo": float(on_values["lo"]),
                                  "hi": float(on_values["hi"]),
                                  "width": float(on_values["hi"] - on_values["lo"]),
                                  "families_that_differ": int(np.sum(np.abs(a - b) > TOLERANCE))},
        })

    attempts_per_scenario = sorted(set(tables.scenario_sample_counts(BASE, records[BASE]).values()))
    return {
        "split": table["splits"][0] if len(table["splits"]) == 1 else table["splits"],
        "families": n_families, "scenarios": table["n_scenarios"],
        "attempts_per_arm": table["n_episodes_per_arm"], "attempts_per_scenario": attempts_per_scenario,
        "styles": sorted({record["style"] for record in records[BASE]}),
        "points_of_one_family": 1.0 / n_families,
        "interval": {"level": level, "n_draws": n_draws, "seed": seed,
                     "paired_family_bootstrap": table["paired_family_bootstrap"]},
        "notes_of_the_frozen_table": table["notes"],
        "arms": per_arm,
        "disagree_rule": {"high": [HIGH.numerator, HIGH.denominator],
                          "low": [LOW.numerator, LOW.denominator]},
        "families_with_fewer_passes_than_sft": worse,
        "families_with_fewer_passes_than_sft_in_every_grpo_arm": worse_in_every,
        "without_those_families": without_them,
        "family_rows": family_rows,
        "comparisons": comparisons,
        "_table": table,
    }


def add_training_styles(split: dict, runs: dict[str, dict], train_styles: list[int]) -> None:
    """Add, per arm, the frozen `all_wordings_pass` on the attempts of the training styles only (the styles
    that some training scenario uses), and the passes of each held-out style in the families with room."""
    split["training_styles"] = list(train_styles)
    for arm, run in runs.items():
        kept = [record for record in run["records"] if record["style"] in train_styles]
        split["arms"][arm]["all_wordings_pass_training_styles"] = (
            float(metrics.all_wordings_pass(kept)) if metrics.valid_records(kept) else None)


def check_against_saved_tables(folder: Path, split: dict) -> list[dict]:
    """Compare this script's numbers with the saved output of the frozen table program, file by file.

    Every `*.json` of `folder` that is a table of the same split is read. For each arm of the same name:
    pass rate, all-wordings-pass, the three variance shares and the counts. For each comparison of the same
    two arms, if the table was made with the same bootstrap settings: the difference and the interval.
    Raises ValueError at the first file with another value. Returns one line per file that was compared."""
    if not folder.is_dir():
        return []
    ours = split["_table"]
    our_arms = {row["arm"]: row for row in ours["arms"]}
    our_pairs = {(row["a"], row["b"]): row for row in ours["pairs"]}
    checked = []
    for path in sorted(folder.glob("*.json")):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(saved, dict) or "arms" not in saved or saved.get("splits") != ours["splits"]:
            continue
        different, n_values = [], 0
        same_bootstrap = all(saved.get(key) == ours[key] for key in ("level", "n_draws", "seed"))
        try:
            for row in saved["arms"]:
                mine = our_arms.get(row.get("arm"))
                if mine is None:
                    continue
                for key in ("n_families", "n_episodes", "n_harness_errors", "pass_rate", "all_wordings_pass"):
                    n_values += 1
                    if not _same(row[key], mine[key]):
                        different.append(f"{row['arm']} {key}: {row[key]} against {mine[key]}")
                for key, value in row["variance_shares"].items():
                    n_values += 1
                    if not _same(value, mine["variance_shares"][key]):
                        different.append(f"{row['arm']} variance_shares {key}: {value} against "
                                         f"{mine['variance_shares'][key]}")
            for row in saved.get("pairs", []) if same_bootstrap else []:
                mine = our_pairs.get((row.get("a"), row.get("b")))
                if mine is None:
                    continue
                for key in ("diff", "lo", "hi"):
                    n_values += 1
                    if not _same(row[key], mine[key]):
                        different.append(f"{row['b']} minus {row['a']} {key}: {row[key]} against {mine[key]}")
        except (AttributeError, KeyError, TypeError):
            continue                     # a JSON file of another kind: not a table of the frozen program
        if different:
            raise ValueError(f"{shown_path(path)} gives other values than this script: "
                             + "; ".join(different[:5]))
        if n_values:
            checked.append({"file": shown_path(path), "sha256": sha256_of_file(path),
                            "values_compared": n_values, "intervals_compared": same_bootstrap,
                            "agrees": True})
    return checked


def family_facts(scenarios: Path, split: str, family_ids: list[str], runs: dict[str, dict]) -> dict | None:
    """What the scenario file says about the named DEV families: the expected close, and the people who
    have the name of the request. None if the file is not there or no family is named.

    Only `dev.jsonl` is ever opened here. Raises ValueError for another split, or if the file is not the
    one that the dev runs name (`data_sha256` of their records)."""
    if split != DEV:
        raise ValueError("the scenario file is read for the dev split only")
    path = scenarios / "dev.jsonl"
    if not family_ids or not path.is_file():
        return None
    file_hash = sha256_of_file(path)
    named = {record["data_sha256"] for run in runs.values() for record in run["records"]
             if record.get("data_sha256")}
    if named and named != {file_hash}:
        raise ValueError(f"{shown_path(path)} is not the scenario file of the dev runs: their records name "
                         "another SHA-256")
    ruleset = next(iter(runs.values()))["records"][0].get("ruleset", "default")
    facts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            family = json.loads(line)
            if family["family_id"] not in family_ids:
                continue
            truth, request = family["truth"], family["request"]
            same_name = [person for person in family["world"]["people"]
                         if person["name"] == request["person_name"]]
            facts[family["family_id"]] = {
                "outcome": family["outcome"],
                "expected_close": family["expected"][ruleset]["request"]["resolution"],
                "person_of_the_request": truth["person_id"],
                "people_with_the_name_of_the_request": len(same_name),
                "status_of_the_other_people_with_that_name": sorted(
                    person["status"] for person in same_name if person["person_id"] != truth["person_id"]),
            }
    return {"file": shown_path(path), "sha256": file_hash,
            "checked_against_the_records": bool(named), "families": facts}


# ---------------------------------------------------------------------------
# Part 5: what the family bootstrap can show
# ---------------------------------------------------------------------------
def families_that_must_gain(n_families: int, n_room: int, *, n_draws: int, seed: int, level: float) -> dict:
    """How many families must gain before the paired family bootstrap can give an interval above zero.

    One draw of the bootstrap takes `n_families` families with replacement. If a gain sits in k families and
    every other family is the same in both arms, a draw that holds none of the k has a difference of exactly
    zero. The share of such draws is (1 - k / n_families) ** n_families. The lower end of the interval is the
    (1 - level) / 2 quantile of the draws, so it is zero while that share is above (1 - level) / 2.
    Each line is also checked with the frozen bootstrap function itself, on made-up family values (a gain of
    1 in the first k families, 0 elsewhere)."""
    tail = (1.0 - level) / 2.0
    bootstrap = impl.get("paired_family_bootstrap")
    lines, needed = [], None
    for k in range(1, max(n_room, 1) + 4):
        if k > n_families:
            break
        share = (1.0 - k / n_families) ** n_families
        gains = np.zeros(n_families)
        gains[:k] = 1.0
        low = bootstrap(np.zeros(n_families), gains, n_draws, seed, level)["lo"]
        lines.append({"families_that_gain": k, "share_of_draws_without_them": share,
                      "lower_end_above_zero_by_the_formula": share < tail,
                      "lower_end_above_zero_by_the_frozen_bootstrap": bool(low > 0.0)})
        if needed is None and share < tail:
            needed = k
    return {"families": n_families, "level": level, "tail": tail, "lines": lines,
            "families_needed": needed, "families_with_room": n_room}


def rank_rule(seeds_per_arm: int) -> dict:
    """The chance, with no effect of the bonus, that every run of arm 2 is above every run of arm 1.

    With no effect the 2 * m runs are in a random order (no ties), and one of the C(2m, m) ways to pick the
    m best runs is "all of arm 2"."""
    return {str(m): {"runs": 2 * m, "ways": math.comb(2 * m, m), "chance": 1.0 / math.comb(2 * m, m)}
            for m in sorted({2, seeds_per_arm})}


# ---------------------------------------------------------------------------
# Part 6: cost
# ---------------------------------------------------------------------------
def plan_lists_the_experiment(path: Path) -> dict | None:
    """Section 3 of the frozen analysis plan (the arms and the planned extras): does it name this experiment?
    Rule: the section runs from the line "## 3." to the next line that starts with "## ". It names the
    experiment if it holds one of PLAN_WORDS (case ignored). Returns None if the file or the section is not
    there."""
    if not path.is_file():
        return None
    section, inside = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line.startswith("## 3.")
            continue
        if inside:
            section.append(line)
    if not section:
        return None
    text = "\n".join(section)
    found = [word for word in PLAN_WORDS if word in text.lower()]
    return {"file": shown_path(path), "sha256": sha256_of_file(path), "section": 3,
            "has_planned_extras": "Planned extras" in text, "words_found": found,
            "lists_the_experiment": bool(found)}


def parse_utc(text: str):
    """A time of the form 2026-11-19T06:08:48Z as an aware datetime. Raises ValueError otherwise."""
    from datetime import datetime, timezone
    if not isinstance(text, str) or not text.endswith("Z"):
        raise ValueError(f"{text!r} is not a time such as 2026-11-19T06:08:48Z")
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def ledger_prices(path: Path, until_utc: str | None) -> dict | None:
    """The lowest and the highest hourly price (USD per hour) of the machines rented up to the end of the last
    saved GRPO training run: the "rent" lines of the spend ledger with a time at or before `until_utc` (the
    latest `ended_utc` of the run manifests of the saved GRPO runs). These are the rentals of the work that
    made the saved runs that the design copies. A machine rented later (an open rental, or one of other work)
    does not move the prices. None without the file, without `until_utc`, or without such a line.
    Raises ValueError if the time of a "rent" line cannot be read."""
    if not path.is_file() or not until_utc:
        return None
    until = parse_utc(until_utc)
    chosen = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("event") != "rent" or not row.get("dph_usd"):
                continue
            try:
                when = parse_utc(row.get("time_utc"))
            except ValueError:
                raise ValueError(f"{shown_path(path)}, line {line}: the time {row.get('time_utc')!r} cannot be "
                                 "read") from None
            if when <= until:
                chosen.append((row.get("instance_id"), float(row["dph_usd"])))
    if not chosen:
        return None
    prices = [price for _, price in chosen]
    return {"file": shown_path(path),
            "rule": "the rent lines with a time at or before the end of the last saved GRPO training run",
            "until_utc": until_utc, "rentals": [instance for instance, _ in chosen],
            "lowest_usd_per_hour": min(prices), "highest_usd_per_hour": max(prices)}


def cost_numbers(training: dict[str, dict], dev_runs: dict[str, dict], grpo_arms: list[str],
                 seeds_per_arm: int, prices: dict | None) -> dict:
    """GPU hours of the saved runs and of the design. One GPU ran each saved run, so its GPU hours are the
    wall-clock seconds of its run manifest divided by 3600. The design is priced as if one of its runs took
    as long as a saved GRPO run plus the dev run of its last checkpoint (mean over the saved GRPO arms)."""
    per_arm = {}
    for arm in grpo_arms:
        train_seconds, dev_seconds = training[arm]["seconds"], dev_runs[arm]["info"]["seconds"]
        if train_seconds is None or dev_seconds is None:
            raise ValueError(f"arm {arm}: a run manifest has no `seconds`")
        per_arm[arm] = {"training_hours": train_seconds / 3600.0, "dev_run_hours": dev_seconds / 3600.0,
                        "training_seconds": train_seconds, "dev_run_seconds": dev_seconds,
                        "gpu": training[arm]["gpu"]}
    hours_per_run = float(np.mean([row["training_hours"] + row["dev_run_hours"] for row in per_arm.values()]))
    designs = {}
    for m in sorted({2, seeds_per_arm}):
        runs = N_ARMS * m
        hours = runs * hours_per_run
        designs[str(m)] = {"seeds_per_arm": m, "runs": runs, "gpu_hours": hours,
                           "usd_low": None if prices is None else hours * prices["lowest_usd_per_hour"],
                           "usd_high": None if prices is None else hours * prices["highest_usd_per_hour"]}
    return {"saved": per_arm, "hours_of_one_design_run": hours_per_run, "designs": designs, "prices": prices,
            "gpus": sorted({str(row["gpu"]) for row in per_arm.values()})}


# ---------------------------------------------------------------------------
# All numbers
# ---------------------------------------------------------------------------
def folder_of(main: Path, overrides: dict[str, Path], key: str) -> Path:
    return overrides.get(key, main / key)


def build_numbers(main: Path, grpo_arms: list[str], overrides: dict[str, Path], *, ledger: Path,
                  scenarios: Path, plan: Path | None = None, n_draws: int, seed: int, level: float, bonus_points: float,
                  attempts_per_wording: int, families_per_step: int, seeds_per_arm: int) -> dict:
    """Everything that the three files hold, as one dict. Raises ValueError for everything that the module
    text names. No file is written here."""
    if not grpo_arms or len(set(grpo_arms)) != len(grpo_arms) or {BASE, SFT} & set(grpo_arms):
        raise ValueError(f"give each GRPO arm once, under a name other than {BASE} and {SFT}: {grpo_arms}")
    if bonus_points <= 0 or attempts_per_wording < 1 or families_per_step < 1 or seeds_per_arm < 1:
        raise ValueError("the numbers of the design must be above zero")
    arms = [BASE, SFT] + grpo_arms

    dev_runs = {arm: load_run(folder_of(main, overrides, f"{arm}_dev"), DEV) for arm in arms}
    dev = split_numbers(dev_runs, grpo_arms, n_draws=n_draws, seed=seed, level=level)
    saved_tables = check_against_saved_tables(folder_of(main, overrides, "tables"), dev)

    training = {arm: saved_groups_with_bonus(folder_of(main, overrides, arm), bonus_points)
                for arm in grpo_arms}
    layouts = sorted({(run["scenarios_per_step"], run["attempts_per_group"], run["steps"],
                       run["learning_rate"], tuple(run["styles"] or ())) for run in training.values()})
    if len(layouts) != 1:
        raise ValueError(f"the GRPO runs differ in their layout (scenarios per step, attempts per group, "
                         f"steps, learning rate, styles): {layouts}")
    saved_layout = next(iter(training.values()))
    train_styles = list(saved_layout["styles"] or [])
    if not train_styles:
        train_styles = [style for style in dev["styles"] if style != metrics.HELD_OUT_STYLE]
    dev["styles_in_no_training_scenario"] = [style for style in dev["styles"] if style not in train_styles]
    add_training_styles(dev, dev_runs, train_styles)

    sft_train = load_run(folder_of(main, overrides, "sft_train"), TRAIN)
    on_saved = design_on_saved_attempts(sft_train["records"], attempts_per_wording, len(train_styles),
                                        bonus_points)
    worse = dev["families_with_fewer_passes_than_sft_in_every_grpo_arm"]
    facts = family_facts(scenarios, DEV, worse, dev_runs)

    n_room = max(dev["arms"][arm]["families_below_one_on_all_wordings"] for arm in grpo_arms)
    spread = {}
    if len(grpo_arms) > 1:
        for key, read in (("pass_rate", lambda row: row["pass_rate"]),
                          ("all_wordings_pass", lambda row: row["all_wordings_pass"]),
                          ("variance_share_of_wording", lambda row: row["variance_shares"]["wording"])):
            found = [read(dev["arms"][arm]) for arm in grpo_arms]
            spread[key] = max(found) - min(found)

    test_folders = {arm: folder_of(main, overrides, f"{arm}_test") for arm in arms}
    missing = [shown_path(folder) for folder in test_folders.values()
               if not (folder / "episodes.jsonl").is_file() or not (folder / "run_manifest.json").is_file()
               or read_json(folder / "run_manifest.json").get("complete") is not True]
    test, test_tables, test_inputs = None, [], None
    if not missing:
        test_runs = {arm: load_run(test_folders[arm], TEST) for arm in arms}
        test = split_numbers(test_runs, grpo_arms, n_draws=n_draws, seed=seed, level=level)
        test["styles_in_no_training_scenario"] = [style for style in test["styles"]
                                                  if style not in train_styles]
        add_training_styles(test, test_runs, train_styles)
        test_tables = check_against_saved_tables(folder_of(main, overrides, "tables"), test)
        test_inputs = {arm: run["info"] for arm, run in test_runs.items()}
        del test["_table"]
    del dev["_table"]

    ends = [run["ended_utc"] for run in training.values()]
    prices = ledger_prices(ledger, max(ends, key=parse_utc) if ends and all(ends) else None)
    design = {
        "numbers": [
            {"part": "arms", "value": N_ARMS, "origin": "the brief: with the bonus and without it"},
            {"part": "wordings in a family group", "value": len(train_styles),
             "origin": "the saved runs: the wording styles that training uses"},
            {"part": "attempts per wording", "value": attempts_per_wording, "origin": "judgement"},
            {"part": "attempts per family group", "value": len(train_styles) * attempts_per_wording,
             "origin": "own calculation"},
            {"part": "family groups per step", "value": families_per_step, "origin": "judgement"},
            {"part": "attempts per step",
             "value": families_per_step * len(train_styles) * attempts_per_wording,
             "origin": "own calculation"},
            {"part": "steps", "value": saved_layout["steps"], "origin": "the saved runs"},
            {"part": "learning rate", "value": saved_layout["learning_rate"], "origin": "the saved runs"},
            {"part": "largest bonus of a pass, in points", "value": bonus_points, "origin": "judgement"},
            {"part": "training runs (seeds) per arm", "value": seeds_per_arm, "origin": "judgement"},
        ],
        "parts": [
            {"part": "start of both arms",
             "value": "the adapter that the saved GRPO runs started from"
                      + (f" ({saved_layout['start_adapter']})" if saved_layout["start_adapter"] else ""),
             "origin": "the saved runs"},
            {"part": "group", "value": "the attempts of one family in all its training wordings",
             "origin": "judgement"},
            {"part": "advantage, both arms",
             "value": "the reward minus the mean reward of the family group; a group with equal rewards is "
                      "dropped, as in the saved runs", "origin": "judgement"},
            {"part": "reward of arm 1", "value": "the reward of the saved runs: pass or fail, with small "
                                                 "penalties for malformed calls and extra turns",
             "origin": "the saved runs"},
            {"part": "other settings", "value": "as in grpo_config of the run manifest of the saved GRPO runs",
             "origin": "the saved runs"},
            {"part": "bonus when another wording has no valid attempt",
             "value": "a pass with no valid attempt in another wording of its group gets no bonus",
             "origin": "judgement"},
            {"part": "reward of arm 2", "value": "the reward of arm 1, and the bonus below for a passing "
                                                 "attempt", "origin": "judgement"},
            {"part": "measures, on dev", "value": "all wordings pass, and the variance due to wording; the "
                                                  "pass rate must not fall", "origin": "judgement"},
            {"part": "rule for the result", "value": "the bonus helps if every run of arm 2 is above every "
                                                     "run of arm 1 on all wordings pass",
             "origin": "judgement"},
        ],
        "bonus_points": bonus_points, "attempts_per_wording": attempts_per_wording,
        "wordings": len(train_styles), "families_per_step": families_per_step,
        "attempts_per_group": len(train_styles) * attempts_per_wording,
        "attempts_per_step": families_per_step * len(train_styles) * attempts_per_wording,
        "seeds_per_arm": seeds_per_arm, "arms": N_ARMS,
        "example": family_example(bonus_points, attempts_per_wording),
        "on_saved_attempts": on_saved,
        "rank_rule": rank_rule(seeds_per_arm),
    }
    return {
        "what": "The stretch item of the brief: numbers from saved runs, and the arithmetic of a design.",
        "experiment_was_run": False,
        "made_by": "scripts/stretch_numbers.py",
        "code": {"versions_of_the_core_functions": impl.mode(),
                 "files_sha256": {name: sha256_of_file(ROOT / name) for name in CODE_FILES}},
        "arms": arms, "grpo_arms": grpo_arms,
        "inputs": {"dev": {arm: run["info"] for arm, run in dev_runs.items()},
                   "sft_train": sft_train["info"], "saved_tables": saved_tables + test_tables,
                   "scenario_file": None if facts is None else {key: facts[key] for key in
                                                                ("file", "sha256",
                                                                 "checked_against_the_records")},
                   "test": test_inputs, "test_runs_not_there": missing},
        "one_wording_group": {
            "example": one_wording_example(EXAMPLE_PASSES, EXAMPLE_FAILS, bonus_points),
            "saved_runs": training,
            "saved_code_refuses_a_group_of_two_scenarios": refuses_a_group_of_two_scenarios(),
        },
        "dev": dev,
        "family_facts": None if facts is None else facts["families"],
        "test": test,
        "design": design,
        "effect": {
            "families_that_must_gain": families_that_must_gain(dev["families"], n_room, n_draws=n_draws,
                                                               seed=seed, level=level),
            "spread_between_the_grpo_arms": spread,
            "largest_room_after_grpo": max(dev["arms"][arm]["room"] for arm in grpo_arms),
            "little_room_is_below": LITTLE_ROOM,
            "room_is_little": max(dev["arms"][arm]["room"] for arm in grpo_arms) < LITTLE_ROOM,
        },
        "cost": cost_numbers(training, dev_runs, grpo_arms, seeds_per_arm, prices),
        "analysis_plan": plan_lists_the_experiment(ROOT / "ANALYSIS_PLAN.md" if plan is None else plan),
    }


# ---------------------------------------------------------------------------
# Words and number formats
# ---------------------------------------------------------------------------
def pct(value: float) -> str:
    """0.3978 -> "39.8%"."""
    return f"{100.0 * value:.1f}%"


def pts(value: float) -> str:
    """A share as points with one decimal: 0.0355 -> "3.6"."""
    return f"{100.0 * value:.1f}"


def signed(value: float, digits: int = 1) -> str:
    """12.5 -> "+12.5"; -3 -> "-3.0"; a value that rounds to zero -> "0.0"."""
    text = f"{value:+.{digits}f}"
    return text[1:] if float(text) == 0.0 else text


def plain(value: float, digits: int = 1) -> str:
    """A number with at most `digits` decimals and no zeros at the end: 20.0 -> "20", 5.625 -> "5.6"."""
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def signed_short(value: float, digits: int = 1) -> str:
    """As `signed`, without zeros at the end: 25 -> "+25"; 48.33 -> "+48.3"; -90 -> "-90"."""
    text = signed(value, digits)
    return text.rstrip("0").rstrip(".") if "." in text else text


def listed(items: list) -> str:
    """["a", "b", "c"] -> "a, b and c"."""
    items = [str(item) for item in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def families_word(n: int) -> str:
    """1 -> "1 family"; 5 -> "5 families"."""
    return f"{n} family" if n == 1 else f"{n} families"


def count_word(n: int) -> str:
    """2 -> "two", 3 -> "three"; any other number as digits."""
    return {1: "one", 2: "two", 3: "three", 4: "four"}.get(n, str(n))


def same_or_listed(texts: list[str], arms: list[str], *, each: str = "in each GRPO arm",
                   short: bool = False) -> str:
    """One text per GRPO arm, as one phrase. If the arms all have the same text it is said once:
    "3.6 points in each GRPO arm" (`short`: only "3.6"). If not: "3.6 (grpo_a) and 4.1 (grpo_b)"."""
    if len(set(texts)) == 1:
        return texts[0] if short or len(texts) == 1 else f"{texts[0]} {each}"
    return listed([f"{text} ({arm})" for text, arm in zip(texts, arms)])


def interval(row: dict) -> str:
    """"+16.2 (+8.1 to +25.4)", in points."""
    return f"{signed(100 * row['diff'])} ({signed(100 * row['lo'])} to {signed(100 * row['hi'])})"


def style_cells(row: dict, styles: list[int]) -> list[str]:
    """Passes of valid attempts for each style, as "7 of 8"."""
    cells = []
    for style in styles:
        counts = row["passes_by_style"].get(str(style))
        cells.append("no attempt" if counts is None else f"{counts[0]} of {counts[1]}")
    return cells


def style_heads(split: dict, *, short: bool = False) -> list[str]:
    """The column heads of the wording styles. A style that training never uses says so."""
    note = " (held out)" if short else " (in no training scenario)"
    return [f"Style {style}" + (note if style in split["styles_in_no_training_scenario"] else "")
            for style in split["styles"]]


def md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def arm_table(split: dict, arms: list[str], *, compact: bool = False) -> list[str]:
    """The table of one split, one row per arm. The table file and the document both use it; `compact` (for
    the document) leaves the column Room out, because the text below the table gives the room."""
    held_out = bool(split.get("styles_in_no_training_scenario"))
    trained = split.get("training_styles") or []
    head = ["Arm", "Pass rate (passed attempts)", "All wordings pass"]
    if held_out:
        head += [f"All wordings pass, styles {listed(trained)} only"]
    head += ["Room, points", "Families with a failed attempt", "Families whose wordings disagree",
             "Variance due to wording"]
    rows = []
    for arm in arms:
        row = split["arms"][arm]
        cells = [arm, f"{pct(row['pass_rate'])} ({row['passed_attempts']} of {row['valid_attempts']})",
                 pct(row["all_wordings_pass"])]
        if held_out:
            value = row.get("all_wordings_pass_training_styles")
            cells += ["no attempt" if value is None else pct(value)]
        cells += [pts(row["room"]), f"{row['families_with_a_failed_attempt']} of {row['families']}",
                  str(len(row["families_whose_wordings_disagree"])), pct(row["variance_shares"]["wording"])]
        rows.append(cells)
    if compact:
        drop = {head.index(name) for name in ("Room, points", "Families with a failed attempt")}
        head = [cell for i, cell in enumerate(head) if i not in drop]
        rows = [[cell for i, cell in enumerate(row) if i not in drop] for row in rows]
    return md_table(head, rows)


def family_table(split: dict, family_id: str, arms: list[str], *, compact: bool = False) -> list[str]:
    """Passes per wording style of one family, one row per arm. `compact` (for the document): if every
    style of every arm has the same number of valid attempts, a cell holds the passes only and the head says
    "passes of 8"; the column of all styles together is left out."""
    family = split["family_rows"][family_id]["arms"]
    sizes = {counts[1] for arm in arms for counts in family[arm]["passes_by_style"].values()}
    if compact and len(sizes) == 1:
        rows = [[arm] + [str((family[arm]["passes_by_style"].get(str(style)) or ["no attempt"])[0])
                         for style in split["styles"]] for arm in arms]
        head = [f"{family_id}: passes of {next(iter(sizes))}"] + style_heads(split, short=True)
        return md_table(head, rows)
    rows = [[arm] + style_cells(family[arm], split["styles"])
            + ([] if compact else [f"{family[arm]['passed_attempts']} of {family[arm]['valid_attempts']}"])
            for arm in arms]
    return md_table([family_id] + style_heads(split, short=compact) + ([] if compact else ["All styles"]),
                    rows)


def low_and_high(row: dict, styles: list[int]) -> tuple[list[int], list[int]]:
    """The styles of one family in one arm that are at or below LOW, and at or above HIGH."""
    low, high = [], []
    for style in styles:
        counts = row["passes_by_style"].get(str(style))
        if counts is None or not counts[1]:
            continue
        share = Fraction(counts[0], counts[1])
        if share <= LOW:
            low.append(style)
        if share >= HIGH:
            high.append(style)
    return low, high


def rule_words(split: dict) -> str:
    """The rule for "the wordings disagree", in the counts of this split when every scenario has the same
    number of attempts."""
    counts = split["attempts_per_scenario"]
    if len(counts) == 1 and (HIGH * counts[0]).denominator == 1 and (LOW * counts[0]).denominator == 1:
        return (f"one style passes {int(HIGH * counts[0])} or more of its {counts[0]} attempts and another "
                f"style {int(LOW * counts[0])} or fewer")
    return (f"one style passes at least {HIGH.numerator} of {HIGH.denominator} of its attempts and another "
            f"style at most {LOW.numerator} of {LOW.denominator}")


# ---------------------------------------------------------------------------
# The table file
# ---------------------------------------------------------------------------
def render_tables(numbers: dict, doc_shown: str) -> str:
    """The table file: every number as a table, with its rule, its unit and its source."""
    arms, grpo_arms, dev = numbers["arms"], numbers["grpo_arms"], numbers["dev"]
    one, design, cost = numbers["one_wording_group"], numbers["design"], numbers["cost"]
    example, runs = one["example"], one["saved_runs"]
    bonus = plain(design["bonus_points"])
    inputs = numbers["inputs"]
    dev_files = ", ".join(f"`{inputs['dev'][arm]['episodes']}`" for arm in arms)
    lines = [
        "# Numbers for the stretch item",
        "",
        "**The stretch experiment was not run.** These tables hold what the saved runs show, and the "
        f"arithmetic of a design. `scripts/stretch_numbers.py` made this file; `{doc_shown}` quotes it. "
        "No model was run to make it.",
        "",
        "Words. A family is one world with one correct end state. A scenario is one family in one wording "
        "style. An attempt is one conversation. A group is the attempts of one scenario in one training "
        "step. The advantage of an attempt is its reward minus the mean reward of its group. A point is a "
        "hundredth of a pass: a pass with no penalty has a reward of 100 points, a fail has 0.",
        "",
        "## T1. A bonus inside a group of one wording (a made-up example)",
        "",
        f"Rule: `group_advantages` (the function of the training, `accessdesk/train/grpo_math.py`) on one "
        f"group of {example['attempts']} attempts with {example['passes']} passes and {example['fails']} "
        f"fails, as it is and with {bonus} points added to every pass. Unit: points.",
        "",
    ]
    lines += md_table(
        ["", "Reward of a pass", "Reward of a fail", "Mean of the group", "Advantage of a pass",
         "Advantage of a fail"],
        [[label, plain(row["reward_of_a_pass"]), plain(row["reward_of_a_fail"]),
          plain(row["mean_of_the_group"]), signed_short(row["advantage_of_a_pass"]),
          signed_short(row["advantage_of_a_fail"])]
         for label, row in (("Without the bonus", example["without"]),
                            (f"With {bonus} points on every pass", example["with"]))])
    lines += [
        "",
        f"Each advantage with the bonus is {plain(example['factor'], 4)} times the advantage without it "
        f"(the formula gives (100 + {bonus}) / 100 = {plain(example['factor_by_formula'], 4)}).",
        "",
        "Why every pass of the group gets the same bonus: the bonus looks at the attempts on the family's "
        "other wordings, and these are the same for every attempt of this group. A bonus that paired each "
        "attempt with one single attempt of another wording would differ between the passes, but only by "
        "chance: the pairing is arbitrary, so its expected value is the same for every pass of the group. "
        "It would add noise to the step, not a preference between attempts.",
        "",
        "## T2. The same bonus on the saved training groups",
        "",
        f"Rule: for every step of a saved GRPO run, the saved attempts give the reward matrix of the "
        f"training (`reward_matrix`), and `group_advantages` is run on it twice: as it is, and with {bonus} "
        "points added to the reward of every passing attempt. Nothing is trained. Unit: the group. Source: "
        "`rollouts/step_*.episodes.jsonl`, `train_log.jsonl` and `run_manifest.json` of each run ("
        + ", ".join(f"`{runs[arm]['folder']}`" for arm in grpo_arms) + ").",
        "",
    ]
    first = runs[grpo_arms[0]]
    labels = [
        ("groups", f"All groups ({first['steps']} steps x {first['scenarios_per_step']} scenarios; "
                   f"{first['attempts_per_group']} attempts each)"),
        ("kept", "Kept by the rule of the training (the rewards are not all equal)"),
        ("kept_with_bonus", "Kept with the bonus"),
        ("keep_decision_changes", "Groups that the bonus moves between kept and dropped"),
        ("kept_order_changes", "Kept groups in which the bonus changes the order of the attempts"),
        ("kept_mixed_full_reward", f"Kept, with passes and fails, every pass at {first['pass_reward']} "
                                   f"points: the bonus multiplies every advantage by "
                                   f"{plain(first['factor'], 4)}"),
        ("kept_one_outcome", "Kept, every attempt passes (the rewards differ by penalties): the bonus "
                             "changes no advantage"),
        ("kept_mixed_other", f"Kept, with passes and fails, a pass below {first['pass_reward']} points: "
                             "the bonus moves all passes by one amount and all fails by another"),
        ("with_a_pass_and_a_fail", "Groups with a pass and a fail"),
        ("with_another_wording_of_the_family_in_the_step", "Groups with another wording of their family "
                                                           "in the same step"),
        ("harness_errors", "Attempts that the harness lost"),
        ("passes", "Passing attempts (not groups)"),
        ("passes_below_full_reward", f"Passing attempts below {first['pass_reward']} points (penalties)"),
        ("fails_not_at_zero", "Failing attempts with a reward other than 0"),
    ]
    lines += md_table(["Count of groups"] + grpo_arms,
                      [[label] + [str(runs[arm][key]) for arm in grpo_arms] for key, label in labels]
                      + [["Lowest reward of a pass, points"]
                         + ["none" if runs[arm]["lowest_pass_reward"] is None
                            else plain(runs[arm]["lowest_pass_reward"]) for arm in grpo_arms]])
    lines += [
        "",
        "The number of kept groups is also the one of the training log (the script stops if they differ). "
        f"The training drops a group only if its rewards are all equal (tie_tol {first['tie_tol']:g}). A bonus "
        "on every pass keeps equal rewards equal and unequal rewards unequal when every fail is below every "
        "pass, so the same groups are kept. With every pass at the full reward the advantages are multiplied; "
        "in general every pass gains b (1 - k/n) and every fail loses b k/n (k passes of n attempts). "
        "The saved training code builds every group from one scenario: its reward matrix "
        + ("refuses" if one["saved_code_refuses_a_group_of_two_scenarios"] else "does NOT refuse")
        + " a group that holds two scenarios (`reward_matrix` in `accessdesk/train/grpo.py`).",
    ]
    lines += split_tables(numbers, dev, "dev", 3, dev_files)
    lines += [
        "",
        "## T8. The design (not run)",
        "",
        "Every value is a choice. The column Origin says where it comes from: the brief, the saved runs, "
        "our own calculation from other values, or a judgement (no run and no source chose it).",
        "",
    ]
    lines += md_table(["Part", "Value", "Origin"],
                      [[row["part"], plain(row["value"], 6) if isinstance(row["value"], float)
                        else str(row["value"]), row["origin"]]
                       for row in design["numbers"] + design["parts"]])
    wordings = design["example"]["wordings"]
    lines += [
        "",
        f"The bonus of arm 2: a passing attempt gets {bonus} x (q - q_mean) points. q is the share of "
        "passes among the family's attempts in the other wordings. q_mean is the mean of q over the "
        "family's passing attempts. So the bonuses of a group add up to zero, the mean reward of the group "
        "stays, and so does the advantage of every fail.",
        "",
        f"A made-up family group ({design['example']['attempts']} attempts). Unit: points. Source: "
        "`family_group` of the script, which calls `group_advantages`.",
        "",
    ]
    lines += md_table(
        ["Wording", "Passes", "q of a pass", "Bonus of a pass", "Arm 1: advantage of a pass",
         "Arm 2: advantage of a pass", "Advantage of a fail, both arms"],
        [[name, f"{row['passes']} of {row['attempts']}",
          plain(row["share_of_passes_in_the_other_wordings"], 3) if "bonus_of_a_pass" in row else "no pass",
          signed(row["bonus_of_a_pass"], 2) if "bonus_of_a_pass" in row else "no pass",
          signed(row["arm1_advantage_of_a_pass"], 2) if "bonus_of_a_pass" in row else "no pass",
          signed(row["arm2_advantage_of_a_pass"], 2) if "bonus_of_a_pass" in row else "no pass",
          signed(row["arm1_advantage_of_a_fail"], 2) if "arm1_advantage_of_a_fail" in row else "no fail"]
         for name, row in wordings.items()])
    lines += [
        "",
        f"q_mean is {plain(design['example']['mean_share'], 4)}; the bonuses of the group add up to "
        f"{plain(abs(design['example']['sum_of_the_bonuses']), 6)}.",
        "",
        "The rule for the result (a judgement): the bonus helps if every run of arm 2 is above every run of "
        "arm 1. With no effect of the bonus, and no ties, this happens by chance in 1 of C(2m, m) cases for "
        "m runs per arm (own calculation): "
        + "; ".join(f"m = {m}: 1 of {row['ways']} ({pct(row['chance'])})"
                    for m, row in design["rank_rule"].items()) + ".",
        "",
        "## T9. The bonus of the design on saved attempts",
        "",
        f"Rule: the saved attempts of the SFT model on training families are cut into family groups of the "
        f"design ({design['wordings']} wordings x {design['attempts_per_wording']} attempts; draw 1 takes "
        "the lowest sample numbers of each wording, draw 2 the next ones), and the arithmetic of both arms "
        "is run on each group. Nothing is trained: this shows the size of the bonus at the start of both "
        f"arms. Unit: the family group. Source: `{inputs['sft_train']['episodes']}` "
        f"({inputs['sft_train']['attempts']} attempts, {inputs['sft_train']['harness_errors']} lost by the "
        "harness).",
        "",
    ]
    draws = design["on_saved_attempts"]
    draw_labels = [
        ("groups", "Family groups"),
        ("groups_with_equal_rewards", "Groups with equal rewards (dropped in both arms)"),
        ("groups_where_the_arms_keep_differently", "Groups that one arm keeps and the other drops"),
        ("groups_with_a_pass_and_a_fail", "Groups with a pass and a fail"),
        ("groups_where_the_bonus_is_not_zero", "Groups in which the bonus is not zero"),
        ("groups_where_the_order_of_the_attempts_changes", "Groups in which the bonus changes the order of "
                                                           "the attempts"),
        ("passes_in_those_groups", "Passing attempts in the groups with a bonus"),
    ]
    rows = [[label] + [str(draw[key]) for draw in draws] for key, label in draw_labels]
    for key, label in (("mean_size_of_the_bonus_points", "Mean size of the bonus of those passes, points"),
                       ("largest_bonus_points", "Largest bonus, points"),
                       ("mean_size_of_the_arm1_advantage_points",
                        "Mean size of the arm-1 advantage in those groups, points")):
        rows.append([label] + ["none" if draw[key] is None else plain(draw[key], 2) for draw in draws])
    lines += md_table([""] + [f"Draw {draw['draw']} (attempts {draw['sample_positions'][0]} to "
                              f"{draw['sample_positions'][1]} of each wording)" for draw in draws], rows)
    lines += [
        "",
        "## T10. Intervals of the saved comparisons (dev)",
        "",
        f"Rule: the difference \"b minus a\" of two arms, family by family, with the paired bootstrap over "
        f"families of the plan (`paired_family_bootstrap`, {dev['interval']['paired_family_bootstrap']} "
        f"version; {plain(100 * dev['interval']['level'])}%, {dev['interval']['n_draws']} draws, seed "
        f"{dev['interval']['seed']}; percentile form). Pass rate: the row of the frozen table program. All "
        "wordings pass: the same function on the family values of that metric (ours; not in the frozen "
        "tables). Unit: points; the family is the unit that is drawn again. A comparison of two GRPO arms is "
        f"not a comparison of the plan: it shows two runs of one recipe. Source: {dev_files}.",
        "",
    ]
    lines += md_table(
        ["Comparison", "Families", "Pass rate: difference (interval)", "Width", "All wordings pass: "
         "difference (interval)", "Width", "Families that differ on all wordings pass", "In the plan"],
        [[f"{row['b']} minus {row['a']}", str(row["families"]), interval(row["pass_rate"]),
          pts(row["pass_rate"]["width"]), interval(row["all_wordings_pass"]),
          pts(row["all_wordings_pass"]["width"]), str(row["all_wordings_pass"]["families_that_differ"]),
          "yes" if row["in_the_plan"] else "no"] for row in dev["comparisons"]])
    needed = numbers["effect"]["families_that_must_gain"]
    lines += [
        "",
        "Smallest detectable effect of the frozen tables (`two_halves_mde`: a difference of two pass rates, "
        "sampling noise only, from one arm's own attempts): "
        + "; ".join(f"{arm} {pts(dev['arms'][arm]['smallest_detectable_effect']['mde'])} points"
                    for arm in arms if dev["arms"][arm]["smallest_detectable_effect"]) + ".",
        "",
        "## T11. How many families must gain before the interval can be above zero",
        "",
        f"Rule: one draw of the bootstrap takes {needed['families']} families with replacement. If a gain "
        "sits in k families and every other family is equal in both arms, a draw without those k families "
        f"has a difference of exactly zero. The share of such draws is (1 - k / {needed['families']}) ^ "
        f"{needed['families']}. The lower end of a {plain(100 * needed['level'])}% interval is the "
        f"{plain(100 * needed['tail'], 2)}% quantile of the draws, so it is zero while that share is above "
        f"{plain(100 * needed['tail'], 2)}%. The last column runs the frozen bootstrap on made-up family "
        "values (a gain in the first k families, zero elsewhere). This is a limit of the method, not a "
        "power calculation.",
        "",
    ]
    lines += md_table(
        ["Families that gain (k)", "Draws without them", "Lower end above zero, by the formula",
         "Lower end above zero, by the frozen bootstrap"],
        [[str(row["families_that_gain"]), pct(row["share_of_draws_without_them"]),
          "yes" if row["lower_end_above_zero_by_the_formula"] else "no",
          "yes" if row["lower_end_above_zero_by_the_frozen_bootstrap"] else "no"] for row in needed["lines"]])
    spread = numbers["effect"]["spread_between_the_grpo_arms"]
    if spread:
        lines += [
            "",
            f"The {len(grpo_arms)} GRPO arms are runs of one recipe with other seeds. Largest minus smallest "
            f"value on dev: pass rate {pts(spread['pass_rate'])} points; all wordings pass "
            f"{pts(spread['all_wordings_pass'])} points; share of the variance due to wording "
            f"{pts(spread['variance_share_of_wording'])} points.",
        ]
    lines += [
        "",
        "## T12. Cost",
        "",
        "Rule: one GPU ran each saved run, so its GPU hours are the wall-clock seconds of its run manifest "
        "(`seconds`) divided by 3600: run time, not rented time. One run of the design is priced as a saved "
        "GRPO run plus the dev run of its last checkpoint (mean over the saved GRPO arms): an estimate, because "
        f"a step of the design has {design['attempts_per_step']} attempts and a saved step has "
        f"{first['attempts_per_step']}. Source: `run_manifest.json` of each run"
        + ("" if cost["prices"] is None else f"; `{cost['prices']['file']}` for the prices") + ".",
        "",
    ]
    lines += md_table(["Saved run", "GPU", "GPU hours of the training", "GPU hours of the dev run"],
                      [[arm, str(row["gpu"]), plain(row["training_hours"], 2), plain(row["dev_run_hours"], 2)]
                       for arm, row in cost["saved"].items()])
    lines += [""]
    lines += md_table(
        ["Design", "Training runs", "GPU hours (estimate)", "USD (estimate)"],
        [[f"{N_ARMS} arms x {row['seeds_per_arm']} seeds", str(row["runs"]), plain(row["gpu_hours"]),
          "no price" if row["usd_low"] is None else f"{plain(row['usd_low'])} to {plain(row['usd_high'])}"]
         for row in cost["designs"].values()])
    if cost["prices"] is not None:
        prices = cost["prices"]
        lines += [
            "",
            "Prices: the lowest and the highest hourly price of the machines rented up to the end of the last "
            f"saved GRPO training run ({prices['until_utc']}, `ended_utc` of its run manifest): "
            f"{plain(prices['lowest_usd_per_hour'], 4)} and {plain(prices['highest_usd_per_hour'], 4)} USD per "
            f"hour (rentals {', '.join(prices['rentals'])}). A machine rented later does not change them. One "
            f"design run: {plain(cost['hours_of_one_design_run'], 2)} GPU hours. The ledger bills a rental "
            "from rent to stop, with start-up and downloads; these amounts leave that time out.",
        ]
    else:
        lines += ["", "No price: the ledger is missing, or a saved GRPO run has no `ended_utc`."]
    lines += ["", "## T13. The test set", ""]
    if numbers["test"] is None:
        lines += ["The test set is not in this file: these runs were not there, or not complete, when the "
                  "file was made: " + ", ".join(f"`{path}`" for path in inputs["test_runs_not_there"])
                  + ". When they are there, the script puts the tables T3 to T7 of the test set here."]
    else:
        test_files = ", ".join(f"`{inputs['test'][arm]['episodes']}`" for arm in arms)
        lines += ["The same tables as T3 to T7, on the test set. The test set was not used for any choice."]
        lines += split_tables(numbers, numbers["test"], "test", None, test_files)
    lines += [
        "",
        "## Files",
        "",
    ]
    lines += md_table(["What", "File", "Start of its SHA-256"],
                      [[f"attempts of {arm} on dev", f"`{inputs['dev'][arm]['episodes']}`",
                        inputs["dev"][arm]["episodes_sha256"][:12]] for arm in arms]
                      + [[f"training log of {arm}", f"`{runs[arm]['folder']}/train_log.jsonl`",
                          runs[arm]["train_log_sha256"][:12]] for arm in grpo_arms]
                      + [[f"{runs[arm]['rollout_files']} step files of {arm} (hash of their `sha256sum` "
                          "lines)", f"`{runs[arm]['folder']}/rollouts/`",
                          runs[arm]["rollout_files_sha256"][:12]] for arm in grpo_arms]
                      + [["attempts of the SFT model on training families",
                          f"`{inputs['sft_train']['episodes']}`",
                          inputs["sft_train"]["episodes_sha256"][:12]]]
                      + [[f"frozen table ({row['values_compared']} values compared, all equal)",
                          f"`{row['file']}`", row["sha256"][:12]] for row in inputs["saved_tables"]]
                      + ([] if inputs["scenario_file"] is None else
                         [["scenario file (dev)", f"`{inputs['scenario_file']['file']}`",
                           inputs["scenario_file"]["sha256"][:12]]])
                      + [["code that computes the numbers", f"`{name}`", file_hash[:12]]
                         for name, file_hash in numbers["code"]["files_sha256"].items()])
    return "\n".join(lines) + "\n"


def split_tables(numbers: dict, split: dict, name: str, first_number: int | None, files: str) -> list[str]:
    """The five tables of one split (T3 to T7 for dev). For the test set they have no number."""
    arms, grpo_arms = numbers["arms"], numbers["grpo_arms"]

    def title(offset: int, text: str) -> str:
        return f"## T{first_number + offset}. {text}" if first_number is not None else f"### {text}"

    counts = split["attempts_per_scenario"]
    per_scenario = str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
    lines = [
        "",
        title(0, f"Per arm, on {name}"),
        "",
        f"{split['families']} families x {len(split['styles'])} wording styles x {per_scenario} attempts = "
        f"{split['attempts_per_arm']} attempts per arm. Source: {files}.",
        "",
    ]
    lines += arm_table(split, arms)
    lines += [
        "",
        "- Pass rate: the mean over families of the family's share of passed attempts (`pass_rate`). In "
        "brackets: the passed attempts of all attempts that the harness did not lose.",
        "- All wordings pass: per family the product of the pass rates of its wording styles, then the mean "
        "over families (`all_wordings_pass`). It is the chance that one attempt per wording passes in every "
        "wording.",
        "- All wordings pass, training styles only: the same frozen function on the attempts of the styles "
        "that training uses (a held-out style is one that no training scenario uses).",
        "- Room: 100% minus all wordings pass, in points.",
        f"- Families whose wordings disagree (our rule): {rule_words(split)}.",
        "- Variance due to wording (`variance_shares`): of the variation of pass or fail inside families, "
        "the share that is between wordings. The variance table below says exactly what it is.",
        "",
        f"Intervals ({plain(100 * split['interval']['level'])}%, families drawn again; "
        "`family_bootstrap_ci`): "
        + "; ".join(f"{arm}: pass rate {pct(split['arms'][arm]['pass_rate_lo'])} to "
                    f"{pct(split['arms'][arm]['pass_rate_hi'])}, all wordings pass "
                    f"{pct(split['arms'][arm]['all_wordings_pass_lo'])} to "
                    f"{pct(split['arms'][arm]['all_wordings_pass_hi'])}" for arm in arms) + ".",
        "",
        title(1, f"Where the room sits after GRPO, on {name}"),
        "",
        f"Rule: a family's part of the room is (1 minus its product of style pass rates) divided by the "
        f"{split['families']} families. One family is at most {pts(split['points_of_one_family'])} points. "
        f"Unit: points of all wordings pass. The document calls the room \"little\" when it is below "
        f"{pts(LITTLE_ROOM)} points in every GRPO arm (a judgement).",
        "",
    ]
    for arm in grpo_arms:
        row = split["arms"][arm]
        lines += [f"{arm}: {pts(row['room'])} points of room, in "
                  f"{row['families_below_one_on_all_wordings']} of {row['families']} families.", ""]
        lines += md_table(["Family", "Expected outcome"] + style_heads(split) + ["Part of the room, points"],
                          [[family_id, str(split["family_rows"][family_id]["outcome"])]
                           + style_cells(split["family_rows"][family_id]["arms"][arm], split["styles"])
                           + [plain(100 * part, 2)] for family_id, part in row["room_by_family"].items()])
        lines += [""]
    lines += [
        title(2, f"Families whose wordings disagree, on {name}"),
        "",
        f"Rule (ours): {rule_words(split)}. Each cell: passes of attempts, per wording style in rising "
        "order. A row is in the table if the family meets the rule in at least one arm; the last column "
        "names those arms.",
        "",
    ]
    disagree = sorted({family_id for arm in arms
                       for family_id in split["arms"][arm]["families_whose_wordings_disagree"]})
    if disagree:
        rows = []
        for family_id in disagree:
            family = split["family_rows"][family_id]
            cells = [" / ".join(str((family["arms"][arm]["passes_by_style"].get(str(style)) or ["-"])[0])
                                for style in split["styles"]) for arm in arms]
            rows.append([family_id, str(family["outcome"])] + cells
                        + [", ".join(arm for arm in arms if family["arms"][arm]["wordings_disagree"])])
        lines += md_table(["Family", "Expected outcome"] + arms + ["Meets the rule in"], rows)
    else:
        lines += ["No family meets the rule in any arm."]
    lines += [
        "",
        "Count per arm: " + "; ".join(f"{arm} {len(split['arms'][arm]['families_whose_wordings_disagree'])}"
                                      for arm in arms) + f" (of {split['families']} families).",
        "",
        title(3, f"Families with fewer passes than under sft, on {name}"),
        "",
        "Rule: the family's share of passed attempts, over all its wording styles, is lower in the GRPO arm "
        "than in sft. Source: the same files.",
        "",
    ]
    for arm in grpo_arms:
        found = split["families_with_fewer_passes_than_sft"][arm]
        lines += [f"- {arm}: " + (listed(found) if found else "no family") + "."]
    every = split["families_with_fewer_passes_than_sft_in_every_grpo_arm"]
    lines += ["- In every GRPO arm: " + (listed(every) if every else "no family") + ".", ""]
    facts = numbers["family_facts"] if name == DEV else None
    for family_id in every:
        lines += family_table(split, family_id, arms)
        lines += [""]
        for arm in grpo_arms:
            row = split["family_rows"][family_id]["arms"][arm]
            failed = row["valid_attempts"] - row["passed_attempts"]
            closes = "; ".join(f"{count} of {failed} close the request as {close}" if close != "no close"
                               else f"{count} of {failed} do not close the request"
                               for close, count in row["closes_of_the_failed_attempts"].items())
            line = f"- {arm}, failed attempts of {family_id}: {closes or 'none'}."
            if facts and family_id in facts and failed:
                right = row["person_of_the_close_in_failed_attempts"].get(
                    facts[family_id]["person_of_the_request"], 0)
                line += (f" {right} of {failed} name the person of the request in the close.")
            lines += [line]
        if facts and family_id in facts:
            fact = facts[family_id]
            others = fact["status_of_the_other_people_with_that_name"]
            lines += [f"- Scenario file: the expected close of {family_id} is {fact['expected_close']}. "
                      f"{fact['people_with_the_name_of_the_request']} people of its world have the name of "
                      "the request"
                      + (f"; the status of the other one is {others[0]}." if len(others) == 1 else
                         f"; the status of the others: {listed(others)}." if others else ".")]
        lines += [""]
    lines += [
        title(4, f"Variance inside families, on {name}"),
        "",
        "Rule (`variance_shares`, frozen): score a pass 1 and a fail 0. Inside each family, the squared "
        "distances of its attempts from the family's pass rate split into two sums: between the pass rates "
        "of its wording styles (each style counted with its number of attempts), and between the attempts "
        "of one style. Both sums are added up over all families. \"Wording\" is the first sum as a share of "
        "both, \"sampling\" is the second. Differences between families are in neither. \"If sampling "
        "only\" is the share to expect when wording has no effect: the style rates still differ by chance. "
        "The share is of the variation that is left, so it says nothing about how much is left: the "
        "column of failed attempts does.",
        "",
    ]
    rows = []
    for arm in arms:
        row = split["arms"][arm]
        cells = [arm, f"{row['failed_attempts']} of {row['valid_attempts']}",
                 pct(row["variance_shares"]["wording"]), pct(row["variance_shares"]["sampling"]),
                 pct(row["variance_shares"]["wording_if_sampling_only"])]
        if every:
            rest = split["without_those_families"].get(arm)
            cells += ["no attempt" if rest is None else pct(rest["variance_shares"]["wording"]),
                      "no attempt" if rest is None else f"{rest['failed_attempts']} of "
                                                         f"{rest['valid_attempts']}"]
        rows.append(cells)
    head = ["Arm", "Failed attempts", "Wording", "Sampling", "If sampling only"]
    if every:
        head += [f"Wording, without {listed(every)}", f"Failed attempts, without {listed(every)}"]
    lines += md_table(head, rows)
    return lines


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def _fail_sentences(numbers: dict, family_id: str) -> list[str]:
    """What the saved attempts show about one family that got worse: in which styles each GRPO arm is at or
    below LOW ("fails"), and how the failed attempts close the request."""
    dev, grpo_arms = numbers["dev"], numbers["grpo_arms"]
    facts = (numbers["family_facts"] or {}).get(family_id)
    sentences, agree_and_fail, closes = [], [], []
    for arm in grpo_arms:
        row = dev["family_rows"][family_id]["arms"][arm]
        low, _ = low_and_high(row, dev["styles"])
        if low and len(low) == len(dev["styles"]):
            passes = listed([str(row["passes_by_style"][str(style)][0]) for style in dev["styles"]])
            sentences.append(f"{arm} fails in all {len(low)} styles ({passes} of "
                             f"{row['passes_by_style'][str(dev['styles'][0])][1]}): by the rule of this note its "
                             "wordings do not disagree.")
            agree_and_fail.append(arm)
        elif low:
            sentences.append(f"{arm} fails in " + ("style " if len(low) == 1 else "styles ") + listed(low)
                             + ".")
        failed = row["valid_attempts"] - row["passed_attempts"]
        if failed:
            close, count = next(iter(row["closes_of_the_failed_attempts"].items()))
            closes.append((arm, close, count, failed))
    one_close = len({close for _, close, _, _ in closes}) == 1 and closes[0][1] != "no close"
    if closes and one_close and all(count == failed for _, _, count, failed in closes):
        sentences.append(f"Every failed attempt on {family_id} "
                         f"{'in these arms' if len(closes) > 1 else 'in ' + closes[0][0]} "
                         f"closes the request as {closes[0][1]} ("
                         + listed([f"{count} of {failed}" for _, _, count, failed in closes]) + ")"
                         + (f"; the correct close is {facts['expected_close']}." if facts else "."))
    else:
        for arm, close, count, failed in closes:
            sentences.append(f"{count} of {failed} failed attempts of {arm} "
                             + ("do not close the request." if close == "no close"
                                else f"close the request as {close}."))
        if facts and closes:
            sentences.append(f"The correct close is {facts['expected_close']}.")
    if agree_and_fail:
        sentences.append("Agreement must be read beside the pass rate.")
    return sentences


def render_doc(numbers: dict, tables_shown: str) -> str:
    """The document for the reviewer. Every number in it is read from `numbers`."""
    arms, grpo_arms, dev = numbers["arms"], numbers["grpo_arms"], numbers["dev"]
    one, design, cost = numbers["one_wording_group"], numbers["design"], numbers["cost"]
    example, runs = one["example"], one["saved_runs"]
    first = runs[grpo_arms[0]]
    bonus = plain(design["bonus_points"])
    trained = [SFT] + grpo_arms
    counts = dev["attempts_per_scenario"]
    per_scenario = str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
    each = {arm: dev["arms"][arm] for arm in arms}
    n_seeds = count_word(len(grpo_arms))

    # --- The head, and parts 1 and 2 ---
    lowest = [runs[arm]["lowest_pass_reward"] for arm in grpo_arms]
    if all(value is not None for value in lowest):
        pass_range_line = (
            f"A saved pass scores {plain(min(lowest))} to {first['pass_reward']} points: {first['pass_reward']} "
            "minus small penalties for malformed calls and extra turns (T2). A fail scores "
            + ("0." if all(runs[arm]["fails_not_at_zero"] == 0 for arm in grpo_arms) else "at most 0."))
    else:
        pass_range_line = (f"A pass scores at most {first['pass_reward']} points (penalties for malformed calls "
                           "and extra turns); a fail scores 0.")
    lines = [
        "# Stretch: a bonus when rewordings give the same correct action",
        "",
        "**The experiment was not run.** This file analyses the saved runs and gives a design. "
        f"`scripts/stretch_numbers.py` makes every number from saved files; `{tables_shown}` gives the rule "
        "and source of each (tables T1 to T13).",
        "",
        "## 1. What the stretch asks",
        "",
        "The brief: add a bonus when reworded versions of one scenario produce the same correct action, and "
        "test whether it helps beyond a reward for correctness alone.",
        "",
        "A *family* is one world with one correct end state. A *scenario* is one family in one wording "
        "style. An *attempt* is one conversation. The verifier accepts one end state per family, so \"the "
        "same correct action\" means: the attempt passes, and so do the attempts on the family's other "
        "wordings. The measure is *all wordings pass*: per family the product of the pass rates of its "
        "wording styles, then the mean over families.",
        "",
        "## 2. What the bonus can do in groups of one wording",
        "",
        f"A GRPO *group* here is the {first['attempts_per_group']} attempts of one scenario in one step. The "
        "*advantage* of an attempt is its reward minus the mean reward of its group. "
        + pass_range_line
        + " A bonus for agreement with the other wordings is one number b for the whole group, because every "
        "attempt of the group has the same other wordings. Add b to every pass of a group with k passes of n "
        "attempts. The mean of the group rises by b k/n, so every pass gains b (1 - k/n) and every fail loses "
        "b k/n.",
        "",
        f"If every pass scores {first['pass_reward']}, this multiplies each advantage by "
        f"({first['pass_reward']} + b)/{first['pass_reward']}. Example (T1): {example['passes']} passes, "
        f"{example['fails']} fails, b = {bonus}. A pass goes from "
        f"{signed_short(example['without']['advantage_of_a_pass'])} to "
        f"{signed_short(example['with']['advantage_of_a_pass'])} and a fail from "
        f"{signed_short(example['without']['advantage_of_a_fail'])} to "
        f"{signed_short(example['with']['advantage_of_a_fail'])}: {plain(example['factor'], 2)} times.",
        "",
        f"With {bonus} points on every pass, the saved training (T2) keeps the same groups ("
        + listed([f"{runs[arm]['kept_with_bonus']} of {runs[arm]['groups']} in {arm}" for arm in grpo_arms])
        + "), and no kept group changes the order of its attempts. In "
        + listed([str(runs[arm]["kept_one_outcome"]) for arm in grpo_arms])
        + " of the kept groups every attempt passes, and the bonus changes nothing.",
        "",
        "So in a group of one wording the bonus cannot make the training prefer one passing attempt over "
        "another: it adds b to each pass-fail gap and leaves the gaps between passes as they are. Where every "
        f"pass scores {first['pass_reward']} ("
        + listed([str(runs[arm]["kept_mixed_full_reward"]) for arm in grpo_arms])
        + " kept groups with passes and fails), that only changes the weight of the group's step, as b "
        "differs between groups; in "
        + listed([str(runs[arm]["kept_mixed_other"]) for arm in grpo_arms])
        + " such groups a pass scores less (T2). For the bonus to differ between passes, a group must mix "
        "the wordings of one family.",
        "",
        "## 3. How much there is to gain",
        "",
        f"Dev: {dev['families']} families x {len(dev['styles'])} wording styles x {per_scenario} attempts = "
        f"{dev['attempts_per_arm']} attempts per arm (T3 to T7).",
        "",
    ]
    lines += arm_table(dev, arms, compact=True)
    held_words = ""
    if dev["styles_in_no_training_scenario"]:
        held = dev["styles_in_no_training_scenario"]
        held_words = (f" A *held-out* style is a wording that no training scenario uses: here "
                      + ("style " if len(held) == 1 else "styles ") + listed(held) + ".")

    # --- Part 3: the room, the variance, the families ---
    top = {arm: next(iter(each[arm]["room_by_family"].items()), None) for arm in grpo_arms}
    one_top = all(top.values()) and len({item[0] for item in top.values()}) == 1
    rooms = [pts(each[arm]["room"]) for arm in grpo_arms]
    with_room = [f"{each[arm]['families_below_one_on_all_wordings']} of {each[arm]['families']}"
                 for arm in grpo_arms]
    if len(grpo_arms) > 1 and len(set(rooms)) == 1 and len(set(with_room)) == 1:
        room_line = (f"**Room after GRPO** (100% minus all wordings pass): {rooms[0]} points in each GRPO "
                     f"arm, in {with_room[0]} families.")
    else:
        room_line = ("**Room after GRPO** (100% minus all wordings pass): "
                     + listed([f"{room} points in {families} families ({arm})"
                               for room, families, arm in zip(rooms, with_room, grpo_arms)]) + ".")
    if one_top:
        room_line += (f" One family, {top[grpo_arms[0]][0]}, holds "
                      + same_or_listed([pts(top[arm][1]) for arm in grpo_arms], grpo_arms, short=True)
                      + " of these points.")
    elif all(top.values()):
        room_line += (" The largest part: " + listed([f"{top[arm][0]} with {pts(top[arm][1])} points in {arm}"
                                                      for arm in grpo_arms]) + ".")
    held = dev["styles_in_no_training_scenario"]
    if held and one_top:
        top_id = top[grpo_arms[0]][0]
        rows_top = dev["family_rows"][top_id]["arms"]
        zero = [style for style in held
                if all((rows_top[arm]["passes_by_style"].get(str(style)) or [None])[0] == 0 for arm in grpo_arms)]
        if zero:
            counts_zero = rows_top[grpo_arms[0]]["passes_by_style"][str(zero[0])][1]
            room_line += (f" In every GRPO arm {top_id} passes 0 of {counts_zero} in its held-out "
                          + ("style " if len(zero) == 1 else "styles ") + listed(zero)
                          + ", so its product stays 0: a bonus on the training styles reaches these points "
                          "only through a gain on a held-out wording.")
    worse = dev["families_with_fewer_passes_than_sft_in_every_grpo_arm"]
    without = dev["without_those_families"]
    after_training = sorted({family_id for arm in trained
                             for family_id in each[arm]["families_whose_wordings_disagree"]})
    n_base = len(each[BASE]["families_whose_wordings_disagree"])
    in_base = f"the {n_base} families of base are in T5" if n_base else "base has none"
    shown_families = (worse or after_training)[:MAX_FAMILIES_IN_DOC]
    if len(worse) == 1 and after_training == worse:
        where = listed([arm for arm in trained if worse[0] in each[arm]["families_whose_wordings_disagree"]])
        family_line = (f"After training only **{worse[0]}** meets the rule (in {where}; {in_base}). It is "
                       "also the only family with fewer passes than under sft in every GRPO arm (T6):")
    else:
        family_line = "**Families (T5, T6).** "
        if after_training:
            met_in = {family_id: [arm for arm in trained
                                  if family_id in each[arm]["families_whose_wordings_disagree"]]
                      for family_id in after_training}
            family_line += ("After training the rule is met by "
                            + listed([f"{family_id} (in {listed(found)})"
                                      for family_id, found in met_in.items()]) + f"; {in_base}.")
        else:
            family_line += f"After training no family meets the rule; {in_base}."
        if worse:
            family_line += (" Fewer passes than under sft in every GRPO arm: " + listed(worse) + ".")
        else:
            family_line += " No family has fewer passes than under sft in every GRPO arm."
        if len(worse or after_training) > MAX_FAMILIES_IN_DOC:
            family_line += f" The first {MAX_FAMILIES_IN_DOC} are shown here."
    lines += [
        "",
        f"Wordings disagree (our rule): {rule_words(dev)}." + held_words,
        "",
        room_line,
        "",
        "**Variance due to wording** (`variance_shares`, fixed before the runs; T7): the share of the pass/fail "
        f"variation inside families that lies between the {len(dev['styles'])} styles, not within a style. "
        "Chance alone gives about "
        f"{pct(each[BASE]['variance_shares']['wording_if_sampling_only'])}. It is a share of what is left: "
        "after GRPO, " + listed([str(each[arm]["failed_attempts"]) for arm in grpo_arms])
        + f" of {each[grpo_arms[0]]['valid_attempts']} attempts fail."
        + ("" if not worse else f" Without {listed(worse)} it is "
           + listed([pct(without[arm]["variance_shares"]["wording"]) for arm in grpo_arms]) + "."),
        "",
        family_line,
        "",
    ]
    for family_id in shown_families:
        lines += family_table(dev, family_id, arms, compact=True)
        lines += [""]
        if family_id in worse:
            lines += [" ".join(_fail_sentences(numbers, family_id)), ""]
    if numbers["test"] is None:
        lines += ["The test set is not in this file: its runs were not there (T13).", ""]
    else:
        test = numbers["test"]
        lines += [
            f"**Test set** (T13; {test['families']} families): all wordings pass is "
            f"{pct(test['arms'][BASE]['all_wordings_pass'])} (base), "
            f"{pct(test['arms'][SFT]['all_wordings_pass'])} (sft), "
            + listed([f"{pct(test['arms'][arm]['all_wordings_pass'])} ({arm})" for arm in grpo_arms])
            + "; room after GRPO: "
            + listed([f"{pts(test['arms'][arm]['room'])} points in "
                      f"{families_word(test['arms'][arm]['families_below_one_on_all_wordings'])} ({arm})"
                      for arm in grpo_arms]) + ".",
            "",
        ]

    # --- Part 4: the design ---
    wordings = design["example"]["wordings"]
    names = list(wordings)
    with_pass = [name for name in names if "bonus_of_a_pass" in wordings[name]]
    a_fail = next((row["arm1_advantage_of_a_fail"] for row in wordings.values()
                   if "arm1_advantage_of_a_fail" in row), None)
    rank = design["rank_rule"]
    draws = design["on_saved_attempts"]
    with_bonus = [draw for draw in draws if draw["mean_size_of_the_bonus_points"] is not None]
    size_line = (
        f"On saved sft attempts cut into such groups in {count_word(len(draws))} "
        f"way{'s' if len(draws) != 1 else ''} (T9), the bonus is not zero in "
        + listed([str(draw["groups_where_the_bonus_is_not_zero"]) for draw in draws])
        + f" of {draws[0]['groups']} groups.")
    if with_bonus:
        size_line += (" There it moves a pass by "
                      + listed([plain(draw["mean_size_of_the_bonus_points"]) for draw in with_bonus])
                      + " points on average, "
                      + plain(max(draw["largest_bonus_points"] for draw in with_bonus)) + " at most.")
    lines += [
        "## 4. The design that would test it",
        "",
        f"{count_word(design['arms']).capitalize()} new arms start from the saved SFT weights and differ "
        "only in the bonus (T8). The saved runs cannot be arm 1: their groups differ."
        + (" The saved code builds each group from one scenario: the design needs new training code."
           if one["saved_code_refuses_a_group_of_two_scenarios"] else ""),
        "",
    ]
    lines += md_table(["Part", "Value", "Origin"], [
        ["Group", f"one family: its {design['wordings']} training wordings x "
                  f"{design['attempts_per_wording']} attempts = {design['attempts_per_group']}; advantage = "
                  "reward minus the group mean", f"{design['attempts_per_wording']}: judgement"],
        ["Step", f"{design['families_per_step']} families = {design['attempts_per_step']} attempts (saved "
                 f"runs: {first['scenarios_per_step']} x {first['attempts_per_group']} = "
                 f"{first['attempts_per_step']}); {first['steps']} steps as saved",
         f"{design['families_per_step']}: judgement"],
        ["Arm 1", "the saved reward (pass or fail, with the penalties of part 2)", "the saved runs"],
        ["Arm 2", f"arm 1; a pass also gets {bonus} x (q - q_mean) points", f"{bonus}: judgement"],
        ["Runs", f"{design['seeds_per_arm']} seeds per arm", "judgement"],
        ["Other settings", f"as in `grpo_config` of `{first['folder']}/run_manifest.json` (learning rate "
                           f"{first['learning_rate']:g}, tie_tol {first['tie_tol']:g}, start "
                           f"`{first['start_adapter']}`)", "the saved runs"],
    ])
    example_line = (
        "q is the share of passes among the family's valid attempts in the other wordings (a pass with no "
        "such attempt gets no bonus); q_mean is its mean over the family's passes. The bonuses of a group add up to zero: fails keep their advantage, passes are "
        "ranked again. Example: wordings " + listed(names) + " pass "
        + listed([wordings[name]["passes"] for name in names])
        + f" of {wordings[names[0]]['attempts']}.")
    if with_pass and a_fail is not None:
        example_line += (
            f" Arm 1: a pass {signed_short(wordings[with_pass[0]]['arm1_advantage_of_a_pass'])}, a fail "
            f"{signed_short(a_fail)}. Arm 2: "
            + listed([f"a pass in {name} {signed_short(wordings[name]['arm2_advantage_of_a_pass'])}"
                      for name in with_pass]))
        strongest = max(with_pass, key=lambda name: wordings[name]["passes"])
        weakest = min(with_pass, key=lambda name: wordings[name]["passes"])
        weaker_counts_more = (wordings[weakest]["arm2_advantage_of_a_pass"]
                              > wordings[strongest]["arm2_advantage_of_a_pass"] + TOLERANCE)
        example_line += ": the weaker wording counts more." if weaker_counts_more else "."
    lines += [
        "",
        example_line,
        "",
        "Measures on dev: all wordings pass (over all styles, and over the training styles only) and the "
        "variance due to wording; the pass rate must not fall. "
        "Rule (judgement): the bonus helps if every run of arm 2 is above every run of arm 1 on all wordings "
        f"pass. By chance alone that happens in 1 of {rank[str(design['seeds_per_arm'])]['ways']} cases (own "
        "calculation).",
        "",
        "## 5. What size of effect it could see",
        "",
        "This is no power calculation: "
        + (f"{n_seeds} saved seeds do not show" if len(grpo_arms) > 1 else "one saved seed does not show")
        + " the spread between runs (T10, T11).",
        "",
    ]

    # --- Part 5: what can be said about the size of an effect ---
    needed = numbers["effect"]["families_that_must_gain"]
    room_counts = sorted({each[arm]["families_below_one_on_all_wordings"] for arm in grpo_arms})
    n_room = listed(room_counts)
    max_room = max(each[arm]["room"] for arm in grpo_arms)
    to_sft = [row for row in dev["comparisons"] if row["a"] == SFT and row["b"] in grpo_arms]
    interval_line = (
        f"- **Interval.** The plan's paired bootstrap ({dev['families']} families drawn again "
        f"{dev['interval']['n_draws']:,} times, {plain(100 * dev['interval']['level'])}%) gives the saved "
        "comparisons with sft intervals "
        + listed([pts(row["pass_rate"]["width"]) for row in to_sft]) + " points wide on the pass rate and "
        + listed([pts(row["all_wordings_pass"]["width"]) for row in to_sft]) + " on all wordings pass.")
    between = [row for row in dev["comparisons"] if row["a"] in grpo_arms and row["b"] in grpo_arms]
    lines += [interval_line]
    by_k = {row["families_that_gain"]: row for row in needed["lines"]}
    if needed["families_needed"] is not None and needed["families_needed"] > 1:
        k = needed["families_needed"]
        families_line = (
            f"- **Families.** A gain in k families is absent from (1 - k/{needed['families']})^"
            f"{needed['families']} of the draws: "
            + listed([f"{pct(by_k[value]['share_of_draws_without_them'])} for k = {value}"
                      for value in sorted({1, k - 1, k}) if value in by_k])
            + f". So a {plain(100 * needed['level'])}% interval is above zero only if at least {k} families "
            "gain. If arm 1 ends where the saved runs did,")
        most_room = max(room_counts)
        if k <= most_room:
            families_line += (f" {k} of the {most_room} families with room would have to gain and no other "
                              "family lose")
        else:
            families_line += (f" only {most_room} {'family has' if most_room == 1 else 'families have'} room, "
                              f"fewer than {k}: the interval can be above zero only if other families gain too")
        families_line += "."
        if between:
            differ = max(row["all_wordings_pass"]["families_that_differ"] for row in between)
            families_line += (
                f" Yet two runs of one recipe already differ in {differ} of {dev['families']} families on all "
                "wordings pass. So a comparison on dev alone is unlikely to give an interval above zero (our "
                "judgement).")
        lines += [families_line]
    lines += [""]

    # --- Part 6: what the experiment would cost ---
    # No reason for not running it is given. Only a fact that a reader can check in the package is stated:
    # section 3 of the frozen analysis plan does not list the experiment (plan_lists_the_experiment).
    saved_cost = cost["saved"]
    main_design = cost["designs"][str(design["seeds_per_arm"])]
    # A sum is printed so that its printed parts add up to its printed total: the parts are the `seconds` of
    # the run manifests at one decimal (as the manifests save them), and the total is the sum of those parts.
    totals = {}
    per_run = []
    for arm in grpo_arms:
        parts = [round(saved_cost[arm]["training_seconds"], 1), round(saved_cost[arm]["dev_run_seconds"], 1)]
        totals[arm] = round(sum(parts), 1)
        per_run.append(f"{arm} {parts[0]:.1f} + {parts[1]:.1f} = {totals[arm]:.1f} s")
    mean_seconds = sum(totals.values()) / len(totals)
    cost_line = (
        "In run time (`seconds` of the run manifests), not rented time (T12), one design run counts as a saved "
        "GRPO run plus its dev run: " + listed(per_run)
        + (" = " if len(grpo_arms) == 1 else f"; mean {mean_seconds:.1f} s = ")
        + f"{plain(cost['hours_of_one_design_run'], 2)} hours. "
        f"{design['arms']} arms x {design['seeds_per_arm']} seeds = {main_design['runs']} runs: "
        f"{main_design['runs']} x {plain(cost['hours_of_one_design_run'], 2)} = about "
        f"{plain(main_design['gpu_hours'])} GPU hours.")
    if main_design["usd_low"] is not None:
        prices = cost["prices"]
        cost_line += (f" At {plain(prices['lowest_usd_per_hour'], 4)} to "
                      f"{plain(prices['highest_usd_per_hour'], 4)} USD per GPU hour (rentals up to the end of "
                      "the last saved GRPO run): "
                      f"{plain(main_design['usd_low'])} to {plain(main_design['usd_high'])} USD.")
    plan = numbers.get("analysis_plan")
    not_run = "The experiment was not run."
    if plan and plan["has_planned_extras"] and not plan["lists_the_experiment"]:
        not_run += (" The frozen `ANALYSIS_PLAN.md` does not list it among its arms or its planned extras "
                    "(section 3).")
    lines += [
        "## 6. What the experiment would cost",
        "",
        not_run,
        "",
        cost_line,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--main", type=Path, default=ROOT / "results" / "main",
                        help="the folder of the saved runs")
    parser.add_argument("--grpo", action="append", default=None, metavar="NAME",
                        help="the name of a GRPO arm; once per arm (default: grpo_seed1 and grpo_seed2)")
    parser.add_argument("--folder", action="append", default=[], metavar="KEY=PATH",
                        help="another place for one standard name, for example sft_dev=/some/folder")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results" / "spend_ledger.csv")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "data" / "scenarios",
                        help="the scenario folder; only dev.jsonl is read")
    parser.add_argument("--plan", type=Path, default=ROOT / "ANALYSIS_PLAN.md",
                        help="the frozen analysis plan; only its section 3 is read")
    parser.add_argument("--out", type=Path, default=None, help=f"folder for {JSON_NAME} and {TABLES_NAME}")
    parser.add_argument("--doc", type=Path, default=None, help="the document for the reviewer")
    parser.add_argument("--draws", type=int, default=10_000, help="bootstrap draws")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--level", type=float, default=0.95, help="level of the intervals")
    parser.add_argument("--bonus", type=float, default=BONUS_POINTS,
                        help="design: the largest bonus of a pass, in points (a judgement)")
    parser.add_argument("--attempts-per-wording", type=int, default=ATTEMPTS_PER_WORDING,
                        help="design: attempts of each wording in a family group (a judgement)")
    parser.add_argument("--families-per-step", type=int, default=FAMILIES_PER_STEP,
                        help="design: family groups in a step (a judgement)")
    parser.add_argument("--seeds-per-arm", type=int, default=SEEDS_PER_ARM,
                        help="design: training runs of each arm (a judgement)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the three files and print where they are. Returns the exit code: 0, or 2 if nothing was
    written."""
    args = parse_args(argv)
    grpo_arms = list(args.grpo) if args.grpo else list(GRPO_ARMS)
    out = args.out if args.out is not None else args.main / "report" / "stretch"
    doc = args.doc if args.doc is not None else args.main.parent / DOC_NAME
    try:
        overrides = {}
        for item in args.folder:
            key, found, value = item.partition("=")
            if not found or not key or not value:
                raise ValueError(f"--folder must look like KEY=PATH, got {item!r}")
            overrides[key] = Path(value)
        numbers = build_numbers(
            args.main, grpo_arms, overrides, ledger=args.ledger, scenarios=args.scenarios, plan=args.plan,
            n_draws=args.draws, seed=args.seed, level=args.level, bonus_points=args.bonus,
            attempts_per_wording=args.attempts_per_wording, families_per_step=args.families_per_step,
            seeds_per_arm=args.seeds_per_arm)
    except KeyError as error:
        print(f"stretch_numbers: a saved file has no key {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as error:
        print(f"stretch_numbers: {error}", file=sys.stderr)
        return 2
    tables_path = out / TABLES_NAME
    numbers["outputs"] = {"numbers": shown_path(out / JSON_NAME), "tables": shown_path(tables_path),
                          "document": shown_path(doc)}
    tables_text = render_tables(numbers, shown_path(doc))
    doc_text = render_doc(numbers, shown_path(tables_path))
    out.mkdir(parents=True, exist_ok=True)
    doc.parent.mkdir(parents=True, exist_ok=True)
    (out / JSON_NAME).write_text(json.dumps(numbers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables_path.write_text(tables_text, encoding="utf-8")
    doc.write_text(doc_text, encoding="utf-8")
    print("\n".join(["The stretch experiment was not run: these files hold an analysis and a design.",
                     f"Dev, per arm (source: the attempts of each arm on dev; rules in {TABLES_NAME}, T3):"]
                    + arm_table(numbers["dev"], numbers["arms"])))
    print(f"wrote {shown_path(out / JSON_NAME)}, {shown_path(tables_path)} and {shown_path(doc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
