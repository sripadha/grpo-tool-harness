"""The formula audit of the saved GRPO training rollouts (ANALYSIS_PLAN.md section 8, line 4).

A GRPO run folder holds `rollouts/step_00012.episodes.jsonl` (the attempts of a step; the attempts of one
scenario are one group) and `train_log.jsonl` (one line per step). After a restart it also holds
`rollouts/start_<n>/` and `train_log.start_<n>.jsonl`: the steps that were run again. Every saved rollout
is audited, also those.

Four questions, for every saved rollout:
  1. No failing attempt has a positive reward. (Also the reverse: no passing attempt has a reward of 0 or
     below. The group rule of the trainer relies on both.)
  2. Every saved reward is the reward that the main and the second version of `reward` give for the saved
     record: its saved `passed`, the counts of its saved turns, its saved turn budget, and the config of
     accessdesk/config.py with the record's `settings`.
  3. The advantages can be made again from the saved rewards with `group_advantages`, and what the train
     log holds agrees with them. The log does NOT hold the advantages. It holds numbers that follow from
     them, and each is compared (LOG_COMPARED):
         scenarios            the scenario ids of the groups, in order
         n_episodes, n_harness_errors, mean_reward
         kept_groups          the number of kept groups
         n_trained_episodes   the valid attempts of the kept groups
         n_loss_tokens        the sampled tokens of those attempts (the sum of `n_tokens` of their turns)
         loss                 only if the step has a token file (`--save-tokens`): the loss, made again
                              from the advantages and the log-probabilities of the SAMPLER, must be near
                              the logged loss. "Near" is exact arithmetic: the trainer's loss uses the
                              trainer's log-probabilities, and the log holds the mean absolute gap between
                              the two (`logprob_gap`), so the two losses differ by at most
                              max|advantage| * gap.n * gap.mean / norm_const (and by the rounding of
                              the trainer's 32-bit sums, LOSS_ROUNDING).
     The main and the second version of `group_advantages` must also give the same advantages.
  4. Every kept group has unequal rewards, and every dropped group has equal rewards or lost more than
     half of its attempts to harness errors. This is asked in plain whole numbers, without
     `group_advantages`. The kept groups are the ones that `group_advantages` keeps with the `tie_tol` of
     the run. A run with a `tie_tol` below 0 keeps groups with equal rewards: the audit reports each.

Not asked here: whether `passed` is right for the final state. That is the replay check
(scripts/replay_check.py), which needs the scenario files; it can be run on every rollout file.

What the audit cannot show. The trainer saves its advantages nowhere, and it does not save which groups
it kept. So questions 3 and 4 are asked of the advantages and the groups that the audit makes again from
the saved rewards. The only witnesses of what the trainer itself used are the numbers of its log line:
the count of kept groups, of trained attempts and of loss tokens, and the loss. The counts are exact. The
loss comparison is a weak check: its bound grows with the gap between the sampler's and the trainer's
log-probabilities, and at a gap that the stop rule allows it can be larger than the loss itself. Then
even advantages with the opposite sign pass. A trainer that kept a group with equal rewards and dropped
one with unequal rewards has the same count of kept groups; it shows only if the count of trained
attempts or of loss tokens differs, or in the loss within its bound. The output holds the bound beside
each step (`loss_checks`) and says all this in plain words (WHAT_THE_CHECKS_SEE).

Nothing here needs torch: `reward_matrix` of accessdesk/train/grpo.py is plain numpy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from accessdesk import impl
from accessdesk.audit import load_script
from accessdesk.harness.records import read_records
from accessdesk.train import grpo

SHOWN = 10                          # differences that the output holds
VERSIONS = ("main", "second")       # the two versions of reward and of group_advantages
CONFIG_KEYS = ("n_scenarios", "group_size", "tie_tol")
CONFIG_FILES = ("run_manifest.json", "latest.json")       # where the GRPO config of a run is read from
LOG_COMPARED = ("scenarios", "n_episodes", "n_harness_errors", "mean_reward", "kept_groups",
                "n_trained_episodes", "n_loss_tokens", "loss")
EPISODES_FILE = re.compile(r"step_(\d+)\.episodes\.jsonl")
START_FOLDER = re.compile(r"start_(\d+)")
FLOAT_SLACK = 1e-9                  # two numbers that should be equal up to rounding
# The trainer adds the log-probabilities up in 32-bit numbers. The rounding of such a sum is at most about
# this share of the sum of the absolute values (a few hundred tokens, each rounded to 6e-8 of its size).
LOSS_ROUNDING = 2e-5
Difference = tuple[str, str]        # (kind, what differs)
# One plain paragraph for the output: what each of the four checks can see, and what it cannot.
WHAT_THE_CHECKS_SEE = (
    "What each of the four checks can see and cannot see. "
    "(1) Signs: every saved attempt is read, so a failed attempt with a reward above 0 is always found. "
    "This check takes the saved pass or fail as it is: it cannot see a wrong verdict. That is the job of "
    "scripts/replay_check.py. "
    "(2) Rewards: every saved reward is made again by two separately written reward functions from the "
    "saved verdict, the saved turns and the saved turn budget. It cannot see a verdict or a turn budget "
    "that is wrong in the record. "
    "(3) Advantages: the trainer's own advantages are saved nowhere. The audit makes them again from the "
    "saved rewards with two separately written functions, which must agree, and compares what the train "
    "log holds: the scenario ids, the counts of attempts and harness errors, the mean reward, the counts "
    "of kept groups, of trained attempts and of loss tokens, and the loss. The counts are exact. The loss "
    "comparison is a WEAK check: the logged loss was made with the trainer's log-probabilities, the audit "
    "has only the sampler's, and the two losses may differ by max|advantage| * gap.n * gap.mean / "
    "norm_const. At a mean gap that the stop rule allows (up to 0.02) this bound can be larger than the "
    "loss itself, and then even advantages with the opposite sign would pass. The bound stands beside "
    "each step in `loss_checks`. A change of the advantages is sure to show only if it moves the loss by "
    "more than twice the bound. So the audit shows that the logged numbers fit the advantages that it "
    "makes again; it does not prove that the trainer used them. "
    "(4) Groups: the kept and the dropped groups are those that the audit makes again from the saved "
    "rewards, and each is checked in whole numbers against the rule. Which groups the trainer itself kept "
    "is saved nowhere. If it kept other groups, this shows only where the count of kept groups, of trained "
    "attempts or of loss tokens in its log differs, or in the loss within the bound."
)


# ---------------------------------------------------------------------------
# One record: questions 1 and 2
# ---------------------------------------------------------------------------
def sign_differences(record: dict) -> list[Difference]:
    """Question 1 for one valid record: a fail with a reward above 0, or a pass with a reward of 0 or
    below."""
    passed, reward = record["passed"], record["reward"]
    if not passed and reward > 0:
        return [("fail_with_positive_reward", f"failed, but its reward is {reward}")]
    if passed and reward <= 0:
        return [("pass_without_positive_reward", f"passed, but its reward is {reward}")]
    return []


def reward_differences(record: dict) -> list[Difference]:
    """Question 2 for one valid record: the saved reward against the main and the second version of
    `reward`. The counts come from the saved turns; a saved counter that is not the count of the turns is
    a difference too. The answer must be a whole number, not a truth value."""
    turns = record["turns"]
    counts = {"n_turns": len(turns), "n_malformed": sum(1 for turn in turns if not turn["parse_ok"])}
    found = [("counters", f"{key}: saved {record[key]!r}, the saved turns give {value}")
             for key, value in counts.items() if record[key] != value]
    try:
        cfg = load_script("replay_check").config_of(record)
    except TypeError as error:
        return found + [("settings", f"the settings of the record are not fields of the config: {error}")]
    answers = {}
    for version in VERSIONS:
        function = impl.main_version("reward") if version == "main" else impl.second_version("reward")
        answers[version] = function(record["passed"], counts["n_malformed"], counts["n_turns"],
                                    record["turn_budget"], cfg)
    saved = record["reward"]
    if any(answer != saved or type(answer) is not type(saved) for answer in answers.values()):
        again = ", ".join(f"{version} {answer!r}" for version, answer in answers.items())
        found.append(("reward", f"saved {saved!r}; the reward functions give: {again}"))
    return found


# ---------------------------------------------------------------------------
# One step: questions 3 and 4
# ---------------------------------------------------------------------------
def group_rule(rewards: list[int], group_size: int) -> tuple[bool, bool]:
    """(the valid rewards of a group are not all equal, at least half of the group's attempts are valid),
    in plain whole numbers. `rewards` are the saved rewards of the group's valid attempts."""
    return len(set(rewards)) > 1, 2 * len(rewards) >= group_size


def group_differences(records: list[dict], keep: list[bool],
                      group_size: int) -> tuple[list[Difference], dict]:
    """Question 4 for the groups of one step: (the differences, the counts).

    `records` are in group order and `keep` says which groups were kept. Counts: `groups`, `kept_groups`,
    `dropped_equal_rewards`, `dropped_lost_attempts` (fewer than half of the attempts are valid)."""
    found: list[Difference] = []
    counts = {"groups": len(keep), "kept_groups": 0, "dropped_equal_rewards": 0, "dropped_lost_attempts": 0}
    for group, kept in enumerate(keep):
        members = records[group * group_size:(group + 1) * group_size]
        rewards = [record["reward"] for record in members if record["harness_error"] is None]
        unequal, enough = group_rule(rewards, group_size)
        name = f"group {group} ({members[0]['scenario_id']}, rewards {sorted(rewards)})"
        if kept:
            counts["kept_groups"] += 1
            if not unequal:
                found.append(("kept_group_with_equal_rewards", f"{name} was kept"))
            if not enough:
                found.append(("kept_group_with_lost_attempts", f"{name} was kept with {len(rewards)} valid "
                                                               f"attempts of {group_size}"))
        elif unequal and enough:
            found.append(("dropped_group_with_unequal_rewards", f"{name} was dropped"))
        else:
            counts["dropped_lost_attempts" if not enough else "dropped_equal_rewards"] += 1
    return found, counts


def advantages_again(records: list[dict], gcfg: dict) -> tuple[np.ndarray, np.ndarray, list[Difference]]:
    """(advantages, keep, differences): the advantages of one step, made again from the saved rewards.

    `reward_matrix` of the trainer puts the rewards into groups, and the main version of
    `group_advantages` gets them with the run's `tie_tol`, as in `plan_update` of the trainer. The second
    version must give the same. Raises ValueError if the records are not the groups of one step."""
    rewards = grpo.reward_matrix(records, gcfg["n_scenarios"], gcfg["group_size"])
    advantages, keep = impl.main_version("group_advantages")(rewards, gcfg["tie_tol"])
    second, second_keep = impl.second_version("group_advantages")(rewards, gcfg["tie_tol"])
    same = (np.array_equal(np.asarray(keep, dtype=bool), np.asarray(second_keep, dtype=bool))
            and np.allclose(advantages, second, rtol=0.0, atol=FLOAT_SLACK))
    found = [] if same else [("advantages", "the main and the second version of group_advantages give "
                                            "other advantages or other kept groups for the saved rewards")]
    return np.asarray(advantages, dtype=float), np.asarray(keep, dtype=bool), found


def selected_of(records: list[dict], keep: np.ndarray, group_size: int) -> list[int]:
    """The positions of the attempts that the update trains on: the valid attempts of the kept groups
    (`selected` of `plan_update` in the trainer)."""
    return [index for index, record in enumerate(records)
            if keep[index // group_size] and record["harness_error"] is None]


def loss_again(advantages: np.ndarray, token_lines: list[dict], selected: list[int],
               norm_const: float) -> float:
    """The GRPO loss of one step from the advantages and the log-probabilities that the SAMPLER saved:
    -(1 / norm_const) * sum over the selected attempts of (advantage * sum of its log-probabilities)."""
    flat = advantages.reshape(-1)
    total = sum(float(flat[index]) * float(sum(token_lines[index]["logprobs"])) for index in selected)
    return -total / norm_const


def loss_check(log: dict, advantages: np.ndarray, token_lines: list[dict], selected: list[int]) -> dict:
    """The numbers of the loss comparison of one step. `log` must hold a loss.

        logged_loss, loss_again   the loss of the log, and the loss made again (`loss_again`)
        difference                the absolute difference of the two
        bound                     how far they may differ: bound_from_gap + bound_from_rounding
        bound_from_gap            max|advantage| * gap.n * gap.mean / norm_const: the logged loss used
                                  the trainer's log-probabilities, the audit has the sampler's
        bound_from_rounding       LOSS_ROUNDING of the size of the sums, over norm_const: the rounding
                                  of the trainer's 32-bit sums
        gap_n, gap_mean, max_abs_advantage, norm_const    what the bound was made from
        n_trained_attempts        the attempts that the loss was made from; 0 if no group was kept (the
                                  loss is then 0, and there is no advantage that could be wrong)
        within_bound              True if difference <= bound: this is what the audit compares
        bound_over_loss           bound / |loss_again|: the bound as a share of the loss itself; None if
                                  the loss made again is 0
        sign_flip_sure_to_show    True if |loss_again| > bound. Advantages with the opposite sign give a
                                  logged loss near -loss_again, so the difference is then 2 * |loss_again|
                                  give or take the bound: it is sure to be above the bound only in this
                                  case. False: such a loss could pass at this step
    """
    gap = log.get("logprob_gap") or {"n": 0, "mean": 0.0}       # no gap: no attempt was trained on
    again = loss_again(advantages, token_lines, selected, log["norm_const"])
    flat = np.abs(advantages).reshape(-1)
    size = sum(float(flat[index]) * sum(abs(value) for value in token_lines[index]["logprobs"])
               for index in selected)
    from_gap = float(flat.max()) * gap["n"] * gap["mean"] / log["norm_const"]
    from_rounding = LOSS_ROUNDING * size / log["norm_const"]
    bound = (float(flat.max()) * gap["n"] * gap["mean"] + LOSS_ROUNDING * size) / log["norm_const"]
    difference = abs(again - log["loss"])
    return {"logged_loss": log["loss"], "loss_again": again, "difference": difference, "bound": bound,
            "bound_from_gap": from_gap, "bound_from_rounding": from_rounding,
            "gap_n": gap["n"], "gap_mean": gap["mean"], "max_abs_advantage": float(flat.max()),
            "norm_const": log["norm_const"], "n_trained_attempts": len(selected),
            "within_bound": bool(difference <= bound),
            "bound_over_loss": bound / abs(again) if again != 0 else None,
            "sign_flip_sure_to_show": bool(abs(again) > bound)}


def loss_differences(log: dict, advantages: np.ndarray, token_lines: list[dict],
                     selected: list[int]) -> list[Difference]:
    """The logged loss against the loss made again (`loss_again`). They may differ by the gap between
    the sampler's and the trainer's log-probabilities (question 3 of the module text) and by the
    rounding of the trainer's 32-bit sums (LOSS_ROUNDING), and by no more. This is a weak check: see
    `loss_check` for the bound and WHAT_THE_CHECKS_SEE for what it cannot see."""
    if log.get("loss") is None:
        return []
    check = loss_check(log, advantages, token_lines, selected)
    if check["within_bound"]:
        return []
    return [("log_loss", f"the log has the loss {log['loss']!r}; the saved rewards and the sampler's "
                         f"log-probabilities give {check['loss_again']!r}; they may differ by "
                         f"{check['bound']:.3g} at most")]


def log_differences(log: dict, records: list[dict], keep: np.ndarray,
                    selected: list[int]) -> list[Difference]:
    """What the log line of a step holds, against the saved rollout and the advantages made again.

    After a fault the trainer made no update: `n_trained_episodes` and `n_loss_tokens` are then 0, or
    the numbers of the selected attempts if the fault was found after the training forward."""
    group_size = len(records) // len(keep)
    valid = [record for record in records if record["harness_error"] is None]
    wanted = {"scenarios": [records[group * group_size]["scenario_id"] for group in range(len(keep))],
              "n_episodes": len(records), "n_harness_errors": len(records) - len(valid),
              "kept_groups": int(keep.sum())}
    found = [(f"log_{key}", f"the log has {log.get(key)!r}; the saved rollout gives {value!r}")
             for key, value in wanted.items() if log.get(key) != value]
    if valid:
        mean_reward = sum(record["reward"] for record in valid) / (grpo.REWARD_SCALE * len(valid))
        if log.get("mean_reward") is None or abs(log["mean_reward"] - mean_reward) > FLOAT_SLACK:
            found.append(("log_mean_reward", f"the log has {log.get('mean_reward')!r}; the saved rewards "
                                             f"give {mean_reward!r}"))
    trained = {"n_trained_episodes": len(selected),
               "n_loss_tokens": sum(turn["n_tokens"] or 0 for index in selected
                                    for turn in records[index]["turns"])}
    allowed_after_fault = (0,) if log.get("fault") is not None else ()
    for key, value in trained.items():
        if log.get(key) != value and log.get(key) not in allowed_after_fault:
            found.append((f"log_{key}", f"the log has {log.get(key)!r}; the kept groups of the saved "
                                        f"rollout give {value!r}"))
    return found


def audit_step(records: list[dict], gcfg: dict, log: dict | None = None,
               token_lines: list[dict] | None = None) -> tuple[list[Difference], dict]:
    """Every difference of one saved rollout, and its counts.

    records      the records of one `step_*.episodes.jsonl`, in file order
    gcfg         `n_scenarios`, `group_size` and `tie_tol` of the run
    log          the log line of the step, or None if the log has none (then nothing is compared with it)
    token_lines  the lines of the step's token file, or None (then the loss is not made again)
    Counts: attempts, harness_errors, failed_attempts, rewards_checked, the counts of
    `group_differences`, and log_lines_compared, loss_checks (0 or 1 each)."""
    valid = [record for record in records if record["harness_error"] is None]
    found: list[Difference] = []
    for record in valid:
        name = f"{record['scenario_id']} sample {record['sample']}"
        found += [(kind, f"{name}: {detail}")
                  for kind, detail in sign_differences(record) + reward_differences(record)]
    counts = {"attempts": len(records), "harness_errors": len(records) - len(valid),
              "failed_attempts": sum(1 for record in valid if not record["passed"]),
              "rewards_checked": len(valid), "groups": 0, "kept_groups": 0, "dropped_equal_rewards": 0,
              "dropped_lost_attempts": 0, "log_lines_compared": 0, "loss_checks": 0}
    try:
        advantages, keep, differ = advantages_again(records, gcfg)
    except ValueError as error:
        return found + [("groups", f"the records are not the groups of one step: {error}")], counts
    selected = selected_of(records, keep, gcfg["group_size"])
    group_found, group_counts = group_differences(records, [bool(kept) for kept in keep], gcfg["group_size"])
    found += differ + group_found
    counts.update(group_counts)
    if log is not None:
        counts["log_lines_compared"] = 1
        found += log_differences(log, records, keep, selected)
        if token_lines is not None and log.get("loss") is not None:
            if [(line["scenario_id"], line["sample"]) for line in token_lines] != \
                    [(record["scenario_id"], record["sample"]) for record in records]:
                found.append(("tokens", "the token file does not hold the attempts of the episodes file, "
                                        "in the same order"))
            else:
                counts["loss_checks"] = 1
                found += loss_differences(log, advantages, token_lines, selected)
    return found, counts


def step_loss_check(records: list[dict], gcfg: dict, log: dict | None,
                    token_lines: list[dict] | None) -> dict | None:
    """The `loss_check` of one step, or None if `audit_step` compares no loss there: the step has no log
    line, no loss in it, or no token file, the token file holds other attempts, or the records are not
    the groups of one step. It makes the advantages again, as `audit_step` does."""
    if log is None or token_lines is None or log.get("loss") is None:
        return None
    if [(line["scenario_id"], line["sample"]) for line in token_lines] != \
            [(record["scenario_id"], record["sample"]) for record in records]:
        return None
    try:
        advantages, keep, _ = advantages_again(records, gcfg)
    except ValueError:
        return None
    return loss_check(log, advantages, token_lines, selected_of(records, keep, gcfg["group_size"]))


# ---------------------------------------------------------------------------
# The whole run folder
# ---------------------------------------------------------------------------
def run_config(folder: Path) -> tuple[dict, str]:
    """(`n_scenarios`, `group_size` and `tie_tol` of the run, the file they were read from): from
    `grpo_config` in run_manifest.json, else in latest.json. Raises ValueError if neither has them."""
    for name in CONFIG_FILES:
        path = folder / name
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8")).get("grpo_config") or {}
            if all(key in config for key in CONFIG_KEYS):
                return {key: config[key] for key in CONFIG_KEYS}, name
    raise ValueError(f"{folder} has no grpo_config with {CONFIG_KEYS} in {' or '.join(CONFIG_FILES)}")


def log_lines(path: Path) -> dict[int, dict]:
    """step -> the log line of the step, from one log file; an empty dict without the file. Of two lines
    of one step the last one stays. A line that is not complete JSON is left out."""
    lines: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            log = json.loads(line)
        except ValueError:
            continue
        if isinstance(log, dict) and isinstance(log.get("step"), int):
            lines[log["step"]] = log
    return lines


def saved_rollouts(folder: Path) -> list[tuple[Path, int, Path]]:
    """(episodes file, step, the log file that holds its line) for every saved rollout of a run folder:
    `rollouts/step_*.episodes.jsonl` with `train_log.jsonl`, then `rollouts/start_<n>/` with
    `train_log.start_<n>.jsonl`, in name order."""
    found = []
    places = [(folder / "rollouts", folder / grpo.LOG_NAME)]
    for start in sorted(path for path in (folder / "rollouts").glob("start_*") if path.is_dir()):
        match = START_FOLDER.fullmatch(start.name)
        if match is not None:
            places.append((start, folder / f"train_log.start_{match.group(1)}.jsonl"))
    for place, log_file in places:
        for path in sorted(place.glob("step_*.episodes.jsonl")):
            match = EPISODES_FILE.fullmatch(path.name)
            if match is not None:
                found.append((path, int(match.group(1)), log_file))
    return found


def audit_run(folder: Path) -> dict:
    """The formula audit of one GRPO run folder: the content of the output JSON.

        run_folder, grpo_config, grpo_config_from   what was audited, with which group settings
        log_keys, log_holds_advantages     the keys of the log lines; whether `advantages` is one of
                           them (if it ever is, this audit does not read it yet)
        log_compared       the keys of the log that the audit compares (see the module text)
        counts             sums over all saved rollouts (`audit_step`), and: rollout_files,
                           steps_without_log_line, steps_without_token_file, log_lines_without_rollout,
                           failed_with_positive_reward, loss_checks_of_steps_that_trained (loss
                           checks of steps with a trained attempt),
                           loss_checks_where_a_sign_flip_is_sure_to_show
        n_differences, by_kind, first_differences   every difference counted; the first SHOWN in full,
                           each with `file`, `step`, `kind`, `detail`
        loss_checks        one entry for each step whose loss was compared: `file`, `step`, and the
                           numbers of `loss_check` (the two losses, their difference and the bound)
        loss_bound         largest: the largest bound of a step, and largest_at_step; and
                           largest_share_of_the_loss: the largest bound_over_loss of a step. Each is
                           None if no loss was compared
        what_the_checks_see    WHAT_THE_CHECKS_SEE: one paragraph in plain words
    Raises ValueError if the folder has no saved rollout or no GRPO config."""
    gcfg, source = run_config(folder)
    rollouts = saved_rollouts(folder)
    if not rollouts:
        raise ValueError(f"{folder} has no saved rollout (rollouts/step_*.episodes.jsonl)")
    counts: dict[str, int] = {"rollout_files": len(rollouts), "steps_without_log_line": 0,
                              "steps_without_token_file": 0, "log_lines_without_rollout": 0}
    differences: list[dict] = []
    loss_checks: list[dict] = []
    logs: dict[Path, dict[int, dict]] = {}
    for path, step, log_file in rollouts:
        log = logs.setdefault(log_file, log_lines(log_file)).get(step)
        token_file = path.with_name(path.name.replace(".episodes.", ".tokens."))
        token_lines = read_records(token_file) if token_file.is_file() else None
        counts["steps_without_log_line"] += log is None
        counts["steps_without_token_file"] += token_lines is None
        records = read_records(path)
        found, step_counts = audit_step(records, gcfg, log, token_lines)
        for key, value in step_counts.items():
            counts[key] = counts.get(key, 0) + value
        if step_counts["loss_checks"]:
            loss_checks.append({"file": path.relative_to(folder).as_posix(), "step": step,
                                **step_loss_check(records, gcfg, log, token_lines)})
        differences += [{"file": path.relative_to(folder).as_posix(), "step": step, "kind": kind,
                         "detail": detail} for kind, detail in found]
    audited = {step for path, step, log_file in rollouts if log_file == folder / grpo.LOG_NAME}
    for step in sorted(set(logs.get(folder / grpo.LOG_NAME, {})) - audited):
        counts["log_lines_without_rollout"] += 1
        differences.append({"file": grpo.LOG_NAME, "step": step, "kind": "missing_rollout",
                            "detail": "the log has a line for this step, but no episodes file is saved"})
    by_kind: dict[str, int] = {}
    for difference in differences:
        by_kind[difference["kind"]] = by_kind.get(difference["kind"], 0) + 1
    counts["failed_with_positive_reward"] = by_kind.get("fail_with_positive_reward", 0)
    counts["loss_checks_of_steps_that_trained"] = sum(
        1 for check in loss_checks if check["n_trained_attempts"] > 0)
    counts["loss_checks_where_a_sign_flip_is_sure_to_show"] = sum(
        1 for check in loss_checks if check["sign_flip_sure_to_show"])
    widest = max(loss_checks, key=lambda check: check["bound"], default=None)
    shares = [check["bound_over_loss"] for check in loss_checks if check["bound_over_loss"] is not None]
    log_keys = sorted({key for lines in logs.values() for log in lines.values() for key in log})
    return {"run_folder": str(folder), "grpo_config": gcfg, "grpo_config_from": source,
            "log_keys": log_keys, "log_holds_advantages": "advantages" in log_keys,
            "log_compared": list(LOG_COMPARED),
            "counts": {key: int(value) for key, value in counts.items()},
            "n_differences": len(differences), "by_kind": by_kind, "first_differences": differences[:SHOWN],
            "loss_bound": {"largest": widest["bound"] if widest else None,
                           "largest_at_step": widest["step"] if widest else None,
                           "largest_share_of_the_loss": max(shares, default=None)},
            "loss_checks": loss_checks, "what_the_checks_see": WHAT_THE_CHECKS_SEE}


def summary_lines(report: dict) -> list[str]:
    """The report of `audit_run` in a few plain lines, for the end of the printed output: the number of
    differences, how weak the loss comparison was in this run, and WHAT_THE_CHECKS_SEE."""
    counts, bound = report["counts"], report["loss_bound"]
    found = f"{report['n_differences']} differences" if report["n_differences"] else "no difference"
    kinds = "".join(f"; {kind} {count}" for kind, count in report["by_kind"].items())
    lines = [f"formula audit of {report['run_folder']}: {found} in {counts['rollout_files']} saved rollouts "
             f"({counts['attempts']} attempts, {counts['failed_attempts']} failed, "
             f"{counts['failed_with_positive_reward']} failed with a reward above 0){kinds}"]
    sure, compared = counts["loss_checks_where_a_sign_flip_is_sure_to_show"], counts["loss_checks"]
    trained = counts["loss_checks_of_steps_that_trained"]
    if compared:
        share = bound["largest_share_of_the_loss"]
        lines += [f"the loss comparison is a WEAK check: {compared} steps compared, {trained} of them with "
                  f"a trained attempt; in {sure} of these {trained} the bound is smaller than the loss made "
                  "again, so advantages with the opposite sign would be sure to show; in the other "
                  f"{trained - sure} they could pass",
                  f"the largest bound of a step: {bound['largest']:.3g} (step {bound['largest_at_step']}); "
                  "the largest bound as a share of the loss made again: "
                  + ("no step has a loss other than 0" if share is None else f"{share:.3g}")]
    else:
        lines.append("the loss comparison is a WEAK check, and in this run no loss was compared at all "
                     f"({counts['steps_without_token_file']} steps without a token file, "
                     f"{counts['steps_without_log_line']} without a log line)")
    return lines + [report["what_the_checks_see"]]
