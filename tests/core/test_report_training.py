"""scripts/training_report.py: the tables about the GRPO runs.

The runs, the scenario file, the ledger and the other run manifests are small and made up, and they are
written into a temporary folder: every expected number below is worked out by hand in a comment. The log
lines of a made-up run are made by the trainer's own functions (`plan_update`, `rollout_stats`), as
`grpo_step` makes them. One test at the end runs the script on the real saved runs if they are there; it
writes into a temporary folder too. Nothing is written into the repository.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from accessdesk.config import DEFAULT
from accessdesk.harness.records import write_records
from accessdesk.train import grpo
from tests.helpers.make_records import make_record, scenario_records

ROOT = Path(__file__).resolve().parents[2]
TOKENS_PER_TURN = 10
GCFG = grpo.GRPOConfig(n_scenarios=3, group_size=4, reference_tokens=10, steps=4, seed=1)
FREEZE = "2026-01-02T00:00:00Z"

# The made-up train split: (family id, expected outcome, (active, holds_role, sensitive), two people of one
# name). Shortest correct path: grant 4 turns, approval 4, already 3, inactive 3 (tests/helpers).
FAMILIES = (
    ("tr-000", "grant", (True, False, False), False),
    ("tr-001", "approval", (True, False, True), True),
    ("tr-002", "already", (True, True, False), False),
    ("tr-003", "already", (True, True, True), True),
    ("tr-004", "inactive", (False, False, False), False),
    ("tr-005", "inactive", (False, True, False), True),
)
OUTCOME_OF = {family_id: outcome for family_id, outcome, _, _ in FAMILIES}

# The made-up run: per step three groups, each (family id, wording style, the rewards of its 4 attempts).
# A reward of 95 is a pass with one malformed message; 88 is a pass with two (10 points) and one turn over
# the budget (2 points); 0 is a fail. "lost" is an attempt that the harness lost.
#   step 0   dropped (all pass)      kept: 2 passes, 2 fails    dropped (all fail)
#   step 1   kept: penalty-only      dropped (all pass)         kept: 3 passes, 1 fail
#   step 2   dropped                 dropped                    dropped (all fail)       no kept group
#   step 3   kept: penalty-only      dropped                    dropped
ALL_PASS, ALL_FAIL = (100, 100, 100, 100), (0, 0, 0, 0)
STEPS = (
    (("tr-000", 1, ALL_PASS), ("tr-001", 1, (100, 0, 0, 100)), ("tr-004", 1, ALL_FAIL)),
    (("tr-002", 1, (100, 95, 100, 100)), ("tr-003", 2, ALL_PASS), ("tr-005", 1, (0, 100, 100, 100))),
    (("tr-000", 2, ALL_PASS), ("tr-001", 2, ALL_PASS), ("tr-004", 2, ALL_FAIL)),
    (("tr-002", 2, (100, 100, 88, 100)), ("tr-003", 1, ALL_PASS), ("tr-005", 2, ALL_PASS)),
)
# A second run for the tables with two runs: one kept group with passes and fails, one penalty-only.
OTHER_STEPS = (
    (("tr-001", 1, (0, 100, 100, 100)), ("tr-002", 1, ALL_PASS), ("tr-004", 1, ALL_PASS)),
    (("tr-003", 1, (100, 100, 100, 95)), ("tr-000", 1, ALL_PASS), ("tr-005", 1, ALL_PASS)),
)

LEDGER_HEADER = ("time_utc,event,instance_id,offer_id,gpu,dph_usd,max_minutes,bandwidth_est_usd,"
                 "projected_usd,actual_usd,note\n")
USUAL = "h x price + download estimate; an estimate, the invoice is the authority."
# The made-up ledger. The freeze is at 2026-01-02T00:00:00Z.
#   E  rented 9 hours by the clock, but the amount is the projected limit: (2.1 - 0.1) / 0.5 = 4 hours
#   A  10 hours at 0.5 plus 0.1 for downloads = 5.1, before the freeze
#   B  16 hours at 1.0 = 16.0, across the freeze: 4 hours (4.0 USD) before it, 12 hours (12.0 USD) after it
#   C  10 hours at 0.4 = 4.0, after the freeze (and after the end of the made-up GRPO run)
#   D  rented and not stopped: in no amount
# Before the freeze: 4 + 10 + 4 = 18 hours, 2.1 + 5.1 + 4.0 = 11.2 USD. After: 12 + 10 = 22 hours, 16.0 USD.
# All closed rentals: 40 hours, 27.2 USD. Price after the freeze: 16 / 22 USD per hour.
LEDGER_ROWS = (
    "2025-12-31T00:00:00Z,rent,E,1,RTX 4090,0.5,240,0.1,2.1,,label e\n"
    "2025-12-31T09:00:00Z,destroy,E,1,RTX 4090,,,0.1,,2.1,already gone; cost = the projected upper limit.\n"
    "2026-01-01T00:00:00Z,rent,A,1,RTX 4090,0.5,900,0.1,7.6,,label a\n"
    f"2026-01-01T10:00:00Z,destroy,A,1,RTX 4090,0.5,,0.1,,5.1,\"10.000 {USUAL}\"\n"
    "2026-01-01T20:00:00Z,rent,B,1,RTX 4090,1.0,1200,0.0,20.0,,label b\n"
    "2026-01-02T06:00:00Z,rent,C,1,RTX 4090,0.4,900,0.0,6.0,,label c\n"
    f"2026-01-02T12:00:00Z,destroy,B,1,RTX 4090,1.0,,0.0,,16.0,\"16.000 {USUAL} destroy-all\"\n"
    f"2026-01-02T16:00:00Z,destroy,C,1,RTX 4090,0.4,,0.0,,4.0,\"10.000 {USUAL}\"\n"
    "2026-01-02T20:00:00Z,rent,D,1,RTX 4090,0.4,900,0.0,6.0,,label d\n"
)
PRICE = 16.0 / 22.0


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script("training_report")


# ---------------------------------------------------------------------------
# Builders of the made-up files
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_scenarios(folder: Path, with_facts: bool = True) -> str:
    """train.jsonl of the made-up families; returns its SHA-256. The folder also gets a dev and a test file
    that are not JSON: the script must never read them."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for family_id, outcome, facts, same_name in FAMILIES:
        family = {"family_id": family_id, "split": "train", "outcome": outcome, "homonym": same_name}
        if with_facts:
            family["facts"] = dict(zip(("active", "holds_role", "sensitive"), facts))
        lines.append(json.dumps(family) + "\n")
    (folder / "train.jsonl").write_text("".join(lines), encoding="utf-8")
    for name in ("dev.jsonl", "test.jsonl"):
        (folder / name).write_text("not JSON: this file must not be read\n", encoding="utf-8")
    return sha256(folder / "train.jsonl")


def group_records(family_id: str, style: int, rewards: tuple, data_sha256: str) -> list[dict]:
    """The attempts of one group, each turn with 10 sampled tokens."""
    records = []
    for sample, reward in enumerate(rewards):
        if reward == "lost":
            records.append(make_record(family_id, style, sample, outcome=OUTCOME_OF[family_id], arm=grpo.ARM,
                                       harness_error="made-up harness error"))
        else:
            records.append(scenario_records(family_id, style, [reward], outcome=OUTCOME_OF[family_id],
                                            arm=grpo.ARM)[0])
            records[-1]["sample"] = sample
    for record in records:
        record["data_sha256"] = data_sha256
        for turn in record["turns"]:
            turn["n_tokens"] = TOKENS_PER_TURN
    return records


def step_records(groups: tuple, data_sha256: str) -> list[dict]:
    return [record for family_id, style, rewards in groups
            for record in group_records(family_id, style, rewards, data_sha256)]


def log_line(step: int, records: list[dict], gcfg: grpo.GRPOConfig = GCFG, **changes) -> dict:
    """The log line that `grpo_step` writes for the records of one step, without a model: the plan and the
    statistics come from the trainer's functions; the loss, the gradient norm and the gap are made up (0
    when no group is kept, as in the trainer). A step takes 12 seconds, 10 of them sampling."""
    plan = grpo.plan_update(records, gcfg)
    with_tokens = [{**record, "tokens": {"logprobs": [-0.5] * len(record["turns"])}} for record in records]
    selected = plan["selected"]
    n_tokens = sum(turn["n_tokens"] for index in selected for turn in records[index]["turns"])
    kept = int(plan["keep"].sum())
    first_of_group = range(0, len(records), gcfg.group_size)
    log = {"step": step, "scenarios": [records[index]["scenario_id"] for index in first_of_group],
           **grpo.rollout_stats(with_tokens), "kept_groups": kept, "kept_share": kept / gcfg.n_scenarios,
           "norm_const": gcfg.norm_const, "lr": gcfg.lr, "loss": 0.001 * kept,
           "n_trained_episodes": len(selected), "n_loss_tokens": n_tokens,
           "logprob_gap": {"n": n_tokens, "mean": 0.001 * (step + 1), "p99": 0.002, "max": 0.003}
           if selected else None,
           "grad_norm": 0.01 * kept, "fault": None, "rollout_seconds": 10.0, "seconds": 12.0,
           "cost_usd": None, "start": 1}
    return {**log, **changes}


def write_run(folder: Path, steps: tuple, data_sha256: str, *, gcfg: grpo.GRPOConfig = GCFG,
              log_changes: dict[int, dict] | None = None, **manifest_changes) -> Path:
    """A run folder as the trainer leaves it: rollouts/, train_log.jsonl, run_manifest.json. The run
    manifest holds the totals of the log, 6 seconds more than the steps, and the hash of the log."""
    (folder / "rollouts").mkdir(parents=True)
    logs = []
    for step, groups in enumerate(steps):
        records = step_records(groups, data_sha256)
        write_records(folder / "rollouts" / f"step_{step:05d}.episodes.jsonl", records)
        # A token file that is not JSON: the script must not read it.
        (folder / "rollouts" / f"step_{step:05d}.tokens.jsonl").write_text("not JSON\n", encoding="utf-8")
        logs.append(log_line(step, records, gcfg, **(log_changes or {}).get(step, {})))
    (folder / "train_log.jsonl").write_text("".join(json.dumps(log) + "\n" for log in logs), encoding="utf-8")
    manifest = {
        "complete": True, "status": "complete", "mode": "seed", "script": "scripts/gpu/train_grpo.py",
        "seed": 1,
        "started_utc": "2026-01-02T01:00:00Z", "ended_utc": "2026-01-02T02:00:00Z",
        "seconds": sum(log["seconds"] for log in logs) + 6.0,
        "grpo_config": gcfg.to_dict(), "norm_const": gcfg.norm_const, "config": DEFAULT.to_dict(),
        "split": "train", "split_file_sha256": data_sha256, "families": 6, "scenarios": 12, "styles": [1, 2],
        "start_adapter": {"folder": "made-up/sft/adapter"}, "starts": [{"result": "complete"}],
        "steps_run_again": {"lines": 0, "steps": [], "episodes": 0, "harness_errors": 0},
        "frozen": {"matches": True, "manifest_sha256": "f" * 64},
        "summary": {"steps": len(logs), "groups": len(logs) * gcfg.n_scenarios,
                    "kept_groups": sum(log["kept_groups"] for log in logs),
                    "episodes": sum(log["n_episodes"] for log in logs),
                    "harness_errors": sum(log["n_harness_errors"] for log in logs),
                    "trained_episodes": sum(log["n_trained_episodes"] for log in logs),
                    "loss_tokens": sum(log["n_loss_tokens"] for log in logs)},
        "outputs": {"log": {"sha256": sha256(folder / "train_log.jsonl")}},
    }
    write_json(folder / "run_manifest.json", {**manifest, **manifest_changes})
    return folder


def rollout_manifest(folder: Path, seconds: float, attempts: int, **changes) -> None:
    """The run manifest of a rollout, with only what the cost table reads."""
    write_json(folder / "run_manifest.json", {
        "complete": True, "command": ["scripts/gpu/rollout.py"], "seconds": seconds,
        "summary": {"episodes": attempts}, "frozen": {"matches": True, "manifest_sha256": "f" * 64},
        **changes})


class Tree:
    """The made-up files of one test: the paths, and the options that give them to the script."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.scenarios = root / "scenarios"
        self.main = root / "main"
        self.calibration = root / "calibration"
        self.ledger = root / "ledger.csv"
        self.smoke = root / "smoke"
        self.freeze = root / "MANIFEST.json"
        self.out = root / "out"
        self.run = self.main / "grpo_seed1"
        self.data_sha256 = ""

    def options(self, *more: str, runs: tuple[tuple[str, Path], ...] | None = None) -> list[str]:
        named = runs if runs is not None else (("seed1", self.run),)
        given = [option for name, folder in named for option in ("--run", name, str(folder))]
        return [*given, "--main", str(self.main), "--calibration", str(self.calibration),
                "--ledger", str(self.ledger), "--smoke", str(self.smoke), "--scenarios", str(self.scenarios),
                "--freeze-manifest", str(self.freeze), "--out", str(self.out), *more]

    def report(self) -> dict:
        return json.loads((self.out / "training_report.json").read_text(encoding="utf-8"))

    def text(self, name: str) -> str:
        return (self.out / name).read_text(encoding="utf-8")


def make_tree(root: Path, steps: tuple = STEPS, **run_options) -> Tree:
    """Everything that the script reads, made up. The seconds of the other runs:

        Stage 1   base_dev 720 (96 attempts), base_train 1440 (192)                 2160 s, 288 attempts
                  sft_data 2, sft 358 (no attempt sampled)                           360 s
                  sft_dev 720 (96)                                                   720 s, 96 attempts
        Checks    pilot 180 (36); layout gradients 90 + 90                           360 s, 36 attempts
        Stage 2   the run 4 x 12 + 6 = 54 (48); dev 360 (96); curve 2 x 36 (2 x 12);
                  second desk 72 (24) + 36 (12)                                      594 s, 204 attempts
        other     odd_run 60 (5); `unfinished` is not complete and is not counted    60 s, 5 attempts
        All run manifests after the freeze: 4254 s = 1.1816... hours.
        calibration   round1/2b 7200 (50 attempts)
    """
    tree = Tree(root)
    tree.data_sha256 = write_scenarios(tree.scenarios)
    write_run(tree.run, steps, tree.data_sha256, **run_options)
    rollouts = (("base_dev", 720.0, 96), ("base_train", 1440.0, 192), ("sft_dev", 720.0, 96),
                ("grpo_seed1_dev", 360.0, 96), ("grpo_seed1_curve_000", 36.0, 12),
                ("grpo_seed1_curve_020", 36.0, 12), ("grpo_seed1_shift_differ", 72.0, 24),
                ("grpo_seed1_shift_control", 36.0, 12), ("checks/pilot", 180.0, 36), ("odd_run", 60.0, 5))
    for name, seconds, attempts in rollouts:
        rollout_manifest(tree.main / name, seconds, attempts)
    frozen = {"matches": True, "manifest_sha256": "f" * 64}
    write_json(tree.main / "sft_data" / "run_manifest.json",
               {"complete": True, "script": "scripts/gpu/sft_data.py", "seconds": 2.0, "frozen": frozen})
    write_json(tree.main / "sft" / "run_manifest.json",
               {"complete": True, "script": "scripts/gpu/train_sft.py", "seconds": 358.0,
                "summary": {"steps": 5}, "frozen": frozen})
    write_json(tree.main / "unfinished" / "run_manifest.json", {"complete": False, "seconds": None})
    write_json(tree.main / "checks" / "layout_gradients" / "layout_gradients.json",
               {"complete": True, "stages": {"bfloat16": {"seconds": 90.0}, "float32": {"seconds": 90.0}},
                "code": {"frozen": frozen}})
    rollout_manifest(tree.calibration / "round1" / "2b", 7200.0, 50, frozen=None)
    tree.ledger.write_text(LEDGER_HEADER + LEDGER_ROWS, encoding="utf-8")
    write_json(tree.freeze, {"frozen_at": FREEZE})
    return tree


@pytest.fixture()
def tree(tmp_path) -> Tree:
    return make_tree(tmp_path)


@pytest.fixture()
def made(tree, script, capsys) -> Tree:
    """The made-up tree after one run of the script, with blocks of 2 steps."""
    assert script.main(tree.options("--block-steps", "2")) == 0
    assert "wrote 6 files" in capsys.readouterr().out
    return tree


@pytest.fixture()
def reads(monkeypatch) -> list[Path]:
    """Every path that is read through pathlib while the test runs."""
    seen: list[Path] = []
    for name in ("read_bytes", "read_text", "open"):
        original = getattr(Path, name)

        def recorded(self, *args, _original=original, **kwargs):
            seen.append(Path(self))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, name, recorded)
    return seen


# ---------------------------------------------------------------------------
# summary.md, the step files and the totals
# ---------------------------------------------------------------------------
def test_the_totals_of_the_made_up_run_are_the_numbers_worked_out_by_hand(made):
    run = made.report()["runs"]["seed1"]
    totals = run["totals"]
    # 4 steps x 3 groups x 4 attempts. Failed attempts: 2 + 4 in step 0, 1 in step 1, 4 in step 2.
    assert (totals["steps"], totals["groups"], totals["attempts"], totals["valid"]) == (4, 12, 48, 48)
    # No scenario comes twice in the made-up run: 12 groups are 12 scenarios.
    assert totals["different_scenarios"] == 12
    assert totals["first_step_with_a_scenario_of_an_earlier_step"] is None
    assert (totals["failed"], totals["passed"], totals["harness_errors"]) == (11, 37, 0)
    assert totals["failed_share_of_valid"] == pytest.approx(11 / 48)
    # Kept: the 2-and-2 group, the 3-and-1 group, and the two penalty-only groups. Dropped: 6 groups in
    # which all attempts pass, and 2 in which all fail.
    assert totals["kept_groups"] == 4 and totals["kept_share_of_groups"] == pytest.approx(4 / 12)
    assert totals["groups_by_type"] == {"pass_and_fail": 2, "penalty_only": 2, "all_fail": 0,
                                        "dropped_all_pass": 6, "dropped_all_fail": 2,
                                        "dropped_lost_attempts": 0, "dropped_other": 0}
    # The frozen function gives the same 8 groups with equal rewards. With the two penalty-only groups,
    # 10 groups have only passes or only fails.
    assert totals["equal_reward_groups"] == {
        "groups_counted": 12, "equal_reward": 8, "equal_reward_all_pass": 6, "equal_reward_all_fail": 2,
        "equal_passfail": 10, "equal_reward_share": pytest.approx(8 / 12),
        "equal_reward_all_pass_share": pytest.approx(0.5),
        "equal_reward_all_fail_share": pytest.approx(2 / 12), "equal_passfail_share": pytest.approx(10 / 12)}
    assert (totals["steps_without_kept_group"], totals["steps_without_kept_group_list"]) == (1, [2])
    assert totals["steps_without_kept_group_and_zero_gradient"] == 1
    assert totals["kept_groups_in_steps_with_a_fault"] == 0
    # Trained: the 16 attempts of the 4 kept groups. Rewards: 600 + 1095 + 800 + 1188 = 3683 points.
    assert totals["trained_attempts"] == 16 and totals["trained_share_of_attempts"] == pytest.approx(16 / 48)
    assert totals["mean_reward"] == pytest.approx(3683 / 4800)
    # Turns: 44 + 37 + 44 + 38 = 163, each of 10 tokens. Kept groups: 16 + 13 + 12 + 14 = 55 turns.
    assert (totals["sampled_tokens"], totals["loss_tokens"]) == (1630, 550)
    assert totals["loss_tokens_share_of_sampled_tokens"] == pytest.approx(550 / 1630)
    assert (totals["attempts_in_dropped_groups"], totals["sampled_tokens_in_dropped_groups"]) == (32, 1080)
    assert totals["reward_values"] == {"0": 11, "88": 1, "95": 1, "100": 35}
    # Seconds: 4 steps of 12 (10 of them sampling); the script as a whole 6 more; step 2 kept no group.
    time = run["time"]
    assert (time["run_seconds"], time["step_seconds"], time["sampling_seconds"]) == (54.0, 48.0, 40.0)
    assert time["update_seconds"] == pytest.approx(8.0)
    assert time["seconds_in_steps_without_kept_group"] == 12.0
    assert time["share_of_step_seconds_in_steps_without_kept_group"] == pytest.approx(0.25)
    assert time["sampling_share_of_step_seconds"] == pytest.approx(40 / 48)
    # Every share and every number of hours that a table prints is in the JSON too.
    assert time["hours"] == {"run": pytest.approx(54 / 3600), "step": pytest.approx(48 / 3600),
                             "sampling": pytest.approx(40 / 3600), "update": pytest.approx(8 / 3600),
                             "in_steps_without_kept_group": pytest.approx(12 / 3600)}
    assert totals["steps_without_kept_group_share_of_steps"] == pytest.approx(0.25)
    assert totals["mean_reward_points"] == pytest.approx(3683 / 48)
    assert totals["dropped_share_of_groups"] == pytest.approx(8 / 12)


def test_the_set_up_and_the_stop_rule_numbers_come_from_the_manifest_and_the_log(made):
    run = made.report()["runs"]["seed1"]
    set_up = run["set_up"]
    assert (set_up["scenarios_per_step"], set_up["attempts_per_scenario"], set_up["tie_tol"]) == (3, 4, 0.0)
    assert (set_up["status"], set_up["seed"], set_up["starts"]) == ("complete", 1, 1)
    assert set_up["reward_settings"] == {
        "pass_reward": 100, "malformed_penalty": 5, "malformed_cap": 15, "turn_penalty": 2, "turn_cap": 10,
        "penalise_malformed_on_fail": False}
    stop = run["stop_rule"]
    assert (stop["steps_with_a_fault"], stop["harness_errors"], stop["harness_error_rate"]) == (0, 0, 0.0)
    # The made-up gap is 0.001 x (step + 1), and step 2 has none; the gradient norm is 0.01 x kept groups.
    assert stop["largest_logprob_gap_mean"] == {"value": pytest.approx(0.004), "step": 3}
    assert stop["largest_grad_norm"] == {"value": pytest.approx(0.02), "step": 1}
    assert (stop["clip_grad_norm"], stop["steps_clipped"]) == (1.0, 0)
    assert (stop["limit_logprob_gap_mean"], stop["limit_invalid_call_rate"]) == (0.02, 0.05)
    # Step 1 has one malformed message in 37 assistant messages, step 3 has two in 38.
    assert stop["invalid_call_rate_step_0"] == 0.0
    assert stop["largest_invalid_call_rate"] == {"value": pytest.approx(2 / 38), "step": 3}


def test_the_pass_rate_of_a_block_is_the_frozen_family_weighted_one(tree, script):
    # Blocks of 3 steps. In the steps 0 to 2 the families tr-000, tr-001 and tr-004 have 8 attempts each
    # (8, 6 and 0 passed) and tr-002, tr-003 and tr-005 have 4 (4, 4 and 3 passed). Family-weighted:
    # (1 + 0.75 + 0 + 1 + 1 + 0.75) / 6 = 0.75. Passed attempts of attempts: 25 of 36, another number.
    assert script.main(tree.options("--block-steps", "3")) == 0
    first, last = tree.report()["runs"]["seed1"]["blocks"]
    assert (first["first_step"], first["last_step"], first["steps"]) == (0, 2, 3)
    assert (first["attempts"], first["passed"], first["failed"]) == (36, 25, 11)
    assert first["pass_rate"] == pytest.approx(0.75)
    assert (first["families"], first["scenarios"]) == (6, 9)
    # inactive: tr-004 with 0 of 8 and tr-005 with 3 of 4: (0 + 0.75) / 2.
    assert first["by_outcome"]["inactive"] == {"pass_rate": pytest.approx(0.375), "passed": 3, "valid": 12}
    assert first["by_outcome"]["approval"] == {"pass_rate": pytest.approx(0.75), "passed": 6, "valid": 8}
    assert (first["kept_groups"], first["groups"], first["steps_without_kept_group"]) == (3, 9, 1)
    # The last block is shorter: step 3 alone.
    assert (last["first_step"], last["last_step"], last["steps"], last["attempts"]) == (3, 3, 1, 12)
    assert last["pass_rate"] == 1.0 and last["groups_by_type"]["penalty_only"] == 1
    assert "| 0 to 2 | 36 | 11 (30.56%) | 75.0% |" in tree.text("summary.md")
    assert "37.5% (3 of 12)" in tree.text("summary.md")


def test_the_blocks_of_two_steps_hold_the_counts_and_the_means(made):
    first, second = made.report()["runs"]["seed1"]["blocks"]
    # Steps 0 and 1: 24 attempts, 7 failed; rewards 600 + 1095 points; 44 + 37 turns of 10 tokens.
    assert (first["attempts"], first["failed"], first["kept_groups"], first["groups"]) == (24, 7, 3, 6)
    assert first["mean_reward"] == pytest.approx(1695 / 2400)
    assert first["tokens_per_attempt"] == pytest.approx(810 / 24)
    assert first["groups_by_type"]["pass_and_fail"] == 2 and first["groups_by_type"]["penalty_only"] == 1
    assert first["pass_rate"] == pytest.approx(17 / 24)
    assert first["failed_share_of_valid"] == pytest.approx(7 / 24)
    assert first["kept_share_of_groups"] == pytest.approx(0.5)
    assert first["mean_reward_points"] == pytest.approx(1695 / 24)
    assert (second["failed"], second["kept_groups"], second["steps_without_kept_group"]) == (4, 1, 1)
    assert "## Pass rate in blocks of 2 steps" in made.text("summary.md")


def test_the_step_file_has_one_row_per_step_with_the_named_columns(made, script):
    rows = list(csv.DictReader(io.StringIO(made.text("steps_seed1.csv"))))
    assert list(rows[0]) == [name for name, _, _ in script.STEP_COLUMNS]
    assert [row["step"] for row in rows] == ["0", "1", "2", "3"]
    assert rows[0]["scenarios"] == "tr-000/s1 tr-001/s1 tr-004/s1"
    assert [row["kept_groups"] for row in rows] == ["1", "2", "0", "1"]
    assert [row["kept_pass_and_fail"] for row in rows] == ["1", "1", "0", "0"]
    assert [row["kept_penalty_only"] for row in rows] == ["0", "1", "0", "1"]
    assert [row["failed_attempts"] for row in rows] == ["6", "1", "4", "0"]
    assert [row["trained_attempts"] for row in rows] == ["4", "8", "0", "4"]
    assert [row["loss_tokens"] for row in rows] == ["160", "250", "0", "140"]
    assert [row["groups"] for row in rows] == ["3"] * 4
    # The log values go into the file as they are; a value that the log does not have is an empty cell.
    assert float(rows[1]["mean_reward"]) == pytest.approx(1095 / 1200)
    assert float(rows[0]["zero_variance_exact"]) == pytest.approx(2 / 3)
    assert float(rows[1]["tokens_per_attempt"]) == pytest.approx(370 / 12)
    assert (rows[2]["logprob_gap_mean"], rows[2]["loss"], rows[2]["grad_norm"]) == ("", "0.0", "0.0")
    assert float(rows[3]["logprob_gap_mean"]) == pytest.approx(0.004)
    assert {row["fault"] for row in rows} == {""}
    assert (rows[0]["rollout_seconds"], rows[0]["seconds"]) == ("10.0", "12.0")
    # The JSON holds the same rows, and summary.md explains every column.
    assert [row["kept_groups"] for row in made.report()["runs"]["seed1"]["steps"]] == [1, 2, 0, 1]
    for name, _, _ in script.STEP_COLUMNS:
        assert f"| `{name}` |" in made.text("summary.md")


def test_the_summary_gives_counts_beside_shares_and_says_where_each_number_comes_from(made):
    text = made.text("summary.md")
    for line in ("| Attempts sampled | 48 | records in the rollout files |",
                 "| Failed attempts | 11 (22.92%) | `passed` is false; share of the valid attempts |",
                 "| Kept groups | 4 of 12 (33.3%) |",
                 "| - with passes and fails | 2 |  |", "| - penalty-only | 2 |  |",
                 "| Dropped groups | 8 of 12 (66.7%) | the groups that are not kept |",
                 "| Groups with equal rewards (zero variance) | 8 of 12 (66.7%) | frozen function "
                 "`zero_variance_share` (`equal_reward`), each step by itself |",
                 "| - every attempt passed | 6 |  |", "| - every attempt failed | 2 |  |",
                 "| Groups in which all attempts passed or all failed | 10 of 12 (83.3%) |",
                 "| Steps with no kept group | 1 of 4 (25.0%) |",
                 "| Trained attempts | 16 (33.3%) |", "| Loss tokens | 550 (33.7%) |",
                 "| Mean reward, of 100 | 76.73 |",
                 "- seed 1: reward 0: 11; reward 88: 1; reward 95: 1; reward 100: 35",
                 "the gradient norm 0 (seed 1: 1 of 1 steps)."):
        assert line in text, line
    # The first lines say the main counts in sentences.
    for line in ("- Attempts sampled: seed 1: 48 in 4 steps.",
                 "- Failed attempts: seed 1: 11 of 48 (22.92%).",
                 "- Kept groups: seed 1: 4 of 12 (33.3%). The other groups gave no learning signal.",
                 "- Steps with no kept group: seed 1: 1 of 4 (25.0%).",
                 "block of steps: seed 1: 70.8% (steps 0 to 1) and 83.3% (steps 2 to 3)."):
        assert line in text, line
    # 54 seconds are 0.015 hours: the table rounds to two decimals.
    assert "| Hours of the run | 0.01 |" in text or "| Hours of the run | 0.02 |" in text
    assert "No number in this file is from the dev set or the test set." in text
    for word in ("- Attempt:", "- Scenario:", "- Step:", "- Group:", "- Reward:", "- Kept group:",
                 "- Penalty-only group:", "- Pass rate: the frozen function `pass_rate`"):
        assert word in text, word
    assert "(`steps`, `groups`, `kept_groups`, `episodes`, `harness_errors`, `trained_episodes`, " \
           "`loss_tokens`) are the totals of the log" in text
    assert "The log is the file that the manifest names by its SHA-256." in text
    assert "is the scenario file that the run manifests and all records name by its SHA-256." in text


def test_the_checks_say_only_what_was_compared(tmp_path, script):
    # A run manifest without totals, without the hash of its log and without the hash of the scenario file.
    tree = make_tree(tmp_path, summary=None, outputs=None, split_file_sha256=None)
    assert script.main(tree.options()) == 0
    text = tree.text("summary.md")
    assert "The run manifests hold no totals that could be compared with the log." in text
    assert "At least one run manifest names no SHA-256 of its log, so that was not compared." in text
    assert "Records that name it by its SHA-256: seed 1: 48 of 48 records. Run manifests that name it: " \
           "seed 1: no." in text
    checks = tree.report()["runs"]["seed1"]["checks"]
    assert checks["run_manifest"] == {"summary_keys_compared": [], "log_sha256_compared": False,
                                      "names_the_scenario_file": False}
    assert (checks["steps_compared_with_the_log"], checks["log_values_made_again"]) == (4, 28)
    assert checks["counts_compared_by_the_frozen_audit"] == ["scenarios", "n_episodes", "n_harness_errors",
                                                             "mean_reward", "kept_groups",
                                                             "n_trained_episodes", "n_loss_tokens"]
    # The made-up manifest names no training code, so the advantage code could not be compared.
    assert set(checks["advantage_code_against_the_run_manifest"].values()) == {"not recorded"}
    assert "are: not recorded." in tree.text("std_division.md")


# ---------------------------------------------------------------------------
# signal.md
# ---------------------------------------------------------------------------
def test_the_kept_groups_are_counted_per_outcome_per_rule_facts_and_per_same_name(made):
    signal = made.report()["runs"]["seed1"]["signal"]
    assert (signal["kept_groups"], signal["failed_attempts"]) == (4, 11)
    by_outcome = {row["outcome"]: row for row in signal["by_outcome"]}
    assert list(by_outcome) == ["grant", "approval", "already", "inactive"]
    assert (by_outcome["grant"]["groups"], by_outcome["grant"]["kept_groups"]) == (2, 0)
    assert (by_outcome["approval"]["kept_groups"], by_outcome["approval"]["pass_and_fail"],
            by_outcome["approval"]["failed_attempts"]) == (1, 1, 2)
    assert (by_outcome["already"]["groups"], by_outcome["already"]["kept_groups"],
            by_outcome["already"]["penalty_only"], by_outcome["already"]["failed_attempts"]) == (4, 2, 2, 0)
    inactive = by_outcome["inactive"]
    assert (inactive["groups"], inactive["attempts"], inactive["kept_groups"], inactive["failed_attempts"]) \
        == (4, 16, 1, 9)
    assert inactive["share_of_all_groups"] == pytest.approx(4 / 12)
    assert inactive["kept_share_of_its_groups"] == pytest.approx(1 / 4)
    assert inactive["share_of_all_kept_groups"] == pytest.approx(1 / 4)
    assert inactive["share_of_all_failed_attempts"] == pytest.approx(9 / 11)
    # Per rule facts, in the order of the outcomes and of the kinds of the generator.
    facts = [(row["outcome"], row["active"], row["holds_role"], row["sensitive"], row["kind"],
              row["train_families"], row["groups"], row["kept_groups"], row["failed_attempts"])
             for row in signal["by_facts"]]
    assert facts == [("grant", True, False, False, "plain", 1, 2, 0, 0),
                     ("approval", True, False, True, "plain", 1, 2, 1, 2),
                     ("already", True, True, False, "plain", 1, 2, 2, 0),
                     ("already", True, True, True, "sensitive", 1, 2, 0, 0),
                     ("inactive", False, False, False, "plain", 1, 2, 0, 8),
                     ("inactive", False, True, False, "held", 1, 2, 1, 1)]
    # Two people of one name: tr-001, tr-003 and tr-005.
    yes, no = signal["by_same_name"]
    keys = ("same_name", "train_families", "groups", "kept_groups", "failed_attempts")
    assert tuple(yes[key] for key in keys) == (True, 3, 6, 2, 3)
    assert tuple(no[key] for key in keys) == (False, 3, 6, 2, 8)
    assert (yes["pass_and_fail"], no["penalty_only"]) == (2, 2)
    assert len(signal["by_facts_and_same_name"]) == 6
    assert all("same_name" in row for row in signal["by_facts_and_same_name"])


def test_a_penalty_only_group_is_split_by_what_the_penalty_was(made):
    signal = made.report()["runs"]["seed1"]["signal"]
    # The attempt with 95 lost 5 points for one malformed message. The attempt with 88 lost 10 points for
    # two malformed messages and 2 points for one turn over the budget.
    assert signal["penalty_only"] == {
        "groups": 2, "share_of_kept_groups": pytest.approx(0.5), "attempts": 8, "penalised_attempts": 2,
        "penalised_share_of_attempts": pytest.approx(0.25),
        "penalised_by_cause": {"malformed_message_only": 1, "turns_over_budget_only": 0, "both": 1},
        "points_lost_for_malformed_messages": 15, "points_lost_for_turns_over_budget": 2}
    assert signal["pass_and_fail"] == {"groups": 2, "share_of_kept_groups": pytest.approx(0.5),
                                       "failed_attempts": 3,
                                       "groups_by_number_of_failed_attempts": {"1": 1, "2": 1}}
    # Steps 0, 1 and 3 have a kept group; only in step 3 are all kept groups penalty-only.
    assert signal["steps_with_a_kept_group"] == 3
    assert signal["steps_with_a_kept_group_share_of_steps"] == pytest.approx(0.75)
    assert signal["steps_whose_kept_groups_are_all_penalty_only"] == 1
    assert signal["steps_whose_kept_groups_are_all_penalty_only_list"] == [3]


def test_the_penalty_parts_come_from_the_frozen_reward_function(script):
    assert script.penalty_parts(make_record("tr-000", 1, 0, n_malformed=1, reward=95)) == (5, 0)
    assert script.penalty_parts(make_record("tr-000", 1, 0, n_malformed=2, reward=88)) == (10, 2)
    # Two turns over the shortest path of 4 and no malformed message: one turn over the budget of 5.
    assert script.penalty_parts(make_record("tr-000", 1, 0, n_turns=6, reward=98)) == (0, 2)
    assert script.penalty_parts(make_record("tr-000", 1, 0)) == (0, 0)
    # The cap of the malformed penalty is 15 points.
    assert script.penalty_parts(make_record("tr-000", 1, 0, n_malformed=4, n_turns=8, reward=79)) == (15, 6)


def test_signal_md_holds_the_tables_with_counts_and_shares(made):
    text = made.text("signal.md")
    for line in ("| Kept groups | 4 of 12 (33.3%) |",
                 "| - with passes and fails (share of the kept groups) | 2 (50.0%) |",
                 "| inactive | 1 of 4 (25.0%) | 25.0% | 1 | 0 | 9 (81.8%) |",
                 "| inactive | no | yes | no | 1 | 1 of 2 (50.0%) | 25.0% | 1 | 0 | 1 (9.1%) |",
                 "| already | yes | yes | no | 1 | 2 of 2 (100.0%) | 50.0% | 0 | 2 | 0 (0.0%) |",
                 "| yes | 3 | 2 of 6 (33.3%) | 50.0% | 2 | 0 | 3 (27.3%) |",
                 "| Penalised attempts in these groups (share of the attempts in these groups) | 2 (25.0%) |",
                 "| - lost points for both | 1 |",
                 "| Steps with a kept group | 3 of 4 (75.0%) |",
                 "| - of these, steps whose kept groups were all penalty-only | 1 |",
                 "| 1 | 1 |", "| 2 | 1 |", "| All groups with passes and fails | 2 |"):
        assert line in text, line
    # The first lines say the same in sentences. The kind with the most failed attempts is tr-004.
    assert "- Kept groups: seed 1: 4 of 12 (33.3%)." in text
    assert "- Penalty-only groups among the kept groups: seed 1: 2 of 4 (50.0%)." in text
    assert ("seed 1: an inactive person who does not hold the role, and a role that is not sensitive "
            "(expected outcome `inactive`). It has 8 of the 11 failed attempts (72.7%) and 2 of the 12 "
            "groups sampled (16.7%).") in text
    assert ("(3 of the 6 train families) gave these shares of the failed attempts: seed 1: 3 of 11 "
            "(27.3%).") in text
    assert "No number in this file is from the dev set or the test set." in text


def test_a_family_of_the_default_design_gets_the_facts_of_its_outcome(tmp_path, script):
    tree = make_tree(tmp_path)
    write_scenarios(tree.scenarios, with_facts=False)       # the same family ids and outcomes, no facts
    families, _ = script.load_train_families(tree.scenarios)
    assert families["tr-004"] == {"outcome": "inactive", "active": False, "holds_role": False,
                                  "sensitive": False, "kind": "plain", "same_name": False}
    assert families["tr-001"]["sensitive"] is True and families["tr-002"]["holds_role"] is True
    # tr-005 is "inactive" too: without its own facts it is a plain inactive family.
    assert families["tr-005"]["holds_role"] is False


# ---------------------------------------------------------------------------
# std_division.md
# ---------------------------------------------------------------------------
def test_division_by_the_standard_deviation_of_one_group(script):
    # Rewards 1, 0, 0, 1: mean 0.5, advantages +0.5 and -0.5, standard deviation 0.5: divided +1 and -1.
    rewards = np.array([1.0, 0.0, 0.0, 1.0])
    assert script.divided_by_std(rewards, rewards - 0.5).tolist() == [1.0, -1.0, -1.0, 1.0]
    # Rewards 1, 0.95, 1, 1: the advantages are +0.0125 three times and -0.0375; the standard deviation is
    # sqrt(0.00046875): divided +0.5774 three times and -1.7321. One attempt of 4 that differs gives this
    # pattern whatever the difference is: the same values for 1, 0, 1, 1.
    penalty = np.array([1.0, 0.95, 1.0, 1.0])
    divided = script.divided_by_std(penalty, penalty - penalty.mean())
    small, large = 1 / math.sqrt(3), math.sqrt(3)
    assert divided.tolist() == pytest.approx([small, -large, small, small])
    fail = np.array([1.0, 0.0, 1.0, 1.0])
    assert script.divided_by_std(fail, fail - fail.mean()).tolist() == pytest.approx(divided.tolist())
    # The other form: n - 1 in the divisor (sqrt(1/3) here) and 0.0001 added.
    other = script.divided_by_std(rewards, rewards - 0.5, ddof=1, eps=1e-4)
    size = 0.5 / (math.sqrt(1 / 3) + 1e-4)
    assert other.tolist() == pytest.approx([size, -size, -size, size])
    # A lost attempt (NaN) is in no mean and gets 0. Rewards 1 and 0 are left: +1 and -1.
    lost = np.array([np.nan, 1.0, 0.0, np.nan])
    assert script.divided_by_std(lost, np.array([0.0, 0.5, -0.5, 0.0])).tolist() == [0.0, 1.0, -1.0, 0.0]
    # Equal rewards have the standard deviation 0: every value is 0, and nothing is divided by 0.
    equal = np.array([1.0, 1.0, 1.0, 1.0])
    assert script.divided_by_std(equal, np.zeros(4)).tolist() == [0.0] * 4
    assert script.divided_by_std(np.array([1.0, np.nan]), np.zeros(2), ddof=1).tolist() == [0.0, 0.0]


def test_the_absolute_advantages_of_the_made_up_run_in_both_versions(made, script):
    numbers = made.report()["runs"]["seed1"]["std_division"]
    root3 = math.sqrt(3)
    # Passes and fails: 1, 0, 0, 1 gives 4 x 0.5 = 2.0 (divided 4 x 1 = 4); 0, 1, 1, 1 gives 0.75 + 3 x 0.25
    # = 1.5 (divided sqrt(3) + 3 / sqrt(3) = 2 sqrt(3)). 8 attempts.
    mixed = numbers["by_type"]["pass_and_fail"]
    assert (mixed["groups"], mixed["attempts"], mixed["share_of_kept_groups"]) == (2, 8, 0.5)
    assert mixed["sum_abs_advantage"]["trainer"] == pytest.approx(3.5)
    assert mixed["sum_abs_advantage"]["divided"] == pytest.approx(4 + 2 * root3)
    assert mixed["mean_abs_advantage"]["trainer"] == pytest.approx(0.4375)
    assert mixed["mean_abs_advantage"]["divided"] == pytest.approx((4 + 2 * root3) / 8)
    assert mixed["max_abs_advantage"]["trainer"] == pytest.approx(0.75)
    assert mixed["max_abs_advantage"]["divided"] == pytest.approx(root3)
    # Penalty-only: 1, 0.95, 1, 1 gives 3 x 0.0125 + 0.0375 = 0.075; 1, 1, 0.88, 1 gives 3 x 0.03 + 0.09 =
    # 0.18. Divided, each is 2 sqrt(3): the pattern is the same as for 0, 1, 1, 1.
    penalty = numbers["by_type"]["penalty_only"]
    assert (penalty["groups"], penalty["attempts"]) == (2, 8)
    assert penalty["sum_abs_advantage"]["trainer"] == pytest.approx(0.255)
    assert penalty["sum_abs_advantage"]["divided"] == pytest.approx(4 * root3)
    assert penalty["mean_abs_advantage"]["trainer"] == pytest.approx(0.031875)
    assert penalty["mean_abs_advantage"]["divided"] == pytest.approx(root3 / 2)
    assert numbers["by_type"]["all_fail"]["groups"] == 0
    assert numbers["by_type"]["all_fail"]["mean_abs_advantage"]["trainer"] is None
    # The share of the penalty-only groups: 0.255 of 3.755 without division, 4 sqrt(3) of 4 + 6 sqrt(3) with.
    assert numbers["total_abs_advantage"]["trainer"] == pytest.approx(3.755)
    assert numbers["total_abs_advantage"]["divided"] == pytest.approx(4 + 6 * root3)
    assert numbers["penalty_only_share_of_total"]["trainer"] == pytest.approx(0.255 / 3.755)
    assert numbers["penalty_only_share_of_total"]["divided"] == pytest.approx(4 * root3 / (4 + 6 * root3))
    assert numbers["ratio_of_means_pass_and_fail_to_penalty_only"]["trainer"] == pytest.approx(3.5 / 0.255)
    assert numbers["ratio_of_means_pass_and_fail_to_penalty_only"]["divided"] \
        == pytest.approx((4 + 2 * root3) / (4 * root3))
    assert (numbers["kept_groups"], numbers["attempts"]) == (4, 16)
    assert numbers["kept_groups_with_zero_divisor"] == 0
    # The check form divides by another number, so its sums are other sums; its share stays a share.
    assert numbers["total_abs_advantage"]["divided_check"] < numbers["total_abs_advantage"]["divided"]
    assert 0.0 < numbers["penalty_only_share_of_total"]["divided_check"] < 1.0
    assert made.report()["std_division_forms"] == {"divided": {"ddof": 0, "eps": 0.0},
                                                   "divided_check": {"ddof": 1, "eps": 1e-4},
                                                   "reward_scale": 100}


def test_the_advantages_of_the_trainer_are_those_of_group_advantages(tree, script):
    # The numbers of std_division.md rest on the trainer's own function: the same sums, made here directly.
    families, data_sha256 = script.load_train_families(tree.scenarios)
    read = script.read_run(tree.run, families, data_sha256)
    total = 0.0
    for groups in STEPS:
        plan = grpo.plan_update(step_records(groups, data_sha256), GCFG)
        total += float(np.abs(plan["advantages"]).sum())
    assert sum(group["abs_advantage"]["trainer"]["sum"] for group in read["groups"]) == pytest.approx(total)
    assert total == pytest.approx(3.755)
    assert [group["group_type"] for group in read["groups"][:6]] == [
        "dropped_all_pass", "pass_and_fail", "dropped_all_fail", "penalty_only", "dropped_all_pass",
        "pass_and_fail"]


def test_the_worked_example_is_the_most_frequent_pattern_of_each_kind(made):
    penalty, mixed = made.report()["worked_examples"]
    # Each kind has two patterns with one group each: the one with the lower rewards comes first.
    assert (penalty["group_type"], penalty["rewards"]) == (
        "penalty_only", [{"reward": 88, "attempts": 1}, {"reward": 100, "attempts": 3}])
    assert penalty["groups_with_this_pattern"] == {"seed1": 1}
    assert penalty["groups_of_this_kind"] == {"seed1": 2}
    assert [(item["reward"], item["attempts"]) for item in penalty["advantage_by_reward"]] \
        == [(88, 1), (100, 3)]
    assert [item["trainer"] for item in penalty["advantage_by_reward"]] == pytest.approx([-0.09, 0.03])
    assert [item["divided"] for item in penalty["advantage_by_reward"]] \
        == pytest.approx([-math.sqrt(3), 1 / math.sqrt(3)])
    assert (mixed["group_type"], mixed["rewards"]) == (
        "pass_and_fail", [{"reward": 0, "attempts": 1}, {"reward": 100, "attempts": 3}])
    assert [item["trainer"] for item in mixed["advantage_by_reward"]] == pytest.approx([-0.75, 0.25])
    # With division the one attempt that differs gets the same advantage in both kinds.
    assert mixed["advantage_by_reward"][0]["divided"] \
        == pytest.approx(penalty["advantage_by_reward"][0]["divided"])
    text = made.text("std_division.md")
    assert "| penalty-only | 3 x 100, 1 x 88 | 1 of 2 | reward 88 (1 attempt) | -0.0900 | -1.732 |" in text
    assert "| passes and fails | 3 x 100, 1 x 0 | 1 of 2 | reward 0 (1 attempt) | -0.7500 | -1.732 |" in text
    assert "|  |  |  | reward 100 (3 attempts) | +0.2500 | +0.577 |" in text


def test_std_division_md_says_in_two_sentences_what_the_division_would_change(made, script):
    report = made.report()
    sentences = script.two_sentences(report)
    assert len(sentences) == 2
    assert all(sentence.endswith(".") and "\n" not in sentence for sentence in sentences)
    assert sentences[0].count(". ") == 0 and sentences[1].count(". ") == 0          # one sentence each
    assert all(len(sentence.split()) <= 30 for sentence in sentences)              # short sentences
    assert "lost 5 points for a malformed message or failed" in sentences[0]
    # With division the penalty-only groups hold 4 sqrt(3) of 4 + 6 sqrt(3): 48.1%. Without: 0.255 of 3.755.
    assert sentences[1] == ("The penalty-only groups would then hold 48.1% (seed 1) of the total absolute "
                            "advantage, not 6.8% (seed 1).")
    text = made.text("std_division.md")
    assert "## What the division would have changed\n\n" + " ".join(sentences) + "\n" in text
    for line in ("| seed 1 | passes and fails | 2 | 8 | 0.4375 | 0.933 |",
                 "| seed 1 | penalty-only | 2 | 8 | 0.0319 | 0.866 |",
                 "| seed 1 | 4 | 2 (50.0%) | 3.8 | ", " (6.8%) | 14.4 | 6.9 (48.1%) |"):
        assert line in text, line
    assert "this trainer 13.7 (seed 1); with division 1.08 (seed 1)." in text
    # The scale of the trainer, the two versions with their sources, and the limits.
    assert "The trainer divides it by 100 (`REWARD_SCALE` in `accessdesk/train/grpo.py`)" in text
    assert "an advantage of 0.05 is 5 reward points" in text
    assert "arXiv:2402.03300, section 4.1.2" in text and "`GRPOTrainer`" in text
    assert "No model was trained with the division" in text
    assert "the number of attempts minus 1 in the divisor, and 0.0001 added" in text
    assert "Nothing was trained for it." in text


def test_without_one_of_the_two_kinds_the_second_sentence_says_so(tmp_path, script):
    only_fails = ((("tr-001", 1, (100, 0, 0, 100)), ("tr-000", 1, (100, 100, 100, 100)),
                   ("tr-004", 1, (0, 0, 0, 0))),)
    tree = make_tree(tmp_path, steps=only_fails)
    assert script.main(tree.options()) == 0
    sentences = script.two_sentences(tree.report())
    assert len(sentences) == 2 and "cannot be compared" in sentences[1]
    numbers = tree.report()["runs"]["seed1"]["std_division"]
    assert numbers["penalty_only_share_of_total"] == {"trainer": 0.0, "divided": 0.0, "divided_check": 0.0}
    assert numbers["ratio_of_means_pass_and_fail_to_penalty_only"]["trainer"] is None
    assert [example["group_type"] for example in tree.report()["worked_examples"]] == ["pass_and_fail"]


# ---------------------------------------------------------------------------
# cost.md
# ---------------------------------------------------------------------------
def test_the_rentals_of_the_ledger_are_split_at_the_freeze(made):
    cost = made.report()["cost"]
    rentals = {rental["instance_id"]: rental for rental in cost["rentals"]}
    assert list(rentals) == ["E", "A", "B", "C", "D"]
    # E: the amount is the projected limit, so the hours are (2.1 - 0.1) / 0.5 = 4, not the 9 of the clock.
    assert (rentals["E"]["hours"], rentals["E"]["clock_hours"], rentals["E"]["at_projected_limit"]) \
        == (pytest.approx(4.0), pytest.approx(9.0), True)
    assert rentals["E"]["note"] == "already gone; cost = the projected upper limit."
    assert tuple(rentals["A"][key] for key in ("hours", "usd_estimate", "phase", "note")) \
        == (pytest.approx(10.0), 5.1, "before", None)
    assert rentals["A"]["at_projected_limit"] is False
    # B is open for 16 hours, 4 of them before the freeze.
    across = rentals["B"]
    assert (across["phase"], across["note"]) == ("across", "destroy-all")
    assert (across["hours_before"], across["usd_before"]) == (pytest.approx(4.0), pytest.approx(4.0))
    assert (across["hours_after"], across["usd_after"]) == (pytest.approx(12.0), pytest.approx(12.0))
    assert (rentals["C"]["phase"], rentals["C"]["hours_before"], rentals["C"]["hours_after"]) \
        == ("after", 0.0, pytest.approx(10.0))
    # D was not stopped: it has no amount and is in no total.
    assert tuple(rentals["D"][key] for key in ("phase", "usd_estimate", "hours")) == ("open", None, None)
    assert cost["open_rentals"] == ["D"]
    assert cost["rentals_by_phase"] == {"before": 2, "across": 1, "after": 1, "open": 1}
    assert cost["rented"]["before_freeze"] == {"hours": pytest.approx(18.0), "usd": pytest.approx(11.2)}
    assert cost["rented"]["after_freeze"] == {"hours": pytest.approx(22.0), "usd": pytest.approx(16.0)}
    assert cost["rented"]["total"] == {"rentals": 4, "hours": pytest.approx(40.0), "usd": pytest.approx(27.2)}
    assert cost["usd_per_hour_after_freeze"] == pytest.approx(PRICE)
    # B (1.0 USD per hour) and C (0.4) were rented at some time after the freeze; D is not closed.
    assert cost["usd_per_hour_of_the_rentals_after_freeze"] == {"lowest": 0.4, "highest": 1.0}
    assert (cost["frozen_at"], cost["ledger_rows"], cost["ledger_other_rows"]) == (FREEZE, 9, 0)
    assert cost["ledger_sha256"] == sha256(made.ledger)


def test_the_stages_after_the_freeze_are_the_seconds_of_the_run_manifests(made):
    cost = made.report()["cost"]
    rows = {(row["stage"], row["part"].split(" (")[0]): row for row in cost["stage_rows"]}
    assert (rows[("Stage 1", "base rollouts")]["seconds"], rows[("Stage 1", "base rollouts")]["attempts"],
            rows[("Stage 1", "base rollouts")]["runs"]) == (2160.0, 288, 2)
    assert (rows[("Stage 1", "SFT")]["seconds"], rows[("Stage 1", "SFT")]["attempts"]) == (360.0, 0)
    assert (rows[("Stage 1", "sft rollouts")]["seconds"], rows[("Stage 1", "sft rollouts")]["attempts"]) \
        == (720.0, 96)
    assert (rows[("Checks", "overfit check")]["runs"], rows[("Checks", "overfit check")]["seconds"]) == (0, 0)
    assert (rows[("Checks", "pilot")]["seconds"], rows[("Checks", "pilot")]["attempts"]) == (180.0, 36)
    layout = rows[("Checks", "layout gradients")]
    assert (layout["seconds"], layout["attempts"]) == (180.0, 0)
    stage = "Stage 2, seed 1"
    training = rows[(stage, "GRPO training")]
    assert (training["seconds"], training["attempts"]) == (54.0, 48)
    assert rows[(stage, "dev rollout of the last checkpoint")]["seconds"] == 360.0
    assert (rows[(stage, "dev curve")]["runs"], rows[(stage, "dev curve")]["seconds"],
            rows[(stage, "dev curve")]["attempts"]) == (2, 72.0, 24)
    assert (rows[(stage, "second desk")]["seconds"], rows[(stage, "second desk")]["attempts"]) == (108.0, 36)
    # A run folder that no stage names is not left out; a run that is not complete is not counted.
    other = rows[("Other runs", "run manifests under --main that no stage above names")]
    assert (other["runs"], other["seconds"], other["attempts"]) == (1, 60.0, 5)
    assert [Path(name).name for name in other["not_counted"]] == ["unfinished"]
    assert [Path(name).name for name in rows[("Stage 1", "base rollouts")]["not_counted"]] \
        == ["base_shift_differ", "base_shift_control"]
    assert rows[(stage, "GRPO training")]["usd"] == pytest.approx(54 / 3600 * PRICE)
    assert cost["hours_in_run_manifests_after_freeze"] == pytest.approx(4254 / 3600)
    # 13 run manifests and the layout check are counted; each says that it ran on the frozen tree.
    assert (cost["run_manifests_counted"], cost["run_manifests_on_the_frozen_tree"]) == (14, 14)
    assert (cost["calibration"]["runs"], cost["calibration"]["seconds"], cost["calibration"]["attempts"]) \
        == (1, 7200.0, 50)
    assert cost["test_run_in_the_table"] is False


def test_the_stage_table_adds_up_to_the_ledger(made):
    cost = made.report()["cost"]
    stages = {stage["stage"]: stage for stage in cost["stages"]}
    assert list(stages) == [
        "Calibration and trials before the freeze", "Stage 1", "Checks", "Stage 2, seed 1", "Other runs",
        "Rented after the freeze, in no run manifest of this table", "Total"]
    before = stages["Calibration and trials before the freeze"]
    assert (before["hours"], before["usd"], before["attempts"], before["source"]) \
        == (pytest.approx(18.0), pytest.approx(11.2), 50, "ledger")
    assert (stages["Stage 1"]["hours"], stages["Stage 1"]["attempts"]) == (pytest.approx(3240 / 3600), 384)
    assert stages["Stage 1"]["usd"] == pytest.approx(0.9 * PRICE)
    assert (stages["Checks"]["hours"], stages["Checks"]["attempts"]) == (pytest.approx(0.1), 36)
    assert (stages["Stage 2, seed 1"]["hours"], stages["Stage 2, seed 1"]["attempts"]) \
        == (pytest.approx(594 / 3600), 204)
    left = stages["Rented after the freeze, in no run manifest of this table"]
    assert left["hours"] == pytest.approx(22 - 4254 / 3600)
    assert left["usd"] == pytest.approx(16 - 4254 / 3600 * PRICE)
    assert cost["not_in_a_run_manifest"] == {"hours": left["hours"], "usd": left["usd"]}
    parts = [stage for name, stage in stages.items() if name != "Total"]
    assert stages["Total"]["hours"] == pytest.approx(40.0) and stages["Total"]["usd"] == pytest.approx(27.2)
    assert sum(stage["hours"] for stage in parts) == pytest.approx(40.0)
    assert sum(stage["usd"] for stage in parts) == pytest.approx(27.2)
    # C was rented at 06:00, after the made-up GRPO run had ended at 02:00.
    assert cost["rentals_after_the_last_grpo_run"] == {
        "last_grpo_run_ended_utc": "2026-01-02T02:00:00Z", "rentals": ["C"], "hours": pytest.approx(10.0),
        "usd": pytest.approx(4.0),
        "hours_left_for_the_other_rentals": pytest.approx(22 - 4254 / 3600 - 10)}


def test_with_a_test_run_in_the_table_no_later_rental_is_called_free_of_its_runs(tree, script):
    # C is rented at 06:00, after the GRPO run ended at 02:00. The test rollouts hold no start time, so
    # they may have run in C: the table must not say what is left "if no run of this table ran in it".
    rollout_manifest(tree.main / "base_test", 900.0, 96)
    assert script.main(tree.options()) == 0
    cost = tree.report()["cost"]
    assert cost["test_run_in_the_table"] is True
    assert cost["rentals_after_the_last_grpo_run"] is None
    text = tree.text("cost.md")
    assert "If no run of this table ran in" not in text
    assert "rented after the last GRPO run had ended" not in text


def test_the_trial_runs_are_counted_by_their_start_time(tree, script):
    # Before the freeze (FREEZE: 2026-01-02T00:00:00Z), after it, and without a start time.
    rollout_manifest(tree.smoke / "a", 360.0, 10, started_utc="2025-12-31T01:00:00Z")
    rollout_manifest(tree.smoke / "b" / "c", 720.0, 20, started_utc="2026-01-03T01:00:00Z")
    rollout_manifest(tree.smoke / "d", 1800.0, 30)
    write_json(tree.smoke / "e" / "run_manifest.json", {"complete": False, "seconds": None})
    assert script.main(tree.options()) == 0
    trials = tree.report()["cost"]["trial_runs"]
    assert (trials["runs"], trials["seconds"], trials["attempts"], trials["not_counted"]) == (3, 2880.0, 60, 1)
    assert (trials["before"]["runs"], trials["before"]["seconds"]) == (1, 360.0)
    assert (trials["after"]["runs"], trials["after"]["seconds"]) == (1, 720.0)
    assert (trials["no_start_time"]["runs"], trials["no_start_time"]["seconds"]) == (1, 1800.0)
    text = tree.text("cost.md")
    assert "have 3 complete run manifests with seconds. They hold 0.80 hours (60 attempts). By " \
           "`started_utc` of the manifest: 1 started before the freeze (0.10 hours), 1 started after it " \
           "(0.20 hours), and 1 have no start time (0.50 hours)." in text
    assert "1 more run manifests are not complete or have no seconds." in text
    assert "set-up and waiting" not in text and "in no run manifest that this script reads" not in text
    # The trial runs are in no row: the stage table still adds up to the ledger.
    stages = tree.report()["cost"]["stages"]
    assert sum(stage["hours"] for stage in stages if stage["stage"] != "Total") == pytest.approx(40.0)


def test_cost_md_says_that_rows_are_rounded_and_names_folders_without_a_run_manifest(tree, script):
    (tree.main / "sft_profile").mkdir()
    (tree.main / "sft_profile" / "profile.md").write_text("made from saved attempts\n", encoding="utf-8")
    (tree.main / "tables").mkdir()
    assert script.main(tree.options()) == 0
    assert tree.report()["cost"]["folders_without_run_manifest"] == ["sft_profile"]
    text = tree.text("cost.md")
    assert "| Total |  | 40.00 | 27.20 | ledger |\n\nRows are rounded; the total is the sum before " \
           "rounding." in text
    assert "so they are in no row of this table: `sft_profile`." in text
    assert "No folder of trial runs was given or found" in text


def test_what_grpo_sampled_and_dropped(made):
    numbers = made.report()["cost"]["grpo"]["seed1"]
    assert (numbers["attempts"], numbers["attempts_in_dropped_groups"]) == (48, 32)
    assert numbers["share_of_attempts_in_dropped_groups"] == pytest.approx(32 / 48)
    assert (numbers["sampled_tokens"], numbers["sampled_tokens_in_dropped_groups"]) == (1630, 1080)
    assert numbers["dropped_groups_by_type"] == {"dropped_all_pass": 6, "dropped_all_fail": 2,
                                                 "dropped_lost_attempts": 0, "dropped_other": 0}
    assert (numbers["steps"], numbers["steps_without_kept_group"]) == (4, 1)
    assert (numbers["seconds_in_steps_without_kept_group"], numbers["share_of_step_seconds"]) == (12.0, 0.25)
    assert numbers["hours_in_steps_without_kept_group"] == pytest.approx(12 / 3600)
    assert numbers["steps_without_kept_group_share_of_steps"] == pytest.approx(0.25)
    assert numbers["usd_of_steps_without_kept_group"] == pytest.approx(12 / 3600 * PRICE)
    assert (numbers["sampling_seconds"], numbers["update_seconds"]) == (40.0, pytest.approx(8.0))


def test_cost_md_says_that_the_amounts_are_estimates_and_how_the_split_is_made(made):
    text = made.text("cost.md")
    assert "**Every amount in USD is an estimate.**" in text
    assert "The invoice of the GPU provider is the authority." in text
    for line in ("| Calibration and trials before the freeze | 50 (calibration rollouts only) | 18.00 | "
                 "11.20 | ledger |",
                 "| Stage 1 | 384 | 0.90 | 0.65 | run manifests |",
                 "| Total |  | 40.00 | 27.20 | ledger |",
                 "| Rented after the freeze, in no run manifest of this table |  | 20.82 | 15.14 | ledger "
                 "minus run manifests |",
                 "| E | 2025-12-31T00:00:00Z | 2025-12-31T09:00:00Z | 4.00 | 0.5000 | 2.10 | before the "
                 "freeze | already gone; cost = the projected upper limit. (The amount is the upper limit of "
                 "the rental, not a measured time.) |",
                 "| B | 2026-01-01T20:00:00Z | 2026-01-02T12:00:00Z | 16.00 | 1.0000 | 16.00 | across the "
                 "freeze | destroy-all |",
                 "| D | 2026-01-02T20:00:00Z | still open | no value | 0.4000 | no value | still open: not yet "
                 "in the total |  |",
                 "| All closed rentals |  |  | 40.00 |  | 27.20 |  |  |",
                 "| Attempts in dropped groups (share of the attempts sampled) | 32 (66.7%) |",
                 "| Sampled tokens in dropped groups | 1,080 of 1,630 (66.3%) |",
                 "| Steps with no kept group | 1 of 4 (25.0%) |"):
        assert line in text, line
    assert "the ledger alone cannot give the cost of a stage. 1 rental was open across the freeze. It is " \
           "split by the clock." in text
    assert "USD of a stage = its hours x 0.7273 USD per hour." in text
    assert "had prices from 0.4000 to 1.0000 USD per hour." in text
    assert "holds 1 rental of 10.00 hours that was rented after the last GRPO run had ended " \
           "(2026-01-02T02:00:00Z" in text
    assert "The folders of the test rollouts were not there when this table was made" in text
    assert ("- Rental D is still open: the ledger had no stop row for it when this file was made. It is not yet "
            "in the total, and in no amount of this file.") in text
    assert "14 of the 14 runs after the freeze say in their manifest that they started on the frozen " \
           "tree." in text
    assert "the gradient norm 0 for these steps (seed 1: 1 of 1)" in text
    assert "Not counted, because the folder is missing, the run is not complete or it has no seconds:" in text


def test_a_test_rollout_is_a_row_of_the_cost_table_only_when_its_folder_is_there(tree, script):
    # The manifest of a test rollout holds results too; only its seconds and its attempts are used.
    for name, pass_rate in (("base_test", 0.123456), ("grpo_seed1_test", 0.654321)):
        rollout_manifest(tree.main / name, 900.0, 96, summary={"episodes": 96, "pass_rate": pass_rate})
    (tree.main / "base_test" / "episodes.jsonl").write_text("not JSON: this file must not be read\n",
                                                            encoding="utf-8")
    assert script.main(tree.options()) == 0
    cost = tree.report()["cost"]
    row = next(row for row in cost["stage_rows"] if row["stage"] == "Test run")
    assert (row["runs"], row["seconds"], row["attempts"]) == (2, 1800.0, 192)
    assert [Path(name).name for name in row["folders"]] == ["base_test", "grpo_seed1_test"]
    assert cost["test_run_in_the_table"] is True
    assert "Test run" in [stage["stage"] for stage in cost["stages"]]
    assert cost["hours_in_run_manifests_after_freeze"] == pytest.approx((4254 + 1800) / 3600)
    text = tree.text("cost.md")
    assert "| Test run | test rollouts | 2 | 192 | 0.50 |" in text
    assert "No result of the test set is read here." in text
    # No output file holds a number of the test results.
    for name in ("cost.md", "summary.md", "signal.md", "std_division.md", "training_report.json"):
        assert "0.123456" not in tree.text(name) and "0.654321" not in tree.text(name)


RENTALS_HEADER = "instance_id,rented_utc,stopped_utc,purpose\n"


def test_without_a_file_of_purposes_the_cost_table_says_so(made):
    text = made.text("cost.md")
    assert "## What each rental was for" in text
    assert f"No file of purposes was found ({made.root / 'rentals.csv'})" in text
    assert made.report()["cost"]["purposes"]["found"] is False


def test_the_purposes_of_the_rentals_come_from_the_file_beside_the_ledger(tree, script):
    """rentals.csv beside the ledger is read by default. A rental of the ledger without a line gets "not in
    <file>"; a line whose times are not those of the ledger, or whose rental the ledger does not hold, is
    named. The open rental D has no stop time in the file: that is no problem."""
    (tree.root / "rentals.csv").write_text(
        RENTALS_HEADER
        + "E,2025-12-31T00:00:00Z,2025-12-31T09:00:00Z,nothing ran\n"
        + "A,2026-01-01T00:00:00Z,2026-01-01T10:00:00Z,\"calibration, round 1\"\n"
        + "B,2026-01-01T20:00:00Z,2026-01-02T11:00:00Z,the GRPO run\n"
        + "D,2026-01-02T20:00:00Z,,not recorded\n"
        + "Z,2026-01-03T00:00:00Z,,a machine of no ledger\n", encoding="utf-8")
    assert script.main(tree.options()) == 0
    purposes = tree.report()["cost"]["purposes"]
    assert purposes["found"] is True
    assert purposes["purposes"] == {"E": "nothing ran", "A": "calibration, round 1", "B": "the GRPO run",
                                    "D": "not recorded", "Z": "a machine of no ledger"}
    rentals = str(tree.root / "rentals.csv")
    assert purposes["problems"] == [
        f"B: stopped_utc is 2026-01-02T11:00:00Z in {rentals} and 2026-01-02T12:00:00Z in the ledger",
        f"Z is in {rentals} but not in the ledger"]
    text = tree.text("cost.md")
    for line in ("| Rental | Rented (UTC) | Stopped (UTC) | Hours | USD (estimate) | Purpose |",
                 "| A | 2026-01-01T00:00:00Z | 2026-01-01T10:00:00Z | 10.00 | 5.10 | calibration, round 1 |",
                 f"| C | 2026-01-02T06:00:00Z | 2026-01-02T16:00:00Z | 10.00 | 4.00 | not in {rentals} |",
                 "| D | 2026-01-02T20:00:00Z | still open | no value | no value | not recorded |",
                 f"The purposes were written by hand from the work log (`{rentals}`); the times, hours and "
                 "amounts are those of the ledger. A stop time is the time of the stop row of the ledger. "
                 "\"not recorded\" means that the log does not say what ran on that machine.",
                 f"- Not as in the ledger: Z is in {rentals} but not in the ledger."):
        assert line in text, line
    assert "| Rental | Rented (UTC) | Stop row (UTC) | Hours | USD per hour |" in text
    # A file without the named columns is refused.
    (tree.root / "rentals.csv").write_text("id,purpose\nA,x\n", encoding="utf-8")
    assert script.main(tree.options()) == 2


def test_a_stop_time_from_the_work_log_is_marked_and_explained(tree, script):
    """Rental E: the ledger wrote its stop row at 09:00, after the machine was gone, and charges the upper
    limit (240 minutes of the rent row: (2.1 - 0.1) / 0.5 = 4 hours). The file of purposes gives the time of
    the work log, with "work log" in the column stopped_from: no problem, and the table marks it. No purpose
    is "not recorded", so the sentence that explains that value is left out."""
    (tree.root / "rentals.csv").write_text(
        "instance_id,rented_utc,stopped_utc,stopped_from,purpose\n"
        "E,2025-12-31T00:00:00Z,2025-12-31T04:00Z,work log,first sessions\n"
        "A,2026-01-01T00:00:00Z,2026-01-01T10:00:00Z,,\"calibration, round 1\"\n"
        "B,2026-01-01T20:00:00Z,2026-01-02T12:00:00Z,ledger,the GRPO run\n", encoding="utf-8")
    assert script.main(tree.options()) == 0
    cost = tree.report()["cost"]
    purposes = cost["purposes"]
    assert purposes["problems"] == [] and purposes["stops_from_the_work_log"] == {"E": "2025-12-31T04:00Z"}
    rental_e = next(rental for rental in cost["rentals"] if rental["instance_id"] == "E")
    assert rental_e["stopped_utc"] == "2025-12-31T09:00:00Z" and rental_e["max_minutes"] == 240
    text = tree.text("cost.md")
    rentals = str(tree.root / "rentals.csv")
    for line in ("| E | 2025-12-31T00:00:00Z | 2025-12-31T04:00Z (work log) | 4.00 | 2.10 | first sessions |",
                 "| A | 2026-01-01T00:00:00Z | 2026-01-01T10:00:00Z | 10.00 | 5.10 | calibration, round 1 |",
                 f"The purposes were written by hand from the work log (`{rentals}`); the times, hours and "
                 "amounts are those of the ledger, except a stop time marked \"(work log)\": that one is the "
                 "time at which the work log says the machine stopped. A stop time without a mark is the time "
                 "of the stop row of the ledger.",
                 "- Rental E: the stop time 2025-12-31T04:00Z is from the work log. The stop row of the ledger "
                 "was written at 2025-12-31T09:00:00Z, and the ledger charges the upper limit of the rental: "
                 "240 minutes (`max_minutes` of its rent row), 4.00 hours.",
                 # The ledger table keeps the time of the stop row, under a header that says so.
                 "| E | 2025-12-31T00:00:00Z | 2025-12-31T09:00:00Z | 4.00 | 0.5000 | 2.10 |"):
        assert line in text, line
    assert "not recorded\" means" not in text
    # A stop time from the work log needs a time; another value of stopped_from is refused.
    for row in ("E,2025-12-31T00:00:00Z,,work log,x\n", "E,2025-12-31T00:00:00Z,2025-12-31T04:00Z,memory,x\n"):
        (tree.root / "rentals.csv").write_text("instance_id,rented_utc,stopped_utc,stopped_from,purpose\n"
                                               + row, encoding="utf-8")
        assert script.main(tree.options()) == 2


def test_the_test_run_and_its_rehearsals_are_placed_in_a_rental_by_the_clock(tree, script):
    """Only the time at the start of each line of a record is read. The test record lies in rental C (open
    from 06:00 to 16:00 on 2026-01-02); a rehearsal record in B; a record across two rentals in none."""
    (tree.main / "test_run_record.txt").write_text(
        "2026-01-02T13:00:00Z ROLLOUT_START arm=base folder=x pass_rate=0.987654\n"
        "2026-01-02T14:00:00Z ROLLOUT_END arm=base exit=0\n"
        "2026-01-02T15:00:00Z TEST_RUN_DONE\n"
        "  a line without a time, attempts=7319\n", encoding="utf-8")
    for name, times in (("dress", ("2026-01-02T01:00:00Z", "2026-01-02T02:00:00Z")),
                        ("dress_across", ("2026-01-02T05:00:00Z", "2026-01-02T15:00:00Z"))):
        folder = tree.smoke / name
        folder.mkdir(parents=True)
        (folder / "test_run_record.txt").write_text("".join(f"{time} STEP x\n" for time in times),
                                                    encoding="utf-8")
    assert script.main(tree.options()) == 0
    records = tree.report()["cost"]["records"]
    assert [(record["kind"], record["lines_with_a_time"], record["rentals"]) for record in records] == [
        ("test run", 3, ["C"]), ("rehearsal", 2, ["B"]), ("rehearsal", 2, [])]
    text = tree.text("cost.md")
    assert (f"- The test run: the times of the 3 lines of `{tree.main / 'test_run_record.txt'}` lie in rental "
            "C.") in text
    assert (f"- A rehearsal of the test chain: the times of the 2 lines of "
            f"`{tree.smoke / 'dress' / 'test_run_record.txt'}` lie in rental B.") in text
    assert "lie in no single rental of the ledger." in text
    assert "was not there when this file was made" not in text
    # Nothing but the times is taken from the record.
    for name in ("cost.md", "training_report.json"):
        assert "0.987654" not in tree.text(name) and "7319" not in tree.text(name)
    # Without the test record the table says that the test run is not placed by the clock.
    (tree.main / "test_run_record.txt").unlink()
    assert script.main(tree.options()) == 0
    assert "so the test run is placed here only by the purposes above, not by the clock." in tree.text("cost.md")


def test_a_ledger_that_cannot_be_read_is_refused(tree, script, capsys):
    later = "2026-01-03T00:00:00Z"
    cases = {
        "has no column": LEDGER_HEADER.replace(",actual_usd", "") + "\n",
        "has two rows 'rent'": LEDGER_HEADER + LEDGER_ROWS + f"{later},rent,A,1,RTX 4090,0.5,,0,,,x\n",
        "is stopped but was not rented": LEDGER_HEADER + f"{later},destroy,Z,1,RTX 4090,0.5,,0,,1.0,x\n",
        "cannot be read": LEDGER_HEADER + "yesterday,rent,A,1,RTX 4090,0.5,,0,,,x\n",
        "not a number": LEDGER_HEADER + f"{later},rent,A,1,RTX 4090,cheap,,0,,,x\n",
    }
    for phrase, text in cases.items():
        tree.ledger.write_text(text, encoding="utf-8")
        assert script.main(tree.options()) == 2
        assert phrase in capsys.readouterr().err, phrase
        assert not tree.out.exists()
    tree.ledger.unlink()
    assert script.main(tree.options()) == 2 and "does not exist" in capsys.readouterr().err
    # A row that is neither a rental nor a stop is counted and left out.
    tree.ledger.write_text(LEDGER_HEADER + LEDGER_ROWS + "2026-01-03T00:00:00Z,note,A,1,,,,,,,a remark\n",
                           encoding="utf-8")
    assert script.main(tree.options()) == 0
    assert (tree.report()["cost"]["ledger_rows"], tree.report()["cost"]["ledger_other_rows"]) == (10, 1)


def test_without_the_time_of_the_freeze_nothing_is_written(tree, script, capsys):
    write_json(tree.freeze, {"frozen_at": "the day before"})
    assert script.main(tree.options()) == 2 and "frozen_at" in capsys.readouterr().err
    tree.freeze.unlink()
    assert script.main(tree.options()) == 2 and "does not exist" in capsys.readouterr().err
    assert not tree.out.exists()


# ---------------------------------------------------------------------------
# Runs that are not as usual
# ---------------------------------------------------------------------------
def test_two_runs_stand_side_by_side_in_every_table(tmp_path, script):
    tree = make_tree(tmp_path)
    other = write_run(tree.main / "grpo_seed2", OTHER_STEPS, tree.data_sha256, seed=2,
                      ended_utc="2026-01-02T03:00:00Z")
    assert script.main(tree.options(runs=(("seed1", tree.run), ("seed2", other)))) == 0
    report = tree.report()
    assert list(report["runs"]) == ["seed1", "seed2"]
    assert [run["label"] for run in report["runs"].values()] == ["seed 1", "seed 2"]
    second = report["runs"]["seed2"]
    assert (second["totals"]["steps"], second["totals"]["attempts"], second["totals"]["failed"]) == (2, 24, 1)
    assert second["totals"]["groups_by_type"]["pass_and_fail"] == 1
    assert second["totals"]["groups_by_type"]["penalty_only"] == 1
    assert "the groups of a run: seed 1: 12 in 12 groups; seed 2: 6 in 6 groups." in tree.text("summary.md")
    assert sorted(path.name for path in tree.out.iterdir()) == [
        "cost.md", "signal.md", "std_division.md", "steps_seed1.csv", "steps_seed2.csv", "summary.md",
        "training_report.json"]
    assert "| Number | seed 1 | seed 2 | Rule and source |" in tree.text("summary.md")
    assert "| Attempts sampled | 48 | 24 | records in the rollout files |" in tree.text("summary.md")
    assert "## seed 2: kept groups per expected outcome" in tree.text("signal.md")
    assert "- Kept groups: seed 1: 4 of 12 (33.3%); seed 2: 2 of 6 (33.3%)." in tree.text("signal.md")
    # 0, 1, 1, 1 and 1, 1, 1, 0.95 have the same pattern: with division each kind holds half of the total.
    assert second["std_division"]["penalty_only_share_of_total"]["divided"] == pytest.approx(0.5)
    assert second["std_division"]["penalty_only_share_of_total"]["trainer"] == pytest.approx(0.075 / 1.575)
    sentences = script.two_sentences(report)
    assert "48.1% (seed 1) and 50.0% (seed 2) of the total absolute advantage, not 6.8% (seed 1) and 4.8% " \
           "(seed 2)" in sentences[1]
    # The second run is a stage of its own, and the last GRPO run ended at 03:00.
    assert "Stage 2, seed 2" in [stage["stage"] for stage in report["cost"]["stages"]]
    later = report["cost"]["rentals_after_the_last_grpo_run"]
    assert later["last_grpo_run_ended_utc"] == "2026-01-02T03:00:00Z"
    assert set(report["cost"]["grpo"]) == {"seed1", "seed2"}
    # The pattern 3 x 100 and 1 x 0 is in both runs: 1 group of seed 1 and 1 of seed 2.
    mixed = report["worked_examples"][1]
    assert mixed["groups_with_this_pattern"] == {"seed1": 1, "seed2": 1}


def test_runs_with_another_step_size_or_another_reward_do_not_share_a_table(tmp_path, script, capsys):
    tree = make_tree(tmp_path)
    cheaper = {**DEFAULT.to_dict(), "malformed_penalty": 3}
    other = write_run(tree.main / "grpo_seed2", OTHER_STEPS, tree.data_sha256, config=cheaper)
    assert script.main(tree.options(runs=(("seed1", tree.run), ("seed2", other)))) == 2
    assert "do not have the same reward_settings" in capsys.readouterr().err
    assert script.main(tree.options(runs=(("seed1", tree.run), ("seed1", tree.run)))) == 2
    assert "give each run once" in capsys.readouterr().err
    assert script.main(tree.options(runs=(("seed 1", tree.run),))) == 2
    assert script.main(tree.options("--block-steps", "0")) == 2
    assert not tree.out.exists()


def test_a_scenario_that_comes_a_second_time_is_counted_once(tmp_path, script):
    again = (STEPS[0], STEPS[1], STEPS[0])           # step 2 samples the scenarios of step 0 again
    tree = make_tree(tmp_path, steps=again)
    assert script.main(tree.options()) == 0
    totals = tree.report()["runs"]["seed1"]["totals"]
    assert (totals["groups"], totals["different_scenarios"]) == (9, 6)
    assert totals["first_step_with_a_scenario_of_an_earlier_step"] == 2
    assert ("the groups of a run: seed 1: 6 in 9 groups; the first step with a scenario of an earlier step "
            "is step 2.") in tree.text("summary.md")


def test_lost_attempts_are_in_no_number_and_a_group_that_lost_too_many_is_dropped(tmp_path, script):
    # Group 1 keeps exactly half of its attempts (a pass and a fail): it stays. Group 2 keeps one of four:
    # it is dropped, whatever its rewards are.
    steps = ((("tr-001", 1, ("lost", 100, 0, "lost")), ("tr-002", 1, ("lost", "lost", 100, "lost")),
              ("tr-000", 1, (100, 100, 100, 100))),)
    tree = make_tree(tmp_path, steps=steps)
    assert script.main(tree.options()) == 0
    run = tree.report()["runs"]["seed1"]
    totals = run["totals"]
    assert (totals["attempts"], totals["valid"], totals["harness_errors"], totals["failed"]) == (12, 7, 5, 1)
    assert totals["groups_by_type"]["pass_and_fail"] == 1
    assert totals["groups_by_type"]["dropped_lost_attempts"] == 1
    assert totals["groups_by_type"]["dropped_all_pass"] == 1
    assert totals["trained_attempts"] == 2
    # Rewards 1 and 0: advantages +0.5 and -0.5; divided by the standard deviation 0.5: +1 and -1.
    mixed = run["std_division"]["by_type"]["pass_and_fail"]
    assert (mixed["attempts"], mixed["sum_abs_advantage"]["trainer"], mixed["sum_abs_advantage"]["divided"]) \
        == (2, pytest.approx(1.0), pytest.approx(2.0))
    assert run["stop_rule"]["harness_errors"] == 5
    text = tree.text("summary.md")
    assert "| Attempts lost to a harness error | 5 |" in text
    assert "| - of these, dropped because the harness lost more than half of the attempts | 1 |  |" in text
    # The frozen function counts a group only if it has 2 valid attempts or more: tr-001 (a pass and a
    # fail) and tr-000 (4 passes). The group with one valid attempt is not counted.
    equal = totals["equal_reward_groups"]
    assert (equal["groups_counted"], equal["equal_reward"], equal["equal_passfail"]) == (2, 1, 1)
    assert "| Groups with equal rewards (zero variance) | 1 of 2 (50.0%) |" in text
    assert "| Dropped groups | 2 of 3 (66.7%) | the groups that are not kept |" in text
    # A group that lost an attempt is not a pattern of the worked example.
    assert tree.report()["worked_examples"] == []
    assert "## A worked example" not in tree.text("std_division.md")


def test_a_run_that_the_stop_rule_ended_is_reported_with_its_fault(tmp_path, script):
    fault = {"fault": "made-up fault", "loss": None, "n_trained_episodes": 0, "n_loss_tokens": 0,
             "logprob_gap": None, "grad_norm": None}
    tree = make_tree(tmp_path, log_changes={3: fault}, status="stopped", complete=False)
    assert script.main(tree.options()) == 0
    run = tree.report()["runs"]["seed1"]
    assert (run["set_up"]["status"], run["stop_rule"]["steps_with_a_fault"]) == ("stopped", 1)
    # The group of step 3 has unequal rewards, but the step made no update: 12 trained attempts, not 16.
    assert (run["totals"]["kept_groups"], run["totals"]["trained_attempts"]) == (4, 12)
    assert run["steps"][3]["fault"] == "made-up fault" and run["steps"][3]["loss"] is None
    row = list(csv.DictReader(io.StringIO(tree.text("steps_seed1.csv"))))[3]
    assert (row["fault"], row["loss"], row["grad_norm"]) == ("made-up fault", "", "")
    # The three files say that this kept group trained nothing.
    assert run["totals"]["kept_groups_in_steps_with_a_fault"] == 1
    for name in ("summary.md", "signal.md", "std_division.md"):
        assert "A step that ended in a fault made no update. Kept groups in such steps: seed 1: 1." \
            in tree.text(name)
    assert "| Status of the run | stopped |" in tree.text("summary.md")
    assert "| Steps that ended in a fault | 1 | a fault ends the run |" in tree.text("summary.md")
    # The run is not complete, so its seconds are not in the cost table.
    row = next(row for row in tree.report()["cost"]["stage_rows"] if row["part"] == "GRPO training")
    assert (row["runs"], row["seconds"]) == (0, 0)


# ---------------------------------------------------------------------------
# Refusals: nothing is written when a number could not be trusted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("change, phrase", [
    ({"kept_groups": 2}, "log_kept_groups"),
    ({"n_trained_episodes": 8}, "log_n_trained_episodes"),
    ({"n_loss_tokens": 1}, "log_n_loss_tokens"),
    ({"mean_reward": 0.9}, "log_mean_reward"),
    ({"scenarios": ["tr-000/s1", "tr-001/s1", "tr-005/s1"]}, "log_scenarios"),
    ({"pass_rate": 0.99}, "the log has pass_rate 0.99"),
    ({"zero_variance_exact": 0.0}, "the log has zero_variance_exact 0.0"),
    ({"invalid_call_rate": 0.5}, "the log has invalid_call_rate 0.5"),
    ({"tokens_per_episode": 1.0}, "the log has tokens_per_episode 1.0"),
    ({"turns_per_episode": 1.0}, "the log has turns_per_episode 1.0"),
])
def test_a_log_line_that_does_not_fit_its_rollout_is_refused(tmp_path, script, capsys, change, phrase):
    tree = make_tree(tmp_path, log_changes={0: change})
    assert script.main(tree.options()) == 2
    error = capsys.readouterr().err
    assert error.startswith("training_report: ") and "step 0" in error and phrase in error
    assert not tree.out.exists()


def test_a_reward_that_the_reward_function_does_not_give_is_refused(tree, script, capsys):
    path = tree.run / "rollouts" / "step_00001.episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1]["reward"] = 96                        # one malformed message costs 5 points, not 4
    write_records(path, records)
    assert script.main(tree.options()) == 2
    error = capsys.readouterr().err
    assert "step 1" in error and "reward" in error
    assert not tree.out.exists()


def test_a_run_folder_that_lacks_a_file_is_refused(tree, script, capsys):
    (tree.run / "rollouts" / "step_00002.episodes.jsonl").unlink()
    assert script.main(tree.options()) == 2
    assert "every step from 0 on needs one of each" in capsys.readouterr().err
    (tree.run / "train_log.jsonl").unlink()
    assert script.main(tree.options()) == 2
    assert "no train_log.jsonl or no saved rollout" in capsys.readouterr().err
    (tree.run / "run_manifest.json").unlink()
    assert script.main(tree.options()) == 2
    assert "has no run_manifest.json" in capsys.readouterr().err
    assert not tree.out.exists()


def test_a_run_manifest_that_does_not_fit_the_log_is_refused(tmp_path, script, capsys):
    tree = make_tree(tmp_path / "totals", summary={"kept_groups": 5, "episodes": 48})
    assert script.main(tree.options()) == 2
    assert "the run manifest has kept_groups 5; the log gives 4" in capsys.readouterr().err
    tree = make_tree(tmp_path / "log", outputs={"log": {"sha256": "0" * 64}})
    assert script.main(tree.options()) == 2
    assert "is not the file that the run manifest names" in capsys.readouterr().err
    tree = make_tree(tmp_path / "split", split="dev")
    assert script.main(tree.options()) == 2
    assert "names the split 'dev', not train" in capsys.readouterr().err
    tree = make_tree(tmp_path / "config", grpo_config=None)
    assert script.main(tree.options()) == 2
    assert "has no grpo_config" in capsys.readouterr().err


def test_another_scenario_file_than_the_one_of_the_run_is_refused(tree, script, capsys):
    # One more family in train.jsonl: the file is no longer the one that the manifest and the records name.
    with (tree.scenarios / "train.jsonl").open("a", encoding="utf-8") as handle:
        facts = {"active": True, "holds_role": False, "sensitive": False}
        handle.write(json.dumps({"family_id": "tr-006", "split": "train", "outcome": "grant",
                                 "homonym": False, "facts": facts}) + "\n")
    assert script.main(tree.options()) == 2
    assert "is not the file that the run manifest names" in capsys.readouterr().err
    assert not tree.out.exists()


def test_records_that_do_not_fit_the_families_are_refused(tmp_path, script, capsys):
    # The manifest names no scenario file here, so the records are what is compared.
    tree = make_tree(tmp_path / "hash", split_file_sha256=None)
    path = tree.run / "rollouts" / "step_00000.episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    write_records(path, [{**record, "data_sha256": "1" * 64} for record in records])
    assert script.main(tree.options()) == 2
    assert "the records name another scenario file" in capsys.readouterr().err
    tree = make_tree(tmp_path / "outcome", split_file_sha256=None)
    lines = (tree.scenarios / "train.jsonl").read_text(encoding="utf-8").replace('"outcome": "grant"',
                                                                                 '"outcome": "approval"')
    (tree.scenarios / "train.jsonl").write_text(lines, encoding="utf-8")
    path = tree.run / "rollouts" / "step_00000.episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    write_records(path, [{key: value for key, value in record.items() if key != "data_sha256"}
                         for record in records])
    assert script.main(tree.options()) == 2
    assert "the record has the outcome grant, its family has approval" in capsys.readouterr().err


def test_a_scenario_folder_without_a_train_file_or_with_another_split_is_refused(tree, script, capsys):
    text = (tree.scenarios / "train.jsonl").read_text(encoding="utf-8")
    (tree.scenarios / "train.jsonl").write_text(text.replace('"split": "train"', '"split": "test"', 1),
                                                encoding="utf-8")
    assert script.main(tree.options()) == 2
    assert "only the train split is read here" in capsys.readouterr().err
    (tree.scenarios / "train.jsonl").unlink()
    assert script.main(tree.options()) == 2
    assert "train.jsonl does not exist" in capsys.readouterr().err
    assert not tree.out.exists()


# ---------------------------------------------------------------------------
# What is read and what is written
# ---------------------------------------------------------------------------
def test_only_the_named_files_are_read_and_the_saved_runs_are_not_changed(tree, script, reads):
    before = {path: path.read_bytes() for path in sorted(tree.root.rglob("*")) if path.is_file()}
    reads.clear()
    assert script.main(tree.options()) == 0
    inside = [path for path in reads if tree.root in path.parents]
    assert inside, "the script read nothing of the made-up tree"
    # No token file, no dev or test scenario file, and of a rollout folder only its run manifest.
    assert not [path for path in inside if path.name.endswith("tokens.jsonl")]
    assert not [path for path in inside if path.name in ("dev.jsonl", "test.jsonl")]
    assert {path.parent.name for path in inside if path.name.endswith("episodes.jsonl")} == {"rollouts"}
    assert {path.name for path in inside if tree.scenarios in path.parents} == {"train.jsonl"}
    # Every file that was there is unchanged, and every new file is in the output folder.
    after = {path: path.read_bytes() for path in sorted(tree.root.rglob("*")) if path.is_file()}
    assert {path: after[path] for path in before} == before
    assert {path.parent for path in after if path not in before} == {tree.out}


def test_the_same_input_gives_the_same_bytes(tree, script):
    assert script.main(tree.options()) == 0
    first = {path.name: path.read_bytes() for path in tree.out.iterdir()}
    assert script.main(tree.options()) == 0
    assert {path.name: path.read_bytes() for path in tree.out.iterdir()} == first
    assert len(first) == 6
    # The JSON names the script that made it, by its hash.
    report = tree.report()
    assert report["script"] == "scripts/training_report.py"
    assert report["script_sha256"] == sha256(ROOT / "scripts" / "training_report.py")
    assert report["inputs"]["train_file_sha256"] == tree.data_sha256
    assert report["inputs"]["train_families"] == 6


def test_the_words_of_the_reward_come_from_the_config_of_the_run(tmp_path, script):
    harder = {**DEFAULT.to_dict(), "penalise_malformed_on_fail": True}
    tree = make_tree(tmp_path, config=harder)
    assert script.main(tree.options()) == 0
    for name in ("summary.md", "signal.md", "std_division.md"):
        assert "- Reward: 0 or less for a failed attempt (it loses the points for its malformed messages). " \
               "100 for a passed attempt, minus 5 for each malformed message (at most 15), minus 2 for " \
               "each turn over the budget (at most 10)." in tree.text(name)


def test_small_helpers(script):
    assert script.label_of("seed1") == "seed 1" and script.label_of("pilot") == "pilot"
    assert script.ratio(1, 4) == 0.25 and script.ratio(1, 0) is None
    assert script.count_of(91, 640) == "91 of 640 (14.2%)"
    assert script.count_share(1456, 10240) == "1,456 (14.2%)"
    assert script.count_share(0, 0) == "0 (no value)"
    assert script.hours_text(5400) == "1.50" and script.hours_text(None) == "no value"
    assert script.joined(["a"]) == "a" and script.joined(["a", "b", "c"]) == "a, b and c"
    assert script.steps_csv([{name: None for name, _, _ in script.STEP_COLUMNS}]).splitlines()[1] \
        == "," * (len(script.STEP_COLUMNS) - 1)
    assert script.parse_utc("2026-11-18T21:48:10Z").isoformat() == "2026-11-18T21:48:10+00:00"
    with pytest.raises(ValueError):
        script.parse_utc("2026-11-18 21:48")


# ---------------------------------------------------------------------------
# The real saved runs, if they are there
# ---------------------------------------------------------------------------
REAL_MAIN = ROOT / "results" / "main"
REAL_RUNS = (("seed1", REAL_MAIN / "grpo_seed1"), ("seed2", REAL_MAIN / "grpo_seed2"))
REAL_FILES = (ROOT / "results" / "spend_ledger.csv", ROOT / "data" / "scenarios" / "train.jsonl",
              ROOT / "MANIFEST.json")


def test_the_real_saved_runs_give_tables_that_fit_their_run_manifests(tmp_path, script, reads, capsys):
    needed = [folder / name for _, folder in REAL_RUNS for name in ("run_manifest.json", "train_log.jsonl")]
    if not all(path.is_file() for path in [*needed, *REAL_FILES]) \
            or not all((folder / "rollouts").is_dir() for _, folder in REAL_RUNS):
        pytest.skip("the saved GRPO runs, the ledger or the scenario file are not in this copy")
    reads.clear()
    assert script.main(["--out", str(tmp_path / "out")]) == 0, capsys.readouterr().err
    report = json.loads((tmp_path / "out" / "training_report.json").read_text(encoding="utf-8"))
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "cost.md", "signal.md", "std_division.md", "steps_seed1.csv", "steps_seed2.csv", "summary.md",
        "training_report.json"]

    for name, folder in REAL_RUNS:
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        run, summary, config = report["runs"][name], manifest["summary"], manifest["grpo_config"]
        totals = run["totals"]
        assert (totals["steps"], totals["groups"], totals["kept_groups"], totals["attempts"]) \
            == (summary["steps"], summary["groups"], summary["kept_groups"], summary["episodes"])
        assert (totals["trained_attempts"], totals["loss_tokens"]) \
            == (summary["trained_episodes"], summary["loss_tokens"])
        assert totals["attempts"] == totals["steps"] * config["n_scenarios"] * config["group_size"]
        assert sum(totals["groups_by_type"].values()) == totals["groups"]
        assert totals["groups_by_type"]["pass_and_fail"] + totals["groups_by_type"]["penalty_only"] \
            + totals["groups_by_type"]["all_fail"] == totals["kept_groups"]
        assert totals["passed"] + totals["failed"] == totals["valid"]
        assert sum(totals["reward_values"].values()) == totals["valid"]
        assert totals["reward_values"].get("0", 0) == totals["failed"]
        # The blocks and the tables of signal.md are the same groups and attempts, counted in parts.
        assert sum(block["attempts"] for block in run["blocks"]) == totals["attempts"]
        assert sum(block["failed"] for block in run["blocks"]) == totals["failed"]
        assert sum(block["kept_groups"] for block in run["blocks"]) == totals["kept_groups"]
        for table in ("by_outcome", "by_facts", "by_same_name", "by_facts_and_same_name"):
            rows = run["signal"][table]
            assert sum(row["groups"] for row in rows) == totals["groups"], table
            assert sum(row["kept_groups"] for row in rows) == totals["kept_groups"], table
            assert sum(row["failed_attempts"] for row in rows) == totals["failed"], table
        assert sum(row["train_families"] for row in run["signal"]["by_facts"]) == manifest["families"]
        # The advantages: both kinds have groups, and the division raises the share of the penalty-only ones.
        division = run["std_division"]
        assert division["kept_groups"] == totals["kept_groups"]
        assert division["attempts"] == totals["trained_attempts"]
        share = division["penalty_only_share_of_total"]
        assert 0.0 < share["trainer"] < share["divided"] < 1.0
        assert abs(share["divided"] - share["divided_check"]) < 0.01
        assert division["kept_groups_with_zero_divisor"] == 0
        assert set(run["checks"]["advantage_code_against_the_run_manifest"].values()) == {"same"}
        assert run["checks"]["steps_compared_with_the_log"] == totals["steps"]
        assert len(run["steps"]) == totals["steps"]
        assert run["time"]["run_seconds"] == manifest["seconds"]
        step_file = (tmp_path / "out" / f"steps_{name}.csv").read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(step_file)))
        assert [int(row["kept_groups"]) for row in rows] == [step["kept_groups"] for step in run["steps"]]
        assert sum(int(row["trained_attempts"]) for row in rows) == totals["trained_attempts"]

    # The cost: the total is the sum of the amounts of the ledger, and the stages add up to it.
    with (ROOT / "results" / "spend_ledger.csv").open(newline="", encoding="utf-8") as handle:
        amounts = [float(row["actual_usd"]) for row in csv.DictReader(handle)
                   if row["event"] == "destroy" and row["actual_usd"]]
    cost = report["cost"]
    assert cost["rented"]["total"]["usd"] == pytest.approx(sum(amounts))
    parts = [stage for stage in cost["stages"] if stage["stage"] != "Total"]
    assert sum(stage["usd"] for stage in parts) == pytest.approx(cost["rented"]["total"]["usd"])
    assert sum(stage["hours"] for stage in parts) == pytest.approx(cost["rented"]["total"]["hours"])
    # Every run of stage 1, of the checks and of stage 2 says that it started on the frozen tree.
    staged = [row for row in cost["stage_rows"] if row["stage"].startswith(("Stage", "Checks"))]
    assert sum(row["runs"] for row in staged if row["part"] == "GRPO training") == len(REAL_RUNS)
    assert all(row["on_frozen_tree"] == row["runs"] for row in staged)
    test_names = ("base_test", "sft_test", "grpo_seed1_test", "grpo_seed2_test")
    test_folders = [REAL_MAIN / name for name in test_names]
    assert cost["test_run_in_the_table"] == any(folder.is_dir() for folder in test_folders)
    # The file of purposes fits the ledger. Rental 61047327 stopped itself at 05:06 PST (13:06 UTC) by the
    # work log (_dev/NIGHT_LOG.md); the ledger wrote its stop row later and charges the 240-minute limit.
    assert cost["purposes"]["problems"] == []
    assert cost["purposes"]["stops_from_the_work_log"] == {"61047327": "2026-11-17T13:06Z"}
    late = next(rental for rental in cost["rentals"] if rental["instance_id"] == "61047327")
    assert (late["stopped_utc"], late["at_projected_limit"], late["max_minutes"]) \
        == ("2026-11-17T16:02:31Z", True, 240.0)
    cost_text = (tmp_path / "out" / "cost.md").read_text(encoding="utf-8")
    assert "| 61047327 | 2026-11-17T09:05:08Z | 2026-11-17T13:06Z (work log) | 4.00 |" in cost_text
    assert "\"not recorded\" means" not in cost_text

    # What was read: no token file, no dev or test scenario, nothing that is kept unread, no key; of
    # the attempts only the training rollouts of the GRPO runs.
    assert reads
    assert not [path for path in reads if path.name.endswith("tokens.jsonl")]
    assert not [path for path in reads if path.name in ("dev.jsonl", "test.jsonl", "vast.env")]
    assert not [path for path in reads if "_sealed" in path.parts]
    assert not [path for path in reads if path.name.startswith(("audit_key", "audit_sheet", "audit_labels"))]
    assert {path.parent.name for path in reads if path.name.endswith("episodes.jsonl")} == {"rollouts"}
    # Every output is words and numbers: the four .md files state that they hold no dev or test number.
    for name in ("summary.md", "signal.md", "std_division.md"):
        text = (tmp_path / "out" / name).read_text(encoding="utf-8")
        assert "No number in this file is from the dev set or the test set." in text
