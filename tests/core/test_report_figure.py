"""The figure of the write-up (scripts/make_figure.py): every plotted value, the caption, and every refusal.

The first part uses made-up runs in a temporary folder: two small GRPO runs (6 steps of 2 groups of 4
attempts) with hand-made attempts, a log whose numbers are typed by hand, and a dev curve of 3 families.
The expected numbers are worked out by hand in this file. Nothing is written into the repository.

The last part runs the script on the saved runs of results/main/, if they are there, in a process of its
own, and makes every value of its CSV again with plain code of this file (no function of the script and
no metric function of the package). It also records every file that the script opens: no file of the
test split, no token file, no sealed result.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.make_records import make_record, scenario_records

ROOT = Path(__file__).resolve().parents[2]
SAVED_RUNS = [ROOT / "results" / "main" / name for name in ("grpo_seed1", "grpo_seed2")]
SAVED_OUT = ROOT / "results" / "main" / "report" / "figure"
DATA_FILES = ("learning_curves.csv", "invalid_call_rate.csv", "caption.md")
FILES = DATA_FILES + ("learning_curves.png", "learning_curves.pdf")
START_ADAPTER = "made_up/sft/adapter"
WEIGHTS_SHA256 = "ab" * 32


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def figure():
    return load_script("make_figure")


# ---------------------------------------------------------------------------
# Made-up runs
# ---------------------------------------------------------------------------
# The rewards of the two groups of each step. A reward of 95 is a pass after one malformed message.
REWARDS_A = [([100, 0, 100, 0], [100, 100, 100, 100]),
             ([100, 100, 100, 100], [100, 100, 100, 100]),
             ([0, 0, 0, 0], [100, 95, 100, 100]),
             ([100, 100, 0, 0], [0, 100, 100, 100]),
             ([100, 100, 100, 100], [100, 100, 100, 100]),
             ([100, 100, 100, 0], [100, 100, 100, 100])]
# What the log of run A holds, worked out by hand from REWARDS_A. Every attempt has 4 assistant messages
# (the pass with reward 95 has 5), and a message of step s has 10 + s tokens.
#   mean reward       the sum of the 8 rewards / 800
#   tokens            (10 + s) * messages / 8;  messages = 32, and 33 in step 2
#   zero variance     the groups with 4 equal rewards / 2;  kept: the other groups
BY_HAND_A = [
    {"mean_reward": 0.75, "tokens": 40.0, "turns": 4.0, "zero": 0.5, "kept": 1, "invalid": 0.0},
    {"mean_reward": 1.0, "tokens": 44.0, "turns": 4.0, "zero": 1.0, "kept": 0, "invalid": 0.0},
    {"mean_reward": 0.49375, "tokens": 49.5, "turns": 4.125, "zero": 0.5, "kept": 1, "invalid": 1 / 33},
    {"mean_reward": 0.625, "tokens": 52.0, "turns": 4.0, "zero": 0.0, "kept": 2, "invalid": 0.0},
    {"mean_reward": 1.0, "tokens": 56.0, "turns": 4.0, "zero": 1.0, "kept": 0, "invalid": 0.0},
    {"mean_reward": 0.875, "tokens": 60.0, "turns": 4.0, "zero": 0.5, "kept": 1, "invalid": 0.0},
]
# Run B: the steps of run A in the reverse order. In its step 5 the harness loses the last attempt of the
# first group (see `made_up`): the rewards are (100, 0, 100, lost) and (100, 100, 100, 100).
REWARDS_B = list(reversed(REWARDS_A))
BY_HAND_B = [
    {"mean_reward": 0.875, "tokens": 40.0, "turns": 4.0, "zero": 0.5, "kept": 1, "invalid": 0.0},
    {"mean_reward": 1.0, "tokens": 44.0, "turns": 4.0, "zero": 1.0, "kept": 0, "invalid": 0.0},
    {"mean_reward": 0.625, "tokens": 48.0, "turns": 4.0, "zero": 0.0, "kept": 2, "invalid": 0.0},
    {"mean_reward": 0.49375, "tokens": 53.625, "turns": 4.125, "zero": 0.5, "kept": 1, "invalid": 1 / 33},
    {"mean_reward": 1.0, "tokens": 56.0, "turns": 4.0, "zero": 1.0, "kept": 0, "invalid": 0.0},
    # step 5: 600 / 700; 7 valid attempts of 4 messages of 15 tokens; the first group still has two
    # different rewards, so it is kept
    {"mean_reward": 6 / 7, "tokens": 60.0, "turns": 4.0, "zero": 0.5, "kept": 1, "invalid": 0.0},
]
# The dev curve: for each checkpoint step, pass (1) or fail (0) of the 2 attempts of the 3 dev families.
# None is an attempt that the harness lost: the family then has one valid attempt. In run B at step 3 the
# family pass rates are (1, 1, 0.5), so the family-weighted pass rate is 2.5 / 3 = 0.833, and not the
# share of the valid attempts that passed (4 of 5 = 0.8).
CURVE_A = {0: [(1, 1), (1, 0), (0, 0)], 3: [(1, 1), (1, 1), (1, 0)], 6: [(1, 1), (1, 1), (1, 1)]}
CURVE_B = {0: [(1, 1), (1, 0), (0, 0)], 3: [(1, 1), (1, None), (1, 0)], 6: [(1, 1), (1, 1), (1, 0)]}
DEV_FAMILIES = ["dv-000", "dv-001", "dv-002"]


def family_rates(passes: list[tuple]) -> list[float]:
    """The pass rate of each family of a made-up curve point, over its attempts that were not lost."""
    valid = [[passed for passed in pair if passed is not None] for pair in passes]
    return [sum(pair) / len(pair) for pair in valid]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_lines(path: Path, objects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(item) + "\n" for item in objects).encode("utf-8"))


def step_records(step: int, rewards: tuple[list[int], list[int]], lost: int | None = None) -> list[dict]:
    """The 8 attempts of one made-up step, in group order. Every message has 10 + step tokens.
    `lost`: the position of an attempt that the harness lost."""
    records = []
    for group, group_rewards in enumerate(rewards):
        records += scenario_records(f"tr-{2 * step + group:03d}", 1, group_rewards, arm="grpo_train")
    for index, record in enumerate(records):
        for turn in record["turns"]:
            turn["n_tokens"] = 10 + step
        if index == lost:
            record["harness_error"] = "made-up: the harness lost this attempt"
    return records


def log_row(step: int, by_hand: dict, n_errors: int = 0) -> dict:
    """One line of a made-up train log: the numbers that were worked out by hand."""
    no_group = by_hand["kept"] == 0
    return {"step": step, "scenarios": [f"tr-{2 * step + group:03d}/s1" for group in (0, 1)],
            "n_episodes": 8, "n_harness_errors": n_errors, "mean_reward": by_hand["mean_reward"],
            "tokens_per_episode": by_hand["tokens"], "turns_per_episode": by_hand["turns"],
            "zero_variance_exact": by_hand["zero"], "invalid_call_rate": by_hand["invalid"],
            "kept_groups": by_hand["kept"], "loss": 0.0 if no_group else -0.01,
            "grad_norm": 0.0 if no_group else 0.02, "fault": None}


def write_manifest(folder: Path, seed: int, **changes) -> None:
    """The run manifest of a made-up GRPO run; it names the SHA-256 of the log that is in the folder."""
    manifest = {
        "complete": True, "status": "complete", "seed": seed, "split": "train", "styles": [1, 2, 3],
        "grpo_config": {"n_scenarios": 2, "group_size": 4, "tie_tol": 0.0, "steps": 6},
        "config": {"pass_reward": 100, "malformed_penalty": 5, "malformed_cap": 15, "turn_penalty": 2,
                   "turn_cap": 10, "penalise_malformed_on_fail": False, "max_new_tokens": 260},
        "start_adapter": {"folder": START_ADAPTER,
                          "files": {"adapter_model.safetensors": {"sha256": WEIGHTS_SHA256}},
                          "trained_by": {"script": "scripts/gpu/train_sft.py"}},
        "outputs": {"log": {"sha256": sha256(folder / "train_log.jsonl")}},
    }
    manifest.update(changes)
    (folder / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_curve_point(run: Path, step: int, passes: list[tuple], families: list[str] | None = None,
                      **changes) -> Path:
    """One point of the made-up dev curve: dev families (the 3 of DEV_FAMILIES if no list is given) in
    wording style 1, 2 attempts each. `passes` has one pair per family."""
    families = DEV_FAMILIES if families is None else families
    folder = run.parent / f"{run.name}_curve_{step:03d}"
    records = [make_record(family_id, 1, sample, passed=bool(passed), arm=folder.name,
                           harness_error=None if passed is not None else "made-up: lost")
               for family_id, pair in zip(families, passes) for sample, passed in enumerate(pair)]
    write_lines(folder / "episodes.jsonl", records)
    adapter = START_ADAPTER if step == 0 else f"made_up/{run.name}/checkpoints/step_{step:05d}/adapter"
    manifest = {"complete": True, "adapter": adapter, "split": "dev", "family_set": "curve_subset",
                "families": len(families), "scenarios": len(families), "samples": 2, "run_seed": 0,
                "command": ["scripts/gpu/rollout.py", "--set", "curve_subset", "--styles", "1"]}
    manifest.update(changes)
    (folder / "run_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    if step == 0:
        record = f"arm={folder.name}\nadapter_model.safetensors={WEIGHTS_SHA256}\n"
        (folder / "chain_record.txt").write_text(record, encoding="utf-8")
    return folder


def write_run(folder: Path, seed: int, rewards: list, by_hand: list[dict], curve: dict,
              lost: tuple[int, int] | None = None) -> Path:
    """A made-up GRPO run folder with its rollouts, its log, its manifest and its dev curve.
    `lost`: (step, position) of one attempt that the harness lost."""
    rows = []
    for step, step_rewards in enumerate(rewards):
        position = lost[1] if lost is not None and lost[0] == step else None
        write_lines(folder / "rollouts" / f"step_{step:05d}.episodes.jsonl",
                    step_records(step, step_rewards, position))
        rows.append(log_row(step, by_hand[step], n_errors=0 if position is None else 1))
    write_lines(folder / "train_log.jsonl", rows)
    write_manifest(folder, seed)
    for step, passes in curve.items():
        write_curve_point(folder, step, passes)
    return folder


@pytest.fixture()
def made_up(tmp_path) -> dict:
    """Two made-up runs (seed 1: REWARDS_A; seed 2: REWARDS_B with one lost attempt) and a scenario folder
    that holds only sets.json."""
    main = tmp_path / "main"
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "sets.json").write_text(json.dumps({"curve_subset": DEV_FAMILIES}), encoding="utf-8")
    run_a = write_run(main / "grpo_a", 1, REWARDS_A, BY_HAND_A, CURVE_A)
    # run B, step 5: the rewards are REWARDS_A[0]; the attempt at position 3 (reward 0) is lost
    run_b = write_run(main / "grpo_b", 2, REWARDS_B, BY_HAND_B, CURVE_B, lost=(5, 3))
    return {"a": run_a, "b": run_b, "scenarios": scenarios, "out": tmp_path / "out"}


def arguments(made_up: dict, *more: str, runs: tuple[str, ...] = ("a", "b")) -> list[str]:
    args = []
    for name in runs:
        args += ["--run", str(made_up[name])]
    return args + ["--scenarios", str(made_up["scenarios"]), "--out", str(made_up["out"]),
                   "--window", "3", "--draws", "2000", *more]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def caption_of(made_up: dict) -> str:
    return (made_up["out"] / "caption.md").read_text(encoding="utf-8")


def bootstrap_by_hand(values: list[float], n_draws: int, seed: int) -> tuple[float, float]:
    """The 95% percentile interval of the mean, by resampling the values: written from the text of
    `family_bootstrap_ci` (one table of indices from np.random.default_rng(seed))."""
    array = np.asarray(values, dtype=float)
    index = np.random.default_rng(seed).integers(0, array.size, size=(n_draws, array.size))
    low, high = np.quantile(array[index].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Small functions
# ---------------------------------------------------------------------------
def test_centred_mean_by_hand(figure):
    assert figure.centred_mean([1.0, 2.0, 6.0, 4.0], 3) == [None, 3.0, 4.0, None]
    assert figure.centred_mean([1.0, 2.0, 6.0, 4.0], 1) == [1.0, 2.0, 6.0, 4.0]
    assert figure.centred_mean([1.0, 2.0], 3) == [None, None]               # the window never fits
    values = [float(i * i) for i in range(11)]
    by_window = figure.centred_mean(values, 9)
    assert by_window[:4] == [None] * 4 and by_window[7:] == [None] * 4      # 4 steps short on each side
    assert by_window[4] == sum(values[0:9]) / 9 and by_window[6] == sum(values[2:11]) / 9
    for window in (0, 2, 4, -3):
        with pytest.raises(figure.FigureError):
            figure.centred_mean(values, window)


def test_the_cut_axis_of_panel_a_and_the_block_of_the_table(figure):
    assert figure.zoom_low(0.875) == 0.85          # the saved runs: the lowest step mean is 0.875
    assert figure.zoom_low(0.9) == 0.85            # a value on a tick gets a tick below it
    assert figure.zoom_low(0.855) == 0.8           # less than 0.01 above the tick: one tick more
    assert figure.zoom_low(0.49375) == 0.45
    assert figure.zoom_low(0.02) == 0.0 and figure.zoom_low(0.0) == 0.0
    assert figure.zoom_low(1.0) == 0.95
    assert [figure.block_size(n) for n in (80, 40, 39, 6, 2, 1)] == [20, 20, 19, 3, 1, 1]


def test_each_run_has_its_own_colour_and_its_own_marker(figure):
    colours = [style["color"] for style in figure.SERIES]
    assert len(colours) == len(set(colours)) == figure.MAX_RUNS == 3
    assert len({(style["marker"], style["filled"]) for style in figure.SERIES}) == 3   # not colour alone


# ---------------------------------------------------------------------------
# The two CSV files of the made-up runs
# ---------------------------------------------------------------------------
def test_every_plotted_value_of_the_made_up_runs_is_in_the_csv(figure, made_up, capsys):
    assert figure.main(arguments(made_up, "--no-image")) == 0
    rows = read_csv(made_up["out"] / "learning_curves.csv")
    assert [(row["run"], row["seed"], row["step"]) for row in rows] == \
        [(name, seed, str(step)) for name, seed in (("grpo_a", "1"), ("grpo_b", "2")) for step in range(7)]
    cases = (("grpo_a", BY_HAND_A, CURVE_A, None), ("grpo_b", BY_HAND_B, CURVE_B, 5))
    for name, by_hand, curve, lost in cases:
        mine = [row for row in rows if row["run"] == name]
        for step, expected in enumerate(by_hand):
            row = mine[step]
            assert (row["attempts"], row["valid_attempts"]) == ("8", "7" if step == lost else "8")
            assert float(row["mean_reward"]) == pytest.approx(expected["mean_reward"], abs=1e-12)
            assert float(row["tokens_per_attempt"]) == pytest.approx(expected["tokens"], abs=1e-12)
            assert float(row["zero_variance_fraction"]) == expected["zero"]
            assert (row["groups"], row["kept_groups"]) == ("2", str(expected["kept"]))
            for column, key in (("mean_reward_ma3", "mean_reward"), ("tokens_per_attempt_ma3", "tokens"),
                                ("zero_variance_fraction_ma3", "zero")):
                if step in (0, 5):                         # the window of 3 steps does not fit at the ends
                    assert row[column] == ""
                else:
                    around = [by_hand[step + shift][key] for shift in (-1, 0, 1)]
                    assert float(row[column]) == pytest.approx(sum(around) / 3, abs=1e-12)
        # the last checkpoint is after the last step of the log: its row has the dev columns only
        assert all(mine[6][column] == "" for column in mine[6]
                   if not column.startswith("dev_") and column not in ("run", "seed", "step"))
        for step in range(7):
            row = mine[step]
            if step not in curve:
                assert all(row[column] == "" for column in row if column.startswith("dev_"))
                continue
            rates = family_rates(curve[step])
            flat = [passed for pair in curve[step] for passed in pair if passed is not None]
            low, high = bootstrap_by_hand(rates, 2000, 0)
            assert float(row["dev_pass_rate"]) == pytest.approx(sum(rates) / 3, abs=1e-12)
            assert float(row["dev_ci_low"]) == pytest.approx(low, abs=1e-12)
            assert float(row["dev_ci_high"]) == pytest.approx(high, abs=1e-12)
            assert (row["dev_attempts"], row["dev_families"]) == ("6", "3")
            assert (row["dev_valid_attempts"], row["dev_passed"]) == (str(len(flat)), str(sum(flat)))
            assert row["dev_adapter"] == (START_ADAPTER if step == 0 else
                                          f"made_up/{name}/checkpoints/step_{step:05d}/adapter")
    # run A once more as plain numbers: 3 of 6 at step 0, 5 of 6 at step 3, 6 of 6 at step 6
    of_a = [row for row in rows if row["run"] == "grpo_a"]
    assert [row["dev_passed"] for row in of_a if row["dev_passed"]] == ["3", "5", "6"]
    assert [row["dev_pass_rate"] for row in of_a if row["dev_pass_rate"]] == ["0.5", repr(2.5 / 3), "1.0"]
    assert (of_a[6]["dev_ci_low"], of_a[6]["dev_ci_high"]) == ("1.0", "1.0")       # all passed: no width
    # run B at step 3: the unit is the family. 4 of the 5 valid attempts passed, and the rate is 2.5 / 3
    of_b = [row for row in rows if row["run"] == "grpo_b"]
    assert (of_b[3]["dev_valid_attempts"], of_b[3]["dev_passed"]) == ("5", "4")
    assert of_b[3]["dev_pass_rate"] == repr(2.5 / 3) != "0.8"
    out = capsys.readouterr().out
    assert "wrote learning_curves.csv, invalid_call_rate.csv, caption.md into" in out
    assert "--no-image: learning_curves.png and learning_curves.pdf were not written" in out
    assert sorted(path.name for path in made_up["out"].iterdir()) == sorted(DATA_FILES)


def test_the_draws_and_the_seed_of_the_command_line_reach_the_bootstrap(figure, made_up):
    """12 families with pass rates of 0, 0.5 and 1, and only 40 draws: the interval moves with the seed.
    (With the 3 families of the other tests it does not: it reaches from the lowest to the highest.)"""
    families = [f"dv-{index:03d}" for index in range(12)]
    passes = [(1, 1), (1, 0), (0, 0), (1, 1), (0, 1), (0, 0), (1, 1), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1)]
    rates = family_rates(passes)
    (made_up["scenarios"] / "sets.json").write_text(json.dumps({"curve_subset": families}), encoding="utf-8")
    for name in ("a", "b"):
        for step in (0, 3, 6):
            write_curve_point(made_up[name], step, passes, families)
    intervals = [bootstrap_by_hand(rates, draws, seed) for draws, seed in ((40, 0), (40, 5), (400, 0))]
    for index, one in enumerate(intervals):                # the three intervals differ by more than rounding
        for other in intervals[index + 1:]:
            assert max(abs(one[0] - other[0]), abs(one[1] - other[1])) > 0.001
    for draws, seed in ((40, 0), (40, 5), (400, 0)):
        if made_up["out"].exists():
            shutil.rmtree(made_up["out"])
        assert figure.main(arguments(made_up, "--no-image", "--draws", str(draws), "--seed", str(seed))) == 0
        low, high = bootstrap_by_hand(rates, draws, seed)
        with_curve = [row for row in read_csv(made_up["out"] / "learning_curves.csv") if row["dev_pass_rate"]]
        assert len(with_curve) == 6
        for row in with_curve:
            assert float(row["dev_pass_rate"]) == pytest.approx(sum(rates) / 12, abs=1e-12)
            assert float(row["dev_ci_low"]) == pytest.approx(low, abs=1e-12)
            assert float(row["dev_ci_high"]) == pytest.approx(high, abs=1e-12)
        assert f"{draws:,} bootstrap draws and the bootstrap seed {seed}" in caption_of(made_up)


def test_the_invalid_call_rate_file_has_the_rate_and_both_counts(figure, made_up):
    assert figure.main(arguments(made_up, "--no-image")) == 0
    rows = read_csv(made_up["out"] / "invalid_call_rate.csv")
    assert len(rows) == 12
    assert list(rows[0]) == ["run", "seed", "step", "invalid_call_rate", "malformed_messages",
                             "assistant_messages", "valid_attempts"]
    for row in rows:
        malformed_step = {"grpo_a": 2, "grpo_b": 3}[row["run"]]             # the step with the reward 95
        lost = row["run"] == "grpo_b" and row["step"] == "5"
        if int(row["step"]) == malformed_step:
            assert (row["malformed_messages"], row["assistant_messages"]) == ("1", "33")
            assert float(row["invalid_call_rate"]) == pytest.approx(1 / 33, abs=1e-15)
        else:
            assert (row["malformed_messages"], row["assistant_messages"]) == ("0", "28" if lost else "32")
            assert float(row["invalid_call_rate"]) == 0.0
        assert row["valid_attempts"] == ("7" if lost else "8")


# ---------------------------------------------------------------------------
# The caption of the made-up runs
# ---------------------------------------------------------------------------
def test_the_caption_states_the_counts_the_ranges_and_the_adapter_of_step_0(figure, made_up, capsys):
    assert figure.main(arguments(made_up, "--no-image")) == 0
    caption = caption_of(made_up)
    assert capsys.readouterr().out.startswith(caption)                     # the caption is printed too
    wanted_sentences = (
        # the caption itself, four sentences; the lowest step mean is 0.49375, so the cut axis starts at 0.45
        "**Caption.** Learning curves of the 2 GRPO runs (seed 1 and seed 2; 6 steps each; a step is 2 "
        "training scenarios x 4 attempts = 8 attempts): for each step, (a) the mean reward (1.00 is the full "
        "reward), (b) the mean number of sampled tokens per attempt, and (c) the share of groups whose 4 "
        "rewards are exactly equal, each with a line that is the mean of 3 steps centred on its step; every "
        "y-axis starts at 0, except the lower part of (a), which shows the range 0.45 to 1.00 again and so "
        "makes small changes look larger. Panel (d) shows the pass rate of 3 dev families at the checkpoints "
        "of steps 0, 3 and 6, with a 95% interval from resampling the families; each point holds 6 attempts "
        "(one wording style, 2 attempts per family), and step 0 is the SFT adapter that both runs started "
        "from. The mean reward was",
        # the notes: one entry per panel
        "- **(a) Reward.** The mean reward of the 8 attempts of a step, divided by 100, so 1.00 is the full "
        "reward. Source: the field `mean_reward` of `train_log.jsonl`.",
        "- **(b) Completion length.** The mean number of sampled tokens per attempt. Source: the field "
        "`tokens_per_episode`",
        "- **(c) Zero-variance fraction.** The share of the 2 groups of a step whose 4 rewards are exactly "
        "equal. Source: the field `zero_variance_exact`",
        "- **(d) Held-out pass rate.** The family-weighted pass rate of 3 dev families at the checkpoints of "
        "steps 0, 3 and 6, with a 95% interval from resampling the 3 families. Each point holds 6 attempts: "
        "3 families in one wording style, 2 attempts per family. Source: `episodes.jsonl` of each point",
        f"The points of step 0 used the adapter `{START_ADAPTER}` (`adapter` in the run manifest of each "
        "point). This is the adapter that each GRPO run started from (`start_adapter` in the run manifest "
        "of the GRPO run): the SFT adapter, trained by `scripts/gpu/train_sft.py`. The SHA-256 of "
        "`adapter_model.safetensors` is the same in `chain_record.txt` of the point and in the manifest of "
        "the GRPO run.",
        # the highest length of a step is 60 tokens, so the axis of (b) ends at 100
        "Every y-axis starts at 0: (a), (c) and (d) go from 0 to 1, and (b) goes from 0 to 100 tokens. The "
        "one exception is the lower part of (a): it shows the range 0.45 to 1.00 of the upper part again.",
        # the mean reward of the first and of the last block of 3 steps, over the valid attempts:
        # A: (6 + 8 + 3.95) / 24 = 0.748 and (5 + 8 + 7) / 24 = 0.833
        # B: (7 + 8 + 5) / 24 = 0.833 and (3.95 + 8 + 6) / 23 = 0.780 (one attempt of 24 was lost)
        "The mean reward was 0.748 (seed 1) and 0.833 (seed 2) in steps 0 to 2, and 0.833 (seed 1) and "
        "0.780 (seed 2) in steps 3 to 5; the trainer kept no group in 2 of the 6 steps of seed 1 and in 2 of "
        "the 6 steps of seed 2.",
        "On the 3 dev families, 3 of 6 attempts passed at step 0 (both seeds), and 6 of 6 (seed 1) and 5 of "
        "6 (seed 2) at step 6.",
        "The harness lost 1 of the 96 attempts of the 12 steps.",
        "An attempt that fails has the reward 0. An attempt that passes has 100, minus 5 for each malformed "
        "message (at most 15), minus 2 for each turn over the budget (at most 10).",
        "One attempt that fails in place of a full reward lowers the mean of a step by 1/8 = 0.125.",
        "A message that the token cap (260 tokens per message, run manifests) cut off counts with the "
        "tokens up to the cap.",
        "Count: 2 groups of 4 attempts per point. A value of the line is a share of 6 groups.",
        "Over all steps the trainer kept 5 of the 12 groups of seed 1 and 5 of the 12 groups of seed 2",
        "In every step the kept groups are exactly the groups whose rewards are not all equal.",
        "In each of these steps the log has a loss of 0 and a gradient norm of 0",
        "These are the steps with the value 1.00 in the panel.",
        "the 3 dev families of the set `curve_subset` in wording style 1, 2 attempts per scenario, sampled "
        "with the run seed 0",
        "The wording style of the curve is one of them. So the curve tests new families, not a new wording "
        "style.",
        "In 1 of the 6 points every attempt passed. The interval then has no width",
        "So the points of step 0 are 2 rollouts of one model with one run seed. Attempts that passed: 3 of "
        "6 (seed 1) and 3 of 6 (seed 2).",
        "The points of the steps 3 and 6 used the checkpoint that the run saved after that number of steps",
        "The points of the seeds are drawn 3.2 steps apart",
        "It is drawn only where all these steps exist: from step 1 to step 4.",
        "the 6 steps of seed 1 used 12 different scenarios and the 6 steps of seed 2 used 12 different "
        "scenarios",
        "They gave the logged values of `n_episodes`, `n_harness_errors`, `mean_reward`, "
        "`tokens_per_episode`, `turns_per_episode`, `zero_variance_exact`, `invalid_call_rate` and "
        "`kept_groups` in all 12 steps (96 attempts).",
        "with a window of 3 steps, 2,000 bootstrap draws and the bootstrap seed 0. It reads no test result.",
        # block of run A, steps 0 to 2: tokens (40 + 44 + 49.5) / 3 = 44.5; equal groups 1 + 2 + 1
        "| 1 | 0 to 2 | 24 | 0.748 | 44.5 | 4 of 6, 66.7% | 2 of 6 | 1 of 3 |",
        # block of run B, steps 3 to 5: tokens (53.625 * 8 + 56 * 8 + 60 * 7) / 23 = 1297 / 23 = 56.4
        "| 2 | 3 to 5 | 23 | 0.780 | 56.4 | 4 of 6, 66.7% | 2 of 6 | 1 of 3 |",
        # family pass rates (1, 0.5, 0): with 3 families a draw of three times the same family has the
        # chance 1 / 27 = 3.7%, more than 2.5%, so the interval reaches from the lowest to the highest
        f"| 1 | 0 | 3 of 6 | 50.0% | 0.0% to 100.0% | `{START_ADAPTER}` |",
        "| 2 | 6 | 5 of 6 | 83.3% | 50.0% to 100.0% | `made_up/grpo_b/checkpoints/step_00006/adapter` |",
        # the family is the unit: 4 of 5 valid attempts, and a pass rate of (1 + 1 + 0.5) / 3
        "| 2 | 3 | 4 of 5 | 83.3% | 50.0% to 100.0% | `made_up/grpo_b/checkpoints/step_00003/adapter` |",
        "The harness lost 1 of the 36 attempts of the 6 points.",
        "the run manifest names this hash", "Status of the run in its manifest: complete.")
    for wanted in wanted_sentences:
        assert wanted in caption, wanted
    assert not re.search(r"\d{4}-\d{2}-\d{2}", caption)        # no date: the same bytes every time
    assert str(made_up["out"]) not in caption                  # the output folder is not named
    assert "whole dev set" not in caption                      # the made-up runs have no folder of tables
    assert len(caption_sentences(caption)) == 4                # the task asks for three or four sentences


def caption_sentences(caption: str) -> list[str]:
    """The sentences of the caption paragraph (the line that starts with **Caption.**). A sentence ends
    with a full stop and a space or the end of the line; numbers such as 0.45 have no space after the
    stop, and (a), (b) end with a parenthesis."""
    paragraph = next(line for line in caption.splitlines() if line.startswith("**Caption.**"))
    body = paragraph[len("**Caption.**"):].strip()
    return [part for part in re.split(r"(?<=\.)\s+", body) if part]


def test_the_caption_paragraph_has_three_or_four_sentences_and_keeps_the_qualifiers(figure, made_up):
    """Review finding: the caption paragraph had 9 sentences. It must have three or four, and still name
    the cut axis of (a), the adapter of step 0, and the attempts per point of (d)."""
    assert figure.main(arguments(made_up, "--no-image")) == 0
    sentences = caption_sentences(caption_of(made_up))
    assert 3 <= len(sentences) <= 4, sentences
    assert "range 0.45 to 1.00 again" in sentences[0] and "mean of 3 steps" in sentences[0]
    assert "step 0 is the SFT adapter" in sentences[1] and "6 attempts" in sentences[1]
    assert "the trainer kept no group" in sentences[2]
    for run in ("a",):                                     # one run alone: still at most four sentences
        shutil.rmtree(made_up["out"])
        assert figure.main(arguments(made_up, "--no-image", runs=(run,))) == 0
        assert 3 <= len(caption_sentences(caption_of(made_up))) <= 4


def test_one_run_alone_and_a_run_without_a_dev_curve(figure, made_up):
    for folder in made_up["a"].parent.glob("grpo_a_curve_*"):
        shutil.rmtree(folder)
    assert figure.main(arguments(made_up, "--no-image", runs=("a",))) == 0
    caption = caption_of(made_up)
    assert "# Figure 1. Learning curves of the 1 GRPO run\n" in caption
    assert "Learning curves of the 1 GRPO run (seed 1; 6 steps; a step is 2 training" in caption
    assert "Panel (d) is empty: no point of the dev curve was found." in caption
    assert "(d) Held-out pass rate" not in caption
    assert "One run describes one checkpoint." in caption and "dev families" not in caption
    rows = read_csv(made_up["out"] / "learning_curves.csv")
    assert len(rows) == 6 and all(row["dev_pass_rate"] == "" for row in rows)


def test_a_step_0_point_from_another_adapter_is_named_and_not_called_the_start_adapter(figure, made_up):
    for name in ("a", "b"):
        write_curve_point(made_up[name], 0, CURVE_A[0], adapter="made_up/other/adapter")
    assert figure.main(arguments(made_up, "--no-image")) == 0
    caption = caption_of(made_up)
    assert ("The points of step 0 used the adapter `made_up/other/adapter` (`adapter` in the run manifest "
            "of each point).") in caption
    assert "each point holds 6 attempts (one wording style, 2 attempts per family). The mean reward" in caption
    assert "started from" not in caption and "rollouts of one model" not in caption
    assert "- SFT:" not in caption


def test_the_caption_calls_an_adapter_the_start_of_both_runs_only_if_both_manifests_say_so(figure,
                                                                                         made_up):
    """Run B says that it started from another adapter. Its point of step 0 still names the adapter of
    run A, so the caption names that adapter and makes no claim about where the runs started."""
    write_manifest(made_up["b"], 2, start_adapter={"folder": "made_up/other/adapter", "files": {}})
    assert figure.main(arguments(made_up, "--no-image")) == 0
    caption = caption_of(made_up)
    assert f"The points of step 0 used the adapter `{START_ADAPTER}` (`adapter` in the run" in caption
    assert "started from" not in caption and "- SFT:" not in caption


def test_without_saved_attempts_the_caption_says_that_the_log_was_not_checked(figure, made_up):
    shutil.rmtree(made_up["a"] / "rollouts")
    shutil.rmtree(made_up["b"] / "rollouts")
    assert figure.main(arguments(made_up, "--no-image")) == 0
    caption = caption_of(made_up)
    assert "Check of the log: none. The run folders have no folder `rollouts`" in caption
    assert "They gave the logged values" not in caption


# ---------------------------------------------------------------------------
# Refusals: exit code 2, the reason on stderr, nothing written
# ---------------------------------------------------------------------------
def change_log(run: Path, change) -> None:
    """Change the lines of a made-up log, and write the manifest again so that it names the new log."""
    lines = (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    write_lines(run / "train_log.jsonl", change([json.loads(line) for line in lines]))
    seed = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["seed"]
    write_manifest(run, seed)


def refused(figure, made_up, capsys, *more: str, runs: tuple[str, ...] = ("a", "b")) -> str:
    assert figure.main(arguments(made_up, "--no-image", *more, runs=runs)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("make_figure: ")
    assert not made_up["out"].exists()
    return captured.err


def test_a_log_that_is_not_the_log_of_the_manifest_is_refused(figure, made_up, capsys):
    with (made_up["a"] / "train_log.jsonl").open("ab") as handle:
        handle.write(b"\n")
    assert "is not the log that the run manifest names" in refused(figure, made_up, capsys)


def test_a_logged_value_that_the_saved_attempts_do_not_give_is_refused(figure, made_up, capsys):
    def raise_reward(rows):
        rows[3]["mean_reward"] = 0.75                       # the attempts of step 3 give 0.625
        return rows
    change_log(made_up["a"], raise_reward)
    err = refused(figure, made_up, capsys)
    assert "does not agree with its saved attempts (1 difference)" in err
    assert "step 3, mean_reward: the log has 0.75, the saved attempts give 0.625" in err


@pytest.mark.parametrize("key, value, made", [
    ("zero_variance_exact", 1.0, "0.5"),                    # step 0: one group of two has equal rewards
    ("kept_groups", 2, "1"),
    ("tokens_per_episode", 41.0, "40.0"),
    ("invalid_call_rate", 0.03125, "0.0"),                  # 1 of 32 messages: a whole number, so it is read
])
def test_each_plotted_field_is_checked_against_the_saved_attempts(figure, made_up, capsys, key, value,
                                                                   made):
    def change(rows):
        rows[0][key] = value
        return rows
    change_log(made_up["a"], change)
    assert f"step 0, {key}: the log has {value!r}, the saved attempts give {made}" in \
        refused(figure, made_up, capsys)


def test_a_log_with_a_missing_step_or_a_cut_line_is_refused(figure, made_up, capsys):
    change_log(made_up["a"], lambda rows: rows[:2] + rows[3:])
    assert "must hold the steps 0, 1, 2, ... each once and in order; it holds [0, 1, 3, 4, 5]" in \
        refused(figure, made_up, capsys)
    text = (made_up["b"] / "train_log.jsonl").read_text(encoding="utf-8")
    (made_up["b"] / "train_log.jsonl").write_text(text[:-40], encoding="utf-8")
    write_manifest(made_up["b"], 2)
    assert "line 6 is not complete JSON" in refused(figure, made_up, capsys, runs=("b",))


def test_a_step_with_another_number_of_attempts_or_without_its_file_is_refused(figure, made_up, capsys):
    def fewer(rows):
        rows[1]["n_episodes"] = 7
        return rows
    change_log(made_up["a"], fewer)
    assert "step 1 has 7 attempts, not 2 groups x 4 attempts = 8" in refused(figure, made_up, capsys)
    (made_up["b"] / "rollouts" / "step_00004.episodes.jsonl").unlink()
    assert "step_00004.episodes.jsonl does not exist" in refused(figure, made_up, capsys, runs=("b",))


def test_a_folder_that_is_not_a_grpo_run_is_refused(figure, made_up, capsys):
    (made_up["a"] / "run_manifest.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
    assert "is not the manifest of a GRPO run" in refused(figure, made_up, capsys)
    (made_up["a"] / "run_manifest.json").unlink()
    assert "run_manifest.json does not exist" in refused(figure, made_up, capsys)


def test_two_runs_with_one_seed_four_runs_and_an_even_window_are_refused(figure, made_up, capsys):
    write_manifest(made_up["b"], 1)
    assert "two runs have the same seed: [1, 1]" in refused(figure, made_up, capsys)
    assert "give 1 to 3 run folders, not 4" in refused(figure, made_up, capsys, runs=("a", "b", "a", "b"))
    assert "a run folder is given twice" in refused(figure, made_up, capsys, runs=("a", "a"))
    write_manifest(made_up["b"], 2)
    assert "--window must be an odd number of at least 1, not 4" in \
        refused(figure, made_up, capsys, "--window", "4")


def test_a_curve_point_that_did_not_use_the_checkpoint_of_its_step_is_refused(figure, made_up, capsys):
    write_curve_point(made_up["a"], 3, CURVE_A[3], adapter="made_up/grpo_a/checkpoints/step_00006/adapter")
    err = refused(figure, made_up, capsys)
    assert "the point of step 3 did not use the checkpoint of that step" in err
    assert "(grpo_a/checkpoints/step_00003/adapter)" in err
    # the checkpoint of another run is not the checkpoint of this run either
    write_curve_point(made_up["a"], 3, CURVE_A[3], adapter="made_up/grpo_b/checkpoints/step_00003/adapter")
    assert "did not use the checkpoint of that step" in refused(figure, made_up, capsys)


def test_a_curve_point_that_is_not_as_its_manifest_says_is_refused(figure, made_up, capsys):
    sets = made_up["scenarios"] / "sets.json"
    sets.write_text(json.dumps({"curve_subset": ["dv-000", "dv-001", "dv-009"]}), encoding="utf-8")
    assert "the families are not the list 'curve_subset'" in refused(figure, made_up, capsys)
    sets.write_text(json.dumps({"curve_subset": DEV_FAMILIES}), encoding="utf-8")
    write_curve_point(made_up["a"], 3, CURVE_A[3], samples=4)
    assert "not every scenario has the 4 attempts of the manifest" in refused(figure, made_up, capsys)
    write_curve_point(made_up["a"], 3, CURVE_A[3], complete=False)
    assert "does not say that the rollout is complete" in refused(figure, made_up, capsys)
    write_curve_point(made_up["a"], 3, CURVE_A[3], run_seed=7)
    assert "the attempts do not all have the run seed 7" in refused(figure, made_up, capsys)
    write_curve_point(made_up["a"], 3, CURVE_A[3],
                      command=["scripts/gpu/rollout.py", "--set", "curve_subset", "--styles", "1,2"])
    assert "the attempts are not in the wording styles (1, 2) of the manifest" in \
        refused(figure, made_up, capsys)


def test_a_curve_point_of_another_split_is_refused_and_its_split_file_is_not_opened(figure, made_up,
                                                                                      capsys):
    """A rollout on the test split under the name of a curve point: refused by its manifest, before its
    attempts are read. The file test.jsonl of the scenario folder cannot be read here (it is a folder),
    so opening it would raise."""
    (made_up["scenarios"] / "test.jsonl").mkdir()
    write_curve_point(made_up["a"], 3, CURVE_A[3], split="test")
    err = refused(figure, made_up, capsys)
    assert "the rollout is of the split 'test'; the curve of the figure is made on the dev split only" in err


def test_curve_points_with_different_coverage_are_refused(figure, made_up, capsys):
    """Every point of run B gets a third attempt per scenario: each point agrees with its own manifest,
    but the points of the two runs do not have the same coverage."""
    for step, passes in CURVE_B.items():
        folder = write_curve_point(made_up["b"], step, passes, samples=3)
        records = [make_record(family_id, 1, sample, arm=folder.name)
                   for family_id in DEV_FAMILIES for sample in range(3)]
        write_lines(folder / "episodes.jsonl", records)
    assert "the points of the dev curve do not all have the same coverage" in \
        refused(figure, made_up, capsys)


def test_the_hash_of_the_dev_file_is_compared_when_the_file_is_there(figure, made_up, capsys):
    """The made-up attempts carry no data_sha256. With a dev.jsonl in the scenario folder they are
    refused; with the hash of that file in every attempt they are taken."""
    dev = made_up["scenarios"] / "dev.jsonl"
    dev.write_text('{"family_id": "dv-000"}\n', encoding="utf-8")
    assert "the attempts were not made from" in refused(figure, made_up, capsys)
    for name in ("a", "b"):
        for folder in made_up[name].parent.glob(f"{made_up[name].name}_curve_*"):
            lines = (folder / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            write_lines(folder / "episodes.jsonl",
                        [{**json.loads(line), "data_sha256": sha256(dev)} for line in lines])
    assert figure.main(arguments(made_up, "--no-image")) == 0


def test_a_step_0_point_with_another_weight_file_than_the_start_adapter_is_refused(figure, made_up,
                                                                                  capsys):
    record = made_up["a"].parent / "grpo_a_curve_000" / "chain_record.txt"
    record.write_text("adapter_model.safetensors=" + "cd" * 32 + "\n", encoding="utf-8")
    assert "chain_record.txt has another SHA-256 of adapter_model.safetensors" in \
        refused(figure, made_up, capsys)


def test_without_matplotlib_only_the_data_can_be_written(figure, made_up, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)                   # `import matplotlib` now fails
    assert figure.main(arguments(made_up)) == 2
    err = capsys.readouterr().err
    assert "matplotlib is not installed" in err and "scripts/requirements-report.txt" in err
    assert "--no-image" in err and not made_up["out"].exists()
    assert figure.main(arguments(made_up, "--no-image")) == 0
    assert sorted(path.name for path in made_up["out"].iterdir()) == sorted(DATA_FILES)


def test_the_requirement_file_pins_one_version_of_matplotlib_and_nothing_else(figure):
    path = ROOT / figure.REQUIREMENTS
    wanted = [line for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]
    assert len(wanted) == 1 and re.fullmatch(r"matplotlib==\d+\.\d+\.\d+", wanted[0])
    # a file "requirements*.txt" in the repository root would be a difference from MANIFEST.json
    assert not (ROOT / "requirements-report.txt").exists()
    # the saved image was made with this version: another one can give other bytes
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib.__version__ == wanted[0].split("==")[1]


# ---------------------------------------------------------------------------
# The image of the made-up runs (needs matplotlib)
# ---------------------------------------------------------------------------
def png_size(path: Path) -> tuple[int, int]:
    """(width, height) in pixels, from the header of a PNG file."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def pdf_width_cm(path: Path) -> float:
    box = re.search(rb"/MediaBox \[ 0 0 ([0-9.]+) ([0-9.]+) \]", path.read_bytes())
    return float(box.group(1)) / 72 * 2.54


def text_boxes(figure_module, data) -> tuple[list[tuple[str, object]], float, float]:
    """(text, box in pixels) of every text of the drawn figure, and the size of the canvas in pixels."""
    from matplotlib import pyplot as plt
    from matplotlib.text import Text

    fig = figure_module.figure_of(data)
    try:
        fig.set_dpi(figure_module.PNG_WIDTH_PX / (figure_module.FIG_WIDTH_CM / 2.54))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in fig.findobj(Text)
                 if text.get_visible() and text.get_text().strip()]
        return boxes, fig.bbox.width, fig.bbox.height
    finally:
        plt.close(fig)


def test_the_same_inputs_give_the_same_bytes_and_the_image_has_its_size(figure, made_up):
    pytest.importorskip("matplotlib")
    assert figure.main(arguments(made_up)) == 0
    first = {name: (made_up["out"] / name).read_bytes() for name in FILES}
    shutil.rmtree(made_up["out"])
    assert figure.main(arguments(made_up)) == 0
    assert {name: (made_up["out"] / name).read_bytes() for name in FILES} == first
    assert png_size(made_up["out"] / "learning_curves.png")[0] == 2000
    assert pdf_width_cm(made_up["out"] / "learning_curves.pdf") == pytest.approx(16.0, abs=0.001)
    pdf = (made_up["out"] / "learning_curves.pdf").read_bytes()
    assert b"CreationDate" not in pdf and b"DejaVuSans" in pdf             # no date; the font is in the file


@pytest.mark.parametrize("names", [("a",), ("a", "b"), ("a", "b", "c")])
def test_no_text_of_the_made_up_figure_leaves_the_page(figure, made_up, names):
    pytest.importorskip("matplotlib")
    made_up["c"] = write_run(made_up["a"].parent / "grpo_c", 3, REWARDS_A, BY_HAND_A, CURVE_B)
    data = figure.build([made_up[name] for name in names], made_up["scenarios"], window=3, n_draws=500)
    boxes, width, height = text_boxes(figure, data)
    assert len(boxes) > 40
    for text, box in boxes:
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= width and box.y1 <= height, text
    shown = [text for text, _ in boxes]
    for title in ("(a) Reward", "(b) Completion length", "(c) Zero-variance fraction",
                  "(d) Held-out pass rate"):
        assert title in shown
    assert [text for text in shown if text.startswith("GRPO seed")] == \
        [f"GRPO seed {index + 1}" for index in range(len(names))]          # one legend entry per run
    assert "3 dev families, 6 attempts per point, 95% intervals" in shown
    # review finding: the figure alone says that step 0 of (d) is the SFT adapter, and that the seeds are
    # drawn side by side
    assert "step 0 = the SFT adapter" in shown
    assert ("the seeds are drawn side by side,\n3.2 steps apart, at each checkpoint" in shown) == (len(names) > 1)
    assert "whole scale, 0 to 1; grey band: 0.45 to 1.00" in shown
    assert "the grey band again:\ny-axis cut, 0.45 to 1.00" in shown


def test_the_count_labels_of_panel_d(figure, made_up):
    data = figure.build([made_up["a"], made_up["b"]], made_up["scenarios"], window=3, n_draws=500)
    at_first, at_last = figure.curve_labels(data["runs"])
    assert at_first == ["3 of 6 passed (both seeds)"]                       # one label where the counts agree
    assert at_last == ["seed 1: 6 of 6 passed", "seed 2: 5 of 6 passed"]
    alone = figure.build([made_up["b"]], made_up["scenarios"], window=3, n_draws=500)
    assert figure.curve_labels(alone["runs"]) == (["seed 2: 3 of 6 passed"], ["seed 2: 5 of 6 passed"])


# ---------------------------------------------------------------------------
# The saved runs of results/main/
# ---------------------------------------------------------------------------
RECORDER = """
import json, os, runpy, sys
opened = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        opened.append(os.path.abspath(os.fsdecode(args[0])))
sys.addaudithook(hook)
script, record, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    with open(record, "w", encoding="utf-8") as handle:
        json.dump(opened, handle)
"""


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> dict:
    """The script, run once as a command on the saved runs, into a temporary folder. The process records
    every file that it opens. Skipped if the saved runs are not there."""
    if not all((run / "train_log.jsonl").is_file() for run in SAVED_RUNS):
        pytest.skip("the saved GRPO runs are not in results/main/")
    pytest.importorskip("matplotlib")
    folder = tmp_path_factory.mktemp("figure_of_saved_runs")
    out, record = folder / "out", folder / "opened.json"
    command = [sys.executable, "-c", RECORDER, str(ROOT / "scripts" / "make_figure.py"), str(record),
               "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=folder)
    assert result.returncode == 0, result.stderr
    return {"out": out, "stdout": result.stdout, "opened": json.loads(record.read_text(encoding="utf-8")),
            "rows": read_csv(out / "learning_curves.csv")}


def lines_of(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def step_by_hand(records: list[dict], group_size: int) -> dict:
    """The plotted numbers of one step, from its saved attempts, with plain code: the groups are the
    blocks of `group_size` attempts of the file, and each block is one scenario."""
    groups = [records[start:start + group_size] for start in range(0, len(records), group_size)]
    assert all(len({record["scenario_id"] for record in group}) == 1 for group in groups)
    valid = [record for record in records if record["harness_error"] is None]
    rewards = [[record["reward"] for record in group if record["harness_error"] is None]
               for group in groups]
    equal = [len(set(group)) == 1 for group in rewards]
    kept = [not same and 2 * len(group) >= group_size for group, same in zip(rewards, equal)]
    return {"mean_reward": sum(record["reward"] for record in valid) / 100 / len(valid),
            "tokens": sum(turn["n_tokens"] for record in valid for turn in record["turns"]) / len(valid),
            "zero": sum(equal) / len(groups), "kept": sum(kept),
            "malformed": sum(record["n_malformed"] for record in valid),
            "messages": sum(len(record["turns"]) for record in valid), "valid": len(valid)}


def test_saved_runs_every_value_of_panels_a_to_c_can_be_made_again(saved):
    assert len(saved["rows"]) == 2 * 81                    # the steps 0 to 79 and the last checkpoint
    invalid = read_csv(saved["out"] / "invalid_call_rate.csv")
    assert len(invalid) == 2 * 80
    for run, seed in zip(SAVED_RUNS, (1, 2)):
        mine = [row for row in saved["rows"] if row["run"] == run.name]
        assert [int(row["step"]) for row in mine] == list(range(81))
        assert {row["seed"] for row in mine} == {str(seed)}
        log = lines_of(run / "train_log.jsonl")
        assert [line["step"] for line in log] == list(range(80))
        from_attempts = []
        for line, row in zip(log, mine):
            path = run / "rollouts" / f"step_{line['step']:05d}.episodes.jsonl"
            again = step_by_hand(lines_of(path), 16)
            from_attempts.append(again)
            # the CSV holds the logged value as it is ...
            assert float(row["mean_reward"]) == line["mean_reward"]
            assert float(row["tokens_per_attempt"]) == line["tokens_per_episode"]
            assert float(row["zero_variance_fraction"]) == line["zero_variance_exact"]
            assert (int(row["kept_groups"]), int(row["groups"])) == (line["kept_groups"], 8)
            assert int(row["attempts"]) == 128
            # ... and the saved attempts of the step give it again
            assert int(row["valid_attempts"]) == again["valid"]
            assert float(row["mean_reward"]) == pytest.approx(again["mean_reward"], abs=1e-12)
            assert float(row["tokens_per_attempt"]) == pytest.approx(again["tokens"], abs=1e-9)
            assert float(row["zero_variance_fraction"]) == again["zero"]
            assert int(row["kept_groups"]) == again["kept"]
        for column, key in (("mean_reward", "mean_reward"), ("tokens_per_attempt", "tokens"),
                            ("zero_variance_fraction", "zero")):
            values = [again[key] for again in from_attempts]
            for step, row in enumerate(mine[:80]):
                if 4 <= step <= 75:                        # the 9 steps that are centred on the step
                    by_hand = sum(values[step - 4:step + 5]) / 9
                    assert float(row[column + "_ma9"]) == pytest.approx(by_hand, abs=1e-9)
                else:
                    assert row[column + "_ma9"] == ""
        for column in ("attempts", "mean_reward", "mean_reward_ma9", "tokens_per_attempt",
                       "zero_variance_fraction", "kept_groups"):
            assert mine[80][column] == ""                  # the last checkpoint has no training step
        of_run = [row for row in invalid if row["run"] == run.name]
        for line, again, row in zip(log, from_attempts, of_run):
            assert int(row["step"]) == line["step"]
            assert float(row["invalid_call_rate"]) == line["invalid_call_rate"]
            assert int(row["malformed_messages"]) == again["malformed"]
            assert int(row["assistant_messages"]) == again["messages"]
            assert float(row["invalid_call_rate"]) == pytest.approx(again["malformed"] / again["messages"],
                                                                    abs=1e-15)


def test_saved_runs_every_value_of_panel_d_can_be_made_again(saved):
    sets = json.loads((ROOT / "data" / "scenarios" / "sets.json").read_text(encoding="utf-8"))
    for run in SAVED_RUNS:
        mine = {int(row["step"]): row for row in saved["rows"] if row["run"] == run.name}
        assert [step for step, row in mine.items() if row["dev_pass_rate"]] == [0, 20, 40, 60, 80]
        for step in (0, 20, 40, 60, 80):
            folder = run.parent / f"{run.name}_curve_{step:03d}"
            records = lines_of(folder / "episodes.jsonl")
            manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
            by_family: dict[str, list[bool]] = {}
            for record in records:
                assert record["harness_error"] is None and record["style"] == 1 and record["split"] == "dev"
                by_family.setdefault(record["family_id"], []).append(record["passed"])
            assert sorted(by_family) == sorted(sets["curve_subset"]) and len(by_family) == 20
            assert all(len(passes) == 4 for passes in by_family.values()) and len(records) == 80
            rates = [sum(by_family[family]) / 4 for family in sorted(by_family)]
            low, high = bootstrap_by_hand(rates, 10_000, 0)
            row = mine[step]
            assert float(row["dev_pass_rate"]) == pytest.approx(sum(rates) / 20, abs=1e-12)
            assert float(row["dev_ci_low"]) == pytest.approx(low, abs=1e-12)
            assert float(row["dev_ci_high"]) == pytest.approx(high, abs=1e-12)
            assert (int(row["dev_attempts"]), int(row["dev_valid_attempts"])) == (80, 80)
            assert int(row["dev_families"]) == 20
            assert int(row["dev_passed"]) == sum(sum(passes) for passes in by_family.values())
            assert row["dev_adapter"] == manifest["adapter"]
            assert manifest["adapter"] == ("results/main/sft/adapter" if step == 0 else
                                           f"results/main/{run.name}/checkpoints/step_{step:05d}/adapter")


def test_saved_runs_the_caption_has_the_counts_that_the_log_gives(saved):
    caption = (saved["out"] / "caption.md").read_text(encoding="utf-8")
    assert saved["stdout"].startswith(caption)
    without, kept, first, last = [], [], [], []
    for run in SAVED_RUNS:
        log = lines_of(run / "train_log.jsonl")
        assert all(line["n_harness_errors"] == 0 and line["n_episodes"] == 128 for line in log)
        without.append(sum(1 for line in log if line["kept_groups"] == 0))
        kept.append(sum(line["kept_groups"] for line in log))
        first.append(sum(line["mean_reward"] for line in log[:20]) / 20)   # every step: 128 valid attempts
        last.append(sum(line["mean_reward"] for line in log[-20:]) / 20)
    passed = {(row["seed"], row["step"]): row["dev_passed"] for row in saved["rows"] if row["dev_passed"]}
    assert passed["1", "0"] == passed["2", "0"] and passed["1", "80"] != passed["2", "80"]
    wanted_sentences = (
        f"in steps 60 to 79; the trainer kept no group in {without[0]} of the 80 steps of seed 1 and in "
        f"{without[1]} of the 80 steps of seed 2.",
        f"Over all steps the trainer kept {kept[0]} of the 640 groups of seed 1 and {kept[1]} of the 640 "
        "groups of seed 2",
        f"The mean reward was {first[0]:.3f} (seed 1) and {first[1]:.3f} (seed 2) in steps 0 to 19, and "
        f"{last[0]:.3f} (seed 1) and {last[1]:.3f} (seed 2) in steps 60 to 79;",
        "Panel (d) shows the pass rate of 20 dev families at the checkpoints of steps 0, 20, 40, 60 and 80, "
        "with a 95% interval from resampling the families; each point holds 80 attempts (one wording style, "
        "4 attempts per family), and step 0 is the SFT adapter that both runs started from.",
        "every y-axis starts at 0, except the lower part of (a), which shows the range 0.85 to 1.00 again "
        "and so makes small changes look larger.",
        # step 0: one model and one run seed, so one count for both seeds (checked above the list)
        f"On the 20 dev families, {passed['1', '0']} of 80 attempts passed at step 0 (both seeds), and "
        f"{passed['1', '80']} of 80 (seed 1) and {passed['2', '80']} of 80 (seed 2) at step 80.",
        "Learning curves of the 2 GRPO runs (seed 1 and seed 2; 80 steps each; a step is 8 training "
        "scenarios x 16 attempts = 128 attempts):",
        "The points of step 0 used the adapter `results/main/sft/adapter` (`adapter` in the run manifest of "
        "each point). This is the adapter that each GRPO run started from (`start_adapter` in the run "
        "manifest of the GRPO run): the SFT adapter, trained by `scripts/gpu/train_sft.py`.",
        "the mean of 9 steps centred on its step",
        "it shows the range 0.85 to 1.00 of the upper part again",
        "(b) goes from 0 to 250 tokens",
        "Each point holds 80 attempts: 20 families in one wording style, 4 attempts per family.",
        "in all 160 steps (20,480 attempts)",
        "The harness lost 0 of the 20,480 attempts of the 160 steps.",
        "The pass rates of the whole dev set are in `results/main/tables/`.",
        "It reads no test result.")
    for wanted in wanted_sentences:
        assert wanted in caption, wanted
    assert len(caption_sentences(caption)) == 4


def test_saved_runs_the_two_tables_of_the_caption_can_be_made_again(saved):
    """Every row of the table by block of 20 steps, from the log with plain code, and every row of the
    table of the dev curve, from the saved attempts of each point. Every step has 128 valid attempts and 8
    groups, so a mean over the steps of a block is the mean over its attempts."""
    caption = (saved["out"] / "caption.md").read_text(encoding="utf-8")
    turns, used = [], []
    for run, seed in zip(SAVED_RUNS, (1, 2)):
        log = lines_of(run / "train_log.jsonl")
        for start in (0, 20, 40, 60):
            block = log[start:start + 20]
            equal = sum(round(line["zero_variance_exact"] * 8) for line in block)
            assert all(line["kept_groups"] == 8 - round(line["zero_variance_exact"] * 8) for line in block)
            row = (f"| {seed} | {start} to {start + 19} | 2,560 | "
                   f"{sum(line['mean_reward'] for line in block) / 20:.3f} | "
                   f"{sum(line['tokens_per_episode'] for line in block) / 20:.1f} | "
                   f"{equal} of 160, {100 * equal / 160:.1f}% | "
                   f"{sum(line['kept_groups'] for line in block)} of 160 | "
                   f"{sum(1 for line in block if line['kept_groups'] == 0)} of 20 |")
            assert row in caption, row
        turns.append(sum(line["turns_per_episode"] for line in log) / 80)
        used.append(len({scenario for line in log for scenario in line["scenarios"]}))
        for step in (0, 20, 40, 60, 80):
            records = lines_of(run.parent / f"{run.name}_curve_{step:03d}" / "episodes.jsonl")
            by_family: dict[str, list[bool]] = {}
            for record in records:
                by_family.setdefault(record["family_id"], []).append(record["passed"])
            rates = [sum(passes) / len(passes) for passes in by_family.values()]
            low, high = bootstrap_by_hand(rates, 10_000, 0)
            adapter = ("results/main/sft/adapter" if step == 0 else
                       f"results/main/{run.name}/checkpoints/step_{step:05d}/adapter")
            row = (f"| {seed} | {step} | {sum(record['passed'] for record in records)} of 80 | "
                   f"{100 * sum(rates) / 20:.1f}% | {100 * low:.1f}% to {100 * high:.1f}% | `{adapter}` |")
            assert row in caption, row
    assert (f"An attempt has {turns[0]:.2f} (seed 1) and {turns[1]:.2f} (seed 2) assistant messages on "
            "average") in caption
    assert (f"the 80 steps of seed 1 used {used[0]} different scenarios and the 80 steps of seed 2 used "
            f"{used[1]} different scenarios") in caption


def test_saved_runs_the_script_opens_no_test_file_no_token_file_and_no_sealed_result(saved):
    inside = [Path(path).relative_to(ROOT).as_posix() for path in saved["opened"]
              if Path(path).is_relative_to(ROOT) and not Path(path).is_relative_to(ROOT / ".venv")]
    data = [path for path in inside if path.startswith(("results/", "data/", "_sealed/"))]
    assert len(data) > 2 * 80 + 2 * 5                      # the logs, the attempts of every step, the curve
    assert sorted({path for path in data if path.startswith("data/")}) == \
        ["data/scenarios/dev.jsonl", "data/scenarios/sets.json"]
    for path in data:
        assert not re.search(r"(^|/)test\.jsonl$|_test(/|_|$)|test_run", path), path   # no test split or run
        assert "tokens" not in path and "_sealed" not in path and "/audit/" not in path, path
        assert path.startswith(("results/main/grpo_seed", "data/scenarios/")), path
    assert not [path for path in inside if path.endswith(".env")]
    # the outputs went to the temporary folder: nothing of them was opened inside the repository
    assert not [path for path in inside if path.startswith("results/main/report")]
    assert sorted(path.name for path in saved["out"].iterdir()) == sorted(FILES)


def test_saved_runs_the_image_has_its_size_and_no_text_leaves_the_page_or_covers_another(figure, saved):
    assert png_size(saved["out"] / "learning_curves.png") == (2000, 1800)
    assert pdf_width_cm(saved["out"] / "learning_curves.pdf") == pytest.approx(16.0, abs=0.001)
    data = figure.build(SAVED_RUNS, ROOT / "data" / "scenarios")
    boxes, width, height = text_boxes(figure, data)
    margin = 12                                            # pixels: about 1 mm on the page
    for text, box in boxes:
        assert box.x0 >= margin and box.y0 >= margin, text
        assert box.x1 <= width - margin and box.y1 <= height - margin, text
    for index, (text, box) in enumerate(boxes):
        for other, other_box in boxes[index + 1:]:
            apart = (box.x1 <= other_box.x0 + 1 or other_box.x1 <= box.x0 + 1
                     or box.y1 <= other_box.y0 + 1 or other_box.y1 <= box.y0 + 1)
            assert apart, (text, other)
    shown = [text for text, _ in boxes]
    without = [sum(1 for line in lines_of(run / "train_log.jsonl") if line["kept_groups"] == 0)
               for run in SAVED_RUNS]
    assert ("steps in which no group was kept (value 1.00):\n"
            f"seed 1: {without[0]} of 80 steps\nseed 2: {without[1]} of 80 steps") in shown
    passed = {(row["seed"], row["step"]): row["dev_passed"] for row in saved["rows"] if row["dev_passed"]}
    assert passed["1", "0"] == passed["2", "0"]            # one model and one run seed at step 0
    assert f"{passed['1', '0']} of 80 passed (both seeds)" in shown
    assert f"seed 1: {passed['1', '80']} of 80 passed" in shown
    assert f"seed 2: {passed['2', '80']} of 80 passed" in shown
    assert "In (a) to (c): a point is one training step; the line is the mean of 9 steps, centred." in shown


def test_saved_runs_the_files_in_results_are_the_ones_that_the_script_makes_now(saved):
    """results/main/report/figure/ holds what the script makes from the saved runs with the packages of
    this environment. If this fails after a change, run: .venv/bin/python scripts/make_figure.py"""
    if not SAVED_OUT.is_dir():
        pytest.skip("results/main/report/figure/ is not there yet")
    for name in FILES:
        assert (SAVED_OUT / name).read_bytes() == (saved["out"] / name).read_bytes(), name
