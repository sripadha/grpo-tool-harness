"""The one figure of the write-up: the learning curves of the GRPO runs, and the data behind it.

Written after the freeze. It only reads saved runs and writes into its own folder. It reads no test
result and no file of the test split.

    .venv/bin/python scripts/make_figure.py

What it writes (default folder: results/main/report/figure):
    learning_curves.png     the figure, 2000 pixels wide; it is drawn at its size on the page (16 cm wide)
    learning_curves.pdf     the same figure as a PDF page of 16 cm width
    learning_curves.csv     every plotted value: one row per seed and step
    invalid_call_rate.csv   the invalid-call rate of every step and seed (logged, not a panel)
    caption.md              the caption, and the rule, the unit, the count and the source of every number

The four panels, one line per GRPO seed, the same colour per seed in every panel:
    (a) reward                   `mean_reward` of train_log.jsonl: the mean reward / 100 of the attempts
                                 of a step
    (b) completion length        `tokens_per_episode`: sampled tokens of all assistant messages per attempt
    (c) zero-variance fraction   `zero_variance_exact`: the share of the groups of a step whose rewards are
                                 exactly equal (the rule by which the trainer drops a group)
    (d) held-out pass rate       the family-weighted pass rate of the dev curve (`<run>_curve_<step>`), with
                                 the interval of `metrics.family_bootstrap_ci`, as in
                                 accessdesk/eval/tables.py
In (a) to (c) a point is one step, and the line is the centred mean of --window steps (default 9). The
line is drawn only where the whole window fits.

What it reads:
    <run>/run_manifest.json   the seed, the groups per step, the attempts per group, the start adapter and
                              the SHA-256 of the log
    <run>/train_log.jsonl     one line per step (`rollout_stats` in accessdesk/train/grpo.py)
    <run>/rollouts/step_*.episodes.jsonl   only as a check: `rollout_stats` and `plan_update` are run
                              again on the saved attempts of every step and must give the logged values.
                              The token files (step_*.tokens.jsonl) are not read.
    <run>_curve_<step>/run_manifest.json, episodes.jsonl, chain_record.txt (if it is there)
    <scenarios>/sets.json     the list `curve_subset`; and the SHA-256 of <scenarios>/dev.jsonl

--run FOLDER    the folder of one GRPO run; give it once per run (at most 3). Default: results/main/grpo_seed1
                and results/main/grpo_seed2. The dev curve of a run is every folder beside it with the name
                <run>_curve_<digits>; the digits are the number of training steps before the checkpoint.
--scenarios     the scenario folder of the runs (default: data/scenarios)
--out           the folder for the five files (it is made if it is missing)
--window        the number of steps in the centred mean, an odd number (default 9)
--draws, --seed the bootstrap of panel (d) (default: 10,000 draws and seed 0, as ANALYSIS_PLAN.md section 4)
--no-image      write the two CSV files and the caption only. Without this option the script needs
                matplotlib: uv pip install --python .venv/bin/python -r scripts/requirements-report.txt

The files do not hold a date or a time: the same inputs give the same bytes with the same packages.

Exit code 0: the files were written. 2: nothing was written; the reason is on stderr. That happens if
  - a run folder has no run manifest or no train log, the log is not the file that the manifest names
    (SHA-256), a line of the log is not complete, the steps are not 0, 1, 2, ... each once, a step has
    another number of attempts than groups x attempts per group, or a step has no valid attempt;
  - the saved attempts of a step do not give the logged values;
  - a point of the dev curve is not of the dev split, is not complete, did not use the checkpoint of its
    step, holds other families than the set of its manifest, holds another number of attempts than its
    manifest says, or the points do not all have the same coverage (set, wording styles, attempts per
    scenario, run seed);
  - matplotlib is not there and --no-image was not given.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.eval import metrics  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402
from accessdesk.train import grpo  # noqa: E402

DEFAULT_RUNS = ("results/main/grpo_seed1", "results/main/grpo_seed2")
DEFAULT_OUT = "results/main/report/figure"
DEFAULT_SCENARIOS = "data/scenarios"
REQUIREMENTS = "scripts/requirements-report.txt"
PNG_NAME, PDF_NAME = "learning_curves.png", "learning_curves.pdf"
CSV_NAME, INVALID_NAME, CAPTION_NAME = "learning_curves.csv", "invalid_call_rate.csv", "caption.md"

WINDOW = 9                      # steps in the centred mean of panels (a) to (c)
BLOCK = 20                      # steps in one block of the table of the caption
N_DRAWS, BOOT_SEED, LEVEL = 10_000, 0, 0.95      # the interval of panel (d): ANALYSIS_PLAN.md section 4
SLACK = 1e-9                    # two numbers that should be equal up to rounding
MAX_RUNS = 3                    # more lines cannot be told apart in panels of this size
# The fields of a log line that are made again from the saved attempts of the step (`check_rollouts`).
CHECKED = ("n_episodes", "n_harness_errors", "mean_reward", "tokens_per_episode", "turns_per_episode",
           "zero_variance_exact", "invalid_call_rate", "kept_groups")
# The fields of a log line that the figure and the two tables need.
NEEDED = ("mean_reward", "tokens_per_episode", "turns_per_episode", "zero_variance_exact",
          "invalid_call_rate", "kept_groups")
REWARD_KEYS = ("pass_reward", "malformed_penalty", "malformed_cap", "turn_penalty", "turn_cap")
CURVE_FOLDER = "{run}_curve_"   # the dev curve of a run: the folders <run>_curve_<digits> beside it
CURVE_SPLIT = "dev"             # the curve is made on dev families only; no other split file is opened
SFT_SCRIPT = "scripts/gpu/train_sft.py"
WEIGHT_FILE = "adapter_model.safetensors"

# The figure. It is drawn at the size that it has on the page, so a font size here is the size in print.
FIG_WIDTH_CM, FIG_HEIGHT_CM = 16.0, 14.4
PNG_WIDTH_PX = 2000
FONT = "DejaVu Sans"            # inside the matplotlib package: the same text on every machine
# Colours: the first three slots of a palette that was checked for colour-blind readers in every pair
# (blue, orange, aqua), on a white page. Text never has a series colour.
SERIES = (
    {"color": "#2a78d6", "marker": "o", "filled": True},
    {"color": "#eb6834", "marker": "s", "filled": False},
    {"color": "#1baf7a", "marker": "^", "filled": True},
)
INK, INK_SECOND = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE, BAND = "#e1e0d9", "#c3c2b7", "#ffffff", "#f3f2ee"
DODGE = 3.2                     # panel (d): the points of two seeds are this many steps apart


class FigureError(ValueError):
    """The figure must not be made from these files. The script stops with exit code 2."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def shown(path: Path | str) -> str:
    """The path as the outputs show it: relative to the repository root, with "/", if it is inside the
    repository; else as it was given."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_of_file(path: Path) -> str:
    """The SHA-256 of the file's bytes, read in pieces of 1 MB."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """The JSON object of a file. Raises FigureError if the file is missing or is not one JSON object."""
    if not path.is_file():
        raise FigureError(f"{shown(path)} does not exist")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise FigureError(f"{shown(path)} is not JSON: {error}") from None
    if not isinstance(value, dict):
        raise FigureError(f"{shown(path)} does not hold one JSON object")
    return value


def centred_mean(values: list[float], window: int) -> list[float | None]:
    """For each position: the mean of the `window` values that are centred on it (the value itself and
    (window - 1) / 2 values on each side). None where the window does not fit, so the first and the last
    (window - 1) / 2 positions are None. `window` is an odd number of at least 1."""
    if window < 1 or window % 2 == 0:
        raise FigureError(f"the window of the centred mean must be an odd number of at least 1, not {window}")
    half = window // 2
    return [math.fsum(values[i - half:i + half + 1]) / window if half <= i < len(values) - half else None
            for i in range(len(values))]


def whole_number(value: float, what: str) -> int:
    """`value` as a whole number. Raises FigureError if it is not one up to rounding."""
    nearest = round(value)
    if abs(value - nearest) > 1e-6:
        raise FigureError(f"{what} should be a whole number, but it is {value!r}")
    return int(nearest)


def percent(value: float) -> str:
    """0.8625 -> "86.2%": one decimal, formatted as the frozen tables format a share."""
    return f"{100.0 * value:.1f}%"


def in_words(items: list[str]) -> str:
    """["a", "b", "c"] -> "a, b and c"."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# One GRPO run: the log of its steps
# ---------------------------------------------------------------------------
def read_run(folder: Path) -> dict:
    """The log of one GRPO run, after every check of the module text. Keys of the result:

        folder, name            the run folder and its last part
        seed                    the GRPO seed (`seed` of run_manifest.json)
        n_scenarios, group_size, tie_tol    groups per step, attempts per group, and the tie tolerance
        planned_steps, status   `grpo_config.steps` and `status` of the manifest
        start_adapter           `start_adapter` of the manifest (folder, files with their SHA-256, trained_by)
        styles, split           the wording styles and the split that the run trained on
        reward_numbers          the reward numbers of the run config (REWARD_KEYS), or None
        fail_is_zero            the run config says that a failed attempt has no penalty: its reward is 0
        token_cap               `max_new_tokens` of the run config: the most tokens of one message, or None
        scenarios_used          the number of different scenario ids in the field `scenarios` of the log
                                lines, or None if a line has no such field
        log, log_sha256         the log file and its SHA-256; log_in_manifest: the manifest names this hash
        rows                    the log lines in step order, each with these keys added: `valid` (attempts
                                without a harness error), `n_messages`, `n_malformed`, and `n_equal` (the
                                groups with equal rewards: zero_variance_exact x groups, or None if that
                                is not a whole number)
    """
    folder = Path(folder)
    manifest = read_json(folder / "run_manifest.json")
    config = manifest.get("grpo_config") or {}
    missing = [key for key in ("n_scenarios", "group_size", "tie_tol", "steps") if key not in config]
    if missing or "seed" not in manifest:
        raise FigureError(f"{shown(folder / 'run_manifest.json')} is not the manifest of a GRPO run: it has "
                          f"no {', '.join(missing) or 'seed'}")
    log_path = folder / grpo.LOG_NAME
    if not log_path.is_file():
        raise FigureError(f"{shown(log_path)} does not exist")
    log_sha256 = sha256_of_file(log_path)
    named = ((manifest.get("outputs") or {}).get("log") or {}).get("sha256")
    if named is not None and named != log_sha256:
        raise FigureError(f"{shown(log_path)} is not the log that the run manifest names: its SHA-256 is "
                          f"{log_sha256}, and the manifest has {named}")
    rows = []
    for number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise FigureError(f"{shown(log_path)}, line {number} is not complete JSON") from None
        rows.append(row)
    steps = [row.get("step") for row in rows]
    if not rows or steps != list(range(len(rows))):
        raise FigureError(f"{shown(log_path)} must hold the steps 0, 1, 2, ... each once and in order; it "
                          f"holds {steps[:12]}{' ...' if len(steps) > 12 else ''}")
    per_step = config["n_scenarios"] * config["group_size"]
    for row in rows:
        where = f"{shown(log_path)}, step {row['step']}"
        if row.get("n_episodes") != per_step:
            raise FigureError(f"{where} has {row.get('n_episodes')} attempts, not {config['n_scenarios']} "
                              f"groups x {config['group_size']} attempts = {per_step}")
        absent = [key for key in NEEDED if not isinstance(row.get(key), (int, float))
                  or isinstance(row.get(key), bool) or not math.isfinite(row[key])]
        if absent:
            raise FigureError(f"{where} has no number for {', '.join(absent)} (a step without a valid "
                              "attempt has none)")
        row["valid"] = row["n_episodes"] - row["n_harness_errors"]
        row["n_messages"] = whole_number(row["turns_per_episode"] * row["valid"],
                                         f"{where}: turns_per_episode x valid attempts")
        row["n_malformed"] = whole_number(row["invalid_call_rate"] * row["n_messages"],
                                          f"{where}: invalid_call_rate x assistant messages")
        equal = row["zero_variance_exact"] * config["n_scenarios"]
        row["n_equal"] = int(round(equal)) if abs(equal - round(equal)) <= 1e-6 else None
    run_config = manifest.get("config") or {}
    return {
        "folder": folder, "name": folder.name, "seed": manifest["seed"],
        "n_scenarios": config["n_scenarios"], "group_size": config["group_size"],
        "tie_tol": config["tie_tol"], "planned_steps": config["steps"], "status": manifest.get("status"),
        "start_adapter": manifest.get("start_adapter") or {},
        "styles": manifest.get("styles"), "split": manifest.get("split"),
        "reward_numbers": ({key: run_config[key] for key in REWARD_KEYS}
                           if all(key in run_config for key in REWARD_KEYS) else None),
        "fail_is_zero": run_config.get("penalise_malformed_on_fail") is False,
        "token_cap": run_config.get("max_new_tokens"),
        "scenarios_used": (len({scenario for row in rows for scenario in row["scenarios"]})
                           if all(isinstance(row.get("scenarios"), list) for row in rows) else None),
        "log": log_path, "log_sha256": log_sha256, "log_in_manifest": named is not None,
        "rows": rows,
    }


def check_rollouts(run: dict) -> dict | None:
    """Make the plotted numbers of every step again from the saved attempts, and compare with the log.

    For each step, `rollouts/step_<step>.episodes.jsonl` of the run is read (128 attempts; one file at a
    time). `rollout_stats` of the trainer, the function that wrote the log line, is run on its records,
    and `plan_update` gives the kept groups again. The fields of CHECKED must be the logged ones (numbers
    up to SLACK). The saved episode files hold no token data, so each record gets an empty list of
    log-probabilities: the one field that needs them (`mean_neg_logprob`) is not compared.

    Returns {"steps": .., "attempts": ..}: what was read. None if the run has no folder `rollouts`: then
    nothing was checked, and the caption says so. Raises FigureError if a step has no file, or with the
    first differences."""
    folder = run["folder"] / "rollouts"
    if not folder.is_dir():
        return None
    gcfg = grpo.GRPOConfig(n_scenarios=run["n_scenarios"], group_size=run["group_size"],
                           tie_tol=run["tie_tol"])
    found, attempts = [], 0
    for row in run["rows"]:
        path = folder / f"step_{row['step']:05d}.episodes.jsonl"
        if not path.is_file():
            raise FigureError(f"{shown(path)} does not exist: the log of step {row['step']} cannot be "
                              "checked against its attempts")
        records = read_records(path)
        attempts += len(records)
        try:
            again = grpo.rollout_stats([{**record, "tokens": {"logprobs": []}} for record in records])
            again["kept_groups"] = int(grpo.plan_update(records, gcfg)["keep"].sum())
        except (KeyError, TypeError, ValueError) as error:
            raise FigureError(f"{shown(path)} cannot be read as the attempts of one step: {error}") from None
        for key in CHECKED:
            logged, made = row.get(key), again.get(key)
            same = (logged == made if isinstance(made, int) and isinstance(logged, int)
                    else made is not None and logged is not None and abs(logged - made) <= SLACK)
            if not same:
                found.append(f"step {row['step']}, {key}: the log has {logged!r}, the saved attempts give "
                             f"{made!r}")
    if found:
        raise FigureError(f"the log of {shown(run['folder'])} does not agree with its saved attempts "
                          f"({plural(len(found), 'difference')}): " + "; ".join(found[:5]))
    return {"steps": len(run["rows"]), "attempts": attempts}


def block_rows(run: dict, block: int) -> list[dict]:
    """The steps of a run in blocks of `block` steps (the last block can be shorter). For each block:

        first, last, steps      the step numbers and their count
        attempts                the valid attempts of the block
        mean_reward             the rewards of all valid attempts of the block / 100, divided by their
                                number (each step's mean, weighted by its valid attempts)
        tokens_per_attempt      the same weighting of `tokens_per_episode`
        zero_variance           the mean of the steps' `zero_variance_exact`
        groups, kept_groups     the groups of the block and how many the trainer kept (`kept_groups`)
        equal_groups            the groups of the block with equal rewards (None if a step has no count)
        steps_without_group     the steps of the block with `kept_groups` 0
    """
    blocks = []
    for start in range(0, len(run["rows"]), block):
        rows = run["rows"][start:start + block]
        attempts = sum(row["valid"] for row in rows)
        blocks.append({
            "first": rows[0]["step"], "last": rows[-1]["step"], "steps": len(rows), "attempts": attempts,
            "mean_reward": math.fsum(row["mean_reward"] * row["valid"] for row in rows) / attempts,
            "tokens_per_attempt": math.fsum(row["tokens_per_episode"] * row["valid"]
                                            for row in rows) / attempts,
            "zero_variance": math.fsum(row["zero_variance_exact"] for row in rows) / len(rows),
            "groups": run["n_scenarios"] * len(rows),
            "kept_groups": sum(row["kept_groups"] for row in rows),
            "equal_groups": (None if any(row["n_equal"] is None for row in rows)
                             else sum(row["n_equal"] for row in rows)),
            "steps_without_group": sum(1 for row in rows if row["kept_groups"] == 0),
        })
    return blocks


def block_size(n_steps: int, wanted: int = BLOCK) -> int:
    """The block of the caption's table: `wanted` steps, or half of a run that is shorter than two blocks."""
    return min(wanted, max(1, n_steps // 2))


# ---------------------------------------------------------------------------
# The dev curve of a run
# ---------------------------------------------------------------------------
def option_of(command: list, name: str) -> str | None:
    """The value after `name` in the command of a run manifest, or None."""
    if isinstance(command, list) and name in command and command.index(name) + 1 < len(command):
        return command[command.index(name) + 1]
    return None


def chain_record(folder: Path) -> dict[str, str]:
    """The lines "key=value" of chain_record.txt of a run folder (the chain script writes there the hashes
    of the adapter files that it gave to the rollout); an empty dict without the file."""
    path = folder / "chain_record.txt"
    if not path.is_file():
        return {}
    pairs = [line.partition("=") for line in path.read_text(encoding="utf-8").splitlines()]
    return {key: value for key, found, value in pairs if found}


def curve_folders(run_folder: Path) -> list[tuple[int, Path]]:
    """(step, folder) for every folder beside the run folder with the name <run>_curve_<digits>, in step
    order. The digits are the number of training steps before the checkpoint."""
    prefix = CURVE_FOLDER.format(run=run_folder.name)
    found = []
    for path in run_folder.parent.glob(prefix + "*"):
        digits = path.name[len(prefix):]
        if path.is_dir() and re.fullmatch(r"\d+", digits):
            found.append((int(digits), path))
    return sorted(found)


def read_curve_point(run: dict, step: int, folder: Path, scenarios: Path, n_draws: int, seed: int) -> dict:
    """One point of the dev curve, after every check of the module text. Keys of the result:

        step, folder, adapter           the checkpoint step, the rollout folder, the adapter of its manifest
        coverage                        (family set, split, wording styles, attempts per scenario, run seed)
        families, attempts, valid, harness_errors, passed     counts
        pass_rate                       `metrics.pass_rate`: the mean of the family pass rates
        lo, hi                          `metrics.family_bootstrap_ci` of the family pass rates
        from_start_adapter              step 0 only: the manifest names the adapter that the run started from
        weights_sha256_agree            True, False or None (not known): the SHA-256 of the weight file in
                                        chain_record.txt against the one in the manifest of the GRPO run
    """
    manifest = read_json(folder / "run_manifest.json")
    where = shown(folder)
    if manifest.get("complete") is not True:
        raise FigureError(f"{where}: the run manifest does not say that the rollout is complete")
    if manifest.get("split") != CURVE_SPLIT:
        raise FigureError(f"{where}: the rollout is of the split {manifest.get('split')!r}; the curve of the "
                          f"figure is made on the {CURVE_SPLIT} split only")
    adapter = manifest.get("adapter")
    start = run["start_adapter"]
    from_start = weights_agree = None
    if step == 0:
        from_start = adapter is not None and adapter == start.get("folder")
        recorded = chain_record(folder).get(WEIGHT_FILE)
        wanted = ((start.get("files") or {}).get(WEIGHT_FILE) or {}).get("sha256")
        if from_start and recorded and wanted:
            weights_agree = recorded == wanted
            if not weights_agree:
                raise FigureError(f"{where}: its manifest names the start adapter of {run['name']}, but "
                                  f"chain_record.txt has another SHA-256 of {WEIGHT_FILE} than the run "
                                  "manifest of the GRPO run")
    else:
        wanted_parts = (run["name"], "checkpoints", f"step_{step:05d}", "adapter")
        if adapter is None or PurePosixPath(adapter).parts[-4:] != wanted_parts:
            raise FigureError(f"{where}: the point of step {step} did not use the checkpoint of that step "
                              f"({'/'.join(wanted_parts)}); its manifest names the adapter {adapter!r}")
    episodes = folder / "episodes.jsonl"
    if not episodes.is_file():
        raise FigureError(f"{shown(episodes)} does not exist")
    records = read_records(episodes)
    styles_text = option_of(manifest.get("command"), "--styles")
    styles = tuple(sorted(int(style) for style in styles_text.split(","))) if styles_text else None
    coverage = (manifest.get("family_set"), manifest.get("split"), styles, manifest.get("samples"),
                manifest.get("run_seed"))
    family_ids = sorted({record["family_id"] for record in records})
    per_scenario: dict[str, int] = {}
    for record in records:
        per_scenario[record["scenario_id"]] = per_scenario.get(record["scenario_id"], 0) + 1
    problems = []
    if len({(record["ruleset"], record["toolset"], record["prompt"]) for record in records}) != 1:
        problems.append("the attempts are not all under one rule set, tool set and prompt")
    if {record["split"] for record in records} != {manifest.get("split")}:
        problems.append(f"the attempts are not all of the split {manifest.get('split')!r} of the manifest")
    if styles is not None and tuple(sorted({record["style"] for record in records})) != styles:
        problems.append(f"the attempts are not in the wording styles {styles} of the manifest")
    if {record["seed"]["run_seed"] for record in records} != {manifest.get("run_seed")}:
        problems.append(f"the attempts do not all have the run seed {manifest.get('run_seed')!r}")
    if (len(family_ids), len(per_scenario)) != (manifest.get("families"), manifest.get("scenarios")):
        problems.append(f"{len(family_ids)} families and {len(per_scenario)} scenarios; the manifest has "
                        f"{manifest.get('families')} and {manifest.get('scenarios')}")
    if set(per_scenario.values()) != {manifest.get("samples")}:
        problems.append(f"not every scenario has the {manifest.get('samples')} attempts of the manifest")
    family_set = manifest.get("family_set")
    if family_set is not None:
        listed = read_json(scenarios / "sets.json").get(family_set)
        if not isinstance(listed, list) or sorted(listed) != family_ids:
            problems.append(f"the families are not the list {family_set!r} of "
                            f"{shown(scenarios / 'sets.json')}")
    split_file = scenarios / f"{CURVE_SPLIT}.jsonl"
    if split_file.is_file():
        if {record.get("data_sha256") for record in records} != {sha256_of_file(split_file)}:
            problems.append(f"the attempts were not made from {shown(split_file)} (data_sha256)")
    if problems:
        raise FigureError(f"{where}: " + "; ".join(problems))
    valid = metrics.valid_records(records)
    if not valid:
        raise FigureError(f"{where}: no attempt without a harness error")
    rates = metrics.family_pass_rates(records)
    interval = metrics.family_bootstrap_ci(list(rates.values()), n_draws, seed, LEVEL)
    return {
        "step": step, "folder": folder, "adapter": adapter, "coverage": coverage,
        "families": len(rates), "attempts": len(records), "valid": len(valid),
        "harness_errors": metrics.harness_error_count(records),
        "passed": sum(1 for record in valid if record["passed"]),
        "pass_rate": metrics.pass_rate(records), "lo": interval["lo"], "hi": interval["hi"],
        "from_start_adapter": from_start, "weights_sha256_agree": weights_agree,
    }


def read_curve(run: dict, scenarios: Path, n_draws: int, seed: int) -> list[dict]:
    """The points of the dev curve of one run, in step order (an empty list if it has none)."""
    points = []
    for step, folder in curve_folders(run["folder"]):
        try:
            points.append(read_curve_point(run, step, folder, scenarios, n_draws, seed))
        except (KeyError, TypeError) as error:
            raise FigureError(f"{shown(folder)} cannot be read as a point of the dev curve: "
                              f"{type(error).__name__}: {error}") from None
    return points


# ---------------------------------------------------------------------------
# Everything that the outputs show
# ---------------------------------------------------------------------------
def zoom_low(lowest: float) -> float:
    """The lower end of the cut y-axis in the lower part of panel (a): the multiple of 0.05 below the
    lowest plotted reward, one more step down if the value is less than 0.01 above it; at least 0."""
    low = math.floor(lowest * 20 + 1e-9) / 20
    if lowest - low < 0.01:
        low -= 0.05
    return max(0.0, round(low, 2))


def build(run_folders: list[Path], scenarios: Path, *, window: int = WINDOW, n_draws: int = N_DRAWS,
          seed: int = BOOT_SEED) -> dict:
    """Read and check every input, and return everything that the five files show. Raises FigureError.

    Keys: `runs` (each as `read_run` gives it, with `smooth` (the centred means of the three log fields),
    `rollout_check`, `curve` and `blocks` added), `window`, `block`, `n_draws`, `seed`, `scenarios`,
    `zoom_low` and `tokens_top` (the two y-ranges that follow the data)."""
    if not 1 <= len(run_folders) <= MAX_RUNS:
        raise FigureError(f"give 1 to {MAX_RUNS} run folders, not {len(run_folders)}")
    if window < 1 or window % 2 == 0:
        raise FigureError(f"--window must be an odd number of at least 1, not {window}")
    if n_draws < 1:
        raise FigureError(f"--draws must be at least 1, not {n_draws}")
    runs = []
    for folder in run_folders:
        try:
            run = read_run(Path(folder))
        except (KeyError, TypeError) as error:
            raise FigureError(f"{shown(folder)} cannot be read as a GRPO run: {type(error).__name__}: "
                              f"{error}") from None
        run["smooth"] = {key: centred_mean([row[key] for row in run["rows"]], window)
                         for key in ("mean_reward", "tokens_per_episode", "zero_variance_exact")}
        run["rollout_check"] = check_rollouts(run)
        run["curve"] = read_curve(run, Path(scenarios), n_draws, seed)
        runs.append(run)
    if len({str(run["folder"].resolve()) for run in runs}) != len(runs):
        raise FigureError("a run folder is given twice")
    if len({run["seed"] for run in runs}) != len(runs):
        raise FigureError(f"two runs have the same seed: {[run['seed'] for run in runs]}; the outputs name "
                          "a run by its seed")
    coverages = {point["coverage"] for run in runs for point in run["curve"]}
    if len(coverages) > 1:
        raise FigureError("the points of the dev curve do not all have the same coverage (family set, split, "
                          f"wording styles, attempts per scenario, run seed): {sorted(map(str, coverages))}")
    counts = {(point["families"], point["attempts"]) for run in runs for point in run["curve"]}
    if len(counts) > 1:
        raise FigureError("the points of the dev curve do not all hold the same number of families and "
                          f"attempts: {sorted(counts)}")
    block = block_size(min(len(run["rows"]) for run in runs))
    for run in runs:
        run["blocks"] = block_rows(run, block)
    lowest = min(row["mean_reward"] for run in runs for row in run["rows"])
    highest_tokens = max(row["tokens_per_episode"] for run in runs for row in run["rows"])
    return {"runs": runs, "window": window, "block": block, "n_draws": n_draws, "seed": seed,
            "scenarios": Path(scenarios), "zoom_low": zoom_low(lowest),
            "tokens_top": 50.0 * max(1, math.ceil(highest_tokens / 50.0 + 1e-9))}


# ---------------------------------------------------------------------------
# The two CSV files
# ---------------------------------------------------------------------------
def cell(value) -> str:
    """One cell: "" for None, the shortest text that gives a float back exactly, else str."""
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else str(value)


def as_csv(header: list[str], rows: list[list]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([[cell(value) for value in row] for row in rows])
    return out.getvalue()


def curves_csv(data: dict) -> str:
    """learning_curves.csv: one row per run and step. The steps of the log first; a checkpoint step after
    the last log step (the last checkpoint) has a row with the dev columns only."""
    window = data["window"]
    header = ["run", "seed", "step", "attempts", "valid_attempts", "mean_reward", f"mean_reward_ma{window}",
              "tokens_per_attempt", f"tokens_per_attempt_ma{window}", "zero_variance_fraction",
              f"zero_variance_fraction_ma{window}", "groups", "kept_groups", "dev_pass_rate", "dev_ci_low",
              "dev_ci_high", "dev_attempts", "dev_valid_attempts", "dev_passed", "dev_families",
              "dev_adapter"]
    lines = []
    for run in data["runs"]:
        by_step = {row["step"]: (row, index) for index, row in enumerate(run["rows"])}
        points = {point["step"]: point for point in run["curve"]}
        for step in sorted(set(by_step) | set(points)):
            line: list = [run["name"], run["seed"], step]
            if step in by_step:
                row, index = by_step[step]
                smooth = run["smooth"]
                line += [row["n_episodes"], row["valid"], row["mean_reward"], smooth["mean_reward"][index],
                         row["tokens_per_episode"], smooth["tokens_per_episode"][index],
                         row["zero_variance_exact"], smooth["zero_variance_exact"][index],
                         run["n_scenarios"], row["kept_groups"]]
            else:
                line += [None] * 10
            if step in points:
                point = points[step]
                line += [point["pass_rate"], point["lo"], point["hi"], point["attempts"], point["valid"],
                         point["passed"], point["families"], point["adapter"]]
            else:
                line += [None] * 8
            lines.append(line)
    return as_csv(header, lines)


def invalid_csv(data: dict) -> str:
    """invalid_call_rate.csv: one row per run and step. The rate is `invalid_call_rate` of the log
    (`metrics.invalid_call_rate`: malformed assistant messages / all assistant messages of the step). The
    two counts follow from the log: messages = turns_per_episode x valid attempts, malformed = rate x
    messages; both are whole numbers up to rounding, or the run is refused."""
    header = ["run", "seed", "step", "invalid_call_rate", "malformed_messages", "assistant_messages",
              "valid_attempts"]
    return as_csv(header, [[run["name"], run["seed"], row["step"], row["invalid_call_rate"],
                            row["n_malformed"], row["n_messages"], row["valid"]]
                           for run in data["runs"] for row in run["rows"]])


# ---------------------------------------------------------------------------
# The caption
# ---------------------------------------------------------------------------
def by_seed(runs: list[dict], values: list[str]) -> str:
    """"0.974 (seed 1) and 0.969 (seed 2)"."""
    return in_words([f"{value} (seed {run['seed']})" for run, value in zip(runs, values)])


def plural(count: int, word: str) -> str:
    """"1 run", "2 runs"."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def last_block(run: dict, block: int) -> dict:
    """The last `block` steps of a run: their first and last step and their mean reward (over all valid
    attempts, as in `block_rows`)."""
    rows = run["rows"][-block:]
    attempts = sum(row["valid"] for row in rows)
    return {"first": rows[0]["step"], "last": rows[-1]["step"],
            "mean_reward": math.fsum(row["mean_reward"] * row["valid"] for row in rows) / attempts}


def shows_sentences(data: dict) -> list[str]:
    """What the curves show, in numbers of the saved files only: one sentence with the mean reward of the
    first and of the last block of steps and the steps without a kept group, and one sentence with the
    first and the last point of the dev curve (where there is one)."""
    runs = data["runs"]
    firsts = [run["blocks"][0] for run in runs]
    lasts = [last_block(run, data["block"]) for run in runs]
    if len({(item["first"], item["last"]) for item in firsts}) == 1 and \
            len({(item["first"], item["last"]) for item in lasts}) == 1:
        early = by_seed(runs, ["{:.3f}".format(item["mean_reward"]) for item in firsts])
        late = by_seed(runs, ["{:.3f}".format(item["mean_reward"]) for item in lasts])
        reward = (f"The mean reward was {early} in steps {firsts[0]['first']} to {firsts[0]['last']}, and "
                  f"{late} in steps {lasts[0]['first']} to {lasts[0]['last']}.")
    else:
        parts = [f"seed {run['seed']}: {a['mean_reward']:.3f} in steps {a['first']} to {a['last']} and "
                 f"{b['mean_reward']:.3f} in steps {b['first']} to {b['last']}"
                 for run, a, b in zip(runs, firsts, lasts)]
        reward = "The mean reward was: " + "; ".join(parts) + "."
    kept = in_words([f"in {sum(1 for row in run['rows'] if row['kept_groups'] == 0)} of the "
                     f"{len(run['rows'])} steps of seed {run['seed']}" for run in runs])
    sentences = [reward[:-1] + f"; the trainer kept no group {kept}."]
    step_sets = {tuple(point["step"] for point in run["curve"]) for run in runs}
    if len(step_sets) == 1 and len(next(iter(step_sets))) >= 2:
        def passed_at(points: list[dict], what: str) -> str:
            """"69 of 80 attempts passed at step 0 (both seeds)" where the runs have the same counts, else
            "79 of 80 (seed 1) and 80 of 80 (seed 2) attempts passed at step 80"."""
            counts = {(point["passed"], point["valid"]) for point in points}
            step = points[0]["step"]
            if len(points) > 1 and len(counts) == 1:
                who = "both seeds" if len(points) == 2 else f"all {len(points)} seeds"
                return f"{points[0]['passed']} of {points[0]['valid']}{what} at step {step} ({who})"
            return (by_seed(runs, [f"{point['passed']} of {point['valid']}" for point in points])
                    + f"{what} at step {step}")

        first = [run["curve"][0] for run in runs]
        final = [run["curve"][-1] for run in runs]
        sentences.append(f"On the {first[0]['families']} dev families, "
                         f"{passed_at(first, ' attempts passed')}, and {passed_at(final, '')}.")
    return sentences


def step0_adapter(runs: list[dict]) -> dict:
    """What the run manifests say about the points of step 0 of the dev curve. Keys:

        points      (run, point) for every point of step 0
        adapters    the adapters that their manifests name, without repeats
        is_start    every point names the adapter that its own GRPO run started from (`start_adapter`)
        shared      and it is one adapter, from which every run of the figure started
        is_sft      and that adapter was trained by scripts/gpu/train_sft.py (`trained_by` of each manifest)
        same_hash   the SHA-256 of the weight file agrees with chain_record.txt of every point of step 0
    """
    zero = [(run, point) for run in runs for point in run["curve"] if point["step"] == 0]
    adapters = sorted({str(point["adapter"]) for _, point in zero})
    is_start = bool(zero) and all(point["from_start_adapter"] for _, point in zero)
    shared = is_start and len(adapters) == 1 and all(run["start_adapter"].get("folder") == adapters[0]
                                                      for run in runs)
    scripts = {(run["start_adapter"].get("trained_by") or {}).get("script") for run in runs}
    return {"points": zero, "adapters": adapters, "is_start": is_start, "shared": shared,
            "is_sft": shared and scripts == {SFT_SCRIPT},
            "same_hash": is_start and all(point["weights_sha256_agree"] for _, point in zero)}


def caption_text(data: dict) -> str:
    """caption.md: the caption, then the rule, the unit, the count and the source of every number."""
    runs, window, block = data["runs"], data["window"], data["block"]
    half = window // 2
    n_runs = len(runs)
    seeds = in_words([f"seed {run['seed']}" for run in runs])
    one_shape = len({(run["n_scenarios"], run["group_size"]) for run in runs}) == 1
    n_scenarios, group_size = runs[0]["n_scenarios"], runs[0]["group_size"]
    per_step = n_scenarios * group_size
    step_counts = [len(run["rows"]) for run in runs]
    points = [point for run in runs for point in run["curve"]]
    first = points[0] if points else None
    curve_steps = sorted({point["step"] for point in points})
    zero = step0_adapter(runs)
    on_train = all(run["split"] == "train" for run in runs)
    lines: list[str] = []
    add = lines.append

    # ---- the caption itself
    add(f"# Figure 1. Learning curves of the {plural(n_runs, 'GRPO run')}")
    add("")
    add(f"![Learning curves of the GRPO runs]({PNG_NAME})")
    add("")
    steps_words = (f"{step_counts[0]} steps each" if len(set(step_counts)) == 1 and n_runs > 1 else
                   plural(step_counts[0], "step") if n_runs == 1 else
                   by_seed(runs, [str(count) for count in step_counts]) + " steps")
    # Three or four sentences: (1) panels (a) to (c), their units, the line and the y-axes; (2) panel (d);
    # (3) what the numbers of (a) and (c) show; (4) the first and the last point of (d). The rule, the
    # unit, the count and the source of every number are in the notes below the caption.
    shape = (f"; a step is {n_scenarios} training scenarios x {group_size} attempts = {per_step} attempts"
             if one_shape else "")
    equal = f"whose {group_size} rewards" if one_shape else "whose rewards"
    lead = (f"Learning curves of the {plural(n_runs, 'GRPO run')} ({seeds}; {steps_words}{shape}): for each "
            f"step, (a) the mean reward (1.00 is the full reward), (b) the mean number of sampled tokens per "
            f"attempt, and (c) the share of groups {equal} are exactly equal, each with a line that is the "
            f"mean of {plural(window, 'step')} centred on its step; every y-axis starts at 0, except the lower "
            f"part of (a), which shows the range {data['zoom_low']:.2f} to 1.00 again and so makes small "
            "changes look larger.")
    if points:
        _, _, styles, samples, _ = first["coverage"]
        per = "family" if styles and len(styles) == 1 else "scenario"
        style_words = ("" if not styles else "one wording style, " if len(styles) == 1
                       else f"{len(styles)} wording styles, ")
        held_out = (f"Panel (d) shows the pass rate of {first['families']} dev families at the checkpoints "
                    f"of steps {in_words([str(step) for step in curve_steps])}, with a 95% interval from "
                    f"resampling the families; each point holds {first['attempts']} attempts ({style_words}"
                    f"{samples} attempts per {per})")
        if zero["shared"]:
            held_out += (", and step 0 is the " + ("SFT " if zero["is_sft"] else "") + "adapter that "
                         + ("the run" if n_runs == 1 else "both runs" if n_runs == 2 else "every run")
                         + " started from")
        held_out += "."
    else:
        held_out = "Panel (d) is empty: no point of the dev curve was found."
    add(f"**Caption.** {lead} {held_out} " + " ".join(shows_sentences(data)))
    add("")

    # ---- words
    add("## Words")
    add("")
    add("- GRPO: the training method of these runs. In each group it makes the attempts with a reward above "
        "the mean of the group more likely, and the attempts below it less likely.")
    add("- Attempt: one conversation of the model with the tools of the access desk. It passes or fails, "
        "and it gets a reward.")
    add("- Family and scenario: a family is one world with one correct end state. A scenario is one family "
        "in one wording style.")
    add(f"- Group: the {group_size if one_shape else ''} attempts of one scenario in one "
        "step.".replace("  ", " "))
    add(f"- Step: the model makes {per_step if one_shape else 'its'} attempts with the weights that it has. "
        "Then the trainer computes one change of the weights from the groups that it keeps.")
    add("- Seed: the number that fixes the order of the scenarios and the sampling of a run. Each seed is "
        "one training run.")
    add("- Adapter and checkpoint: the adapter is the small set of trained weights (LoRA). A checkpoint is "
        "the adapter as it was saved after a number of steps.")
    if zero["is_sft"]:
        add("- SFT: the stage before GRPO. It trains the adapter on attempts of the untrained model that "
            "passed (`ANALYSIS_PLAN.md`, section 7).")
    add("- Sampled token: a token that the model wrote. Prompts and tool results are not sampled.")
    add("- Malformed message: an assistant message that is not exactly one valid tool call.")
    add("- Harness: the program that runs the attempts. An attempt that it loses is in no number.")
    add("- Dev family: a family of the dev split."
        + (" The GRPO runs trained on the train split (run manifests)." if on_train else ""))
    add("- Family-weighted pass rate: the pass rate of each family first, then the mean over the families.")
    add("")

    # ---- panel by panel
    add("## Notes to the caption: what each panel counts")
    add("")
    add("The x-axis is the training step. A point at step s in (a) to (c) shows attempts that the model made "
        "after s earlier steps (the steps 0 to s - 1). A checkpoint of step s holds the weights after s "
        "steps. So a point of (a) to (c) and a checkpoint at the same x have the same weights.")
    add("")
    add(f"Every y-axis starts at 0: (a), (c) and (d) go from 0 to 1, and (b) goes from 0 to "
        f"{data['tokens_top']:.0f} tokens. The one exception is the lower part of (a): it shows the range "
        f"{data['zoom_low']:.2f} to 1.00 of the upper part again.")
    add("")
    n_steps_all = sum(step_counts)
    n_errors = sum(row["n_harness_errors"] for run in runs for row in run["rows"])
    n_attempts_all = sum(row["n_episodes"] for run in runs for row in run["rows"])
    add(f"- **(a) Reward.** The mean reward of the {per_step if one_shape else ''} attempts of a step, "
        "divided by 100, so 1.00 is the full reward. ".replace("  ", " ")
        + "Source: the field `mean_reward` of `train_log.jsonl`. Rule: the rewards of the "
        "valid attempts of the step, divided by 100, then their mean. Unit: reward / 100. "
        + (f"Count: {per_step} attempts per point. " if one_shape else "")
        + f"The harness lost {n_errors} of the {n_attempts_all:,} attempts of the {n_steps_all} steps.")
    numbers = runs[0]["reward_numbers"]
    if numbers is not None and all(run["reward_numbers"] == numbers and run["fail_is_zero"] for run in runs):
        add(f"  An attempt that fails has the reward 0. An attempt that passes has "
            f"{numbers['pass_reward']}, minus {numbers['malformed_penalty']} for each malformed message (at "
            f"most {numbers['malformed_cap']}), minus {numbers['turn_penalty']} for each turn over the "
            f"budget (at most {numbers['turn_cap']}). These numbers are in the run manifests (`config`). "
            "The budget is the shortest correct path plus one turn (`ANALYSIS_PLAN.md`, section 6).")
    if one_shape:
        add(f"  One attempt that fails in place of a full reward lowers the mean of a step by 1/{per_step} "
            f"= {1 / per_step:.3f}.")
    add(f"  y-axis: the upper part shows the whole scale, from 0 to 1. The lower part shows the same values "
        f"again with a cut y-axis, from {data['zoom_low']:.2f} to 1.00. The cut makes a change look larger "
        "than it is on the whole scale.")
    mean_turns = [math.fsum(row["n_messages"] for row in run["rows"])
                  / sum(row["valid"] for row in run["rows"]) for run in runs]
    caps = {run["token_cap"] for run in runs}
    cap_words = (f"the token cap ({next(iter(caps))} tokens per message, run manifests)"
                 if len(caps) == 1 and None not in caps else "the token cap")
    add("- **(b) Completion length.** The mean number of sampled tokens per attempt. "
        "Source: the field `tokens_per_episode` (`rollout_stats` in "
        "`accessdesk/train/grpo.py`). Rule: the sampled tokens of all assistant messages of the valid "
        "attempts of the step, divided by the number of valid attempts. Unit: tokens per attempt."
        + (f" Count: {per_step} attempts per point." if one_shape else ""))
    add(f"  A message counts with its end-of-turn token. A malformed message counts too. A message that "
        f"{cap_words} cut off counts with the tokens up to the cap.")
    add(f"  An attempt has {by_seed(runs, [f'{value:.2f}' for value in mean_turns])} assistant messages on "
        "average (the field `turns_per_episode`, over all steps). The length of an attempt is the sum of "
        "the lengths of its messages.")
    add(f"  y-axis: from 0 to {data['tokens_top']:.0f} tokens.")
    add(f"- **(c) Zero-variance fraction.** The share of the {n_scenarios if one_shape else ''} groups of "
        f"a step whose {group_size if one_shape else ''} rewards are exactly equal. ".replace("  ", " ")
        + "Source: the field `zero_variance_exact` "
        "(`metrics.zero_variance_share`, `equal_reward`). Rule: the number of groups whose valid attempts "
        "all have exactly the same reward, divided by the number of groups."
        + (f" Count: {n_scenarios} groups of {group_size} attempts per point. A value of the line is a "
           f"share of {window * n_scenarios} groups." if one_shape else ""))
    if {run["tie_tol"] for run in runs} == {0.0}:
        add("  This is the rule of the trainer: it drops a group whose rewards are all equal. Its tie "
            "tolerance is 0, so only exactly equal rewards count as equal. A dropped group adds nothing to "
            "the change of the weights.")
    for sentence in no_group_sentences(runs):
        add("  " + sentence)
    add("  y-axis: the whole scale, from 0 to 1.")
    if points:
        for line in panel_d_lines(data, points, zero):
            add(line)
    add("")
    add(f"The line of (a) to (c): for step s, the mean of the values of the steps s - {half} to s + {half} "
        f"({plural(window, 'step')}). It is drawn only where all these steps exist"
        + (f": from step {half} to step {step_counts[0] - 1 - half}."
           if len(set(step_counts)) == 1 and step_counts[0] >= window else "."))
    add("")

    # ---- the numbers
    add(f"## The numbers by block of {plural(block, 'step')}")
    add("")
    add("From `train_log.jsonl` of each run. Mean reward and tokens per attempt: over all valid attempts of "
        "the block. Groups with equal rewards: their number in the block, and the mean of the "
        "zero-variance fractions of its steps.")
    add("")
    add("| Seed | Steps | Attempts | Mean reward | Tokens per attempt | Groups with equal rewards | "
        "Kept groups | Steps with no kept group |")
    add("|---|---|---|---|---|---|---|---|")
    for run in runs:
        for item in run["blocks"]:
            equal = ("" if item["equal_groups"] is None else f"{item['equal_groups']} of {item['groups']}, ")
            add(f"| {run['seed']} | {item['first']} to {item['last']} | {item['attempts']:,} | "
                f"{item['mean_reward']:.3f} | {item['tokens_per_attempt']:.1f} | "
                f"{equal}{percent(item['zero_variance'])} | {item['kept_groups']} of {item['groups']} | "
                f"{item['steps_without_group']} of {item['steps']} |")
    if points:
        add("")
        add(f"The dev curve, from `episodes.jsonl` of each point ({first['families']} families, "
            f"{first['attempts']} attempts per point):")
        add("")
        add("| Seed | Checkpoint step | Attempts passed | Pass rate | 95% interval | Adapter |")
        add("|---|---|---|---|---|---|")
        for run in runs:
            for point in run["curve"]:
                add(f"| {run['seed']} | {point['step']} | {point['passed']} of {point['valid']} | "
                    f"{percent(point['pass_rate'])} | {percent(point['lo'])} to {percent(point['hi'])} | "
                    f"`{point['adapter']}` |")
    add("")

    # ---- limits
    add("## What the figure cannot show")
    add("")
    add(f"- The figure shows {plural(n_runs, 'training run')}. "
        + ("Two runs describe two checkpoints. " if n_runs == 2 else
           "One run describes one checkpoint. " if n_runs == 1 else "They describe their checkpoints. ")
        + "This does not measure how much the result changes from run to run.")
    if one_shape and all(run["scenarios_used"] is not None for run in runs):
        used = in_words([f"the {plural(len(run['rows']), 'step')} of seed {run['seed']} used "
                         f"{run['scenarios_used']} different scenarios" for run in runs])
        add(f"- A step holds {n_scenarios} scenarios, and the scenarios change from step to step: {used} "
            "(the field `scenarios` of the log). So a part of the change from one point to the next comes "
            "from the scenarios, not from the training.")
    if on_train and points:
        add("- Panels (a) to (c) are made from training scenarios. Only panel (d) is made from families that "
            "the GRPO runs did not train on.")
    if points:
        tables = runs[0]["folder"].parent / "tables"
        add(f"- Panel (d) is a small sample: {first['attempts']} attempts per point."
            + (f" The pass rates of the whole dev set are in `{shown(tables)}/`." if tables.is_dir() else "")
            + " By the plan, no checkpoint is chosen by this curve: the result of a run is its last "
              "checkpoint (`ANALYSIS_PLAN.md`, sections 7 and 10).")
    add("")

    # ---- sources
    add("## Source files")
    add("")
    for run in runs:
        named = " (the run manifest names this hash)" if run["log_in_manifest"] else ""
        add(f"- Seed {run['seed']}: `{shown(run['log'])}`, {plural(len(run['rows']), 'line')}, SHA-256 "
            f"`{run['log_sha256'][:16]}...`{named}. Status of the run in its manifest: {run['status']}.")
        for row in run["rows"]:
            if row.get("fault") is not None:
                add(f"  The stop rule ended this run at step {row['step']}: {row['fault']}")
    if points:
        folders = in_words([f"`{shown(run['folder'])}_curve_<step>`" for run in runs if run["curve"]])
        add(f"- The dev curve: the folders {folders} (`run_manifest.json` and `episodes.jsonl`), and the "
            f"list `{first['coverage'][0]}` of `{shown(data['scenarios'] / 'sets.json')}`.")
    add("- " + check_sentence(runs))
    add(f"- Made by `scripts/make_figure.py` with a window of {plural(window, 'step')}, "
        f"{data['n_draws']:,} bootstrap draws and the bootstrap seed {data['seed']}. It reads no test "
        "result.")
    add("")

    # ---- columns
    add(f"## Columns of `{CSV_NAME}`")
    add("")
    add("One row per run and step. A cell is empty where a step has no such value.")
    add("")
    add("- `run`, `seed`, `step`: the run folder, its GRPO seed, and the step.")
    add("- `attempts`, `valid_attempts`: the attempts of the step, and those that the harness did not lose.")
    add(f"- `mean_reward`, `tokens_per_attempt`, `zero_variance_fraction`: the points of (a), (b) and (c). "
        f"The same names with `_ma{window}` at the end: the lines.")
    add("- `groups`, `kept_groups`: the groups of the step, and the groups that the trainer kept.")
    add("- `dev_pass_rate`, `dev_ci_low`, `dev_ci_high`: the points and the intervals of (d). "
        "`dev_attempts`, `dev_valid_attempts`, `dev_passed`, `dev_families`: their counts. `dev_adapter`: "
        "the adapter of the point.")
    add("")
    add(f"`{INVALID_NAME}` holds the invalid-call rate of every step: the malformed assistant messages "
        "divided by all assistant messages of the step (the field `invalid_call_rate`), with both counts. "
        "It is not a panel.")
    return "\n".join(lines) + "\n"


def no_group_sentences(runs: list[dict]) -> list[str]:
    """How many groups the trainer kept, how many steps kept no group, and what the log shows for them."""
    kept = in_words([f"{sum(row['kept_groups'] for row in run['rows'])} of the "
                     f"{run['n_scenarios'] * len(run['rows'])} groups of seed {run['seed']}" for run in runs])
    sentences = [f"Over all steps the trainer kept {kept} (the field `kept_groups`)."]
    rows = [row for run in runs for row in run["rows"]]
    if all(row["n_equal"] is not None and row["kept_groups"] == run["n_scenarios"] - row["n_equal"]
           for run in runs for row in run["rows"]):
        sentences.append("In every step the kept groups are exactly the groups whose rewards are not all "
                         "equal.")
    without = [[row for row in run["rows"] if row["kept_groups"] == 0] for run in runs]
    counts = in_words([f"{len(found)} of the {len(run['rows'])} steps of seed {run['seed']}"
                       for run, found in zip(runs, without)])
    text = f"The trainer kept no group in {counts}."
    every = [row for found in without for row in found]
    if every and all(row.get("loss") == 0.0 and row.get("grad_norm") == 0.0 for row in every):
        text += (" In each of these steps the log has a loss of 0 and a gradient norm of 0: the step gave "
                 "the trainer nothing to learn from.")
    if every and {id(row) for row in every} == {id(row) for row in rows if row["zero_variance_exact"] == 1.0}:
        text += " These are the steps with the value 1.00 in the panel."
    sentences.append(text)
    return sentences


def panel_d_lines(data: dict, points: list[dict], zero: dict) -> list[str]:
    """The lines of panel (d): the rule, the counts, the interval, the adapters."""
    runs = data["runs"]
    first = points[0]
    family_set, split, styles, samples, run_seed = first["coverage"]
    style_words = ("" if not styles else f" in wording style {styles[0]}" if len(styles) == 1 else
                   f" in the wording styles {in_words([str(style) for style in styles])}")
    steps = sorted({point["step"] for point in points})
    in_short = ("" if not styles else " in one wording style" if len(styles) == 1
                else f" in {len(styles)} wording styles")
    lines = [f"- **(d) Held-out pass rate.** The family-weighted pass rate of {first['families']} {split} "
             f"families at the checkpoints of steps {in_words([str(step) for step in steps])}, with a 95% "
             f"interval from resampling the {first['families']} families. Each point holds "
             f"{first['attempts']} attempts: {first['families']} families{in_short}, {samples} attempts per "
             f"{'family' if styles and len(styles) == 1 else 'scenario'}. "
             f"Source: `episodes.jsonl` of each point of the dev curve. Rule: "
             f"the pass rate of each family over its attempts, then the mean over the families "
             f"(`metrics.pass_rate`). Unit: the family. Count: {first['attempts']} attempts per point: the "
             f"{first['families']} {split} families of the set `{family_set}`{style_words}, {samples} "
             f"attempts per scenario, sampled with the run seed {run_seed} (run manifests). The harness lost "
             f"{sum(point['harness_errors'] for point in points)} of the "
             f"{sum(point['attempts'] for point in points):,} attempts of the "
             f"{plural(len(points), 'point')}."]
    trained = {(run["split"], tuple(run["styles"] or ())) for run in runs}
    if len(trained) == 1 and runs[0]["split"] == "train" and styles and runs[0]["styles"]:
        seen = set(styles) <= set(runs[0]["styles"])
        lines.append(
            f"  Held out means: no GRPO step used a dev family. The runs trained on the train split, in the "
            f"wording styles {in_words([str(style) for style in runs[0]['styles']])} (run manifests). "
            + (f"The wording {'style' if len(styles) == 1 else 'styles'} of the curve "
               f"{'is one' if len(styles) == 1 else 'are'} of them. So the curve tests new families, not a "
               "new wording style." if seen else
               "The curve also has a wording style that the runs did not train on."))
    text = (f"  Interval: 95%, from {data['n_draws']:,} draws. Each draw takes {first['families']} families "
            f"with replacement from the {first['families']}, and their mean pass rate "
            f"(`metrics.family_bootstrap_ci`, seed {data['seed']}, the percentile form).")
    full = sum(1 for point in points if point["passed"] == point["valid"])
    if full:
        text += (f" In {full} of the {plural(len(points), 'point')} every attempt passed. The interval then "
                 "has no width: this method cannot show the doubt that is left.")
    lines.append(text)
    if not zero["points"]:
        lines.append("  The dev curve has no point of step 0.")
    elif len(zero["adapters"]) > 1:
        lines.append("  The points of step 0 used these adapters (run manifests): "
                     + in_words([f"`{point['adapter']}` (seed {run['seed']})"
                                 for run, point in zero["points"]]) + ".")
    else:
        many = len(zero["points"]) > 1
        text = (f"  The {'points' if many else 'point'} of step 0 used the adapter `{zero['adapters'][0]}` "
                f"(`adapter` in the run manifest of {'each' if many else 'the'} point).")
        if zero["is_start"] and not zero["shared"]:
            text += (" Each of these points names the adapter that its own GRPO run started from "
                     "(`start_adapter` in the run manifest of the GRPO run).")
        if zero["shared"]:
            text += (f" This is the adapter that {'each GRPO run' if len(runs) > 1 else 'the GRPO run'} "
                     "started from (`start_adapter` in the run manifest of the GRPO run)"
                     + (f": the SFT adapter, trained by `{SFT_SCRIPT}`." if zero["is_sft"] else "."))
        if zero["is_start"]:
            if zero["same_hash"]:
                text += (f" The SHA-256 of `{WEIGHT_FILE}` is the same in `chain_record.txt` of the point "
                         "and in the manifest of the GRPO run.")
            if many and len({point["coverage"] for _, point in zero["points"]}) == 1:
                passed = in_words([f"{point['passed']} of {point['valid']} (seed {run['seed']})"
                                   for run, point in zero["points"]])
                text += (f" So the points of step 0 are {len(zero['points'])} rollouts of one model with "
                         f"one run seed. Attempts that passed: {passed}.")
        lines.append(text)
    later = sorted({point["step"] for point in points if point["step"] > 0})
    if later:
        lines.append(f"  The points of the steps {in_words([str(step) for step in later])} used the "
                     "checkpoint that the run saved after that number of steps "
                     "(`checkpoints/step_<step>/adapter`, run manifests).")
    lines.append("  The labels in the panel give the attempts that passed at the first and at the last "
                 "checkpoint. The table below has all points.")
    if len(runs) > 1:
        lines.append(f"  The points of the seeds are drawn {DODGE:g} steps apart, to the left and to the "
                     "right of the checkpoint step, so that every interval can be seen.")
    lines.append("  y-axis: the whole scale, from 0 to 1.")
    return lines


def check_sentence(runs: list[dict]) -> str:
    """What was checked against the saved attempts of the steps."""
    checked = [run for run in runs if run["rollout_check"] is not None]
    if not checked:
        return ("Check of the log: none. The run folders have no folder `rollouts`, so the logged values "
                "were not made again from the saved attempts.")
    steps = sum(run["rollout_check"]["steps"] for run in checked)
    attempts = sum(run["rollout_check"]["attempts"] for run in checked)
    text = (f"Check of the log: for every step, the function that wrote the log line (`rollout_stats`) and "
            f"the function that chooses the kept groups (`plan_update`) were run again on the saved attempts "
            f"(`rollouts/step_<step>.episodes.jsonl`). They gave the logged values of "
            f"{in_words([f'`{key}`' for key in CHECKED])} in all {plural(steps, 'step')} ({attempts:,} "
            "attempts).")
    if len(checked) < len(runs):
        text += (" Not checked, because the folder `rollouts` is missing: "
                 + in_words([f"seed {run['seed']}" for run in runs if run["rollout_check"] is None]) + ".")
    return text


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def curve_labels(runs: list[dict]) -> tuple[list[str], list[str]]:
    """The two sets of count labels of panel (d): for the first and for the last checkpoint step.

    Only these two steps get labels: a number at every point would be too much, and the caption has the
    table of all points. A step where every run has the same counts gets one label for all of them. The
    labels of a step are in the order of the runs."""
    def of_step(step: int) -> list[str]:
        found = [(run, point) for run in runs for point in run["curve"] if point["step"] == step]
        counts = {(point["passed"], point["valid"]) for _, point in found}
        if len(found) > 1 and len(counts) == 1:
            passed, valid = next(iter(counts))
            who = "both seeds" if len(found) == 2 else f"all {len(found)} seeds"
            return [f"{passed} of {valid} passed ({who})"]
        return [f"seed {run['seed']}: {point['passed']} of {point['valid']} passed" for run, point in found]

    steps = sorted({point["step"] for run in runs for point in run["curve"]})
    if not steps:
        return [], []
    return of_step(steps[0]), (of_step(steps[-1]) if len(steps) > 1 else [])


def figure_of(data: dict):
    """The matplotlib figure of the four panels, not yet saved. matplotlib is imported here, with the Agg
    backend. The caller saves the figure and closes it (`draw`)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    runs, window = data["runs"], data["window"]
    styles = SERIES[:len(runs)]
    last_step = max([run["rows"][-1]["step"] for run in runs]
                    + [point["step"] for run in runs for point in run["curve"]])
    widest = DODGE * (len(runs) - 1) / 2
    x_low, x_high = -0.03 * max(last_step, 1) - 1.0 - widest / 2, 1.03 * max(last_step, 1) + 1.0 + widest / 2
    tick_step = next(size for size in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10 ** 9)
                     if last_step / size <= 8)
    x_ticks = list(range(0, last_step + 1, tick_step))
    rc = {"font.family": FONT, "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
          "ps.fonttype": 42, "axes.unicode_minus": False, "figure.facecolor": SURFACE,
          "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK}

    def style_axes(ax, y_low, y_high, y_ticks, decimals, face=SURFACE):
        ax.set_facecolor(face)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.{decimals}f}"))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(axis="both", length=2.5, width=0.6, color=AXIS, labelcolor=INK_SECOND, labelsize=7,
                       pad=2)
        ax.grid(axis="y", color=GRID if face == SURFACE else SURFACE, linewidth=0.5)
        ax.set_axisbelow(True)

    def header(ax, title, subtitle):
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 15.5), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color=INK, ha="left", va="baseline")
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5.5),
                    textcoords="offset points", fontsize=6.6, color=INK_SECOND, ha="left", va="baseline")

    def note(ax, x, y, text, **kwargs):
        ax.text(x, y, text, transform=ax.transAxes, fontsize=6.6, color=INK_SECOND, linespacing=1.35,
                **{"ha": "left", "va": "bottom", **kwargs})

    def points_and_line(ax, key):
        for run, style in zip(runs, styles):
            filled = style["filled"]
            ax.plot([row["step"] for row in run["rows"]], [row[key] for row in run["rows"]],
                    linestyle="none", marker=style["marker"], markersize=2.9 if filled else 3.3,
                    markerfacecolor=style["color"] if filled else "none",
                    markeredgecolor=SURFACE if filled else style["color"],
                    markeredgewidth=0.25 if filled else 0.7, alpha=0.62, zorder=2, clip_on=False)
        for run, style in zip(runs, styles):
            pairs = [(row["step"], value) for row, value in zip(run["rows"], run["smooth"][key])
                     if value is not None]
            line, = ax.plot([step for step, _ in pairs], [value for _, value in pairs], color=style["color"],
                            linewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            line.set_path_effects([patheffects.Stroke(linewidth=2.3, foreground=SURFACE),
                                   patheffects.Normal()])

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(FIG_WIDTH_CM / 2.54, FIG_HEIGHT_CM / 2.54))
        outer = fig.add_gridspec(2, 2, left=0.082, right=0.985, top=0.895, bottom=0.068, hspace=0.40,
                                 wspace=0.24)
        inner = outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 2.1], hspace=0.26)
        ax_full, ax_zoom = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        ax_b, ax_c, ax_d = (fig.add_subplot(outer[0, 1]), fig.add_subplot(outer[1, 0]),
                            fig.add_subplot(outer[1, 1]))

        # (a) reward: the whole scale, then the same values with a cut y-axis
        low = data["zoom_low"]
        span = 1.0 - low
        zoom_tick = 0.05 if span <= 0.3 + 1e-9 else 0.1 if span <= 0.6 + 1e-9 else 0.2
        zoom_ticks = [round(1.0 - zoom_tick * i, 2) for i in range(int(span / zoom_tick + 1e-9) + 1)]
        style_axes(ax_full, 0.0, 1.0, [0.0, 0.5, 1.0], 2)
        style_axes(ax_zoom, low, 1.0, zoom_ticks, 2, face=BAND)
        ax_full.axhspan(low, 1.0, color=BAND, zorder=0, linewidth=0)
        points_and_line(ax_full, "mean_reward")
        points_and_line(ax_zoom, "mean_reward")
        ax_full.tick_params(axis="x", labelbottom=False)
        header(ax_full, "(a) Reward", "mean reward of the attempts of a step (1.00 = full reward)")
        note(ax_full, 0.985, 0.1, f"whole scale, 0 to 1; grey band: {low:.2f} to 1.00", ha="right")
        note(ax_zoom, 0.985, 0.05, f"the grey band again:\ny-axis cut, {low:.2f} to 1.00", ha="right")
        ax_zoom.set_ylabel("mean reward", fontsize=7, color=INK_SECOND, labelpad=3)

        # (b) completion length
        top = data["tokens_top"]
        style_axes(ax_b, 0.0, top, [50.0 * i for i in range(int(round(top / 50.0)) + 1)], 0)
        points_and_line(ax_b, "tokens_per_episode")
        header(ax_b, "(b) Completion length", "mean number of sampled tokens per attempt")
        ax_b.set_ylabel("tokens per attempt", fontsize=7, color=INK_SECOND, labelpad=3)

        # (c) zero-variance share
        style_axes(ax_c, 0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0], 2)
        points_and_line(ax_c, "zero_variance_exact")
        shape = {(run["n_scenarios"], run["group_size"]) for run in runs}
        subtitle = (f"share of the {runs[0]['n_scenarios']} groups of a step with {runs[0]['group_size']} "
                    "equal rewards" if len(shape) == 1 else
                    "share of the groups of a step with equal rewards")
        header(ax_c, "(c) Zero-variance fraction", subtitle)
        without = [f"seed {run['seed']}: {sum(1 for row in run['rows'] if row['kept_groups'] == 0)} of "
                   f"{len(run['rows'])} steps" for run in runs]
        at_one = all((row["kept_groups"] == 0) == (row["zero_variance_exact"] == 1.0)
                     for run in runs for row in run["rows"])
        note(ax_c, 0.985, 0.04, "steps in which no group was kept" + (" (value 1.00)" if at_one else "")
             + ":\n" + "\n".join(without), ha="right")
        ax_c.set_ylabel("share of groups", fontsize=7, color=INK_SECOND, labelpad=3)
        ax_c.set_xlabel("training step", fontsize=7, color=INK_SECOND, labelpad=2)

        # (d) the dev curve
        style_axes(ax_d, 0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0], 2)
        all_points = [point for run in runs for point in run["curve"]]
        offsets = [DODGE * (index - (len(runs) - 1) / 2) for index in range(len(runs))]
        drawn = [(run, style, offset) for run, style, offset in zip(runs, styles, offsets) if run["curve"]]
        for run, style, offset in drawn:            # the intervals and the lines first, the markers on top
            xs = [point["step"] + offset for point in run["curve"]]
            ys = [point["pass_rate"] for point in run["curve"]]
            ax_d.errorbar(xs, ys, yerr=[[point["pass_rate"] - point["lo"] for point in run["curve"]],
                                        [point["hi"] - point["pass_rate"] for point in run["curve"]]],
                          fmt="none", ecolor=style["color"], elinewidth=1.0, capsize=2.0, capthick=1.0,
                          zorder=2, clip_on=False)
            line, = ax_d.plot(xs, ys, color=style["color"], linewidth=1.5, solid_capstyle="round",
                              solid_joinstyle="round", zorder=3, clip_on=False)
            line.set_path_effects([patheffects.Stroke(linewidth=2.3, foreground=SURFACE),
                                   patheffects.Normal()])
        for run, style, offset in drawn:
            filled = style["filled"]
            ax_d.plot([point["step"] + offset for point in run["curve"]],
                      [point["pass_rate"] for point in run["curve"]], linestyle="none",
                      marker=style["marker"], markersize=4.4,
                      markerfacecolor=style["color"] if filled else SURFACE,
                      markeredgecolor=SURFACE if filled else style["color"],
                      markeredgewidth=0.6 if filled else 1.0, zorder=4, clip_on=False)
        if all_points:
            first = all_points[0]
            header(ax_d, "(d) Held-out pass rate", f"{first['families']} dev families, {first['attempts']} "
                                                   "attempts per point, 95% intervals")
            if len(drawn) > 1:
                note(ax_d, 0.985, 0.04, f"the seeds are drawn side by side,\n{DODGE:g} steps apart, at each "
                                        "checkpoint", ha="right")
            steps = sorted({point["step"] for point in all_points})
            at_first, at_last = curve_labels(runs)
            if steps[0] == 0 and step0_adapter(runs)["is_sft"]:     # the figure alone says where (d) starts
                at_first = ["step 0 = the SFT adapter"] + at_first
            line_height = 0.075

            def stack(labels, x, top_y, ha):
                top_y = max(top_y, 0.02 + line_height * len(labels))          # the labels stay in the panel
                for index, label in enumerate(labels):
                    ax_d.text(x, top_y - index * line_height, label, fontsize=6.6, color=INK_SECOND, ha=ha,
                              va="top")

            first_points = [point for point in all_points if point["step"] == steps[0]]
            last_points = [point for point in all_points if point["step"] == steps[-1]]
            stack(at_first, x_low + 1.0, min(point["lo"] for point in first_points) - 0.03, "left")
            stack(at_last, x_high - 0.5, min(point["lo"] for point in last_points) - 0.05, "right")
        else:
            header(ax_d, "(d) Held-out pass rate", "no point of the dev curve was found")
        ax_d.set_ylabel("pass rate", fontsize=7, color=INK_SECOND, labelpad=3)
        ax_d.set_xlabel("training step (checkpoint)", fontsize=7, color=INK_SECOND, labelpad=2)

        # one legend for the four panels, above them: the seeds on the left, the marks of (a) to (c) on
        # the right
        handles = []
        for run, style in zip(runs, styles):
            filled = style["filled"]
            handles.append(Line2D([], [], color=style["color"], linewidth=1.5, marker=style["marker"],
                                  markersize=4.4, markerfacecolor=style["color"] if filled else SURFACE,
                                  markeredgecolor=SURFACE if filled else style["color"],
                                  markeredgewidth=0.6 if filled else 1.0, label=f"GRPO seed {run['seed']}"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.999), ncol=len(handles),
                   frameon=False, fontsize=7.5, handlelength=1.9, handletextpad=0.5, columnspacing=1.4,
                   borderaxespad=0.2, labelcolor=INK)
        fig.text(0.985, 0.9805, f"In (a) to (c): a point is one training step; the line is the mean of "
                                f"{window} steps, centred.", fontsize=7.0, color=INK_SECOND, ha="right",
                 va="center")
    return fig


def draw(data: dict, png: Path, pdf: Path) -> None:
    """Draw the figure and write it as PNG (PNG_WIDTH_PX pixels wide) and as PDF (FIG_WIDTH_CM wide).
    Neither file holds a date."""
    from matplotlib import pyplot as plt

    fig = figure_of(data)
    try:
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=PNG_WIDTH_PX / (FIG_WIDTH_CM / 2.54), metadata={"Software": None})
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--run", type=Path, action="append", default=None, metavar="FOLDER",
                        help="the folder of one GRPO run; give it once per run (default: "
                             + " and ".join(DEFAULT_RUNS) + ")")
    parser.add_argument("--scenarios", type=Path, default=ROOT / DEFAULT_SCENARIOS,
                        help="the scenario folder of the runs (sets.json is read; the test split never)")
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_OUT, help="the folder for the five files")
    parser.add_argument("--window", type=int, default=WINDOW, help="steps in the centred mean (odd)")
    parser.add_argument("--draws", type=int, default=N_DRAWS, help="bootstrap draws of panel (d)")
    parser.add_argument("--seed", type=int, default=BOOT_SEED, help="bootstrap seed of panel (d)")
    parser.add_argument("--no-image", action="store_true",
                        help="write the two CSV files and the caption only (needs no matplotlib)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the files and print the caption. Returns the exit code: 0, or 2 if nothing was written."""
    args = parse_args(argv)
    run_folders = args.run if args.run else [ROOT / name for name in DEFAULT_RUNS]
    try:
        if not args.no_image:
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                raise FigureError("matplotlib is not installed. Install it with: uv pip install --python "
                                  f".venv/bin/python -r {REQUIREMENTS} ; or give --no-image to write the "
                                  "two CSV files and the caption without the figure") from None
        data = build(run_folders, args.scenarios, window=args.window, n_draws=args.draws, seed=args.seed)
        caption = caption_text(data)
        curves, invalid = curves_csv(data), invalid_csv(data)
    except FigureError as error:
        print(f"make_figure: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if not args.no_image:      # the image first: if the drawing fails, no new table is beside an old image
        draw(data, args.out / PNG_NAME, args.out / PDF_NAME)
        written += [PNG_NAME, PDF_NAME]
    for name, text in ((CSV_NAME, curves), (INVALID_NAME, invalid), (CAPTION_NAME, caption)):
        (args.out / name).write_bytes(text.encode("utf-8"))
        written.append(name)
    print(caption, end="")
    print(f"wrote {', '.join(written)} into {shown(args.out)}")
    if args.no_image:
        print(f"--no-image: {PNG_NAME} and {PDF_NAME} were not written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
