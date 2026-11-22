"""Choose the episodes that SFT trains on, from a rollout of the untrained model on the TRAIN split.

    python scripts/gpu/sft_data.py --episodes results/main/base_train/episodes.jsonl \\
        --tokens results/main/base_train/tokens.jsonl --scenarios data/scenarios --out results/main/sft_data

It needs no GPU and no torch, so it also runs on a laptop. The two input files are the ones that
scripts/gpu/rollout.py wrote for `--split train`. The choice itself is made by the library
(`accessdesk.train.sft`): `attach_tokens` joins each episode with its token row, and `select_sft_records`
keeps the passes without a penalty and without a refused call, at most one per scenario. A refused call is
a well-formed call that the tool answered with an error, for example a call with a made-up id; an attempt
with one is never chosen, also with `--allow-penalised`. This script only checks the inputs, calls those two
functions and writes the result. `--allow-penalised` is passed on to `select_sft_records`.

Written into --out:
    sft_records.jsonl   the chosen episode records, each with its `tokens` (ids, segments, log-probabilities),
                        in the order of their scenario ids. scripts/gpu/train_sft.py reads this file.
    sft_report.json     the report of `select_sft_records`, and a few numbers of this script (see
                        `build_report`)
    sft_report.md       the same per outcome as a small table: families in the input, families with a chosen
                        episode, chosen scenarios, pass rate of the input
    run_manifest.json   the command, the config and its hash, the SHA-256 of every file that was read and
                        written, the hash of the frozen code and the state of the freeze (`code_sha256`
                        and `frozen`, from accessdesk/freeze.py; None with a note in `frozen_note` if
                        that module does not give them) and the hashes of the training code.
                        `"complete": true` is in it only when all files are written.

The run is refused (exit code 2, nothing is written) if
  - the output folder already holds one of these files;
  - a record is of another split than train or of a wording style that is not a training style, or the
    records come from different set-ups (the library says so);
  - the scenario folder has no summary.json, or its train.jsonl is not the file that the build wrote;
  - a record names another scenario file (`data_sha256`) or another text file (`text_sha256`) than the folder;
  - the records were not all made with one config, or their reward numbers are not those of the config here;
  - the rollout's own run_manifest.json lies beside the episodes file and says that the rollout is not
    complete, or names another number of episodes;
  - the scenarios were built from the placeholder text and --out is not under results/_smoke/.
Exit code 1 (nothing is written, the counts are printed): no episode was chosen.

The config. If run_manifest.json of the rollout lies beside the episodes file, its config is used (the token
cap of the rollout may differ from the default). If not, the default config is used, and the manifest says
so. The selection reads one number of it: `pass_reward`.

The functions `utc_now`, `file_entry`, `dict_sha256`, `frozen_code`, `warn_if_not_as_frozen` and
`training_code` are also used by train_sft.py and train_grpo.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for rollout.py, next to this file

import rollout as rollout_script  # noqa: E402
from accessdesk import impl  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.scenarios import load_families, scenario_ids  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.harness.records import read_records, write_records  # noqa: E402
from accessdesk.train.sft import (TRAIN_SPLIT, attach_tokens, select_sft_records,  # noqa: E402
                                  training_example)

SCRIPT = "scripts/gpu/sft_data.py"
MANIFEST_NAME = "run_manifest.json"
RECORDS_NAME = "sft_records.jsonl"
REPORT_NAME = "sft_report.json"
TABLE_NAME = "sft_report.md"
OUTPUT_FILES = (RECORDS_NAME, REPORT_NAME, TABLE_NAME, MANIFEST_NAME)
TRAIN_FILE = f"{TRAIN_SPLIT}.jsonl"
# The config fields that decide the reward of an episode. `select_sft_records` calls a pass "clean" when its
# reward equals `pass_reward`, so the records must have been scored with these numbers.
REWARD_FIELDS = ("pass_reward", "malformed_penalty", "malformed_cap", "turn_penalty", "turn_cap",
                 "penalise_malformed_on_fail")
# What is copied from the rollout's run manifest into this script's manifest.
SOURCE_KEYS = ("complete", "model", "revision", "adapter", "split", "samples", "run_seed", "families",
               "scenarios", "config_sha256", "text_sha256", "split_file_sha256")
# Manifest field -> the function of accessdesk/freeze.py that gives its value.
FREEZE_FUNCTIONS = {"code_sha256": "code_sha256", "frozen": "frozen_state"}
# The training code. accessdesk/freeze.py leaves it out of the frozen files on purpose and says that it is
# recorded at each training run, in the manifest of that run. `training_code` does that.
TRAINING_CODE = ("accessdesk/train/*.py", "accessdesk/_second/grpo_math.py",
                 "accessdesk/_second/loss_mask.py", "scripts/gpu/sft_data.py", "scripts/gpu/train_sft.py",
                 "scripts/gpu/train_grpo.py", "scripts/gpu/chain_stage1.sh", "scripts/gpu/chain_stage2.sh")


class Refused(Exception):
    """The run must not start. The message says why."""


# ---------------------------------------------------------------------------
# Small helpers, shared with train_sft.py and train_grpo.py
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """The time now in UTC, such as 2026-11-18T11:02:03Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_entry(path: Path) -> dict:
    """What a manifest says about one file: its path as given, its SHA-256 and its size in bytes."""
    raw = Path(path).read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def dict_sha256(values: dict) -> str:
    """SHA-256 of json.dumps(values, sort_keys=True): the formula of `RunConfig.sha256`."""
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def command_line(script: str, argv: list[str] | None) -> list[str]:
    """The command for the manifest: sys.argv, or the script's name and `argv` if main() was called with
    one."""
    return list(sys.argv) if argv is None else [script, *argv]


def frozen_code() -> dict:
    """The fields of a run manifest that name the frozen code, with the names of _dev/freeze/WIRING.md:

        code_sha256   `accessdesk.freeze.code_sha256()`: one hash over the code that makes and scores an
                      episode and computes a number, as the tree is now
        frozen        `accessdesk.freeze.frozen_state()`: the hash of MANIFEST.json (None before the
                      freeze), the code hash, whether the tree is as the manifest says, and the differences
        frozen_note   why one of the two is None; None if both are there

    A function is called without arguments, and with the repository folder if it does not take that.
    Nothing here can stop a run: a missing module, a missing function and an error inside a function all
    give None and a note. The training code is not part of the frozen code: see `training_code`.
    """
    fields: dict[str, Any] = {"code_sha256": None, "frozen": None, "frozen_note": None}
    notes: list[str] = []
    try:
        freeze = _freeze_module()
    except Exception as error:      # not there, or half-written
        fields["frozen_note"] = (f"accessdesk.freeze could not be imported ({type(error).__name__}: "
                                 f"{error}): the hash of the frozen code was not taken")[:400]
        return fields
    for key, name in FREEZE_FUNCTIONS.items():
        function = getattr(freeze, name, None)
        if function is None:
            notes.append(f"accessdesk.freeze has no {name}")
            continue
        try:
            fields[key] = _call(function) if callable(function) else function
        except Exception as error:
            notes.append(f"accessdesk.freeze.{name} failed ({type(error).__name__}: {error})"[:300])
    if fields["code_sha256"] is None and isinstance(fields["frozen"], dict):
        fields["code_sha256"] = fields["frozen"].get("code_sha256")
    fields["frozen_note"] = "; ".join(notes) or None
    return fields


def warn_if_not_as_frozen(fields: dict) -> None:
    """Say it on stderr if the tree is frozen (MANIFEST.json is there) and differs from that manifest.

    The run goes on: its own manifest holds the differences (`frozen`), so nothing is hidden. A stop here
    could end a night's chain after hours of rollouts for a file that makes no number.
    """
    frozen = fields.get("frozen")
    if not isinstance(frozen, dict) or frozen.get("manifest_sha256") is None or frozen.get("matches"):
        return
    differences = list(frozen.get("differences") or [])
    print(f"WARNING: the tree differs from MANIFEST.json in {len(differences)} places. The run manifest "
          "lists them. Record a wanted change with scripts/freeze.py amend.", file=sys.stderr)
    for line in differences[:10]:
        print(f"- {line}", file=sys.stderr)


def _freeze_module() -> Any:
    return importlib.import_module("accessdesk.freeze")


def _call(function: Any) -> Any:
    try:
        return function()
    except TypeError:
        return function(ROOT)


def training_code() -> dict:
    """The hashes of the training code as it is now, for a run manifest.

    `files`: path -> SHA-256 of the file's bytes, for the files of TRAINING_CODE, in path order.
    `sha256`: one hash over them, by the rule of accessdesk/freeze.py for a group of files: the SHA-256 of
    the lines "<file hash>  <path>\n". These lines are what `sha256sum` prints, so
    `sha256sum PATHS | sha256sum` gives the same value.
    """
    paths = sorted({path.relative_to(ROOT).as_posix() for pattern in TRAINING_CODE
                    for path in ROOT.glob(pattern) if path.is_file()})
    files = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    lines = "".join(f"{digest}  {path}\n" for path, digest in files.items())
    return {"sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(), "files": files}


# ---------------------------------------------------------------------------
# Checks. Each returns what the manifest needs, or raises Refused.
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--episodes", type=Path, required=True,
                        help="episodes.jsonl of the rollout on the train split")
    parser.add_argument("--tokens", type=Path, required=True,
                        help="tokens.jsonl of the same rollout (the same lines, in the same order)")
    parser.add_argument("--scenarios", type=Path, required=True,
                        help="the scenario folder that the rollout read (summary.json and train.jsonl)")
    parser.add_argument("--out", type=Path, required=True, help="output folder; must not hold data yet")
    parser.add_argument("--allow-penalised", action="store_true",
                        help="also take passes with a penalty (their malformed turns are left out of the "
                             "loss)")
    return parser.parse_args(argv)


def check_output_folder(out_dir: Path) -> None:
    held = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if held:
        raise Refused(f"{out_dir} already holds {', '.join(held)}; choose a new output folder")


def check_placeholder_output(text_is_placeholder: object, out_dir: Path) -> None:
    """Data that was made with the placeholder text is written under results/_smoke/ only: the rule of
    scripts/gpu/rollout.py, so that it never lies among the results."""
    smoke = rollout_script.SMOKE_DIR.resolve()
    if text_is_placeholder and smoke not in out_dir.resolve().parents:
        raise Refused(f"the data was made with the placeholder text: the output folder must be under {smoke}")


def read_jsonl(path: Path, what: str) -> list[dict]:
    if not path.is_file():
        raise Refused(f"{what}: {path} is missing")
    try:
        return read_records(path)
    except ValueError as error:      # a line that is not JSON, for example a line that was cut off
        raise Refused(f"{what}: {path} is not a file of JSON lines ({error})") from error


def check_scenario_folder(scenario_dir: Path) -> dict:
    """The folder must have summary.json, and train.jsonl must be the file that the build wrote (its SHA-256
    is in summary.json, `files_sha256`). This is the check of scripts/gpu/rollout.py (`check_scenarios`),
    without a text file: the text hash is compared with the records instead.

    Returns `summary` (the parsed file), `train_sha256` and `scenario_ids` (the set of all train scenarios).
    """
    summary_file = scenario_dir / "summary.json"
    train_file = scenario_dir / TRAIN_FILE
    if not summary_file.is_file():
        raise Refused(f"{summary_file} is missing; build the scenarios with scripts/build_scenarios.py")
    if not train_file.is_file():
        raise Refused(f"{train_file} is missing")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    train_sha256 = rollout_script.sha256_of_file(train_file)
    if train_sha256 != summary.get("files_sha256", {}).get(TRAIN_FILE):
        raise Refused(f"{train_file} is not the file that the build wrote (summary.json names another "
                      "SHA-256): build the scenarios again")
    ids = {scenario_id for family in load_families(train_file) for scenario_id in scenario_ids(family)}
    return {"summary": summary, "train_sha256": train_sha256, "scenario_ids": ids}


def check_records_against_folder(records: list[dict], folder: dict) -> None:
    """Every record must name the folder's train.jsonl (`data_sha256`) and the folder's text file
    (`text_sha256`), and must be an episode of a scenario of that file."""
    other_data = [r for r in records if r.get("data_sha256") != folder["train_sha256"]]
    if other_data:
        found = sorted({str(r.get("data_sha256"))[:12] for r in other_data})
        raise Refused(f"{len(other_data)} of {len(records)} records name another scenario file than "
                      f"{TRAIN_FILE} of the folder (data_sha256 {found}, the folder has "
                      f"{folder['train_sha256'][:12]})")
    text_sha256 = folder["summary"]["text_sha256"]
    other_text = [r for r in records if r.get("text_sha256") != text_sha256]
    if other_text:
        found = sorted({str(r.get("text_sha256"))[:12] for r in other_text})
        raise Refused(f"{len(other_text)} of {len(records)} records were made with another text file than "
                      f"the scenario folder (text_sha256 {found}, the folder has {text_sha256[:12]})")
    unknown = sorted({r["scenario_id"] for r in records} - folder["scenario_ids"])
    if unknown:
        raise Refused(f"{len(unknown)} scenarios of the records are not in {TRAIN_FILE} of the folder, "
                      f"for example {unknown[0]}")


def source_run(episodes_file: Path, n_records: int) -> dict | None:
    """What the rollout's own run_manifest.json (beside the episodes file) says, or None if there is none.

    Returns the keys of SOURCE_KEYS, `manifest` (path, SHA-256, size) and `config` (the rollout's config).
    Refused if that manifest says that the rollout is not complete, or names another number of episodes.
    """
    path = episodes_file.parent / MANIFEST_NAME       # scripts/gpu/rollout.py uses the same file name
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise Refused(f"{path} says that the rollout is not complete: SFT data is taken from a complete "
                      "rollout only")
    scenarios, samples = manifest.get("scenarios"), manifest.get("samples")
    if isinstance(scenarios, int) and isinstance(samples, int) and scenarios * samples != n_records:
        raise Refused(f"{episodes_file} holds {n_records} records, but {path} names {scenarios} scenarios "
                      f"with {samples} attempts each ({scenarios * samples})")
    return {"manifest": file_entry(path), **{key: manifest.get(key) for key in SOURCE_KEYS},
            "config": manifest.get("config")}


def choose_config(records: list[dict], source: dict | None) -> tuple[RunConfig, str]:
    """(the config of the run that made the records, where it comes from).

    All records must carry one `config_sha256` and one `settings` value: they are of one run. With the
    rollout's manifest, its config is built again and must have the hash that the records carry. Without
    it, the default config is taken. In both cases the reward numbers of the records (`settings`) must be
    the config's: only then is "reward equal to pass_reward" a pass without a penalty.
    """
    hashes = sorted({str(r.get("config_sha256")) for r in records})
    settings = _distinct([r.get("settings") for r in records])
    if None in settings:
        raise Refused("a record carries no `settings` (the reward numbers of its run), so it cannot be "
                      "checked that reward 100 means a pass without a penalty")
    if len(hashes) != 1 or len(settings) != 1:
        raise Refused(f"the records were not all made with one config: they carry {len(hashes)} config "
                      f"hashes and {len(settings)} different settings")
    if source is not None and source["config"] is not None:
        try:
            cfg = RunConfig(**source["config"])
        except TypeError as error:
            raise Refused(f"the config in the rollout's run manifest is not a config of this code: {error}")
        if cfg.sha256() != source["config_sha256"] or cfg.sha256() != hashes[0]:
            raise Refused("the config in the rollout's run manifest does not have the hash that the manifest "
                          f"and the records name ({cfg.sha256()[:12]}, {str(source['config_sha256'])[:12]}, "
                          f"{hashes[0][:12]}): the manifest is of another run, or the config code changed")
        origin = "run_manifest.json of the rollout"
    else:
        cfg = RunConfig()
        origin = ("the default config (no run_manifest.json of the rollout with a config beside the "
                  "episodes file)")
    different = [name for name in REWARD_FIELDS if settings[0].get(name) != getattr(cfg, name)]
    if different:
        shown = {name: (settings[0].get(name), getattr(cfg, name)) for name in different}
        raise Refused(f"the records were scored with other reward numbers than the config (record, config): "
                      f"{shown}")
    return cfg, origin


def _distinct(values: list) -> list:
    """The values without repeats, compared with ==, in the order of their first appearance."""
    found: list = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def build_report(records: list[dict], chosen: list[dict], selection: dict) -> dict:
    """The content of sft_report.json.

        selection                 the report of `select_sft_records`, as the library gives it
        pass_rate                 `metrics.pass_rate` of the input: the mean over families of the family's
                                  share of passed episodes (None if the input has no valid record)
        attempts_per_scenario     number of records of a scenario -> how many scenarios have that number
        n_chosen_tokens           all tokens of the chosen episodes (what the model reads in training)
        n_loss_tokens             the tokens of the chosen episodes with weight 1 in `training_example`:
                                  the tokens that the SFT loss is taken on
    """
    examples = [training_example(record) for record in chosen]
    per_scenario: dict[str, int] = {}
    for record in records:
        per_scenario[record["scenario_id"]] = per_scenario.get(record["scenario_id"], 0) + 1
    counts: dict[str, int] = {}
    for n in sorted(per_scenario.values()):
        counts[str(n)] = counts.get(str(n), 0) + 1
    return {
        "selection": selection,
        "pass_rate": metrics.pass_rate(records) if metrics.valid_records(records) else None,
        "attempts_per_scenario": counts,
        "n_chosen_tokens": sum(len(example["ids"]) for example in examples),
        "n_loss_tokens": sum(sum(example["weights"]) for example in examples),
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def to_markdown(report: dict) -> str:
    """sft_report.md: the counts of the selection, then one table row per outcome and one row for all."""
    selection = report["selection"]
    by_outcome = selection.get("by_outcome", {})
    rates = selection.get("pass_rate_by_outcome", {})
    # Both forms name the rule "no attempt with a refused call": it holds with and without a penalty.
    kept = "passes without a penalty and without a refused call"
    if selection.get("allow_penalised"):
        kept = "every pass without a refused call (a penalty is allowed)"
    lines = ["# SFT data: the selection", "",
             f"Input: a rollout of the model on the train split. Kept: {kept}, at most one per scenario. "
             "All rules are in `accessdesk/train/sft.py` (`select_sft_records`).",
             "", "| Outcome | Families in the input | Families covered | Scenarios chosen | "
             "Pass rate of the input |", "|---|---|---|---|---|"]
    for outcome, row in by_outcome.items():
        lines.append(f"| {outcome} | {row['families']} | {row['families_covered']} | "
                     f"{row['scenarios_chosen']} | {_percent(rates.get(outcome))} |")
    lines.append(f"| all | {selection.get('n_families')} | {selection.get('n_families_covered')} | "
                 f"{selection.get('n_chosen')} | {_percent(report['pass_rate'])} |")
    lines += ["", "A family is covered if at least one of its scenarios has a chosen episode. The pass rate "
              "is the mean over families of the family's share of passed episodes (harness errors left out).",
              "", "Counts of the selection:", ""]
    lines += [f"- {key}: {value}" for key, value in selection.items() if not isinstance(value, (dict, list))]
    lines += ["", "Records per scenario in the input (records: scenarios): "
              f"{report['attempts_per_scenario']}",
              f"Tokens of the chosen episodes: {report['n_chosen_tokens']}, of which in the loss: "
              f"{report['n_loss_tokens']}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace, command: list[str]) -> int:
    started_utc, started = utc_now(), time.perf_counter()
    check_output_folder(args.out)
    frozen = frozen_code()
    warn_if_not_as_frozen(frozen)
    episode_records = read_jsonl(args.episodes, "--episodes")
    token_rows = read_jsonl(args.tokens, "--tokens")
    try:
        records = attach_tokens(episode_records, token_rows)
    except (ValueError, KeyError) as error:
        raise Refused(f"the two input files do not belong together: {error}") from error
    if not records:
        raise Refused(f"{args.episodes} holds no record")
    source = source_run(args.episodes, len(records))
    cfg, config_source = choose_config(records, source)
    try:
        chosen, selection = select_sft_records(records, cfg, allow_penalised=args.allow_penalised)
    except ValueError as error:       # another split than train, or records of different set-ups
        raise Refused(str(error)) from error
    folder = check_scenario_folder(args.scenarios)
    check_records_against_folder(records, folder)
    check_placeholder_output(folder["summary"].get("text_is_placeholder"), args.out)
    if not chosen:
        print(json.dumps(selection, indent=2, ensure_ascii=False))
        print("no episode was chosen: there is nothing to train on (the counts are above). Nothing was "
              "written.", file=sys.stderr)
        return 1
    try:
        report = build_report(records, chosen, selection)
    except ValueError as error:       # token data that cannot be trained on
        raise Refused(f"a chosen episode cannot become a training example: {error}") from error

    args.out.mkdir(parents=True, exist_ok=True)
    write_records(args.out / RECORDS_NAME, chosen)
    rollout_script.write_json(args.out / REPORT_NAME, report)
    (args.out / TABLE_NAME).write_bytes(to_markdown(report).encode("utf-8"))
    summary = folder["summary"]
    manifest = {
        "complete": True,
        "script": SCRIPT, "command": command,
        "started_utc": started_utc, "ended_utc": utc_now(),
        "seconds": round(time.perf_counter() - started, 2),
        "python": sys.version.split()[0],
        "allow_penalised": args.allow_penalised,
        "config": cfg.to_dict(), "config_sha256": cfg.sha256(), "config_source": config_source,
        "records_config_sha256": records[0]["config_sha256"], "records_settings": records[0]["settings"],
        "impl": impl.status(),
        **frozen, "training_code": training_code(),
        "split": TRAIN_SPLIT, "text_sha256": summary["text_sha256"],
        "text_is_placeholder": summary.get("text_is_placeholder"),
        "data_sha256": folder["train_sha256"], "scenarios_config_sha256": summary.get("config_sha256"),
        "source_run": None if source is None else {key: value for key, value in source.items()
                                                   if key != "config"},
        "inputs": {"episodes": file_entry(args.episodes), "tokens": file_entry(args.tokens),
                   "summary": file_entry(args.scenarios / "summary.json"),
                   "train_file": file_entry(args.scenarios / TRAIN_FILE)},
        "outputs": {name: file_entry(args.out / name) for name in OUTPUT_FILES if name != MANIFEST_NAME},
        "counts": {"records": len(records), "chosen": len(chosen), "loss_tokens": report["n_loss_tokens"]},
    }
    rollout_script.write_json(args.out / MANIFEST_NAME, manifest)
    print(to_markdown(report))
    print(f"chose {len(chosen)} of {len(records)} episodes ({report['n_loss_tokens']} tokens in the loss); "
          f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args, command_line(SCRIPT, argv))
    except Refused as refusal:
        print(str(refusal), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
