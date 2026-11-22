"""Stage 2 training: GRPO on the train split, on top of the SFT adapter. Runs on the rented GPU.

    python scripts/gpu/train_grpo.py --model Qwen/Qwen3.5-2B --adapter results/main/sft/adapter \\
        --scenarios data/scenarios --out results/main/grpo_seed1 --seed 1 --max-new-tokens 260

The training itself is the library's (`accessdesk.train.grpo.train_grpo`): each step samples `group size`
attempts on each of `scenarios per step` training scenarios, takes reward minus the mean of the group as the
advantage, drops groups whose rewards are all equal, and makes one update. This script checks the inputs,
loads the model with the SFT adapter as a trainable adapter (it is trained further; no second adapter is
added), calls `train_grpo`, and writes the run manifest. It copies none of the library's logic.

The defaults are the values of the plan (PLAN.md section 5.6): 80 steps, 16 scenarios per step, 8 attempts per
scenario, learning rate 1e-5, gradient norm clipped at 1.0, a checkpoint every 10 steps, the stop rule with
its three limits, and the three training styles (1, 2, 3) of the train split.

The two checks of the plan before the seeds:
    --pilot          15 steps at the real settings (the pilot gate).
    --overfit FILE   train only on the scenarios of the families named in FILE (one family id a line). Without
                     --steps it runs 30 steps, and without --scenarios-per-step every step uses all of these
                     scenarios (at most 16). Give --styles 1 to keep one wording of each family.
                     With few episodes per step one harness error is above the 1% limit of the stop rule;
                     --max-harness-error-rate changes that limit (for a check only, never for a seed).

The shared config (token cap, turn limit, sampling settings, reward numbers) is the default config, with
--max-new-tokens and --chunk-size as in scripts/gpu/rollout.py. If run_manifest.json of the SFT run lies
beside the adapter folder, the run is refused when its settings differ from this run's
(--allow-other-settings accepts that), so that training and evaluation cannot use different token caps
without a word.

Written into --out by the library: train_log.jsonl (one line per step), rollouts/ (the episodes of every
step; the token data too with --save-tokens), checkpoints/ and latest.json, starts.jsonl (one line for every
start of the run and one for every fault), stopped.json if the stop rule ended the run. Written by this
script: run_manifest.json with the command, model and revision, the settings of GRPOConfig and RunConfig with
their hashes, the hashes of the adapter, of the text file and of the scenario file, the seed and the run seed
that the rollouts of the training give to the harness (`sampling_run_seed`), package versions, the GPU,
times, the hash of the frozen code and the state of the freeze (`code_sha256` and `frozen`; None with a note
in `frozen_note` if accessdesk/freeze.py does not give them), the hashes of the training code, every start
of the run (`starts`), and at the end a summary of the log and `final_adapter`, the adapter of the last
checkpoint. `"complete": true` is in it only when all steps are done and the log passed its checks.
`status` is then `complete`, or `complete_after_fault` if a start of the run went on after a stop of the
stop rule. `summary` counts the final log (every step once). `steps_run_again` counts the log lines that a
restart moved away, so the harness errors of all starts together are the sum of the two counts.
`outputs.history` holds the hash of starts.jsonl and of every file that a restart moved away.

What can be traced. Every episode record of the training carries `data_sha256`, the hash of train.jsonl, as
the records of scripts/gpu/rollout.py do. `start_adapter` in the manifest holds the hashes of the adapter
files that the run started from and `lora_fingerprint_step0`, the fingerprint of the LoRA weights before
step 0; the library writes both into every checkpoint (state.json) too, so two seeds can be shown to start
from one adapter. --save-tokens also saves the token ids and the sampler's log-probabilities of every step
(rollouts/step_*.tokens.jsonl). Only with them can the logged `loss` and `logprob_gap` be computed again
from the saved files. They cost disk space: about 15 kB per episode, so about 1.9 MB per step of 128
episodes and about 155 MB for 80 steps. This is an estimate, not a measurement of a training run. (Own
calculation from one measured file: tokens.jsonl of the untrained 2B in calibration round 3, at token cap
256, has 23.1 MB for 1,536 episodes. The token lines of a training step have the same form. Episodes after
SFT have another length, and another token cap changes the size too, both about in proportion to the tokens
of an episode.) A resume must give --save-tokens if the first start did, and must not if it did not.

--resume continues a run that stopped (a lost machine): the model is loaded with the adapter of the last
checkpoint, and the library reads the optimiser state and the totals back and checks the adapter's
fingerprint. The scenarios and the sampling seeds of a step depend only on (seed, step), so no step is
replayed with other scenarios and none is skipped; this script checks that on the finished log. A resume is
refused if the command differs from the first start in the model, the adapter, the data or the settings.
The steps after the last checkpoint are run again. What the start before wrote for them is not deleted: the
library moves it to train_log.start_<n>.jsonl, rollouts/start_<n>/ and stopped.start_<n>.json.

A run that the stop rule ended is reported as stopped (PLAN 5.6). --resume refuses to continue it (exit
code 2). --allow-resume-after-fault continues it all the same, from its last checkpoint: for a fault that
was the machine's and not the run's. The record of the stop stays on disk, and the manifest says in
`resumed_after_fault`, and in the start's entry in `starts`, that the run went on after it. When such a run
has done all its steps, its `status` is `complete_after_fault`, not `complete`; `"complete": true` only
says that all steps are done. Whether such a run may stand in a table is for the write-up to say.

Exit codes: 0 the run is complete; 2 the run was refused before the model loaded; 3 the stop rule of the
plan ended the run (the reason is printed and is in stopped.json and in the manifest); 1 any other failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for the scripts next to this file

import rollout as rollout_script  # noqa: E402
import train_sft as train_sft_script  # noqa: E402
from accessdesk import impl  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.generator import TRAIN_STYLES  # noqa: E402
from accessdesk.env.scenarios import load_families, scenario_ids  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.harness.hf_backend import resolve_revision  # noqa: E402
from accessdesk.text import TextBundle, load_text, model_text_path  # noqa: E402
from accessdesk.train.grpo import (LOG_NAME, SAME_ON_RESUME, STARTS_NAME, STOPPED_NAME,  # noqa: E402
                                   GRPOConfig, latest_checkpoint, sampling_run_seed, step_scenarios,
                                   stop_of_last_start)
from accessdesk.train.sft import TRAIN_SPLIT  # noqa: E402
from sft_data import (Refused, command_line, dict_sha256, file_entry, frozen_code,  # noqa: E402
                      training_code, utc_now, warn_if_not_as_frozen)

SCRIPT = "scripts/gpu/train_grpo.py"
MANIFEST_NAME = "run_manifest.json"
TRAIN_FILE = f"{TRAIN_SPLIT}.jsonl"
PILOT_STEPS = 15              # PLAN 5.6, the pilot gate
OVERFIT_STEPS = 30            # PLAN 5.6: the overfit check reaches a pass rate near 1 "in about 30 steps"
EXIT_STOPPED = 3              # the stop rule of the plan ended the run
RUN_FILES = (MANIFEST_NAME, LOG_NAME, "latest.json", "checkpoints", "rollouts", STOPPED_NAME, STARTS_NAME)
# What a restart moved away and kept (the library's `begin_start`): the log lines and the stopped.json of
# an earlier start, and the episode and token files of the steps that the restart ran again.
KEPT_LOGS = "train_log.start_*.jsonl"
KEPT_FILES = (KEPT_LOGS, "stopped.start_*.json", "rollouts/start_*/*.jsonl")
# The two values of `status` that go with `"complete": true`. The second one: all steps are done, but a
# start of the run went on after a stop of the stop rule (--allow-resume-after-fault).
COMPLETE, COMPLETE_AFTER_FAULT = "complete", "complete_after_fault"
# The numbers of a step that the summary repeats for the first and the last step.
STEP_KEYS = ("step", "mean_reward", "pass_rate", "kept_groups", "zero_variance_exact",
             "zero_variance_pass_fail", "invalid_call_rate", "grad_norm")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    defaults = GRPOConfig()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", required=True, help="model id on the Hub, or a local folder")
    parser.add_argument("--revision", default=None, help="model revision (default: the pinned one, if known)")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="folder of the SFT adapter; it is trained further")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file (YAML)")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="accept the placeholder text; outputs must then be under results/_smoke/")
    parser.add_argument("--scenarios", type=Path, required=True, help="folder with the split files")
    parser.add_argument("--out", type=Path, required=True, help="output folder; must not hold a run yet")
    parser.add_argument("--seed", type=int, required=True, help="seed of the scenario order and of sampling")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"training steps (default: {defaults.steps}; {PILOT_STEPS} with --pilot; "
                             f"{OVERFIT_STEPS} with --overfit)")
    parser.add_argument("--scenarios-per-step", type=int, default=None,
                        help=f"groups per step (default: {defaults.n_scenarios})")
    parser.add_argument("--group-size", type=int, default=defaults.group_size, help="attempts per scenario")
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--micro-batch", type=int, default=defaults.micro_batch,
                        help="episodes per training forward pass")
    parser.add_argument("--reference-tokens", type=int, default=defaults.reference_tokens,
                        help="the fixed token count in the constant of the loss")
    parser.add_argument("--tie-tol", type=float, default=defaults.tie_tol,
                        help="0.0: only groups with exactly equal rewards are dropped")
    parser.add_argument("--save-every", type=int, default=defaults.save_every,
                        help="steps between checkpoints")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="switch gradient checkpointing off (it is on by default; it saves memory)")
    parser.add_argument("--max-harness-error-rate", type=float, default=defaults.max_harness_error_rate,
                        help="stop rule: harness errors / episodes of the run so far")
    parser.add_argument("--max-logprob-gap", type=float, default=defaults.max_logprob_gap,
                        help="stop rule: mean |sampler - trainer| log-probability of one step")
    parser.add_argument("--max-invalid-call-rate", type=float, default=defaults.max_invalid_call_rate,
                        help="stop rule: invalid-call rate of one step (and at least twice that of step 0)")
    parser.add_argument("--usd-per-hour", type=float, default=None,
                        help="price of the machine, only to log the cost of each step")
    parser.add_argument("--styles", default=",".join(str(style) for style in TRAIN_STYLES),
                        help="wording styles to train on (default: the three training styles)")
    parser.add_argument("--overfit", type=Path, default=None, metavar="FAMILY_IDS_FILE",
                        help="the overfit check: train only on the scenarios of these families")
    parser.add_argument("--pilot", action="store_true", help=f"the pilot gate: {PILOT_STEPS} steps")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in --out from its last checkpoint")
    parser.add_argument("--allow-resume-after-fault", action="store_true",
                        help="with --resume: also continue a run that the stop rule ended; the manifest "
                             "says that this was done")
    parser.add_argument("--save-tokens", action="store_true",
                        help="also save the token data of every step (an estimate: about 1.9 MB per step of "
                             "128 episodes)")
    parser.add_argument("--allow-other-settings", action="store_true",
                        help="accept an SFT adapter whose run used other settings than this run")
    parser.add_argument("--chunk-size", type=int, default=None, help="episodes per generate call")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="token cap per turn")
    parser.add_argument("--run", default=None, help="run name in the records (default: the name of --out)")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def run_mode(args: argparse.Namespace) -> str:
    """"pilot", "overfit" or "seed" (a real training run)."""
    if args.pilot and args.overfit is not None:
        raise Refused("--pilot and --overfit are two different checks: give one of them")
    if args.pilot and args.steps is not None:
        raise Refused(f"--pilot runs {PILOT_STEPS} steps: do not give --steps with it")
    return "pilot" if args.pilot else "overfit" if args.overfit is not None else "seed"


def grpo_config(args: argparse.Namespace, n_available: int) -> GRPOConfig:
    """The GRPO settings of this run. `n_available` is the number of training scenarios after --styles and
    --overfit; it is used only for the overfit check, where a step takes all of them (at most the default)."""
    defaults = GRPOConfig()
    mode = run_mode(args)
    steps = args.steps
    if steps is None:
        steps = {"pilot": PILOT_STEPS, "overfit": OVERFIT_STEPS, "seed": defaults.steps}[mode]
    n_scenarios = args.scenarios_per_step
    if n_scenarios is None:
        n_scenarios = min(defaults.n_scenarios, n_available) if mode == "overfit" else defaults.n_scenarios
    return GRPOConfig(
        n_scenarios=n_scenarios, group_size=args.group_size, reference_tokens=args.reference_tokens,
        steps=steps, lr=args.lr, max_grad_norm=args.max_grad_norm, micro_batch=args.micro_batch,
        gradient_checkpointing=not args.no_gradient_checkpointing, tie_tol=args.tie_tol,
        save_every=args.save_every, seed=args.seed, max_harness_error_rate=args.max_harness_error_rate,
        max_logprob_gap=args.max_logprob_gap, max_invalid_call_rate=args.max_invalid_call_rate,
        usd_per_hour=args.usd_per_hour)


# ---------------------------------------------------------------------------
# Checks before the model loads. Each returns what the manifest needs, or raises Refused.
# ---------------------------------------------------------------------------
def check_text_and_output(text: TextBundle, args: argparse.Namespace) -> None:
    """The rule of scripts/gpu/rollout.py for the placeholder text."""
    smoke = rollout_script.SMOKE_DIR.resolve()
    if text.is_placeholder and not args.allow_placeholder:
        raise Refused("the text file is the placeholder; pass --allow-placeholder for a smoke run")
    if text.is_placeholder and smoke not in args.out.resolve().parents:
        raise Refused(f"with the placeholder text the output folder must be under {smoke}")


def training_families(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """The families that `train_grpo` gets, and what the manifest says about them.

    The families are those of train.jsonl (with --overfit: only the ones named in the file), each with only
    its wordings in the styles of --styles. The held-out style is refused. Every wording of every family that
    is passed on is one training scenario.
    """
    try:
        styles = sorted({int(style) for style in args.styles.split(",")})
    except ValueError as error:
        raise Refused(f"--styles must be numbers with commas between them, such as 1,2,3: {error}") from error
    if metrics.HELD_OUT_STYLE in styles:
        raise Refused(f"style {metrics.HELD_OUT_STYLE} is the held-out style: it is never trained on")
    families = load_families(args.scenarios / TRAIN_FILE)
    overfit = None
    if args.overfit is not None:
        if not args.overfit.is_file():
            raise Refused(f"--overfit: {args.overfit} is missing")
        lines = args.overfit.read_text(encoding="utf-8").splitlines()
        wanted = [line.strip() for line in lines if line.strip()]
        unknown = sorted(set(wanted) - {family["family_id"] for family in families})
        if unknown or not wanted:
            raise Refused(f"--overfit: {args.overfit} must name families of the train split; not in it: "
                          f"{unknown[:5]}")
        families = [family for family in families if family["family_id"] in set(wanted)]
        overfit = {"file": file_entry(args.overfit),
                   "family_ids": [family["family_id"] for family in families]}
    kept = [{**family, "wordings": [wording for wording in family["wordings"] if wording["style"] in styles]}
            for family in families]
    empty = [family["family_id"] for family in kept if not family["wordings"]]
    if empty:
        raise Refused(f"{len(empty)} families have no wording in the styles {styles}, for example {empty[0]}")
    ids = sorted(scenario_id for family in kept for scenario_id in scenario_ids(family))
    info = {"families": len(kept), "scenarios": len(ids), "styles": styles, "overfit": overfit,
            "scenario_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()}
    return kept, info


def sft_adapter(args: argparse.Namespace, cfg: RunConfig) -> dict:
    """The hashes of the adapter that the run starts from, and what the manifest of the run that trained it
    (run_manifest.json beside the adapter folder) says.

    Refused if the folder holds no adapter; and, with that manifest: if its run is not complete, if it
    trained another model or revision, or if its settings (`RunConfig.compared_settings`) differ from this
    run's and --allow-other-settings is not given.
    """
    files = train_sft_script.adapter_entries(args.adapter)
    if None in files.values():
        raise Refused(f"{args.adapter} is not a saved adapter: {sorted(files)} must be in it")
    info: dict[str, Any] = {"folder": str(args.adapter), "files": files, "trained_by": None, "note": None}
    manifest_file = args.adapter.parent / MANIFEST_NAME
    if not manifest_file.is_file():
        info["note"] = "no run_manifest.json beside the adapter folder: the run that trained it is not known"
        return info
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise Refused(f"{manifest_file} says that the run that trained the adapter is not complete")
    wanted = (args.model, resolve_revision(args.model, args.revision))
    found = (manifest.get("model"), manifest.get("revision"))
    if found != wanted:
        raise Refused(f"the adapter was trained on (model, revision) {found}, not on {wanted}")
    theirs = manifest.get("config") or {}
    different = {name: (theirs.get(name), value) for name, value in cfg.compared_settings().items()
                 if theirs.get(name) != value}
    if different:
        message = (f"the run that trained the adapter used other settings than this run (theirs, this "
                   f"run's): {different}")
        if not args.allow_other_settings:
            raise Refused(message + "; give the same --max-new-tokens, or pass --allow-other-settings")
        info["note"] = message + "; accepted with --allow-other-settings"
    info["trained_by"] = {"manifest": file_entry(manifest_file), "script": manifest.get("script"),
                          "seed": manifest.get("seed"), "config_sha256": manifest.get("config_sha256"),
                          "sft_config_sha256": manifest.get("sft_config_sha256")}
    return info


def earlier_run(out_dir: Path, resume: bool, identity: dict) -> tuple[dict | None, dict | None]:
    """(the manifest of the run in `out_dir`, its latest checkpoint) with --resume; (None, None) for a new
    run. Refused: a new run in a folder that holds a run; a resume without a manifest or a checkpoint; a
    resume whose command differs from the first start in one of the values of `identity`."""
    if not resume:
        held = [name for name in RUN_FILES if (out_dir / name).exists()]
        if held:
            raise Refused(f"{out_dir} already holds {', '.join(held)}: pass --resume to continue that run, "
                          "or choose a new output folder")
        return None, None
    manifest_file = out_dir / MANIFEST_NAME
    saved = latest_checkpoint(out_dir)
    if not manifest_file.is_file() or saved is None:
        raise Refused(f"--resume: {out_dir} has no run manifest or no checkpoint (latest.json) to continue "
                      "from; start the run again in a new output folder")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    before = manifest.get("identity") or {}
    different = sorted(key for key, value in identity.items() if before.get(key) != value)
    if different:
        raise Refused(f"--resume: this command differs from the run in {out_dir} in: {', '.join(different)}")
    return manifest, saved


def stop_before(out_dir: Path, before: dict, allow_resume_after_fault: bool) -> dict | None:
    """With --resume: the stop of the stop rule that ended the last start of the run in `out_dir` (`step`,
    `reason`, `start`), or None if the rule did not end it.

    A run that the stop rule ended is reported as stopped (PLAN 5.6), so it is not continued: Refused,
    unless --allow-resume-after-fault is given. The stop is read from the folder by the library
    (`stop_of_last_start`), and from the status of the earlier run manifest (`before`).
    """
    stop = stop_of_last_start(out_dir)
    if stop is None and before.get("status") == "stopped_by_rule":
        stop = {"step": None, "reason": None, "start": None, **(before.get("fault") or {})}
    if stop is not None and not allow_resume_after_fault:
        raise Refused(f"--resume: the stop rule ended the run in {out_dir} at step {stop['step']}: "
                      f"{stop['reason']}. A stopped run is reported as stopped (PLAN 5.6) and is not "
                      "continued. If the fault was the machine's and not the run's, pass "
                      "--allow-resume-after-fault; the manifest will say that the run went on after a stop.")
    return stop


def history_files(out_dir: Path) -> dict:
    """The files that hold the history of the starts of the run, each with its hash: starts.jsonl, and what
    a restart moved away (the log lines and the stopped.json of an earlier start, and the episode and token
    files of the steps that were run again). The key of a file is its path from the run folder."""
    kept = [path for pattern in KEPT_FILES for path in sorted(out_dir.glob(pattern))]
    return {path.relative_to(out_dir).as_posix(): file_entry(path)
            for path in [out_dir / STARTS_NAME, *kept] if path.is_file()}


def steps_run_again(out_dir: Path) -> dict:
    """Totals of the log lines that a restart moved away (train_log.start_<n>.jsonl): the steps after a
    checkpoint that a later start ran again.

    `summarise` does not count these lines: it counts the final log, in which every step stands once, as
    the last start that ran it wrote it. The stop rule of the later start did not count them either: its
    totals began again at the checkpoint. So the harness errors of all starts together are
    `summary.harness_errors` plus `harness_errors` here.

        lines             the log lines that were moved away
        steps             their step numbers, each once, in order
        episodes, harness_errors   the sums of `n_episodes` and `n_harness_errors` over those lines
        unreadable_lines  lines that are not a JSON object (a start that died in the middle of a write)

    A line that stands twice in one file is counted once: a restart that died in the middle of its file
    steps can leave the same lines there twice (the library's `begin_start`).
    """
    logs, unreadable = [], 0
    for path in sorted(out_dir.glob(KEPT_LOGS)):
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line in seen:
                continue
            seen.add(line)
            try:
                log = json.loads(line)
            except ValueError:
                log = None
            if isinstance(log, dict) and "step" in log:
                logs.append(log)
            else:
                unreadable += 1
    return {"lines": len(logs), "steps": sorted({log["step"] for log in logs}),
            "episodes": sum(log.get("n_episodes", 0) for log in logs),
            "harness_errors": sum(log.get("n_harness_errors", 0) for log in logs),
            "unreadable_lines": unreadable}


# ---------------------------------------------------------------------------
# The GPU part
# ---------------------------------------------------------------------------
def setup_model(args: argparse.Namespace, cfg: RunConfig, adapter_dir: Path) -> dict:
    """Load the model with the adapter of `adapter_dir` as a trainable adapter, and check its coverage.

    Returns `model`, `tokenizer`, and the manifest parts `lora`, `environment`, `kernels` and `sampler`.
    The sampler built here is only asked for its settings; `train_grpo` builds its own from the same config.
    """
    from accessdesk.harness.hf_backend import (Sampler, causal_lm, environment_report, kernel_report,
                                               load_model)
    from accessdesk.train.lora import (assert_coverage, coverage_summary, coverage_table, lora_fingerprint,
                                       trainable_parameters)

    model, tokenizer = load_model(args.model, args.revision, adapter_dir=str(adapter_dir),
                                  adapter_trainable=True, device=args.device)
    table = coverage_table(model)
    assert_coverage(table, causal_lm(model).config)
    lora = {"loaded_from": str(adapter_dir), "wrapped_modules": sum(1 for row in table if row["wrapped"]),
            "trainable_parameters": sum(param.numel() for param in trainable_parameters(model)),
            "fingerprint_at_start": lora_fingerprint(model), "coverage": coverage_summary(table)}
    return {"model": model, "tokenizer": tokenizer, "lora": lora, "environment": environment_report(),
            "kernels": kernel_report(model), "sampler": Sampler(model, tokenizer, cfg).settings()}


def run_training(setup: dict, families: list[dict], text: TextBundle, cfg: RunConfig, gcfg: GRPOConfig,
                 args: argparse.Namespace) -> list[dict]:
    """`train_grpo` of the library with this run's settings. Returns its log lines.

    `setup["origin"]` (put there by `run`) holds what the library records about where the run comes from:
    `data_sha256`, the hash of train.jsonl, which goes into every episode record of the training; and
    `start_adapter`, the hashes of the files of the adapter that the run started from.
    """
    from accessdesk.train.grpo import train_grpo

    origin = setup.get("origin") or {}
    return train_grpo(setup["model"], setup["tokenizer"], families, text, cfg, gcfg, out_dir=args.out,
                      run=args.run or args.out.name, resume=args.resume, save_tokens=args.save_tokens,
                      allow_resume_after_fault=args.allow_resume_after_fault,
                      data_sha256=origin.get("data_sha256"), start_adapter=origin.get("start_adapter"))


def peak_gpu_memory_gb() -> dict | None:
    return train_sft_script.peak_gpu_memory_gb()


# ---------------------------------------------------------------------------
# After the training (no torch)
# ---------------------------------------------------------------------------
def log_problems(logs: list[dict], ids: list[str], gcfg: GRPOConfig, saved: dict | None) -> list[str]:
    """What is wrong with the log of a run that ended without a fault; an empty list if all is well.

      - the log holds the steps 0, 1, ..., steps - 1, each once and in order;
      - every step used exactly the scenarios that the library's `step_scenarios` gives for (seed, step), so
        a restart replayed no step with other scenarios and skipped none;
      - the latest checkpoint is the one after the last step.
    """
    problems = []
    steps = [log["step"] for log in logs]
    if steps != list(range(gcfg.steps)):
        problems.append(f"the log holds the steps {steps[:5]}... ({len(steps)} lines), not 0 to "
                        f"{gcfg.steps - 1}")
    wrong = [log["step"] for log in logs
             if log["scenarios"] != step_scenarios(ids, gcfg.n_scenarios, gcfg.seed, log["step"])]
    if wrong:
        problems.append(f"{len(wrong)} steps did not use the scenarios of the seeded order, first: "
                        f"{wrong[0]}")
    if saved is None or saved["completed_steps"] != gcfg.steps:
        done = None if saved is None else saved["completed_steps"]
        problems.append(f"the latest checkpoint is after step {done}, not after step {gcfg.steps}")
    return problems


def summarise(logs: list[dict], gcfg: GRPOConfig) -> dict:
    """Totals of the training log. Every number is taken from the log lines of the library.

    They are the totals of the final log: every step once, as the last start that ran it wrote it. The lines
    of steps that a restart ran again are not counted here; `steps_run_again` counts them.

        steps               log lines
        groups, kept_groups groups sampled (steps x scenarios per step), and those that were not dropped
        episodes            episodes sampled, harness errors included; harness_errors: how many of them
        trained_episodes    episodes that went through the training forward (valid episodes of kept groups)
        loss_tokens         their sampled tokens
        seconds_in_steps    rollout and update of all steps, without loading and checkpoints
        cost_usd            the sum of the steps' cost, or None without --usd-per-hour
        first_step, last_step   a few numbers of that step, with the names of the log. They describe the
                            TRAINING episodes of the step (sampled on train scenarios), not the dev split.
    """
    costs = [log["cost_usd"] for log in logs if log.get("cost_usd") is not None]

    def of_step(log: dict) -> dict:
        gap = log.get("logprob_gap")
        return {**{key: log.get(key) for key in STEP_KEYS}, "logprob_gap_mean": gap["mean"] if gap else None}

    return {
        "steps": len(logs), "groups": len(logs) * gcfg.n_scenarios,
        "kept_groups": sum(log["kept_groups"] for log in logs),
        "episodes": sum(log["n_episodes"] for log in logs),
        "harness_errors": sum(log["n_harness_errors"] for log in logs),
        "trained_episodes": sum(log["n_trained_episodes"] for log in logs),
        "loss_tokens": sum(log["n_loss_tokens"] for log in logs),
        "seconds_in_steps": round(sum(log["seconds"] for log in logs), 1),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "first_step": of_step(logs[0]) if logs else None, "last_step": of_step(logs[-1]) if logs else None,
    }


def run(args: argparse.Namespace, command: list[str]) -> int:
    started_utc, started = utc_now(), time.perf_counter()
    mode = run_mode(args)
    if args.allow_resume_after_fault and not args.resume:
        raise Refused("--allow-resume-after-fault is an option of --resume: give both, or none")
    text = load_text(args.text)
    cfg = rollout_script.run_config(args)
    check_text_and_output(text, args)
    problem = rollout_script.check_scenarios(args.scenarios, text, (TRAIN_FILE,))
    if problem:
        raise Refused(problem)
    families, selection = training_families(args)
    gcfg = grpo_config(args, selection["scenarios"])
    if selection["scenarios"] < gcfg.n_scenarios:
        raise Refused(f"{selection['scenarios']} training scenarios are not enough for steps of "
                      f"{gcfg.n_scenarios}; give a smaller --scenarios-per-step")
    adapter = sft_adapter(args, cfg)
    data_sha256 = rollout_script.sha256_of_file(args.scenarios / TRAIN_FILE)
    revision = resolve_revision(args.model, args.revision)
    # What a resumed run must share with its first start. `steps` is not in it: the library lets it change.
    # `save_tokens` is in it, so that a run has the token data of all its steps or of none.
    identity = {
        "model": args.model, "revision": revision,
        "adapter_sha256": {name: entry["sha256"] for name, entry in adapter["files"].items()},
        "text_sha256": text.sha256, "train_file_sha256": data_sha256,
        "scenario_ids_sha256": selection["scenario_ids_sha256"], "config_sha256": cfg.sha256(),
        "grpo_config": {key: gcfg.to_dict()[key] for key in SAME_ON_RESUME},
        "save_tokens": args.save_tokens,
    }
    before, saved = earlier_run(args.out, args.resume, identity)
    if before is not None and before.get("complete") is True and saved["completed_steps"] >= gcfg.steps:
        print(f"{args.out} is already complete ({saved['completed_steps']} steps): nothing to do")
        return 0
    stop = None if before is None else stop_before(args.out, before, args.allow_resume_after_fault)
    adapter_dir = args.adapter if saved is None else args.out / saved["checkpoint"] / "adapter"
    frozen = frozen_code()
    warn_if_not_as_frozen(frozen)

    setup = setup_model(args, cfg, adapter_dir)          # from here on the GPU is used
    setup["origin"] = {"data_sha256": data_sha256, "start_adapter": identity["adapter_sha256"]}
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = (setup.get("lora") or {}).get("fingerprint_at_start")      # of this start's weights
    # The fingerprint of the LoRA weights before step 0: this start's on a first start, and on a resume the
    # one that the first start wrote into the manifest.
    adapter["lora_fingerprint_step0"] = (fingerprint if before is None else
                                         (before.get("start_adapter") or {}).get("lora_fingerprint_step0"))
    start = {"started_utc": started_utc, "ended_utc": None, "seconds": None, "command": command,
             "resumed_after_step": None if saved is None else saved["completed_steps"],
             "allow_resume_after_fault": args.allow_resume_after_fault, "resumed_after_fault": stop,
             "lora_fingerprint": fingerprint,
             "steps": gcfg.steps, "result": "running", "fault": None,
             "gpu_name": (setup["environment"] or {}).get("gpu_name")}
    starts = ([] if before is None else list(before.get("starts", []))) + [start]
    manifest: dict[str, Any] = {
        "complete": False, "status": "running", "mode": mode,
        "script": SCRIPT, "command": command,
        "started_utc": starts[0]["started_utc"], "ended_utc": None, "seconds": None,
        "model": args.model, "revision": revision, "seed": args.seed,
        "sampling_run_seed": sampling_run_seed(args.seed), "save_tokens": args.save_tokens,
        "grpo_config": gcfg.to_dict(), "grpo_config_sha256": dict_sha256(gcfg.to_dict()),
        "norm_const": gcfg.norm_const,
        "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "text_file": Path(text.path).name, "text_sha256": text.sha256,
        "text_is_placeholder": text.is_placeholder,
        "split": TRAIN_SPLIT, "scenario_folder": str(args.scenarios), "split_file_sha256": data_sha256,
        **selection,
        "start_adapter": adapter, "lora": setup["lora"],
        "impl": impl.status(),
        **frozen, "training_code": training_code(),
        "environment": setup["environment"], "kernels": setup["kernels"], "sampler": setup["sampler"],
        "identity": identity, "starts": starts,
        # Every stop of the stop rule that a later start went on after (--allow-resume-after-fault). An
        # empty list: no start of this run went on after a stop.
        "resumed_after_fault": [item["resumed_after_fault"] for item in starts
                                if item.get("resumed_after_fault")],
        "fault": None, "error": None, "summary": None, "final_adapter": None,
    }
    rollout_script.write_json(args.out / MANIFEST_NAME, manifest)

    print(f"GRPO ({mode}) of {args.model}: {gcfg.steps} steps of {gcfg.n_scenarios} scenarios x "
          f"{gcfg.group_size} attempts on {selection['scenarios']} training scenarios, seed {gcfg.seed}, "
          f"learning rate {gcfg.lr}; results in {args.out}", flush=True)
    logs: list[dict] = []
    error = fault = None
    try:
        logs = run_training(setup, families, text, cfg, gcfg, args)
        faulty = [log for log in logs if log["fault"] is not None]
        if faulty:
            fault = {"step": faulty[0]["step"], "reason": faulty[0]["fault"]}
        else:
            ids = sorted(scenario_id for family in families for scenario_id in scenario_ids(family))
            problems = log_problems(logs, ids, gcfg, latest_checkpoint(args.out))
            error = "; ".join(problems) or None
    except Exception as exc:      # the manifest must say that the run failed
        error = f"{type(exc).__name__}: {exc}"[:2000]
        print(traceback.format_exc(), file=sys.stderr)

    # All steps are done and the log passed its checks: "complete". If a start of this run went on after a
    # stop of the stop rule, the status says so, so that nobody takes the run for one without a stop.
    done = COMPLETE_AFTER_FAULT if manifest["resumed_after_fault"] else COMPLETE
    status = "stopped_by_rule" if fault else "failed" if error else done
    last = latest_checkpoint(args.out)
    start.update({"ended_utc": utc_now(), "seconds": round(time.perf_counter() - started, 1),
                  "result": status, "fault": fault, "peak_gpu_memory_gb": peak_gpu_memory_gb()})
    manifest.update({
        "complete": status in (COMPLETE, COMPLETE_AFTER_FAULT), "status": status, "fault": fault,
        "error": error,
        "ended_utc": start["ended_utc"], "seconds": round(sum(item["seconds"] or 0 for item in starts), 1),
        "summary": summarise(logs, gcfg) if logs else None,
        "steps_run_again": steps_run_again(args.out),
        "last_checkpoint": last,
        "outputs": {"log": file_entry(args.out / LOG_NAME) if (args.out / LOG_NAME).is_file() else None,
                    "history": history_files(args.out)},
    })
    if manifest["complete"]:
        final = Path(last["checkpoint"]) / "adapter"
        manifest["final_adapter"] = {"folder": str(final),
                                     "files": train_sft_script.adapter_entries(args.out / final)}
    rollout_script.write_json(args.out / MANIFEST_NAME, manifest)

    if fault:
        kept = "no checkpoint was saved" if last is None else f"the last checkpoint is {last['checkpoint']}"
        print(f"STOPPED BY THE STOP RULE at step {fault['step']}: {fault['reason']}\n"
              f"The run is reported as stopped (PLAN 5.6); {kept}. See {args.out / 'stopped.json'}.",
              file=sys.stderr)
        return EXIT_STOPPED
    if error:
        print(f"GRPO FAILED: {error}", file=sys.stderr)
        return 1
    summary = manifest["summary"]
    print(f"{summary['steps']} steps in {manifest['seconds']} s; kept {summary['kept_groups']} of "
          f"{summary['groups']} groups; train reward {summary['first_step']['mean_reward']} at step 0 and "
          f"{summary['last_step']['mean_reward']} at the last step; adapter in "
          f"{args.out / manifest['final_adapter']['folder']}")
    for earlier in manifest["resumed_after_fault"]:
        print(f"NOTE: this run went on after the stop rule had ended it at step {earlier['step']} "
              f"({earlier['reason']}). The write-up must say so; the record is in {args.out / STARTS_NAME}.")
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
