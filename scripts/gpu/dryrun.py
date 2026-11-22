"""A tiny end-to-end run of every stage. It proves only that the pieces fit together.

    python scripts/gpu/dryrun.py --model Qwen/Qwen3.5-2B --text text/placeholder_text.yaml --allow-placeholder

NONE OF ITS NUMBERS IS A RESULT. The run is far too small, and with the placeholder text it uses text that no
reported run uses. Stages, in order:

  0. setup         load the model, add the LoRA adapters, check their coverage, build the sampler
  0b. dev_before   one attempt on every wording of 8 dev families with the untrained model (adapters attached,
                   zero); stage 6 repeats it with the same sampling seeds after training
  1. rollout       4 attempts on each of 40 train scenarios with the untrained model (adapters attached, zero)
  2. sft_data      the SFT selection on those episodes. With fewer than 8 clean passes it says so and also
                   takes passes with penalties. If the model did not pass a single episode, it says so and
                   uses the scripted correct solutions of 8 scenarios instead, only so that the SFT code runs.
  3. sft           --sft-epochs epochs (default 2) of SFT on what stage 2 chose; the adapter is saved
  4. grpo          two GRPO steps of 4 scenarios x 8 attempts with a checkpoint after each step; then the
                   model is thrown away, loaded again from the Hub cache with the adapter of the last
                   checkpoint (trainable), and the loop is restarted for the third step. This runs the
                   checkpoint, the adapter save and load, and the resume code as a real restart would.
                   The loop gets the hash of the train families and the hashes of the adapter of stage 3,
                   as scripts/gpu/train_grpo.py gives them (`grpo_origin`).
  5. gradient      only if no group was kept in stage 4 (all attempts of every scenario tied): one forward and
                   backward pass of the GRPO loss with made-up advantages, so that this code has run too
  6. dev_after     one attempt on every wording of 8 dev families with the trained adapter
  7. weights       how far the adapters of each block moved since the start (must not be zero)

Every stage prints one line with its numbers. If a stage fails, the error is printed, the stages after it are
not run, and the exit code is 1. Written into results/_smoke/<name>/dryrun/: dryrun.json, run_manifest.json,
and the files of each stage in a folder of its name.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for smoke.py and rollout.py, next to this

import rollout as rollout_script  # noqa: E402
import smoke  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.scenarios import save_families  # noqa: E402
from accessdesk.harness.episode import Episode  # noqa: E402
from accessdesk.harness.tokens import TokenLayout  # noqa: E402
from accessdesk.train.grpo import GRPOConfig  # noqa: E402
from accessdesk.train.sft import SFTConfig, select_sft_records, training_example  # noqa: E402

MIN_CLEAN_PASSES = 8
RUN_SEED = 0
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")     # the two files of a saved adapter


def roll(ctx: dict, scenarios: list[tuple[dict, int]], samples: int, arm: str, folder: Path) -> list[dict]:
    """`samples` attempts on every scenario, one pass per attempt; the episodes are saved in `folder`.
    Returns the records with their token data."""
    from accessdesk.harness.hf_backend import rollout

    folder.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for sample in range(samples):
        episodes = [Episode(family, style, ctx["text"], ctx["cfg"]) for family, style in scenarios]
        new = rollout(ctx["sampler"], ctx["layouts"], episodes, run_seed=RUN_SEED, pass_index=sample,
                      meta={"run": ctx["out_dir"].parent.name, "arm": arm, "first_sample": sample})
        rollout_script.append_jsonl(folder / "tokens.jsonl",
                                    [rollout_script.token_line(record) for record in new])
        rollout_script.append_jsonl(folder / "episodes.jsonl",
                                    [{key: value for key, value in record.items() if key != "tokens"}
                                     for record in new])
        records.extend(new)
    return records


def summary_of(records: list[dict]) -> dict:
    """The numbers of scripts/gpu/rollout.py for a list of records (with or without token data)."""
    return rollout_script.summarise([{key: value for key, value in record.items() if key != "tokens"}
                                     for record in records])


# ---------------------------------------------------------------------------
# The stages. Each takes the shared dict `ctx` and returns its numbers.
# ---------------------------------------------------------------------------
def stage_setup(ctx: dict) -> dict:
    from accessdesk.harness.hf_backend import Sampler, causal_lm, load_model
    from accessdesk.train.lora import (add_lora, assert_coverage, coverage_table, lora_state,
                                       trainable_parameters)

    args, text, cfg = ctx["args"], ctx["text"], ctx["cfg"]
    model, tokenizer = load_model(args.model, args.revision, device=args.device)
    model = add_lora(model, seed=0)
    table = coverage_table(model)
    assert_coverage(table, causal_lm(model).config)
    sampler = Sampler(model, tokenizer, cfg)
    ctx.update({"model": model, "tokenizer": tokenizer, "sampler": sampler,
                "layouts": {"five": TokenLayout(tokenizer, text.tool_specs("five"))},
                "initial_lora": lora_state(model)})
    note = "a dry run: none of its numbers is a result"
    manifest = smoke.run_manifest(args, cfg, text, model=model, sampler=sampler,
                                  extra={"complete": False, "note": note})
    ctx["manifest"] = manifest
    rollout_script.write_json(ctx["out_dir"] / "run_manifest.json", manifest)
    return {"wrapped_modules": sum(1 for row in table if row["wrapped"]),
            "trainable_parameters": sum(param.numel() for param in trainable_parameters(model)),
            "gated_delta_rule": manifest["kernels"]["gated_delta_rule"],
            "causal_conv1d": manifest["kernels"]["causal_conv1d"], "gpu": manifest["environment"]["gpu_name"]}


def stage_rollout(ctx: dict) -> dict:
    args = ctx["args"]
    train = ctx["families"]["train"]
    scenarios = [(family, wording["style"]) for family in train for wording in family["wordings"]]
    scenarios = scenarios[:args.train_scenarios]
    started = time.perf_counter()
    ctx["train_records"] = roll(ctx, scenarios, args.samples, "dryrun_base_train", ctx["out_dir"] / "rollout")
    seconds = time.perf_counter() - started
    summary = summary_of(ctx["train_records"])
    return {"scenarios": len(scenarios), "samples": args.samples, **summary, "seconds": round(seconds, 1),
            "episodes_per_hour": round(len(ctx["train_records"]) / seconds * 3600)}


def stage_sft_data(ctx: dict) -> dict:
    cfg, text = ctx["cfg"], ctx["text"]
    chosen, report = select_sft_records(ctx["train_records"], cfg)
    source = "clean passes"
    if report["n_chosen"] < MIN_CLEAN_PASSES:
        print(f"      only {report['n_chosen']} scenarios have a clean pass (fewer than {MIN_CLEAN_PASSES}): "
              "passes with penalties are used too")
        chosen, report = select_sft_records(ctx["train_records"], cfg, allow_penalised=True)
        source = "passes, with penalties allowed"
    examples = [training_example(record) for record in chosen]
    if not examples:
        print("      the model passed no episode: the scripted correct solutions of 8 train scenarios are "
              "used instead, ONLY so that the SFT code runs")
        families = ctx["families"]["train"][:MIN_CLEAN_PASSES]
        examples = [smoke.oracle_example(family, 1, text, cfg, ctx["layouts"]["five"]) for family in families]
        source = "scripted solutions (the model had no pass)"
    ctx["sft_examples"] = examples
    rollout_script.write_json(ctx["out_dir"] / "sft_data_report.json", {"source": source, **report})
    return {"source": source, "examples": len(examples),
            "loss_tokens": sum(sum(example["weights"]) for example in examples),
            "clean_passes": report["n_clean_passes"], "passes": report["n_passed"],
            "families_covered": report["n_families_covered"], "families": report["n_families"],
            "by_outcome": report["by_outcome"]}


def stage_sft(ctx: dict) -> dict:
    from accessdesk.train.sft import train_sft

    sft_cfg = SFTConfig(epochs=ctx["args"].sft_epochs)
    logs = train_sft(ctx["model"], ctx["sft_examples"], sft_cfg, pad_id=ctx["sampler"].pad_id,
                     out_dir=ctx["out_dir"] / "sft")
    # The mean loss of each epoch (weighted by loss tokens): every epoch sees the same examples once, so
    # these numbers can be compared; the first and the last step are different batches.
    by_epoch = []
    for epoch in range(sft_cfg.epochs):
        steps = [log for log in logs if log["epoch"] == epoch]
        by_epoch.append(sum(log["loss"] * log["n_tokens"] for log in steps)
                        / sum(log["n_tokens"] for log in steps))
    return {"steps": len(logs), "first_loss": logs[0]["loss"], "last_loss": logs[-1]["loss"],
            "mean_loss_by_epoch": by_epoch,
            "losses": [log["loss"] for log in logs], "grad_norms": [log["grad_norm"] for log in logs],
            "first_grad_norm": logs[0]["grad_norm"], "seconds": round(sum(log["seconds"] for log in logs), 1),
            "config": sft_cfg.to_dict()}


def reload_from_checkpoint(ctx: dict, adapter_dir: Path) -> dict:
    """Throw the model away and load it again with the saved adapter, trainable: what a restart in a new
    process does. Returns the fingerprints of the LoRA weights before and after."""
    from accessdesk.harness.hf_backend import Sampler, load_model
    from accessdesk.train.lora import lora_fingerprint, trainable_parameters

    args = ctx["args"]
    before = lora_fingerprint(ctx["model"])
    for key in ("model", "sampler"):                 # every reference to the old model
        ctx.pop(key)
    smoke.free_gpu_memory()
    model, tokenizer = load_model(args.model, args.revision, adapter_dir=str(adapter_dir),
                                  adapter_trainable=True, device=args.device)
    ctx.update({"model": model, "tokenizer": tokenizer, "sampler": Sampler(model, tokenizer, ctx["cfg"])})
    return {"adapter": str(adapter_dir.relative_to(ctx["out_dir"])), "fingerprint_before": before,
            "fingerprint_after": lora_fingerprint(model),
            "trainable_parameters_after": sum(param.numel() for param in trainable_parameters(model))}


def grpo_origin(ctx: dict) -> dict:
    """The two arguments of `train_grpo` that say where a run comes from, as scripts/gpu/train_grpo.py
    gives them: `data_sha256`, the hash of the scenario file, and `start_adapter`, the hashes of the two
    files of the adapter that the run starts from.

    The dry run builds its families in memory and has no scenario file. So the train families are written
    to grpo_train.jsonl in the output folder first (`save_families`, as the build writes them), and that
    file is hashed. The adapter is the one that stage 3 saved in sft/adapter: the model still holds its
    weights when stage 4 starts."""
    train_file = ctx["out_dir"] / "grpo_train.jsonl"
    save_families(train_file, ctx["families"]["train"])
    adapter_dir = ctx["out_dir"] / "sft" / "adapter"
    return {"data_sha256": rollout_script.sha256_of_file(train_file),
            "start_adapter": {name: rollout_script.sha256_of_file(adapter_dir / name)
                              for name in ADAPTER_FILES}}


def stage_grpo(ctx: dict) -> dict:
    """Two steps, then a restart for the third: the model is loaded again with the adapter of the last
    checkpoint (`reload_from_checkpoint`), and the optimiser state and the run's totals are read back from
    the checkpoint files by `train_grpo(resume=True)`, which also checks the fingerprint of the adapter.
    Both calls get the same `data_sha256` and `start_adapter` (`grpo_origin`): the restart is refused if
    one of them is not the value of the first start."""
    from accessdesk.train.grpo import latest_checkpoint, train_grpo

    args, out = ctx["args"], ctx["out_dir"] / "grpo"
    # A dry run of 32 episodes per step must not stop for one harness error; the other stop rules stay.
    gcfg = GRPOConfig(n_scenarios=4, group_size=8, steps=args.grpo_steps - 1, save_every=1, seed=RUN_SEED,
                      max_harness_error_rate=0.25)
    common = {"out_dir": out, "run": ctx["out_dir"].parent.name, "save_tokens": True, **grpo_origin(ctx)}
    train = ctx["families"]["train"]
    logs = train_grpo(ctx["model"], ctx["tokenizer"], train, ctx["text"], ctx["cfg"], gcfg, **common)
    faults = [log["fault"] for log in logs if log["fault"] is not None]
    reload = None
    if not faults:
        saved = latest_checkpoint(out)
        if saved is None or saved["completed_steps"] != gcfg.steps:
            raise RuntimeError(f"expected a checkpoint after step {gcfg.steps}, found {saved}")
        reload = reload_from_checkpoint(ctx, out / saved["checkpoint"] / "adapter")
        print(f"      restart: {reload}")
        resumed = dataclasses.replace(gcfg, steps=args.grpo_steps)
        logs = train_grpo(ctx["model"], ctx["tokenizer"], train, ctx["text"], ctx["cfg"], resumed,
                          resume=True, **common)
        faults = [log["fault"] for log in logs if log["fault"] is not None]
    ctx["grpo_logs"] = logs
    for log in logs:
        gap = log["logprob_gap"]["mean"] if log["logprob_gap"] else None
        print(f"      step {log['step']}: reward {log.get('mean_reward')}, pass rate {log.get('pass_rate')}, "
              f"kept groups {log['kept_groups']}/4, zero variance exact {log.get('zero_variance_exact')} / "
              f"pass-fail {log.get('zero_variance_pass_fail')}, invalid calls {log.get('invalid_call_rate')}, "
              f"trained episodes {log['n_trained_episodes']}, loss {log['loss']}, gap {gap}, "
              f"grad norm {log['grad_norm']}, rollout {log['rollout_seconds']} s, step {log['seconds']} s")
    if faults:
        raise RuntimeError(f"the stop rule ended the GRPO run: {faults[0]}")
    if [log["step"] for log in logs] != list(range(args.grpo_steps)):
        raise RuntimeError(f"the log holds the steps {[log['step'] for log in logs]}")
    used = [scenario for log in logs for scenario in log["scenarios"]]
    if len(set(used)) != len(used):
        raise RuntimeError("a scenario was used in two steps: the restart replayed a step")
    return {"steps": len(logs), "kept_groups": [log["kept_groups"] for log in logs],
            "mean_reward": [log.get("mean_reward") for log in logs],
            "zero_variance_exact": [log.get("zero_variance_exact") for log in logs],
            "zero_variance_pass_fail": [log.get("zero_variance_pass_fail") for log in logs],
            "invalid_call_rate": [log.get("invalid_call_rate") for log in logs],
            "logprob_gap_mean": [log["logprob_gap"]["mean"] if log["logprob_gap"] else None for log in logs],
            "grad_norm": [log["grad_norm"] for log in logs],
            "loss": [log["loss"] for log in logs],
            "trained_episodes_per_step": [log["n_trained_episodes"] for log in logs],
            "trained_episodes": sum(log["n_trained_episodes"] for log in logs),
            "max_logprob_gap_mean": max((log["logprob_gap"]["mean"] for log in logs if log["logprob_gap"]),
                                        default=None),
            "rollout_seconds_per_step": [log["rollout_seconds"] for log in logs],
            "seconds_per_step": [log["seconds"] for log in logs],
            "restart": reload,
            "checkpoint": latest_checkpoint(out)["checkpoint"]}


def stage_gradient(ctx: dict) -> dict:
    """The GRPO forward and backward pass with made-up advantages, if stage 4 never kept a group."""
    from accessdesk.train.grpo import accumulate_policy_gradient, zero_gradients
    from accessdesk.train.lora import gradient_report, trainable_parameters

    if sum(log["n_trained_episodes"] for log in ctx["grpo_logs"]) > 0:
        return {"needed": False}
    model = ctx["model"]
    records = [record for record in ctx["train_records"] if record["harness_error"] is None][:8]
    if not records:
        raise RuntimeError("stage 1 left no valid episode to run the gradient on")
    advantages = [0.5 if index % 2 == 0 else -0.5 for index in range(len(records))]
    try:
        model.train()
        zero_gradients(model)
        numbers = accumulate_policy_gradient(model, records, advantages, list(range(len(records))),
                                             norm_const=GRPOConfig(n_scenarios=1, group_size=8).norm_const,
                                             pad_id=ctx["sampler"].pad_id, micro_batch=4)
        report = gradient_report(model)
    finally:
        for param in trainable_parameters(model):
            param.grad = None
        model.eval()
    if not all(norm > 0 for norm in report["by_block"].values()):
        raise RuntimeError(f"a block got no gradient: {report['by_block']}")
    return {"needed": True, "episodes": numbers["n_episodes"], "loss": numbers["loss"],
            "logprob_gap_mean": numbers["logprob_gap"]["mean"], "lora_B_by_block": report["by_block"]}


def dev_rollout(ctx: dict, arm: str, folder: str) -> dict:
    """One attempt on every wording of the first `--dev-families` dev families. The sampling seeds depend on
    the run seed, the pass, the chunk and the turn only, so the two dev stages use the same seeds."""
    args = ctx["args"]
    dev = ctx["families"]["dev"][:args.dev_families]
    scenarios = [(family, wording["style"]) for family in dev for wording in family["wordings"]]
    started = time.perf_counter()
    records = roll(ctx, scenarios, 1, arm, ctx["out_dir"] / folder)
    return {"families": len(dev), **summary_of(records), "seconds": round(time.perf_counter() - started, 1)}


def stage_dev_before(ctx: dict) -> dict:
    return dev_rollout(ctx, "dryrun_base_dev", "dev_rollout_before")


def stage_dev_after(ctx: dict) -> dict:
    return dev_rollout(ctx, "dryrun_trained_dev", "dev_rollout_after")


def stage_weights(ctx: dict) -> dict:
    from accessdesk.train.lora import weight_change_report

    change = weight_change_report(ctx["model"], ctx["initial_lora"])
    if not all(value > 0 for value in change["by_block"].values()):
        raise RuntimeError(f"the adapters of a block did not change: {change['by_block']}")
    return {"weight_change_by_block": change["by_block"], "weight_change_total": change["total"]}


STAGES: list[tuple[str, Callable[[dict], dict]]] = [
    ("setup", stage_setup), ("dev_before", stage_dev_before), ("rollout", stage_rollout),
    ("sft_data", stage_sft_data), ("sft", stage_sft), ("grpo", stage_grpo), ("gradient", stage_gradient),
    ("dev_after", stage_dev_after), ("weights", stage_weights),
]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    smoke.add_common_arguments(parser)
    parser.add_argument("--train-scenarios", type=int, default=40)
    parser.add_argument("--samples", type=int, default=4, help="attempts per train scenario in stage 1")
    parser.add_argument("--grpo-steps", type=int, default=3,
                        help="at least 2: the last one runs after a restart")
    parser.add_argument("--dev-families", type=int, default=8)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="delete the dryrun folder of an earlier run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = smoke.load_text_or_exit(args)
    out_dir = smoke.output_dir(args) / "dryrun"
    if args.grpo_steps < 2:
        print("--grpo-steps must be at least 2", file=sys.stderr)
        return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            print(f"{out_dir} is not empty; pass --overwrite or another --name", file=sys.stderr)
            return 2
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig()
    ctx: dict[str, Any] = {"args": args, "text": text, "cfg": cfg, "out_dir": out_dir,
                           "families": smoke.build_families(text, cfg)}

    print(f"dry run of {args.model}; results in {out_dir}")
    stages: list[dict] = []
    failed = None
    for name, function in STAGES:
        started = time.perf_counter()
        try:
            numbers, error = function(ctx), None
        except Exception as exc:  # the first failing stage ends the dry run
            numbers, error = {}, f"{type(exc).__name__}: {exc}"[:2000]
            print("      " + traceback.format_exc().strip().replace("\n", "\n      ")[-3000:])
        seconds = round(time.perf_counter() - started, 1)
        stages.append({"stage": name, "passed": error is None, "seconds": seconds, "error": error,
                       "numbers": numbers})
        print(f"{'DONE' if error is None else 'FAIL'}  {name:12s} {seconds:7.1f} s  "
              f"{error or smoke.brief(numbers)}", flush=True)
        rollout_script.write_json(out_dir / "dryrun.json", {"complete": False, "stages": stages})
        if error is not None:
            failed = name
            break
    not_run = [name for name, _ in STAGES[len(stages):]]
    rollout_script.write_json(out_dir / "dryrun.json", {"complete": failed is None, "failed": failed,
                                                        "not_run": not_run, "stages": stages})
    if "manifest" in ctx:
        ctx["manifest"]["complete"] = failed is None
        rollout_script.write_json(out_dir / "run_manifest.json", ctx["manifest"])
    if failed is None:
        print(f"DRYRUN PASSED: all {len(stages)} stages ran. Its numbers are not results.")
        return 0
    print(f"DRYRUN FAILED at stage {failed}; not run: {', '.join(not_run) or 'none'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
