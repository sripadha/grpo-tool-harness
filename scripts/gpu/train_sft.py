"""Stage 1 training: SFT of a new LoRA adapter on the episodes that sft_data.py chose. Runs on the rented GPU.

    python scripts/gpu/train_sft.py --model Qwen/Qwen3.5-2B --data results/main/sft_data \\
        --out results/main/sft --seed 0

The training itself is the library's (`accessdesk.train.sft.train_sft`): the loss is the mean negative
log-likelihood over the tokens that the model sampled itself, and every other token has weight 0. This script
loads the model as scripts/gpu/dryrun.py does, adds the LoRA adapters, checks that both layer types have them
(`assert_coverage`), builds one training example per chosen episode (`training_example`: the stored token ids,
never tokenised again), calls `train_sft`, and writes the run manifest.

The defaults are the values of the plan (PLAN.md sections 2 and 5.5): learning rate 1e-4, 2 epochs, 8 episodes
per optimiser step in micro-batches of 2, gradient norm clipped at 1.0, gradient checkpointing on; LoRA rank
16, alpha 32, dropout 0, on the ten projections of `accessdesk.train.lora.DEFAULT_TARGETS`.
An SFT run has two seeds, and --seed sets both: the seed of the adapter's random start values (`add_lora`
draws lora_A with it) and the seed of the order of the examples in each epoch (`SFTConfig.seed`). The
manifest names both in `seeds`, each with its value, the option that set it and what it is used for. So a
second SFT seed (another --seed) changes both, and its manifest says so.

The shared config (token cap, turn limit, sampling settings, reward numbers) is not given here. It is read
from run_manifest.json of the data folder: the config of the rollout that sampled the episodes.

Written into --out:
    adapter/            the adapter after the last step (the library writes it; there is no best-step choice)
    sft_log.jsonl       one line per optimiser step, as the library gives it: step, epoch, loss, n_tokens,
                        n_episodes, grad_norm (before clipping), lr, seconds
    lora_coverage.json  one row per linear layer of the model: wrapped or not, shape, adapter weights
    run_manifest.json   the command, model and revision, the settings of SFTConfig and RunConfig with their
                        hashes, the LoRA settings, the hashes of the data files, the seed, the versions of
                        Python and the packages, the GPU, start and end time, seconds, the hash of the frozen
                        code and the state of the freeze (`code_sha256` and `frozen`; None with a note in
                        `frozen_note` if accessdesk/freeze.py does not give them), the hashes of the
                        training code, and after the training: a summary of the log, how far the adapters
                        of each block moved, the hashes of the adapter files. `"complete": true` is in it
                        only after all of that.

The run is refused (exit code 2, before the model loads) if --out already holds a run, if the data folder is
not a complete output of sft_data.py or its record file changed, and if the data was sampled by another model
or revision than --model, or by a model with an adapter (--allow-other-source accepts that). Data that was
made with the placeholder text is written under results/_smoke/ only.
Exit code 1: the training failed, or the adapters of one block did not move at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for sft_data.py and rollout.py, next to this

import rollout as rollout_script  # noqa: E402
import sft_data as sft_data_script  # noqa: E402
from accessdesk import impl  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.harness.hf_backend import resolve_revision  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402
from accessdesk.train.lora import DEFAULT_TARGETS, GATE_TARGETS  # noqa: E402
from accessdesk.train.sft import SFTConfig, training_example  # noqa: E402
from sft_data import (Refused, command_line, dict_sha256, file_entry, frozen_code,  # noqa: E402
                      training_code, utc_now, warn_if_not_as_frozen)

SCRIPT = "scripts/gpu/train_sft.py"
MANIFEST_NAME = "run_manifest.json"
LOG_NAME = "sft_log.jsonl"                    # written by the library
ADAPTER_DIR = "adapter"                       # written by the library
COVERAGE_NAME = "lora_coverage.json"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
LORA_RANK, LORA_ALPHA = 16, 32                # PLAN section 2; the defaults of `add_lora` are the same


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    defaults = SFTConfig()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", required=True, help="model id on the Hub, or a local folder")
    parser.add_argument("--revision", default=None, help="model revision (default: the pinned one, if known)")
    parser.add_argument("--data", type=Path, required=True,
                        help="the output folder of scripts/gpu/sft_data.py")
    parser.add_argument("--out", type=Path, required=True, help="output folder; must not hold a run yet")
    parser.add_argument("--seed", type=int, required=True,
                        help="seed of the adapter's random start and of the order of the examples")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--batch-episodes", type=int, default=defaults.batch_episodes,
                        help="episodes per optimiser step")
    parser.add_argument("--micro-batch", type=int, default=defaults.micro_batch,
                        help="episodes per forward pass; the gradients of a step add up over these")
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="switch gradient checkpointing off (it is on by default; it saves memory)")
    parser.add_argument("--rank", type=int, default=LORA_RANK, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=LORA_ALPHA, help="LoRA alpha; the scale is alpha / rank")
    parser.add_argument("--gate-projections", action="store_true",
                        help="also adapt in_proj_a and in_proj_b of the linear-attention layers")
    parser.add_argument("--allow-other-source", action="store_true",
                        help="accept data that another model or revision sampled, or a model with an adapter")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def sft_config(args: argparse.Namespace) -> SFTConfig:
    return SFTConfig(lr=args.lr, epochs=args.epochs, batch_episodes=args.batch_episodes,
                     micro_batch=args.micro_batch, max_grad_norm=args.max_grad_norm,
                     gradient_checkpointing=not args.no_gradient_checkpointing, seed=args.seed)


def lora_targets(args: argparse.Namespace) -> tuple[str, ...]:
    return DEFAULT_TARGETS + (GATE_TARGETS if args.gate_projections else ())


def seeds_of_the_run(args: argparse.Namespace, sft_cfg: SFTConfig) -> dict:
    """The two seeds of an SFT run, for the manifest: each with its value, the option that set it and what
    it is used for. Today one option sets both; the values are read from where each one is used, so the
    manifest stays right if that changes."""
    return {
        "lora_start_values": {
            "value": args.seed, "set_by": "--seed",
            "used_for": "the random start values of lora_A (accessdesk.train.lora.add_lora, argument seed)"},
        "example_order": {
            "value": sft_cfg.seed, "set_by": "--seed",
            "used_for": "the order of the examples in each epoch (SFTConfig.seed, "
                        "accessdesk.train.sft.epoch_order)"},
    }


def check_output_folder(out_dir: Path) -> None:
    names = (MANIFEST_NAME, LOG_NAME, ADAPTER_DIR, COVERAGE_NAME)
    held = [name for name in names if (out_dir / name).exists()]
    if held:
        raise Refused(f"{out_dir} already holds {', '.join(held)}; choose a new output folder")


def load_data(data_dir: Path) -> dict:
    """The output folder of sft_data.py: its manifest, its records and the config of the rollout.

    Refused if the manifest is missing or not complete, if sft_records.jsonl does not have the SHA-256 that
    the manifest names, or if the manifest's config does not give its own hash again under this code.
    Returns `manifest`, `records`, `cfg` and `files` (the manifest file and the record file, with hashes).
    """
    manifest_file = data_dir / sft_data_script.MANIFEST_NAME
    records_file = data_dir / sft_data_script.RECORDS_NAME
    if not manifest_file.is_file() or not records_file.is_file():
        raise Refused(f"{data_dir} is not an output folder of scripts/gpu/sft_data.py: "
                      f"{sft_data_script.MANIFEST_NAME} or {sft_data_script.RECORDS_NAME} is missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("script") != sft_data_script.SCRIPT:
        raise Refused(f"{manifest_file} is not the manifest of a complete run of {sft_data_script.SCRIPT}")
    records_entry = file_entry(records_file)
    named = manifest["outputs"][sft_data_script.RECORDS_NAME]["sha256"]
    if records_entry["sha256"] != named:
        raise Refused(f"{records_file} is not the file that sft_data.py wrote (its manifest names another "
                      "SHA-256)")
    try:
        cfg = RunConfig(**manifest["config"])
    except TypeError as error:
        raise Refused(f"the config in {manifest_file} is not a config of this code: {error}") from error
    if cfg.sha256() != manifest["config_sha256"]:
        raise Refused(f"the config in {manifest_file} does not give its own hash again: the config code "
                      "changed since the data was made")
    return {"manifest": manifest, "records": read_records(records_file), "cfg": cfg,
            "files": {"manifest": file_entry(manifest_file), "records": records_entry}}


def check_source(args: argparse.Namespace, source: dict | None) -> str | None:
    """SFT trains a model on its OWN passes. `source` is what the data manifest says about the rollout that
    sampled the episodes (None if that is not known). Refused if it was another model, another revision, or
    a model with an adapter, unless --allow-other-source is given. Returns a note for the manifest, or None.
    """
    if source is None:
        return "the data folder does not say which model sampled the episodes: not checked"
    wanted = (args.model, resolve_revision(args.model, args.revision), None)
    found = (source.get("model"), source.get("revision"), source.get("adapter"))
    if found == wanted:
        return None
    message = (f"the episodes were sampled by (model, revision, adapter) {found}, but this run trains "
               f"{wanted[:2]} from its start weights")
    if not args.allow_other_source:
        raise Refused(message + "; pass --allow-other-source if that is meant")
    return message + "; accepted with --allow-other-source"


def make_examples(records: list[dict]) -> list[dict]:
    try:
        return [training_example(record) for record in records]
    except (ValueError, KeyError) as error:
        raise Refused(f"a record of the data folder cannot become a training example: {error}") from error


def setup_model(args: argparse.Namespace, cfg: RunConfig, targets: tuple[str, ...]) -> dict:
    """The GPU part before the training, in the order of scripts/gpu/dryrun.py: load the model, add the LoRA
    adapters, check their coverage, build the sampler (for the pad token and for the manifest).

    Returns `model`, `tokenizer`, `pad_id`, `initial_lora` (a copy of the adapter weights at the start),
    `coverage_table`, and the manifest parts `lora`, `environment`, `kernels` and `sampler`.
    """
    from accessdesk.harness.hf_backend import (Sampler, causal_lm, environment_report, kernel_report,
                                               load_model)
    from accessdesk.train.lora import (add_lora, assert_coverage, coverage_summary, coverage_table,
                                       lora_state, trainable_parameters)

    model, tokenizer = load_model(args.model, args.revision, device=args.device)
    model = add_lora(model, rank=args.rank, alpha=args.alpha, targets=targets, seed=args.seed)
    table = coverage_table(model)
    assert_coverage(table, causal_lm(model).config)
    sampler = Sampler(model, tokenizer, cfg)
    lora = {"rank": args.rank, "alpha": args.alpha, "dropout": 0.0, "targets": list(targets),
            "seed": args.seed, "wrapped_modules": sum(1 for row in table if row["wrapped"]),
            "trainable_parameters": sum(param.numel() for param in trainable_parameters(model)),
            "coverage": coverage_summary(table)}
    return {"model": model, "tokenizer": tokenizer, "pad_id": sampler.pad_id,
            "initial_lora": lora_state(model), "coverage_table": table, "lora": lora,
            "environment": environment_report(), "kernels": kernel_report(model),
            "sampler": sampler.settings()}


def run_training(setup: dict, examples: list[dict], sft_cfg: SFTConfig,
                 out_dir: Path) -> tuple[list[dict], dict]:
    """`train_sft` of the library, then how far the adapters moved. Returns (the log lines, that report)."""
    from accessdesk.train.lora import weight_change_report
    from accessdesk.train.sft import train_sft

    logs = train_sft(setup["model"], examples, sft_cfg, pad_id=setup["pad_id"], out_dir=out_dir)
    return logs, weight_change_report(setup["model"], setup["initial_lora"])


def summarise(logs: list[dict], sft_cfg: SFTConfig) -> dict:
    """A few numbers of the training log. Every one is computed from the log lines of the library.

        steps                   optimiser steps
        first_loss, last_loss   the loss of the first and of the last step (two different batches)
        mean_loss_by_epoch      per epoch: the mean loss over its steps, weighted by loss tokens. Every epoch
                                sees the same examples once, so these numbers can be compared with each other
        max_grad_norm_seen      the largest gradient norm before clipping
        clipped_steps           steps whose gradient norm was above the clip value
        seconds_in_steps        the sum of the steps' seconds (without loading and saving)
    """
    by_epoch = []
    for epoch in range(sft_cfg.epochs):
        steps = [log for log in logs if log["epoch"] == epoch]
        tokens = sum(log["n_tokens"] for log in steps)
        by_epoch.append(sum(log["loss"] * log["n_tokens"] for log in steps) / tokens if tokens else None)
    return {"steps": len(logs), "first_loss": logs[0]["loss"], "last_loss": logs[-1]["loss"],
            "mean_loss_by_epoch": by_epoch,
            "max_grad_norm_seen": max(log["grad_norm"] for log in logs),
            "clipped_steps": sum(1 for log in logs if log["grad_norm"] > sft_cfg.max_grad_norm),
            "seconds_in_steps": round(sum(log["seconds"] for log in logs), 1)}


def adapter_entries(adapter_dir: Path) -> dict:
    """The hashes of the adapter's two files, for a manifest. A file that is missing has the value None."""
    return {name: file_entry(adapter_dir / name) if (adapter_dir / name).is_file() else None
            for name in ADAPTER_FILES}


def peak_gpu_memory_gb() -> dict | None:
    """The peak GPU memory of this process, as scripts/gpu/rollout.py reports it; None if it cannot be read
    (after a failed run the manifest must still be written)."""
    try:
        return rollout_script.peak_gpu_memory_gb()
    except Exception:
        return None


def run(args: argparse.Namespace, command: list[str]) -> int:
    started_utc, started = utc_now(), time.perf_counter()
    check_output_folder(args.out)
    data = load_data(args.data)
    data_manifest, cfg = data["manifest"], data["cfg"]
    sft_data_script.check_placeholder_output(data_manifest.get("text_is_placeholder"), args.out)
    source_note = check_source(args, data_manifest.get("source_run"))
    examples = make_examples(data["records"])
    if not examples:
        raise Refused(f"{args.data} holds no record: there is nothing to train on")
    sft_cfg = sft_config(args)
    targets = lora_targets(args)
    frozen = frozen_code()
    warn_if_not_as_frozen(frozen)

    setup = setup_model(args, cfg, targets)          # from here on the GPU is used
    args.out.mkdir(parents=True, exist_ok=True)
    rollout_script.write_json(args.out / COVERAGE_NAME, {"rows": setup["coverage_table"]})
    manifest: dict[str, Any] = {
        "complete": False, "status": "running",
        "script": SCRIPT, "command": command,
        "started_utc": started_utc, "ended_utc": None, "seconds": None,
        "model": args.model, "revision": resolve_revision(args.model, args.revision), "seed": args.seed,
        "seeds": seeds_of_the_run(args, sft_cfg),
        "sft_config": sft_cfg.to_dict(), "sft_config_sha256": dict_sha256(sft_cfg.to_dict()),
        "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "config_source": "run_manifest.json of the data folder (the config of the rollout)",
        "lora": setup["lora"],
        "data": {"folder": str(args.data), **data["files"], "examples": len(examples),
                 "loss_tokens": sum(sum(example["weights"]) for example in examples),
                 "text_sha256": data_manifest.get("text_sha256"),
                 "text_is_placeholder": data_manifest.get("text_is_placeholder"),
                 "data_sha256": data_manifest.get("data_sha256"),
                 "allow_penalised": data_manifest.get("allow_penalised"),
                 "source_run": data_manifest.get("source_run"), "source_note": source_note},
        "impl": impl.status(),
        **frozen, "training_code": training_code(),
        "environment": setup["environment"], "kernels": setup["kernels"], "sampler": setup["sampler"],
    }
    rollout_script.write_json(args.out / MANIFEST_NAME, manifest)

    print(f"SFT of {args.model} on {len(examples)} episodes ({manifest['data']['loss_tokens']} tokens in the "
          f"loss), {sft_cfg.epochs} epochs, learning rate {sft_cfg.lr}; results in {args.out}", flush=True)
    error, logs, change = None, [], None
    try:
        logs, change = run_training(setup, examples, sft_cfg, args.out)
        still = sorted(block for block, value in change["by_block"].items() if not value > 0)
        if still:
            error = f"the adapters of {still} did not move at all in the training"
    except Exception as exc:      # the manifest must say that the run failed
        error = f"{type(exc).__name__}: {exc}"[:2000]
        print(traceback.format_exc(), file=sys.stderr)
    manifest.update({
        "complete": error is None, "status": "complete" if error is None else "failed", "error": error,
        "ended_utc": utc_now(), "seconds": round(time.perf_counter() - started, 1),
        "summary": summarise(logs, sft_cfg) if logs else None,
        "weight_change": change,
        "outputs": {"log": file_entry(args.out / LOG_NAME) if (args.out / LOG_NAME).is_file() else None,
                    "coverage": file_entry(args.out / COVERAGE_NAME),
                    "adapter": adapter_entries(args.out / ADAPTER_DIR)},
        "peak_gpu_memory_gb": peak_gpu_memory_gb(),
    })
    rollout_script.write_json(args.out / MANIFEST_NAME, manifest)
    if error is not None:
        print(f"SFT FAILED: {error}", file=sys.stderr)
        return 1
    summary = manifest["summary"]
    print(f"{summary['steps']} steps in {manifest['seconds']} s; mean loss by epoch "
          f"{summary['mean_loss_by_epoch']}; weight change by block {change['by_block']}; adapter in "
          f"{args.out / ADAPTER_DIR}")
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
