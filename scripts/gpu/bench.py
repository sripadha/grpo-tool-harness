"""Speed and memory on the rented machine. Its numbers decide the batch size and the budget (PLAN section 7).

    python scripts/gpu/bench.py --model Qwen/Qwen3.5-4B --text text/placeholder_text.yaml --allow-placeholder

Four parts:
  (a) sampling     episodes per hour, sampled tokens per second and peak memory of the rollout, for the chunk
                   sizes 16, 32, 64, 96, 128 on the dev scenarios, with the (still zero) LoRA adapters
                   attached, as in training. Each setting runs until it has --sampling-episodes episodes
                   (whole chunks). With --base-sampling one extra setting at chunk size 32 runs first, on the
                   model without adapters, to show what the adapters cost.
  (b) training     the update part of one GRPO step: seconds and peak memory of the forward pass with
                   gradients, the backward pass and the optimiser step (clipping and AdamW) over 16 x 8 and
                   8 x 8 episodes, with and without gradient checkpointing, for the micro-batch sizes 2, 4, 8,
                   16. The episodes come from ONE real rollout of 16 x 8 train episodes, which is timed as
                   well. For the timing every valid episode gets a made-up advantage (+0.5 or -0.5), so that
                   all of them go through the model: the most expensive case of a step. With real advantages
                   the tied groups are left out and a step is cheaper; the share of groups that the real
                   rewards would keep is reported. The adapter is put back to its start values afterwards.
  (c) SFT          one SFT step on 8 scripted episodes (micro-batch 2, gradient checkpointing on), with the
                   optimiser step; the adapter is then put back to its start values.
  (d) whole step   one whole GRPO step in one piece, as the real loop runs it: a fresh rollout of n x 8 train
                   episodes, then the update over ALL valid episodes (made-up advantages, as in (b)) with the
                   fastest setting of (b) that fitted, and the optimiser step. Nothing is freed between the
                   rollout and the update. Timed for 16 x 8 and for 8 x 8.

The script protects itself. Every setting runs inside a time limit and inside try/except: a setting that runs
out of memory or of time is recorded as "oom" or "timeout" and the script goes on. Between settings the cached
GPU memory is given back and the peak-memory counter is reset. After an out-of-memory error the larger
settings of the same kind are recorded as "skipped". Memory is reported twice: `peak_memory_gb` is what
tensors used (the model's weights included), `peak_reserved_gb` is what torch held from the driver (GiB).

Written into results/_smoke/<name>/: bench.json (rewritten after every setting) and run_manifest.json.
A short table is printed at the end. Its numbers are for planning only and are never reported as results.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for smoke.py, the script next to this one

import smoke  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.scenarios import scenario_ids  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.harness.episode import Episode  # noqa: E402
from accessdesk.harness.tokens import TokenLayout  # noqa: E402
from accessdesk.text import TextBundle  # noqa: E402
from accessdesk.train.grpo import GRPOConfig, make_groups, plan_update, step_scenarios  # noqa: E402

BENCH_SEED = 0
GROUP_SIZE = 8


class TimeLimit(BaseException):
    """Raised by the alarm of `time_limit`. It is not an Exception on purpose: `rollout` catches Exception
    around every sampling call (to mark harness errors), and the limit must get through that."""


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Raise TimeLimit in the code inside the `with` block when it has run for `seconds`.

    The alarm is delivered between two Python instructions, so a single call into the GPU library is never
    cut in half; the limit is therefore late by at most one such call.
    """
    def on_alarm(signum: int, frame: Any) -> None:
        raise TimeLimit(f"stopped after {seconds:.0f} s")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def is_out_of_memory(error: BaseException | str) -> bool:
    """True for torch's out-of-memory error, by its class or by its message."""
    import torch

    classes = tuple(cls for cls in (getattr(torch, "OutOfMemoryError", None),
                                    getattr(torch.cuda, "OutOfMemoryError", None)) if cls is not None)
    by_class = not isinstance(error, str) and isinstance(error, classes)
    return by_class or "out of memory" in str(error).lower()


def peak_reserved_gb() -> float | None:
    """The largest amount of GPU memory that torch held from the driver since the last reset, in GiB."""
    import torch

    return round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None


def synchronize() -> None:
    """Wait until the GPU has finished what was sent to it, so that a clock read after it is right."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def restore_lora(model: Any, initial: dict) -> None:
    """Put every LoRA weight back to the values of `initial` (from `lora_state`)."""
    import torch

    from accessdesk.harness.hf_backend import causal_lm

    with torch.no_grad():
        for name, param in causal_lm(model).named_parameters():
            if name in initial:
                param.copy_(initial[name].to(param.device))


def guarded(setting: dict, limit_seconds: float, function: Callable[[], dict]) -> dict:
    """Run one setting. Returns `setting` with `status` ("ok", "oom", "timeout" or "error: ..."), `seconds`,
    `peak_memory_gb`, and the numbers that `function` returned. Never raises (except for Ctrl-C)."""
    smoke.free_gpu_memory()
    smoke.reset_peak_memory()
    result = dict(setting)
    started = time.perf_counter()
    try:
        with time_limit(limit_seconds):
            result.update(function())
        result.setdefault("status", "ok")
    except TimeLimit:
        result["status"] = "timeout"
    except Exception as error:  # the message is kept; the error object, which holds GPU tensors, is not
        message = f"error: {type(error).__name__}: {error}"[:300]
        result["status"] = "oom" if is_out_of_memory(error) else message
    result["seconds"] = round(time.perf_counter() - started, 2)
    result["peak_memory_gb"] = smoke.peak_memory_gb()
    result["peak_reserved_gb"] = peak_reserved_gb()
    smoke.free_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# (a) Sampling
# ---------------------------------------------------------------------------
def sampling_setting(model: Any, tokenizer: Any, text: TextBundle, scenarios: list[tuple[dict, int]],
                     chunk_size: int, min_episodes: int, label: str) -> dict:
    """Roll out chunks of `chunk_size` episodes, one chunk per call, until at least `min_episodes` episodes
    are done (at least one chunk).

    The scenarios are taken in order and start again from the first when the list is used up. A chunk in
    which a sampling call ran out of memory ends the setting with status "oom".
    """
    from accessdesk.harness.hf_backend import Sampler, rollout

    cfg = dataclasses.replace(RunConfig(), chunk_size=chunk_size)
    sampler = Sampler(model, tokenizer, cfg)
    layouts = {"five": TokenLayout(tokenizer, text.tool_specs("five"))}
    records: list[dict] = []
    started = time.perf_counter()
    status, call = "ok", 0
    while call == 0 or len(records) < min_episodes:
        chosen = [scenarios[(call * chunk_size + i) % len(scenarios)] for i in range(chunk_size)]
        episodes = [Episode(family, style, text, cfg) for family, style in chosen]
        new = rollout(sampler, layouts, episodes, run_seed=BENCH_SEED, pass_index=call,
                      meta={"run": label, "arm": "bench"})
        records.extend({key: value for key, value in record.items() if key != "tokens"} for record in new)
        call += 1
        if any(is_out_of_memory(record["harness_error"] or "") for record in new):
            status = "oom"
            break
    seconds = time.perf_counter() - started
    valid = metrics.valid_records(records)
    sampled_tokens = sum(turn["n_tokens"] for record in valid for turn in record["turns"])
    numbers = {"status": status, "rollout_calls": call, "episodes": len(records),
               "harness_errors": len(records) - len(valid), "sampling_seconds": round(seconds, 1),
               "episodes_per_hour": round(len(valid) / seconds * 3600), "sampled_tokens": sampled_tokens,
               "sampled_tokens_per_second": round(sampled_tokens / seconds, 1)}
    if valid:
        numbers.update({"turns_per_episode": metrics.turn_stats(records)["mean_turns"],
                        "tokens_per_turn": sampled_tokens / sum(record["n_turns"] for record in valid),
                        "pass_rate": metrics.pass_rate(records),
                        "invalid_call_rate": metrics.invalid_call_rate(records)})
    return numbers


def bench_sampling(model: Any, tokenizer: Any, text: TextBundle, families: dict, args: argparse.Namespace,
                   label: str, chunk_sizes: list[int], record: Callable[[dict], None]) -> None:
    """The sampling settings for the given chunk sizes, smallest first; larger ones are skipped after an
    out-of-memory error."""
    scenarios = dev_scenarios(families)
    out_of_memory = False
    for chunk_size in sorted(chunk_sizes):
        setting = {"part": "sampling", "label": label, "chunk_size": chunk_size}
        if out_of_memory:
            record({**setting, "status": "skipped: a smaller chunk size ran out of memory"})
            continue
        result = guarded(setting, args.sampling_limit, lambda: sampling_setting(
            model, tokenizer, text, scenarios, chunk_size, args.sampling_episodes, label))
        out_of_memory = result["status"] == "oom"
        record(result)


def dev_scenarios(families: dict) -> list[tuple[dict, int]]:
    """(family, style) for every wording of every dev family, in order."""
    return [(family, wording["style"]) for family in families["dev"] for wording in family["wordings"]]


def warm_up(model: Any, tokenizer: Any, text: TextBundle, families: dict) -> dict:
    """A rollout of four episodes that is not part of any number: the first use of the model on a machine
    compiles kernels, and that time must not land in the first timed setting."""
    numbers = sampling_setting(model, tokenizer, text, dev_scenarios(families), 4, 1, "warm-up")
    return {key: numbers[key] for key in ("episodes", "harness_errors", "status")}


# ---------------------------------------------------------------------------
# (b) The training part of a GRPO step
# ---------------------------------------------------------------------------
def step_rollout(model: Any, tokenizer: Any, text: TextBundle, families: dict, n_scenarios: int,
                 chunk_size: int, step_index: int = 0) -> dict:
    """One rollout as a GRPO step makes it: the `n_scenarios` train scenarios of step `step_index` x 8
    attempts, in chunks of `chunk_size`. The model's mode is left as it is (the sampler samples in eval mode
    and restores the mode)."""
    from accessdesk.harness.hf_backend import Sampler, rollout

    cfg = dataclasses.replace(RunConfig(), chunk_size=chunk_size)
    sampler = Sampler(model, tokenizer, cfg)
    layouts = {"five": TokenLayout(tokenizer, text.tool_specs("five"))}
    train = families["train"]
    ids = sorted(scenario_id for family in train for scenario_id in scenario_ids(family))
    step_ids = step_scenarios(ids, n_scenarios, BENCH_SEED, step_index)
    episodes = make_groups(step_ids, {family["family_id"]: family for family in train}, text, cfg, GROUP_SIZE)
    started = time.perf_counter()
    records = rollout(sampler, layouts, episodes, run_seed=BENCH_SEED, pass_index=step_index,
                      meta={"run": "bench", "arm": "bench_step"})
    synchronize()
    seconds = time.perf_counter() - started
    plan = plan_update(records, GRPOConfig(n_scenarios=n_scenarios, group_size=GROUP_SIZE))
    valid = metrics.valid_records(records)
    n_valid = max(1, len(valid))
    sampled = sum(len(record["tokens"]["logprobs"]) for record in valid)
    return {"records": records, "pad_id": sampler.pad_id, "rollout_seconds": round(seconds, 1),
            "chunk_size": chunk_size, "episodes": len(records), "harness_errors": len(records) - len(valid),
            "episodes_per_hour": round(len(valid) / seconds * 3600),
            "sampled_tokens_per_second": round(sampled / seconds, 1),
            "longest_episode_tokens": max((len(record["tokens"]["ids"]) for record in valid), default=0),
            "groups_kept_by_the_real_rewards": float(plan["keep"].mean()),
            "pass_rate": metrics.pass_rate(records) if valid else None,
            "tokens_per_episode": sum(len(record["tokens"]["ids"]) for record in valid) / n_valid,
            "sampled_tokens_per_episode":
                sum(len(record["tokens"]["logprobs"]) for record in valid) / n_valid}


def update_part(model: Any, records: list[dict], pad_id: int, micro_batch: int) -> dict:
    """The update part of a GRPO step over every valid episode of `records`, with made-up advantages:
    forward with gradients, backward, clipping and one AdamW step. The model is in train mode already.
    Returns the timings and the numbers of the pass. The adapter is changed (the caller puts it back)."""
    from accessdesk.train.grpo import accumulate_policy_gradient, zero_gradients
    from accessdesk.train.sft import clip_and_step, make_optimizer

    gcfg = GRPOConfig(n_scenarios=max(1, len(records) // GROUP_SIZE), group_size=GROUP_SIZE)
    selected = [index for index, record in enumerate(records) if record["harness_error"] is None]
    advantages = [0.5 if index % 2 == 0 else -0.5 for index in range(len(records))]
    optimizer = make_optimizer(model, gcfg.lr)
    try:
        synchronize()
        started = time.perf_counter()
        zero_gradients(model)
        numbers = accumulate_policy_gradient(model, records, advantages, selected, norm_const=gcfg.norm_const,
                                             pad_id=pad_id, micro_batch=micro_batch)
        synchronize()
        forward_backward = time.perf_counter() - started
        grad_norm = clip_and_step(model, optimizer, gcfg.max_grad_norm)
        synchronize()
        seconds = time.perf_counter() - started
    finally:
        optimizer.zero_grad(set_to_none=True)
    gap = numbers["logprob_gap"] or {}
    return {"training_seconds": round(seconds, 2), "forward_backward_seconds": round(forward_backward, 2),
            "optimizer_step_seconds": round(seconds - forward_backward, 3),
            "trained_episodes": numbers["n_episodes"], "loss_tokens": numbers["n_tokens"],
            "seconds_per_episode": round(seconds / max(1, numbers["n_episodes"]), 3),
            "grad_norm": grad_norm, "logprob_gap_mean": gap.get("mean"), "logprob_gap_max": gap.get("max")}


def training_setting(model: Any, records: list[dict], pad_id: int, micro_batch: int,
                     checkpointing: bool) -> dict:
    """`update_part` in train mode with the given gradient checkpointing; then the adapter is put back to
    the values it had before, and the model to eval mode."""
    from accessdesk.train.forward import set_gradient_checkpointing
    from accessdesk.train.lora import lora_state, trainable_parameters

    initial = lora_state(model)
    try:
        set_gradient_checkpointing(model, checkpointing)
        model.train()
        return update_part(model, records, pad_id, micro_batch)
    finally:
        for param in trainable_parameters(model):
            param.grad = None
        set_gradient_checkpointing(model, False)
        model.eval()
        restore_lora(model, initial)


def whole_step(model: Any, tokenizer: Any, text: TextBundle, families: dict, n_scenarios: int,
               chunk_size: int, micro_batch: int, checkpointing: bool, step_index: int) -> dict:
    """One whole GRPO step in one piece, as `train_grpo` runs it: the model in train mode with gradient
    checkpointing as given; a fresh rollout of `n_scenarios` x 8 train episodes (the sampler switches to eval
    mode and back); then `update_part` over ALL valid episodes. Nothing is freed in between."""
    from accessdesk.train.forward import set_gradient_checkpointing
    from accessdesk.train.lora import lora_state, trainable_parameters

    initial = lora_state(model)
    try:
        set_gradient_checkpointing(model, checkpointing)
        model.train()
        synchronize()
        started = time.perf_counter()
        step = step_rollout(model, tokenizer, text, families, n_scenarios, chunk_size, step_index)
        update = update_part(model, step["records"], step["pad_id"], micro_batch)
        seconds = time.perf_counter() - started
    finally:
        for param in trainable_parameters(model):
            param.grad = None
        set_gradient_checkpointing(model, False)
        model.eval()
        restore_lora(model, initial)
    numbers = {key: value for key, value in step.items() if key not in ("records", "pad_id")}
    return {**numbers, **update, "whole_step_seconds": round(seconds, 1)}


def bench_training(model: Any, step: dict, args: argparse.Namespace, record: Callable[[dict], None]) -> None:
    """Every training setting: episodes x gradient checkpointing x micro-batch size, cheapest memory first."""
    for n_scenarios in (args.grid_step_sizes or args.step_sizes):
        records = step["records"][:n_scenarios * GROUP_SIZE]
        for checkpointing in [mode == "on" for mode in args.checkpointing]:
            out_of_memory = False
            # Without checkpointing one episode of 1,500 tokens already needs about 12 GiB on the 4B (smoke
            # test, 2026-11-17), so micro-batch 1 is tried there as well.
            sizes = sorted(set(args.micro_batches) | (set() if checkpointing else {1}))
            for micro_batch in sizes:
                setting = {"part": "training", "episodes": len(records), "n_scenarios": n_scenarios,
                           "gradient_checkpointing": checkpointing, "micro_batch": micro_batch}
                if out_of_memory:
                    record({**setting, "status": "skipped: a smaller micro-batch ran out of memory"})
                    continue
                result = guarded(setting, args.train_seconds, lambda: training_setting(
                    model, records, step["pad_id"], micro_batch, checkpointing))
                out_of_memory = result["status"] == "oom"
                record(result)


# ---------------------------------------------------------------------------
# (c) One SFT step
# ---------------------------------------------------------------------------
def sft_setting(model: Any, tokenizer: Any, text: TextBundle, families: dict) -> dict:
    """One SFT step on 8 scripted episodes (two per outcome), with the optimiser step; then the adapter is
    put back to the values it had before."""
    from accessdesk.train.forward import set_gradient_checkpointing
    from accessdesk.train.lora import lora_state
    from accessdesk.train.sft import SFTConfig, clip_and_step, make_optimizer, sft_gradient

    cfg, sft_cfg = RunConfig(), SFTConfig()
    layout = TokenLayout(tokenizer, text.tool_specs("five"))
    by_outcome: dict[str, list[dict]] = {}
    for family in families["train"]:
        by_outcome.setdefault(family["outcome"], []).append(family)
    chosen = [family for outcome in sorted(by_outcome) for family in by_outcome[outcome][:2]]
    batch = [smoke.oracle_example(family, 1, text, cfg, layout) for family in chosen]
    initial = lora_state(model)
    optimizer = make_optimizer(model, sft_cfg.lr)
    try:
        set_gradient_checkpointing(model, sft_cfg.gradient_checkpointing)
        model.train()
        synchronize()
        started = time.perf_counter()
        numbers = sft_gradient(model, batch, pad_id=layout.pad_id, micro_batch=sft_cfg.micro_batch)
        grad_norm = clip_and_step(model, optimizer, sft_cfg.max_grad_norm)
        synchronize()
        seconds = time.perf_counter() - started
    finally:
        optimizer.zero_grad(set_to_none=True)
        set_gradient_checkpointing(model, False)
        model.eval()
        restore_lora(model, initial)
    return {"sft_step_seconds": round(seconds, 2), "episodes": numbers["n_episodes"],
            "loss_tokens": numbers["n_tokens"], "tokens": sum(len(example["ids"]) for example in batch),
            "loss": numbers["loss"], "grad_norm": grad_norm, "micro_batch": sft_cfg.micro_batch,
            "gradient_checkpointing": sft_cfg.gradient_checkpointing}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_row(result: dict) -> None:
    """One line per setting, as soon as it is done."""
    if result["part"] == "sampling":
        what = f"sampling  {result['label']:13s} chunk {result['chunk_size']:3d}"
        numbers = (f"{result.get('episodes', '-')} episodes in {result.get('sampling_seconds', '-')} s  "
                   f"{result.get('episodes_per_hour', '-')} episodes/hour  "
                   f"{result.get('sampled_tokens_per_second', '-')} sampled tokens/s")
    elif result["part"] == "training":
        checkpointing = "on " if result["gradient_checkpointing"] else "off"
        what = (f"training  {result['n_scenarios']:2d} x {GROUP_SIZE}  checkpointing {checkpointing}  "
                f"micro-batch {result['micro_batch']:2d}")
        numbers = (f"{result.get('training_seconds', '-')} s for {result.get('trained_episodes', '-')} "
                   f"episodes (optimiser step {result.get('optimizer_step_seconds', '-')} s)")
    elif result["part"] == "whole_step":
        checkpointing = "on" if result["gradient_checkpointing"] else "off"
        what = (f"whole step {result['n_scenarios']:2d} x {GROUP_SIZE}  chunk {result['chunk_size']}  "
                f"micro-batch {result['micro_batch']}  ckpt {checkpointing}")
        numbers = (f"{result.get('whole_step_seconds', '-')} s = rollout {result.get('rollout_seconds', '-')} s "
                   f"+ update {result.get('training_seconds', '-')} s "
                   f"({result.get('trained_episodes', '-')} episodes trained)")
    else:
        what = f"{result['part']}"
        numbers = "  ".join(f"{key}={value}" for key, value in result.items()
                            if key.endswith("seconds") and key != "seconds")
    print(f"{what:58s} {result['status'][:40]:10s} {numbers}  peak {result.get('peak_memory_gb', '-')} GiB "
          f"allocated, {result.get('peak_reserved_gb', '-')} GiB reserved", flush=True)


def best_training(results: list[dict], n_scenarios: int) -> dict | None:
    """The fastest training setting of `n_scenarios` x 8 that ran, or None."""
    done = [r for r in results if r["part"] == "training" and r["status"] == "ok"
            and r["n_scenarios"] == n_scenarios]
    return min(done, key=lambda r: r["training_seconds"]) if done else None


def best_chunk_size(results: list[dict], default: int) -> int:
    """The chunk size with the most episodes per hour among the sampling settings (with LoRA) that ran."""
    done = [r for r in results if r["part"] == "sampling" and r["status"] == "ok" and r["label"] == "with LoRA"]
    return max(done, key=lambda r: r["episodes_per_hour"])["chunk_size"] if done else default


def print_summary(results: list[dict]) -> None:
    """What the speed rule of PLAN section 7 reads: the time of one whole step at 16 x 8 and at 8 x 8."""
    print("\nOne GRPO step, parts added up = the rollout of 16 x 8 (scaled for 8 x 8) + the fastest update "
          "setting that ran (every episode trained):")
    rollouts = {r["n_scenarios"]: r for r in results if r["part"] == "step_rollout" and r["status"] == "ok"}
    for n_scenarios in sorted(rollouts, reverse=True):
        best = best_training(results, n_scenarios)
        if best is None:
            print(f"  {n_scenarios:2d} x {GROUP_SIZE}: no training setting ran")
            continue
        rollout_seconds = rollouts[n_scenarios]["rollout_seconds"]
        total = rollout_seconds + best["training_seconds"]
        checkpointing = "on" if best["gradient_checkpointing"] else "off"
        print(f"  {n_scenarios:2d} x {GROUP_SIZE}: {total:6.0f} s = rollout {rollout_seconds:.0f} s + "
              f"update {best['training_seconds']:.0f} s (micro-batch {best['micro_batch']}, checkpointing "
              f"{checkpointing}, peak {best['peak_memory_gb']} GiB)")
    print("One GRPO step, measured in one piece (fresh rollout, then the update over every episode):")
    for r in results:
        if r["part"] == "whole_step":
            print(f"  {r['n_scenarios']:2d} x {GROUP_SIZE}: {r.get('whole_step_seconds', '-')} s  ({r['status']}; "
                  f"rollout {r.get('rollout_seconds', '-')} s at chunk {r['chunk_size']}, update "
                  f"{r.get('training_seconds', '-')} s; the real rewards would keep "
                  f"{r.get('groups_kept_by_the_real_rewards', '-')} of the groups)")
    print("PLAN section 7: up to 180 s for 16 x 8 keeps 16 x 8; 180 to 300 s means 8 x 8; more means the 2B "
          "model.")


def int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    smoke.add_common_arguments(parser)
    parser.add_argument("--chunk-sizes", type=int_list, default=[16, 32, 64, 96, 128])
    parser.add_argument("--sampling-episodes", type=int, default=128,
                        help="at least this many episodes per chunk size (whole chunks)")
    parser.add_argument("--sampling-limit", type=float, default=900.0,
                        help="time limit of one sampling setting, in seconds")
    parser.add_argument("--base-sampling", action="store_true",
                        help="also time sampling at chunk size 32 on the model without adapters")
    parser.add_argument("--step-sizes", type=int_list, default=[16, 8],
                        help="scenarios per GRPO step to time")
    parser.add_argument("--grid-step-sizes", type=int_list, default=None,
                        help="step sizes of the training grid of part (b) (default: --step-sizes)")
    parser.add_argument("--micro-batches", type=int_list, default=[2, 4, 8, 16])
    parser.add_argument("--checkpointing", type=lambda value: [part.strip() for part in value.split(",")],
                        default=["on", "off"],
                        help="gradient checkpointing modes of the training grid of part (b): on, off or on,off")
    parser.add_argument("--step-chunk-size", type=int, default=None,
                        help="chunk size of the step rollouts (default: the fastest of part (a), else 32)")
    parser.add_argument("--train-seconds", type=float, default=900.0,
                        help="time limit of one training setting")
    parser.add_argument("--skip", default="",
                        help="parts to leave out: sampling, training, sft, whole (commas)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = smoke.load_text_or_exit(args)
    skip = {part.strip() for part in args.skip.split(",") if part.strip()}
    if not args.checkpointing or set(args.checkpointing) - {"on", "off"}:
        print("--checkpointing takes on, off or on,off", file=sys.stderr)
        return 2
    if skip - {"sampling", "training", "sft", "whole"} or max(args.step_sizes, default=0) < 1:
        print("--skip takes sampling, training, sft, whole; --step-sizes needs at least one number",
              file=sys.stderr)
        return 2
    cfg = RunConfig()
    out_dir = smoke.output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    families = smoke.build_families(text, cfg)

    from accessdesk.harness.hf_backend import load_model
    from accessdesk.train.lora import add_lora

    model, tokenizer = load_model(args.model, args.revision, device=args.device)
    manifest = smoke.run_manifest(args, cfg, text, model=model, extra={"complete": False})
    smoke.write_json(out_dir / "run_manifest.json", manifest)
    results: list[dict] = []

    def record(result: dict) -> None:
        """Keep a finished setting: print its line and rewrite bench.json."""
        results.append(result)
        print_row(result)
        smoke.write_json(out_dir / "bench.json",
                         {"complete": False, "model": args.model, "settings": results})

    print(f"benchmark of {args.model}; results in {out_dir}")
    record(guarded({"part": "warm_up"}, 900.0, lambda: warm_up(model, tokenizer, text, families)))
    if "sampling" not in skip and args.base_sampling:
        bench_sampling(model, tokenizer, text, families, args, "without LoRA", [32], record)
    model = add_lora(model, seed=0)
    if "sampling" not in skip:
        bench_sampling(model, tokenizer, text, families, args, "with LoRA", args.chunk_sizes, record)
    chunk_size = args.step_chunk_size or best_chunk_size(results, cfg.chunk_size)
    if "training" not in skip:
        largest = max(args.step_sizes)
        step: dict = {}

        def run_step_rollout() -> dict:
            step.update(step_rollout(model, tokenizer, text, families, largest, chunk_size))
            return {key: value for key, value in step.items() if key not in ("records", "pad_id")}

        measured = guarded({"part": "step_rollout", "n_scenarios": largest}, 1800.0, run_step_rollout)
        record(measured)
        if measured["status"] == "ok":
            for n_scenarios in args.step_sizes:            # the rollout of a smaller step is not run again
                if n_scenarios != largest:
                    record({**measured, "n_scenarios": n_scenarios, "episodes": n_scenarios * GROUP_SIZE,
                            "rollout_seconds": round(measured["rollout_seconds"] * n_scenarios / largest, 1),
                            "note": f"not measured: the time of {largest} x {GROUP_SIZE}, scaled"})
            bench_training(model, step, args, record)
        step.clear()
    if "whole" not in skip:
        for index, n_scenarios in enumerate(args.step_sizes):
            best = best_training(results, n_scenarios) or best_training(results, max(args.step_sizes))
            micro_batch = best["micro_batch"] if best else GRPOConfig().micro_batch
            checkpointing = best["gradient_checkpointing"] if best else GRPOConfig().gradient_checkpointing
            setting = {"part": "whole_step", "n_scenarios": n_scenarios, "chunk_size": chunk_size,
                       "micro_batch": micro_batch, "gradient_checkpointing": checkpointing,
                       "step_index": index + 1}
            record(guarded(setting, 1800.0, lambda: whole_step(
                model, tokenizer, text, families, n_scenarios, chunk_size, micro_batch, checkpointing,
                index + 1)))
    if "sft" not in skip:
        record(guarded({"part": "sft_step"}, args.train_seconds,
                       lambda: sft_setting(model, tokenizer, text, families)))
    smoke.write_json(out_dir / "bench.json", {"complete": True, "model": args.model, "settings": results})
    manifest["complete"] = True
    smoke.write_json(out_dir / "run_manifest.json", manifest)
    print_summary(results)
    failed = [r for r in results if r["status"] != "ok" and not r["status"].startswith("skipped")]
    print(f"{len(results)} settings; {len(failed)} did not finish (oom, timeout or error); "
          f"wrote {out_dir / 'bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
