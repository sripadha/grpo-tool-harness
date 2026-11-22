"""The first script to run on a rented machine: does every piece work on this GPU, with this model?

    python scripts/gpu/smoke.py --model Qwen/Qwen3.5-2B --text text/placeholder_text.yaml --allow-placeholder

It runs a fixed list of checks, the cheapest and most basic first. Every check prints one line,

    PASS  name   seconds   the numbers it measured
    FAIL  name   seconds   the numbers it measured, or the error

A check that raises is a FAIL; the script goes on with the next check. Checks that need something that an
earlier check could not build (the model, for example) fail with a message that says so. At the end the script
prints which checks failed and exits with code 1 if any did (0 if all passed, 2 for a wrong command line).

Written into results/_smoke/<name>/ (always there, whatever the text file):
    smoke.json           every check with its numbers, rewritten after each check
    run_manifest.json    the command, config, text hash, model, package versions, kernels, sampler settings
    smoke_episodes.jsonl, smoke_tokens.jsonl    the episodes of the rollout check
    adapter_round_trip/  the adapter that the last check saves and loads again

The bounds of the numerical checks are the ones of tests/gpu: see the constants below, with the numbers
measured on the first run (2026-11-17) beside them. A bound that is missed by a little is a reason to look at
the numbers, not a proof of a bug; a padding or indexing mistake misses them by a lot.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for rollout.py, the script next to this one

import rollout as rollout_script  # noqa: E402
from accessdesk import impl  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.generator import load_name_pools, make_all_splits  # noqa: E402
from accessdesk.harness.episode import Episode  # noqa: E402
from accessdesk.harness.policies import oracle, run_policy  # noqa: E402
from accessdesk.harness.records import replay  # noqa: E402
from accessdesk.harness.tokens import TokenLayout, buffer_from_record  # noqa: E402
from accessdesk.text import TextBundle, load_text, model_text_path  # noqa: E402
from accessdesk.train.grpo import abs_stats  # noqa: E402
from accessdesk.train.lora import target_pattern  # noqa: E402
from accessdesk.types import EXTRA_TOOL_KEY  # noqa: E402

SMOKE_DIR = rollout_script.SMOKE_DIR
write_json = rollout_script.write_json

# Bounds on |difference of two log-probabilities of the same token|: (mean, maximum).
SAME_COMPUTATION = (1e-3, 0.25)     # the same forward pass twice, or with adapters that are still zero
# A sequence alone against the same sequence as the second row of a left-padded batch of two: bounds on the
# (mean, 99th percentile) over 399 PROMPT tokens. In bfloat16 a row's numbers depend on the shape of its batch,
# pads or no pads, and prompt tokens (unlikely ones) show it most. Measured on 2026-11-17 (RTX 4090), mean /
# 99th percentile: 2B 0.047 / 0.50 to 0.88, 4B 0.103 to 0.123 / 1.1 to 1.9, and the same without any pad
# token (2B 0.046, 4B 0.108); with the pads left unmasked on purpose 0.82 / 13 (2B) and 1.81 / 17.5 (4B)
# (tests/gpu/test_forward.py). The bounds are about three times the largest measured value. The first version
# asked for mean < 0.01 and max < 0.5 on 39 tokens: a guess; the first run gave mean 0.051, max 0.375 (2B).
PADDED_AGAINST_ALONE = (0.3, 4.0)
SAMPLER_AGAINST_TRAINER = (0.02, 1.0)   # recorded while sampling against the training forward (PLAN 5.6)
ADAPTER_EFFECT_FLOOR = 2e-5         # a tenth of the smallest measured effect of one SFT step (4B: 0.00023)
END_OF_TURN = "<|im_end|>"
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
SUFFIX_HEAD = "\n<|im_start|>user\n<tool_response>\n"
SUFFIX_TAIL = "\n</tool_response><|im_end|>\n<|im_start|>assistant\n" + EMPTY_THINK_BLOCK


# ---------------------------------------------------------------------------
# Helpers that bench.py and dryrun.py use too
# ---------------------------------------------------------------------------
def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="model id on the Hub, or a local folder")
    parser.add_argument("--revision", default=None, help="model revision (default: the pinned one, if known)")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file (YAML)")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="accept the placeholder text (outputs are under results/_smoke/ in any case)")
    parser.add_argument("--name", default=None, help="name of the output folder under results/_smoke/")
    parser.add_argument("--device", default="cuda")


def output_dir(args: argparse.Namespace) -> Path:
    """results/_smoke/<name>/; the name defaults to the last part of the model id, in lower case."""
    name = args.name or re.sub(r"[^a-z0-9._-]+", "-", args.model.rstrip("/").split("/")[-1].lower())
    return SMOKE_DIR / name


def load_text_or_exit(args: argparse.Namespace) -> TextBundle:
    """The text bundle; exit code 2 with a message if it is the placeholder and that was not allowed."""
    text = load_text(args.text)
    if text.is_placeholder and not args.allow_placeholder:
        print("the text file is the placeholder; pass --allow-placeholder for a smoke run", file=sys.stderr)
        raise SystemExit(2)
    return text


def build_families(text: TextBundle, cfg: RunConfig) -> dict[str, list[dict]]:
    """split -> family records, built in memory from the text file and data/names (nothing is read from
    data/scenarios*, so these scripts work with either text file)."""
    return make_all_splits(text, load_name_pools(ROOT / "data" / "names"), cfg)


def run_manifest(args: argparse.Namespace, cfg: RunConfig, text: TextBundle, *, model: Any = None,
                 sampler: Any = None, extra: dict | None = None) -> dict:
    """What a script writes as run_manifest.json. The parts that need torch are there if `model` is given."""
    from accessdesk.harness.hf_backend import environment_report, kernel_report, resolve_revision

    manifest = {
        "command": sys.argv,
        "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "text_file": Path(text.path).name, "text_sha256": text.sha256,
        "text_is_placeholder": text.is_placeholder,
        "impl": impl.status(),
        "model": args.model, "revision": resolve_revision(args.model, args.revision),
    }
    if model is not None:
        manifest.update({"environment": environment_report(), "kernels": kernel_report(model)})
    if sampler is not None:
        manifest["sampler"] = sampler.settings()
    manifest.update(extra or {})
    return manifest


def peak_memory_gb() -> float | None:
    """The largest amount of GPU memory that tensors used since the last reset, in GB; None without a GPU."""
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None


def reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def free_gpu_memory() -> None:
    """Drop what Python no longer uses and give the cached GPU memory back. Never raises."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # no torch, or a broken CUDA state: nothing more can be done here
        pass


def all_but_first(ids: list[int]) -> list[int]:
    """Weights that put every token but the first in the loss (the first has nothing before it)."""
    return [0] + [1] * (len(ids) - 1)


def oracle_example(family: dict, style: int, text: TextBundle, cfg: RunConfig, layout: TokenLayout) -> dict:
    """A training example made without a model: the scripted correct solution of one scenario.

    Returns `ids`, `weights` (from `impl.get("build_loss_mask")`), and the `record`, `buffer` and `family` it
    was built from. For checks and timing only: nothing that is reported is trained on scripted episodes.
    """
    record = run_policy(Episode(family, style, text, cfg), oracle(text)).record()
    buffer = buffer_from_record(layout, record, family, text)
    weights = impl.get("build_loss_mask")(len(buffer.ids), buffer.segments)
    return {"ids": list(buffer.ids), "weights": list(weights), "record": record, "buffer": buffer,
            "family": family}


def brief(numbers: dict) -> str:
    """The scalar numbers of a check on one line; lists and nested tables are only in smoke.json."""
    parts = []
    for key, value in numbers.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        elif isinstance(value, str):
            parts.append(f"{key}={value!r}")
        elif isinstance(value, (bool, int)) or value is None:
            parts.append(f"{key}={value}")
        elif isinstance(value, dict) and value and all(isinstance(v, (int, float)) for v in value.values()):
            parts.append(key + "={" + ", ".join(f"{k}: {v:.4g}" for k, v in value.items()) + "}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# The checks. Each takes the shared dict `ctx` and returns (passed, numbers).
# ---------------------------------------------------------------------------
def check_environment(ctx: dict) -> tuple[bool, dict]:
    """A GPU is there, and the packages are the pinned versions of requirements-gpu.txt."""
    from accessdesk.harness.hf_backend import environment_report

    report = environment_report()
    ctx["environment"] = report
    pinned = {}
    for line in (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, _, version = line.strip().partition("==")
            pinned[name] = version
    found = report["packages"]
    not_as_pinned = {name: found[name] for name in found
                     if name in pinned and (found[name] or "").split("+")[0] != pinned[name]}
    numbers = {"gpu": report["gpu_name"], "gpu_memory_gb": report["gpu_memory_gb"],
               "cuda": report["cuda_version"], "python": report["python"], "packages": found,
               "not_as_pinned": not_as_pinned, "n_not_as_pinned": len(not_as_pinned)}
    return report["gpu_name"] is not None and not not_as_pinned, numbers


def check_text_and_impl(ctx: dict) -> tuple[bool, dict]:
    """Which text file and which versions of the six hand-written functions this run uses."""
    text = ctx["text"]
    status = impl.status()
    numbers = {"text_file": Path(text.path).name, "placeholder": text.is_placeholder,
               "text_sha256": text.sha256[:12], "impl": status,
               "impl_versions": ", ".join(sorted(set(status.values())))}
    return True, numbers


def check_target_pattern(ctx: dict) -> tuple[bool, dict]:
    """The LoRA target expression matches projections of decoder layers and nothing else (no model needed)."""
    pattern = target_pattern()
    inside = ["model.layers.0.linear_attn.in_proj_qkv", "model.layers.1.linear_attn.in_proj_z",
              "model.layers.2.linear_attn.out_proj", "model.layers.3.self_attn.q_proj",
              "model.layers.3.self_attn.o_proj", "model.layers.23.mlp.down_proj"]
    outside = ["lm_head", "model.embed_tokens", "model.layers.0.linear_attn.in_proj_a",
               "model.layers.0.self_attn.out_proj", "model.visual.blocks.0.mlp.gate_proj",
               "base_model.model.model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj.base_layer"]
    wrong = [name for name in inside if not re.fullmatch(pattern, name)]
    wrong += [name for name in outside if re.fullmatch(pattern, name)]
    return not wrong, {"pattern": pattern, "wrong": wrong, "n_wrong": len(wrong)}


def check_tokenizer(ctx: dict) -> tuple[bool, dict]:
    """The tokenizer loads; its end-of-turn and pad tokens are the expected ones; four real prompts are
    built."""
    from transformers import AutoTokenizer

    from accessdesk.harness.hf_backend import resolve_revision

    args, text = ctx["args"], ctx["text"]
    revision = resolve_revision(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **({"revision": revision} if revision else {}))
    _set_tokenizer(ctx, tokenizer)
    layout = ctx["layout"]
    numbers = {"end_of_turn": tokenizer.eos_token, "end_of_turn_id": layout.im_end_id,
               "pad_id": layout.pad_id, "vocabulary": len(tokenizer),
               "prompt_lengths": [len(prompt) for prompt in ctx["prompts"]]}
    ok = tokenizer.eos_token == END_OF_TURN and layout.im_end_id != layout.pad_id
    return ok and len(set(numbers["prompt_lengths"])) > 1, numbers


def _set_tokenizer(ctx: dict, tokenizer: Any) -> None:
    """Put the tokenizer, the five-tool layout and four first prompts of different lengths into `ctx`."""
    text = ctx["text"]
    layout = TokenLayout(tokenizer, text.tool_specs("five"))
    dev = ctx["families"]["dev"]
    chosen = [(dev[0], 0), (dev[1], 1), (dev[2], 0), (dev[3], 3)]         # (family, index of the wording)
    prompts = [layout.first_prompt(text.system_prompt, family["wordings"][w]["text"]) for family, w in chosen]
    ctx.update({"tokenizer": tokenizer, "layout": layout, "prompts": prompts})


def check_first_prompt(ctx: dict) -> tuple[bool, dict]:
    """The first prompt of a scenario, decoded: every tool schema, the rules, the request, no sixth tool,
    and the empty think block at the very end (thinking is off)."""
    text, cfg, layout = ctx["text"], ctx["cfg"], ctx["layout"]
    episode = Episode(ctx["families"]["dev"][0], 1, text, cfg)
    ids = layout.first_prompt(episode.system_prompt, episode.request_text)
    decoded = ctx["tokenizer"].decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    missing = []
    for spec in episode.tool_specs:
        pieces = {"name": spec.name, "description": spec.description}
        for param in spec.params:
            pieces[f"parameter {param.key}"] = param.name
            pieces[f"description of {param.key}"] = param.description
            for value in param.enum or ():
                pieces[f"value {value} of {param.key}"] = value
        # The schema is written into the prompt as JSON, so a piece is looked for in its JSON spelling.
        missing += [f"{spec.key}: {what}" for what, piece in pieces.items()
                    if json.dumps(piece, ensure_ascii=False)[1:-1] not in decoded]
    if episode.system_prompt not in decoded:
        missing.append("the system prompt (the rules)")
    if episode.request_text not in decoded:
        missing.append("the request")
    if not decoded.endswith(EMPTY_THINK_BLOCK):
        missing.append("the empty think block at the end")
    extra_tool = json.dumps(text.tools[EXTRA_TOOL_KEY].name, ensure_ascii=False)
    numbers = {"n_tokens": len(ids), "n_tools": len(episode.tool_specs), "missing": missing,
               "n_missing": len(missing), "sixth_tool_in_prompt": extra_tool in decoded,
               "last_characters": decoded[-40:]}
    return not missing and not numbers["sixth_tool_in_prompt"], numbers


def check_episode_layout(ctx: dict) -> tuple[bool, dict]:
    """The token layout of a scripted episode (the correct solution of an approval scenario, four turns):
    the segments, what each one holds, and the loss mask."""
    text, cfg, layout, tokenizer = ctx["text"], ctx["cfg"], ctx["layout"], ctx["tokenizer"]
    family = next(f for f in ctx["families"]["dev"] if f["outcome"] == "approval")
    example = oracle_example(family, 1, text, cfg, layout)
    ctx["example"] = example
    record, buffer, weights = example["record"], example["buffer"], example["weights"]
    segments, turns = buffer.segments, record["turns"]

    def decode(start: int, end: int) -> str:
        return tokenizer.decode(buffer.ids[start:end], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)

    problems = []
    if [s.kind for s in segments] != ["prompt"] + ["sampled", "suffix"] * (len(turns) - 1) + ["sampled"]:
        problems.append(f"segment kinds: {[s.kind for s in segments]}")
    if any(a.end != b.start for a, b in zip(segments, segments[1:])) or segments[0].start != 0 \
            or segments[-1].end != len(buffer.ids):
        problems.append("the segments do not tile the buffer")
    for segment in segments:
        inside = weights[segment.start:segment.end]
        if segment.kind == "sampled":
            turn = turns[segment.turn]
            if decode(segment.start, segment.end) != turn["assistant_text"] + END_OF_TURN:
                problems.append(f"turn {segment.turn}: the sampled tokens are not the assistant text and "
                                "its end-of-turn token")
            if buffer.ids[segment.end - 1] != layout.im_end_id or set(inside) != {1}:
                problems.append(f"turn {segment.turn}: end-of-turn token or weights of the sampled tokens")
        else:
            if set(inside) != {0}:
                problems.append(f"a {segment.kind} segment has a token with weight 1")
        if segment.kind == "suffix":
            expected = SUFFIX_HEAD + turns[segment.turn]["tool_response"].strip() + SUFFIX_TAIL
            if decode(segment.start, segment.end) != expected:
                problems.append(f"turn {segment.turn}: the suffix is not the tool result and the next header")
    if not decode(segments[0].start, segments[0].end).endswith(EMPTY_THINK_BLOCK):
        problems.append("the first prompt does not end with the empty think block")
    if not (record["passed"] and record["reward"] == cfg.pass_reward):
        problems.append(f"the scripted solution did not pass cleanly (reward {record['reward']})")
    numbers = {"family": family["family_id"], "n_turns": len(turns), "n_tokens": len(buffer.ids),
               "n_loss_tokens": sum(weights), "reward": record["reward"], "problems": problems,
               "n_problems": len(problems)}
    return not problems, numbers


def check_model_load(ctx: dict) -> tuple[bool, dict]:
    """The model loads in bfloat16 as the text-only class, in eval mode, with its tokenizer padding on the
    left."""
    import torch

    from accessdesk.harness.hf_backend import TEXT_ONLY_CLASS, causal_lm, load_model

    args = ctx["args"]
    model, tokenizer = load_model(args.model, args.revision, device=args.device)
    ctx["model"] = model
    _set_tokenizer(ctx, tokenizer)
    core = causal_lm(model)
    dtypes = sorted({str(param.dtype) for param in core.parameters()})
    layer_types = list(core.config.layer_types)
    numbers = {
        "class": type(core).__name__, "dtypes": dtypes, "device": str(core.device),
        "parameters": sum(param.numel() for param in core.parameters()),
        "layers": len(core.model.layers),
        "linear_attention_layers": layer_types.count("linear_attention"),
        "full_attention_layers": layer_types.count("full_attention"),
        "hidden_size": core.config.hidden_size, "vocabulary": core.config.vocab_size,
        "head_tied_to_embeddings": core.lm_head.weight is core.model.embed_tokens.weight,
        "padding_side": tokenizer.padding_side, "training_mode": core.training,
        "gpu_memory_gb": (round(torch.cuda.memory_allocated() / 2**30, 2)
                          if torch.cuda.is_available() else None),
    }
    ok = (numbers["class"] == TEXT_ONLY_CLASS and dtypes == ["torch.bfloat16"] and not core.training
          and tokenizer.padding_side == "left" and len(layer_types) == len(core.model.layers))
    return ok, numbers


def check_kernels(ctx: dict) -> tuple[bool, dict]:
    """Which Gated DeltaNet kernels run. The slow torch path for the delta rule is a FAIL; the convolution
    on the torch path is expected when the optional package causal-conv1d is not installed. `use_kernels`
    must be off: with it, transformers may replace the whole forward of a linear-attention layer by a hub
    kernel (modeling_qwen3_5.py line 495), and the LoRA wrappers inside it would not be called."""
    from accessdesk.harness.hf_backend import kernel_report

    report = kernel_report(ctx["model"])
    ctx["kernels"] = report
    numbers = {"gated_delta_rule": report["gated_delta_rule"], "causal_conv1d": report["causal_conv1d"],
               "fast_path": report["fast_path"], "use_kernels": report["use_kernels"],
               "torch_fallback_functions": report["torch_fallback_functions"],
               "imports": {name: result["ok"] for name, result in report["imports"].items()}}
    return report["gated_delta_rule"] != "torch" and not report["use_kernels"], numbers


def _logprobs(ctx: dict, seqs: list[list[int]], weights: list[list[int]],
              micro_batch: int) -> list[list[float]]:
    """`token_logprobs` without gradients on the model of `ctx`, as plain lists of numbers."""
    from accessdesk.train.forward import token_logprobs

    result = token_logprobs(ctx["model"], seqs, weights, pad_id=ctx["layout"].pad_id, micro_batch=micro_batch,
                            with_grad=False)
    return [values.tolist() for values in result]


def _within(stats: dict, bounds: tuple[float, float]) -> bool:
    return stats["mean"] < bounds[0] and stats["max"] < bounds[1]


def check_padding(ctx: dict) -> tuple[bool, dict]:
    """400 prompt tokens alone, and as the second row of a batch of two: once beside a partner of the same
    length (no pad token) and once beside a longer partner (80 pad tokens on their left).

    Passes if padding adds nothing to the rounding difference that a batch of two has anyway (padded mean
    below twice the no-pad mean plus 0.01) and the padded numbers are inside `PADDED_AGAINST_ALONE`.
    The same check, with a leak made on purpose, is tests/gpu/test_forward.py."""
    short = ctx["prompts"][0][:400]
    same_length, longer = ctx["prompts"][1][7:407], ctx["prompts"][1][:480]

    def second_row(partner: list[int]) -> list[float]:
        return _logprobs(ctx, [partner, short], [all_but_first(partner), all_but_first(short)], 2)[1]

    alone = _logprobs(ctx, [short], [all_but_first(short)], 1)[0]
    no_pad = abs_stats([a - b for a, b in zip(second_row(same_length), alone)])
    with_pad = abs_stats([a - b for a, b in zip(second_row(longer), alone)])
    numbers = {"n": with_pad["n"], "padded_mean": with_pad["mean"], "padded_p99": with_pad["p99"],
               "padded_max": with_pad["max"], "no_pad_mean": no_pad["mean"], "no_pad_p99": no_pad["p99"]}
    ok = (len(alone) == 399 and with_pad["mean"] < 2 * no_pad["mean"] + 0.01
          and with_pad["mean"] < PADDED_AGAINST_ALONE[0] and with_pad["p99"] < PADDED_AGAINST_ALONE[1])
    return ok, numbers


def check_sampler(ctx: dict) -> tuple[bool, dict]:
    """The sampler starts (it refuses anything but plain sampling) and its settings are the config's."""
    from accessdesk.harness.hf_backend import Sampler

    cfg = ctx["cfg"]
    sampler = Sampler(ctx["model"], ctx["tokenizer"], cfg)
    ctx["sampler"] = sampler
    settings = sampler.settings()
    ctx["sampler_settings"] = settings
    effective = settings["generate_effective_config"]
    numbers = {key: settings[key] for key in ("temperature", "top_k", "top_p", "max_new_tokens", "chunk_size",
                                              "stop_id", "pad_id", "padding_side", "logits_processors")}
    numbers.update({"generate_do_sample": effective.get("do_sample"),
                    "generate_num_beams": effective.get("num_beams"),
                    "generate_eos_token_id": effective.get("eos_token_id"),
                    "generate_repetition_penalty": effective.get("repetition_penalty")})
    as_config = (cfg.temperature, cfg.top_k, cfg.top_p, cfg.max_new_tokens)
    found = (settings["temperature"], settings["top_k"], settings["top_p"], settings["max_new_tokens"])
    ok = (found == as_config and settings["logits_processors"][-1] == "LogprobRecorder"
          and settings["stop_id"] == ctx["layout"].im_end_id and effective.get("do_sample") is True)
    return ok, numbers


def _short_sampler(ctx: dict) -> Any:
    """A sampler with a token cap of 64, for the checks that sample one turn."""
    from accessdesk.harness.hf_backend import Sampler

    if "short_sampler" not in ctx:
        cfg = dataclasses.replace(ctx["cfg"], max_new_tokens=64)
        ctx["short_sampler"] = Sampler(ctx["model"], ctx["tokenizer"], cfg)
    return ctx["short_sampler"]


def check_determinism(ctx: dict) -> tuple[bool, dict]:
    """The same prompts with the same seed give the same tokens; another seed gives other tokens.

    The second half is checked with a HOT sampler (temperature 2.0, as tests/gpu/test_sampler.py does). At the
    config's temperature 1.0 it is not a valid check: a model that is nearly certain of its first call gives
    the same tokens for most seeds. First run, 2026-11-17: the 4B gave the same four first turns for seeds 21
    and 22 (the check failed); the summed log-probability of those turns is in `first_turns_logprob`, and
    exp of it is the chance that another seed repeats them (`chance_of_a_repeat`; measured 0.97 on the 4B,
    summed log-probability -0.033 over 114 tokens; the number of each run is in smoke.json)."""
    from accessdesk.harness.hf_backend import Sampler

    sampler, prompts = _short_sampler(ctx), ctx["prompts"]
    first, again, other = (sampler.sample(prompts, seed=seed) for seed in (21, 21, 22))
    ctx["turns"] = first
    hot_cfg = dataclasses.replace(ctx["cfg"], max_new_tokens=32, temperature=2.0)
    hot = Sampler(ctx["model"], ctx["tokenizer"], hot_cfg)
    hot_first, hot_again, hot_other = (hot.sample(prompts, seed=seed) for seed in (21, 21, 22))
    logprob_gap = abs_stats([a - b for x, y in zip(first, again) for a, b in zip(x.logprobs, y.logprobs)])
    total_logprob = sum(sum(x.logprobs) for x in first)
    numbers = {"same_seed_same_tokens": [x.ids for x in first] == [x.ids for x in again],
               "hot_same_seed_same_tokens": [x.ids for x in hot_first] == [x.ids for x in hot_again],
               "hot_other_seed_other_tokens": [x.ids for x in hot_first] != [x.ids for x in hot_other],
               "other_seed_other_tokens_at_temperature_1": [x.ids for x in first] != [x.ids for x in other],
               "first_turns_logprob": total_logprob, "chance_of_a_repeat": math.exp(total_logprob),
               "same_seed_max_logprob_difference": logprob_gap["max"],
               "turns_ended_by_the_model": sum(1 for x in first if not x.truncated),
               "tokens": [len(x.ids) for x in first]}
    ok = (numbers["same_seed_same_tokens"] and numbers["hot_same_seed_same_tokens"]
          and numbers["hot_other_seed_other_tokens"])
    return ok, numbers


def check_parity(ctx: dict, mode: str) -> tuple[bool, dict]:
    """PLAN 5.6: the log-probabilities recorded while sampling against the training forward on the same ids,
    with the model in eval mode or in train mode."""
    from accessdesk.harness.hf_backend import causal_lm

    prompts = ctx["prompts"]
    turns = ctx.get("turns") or _short_sampler(ctx).sample(prompts, seed=21)
    seqs = [prompt + turn.ids for prompt, turn in zip(prompts, turns)]
    weights = [[0] * len(prompt) + [1] * len(turn.ids) for prompt, turn in zip(prompts, turns)]
    core = causal_lm(ctx["model"])
    try:
        core.train(mode == "train")
        ours = _logprobs(ctx, seqs, weights, 2)
    finally:
        core.eval()
    stats = abs_stats([a - b for mine, turn in zip(ours, turns) for a, b in zip(mine, turn.logprobs)])
    every_token = stats["n"] == sum(len(turn.ids) for turn in turns)
    return every_token and _within(stats, SAMPLER_AGAINST_TRAINER), stats


def check_lora_coverage(ctx: dict) -> tuple[bool, dict]:
    """Add the adapters. Every required projection of both layer types and of the MLP has one, nothing outside
    the decoder layers has one, and only float32 LoRA weights are trainable. Prints the coverage table."""
    from accessdesk.harness.hf_backend import causal_lm
    from accessdesk.train.lora import (add_lora, coverage_problems, coverage_summary, coverage_table,
                                       trainable_parameters)

    example = ctx["example"]
    ctx["base_logprobs"] = _logprobs(ctx, [example["ids"]], [example["weights"]], 1)[0]
    ctx["model"] = add_lora(ctx["model"], seed=0)
    ctx["lora"] = True
    core = causal_lm(ctx["model"])
    table = coverage_table(ctx["model"])
    problems = coverage_problems(table, core.config)
    params = trainable_parameters(ctx["model"])
    summary = coverage_summary(table)
    print("      layer type        block        module       wrapped  shape             adapter weights")
    for row in summary:
        print(f"      {row['layer_type']!s:17} {row['block']!s:12} {row['module']:12} "
              f"{row['wrapped']:3d}/{row['of']:<3d}  {row['shape']!s:17} {row['lora_params']:>12,d}")
    numbers = {"wrapped_modules": sum(1 for row in table if row["wrapped"]),
               "trainable_parameters": sum(param.numel() for param in params),
               "all_parameters": sum(param.numel() for param in core.parameters()),
               "not_wrapped": sorted({row["module"] for row in table if not row["wrapped"]}),
               "problems": problems[:10], "n_problems": len(problems), "table": summary}
    return not problems, numbers


def check_lora_is_a_no_op(ctx: dict) -> tuple[bool, dict]:
    """Fresh adapters (lora_B is zero) change no log-probability."""
    example = ctx["example"]
    after = _logprobs(ctx, [example["ids"]], [example["weights"]], 1)[0]
    stats = abs_stats([a - b for a, b in zip(after, ctx["base_logprobs"])])
    return _within(stats, SAME_COMPUTATION), stats


def check_backward(ctx: dict, checkpointing: bool) -> tuple[bool, dict]:
    """One backward pass on the scripted episode with the loss mask. The lora_B gradient is not zero for
    linear_attn, for self_attn and for the MLP; the lora_A gradient is exactly zero (lora_B is still zero)."""
    from accessdesk.train.forward import set_gradient_checkpointing
    from accessdesk.train.lora import gradient_report, trainable_parameters
    from accessdesk.train.sft import sft_gradient

    model, example = ctx["model"], ctx["example"]
    reset_peak_memory()
    try:
        set_gradient_checkpointing(model, checkpointing)
        model.train()
        for param in trainable_parameters(model):
            param.grad = None
        loss = sft_gradient(model, [example], pad_id=ctx["layout"].pad_id, micro_batch=1)
        report = gradient_report(model)
    finally:
        for param in trainable_parameters(model):
            param.grad = None
        set_gradient_checkpointing(model, False)
        model.eval()
    norms = report["by_block"]
    numbers = {"loss": loss["loss"], "n_loss_tokens": loss["n_tokens"], "n_tokens": len(example["ids"]),
               "lora_B_by_block": norms, "lora_B_by_layer_type": report["by_layer_type"],
               "lora_B_by_module": report["by_module"], "lora_B": report["lora_B"],
               "lora_A": report["lora_A"], "without_gradient": report["without_gradient"]["n"],
               "peak_memory_gb": peak_memory_gb()}
    ok = (sorted(norms) == ["linear_attn", "mlp", "self_attn"]
          and all(math.isfinite(norm) and norm > 0 for norm in norms.values())
          and report["lora_A"] == 0.0 and report["without_gradient"]["n"] == 0
          and math.isfinite(loss["loss"]))
    if checkpointing and "lora_B_norm" in ctx:
        relative = abs(report["lora_B"] - ctx["lora_B_norm"]) / ctx["lora_B_norm"]
        numbers["relative_difference_to_no_checkpointing"] = relative
        ok = ok and relative < 0.2
    if not checkpointing:
        ctx["lora_B_norm"] = report["lora_B"]
    return ok, numbers


def check_rollout(ctx: dict) -> tuple[bool, dict]:
    """A rollout of the model on dev scenarios (with the zero adapters attached, as in training): no harness
    error, at least one well-formed call, and every saved episode replays to its saved final state."""
    from accessdesk.harness.hf_backend import rollout

    args, text, cfg = ctx["args"], ctx["text"], ctx["cfg"]
    scenarios = [(family, wording["style"]) for family in ctx["families"]["dev"]
                 for wording in family["wordings"]][:args.episodes]
    episodes = [Episode(family, style, text, cfg) for family, style in scenarios]
    reset_peak_memory()
    started = time.perf_counter()
    records = rollout(ctx["sampler"], {"five": ctx["layout"]}, episodes, run_seed=0, pass_index=0,
                      meta={"run": ctx["out_dir"].name, "arm": "smoke"})
    seconds = time.perf_counter() - started
    ctx["rollout_records"] = records
    without_tokens = [{key: value for key, value in record.items() if key != "tokens"} for record in records]
    for name, rows in (("smoke_episodes.jsonl", without_tokens),
                       ("smoke_tokens.jsonl", [rollout_script.token_line(record) for record in records])):
        (ctx["out_dir"] / name).unlink(missing_ok=True)
        rollout_script.append_jsonl(ctx["out_dir"] / name, rows)
    summary = rollout_script.summarise(without_tokens)
    errors = [record["harness_error"] for record in records if record["harness_error"] is not None]
    not_replayed = sum(1 for record, (family, _) in zip(records, scenarios) if record["harness_error"] is None
                       and replay(record, family, text, cfg) != record["final_state"])
    invalid = summary.get("invalid_call_rate")
    numbers = {"episodes": len(records), "harness_errors": len(errors),
               "first_harness_error": (errors or [None])[0], "pass_rate": summary.get("pass_rate"),
               "well_formed_share": None if invalid is None else 1 - invalid,
               "tokens_per_turn": summary.get("tokens_per_turn"),
               "turns_per_episode": summary.get("mean_turns"),
               "turns_cut_off": summary.get("truncated_turn_rate"), "close_rate": summary.get("share_closed"),
               "pass_rate_by_outcome": summary.get("pass_rate_by_outcome"), "not_replayed": not_replayed,
               "seconds": round(seconds, 1), "episodes_per_hour": round(len(records) / seconds * 3600),
               "peak_memory_gb": peak_memory_gb()}
    ok = not errors and not_replayed == 0 and (numbers["well_formed_share"] or 0) > 0
    return ok, numbers


def check_episode_parity(ctx: dict) -> tuple[bool, dict]:
    """The gap that GRPO logs at every step, on whole multi-turn episodes of the rollout: sampler
    log-probabilities against the training forward on the stored ids, with the loss mask."""
    from accessdesk.train.sft import token_weights

    records = [record for record in ctx["rollout_records"] if record["harness_error"] is None][:16]
    seqs = [record["tokens"]["ids"] for record in records]
    weights = [token_weights(record["tokens"]) for record in records]
    ours = _logprobs(ctx, seqs, weights, 4)
    differences = [a - b for mine, record in zip(ours, records)
                   for a, b in zip(mine, record["tokens"]["logprobs"])]
    stats = abs_stats(differences)
    same_length = all(len(mine) == len(record["tokens"]["logprobs"]) for mine, record in zip(ours, records))
    numbers = {**stats, "episodes": len(records), "longest_episode_tokens": max(len(seq) for seq in seqs)}
    return same_length and _within(stats, SAMPLER_AGAINST_TRAINER), numbers


def check_optimizer_step(ctx: dict) -> tuple[bool, dict]:
    """One AdamW step at the SFT learning rate on the scripted episode: the adapters of all three blocks
    change, and the loss on that episode goes down."""
    from accessdesk.train.lora import lora_state, weight_change_report
    from accessdesk.train.sft import clip_and_step, make_optimizer, sft_gradient

    model, example = ctx["model"], ctx["example"]
    initial = lora_state(model)
    optimizer = make_optimizer(model, lr=1e-4)
    try:
        model.train()
        before = sft_gradient(model, [example], pad_id=ctx["layout"].pad_id, micro_batch=1)
        grad_norm = clip_and_step(model, optimizer, 1.0)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
    ctx["trained_logprobs"] = _logprobs(ctx, [example["ids"]], [example["weights"]], 1)[0]
    loss_after = -sum(ctx["trained_logprobs"]) / len(ctx["trained_logprobs"])
    change = weight_change_report(model, initial)
    numbers = {"loss_before": before["loss"], "loss_after": loss_after,
               "grad_norm_before_clipping": grad_norm,
               "weight_change_by_block": change["by_block"], "weight_change_total": change["total"]}
    ok = (sorted(change["by_block"]) == ["linear_attn", "mlp", "self_attn"]
          and all(value > 0 for value in change["by_block"].values())
          and math.isfinite(loss_after) and loss_after < before["loss"])
    return ok, numbers


def check_adapter_round_trip(ctx: dict) -> tuple[bool, dict]:
    """Save the trained adapter, free the model, load the model again with the saved adapter: it gives the
    trained model's log-probabilities, and they differ from the base model's (the adapter has an effect)."""
    from safetensors.torch import load_file

    from accessdesk.harness.hf_backend import load_model
    from accessdesk.train.lora import coverage_table, lora_fingerprint, save_adapter

    args, example = ctx["args"], ctx["example"]
    folder = ctx["out_dir"] / "adapter_round_trip"
    save_adapter(ctx["model"], folder)
    fingerprint = lora_fingerprint(ctx["model"])
    n_wrapped = sum(1 for row in coverage_table(ctx["model"]) if row["wrapped"])
    saved = load_file(str(folder / "adapter_model.safetensors"))
    keys_ok = len(saved) == 2 * n_wrapped and all(".lora_A." in key or ".lora_B." in key for key in saved)
    megabytes = (folder / "adapter_model.safetensors").stat().st_size / 2**20
    del saved
    for key in ("model", "sampler", "short_sampler"):        # every reference to the first model
        ctx.pop(key, None)
    free_gpu_memory()
    ctx["model"], _ = load_model(args.model, args.revision, adapter_dir=str(folder), device=args.device)
    reloaded = _logprobs(ctx, [example["ids"]], [example["weights"]], 1)[0]
    to_trained = abs_stats([a - b for a, b in zip(reloaded, ctx["trained_logprobs"])])
    to_base = abs_stats([a - b for a, b in zip(reloaded, ctx["base_logprobs"])])
    fingerprint_again = lora_fingerprint(ctx["model"])
    numbers = {"file_mb": round(megabytes, 1), "saved_tensors": 2 * n_wrapped, "keys_ok": keys_ok,
               "reloaded_against_trained_mean": to_trained["mean"],
               "reloaded_against_trained_max": to_trained["max"],
               "reloaded_against_base_mean": to_base["mean"],
               "fingerprint_saved": fingerprint, "fingerprint_loaded": fingerprint_again}
    # "The adapter has an effect" is asked relative to the reload error: the reloaded model must be at least
    # ten times nearer to the trained model than to the base model, and not equal to the base model.
    # Measured on 2026-11-17, mean |difference| on the 213 loss tokens of the scripted episode:
    #   2B: reloaded against trained 0 (exactly), against base 0.0042 to 0.0043
    #   4B: reloaded against trained 0 (exactly), against base 0.00023
    # The first version asked for "against base > 0.001", a guess: the untrained 4B already gives this
    # episode a mean log-probability of -0.00023 per token, so one step cannot move it by more than that.
    # An adapter that was saved but not loaded gives "against base" = 0 and fails here.
    ok = (keys_ok and _within(to_trained, SAME_COMPUTATION) and to_base["mean"] > ADAPTER_EFFECT_FLOOR
          and to_base["mean"] > 10 * to_trained["mean"]
          and math.isclose(fingerprint, fingerprint_again, rel_tol=1e-6))
    return ok, numbers


# (name, function, the keys of `ctx` that it needs from earlier checks)
CHECKS: list[tuple[str, Callable[[dict], tuple[bool, dict]], tuple[str, ...]]] = [
    ("environment", check_environment, ()),
    ("text_and_impl", check_text_and_impl, ()),
    ("target_pattern", check_target_pattern, ()),
    ("tokenizer", check_tokenizer, ()),
    ("first_prompt", check_first_prompt, ("tokenizer", "layout")),
    ("episode_layout", check_episode_layout, ("tokenizer", "layout")),
    ("model_load", check_model_load, ()),
    ("kernels", check_kernels, ("model",)),
    ("padding", check_padding, ("model", "prompts")),
    ("sampler", check_sampler, ("model", "layout")),
    ("determinism", check_determinism, ("model", "prompts")),
    ("parity_eval_mode", lambda ctx: check_parity(ctx, "eval"), ("model", "prompts")),
    ("parity_train_mode", lambda ctx: check_parity(ctx, "train"), ("model", "prompts")),
    ("lora_coverage", check_lora_coverage, ("model", "example")),
    ("lora_is_a_no_op", check_lora_is_a_no_op, ("lora", "example", "base_logprobs")),
    ("backward", lambda ctx: check_backward(ctx, False), ("lora", "example")),
    ("backward_checkpointing", lambda ctx: check_backward(ctx, True), ("lora", "example")),
    ("rollout", check_rollout, ("model", "sampler")),
    ("episode_parity", check_episode_parity, ("model", "rollout_records")),
    ("optimizer_step", check_optimizer_step, ("lora", "example")),
    ("adapter_round_trip", check_adapter_round_trip, ("lora", "trained_logprobs", "base_logprobs")),
]


# ---------------------------------------------------------------------------
# Running the checks
# ---------------------------------------------------------------------------
def run_check(name: str, function: Callable[[dict], tuple[bool, dict]], needs: tuple[str, ...],
              ctx: dict) -> dict:
    """Run one check; never raises (except for Ctrl-C). Returns its result and prints its line."""
    started = time.perf_counter()
    ok, numbers, error = False, {}, None
    missing = [key for key in needs if key not in ctx]
    if missing:
        error = f"not run: it needs {missing} from an earlier check that failed or was left out"
    else:
        try:
            ok, numbers = function(ctx)
        except Exception as exc:  # every failure of a check, an out-of-memory error included
            error = f"{type(exc).__name__}: {exc}"[:2000]
            print("      " + traceback.format_exc().strip().replace("\n", "\n      ")[-3000:])
    if error is not None:
        free_gpu_memory()
    seconds = time.perf_counter() - started
    print(f"{'PASS' if ok else 'FAIL'}  {name:24s} {seconds:7.1f} s  {error or brief(numbers)}", flush=True)
    return {"name": name, "passed": bool(ok), "seconds": round(seconds, 2), "error": error,
            "numbers": numbers}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--episodes", type=int, default=32, help="episodes of the rollout check")
    parser.add_argument("--only", default=None, help="run only these checks, by name, separated by commas "
                        "(checks that need an earlier one will fail)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = load_text_or_exit(args)
    cfg = RunConfig()
    names = [name for name, _, _ in CHECKS]
    wanted = names if args.only is None else [name.strip() for name in args.only.split(",")]
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        print(f"unknown checks {unknown}; known: {names}", file=sys.stderr)
        return 2
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx: dict = {"args": args, "text": text, "cfg": cfg, "out_dir": out_dir,
                 "families": build_families(text, cfg)}
    print(f"smoke test of {args.model}; results in {out_dir}")
    results: list[dict] = []
    for name, function, needs in CHECKS:
        if name not in wanted:
            continue
        results.append(run_check(name, function, needs, ctx))
        failed = [result["name"] for result in results if not result["passed"]]
        write_json(out_dir / "smoke.json", {"complete": False, "model": args.model, "passed": not failed,
                                            "failed": failed, "checks": results})
    failed = [result["name"] for result in results if not result["passed"]]
    write_json(out_dir / "smoke.json", {"complete": True, "model": args.model, "passed": not failed,
                                        "failed": failed, "checks": results})
    try:
        manifest = run_manifest(args, cfg, text, extra={
            "environment": ctx.get("environment"), "kernels": ctx.get("kernels"),
            "sampler": ctx.get("sampler_settings"), "smoke_failed": failed})
        write_json(out_dir / "run_manifest.json", manifest)
    except Exception as exc:  # the manifest must not hide the result of the checks
        print(f"run_manifest.json was not written: {type(exc).__name__}: {exc}")
    print(f"SMOKE {'PASSED' if not failed else 'FAILED'}: {len(results) - len(failed)} of {len(results)} "
          "checks passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
