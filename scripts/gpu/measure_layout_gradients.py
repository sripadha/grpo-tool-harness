"""Measure how far the gradient of one saved GRPO step depends on the micro-batch layout. Runs on the GPU.

    python scripts/gpu/measure_layout_gradients.py --model Qwen/Qwen3.5-2B \\
        --adapter results/main/sft/adapter \\
        --episodes results/main/checks/pilot/rollouts/step_00000.episodes.jsonl \\
        --tokens results/main/checks/pilot/rollouts/step_00000.tokens.jsonl \\
        --out results/main/checks/layout_gradients

WRITTEN ON A LAPTOP WITHOUT TORCH, 2026-11-18. THE GPU PART (the class `ModelBackend`) HAS NOT RUN YET.
Everything else is tested without a model in tests/core/test_measure_layout_gradients.py.

The question. The model computes in bfloat16. A control on made-up token sequences
(_dev/gpu_trial/gradient_controls_result.md) showed that the gradient of the same sequences differs by 16 to
54% between two micro-batch layouts in bfloat16, and by about 1% in float32. This script measures the same
on the real episodes of one GRPO step, and beside it the sampling noise of the gradient.

The input is one step of a run of scripts/gpu/train_grpo.py that was made with --save-tokens: its episodes
and its token data. The advantages come from the saved rewards through the library (`plan_update`), and the
loss constant is the run's (`GRPOConfig.norm_const`). The gradient of the GRPO loss of the step is taken at
all LoRA weights, as one long vector, in these ways:

    bfloat16_trainer_layout   bfloat16, micro-batches as the trainer makes them: the library's
                              `accumulate_policy_gradient` with the run's micro-batch (4), in its order
    bfloat16_one_episode      bfloat16, one episode per micro-batch (no padding)
    float32_one_episode       float32, one episode per micro-batch: the reference
    halves                    the yardstick. As bfloat16_trainer_layout, on two halves of the kept groups;
                              the groups are split at random with a seed, in up to 20 different splits. The
                              two halves give the sampling noise of the whole step: see "How to read it".

Then the same four ways with an SFT-style loss on all valid episodes of the step: weight 1 on every sampled
token, the mean over the tokens (the library's `sft_gradient`). There the halves split the episodes. The
trainer's layout is the same for both losses: the micro-batch of the GRPO run. scripts/gpu/train_sft.py
itself trains with micro-batches of 2 (`SFTConfig`); that layout is not measured here.

Saving first. A GPU stage writes every way to disk as soon as it is measured: first its arrays
(stage_<stage>.<loss>.<way>.npz), then its numbers (stage_<stage>.<loss>.<way>.json). The halves of a loss
are written again after every split (stage_bfloat16.<loss>.halves.json). Only after all of that does the
stage read what merely describes the machine (package versions, the GPU, the kernels, the TF32 switches).
Each of those reads is in a try block: one that fails is written as its error text, and the stage goes on.
The two calls that only serve the seconds and the peak memory of a pass are guarded in the same way
(`timing_and_memory_errors`). stage_<stage>.json is written last. If a stage fails, start it again with
the same command and --stage <stage>: it keeps every way and every split that the earlier start saved
from the same inputs, settings and code, and measures only the rest. A piece that was saved from other
inputs, settings or code is measured again and written over.

A gradient is read from `.grad` of the LoRA weights (float32, as the optimiser would get it). Every
comparison is computed on the CPU in float64. Nothing is trained: there is no optimiser and no optimiser
step, and the fingerprint of the LoRA weights and the hashes of the adapter files are checked before and
after.

The stages. --stage all (the default) does three things one after another:
    bfloat16   a new Python process: the three bfloat16 ways for both losses
    float32    a new Python process: the reference for both losses
    compare    this process, without torch: reads what the two stages saved and writes the result
Each stage can also be run alone (--stage bfloat16, then --stage float32, then --stage compare, with the
same other arguments). Why two processes, and not one that frees the bfloat16 model and then loads the
float32 one: a process that ends gives all its GPU memory back, for certain. Inside one process a single
forgotten reference keeps the first model on the card (about 4.5 GB), and the float32 model (about 9 GB of
weights) and its work would have to fit beside it and beside torch's cache: on 24 GB that can fail. With two
processes the float32 stage has the whole card, the peak memory of each dtype is a clean number, and a
failure of the float32 stage does not lose the bfloat16 numbers. The two sizes are estimates of the review
in _dev/gpu_trial/bf16_judgement.json, not measurements.

The float32 model is the bfloat16 model with every weight turned into float32 (`model.float()` after the
library's `load_model`). So it has exactly the same weights, and only the arithmetic is more exact. The
stage tries to switch TF32 off with torch's two old switches; a switch that cannot be set is written as
its error text, and the stage goes on.

The float32 reference is not exact. A control on made-up sequences (C3 in
_dev/gpu_trial/gradient_controls_result.md) gave a relative difference of 0.003 to 0.012 between two
micro-batch layouts in float32, with torch's TF32 switch off for matmul and on for cuDNN. So a relative
difference against float32_one_episode below about 0.01 cannot be told from the reference's own error. The
result says so beside those numbers (`float32_reference`). A possible cause, which is not verified: the
Triton kernels of flash-linear-attention may multiply float32 matrices in TF32, and torch's switches do
not reach inside a Triton kernel. Each stage writes down what can be read about this: `tf32` (torch's
switches), `kernels` (which kernels ran) and `kernel_sources` (the version of Triton, its environment
variable for the precision, and what the source files say).

The settings of the step (scenarios per step, attempts per scenario, the token count of the loss constant,
the micro-batch, the tie tolerance) are read from run_manifest.json of the run, if --episodes lies in its
rollouts/ folder. An option on the command line that differs from the run is refused. Without that manifest
the defaults are 16, 8, 200, 4 and 0.0 (ANALYSIS_PLAN.md section 7 and `GRPOConfig`).

Written into --out:
    stage_<stage>.<loss>.<way>.npz             one way of a stage: its gradient and its log-probabilities
    stage_<stage>.<loss>.<way>.json            the numbers of that way, and the hash of its .npz file
    stage_bfloat16.<loss>.halves.json          the halves of one loss, written again after every split
    stage_bfloat16.json, stage_float32.json    a stage, written last: all its numbers, the hashes of its
                                               .npz files, and what describes the machine
    layout_gradients.json                      the result. Its keys:
        ways                 per loss and way: dtype, micro-batch, episodes, loss tokens, loss, norm of the
                             gradient; seconds and peak GPU memory of the gradient pass and of the pass
                             that computes the log-probabilities, each under a name that says which pass
        comparisons          per loss, for each pair of ways: cosine and relative difference
                             |other - reference| / |reference|, over all LoRA weights and for the LoRA
                             weights of the first, a middle and the last decoder layer
        float32_reference    the smallest relative difference that the float32 reference can resolve, and
                             where that number comes from
        halves               per loss: every split with its cosine and its relative noise; the median,
                             mean, smallest and largest of each; and `meaning`: what the two numbers say
        logprob_differences  per loss, for each pair of ways: mean, 99th percentile and largest difference
                             of the log-probability of one sampled token
        table                one row per loss and set of weights: the numbers of the pairs beside the
                             numbers of the halves. The script prints it.
        step, settings, inputs, weights, checks, stages, code    what was measured, on what, by which code
    The .npz files hold six gradients of about 63 MB each (own calculation: 4 bytes times the
    15,630,336 LoRA weights that run_manifest.json of results/_smoke/trial_step counts). With them
    --stage compare can be run again anywhere, without a GPU. Delete them when they are not needed.

How to read it. Two numbers of the same kind stand side by side. Each is the length of a difference,
divided by the length of the gradient of the whole step:
    relative_difference       of bfloat16_trainer_layout against float32_one_episode: how far bfloat16 and
                              the trainer's layout move the gradient
    relative_noise            of the halves: how far the sampling noise moves it. For the GRPO loss both
                              halves are divided by the constant of the whole step, so the gradient of the
                              whole step is g_A + g_B, and the number is |g_A - g_B| / |g_A + g_B|. Pure
                              noise gives about 1, no noise gives 0. For the SFT-style loss each half is a
                              mean over its own tokens (n_A and n_B of them); there the number is
                              sqrt(n_A * n_B) / (n_A + n_B) * |g_A - g_B| / |gradient of the whole|, which
                              is right also for halves of unequal size (`relative_noise_of_a_mean`).
The layout does no harm if the first number is much smaller than the second. The script prints both and
judges nothing. Three things to know. The cosine of the two halves is kept, but it is no yardstick for the
cosine of two layouts: pure noise gives a halves cosine near 0, so "nearer to 1 than the halves" is met
almost for free when few groups are kept. The halves are computed in bfloat16 in the trainer's layout, so
relative_noise also holds the arithmetic error of that layout. And with an odd number of units one half
holds one unit more, so the difference then also holds about one part in `n_units` of the true gradient.

The log-probabilities of a way are taken in a second forward pass in eval mode without gradients, in the
same micro-batches. For the GRPO loss the library reports how far its own gradient pass is from the
sampler (`logprob_gap`); the same numbers of the second pass stand beside them, so a reader can see that the
two passes agree. The peak GPU memory is read for each of the two passes by itself. `peak_gpu_memory_gb` of
a stage is the largest over loading the model, every gradient pass (the halves too) and every
log-probability pass.

Exit codes: 0 the stage is done; 2 the run was refused before a model loaded; 1 any other failure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for the scripts next to this file

import rollout as rollout_script  # noqa: E402
import train_sft as train_sft_script  # noqa: E402
from accessdesk import impl  # noqa: E402
from accessdesk.harness.hf_backend import resolve_revision  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402
from accessdesk.rng import Rng, derive_seed  # noqa: E402
from accessdesk.train.grpo import GRPOConfig, abs_stats, plan_update  # noqa: E402
from accessdesk.train.sft import attach_tokens, token_weights  # noqa: E402
from sft_data import (Refused, command_line, file_entry, frozen_code, training_code,  # noqa: E402
                      utc_now, warn_if_not_as_frozen)

SCRIPT = "scripts/gpu/measure_layout_gradients.py"
RESULT_NAME = "layout_gradients.json"
GPU_STAGES = ("bfloat16", "float32")          # a GPU stage is named after the dtype of its model
STAGES = ("all", *GPU_STAGES, "compare")
GRPO, SFT_STYLE = "grpo", "sft_style"
LOSSES = (GRPO, SFT_STYLE)
BF16_TRAINER, BF16_SINGLE, FP32_SINGLE = ("bfloat16_trainer_layout", "bfloat16_one_episode",
                                          "float32_one_episode")
WAYS_OF_STAGE = {"bfloat16": (BF16_TRAINER, BF16_SINGLE), "float32": (FP32_SINGLE,)}
# (other, reference): the pairs of ways that are compared. The float32 way is the reference where it is in.
PAIRS = ((BF16_TRAINER, FP32_SINGLE), (BF16_SINGLE, FP32_SINGLE), (BF16_TRAINER, BF16_SINGLE))
ALL_WEIGHTS = "all LoRA weights"
SPLIT_LABEL = "layout_halves"                 # the label in the seed of a split (`split_in_halves`)
DEFAULT_SPLITS = 20
# Option -> the field of GRPOConfig that it sets. Read from the run's manifest if the option is not given.
STEP_OPTIONS = {"scenarios_per_step": "n_scenarios", "group_size": "group_size",
                "reference_tokens": "reference_tokens", "micro_batch": "micro_batch", "tie_tol": "tie_tol"}
# ANALYSIS_PLAN.md section 7: every GRPO run is started with --reference-tokens 200. The other values of
# a step that has no run manifest are the defaults of GRPOConfig.
PLAN_REFERENCE_TOKENS = 200
# What is copied from the line of the step in the run's train_log.jsonl, to stand beside this script's
# numbers of the way bfloat16_trainer_layout.
LOG_KEYS = ("loss", "grad_norm", "kept_groups", "n_trained_episodes", "n_loss_tokens", "norm_const",
            "logprob_gap")
# Two fingerprints of the LoRA weights (`lora_fingerprint`: the sum of their absolute values, about 62,000
# for the adapters of the trial runs) count as equal within this relative distance. One optimiser step of
# the trial moved the fingerprint by 7e-06 of its value (_dev/gpu_trial/gpu_trial_report.json: 62293.587
# before a step, 62294.028 after), so a step would be seen.
FINGERPRINT_TOLERANCE = 1e-9
# The float32 reference cannot resolve a relative difference below about this number. Where it comes
# from: the last line of the table in _dev/gpu_trial/gradient_controls_result.md (control C3, float32, one
# sequence per micro-batch against one micro-batch of two) gives 3.29e-03, 8.69e-03, 1.01e-02 and 1.20e-02
# at four weights. The round number 0.01 is a judgement on those four.
FLOAT32_REFERENCE = {
    "resolves_relative_differences_above": 0.01,
    "meaning": "A relative difference against float32_one_episode below about 0.01 cannot be told from "
               "the error of the float32 reference itself.",
    "measured_in": "_dev/gpu_trial/gradient_controls_result.md, control C3: in float32, one sequence per "
                   "micro-batch against one micro-batch of two gave a relative difference of 3.29e-03, "
                   "8.69e-03, 1.01e-02 and 1.20e-02 at four weights, on two made-up sequences without an "
                   "adapter. In that control torch's TF32 switch was off for matmul and on for cuDNN "
                   "(_dev/gpu_trial/logs/gradcontrols.log, line 76). 0.01 is a round number for these "
                   "four, not a measurement on this step.",
    "possible_cause_not_verified": "The Triton kernels of flash-linear-attention may multiply float32 "
                                   "matrices in TF32. torch's TF32 switches do not reach inside a Triton "
                                   "kernel. See tf32, kernels and kernel_sources of the stage float32.",
}
# What the two numbers of the halves mean, per loss: one sentence each, written into the result.
_COSINE_MEANING = ("The cosine of the gradients of the two halves: near 0 if the gradient is only sampling "
                   "noise, 1 if there is no sampling noise; it cannot be compared with a relative "
                   "difference.")
_NOISE_LIMITS = ["The halves are computed in bfloat16 in the trainer's layout, so relative_noise also holds "
                 "the arithmetic error of that layout.",
                 "With an odd number of units one half holds one unit more, and the difference then also "
                 "holds about one part in n_units of the true gradient.",
                 "All splits are splits of the same step, so their range is not a confidence interval."]
HALVES_MEANING = {
    GRPO: {
        "halves_and_whole": "Both halves are divided by the loss constant of the whole step (norm_const), "
                            "so in exact arithmetic the gradient of the whole step is g_A + g_B.",
        "cosine": _COSINE_MEANING,
        "relative_noise": "|g_A - g_B| / |g_A + g_B|: the sampling noise of the gradient of the whole step "
                          "as a share of its length, to be compared with relative_difference of the "
                          "comparisons; near 1 if the gradient is only sampling noise, 0 if there is no "
                          "sampling noise.",
        "limits": _NOISE_LIMITS,
    },
    SFT_STYLE: {
        "halves_and_whole": "Each half is a mean over its own loss tokens (n_A and n_B), so in exact "
                            "arithmetic the gradient of the whole is s_A * g_A + s_B * g_B, with the "
                            "shares s_A = n_A / (n_A + n_B) and s_B = n_B / (n_A + n_B) "
                            "(share_of_the_whole of each split).",
        "cosine": _COSINE_MEANING,
        "relative_noise": "sqrt(n_A * n_B) / (n_A + n_B) * |g_A - g_B| / |s_A * g_A + s_B * g_B|: an "
                          "estimate of the sampling noise of the gradient of the whole as a share of its "
                          "length, to be compared with relative_difference of the comparisons; near 1 if "
                          "the gradient is only sampling noise, 0 if there is no sampling noise. The "
                          "factor makes the number right for halves with unequal numbers of loss tokens "
                          "(relative_noise_of_a_mean); it takes the noise of a half to grow with its "
                          "number of loss tokens.",
        "limits": _NOISE_LIMITS,
    },
}
# torch's switches for TF32 (float32 products rounded to about 10 binary digits on the GPU). The two old
# ones are set in the float32 stage. All are read after the measurement, each by itself. The names of the
# newer ones (`fp32_precision`) are typed from memory of the torch documentation and are not verified:
# a name that does not exist is written as its error text.
TF32_OLD_SWITCHES = ("backends.cuda.matmul", "backends.cudnn")          # each has `allow_tf32`
TF32_NEW_SWITCHES = ("backends.fp32_precision", "backends.cuda.matmul.fp32_precision",
                     "backends.cudnn.fp32_precision", "backends.cudnn.conv.fp32_precision",
                     "backends.cudnn.rnn.fp32_precision")
# What is counted in the source files of flash-linear-attention: the calls of Triton's matrix product, and
# the two arguments with which such a call can name its precision. The sub-folders of fla/ops whose files
# are listed one by one are typed from memory of that package and are not verified.
DOT_WORDS = ("tl.dot(", "allow_tf32", "input_precision")
FLA_FOLDERS_LISTED = ("gated_delta_rule", "common", "utils")
ERROR_LENGTH = 300                            # an error text is cut to this many characters
_STEP_FILE = re.compile(r"step_(\d+)\.episodes\.jsonl")
_LAYER_OF_WEIGHT = re.compile(r"model\.layers\.(\d+)\.")


def stage_file(stage: str) -> str:
    """The file that a GPU stage writes last: all its numbers, with `"complete": true`."""
    return f"stage_{stage}.json"


def way_files(stage: str, loss: str, way: str) -> tuple[str, str]:
    """The two files of one way of a stage, written as soon as the way is measured: its numbers, and its
    arrays."""
    return f"stage_{stage}.{loss}.{way}.json", f"stage_{stage}.{loss}.{way}.npz"


def halves_file(stage: str, loss: str) -> str:
    """The file of the halves of one loss in a stage. It is written again after every split."""
    return f"stage_{stage}.{loss}.halves.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", required=True, help="model id on the Hub, or a local folder")
    parser.add_argument("--revision", default=None, help="model revision (default: the pinned one, if known)")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="folder of the adapter that the step was run with; it is read, never written")
    parser.add_argument("--episodes", type=Path, required=True,
                        help="rollouts/step_XXXXX.episodes.jsonl of a GRPO run made with --save-tokens")
    parser.add_argument("--tokens", type=Path, required=True,
                        help="rollouts/step_XXXXX.tokens.jsonl of the same step")
    parser.add_argument("--out", type=Path, required=True, help="output folder; must not hold a result yet")
    parser.add_argument("--scenarios-per-step", type=int, default=None,
                        help="groups of the step (default: the run's value, else 16)")
    parser.add_argument("--group-size", type=int, default=None,
                        help="attempts per scenario (default: the run's value, else 8)")
    parser.add_argument("--reference-tokens", type=int, default=None,
                        help=f"the token count in the constant of the loss (default: the run's value, else "
                             f"{PLAN_REFERENCE_TOKENS})")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="episodes per forward pass in the trainer's layout (default: the run's value, "
                             "else 4)")
    parser.add_argument("--tie-tol", type=float, default=None,
                        help="groups whose rewards lie within this distance are dropped (default: the "
                             "run's value, else 0.0)")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="switch gradient checkpointing off (default: as the run had it, else on)")
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS,
                        help="different splits into two halves, for each loss")
    parser.add_argument("--split-seed", type=int, default=0, help="seed of the splits")
    parser.add_argument("--stage", choices=STAGES, default="all",
                        help="all: bfloat16 and float32, each in a new process, then compare")
    parser.add_argument("--allow-other-settings", action="store_true",
                        help="accept settings, a model or a revision that differ from the run's manifest")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# The splits into two halves (no torch)
# ---------------------------------------------------------------------------
def split_in_halves(n_items: int, seed: int, index: int) -> tuple[list[int], list[int]]:
    """Split the positions 0 to n_items - 1 at random into two halves; each half is sorted.

    The first half holds n_items // 2 positions and the second half the others, so with at least two
    positions no half is empty. The split depends only on (n_items, seed, index): the same three numbers
    give the same split on every machine. Raises ValueError for fewer than two positions.
    """
    if n_items < 2:
        raise ValueError(f"two halves need at least 2 positions, not {n_items}")
    order = Rng(derive_seed(seed, SPLIT_LABEL, index)).shuffle(list(range(n_items)))
    return sorted(order[:n_items // 2]), sorted(order[n_items // 2:])


def n_possible_splits(n_items: int) -> int:
    """How many different splits of `n_items` positions `split_in_halves` can give. Two splits with the
    same two halves are one split, whichever half is named first. 0 for fewer than two positions.

    Example: 4 positions give 3 splits (01|23, 02|13, 03|12); 5 positions give 10.
    """
    if n_items < 2:
        return 0
    return math.comb(n_items, n_items // 2) // (2 if n_items % 2 == 0 else 1)


def different_splits(n_items: int, n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Up to `n_splits` splits of the positions 0 to n_items - 1 into two halves, no split twice.

    The splits are those of `split_in_halves` for the indices 0, 1, 2, ..., without the ones that were
    there already. If fewer than `n_splits` different splits exist, all of them are returned. An empty
    list for fewer than two positions. The result depends only on the three arguments.
    """
    wanted = min(n_splits, n_possible_splits(n_items))
    found: list[tuple[list[int], list[int]]] = []
    seen: set[frozenset] = set()
    index = 0
    while len(found) < wanted:
        halves = split_in_halves(n_items, seed, index)
        index += 1
        key = frozenset(tuple(half) for half in halves)
        if key not in seen:
            seen.add(key)
            found.append(halves)
    return found


# ---------------------------------------------------------------------------
# Comparing two gradients (no torch)
# ---------------------------------------------------------------------------
def _vector(values: Any) -> np.ndarray:
    """`values` (a list, or an array of any float type) as a flat float64 array. Raises ValueError if a
    value is not a finite number."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError("a value is not a finite number")
    return vector


def _same_length(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"the two vectors have {a.size} and {b.size} entries")


def norm(values: Any) -> float:
    """The length of a vector: the square root of the sum of its squared entries, in float64."""
    return float(np.linalg.norm(_vector(values)))


def cosine(a: Any, b: Any) -> float | None:
    """The cosine of the angle between two vectors, in float64: (a . b) / (|a| |b|).

    1.0: the same direction. 0.0: at a right angle. The lengths play no part. None if one of the vectors
    is all zeros: the angle is then not defined. Raises ValueError for vectors of different lengths.
    """
    a, b = _vector(a), _vector(b)
    _same_length(a, b)
    length_a, length_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if length_a == 0.0 or length_b == 0.0:
        return None
    return max(-1.0, min(1.0, float(np.dot(a, b)) / (length_a * length_b)))


def relative_difference(other: Any, reference: Any) -> float | None:
    """|other - reference| / |reference|, in float64, with |.| the length of a vector (`norm`).

    0.0: the two are equal. None if the reference is all zeros. This is the "relative difference" of
    _dev/gpu_trial/gradient_controls_result.md. Raises ValueError for vectors of different lengths.
    """
    other, reference = _vector(other), _vector(reference)
    _same_length(other, reference)
    length = float(np.linalg.norm(reference))
    return None if length == 0.0 else float(np.linalg.norm(other - reference)) / length


def relative_noise(part_a: Any, part_b: Any) -> float | None:
    """|part_a - part_b| / |part_a + part_b|, in float64, with |.| the length of a vector (`norm`).

    The two parts are the shares of two halves of a step in the gradient of the whole step, so their sum
    is that gradient. If the halves are equally large and hold independent draws, the difference of the
    parts is spread like the sampling noise in their sum. So the number is one draw of the sampling noise
    of the whole step, as a share of its gradient: about 1 if the gradient is only noise, 0.0 if the two
    parts are equal. It is read like `relative_difference`. None if the sum is all zeros. Raises
    ValueError for vectors of different lengths.
    """
    part_a, part_b = _vector(part_a), _vector(part_b)
    _same_length(part_a, part_b)
    length = float(np.linalg.norm(part_a + part_b))
    return None if length == 0.0 else float(np.linalg.norm(part_a - part_b)) / length


def relative_noise_of_a_mean(mean_a: Any, mean_b: Any, n_a: int, n_b: int) -> float | None:
    """The sampling noise of a mean over n_a + n_b units as a share of its length, from the means of two
    halves with n_a and n_b units:

        sqrt(n_a * n_b) / (n_a + n_b) * |mean_a - mean_b| / |whole|,
        whole = (n_a * mean_a + n_b * mean_b) / (n_a + n_b).

    Why this factor. If every unit adds independent noise of variance v, the difference of the two means
    has the variance v * (1 / n_a + 1 / n_b) = v * (n_a + n_b) / (n_a * n_b), and the mean of the whole has
    the variance v / (n_a + n_b). The ratio of the two standard deviations is sqrt(n_a * n_b) / (n_a + n_b).
    With equal halves the factor is 1/2, and the number is `relative_noise` of the two halves, each times
    its share. With unequal halves that simpler form also counts (share_a - share_b) times the true
    gradient as noise (found by a reviewer on 2026-11-18: 0.415 against a true 0.264 for halves of 10 and
    20 units); this form does not. It takes the noise of a half to grow with its number of units; here
    the units are loss tokens, and tokens of one episode are not independent, so this is an estimate.
    None if a half has no unit or the whole is all zeros. Raises ValueError for vectors of different
    lengths.
    """
    mean_a, mean_b = _vector(mean_a), _vector(mean_b)
    _same_length(mean_a, mean_b)
    if n_a <= 0 or n_b <= 0:
        return None
    total = n_a + n_b
    whole = (n_a * mean_a + n_b * mean_b) / total
    length = float(np.linalg.norm(whole))
    if length == 0.0:
        return None
    return float(np.sqrt(n_a * n_b) / total * np.linalg.norm(mean_a - mean_b) / length)


def shares_of_the_whole(loss: str, n_loss_tokens: Sequence[int]) -> list[float]:
    """The factor that turns the gradient of each half into its part of the gradient of the whole.

    "grpo": 1.0 for each half. `accumulate_policy_gradient` divides each half by the loss constant of the
    whole step, so the gradient of the whole step is the sum of the two halves.
    "sft_style": the loss tokens of the half divided by the loss tokens of both halves. `sft_gradient`
    gives each half the mean over its own tokens, and the mean over all tokens is
    (n_a * mean_a + n_b * mean_b) / (n_a + n_b).
    """
    if loss == GRPO:
        return [1.0 for _ in n_loss_tokens]
    total = sum(n_loss_tokens)
    return [n_tokens / total for n_tokens in n_loss_tokens]


def weight_sets(weights: Sequence[tuple[str, int]], n_layers: int) -> dict[str, list[tuple[int, int]]]:
    """Where the sets of weights lie in the long gradient vector: label -> list of (start, stop).

    `weights` are (name, number of entries) of the LoRA weights in the order of the vector; a name is
    that of the model's parameter, such as "model.layers.3.self_attn.q_proj.lora_A.default.weight".
    The sets: all LoRA weights, and the LoRA weights of the first decoder layer (0), of a middle one
    (n_layers // 2) and of the last one (n_layers - 1), the layers of the control tests in
    tests/gpu/test_train_fixes_gpu.py. `stop` is not part of a range; ranges that touch are joined.
    Raises ValueError for a name that is not of a decoder layer, and if one of the three layers has no
    weight.
    """
    layers = {"first layer (0)": 0, f"middle layer ({n_layers // 2})": n_layers // 2,
              f"last layer ({n_layers - 1})": n_layers - 1}
    sets: dict[str, list[tuple[int, int]]] = {ALL_WEIGHTS: [], **{label: [] for label in layers}}
    start = 0
    for name, size in weights:
        match = _LAYER_OF_WEIGHT.match(name)
        if match is None:
            raise ValueError(f"{name!r} is not a weight of a decoder layer")
        labels = [ALL_WEIGHTS] + [label for label, layer in layers.items() if layer == int(match.group(1))]
        for label in labels:
            ranges = sets[label]
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], start + size)
            else:
                ranges.append((start, start + size))
        start += size
    empty = [label for label, ranges in sets.items() if not ranges]
    if empty:
        raise ValueError(f"no LoRA weight in: {empty}")
    return sets


def take(vector: np.ndarray, ranges: Sequence[Sequence[int]]) -> np.ndarray:
    """The entries of `vector` in the ranges (start, stop), in order, as one array."""
    pieces = [vector[start:stop] for start, stop in ranges]
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def n_in(ranges: Sequence[Sequence[int]]) -> int:
    return sum(stop - start for start, stop in ranges)


def compare_gradients(gradients: dict[str, Any], sets: dict[str, Sequence[Sequence[int]]],
                      pairs: Sequence[tuple[str, str]] = PAIRS) -> list[dict]:
    """For every pair (other, reference) of ways, and every set of weights: the cosine of the two
    gradients, their relative difference, and the length of each.

    `gradients` maps a way to its gradient vector, `sets` a label to its ranges (`weight_sets`). Returns
    one dict per pair, in the order of `pairs`: `other`, `reference`, and `weights`, which maps each label
    to `n_weights`, `cosine`, `relative_difference`, `norm_other` and `norm_reference`. A cosine or a
    relative difference that is not defined (a gradient that is all zeros) is None.
    """
    rows = []
    for other, reference in pairs:
        a, b = _vector(gradients[other]), _vector(gradients[reference])
        _same_length(a, b)
        by_set = {}
        for label, ranges in sets.items():
            part_a, part_b = take(a, ranges), take(b, ranges)
            by_set[label] = {"n_weights": n_in(ranges), "cosine": cosine(part_a, part_b),
                             "relative_difference": relative_difference(part_a, part_b),
                             "norm_other": norm(part_a), "norm_reference": norm(part_b)}
        rows.append({"other": other, "reference": reference, "weights": by_set})
    return rows


def summary_of(values: Sequence[float | None]) -> dict:
    """n, mean, median, smallest and largest of the values that are numbers; `n_not_defined` counts the
    None values. With no number the four statistics are None."""
    numbers = sorted(float(value) for value in values if value is not None)
    summary: dict = {"n": len(numbers), "n_not_defined": len(values) - len(numbers),
                     "mean": None, "median": None, "min": None, "max": None}
    if numbers:
        middle = len(numbers) // 2
        median = numbers[middle] if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2
        summary.update({"mean": sum(numbers) / len(numbers), "median": median, "min": numbers[0],
                        "max": numbers[-1]})
    return summary


def logprob_differences(logprobs: dict[str, Any], pairs: Sequence[tuple[str, str]] = PAIRS) -> list[dict]:
    """For every pair (other, reference) of ways: n, mean, 99th percentile and largest of the absolute
    difference of the log-probability of one sampled token (the library's `abs_stats`). The four numbers
    are None for a pair without a token."""
    rows = []
    for other, reference in pairs:
        a, b = _vector(logprobs[other]), _vector(logprobs[reference])
        _same_length(a, b)
        stats = abs_stats((a - b).tolist()) or {"n": 0, "mean": None, "p99": None, "max": None}
        rows.append({"other": other, "reference": reference, **stats})
    return rows


def pair_name(other: str, reference: str) -> str:
    return f"{other} against {reference}"


def result_table(comparisons: dict[str, list[dict]], halves: dict[str, dict]) -> list[dict]:
    """The table of the result: one row per loss and set of weights.

    A row has `loss`, `weights`, `n_weights`, `cosine` and `relative_difference` (each maps the name of a
    pair, "<other> against <reference>", to its number), and `halves_relative_noise` and `halves_cosine`
    (`summary_of` over the splits of that loss, for that set of weights). `halves_relative_noise` is the
    number to hold against `relative_difference`. The rows are in the order of the losses and of the sets.
    """
    rows = []
    for loss, pairs in comparisons.items():
        for label in pairs[0]["weights"]:
            names = [pair_name(pair["other"], pair["reference"]) for pair in pairs]
            cells = [pair["weights"][label] for pair in pairs]
            rows.append({"loss": loss, "weights": label, "n_weights": cells[0]["n_weights"],
                         "cosine": {name: cell["cosine"] for name, cell in zip(names, cells)},
                         "relative_difference": {name: cell["relative_difference"]
                                                 for name, cell in zip(names, cells)},
                         "halves_relative_noise": halves[loss]["relative_noise_summary"][label],
                         "halves_cosine": halves[loss]["cosine_summary"][label]})
    return rows


def _shown(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def format_table(rows: list[dict]) -> str:
    """The table as lines of text: for each row the cosine and the relative difference of every pair,
    then the relative noise of the whole step and the cosine of the two halves."""
    lines = []
    for row in rows:
        lines.append(f"{row['loss']}, {row['weights']} ({row['n_weights']} weights)")
        for name, value in row["cosine"].items():
            lines.append(f"    cosine {name:<60} {_shown(value)}   relative difference "
                         f"{_shown(row['relative_difference'][name])}")
        noise = row["halves_relative_noise"]
        lines.append(f"    relative noise of the whole step, from the two halves, {noise['n']} splits: "
                     f"median {_shown(noise['median'])}, smallest {_shown(noise['min'])}, largest "
                     f"{_shown(noise['max'])}")
        halves = row["halves_cosine"]
        lines.append(f"    cosine of the two halves, {halves['n']} splits: mean {_shown(halves['mean'])}, "
                     f"median {_shown(halves['median'])}, smallest {_shown(halves['min'])}, largest "
                     f"{_shown(halves['max'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# What only describes the machine (no torch). Nothing here may stop a stage: see `read_or_error`.
# ---------------------------------------------------------------------------
def read_or_error(read: Callable[[], Any]) -> Any:
    """What `read()` returns. If it raises, {"error": "<type of the error>: <its text>"}, cut to
    ERROR_LENGTH characters: the error is written down, and the caller goes on."""
    try:
        return read()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:ERROR_LENGTH]}


def read_attribute(root: Any, dotted: str) -> Any:
    """`root.a.b.c` for the name "a.b.c"."""
    value = root
    for name in dotted.split("."):
        value = getattr(value, name)
    return value


def _plain(value: Any) -> Any:
    """A value as a JSON file can hold it: a bool, a number, a text or None as it is, anything else as
    its text."""
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


def switch_tf32_off(torch: Any) -> dict:
    """Set `allow_tf32` to False on the two old TF32 switches of `torch` (TF32_OLD_SWITCHES). Returns,
    per switch, "set to False" or {"error": ...}: a switch that cannot be set does not stop the other one
    and does not stop the stage."""
    def switch_off(name: str) -> str:
        setattr(read_attribute(torch, name), "allow_tf32", False)
        return "set to False"

    return {f"torch.{name}.allow_tf32": read_or_error(lambda name=name: switch_off(name))
            for name in TF32_OLD_SWITCHES}


def tf32_switches(torch: Any) -> dict:
    """What the TF32 switches of `torch` say now: the two old ones, the matmul precision, and the newer
    `fp32_precision` ones. Each is read by itself; one that cannot be read is {"error": ...}."""
    names = [f"{name}.allow_tf32" for name in TF32_OLD_SWITCHES]
    switches = {f"torch.{name}": read_or_error(lambda name=name: _plain(read_attribute(torch, name)))
                for name in names}
    switches["torch.get_float32_matmul_precision()"] = read_or_error(
        lambda: _plain(torch.get_float32_matmul_precision()))
    for name in TF32_NEW_SWITCHES:
        switches[f"torch.{name}"] = read_or_error(lambda name=name: _plain(read_attribute(torch, name)))
    return switches


def package_folder(name: str) -> Path:
    """The folder of an installed package. The package is not imported. Raises FileNotFoundError if it
    is not installed."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(f"the package {name} is not installed")
    return Path(list(spec.submodule_search_locations)[0])


def _source_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    return sorted(folder.rglob("*.py"))


def count_words(folder: Path, words: Sequence[str], listed: Sequence[str]) -> dict:
    """How often each of `words` stands in the .py files under `folder`. Keys: `folder`, `n_files` (the
    files that were read), `total` (word -> count over all files) and `files` (file -> word -> count, for
    each file that holds one of the words and lies in one of the sub-folders `listed`). Raises
    FileNotFoundError if `folder` is not a folder."""
    paths = _source_files(folder)
    total = {word: 0 for word in words}
    files = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        counts = {word: text.count(word) for word in words}
        for word in words:
            total[word] += counts[word]
        relative = path.relative_to(folder)
        if any(counts.values()) and relative.parts[0] in listed:
            files[relative.as_posix()] = counts
    return {"folder": str(folder), "n_files": len(paths), "total": total, "files": files}


def lines_with(folder: Path, words: Sequence[str], limit: int = 40) -> dict:
    """The lines of the .py files under `folder` that hold one of `words`; capitals play no part, and
    the words are given in small letters. Keys: `folder`, `n_lines` (all such lines) and `lines` (the
    first `limit` of them as "<file>:<line number>: <the line, at most 200 characters>"). Raises
    FileNotFoundError if `folder` is not a folder."""
    found = []
    for path in _source_files(folder):
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if any(word in line.lower() for word in words):
                found.append(f"{path.relative_to(folder).as_posix()}:{number}: {line.strip()[:200]}")
    return {"folder": str(folder), "n_lines": len(found), "lines": found[:limit]}


def kernel_sources_report() -> dict:
    """What can be read, without a GPU, about TF32 inside the Triton kernels of flash-linear-attention
    (the package `fla`). torch's TF32 switches do not reach inside a Triton kernel: there the precision
    of a float32 matrix product is an argument of `tl.dot`, with a default that Triton chooses. Keys:

        triton_version            the installed version of Triton
        TRITON_F32_DEFAULT        the value of this environment variable, None if it is not set. From
                                  memory of Triton's source, not verified: it sets the default precision
                                  of `tl.dot`
        tf32_in_triton_language   the lines of triton/language/*.py that name TF32 (`lines_with`): they
                                  show the default of the installed version
        dot_calls_in_fla_ops      how often fla/ops calls `tl.dot(`, and how often it names `allow_tf32`
                                  or `input_precision` (`count_words`)
    Each of the four is read by itself; one that cannot be read is {"error": ...}. Nothing is imported.
    """
    return {
        "triton_version": read_or_error(lambda: importlib.metadata.version("triton")),
        "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT"),
        "tf32_in_triton_language": read_or_error(
            lambda: lines_with(package_folder("triton") / "language", ("tf32",))),
        "dot_calls_in_fla_ops": read_or_error(
            lambda: count_words(package_folder("fla") / "ops", DOT_WORDS, FLA_FOLDERS_LISTED)),
    }


def write_json_whole(path: Path, obj: dict) -> None:
    """Write a JSON file so that `path` holds the old file or the whole new one, never a part of one: the
    text goes into a file beside it, and that file then takes the name. A file that is written again
    after every split must not be lost by a failure in the middle of a write."""
    part = path.with_name(path.name + ".part")
    rollout_script.write_json(part, obj)
    os.replace(part, path)


# ---------------------------------------------------------------------------
# The step: its settings, its episodes, what each loss is taken on (no torch)
# ---------------------------------------------------------------------------
def run_of(episodes_file: Path) -> tuple[Path | None, int | None]:
    """(the run folder, the step number) if the file is rollouts/step_<number>.episodes.jsonl of a run
    folder; (None, None) for any other path."""
    match = _STEP_FILE.fullmatch(episodes_file.name)
    if match is None or episodes_file.parent.name != "rollouts":
        return None, None
    return episodes_file.parent.parent, int(match.group(1))


def _read_json(path: Path) -> dict | None:
    """The file as a dict; None if it is not there or is not a JSON object."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def step_settings(args: argparse.Namespace, run_config: dict | None) -> tuple[GRPOConfig, dict]:
    """(the GRPO settings of the step, where each of them comes from).

    `run_config` is `grpo_config` of the run's manifest, or None. For each setting: the option of the
    command line if it is given, else the run's value, else the default (STEP_OPTIONS,
    PLAN_REFERENCE_TOKENS). Refused if an option differs from the run's value and --allow-other-settings
    is not given: the measured loss would not be the loss of the run. Only the fields that the gradient
    of a step depends on are set; the other fields of the returned config are not used.
    """
    defaults = {**GRPOConfig().to_dict(), "reference_tokens": PLAN_REFERENCE_TOKENS}
    run_config = run_config or {}
    values, sources, different = {}, {}, {}
    for option, field in STEP_OPTIONS.items():
        given, of_run = getattr(args, option), run_config.get(field)
        if given is not None and of_run is not None and given != of_run:
            different[f"--{option.replace('_', '-')}"] = (given, of_run)
        if given is not None:
            values[field], sources[field] = given, "the command line"
        elif of_run is not None:
            values[field], sources[field] = of_run, "run_manifest.json of the run"
        else:
            values[field], sources[field] = defaults[field], "the default"
    if different and not args.allow_other_settings:
        raise Refused(f"options that differ from the run's manifest (given, the run's): {different}; "
                      "leave them out, or pass --allow-other-settings")
    of_run = run_config.get("gradient_checkpointing")
    if args.no_gradient_checkpointing:
        values["gradient_checkpointing"], sources["gradient_checkpointing"] = False, "the command line"
    elif of_run is not None:
        values["gradient_checkpointing"], sources["gradient_checkpointing"] = (
            bool(of_run), "run_manifest.json of the run")
    else:
        values["gradient_checkpointing"], sources["gradient_checkpointing"] = True, "the default"
    for field in ("n_scenarios", "group_size", "reference_tokens", "micro_batch"):
        if values[field] < 1:
            raise Refused(f"{field} must be at least 1, not {values[field]}")
    return GRPOConfig(**values), sources


def step_plan(records: list[dict], gcfg: GRPOConfig) -> dict:
    """What each loss is taken on, from the records of the step (each with its `tokens`). Keys:

        advantages    [groups, attempts] from the library's `plan_update`: reward minus the mean of the group
        kept_groups   the numbers of the groups that are not dropped, in order
        scenarios     the scenario id of every group
        episodes      per loss, the indices into `records` that go through the model: for "grpo" the valid
                      episodes of the kept groups (`selected` of `plan_update`), for "sft_style" every
                      valid episode. A harness error is in neither list.
        weights       index -> one weight per token of the episode, 1 on a sampled token (`token_weights`)
        examples      index -> the episode as `sft_gradient` takes it: `ids` and `weights`
    Refused if the records are not `n_scenarios` groups of `group_size` attempts of different scenarios,
    or if an episode has another number of sampled tokens than of sampler log-probabilities.
    """
    try:
        plan = plan_update(records, gcfg)
    except ValueError as error:
        raise Refused(f"the episodes are not {gcfg.n_scenarios} groups of {gcfg.group_size} attempts: "
                      f"{error}") from error
    scenarios = [records[group * gcfg.group_size]["scenario_id"] for group in range(gcfg.n_scenarios)]
    if len(set(scenarios)) != len(scenarios):
        raise Refused(f"the episodes hold {len(set(scenarios))} scenarios, not {gcfg.n_scenarios}: a step "
                      "never holds a scenario twice. Give the --scenarios-per-step and --group-size of "
                      "the run")
    valid = [index for index, record in enumerate(records) if record["harness_error"] is None]
    weights = {}
    for index in valid:
        tokens = records[index]["tokens"]
        try:
            weights[index] = token_weights(tokens)
        except ValueError as error:
            raise Refused(f"episode {index}: its token data cannot be read: {error}") from error
        if sum(weights[index]) != len(tokens["logprobs"]):
            raise Refused(f"episode {index} has {sum(weights[index])} sampled tokens but "
                          f"{len(tokens['logprobs'])} sampler log-probabilities")
    return {"advantages": plan["advantages"],
            "kept_groups": [int(group) for group in np.flatnonzero(plan["keep"])],
            "scenarios": scenarios,
            "episodes": {GRPO: [int(index) for index in plan["selected"]], SFT_STYLE: valid},
            "weights": weights,
            "examples": {index: {"ids": records[index]["tokens"]["ids"], "weights": weights[index]}
                         for index in valid}}


def describe_step(records: list[dict], plan: dict, gcfg: GRPOConfig) -> dict:
    """The counts of the step, for the result: episodes, harness errors, groups, the kept groups with
    their scenarios, and per loss the episodes and the loss tokens that go through the model."""
    def counts(indices: list[int]) -> dict:
        return {"n_episodes": len(indices),
                "n_loss_tokens": sum(sum(plan["weights"][index]) for index in indices)}

    return {"n_episodes": len(records), "n_harness_errors": len(records) - len(plan["episodes"][SFT_STYLE]),
            "n_groups": gcfg.n_scenarios, "group_size": gcfg.group_size,
            "n_kept_groups": len(plan["kept_groups"]),
            "kept_groups": [{"group": group, "scenario_id": plan["scenarios"][group]}
                            for group in plan["kept_groups"]],
            "norm_const": gcfg.norm_const,
            GRPO: counts(plan["episodes"][GRPO]), SFT_STYLE: counts(plan["episodes"][SFT_STYLE])}


def weights_of_the_step(step_number: int | None, adapter: dict, run_manifest: dict | None) -> bool | None:
    """True if --adapter is known to hold the weights with which the trainer ran this step: the step is
    step 0 and the adapter files are the files that the run started from. False if the step is step 0 and
    they are other files. None if it cannot be told (another step, or no run manifest)."""
    recorded = ((run_manifest or {}).get("start_adapter") or {}).get("files")
    if step_number != 0 or not isinstance(recorded, dict):
        return None
    return all((recorded.get(name) or {}).get("sha256") == entry["sha256"] for name, entry in adapter.items())


def logged_step(run_dir: Path | None, step_number: int | None) -> dict | None:
    """The numbers of LOG_KEYS that the trainer wrote for this step into train_log.jsonl of the run; None
    if there is no such line."""
    if run_dir is None or not (run_dir / "train_log.jsonl").is_file():
        return None
    for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            log = json.loads(line)
        except ValueError:
            continue
        if isinstance(log, dict) and log.get("step") == step_number:
            return {key: log.get(key) for key in LOG_KEYS}
    return None


def prepare(args: argparse.Namespace) -> dict:
    """Everything that a stage needs and that asks for no GPU: the inputs with their hashes, the settings,
    the records and what each loss is taken on. Refused if an input is missing or does not fit. Keys:
    `records`, `plan`, `gcfg`, `identity` (what the two GPU stages must share), `step`, `sources`,
    `run` (what the run folder says about the step) and `split` (number and seed of the splits).
    """
    for path, option in ((args.episodes, "--episodes"), (args.tokens, "--tokens")):
        if not path.is_file():
            raise Refused(f"{option}: {path} is missing")
    adapter = train_sft_script.adapter_entries(args.adapter)
    if None in adapter.values():
        raise Refused(f"{args.adapter} is not a saved adapter: {sorted(adapter)} must be in it")
    if args.splits < 1:
        raise Refused(f"--splits must be at least 1, not {args.splits}")
    run_dir, step_number = run_of(args.episodes)
    run_manifest = None if run_dir is None else _read_json(run_dir / "run_manifest.json")
    revision = resolve_revision(args.model, args.revision)
    if run_manifest is not None and not args.allow_other_settings:
        found = (run_manifest.get("model"), run_manifest.get("revision"))
        if found != (args.model, revision):
            raise Refused(f"the run was made with (model, revision) {found}, not with "
                          f"{(args.model, revision)}; pass --allow-other-settings if that is meant")
    gcfg, sources = step_settings(args, (run_manifest or {}).get("grpo_config"))
    try:
        records = attach_tokens(read_records(args.episodes), read_records(args.tokens))
    except (ValueError, KeyError) as error:
        raise Refused(f"the two input files do not belong together: {error}") from error
    plan = step_plan(records, gcfg)
    if len(plan["episodes"][SFT_STYLE]) < 2:
        raise Refused("the step has fewer than 2 valid episodes: there is nothing to compare")
    settings = {field: getattr(gcfg, field) for field in (*STEP_OPTIONS.values(), "gradient_checkpointing")}
    step = describe_step(records, plan, gcfg)
    identity = {"model": args.model, "revision": revision,
                "episodes_sha256": file_entry(args.episodes)["sha256"],
                "tokens_sha256": file_entry(args.tokens)["sha256"],
                "adapter_sha256": {name: entry["sha256"] for name, entry in adapter.items()},
                "settings": settings, "splits": args.splits, "split_seed": args.split_seed, "step": step}
    run = {"folder": None if run_dir is None else str(run_dir), "step": step_number,
           "manifest": None if run_manifest is None else file_entry(run_dir / "run_manifest.json"),
           "logged": logged_step(run_dir, step_number),
           "adapter_holds_the_weights_of_the_step": weights_of_the_step(step_number, adapter, run_manifest)}
    return {"records": records, "plan": plan, "gcfg": gcfg, "identity": identity, "step": step,
            "sources": sources, "run": run,
            "inputs": {"episodes": file_entry(args.episodes), "tokens": file_entry(args.tokens),
                       "adapter": {"folder": str(args.adapter), "files": adapter}}}


# ---------------------------------------------------------------------------
# One GPU stage. `measure_stage` needs no torch: it asks a backend for gradients and log-probabilities.
# ---------------------------------------------------------------------------
def array_name(loss: str, way: str, kind: str) -> str:
    """The name of an array in the .npz file of a stage; `kind` is "gradient" or "logprobs"."""
    return f"{loss}__{way}__{kind}"


def micro_batch_of(way: str, gcfg: GRPOConfig) -> int:
    """Episodes per forward pass of a way: the run's micro-batch in the trainer's layout, else 1."""
    return gcfg.micro_batch if way == BF16_TRAINER else 1


def sampler_logprobs(records: list[dict], episodes: list[int]) -> np.ndarray:
    """The log-probabilities that the sampler recorded for the sampled tokens of these episodes, in order."""
    return np.asarray([value for index in episodes for value in records[index]["tokens"]["logprobs"]],
                      dtype=np.float64)


def half_episodes(loss: str, half: list[int], prepared: dict) -> list[int]:
    """The episodes of one half, in record order. `half` holds positions into the units of the loss: the
    kept groups for "grpo" (a half then holds the valid episodes of its groups), the valid episodes for
    "sft_style"."""
    plan = prepared["plan"]
    if loss == SFT_STYLE:
        return [plan["episodes"][SFT_STYLE][position] for position in half]
    groups = {plan["kept_groups"][position] for position in half}
    return [index for index in plan["episodes"][GRPO] if index // prepared["gcfg"].group_size in groups]


def measure_halves(backend: Any, loss: str, prepared: dict, sets: dict, n_splits: int, seed: int,
                   kept: Sequence[dict] = (), save: Callable[[dict], None] | None = None) -> dict:
    """The yardstick of one loss: for each split, the gradient of each half in the trainer's layout, and
    from the two the cosine and the relative noise of the whole step, for every set of weights.

    `kept` are the first splits of this loss as an earlier start of the stage saved them, with the same
    seed and the same number of splits; they are not measured again. `save` is called with the result so
    far after every split that is measured, so that a failure in a later split loses no earlier one.

    Returns `unit` (what is split), `n_units`, `n_splits_asked`, `n_splits_planned` (fewer than asked if
    fewer exist), `n_splits` (those that are done), `seed`, `micro_batch`, `meaning` (HALVES_MEANING),
    `splits` and, per set of weights, `cosine_summary` and `relative_noise_summary` (`summary_of`). Per
    split: the units, the number of episodes and of loss tokens of each half, `share_of_the_whole`
    (`shares_of_the_whole`), `cosine` (of the two half gradients), `relative_noise` (for the GRPO loss
    `relative_noise` of the two halves; for the SFT-style loss `relative_noise_of_a_mean` of the two half
    means with their numbers of loss tokens) and the seconds. With fewer than two units there is no split
    and every summary is empty.
    """
    plan, gcfg = prepared["plan"], prepared["gcfg"]
    units = plan["kept_groups"] if loss == GRPO else plan["episodes"][SFT_STYLE]
    micro_batch = micro_batch_of(BF16_TRAINER, gcfg)
    planned = different_splits(len(units), n_splits, seed)
    splits = list(kept)[:len(planned)]

    def result() -> dict:
        return {"unit": "kept group" if loss == GRPO else "valid episode", "n_units": len(units),
                "n_splits_asked": n_splits, "n_splits_planned": len(planned), "n_splits": len(splits),
                "seed": seed, "micro_batch": micro_batch, "meaning": HALVES_MEANING[loss],
                "splits": splits,
                "cosine_summary": {label: summary_of([split["cosine"][label] for split in splits])
                                   for label in sets},
                "relative_noise_summary": {
                    label: summary_of([split["relative_noise"][label] for split in splits])
                    for label in sets}}

    for halves in planned[len(splits):]:
        started = time.perf_counter()
        episodes = [half_episodes(loss, half, prepared) for half in halves]
        measured = [backend.gradient(loss, part, micro_batch) for part in episodes]
        gradients = [half["gradient"] for half in measured]
        n_loss_tokens = [int(half["n_loss_tokens"]) for half in measured]
        shares = shares_of_the_whole(loss, n_loss_tokens)
        # Each half as its part of the gradient of the whole, in float64.
        parts = [share * np.asarray(gradient, dtype=np.float64) for share, gradient in zip(shares, gradients)]
        splits.append({
            "half_a": [units[position] for position in halves[0]],
            "half_b": [units[position] for position in halves[1]],
            "n_episodes": [len(part) for part in episodes],
            "n_loss_tokens": n_loss_tokens,
            "share_of_the_whole": shares,
            "cosine": {label: cosine(take(gradients[0], ranges), take(gradients[1], ranges))
                       for label, ranges in sets.items()},
            "relative_noise": {
                label: (relative_noise(take(parts[0], ranges), take(parts[1], ranges)) if loss == GRPO
                        else relative_noise_of_a_mean(take(gradients[0], ranges), take(gradients[1], ranges),
                                                      n_loss_tokens[0], n_loss_tokens[1]))
                for label, ranges in sets.items()},
            "seconds": round(time.perf_counter() - started, 2)})
        if save is not None:
            save(result())
    return result()


def measure_way(backend: Any, loss: str, way: str, stage: str, episodes: list[int], recorded: np.ndarray,
                micro_batch: int) -> tuple[dict, dict]:
    """One way of one loss: (its numbers, its arrays).

    The arrays are the gradient and the log-probabilities, under the names of `array_name`. The numbers
    are those of `ModelBackend.gradient`, with the seconds and the peak GPU memory of the gradient pass
    and of the pass that computes the log-probabilities under names that say which pass; the norm of the
    gradient; how far the second pass is from the sampler; and `lora_fingerprint_after`, the fingerprint
    of the LoRA weights after the way. `recorded` are the sampler's log-probabilities of the same tokens.
    """
    gradient_pass = dict(backend.gradient(loss, episodes, micro_batch))
    gradient = gradient_pass.pop("gradient")
    logprob_pass = backend.logprobs(episodes, micro_batch)
    logprobs = np.asarray(logprob_pass["logprobs"], dtype=np.float64)
    if logprobs.shape != recorded.shape:
        raise RuntimeError(f"{way}, {loss}: {logprobs.size} log-probabilities for {recorded.size} "
                           "sampled tokens")
    numbers = {"dtype": stage, "micro_batch": micro_batch,
               **{key: value for key, value in gradient_pass.items()
                  if key not in ("seconds", "peak_gpu_memory_gb")},
               "seconds_gradient_pass": gradient_pass["seconds"],
               "seconds_logprob_pass": logprob_pass["seconds"],
               "peak_gpu_memory_gb_gradient_pass": gradient_pass["peak_gpu_memory_gb"],
               "peak_gpu_memory_gb_logprob_pass": logprob_pass["peak_gpu_memory_gb"],
               "gradient_norm": norm(gradient),
               "logprob_gap_to_sampler_second_pass": abs_stats((logprobs - recorded).tolist()),
               "lora_fingerprint_after": backend.fingerprint()}
    arrays = {array_name(loss, way, "gradient"): gradient, array_name(loss, way, "logprobs"): logprobs}
    return numbers, arrays


class StageFiles:
    """The files of one GPU stage in the output folder, written piece by piece while the stage runs.

    A piece is one way of one loss (`way_files`: its arrays, then its numbers) or the halves of one loss
    (`halves_file`, written again after every split). Each JSON file holds `stamp`: the inputs and
    settings of the stage (`identity` of `prepare`) and the code (`code_state`). A later start of the same
    stage keeps a piece only if its stamp is the stamp of that start, and for a way only if the .npz file
    is still the one that was written with it. `kept` lists the files that this start kept.
    """

    def __init__(self, out_dir: Path, stage: str, stamp: dict, start: dict) -> None:
        self.out_dir, self.stage = out_dir, stage
        self.stamp = json.loads(json.dumps(stamp, default=str))       # as a file holds it
        self.start = start                    # `command` and `started_utc` of this start, for every file
        self.kept: list[str] = []

    def _header(self, loss: str) -> dict:
        return {"script": SCRIPT, "stage": self.stage, "loss": loss, **self.start, "saved_utc": utc_now(),
                "stamp": self.stamp}

    def _not_kept(self, name: str) -> None:
        print(f"{self.stage}: {name} of an earlier start is not kept (it was saved from other inputs, "
              "settings or code, or it is not whole): it is measured again", flush=True)

    def saved_way(self, loss: str, way: str) -> dict | None:
        """The numbers of a way that an earlier start saved, if they can be kept; else None."""
        json_name, npz_name = way_files(self.stage, loss, way)
        saved = _read_json(self.out_dir / json_name)
        if saved is None:
            return None
        numbers = saved.get("numbers")
        whole = (saved.get("complete") is True and saved.get("stamp") == self.stamp
                 and isinstance(numbers, dict) and isinstance(numbers.get("arrays"), dict)
                 and (self.out_dir / npz_name).is_file()
                 and file_entry(self.out_dir / npz_name)["sha256"] == numbers["arrays"].get("sha256"))
        if not whole:
            self._not_kept(json_name)
            return None
        self.kept.append(json_name)
        print(f"{self.stage}: {loss}, {way}: kept from an earlier start ({json_name})", flush=True)
        return numbers

    def save_way(self, loss: str, way: str, numbers: dict, arrays: dict) -> dict:
        """Write the arrays of a way and then its numbers. Returns the numbers with `arrays`: the path,
        the SHA-256 and the size of the .npz file."""
        json_name, npz_name = way_files(self.stage, loss, way)
        np.savez(self.out_dir / npz_name, **arrays)
        numbers = {**numbers, "arrays": file_entry(self.out_dir / npz_name)}
        write_json_whole(self.out_dir / json_name, {"complete": True, **self._header(loss), "way": way,
                                                    "numbers": numbers})
        return numbers

    def saved_splits(self, loss: str) -> list[dict]:
        """The splits of the halves of a loss that an earlier start saved, if they can be kept; else []."""
        name = halves_file(self.stage, loss)
        saved = _read_json(self.out_dir / name)
        if saved is None:
            return []
        splits = (saved.get("halves") or {}).get("splits")
        if saved.get("stamp") != self.stamp or not isinstance(splits, list):
            self._not_kept(name)
            return []
        if splits:
            self.kept.append(name)
            print(f"{self.stage}: {loss}, halves: {len(splits)} splits kept from an earlier start ({name})",
                  flush=True)
        return splits

    def save_halves(self, loss: str, halves: dict) -> None:
        """Write the halves of a loss as they are now; `complete` says if every planned split is done."""
        write_json_whole(self.out_dir / halves_file(self.stage, loss), {
            "complete": halves["n_splits"] == halves["n_splits_planned"], **self._header(loss),
            "halves": halves})


def measure_stage(backend: Any, prepared: dict, stage: str, n_splits: int, seed: int,
                  files: StageFiles) -> dict:
    """Everything that one GPU stage measures. Every piece is written through `files` as soon as it is
    measured, and a piece that `files` holds from an earlier start is not measured again.

    `backend` gives `gradient(loss, episodes, micro_batch)`, `logprobs(episodes, micro_batch)`,
    `weights()`, `n_layers`, `fingerprint()` and `largest_peak()`: see `ModelBackend`. Returns the numbers
    of the stage: `ways` (per loss and way: `measure_way`, with `arrays`), `halves` (the bfloat16 stage
    only: `measure_halves`), `weights`, `lora_fingerprint` (before the first way and after the last
    piece; `unchanged` is true only if also every way saw the same weights) and `peak_gpu_memory_gb`
    (the largest of this start).
    """
    plan, gcfg = prepared["plan"], prepared["gcfg"]
    weights = [(name, int(size)) for name, size in backend.weights()]
    sets = weight_sets(weights, backend.n_layers)
    before = backend.fingerprint()
    ways: dict = {loss: {} for loss in LOSSES}
    halves: dict = {}
    for loss in LOSSES:
        episodes = plan["episodes"][loss]
        recorded = sampler_logprobs(prepared["records"], episodes)
        for way in WAYS_OF_STAGE[stage]:
            numbers = files.saved_way(loss, way)
            if numbers is None:
                numbers, arrays = measure_way(backend, loss, way, stage, episodes, recorded,
                                              micro_batch_of(way, gcfg))
                numbers = files.save_way(loss, way, numbers, arrays)
                del arrays                    # they are on disk; the next way needs the memory
                print(f"{stage}: {loss}, {way}: {numbers['n_episodes']} episodes in "
                      f"{numbers['seconds_gradient_pass']} s; saved", flush=True)
            ways[loss][way] = numbers
        if stage == "bfloat16":
            halves[loss] = measure_halves(
                backend, loss, prepared, sets, n_splits, seed, kept=files.saved_splits(loss),
                save=lambda result, loss=loss: files.save_halves(loss, result))
            print(f"{stage}: {loss}, halves: {halves[loss]['n_splits']} splits", flush=True)
    after = backend.fingerprint()
    seen = [after] + [of_way["lora_fingerprint_after"] for of_loss in ways.values()
                      for of_way in of_loss.values()]
    return {"ways": ways, "halves": halves,
            "weights": {"n_layers": backend.n_layers, "n_weights": sum(size for _, size in weights),
                        "n_tensors": len(weights), "names_sha256": names_sha256(weights),
                        "sets": {label: {"n_weights": n_in(ranges), "ranges": [list(r) for r in ranges]}
                                 for label, ranges in sets.items()}},
            "lora_fingerprint": {"before": before, "after": after,
                                 "unchanged": all(math.isclose(before, value, rel_tol=FINGERPRINT_TOLERANCE)
                                                  for value in seen)},
            "peak_gpu_memory_gb": backend.largest_peak()}


def names_sha256(weights: Sequence[tuple[str, int]]) -> str:
    """One hash for the names and sizes of the LoRA weights, in order: the SHA-256 of the lines
    "<name> <size>\\n". Two stages with the same hash put the same weight at the same place of the vector."""
    lines = "".join(f"{name} {size}\n" for name, size in weights)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


class ModelBackend:
    """The GPU side of one stage: the model with the adapter, and what `measure_stage` asks of it.

    NOT RUN YET (written without torch). Every call into the library is one that the trainer makes too:
    `load_model` with a trainable adapter and `assert_coverage` as scripts/gpu/train_grpo.py,
    `set_gradient_checkpointing` and train mode as `train_grpo`, `zero_gradients` and
    `accumulate_policy_gradient` as `grpo_step`, `sft_gradient` as `train_sft`. No optimiser is built.

    `dtype` "bfloat16": the model as the library loads it. "float32": the same model with every weight
    turned into float32 by `model.float()`, and TF32 switched off as far as torch's two old switches go
    (`switch_tf32_off`), so that torch does not round float32 products to about 10 binary digits. A
    switch that cannot be set is kept as its error text in `tf32_switched_off`, and the stage goes on.
    """

    def __init__(self, args: argparse.Namespace, dtype: str, prepared: dict) -> None:
        import torch

        from accessdesk.harness.hf_backend import causal_lm, load_model
        from accessdesk.train.forward import set_gradient_checkpointing
        from accessdesk.train.lora import assert_coverage, coverage_table, trainable_parameters

        self.torch = torch
        self.dtype = dtype
        self.records = prepared["records"]
        self.plan = prepared["plan"]
        self.gcfg = prepared["gcfg"]
        # Per switch: "set to False", or the error text. Empty in the bfloat16 stage: nothing is set there.
        self.tf32_switched_off: dict = switch_tf32_off(torch) if dtype == "float32" else {}
        # Call -> error text, for a call that only serves the seconds or the peak memory (`_quietly`).
        self.timing_and_memory_errors: dict[str, str] = {}
        model, tokenizer = load_model(args.model, args.revision, adapter_dir=str(args.adapter),
                                      adapter_trainable=True, device=args.device)
        if dtype == "float32":
            model.float()                            # every weight as float32; the values do not change
        self.model = model
        self.core = causal_lm(model)
        assert_coverage(coverage_table(model), self.core.config)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise RuntimeError("the tokenizer has no pad token")
        # The LoRA weights in model order: the order of the gradient vector. `trainable_parameters` raises
        # if a trainable weight is not a float32 LoRA weight of a decoder layer.
        self.named = [(name, param) for name, param in self.core.named_parameters() if param.requires_grad]
        checked = trainable_parameters(model)
        if len(checked) != len(self.named) or any(a is not b for a, (_, b) in zip(checked, self.named)):
            raise RuntimeError("the trainable weights are not the ones that the library lists")
        self.n_layers = len(self.core.model.layers)
        self.base_dtypes: dict[str, int] = {}        # dtype -> number of weights that are not trained
        for param in self.core.parameters():
            if not param.requires_grad:
                key = str(param.dtype)
                self.base_dtypes[key] = self.base_dtypes.get(key, 0) + param.numel()
        if dtype == "float32" and set(self.base_dtypes) != {"torch.float32"}:
            raise RuntimeError(f"the float32 model holds weights of {sorted(self.base_dtypes)}")
        set_gradient_checkpointing(model, self.gcfg.gradient_checkpointing)
        model.train()
        self.peaks = [self._peak()]                  # the first entry is the peak of loading

    # -- what `measure_stage` asks for --------------------------------------------------------------
    def weights(self) -> list[tuple[str, int]]:
        """(name, number of entries) of every LoRA weight, in the order of the gradient vector."""
        return [(name, param.numel()) for name, param in self.named]

    def fingerprint(self) -> float:
        """The library's `lora_fingerprint`: one number for the LoRA weights as they are now."""
        from accessdesk.train.lora import lora_fingerprint

        return lora_fingerprint(self.model)

    def gradient(self, loss: str, episodes: list[int], micro_batch: int) -> dict:
        """The gradient of `loss` on the episodes `episodes` (indices into the records), in micro-batches
        of `micro_batch` in the given order. Keys: `gradient` (a float32 numpy vector over all LoRA
        weights: the values of `.grad`), `loss`, `n_episodes`, `n_loss_tokens`, `logprob_gap_to_sampler`
        (the library's `logprob_gap`; None for the SFT-style loss), `seconds` and `peak_gpu_memory_gb`.
        The gradients are set to zero first, so nothing of an earlier call is left in them.
        """
        from accessdesk.train.grpo import accumulate_policy_gradient, zero_gradients
        from accessdesk.train.sft import sft_gradient

        zero_gradients(self.model)
        self._reset_peak()
        started = time.perf_counter()
        if loss == GRPO:
            numbers = accumulate_policy_gradient(
                self.model, self.records, self.plan["advantages"], episodes,
                norm_const=self.gcfg.norm_const, pad_id=self.pad_id, micro_batch=micro_batch)
            gap = numbers["logprob_gap"]
        else:
            batch = [self.plan["examples"][index] for index in episodes]
            numbers = sft_gradient(self.model, batch, pad_id=self.pad_id, micro_batch=micro_batch)
            gap = None
        self._synchronize()
        seconds = time.perf_counter() - started
        self.peaks.append(self._peak())
        return {"gradient": self._read_gradient(), "loss": float(numbers["loss"]),
                "n_episodes": int(numbers["n_episodes"]), "n_loss_tokens": int(numbers["n_tokens"]),
                "logprob_gap_to_sampler": gap, "seconds": round(seconds, 2),
                "peak_gpu_memory_gb": self.peaks[-1]}

    def logprobs(self, episodes: list[int], micro_batch: int) -> dict:
        """The log-probability of every sampled token of the episodes: the library's `token_logprobs`
        without gradients, in micro-batches of `micro_batch`. Keys: `logprobs` (a float64 numpy vector, in
        the order of the tokens), `seconds` and `peak_gpu_memory_gb` of this pass (None without a CUDA
        device). The model is in eval mode for this call, as in the parity tests of
        tests/gpu/test_forward.py, and in train mode after it.
        """
        from accessdesk.train.forward import token_logprobs

        if not episodes:
            return {"logprobs": np.zeros(0, dtype=np.float64), "seconds": 0.0, "peak_gpu_memory_gb": None}
        self._reset_peak()
        started = time.perf_counter()
        self.model.eval()
        try:
            per_episode = token_logprobs(
                self.model, [self.records[index]["tokens"]["ids"] for index in episodes],
                [self.plan["weights"][index] for index in episodes],
                pad_id=self.pad_id, micro_batch=micro_batch, with_grad=False)
        finally:
            self.model.train()
        self._synchronize()
        seconds = time.perf_counter() - started
        self.peaks.append(self._peak())
        logprobs = np.concatenate([episode.detach().double().cpu().numpy() for episode in per_episode])
        return {"logprobs": logprobs, "seconds": round(seconds, 2), "peak_gpu_memory_gb": self.peaks[-1]}

    def largest_peak(self) -> dict | None:
        """The largest GPU memory of the process over loading, every gradient call and every
        log-probability call, in GiB: `allocated` and `reserved`, as scripts/gpu/rollout.py reports them.
        None without a CUDA device."""
        peaks = [peak for peak in self.peaks if peak is not None]
        if not peaks:
            return None
        return {key: max(peak[key] for peak in peaks) for key in ("allocated", "reserved")}

    def describe(self) -> dict:
        """For the stage's file, after everything is measured and saved: the dtypes of the weights that
        are not trained (dtype -> number of weights), the pad token, the TF32 switches (what this stage
        set, and what all of them say now), the library's environment and kernel reports, and what the
        source files say about TF32 in the Triton kernels (`kernel_sources_report`). Every read is by
        itself in `read_or_error`: one that fails is written as its error text and stops nothing."""
        from accessdesk.harness.hf_backend import environment_report, kernel_report

        def kernels() -> dict:
            self.model.eval()
            return kernel_report(self.model)

        return {"base_weight_dtypes": self.base_dtypes, "pad_id": self.pad_id,
                "timing_and_memory_errors": self.timing_and_memory_errors,
                "tf32": {"switched_off_by_this_stage": self.tf32_switched_off,
                         "switches": read_or_error(lambda: tf32_switches(self.torch))},
                "environment": read_or_error(environment_report),
                "kernels": read_or_error(kernels),
                "kernel_sources": read_or_error(kernel_sources_report)}

    # -- small helpers ------------------------------------------------------------------------------
    def _read_gradient(self) -> np.ndarray:
        missing = [name for name, param in self.named if param.grad is None]
        if missing:
            raise RuntimeError(f"{len(missing)} LoRA weights have no gradient, for example {missing[0]!r}")
        vector = np.concatenate([param.grad.detach().float().flatten().cpu().numpy()
                                 for _, param in self.named])
        if not np.isfinite(vector).all():
            raise FloatingPointError("the gradient holds a value that is not a finite number")
        return vector

    def _synchronize(self) -> None:
        """Wait for the GPU, so that the seconds of a pass cover all of it."""
        def synchronize() -> None:
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()

        self._quietly("torch.cuda.synchronize", synchronize)

    def _reset_peak(self) -> None:
        """Start the peak GPU memory anew, so that the next reading covers one pass only."""
        def reset() -> None:
            if self.torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()

        self._quietly("torch.cuda.reset_peak_memory_stats", reset)

    def _quietly(self, name: str, call: Callable[[], None]) -> None:
        """Make a call that only serves the seconds or the peak memory of a pass. If it raises, its error
        text is kept in `timing_and_memory_errors` (written into the stage's file) and the measurement
        goes on: such a call must not cost a gradient. The seconds or the peak memory are then less exact."""
        result = read_or_error(call)
        if isinstance(result, dict) and "error" in result:
            self.timing_and_memory_errors[name] = result["error"]

    def _peak(self) -> dict | None:
        return train_sft_script.peak_gpu_memory_gb()


def open_backend(args: argparse.Namespace, dtype: str, prepared: dict) -> Any:
    """The backend of a GPU stage. From here on the GPU is used. (The tests put a stand-in here.)"""
    return ModelBackend(args, dtype, prepared)


def code_state() -> dict:
    """Which code made the numbers: the hash of the frozen code and the state of the freeze, the hashes
    of the training code (both as the training scripts record them), and the hash of this script."""
    frozen = frozen_code()
    warn_if_not_as_frozen(frozen)
    return {**frozen, "training_code": training_code(), "impl": impl.status(),
            "this_script": file_entry(ROOT / SCRIPT)}


def refuse_used_folder(out_dir: Path, names: Sequence[str], hint: str = "") -> None:
    held = [name for name in names if (out_dir / name).exists()]
    if held:
        raise Refused(f"{out_dir} already holds {', '.join(held)}; choose a new output folder{hint}")


def gpu_stage(args: argparse.Namespace, prepared: dict, command: list[str]) -> int:
    """One GPU stage: load the model, measure and save piece by piece, then describe the machine, then
    write the file of the stage.

    Every way and every split is on disk before the next one starts (`StageFiles`), and all of them
    before `describe` is called. `describe` only reads what describes the machine. If it fails, its
    error text is written as `description_error`, and the stage is complete all the same. The file of the
    stage is written last and holds `"complete": true`, so a stage that failed leaves no such file; what
    it saved is kept by the next start of the same stage (`kept_from_an_earlier_start`).
    """
    stage = args.stage
    json_name = stage_file(stage)
    refuse_used_folder(args.out, (json_name, RESULT_NAME))
    started_utc, started = utc_now(), time.perf_counter()
    code = code_state()
    args.out.mkdir(parents=True, exist_ok=True)
    files = StageFiles(args.out, stage, {"identity": prepared["identity"], "code": code},
                       {"command": command, "started_utc": started_utc})
    backend = open_backend(args, stage, prepared)             # from here on the GPU is used
    numbers = measure_stage(backend, prepared, stage, args.splits, args.split_seed, files)
    # Everything that was measured is on disk now. What follows cannot lose it.
    try:
        description = {"description_error": None, **backend.describe()}
    except Exception as error:
        description = {"description_error": f"{type(error).__name__}: {error}"[:ERROR_LENGTH]}
    arrays = {Path(way["arrays"]["path"]).name: way["arrays"]
              for of_loss in numbers["ways"].values() for way in of_loss.values()}
    reference = {"float32_reference": FLOAT32_REFERENCE} if stage == "float32" else {}
    write_json_whole(args.out / json_name, {
        "complete": True, "script": SCRIPT, "stage": stage, "dtype": stage, "command": command,
        "started_utc": started_utc, "ended_utc": utc_now(),
        "seconds": round(time.perf_counter() - started, 1),
        "kept_from_an_earlier_start": files.kept,
        "identity": prepared["identity"], **numbers, **reference, **description, "code": code,
        "arrays": arrays})
    if not numbers["lora_fingerprint"]["unchanged"]:
        print(f"{stage}: THE LORA WEIGHTS CHANGED during the measurement: {numbers['lora_fingerprint']}",
              file=sys.stderr)
        return 1
    print(f"{stage}: done in {round(time.perf_counter() - started, 1)} s; peak GPU memory "
          f"{numbers['peak_gpu_memory_gb']} GiB; wrote {args.out / json_name}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# The comparison of the two stages (no torch)
# ---------------------------------------------------------------------------
def read_stage(out_dir: Path, stage: str, identity: dict) -> tuple[dict, dict]:
    """(the numbers, the arrays) that a GPU stage saved. Refused if the file of the stage is missing or
    not complete, if it was run on other inputs or settings than this call (`identity`), or if the
    arrays of one of its ways are missing or are not the file that the stage wrote."""
    numbers = _read_json(out_dir / stage_file(stage))
    if numbers is None or numbers.get("complete") is not True:
        raise Refused(f"{out_dir} holds no complete stage {stage}: run --stage {stage} first")
    wanted = json.loads(json.dumps(identity))                 # as the stage's file holds it
    different = sorted(key for key in wanted if numbers["identity"].get(key) != wanted[key])
    if different:
        raise Refused(f"the stage {stage} in {out_dir} was run with another {', '.join(different)} than "
                      "this call")
    arrays = {}
    for loss in LOSSES:
        for way in WAYS_OF_STAGE[stage]:
            npz = out_dir / way_files(stage, loss, way)[1]
            written = numbers["ways"][loss][way]["arrays"]["sha256"]
            if not npz.is_file() or file_entry(npz)["sha256"] != written:
                raise Refused(f"{npz} is missing or is not the file that the stage {stage} wrote")
            with np.load(npz) as data:
                arrays.update({name: data[name] for name in data.files})
    return numbers, arrays


def against_the_log(run: dict, way: dict, step: dict) -> dict:
    """The trainer's own numbers of the step beside this script's numbers of bfloat16_trainer_layout for
    the GRPO loss. They are the same computation only if --adapter holds the weights of the step
    (`adapter_holds_the_weights_of_the_step`); `logged` is None without the run's log."""
    return {"adapter_holds_the_weights_of_the_step": run["adapter_holds_the_weights_of_the_step"],
            "logged": run["logged"],
            "this_script": {"loss": way["loss"], "grad_norm": way["gradient_norm"],
                            "kept_groups": step["n_kept_groups"], "n_trained_episodes": way["n_episodes"],
                            "n_loss_tokens": way["n_loss_tokens"], "norm_const": step["norm_const"],
                            "logprob_gap": way["logprob_gap_to_sampler"]}}


def build_result(prepared: dict, stages: dict[str, tuple[dict, dict]], adapter_after: dict) -> dict:
    """The content of layout_gradients.json, from what the two stages saved. `stages` maps a stage to its
    (numbers, arrays); `adapter_after` are the hashes of the adapter files now."""
    numbers = {stage: saved[0] for stage, saved in stages.items()}
    arrays = {name: array for _, saved in stages.items() for name, array in saved[1].items()}
    bf16, fp32 = numbers["bfloat16"], numbers["float32"]
    if bf16["weights"]["names_sha256"] != fp32["weights"]["names_sha256"]:
        raise Refused("the two stages do not have the same LoRA weights in the same order")
    sets = {label: entry["ranges"] for label, entry in bf16["weights"]["sets"].items()}
    ways = {loss: {**bf16["ways"][loss], **fp32["ways"][loss]} for loss in LOSSES}
    comparisons, differences = {}, {}
    limit = FLOAT32_REFERENCE["resolves_relative_differences_above"]
    for loss in LOSSES:
        names = [way for stage in GPU_STAGES for way in WAYS_OF_STAGE[stage]]
        comparisons[loss] = compare_gradients(
            {way: arrays[array_name(loss, way, "gradient")] for way in names}, sets)
        for pair in comparisons[loss]:                        # the limit, beside the float32 numbers
            if pair["reference"] == FP32_SINGLE:
                pair["reference_resolves_relative_differences_above"] = limit
        differences[loss] = logprob_differences(
            {way: arrays[array_name(loss, way, "logprobs")] for way in names})
    adapter_before = prepared["identity"]["adapter_sha256"]
    fingerprints = {stage: numbers[stage]["lora_fingerprint"] for stage in GPU_STAGES}
    checks = {
        "adapter_files_unchanged": adapter_after == adapter_before,
        "lora_fingerprint": fingerprints,
        "lora_weights_unchanged": all(entry["unchanged"] for entry in fingerprints.values()),
        "same_lora_weights_in_both_stages": math.isclose(
            fingerprints["bfloat16"]["before"], fingerprints["float32"]["before"],
            rel_tol=FINGERPRINT_TOLERANCE),
        "against_the_training_log": against_the_log(prepared["run"], ways[GRPO][BF16_TRAINER],
                                                    prepared["step"]),
    }
    stage_keys = ("dtype", "command", "started_utc", "ended_utc", "seconds", "kept_from_an_earlier_start",
                  "peak_gpu_memory_gb", "description_error", "timing_and_memory_errors",
                  "base_weight_dtypes", "tf32", "pad_id", "environment", "kernels", "kernel_sources",
                  "arrays", "code")
    return {
        "script": SCRIPT, "model": prepared["identity"]["model"],
        "revision": prepared["identity"]["revision"],
        "inputs": prepared["inputs"], "run": prepared["run"],
        "settings": prepared["identity"]["settings"], "settings_from": prepared["sources"],
        "step": prepared["step"], "weights": bf16["weights"],
        "ways": ways, "comparisons": comparisons, "float32_reference": FLOAT32_REFERENCE,
        "halves": bf16["halves"],
        "logprob_differences": differences, "table": result_table(comparisons, bf16["halves"]),
        "checks": checks,
        "stages": {stage: {key: numbers[stage].get(key) for key in stage_keys} for stage in GPU_STAGES},
    }


def compare_stage(args: argparse.Namespace, prepared: dict, command: list[str]) -> int:
    """Read what the two GPU stages saved, compare, write layout_gradients.json and print the table.
    Exit code 1 if the LoRA weights or the adapter files changed during the measurement."""
    refuse_used_folder(args.out, (RESULT_NAME,))
    stages = {stage: read_stage(args.out, stage, prepared["identity"]) for stage in GPU_STAGES}
    adapter_after = {name: None if entry is None else entry["sha256"]
                     for name, entry in train_sft_script.adapter_entries(args.adapter).items()}
    result = build_result(prepared, stages, adapter_after)
    checks = result["checks"]
    unchanged = checks["adapter_files_unchanged"] and checks["lora_weights_unchanged"]
    write_json_whole(args.out / RESULT_NAME, {
        "complete": unchanged, "command": command, "written_utc": utc_now(), **result, "code": code_state()})
    print(format_table(result["table"]))
    print("Hold relative difference against the relative noise of the whole step. "
          + FLOAT32_REFERENCE["meaning"])
    if not unchanged:
        print(f"THE ADAPTER CHANGED during the measurement: {checks}", file=sys.stderr)
        return 1
    print(f"wrote {args.out / RESULT_NAME}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def stage_command(argv: list[str], stage: str) -> list[str]:
    """The command that runs one stage of this script in a new Python process: the same arguments, with
    --stage at the end. The parser takes the last --stage, so one that is in `argv` already has no effect."""
    return [sys.executable, str(ROOT / SCRIPT), *argv, "--stage", stage]


def run_in_new_process(argv: list[str], stage: str) -> int:
    """Run one stage in a new process and return its exit code. When the process ends, all its GPU memory
    is given back."""
    return subprocess.run(stage_command(argv, stage), check=False).returncode


def run(args: argparse.Namespace, argv: list[str], command: list[str]) -> int:
    prepared = prepare(args)
    if args.stage == "compare":
        return compare_stage(args, prepared, command)
    if args.stage in GPU_STAGES:
        return gpu_stage(args, prepared, command)
    every_file = [stage_file(stage) for stage in GPU_STAGES] + [RESULT_NAME]
    refuse_used_folder(args.out, every_file, hint=", or run the stages that are missing one by one with "
                                                  "--stage")
    for stage in GPU_STAGES:
        code = run_in_new_process(argv, stage)
        if code != 0:
            print(f"the stage {stage} ended with exit code {code}: the stages after it were not run. "
                  f"The pieces that it saved are in {args.out}. The same command with --stage {stage} "
                  "keeps them and measures the rest; then run the stages after it", file=sys.stderr)
            return code
    return compare_stage(args, prepared, command)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    given = list(sys.argv[1:]) if argv is None else list(argv)
    try:
        return run(args, given, command_line(SCRIPT, argv))
    except Refused as refusal:
        print(str(refusal), file=sys.stderr)
        return 2
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
