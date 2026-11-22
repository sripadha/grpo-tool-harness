"""SFT on the model's own passes (rejection sampling): the data selection and the training loop.

The data. The untrained model is rolled out on the TRAIN scenarios (`scripts/gpu/rollout.py`). From those
episodes this module keeps the ones that passed without any penalty and without a refused call (a
well-formed call that the tool answered with an error, for example a call with a made-up id), at most one
per scenario. The rules are written out at `select_sft_records`. Only episodes of the train split in the
training styles can become training data: `select_sft_records` and `training_example` both refuse any other
record. A training example is the episode's STORED token ids with one weight per token; text is never
tokenised again. The weight is 1 on the tokens that the model sampled and 0 everywhere else
(`impl.get("build_loss_mask")`).

The loss of one optimiser step is the mean negative log-likelihood over the weight-1 tokens of its batch:

    loss = -(1 / N) * sum over the episodes of the batch, sum over their weight-1 tokens t, of log p(token t)

N is the number of weight-1 tokens of the batch. It is known before the forward pass, so the batch can be run
in small slices: each slice adds `-(sum of its log-probabilities) / N` to the loss and its gradient to the
gradients, and the sum over the slices is the loss above.

What is read from the library source:
  - `token_logprobs(with_grad=True)` takes one slice per call (accessdesk/train/forward.py); the caller runs
    `backward()` before the next slice.
  - Gradient checkpointing only acts on a model in train mode (transformers modeling_layers.py line 93), so
    the loop puts the model in train mode. The LoRA dropout is 0 and the model's own attention dropout is 0
    (config.json), so train mode changes nothing else.
torch itself is not on the machine where this file was written, so its source could not be read. The torch
calls used here are `torch.optim.AdamW(params, lr, betas, eps, weight_decay)` and
`torch.nn.utils.clip_grad_norm_(parameters, max_norm)`, which returns the norm BEFORE clipping. AdamW's own
default weight decay is 0.01, so `weight_decay=0.0` is always given.

The selection functions need no torch. torch is imported inside the functions that train.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from accessdesk import impl
from accessdesk.config import RunConfig
from accessdesk.env.generator import TRAIN_STYLES
from accessdesk.eval import metrics
from accessdesk.rng import Rng, derive_seed
from accessdesk.train.forward import set_gradient_checkpointing, token_logprobs
from accessdesk.train.lora import save_adapter, trainable_parameters
from accessdesk.types import OUTCOMES, Segment

TRAIN_SPLIT = "train"


@dataclass(frozen=True)
class SFTConfig:
    lr: float = 1e-4
    epochs: int = 2
    batch_episodes: int = 8            # episodes per optimiser step
    micro_batch: int = 2               # episodes per forward pass; the gradients of a step add up over these
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    # The order of the examples in each epoch (`epoch_order`). An SFT run has a second seed, which is not in
    # this config: the seed of the adapter's random start values, an argument of `add_lora`
    # (accessdesk/train/lora.py).
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data selection (no torch)
# ---------------------------------------------------------------------------
def attach_tokens(records: list[dict], token_rows: list[dict]) -> list[dict]:
    """The records of `episodes.jsonl` with the key `tokens` from the same line of `tokens.jsonl`.

    The two files are written in the same order. Raises ValueError if the lines do not belong together
    (scenario, sample number or seed differ) or if the numbers of lines differ.
    """
    if len(records) != len(token_rows):
        raise ValueError(f"{len(records)} episode records but {len(token_rows)} token rows")
    joined = []
    for index, (record, row) in enumerate(zip(records, token_rows)):
        for key in ("scenario_id", "sample", "seed"):
            if record[key] != row[key]:
                raise ValueError(f"line {index}: the episode and the token row differ in {key!r}")
        tokens = {key: row[key] for key in ("ids", "segments", "logprobs")}
        joined.append({**record, "tokens": tokens})
    return joined


def action_sequence(record: dict) -> str:
    """What the episode did, as one string: the (tool, arguments) of every turn, in order.

    Two episodes of one scenario with the same string made the same calls. A malformed turn is (None, None).
    """
    calls = [[turn["tool_key"], turn["args"]] for turn in record["turns"]]
    return json.dumps(calls, sort_keys=True, ensure_ascii=False)


def token_weights(tokens: dict, exclude_turns: frozenset[int] = frozenset()) -> list[int]:
    """One weight per stored token id: 1 on sampled tokens, else 0, from `impl.get("build_loss_mask")`.

    `tokens` is the token data of one episode: `ids` and `segments` ([kind, start, end, turn] each).
    """
    segments = [Segment(kind, start, end, turn) for kind, start, end, turn in tokens["segments"]]
    weights = impl.get("build_loss_mask")(len(tokens["ids"]), segments, exclude_turns)
    if len(weights) != len(tokens["ids"]):
        raise ValueError(f"{len(weights)} weights for {len(tokens['ids'])} tokens")
    return list(weights)


def refuse_other_data(records: list[dict]) -> None:
    """Raise ValueError unless every record is an episode of a training scenario: of the train split, and
    in one of the training styles (`accessdesk.env.generator.TRAIN_STYLES`).

    Nothing from dev, test or calibration may become training data, and neither may the held-out wording
    style. The generator gives train families the training styles only, so the second check fires only if
    a scenario file or a record was changed by hand.
    """
    wrong_split = sorted({record["split"] for record in records} - {TRAIN_SPLIT})
    if wrong_split:
        raise ValueError(f"SFT data may come from the train split only; found records of {wrong_split}")
    wrong_style = sorted({record["style"] for record in records} - set(TRAIN_STYLES))
    if wrong_style:
        raise ValueError(f"SFT data may come from the training styles {TRAIN_STYLES} only; found records "
                         f"of the styles {wrong_style}")


def training_example(record: dict) -> dict:
    """The SFT example of one episode record (with its `tokens`): the stored ids and their weights.

    A turn that was not a valid call is left out of the loss (weight 0 on its tokens). A clean pass has no
    such turn. Raises ValueError for a harness error, and if no token is left in the loss.

    It also raises ValueError for a record that is not of the train split or not in a training style
    (`refuse_other_data`). This is a second guard behind `select_sft_records`: every example passes through
    this function, whatever chose its record.
    """
    refuse_other_data([record])
    if record["harness_error"] is not None:
        raise ValueError(f"{record['scenario_id']} is a harness error and must not be trained on")
    malformed = frozenset(turn["turn"] for turn in record["turns"] if not turn["parse_ok"])
    weights = token_weights(record["tokens"], malformed)
    if sum(weights) == 0:
        raise ValueError(f"{record['scenario_id']}: no token is in the loss")
    return {"scenario_id": record["scenario_id"], "family_id": record["family_id"],
            "outcome": record["outcome"], "sample": record["sample"], "reward": record["reward"],
            "ids": list(record["tokens"]["ids"]), "weights": weights}


def select_sft_records(records: list[dict], cfg: RunConfig, *,
                       allow_penalised: bool = False) -> tuple[list[dict], dict]:
    """Choose the episodes that SFT trains on. Returns (the chosen records, a report).

    `records` are episode records of the TRAIN split with their `tokens` (see `attach_tokens`). Any other
    split raises ValueError: nothing from dev, test or calibration may become training data. So does a
    record in a wording style that is not a training style (`refuse_other_data`). All records must come from
    one set-up (rule set, tool set, prompt).

    The rules, in order:
      1. harness errors are dropped;
      2. kept are the clean passes: `passed` and reward equal to `cfg.pass_reward` (no penalty). With
         `allow_penalised=True` every pass is kept (its malformed turns get weight 0 in `training_example`);
      3. an attempt with a refused call is never chosen, also with `allow_penalised=True`. A refused call
         is a well-formed call that the tool answered with an error (`n_tool_errors` above 0), for example
         a call with a made-up id. It wrote nothing, so the final state and the reward do not show it, and
         the attempt can still pass with the full reward. A made-up id that the tool refused must not
         become training data, even when the final state is right;
      4. inside a scenario, episodes with the same action sequence are duplicates: the one with the lowest
         sample number stays;
      5. at most one episode per scenario: the highest reward, then the fewest turns, then the lowest sample
         number.
    The chosen records are in the order of their scenario ids, so the result does not depend on the order of
    the input.

    What rule 3 does not cover: an id that was guessed and happens to exist. The tool serves such a call, so
    `n_tool_errors` does not count it, and the attempt can be chosen. The monitor `id_not_from_a_result`
    (accessdesk/eval/monitors.py) finds such a call in a saved record. This function does not use it.

    The report counts every stage, and gives per outcome the number of families in the input, the number of
    families with at least one chosen episode, the number of chosen scenarios, and the pass rate of the input
    (family-weighted, from `metrics.pass_rate_by`). `n_with_refused_call` is the number of passing attempts
    that rule 2 kept and rule 3 removed, and `n_candidates` is what is left after rule 3.
    """
    refuse_other_data(records)
    setups = sorted({(record["ruleset"], record["toolset"], record["prompt"]) for record in records})
    if len(setups) > 1:
        raise ValueError(f"the records come from {len(setups)} different set-ups: {setups}")

    valid = metrics.valid_records(records)
    passes = [record for record in valid if record["passed"]]
    clean = [record for record in passes if record["reward"] == cfg.pass_reward]
    kept = passes if allow_penalised else clean
    candidates = [record for record in kept if record["n_tool_errors"] == 0]

    by_scenario: dict[str, dict[str, dict]] = {}       # scenario id -> action sequence -> record
    for record in sorted(candidates, key=lambda r: (r["scenario_id"], r["sample"])):
        by_scenario.setdefault(record["scenario_id"], {}).setdefault(action_sequence(record), record)
    chosen = [min(distinct.values(), key=lambda r: (-r["reward"], r["n_turns"], r["sample"]))
              for _, distinct in sorted(by_scenario.items())]

    report = {
        "n_records": len(records), "n_harness_errors": len(records) - len(valid),
        "n_passed": len(passes), "n_clean_passes": len(clean),
        "allow_penalised": allow_penalised, "n_with_refused_call": len(kept) - len(candidates),
        "n_candidates": len(candidates),
        "n_distinct": sum(len(distinct) for distinct in by_scenario.values()),
        "n_chosen": len(chosen),
        "n_chosen_with_penalty": sum(1 for record in chosen if record["reward"] != cfg.pass_reward),
        "n_scenarios": len({record["scenario_id"] for record in records}),
        "n_families": len({record["family_id"] for record in records}),
        "n_families_covered": len({record["family_id"] for record in chosen}),
        "by_outcome": _coverage_by_outcome(records, chosen),
        "pass_rate_by_outcome": metrics.pass_rate_by(records, "outcome") if valid else {},
    }
    return chosen, report


def _coverage_by_outcome(records: list[dict], chosen: list[dict]) -> dict[str, dict]:
    """Per outcome: families in the input, families with a chosen episode, chosen scenarios."""
    names = [o for o in OUTCOMES] + sorted({r["outcome"] for r in records} - set(OUTCOMES))
    coverage = {}
    for outcome in names:
        families = {r["family_id"] for r in records if r["outcome"] == outcome}
        picked = [r for r in chosen if r["outcome"] == outcome]
        if families:
            coverage[outcome] = {"families": len(families),
                                 "families_covered": len({r["family_id"] for r in picked}),
                                 "scenarios_chosen": len(picked)}
    return coverage


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_optimizer(model: Any, lr: float) -> Any:
    """AdamW over the LoRA weights only, without weight decay. SFT and GRPO both use this function."""
    import torch

    return torch.optim.AdamW(trainable_parameters(model), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0.0)


def clip_and_step(model: Any, optimizer: Any, max_grad_norm: float) -> float:
    """One optimiser step with the gradients that are in `.grad` now. Returns the gradient norm BEFORE
    clipping.

    The norm is taken over all LoRA weights together. If it is larger than `max_grad_norm`, every gradient is
    scaled down by the same factor so that the norm becomes `max_grad_norm`. The gradients are cleared after
    the step. If the norm is not a finite number, no step is made and FloatingPointError is raised.
    """
    import torch

    norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm))
    if not math.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"the gradient norm is {norm}; no optimiser step was made")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def sft_gradient(model: Any, batch: list[dict], *, pad_id: int, micro_batch: int) -> dict:
    """Add the gradient of the batch's loss to the `.grad` of the LoRA weights. No optimiser step.

    `batch` holds examples from `training_example`. The loss is the mean negative log-likelihood over all
    weight-1 tokens of the batch (see the module docstring). Returns `loss`, `n_tokens` and `n_episodes`.
    The model must be in the mode that the caller wants (train mode for gradient checkpointing).
    """
    n_tokens = sum(sum(example["weights"]) for example in batch)
    if n_tokens == 0:
        raise ValueError("the batch has no token in the loss")
    loss = 0.0
    for start in range(0, len(batch), micro_batch):
        part = batch[start:start + micro_batch]
        logprobs = token_logprobs(model, [example["ids"] for example in part],
                                  [example["weights"] for example in part],
                                  pad_id=pad_id, micro_batch=micro_batch, with_grad=True)
        part_loss = -sum(episode.sum() for episode in logprobs) / n_tokens    # this slice's share of the mean
        part_loss.backward()                                                  # frees the slice's graph
        loss += float(part_loss.detach())
    return {"loss": loss, "n_tokens": n_tokens, "n_episodes": len(batch)}


def epoch_order(n_examples: int, seed: int, epoch: int) -> list[int]:
    """The order of the examples in one epoch: a shuffle that depends only on the seed and the epoch."""
    return Rng(derive_seed(seed, "sft", epoch)).shuffle(list(range(n_examples)))


def train_sft(model: Any, examples: list[dict], cfg: SFTConfig, *, pad_id: int,
              out_dir: str | Path) -> list[dict]:
    """Train the LoRA adapter of `model` on `examples` and save it. Returns the log lines, one per step.

    `model` is the peft wrapper from `add_lora`. Each epoch visits every example once, in a seeded order, in
    batches of `cfg.batch_episodes` (the last batch of an epoch may be smaller). One batch is one optimiser
    step. Written into `out_dir`:
        sft_log.jsonl   one line per step: step, epoch, loss, n_tokens, n_episodes, grad_norm (before
                        clipping), lr, seconds
        adapter/        the adapter after the last step (there is no selection of a best step)
    The model is in eval mode when the function returns. Gradient checkpointing stays as `cfg` set it; it has
    no effect in eval mode.
    """
    if not examples:
        raise ValueError("no SFT example")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sft_log.jsonl"
    log_path.write_bytes(b"")
    optimizer = make_optimizer(model, cfg.lr)
    optimizer.zero_grad(set_to_none=True)
    set_gradient_checkpointing(model, cfg.gradient_checkpointing)
    logs: list[dict] = []
    model.train()
    try:
        for epoch in range(cfg.epochs):
            order = epoch_order(len(examples), cfg.seed, epoch)
            for start in range(0, len(order), cfg.batch_episodes):
                batch = [examples[i] for i in order[start:start + cfg.batch_episodes]]
                started = time.perf_counter()
                numbers = sft_gradient(model, batch, pad_id=pad_id, micro_batch=cfg.micro_batch)
                grad_norm = clip_and_step(model, optimizer, cfg.max_grad_norm)
                log = {"step": len(logs), "epoch": epoch, **numbers, "grad_norm": grad_norm, "lr": cfg.lr,
                       "seconds": round(time.perf_counter() - started, 3)}
                logs.append(log)
                with log_path.open("ab") as handle:
                    handle.write((json.dumps(log) + "\n").encode("utf-8"))
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
    save_adapter(model, out_dir / "adapter")
    return logs
