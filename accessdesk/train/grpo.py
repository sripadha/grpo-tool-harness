"""The GRPO step and the GRPO loop.

One step, in order:
  1. take `n_scenarios` training scenarios from a seeded order (`step_scenarios`);
  2. roll the model out `group_size` times on each of them, in eval mode, with the shared sampler. The
     attempts of one scenario are one GROUP;
  3. rewards / 100 go into an array [n_scenarios, group_size]; an attempt lost to a harness error is NaN;
  4. `impl.get("group_advantages")`: advantage = reward minus the mean of its group; a group whose rewards are
     all equal, or that lost more than half of its attempts, is DROPPED (all its advantages are 0);
  5. for the episodes of the kept groups: the log-probability of every sampled token under the current
     weights, with gradients (`token_logprobs`), in train mode;
  6. `impl.get("policy_loss")`, backward, gradient clipping, one optimiser step.

The loss, written out (A_i = advantage of episode i, a fixed number; logp = log-probability under the model):

    loss = -(1 / C) * sum over episodes i of ( A_i * sum over the sampled tokens t of episode i of logp(i,t) )

    C = n_scenarios * group_size * reference_tokens        (`GRPOConfig.norm_const`)

C is a constant of the run. It is computed from the config and from nothing else: not from the number of kept
groups, not from the number of episodes or tokens that went through the model. Two things follow.
  - A dropped group has A_i = 0 for all its episodes, so its terms in the sum are zero. Leaving its episodes
    out of step 5 gives exactly the same loss and the same gradient, and only saves compute.
  - The loss is a sum over episodes, so it can be computed in slices: each slice gives its part of the sum
    divided by the same C, and `backward()` on each part adds up to the gradient of the whole.
`tests/gpu/test_grpo_step.py` checks both on a hand-made batch.

If no group is kept, the gradient of the step is exactly zero. The optimiser step is still made, with that
zero gradient, because this is what "its terms are zero" means for the optimiser too (AdamW's running
averages go on; the weights can move a little from earlier steps' momentum).

The sampling policy and the trained policy are the same weights (one update per fresh batch), so there is no
importance ratio and no clipping of it. What remains is a small numerical gap between the log-probability
that the sampler recorded and the one that the training forward computes for the same token. It is measured
in step 5 on the kept groups, before the update, and logged (`logprob_gap`).

The files of a run, and what a restart does with them. Nothing that a start wrote is ever deleted.
    train_log.jsonl             one line per step, each step once. `start` in a line is the start that ran it
    rollouts/step_00012.*       the episodes of a step (and its token data with `save_tokens`)
    checkpoints/, latest.json   the adapter, the optimiser state and the run's totals; see `save_checkpoint`
    stopped.json                written when the stop rule ends a start: step, reason, start
    starts.jsonl                one line for every start and one for every fault; lines are only added
A restart (`resume=True`) goes on after the last checkpoint, so it runs the steps after that checkpoint
again. What the start before wrote for those steps is kept under the number of that start:
    train_log.start_1.jsonl     the log lines of those steps, the line with the fault among them
    rollouts/start_1/           their episode files
    stopped.start_1.json        the `stopped.json` of start 1
A run that the stop rule ended is not started again unless the caller says so
(`allow_resume_after_fault=True`); `starts.jsonl` then says that this was done. See `begin_start`.

The step functions that need no model (`step_scenarios`, `reward_matrix`, `plan_update`, `rollout_stats`,
`find_fault`, `abs_stats`) and the file steps of a start (`stop_of_last_start`, `begin_start`,
`record_step`) need no torch. torch is imported inside the functions that train.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from accessdesk import impl
from accessdesk.config import RunConfig
from accessdesk.env.generator import TRAIN_STYLES
from accessdesk.env.scenarios import scenario_ids, split_scenario_id
from accessdesk.eval import metrics
from accessdesk.harness.episode import Episode
from accessdesk.harness.hf_backend import Sampler, causal_lm, rollout
from accessdesk.harness.records import write_records
from accessdesk.harness.tokens import TokenLayout
from accessdesk.rng import Rng, derive_seed
from accessdesk.text import TextBundle
from accessdesk.train.forward import set_gradient_checkpointing, token_logprobs
from accessdesk.train.lora import lora_fingerprint, save_adapter, trainable_parameters
from accessdesk.train.sft import TRAIN_SPLIT, clip_and_step, make_optimizer, token_weights

REWARD_SCALE = 100            # a reward is a whole number of hundredths; the advantages use reward / 100
ARM = "grpo_train"            # the `arm` field of the episodes that a training step samples
LOG_NAME = "train_log.jsonl"
STARTS_NAME = "starts.jsonl"       # one line for every start of a run and for every fault; never rewritten
STOPPED_NAME = "stopped.json"      # written when the stop rule ends a start
SAMPLING_LABEL = "grpo_sampling"   # the label in the run seed of a GRPO run's rollouts (`sampling_run_seed`)
RUN_SEED_BITS = 53                 # that run seed is below 2**53: a double holds it exactly
# The fields of the config that a restarted run must share with the run that wrote the checkpoint.
SAME_ON_RESUME = ("n_scenarios", "group_size", "reference_tokens", "lr", "max_grad_norm", "tie_tol", "seed")
STATE_KEYS = ("completed_steps", "n_episodes", "n_harness_errors", "invalid_call_rate_step0")
# What is fixed at the first start and written into every checkpoint: the adapter that the run started from
# (`start_adapter`) and the scenario file of its families (`data_sha256`). See `train_grpo`.
ORIGIN_KEYS = ("start_adapter", "data_sha256")
_ROLLOUT_FILE = re.compile(r"step_(\d+)\.(?:episodes|tokens)\.jsonl")


class StoppedRunError(RuntimeError):
    """A restart was asked for of a run that the stop rule ended, without `allow_resume_after_fault`."""


@dataclass(frozen=True)
class GRPOConfig:
    n_scenarios: int = 16              # groups per step
    group_size: int = 8                # attempts per scenario
    # The token count in the constant C of the loss. It is a fixed constant of the run: it is set once,
    # before the run starts, and never from a batch. The maths is right for any fixed value. The value only
    # sets the scale of the gradient, and so how often the clip at `max_grad_norm` acts.
    # Where a value should come from: the TRAIN rollout that feeds SFT (about the number of sampled tokens
    # of one of its episodes). It is never taken from dev or test episodes: no training setting may depend
    # on them. The default is near the 165 sampled tokens that the scripted solution has on average on 40
    # train families with the placeholder text (counted when the loop was written).
    reference_tokens: int = 160
    steps: int = 80
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    micro_batch: int = 4               # episodes per training forward pass
    gradient_checkpointing: bool = True
    tie_tol: float = 0.0               # 0.0: only groups with exactly equal rewards are dropped
    save_every: int = 10               # steps between checkpoints
    seed: int = 0                      # scenario order (`step_scenarios`) and sampling (`sampling_run_seed`)
    # The stop rule (PLAN 5.6). A run is stopped only for a fault.
    max_harness_error_rate: float = 0.01      # harness errors / episodes, over the whole run so far
    max_logprob_gap: float = 0.02             # mean |sampler - trainer| log-probability of one step
    max_invalid_call_rate: float = 0.05       # and at least twice the rate of step 0
    usd_per_hour: float | None = None         # price of the machine, only to log the cost of each step

    @property
    def norm_const(self) -> float:
        """C of the loss: n_scenarios * group_size * reference_tokens. The same for every step of a run."""
        return float(self.n_scenarios * self.group_size * self.reference_tokens)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Which scenarios a step uses (no torch)
# ---------------------------------------------------------------------------
def step_scenarios(scenarios: list[str], n_scenarios: int, seed: int, step: int) -> list[str]:
    """The scenario ids of training step `step` (0, 1, 2, ...). It depends on its arguments only.

    The sorted ids are shuffled; the shuffled list is cut into consecutive pieces of `n_scenarios`; each step
    takes the next piece. When no full piece is left, one PASS is over: the list is shuffled again (with the
    pass number in the seed) and the next step takes the first piece of the new order. The ids at the end of
    a pass that do not fill a piece are not used in that pass (with 600 scenarios and 16 per step: 8 of them,
    other ones in each pass).

    So inside one pass no scenario is used twice, and one step never holds a scenario twice. Because the
    result is a function of (seed, step), a run that is restarted at step k goes on with exactly the scenarios
    that an uninterrupted run would have used: nothing is replayed and nothing is skipped.
    """
    ordered = sorted(set(scenarios))
    if len(ordered) != len(scenarios):
        raise ValueError("a scenario id is given twice")
    steps_per_pass = len(ordered) // n_scenarios if n_scenarios > 0 else 0
    if steps_per_pass == 0:
        raise ValueError(f"{len(ordered)} scenarios are not enough for steps of {n_scenarios}")
    if step < 0:
        raise ValueError(f"step must be 0 or more, not {step}")
    pass_index, position = divmod(step, steps_per_pass)
    shuffled = Rng(derive_seed(seed, "grpo_scenarios", pass_index)).shuffle(ordered)
    return shuffled[position * n_scenarios:(position + 1) * n_scenarios]


def sampling_run_seed(seed: int) -> int:
    """The run seed that the rollouts of a GRPO run with the seed `seed` give to the harness.

    The harness makes the torch seed of every generate call from (run seed, pass or step number, chunk, turn)
    and has no label for the kind of run. With the plain seed, step s of a GRPO run with seed S would draw
    from the same torch seeds as pass s of a rollout with run seed S (scripts/gpu/rollout.py). So the GRPO
    path has a run seed of its own: the lowest 53 bits of `derive_seed(seed, "grpo_sampling")`.

    It depends on `seed` only. So the sampling seeds of a step still depend only on (seed, step), and a
    restarted run draws what a run without a break would have drawn. The episode records of a training step
    hold this number as `seed["run_seed"]`, because the harness writes the run seed that it was given. The
    scenario order does not use it: `step_scenarios` takes the plain seed.

    Why 53 bits. `derive_seed` gives a 64-bit number, and the records hold the run seed. A number from 2**63
    on does not fit a signed 64-bit integer (numpy, pandas), and a reader that keeps numbers as doubles (jq,
    JavaScript) rounds whole numbers above 2**53. Below 2**53 every reader gets the number exactly.
    """
    return derive_seed(seed, SAMPLING_LABEL) % 2 ** RUN_SEED_BITS


def make_groups(step_ids: list[str], families_by_id: dict[str, dict], text: TextBundle, cfg: RunConfig,
                group_size: int) -> list[Episode]:
    """Fresh episodes for one step: `group_size` of the first scenario, then of the second, and so on.

    This order is what `reward_matrix` relies on: episode number i belongs to group i // group_size.
    """
    episodes: list[Episode] = []
    for scenario_id in step_ids:
        family_id, style = split_scenario_id(scenario_id)
        episodes.extend(Episode(families_by_id[family_id], style, text, cfg) for _ in range(group_size))
    return episodes


# ---------------------------------------------------------------------------
# From the episodes of a step to advantages (no torch)
# ---------------------------------------------------------------------------
def reward_matrix(records: list[dict], n_scenarios: int, group_size: int) -> np.ndarray:
    """Rewards / 100 as [n_scenarios, group_size]; NaN where the episode is a harness error.

    `records` are in the order of `make_groups`. Raises ValueError if their number is wrong or if a row would
    hold episodes of two scenarios.
    """
    if len(records) != n_scenarios * group_size:
        raise ValueError(f"{len(records)} records for {n_scenarios} groups of {group_size}")
    rewards = np.full((n_scenarios, group_size), np.nan, dtype=float)
    for index, record in enumerate(records):
        group, position = divmod(index, group_size)
        if record["scenario_id"] != records[group * group_size]["scenario_id"]:
            raise ValueError(f"record {index} is of scenario {record['scenario_id']}, but its group is of "
                             f"{records[group * group_size]['scenario_id']}")
        if record["harness_error"] is None:
            rewards[group, position] = record["reward"] / REWARD_SCALE
    return rewards


def plan_update(records: list[dict], gcfg: GRPOConfig) -> dict:
    """What the update of one step trains on. Keys:

        rewards      [n_scenarios, group_size], reward / 100, NaN for a harness error
        advantages   the same shape, from `impl.get("group_advantages")`; 0 in dropped groups and for NaN
        keep         [n_scenarios] booleans: the groups that are not dropped
        selected     indices into `records` of the episodes that go through the training forward: the valid
                     episodes of the kept groups, in record order
    A harness error is never selected: its token data may be incomplete.
    """
    rewards = reward_matrix(records, gcfg.n_scenarios, gcfg.group_size)
    advantages, keep = impl.get("group_advantages")(rewards, gcfg.tie_tol)
    advantages = np.asarray(advantages, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    if advantages.shape != rewards.shape or keep.shape != (gcfg.n_scenarios,):
        raise ValueError(f"group_advantages returned shapes {advantages.shape} and {keep.shape}")
    if not np.isfinite(advantages).all():
        raise ValueError("group_advantages returned a value that is not a finite number")
    selected = [index for index, record in enumerate(records)
                if keep[index // gcfg.group_size] and record["harness_error"] is None]
    return {"rewards": rewards, "advantages": advantages, "keep": keep, "selected": selected}


def abs_stats(differences: list[float]) -> dict | None:
    """n, mean, 99th percentile and maximum of the absolute values of a list; None for an empty list."""
    values = sorted(abs(float(x)) for x in differences)
    if not values:
        return None
    return {"n": len(values), "mean": sum(values) / len(values),
            "p99": values[min(len(values) - 1, int(0.99 * len(values)))], "max": values[-1]}


# ---------------------------------------------------------------------------
# The gradient of one step
# ---------------------------------------------------------------------------
def stack_logprobs(per_episode: list[Any]) -> tuple[Any, Any]:
    """1-D tensors of different lengths as one [episodes, longest] tensor, and its mask.

    Returns (logprobs, mask). Row i holds the values of `per_episode[i]` from the left and 0.0 after them;
    the mask is 1.0 on the values and 0.0 on the filling. The filling is only there to make a rectangle:
    with mask 0 it adds nothing to the loss. Gradients flow through `logprobs` to the inputs.
    """
    import torch

    width = max(len(values) for values in per_episode)
    padded = [torch.nn.functional.pad(values, (0, width - len(values))) for values in per_episode]
    logprobs = torch.stack(padded)
    mask = torch.zeros(logprobs.shape, dtype=logprobs.dtype, device=logprobs.device)
    for row, values in enumerate(per_episode):
        mask[row, :len(values)] = 1.0
    return logprobs, mask


def zero_gradients(model: Any) -> None:
    """Set the gradient of every LoRA weight to zeros (not to None), so that a step without any kept group
    still gives the optimiser a gradient: exactly zero."""
    import torch

    for param in trainable_parameters(model):
        param.grad = torch.zeros_like(param)


def accumulate_policy_gradient(model: Any, records: list[dict], advantages: Any, selected: list[int], *,
                               norm_const: float, pad_id: int, micro_batch: int) -> dict:
    """Add the gradient of the GRPO loss to the `.grad` of the LoRA weights. No optimiser step.

    records      the episodes of the step, each with its `tokens` (ids, segments, sampler log-probabilities)
    advantages   one number per record (any shape with len(records) entries, in record order)
    selected     indices of the records that are run through the model (`plan_update`); the others are taken
                 as zero terms, which is exact when their advantage is 0
    norm_const   C of the loss. It is used as given; nothing here counts episodes or tokens to change it.

    The selected episodes are run in slices of `micro_batch`. For each slice: `token_logprobs` with gradients
    on the stored token ids, with weight 1 on every sampled token; the slice's log-probabilities as
    [episodes, tokens] with a mask; `impl.get("policy_loss")(logprobs, mask, advantages, norm_const)`;
    `backward()`.

    Returns `loss` (the sum of the slices' losses), `n_episodes` and `n_tokens` that went through the model,
    and `logprob_gap`: n, mean, 99th percentile and maximum of |trainer - sampler| over those tokens (None if
    nothing was selected). The model's mode is not changed here.
    """
    import torch

    policy_loss = impl.get("policy_loss")
    flat = np.asarray(advantages, dtype=float).reshape(-1)
    if len(flat) != len(records):
        raise ValueError(f"{len(flat)} advantages for {len(records)} records")
    loss, n_tokens, differences = 0.0, 0, []
    for start in range(0, len(selected), micro_batch):
        part = selected[start:start + micro_batch]
        tokens = [records[index]["tokens"] for index in part]
        per_episode = token_logprobs(model, [t["ids"] for t in tokens], [token_weights(t) for t in tokens],
                                     pad_id=pad_id, micro_batch=micro_batch, with_grad=True)
        logprobs, mask = stack_logprobs(per_episode)                          # [episodes, tokens]
        part_advantages = torch.tensor([float(flat[index]) for index in part], dtype=logprobs.dtype,
                                       device=logprobs.device)               # [episodes], no gradient
        part_loss = policy_loss(logprobs, mask, part_advantages, norm_const)
        part_loss.backward()                                                  # frees the slice's graph
        loss += float(part_loss.detach())
        n_tokens += sum(len(values) for values in per_episode)
        for mine, t in zip(per_episode, tokens):
            if len(mine) != len(t["logprobs"]):
                raise ValueError(f"{len(mine)} trained tokens but {len(t['logprobs'])} sampler "
                                 "log-probabilities")
            differences.extend((mine.detach().cpu() - torch.tensor(t["logprobs"], dtype=mine.dtype)).tolist())
    if not math.isfinite(loss):
        raise FloatingPointError(f"the GRPO loss is {loss}")
    return {"loss": loss, "n_episodes": len(selected), "n_tokens": n_tokens,
            "logprob_gap": abs_stats(differences)}


# ---------------------------------------------------------------------------
# The numbers of one step (no torch)
# ---------------------------------------------------------------------------
def rollout_stats(records: list[dict]) -> dict:
    """What the episodes of one step look like. Harness errors are counted and are in no other number.

    Unit of each number:
        n_episodes, n_harness_errors             episodes
        mean_reward                              mean over valid episodes of reward / 100
        pass_rate, pass_rate_by_outcome          `metrics.pass_rate`, `metrics.pass_rate_by` (family-weighted)
        turns_per_episode, close_rate,
        share_at_turn_limit                      `metrics.turn_stats`
        calls_per_episode                        well-formed calls per valid episode
        tokens_per_turn, tokens_per_episode      sampled tokens, counted by the harness (`n_tokens`), of
                                                 ALL assistant messages: malformed and cut-off ones too.
                                                 (The calibration report's token numbers are another
                                                 quantity: tokens per well-formed call.)
        share_at_token_cap                       share of valid episodes with at least one turn cut off
        invalid_call_rate                        `metrics.invalid_call_rate` (malformed messages / messages)
        outcome_labels                           `metrics.outcome_labels` (shares that add to 1)
        zero_variance_exact                      share of scenarios whose valid attempts all have exactly the
                                                 same reward (`metrics.zero_variance_share`, equal_reward)
        zero_variance_pass_fail                  the same for pass or fail only (equal_passfail)
        mean_neg_logprob                         minus the mean sampler log-probability of the sampled tokens.
                                                 With temperature 1 and no filter this is an unbiased
                                                 estimate of the mean token entropy of the policy; it is not
                                                 the exact entropy, which needs the whole distribution.
    """
    valid = metrics.valid_records(records)
    stats: dict = {"n_episodes": len(records), "n_harness_errors": len(records) - len(valid)}
    if not valid:
        return stats
    turns = [turn for record in valid for turn in record["turns"]]
    logprobs = [value for record in valid for value in record["tokens"]["logprobs"]]
    turn_stats = metrics.turn_stats(records)
    zero = metrics.zero_variance_share(records)
    stats.update({
        "mean_reward": sum(record["reward"] for record in valid) / (REWARD_SCALE * len(valid)),
        "pass_rate": metrics.pass_rate(records),
        "pass_rate_by_outcome": metrics.pass_rate_by(records, "outcome"),
        "turns_per_episode": turn_stats["mean_turns"],
        "close_rate": turn_stats["share_closed"],
        "share_at_turn_limit": turn_stats["share_at_turn_limit"],
        "calls_per_episode": sum(1 for turn in turns if turn["parse_ok"]) / len(valid),
        "tokens_per_turn": sum(turn["n_tokens"] for turn in turns) / len(turns),
        "tokens_per_episode": sum(turn["n_tokens"] for turn in turns) / len(valid),
        "share_at_token_cap": sum(1 for record in valid
                                  if any(turn["truncated"] for turn in record["turns"])) / len(valid),
        "invalid_call_rate": metrics.invalid_call_rate(records),
        "outcome_labels": metrics.outcome_labels(records),
        "zero_variance_exact": zero["equal_reward"],
        "zero_variance_pass_fail": zero["equal_passfail"],
        "mean_neg_logprob": -sum(logprobs) / len(logprobs) if logprobs else None,
    })
    return stats


def find_fault(log: dict, state: dict, gcfg: GRPOConfig) -> str | None:
    """The reason to stop the run after this step, or None. The stop rule of PLAN 5.6, and nothing else.

    `log` is the step's log so far; `state` holds the totals of the run INCLUDING this step and the
    invalid-call rate of step 0. A rule whose number is not in `log` yet (the gap, before the training
    forward) is not checked. A loss or gradient that is not finite is caught where it happens (`grpo_step`).
    """
    error_rate = state["n_harness_errors"] / state["n_episodes"]
    if error_rate > gcfg.max_harness_error_rate:
        return (f"harness errors: {state['n_harness_errors']} of {state['n_episodes']} episodes so far "
                f"({error_rate:.4f}) are above {gcfg.max_harness_error_rate}")
    rate, first = log.get("invalid_call_rate"), state["invalid_call_rate_step0"]
    if rate is not None and first is not None and rate > gcfg.max_invalid_call_rate and rate >= 2 * first:
        return (f"invalid-call rate {rate:.4f} is above {gcfg.max_invalid_call_rate} and at least twice its "
                f"step-0 value {first:.4f}")
    gap = log.get("logprob_gap")
    if gap is not None and gap["mean"] > gcfg.max_logprob_gap:
        return f"mean log-probability gap {gap['mean']:.4f} is above {gcfg.max_logprob_gap}"
    return None


# ---------------------------------------------------------------------------
# One step
# ---------------------------------------------------------------------------
def grpo_step(model: Any, optimizer: Any, sampler: Sampler, layouts: dict, families_by_id: dict[str, dict],
              text: TextBundle, scenarios: list[str], step: int, gcfg: GRPOConfig, state: dict, *,
              run: str | None, data_sha256: str | None = None) -> tuple[dict, list[dict]]:
    """One GRPO step: rollout, advantages, gradient, one optimiser step. Returns (log line, episode records).

    `state` (the totals of the run) is updated with this step's episodes. If the stop rule fires, or a loss
    or gradient is not finite, `log["fault"]` says why, no optimiser step is made for this step, and the
    caller stops the run. The model must be in train mode; the sampler switches to eval mode for the rollout
    and back.

    The rollout gets the run seed `sampling_run_seed(gcfg.seed)` and the step number as its pass number.
    `data_sha256` (the SHA-256 of the scenario file of the families) goes into every episode record, as in
    the records of scripts/gpu/rollout.py; with None the records do not have that key.
    """
    started = time.perf_counter()
    step_ids = step_scenarios(scenarios, gcfg.n_scenarios, gcfg.seed, step)
    episodes = make_groups(step_ids, families_by_id, text, sampler.cfg, gcfg.group_size)
    meta = {"run": run, "arm": ARM}
    if data_sha256 is not None:
        meta["data_sha256"] = data_sha256
    records = rollout(sampler, layouts, episodes, run_seed=sampling_run_seed(gcfg.seed), pass_index=step,
                      meta=meta)
    rollout_seconds = time.perf_counter() - started

    plan = plan_update(records, gcfg)
    log: dict = {"step": step, "scenarios": step_ids, **rollout_stats(records),
                 "kept_groups": int(plan["keep"].sum()), "kept_share": float(plan["keep"].mean()),
                 "norm_const": gcfg.norm_const, "lr": gcfg.lr,
                 "loss": None, "n_trained_episodes": 0, "n_loss_tokens": 0, "logprob_gap": None,
                 "grad_norm": None}
    state["n_episodes"] += log["n_episodes"]
    state["n_harness_errors"] += log["n_harness_errors"]
    if step == 0:
        state["invalid_call_rate_step0"] = log.get("invalid_call_rate")

    fault = find_fault(log, state, gcfg)
    if fault is None:
        try:
            zero_gradients(model)
            numbers = accumulate_policy_gradient(
                model, records, plan["advantages"], plan["selected"], norm_const=gcfg.norm_const,
                pad_id=sampler.pad_id, micro_batch=gcfg.micro_batch)
            log.update({"loss": numbers["loss"], "n_trained_episodes": numbers["n_episodes"],
                        "n_loss_tokens": numbers["n_tokens"], "logprob_gap": numbers["logprob_gap"]})
            fault = find_fault(log, state, gcfg)
            if fault is None:
                log["grad_norm"] = clip_and_step(model, optimizer, gcfg.max_grad_norm)
        except FloatingPointError as error:
            fault = f"not finite: {error}"
        if fault is not None:
            optimizer.zero_grad(set_to_none=True)
    seconds = time.perf_counter() - started
    cost = None if gcfg.usd_per_hour is None else round(seconds / 3600 * gcfg.usd_per_hour, 4)
    log.update({"fault": fault, "rollout_seconds": round(rollout_seconds, 2), "seconds": round(seconds, 2),
                "cost_usd": cost})
    return log, records


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(model: Any, optimizer: Any, out_dir: Path, state: dict, gcfg: GRPOConfig,
                    run_cfg: RunConfig) -> Path:
    """Write the adapter, the optimiser state and the run's state after `state["completed_steps"]` steps.

    Folder: `out_dir/checkpoints/step_00010/` with `adapter/`, `optimizer.pt` and `state.json`. Then
    `out_dir/latest.json` is replaced in one move, so it always names a complete checkpoint.

    `state.json` also says where the run comes from (the keys of ORIGIN_KEYS, fixed at the first start) and
    which run seed its rollouts give to the harness (`sampling_run_seed`).
    """
    name = f"checkpoints/step_{state['completed_steps']:05d}"
    folder = out_dir / name
    save_adapter(model, folder / "adapter")
    _save_optimizer(optimizer, folder / "optimizer.pt")
    saved = {**{key: state[key] for key in STATE_KEYS}, "checkpoint": name,
             "lora_fingerprint": lora_fingerprint(model), "grpo_config": gcfg.to_dict(),
             "run_config_sha256": run_cfg.sha256(), "sampling_run_seed": sampling_run_seed(gcfg.seed),
             **{key: state.get(key) for key in ORIGIN_KEYS}}
    _write_json(folder / "state.json", saved)
    _write_json(out_dir / "latest.json.tmp", saved)
    os.replace(out_dir / "latest.json.tmp", out_dir / "latest.json")
    return folder


def latest_checkpoint(out_dir: str | Path) -> dict | None:
    """The content of `out_dir/latest.json` (see `save_checkpoint`), or None if the run has no checkpoint.

    `out_dir / result["checkpoint"] / "adapter"` is the adapter folder to give to `load_model`.
    """
    path = Path(out_dir) / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def resume_state(out_dir: Path, model: Any, optimizer: Any, gcfg: GRPOConfig, run_cfg: RunConfig) -> dict:
    """Load the optimiser state of the latest checkpoint and return the run's state at that point.

    The caller has already loaded that checkpoint's adapter into `model` (trainable). This is checked with
    the fingerprint of the LoRA weights. The config must be the one of the first start, except for `steps`
    and the fields that change no number of the training. The returned state holds the keys of STATE_KEYS
    and of ORIGIN_KEYS (None for an ORIGIN_KEYS key that the checkpoint does not have).
    """
    saved = latest_checkpoint(out_dir)
    if saved is None:
        raise FileNotFoundError(f"{out_dir} has no checkpoint to resume from")
    changed = [key for key in SAME_ON_RESUME if saved["grpo_config"][key] != gcfg.to_dict()[key]]
    if changed or saved["run_config_sha256"] != run_cfg.sha256():
        raise ValueError(f"the config differs from the one of the checkpoint: {changed or 'the run config'}")
    # A checkpoint of other code can name another run seed for the same seed (the formula of
    # `sampling_run_seed` changed on 2026-11-18). Going on from it would mix two sampling streams in one run.
    recorded = saved.get("sampling_run_seed")
    if recorded is not None and recorded != sampling_run_seed(gcfg.seed):
        raise ValueError(f"the checkpoint was sampled with the run seed {recorded}, but this code gives the "
                         f"seed {gcfg.seed} the run seed {sampling_run_seed(gcfg.seed)}: the run would go on "
                         "with other sampling seeds. Start the run again in a new folder")
    fingerprint = lora_fingerprint(model)
    if not math.isclose(fingerprint, saved["lora_fingerprint"], rel_tol=1e-6):
        raise RuntimeError(f"the model does not hold the adapter of {saved['checkpoint']} (fingerprint "
                           f"{fingerprint} against {saved['lora_fingerprint']}); load it first")
    _load_optimizer(optimizer, out_dir / saved["checkpoint"] / "optimizer.pt", model)
    return {**{key: saved[key] for key in STATE_KEYS}, **{key: saved.get(key) for key in ORIGIN_KEYS}}


def _save_optimizer(optimizer: Any, path: Path) -> None:
    """The optimiser state as one file (torch.save of its state dict)."""
    import torch

    torch.save(optimizer.state_dict(), path)


def _load_optimizer(optimizer: Any, path: Path, model: Any) -> None:
    """Read the file of `_save_optimizer` back into `optimizer`, onto the device of `model`."""
    import torch

    optimizer.load_state_dict(torch.load(path, map_location=causal_lm(model).device, weights_only=True))


def origin_problem(state: dict, start_adapter: dict | None, data_sha256: str | None) -> str | None:
    """Why a restart must not go on with these two arguments of `train_grpo`, or None if it may.

    `state` is the state of the last checkpoint. It holds what the first start recorded: the hashes of the
    files of the adapter that the run started from, and the hash of the scenario file. An argument that is
    None is not compared: the run keeps what the first start recorded. An argument that differs from the
    recorded value is a problem: the run would go on from another adapter, or on other scenarios.
    """
    recorded = (state.get("start_adapter") or {}).get("files_sha256")
    if start_adapter is not None and start_adapter != recorded:
        return (f"the start adapter differs from the one of the first start: its files have the hashes "
                f"{start_adapter}, and the first start recorded {recorded}")
    if data_sha256 is not None and data_sha256 != state.get("data_sha256"):
        return (f"the scenario file differs from the one of the first start: data_sha256 {data_sha256}, and "
                f"the first start recorded {state.get('data_sha256')}")
    return None


# ---------------------------------------------------------------------------
# The files of a start (no torch). Nothing is deleted here.
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """The time now in UTC, such as 2026-11-18T11:02:03Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: dict) -> None:
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _append_line(path: Path, obj: dict) -> None:
    with path.open("ab") as handle:
        handle.write((json.dumps(obj) + "\n").encode("utf-8"))


def _json_object(text: str) -> dict | None:
    """`text` as a dict; None if it is not one complete JSON object (a line that was cut off, for example)."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _json_lines(path: Path) -> list[tuple[str, dict | None]]:
    """(the text of the line, the line as a dict or None) for every line of a file that is not empty; an
    empty list if the file is not there."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return [(line, _json_object(line)) for line in lines if line]


def _rollout_files(folder: Path, first_step: int) -> list[Path]:
    """The episode and token files of the steps from `first_step` on in `rollouts/`, in name order."""
    found = []
    for path in sorted(folder.glob("step_*.jsonl")) if folder.is_dir() else []:
        match = _ROLLOUT_FILE.fullmatch(path.name)
        if match is not None and int(match.group(1)) >= first_step:
            found.append(path)
    return found


def _move(source: Path, target: Path) -> None:
    """Give `source` the name `target`. A file that is already there is never replaced."""
    if target.exists():
        raise FileExistsError(f"{target} is there already, so {source} was not moved. Nothing was deleted: "
                              "look at both files before the next start")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def read_starts(out_dir: str | Path) -> list[dict]:
    """The lines of `starts.jsonl` of the run in `out_dir`, in file order; an empty list without the file.

    A line has `event`, "start" or "fault", and `start`, the number of the start (1, 2, ...).
    A start line also has: `time_utc`, `resume`, `resumed_after_step`, `allow_resume_after_fault`,
    `resumed_after_fault` (the stop that this start went on after, or None), `moved` (what `begin_start`
    moved away) and the details that `train_grpo` adds. A fault line also has `step`, `reason`, `time_utc`.
    """
    return [event for _, event in _json_lines(Path(out_dir) / STARTS_NAME) if event is not None]


def stop_of_last_start(out_dir: str | Path) -> dict | None:
    """The stop that ended the last start of the run in `out_dir`, as `step`, `reason` and `start` (each
    None if it is not known). None if the stop rule did not end that start.

    Three places are read, and one of them is enough:
      - the last line of `starts.jsonl` is a fault: no start came after it;
      - `stopped.json` is there;
      - a line of `train_log.jsonl` holds a fault.
    An allowed restart moves `stopped.json` and that log line away under the number of the stopped start and
    adds a start line to `starts.jsonl` (`begin_start`). After it this function returns None, until the rule
    fires again.
    """
    out_dir = Path(out_dir)
    events = [event for event in read_starts(out_dir) if event.get("event") in ("start", "fault")]
    found: dict | None = None
    if events and events[-1]["event"] == "fault":
        found = events[-1]
    elif (out_dir / STOPPED_NAME).is_file():
        found = _json_object((out_dir / STOPPED_NAME).read_text(encoding="utf-8")) or {}
    else:
        faults = [log for _, log in _json_lines(out_dir / LOG_NAME) if log and log.get("fault") is not None]
        if faults:
            found = {"step": faults[-1].get("step"), "reason": faults[-1]["fault"],
                     "start": faults[-1].get("start")}
    if found is None:
        return None
    return {"step": found.get("step"), "reason": found.get("reason"), "start": found.get("start")}


def refuse_stopped_run(out_dir: str | Path, allow_resume_after_fault: bool) -> dict | None:
    """Raise StoppedRunError if the stop rule ended the last start of the run in `out_dir` and the caller
    did not allow a restart after a fault. Returns that stop (None if there was none). No file is touched."""
    stop = stop_of_last_start(out_dir)
    if stop is not None and not allow_resume_after_fault:
        raise StoppedRunError(
            f"the stop rule ended the run in {out_dir} at step {stop['step']}: {stop['reason']}. A stopped "
            "run is reported as stopped (PLAN 5.6), so it is not started again. To go on with it all the "
            "same, pass allow_resume_after_fault=True (scripts/gpu/train_grpo.py: "
            "--allow-resume-after-fault). The record of the stop is kept, and starts.jsonl will say that "
            "the run went on after it.")
    return stop


def refuse_used_folder(out_dir: str | Path) -> None:
    """Raise FileExistsError if `out_dir` holds files of a run: a new run must not write over them."""
    out_dir = Path(out_dir)
    held = [name for name in (LOG_NAME, STARTS_NAME, STOPPED_NAME) if (out_dir / name).exists()]
    held += [f"rollouts/{path.name}" for path in _rollout_files(out_dir / "rollouts", 0)[:1]]
    if held:
        raise FileExistsError(f"{out_dir} already holds {', '.join(held)}: choose a new folder, or pass "
                              "resume=True")


def begin_start(out_dir: str | Path, *, resume: bool, completed_steps: int = 0,
                allow_resume_after_fault: bool = False,
                details: dict | None = None) -> tuple[int, list[dict]]:
    """The file steps at the beginning of a start of the run in `out_dir`. Returns (the number of this
    start, the log lines that stay in `train_log.jsonl`). No torch. No file is deleted.

    A new run (`resume=False`) is start 1. Its folder must not hold a run (FileExistsError).

    A restart (`resume=True`) goes on after `completed_steps` steps: the steps of the last checkpoint.
      1. If the stop rule ended the start before, StoppedRunError is raised and no file is touched, unless
         `allow_resume_after_fault` is True (`refuse_stopped_run`).
      2. The steps from `completed_steps` on will be run again. What the start before (its number is n)
         wrote for them is moved away, under that number:
           - their lines of `train_log.jsonl` are added to `train_log.start_<n>.jsonl`, as they are. A line
             that is not complete JSON goes there too. `train_log.jsonl` keeps the lines of the steps
             before `completed_steps`, unchanged;
           - their files in `rollouts/` go to `rollouts/start_<n>/`;
           - `stopped.json` becomes `stopped.start_<n>.json`.
         A file that is already there under such a name is never replaced (FileExistsError).
      3. One start line is added to `starts.jsonl` (`read_starts`), with `details` in it.

    The number of a start is one more than the highest start number in `starts.jsonl`. A restart of a run
    that has no such file (a run of older code) is start 2: its earlier starts count as one.

    If the process dies in the middle of these steps, the next restart does the rest of them under the
    same number: the start line is written last. The log lines that were cut can then be in the kept file
    twice. Nothing is lost.
    """
    out_dir = Path(out_dir)
    stop = None
    if resume:
        stop = refuse_stopped_run(out_dir, allow_resume_after_fault)
    else:
        refuse_used_folder(out_dir)
    numbers = [event["start"] for event in read_starts(out_dir)
               if event.get("event") == "start" and isinstance(event.get("start"), int)]
    start = max(numbers) + 1 if numbers else (2 if resume else 1)
    before = start - 1
    (out_dir / "rollouts").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / LOG_NAME
    kept_lines, kept, cut_lines = [], [], []
    for line, log in _json_lines(log_path):
        if log is not None and isinstance(log.get("step"), int) and log["step"] < completed_steps:
            kept_lines.append(line)
            kept.append(log)
        else:
            cut_lines.append(line)
    moved: dict = {"log_lines": None, "n_log_lines": len(cut_lines), "rollout_files": None,
                   "n_rollout_files": 0, "stopped": None}
    if cut_lines:
        moved["log_lines"] = f"train_log.start_{before}.jsonl"
        with (out_dir / moved["log_lines"]).open("ab") as handle:
            handle.write("".join(line + "\n" for line in cut_lines).encode("utf-8"))
        temporary = out_dir / (LOG_NAME + ".tmp")
        temporary.write_bytes("".join(line + "\n" for line in kept_lines).encode("utf-8"))
        os.replace(temporary, log_path)              # in one move: the log is never half-written
    for path in _rollout_files(out_dir / "rollouts", completed_steps) if resume else []:
        moved["rollout_files"] = f"rollouts/start_{before}"
        _move(path, out_dir / moved["rollout_files"] / path.name)
        moved["n_rollout_files"] += 1
    if resume and (out_dir / STOPPED_NAME).is_file():
        writer = stop["start"] if stop is not None and isinstance(stop["start"], int) else before
        moved["stopped"] = f"stopped.start_{writer}.json"
        _move(out_dir / STOPPED_NAME, out_dir / moved["stopped"])
    _append_line(out_dir / STARTS_NAME, {
        "event": "start", "start": start, "time_utc": utc_now(), "resume": resume,
        "resumed_after_step": completed_steps if resume else None,
        "allow_resume_after_fault": allow_resume_after_fault, "resumed_after_fault": stop,
        "moved": moved, **(details or {})})
    return start, kept


def _save_rollout(out_dir: Path, step: int, records: list[dict], save_tokens: bool) -> None:
    """The episodes of one step: `rollouts/step_00012.episodes.jsonl`, and the token data if asked for
    (`rollouts/step_00012.tokens.jsonl`, the same lines as tokens.jsonl of scripts/gpu/rollout.py)."""
    base = out_dir / "rollouts" / f"step_{step:05d}"
    without_tokens = [{key: value for key, value in record.items() if key != "tokens"} for record in records]
    write_records(f"{base}.episodes.jsonl", without_tokens)
    if save_tokens:
        keys = ("scenario_id", "sample", "seed", "harness_error")
        write_records(f"{base}.tokens.jsonl",
                      [{**{key: record[key] for key in keys}, **record["tokens"]} for record in records])


def record_step(out_dir: Path, start: int, log: dict, records: list[dict], save_tokens: bool) -> bool:
    """Write what one step leaves on disk. Returns True if the step has a fault: the run must stop.

    In this order: the episodes of the step (`_save_rollout`); its log line, with `start` (the number of
    the start that ran the step) added to it; and, for a fault, `stopped.json` (step, reason, start) and a
    fault line in `starts.jsonl`.
    """
    log["start"] = start
    _save_rollout(out_dir, log["step"], records, save_tokens)
    _append_line(out_dir / LOG_NAME, log)
    if log["fault"] is None:
        return False
    stop = {"step": log["step"], "reason": log["fault"], "start": start}
    _write_json(out_dir / STOPPED_NAME, stop)
    _append_line(out_dir / STARTS_NAME, {"event": "fault", "time_utc": utc_now(), **stop})
    return True


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def training_scenarios(families: list[dict]) -> list[str]:
    """The scenario ids that GRPO trains on: every wording of every family, sorted. No torch.

    Two guards. Every family must be of the train split, and every wording must be in a training style
    (`accessdesk.env.generator.TRAIN_STYLES`): the held-out style is never trained on. Raises ValueError
    otherwise; a caller that wants fewer styles leaves the other wordings out of the families.
    """
    wrong_split = sorted({family["split"] for family in families} - {TRAIN_SPLIT})
    if wrong_split:
        raise ValueError(f"GRPO trains on the train split only; found families of {wrong_split}")
    scenarios = sorted(scenario_id for family in families for scenario_id in scenario_ids(family))
    other = [sid for sid in scenarios if split_scenario_id(sid)[1] not in TRAIN_STYLES]
    if other:
        styles = sorted({split_scenario_id(sid)[1] for sid in other})
        raise ValueError(f"GRPO trains on the training styles {TRAIN_STYLES} only; found {len(other)} "
                         f"scenarios of the styles {styles}, for example {other[0]}. Leave these wordings "
                         "out of the families.")
    return scenarios


def train_grpo(model: Any, tokenizer: Any, families: list[dict], text: TextBundle, run_cfg: RunConfig,
               gcfg: GRPOConfig, *, out_dir: str | Path, run: str | None = None, resume: bool = False,
               save_tokens: bool = False, allow_resume_after_fault: bool = False,
               data_sha256: str | None = None, start_adapter: dict | None = None) -> list[dict]:
    """Train the LoRA adapter of `model` with GRPO for `gcfg.steps` steps. Returns the log lines.

    model      the peft wrapper with trainable LoRA weights (from `add_lora`, or from `load_model` with
               `adapter_trainable=True`, for example the adapter that SFT saved)
    families   family records of the TRAIN split with wordings in the training styles only (anything else
               raises ValueError, see `training_scenarios`); every wording of every family is one training
               scenario
    run_cfg    the shared config: sampling settings, token cap, turn limit, parser, chunk size
    out_dir    receives `train_log.jsonl` (one line per step), `rollouts/` (the episodes of every step),
               `checkpoints/` and `latest.json` (every `gcfg.save_every` steps and at the end),
               `starts.jsonl` (one line per start and per fault), and `stopped.json` if the stop rule ended
               the run
    save_tokens   also write the token data of every step (`rollouts/step_*.tokens.jsonl`: ids, segments
               and sampler log-probabilities). Only with it can the logged `loss` and `logprob_gap` be
               computed again from the saved files
    data_sha256   the SHA-256 of the scenario file that `families` were loaded from. It goes into every
               episode record of the training, into the checkpoints and into `starts.jsonl`
    start_adapter   what the caller knows of the adapter that `model` holds at the first start: the SHA-256
               of each of its files, by file name. It is written, with the fingerprint of the LoRA weights
               before step 0 (`lora_fingerprint`), into every checkpoint and into `starts.jsonl`, so that
               two runs can be shown to start from one adapter

    Restart: load the adapter of `latest_checkpoint(out_dir)` into the model and call this function with
    `resume=True`. The run goes on after the last saved step with the saved optimiser state. The scenarios
    and the sampling seeds of a step depend only on (gcfg.seed, step), so they are the ones an uninterrupted
    run would have used. Steps that were done after the last checkpoint are done again; what the start
    before wrote for them is kept under other names (`begin_start`). `data_sha256` and `start_adapter` must
    be the ones of the first start, or None (`origin_problem`).

    A run that the stop rule ended is reported as stopped (PLAN 5.6). A restart of it raises
    StoppedRunError, before anything is loaded or written, unless `allow_resume_after_fault=True` is given.
    With it the run goes on from its last checkpoint; the record of the stop stays on disk, and
    `starts.jsonl` says that the run went on after a fault.

    The model is in eval mode when the function returns.
    """
    out_dir = Path(out_dir)
    scenarios = training_scenarios(families)
    families_by_id = {family["family_id"]: family for family in families}
    if resume:
        refuse_stopped_run(out_dir, allow_resume_after_fault)
    else:
        refuse_used_folder(out_dir)

    optimizer = make_optimizer(model, gcfg.lr)
    fingerprint = lora_fingerprint(model)                    # of the weights that this start begins with
    if resume:
        state = resume_state(out_dir, model, optimizer, gcfg, run_cfg)
        problem = origin_problem(state, start_adapter, data_sha256)
        if problem is not None:
            raise ValueError(problem)
    else:
        state = {"completed_steps": 0, "n_episodes": 0, "n_harness_errors": 0,
                 "invalid_call_rate_step0": None,
                 "start_adapter": {"lora_fingerprint": fingerprint, "files_sha256": start_adapter},
                 "data_sha256": data_sha256}
    start, logs = begin_start(
        out_dir, resume=resume, completed_steps=state["completed_steps"],
        allow_resume_after_fault=allow_resume_after_fault,
        details={"run": run, "seed": gcfg.seed, "sampling_run_seed": sampling_run_seed(gcfg.seed),
                 "steps": gcfg.steps, "save_tokens": save_tokens, "lora_fingerprint": fingerprint,
                 **{key: state[key] for key in ORIGIN_KEYS}})
    set_gradient_checkpointing(model, gcfg.gradient_checkpointing)
    sampler = Sampler(model, tokenizer, run_cfg)
    layouts = {"five": TokenLayout(tokenizer, text.tool_specs("five"))}
    model.train()
    try:
        for step in range(state["completed_steps"], gcfg.steps):
            log, records = grpo_step(model, optimizer, sampler, layouts, families_by_id, text, scenarios,
                                     step, gcfg, state, run=run, data_sha256=state["data_sha256"])
            logs.append(log)
            if record_step(out_dir, start, log, records, save_tokens):
                break
            state["completed_steps"] = step + 1
            if state["completed_steps"] % gcfg.save_every == 0 or state["completed_steps"] == gcfg.steps:
                save_checkpoint(model, optimizer, out_dir, state, gcfg, run_cfg)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
    return logs
