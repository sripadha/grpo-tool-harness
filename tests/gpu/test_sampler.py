"""Tests of the sampler: what a sampled token is, and that the recorded log-probabilities are exact.

The first tests use hand-made numbers and no model. The others use the real model on the GPU.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import subprocess
import sys
from pathlib import Path

import pytest

from accessdesk.config import RunConfig
from accessdesk.harness.hf_backend import (LogprobRecorder, Sampler, cut_at_stop, kernel_report, pad_batch,
                                           set_sampling_seed)

STOP = 7      # stand-ins for <|im_end|> and the pad token in the hand-made cases
PAD = 3


# ---------------------------------------------------------------------------
# Hand-made cases, no model
# ---------------------------------------------------------------------------
def test_collecting_this_folder_does_not_import_torch():
    """tests/gpu/conftest.py is imported when pytest collects, before any test of any folder runs. If it
    imported torch, a test elsewhere that checks `"torch" not in sys.modules` would fail on the rented
    machine only. The conftest is run here in a fresh interpreter, as collection does."""
    conftest = Path(__file__).resolve().parent / "conftest.py"
    code = ("import runpy, sys; g = runpy.run_path(sys.argv[1]); "
            "assert g['SKIP_REASON'] is None, g['SKIP_REASON']; assert 'torch' not in sys.modules")
    done = subprocess.run([sys.executable, "-c", code, str(conftest)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-500:]


def test_ids_end_at_the_first_stop_token_and_padding_is_dropped():
    turn = cut_at_stop([5, 9, STOP, PAD, PAD], [-0.1, -0.2, -0.3, -9.0, -9.0], STOP, max_new_tokens=5)
    assert turn.ids == [5, 9, STOP]
    assert turn.logprobs == [-0.1, -0.2, -0.3]
    assert turn.truncated is False


def test_a_second_stop_token_after_the_first_is_dropped_too():
    turn = cut_at_stop([STOP, 5, STOP, PAD], [-0.5, -1.0, -2.0, -9.0], STOP, max_new_tokens=4)
    assert (turn.ids, turn.logprobs, turn.truncated) == ([STOP], [-0.5], False)


def test_a_row_without_stop_token_is_truncated_and_has_exactly_the_cap():
    turn = cut_at_stop([5, 9, 1, 2, 4], [-0.1, -0.2, -0.3, -0.4, -0.5], STOP, max_new_tokens=5)
    assert turn.ids == [5, 9, 1, 2, 4] and len(turn.ids) == 5
    assert turn.logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5]
    assert turn.truncated is True


def test_a_pad_token_that_the_model_wrote_before_the_stop_token_is_kept():
    turn = cut_at_stop([5, PAD, STOP, PAD], [-0.1, -0.2, -0.3, -9.0], STOP, max_new_tokens=4)
    assert (turn.ids, turn.truncated) == ([5, PAD, STOP], False)


def test_a_stop_token_on_the_last_allowed_step_is_not_truncated():
    turn = cut_at_stop([5, 9, 1, 2, STOP], [-0.1, -0.2, -0.3, -0.4, -0.5], STOP, max_new_tokens=5)
    assert (len(turn.ids), turn.truncated) == (5, False)


def test_a_short_row_without_stop_token_is_an_error():
    with pytest.raises(RuntimeError):
        cut_at_stop([5, 9], [-0.1, -0.2], STOP, max_new_tokens=5)


def test_recorder_on_hand_made_scores():
    """Two rows, two steps. Scores [0, ln 3] are the probabilities [1/4, 3/4]; [0, 0, ...] is uniform."""
    import torch

    step0 = torch.tensor([[0.0, math.log(3.0), -math.inf], [0.0, 0.0, 0.0]])
    step1 = torch.tensor([[math.log(3.0), 0.0, -math.inf], [0.0, math.log(2.0), math.log(2.0)]])
    prompt = torch.tensor([[2, 2], [2, 2]])
    drawn0 = torch.tensor([[1], [2]])      # row 0 draws token 1 (p = 3/4), row 1 draws token 2 (p = 1/3)
    drawn1 = torch.tensor([[1], [0]])      # row 0 draws token 1 (p = 1/4), row 1 draws token 0 (p = 1/5)
    recorder = LogprobRecorder()
    assert recorder(prompt, step0) is step0                            # the scores go through unchanged
    assert recorder(torch.cat([prompt, drawn0], dim=1), step1) is step1
    logprobs = recorder.finish(torch.cat([drawn0, drawn1], dim=1))
    expected = torch.tensor([[math.log(3 / 4), math.log(1 / 4)], [math.log(1 / 3), math.log(1 / 5)]])
    assert logprobs.shape == (2, 2)
    assert torch.allclose(logprobs, expected, atol=1e-6)


def test_recorder_refuses_a_wrong_number_of_steps():
    import torch

    recorder = LogprobRecorder()
    with pytest.raises(RuntimeError):
        recorder.finish(torch.tensor([[1]]))                           # never called
    recorder = LogprobRecorder()
    recorder(torch.tensor([[2]]), torch.zeros(1, 3))
    with pytest.raises(RuntimeError):
        recorder.finish(torch.tensor([[1, 1]]))                        # one step recorded, two tokens given


def test_pad_batch_left_and_right():
    import torch

    ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=PAD, side="left")
    assert ids.tolist() == [[5, 6, 7], [PAD, PAD, 8]] and mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert ids.dtype == torch.long and mask.dtype == torch.long
    ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=PAD, side="right")
    assert ids.tolist() == [[5, 6, 7], [8, PAD, PAD]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    with pytest.raises(ValueError):
        pad_batch([[5], []], pad_id=PAD)
    with pytest.raises(ValueError):
        pad_batch([[5]], pad_id=PAD, side="middle")


# ---------------------------------------------------------------------------
# With the model
# ---------------------------------------------------------------------------
def _reference(sampler: Sampler, prompts: list[list[int]], seed: int):
    """The same generate call as `Sampler.sample`, without the recorder, with every step's scores kept.

    Returns (new tokens [rows, steps], log-probabilities from the final scores, the same from the raw logits).
    """
    import torch
    from transformers import LogitsProcessorList

    input_ids, attention_mask = pad_batch(prompts, sampler.pad_id, side="left")
    config = copy.deepcopy(sampler.generation_config)
    config.return_dict_in_generate = True
    config.output_scores = True
    config.output_logits = True
    set_sampling_seed(seed)
    device = sampler.core.device
    with torch.no_grad():
        out = sampler.core.generate(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
            generation_config=config, logits_processor=LogitsProcessorList(list(sampler.warpers)))
    new_tokens = out.sequences[:, input_ids.shape[1]:]

    def chosen(per_step):
        """log_softmax of every step's [rows, vocabulary] tensor, at the token that was drawn in that step."""
        columns = [torch.log_softmax(step.float(), dim=-1).gather(1, new_tokens[:, k:k + 1]).squeeze(1)
                   for k, step in enumerate(per_step)]
        return torch.stack(columns, dim=1)

    return new_tokens, chosen(out.scores), chosen(out.logits)


@pytest.mark.parametrize("settings", [
    {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
    {"temperature": 2.5, "top_k": 50, "top_p": 0.95},      # far from the model's own distribution
])
def test_recorded_logprobs_equal_the_scores_that_generate_samples_from(model, tokenizer, prompts, settings):
    cfg = dataclasses.replace(RunConfig(), max_new_tokens=48, **settings)
    sampler = Sampler(model, tokenizer, cfg)
    turns = sampler.sample(prompts, seed=11)
    new_tokens, from_scores, from_logits = _reference(sampler, prompts, seed=11)
    assert len(turns) == len(prompts)
    for row, turn in enumerate(turns):
        n = len(turn.ids)
        assert turn.ids == new_tokens[row, :n].tolist()                # same seed, same tokens
        assert len(turn.logprobs) == n
        assert (turn.ids[-1] == sampler.stop_id) != turn.truncated
        assert sampler.stop_id not in turn.ids[:-1]
        if turn.truncated:
            assert n == cfg.max_new_tokens
        recorded = new_tokens.new_tensor(turn.logprobs, dtype=from_scores.dtype)
        assert (recorded - from_scores[row, :n]).abs().max().item() < 1e-4
        assert all(value <= 1e-6 for value in turn.logprobs)           # a log-probability is never above 0
        if settings["temperature"] == 1.0:
            assert (recorded - from_logits[row, :n]).abs().max().item() < 1e-4
    if settings["temperature"] != 1.0:
        # The recorded numbers are those of the changed distribution, not of the raw logits: somewhere in
        # the batch the two differ clearly. (So a recorder placed before the filters would fail above.)
        gaps = [(from_scores[row, :len(turn.ids)] - from_logits[row, :len(turn.ids)]).abs().max().item()
                for row, turn in enumerate(turns)]
        assert max(gaps) > 1e-2


def test_same_seed_gives_the_same_tokens_and_another_seed_does_not(model, tokenizer, prompts):
    cfg = dataclasses.replace(RunConfig(), max_new_tokens=32, temperature=2.0)    # hot, so that seeds matter
    sampler = Sampler(model, tokenizer, cfg)
    first = sampler.sample(prompts, seed=5)
    again = sampler.sample(prompts, seed=5)
    other = sampler.sample(prompts, seed=6)
    assert [t.ids for t in first] == [t.ids for t in again]
    gap = max(abs(a - b) for x, y in zip(first, again) for a, b in zip(x.logprobs, y.logprobs))
    print(f"\nsame seed twice: largest difference of a recorded log-probability = {gap:.2e}")
    assert gap < 1e-3
    assert [t.ids for t in first] != [t.ids for t in other]


def test_a_row_sampled_in_a_padded_batch_scores_the_same_alone(model, tokenizer, prompts):
    """Left padding must not change the model's view of a row: score the batch's tokens again, row by row."""
    from accessdesk.train.forward import token_logprobs

    sampler = Sampler(model, tokenizer, dataclasses.replace(RunConfig(), max_new_tokens=32))
    turns = sampler.sample(prompts, seed=3)
    shortest = min(range(len(prompts)), key=lambda i: len(prompts[i]))         # the row with the most padding
    seq = prompts[shortest] + turns[shortest].ids
    weights = [0] * len(prompts[shortest]) + [1] * len(turns[shortest].ids)
    alone = token_logprobs(model, [seq], [weights], pad_id=sampler.pad_id, micro_batch=1, with_grad=False)[0]
    recorded = alone.new_tensor(turns[shortest].logprobs)
    gaps = (alone - recorded).abs()
    print(f"\nsampled in a padded batch of {len(prompts)} against scored alone: n = {len(gaps)}, "
          f"{len(prompts[shortest])} prompt tokens, {max(map(len, prompts)) - len(prompts[shortest])} pads, "
          f"mean = {gaps.mean().item():.2e}, max = {gaps.max().item():.2e}")
    # Measured on 2026-11-17 (RTX 4090): 28 tokens after 1151 prompt tokens, 18 pads in the batch;
    # 2B: mean 3.1e-05, largest 7.4e-04; 4B: mean 1.8e-05, largest 4.3e-04.
    assert gaps.mean().item() < 0.02
    assert gaps.max().item() < 1.0


def test_sampler_restores_the_mode_of_the_model(model, tokenizer, prompts):
    sampler = Sampler(model, tokenizer, dataclasses.replace(RunConfig(), max_new_tokens=4))
    core = sampler.core
    try:
        core.train()
        sampler.sample(prompts[:1], seed=1)
        assert core.training is True
        core.eval()
        sampler.sample(prompts[:1], seed=1)
        assert core.training is False
    finally:
        core.eval()


def test_effective_generation_config_is_plain_sampling(model, tokenizer):
    effective = Sampler(model, tokenizer, RunConfig()).settings()["generate_effective_config"]
    assert effective["do_sample"] is True and effective["num_beams"] == 1
    assert (effective["temperature"], effective["top_k"], effective["top_p"]) == (1.0, 0, 1.0)
    assert effective["repetition_penalty"] == 1.0
    assert effective["eos_token_id"] == tokenizer.eos_token_id
    assert effective["pad_token_id"] == tokenizer.pad_token_id != tokenizer.eos_token_id
    assert effective["max_new_tokens"] == RunConfig().max_new_tokens


def test_sampler_refuses_a_hidden_setting_of_the_models_own_generation_config(model, tokenizer):
    core = Sampler(model, tokenizer, RunConfig()).core
    before = core.generation_config.suppress_tokens
    try:
        core.generation_config.suppress_tokens = [1]
        with pytest.raises(RuntimeError):
            Sampler(model, tokenizer, RunConfig())
    finally:
        core.generation_config.suppress_tokens = before
    Sampler(model, tokenizer, RunConfig())                             # and it starts again afterwards


def test_kernel_report_is_consistent_and_repeatable(model):
    report = kernel_report(model)
    again = kernel_report(model)
    assert report == again                       # the once-per-process warning is seen on every call
    assert report["fast_path"] == (not report["fallback_warnings"])
    fla_ok = report["imports"]["fla.ops.gated_delta_rule"]["ok"]
    conv_ok = report["imports"]["causal_conv1d"]["ok"]
    assert report["gated_delta_rule"] == ("fla" if fla_ok else "torch")
    assert report["causal_conv1d"] == ("causal_conv1d" if conv_ok else "torch")
    print("\nkernel report:", report)
