"""Tests of `accessdesk/train/forward.py`: which positions are read, padding, and parity with the sampler.

The first tests use hand-made numbers and a four-token toy model; they need torch but no GPU work. The others
use the real model.

About the tolerances of the tests with the real model (measured on 2026-11-17, RTX 4090, torch 2.15.0,
transformers 5.23.0, fla 0.6.0, causal-conv1d 1.7.0; the first run of this file on a GPU).
The model runs in bfloat16, in the sampler and in the trainer alike. In bfloat16 the result for one row
depends on the SHAPE of the batch it is in (how many rows, how wide), and on nothing else:
  - the same 400 prompt tokens alone and as the second row of a batch of two WITHOUT any pad token: the
    log-probabilities differ by 0.046 on average on the 2B and 0.108 on the 4B (99th percentile 0.50 and
    1.34, largest 1.9 and 5.7);
  - with 80 pad tokens on the left the numbers are the same (2B 0.047, 4B 0.103): padding adds nothing.
    The last hidden states of that row then differ from those of the row alone by about 3 % on average
    (relative L2 norm per position);
  - with the other row replaced by other tokens of the same length the difference is exactly 0;
  - the 2B in float32: 0.001 to 0.002 on average. So this is rounding, not a leak.
These differences are large on tokens that the model finds unlikely (prompt text) and small on tokens that the
model sampled itself, which are the ones training uses: sampler against trainer on whole episodes gives a mean
of 0.0017 (2B) and 0.0015 (4B). The tests print mean, 99th percentile and maximum (run pytest with -s). They
bound the mean and the 99th percentile, never the maximum of prompt tokens. A real padding mistake is far
outside the bounds: with the pad tokens left unmasked the mean is 0.82 (2B) and 1.81 (4B), the 99th
percentile 13 and 17.5.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest

from accessdesk.config import RunConfig
from accessdesk.harness.hf_backend import Sampler, causal_lm
from accessdesk.train.forward import (loss_positions, pad_batch, position_ids, set_gradient_checkpointing,
                                      token_logprobs)

# P[next][current] of the toy model: column c is the distribution of the token after token c.
TOY_P = [[0.1, 0.4, 0.25, 0.7],
         [0.2, 0.3, 0.25, 0.1],
         [0.3, 0.2, 0.25, 0.1],
         [0.4, 0.1, 0.25, 0.1]]
TOY_PAD = 3


# ---------------------------------------------------------------------------
# Hand-made cases
# ---------------------------------------------------------------------------
def test_loss_positions_hand_made():
    # Width 3. Row 1 is [pad, 8, 9]. Token 7 sits in column 2 and is predicted from column 1.
    # Token 9 sits in column 2 of its row and is predicted from column 1, where token 8 is.
    assert loss_positions([[5, 6, 7], [8, 9]], [[0, 0, 1], [0, 1]]) == ([0, 1], [1, 1], [7, 9], [1, 1])


def test_loss_positions_with_two_pads_and_several_tokens():
    # Width 4. Row 0 has no pad: tokens 1 and 3 are predicted from columns 0 and 2.
    # Row 1 is [pad, pad, 2, 0]: its token 1 (id 0) is in column 3 and is predicted from column 2.
    # Row 2 has no selected token.
    seqs = [[0, 1, 2, 1], [2, 0], [1, 1, 1]]
    weights = [[0, 1, 0, 1], [0, 1], [0, 0, 0]]
    assert loss_positions(seqs, weights) == ([0, 0, 1], [0, 2, 2], [1, 1, 0], [2, 1, 0])


@pytest.mark.parametrize("seqs, weights", [
    ([[4, 5]], [[1, 0]]),          # the first token can never be selected
    ([[4, 5]], [[0]]),             # one weight per token
    ([[4, 5]], [[0, 2]]),          # weights are 0 or 1
    ([[4, 5]], [[0, 1], [0, 1]]),  # one weight list per sequence
    ([[]], [[]]),                  # no empty sequence
])
def test_loss_positions_rejects_bad_input(seqs, weights):
    with pytest.raises(ValueError):
        loss_positions(seqs, weights)


def test_position_ids_count_real_tokens_only():
    import torch

    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert position_ids(mask).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


def _toy_model():
    """A model whose next-token distribution depends only on the current token: log P[next][current].

    Its hidden state at a position is the one-hot vector of the token there, pad tokens included. So a result
    read from the wrong column, or from a pad column, gives a different number than the expected one.
    """
    import torch

    class Qwen3_5ForCausalLM(torch.nn.Module):           # the class name that causal_lm() asks for
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.log(torch.tensor(TOY_P)))
            self.calls: list[dict] = []

        @property
        def device(self):
            return self.lm_head.weight.device

        def model(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
            self.calls.append({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist(),
                               "position_ids": position_ids.tolist(), "use_cache": use_cache})
            return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 4).float())

    return Qwen3_5ForCausalLM()


def test_token_logprobs_on_the_toy_model():
    import torch

    toy = _toy_model()
    seqs = [[0, 1, 2, 1], [2, 0]]
    weights = [[0, 1, 0, 1], [0, 1]]
    result = token_logprobs(toy, seqs, weights, pad_id=TOY_PAD, micro_batch=2, with_grad=False)
    # Sequence 0: token 1 after token 0 is P[1][0] = 0.2; token 1 after token 2 is P[1][2] = 0.25.
    # Sequence 1: token 0 after token 2 is P[0][2] = 0.25. (Read one column too far to the left it would be
    # the pad token, P[0][3] = 0.7; read without the padding offset it would be P[0][3] as well.)
    assert [len(r) for r in result] == [2, 1]
    assert torch.allclose(result[0], torch.tensor([math.log(0.2), math.log(0.25)]), atol=1e-6)
    assert torch.allclose(result[1], torch.tensor([math.log(0.25)]), atol=1e-6)
    assert all(r.dtype == torch.float32 and not r.requires_grad for r in result)
    assert toy.calls == [{"input_ids": [[0, 1, 2, 1], [TOY_PAD, TOY_PAD, 2, 0]],
                          "attention_mask": [[1, 1, 1, 1], [0, 0, 1, 1]],          # 2-D, left padded
                          "position_ids": [[0, 1, 2, 3], [0, 0, 0, 1]],
                          "use_cache": False}]


def test_token_logprobs_micro_batches_and_empty_selection():
    import torch

    toy = _toy_model()
    seqs = [[0, 1, 2, 1], [2, 0], [1, 1, 1]]
    weights = [[0, 1, 0, 1], [0, 1], [0, 0, 0]]
    together = token_logprobs(toy, seqs, weights, pad_id=TOY_PAD, micro_batch=3, with_grad=False)
    one_by_one = token_logprobs(toy, seqs, weights, pad_id=TOY_PAD, micro_batch=1, with_grad=False)
    assert [r.tolist() for r in together] == [r.tolist() for r in one_by_one]
    assert [len(r) for r in together] == [2, 1, 0]
    assert len(toy.calls) == 1 + 2      # one batch of three; then two single rows (the third selects nothing)
    assert torch.allclose(together[1], torch.tensor([math.log(0.25)]), atol=1e-6)


def test_token_logprobs_with_gradients_takes_one_micro_batch():
    toy = _toy_model()
    seqs = [[0, 1, 2, 1], [2, 0]]
    weights = [[0, 1, 0, 1], [0, 1]]
    with pytest.raises(ValueError):
        token_logprobs(toy, seqs, weights, pad_id=TOY_PAD, micro_batch=1, with_grad=True)
    result = token_logprobs(toy, seqs, weights, pad_id=TOY_PAD, micro_batch=2, with_grad=True)
    assert all(r.requires_grad for r in result)
    (-(result[0].sum() + result[1].sum())).backward()
    grad = toy.lm_head.weight.grad
    # d(-log softmax(w[:, c])[n]) / d w[m][c] = P[m][c] - (1 if m == n else 0), summed over the chosen tokens:
    # column 0 is used once (next token 1), column 2 twice (next tokens 1 and 0), columns 1 and 3 never.
    assert grad[:, 1].abs().sum().item() == 0.0 and grad[:, 3].abs().sum().item() == 0.0
    assert grad[1, 0].item() == pytest.approx(0.2 - 1.0, abs=1e-6)
    assert grad[0, 0].item() == pytest.approx(0.1, abs=1e-6)
    assert grad[0, 2].item() == pytest.approx(2 * 0.25 - 1.0, abs=1e-6)
    assert grad[3, 2].item() == pytest.approx(2 * 0.25, abs=1e-6)


def test_pad_batch_is_the_samplers_function():
    from accessdesk.harness import hf_backend

    assert pad_batch is hf_backend.pad_batch             # one padding function for sampling and training


# ---------------------------------------------------------------------------
# With the model
# ---------------------------------------------------------------------------
def _stats(differences) -> dict:
    values = sorted(abs(float(x)) for x in differences)
    return {"n": len(values), "mean": sum(values) / len(values),
            "p99": values[min(len(values) - 1, int(0.99 * len(values)))], "max": values[-1]}


def _all_but_first(seq: list[int]) -> list[int]:
    return [0] + [1] * (len(seq) - 1)


def test_equal_to_the_plain_logits_path(model, tokenizer, prompts):
    """The reference is the model's own forward: full logits, log_softmax, the next token of each position."""
    import torch

    core = causal_lm(model)
    seqs = [prompts[0][:48], prompts[1][:31]]
    ours = token_logprobs(model, seqs, [_all_but_first(s) for s in seqs], pad_id=tokenizer.pad_token_id,
                          micro_batch=1, with_grad=False)
    differences = []
    for seq, mine in zip(seqs, ours):
        ids = torch.tensor([seq], device=core.device)
        with torch.no_grad():
            logits = core(input_ids=ids).logits[0]                               # [length, vocabulary]
        positions = torch.arange(len(seq) - 1, device=core.device)          # position p predicts token p + 1
        reference = torch.log_softmax(logits.float(), dim=-1)[positions, ids[0, 1:]]
        assert mine.shape == reference.shape == (len(seq) - 1,)
        differences.extend((mine - reference).tolist())
    stats = _stats(differences)
    print("\nplain path against token_logprobs:", stats)
    assert stats["mean"] < 1e-3 and stats["max"] < 0.25


def _logprobs_with_pads_visible(core, seqs: list[list[int]], row: int, pad_id: int):
    """A padding MISTAKE, made on purpose: the left-padded batch is run with an attention mask of ones, so
    the pad tokens are ordinary tokens. Returns the log-probabilities of all tokens but the first of `seqs[row]`.
    Only `test_left_padding_does_not_change_a_sequence` uses it, to show that it would notice a leak."""
    import torch

    input_ids, attention_mask = pad_batch(seqs, pad_id, side="left")
    seq = seqs[row]
    with torch.no_grad():
        hidden = core.model(input_ids=input_ids.to(core.device),
                            attention_mask=torch.ones_like(attention_mask).to(core.device),
                            position_ids=position_ids(attention_mask).to(core.device),
                            use_cache=False, return_dict=True).last_hidden_state
        logits = core.lm_head(hidden[row, -len(seq):-1])
        targets = torch.tensor(seq[1:], device=core.device)
        return torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)


def test_left_padding_does_not_change_a_sequence(model, tokenizer, prompts):
    """400 prompt tokens alone, and as the second row of a batch of two: without pads, and with 80 pads.

    What is asserted, and where each bound comes from (see the module docstring for the mechanism):
      1. The content of the other row never matters: exactly the same numbers with another partner.
      2. Padding adds nothing to the rounding difference that a batch of two has anyway.
         Measured mean, alone against batch of two: no pad 0.0462 (2B), 0.1077 (4B); 80 pads 0.0475 (2B),
         0.1031 (4B). For 0 to 500 pads the 2B stayed between 0.040 and 0.056. Bound: twice the no-pad mean.
      3. Absolute bounds, about three times the largest value measured: mean < 0.3 (measured 0.047 on the
         2B, 0.103 to 0.123 on the 4B), 99th percentile < 4.0 (measured 0.50 to 0.88 on the 2B, 1.1 to 1.9 on
         the 4B). The maximum is printed and not bounded: one unlikely token moved by 2.5 (2B) and 6.9 (4B).
      4. The test can see a leak: with the pads unmasked the mean is 0.82 (2B) and 1.81 (4B), the 99th
         percentile 13.0 and 17.5. Bound: at least five times the padded mean, and above the bounds of 3.
    The first version of this test asked for mean < 0.01 and max < 0.5 on 40 tokens; that was a guess, and
    the first measurement was mean 0.051, max 0.375 (2B), the same as without any padding (0.047).
    """
    short = prompts[0][:400]
    same_length = prompts[1][7:407]          # a partner of equal length: a batch of two without a pad token
    long_a = prompts[1][:480]                # a longer partner: 80 pad tokens on the left of `short`
    long_b = prompts[2][5:485]               # other tokens, the same length as long_a
    assert same_length != short and long_a != long_b
    pad_id = tokenizer.pad_token_id

    def second_row(partner: list[int]):
        return token_logprobs(model, [partner, short], [_all_but_first(partner), _all_but_first(short)],
                              pad_id=pad_id, micro_batch=2, with_grad=False)[1]

    alone = token_logprobs(model, [short], [_all_but_first(short)], pad_id=pad_id, micro_batch=1,
                           with_grad=False)[0]
    padded = second_row(long_a)
    no_pad = _stats((second_row(same_length) - alone).tolist())
    with_pad = _stats((padded - alone).tolist())
    other_partner = _stats((padded - second_row(long_b)).tolist())
    leak = _stats((_logprobs_with_pads_visible(causal_lm(model), [long_a, short], 1, pad_id) - alone).tolist())
    print("\nalone against batch of two, no pad:  ", no_pad)
    print("alone against left-padded (80 pads): ", with_pad)
    print("left-padded, partner A against B:    ", other_partner)
    print("MISTAKE on purpose, pads not masked: ", leak)
    assert other_partner["max"] == 0.0                                           # 1
    assert with_pad["mean"] < 2 * no_pad["mean"] + 0.01                          # 2
    assert with_pad["mean"] < 0.3 and with_pad["p99"] < 4.0                      # 3
    assert leak["mean"] > 5 * with_pad["mean"] and leak["mean"] > 0.3 and leak["p99"] > 4.0      # 4


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_parity_of_sampler_and_trainer_logprobs(model, tokenizer, prompts, mode):
    """PLAN 5.6: the log-probabilities recorded while sampling against `token_logprobs` on the same ids."""
    cfg = dataclasses.replace(RunConfig(), max_new_tokens=64)
    sampler = Sampler(model, tokenizer, cfg)
    turns = sampler.sample(prompts, seed=21)
    seqs = [prompt + turn.ids for prompt, turn in zip(prompts, turns)]
    weights = [[0] * len(prompt) + [1] * len(turn.ids) for prompt, turn in zip(prompts, turns)]
    core = causal_lm(model)
    try:
        core.train(mode == "train")
        ours = token_logprobs(model, seqs, weights, pad_id=sampler.pad_id, micro_batch=2, with_grad=False)
    finally:
        core.eval()
    differences = [a - b for mine, turn in zip(ours, turns) for a, b in zip(mine.tolist(), turn.logprobs)]
    assert len(differences) == sum(len(turn.ids) for turn in turns)
    stats = _stats(differences)
    print(f"\nsampler against trainer ({mode} mode):", stats)
    # Measured on 2026-11-17 (RTX 4090), the same numbers in eval and in train mode:
    #   2B: 152 tokens, mean 5.1e-05, 99th percentile 1.5e-03, largest 2.3e-03
    #   4B: 115 tokens, mean 2.2e-05, 99th percentile 7.1e-04, largest 9.1e-04
    # These are first turns, where the model is nearly certain of every token. On whole episodes the same
    # comparison gives a mean of 0.0017 and a largest value of 0.37 (tests/gpu/test_rollout.py), so the
    # bounds below are those of the plan (PLAN 5.6) and were left as they are.
    assert stats["mean"] < 0.02 and stats["max"] < 1.0


def test_gradients_flow_with_and_without_gradient_checkpointing(model, tokenizer, prompts):
    """One backward pass down to the first decoder layer, with checkpointing off and on: same gradient."""
    import torch

    core = causal_lm(model)
    seq = prompts[0][:64]
    probe = core.model.layers[0].input_layernorm.weight     # its gradient travels through every decoder layer
    flags = [(p, p.requires_grad) for p in core.parameters()]
    grads = {}
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        probe.requires_grad_(True)
        core.train()
        for checkpointing in (False, True):
            set_gradient_checkpointing(model, checkpointing)
            assert core.is_activation_checkpointing is checkpointing
            probe.grad = None
            (logprobs,) = token_logprobs(model, [seq], [_all_but_first(seq)], pad_id=tokenizer.pad_token_id,
                                         micro_batch=1, with_grad=True)
            assert logprobs.requires_grad
            (-logprobs.mean()).backward()
            grads[checkpointing] = probe.grad.detach().float().clone()
    finally:
        set_gradient_checkpointing(model, False)
        core.eval()
        probe.grad = None
        for p, flag in flags:
            p.requires_grad_(flag)
    for grad in grads.values():
        assert bool(torch.isfinite(grad).all()) and grad.abs().sum().item() > 0
    cosine = torch.nn.functional.cosine_similarity(grads[False], grads[True], dim=0).item()
    print(f"\ngradient with against without checkpointing: cosine = {cosine:.6f}")
    assert cosine > 0.99


def test_no_graph_without_gradients(model, tokenizer, prompts):
    seq = prompts[0][:32]
    (logprobs,) = token_logprobs(model, [seq], [_all_but_first(seq)], pad_id=tokenizer.pad_token_id,
                                 micro_batch=1, with_grad=False)
    assert not logprobs.requires_grad and logprobs.dtype.is_floating_point
    assert all(value <= 1e-6 for value in logprobs.tolist())
