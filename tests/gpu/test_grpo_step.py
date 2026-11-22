"""The arithmetic of one GRPO step, on a hand-made batch and a four-token toy model.

Two statements of accessdesk/train/grpo.py are proved here:
  1. the normaliser of the loss is a constant of the config (n_scenarios * group_size * reference_tokens). It
     does not change with the number of kept groups, of episodes or of tokens that go through the model;
  2. a dropped group changes nothing in the loss or in the gradient, except that its own terms are zero.

The toy model stands in for the real one (no torch tricks: it is a plain module). Its hidden state at a
position is the one-hot vector of the token there, and its output layer W starts as log P, so the
log-probability of token n after token c is log P[n][c], and

    d log p(n | c) / d W[m][c] = (1 if m == n else 0) - P[m][c]

With that, the loss and its gradient are worked out by hand below, from a table of (advantage, current token,
next token) that is written down separately from the code under test.

The toy's output layer is stored under the parameter name that peft gives to a LoRA weight
("model.layers.0.mlp.down_proj.lora_B.default.weight"), so that `make_optimizer`, `zero_gradients` and
`clip_and_step`, which accept LoRA weights only, run on it unchanged.

These tests need torch but no GPU work and no model download. The tests of the scenario order and of
`plan_update` need no torch at all.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from accessdesk.train.forward import token_logprobs
from accessdesk.train.grpo import (GRPOConfig, accumulate_policy_gradient, find_fault, plan_update,
                                   reward_matrix, stack_logprobs, step_scenarios, zero_gradients)
from accessdesk.train.sft import clip_and_step, make_optimizer

# P[next][current] of the toy model: column c is the distribution of the token after token c.
TOY_P = [[0.1, 0.4, 0.25, 0.7],
         [0.2, 0.3, 0.25, 0.1],
         [0.3, 0.2, 0.25, 0.1],
         [0.4, 0.1, 0.25, 0.1]]
PAD = 3
GCFG = GRPOConfig(n_scenarios=3, group_size=2, reference_tokens=5)        # C = 3 * 2 * 5 = 30

# The batch: three groups of two episodes. The first token of an episode is its "prompt" (weight 0); the
# others were "sampled" (weight 1).
#   group A  rewards 100, 0     mixed -> kept, advantages +0.5, -0.5
#   group B  rewards 100, 100   tied  -> dropped
#   group C  rewards 100, 95    a penalty only -> kept, mean 0.975, advantages +0.025, -0.025
EPISODES = [("A", [0, 1, 2], 100), ("A", [2, 0], 0),
            ("B", [1, 1, 1], 100), ("B", [0, 2], 100),
            ("C", [1, 0, 0], 100), ("C", [2, 2], 95)]
# Every sampled token of the kept groups as (advantage of its episode, current token, next token).
TERMS = [(0.5, 0, 1), (0.5, 1, 2),            # A0: 1 after 0, 2 after 1
         (-0.5, 2, 0),                        # A1: 0 after 2
         (0.025, 1, 0), (0.025, 0, 0),        # C0: 0 after 1, 0 after 0
         (-0.025, 2, 2)]                      # C1: 2 after 2
# The same for group B if its rewards were 100 and 0 (advantages +0.5, -0.5).
TERMS_OF_B_IF_MIXED = [(0.5, 1, 1), (0.5, 1, 1), (-0.5, 0, 2)]
# loss = -(1/30) * [0.5 * (ln 0.2 + ln 0.2) - 0.5 * ln 0.25 + 0.025 * (ln 0.4 + ln 0.1) - 0.025 * ln 0.25]
LOSS_BY_HAND = 0.03207017561559542


def _expected_gradient(terms: list[tuple[float, int, int]], norm_const: float) -> list[list[float]]:
    """d loss / d W[m][c] = -(1 / C) * sum over the terms with current token c of A * ((m == n) - P[m][c])."""
    grad = [[0.0] * 4 for _ in range(4)]
    for advantage, current, nxt in terms:
        for m in range(4):
            is_next = 1.0 if m == nxt else 0.0
            grad[m][current] += -(1.0 / norm_const) * advantage * (is_next - TOY_P[m][current])
    return grad


def _true_logprobs(ids: list[int]) -> list[float]:
    return [math.log(TOY_P[ids[t]][ids[t - 1]]) for t in range(1, len(ids))]


def _record(scenario: str, ids: list[int], reward: int, *, harness_error: str | None = None) -> dict:
    """The fields of an episode record that the GRPO step reads."""
    tokens = {"ids": ids, "segments": [["prompt", 0, 1, -1], ["sampled", 1, len(ids), 0]],
              "logprobs": _true_logprobs(ids)}
    return {"scenario_id": scenario, "harness_error": harness_error, "reward": reward, "tokens": tokens}


def _batch(rewards_of_b: tuple[int, int] = (100, 100)) -> list[dict]:
    records = [_record(*episode) for episode in EPISODES]
    records[2]["reward"], records[3]["reward"] = rewards_of_b
    return records


def _toy_model():
    import torch

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            down_proj = torch.nn.Module()
            down_proj.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 4, bias=False)})
            mlp = torch.nn.Module()
            mlp.down_proj = down_proj
            layer = torch.nn.Module()
            layer.mlp = mlp
            self.layers = torch.nn.ModuleList([layer])

        def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
            return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 4).float())

    class Qwen3_5ForCausalLM(torch.nn.Module):           # the class name that causal_lm() asks for
        def __init__(self) -> None:
            super().__init__()
            self.model = Decoder()
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.log(torch.tensor(TOY_P)))

        @property
        def lm_head(self):
            return self.model.layers[0].mlp.down_proj.lora_B["default"]

        @property
        def device(self):
            return self.lm_head.weight.device

    return Qwen3_5ForCausalLM()


def _run(records: list[dict], advantages, selected: list[int], *, norm_const: float = GCFG.norm_const,
         micro_batch: int = 2):
    """(numbers, gradient of the toy's output layer) of one `accumulate_policy_gradient` call on a new toy."""
    import torch

    toy = _toy_model()
    numbers = accumulate_policy_gradient(toy, records, advantages, selected, norm_const=norm_const,
                                         pad_id=PAD, micro_batch=micro_batch)
    grad = toy.lm_head.weight.grad
    return numbers, (torch.zeros(4, 4) if grad is None else grad.clone())


# ---------------------------------------------------------------------------
# No torch
# ---------------------------------------------------------------------------
def test_plan_of_the_hand_made_batch():
    plan = plan_update(_batch(), GCFG)
    assert GCFG.norm_const == 30.0
    assert plan["rewards"].tolist() == [[1.0, 0.0], [1.0, 1.0], [1.0, 0.95]]
    assert plan["keep"].tolist() == [True, False, True]
    assert np.allclose(plan["advantages"], [[0.5, -0.5], [0.0, 0.0], [0.025, -0.025]], atol=1e-12)
    assert plan["selected"] == [0, 1, 4, 5]                 # the episodes of the kept groups A and C


def test_a_harness_error_is_not_selected_and_its_reward_is_not_read():
    gcfg = GRPOConfig(n_scenarios=1, group_size=4, reference_tokens=5)
    records = [_record("A", [0, 1, 2], 100), _record("A", [2, 0], 0),
               _record("A", [0, 1], 100, harness_error="made up"), _record("A", [1, 1], 100)]
    records[2]["tokens"] = {"ids": [], "segments": [], "logprobs": []}       # incomplete, as after a failure
    plan = plan_update(records, gcfg)
    assert np.isnan(plan["rewards"][0, 2])
    # Three valid rewards 1, 0, 1: mean 2/3.
    assert np.allclose(plan["advantages"], [[1 / 3, -2 / 3, 0.0, 1 / 3]], atol=1e-12)
    assert plan["keep"].tolist() == [True] and plan["selected"] == [0, 1, 3]


def test_reward_matrix_refuses_a_group_of_two_scenarios():
    records = _batch()
    with pytest.raises(ValueError):
        reward_matrix(records[1:] + records[:1], 3, 2)
    with pytest.raises(ValueError):
        reward_matrix(records[:5], 3, 2)


def test_scenario_order_has_no_repeat_in_a_pass_and_goes_on_after_a_restart():
    scenarios = [f"tr-{family:03d}/s{style}" for family in range(200) for style in (1, 2, 3)]      # 600
    steps_per_pass = 600 // 16                                                                    # 37
    run = [step_scenarios(scenarios, 16, 7, step) for step in range(80)]
    assert all(len(step) == len(set(step)) == 16 for step in run)
    first = [scenario for step in run[:steps_per_pass] for scenario in step]
    second = [scenario for step in run[steps_per_pass:2 * steps_per_pass] for scenario in step]
    assert len(set(first)) == len(first) == 592 and len(set(second)) == len(second) == 592
    assert first != second                                   # the second pass has a new order
    # A run that is restarted at step 30 uses, from there on, exactly what the first run used: no scenario
    # of the steps 0 to 29 of this pass comes again, and none is skipped.
    assert [step_scenarios(scenarios, 16, 7, step) for step in range(30, 80)] == run[30:]
    assert not set(first[:30 * 16]) & {s for step in run[30:steps_per_pass] for s in step}
    assert step_scenarios(list(reversed(scenarios)), 16, 7, 3) == run[3]       # the input order is ignored
    assert step_scenarios(scenarios, 16, 8, 0) != run[0]                       # the seed does
    for bad in ((scenarios[:5], 16, 0, 0), (scenarios + scenarios[:1], 16, 0, 0), (scenarios, 16, 0, -1)):
        with pytest.raises(ValueError):
            step_scenarios(*bad)


def test_stop_rule_by_hand():
    # The limits of the default config: harness errors 1%, gap 0.02, invalid calls 5% and twice step 0.
    gcfg = GRPOConfig()
    state = {"n_episodes": 1280, "n_harness_errors": 12, "invalid_call_rate_step0": 0.03}
    one_more = {**state, "n_harness_errors": 13}
    assert find_fault({"invalid_call_rate": 0.04}, state, gcfg) is None                    # 12 / 1280 = 0.94%
    assert "harness errors" in find_fault({"invalid_call_rate": 0.04}, one_more, gcfg)     # 13 / 1280 = 1.02%
    assert find_fault({"invalid_call_rate": 0.059}, state, gcfg) is None         # above 5%, but not twice 3%
    assert "invalid-call" in find_fault({"invalid_call_rate": 0.06}, state, gcfg)         # above 5%, twice 3%
    low_start = {**state, "invalid_call_rate_step0": 0.02}
    assert find_fault({"invalid_call_rate": 0.045}, low_start, gcfg) is None     # twice 2%, but not above 5%
    assert find_fault({"invalid_call_rate": 0.0, "logprob_gap": {"mean": 0.02}}, state, gcfg) is None
    assert "gap" in find_fault({"invalid_call_rate": 0.0, "logprob_gap": {"mean": 0.021}}, state, gcfg)
    assert find_fault({"invalid_call_rate": 0.0, "logprob_gap": None}, state, gcfg) is None    # no training


# ---------------------------------------------------------------------------
# With torch and the toy model
# ---------------------------------------------------------------------------
def test_stack_logprobs_by_hand():
    import torch

    short = torch.tensor([-1.0], requires_grad=True)
    long = torch.tensor([-2.0, -3.0, -4.0], requires_grad=True)
    logprobs, mask = stack_logprobs([short, long])
    assert logprobs.tolist() == [[-1.0, 0.0, 0.0], [-2.0, -3.0, -4.0]]
    assert mask.tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]] and not mask.requires_grad
    (logprobs * mask).sum().backward()
    assert short.grad.tolist() == [1.0] and long.grad.tolist() == [1.0, 1.0, 1.0]


def test_loss_and_gradient_by_hand():
    import torch

    records = _batch()
    plan = plan_update(records, GCFG)
    numbers, grad = _run(records, plan["advantages"], plan["selected"])
    assert numbers["loss"] == pytest.approx(LOSS_BY_HAND, abs=1e-7)
    assert numbers["n_episodes"] == 4 and numbers["n_tokens"] == 6
    assert torch.allclose(grad, torch.tensor(_expected_gradient(TERMS, 30.0)), atol=1e-7)
    # Three entries by hand. W[1][0]: -(1/30) * (0.5 * (1 - 0.2) + 0.025 * (0 - 0.2)) = -0.395 / 30.
    assert grad[1, 0].item() == pytest.approx(-0.395 / 30, abs=1e-7)
    # W[3][0]: -(1/30) * (0.5 * (0 - 0.4) + 0.025 * (0 - 0.4)) = 0.21 / 30 = 0.007.
    assert grad[3, 0].item() == pytest.approx(0.007, abs=1e-7)
    # W[0][2]: -(1/30) * (-0.5 * (1 - 0.25) - 0.025 * (0 - 0.25)) = 0.36875 / 30.
    assert grad[0, 2].item() == pytest.approx(0.36875 / 30, abs=1e-7)
    assert torch.count_nonzero(grad[:, 3]).item() == 0       # the pad token is never a "current" token


def test_the_normaliser_is_the_config_constant_whatever_goes_through_the_model():
    """Statement 1. Four ways to run the same batch give one gradient; only the config constant scales it."""
    import torch

    records = _batch()
    plan = plan_update(records, GCFG)
    _, reference = _run(records, plan["advantages"], plan["selected"])

    # (a) The dropped group is run through the model as well, with its advantages of 0: 6 episodes and 9
    #     tokens instead of 4 and 6. A normaliser that counted episodes or tokens would change the result.
    numbers, grad = _run(records, plan["advantages"], list(range(6)))
    assert numbers["n_episodes"] == 6 and numbers["n_tokens"] == 9
    assert numbers["loss"] == pytest.approx(LOSS_BY_HAND, abs=1e-7)
    assert torch.allclose(grad, reference, atol=1e-7)

    # (b) The dropped group is not in the lists at all: only the four episodes of A and C exist.
    four = [records[i] for i in plan["selected"]]
    four_advantages = [plan["advantages"].reshape(-1)[i] for i in plan["selected"]]
    numbers, grad = _run(four, four_advantages, [0, 1, 2, 3])
    assert numbers["loss"] == pytest.approx(LOSS_BY_HAND, abs=1e-7)
    assert torch.allclose(grad, reference, atol=1e-7)

    # (c) Slices of 1, 2 and 4 episodes: the parts of the sum add up to the same gradient.
    for micro_batch in (1, 4):
        numbers, grad = _run(records, plan["advantages"], plan["selected"], micro_batch=micro_batch)
        assert numbers["loss"] == pytest.approx(LOSS_BY_HAND, abs=1e-7)
        assert torch.allclose(grad, reference, atol=1e-7)

    # (d) The constant itself: twice the reference token count, half the loss and half the gradient.
    doubled = GRPOConfig(n_scenarios=3, group_size=2, reference_tokens=10)
    numbers, grad = _run(records, plan["advantages"], plan["selected"], norm_const=doubled.norm_const)
    assert numbers["loss"] == pytest.approx(LOSS_BY_HAND / 2, abs=1e-7)
    assert torch.allclose(grad, reference / 2, atol=1e-7)


def test_a_dropped_group_only_removes_its_own_terms():
    """Statement 2. Group B once tied (dropped) and once mixed (kept): the difference of the two gradients
    is exactly the terms of B, with the same 1/30 in front. The terms of A and C are not rescaled."""
    import torch

    tied = _batch((100, 100))
    mixed = _batch((100, 0))
    plan_tied, plan_mixed = plan_update(tied, GCFG), plan_update(mixed, GCFG)
    assert plan_tied["keep"].tolist() == [True, False, True] and plan_mixed["keep"].tolist() == [True] * 3
    numbers_tied, grad_tied = _run(tied, plan_tied["advantages"], plan_tied["selected"])
    numbers_mixed, grad_mixed = _run(mixed, plan_mixed["advantages"], plan_mixed["selected"])
    terms_of_b = torch.tensor(_expected_gradient(TERMS_OF_B_IF_MIXED, 30.0))
    assert torch.count_nonzero(terms_of_b).item() > 0
    assert torch.allclose(grad_mixed - grad_tied, terms_of_b, atol=1e-7)
    assert torch.allclose(grad_tied, torch.tensor(_expected_gradient(TERMS, 30.0)), atol=1e-7)
    # The loss: B's own part is -(1/30) * (0.5 * (ln 0.3 + ln 0.3) - 0.5 * ln 0.3) = -(1/30) * 0.5 * ln 0.3.
    assert numbers_mixed["loss"] - numbers_tied["loss"] == pytest.approx(-0.5 * math.log(0.3) / 30, abs=1e-7)


def test_the_logprob_gap_is_measured_on_the_tokens_that_are_trained():
    records = _batch()
    records[1]["tokens"]["logprobs"][0] += 0.1               # A1: its one token
    records[5]["tokens"]["logprobs"][0] -= 0.3               # C1: its one token
    records[2]["tokens"]["logprobs"][0] += 9.0               # B0 is dropped: it is not measured
    plan = plan_update(records, GCFG)
    numbers, _ = _run(records, plan["advantages"], plan["selected"])
    gap = numbers["logprob_gap"]
    assert gap["n"] == 6 and gap["max"] == pytest.approx(0.3, abs=1e-6)
    assert gap["mean"] == pytest.approx(0.4 / 6, abs=1e-6)
    records[0]["tokens"]["logprobs"].append(-1.0)            # one sampler value too many
    with pytest.raises(ValueError):
        _run(records, plan["advantages"], plan["selected"])


def test_one_update_moves_the_logprobs_in_the_direction_of_the_advantage():
    import torch

    toy = _toy_model()
    optimizer = make_optimizer(toy, lr=0.05)
    records = _batch()
    plan = plan_update(records, GCFG)
    zero_gradients(toy)
    accumulate_policy_gradient(toy, records, plan["advantages"], plan["selected"], norm_const=GCFG.norm_const,
                               pad_id=PAD, micro_batch=2)
    norm = clip_and_step(toy, optimizer, 0.01)
    # The returned norm is the one BEFORE clipping: the norm of the hand-made gradient, not the limit 0.01.
    by_hand = math.sqrt(sum(value ** 2 for row in _expected_gradient(TERMS, 30.0) for value in row))
    assert by_hand == pytest.approx(0.026356925, abs=1e-8)
    assert norm == pytest.approx(by_hand, abs=1e-6)
    assert toy.lm_head.weight.grad is None                   # cleared after the step
    good, bad = [0, 1, 2], [2, 0]                            # A0 (advantage +0.5) and A1 (advantage -0.5)
    after = token_logprobs(toy, [good, bad], [[0, 1, 1], [0, 1]], pad_id=PAD, micro_batch=2, with_grad=False)
    assert after[0].sum().item() > sum(_true_logprobs(good))
    assert after[1].sum().item() < sum(_true_logprobs(bad))
    assert torch.isfinite(toy.lm_head.weight).all()


def test_a_step_without_a_kept_group_has_an_exactly_zero_gradient():
    import torch

    toy = _toy_model()
    optimizer = make_optimizer(toy, lr=0.05)
    before = toy.lm_head.weight.detach().clone()
    records = [_record(scenario, ids, 100) for scenario, ids, _ in EPISODES]       # every group is tied
    plan = plan_update(records, GCFG)
    assert not plan["keep"].any() and plan["selected"] == []
    zero_gradients(toy)
    numbers = accumulate_policy_gradient(toy, records, plan["advantages"], plan["selected"],
                                         norm_const=GCFG.norm_const, pad_id=PAD, micro_batch=2)
    assert numbers == {"loss": 0.0, "n_episodes": 0, "n_tokens": 0, "logprob_gap": None}
    grad = toy.lm_head.weight.grad
    assert grad is not None and torch.count_nonzero(grad).item() == 0
    assert clip_and_step(toy, optimizer, 1.0) == 0.0
    assert torch.equal(toy.lm_head.weight, before)           # no history and a zero gradient: nothing moves


def test_after_a_real_step_a_zero_gradient_step_still_moves_the_weights():
    """This documents a choice. A step whose groups are all dropped gives AdamW a zero gradient; AdamW then
    still moves the weights with what is left of its running average of earlier gradients."""
    import torch

    toy = _toy_model()
    optimizer = make_optimizer(toy, lr=0.05)
    start = toy.lm_head.weight.detach().clone()
    records = _batch()
    plan = plan_update(records, GCFG)
    zero_gradients(toy)
    accumulate_policy_gradient(toy, records, plan["advantages"], plan["selected"], norm_const=GCFG.norm_const,
                               pad_id=PAD, micro_batch=2)
    clip_and_step(toy, optimizer, 1.0)
    after_real_step = toy.lm_head.weight.detach().clone()
    zero_gradients(toy)                                      # the next step: nothing kept, gradient zero
    assert clip_and_step(toy, optimizer, 1.0) == 0.0
    first_move = after_real_step - start
    second_move = toy.lm_head.weight.detach() - after_real_step
    moved = first_move != 0
    assert moved.any() and torch.equal(second_move != 0, moved)
    assert bool((torch.sign(second_move[moved]) == torch.sign(first_move[moved])).all())
    assert bool((second_move[moved].abs() < first_move[moved].abs()).all())     # and it is a smaller move


def test_a_gradient_that_is_not_finite_makes_no_step():
    import torch

    toy = _toy_model()
    optimizer = make_optimizer(toy, lr=0.05)
    before = toy.lm_head.weight.detach().clone()
    zero_gradients(toy)
    toy.lm_head.weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        clip_and_step(toy, optimizer, 1.0)
    assert torch.equal(toy.lm_head.weight, before) and toy.lm_head.weight.grad is None
