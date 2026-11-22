"""Tests of accessdesk/train/lora.py: which modules get an adapter, what is trainable, gradients, saving
and loading.

The first tests need no torch: the target expression against module names read from the model's source
(transformers 5.23.0, models/qwen3_5/modeling_qwen3_5.py), and `coverage_problems` on hand-made tables.

The others use the real model of the session. `add_lora` changes a model in place, and the other test files
share that model, so the fixture `lora_model` puts everything back when this file is done: the adapters are
removed with peft's `unload()` and every `requires_grad` flag is restored.
"""
from __future__ import annotations

import copy
import json
import os
import re
from types import SimpleNamespace

import pytest

from accessdesk.harness.hf_backend import causal_lm, load_model
from accessdesk.train.forward import set_gradient_checkpointing, token_logprobs
from accessdesk.train.lora import (ADAPTER, BLOCK_OF_LAYER_TYPE, DEFAULT_TARGETS, GATE_TARGETS, REQUIRED,
                                   add_lora, assert_coverage, coverage_problems, coverage_summary,
                                   coverage_table, gradient_report, lora_fingerprint, lora_state,
                                   save_adapter, target_pattern, trainable_parameters, weight_change_report)
from accessdesk.train.sft import clip_and_step, make_optimizer, sft_gradient

RANK = 16
BLOCKS = ("linear_attn", "self_attn", "mlp")

# ---------------------------------------------------------------------------
# No torch: the target expression
# ---------------------------------------------------------------------------
INSIDE = ["model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn.in_proj_z",
          "model.layers.2.linear_attn.out_proj", "model.layers.3.self_attn.q_proj",
          "model.layers.3.self_attn.k_proj", "model.layers.3.self_attn.v_proj",
          "model.layers.31.self_attn.o_proj", "model.layers.0.mlp.gate_proj", "model.layers.17.mlp.up_proj",
          "model.layers.31.mlp.down_proj"]
NOT_TARGETS = [
    "lm_head", "model.embed_tokens", "model.norm", "model.layers.0", "model.layers.0.mlp",
    # the two gates: off by default
    "model.layers.0.linear_attn.in_proj_a", "model.layers.0.linear_attn.in_proj_b",
    "model.layers.0.linear_attn.conv1d", "model.layers.0.linear_attn.norm",
    # a name in the wrong block
    "model.layers.0.self_attn.out_proj", "model.layers.0.linear_attn.q_proj",
    # names of the model with the vision tower
    "model.visual.blocks.0.mlp.gate_proj", "model.language_model.layers.0.mlp.gate_proj",
    # the expression must match from the start of the name ...
    "base_model.model.model.layers.0.mlp.gate_proj", "xmodel.layers.0.mlp.gate_proj",
    # ... and to its end
    "model.layers.0.mlp.gate_proj.lora_A.default", "model.layers.0.mlp.gate_proj.base_layer",
    "model.layers.x.mlp.gate_proj", "model.layers.0.mlp.gate_projx",
]


def test_target_pattern_matches_the_projections_of_the_decoder_layers_only():
    pattern = target_pattern()
    assert pattern == (r"model\.layers\.\d+\.(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
                       r"|mlp\.(?:gate_proj|up_proj|down_proj)|self_attn\.(?:q_proj|k_proj|v_proj|o_proj))")
    assert all(re.fullmatch(pattern, name) for name in INSIDE)           # peft uses re.fullmatch
    assert not any(re.fullmatch(pattern, name) for name in NOT_TARGETS)


def test_target_pattern_with_the_gate_projections():
    pattern = target_pattern(DEFAULT_TARGETS + GATE_TARGETS)
    assert re.fullmatch(pattern, "model.layers.0.linear_attn.in_proj_a")
    assert re.fullmatch(pattern, "model.layers.5.linear_attn.in_proj_b")
    assert not re.fullmatch(pattern, "model.layers.0.self_attn.in_proj_a")
    assert not re.fullmatch(pattern, "lm_head")


@pytest.mark.parametrize("targets", [[], ["q_proj", "q_proj"], ["lm_head"], ["embed_tokens"], ["all-linear"]])
def test_target_pattern_rejects_what_is_not_a_projection(targets):
    with pytest.raises(ValueError):
        target_pattern(targets)


# ---------------------------------------------------------------------------
# No torch: coverage_problems on hand-made tables
# ---------------------------------------------------------------------------
LINEAR_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
FULL_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")


def _row(layer, layer_type, block, module, wrapped) -> dict:
    path = module if layer is None else f"model.layers.{layer}.{block}.{module}"
    return {"path": path, "layer": layer, "layer_type": layer_type, "block": block, "module": module,
            "wrapped": wrapped, "shape": [8, 8], "lora_params": RANK * 16 if wrapped else 0}


def _hand_table(layer_types: list[str]) -> list[dict]:
    """A correct table for a tiny model: the default targets wrapped, the gates and lm_head not."""
    rows = []
    for layer, layer_type in enumerate(layer_types):
        block = BLOCK_OF_LAYER_TYPE[layer_type]
        for module in (LINEAR_MODULES if block == "linear_attn" else FULL_MODULES):
            rows.append(_row(layer, layer_type, block, module, module in DEFAULT_TARGETS))
        rows.extend(_row(layer, layer_type, "mlp", module, True) for module in MLP_MODULES)
    return rows + [_row(None, None, None, "lm_head", False)]


TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
CONFIG = SimpleNamespace(layer_types=TYPES)


def _find(table: list[dict], path: str) -> dict:
    return next(row for row in table if row["path"] == path)


def test_a_correct_table_has_no_problem():
    table = _hand_table(TYPES)
    assert coverage_problems(table, CONFIG) == []
    assert_coverage(table, CONFIG)
    _find(table, "model.layers.1.linear_attn.in_proj_a").update(wrapped=True, lora_params=5)   # allowed
    assert coverage_problems(table, CONFIG) == []


@pytest.mark.parametrize("path", [
    "model.layers.0.linear_attn.in_proj_qkv", "model.layers.1.linear_attn.in_proj_z",
    "model.layers.2.linear_attn.out_proj", "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.o_proj", "model.layers.0.mlp.gate_proj", "model.layers.3.mlp.down_proj"])
def test_a_missing_adapter_is_found(path):
    table = _hand_table(TYPES)
    _find(table, path).update(wrapped=False, lora_params=0)
    problems = coverage_problems(table, CONFIG)
    assert len(problems) == 1 and path.split(".", 3)[3] in problems[0]
    with pytest.raises(AssertionError):
        assert_coverage(table, CONFIG)


def test_the_silent_miss_of_a_llama_style_list_is_found():
    """Only q, k, v, o and the MLP wrapped: every linear_attention layer misses its three projections."""
    table = _hand_table(TYPES)
    for row in table:
        if row["block"] == "linear_attn":
            row.update(wrapped=False, lora_params=0)
    assert len(coverage_problems(table, CONFIG)) == 3 * 3


def test_an_adapter_outside_the_decoder_layers_is_found():
    table = _hand_table(TYPES)
    _find(table, "lm_head").update(wrapped=True, lora_params=7)
    assert any("lm_head" in problem for problem in coverage_problems(table, CONFIG))
    table = _hand_table(TYPES) + [_row(None, None, None, "model.visual.blocks.0.mlp.linear_fc1", True)]
    assert any("visual" in problem for problem in coverage_problems(table, CONFIG))


def test_wrong_layers_are_found():
    assert coverage_problems(_hand_table(TYPES[:3]), CONFIG)                     # a layer is missing
    swapped = ["full_attention", "linear_attention", "linear_attention", "linear_attention"]
    assert coverage_problems(_hand_table(swapped), CONFIG)                       # same counts, other places
    assert coverage_problems(_hand_table(["linear_attention"] * 4), CONFIG)      # other counts
    table = _hand_table(TYPES)
    table.append(_row(0, "linear_attention", "linear_attn", "something_else", True))
    assert any("something_else" in problem for problem in coverage_problems(table, CONFIG))
    table = _hand_table(TYPES)
    _find(table, "model.layers.0.mlp.up_proj")["lora_params"] = 0
    assert coverage_problems(table, CONFIG)                                      # wrapped, without a weight


def test_coverage_summary_folds_the_layers():
    summary = coverage_summary(_hand_table(TYPES))
    by_name = {(row["block"], row["module"]): row for row in summary}
    wrapped_of = {key: (row["wrapped"], row["of"]) for key, row in by_name.items()}
    assert wrapped_of[("linear_attn", "in_proj_qkv")] == (3, 3)
    assert wrapped_of[("linear_attn", "in_proj_a")] == (0, 3)
    assert wrapped_of[("self_attn", "q_proj")] == (1, 1) and wrapped_of[(None, "lm_head")] == (0, 1)
    # The MLP rows are kept apart by layer type: 3 linear_attention layers and 1 full_attention layer.
    mlp = sorted(row["of"] for row in summary if row["module"] == "gate_proj")
    assert mlp == [1, 3] and summary[-1]["module"] == "lm_head"


# ---------------------------------------------------------------------------
# With the model
# ---------------------------------------------------------------------------
def _example(prompts: list[list[int]]) -> dict:
    """A short training example: the first 96 tokens of a real prompt; every token but the first is in the
    loss."""
    ids = prompts[0][:96]
    return {"ids": ids, "weights": [0] + [1] * (len(ids) - 1)}


def _logprobs(model, tokenizer, example: dict):
    return token_logprobs(model, [example["ids"]], [example["weights"]], pad_id=tokenizer.pad_token_id,
                          micro_batch=1, with_grad=False)[0]


@pytest.fixture(scope="module")
def lora_model(model, tokenizer, prompts):
    """(the peft wrapper, the log-probabilities of `_example` under the model before it got adapters)."""
    core = causal_lm(model)
    flags = [(param, param.requires_grad) for param in core.parameters()]
    before = _logprobs(model, tokenizer, _example(prompts))
    wrapped = add_lora(model, rank=RANK, alpha=32, seed=0)
    yield wrapped, before
    set_gradient_checkpointing(wrapped, False)
    wrapped.unload()                         # peft puts the original layers back (tuners_utils.py line 780)
    for param, flag in flags:
        param.requires_grad_(flag)
        param.grad = None
    core.eval()
    assert not any("lora_" in name for name, _ in core.named_parameters())


def test_add_lora_returns_a_wrapper_of_the_same_model(model, lora_model):
    wrapped, _ = lora_model
    assert causal_lm(wrapped) is model
    with pytest.raises(ValueError):
        add_lora(wrapped)                    # a second call needs the bare model


def test_coverage_of_the_real_model(model, lora_model):
    wrapped, _ = lora_model
    config = causal_lm(wrapped).config
    table = coverage_table(wrapped)
    assert_coverage(table, config)
    n_linear = config.layer_types.count("linear_attention")
    n_full = config.layer_types.count("full_attention")
    assert n_linear + n_full == config.num_hidden_layers == len(causal_lm(wrapped).model.layers)
    n_wrapped = sum(1 for row in table if row["wrapped"])
    assert n_wrapped == 3 * n_linear + 4 * n_full + 3 * config.num_hidden_layers
    not_wrapped = sorted({row["module"] for row in table if not row["wrapped"]})
    assert not_wrapped == ["in_proj_a", "in_proj_b", "lm_head"]         # every other linear layer has one
    assert not any("visual" in row["path"] or "embed" in row["path"] for row in table)
    for row in table:
        if row["wrapped"]:
            out_features, in_features = row["shape"]
            assert row["lora_params"] == RANK * (in_features + out_features)
    print("\nLoRA coverage (layer type, block, module, wrapped/of, shape, adapter weights):")
    for row in coverage_summary(table):
        print(f"  {row['layer_type']!s:17} {row['block']!s:12} {row['module']:12} "
              f"{row['wrapped']:3d}/{row['of']:<3d} {row['shape']!s:15} {row['lora_params']:>10,d}")


def test_assert_coverage_fails_on_a_changed_real_table(lora_model):
    wrapped, _ = lora_model
    config = causal_lm(wrapped).config
    table = coverage_table(wrapped)
    missing = copy.deepcopy(table)
    next(row for row in missing if row["module"] == "in_proj_qkv").update(wrapped=False, lora_params=0)
    with pytest.raises(AssertionError):
        assert_coverage(missing, config)
    outside = copy.deepcopy(table)
    next(row for row in outside if row["module"] == "lm_head").update(wrapped=True, lora_params=1)
    with pytest.raises(AssertionError):
        assert_coverage(outside, config)


def test_only_float32_lora_weights_are_trainable(lora_model):
    import torch

    wrapped, _ = lora_model
    core = causal_lm(wrapped)
    params = trainable_parameters(wrapped)
    table = coverage_table(wrapped)
    assert sum(param.numel() for param in params) == sum(row["lora_params"] for row in table)
    assert len(params) == 2 * sum(1 for row in table if row["wrapped"])          # lora_A and lora_B of each
    assert all(param.dtype == torch.float32 for param in params)
    for name, param in core.named_parameters():
        assert param.requires_grad == (".lora_A." in name or ".lora_B." in name), name
    assert core.lm_head.weight.dtype == torch.bfloat16 and not core.lm_head.weight.requires_grad
    # lora_B starts at zero, lora_A does not.
    named = dict(core.named_parameters())
    b_name = next(name for name in named if ".lora_B." in name)
    assert torch.count_nonzero(named[b_name]).item() == 0
    assert torch.count_nonzero(named[b_name.replace(".lora_B.", ".lora_A.")]).item() > 0


def test_the_adapter_changes_nothing_before_training(lora_model, tokenizer, prompts):
    wrapped, before = lora_model
    after = _logprobs(wrapped, tokenizer, _example(prompts))
    difference = (after - before).abs()
    print(f"\nbase against base with zero adapters: mean {difference.mean().item():.2e}, "
          f"max {difference.max().item():.2e}")
    assert difference.mean().item() < 1e-3 and difference.max().item() < 0.25


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradients_reach_both_layer_types_and_the_mlp(lora_model, tokenizer, prompts, checkpointing):
    import math

    wrapped, _ = lora_model
    core = causal_lm(wrapped)
    example = _example(prompts)
    try:
        set_gradient_checkpointing(wrapped, checkpointing)
        wrapped.train()
        for param in trainable_parameters(wrapped):
            param.grad = None
        numbers = sft_gradient(wrapped, [example], pad_id=tokenizer.pad_token_id, micro_batch=1)
        report = gradient_report(wrapped)
    finally:
        for param in trainable_parameters(wrapped):
            param.grad = None
        set_gradient_checkpointing(wrapped, False)
        wrapped.eval()
    print(f"\ncheckpointing {checkpointing}: loss {numbers['loss']:.4f}, "
          f"lora_B gradient norms {report['by_block']}")
    assert math.isfinite(numbers["loss"]) and numbers["n_tokens"] == 95
    assert sorted(report["by_block"]) == sorted(BLOCKS)
    assert all(math.isfinite(norm) and norm > 0 for norm in report["by_block"].values())
    assert sorted(report["by_layer_type"]) == ["full_attention", "linear_attention"]
    assert all(norm > 0 for norm in report["by_layer_type"].values())
    wanted = {name for names in REQUIRED.values() for name in names}
    assert set(report["by_module"]) == wanted and all(norm > 0 for norm in report["by_module"].values())
    assert report["without_gradient"]["n"] == 0
    assert report["lora_A"] == 0.0                         # exactly zero while lora_B is zero
    assert all(param.grad is None for name, param in core.named_parameters() if "lora_" not in name)


def test_training_changes_the_adapter_and_it_survives_save_and_load(lora_model, tokenizer, prompts, tmp_path):
    import torch
    from safetensors.torch import load_file

    wrapped, before = lora_model
    example = _example(prompts)
    initial = lora_state(wrapped)
    assert weight_change_report(wrapped, initial)["total"] == 0.0
    optimizer = make_optimizer(wrapped, lr=1e-4)             # the SFT learning rate
    try:
        wrapped.train()
        sft_gradient(wrapped, [example], pad_id=tokenizer.pad_token_id, micro_batch=1)
        grad_norm = clip_and_step(wrapped, optimizer, 1.0)
        wrapped.eval()
        change = weight_change_report(wrapped, initial)
        trained = _logprobs(wrapped, tokenizer, example)
        print(f"\ngradient norm {grad_norm:.4f}; weight change per block {change['by_block']}; "
              f"mean log-probability {before.mean().item():.4f} -> {trained.mean().item():.4f}")
        assert sorted(change["by_block"]) == sorted(BLOCKS)
        assert all(value > 0 for value in change["by_block"].values()) and change["total"] > 0
        assert (trained - before).abs().mean().item() > 1e-3            # the adapter now has an effect
        assert trained.mean().item() > before.mean().item()             # a step on a text makes it likelier

        folder = tmp_path / "adapter"
        save_adapter(wrapped, folder)
        saved = load_file(str(folder / "adapter_model.safetensors"))
        n_wrapped = sum(1 for row in coverage_table(wrapped) if row["wrapped"])
        assert len(saved) == 2 * n_wrapped
        assert all(re.fullmatch(r"base_model\.model\.model\.layers\.\d+\.\w+\.\w+\.lora_[AB]\.weight", key)
                   for key in saved), sorted(saved)[:3]
        n_saved_numbers = sum(value.numel() for value in saved.values())
        assert os.path.getsize(folder / "adapter_model.safetensors") < 4.2 * n_saved_numbers     # float32
        config = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
        assert config["target_modules"] == target_pattern()
        assert config["r"] == RANK and config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.0 and config["bias"] == "none"

        # A second copy of the model with the saved adapter gives the trained model's numbers.
        name = os.environ.get("ACCESSDESK_TEST_MODEL", "Qwen/Qwen3.5-2B")
        for trainable in (False, True):
            loaded, _ = load_model(name, adapter_dir=str(folder), adapter_trainable=trainable)
            try:
                reloaded = _logprobs(loaded, tokenizer, example)
                assert (reloaded - trained).abs().mean().item() < 1e-3
                assert lora_fingerprint(loaded) == pytest.approx(lora_fingerprint(wrapped), rel=1e-6)
                assert coverage_problems(coverage_table(loaded), causal_lm(loaded).config) == []
                if trainable:
                    assert len(trainable_parameters(loaded)) == 2 * n_wrapped
                else:
                    with pytest.raises(RuntimeError):
                        trainable_parameters(loaded)
            finally:
                del loaded
                torch.cuda.empty_cache()
    finally:
        optimizer.zero_grad(set_to_none=True)
        wrapped.eval()
        with torch.no_grad():                 # put the adapter back to its start values for the other tests
            for name, param in causal_lm(wrapped).named_parameters():
                if name in initial:
                    param.copy_(initial[name].to(param.device))
    assert weight_change_report(wrapped, initial)["total"] == 0.0
    assert ADAPTER == "default"
