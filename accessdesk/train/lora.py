"""LoRA on the decoder layers of Qwen3.5: the set-up, the coverage table, and two reports.

Qwen3.5 has two kinds of decoder layer, and each kind has its own projections (transformers 5.23.0,
models/qwen3_5/modeling_qwen3_5.py):

    layer type          block        linear projections of the block                          source lines
    linear_attention    linear_attn  in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj   535-542
    full_attention      self_attn    q_proj, k_proj, v_proj, o_proj                           754-765
    both                mlp          gate_proj, up_proj, down_proj                            823-825

A layer has `linear_attn` or `self_attn`, never both (lines 857-863), and `mlp` always. Every one of these
projections is CALLED as a module in the forward pass (lines 563-570 and 655; 781-813; 829), so a wrapper put
around it takes part in the computation. (peft refuses `out_proj` for Mamba models, whose fast path reads the
weight directly; `qwen3_5_text` is not on that list: peft tuners_utils.py lines 145-164.) The layers are
`Qwen3_5ForCausalLM.model.layers` (lines 1660 and 1209), so the full name of a projection is, for example,
`model.layers.3.self_attn.q_proj`. Outside the layers there are two more weight matrices: `model.embed_tokens`
and `lm_head` (lines 1208 and 1662; they share one tensor). Neither may get an adapter.

How peft 0.21.3 picks the modules (peft/tuners/tuners_utils.py). A LIST of names is matched by suffix (line
2376), so "out_proj" would match a module of that name anywhere in the model. A STRING is one regular
expression that must match the whole module name (`re.fullmatch`, line 2371 and peft/utils/other.py line
1521). `target_pattern` builds such an expression. It starts with `model\\.layers\\.<number>\\.`, so it cannot
match a module outside the decoder layers, and it names the block of every projection, so `out_proj` can only
be the one of `linear_attn`. `coverage_table` then lists what was really wrapped, and `assert_coverage`
checks it.

What `get_peft_model` does to the model (peft/mapping_func.py line 105, tuners_utils.py lines 799-1153):
  - it changes the model IN PLACE: each matched `nn.Linear` is replaced by a wrapper that holds the old layer
    as `base_layer` and two new small layers `lora_A.default` and `lora_B.default` (lora/layer.py lines
    263-264);
  - it sets `requires_grad = False` on every parameter that is not an adapter weight (lines 506-523), so after
    it only the LoRA weights are trainable;
  - `lora_B` starts at zero (lora/layer.py line 343), so the adapter changes nothing until it is trained;
  - the adapter weights are float32 although the model is bfloat16 (`autocast_adapter_dtype=True`, line 2705).
    This matters: a bfloat16 number has about three decimal digits, so an update of 1e-5 would be lost.
    `trainable_parameters` raises if an adapter weight is not float32.
The wrapper's forward is `base(x) + lora_B(lora_A(x)) * (alpha / rank)`, cast back to the dtype of `base(x)`
(lora/layer.py lines 1087-1113; the scale is set in line 281).

torch and peft are imported inside the functions. `target_pattern` and `coverage_problems` need neither.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from accessdesk.harness.hf_backend import causal_lm

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                   "gate_proj", "up_proj", "down_proj")
GATE_TARGETS = ("in_proj_a", "in_proj_b")     # the two small gate projections; left out by default

# The block of a decoder layer in which each projection lives (see the table in the module docstring).
BLOCK_OF_TARGET = {
    "q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
    "in_proj_qkv": "linear_attn", "in_proj_z": "linear_attn", "out_proj": "linear_attn",
    "in_proj_a": "linear_attn", "in_proj_b": "linear_attn",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
}
# The token-mixing block of each layer type (`config.layer_types`; modeling_qwen3_5.py lines 858-862).
BLOCK_OF_LAYER_TYPE = {"linear_attention": "linear_attn", "full_attention": "self_attn"}
# The projections that must have an adapter, per block. `assert_coverage` checks exactly these.
REQUIRED = {
    "linear_attn": ("in_proj_qkv", "in_proj_z", "out_proj"),
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}
ADAPTER = "default"                           # peft's name for the first adapter of a model

# "model.layers.<layer>.<block>.<module>": a direct child of a block of a decoder layer.
_IN_LAYER = re.compile(r"model\.layers\.(\d+)\.(\w+)\.(\w+)")
# The same, followed by the name that peft gives to an adapter weight.
_LORA_WEIGHT = re.compile(r"model\.layers\.(\d+)\.(\w+)\.(\w+)\.lora_([AB])\." + ADAPTER + r"\.weight")


# ---------------------------------------------------------------------------
# Set-up
# ---------------------------------------------------------------------------
def target_pattern(targets: Iterable[str] = DEFAULT_TARGETS) -> str:
    """The regular expression that peft matches against whole module names (`re.fullmatch`).

    For the default targets it is, on one line and without the spaces:
        model\\.layers\\.\\d+\\.(?: linear_attn\\.(?:in_proj_qkv|in_proj_z|out_proj)
                               | mlp\\.(?:gate_proj|up_proj|down_proj)
                               | self_attn\\.(?:q_proj|k_proj|v_proj|o_proj) )
    Raises ValueError for a name that is not a projection of a decoder layer, for a
    name given twice, and for an empty list.
    """
    names = list(targets)
    unknown = sorted(set(names) - set(BLOCK_OF_TARGET))
    if unknown:
        raise ValueError(f"not a projection of a decoder layer: {unknown}; known: {sorted(BLOCK_OF_TARGET)}")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"the targets must be a non-empty list without repeats, got {names}")
    by_block: dict[str, list[str]] = {}
    for name in names:
        by_block.setdefault(BLOCK_OF_TARGET[name], []).append(name)
    blocks = [rf"{block}\.(?:{'|'.join(by_block[block])})" for block in sorted(by_block)]
    return rf"model\.layers\.\d+\.(?:{'|'.join(blocks)})"


def add_lora(model: Any, *, rank: int = 16, alpha: int = 32, targets: Iterable[str] = DEFAULT_TARGETS,
             seed: int = 0) -> Any:
    """Add LoRA adapters to the decoder layers of `model` and return the peft wrapper.

    `model` is the bare `Qwen3_5ForCausalLM` (as `load_model` returns it without an adapter). It is changed in
    place; the returned wrapper holds the same object, and `causal_lm(wrapper)` gives it back.
    Dropout is 0 and no bias is trained. `lora_A` is drawn from torch's random generator, which is seeded with
    `seed` here; `lora_B` is zero. After the call only the adapter weights are trainable.

    Call this before gradient checkpointing is switched on.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    if causal_lm(model) is not model:
        raise ValueError("add_lora takes the bare model; this one already has a peft wrapper")
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=target_pattern(targets),     # one regular expression: see the module docstring
        init_lora_weights=True,                     # lora_A random, lora_B zero
        use_rslora=False, use_dora=False,           # plain LoRA: the scale is alpha / rank
        task_type="CAUSAL_LM",
    )
    torch.manual_seed(seed)
    wrapped = get_peft_model(model, config, adapter_name=ADAPTER, autocast_adapter_dtype=True)
    trainable_parameters(wrapped)                   # raises if anything is not as stated above
    return wrapped


def trainable_parameters(model: Any) -> list[Any]:
    """The parameters that the optimiser gets: every parameter with `requires_grad`, in model order.

    Raises RuntimeError if there is none, if one of them is not a LoRA weight of a decoder layer, or if one
    of them is not float32. So a base weight can never be trained by mistake.
    """
    import torch

    named = [(name, param) for name, param in causal_lm(model).named_parameters() if param.requires_grad]
    if not named:
        raise RuntimeError("the model has no trainable parameter: add_lora was not called, or the adapter "
                           "was loaded frozen")
    not_lora = [name for name, _ in named if _LORA_WEIGHT.fullmatch(name) is None]
    if not_lora:
        raise RuntimeError(f"{len(not_lora)} trainable parameters are not LoRA weights of a decoder layer, "
                           f"for example {not_lora[0]!r}")
    not_float32 = [name for name, param in named if param.dtype != torch.float32]
    if not_float32:
        raise RuntimeError(f"{len(not_float32)} LoRA weights are not float32, for example {not_float32[0]!r}")
    return [param for _, param in named]


def save_adapter(model: Any, directory: str | Path) -> None:
    """Write the adapter weights and their config into `directory` (peft's own format).

    `load_model(model_id, adapter_dir=directory)` reads it back. Only the LoRA weights are written.
    `save_embedding_layers=False` is given on purpose: peft's default "auto" asks the Hub for the base model's
    config at every save and can decide to add the whole embedding matrix to the file (peft
    utils/save_and_load.py lines 292-335).
    """
    if causal_lm(model) is model:
        raise TypeError("save_adapter needs the peft wrapper that add_lora or load_model returned")
    model.save_pretrained(str(directory), safe_serialization=True, save_embedding_layers=False)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def coverage_table(model: Any) -> list[dict]:
    """One row for every linear layer of the model, and for every module that has an adapter.

    Row keys:
        path         the module name inside the model, for example "model.layers.3.self_attn.q_proj"
        layer        index of the decoder layer, or None for a module that is not a projection of one
        layer_type   "linear_attention" or "full_attention", read from the layer itself (`block_type`);
                     None outside the decoder layers
        block        "linear_attn", "self_attn" or "mlp"; else None
        module       the last part of the path
        wrapped      True if the module has a LoRA adapter
        shape        [out_features, in_features] of the original weight
        lora_params  number of adapter weights (0 if not wrapped); for rank r it is r * (in + out)
    Rows are in model order. The projections that are NOT wrapped are in the table too (`wrapped` False), and
    so is `lm_head`. The inside of a wrapper (its `base_layer`, `lora_A`, `lora_B`) gives no rows of its own.
    """
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer

    core = causal_lm(model)
    rows: list[dict] = []
    inside = None                                    # the wrapper whose inner modules are being skipped
    for path, module in core.named_modules():        # parents come before their children
        if inside is not None and path.startswith(inside + "."):
            continue
        wrapped = isinstance(module, BaseTunerLayer)
        if wrapped:
            inside = path
        elif not isinstance(module, torch.nn.Linear):
            continue
        base = module.get_base_layer() if wrapped else module
        match = _IN_LAYER.fullmatch(path)
        layer = int(match.group(1)) if match else None
        rows.append({
            "path": path, "layer": layer,
            "layer_type": core.model.layers[layer].block_type if match else None,
            "block": match.group(2) if match else None,
            "module": path.rpartition(".")[2],
            "wrapped": wrapped,
            "shape": list(base.weight.shape),
            "lora_params": sum(param.numel() for name, param in module.named_parameters()
                               if name.startswith("lora_")) if wrapped else 0,
        })
    return rows


def coverage_problems(table: list[dict], config: Any) -> list[str]:
    """Everything that is wrong with a coverage table, as sentences; an empty list if all is well.

    `config` is the model's config; only `config.layer_types` is read. Checked:
      1. the table has every decoder layer of the config, and each layer has the type that the config gives it
         (so the number of layers of each type is the config's);
      2. every linear_attention layer has in_proj_qkv, in_proj_z and out_proj wrapped; every full_attention
         layer has q_proj, k_proj, v_proj and o_proj wrapped; every layer has gate_proj, up_proj and down_proj
         wrapped (in_proj_a and in_proj_b may be wrapped or not);
      3. nothing outside the decoder layers is wrapped, and inside them only known projections are wrapped,
         each with at least one adapter weight.
    """
    problems: list[str] = []
    layer_types = list(config.layer_types)
    type_of_layer = {row["layer"]: row["layer_type"] for row in table if row["layer"] is not None}
    wrapped = {(row["layer"], row["block"], row["module"]) for row in table if row["wrapped"]}

    if sorted(type_of_layer) != list(range(len(layer_types))):
        problems.append(f"the table has the decoder layers {sorted(type_of_layer)}, but the config has "
                        f"{len(layer_types)} layers")
    if Counter(type_of_layer.values()) != Counter(layer_types):
        problems.append(f"layers per type: {dict(Counter(type_of_layer.values()))} in the model, "
                        f"{dict(Counter(layer_types))} in the config")
    for layer, layer_type in enumerate(layer_types):
        if layer in type_of_layer and type_of_layer[layer] != layer_type:
            problems.append(f"layer {layer} is {type_of_layer[layer]!r} in the model but {layer_type!r} in "
                            "the config")
        if layer_type not in BLOCK_OF_LAYER_TYPE:
            problems.append(f"layer {layer} has the unknown type {layer_type!r}")
            continue
        for block in (BLOCK_OF_LAYER_TYPE[layer_type], "mlp"):
            for name in REQUIRED[block]:
                if (layer, block, name) not in wrapped:
                    problems.append(f"layer {layer} ({layer_type}): {block}.{name} has no adapter")
    for row in table:
        if not row["wrapped"]:
            continue
        if row["layer"] is None:
            problems.append(f"{row['path']} has an adapter but is not a projection of a decoder layer")
        elif BLOCK_OF_TARGET.get(row["module"]) != row["block"]:
            problems.append(f"{row['path']} has an adapter but is not a known projection")
        if row["lora_params"] <= 0:
            problems.append(f"{row['path']} is wrapped but has no adapter weight")
    return problems


def assert_coverage(table: list[dict], config: Any) -> None:
    """Raise AssertionError with every problem that `coverage_problems` finds."""
    problems = coverage_problems(table, config)
    if problems:
        shown = "\n  ".join(problems[:20])
        raise AssertionError(f"LoRA coverage: {len(problems)} problems\n  {shown}")


def coverage_summary(table: list[dict]) -> list[dict]:
    """The table folded over the layers: one row per (layer type, block, module), for printing.

    Keys: layer_type, block, module, wrapped (how many layers have the adapter), of (how many layers have the
    module), shape, lora_params (the sum over the layers). Modules outside the decoder layers come last, with
    layer_type and block None.
    """
    groups: dict[tuple, dict] = {}
    for row in table:
        key = (row["layer_type"], row["block"], row["module"])
        group = groups.setdefault(key, {"layer_type": row["layer_type"], "block": row["block"],
                                        "module": row["module"], "wrapped": 0, "of": 0,
                                        "shape": row["shape"], "lora_params": 0})
        group["wrapped"] += int(row["wrapped"])
        group["of"] += 1
        group["lora_params"] += row["lora_params"]
    return [groups[key] for key in sorted(groups, key=lambda k: (k[0] is None, str(k[0]), str(k[1]), k[2]))]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def gradient_report(model: Any) -> dict:
    """After one backward pass: how large the gradient of the LoRA weights is, per layer type and per module.

    A "norm" here is the L2 norm of all the gradients of a group taken together: the square root of the sum
    of their squared entries. Only `lora_B` gradients are in the groups:
        by_layer_type   {"linear_attention": .., "full_attention": ..}: every adapter of the layers of that
                        type, the MLP adapters included
        by_block        {"linear_attn": .., "self_attn": .., "mlp": ..}
        by_module       {"q_proj": .., "in_proj_qkv": .., ...}
        lora_B          the norm over all of them
        lora_A          the norm of all `lora_A` gradients
        without_gradient  names of LoRA weights whose `.grad` is None (the first 5) and their number

    At the first step `lora_A` is exactly 0, and that is correct: `lora_B` starts at zero, the adapter's
    output is `lora_B(lora_A(x))`, so its derivative with respect to `lora_A` has the factor `lora_B` = 0.
    The gradient that shows that an adapter is connected to the loss is therefore the one of `lora_B`.
    A block or layer type with norm 0 here means that its adapters get no gradient.
    """
    core = causal_lm(model)
    squares: dict[str, dict[str, float]] = {"by_layer_type": {}, "by_block": {}, "by_module": {}}
    square_a = square_b = 0.0
    missing: list[str] = []
    for name, param in core.named_parameters():
        match = _LORA_WEIGHT.fullmatch(name)
        if match is None:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        square = float(param.grad.detach().float().pow(2).sum())
        if match.group(4) == "A":
            square_a += square
            continue
        square_b += square
        layer_type = core.model.layers[int(match.group(1))].block_type
        for key, group in (("by_layer_type", layer_type), ("by_block", match.group(2)),
                           ("by_module", match.group(3))):
            squares[key][group] = squares[key].get(group, 0.0) + square
    report: dict = {key: {group: math.sqrt(value) for group, value in sorted(groups.items())}
                    for key, groups in squares.items()}
    report.update({"lora_B": math.sqrt(square_b), "lora_A": math.sqrt(square_a),
                   "without_gradient": {"n": len(missing), "first": missing[:5]}})
    return report


def lora_state(model: Any) -> dict[str, Any]:
    """A copy on the CPU of every LoRA weight, by parameter name: what `weight_change_report` takes as
    `initial_state`."""
    return {name: param.detach().cpu().clone()
            for name, param in causal_lm(model).named_parameters() if _LORA_WEIGHT.fullmatch(name)}


def lora_fingerprint(model: Any) -> float:
    """One number for the current LoRA weights: the sum of their absolute values (in float64).

    Saved with a checkpoint and compared at a restart, so that an optimiser state is never continued on other
    weights than the ones it was saved with. It is not a hash: it only has to tell two training states apart.
    """
    return sum(float(param.detach().double().abs().sum())
               for name, param in causal_lm(model).named_parameters() if _LORA_WEIGHT.fullmatch(name))


def weight_change_report(model: Any, initial_state: dict[str, Any]) -> dict:
    """After training: how far the weight change of the adapters has moved since `initial_state`.

    For one wrapped projection the adapter adds `scale * lora_B @ lora_A` to the weight (scale = alpha /
    rank). This function takes that matrix now, minus the same matrix built from `initial_state` (from
    `lora_state`), and reports the Frobenius norm of the difference, pooled like the groups of
    `gradient_report`:
    `by_layer_type`, `by_block`, `by_module`, and `total`. All zeros means that training changed nothing.
    """
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer

    core = causal_lm(model)
    squares: dict[str, dict[str, float]] = {"by_layer_type": {}, "by_block": {}, "by_module": {}}
    total = 0.0
    with torch.no_grad():
        for path, module in core.named_modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            match = _IN_LAYER.fullmatch(path)
            if match is None:
                raise RuntimeError(f"{path} has an adapter but is not a projection of a decoder layer")
            now_a = module.lora_A[ADAPTER].weight.float()
            now_b = module.lora_B[ADAPTER].weight.float()
            old_a = initial_state[f"{path}.lora_A.{ADAPTER}.weight"].to(now_a.device).float()
            old_b = initial_state[f"{path}.lora_B.{ADAPTER}.weight"].to(now_b.device).float()
            change = module.scaling[ADAPTER] * (now_b @ now_a - old_b @ old_a)       # [out, in]
            square = float(change.pow(2).sum())
            total += square
            layer_type = core.model.layers[int(match.group(1))].block_type
            for key, group in (("by_layer_type", layer_type), ("by_block", match.group(2)),
                               ("by_module", match.group(3))):
                squares[key][group] = squares[key].get(group, 0.0) + square
    report: dict = {key: {group: math.sqrt(value) for group, value in sorted(groups.items())}
                    for key, groups in squares.items()}
    report["total"] = math.sqrt(total)
    return report
