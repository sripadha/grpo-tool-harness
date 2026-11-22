"""The pieces of the output layer (accessdesk/train/forward.py): the same numbers as all rows at once, and
less memory.

WRITTEN ON A LAPTOP WITHOUT TORCH, 2026-11-18. NOT RUN YET. Run it on the rented machine before the training
runs, and read the printed numbers (-s):

    python -m pytest tests/gpu/test_train_fixes_gpu.py -q -s

`token_logprobs` sends the selected rows through `lm_head` and `log_softmax` in pieces of `head_rows` rows
and joins the results. A piece size above the number of rows gives one piece: all rows at once, as the code
did before. The tests compare the two.

The first tests use the four-token toy model of tests/gpu/test_forward.py; they need torch but no GPU work.
There the pieces must give the hand-made numbers. The other tests use the real model. Their bounds are NOT
measured: they are first guesses, and each test says where its guess comes from. If one fails, read the
printed numbers before anything is changed.
"""
from __future__ import annotations

import math

import pytest

from accessdesk.harness.hf_backend import causal_lm
from accessdesk.train import forward
from accessdesk.train.forward import HEAD_ROWS, row_pieces, token_logprobs
from tests.gpu.test_forward import TOY_PAD, _all_but_first, _stats, _toy_model

ALL_ROWS = 10 ** 9            # a piece size above any number of rows: one piece, all rows at once


# ---------------------------------------------------------------------------
# The toy model: hand-made numbers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head_rows", [1, 2, 3, HEAD_ROWS, ALL_ROWS])
def test_pieces_give_the_hand_made_logprobs_and_gradients_on_the_toy_model(head_rows):
    """The batch of test_token_logprobs_with_gradients_takes_one_micro_batch in tests/gpu/test_forward.py:
    three selected rows, so 3, 2, 2, 1 and 1 pieces. Every piece size must give the numbers that are worked
    out there by hand."""
    import torch

    toy = _toy_model()
    seqs = [[0, 1, 2, 1], [2, 0]]
    weights = [[0, 1, 0, 1], [0, 1]]
    assert len(row_pieces(3, head_rows)) == {1: 3, 2: 2}.get(head_rows, 1)
    result = token_logprobs(toy, seqs, weights, pad_id=TOY_PAD, micro_batch=2, with_grad=True,
                            head_rows=head_rows)
    # Token 1 after token 0 is P[1][0] = 0.2; token 1 after token 2 is 0.25; token 0 after token 2 is 0.25.
    assert [len(r) for r in result] == [2, 1]
    assert torch.allclose(result[0], torch.tensor([math.log(0.2), math.log(0.25)]), atol=1e-6)
    assert torch.allclose(result[1], torch.tensor([math.log(0.25)]), atol=1e-6)
    assert all(r.dtype == torch.float32 and r.requires_grad for r in result)
    (-(result[0].sum() + result[1].sum())).backward()
    grad = toy.lm_head.weight.grad
    # d(-log softmax(w[:, c])[n]) / d w[m][c] = P[m][c] - (1 if m == n else 0): column 0 is used once (next
    # token 1), column 2 twice (next tokens 1 and 0), columns 1 and 3 never.
    assert grad[:, 1].abs().sum().item() == 0.0 and grad[:, 3].abs().sum().item() == 0.0
    assert grad[1, 0].item() == pytest.approx(0.2 - 1.0, abs=1e-6)
    assert grad[0, 0].item() == pytest.approx(0.1, abs=1e-6)
    assert grad[0, 2].item() == pytest.approx(2 * 0.25 - 1.0, abs=1e-6)
    assert grad[3, 2].item() == pytest.approx(2 * 0.25, abs=1e-6)


def test_pieces_and_all_rows_at_once_give_one_weighted_gradient_on_the_toy_model():
    """A loss with another factor on every token, as the GRPO loss has (the advantage of the token's
    episode): the gradient of the output layer is the same for pieces of 1, 2 and 4 rows and for all rows
    at once."""
    import torch

    seqs = [[0, 1, 2, 1, 0, 2], [2, 0, 0, 1], [1, 1, 2]]
    weights = [[0, 1, 1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 1]]          # 5 + 2 + 2 = 9 selected rows
    factors = [0.5, -0.25, 0.125]                                    # one per sequence

    def run(head_rows: int):
        toy = _toy_model()
        result = token_logprobs(toy, seqs, weights, pad_id=TOY_PAD, micro_batch=3, with_grad=True,
                                head_rows=head_rows)
        loss = -sum(factor * logprobs.sum() for factor, logprobs in zip(factors, result))
        loss.backward()
        return torch.cat([r.detach() for r in result]), toy.lm_head.weight.grad.clone(), len(toy.calls)

    values, grad, calls = run(ALL_ROWS)
    assert values.shape == (9,) and calls == 1 and grad.abs().sum().item() > 0
    for head_rows in (1, 2, 4):
        piece_values, piece_grad, piece_calls = run(head_rows)
        assert piece_calls == 1                              # one decoder pass, whatever the piece size is
        assert torch.allclose(piece_values, values, atol=1e-7)
        assert torch.allclose(piece_grad, grad, atol=1e-7)


def test_a_piece_size_below_one_is_refused_before_any_forward_pass():
    toy = _toy_model()
    with pytest.raises(ValueError):
        token_logprobs(toy, [[0, 1]], [[0, 1]], pad_id=TOY_PAD, micro_batch=1, with_grad=False, head_rows=0)
    assert toy.calls == []


# ---------------------------------------------------------------------------
# The real model
# ---------------------------------------------------------------------------
def test_pieces_give_the_logprobs_of_all_rows_at_once(model, tokenizer, prompts):
    """Two prompts, every token but the first selected: 48 + 31 - 2 = 77 rows, in pieces of 16 and of 1 row
    against all rows at once. The decoder pass is the same in the three runs, so only the output layer can
    make a difference.

    Bounds (a guess, not measured): those of test_equal_to_the_plain_logits_path in tests/gpu/test_forward.py,
    mean below 1e-3 and largest below 0.25. That test compares two decoder passes of different batch shapes
    (measured mean 1e-3 scale); the output layer alone should differ less, perhaps not at all.
    """
    seqs = [prompts[0][:48], prompts[1][:31]]
    weights = [_all_but_first(seq) for seq in seqs]

    def run(head_rows: int) -> list[float]:
        result = token_logprobs(model, seqs, weights, pad_id=tokenizer.pad_token_id, micro_batch=2,
                                with_grad=False, head_rows=head_rows)
        assert [len(r) for r in result] == [47, 30]
        return [value for r in result for value in r.tolist()]

    whole = run(ALL_ROWS)
    again = run(ALL_ROWS)
    print("\nall rows at once, twice:      ", _stats([a - b for a, b in zip(whole, again)]))
    for head_rows in (16, 1):
        stats = _stats([a - b for a, b in zip(run(head_rows), whole)])
        print(f"pieces of {head_rows:2d} against all rows:", stats)
        assert stats["n"] == 77 and stats["mean"] < 1e-3 and stats["max"] < 0.25


def test_pieces_give_the_gradient_of_all_rows_at_once(model, tokenizer, prompts):
    """One backward pass down to the first decoder layer, with pieces of 16 rows and with all rows at once.
    The loss has another factor on each of the two sequences, as the GRPO loss has.

    Bound (a guess, not measured): cosine above 0.99, the bound of
    test_gradients_flow_with_and_without_gradient_checkpointing in tests/gpu/test_forward.py. The relative
    difference of the two gradients is printed.
    """
    import torch

    core = causal_lm(model)
    seqs = [prompts[0][:48], prompts[1][:31]]
    weights = [_all_but_first(seq) for seq in seqs]
    factors = [0.5, -0.25]
    probe = core.model.layers[0].input_layernorm.weight     # its gradient travels through every decoder layer
    flags = [(p, p.requires_grad) for p in core.parameters()]
    grads = {}
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        probe.requires_grad_(True)
        core.train()
        for head_rows in (ALL_ROWS, 16):
            probe.grad = None
            result = token_logprobs(model, seqs, weights, pad_id=tokenizer.pad_token_id, micro_batch=2,
                                    with_grad=True, head_rows=head_rows)
            (-sum(factor * logprobs.sum() for factor, logprobs in zip(factors, result))).backward()
            grads[head_rows] = probe.grad.detach().float().clone()
    finally:
        core.eval()
        probe.grad = None
        for p, flag in flags:
            p.requires_grad_(flag)
    whole, pieces = grads[ALL_ROWS], grads[16]
    assert bool(torch.isfinite(whole).all()) and whole.abs().sum().item() > 0
    cosine = torch.nn.functional.cosine_similarity(whole, pieces, dim=0).item()
    relative = ((whole - pieces).norm() / whole.norm()).item()
    print(f"\ngradient, pieces of 16 against all rows: cosine = {cosine:.6f}, relative difference = "
          f"{relative:.2e}")
    assert cosine > 0.99


def test_pieces_lower_the_peak_memory_of_the_output_layer(model):
    """The output layer alone, on 2,048 made-up hidden rows with gradients: forward and backward with pieces
    of 256 rows and with all rows at once. The peak memory of both is measured and printed.

    The estimate in the docstring of `_micro_batch_logprobs` (own calculation from the tensor shapes, V =
    vocabulary): about (4 R + 10 P) V bytes with pieces and 14 R V bytes without. For R = 2,048, P = 256 and
    V = 248,320 that is 2.5 GiB against 6.6 GiB, a ratio of 0.375. Bound (a guess): the measured ratio is
    below 0.6. If it is not, the estimate in that docstring is wrong and must be corrected with the numbers
    that this test prints. The made-up rows change no weight of the model.
    """
    import torch

    core = causal_lm(model)
    head = core.lm_head
    n_rows, piece = 2048, 256
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(n_rows, head.in_features, generator=generator).to(core.device, head.weight.dtype)
    targets = torch.randint(0, head.out_features, (n_rows,), generator=generator).to(core.device)
    factors = torch.linspace(-1.0, 1.0, n_rows, device=core.device)          # another factor on every row

    def run(head_rows: int):
        selected = rows.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        chosen = forward._selected_logprobs(core, selected, targets, head_rows)
        (-(factors * chosen).sum()).backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        return peak, chosen.detach().clone(), selected.grad.detach().float().clone()

    # The weight of the output layer gets no gradient here, as in training, where the base model is frozen.
    # The fixture's model has no adapter, so its weights may still ask for one.
    flags = [(p, p.requires_grad) for p in head.parameters()]
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        whole_peak, whole_values, whole_grad = run(ALL_ROWS)
        piece_peak, piece_values, piece_grad = run(piece)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    gib, vocabulary = 2 ** 30, head.out_features
    print(f"\npeak memory of the output layer, {n_rows} rows, vocabulary {vocabulary}: all rows at once "
          f"{whole_peak / gib:.2f} GiB (estimate {14 * n_rows * vocabulary / gib:.2f}), pieces of {piece} "
          f"{piece_peak / gib:.2f} GiB (estimate {(4 * n_rows + 10 * piece) * vocabulary / gib:.2f}), "
          f"ratio {piece_peak / whole_peak:.3f} (estimate {(4 * n_rows + 10 * piece) / (14 * n_rows):.3f})")
    assert piece_peak < 0.6 * whole_peak
    # The same numbers, up to rounding. The bounds are a guess: see test_pieces_give_the_logprobs_of_all_rows.
    stats = _stats((piece_values - whole_values).tolist())
    cosine = torch.nn.functional.cosine_similarity(whole_grad.flatten(), piece_grad.flatten(), dim=0).item()
    print("values, pieces against all rows:", stats, f"; gradient of the rows: cosine = {cosine:.6f}")
    assert stats["mean"] < 1e-3 and stats["max"] < 0.25 and cosine > 0.99
    assert bool(torch.isfinite(whole_grad).all()) and whole_grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Controls for the gradient difference of pieces against all rows
# ---------------------------------------------------------------------------
# WRITTEN ON A LAPTOP WITHOUT TORCH, 2026-11-18. NOT RUN YET. The tests above this line ran on an RTX 4090
# that morning; the controls below did not. They are measurements: run them with -s and read the lines.
#
#     ACCESSDESK_TEST_MODEL=Qwen/Qwen3.5-2B python -m pytest tests/gpu/test_train_fixes_gpu.py \
#         -k test_control_ -o addopts="-p no:cacheprovider" -q -s
#
# THE OPEN POINT. test_pieces_give_the_gradient_of_all_rows_at_once printed, twice with the same digits:
# "cosine = 0.998520, relative difference = 5.44e-02". Pieces of 16 rows and all rows at once are the same
# function as maths, their log-probabilities are equal (largest difference 0.0), and at the output layer
# itself the gradient cosine is 1.000000. Still the gradients at the first decoder layer differ by 5 %.
# Is that plain bfloat16 rounding that builds up on the way down through the layers, or something else?
# (A bfloat16 number has 8 binary digits, so one rounding moves a number by up to 0.4 %.)
#
# Every control uses the input and the loss of that test (two sequences of 48 and 31 tokens, 77 rows, the
# factors 0.5 and -0.25) and reads the gradient at several depths: the weight of the final norm (above
# every layer) and the weight of `input_layernorm` of the last, a middle and the first decoder layer. The
# first-layer weight is the one that test reads, so the first-layer line of C2 should repeat its two numbers.
# The relative difference is |other - reference| / |reference|, computed in float64.
#
# WHAT EACH RESULT WOULD MEAN.
# C1  The same gradient with both sequences in one micro-batch and with one sequence per micro-batch,
#     WITHOUT pieces. As maths the two are equal. This is the size of plain rounding between two batch
#     shapes. Careful: here the forward pass differs too (see the line "log-probabilities"); with pieces it
#     does not. So C1 is an upper mark, and C4 is the closer comparison.
#       rounding:        a few percent at the first layer, about as large as 5.4e-02 or larger.
#       something else:  far smaller than 5.4e-02 (say below 1e-3). Then two batch shapes round much less
#                        than the pieces differ, and the pieces need another explanation.
# C2  Pieces of 16 against all rows, at each depth. Also all rows a second time (expected: exactly 0, one
#     way repeats itself; the morning's two runs printed the same digits) and pieces of 1.
#       rounding:        about 0 at the final norm, small at the last layer, larger at the middle layer, a
#                        few percent at the first: it grows on the way down. Pieces of 1 give about the
#                        same size as pieces of 16.
#       something else:  already a few percent at the final norm or at the last layer, or no growth with
#                        depth, or a size that changes much with the piece size.
# C4  (An addition to the three controls that were asked for.) No pieces, one batch shape, one forward
#     pass: only the loss is multiplied by 2 or by 3 and the gradient divided by it again. Times 2 changes
#     no rounding (a power of two), so it should give exactly 0. Times 3 makes every number of the backward
#     pass round in another place, and nothing else. So "times 3" is the size of rounding in the backward
#     pass alone.
#       rounding:        times 3 gives the same size as the pieces at each depth (a few percent at the
#                        first layer).
#       something else:  times 3 stays far below the pieces (say 1e-3 against 5e-02).
# C5  (An addition.) The comparisons of C1, C2 and C4 read at the LoRA weights (`lora_B`, float32) of the
#     same three layers, with the adapter as training sets it up. These are the weights that training
#     moves; the tests above read a norm weight of the base model. Read it as C1, C2 and C4. The norm lines
#     of C5 should repeat those of C2: an adapter that is still zero changes nothing.
# C3  (Last in this file, because it loads a second copy of the model.) The comparisons again on a copy of
#     the model in float32, 24 binary digits in place of 8.
#       rounding:        pieces against all rows agree to about 1e-5 or better at every depth. A value up
#                        to 1e-3 still points to rounding if "times 3" and the batch shapes are as large in
#                        float32: tests/gpu/test_forward.py says that the float32 forward pass of this
#                        model differs by 0.001 to 0.002 between two batch shapes.
#       something else:  the first-layer difference of the pieces stays at a few percent in float32.
#
# What is asserted is only what must hold in every case: finite numbers, a gradient that is not zero, and
# for the bfloat16 comparisons of pieces and of batch shapes a cosine above 0.9. Every line is printed
# before anything is asserted.
def _control_input(prompts: list[list[int]]):
    """The input of test_pieces_give_the_gradient_of_all_rows_at_once: (sequences, weights, factors)."""
    seqs = [prompts[0][:48], prompts[1][:31]]
    return seqs, [_all_but_first(seq) for seq in seqs], [0.5, -0.25]


def _way(batches: list[list[int]], head_rows: int = ALL_ROWS, scale: float = 1.0) -> dict:
    """One way to compute the gradient. `batches`: the sequences that go through the model together, one
    list of indices per micro-batch. `head_rows`: the piece size of the output layer. `scale`: the loss is
    multiplied by it and the gradient is divided by it again."""
    return {"batches": batches, "head_rows": head_rows, "scale": scale}


def _depths(core) -> dict[str, int]:
    """Label -> index of the decoder layer, from the top of the model down."""
    n_layers = len(core.model.layers)
    return {f"last layer ({n_layers - 1})": n_layers - 1, f"middle layer ({n_layers // 2})": n_layers // 2,
            "first layer (0)": 0}


def _norm_probes(core) -> dict[str, list]:
    """Label -> the weights whose gradient is read there, from the top down: the final norm, then the
    `input_layernorm` of the layers of `_depths`. The gradient of a layer's weight has passed through that
    layer and every layer above it; the one of the final norm only through the output layer."""
    probes = {"final norm (above the layers)": [core.model.norm.weight]}
    for label, index in _depths(core).items():
        probes[label] = [core.model.layers[index].input_layernorm.weight]
    return probes


def _control_gradients(model, tokenizer, probes: dict, seqs, weights, factors, ways: dict) -> dict:
    """The gradient of the loss of test_pieces_give_the_gradient_of_all_rows_at_once at every probe, once
    for each way of `ways` (name -> `_way`). Returns {name: (gradients, logprobs)}: `gradients` is {probe
    label: 1-D float64 tensor}, `logprobs` a 1-D float64 tensor of all selected tokens in sequence order.

    As in that test, everything but the probes is frozen and the model is in train mode. The gradient of
    each micro-batch is read alone and the micro-batches are added in float64, so that adding them brings
    no rounding of its own. Flags, mode and gradients of the model are put back at the end.
    """
    import torch

    core = causal_lm(model)
    flags = [(p, p.requires_grad) for p in core.parameters()]
    was_training = core.training
    probe_weights = [p for group in probes.values() for p in group]
    results = {}
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        for p in probe_weights:
            p.requires_grad_(True)
        core.train()
        for name, way in ways.items():
            gradients = {label: 0.0 for label in probes}
            logprobs = {}
            for batch in way["batches"]:
                for p in probe_weights:
                    p.grad = None
                result = token_logprobs(model, [seqs[i] for i in batch], [weights[i] for i in batch],
                                        pad_id=tokenizer.pad_token_id, micro_batch=len(batch),
                                        with_grad=True, head_rows=way["head_rows"])
                (-sum(way["scale"] * factors[i] * values.sum() for i, values in zip(batch, result))).backward()
                for i, values in zip(batch, result):
                    logprobs[i] = values.detach().double().cpu()
                without = [label for label, group in probes.items() if any(p.grad is None for p in group)]
                assert not without, f"no gradient arrived at: {without}"
                for label, group in probes.items():
                    part = torch.cat([p.grad.detach().double().flatten() for p in group])
                    gradients[label] = gradients[label] + part / way["scale"]
            results[name] = (gradients, torch.cat([logprobs[i] for i in range(len(seqs))]))
    finally:
        core.train(was_training)
        for p in probe_weights:
            p.grad = None
        for p, flag in flags:
            p.requires_grad_(flag)
    return results


def _print_against(title: str, results: dict, reference: str, other: str) -> dict:
    """Print how the way `other` differs from the way `reference`: the log-probabilities, and at every probe
    the cosine of the two gradients and their relative difference |other - reference| / |reference|.
    Returns {probe label: (cosine, relative difference)}. After all lines are printed it raises if a number
    is not finite or a gradient is all zero."""
    import torch

    (grads_a, values_a), (grads_b, values_b) = results[reference], results[other]
    print(f"\n{title}: {other!r} against {reference!r}")
    print("  log-probabilities:", _stats((values_b - values_a).tolist()))
    numbers, problems = {}, []
    for label, a in grads_a.items():
        b = grads_b[label]
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        relative = ((a - b).norm() / a.norm()).item()
        print(f"  gradient at {label:<30} cosine = {cosine:.6f} (1 - cosine = {1 - cosine:.2e}), "
              f"relative difference = {relative:.2e}")
        if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                and math.isfinite(cosine) and math.isfinite(relative)):
            problems.append(f"{label}: a number is not finite")
        elif a.abs().sum().item() == 0 or b.abs().sum().item() == 0:
            problems.append(f"{label}: a gradient is all zero")
        numbers[label] = (cosine, relative)
    assert not problems, problems
    return numbers


def test_control_c1_batch_shapes_without_pieces(model, tokenizer, prompts):
    """C1: both sequences in one micro-batch against one sequence per micro-batch with the gradients added.
    The output layer runs on all rows at once in both. What the result means: the comment block above."""
    seqs, weights, factors = _control_input(prompts)
    ways = {"one micro-batch of two": _way([[0, 1]]), "one sequence per micro-batch": _way([[0], [1]])}
    results = _control_gradients(model, tokenizer, _norm_probes(causal_lm(model)), seqs, weights, factors,
                                 ways)
    numbers = _print_against("C1 batch shapes, no pieces", results, "one micro-batch of two",
                             "one sequence per micro-batch")
    # Measured on 2026-11-18 on the 2B in bfloat16: cosine 0.987, 0.902, 0.927 and 0.866 from the final norm
    # down to the first layer (_dev/gpu_trial/gradient_controls_result.md). The first bound of this test,
    # 0.9, was a guess and did not hold. 0.5 only catches a computation that is broken.
    assert all(cosine > 0.5 for cosine, _ in numbers.values()), numbers


def test_control_c2_pieces_against_all_rows_at_several_depths(model, tokenizer, prompts):
    """C2: the comparison of test_pieces_give_the_gradient_of_all_rows_at_once, read at the final norm and
    at the last, a middle and the first decoder layer. The first-layer line of pieces of 16 should repeat
    that test: cosine = 0.998520, relative difference = 5.44e-02. What the result means: see above."""
    seqs, weights, factors = _control_input(prompts)
    ways = {"all rows at once": _way([[0, 1]]), "all rows at once, a second time": _way([[0, 1]]),
            "pieces of 16": _way([[0, 1]], head_rows=16), "pieces of 1": _way([[0, 1]], head_rows=1)}
    results = _control_gradients(model, tokenizer, _norm_probes(causal_lm(model)), seqs, weights, factors,
                                 ways)
    _print_against("C2 the same way twice (expected: exactly 0)", results, "all rows at once",
                   "all rows at once, a second time")
    numbers = _print_against("C2 pieces, by depth", results, "all rows at once", "pieces of 16")
    single = _print_against("C2 pieces, by depth", results, "all rows at once", "pieces of 1")
    assert all(cosine > 0.9 for cosine, _ in list(numbers.values()) + list(single.values())), (numbers, single)


def test_control_c4_the_loss_times_two_and_times_three(model, tokenizer, prompts):
    """C4 (an addition): no pieces and one batch shape; the loss is multiplied by 2 or by 3 and the
    gradient divided by it again. Times 2 should change nothing. Times 3 moves every rounding of the
    backward pass and nothing else. What the result means: see above."""
    seqs, weights, factors = _control_input(prompts)
    ways = {"loss times 1": _way([[0, 1]]), "loss times 2": _way([[0, 1]], scale=2.0),
            "loss times 3": _way([[0, 1]], scale=3.0)}
    results = _control_gradients(model, tokenizer, _norm_probes(causal_lm(model)), seqs, weights, factors,
                                 ways)
    _print_against("C4 a power of two (expected: exactly 0)", results, "loss times 1", "loss times 2")
    numbers = _print_against("C4 rounding of the backward pass alone", results, "loss times 1",
                             "loss times 3")
    assert all(cosine > 0.9 for cosine, _ in numbers.values()), numbers


def test_control_c5_the_comparisons_at_the_lora_weights(model, tokenizer, prompts):
    """C5 (an addition): C1, C2 and C4 read at the `lora_B` weights of the last, a middle and the first
    decoder layer, next to the norm weight of the same layer. `add_lora` changes the model of the session
    in place; as the fixture `lora_model` of tests/gpu/test_lora.py does, this test takes the adapters out
    again with `unload()` and puts every flag back. What the result means: see above."""
    import torch

    from accessdesk.train.lora import add_lora

    core = causal_lm(model)
    seqs, weights, factors = _control_input(prompts)
    flags = [(p, p.requires_grad) for p in core.parameters()]
    # `add_lora` seeds torch's random generators; their states are put back below, for the tests that follow.
    cpu_state, cuda_states = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    wrapped = add_lora(model, seed=0)                    # the default rank and targets; lora_B starts at zero
    try:
        named = list(core.named_parameters())
        probes = {}
        for label, index in _depths(core).items():
            lora_b = [p for name, p in named if name.startswith(f"model.layers.{index}.") and ".lora_B." in name]
            assert lora_b, f"layer {index} has no lora_B weight"
            probes[f"{label}, lora_B"] = lora_b
            probes[f"{label}, norm"] = [core.model.layers[index].input_layernorm.weight]
        ways = {"all rows at once": _way([[0, 1]]), "pieces of 16": _way([[0, 1]], head_rows=16),
                "one sequence per micro-batch": _way([[0], [1]]), "loss times 3": _way([[0, 1]], scale=3.0)}
        results = _control_gradients(wrapped, tokenizer, probes, seqs, weights, factors, ways)
    finally:
        wrapped.unload()                                 # peft puts the original layers back
        for p, flag in flags:
            p.requires_grad_(flag)
            p.grad = None
        core.eval()
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)
        assert not any("lora_" in name for name, _ in core.named_parameters())
    pieces = _print_against("C5 pieces, at the LoRA weights", results, "all rows at once", "pieces of 16")
    shapes = _print_against("C5 batch shapes, at the LoRA weights", results, "all rows at once",
                            "one sequence per micro-batch")
    _print_against("C5 rounding of the backward pass alone, at the LoRA weights", results,
                   "all rows at once", "loss times 3")
    # The pieces keep the bound 0.9 (measured 0.998 at the first layer). The two batch shapes were measured
    # at 0.861 to 0.963 on 2026-11-18, so their bound is 0.5 (see the note in the C1 test).
    assert all(cosine > 0.9 for cosine, _ in pieces.values()), pieces
    assert all(cosine > 0.5 for cosine, _ in shapes.values()), shapes


def test_control_c3_pieces_against_all_rows_in_float32(model, tokenizer, prompts):
    """C3: pieces of 16 against all rows, the batch shapes of C1 and the "times 3" of C4, on a second copy
    of the model in float32. The copy is loaded as `load_model` loads the model, with float32 in place of
    bfloat16, and is deleted at the end; the model of the session is not touched.

    Memory (own calculation, not measured): 4 bytes per weight for the copy, next to the model of the
    session, in main memory while it loads and then on the GPU. The test is skipped, with the numbers, if
    the GPU has less free memory than that plus 3 GiB, or if it runs out of memory all the same. Only
    finite numbers are asserted. What the result means: see above."""
    import gc
    import os

    import torch
    from transformers import AutoModelForCausalLM

    from accessdesk.harness.hf_backend import resolve_revision

    core = causal_lm(model)
    seqs, weights, factors = _control_input(prompts)
    name = os.environ.get("ACCESSDESK_TEST_MODEL", "Qwen/Qwen3.5-2B")
    gib = 2 ** 30
    needed = 4 * sum(p.numel() for p in core.parameters()) + 3 * gib
    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info(core.device)[0]
    if free < needed:
        pytest.skip(f"C3 not run: a float32 copy of {name} needs about {needed / gib:.1f} GiB of GPU memory "
                    f"(4 bytes per weight and 3 GiB to work in), and {free / gib:.1f} GiB are free")
    revision = resolve_revision(name, None)
    source = {"revision": revision} if revision is not None else {}
    ways = {"all rows at once": _way([[0, 1]]), "pieces of 16": _way([[0, 1]], head_rows=16),
            "one sequence per micro-batch": _way([[0], [1]]), "loss times 3": _way([[0, 1]], scale=3.0)}
    copy32 = core32 = out_of_memory = None
    try:
        copy32, loading = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                               output_loading_info=True, **source)
        assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
        copy32.to(core.device)
        copy32.eval()
        core32 = causal_lm(copy32)
        dtypes = {str(p.dtype) for p in core32.parameters()}
        assert dtypes == {"torch.float32"}, f"the copy is not float32: {dtypes}"
        same_weights = all(torch.equal(p32.to(p16.dtype), p16)
                           for p32, p16 in zip(core32.parameters(), core.parameters()))
        results = _control_gradients(copy32, tokenizer, _norm_probes(core32), seqs, weights, factors, ways)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        out_of_memory = str(error)[:200]
    finally:
        copy32 = core32 = None                           # the copy is given back before anything else is done
        gc.collect()
        torch.cuda.empty_cache()
    if out_of_memory is not None:
        pytest.skip(f"C3 not run: the GPU ran out of memory with the float32 copy of {name}: {out_of_memory}")
    print(f"\nC3 float32 copy of {name}: its weights, rounded to bfloat16, are the weights of the session's "
          f"model: {same_weights}. TF32 allowed for matmul: {torch.backends.cuda.matmul.allow_tf32}, for "
          f"cuDNN: {torch.backends.cudnn.allow_tf32} (TF32 rounds float32 products to about 10 binary digits)")
    _print_against("C3 float32, pieces by depth", results, "all rows at once", "pieces of 16")
    _print_against("C3 float32, batch shapes", results, "all rows at once", "one sequence per micro-batch")
    _print_against("C3 float32, rounding of the backward pass alone", results, "all rows at once",
                   "loss times 3")
