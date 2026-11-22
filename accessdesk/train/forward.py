"""Log-probabilities of chosen tokens under the model: the forward pass of training and of the parity check.

An episode is a list of token ids with one weight per token (0 or 1). `token_logprobs` returns, for each
episode, log p(token t | tokens before t) for every token t whose weight is 1, in order.

The index arithmetic (the part to check when you read this file). For a sequence s[0], ..., s[n-1]:
  - the hidden state at position p has seen s[0..p], so it predicts the token at position p + 1;
  - so the log-probability of token t is read from the hidden state at position t - 1;
  - token 0 has no token before it: it can never be selected, and a weight of 1 there is an error;
  - in a batch padded on the LEFT to width W the sequence starts at column W - n, so token t is in column
    (W - n) + t and the hidden state that predicts it is in column (W - n) + t - 1.
`loss_positions` is this arithmetic and nothing else. It needs no torch, so it can be checked by hand.

What is read from the library source (transformers 5.23.0, models/qwen3_5/modeling_qwen3_5.py):
  - `Qwen3_5ForCausalLM.model` is the decoder (`Qwen3_5TextModel`, line 1660) and `.lm_head` the output layer
    (line 1662). The decoder returns `last_hidden_state` after the final norm (lines 1283-1288), and the
    model's own forward does `lm_head(last_hidden_state)` (lines 1708-1710). Here the decoder is called
    directly and `lm_head` is applied to the selected rows only: the full [batch, length, 248320] logits are
    never built. The selected rows go through `lm_head` in pieces of `HEAD_ROWS` rows (`row_pieces`); what
    that saves and what it does not is written at `_micro_batch_logprobs`.
  - The 2-D attention mask is what makes left padding safe: the Gated DeltaNet layers multiply their input by
    it (lines 222-231 and 552; masking_utils.py lines 1499-1529). The layer's projections and its convolution
    have no bias (lines 517-542), so a zeroed pad position gives a zero key and writes nothing into the
    recurrent state (lines 481-484). A 4-D mask switches that off (masking_utils.py line 1521) and is never
    passed here.
  - Without `position_ids` the decoder numbers the columns 0..W-1 for every row (lines 1241-1244), pads
    included. `generate` instead counts real tokens only (generation/utils.py lines 905-908). This file builds
    the positions exactly as `generate` does, so that the trainer and the sampler see the same positions.

torch is imported inside the functions, so the module can be imported and `loss_positions` and `row_pieces`
tested without it.
"""
from __future__ import annotations

from typing import Any

from accessdesk.harness.hf_backend import causal_lm, pad_batch

__all__ = ["pad_batch", "position_ids", "loss_positions", "row_pieces", "token_logprobs",
           "set_gradient_checkpointing", "HEAD_ROWS"]

# The selected rows of a micro-batch go through the output layer in pieces of this many rows. One row of the
# output layer has one number per vocabulary entry (248,320 for the 2B), so a piece of 1,024 rows is about
# 254 million numbers. See `_micro_batch_logprobs` for the memory.
HEAD_ROWS = 1024


def position_ids(attention_mask: Any) -> Any:
    """Position of each token, counted over real tokens only: 0, 1, 2, ... from the first real one; 0 on pads.

    The same formula as `GenerationMixin._prepare_position_ids_for_generation` (generation/utils.py 905-908).
    """
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


def loss_positions(seqs: list[list[int]],
                   weights: list[list[int]]) -> tuple[list[int], list[int], list[int], list[int]]:
    """Where to read the log-probability of every weight-1 token in a LEFT-padded batch of `seqs`.

    Returns (rows, columns, targets, counts), the first three with one entry per weight-1 token, in batch
    order and then in token order:
        rows[k], columns[k]   the hidden state that predicts the token: the position just before it
        targets[k]            the id of the token itself
    counts[i] is the number of weight-1 tokens of sequence i, so that the flat lists can be cut back into
    sequences.

    Example: seqs = [[5, 6, 7], [8, 9]], weights = [[0, 0, 1], [0, 1]]. The width is 3, so the second row is
    [pad, 8, 9]. Token 7 (row 0, column 2) is predicted from column 1. Token 9 (row 1, column 2) is predicted
    from column 1 as well, where token 8 sits. Result: ([0, 1], [1, 1], [7, 9], [1, 1]).
    """
    if len(seqs) != len(weights):
        raise ValueError(f"{len(seqs)} sequences but {len(weights)} weight lists")
    width = max((len(seq) for seq in seqs), default=0)
    rows: list[int] = []
    columns: list[int] = []
    targets: list[int] = []
    counts: list[int] = []
    for row, (seq, seq_weights) in enumerate(zip(seqs, weights)):
        _check_weights(row, seq, seq_weights)
        first_column = width - len(seq)              # the number of pad tokens on the left of this row
        selected = [t for t, weight in enumerate(seq_weights) if weight == 1]
        for t in selected:
            rows.append(row)
            columns.append(first_column + t - 1)     # the hidden state at token t - 1 predicts token t
            targets.append(seq[t])
        counts.append(len(selected))
    return rows, columns, targets, counts


def _check_weights(row: int, seq: list[int], seq_weights: list[int]) -> None:
    if len(seq) == 0:
        raise ValueError(f"sequence {row} is empty")
    if len(seq_weights) != len(seq):
        raise ValueError(f"sequence {row} has {len(seq)} tokens but {len(seq_weights)} weights")
    if any(weight not in (0, 1) for weight in seq_weights):
        raise ValueError(f"sequence {row}: every weight must be 0 or 1")
    if seq_weights[0] == 1:
        raise ValueError(f"sequence {row}: token 0 has weight 1, but nothing comes before it to predict it")


def row_pieces(n_rows: int, piece: int = HEAD_ROWS) -> list[tuple[int, int]]:
    """Cut the rows 0 to n_rows - 1 into consecutive pieces of at most `piece` rows: a list of (start, stop),
    with `stop` not included. Every row is in exactly one piece, and the pieces are in row order.

    Example: row_pieces(2500, 1024) == [(0, 1024), (1024, 2048), (2048, 2500)]. No row gives no piece.
    """
    if piece < 1:
        raise ValueError(f"a piece must have at least 1 row, not {piece}")
    if n_rows < 0:
        raise ValueError(f"the number of rows must be 0 or more, not {n_rows}")
    return [(start, min(start + piece, n_rows)) for start in range(0, n_rows, piece)]


def token_logprobs(model: Any, seqs: list[list[int]], weights: list[list[int]], *, pad_id: int,
                   micro_batch: int, with_grad: bool, head_rows: int = HEAD_ROWS) -> list[Any]:
    """For each sequence: a 1-D float32 tensor with the log-probability of every weight-1 token, in order,
    given all the tokens before it. Temperature 1, the whole vocabulary: the model's own distribution.

    `model` is the transformers model or its peft wrapper. Its mode is not changed here: call `model.train()`
    or `model.eval()` before. The sequences are run in micro-batches of `micro_batch`, each padded on the
    left. Only the selected positions go through the output layer, in pieces of `head_rows` rows. The piece
    size changes the memory and not the result (`_micro_batch_logprobs`).

    with_grad=False: no graph is built; any number of sequences.
    with_grad=True:  the results carry the graph. The graph of a micro-batch must be freed by `backward()`
        before the next micro-batch is run, or the memory of all of them adds up. So with gradients this
        function takes ONE micro-batch per call (len(seqs) <= micro_batch) and raises otherwise. The training
        loop slices the episodes, and for each slice calls this function, builds its part of the loss and
        calls `backward()`. Gradients add up over the slices because the loss is a sum over episodes.
    """
    import torch

    if micro_batch < 1:
        raise ValueError(f"micro_batch must be at least 1, not {micro_batch}")
    if with_grad and len(seqs) > micro_batch:
        raise ValueError(
            f"with_grad=True takes one micro-batch per call, but got {len(seqs)} sequences for micro_batch="
            f"{micro_batch}. Call it once per slice, and run backward() on that loss before the next call.")
    loss_positions(seqs, weights)                    # checks every sequence before any forward pass
    row_pieces(0, head_rows)                         # checks the piece size before any forward pass
    core = causal_lm(model)
    results: list[Any] = []
    with torch.set_grad_enabled(with_grad):
        for start in range(0, len(seqs), micro_batch):
            stop = start + micro_batch
            results.extend(_micro_batch_logprobs(core, seqs[start:stop], weights[start:stop], pad_id,
                                                 head_rows))
    return results


def _head_logprobs(core: Any, selected: Any, targets: Any) -> Any:
    """One piece: the log-probability of token `targets[k]` read from the hidden state `selected[k]`, for
    every row k. [rows, hidden size] and [rows] in, [rows] float32 out.

    The piece is a function of its own, so that its large tensors (the logits and their float32 copy) are
    freed when it returns.
    """
    import torch

    logits = core.lm_head(selected)                                       # [rows, vocabulary]
    all_logprobs = torch.log_softmax(logits.float(), dim=-1)              # float32, as in generate()
    return all_logprobs.gather(1, targets[:, None]).squeeze(1)            # [rows]


def _selected_logprobs(core: Any, selected: Any, targets: Any, head_rows: int) -> Any:
    """All selected rows through the output layer, in pieces of `head_rows` rows, joined again in row order.
    [n_selected, hidden size] and [n_selected] in, [n_selected] float32 out; at least one row."""
    import torch

    return torch.cat([_head_logprobs(core, selected[start:stop], targets[start:stop])
                      for start, stop in row_pieces(len(targets), head_rows)])


def _micro_batch_logprobs(core: Any, seqs: list[list[int]], weights: list[list[int]], pad_id: int,
                          head_rows: int = HEAD_ROWS) -> list[Any]:
    """`token_logprobs` for one padded batch. Gradients are on or off as set by the caller.

    The output layer is the large part: one selected row gives one number per vocabulary entry. So the
    selected rows go through `lm_head` and `log_softmax` in pieces of `head_rows` rows (`row_pieces`), and
    the log-probabilities of the pieces are joined again in row order.

    Why the pieces give the same numbers. `lm_head` and `log_softmax` work on each row alone: the
    log-probability of a token is computed from the one hidden state that predicts it, and from no other
    row. So, as maths, the pieces give exactly what all rows at once give. The caller's loss is a sum over
    tokens, so it is the same sum, and its gradient is the same sum of one gradient per token. On a GPU the
    last bits can differ, and only by floating-point rounding: the matrix product of `lm_head` may add up
    its terms in another order when it gets another number of rows. tests/gpu/test_train_fixes_gpu.py
    compares pieces against all rows at once, values and gradients.

    Memory of this part. The numbers are an own calculation from the tensor shapes. They were not measured
    on a GPU when this was written; the last test of tests/gpu/test_train_fixes_gpu.py measures the peak
    with pieces and without. V is the vocabulary (248,320 for the 2B), R the selected rows of the
    micro-batch, P the rows of a piece.
      - While a piece is computed it has its bfloat16 logits, their float32 copy and the float32 result of
        `log_softmax`: 10 bytes per entry, for P rows. The first two are freed when the piece is done.
      - With gradients, the float32 result of `log_softmax` of EVERY piece stays until the caller's
        `backward()`, because the backward pass of `log_softmax` reads it: 4 bytes per entry, about
        0.95 MiB per selected row. The pieces do NOT save this.
      - In `backward()` the pieces are handled one after another: torch runs a graph from its newest node
        back, so one piece is finished before the piece before it is begun. A piece needs about 10 more
        bytes per entry for its P rows while it is handled.
    So the peak is about (4 R + 10 P) V bytes with pieces, and about 14 R V bytes with all rows at once.
    For R = 3,237 (the largest micro-batch of 4 in the saved 2B episodes at token cap 256, as the review of
    the training maths counted it): 5.4 GiB in place of 10.5 GiB. For R = 8,192 (8 turns x 256 tokens x 4
    episodes, the largest that the settings allow): 9.9 GiB in place of 26.5 GiB.
    What is left is the 4 R V bytes. To save them too, the backward pass of the output layer would have to
    run piece by piece inside this function (it would then need each token's factor in the loss, so both
    loss functions would change), or each piece would have to be computed a second time in the backward
    pass (torch.utils.checkpoint). Neither is done here. The hidden states are not the problem: they come
    from one decoder pass, stay in memory until `backward()` in any case, and are small (R rows of the
    hidden size).
    """
    import torch

    device = core.device
    rows, columns, targets, counts = loss_positions(seqs, weights)
    if not targets:                                  # nothing selected: no forward pass is needed
        return [torch.zeros(0, dtype=torch.float32, device=device) for _ in seqs]
    input_ids, attention_mask = pad_batch(seqs, pad_id, side="left")
    hidden = core.model(                             # [batch, width, hidden size], after the final norm
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),    # 2-D, never 4-D (see the module docstring)
        position_ids=position_ids(attention_mask).to(device),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    row_index, column_index = torch.tensor(rows, device=device), torch.tensor(columns, device=device)
    selected = hidden[row_index, column_index]                            # [n_selected, hidden size]
    target_index = torch.tensor(targets, device=device)                   # [n_selected]
    chosen = _selected_logprobs(core, selected, target_index, head_rows)  # [n_selected]
    if not bool(torch.isfinite(chosen).all()):
        raise FloatingPointError("a log-probability is not finite (NaN or infinite): check padding and dtype")
    per_sequence: list[Any] = []
    start = 0
    for count in counts:                             # cut the flat result back into one tensor per sequence
        per_sequence.append(chosen[start:start + count])
        start += count
    return per_sequence


def set_gradient_checkpointing(model: Any, enabled: bool) -> None:
    """Switch gradient checkpointing of the decoder layers on or off.

    Qwen3_5PreTrainedModel has `supports_gradient_checkpointing = True` (modeling_qwen3_5.py line 913) and its
    decoder layers are `GradientCheckpointingLayer`s (line 854). In transformers 5.23 the switch is called
    `activation_checkpointing_enable` (modeling_utils.py line 3110; `gradient_checkpointing_enable` is a
    deprecated alias). Its default is `use_reentrant=False` (line 3149), which also works when no input needs
    a gradient, as with a frozen base model and LoRA. A layer is checkpointed only while it is in train mode
    (modeling_layers.py line 93), so the setting has no effect on a model in eval mode.
    """
    core = causal_lm(model)
    if enabled:
        core.activation_checkpointing_enable()
    else:
        core.activation_checkpointing_disable()
