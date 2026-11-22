"""Second version of `build_loss_mask`, written from the contract in the function's docstring only, without
reading the main version. Used by tests only.

How it computes: first it checks every segment (kind, tiling) and collects one weight and one length per
segment; then numpy repeats each segment's weight over its length.

Choices where the contract was silent when this version was written (since the comparison of the two
versions, 2026-11-17, the contract says that turns and the order of kinds are not checked, and that other
types than Python ints are outside it):
- n_tokens, start and end must be integers (anything `operator.index` accepts); otherwise TypeError.
- `turn` is not checked, and the order of kinds is not checked: the weight depends only on kind and turn.
- exclude_turns may be any container that supports `in` (frozenset, set, list, tuple).
- the result holds plain Python ints.
"""
from __future__ import annotations

import operator

import numpy as np

from accessdesk.types import SEGMENT_KINDS, Segment


def build_loss_mask(n_tokens: int, segments: list[Segment],
                    exclude_turns: frozenset[int] = frozenset()) -> list[int]:
    """One weight per token of an episode's token buffer: 1 where the loss is taken, 0 everywhere else.

    Arguments
        n_tokens       the number of tokens in the buffer
        segments       the buffer's segments, in the order of the buffer (`TokenBuffer.segments` in
                       accessdesk/harness/tokens.py). A Segment (accessdesk/types.py) has four fields:
                       kind, start, end, turn. It covers the tokens start, start + 1, ..., end - 1;
                       `end` itself is not part of it.
        exclude_turns  numbers of assistant turns whose sampled tokens get weight 0 as well, for example
                       the turns that were cut off by the token cap. The default excludes no turn.

    The four kinds of segment (types.SEGMENT_KINDS)
        "prompt"      the first prompt: the system turn with the tools, the request, and the header of the
                      first assistant turn. Its turn is -1.
        "sampled"     the tokens that the model sampled in one assistant turn, with the `<|im_end|>` that
                      ends the turn if the model sampled it. `turn` is the turn's number, from 0.
        "suffix"      what the harness wrote after a turn: the tool result and the header of the next
                      assistant turn.
        "forced_eos"  one `<|im_end|>` that the harness added because the turn reached the token cap. The
                      model did not choose this token.

    Result: a list of n_tokens ints (0 or 1).
        weight 1   a token of a "sampled" segment whose turn is not in exclude_turns
        weight 0   every other token: "prompt", "suffix" and "forced_eos" always, and "sampled" when its
                   turn is in exclude_turns

    Raises ValueError if
        - a segment's kind is not one of the four kinds, or
        - the segments do not tile [0, n_tokens) exactly, in order. That is: the first segment starts at 0,
          every next segment starts where the one before it ended, the last segment ends at n_tokens, and
          every segment holds at least one token (start < end).
    No segments together with n_tokens = 0 is a valid, empty buffer: the result is [].

    A number in exclude_turns that no sampled segment has changes nothing. The arguments are not changed.
    Nothing else is checked: not the turn numbers, and not the order or the number of the kinds.
    n_tokens, start and end are Python ints, and kind is a string. Arguments of other types are outside
    this contract: what the call does then is not defined.
    """
    n = operator.index(n_tokens)
    if n < 0:
        raise ValueError(f"n_tokens is negative: {n}")

    weights: list[int] = []
    lengths: list[int] = []
    expected_start = 0
    for i, seg in enumerate(segments):
        if not isinstance(seg.kind, str) or seg.kind not in SEGMENT_KINDS:
            raise ValueError(f"segment {i}: unknown kind {seg.kind!r}")
        start = operator.index(seg.start)
        end = operator.index(seg.end)
        if start != expected_start:
            raise ValueError(f"segment {i}: starts at {start}, expected {expected_start}")
        if end <= start:
            raise ValueError(f"segment {i}: empty or reversed ({start}, {end})")
        weight = 1 if (seg.kind == "sampled" and seg.turn not in exclude_turns) else 0
        weights.append(weight)
        lengths.append(end - start)
        expected_start = end
    if expected_start != n:
        raise ValueError(f"the segments end at {expected_start}, but n_tokens is {n}")

    if n == 0:
        return []
    mask = np.repeat(np.array(weights, dtype=np.int64), np.array(lengths, dtype=np.int64))
    return [int(w) for w in mask]
