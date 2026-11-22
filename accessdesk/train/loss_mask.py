"""The loss mask: which tokens of an episode take part in the training loss.

This is the main version of `build_loss_mask`. The rest of the code reaches it through
`accessdesk.impl.get("build_loss_mask")`. An independent second version, written from the docstring below
only, is in `accessdesk/_second/loss_mask.py`; tests compare the two.
Tests: `.venv/bin/python -m pytest tests/core/test_build_loss_mask.py
tests/core/test_differential_loss_mask.py`
A decoded mask of a real episode, to read by eye:
`.venv/bin/python -m pytest tests/core/test_tokens.py -k decoded_mask -s`
"""
from __future__ import annotations

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
    mask: list[int] = []
    position = 0                    # the token at which the next segment must start

    for segment in segments:
        # The three checks: a known kind, no gap and no overlap, at least one token.
        if segment.kind not in SEGMENT_KINDS:
            raise ValueError(f"unknown segment kind {segment.kind!r}; known: {SEGMENT_KINDS}")
        if segment.start != position:
            raise ValueError(f"a segment starts at {segment.start}, but the one before it ended at "
                             f"{position}")
        if segment.end <= segment.start:
            raise ValueError(f"a segment must hold at least one token, not [{segment.start}, {segment.end})")

        # Only tokens that the model sampled itself are in the loss, and only in turns that are not excluded.
        in_loss = segment.kind == "sampled" and segment.turn not in exclude_turns
        weight = 1 if in_loss else 0

        # Every token of the segment gets the same weight.
        mask.extend([weight] * (segment.end - segment.start))
        position = segment.end

    # The last segment must end exactly where the buffer ends.
    if position != n_tokens:
        raise ValueError(f"the segments cover {position} tokens, but the buffer has {n_tokens}")
    return mask
