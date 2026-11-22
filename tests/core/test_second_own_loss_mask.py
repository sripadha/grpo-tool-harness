"""Hand-made cases for the second version of build_loss_mask (accessdesk/_second/loss_mask.py).

Every expected output is worked out by hand from the contract, in the comments. These tests import the
second version directly; they never look at the main version.
"""
from __future__ import annotations

import pytest

from accessdesk._second.loss_mask import build_loss_mask
from accessdesk.types import Segment


def test_empty_buffer_is_empty_list():
    # Contract: "No segments together with n_tokens = 0 is a valid, empty buffer: the result is []."
    assert build_loss_mask(0, []) == []


def test_two_turn_episode():
    # prompt [0,3) turn -1 -> 0 0 0
    # sampled [3,5) turn 0 -> 1 1
    # suffix [5,8) turn 0 -> 0 0 0
    # sampled [8,10) turn 1 -> 1 1
    segs = [Segment("prompt", 0, 3, -1), Segment("sampled", 3, 5, 0),
            Segment("suffix", 5, 8, 0), Segment("sampled", 8, 10, 1)]
    assert build_loss_mask(10, segs) == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]


def test_forced_eos_is_zero():
    # prompt [0,2) -> 0 0; sampled [2,4) turn 0 -> 1 1; forced_eos [4,5) -> 0 (the model did not choose it)
    segs = [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0), Segment("forced_eos", 4, 5, 0)]
    assert build_loss_mask(5, segs) == [0, 0, 1, 1, 0]


def test_exclude_turns_zeroes_that_turn_only():
    # Same as the two-turn episode, but turn 1 excluded: tokens 8, 9 become 0; turn 0 stays 1.
    segs = [Segment("prompt", 0, 3, -1), Segment("sampled", 3, 5, 0),
            Segment("suffix", 5, 8, 0), Segment("sampled", 8, 10, 1)]
    assert build_loss_mask(10, segs, frozenset({1})) == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    # Excluding both turns: all zero.
    assert build_loss_mask(10, segs, frozenset({0, 1})) == [0] * 10


def test_exclude_turn_that_no_sampled_segment_has_changes_nothing():
    # Contract: "A number in exclude_turns that no sampled segment has changes nothing."
    # -1 is the prompt's turn (prompt is 0 anyway); 7 matches nothing.
    segs = [Segment("prompt", 0, 1, -1), Segment("sampled", 1, 3, 0)]
    assert build_loss_mask(3, segs, frozenset({-1, 7})) == [0, 1, 1]


def test_all_sampled_buffer():
    # One sampled segment covering the whole buffer: every token weight 1.
    assert build_loss_mask(4, [Segment("sampled", 0, 4, 0)]) == [1, 1, 1, 1]


def test_single_token_segments():
    # Each segment holds exactly one token (start < end holds): 0 1 0 0
    segs = [Segment("prompt", 0, 1, -1), Segment("sampled", 1, 2, 0),
            Segment("forced_eos", 2, 3, 0), Segment("suffix", 3, 4, 0)]
    assert build_loss_mask(4, segs) == [0, 1, 0, 0]


def test_result_is_plain_python_ints_and_right_length():
    out = build_loss_mask(3, [Segment("prompt", 0, 1, -1), Segment("sampled", 1, 3, 0)])
    assert len(out) == 3
    assert all(type(w) is int for w in out)


def test_arguments_are_not_changed():
    segs = [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 3, 0)]
    segs_copy = list(segs)
    excl = frozenset({5})
    build_loss_mask(3, segs, excl)
    assert segs == segs_copy
    assert excl == frozenset({5})


@pytest.mark.parametrize("kind", ["Sampled", "tool", "", "prompt "])
def test_unknown_kind_raises(kind):
    # Contract: ValueError if "a segment's kind is not one of the four kinds".
    with pytest.raises(ValueError):
        build_loss_mask(2, [Segment("prompt", 0, 1, -1), Segment(kind, 1, 2, 0)])


@pytest.mark.parametrize("n_tokens,segs", [
    # first segment does not start at 0
    (3, [Segment("prompt", 1, 3, -1)]),
    # gap between segments: [0,2) then [3,4)
    (4, [Segment("prompt", 0, 2, -1), Segment("sampled", 3, 4, 0)]),
    # overlap: [0,3) then [2,4)
    (4, [Segment("prompt", 0, 3, -1), Segment("sampled", 2, 4, 0)]),
    # last segment ends before n_tokens
    (5, [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0)]),
    # last segment ends after n_tokens
    (3, [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0)]),
    # empty segment (start == end) in the middle, even though the rest tiles
    (4, [Segment("prompt", 0, 2, -1), Segment("suffix", 2, 2, 0), Segment("sampled", 2, 4, 0)]),
    # reversed segment
    (2, [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 1, 0)]),
    # out of order: the right pieces, wrong order
    (4, [Segment("sampled", 2, 4, 0), Segment("prompt", 0, 2, -1)]),
    # tokens but no segments
    (3, []),
    # segments but n_tokens = 0
    (0, [Segment("prompt", 0, 1, -1)]),
    # negative n_tokens
    (-1, []),
])
def test_bad_tiling_raises(n_tokens, segs):
    with pytest.raises(ValueError):
        build_loss_mask(n_tokens, segs)
