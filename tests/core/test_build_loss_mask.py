"""build_loss_mask(): hand-made cases and properties, run on the main version and on the second version.

Every expected mask below was typed by hand from the contract in the function's docstring
(accessdesk/train/loss_mask.py), not by running any implementation. Each test takes the function under test
as the argument `build_loss_mask`: the main version, and the second version of accessdesk/_second once it is
written (until then those tests are skipped).
"""
from __future__ import annotations

import itertools

import pytest

from accessdesk import impl
from accessdesk.types import SEGMENT_KINDS, Segment
from tests.conftest import VERSIONS, function_under_test


@pytest.fixture(params=VERSIONS)
def build_loss_mask(request):
    """The function under test: the main version, then the second version."""
    return function_under_test("build_loss_mask", request.param)


NONE: frozenset[int] = frozenset()

# A two-turn episode of 10 tokens:   P P P | S S | X X X | S S      (P prompt, S sampled, X suffix)
TWO_TURNS = [Segment("prompt", 0, 3, -1), Segment("sampled", 3, 5, 0), Segment("suffix", 5, 8, 0),
             Segment("sampled", 8, 10, 1)]
# A three-turn episode of 12 tokens: P P | S S | X | S S S | X X | S S
THREE_TURNS = [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0), Segment("suffix", 4, 5, 0),
               Segment("sampled", 5, 8, 1), Segment("suffix", 8, 10, 1), Segment("sampled", 10, 12, 2)]
# Turn 0 reached the token cap, so the harness added one end-of-turn token (F). 9 tokens:
#                                    P P | S S S | F | X X | S
CUT_OFF_TURN = [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 5, 0), Segment("forced_eos", 5, 6, 0),
                Segment("suffix", 6, 8, 0), Segment("sampled", 8, 9, 1)]

CASES = [
    # (label, n_tokens, segments, exclude_turns, the right mask)
    ("an empty buffer", 0, [], NONE, []),
    ("a prompt alone", 3, [Segment("prompt", 0, 3, -1)], NONE, [0, 0, 0]),
    ("a prompt of one token and one sampled token", 2,
     [Segment("prompt", 0, 1, -1), Segment("sampled", 1, 2, 0)], NONE, [0, 1]),
    ("one turn", 7, [Segment("prompt", 0, 4, -1), Segment("sampled", 4, 7, 0)], NONE, [0, 0, 0, 0, 1, 1, 1]),
    ("two turns", 10, TWO_TURNS, NONE, [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]),
    ("three turns", 12, THREE_TURNS, NONE, [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]),
    ("a buffer that ends with a suffix (the next turn is not sampled yet)", 6,
     [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0), Segment("suffix", 4, 6, 0)], NONE,
     [0, 0, 1, 1, 0, 0]),
    # --- the forced end-of-turn token is never in the loss ---
    ("a cut-off turn: its sampled tokens count, the forced token does not", 9, CUT_OFF_TURN, NONE,
     [0, 0, 1, 1, 1, 0, 0, 0, 1]),
    ("a cut-off last turn", 4,
     [Segment("prompt", 0, 1, -1), Segment("sampled", 1, 3, 0), Segment("forced_eos", 3, 4, 0)], NONE,
     [0, 1, 1, 0]),
    # --- exclude_turns ---
    ("two turns, turn 0 excluded", 10, TWO_TURNS, frozenset({0}), [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
    ("two turns, turn 1 excluded", 10, TWO_TURNS, frozenset({1}), [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
    ("two turns, both excluded", 10, TWO_TURNS, frozenset({0, 1}), [0] * 10),
    ("three turns, the middle one excluded", 12, THREE_TURNS, frozenset({1}),
     [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]),
    ("three turns, the first and the last excluded", 12, THREE_TURNS, frozenset({0, 2}),
     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]),
    ("a turn number that the episode does not have changes nothing", 10, TWO_TURNS, frozenset({2, 7}),
     [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]),
    ("the prompt's turn number, -1, changes nothing", 10, TWO_TURNS, frozenset({-1}),
     [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]),
    ("the cut-off turn excluded: only the other turn is left", 9, CUT_OFF_TURN, frozenset({0}),
     [0, 0, 0, 0, 0, 0, 0, 0, 1]),
    ("the turn after the cut-off turn excluded", 9, CUT_OFF_TURN, frozenset({1}),
     [0, 0, 1, 1, 1, 0, 0, 0, 0]),
]

BAD_CASES = [
    # (label, n_tokens, segments): each one must raise ValueError
    # --- the segments do not tile [0, n_tokens) ---
    ("a gap between two segments", 5, [Segment("prompt", 0, 2, -1), Segment("sampled", 3, 5, 0)]),
    ("two segments overlap", 5, [Segment("prompt", 0, 3, -1), Segment("sampled", 2, 5, 0)]),
    ("the first segment does not start at 0", 3, [Segment("prompt", 1, 3, -1)]),
    ("the segments end before the buffer ends", 5,
     [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0)]),
    ("the segments end after the buffer ends", 3,
     [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0)]),
    ("the segments are not in order", 4, [Segment("sampled", 2, 4, 0), Segment("prompt", 0, 2, -1)]),
    ("the same segment twice", 4, [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 4, 0),
                                  Segment("sampled", 2, 4, 0)]),
    ("no segments, but the buffer has tokens", 2, []),
    ("segments, but the buffer has no tokens", 0, [Segment("prompt", 0, 2, -1)]),
    ("a segment without a token", 4, [Segment("prompt", 0, 2, -1), Segment("sampled", 2, 2, 0),
                                     Segment("suffix", 2, 4, 0)]),
    ("a segment that ends before it starts", 4, [Segment("prompt", 0, 3, -1), Segment("sampled", 3, 2, 0),
                                                Segment("suffix", 2, 4, 0)]),
    # --- an unknown kind ---
    ("an unknown kind", 4, [Segment("prompt", 0, 2, -1), Segment("assistant", 2, 4, 0)]),
    ("a known kind in capital letters", 4, [Segment("prompt", 0, 2, -1), Segment("SAMPLED", 2, 4, 0)]),
    ("an unknown kind in a turn that is excluded anyway", 4,
     [Segment("prompt", 0, 2, -1), Segment("generated", 2, 4, 0)]),
    ("an empty kind", 2, [Segment("", 0, 2, -1)]),
]


@pytest.mark.parametrize("label, n_tokens, segments, exclude_turns, answer", CASES,
                         ids=[case[0] for case in CASES])
def test_build_loss_mask_hand_made(build_loss_mask, label, n_tokens, segments, exclude_turns, answer):
    mask = build_loss_mask(n_tokens, list(segments), exclude_turns)
    assert mask == answer
    assert type(mask) is list and all(type(weight) is int for weight in mask)


@pytest.mark.parametrize("label, n_tokens, segments", BAD_CASES, ids=[case[0] for case in BAD_CASES])
def test_build_loss_mask_refuses(build_loss_mask, label, n_tokens, segments):
    with pytest.raises(ValueError):
        build_loss_mask(n_tokens, list(segments), frozenset({0}))
    with pytest.raises(ValueError):
        build_loss_mask(n_tokens, list(segments))


def test_the_hand_made_cases_reach_every_part_of_the_contract():
    kinds = {segment.kind for case in CASES for segment in case[2]}
    assert kinds == set(SEGMENT_KINDS)
    assert sum(1 for case in CASES if case[3]) >= 6                     # cases with an excluded turn
    assert any(case[4] == [] for case in CASES)                        # the empty buffer
    assert any(0 in case[4] and 1 in case[4] for case in CASES)
    # A weight of 0 is seen on a sampled token (excluded), and on a token of each other kind.
    zero_kinds = {segment.kind for case in CASES for segment in case[2]
                  if case[4][segment.start] == 0}
    assert zero_kinds == set(SEGMENT_KINDS)
    bad_kinds = [case for case in BAD_CASES if any(s.kind not in SEGMENT_KINDS for s in case[2])]
    assert len(bad_kinds) >= 3 and len(BAD_CASES) - len(bad_kinds) >= 9


def test_no_turn_is_excluded_by_default(build_loss_mask):
    assert build_loss_mask(10, list(TWO_TURNS)) == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]


def test_the_mask_is_a_list_of_whole_numbers(build_loss_mask):
    mask = build_loss_mask(9, list(CUT_OFF_TURN), frozenset({1}))
    assert type(mask) is list
    assert [type(weight) for weight in mask] == [int] * 9          # not bool, not float


def test_the_arguments_are_not_changed(build_loss_mask):
    segments = list(TWO_TURNS)
    exclude_turns = frozenset({1})
    build_loss_mask(10, segments, exclude_turns)
    assert segments == TWO_TURNS and exclude_turns == frozenset({1})
    segments = list(THREE_TURNS)
    build_loss_mask(12, segments, frozenset({1}))
    assert segments == THREE_TURNS


def test_two_calls_do_not_share_a_list(build_loss_mask):
    first = build_loss_mask(10, list(TWO_TURNS))
    first[0] = 1
    assert build_loss_mask(10, list(TWO_TURNS)) == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]


def slow_mask(n_tokens: int, segments: list[Segment], exclude_turns: frozenset[int]) -> list[int]:
    """The contract, token by token: find the one segment that holds the token; read its kind and turn."""
    weights = []
    for token in range(n_tokens):
        (segment,) = [s for s in segments if s.start <= token < s.end]
        weights.append(1 if segment.kind == "sampled" and segment.turn not in exclude_turns else 0)
    return weights


def tilings(n_tokens: int) -> list[list[tuple[int, int]]]:
    """Every way to cut [0, n_tokens) into runs of at least one token, as lists of (start, end)."""
    result = []
    for n_cuts in range(n_tokens):
        for cuts in itertools.combinations(range(1, n_tokens), n_cuts):
            edges = [0, *cuts, n_tokens]
            result.append(list(zip(edges[:-1], edges[1:])))
    return result if n_tokens else [[]]


def test_tilings_are_counted_right():
    assert tilings(0) == [[]]
    assert tilings(1) == [[(0, 1)]]
    assert sorted(tilings(3)) == [[(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 3)], [(0, 2), (2, 3)], [(0, 3)]]
    assert len(tilings(5)) == 16


def test_the_mask_equals_the_token_by_token_definition_on_every_small_buffer(build_loss_mask):
    """All buffers of up to 5 tokens, with every kind in every segment and every set of excluded turns."""
    n_checked = 0
    for n_tokens in range(0, 6):
        for runs in tilings(n_tokens):
            for kinds in itertools.product(SEGMENT_KINDS, repeat=len(runs)):
                # the turn of a segment is its position in the buffer minus 1, so turns -1, 0, 1, ... occur
                segments = [Segment(kind, start, end, index - 1)
                            for index, (kind, (start, end)) in enumerate(zip(kinds, runs))]
                for exclude_turns in (frozenset(), frozenset({0}), frozenset({1, 2}), frozenset({-1, 0, 3})):
                    mask = build_loss_mask(n_tokens, segments, exclude_turns)
                    assert mask == slow_mask(n_tokens, segments, exclude_turns)
                    n_checked += 1
    assert n_checked > 10_000


@pytest.mark.parametrize("segments, n_tokens", [(TWO_TURNS, 10), (THREE_TURNS, 12), (CUT_OFF_TURN, 9)])
def test_excluding_a_turn_removes_exactly_that_turns_sampled_tokens(build_loss_mask, segments, n_tokens):
    full = build_loss_mask(n_tokens, segments)
    for segment in segments:
        if segment.kind != "sampled":
            continue
        without = build_loss_mask(n_tokens, segments, frozenset({segment.turn}))
        changed = [token for token in range(n_tokens) if without[token] != full[token]]
        assert changed == list(range(segment.start, segment.end))
        assert all(without[token] == 0 for token in changed)


def test_the_sum_of_the_mask_is_the_number_of_sampled_tokens(build_loss_mask):
    assert sum(build_loss_mask(10, TWO_TURNS)) == 4
    assert sum(build_loss_mask(12, THREE_TURNS)) == 7
    assert sum(build_loss_mask(9, CUT_OFF_TURN)) == 4            # the forced token is not counted
    assert sum(build_loss_mask(9, CUT_OFF_TURN, frozenset({0}))) == 1


def test_the_probe_input_of_impl_is_a_valid_buffer(build_loss_mask):
    assert build_loss_mask(*impl._probe_args("build_loss_mask")) == [0, 1]
