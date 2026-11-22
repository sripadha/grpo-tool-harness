"""The second version of build_loss_mask() against the main version, on generated buffers.

The comparison at the top is skipped while the second version (accessdesk/_second/loss_mask.py) is not
written. So that a skipped test is not an untested test, the rest of the file runs today: the inputs are
checked, the main version agrees with itself, and the comparison catches each of several wrong functions.

600 of the generated inputs are valid buffers, and the second version must return the same mask. The other
630 are buffers with one fault each (70 for each of nine faults), and the second version must raise
ValueError as the main version does.

Further down, added by the comparison of the two versions:
  - edge inputs: 480 valid buffers of special shapes (one segment, one kind only, long segments, many
    segments, turn numbers that repeat or are negative, everything excluded) and 600 broken segment lists
    with twelve more faults (a negative n_tokens, no segments, a list that does not start at 0, a gap,
    an overlap, a reversed segment, an empty segment at an end, a misspelled kind, two faults at once).
    For the valid ones the right mask is also found a third way, token by token, so a mistake that the two
    versions share would show.
  - real token buffers (they need the tokenizer files): the scripted episode of every dev family in every
    wording, with and without excluded turns, with turns cut off at the token cap, and with broken copies
    of the real segment lists. The masks of the two versions are compared with each other and, on episodes
    without a cut-off turn, with the scan of the token ids (tests/helpers/scan_mask.py).
"""
from __future__ import annotations

import functools

import pytest

from accessdesk import impl
from accessdesk.rng import Rng
from accessdesk.train.loss_mask import build_loss_mask as main_mask
from accessdesk.types import SEGMENT_KINDS, Segment
from tests.conftest import VERSIONS, function_under_test, skip_unless_second_is_written

N_VALID = 600
FAULTS = ("more_tokens", "fewer_tokens", "moved_start", "moved_end", "dropped", "doubled", "swapped",
          "unknown_kind", "empty_segment")
N_PER_FAULT = 70
Input = tuple[int, list[Segment], frozenset[int]]


def valid_buffer(rng: Rng, min_segments: int = 0) -> tuple[int, list[Segment]]:
    """A buffer cut into 0 to 8 segments of 1 to 5 tokens. Kinds and turn numbers are mixed freely: the
    contract asks for a tiling and known kinds, and for nothing about the order of the kinds."""
    segments: list[Segment] = []
    position = 0
    for _ in range(min_segments + rng.below(9 - min_segments)):
        length = 1 + rng.below(5)
        kind = rng.choice(("sampled", "sampled") + SEGMENT_KINDS)        # half of the segments are sampled
        segments.append(Segment(kind, position, position + length, rng.below(5) - 1))    # turns -1 .. 3
        position += length
    return position, segments


def exclude_set(rng: Rng) -> frozenset[int]:
    return frozenset(turn for turn in range(-1, 5) if rng.chance(0.3))


def with_fault(fault: str, n_tokens: int, segments: list[Segment], rng: Rng) -> tuple[int, list[Segment]]:
    """A copy of a valid buffer of at least two segments with one fault. Every result must be refused."""
    segments = list(segments)
    i = rng.below(len(segments))
    chosen = segments[i]
    if fault == "more_tokens":
        return n_tokens + 1 + rng.below(3), segments
    if fault == "fewer_tokens":
        return n_tokens - 1 - rng.below(min(3, n_tokens)), segments
    if fault == "moved_start":
        segments[i] = Segment(chosen.kind, chosen.start + rng.choice((-1, 1)), chosen.end, chosen.turn)
    elif fault == "moved_end":
        segments[i] = Segment(chosen.kind, chosen.start, chosen.end + rng.choice((-1, 1)), chosen.turn)
    elif fault == "dropped":
        del segments[i]
    elif fault == "doubled":
        segments.insert(i, chosen)
    elif fault == "swapped":
        j = rng.below(len(segments) - 1)
        segments[j], segments[j + 1] = segments[j + 1], segments[j]
    elif fault == "unknown_kind":
        segments[i] = Segment(rng.choice(("assistant", "Sampled", "eos", "")), chosen.start, chosen.end,
                              chosen.turn)
    elif fault == "empty_segment":
        segments.insert(i, Segment(chosen.kind, chosen.start, chosen.start, chosen.turn))
    else:
        raise KeyError(fault)
    return n_tokens, segments


def generated_inputs() -> list[Input]:
    """N_VALID valid buffers, then N_PER_FAULT buffers for each fault. The same list on every call."""
    rng = Rng(20261117)
    inputs: list[Input] = []
    for _ in range(N_VALID):
        n_tokens, segments = valid_buffer(rng)
        inputs.append((n_tokens, segments, exclude_set(rng)))
    for fault in FAULTS:
        for _ in range(N_PER_FAULT):
            n_tokens, segments = with_fault(fault, *valid_buffer(rng, min_segments=2), rng)
            inputs.append((n_tokens, segments, exclude_set(rng)))
    return inputs


def result_of(function, n_tokens: int, segments: list[Segment], exclude_turns: frozenset[int]) -> object:
    """What a call gives: the mask, or the name of the error that it raised."""
    try:
        return function(n_tokens, list(segments), exclude_turns)
    except Exception as error:                      # any error is a result here, so that it can be compared
        return type(error).__name__


def is_mask(result: object) -> bool:
    return type(result) is list and all(type(weight) is int for weight in result)


def disagreements(other, main=main_mask) -> list[tuple]:
    """The generated inputs on which `other` does not give the result of the main version."""
    wrong = []
    for args in generated_inputs():
        got, wanted = result_of(other, *args), result_of(main, *args)
        same = got == wanted if isinstance(got, (list, str)) else False
        if not same or not (is_mask(got) or got == "ValueError"):
            wrong.append((args, got, wanted))
    return wrong


def test_the_second_mask_agrees_with_the_main_version():
    skip_unless_second_is_written("build_loss_mask")
    wrong = disagreements(impl.second_version("build_loss_mask"))
    assert wrong == [], f"{len(wrong)} of {len(generated_inputs())} inputs differ; the first one: {wrong[0]}"


# --- the comparison itself, tested today ---------------------------------------------------------
def test_the_generated_inputs_are_what_they_claim_to_be():
    inputs = generated_inputs()
    assert len(inputs) == N_VALID + len(FAULTS) * N_PER_FAULT
    assert inputs == generated_inputs()                               # the same inputs on every run
    results = [result_of(main_mask, *args) for args in inputs]
    assert all(is_mask(result) for result in results[:N_VALID])       # every valid buffer gets a mask
    assert results[N_VALID:] == ["ValueError"] * (len(FAULTS) * N_PER_FAULT)     # every fault is refused
    masks = results[:N_VALID]
    assert sum(1 for mask in masks if 0 in mask and 1 in mask) > 300
    assert any(mask == [] for mask in masks)
    kinds = {segment.kind for _, segments, _ in inputs[:N_VALID] for segment in segments}
    assert kinds == set(SEGMENT_KINDS)
    # Excluding turns changes the mask for many inputs, and leaves it alone for many.
    changed = sum(1 for (n, segments, exclude), mask in zip(inputs, masks)
                  if mask != main_mask(n, segments, frozenset()))
    assert 150 < changed < N_VALID - 100


def test_each_fault_changes_the_buffer():
    rng = Rng(5)
    for fault in FAULTS:
        n_tokens, segments = valid_buffer(rng, min_segments=2)
        assert with_fault(fault, n_tokens, segments, rng) != (n_tokens, segments)


def test_the_main_version_agrees_with_itself():
    assert disagreements(main_mask) == []


def _weights(segments: list[Segment], in_loss) -> list[int]:
    """A mask without any check: the weight of each segment is decided by `in_loss(position, segment)`."""
    return [1 if in_loss(index, segment) else 0
            for index, segment in enumerate(segments) for _ in range(segment.start, segment.end)]


def forced_token_counts(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: the token that the harness added is trained on."""
    main_mask(n_tokens, segments, exclude_turns)
    kinds = ("sampled", "forced_eos")
    return _weights(segments, lambda _, s: s.kind in kinds and s.turn not in exclude_turns)


def suffix_counts(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: the tool results are trained on."""
    main_mask(n_tokens, segments, exclude_turns)
    kinds = ("sampled", "suffix")
    return _weights(segments, lambda _, s: s.kind in kinds and s.turn not in exclude_turns)


def all_but_the_prompt(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: everything after the first prompt is trained on."""
    main_mask(n_tokens, segments, exclude_turns)
    return _weights(segments, lambda _, s: s.kind != "prompt" and s.turn not in exclude_turns)


def exclusion_ignored(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: exclude_turns is not read."""
    return main_mask(n_tokens, segments)


def exclusion_turned_round(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: only the excluded turns are kept."""
    main_mask(n_tokens, segments, exclude_turns)
    return _weights(segments, lambda _, s: s.kind == "sampled" and s.turn in exclude_turns)


def exclusion_by_position(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: the k-th sampled segment is taken for turn k; the segment's own turn is not read."""
    main_mask(n_tokens, segments, exclude_turns)
    sampled = [index for index, segment in enumerate(segments) if segment.kind == "sampled"]
    return _weights(segments,
                    lambda index, s: s.kind == "sampled" and sampled.index(index) not in exclude_turns)


def no_checks(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: nothing is refused."""
    weights = [0] * n_tokens
    for segment in segments:
        if segment.kind == "sampled" and segment.turn not in exclude_turns:
            for token in range(max(0, segment.start), min(segment.end, n_tokens)):
                weights[token] = 1
    return weights


def unknown_kind_is_weight_zero(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: a kind that does not exist is taken for a part of the prompt."""
    known = [Segment(s.kind if s.kind in SEGMENT_KINDS else "prompt", s.start, s.end, s.turn)
             for s in segments]
    return main_mask(n_tokens, known, exclude_turns)


def empty_segments_allowed(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: a segment without a token is passed over."""
    for segment in segments:
        if segment.kind not in SEGMENT_KINDS:
            raise ValueError(segment.kind)
    return main_mask(n_tokens, [s for s in segments if s.start != s.end], exclude_turns)


def length_not_checked(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: n_tokens is not compared with the end of the last segment."""
    return main_mask(segments[-1].end if segments else 0, segments, exclude_turns)


def true_and_false(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: booleans, not whole numbers."""
    return [weight == 1 for weight in main_mask(n_tokens, segments, exclude_turns)]


def other_error(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: a bad buffer raises an error that is not ValueError."""
    try:
        return main_mask(n_tokens, segments, exclude_turns)
    except ValueError as error:
        raise AssertionError("bad segments") from error


WRONG_FUNCTIONS = [forced_token_counts, suffix_counts, all_but_the_prompt, exclusion_ignored,
                   exclusion_turned_round, exclusion_by_position, no_checks, unknown_kind_is_weight_zero,
                   empty_segments_allowed, length_not_checked, true_and_false, other_error]


@pytest.mark.parametrize("wrong_function", WRONG_FUNCTIONS, ids=[f.__name__ for f in WRONG_FUNCTIONS])
def test_the_comparison_catches_wrong_functions(wrong_function):
    assert len(disagreements(wrong_function)) >= 20


# ---------------------------------------------------------------------------
# Edge inputs: valid buffers of special shapes, and ten more faults
# ---------------------------------------------------------------------------
EDGE_SHAPES = ("one_segment", "one_kind_only", "long_segments", "many_segments", "one_token_each",
               "same_turn_everywhere", "negative_turns", "everything_excluded")
N_PER_SHAPE = 60
EDGE_FAULTS = ("negative_n_tokens", "negative_n_tokens_no_segments", "tokens_without_segments",
               "segments_without_tokens", "starts_after_zero", "starts_before_zero", "gap", "overlap",
               "reversed_segment", "empty_segment_at_an_end", "misspelled_kind", "two_faults")
N_PER_EDGE_FAULT = 50
MISSPELLED = ("SAMPLED", " sampled", "sampled ", "sampled\n", "samples", "forced-eos", "forcedeos",
              "suffix_", "prompt.", "Prompt")


def tiled(lengths: list[int], kinds: list[str], turns: list[int]) -> tuple[int, list[Segment]]:
    """Segments of the given lengths, one after the other from token 0."""
    segments, position = [], 0
    for length, kind, turn in zip(lengths, kinds, turns):
        segments.append(Segment(kind, position, position + length, turn))
        position += length
    return position, segments


def edge_buffer(shape: str, rng: Rng) -> Input:
    """A valid buffer of the named shape, with the turns to exclude."""
    if shape == "one_segment":
        return *tiled([1 + rng.below(4)], [rng.choice(SEGMENT_KINDS)], [rng.below(3) - 1]), exclude_set(rng)
    count = 2 + rng.below(7)
    lengths = [1 + rng.below(5) for _ in range(count)]
    kinds = [rng.choice(("sampled",) + SEGMENT_KINDS) for _ in range(count)]
    turns = [rng.below(5) - 1 for _ in range(count)]
    exclude = exclude_set(rng)
    if shape == "one_kind_only":
        kinds = [rng.choice(SEGMENT_KINDS)] * count
    elif shape == "long_segments":
        lengths = [200 + rng.below(1800) for _ in range(count)]
    elif shape == "many_segments":
        count = 150 + rng.below(100)
        lengths = [1 + rng.below(3) for _ in range(count)]
        kinds = [rng.choice(("sampled",) + SEGMENT_KINDS) for _ in range(count)]
        turns = [rng.below(60) - 1 for _ in range(count)]
        exclude = frozenset(turn for turn in range(-1, 60) if rng.chance(0.3))
    elif shape == "one_token_each":
        lengths = [1] * count
    elif shape == "same_turn_everywhere":
        turns = [rng.below(3)] * count
        kinds = ["sampled"] * count
    elif shape == "negative_turns":
        turns = [-1 - rng.below(3) for _ in range(count)]
        exclude = frozenset(turn for turn in range(-3, 0) if rng.chance(0.4))
    elif shape == "everything_excluded":
        exclude = frozenset(turns)
    else:
        raise KeyError(shape)
    return *tiled(lengths, kinds, turns), exclude


def moved(segments: list[Segment], first: int, by: int) -> list[Segment]:
    """The segments from position `first` on, moved by `by` tokens; the ones before stay."""
    return segments[:first] + [Segment(s.kind, s.start + by, s.end + by, s.turn) for s in segments[first:]]


def with_edge_fault(fault: str, n_tokens: int, segments: list[Segment],
                    rng: Rng) -> tuple[int, list[Segment]]:
    """A copy of a valid buffer of at least two segments with the named fault. It must be refused."""
    i = rng.below(len(segments))
    later = 1 + rng.below(len(segments) - 1)                 # a segment that is not the first one
    chosen = segments[i]
    if fault == "negative_n_tokens":
        return -1 - rng.below(3), list(segments)
    if fault == "negative_n_tokens_no_segments":
        return -1 - rng.below(3), []
    if fault == "tokens_without_segments":
        return n_tokens, []
    if fault == "segments_without_tokens":
        return 0, list(segments)
    if fault == "starts_after_zero":                         # the whole list moved up: it ends at n_tokens
        by = 1 + rng.below(3)
        return n_tokens + by, moved(segments, 0, by)
    if fault == "starts_before_zero":
        return n_tokens - 1, moved(segments, 0, -1)
    if fault == "gap":                                       # one token between two segments belongs to none
        return n_tokens + 1, moved(segments, later, 1)
    if fault == "overlap":                                   # one token belongs to two segments
        return n_tokens - 1, moved(segments, later, -1)
    if fault == "empty_segment_at_an_end":                   # before the first, after the last, or alone
        where = rng.below(3)
        if where == 0:
            return n_tokens, [Segment(chosen.kind, 0, 0, chosen.turn)] + list(segments)
        if where == 1:
            return n_tokens, list(segments) + [Segment(chosen.kind, n_tokens, n_tokens, chosen.turn)]
        return 0, [Segment(chosen.kind, 0, 0, chosen.turn)]
    changed = list(segments)
    if fault == "reversed_segment":
        changed[i] = Segment(chosen.kind, chosen.end, chosen.start, chosen.turn)
    elif fault == "misspelled_kind":
        changed[i] = Segment(rng.choice(MISSPELLED), chosen.start, chosen.end, chosen.turn)
    elif fault == "two_faults":
        changed[i] = Segment(rng.choice(MISSPELLED), chosen.start, chosen.end, chosen.turn)
        return n_tokens + 1 + rng.below(2), moved(changed, later, 1)
    else:
        raise KeyError(fault)
    return n_tokens, changed


@functools.lru_cache(maxsize=None)
def edge_inputs() -> list[Input]:
    """N_PER_SHAPE valid buffers for each shape, then N_PER_EDGE_FAULT buffers for each edge fault."""
    rng = Rng(20261118)
    inputs = [edge_buffer(shape, rng) for shape in EDGE_SHAPES for _ in range(N_PER_SHAPE)]
    for fault in EDGE_FAULTS:
        for _ in range(N_PER_EDGE_FAULT):
            n_tokens, segments = with_edge_fault(fault, *valid_buffer(rng, min_segments=2), rng)
            inputs.append((n_tokens, segments, exclude_set(rng)))
    return inputs


N_EDGE_VALID = len(EDGE_SHAPES) * N_PER_SHAPE


def token_by_token(n_tokens: int, segments: list[Segment], exclude_turns: frozenset[int]) -> list[int]:
    """The mask of a VALID buffer found a third way: for each token on its own, look up the one segment
    that covers it. No checks, no running position."""
    mask = []
    for token in range(n_tokens):
        (cover,) = [s for s in segments if s.start <= token < s.end]
        mask.append(int(cover.kind == "sampled" and cover.turn not in exclude_turns))
    return mask


def disagreements_on(inputs: list[Input], other, main=main_mask) -> list[tuple]:
    """The inputs on which `other` does not give the result of `main`: the same mask of ints, or ValueError
    where `main` raises ValueError."""
    wrong = []
    for args in inputs:
        got, wanted = result_of(other, *args), result_of(main, *args)
        if got != wanted or not (is_mask(got) or got == "ValueError"):
            wrong.append((args, got, wanted))
    return wrong


def test_the_second_mask_agrees_with_the_main_version_on_the_edge_inputs():
    skip_unless_second_is_written("build_loss_mask")
    wrong = disagreements_on(edge_inputs(), impl.second_version("build_loss_mask"))
    assert wrong == [], f"{len(wrong)} of {len(edge_inputs())} inputs differ; the first one: {wrong[0]}"


@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_gives_the_mask_found_token_by_token(version):
    function = function_under_test("build_loss_mask", version)
    valid = generated_inputs()[:N_VALID] + edge_inputs()[:N_EDGE_VALID]
    wrong = disagreements_on(valid, function, main=token_by_token)
    assert wrong == [], f"{len(wrong)} of {len(valid)} masks differ; the first one: {wrong[0]}"
    broken = generated_inputs()[N_VALID:] + edge_inputs()[N_EDGE_VALID:]
    assert [result_of(function, *args) for args in broken] == ["ValueError"] * len(broken)


def test_the_edge_inputs_are_what_they_claim_to_be():
    inputs = edge_inputs()
    assert len(inputs) == N_EDGE_VALID + len(EDGE_FAULTS) * N_PER_EDGE_FAULT == 480 + 600
    assert inputs == edge_inputs.__wrapped__()                        # the same inputs on every run
    results = [result_of(main_mask, *args) for args in inputs]
    masks = results[:N_EDGE_VALID]
    assert all(is_mask(mask) and len(mask) == n for mask, (n, _, _) in zip(masks, inputs))
    assert results[N_EDGE_VALID:] == ["ValueError"] * (len(EDGE_FAULTS) * N_PER_EDGE_FAULT)
    by_shape = {shape: masks[i * N_PER_SHAPE:(i + 1) * N_PER_SHAPE] for i, shape in enumerate(EDGE_SHAPES)}
    assert all(sum(mask) == 0 for mask in by_shape["everything_excluded"])
    assert max(len(mask) for mask in by_shape["long_segments"]) > 5000
    assert all(len(mask) >= 150 for mask in by_shape["many_segments"])
    assert sum(1 for mask in by_shape["same_turn_everywhere"] if sum(mask) == len(mask)) > 10     # all ones
    assert sum(1 for mask in by_shape["same_turn_everywhere"] if sum(mask) == 0) > 10            # all zeros
    assert sum(1 for mask in by_shape["negative_turns"] if 0 < sum(mask)) > 20    # a sampled turn -1 counts
    # Every edge fault but the misspelled kind is a fault of the tiling alone: the kinds are all known.
    for index, fault in enumerate(EDGE_FAULTS):
        start = N_EDGE_VALID + index * N_PER_EDGE_FAULT
        kinds_known = all(s.kind in SEGMENT_KINDS for _, segments, _ in inputs[start:start + N_PER_EDGE_FAULT]
                          for s in segments)
        assert kinds_known is (fault not in ("misspelled_kind", "two_faults")), fault


def negative_length_is_empty(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: a negative n_tokens without segments is taken for an empty buffer."""
    return [] if n_tokens < 0 and not segments else main_mask(n_tokens, segments, exclude_turns)


def only_the_end_is_checked(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: the list may start anywhere and be in any order, if its last segment ends at
    n_tokens and no segment is empty."""
    for segment in segments:
        if segment.kind not in SEGMENT_KINDS or segment.end <= segment.start:
            raise ValueError("bad segment")
    if (segments[-1].end if segments else 0) != n_tokens:
        raise ValueError("bad length")
    weights = [0] * n_tokens
    for segment in segments:
        if segment.kind == "sampled" and segment.turn not in exclude_turns:
            weights[max(0, segment.start):segment.end] = [1] * (segment.end - max(0, segment.start))
    return weights[:n_tokens]


def kinds_without_case(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: "SAMPLED" and " sampled" are taken for "sampled"."""
    cleaned = [Segment(s.kind.strip().lower(), s.start, s.end, s.turn) for s in segments]
    return main_mask(n_tokens, cleaned, exclude_turns)


def negative_turns_never_count(n_tokens, segments, exclude_turns=frozenset()):
    """Wrong on purpose: a sampled segment with a turn below 0 gets weight 0 (the contract does not check
    the turn numbers)."""
    return main_mask(n_tokens, segments, frozenset(exclude_turns) | {s.turn for s in segments if s.turn < 0})


EDGE_WRONG_FUNCTIONS = WRONG_FUNCTIONS + [negative_length_is_empty, only_the_end_is_checked,
                                          kinds_without_case, negative_turns_never_count]


@pytest.mark.parametrize("wrong_function", EDGE_WRONG_FUNCTIONS,
                         ids=[function.__name__ for function in EDGE_WRONG_FUNCTIONS])
def test_the_edge_comparison_catches_wrong_functions(wrong_function):
    assert len(disagreements_on(edge_inputs(), wrong_function)) >= 20


# ---------------------------------------------------------------------------
# Real token buffers (needs the tokenizer files in .cache/hf; no model weights)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_inputs(tokenizer, families, text) -> list[tuple[Input, list[int] | None]]:
    """(input, the scan of the token ids or None) for real buffers of the scripted dev episodes.

    Per episode: no turn excluded, the first turn, the last turn, every turn. For every fourth episode also a
    copy with one turn cut off (a "forced_eos" segment; the scan cannot know such a token, so None), and
    three broken copies of the real segment list."""
    from accessdesk.harness.tokens import TokenLayout, buffer_from_record
    from tests.core.test_tokens import scripted_record
    from tests.helpers.scan_mask import scan_mask

    layout = TokenLayout(tokenizer, text.tool_specs("five"))
    rng = Rng(20261119)
    inputs: list[tuple[Input, list[int] | None]] = []
    episodes = [(family, wording["style"]) for family in families["dev"] for wording in family["wordings"]]
    for number, (family, style) in enumerate(episodes):
        record = scripted_record(family, text, style=style)
        buffer = buffer_from_record(layout, record, family, text)
        n_tokens, segments, n_turns = len(buffer.ids), list(buffer.segments), len(record["turns"])
        scanned = scan_mask(buffer.ids, tokenizer)
        inputs.append(((n_tokens, segments, frozenset()), scanned))
        for exclude in (frozenset({0}), frozenset({n_turns - 1}), frozenset(range(n_turns))):
            inputs.append(((n_tokens, segments, exclude), None))
        if number % 4:
            continue
        cut = rng.below(n_turns)
        turns = list(record["turns"])
        turns[cut] = {**turns[cut], "assistant_text": turns[cut]["assistant_text"][:20], "truncated": True}
        cut_buffer = buffer_from_record(layout, {**record, "turns": turns}, family, text)
        assert "forced_eos" in [segment.kind for segment in cut_buffer.segments]
        for exclude in (frozenset(), frozenset({cut})):
            inputs.append(((len(cut_buffer.ids), list(cut_buffer.segments), exclude), None))
        for fault in ("dropped", "swapped", "fewer_tokens"):
            inputs.append(((*with_fault(fault, n_tokens, segments, rng), frozenset()), None))
    return inputs


@pytest.mark.tokenizer
def test_the_real_buffers_are_many_and_of_every_kind(real_inputs, cfg):
    assert len(real_inputs) == cfg.n_dev * 4 * 4 + cfg.n_dev * 5
    results = [result_of(main_mask, *args) for args, _ in real_inputs]
    assert results.count("ValueError") == cfg.n_dev * 3
    assert sum(1 for _, scanned in real_inputs if scanned is not None) == cfg.n_dev * 4
    assert all(n_tokens > 1000 for (n_tokens, _, _), _ in real_inputs)             # real prompts are long
    kinds = {segment.kind for (_, segments, _), _ in real_inputs for segment in segments}
    assert kinds == set(SEGMENT_KINDS)


@pytest.mark.tokenizer
@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_gives_the_scanned_mask_on_real_buffers(real_inputs, version):
    function = function_under_test("build_loss_mask", version)
    for (n_tokens, segments, exclude), scanned in real_inputs:
        if scanned is not None:
            assert function(n_tokens, segments, exclude) == scanned
        elif result_of(main_mask, n_tokens, segments, exclude) != "ValueError":
            assert function(n_tokens, segments, exclude) == token_by_token(n_tokens, segments, exclude)


@pytest.mark.tokenizer
def test_the_second_mask_agrees_with_the_main_version_on_real_buffers(real_inputs):
    skip_unless_second_is_written("build_loss_mask")
    inputs = [args for args, _ in real_inputs]
    wrong = disagreements_on(inputs, impl.second_version("build_loss_mask"))
    assert wrong == [], f"{len(wrong)} of {len(inputs)} real buffers differ; the first one: {wrong[0][0][0]}"


@pytest.mark.tokenizer
@pytest.mark.parametrize("wrong_function", [forced_token_counts, suffix_counts, all_but_the_prompt,
                                            exclusion_ignored, no_checks, true_and_false],
                         ids=lambda function: function.__name__)
def test_the_real_buffers_catch_wrong_functions(real_inputs, wrong_function):
    assert len(disagreements_on([args for args, _ in real_inputs], wrong_function)) >= 20
