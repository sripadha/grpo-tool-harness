"""The second version of verify() against the main version, on many generated pairs of states.

The comparison is skipped while the second version (accessdesk/_second/verifier.py) is not written.
The generator of the pairs and the comparison itself are tested here in any case, with wrong functions.

Two sets of inputs:
  - 3000 random pairs (a state and a slightly changed copy, or two independent states). The two versions
    must give the same answer.
  - about 9,800 pairs with exactly one change (one record more, one less, one twice, one id changed, order
    only, one request field, one missing part, ...). Here the right answer is known from the change that
    was made, so both versions are compared with that answer and not only with each other: a mistake that
    the two versions share would show.
"""
from __future__ import annotations

import copy
import functools

import pytest

from accessdesk import impl
from accessdesk.env.verifier import verify as main_verify
from accessdesk.rng import Rng
from tests.conftest import VERSIONS, function_under_test, skip_unless_second_is_written

PEOPLE = ("p1", "p2", "p3")
ROLES = ("r1", "r2", "r3")
RESOLUTIONS_OR_NONE = ("granted", "pending", "already", "denied", None)
N_PAIRS = 3000


def random_state(rng: Rng) -> dict:
    """A valid state dict: access without repeats; approvals and grants may repeat; any request."""
    pairs = [[person, role] for person in PEOPLE for role in ROLES]
    closed = rng.chance(0.8)
    return {
        "access": rng.sample(pairs, rng.below(5)),
        "approvals": [rng.choice(pairs) + [rng.choice(PEOPLE)] for _ in range(rng.below(3))],
        "grants": [list(rng.choice(pairs)) for _ in range(rng.below(3))],
        "request": {"resolution": rng.choice(RESOLUTIONS_OR_NONE) if closed else None,
                    "person_id": rng.choice(PEOPLE) if closed else None,
                    "role_id": rng.choice(ROLES) if closed else None},
    }


def changed_a_little(state: dict, rng: Rng) -> dict:
    """A copy with every list in a new order and, half of the time, one small change that matters."""
    other = copy.deepcopy(state)
    for part in ("access", "approvals", "grants"):
        other[part] = rng.shuffle(other[part])
    kind = rng.below(9)
    if kind == 0 and other["grants"]:
        other["grants"].append(list(other["grants"][0]))            # a record a second time
    elif kind == 1 and other["approvals"]:
        other["approvals"].pop()                                    # a record less
    elif kind == 2:
        other["request"]["resolution"] = rng.choice(RESOLUTIONS_OR_NONE)
    elif kind == 3:
        other["request"]["person_id"] = rng.choice(PEOPLE)
    elif kind == 4:                                                 # tuples for lists: the same records
        for part in ("access", "approvals", "grants"):
            other[part] = [tuple(record) for record in other[part]]
    return other


def generated_pairs(seed: int = 20261116) -> list[tuple[dict, dict]]:
    """Half of the pairs are a state and a slightly changed copy; the others are two independent states."""
    rng = Rng(seed)
    pairs = []
    for i in range(N_PAIRS):
        expected = random_state(rng)
        final = changed_a_little(expected, rng) if i % 2 == 0 else random_state(rng)
        pairs.append((final, expected))
    return pairs


def disagreements(other, main=main_verify) -> list[tuple[dict, dict]]:
    """The generated pairs on which `other` does not give the answer of the main version."""
    return [(final, expected) for final, expected in generated_pairs()
            if other(copy.deepcopy(final), copy.deepcopy(expected)) is not main(final, expected)]


def test_the_second_verify_agrees_with_the_main_version():
    skip_unless_second_is_written("verify")
    wrong = disagreements(impl.second_version("verify"))
    assert wrong == [], f"{len(wrong)} of {N_PAIRS} pairs differ; the first one: {wrong[0]}"


# --- the comparison itself, tested today ---------------------------------------------------------
def test_generated_pairs_hold_many_passes_and_many_fails():
    answers = [main_verify(final, expected) for final, expected in generated_pairs()]
    assert len(answers) == N_PAIRS
    assert answers.count(True) > 500 and answers.count(False) > 1500
    assert generated_pairs() == generated_pairs()                    # the same pairs on every run


def test_the_main_version_agrees_with_itself():
    assert disagreements(main_verify) == []


def as_sets(final: dict, expected: dict) -> bool:
    """Wrong on purpose: forgets the count of repeated records."""
    same = all({tuple(r) for r in final[part]} == {tuple(r) for r in expected[part]}
               for part in ("access", "approvals", "grants"))
    return same and final["request"] == expected["request"]


def in_order(final: dict, expected: dict) -> bool:
    """Wrong on purpose: order matters."""
    return final == expected


def without_person(final: dict, expected: dict) -> bool:
    """Wrong on purpose: forgets the request's person id."""
    return main_verify({**final, "request": {**final["request"], "person_id": None}},
                       {**expected, "request": {**expected["request"], "person_id": None}})


def lists_only(final: dict, expected: dict) -> bool:
    """Wrong on purpose: a record written as a tuple is not the record written as a list."""
    same = all(sorted(final[part], key=list) == sorted(expected[part], key=list)
               for part in ("access", "approvals", "grants"))
    return same and final["request"] == expected["request"]


@pytest.mark.parametrize("wrong_function", [as_sets, in_order, without_person, lists_only,
                                            lambda final, expected: True, lambda final, expected: False])
def test_the_comparison_catches_wrong_functions(wrong_function):
    assert len(disagreements(wrong_function)) >= 20


# ---------------------------------------------------------------------------
# Pairs with exactly one change. The right answer follows from the change, not from a version of verify().
# ---------------------------------------------------------------------------
N_BASES = 150
RECORD_PARTS = ("access", "approvals", "grants")
ALL_PARTS = RECORD_PARTS + ("request",)
FIELDS = ("resolution", "person_id", "role_id")
FIELD_VALUES = {"resolution": RESOLUTIONS_OR_NONE, "person_id": PEOPLE + (None,), "role_id": ROLES + (None,)}
OneChange = tuple[str, dict, dict, object]          # (name of the change, final, expected, the right answer)


def _another(value: object, values: tuple, rng: Rng) -> object:
    """A value of `values` that is not `value`."""
    return rng.choice([other for other in values if other != value])


def one_changes_of(expected: dict, rng: Rng) -> list[tuple[str, dict, object]]:
    """Copies of a valid state with one change each: (name, the changed copy, the right answer).

    The right answer is True (nothing that counts was changed), False (one thing that counts was changed)
    or "KeyError" (a part or a request field is missing)."""
    changes: list[tuple[str, dict, object]] = []

    def changed(name: str, answer: object) -> dict:
        final = copy.deepcopy(expected)
        changes.append((name, final, answer))
        return final

    changed("nothing", True)
    final = changed("order only", True)
    for part in RECORD_PARTS:
        final[part] = list(reversed(rng.shuffle(final[part])))
    final = changed("records as tuples", True)
    for part in RECORD_PARTS:
        final[part] = [tuple(record) for record in final[part]]
    final = changed("keys that no state has", True)
    final["closed"] = True
    final["request"]["note"] = "x"

    all_pairs = [[person, role] for person in PEOPLE for role in ROLES]
    for part in RECORD_PARTS:
        # One record more. In access it is a pair that the state does not hold (a state never holds a pair
        # twice); in approvals and grants it is any record, so sometimes a second copy of one that is there.
        if part == "access":
            record = rng.choice([pair for pair in all_pairs if pair not in expected["access"]])
        else:
            record = rng.choice(all_pairs) + ([rng.choice(PEOPLE)] if part == "approvals" else [])
        changed(f"one record more in {part}", False)[part].append(list(record))
        if not expected[part]:
            continue
        index = rng.below(len(expected[part]))
        changed(f"one record less in {part}", False)[part].pop(index)
        if part != "access":
            final = changed(f"one record twice in {part}", False)
            final[part].insert(rng.below(len(final[part]) + 1), list(final[part][index]))
        position = rng.below(len(expected[part][index]))
        final = changed(f"one id changed in {part}", False)
        final[part][index][position] = _another(final[part][index][position],
                                                ROLES if position == 1 else PEOPLE, rng)
        final = changed(f"person and role of one record swapped in {part}", False)
        final[part][index][0], final[part][index][1] = final[part][index][1], final[part][index][0]
    if expected["approvals"]:
        final = changed("an approval without its approver", False)
        final["approvals"][0] = final["approvals"][0][:2]

    for field in FIELDS:
        final = changed(f"request field {field} changed", False)
        final["request"][field] = _another(final["request"][field], FIELD_VALUES[field], rng)
    final = changed("the request never closed", expected["request"] == dict.fromkeys(FIELDS))
    final["request"] = dict.fromkeys(FIELDS)

    for part in ALL_PARTS:
        del changed(f"part {part} missing", "KeyError")[part]
        final = changed(f"part {part} missing and another part changed", "KeyError")
        del final[part]
        if part == "grants":
            final["access"].append(["p9", "r9"])
        else:
            final["grants"].append(["p9", "r9"])
    for field in FIELDS:
        del changed(f"request field {field} missing", "KeyError")["request"][field]
        final = changed(f"request field {field} missing and another part changed", "KeyError")
        del final["request"][field]
        final["grants"].append(["p9", "r9"])
    return changes


@functools.lru_cache(maxsize=None)
def one_change_pairs(seed: int = 20261117) -> list[OneChange]:
    """Every one-change copy of N_BASES random states, each as (final, expected) and as (expected, final):
    the right answer is the same in both directions. Built once; `answer_of` hands out copies."""
    rng = Rng(seed)
    pairs: list[OneChange] = []
    for _ in range(N_BASES):
        expected = random_state(rng)
        for name, final, answer in one_changes_of(expected, rng):
            pairs.append((name, final, expected, answer))
            pairs.append((name + " (arguments swapped)", copy.deepcopy(expected), final, answer))
    return pairs


def answer_of(function, final: dict, expected: dict) -> object:
    """What a call gives: True, False, or "KeyError". Any other error is not caught."""
    try:
        return function(copy.deepcopy(final), copy.deepcopy(expected))
    except KeyError:
        return "KeyError"


def wrong_answers(function) -> list[tuple]:
    """The one-change pairs on which `function` does not give the right answer."""
    wrong = []
    for name, final, expected, answer in one_change_pairs():
        got = answer_of(function, final, expected)
        if type(got) is not type(answer) or got != answer:          # 1 is not True, 0 is not False
            wrong.append((name, final, expected, answer, got))
    return wrong


@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_gives_the_right_answer_on_every_one_change_pair(version):
    wrong = wrong_answers(function_under_test("verify", version))
    assert wrong == [], f"{len(wrong)} of {len(one_change_pairs())} answers are wrong; the first: {wrong[0]}"


def test_the_one_change_pairs_hold_every_kind_of_change_many_times():
    pairs = one_change_pairs()
    assert len(pairs) > 9_000
    assert pairs == one_change_pairs.__wrapped__()                   # the same pairs on every run
    names = [name for name, _, _, _ in pairs if not name.endswith("(arguments swapped)")]
    assert len(names) * 2 == len(pairs)
    assert len(set(names)) == 4 + 14 + 1 + 4 + 8 + 6                 # every change above, by name
    for name in set(names):
        assert names.count(name) >= 80, name                         # the rarest needs a non-empty part
    answers = [answer for _, _, _, answer in pairs]
    assert answers.count(True) > 1000 and answers.count(False) > 4000 and answers.count("KeyError") > 4000
    # Only a state that never closed can still match after "the request never closed".
    never_closed = [answer for name, _, _, answer in pairs if name == "the request never closed"]
    assert 10 < never_closed.count(True) < never_closed.count(False)


def without_role(final: dict, expected: dict) -> bool:
    """Wrong on purpose: forgets the request's role id."""
    return main_verify({**final, "request": {**final["request"], "role_id": None}},
                       {**expected, "request": {**expected["request"], "role_id": None}})


def missing_is_empty(final: dict, expected: dict) -> bool:
    """Wrong on purpose: a missing part is taken for an empty list, a missing field for None."""
    def filled(state: dict) -> dict:
        request = {field: state.get("request", {}).get(field) for field in FIELDS}
        return {**{part: state.get(part, []) for part in RECORD_PARTS}, "request": request}
    return main_verify(filled(final), filled(expected))


def stops_at_the_first_difference(final: dict, expected: dict) -> bool:
    """Wrong since the decision of 2026-11-17 (this is what the main version did before): a missing part
    is noticed only if no earlier part differs."""
    for part in RECORD_PARTS:
        if sorted(tuple(r) for r in final[part]) != sorted(tuple(r) for r in expected[part]):
            return False
    return all(final["request"][field] == expected["request"][field] for field in FIELDS)


def counts_only(final: dict, expected: dict) -> bool:
    """Wrong on purpose: compares how many records each part holds, and the request."""
    return all(len(final[part]) == len(expected[part]) for part in RECORD_PARTS) \
        and final["request"] == expected["request"]


ONE_CHANGE_WRONG = [as_sets, in_order, without_person, without_role, lists_only, missing_is_empty,
                    stops_at_the_first_difference, counts_only, lambda final, expected: True,
                    lambda final, expected: False]


@pytest.mark.parametrize("wrong_function", ONE_CHANGE_WRONG)
def test_the_one_change_pairs_catch_wrong_functions(wrong_function):
    assert len(wrong_answers(wrong_function)) >= 20
