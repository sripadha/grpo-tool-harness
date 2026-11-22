"""verify(): hand-made cases and properties, run on the main version and on the second version.

Every expected answer below was worked out by hand from the contract in the function's docstring
(accessdesk/env/verifier.py), not by running any implementation. Each test takes the function under test as
the argument `verify`: the main version, and the second version of accessdesk/_second once it is written
(until then those tests are skipped).
"""
from __future__ import annotations

import copy

import pytest

from tests.conftest import VERSIONS, function_under_test


@pytest.fixture(params=VERSIONS)
def verify(request):
    """The function under test: the main version, then the second version."""
    return function_under_test("verify", request.param)


def state(access: list, approvals: list, grants: list, resolution, person_id, role_id) -> dict:
    """A state dict. The lists are copied, so no two states share a list."""
    return {"access": [list(pair) for pair in access], "approvals": [list(record) for record in approvals],
            "grants": [list(record) for record in grants],
            "request": {"resolution": resolution, "person_id": person_id, "role_id": role_id}}


# A small world. At the start p1 holds r1 and p3 holds r2. p3 is the manager of p1 and of p2.
START = [["p1", "r1"], ["p3", "r2"]]
P1_GOT_R3 = [["p1", "r1"], ["p1", "r3"], ["p3", "r2"]]      # START plus (p1, r3)
P1_GOT_R2 = [["p1", "r1"], ["p1", "r2"], ["p3", "r2"]]      # START plus (p1, r2)
P2_GOT_R3 = [["p1", "r1"], ["p2", "r3"], ["p3", "r2"]]      # START plus (p2, r3)

# The correct end states of four requests.
GRANTED = state(P1_GOT_R3, [], [["p1", "r3"]], "granted", "p1", "r3")      # p1 asks for r3: grant
PENDING = state(START, [["p1", "r2", "p3"]], [], "pending", "p1", "r2")    # p1 asks for r2 (sensitive)
ALREADY = state(START, [], [], "already", "p1", "r1")                      # p1 asks for r1, which p1 holds
DENIED = state(START, [], [], "denied", "p2", "r3")                        # p2 (inactive) asks for r3

UNTOUCHED = state(START, [], [], None, None, None)
TWO_APPROVALS = [["p1", "r1", "p3"], ["p1", "r2", "p3"]]
TWO_GRANTS = [["p1", "r3"], ["p2", "r3"]]
BOTH_GOT_R3 = [["p1", "r1"], ["p1", "r3"], ["p2", "r3"], ["p3", "r2"]]
SAME_GRANT_TWICE = state(P1_GOT_R3, [], [["p1", "r3"], ["p1", "r3"]], "granted", "p1", "r3")

CASES = [
    # (label, final state, expected state, the right answer)
    # --- passes ---
    ("grant done right", copy.deepcopy(GRANTED), GRANTED, True),
    ("approval done right", copy.deepcopy(PENDING), PENDING, True),
    ("already has access, closed as such", copy.deepcopy(ALREADY), ALREADY, True),
    ("inactive, closed as denied", copy.deepcopy(DENIED), DENIED, True),
    ("access listed in another order",
     state([["p3", "r2"], ["p1", "r3"], ["p1", "r1"]], [], [["p1", "r3"]], "granted", "p1", "r3"),
     GRANTED, True),
    ("two expected approvals, written in the other order",
     state(START, list(reversed(TWO_APPROVALS)), [], "pending", "p1", "r2"),
     state(START, TWO_APPROVALS, [], "pending", "p1", "r2"), True),
    ("two expected grants, written in the other order",
     state(BOTH_GOT_R3, [], list(reversed(TWO_GRANTS)), "granted", "p1", "r3"),
     state(BOTH_GOT_R3, [], TWO_GRANTS, "granted", "p1", "r3"), True),
    ("the same grant expected twice and written twice",
     copy.deepcopy(SAME_GRANT_TWICE), SAME_GRANT_TWICE, True),
    # --- nothing done, or not closed ---
    ("untouched world", UNTOUCHED, GRANTED, False),
    ("untouched world against a denial", UNTOUCHED, DENIED, False),
    ("all the right writes, but never closed",
     state(P1_GOT_R3, [], [["p1", "r3"]], None, None, None), GRANTED, False),
    # --- the request fields ---
    ("closed with the wrong resolution",
     state(P1_GOT_R3, [], [["p1", "r3"]], "pending", "p1", "r3"), GRANTED, False),
    ("closed with the wrong person id",
     state(P1_GOT_R3, [], [["p1", "r3"]], "granted", "p3", "r3"), GRANTED, False),
    ("closed with the wrong role id",
     state(P1_GOT_R3, [], [["p1", "r3"]], "granted", "p1", "r2"), GRANTED, False),
    ("denied for the right person but the wrong role",
     state(START, [], [], "denied", "p2", "r1"), DENIED, False),
    ("already has access, but closed as granted",
     state(START, [], [], "granted", "p1", "r1"), ALREADY, False),
    ("resolution missing, ids right",
     state(START, [], [], None, "p2", "r3"), DENIED, False),
    # --- grants ---
    ("closed as granted with no grant",
     state(START, [], [], "granted", "p1", "r3"), GRANTED, False),
    ("the right grant made twice",
     copy.deepcopy(SAME_GRANT_TWICE), GRANTED, False),
    ("a second, needless grant of another role",
     state([["p1", "r1"], ["p1", "r2"], ["p1", "r3"], ["p3", "r2"]], [], [["p1", "r3"], ["p1", "r2"]],
           "granted", "p1", "r3"), GRANTED, False),
    ("a grant on a role already held, then closed as already",
     state(START, [], [["p1", "r1"]], "already", "p1", "r1"), ALREADY, False),
    ("the role granted to the wrong person",
     state(P2_GOT_R3, [], [["p2", "r3"]], "granted", "p1", "r3"), GRANTED, False),
    ("a grant where a denial was right",
     state(P2_GOT_R3, [], [["p2", "r3"]], "denied", "p2", "r3"), DENIED, False),
    ("a grant record without the access",
     state(START, [], [["p1", "r3"]], "granted", "p1", "r3"), GRANTED, False),
    ("the access without a grant record",
     state(P1_GOT_R3, [], [], "granted", "p1", "r3"), GRANTED, False),
    ("one grant expected twice but written once",
     copy.deepcopy(GRANTED), SAME_GRANT_TWICE, False),
    # --- approvals ---
    ("a sensitive role granted without approval",
     state(P1_GOT_R2, [], [["p1", "r2"]], "granted", "p1", "r2"), PENDING, False),
    ("approval asked from the wrong approver",
     state(START, [["p1", "r2", "p2"]], [], "pending", "p1", "r2"), PENDING, False),
    ("approval asked for the wrong role",
     state(START, [["p1", "r3", "p3"]], [], "pending", "p1", "r2"), PENDING, False),
    ("approval asked twice",
     state(START, [["p1", "r2", "p3"], ["p1", "r2", "p3"]], [], "pending", "p1", "r2"), PENDING, False),
    ("closed as pending with no approval record",
     state(START, [], [], "pending", "p1", "r2"), PENDING, False),
    ("approval asked and the role granted as well",
     state(P1_GOT_R2, [["p1", "r2", "p3"]], [["p1", "r2"]], "pending", "p1", "r2"), PENDING, False),
    ("a needless approval before a right grant",
     state(P1_GOT_R3, [["p1", "r3", "p3"]], [["p1", "r3"]], "granted", "p1", "r3"), GRANTED, False),
    ("a needless approval before a right denial",
     state(START, [["p2", "r3", "p3"]], [], "denied", "p2", "r3"), DENIED, False),
    # --- access ---
    ("a role lost from access",
     state([["p1", "r1"]], [], [], "already", "p1", "r1"), ALREADY, False),
    ("another person's access changed",
     state([["p1", "r1"], ["p2", "r1"], ["p3", "r2"]], [], [], "already", "p1", "r1"), ALREADY, False),
]


@pytest.mark.parametrize("label, final, expected, answer", CASES, ids=[case[0] for case in CASES])
def test_verify_hand_made(verify, label, final, expected, answer):
    assert verify(final, expected) is answer


def test_the_hand_made_cases_hold_passes_and_fails_of_every_part():
    answers = [case[3] for case in CASES]
    assert answers.count(True) >= 8 and answers.count(False) >= 25
    # For each part, at least one failing case differs from its expected state in that part alone.
    alone = {"access": 0, "approvals": 0, "grants": 0, "request": 0}
    for _, final, expected, answer in CASES:
        differing = [part for part in alone if sorted_part(final, part) != sorted_part(expected, part)]
        if not answer and len(differing) == 1:
            alone[differing[0]] += 1
    assert all(count >= 2 for count in alone.values()), alone


def sorted_part(state_dict: dict, part: str) -> object:
    return state_dict[part] if part == "request" else sorted(state_dict[part])


def test_verify_returns_a_bool(verify):
    assert type(verify(copy.deepcopy(GRANTED), GRANTED)) is bool
    assert type(verify(copy.deepcopy(UNTOUCHED), GRANTED)) is bool


@pytest.mark.parametrize("expected", [GRANTED, PENDING, ALREADY, DENIED])
def test_every_state_matches_itself_and_its_copy(verify, expected):
    assert verify(expected, expected) is True
    assert verify(copy.deepcopy(expected), expected) is True


@pytest.mark.parametrize("field, value", [("resolution", None), ("resolution", "denied"), ("person_id", "p3"),
                                          ("person_id", None), ("role_id", "r1"), ("role_id", None)])
def test_each_request_field_is_checked_on_its_own(verify, field, value):
    final = copy.deepcopy(GRANTED)
    final["request"][field] = value
    assert verify(final, GRANTED) is False


@pytest.mark.parametrize("part, record", [("access", ["p2", "r1"]), ("approvals", ["p1", "r3", "p3"]),
                                          ("grants", ["p1", "r3"]), ("grants", ["p2", "r1"])])
def test_one_record_more_or_less_fails_in_both_directions(verify, part, record):
    more = copy.deepcopy(GRANTED)
    more[part].append(record)
    assert verify(more, GRANTED) is False        # the episode wrote one record too many
    assert verify(GRANTED, more) is False        # the episode wrote one record too few


def test_verify_is_symmetric_on_the_hand_made_cases(verify):
    for _, final, expected, answer in CASES:
        assert verify(expected, final) is answer


def test_a_record_may_be_a_list_or_a_tuple(verify):
    # The same end state, once with records as lists (as JSON gives them) and once as tuples.
    as_tuples = {"access": [tuple(pair) for pair in PENDING["access"]], "approvals": [("p1", "r2", "p3")],
                 "grants": [], "request": dict(PENDING["request"])}
    assert verify(as_tuples, PENDING) is True
    assert verify(PENDING, as_tuples) is True
    # A tuple with another approver is still another record.
    as_tuples["approvals"] = [("p1", "r2", "p2")]
    assert verify(as_tuples, PENDING) is False


def test_verify_changes_neither_argument(verify):
    final = state([["p3", "r2"], ["p1", "r3"], ["p1", "r1"]], [], [["p1", "r3"]], "granted", "p1", "r3")
    expected = copy.deepcopy(GRANTED)
    final_before, expected_before = copy.deepcopy(final), copy.deepcopy(expected)
    assert verify(final, expected) is True
    assert final == final_before and expected == expected_before


def test_a_state_without_one_of_its_parts_is_an_error_and_not_a_fail(verify):
    broken = copy.deepcopy(GRANTED)
    del broken["grants"]
    with pytest.raises(KeyError):
        verify(broken, GRANTED)


# --- cases that the comparison of the two versions decided (2026-11-17) -------------------------------------
# The contract said only: a missing part raises KeyError "when two states agree in everything else". The
# main version answered False when it met another difference first; the second version raised KeyError.
# Decided: a missing part or request field is always a KeyError, in either argument, whatever else differs.
def without(state_dict: dict, part: str) -> dict:
    """A copy of the state without one of its four parts."""
    copy_ = copy.deepcopy(state_dict)
    del copy_[part]
    return copy_


def without_field(state_dict: dict, field: str) -> dict:
    """A copy of the state without one of the three request fields."""
    copy_ = copy.deepcopy(state_dict)
    del copy_["request"][field]
    return copy_


@pytest.mark.parametrize("part", ["access", "approvals", "grants", "request"])
def test_a_missing_part_is_an_error_whatever_else_the_states_hold(verify, part):
    # DENIED and GRANTED differ in access, in grants and in all three request fields. Before the decision the
    # main version returned False here for every part but "access", because "access" already differed.
    for final, expected in ((without(DENIED, part), GRANTED), (DENIED, without(GRANTED, part)),
                            (without(GRANTED, part), GRANTED), (GRANTED, without(GRANTED, part)),
                            (without(GRANTED, part), without(GRANTED, part))):
        with pytest.raises(KeyError):
            verify(final, expected)


@pytest.mark.parametrize("field", ["resolution", "person_id", "role_id"])
def test_a_missing_request_field_is_an_error_whatever_else_the_states_hold(verify, field):
    less = without_field(GRANTED, field)
    for final, expected in ((without_field(DENIED, field), GRANTED), (DENIED, less), (less, GRANTED),
                            (GRANTED, less), (without_field(UNTOUCHED, field), GRANTED)):
        with pytest.raises(KeyError):
            verify(final, expected)


def test_a_missing_part_is_not_an_empty_part_and_a_missing_field_is_not_none(verify):
    # ALREADY has no approvals and no grants; UNTOUCHED has None in all three request fields.
    for part in ("approvals", "grants"):
        with pytest.raises(KeyError):
            verify(without(ALREADY, part), ALREADY)
    for field in ("resolution", "person_id", "role_id"):
        with pytest.raises(KeyError):
            verify(without_field(UNTOUCHED, field), UNTOUCHED)


def test_keys_beyond_the_four_parts_and_the_three_fields_are_not_read(verify):
    final = copy.deepcopy(GRANTED)
    final["closed"] = True                       # a key that no state has
    final["request"]["note"] = "x"               # a request field that no state has
    assert verify(final, GRANTED) is True
    assert verify(GRANTED, final) is True
    assert verify(final, PENDING) is False


def test_start_is_the_access_of_the_untouched_world():
    assert state(START, [], [], None, None, None)["access"] == [["p1", "r1"], ["p3", "r2"]]
