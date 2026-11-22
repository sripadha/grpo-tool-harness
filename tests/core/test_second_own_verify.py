"""Hand-made tests of the SECOND version of verify(), written from the contract only.

Every expected value is worked out by hand from the contract, in a comment beside the case.
"""
from __future__ import annotations

import copy

import pytest

from accessdesk._second.verifier import verify


def base_state() -> dict:
    """An "approval" end state: p1 asked for sensitive r2, manager m1 asked, request pending."""
    return {
        "access": [["p1", "r1"], ["m1", "r2"]],
        "approvals": [["p1", "r2", "m1"]],
        "grants": [],
        "request": {"resolution": "pending", "person_id": "p1", "role_id": "r2"},
    }


def grant_state() -> dict:
    """A "grant" end state: p1 got r3 by one grant call, request granted."""
    return {
        "access": [["p1", "r1"], ["p1", "r3"]],
        "approvals": [],
        "grants": [["p1", "r3"]],
        "request": {"resolution": "granted", "person_id": "p1", "role_id": "r3"},
    }


# --- equal states ------------------------------------------------------------------------------------------

def test_identical_states_pass():
    # All four parts are equal -> True.
    assert verify(base_state(), base_state()) is True
    assert verify(grant_state(), grant_state()) is True


def test_order_of_records_does_not_matter():
    # access: same two pairs in another order; approvals: same multiset {(p1,r2,m1) x1, (p2,r2,m1) x1}
    # in another order -> all parts match -> True.
    final = base_state()
    expected = base_state()
    final["access"] = [["m1", "r2"], ["p1", "r1"]]
    final["approvals"] = [["p2", "r2", "m1"], ["p1", "r2", "m1"]]
    expected["approvals"] = [["p1", "r2", "m1"], ["p2", "r2", "m1"]]
    assert verify(final, expected) is True


def test_list_and_tuple_records_are_the_same():
    # Contract: ["p1", "r1"] and ("p1", "r1") are the same record -> True.
    final = grant_state()
    final["access"] = [("p1", "r1"), ("p1", "r3")]
    final["grants"] = [("p1", "r3")]
    assert verify(final, grant_state()) is True


def test_tuple_parts_are_accepted():
    # A part given as a tuple of records holds the same records -> True.
    final = grant_state()
    final["access"] = (("p1", "r3"), ["p1", "r1"])
    assert verify(final, grant_state()) is True


def test_empty_parts_on_both_sides_match():
    # "already" state: no writes at all; empty == empty in every part -> True.
    state = {"access": [["p1", "r1"]], "approvals": [], "grants": [],
             "request": {"resolution": "already", "person_id": "p1", "role_id": "r1"}}
    assert verify(state, copy.deepcopy(state)) is True


def test_extra_top_level_key_is_ignored():
    # Only the four parts are compared (my choice for a contract gap) -> True.
    final = base_state()
    final["note"] = "anything"
    assert verify(final, base_state()) is True


# --- counts matter in approvals and grants -----------------------------------------------------------------

def test_grant_written_twice_fails():
    # grants: final {(p1,r3) x2}, expected {(p1,r3) x1}: counts differ -> False (repeated grant fails).
    final = grant_state()
    final["grants"] = [["p1", "r3"], ["p1", "r3"]]
    assert verify(final, grant_state()) is False


def test_grant_expected_twice_but_written_once_fails():
    # The same comparison seen from the other side: {x1} vs {x2} -> False.
    expected = grant_state()
    expected["grants"] = [["p1", "r3"], ["p1", "r3"]]
    assert verify(grant_state(), expected) is False


def test_approval_written_twice_fails():
    # approvals: {(p1,r2,m1) x2} vs {x1} -> False.
    final = base_state()
    final["approvals"] = [["p1", "r2", "m1"], ["p1", "r2", "m1"]]
    assert verify(final, base_state()) is False


def test_same_length_different_multiset_fails():
    # grants {A, A} vs {A, B}: same length, but B has no partner -> False. (A set comparison would also
    # fail here; a "same length and every left record is somewhere on the right" check would wrongly pass.)
    final = grant_state()
    expected = grant_state()
    final["grants"] = [["p1", "r3"], ["p1", "r3"]]
    expected["grants"] = [["p1", "r3"], ["p1", "r4"]]
    assert verify(final, expected) is False
    assert verify(expected, final) is False


def test_needless_grant_on_held_role_fails():
    # An "already" family where the policy granted the held role anyway: access unchanged (pair already there),
    # but grants has one extra record -> False.
    expected = {"access": [["p1", "r1"]], "approvals": [], "grants": [],
                "request": {"resolution": "already", "person_id": "p1", "role_id": "r1"}}
    final = copy.deepcopy(expected)
    final["grants"] = [["p1", "r1"]]
    assert verify(final, expected) is False


# --- any changed, extra or missing record ------------------------------------------------------------------

@pytest.mark.parametrize("part, value", [
    ("access", [["p1", "r1"]]),                                    # missing pair (m1, r2)
    ("access", [["p1", "r1"], ["m1", "r2"], ["p1", "r2"]]),        # extra pair: grant without approval
    ("access", [["p1", "r1"], ["m1", "r9"]]),                      # changed role id
    ("approvals", []),                                             # missing approval
    ("approvals", [["p1", "r2", "m2"]]),                           # wrong approver
    ("approvals", [["p9", "r2", "m1"]]),                           # wrong person
    ("approvals", [["p1", "r2"]]),                                 # record too short
    ("grants", [["p1", "r2"]]),                                    # extra grant
])
def test_any_record_difference_fails(part, value):
    # Exactly one part differs from base_state -> that part does not match -> False.
    final = base_state()
    final[part] = value
    assert verify(final, base_state()) is False
    assert verify(base_state(), final) is False


@pytest.mark.parametrize("field, value", [
    ("resolution", "granted"),     # wrong resolution
    ("resolution", None),          # never closed
    ("person_id", "m1"),           # wrong person (the manager)
    ("person_id", None),
    ("role_id", "r1"),             # wrong role
    ("role_id", "R2"),             # case matters: no normalisation
    ("role_id", " r2"),            # white space matters
])
def test_any_request_difference_fails(field, value):
    # One request field differs -> part 4 does not match -> False.
    final = base_state()
    final["request"][field] = value
    assert verify(final, base_state()) is False


def test_never_closed_episode_fails():
    # A world untouched and never closed: request all None vs expected "pending" -> False.
    final = base_state()
    final["approvals"] = []
    final["request"] = {"resolution": None, "person_id": None, "role_id": None}
    assert verify(final, base_state()) is False


def test_extra_request_field_is_ignored():
    # Only resolution, person_id and role_id are compared (my choice) -> True.
    final = base_state()
    final["request"]["note"] = "x"
    assert verify(final, base_state()) is True


def test_duplicated_access_pair_fails():
    # Contract gap: a valid state never holds a pair twice. I compare access as a multiset, so
    # {(p1,r1) x2, (m1,r2)} vs {(p1,r1), (m1,r2)} -> False.
    final = base_state()
    final["access"] = [["p1", "r1"], ["p1", "r1"], ["m1", "r2"]]
    assert verify(final, base_state()) is False


# --- missing parts and fields raise KeyError ---------------------------------------------------------------

@pytest.mark.parametrize("part", ["access", "approvals", "grants", "request"])
@pytest.mark.parametrize("side", ["final", "expected"])
def test_missing_part_raises_key_error(part, side):
    # Everything else agrees; a missing part is not an empty one -> KeyError.
    final, expected = base_state(), base_state()
    del (final if side == "final" else expected)[part]
    with pytest.raises(KeyError):
        verify(final, expected)


def test_missing_empty_part_is_not_taken_for_empty():
    # grants is [] on one side and absent on the other: still KeyError, not True.
    final, expected = grant_state(), grant_state()
    final["approvals"] = []
    del expected["approvals"]
    with pytest.raises(KeyError):
        verify(final, expected)


@pytest.mark.parametrize("field", ["resolution", "person_id", "role_id"])
@pytest.mark.parametrize("side", ["final", "expected"])
def test_missing_request_field_raises_key_error(field, side):
    # A missing request field is not None -> KeyError.
    final, expected = base_state(), base_state()
    del (final if side == "final" else expected)["request"][field]
    with pytest.raises(KeyError):
        verify(final, expected)


def test_missing_none_field_is_not_taken_for_none():
    # final has role_id None, expected lacks role_id: KeyError, not a comparison of None with None.
    final, expected = base_state(), base_state()
    final["request"]["role_id"] = None
    expected["request"]["role_id"] = None
    del expected["request"]["role_id"]
    with pytest.raises(KeyError):
        verify(final, expected)


def test_missing_part_raises_even_if_other_parts_differ():
    # Contract gap: my choice is KeyError whenever a part is missing, also when another part differs.
    final, expected = base_state(), base_state()
    final["grants"] = [["p1", "r2"]]
    del expected["access"]
    with pytest.raises(KeyError):
        verify(final, expected)


# --- wrong types raise TypeError (my choice) -------------------------------------------------------------

def test_record_that_is_a_string_raises_type_error():
    # "p1r1" is neither a list nor a tuple -> TypeError (not split into characters).
    final = grant_state()
    final["grants"] = ["p1r3"]
    with pytest.raises(TypeError):
        verify(final, grant_state())


def test_part_that_is_none_raises_type_error():
    final = grant_state()
    final["approvals"] = None
    with pytest.raises(TypeError):
        verify(final, grant_state())


# --- result type and no mutation ---------------------------------------------------------------------------

def test_returns_bool_not_number():
    result_true = verify(base_state(), base_state())
    result_false = verify(base_state(), grant_state())
    assert type(result_true) is bool and result_true is True
    assert type(result_false) is bool and result_false is False


def test_arguments_are_not_changed():
    final = base_state()
    final["approvals"] = [("p2", "r2", "m1"), ["p1", "r2", "m1"]]
    expected = base_state()
    expected["approvals"] = [["p1", "r2", "m1"], ["p2", "r2", "m1"]]
    final_before, expected_before = copy.deepcopy(final), copy.deepcopy(expected)
    verify(final, expected)
    verify(final, base_state())
    assert final == final_before and expected == expected_before
    # tuples stay tuples (deep equality treats ("a",) != ["a"], so the line above already checks this)
    assert isinstance(final["approvals"][0], tuple)


def test_symmetric_on_hand_cases():
    # Swapping the two arguments gives the same answer on these cases.
    a, b = base_state(), grant_state()
    assert verify(a, b) == verify(b, a) is False
    assert verify(a, base_state()) == verify(base_state(), a) is True
