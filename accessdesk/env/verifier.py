"""The verifier: does the final state of an episode match the correct end state?

This is the main version of `verify`. The rest of the code reaches it through `accessdesk.impl.get("verify")`.
An independent second version, written from the docstring below only, is in `accessdesk/_second/verifier.py`;
tests compare the two.
Tests: `.venv/bin/python -m pytest tests/core/test_verify.py tests/core/test_differential_verify.py`
"""
from __future__ import annotations


def verify(final_state: dict, expected_state: dict) -> bool:
    """True only if all four parts of the final state match the expected state.

    Both arguments are state dicts (the shape that `World.state()` and `rules.expected_state()` return):

        {"access":    [[person_id, role_id], ...],                 who holds which role
         "approvals": [[person_id, role_id, approver_id], ...],    one record per approval request
         "grants":    [[person_id, role_id], ...],                 one record per grant call
         "request":   {"resolution": "granted" | "pending" | "already" | "denied" | None,
                       "person_id": str | None, "role_id": str | None}}

    The four parts:
      1. access:    the same pairs (a state never holds a pair twice; order does not matter).
      2. approvals: the same records, compared as multisets: order does not matter, count does.
      3. grants:    the same records, compared as multisets: order does not matter, count does.
      4. request:   the same resolution, the same person_id and the same role_id.

    Consequences that the tests check:
      - A record written twice does not match a record expected once (a repeated or needless grant fails).
      - An episode that never closed has resolution None, and no expected state has that, so it fails.
      - Any extra record, missing record or changed id in any part gives False.

    A record may be a list or a tuple: ["p1", "r1"] and ("p1", "r1") are the same record.
    A missing part or a missing request field is never taken for an empty one: the call raises KeyError.
    This is checked for both arguments before anything is compared, so it does not matter what else the two
    states hold. Keys beyond the four parts and the three request fields are not read.
    The ids are strings (None only in the request), the parts are lists, and a record is a list or a tuple.
    Arguments of other types are outside this contract: what the call does then is not defined.
    The function reads its arguments and changes neither of them. It returns a bool, never a number.
    It does not know how the state was reached: turns and malformed calls belong to the reward.
    """
    # First: nothing may be missing, in either state. A state without one of its parts is a mistake of the
    # caller, and it must not pass for a fail (or for a pass) because the comparison stopped before it.
    for state in (final_state, expected_state):
        for part in ("access", "approvals", "grants", "request"):
            if part not in state:
                raise KeyError(part)
        for field in ("resolution", "person_id", "role_id"):
            if field not in state["request"]:
                raise KeyError(field)

    # Parts 1 to 3: the three lists of records. Two lists hold the same records the same number of times
    # exactly when they are equal after sorting. `sorted` makes a new list, so the arguments stay as they are.
    # Each record becomes a tuple first, so that a record stored as a list and one stored as a tuple compare.
    for part in ("access", "approvals", "grants"):
        final_records = sorted(tuple(record) for record in final_state[part])
        expected_records = sorted(tuple(record) for record in expected_state[part])
        if final_records != expected_records:
            return False

    # Part 4: the three fields of the request, one by one. An episode that never closed has None in all
    # three, and no expected state has a resolution of None, so it fails here.
    for field in ("resolution", "person_id", "role_id"):
        if final_state["request"][field] != expected_state["request"][field]:
            return False

    # Nothing differs.
    return True
