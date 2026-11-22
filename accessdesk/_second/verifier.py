"""Second version of `verify`, written from the contract in the function's docstring only.

Another model wrote it without reading the main version, so that a test can compare the two on generated
inputs. Used by tests only.
"""
from __future__ import annotations

from collections.abc import Mapping


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
    # Choices where the contract was silent when this version was written. The comparison of the two
    # versions (2026-11-17) put the first two into the contract; the others are outside it (other types).
    # - a missing part or request field in EITHER argument raises KeyError, before anything is compared;
    # - keys other than the four parts and the three request fields are ignored;
    # - access is compared as a multiset too (a duplicated pair, which a valid state never has, fails);
    # - a part or a record that is not a list or a tuple raises TypeError;
    # - values are compared with plain ==, with no stripping or case folding.
    # - an argument or a request that is not a mapping raises TypeError.
    if not isinstance(final_state, Mapping) or not isinstance(expected_state, Mapping):
        raise TypeError("both states must be dicts")
    for name in _PARTS:
        if name not in final_state or name not in expected_state:
            raise KeyError(name)
    if not isinstance(final_state["request"], Mapping) or not isinstance(expected_state["request"], Mapping):
        raise TypeError("the request part must be a dict")
    for field in _REQUEST_FIELDS:
        if field not in final_state["request"] or field not in expected_state["request"]:
            raise KeyError(field)

    same = True
    for name in ("access", "approvals", "grants"):
        if not _same_multiset(final_state[name], expected_state[name]):
            same = False
    for field in _REQUEST_FIELDS:
        if not (final_state["request"][field] == expected_state["request"][field]):
            same = False
    return bool(same)


_PARTS = ("access", "approvals", "grants", "request")
_REQUEST_FIELDS = ("resolution", "person_id", "role_id")


def _records(part: object) -> list[tuple]:
    """A new list of the part's records, each as a tuple. TypeError for a part or record of another type."""
    if not isinstance(part, (list, tuple)):
        raise TypeError(f"a state part must be a list or tuple, got {type(part).__name__}")
    out = []
    for record in part:
        if not isinstance(record, (list, tuple)):
            raise TypeError(f"a record must be a list or tuple, got {type(record).__name__}")
        out.append(tuple(record))
    return out


def _same_multiset(left: object, right: object) -> bool:
    """True if every record of `left` can be paired with one equal record of `right`, with none left over.

    Done by crossing records off a copy of the right list (no hashing, no sorting): the inputs stay as they
    are.
    """
    left_records = _records(left)
    remaining = _records(right)
    if len(left_records) != len(remaining):
        return False
    for record in left_records:
        for i, candidate in enumerate(remaining):
            if candidate == record:
                del remaining[i]
                break
        else:
            return False
    return not remaining
