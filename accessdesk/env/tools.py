"""The six tools of the desk. Each one reads or writes a World and returns the JSON text that the model reads.

No model-visible string is written here: result keys, status words, confirmations and error texts come from
the text bundle; ids, names and departments come from the world.

An error writes nothing. Arguments are checked in the tool's parameter order, and the first problem is
reported. An error text may repeat the value that was not found. That value is text that the model wrote,
so `TextBundle.error_text` passes it through `accessdesk.text.safe_echo` first: it is cut to 40 characters,
and every character that is not a letter, a digit, a space or one of  _ - . ' ,  becomes "?". So a value
that spells a chat-format string (for example the end-of-turn token) cannot put that string into a tool
result. No other part of a tool result repeats an argument: everything else comes from the world or from
the text bundle.

The tools allow wrong but well-formed actions (a wrong approver, a grant on a sensitive role, a
grant to an inactive person, a wrong resolution): the final state then shows them, and the verifier decides.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from accessdesk.env.world import World
from accessdesk.text import TOOL_PARAMS, TextBundle, normalise
from accessdesk.types import RESOLUTIONS


@dataclass(frozen=True)
class ToolOutcome:
    response: str            # the JSON text that the model reads
    error: str | None        # internal error key, or None
    closes: bool             # True only for a successful close_request


class _ToolError(Exception):
    """A well-formed call that the world cannot serve: the internal error key and the value not found."""

    def __init__(self, key: str, value: str) -> None:
        super().__init__(key)
        self.key = key
        self.value = value


def execute(world: World, text: TextBundle, tool_key: str, args: dict[str, str]) -> ToolOutcome:
    """Run one tool call. `args` maps every internal parameter key of the tool to a string.

    For close_request the resolution is the internal resolution key (the caller has mapped the visible value).
    Raises ValueError for a call that the parser can never produce (unknown tool key, wrong parameter set,
    unknown resolution key): that is a fault of the harness and must not look like a tool error.
    """
    if tool_key not in _TOOLS:
        raise ValueError(f"unknown tool key {tool_key!r}")
    if sorted(args) != sorted(TOOL_PARAMS[tool_key]):
        raise ValueError(f"{tool_key} needs the arguments {TOOL_PARAMS[tool_key]}, not {sorted(args)}")
    try:
        result = _TOOLS[tool_key](world, text, args)
    except _ToolError as exc:
        body = {text.result_keys["error"]: text.error_text(exc.key, exc.value)}
        return ToolOutcome(response=_to_json(body), error=exc.key, closes=False)
    return ToolOutcome(response=_to_json(result), error=None, closes=tool_key == "close_request")


def _to_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Argument checks. Each returns the id that is known to the world, or raises _ToolError.
# ---------------------------------------------------------------------------
def _person_id(world: World, value: str, error_key: str = "unknown_person_id") -> str:
    if value not in world.people:
        raise _ToolError(error_key, value)
    return value


def _role_id(world: World, value: str) -> str:
    if value not in world.roles:
        raise _ToolError("unknown_role_id", value)
    return value


def _role_id_by_name(world: World, name: str) -> str:
    for role_id, role in world.roles.items():
        if normalise(role["name"]) == normalise(name):
            return role_id
    raise _ToolError("unknown_role_name", name)


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------
def _lookup_person(world: World, text: TextBundle, args: dict[str, str]) -> dict:
    keys = text.result_keys
    wanted = normalise(args["name"])
    matches = [
        {keys["person_id"]: person["person_id"], keys["name"]: person["name"],
         keys["department"]: person["department"], keys["status"]: text.status[person["status"]],
         keys["manager_id"]: person["manager_id"]}
        for person in world.people.values() if normalise(person["name"]) == wanted
    ]
    return {keys["matches"]: matches}


def _check_access(world: World, text: TextBundle, args: dict[str, str]) -> dict:
    keys = text.result_keys
    person_id = _person_id(world, args["person_id"])
    role_id = _role_id_by_name(world, args["role_name"])
    return {keys["role_id"]: role_id, keys["sensitive"]: bool(world.roles[role_id]["sensitive"]),
            keys["has_access"]: (person_id, role_id) in world.access}


def _request_approval(world: World, text: TextBundle, args: dict[str, str]) -> dict:
    person_id = _person_id(world, args["person_id"])
    role_id = _role_id(world, args["role_id"])
    approver_id = _person_id(world, args["approver_id"], "unknown_approver_id")
    world.approvals.append((person_id, role_id, approver_id))
    return {text.result_keys["result"]: text.confirmation["approval_requested"]}


def _grant_role(world: World, text: TextBundle, args: dict[str, str]) -> dict:
    person_id = _person_id(world, args["person_id"])
    role_id = _role_id(world, args["role_id"])
    world.access.add((person_id, role_id))
    world.grants.append((person_id, role_id))      # also when the pair was already in access
    return {text.result_keys["result"]: text.confirmation["role_granted"]}


def _close_request(world: World, text: TextBundle, args: dict[str, str]) -> dict:
    if args["resolution"] not in RESOLUTIONS:
        raise ValueError(f"resolution must be an internal key {RESOLUTIONS}, not {args['resolution']!r}")
    person_id = _person_id(world, args["person_id"])
    role_id = _role_id(world, args["role_id"])
    world.request = {"resolution": args["resolution"], "person_id": person_id, "role_id": role_id}
    world.closed = True
    return {text.result_keys["result"]: text.confirmation["request_closed"]}


def _department_info(world: World, text: TextBundle, args: dict[str, str]) -> dict:
    wanted = normalise(args["department"])
    members = [p for p in world.people.values() if normalise(p["department"]) == wanted]
    if not members:
        raise _ToolError("unknown_department", args["department"])
    return {text.result_keys["department"]: members[0]["department"],
            text.result_keys["member_count"]: len(members)}


_TOOLS = {
    "lookup_person": _lookup_person,
    "check_access": _check_access,
    "request_approval": _request_approval,
    "grant_role": _grant_role,
    "close_request": _close_request,
    "department_info": _department_info,
}
