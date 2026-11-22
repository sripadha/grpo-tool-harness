"""The rules of the desk, and the correct end state of a family under each rule set.

Nothing here runs a tool. `expected_state` is built directly from the family's world and truth, so that the
scripted correct policy (which does run the tools) is an independent check of it.
"""
from __future__ import annotations

RULESETS = ("default", "all_grants_need_approval", "sensitive_denied")

ACTIONS = ("deny", "already", "approval", "grant")

# The resolution (internal key) that closes the request after each action.
RESOLUTION_OF_ACTION = {"deny": "denied", "already": "already", "approval": "pending", "grant": "granted"}


def decide(active: bool, has_access: bool, sensitive: bool, ruleset: str = "default") -> str:
    """The correct action: "deny", "already", "approval" or "grant". The first rule that applies wins.

    default:                  not active -> deny; has access -> already; sensitive -> approval; else grant.
    all_grants_need_approval: not active -> deny; has access -> already; else approval.
    sensitive_denied:         not active -> deny; has access -> already; sensitive -> deny; else grant.
    """
    if ruleset not in RULESETS:
        raise ValueError(f"unknown rule set {ruleset!r}; known: {RULESETS}")
    if not active:
        return "deny"
    if has_access:
        return "already"
    if ruleset == "all_grants_need_approval":
        return "approval"
    if sensitive:
        return "deny" if ruleset == "sensitive_denied" else "approval"
    return "grant"


def target_facts(family: dict) -> tuple[bool, bool, bool]:
    """(active, has_access, sensitive) of the request's person and role, read from the family's world."""
    world, truth = family["world"], family["truth"]
    person = _only([p for p in world["people"] if p["person_id"] == truth["person_id"]], "person")
    role = _only([r for r in world["roles"] if r["role_id"] == truth["role_id"]], "role")
    has_access = [truth["person_id"], truth["role_id"]] in [list(pair) for pair in world["access"]]
    return person["status"] == "active", has_access, bool(role["sensitive"])


def _only(items: list[dict], what: str) -> dict:
    if len(items) != 1:
        raise ValueError(f"the world must hold the request's {what} exactly once, not {len(items)} times")
    return items[0]


def action_for(family: dict, ruleset: str = "default") -> str:
    """The correct action for a family under a rule set."""
    active, has_access, sensitive = target_facts(family)
    return decide(active, has_access, sensitive, ruleset)


def reference_plan(family: dict, ruleset: str = "default") -> list[tuple[str, dict[str, str]]]:
    """The shortest correct list of (tool key, arguments).

    Arguments use internal parameter keys, and the resolution is the internal resolution key.
    The plan is: lookup_person, check_access, then request_approval or grant_role if the action needs a write,
    then close_request.
    """
    request, truth = family["request"], family["truth"]
    ids = {"person_id": truth["person_id"], "role_id": truth["role_id"]}
    action = action_for(family, ruleset)
    plan: list[tuple[str, dict[str, str]]] = [
        ("lookup_person", {"name": request["person_name"]}),
        ("check_access", {"person_id": truth["person_id"], "role_name": request["role_name"]}),
    ]
    if action == "approval":
        plan.append(("request_approval", {**ids, "approver_id": truth["manager_id"]}))
    if action == "grant":
        plan.append(("grant_role", dict(ids)))
    plan.append(("close_request", {"resolution": RESOLUTION_OF_ACTION[action], **ids}))
    return plan


def expected_state(family: dict, ruleset: str = "default") -> dict:
    """The correct final state dict (section 4 of the build specification) of a family under a rule set.

    It is the initial state plus the end state of the correct action:
    deny, already: no write. approval: one approval record (person, role, the person's manager).
    grant: the pair added to access, and one grant record.
    The request always holds the resolution, the person's id and the role's id.
    """
    truth = family["truth"]
    pair = [truth["person_id"], truth["role_id"]]
    action = action_for(family, ruleset)
    access = [list(p) for p in family["world"]["access"]]
    approvals: list[list[str]] = []
    grants: list[list[str]] = []
    if action == "approval":
        approvals.append(pair + [truth["manager_id"]])
    if action == "grant":            # "grant" is decided only when the pair is not in access yet
        access.append(list(pair))
        grants.append(list(pair))
    return {
        "access": sorted(access),
        "approvals": approvals,
        "grants": grants,
        "request": {"resolution": RESOLUTION_OF_ACTION[action],
                    "person_id": truth["person_id"], "role_id": truth["role_id"]},
    }


def turn_budget(family: dict, ruleset: str = "default") -> int:
    """The number of turns allowed without a turn penalty: the shortest correct path plus one turn."""
    return len(reference_plan(family, ruleset)) + 1
