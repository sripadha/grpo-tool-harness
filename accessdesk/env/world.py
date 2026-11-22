"""The state of one episode, held in memory only."""
from __future__ import annotations


class World:
    """Mutable state of one episode. Built fresh from a family's world dict for every episode.

    The world dict is not changed and nothing of it is shared: every episode starts from the same state.
    """

    def __init__(self, world: dict) -> None:
        self.people: dict[str, dict] = {p["person_id"]: dict(p) for p in world["people"]}
        self.roles: dict[str, dict] = {r["role_id"]: dict(r) for r in world["roles"]}
        if len(self.people) != len(world["people"]) or len(self.roles) != len(world["roles"]):
            raise ValueError("person ids and role ids must be unique in a world")
        self.access: set[tuple[str, str]] = {(person_id, role_id) for person_id, role_id in world["access"]}
        self.approvals: list[tuple[str, str, str]] = []   # (person_id, role_id, approver_id), in call order
        self.grants: list[tuple[str, str]] = []           # (person_id, role_id), in call order
        self.request: dict[str, str | None] = {"resolution": None, "person_id": None, "role_id": None}
        self.closed: bool = False

    def state(self) -> dict:
        """The state dict: a fresh, JSON-serialisable copy with every list sorted.

        {"access": [[person_id, role_id], ...], "approvals": [[person_id, role_id, approver_id], ...],
         "grants": [[person_id, role_id], ...],
         "request": {"resolution": .., "person_id": .., "role_id": ..}}
        Equal states give equal JSON. A record that was written twice appears twice.
        """
        return {
            "access": sorted(list(pair) for pair in self.access),
            "approvals": sorted(list(record) for record in self.approvals),
            "grants": sorted(list(record) for record in self.grants),
            "request": dict(self.request),
        }
