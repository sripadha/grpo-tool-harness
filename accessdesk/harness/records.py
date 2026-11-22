"""Episode records on disk (one JSON object per line), and the replay of a saved episode.

Every number of the write-up is computed from these files. `replay` is the check that a saved episode can be
trusted: its saved calls, run again on a fresh world, must give the saved tool results and the saved final
state.
"""
from __future__ import annotations

import json
from pathlib import Path

from accessdesk.config import RunConfig
from accessdesk.env.tools import execute
from accessdesk.env.world import World
from accessdesk.text import TextBundle


def write_records(path: str | Path, records: list[dict]) -> None:
    """One JSON object per line, keys in record order, UTF-8, "\\n" line ends. Replaces an existing file."""
    lines = [json.dumps(record, sort_keys=False, ensure_ascii=False) + "\n" for record in records]
    Path(path).write_bytes("".join(lines).encode("utf-8"))


def read_records(path: str | Path) -> list[dict]:
    """The records of a file written by write_records, in file order."""
    lines = Path(path).read_bytes().decode("utf-8").split("\n")
    return [json.loads(line) for line in lines if line]


def replay(record: dict, family: dict, text: TextBundle, cfg: RunConfig) -> dict:
    """Run the saved calls of an episode record again on a fresh world; return the final state dict.

    Only the saved parsed calls are used (`tool_key` and `args` of every turn with `parse_ok`), in order. The
    assistant texts are not parsed again, and the saved final state is not read.

    Raises ValueError if the record cannot be an episode of this family, text and config: another family id,
    more turns than cfg.turn_limit, a call after the request was closed, or a tool result or tool error that
    differs from the saved one.
    """
    if record["family_id"] != family["family_id"]:
        raise ValueError(f"the record is of family {record['family_id']}, not of {family['family_id']}")
    if len(record["turns"]) > cfg.turn_limit:
        raise ValueError(f"{len(record['turns'])} turns, but the turn limit is {cfg.turn_limit}")
    world = World(family["world"])
    for turn in record["turns"]:
        if world.closed:
            raise ValueError(f"turn {turn['turn']} comes after the request was closed")
        if not turn["parse_ok"]:
            continue
        outcome = execute(world, text, turn["tool_key"], turn["args"])
        if (outcome.response, outcome.error) != (turn["tool_response"], turn["tool_error"]):
            raise ValueError(f"turn {turn['turn']}: the tool result of the replay differs from the saved one")
    return world.state()
