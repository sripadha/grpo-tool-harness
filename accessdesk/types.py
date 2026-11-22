"""Small shared data types. No logic that needs a test lives here.

Internal keys and model-visible strings are kept apart on purpose:
- an internal *key* (for example "lookup_person", "person_id", "granted") is used by code;
- a model-visible *name* or *value* comes from the text file and is what the model reads and writes.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Internal keys (never shown to the model)
# ---------------------------------------------------------------------------
TOOL_KEYS = ("lookup_person", "check_access", "request_approval", "grant_role", "close_request")
EXTRA_TOOL_KEY = "department_info"  # read-only tool used only in the extra-tool set

OUTCOMES = ("grant", "approval", "already", "inactive")          # one per family
RESOLUTIONS = ("granted", "pending", "already", "denied")        # internal resolution keys
STATUSES = ("active", "inactive")

# Why an assistant message could not be used as a tool call.
PARSE_ERRORS = (
    "no_call",             # no <tool_call> block
    "multiple_calls",      # more than one <tool_call> block
    "bad_format",          # the block does not follow the format
    "unknown_tool",        # function name is not in the tool list
    "missing_argument",    # a required parameter is absent
    "unknown_argument",    # a parameter that the tool does not have
    "duplicate_argument",  # the same parameter twice
    "empty_argument",      # a parameter with an empty value
    "bad_value",           # a value outside the allowed list (close_request resolution)
    "text_after_call",     # anything but white space after </tool_call>
    "truncated",           # the turn was cut off by the token cap
)

SEGMENT_KINDS = ("prompt", "sampled", "suffix", "forced_eos")


@dataclass(frozen=True)
class ParamSpec:
    key: str                              # internal key
    name: str                             # model-visible parameter name
    description: str                      # model-visible
    enum: tuple[str, ...] | None = None   # model-visible allowed values, or None


@dataclass(frozen=True)
class ToolSpec:
    key: str                              # internal key
    name: str                             # model-visible tool name
    description: str                      # model-visible
    params: tuple[ParamSpec, ...]         # every parameter is required and is a string

    def schema(self) -> dict:
        """The function schema that the chat template writes into the system turn."""
        properties: dict[str, dict] = {}
        for p in self.params:
            prop: dict = {"type": "string", "description": p.description}
            if p.enum is not None:
                prop["enum"] = list(p.enum)
            properties[p.name] = prop
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [p.name for p in self.params],
                },
            },
        }

    def param_by_name(self, name: str) -> ParamSpec | None:
        for p in self.params:
            if p.name == name:
                return p
        return None


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    tool_key: str | None = None           # internal tool key when ok
    args: dict[str, str] | None = None    # internal parameter key -> value (model-visible string, stripped)
    error: str | None = None              # one of PARSE_ERRORS when not ok
    preamble: str = ""                    # text before the call (allowed by the chat template)


@dataclass(frozen=True)
class Segment:
    """A run of tokens in an episode's token buffer: [start, end)."""
    kind: str    # one of SEGMENT_KINDS
    start: int
    end: int
    turn: int    # assistant turn index (0-based); -1 for the first prompt
