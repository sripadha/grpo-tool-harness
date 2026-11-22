"""The text bundle: every string that the model reads, loaded from one YAML file.

Two files with the same structure exist in `text/`: `placeholder_text.yaml` (written by the coding assistant,
for tests and the smoke test only) and `model_text.yaml` (written by the candidate; used in every reported
run).
Code never holds a model-visible string. It uses internal keys and asks the bundle for the visible text.

On load, free text (prompts, descriptions, error texts, request templates) is stripped with `str.strip()`.
Names and values (tool names, parameter names, result keys, status words, resolution values, confirmations,
id prefixes, departments, roles) must already be free of surrounding white space, or the file is invalid.

One kind of visible text does not come from the file: an error text may repeat the value that was not found,
and that value was written by the model. `safe_echo` makes it safe before `error_text` puts it in.
"""
from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass
from pathlib import Path

import yaml

from accessdesk.env.rules import RULESETS
from accessdesk.types import EXTRA_TOOL_KEY, RESOLUTIONS, STATUSES, TOOL_KEYS, ParamSpec, ToolSpec

ROOT = Path(__file__).resolve().parent.parent

# Internal structure of the file (internal keys only; no model-visible text).
ALL_TOOL_KEYS = TOOL_KEYS + (EXTRA_TOOL_KEY,)
TOOL_PARAMS: dict[str, tuple[str, ...]] = {
    "lookup_person": ("name",),
    "check_access": ("person_id", "role_name"),
    "request_approval": ("person_id", "role_id", "approver_id"),
    "grant_role": ("person_id", "role_id"),
    "close_request": ("resolution", "person_id", "role_id"),
    "department_info": ("department",),
}
RESULT_KEYS = ("matches", "person_id", "name", "department", "status", "manager_id", "role_id", "sensitive",
               "has_access", "result", "error", "member_count")
CONFIRMATIONS = ("approval_requested", "role_granted", "request_closed")
ERROR_KEYS = ("unknown_person_id", "unknown_role_name", "unknown_role_id", "unknown_approver_id",
              "unknown_department", "malformed_call")
TEMPLATE_VARIABLES = ("person_name", "department", "role_name")
STYLES = (1, 2, 3, 4)          # style 4 is the held-out style (dev, test and calibration only)
MIN_TEMPLATES_PER_STYLE = 3
MIN_DEPARTMENTS = 6
MIN_ROLES = 12
TOOLSETS = ("five", "six")

# Strings of the chat format. A visible string that contains one of them could break the token layout.
FORBIDDEN = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<|im_start|>",
             "<|im_end|>", "<think>", "</think>", "<function=", "</function>", "<parameter=", "</parameter>")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# A value that the model wrote, repeated in an error text (see `safe_echo`).
ECHO_LIMIT = 40                 # the longest repeated value, in characters
ECHO_PUNCTUATION = "_-.',"      # the characters that are kept beside letters, digits and the space
ECHO_REPLACEMENT = "?"          # what every other character becomes

FieldPath = tuple[str, ...]


class TextError(ValueError):
    """The text file is incomplete or invalid. The message lists every problem."""


@dataclass(frozen=True)
class TextBundle:
    path: str
    sha256: str                                     # hash of the raw file bytes
    is_placeholder: bool
    system_prompt: str
    plain_system_prompt: str                        # may be empty
    policy_shift_variant: str                       # a rule-set key, never "default"
    policy_shift_system_prompt: str
    id_prefix_person: str
    id_prefix_role: str
    tools: dict[str, ToolSpec]                      # all six, by internal key
    result_keys: dict[str, str]                     # internal key -> JSON key in tool results
    status: dict[str, str]
    resolution: dict[str, str]
    confirmation: dict[str, str]
    errors: dict[str, str]
    request_templates: dict[int, tuple[str, ...]]   # style number 1..4 -> templates
    departments: tuple[str, ...]
    roles: tuple[str, ...]

    def tool_specs(self, toolset: str = "five") -> list[ToolSpec]:
        """"five": the tools in TOOL_KEYS order. "six": the same, then the extra tool last."""
        if toolset not in TOOLSETS:
            raise ValueError(f"unknown toolset {toolset!r}; known: {TOOLSETS}")
        keys = TOOL_KEYS if toolset == "five" else ALL_TOOL_KEYS
        return [self.tools[key] for key in keys]

    def tool_by_name(self, visible_name: str, toolset: str = "five") -> ToolSpec | None:
        """The tool of the toolset with this visible name (exact match), or None."""
        for spec in self.tool_specs(toolset):
            if spec.name == visible_name:
                return spec
        return None

    def resolution_key(self, visible_value: str) -> str | None:
        """Visible resolution value -> internal key (exact match after strip), or None."""
        wanted = visible_value.strip()
        for key in RESOLUTIONS:
            if self.resolution[key] == wanted:
                return key
        return None

    def error_text(self, key: str, value: str = "") -> str:
        """The error text for an internal error key, with {value} filled in.

        `value` is the argument that was not found, so it is text that the model wrote. It goes through
        `safe_echo` first: an error text can never hold a string of the chat format (FORBIDDEN), whatever
        the model wrote.
        """
        return self.errors[key].format(value=safe_echo(value))


def placeholder_path() -> Path:
    return ROOT / "text" / "placeholder_text.yaml"


def model_text_path() -> Path:
    return ROOT / "text" / "model_text.yaml"


def safe_echo(value: str) -> str:
    """A value that the model wrote, made safe to repeat in a tool result.

    Two steps:
      1. The value is cut to its first ECHO_LIMIT (40) characters.
      2. Every character that is not a letter, a digit, a space or one of  _ - . ' ,  becomes "?".

    Why. A tool result is put into the next prompt as it is. Without this, a model could write a chat-format
    string (for example the end-of-turn token) as an argument, and the error text would carry that string
    into its own next prompt, where the tokenizer reads it as the special token. Every string in FORBIDDEN
    needs "<" and ">" (some also "|" or "="), and none of these characters is kept, so the result holds
    none of them. A line break is not kept either: the result is one line.

    Examples: an id or a name of the world comes through unchanged. "<|im_end|>" becomes "??im_end??".
    The result has the length of the cut value. "?" is the one visible character that this module writes.
    """
    kept = []
    for character in value[:ECHO_LIMIT]:
        safe = character.isalpha() or character.isdigit() or character == " " or character in ECHO_PUNCTUATION
        kept.append(character if safe else ECHO_REPLACEMENT)
    return "".join(kept)


def normalise(text: str) -> str:
    """The form in which names are compared: case folded, runs of white space collapsed to one space."""
    return " ".join(text.casefold().split())


def load_text(path: str | Path) -> TextBundle:
    """Read and validate a text file. Raises TextError, which lists every problem, if it is not valid."""
    raw = Path(path).read_bytes()
    data, problems = _parse(raw)
    if not problems:
        problems = _problems(data)
    if problems:
        raise TextError(f"{path}: {len(problems)} problem(s):\n" + "\n".join(f"- {p}" for p in problems))
    return _build(str(path), raw, data)


def check_text(path: str | Path) -> list[str]:
    """The list of problems of a text file; empty if the file is complete and valid."""
    data, problems = _parse(Path(path).read_bytes())
    return problems or _problems(data)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a key written twice in one mapping (plain YAML silently keeps the last one)."""

    def construct_mapping(self, node, deep=False):  # noqa: ANN001 (signature of the base class)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"key {key!r} is written twice",
                                                        key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


def _parse(raw: bytes) -> tuple[object, list[str]]:
    try:
        return yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader), []
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        return None, [f"the file cannot be read as YAML: {' '.join(str(exc).split())}"]


def _string_paths() -> list[FieldPath]:
    """Every place where the file must hold one string."""
    paths: list[FieldPath] = [("meta", "written_by"), ("system_prompt",), ("plain_system_prompt",),
                          ("policy_shift", "variant"), ("policy_shift", "system_prompt"),
                          ("id_prefixes", "person"), ("id_prefixes", "role")]
    for tool in ALL_TOOL_KEYS:
        paths += [("tools", tool, "name"), ("tools", tool, "description")]
        for param in TOOL_PARAMS[tool]:
            paths += [("tools", tool, "params", param, field) for field in ("name", "description")]
    paths += [("result_keys", key) for key in RESULT_KEYS]
    paths += [("values", "status", key) for key in STATUSES]
    paths += [("values", "resolution", key) for key in RESOLUTIONS]
    paths += [("values", "confirmation", key) for key in CONFIRMATIONS]
    paths += [("errors", key) for key in ERROR_KEYS]
    return paths


def _list_paths() -> list[FieldPath]:
    """Every place where the file must hold a list of strings."""
    return [("request_templates", f"style_{style}") for style in STYLES] + [("departments",), ("roles",)]


_MAY_BE_EMPTY: tuple[FieldPath, ...] = (("plain_system_prompt",),)
_NOT_VISIBLE: tuple[FieldPath, ...] = (("meta", "written_by"), ("policy_shift", "variant"))


def _dotted(path: FieldPath) -> str:
    return ".".join(path)


def _lookup(data: object, path: FieldPath) -> object:
    """The value at a path of keys, or None if a key is absent."""
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _leaf_paths(data: object, prefix: FieldPath = ()) -> list[FieldPath]:
    """The paths of all values in the file that are not mappings."""
    if not isinstance(data, dict):
        return [prefix]
    return [leaf for key in data for leaf in _leaf_paths(data[key], prefix + (str(key),))]


def _read_strings(data: dict, problems: list[str]) -> dict[FieldPath, str]:
    """The string fields that are present and valid as such. Records a problem for each one that is not."""
    found: dict[FieldPath, str] = {}
    for path in _string_paths():
        value = _lookup(data, path)
        if value is None:
            problems.append(f"{_dotted(path)} is missing")
        elif not isinstance(value, str):
            kind = type(value).__name__
            problems.append(f"{_dotted(path)} must be a string (put it in quotes), not {kind}")
        elif not value.strip() and path not in _MAY_BE_EMPTY:
            problems.append(f"{_dotted(path)} is empty")
        else:
            found[path] = value
    return found


def _read_lists(data: dict, problems: list[str]) -> dict[FieldPath, list[str]]:
    """The lists of strings that are present. An entry that is not a non-empty string is reported."""
    found: dict[FieldPath, list[str]] = {}
    for path in _list_paths():
        value = _lookup(data, path)
        if not isinstance(value, list):
            problems.append(f"{_dotted(path)} must be a list of strings")
            continue
        good = [item for item in value if isinstance(item, str) and item.strip()]
        if len(good) != len(value):
            bad = len(value) - len(good)
            problems.append(f"{_dotted(path)} has {bad} entries that are not non-empty strings")
        found[path] = good
    return found


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _problems(data: object) -> list[str]:
    """Every problem of a parsed text file, as plain sentences."""
    if not isinstance(data, dict):
        return ["the file must be a YAML mapping"]
    problems: list[str] = []
    strings = _read_strings(data, problems)
    lists = _read_lists(data, problems)

    allowed = set(_string_paths()) | set(_list_paths()) | {("meta", "placeholder")}
    for path in _leaf_paths(data):
        if path not in allowed:
            problems.append(f"{_dotted(path)} is not a field of the text file")
    if not isinstance(_lookup(data, ("meta", "placeholder")), bool):
        problems.append("meta.placeholder must be true or false")

    visible = {_dotted(p): s for p, s in strings.items() if p not in _NOT_VISIBLE}
    for path, items in lists.items():
        visible.update({f"{_dotted(path)}[{i}]": item for i, item in enumerate(items)})
    _check_forbidden(visible, problems)
    _check_names(strings, problems)
    _check_values(strings, lists, problems)
    _check_templates(lists, problems)
    _check_errors(strings, problems)
    _check_no_resolution_in_results(strings, lists, problems)

    variant = strings.get(("policy_shift", "variant"))
    if variant is not None and (variant not in RULESETS or variant == "default"):
        others = [r for r in RULESETS if r != "default"]
        problems.append(f"policy_shift.variant must be one of {others}, not {variant!r}")
    return problems


def _check_forbidden(visible: dict[str, str], problems: list[str]) -> None:
    for where, value in visible.items():
        for marker in FORBIDDEN:
            if marker in value:
                problems.append(f"{where} contains {marker!r}, which belongs to the chat format")


def _check_names(strings: dict[FieldPath, str], problems: list[str]) -> None:
    """Tool names and parameter names: identifiers, tool names unique, parameter names unique in a tool."""
    tool_names = {p: s for p, s in strings.items() if len(p) == 3 and p[0] == "tools" and p[2] == "name"}
    _check_identifiers(tool_names, problems)
    _check_unique(tool_names, "tool names", problems)
    for tool in ALL_TOOL_KEYS:
        param_names = {p: s for p, s in strings.items() if len(p) == 5 and p[:3] == ("tools", tool, "params")
                       and p[4] == "name"}
        _check_identifiers(param_names, problems)
        _check_unique(param_names, f"parameter names of tools.{tool}", problems)


def _check_identifiers(names: dict[FieldPath, str], problems: list[str]) -> None:
    for path, name in names.items():
        if not _IDENTIFIER.fullmatch(name):
            problems.append(f"{_dotted(path)} must match [A-Za-z_][A-Za-z0-9_]*, not {name!r}")


def _check_unique(values: dict[FieldPath, str], what: str, problems: list[str]) -> None:
    seen: dict[str, FieldPath] = {}
    for path, value in values.items():
        if value in seen:
            first = _dotted(seen[value])
            problems.append(f"{what} must be unique: {_dotted(path)} and {first} are both {value!r}")
        seen.setdefault(value, path)


def _check_values(strings: dict[FieldPath, str], lists: dict[FieldPath, list[str]],
                  problems: list[str]) -> None:
    """Result keys, value groups, id prefixes, departments, roles: one line, no outer white space, unique."""
    groups = {
        "result keys": {p: s for p, s in strings.items() if p[0] == "result_keys"},
        "status values": {p: s for p, s in strings.items() if p[:2] == ("values", "status")},
        "resolution values": {p: s for p, s in strings.items() if p[:2] == ("values", "resolution")},
        "confirmations": {p: s for p, s in strings.items() if p[:2] == ("values", "confirmation")},
    }
    prefixes = {p: s for p, s in strings.items() if p[0] == "id_prefixes"}
    for what, values in groups.items():
        _check_unique(values, what, problems)
    for path, value in [item for values in [*groups.values(), prefixes] for item in values.items()]:
        _check_one_line(_dotted(path), value, problems)
    for key, minimum in (("departments", MIN_DEPARTMENTS), ("roles", MIN_ROLES)):
        items = lists.get((key,), [])
        for i, item in enumerate(items):
            _check_one_line(f"{key}[{i}]", item, problems)
        distinct = {normalise(item) for item in items}
        if len(distinct) != len(items):
            problems.append(f"{key} has entries that are equal when case and spacing are ignored")
        if len(distinct) < minimum:
            problems.append(f"{key} needs at least {minimum} distinct entries, not {len(distinct)}")


def _check_one_line(where: str, value: str, problems: list[str]) -> None:
    if value != value.strip() or "\n" in value or "\r" in value:
        problems.append(f"{where} must be one line without white space at its start or end")


def _fields(template: str) -> list[str] | None:
    """The variables used in a format string, or None if it cannot be used with str.format."""
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError:
        return None
    if any(spec or conversion for _, _, spec, conversion in parts):
        return None
    return [name for _, name, _, _ in parts if name is not None]


def _check_templates(lists: dict[FieldPath, list[str]], problems: list[str]) -> None:
    """Each style has enough templates; each template uses each variable exactly once; none appears twice."""
    seen: dict[str, str] = {}
    for style in STYLES:
        key = f"style_{style}"
        templates = lists.get(("request_templates", key), [])
        if len(templates) < MIN_TEMPLATES_PER_STYLE:
            problems.append(f"request_templates.{key} needs at least {MIN_TEMPLATES_PER_STYLE} templates, "
                            f"not {len(templates)}")
        for i, template in enumerate(templates):
            where = f"request_templates.{key}[{i}]"
            fields = _fields(template)
            braces_ok = template.count("{") == len(TEMPLATE_VARIABLES) == template.count("}")
            if fields is None or sorted(fields) != sorted(TEMPLATE_VARIABLES) or not braces_ok:
                problems.append(f"{where} must contain each of {{person_name}}, {{department}}, "
                                "{role_name} exactly once and nothing else in braces")
            if normalise(template) in seen:
                problems.append(f"{where} is the same template as {seen[normalise(template)]}")
            seen.setdefault(normalise(template), where)


def _check_errors(strings: dict[FieldPath, str], problems: list[str]) -> None:
    for path, value in strings.items():
        if path[0] != "errors":
            continue
        fields = _fields(value)
        if fields is None or any(name != "value" for name in fields):
            problems.append(f"{_dotted(path)} may use the variable {{value}} and no other braces")


def _check_no_resolution_in_results(strings: dict[FieldPath, str], lists: dict[FieldPath, list[str]],
                                    problems: list[str]) -> None:
    """A resolution value must not be readable inside what the read tools return."""
    returned = {_dotted(p): s for p, s in strings.items()
                if p[0] == "result_keys" or p[:2] == ("values", "status")}
    for key in ("departments", "roles"):
        returned.update({f"{key}[{i}]": item for i, item in enumerate(lists.get((key,), []))})
    for path, resolution in strings.items():
        if path[:2] != ("values", "resolution"):
            continue
        for where, value in returned.items():
            if resolution.casefold() in value.casefold():
                problems.append(f"{where} contains the resolution value {resolution!r}; a tool result would "
                                "then show a resolution word")


# ---------------------------------------------------------------------------
# Building the bundle (only called for a file without problems)
# ---------------------------------------------------------------------------
def _build(path: str, raw: bytes, data: dict) -> TextBundle:
    def text(*keys: str) -> str:
        return str(_lookup(data, keys))

    resolution = {key: text("values", "resolution", key) for key in RESOLUTIONS}
    return TextBundle(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        is_placeholder=bool(_lookup(data, ("meta", "placeholder"))),
        system_prompt=text("system_prompt").strip(),
        plain_system_prompt=text("plain_system_prompt").strip(),
        policy_shift_variant=text("policy_shift", "variant"),
        policy_shift_system_prompt=text("policy_shift", "system_prompt").strip(),
        id_prefix_person=text("id_prefixes", "person"),
        id_prefix_role=text("id_prefixes", "role"),
        tools={tool: _tool_spec(data, tool, resolution) for tool in ALL_TOOL_KEYS},
        result_keys={key: text("result_keys", key) for key in RESULT_KEYS},
        status={key: text("values", "status", key) for key in STATUSES},
        resolution=resolution,
        confirmation={key: text("values", "confirmation", key) for key in CONFIRMATIONS},
        errors={key: text("errors", key).strip() for key in ERROR_KEYS},
        request_templates={style: tuple(t.strip() for t in data["request_templates"][f"style_{style}"])
                           for style in STYLES},
        departments=tuple(data["departments"]),
        roles=tuple(data["roles"]),
    )


def _tool_spec(data: dict, tool: str, resolution: dict[str, str]) -> ToolSpec:
    entry = data["tools"][tool]
    params = []
    for key in TOOL_PARAMS[tool]:
        is_resolution = tool == "close_request" and key == "resolution"
        enum = tuple(resolution[r] for r in RESOLUTIONS) if is_resolution else None
        params.append(ParamSpec(key=key, name=entry["params"][key]["name"],
                                description=entry["params"][key]["description"].strip(), enum=enum))
    return ToolSpec(key=tool, name=entry["name"], description=entry["description"].strip(),
                    params=tuple(params))
