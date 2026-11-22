"""Reads one assistant message and says whether it is exactly one usable tool call.

STRICT MODE (the only mode that the reward uses) accepts exactly this text, and nothing else:

    PREAMBLE <tool_call> _ <function=NAME> _ PARAMETERS </function> _ </tool_call> _

    PARAMETERS   zero or more times:  <parameter=KEY> VALUE </parameter> _

    PREAMBLE  any text, also none. It is returned as `preamble`, as written.
    _         white space only (blanks, tabs, line breaks; what str.isspace() accepts), also none.
    NAME      the text up to the next ">". It must be exactly the visible name of one tool in `tools`.
    KEY       the text up to the next ">". It must be exactly a visible parameter name of that tool.
    VALUE     the text up to the first "</parameter>", with white space removed at both ends (str.strip()).
              It may contain any character, also line breaks, "<", "</function>" or "<parameter=".
              It must not be empty.

    More rules:
    - "<tool_call>" must be in the whole message exactly once, wherever it is (also inside a value).
    - Every parameter of the tool must be given, each one once, in any order.
    - A parameter with a list of allowed values (an enum) must have exactly one of them as its value.
    - A turn that was cut off by the token cap (`truncated=True`) is rejected, whatever its text is.

The first problem that is found decides the error code. The checks run in this order:

    1. truncated          the caller says that the turn was cut off
    2. no_call            "<tool_call>" is not in the text
    3. multiple_calls     "<tool_call>" is in the text more than once
    4. bad_format         the text from "<tool_call>" on does not follow the shape above
    5. text_after_call    something other than white space follows "</tool_call>"
    6. unknown_tool       NAME is not a tool
    7. for each parameter, in the order written:
       unknown_argument, duplicate_argument, empty_argument, bad_value
    8. missing_argument   a parameter of the tool was not given

So the shape is checked first (steps 1 to 5) and the meaning after it (steps 6 to 8).

On success `tool_key` is the tool's internal key and `args` maps each internal parameter key to its value, in
the order of the tool's parameters. A value stays as the model wrote it (after strip()): it is not translated.

LENIENT MODE (`strict=False`) is used only in later analysis, to tell a format failure from a wrong decision.
It runs the same steps with these differences, and everything that strict mode accepts it accepts with the
same result:
    - several "<tool_call>": the first one is read and the others are ignored;
    - reading stops at "</function>": whatever follows it is ignored (text after the call, a missing
      "</tool_call>");
    - the parameter list may also end at "</tool_call>" or at the end of the text (a missing "</function>");
    - a parameter that the tool does not have is ignored;
    - a tool name or parameter name that has no exact match is compared without regard to case;
    - if the first character after "<tool_call>" and white space is "{", the call is read as a JSON object
      {"name": NAME, "arguments": {KEY: VALUE, ...}} with string values; what follows the object is ignored.

The text is read once from left to right (str.count, str.find, str.startswith, and a loop over white space),
so the time is linear in the length of the text. No regular expression is used.
"""
from __future__ import annotations

import json
from typing import Sequence, TypeVar

from accessdesk.harness.render import (CLOSE_CALL, CLOSE_FUNCTION, CLOSE_PARAMETER, OPEN_CALL, OPEN_FUNCTION,
                                       OPEN_PARAMETER)
from accessdesk.types import ParamSpec, ParseResult, ToolSpec

Named = TypeVar("Named", ToolSpec, ParamSpec)      # both have a visible `name`


class _Reject(Exception):
    """Internal: the message is not a usable call. Carries one code of types.PARSE_ERRORS.

    It is caught in parse_assistant_message and never leaves this module.
    """

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def parse_assistant_message(text: str, tools: list[ToolSpec], *, strict: bool = True, truncated: bool = False,
                            resolution_values: dict[str, str] | None = None) -> ParseResult:
    """Parses one assistant message (without its final `<|im_end|>`). See the module text for the grammar.

    Returns ParseResult(ok=True, tool_key, args, preamble) or ParseResult(ok=False, error, preamble).
    `preamble` is the text before the first `<tool_call>`, or "" if there is none or the turn was truncated.

    Never raises for any string. A `text` that is not a string is a bug of the caller and not a mistake of the
    model, so it raises TypeError and is not counted as a malformed call.

    `resolution_values` is in the signature of the build specification and is not used: the allowed values of
    a parameter come from its `enum`, and the caller maps a visible value to its internal key.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, not {type(text).__name__}")
    if truncated:
        return ParseResult(ok=False, error="truncated")
    n_calls = text.count(OPEN_CALL)
    if n_calls == 0:
        return ParseResult(ok=False, error="no_call")
    start = text.find(OPEN_CALL)
    preamble = text[:start]
    try:
        if strict and n_calls > 1:
            raise _Reject("multiple_calls")
        pos = _skip_space(text, start + len(OPEN_CALL))
        if not strict and text.startswith("{", pos):
            name, pairs = _read_json_call(text, pos)
        else:
            name, pairs = _read_call(text, pos, strict)
        tool_key, args = _check_call(name, pairs, tools, strict)
    except _Reject as reject:
        return ParseResult(ok=False, error=reject.code, preamble=preamble)
    return ParseResult(ok=True, tool_key=tool_key, args=args, preamble=preamble)


def _skip_space(text: str, pos: int) -> int:
    """The first position at or after `pos` that is not white space (len(text) if there is none)."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _read_name(text: str, pos: int, opening: str) -> tuple[str, int]:
    """Reads `opening`, a name and ">" at `pos`. Returns the name and the position after ">"."""
    if not text.startswith(opening, pos):
        raise _Reject("bad_format")
    start = pos + len(opening)
    end = text.find(">", start)
    if end == -1:
        raise _Reject("bad_format")
    return text[start:end], end + 1


def _read_call(text: str, pos: int, strict: bool) -> tuple[str, list[tuple[str, str]]]:
    """Reads `<function=NAME>`, the parameters and the closing tags, from `pos` on.

    Returns NAME and the (KEY, stripped VALUE) pairs in the order written. Checks the shape only.
    """
    name, pos = _read_name(text, pos, OPEN_FUNCTION)
    pairs: list[tuple[str, str]] = []
    while True:
        pos = _skip_space(text, pos)
        if text.startswith(OPEN_PARAMETER, pos):
            key, pos = _read_name(text, pos, OPEN_PARAMETER)
            end = text.find(CLOSE_PARAMETER, pos)
            if end == -1:
                raise _Reject("bad_format")
            pairs.append((key, text[pos:end].strip()))
            pos = end + len(CLOSE_PARAMETER)
        elif text.startswith(CLOSE_FUNCTION, pos):
            pos += len(CLOSE_FUNCTION)
            break
        elif not strict and (pos == len(text) or text.startswith(CLOSE_CALL, pos)):
            return name, pairs
        else:
            raise _Reject("bad_format")
    if strict:
        pos = _skip_space(text, pos)
        if not text.startswith(CLOSE_CALL, pos):
            raise _Reject("bad_format")
        if _skip_space(text, pos + len(CLOSE_CALL)) != len(text):
            raise _Reject("text_after_call")
    return name, pairs


def _read_json_call(text: str, pos: int) -> tuple[str, list[tuple[str, str]]]:
    """Lenient mode only: reads a JSON object {"name": .., "arguments": {..}} that starts at `pos`.

    Returns the same as `_read_call`. Every argument value must be a JSON string. If a key is written twice,
    JSON keeps the last one. The caller has checked that text[pos] is "{", so valid JSON here is an object.
    """
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, pos)
    except (ValueError, RecursionError):      # not JSON, or nested too deeply
        raise _Reject("bad_format") from None
    name, arguments = obj.get("name"), obj.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise _Reject("bad_format")
    pairs: list[tuple[str, str]] = []
    for key, value in arguments.items():
        if not isinstance(value, str):
            raise _Reject("bad_format")
        pairs.append((key, value.strip()))
    return name, pairs


def _find(items: Sequence[Named], name: str, strict: bool) -> Named | None:
    """The first tool or parameter whose visible name is exactly `name`. In lenient mode, if there is none:
    the first one whose name is equal without regard to case. None if nothing matches."""
    for item in items:
        if item.name == name:
            return item
    if not strict:
        for item in items:
            if item.name.lower() == name.lower():
                return item
    return None


def _check_call(name: str, pairs: list[tuple[str, str]], tools: list[ToolSpec],
                strict: bool) -> tuple[str, dict[str, str]]:
    """Checks the meaning of a call that has the right shape. Returns the internal tool key and the arguments
    by internal parameter key, in the order of the tool's parameters."""
    tool = _find(tools, name, strict)
    if tool is None:
        raise _Reject("unknown_tool")
    values: dict[str, str] = {}
    for key, value in pairs:
        param = _find(tool.params, key, strict)
        if param is None:
            if strict:
                raise _Reject("unknown_argument")
            continue
        if param.key in values:
            raise _Reject("duplicate_argument")
        if value == "":
            raise _Reject("empty_argument")
        if param.enum is not None and value not in param.enum:
            raise _Reject("bad_value")
        values[param.key] = value
    for param in tool.params:
        if param.key not in values:
            raise _Reject("missing_argument")
    return tool.key, {param.key: values[param.key] for param in tool.params}
