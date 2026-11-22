"""The text of one tool call, exactly as the model's chat template writes it.

The format belongs to the chat template of the model (`.cache/hf/Qwen3.5-4B/chat_template.jinja`). The six
tags below are the only place in the repository where it is written down. The parser imports them from here.
"""
from __future__ import annotations

OPEN_CALL = "<tool_call>"
CLOSE_CALL = "</tool_call>"
OPEN_FUNCTION = "<function="      # followed by the tool name and ">"
CLOSE_FUNCTION = "</function>"
OPEN_PARAMETER = "<parameter="    # followed by the parameter name and ">"
CLOSE_PARAMETER = "</parameter>"


def render_tool_call(tool_name: str, args: dict[str, str]) -> str:
    """The assistant text for one tool call, with nothing before or after it.

    `tool_name` is the model-visible tool name. `args` maps model-visible parameter names to values, in the
    order in which they are written. The result is byte-for-byte what the chat template renders for an
    assistant message with empty content and this one tool call:

        <tool_call>
        <function=NAME>
        <parameter=KEY>
        VALUE
        </parameter>
        </function>
        </tool_call>

    There is no line break after `</tool_call>` and no `<|im_end|>`: the template adds the end-of-turn token.

    Nothing is checked or escaped, as in the template. A value is written as it is, with its own white space.
    The parser strips white space around a value, ends a value at the first `</parameter>` and counts every
    `<tool_call>`. So `parse(render(name, args))` gives back `args` only if no value has white space at its
    ends, is empty, or contains `</parameter>` or `<tool_call>`, and if no name contains `>`.
    """
    parts = [f"{OPEN_CALL}\n{OPEN_FUNCTION}{tool_name}>\n"]
    for key, value in args.items():
        parts.append(f"{OPEN_PARAMETER}{key}>\n{value}\n{CLOSE_PARAMETER}\n")
    parts.append(f"{CLOSE_FUNCTION}\n{CLOSE_CALL}")
    return "".join(parts)
