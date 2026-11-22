"""render_tool_call: the exact text, by hand and against the real chat template of the model.

All tool and parameter names here are made up on purpose ("tool_alpha", "param_one"): the tests are about the
format, not about the desk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from accessdesk.harness.render import render_tool_call
from accessdesk.types import ParamSpec, ToolSpec

TOKENIZER_DIR = Path(__file__).resolve().parents[2] / ".cache" / "hf" / "Qwen3.5-4B"

# What the template writes before an assistant turn that comes after the last user request.
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

TOOL_GAMMA = ToolSpec(
    key="key_gamma", name="tool_gamma", description="Made-up tool with three parameters.",
    params=(ParamSpec("key_one", "param_one", "Made-up first parameter."),
            ParamSpec("key_two", "param_two", "Made-up second parameter."),
            ParamSpec("key_three", "param_three", "Made-up third parameter.")))

# (label, arguments by visible parameter name). One, two and three parameters, and values that could break a
# careless renderer or a template that escapes.
TEMPLATE_CASES: list[tuple[str, dict[str, str]]] = [
    ("one_parameter", {"param_one": "abc"}),
    ("two_parameters", {"param_one": "abc", "param_two": "def"}),
    ("three_parameters", {"param_one": "abc", "param_two": "def", "param_three": "ghi"}),
    ("no_parameters", {}),
    ("order_as_given", {"param_three": "3", "param_one": "1", "param_two": "2"}),
    ("spaces", {"param_one": "two words", "param_two": "three  spaced   words"}),
    ("double_and_single_quotes", {"param_one": 'she said "yes"', "param_two": "it's"}),
    ("angle_brackets_and_ampersand", {"param_one": "<b>bold</b> & a < b > c", "param_two": "</function>"}),
    ("line_breaks", {"param_one": "line one\nline two\n\nline four", "param_two": "a\r\nb\tc"}),
    ("white_space_at_the_ends", {"param_one": "  padded  ", "param_two": "\n\nblank lines around\n\n"}),
    ("only_white_space", {"param_one": " ", "param_two": "\n"}),
    ("looks_like_json", {"param_one": '{"a": 1}', "param_two": "true", "param_three": "123"}),
    ("unicode", {"param_one": "Zoë Ünal", "param_two": "東京 🙂"}),
    ("backslashes_and_braces", {"param_one": "a\\nb {{ x }} {% y %}", "param_two": "100%"}),
]


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.is_dir():
        pytest.skip(f"tokenizer files are missing: {TOKENIZER_DIR}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def _assistant_call(tool_name: str, args: dict[str, str]) -> dict:
    """An assistant message with empty content and one structured tool call, as the template expects it."""
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": tool_name, "arguments": args}}]}


# ---------------------------------------------------------------------------
# By hand (no tokenizer needed)
# ---------------------------------------------------------------------------

def test_exact_text_for_one_parameter():
    assert render_tool_call("tool_alpha", {"param_one": "abc"}) == (
        "<tool_call>\n<function=tool_alpha>\n<parameter=param_one>\nabc\n</parameter>\n</function>\n"
        "</tool_call>")


def test_exact_text_for_three_parameters():
    text = render_tool_call("tool_gamma", {"param_one": "a b", "param_two": "c\nd", "param_three": "<e>"})
    assert text.split("\n") == [
        "<tool_call>",
        "<function=tool_gamma>",
        "<parameter=param_one>", "a b", "</parameter>",
        "<parameter=param_two>", "c", "d", "</parameter>",
        "<parameter=param_three>", "<e>", "</parameter>",
        "</function>",
        "</tool_call>",
    ]


def test_exact_text_without_parameters():
    text = render_tool_call("tool_alpha", {})
    assert text == "<tool_call>\n<function=tool_alpha>\n</function>\n</tool_call>"


def test_parameters_are_written_in_the_order_given():
    text = render_tool_call("tool_gamma", {"param_two": "2", "param_one": "1"})
    assert text.index("<parameter=param_two>") < text.index("<parameter=param_one>")


def test_value_is_written_unchanged():
    """Nothing is stripped or escaped: the template does not do it either."""
    value = "  \n <x> & \"y\" \n\n"
    assert render_tool_call("tool_alpha", {"param_one": value}) == (
        "<tool_call>\n<function=tool_alpha>\n<parameter=param_one>\n" + value + "\n</parameter>\n"
        "</function>\n</tool_call>")


def test_no_text_before_or_after_the_call():
    text = render_tool_call("tool_alpha", {"param_one": "abc"})
    assert text.startswith("<tool_call>")
    assert text.endswith("</tool_call>")      # no final line break and no <|im_end|>


# ---------------------------------------------------------------------------
# Against the real chat template
# ---------------------------------------------------------------------------

@pytest.mark.tokenizer
@pytest.mark.parametrize("label, args", TEMPLATE_CASES, ids=[case[0] for case in TEMPLATE_CASES])
def test_same_bytes_as_the_chat_template(tokenizer, label, args):
    """The template's render of [system, user, assistant call] ends with header + our text + end of turn."""
    messages = [{"role": "system", "content": "Made-up system text."},
                {"role": "user", "content": "Made-up request."},
                _assistant_call("tool_gamma", args)]
    rendered = tokenizer.apply_chat_template(messages, tools=[TOOL_GAMMA.schema()], tokenize=False)
    call = render_tool_call("tool_gamma", args)
    assert rendered.endswith(ASSISTANT_HEADER + call + "<|im_end|>\n")
    assert rendered.count("<tool_call>\n<function=tool_gamma>") == 1


@pytest.mark.tokenizer
def test_same_bytes_in_the_middle_of_a_conversation(tokenizer):
    """Two calls with a tool result between them: both are at the right place, with nothing added."""
    first = {"param_one": "first value", "param_two": "x < y", "param_three": "line\nbreak"}
    second = {"param_one": 'second "value"', "param_two": "z", "param_three": "w"}
    response = '{"made_up_key": "made-up result"}'
    messages = [{"role": "system", "content": "Made-up system text."},
                {"role": "user", "content": "Made-up request."},
                _assistant_call("tool_gamma", first),
                {"role": "tool", "content": response},
                _assistant_call("tool_gamma", second)]
    rendered = tokenizer.apply_chat_template(messages, tools=[TOOL_GAMMA.schema()], tokenize=False)
    expected_end = (
        "<|im_start|>user\nMade-up request.<|im_end|>\n"
        + ASSISTANT_HEADER + render_tool_call("tool_gamma", first) + "<|im_end|>\n"
        + "<|im_start|>user\n<tool_response>\n" + response + "\n</tool_response><|im_end|>\n"
        + ASSISTANT_HEADER + render_tool_call("tool_gamma", second) + "<|im_end|>\n")
    assert rendered.endswith(expected_end)


@pytest.mark.tokenizer
def test_same_bytes_without_a_tool_list(tokenizer):
    """The call text does not depend on whether tool schemas are in the system turn."""
    args = {"param_one": "abc", "param_two": "d e"}
    messages = [{"role": "user", "content": "Made-up request."}, _assistant_call("tool_gamma", args)]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    assert rendered == ("<|im_start|>user\nMade-up request.<|im_end|>\n"
                        + ASSISTANT_HEADER + render_tool_call("tool_gamma", args) + "<|im_end|>\n")


@pytest.mark.tokenizer
def test_same_format_as_the_example_in_the_templates_instructions(tokenizer):
    """The system turn that the template writes shows the model an example call. Ours has the same shape."""
    messages = [{"role": "user", "content": "Made-up request."}]
    rendered = tokenizer.apply_chat_template(messages, tools=[TOOL_GAMMA.schema()], tokenize=False)
    example = render_tool_call("example_function_name", {
        "example_parameter_1": "value_1",
        "example_parameter_2": "This is the value for the second parameter\nthat can span\nmultiple lines"})
    assert rendered.count(example) == 1
