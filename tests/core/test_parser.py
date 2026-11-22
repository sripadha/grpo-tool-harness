"""parse_assistant_message: every error code, the round trip with the renderer, strict against lenient mode,
and hostile input.

All tool and parameter names are made up ("tool_alpha", "param_one"), and every internal key differs from its
visible name ("key_alpha", "key_one"), so a mix-up of keys and names fails a test.

The four tables below are also the input of one property test: whatever strict mode accepts, lenient mode
accepts with the same result.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from accessdesk.harness.parser import parse_assistant_message
from accessdesk.harness.render import render_tool_call
from accessdesk.rng import Rng
from accessdesk.types import PARSE_ERRORS, ParamSpec, ParseResult, ToolSpec

TOKENIZER_DIR = Path(__file__).resolve().parents[2] / ".cache" / "hf" / "Qwen3.5-4B"

P_ONE = ParamSpec("key_one", "param_one", "Made-up first parameter.")
P_TWO = ParamSpec("key_two", "param_two", "Made-up second parameter.")
P_COLOUR = ParamSpec("key_colour", "param_colour", "Made-up parameter with a fixed list of values.",
                     enum=("value_red", "value green", "Value_Blue"))
TOOL_NONE = ToolSpec("key_none", "tool_none", "Made-up tool without parameters.", ())
TOOL_ALPHA = ToolSpec("key_alpha", "tool_alpha", "Made-up tool with one parameter.", (P_ONE,))
TOOL_BETA = ToolSpec("key_beta", "tool_beta", "Made-up tool with two parameters.", (P_ONE, P_TWO))
TOOL_GAMMA = ToolSpec("key_gamma", "tool_gamma", "Made-up tool with three parameters.",
                      (P_ONE, P_TWO, P_COLOUR))
TOOLS = [TOOL_NONE, TOOL_ALPHA, TOOL_BETA, TOOL_GAMMA]


# Call texts written by this test file, not by the renderer under test.
def _param(name: str, value: str) -> str:
    return f"<parameter={name}>\n{value}\n</parameter>\n"


def _one(value: str) -> str:
    return _param("param_one", value)


def _two(value: str) -> str:
    return _param("param_two", value)


def _colour(value: str) -> str:
    return _param("param_colour", value)


def _nine(value: str) -> str:
    """A parameter that no tool has."""
    return _param("param_nine", value)


def _call(name: str, *params: str) -> str:
    return f"<tool_call>\n<function={name}>\n" + "".join(params) + "</function>\n</tool_call>"


def _gamma(colour: str) -> str:
    """A call of the three-parameter tool; only the value of the enum parameter varies."""
    return _call("tool_gamma", _one("a"), _two("b"), _colour(colour))


def _ok(tool_key: str, args: dict[str, str], preamble: str = "") -> ParseResult:
    return ParseResult(ok=True, tool_key=tool_key, args=args, preamble=preamble)


def _error(code: str, preamble: str = "") -> ParseResult:
    return ParseResult(ok=False, error=code, preamble=preamble)


ALPHA = _call("tool_alpha", _one("abc"))
BETA = _call("tool_beta", _one("abc"), _two("d e"))
ALPHA_ARGS = {"key_one": "abc"}
BETA_ARGS = {"key_one": "abc", "key_two": "d e"}
OK_ALPHA = _ok("key_alpha", ALPHA_ARGS)
OK_BETA = _ok("key_beta", BETA_ARGS)
TAG_LIKE_PREAMBLE = "</tool_call> <function=tool_beta> x </function>\n"
JSON_ALPHA = '{"name": "tool_alpha", "arguments": {"param_one": "abc"}}'
TEMPLATE_EXAMPLE_VALUE = "This is the value for the second parameter\nthat can span\nmultiple lines"

# ---------------------------------------------------------------------------
# Table 1: texts that strict mode accepts. (label, text, expected result)
# ---------------------------------------------------------------------------
STRICT_OK: list[tuple[str, str, ParseResult]] = [
    ("one_parameter", ALPHA, OK_ALPHA),
    ("two_parameters", BETA, OK_BETA),
    ("three_parameters_with_enum", _gamma("value green"),
     _ok("key_gamma", {"key_one": "a", "key_two": "b", "key_colour": "value green"})),
    ("no_parameters", _call("tool_none"), _ok("key_none", {})),
    ("all_on_one_line",
     "<tool_call><function=tool_beta><parameter=param_one>abc</parameter>"
     "<parameter=param_two>d e</parameter></function></tool_call>", OK_BETA),
    ("extra_white_space_between_the_parts",
     "<tool_call> \n\n\t<function=tool_beta>\n\n <parameter=param_one>  abc  </parameter> \t\n\n"
     "<parameter=param_two>\n\nd e\n\n</parameter>\n \n</function>\n\n\n</tool_call>", OK_BETA),
    ("white_space_after_the_call", ALPHA + " \n\t\n", OK_ALPHA),
    ("text_before_the_call_is_the_preamble", "I look the person up first.\n\n" + ALPHA,
     _ok("key_alpha", ALPHA_ARGS, "I look the person up first.\n\n")),
    ("preamble_with_tag_like_text", TAG_LIKE_PREAMBLE + ALPHA,
     _ok("key_alpha", ALPHA_ARGS, TAG_LIKE_PREAMBLE)),
    ("closing_function_tag_inside_a_value", _call("tool_alpha", _one("a </function> b")),
     _ok("key_alpha", {"key_one": "a </function> b"})),
    ("closing_call_tag_inside_a_value", _call("tool_alpha", _one("a </tool_call> b")),
     _ok("key_alpha", {"key_one": "a </tool_call> b"})),
    ("opening_parameter_tag_inside_a_value",
     _call("tool_beta", _one("<parameter=param_two>\nnested"), _two("real")),
     _ok("key_beta", {"key_one": "<parameter=param_two>\nnested", "key_two": "real"})),
    ("function_block_inside_a_value", _call("tool_alpha", _one("<function=tool_none>\n</function>")),
     _ok("key_alpha", {"key_one": "<function=tool_none>\n</function>"})),
    ("blank_lines_around_a_value_are_removed", _call("tool_alpha", _one("\n\n\n  two words \t\n\n\n")),
     _ok("key_alpha", {"key_one": "two words"})),
    ("white_space_inside_a_value_is_kept", _call("tool_alpha", _one("a  b\n\n\tc")),
     _ok("key_alpha", {"key_one": "a  b\n\n\tc"})),
    ("unicode_value", _call("tool_alpha", _one("Zoë Ünal 東京 🙂")),
     _ok("key_alpha", {"key_one": "Zoë Ünal 東京 🙂"})),
    ("angle_brackets_and_quotes", _call("tool_alpha", _one("a < b > \"c\" 'd' <e/> &amp;")),
     _ok("key_alpha", {"key_one": "a < b > \"c\" 'd' <e/> &amp;"})),
    ("json_text_as_a_value", _call("tool_alpha", _one(JSON_ALPHA)),
     _ok("key_alpha", {"key_one": JSON_ALPHA})),
    ("parameters_in_another_order_come_back_in_the_tools_order",
     _call("tool_gamma", _colour("Value_Blue"), _two("b"), _one("a")),
     _ok("key_gamma", {"key_one": "a", "key_two": "b", "key_colour": "Value_Blue"})),
    ("the_example_in_the_chat_templates_instructions",
     "<tool_call>\n<function=tool_beta>\n<parameter=param_one>\nvalue_1\n</parameter>\n"
     "<parameter=param_two>\n" + TEMPLATE_EXAMPLE_VALUE + "\n</parameter>\n</function>\n</tool_call>",
     _ok("key_beta", {"key_one": "value_1", "key_two": TEMPLATE_EXAMPLE_VALUE})),
]

# ---------------------------------------------------------------------------
# Table 2: texts that strict mode rejects. (label, text, error code)
# ---------------------------------------------------------------------------
STRICT_ERRORS: list[tuple[str, str, str]] = [
    # no_call
    ("empty_text", "", "no_call"),
    ("only_white_space", " \n\t", "no_call"),
    ("plain_answer", "The request is done.", "no_call"),
    ("function_block_without_the_call_tag", "<function=tool_none>\n</function>", "no_call"),
    ("call_tag_in_other_case", ALPHA.replace("tool_call", "TOOL_CALL"), "no_call"),
    ("only_a_closing_call_tag", "</tool_call>", "no_call"),
    # multiple_calls
    ("two_calls", ALPHA + "\n" + BETA, "multiple_calls"),
    ("the_same_call_twice", ALPHA + ALPHA, "multiple_calls"),
    ("opening_call_tag_inside_a_value", _call("tool_alpha", _one("a <tool_call> b")), "multiple_calls"),
    ("opening_call_tag_in_the_preamble", "I write a <tool_call> now.\n" + ALPHA, "multiple_calls"),
    ("two_calls_and_the_first_is_broken", "<tool_call>\n" + ALPHA, "multiple_calls"),
    # bad_format
    ("call_tag_alone", "<tool_call>", "bad_format"),
    ("empty_block", "<tool_call>\n</tool_call>", "bad_format"),
    ("missing_closing_call_tag", ALPHA.removesuffix("</tool_call>"), "bad_format"),
    ("missing_closing_function_tag", ALPHA.replace("</function>\n", ""), "bad_format"),
    ("missing_both_closing_tags", ALPHA.removesuffix("</function>\n</tool_call>"), "bad_format"),
    ("missing_closing_parameter_tag", ALPHA.replace("</parameter>\n", ""), "bad_format"),
    ("function_tag_never_closed", "<tool_call>\n<function=tool_none\n</function\n</tool_call", "bad_format"),
    ("function_tag_without_equals_sign", "<tool_call>\n<function>tool_none</function>\n</tool_call>",
     "bad_format"),
    ("text_between_call_tag_and_function", _call("tool_none").replace("<function", "here:\n<function"),
     "bad_format"),
    ("text_between_function_and_parameter", ALPHA.replace("<parameter", "abc\n<parameter"), "bad_format"),
    ("text_between_two_parameters", BETA.replace("<parameter=param_two", "and\n<parameter=param_two"),
     "bad_format"),
    ("text_between_function_end_and_call_end", ALPHA.replace("</tool_call>", "end\n</tool_call>"),
     "bad_format"),
    ("closing_tags_in_the_wrong_order",
     ALPHA.replace("</function>\n</tool_call>", "</tool_call>\n</function>"), "bad_format"),
    ("second_function_in_one_call", _call("tool_none", "</function>\n<function=tool_none>\n"), "bad_format"),
    ("closing_parameter_tag_inside_a_value_ends_it_early", _call("tool_alpha", _one("a </parameter> b")),
     "bad_format"),
    ("json_body", "<tool_call>\n" + JSON_ALPHA + "\n</tool_call>", "bad_format"),
    ("parameter_given_as_an_attribute", '<tool_call>\n<function=tool_alpha param_one="abc"/>\n</tool_call>',
     "bad_format"),
    ("bad_format_is_reported_before_an_unknown_tool", "<tool_call>\n<function=tool_omega>\n</function>",
     "bad_format"),
    # unknown_tool
    ("tool_that_does_not_exist", _call("tool_omega", _one("abc")), "unknown_tool"),
    ("tool_name_in_other_case", ALPHA.replace("tool_alpha", "Tool_Alpha"), "unknown_tool"),
    ("tool_name_with_a_blank", ALPHA.replace("=tool_alpha>", "= tool_alpha>"), "unknown_tool"),
    ("internal_key_used_as_the_tool_name", ALPHA.replace("tool_alpha", "key_alpha"), "unknown_tool"),
    ("empty_tool_name", _call(""), "unknown_tool"),
    ("unknown_tool_is_reported_before_argument_problems", _call("tool_omega", _one(" ")), "unknown_tool"),
    # missing_argument
    ("one_of_two_parameters", _call("tool_beta", _one("abc")), "missing_argument"),
    ("the_other_one_of_two_parameters", _call("tool_beta", _two("abc")), "missing_argument"),
    ("no_parameter_at_all", _call("tool_alpha"), "missing_argument"),
    # unknown_argument
    ("extra_parameter", _call("tool_alpha", _one("abc"), _nine("x")), "unknown_argument"),
    ("parameter_of_another_tool", _call("tool_alpha", _one("abc"), _two("x")), "unknown_argument"),
    ("parameter_name_in_other_case", _call("tool_alpha", _param("Param_One", "abc")), "unknown_argument"),
    ("internal_key_used_as_the_parameter_name", _call("tool_alpha", _param("key_one", "abc")),
     "unknown_argument"),
    ("parameter_for_a_tool_without_parameters", _call("tool_none", _one("abc")), "unknown_argument"),
    ("unknown_parameter_written_before_an_empty_one", _call("tool_beta", _nine("x"), _one(" ")),
     "unknown_argument"),
    # duplicate_argument
    ("same_parameter_twice", _call("tool_alpha", _one("abc"), _one("abc")), "duplicate_argument"),
    ("same_parameter_twice_with_two_values", _call("tool_beta", _one("a"), _two("b"), _one("c")),
     "duplicate_argument"),
    ("duplicate_is_reported_before_a_missing_parameter", _call("tool_beta", _one("a"), _one("b")),
     "duplicate_argument"),
    # empty_argument
    ("empty_value", _call("tool_alpha", "<parameter=param_one></parameter>\n"), "empty_argument"),
    ("value_of_blanks_and_line_breaks", _call("tool_alpha", _one("  \n\t \n")), "empty_argument"),
    ("value_of_a_no_break_space", _call("tool_alpha", _one(" ")), "empty_argument"),
    ("empty_parameter_written_before_an_unknown_one", _call("tool_beta", _one(" "), _nine("x")),
     "empty_argument"),
    # bad_value
    ("value_outside_the_enum", _gamma("value_purple"), "bad_value"),
    ("enum_value_in_other_case", _gamma("VALUE_RED"), "bad_value"),
    ("enum_value_with_more_text", _gamma("value_red."), "bad_value"),
    ("two_enum_values", _gamma("value_red value green"), "bad_value"),
    # text_after_call
    ("sentence_after_the_call", ALPHA + "\nDone.", "text_after_call"),
    ("end_of_turn_token_not_removed", ALPHA + "<|im_end|>", "text_after_call"),
    ("closing_call_tag_twice", ALPHA + "</tool_call>", "text_after_call"),
    ("text_after_call_is_reported_before_an_unknown_tool", _call("tool_omega") + " x", "text_after_call"),
]

# ---------------------------------------------------------------------------
# Table 3: where the two modes differ, and where lenient mode must stay firm.
# (label, text, strict error code, lenient result)
# ---------------------------------------------------------------------------
LENIENT: list[tuple[str, str, str, ParseResult]] = [
    ("text_after_the_call", ALPHA + "\nDone.", "text_after_call", OK_ALPHA),
    ("several_calls_takes_the_first", "Two steps.\n" + ALPHA + "\n" + BETA, "multiple_calls",
     _ok("key_alpha", ALPHA_ARGS, "Two steps.\n")),
    ("several_calls_and_the_first_is_wrong", _call("tool_omega") + "\n" + ALPHA, "multiple_calls",
     _error("unknown_tool")),
    ("missing_closing_call_tag", ALPHA.removesuffix("</tool_call>"), "bad_format", OK_ALPHA),
    ("missing_closing_function_tag", BETA.replace("</function>\n", ""), "bad_format", OK_BETA),
    ("missing_both_closing_tags", BETA.removesuffix("</function>\n</tool_call>"), "bad_format", OK_BETA),
    ("text_between_function_end_and_call_end", ALPHA.replace("</tool_call>", "end\n</tool_call>"),
     "bad_format", OK_ALPHA),
    ("unknown_extra_parameter_is_ignored", _call("tool_alpha", _nine(" "), _one("abc"), _nine("x")),
     "unknown_argument", OK_ALPHA),
    ("tool_name_in_other_case", ALPHA.replace("tool_alpha", "TOOL_Alpha"), "unknown_tool", OK_ALPHA),
    ("parameter_name_in_other_case", BETA.replace("param_two", "PARAM_Two"), "unknown_argument", OK_BETA),
    ("json_body", "<tool_call>\n" + JSON_ALPHA + "\n</tool_call>", "bad_format", OK_ALPHA),
    ("json_body_with_preamble_text_after_and_no_closing_tag", "Next.\n<tool_call>" + JSON_ALPHA + " etc.",
     "bad_format", _ok("key_alpha", ALPHA_ARGS, "Next.\n")),
    ("json_body_with_names_in_other_case_extra_key_and_padded_value",
     '<tool_call>{"name": "Tool_Beta", "arguments": '
     '{"PARAM_ONE": " abc\\n", "param_two": "d e", "param_nine": "x"}}', "bad_format", OK_BETA),
    # Lenient mode stays firm on these.
    ("no_call", "The request is done.", "no_call", _error("no_call")),
    ("missing_closing_parameter_tag", ALPHA.replace("</parameter>\n", ""), "bad_format",
     _error("bad_format")),
    ("text_between_call_tag_and_function", _call("tool_none").replace("<function", "here:\n<function"),
     "bad_format", _error("bad_format")),
    ("text_between_two_parameters", BETA.replace("<parameter=param_two", "and\n<parameter=param_two"),
     "bad_format", _error("bad_format")),
    ("unknown_tool", _call("tool_omega"), "unknown_tool", _error("unknown_tool")),
    ("missing_parameter", _call("tool_beta", _one("abc")), "missing_argument", _error("missing_argument")),
    ("missing_parameter_and_no_closing_tags", "<tool_call>\n<function=tool_beta>\n" + _one("abc"),
     "bad_format", _error("missing_argument")),
    ("same_parameter_twice", _call("tool_alpha", _one("a"), _one("a")), "duplicate_argument",
     _error("duplicate_argument")),
    ("same_parameter_twice_in_two_cases", _call("tool_alpha", _one("a"), _param("PARAM_ONE", "a")),
     "unknown_argument", _error("duplicate_argument")),
    ("empty_value", _call("tool_alpha", _one(" ")), "empty_argument", _error("empty_argument")),
    ("enum_value_in_other_case", _gamma("VALUE_RED"), "bad_value", _error("bad_value")),
    ("json_unknown_tool", '<tool_call>{"name": "tool_omega", "arguments": {}}</tool_call>', "bad_format",
     _error("unknown_tool")),
    ("json_missing_parameter", '<tool_call>{"name": "tool_beta", "arguments": {"param_one": "abc"}}',
     "bad_format", _error("missing_argument")),
    ("json_empty_value", '<tool_call>{"name": "tool_alpha", "arguments": {"param_one": "  "}}',
     "bad_format", _error("empty_argument")),
    ("json_value_outside_the_enum",
     '<tool_call>{"name": "tool_gamma", "arguments": '
     '{"param_one": "a", "param_two": "b", "param_colour": "x"}}', "bad_format", _error("bad_value")),
    ("json_value_that_is_not_a_string", '<tool_call>{"name": "tool_alpha", "arguments": {"param_one": 7}}',
     "bad_format", _error("bad_format")),
    ("json_arguments_that_are_not_an_object", '<tool_call>{"name": "tool_alpha", "arguments": "abc"}',
     "bad_format", _error("bad_format")),
    ("json_without_arguments", '<tool_call>{"name": "tool_none"}</tool_call>', "bad_format",
     _error("bad_format")),
    ("json_without_a_name", '<tool_call>{"arguments": {"param_one": "abc"}}</tool_call>', "bad_format",
     _error("bad_format")),
    ("json_name_that_is_not_a_string", '<tool_call>{"name": 5, "arguments": {}}</tool_call>', "bad_format",
     _error("bad_format")),
    ("json_cut_off", '<tool_call>{"name": "tool_alpha", "arguments": {"param_one": "ab', "bad_format",
     _error("bad_format")),
    ("json_list_instead_of_an_object", "<tool_call>[" + JSON_ALPHA + "]</tool_call>", "bad_format",
     _error("bad_format")),
]

# ---------------------------------------------------------------------------
# Table 4: hostile input of about 200,000 characters.
# (label, text, strict result: an error code or "ok", lenient result)
# ---------------------------------------------------------------------------
BIG = 200_000
OPEN_ALPHA = "<tool_call><function=tool_alpha>"
OPEN_NONE = "<tool_call><function=tool_none>"
CLOSE = "</function></tool_call>"
HOSTILE: list[tuple[str, str, str, str]] = [
    ("empty_string", "", "no_call", "no_call"),
    ("opening_brackets", "<" * BIG, "no_call", "no_call"),
    ("closing_brackets", ">" * BIG, "no_call", "no_call"),
    ("null_characters", "\x00" * BIG, "no_call", "no_call"),
    ("opening_call_tags", "<tool_call>" * (BIG // 11), "multiple_calls", "bad_format"),
    ("closing_call_tags", "</tool_call>" * (BIG // 12), "no_call", "no_call"),
    ("opening_parameter_tags", "<parameter=a>" * (BIG // 13), "no_call", "no_call"),
    ("nested_unclosed_tags_repeated", "<tool_call><function=a><parameter=b>" * (BIG // 36),
     "multiple_calls", "bad_format"),
    ("one_call_tag_then_function_tags", "<tool_call>" + "<function=tool_alpha>" * (BIG // 21),
     "bad_format", "bad_format"),
    ("one_call_then_unclosed_parameters", OPEN_ALPHA + "<parameter=param_one>" * (BIG // 21),
     "bad_format", "bad_format"),
    ("unclosed_parameters_then_one_closing_tag",
     OPEN_ALPHA + "<parameter=param_one>" * (BIG // 21) + "</parameter>", "bad_format", "ok"),
    ("tags_that_never_end", "<tool_call>" + "<function=a<parameter=b" * (BIG // 23),
     "bad_format", "bad_format"),
    ("very_long_tool_name", "<tool_call><function=" + "a" * BIG + ">" + CLOSE,
     "unknown_tool", "unknown_tool"),
    ("white_space_after_the_call_tag", "<tool_call>" + " " * BIG, "bad_format", "bad_format"),
    ("white_space_everywhere",
     "<tool_call>" + "\n" * BIG + "<function=tool_none>" + " " * BIG + "</function>" + "\t" * BIG
     + "</tool_call>" + "\n" * BIG, "ok", "ok"),
    ("white_space_then_text_after_the_call", _call("tool_none") + " " * BIG + "x", "text_after_call", "ok"),
    ("the_same_parameter_many_times",
     OPEN_ALPHA + "<parameter=param_one>x</parameter>" * (BIG // 34) + CLOSE,
     "duplicate_argument", "duplicate_argument"),
    ("an_unknown_parameter_many_times",
     OPEN_NONE + "<parameter=param_nine>x</parameter>" * (BIG // 35) + CLOSE,
     "unknown_argument", "ok"),
    ("closing_function_tags_inside_an_unclosed_value",
     OPEN_ALPHA + "<parameter=param_one>" + "</function>" * (BIG // 11), "bad_format", "bad_format"),
    ("json_braces", "<tool_call>" + "{" * BIG, "bad_format", "bad_format"),
    ("json_deeply_nested_lists", '<tool_call>{"name": ' + "[" * BIG, "bad_format", "bad_format"),
    ("json_deeply_nested_objects", "<tool_call>" + '{"a":' * (BIG // 5), "bad_format", "bad_format"),
    ("json_very_long_string", '<tool_call>{"name": "' + "a" * BIG, "bad_format", "bad_format"),
]

# ---------------------------------------------------------------------------
# Generated calls for the round trip with the renderer
# ---------------------------------------------------------------------------
VALUE_PIECES = [
    "abc", "two words", "x", "7", "</function>", "</tool_call>", "<parameter=param_one>",
    "<function=tool_beta>", "</param", "<", ">", "a > b", '"quoted"', "it's", "<b>bold</b>", "&amp;", "Zoë",
    "東京", "🙂", "\\n", "{}", '{"name": "tool_alpha"}', "param_one=1", "<|im_end|>", "<think>", "value_red",
]
VALUE_JOINTS = [" ", "\n", "\n\n", "\t", "", "  "]


def _generated_value(rng: Rng) -> str:
    """One to three pieces with joints between them. No white space at the ends, never empty."""
    value = rng.choice(VALUE_PIECES)
    for _ in range(rng.below(3)):
        value += rng.choice(VALUE_JOINTS) + rng.choice(VALUE_PIECES)
    return value


def _generated_calls(n: int, seed: int) -> list[tuple[ToolSpec, dict[str, str], str]]:
    """(tool, arguments by internal key in the tool's order, rendered text with the parameters shuffled)."""
    rng = Rng(seed)
    calls = []
    for _ in range(n):
        tool = rng.choice(TOOLS)
        by_key = {p.key: rng.choice(p.enum) if p.enum is not None else _generated_value(rng)
                  for p in tool.params}
        written = {p.name: by_key[p.key] for p in rng.shuffle(tool.params)}
        calls.append((tool, by_key, render_tool_call(tool.name, written)))
    return calls


GENERATED = _generated_calls(400, seed=20261116)

ALL_TEXTS: list[str] = (
    [row[1] for row in STRICT_OK] + [row[1] for row in STRICT_ERRORS] + [row[1] for row in LENIENT]
    + [row[1] for row in HOSTILE] + [row[2] for row in GENERATED])


def _ids(table: list[tuple]) -> list[str]:
    return [row[0] for row in table]


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label, text, expected", STRICT_OK, ids=_ids(STRICT_OK))
def test_strict_accepts(label, text, expected):
    result = parse_assistant_message(text, TOOLS)
    assert result == expected
    assert list(result.args) == list(expected.args)     # the order of the tool's parameters, not as written


@pytest.mark.parametrize("label, text, code", STRICT_ERRORS, ids=_ids(STRICT_ERRORS))
def test_strict_rejects(label, text, code):
    result = parse_assistant_message(text, TOOLS)
    assert (result.ok, result.error, result.tool_key, result.args) == (False, code, None, None)


# One short input for each error code, written out in full.
# `True` in the second place means: the turn was cut off by the token cap.
ONE_PER_CODE: dict[str, tuple[str, bool]] = {
    "no_call": ("The request is done.", False),
    "multiple_calls": (
        "<tool_call>\n<function=tool_none>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=tool_none>\n</function>\n</tool_call>", False),
    "bad_format": ("<tool_call>\n<function=tool_none>\n</function>", False),
    "unknown_tool": ("<tool_call>\n<function=tool_omega>\n</function>\n</tool_call>", False),
    "missing_argument": ("<tool_call>\n<function=tool_alpha>\n</function>\n</tool_call>", False),
    "unknown_argument": (
        "<tool_call>\n<function=tool_none>\n<parameter=param_nine>\nabc\n</parameter>\n</function>\n"
        "</tool_call>", False),
    "duplicate_argument": (
        "<tool_call>\n<function=tool_alpha>\n<parameter=param_one>\nabc\n</parameter>\n"
        "<parameter=param_one>\nabc\n</parameter>\n</function>\n</tool_call>", False),
    "empty_argument": (
        "<tool_call>\n<function=tool_alpha>\n<parameter=param_one>\n\n</parameter>\n</function>\n"
        "</tool_call>", False),
    "bad_value": (
        "<tool_call>\n<function=tool_gamma>\n<parameter=param_one>\na\n</parameter>\n"
        "<parameter=param_two>\nb\n</parameter>\n<parameter=param_colour>\nvalue_purple\n</parameter>\n"
        "</function>\n</tool_call>", False),
    "text_after_call": ("<tool_call>\n<function=tool_none>\n</function>\n</tool_call>\nDone.", False),
    "truncated": ("<tool_call>\n<function=tool_alpha>\n<parameter=param_one>\nab", True),
}


@pytest.mark.parametrize("code", PARSE_ERRORS)
def test_each_error_code_has_an_input_that_triggers_it(code):
    text, truncated = ONE_PER_CODE[code]
    assert parse_assistant_message(text, TOOLS, truncated=truncated) == _error(code)


def test_the_error_examples_cover_exactly_the_error_codes():
    assert tuple(ONE_PER_CODE) == PARSE_ERRORS
    assert {row[2] for row in STRICT_ERRORS} == set(PARSE_ERRORS) - {"truncated"}


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("text", [ALPHA, "", "no call here", ALPHA + ALPHA, "<tool_call>"])
def test_truncated_turn_is_rejected_whatever_the_text_is(text, strict):
    assert parse_assistant_message(text, TOOLS, strict=strict, truncated=True) == _error("truncated")


def test_preamble_is_kept_when_the_call_is_rejected():
    result = parse_assistant_message("Some thought.\n" + _call("tool_omega"), TOOLS)
    assert result == _error("unknown_tool", preamble="Some thought.\n")


def test_preamble_is_empty_without_a_call():
    assert parse_assistant_message("Some thought.", TOOLS) == _error("no_call", preamble="")


@pytest.mark.parametrize("value", P_COLOUR.enum)
@pytest.mark.parametrize("strict", [True, False])
def test_each_enum_value_is_accepted(value, strict):
    result = parse_assistant_message(_gamma(value), TOOLS, strict=strict)
    assert result == _ok("key_gamma", {"key_one": "a", "key_two": "b", "key_colour": value})


def test_a_parameter_without_enum_takes_any_value():
    """The value that is outside the enum of param_colour is fine for param_one."""
    text = _call("tool_alpha", _one("value_purple"))
    assert parse_assistant_message(text, TOOLS) == _ok("key_alpha", {"key_one": "value_purple"})


def test_only_the_tools_that_are_passed_are_known():
    assert parse_assistant_message(ALPHA, [TOOL_NONE, TOOL_BETA]).error == "unknown_tool"
    assert parse_assistant_message(ALPHA, []).error == "unknown_tool"


def test_resolution_values_argument_changes_nothing():
    for text in (ALPHA, BETA, "no call"):
        assert (parse_assistant_message(text, TOOLS, resolution_values={"key_x": "abc"})
                == parse_assistant_message(text, TOOLS))


@pytest.mark.parametrize("not_text", [None, b"<tool_call>", 5, ["<tool_call>"]])
def test_input_that_is_not_a_string_is_a_callers_bug(not_text):
    """It must not be counted as a malformed call of the model, so it raises."""
    with pytest.raises(TypeError):
        parse_assistant_message(not_text, TOOLS)


# ---------------------------------------------------------------------------
# Round trip with the renderer
# ---------------------------------------------------------------------------

def test_round_trip_of_generated_calls():
    """parse(render(call)) gives the call back, for 400 generated calls with the parameters shuffled."""
    for tool, by_key, text in GENERATED:
        for strict in (True, False):
            result = parse_assistant_message(text, TOOLS, strict=strict)
            assert result == _ok(tool.key, by_key), text
            assert list(result.args) == [p.key for p in tool.params], text


def test_generated_calls_are_varied():
    """Guards the round-trip test against a generator that produces nearly nothing."""
    texts = {text for _, _, text in GENERATED}
    values = [value for _, by_key, _ in GENERATED for value in by_key.values()]
    assert len(texts) >= 250          # about a quarter of the 400 are the tool without parameters: one text
    assert {tool.key for tool, _, _ in GENERATED} == {tool.key for tool in TOOLS}
    for needle in ("\n", "</function>", "<parameter=param_one>", "</tool_call>", "東京", '"'):
        assert any(needle in value for value in values), needle
    assert all(value == value.strip() and value != "" for value in values)
    assert not any("</parameter>" in value or "<tool_call>" in value for value in values)


def test_round_trip_with_a_preamble_and_final_line_break():
    text = "Thought.\n\n" + render_tool_call("tool_beta", {"param_two": "d e", "param_one": "abc"}) + "\n"
    assert parse_assistant_message(text, TOOLS) == _ok("key_beta", BETA_ARGS, "Thought.\n\n")


def test_round_trip_removes_white_space_around_a_value():
    text = render_tool_call("tool_alpha", {"param_one": "\n\n  abc \n"})
    assert parse_assistant_message(text, TOOLS) == OK_ALPHA


# ---------------------------------------------------------------------------
# Lenient mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label, text, strict_code, lenient_result", LENIENT, ids=_ids(LENIENT))
def test_lenient_against_strict(label, text, strict_code, lenient_result):
    assert parse_assistant_message(text, TOOLS).error == strict_code
    assert parse_assistant_message(text, TOOLS, strict=False) == lenient_result


def test_lenient_prefers_the_exact_name():
    """Two tools and two parameters that differ only by case: the exact one wins, wherever it is."""
    p_lower = ParamSpec("key_lower", "param_x", "Made-up.")
    p_upper = ParamSpec("key_upper", "PARAM_X", "Made-up.")
    lower = ToolSpec("key_tool_lower", "tool_x", "Made-up.", (p_lower,))
    upper = ToolSpec("key_tool_upper", "TOOL_X", "Made-up.", (p_upper,))
    for tools in ([lower, upper], [upper, lower]):
        assert (parse_assistant_message(_call("TOOL_X", _param("PARAM_X", "v")), tools, strict=False)
                == _ok("key_tool_upper", {"key_upper": "v"}))
        assert (parse_assistant_message(_call("tool_x", _param("param_x", "v")), tools, strict=False)
                == _ok("key_tool_lower", {"key_lower": "v"}))
    both = ToolSpec("key_both", "tool_both", "Made-up.", (p_lower, p_upper))
    text = _call("tool_both", _param("PARAM_X", "upper"), _param("param_x", "lower"))
    for strict in (True, False):
        assert (parse_assistant_message(text, [both], strict=strict)
                == _ok("key_both", {"key_lower": "lower", "key_upper": "upper"}))


def test_strict_ok_implies_lenient_ok_with_the_same_result():
    """The property, over every text of this file."""
    n_strict_ok = n_lenient_only = 0
    for text in ALL_TEXTS:
        strict = parse_assistant_message(text, TOOLS)
        lenient = parse_assistant_message(text, TOOLS, strict=False)
        if strict.ok:
            n_strict_ok += 1
            assert lenient == strict, text[:200]
            assert list(lenient.args) == list(strict.args), text[:200]
        elif lenient.ok:
            n_lenient_only += 1
    assert n_strict_ok >= len(STRICT_OK) + len(GENERATED)                # the property was really checked
    assert n_lenient_only >= sum(1 for row in LENIENT if row[3].ok)      # and lenient mode is really wider


# ---------------------------------------------------------------------------
# Hostile input: never raises, linear time
# ---------------------------------------------------------------------------

def _code(result: ParseResult) -> str:
    return "ok" if result.ok else result.error


def _timed_code(text: str, strict: bool) -> tuple[str, float]:
    started = time.perf_counter()
    result = parse_assistant_message(text, TOOLS, strict=strict)
    return _code(result), time.perf_counter() - started


@pytest.mark.parametrize("label, text, strict_code, lenient_code", HOSTILE, ids=_ids(HOSTILE))
def test_hostile_input_is_answered_fast_and_without_an_exception(label, text, strict_code, lenient_code):
    assert text == "" or len(text) >= 0.99 * BIG
    for strict, expected in ((True, strict_code), (False, lenient_code)):
        code, seconds = _timed_code(text, strict)
        assert code == expected
        assert seconds < 1.0, f"{seconds:.2f} s with strict={strict}"


def test_time_stays_linear_on_input_ten_times_longer():
    """2,000,000 characters with about 57,000 parameters. Measured on the build machine: 0.15 s.

    A version that copies the rest of the text at every step (text[pos:].lstrip()) passes the
    200,000-character test above (0.05 s) and needs several seconds here, so this is the test that catches it.
    """
    unit = "<parameter=param_nine>x</parameter>\n"
    text = OPEN_NONE + unit * (10 * BIG // len(unit)) + CLOSE
    for strict, expected in ((True, "unknown_argument"), (False, "ok")):
        code, seconds = _timed_code(text, strict)
        assert code == expected
        assert seconds < 2.0, f"{seconds:.2f} s with strict={strict}"


def test_every_result_is_well_formed():
    """Over every text of this file, in both modes: ok with key and arguments, or one known error code."""
    for text in ALL_TEXTS:
        for strict in (True, False):
            result = parse_assistant_message(text, TOOLS, strict=strict)
            assert isinstance(result, ParseResult)
            if result.ok:
                tool = next(tool for tool in TOOLS if tool.key == result.tool_key)
                assert result.error is None
                assert list(result.args) == [p.key for p in tool.params]
                assert all(isinstance(v, str) and v == v.strip() and v != "" for v in result.args.values())
            else:
                assert result.error in PARSE_ERRORS
                assert result.tool_key is None and result.args is None
            assert text.startswith(result.preamble)


# ---------------------------------------------------------------------------
# With the real tokenizer: what the model writes as tokens parses back
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.is_dir():
        pytest.skip(f"tokenizer files are missing: {TOKENIZER_DIR}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@pytest.mark.tokenizer
def test_round_trip_through_token_ids(tokenizer):
    """Render, encode with the end-of-turn token, decode, strip that token (as the harness does), parse."""
    end = "<|im_end|>"
    for tool, by_key, text in GENERATED[:100]:
        ids = tokenizer.encode(text + end, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded.endswith(end)
        assert parse_assistant_message(decoded.removesuffix(end), TOOLS) == _ok(tool.key, by_key), text
