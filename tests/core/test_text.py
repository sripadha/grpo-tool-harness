"""accessdesk/text.py and the two files in text/."""
from __future__ import annotations

import copy
import hashlib
import unicodedata
from pathlib import Path

import pytest
import yaml

from accessdesk.text import (ALL_TOOL_KEYS, ECHO_LIMIT, ECHO_PUNCTUATION, ERROR_KEYS, FORBIDDEN, RESULT_KEYS,
                             TOOL_PARAMS, TextBundle, TextError, check_text, load_text, model_text_path,
                             normalise, placeholder_path, safe_echo)
from accessdesk.types import EXTRA_TOOL_KEY, RESOLUTIONS, TOOL_KEYS

PLACEHOLDER_DATA = yaml.safe_load(placeholder_path().read_text(encoding="utf-8"))
HEADER = ("Placeholder written by the coding assistant (a closed API model) "
          "for tests and the smoke test only. "
          "Nothing here is used in any reported run. "
          "Do not copy a sentence from this file into model_text.yaml.")


def write_changed(tmp_path: Path, change) -> Path:
    """A copy of the placeholder file with one change applied to its parsed content."""
    data = copy.deepcopy(PLACEHOLDER_DATA)
    change(data)
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def put(*path_and_value):
    """A change that sets data[path[0]][path[1]]... = value."""
    *path, value = path_and_value

    def change(data: dict) -> None:
        for key in path[:-1]:
            data = data[key]
        data[path[-1]] = value
    return change


def drop(*path):
    def change(data: dict) -> None:
        for key in path[:-1]:
            data = data[key]
        del data[path[-1]]
    return change


# ---------------------------------------------------------------------------
# The placeholder file
# ---------------------------------------------------------------------------
def test_placeholder_file_is_complete_and_valid():
    assert check_text(placeholder_path()) == []


def test_unchanged_copy_of_the_placeholder_is_valid(tmp_path):
    # The base case of every "one change gives a problem" test below.
    assert check_text(write_changed(tmp_path, lambda data: None)) == []


def test_placeholder_header_comment():
    raw = placeholder_path().read_text(encoding="utf-8")
    comment_lines = []
    for line in raw.splitlines():
        if not line.startswith("#"):
            break
        comment_lines.append(line.lstrip("#").strip())
    assert " ".join(" ".join(comment_lines).split()).startswith(HEADER)


def test_bundle_meta(text):
    assert isinstance(text, TextBundle)
    assert text.is_placeholder is True
    assert text.path == str(placeholder_path())
    assert text.sha256 == hashlib.sha256(placeholder_path().read_bytes()).hexdigest()
    assert text.policy_shift_variant == "all_grants_need_approval"


def test_tool_specs_order_and_toolsets(text):
    assert [spec.key for spec in text.tool_specs()] == list(TOOL_KEYS)
    assert [spec.key for spec in text.tool_specs("five")] == list(TOOL_KEYS)
    assert [spec.key for spec in text.tool_specs("six")] == list(TOOL_KEYS) + [EXTRA_TOOL_KEY]
    assert set(text.tools) == set(ALL_TOOL_KEYS)
    with pytest.raises(ValueError):
        text.tool_specs("seven")


def test_tool_params_follow_the_args_column_of_the_specification(text):
    assert TOOL_PARAMS == {
        "lookup_person": ("name",),
        "check_access": ("person_id", "role_name"),
        "request_approval": ("person_id", "role_id", "approver_id"),
        "grant_role": ("person_id", "role_id"),
        "close_request": ("resolution", "person_id", "role_id"),
        "department_info": ("department",),
    }
    for key in ALL_TOOL_KEYS:
        spec = text.tools[key]
        assert spec.key == key
        assert tuple(param.key for param in spec.params) == TOOL_PARAMS[key]
        assert spec.name == PLACEHOLDER_DATA["tools"][key]["name"]
        assert spec.description == PLACEHOLDER_DATA["tools"][key]["description"]
        for param in spec.params:
            assert param.name == PLACEHOLDER_DATA["tools"][key]["params"][param.key]["name"]
            assert param.description == PLACEHOLDER_DATA["tools"][key]["params"][param.key]["description"]


def test_only_the_resolution_parameter_has_an_enum_in_resolutions_order(text):
    for key in ALL_TOOL_KEYS:
        for param in text.tools[key].params:
            if (key, param.key) == ("close_request", "resolution"):
                assert param.enum == tuple(text.resolution[r] for r in RESOLUTIONS)
                assert param.enum == ("granted", "pending_approval", "already_has_role", "denied")
            else:
                assert param.enum is None
    schema = text.tools["close_request"].schema()
    assert schema["function"]["parameters"]["properties"]["outcome"]["enum"] == [
        "granted", "pending_approval", "already_has_role", "denied"]


def test_tool_by_name(text):
    for spec in text.tool_specs("six"):
        assert text.tool_by_name(spec.name, "six") is spec
    extra = text.tools[EXTRA_TOOL_KEY]
    assert text.tool_by_name(extra.name, "five") is None
    assert text.tool_by_name(extra.name) is None
    # The placeholder's visible names differ from the internal keys, so a key must not be found as a name.
    assert text.tool_by_name("lookup_person", "six") is None
    assert text.tool_by_name("find_person", "five").key == "lookup_person"
    assert text.tool_by_name("Find_person", "five") is None
    assert text.tool_by_name(" find_person", "five") is None


def test_resolution_key(text):
    for key in RESOLUTIONS:
        assert text.resolution_key(text.resolution[key]) == key
        assert text.resolution_key(f"  {text.resolution[key]}\n") == key
    assert text.resolution_key("pending_approval") == "pending"
    assert text.resolution_key("pending") is None           # the internal key is not a visible value here
    assert text.resolution_key("GRANTED") is None
    assert text.resolution_key("") is None


def test_error_text(text):
    assert text.error_text("unknown_person_id", "per_abc234") == "No person has the id per_abc234."
    assert text.error_text("unknown_role_name", "Wiki Editor") == "No role has the name Wiki Editor."
    assert text.error_text("malformed_call") == PLACEHOLDER_DATA["errors"]["malformed_call"]
    assert set(text.errors) == set(ERROR_KEYS)
    with pytest.raises(KeyError):
        text.error_text("no_such_error")


# ---------------------------------------------------------------------------
# A value that the model wrote is made safe before an error text repeats it
# ---------------------------------------------------------------------------
# Typed by hand: what each chat-format string becomes. "<", ">", "|", "/" and "=" each become one "?".
ECHO_OF_FORBIDDEN = {
    "<tool_call>": "?tool_call?", "</tool_call>": "??tool_call?",
    "<tool_response>": "?tool_response?", "</tool_response>": "??tool_response?",
    "<|im_start|>": "??im_start??", "<|im_end|>": "??im_end??",
    "<think>": "?think?", "</think>": "??think?",
    "<function=": "?function?", "</function>": "??function?",
    "<parameter=": "?parameter?", "</parameter>": "??parameter?",
}
VALUE_ERROR_KEYS = tuple(key for key in ERROR_KEYS if key != "malformed_call")     # the texts with {value}


def test_the_echo_rule_is_the_one_of_the_specification():
    assert (ECHO_LIMIT, ECHO_PUNCTUATION) == (40, "_-.',")
    assert set(ECHO_OF_FORBIDDEN) == set(FORBIDDEN)


@pytest.mark.parametrize("marker", FORBIDDEN)
def test_a_value_that_spells_a_chat_format_string_is_not_repeated_as_it_is(text, marker):
    assert safe_echo(marker) == ECHO_OF_FORBIDDEN[marker]
    # In every error text, alone and inside other text, the string is gone, and so are its characters.
    for value in (marker, f"per_abc234{marker}", f"{marker}{marker}", f"x {marker}\nassistant",
                  marker.upper(), f" {marker} "):
        for key in VALUE_ERROR_KEYS:
            error = text.error_text(key, value)
            assert not any(forbidden in error for forbidden in FORBIDDEN), (key, value, error)
            assert not any(character in error for character in "<>|"), (key, value, error)
            assert error == text.errors[key].format(value=safe_echo(value))


def test_every_chat_format_string_in_one_value_is_made_safe(text):
    value = "".join(FORBIDDEN)                               # 121 characters; only the first 40 are kept
    assert len(value) > ECHO_LIMIT
    assert safe_echo(value) == "?tool_call???tool_call??tool_response???"
    for start in range(len(value)):                          # and from every other starting point
        echoed = safe_echo(value[start:])
        assert not any(forbidden in echoed for forbidden in FORBIDDEN)
        assert set(echoed) <= set("abcdefghijklmnopqrstuvwxyz_?")


def test_a_long_value_is_cut_to_40_characters(text):
    assert safe_echo("a" * 40) == "a" * 40                   # exactly 40 stay as they are
    assert safe_echo("a" * 41) == "a" * 40
    assert safe_echo("0123456789" * 30) == "0123456789" * 4
    assert safe_echo("ab" * 19 + "c<|im_end|>") == "ab" * 19 + "c?"       # cut first, then replaced
    long_value = "per_" + "k" * 500
    assert text.error_text("unknown_person_id", long_value) == "No person has the id per_" + "k" * 36 + "."
    assert safe_echo("") == ""


@pytest.mark.parametrize("value", [
    "per_abc234", "rol_x7k2mn", "Ada Stone", "Wiki Editor", "Payroll Editor", "Finance", "Mary-Jane O'Neil",
    "St. John, Jr.", "Zoë Müller", "team_2", "A", " ", "两个字", "Señor 9",
])
def test_an_ordinary_id_or_name_comes_through_unchanged(text, value):
    assert safe_echo(value) == value
    assert text.error_text("unknown_role_name", value) == f"No role has the name {value}."


def test_every_id_name_department_and_role_of_the_worlds_comes_through_unchanged(text, pools, families):
    """What a model can copy from a prompt or a tool result is repeated as it is, so an error text about a
    real id or name (for example a role id given where a person id belongs) stays readable."""
    values = set(text.departments) | set(text.roles)
    for first_names, last_names in pools.values():
        values |= set(first_names) | set(last_names)
    for split in families:
        for family in families[split]:
            values |= {person["person_id"] for person in family["world"]["people"]}
            values |= {person["name"] for person in family["world"]["people"]}
            values |= {role["role_id"] for role in family["world"]["roles"]}
    assert len(values) > 1000
    changed = sorted(value for value in values if safe_echo(value) != value)
    assert changed == []


@pytest.mark.parametrize("value, wanted", [
    ("{x}", "?x?"), ("{value}", "?value?"), ("a\nb", "a?b"), ("a\tb", "a?b"), ("a\u00a0b", "a?b"),
    ("a/b\\c", "a?b?c"), ('say "hi"', "say ?hi?"), ("a=b;c:d", "a?b?c?d"), ("x@y.z", "x?y.z"),
    ("\uff1cthink\uff1e", "?think?"), ("[1]+(2)*3", "?1???2??3"), ("it's 5,5 - ok_", "it's 5,5 - ok_"),
    ("\x00\x1b[0m", "???0m"), ("&lt;think&gt;", "?lt?think?gt?"),
])
def test_every_other_character_becomes_a_question_mark(value, wanted):
    assert safe_echo(value) == wanted
    assert len(safe_echo(value)) == len(value)               # one character for one character


def test_no_kept_character_can_turn_into_a_character_of_the_chat_format():
    """Letters and digits of every script are kept. None of them is "<", ">", "|", "/" or "=", and none
    becomes one of them when a tokenizer brings text to a normal form (all four Unicode forms are tried)."""
    characters = [chr(code) for code in range(0x110000)]
    kept = [c for c in characters if c.isalpha() or c.isdigit() or c == " " or c in ECHO_PUNCTUATION]
    assert len(kept) > 100_000
    assert all(safe_echo(character) == character for character in kept[::997])
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        normal = unicodedata.normalize(form, "".join(kept))
        assert not set(normal) & set("<>|/=\n\r"), form


def test_the_echo_does_not_change_an_error_text_without_a_value(text):
    assert text.error_text("malformed_call", "<|im_end|>") == PLACEHOLDER_DATA["errors"]["malformed_call"]
    for key in VALUE_ERROR_KEYS:
        assert "{value}" in PLACEHOLDER_DATA["errors"][key]


def test_groups_hold_exactly_the_internal_keys(text):
    assert tuple(text.result_keys) == RESULT_KEYS
    assert tuple(text.status) == ("active", "inactive")
    assert tuple(text.resolution) == RESOLUTIONS
    assert tuple(text.confirmation) == ("approval_requested", "role_granted", "request_closed")
    assert sorted(text.request_templates) == [1, 2, 3, 4]
    for templates in text.request_templates.values():
        assert isinstance(templates, tuple) and len(templates) >= 3
    assert len(text.departments) >= 6 and len(text.roles) >= 12
    assert text.id_prefix_person == "per_" and text.id_prefix_role == "rol_"


def test_prompts_are_stripped_and_differ(text):
    prompts = [text.system_prompt, text.plain_system_prompt, text.policy_shift_system_prompt]
    assert all(prompt == prompt.strip() and prompt for prompt in prompts)
    assert len(set(prompts)) == 3
    assert "{" not in text.system_prompt and "{" not in text.policy_shift_system_prompt


def test_placeholder_system_prompt_states_the_rules_in_plan_order(text):
    """PLAN 5.1: look up; check access; inactive -> denied; has access -> already; sensitive -> approval and
    pending; otherwise grant and granted."""
    prompt = text.system_prompt
    name = {key: text.tools[key].name for key in TOOL_KEYS}
    in_order = [name["lookup_person"], name["check_access"],
                text.resolution["denied"], text.resolution["already"],
                name["request_approval"], text.resolution["pending"],
                name["grant_role"], text.resolution["granted"]]
    positions = [prompt.find(word) for word in in_order]
    assert -1 not in positions, dict(zip(in_order, positions))
    assert positions == sorted(positions), dict(zip(in_order, positions))
    assert name["close_request"] in prompt
    assert text.tools[EXTRA_TOOL_KEY].name not in prompt
    assert "exactly one tool" in prompt and "Use only ids that a tool has returned" in prompt


def test_placeholder_policy_shift_prompt_describes_its_variant(text):
    prompt = text.policy_shift_system_prompt
    for key in ("lookup_person", "check_access", "request_approval", "close_request"):
        assert text.tools[key].name in prompt
    for key in ("denied", "already", "pending"):
        assert text.resolution[key] in prompt
    assert text.resolution["granted"] not in prompt      # under this rule set nothing is granted directly
    assert "exactly one tool" in prompt


def test_plain_prompt_has_no_rule_list(text):
    for value in text.resolution.values():
        assert value not in text.plain_system_prompt
    for spec in text.tool_specs("six"):
        assert spec.name not in text.plain_system_prompt


def test_templates_fill_in(text):
    for templates in text.request_templates.values():
        for template in templates:
            filled = template.format(person_name="PERSON", department="DEPARTMENT", role_name="ROLE")
            assert all(filled.count(word) == 1 for word in ("PERSON", "DEPARTMENT", "ROLE"))
            assert "{" not in filled and "}" not in filled


def test_normalise():
    assert normalise("  Ada   LOVELACE\n") == "ada lovelace"
    assert normalise("Straße") == normalise("STRASSE")
    assert normalise("a b") != normalise("ab")


# ---------------------------------------------------------------------------
# The candidate's file: same structure, nothing written yet, nothing copied
# ---------------------------------------------------------------------------
def structure(node: object, prefix: tuple = ()) -> dict[tuple, str]:
    """path -> kind of value ("list", "bool" or "str") for every value that is not a mapping."""
    if isinstance(node, dict):
        found: dict[tuple, str] = {}
        for key, value in node.items():
            found.update(structure(value, prefix + (key,)))
        return found
    return {prefix: type(node).__name__}


MODEL_DATA = yaml.safe_load(model_text_path().read_text(encoding="utf-8"))


def test_model_text_has_the_same_structure_as_the_placeholder():
    assert structure(MODEL_DATA) == structure(PLACEHOLDER_DATA)


def leaf_values(node: object, prefix: tuple = ()) -> dict[tuple, object]:
    """path -> value for every leaf of a parsed text file (a list is one leaf)."""
    if isinstance(node, dict):
        found: dict[tuple, object] = {}
        for key, value in node.items():
            found.update(leaf_values(value, prefix + (key,)))
        return found
    return {prefix: node}


def test_model_text_is_complete_and_is_not_the_placeholder():
    """Since 2026-11-17 text/model_text.yaml holds the text that an open-weight model wrote
    (scripts/gpu/draft_text.py). It is complete and valid, it is not marked as the placeholder, and it is not
    the placeholder's text under another name."""
    assert check_text(model_text_path()) == []
    bundle = load_text(model_text_path())
    assert bundle.is_placeholder is False
    assert bundle.sha256 != load_text(placeholder_path()).sha256
    assert MODEL_DATA["meta"]["placeholder"] is False
    assert MODEL_DATA["policy_shift"]["variant"] in ("all_grants_need_approval", "sensitive_denied")
    for path, value in leaf_values(MODEL_DATA).items():
        if path == ("meta", "placeholder"):
            continue
        if isinstance(value, list):
            assert value and all(isinstance(item, str) and item.strip() for item in value), path
        else:
            assert isinstance(value, str) and value.strip(), path
    # the three prompts and every template are the writer's own, not the placeholder's
    placeholder = leaf_values(PLACEHOLDER_DATA)
    for path in (("system_prompt",), ("plain_system_prompt",), ("policy_shift", "system_prompt")):
        assert leaf_values(MODEL_DATA)[path] != placeholder[path], path
    for style in ("style_1", "style_2", "style_3", "style_4"):
        ours = set(MODEL_DATA["request_templates"][style])
        assert len(ours) >= 3 and not ours & set(PLACEHOLDER_DATA["request_templates"][style]), style


def test_model_text_names_its_writer_as_an_open_weight_model():
    written_by = MODEL_DATA["meta"]["written_by"]
    assert "open-weight model" in written_by and "revision" in written_by
    assert "coding assistant" not in written_by and "closed" not in written_by


def test_an_emptied_model_text_is_reported_as_incomplete(tmp_path):
    """The model text with its values taken out again: every missing value is named, and loading fails."""
    def emptied(node: object, path: tuple = ()) -> object:
        if isinstance(node, dict):
            return {key: emptied(value, path + (key,)) for key, value in node.items()}
        if path in (("meta", "placeholder"), ("policy_shift", "variant")):
            return node
        return [] if isinstance(node, list) else ""

    path = tmp_path / "emptied.yaml"
    path.write_text(yaml.safe_dump(emptied(MODEL_DATA), sort_keys=False, allow_unicode=True), encoding="utf-8")
    problems = check_text(path)
    assert "system_prompt is empty" in problems
    assert "tools.lookup_person.name is empty" in problems
    assert "errors.malformed_call is empty" in problems
    assert "plain_system_prompt is empty" not in problems
    assert any(problem.startswith("departments needs at least 6") for problem in problems)
    assert any(problem.startswith("request_templates.style_4 needs at least 3") for problem in problems)
    with pytest.raises(TextError) as error:
        load_text(path)
    assert all(problem in str(error.value) for problem in problems)


def placeholder_sentences() -> list[str]:
    """Every line of placeholder text that is long enough to be a sentence and not only a name."""
    lines: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            lines.extend(line.strip() for line in node.splitlines())
    walk({key: value for key, value in PLACEHOLDER_DATA.items() if key != "meta"})
    return [line for line in lines if len(line) >= 20 and " " in line]


def test_model_text_copies_no_sentence_of_the_placeholder():
    model_raw = " ".join(model_text_path().read_text(encoding="utf-8").split()).casefold()
    sentences = placeholder_sentences()
    assert len(sentences) > 40                                 # the check below is not empty
    copied = [s for s in sentences if " ".join(s.split()).casefold() in model_raw]
    assert copied == []


def test_model_text_header_says_who_wrote_it_and_how():
    """The header comment of the file: the writer (an open-weight model with a pinned revision), the
    instruction file, the selection rule, the source of the five tool names, and the line for the reader."""
    raw = model_text_path().read_text(encoding="utf-8")
    header = " ".join(line.lstrip("# ") for line in raw.split("\nmeta:")[0].splitlines())
    assert "An open-weight model wrote the values in this file" in header
    assert "Written by: Qwen/Qwen3.5-9B, revision " in header
    assert "text/drafting_instructions.yaml" in header and "text/drafting_log.jsonl" in header
    assert "the first valid candidate in index order was taken" in header
    assert "Nobody chose between candidates" in header
    assert "Largest shared run among the values that were taken" in header
    assert "Read and approved by:" in header
    assert "The candidate writes every value in this file" not in header      # the header of the empty file
    for name in ("lookup_person", "check_access", "request_approval", "grant_role", "close_request"):
        assert name in header


# ---------------------------------------------------------------------------
# Validation: one change to a valid file gives the problem that names the field
# ---------------------------------------------------------------------------
def set_template(style: int, index: int, template: str):
    def change(data: dict) -> None:
        data["request_templates"][f"style_{style}"][index] = template
    return change


ONE_PROBLEM = [
    # (label, change, text that the single problem must contain)
    ("missing prompt", drop("system_prompt"), "system_prompt is missing"),
    ("missing shift prompt", drop("policy_shift", "system_prompt"), "policy_shift.system_prompt is missing"),
    ("empty description", put("tools", "grant_role", "description", "  "),
     "tools.grant_role.description is empty"),
    ("missing param", drop("tools", "check_access", "params", "role_name", "description"),
     "tools.check_access.params.role_name.description is missing"),
    ("missing result key", drop("result_keys", "member_count"), "result_keys.member_count is missing"),
    ("missing error", drop("errors", "unknown_department"), "errors.unknown_department is missing"),
    ("value is not a string", put("values", "status", "active", True),
     "values.status.active must be a string"),
    ("number as a name", put("id_prefixes", "person", 7), "id_prefixes.person must be a string"),
    ("placeholder flag", put("meta", "placeholder", "yes"), "meta.placeholder must be true or false"),
    ("unknown top field", put("extra_notes", "x"), "extra_notes is not a field"),
    ("unknown tool field", put("tools", "grant_role", "returns", "x"),
     "tools.grant_role.returns is not a field"),
    ("fifth style", put("request_templates", "style_5", ["a {person_name} {department} {role_name}"]),
     "request_templates.style_5 is not a field"),
    ("tool name with space", put("tools", "grant_role", "name", "give role"),
     "tools.grant_role.name must match"),
    ("tool name with digit first", put("tools", "grant_role", "name", "1give"),
     "tools.grant_role.name must match"),
    ("tool name with line break", put("tools", "grant_role", "name", "give_role\n"),
     "tools.grant_role.name must match"),
    ("tool name with hyphen", put("tools", "grant_role", "name", "give-role"),
     "tools.grant_role.name must match"),
    ("param name with dot", put("tools", "grant_role", "params", "role_id", "name", "role.id"),
     "tools.grant_role.params.role_id.name must match"),
    ("two tools with one name", put("tools", "grant_role", "name", "find_person"),
     "tool names must be unique"),
    ("extra tool with a taken name", put("tools", "department_info", "name", "close_ticket"),
     "tool names must be unique"),
    ("two params with one name", put("tools", "grant_role", "params", "role_id", "name", "person_id"),
     "parameter names of tools.grant_role must be unique"),
    ("two result keys alike", put("result_keys", "result", "error"), "result keys must be unique"),
    ("two status values alike", put("values", "status", "inactive", "active"),
     "status values must be unique"),
    ("two resolutions alike", put("values", "resolution", "denied", "granted"),
     "resolution values must be unique"),
    ("two confirmations alike", put("values", "confirmation", "role_granted", "ticket closed"),
     "confirmations must be unique"),
    ("resolution with line break", put("values", "resolution", "denied", "den\nied"),
     "values.resolution.denied must be one line"),
    ("resolution with space at end", put("values", "resolution", "denied", "denied "),
     "values.resolution.denied must be one line"),
    ("status with space at start", put("values", "status", "active", " active"),
     "values.status.active must be one line"),
    ("confirmation with line break", put("values", "confirmation", "role_granted", "role\ngiven"),
     "values.confirmation.role_granted must be one line"),
    ("result key with space at end", put("result_keys", "matches", "people "),
     "result_keys.matches must be one line"),
    ("id prefix with line break", put("id_prefixes", "role", "rol_\n"), "id_prefixes.role must be one line"),
    ("department with space at end", put("departments", 0, "Finance "), "departments[0] must be one line"),
    ("role with line break", put("roles", 3, "Source\nCode Reader"), "roles[3] must be one line"),
    ("template without a variable", set_template(1, 0, "Give {person_name} the {role_name} role."),
     "request_templates.style_1[0] must contain each of"),
    ("template with a variable twice",
     set_template(2, 1, "{person_name} {person_name} of {department} needs {role_name}."),
     "request_templates.style_2[1] must contain each of"),
    ("template with a fourth variable",
     set_template(3, 2, "{person_name} of {department} needs {role_name} by {date}."),
     "request_templates.style_3[2] must contain each of"),
    ("template with doubled braces",
     set_template(4, 0, "{person_name} of {department} needs {role_name} {{now}}."),
     "request_templates.style_4[0] must contain each of"),
    ("template with a lone brace", set_template(1, 1, "{person_name} of {department} needs {role_name} }."),
     "request_templates.style_1[1] must contain each of"),
    ("template with a format spec",
     set_template(1, 2, "{person_name:>30} of {department} needs {role_name}."),
     "request_templates.style_1[2] must contain each of"),
    ("template with a conversion", set_template(1, 2, "{person_name!r} of {department} needs {role_name}."),
     "request_templates.style_1[2] must contain each of"),
    ("template with an attribute", set_template(1, 2, "{person_name.x} of {department} needs {role_name}."),
     "request_templates.style_1[2] must contain each of"),
    ("too few templates",
     put("request_templates", "style_3", PLACEHOLDER_DATA["request_templates"]["style_3"][:2]),
     "request_templates.style_3 needs at least 3 templates, not 2"),
    ("held-out template copied into a train style",
     set_template(1, 0, PLACEHOLDER_DATA["request_templates"]["style_4"][2]),
     "request_templates.style_4[2] is the same template as request_templates.style_1[0]"),
    ("a template twice in one style, in other case",
     set_template(2, 2, PLACEHOLDER_DATA["request_templates"]["style_2"][0].upper()
                  .replace("{PERSON_NAME}", "{person_name}").replace("{DEPARTMENT}", "{department}")
                  .replace("{ROLE_NAME}", "{role_name}")),
     "request_templates.style_2[2] is the same template as request_templates.style_2[0]"),
    ("too few departments", put("departments", PLACEHOLDER_DATA["departments"][:5]),
     "departments needs at least 6 distinct entries, not 5"),
    ("too few roles", put("roles", PLACEHOLDER_DATA["roles"][:11]),
     "roles needs at least 12 distinct entries, not 11"),
    ("department twice in other case", put("departments", 1, "FINANCE"),
     "departments has entries that are equal"),
    ("role twice with other spacing", put("roles", 1, "Billing  Viewer"), "roles has entries that are equal"),
    ("error text with another variable", put("errors", "unknown_role_id", "No role {role}."),
     "errors.unknown_role_id may use the variable {value}"),
    ("error text with a lone brace", put("errors", "malformed_call", "Bad call {"),
     "errors.malformed_call may use the variable {value}"),
    ("error text with a format spec", put("errors", "unknown_role_id", "No role {value:>9}."),
     "errors.unknown_role_id may use the variable {value}"),
    ("variant is the default rule set", put("policy_shift", "variant", "default"),
     "policy_shift.variant must be"),
    ("variant is unknown", put("policy_shift", "variant", "approve_everything"),
     "policy_shift.variant must be"),
    ("resolution word inside a status value", put("values", "status", "inactive", "account denied"),
     "values.status.inactive contains the resolution value 'denied'"),
    ("resolution word inside a role name", put("roles", 0, "Granted Access Viewer"),
     "roles[0] contains the resolution value 'granted'"),
    ("resolution word inside a result key", put("result_keys", "has_access", "is_granted"),
     "result_keys.has_access contains the resolution value 'granted'"),
    ("resolution word inside a department", put("departments", 2, "Denied Claims"),
     "departments[2] contains the resolution value 'denied'"),
]


@pytest.mark.parametrize("label, change, wanted", ONE_PROBLEM, ids=[case[0] for case in ONE_PROBLEM])
def test_one_change_gives_exactly_the_problem_that_names_the_field(tmp_path, label, change, wanted):
    problems = check_text(write_changed(tmp_path, change))
    assert len(problems) == 1, problems
    assert wanted in problems[0]


NO_PROBLEM = [
    ("empty plain prompt", put("plain_system_prompt", "")),
    ("other variant", put("policy_shift", "variant", "sensitive_denied")),
    ("error text without a variable", put("errors", "unknown_role_id", "No such role.")),
    ("error text with the variable twice", put("errors", "unknown_role_id", "{value}? No role {value}.")),
    ("the same parameter name in two tools",
     put("tools", "lookup_person", "params", "name", "name", "person_id")),
    ("a fourth template", put("request_templates", "style_1", PLACEHOLDER_DATA["request_templates"]["style_1"]
                              + ["{role_name} for {person_name}, {department}."])),
    ("braces in a prompt", put("system_prompt", "Rules {in braces} are plain text here.")),
    ("placeholder flag off", put("meta", "placeholder", False)),
]


@pytest.mark.parametrize("label, change", NO_PROBLEM, ids=[case[0] for case in NO_PROBLEM])
def test_allowed_changes_give_no_problem(tmp_path, label, change):
    assert check_text(write_changed(tmp_path, change)) == []


TAG = "</tool_call>"
FIELDS_WITH_A_TAG = [
    put("system_prompt", f"Rules. {TAG} More."),
    put("plain_system_prompt", TAG),
    put("policy_shift", "system_prompt", f"Rules {TAG}"),
    put("tools", "close_request", "description", f"Close {TAG} it."),
    put("tools", "close_request", "params", "resolution", "description", TAG),
    put("errors", "malformed_call", f"Bad call {TAG}"),
    put("values", "confirmation", "request_closed", f"closed {TAG}"),
    put("values", "status", "active", f"active{TAG}"),
    put("id_prefixes", "person", TAG),
    put("result_keys", "matches", f"people{TAG}"),
    put("roles", 5, f"Role {TAG}"),
    put("departments", 5, f"Dept {TAG}"),
    set_template(4, 1, "{person_name} {department} {role_name} " + TAG),
]


@pytest.mark.parametrize("marker", FORBIDDEN)
def test_every_chat_format_string_is_refused(tmp_path, marker):
    change = put("tools", "grant_role", "description", f"Give {marker} now.")
    problems = check_text(write_changed(tmp_path, change))
    assert problems == [f"tools.grant_role.description contains {marker!r}, which belongs to the chat format"]


def test_the_list_of_chat_format_strings_is_the_specified_one():
    assert set(FORBIDDEN) == {"<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
                              "<|im_start|>", "<|im_end|>", "<think>", "</think>",
                              "<function=", "</function>", "<parameter=", "</parameter>"}


@pytest.mark.parametrize("index", range(len(FIELDS_WITH_A_TAG)))
def test_chat_format_strings_are_refused_in_every_kind_of_field(tmp_path, index):
    problems = check_text(write_changed(tmp_path, FIELDS_WITH_A_TAG[index]))
    assert len(problems) == 1 and f"contains {TAG!r}" in problems[0], problems


def test_free_text_is_stripped_and_names_are_kept(tmp_path):
    def change(data: dict) -> None:
        data["system_prompt"] = "\n\n  Rules here.  \n\n"
        data["tools"]["grant_role"]["description"] = "Give it.\n"
        data["tools"]["grant_role"]["params"]["role_id"]["description"] = "  The role.\n"
        data["errors"]["unknown_role_id"] = "No role {value}.\n"
        data["request_templates"]["style_1"][0] = "  {person_name} {department} {role_name}\n"
    bundle = load_text(write_changed(tmp_path, change))
    assert bundle.system_prompt == "Rules here."
    assert bundle.tools["grant_role"].description == "Give it."
    assert bundle.tools["grant_role"].params[1].description == "The role."
    assert bundle.errors["unknown_role_id"] == "No role {value}."
    assert bundle.request_templates[1][0] == "{person_name} {department} {role_name}"


def test_load_text_lists_every_problem(tmp_path):
    def change(data: dict) -> None:
        del data["system_prompt"]
        data["values"]["status"]["inactive"] = "active"
        data["roles"] = data["roles"][:3]
    path = write_changed(tmp_path, change)
    problems = check_text(path)
    assert len(problems) == 3
    with pytest.raises(TextError) as error:
        load_text(path)
    message = str(error.value)
    assert "3 problem(s)" in message and all(problem in message for problem in problems)
    assert isinstance(error.value, ValueError)


def test_a_key_written_twice_is_a_problem(tmp_path):
    raw = placeholder_path().read_text(encoding="utf-8")
    assert raw.count('\n  style_1:\n') == 1
    path = tmp_path / "twice.yaml"
    path.write_text(raw.replace("\n  style_2:\n", "\n  style_1:\n"), encoding="utf-8")
    problems = check_text(path)
    assert len(problems) == 1 and "'style_1' is written twice" in problems[0]
    with pytest.raises(TextError):
        load_text(path)


@pytest.mark.parametrize("raw, wanted", [
    ("tools: [unclosed", "cannot be read as YAML"),
    ("- a\n- b\n", "must be a YAML mapping"),
    ("", "must be a YAML mapping"),
])
def test_files_that_are_not_a_mapping(tmp_path, raw, wanted):
    path = tmp_path / "bad.yaml"
    path.write_text(raw, encoding="utf-8")
    problems = check_text(path)
    assert len(problems) == 1 and wanted in problems[0]


def test_missing_file_is_an_error_and_not_a_problem_list(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_text(tmp_path / "nowhere.yaml")
