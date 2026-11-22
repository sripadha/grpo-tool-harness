"""accessdesk/drafting.py, text/drafting_instructions.yaml and scripts/gpu/draft_text.py, with fake writers.

No model is used here. A fake writer knows which field it is asked for from the seed (every candidate of every
field has its own seed), and answers with the placeholder's value for that field, or with a reply that the
test gives it. The placeholder is used only as test data: the real writer never sees it.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from accessdesk import drafting
from accessdesk import text as text_module
from accessdesk.drafting import CUT_OFF_REASON, ReplyCutOff, candidate_seed, clean_reply
from accessdesk.rng import Rng
from accessdesk.text import check_text, load_text, placeholder_path
from accessdesk.types import TOOL_KEYS

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = yaml.safe_load(placeholder_path().read_text(encoding="utf-8"))
MODEL, REVISION = "Open/Writer-9B", "0123abcd"
N = 8
WHICH_PERSON = "tells you which person is meant"      # words of the placeholder's step that finds the person
FIRST_CASE = "Then take the first case that applies"  # words of the placeholder's step that opens the cases
INACTIVE_CASE = "The person's account is inactive"    # words of the placeholder's case a

# The values that stood in text/model_text.yaml when the calibration of 2026-11-17 was run (written by
# Qwen3.5-9B, run seeds 1 and 2), copied here as test data: the five places that were unclear, a neighbour
# of them that was clear, and the visible names that the checks read. Each of the five places must be
# refused by the check that was added for it (C1 to C5 in accessdesk/drafting.py).
PRESENT_PARAGRAPHS = (
    (
        "You work at the IT access desk of a company that manages people and roles. Each conversation "
        "you handle is exactly one request from a sender. The sender asks that a specific person from "
        "a specific department receives a specific role. You must settle this request using your tools "
        "without asking the sender any questions."
    ),
    (
        "You must follow two strict rules. First, each of your messages contains exactly one tool call "
        "and nothing else. Second, you must never make up an id or guess an id. You may only use an id "
        "if a tool result shows it to you."
    ),
    (
        "Always follow these steps in this exact order. Step one is to find the person by searching "
        "for their full name from the request. If several people have the same name, you must use the "
        "department from the request to decide which one is correct. Step two is to look up the "
        "requested role for that person by searching for the role name. This step returns the role id, "
        "whether the role is sensitive, and whether the person already holds the role."
    ),
    (
        "You then check four cases in this order and act on the first case that fits. Case a applies "
        "if the person's account state is inactive. In this case, do not give the role and do not ask "
        "for approval. Close the request with the resolution value request_denied. Case b applies if "
        "the person already holds the role. In this case, do not give the role and do not ask for "
        "approval. Close the request with the resolution value role_already_held. Case c applies if "
        "the role is sensitive. In this case, ask for approval using the manager id found in the step "
        "one result. Do not give the role yet. Close the request with the resolution value "
        "request_closing. Case d applies if none of the previous cases fit. In this case, give the "
        "role to the person using the grant_role tool. Close the request with the resolution value "
        "role_granted."
    ),
    (
        "Your last tool call must always be close_request. This call requires the resolution value, "
        "the person id, and the role id. Never give a role twice in one conversation and never request "
        "approval twice. Use the tool names lookup_person, check_access, request_approval, grant_role, "
        "and close_request exactly as written."
    ),
)
PRESENT_SHIFT_CASES = (
    "You then check four cases in this order and act on the first case that fits. Case a applies if "
    "the person's account state is inactive. In this case, do not give the role and do not ask for "
    "approval. Close the request with the resolution value request_denied. Case b applies if the "
    "person already holds the role. In this case, do not give the role and do not ask for approval. "
    "Close the request with the resolution value role_already_held. Case c applies if the person's "
    "account is in use and the person does not yet hold the role. In this case, ask for approval using "
    "the manager id found in the step one result. Do not give the role yet. Close the request with the "
    "resolution value request_closing. Case d applies if none of the previous cases fit. In this case, "
    "do not give the role directly. Instead, close the request with the resolution value "
    "request_closing to indicate the action is pending."
)
PRESENT_RESOLUTION_DESCRIPTION = (
    "The resolution parameter accepts one of four fixed values indicating how the request ended."
)
PRESENT_LOOKUP_DESCRIPTION = (
    "The lookup_person tool takes a person_full_name and returns a list of people. Each person in the "
    "list includes person_id, person_full_name, department_name, account_state, and manager_id."
)
PRESENT_STYLE_3 = [
    "hey, could you get {person_name} the {role_name} access in {department}? thanks",
    "hi, please assign {role_name} to {person_name} from {department} as soon as possible",
    "just a quick ask, can we give {person_name} in {department} the {role_name} role",
]
PRESENT_NAMES = {
    "tools.department_info.name": "count_department",
    "tools.lookup_person.params.name.name": "person_full_name",
    "tools.check_access.params.person_id.name": "person_id",
    "tools.check_access.params.role_name.name": "role_name",
    "tools.request_approval.params.person_id.name": "person_id",
    "tools.request_approval.params.role_id.name": "role_id",
    "tools.request_approval.params.approver_id.name": "approver_id",
    "tools.grant_role.params.person_id.name": "person_id",
    "tools.grant_role.params.role_id.name": "role_id",
    "tools.close_request.params.resolution.name": "request_resolution",
    "tools.close_request.params.person_id.name": "person_id",
    "tools.close_request.params.role_id.name": "role_id",
    "tools.department_info.params.department.name": "department_name", "result_keys.matches": "people",
    "result_keys.person_id": "person_id", "result_keys.name": "person_full_name",
    "result_keys.department": "department_name", "result_keys.status": "account_state",
    "result_keys.manager_id": "manager_id", "result_keys.role_id": "role_id",
    "result_keys.sensitive": "is_sensitive", "result_keys.has_access": "holds_role",
    "result_keys.result": "success_message", "result_keys.error": "error_in_request",
    "result_keys.member_count": "department_people_count", "values.status.active": "active",
    "values.status.inactive": "inactive", "values.resolution.granted": "role_granted",
    "values.resolution.pending": "request_closing", "values.resolution.already": "role_already_held",
    "values.resolution.denied": "request_denied",
}
PRESENT_MAIN = "\n\n".join(PRESENT_PARAGRAPHS)
PRESENT_SHIFT = "\n\n".join(PRESENT_PARAGRAPHS[:3] + (PRESENT_SHIFT_CASES,) + PRESENT_PARAGRAPHS[4:])
# Small repairs of those two prompts: name -> (old words, new words). Each one mends one thing that a check
# wants, so a test can show that a prompt passes with all of them and is refused without any one of them.
REPAIRS = {
    "tools": [("find the person by searching", "find the person with lookup_person by searching"),
              ("look up the requested role", "call check_access for the requested role"),
              ("ask for approval using", "ask for approval with request_approval using")],
    "step lines": [(" Step one", "\nStep one"), (" Step two", "\nStep two")],
    "case lines": [(" Case ", "\nCase ")],
    "cases c and d": [(" Case c", "\nCase c"), (" Case d", "\nCase d")],
    "no list": [(" Use the tool names lookup_person, check_access, request_approval, grant_role, and "
                 "close_request exactly as written.", "")],
    "returned list": [("If several people have the same name", "If the returned list holds several people")],
    "never called": [("Your last tool call must",
                      "Never call grant_role at this desk. Your last tool call must")],
    "never called in case c": [("Do not give the role yet.",
                                "Do not give the role. Never call grant_role, and never use role_granted.")],
    "close in cases": [("Close the request with", "Call close_request with")],
    "never twice": [("Never give a role twice", "Never call grant_role twice")],
    "three cases": [("four cases", "three cases")],
    "no case d": [("\nCase d applies if none of the previous cases fit. In this case, do not give the role "
                   "directly. Instead, close the request with the resolution value request_closing to "
                   "indicate the action is pending.", "")],
    "named again": [("and never request approval twice.",
                     "and never request approval twice. The tool grant_role stays in your list.")],
    # check C7: case a holds the result key of the account state and says where it is read
    "account source": [("the person's account state is inactive",
                        "the account_state in the step one result is inactive")],
    # the checks of the third calibration (C12 to C20): what the first prompts lacked for them
    "every request": [("This step returns", "Do this for every request. This step returns")],       # C16
    "held key": [("Case b applies if the person already holds the role",
                  "Case b applies if holds_role is true")],                                         # C15
    "sensitive key": [("Case c applies if the role is sensitive", "Case c applies if is_sensitive is true")],
    "two messages c": [("Do not give the role yet.", "Do not give the role yet. Then send a new message.")],
    "two messages d": [("using the grant_role tool.", "using the grant_role tool. Then send a new message.")],
    "no second giving": [("Never give a role twice in one conversation and never request approval twice.",
                          "Never request approval twice.")],                                       # C18 (i)
    "account word": [("the person's account is in use", "the person's account_state is active")],  # C18 (ii)
}
# The values of round 2 (six fields written again on 2026-11-17, run seed 3), which the second calibration
# used. Read only. Each of the five faults that it exposed must be refused by its check (C6 to C10).
ROUND2_PATH = ROOT / "results" / "drafting" / "model_text_round2_sha711a1a96.yaml"
ROUND2_DATA = yaml.safe_load(ROUND2_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ins() -> drafting.Instructions:
    return drafting.load_instructions()


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("draft_text", ROOT / "scripts" / "gpu" / "draft_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def good_values(ins: drafting.Instructions) -> dict:
    """A valid value for every field that is asked from the model: the placeholder's value.

    In the three prompts the placeholder's tool names are replaced by the five fixed names, because a prompt
    must name the tools as the instruction file fixes them. The two rule prompts get the markers of the
    line layout (`with_markers`) and are mended in five places (`mended`), because the placeholder file
    itself fails the check of the marker lines and checks C2, C6, C7, C15 and C16: see
    `test_the_new_checks_over_the_placeholder_text`. Its prompt of the second desk is mended in a sixth
    place, for check C21: its case c says "for a sensitive role and for any other role alike". Its plain
    prompt has 41 words and is mended for check C19 (at most 40).
    """
    renames = {PLACEHOLDER["tools"][tool]["name"]: ins.fields[f"tools.{tool}.name"].value
               for tool in TOOL_KEYS}
    values: dict = {}
    for key, field in ins.fields.items():
        if not field.asked:
            continue
        node = PLACEHOLDER
        for part in key.split("."):
            node = node[part]
        if key.endswith("system_prompt"):
            for old, new in renames.items():
                node = node.replace(old, new)
            node = node.strip()
        if key in ("system_prompt", "policy_shift.system_prompt"):
            node = mended(with_markers(node), "C2", "C6", "C7", "C15", "C16")
        if key == "policy_shift.system_prompt":
            node = mended(node, "C21")
        if key == "plain_system_prompt":
            node = mended(node, "C19")
        values[key] = node
    return values


def with_markers(prompt: str) -> str:
    """A rule prompt of the placeholder with the markers of the line layout: at the start of a line "1."
    becomes "Step 1." and "a." becomes "Case a."."""
    prompt = re.sub(r"(?m)^(\d)\. ", r"Step \1. ", prompt)
    return re.sub(r"(?m)^   ([a-d])\. ", r"   Case \1. ", prompt)


# What each new prompt check wants from a rule prompt of the placeholder: check -> (old words, new words).
# C2: the step that finds the person speaks of the returned list. C6: the line above the cases says that
# they are tested in order. C7: case a holds the result key of the account state and names step 1.
MENDS = {
    "C2": (WHICH_PERSON, WHICH_PERSON.replace("person", "person in the returned list")),
    "C6": (FIRST_CASE, "Then test the cases in this order and take the first case that applies"),
    "C7": (INACTIVE_CASE, f"The {PLACEHOLDER['result_keys']['status']} in the result of step 1 is inactive"),
    # the checks of the third calibration. C15: case b holds the result key for a held role. C16: step 2
    # says that it is always done. C19: the plain prompt has at most 40 words (the placeholder has 41).
    "C15": ("The person already has the role", f"{PLACEHOLDER['result_keys']['has_access']} is true"),
    "C16": ("Check the requested role for that person", "Always check the requested role for that person"),
    # Its start is cut to 30 words in all, so that the tests which add one short sentence stay under 40.
    "C19": ("You work at the IT access desk of a company. Each conversation is one request: a person needs "
            "a role.\nResolve the request with the tools.",
            "You work at an IT access desk. Each conversation is one request for a role."),
    # the check of text round 4. C21: case c of the second desk does not speak of a sensitive role. The
    # placeholder's words are true at that desk, and the check refuses them all the same.
    "C21": ("In every other case, for a sensitive role and for any other role alike:", "In every other case:"),
}


def mended(prompt: str, *checks: str) -> str:
    """A rule prompt of the placeholder with the mends for the named checks. Each finds its place once."""
    for check in checks:
        old, new = MENDS[check]
        assert prompt.count(old) == 1, (check, old)
        prompt = prompt.replace(old, new)
    return prompt


class FakeWriter:
    """A writer that answers every field with its good value, except where `replies` says otherwise.

    `replies` maps (field key, candidate index) to a reply (a string, a list of lines, an exception to
    raise, or a function of the messages that gives one of these). `calls` records every (field key,
    candidate index) that was asked for.
    """

    def __init__(self, ins: drafting.Instructions, replies: dict | None = None, run_seed: int = 1) -> None:
        self.by_seed = {candidate_seed(run_seed, key, i): (key, i) for key in ins.fields for i in range(N)}
        self.good = good_values(ins)
        self.replies = replies or {}
        self.calls: list[tuple[str, int]] = []

    def __call__(self, messages: list[dict], seed: int, max_new_tokens: int) -> str:
        key, index = self.by_seed[seed]
        self.calls.append((key, index))
        reply = self.replies.get((key, index), self.good[key])
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        return "\n".join(reply) if isinstance(reply, list) else reply


def draft(ins: drafting.Instructions, replies: dict | None = None, **options):
    writer = FakeWriter(ins, replies)
    lines: list[dict] = []
    result = drafting.draft_all(ins, writer, writer_id="fake@0", run_seed=1, n_candidates=N,
                                emit=lines.append, **options)
    return result, lines, writer


def candidates(lines: list[dict], key: str) -> list[dict]:
    return [line for line in lines if line["kind"] == "candidate" and line["field"] == key]


def file_content(ins: drafting.Instructions, chosen: dict, lines: list[dict] | None = None) -> str:
    """The text file as the script writes it. `lines`: the log that the header reports on (default: empty)."""
    header = drafting.header_comment(ins, model_id=MODEL, revision=REVISION,
                                     written_at_utc="2026-11-17 10:00:00 UTC", run_seed=1, n_candidates=N,
                                     sampling="thinking off",
                                     instructions_file="text/drafting_instructions.yaml",
                                     log_file="text/drafting_log.jsonl",
                                     report=drafting.log_report(ins, lines or []))
    return header + "\n" + drafting.render_yaml(drafting.assemble(chosen, f"{MODEL} (revision {REVISION})"))


# ---------------------------------------------------------------------------
# The instruction file
# ---------------------------------------------------------------------------
def test_the_instruction_file_covers_exactly_the_fields_of_the_text_file(ins):
    strings, lists = drafting.text_fields()
    assert sorted(ins.fields) == sorted(strings + lists)
    # text_fields is built from the public constants of text.py; it must agree with text.py's own lists
    dotted = [".".join(path) for path in text_module._string_paths()]
    assert sorted(strings + ["meta.written_by"]) == sorted(dotted)
    assert lists == [".".join(path) for path in text_module._list_paths()]


def test_the_identifier_pattern_is_the_one_of_text_py():
    assert drafting.IDENTIFIER.pattern == text_module._IDENTIFIER.pattern


def test_the_five_tool_names_are_fixed_to_the_brief_list_and_say_so(ins):
    for tool in TOOL_KEYS:
        field = ins.fields[f"tools.{tool}.name"]
        assert (field.kind, field.value) == ("fixed", tool)
        assert "brief" in field.source and "not asked" in field.source
    assert ins.fields["tools.department_info.name"].asked
    assert ins.fields["policy_shift.variant"].value == "all_grants_need_approval"


def test_what_is_asked_and_how_much(ins):
    asked = [key for key, field in ins.fields.items() if field.asked]
    copies = [key for key, field in ins.fields.items() if field.kind == "copy"]
    assert len(asked) == 64 and len(copies) == 5 and len(ins.fields) == 75
    assert all(ins.fields[key].want == 3 for key in ins.fields if key.startswith("request_templates."))
    assert (ins.fields["departments"].want, ins.fields["roles"].want) == (8, 14)
    assert [ins.fields[key].kind for key in ("system_prompt", "policy_shift.system_prompt",
                                             "plain_system_prompt")] == ["text"] * 3
    for key in copies:                                     # a copy is the same parameter in another tool
        assert key.split(".")[3:] == ins.fields[key].source.split(".")[3:]


def test_no_field_depends_on_a_field_that_is_written_later(ins):
    earlier: set[str] = set()
    for key, field in ins.fields.items():
        assert set(field.depends_on()) <= earlier, key
        earlier.add(key)


def test_a_prompt_can_be_built_from_earlier_values_only(ins):
    values = drafting.standin_values(ins)
    chosen: dict = {}
    n_shown = 0
    for key, field in ins.fields.items():
        if field.asked:
            user = drafting.build_messages(ins, key, chosen)[1]["content"]      # no KeyError: all are earlier
            assert all(str(chosen[name] if isinstance(chosen[name], str) else chosen[name][0]) in user
                       for name in field.uses)
            n_shown += len(field.uses)
        chosen[key] = values[key]
    assert n_shown > 150                                                        # the check above is not empty
    with pytest.raises(KeyError):                                             # a value that is not there yet
        drafting.build_messages(ins, "system_prompt", {})


def test_the_user_message_is_desk_then_instruction_then_earlier_values(ins):
    values = drafting.standin_values(ins)
    system, user = drafting.build_messages(ins, "policy_shift.system_prompt", values)
    assert system == {"role": "system", "content": ins.system_message}
    assert "requested text only" in ins.system_message
    content = user["content"]
    field = ins.fields["policy_shift.system_prompt"]
    assert content.startswith(ins.desk)
    assert content.index(ins.desk) < content.index(ins.task_heading) < content.index(field.instruction) \
        < content.index(ins.earlier_heading) < content.index("<<system_prompt>>")
    assert "- name of the tool that finds people by their full name: lookup_person" in content
    # a value of several lines stands between two lines of dashes; a list has one item per line
    shown = drafting.build_messages(ins, "policy_shift.system_prompt", {**values, "system_prompt": "a\nb"})
    assert "----------\na\nb\n----------" in shown[1]["content"]
    listed = drafting.build_messages(ins, "request_templates.style_4", values)[1]["content"]
    assert "    <<request_templates.style_1, item 1>>\n    <<request_templates.style_1, item 2>>" in listed


def changed_instructions(tmp_path: Path, change) -> Path:
    data = yaml.safe_load(drafting.INSTRUCTIONS_PATH.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "instructions.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_an_unchanged_copy_of_the_instruction_file_loads(tmp_path, ins):
    copy = drafting.load_instructions(changed_instructions(tmp_path, lambda data: None))
    assert copy.fields == ins.fields


@pytest.mark.parametrize("label, change, wanted", [
    ("uses a later field", lambda d: d["fields"]["id_prefixes.person"].update(uses=["id_prefixes.role"]),
     "id_prefixes.person uses 'id_prefixes.role'"),
    ("uses an unknown name", lambda d: d["fields"]["roles"].update(uses=["no_such_group"]), "no_such_group"),
    ("a field is missing", lambda d: d["fields"].pop("errors.malformed_call"),
     "errors.malformed_call has no entry"),
    ("an unknown field",
     lambda d: d["fields"].update({"errors.other": {"label": "x", "fixed": "y", "source": "z"}}),
     "errors.other is not a field"),
    ("a tool name is asked", lambda d: d["fields"]["tools.grant_role.name"].update(
        {"ask": "line", "instruction": "x"}) or d["fields"]["tools.grant_role.name"].pop("fixed"),
     "tools.grant_role.name must be fixed"),
    ("too few roles", lambda d: d["fields"]["roles"].update(want=11), "roles: want must be at least 12"),
    ("a list asked as a line", lambda d: d["fields"]["departments"].update(ask="line"),
     "departments: ask must be"),
    ("an unknown form", lambda d: d["fields"]["roles"].update(form="no_such_form"),
     "roles: ask must be one of"),
    ("no instruction", lambda d: d["fields"]["roles"].update(instruction=" "), "roles: ask must be one of"),
    ("no label", lambda d: d["fields"]["roles"].pop("label"), "roles has no label"),
    ("a copy of another parameter", lambda d: d["fields"]["tools.grant_role.params.role_id.name"].update(
        same_as="tools.check_access.params.person_id.name"), "same_as must name"),
    ("the prompt before the values it must hold", lambda d: d["fields"].update(
        {"values.resolution.denied": d["fields"].pop("values.resolution.denied")}),
     "- system_prompt needs ['values.resolution.denied']"),
    ("an upper-case word in the list", lambda d: d["sensitivity_words"].append("Admin"), "sensitivity_words"),
])
def test_a_wrong_instruction_file_is_refused_with_the_reason(tmp_path, label, change, wanted):
    with pytest.raises(drafting.InstructionError) as error:
        drafting.load_instructions(changed_instructions(tmp_path, change))
    assert wanted in str(error.value), label


def test_a_key_written_twice_in_the_instruction_file_is_refused(tmp_path):
    raw = drafting.INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    path = tmp_path / "twice.yaml"
    path.write_text(raw + "\n  roles:\n    label: x\n    fixed: y\n    source: z\n", encoding="utf-8")
    with pytest.raises(drafting.InstructionError, match="twice"):
        drafting.load_instructions(path)


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_{}']+", text.casefold())


def runs_of(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def test_no_instruction_shares_a_run_of_eight_words_with_the_placeholder(ins):
    placeholder = runs_of(words_of(placeholder_path().read_text(encoding="utf-8")), 8)
    assert len(placeholder) > 500
    # everything that the writing model can be shown: every message, and every label
    shown = [drafting.dry_run_text(ins, 1)] + [field.label for field in ins.fields.values()]
    shown += [ins.system_message, ins.desk, ins.task_heading, ins.earlier_heading]
    for part in shown:
        assert runs_of(words_of(part), 8) & placeholder == set()
    # the check can fail: a sentence of the placeholder is found
    copied = PLACEHOLDER["tools"]["lookup_person"]["description"]
    assert runs_of(words_of("Write this: " + copied), 8) & placeholder


def test_the_instruction_file_holds_no_line_of_the_placeholder(ins):
    """No instruction quotes the placeholder: none of its lines (of five words or more) is in the file."""
    raw = " ".join(drafting.INSTRUCTIONS_PATH.read_text(encoding="utf-8").split()).casefold()
    lines: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            lines.extend(" ".join(line.split()).casefold() for line in node.splitlines())
    walk({key: value for key, value in PLACEHOLDER.items() if key != "meta"})
    sentences = [line for line in lines if len(line.split()) >= 5]
    assert len(sentences) > 40
    assert [line for line in sentences if line in raw] == []


def test_the_role_instruction_shows_every_word_of_the_sensitivity_list(ins):
    assert len(ins.sensitivity_words) >= 20
    instruction = ins.fields["roles"].instruction
    assert all(drafting.mentions(instruction, word) for word in ins.sensitivity_words)


def test_every_template_instruction_names_the_three_variables(ins):
    for style in (1, 2, 3, 4):
        instruction = ins.fields[f"request_templates.style_{style}"].instruction
        for name in ("person_name", "department", "role_name"):
            assert instruction.count("{" + name + "}") == 1
    held_out = ins.fields["request_templates.style_4"]
    assert held_out.uses == tuple(f"request_templates.style_{style}" for style in (1, 2, 3))


def test_the_prompt_instructions_state_the_rules_and_use_the_chosen_values(ins):
    main = ins.fields["system_prompt"]
    assert {f"tools.{tool}.name" for tool in TOOL_KEYS} <= set(main.uses)
    assert {f"values.resolution.{key}" for key in ("granted", "pending", "already", "denied")} <= set(
        main.uses)
    assert "tools.department_info.name" not in main.uses
    for needed in ("= 1 tool call, exactly", "switched off", "holds role already", "sensitive", "manager",
                   "shown by a tool result: yes; any other: no; made up or guessed: never"):
        assert needed in main.instruction, needed
    # the numbered structure: situation, two working rules, the steps, the four cases in order, the end
    marks = ("(1) SITUATION", "(2) TWO WORKING RULES", "(3) STEPS", "step 1:", "step 2:", "step 3:",
             "a. IF account of person: switched off", "b. IF person: holds role already",
             "c. IF role: sensitive", "d. IF a, b, c: none fits", "(4) END")
    order = [main.instruction.index(mark) for mark in marks]
    assert order == sorted(order)
    # each case closes with the resolution that `rules.decide` gives for it (default rule set)
    cases = main.instruction[main.instruction.index("a. IF"):main.instruction.index("(4) END")]
    assert [found for found in re.findall(r"the\s+one\s+meaning\s+([A-Z ]+[A-Z])", cases)] == [
        "REFUSED", "HELD BEFORE", "WAITS FOR APPROVAL", "GIVEN"]
    for tag, key in (("REFUSED", "denied"), ("HELD BEFORE", "already"), ("WAITS FOR APPROVAL", "pending"),
                     ("GIVEN", "granted")):                        # the label of the value carries the tag
        assert f"meaning {tag} (" in ins.fields[f"values.resolution.{key}"].label
    shift = ins.fields["policy_shift.system_prompt"]
    assert "system_prompt" in shift.uses and "gives a role directly: never" in shift.instruction
    assert ins.fields["plain_system_prompt"].uses == ()
    # the three prompt instructions say that the notes are not the wording
    for key in ("system_prompt", "policy_shift.system_prompt", "plain_system_prompt"):
        assert "not the wording" in " ".join(ins.fields[key].instruction.split()), key
    for key in ("system_prompt", "policy_shift.system_prompt"):
        assert "They are not its wording: write every sentence yourself" in " ".join(
            ins.fields[key].instruction.split())


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def test_seeds_differ_between_fields_and_candidates_and_are_stable(ins):
    seeds = {(key, i): candidate_seed(1, key, i) for key in ins.fields for i in range(N)}
    assert len(set(seeds.values())) == len(seeds)
    assert candidate_seed(1, "system_prompt", 0) == 1472705299217925469       # a fixed number: must not move
    assert candidate_seed(1, "system_prompt", 0) != candidate_seed(2, "system_prompt", 0)
    _, lines, _ = draft(ins)
    for line in lines:
        if line["kind"] == "candidate":
            assert line["seed"] == seeds[(line["field"], line["index"])]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, raw, cleaned", [
    ("line", "  role_id \n", "role_id"),
    ("line", '"role_id"', "role_id"),
    ("line", "'role_id'", "role_id"),
    ("line", "`role_id`", "role_id"),
    ("line", "\u201crole_id\u201d", "role_id"),
    ("line", "```\nrole_id\n```", "role_id"),
    ("line", "```text\nrole_id\n```", "role_id"),
    ("line", "```role_id```", "role_id"),
    ("line", "\n\nrole_id\n\nI chose this because it is short.", "role_id"),
    ("line", "role_id\r\nsecond line", "role_id"),
    ("line", '"No person has the id {value}."\n\nNote: done.', "No person has the id {value}."),
    ("line", "It's the person's id", "It's the person's id"),
    ("line", "'it's'", "'it's'"),                                  # an apostrophe inside: not a pair
    ("line", '"Use "a" and "b"."', 'Use "a" and "b".'),            # balanced marks inside: the pair goes
    ("line", '"', '"'),
    ("line", "", ""),
    ("line", "   \n  ", ""),
    ("text", "  You work here.  \nSecond line.\t\n\nThird.\n", "You work here.\nSecond line.\n\nThird."),
    ("text", '"You work here.\nSecond line."', "You work here.\nSecond line."),
    ("text", "```\nYou work here.\nSecond line.\n```", "You work here.\nSecond line."),
    ("list", "Finance\n\n  Sales \n", ["Finance", "Sales"]),
    ("list", "- Finance\n* Sales\n\u2022 Legal\n1. Marketing\n2) Operations\n12. Support",
     ["Finance", "Sales", "Legal", "Marketing", "Operations", "Support"]),
    ("list", "Two - Words\nStep 1. Go", ["Two - Words", "Step 1. Go"]),        # a marker only at the start
    ("list", '"Finance"\n\'Sales\'', ["Finance", "Sales"]),
    ("list", "```\nFinance\nSales\n```", ["Finance", "Sales"]),
    ("list", "Self-Service\n-\n", ["Self-Service", "-"]),         # a hyphen in a word is not a list marker
])
def test_cleaning(kind, raw, cleaned):
    assert clean_reply(kind, raw) == cleaned


def test_cleaning_refuses_an_unknown_kind():
    with pytest.raises(ValueError):
        clean_reply("fixed", "x")


def test_cleaning_a_long_reply_is_fast():
    """A reply is not trusted text: none of the cleaning steps may take more than linear time."""
    for kind in drafting.ASK_KINDS:
        clean_reply(kind, "- " * 100_000 + "\n" + "1." * 100_000 + '"' * 100_001)


# ---------------------------------------------------------------------------
# A writer that always gives valid text
# ---------------------------------------------------------------------------
def test_a_writer_that_always_gives_valid_text_fills_the_whole_file(ins, tmp_path):
    result, lines, writer = draft(ins)
    assert result.complete and result.failed == {}
    assert sorted(result.chosen) == sorted(ins.fields)
    asked = [key for key, field in ins.fields.items() if field.asked]
    assert writer.calls == [(key, 0) for key in asked]            # one candidate each, in the file's order
    assert {key: result.chosen[key] for key in asked} == writer.good
    for key, field in ins.fields.items():                           # fixed values and copies
        if field.kind == "fixed":
            assert result.chosen[key] == field.value
        if field.kind == "copy":
            assert result.chosen[key] == result.chosen[field.source]

    out = tmp_path / "model_text.yaml"
    assert drafting.write_text_file(out, file_content(ins, result.chosen)) == []
    assert check_text(out) == []
    bundle = load_text(out)
    assert not bundle.is_placeholder
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["meta"] == {
        "written_by": f"{MODEL} (revision {REVISION})", "placeholder": False}
    assert bundle.system_prompt == result.chosen["system_prompt"]
    assert bundle.policy_shift_system_prompt == result.chosen["policy_shift.system_prompt"]
    assert bundle.plain_system_prompt == result.chosen["plain_system_prompt"]
    assert bundle.policy_shift_variant == "all_grants_need_approval"
    assert [spec.name for spec in bundle.tool_specs("six")] == list(TOOL_KEYS) + [
        PLACEHOLDER["tools"]["department_info"]["name"]]
    assert [p.name for p in bundle.tools["close_request"].params] == ["outcome", "person_id", "role_id"]
    assert bundle.request_templates[4] == tuple(result.chosen["request_templates.style_4"])
    assert bundle.departments == tuple(result.chosen["departments"])
    assert bundle.roles == tuple(result.chosen["roles"]) and len(bundle.roles) == 14
    assert bundle.errors["malformed_call"] == result.chosen["errors.malformed_call"]
    assert bundle.resolution == {key: result.chosen[f"values.resolution.{key}"] for key in bundle.resolution}
    assert not (tmp_path / "model_text.yaml.checking").exists()


def test_the_written_yaml_gives_back_every_value_exactly(ins):
    """Values that YAML could misread (a word like `no`, a colon, quotation marks, several lines) survive."""
    chosen = dict(draft(ins)[0].chosen)
    chosen.update({"values.status.active": "yes", "values.status.inactive": "no", "id_prefixes.person": "n",
                   "errors.unknown_role_id": 'Role: "{value}" # not found',
                   "plain_system_prompt": "First line: here.\n\n  Indented line.\nLast - line.",
                   "departments": ["Finance", "No", "On-Site Support"] + chosen["departments"][3:]})
    data = yaml.safe_load(file_content(ins, chosen))
    assert data == drafting.assemble(chosen, f"{MODEL} (revision {REVISION})")
    assert data["values"]["status"] == {"active": "yes", "inactive": "no"}
    assert list(data) == ["meta", "system_prompt", "plain_system_prompt", "policy_shift", "id_prefixes",
                          "tools", "result_keys", "values", "errors", "request_templates", "departments",
                          "roles"]
    assert "system_prompt: |" in file_content(ins, chosen)         # a prompt is written as a readable block


def test_the_header_comment_holds_every_required_statement(ins):
    content = file_content(ins, draft(ins)[0].chosen)
    header = [line for line in content.split("\n\n")[0].splitlines()]
    assert all(line.startswith("#") for line in header) and len(header) >= 12
    flat = " ".join(" ".join(line.lstrip("# ") for line in header).split())
    for statement in (
        f"Written by: {MODEL}, revision {REVISION}",
        "Date and time (UTC): 2026-11-17 10:00:00 UTC",
        "Run seed: 1",
        f"SHA-256 {ins.sha256}",
        "Instructions: written by the coding assistant (a closed API model). They are in "
        "text/drafting_instructions.yaml",
        "the names of the five tools follow the brief's example list (lookup_person, check_access, "
        "request_approval, grant_role, close_request)",
        "Person names: they are not in this file. They come from the public lists in data/names/.",
        "Nobody chose between candidates: no person and no closed model.",
        "Every candidate is in text/drafting_log.jsonl",
        "Line layout: in the two system prompts that tell the rules, the program starts a new line in front "
        "of each sentence that begins with the word Step and a number, with the word Case and a letter, or "
        "with the word Finally. This changes no word: only white space becomes a line break.",
        "Read and approved by: (not yet)",
    ):
        assert statement in flat, statement
    assert len(ins.sha256) == 64
    assert ins.sha256 == __import__("hashlib").sha256(drafting.INSTRUCTIONS_PATH.read_bytes()).hexdigest()


def test_a_text_that_check_text_refuses_is_not_written(ins, tmp_path):
    chosen = dict(draft(ins)[0].chosen)
    chosen["result_keys.error"] = chosen["result_keys.result"]        # two result keys are equal
    out = tmp_path / "model_text.yaml"
    problems = drafting.write_text_file(out, file_content(ins, chosen))
    assert any("result keys must be unique" in problem for problem in problems)
    assert list(tmp_path.iterdir()) == []
    out.write_text("old", encoding="utf-8")                           # an existing file is left as it is
    assert drafting.write_text_file(out, file_content(ins, chosen)) != []
    assert out.read_text(encoding="utf-8") == "old"


# ---------------------------------------------------------------------------
# Selection: the first valid candidate, in index order
# ---------------------------------------------------------------------------
def test_the_first_valid_candidate_is_taken_and_no_later_one_is_asked_for(ins):
    key = "values.confirmation.role_granted"
    replies = {(key, 0): "", (key, 1): "", (key, 2): "first valid text", (key, 3): "second valid text"}
    result, lines, writer = draft(ins, replies)
    assert result.chosen[key] == "first valid text"
    assert [call for call in writer.calls if call[0] == key] == [(key, 0), (key, 1), (key, 2)]
    assert [(line["index"], line["valid"], line["taken"]) for line in candidates(lines, key)] == [
        (0, False, False), (1, False, False), (2, True, True)]


def test_a_later_field_is_given_the_value_that_was_taken(ins):
    key = "tools.department_info.name"
    result, lines, _ = draft(ins, {(key, 0): "2bad", (key, 1): "count_department_people"})
    assert result.chosen[key] == "count_department_people"
    later = candidates(lines, "tools.department_info.description")[0]
    user = later["messages"][1]["content"]
    assert "which counts the people of a department: count_department_people" in user
    assert "2bad" not in user


@pytest.fixture(scope="module")
def good(ins) -> dict:
    return good_values(ins)


def just_over_a_third(main: str) -> str:
    """The start of `main` with one character more than a third of it; it does not end in white space,
    which the cleaning of a reply would remove."""
    n = len(main) // 3 + 1
    while main[n - 1].isspace():
        n += 1
    return main[:n]


def bad_prompt(kind: str):
    """A function good values -> an invalid reply, for the cases that need the good prompt."""
    def make(good: dict) -> str:
        main, shift, plain = (good[key] for key in ("system_prompt", "policy_shift.system_prompt",
                                                    "plain_system_prompt"))
        return {
            "short": "Use lookup_person, check_access, request_approval, grant_role and close_request.",
            "long": main + " word" * 400,
            "no grant tool": main.replace("grant_role", "the granting tool"),
            "no already value": main.replace("already_has_role", "has it"),
            "extra tool": main + "\nNever call Department_Summary.",
            "equal": main,
            "shift without approval tool": shift.replace("request_approval", "the approval tool"),
            "shift without pending value": shift.replace("pending_approval", "waiting"),
            "plain too long": just_over_a_third(main),
            "plain with a value": plain + " Denied.",
        }[kind]
    return make


def plain_with(sentence: str):
    """A function good values -> the good plain prompt with one more sentence (it stays short enough)."""
    def make(good: dict) -> str:
        value = good["plain_system_prompt"] + " " + sentence
        assert 3 * len(value) <= len(good["system_prompt"])       # so the length rule is not what refuses it
        return value
    return make


INVALID = [
    # every field
    ("tools.department_info.name", "", "is empty"),
    ("result_keys.matches", "<think>people", "a string of the chat format"),
    ("errors.unknown_person_id", "No person <|im_end|> has it.", "a string of the chat format"),
    ("errors.unknown_person_id", "No person has this id.<|endoftext|>", "holds '<|', a string of the chat"),
    ("plain_system_prompt", "You work at the desk. |>", "holds '|>', a string of the chat format"),
    # one-line fields
    ("tools.lookup_person.description", "Finds people. " * 15, "one line has at most 200"),
    ("errors.unknown_role_id", "Here is the error text:", "ends with a colon"),
    # id prefixes
    ("id_prefixes.person", "person id", "is not a short id prefix"),
    ("id_prefixes.person", "per__", "is not a short id prefix"),
    ("id_prefixes.person", "7p_", "is not a short id prefix"),
    ("id_prefixes.role", "per", "begins like the other id prefix"),
    ("id_prefixes.role", "PER_X", "is not a short id prefix"),
    ("id_prefixes.role", "Per_", "begins like the other id prefix"),
    # names
    ("tools.department_info.name", "department summary", "is not an identifier"),
    ("tools.department_info.name", "2nd_tool", "is not an identifier"),
    ("tools.department_info.name", "grant_role", "is already the name of another tool"),
    ("tools.department_info.name", "lookup_person_count", "contains the tool name 'lookup_person'"),
    ("tools.department_info.name", "lookup", "contains the tool name 'lookup_person'"),
    ("tools.check_access.params.role_name.name", "role-name", "is not an identifier"),
    ("tools.check_access.params.role_name.name", "person_id", "is already the name of another parameter"),
    ("tools.request_approval.params.role_id.name", "role_name", "is already the name of another parameter"),
    ("tools.request_approval.params.approver_id.name", "role_id", "is already the name of another parameter"),
    ("tools.close_request.params.resolution.name", "person_id", "is already the name of another parameter"),
    # result keys and values
    ("result_keys.name", "full name", "is not an identifier"),
    ("result_keys.name", "people", "is already the value of result_keys.matches"),
    ("values.status.active", "is active", "is not an identifier"),
    ("values.status.inactive", "active", "is already the value of values.status.active"),
    ("values.resolution.granted", "Granted!", "is not an identifier"),
    ("values.resolution.pending", "granted", "is already the value of values.resolution.granted"),
    ("values.resolution.granted", "error", "stands inside result_keys.error"),
    ("values.resolution.denied", "Active", "stands inside values.status.active"),
    ("values.resolution.already", "has", "stands inside result_keys.has_access"),
    ("values.confirmation.role_granted", "approval requested", "is already the value of"),
    # descriptions
    ("tools.grant_role.description", "Gives a role. Call check_access first.",
     "names another tool (check_access)"),
    ("tools.lookup_person.description", "Finds people, like Department_Summary.", "names another tool"),
    ("tools.grant_role.params.role_id.description", "The id that check_access returned.",
     "names another tool"),
    # error texts
    ("errors.unknown_person_id", "No person has the id {person_id}.", "holds braces other than {value}"),
    ("errors.unknown_person_id", "No person has the id {value!r}.", "holds braces other than {value}"),
    ("errors.unknown_person_id", "No person has the id {{value}}.", "holds braces other than {value}"),
    ("errors.unknown_person_id", "No person has the id {value.", "holds braces other than {value}"),
    ("errors.unknown_person_id", "The id {value} is unknown: {value}.", "holds {value} more than once"),
    ("errors.malformed_call", "The call {value} is not valid.", "has no variable"),
    # prompts
    ("system_prompt", bad_prompt("short"), "has 7 words; a system prompt has 80 to 400"),
    ("system_prompt", bad_prompt("long"), "words; a system prompt has 80 to 400"),
    ("system_prompt", lambda good: " ".join(good["system_prompt"].split()[:79]), "has 79 words"),
    ("system_prompt", bad_prompt("no grant tool"), "does not name the tool grant_role"),
    ("system_prompt", bad_prompt("no already value"),
     "does not hold the resolution value 'already_has_role'"),
    ("system_prompt", bad_prompt("extra tool"), "names the extra tool (department_summary)"),
    ("policy_shift.system_prompt", bad_prompt("equal"), "is equal to the main system prompt"),
    ("policy_shift.system_prompt", bad_prompt("short"), "has 7 words"),
    ("policy_shift.system_prompt", bad_prompt("shift without approval tool"),
     "does not name the tool request_approval"),
    ("policy_shift.system_prompt", bad_prompt("shift without pending value"),
     "does not hold the resolution value 'pending_approval'"),
    ("plain_system_prompt", bad_prompt("plain too long"), "more than a third of the main system prompt"),
    ("plain_system_prompt", bad_prompt("plain with a value"), "holds the resolution value 'denied'"),
    ("plain_system_prompt", "", "is empty"),
    # repairs of the independent check (DRAFT-CHECK)
    ("tools.grant_role.description", "```\nGives a role to a person.\n```\nThis says what the tool does.",
     "the rest of a code fence"),                     # the fence is not closed at the end: "```" was taken
    ("tools.grant_role.description", "```text\nGives a role to a person.", "the rest of a code fence"),
    ("system_prompt", lambda good: "```\n" + good["system_prompt"] + "\n```\nThe prompt follows the notes.",
     "the rest of a code fence"),
    ("system_prompt", lambda good: "Here is the system prompt:\n\n" + good["system_prompt"],
     "its first line ends with a colon"),
    ("system_prompt", lambda good: good["system_prompt"] + "\nNever call Department_Summary_Tool.",
     "names the extra tool (department_summary)"),
    ("policy_shift.system_prompt", lambda good: good["system_prompt"].replace(". ", ".  ") + " ",
     "is equal to the main system prompt"),
    ("policy_shift.system_prompt", lambda good: good["system_prompt"].replace("\n", "\n\n"),
     "is equal to the main system prompt"),
    # the checks added after the second independent check: a resolution value in an error text ...
    ("errors.unknown_role_id", "The role id {value} was Denied.", "holds the resolution value 'denied'"),
    ("errors.unknown_person_id", "No person {value}: ALREADY_HAS_ROLE.",
     "holds the resolution value 'already_has_role'"),
    ("errors.malformed_call", "Send one tool call. Nothing was granted.",
     "holds the resolution value 'granted'"),
    # ... words of the rules in the plain prompt ...
    ("plain_system_prompt", plain_with("Stay Active."), "holds the status word 'active'"),
    ("plain_system_prompt", plain_with("Skip inactive people."), "holds the status word 'inactive'"),
    ("plain_system_prompt", plain_with("Start with Lookup_Person."), "names the tool lookup_person"),
    ("plain_system_prompt", plain_with("End with close_request."), "names the tool close_request"),
    ("plain_system_prompt", plain_with("Never use department_summary."), "names the tool department_summary"),
    ("plain_system_prompt", plain_with("Some roles are Sensitive."), "begins with 'sensitiv'"),
    ("plain_system_prompt", plain_with("Mind the sensitivity of a role."), "begins with 'sensitiv'"),
    ("plain_system_prompt", plain_with("Ask the person's manager."), "begins with 'manager'"),
    ("plain_system_prompt", plain_with("MANAGERS decide."), "begins with 'manager'"),
    ("plain_system_prompt", plain_with("Get approval first."), "begins with 'approv'"),
    ("plain_system_prompt", plain_with("Somebody must approve."), "begins with 'approv'"),
    ("plain_system_prompt", plain_with("Nothing is pre-Approved."), "begins with 'approv'"),
    ("plain_system_prompt", plain_with("Find the approver."), "begins with 'approv'"),
    ("plain_system_prompt", plain_with("Wait for approving."), "begins with 'approv'"),
    # check C10: what the desk does with a request
    ("plain_system_prompt", plain_with("A request means giving a role."), "begins with 'giv'"),
    ("plain_system_prompt", plain_with("Grant nothing by yourself."), "begins with 'grant'"),
    ("plain_system_prompt", plain_with("Think of the assignment."), "begins with 'assign'"),
    # ... and the arrow of the notes
    ("tools.grant_role.params.person_id.description", "id -> the one who gets the role",
     "holds '->', the arrow of the notes"),
    ("system_prompt", lambda good: good["system_prompt"] + "\nrule: id made up -> never",
     "holds '->', the arrow of the notes"),
    # check C3: the word before "parameter" is the visible name or no name at all; a tool's description
    # holds no parameter name of another tool (here the names are the placeholder's: "outcome", "person_id")
    ("tools.close_request.params.resolution.description", "The resolution parameter takes one of four words.",
     "calls a parameter 'resolution', which is not its visible name (outcome)"),
    ("tools.close_request.params.resolution.description", "The Outcome parameter takes one of four words.",
     "calls a parameter 'Outcome', which is not its visible name (outcome)"),
    ("tools.grant_role.params.person_id.description", "The person-id parameter: who gets the role.",
     "calls a parameter 'id', which is not its visible name (person_id)"),
    ("tools.grant_role.params.person_id.description", "The role_id parameter: who gets the role.",
     "calls a parameter 'role_id', which is not its visible name (person_id)"),
    ("tools.close_request.description", "Closes the request. The resolution parameter says how it ended.",
     "calls a parameter 'resolution', which is not its visible name (outcome, person_id, role_id)"),
    ("tools.grant_role.description", "Gives the role with that role_name to a person now.",
     "names the parameter 'role_name', which this tool does not have"),
    ("tools.lookup_person.description", "Finds people by full_name, never by Approver_Id.",
     "names the parameter 'approver_id', which this tool does not have"),
    # a reply that was cut off by the token limit
    ("system_prompt", ReplyCutOff("You work at"), CUT_OFF_REASON),
    ("result_keys.status", ReplyCutOff("account_status"), CUT_OFF_REASON),
]


@pytest.mark.parametrize("key, reply, reason", INVALID, ids=[f"{k}:{r}"[:60] for k, _, r in INVALID])
def test_an_invalid_first_reply_is_refused_for_the_right_reason_and_the_next_is_taken(ins, good, key, reply,
                                                                                    reason):
    reply = reply(good) if callable(reply) else reply
    result, lines, writer = draft(ins, {(key, 0): reply})
    first, second = candidates(lines, key)
    assert (first["valid"], first["taken"]) == (False, False)
    assert reason in first["reason"], first["reason"]
    assert (second["index"], second["valid"], second["reason"], second["taken"]) == (1, True, None, True)
    assert result.chosen[key] == good[key] and result.complete
    assert first["cut_off"] == isinstance(reply, ReplyCutOff)
    assert first["raw"] == (reply.text if isinstance(reply, ReplyCutOff) else reply)


@pytest.mark.parametrize("key, reply", [
    ("tools.grant_role.description", "grant_role gives a role to a person now and writes a record."),
    ("tools.grant_role.params.role_id.description", "The id of the role that grant_role gives."),
    ("tools.grant_role.description", "Gives the role; the checking tool (check_accesses) is another one."),
    ("errors.unknown_person_id", "No such person."),
    ("values.resolution.granted", "has_role_now"),                 # holds a result key; is not inside one
    ("result_keys.result", "resolution"),
    ("id_prefixes.person", "P-"),
    ("id_prefixes.role", "role_"),
    # near cases of the added checks: a word that only begins like a resolution value or holds one inside
    ("errors.unknown_role_id", "Grant failed: no role has the id {value}."),
    ("errors.unknown_role_id", "No role has the id {value} (approval still pending)."),
    ("errors.malformed_call", "Send one tool call. Undenied and regranted are not words."),
    # near cases for the plain prompt: the words stand inside other words, or are other forms
    ("plain_system_prompt", plain_with("Be proactive and manage each request.")),
    ("plain_system_prompt", plain_with("Look up the person. Close the request.")),
    ("plain_system_prompt", plain_with("Stay sensible; prove nothing, improve nothing.")),
    ("plain_system_prompt", plain_with("Be a good comanager of insensitive work.")),   # inside a word
    ("plain_system_prompt", plain_with("Somebody gave a gift; forgive unassigned misgivings.")),
    ("plain_system_prompt", plain_with("Department summary is no rule.")),
    # near cases of check C3: the visible name, a word that is no name, the name after the word, a plural,
    # and names of other tools' parameters that are result keys too (what a tool returns)
    ("tools.close_request.params.resolution.description", "The outcome parameter takes one of four words."),
    ("tools.close_request.params.resolution.description", "This required parameter says how it all ended."),
    ("tools.close_request.params.resolution.description", "The parameter outcome: how the request ended."),
    ("tools.close_request.description", "Closes the request. Its three parameters are written on it."),
    ("tools.close_request.description", "Closes the request. The outcome parameter says how it ended."),
    ("tools.lookup_person.description", "Finds people by full_name. Each one has a person_id and a role_id."),
    ("tools.grant_role.description", "Gives a role now. Needs the person_id and the role_id."),
    # the arrow is refused only where the instruction has notes with it
    ("request_templates.style_2", ["Role -> {role_name} | For -> {person_name} | Team -> {department}",
                                   "Ticket; who: {person_name}; where: {department}; what: {role_name}",
                                   "ROLE={role_name} / NAME={person_name} / DEPT={department}"]),
])
def test_replies_that_only_look_wrong_are_taken(ins, good, key, reply):
    reply = reply(good) if callable(reply) else reply
    result, lines, _ = draft(ins, {(key, 0): reply})
    assert result.chosen[key] == reply and candidates(lines, key)[0]["taken"]


def test_the_policy_shift_prompt_needs_only_what_its_rule_set_can_use(ins, good):
    assert drafting.ruleset_tools("default") == list(TOOL_KEYS)
    assert drafting.ruleset_resolutions("default") == ["granted", "pending", "already", "denied"]
    assert drafting.ruleset_tools("all_grants_need_approval") == ["lookup_person", "check_access",
                                                                  "request_approval", "close_request"]
    assert drafting.ruleset_resolutions("all_grants_need_approval") == ["pending", "already", "denied"]
    assert drafting.ruleset_tools("sensitive_denied") == ["lookup_person", "check_access", "grant_role",
                                                          "close_request"]
    assert drafting.ruleset_resolutions("sensitive_denied") == ["granted", "already", "denied"]
    # the role-giving tool is not needed as a tool of the desk, and the value for a given role is not needed
    # at all; but since check C5 the prompt must name that tool once, to say that it is not called
    shift = good["policy_shift.system_prompt"]
    assert "granted" not in shift and shift.count("grant_role") == 1
    without_grant = shift.replace("grant_role", "the granting tool")
    result, lines, _ = draft(ins, {("policy_shift.system_prompt", 0): without_grant})
    assert candidates(lines, "policy_shift.system_prompt")[0]["reason"] == (
        "does not say that grant_role is not called at this desk: no sentence names it")
    assert result.chosen["policy_shift.system_prompt"] == shift


def test_a_value_is_checked_against_values_that_were_written_later_too(ins):
    """With --only, a field is written again while later fields keep their values."""
    chosen = dict(draft(ins)[0].chosen)
    del chosen["result_keys.status"]
    assert drafting.check_value(ins, "result_keys.status", "account_status", chosen) is None
    assert "holds the resolution value 'denied'" in drafting.check_value(ins, "result_keys.status",
                                                                         "denied_or_not", chosen)
    assert "is already the value of result_keys.error" in drafting.check_value(ins, "result_keys.status",
                                                                               "error", chosen)
    del chosen["values.resolution.denied"]
    assert "stands inside roles[8]" in drafting.check_value(ins, "values.resolution.denied", "desk", chosen)
    assert "stands inside departments[0]" in drafting.check_value(ins, "values.resolution.denied", "FINANCE",
                                                                  chosen)


def test_a_check_reads_only_the_values_that_it_names(ins, good):
    """`validation_needs` is what keeps the order of writing safe, so it must name every value that a check
    cannot do without: given those values alone, the check of the good value runs and finds nothing."""
    full = draft(ins)[0].chosen
    n_needs = 0
    for key, field in ins.fields.items():
        if field.asked and field.kind != "list":
            needs = drafting.validation_needs(key)
            assert drafting.check_value(ins, key, good[key], {name: full[name] for name in needs}) is None
            n_needs += len(needs)
    for key in ("departments", "roles", "request_templates.style_4", "request_templates.style_2"):
        assert drafting.validation_needs(key) == ()
        assert all(drafting.check_item(ins, key, item, {}, []) is None for item in good[key])
    # 6 tool descriptions with the six tool names, the 12 parameter names and the 12 result keys; 12
    # parameter descriptions with the six tool names and their own parameter name; the three prompts (since
    # check C7 the two rule prompts need the result key of the account state and the two account words,
    # and since checks C12 and C15 the result keys of the manager id, of a sensitive role and of a held
    # role: 16 and 18 values, not 10 and 12); the six error texts with the four resolution values
    assert n_needs == 6 * (6 + 12 + 12) + 12 * (6 + 1) + 16 + 18 + (5 + 2 + 6) + 6 * 4
    for key in drafting.RULE_PROMPTS:
        assert {"result_keys.status", "values.status.active", "values.status.inactive"} <= set(
            drafting.validation_needs(key))
    assert set(drafting.validation_needs("tools.grant_role.description")) >= {
        "tools.check_access.params.role_name.name", "result_keys.role_id"}
    assert drafting.validation_needs("tools.close_request.params.resolution.description")[-1] == (
        "tools.close_request.params.resolution.name")
    assert set(drafting.validation_needs("plain_system_prompt")) >= {
        "values.status.active", "values.status.inactive", "tools.department_info.name",
        "tools.grant_role.name"}
    assert drafting.validation_needs("errors.malformed_call") == drafting.validation_needs(
        "errors.unknown_role_id") == tuple(f"values.resolution.{key}" for key in (
            "granted", "pending", "already", "denied"))
    with pytest.raises(KeyError):                            # and a check does fail without them
        drafting.check_value(ins, "system_prompt", good["system_prompt"], {})


def one_bad_value(ins, chosen: dict, key: str, value) -> tuple[list[str], str | None]:
    """(problems of check_text for the file with this one value, the reason of the field check)."""
    data = drafting.assemble({**chosen, key: value}, "x")
    problems = text_module._problems(yaml.safe_load(drafting.render_yaml(data)))
    others = {name: found for name, found in chosen.items() if name != key}
    if isinstance(value, list):
        reason = drafting.check_item(ins, key, value[-1], others, value[:-1])
    else:
        reason = drafting.check_value(ins, key, value, others)
    return problems, reason


@pytest.mark.parametrize("key, value", [
    ("tools.department_info.name", "lookup_person"),
    ("tools.department_info.name", "department summary"),
    ("tools.close_request.params.resolution.name", "role_id"),
    ("tools.lookup_person.params.name.name", "full-name"),
    ("result_keys.manager_id", "role_id"),
    ("result_keys.manager_id", "manager_denied"),
    ("values.status.inactive", "active"),
    ("values.status.inactive", "access_denied"),
    ("values.resolution.denied", "granted"),
    ("values.resolution.denied", "role"),
    ("values.confirmation.request_closed", "role given"),
    ("errors.unknown_role_name", "No role {name}."),
    ("errors.unknown_role_name", "No role {value:>5}."),
    ("tools.lookup_person.description", "Text with </tool_call> in it."),
    ("system_prompt", "Rules <|im_start|>"),
    ("departments", ["Finance", "Sales", "Legal", "Marketing", "Operations", "Engineering", "finance"]),
    ("departments", ["Finance", "Sales", "Legal", "Marketing", "Operations", "Engineering",
                     "Granted Things"]),
    ("roles", [f"Role {chr(65 + i)}" for i in range(13)] + ["Denied Viewer"]),
    ("request_templates.style_1", ["{person_name} {department}"]),
    ("request_templates.style_1", ["{person_name} {department} {role_name} {role_name}"]),
    ("request_templates.style_1", ["{person_name} {department} {role_name} {other}"]),
    ("request_templates.style_1", ["{person_name} {department} {role_name} <tool_call>"]),
    ("request_templates.style_2", PLACEHOLDER["request_templates"]["style_2"][:2]
     + [PLACEHOLDER["request_templates"]["style_1"][0].upper().replace("{PERSON_NAME}", "{person_name}")
        .replace("{DEPARTMENT}", "{department}").replace("{ROLE_NAME}", "{role_name}")]),
])
def test_what_check_text_refuses_in_one_field_the_field_check_refuses_too(ins, key, value):
    chosen = draft(ins)[0].chosen
    problems, reason = one_bad_value(ins, chosen, key, value)
    assert problems != [], "the case must be one that check_text refuses"
    assert reason is not None


def test_the_good_values_pass_both_checks(ins):
    """The control of the test above: with the good value both checks are silent."""
    chosen = draft(ins)[0].chosen
    for key in ("tools.department_info.name", "result_keys.manager_id", "values.resolution.denied",
                "errors.unknown_role_name", "system_prompt"):
        assert one_bad_value(ins, chosen, key, chosen[key]) == ([], None)


# ---------------------------------------------------------------------------
# List fields
# ---------------------------------------------------------------------------
def items_of(line: dict) -> list[tuple[str, bool, bool]]:
    return [(entry["value"], entry["valid"], entry["taken"]) for entry in line["items"]]


def test_a_list_keeps_the_first_valid_items_and_drops_the_rest(ins, good):
    key = "departments"
    names = good[key]
    reply = ["Departments:", names[0], "Research And Development Core Team", names[1], "Team 7",
             names[0].upper(), "R&D", "Granted Requests", f"1. {names[2]}", "Self-Service", "-Bad",
             "Two  Spaces"] + names[3:] + ["One More"]
    result, lines, writer = draft(ins, {(key, 0): reply})
    wanted = names[:3] + ["Self-Service"] + names[3:7]
    assert result.chosen[key] == wanted and len(wanted) == 8
    (line,) = candidates(lines, key)                                # one reply was enough
    reasons = {entry["value"]: entry["reason"] for entry in line["items"]}
    assert reasons["Departments:"] == "ends with a colon"
    assert "is not 1 to 4 words" in reasons["Research And Development Core Team"]
    assert "is not 1 to 4 words" in reasons["Team 7"]
    assert "is not 1 to 4 words" in reasons["R&D"]
    assert "is not 1 to 4 words" in reasons["-Bad"]
    assert "is not 1 to 4 words" in reasons["Two  Spaces"]
    assert reasons[names[0].upper()] == "is in the list already"
    assert reasons["Granted Requests"] == "holds the resolution value 'granted'"
    assert reasons[names[2]] is None                                # the list marker was removed
    # valid items after the eighth are not taken: nobody picks among them
    assert items_of(line)[-2:] == [(names[7], True, False), ("One More", True, False)]
    assert (line["valid"], line["taken"], line["reason"]) == (True, True, None)
    assert line["cleaned"] == [entry["value"] for entry in line["items"]]


def test_a_list_with_too_few_items_asks_again_and_appends(ins, good):
    key = "request_templates.style_3"
    a, b, c = good[key]
    extra = "yo, {role_name} for {person_name} in {department} pls"
    replies = {(key, 0): [a, "no variables here", b],              # two valid items
               (key, 1): ["nothing useful"],                       # none
               (key, 2): [b, extra, c]}                            # one repeated, then two new: one is needed
    result, lines, writer = draft(ins, replies)
    assert result.chosen[key] == [a, b, extra]
    first, second, third = candidates(lines, key)
    assert [line["index"] for line in (first, second, third)] == [0, 1, 2]
    assert items_of(first) == [(a, True, True), ("no variables here", False, False), (b, True, True)]
    assert (second["valid"], second["taken"], second["reason"]) == (False, False,
                                                                    "the reply holds no valid new item")
    assert items_of(third) == [(b, False, False), (extra, True, True), (c, True, False)]
    assert third["items"][0]["reason"] == "is equal to another template when the variables are removed"
    assert first["messages"] == second["messages"] == third["messages"]
    assert len({line["seed"] for line in (first, second, third)}) == 3
    assert (key, 3) not in writer.calls


TEMPLATE = "Please add {role_name} for {person_name} of {department}."


@pytest.mark.parametrize("item, reason", [
    ("Please add the role for {person_name} of {department}.", "three variables"),
    ("Add {role_name} for {person_name} ({person_name}) of {department}.", "three variables"),
    ("Add {role_name} for {person_name} of {department} by {date}.", "three variables"),
    ("Add {role_name} for {person_name} of {department} {{now}}.", "three variables"),
    ("Add {role_name!r} for {person_name} of {department}.", "three variables"),
    ("Add {role_name} for {person_name} of {department}. }", "three variables"),
    ("Add {} for {person_name} of {department}.", "three variables"),
    ("please ADD {person_name}   for {department} of {role_name}.", "equal to another template"),
    ("Add {role_name} for {person_name} of {department}. </tool_call>", "chat format"),
    ("Add {role_name} for {person_name} of {department}, " + "really " * 30 + "soon.", "at most 200"),
    # check C4: what stands directly after {role_name}
    (PRESENT_STYLE_3[0], "the word 'access' stands directly after {role_name}"),
    ("{person_name} of {department}: {role_name} permissions wanted", "the word 'permissions' stands"),
    ('Give "{role_name}" rights to {person_name} of {department}.', "the word 'rights' stands directly"),
    ("({role_name}) access for {person_name} of {department}", "the word 'access' stands directly after"),
    ("{role_name}-access for {person_name} of {department}", "the word 'access' stands directly after"),
    ("{role_name}'s holder shall be {person_name} of {department}", "the word 's' stands directly after"),
    ("{person_name} wants {role_name} {department}", "another variable stands directly after {role_name}"),
    ("{person_name}: {role_name}{department}", "another variable stands directly after {role_name}"),
])
def test_template_rules(ins, item, reason):
    assert reason in drafting.check_item(ins, "request_templates.style_2", item, {}, [TEMPLATE])
    assert drafting.check_item(ins, "request_templates.style_2", TEMPLATE, {}, []) is None


def test_a_template_must_differ_from_the_templates_of_the_other_styles(ins, good):
    key = "request_templates.style_2"
    copied = good["request_templates.style_1"][0]
    swapped = copied.replace("{person_name}", "{x}").replace("{role_name}", "{person_name}").replace(
        "{x}", "{role_name}")                                       # the same words, the variables exchanged
    result, lines, _ = draft(ins, {(key, 0): [copied, swapped] + good[key]})
    assert result.chosen[key] == good[key]
    line = candidates(lines, key)[0]
    assert [entry["reason"] for entry in line["items"][:2]] == [
        "is equal to another template when the variables are removed"] * 2


@pytest.mark.parametrize("name, reason", [
    ("Database Admin", "holds the word 'admin'"),
    ("Top-Secret Files Reader", "holds the word 'secret'"),
    ("PUBLIC Wiki Editor", "holds the word 'public'"),
    ("Read-Only Ledger", "holds the word 'only'"),
    ("Secrets Vault Reader", "holds the word 'secret'"),          # plural forms (DRAFT-CHECK repair)
    ("Payroll Admins", "holds the word 'admin'"),
    ("Superusers Group", "holds the word 'superuser'"),
    ("Masteres Files", "holds the word 'master'"),
    ("Ledger Reader 2", "is not 1 to 4 words"),
    ("Very Long Role Name Here", "is not 1 to 4 words"),
    ("wiki   editor", "is not 1 to 4 words"),
    ("Wiki editor", "is in the list already"),
])
def test_role_name_rules(ins, name, reason):
    chosen = {"values.resolution.denied": "denied"}
    assert reason in drafting.check_item(ins, "roles", name, chosen, ["Wiki Editor"])
    # the same name is a valid department, unless the rule is one for both kinds of name
    as_department = drafting.check_item(ins, "departments", name, chosen, ["Wiki Editor"])
    assert (as_department is None) == ("holds the word" in reason)
    assert drafting.check_item(ins, "roles", "Administration Files Clerk", chosen, []) is None
    # words that only end like a plural of a listed word are not refused
    for name in ("Access Records Clerk", "Process Logs Viewer", "Sales Ledger Editor"):
        assert drafting.check_item(ins, "roles", name, chosen, []) is None, name


def test_the_extra_tool_check_does_not_refuse_a_word_that_holds_its_name_inside(ins, good):
    """The main prompt may not hold the extra tool's name, also at the start of a longer word; but a short
    name inside a longer word ("count" in "account") is not a mention, or every prompt would be refused."""
    chosen = {**draft(ins)[0].chosen, "tools.department_info.name": "count"}
    prompt = good["system_prompt"] + "\nThe account of each person is checked."
    assert "account" in prompt
    assert drafting.check_value(ins, "system_prompt", prompt, chosen) is None
    assert "names the extra tool" in drafting.check_value(ins, "system_prompt", prompt + " Count_people.",
                                                          chosen)


def test_a_first_line_with_a_colon_is_refused_only_in_the_first_line(ins, good):
    chosen = draft(ins)[0].chosen
    assert ":" in good["system_prompt"].split("\n", 2)[2]           # later lines may end with a colon
    assert drafting.check_value(ins, "system_prompt", good["system_prompt"], chosen) is None


def test_the_account_word_instructions_state_the_fact_and_not_the_rule(ins):
    """The status words stand in every person lookup, also under the plain prompt. An instruction that says
    what the desk may do for such a person ("can be given no role") would put the rule into a tool result."""
    for key in ("values.status.active", "values.status.inactive"):
        instruction = ins.fields[key].instruction
        assert "can be given" not in instruction and "says nothing about what the desk" in instruction
    assert "in use" in ins.fields["values.status.active"].instruction
    assert "switched off" in ins.fields["values.status.inactive"].instruction


def test_the_desk_description_starts_with_its_heading_on_a_line_of_its_own(ins):
    assert ins.desk.split("\n", 1)[0] == "BACKGROUND"


def test_a_cut_off_list_reply_gives_no_item(ins, good):
    key = "roles"
    result, lines, _ = draft(ins, {(key, 0): ReplyCutOff("\n".join(good[key][:5]) + "\nBilli")})
    first, second = candidates(lines, key)
    assert (first["valid"], first["taken"], first["reason"]) == (False, False, CUT_OFF_REASON)
    assert all(not entry["taken"] and entry["reason"] == CUT_OFF_REASON for entry in first["items"])
    assert len(first["items"]) == 6
    assert result.chosen[key] == good[key] and second["taken"]


# ---------------------------------------------------------------------------
# A field with no valid candidate
# ---------------------------------------------------------------------------
def test_a_field_with_no_valid_candidate_is_reported_and_the_run_goes_on(ins):
    key = "values.resolution.pending"
    result, lines, writer = draft(ins, {(key, i): "not valid!" for i in range(N)})
    assert not result.complete
    assert result.failed[key] == "no valid candidate among 8"
    assert len(candidates(lines, key)) == N and not any(line["taken"] for line in candidates(lines, key))
    assert key not in result.chosen
    # fields that need the value are not written, and say why; all other fields are written
    for dependent in ("values.resolution.already", "system_prompt", "policy_shift.system_prompt",
                      "plain_system_prompt", "errors.malformed_call"):
        assert key in result.failed[dependent] and dependent not in result.chosen
        assert candidates(lines, dependent) == []
    assert "tools.close_request.description" in result.chosen and "roles" in result.chosen
    assert sorted(result.failed) == sorted(line["field"] for line in lines if line["kind"] == "failed")
    assert set(result.failed) | set(result.chosen) == set(ins.fields)


def test_a_list_that_never_gets_enough_items_fails(ins, good):
    key = "departments"
    result, lines, writer = draft(ins, {(key, i): good[key][:7] for i in range(N)})
    assert result.failed == {key: "no valid candidate among 8"}
    assert len(candidates(lines, key)) == N
    assert sum(entry["taken"] for line in candidates(lines, key) for entry in line["items"]) == 7


def test_fewer_candidates_can_be_asked_for(ins):
    key = "id_prefixes.person"
    assert key in draft(ins, {(key, 0): "?", (key, 1): "?"})[0].chosen     # with 8 candidates: the third
    writer = FakeWriter(ins, {(key, 0): "?", (key, 1): "?"})
    result = drafting.draft_all(ins, writer, writer_id="fake@0", run_seed=1, n_candidates=2)
    assert result.failed[key] == "no valid candidate among 2"
    assert [call for call in writer.calls if call[0] == key] == [(key, 0), (key, 1)]


# ---------------------------------------------------------------------------
# The copy rule: a value may not repeat a long run of the words that its writer was shown
# ---------------------------------------------------------------------------
def longest_by_brute_force(words: list[str], texts: list[list[str]]) -> int:
    """The length of the longest shared run, found the slow way with the helpers of this file."""
    return max((n for n in range(1, len(words) + 1) if any(runs_of(words, n) & runs_of(text, n)
                                                           for text in texts)), default=0)


def instruction_sentence(ins: drafting.Instructions, start: str):
    """A reply function for the fake writer: the one sentence of the shown task that begins with `start`.

    It reads the sentence out of the user message, between the heading of the task and the heading of the
    earlier values, so the reply is a piece of what this very writer was shown.
    """
    def reply(messages: list[dict]) -> str:
        task = messages[1]["content"].split(ins.task_heading + "\n", 1)[1].split(ins.earlier_heading)[0]
        (sentence,) = [found for found in re.split(r"(?<=[.])\s+", " ".join(task.split()))
                       if found.startswith(start)]
        return sentence
    return reply


def test_words_are_compared_in_lower_case_without_punctuation_and_variables_are_words():
    assert drafting.words_of("The person's Role-ID: {value}, {person_name}; lookup_person -> X.") == [
        "the", "persons", "role", "id", "{value}", "{person_name}", "lookup_person", "x"]
    assert drafting.words_of("{department} {role_name} {other} department value") == [
        "{department}", "{role_name}", "other", "department", "value"]
    assert drafting.words_of("it’s café-time, 2 ids") == ["its", "café", "time", "2", "ids"]
    assert drafting.words_of(" \n ") == []


def test_the_longest_shared_run_is_one_piece_of_one_text():
    run = drafting.longest_shared_run
    words = "one two three four five six".split()
    assert run(words, ["zero one two three nine".split(), "three four five six seven".split()]) == [
        "three", "four", "five", "six"]
    assert run(words, ["one two".split(), "three four".split()]) == ["one", "two"]    # not across two texts
    assert run(words, ["five six x one two".split()]) == ["one", "two"]             # of equals: the first
    assert run("x one two x one two three".split(), [words]) == ["one", "two", "three"]
    assert run(words, []) == [] and run([], [words]) == [] and run(words, [["seven"]]) == []
    rng = Rng(7)
    lengths: set[int] = set()
    for _ in range(300):                                           # against the slow way, on small alphabets
        made = [[str(rng.below(3)) for _ in range(rng.below(12) + 1)] for _ in range(3)]
        found = run(made[0], made[1:])
        assert len(found) == longest_by_brute_force(made[0], made[1:])
        assert not found or any(tuple(found) in runs_of(text, len(found)) for text in made[1:])
        lengths.add(len(found))
    assert len(lengths) >= 5


def test_the_shown_texts_are_everything_that_the_writer_reads_but_the_fixed_values(ins):
    """Every line of eight or more words in a request to the writer counts as a copy if it comes back; of a
    line in the block of fixed values only the label counts, not the value."""
    values = drafting.standin_values(ins)
    n_lines = n_labels = 0
    for key, field in ins.fields.items():
        if not field.asked:
            continue
        system, user = drafting.build_messages(ins, key, values)
        before, heading, block = user["content"].partition(ins.earlier_heading)
        assert (heading != "") == (block != "") == bool(field.uses)
        for line in (system["content"] + "\n" + before + "\n" + heading).splitlines():
            if len(drafting.words_of(line)) >= drafting.COPY_LIMIT:
                assert drafting.words_of(line) == drafting.shared_run(ins, key, line), (key, line)
                assert drafting.copy_problem(ins, key, line).startswith("copies the instruction")
                n_lines += 1
        # the message up to the block of fixed values is one of the shown texts, exactly as it was sent
        assert before.strip() in drafting.shown_texts(ins, key)
        for name in field.uses:
            label = ins.fields[name].label
            assert f"- {label}:" in block
            if len(drafting.words_of(label)) >= drafting.COPY_LIMIT:
                assert drafting.copy_problem(ins, key, label) is not None
                n_labels += 1
        assert not any("<<" in text for text in drafting.shown_texts(ins, key))       # no value among them
    assert n_lines > 1000 and n_labels > 100


def test_a_run_may_cross_from_the_heading_into_the_instruction_but_not_into_what_was_not_shown(ins):
    """The desk, the heading of the task and the instruction are one text, as in the message: a run that
    begins in one and ends in the next is one run. The heading of the fixed values counts only for a writer
    that was shown it."""
    key = "errors.unknown_role_id"
    field = ins.fields[key]
    assert field.uses == ()
    heading, start = drafting.words_of(ins.task_heading), drafting.words_of(field.instruction)[:4]
    assert len(heading) == 4
    assert drafting.shared_run(ins, key, " ".join(heading + start) + " zzz") == heading + start
    assert drafting.copy_problem(ins, key, " ".join(heading + start) + " zzz") is not None
    assert drafting.copy_problem(ins, key, " ".join(heading[1:] + start) + " zzz") is None     # seven words
    end_of_desk = drafting.words_of(ins.desk)[-4:]
    assert drafting.shared_run(ins, key, " ".join(end_of_desk + heading)) == end_of_desk + heading
    fixed = drafting.words_of(ins.earlier_heading)
    assert len(fixed) >= 12
    assert len(drafting.shared_run(ins, key, " ".join(fixed))) < 4             # a few words, by chance
    assert drafting.copy_problem(ins, key, " ".join(fixed)) is None
    with_values = "tools.grant_role.description"
    assert ins.fields[with_values].uses and drafting.shared_run(ins, with_values, " ".join(fixed)) == fixed


@pytest.mark.parametrize("key, start, n_words", [
    ("tools.department_info.params.department.description", "Write the description of the one parameter", 11),
    ("tools.close_request.description", "The assistant reads it", 16),
    ("values.confirmation.role_granted", "When the tool that gives a role", 16),
    ("errors.unknown_role_name", "Write a normal sentence of your own", 17),
    # The plain prompt stood here until 2026-11-18 ("A comparison test uses it", 26 words). Since check C17
    # a sentence of the instruction is no valid plain prompt even without the copy rule: it does not tell
    # the two working rules. The copy rule of the plain prompt is tested in
    # `test_the_same_limit_of_eight_holds_for_the_three_system_prompts`.
])
def test_a_writer_that_returns_a_sentence_of_its_instruction_is_refused_as_a_copy(ins, good, key, start,
                                                                                  n_words):
    result, lines, _ = draft(ins, {(key, 0): instruction_sentence(ins, start)})
    first, second = candidates(lines, key)
    sentence = first["cleaned"]
    assert sentence.startswith(start) and sentence in " ".join(ins.fields[key].instruction.split())
    assert (first["valid"], first["taken"]) == (False, False)
    assert first["reason"].startswith("copies the instruction: it shares a run of "), first["reason"]
    # the whole sentence is the shared run, and the run itself is in the log
    assert first["shared_run"] == {"length": n_words, "words": " ".join(drafting.words_of(sentence))}
    assert first["shared_run"]["words"] in first["reason"]
    # no other rule refuses the sentence: without the copy rule it would have been taken
    others = {name: value for name, value in result.chosen.items() if name != key}
    assert drafting._value_problem(ins, key, sentence, others) is None
    assert (second["index"], second["valid"], second["reason"], second["taken"]) == (1, True, None, True)
    assert result.chosen[key] == good[key] and result.complete


@pytest.mark.parametrize("key, start", [
    ("system_prompt", "It is the only place where the assistant learns"),
    ("system_prompt", "Do not speak of tests"),
    ("policy_shift.system_prompt", "The new prompt must be complete in itself"),
])
def test_a_system_prompt_that_takes_a_sentence_of_its_instruction_over_is_refused(ins, good, key, start):
    sentence = instruction_sentence(ins, start)
    result, lines, _ = draft(ins, {(key, 0): lambda messages: good[key] + "\n" + sentence(messages)})
    first, second = candidates(lines, key)
    assert first["reason"].startswith("copies the instruction") and not first["taken"]
    assert first["shared_run"]["words"] == " ".join(drafting.words_of(sentence(first["messages"])))
    assert first["shared_run"]["length"] >= 10
    assert second["taken"] and result.chosen[key] == good[key]


def test_notes_without_their_arrows_are_refused_and_short_fragments_in_other_words_are_not(ins, good):
    """What the copy rule does, and what it cannot do.

    A line of the notes that comes back with its arrow and its semicolons exchanged is still a copy: the
    words are compared without punctuation. But a writer can put two short fragments of the notes into a
    phrase of its own. That value is taken, and the log shows how many words it shares.
    """
    key = "tools.lookup_person.description"
    (line,) = [found.strip() for found in ins.fields[key].instruction.splitlines()
               if found.strip().startswith("per person in list ->")]
    disguised = line.replace(" ->", ":").replace(";", ",") + "."
    assert "->" not in disguised and len(disguised) < 100
    result, lines, _ = draft(ins, {(key, 0): disguised})
    first, second = candidates(lines, key)
    assert first["reason"].startswith("copies the instruction") and not first["taken"]
    assert first["shared_run"] == {"length": len(drafting.words_of(line)), "words": " ".join(
        drafting.words_of(line))} and first["shared_run"]["length"] >= 10
    assert second["taken"] and result.chosen[key] == good[key]

    key = "tools.grant_role.params.person_id.description"
    notes = " ".join(ins.fields[key].instruction.split())
    fragment = notes.split("whose -> ", 1)[1].split(".", 1)[0]
    assert 3 <= len(fragment.split()) < drafting.COPY_LIMIT
    joined = f"Person id of the {fragment}."
    result, lines, _ = draft(ins, {(key, 0): joined})
    (only,) = candidates(lines, key)
    assert only["taken"] and result.chosen[key] == joined
    assert only["shared_run"] == {"length": len(fragment.split()), "words": fragment}


def test_a_run_of_seven_shared_words_passes_and_a_run_of_eight_is_refused(ins):
    assert drafting.COPY_LIMIT == 8
    key = "tools.grant_role.description"
    words = drafting.words_of(ins.fields[key].instruction)
    assert words[:9] == "write the description of the tool that gives a".split()
    seven, eight = (" ".join(words[:n]) + " zzz" for n in (7, 8))
    result, lines, _ = draft(ins, {(key, 0): eight, (key, 1): seven})
    first, second = candidates(lines, key)
    assert first["shared_run"] == {"length": 8, "words": " ".join(words[:8])}
    assert (first["valid"], first["taken"]) == (False, False)
    assert first["reason"] == ("copies the instruction: it shares a run of 8 words with the text that the "
                               f"writer was shown ({' '.join(words[:8])!r}); the limit is 8")
    assert second["shared_run"] == {"length": 7, "words": " ".join(words[:7])}
    assert (second["valid"], second["reason"], second["taken"]) == (True, None, True)
    assert result.chosen[key] == seven


@pytest.mark.parametrize("key", ["system_prompt", "policy_shift.system_prompt", "plain_system_prompt"])
def test_the_same_limit_of_eight_holds_for_the_three_system_prompts(ins, good, key):
    chosen = draft(ins)[0].chosen
    words = drafting.words_of(ins.fields[key].instruction)
    seven, eight = (good[key] + " Zzz " + " ".join(words[:n]) + " zzz." for n in (7, 8))
    assert drafting.check_value(ins, key, seven, chosen) is None
    assert drafting.shared_run(ins, key, seven) == words[:7]
    assert drafting.check_value(ins, key, eight, chosen).startswith("copies the instruction")
    assert drafting.shared_run(ins, key, eight) == words[:8]


def test_a_template_variable_counts_as_a_word_of_a_shared_run(ins):
    key = "request_templates.style_1"
    assert "{person_name} for the full name of the person" in " ".join(ins.fields[key].instruction.split())
    eight = "{person_name} for the full name of the person, then {department}, then {role_name}"
    seven = "{department} for the full name of the person, then {person_name}, then {role_name}"
    assert drafting.check_item(ins, key, eight, {}, []).startswith("copies the instruction")
    assert drafting.shared_run(ins, key, eight) == ["{person_name}", "for", "the", "full", "name", "of",
                                                    "the", "person"]
    assert drafting.check_item(ins, key, seven, {}, []) is None
    assert drafting.shared_run(ins, key, seven) == ["for", "the", "full", "name", "of", "the", "person"]


def test_a_copied_list_item_is_refused_and_the_log_line_holds_the_longest_run_of_its_items(ins, good):
    key = "request_templates.style_1"
    copied = "{person_name} for the full name of the person, then {department}, then {role_name}"
    result, lines, _ = draft(ins, {(key, 0): [copied] + good[key]})
    (line,) = candidates(lines, key)
    assert result.chosen[key] == good[key]
    first = line["items"][0]
    assert (first["valid"], first["taken"]) == (False, False)
    assert first["reason"].startswith("copies the instruction")
    assert first["shared_run"] == {"length": 8, "words": "{person_name} for the full name of the person"}
    assert line["shared_run"] == first["shared_run"]
    assert all(entry["taken"] and entry["shared_run"]["length"] < 8 for entry in line["items"][1:])


def test_words_of_values_that_are_already_fixed_do_not_count(ins, good):
    result, lines, _ = draft(ins)
    main, shift = (result.chosen[key] for key in ("system_prompt", "policy_shift.system_prompt"))
    (line,) = candidates(lines, "policy_shift.system_prompt")
    assert main in line["messages"][1]["content"]                 # the writer was shown the first prompt ...
    with_main = drafting.longest_shared_run(drafting.words_of(shift), [drafting.words_of(main)])
    assert len(with_main) > 60                                    # ... and repeats a long run of it, as told
    assert line["taken"] and line["shared_run"]["length"] < drafting.COPY_LIMIT
    assert not any(main in text for text in drafting.shown_texts(ins, "policy_shift.system_prompt"))
    # nine fixed names in a row are nine words that the writer was shown, and none of them counts
    names = [result.chosen[f"tools.{tool}.name"] for tool in TOOL_KEYS]
    names += [result.chosen[f"values.resolution.{key}"]
              for key in ("granted", "pending", "already", "denied")]
    (main_line,) = candidates(lines, "system_prompt")
    assert all(f": {name}\n" in main_line["messages"][1]["content"] + "\n" for name in names)
    listed = main + "\nUse these names only: " + ", ".join(names) + "."
    assert drafting.check_value(ins, "system_prompt", listed, result.chosen) is None
    assert drafting.shared_run(ins, "system_prompt", listed) == drafting.shared_run(ins, "system_prompt",
                                                                                    main)


def test_the_labels_of_the_fixed_values_are_our_words_and_do_count(ins, good):
    chosen = draft(ins)[0].chosen
    label = ins.fields["result_keys.has_access"].label
    assert "result_keys.has_access" in ins.fields["system_prompt"].uses
    assert label in drafting.shown_texts(ins, "system_prompt")
    words = drafting.words_of(label)
    assert len(words) >= 12
    seven, eight = (good["system_prompt"] + "\nZzz " + " ".join(words[2:2 + n]) + " zzz." for n in (7, 8))
    assert drafting.check_value(ins, "system_prompt", seven, chosen) is None
    assert drafting.check_value(ins, "system_prompt", eight, chosen).startswith("copies the instruction")
    # a label that this writer is not shown does not count for it
    other = drafting.words_of(ins.fields["request_templates.style_4"].label)
    assert len(other) >= 9 and "request_templates.style_4" not in ins.fields["system_prompt"].uses
    unseen = good["system_prompt"] + "\nZzz " + " ".join(other[:9]) + " zzz."
    assert drafting.check_value(ins, "system_prompt", unseen, chosen) is None


def test_the_log_holds_the_shared_run_of_every_candidate_valid_or_not(ins, good):
    key = "tools.grant_role.description"
    replies = {(key, 0): "", (key, 1): instruction_sentence(ins, "The assistant reads it"),
               (key, 2): ReplyCutOff("Gives a role to a"),
               ("roles", 0): ReplyCutOff("\n".join(good["roles"][:3])),
               ("departments", 0): ["Finance:", "R&D"] + good["departments"]}
    result, lines, _ = draft(ins, replies)
    logged = [line for line in json.loads(json.dumps(lines)) if line["kind"] == "candidate"]
    assert len(logged) == 64 + 3 + 1 and result.complete
    for line in logged:
        run = line["shared_run"]
        assert set(run) == {"length", "words"} and run["length"] == len(run["words"].split())
        shown = [drafting.words_of(text) for text in drafting.shown_texts(ins, line["field"])]
        values = [entry["value"] for entry in line["items"]] if "items" in line else [line["cleaned"]]
        slow = [longest_by_brute_force(drafting.words_of(value), shown) for value in values]
        assert run["length"] == max(slow, default=0), line["field"]
        if "items" in line:
            assert [entry["shared_run"]["length"] for entry in line["items"]] == slow
    empty, copied, cut, taken = candidates(lines, key)
    assert empty["shared_run"] == {"length": 0, "words": ""} and empty["reason"] == "is empty"
    assert copied["shared_run"]["length"] == 16 and copied["reason"].startswith("copies the instruction")
    assert cut["shared_run"] == {"length": 5, "words": "gives a role to a"}
    assert cut["reason"] == CUT_OFF_REASON
    assert taken["taken"] and taken["shared_run"]["length"] == 5
    (cut_list, _) = candidates(lines, "roles")
    assert cut_list["shared_run"]["length"] == 0 and len(cut_list["items"]) == 3
    # no valid candidate reaches the limit, and the valid ones are measured too (not all of them are 0)
    valid = [line["shared_run"]["length"] for line in logged if line["valid"]]
    assert len(valid) == 64 and max(valid) < drafting.COPY_LIMIT
    assert sum(length > 0 for length in valid) > 20


def without_measurements(lines: list[dict]) -> list[dict]:
    """A log as it was written before the copy rule existed: no line and no item holds `shared_run`."""
    old = json.loads(json.dumps(lines))
    for line in old:
        line.pop("shared_run", None)
        for entry in line.get("items", []):
            entry.pop("shared_run")
    return old


def test_the_header_names_the_largest_shared_run_among_the_taken_values(ins, good):
    key = "tools.grant_role.description"
    words = drafting.words_of(ins.fields[key].instruction)
    seven, eight = (" ".join(words[:n]) + " zzz" for n in (7, 8))
    result, lines, _ = draft(ins, {(key, 0): eight, (key, 1): seven})
    assert candidates(lines, key)[0]["reason"].startswith("copies the instruction")
    report = drafting.log_report(ins, lines)
    # the refused candidate, with its run of 8, is not a taken value
    assert report["largest_run"] == {"field": key, "length": 7, "words": " ".join(words[:7])}
    assert report["unmeasured"] == []
    header = file_content(ins, result.chosen, lines).split("\nmeta:")[0]
    wanted = ("# Largest shared run among the values that were taken: "
              f'7 words in {key} ("{" ".join(words[:7])}")\n')
    assert wanted in header and header.count("Largest shared run") == 1
    assert "# Copy check: a candidate that shared a run of 8 or more words with the text shown" in header
    assert header.rstrip().endswith("# Read and approved by: (not yet)")
    # the run of a list field is the longest among its TAKEN items
    listed = "request_templates.style_1"
    item = "{person_name} has the full name of the person, sits in {department}, wants {role_name}"
    # (since check C9 the fourth item takes its seven words from another place of the instruction: with
    # "the full name of the person" it would be refused as nearly the same sentence as the first item)
    extra = "{role_name}, then {person_name}; for the name of the person's department take {department}"
    _, lines, _ = draft(ins, {(listed, 0): [item] + good[listed][:2] + [extra]})
    last = candidates(lines, listed)[0]["items"][3]
    assert (last["valid"], last["taken"], last["shared_run"]["length"]) == (True, False, 7)
    assert drafting.taken_runs(lines)[listed] == {"length": 6, "words": "the full name of the person"}
    assert drafting.log_report(ins, lines)["largest_run"] == {
        "field": listed, "length": 6, "words": "the full name of the person"}
    # the fake writer's usual values: the largest run is found by a search of the test's own
    _, lines, _ = draft(ins)
    runs = {line["field"]: line["shared_run"]["length"] for line in lines if line["kind"] == "candidate"}
    largest = drafting.log_report(ins, lines)["largest_run"]
    assert largest["length"] == max(runs.values()) == runs[largest["field"]] == 5
    # a log from before the copy rule: the header says that nothing was measured
    old = drafting.log_report(ins, without_measurements(lines))
    assert old["largest_run"] is None and sorted(old["unmeasured"]) == sorted(runs)
    header = file_content(ins, result.chosen, without_measurements(lines))
    assert "# Largest shared run among the values that were taken: not measured\n" in header
    assert "#   Not measured (their log lines are older than the copy check): [" in header
    assert "values that were taken: not measured\n" in file_content(ins, result.chosen)   # an empty log


def test_the_placeholder_report_gives_the_longest_run_shared_with_the_placeholder_file(ins, good, tmp_path):
    result, _, _ = draft(ins)
    report = drafting.placeholder_report(ins, result.chosen)
    # the fake writer answers with the placeholder's own texts, so long runs must be found; the plain prompt
    # holds no tool name. Since check C19 its start is cut (`good_values`), so only its last two
    # sentences stand in the placeholder as they are: 15 words
    plain = drafting.words_of(good["plain_system_prompt"])
    assert report["length"] >= 15 and len(plain) == 30
    texts = [drafting.words_of(text) for text in drafting._leaf_strings(PLACEHOLDER)]
    slow = {key: max(longest_by_brute_force(drafting.words_of(item), texts)
                     for item in (value if isinstance(value, list) else [value]))
            for key, value in good.items()}
    assert slow["plain_system_prompt"] == 15
    assert report["length"] == max(slow.values()) == slow[report["field"]]
    assert report["words"] in " ".join(drafting.words_of(placeholder_path().read_text(encoding="utf-8")))
    assert len(report["words"].split()) == report["length"]
    # it is a report and not a rule: every one of these values was taken
    assert result.complete
    # only values that the model wrote are measured, and a value that shares nothing gives no field
    assert drafting.placeholder_report(ins, {"errors.unknown_role_id": "zzz qqq"}) == {
        "field": None, "length": 0, "words": ""}
    assert drafting.placeholder_report(ins, {"errors.unknown_role_id": "zzz person has the qqq"}) == {
        "field": "errors.unknown_role_id", "length": 3, "words": "person has the"}
    assert "people" in texts[[text == ["people"] for text in texts].index(True)]
    assert drafting.placeholder_report(ins, {"tools.grant_role.name": "people"})["field"] is None  # fixed
    assert drafting.placeholder_report(ins, {"result_keys.matches": "people"})["length"] == 1
    # another file, and a file that is not there
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({"a": {"b": "one zzz qqq www two"}, "c": ["www two three"]}),
                     encoding="utf-8")
    assert drafting.placeholder_report(ins, {"departments": ["Finance", "qqq www two three"]}, other) == {
        "field": "departments", "length": 3, "words": "qqq www two"}      # one value of the file, not two
    assert drafting.placeholder_report(ins, result.chosen, tmp_path / "none.yaml") is None


def test_draws_are_counted_per_run_and_a_reused_reply_is_not_a_draw(ins):
    def candidate(field: str, reused: bool = False) -> dict:
        return {"kind": "candidate", "field": field, "reused": reused, "taken": True}
    lines = [candidate("a"),                                               # before any "run" line: one run
             {"kind": "run"}, candidate("a"), candidate("a"), candidate("b"),
             {"kind": "run"}, candidate("a", True), candidate("b", True), candidate("b"), candidate("c"),
             {"kind": "run"}, {"kind": "kept", "field": "a"}, {"kind": "fixed", "field": "d"}, candidate("c")]
    assert drafting.draw_counts(lines) == {"a": 2, "b": 2, "c": 2}
    assert drafting.draw_counts(lines[1:5]) == {"a": 1, "b": 1}
    assert drafting.draw_counts([]) == {}
    # the report lists every asked field, in the order of the instruction file, also one that was never drawn
    _, drafted, _ = draft(ins)
    draws = drafting.log_report(ins, [{"kind": "run"}] + drafted)["draws"]
    assert list(draws) == [key for key, field in ins.fields.items() if field.asked]
    assert set(draws.values()) == {1} and len(draws) == 64
    assert set(drafting.log_report(ins, [])["draws"].values()) == {0}


def test_the_content_of_every_sentence_field_is_given_as_notes(ins):
    noted = [key for key in ins.fields if key.startswith(("values.confirmation.", "errors."))
             or key.endswith((".description", "system_prompt"))]
    assert len(noted) == 3 + 6 + 18 + 3
    for key in noted:
        instruction = " ".join(ins.fields[key].instruction.split())
        assert "->" in instruction and "not the wording" in instruction, key
        for sentence_form in ("The value it expects:", "What it must tell:", "The text confirms that",
                              "What the prompt must contain", "What it must contain, and nothing more:"):
            assert sentence_form not in instruction, key
    assert "NOTES" in ins.system_message and "Never copy a note" in ins.system_message
    # a field whose value is a name, a word or a list has no notes, so the arrow is not refused there
    plain = [key for key, field in ins.fields.items() if field.asked and key not in noted]
    assert len(plain) == 64 - 30 and not any("->" in ins.fields[key].instruction for key in plain)


# ---------------------------------------------------------------------------
# The five checks added after the calibration (C1 to C5 in accessdesk/drafting.py)
# ---------------------------------------------------------------------------
def repaired(prompt: str, *names: str) -> str:
    """`prompt` after the named repairs (REPAIRS). Every repair must find its place in the prompt."""
    for name in names:
        for old, new in REPAIRS[name]:
            assert old in prompt, (name, old)
            prompt = prompt.replace(old, new)
    return prompt


def prompt_lines(prompt: str) -> list[str]:
    return [line for line in prompt.split("\n") if line.strip()]


def flat_text(data: dict) -> dict:
    """A text file as dotted key -> value, for every field of the text file."""
    strings, lists = drafting.text_fields()
    values: dict = {}
    for key in strings + lists:
        node = data
        for part in key.split("."):
            node = node[part]
        values[key] = node.strip() if isinstance(node, str) else node
    return values


@pytest.fixture(scope="module")
def present(ins) -> dict:
    """The values of a good draft, with the visible names of the first text laid over them."""
    return {**draft(ins)[0].chosen, **PRESENT_NAMES}


def problem_of(ins, chosen: dict, key: str, value) -> str | None:
    """The reason of the field check for one value (for a list field: for one item, as the first one)."""
    others = {name: found for name, found in chosen.items() if name != key}
    if ins.fields[key].kind == "list":
        return drafting.check_item(ins, key, value, others, [])
    return drafting.check_value(ins, key, value, others)


def test_the_fixtures_are_the_two_prompts_of_the_first_text():
    assert (len(PRESENT_MAIN.split()), len(PRESENT_SHIFT.split())) == (378, 392)
    assert len(prompt_lines(PRESENT_MAIN)) == len(prompt_lines(PRESENT_SHIFT)) == 5     # five paragraphs
    assert PRESENT_MAIN.count("lookup_person") == PRESENT_MAIN.count("check_access") == 1
    assert len(PRESENT_NAMES) == 1 + 12 + 12 + 2 + 4


@pytest.mark.parametrize("key, value, reason", [
    # since 2026-11-18 the check of the marker lines stands first: the first prompts were paragraphs, with
    # no line that begins with a marker. Check C1 by itself: the test after the next one.
    ("system_prompt", PRESENT_MAIN,
     "has too few lines that begin with a marker: 0 line(s) begin with 'Step'"),
    ("policy_shift.system_prompt", PRESENT_SHIFT, "has too few lines that begin with a marker"),
    ("tools.close_request.params.resolution.description", PRESENT_RESOLUTION_DESCRIPTION,
     "calls a parameter 'resolution', which is not its visible name (request_resolution)"),
    ("request_templates.style_3", PRESENT_STYLE_3[0], "the word 'access' stands directly after {role_name}"),
])
def test_each_of_the_five_places_of_the_first_text_is_refused(ins, present, key, value, reason):
    assert reason in problem_of(ins, present, key, value)


def test_what_was_clear_in_the_first_text_is_not_refused(ins, present):
    """The control of the test above: the neighbours of the five places pass the same checks."""
    for item in PRESENT_STYLE_3[1:]:
        assert problem_of(ins, present, "request_templates.style_3", item) is None
    # the description of the people-finding tool names person_id and department_name, which are parameters
    # of other tools; they are result keys too, and a description may say what a tool returns
    assert "person_id" in PRESENT_LOOKUP_DESCRIPTION and "department_name" in PRESENT_LOOKUP_DESCRIPTION
    assert problem_of(ins, present, "tools.lookup_person.description", PRESENT_LOOKUP_DESCRIPTION) is None
    no_keys = {**present, "result_keys.person_id": "pid", "result_keys.department": "dept"}
    assert problem_of(ins, no_keys, "tools.lookup_person.description", PRESENT_LOOKUP_DESCRIPTION) == (
        "names the parameter 'department_name', which this tool does not have")


def test_every_prompt_check_refuses_the_first_prompts_by_itself(present):
    """C1 (two parts), C2 and C5, each called alone: none of them hides behind another."""
    main, shift = prompt_lines(PRESENT_MAIN), prompt_lines(PRESENT_SHIFT)
    for lines, ruleset in ((main, "default"), (shift, "all_grants_need_approval")):
        assert "only in a list of tools" in drafting._tool_line_problem(lines, present, ruleset)
        assert "on no line of its own" in drafting._case_line_problem(lines, present, ruleset)
        assert "no line that names lookup_person" in drafting._lookup_line_problem(lines, present)
    # what the first main prompt named where: one tool in the line of the cases, all five in the last line
    named = [[tool for tool in TOOL_KEYS if drafting.mentions(line, tool)] for line in main]
    assert named == [[], [], [], ["grant_role"], list(TOOL_KEYS)]
    assert drafting._never_grants_problem(shift, present) == (
        "names grant_role in a sentence without 'not' or 'never'")
    # why C5 looks at sentences: the LINE that lists grant_role holds "never", in another sentence
    (line,) = [line for line in shift if "grant_role" in line]
    assert drafting.mentions(line, "never", any_case=True)
    # and the fourth case is found once the list of names is gone
    assert drafting._never_grants_problem(
        prompt_lines(repaired(PRESENT_SHIFT, "no list", "never called")), present) == (
            "speaks of a fourth case ('four cases'); this desk has three cases: a, b and c")


# Since check C7 the repair "account source" is one of them: case a of the first text said "account state"
# in two words and named no source (see `test_the_first_prompts_and_the_checks_of_the_second_calibration`).
ALL_BUT_C5 = ("tools", "step lines", "case lines", "no list", "returned list", "account source")
# Since the third calibration a prompt needs more: the repairs for checks C14, C15 and C16 (main prompt),
# and for C14, C15, C16 and C18 (the prompt of the second desk). They keep the words that older repairs
# look for, so they stand in front of those.
ROUND3_MAIN = ("every request", "held key", "sensitive key", "two messages c", "two messages d")
ROUND3_SHIFT = ("every request", "held key", "two messages c", "no second giving", "account word")


FEW_MARKERS = "has too few lines that begin with a marker"


@pytest.mark.parametrize("repairs, reason", [
    # the check of the marker lines stands first: no line of the first prompt begins with a marker
    ((), FEW_MARKERS + ": 0 line(s) begin with 'Step' and a number (at least 2 are needed) and 0 line(s) "
     "begin with 'Case' and a letter (at least 4 are needed)"),
    (("tools",), FEW_MARKERS),
    (("tools", "step lines"), FEW_MARKERS + ": 2 line(s) begin with 'Step' and a number (at least 2 are "
     "needed) and 0 line(s) begin with 'Case'"),
    (("tools", "step lines", "case lines"), "names the tool close_request only in a list of tools"),
    (("tools", "step lines", "case lines", "no list"),
     "no line that names lookup_person, with the line after it, holds the words 'list' and 'department'"),
    # check C7: without its repair the result key of the account state stands nowhere
    (("tools", "step lines", "case lines", "no list", "returned list"),
     "does not show where the account state is read: the result key 'account_state' stands neither on a "
     "line that names lookup_person"),
    # the checks of the third calibration stand behind the older ones: step 2 is the first to fail
    (ALL_BUT_C5, "the line of step 2 does not say that the step is done for every request"),
    (ALL_BUT_C5 + ("every request",), "does not hold the result key 'holds_role', letter for letter"),
    (ALL_BUT_C5 + ("every request", "held key"), "does not hold the result key 'is_sensitive'"),
    (ALL_BUT_C5 + ("every request", "held key", "sensitive key"),
     "names two calls, request_approval and the closing call"),
    (ALL_BUT_C5 + ("every request", "held key", "sensitive key", "two messages c"),
     "names two calls, grant_role and the closing call"),
    (ALL_BUT_C5 + ROUND3_MAIN, None),
    # a case may name the closing tool beside the tool of its own action: that tool is not counted
    (ALL_BUT_C5 + ROUND3_MAIN + ("close in cases",), None),
    # each repair is needed: without it the prompt is refused, whatever else is repaired
    (("step lines", "case lines", "no list", "returned list"), "does not name the tool lookup_person"),
    (("tools", "case lines", "no list", "returned list"), FEW_MARKERS + ": 0 line(s) begin with 'Step'"),
    (("tools", "step lines", "no list", "returned list"), FEW_MARKERS),
    (("tools", "step lines", "cases c and d", "no list", "returned list"),
     "and 2 line(s) begin with 'Case' and a letter (at least 4 are needed)"),
    (("tools", "step lines", "case lines", "returned list"), "names the tool close_request only in a list"),
])
def test_the_first_main_prompt_passes_after_one_small_repair_for_each_demand(ins, present, repairs, reason):
    found = problem_of(ins, present, "system_prompt", repaired(PRESENT_MAIN, *repairs))
    assert found is None if reason is None else reason in found, found


@pytest.mark.parametrize("repairs, check, reason", [
    # the tools are named in the steps, but steps one and two are one line: two tools on it
    (("tools",), "_tool_line_problem", "names the tool lookup_person only in a list of tools"),
    (("tools", "step lines"), "_tool_line_problem",
     "names the tool request_approval only in a list of tools"),
    (("tools", "step lines", "cases c and d", "no list", "returned list"), "_case_line_problem",
     "holds the resolution value 'role_already_held' on no line of its own"),
])
def test_what_check_c1_says_where_the_check_of_the_marker_lines_now_stands_first(present, repairs, check,
                                                                                 reason):
    """Three rows of the test above are refused by the check of the marker lines since 2026-11-18. Check C1
    by itself still refuses each of them for the reason that the row had before."""
    lines = prompt_lines(repaired(PRESENT_MAIN, *repairs))
    assert reason in getattr(drafting, check)(lines, present, "default")


@pytest.mark.parametrize("repairs, reason", [
    ((), FEW_MARKERS),
    (ALL_BUT_C5, "does not say that grant_role is not called at this desk: no sentence names it"),
    (ALL_BUT_C5 + ("never called",), "speaks of a fourth case ('four cases')"),
    (ALL_BUT_C5 + ("never called", "three cases"), "speaks of a fourth case ('Case d')"),
    (ALL_BUT_C5 + ("never called", "three cases", "no case d"), "(check C16)"),
    # the checks of the third calibration: the second desk forbids no second giving (C18, part i), and
    # its case c says "active", the word of the tool result (C18, part ii)
    (ALL_BUT_C5 + ROUND3_SHIFT[:3] + ("never called", "three cases", "no case d"),
     "forbids a second giving of a role, and at this desk no role is given even once (check C18)"),
    (ALL_BUT_C5 + ROUND3_SHIFT[:4] + ("never called", "three cases", "no case d"),
     "speaks of the account and does not hold the account word 'active'"),
    (ALL_BUT_C5 + ROUND3_SHIFT + ("never called", "three cases", "no case d"), None),
    # the sentence that forbids the tool, and the value of a given role, may stand in the line of a case:
    # only the tools and the resolution values of the desk are counted on a line
    (ALL_BUT_C5 + ROUND3_SHIFT + ("never called in case c", "three cases", "no case d"), None),
    # a list of names that holds the role-giving tool is refused, also beside the sentence that forbids it
    (("tools", "step lines", "case lines", "returned list", "never called", "three cases", "no case d"),
     "names the tool close_request only in a list of tools"),
    (ALL_BUT_C5 + ("never called", "three cases", "no case d", "named again"),
     "names grant_role in a sentence without 'not' or 'never'"),
    # the sentence of the main prompt against a second giving, with the tool's name: "never", but only twice
    (ALL_BUT_C5 + ("never twice", "three cases", "no case d"),
     "names grant_role in a sentence with 'twice', which forbids a second call only"),
])
def test_the_first_policy_shift_prompt_passes_after_its_repairs(ins, present, repairs, reason):
    found = problem_of(ins, present, "policy_shift.system_prompt", repaired(PRESENT_SHIFT, *repairs))
    assert found is None if reason is None else reason in found, found


def test_the_checks_for_a_desk_that_never_gives_a_role_hold_for_such_a_rule_set_only(ins, present):
    """The repaired main prompt names the role-giving tool without "not" and has a case d. That is right for
    a rule set that calls the tool, and refused for one that never does."""
    main = repaired(PRESENT_MAIN, *ALL_BUT_C5)
    assert drafting._prompt_line_problem(main, present, "default") is None
    assert drafting._prompt_line_problem(main, present, "sensitive_denied") is None
    assert drafting._prompt_line_problem(main, present, "all_grants_need_approval") == (
        "names grant_role in a sentence without 'not' or 'never'")


@pytest.mark.parametrize("lines, reason", [
    (["1. Call lookup_person with the name. It returns a list. Take the person of the right department."],
     None),
    (["1. Call lookup_person with the full name.", "From the LIST take the one whose department_name fits."],
     None),                                           # the line after it; any case; a word that begins so
    (["1. Call lookup_person with the full name.", "2. Then go on.", "Use the list and the department."],
     "no line that names lookup_person"),             # two lines after it
    (["1. Call lookup_person. It lists the people; the department decides."], "no line that names"),
    (["1. Call lookup_person. Take the right one from the list."], "no line that names"),
    (["1. Find the person in the list by department.", "2. Call check_access."], "no line that names"),
    # the result key of the list was "people": an ordinary word, which is why the key is not accepted
    (["1. Call lookup_person. If several people have the name, use the department."], "no line that names"),
])
def test_the_step_that_finds_the_person_must_speak_of_the_list_and_the_department(present, lines, reason):
    found = drafting._lookup_line_problem(lines, present)
    assert found is None if reason is None else reason in found
    assert present["result_keys.matches"] == "people"


@pytest.mark.parametrize("lines, reason", [
    (["c. Ask for approval. You do not call grant_role at this desk."], None),
    (["NEVER use grant_role. In case department names differ, the list decides.", "Do not guess."], None),
    (["Never give a role. The tools are grant_role and four others."], "in a sentence without 'not'"),
    (["You cannot call Grant_Role here."], "in a sentence without 'not'"),      # "cannot" is not "not"
    (["Close the request."], "no sentence names it"),
    (["Never make a Second call of grant_role."], "in a sentence with 'second'"),
    (["Never call grant_role. Never ask for approval twice."], None),           # "twice" in another sentence
    (["Never call grant_role.", "d. In every other case close the request."], "a fourth case ('d.')"),
    (["Never call grant_role.", "  (d) Otherwise close the request."], "a fourth case ('(d)')"),
    (["Never call grant_role. Then comes case D."], "a fourth case ('case D')"),
    (["Never call grant_role. There are 4 cases."], "a fourth case ('4 cases')"),
    (["Never call grant_role. The fourth  case is the last."], "a fourth case ('fourth  case')"),
])
def test_the_prompt_of_a_desk_that_never_gives_a_role(present, lines, reason):
    found = drafting._never_grants_problem(lines, present)
    assert found is None if reason is None else reason in found, found


@pytest.mark.parametrize("item", [
    "hey, could you get {person_name} the {role_name} role in {department}? thanks",
    "hey, could you get {person_name} the {role_name} in {department}? thanks",
    "hey, could you get {person_name} access to {role_name}, in {department}? thanks",
    "{role_name} - {person_name} - {department}",
    "access wanted ({department}, {person_name}): {role_name}",
    'please give "{role_name}" to {person_name} of {department}',
    "{person_name} of {department} needs the access role {role_name}",        # a word BEFORE it is free
    "{role_name}, access for {person_name} of {department}",                  # after a punctuation mark too
    "{role_name} ASAP for {person_name} of {department}",
    "{role_name}-{department}-{person_name}",
])
def test_what_may_stand_after_the_role_variable(ins, item):
    assert drafting._after_role_problem(item) is None                    # check C4 by itself
    found = drafting.check_item(ins, "request_templates.style_3", item, {}, [])
    if "{role_name} role in {department}" in item or "{role_name} in {department}" in item \
            or "{role_name}-{department}" in item:
        # since check C8 these are refused: fewer than four words before the department, and no mark that
        # counts (a hyphen is no such mark)
        assert TOO_NEAR in found
    else:
        assert found is None


def test_every_listed_word_may_follow_the_role_variable_and_a_noun_may_not(ins):
    words = drafting.AFTER_ROLE_WORDS
    assert list(words) == sorted(set(words)) and all(word == word.casefold() for word in words)
    for word in words + ("Role", "PLEASE"):
        item = "{person_name} of {department}: {role_name} " + word + " zzz"
        assert drafting.check_item(ins, "request_templates.style_3", item, {}, []) is None, word
    for word in ("access", "permission", "permissions", "rights", "privileges", "needed", "account", "roles",
                 "access_role", "2"):
        item = "{person_name} of {department}: {role_name} " + word + " zzz"
        assert f"the word {word!r} stands directly after {{role_name}}" in drafting.check_item(
            ins, "request_templates.style_3", item, {}, []), word
    # the other two variables have no such rule: no template of the first text was unclear there
    assert drafting.check_item(ins, "request_templates.style_3",
                               "{person_name} access in {department} team for {role_name}", {}, []) is None


def test_the_new_checks_over_the_placeholder_text(ins):
    """The control in the other direction: a text that we wrote to be clear, checked with its own names.
    Its templates, its descriptions and its plain prompt pass every new check. Its two rule prompts fail
    four of them, one after the other: the check of the marker lines (its steps begin with "1." and its
    cases with "a."), then C2 (its step that finds the person speaks of the department and not of the
    returned list, which is the very gap that was measured in the first text), then C6 (its line above the
    cases says "first case" but not "order"), then C7 (the result key of the account state stands
    nowhere), then, since the third calibration, C16 (its step 2 does not say that it is always done) and
    C15 (its case b names no result key). With the markers and the five mends of `good_values` the main
    prompt passes. The prompt of the second desk then fails check C21 (its case c says "for a sensitive
    role and for any other role alike", which is true and which the fixed rule refuses), and passes with
    a sixth mend.
    Its plain prompt has 41 words, one more than check C19 allows; `good_values` cuts its start. (Three of its
    parameter descriptions name another tool, which an older rule of the design refuses; that is not one
    of the new checks, and they are left out here.)"""
    chosen = flat_text(PLACEHOLDER)
    refused: dict = {}
    for key, field in ins.fields.items():
        if not field.asked:
            continue
        others = {name: value for name, value in chosen.items() if name != key}
        if field.kind == "list":
            reasons = [drafting._item_problem(ins, key, item, others, chosen[key][:n])
                       for n, item in enumerate(chosen[key])]
        else:
            reasons = [drafting._value_problem(ins, key, chosen[key], others)]
        refused.update({key: reason for reason in reasons if reason is not None})
    older = sorted(key for key, reason in refused.items() if reason.startswith("names another tool ("))
    assert older == ["tools.check_access.params.person_id.description",
                     "tools.grant_role.params.role_id.description",
                     "tools.request_approval.params.role_id.description"]
    refused = {key: reason for key, reason in refused.items() if key not in older}
    assert refused.pop("plain_system_prompt") == "has 41 words; the short prompt has at most 40 (check C19)"
    plain_others = {name: value for name, value in chosen.items() if name != "plain_system_prompt"}
    assert drafting._value_problem(ins, "plain_system_prompt", mended(chosen["plain_system_prompt"], "C19"),
                                   plain_others) is None
    assert sorted(refused) == ["policy_shift.system_prompt", "system_prompt"]
    assert all("has too few lines that begin with a marker: 0 line(s) begin with 'Step' and a number (at "
               "least 2 are needed) and 0 line(s) begin with 'Case'" in reason for reason in refused.values())
    for key in refused:
        others = {name: value for name, value in chosen.items() if name != key}
        marked = with_markers(chosen[key])
        assert "no line that names find_person, with the line after it, holds the words 'list' and " \
               "'department'" in drafting._value_problem(ins, key, marked, others)
        assert NO_ORDER_LINE in drafting._value_problem(ins, key, mended(marked, "C2"), others)
        assert "does not show where the account state is read: the result key 'account_status' stands " \
               "neither" in drafting._value_problem(ins, key, mended(marked, "C2", "C6"), others)
        assert "(check C16)" in drafting._value_problem(ins, key, mended(marked, "C2", "C6", "C7"), others)
        assert "(check C15)" in drafting._value_problem(
            ins, key, mended(marked, "C2", "C6", "C7", "C16"), others)
        five = mended(marked, "C2", "C6", "C7", "C15", "C16")
        if key == "system_prompt":
            assert drafting._value_problem(ins, key, five, others) is None
        else:
            # the placeholder's result key for a sensitive role is the word "sensitive" itself
            found = drafting._value_problem(ins, key, five, others)
            assert "holds the result key 'sensitive'" in found and "(check C21)" in found
            assert drafting._value_problem(ins, key, mended(five, "C21"), others) is None


def test_the_instructions_state_what_the_new_checks_demand(ins):
    """A writer can follow a check only if its instruction states it: the word lists and the numbers in the
    instruction file are those of the program."""
    flat = {key: " ".join(field.instruction.split()) for key, field in ins.fields.items()}
    after_role = "one of these words: " + ", ".join(drafting.AFTER_ROLE_WORDS) + "."
    for style in (1, 2, 3, 4):
        assert after_role in flat[f"request_templates.style_{style}"]
    plain_words = "one of these words: " + ", ".join(drafting.BEFORE_PARAMETER_WORDS) + "."
    descriptions = [key for key in flat if key.endswith(".description")]
    assert len(descriptions) == 18 and all(plain_words in flat[key] for key in descriptions)
    assert (drafting.LISTING_LINE_TOOLS, drafting.NEGATIONS) == (2, ("not", "never"))
    assert drafting.SECOND_CALL_WORDS == ("twice", "second")
    assert (drafting.LOOKUP_LINE_WORD, drafting.LOOKUP_LINE_STEM) == ("list", "department")
    for key in ("system_prompt", "policy_shift.system_prompt"):
        for stated in (f"The program refuses a prompt of more than {drafting.PROMPT_WORDS[1]} words.",
                       "begin the sentence of each step with the word Step and its number",
                       "the sentence of each case with the word Case and its letter",
                       "Begin the end rules (the last call, and what is done once at most) with the word "
                       "Finally",
                       "The program starts a new line in front of each of these sentences, so you need not "
                       "write line breaks, but you may",
                       "a line that names two or more of the tools that the desk uses, the closing tool not "
                       "counted, is a list of tools",
                       "must stand on at least one line that holds no other resolution value that the desk "
                       "uses",
                       'or the line directly after it, must hold the word "list" and the word "department"'):
            assert stated in flat[key], (key, stated)
    for note in ("Input of that tool -> 1 thing only: full name from request",
                 "Department -> NOT an input of that tool",
                 "person to take -> the one in the list whose department is that department",
                 "Tool of this step -> the one that checks a role",
                 "THEN ask approval, tool for it -> the one that records a request for approval",
                 "THEN give role to person, tool for it -> the one that gives a role",
                 "a sentence at the end that only lists the tool names is not enough"):
        assert note in flat["system_prompt"], note
    for note in ("number of cases at the second desk -> 3, exactly: a, b, c",
                 "Finally line (the end rules) of the second desk -> 3 things, in this order, and nothing else",
                 'every sentence that holds the name must also hold the word "not" or the word "never"',
                 'may hold neither the word "twice" nor the word "second"',
                 "speaks of four cases, of a fourth case or of a case d"):
        assert note in flat["policy_shift.system_prompt"], note
    # checks C6 to C10: the same for the second calibration
    assert drafting.ORDER_LINE_WORDS + (drafting.ORDER_LINE_STEM,) == ("order", "first", "case")
    assert (drafting.ROLE_TO_DEPARTMENT_WORDS, drafting.SAME_RUN_LIMIT) == (4, 5)
    assert drafting.MIN_STEP_LINES == 2
    assert drafting.ACTION_STEMS == ("giv", "grant", "assign")
    assert drafting.PLAIN_PROMPT_STEMS == ("sensitiv", "manager", "approv") + drafting.ACTION_STEMS
    for key in ("system_prompt", "policy_shift.system_prompt"):
        three = (", ".join(f'"{word}"' for word in drafting.ORDER_LINE_WORDS)
                 + f' and "{drafting.ORDER_LINE_STEM}"')
        for stated in ("The program checks these things about the lines",
                       "Check of the markers: at least two lines begin with the word Step and a number, and "
                       "for each case one line begins with the word Case and a letter.",
                       "Fourth check: above the line of the first case, one sentence tells in which order "
                       "the cases are tested and which case the assistant follows (the first one that "
                       "fits). It names no tool.",
                       f"looks for the words {three} in one sentence, or in two sentences that stand one "
                       "directly after the other on one line",
                       "Fifth check: the lines of the cases stand in the order in which the cases are "
                       "tested.",
                       "the program takes the first line that holds it",
                       "Sixth check: the line of the case for a switched-off account holds the account word "
                       "for a switched-off account, and not the account word for an account in use.",
                       "the result key of the account state, letter for letter, stands in the line that "
                       "names the people-finding tool or in the line directly after it",
                       'together with the name of the people-finding tool or with "step 1"'):
            assert stated in flat[key], (key, stated)
    for note in ("What the result holds for each person of the list (a FACT about the result; nothing that "
                 "the assistant writes, lists or shows) -> department; state of account, under its result "
                 "key; id of manager, under its result key",
                 "a. IF account of person: switched off (state of account -> in result of step 1)",
                 "order of testing -> a, b, c, d; first case that fits -> the one to follow",
                 "must stand in the line of step 1",
                 "The other account word may not stand in the line of case a"):
        assert note in flat["system_prompt"], note
    for note in ("line that tells the order of the cases (order of testing; first case that fits) -> stays",
                 "order of testing at the second desk -> a, b, c",
                 "each keeps its line, with its marker and its account word"):
        assert note in flat["policy_shift.system_prompt"], note
    assert ("cut at the underscores, the value must hold a word that begins with one of these: "
            + ", ".join(drafting.PENDING_STEMS) + ".") in flat["values.resolution.pending"]
    for style in (1, 2, 3, 4):
        for stated in ("there must stand at least four words, or one of these marks: "
                       + " ".join(drafting.ROLE_TO_DEPARTMENT_MARKS) + " (a hyphen, a slash",
                       "never stands directly after the variable for the department with only a space "
                       "between them",
                       "shares a run of five or more words in a row with an earlier template of the same "
                       "list. A variable counts as one word"):
            assert stated in flat[f"request_templates.style_{style}"], (style, stated)
    stems = drafting.ACTION_STEMS
    for stated in ("request -> asks: a role for a person (what is asked, not what the desk does about it)",
                   f"No word that begins with {stems[0]}, {stems[1]} or {stems[2]}"):
        assert stated in flat["plain_system_prompt"], stated
    # the instruction of the parameter that was called by a wrong name no longer shows that wrong form
    key = "tools.close_request.params.resolution.description"
    assert "resolution parameter" not in flat[key] and "uses that name, letter for letter" in flat[key]
    assert drafting._parameter_word_problem(flat[key], ["request_resolution"]) is None


def test_the_dry_run_shows_each_new_rule_to_the_writers_of_its_fields_and_every_field_once(script, capsys,
                                                                                           ins):
    assert script.main(["--dry-run", "--run-seed", "3"]) == 0
    blocks = re.split(r"(?m)^=== ", capsys.readouterr().out)[1:]
    assert [block.split()[1] for block in blocks] == list(ins.fields)            # every field, once, in order
    shown = {block.split()[1]: " ".join(block.split()) for block in blocks}

    def holding(text: str) -> list[str]:
        return [key for key, block in shown.items() if text in block]
    assert holding("Third check: the line that names the people-finding tool") == [
        "system_prompt", "policy_shift.system_prompt"]
    assert holding(", ".join(drafting.AFTER_ROLE_WORDS)) == [f"request_templates.style_{n}"
                                                             for n in (1, 2, 3, 4)]
    assert len(holding(", ".join(drafting.BEFORE_PARAMETER_WORDS))) == 18
    assert holding("speaks of four cases") == ["policy_shift.system_prompt"]
    assert holding("The condition in the line of case c is only that no case above fits") == [
        "policy_shift.system_prompt"]                                   # check C21: the second desk only
    for rule in ("begin the sentence of each step with the word Step and its number",
                 "Check of the markers: at least two lines begin with the word Step",
                 "Fourth check: above the line of the first case", "Fifth check: the lines of the cases",
                 "Sixth check: the line of the case for a switched-off account"):
        assert holding(rule) == ["system_prompt", "policy_shift.system_prompt"], rule
    for rule in ("at least four words, or one of these marks", "a run of five or more words in a row"):
        assert holding(rule) == [f"request_templates.style_{n}" for n in (1, 2, 3, 4)], rule
    assert holding("begins with one of these: pend, approv, wait, await") == ["values.resolution.pending"]
    # the BACKGROUND in note form stands in every user message
    assert len(holding("request -> asks: a role for a person; things named in request -> 3")) == sum(
        field.asked for field in ins.fields.values())
    assert holding("No word that begins with giv, grant or assign") == ["plain_system_prompt"]
    assert f"seed of candidate 0: {candidate_seed(3, 'system_prompt', 0)}" in shown["system_prompt"]


# ---------------------------------------------------------------------------
# The five checks added after the second calibration (C6 to C10 in accessdesk/drafting.py)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def round2() -> dict:
    """Every value of the round-2 text, as dotted key -> value."""
    return flat_text(ROUND2_DATA)


# Hand-made lines of a rule prompt, with the visible names of the first text (`present`).
ORDER_LINE = "Test the cases in this order and follow the first case that fits."
CASE_A = "a. If account_state in the Step 1 result is inactive, call close_request with request_denied."
CASE_B = "b. If holds_role is true, call close_request with role_already_held."
CASE_C = "c. If is_sensitive is true, call request_approval, then close_request with request_closing."
CASE_D = "d. In every other case call grant_role, then close_request with role_granted."
CASES = [CASE_A, CASE_B, CASE_C, CASE_D]
NO_ORDER_LINE = "above the first line with a resolution value, no sentence that names no tool"
WRONG_ORDER = "its cases do not stand in the order in which they are tested"
NO_SOURCE = "does not show where the account state is read: the result key 'account_state' stands neither"
NOT_OFF = "does not hold the account word 'inactive'"
IN_USE = "holds the account word 'active', which is the word for an account in use"
TOO_NEAR = "none of the marks . , ; : ( ) ? ! | stand between {role_name} and {department}"
BY_C10 = "which tells what the desk does with a request (check C10)"


def test_each_fault_of_the_round_2_text_is_refused_by_its_check(ins, round2):
    # C6, part 1: both rule prompts
    for key in ("system_prompt", "policy_shift.system_prompt"):
        assert NO_ORDER_LINE in problem_of(ins, round2, key, round2[key])
    # C7: C6 stands before it in the order of the checks, so it is called by itself
    for key in ("system_prompt", "policy_shift.system_prompt"):
        assert NO_SOURCE in drafting._account_source_problem(prompt_lines(round2[key]), round2)
    # C8: the first template of the third manner
    style_3 = round2["request_templates.style_3"]
    assert style_3[0] == "hey hi, can you please give {person_name} the {role_name} in {department}?"
    assert problem_of(ins, round2, "request_templates.style_3", style_3[0]).startswith(
        "only 1 word(s) ('in') and " + TOO_NEAR)
    for item in style_3[1:]:
        assert problem_of(ins, round2, "request_templates.style_3", item) is None
    # C9: the second template of the fourth manner, after the first was accepted
    key = "request_templates.style_4"
    first, second, third = round2[key]
    others = {name: value for name, value in round2.items() if name != key}
    assert drafting.check_item(ins, key, first, others, []) is None
    assert drafting.check_item(ins, key, second, others, [first]) == (
        "is nearly the same sentence as a template that was accepted before it: the two share a run of 6 "
        "words ('i need the {role_name} role to'); the limit is 5")
    # the two share a second run, of five words: their start
    assert drafting.words_of(first)[:5] == drafting.words_of(second)[:5] == [
        "i", "am", "{person_name}", "from", "{department}"]
    assert drafting.check_item(ins, key, third, others, [first]) is None
    # C10: the plain prompt
    assert "one request means giving a role to a person" in round2["plain_system_prompt"]
    assert problem_of(ins, round2, "plain_system_prompt", round2["plain_system_prompt"]) == (
        "holds a word that begins with 'giv', " + BY_C10)


def test_the_round_2_prompts_pass_after_three_small_repairs(ins, round2):
    """A line on the order above case a, and the source of the account state in case a: once as the name
    of the people-finding tool, once as "Step 1". And, since the third calibration, a word in step 2 which
    says that the step is always done (check C16). Nothing else of the round-2 prompts is refused: they
    named the result keys in cases b and c and said "Then call" in front of the closing call."""
    for key, source in (("system_prompt", "lookup_person"), ("policy_shift.system_prompt", "Step 1")):
        prompt = round2[key]
        assert prompt.count("\n\nCase a: If the account_state is inactive") == 1
        with_order = prompt.replace("\n\nCase a:", "\n\n" + ORDER_LINE + "\n\nCase a:")
        with_both = with_order.replace("If the account_state is inactive",
                                       f"If the account_state in the {source} result is inactive")
        assert NO_ORDER_LINE in problem_of(ins, round2, key, prompt)
        assert NO_SOURCE in problem_of(ins, round2, key, with_order)
        assert "(check C16)" in problem_of(ins, round2, key, with_both)
        assert with_both.count("Step 2: Call check_access") == 1
        with_all = with_both.replace("Step 2: Call check_access", "Step 2: Always call check_access")
        assert problem_of(ins, round2, key, with_all) is None


def test_the_first_prompts_and_the_checks_of_the_second_calibration(present):
    """The round-1 prompts had the sentence on the order, but inside the paragraph of the cases; and their
    case a said "account state" in two words, with no source."""
    for prompt, ruleset in ((PRESENT_MAIN, "default"), (PRESENT_SHIFT, "all_grants_need_approval")):
        assert "act on the first case that fits" in prompt
        assert NO_ORDER_LINE in drafting._case_order_problem(prompt_lines(prompt), present, ruleset)
        lines = prompt_lines(repaired(prompt, "case lines"))
        assert drafting._case_order_problem(lines, present, ruleset) is None
        assert NO_SOURCE in drafting._account_source_problem(lines, present)
        lines = prompt_lines(repaired(prompt, "case lines", "account source"))
        assert drafting._account_source_problem(lines, present) is None


def test_the_order_of_the_cases_is_the_order_in_which_the_rules_test_the_facts():
    from accessdesk.env.rules import RESOLUTION_OF_ACTION, RULESETS, decide
    # (active, holds the role, sensitive): the facts in which a case and every later case fit
    facts = [(False, True, True), (True, True, True), (True, False, True), (True, False, False)]
    assert tuple(RESOLUTION_OF_ACTION[decide(*row)] for row in facts) == drafting.CASE_ORDER
    for ruleset in RULESETS:                           # and under each rule set the same order, with gaps
        found = list(dict.fromkeys(RESOLUTION_OF_ACTION[decide(*row, ruleset)] for row in facts))
        assert found == [key for key in drafting.CASE_ORDER if key in drafting.ruleset_resolutions(ruleset)]


@pytest.mark.parametrize("lines, ruleset, reason", [
    ([ORDER_LINE] + CASES, "default", None),
    (["The ORDER of the Cases counts: the First one that fits wins."] + CASES, "default", None),
    (CASES + [ORDER_LINE], "default", NO_ORDER_LINE),                              # after the cases
    ([CASE_A + " " + ORDER_LINE] + CASES[1:], "default", NO_ORDER_LINE),           # inside the line of a case
    (["Test the cases in this order.", "Follow the first case that fits."] + CASES, "default",
     NO_ORDER_LINE),                                                               # spread over two lines
    (["Work in an orderly way: the first case that fits wins."] + CASES, "default", NO_ORDER_LINE),
    (["Keep to this order: what fits first is followed."] + CASES, "default", NO_ORDER_LINE),
    ([ORDER_LINE, CASE_B, CASE_A, CASE_C, CASE_D], "default",
     WRONG_ORDER + ": the first line that holds 'role_already_held' does not stand below the first line "
     "that holds 'request_denied'"),
    ([ORDER_LINE, CASE_A, CASE_B, CASE_D, CASE_C], "default",
     "the first line that holds 'role_granted' does not stand below the first line that holds "
     "'request_closing'"),
    # a desk that never gives a role: the value of a given role is not counted, as in check C1
    ([ORDER_LINE, CASE_A, CASE_B, CASE_D, CASE_C], "all_grants_need_approval", None),
    ([ORDER_LINE, CASE_A, CASE_B, CASE_C], "all_grants_need_approval", None),
    ([ORDER_LINE, CASE_A, CASE_C, CASE_B], "all_grants_need_approval",
     "the first line that holds 'request_closing' does not stand below the first line that holds "
     "'role_already_held'"),
    # a line above the cases that lists the values: it is the first line of every value
    ([ORDER_LINE, "The values: request_denied, role_already_held, request_closing, role_granted."] + CASES,
     "default", WRONG_ORDER),
])
def test_the_order_of_the_cases_is_told_above_them_and_kept(present, lines, ruleset, reason):
    found = drafting._case_order_problem(lines, present, ruleset)
    assert found is None if reason is None else reason in found, found


STEP_1 = "Step 1: Call lookup_person with the full name. It returns a list; the department decides."
STEP_1_KEY = STEP_1 + " Each person is shown with the department, the account_state and the manager_id."
STEP_2 = "Step 2: Call check_access with the person id and the role name."
WORDS_A = "Case a: If the account state is inactive, close with request_denied."


@pytest.mark.parametrize("lines, reason", [
    # part (ii), place A: the key stands where the people-finding tool is named, so case a may speak of
    # the account state in words (the example of the task)
    ([STEP_1_KEY, STEP_2, WORDS_A], None),
    ([STEP_1, "You read the account_state of each person there.", STEP_2, WORDS_A], None),  # the line after
    ([STEP_1, STEP_2, "You read the account_state in step 1.", WORDS_A], NO_SOURCE),   # two lines after it
    # place B: the key in the line of case a, with a source on that line
    ([STEP_1, STEP_2, CASE_A], None),                                              # the key and "Step 1"
    ([STEP_1, STEP_2, "a. If account_state from step one is inactive, close with request_denied."], None),
    ([STEP_1, STEP_2, "a. If account_state (STEP  1) is inactive, close with request_denied."], None),
    ([STEP_1, STEP_2, "a. If account_state in the lookup_person result is inactive, close with "
      "request_denied."], None),
    # the fault of round 2: the key in case a with no source. And no key at all
    ([STEP_1, STEP_2, "a. If account_state is inactive, call close_request with request_denied."],
     NO_SOURCE),
    ([STEP_1, STEP_2, WORDS_A], NO_SOURCE),
    ([STEP_1, STEP_2, "a. If Account_State in the Step 1 result is inactive, close with request_denied."],
     NO_SOURCE),
    ([STEP_1, STEP_2, "a. If account_state from step 2 is inactive, close with request_denied."],
     NO_SOURCE),
    ([STEP_1, STEP_2, "a. If account_state from steps 1 and 2 is inactive, close with request_denied."],
     NO_SOURCE),
    ([STEP_1, STEP_2, "a. If account_state from step 10 is inactive, close with request_denied."],
     NO_SOURCE),
    ([STEP_1, STEP_2, "a. If account_state from Lookup_Person is inactive, close with request_denied."],
     NO_SOURCE),
    # part (i): the account word. The reviewer's example holds the word for an account in use
    ([STEP_1_KEY, STEP_2, "a. If account_state in the Step 1 result is active, call close_request with "
      "request_denied."], NOT_OFF),
    ([STEP_1_KEY, STEP_2, "a. If the account is not Active but inactive, close with request_denied."],
     IN_USE),
    ([STEP_1_KEY, STEP_2, "a. If the account is INACTIVE, close with request_denied."], None),
    ([STEP_1_KEY, STEP_2, "a. If the account is switched off, close with request_denied."], NOT_OFF),
    ([STEP_1_KEY, STEP_2, "a. If the account is inactivated, close with request_denied."], NOT_OFF),
])
def test_the_case_of_the_switched_off_account_says_where_the_state_is_read(present, lines, reason):
    # only the first line that holds the value is read as the case: a line below it does not count
    lines = lines + [CASE_B, "Again: request_denied is for an inactive account_state from Step 1."]
    found = drafting._account_source_problem(lines, present)
    assert found is None if reason is None else reason in found, found


def test_check_c7_reads_the_account_words_and_the_key_of_the_text(present):
    """The words and the key are the chosen values, not fixed English words."""
    other = {**present, "values.status.inactive": "disabled", "values.status.active": "enabled",
             "result_keys.status": "acct"}
    lines = [STEP_1.replace("decides.", "decides. It shows acct."), STEP_2,
             "Case a: If the account is disabled, close with request_denied."]
    assert drafting._account_source_problem(lines, other) is None
    assert "does not hold the account word 'disabled'" in drafting._account_source_problem(
        [lines[0], STEP_2, WORDS_A], other)
    assert "holds the account word 'enabled'" in drafting._account_source_problem(
        lines[:2] + ["Case a: If the account is disabled, not enabled, close with request_denied."], other)


def test_a_case_line_with_the_people_finding_tool_and_the_closing_tool_is_no_list_of_tools(present):
    """Check C1 does not count the closing tool, so check C7 can be met with the tool's name."""
    line = "a. If account_state in the lookup_person result is inactive, call close_request, request_denied."
    rest = ["2. Call check_access.", "c. Call request_approval.", "d. Call grant_role."]
    assert drafting._tool_line_problem([line] + rest, present, "default") is None
    listing = line.replace("call close_request", "call check_access and close_request")
    assert "names the tool lookup_person only in a list of tools" in drafting._tool_line_problem(
        [listing] + rest, present, "default")


@pytest.mark.parametrize("item, reason", [
    ("hey, give {person_name} the {role_name} in {department}", "only 1 word(s) ('in')"),
    ("hey, give {person_name} the {role_name} role in {department}", "only 2 word(s) ('role in')"),
    ("hey, give {person_name} the {role_name} role over in {department}", "only 3 word(s) ('role over in')"),
    # a template that run seed 4 took (text/drafting_log.jsonl): three words
    ("hey hi, can you please help {person_name} get the {role_name} role in the {department}?",
     "only 3 word(s) ('role in the')"),
    # a kept template of the text: four words
    ("Kindly provide {person_name} with the {role_name} while they work in {department}.", None),
    ("hey, give {person_name} the {role_name}, in {department}", None),                   # a mark that counts
    ("hey, give {person_name} the {role_name} (in {department})", None),
    ("{role_name} | {department} | {person_name}", None),
    ("is it {role_name}? {department}! {person_name}.", None),
    # marks that do not count: a quotation mark, a hyphen, a slash, an "and" sign (the reviewer's examples)
    ('hey, give {person_name} the "{role_name}" in {department}', "only 1 word(s) ('in')"),
    ("{role_name} - {department} - {person_name}", "only 0 word(s) ('')"),
    ("give {person_name} the {role_name}-{department} role", "only 0 word(s) ('')"),
    ("the {role_name}/{department} please, for {person_name}", "only 0 word(s) ('')"),
    ("the {role_name} & {department} thing for {person_name}", "only 0 word(s) ('')"),
    ("the {role_name} for {person_name} in {department}", None),               # {person_name} between them
    ("{person_name} in {department} needs the {role_name}", None),             # the department comes first
    ("{role_name} in {department}: {person_name}", "only 1 word(s) ('in')"),              # a mark after it
    # an apostrophe or a hyphen inside a word: one word
    ("{role_name} for HR's {department}, for {person_name}", "only 2 word(s) ('for HRs')"),
    ("{role_name} in-house {department}, for {person_name}", "only 1 word(s) ('inhouse')"),
])
def test_the_department_does_not_stand_right_after_the_role_name(ins, item, reason):
    found = drafting._department_after_role_problem(item)
    assert found is None if reason is None else found.startswith(f"{reason} and {TOO_NEAR}"), found
    assert drafting.check_item(ins, "request_templates.style_3", item, {}, []) == found
    if found is not None:
        assert found.endswith("at least 4 words or one of these marks must stand there")


@pytest.mark.parametrize("item, refused", [
    ("give {person_name} of {department} {role_name} now", True),              # the reviewer's example
    ("{person_name}: {department}{role_name}", True),
    ("give {person_name} of {department} the {role_name} now", False),
    ("{person_name} ({department}) {role_name}", False),
    ("{department}, {role_name}, {person_name}", False),
])
def test_the_role_name_does_not_stand_right_after_the_department(ins, item, refused):
    """Check C8, the mirrored order: 6 of the 14 role names of the text begin with a department word, so
    "{department} {role_name}" can read as one name ("Finance Finance Ledger Entry")."""
    found = drafting.check_item(ins, "request_templates.style_3", item, {}, [])
    if refused:
        assert found.startswith("{role_name} stands directly after {department}"), found
        assert drafting._after_role_problem(item) is None and drafting._department_after_role_problem(
            item) is None                                    # no older check refuses it
    else:
        assert found is None
    roles, departments = ROUND2_DATA["roles"], ROUND2_DATA["departments"]
    assert sum(any(role.startswith(name.split()[0] + " ") for name in departments) for role in roles) == 6


def test_two_templates_of_one_field_are_not_nearly_the_same_sentence(ins):
    key = "request_templates.style_4"
    first = "I am {person_name} from {department} and need {role_name}."
    four = "Hello, I am {person_name} from the team {department}. My work needs {role_name} now."
    five = "I am {person_name} from {department}; my work needs {role_name} today."
    loud = "I AM {person_name}, FROM {department}! My work needs {role_name} today."
    assert drafting.check_item(ins, key, four, {}, [first]) is None                 # a run of four words
    for item in (five, loud):                                  # five; capitals and marks make no difference
        assert drafting.check_item(ins, key, item, {}, [first]) == (
            "is nearly the same sentence as a template that was accepted before it: the two share a run "
            "of 5 words ('i am {person_name} from {department}'); the limit is 5")
    # a variable is one word, and it is not the plain word
    assert drafting._same_run_problem("i am person_name from department x {person_name} {department} "
                                      "{role_name}", [first]) is None
    # only the templates of the same field that were accepted before it are compared
    assert drafting.check_item(ins, key, five, {"request_templates.style_1": [first]}, []) is None
    assert drafting.check_item(ins, key, five, {}, []) is None
    # a run may go over the end of a sentence: "{department}. My work needs {role_name}"
    assert "a run of 5 words ('{department} my work needs {role_name}')" in drafting.check_item(
        ins, key, five, {}, [four])


def test_the_kept_templates_of_the_present_text_pass_the_two_new_template_checks(ins):
    data = flat_text(yaml.safe_load(text_module.model_text_path().read_text(encoding="utf-8")))
    kept = "Kindly provide {person_name} with the {role_name} while they work in {department}."
    assert kept in data["request_templates.style_1"]
    for key in ("request_templates.style_1", "request_templates.style_2"):
        for n, item in enumerate(data[key]):
            assert drafting._department_after_role_problem(item) is None, item
            assert drafting._same_run_problem(item, data[key][:n]) is None, item


HARMLESS_PLAIN = ("You work at an IT desk. Each conversation is one request: somebody asks for a role for a "
                  "person. Settle it with your tools, one tool call in each message. Use an id only if a "
                  "tool result shows it.")


@pytest.mark.parametrize("sentence, stem", [
    ("One request means giving a role to a person.", "giv"),
    ("A role is Given on request.", "giv"),
    ("Access is granted on request.", "grant"),
    ("Each request is about the assignment of a role.", "assign"),
    ("You assign roles.", "assign"),
    ("He gave a gift. Forgive the unassigned and the regranted; have no misgivings.", None),
    ("The person gets, receives or needs the role.", None),
])
def test_the_plain_prompt_does_not_tell_what_the_desk_does_with_a_request(round2, sentence, stem):
    assert drafting._plain_prompt_problem(HARMLESS_PLAIN, round2) is None
    found = drafting._plain_prompt_problem(f"{HARMLESS_PLAIN} {sentence}", round2)
    assert found == (None if stem is None else f"holds a word that begins with {stem!r}, {BY_C10}")


# ---------------------------------------------------------------------------
# Kept values after a change of the checks or of the instruction file
# ---------------------------------------------------------------------------
def test_kept_values_are_measured_against_the_checks_as_they_are_now(ins, good):
    chosen = draft(ins)[0].chosen
    assert drafting.kept_problems(ins, chosen, list(ins.fields)) == {}        # every good value passes
    template = "please get {person_name} of {department} the {role_name} access"
    description = "The resolution parameter says how the request ended."
    listed, described = "request_templates.style_1", "tools.close_request.params.resolution.description"
    changed = {**chosen, listed: [template] + good[listed][1:], described: description}
    found = drafting.kept_problems(ins, changed, list(ins.fields))
    assert found == {
        described: "calls a parameter 'resolution', which is not its visible name (outcome)",
        listed: f"{template!r} the word 'access' stands directly after {{role_name}}, where a reader could "
                "take it for a part of the role's name"}
    # only the named fields are measured, and a fixed value is never measured
    assert drafting.kept_problems(ins, changed, ["roles", "tools.grant_role.name", "departments"]) == {}
    header = drafting.header_comment(
        ins, model_id=MODEL, revision=REVISION, written_at_utc="2026-11-17 10:00:00 UTC", run_seed=3,
        n_candidates=N, sampling="thinking off", instructions_file="text/drafting_instructions.yaml",
        log_file="text/drafting_log.jsonl", report=drafting.log_report(ins, []),
        kept={"run_seeds": [1, 2], "instructions_sha256": ["0" * 64]}, written=["roles"], refused_now=found)
    assert "this line is a report): 2 kept value(s) would be refused:\n" in header
    assert f"#     {described}: calls a parameter 'resolution'" in header
    assert ("#   Fields that the run named above wrote (every other value that the model wrote is a\n"
            "#   kept value):\n#     roles\n") in header


def test_only_after_a_change_of_the_instruction_file_states_both_hashes_plainly(script, capsys, ins,
                                                                                tmp_path):
    """The next real run: three fields are named, the program writes six, and 58 values written by the
    model are kept from a run under an earlier instruction file."""
    earlier = tmp_path / "earlier_instructions.yaml"
    earlier.write_bytes(drafting.INSTRUCTIONS_PATH.read_bytes() + b"# this line was removed later\n")
    old_sha = drafting.load_instructions(earlier).sha256
    assert old_sha != ins.sha256
    assert run_script(script, capsys, FakeWriter(ins), tmp_path, "--instructions", str(earlier))[0] == 0
    first = (tmp_path / "out.yaml").read_text(encoding="utf-8")
    assert old_sha in first and ins.sha256 not in first and "Kept values" not in first

    only = ["system_prompt", "tools.close_request.params.resolution.description", "request_templates.style_3"]
    wrote = ["tools.close_request.params.resolution.description", "system_prompt",
             "policy_shift.system_prompt", "plain_system_prompt", "request_templates.style_3",
             "request_templates.style_4"]
    assert drafting.fields_to_write(ins, only) == wrote
    writer = FakeWriter(ins, run_seed=3)
    code, out, _ = run_script(script, capsys, writer, tmp_path, "--run-seed", "3", "--only", *only)
    assert code == 0 and out.startswith("6 fields to write")
    assert [call[0] for call in writer.calls] == wrote
    header = (tmp_path / "out.yaml").read_text(encoding="utf-8").split("\nmeta:")[0]
    for stated in (
        "# Run seed: 3\n",
        f"(SHA-256 {ins.sha256}).\n",
        "Their run seeds: [1].\n#   SHA-256 of their instruction files:\n" + f"#     {old_sha}\n"
        "#   The instruction file was changed between runs: a hash in this list differs from the\n"
        f"#   hash at 'Instructions' above ({ins.sha256}).\n"
        "#   The kept values were written under the files of this list.",
        "shown for each kept value stands in the 'messages' of its line in " + str(tmp_path / "log.jsonl"),
        "#   Fields that the run named above wrote (every other value that the model wrote is a\n"
        "#   kept value):\n" + "".join(f"#     {key}\n" for key in wrote)
        + "#   Kept values and the checks as they are now",
        "this line is a report): every kept value passes them.\n",
    ):
        assert stated in header, stated
    assert header.count(old_sha) == 1 and header.count(ins.sha256) == 2
    runs = [line for line in log_lines(tmp_path) if line["kind"] == "run"]
    assert [run["instructions_sha256"] for run in runs] == [old_sha, ins.sha256]
    assert sum(line["kind"] == "kept" for line in log_lines(tmp_path)) == 75 - 6 - len(wrote) == 63
    assert header_draws(tmp_path / "out.yaml") == {
        key: 2 if key in wrote else 1 for key, field in ins.fields.items() if field.asked}


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------
def test_the_log_holds_every_candidate_with_everything_about_it(ins, good):
    key = "errors.unknown_department"
    replies = {(key, 0): '"No department {name}."', (key, 1): "```\nNo department has the name {value}.\n```"}
    result, lines, writer = draft(ins, replies)
    lines = json.loads(json.dumps(lines))                            # every line can be written as JSON
    logged = [line for line in lines if line["kind"] == "candidate"]
    assert [(line["field"], line["index"]) for line in logged] == writer.calls
    assert len(logged) == 64 + 1
    for line in logged:
        assert set(line) >= {"field", "index", "seed", "writer", "max_new_tokens", "messages", "raw",
                             "cut_off", "cleaned", "valid", "reason", "taken", "reused", "shared_run"}
        assert line["messages"] == drafting.build_messages(ins, line["field"], result.chosen)
        assert line["max_new_tokens"] == drafting.MAX_NEW_TOKENS[ins.fields[line["field"]].kind]
        assert line["writer"] == "fake@0" and line["reused"] is False
    first, second = candidates(lines, key)
    assert (first["raw"], first["cleaned"], first["valid"], first["taken"]) == (
        '"No department {name}."', "No department {name}.", False, False)
    assert first["reason"] == "holds braces other than {value}"
    assert (second["cleaned"], second["valid"], second["reason"], second["taken"]) == (
        "No department has the name {value}.", True, None, True)
    # every field is in the log once as fixed, copied, or with its candidates
    kinds = {line["field"]: line["kind"] for line in lines}
    assert kinds == {key: {"fixed": "fixed", "copy": "copied"}.get(field.kind, "candidate")
                     for key, field in ins.fields.items()}
    fixed = next(line for line in lines if line["field"] == "tools.grant_role.name")
    assert fixed["value"] == "grant_role" and "brief" in fixed["source"]
    copied = next(line for line in lines if line["field"] == "tools.grant_role.params.person_id.name")
    assert (copied["value"], copied["from"]) == ("person_id", "tools.check_access.params.person_id.name")


def test_values_and_replies_can_be_read_back_from_a_log(ins, good):
    key = "departments"
    first_run = [{"kind": "run"}] + draft(ins, {(key, 0): good[key][:5], (key, 1): good[key][5:]})[1]
    assert drafting.previous_values(first_run) == draft(ins)[0].chosen
    cache = drafting.reply_cache(first_run)
    assert len(cache) == 64 + 1
    # a second run in the same log that writes one field again replaces its value ...
    again = "tools.department_info.name"
    line = {"kind": "candidate", "field": again, "taken": True, "cleaned": "people_per_department"}
    values = drafting.previous_values(first_run + [{"kind": "run"}, line])
    assert values[again] == "people_per_department" and values[key] == good[key]
    # ... and a second run that found nothing leaves the field without a value
    line = {"kind": "candidate", "field": again, "taken": False, "cleaned": "?"}
    assert again not in drafting.previous_values(first_run + [{"kind": "run"}, line])
    assert drafting.read_log(Path("no-such-folder") / "log.jsonl") == []


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------
def run_script(script, capsys, writer, tmp_path: Path, *extra: str) -> tuple[int, str, str]:
    argv = ["--model", MODEL, "--revision", REVISION, "--run-seed", "1", "--out", str(tmp_path / "out.yaml"),
            "--log", str(tmp_path / "log.jsonl"), *extra]
    code = script.main(argv, writer=writer)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def log_lines(tmp_path: Path) -> list[dict]:
    return drafting.read_log(tmp_path / "log.jsonl")


def test_the_script_writes_the_file_and_the_log(script, capsys, ins, tmp_path):
    writer = FakeWriter(ins, {("id_prefixes.role", 0): "role id"})
    code, out, err = run_script(script, capsys, writer, tmp_path)
    assert (code, err) == (0, "")
    assert "candidate 0 of id_prefixes.role: refused: is not a short id prefix" in out
    assert "candidate 1 of id_prefixes.role: taken" in out
    written = tmp_path / "out.yaml"
    assert check_text(written) == []
    bundle = load_text(written)
    assert not bundle.is_placeholder and bundle.id_prefix_role == "rol_"
    raw = written.read_text(encoding="utf-8")
    meta = yaml.safe_load(raw)["meta"]
    assert MODEL in meta["written_by"] and REVISION in meta["written_by"] and meta["placeholder"] is False
    assert f"# Written by: {MODEL}, revision {REVISION}" in raw
    assert f"SHA-256 {ins.sha256}" in raw and "# Run seed: 1" in raw
    assert "temperature 0.7, top_p 0.8, top_k 20" in raw and "thinking off" in raw
    assert re.search(r"# Date and time \(UTC\): 20\d\d-\d\d-\d\d \d\d:\d\d:\d\d UTC\n", raw)
    assert "# Read and approved by: (not yet)" in raw and "Kept values" not in raw
    lines = log_lines(tmp_path)
    assert [line["kind"] for line in lines][0] == "run" and lines[-1]["kind"] == "written"
    assert lines[0]["instructions_sha256"] == ins.sha256
    assert (lines[0]["model"], lines[0]["revision"], lines[0]["run_seed"], lines[0]["candidates"]) == (
        MODEL, REVISION, 1, 8)
    assert sum(line["kind"] == "candidate" for line in lines) == len(writer.calls) == 65
    assert all(line["writer"] == f"{MODEL}@{REVISION}" for line in lines if line["kind"] == "candidate")
    assert lines[-1]["sha256"] == __import__("hashlib").sha256(written.read_bytes()).hexdigest()


def table_of_draws(out: str) -> dict[str, int]:
    """The table that the script prints: field -> number of runs in which a value for it was drawn."""
    printed = out.split("runs in which a value was drawn, for each field:\n", 1)[1]
    rows = [line.split() for line in printed.split("\nfields drawn in more than one run:")[0].splitlines()]
    assert all(len(row) == 2 for row in rows)
    return {key: int(count) for key, count in rows}


def header_draws(path: Path) -> dict[str, int]:
    """The same table as it stands in the header of a written text file."""
    header = path.read_text(encoding="utf-8").split("\nmeta:")[0]
    table = header.split("was written again):\n", 1)[1].split("#\n# Read and approved by:")[0]
    rows = [line.removeprefix("#   ").split(": ") for line in table.splitlines()]
    return {key: int(count) for key, count in rows}


def test_the_script_reports_the_shared_runs_in_the_header_the_log_and_the_summary(script, capsys, ins,
                                                                                  tmp_path):
    key = "tools.grant_role.description"
    words = drafting.words_of(ins.fields[key].instruction)
    seven, eight = (" ".join(words[:n]) + " zzz" for n in (7, 8))
    code, out, _ = run_script(script, capsys, FakeWriter(ins, {(key, 0): eight, (key, 1): seven}), tmp_path)
    # code 0: the placeholder measurement is a report and not a rule. The fake writer's values ARE the
    # placeholder's texts, and the file is written all the same.
    assert code == 0
    assert f"candidate 0 of {key}: refused: copies the instruction: it shares a run of 8 words" in out
    run = f'7 words in {key} ("{" ".join(words[:7])}")'
    header = (tmp_path / "out.yaml").read_text(encoding="utf-8").split("\nmeta:")[0]
    assert f"# Largest shared run among the values that were taken: {run}\n" in header
    assert ("longest run of words shared with the shown text, among the taken values (limit 8): "
            f"{run}\n") in out
    lines = log_lines(tmp_path)
    last = lines[-1]
    assert last["kind"] == "written"
    assert last["shared_run"] == {"field": key, "length": 7, "words": " ".join(words[:7])}
    # the placeholder: in the last line of the log and in the printed summary, and not in the header
    expected = drafting.placeholder_report(ins, drafting.previous_values(lines))
    assert last["placeholder_run"] == expected and expected["length"] > 30
    assert expected["field"] in ("system_prompt", "policy_shift.system_prompt", "plain_system_prompt")
    assert ("longest run of words shared with the placeholder text file (a report, not a rule): "
            f"{expected['length']} words in {expected['field']} (\"{expected['words']}\")\n") in out
    assert "placeholder" not in header
    assert all("shared_run" in line for line in lines if line["kind"] == "candidate")


def test_the_header_and_the_summary_say_in_how_many_runs_each_field_was_drawn(script, capsys, ins, tmp_path):
    asked = [key for key, field in ins.fields.items() if field.asked]
    code, out, _ = run_script(script, capsys, FakeWriter(ins), tmp_path)
    assert code == 0
    assert header_draws(tmp_path / "out.yaml") == table_of_draws(out) == {key: 1 for key in asked}
    assert list(header_draws(tmp_path / "out.yaml")) == asked           # fixed and copied fields: not there
    assert "fields drawn in more than one run: none\n" in out
    # a field that is written again, and again, cannot hide it: the count goes up with every run
    key = "values.confirmation.role_granted"
    todo = [name for name in drafting.fields_to_write(ins, [key]) if ins.fields[name].asked]
    assert todo == [key, "values.confirmation.request_closed"]
    for n_runs, value in ((2, "role now held"), (3, "the role was added")):
        code, out, _ = run_script(script, capsys, FakeWriter(ins, {(key, 0): value}), tmp_path, "--only", key)
        assert code == 0 and load_text(tmp_path / "out.yaml").confirmation["role_granted"] == value
        wanted = {name: n_runs if name in todo else 1 for name in asked}
        assert header_draws(tmp_path / "out.yaml") == table_of_draws(out) == wanted
        assert f"fields drawn in more than one run: {todo}\n" in out
    assert drafting.draw_counts(log_lines(tmp_path))[key] == 3


def test_the_script_writes_no_file_when_a_field_has_no_valid_candidate(script, capsys, ins, tmp_path):
    key = "values.confirmation.request_closed"
    writer = FakeWriter(ins, {(key, i): "" for i in range(N)})
    code, out, _ = run_script(script, capsys, writer, tmp_path)
    assert code == 1
    assert "NOT WRITTEN" in out and key in out
    assert f"FAILED    {key}: no valid candidate among 8" in out
    assert not (tmp_path / "out.yaml").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["log.jsonl"]
    last = log_lines(tmp_path)[-1]
    assert last["kind"] == "not_written" and list(last["failed"]) == [key]
    # the reports are made for a run that wrote nothing too
    assert last["placeholder_run"]["length"] > 30 and last["shared_run"]["length"] == 5
    assert "shared with the placeholder text file (a report, not a rule): " in out
    assert table_of_draws(out)[key] == 1 and "fields drawn in more than one run: none" in out


def test_the_script_does_not_overwrite_a_text_file_with_a_refused_one(script, capsys, ins, tmp_path,
                                                                      monkeypatch):
    """Every field has a value, but the whole file does not pass check_text: nothing is written."""
    (tmp_path / "out.yaml").write_text("old", encoding="utf-8")
    monkeypatch.setattr(drafting, "check_text", lambda path: ["a made-up problem"])
    code, out, _ = run_script(script, capsys, FakeWriter(ins), tmp_path)
    assert code == 1 and "a made-up problem" in out
    assert (tmp_path / "out.yaml").read_text(encoding="utf-8") == "old"
    last = log_lines(tmp_path)[-1]
    assert {key: last[key] for key in ("kind", "out", "problems")} == {
        "kind": "not_written", "out": str(tmp_path / "out.yaml"), "problems": ["a made-up problem"]}
    assert set(last) == {"kind", "out", "problems", "shared_run", "placeholder_run"}


def test_dry_run_prints_every_field_once_and_needs_no_model(script, capsys, ins, tmp_path):
    def no_writer(*args):
        raise AssertionError("the dry run must not ask a writer")
    code, out, err = run_script(script, capsys, no_writer, tmp_path, "--dry-run")
    assert (code, err) == (0, "")
    heads = [line for line in out.splitlines() if line.startswith("=== ")]
    assert [head.split()[2] for head in heads] == list(ins.fields)
    assert [head.split()[1] for head in heads] == [f"{n}/75" for n in range(1, 76)]
    assert out.count("--- system message ---") == out.count("--- user message ---") == 64
    assert sum("(fixed, not asked)" in head for head in heads) == 6
    assert sum("(copied by the program, not asked)" in head for head in heads) == 5
    for key, field in ins.fields.items():
        if field.asked:
            assert out.count(field.instruction) == 1, key           # each instruction is printed, once
    assert "<<values.resolution.granted>>" in out                     # stand-ins for earlier values
    assert f"seed of candidate 0: {candidate_seed(1, 'roles', 0)}" in out
    assert list(tmp_path.iterdir()) == []                             # nothing is written
    assert "torch" not in sys.modules


def test_the_script_needs_no_arguments_for_a_dry_run(script, capsys):
    assert script.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("=== ") == 75
    assert (script.DEFAULT_MODEL, script.DEFAULT_REVISION) == ("Qwen/Qwen3.5-9B",
                                                               "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    assert script.SAMPLING == {"enable_thinking": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def test_an_existing_log_stops_a_plain_run(script, capsys, ins, tmp_path):
    assert run_script(script, capsys, FakeWriter(ins), tmp_path)[0] == 0
    writer = FakeWriter(ins)
    code, _, err = run_script(script, capsys, writer, tmp_path)
    assert code == 2 and "--resume" in err and writer.calls == []


def test_resume_asks_only_for_what_the_log_does_not_hold(script, capsys, ins, tmp_path):
    class Stop(Exception):
        pass
    key = "values.status.active"
    first = FakeWriter(ins, {(key, 0): "not valid", (key, 1): Stop("the machine went away")})
    with pytest.raises(Stop):
        run_script(script, capsys, first, tmp_path)
    capsys.readouterr()
    assert not (tmp_path / "out.yaml").exists()
    n_done = len(first.calls) - 1                                     # the last call did not come back
    assert sum(line["kind"] == "candidate" for line in log_lines(tmp_path)) == n_done > 20

    second = FakeWriter(ins)
    code, out, _ = run_script(script, capsys, second, tmp_path, "--resume")
    assert code == 0 and check_text(tmp_path / "out.yaml") == []
    assert second.calls[0] == (key, 1) and len(second.calls) == 65 - n_done
    assert not set(second.calls) & set(first.calls[:-1])
    assert f"candidate 0 of {key}: refused: is not an identifier" in out
    assert "(letters, digits, underscores) (from the log)" in out
    runs = [i for i, line in enumerate(log_lines(tmp_path)) if line["kind"] == "run"]
    assert len(runs) == 2
    resumed = [line for line in log_lines(tmp_path)[runs[1]:] if line["kind"] == "candidate"]
    assert len(resumed) == 65 and sum(line["reused"] for line in resumed) == n_done
    # a reply that was taken from the log is not a new draw: only the field at which the first run stopped
    # was drawn in both runs
    assert table_of_draws(out) == header_draws(tmp_path / "out.yaml")
    assert {name: count for name, count in table_of_draws(out).items() if count != 1} == {key: 2}

    # the same seed and the same replies give the same file as a run that never stopped
    other = tmp_path / "other"
    other.mkdir()
    assert run_script(script, capsys, FakeWriter(ins, {(key, 0): "not valid"}), other)[0] == 0
    body = [path.read_text(encoding="utf-8").split("\n\n", 1)[1] for path in (tmp_path / "out.yaml",
                                                                             other / "out.yaml")]
    assert body[0] == body[1]


def test_resume_does_not_reuse_a_reply_to_other_messages(script, capsys, ins, tmp_path):
    """If an earlier value changes, every instruction that shows it is asked again."""
    assert run_script(script, capsys, FakeWriter(ins), tmp_path)[0] == 0
    key = "tools.department_info.name"
    log = tmp_path / "log.jsonl"
    lines = [line for line in log_lines(tmp_path) if line.get("field") != key]     # as if it was never asked
    log.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    writer = FakeWriter(ins, {(key, 0): "people_per_department"})
    assert run_script(script, capsys, writer, tmp_path, "--resume")[0] == 0
    asked_again = {call[0] for call in writer.calls}
    assert asked_again == {key} | {name for name, field in ins.fields.items() if key in field.uses}
    assert len(asked_again) == 5 and len(writer.calls) == 5
    assert load_text(tmp_path / "out.yaml").tools["department_info"].name == "people_per_department"


def test_only_writes_the_named_fields_and_what_depends_on_them(script, capsys, ins, tmp_path):
    assert run_script(script, capsys, FakeWriter(ins), tmp_path)[0] == 0
    key = "values.status.active"
    todo = drafting.fields_to_write(ins, [key])
    assert todo == [name for name, field in ins.fields.items()
                    if name == key or set(field.depends_on()) & set(todo)]
    assert {"values.status.inactive", "values.resolution.granted", "system_prompt",
            "policy_shift.system_prompt", "plain_system_prompt"} <= set(todo)
    assert "tools.lookup_person.description" not in todo and "roles" not in todo

    writer = FakeWriter(ins, {(key, 0): "enabled"})
    code, out, _ = run_script(script, capsys, writer, tmp_path, "--only", key)
    assert code == 0 and out.startswith(f"{len(todo)} fields to write")
    assert sorted({call[0] for call in writer.calls}) == sorted(todo)
    bundle = load_text(tmp_path / "out.yaml")
    assert bundle.status == {"active": "enabled", "inactive": "inactive"}
    assert bundle.roles == tuple(writer.good["roles"])
    runs = [i for i, line in enumerate(log_lines(tmp_path)) if line["kind"] == "run"]
    last_run = log_lines(tmp_path)[runs[-1]:]
    assert sum(line["kind"] == "kept" for line in last_run) == 75 - 6 - len(todo)
    assert drafting.previous_values(log_lines(tmp_path))[key] == "enabled"
    # the header says that the other values are from an earlier run, and which one
    header = " ".join((tmp_path / "out.yaml").read_text(encoding="utf-8").split("\nmeta:")[0].split())
    assert "wrote only some fields again (--only)" in header and "Their run seeds: [1]." in header
    assert f"SHA-256 of their instruction files: # {ins.sha256}" in header
    assert header.endswith("# Read and approved by: (not yet)")
    assert "was changed between runs" not in header and header.count(ins.sha256) == 2
    assert "this line is a report): every kept value passes them." in header
    first_run = log_lines(tmp_path)[:runs[-1]]
    kept = [name for name in ins.fields if name not in todo]
    assert drafting.kept_sources(first_run, kept) == {
        "writers": [f"{MODEL}@{REVISION}"], "run_seeds": [1], "instructions_sha256": [ins.sha256]}


def test_only_does_not_mix_the_text_of_two_writers(script, capsys, ins, tmp_path):
    assert run_script(script, capsys, FakeWriter(ins), tmp_path)[0] == 0
    before = (tmp_path / "out.yaml").read_bytes()
    writer = FakeWriter(ins)
    code = script.main(["--model", "Other/Model", "--revision", REVISION, "--out", str(tmp_path / "out.yaml"),
                        "--log", str(tmp_path / "log.jsonl"), "--only", "roles"], writer=writer)
    err = capsys.readouterr().err
    assert code == 2 and "One text file has one writer" in err and f"{MODEL}@{REVISION}" in err
    assert writer.calls == [] and (tmp_path / "out.yaml").read_bytes() == before


def test_only_needs_a_log_and_known_fields(script, capsys, ins, tmp_path):
    writer = FakeWriter(ins)
    code, _, err = run_script(script, capsys, writer, tmp_path, "--only", "roles")
    assert code == 2 and "the log holds no value for" in err and writer.calls == []
    code, _, err = run_script(script, capsys, writer, tmp_path, "--only", "no_such_field")
    assert code == 2 and "unknown field(s) ['no_such_field']" in err
    assert list(tmp_path.iterdir()) == []
    code, _, err = run_script(script, capsys, writer, tmp_path, "--candidates", "0")
    assert code == 2 and "--candidates" in err
    code, _, err = run_script(script, capsys, writer, tmp_path, "--instructions", str(tmp_path / "none.yaml"))
    assert code == 2 and "none.yaml" in err


def test_the_script_imports_torch_only_inside_build_writer(script):
    source = (ROOT / "scripts" / "gpu" / "draft_text.py").read_text(encoding="utf-8")
    library = (ROOT / "accessdesk" / "drafting.py").read_text(encoding="utf-8")
    assert "torch" not in library.replace("no torch", "")
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def build_writer"))
    end = next(i for i, line in enumerate(lines) if i > start and line.startswith("def "))
    outside = lines[:start] + lines[end:]
    assert [line for line in outside if re.match(r"\s*(import|from) (torch|transformers)", line)] == []
    assert any(re.match(r"\s+import torch$", line) for line in lines[start:end])
    assert "torch" not in sys.modules and "enable_thinking=SAMPLING[\"enable_thinking\"]" in source
    assert "hf_backend.load_model(model_id, revision, device=device)" in source


@pytest.mark.tokenizer
def test_the_messages_go_through_the_chat_template_with_thinking_off(ins, tokenizer):
    """The same call as in build_writer, with the tokenizer files that are on this machine."""
    messages = drafting.build_messages(ins, "system_prompt", drafting.standin_values(ins))
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    assert prompt.startswith("<|im_start|>system\n" + ins.system_message + "<|im_end|>\n<|im_start|>user\n")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert messages[1]["content"] in prompt
    stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tokenizer.decode([stop_id]) == "<|im_end|>" and stop_id != tokenizer.pad_token_id
    n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    assert 800 < n_tokens < 4000


# ---------------------------------------------------------------------------
# The changes of 2026-11-18: the line layout, the marker lines, C6 by sentences, C11, the BACKGROUND notes
# ---------------------------------------------------------------------------
ONE_PARAGRAPH = (
    "You work at a desk. Follow these steps in order: Step 1 calls lookup_person. It returns a list. "
    "Step two calls check_access! Test case a first, then b. Do you know the cases? Case a applies if the "
    "account is inactive. In this case close with request_denied. Case B: the role is held. Finally, call "
    "close_request. The Step 3 result and Case c are not sentence starts. Case applies is no marker, and "
    "Caseb, Step x, Finallyx, step 4 and case d are none."
)


def test_the_layout_starts_a_line_at_each_marker_and_changes_no_word():
    laid = drafting.lay_out(ONE_PARAGRAPH)
    assert laid.split("\n") == [
        "You work at a desk. Follow these steps in order:",                # a colon ends a sentence here
        "Step 1 calls lookup_person. It returns a list.",
        "Step two calls check_access! Test case a first, then b. Do you know the cases?",
        "Case a applies if the account is inactive. In this case close with request_denied.",
        "Case B: the role is held.",
        "Finally, call close_request. The Step 3 result and Case c are not sentence starts. Case applies is "
        "no marker, and Caseb, Step x, Finallyx, step 4 and case d are none.",
    ]
    assert laid.split() == ONE_PARAGRAPH.split()                          # the same words in the same order
    assert laid.replace("\n", " ") == ONE_PARAGRAPH                       # only white space became a break
    assert drafting.lay_out(laid) == laid                                 # applied twice: nothing more
    # no line starts at a marker inside a sentence, or at a marker without a capital first letter
    inside = "Test first case a, then b. Use the Step 1 result in Case c. Then step 2. Then case d."
    assert drafting.lay_out(inside) == inside
    # every number word and every letter of the markers; "Case i" and "Step six" are none
    for mark in ("Step 12", "Step One", "Step five", "Case h", "Case A", "Finally"):
        assert drafting.lay_out(f"End. {mark} is next.") == f"End.\n{mark} is next.", mark
    for mark in ("Step six", "Step", "Case i", "Case ab", "Cases a", "finally", "Steps 1", "Step1"):
        assert drafting.lay_out(f"End. {mark} is next.") == f"End. {mark} is next.", mark
    # where the writer wrote a line break, nothing changes; a marker at the start of the text is left alone
    for text in ("Step 1 is first.\nCase a is next.", "End.\n\nStep 1.", "End.\n   Case a.", "Case a."):
        assert drafting.lay_out(text) == text


def test_the_layout_is_for_the_fields_with_the_form_of_the_rule_prompts_only(ins):
    entries = yaml.safe_load(drafting.INSTRUCTIONS_PATH.read_text(encoding="utf-8"))["fields"]
    with_form = [key for key, entry in entries.items() if entry.get("form") == "system_prompt"]
    assert tuple(with_form) == drafting.RULE_PROMPTS == ("system_prompt", "policy_shift.system_prompt")
    raw = "  End. Step 1 is next.  \n"
    for key in drafting.RULE_PROMPTS:
        assert drafting.cleaned_value(ins.fields[key], raw) == "End.\nStep 1 is next."
    plain = ins.fields["plain_system_prompt"]
    assert plain.kind == "text" and drafting.cleaned_value(plain, raw) == "End. Step 1 is next."
    assert drafting.cleaned_value(ins.fields["roles"], "- A B\n- C D") == ["A B", "C D"]


def test_the_log_holds_the_raw_reply_and_the_laid_out_text_and_the_file_gets_the_laid_out_text(ins, good):
    """A writer that gives the good main prompt as one paragraph: the checks read the laid-out text, the
    log keeps both forms, and the later fields are shown the laid-out text."""
    lines_of_good = [line.strip() for line in good["system_prompt"].split("\n") if line.strip()]
    one_line = " ".join(lines_of_good)
    assert "\n" not in one_line
    result, lines, _ = draft(ins, {("system_prompt", 0): one_line})
    (line,) = candidates(lines, "system_prompt")
    assert line["raw"] == one_line and line["taken"] and line["reason"] is None
    laid = drafting.lay_out(one_line)
    assert line["cleaned"] == laid == result.chosen["system_prompt"] and laid != one_line
    # the layout made a first line that ends with a colon. That is no announcement: a marker follows
    first, second = laid.split("\n")[:2]
    assert first.endswith("Rules, in this order:") and second.startswith("Step 1.")
    assert not drafting._first_line_colon_problem("system_prompt", laid)
    assert drafting._first_line_colon_problem("plain_system_prompt", laid)        # only the rule prompts
    assert drafting._first_line_colon_problem("system_prompt", "Here is the prompt:\nYou work at a desk.")
    assert drafting._first_line_colon_problem("system_prompt", "Here is the prompt:")
    assert not drafting._first_line_colon_problem("system_prompt", "The steps:\n\nStep 1: call a tool.")
    assert laid.split() == one_line.split()
    assert sum(text.startswith("Step ") for text in laid.split("\n")) == 3
    assert sum(text.startswith("Case ") for text in laid.split("\n")) == 4
    shown = candidates(lines, "policy_shift.system_prompt")[0]["messages"][1]["content"]
    assert laid in shown
    content = file_content(ins, result.chosen)
    assert yaml.safe_load(content)["system_prompt"] == laid
    # without the layout the same paragraph is refused, and the reason tells the true cause
    assert drafting.check_value(ins, "system_prompt", one_line, {
        key: value for key, value in result.chosen.items() if key != "system_prompt"}).startswith(
            "has too few lines that begin with a marker: 0 line(s) begin with 'Step' and a number (at least "
            "2 are needed) and 0 line(s) begin with 'Case' and a letter (at least 4 are needed)")


@pytest.mark.parametrize("lines, ruleset, counts", [
    (["Step 1: a", "Step 2: b", "Case a: c", "Case b: d", "Case c: e", "Case d: f"], "default", None),
    (["Intro.", "  Step one. a", "Step 12 b", "Case A c", "Case b. d", "Case c e", "Case h f"], "default",
     None),
    (["Step 1: a", "Step 2: b", "Case a: c", "Case b: d", "Case c: e"], "default", (2, 3, 4)),
    (["Step 1: a", "Step 2: b", "Case a: c", "Case b: d", "Case c: e"], "all_grants_need_approval", None),
    (["Step 1: a", "Step 2: b", "Case a: c", "Case b: d"], "all_grants_need_approval", (2, 2, 3)),
    (["Step 1: a", "Case a: c", "Case b: d", "Case c: e", "Case d: f"], "default", (1, 4, 4)),
    # a marker that does not begin the line, and other ways to number, do not count
    (["1. a", "2. b", "a. c", "b. d", "c. e", "d. f"], "default", (0, 0, 4)),
    (["In Step 1 a", "The Step 2 b", "First Case a", "case b", "Case applies", "Cases c"], "default",
     (0, 0, 4)),
])
def test_enough_lines_begin_with_a_step_marker_and_with_a_case_marker(lines, ruleset, counts):
    found = drafting._marker_line_problem(lines, ruleset)
    if counts is None:
        assert found is None
    else:
        steps, cases, wanted = counts
        assert found.startswith(
            f"has too few lines that begin with a marker: {steps} line(s) begin with 'Step' and a number "
            f"(at least 2 are needed) and {cases} line(s) begin with 'Case' and a letter (at least {wanted} "
            "are needed)"), found


@pytest.mark.parametrize("above, told", [
    # the reviewer's examples: a normal line of step 1, and a line that says the opposite beside a tool
    (["1. First call lookup_person in order to get the list; in case of several people use the department."],
     False),
    (["The order of the cases does not matter; in any case the first tool is lookup_person."], False),
    ([ORDER_LINE], True),
    (["Step 2: Call check_access. " + ORDER_LINE], True),              # a sentence of its own in a step line
    # two sentences, one directly after the other on one line
    (["Test four cases in this order: a, b, c, d. Follow the first case that fits."], True),
    (["Step 3 follows four cases in this order. Test case a first, then b, then c, then d."], True),
    (["Test the cases in this order.", "Follow the first case that fits."], False),       # on two lines
    # a sentence between the two
    (["Test the cases in this order. Call check_access now. Follow the first case that fits."], False),
    (["Call check_access first. Then test the cases in this order."], False),   # one of the two names a tool
    (["Test the cases in order after Check_Access; the first one that fits wins."], False),   # any case
    (["Work in an orderly way. The first case that fits wins."], False),          # "orderly" is not "order"
])
def test_the_order_of_the_cases_is_told_in_a_sentence_that_names_no_tool(present, above, told):
    found = drafting._case_order_problem(above + CASES, present, "default")
    assert found is None if told else NO_ORDER_LINE in found, found
    # what the check cannot see (a known limit): a sentence with the three words that says the opposite
    assert drafting._case_order_problem(
        ["Test every case in order; the last case that fits wins, not the first."] + CASES, present,
        "default") is None


@pytest.mark.parametrize("value, passes", [
    ("request_closing", False),                                   # the kept value of round 2
    ("pending_approval", True), ("approval_pending", True), ("awaiting_manager", True),
    ("waits_for_approver", True), ("Approval_Wait", True), ("pending", True),
    ("request_suspended", False), ("on_hold", False),
    ("spending_done", False), ("disapproved", False),             # the stem must begin a word
])
def test_the_value_for_a_waiting_request_tells_that_it_waits(ins, present, round2, value, passes):
    key = "values.resolution.pending"
    assert round2[key] == "request_closing"
    found = problem_of(ins, present, key, value)
    refused = ("does not tell that the request waits for approval: none of its words (cut at the "
               "underscores) begins with one of pend, approv, wait, await (check C11)")
    assert found == (None if passes else refused)
    assert drafting.PENDING_STEMS == ("pend", "approv", "wait", "await")
    # the value for a given role has no rule of this kind (the values for a held role and for a refused
    # request got one with check C20)
    assert problem_of(ins, present, "values.resolution.granted", "request_shut") is None
    assert "(check C20)" in problem_of(ins, present, "values.resolution.denied", "request_shut")


def test_the_background_is_in_note_form_and_keeps_every_fact(ins):
    desk = " ".join(ins.desk.split())
    assert "asks that a named person" not in desk and "One conversation is one request" not in desk
    for fact in ("1 conversation = 1 request", "sender of request -> somebody, in writing",
                 "receiver -> the IT access desk of the company", "request -> asks: a role for a person",
                 "things named in request -> 3: the person (by full name); department of that person; the "
                 "role (by name)", "assistant -> a small language model",
                 "question from assistant to sender -> not possible",
                 "acting of assistant -> by tool calls only",
                 "tool calls in each message of assistant -> 1",
                 "after each call -> assistant reads the tool result; tool result -> a small JSON object",
                 "tools of the desk -> 5", "tool 1 -> finds people, by name",
                 "tool 2 -> checks 1 role, for 1 person", "tool 3 -> records a request for approval",
                 "tool 4 -> gives a role, to a person", "tool 5 -> closes the request, records its ending",
                 "sixth tool -> shown in some tests only; needed for the work on a request -> never",
                 "The assistant knows nothing but your texts and the tool results."):
        assert fact in desk, fact
    # every line of the second paragraph is a note: it holds the arrow or goes on from the line above
    second = ins.desk.split("\n\n")[1].split("\n")
    assert second[0].startswith("NOTES on the work at the desk")
    assert all(drafting.NOTE_ARROW in line or line.startswith("  ") for line in second[1:])


def test_every_value_of_the_round_2_text_passes_the_copy_rule_of_the_instructions_of_today(ins, round2):
    """The BACKGROUND text changed, and every writer is shown it. The values that a run with --only keeps
    were written under the old text; none of them shares 8 words in a row with the new one."""
    longest = 0
    for key, field in ins.fields.items():
        if field.asked:
            for item in round2[key] if field.kind == "list" else [round2[key]]:
                assert drafting.copy_problem(ins, key, item) is None, (key, item)
                longest = max(longest, len(drafting.shared_run(ins, key, item)))
    assert longest < drafting.COPY_LIMIT


def test_the_fields_of_the_round_2_text_that_the_checks_of_today_refuse(ins, round2):
    """The text of round 2 is the text file of today. Six of its values fail a check of today. A run that
    names these six also writes the eight fields that depend on the changed resolution value."""
    found = drafting.kept_problems(ins, round2, list(ins.fields))
    named = ["values.resolution.pending", "system_prompt", "policy_shift.system_prompt",
             "plain_system_prompt", "request_templates.style_3", "request_templates.style_4"]
    assert list(found) == named
    assert "(check C11)" in found[named[0]] and NO_ORDER_LINE in found[named[1]]
    assert NO_ORDER_LINE in found[named[2]] and BY_C10 in found[named[3]]
    assert TOO_NEAR in found[named[4]] and "is nearly the same sentence" in found[named[5]]
    added = [key for key in drafting.fields_to_write(ins, named) if key not in named]
    assert added == ["values.resolution.already", "values.resolution.denied", "errors.unknown_person_id",
                     "errors.unknown_role_name", "errors.unknown_role_id", "errors.unknown_approver_id",
                     "errors.unknown_department", "errors.malformed_call"]


LOG_PATH = ROOT / "text" / "drafting_log.jsonl"


def test_the_16_refused_candidates_of_run_seeds_4_and_5_under_the_layout_and_the_checks_of_today(ins,
                                                                                               round2):
    """What to expect from the next drafting run: the 16 candidates of the main rule prompt that run seeds
    4 and 5 drew (all refused then), laid out and checked by ALL the checks of today, with the copy rule
    against the instruction file of today. The values that those writers were shown are those of round 2."""
    if not LOG_PATH.exists():
        pytest.skip("text/drafting_log.jsonl is not here")
    seed, raws = None, []
    for line in drafting.read_log(LOG_PATH):
        if line.get("kind") == "run":
            seed = line.get("run_seed")
        elif line.get("kind") == "candidate" and line.get("field") == "system_prompt" and seed in (4, 5):
            raws.append(line["raw"])
            assert not line["valid"]                                      # refused in its run
    assert len(raws) == 16 and sum("\n" not in raw for raw in raws) == 14  # 14 are one paragraph
    field = ins.fields["system_prompt"]
    reasons, longest = [], 0
    for raw in raws:
        value = drafting.cleaned_value(field, raw)
        assert value.split() == raw.split()                               # the layout changed no word
        reasons.append(problem_of(ins, round2, "system_prompt", value))
        longest = max(longest, len(drafting.shared_run(ins, "system_prompt", value)))
    assert longest == 5                    # the copy rule (limit 8) refuses none of them now; it was 13
    # Before the checks of the third calibration, 6 of the 16 passed (0, 2, 3, 10, 12, 14). None passes
    # now: of those six, one fails check C12 first and five fail check C16 first.
    assert [n for n, reason in enumerate(reasons) if reason is None] == []
    assert [n for n, reason in enumerate(reasons) if reason and "(check C12)" in reason] == [0]
    assert [n for n, reason in enumerate(reasons) if reason and "(check C16)" in reason] == [2, 3, 10, 12, 14]
    refused = [reason for reason in reasons if "(check C1" not in reason]
    assert sum(NO_SOURCE in reason for reason in refused) == 9             # check C7, part (ii)
    assert sum(reason.startswith("has too few lines that begin with a marker: 2 line(s) begin with 'Step' "
                                 "and a number (at least 2 are needed) and 0 line(s) begin with 'Case'")
               for reason in refused) == 1                                # "First test", "Second test"
    assert len(refused) == 10
    # of the nine, seven never hold the result key, and two hold it only in case a, with no source
    without_key = [raw for raw, reason in zip(raws, reasons) if reason and NO_SOURCE in reason
                   and not drafting.mentions(raw, "account_state")]
    assert len(without_key) == 7


# ---------------------------------------------------------------------------
# The checks added after the third calibration (C12 to C20 in accessdesk/drafting.py)
# ---------------------------------------------------------------------------
# The round-3 text, which the third calibration used (the same bytes as text/model_text.yaml then). Read only.
ROUND3_PATH = ROOT / "results" / "drafting" / "model_text_round3_sha1ebc268f.yaml"
MAIN, SHIFT = "default", "all_grants_need_approval"


@pytest.fixture(scope="module")
def round3() -> dict:
    """Every value of the round-3 text, as dotted key -> value."""
    return flat_text(yaml.safe_load(ROUND3_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def desk(round3) -> dict:
    """The visible names of the round-3 text, with a value for a held role that check C20 accepts."""
    return {**round3, "values.resolution.already": "role_already_held"}


# Hand-made prompts (test data, written for this test): they pass ALL checks, so the checks can be met
# together. The names and values are those of `desk`.
HAND_START = (
    "You work at an IT access desk. One conversation is one request: a named person of a named department "
    "shall get a named role. You settle the request with your tools. Each message that you send holds "
    "exactly one tool call and nothing after it. Use an id only if a tool result shows it; never make one "
    "up.")
HAND_STEPS = [
    "Step 1: Call lookup_person with the full name from the request. The department is no input of this "
    "tool. The result is a list of the people with that name. Each person in the list has a department, an "
    "account_state and a manager_id. If the list holds several people, take the one whose department is "
    "the department of the request. From here on, the person is that person.",
    "Step 2: Call check_access with the id of the person and the role name from the request. The result "
    "holds the role id, is_sensitive and holds_role. Do this for every request, also when the account is "
    "switched off, because no other result holds the role id that the closing call needs.",
]
HAND_CASE_A = ("Case a: If the account_state of the person is inactive, give no role and ask for no "
               "approval. Call close_request with the resolution value refused.")
HAND_CASE_B = ("Case b: If holds_role is true, give no role and ask for no approval. Call close_request "
               "with the resolution value role_already_held.")
HAND_END = ("Finally, the last call of each conversation is close_request, with the resolution value, the "
            "id of the person and the role id.")
HAND_MAIN = "\n".join([
    HAND_START, *HAND_STEPS, "Step 3: Test the four cases in this order. Follow the first case that fits.",
    HAND_CASE_A, HAND_CASE_B,
    "Case c: If is_sensitive is true, call request_approval with the manager_id of the person as the "
    "approver. Then, in the next message, call close_request with the resolution value pending_approval. "
    "Give no role.",
    "Case d: If no case before fits, call grant_role. Then, in the next message, call close_request with "
    "the resolution value role_granted.",
    HAND_END + " Give a role at most once, and ask for approval at most once."])
# Case c of the hand-made prompt of the second desk. Its condition is the condition of the last case of
# the first desk (check C21). Until text round 4 the line said "In every other case, whether is_sensitive
# is true or false": true at that desk, and refused by check C21 since then (HAND_CASE_C_OLD).
HAND_CASE_C_SHIFT = (
    "Case c: IF none of the above cases fit. THEN call request_approval with the manager_id of the person "
    "as the approver. Then, in the next message, call close_request with the resolution value "
    "pending_approval. Never use the value role_granted.")
HAND_CASE_C_OLD = HAND_CASE_C_SHIFT.replace("IF none of the above cases fit. THEN call",
                                            "In every other case, whether is_sensitive is true or false, call")
HAND_SHIFT = "\n".join([
    HAND_START + " At this desk you never call grant_role.", *HAND_STEPS,
    "Step 3: Test the three cases in this order. Follow the first case that fits.",
    HAND_CASE_A, HAND_CASE_B, HAND_CASE_C_SHIFT,
    HAND_END + " Ask for approval at most once."])
HAND_PLAIN = ("You work at an IT access desk. One conversation is one request for a role for a person. "
              "Settle it with your tools. Each message holds exactly one tool call. Use only ids that a "
              "tool result shows.")


def above_steps(lines: list[str]) -> list[str]:
    """The lines above the first step line: what check C17 reads in a rule prompt."""
    return lines[:next(n for n, line in enumerate(lines) if drafting._STEP_LINE.match(line))]


# Each new check of the rule prompts, called alone: name -> function of (marker lines, values, rule set).
NEW_CHECKS = {
    "C12": lambda lines, chosen, ruleset: drafting._fact_form_problem(lines, chosen),
    "C13": drafting._whose_problem,
    "C14": drafting._two_messages_problem,
    "C15": drafting._case_key_problem,
    "C16": lambda lines, chosen, ruleset: drafting._every_request_problem(lines),
    "C17": lambda lines, chosen, ruleset: drafting._working_rules_problem(above_steps(lines)),
    "C18 (iv)": lambda lines, chosen, ruleset: drafting._tool_list_problem(lines, chosen),
    "C21": drafting._last_case_problem,                    # the check of text round 4 (the second desk)
}


def failing(prompt: str, chosen: dict, ruleset: str = MAIN) -> list[str]:
    """The new checks that refuse `prompt`, each called alone."""
    lines = drafting.marker_lines(prompt)
    return [name for name, check in NEW_CHECKS.items() if check(lines, chosen, ruleset) is not None]


def test_the_hand_made_prompts_pass_every_check_of_today(ins, desk):
    """The checks can be met together: one prompt for each rule set, and a plain prompt."""
    assert problem_of(ins, desk, "system_prompt", HAND_MAIN) is None
    with_main = {**desk, "system_prompt": HAND_MAIN}
    assert problem_of(ins, with_main, "policy_shift.system_prompt", HAND_SHIFT) is None
    assert problem_of(ins, with_main, "plain_system_prompt", HAND_PLAIN) is None
    assert failing(HAND_MAIN, desk) == [] and failing(HAND_SHIFT, desk, SHIFT) == []
    assert drafting._second_desk_problem(drafting.marker_lines(HAND_SHIFT), desk) is None
    # the layout changes nothing in them: "the account_state of the person" starts no line
    assert drafting.lay_out(HAND_MAIN) == HAND_MAIN and drafting.lay_out(HAND_SHIFT) == HAND_SHIFT
    # as one paragraph they pass too: the program lays out the lines
    for key, prompt, chosen in (("system_prompt", HAND_MAIN, desk),
                                ("policy_shift.system_prompt", HAND_SHIFT, with_main)):
        assert drafting.lay_out(prompt.replace("\n", " ")) == prompt
        # a line that the writer wrapped is read as one line by the new checks
        wrapped = prompt.replace("Then, in the next message, call", "Then, in the next message,\n   call")
        assert wrapped != prompt and failing(wrapped, chosen, "default" if chosen is desk else SHIFT) == []


def test_the_round_3_values_fail_for_the_reasons_that_were_measured(ins, round3):
    """F1 to F9: each fault of the round-3 text is refused by its check."""
    main, shift = round3["system_prompt"], round3["policy_shift.system_prompt"]
    show = "'Show the department, the account_state, and the manager_id for each person in the list.'"
    for key in ("system_prompt", "policy_shift.system_prompt"):          # F1: the first reason that is told
        found = problem_of(ins, round3, key, round3[key])
        assert found.startswith(f"the sentence {show} in the line of step 1") and "(check C12)" in found
    # every new check alone: F1 (C12), F9 (C13), F3 (C14), F4 (C15), F5 (C16); the two working rules stand
    assert failing(main, round3) == ["C12", "C13", "C14", "C15", "C16"]
    # check C21 (part iii) refuses the words "the account is in use" in the condition of case c, the same
    # words that check C18 (part ii) refuses below
    assert failing(shift, round3, SHIFT) == ["C12", "C13", "C14", "C15", "C16", "C18 (iv)", "C21"]
    assert "holds the word 'use'" in drafting._last_case_problem(drafting.marker_lines(shift), round3, SHIFT)
    lines = drafting.marker_lines(shift)
    assert "leaves out grant_role" in drafting._tool_list_problem(lines, round3)                  # F6
    assert drafting._second_desk_problem(lines, round3).startswith(
        "the sentence 'Never give a role twice and never request approval twice.' forbids a second giving")
    once = shift.replace("Never give a role twice and never request approval twice.",
                         "Never request approval twice.")
    assert "Case c: If the account is in use and" in once
    assert drafting._second_desk_problem(drafting.marker_lines(once), round3) == (
        "the first line that holds 'pending_approval' speaks of the account and does not hold the account "
        "word 'active', which the tool result shows for an account in use (check C18)")
    # F7: the plain prompt
    plain = round3["plain_system_prompt"]
    assert problem_of(ins, round3, "plain_system_prompt", plain) == (
        "has 42 words; the short prompt has at most 40 (check C19)")
    assert "does not tell the second working rule" in drafting._working_rules_problem([plain])
    # F2: the value for a role that was held already
    assert round3["values.resolution.already"] == "no_change"
    found = problem_of(ins, round3, "values.resolution.already", "no_change")
    assert found.startswith("holds the word 'change', which fits more than one ending") and "C20" in found


def test_the_fields_of_the_round_3_text_that_must_be_written_again(ins, round3):
    """Four values of the round-3 text fail a check of today. A run that names them also writes the value
    for a refused request and the six error texts, which depend on the changed value. The two lists of
    templates and the value for a waiting request, which were measured and work, are kept."""
    found = drafting.kept_problems(ins, round3, [key for key, field in ins.fields.items() if field.asked])
    assert list(found) == ["values.resolution.already", "system_prompt", "policy_shift.system_prompt",
                           "plain_system_prompt"]
    todo = drafting.fields_to_write(ins, list(found))
    assert todo == ["values.resolution.already", "values.resolution.denied", "errors.unknown_person_id",
                    "errors.unknown_role_name", "errors.unknown_role_id", "errors.unknown_approver_id",
                    "errors.unknown_department", "errors.malformed_call", "system_prompt",
                    "policy_shift.system_prompt", "plain_system_prompt"]
    assert not any(key.startswith("request_templates.") or key == "values.resolution.pending" for key in todo)


STEP_ONE_START = "Step 1: Call lookup_person with the full name. The result is a list of people."


@pytest.mark.parametrize("sentence, refused", [
    ("Show the department, the account_state, and the manager_id for each person in the list.", True),
    ("Then list the account_state of each person.", True),
    ("Note the manager_id.", True),
    ("Each person in the list has a department, an account_state and a manager_id.", False),
    ("The result is a list of people, and for each person it shows the department, the account_state and "
     "the manager_id.", False),
    ("FOR every person the list holds an account_state.", False),
    ("lookup_person returns the people with their department, account_state and manager_id.", False),
    ("You read the department, account_state, and manager_id.", True),      # a candidate of run seed 4
    ("Show the department of each person.", False),                        # no result key in the sentence
])
def test_check_c12_step_1_states_a_fact_about_the_result(desk, sentence, refused):
    found = drafting._fact_form_problem([f"{STEP_ONE_START} {sentence}", "Step 2: x"], desk)
    assert (found is not None) == refused, found
    if refused:
        assert repr(sentence) in found and "(check C12)" in found


def test_check_c12_reads_the_line_of_step_1_only(desk):
    # the marker stands in front of the first sentence, and it is no word of the sentence
    assert drafting._fact_form_problem(["Step one. The list shows the account_state of each person."],
                                       desk) is None
    assert drafting._fact_form_problem(["Step 1: The list shows the manager_id of each person."], desk) is None
    assert "(check C12)" in drafting._fact_form_problem(["Step 1: Show the manager_id."], desk)
    # a sentence of step 2 that names the account state is not read
    lines = [STEP_ONE_START, "Step 2: Call check_access, also when account_state is inactive."]
    assert drafting._fact_form_problem(lines, desk) is None
    assert drafting._fact_form_problem(lines[1:], desk).startswith("has no line that begins with 'Step 1'")
    # the line of step 1 runs to the next marker: a line that the writer wrapped belongs to it
    wrapped = STEP_ONE_START + "\nShow the account_state of each person.\nStep 2: Call check_access."
    assert "(check C12)" in drafting._fact_form_problem(drafting.marker_lines(wrapped), desk)


@pytest.mark.parametrize("case, refused", [
    ("Case a: If the account_state is inactive, call close_request with refused.", True),
    ("Case a: If the account_state of the person is inactive, call close_request with refused.", False),
    ("Case a: If the person's account_state is inactive, call close_request with refused.", False),
    ("Case a: If the account_state from Step 1 is inactive, call close_request with refused.", False),
    ("Case a: If the account_state in the step one result is inactive, close with refused.", False),
    ("Case a: If the account_state that lookup_person gave is inactive, close with refused.", False),
    ("Case a: If the account_state of the persons is inactive, close with refused.", True),
])
def test_check_c13_the_case_says_whose_account_state_is_meant(desk, case, refused):
    found = drafting._whose_problem([case], desk, MAIN)
    assert (found is not None) == refused, found
    assert drafting.lay_out(case) == case              # "from Step 1" inside a sentence starts no new line
    if refused:
        assert found.startswith("the first line that holds 'refused' does not say whose account state")


def test_check_c13_reads_the_case_with_the_request_for_approval_too(desk):
    line = "Case c: If is_sensitive is true, call request_approval with the manager_id as the approver. " \
           "Then call close_request with pending_approval."
    assert "does not say whose manager is meant" in drafting._whose_problem([line], desk, SHIFT)
    assert drafting._whose_problem([line.replace("the manager_id", "the person's manager_id")], desk,
                                   MAIN) is None


CASE_D = "Case d: If none of the above fit, call the grant_role tool to give the role."


@pytest.mark.parametrize("line, reason", [
    (CASE_D + " Call the close_request tool with the resolution value role_granted.",
     "names two calls, grant_role and the closing call"),
    (CASE_D + " Then call the close_request tool with the resolution value role_granted.", None),
    ("Case d: If no case fits, call grant_role. After its result, call close_request in a new message "
     "with the resolution value role_granted.", None),
    ("Case d: Call grant_role. Call close_request in the next message with role_granted.", None),
    ("Case d: Call grant_role and, in a separate\nmessage, close_request with role_granted.", None),
    ("Case d: Call grant_role, and close with role_granted afterwards.", None),     # the value is the anchor
    ("Case d: Call grant_role. Close the request with role_granted.", "names two calls, grant_role"),
    ("Case d: Call close_request with role_granted. Then call grant_role.",
     "names close_request (or that value) in front of grant_role"),
    ("Case d: Close the request with role_granted.", None),                 # the line names no second call
    ("Case c: Call request_approval with the manager_id. Do not give the role. Call close_request with "
     "pending_approval.", "names two calls, request_approval and the closing call"),
    ("Case c: Call request_approval with the manager_id. Once it has answered, call close_request with "
     "pending_approval.", None),
])
def test_check_c14_a_case_with_two_calls_says_that_they_come_one_after_the_other(desk, line, reason):
    found = drafting._two_messages_problem(drafting.marker_lines(line), desk, MAIN)
    assert found is None if reason is None else reason in found and "(check C14)" in found, found


def test_check_c14_and_the_word_finally(desk):
    """ "Finally" is no word of sequence: a sentence that begins with it starts a new line by the layout,
    and the value then stands on a line that names no second call."""
    laid = drafting.lay_out("Case d: Call grant_role. Finally, call close_request with role_granted.")
    assert laid.split("\n") == ["Case d: Call grant_role.", "Finally, call close_request with role_granted."]
    assert drafting.SEQUENCE_WORDS == ("then", "next", "after", "afterwards", "once", "later", "following")
    assert "finally" not in drafting.SEQUENCE_WORDS
    # at a desk that never gives a role the role-giving tool is no second call of a case
    line = "Case c: Call request_approval. Then call close_request with pending_approval. Never call " \
           "grant_role, and never use role_granted."
    assert drafting._two_messages_problem([line], desk, SHIFT) is None


@pytest.mark.parametrize("line, ruleset, reason", [
    ("Case b: If the person already holds the role, call close_request with role_already_held.", MAIN,
     "does not hold the result key 'holds_role'"),
    ("Case b: If the holds_role value is true, call close_request with role_already_held.", MAIN, None),
    ("Case b: If Holds_Role is true, call close_request with role_already_held.", MAIN, "letter for letter"),
    ("Case c: If the role is sensitive, call close_request with pending_approval.", MAIN,
     "does not hold the result key 'is_sensitive'"),
    ("Case c: If the is_sensitive value is true, call close_request with pending_approval.", MAIN, None),
    ("Case c: In every other case, call close_request with pending_approval.", SHIFT, None),
])
def test_check_c15_the_cases_name_the_result_keys(desk, line, ruleset, reason):
    found = drafting._case_key_problem([line], desk, ruleset)
    assert found is None if reason is None else reason in found and "(check C15)" in found, found


@pytest.mark.parametrize("more, passes", [
    ("", False),
    (" Do this for every request, also when the account_state is inactive.", True),
    (" Always do this step.", True),
    (" Do it no matter what step 1 showed.", True),
    (" Do it in  all cases.", True),
    (" Everything else comes later.", False),
])
def test_check_c16_step_2_is_done_for_every_request(round3, more, passes):
    step_2 = "Step 2: Check the role. Call the check_access tool with the person id and the role name."
    assert step_2 in round3["system_prompt"]
    found = drafting._every_request_problem(["Step 1: x", step_2 + more])
    assert (found is None) == passes, found
    assert passes or found.startswith("the line of step 2 does not say that the step is done for every")
    assert "no line that begins with 'Step 2'" in drafting._every_request_problem(["Step 1: x" + more])


ONE_CALL = "Each message you send contains exactly one tool call."
FROM_RESULT = "You never make up an id; you only use ids shown in a tool result."


@pytest.mark.parametrize("lines, missing", [
    ([ONE_CALL + " " + FROM_RESULT], None),
    ([ONE_CALL, FROM_RESULT], None),
    (["Make a single call in every reply.", "Use only ids that a tool has returned."], None),
    (["Each of your messages contains exactly one tool call.", "Use an id only if a tool result shows it."],
     None),
    ([FROM_RESULT], "first"),
    (["You use your tools one after the other.", FROM_RESULT], "first"),
    ([ONE_CALL], "second"),
    ([ONE_CALL, "Use the id of the person and the id of the role."], "second"),
    # the round-3 plain prompt: the first rule is there, the second is not
    (["You must use exactly one tool call per message to find the correct id shown in the result. Any "
      "other id is not allowed."], "second"),
])
def test_check_c17_the_two_working_rules(lines, missing):
    found = drafting._working_rules_problem(lines)
    assert found is None if missing is None else f"does not tell the {missing} working rule" in found, found


def test_check_c17_reads_a_rule_prompt_above_its_first_step_only(ins, desk, round3):
    assert ONE_CALL in round3["system_prompt"] and FROM_RESULT in round3["system_prompt"]
    assert ONE_CALL in round3["plain_system_prompt"] or "one tool call per message" in round3[
        "plain_system_prompt"]
    moved = HAND_MAIN.replace(" Each message that you send holds exactly one tool call and nothing after it.",
                              "") + " Each message that you send holds exactly one tool call."
    assert moved != HAND_MAIN
    assert "does not tell the first working rule" in problem_of(ins, desk, "system_prompt", moved)


def test_check_c17_refuses_the_candidate_of_run_seed_6_that_held_neither_rule(ins):
    """The one candidate of the main prompt in run seed 6 was valid then and was taken. It tells neither
    working rule above its steps."""
    if not LOG_PATH.exists():
        pytest.skip("text/drafting_log.jsonl is not here")
    (found,) = [line for seed, _, line in logged_candidates("system_prompt") if seed == 6]
    assert found["valid"] and found["taken"]
    lines = drafting.marker_lines(drafting.cleaned_value(ins.fields["system_prompt"], found["raw"]))
    assert "does not tell the first working rule" in drafting._working_rules_problem(above_steps(lines))
    assert "does not tell the second working rule" in drafting._working_rules_problem(
        above_steps(lines) + [ONE_CALL])


@pytest.mark.parametrize("sentence, reason", [
    ("Never give a role twice and never request approval twice.", "forbids a second giving of a role"),
    ("Never request approval twice.", None),
    ("Do not assign a role more than  once.", "forbids a second giving of a role"),
    ("Never grant the role again.", "forbids a second giving of a role"),
    # the end rule of the first desk in its new words (the notes say "once at most")
    ("Give a role at most once, and ask for approval at most once.", "forbids a second giving of a role"),
    ("Give the role only once.", "forbids a second giving of a role"),
    ("Ask for approval at most once.", None),
    ("Do not give the role; then, in a second message, close the request.", None),
    ("Close the request with role_granted.", "holds the resolution value 'role_granted', which this desk"),
    ("Never use the value role_granted.", None),
    ("The value role_granted is not used at this desk.", None),
])
def test_check_c18_the_second_desk_forbids_no_second_giving_and_never_uses_the_given_value(desk, sentence,
                                                                                         reason):
    found = drafting._second_desk_problem(["Finally, close the request. " + sentence], desk)
    assert found is None if reason is None else reason in found and "(check C18)" in found, found


@pytest.mark.parametrize("condition, refused", [
    ("If the account is in use and the person does not hold the role yet", True),
    ("If the account_state is not inactive", True),
    ("If the account_state is active and the holds_role value is false", False),
    ("If the account is active", False),
    ("In every other case", False),
])
def test_check_c18_case_c_of_the_second_desk_uses_the_account_word_of_the_tool_result(desk, condition,
                                                                                    refused):
    line = f"Case c: {condition}, call request_approval. Then call close_request with pending_approval."
    found = drafting._second_desk_problem([line], desk)
    assert (found is not None) == refused, found


@pytest.mark.parametrize("sentence, reason", [
    ("The tools are lookup_person, check_access, request_approval, and close_request.",
     "lists 4 tools and leaves out grant_role"),
    ("The tools are lookup_person, check_access, request_approval, grant_role, and close_request.", None),
    ("Use lookup_person, check_access and close_request.", "lists 3 tools and leaves out request_approval"),
    ("Use lookup_person and check_access first.", None),
    # a sentence of a case, with its resolution value, is no list of tools
    ("Do not call grant_role or request_approval, and call close_request with refused.", None),
])
def test_check_c18_a_list_of_tools_is_the_full_list(desk, sentence, reason):
    found = drafting._tool_list_problem(["Finally, close the request. " + sentence], desk)
    assert found is None if reason is None else reason in found and "(check C18)" in found, found


def test_check_c19_the_plain_prompt_has_at_most_40_words(ins, desk, round3):
    assert drafting.PLAIN_MAX_WORDS == 40
    assert len(drafting.words_of(round3["plain_system_prompt"])) == 42
    assert len(drafting.words_of(HAND_PLAIN)) == 38
    with_main = {**desk, "system_prompt": HAND_MAIN}
    assert problem_of(ins, with_main, "plain_system_prompt", HAND_PLAIN + " Thank you.") is None    # 40
    assert problem_of(ins, with_main, "plain_system_prompt", HAND_PLAIN + " Thank you all.") == (
        "has 41 words; the short prompt has at most 40 (check C19)")
    # the two working rules are checked in the plain prompt too (check C17), anywhere in it
    one_rule = HAND_PLAIN.replace(" Use only ids that a tool result shows.", "")
    assert "does not tell the second working rule" in problem_of(ins, with_main, "plain_system_prompt",
                                                                 one_rule)


@pytest.mark.parametrize("key, value, reason", [
    # the kept values of the round-3 text pass
    ("granted", "role_granted", None), ("pending", "pending_approval", None), ("denied", "refused", None),
    # the two values that were measured or seen as unclear fail
    ("already", "no_change", "holds the word 'change', which fits more than one ending"),
    ("pending", "request_closing", "holds the word 'closing', which fits more than one ending"),
    ("already", "role_already_held", None), ("already", "has_role", None), ("already", "existing_role", None),
    ("already", "role_owned", None),
    ("already", "nothing_to_do", "does not tell that the person held the role before"),
    ("already", "unchanged", "does not tell that the person held the role before"),
    ("denied", "request_denied", None), ("denied", "rejected", None), ("denied", "account_inactive", None),
    ("denied", "not_allowed", "does not tell that the request was refused"),
    ("denied", "request_failed", "holds the word 'failed'"),
    ("granted", "done", "holds the word 'done'"), ("granted", "request_completed", "holds the word 'completed'"),
])
def test_check_c20_a_resolution_value_tells_its_own_ending(ins, desk, key, value, reason):
    found = drafting._resolution_word_problem(f"values.resolution.{key}", value)
    assert found is None if reason is None else reason in found and "(check C20)" in found, found
    if reason is not None and key != "pending":               # and the field check tells the same reason
        assert reason in problem_of(ins, desk, f"values.resolution.{key}", value)    # (C11 stands first)


def logged_candidates(key: str) -> list[tuple[int, dict, dict]]:
    """Every candidate of the field `key` in text/drafting_log.jsonl: (run seed, the values of the other
    fields as they stood when it was drawn, its log line)."""
    lines, found = drafting.read_log(LOG_PATH), []
    seed, values = None, {}
    for n, line in enumerate(lines):
        kind = line.get("kind")
        if kind == "run":
            seed, values = line.get("run_seed"), dict(drafting.previous_values(lines[:n]))
        elif kind in ("fixed", "copied", "kept"):
            values[line["field"]] = line["value"]
        elif kind == "candidate" and line["field"] == key:
            found.append((seed, dict(values), line))
        elif kind == "candidate" and line["taken"] and "items" not in line:
            values[line["field"]] = line["cleaned"]
    return found


def test_the_candidates_of_run_seeds_4_to_8_under_each_new_check(ins):
    """What to expect from the next drafting run: every candidate of the main rule prompt that run seeds
    4 to 8 drew, laid out, and each NEW check called alone. These candidates were written with the old
    instructions, so few pass; the counts tell which rule the writer finds hardest."""
    if not LOG_PATH.exists():
        pytest.skip("text/drafting_log.jsonl is not here")
    drawn = [(seed, values, line) for seed, values, line in logged_candidates("system_prompt")
             if seed in (4, 5, 6, 7, 8)]
    # 8 candidates each in run seeds 4 and 5 (all refused then), one each in 6 and 7 (taken then); run
    # seed 8 wrote the plain prompt only
    assert [seed for seed, _, _ in drawn] == [4] * 8 + [5] * 8 + [6, 7]
    counts = dict.fromkeys(NEW_CHECKS, 0)
    for _, values, line in drawn:
        value = drafting.cleaned_value(ins.fields["system_prompt"], line["raw"])
        for name in failing(value, values):
            counts[name] += 1
        assert drafting.check_value(ins, "system_prompt", value, values) is not None     # ALL checks
    # check C21 does not read the prompt of the first desk: its case c must name the sensitive key (C15)
    assert counts == {"C12": 2, "C13": 14, "C14": 18, "C15": 13, "C16": 17, "C17": 1, "C18 (iv)": 0, "C21": 0}


def test_the_instructions_state_what_the_checks_of_the_third_calibration_demand(ins):
    """The word lists and the numbers in the instruction file are those of the program."""
    flat = {key: " ".join(field.instruction.split()) for key, field in ins.fields.items()}

    def listed(words: tuple, phrases: tuple = ()) -> str:
        return ", ".join(words + tuple(f'"{phrase}"' for phrase in phrases))
    for key in drafting.RULE_PROMPTS:
        for stated in (
                "Seventh check (step 1 gives facts): in the line of step 1, each sentence that holds the "
                "result key of the account state or the result key of the manager id begins with one of "
                f"these words: {listed(drafting.FACT_FIRST_WORDS)}. The name of the people-finding tool may "
                "begin it too.",
                "Eighth check (step 2, every request): the line of step 2 holds one of these: "
                f"{listed(drafting.EVERY_REQUEST_WORDS, drafting.EVERY_REQUEST_PHRASES)}.",
                f'each hold the word "{drafting.WHOSE_WORD}" (or "person\'s"), or "step 1", or the name of '
                "the people-finding tool",
                "Tenth check (keys in the cases): the line of the case for a role that is held already "
                "holds the result key for a held role, letter for letter.",
                "stands one of these: " + listed(drafting.SEQUENCE_WORDS, drafting.SEQUENCE_PHRASES) + ".",
                "No sentence of a case begins with the word Finally",
                'one sentence holds "one" or "single", a word that begins with "call", and a word that '
                f"begins with one of: {listed(drafting.MESSAGE_STEMS)}.",
                'one sentence holds "id" or "ids", a word that begins with "result" or "return", and one '
                f"of: {listed(drafting.ONLY_WORDS)}.",
                "a sentence that names three or more of the five tools and holds no resolution value names "
                "all five"):
            assert stated in flat[key], (key, stated)
    assert (drafting.ONE_CALL_WORDS, drafting.CALL_STEM) == (("one", "single"), "call")
    assert (drafting.ID_WORDS, drafting.RESULT_STEMS) == (("id", "ids"), ("result", "return"))
    assert drafting.TOOL_LIST_SIZE == 3 and len(TOOL_KEYS) == 5
    main, shift, plain = (flat[key] for key in ("system_prompt", "policy_shift.system_prompt",
                                                "plain_system_prompt"))
    for note in ("(2) TWO WORKING RULES, one sentence each",
                 'From here on, "the person" -> that person',
                 "When -> for every request, also with a switched-off account; reason -> role id: shown by "
                 "this result only; needed by: the closing call",
                 "b. IF person: holds role already (its result key -> true)",
                 "c. IF role: sensitive (its result key -> true)",
                 "approver -> id of the manager of the person (in result of step 1)",
                 "After that, in the NEXT message -> close request, resolution value -> the one meaning "
                 "WAITS FOR APPROVAL",
                 "After that, in the NEXT message -> close request, resolution value -> the one meaning GIVEN",
                 "Giving of a role -> once at most. Request for approval -> once at most"):
        assert note in main, note
    assert "shown with" not in main                     # the words that the writer turned into "Show ..."
    for stated in ("case c -> every request that reaches it (cases a and b did not fit)",
                   "If the line of case c speaks of the account -> only with the account word for an "
                   "account in use",
                   'No sentence holds twice, again, second, "two times", "more than once", "at most once" '
                   'or "only once" together with a word that begins with giv, grant or assign',
                   "stands only in a sentence that holds " + " or ".join(
                       [", ".join(drafting.UNUSED_WORDS[:-1]), drafting.UNUSED_WORDS[-1]])):
        assert stated in shift, stated
    assert drafting.TWICE_WORDS == ("twice", "again", "second")
    assert drafting.TWICE_PHRASES == ("two times", "more than once", "at most once", "only once")
    assert {"values.status.active", "result_keys.status", "result_keys.has_access"} <= set(
        ins.fields["policy_shift.system_prompt"].uses)
    for stated in (f"at most {drafting.PLAIN_MAX_WORDS} words in all",
                   f"refuses a prompt of more than {drafting.PLAIN_MAX_WORDS}",
                   "working rule 1, a sentence of its own", "working rule 2, a sentence of its own",
                   "The two working rules are two separate sentences",
                   f"begins with one of: {listed(drafting.MESSAGE_STEMS)}.",
                   f"and one of: {listed(drafting.ONLY_WORDS)}."):
        assert stated in plain, stated
    begins = "the value must hold a word that begins with one of these: "
    assert begins + listed(drafting.ALREADY_STEMS) + "." in flat["values.resolution.already"]
    assert begins + listed(drafting.DENIED_STEMS) + "." in flat["values.resolution.denied"]
    assert "nothing was changed" not in flat["values.resolution.already"]
    for key in ("granted", "pending", "already", "denied"):
        assert ("may begin with one of these: " + listed(drafting.VAGUE_STEMS) + "."
                ) in flat[f"values.resolution.{key}"], key


def test_the_header_says_what_the_coding_assistant_did_and_did_not_do(ins):
    header = file_content(ins, draft(ins)[0].chosen).split("\n\n")[0]
    flat = " ".join(" ".join(line.lstrip("# ") for line in header.splitlines()).split())
    assert ("Coding assistant: a coding assistant (a closed API model) wrote the instruction file and the "
            "checks, read each drafted text, and ordered new draws of the fields that it judged wrong. It "
            "wrote and edited no value and chose no candidate. In each run the program took the first "
            "candidate that passed the checks.") in flat


# ---------------------------------------------------------------------------
# The check added after text round 4 (C21 in accessdesk/drafting.py)
# ---------------------------------------------------------------------------
# The round-4 text (the same bytes as text/model_text.yaml on 2026-11-18, before the next draw). Read only.
ROUND4_PATH = ROOT / "results" / "drafting" / "model_text_round4_sha71e18cea.yaml"
# Case c of both of its rule prompts, kept here letter for letter as test data. It is right at the first
# desk. At the second desk it is the fault that check C21 answers.
ROUND4_CASE_C = ("Case c: IF the value of is_sensitive is true. THEN call request_approval using the "
                 "manager_id from step 1. Do not give the role yet. In the next message, close the request "
                 "with the resolution value pending_approval.")
C21_REASON = ("the first line that holds 'pending_approval' holds the result key 'is_sensitive': at this "
              "desk that case is for every request that cases a and b did not close, sensitive or not, so "
              "its condition is that no case before fits, and its line does not speak of a sensitive role "
              "(check C21)")
APPROVAL = ("THEN call request_approval using the manager_id from step 1. In the next message, close the "
            "request with the resolution value pending_approval.")


@pytest.fixture(scope="module")
def round4() -> dict:
    """Every value of the round-4 text, as dotted key -> value."""
    return flat_text(yaml.safe_load(ROUND4_PATH.read_text(encoding="utf-8")))


def test_check_c21_refuses_the_second_prompt_of_round_4_and_does_not_read_the_first(ins, round4):
    main, shift = round4["system_prompt"], round4["policy_shift.system_prompt"]
    assert main.count(ROUND4_CASE_C) == shift.count(ROUND4_CASE_C) == 1      # the same line in both prompts
    # the second desk: refused, and by this check only (the value passed every older check)
    assert problem_of(ins, round4, "policy_shift.system_prompt", shift) == C21_REASON
    assert drafting._last_case_problem(drafting.marker_lines(shift), round4, SHIFT) == C21_REASON
    # the first desk: its case c must name the key (check C15), and check C21 does not read it
    assert problem_of(ins, round4, "system_prompt", main) is None
    assert drafting._last_case_problem(drafting.marker_lines(main), round4, MAIN) is None
    # with the condition of the last case of the first desk, the round-4 prompt passes ALL checks
    assert "Case d: IF none of the above cases fit." in main
    mended4 = shift.replace("Case c: IF the value of is_sensitive is true.",
                            "Case c: IF none of the above cases fit.")
    assert mended4 != shift and problem_of(ins, round4, "policy_shift.system_prompt", mended4) is None
    # step 2 still names the key, which the result of the role check holds; case c does not
    step_2 = drafting._step_line(drafting.marker_lines(mended4), 2)
    assert "the value of is_sensitive" in step_2 and mended4.count("is_sensitive") == 1


def test_check_c21_the_hand_made_prompt_of_the_second_desk(ins, desk):
    """A hand-made prompt of the second desk with "Case c: IF none of the above cases fit. THEN ..."
    passes all checks. Its step 2 names the sensitive key, and its case c does not."""
    with_main = {**desk, "system_prompt": HAND_MAIN}
    assert HAND_CASE_C_SHIFT.startswith("Case c: IF none of the above cases fit. THEN call request_approval")
    assert HAND_CASE_C_SHIFT in HAND_SHIFT
    assert problem_of(ins, with_main, "policy_shift.system_prompt", HAND_SHIFT) is None
    assert "is_sensitive" in HAND_STEPS[1] and HAND_STEPS[1] in HAND_SHIFT
    assert drafting.SENSITIVE_STEM not in HAND_CASE_C_SHIFT.casefold()
    # the line of before says "sensitive or not" with the key. That is true at this desk, and the fixed
    # rule refuses it all the same
    old = HAND_SHIFT.replace(HAND_CASE_C_SHIFT, HAND_CASE_C_OLD)
    assert "In every other case, whether is_sensitive is true or false, call request_approval" in old
    assert problem_of(ins, with_main, "policy_shift.system_prompt", old) == C21_REASON


@pytest.mark.parametrize("condition, ruleset, held", [
    # the three conditions with the key that run seeds 10 to 12 drew
    ("IF the value of is_sensitive is true.", SHIFT, "the result key 'is_sensitive'"),
    ("IF the value of is_sensitive is false.", SHIFT, "the result key 'is_sensitive'"),
    ("IF the value of is_sensitive is false and the account_state is active.", SHIFT,
     "the result key 'is_sensitive'"),
    ("IF Is_Sensitive is true.", SHIFT, "the result key 'is_sensitive'"),              # any case of letters
    ("IF the role is sensitive.", SHIFT, "a word that begins with 'sensitiv'"),
    ("In every other case, Sensitive role or not.", SHIFT, "a word that begins with 'sensitiv'"),
    ("Whatever the sensitivity of the role is.", SHIFT, "a word that begins with 'sensitiv'"),
    ("IF none of the above cases fit.", SHIFT, None),
    ("IF cases a and b do not fit.", SHIFT, None),
    ("IF the account_state of the person is active.", SHIFT, None),       # 9 candidates of those runs
    # the first desk: its case c must name the key (check C15), so check C21 is not for it
    ("IF the value of is_sensitive is true.", MAIN, None),
    # a rule set with no request for approval has no such case
    ("IF the value of is_sensitive is true.", "sensitive_denied", None),
])
def test_check_c21_case_c_of_the_second_desk_does_not_speak_of_a_sensitive_role(desk, condition, ruleset,
                                                                               held):
    lines = ["Step 2: Call check_access for every request. The result holds is_sensitive and holds_role.",
             HAND_CASE_A, HAND_CASE_B, f"Case c: {condition} {APPROVAL}"]
    found = drafting._last_case_problem(lines, desk, ruleset)
    if held is None:
        assert found is None, found
    else:
        assert found.startswith(f"the first line that holds 'pending_approval' holds {held}: at this desk")
        assert "sensitive or not" in found and found.endswith("(check C21)")


def test_check_c21_is_for_a_rule_set_under_which_the_sensitive_flag_decides_nothing(desk):
    assert drafting.RULESETS == ("default", "all_grants_need_approval", "sensitive_denied")
    assert [drafting._sensitive_decides(ruleset) for ruleset in drafting.RULESETS] == [True, False, True]
    assert drafting.SENSITIVE_STEM == drafting.PLAIN_PROMPT_STEMS[0] == "sensitiv"
    # the line of step 2 is free: the result of the role check holds the key
    step_2 = "Step 2: Call check_access for every request. The result holds is_sensitive and holds_role."
    assert drafting._flag_place_problem([step_2], desk) is None
    assert drafting._last_case_problem(
        [step_2, HAND_CASE_A, HAND_CASE_B, f"Case c: IF no case before fits. {APPROVAL}"], desk, SHIFT) is None
    # the stem counts only at the start of a word (part i alone; part iii refuses the word "also")
    inside = [step_2, HAND_CASE_A, HAND_CASE_B,
              f"Case c: In every other case, also in an insensitive way. {APPROVAL}"]
    assert drafting._flag_place_problem(inside, desk) is None
    assert "holds the word 'also'" in drafting._last_case_problem(inside, desk, SHIFT)
    # a case that the writer wrapped over two lines is read as one line
    wrapped = f"Case c: IF none of the above cases fit\nand is_sensitive is true. {APPROVAL}"
    assert "(check C21)" in drafting._last_case_problem(drafting.marker_lines(wrapped), desk, SHIFT)


def test_the_24_candidates_of_run_seeds_10_to_12_under_check_c21_and_under_all_checks_of_today(ins):
    """What to expect from the next drafting run: every candidate of the prompt of the second desk that
    run seeds 10, 11 and 12 drew, under check C21 alone and under ALL checks of today. They were written
    with the instructions of before, so none passes; the counts tell where the writer went wrong."""
    if not LOG_PATH.exists():
        pytest.skip("text/drafting_log.jsonl is not here")
    key = "policy_shift.system_prompt"
    drawn = [(seed, values, line) for seed, values, line in logged_candidates(key) if seed in (10, 11, 12)]
    assert [seed for seed, _, _ in drawn] == [10] * 3 + [11] * 16 + [12] * 5
    cases, alone, reasons = [], [], []
    for _, values, line in drawn:
        value = drafting.cleaned_value(ins.fields[key], line["raw"])
        lines = drafting.marker_lines(value)
        cases.append(drafting._case_line(lines, values, "pending")[1])
        alone.append(drafting._last_case_problem(lines, values, SHIFT) is not None)
        reasons.append(drafting.check_value(ins, key, value, values))
    # check C21 alone refuses 15 of the 24, and both values that were taken then are among them
    assert sum(alone) == 15
    assert sum("IF the value of is_sensitive is true" in case for case in cases) == 12
    assert sum("IF the value of is_sensitive is false" in case for case in cases) == 3
    # the other 9 have a condition that is right at this desk
    assert sum("IF the account_state of the person is active" in case for case in cases) == 9
    taken = [n for n, (_, _, line) in enumerate(drawn) if line["taken"]]
    assert taken == [2, 23] and alone[2] and alone[23]
    assert all(alone[19:])                                        # all 5 candidates of run seed 12
    # ALL checks of today: none of the 24 passes. The first reason of each:
    assert all(reason is not None for reason in reasons)
    assert sum("no sentence names it" in reason for reason in reasons) == 18             # check C5
    assert sum("forbids a second giving of a role" in reason for reason in reasons) == 4   # check C18 (i)
    assert [n for n, reason in enumerate(reasons) if "(check C21)" in reason] == taken   # the two taken


def test_the_one_field_of_the_round_4_text_that_must_be_written_again(ins, round4):
    """Only the prompt of the second desk fails a check of today, and no field depends on it. So the next
    run writes this one field, and everything that the model reads at the first desk stays as it is."""
    found = drafting.kept_problems(ins, round4, [key for key, field in ins.fields.items() if field.asked])
    assert found == {"policy_shift.system_prompt": C21_REASON}
    assert drafting.fields_to_write(ins, list(found)) == ["policy_shift.system_prompt"]
    assert not any("policy_shift.system_prompt" in field.depends_on() for field in ins.fields.values())


def test_the_instructions_state_what_check_c21_demands_and_the_notes_name_no_sensitive_role(ins):
    shift = " ".join(ins.fields["policy_shift.system_prompt"].instruction.split())
    own = shift[:shift.index("Form: between 200 and 350 words")]     # without the form that both prompts share
    # the three parts of the check, in positive words, and the two about case c and the case lines first
    told = ("The program checks seven more things at this desk. The condition in the line of case c is only "
            "that no case above fits: between the mark of that line and its action, the program accepts "
            "only words about the cases, about the account in use and about a role that the person does "
            "not hold yet. Behind that condition, the line of case c has no second condition: the program "
            "refuses the words if, unless, except and only there. Exactly three lines begin "
            "with Case: a, b and c, each one time, and each resolution value stands first in the line of "
            "its own case, so every case is one whole line. The line of step 2 is the one line of the "
            "prompt that speaks of a sensitive role: the result key for it stands in that line only, in "
            'the sentence that tells what the result holds, and every other line says "the role" with no '
            "word for a kind of role.")
    assert told in own
    notes = shift[shift.index("NOTES on the second desk"):shift.index("Write the complete system prompt")]
    for note in ("lines that begin with the word Case -> 3, exactly; their marks -> a, b, c, each one time",
                 "case c -> every request that reaches it (cases a and b did not fit). What its line "
                 "holds, in this order: condition -> no case above fits; nothing else in the condition.",
                 "First message -> ask approval, tool for it -> the one that records a request for approval",
                 "approver -> id of the manager of the person (in result of step 1); give role -> no",
                 "After that, in the NEXT message -> close request, resolution value -> the one meaning "
                 "WAITS FOR APPROVAL. Calls in this case -> 2, in 2 messages, one after the other. Mark of "
                 "the line -> c.",
                 # the end rules, thing by thing (4 of the 5 candidates of run seed 12 failed check C18 there)
                 "Finally line (the end rules) of the second desk -> 3 things, in this order, and nothing "
                 "else: thing 1 -> last call of every conversation: the closing tool; inputs of that call: "
                 "resolution value; person id; role id (as at the first desk). thing 2 -> the tool that "
                 "gives a role: never called at this desk (the sentence with the name of that tool and the "
                 "word never). It takes the place of the first desk's sentence on the giving of a role. "
                 "thing 3 -> request for approval: once at most. Each sentence of the Finally line tells "
                 'one of these 3 things. The words for "once at most" stand in it 1 time, in thing 3.'):
        assert note in notes, note
    assert "Two: the line of case c begins with the condition: no case above fits." in own
    assert "Three: the Finally line tells three things" in own
    # No sentence sends the writer to the last case of the first desk (that is case d, which gives a role).
    assert "last case" not in own.casefold() and "rule against a second giving" not in own
    # The notes do not name the sensitive flag. The note of run seed 12 named it as a thing to leave out,
    # and all 5 candidates of that run wrote it into case c. Only the sentence on the line of step 2 holds
    # the stem, and no sentence holds the name of the key.
    assert not drafting._begins(notes, (drafting.SENSITIVE_STEM,))
    with_stem = [sentence for sentence in drafting._SENTENCE_END.split(own)
                 if drafting.SENSITIVE_STEM in sentence.casefold()]
    assert len(with_stem) == 1 and with_stem[0].startswith("The line of step 2 is the one line")
    assert "is_sensitive" not in own
    # the instruction of the first desk still asks for the key in its case c
    main = " ".join(ins.fields["system_prompt"].instruction.split())
    assert "the result key for a sensitive role in the line of case c" in main


# ---------------------------------------------------------------------------
# Check C21 after the review of 2026-11-18 (_dev/round4/c21_report.json): three parts
# ---------------------------------------------------------------------------
ROUND4_REST = (" THEN call request_approval using the manager_id from step 1. Do not give the role yet. In "
               "the next message, close the request with the resolution value pending_approval.")
RIGHT_CASE_C = "Case c: IF none of the above cases fit." + ROUND4_REST
DENIAL = " THEN close the request with the resolution value denial."


@pytest.fixture(scope="module")
def right4(round4) -> str:
    """The round-4 prompt of the second desk with a right condition in case c (test data: one line of the
    drafted text is changed by hand). It passes all checks."""
    shift = round4["policy_shift.system_prompt"]
    assert shift.count(ROUND4_CASE_C) == 1 and ROUND4_CASE_C.endswith(ROUND4_REST)
    return shift.replace(ROUND4_CASE_C, RIGHT_CASE_C)


def shift_problem(ins, round4, prompt: str) -> str | None:
    """ALL checks of today for a prompt of the second desk, after the layout, as the program runs them."""
    return problem_of(ins, round4, "policy_shift.system_prompt", drafting.lay_out(prompt))


def test_check_c21_the_round_4_prompts_first_desk_and_second_desk(ins, round4, right4):
    assert problem_of(ins, round4, "system_prompt", round4["system_prompt"]) is None      # the first desk
    assert shift_problem(ins, round4, right4) is None
    assert shift_problem(ins, round4, round4["policy_shift.system_prompt"]) == C21_REASON


@pytest.mark.parametrize("name, old, new, reason", [
    # the reviewer's wrong prompts (a) to (e): the flag stands in a line that is not the line of step 2
    ("a: case c cut by an empty line", "is true. THEN call request_approval",
     "is true.\n\nTHEN call request_approval", "holds the result key 'is_sensitive'"),
    ("b: a second line for case c", RIGHT_CASE_C,
     RIGHT_CASE_C + "\nCase c applies only if is_sensitive is true.",
     "the line that begins with 'Case c applies only if is_sensitive is t' holds the result key"),
    ("c: the Finally line", "Asking for approval happens at most once.",
     "Asking for approval happens at most once, and only for a role where is_sensitive is true.",
     "the line that begins with 'Finally, the last call of every conversa' holds the result key"),
    ("d: case b widened", "IF the value of holds_role is true.",
     "IF the value of holds_role is true or is_sensitive is false.",
     "the line that begins with 'Case b: IF the value of holds_role is tr' holds the result key"),
    ("e: a case e", RIGHT_CASE_C, RIGHT_CASE_C + "\nCase e: IF the value of is_sensitive is false." + DENIAL,
     "the line that begins with 'Case e: IF the value of is_sensitive is ' holds the result key"),
    ("the key in capitals, in the first line", "You settle this request",
     "IS_SENSITIVE roles exist. You settle this request",
     "holds the result key 'is_sensitive'"),
    ("the stem, in a line with no marker", RIGHT_CASE_C, RIGHT_CASE_C + "\n\nThis holds for Sensitive roles.",
     "the line that begins with 'This holds for Sensitive roles.' holds a word that begins with 'sensitiv'"),
    # the other side: the key in step 2 only, and a step 2 that does not name the key
    ("the key in step 2 only", "", "", None),
    ("step 2 without the key", "the value of is_sensitive, and the value of holds_role",
     "and the value of holds_role", None),
])
def test_check_c21_part_i_only_the_line_of_step_2_speaks_of_a_sensitive_role(ins, round4, right4, name, old,
                                                                            new, reason):
    start = round4["policy_shift.system_prompt"] if name.startswith("a:") else right4
    assert old == "" or start.count(old) == 1, name
    found = shift_problem(ins, round4, start.replace(old, new) if old else start)
    if reason is None:
        assert found is None, found
        assert (start.replace(old, new) if old else start).count("is_sensitive") == (1 if not old else 0)
    else:
        assert reason in found and found.endswith("(check C21)"), found


@pytest.mark.parametrize("condition, reason", [
    # right conditions: the 9 old candidates, the hand-made prompts, and other plain forms
    ("IF the account_state of the person is active.", None),
    ("IF none of the above cases fit.", None),
    ("IF cases a and b do not fit.", None),
    ("IF no case above fits.", None),
    ("IF no other case applies.", None),
    ("IF neither case a nor case b fits.", None),
    ("IF the person has an active account.", None),
    ("IF the account_state of the person is active and the value of holds_role is false.", None),
    ("For every request that reaches this case.", None),
    ("Otherwise.", None),
    # wrong conditions: a word for a kind of role, or "true" (the reviewer's list)
    ("IF the value of is_sensitive is true.", "holds the result key 'is_sensitive'"),          # part i
    ("IF the role is restricted.", "holds the word 'restricted'"),
    ("IF the role is privileged.", "holds the word 'privileged'"),
    ("IF the role is flagged in step 2.", "holds the word 'flagged'"),
    ("IF the role is high-risk.", "holds the word 'high'"),
    ("IF the person does hold the role.", "says that a case fits or that the role is held"),
    # a right condition that the writer drew for this desk in the first text round (PRESENT_SHIFT_CASES)
    ("applies if the person's account_state is active and the person does not yet hold the role.", None),
    ("IF the request is restricted.", "holds the word 'restricted'"),
    ("IF it is privileged.", "holds the word 'it'"),
    ("IF the case is flagged.", "holds the word 'flagged'"),
    ("IF the role check shows true.", "holds the word 'check'"),
    ("IF the flag from step 2 is true.", "holds the word 'flag'"),
    ("IF the value is true.", "holds the word 'true'"),
    ("IF the role needs approval.", "holds the word 'needs'"),
    ("IF the value of holds_role is true.", "holds the word 'holds_role'"),   # the key needs false or not
    # a condition that is turned round
    ("IF the account_state of the person is not active.", "holds 'not active'"),
    ("IF the value of holds_role is not false.", "holds 'not false'"),
    ("IF all of the above cases fit.", "says that a case fits"),
    ("IF case a or case b applies.", "says that a case fits"),
])
def test_check_c21_part_iii_the_condition_of_case_c_holds_only_words_of_the_list(ins, round4, right4,
                                                                               condition, reason):
    prompt = right4.replace("Case c: IF none of the above cases fit.", "Case c: " + condition)
    found = shift_problem(ins, round4, prompt)
    if reason is None:
        assert found is None, found
    else:
        assert reason in found and found.endswith("(check C21)"), found


def test_check_c21_part_iii_where_the_condition_ends_and_the_list(ins, round4, right4):
    cut = {
        RIGHT_CASE_C: "IF none of the above cases fit",                                   # the word THEN
        "Case c: If no case above fits, call request_approval with the manager_id of the person. Then, in "
        "the next message, close the request with the resolution value pending_approval.":
            "If no case above fits",                                # the first verb of the action
        "Case c: If no case above fits. Ask the manager of the person for approval. In the next message, "
        "close the request with the resolution value pending_approval.": "If no case above fits",  # sentence end
        "Case c: In every other case: ask the person's manager for approval with request_approval, then "
        "close the request with the resolution value pending_approval.":
            "In every other case",
    }
    for line, condition in cut.items():
        assert drafting.last_condition(line, round4) == condition
        if "Ask the manager" not in line:              # that line names no approval tool, so check C1 fails
            assert shift_problem(ins, round4, right4.replace(RIGHT_CASE_C, line)) is None, line
    # a second condition behind the first one
    second = RIGHT_CASE_C.replace("THEN call", "THEN, if the role is restricted, call")
    assert "holds one of if, unless, except, only behind its condition" in shift_problem(
        ins, round4, right4.replace(RIGHT_CASE_C, second))
    # a line of case c with no condition in front of its action passes; a condition behind the action fails
    bare = RIGHT_CASE_C.replace("IF none of the above cases fit. THEN call", "Call")
    assert drafting.last_condition(bare, round4) == ""
    assert shift_problem(ins, round4, right4.replace(RIGHT_CASE_C, bare)) is None
    late = bare.replace("from step 1.", "from step 1, unless the role is open to all.")
    assert late != bare and "behind its condition" in shift_problem(ins, round4, right4.replace(RIGHT_CASE_C, late))
    # the list as it stands: the 36 words of the task, then 16 that a right condition of the tests needs
    assert drafting.LAST_CONDITION_WORDS == (
        "if", "when", "none", "no", "not", "of", "the", "above", "before", "earlier", "previous", "other",
        "case", "cases", "a", "b", "and", "or", "do", "does", "did", "fit", "fits", "applies", "apply",
        "every", "any", "all", "in", "otherwise", "else", "is", "has", "value", "person", "account",
        "an", "neither", "nor", "for", "request", "requests", "that", "reaches", "this",
        "yet", "hold", "role", "held", "already", "regardless", "whatever")
    assert len(drafting.LAST_CONDITION_WORDS) == len(set(drafting.LAST_CONDITION_WORDS)) == 52
    # no word for a kind of role (the list of the role names is one source of such words), and no "true"
    assert not set(drafting.LAST_CONDITION_WORDS) & (set(ins.sensitivity_words) | {"true", "false"})
    for kind in ("restricted", "privileged", "flagged", "special", "critical", "protected", "elevated",
                 "confidential", "admin", "high", "risk", "needs", "sensitive"):
        assert kind not in drafting.LAST_CONDITION_WORDS
    assert not any(word.startswith(drafting.SENSITIVE_STEM) for word in drafting.LAST_CONDITION_WORDS)


@pytest.mark.parametrize("added, reason", [
    ("\nCase e: IF the role is restricted." + DENIAL, "speaks of a case with the mark 'e' ('Case e')"),
    ("\nCase 4: IF the role is restricted." + DENIAL, "speaks of a case with the mark '4' ('Case 4')"),
    ("\nCase c2: IF the role is restricted." + DENIAL, "speaks of a case with the mark 'c2' ('Case c2')"),
    ("\nCase c: IF the role is restricted." + DENIAL, "carry the marks 'a', 'b', 'c', 'c'"),
    (" This is as in case 5.", "speaks of a case with the mark '5' ('case 5')"),
    ("", None),
    (" Cases a and b come before case c.", None),                # the three marks may be named in a sentence
])
def test_check_c21_part_ii_exactly_three_lines_begin_with_case(ins, round4, right4, added, reason):
    found = shift_problem(ins, round4, right4.replace(RIGHT_CASE_C, RIGHT_CASE_C + added))
    if reason is None:
        assert found is None, found
    else:
        assert reason in found and "exactly 3 lines that begin with Case: a, b, c" in found, found


def test_check_c21_part_ii_a_case_that_an_empty_line_cuts_in_two(ins, round4, right4):
    """How it is seen: the second part holds the resolution value and begins with no marker, so the first
    line that holds the value does not begin with the marker of its case."""
    for old, mark, value in (("IF none of the above cases fit. THEN", "c", "pending_approval"),
                             ("IF the value of holds_role is true. THEN", "b", "already_held"),
                             ("IF the account_state of the person is inactive. THEN", "a", "denial")):
        assert right4.count(old) == 1
        cut = right4.replace(old, old.replace(" THEN", "\n\nTHEN"))
        found = drafting._last_case_problem(drafting.marker_lines(cut), round4, SHIFT)
        assert found.startswith(f"the first line that holds {value!r} does not begin with 'Case {mark}'")
        assert shift_problem(ins, round4, cut) is not None
        # a line that is only wrapped (no empty line) is one line still for this check
        wrapped = right4.replace(old, old.replace(" THEN", "\n   THEN"))
        assert drafting._last_case_problem(drafting.marker_lines(wrapped), round4, SHIFT) is None
    # the key of the account state stands in the condition only with the account word for an account in use
    off = [line.replace("IF none of the above cases fit", "IF the account_state of the person is inactive")
           for line in drafting.marker_lines(right4)]
    assert "holds the word 'account_state'" in drafting._last_condition_problem(off, round4)
    lines = drafting.marker_lines(right4)
    assert drafting._case_marks_problem(lines, round4, SHIFT) is None
    assert [drafting._CASE_HEAD.match(line).group(1) for line in lines
            if line.startswith("Case")] == ["a", "b", "c"]


def test_the_24_candidates_of_run_seeds_10_to_12_under_each_check_of_today(ins):
    """Every check that refuses each of the 24 old candidates of the prompt of the second desk, with the
    instruction file of today, and how many have a right condition in case c (check C21, part iii)."""
    if not LOG_PATH.exists():
        pytest.skip("text/drafting_log.jsonl is not here")
    key = "policy_shift.system_prompt"
    drawn = [(seed, values, line) for seed, values, line in logged_candidates(key) if seed in (10, 11, 12)]
    assert len(drawn) == 24
    parts, refused_by, passing = dict.fromkeys(("i", "ii", "iii", "iv"), 0), {}, 0
    for seed, values, line in drawn:
        value = drafting.cleaned_value(ins.fields[key], line["raw"])
        lines, plain = drafting.marker_lines(value), [text for text in value.split("\n") if text.strip()]
        found = {"i": drafting._flag_place_problem(lines, values),
                 "ii": drafting._case_marks_problem(lines, values, SHIFT),
                 "iii": drafting._last_condition_problem(lines, values),
                 "iv": drafting._role_kind_problem(lines, values) or drafting._later_giving_problem(lines, values)}
        for part in parts:
            parts[part] += found[part] is not None
        checks = {"C5": drafting._never_grants_problem(plain, values),
                  "C18": drafting._second_desk_problem(lines, values),
                  "C21": drafting._last_case_problem(lines, values, SHIFT),
                  "copy rule": drafting.copy_problem(ins, key, value)}
        names = (seed, tuple(name for name, reason in checks.items() if reason is not None))
        refused_by[names] = refused_by.get(names, 0) + 1
        passing += drafting.check_value(ins, key, value, values) is None
        # the condition is right by part iii exactly where the writer wrote the account in use
        right = "Case c: IF the account_state of the person is active. THEN" in value
        assert (found["iii"] is None) == right == (found["i"] is None)
    assert parts == {"i": 15, "ii": 0, "iii": 15, "iv": 0}     # 9 of the 24 have a right condition
    assert passing == 0
    assert refused_by == {(10, ("C5", "C21")): 1, (10, ("C5",)): 1, (10, ("C21",)): 1,
                          (11, ("C5",)): 8, (11, ("C5", "C21")): 7, (11, ("C5", "C18", "C21")): 1,
                          (12, ("C18", "C21")): 4, (12, ("C21",)): 1}


# ---------------------------------------------------------------------------
# Check C21 after the second review of 2026-11-18: three more holes, each on both sides of its rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition, reason", [
    # hole 1: a wrong condition made only of words that the list held before (manager, approval, role)
    ("IF the role is a manager role.", "holds the word 'manager'"),
    ("IF the request is for a manager role.", "holds the word 'manager'"),
    ("IF the role is for a manager.", "holds the word 'manager'"),
    ("IF the person is a manager.", "holds the word 'manager'"),
    ("IF the request has approval.", "holds the word 'approval'"),
    ("IF the role has approval.", "holds the word 'approval'"),
    ("IF the person has no manager.", "holds the word 'manager'"),
    # the word "role" stands only where the condition says that the person does not hold the role
    ("IF no case above fits and the person does hold the role.", "speaks of the role"),
    ("IF the account_state of the person is active and the role is not a b role.", "speaks of the role"),
    ("IF the person has the role.", "speaks of the role"),
    ("IF the role is already held.", "speaks of the role"),
    ("IF the request is for a role of the person.", "speaks of the role"),
    # "not" or "no" with one small word between it and the account word
    ("IF the account of the person is not an active account.", "holds 'not an active'"),
    # the other side: natural right conditions, also four that the list refused before
    ("IF the account is active and the role is not held yet.", None),
    ("IF the account_state is active and the person does not already hold the role.", None),
    ("IF none of the above cases fit, regardless of the role.", None),
    ("IF cases a and b do not fit, whatever the role.", None),
    ("IF no earlier case fits, for any role.", None),
    ("IF the person's account is active and the person does not hold the role.", None),
    ("IF the account is active and holds_role is false.", None),
    ("In every other case.", None),
    ("For every other request.", None),
    ("All other requests.", None),
    ("IF none of the previous cases apply.", None),
    ("IF neither case a nor case b fits.", None),
])
def test_check_c21_second_review_listed_words_cannot_limit_case_c(ins, round4, right4, condition, reason):
    found = shift_problem(ins, round4, right4.replace("Case c: IF none of the above cases fit.",
                                                      "Case c: " + condition))
    if reason is None:
        assert found is None, found
    else:
        assert reason in found and found.endswith("(check C21)"), found


@pytest.mark.parametrize("name, old, new, reason", [
    # hole 2: a kind of role outside the line of case c, in words that are neither the key nor the stem
    ("step 3", "Follow the first case that fits.",
     "Follow the first case that fits. Only restricted roles reach case c.", "kind of role ('restricted')"),
    ("step 3, the other kind", "Follow the first case that fits.",
     "Follow the first case that fits. A request for an ordinary role is closed in case a.",
     "kind of role ('ordinary')"),
    ("case a", "Case a: IF the account_state of the person is inactive.",
     "Case a: IF the account_state of the person is inactive or the role is not restricted.",
     "kind of role ('restricted')"),
    ("case b", "Case b: IF the value of holds_role is true.",
     "Case b: IF the value of holds_role is true or the role is unrestricted.", "kind of role ('unrestricted')"),
    ("the first paragraph", "You settle this request using your tools.",
     "You settle this request using your tools. Only privileged roles need approval at this desk.",
     "kind of role ('privileged')"),
    ("the Finally line", "Asking for approval happens at most once.",
     "Asking for approval happens at most once, and only for a privileged role.", "kind of role ('privileged')"),
    # the line of step 2 tells what the result holds; a sentence that also tells what to do is refused
    ("step 2, the key decides", "and the value of holds_role.",
     "and the value of holds_role. Only a role with is_sensitive true goes to approval.",
     "a sentence of the line of step 2 holds the result key 'is_sensitive' together with"),
    ("step 2, a kind word decides", "and the value of holds_role.",
     "and the value of holds_role. A restricted role is for case c.",
     "a sentence of the line of step 2 holds a word for a kind of role ('restricted') together with"),
    # hole 3: a limit behind the action of case c
    ("case c, behind the action", "THEN call request_approval using the manager_id from step 1.",
     "THEN call request_approval using the manager_id from step 1, for restricted roles.",
     "kind of role ('restricted')"),
    ("case c, the reviewer's words", "THEN call request_approval using the manager_id from step 1.",
     "THEN call request_approval using the manager_id from step 1, for restricted roles only.",
     "holds one of if, unless, except, only behind its condition"),
    ("case c, only", "THEN call request_approval using the manager_id from step 1.",
     "THEN call request_approval using the manager_id from step 1, for manager roles only.",
     "holds one of if, unless, except, only behind its condition"),
    # the other side: right prompts
    ("case c, even if", "Do not give the role yet.", "Do not give the role, even if the manager agrees.", None),
    ("case c, every role", "Do not give the role yet.", "Do not give the role yet. This applies to every role.",
     None),
    ("the Finally line, never given", "The tool grant_role is never called at this desk.",
     "The tool grant_role is never called at this desk, because a role is never given here.", None),
    ("step 2, the key changes nothing", "and the value of holds_role.",
     "and the value of holds_role. At this desk the value of is_sensitive changes nothing.", None),
    ("case c, no IF", "Case c: IF none of the above cases fit. THEN call", "Case c: Otherwise, call", None),
    # part v: the giving tool with a word that says "later"
    ("case c, until", "Do not give the role yet.", "Do not call grant_role until the manager approves.",
     "names grant_role in a sentence with one of until, till, before, after, unless"),
    ("case c, before", "Do not give the role yet.", "Never call grant_role before the approval comes.",
     "names grant_role in a sentence with one of until, till, before, after, unless"),
    ("case c, the giving tool, right", "Do not give the role yet.", "Do not call grant_role.", None),
    ("case c, even after", "Do not give the role yet.",
     "Do not call grant_role, not even after the manager agrees.", None),
    ("case c, a small then", "Case c: IF none of the above cases fit. THEN call",
     "Case c: If no case above fits, then call", None),
])
def test_check_c21_second_review_a_kind_of_role_outside_the_condition(ins, round4, right4, name, old, new,
                                                                     reason):
    assert right4.count(old) == 1, name
    found = shift_problem(ins, round4, right4.replace(old, new))
    if reason is None:
        assert found is None, found
    else:
        assert reason in found and found.endswith("(check C21)"), found


def test_check_c21_second_review_the_condition_ends_at_the_first_action(ins, round4):
    """In a line without THEN the action begins at its verb, so the verb and the words behind it are not
    words of the condition. The list needs no word of an action then."""
    for line, condition in (
            ("Case c: If no case above fits, call request_approval with the manager_id.", "If no case above fits"),
            ("Case c: In every other case: ask the person's manager for approval with request_approval.",
             "In every other case"),
            ("Case c: In every other case, request approval with request_approval.", "In every other case"),
            ("Case c: Call request_approval with the manager_id.", ""),
            ("Case c: IF cases a and b do not fit. THEN call request_approval.", "IF cases a and b do not fit")):
        assert drafting.last_condition(line, round4) == condition
    # a verb in the middle of a clause ends nothing: the words behind it are read
    assert drafting.last_condition("Case c: IF the person did give a reason. THEN call request_approval.",
                                   round4) == "IF the person did give a reason"
    assert not {"call", "ask", "manager", "approval", "with"} & set(drafting.LAST_CONDITION_WORDS)
    assert {"held", "already", "regardless", "whatever"} <= set(drafting.LAST_CONDITION_WORDS)
    assert drafting.SECOND_CONDITION_WORDS == ("if", "unless", "except", "only")
    assert not any(drafting._begins(word, drafting.ROLE_KIND_STEMS) for word in drafting.LAST_CONDITION_WORDS)
    # the stems for a kind of role and the other small lists of the second review, letter for letter
    assert drafting.ROLE_KIND_STEMS == (
        "restrict", "unrestrict", "nonrestrict", "privileg", "unprivileg", "flag", "unflag", "special",
        "critical", "protected", "unprotected", "elevated", "confidential", "admin", "ordinary", "normal",
        "regular", "risk", "secret", "classified")
    assert drafting.STEP_2_NOT_WITH == ("approv", "case", "only", "manager")
    assert drafting.ACTION_WORDS == ("call", "ask", "request", "close", "give", "record", "use", "send", "do")
    assert drafting.LATER_WORDS == ("until", "till", "before", "after", "unless")
    assert drafting.ROLE_WORDS == ("role", "hold", "held")
