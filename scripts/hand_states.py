"""The form for the six end states that a person writes by hand, and the check of the filled form.

    .venv/bin/python scripts/hand_states.py make  --scenarios DIR --split calib --seed 0 --out FILE
    .venv/bin/python scripts/hand_states.py check --form FILE --scenarios DIR --split calib

PLAN 5.2: "For 6 scenarios you write the expected final state by hand, without the generator, and it must
match." The point: a person who reads only the rules of the desk must arrive at the end state that the
generator stored. If not, the rules and the generator do not mean the same thing.

make. Six families of the split are chosen by a fixed rule and a seed (`choose_six`), so nobody picks easy
ones. The form is one YAML file. Its comments show, for each family: the request, the people who have the
full name of the request, the role of the request, and who of these people holds the role. Four answer
fields are empty. The form shows nothing of what the generator expects: `question` reads only the request
and the world of a family. The rules at the top of the form are printed from accessdesk/env/rules.py
(`ordered_rules`), so the form cannot drift from the code.
An existing file is never written over: it may hold answers.

check. For each of the six families the end state is built from the four answers with the tools of the desk
(accessdesk/env/tools.py): a grant if the answer says so, an approval record if the answer names somebody,
then the closing. It is compared with the stored expected state by the verifier (accessdesk.impl
get("verify")). The generator's own rules code (decide, expected_state) is not called for this.
Each family gets one line: MATCH, DIFFERS (with the answers that differ), NOT ANSWERED or CANNOT BE READ.
The last line is "6 of 6 match", or it names what is open.

One form has one rule set. It stands once at the top of the form as data and once in each family. The
check refuses a form in which these lines do not agree, and the first line of its output names the rule
set. (Before, each family line was read alone: one changed line turned a wrong answer into a match.)

Each run that gave verdicts is added to a log beside the form (FORM.checks.log) with its number, the time
and the SHA-256 of the form. The output names the run. The first run is the one that counts: a person who
changes an answer until the check says MATCH has not written the state by hand.
A check while every family still has an empty answer is no run: no state is built, no answer is compared,
and the log gets nothing. So a check of the form before it is filled in does not use up the first run.
As soon as one family has its four answers it gets a verdict, and the check is a run, also if other
families are still open: else answers could be tried one by one without a count.

The check reads only the data lines of the form (`key: value`). A comment line that was added or changed,
also a line of a table, is not noticed: the facts of a family are read from the split file by its id.

Exit code of check: 0 if all six match, else 1. If the check could not run (the form or the scenario files
cannot be used), the reason is on stderr, the last line is "0 of 6 match; open: the check did not run",
and the code is 1 as well. make ends with 0, or with 2 if it wrote nothing.

The text file (--text) is only needed to call the tools: no word of it reaches an end state. Without
--text the placeholder text is used. The split file must be the file that the build wrote (its SHA-256 is
in the folder's summary.json), and the form remembers that SHA-256. Families of the test split are refused
unless --allow-test is given, as in scripts/play.py.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.env.generator import SPLITS  # noqa: E402
from accessdesk.env.rules import RULESETS, reference_plan, target_facts  # noqa: E402
from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.env.tools import execute  # noqa: E402
from accessdesk.env.world import World  # noqa: E402
from accessdesk.rng import Rng, derive_seed  # noqa: E402
from accessdesk.text import TextBundle, load_text, model_text_path, normalise, placeholder_path  # noqa: E402
from accessdesk.types import OUTCOMES, RESOLUTIONS  # noqa: E402

FORM_NAME = "hand_states"
FORM_VERSION = "2"                      # 2: the rule set stands once at the top as data; no folder line
N_FAMILIES = 6
N_EXTRA = N_FAMILIES - len(OUTCOMES)    # the families beyond one per outcome: 2
N_SAME_NAME = 2                         # at least so many of the six have two people with one full name
SEED_LABEL = "hand states"

# The words of the form for the four resolutions (internal key -> word). They are not the words of the
# code and not the values that the model writes: no word of the form's own text holds an outcome word or
# a resolution key (a test checks it).
WORD_OF_RESOLUTION = {"granted": "role given", "pending": "manager asked", "already": "held before",
                      "denied": "refused"}
YES, NO, NOBODY = "yes", "no", "none"
# What a person also types for "nobody was asked". The form names only the first word.
NOBODY_WORDS = (NOBODY, "no", "nobody", "no one", "-")
ANSWER_KEYS = ("person", "resolution", "role_given", "asked_to_approve")
ENTRY_KEYS = ("family", "ruleset") + ANSWER_KEYS
TOP_KEYS = ("form", "version", "split", "split_file_sha256", "seed", "ruleset", "families")
# One answer line of the form: the key with its colon, then what the person typed up to the line end.
ANSWER_LINE = re.compile(r"^([ \t]*(?:%s):)([^\r\n]*)" % "|".join(ANSWER_KEYS), re.MULTILINE)
ONLY_A_DASH = re.compile(r"\s*-+\s*(#.*)?")
LOG_SUFFIX = ".checks.log"              # the log of the check runs: the name of the form plus this
RUN_MARK = "=== run "                   # each run in the log starts with a line that starts like this
# What the check prints in place of the run line while every family still has an empty answer.
NO_RUN = ("this is no run of the form: no family has all four answers, so no state was built and no "
          "answer was compared. Nothing is added to the log.")

# The three rule facts, in the order of rules.target_facts: (active, holds the role, sensitive). Each
# line: (position of the fact, the value that a rule asks for, the condition in words).
CONDITIONS = ((0, False, "The account of the person is not active."),
              (1, True, "The person holds the role now."),
              (2, True, "The role is sensitive."))
OTHERWISE = "In every other case."

MATCH, DIFFERS, NOT_ANSWERED, CANNOT_BE_READ = "MATCH", "DIFFERS", "NOT ANSWERED", "CANNOT BE READ"
OTHER_PART = ("a part of the state that the four answers do not set (the role id, the access list or the "
              "pair in a record)")
DID_NOT_RUN = "the check did not run"
RULE = "# " + "-" * 78

Facts = tuple[bool, bool, bool]
Effect = tuple[bool, bool, str]         # (the desk gives the role, it asks the manager, the resolution key)


class FormError(ValueError):
    """The form, or one answer in it, cannot be used. The message says why in plain words."""


class Verdict(NamedTuple):
    family_id: str
    status: str         # MATCH, DIFFERS, NOT_ANSWERED or CANNOT_BE_READ
    detail: str         # what differs, which fields are empty, or what cannot be read; "" for a match


# ---------------------------------------------------------------------------
# What a family shows
# ---------------------------------------------------------------------------
def same_name_people(family: dict) -> list[dict]:
    """The people of the world who have the full name of the request, in the order of the world. This is
    the list that the lookup tool returns for that name."""
    wanted = normalise(family["request"]["person_name"])
    return [person for person in family["world"]["people"] if normalise(person["name"]) == wanted]


def requested_role(family: dict) -> dict:
    """The role of the world that has the role name of the request. Raises ValueError unless the world
    holds exactly one such role."""
    wanted = normalise(family["request"]["role_name"])
    roles = [role for role in family["world"]["roles"] if normalise(role["name"]) == wanted]
    if len(roles) != 1:
        raise ValueError(f"family {family['family_id']}: the world must hold the role of the request "
                         f"exactly once, not {len(roles)} times")
    return roles[0]


def question(family: dict) -> dict:
    """What the form shows of a family, and nothing else.

    {"family_id", "wording": the first wording of the request, "person_name",
     "people": one dict per person with the full name of the request (person_id, department, active,
               manager_id, holds_role), "role": {role_id, name, sensitive}}

    Only the keys "family_id", "wordings", "request" and "world" of the family are read. The outcome, the
    facts, the expected states, the turn budgets and "truth" are not read, so the form cannot show them.
    """
    role = requested_role(family)
    access = {tuple(pair) for pair in family["world"]["access"]}
    people = [{"person_id": person["person_id"], "department": person["department"],
               "active": person["status"] == "active", "manager_id": person["manager_id"],
               "holds_role": (person["person_id"], role["role_id"]) in access}
              for person in same_name_people(family)]
    return {"family_id": family["family_id"], "wording": family["wordings"][0]["text"],
            "person_name": family["request"]["person_name"], "people": people,
            "role": {"role_id": role["role_id"], "name": role["name"], "sensitive": bool(role["sensitive"])}}


# ---------------------------------------------------------------------------
# The choice of the six families
# ---------------------------------------------------------------------------
def rule_facts_true(family: dict) -> int:
    """How many of the three rule facts are true for the person and the role that the request means: the
    account is not active, the person holds the role, the role is sensitive. 0 to 3."""
    active, holds_role, sensitive = target_facts(family)
    return (not active) + holds_role + sensitive


def has_same_name_pair(family: dict) -> bool:
    """True if two or more people of the world have the full name of the request."""
    return len(same_name_people(family)) > 1


def choose_six(families: list[dict], seed: int) -> list[dict]:
    """The six families of the form, in the order of the form. The same seed and the same families give
    the same six; the order of `families` does not matter.

    The limits, in the order in which they are served:
    1. One family of each of the four outcomes.
    2. Two more families in which more than one rule fact is true. The two are of different outcomes if
       the split has such a pair. In the mixed design of the generator one of them then holds a
       sensitive role (the rule for a held role must come before the rule for a sensitive role), and
       in the other the account is not active (that rule must come first). Without such a pair the two
       have different facts, or are any two. If the split has too few such families, the places that
       are left go to families of different outcomes.
    3. At least two of the six have two people with one full name, as far as the split allows: while
       there are too few, a chosen family without such a pair is replaced by one with a pair that has
       the same outcome and the same rule facts (so limits 1 and 2 still hold).
    Inside these limits every draw comes from Rng(derive_seed(seed, "hand states")). At the end the six
    are shuffled, so the place of a family in the form says nothing about it.
    Raises ValueError if an outcome has no family or the split has fewer than six families.
    """
    by_id = sorted(families, key=lambda family: family["family_id"])
    if len({family["family_id"] for family in by_id}) != len(by_id):
        raise ValueError("a family id is in the split twice")
    rng = Rng(derive_seed(seed, SEED_LABEL))
    chosen: list[dict] = []
    for outcome in OUTCOMES:
        of_outcome = [family for family in by_id if family["outcome"] == outcome]
        if not of_outcome:
            raise ValueError(f"the split has no family with the outcome {outcome!r}")
        chosen.append(rng.choice(of_outcome))
    for _ in range(N_EXTRA):
        chosen.append(_one_more(by_id, chosen, rng))
    _bring_in_same_name_pairs(by_id, chosen, rng)
    return rng.shuffle(chosen)


def _one_more(families: list[dict], chosen: list[dict], rng: Rng) -> dict:
    """One more family for limit 2: the first family, in a seeded order of the families not chosen yet,
    of the best group that is not empty. The groups, best first:
    more than one rule fact true and another outcome than the extra family before it;
    more than one rule fact true and other facts than the extra family before it;
    more than one rule fact true;
    another outcome than the extra family before it;
    any family."""
    taken = {family["family_id"] for family in chosen}
    free = rng.shuffle([family for family in families if family["family_id"] not in taken])
    if not free:
        raise ValueError(f"the split has fewer than {N_FAMILIES} families")
    extras = chosen[len(OUTCOMES):]

    def other_outcome(family: dict) -> bool:
        return all(family["outcome"] != extra["outcome"] for extra in extras)

    def other_facts(family: dict) -> bool:
        return all(target_facts(family) != target_facts(extra) for extra in extras)

    mixed = [family for family in free if rule_facts_true(family) > 1]
    groups = [[family for family in mixed if other_outcome(family)],
              [family for family in mixed if other_facts(family)],
              mixed,
              [family for family in free if other_outcome(family)],
              free]
    return next(group for group in groups if group)[0]


def _bring_in_same_name_pairs(families: list[dict], chosen: list[dict], rng: Rng) -> None:
    """Limit 3 of `choose_six`: change `chosen` in place until N_SAME_NAME of its families have a
    same-name pair, or until no place can be changed. The places are tried in a seeded order."""
    for place in rng.shuffle(range(len(chosen))):
        if sum(has_same_name_pair(family) for family in chosen) >= N_SAME_NAME:
            return
        current, taken = chosen[place], {family["family_id"] for family in chosen}
        if has_same_name_pair(current):
            continue
        twins = [family for family in families
                 if family["family_id"] not in taken and has_same_name_pair(family)
                 and family["outcome"] == current["outcome"]
                 and target_facts(family) == target_facts(current)]
        if twins:
            chosen[place] = rng.choice(twins)


def limits_not_met(six: list[dict]) -> list[str]:
    """One plain line for each limit of `choose_six` that the six families do not meet (the split had too
    few families of a kind). Empty if all limits are met."""
    lines = []
    others_cover = [set(f["outcome"] for f in six if f not in pair) == set(OUTCOMES)
                    for pair in itertools.combinations([f for f in six if rule_facts_true(f) > 1], N_EXTRA)]
    if not any(others_cover):
        lines.append(f"fewer than {N_EXTRA} families beyond one per outcome have more than one rule fact "
                     "true: the split has too few of them")
    with_pair = sum(has_same_name_pair(family) for family in six)
    if with_pair < N_SAME_NAME:
        lines.append(f"only {with_pair} of the {len(six)} families have two people with one full name: "
                     "the split has no more that fit")
    return lines


# ---------------------------------------------------------------------------
# The rules of the desk, read from the code
# ---------------------------------------------------------------------------
def effect_of_rules(facts: Facts, ruleset: str) -> Effect:
    """What the desk does when the three rule facts are `facts`: (it gives the role, it asks the manager
    of the person to approve, the resolution key of the closing).

    It is read from rules.reference_plan, run on a small world of one person, one manager and one role
    that has exactly these facts. Raises ValueError if the plan is one that the form has no words for (two
    writes of one kind, an approver who is not the manager, or no closing at the end).
    """
    active, holds_role, sensitive = facts
    pair = {"person_id": "person", "role_id": "role"}
    probe = {"world": {"people": [{"person_id": "person", "name": "P", "department": "D",
                                   "status": "active" if active else "inactive", "manager_id": "manager"},
                                  {"person_id": "manager", "name": "M", "department": "D",
                                   "status": "active", "manager_id": None}],
                       "roles": [{"role_id": "role", "name": "R", "sensitive": sensitive}],
                       "access": [["person", "role"]] if holds_role else []},
             "request": {"person_name": "P", "department": "D", "role_name": "R"},
             "truth": {**pair, "manager_id": "manager"}}
    plan = reference_plan(probe, ruleset)
    writes = [call for call in plan if call[0] in ("grant_role", "request_approval")]
    gives = ("grant_role", pair) in writes
    asks = ("request_approval", {**pair, "approver_id": "manager"}) in writes
    closing = plan[-1] if plan else ("", {})
    resolution = closing[1].get("resolution")
    if (len(writes) != gives + asks or closing[0] != "close_request" or resolution not in WORD_OF_RESOLUTION
            or {key: closing[1].get(key) for key in pair} != pair):
        raise ValueError(f"the form has no words for the plan of the rule set {ruleset!r} with the facts "
                         f"{facts}: {plan}")
    return gives, asks, resolution


def ordered_rules(ruleset: str) -> list[tuple[int | None, Effect]]:
    """The rules of a rule set as a list in order: (the number of a line of CONDITIONS, or None for "in
    every other case"; what the desk then does). The first rule whose condition holds decides.

    The list is worked out from `effect_of_rules` for all eight sets of facts, so it follows the code of
    accessdesk/env/rules.py. A condition comes next in the list if, among the sets of facts that no
    earlier rule took, all those with the condition have one effect. For the rule sets of rules.py this
    leaves one list only. Raises ValueError if the rules cannot be written as such a list.
    """
    table = {facts: effect_of_rules(facts, ruleset) for facts in itertools.product((True, False), repeat=3)}
    rows, rules = list(table), []
    while len({table[row] for row in rows}) > 1:
        found = _next_rule(rows, table)
        if found is None:
            raise ValueError(f"the rules of the rule set {ruleset!r} cannot be written as a list in which "
                             "each rule looks at one fact: the form needs new words")
        number, taken = found
        rules.append((number, table[taken[0]]))
        rows = [row for row in rows if row not in taken]
    return rules + [(None, table[rows[0]])]


def _next_rule(rows: list[Facts], table: dict[Facts, Effect]) -> tuple[int, list[Facts]] | None:
    """(the number of the first condition that takes some but not all of `rows` and gives them all one
    effect, the rows that it takes), or None if no condition does."""
    for number, (position, value, _) in enumerate(CONDITIONS):
        taken = [row for row in rows if row[position] == value]
        if taken and len(taken) < len(rows) and len({table[row] for row in taken}) == 1:
            return number, taken
    return None


def effect_words(effect: Effect) -> str:
    """One effect in the words of the form, for example "The desk gives the role. It asks nobody.
    Resolution: role given." """
    gives, asks, resolution = effect
    give = "The desk gives the role." if gives else "The desk does not give the role."
    ask = "It asks the manager of the person to approve." if asks else "It asks nobody."
    return f"{give} {ask} Resolution: {WORD_OF_RESOLUTION[resolution]}."


def rule_lines(ruleset: str) -> list[str]:
    """The rules of a rule set as numbered lines of text, in order: the condition, then the effect."""
    lines = []
    for position, (number, effect) in enumerate(ordered_rules(ruleset), start=1):
        condition = OTHERWISE if number is None else CONDITIONS[number][2]
        lines += [f"  {position}. {condition}", f"       {effect_words(effect)}"]
    return lines


# ---------------------------------------------------------------------------
# The text of the form
# ---------------------------------------------------------------------------
def table_lines(head: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    """A small table as lines of text: the head, then one line per row, the columns padded to one width."""
    lines = [head, *rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
    return ["   ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines]


def yes_no(value: bool) -> str:
    return YES if value else NO


def prompt_file() -> str:
    """The file with the text that the model reads, as a path from the top folder of the repository."""
    return os.path.relpath(model_text_path(), ROOT)


def prompt_key(ruleset: str) -> str:
    """The key of the text file under which the system prompt with the rules of this rule set stands."""
    if ruleset == "default":
        return "system_prompt"
    return "policy_shift, then system_prompt (if the variant named there is this rule set)"


def header_lines(ruleset: str, check_command: str) -> list[str]:
    """The top of the form as lines of text (without the comment sign): the task in five sentences, who
    is meant, the rules of the rule set in order, and the four answers."""
    words = " | ".join(WORD_OF_RESOLUTION[key] for key in RESOLUTIONS)
    return [
        "The end states of six requests, written by hand",
        "",
        "What to do",
        "  1. This form holds six requests to the access desk; a fixed rule and a seed chose them.",
        "  2. Each request comes with the facts that the tools of the desk would show.",
        "  3. Work out with the rules below what the desk must do, and write four answers under each one.",
        "  4. Do not open the scenario file and do not run the generator before the form is checked.",
        "  5. Then run the check: it builds the end state from your answers with the tools of the desk and",
        "     compares it with the end state that the generator stored.",
        "",
        f"     {check_command}",
        "",
        "     Each run of the check is added to a log beside this form. The first run is the one that",
        "     counts: if an answer differs, find out why before you change it.",
        "",
        "Who is meant",
        "  The request names a person and a department. The person meant is the one with this full name.",
        "  If two people have this full name, it is the one who works in the department of the request.",
        "",
        f"The rules of the desk (rule set: {ruleset})",
        "  Take the rules in this order. The first rule that fits decides, and no later rule is looked at.",
        *rule_lines(ruleset),
        "  This list is printed from the code (accessdesk/env/rules.py): it is what the generator follows.",
        "  Read it once against the rules in the system prompt that the model reads. That prompt is in",
        f"  the file {prompt_file()}, under the key {prompt_key(ruleset)}.",
        "  If the two differ, stop: that is a finding, and the answers below would only repeat it.",
        "",
        "The four answers",
        "  person            the person id of the person meant",
        f"  resolution        one of: {words}",
        f"  role_given        {YES} if the desk gives the role in this request, else {NO}",
        f"  asked_to_approve  the person id of the manager whom the desk asks, or {NOBODY}",
        "  Write each answer behind the colon, with a space after the colon. Leave no answer empty.",
        "  Change nothing else in this file.",
    ]


def entry_lines(family: dict, ruleset: str, position: int) -> list[str]:
    """The part of the form for one family: the two lines that name the family and the rule set, what
    `question` shows as comments, and the four empty answers."""
    shown = question(family)
    people = [(person["person_id"], person["department"], "active" if person["active"] else "not active",
               person["manager_id"] or NOBODY, yes_no(person["holds_role"])) for person in shown["people"]]
    role = shown["role"]
    role_row = (role["role_id"], role["name"], yes_no(role["sensitive"]))
    info = [f"Request {position} of {N_FAMILIES}: the next two lines name it and its rule set. "
            "Do not change them."]
    data = [f"- family: {json.dumps(shown['family_id'])}", f"  ruleset: {ruleset}"]
    body = ["The request:", *[f"  {line}" for line in shown["wording"].splitlines()], "",
            f"People with the full name {json.dumps(shown['person_name'], ensure_ascii=False)}:",
            *["  " + line for line in table_lines(
                ("person id", "department", "account", "manager id", "holds the role now"), people)], "",
            "The role of the request:",
            *["  " + line for line in table_lines(("role id", "name", "sensitive"), [role_row])],
            "", "Your answers (the words to use are at the top of the form):"]
    answers = [f"  {key}:" for key in ANSWER_KEYS]
    return [RULE, *_comment(info), RULE, *data, *_comment(body, indent="  "), *answers, ""]


def _comment(lines: list[str], indent: str = "") -> list[str]:
    """The lines as YAML comments."""
    return [(f"{indent}# {line}").rstrip() for line in lines]


def form_text(six: list[dict], *, split: str, seed: int, ruleset: str, split_file_sha256: str,
              check_command: str) -> str:
    """The whole form as YAML text: the top as comments, the data lines that the check reads again, and
    one part per family in the order of `six`. The rule set stands once at the top as data and once in
    each family: the check refuses a form in which these lines do not agree."""
    if ruleset not in RULESETS:
        raise ValueError(f"unknown rule set {ruleset!r}; known: {RULESETS}")
    lines = ["# " + "=" * 78, *_comment(header_lines(ruleset, check_command)), "# " + "=" * 78, "",
             f"form: {FORM_NAME}", f"version: {json.dumps(FORM_VERSION)}", f"split: {split}",
             f"split_file_sha256: {json.dumps(split_file_sha256)}", f"seed: {json.dumps(str(seed))}",
             f"ruleset: {ruleset}", "families:", ""]
    for position, family in enumerate(six, start=1):
        lines += entry_lines(family, ruleset, position)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading a form that a person has filled
# ---------------------------------------------------------------------------
class _FormLoader(yaml.BaseLoader):
    """Every value stays the text that was typed (yes is not turned into True, an id never into a number),
    and a key that is written twice in one place is refused (plain YAML keeps the last one silently)."""

    def construct_mapping(self, node, deep=False):  # noqa: ANN001 (signature of the base class)
        seen: list[object] = []
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"the key {key!r} is written twice",
                                                        key_node.start_mark)
            seen.append(key)
        return super().construct_mapping(node, deep)


def typed(value: object) -> str:
    """One value of the form as the person typed it: without white space and without quote signs at its
    two ends. An empty field is "". Raises FormError for a value that is a list or a mapping."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise FormError(f"a field holds a list or a mapping where one word is expected: {value!r}")
    return value.strip().strip("\"'\u201c\u201d\u2018\u2019").strip()


def readable_answers(text: str) -> str:
    """The text of the form with two habits of a person made readable for YAML. Only the four answer
    lines are touched, and only behind their colon: a Tab becomes a space (YAML refuses a Tab there), and
    an answer that is only a dash is put in quotes (YAML reads a bare dash as the start of a list)."""
    def repaired(line: re.Match) -> str:
        answer = line.group(2).replace("\t", " ")
        dash = ONLY_A_DASH.fullmatch(answer)
        if dash:
            answer = f' "-" {dash.group(1) or ""}'.rstrip()
        return line.group(1) + answer

    return ANSWER_LINE.sub(repaired, text)


def read_form(text: str) -> dict:
    """The filled form as data: {"split", "seed" (int), "split_file_sha256", "ruleset",
    "families": [{"family", "ruleset", "person", "resolution", "role_given", "asked_to_approve"}, ...]},
    every answer as typed (`typed`), "" if it is empty.

    Raises FormError, with a plain message, if the text is not YAML, is not a form of this script, lacks a
    key, has a key that the form does not have, or does not hold six different families.
    """
    try:
        data = yaml.load(readable_answers(text), Loader=_FormLoader)
    except yaml.YAMLError as error:
        raise FormError("the form cannot be read as YAML (each answer needs a space after its colon, no "
                        "line may start with a Tab, and no other line may be changed): "
                        f"{' '.join(str(error).split())}") from None
    if not isinstance(data, dict) or data.get("form") != FORM_NAME:
        raise FormError(f"this is not a form of scripts/hand_states.py (form: {FORM_NAME}, version: "
                        f"{FORM_VERSION})")
    if data.get("version") != FORM_VERSION:
        raise FormError(f"this form is of version {typed(data.get('version'))!r}, and the script reads "
                        f"version {FORM_VERSION!r}: make the form again")
    _check_keys(data, TOP_KEYS, "the form")
    entries = data["families"]
    if not isinstance(entries, list) or len(entries) != N_FAMILIES:
        raise FormError(f"the form must hold {N_FAMILIES} families")
    for entry in entries:
        _check_keys(entry, ENTRY_KEYS, "a family of the form")
    families = [{key: typed(entry[key]) for key in ENTRY_KEYS} for entry in entries]
    if len({family["family"] for family in families}) != N_FAMILIES:
        raise FormError("a family is in the form twice")
    seed = typed(data["seed"])
    if not seed.lstrip("-").isdigit():
        raise FormError(f"the seed of the form must be a whole number, not {seed!r}")
    return {"split": typed(data["split"]), "seed": int(seed), "ruleset": typed(data["ruleset"]),
            "split_file_sha256": typed(data["split_file_sha256"]), "families": families}


def _check_keys(mapping: object, keys: tuple[str, ...], what: str) -> None:
    """Raise FormError unless `mapping` is a mapping with exactly these keys."""
    if not isinstance(mapping, dict):
        raise FormError(f"{what} is not a list of lines of the kind `key: value`")
    missing, extra = [k for k in keys if k not in mapping], [k for k in mapping if k not in keys]
    if missing or extra:
        raise FormError(f"{what} must have the keys {list(keys)}: missing {missing}, not known {extra}")


def as_word(value: str) -> str:
    """An answer that is a word, as it is compared: small letters, one space between words, and no full
    stop at its end."""
    return " ".join(value.casefold().rstrip(". ").split())


def as_person_id(value: str, people: set[str]) -> str | None:
    """The person id that was typed, or None if `people` has no such id. An id must be typed exactly; a
    full stop at its end is dropped only if the id with the full stop is no person."""
    return next((candidate for candidate in (value, value.rstrip(". ")) if candidate in people), None)


def answers_of(entry: dict, family: dict) -> dict:
    """The four answers of one family in the form that the tools take:
    {"person_id", "resolution": the internal resolution key, "gives": bool, "approver_id": str or None}.

    Case, the number of spaces and a full stop at the end do not matter in the three answers that are
    words; a person id must be typed exactly. For "nobody was asked" every word of NOBODY_WORDS is read
    (a person types `no` under a field that takes yes or no).
    Raises FormError, naming every answer that cannot be read: a person id that the family does not
    have, a resolution that is not one of the four words, a `role_given` that is not yes or no.
    """
    people = {person["person_id"] for person in family["world"]["people"]}
    key_of_word = {word: key for key, word in WORD_OF_RESOLUTION.items()}
    resolution = as_word(entry["resolution"].replace("_", " ").replace("-", " "))
    given = as_word(entry["role_given"])
    person = as_person_id(entry["person"], people)
    nobody = as_word(entry["asked_to_approve"]) in NOBODY_WORDS
    approver = None if nobody else as_person_id(entry["asked_to_approve"], people)
    problems = []
    if person is None:
        problems.append(f"person: no person of this family has the id {entry['person']!r}")
    if resolution not in key_of_word:
        problems.append(f"resolution: {entry['resolution']!r} is not one of {list(key_of_word)}")
    if given not in (YES, NO):
        problems.append(f"role_given: {entry['role_given']!r} is not {YES} or {NO}")
    if not nobody and approver is None:
        problems.append(f"asked_to_approve: {entry['asked_to_approve']!r} is not {NOBODY} and not a person "
                        "id of this family")
    if problems:
        raise FormError("; ".join(problems))
    return {"person_id": person, "resolution": key_of_word[resolution], "gives": given == YES,
            "approver_id": approver}


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
def build_state(family: dict, answers: dict, text: TextBundle) -> dict:
    """The end state that the answers describe, built with the tools of the desk on the family's world:
    grant_role if the answers say that the desk gives the role, request_approval if they name an approver,
    then close_request. The role is the role of the request. No rule of the desk is applied here.
    Raises FormError if a tool refuses a call (it then wrote nothing)."""
    world = World(family["world"])
    ids = {"person_id": answers["person_id"], "role_id": requested_role(family)["role_id"]}
    calls: list[tuple[str, dict[str, str]]] = []
    if answers["gives"]:
        calls.append(("grant_role", ids))
    if answers["approver_id"] is not None:
        calls.append(("request_approval", {**ids, "approver_id": answers["approver_id"]}))
    calls.append(("close_request", {"resolution": answers["resolution"], **ids}))
    for tool_key, args in calls:
        outcome = execute(world, text, tool_key, args)
        if outcome.error is not None:
            raise FormError(f"the tool {tool_key} refused the answers ({outcome.error})")
    return world.state()


def differing_answers(built: dict, expected: dict) -> list[str]:
    """Which of the four answers differ between two end states that the verifier found different: "the
    person", "the resolution", "the grant" (a grant record in one state only), "the approval" (not the
    same approver). If none of the four differs, the one entry is OTHER_PART: the states then differ in
    something that no answer sets, which the tools cannot reach from this world."""
    parts = []
    for field, name in (("person_id", "the person"), ("resolution", "the resolution")):
        if built["request"][field] != expected["request"][field]:
            parts.append(name)
    if len(built["grants"]) != len(expected["grants"]):
        parts.append("the grant")
    approvers = [sorted(tuple(record[2:]) for record in state["approvals"]) for state in (built, expected)]
    if approvers[0] != approvers[1]:
        parts.append("the approval")
    return parts or [OTHER_PART]


def check_family(entry: dict, family: dict, text: TextBundle) -> Verdict:
    """The verdict for one family of the form. MATCH only if the verifier says that the state built from
    the answers and the stored expected state of the entry's rule set are the same."""
    family_id = family["family_id"]
    empty = [key for key in ANSWER_KEYS if not entry[key]]
    if empty:
        return Verdict(family_id, NOT_ANSWERED, ", ".join(empty))
    try:
        built = build_state(family, answers_of(entry, family), text)
    except FormError as error:
        return Verdict(family_id, CANNOT_BE_READ, str(error))
    expected = family["expected"][entry["ruleset"]]
    if impl.get("verify")(built, expected):
        return Verdict(family_id, MATCH, "")
    return Verdict(family_id, DIFFERS, ", ".join(differing_answers(built, expected)))


def check_form(form: dict, families: list[dict], text: TextBundle) -> list[Verdict]:
    """One verdict per family of the form, in the order of the form.

    A form has one rule set, `form["ruleset"]`, and all six families are checked against it. So a family
    line that names another rule set stops the check: else one changed line could turn a wrong answer
    into a match.
    Raises FormError if the form cannot be checked against these families: an unknown rule set, a family
    line with another rule set than the form's, a family id that the split does not have, a family that
    stores no end state for the rule set, or six families that are not the six that the form's seed
    chooses.
    """
    ruleset = form["ruleset"]
    if ruleset not in RULESETS:
        raise FormError(f"unknown rule set {ruleset!r}; known: {RULESETS}")
    by_id = {family["family_id"]: family for family in families}
    for entry in form["families"]:
        if entry["ruleset"] != ruleset:
            raise FormError(f"family {entry['family']} names the rule set {entry['ruleset']!r}, but the "
                            f"form is for the rule set {ruleset!r}: a form has one rule set, and the lines "
                            "that name it must not be changed")
        if entry["family"] not in by_id:
            raise FormError(f"the split has no family {entry['family']!r}")
        if ruleset not in by_id[entry["family"]].get("expected", {}):
            raise FormError(f"family {entry['family']} stores no end state for the rule set {ruleset!r}")
    wanted = sorted(family["family_id"] for family in choose_six(families, form["seed"]))
    if sorted(entry["family"] for entry in form["families"]) != wanted:
        raise FormError(f"the families of the form are not the six that the seed {form['seed']} chooses in "
                        f"this split ({', '.join(wanted)})")
    return [check_family(entry, by_id[entry["family"]], text) for entry in form["families"]]


def verdict_line(verdict: Verdict) -> str:
    return f"{verdict.family_id:8s} {verdict.status}" + (f": {verdict.detail}" if verdict.detail else "")


def last_line(verdicts: list[Verdict]) -> str:
    """"6 of 6 match", or the number that match and every family that is open with its status."""
    matches = sum(verdict.status == MATCH for verdict in verdicts)
    line = f"{matches} of {len(verdicts)} match"
    still_open = [f"{verdict.family_id} ({verdict.status.lower()})" for verdict in verdicts
                  if verdict.status != MATCH]
    return line + (f"; open: {', '.join(still_open)}" if still_open else "")


# ---------------------------------------------------------------------------
# Files and the command line
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    """The SHA-256 of the file's bytes, as 64 hexadecimal digits."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_file_problem(scenarios_dir: Path, split: str, allow_test: bool) -> str | None:
    """Why the split file of this folder cannot be used, or None if it can. The file must be there and
    must be the file that the build wrote (summary.json names its SHA-256), and the test split needs
    `allow_test`."""
    if split == "test" and not allow_test:
        return ("the test set is looked at once, at the end. Use the calibration or the dev split, or pass "
                "--allow-test and say so in the notes.")
    summary_path, split_file = scenarios_dir / "summary.json", scenarios_dir / f"{split}.jsonl"
    if not summary_path.is_file():
        return f"{summary_path} is missing: build the scenarios with scripts/build_scenarios.py"
    if not split_file.is_file():
        return f"{split_file} is missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256_of_file(split_file) != summary.get("files_sha256", {}).get(split_file.name):
        return (f"{split_file} is not the file that the build wrote (summary.json names another SHA-256): "
                "build the scenarios again")
    return None


def record_run(form_file: Path, form_sha256: str, lines: list[str]) -> str:
    """Add this run of the check to the log beside the form (the name of the form plus LOG_SUFFIX), and
    return the line that the output shows for it: the number of the run and the name of the log.

    The log gets, for each run: its number, the time (UTC), the SHA-256 of the form as it was checked,
    and `lines` (what the check prints). So a reader sees how often the form was checked and whether the
    answers changed between two runs. The runs are counted in the log: the count starts again if the
    log is removed. The log is a trace for an honest reader, not a proof.
    Nothing is raised if the log cannot be written: the returned line then says so.
    """
    log = form_file.with_name(form_file.name + LOG_SUFFIX)
    try:
        before = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        number = sum(line.startswith(RUN_MARK) for line in before.splitlines()) + 1
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        block = [f"{RUN_MARK}{number} at {when}, SHA-256 of the form {form_sha256}", *lines, ""]
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(block) + "\n")
    except OSError as error:
        return f"this run is in no log ({error}): keep this output yourself"
    return f"run {number} of this form. Each run is added to {log}; the first run is the one that counts."


def make(args: argparse.Namespace) -> int:
    """Write the empty form. 0 if it was written, 2 (with the reason on stderr) if not."""
    problem = split_file_problem(args.scenarios, args.split, args.allow_test)
    if problem is None and args.out.exists():
        problem = f"{args.out} exists and may hold answers: it is not written over. Give another name."
    if problem:
        print(problem, file=sys.stderr)
        return 2
    split_file = args.scenarios / f"{args.split}.jsonl"
    try:
        six = choose_six(load_families(split_file), args.seed)
        command = (f".venv/bin/python scripts/hand_states.py check --form {args.out} "
                   f"--scenarios {args.scenarios} --split {args.split}")
        text = form_text(six, split=args.split, seed=args.seed, ruleset=args.ruleset,
                         split_file_sha256=sha256_of_file(split_file), check_command=command)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(text.encode("utf-8"))
    print(f"wrote {args.out}: {N_FAMILIES} families of the split {args.split}, seed {args.seed}, rule set "
          f"{args.ruleset}: {', '.join(family['family_id'] for family in six)}")
    for line in limits_not_met(six):
        print(f"NOTE: {line}")
    return 0


def check(args: argparse.Namespace) -> int:
    """Check the filled form and print one line per family and the last line. 0 if all six match, else
    1 (also if the check could not run: the reason is then on stderr). A run that gave verdicts is added
    to the log beside the form (`record_run`); a check that could not run gives no verdict and no log.
    If every family is NOT ANSWERED, nothing was built or compared: the output says NO_RUN in place of
    the run line, and the log gets nothing."""
    try:
        if not args.form.is_file():
            raise FormError(f"{args.form} is missing")
        form_bytes = args.form.read_bytes()
        form = read_form(form_bytes.decode("utf-8-sig"))                  # a byte order mark is dropped
        if form["split"] != args.split:
            raise FormError(f"the form was made from the split {form['split']!r}, not from {args.split!r}")
        problem = split_file_problem(args.scenarios, args.split, allow_test=True)
        if problem:
            raise FormError(problem)
        split_file = args.scenarios / f"{args.split}.jsonl"
        if sha256_of_file(split_file) != form["split_file_sha256"]:
            raise FormError(f"the form was made from another file than {split_file} (its SHA-256 is not "
                            "the one in the form): make the form again")
        verdicts = check_form(form, load_families(split_file), load_text(args.text))
    except (ValueError, OSError) as error:                  # FormError and TextError are ValueErrors
        print(f"the check did not run: {error}", file=sys.stderr)
        print(f"0 of {N_FAMILIES} match; open: {DID_NOT_RUN}")
        return 1
    head = [f"form {args.form}: seed {form['seed']}, split {args.split}, rule set {form['ruleset']}",
            f"split file {split_file}, SHA-256 {form['split_file_sha256']}",
            f"verifier: the {impl.which('verify')} version"]
    result = [*[verdict_line(verdict) for verdict in verdicts], last_line(verdicts)]
    if all(verdict.status == NOT_ANSWERED for verdict in verdicts):
        run_line = NO_RUN                 # `check_family` built no state and asked the verifier nothing
    else:
        run_line = record_run(args.form, hashlib.sha256(form_bytes).hexdigest(), head + result)
    print("\n".join([*head, run_line, *result]))
    return 0 if all(verdict.status == MATCH for verdict in verdicts) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, run in (("make", make), ("check", check)):
        command = commands.add_parser(name)
        command.set_defaults(run=run)
        command.add_argument("--scenarios", type=Path, required=True, help="the scenario folder")
        command.add_argument("--split", choices=SPLITS, required=True)
        if name == "make":
            command.add_argument("--seed", type=int, required=True, help="the seed of the choice")
            command.add_argument("--out", type=Path, required=True, help="the file for the empty form")
            command.add_argument("--ruleset", choices=RULESETS, default="default",
                                 help="the rule set of the six requests")
            command.add_argument("--allow-test", action="store_true", help="allow the test split")
        else:
            command.add_argument("--form", type=Path, required=True, help="the filled form")
            command.add_argument("--text", type=Path, default=placeholder_path(),
                                 help="the text file (YAML); only needed to call the tools")
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
