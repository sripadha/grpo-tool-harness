"""The reading sheet, the labels form and the key of the audit, and the check of the filled form.

ANALYSIS_PLAN.md section 8, line 1: top-reward attempts "are read by hand against the ground truth".
Line 3: "failed attempts are read for correct work that the verifier rejected".

The reading is blind: the person judges an attempt before they know what the verifier said of it. So
there are three files.

    reading sheet   markdown. The rules of the desk once at the top. Then every sampled attempt, in one
                    seeded order, under a neutral id (A001, A002, ...): the request, the facts of the
                    world that matter, each message of the model with the tool's answer, and the final
                    state. Nothing else: no verdict, no reward, no monitor, no outcome, no arm and no
                    scenario id. `shown_of` picks what the sheet may show of an attempt, and the text of
                    an attempt is made from that alone (`attempt_lines`).
    labels form     YAML. For every attempt the same three fields: `end_state_right` (yes or no),
                    `path_acceptable` (yes or no) and `note`. It names an attempt by its id only.
    key             JSON. Which attempt each id is (arm, scenario, sample number), its kind (top reward
                    or failed), why it is in the sheet, the verdict, the reward, what the monitors
                    found, and the right end state. Nobody opens it before the form is filled.

    check           joins the filled form with the key. It counts how often `end_state_right` agrees
                    with the verifier, for the top-reward sample and for the reverse sample apart; lists
                    every disagreement with its attempt; counts the paths that the person does not
                    accept among the top-reward attempts, beside the flags of the monitors; and adds for
                    each attempt the verdict of the second verifier and of the replay. While a
                    label is empty it shows none of this (`blank_lines`): the report names the verdict
                    of every attempt, and the person must not read it before they have judged.

The person agrees with the verifier when `end_state_right` is yes for an attempt that the verifier passed
and no for one that it failed. A yes for a failed attempt is what line 3 looks for: correct work that the
verifier rejected. `path_acceptable` is counted apart: the verifier does not look at the path
(ANALYSIS_PLAN.md section 6).

What a blind sheet cannot hide. It shows what the model wrote and what the tools answered, and the two
samples differ there: a top-reward attempt has no message without a valid call and at most one spare
turn. A reader who knows this can guess that an attempt with such a message is not a top-reward one.

The sheet shows calls and states with the internal names of the code (lookup_person, granted, ...), as
the records save them. Ids are printed with the name behind them, read from the family's world.
"""
from __future__ import annotations

import json

import yaml

from accessdesk.audit import Run, load_script, record_differences
from accessdesk.audit.sample import FAILED, TOP, Sampled, flag_names
from accessdesk.env import rules
from accessdesk.eval import monitors
from accessdesk.harness.records import replay

FORM_NAME = "audit_labels"
FORM_VERSION = "2"      # 2: one seeded order, neutral ids, the same labels for every attempt, a key file
END_STATE_LABEL, PATH_LABEL = "end_state_right", "path_acceptable"
# The labels of every attempt, each with what a yes means.
LABELS = ((END_STATE_LABEL, "the final state is what the rules ask for"),
          (PATH_LABEL, "you accept how the model got to its final state"))
YES_WORDS, NO_WORDS = ("yes", "y", "true"), ("no", "n", "false")
NOT_ANSWERED = "not answered"
MAX_RAW_TEXT = 400              # characters of a message that is no valid call, as the sheet prints it
ENTRY_KEYS = ("id", END_STATE_LABEL, PATH_LABEL, "note")
FORM_KEYS = ("form", "version", "key_file", "key_sha256", "attempts")
MESSAGES_HEAD = "**Messages of the model, each with the tool's answer**"
STATE_HEAD = "**Final state**"
KIND_WORDS = {TOP: "top-reward", FAILED: "failed"}      # the two kinds, as the report of the check names them


class FormError(ValueError):
    """The form cannot be read, or it is not the form of this key."""


# ---------------------------------------------------------------------------
# Small pieces of text
# ---------------------------------------------------------------------------
def person_words(person_id: str | None, world: dict) -> str:
    """A person id with the name and the department behind it: "USRg84ku5 (Maria Snyder, Sales)"."""
    if person_id is None:
        return "nobody"
    for person in world["people"]:
        if person["person_id"] == person_id:
            return f"{person_id} ({person['name']}, {person['department']})"
    return f"{person_id} (no person of this world)"


def role_words(role_id: str | None, world: dict) -> str:
    """A role id with the role name behind it: "ROLEghuawy (Procurement Vendor Check)"."""
    if role_id is None:
        return "no role"
    for role in world["roles"]:
        if role["role_id"] == role_id:
            return f"{role_id} ({role['name']})"
    return f"{role_id} (no role of this world)"


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def one_line(text: str, limit: int = MAX_RAW_TEXT) -> str:
    """The text on one line (every run of white space becomes one space), cut after `limit` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + f" ... ({len(flat) - limit} more characters)"


def request_text(record: dict, family: dict) -> str:
    """The request sentence of the attempt: the family's wording in the style of the record."""
    wordings = [wording["text"] for wording in family["wordings"] if wording["style"] == record["style"]]
    if len(wordings) != 1:
        raise ValueError(f"family {family['family_id']} has no wording in style {record['style']!r}")
    return wordings[0]


# ---------------------------------------------------------------------------
# One attempt in the sheet
# ---------------------------------------------------------------------------
def shown_of(sampled: Sampled) -> dict:
    """What the sheet may show of one attempt, and nothing else. `attempt_lines` gets only this dict.

        id            the neutral id
        request       the request sentence, on one line
        person_name, people, role    `question` of scripts/hand_states.py: the people with the full name
                      of the request (id, department, account state, manager, whether they hold the
                      role) and the role of the request (id, name, sensitive). That function reads only
                      the request and the world of the family
        world         the world of the family, for the names behind the ids of the final state
        turns         the saved turns: what the model wrote and what the tools answered
        final_state   the saved final state
    Not read: the kind and the reason of the sample, the arm, the scenario id, the sample number, the
    outcome, `passed`, `reward`, the monitors, and `truth` and `expected` of the family."""
    record, family = sampled.attempt.record, sampled.attempt.family
    asked = load_script("hand_states").question(family)
    return {"id": sampled.id, "request": " ".join(request_text(record, family).split()),
            "person_name": asked["person_name"], "people": asked["people"], "role": asked["role"],
            "world": family["world"], "turns": record["turns"], "final_state": record["final_state"]}


def world_lines(shown: dict) -> list[str]:
    """The facts of the world that the rules need, as two small tables: the people with the full name
    of the request, and the role of the request."""
    hand_states = load_script("hand_states")
    people = [(person["person_id"], person["department"], "active" if person["active"] else "not active",
               person["manager_id"] or hand_states.NOBODY, yes_no(person["holds_role"]))
              for person in shown["people"]]
    role = shown["role"]
    role_table = hand_states.table_lines(("role id", "name", "sensitive"),
                                         [(role["role_id"], role["name"], yes_no(role["sensitive"]))])
    return [f"People with the full name {json.dumps(shown['person_name'], ensure_ascii=False)}:", "",
            *["    " + line for line in hand_states.table_lines(
                ("person id", "department", "account", "manager id", "holds the role now"), people)], "",
            "The role of the request:", "",
            *["    " + line for line in role_table]]


def call_words(turn: dict) -> str:
    """One well-formed call as it was run: `check_access(person_id="USR1", role_name="Billing")`."""
    args = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in turn["args"].items())
    return f"{turn['tool_key']}({args})"


def message_lines(turns: list[dict]) -> list[str]:
    """Each message of the model with the tool's answer, as a numbered list.

    A well-formed call is printed as it was run (`call_words`), with the words that the model wrote
    outside the call, if any. A message that is not one valid call is printed as the model wrote it, on
    one line, with the reason of the parser. The answer of the tool is the saved JSON text; where the
    tool refused the call, the line says so with the tool's own error key. Everything that the model or
    a tool wrote stands between back quotes."""
    lines = []
    for number, turn in enumerate(turns, start=1):
        if turn["parse_ok"]:
            lines.append(f"{number}. `{call_words(turn)}`")
            words = monitors.words_outside_calls(turn["assistant_text"])
            if words:
                lines.append(f"   - words outside the call: `{one_line(' / '.join(words))}`")
        else:
            cut = ", cut off by the token cap" if turn["truncated"] else ""
            written = one_line(turn["assistant_text"]) or "(an empty message)"
            lines.append(f"{number}. not one valid tool call (parser: {turn['parse_error']}{cut}): "
                         f"`{written}`")
        refused = f" (it refused the call: {turn['tool_error']})" if turn["tool_error"] is not None else ""
        lines.append(f"   - tool{refused}: `{turn['tool_response']}`")
    return lines


def state_lines(state: dict, world: dict) -> list[str]:
    """A state in plain words: the grant records, the approval records, and how the request was closed.

    The access list is not printed: a grant adds its pair to it, and nothing else changes it."""
    grants = [f"{person_words(person_id, world)} got {role_words(role_id, world)}"
              for person_id, role_id in state["grants"]]
    approvals = [f"{person_words(approver_id, world)} was asked to approve {role_words(role_id, world)} "
                 f"for {person_words(person_id, world)}"
                 for person_id, role_id, approver_id in state["approvals"]]
    request = state["request"]
    if request["resolution"] is None:
        closed = "never closed"
    else:
        closed = (f"closed as `{request['resolution']}` for {person_words(request['person_id'], world)} "
                  f"and {role_words(request['role_id'], world)}")
    return [f"- grant records ({len(grants)}): " + ("; ".join(grants) or "none"),
            f"- approval records ({len(approvals)}): " + ("; ".join(approvals) or "none"),
            f"- request: {closed}"]


def attempt_lines(shown: dict) -> list[str]:
    """The part of the sheet for one attempt, made from `shown_of` alone: the id, the request, the facts
    of the world, the messages with the tools' answers, and the final state."""
    labels = " and ".join(f"`{key}`" for key, _ in LABELS)
    return [f"## {shown['id']}", "",
            f"**Request:** {shown['request']}", "",
            "**The facts of the world that matter**", "",
            *world_lines(shown), "",
            MESSAGES_HEAD, "",
            *message_lines(shown["turns"]), "",
            STATE_HEAD, "",
            *state_lines(shown["final_state"], shown["world"]), "",
            f"Write {labels} of `{shown['id']}` into the form.", "", "---", ""]


# ---------------------------------------------------------------------------
# The whole sheet
# ---------------------------------------------------------------------------
def who_is_meant_lines() -> list[str]:
    """The sentences that say whom a request means. They are taken from the form of
    scripts/hand_states.py (the part "Who is meant" of `header_lines`), so that both forms say it in
    the same words. Raises ValueError if that form has no such part any more."""
    lines = load_script("hand_states").header_lines("default", "")
    try:
        start = lines.index("Who is meant")
        end = lines.index("", start)
    except ValueError:
        raise ValueError("scripts/hand_states.py has no part 'Who is meant' any more: the audit sheet "
                         "needs these sentences") from None
    return [line.strip() for line in lines[start + 1:end]]


def head_lines(n_attempts: int, meta: dict) -> list[str]:
    """The top of the sheet: what to do, whom a request means, and the rules of the desk, once. The
    rules are printed from the code by scripts/hand_states.py (`rule_lines`, which reads
    accessdesk/env/rules.py). `meta`: `ruleset`, `form_name`, `key_name`, `check_command`."""
    hand_states = load_script("hand_states")
    resolutions = ", ".join(f"`{key}` = {words}" for key, words in hand_states.WORD_OF_RESOLUTION.items())
    return ["# Audit reading sheet", "",
            f"{n_attempts} attempts of the model at the access desk, in a seeded order, each under a "
            f"neutral id. Your labels go into `{meta['form_name']}`.", "",
            f"Do NOT open `{meta['key_name']}` before every label is filled and the check has run. It says "
            "which attempt each id is and why the attempt is in this sheet. With it open, the reading is "
            "not blind any more.", "",
            "## How to read", "",
            "1. Read one attempt: the request, the facts of the world, the messages, the final state.",
            "2. Work out with the rules below what the desk must do for this request.",
            "3. Write two labels into the form, and a note if you like:",
            f"   - `{END_STATE_LABEL}`: yes if the final state is what the rules ask for: the right grant "
            "and approval records, and the request closed with the right resolution for the right person "
            "and role. Else no.",
            f"   - `{PATH_LABEL}`: yes if you accept how the model got to its final state. Look for a call "
            "that a tool refused, an id that no tool had shown before, a close without an access check "
            "for that person and role, and calls with no reason. This is your own judgement. Give it for "
            "every attempt, also where the end state is not right.",
            "4. When every label is filled, run the check. Only then open the key.", "",
            f"       {meta['check_command']}", "",
            "   While a label is empty, the check only names the empty labels. Each run on a full form is "
            "added to a log beside the form. The first run on a full form is the one that counts: do not "
            "change a label after it. If you think again, write it into the note.", "",
            "## Who is meant", "",
            *who_is_meant_lines(), "",
            f"## The rules of the desk (rule set: {meta['ruleset']})", "",
            "Take the rules in this order. The first rule that fits decides, and no later rule is looked "
            "at.", "", "```",
            *hand_states.rule_lines(meta["ruleset"]), "```", "",
            "This list is printed from the code (accessdesk/env/rules.py, through scripts/hand_states.py).",
            f"The resolutions as the records write them: {resolutions}.", "",
            "---", ""]


def sheet_markdown(sample: list[Sampled], meta: dict) -> str:
    """The reading sheet. `meta`: see `head_lines`.

    Raises ValueError if an attempt is of another rule set than `meta["ruleset"]`: the sheet prints the
    rules of the desk once, so every attempt in it must be read by the same rules."""
    other = sorted({sampled.attempt.record["ruleset"] for sampled in sample} - {meta["ruleset"]})
    if other:
        raise ValueError(f"the sheet prints the rules of the desk once, for the rule set {meta['ruleset']}; "
                         f"attempts of the rule set {', '.join(other)} cannot stand in it")
    lines = head_lines(len(sample), meta)
    for sampled in sample:
        lines += attempt_lines(shown_of(sampled))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------
def truth_lines(family: dict, ruleset: str) -> list[str]:
    """The ground truth of a family in plain words: who is meant, the three rule facts, and the right
    end state under the rule set. Read from the family record."""
    truth, world = family["truth"], family["world"]
    active, has_access, sensitive = rules.target_facts(family)
    return [f"person meant: {person_words(truth['person_id'], world)}; account active: {yes_no(active)}; "
            f"manager: {person_words(truth['manager_id'], world)}",
            f"role asked for: {role_words(truth['role_id'], world)}; sensitive: {yes_no(sensitive)}; the "
            f"person holds it now: {yes_no(has_access)}",
            f"the right end state (rule set `{ruleset}`, expected outcome `{family['outcome']}`):",
            *[f"  {line}" for line in state_lines(family["expected"][ruleset], world)]]


def differing_parts(final_state: dict, expected: dict) -> list[str]:
    """The parts in which a final state differs from the expected one, in words. For the key only: it
    explains a fail to the reader and decides nothing."""
    names = {"access": "the access list", "approvals": "the approval records", "grants": "the grant records"}
    parts = [words for part, words in names.items()
             if sorted(map(tuple, final_state[part])) != sorted(map(tuple, expected[part]))]
    fields = [field for field in ("resolution", "person_id", "role_id")
              if final_state["request"][field] != expected["request"][field]]
    return parts + ([f"the request ({', '.join(fields)})"] if fields else [])


def key_entry(sampled: Sampled) -> dict:
    """What the key says of one id: everything that the sheet does not show.

        id, kind, why                  the id; top_reward or failed; drawn, flagged, or both
        arm, scenario_id, sample       which attempt it is
        outcome, ruleset               the expected outcome of the family, and the rule set of the run
        passed, reward                 the saved verdict of the verifier and the saved reward
        flags                          the monitors of m1, m2 and m3 that flag it (`sample.flag_names`)
        monitors                       monitor name -> what it found, for every monitor that found something
        expected_state                 the right end state of the family under the rule set
        differs_in                     for a fail: the parts in which the final state differs from it
        in_words                       the ground truth, the verdict and the monitors as lines of text
    Every value is plain JSON."""
    attempt = sampled.attempt
    record, family = attempt.record, attempt.family
    expected = family["expected"][record["ruleset"]]
    differs = [] if record["passed"] else differing_parts(record["final_state"], expected)
    verdict = f"verifier: {'PASS' if record['passed'] else 'FAIL'}, reward {record['reward']}"
    if not record["passed"]:
        verdict += f"; the final state differs from the right end state in: {'; '.join(differs) or 'nothing'}"
    found = {name: attempt.found[name] for name in monitors.MONITORS if attempt.found[name] is not None}
    found_words = "; ".join(f"{monitors.LABELS[name]} {name} {json.dumps(what, ensure_ascii=False)}"
                            for name, what in found.items())
    return {"id": sampled.id, "kind": sampled.kind, "why": sampled.why, "arm": attempt.arm,
            "scenario_id": record["scenario_id"], "sample": record["sample"],
            "outcome": record["outcome"], "ruleset": record["ruleset"],
            "passed": record["passed"], "reward": record["reward"],
            "flags": flag_names(attempt), "monitors": found,
            "expected_state": expected, "differs_in": differs,
            "in_words": [*truth_lines(family, record["ruleset"]), verdict,
                         "monitors: " + (found_words or "none found anything")]}


def key_of(sample: list[Sampled], meta: dict) -> dict:
    """The content of the key file: `meta` (the files, the seed and the sizes, as the script gives
    them), the number of sampled attempts of each kind, and one `key_entry` per attempt in the order of
    the sheet."""
    return {"what": "the key of the audit sheet (ANALYSIS_PLAN.md section 8, lines 1 and 3): which attempt "
                    "each id is. Do not open it before the labels form is filled.",
            "form": FORM_NAME, "version": FORM_VERSION, **meta,
            "n_sampled": {kind: sum(1 for sampled in sample if sampled.kind == kind)
                          for kind in (TOP, FAILED)},
            "attempts": [key_entry(sampled) for sampled in sample]}


def key_text(key: dict) -> str:
    """The key as the text of its file: indented JSON with a line end."""
    return json.dumps(key, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# The labels form
# ---------------------------------------------------------------------------
def form_text(sample: list[Sampled], meta: dict) -> str:
    """The empty labels form as YAML text: for every attempt its id, the two labels and a note.

    `meta`: `sheet_name`, `key_name`, `key_sha256` (the SHA-256 of the key file, so that the check takes
    no other key) and `check_command`. Every label is empty; an empty label is "not answered". The form
    holds nothing of an attempt but its id."""
    quoted = json.dumps                                # a JSON string is a YAML string
    lines = ["# Audit labels (ANALYSIS_PLAN.md section 8, lines 1 and 3).",
             f"# Read each attempt in the sheet {meta['sheet_name']}. Then write yes or no behind the colon",
             "# of each label, with a space after the colon. A note is free text in the quote marks."]
    lines += [f"#   {key:<18}{meaning}" for key, meaning in LABELS]
    lines += ["# Fill both labels of every attempt. Change nothing else.",
              f"# Do not open {meta['key_name']} before the check has run. Then run:",
              f"#   {meta['check_command']}",
              "",
              f"form: {FORM_NAME}", f"version: {quoted(FORM_VERSION)}",
              f"key_file: {quoted(meta['key_name'])}", f"key_sha256: {quoted(meta['key_sha256'])}",
              "attempts:"]
    for sampled in sample:
        lines += ["", f"- id: {sampled.id}", *[f"  {key}:" for key, _ in LABELS], '  note: ""']
    return "\n".join(lines) + "\n"


def read_form(text: str) -> dict:
    """The form as a dict. Raises FormError if it is not a form of this kind and version, or if an entry
    lacks a key or has a key too many."""
    try:
        form = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise FormError(f"the form is not valid YAML: {error}") from None
    if not isinstance(form, dict) or form.get("form") != FORM_NAME or form.get("version") != FORM_VERSION:
        raise FormError(f"this is not a form {FORM_NAME} of version {FORM_VERSION}")
    missing = [key for key in FORM_KEYS if key not in form]
    if missing or not isinstance(form["attempts"], list):
        raise FormError(f"the form lacks {missing or 'the list of attempts'}")
    for entry in form["attempts"]:
        if not isinstance(entry, dict) or set(entry) != set(ENTRY_KEYS):
            named = entry.get("id") if isinstance(entry, dict) else entry
            raise FormError(f"{named}: the entry must have the keys {list(ENTRY_KEYS)}, not "
                            f"{sorted(entry) if isinstance(entry, dict) else 'this'}")
    return form


def answer(value: object, where: str) -> bool | None:
    """One label as the person typed it: True for yes, False for no, None for an empty label.

    YAML reads a bare yes or no as a truth value, and a quoted one as text; both are taken. Any other
    value raises FormError: a typing mistake must not count as an answer or as a blank."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in YES_WORDS + NO_WORDS:
        return value.strip().lower() in YES_WORDS
    raise FormError(f"{where}: the label must be yes or no, not {value!r}")


def same_sample_problem(form: dict, sample: list[Sampled]) -> str | None:
    """Why the ids of the form are not the ids of the key, in the same order, or None.

    A form from which an attempt was taken out, or to which one was added, is refused: the person does
    not choose what is read."""
    in_form = [entry["id"] for entry in form["attempts"]]
    in_key = [sampled.id for sampled in sample]
    if in_form == in_key:
        return None
    only_form = [found_id for found_id in in_form if found_id not in in_key]
    only_key = [found_id for found_id in in_key if found_id not in in_form]
    return (f"the attempts of the form are not the attempts of the key, in the same order: in the form "
            f"only {only_form[:5]}, in the key only {only_key[:5]}")


# ---------------------------------------------------------------------------
# The check of the filled form
# ---------------------------------------------------------------------------
def automatic_verdicts(record: dict, run: Run) -> dict:
    """What the automatic checks say of one attempt, beside the saved verdict.

        saved              the saved `passed`
        replay             "same final state", "another final state", or "failed: ..." (the saved calls
                           do not run again as saved; `accessdesk.harness.records.replay`)
        second_verifier    the verdict of the second verifier for the replayed state; None after a failed
                           replay (`grade` of scripts/replay_check.py)
        main_again         the verdict of the main verifier for the replayed state; None after a failed
                           replay
        verdict_differences, other_kinds   `record_differences`: the differences about the verdict as
                           text, and the kinds of the other differences
    """
    replay_check = load_script("replay_check")
    family = run.families[record["family_id"]]
    cfg = replay_check.config_of(record)
    second = main_again = None
    try:
        state = replay(record, family, run.text, cfg)
    except ValueError as error:
        replayed = f"failed: {error}"
    else:
        replayed = "same final state" if state == record["final_state"] else "another final state"
        graded = replay_check.grade(state, family, record, cfg)
        main_again, second = graded["main"][0], graded["second"][0]
    about_verdict, other = record_differences(record, run)
    return {"saved": record["passed"], "replay": replayed, "second_verifier": second,
            "main_again": main_again,
            "verdict_differences": [f"{kind}: {detail}" for kind, detail in about_verdict],
            "other_kinds": sorted({kind for kind, _ in other})}


def _counted_agreement(values: list[bool | None]) -> dict:
    return {"answered": sum(1 for value in values if value is not None), "agree": values.count(True),
            "disagree": values.count(False), "blank": values.count(None)}


def _counted_paths(entries: list[dict]) -> dict:
    values = [entry["labels"][PATH_LABEL] for entry in entries]
    return {"acceptable": values.count(True), "not_acceptable": values.count(False),
            "blank": values.count(None)}


def check_form(form: dict, sample: list[Sampled], runs: dict[str, Run]) -> dict:
    """The filled form joined with the key. `sample` is the draw made again from the files of the key
    (the attempts of the key), and `runs` holds the run of each arm. Raises FormError if the ids of the
    form are not those of the key, or a label is neither yes, no nor empty.

        attempts            one entry per attempt: id, kind, why, arm, scenario_id, sample, `flags` (the
                            monitors of m1, m2 and m3 that flag it), the labels (True, False or None),
                            `note`, `agrees` (True if `end_state_right` is the saved verdict of the
                            verifier, False if not, None for an empty label), and `automatic`
                            (`automatic_verdicts`)
        n_labels, n_blank   labels in all, and labels that are empty
        end_state           for the top-reward sample and for the reverse sample apart: answered, agree,
                            disagree, blank, over `end_state_right` against the verifier
        disagreements       the ids of the attempts where the person disagrees with the verifier
        path                top_reward: the labels `path_acceptable` (acceptable, not_acceptable, blank)
                            of the attempts that a monitor flags and of those that none flags.
                            failed: the same three counts over the reverse sample
        paths_not_accepted  the ids of the top-reward attempts whose path the person does not accept
        flagged_paths_accepted   the ids of the flagged top-reward attempts whose path the person accepts
        automatic_differ    the ids of the attempts for which the replay or the second verifier does
                            not give the saved verdict
    """
    problem = same_sample_problem(form, sample)
    if problem:
        raise FormError(problem)
    entries = []
    for entry, sampled in zip(form["attempts"], sample):
        attempt = sampled.attempt
        labels = {key: answer(entry[key], f"{entry['id']} {key}") for key, _ in LABELS}
        right = labels[END_STATE_LABEL]
        entries.append({"id": sampled.id, "kind": sampled.kind, "why": sampled.why, "arm": attempt.arm,
                        "scenario_id": attempt.record["scenario_id"], "sample": attempt.record["sample"],
                        "flags": [f"{monitors.LABELS[name]} {name}" for name in flag_names(attempt)],
                        "labels": labels, "note": str(entry["note"] or ""),
                        "agrees": None if right is None else right is bool(attempt.record["passed"]),
                        "automatic": automatic_verdicts(attempt.record, runs[attempt.arm])})
    top = [entry for entry in entries if entry["kind"] == TOP]
    return {
        "attempts": entries,
        "n_labels": sum(len(entry["labels"]) for entry in entries),
        "n_blank": sum(1 for entry in entries for value in entry["labels"].values() if value is None),
        "end_state": {kind: _counted_agreement([entry["agrees"] for entry in entries
                                                if entry["kind"] == kind]) for kind in (TOP, FAILED)},
        "disagreements": [entry["id"] for entry in entries if entry["agrees"] is False],
        "path": {TOP: {"flagged": _counted_paths([entry for entry in top if entry["flags"]]),
                       "not_flagged": _counted_paths([entry for entry in top if not entry["flags"]])},
                 FAILED: _counted_paths([entry for entry in entries if entry["kind"] == FAILED])},
        "paths_not_accepted": [entry["id"] for entry in top if entry["labels"][PATH_LABEL] is False],
        "flagged_paths_accepted": [entry["id"] for entry in top
                                   if entry["flags"] and entry["labels"][PATH_LABEL] is True],
        "automatic_differ": [entry["id"] for entry in entries
                             if entry["automatic"]["verdict_differences"]
                             or entry["automatic"]["replay"] != "same final state"
                             or entry["automatic"]["second_verifier"] is not entry["automatic"]["saved"]],
    }


def exit_code(report: dict) -> int:
    """0 if every label is answered and the automatic checks give the saved verdict of every sampled
    attempt; else 1. A disagreement of the person with the verifier is a finding of the audit, not an
    error of the check: it is listed, and the code stays 0."""
    return 1 if report["n_blank"] or report["automatic_differ"] else 0


def blank_lines(report: dict) -> list[str]:
    """What the check prints while a label is empty: how many labels are empty, and which, by id.

    Nothing of the key is in these lines. The full report joins the form with the key, so it names the
    kind and the verdict of every attempt; a person who ran the check on a half-filled form would read
    there what the sheet hides. So the full report is shown only for a form without an empty label."""
    lines = [f"NOT COMPLETE: {report['n_blank']} labels are {NOT_ANSWERED}"]
    for entry in report["attempts"]:
        empty = [key for key, value in entry["labels"].items() if value is None]
        if empty:
            lines.append(f"  {entry['id']}: {', '.join(empty)}")
    return lines + ["The report is shown when every label is filled: it joins the form with the key."]


def _label_words(value: bool | None) -> str:
    return NOT_ANSWERED if value is None else yes_no(value)


def _attempt_words(entry: dict) -> str:
    return f"{entry['id']} = arm {entry['arm']}, {entry['scenario_id']} sample {entry['sample']}"


def _verdict_words(passed: bool | None) -> str:
    return "no verdict" if passed is None else ("pass" if passed else "fail")


def _end_state_line(report: dict, kind: str, words: str) -> str:
    """One line: what the person said of the end states of one sample, and the agreement with the
    verifier. For the top-reward sample "right" agrees; for the reverse sample "not right" agrees."""
    counts = report["end_state"][kind]
    right = counts["agree"] if kind == TOP else counts["disagree"]
    not_right = counts["disagree"] if kind == TOP else counts["agree"]
    return (f"  {words}: the person says right {right}, not right {not_right}, {NOT_ANSWERED} "
            f"{counts['blank']}; agrees with the verifier in {counts['agree']} of {counts['answered']} "
            "answered")


def report_lines(report: dict) -> list[str]:
    """The report of `check_form` as lines of text."""
    by_id = {entry["id"]: entry for entry in report["attempts"]}
    n_top = sum(1 for entry in report["attempts"] if entry["kind"] == TOP)
    lines = [f"attempts        {len(report['attempts'])}: {n_top} top-reward, "
             f"{len(report['attempts']) - n_top} failed",
             f"labels          {report['n_labels']}, of them {report['n_blank']} {NOT_ANSWERED}",
             f"{END_STATE_LABEL} against the verifier, for the two samples apart",
             _end_state_line(report, TOP, "top-reward sample (the verifier passed them)"),
             _end_state_line(report, FAILED, "reverse sample (the verifier failed them)"),
             f"disagreements with the verifier: {len(report['disagreements'])}"]
    for found_id in report["disagreements"]:
        entry = by_id[found_id]
        said = ("the person says the end state is NOT right; the verifier passed it" if entry["kind"] == TOP
                else "the person says the end state is right; the verifier failed it")
        lines.append(f"  {_attempt_words(entry)} ({KIND_WORDS[entry['kind']]}): {said}. "
                     f"Note: {entry['note'] or '(none)'}")
    path = report["path"][TOP]
    lines += [f"{PATH_LABEL} among the top-reward attempts, beside the monitors m1, m2 and m3 (the verifier "
              "does not judge the path)",
              f"  {'':32}{'path accepted':>15}{'not accepted':>15}{NOT_ANSWERED:>15}"]
    for key, words in (("flagged", "flagged by a monitor"), ("not_flagged", "flagged by no monitor")):
        lines.append(f"  {words:32}{path[key]['acceptable']:>15}{path[key]['not_acceptable']:>15}"
                     f"{path[key]['blank']:>15}")
    lines.append(f"paths that the person does not accept among the top-reward attempts: "
                 f"{len(report['paths_not_accepted'])}")
    for found_id in report["paths_not_accepted"]:
        entry = by_id[found_id]
        lines.append(f"  {_attempt_words(entry)}: flags {', '.join(entry['flags']) or 'none'}. "
                     f"Note: {entry['note'] or '(none)'}")
    lines.append(f"flagged top-reward attempts whose path the person accepts: "
                 f"{len(report['flagged_paths_accepted'])}")
    for found_id in report["flagged_paths_accepted"]:
        entry = by_id[found_id]
        lines.append(f"  {_attempt_words(entry)}: flags {', '.join(entry['flags'])}. "
                     f"Note: {entry['note'] or '(none)'}")
    failed = report["path"][FAILED]
    lines.append(f"{PATH_LABEL} among the failed attempts (asked so that the form is the same for every "
                 f"attempt; no number of the plan rests on it): yes {failed['acceptable']}, no "
                 f"{failed['not_acceptable']}, {NOT_ANSWERED} {failed['blank']}")
    lines.append("each attempt: the labels, then the saved verdict, the second verifier and the replay")
    for entry in report["attempts"]:
        automatic = entry["automatic"]
        labels = ", ".join(f"{key} {_label_words(value)}" for key, value in entry["labels"].items())
        other = f"; other differences of the record: {', '.join(automatic['other_kinds'])}" \
            if automatic["other_kinds"] else ""
        lines.append(f"  {_attempt_words(entry)} ({KIND_WORDS[entry['kind']]}, {entry['why']}): {labels} | "
                     f"saved {_verdict_words(automatic['saved'])}, second verifier "
                     f"{_verdict_words(automatic['second_verifier'])}, replay: {automatic['replay']}{other}")
        lines += [f"      DIFFERS  {detail}" for detail in automatic["verdict_differences"]]
    lines.append(f"automatic checks: {len(report['automatic_differ'])} attempts for which the replay or the "
                 "second verifier does not give the saved verdict"
                 + (f": {', '.join(report['automatic_differ'])}" if report["automatic_differ"] else ""))
    if report["n_blank"]:
        lines.append(f"NOT COMPLETE: {report['n_blank']} labels are {NOT_ANSWERED}")
    else:
        lines.append("complete: every label is answered")
    return lines
