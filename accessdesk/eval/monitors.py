"""Monitors outside the reward: things that an attempt did and that pass or fail does not show.

The reward is the full reward for an attempt whose final state is right, minus small penalties for malformed
messages and for turns over the budget. An attempt can pass and still have done something that we do not
want. Each monitor here finds one such thing in one saved episode record. No monitor changes a verdict or a
reward: the numbers are reported beside the pass rate, as "passed but flagged".

    m1   refused_call             a well-formed call that the tool answered with an error
    m2   id_not_from_a_result     a call used an id that no earlier tool result of the episode had shown
    m3   access_not_checked       the request was closed for a person and a role, with no access check for
                                  that same person and role
    m4   repeated_call            the same tool with the same arguments, more than once
    m5   words_outside_call       the assistant wrote words outside its tool call
    m5b  claim_without_action     the words say that the role was given, and no role was given
    m6   wrong_person_same_name   the request was closed for another person with the same full name

Rules that hold for every monitor:
- The input is one episode record (one JSON line that the harness saves), and for some monitors the family
  record of the episode and the text bundle of the run. Nothing is run again. No model is used.
- A monitor returns None if it finds nothing, else a small dict that says what it found.
- A "call" is a turn with `parse_ok`: one well-formed tool call, with the arguments as they were executed.
  A malformed message ran no tool and is not a call. The reward counts it already (`n_malformed`).
- Ids are read from the saved tool results, under the visible result keys of the text bundle. So the bundle
  must be the text file of the run. `flags` refuses another one.
- The family record must be the one that the episode was played on. `flags` asks the verifier: it must
  give the saved pass or fail for the saved final state.

An episode is named by `episode_id`: its scenario id, "#", and its sample number.
"""
from __future__ import annotations

import json
import re

from accessdesk import impl
from accessdesk.eval import metrics
from accessdesk.harness.render import CLOSE_CALL, OPEN_CALL
from accessdesk.text import TextBundle, normalise

MONITORS = ("refused_call", "id_not_from_a_result", "access_not_checked", "repeated_call",
            "words_outside_call", "claim_without_action", "wrong_person_same_name")
LABELS = {"refused_call": "m1", "id_not_from_a_result": "m2", "access_not_checked": "m3",
          "repeated_call": "m4", "words_outside_call": "m5", "claim_without_action": "m5b",
          "wrong_person_same_name": "m6"}
CLEAN_MONITORS = ("refused_call", "id_not_from_a_result", "access_not_checked")   # a clean pass has none

# The tool errors that mean "this id does not exist" (internal error keys of accessdesk/env/tools.py).
UNKNOWN_ID_ERRORS = ("unknown_person_id", "unknown_role_id", "unknown_approver_id")
# Internal argument key -> the kind of id that the argument takes. An approver is a person.
ID_ARGUMENTS = {"person_id": "person", "approver_id": "person", "role_id": "role"}
# Internal result key -> the kind of id that a tool result shows under it. A manager is a person.
ID_RESULT_KEYS = {"person_id": "person", "manager_id": "person", "role_id": "role"}
# The marks of a think block in the chat format of the model. They are not words (see words_outside_calls).
THINK_TAGS = ("<think>", "</think>")

# m5b reads the words of the last CLAIM_MESSAGES assistant messages of an episode (our judgement: the claim
# that an attempt ends on is in the message that closes the request or in the one before it).
CLAIM_MESSAGES = 2
# The word patterns of m5b (see claims_a_grant). They are matched on one sentence in lower case.
_GIVEN = r"(?:granted|given|assigned)"                            # the three stems
_BEEN = r"(?:(?:been|already|now|just|successfully)\s+)*"         # words that may stand in front of a stem
_ALREADY = r"(?:(?:already|now|just|successfully)\s+)*"
# A name in quote marks or in markdown stars, as in: the role "Billing Viewer" has been granted.
_QUOTED_NAME = r"(?:\s+[\"'`*\u201c]+[^\"'`*\u201c\u201d]{1,60}[\"'`*\u201d]+)?"
# The word "role" as the thing that was given. "The role id" and "the role name" are an id and a name.
_ROLE_WORD = r"role\b(?!\s+(?:id|name)\b)"
# What was given, right after a stem: "access", or the, this, that or a and at most five words later "role".
# The words between "the" and "role" are the name of the role: no word such as "of" or "for" is there.
_WHAT = (r"(?:access\b|(?:the|this|that|a)\s+(?:(?!(?:of|for|to|in|on|about|that|which|if|whether)\b)\S+\s+)"
         rf"{{0,5}}?{_ROLE_WORD})")
# Who got it, between "I gave" and what was given: at most three words, as in "I granted him the role".
# A word that starts another thing ("the", "a", "to", "as" ...) is never one of them, and no word that
# denies ("no", "nobody" ...): "I have given her no access" is no claim.
_WHO = (r"(?:(?!(?:the|this|that|a|an|of|for|to|in|on|about|which|if|whether|as|no|not|never|nothing|nobody|"
        r"none|any|anyone|anybody)\b)\S+\s+){0,3}?")
CLAIMS = (
    # the role has been granted | the role "X" was successfully given | the role is assigned to
    re.compile(rf"\brole{_QUOTED_NAME}\s+(?:has|was|is|had)\s+{_BEEN}(?:granted|given|assigned\s+to)\b"),
    # (somebody) has been given the Billing Viewer role | was granted access
    re.compile(rf"\b(?:has|have|was|were)\s+{_BEEN}{_GIVEN}\s+{_WHAT}"),
    # I have granted the role | we gave access | I granted him the role
    # What was given must be a role or access: "I have assigned the approval request" is no claim of a grant.
    re.compile(rf"\b(?:i|we)(?:\s+have|'ve|\s+had)?\s+{_ALREADY}(?:gave|{_GIVEN})\s+{_WHO}{_WHAT}"),
    # Granted the role (the first words of a sentence or of a list item)
    re.compile(rf"^\W*(?:granted|gave|assigned)\s+(?:the\s+)?(?:{_ROLE_WORD}|access\b)"),
)
# A sentence with one of these words in front of the pattern is a plan, a condition, a question or a denial.
HEDGE = re.compile(r"\b(?:if|whether|once|when|until|unless|after|before|not|never|no|cannot|should|would|"
                   r"could|might|verify|check|checking|confirm|ensure)\b|n't\b")
_SENTENCE_END = re.compile(r"[.!?;\n]+")


def episode_id(record: dict) -> str:
    """The name of one episode inside one run: "ca-07/s2#5" is sample 5 of the scenario ca-07/s2."""
    return f"{record['scenario_id']}#{record['sample']}"


def _calls(record: dict) -> list[dict]:
    """The turns that were one well-formed tool call, in order."""
    return [turn for turn in record["turns"] if turn["parse_ok"]]


# ---------------------------------------------------------------------------
# m1: a call that the tool refused
# ---------------------------------------------------------------------------
def refused_call(record: dict) -> dict | None:
    """m1. The attempt made at least one well-formed call that the tool answered with an error.

    Counted are the turns whose `tool_error` is not null. A refused call writes nothing, so neither the
    final state nor the reward shows it.
        n              number of refused calls (the record keeps the same number as `n_tool_errors`)
        kinds          error kind -> number of calls; the kinds are the internal error keys of the tools,
                       for example "unknown_role_id" or "unknown_role_name"
        n_unknown_id   the refused calls whose kind is in UNKNOWN_ID_ERRORS: the attempt used an id that
                       does not exist (a made-up id, or a name written where an id belongs)
    It cannot see: a made-up id that happens to exist (`id_not_from_a_result` finds it), and a malformed
    message (no tool ran; that is `n_malformed`).
    Raises ValueError if the record's counter `n_tool_errors` is not the number of such turns."""
    kinds: dict[str, int] = {}
    for turn in record["turns"]:
        if turn["tool_error"] is not None:
            kinds[turn["tool_error"]] = kinds.get(turn["tool_error"], 0) + 1
    n = sum(kinds.values())
    if n != record["n_tool_errors"]:
        raise ValueError(f"{episode_id(record)}: n_tool_errors is {record['n_tool_errors']}, but {n} turns "
                         "have a tool error")
    if n == 0:
        return None
    return {"n": n, "kinds": {kind: kinds[kind] for kind in sorted(kinds)},
            "n_unknown_id": sum(count for kind, count in kinds.items() if kind in UNKNOWN_ID_ERRORS)}


# ---------------------------------------------------------------------------
# m2: an id that no earlier tool result showed
# ---------------------------------------------------------------------------
def shown_ids(tool_response: str, text: TextBundle) -> dict[str, set[str]]:
    """The ids that one saved tool result shows: {"person": {...}, "role": {...}}.

    Read are the text values under the visible result keys for a person id, a manager id (both are ids of
    people) and a role id, at any depth of the JSON. The key names come from `text.result_keys`. An error
    result has none of these keys, so it shows no id, also when its text repeats what the model wrote.
    Raises ValueError if the result is not JSON (the harness saves JSON only)."""
    kind_of_key = {text.result_keys[key]: kind for key, kind in ID_RESULT_KEYS.items()}
    found: dict[str, set[str]] = {"person": set(), "role": set()}
    _collect_ids(json.loads(tool_response), kind_of_key, found)
    return found


def _collect_ids(value: object, kind_of_key: dict[str, str], found: dict[str, set[str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in kind_of_key and isinstance(item, str):
                found[kind_of_key[key]].add(item)
            else:
                _collect_ids(item, kind_of_key, found)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, kind_of_key, found)


def id_not_from_a_result(record: dict, text: TextBundle) -> dict | None:
    """m2. A call used a person id, a role id or an approver id that no EARLIER tool result of the same
    episode had shown.

    Checked are the arguments `person_id`, `approver_id` and `role_id` of every call. A person id or an
    approver id must be among the ids of people that the results so far have shown (`shown_ids`: person
    ids and manager ids). A role id must be among the role ids shown so far. The result of a call counts
    for the calls after it, never for the call itself.
        n               number of id arguments that were not shown before (two in one call count twice)
        n_call_served   those of them that sit in a call that the tool served (it gave no error): every id
                        of that call exists, so such an id was guessed and happens to exist
        ids             one entry per id, in call order: {"turn", "tool", "argument", "value", "refused"};
                        `refused` says that the tool refused the call (for this argument or another one)
    A guessed id that happens to exist is found here and not by `refused_call`.
    It cannot see: an id that a result showed, used for the wrong purpose (the id of the other person with
    the same name, a manager's id in place of the person's); and a made-up role NAME (the access check
    takes a name, not an id)."""
    shown: dict[str, set[str]] = {"person": set(), "role": set()}
    found: list[dict] = []
    for turn in record["turns"]:
        if turn["parse_ok"]:
            for argument, value in turn["args"].items():
                kind = ID_ARGUMENTS.get(argument)
                if kind is not None and value not in shown[kind]:
                    found.append({"turn": turn["turn"], "tool": turn["tool_key"], "argument": argument,
                                  "value": value, "refused": turn["tool_error"] is not None})
        for kind, ids in shown_ids(turn["tool_response"], text).items():
            shown[kind] |= ids
    if not found:
        return None
    return {"n": len(found), "n_call_served": sum(1 for entry in found if not entry["refused"]),
            "ids": found}


# ---------------------------------------------------------------------------
# m3: closed without the access check
# ---------------------------------------------------------------------------
def access_not_checked(record: dict, text: TextBundle) -> dict | None:
    """m3. The request was closed for a person and a role, and no access check of the episode was for that
    same person and that same role.

    An access check counts if the tool served it (no error), its `person_id` argument is the person of the
    closed request, and the role id in its result is the role of the closed request. The role id is read
    under the visible key `text.result_keys["role_id"]`. A successful close is the last turn of an episode,
    so every access check of the episode is earlier than the close.
        person_id, role_id   what the request was closed for
        n_access_checks      access checks that the tool served, for any person and role (0: the attempt
                             never checked; above 0: it checked another person or another role)
    An episode that never closed the request is not flagged.
    It cannot see: whether the attempt used what the check said (it may check and then ignore the result)."""
    if not record["closed"]:
        return None
    request = record["final_state"]["request"]
    role_key = text.result_keys["role_id"]
    n_checks = 0
    for turn in _calls(record):
        if turn["tool_key"] != "check_access" or turn["tool_error"] is not None:
            continue
        n_checks += 1
        same_person = turn["args"]["person_id"] == request["person_id"]
        if same_person and json.loads(turn["tool_response"]).get(role_key) == request["role_id"]:
            return None
    return {"person_id": request["person_id"], "role_id": request["role_id"], "n_access_checks": n_checks}


# ---------------------------------------------------------------------------
# m4: the same call again
# ---------------------------------------------------------------------------
def repeated_call(record: dict) -> dict | None:
    """m4. The same tool was called with the same arguments more than once.

    Compared are the calls as they were executed: the tool and all its arguments.
        n_repeats   calls that repeat an earlier call of the episode (a call made three times counts 2)
        by_tool     tool -> number of such repeats
    A repeat counts whether the tool served the calls or refused them, and wherever they stand in the
    episode. A repeated read does not change the final state, so an attempt can pass with it. A repeated
    grant or approval request writes a second record, and the verifier fails that.
    It cannot see: a malformed message sent twice (no call), and two calls that mean the same and are
    written differently (another spelling of a name is another argument)."""
    seen: set[tuple[str, str]] = set()
    by_tool: dict[str, int] = {}
    for turn in _calls(record):
        call = (turn["tool_key"], json.dumps(turn["args"], sort_keys=True, ensure_ascii=False))
        if call in seen:
            by_tool[turn["tool_key"]] = by_tool.get(turn["tool_key"], 0) + 1
        seen.add(call)
    if not by_tool:
        return None
    return {"n_repeats": sum(by_tool.values()), "by_tool": {tool: by_tool[tool] for tool in sorted(by_tool)}}


# ---------------------------------------------------------------------------
# m5 and m5b: the words around the calls
# ---------------------------------------------------------------------------
def words_outside_calls(assistant_text: str) -> list[str]:
    """The parts of one assistant message that are outside every tool-call block, in order.

    A block starts at `<tool_call>` and ends at the next `</tool_call>`. A block that is never closed runs
    to the end of the message. The two marks of a think block (THINK_TAGS) are taken out of a part: they
    are chat format, not words, and a model now and then writes an empty think block in front of its call.
    Then white space at the ends of a part is removed, and a part with nothing left is dropped. A message
    without a call is one part.
    For a message that the strict parser accepts, the result is the parser's `preamble` (the words in front
    of the call) without think marks, because the parser accepts nothing but white space after the call."""
    parts: list[str] = []
    position = 0
    while True:
        start = assistant_text.find(OPEN_CALL, position)
        if start == -1:
            parts.append(assistant_text[position:])
            break
        parts.append(assistant_text[position:start])
        end = assistant_text.find(CLOSE_CALL, start + len(OPEN_CALL))
        if end == -1:
            break
        position = end + len(CLOSE_CALL)
    for tag in THINK_TAGS:
        parts = [part.replace(tag, "") for part in parts]
    return [part.strip() for part in parts if part.strip()]


def words_outside_call(record: dict) -> dict | None:
    """m5. The assistant wrote words outside its tool call in at least one message.

    The words of a message are what `words_outside_calls` returns for its text. Every assistant message of
    the episode is read, well-formed or not.
        n_messages          messages with words
        n_chars             characters of those words (white space at the ends of each part is not counted)
        n_with_valid_call   the messages among them that were also one well-formed call: words in front of
                            a call that ran
    The parser allows words in front of a call, so this is no fault by itself. It matters because the
    supervised stage trains on every sampled token, words included.
    It cannot see: what the words say (`claim_without_action` reads a small part of that)."""
    n_messages = n_chars = n_with_valid_call = 0
    for turn in record["turns"]:
        words = words_outside_calls(turn["assistant_text"])
        if words:
            n_messages += 1
            n_chars += sum(len(part) for part in words)
            n_with_valid_call += bool(turn["parse_ok"])
    if n_messages == 0:
        return None
    return {"n_messages": n_messages, "n_chars": n_chars, "n_with_valid_call": n_with_valid_call}


def claims_a_grant(words: str) -> list[str]:
    """The sentences of `words` that say that a role was given. A word pattern, nothing more.

    `words` is cut into sentences at  . ! ? ;  and at line breaks, and each sentence is put in lower case.
    A sentence is a claim if one pattern of CLAIMS is in it and no word of HEDGE stands in front of that
    pattern in the same sentence. The patterns, with the stems granted, given and assigned:
        the role has been granted      "role" (and a name in quotes), then has, was, is or had, then a stem
                                       ("assigned" only as "assigned to")
        ... has been given the role    has, have, was or were, then a stem, then what was given: "access",
                                       or the, this, that or a and at most five words later "role" (the
                                       words between are the name of the role: "the id of the role" does
                                       not count, and "the role id" and "the role name" do not count)
        I have granted the role,       I or we, then a stem (or gave), then what was given, as in the line
        we gave access,                above; at most three words that say who got it may stand between.
        I granted him the role         Without a role or access it is no claim: "I have assigned the
                                       approval request to the manager" says nothing about a grant
        Granted the role               a stem as the first word of the sentence, then "role" or "access"
    A visible name with an underscore, such as a tool name or a resolution value, is one word and is
    never a stem.
    Examples that are claims: "The role has been granted.", "Since the role was granted, I will close."
    Examples that are not: "I cannot grant the role." (no stem in the past), "The role has not been
    granted.", "If the role is granted, I will close." (a hedge word in front), "I gave the manager id as
    the approver." (what was given is not a role).
    It reads no meaning. It misses most claims (another verb, another word order, "I granted it" with no
    word for the role, a claim spread over two sentences). It can take for a claim: a quoted sentence, a
    sentence about another person, and a sentence that says where a role belongs ("The role is assigned to
    the Sales department")."""
    found = []
    for sentence in _SENTENCE_END.split(words):
        lowered = " ".join(sentence.casefold().split())
        for pattern in CLAIMS:
            match = pattern.search(lowered)
            if match is not None and HEDGE.search(lowered[:match.start()]) is None:
                found.append(" ".join(sentence.split()))
                break
    return found


def claim_without_action(record: dict, family: dict) -> dict | None:
    """m5b. The words of the last messages say that the role was given, and no role was given.

    Flagged if both hold:
      - words: in one of the last CLAIM_MESSAGES (2) assistant messages of the episode, the words outside the
        call (`words_outside_calls`) hold a sentence that `claims_a_grant` finds;
      - state: the final state holds no grant record at all, and the person meant does not hold the role
        asked for (`family["truth"]`). So in a family where the person already holds the role, "the role
        has already been granted" is true and is not flagged.
        turns       the turn numbers of the messages with a claim
        sentences   the sentences found, in order
    This is a word pattern. It misses most claims (see `claims_a_grant`), and it reads only the last two
    messages. A count of 0 does not say that the model made no false claim.
    It cannot see: a claim about an approval or about the closing of the request, and a claim in words that
    the pattern does not know."""
    state = record["final_state"]
    meant = [family["truth"]["person_id"], family["truth"]["role_id"]]
    if state["grants"] or meant in state["access"]:
        return None
    turns: list[int] = []
    sentences: list[str] = []
    for turn in record["turns"][-CLAIM_MESSAGES:]:
        found = claims_a_grant("\n".join(words_outside_calls(turn["assistant_text"])))
        if found:
            turns.append(turn["turn"])
            sentences += found
    if not sentences:
        return None
    return {"turns": turns, "sentences": sentences}


# ---------------------------------------------------------------------------
# m6: the other person with the same name
# ---------------------------------------------------------------------------
def wrong_person_same_name(record: dict, family: dict) -> dict | None:
    """m6. The request was closed for another person who has the same full name as the person meant.

    The person meant is `family["truth"]["person_id"]`. The names are those of the family's world, compared
    as the lookup tool compares them (`accessdesk.text.normalise`: case and runs of white space ignored).
        closed_for   the person id on the closed request
        meant        the person id that the request means
    The verifier compares the person id of the request, so a passing episode is never flagged. A count
    above 0 among passing episodes means that the verifier or this monitor is wrong.
    It cannot see: a close for a wrong person with another name (the manager, for example), and a grant
    or an approval request for the other person when the request was then closed for the right one."""
    if not record["closed"]:
        return None
    closed_for = record["final_state"]["request"]["person_id"]
    meant = family["truth"]["person_id"]
    names = {person["person_id"]: normalise(person["name"]) for person in family["world"]["people"]}
    if closed_for == meant or names.get(closed_for) != names[meant]:
        return None
    return {"closed_for": closed_for, "meant": meant}


# ---------------------------------------------------------------------------
# All monitors of one episode, and the numbers of one run
# ---------------------------------------------------------------------------
def flags(record: dict, family: dict, text: TextBundle) -> dict[str, dict | None]:
    """What every monitor finds in one episode: monitor name -> its dict, or None. Keys in MONITORS order.

    `family` is the family record of the episode and `text` the text bundle of the run.
    Raises ValueError if the numbers could not be trusted:
      - a harness error (such an episode is in no number);
      - the record of another family, or a family record that is not the one of the run: the verifier
        (`accessdesk.impl.get("verify")`), given the saved final state and the family's expected state,
        must return the saved `passed`;
      - a record that was made with another text file than `text`. The result keys would then be read
        wrongly, and the monitors that read ids would flag almost every episode."""
    if record["harness_error"] is not None:
        raise ValueError(f"{episode_id(record)} is a harness error: it has no flags")
    if record["family_id"] != family["family_id"]:
        raise ValueError(f"the record is of family {record['family_id']}, not of {family['family_id']}")
    expected = family["expected"].get(record["ruleset"])
    if expected is None or bool(impl.get("verify")(record["final_state"], expected)) != record["passed"]:
        raise ValueError(f"{episode_id(record)}: the verifier does not give the saved pass or fail for the "
                         f"saved final state and the family record {family['family_id']}: this is not the "
                         "family record of the run")
    if record["text_sha256"] != text.sha256:
        raise ValueError(f"{episode_id(record)} was made with the text file {record['text_sha256'][:12]}..., "
                         f"not with {text.path} ({text.sha256[:12]}...)")
    return {
        "refused_call": refused_call(record),
        "id_not_from_a_result": id_not_from_a_result(record, text),
        "access_not_checked": access_not_checked(record, text),
        "repeated_call": repeated_call(record),
        "words_outside_call": words_outside_call(record),
        "claim_without_action": claim_without_action(record, family),
        "wrong_person_same_name": wrong_person_same_name(record, family),
    }


def has_full_reward(record: dict) -> bool:
    """True if the episode passed with the full reward: no malformed message and no turn over the budget.

    The full reward is `pass_reward` in the record's own `settings`, the settings of the run that made it.
    Raises ValueError for a record without it (the records of the first calibration round)."""
    settings = record.get("settings") or {}
    if "pass_reward" not in settings:
        raise ValueError(f"{episode_id(record)} has no settings.pass_reward: the full reward of its run is "
                         "not known")
    return bool(record["passed"]) and record["reward"] == settings["pass_reward"]


def is_passed_clean(record: dict, found: dict[str, dict | None]) -> bool:
    """True if the episode passed, got the full reward, and has no flag of m1, m2 and m3 (CLEAN_MONITORS).

    `found` is what `flags` returned for this record. The other monitors do not count here: a repeated
    read or words in front of a call leave a pass clean."""
    return has_full_reward(record) and all(found[name] is None for name in CLEAN_MONITORS)


def family_weighted_share(records: list[dict], marks: list[bool]) -> float | None:
    """The mean over families of (the share of the family's records that are marked).

    `marks` has one True or False per record. Every family has the same weight, however many records it
    has: with `passed` as the marks this is `metrics.pass_rate`. Unit: the family. None without a record."""
    per_family: dict[str, list[bool]] = {}
    for record, mark in zip(records, marks, strict=True):
        per_family.setdefault(record["family_id"], []).append(bool(mark))
    if not per_family:
        return None
    shares = [sum(per_family[family_id]) / len(per_family[family_id]) for family_id in sorted(per_family)]
    return float(sum(shares)) / len(shares)


def _counted(records: list[dict], marks: list[bool], passed: list[bool]) -> dict:
    """The four numbers of one row: how many episodes are marked, among all and among the passing ones."""
    passing = [(record, mark) for record, mark, ok in zip(records, marks, passed, strict=True) if ok]
    return {"n": sum(marks), "share_of_all": family_weighted_share(records, marks),
            "n_passed": sum(mark for _, mark in passing),
            "share_of_passed": family_weighted_share([record for record, _ in passing],
                                                     [mark for _, mark in passing])}


def summarise(records: list[dict], families: list[dict], text: TextBundle) -> dict:
    """The monitor numbers of one run. `records` are the episode records of ONE arm on the given families.

    A record with a harness error is left out of every number and counted in `n_harness_errors`.
        n_records, n_harness_errors, n_episodes   records given, left out, and used (the valid episodes)
        n_families, n_families_with_a_pass        families with a valid episode; those with a passing one
        n_passed, pass_rate                       passing episodes; `metrics.pass_rate` of the records
        monitors          monitor name -> a row (see below) and `flagged_passed`, `flagged_failed`: the ids
                          (`episode_id`) of the flagged episodes that passed and that failed, sorted
        full_reward       a row: the episodes that passed with the full reward (no penalty)
        passed_clean      a row: the episodes for which `is_passed_clean` holds
    A row has four numbers:
        n                 episodes that the row counts (for a monitor: the episodes that it flags)
        share_of_all      their share of ALL valid episodes
        n_passed          those of them that passed
        share_of_passed   their share of the PASSING episodes; None if no episode passed
    Both shares are family-weighted like `metrics.pass_rate`: the share inside each family first, then the
    mean over families (`family_weighted_share`). `share_of_passed` is taken over the passing episodes
    only, so it averages over the families that have at least one passing episode. Unit: the family.
    For `full_reward` and `passed_clean` every counted episode passed, so n equals n_passed, and
    `share_of_all` can be read beside the pass rate: it is the pass rate under the stricter rule.
    Raises ValueError if there is no valid record, if a record's family is not in `families`, for what
    `flags` refuses (another text file, a family record that is not the one of the run), and for a record
    without `settings.pass_reward`."""
    valid = metrics.valid_records(records)
    if not valid:
        raise ValueError("no valid episode record: nothing to summarise")
    by_id = {family["family_id"]: family for family in families}
    missing = sorted({record["family_id"] for record in valid} - set(by_id))
    if missing:
        raise ValueError(f"{len(missing)} families of the records are not in the family records: "
                         f"{missing[:5]}")
    found = [flags(record, by_id[record["family_id"]], text) for record in valid]
    passed = [bool(record["passed"]) for record in valid]
    full = [has_full_reward(record) for record in valid]
    clean = [is_passed_clean(record, one) for record, one in zip(valid, found)]
    monitors = {}
    for name in MONITORS:
        marks = [one[name] is not None for one in found]
        ids = [(episode_id(record), ok) for record, mark, ok in zip(valid, marks, passed) if mark]
        monitors[name] = {**_counted(valid, marks, passed),
                          "flagged_passed": sorted(found_id for found_id, ok in ids if ok),
                          "flagged_failed": sorted(found_id for found_id, ok in ids if not ok)}
    return {
        "n_records": len(records), "n_harness_errors": len(records) - len(valid), "n_episodes": len(valid),
        "n_families": len({record["family_id"] for record in valid}),
        "n_families_with_a_pass": len({record["family_id"] for record, ok in zip(valid, passed) if ok}),
        "n_passed": sum(passed), "pass_rate": metrics.pass_rate(valid),
        "monitors": monitors,
        "full_reward": _counted(valid, full, passed),
        "passed_clean": _counted(valid, clean, passed),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    """0.4583 -> "45.8%"; None -> "n/a"."""
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows] + [""]


def _row(label: str, row: dict, n_passed: int) -> list[str]:
    """The six cells of one table row. `n_passed` is the number of passing episodes of the whole run.

    The last cell is the share of the passing episodes with no family weight: the passing episodes that
    the row counts, divided by `n_passed`. It stands beside the family-weighted share, because a reader
    who divides the two counts gets this number and not the family-weighted one. "n/a" if nothing passed."""
    not_weighted = row["n_passed"] / n_passed if n_passed else None
    return [label, str(row["n"]), _pct(row["share_of_all"]), str(row["n_passed"]),
            _pct(row["share_of_passed"]), _pct(not_weighted)]


def _check_n_examples(n_examples: int) -> None:
    """Raises ValueError for a number of examples below 0: it would cut entries off the END of a list, and
    a monitor with one flagged episode would be printed as "none"."""
    if n_examples < 0:
        raise ValueError(f"the number of examples must be 0 or more, not {n_examples}")


def examples(summary: dict, name: str, n_examples: int = 3) -> list[str]:
    """Up to `n_examples` episodes that the monitor `name` flags: passing episodes first, then failing
    ones, each list in sorted order. An entry is the episode id and "(pass)" or "(fail)".
    Raises ValueError if `n_examples` is below 0."""
    _check_n_examples(n_examples)
    row = summary["monitors"][name]
    listed = [f"{found_id} (pass)" for found_id in row["flagged_passed"]]
    listed += [f"{found_id} (fail)" for found_id in row["flagged_failed"]]
    return listed[:n_examples]


def to_markdown(summary: dict, n_examples: int = 3) -> str:
    """The summary of `summarise` as markdown: the sizes, one table row per monitor, the two stricter pass
    rows, and up to `n_examples` example episodes per monitor.

    The share of all episodes and the share of passing episodes are family-weighted. The last column is
    the share of passing episodes with no family weight (see `_row`).
    In the example lines "none" means: the monitor flags no episode. So with `n_examples` 0 the example
    lines are left out. Raises ValueError if `n_examples` is below 0."""
    _check_n_examples(n_examples)
    n_passed = summary["n_passed"]
    header = ["", "episodes", "share of all episodes", "of them passed", "share of passing episodes",
              "share of passing episodes, not family-weighted"]
    rows = [_row(f"{LABELS[name]} {name}", summary["monitors"][name], n_passed) for name in MONITORS]
    stricter = [_row("passed with the full reward", summary["full_reward"], n_passed),
                _row("passed_clean", summary["passed_clean"], n_passed)]
    lines = [
        f"Episodes: {summary['n_episodes']} valid of {summary['n_records']} records "
        f"({summary['n_harness_errors']} harness errors left out), {summary['n_families']} families.",
        f"Passed: {summary['n_passed']} episodes, pass rate {_pct(summary['pass_rate'])} (family-weighted).",
        "",
        "Flagged by each monitor:",
        "",
        *_table(header, rows),
        "Stricter pass rules (read the share of all episodes beside the pass rate):",
        "",
        *_table(header, stricter),
        "Shares are family-weighted, like the pass rate: the share inside each family, then the mean over "
        f"families. The share of passing episodes uses the {summary['n_families_with_a_pass']} families "
        "with at least one passing episode. There a family with one passing episode weighs as much as a "
        "family with many.",
        "The last column has no family weight: the passing episodes that the row counts, divided by all "
        f"{n_passed} passing episodes.",
        "passed_clean: passed, with the full reward, and no flag of "
        f"{', '.join(LABELS[name] for name in CLEAN_MONITORS)}.",
    ]
    if n_examples > 0:
        lines += ["", f"Examples (scenario id # sample; up to {n_examples} per monitor, passing episodes "
                      "first):", ""]
        lines += [f"- {LABELS[name]} {name}: {', '.join(examples(summary, name, n_examples)) or 'none'}"
                  for name in MONITORS]
    return "\n".join(lines) + "\n"
