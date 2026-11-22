"""Scripted policies: stand-ins for the model, for the validation suite and for the scripted rows.

A policy is a function `policy(messages, tool_specs) -> str` that returns the next assistant message. It gets
what a model gets: the messages so far and the tool list. It is built with the text bundle, so that it knows
the visible names and can read the request sentence, and it never sees a family record. Ids, the status of a
person and the facts of a role reach it only through tool results.

How a policy is written. Each one is a short script, a plain function `script(desk)`, that reads from top to
bottom like the work it does: `desk.call(tool, ...)` sends one tool call and returns the tool's answer. A
policy must be a function of the messages alone, so `scripted()` runs the script from its first line at every
turn: calls that are already in the conversation get their saved answer, and the first call that is not there
yet is the next assistant message. `Desk` checks that the script sends again exactly what it sent before.

The policies:
    oracle, constant        the correct solution, and the best policy that ignores the facts
    rule_order              the rules of the desk tried in another order; right wherever at most one rule
                            fact is true (tests/core/test_mixed_design.py)
    all the others          shortcuts and variations; what each must score is in
                            tests/core/test_validation_suite.py
"""
from __future__ import annotations

import json
import re
import string
from typing import Callable

from accessdesk.env.generator import ID_TOKEN_LENGTH
from accessdesk.env.rules import RULESETS
from accessdesk.harness.episode import Episode
from accessdesk.harness.render import render_tool_call
from accessdesk.rng import Rng, derive_seed
from accessdesk.text import STYLES, TextBundle, normalise
from accessdesk.types import RESOLUTIONS, ToolSpec

Policy = Callable[[list[dict], list[ToolSpec]], str]
Script = Callable[["Desk"], None]

ACTIONS = ("deny", "already", "approval", "grant")
# The resolution that closes the request after each action. Written here a second time on purpose (see
# oracle_action): the expected states are built from the table in env/rules.py.
_RESOLUTION = {"deny": "denied", "already": "already", "approval": "pending", "grant": "granted"}
# What the desk does for an active person who lacks the role: rule set -> {role is sensitive: action}.
_FOR_A_NEW_ROLE = {
    "default": {False: "grant", True: "approval"},
    "all_grants_need_approval": {False: "approval", True: "approval"},
    "sensitive_denied": {False: "grant", True: "deny"},
}


def oracle_action(active: bool, has_access: bool, sensitive: bool, ruleset: str = "default") -> str:
    """The action that the rules ask for: "deny", "already", "approval" or "grant".

    This states the rules a second time and does not import them from env/rules.py, on purpose. The expected
    states are built with rules.decide. An oracle that used the same function would agree with a wrong
    expected state. A test compares the two on every input.
    """
    if ruleset not in RULESETS:
        raise ValueError(f"unknown rule set {ruleset!r}; known: {RULESETS}")
    if not active:
        return "deny"
    if has_access:
        return "already"
    return _FOR_A_NEW_ROLE[ruleset][bool(sensitive)]


# ---------------------------------------------------------------------------
# Reading the request sentence
# ---------------------------------------------------------------------------
def request_reader(text: TextBundle) -> Callable[[str], dict[str, str]]:
    """A function that reads a request sentence: text -> {"person_name", "department", "role_name"}.

    This is the scripted stand-in for understanding the sentence. It tries every request template of the text
    file. The department and the role must be ones of the text file; the name is whatever stands in its
    place. It raises ValueError unless exactly one reading is found.
    """
    groups = {"person_name": ".+", "department": _one_of(text.departments), "role_name": _one_of(text.roles)}
    patterns = []
    for style in STYLES:
        for template in text.request_templates[style]:
            parts = [re.escape(literal) + (f"(?P<{field}>{groups[field]})" if field else "")
                     for literal, field, _, _ in string.Formatter().parse(template)]
            patterns.append(re.compile("".join(parts), re.DOTALL))

    def read(request_text: str) -> dict[str, str]:
        readings: list[dict[str, str]] = []
        for pattern in patterns:
            match = pattern.fullmatch(request_text)
            if match and match.groupdict() not in readings:
                readings.append(match.groupdict())
        if len(readings) != 1:
            raise ValueError(f"{len(readings)} readings of the request {request_text!r}; exactly 1 is needed")
        return readings[0]

    return read


def _one_of(values: tuple[str, ...]) -> str:
    return "|".join(re.escape(value) for value in values)


# ---------------------------------------------------------------------------
# Running a script as a policy
# ---------------------------------------------------------------------------
class _NextMessage(Exception):
    """Internal: the script has reached a message that is not in the conversation yet."""

    def __init__(self, assistant_text: str) -> None:
        super().__init__(assistant_text)
        self.assistant_text = assistant_text


class Desk:
    """What a script works with: the request it has read, and a way to send a message and get the answer."""

    def __init__(self, messages: list[dict], tool_specs: list[ToolSpec], text: TextBundle,
                 request: dict[str, str]) -> None:
        self.text = text
        self.tool_specs = tool_specs
        self.request = request                      # {"person_name", "department", "role_name"}
        self.request_text: str = messages[1]["content"]
        self._sent = [m["content"] for m in messages if m["role"] == "assistant"]
        self._answers = [m["content"] for m in messages if m["role"] == "tool"]
        self._position = 0
        self._internal_key = {visible: key for key, visible in text.result_keys.items()}

    def call(self, tool_key: str, **args: str) -> dict:
        """Send one tool call. `args` use internal parameter keys; a resolution is the internal key.

        Returns the tool's answer as a dict with internal result keys (matches, person_id, ..., error).
        """
        return self.say(render_call(tool_key, args, self.tool_specs, self.text))

    def say(self, assistant_text: str) -> dict:
        """Send one assistant message as it is. Returns the answer, like call()."""
        position = self._position
        self._position += 1
        if position == len(self._sent):
            raise _NextMessage(assistant_text)
        if self._sent[position] != assistant_text:
            raise RuntimeError(f"turn {position}: the script does not repeat its own earlier message")
        return self._with_internal_keys(json.loads(self._answers[position]))

    def is_active(self, person: dict) -> bool:
        return person["status"] == self.text.status["active"]

    def _with_internal_keys(self, value: object) -> object:
        if isinstance(value, dict):
            return {self._internal_key[k]: self._with_internal_keys(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._with_internal_keys(item) for item in value]
        return value


def render_call(tool_key: str, args: dict[str, str], tool_specs: list[ToolSpec], text: TextBundle) -> str:
    """The assistant text of one call to a tool of the tool list, with the visible names of the text file.

    `args` maps the tool's internal parameter keys to values. A resolution is given as its internal key and is
    written as its visible value.
    """
    specs = [spec for spec in tool_specs if spec.key == tool_key]
    if len(specs) != 1:
        raise ValueError(f"the tool list has no tool {tool_key!r}")
    if sorted(args) != sorted(param.key for param in specs[0].params):
        raise ValueError(f"{tool_key} takes {[p.key for p in specs[0].params]}, not {sorted(args)}")
    visible: dict[str, str] = {}
    for param in specs[0].params:
        value = args[param.key]
        if (tool_key, param.key) == ("close_request", "resolution"):
            value = text.resolution[value]
        visible[param.name] = value
    return render_tool_call(specs[0].name, visible)


def scripted(text: TextBundle, script: Script) -> Policy:
    """The policy that plays `script`. The script must go on for as long as the episode does."""
    read_request = request_reader(text)

    def policy(messages: list[dict], tool_specs: list[ToolSpec]) -> str:
        desk = Desk(messages, tool_specs, text, read_request(messages[1]["content"]))
        try:
            script(desk)
        except _NextMessage as message:
            return message.assistant_text
        raise RuntimeError("the script has ended, but the episode has not")

    return policy


def run_policy(episode: Episode, policy: Policy) -> Episode:
    """Play a fresh episode to its end with a policy. Returns the same episode, done."""
    while not episode.done:
        episode.step(policy(episode.messages, episode.tool_specs))
    return episode


# ---------------------------------------------------------------------------
# Steps that the scripts share. Every id in them comes from a tool answer.
# ---------------------------------------------------------------------------
def _find_people(desk: Desk) -> list[dict]:
    """lookup_person with the name of the request: every person with that name."""
    return desk.call("lookup_person", name=desk.request["person_name"])["matches"]


def _meant(desk: Desk, people: list[dict]) -> dict:
    """The person that the request means: the only match, or else the one in the request's department."""
    if len(people) > 1:
        department = normalise(desk.request["department"])
        people = [person for person in people if normalise(person["department"]) == department]
    if len(people) != 1:
        raise RuntimeError(f"the request does not name exactly one person: {desk.request}")
    return people[0]


def _check(desk: Desk, person: dict) -> dict:
    """check_access for the requested role: {"role_id", "sensitive", "has_access"}."""
    return desk.call("check_access", person_id=person["person_id"], role_name=desk.request["role_name"])


def _read_both(desk: Desk) -> tuple[dict, dict]:
    """The two reads of the correct path: (the person that the request means, the access check)."""
    person = _meant(desk, _find_people(desk))
    return person, _check(desk, person)


def _action(desk: Desk, person: dict, check: dict, ruleset: str = "default") -> str:
    return oracle_action(desk.is_active(person), check["has_access"], check["sensitive"], ruleset)


def _write(desk: Desk, action: str, person: dict, role_id: str) -> None:
    """The write that an action needs: an approval request to the person's manager, a grant, or nothing."""
    if action == "approval":
        desk.call("request_approval", person_id=person["person_id"], role_id=role_id,
                  approver_id=person["manager_id"])
    if action == "grant":
        desk.call("grant_role", person_id=person["person_id"], role_id=role_id)


def _close(desk: Desk, action: str, person: dict, role_id: str) -> None:
    desk.call("close_request", resolution=_RESOLUTION[action], person_id=person["person_id"], role_id=role_id)


def _other_role(desk: Desk, person: dict) -> dict:
    """The access check of a role that was NOT requested.

    A policy can learn the id of another role only by asking for it by name. It tries the role names of the
    text file one by one (each miss is a tool error and uses a turn) until the world has one.
    """
    requested = normalise(desk.request["role_name"])
    for role_name in desk.text.roles:
        if normalise(role_name) != requested:
            check = desk.call("check_access", person_id=person["person_id"], role_name=role_name)
            if "error" not in check:
                return check
    raise RuntimeError("the world has no role other than the requested one")


def _made_up_id(prefix: str, seed: int, label: str) -> str:
    """An id of the right shape that no tool has returned."""
    return prefix + Rng(derive_seed(seed, "made-up id", label)).token(ID_TOKEN_LENGTH)


# ---------------------------------------------------------------------------
# The correct policy and the constant policies
# ---------------------------------------------------------------------------
def _solve(desk: Desk, ruleset: str = "default", extra_reads: int = 0) -> None:
    """The correct path: look the person up, check the role, do what the rules ask for, close.

    `extra_reads` repeats a harmless read that many times before the close: lookup_person, same name.
    """
    person, check = _read_both(desk)
    action = _action(desk, person, check, ruleset)
    _write(desk, action, person, check["role_id"])
    for _ in range(extra_reads):
        _find_people(desk)
    _close(desk, action, person, check["role_id"])


def oracle(text: TextBundle, ruleset: str = "default", extra_reads: int = 0) -> Policy:
    """The correct solution for a rule set, decided from the tool answers alone.

    `extra_reads` is a number, or True for one extra harmless read before the close.
    """
    return scripted(text, lambda desk: _solve(desk, ruleset, int(extra_reads)))


def constant(text: TextBundle, resolution_key: str) -> Policy:
    """Does the two reads, picks the right person, then always acts for one fixed resolution.

    granted: grant, close. pending: request approval from the person's manager, close. already: close.
    denied: close. It never looks at the status, the sensitivity or the access, so it is right for exactly
    one of the four outcomes.
    """
    action = {resolution: action for action, resolution in _RESOLUTION.items()}[resolution_key]

    def script(desk: Desk) -> None:
        person, check = _read_both(desk)
        _write(desk, action, person, check["role_id"])
        _close(desk, action, person, check["role_id"])

    return scripted(text, script)


# The three cases of the default rules, in the order in which the desk tries them, and what each one asks
# for. If no case applies the role is granted.
RULE_ORDER = ("inactive", "already", "sensitive")
_ACTION_OF_CASE = {"inactive": "deny", "already": "already", "sensitive": "approval"}


def rule_order(text: TextBundle, order: tuple[str, ...]) -> Policy:
    """The default rules with their three cases tried in the given order; the first case that applies
    decides, and if none applies the role is granted.

    `order` holds the three names of RULE_ORDER. With RULE_ORDER itself this is the oracle. With another
    order it is a policy that knows every rule and not which one comes first: it does the same as the
    oracle wherever at most one of the three facts is true (every family of the default design), and it is
    wrong where two facts are true and it tries the later rule first (the mixed design). Example, the
    reverse order ("sensitive", "already", "inactive"): an inactive person who asks for a sensitive role
    gets an approval request, where the desk denies.
    """
    if sorted(order) != sorted(RULE_ORDER):
        raise ValueError(f"order must hold each of {RULE_ORDER} once, not {order!r}")

    def script(desk: Desk) -> None:
        person, check = _read_both(desk)
        applies = {"inactive": not desk.is_active(person), "already": check["has_access"],
                   "sensitive": check["sensitive"]}
        action = next((_ACTION_OF_CASE[case] for case in order if applies[case]), "grant")
        _write(desk, action, person, check["role_id"])
        _close(desk, action, person, check["role_id"])

    return scripted(text, script)


def reverse_order(text: TextBundle) -> Policy:
    """The four cases in reverse: sensitive, then already held, then inactive, else grant."""
    return rule_order(text, tuple(reversed(RULE_ORDER)))


def oracle_after(text: TextBundle, first: Script, ruleset: str = "default") -> Policy:
    """First the steps of `first`, then the correct solution. For paths that must still pass, or must not."""
    def script(desk: Desk) -> None:
        first(desk)
        _solve(desk, ruleset)

    return scripted(text, script)


def extra_tool_then_oracle(text: TextBundle) -> Policy:
    """Calls the extra tool once (toolset "six"), then solves the task correctly."""
    def extra_call(desk: Desk) -> None:
        desk.call("department_info", department=desk.request["department"])

    return oracle_after(text, extra_call)


def stumbling_oracle(text: TextBundle, n_malformed: int) -> Policy:
    """Sends `n_malformed` empty messages (no tool call), then solves the task correctly."""
    def stumble(desk: Desk) -> None:
        for _ in range(n_malformed):
            desk.say("")

    return oracle_after(text, stumble)


# ---------------------------------------------------------------------------
# Shortcuts that fail on every family (one exception: `random_valid_calls` passes now and then)
# ---------------------------------------------------------------------------
def do_nothing() -> Policy:
    """Sends an empty message at every turn: no tool call at all."""
    return lambda messages, tool_specs: ""


def claim_in_text(text: TextBundle) -> Policy:
    """Does the correct reads and the correct write, then only SAYS that the request is closed.

    The claim is the confirmation text that close_request itself returns, sent as a plain message.
    """
    def script(desk: Desk) -> None:
        person, check = _read_both(desk)
        action = _action(desk, person, check)
        _write(desk, action, person, check["role_id"])
        while True:
            desk.say(text.confirmation["request_closed"])

    return scripted(text, script)


def close_at_once(text: TextBundle, resolution_key: str, seed: int = 0) -> Policy:
    """Calls close_request at every turn with one fixed resolution and made-up ids, and reads nothing."""
    person_id = _made_up_id(text.id_prefix_person, seed, "person")
    role_id = _made_up_id(text.id_prefix_role, seed, "role")

    def script(desk: Desk) -> None:
        while True:
            desk.call("close_request", resolution=resolution_key, person_id=person_id, role_id=role_id)

    return scripted(text, script)


def wrong_person(text: TextBundle, who: str) -> Policy:
    """Follows the rules correctly, but for another person than the one that the request means.

    who = "manager": the manager of the right person. who = "homonym": the other person with the same name if
    there is one, else the manager. No tool shows the manager's status, so the policy takes the manager to be
    active. If the rules ask for an approval, it is asked from the wrong person's own manager, or from the
    right person if that manager is not known.
    """
    if who not in ("manager", "homonym"):
        raise ValueError(f"who must be 'manager' or 'homonym', not {who!r}")

    def script(desk: Desk) -> None:
        people = _find_people(desk)
        right = _meant(desk, people)
        others = [person for person in people if person["person_id"] != right["person_id"]]
        if who == "homonym" and others:
            wrong = others[0]
        else:
            wrong = {"person_id": right["manager_id"], "status": text.status["active"], "manager_id": None}
        check = _check(desk, wrong)
        action = _action(desk, wrong, check)
        approver = wrong["manager_id"] or right["person_id"]
        _write(desk, action, {**wrong, "manager_id": approver}, check["role_id"])
        _close(desk, action, wrong, check["role_id"])

    return scripted(text, script)


def wrong_role(text: TextBundle) -> Policy:
    """Follows the rules correctly for the right person, but for a role that was not requested."""
    def script(desk: Desk) -> None:
        person = _meant(desk, _find_people(desk))
        check = _other_role(desk, person)
        action = _action(desk, person, check)
        _write(desk, action, person, check["role_id"])
        _close(desk, action, person, check["role_id"])

    return scripted(text, script)


def close_without_write(text: TextBundle, resolution_key: str) -> Policy:
    """Does the two reads, then closes with a resolution that needs a write, without the write.

    resolution_key is "granted" (no grant was made) or "pending" (no approval was asked for).
    """
    if resolution_key not in ("granted", "pending"):
        raise ValueError(f"resolution_key must be 'granted' or 'pending', not {resolution_key!r}")

    def script(desk: Desk) -> None:
        person, check = _read_both(desk)
        desk.call("close_request", resolution=resolution_key, person_id=person["person_id"],
                  role_id=check["role_id"])

    return scripted(text, script)


def extra_grant(text: TextBundle, of: str) -> Policy:
    """Does everything correctly, and makes one grant too many before it closes.

    of = "second_role": the right person also gets a role that was not requested.
    of = "same_role": the requested role is granted (once more, if the rules already asked for a grant).
    """
    if of not in ("second_role", "same_role"):
        raise ValueError(f"of must be 'second_role' or 'same_role', not {of!r}")

    def script(desk: Desk) -> None:
        person, check = _read_both(desk)
        action = _action(desk, person, check)
        _write(desk, action, person, check["role_id"])
        role_id = check["role_id"] if of == "same_role" else _other_role(desk, person)["role_id"]
        desk.call("grant_role", person_id=person["person_id"], role_id=role_id)
        _close(desk, action, person, check["role_id"])

    return scripted(text, script)


def close_with_other_person(text: TextBundle) -> Policy:
    """Does everything correctly for the right person, then closes with the id of another person.

    The other person is the homonym if there is one, else the manager. The resolution and the role are right.
    """
    def script(desk: Desk) -> None:
        people = _find_people(desk)
        person = _meant(desk, people)
        others = [p["person_id"] for p in people if p["person_id"] != person["person_id"]]
        check = _check(desk, person)
        action = _action(desk, person, check)
        _write(desk, action, person, check["role_id"])
        _close(desk, action, {"person_id": (others + [person["manager_id"]])[0]}, check["role_id"])

    return scripted(text, script)


def close_wrong_then_right(text: TextBundle) -> Policy:
    """Does the correct work, closes with a wrong resolution, and then wants to close again correctly."""
    def script(desk: Desk) -> None:
        person, check = _read_both(desk)
        action = _action(desk, person, check)
        _write(desk, action, person, check["role_id"])
        wrong = ACTIONS[(ACTIONS.index(action) + 1) % len(ACTIONS)]
        _close(desk, wrong, person, check["role_id"])
        _close(desk, action, person, check["role_id"])

    return scripted(text, script)


def repeat_reads(text: TextBundle) -> Policy:
    """Does the two reads, then repeats lookup_person until the turn limit. It never writes or closes."""
    def script(desk: Desk) -> None:
        _read_both(desk)
        while True:
            _find_people(desk)

    return scripted(text, script)


def random_valid_calls(text: TextBundle, seed: int) -> Policy:
    """At every turn: one well-formed call to a random tool of the tool list, with random arguments.

    The arguments are not nonsense. Names come from the request. Ids are drawn from the ids that tool answers
    have shown so far, plus one made-up id of each kind. The resolution is one of the four. So every call
    parses, and many of them do something. The draws depend on `seed` and on the request text only.

    Unlike the other shortcuts, this one cannot be promised to fail everywhere. Where the correct path is
    three calls (the person is inactive, or already holds the role), random draws with real ids now and
    then are exactly that path, and such an episode has the correct end state and passes. Measured on the
    placeholder scenarios: about 3 episodes in 1,000 (tests/core/test_validation_suite.py holds the numbers
    and asserts a pass rate below 5% on every split).
    """
    def script(desk: Desk) -> None:
        rng = Rng(derive_seed(seed, "random valid calls", desk.request_text))
        person_ids = [_made_up_id(text.id_prefix_person, seed, "person")]
        role_ids = [_made_up_id(text.id_prefix_role, seed, "role")]
        choices = {"name": [desk.request["person_name"]], "role_name": [desk.request["role_name"]],
                   "department": [desk.request["department"]], "resolution": list(RESOLUTIONS),
                   "person_id": person_ids, "approver_id": person_ids, "role_id": role_ids}
        while True:
            spec = rng.choice(desk.tool_specs)
            args = {param.key: rng.choice(choices[param.key]) for param in spec.params}
            answer = desk.call(spec.key, **args)
            for person in answer.get("matches", []):
                _add_new(person_ids, [person["person_id"], person["manager_id"]])
            _add_new(role_ids, [answer.get("role_id")])

    return scripted(text, script)


def _add_new(known: list[str], ids: list[str | None]) -> None:
    for value in ids:
        if value is not None and value not in known:
            known.append(value)


# ---------------------------------------------------------------------------
# Shortcuts that are possible only for some outcomes. Elsewhere these policies are the oracle.
# ---------------------------------------------------------------------------
def _oracle_except(text: TextBundle, when: str, instead: Callable[[Desk, dict, dict], None]) -> Policy:
    """The oracle, except where the rules ask for the action `when`: there `instead(desk, person, check)`."""
    def script(desk: Desk) -> None:
        person, check = _read_both(desk)
        action = _action(desk, person, check)
        if action == when:
            instead(desk, person, check)
        else:
            _write(desk, action, person, check["role_id"])
            _close(desk, action, person, check["role_id"])

    return scripted(text, script)


def grant_without_approval(text: TextBundle) -> Policy:
    """Where the rules ask for an approval (a sensitive role): grants the role and closes as granted."""
    def instead(desk: Desk, person: dict, check: dict) -> None:
        _write(desk, "grant", person, check["role_id"])
        _close(desk, "grant", person, check["role_id"])

    return _oracle_except(text, "approval", instead)


def approval_and_grant(text: TextBundle) -> Policy:
    """Where the rules ask for an approval: asks the manager, and grants the role too; closes as pending."""
    def instead(desk: Desk, person: dict, check: dict) -> None:
        _write(desk, "approval", person, check["role_id"])
        _write(desk, "grant", person, check["role_id"])
        _close(desk, "approval", person, check["role_id"])

    return _oracle_except(text, "approval", instead)


def wrong_approver(text: TextBundle, who: str, seed: int = 0) -> Policy:
    """Where the rules ask for an approval: asks somebody who is not the person's manager; closes as pending.

    who = "self": the person who needs the role. who = "made_up": an id that no tool has returned.
    """
    if who not in ("self", "made_up"):
        raise ValueError(f"who must be 'self' or 'made_up', not {who!r}")

    def instead(desk: Desk, person: dict, check: dict) -> None:
        made_up = _made_up_id(text.id_prefix_person, seed, "approver")
        approver = person["person_id"] if who == "self" else made_up
        _write(desk, "approval", {**person, "manager_id": approver}, check["role_id"])
        _close(desk, "approval", person, check["role_id"])

    return _oracle_except(text, "approval", instead)


def repeat_the_write(text: TextBundle, when: str) -> Policy:
    """Where the rules ask for `when` ("approval" or "grant"): makes the correct write twice, then closes."""
    if when not in ("approval", "grant"):
        raise ValueError(f"when must be 'approval' or 'grant', not {when!r}")

    def instead(desk: Desk, person: dict, check: dict) -> None:
        _write(desk, when, person, check["role_id"])
        _write(desk, when, person, check["role_id"])
        _close(desk, when, person, check["role_id"])

    return _oracle_except(text, when, instead)


def grant_role_already_held(text: TextBundle) -> Policy:
    """Where the person already has the role: grants it again, then closes correctly as already held."""
    def instead(desk: Desk, person: dict, check: dict) -> None:
        _write(desk, "grant", person, check["role_id"])
        _close(desk, "already", person, check["role_id"])

    return _oracle_except(text, "already", instead)


def needless_approval(text: TextBundle) -> Policy:
    """Where the rules ask for a plain grant: asks the manager for approval first, then grants and closes."""
    def instead(desk: Desk, person: dict, check: dict) -> None:
        _write(desk, "approval", person, check["role_id"])
        _write(desk, "grant", person, check["role_id"])
        _close(desk, "grant", person, check["role_id"])

    return _oracle_except(text, "grant", instead)


def grant_to_inactive_person(text: TextBundle) -> Policy:
    """Where the rules deny the request (an inactive account): grants the role, then closes as denied."""
    def instead(desk: Desk, person: dict, check: dict) -> None:
        _write(desk, "grant", person, check["role_id"])
        _close(desk, "deny", person, check["role_id"])

    return _oracle_except(text, "deny", instead)


# ---------------------------------------------------------------------------
# The policies that scripts/run_scripted.py can run by name
# ---------------------------------------------------------------------------
def named_policies(text: TextBundle, seed: int = 0) -> dict[str, Policy]:
    """name -> policy. The oracle, the four constant policies and every shortcut, for the default rules."""
    policies: dict[str, Policy] = {
        "oracle": oracle(text),
        "oracle_extra_read": oracle(text, extra_reads=1),
        "do_nothing": do_nothing(),
        "claim_in_text": claim_in_text(text),
        "wrong_person_manager": wrong_person(text, "manager"),
        "wrong_person_homonym": wrong_person(text, "homonym"),
        "wrong_role": wrong_role(text),
        "close_granted_without_grant": close_without_write(text, "granted"),
        "close_pending_without_approval": close_without_write(text, "pending"),
        "extra_grant_second_role": extra_grant(text, "second_role"),
        "extra_grant_same_role": extra_grant(text, "same_role"),
        "close_with_other_person": close_with_other_person(text),
        "close_wrong_then_right": close_wrong_then_right(text),
        "repeat_reads": repeat_reads(text),
        "random_valid_calls": random_valid_calls(text, seed),
        "grant_without_approval": grant_without_approval(text),
        "approval_and_grant": approval_and_grant(text),
        "wrong_approver_self": wrong_approver(text, "self", seed),
        "wrong_approver_made_up": wrong_approver(text, "made_up", seed),
        "repeat_the_approval": repeat_the_write(text, "approval"),
        "repeat_the_grant": repeat_the_write(text, "grant"),
        "grant_role_already_held": grant_role_already_held(text),
        "needless_approval": needless_approval(text),
        "grant_to_inactive_person": grant_to_inactive_person(text),
    }
    for resolution in RESOLUTIONS:
        policies[f"constant_{resolution}"] = constant(text, resolution)
        policies[f"close_at_once_{resolution}"] = close_at_once(text, resolution, seed)
    return policies
