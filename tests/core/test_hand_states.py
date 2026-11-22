"""scripts/hand_states.py: the form for the six end states that a person writes by hand, and its check.

What the tests prove:
    the choice   one family of each outcome, two more with rule facts that are true together, two with a
                 same-name pair; fixed by the seed; the place in the form says nothing
    no leak      the text of the form does not depend on the answer that is stored with a family, and no
                 outcome word and no resolution value of the code stands in it
    the rules    the rule list in the form is the list of the plan, and it follows rules.py if that changes
    the check    right answers match; a wrong person, resolution, grant or approval is found and named; an
                 empty answer and an answer that cannot be read are found; a wrong stored state is found
    the file     the form loads again after a person edits it; a broken form or another file stops the check

The families come from three places: four families typed by hand below (the test knows their answers),
small families made for one test, and generated families built in memory (`mixed`, `plain`).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path

import pytest

from accessdesk import impl, types
from accessdesk.env.rules import RULESETS
from accessdesk.env.scenarios import save_families
from tests.conftest import skip_unless_second_is_written

ROOT = Path(__file__).resolve().parents[2]

# Typed by hand: the words of the code for the answer, and the words of the form for the four resolutions.
OUTCOME_WORDS = ("grant", "approval", "already", "inactive")
RESOLUTION_VALUES = ("granted", "pending", "already", "denied")
FORM_WORD = {"granted": "role given", "pending": "manager asked", "already": "held before",
             "denied": "refused"}
ANSWER_LINE = re.compile(r"^  (person|resolution|role_given|asked_to_approve):\s*$")
# All that `check` prints on stdout when it cannot use the form or the scenario files (exit code 1).
DID_NOT_RUN = ["0 of 6 match; open: the check did not run"]


@pytest.fixture(scope="module")
def hand_states():
    spec = importlib.util.spec_from_file_location("hand_states", ROOT / "scripts" / "hand_states.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mixed(text, pools, cfg) -> list[dict]:
    """48 calibration families of the harder design, built in memory: rule facts that are true together,
    and two people of one name in half of the families."""
    from accessdesk.env.generator import make_split
    harder = dataclasses.replace(cfg, mix_facts=True, homonym_share=0.5, n_calib=48)
    return make_split("calib", text, pools, harder)


@pytest.fixture(scope="module")
def plain(text, pools, cfg) -> list[dict]:
    """The 24 calibration families of the design before the freeze day, built in memory: no two rule facts
    are true together, and one family in five has two people of one name. The two values are set here by
    hand: since 2026-11-18 the defaults of accessdesk/config.py are the harder design."""
    from accessdesk.env.generator import make_split
    return make_split("calib", text, pools, dataclasses.replace(cfg, mix_facts=False, homonym_share=0.2))


# ---------------------------------------------------------------------------
# Four families typed by hand, one per outcome. The test knows their answers.
# ---------------------------------------------------------------------------
WORLD = {
    "people": [
        {"person_id": "p1", "name": "Ada Stone", "department": "Finance", "status": "active",
         "manager_id": "p3"},
        {"person_id": "p2", "name": "Ada Stone", "department": "Sales", "status": "inactive",
         "manager_id": "p4"},
        {"person_id": "p3", "name": "Bo Reed", "department": "Finance", "status": "active",
         "manager_id": None},
        {"person_id": "p4", "name": "Cy Lake", "department": "Sales", "status": "active",
         "manager_id": "p3"},
    ],
    "roles": [
        {"role_id": "r1", "name": "Wiki Editor", "sensitive": False},
        {"role_id": "r2", "name": "Payroll Editor", "sensitive": True},
        {"role_id": "r3", "name": "Help Desk Agent", "sensitive": False},
    ],
    "access": [["p1", "r1"], ["p2", "r2"], ["p3", "r2"]],
}
ACCESS = [["p1", "r1"], ["p2", "r2"], ["p3", "r2"]]


def state(access: list, approvals: list, grants: list, resolution: str, person_id: str, role_id: str) -> dict:
    return {"access": access, "approvals": approvals, "grants": grants,
            "request": {"resolution": resolution, "person_id": person_id, "role_id": role_id}}


def hand_family(family_id: str, person_id: str, role_id: str, outcome: str, expected: dict) -> dict:
    """A family record about one person and one role of WORLD, with the end state typed by hand."""
    person = next(p for p in WORLD["people"] if p["person_id"] == person_id)
    role = next(r for r in WORLD["roles"] if r["role_id"] == role_id)
    request = {"person_name": person["name"], "department": person["department"], "role_name": role["name"]}
    wording = "Please give {person_name} from {department} access to the {role_name} role.".format(**request)
    return {"family_id": family_id, "split": "calib", "outcome": outcome, "world": copy.deepcopy(WORLD),
            "request": request,
            "truth": {"person_id": person_id, "role_id": role_id, "manager_id": person["manager_id"]},
            "expected": {"default": expected},
            "wordings": [{"style": 1, "template": 0, "text": wording}]}


# Ada Stone of Finance (p1) is active. Ada Stone of Sales (p2) is not active, holds r2, and r2 is sensitive.
GRANT = hand_family("hf-1", "p1", "r3", "grant",
                    state([["p1", "r1"], ["p1", "r3"], ["p2", "r2"], ["p3", "r2"]], [], [["p1", "r3"]],
                          "granted", "p1", "r3"))
APPROVAL = hand_family("hf-2", "p1", "r2", "approval",
                       state(ACCESS, [["p1", "r2", "p3"]], [], "pending", "p1", "r2"))
ALREADY = hand_family("hf-3", "p1", "r1", "already", state(ACCESS, [], [], "already", "p1", "r1"))
INACTIVE = hand_family("hf-4", "p2", "r2", "inactive", state(ACCESS, [], [], "denied", "p2", "r2"))
HAND_FAMILIES = (GRANT, APPROVAL, ALREADY, INACTIVE)

# What a person who reads the rules writes into the form for each of the four, typed by hand.
RIGHT = {
    "hf-1": {"person": "p1", "resolution": "role given", "role_given": "yes", "asked_to_approve": "none"},
    "hf-2": {"person": "p1", "resolution": "manager asked", "role_given": "no", "asked_to_approve": "p3"},
    "hf-3": {"person": "p1", "resolution": "held before", "role_given": "no", "asked_to_approve": "none"},
    "hf-4": {"person": "p2", "resolution": "refused", "role_given": "no", "asked_to_approve": "none"},
}


def entry(family: dict, ruleset: str = "default", **changed: str) -> dict:
    """One family of a read form: the right answers of a hand family, with `changed` on top."""
    return {"family": family["family_id"], "ruleset": ruleset, **RIGHT[family["family_id"]], **changed}


# ---------------------------------------------------------------------------
# Helpers that do what a person, or a small script of the test kind, does
# ---------------------------------------------------------------------------
def right_answers(family: dict, ruleset: str = "default") -> dict[str, str]:
    """The four answers that agree with the stored expected state of a generated family."""
    expected = family["expected"][ruleset]
    return {"person": expected["request"]["person_id"],
            "resolution": FORM_WORD[expected["request"]["resolution"]],
            "role_given": "yes" if expected["grants"] else "no",
            "asked_to_approve": expected["approvals"][0][2] if expected["approvals"] else "none"}


def fill(form: str, answers: dict[str, dict[str, str]]) -> str:
    """The form with answers typed behind the colons. `answers`: family id -> {field: the text to type}.
    A field that is not named stays empty."""
    lines, family_id = [], None
    for line in form.split("\n"):
        if line.startswith("- family:"):
            family_id = line.split(":", 1)[1].strip().strip('"')
        match = ANSWER_LINE.match(line)
        if match and match.group(1) in answers.get(family_id, {}):
            line = f"{line} {answers[family_id][match.group(1)]}"
        lines.append(line)
    return "\n".join(lines)


def ids_in(form: str) -> list[str]:
    return re.findall(r'^- family: "(.+)"$', form, flags=re.MULTILINE)


def facts_true(family: dict) -> int:
    """How many rule facts are true for the person and the role meant, read from the world here: the
    account is not active, the person holds the role, the role is sensitive."""
    truth, world = family["truth"], family["world"]
    person = next(p for p in world["people"] if p["person_id"] == truth["person_id"])
    role = next(r for r in world["roles"] if r["role_id"] == truth["role_id"])
    holds = [truth["person_id"], truth["role_id"]] in [list(pair) for pair in world["access"]]
    return (person["status"] == "inactive") + holds + bool(role["sensitive"])


def has_pair(family: dict) -> bool:
    names = [person["name"] for person in family["world"]["people"]]
    return names.count(family["request"]["person_name"]) > 1


def two_more_with_mixed_facts(six: list[dict], of_other_outcomes: bool = False) -> bool:
    """True if two families with more than one rule fact true can be set aside so that the four that are
    left are one of each outcome. With `of_other_outcomes` the two must also differ in their outcome."""
    mixed_ones = [family for family in six if facts_true(family) > 1]
    for a, b in itertools.combinations(mixed_ones, 2):
        rest = sorted(family["outcome"] for family in six if family is not a and family is not b)
        if rest == sorted(OUTCOME_WORDS) and (a["outcome"] != b["outcome"] or not of_other_outcomes):
            return True
    return False


def small_family(family_id: str, outcome: str, facts: tuple[bool, bool, bool], pair: bool = False) -> dict:
    """A family with only what the choice reads: one person meant with these facts (active, holds the
    role, sensitive), her manager, and with `pair` a second person of her name."""
    active, holds, sensitive = facts
    people = [{"person_id": "t", "name": "Ada Stone", "department": "Finance",
               "status": "active" if active else "inactive", "manager_id": "m"},
              {"person_id": "m", "name": "Bo Reed", "department": "Finance", "status": "active",
               "manager_id": None}]
    if pair:
        people.append({"person_id": "h", "name": "Ada Stone", "department": "Sales", "status": "active",
                       "manager_id": "m"})
    role = {"role_id": "r", "name": "Wiki Editor", "sensitive": sensitive}
    return {"family_id": family_id, "outcome": outcome,
            "world": {"people": people, "roles": [role], "access": [["t", "r"]] if holds else []},
            "request": {"person_name": "Ada Stone", "department": "Finance", "role_name": "Wiki Editor"},
            "truth": {"person_id": "t", "role_id": "r", "manager_id": "m"}}


PLAIN_FACTS = {"grant": (True, False, False), "approval": (True, False, True), "already": (True, True, False),
               "inactive": (False, False, False)}


def small_split(per_outcome: int = 3, pair: bool = False) -> list[dict]:
    """`per_outcome` families of each outcome in which no two rule facts are true together."""
    return [small_family(f"{outcome}-{i}", outcome, PLAIN_FACTS[outcome], pair)
            for outcome in OUTCOME_WORDS for i in range(per_outcome)]


def form_of(hand_states, families: list[dict], ruleset: str = "default") -> str:
    return hand_states.form_text(families, split="calib", seed=0, ruleset=ruleset,
                                 split_file_sha256="0" * 64, check_command="the check command")


def scenario_folder(tmp_path: Path, families: list[dict], split: str = "calib") -> Path:
    """A scenario folder with one split file and a summary.json that names its SHA-256, as a build writes."""
    folder = tmp_path / "scenarios"
    folder.mkdir()
    save_families(folder / f"{split}.jsonl", families)
    write_summary(folder, split)
    return folder


def write_summary(folder: Path, split: str = "calib") -> None:
    digest = hashlib.sha256((folder / f"{split}.jsonl").read_bytes()).hexdigest()
    (folder / "summary.json").write_text(json.dumps({"files_sha256": {f"{split}.jsonl": digest}}),
                                         encoding="utf-8")


def run(hand_states, capsys, *argv: object) -> tuple[int, str, str]:
    code = hand_states.main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def make_form(hand_states, capsys, folder: Path, out: Path, *more: object) -> str:
    """Run `make` with seed 0 on the calib split of `folder` and return the text of the form."""
    code, _, err = run(hand_states, capsys, "make", "--scenarios", folder, "--split", "calib", "--seed", 0,
                       "--out", out, *more)
    assert code == 0, err
    return out.read_text(encoding="utf-8")


def check_form_file(hand_states, capsys, folder: Path, form_file: Path,
                    form: str) -> tuple[int, list[str], str]:
    """Write `form` to the file, run `check`, and return (exit code, the lines of the output, stderr)."""
    form_file.write_text(form, encoding="utf-8")
    code, out, err = run(hand_states, capsys, "check", "--form", form_file, "--scenarios", folder,
                         "--split", "calib")
    return code, out.splitlines(), err


def verdict_of(lines: list[str], family_id: str) -> str:
    """The text behind the family id on the family's line of the check's output."""
    found = [line for line in lines if line.split()[:1] == [family_id]]
    assert len(found) == 1, lines
    return found[0].split(None, 1)[1]


# ---------------------------------------------------------------------------
# The choice of the six
# ---------------------------------------------------------------------------
def test_the_words_typed_by_hand_are_the_words_of_the_code():
    assert types.OUTCOMES == OUTCOME_WORDS and types.RESOLUTIONS == RESOLUTION_VALUES


@pytest.mark.parametrize("seed", range(25))
def test_the_six_are_one_of_each_outcome_and_two_more_with_mixed_facts(hand_states, mixed, seed):
    six = hand_states.choose_six(mixed, seed)
    assert len(six) == 6 and len({family["family_id"] for family in six}) == 6
    assert all(family in mixed for family in six)
    assert two_more_with_mixed_facts(six)
    # One of the two holds the role and the role is sensitive; in the other the account is not active.
    assert two_more_with_mixed_facts(six, of_other_outcomes=True)


@pytest.mark.parametrize("seed", range(25))
def test_at_least_two_of_the_six_have_two_people_of_one_name(hand_states, mixed, plain, seed):
    # `plain` has only one such family per outcome (4 of 24): a plain draw of six rarely holds two.
    assert sum(has_pair(family) for family in plain) == 4
    for families in (mixed, plain):
        six = hand_states.choose_six(families, seed)
        assert sum(has_pair(family) for family in six) >= 2
        assert hand_states.limits_not_met(six) == ([] if families is mixed else [
            "fewer than 2 families beyond one per outcome have more than one rule fact true: the split "
            "has too few of them"])


def test_the_choice_is_fixed_by_the_seed_and_not_by_the_order_of_the_file(hand_states, mixed):
    ids = [[family["family_id"] for family in hand_states.choose_six(mixed, seed)] for seed in range(10)]
    again = [[family["family_id"] for family in hand_states.choose_six(mixed, seed)] for seed in range(10)]
    backwards = [[family["family_id"] for family in hand_states.choose_six(mixed[::-1], seed)]
                 for seed in range(10)]
    assert ids == again == backwards
    assert len({tuple(sorted(six)) for six in ids}) == 10        # ten seeds, ten different sets of six
    assert hand_states.choose_six(mixed, 3) == hand_states.choose_six(copy.deepcopy(mixed), 3)


def test_the_place_in_the_form_does_not_tell_the_outcome(hand_states, mixed):
    first = {hand_states.choose_six(mixed, seed)[0]["outcome"] for seed in range(40)}
    last = {hand_states.choose_six(mixed, seed)[5]["outcome"] for seed in range(40)}
    assert first == last == set(OUTCOME_WORDS)
    places_of_grant = {[f["outcome"] for f in hand_states.choose_six(mixed, seed)].index("grant")
                       for seed in range(40)}
    assert places_of_grant == set(range(6))


@pytest.mark.parametrize("seed", range(10))
def test_without_mixed_facts_the_two_more_are_of_different_outcomes(hand_states, plain, seed):
    six = hand_states.choose_six(plain, seed)
    assert all(facts_true(family) <= 1 for family in plain)
    assert sorted(sorted(family["outcome"] for family in six).count(o) for o in OUTCOME_WORDS) == [1, 1, 2, 2]


def test_one_family_with_mixed_facts_is_taken_and_the_other_place_gets_another_outcome(hand_states):
    only = small_family("mixed-0", "already", (True, True, True))
    for seed in range(10):
        six = hand_states.choose_six(small_split() + [only], seed)
        outcomes = [family["outcome"] for family in six]
        assert only in six                     # in the place of its outcome, or as one of the two more
        assert sorted(outcomes.count(o) for o in OUTCOME_WORDS) == [1, 1, 2, 2]


def test_a_split_without_same_name_pairs_still_gives_six_and_says_so(hand_states):
    six = hand_states.choose_six(small_split(), 0)
    assert len(six) == 6
    assert hand_states.limits_not_met(six)[-1] == ("only 0 of the 6 families have two people with one full "
                                                   "name: the split has no more that fit")
    # One family with a pair in the split: it is always brought in, and the note names the count.
    with_pair = small_family("grant-pair", "grant", PLAIN_FACTS["grant"], pair=True)
    for seed in range(10):
        six = hand_states.choose_six(small_split() + [with_pair], seed)
        assert with_pair in six and sorted(f["outcome"] for f in six).count("grant") <= 2
        assert hand_states.limits_not_met(six)[-1].startswith("only 1 of the 6 families have two people")
    assert all(has_pair(f) for f in hand_states.choose_six(small_split(pair=True), 0))


def test_a_split_that_is_too_small_is_refused(hand_states):
    with pytest.raises(ValueError, match="fewer than 6 families"):
        hand_states.choose_six(small_split(per_outcome=1), 0)
    without_grant = [family for family in small_split() if family["outcome"] != "grant"]
    with pytest.raises(ValueError, match="no family with the outcome 'grant'"):
        hand_states.choose_six(without_grant, 0)
    with pytest.raises(ValueError, match="twice"):
        hand_states.choose_six(small_split() + small_split(), 0)


# ---------------------------------------------------------------------------
# What the form shows, and what it does not show
# ---------------------------------------------------------------------------
def test_the_form_shows_the_request_the_people_of_that_name_and_the_role(hand_states):
    form = form_of(hand_states, [INACTIVE, GRANT])
    # Typed by hand from WORLD: both people called Ada Stone, in the order of the world.
    assert """\
- family: "hf-4"
  ruleset: default
  # The request:
  #   Please give Ada Stone from Sales access to the Payroll Editor role.
  #
  # People with the full name "Ada Stone":
  #   person id   department   account      manager id   holds the role now
  #   p1          Finance      active       p3           no
  #   p2          Sales        not active   p4           yes
  #
  # The role of the request:
  #   role id   name             sensitive
  #   r2        Payroll Editor   yes
  #
  # Your answers (the words to use are at the top of the form):
  person:
  resolution:
  role_given:
  asked_to_approve:
""" in form
    assert """\
  #   Please give Ada Stone from Finance access to the Help Desk Agent role.
  #
  # People with the full name "Ada Stone":
  #   person id   department   account      manager id   holds the role now
  #   p1          Finance      active       p3           no
  #   p2          Sales        not active   p4           no
  #
  # The role of the request:
  #   role id   name              sensitive
  #   r3        Help Desk Agent   no
""" in form
    assert "Request 1 of 6" in form and "Request 2 of 6" in form
    # A person without a manager, and only one person of the name: one row.
    about_bo = hand_family("hf-5", "p3", "r1", "grant", state([], [], [], "granted", "p3", "r1"))
    assert "  #   p3          Finance      active    none         no\n" in form_of(hand_states, [about_bo])


def test_the_form_reads_only_the_request_and_the_world_of_a_family(hand_states, mixed):
    """The strongest form of "no leak": the text is the same when the stored answer is another one, and
    when no answer is stored at all."""
    answer_keys = ("outcome", "homonym", "facts", "truth", "expected", "turn_budget")
    for family, other in zip(mixed, mixed[1:] + mixed[:1]):
        shown = form_of(hand_states, [family])
        bare = {key: family[key] for key in ("family_id", "wordings", "request", "world")}
        with_another_answer = {**family, **{key: other[key] for key in answer_keys}}
        assert set(family) == set(bare) | set(answer_keys) | {"split", "world_seed"}
        assert form_of(hand_states, [bare]) == shown
        assert form_of(hand_states, [with_another_answer]) == shown


@pytest.mark.parametrize("family", HAND_FAMILIES, ids=OUTCOME_WORDS)
def test_no_outcome_word_and_no_resolution_value_stands_in_the_form(hand_states, family):
    form = form_of(hand_states, [family]).casefold()
    assert family["outcome"] not in form
    assert family["expected"]["default"]["request"]["resolution"] not in form
    # More than that: the form holds none of these words of the code, for any family of the four.
    whole = form_of(hand_states, list(HAND_FAMILIES)).casefold()
    assert [word for word in OUTCOME_WORDS + RESOLUTION_VALUES if word in whole] == []
    assert "expected" not in whole and "truth" not in whole and "outcome" not in whole


def test_the_top_of_the_form_is_the_same_text_for_any_families(hand_states, mixed):
    tops = {form_of(hand_states, hand_states.choose_six(mixed, seed)).split("\nfamilies:\n")[0]
            for seed in range(5)}
    assert len(tops) == 1 and tops == {form_of(hand_states, [GRANT]).split("\nfamilies:\n")[0]}
    top = tops.pop()
    assert top.count("\n#   1. ") == 2 and top.count("\n#   5. ") == 1      # five sentences, then the rules
    assert "\n#      the check command\n" in top
    assert "role given | manager asked | held before | refused" in top


# ---------------------------------------------------------------------------
# The rules in the form
# ---------------------------------------------------------------------------
# Typed by hand from PLAN.md 5.1: "inactive account, close as denied; already has access, close as such;
# sensitive role, request approval from the person's manager and close as pending; otherwise grant and close
# as granted." The first rule that applies wins.
NOT_ACTIVE = ["The account of the person is not active.",
              "The desk does not give the role. It asks nobody. Resolution: refused."]
HELD = ["The person holds the role now.",
        "The desk does not give the role. It asks nobody. Resolution: held before."]
ASK = ("The desk does not give the role. It asks the manager of the person to approve. "
       "Resolution: manager asked.")
GIVE = "The desk gives the role. It asks nobody. Resolution: role given."
RULES_BY_HAND = {
    "default": [NOT_ACTIVE, HELD, ["The role is sensitive.", ASK], ["In every other case.", GIVE]],
    "all_grants_need_approval": [NOT_ACTIVE, HELD, ["In every other case.", ASK]],
    "sensitive_denied": [NOT_ACTIVE, HELD, ["The role is sensitive.", NOT_ACTIVE[1]],
                         ["In every other case.", GIVE]],
}


def numbered(rules: list[list[str]]) -> list[str]:
    """The rule lines as the form prints them: "  1. condition", then the effect on the next line."""
    return [line for number, (condition, effect) in enumerate(rules, start=1)
            for line in (f"  {number}. {condition}", f"       {effect}")]


@pytest.mark.parametrize("ruleset", RULESETS)
def test_the_rules_in_the_form_are_the_rules_of_the_plan_in_their_order(hand_states, ruleset):
    assert sorted(RULES_BY_HAND) == sorted(RULESETS)
    assert hand_states.rule_lines(ruleset) == numbered(RULES_BY_HAND[ruleset])
    form = form_of(hand_states, [GRANT], ruleset)
    assert "".join(f"# {line}\n" for line in numbered(RULES_BY_HAND[ruleset])) in form
    assert f"# The rules of the desk (rule set: {ruleset})\n" in form


def test_the_rule_list_follows_rules_py_when_the_rules_change(hand_states, monkeypatch):
    def sensitive_before_held(active, has_access, sensitive, ruleset="default"):
        if not active:
            return "deny"
        if sensitive:
            return "approval"
        return "already" if has_access else "grant"

    monkeypatch.setattr("accessdesk.env.rules.decide", sensitive_before_held)
    changed = [NOT_ACTIVE, ["The role is sensitive.", ASK], HELD, ["In every other case.", GIVE]]
    assert hand_states.rule_lines("default") == numbered(changed)


def test_rules_that_the_form_has_no_words_for_stop_it(hand_states, monkeypatch):
    def two_facts_at_once(active, has_access, sensitive, ruleset="default"):
        return "grant" if has_access == sensitive else "deny"

    monkeypatch.setattr("accessdesk.env.rules.decide", two_facts_at_once)
    with pytest.raises(ValueError, match="cannot be written as a list"):
        form_of(hand_states, [GRANT])
    monkeypatch.undo()

    def approver_is_the_person(family, ruleset="default"):
        ids = {"person_id": "person", "role_id": "role"}
        return [("request_approval", {**ids, "approver_id": "person"}),
                ("close_request", {"resolution": "pending", **ids})]

    monkeypatch.setattr(hand_states, "reference_plan", approver_is_the_person)
    with pytest.raises(ValueError, match="no words for the plan"):
        form_of(hand_states, [GRANT])


# ---------------------------------------------------------------------------
# The check, on the four families typed by hand
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", HAND_FAMILIES, ids=OUTCOME_WORDS)
def test_right_answers_match(hand_states, text, family):
    verdict = hand_states.check_family(entry(family), family, text)
    assert verdict == (family["family_id"], "MATCH", "")


WRONG = [
    # (family, the answers that are changed, what the check must name)
    ("the other person of the name", GRANT, {"person": "p2"}, "the person"),
    ("a person of another name", ALREADY, {"person": "p3"}, "the person"),
    ("another resolution", APPROVAL, {"resolution": "refused"}, "the resolution"),
    ("role given, but no grant", GRANT, {"role_given": "no"}, "the grant"),
    ("a grant of a role that is held", ALREADY, {"role_given": "yes"}, "the grant"),
    ("a grant to an account that is not active", INACTIVE, {"role_given": "yes"}, "the grant"),
    ("nobody asked", APPROVAL, {"asked_to_approve": "none"}, "the approval"),
    ("the wrong manager asked", APPROVAL, {"asked_to_approve": "p4"}, "the approval"),
    ("a manager asked for nothing", GRANT, {"asked_to_approve": "p3"}, "the approval"),
    ("a grant and no approval for a sensitive role", APPROVAL,
     {"resolution": "role given", "role_given": "yes", "asked_to_approve": "none"},
     "the resolution, the grant, the approval"),
    ("the rules applied to the other person", INACTIVE,
     {"person": "p1", "resolution": "manager asked", "asked_to_approve": "p3"},
     "the person, the resolution, the approval"),
]


@pytest.mark.parametrize("label, family, changed, named", WRONG, ids=[case[0] for case in WRONG])
def test_each_kind_of_wrong_answer_is_found_and_named(hand_states, text, label, family, changed, named):
    verdict = hand_states.check_family(entry(family, **changed), family, text)
    assert verdict == (family["family_id"], "DIFFERS", named)


@pytest.mark.parametrize("field", ["person", "resolution", "role_given", "asked_to_approve"])
def test_an_empty_answer_is_not_answered_and_never_a_match(hand_states, text, field):
    for family in HAND_FAMILIES:
        verdict = hand_states.check_family(entry(family, **{field: ""}), family, text)
        assert verdict == (family["family_id"], "NOT ANSWERED", field)
    nothing = {key: "" for key in RIGHT["hf-3"]}
    assert hand_states.check_family(entry(ALREADY, **nothing), ALREADY, text).detail == \
        "person, resolution, role_given, asked_to_approve"


def test_an_answer_that_cannot_be_read_is_named_and_never_a_match(hand_states, text):
    cases = [({"person": "P1"}, "person: no person of this family has the id 'P1'"),
             ({"resolution": "given"}, "resolution: 'given' is not one of ['role given', 'manager asked', "
                                       "'held before', 'refused']"),
             ({"role_given": "maybe"}, "role_given: 'maybe' is not yes or no"),
             ({"asked_to_approve": "the manager"},
              "asked_to_approve: 'the manager' is not none and not a person id of this family")]
    for changed, message in cases:
        verdict = hand_states.check_family(entry(GRANT, **changed), GRANT, text)
        assert verdict == ("hf-1", "CANNOT BE READ", message)
    both = hand_states.check_family(entry(GRANT, role_given="y", person="x"), GRANT, text)
    assert both.status == "CANNOT BE READ" and "person:" in both.detail and "role_given:" in both.detail


def test_a_wrong_stored_state_is_found_by_right_answers(hand_states, text):
    """What the check is for: the person is right, and the state that the generator stored is not."""
    def with_expected(family: dict, **parts) -> dict:
        broken = copy.deepcopy(family)
        request = {**broken["expected"]["default"]["request"], **parts.pop("request", {})}
        broken["expected"]["default"] = {**broken["expected"]["default"], **parts, "request": request}
        return broken

    cases = [(with_expected(GRANT, grants=[]), "the grant"),                   # the grant record forgotten
             (with_expected(ALREADY, grants=[["p1", "r1"]]), "the grant"),     # a needless grant stored
             (with_expected(APPROVAL, approvals=[["p1", "r2", "p4"]]), "the approval"),
             (with_expected(APPROVAL, approvals=[]), "the approval"),
             (with_expected(INACTIVE, request={"resolution": "already"}), "the resolution"),
             (with_expected(INACTIVE, request={"person_id": "p1"}), "the person"),
             # Parts that no answer sets: the access list, the role, the pair in a record.
             (with_expected(GRANT, access=ACCESS), hand_states.OTHER_PART),
             (with_expected(ALREADY, access=ACCESS + [["p4", "r1"]]), hand_states.OTHER_PART),
             (with_expected(ALREADY, request={"role_id": "r3"}), hand_states.OTHER_PART),
             (with_expected(GRANT, grants=[["p1", "r1"]]), hand_states.OTHER_PART)]
    for broken, named in cases:
        verdict = hand_states.check_family(entry(broken), broken, text)
        assert verdict == (broken["family_id"], "DIFFERS", named)


def test_the_verdict_is_the_verdict_of_the_verifier(hand_states, text, monkeypatch):
    monkeypatch.setattr(hand_states.impl, "get", lambda name: {"verify": lambda final, expected: False}[name])
    verdict = hand_states.check_family(entry(GRANT), GRANT, text)
    assert verdict == ("hf-1", "DIFFERS", hand_states.OTHER_PART)       # no answer differs, and it says so
    monkeypatch.setattr(hand_states.impl, "get", lambda name: {"verify": lambda final, expected: True}[name])
    assert hand_states.check_family(entry(GRANT, role_given="no"), GRANT, text).status == "MATCH"


def test_the_second_version_of_the_verifier_gives_the_same_verdicts(hand_states, text, monkeypatch):
    skip_unless_second_is_written("verify")
    monkeypatch.setenv("ACCESSDESK_IMPL", "second")
    assert all(hand_states.check_family(entry(family), family, text).status == "MATCH"
               for family in HAND_FAMILIES)
    assert [hand_states.check_family(entry(family, **changed), family, text).detail
            for _, family, changed, _ in WRONG] == [named for *_, named in WRONG]


def test_the_state_is_built_by_the_tools_from_the_answers_alone(hand_states, text):
    answers = {"person_id": "p2", "resolution": "granted", "gives": True, "approver_id": "p4"}
    # Wrong on purpose: an account that is not active gets a sensitive role, and a manager is asked too.
    assert hand_states.build_state(GRANT, answers, text) == state(
        [["p1", "r1"], ["p2", "r2"], ["p2", "r3"], ["p3", "r2"]], [["p2", "r3", "p4"]], [["p2", "r3"]],
        "granted", "p2", "r3")
    assert GRANT["world"] == WORLD                                      # the family is not changed


# ---------------------------------------------------------------------------
# The check, on generated families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ruleset", RULESETS)
def test_right_answers_match_in_every_generated_family(hand_states, text, mixed, plain, ruleset):
    for family in mixed + plain:
        answers = {"family": family["family_id"], "ruleset": ruleset, **right_answers(family, ruleset)}
        assert hand_states.check_family(answers, family, text).status == "MATCH"


def test_the_check_calls_no_rules_code_of_the_generator(hand_states, text, mixed, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the check must not ask the generator's rules for the answer")

    for name in ("decide", "action_for", "expected_state", "reference_plan", "turn_budget"):
        monkeypatch.setattr(f"accessdesk.env.rules.{name}", forbidden)
    monkeypatch.setattr(hand_states, "reference_plan", forbidden)
    six = hand_states.choose_six(mixed, 0)
    form = {"seed": 0, "ruleset": "default",
            "families": [{"family": family["family_id"], "ruleset": "default", **right_answers(family)}
                         for family in six]}
    verdicts = hand_states.check_form(form, mixed, text)
    assert [verdict.status for verdict in verdicts] == ["MATCH"] * 6
    assert hand_states.last_line(verdicts) == "6 of 6 match"


def test_a_form_with_other_families_than_the_seed_chooses_is_refused(hand_states, text, mixed):
    six = hand_states.choose_six(mixed, 0)
    easy = next(family for family in mixed if family not in six)
    entries = [{"family": family["family_id"], "ruleset": "default", **right_answers(family)}
               for family in six[:5] + [easy]]
    with pytest.raises(hand_states.FormError, match="not the six that the seed 0 chooses"):
        hand_states.check_form({"seed": 0, "ruleset": "default", "families": entries}, mixed, text)
    entries[5]["family"] = "ca-99"
    with pytest.raises(hand_states.FormError, match="no family 'ca-99'"):
        hand_states.check_form({"seed": 0, "ruleset": "default", "families": entries}, mixed, text)
    entries = [{"family": family["family_id"], "ruleset": "no_rules", **right_answers(family)}
               for family in six]
    with pytest.raises(hand_states.FormError, match="unknown rule set 'no_rules'"):
        hand_states.check_form({"seed": 0, "ruleset": "no_rules", "families": entries}, mixed, text)


# ---------------------------------------------------------------------------
# make and check as commands, with files
# ---------------------------------------------------------------------------
def test_the_filled_form_of_six_matches(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "forms" / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    six = [family for family_id in ids_in(form) for family in mixed if family["family_id"] == family_id]
    assert [family["family_id"] for family in six] == \
        [family["family_id"] for family in hand_states.choose_six(mixed, 0)]

    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, form)       # nothing filled in
    assert code == 1 and err == "" and lines[-1].startswith("0 of 6 match; open: ")
    assert all(verdict_of(lines, family["family_id"]) ==
               "NOT ANSWERED: person, resolution, role_given, asked_to_approve" for family in six)

    filled = fill(form, {family["family_id"]: right_answers(family) for family in six})
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, filled)
    assert code == 0 and err == "" and lines[-1] == "6 of 6 match"
    assert lines[-7:-1] == [f"{family['family_id']:8s} MATCH" for family in six]
    digest = hashlib.sha256((folder / "calib.jsonl").read_bytes()).hexdigest()
    assert lines[1] == f"split file {folder / 'calib.jsonl'}, SHA-256 {digest}"
    assert lines[2] == f"verifier: the {impl.which('verify')} version"


def one_wrong(family: dict, kind: str) -> dict[str, str]:
    """The right answers of a generated family with one answer of the given kind changed."""
    answers = right_answers(family)
    if kind == "person":
        answers["person"] = next(p["person_id"] for p in family["world"]["people"]
                                 if p["name"] == family["request"]["person_name"]
                                 and p["person_id"] != answers["person"])
    elif kind == "resolution":
        answers["resolution"] = "refused" if answers["resolution"] != "refused" else "held before"
    elif kind == "grant":
        answers["role_given"] = "no" if answers["role_given"] == "yes" else "yes"
    else:
        answers["asked_to_approve"] = answers["person"] if answers["asked_to_approve"] == "none" else "none"
    return answers


@pytest.mark.parametrize("kind", ["person", "resolution", "grant", "approval"])
def test_one_wrong_answer_in_the_file_is_found_named_and_gives_exit_code_1(hand_states, capsys, tmp_path,
                                                                           mixed, kind):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    six = [family for family_id in ids_in(form) for family in mixed if family["family_id"] == family_id]
    for family in six:
        if kind == "person" and not has_pair(family):
            continue
        answers = {other["family_id"]: right_answers(other) for other in six}
        answers[family["family_id"]] = one_wrong(family, kind)
        code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, fill(form, answers))
        assert code == 1
        assert verdict_of(lines, family["family_id"]) == f"DIFFERS: the {kind}"
        assert lines[-1] == f"5 of 6 match; open: {family['family_id']} (differs)"
        assert sum(line.endswith(" MATCH") for line in lines) == 5


def test_an_empty_answer_in_the_file_is_reported_and_gives_exit_code_1(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    six = [family for family_id in ids_in(form) for family in mixed if family["family_id"] == family_id]
    answers = {family["family_id"]: right_answers(family) for family in six}
    del answers[six[2]["family_id"]]["asked_to_approve"]
    answers[six[4]["family_id"]]["role_given"] = "perhaps"
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, fill(form, answers))
    assert code == 1
    assert verdict_of(lines, six[2]["family_id"]) == "NOT ANSWERED: asked_to_approve"
    assert verdict_of(lines, six[4]["family_id"]) == "CANNOT BE READ: role_given: 'perhaps' is not yes or no"
    assert lines[-1] == (f"4 of 6 match; open: {six[2]['family_id']} (not answered), "
                         f"{six[4]['family_id']} (cannot be read)")


def test_the_form_loads_again_after_a_person_edits_it(hand_states, capsys, tmp_path, mixed):
    """Extra spaces, capital letters, quotes, a comment behind an answer, and the line ends and the byte
    order mark of another editor."""
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    six = [family for family_id in ids_in(form) for family in mixed if family["family_id"] == family_id]
    curly = "\u201c{}\u201d"                             # the quote signs that a word processor types
    spellings = {"yes": ["Yes", '"yes"', "  YES   "], "no": ["No", "'no'", "NO   # sure"],
                 "none": ["None", '"none"', "   NONE"],
                 "role given": ["Role Given", "role   given", '"role_given"'],
                 "manager asked": ["Manager asked", "'manager asked'", "manager-asked"],
                 "held before": ["HELD BEFORE", curly.format("held before"), '" held before "'],
                 "refused": ["Refused", curly.format("refused"), "refused # rule 1"]}
    for variant in range(3):
        answers = {}
        for family in six:
            right = right_answers(family)
            answers[family["family_id"]] = {
                "person": [right["person"], f'"{right["person"]}"', f"   {right['person']}   "][variant],
                "resolution": spellings[right["resolution"]][variant],
                "role_given": spellings[right["role_given"]][variant],
                "asked_to_approve": spellings.get(right["asked_to_approve"],
                                                  [right["asked_to_approve"]] * 3)[variant]}
        edited = fill(form, answers)
        if variant == 2:                 # saved by an editor that writes a byte order mark and CR LF
            edited = chr(0xFEFF) + edited.replace("\n", "\r\n")
        code, lines, err = check_form_file(hand_states, capsys, folder, form_file, edited)
        assert (code, err, lines[-1]) == (0, "", "6 of 6 match")
    read = hand_states.read_form(fill(form, {six[0]["family_id"]: {"role_given": "Yes", "person": '"x y"'}}))
    assert read["families"][0] == {"family": six[0]["family_id"], "ruleset": "default", "person": "x y",
                                   "resolution": "", "role_given": "Yes", "asked_to_approve": ""}
    assert read["seed"] == 0 and read["split"] == "calib" and len(read["families"]) == 6


def test_a_broken_form_stops_the_check_with_a_plain_message(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    first = ids_in(form)[0]
    without_the_last = form[:form.rindex("# ---", 0, form.rindex("- family:"))]
    broken = {
        "no space after the colon": (form.replace("  person:", "  person:p1", 1), "cannot be read as YAML"),
        "an answer line twice": (form.replace("  person:\n", "  person: a\n  person: b\n", 1),
                                 "the key 'person' is written twice"),
        "an answer line removed": (form.replace("  role_given:\n", "", 1), "missing ['role_given']"),
        "a new line": (form.replace("  person:\n", "  person:\n  note: hard\n", 1), "not known ['note']"),
        "a family removed": (without_the_last, "must hold 6 families"),
        "a family twice": (form.replace(f'"{ids_in(form)[1]}"', f'"{first}"'), "is in the form twice"),
        "another file": ("just: text\n", "not a form of scripts/hand_states.py"),
        "a list as an answer": (form.replace("  person:\n", "  person: [p1, p2]\n", 1),
                                "a list or a mapping"),
        "another seed": (form.replace('seed: "0"', 'seed: "1"'), "not the six that the seed 1 chooses"),
        "no seed": (form.replace('seed: "0"', 'seed: "soon"'), "must be a whole number"),
        "another split": (form.replace("split: calib", "split: dev"), "made from the split 'dev'"),
    }
    for label, (text_of_form, message) in broken.items():
        code, lines, err = check_form_file(hand_states, capsys, folder, form_file, text_of_form)
        assert (code, lines) == (1, DID_NOT_RUN), label          # no family gets a verdict
        assert err.startswith("the check did not run: ") and message in err, (label, err)
    code, out, err = run(hand_states, capsys, "check", "--form", tmp_path / "none.yaml", "--scenarios",
                         folder, "--split", "calib")
    assert (code, out.splitlines()) == (1, DID_NOT_RUN) and "none.yaml is missing" in err


def test_the_form_belongs_to_the_split_file_it_was_made_from(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    digest = hashlib.sha256((folder / "calib.jsonl").read_bytes()).hexdigest()
    assert f'\nsplit_file_sha256: "{digest}"\n' in form
    # The folder is named in the check command at the top. It is no data line: the check never read it.
    assert f" --scenarios {folder} --split calib\n" in form and "\nscenarios:" not in form
    filled = fill(form, {family["family_id"]: right_answers(family) for family in mixed})
    assert check_form_file(hand_states, capsys, folder, form_file, filled)[0] == 0

    # The split file is changed after the form was made: first without the summary, then with a new one.
    changed = copy.deepcopy(mixed)
    changed[0]["world_seed"] += 1
    save_families(folder / "calib.jsonl", changed)
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, filled)
    assert (code, lines) == (1, DID_NOT_RUN) and "is not the file that the build wrote" in err
    write_summary(folder)
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, filled)
    assert (code, lines) == (1, DID_NOT_RUN) and "the form was made from another file" in err
    (folder / "summary.json").unlink()
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, filled)
    assert (code, lines) == (1, DID_NOT_RUN) and "summary.json is missing" in err


def test_make_never_writes_over_a_form_and_refuses_what_it_cannot_use(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    filled = fill(form, {family["family_id"]: right_answers(family) for family in mixed})
    form_file.write_text(filled, encoding="utf-8")
    common = ("make", "--scenarios", folder, "--seed", 0)
    code, out, err = run(hand_states, capsys, *common, "--split", "calib", "--out", form_file)
    assert (code, out) == (2, "") and "may hold answers" in err
    assert form_file.read_text(encoding="utf-8") == filled
    # The same seed gives the same bytes; another seed another form.
    again = make_form(hand_states, capsys, folder, tmp_path / "again.yaml")
    assert again.replace("again.yaml", "form.yaml") == form
    other_seed = tmp_path / "seed1.yaml"
    code, _, _ = run(hand_states, capsys, *common[:3], "--seed", 1, "--split", "calib", "--out", other_seed)
    assert code == 0 and ids_in(other_seed.read_text(encoding="utf-8")) != ids_in(form)

    code, out, err = run(hand_states, capsys, *common, "--split", "dev", "--out", tmp_path / "dev.yaml")
    assert (code, out) == (2, "") and "dev.jsonl is missing" in err
    code, out, err = run(hand_states, capsys, *common, "--split", "test", "--out", tmp_path / "test.yaml")
    assert (code, out) == (2, "") and "the test set is looked at once" in err
    save_families(folder / "calib.jsonl", mixed[:5])
    write_summary(folder)
    code, out, err = run(hand_states, capsys, *common, "--split", "calib", "--out", tmp_path / "small.yaml")
    assert (code, out) == (2, "") and not (tmp_path / "small.yaml").exists()
    with pytest.raises(SystemExit) as stop:
        hand_states.main(["make", "--scenarios", str(folder), "--split", "calib", "--out", "x"])    # no seed
    assert stop.value.code == 2
    capsys.readouterr()


def test_the_test_split_needs_a_flag_and_is_then_checked_like_any_other(hand_states, capsys, tmp_path, mixed):
    folder = scenario_folder(tmp_path, mixed, split="test")
    form_file = tmp_path / "test_form.yaml"
    code, out, _ = run(hand_states, capsys, "make", "--scenarios", folder, "--split", "test", "--seed", 4,
                       "--out", form_file, "--allow-test")
    assert code == 0 and out.startswith(f"wrote {form_file}: 6 families of the split test, seed 4")
    filled = fill(form_file.read_text(encoding="utf-8"),
                  {family["family_id"]: right_answers(family) for family in mixed})
    form_file.write_text(filled, encoding="utf-8")
    code, out, _ = run(hand_states, capsys, "check", "--form", form_file, "--scenarios", folder,
                       "--split", "test")
    assert code == 0 and out.splitlines()[-1] == "6 of 6 match"


def test_a_form_for_the_rule_set_of_the_policy_shift(hand_states, capsys, tmp_path, mixed):
    """Under all_grants_need_approval the desk never gives a role: the manager is asked."""
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "shift.yaml"
    form = make_form(hand_states, capsys, folder, form_file, "--ruleset", "all_grants_need_approval")
    six = [family for family_id in ids_in(form) for family in mixed if family["family_id"] == family_id]
    assert form.count("\n  ruleset: all_grants_need_approval\n") == 6
    by_new_rule = {family["family_id"]: right_answers(family, "all_grants_need_approval") for family in six}
    assert check_form_file(hand_states, capsys, folder, form_file, fill(form, by_new_rule))[0] == 0
    assert all(answers["role_given"] == "no" for answers in by_new_rule.values())
    # The answers of the default rules are wrong here for the family whose role would be given.
    by_old_rule = {family["family_id"]: right_answers(family) for family in six}
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, fill(form, by_old_rule))
    given = [family["family_id"] for family in six if family["outcome"] == "grant"]
    assert code == 1 and len(given) == 1
    assert verdict_of(lines, given[0]) == "DIFFERS: the resolution, the grant, the approval"
    assert lines[-1] == f"5 of 6 match; open: {given[0]} (differs)"


def test_the_script_starts_as_a_program_and_without_its_docstring(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-OO", str(ROOT / "scripts" / "hand_states.py"), "check",
                             "--form", str(tmp_path / "none.yaml"), "--scenarios", str(tmp_path),
                             "--split", "calib"], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1 and result.stderr.startswith("the check did not run: ")
    assert "Traceback" not in result.stderr and result.stdout.splitlines() == DID_NOT_RUN


# ---------------------------------------------------------------------------
# One rule set per form (a review found that one changed line turned a wrong answer into a match)
# ---------------------------------------------------------------------------
def six_of(form: str, families: list[dict]) -> list[dict]:
    """The families of the form, in the order of the form."""
    return [family for family_id in ids_in(form) for family in families if family["family_id"] == family_id]


def test_a_changed_rule_set_line_of_one_family_stops_the_check(hand_states, capsys, tmp_path, mixed):
    """A sensitive role for an active person who does not hold it: under the default rules the manager is
    asked. "refused, nobody asked" is wrong. It is the right answer of the rule set sensitive_denied, so
    the person changes the rule set line of that one family. That must not give a match."""
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    six = six_of(form, mixed)
    asked = next(family for family in six if family["outcome"] == "approval")
    answers = {family["family_id"]: right_answers(family) for family in six}
    answers[asked["family_id"]] = right_answers(asked, "sensitive_denied")
    assert answers[asked["family_id"]]["resolution"] == "refused"
    wrong = fill(form, answers)
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, wrong)
    assert code == 1 and verdict_of(lines, asked["family_id"]) == "DIFFERS: the resolution, the approval"

    line_of_family = f'- family: "{asked["family_id"]}"\n  ruleset: '
    edited = wrong.replace(line_of_family + "default\n", line_of_family + "sensitive_denied\n")
    assert edited != wrong and edited.count("ruleset: sensitive_denied") == 1
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, edited)
    assert (code, lines) == (1, DID_NOT_RUN)                # no family gets a verdict, least of all MATCH
    assert err.startswith("the check did not run: ")
    assert asked["family_id"] in err and "sensitive_denied" in err and "default" in err


def test_the_output_names_the_rule_set_of_the_form(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    six = six_of(form, mixed)
    assert form.count("\nruleset: default\n") == 1          # once at the top, as data
    filled = fill(form, {family["family_id"]: right_answers(family) for family in six})
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, filled)
    assert code == 0 and lines[0] == f"form {form_file}: seed 0, split calib, rule set default"

    # Every rule set line is changed, the one at the top too. Then it is a form of that rule set: the
    # check runs, and the first line of its output says which rule set it was.
    other = form.replace("ruleset: default\n", "ruleset: sensitive_denied\n")
    assert other.count("ruleset: sensitive_denied\n") == 7
    by_other = fill(other, {family["family_id"]: right_answers(family, "sensitive_denied") for family in six})
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, by_other)
    assert code == 0 and lines[0] == f"form {form_file}: seed 0, split calib, rule set sensitive_denied"
    # The top line alone is changed: the six family lines do not agree with it.
    top_only = filled.replace("\nruleset: default\n", "\nruleset: sensitive_denied\n")
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, top_only)
    assert (code, lines) == (1, DID_NOT_RUN) and "sensitive_denied" in err


def test_check_form_takes_one_rule_set_for_all_six(hand_states, text, mixed):
    six = hand_states.choose_six(mixed, 0)
    entries = [{"family": family["family_id"], "ruleset": "default", **right_answers(family)}
               for family in six]
    form = {"seed": 0, "ruleset": "default", "families": entries}
    assert [verdict.status for verdict in hand_states.check_form(form, mixed, text)] == ["MATCH"] * 6
    for place in range(6):
        changed = [{**entry, "ruleset": "sensitive_denied"} if i == place else entry
                   for i, entry in enumerate(entries)]
        with pytest.raises(hand_states.FormError, match="sensitive_denied"):
            hand_states.check_form({**form, "families": changed}, mixed, text)
    # A family that stores no end state for the rule set of the form: a plain message, no traceback.
    without = copy.deepcopy(mixed)
    stored = next(family for family in without if family["family_id"] == six[3]["family_id"])
    del stored["expected"]["all_grants_need_approval"]
    shift = {"seed": 0, "ruleset": "all_grants_need_approval",
             "families": [{**entry, "ruleset": "all_grants_need_approval"} for entry in entries]}
    with pytest.raises(hand_states.FormError, match=f"{six[3]['family_id']} stores no end state"):
        hand_states.check_form(shift, without, text)


def test_a_form_of_the_first_layout_is_refused_and_must_be_made_again(hand_states, capsys, tmp_path, mixed):
    """The first layout had no rule set line at the top (and a folder line that the check never read)."""
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    filled = fill(form, {family["family_id"]: right_answers(family) for family in mixed})
    first_layout = filled.replace('\nversion: "2"\n', '\nversion: "1"\nscenarios: "some/folder"\n') \
                         .replace("\nruleset: default\nfamilies:\n", "\nfamilies:\n")
    assert first_layout.count("ruleset: default") == 6 and '\nversion: "1"\n' in first_layout
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, first_layout)
    assert (code, lines) == (1, DID_NOT_RUN) and "make the form again" in err
    # The new layout without its rule set line at the top is refused as well.
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file,
                                       filled.replace("\nruleset: default\nfamilies:\n", "\nfamilies:\n"))
    assert (code, lines) == (1, DID_NOT_RUN) and "missing ['ruleset']" in err


# ---------------------------------------------------------------------------
# The log of the check runs
# ---------------------------------------------------------------------------
RUN_START = re.compile(r"=== run (\d+) at \d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC, "
                       r"SHA-256 of the form ([0-9a-f]{64})")


def test_each_run_of_the_check_is_numbered_and_kept_in_a_log_beside_the_form(hand_states, capsys, tmp_path,
                                                                             mixed):
    """A person can change an answer until the check says MATCH. The log shows it: two runs, and the form
    of the second run is not the form of the first."""
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    log = tmp_path / "form.yaml.checks.log"
    form = make_form(hand_states, capsys, folder, form_file)
    six = six_of(form, mixed)
    assert not log.exists()                                             # `make` starts no log
    answers = {family["family_id"]: right_answers(family) for family in six}
    wrong = fill(form, {**answers, six[0]["family_id"]: one_wrong(six[0], "grant")})
    right = fill(form, answers)

    code, first, _ = check_form_file(hand_states, capsys, folder, form_file, wrong)
    assert code == 1 and first[3] == (f"run 1 of this form. Each run is added to {log}; the first run is "
                                      "the one that counts.")
    code, second, _ = check_form_file(hand_states, capsys, folder, form_file, right)
    assert code == 0 and second[-1] == "6 of 6 match"
    assert second[3].startswith("run 2 of this form. ")

    blocks = log.read_text(encoding="utf-8").split("\n\n")              # a blank line ends each run
    assert len(blocks) == 3 and blocks[2] == ""
    for block, printed, checked, number in zip(blocks, (first, second), (wrong, right), ("1", "2")):
        start, *lines = block.split("\n")
        digest = hashlib.sha256(checked.encode("utf-8")).hexdigest()
        assert RUN_START.fullmatch(start).groups() == (number, digest)
        assert lines == printed[:3] + printed[4:]                       # what the check printed
    assert f"{six[0]['family_id']:8s} DIFFERS: the grant" in blocks[0].split("\n")
    assert blocks[1].endswith("\n6 of 6 match")

    # A check that could not run gives no verdict, so it is no run of the log.
    before = log.read_bytes()
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, "just: text\n")
    assert (code, lines) == (1, DID_NOT_RUN) and log.read_bytes() == before


# The line that the check prints in place of the run line, typed by hand.
NO_RUN_LINE = ("this is no run of the form: no family has all four answers, so no state was built and no "
               "answer was compared. Nothing is added to the log.")


def test_a_check_before_any_family_is_answered_in_full_is_no_run(hand_states, capsys, tmp_path, mixed):
    """The first run is the one that counts. A check of the form before the answers are written builds no
    state and compares no answer, so it must not use up run 1. (Found in the rehearsal of the freeze: a
    check of the empty form was logged as "run 1".) As soon as one family has its four answers the check
    gives a verdict on it, and that is a run: else answers could be tried one by one without a count."""
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    log = tmp_path / "form.yaml.checks.log"
    form = make_form(hand_states, capsys, folder, form_file)
    six = six_of(form, mixed)
    answers = {family["family_id"]: right_answers(family) for family in six}

    # Nothing filled in: six lines NOT ANSWERED, no run, no log.
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file, form)
    assert code == 1 and err == "" and len(lines) == 11 and lines[-1].startswith("0 of 6 match; open: ")
    assert lines[3] == NO_RUN_LINE and not log.exists()

    # Three of the four answers in every family: still nothing is built or compared, so still no run.
    holes = {family_id: {key: value for key, value in given.items() if key != "resolution"}
             for family_id, given in answers.items()}
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, fill(form, holes))
    assert code == 1 and lines[3] == NO_RUN_LINE and not log.exists()
    assert all(verdict_of(lines, family["family_id"]) == "NOT ANSWERED: resolution" for family in six)

    # One family answered in full and five open: the check gives a verdict, so this is run 1.
    first_id = six[0]["family_id"]
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, fill(form, {first_id: answers[first_id]}))
    assert code == 1 and lines[3].startswith("run 1 of this form. ")
    assert verdict_of(lines, first_id) == "MATCH" and lines[-1].startswith("1 of 6 match; open: ")

    # An answer that cannot be read is an answered family too: the tools were called for it.
    unreadable = fill(form, {first_id: {**answers[first_id], "role_given": "perhaps"}})
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, unreadable)
    assert code == 1 and lines[3].startswith("run 2 of this form. ")

    # The full form is then run 3, and the log holds three runs.
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, fill(form, answers))
    assert code == 0 and lines[3].startswith("run 3 of this form. ") and lines[-1] == "6 of 6 match"
    starts = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("=== run ")]
    assert [RUN_START.fullmatch(line).group(1) for line in starts] == ["1", "2", "3"]


def test_a_log_that_cannot_be_written_is_named_and_the_verdicts_stand(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    (tmp_path / "form.yaml.checks.log").mkdir()                         # a folder stands where the log goes
    filled = fill(form, {family["family_id"]: right_answers(family) for family in mixed})
    code, lines, _ = check_form_file(hand_states, capsys, folder, form_file, filled)
    assert code == 0 and lines[-1] == "6 of 6 match" and len(lines) == 11
    assert lines[3].startswith("this run is in no log (") and lines[3].endswith("keep this output yourself")


def test_the_form_tells_of_the_log_and_says_where_the_system_prompt_is(hand_states):
    from accessdesk.text import model_text_path
    top = form_of(hand_states, [GRANT]).split("\nfamilies:\n")[0]
    as_one_line = " ".join(line.lstrip("#").strip() for line in top.split("\n"))
    assert ("Each run of the check is added to a log beside this form. The first run is the one that "
            "counts: if an answer differs, find out why before you change it.") in as_one_line
    assert "#   the file text/model_text.yaml, under the key system_prompt.\n" in top
    assert model_text_path() == ROOT / "text" / "model_text.yaml"       # the file that the code reads
    shift = form_of(hand_states, [GRANT], "all_grants_need_approval").split("\nfamilies:\n")[0]
    assert "#   the file text/model_text.yaml, under the key policy_shift, then system_prompt " in shift


# ---------------------------------------------------------------------------
# Answers typed in other natural ways (a review found that these were refused)
# ---------------------------------------------------------------------------
TAB_BEHIND_THE_COLON = re.compile(r"^(  (?:person|resolution|role_given|asked_to_approve):) (.*)$",
                                  re.MULTILINE)


def test_a_tab_a_full_stop_and_other_words_for_nobody_are_read(hand_states, capsys, tmp_path, mixed):
    folder, form_file = scenario_folder(tmp_path, mixed), tmp_path / "form.yaml"
    form = make_form(hand_states, capsys, folder, form_file)
    six = six_of(form, mixed)
    assert sum(right_answers(family)["asked_to_approve"] == "none" for family in six) >= 3
    for variant, nobody in enumerate(["no", "Nobody", "-", "no one", "-   # nobody", "None."]):
        stop = "." if variant % 2 else ""
        answers = {}
        for family in six:
            right = right_answers(family)
            asked = right["asked_to_approve"]
            answers[family["family_id"]] = {"person": right["person"] + stop,
                                            "resolution": right["resolution"] + ".",
                                            "role_given": right["role_given"].capitalize() + stop,
                                            "asked_to_approve": nobody if asked == "none" else asked + stop}
        edited = fill(form, answers)
        if variant >= 3:                # a Tab in the place of the space behind the colon, and one at the end
            edited = TAB_BEHIND_THE_COLON.sub("\\1\t\\2\t", edited)
            assert edited.count(":\t") == 24
        code, lines, err = check_form_file(hand_states, capsys, folder, form_file, edited)
        assert (code, err, lines[-1]) == (0, "", "6 of 6 match"), (nobody, lines)

    # A Tab at the start of a line is still not YAML. The message names it.
    filled = fill(form, {family["family_id"]: right_answers(family) for family in six})
    code, lines, err = check_form_file(hand_states, capsys, folder, form_file,
                                       filled.replace("  person:", "\tperson:", 1))
    assert (code, lines) == (1, DID_NOT_RUN) and "no line may start with a Tab" in err


@pytest.mark.parametrize("nobody", ["no", "nobody", "-", "No.", "no one"])
def test_another_word_for_nobody_is_never_a_wrong_match(hand_states, text, nobody):
    """The manager must be asked in hf-2 and nobody in hf-1: the new words mean "nobody" and nothing else."""
    assert hand_states.check_family(entry(APPROVAL, asked_to_approve=nobody), APPROVAL, text) == \
        ("hf-2", "DIFFERS", "the approval")
    assert hand_states.check_family(entry(GRANT, asked_to_approve=nobody), GRANT, text).status == "MATCH"
    # A word for nobody is no person. In the yes-or-no field only `no` is read, and it means no.
    assert hand_states.check_family(entry(GRANT, person=nobody), GRANT, text).status == "CANNOT BE READ"
    as_yes_or_no = hand_states.check_family(entry(GRANT, role_given=nobody), GRANT, text)
    if nobody in ("no", "No."):
        assert as_yes_or_no == ("hf-1", "DIFFERS", "the grant")
    else:
        assert as_yes_or_no.status == "CANNOT BE READ"


def test_a_full_stop_behind_an_id_is_dropped_and_the_id_is_still_exact(hand_states, text):
    assert hand_states.check_family(entry(GRANT, person="p1."), GRANT, text).status == "MATCH"
    assert hand_states.check_family(entry(APPROVAL, asked_to_approve="p3."), APPROVAL, text).status == "MATCH"
    assert hand_states.check_family(entry(GRANT, person="P1."), GRANT, text).status == "CANNOT BE READ"
    assert hand_states.as_person_id("p1.", {"p1", "p1."}) == "p1."      # an id that ends so is kept whole
    assert hand_states.as_person_id("p1.", {"p1"}) == "p1" and hand_states.as_person_id("p9", {"p1"}) is None
