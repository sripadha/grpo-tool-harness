"""The scenario generator: name pools, one family, one split.

A family is one outcome plus one world seed. Everything random in a family comes from Rng(world_seed), and
world_seed = derive_seed(master seed, split, index). So a family depends only on the text file, the name
files, the config and its own (split, index, outcome, homonym flag).

The default design (cfg.mix_facts False). The outcome fixes three facts of the request's person and role,
and nothing else is mixed in:
    grant:     person active,   lacks the role, role not sensitive
    approval:  person active,   lacks the role, role sensitive
    already:   person active,   has the role,   role not sensitive
    inactive:  person inactive, lacks the role, role not sensitive

The mixed design (cfg.mix_facts True). The outcome is still the first rule of the desk that applies, and
the outcomes are still a quarter each, but a fact that only a LATER rule would look at may be true as
well: an inactive person may ask for a sensitive role, or for a role that the person holds. So a policy
that tries the rules in another order gets some families wrong. MIXED_KINDS holds the kinds of family of
each outcome and their shares; `split_facts` gives every family of a split its kind.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

from accessdesk.config import RunConfig
from accessdesk.env.rules import RULESETS, expected_state, turn_budget
from accessdesk.rng import Rng, derive_seed
from accessdesk.text import STYLES, TextBundle, normalise
from accessdesk.types import OUTCOMES

SPLITS = ("train", "dev", "test", "calib")
ID_PREFIX_OF_SPLIT = {"train": "tr", "dev": "dv", "test": "te", "calib": "ca"}
ID_DIGITS_OF_SPLIT = {"train": 3, "dev": 2, "test": 2, "calib": 2}
TRAIN_STYLES = (1, 2, 3)                      # style 4 is held out
NAMES_PER_FILE = 200
# Positions in each name list, after its seeded shuffle, that belong to each split: [start, end). Disjoint,
# so no name is in two splits.
NAME_LINES_OF_SPLIT = {"train": (0, 110), "dev": (110, 140), "test": (140, 170), "calib": (170, 200)}
NAME_SHUFFLE_LABEL = "name pools"     # the shuffle is fixed: it does not depend on the data seed

OTHER_PERSON_ACTIVE = 0.85      # probability that a person is active (not target, manager or homonym)
HOMONYM_INACTIVE = 0.25         # probability that the homonym is inactive, whatever the target's state is
OTHER_ROLE_SENSITIVE = 0.4      # probability that a role other than the requested one is sensitive
OTHER_ACCESS = 0.25             # probability of each (person, role) pair other than (target, requested role)
NO_MANAGER = 0.2                # probability that a person has no manager (not the target, not its homonym)
ID_TOKEN_LENGTH = 6

# The three rule facts of a family: (the person is active, the person holds the requested role, the
# requested role is sensitive). The desk tries its rules in this order: inactive, already held, sensitive.
Facts = tuple[bool, bool, bool]
FACT_KEYS = ("active", "holds_role", "sensitive")          # the keys of a family record's "facts"

# The default design: the facts of each outcome. No two rule facts are true together.
FACTS_OF_OUTCOME: dict[str, Facts] = {
    "grant": (True, False, False),
    "approval": (True, False, True),
    "already": (True, True, False),
    "inactive": (False, False, False),
}

# The mixed design: outcome -> ((facts, parts), ...), the kinds of family of an outcome. A kind gets
# parts / (sum of the parts) of the outcome's families in a split, as a whole number (largest_remainders).
#   inactive: 1/3 plain, 1/3 with a sensitive role, 1/6 with the role already held, 1/6 with both.
#   already:  1/2 plain, 1/2 with a sensitive role.
#   approval and grant: as in the default design (a true fact more would change the outcome).
# In every kind the outcome is the first rule that applies to the facts (a test checks it by hand).
MIXED_KINDS: dict[str, tuple[tuple[Facts, int], ...]] = {
    "grant": (((True, False, False), 1),),
    "approval": (((True, False, True), 1),),
    "already": (((True, True, False), 1), ((True, True, True), 1)),
    "inactive": (((False, False, False), 2), ((False, False, True), 2),
                 ((False, True, False), 1), ((False, True, True), 1)),
}
# The mixed design: probability that the homonym holds the requested role. It is the share of the families
# in which the person meant holds it: 1/4 (already) + 1/4 * (1/6 + 1/6) (inactive, role held) = 1/3. With
# any other number, "of two matches, take the one that holds the role" would be right more often than
# wrong. In the default design that share is 1/4, which is OTHER_ACCESS.
HOMONYM_HOLDS_ROLE_MIXED = 1 / 3

NamePool = tuple[tuple[str, ...], tuple[str, ...]]             # (first names, last names) of one split
Pools = dict[str, NamePool]                                    # split -> (first names, last names)


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------
def load_name_pools(names_dir: str | Path) -> Pools:
    """split -> (first names, last names), from first_names.txt and last_names.txt (200 lines each).

    Each list is shuffled with a fixed seed and then cut by position (NAME_LINES_OF_SPLIT). The files are in
    order of how common a name is. Cut without the shuffle, the training split would hold the 110 most
    common names and dev, test and calibration the less common ones: a difference between the splits that
    nobody intended. The shuffle depends only on the file's own lines, so the pools are the same in every run.
    If a file is missing, made-up tokens (Fn000.., Ln000..) are used and a warning is given.
    """
    first = _shuffled(_read_names(Path(names_dir) / "first_names.txt", "Fn"), "first_names")
    last = _shuffled(_read_names(Path(names_dir) / "last_names.txt", "Ln"), "last_names")
    return {split: (first[start:end], last[start:end]) for split, (start, end) in NAME_LINES_OF_SPLIT.items()}


def _shuffled(names: tuple[str, ...], which: str) -> tuple[str, ...]:
    return tuple(Rng(derive_seed(NAME_SHUFFLE_LABEL, which)).shuffle(names))


def _read_names(path: Path, fallback_prefix: str) -> tuple[str, ...]:
    if not path.is_file():
        warnings.warn(f"{path} is missing: made-up name tokens ({fallback_prefix}000..) are used instead",
                      stacklevel=3)
        return tuple(f"{fallback_prefix}{i:03d}" for i in range(NAMES_PER_FILE))
    names = tuple(path.read_text(encoding="utf-8").splitlines())
    if len(names) != NAMES_PER_FILE:
        raise ValueError(f"{path} must have exactly {NAMES_PER_FILE} lines, not {len(names)}")
    if any(not name or name != name.strip() for name in names):
        raise ValueError(f"{path} has an empty line or a name with white space at its start or end")
    if len({normalise(name) for name in names}) != len(names):
        raise ValueError(f"{path} holds a name twice (case and spacing ignored)")
    return names


# ---------------------------------------------------------------------------
# One split
# ---------------------------------------------------------------------------
def split_size(split: str, cfg: RunConfig) -> int:
    return {"train": cfg.n_train, "dev": cfg.n_dev, "test": cfg.n_test, "calib": cfg.n_calib}[split]


def master_seed(split: str, cfg: RunConfig) -> int:
    """The calibration families have their own master seed; train, dev and test share the data seed."""
    return cfg.calib_seed if split == "calib" else cfg.data_seed


def family_id(split: str, index: int) -> str:
    return f"{ID_PREFIX_OF_SPLIT[split]}-{index:0{ID_DIGITS_OF_SPLIT[split]}d}"


def split_outcomes(split: str, cfg: RunConfig) -> list[str]:
    """The outcome of every family index: exactly a quarter each, in a seeded order (not in blocks)."""
    n = split_size(split, cfg)
    if n % len(OUTCOMES) != 0:
        raise ValueError(f"the size of split {split!r} must be a multiple of {len(OUTCOMES)}, not {n}")
    balanced = [outcome for outcome in OUTCOMES for _ in range(n // len(OUTCOMES))]
    return Rng(derive_seed(master_seed(split, cfg), split, "outcomes")).shuffle(balanced)


def split_homonyms(split: str, outcomes: list[str], cfg: RunConfig) -> list[bool]:
    """For every family index: does it have a homonym? round(homonym_share * n) of the n families of each
    outcome, chosen by the seed. `round` is Python's: a number that ends in .5 goes to the even neighbour
    (2.5 -> 2, 3.5 -> 4). Every outcome of a split has the same n, so the same number of homonyms."""
    flags = [False] * len(outcomes)
    for outcome in OUTCOMES:
        indices = [i for i, o in enumerate(outcomes) if o == outcome]
        rng = Rng(derive_seed(master_seed(split, cfg), split, "homonyms", outcome))
        for i in rng.sample(indices, round(cfg.homonym_share * len(indices))):
            flags[i] = True
    return flags


def split_homonym_places(split: str, outcomes: list[str], homonyms: list[bool],
                         cfg: RunConfig) -> list[bool | None]:
    """For every family index with a homonym: is the person meant listed BEFORE the homonym in the world
    (True) or after it (False)? None for a family without a homonym.

    `lookup_person` returns the matches in world order. Left to the shuffle, the person meant was the first
    match in 6 of the 8 homonym families of one split and in 1 of 8 of another, so a model with the habit
    "take the first match" got a difference between splits that no training made. Here the place is
    balanced: in each outcome half of the homonym families have the person meant first, in a seeded order.
    With an odd number in an outcome, the one left over is first and last in turn from outcome to outcome
    (the turn starts with a seeded draw), so the whole split is balanced to within one family.
    """
    places: list[bool | None] = [None] * len(outcomes)
    seed = master_seed(split, cfg)
    extra_is_first = Rng(derive_seed(seed, split, "homonym places")).chance(0.5)
    for outcome in OUTCOMES:
        indices = [i for i, o in enumerate(outcomes) if o == outcome and homonyms[i]]
        half, odd = divmod(len(indices), 2)
        balanced = [True] * half + [False] * half + ([extra_is_first] if odd else [])
        if odd:
            extra_is_first = not extra_is_first
        shuffled = Rng(derive_seed(seed, split, "homonym places", outcome)).shuffle(balanced)
        for i, first in zip(indices, shuffled):
            places[i] = first
    return places


def kind_name(outcome: str, facts: Facts) -> str:
    """A kind of family in words: "plain" if the facts are those of the outcome in the default design, else
    the facts that are true on top of them: "sensitive", "held" or "held+sensitive"."""
    words = (None, "held", "sensitive")                      # in the order of Facts; `active` never differs
    extra = [word for word, is_true, was_true in zip(words, facts, FACTS_OF_OUTCOME[outcome])
             if word and is_true and not was_true]
    return "+".join(extra) or "plain"


def largest_remainders(total: int, parts: Sequence[int]) -> list[int]:
    """`total` divided in proportion to `parts`: whole numbers that add up to `total`.

    Each share total * part / sum(parts) is rounded down. The units that are then left over go, one each,
    to the shares with the largest remainders; of two equal remainders the one that comes first in `parts`
    is served first. Example: 50 in the parts (2, 2, 1, 1) is 16.67, 16.67, 8.33, 8.33 -> 17, 17, 8, 8.
    Whole-number arithmetic only, so the result is the same on every machine.
    """
    whole = sum(parts)
    if total == 0:
        return [0] * len(parts)
    if total < 0 or whole <= 0 or min(parts) < 0:
        raise ValueError(f"cannot divide {total} in the parts {list(parts)}")
    counts = [total * part // whole for part in parts]
    remainders = [total * part % whole for part in parts]
    by_remainder = sorted(range(len(parts)), key=lambda i: (-remainders[i], i))
    for i in by_remainder[:total - sum(counts)]:
        counts[i] += 1
    return counts


def split_facts(split: str, outcomes: list[str], homonyms: list[bool], cfg: RunConfig) -> list[Facts | None]:
    """For every family index: its three rule facts in the mixed design. None for every family if
    cfg.mix_facts is False (the facts are then those of FACTS_OF_OUTCOME, and no record names them).

    Exact numbers, no coin flips. For each outcome:
    1. The number of families of each kind is largest_remainders(families of the outcome, the parts of
       MIXED_KINDS), in the order of MIXED_KINDS.
    2. The homonym flag is kept independent of the kind as far as whole numbers allow: the families WITH a
       homonym get largest_remainders(their number, the kind counts of step 1) of each kind, and the
       families without a homonym get the rest. So with a homonym in half of the families, half of the
       families of every kind have one (to within one family).
    3. Which family gets which kind is a seeded shuffle inside each of the two groups.
    """
    facts: list[Facts | None] = [None] * len(outcomes)
    if not cfg.mix_facts:
        return facts
    seed = master_seed(split, cfg)
    for outcome in OUTCOMES:
        kinds = [kind for kind, _ in MIXED_KINDS[outcome]]
        indices = [i for i, o in enumerate(outcomes) if o == outcome]
        counts = largest_remainders(len(indices), [parts for _, parts in MIXED_KINDS[outcome]])
        with_homonym = [i for i in indices if homonyms[i]]
        without_homonym = [i for i in indices if not homonyms[i]]
        counts_with = largest_remainders(len(with_homonym), counts)
        counts_without = [count - taken for count, taken in zip(counts, counts_with)]
        for label, group, group_counts in (("with a homonym", with_homonym, counts_with),
                                           ("without a homonym", without_homonym, counts_without)):
            in_blocks = [kind for kind, count in zip(kinds, group_counts) for _ in range(count)]
            shuffled = Rng(derive_seed(seed, split, "facts", outcome, label)).shuffle(group)
            for i, kind in zip(shuffled, in_blocks):
                facts[i] = kind
    return facts


def make_split(split: str, text: TextBundle, pools: Pools, cfg: RunConfig) -> list[dict]:
    """The family records of one split, in index order."""
    outcomes = split_outcomes(split, cfg)
    homonyms = split_homonyms(split, outcomes, cfg)
    places = split_homonym_places(split, outcomes, homonyms, cfg)
    facts = split_facts(split, outcomes, homonyms, cfg)
    return [make_family(split, i, outcomes[i], homonyms[i], text, pools, cfg, target_first=places[i],
                        facts=facts[i])
            for i in range(len(outcomes))]


def make_all_splits(text: TextBundle, pools: Pools, cfg: RunConfig) -> dict[str, list[dict]]:
    """split -> family records, for the four splits. Checks that no world seed is used by two families."""
    families = {split: make_split(split, text, pools, cfg) for split in SPLITS}
    seeds = [family["world_seed"] for split in SPLITS for family in families[split]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("two families have the same world seed")
    return families


# ---------------------------------------------------------------------------
# One family
# ---------------------------------------------------------------------------
def make_family(split: str, index: int, outcome: str, homonym: bool, text: TextBundle, pools: Pools,
                cfg: RunConfig, *, target_first: bool | None = None, facts: Facts | None = None) -> dict:
    """One family record (section 6 of the build specification).

    `target_first` is used only for a family with a homonym: True puts the person meant before the homonym
    in the world's list of people, False after it (the two change places if the seeded shuffle put them the
    other way round; nothing else moves, and no random number is used for it). None leaves the shuffle as
    it is. `make_split` gives the balanced values of `split_homonym_places`.

    `facts` is given only in the mixed design: the three rule facts of the family, one of the kinds of
    MIXED_KINDS[outcome] (`make_split` gives the values of `split_facts`). The record then has the key
    "facts", which no prompt and no tool shows. None is the default design: the facts are those of
    FACTS_OF_OUTCOME[outcome], and the record has no such key. No random number is drawn for a fact, so
    a family index with the same homonym flag has the same names, ids, departments and managers in both
    designs.
    The outcome is only a label here. The expected states are built by rules.expected_state, which reads
    the three facts from the world.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}; known: {OUTCOMES}")
    if facts is not None and facts not in [kind for kind, _ in MIXED_KINDS[outcome]]:
        raise ValueError(f"the facts {facts} are not a kind of the outcome {outcome!r} (MIXED_KINDS)")
    active, holds_role, sensitive = FACTS_OF_OUTCOME[outcome] if facts is None else facts
    world_seed = derive_seed(master_seed(split, cfg), split, index)
    rng = Rng(world_seed)
    people, target, manager, namesake = _make_people(rng, active, homonym, text, pools[split], cfg)
    roles, requested = _make_roles(rng, sensitive, text, cfg)
    namesake_holds = OTHER_ACCESS if facts is None else HOMONYM_HOLDS_ROLE_MIXED
    access = _make_access(rng, holds_role, people, roles, target, requested, namesake, namesake_holds)
    _assign_ids(rng, people, roles, text)
    listed = rng.shuffle(people)
    if homonym and target_first is not None:
        _place_target(listed, target, target_first)

    family: dict = {"family_id": family_id(split, index), "split": split, "outcome": outcome,
                    "homonym": homonym}
    if facts is not None:
        family["facts"] = dict(zip(FACT_KEYS, facts))
    family.update({
        "world_seed": world_seed,
        "world": {
            "people": [_person_record(p) for p in listed],
            "roles": [_role_record(r) for r in rng.shuffle(roles)],
            "access": sorted([p["person_id"], r["role_id"]] for p, r in access),
        },
        "request": {"person_name": target["name"], "department": target["department"],
                    "role_name": requested["name"]},
        "truth": {"person_id": target["person_id"], "role_id": requested["role_id"],
                  "manager_id": manager["person_id"]},
    })
    family["expected"] = {ruleset: expected_state(family, ruleset) for ruleset in RULESETS}
    family["turn_budget"] = {ruleset: turn_budget(family, ruleset) for ruleset in RULESETS}
    family["wordings"] = _make_wordings(rng, split, family["request"], text)
    return family


def _make_people(rng: Rng, active: bool, homonym: bool, text: TextBundle, pool: NamePool,
                 cfg: RunConfig) -> tuple[list[dict], dict, dict, dict | None]:
    """(people, target, the target's manager, the target's homonym or None). `active` is the target's
    state. People are dicts that still lack an id; "manager" is a dict."""
    minimum = 3 if homonym else 2
    if cfg.n_people < minimum or cfg.n_departments < 2:
        raise ValueError("a world needs 2 departments and room for the target, the manager and a homonym")
    departments = rng.sample(text.departments, cfg.n_departments)
    names = _full_names(rng, pool, cfg.n_people - 1 if homonym else cfg.n_people)

    target = {"name": names[0], "department": rng.choice(departments),
              "active": active, "manager": None}
    manager = {"name": names[1], "department": rng.choice(departments), "active": True, "manager": None}
    people = [target, manager]
    namesake = None
    if homonym:
        other_departments = [d for d in departments if d != target["department"]]
        # The homonym's state is drawn without a look at the target's state. The target is inactive in one
        # family in four (the outcome `inactive`, in both designs), and so is the homonym: when exactly one
        # of the two is inactive, it is the target half of the time, and the state says nothing about who
        # is meant.
        namesake = {"name": target["name"], "department": rng.choice(other_departments),
                    "active": not rng.chance(HOMONYM_INACTIVE), "manager": None}
        people.append(namesake)
    for name in names[2:]:
        people.append({"name": name, "department": rng.choice(departments),
                       "active": rng.chance(OTHER_PERSON_ACTIVE), "manager": None})
    _assign_managers(rng, people, target, manager, namesake)
    return people, target, manager, namesake


def _full_names(rng: Rng, pool: NamePool, count: int) -> list[str]:
    """`count` different full names "First Last" from the split's own first names and last names."""
    first_names, last_names = pool
    if count > len(first_names) * len(last_names):
        raise ValueError("the name pool is too small for this world size")
    names: list[str] = []
    while len(names) < count:
        name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        if name not in names:
            names.append(name)
    return names


def _assign_managers(rng: Rng, people: list[dict], target: dict, manager: dict,
                     namesake: dict | None = None) -> None:
    """Give every person a manager who comes earlier in a seeded order, or none.

    The first person of the order has no manager, so at least one person has none, nobody is their own manager
    and there is no cycle. The target's manager is fixed and is placed before the target.

    `namesake` is the homonym of the target, if the family has one. The target always has a manager, so the
    homonym always gets one too: otherwise "the match that has a manager" would pick the right person
    without the department. If the homonym is first in the order, it changes places with the target's
    manager (who then has no manager, and still comes before the target).
    The homonym's manager is never the target. The target's manager is never the homonym, so if the homonym
    could report to the target, "of two matches of which one manages the other, take the manager" would
    pick the right person without the department.
    """
    order = rng.shuffle(people)
    at_manager, at_target = _position(order, manager), _position(order, target)
    if at_manager > at_target:
        order[at_manager], order[at_target] = target, manager
    if namesake is not None and order[0] is namesake:
        at_manager = _position(order, manager)       # found before the swap: the swap moves the manager
        order[0], order[at_manager] = manager, namesake
    for i, person in enumerate(order):
        if person is target:
            person["manager"] = manager
        elif person is namesake:
            # Never first in the order, and the first one is never the target: the list is not empty.
            person["manager"] = rng.choice([earlier for earlier in order[:i] if earlier is not target])
        elif i > 0 and not rng.chance(NO_MANAGER):
            person["manager"] = rng.choice(order[:i])


def _place_target(listed: list[dict], target: dict, target_first: bool) -> None:
    """Make the target come before its homonym in `listed` (target_first) or after it, by letting the two
    change places if they are the other way round."""
    namesake = next(p for p in listed if p is not target and p["name"] == target["name"])
    at_target, at_namesake = _position(listed, target), _position(listed, namesake)
    if (at_target < at_namesake) != target_first:
        listed[at_target], listed[at_namesake] = namesake, target


def _position(people: list[dict], person: dict) -> int:
    """The index of this very person (the same object, not an equal one) in the list."""
    return next(i for i, candidate in enumerate(people) if candidate is person)


def _make_roles(rng: Rng, sensitive: bool, text: TextBundle, cfg: RunConfig) -> tuple[list[dict], dict]:
    """(roles, the requested role). `sensitive` is the fact of the requested role; no number is drawn for
    it. In the default design only the `approval` outcome has a sensitive requested role."""
    names = rng.sample(text.roles, cfg.n_roles)
    requested = {"name": names[0], "sensitive": sensitive}
    others = [{"name": name, "sensitive": rng.chance(OTHER_ROLE_SENSITIVE)} for name in names[1:]]
    return [requested] + others, requested


def _make_access(rng: Rng, holds_role: bool, people: list[dict], roles: list[dict], target: dict,
                 requested: dict, namesake: dict | None = None,
                 namesake_holds: float = OTHER_ACCESS) -> list[tuple[dict, dict]]:
    """The (person, role) pairs that hold at the start.

    The pair (target, requested role) is the fact `holds_role`; no number is drawn for it. In the default
    design it is present only for `already`.
    Every other pair is present with probability OTHER_ACCESS, except the pair (homonym, requested role),
    which is present with probability `namesake_holds` (one draw, like every other pair).
    """
    access = []
    for person in people:
        for role in roles:
            if person is target and role is requested:
                held = holds_role
            elif person is namesake and role is requested:
                held = rng.chance(namesake_holds)
            else:
                held = rng.chance(OTHER_ACCESS)
            if held:
                access.append((person, role))
    return access


def _assign_ids(rng: Rng, people: list[dict], roles: list[dict], text: TextBundle) -> None:
    """Random ids: prefix + 6 characters, unique in the world."""
    used: set[str] = set()
    for item, key, prefix in ([(p, "person_id", text.id_prefix_person) for p in people]
                              + [(r, "role_id", text.id_prefix_role) for r in roles]):
        new_id = prefix + rng.token(ID_TOKEN_LENGTH)
        while new_id in used:
            new_id = prefix + rng.token(ID_TOKEN_LENGTH)
        used.add(new_id)
        item[key] = new_id


def _person_record(person: dict) -> dict:
    manager = person["manager"]
    return {"person_id": person["person_id"], "name": person["name"], "department": person["department"],
            "status": "active" if person["active"] else "inactive",
            "manager_id": None if manager is None else manager["person_id"]}


def _role_record(role: dict) -> dict:
    return {"role_id": role["role_id"], "name": role["name"], "sensitive": role["sensitive"]}


def _make_wordings(rng: Rng, split: str, request: dict, text: TextBundle) -> list[dict]:
    """One wording per style: train families get styles 1 to 3, all other splits get styles 1 to 4."""
    wordings = []
    for style in (TRAIN_STYLES if split == "train" else STYLES):
        templates = text.request_templates[style]
        index = rng.below(len(templates))
        wordings.append({"style": style, "template": index, "text": templates[index].format(**request)})
    return wordings
