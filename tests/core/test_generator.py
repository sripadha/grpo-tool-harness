"""accessdesk/env/generator.py, accessdesk/env/scenarios.py and scripts/build_scenarios.py."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import pytest

import yaml

from accessdesk.config import RunConfig
from accessdesk.env import generator
from accessdesk.env.generator import (NAME_LINES_OF_SPLIT, SPLITS, largest_remainders, load_name_pools,
                                      make_all_splits, make_family, make_split, split_facts,
                                      split_homonym_places, split_homonyms, split_outcomes)
from accessdesk.env.scenarios import load_families, make_sets, save_families, scenario_ids, split_scenario_id
from accessdesk.env.tools import execute
from accessdesk.env.world import World
from accessdesk.rng import TOKEN_ALPHABET, derive_seed
from accessdesk.text import placeholder_path
from accessdesk.types import OUTCOMES
from tests.conftest import MARKER_PATTERN

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build_scenarios.py"
BUILT_FILES = ["train.jsonl", "dev.jsonl", "test.jsonl", "calib.jsonl", "sets.json", "summary.json"]

# Typed by hand from the specification and the plan.
SIZES = {"train": 200, "dev": 48, "test": 48, "calib": 24}
SCENARIOS = {"train": 600, "dev": 192, "test": 192, "calib": 96}
HOMONYMS_PER_OUTCOME = {"train": 10, "dev": 2, "test": 2, "calib": 1}     # round(0.2 * families per outcome)
ID_PATTERN = {"train": r"tr-\d{3}", "dev": r"dv-\d{2}", "test": r"te-\d{2}", "calib": r"ca-\d{2}"}
# outcome -> (target is active, target has the role, role is sensitive)
FACTS_OF_OUTCOME = {
    "grant": (True, False, False),
    "approval": (True, False, True),
    "already": (True, True, False),
    "inactive": (False, False, False),
}


# The design before the freeze day (2026-11-18): the outcome of a family fixes its three rule facts, and
# one family in five holds two people of one full name. Since that day the defaults of accessdesk/config.py
# are the final design (mix_facts True, homonym_share 0.5), and the fixtures `cfg` and `families` of
# tests/conftest.py follow the defaults. A test of this file that is about the earlier design, with its
# counts typed by hand, asks for `plain_cfg` and `plain_families`: they set the two old values by hand.
PLAIN_DESIGN = {"mix_facts": False, "homonym_share": 0.2}


@pytest.fixture(scope="module")
def plain_cfg(cfg) -> RunConfig:
    """The config of the tests with the two values of the design before the freeze day."""
    return dataclasses.replace(cfg, **PLAIN_DESIGN)


@pytest.fixture(scope="module")
def plain_families(text, pools, plain_cfg) -> dict[str, list[dict]]:
    """The families of every split in the design before the freeze day. Read-only, like `families`."""
    return make_all_splits(text, pools, plain_cfg)


def all_families(families: dict) -> list[dict]:
    return [family for split in SPLITS for family in families[split]]


def target_of(family: dict) -> dict:
    return next(p for p in family["world"]["people"] if p["person_id"] == family["truth"]["person_id"])


def requested_role_of(family: dict) -> dict:
    return next(r for r in family["world"]["roles"] if r["role_id"] == family["truth"]["role_id"])


def to_bytes(families: list[dict]) -> bytes:
    return "".join(json.dumps(family, ensure_ascii=False) + "\n" for family in families).encode("utf-8")


def load_build_script():
    spec = importlib.util.spec_from_file_location("build_scenarios", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------
def write_name_files(folder: Path, first: list[str] | None = None, last: list[str] | None = None) -> Path:
    first = [f"First{i}" for i in range(200)] if first is None else first
    last = [f"Last{i}" for i in range(200)] if last is None else last
    (folder / "first_names.txt").write_text("\n".join(first) + "\n", encoding="utf-8")
    (folder / "last_names.txt").write_text("\n".join(last) + "\n", encoding="utf-8")
    return folder


def line_numbers(names: tuple[str, ...]) -> list[int]:
    """The line numbers of made-up names such as "First17" or "Ln003" in the file they came from."""
    return [int("".join(character for character in name if character.isdigit())) for name in names]


def test_name_pools_are_a_seeded_shuffle_of_the_file_cut_by_position(tmp_path):
    assert NAME_LINES_OF_SPLIT == {"train": (0, 110), "dev": (110, 140), "test": (140, 170),
                                   "calib": (170, 200)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                           # real files: no fallback warning
        pools = load_name_pools(write_name_files(tmp_path))
    assert list(pools) == ["train", "dev", "test", "calib"]
    assert [(len(pools[split][0]), len(pools[split][1])) for split in SPLITS] == [(110, 110)] + [(30, 30)] * 3
    for kind, prefix in ((0, "First"), (1, "Last")):
        in_pool_order = [name for split in SPLITS for name in pools[split][kind]]
        assert sorted(in_pool_order) == sorted(f"{prefix}{i}" for i in range(200))     # every name, once
        assert in_pool_order != [f"{prefix}{i}" for i in range(200)]                   # not in file order
    # First names and last names are shuffled differently: line i of one file does not stay with line i
    # of the other.
    assert line_numbers(pools["dev"][0]) != line_numbers(pools["dev"][1])
    # The same file gives the same pools again, and the pools do not depend on the data seed.
    assert load_name_pools(tmp_path) == pools


def test_every_split_gets_common_and_less_common_names_alike(tmp_path):
    """The name files are in order of how common a name is. Cut by line number, train would hold lines
    0 to 109 (mean line 54.5) and calibration lines 170 to 199 (mean 184.5). After the shuffle the mean line
    of every pool is near 99.5, the mean of the whole file. For 30 names drawn from 200 lines the standard
    deviation of that mean is about 9.7, for 110 names about 3.7; the limits below are 3 standard
    deviations and more."""
    pools = load_name_pools(write_name_files(tmp_path))
    for split in SPLITS:
        for kind in (0, 1):
            lines = line_numbers(pools[split][kind])
            mean = sum(lines) / len(lines)
            assert 99.5 - 30 < mean < 99.5 + 30, (split, kind, mean)
            assert min(lines) < 40 and max(lines) > 160, (split, kind)      # from both ends of the file
    by_line_number = [list(range(start, end)) for start, end in NAME_LINES_OF_SPLIT.values()]
    assert [sum(lines) / len(lines) for lines in by_line_number] == [54.5, 124.5, 154.5, 184.5]


def test_name_pools_fall_back_to_made_up_tokens_with_a_warning(tmp_path):
    with pytest.warns(UserWarning, match="is missing") as caught:
        pools = load_name_pools(tmp_path)
    assert len(caught) == 2                                      # one per missing file
    firsts = [name for split in SPLITS for name in pools[split][0]]
    lasts = [name for split in SPLITS for name in pools[split][1]]
    assert sorted(firsts) == [f"Fn{i:03d}" for i in range(200)]
    assert sorted(lasts) == [f"Ln{i:03d}" for i in range(200)]
    assert [len(pools[split][0]) for split in SPLITS] == [110, 30, 30, 30]


def test_one_missing_file_falls_back_for_that_file_only(tmp_path):
    write_name_files(tmp_path)
    (tmp_path / "last_names.txt").unlink()
    with pytest.warns(UserWarning, match="last_names.txt is missing") as caught:
        pools = load_name_pools(tmp_path)
    assert len(caught) == 1
    assert all(name.startswith("First") for name in pools["dev"][0])
    assert all(name.startswith("Ln") for name in pools["dev"][1])


@pytest.mark.parametrize("first, wanted", [
    ([f"First{i}" for i in range(199)], "exactly 200 lines"),
    ([f"First{i}" for i in range(201)], "exactly 200 lines"),
    ([f"First{i}" for i in range(199)] + [""], "empty line"),
    ([f"First{i}" for i in range(199)] + ["First7 "], "white space"),
    ([f"First{i}" for i in range(199)] + ["FIRST7"], "twice"),
])
def test_bad_name_files_are_refused(tmp_path, first, wanted):
    with pytest.raises(ValueError, match=wanted):
        load_name_pools(write_name_files(tmp_path, first=first))


def test_pools_of_two_splits_share_no_name(pools):
    for index, kind in ((0, "first"), (1, "last")):
        seen: dict[str, str] = {}
        for split in SPLITS:
            for name in pools[split][index]:
                assert name not in seen, f"{kind} name {name!r} is in {seen.get(name)} and {split}"
                seen[name] = split
        assert len(seen) == 200


def test_real_name_files_are_valid_if_they_exist():
    names_dir = ROOT / "data" / "names"
    if not all((names_dir / name).is_file() for name in ("first_names.txt", "last_names.txt")):
        pytest.skip("data/names/*.txt do not exist yet")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pools = load_name_pools(names_dir)
    assert sum(len(pools[split][0]) for split in SPLITS) == 200
    # No word is a first name in one split and a last name in another split.
    split_of: dict[str, str] = {}
    for split in SPLITS:
        for name in pools[split][0] + pools[split][1]:
            assert split_of.setdefault(name.casefold(), split) == split, name
    assert len(split_of) == 400


# ---------------------------------------------------------------------------
# Splits: sizes, ids, seeds, balance
# ---------------------------------------------------------------------------
def test_split_sizes_and_scenario_counts(families):
    assert {split: len(families[split]) for split in SPLITS} == SIZES
    assert {split: sum(len(f["wordings"]) for f in families[split]) for split in SPLITS} == SCENARIOS
    assert sum(SIZES[split] for split in ("train", "dev", "test")) == 296


def test_family_record_keys_and_order(families, plain_families):
    keys = ["family_id", "split", "outcome", "homonym", "world_seed", "world", "request",
            "truth", "expected", "turn_budget", "wordings"]
    # The mixed design (the default since the freeze day) has one more key: "facts", after "homonym".
    for built, wanted in ((plain_families, keys), (families, keys[:4] + ["facts"] + keys[4:])):
        for family in all_families(built):
            assert list(family) == wanted
            assert list(family["world"]) == ["people", "roles", "access"]
            assert list(family["request"]) == ["person_name", "department", "role_name"]
            assert list(family["truth"]) == ["person_id", "role_id", "manager_id"]
            for person in family["world"]["people"]:
                assert list(person) == ["person_id", "name", "department", "status", "manager_id"]
                assert person["status"] in ("active", "inactive")
            for role in family["world"]["roles"]:
                assert list(role) == ["role_id", "name", "sensitive"]
                assert type(role["sensitive"]) is bool
            assert type(family["homonym"]) is bool
            for wording in family["wordings"]:
                assert list(wording) == ["style", "template", "text"]


def test_family_ids_and_split_field(families):
    for split in SPLITS:
        for index, family in enumerate(families[split]):
            assert family["split"] == split
            assert re.fullmatch(ID_PATTERN[split], family["family_id"])
            assert int(family["family_id"].split("-")[1]) == index
    assert families["train"][0]["family_id"] == "tr-000" and families["train"][199]["family_id"] == "tr-199"
    assert families["dev"][7]["family_id"] == "dv-07" and families["test"][47]["family_id"] == "te-47"
    assert families["calib"][23]["family_id"] == "ca-23"
    ids = [family["family_id"] for family in all_families(families)]
    assert len(set(ids)) == len(ids) == 320


def test_world_seed_is_derived_from_master_seed_split_and_index(families, cfg):
    for split in SPLITS:
        master = cfg.calib_seed if split == "calib" else cfg.data_seed
        for index, family in enumerate(families[split]):
            assert family["world_seed"] == derive_seed(master, split, index)


def test_world_seeds_are_unique_over_all_families(families):
    seeds = [family["world_seed"] for family in all_families(families)]
    assert len(set(seeds)) == len(seeds) == 320


def test_make_all_splits_refuses_a_repeated_world_seed(text, pools, cfg, monkeypatch):
    monkeypatch.setattr("accessdesk.env.generator.derive_seed", lambda *parts: 5)
    with pytest.raises(ValueError, match="same world seed"):
        make_all_splits(text, pools, cfg)


def test_outcomes_are_exactly_balanced_in_every_split(families):
    for split in SPLITS:
        counts = Counter(family["outcome"] for family in families[split])
        assert counts == {outcome: SIZES[split] // 4 for outcome in OUTCOMES}, split


def test_outcomes_are_not_in_blocks(families):
    for split in SPLITS:
        outcomes = [family["outcome"] for family in families[split]]
        changes = sum(a != b for a, b in zip(outcomes, outcomes[1:]))
        assert changes > len(outcomes) // 2, (split, changes)     # blocks would give 3 changes
        assert len(set(outcomes[:len(outcomes) // 4])) > 1


def test_a_split_size_that_is_not_a_multiple_of_four_is_refused(cfg):
    with pytest.raises(ValueError, match="multiple of 4"):
        split_outcomes("dev", dataclasses.replace(cfg, n_dev=50))


def test_homonym_counts_are_exact_per_outcome_and_split(families, plain_families):
    for split in SPLITS:
        counts = Counter(family["outcome"] for family in plain_families[split] if family["homonym"])
        assert counts == {outcome: HOMONYMS_PER_OUTCOME[split] for outcome in OUTCOMES}, split
        # The default share since the freeze day is 0.5: half of the families of each outcome.
        counts = Counter(family["outcome"] for family in families[split] if family["homonym"])
        assert counts == {outcome: SIZES[split] // 8 for outcome in OUTCOMES}, split


def test_homonym_flags_follow_the_share_in_the_config(cfg):
    outcomes = split_outcomes("train", cfg)
    for share, per_outcome in ((0.0, 0), (0.2, 10), (0.5, 25), (1.0, 50)):
        flags = split_homonyms("train", outcomes, dataclasses.replace(cfg, homonym_share=share))
        counts = Counter(outcome for outcome, flag in zip(outcomes, flags) if flag)
        assert counts == ({outcome: per_outcome for outcome in OUTCOMES} if per_outcome else {})


# ---------------------------------------------------------------------------
# A larger share of same-name families (0.5 and 1.0)
# ---------------------------------------------------------------------------
# Typed by hand: families with a homonym per outcome (50, 12, 12 and 6 families per outcome), and families
# in the whole split in which the person meant is the first of the two matches (half of the homonyms).
HOMONYMS_AT_SHARE = {0.5: {"train": 25, "dev": 6, "test": 6, "calib": 3},
                     1.0: {"train": 50, "dev": 12, "test": 12, "calib": 6}}
FIRST_AT_SHARE = {0.5: {"train": 50, "dev": 12, "test": 12, "calib": 6},
                  1.0: {"train": 100, "dev": 24, "test": 24, "calib": 12}}


@pytest.fixture(scope="module", params=[0.5, 1.0])
def at_share(request, text, pools, plain_cfg) -> tuple[float, dict]:
    """(share, the families of every split built with that share of homonyms; the switch mix_facts is off)."""
    share = request.param
    return share, make_all_splits(text, pools, dataclasses.replace(plain_cfg, homonym_share=share))


def test_a_larger_share_gives_exact_homonym_counts_and_balanced_places(at_share):
    share, built = at_share
    for split in SPLITS:
        counts = Counter(family["outcome"] for family in built[split] if family["homonym"])
        assert counts == {outcome: HOMONYMS_AT_SHARE[share][split] for outcome in OUTCOMES}, split
        assert_places_are_balanced(built[split])
        n_first = sum(person_meant_is_the_first_match(f) for f in built[split] if f["homonym"])
        assert n_first == FIRST_AT_SHARE[share][split], split
        outcomes = Counter(family["outcome"] for family in built[split])
        assert outcomes == {outcome: SIZES[split] // 4 for outcome in OUTCOMES}


def test_a_larger_share_keeps_every_rule_of_a_world(at_share, pools):
    """The checks of the default share, run again on the families of the larger share."""
    share, built = at_share
    test_outcome_fixes_the_three_facts_and_never_mixes_them(built)
    test_request_names_the_target_and_only_the_target(built)
    test_full_names_are_unique_in_a_world_except_for_the_homonym_pair(built)
    test_manager_rules(built)
    test_names_come_from_the_pools_of_the_family_s_own_split(built, pools)
    test_no_name_seed_or_family_id_is_in_two_splits(built, pools)
    seen = 0
    for family in all_families(built):
        assert len(family["world"]["people"]) == 8
        if family["homonym"]:
            target, other = target_of(family), homonym_of(family)
            assert other["department"] != target["department"]
            assert other["manager_id"] is not None and other["manager_id"] != target["person_id"]
            assert family["truth"]["manager_id"] != other["person_id"]
            seen += 1
    assert seen == 4 * sum(HOMONYMS_AT_SHARE[share].values()) == round(share * 320)


def test_the_name_pools_are_large_enough_for_a_homonym_in_every_family(pools, cfg):
    """A world needs 8 full names (7 with a homonym) out of first names x last names of the split's pool:
    110 x 110 for train, 30 x 30 for the other splits. Names are drawn for one world at a time, so the
    share of homonyms does not use the pools up."""
    assert cfg.n_people == 8
    for split in SPLITS:
        first_names, last_names = pools[split]
        assert len(first_names) * len(last_names) >= 900 > 100 * cfg.n_people


def split_of_homonym_families(text, pools, cfg, **changes) -> list[dict]:
    """The families with a homonym of one large train split, built by `make_split` itself."""
    built = make_split("train", text, pools, dataclasses.replace(cfg, **changes))
    return [family for family in built if family["homonym"]]


@pytest.mark.parametrize("changes", [
    {"homonym_share": 1.0, "n_train": 4000},
    {"homonym_share": 0.5, "n_train": 8000},
    {"homonym_share": 1.0, "n_train": 4000, "mix_facts": True},
    {"homonym_share": 0.5, "n_train": 8000, "mix_facts": True},
], ids=["share 1.0", "share 0.5", "share 1.0 mixed", "share 0.5 mixed"])
def test_no_simple_rule_tells_the_target_from_its_homonym_at_a_larger_share(text, pools, cfg, changes):
    """The shortcut rules of test_no_simple_rule_tells_the_target_from_its_homonym, on 4,000 families
    with a homonym that `make_split` built with a share of 0.5 or 1.0, in both designs.

    Measured on 2026-11-17 (placeholder text, placeholder data seed 9001), as right / decided = share:
                              the inactive match     the match that holds the requested role
        share 1.0             762 / 1520 = 0.501     720 / 1509 = 0.477
        share 0.5             753 / 1491 = 0.505     730 / 1499 = 0.487
        share 1.0, mixed      762 / 1520 = 0.501     832 / 1749 = 0.476
        share 0.5, mixed      753 / 1491 = 0.505     877 / 1807 = 0.485
    (The two designs draw the same random numbers, so their rows move together. Over six other data
    seeds, 11 to 16, share 1.0: holds the role 4506 / 8942 = 0.504 default, 5341 / 10618 = 0.503 mixed.)
    "The first match" is right in exactly 2,000 of 4,000: `make_split` balances the place.
    """
    scores = shortcut_scores(split_of_homonym_families(text, pools, cfg, **changes))
    for rule in ("inactive", "holds_role"):
        decided, right = scores[rule]
        assert decided >= 1200, (rule, decided)
        assert 0.44 <= right / decided <= 0.56, (rule, right, decided)
    assert scores["first"] == (4000, 2000)
    assert scores["has_manager"] == (0, 0)
    assert scores["manages_the_other"] == (0, 0)


def test_in_the_mixed_design_the_homonym_holds_the_role_as_often_as_the_person_meant(text, pools, cfg,
                                                                                    monkeypatch):
    """In the mixed design the person meant holds the requested role in 1/4 (already) + 1/4 * 1/3
    (inactive, role held) = 1/3 of the families, so the homonym holds it with probability 1/3 as well.
    With the 0.25 of every other (person, role) pair, "of two matches, take the one that holds the role"
    would be right in 6 of 10 pairs in which it can decide (0.25 / (0.25 + 2/3 * 0.25)): shown here by
    putting 0.25 back. Measured with the placeholder data seed 9001: 962 / 1659 = 0.580."""
    held = sum(sum(parts for facts, parts in kinds if facts[1]) / sum(parts for _, parts in kinds)
               for kinds in generator.MIXED_KINDS.values()) / len(OUTCOMES)
    assert held == pytest.approx(1 / 3) and generator.HOMONYM_HOLDS_ROLE_MIXED == pytest.approx(1 / 3)
    assert generator.OTHER_ACCESS == 0.25
    changes = {"homonym_share": 1.0, "n_train": 4000, "mix_facts": True}
    monkeypatch.setattr(generator, "HOMONYM_HOLDS_ROLE_MIXED", generator.OTHER_ACCESS)
    decided, right = shortcut_scores(split_of_homonym_families(text, pools, cfg, **changes))["holds_role"]
    assert decided >= 1200 and right / decided > 0.56, (right, decided)      # outside the band of "chance"


# ---------------------------------------------------------------------------
# The mixed design: which family gets which facts
# ---------------------------------------------------------------------------
# Typed by hand: families of an outcome in a split -> families of each kind, in the order of
# generator.MIXED_KINDS. inactive: plain, sensitive, held, held+sensitive (1/3, 1/3, 1/6, 1/6).
# already: plain, sensitive (1/2, 1/2). 50 inactive families: 16.67, 16.67, 8.33, 8.33 -> 17, 17, 8, 8.
QUOTAS = {
    6: {"inactive": (2, 2, 1, 1), "already": (3, 3), "approval": (6,), "grant": (6,)},          # split of 24
    12: {"inactive": (4, 4, 2, 2), "already": (6, 6), "approval": (12,), "grant": (12,)},       # split of 48
    50: {"inactive": (17, 17, 8, 8), "already": (25, 25), "approval": (50,), "grant": (50,)},   # split of 200
}
KINDS = {      # (active, holds the role, sensitive), typed by hand
    "grant": [(True, False, False)],
    "approval": [(True, False, True)],
    "already": [(True, True, False), (True, True, True)],
    "inactive": [(False, False, False), (False, False, True), (False, True, False), (False, True, True)],
}


def first_rule_that_applies(active: bool, holds_role: bool, sensitive: bool) -> str:
    """The outcome under the default rules, written here by hand."""
    if not active:
        return "inactive"
    if holds_role:
        return "already"
    return "approval" if sensitive else "grant"


def test_largest_remainders_by_hand():
    assert largest_remainders(50, (2, 2, 1, 1)) == [17, 17, 8, 8]     # 16.67 16.67 8.33 8.33: two units left
    assert largest_remainders(12, (2, 2, 1, 1)) == [4, 4, 2, 2]       # whole numbers: nothing left over
    assert largest_remainders(7, (2, 2, 1, 1)) == [3, 2, 1, 1]        # 2.33 2.33 1.17 1.17: one unit, first
    assert largest_remainders(3, (2, 2, 1, 1)) == [1, 1, 1, 0]        # 1 1 0.5 0.5: one unit, the first 0.5
    assert largest_remainders(1, (2, 2, 1, 1)) == [1, 0, 0, 0]
    assert largest_remainders(5, (1, 1)) == [3, 2]
    assert largest_remainders(25, (17, 17, 8, 8)) == [9, 8, 4, 4]     # 8.5 8.5 4 4: one unit, the first 8.5
    assert largest_remainders(4, (1, 0, 3)) == [1, 0, 3]              # a part of 0 gets nothing
    assert largest_remainders(0, (2, 1)) == [0, 0] and largest_remainders(0, (0, 0)) == [0, 0]
    for total in range(0, 40):
        for parts in ((2, 2, 1, 1), (1, 1), (1,), (5, 3, 1)):
            counts = largest_remainders(total, parts)
            assert sum(counts) == total
            exact = [total * part / sum(parts) for part in parts]
            assert all(abs(count - share) < 1 for count, share in zip(counts, exact))
    for total, parts in ((3, (0, 0)), (-1, (1, 1)), (3, (2, -1))):
        with pytest.raises(ValueError):
            largest_remainders(total, parts)


def test_the_kinds_of_the_mixed_design_are_the_specified_ones():
    """Every kind is listed under the outcome that the rules give it, the plain kind comes first, and the
    parts are the shares of the task: inactive 1/3, 1/3, 1/6, 1/6; already 1/2, 1/2."""
    assert generator.FACTS_OF_OUTCOME == FACTS_OF_OUTCOME
    assert generator.FACT_KEYS == ("active", "holds_role", "sensitive")
    listed = {outcome: [facts for facts, _ in kinds] for outcome, kinds in generator.MIXED_KINDS.items()}
    assert listed == KINDS
    for outcome, kinds in generator.MIXED_KINDS.items():
        assert all(first_rule_that_applies(*facts) == outcome for facts, _ in kinds)
        assert kinds[0][0] == FACTS_OF_OUTCOME[outcome]
    shares = {outcome: [parts / sum(p for _, p in kinds) for _, parts in kinds]
              for outcome, kinds in generator.MIXED_KINDS.items()}
    assert shares == {"grant": [1.0], "approval": [1.0], "already": [0.5, 0.5],
                      "inactive": [pytest.approx(1 / 3), pytest.approx(1 / 3), pytest.approx(1 / 6),
                                   pytest.approx(1 / 6)]}
    names = {outcome: [generator.kind_name(outcome, facts) for facts in KINDS[outcome]]
             for outcome in OUTCOMES}
    assert names == {"grant": ["plain"], "approval": ["plain"], "already": ["plain", "sensitive"],
                     "inactive": ["plain", "sensitive", "held", "held+sensitive"]}
    # All eight combinations of the three facts occur, each once.
    assert len({facts for kinds in KINDS.values() for facts in kinds}) == 8


@pytest.mark.parametrize("size", [24, 48, 200])
@pytest.mark.parametrize("share", [0.0, 0.2, 0.5, 1.0])
def test_the_quotas_of_the_mixed_design_are_exact_in_every_split(cfg, size, share):
    mixed = dataclasses.replace(cfg, mix_facts=True, homonym_share=share, n_train=size, n_dev=size,
                                n_test=size, n_calib=size)
    for split in SPLITS:
        outcomes = split_outcomes(split, mixed)
        homonyms = split_homonyms(split, outcomes, mixed)
        facts = split_facts(split, outcomes, homonyms, mixed)
        assert len(facts) == size and Counter(outcomes) == {outcome: size // 4 for outcome in OUTCOMES}
        for outcome in OUTCOMES:
            of_outcome = [(f, h) for f, h, o in zip(facts, homonyms, outcomes) if o == outcome]
            counts = Counter(f for f, _ in of_outcome)
            wanted = list(QUOTAS[size // 4][outcome])
            assert [counts[kind] for kind in KINDS[outcome]] == wanted, (split, outcome)
            assert set(counts) <= set(KINDS[outcome])
            # The homonym flag is independent of the kind as far as whole numbers allow: of the c families
            # of a kind, c * (homonyms of the outcome) / (families of the outcome) have a homonym, rounded
            # down or up.
            n_homonyms = sum(h for _, h in of_outcome)
            with_homonym = Counter(f for f, h in of_outcome if h)
            for kind in KINDS[outcome]:
                exact = counts[kind] * n_homonyms / len(of_outcome)
                assert abs(with_homonym[kind] - exact) < 1, (split, outcome, kind, with_homonym[kind], exact)


def test_the_homonym_flags_of_each_kind_by_hand(cfg):
    """Train split of 200 with a homonym in half of the families: 25 of the 50 families of an outcome.
    inactive: 17, 17, 8, 8 families of the four kinds, so 8.5, 8.5, 4, 4 with a homonym -> 9, 8, 4, 4.
    already: 25, 25 -> 12.5, 12.5 -> 13, 12. Dev split of 48: 12 per outcome, 6 with a homonym: inactive
    4, 4, 2, 2 -> 2, 2, 1, 1; already 6, 6 -> 3, 3."""
    mixed = dataclasses.replace(cfg, mix_facts=True, homonym_share=0.5)
    wanted = {"train": {"inactive": [9, 8, 4, 4], "already": [13, 12], "approval": [25], "grant": [25]},
              "dev": {"inactive": [2, 2, 1, 1], "already": [3, 3], "approval": [6], "grant": [6]}}
    for split in wanted:
        outcomes = split_outcomes(split, mixed)
        homonyms = split_homonyms(split, outcomes, mixed)
        facts = split_facts(split, outcomes, homonyms, mixed)
        for outcome in OUTCOMES:
            with_homonym = Counter(f for f, h, o in zip(facts, homonyms, outcomes) if o == outcome and h)
            counted = [with_homonym[kind] for kind in KINDS[outcome]]
            assert counted == wanted[split][outcome], (split, outcome)


def facts_of_split(split: str, cfg) -> tuple[list[str], list]:
    """(the outcome of every family index, its facts in the mixed design)."""
    outcomes = split_outcomes(split, cfg)
    return outcomes, split_facts(split, outcomes, split_homonyms(split, outcomes, cfg), cfg)


def test_the_facts_are_chosen_by_the_seed_and_are_not_in_blocks(cfg):
    mixed = dataclasses.replace(cfg, mix_facts=True, homonym_share=0.5)
    outcomes, facts = facts_of_split("train", mixed)
    assert facts_of_split("train", mixed) == (outcomes, facts)
    # Train follows the data seed and the calibration split the calibration seed.
    other_data = dataclasses.replace(mixed, data_seed=mixed.data_seed + 1)
    other_calib = dataclasses.replace(mixed, calib_seed=mixed.calib_seed + 1)
    assert facts_of_split("train", other_data) != (outcomes, facts)
    assert facts_of_split("train", other_calib) == (outcomes, facts)
    assert facts_of_split("calib", other_data) == facts_of_split("calib", mixed)
    assert facts_of_split("calib", other_calib) != facts_of_split("calib", mixed)
    # Not in blocks: along the 50 inactive families the kind changes often (blocks would give 3 changes).
    inactive = [f for f, o in zip(facts, outcomes) if o == "inactive"]
    assert sum(a != b for a, b in zip(inactive, inactive[1:])) > 20


def test_with_the_switch_off_no_family_gets_facts_and_no_record_names_them(plain_families, plain_cfg):
    assert plain_cfg.mix_facts is False
    outcomes = split_outcomes("dev", plain_cfg)
    homonyms = split_homonyms("dev", outcomes, plain_cfg)
    assert split_facts("dev", outcomes, homonyms, plain_cfg) == [None] * 48
    assert not any("facts" in family for family in all_families(plain_families))


def test_a_mixed_family_is_the_default_family_with_other_facts(text, pools, cfg):
    """No random number is drawn for a fact. So with the facts of the default design the mixed family is
    the default family plus the key "facts"; with other facts only the three facts change: the requested
    role's flag, the pair (person meant, requested role) in the access list, the person's status never
    (the outcome fixes it), and what follows from them (expected states, turn budgets)."""
    for index, outcome in ((0, "inactive"), (1, "already"), (2, "grant"), (3, "approval")):
        default = make_family("dev", index, outcome, False, text, pools, cfg)
        same = make_family("dev", index, outcome, False, text, pools, cfg, facts=FACTS_OF_OUTCOME[outcome])
        assert list(same) == ["family_id", "split", "outcome", "homonym", "facts", "world_seed", "world",
                              "request", "truth", "expected", "turn_budget", "wordings"]
        assert same["facts"] == dict(zip(("active", "holds_role", "sensitive"), FACTS_OF_OUTCOME[outcome]))
        assert {key: value for key, value in same.items() if key != "facts"} == default
        assert "facts" not in default
    default = make_family("dev", 0, "inactive", False, text, pools, cfg)
    both = make_family("dev", 0, "inactive", False, text, pools, cfg, facts=(False, True, True))
    assert both["facts"] == {"active": False, "holds_role": True, "sensitive": True}
    for key in ("family_id", "split", "outcome", "homonym", "world_seed", "request", "truth", "wordings"):
        assert both[key] == default[key], key
    assert both["world"]["people"] == default["world"]["people"]
    pair = [default["truth"]["person_id"], default["truth"]["role_id"]]
    assert pair not in default["world"]["access"]
    assert both["world"]["access"] == sorted(default["world"]["access"] + [pair])
    flipped = [dict(role, sensitive=True) if role["role_id"] == pair[1] else role
               for role in default["world"]["roles"]]
    assert both["world"]["roles"] == flipped != default["world"]["roles"]
    # Still denied, with no write: the first rule that applies decides.
    assert both["expected"]["default"]["request"]["resolution"] == "denied"
    assert both["expected"]["default"]["access"] == both["world"]["access"]
    assert both["expected"]["default"]["approvals"] == both["expected"]["default"]["grants"] == []


def test_make_family_refuses_facts_that_are_not_a_kind_of_the_outcome(text, pools, cfg):
    for outcome, facts in (("inactive", (True, False, False)), ("grant", (True, False, True)),
                           ("already", (True, False, True)), ("approval", (True, True, True))):
        with pytest.raises(ValueError, match="not a kind of the outcome"):
            make_family("dev", 0, outcome, False, text, pools, cfg, facts=facts)


# ---------------------------------------------------------------------------
# One family: the world
# ---------------------------------------------------------------------------
def test_outcome_fixes_the_three_facts_and_never_mixes_them(plain_families):
    """Read straight from the stored world, without rules.py. (The design with the switch mix_facts off.)"""
    for family in all_families(plain_families):
        target, role = target_of(family), requested_role_of(family)
        has_role = [target["person_id"], role["role_id"]] in family["world"]["access"]
        facts = (target["status"] == "active", has_role, role["sensitive"])
        assert facts == FACTS_OF_OUTCOME[family["outcome"]], family["family_id"]


def test_request_names_the_target_and_only_the_target(families):
    for family in all_families(families):
        request, world = family["request"], family["world"]
        matches = [p for p in world["people"]
                   if (p["name"], p["department"]) == (request["person_name"], request["department"])]
        assert [p["person_id"] for p in matches] == [family["truth"]["person_id"]]
        roles = [r for r in world["roles"] if r["name"] == request["role_name"]]
        assert [r["role_id"] for r in roles] == [family["truth"]["role_id"]]


def test_world_size_and_text_sources(families, text, cfg):
    for family in all_families(families):
        world = family["world"]
        assert len(world["people"]) == cfg.n_people == 8
        assert len(world["roles"]) == cfg.n_roles == 6
        departments = {p["department"] for p in world["people"]}
        assert departments <= set(text.departments) and len(departments) <= cfg.n_departments == 3
        role_names = [r["name"] for r in world["roles"]]
        assert set(role_names) <= set(text.roles) and len(set(role_names)) == 6


def test_every_department_and_role_of_the_text_is_used_somewhere(families, text):
    used_departments = {p["department"] for f in all_families(families) for p in f["world"]["people"]}
    used_roles = {r["name"] for f in all_families(families) for r in f["world"]["roles"]}
    requested = {f["request"]["role_name"] for f in all_families(families)}
    assert used_departments == set(text.departments)
    assert used_roles == set(text.roles) == requested


def test_ids_have_the_prefix_and_six_characters_and_are_unique_in_the_world(families, text):
    token = f"[{TOKEN_ALPHABET}]{{6}}"
    for family in all_families(families):
        person_ids = [p["person_id"] for p in family["world"]["people"]]
        role_ids = [r["role_id"] for r in family["world"]["roles"]]
        assert all(re.fullmatch(re.escape(text.id_prefix_person) + token, i) for i in person_ids)
        assert all(re.fullmatch(re.escape(text.id_prefix_role) + token, i) for i in role_ids)
        assert len(set(person_ids + role_ids)) == 14


def test_access_is_sorted_unique_and_uses_ids_of_the_world(families):
    for family in all_families(families):
        world = family["world"]
        person_ids = {p["person_id"] for p in world["people"]}
        role_ids = {r["role_id"] for r in world["roles"]}
        assert world["access"] == sorted(world["access"])
        assert len({tuple(pair) for pair in world["access"]}) == len(world["access"])
        assert all(p in person_ids and r in role_ids for p, r in world["access"])


def test_names_come_from_the_pools_of_the_family_s_own_split(families, pools):
    for split in SPLITS:
        first_names, last_names = set(pools[split][0]), set(pools[split][1])
        full_names = {f"{first} {last}" for first in first_names for last in last_names}
        for family in families[split]:
            assert all(p["name"] in full_names for p in family["world"]["people"])


def test_full_names_are_unique_in_a_world_except_for_the_homonym_pair(families):
    for family in all_families(families):
        counts = Counter(p["name"] for p in family["world"]["people"])
        target_name = family["request"]["person_name"]
        if family["homonym"]:
            assert counts[target_name] == 2
            assert all(count == 1 for name, count in counts.items() if name != target_name)
        else:
            assert all(count == 1 for count in counts.values())


def homonym_of(family: dict) -> dict:
    """The other person with the target's full name (only for a family with a homonym)."""
    target = target_of(family)
    (other,) = [p for p in family["world"]["people"]
                if p["name"] == target["name"] and p["person_id"] != target["person_id"]]
    return other


def test_homonym_has_another_department(families, plain_families):
    # Both designs. The share 0.2 (before the freeze day): 10 + 2 + 2 + 1 homonym families per outcome.
    # The share 0.5 (the default since the freeze day): 25 + 6 + 6 + 3 per outcome.
    for built, wanted in ((plain_families, 4 * (10 + 2 + 2 + 1)), (families, 4 * (25 + 6 + 6 + 3))):
        seen = 0
        for family in all_families(built):
            if not family["homonym"]:
                continue
            assert homonym_of(family)["department"] != target_of(family)["department"]
            seen += 1
        assert seen == wanted


def test_manager_rules(families):
    for family in all_families(families):
        people = {p["person_id"]: p for p in family["world"]["people"]}
        target = target_of(family)
        assert family["truth"]["manager_id"] == target["manager_id"] is not None
        manager = people[target["manager_id"]]
        assert manager["status"] == "active"
        assert manager["person_id"] != target["person_id"]
        assert manager["name"] != target["name"]                          # so the manager is not the homonym
        assert all(p["manager_id"] is None or p["manager_id"] in people for p in people.values())
        assert all(p["manager_id"] != p["person_id"] for p in people.values())
        assert any(p["manager_id"] is None for p in people.values())
        for person in people.values():                                    # no cycle: every chain ends
            steps, current = 0, person
            while current["manager_id"] is not None:
                current = people[current["manager_id"]]
                steps += 1
                assert steps <= len(people), family["family_id"]


def test_the_target_s_manager_is_not_marked_by_a_missing_manager_of_its_own(families):
    """If the target's manager always had no manager, "the person without a manager" would be a shortcut."""
    with_manager = 0
    for family in all_families(families):
        people = {p["person_id"]: p for p in family["world"]["people"]}
        with_manager += people[family["truth"]["manager_id"]]["manager_id"] is not None
    assert 0.2 * 320 < with_manager < 0.8 * 320, with_manager


def test_target_and_requested_role_are_in_no_fixed_position(families):
    person_positions = Counter()
    role_positions = Counter()
    for family in families["train"]:
        person_ids = [p["person_id"] for p in family["world"]["people"]]
        role_ids = [r["role_id"] for r in family["world"]["roles"]]
        person_positions[person_ids.index(family["truth"]["person_id"])] += 1
        role_positions[role_ids.index(family["truth"]["role_id"])] += 1
    assert sorted(person_positions) == list(range(8)) and max(person_positions.values()) < 50   # mean is 25
    assert sorted(role_positions) == list(range(6)) and max(role_positions.values()) < 60       # mean is 33


def test_the_random_parts_of_the_world_have_the_specified_rates(families):
    active = sensitive = held = n_people = n_roles = n_pairs = 0
    for family in all_families(families):
        target_id, role_id = family["truth"]["person_id"], family["truth"]["role_id"]
        # Not drawn at the rate of 0.85: the target (fixed by the outcome), the manager (always active) and
        # the homonym (active with 0.75; see the test of the homonym's state below).
        fixed_people = {target_id, family["truth"]["manager_id"]}
        if family["homonym"]:
            fixed_people.add(homonym_of(family)["person_id"])
        others = [p for p in family["world"]["people"] if p["person_id"] not in fixed_people]
        active += sum(p["status"] == "active" for p in others)
        n_people += len(others)
        other_roles = [r for r in family["world"]["roles"] if r["role_id"] != role_id]
        sensitive += sum(r["sensitive"] for r in other_roles)
        n_roles += len(other_roles)
        held += sum(pair != [target_id, role_id] for pair in family["world"]["access"])
        n_pairs += 8 * 6 - 1
    assert 0.81 < active / n_people < 0.89, active / n_people              # 0.85; sd is about 0.008
    assert 0.35 < sensitive / n_roles < 0.45, sensitive / n_roles          # 0.40; sd is about 0.012
    assert 0.235 < held / n_pairs < 0.265, held / n_pairs                  # 0.25; sd is about 0.0035


def test_homonym_pair_is_in_no_fixed_order(families, plain_families):
    """If the target always came first of the two, "take the first match" would be a shortcut."""
    # Both designs. The share 0.2 (before the freeze day): 60 homonym families, 30 expected (the bounds
    # are from the time when a shuffle set the order; sd was about 3.9). The share 0.5 (the default since
    # the freeze day): 160 homonym families, 80 expected, with bounds of the same width in shares. The
    # exact numbers per split are pinned in
    # test_the_person_meant_is_the_first_match_in_half_of_the_homonym_families.
    for built, wanted, lowest, highest in ((plain_families, 60, 18, 42), (families, 160, 48, 112)):
        target_first = total = 0
        for family in all_families(built):
            if family["homonym"]:
                same_name = [p["person_id"] for p in family["world"]["people"]
                             if p["name"] == family["request"]["person_name"]]
                target_first += same_name[0] == family["truth"]["person_id"]
                total += 1
        assert total == wanted and lowest <= target_first <= highest, (total, target_first)


def test_a_missing_manager_does_not_tell_the_homonym_from_the_target(families, plain_families):
    """The target always has a manager. If its homonym could have none, "take the match that has a manager"
    would pick the right person without reading the department. (Found at integration: with a homonym that
    had no manager in 18 of 60 families, that rule picked the target in 41 of 60.)"""
    # Both designs: 60 homonym families at the share 0.2, and 160 at the share 0.5 (the default since the
    # freeze day).
    for built, wanted in ((plain_families, 60), (families, 160)):
        seen = 0
        for family in all_families(built):
            if family["homonym"]:
                same_name = [p for p in family["world"]["people"]
                             if p["name"] == family["request"]["person_name"]]
                assert len(same_name) == 2, family["family_id"]
                assert all(p["manager_id"] is not None for p in same_name), family["family_id"]
                seen += 1
        assert seen == wanted


# ---------------------------------------------------------------------------
# A homonym pair on a large sample: no simple rule finds the target without the department
# ---------------------------------------------------------------------------
N_HOMONYM_FAMILIES = 4000
WORLD_SIZES = {"small": {"n_people": 4, "n_roles": 2, "n_departments": 2}, "default": {}}


def homonym_families(text, pools, cfg, size: str) -> list[dict]:
    """4,000 families that all have a homonym, 1,000 of each outcome, as `make_family` builds them."""
    cfg = dataclasses.replace(cfg, **WORLD_SIZES[size])
    return [make_family("train", index, OUTCOMES[index % 4], True, text, pools, cfg)
            for index in range(N_HOMONYM_FAMILIES)]


def shortcut_scores(families: list[dict]) -> dict[str, tuple[int, int]]:
    """rule -> (pairs in which the rule can decide, pairs of these in which it picks the target).

    Each rule looks at the two people who have the requested name, in the order of the world (the order of
    a lookup result), and picks one of them without reading the department. A rule "can decide" in a pair
    if it points at exactly one of the two.
    """
    rules = ("inactive", "first", "has_manager", "holds_role", "manages_the_other")
    scores = {rule: [0, 0] for rule in rules}
    for family in families:
        target_id, role_id = family["truth"]["person_id"], family["truth"]["role_id"]
        wanted_name = family["request"]["person_name"]
        first, second = [p for p in family["world"]["people"] if p["name"] == wanted_name]
        holders = {person_id for person_id, held_role in family["world"]["access"] if held_role == role_id}
        picks = {     # rule -> for (first, second): does the rule point at this person?
            "inactive": [p["status"] == "inactive" for p in (first, second)],
            "first": [True, False],
            "has_manager": [p["manager_id"] is not None for p in (first, second)],
            "holds_role": [p["person_id"] in holders for p in (first, second)],
            "manages_the_other": [second["manager_id"] == first["person_id"],
                                  first["manager_id"] == second["person_id"]],
        }
        for rule, (at_first, at_second) in picks.items():
            if at_first != at_second:
                picked = first if at_first else second
                scores[rule][0] += 1
                scores[rule][1] += picked["person_id"] == target_id
    return {rule: (decided, right) for rule, (decided, right) in scores.items()}


def test_shortcut_scores_by_hand():
    def family(people: list[tuple], access: list[list[str]]) -> dict:
        return {"truth": {"person_id": "t", "role_id": "r"}, "request": {"person_name": "Ada Stone"},
                "world": {"access": access,
                          "people": [{"person_id": person_id, "name": name, "status": status,
                                      "manager_id": manager_id}
                                     for person_id, name, status, manager_id in people]}}
    # Family 1: the target "t" comes second, is the inactive one, has no manager, holds the role, and is
    # the manager of the other match. So: inactive right, first wrong, has_manager wrong, holds_role right,
    # manages_the_other right.
    one = family([("m", "Bo Reed", "active", None), ("h", "Ada Stone", "active", "t"),
                  ("t", "Ada Stone", "inactive", None)], [["t", "r"], ["h", "x"]])
    # Family 2: the target comes first; both are active, both have a manager, neither holds the role, and
    # neither manages the other. Only "first" can decide, and it is right.
    two = family([("t", "Ada Stone", "active", "m"), ("m", "Bo Reed", "inactive", None),
                  ("h", "Ada Stone", "active", "m")], [["m", "r"]])
    assert shortcut_scores([one, two]) == {"inactive": (1, 1), "first": (2, 1), "has_manager": (1, 0),
                                           "holds_role": (1, 1), "manages_the_other": (1, 1)}


@pytest.mark.parametrize("size", sorted(WORLD_SIZES))
def test_no_simple_rule_tells_the_target_from_its_homonym(text, pools, cfg, size):
    """SPEC_CHANGES section 5. Over 4,000 families with a homonym, each rule that picks one of the two
    matches without the department is right in about half of the pairs in which it can decide.

    Measured on 2026-11-17 (placeholder text, placeholder data seed 9001), as right / decided = share:
                                     small world (4 people, 2 roles)   default world (8 people, 6 roles)
        the inactive match           742 / 1500 = 0.495                765 / 1526 = 0.501
        the first match              1955 / 4000 = 0.489               1999 / 4000 = 0.500
        the match that holds the     743 / 1481 = 0.502                733 / 1535 = 0.478
          requested role
        the match that has a manager        never decides: both matches always have a manager
        the match that manages the          never decides: neither match is the manager of the other
          other match
    Before the change of section 5 a homonym was active whenever the target was inactive, and "the inactive
    match" was the target in every such pair. Before the homonym was kept from reporting to the target,
    "the match that manages the other match" decided in 242 (default) and 500 (small) of the 4,000 pairs
    and was right in all of them.
    """
    scores = shortcut_scores(homonym_families(text, pools, cfg, size))
    for rule, at_least in (("inactive", 1200), ("first", N_HOMONYM_FAMILIES), ("holds_role", 1200)):
        decided, right = scores[rule]
        assert decided >= at_least, (rule, decided)
        assert 0.44 <= right / decided <= 0.56, (rule, right, decided)
    assert scores["has_manager"] == (0, 0)
    assert scores["manages_the_other"] == (0, 0)


def test_the_homonym_is_inactive_in_a_quarter_of_the_families_whatever_the_target_is(text, pools, cfg):
    families = homonym_families(text, pools, cfg, "small")
    for outcome in OUTCOMES:
        of_outcome = [family for family in families if family["outcome"] == outcome]
        inactive = sum(homonym_of(family)["status"] == "inactive" for family in of_outcome)
        assert len(of_outcome) == 1000
        assert 0.21 <= inactive / 1000 <= 0.29, (outcome, inactive)       # 0.25; sd is about 0.014
    # Both states of both people occur together: an inactive target can have an inactive homonym.
    both_inactive = sum(family["outcome"] == "inactive" and homonym_of(family)["status"] == "inactive"
                        for family in families)
    assert 200 <= both_inactive <= 300                                    # 250 expected


def test_read_tools_show_the_facts_of_the_outcome_and_no_resolution_word(families, plain_families, text):
    """Through the real tools: the lookup and the access check return the three rule facts of the family,
    no read tool names a resolution, and no read tool changes the world. Both designs.
    With the switch mix_facts off the outcome fixes the three facts (FACTS_OF_OUTCOME, typed by hand). In
    the default design since the freeze day they are the "facts" of the record; their kinds are pinned by
    hand in tests/core/test_mixed_design.py."""
    keys = text.result_keys
    seen = Counter()
    for built in (plain_families, families):
        for family in all_families(built):
            world, request, truth = World(family["world"]), family["request"], family["truth"]
            lookup = execute(world, text, "lookup_person", {"name": request["person_name"]})
            matches = json.loads(lookup.response)[keys["matches"]]
            assert len(matches) == (2 if family["homonym"] else 1)
            meant = [m for m in matches if m[keys["department"]] == request["department"]]
            assert [m[keys["person_id"]] for m in meant] == [truth["person_id"]]
            assert meant[0][keys["manager_id"]] == truth["manager_id"]

            check = execute(world, text, "check_access",
                            {"person_id": truth["person_id"], "role_name": request["role_name"]})
            if built is plain_families:
                active, has_role, sensitive = FACTS_OF_OUTCOME[family["outcome"]]
            else:
                active, has_role, sensitive = (family["facts"][key]
                                               for key in ("active", "holds_role", "sensitive"))
                seen[(active, has_role, sensitive)] += 1
            assert meant[0][keys["status"]] == text.status["active" if active else "inactive"]
            assert json.loads(check.response) == {keys["role_id"]: truth["role_id"],
                                                  keys["sensitive"]: sensitive, keys["has_access"]: has_role}

            info = execute(world, text, "department_info", {"department": request["department"]})
            for outcome in (lookup, check, info):
                assert outcome.error is None
                assert not any(value in outcome.response
                               for value in text.resolution.values()), outcome.response
            assert world.state() == World(family["world"]).state()
    # The default design holds every one of the eight kinds of facts, so the loop above looked at each.
    assert len(seen) == 8 and sum(seen.values()) == sum(SIZES.values())


# ---------------------------------------------------------------------------
# One family: wordings
# ---------------------------------------------------------------------------
def test_train_has_three_styles_and_the_other_splits_have_four(families):
    for split in SPLITS:
        wanted = [1, 2, 3] if split == "train" else [1, 2, 3, 4]
        for family in families[split]:
            assert [wording["style"] for wording in family["wordings"]] == wanted
    assert not any(w["style"] == 4 for family in families["train"] for w in family["wordings"])


def test_wording_text_is_the_chosen_template_filled_with_the_request(families, text):
    for family in all_families(families):
        for wording in family["wordings"]:
            template = text.request_templates[wording["style"]][wording["template"]]
            assert wording["text"] == template.format(**family["request"])
            for value in family["request"].values():
                assert value in wording["text"]


def test_no_held_out_template_text_reaches_a_train_wording(families, text):
    """Fill the style-4 templates with each train request: no train wording may be equal to one of them."""
    for family in families["train"]:
        held_out = {template.format(**family["request"]) for template in text.request_templates[4]}
        assert not held_out & {wording["text"] for wording in family["wordings"]}


def test_every_template_of_every_style_is_used(families, text):
    used = Counter((w["style"], w["template"]) for f in all_families(families) for w in f["wordings"])
    wanted = {(style, index) for style, templates in text.request_templates.items()
              for index in range(len(templates))}
    assert set(used) == wanted
    train = Counter((w["style"], w["template"]) for family in families["train"] for w in family["wordings"])
    assert all(40 < train[key] < 95 for key in wanted if key[0] != 4), train        # 200 / 3 is about 67


def test_scenario_ids(families):
    family = families["dev"][7]
    assert scenario_ids(family) == ["dv-07/s1", "dv-07/s2", "dv-07/s3", "dv-07/s4"]
    assert scenario_ids(families["train"][123]) == ["tr-123/s1", "tr-123/s2", "tr-123/s3"]
    assert split_scenario_id("dv-07/s4") == ("dv-07", 4)
    assert split_scenario_id("tr-123/s2") == ("tr-123", 2)
    all_ids = [sid for f in all_families(families) for sid in scenario_ids(f)]
    assert len(set(all_ids)) == len(all_ids) == 600 + 192 + 192 + 96
    for f in all_families(families):
        pairs = [(f["family_id"], w["style"]) for w in f["wordings"]]
        assert [split_scenario_id(sid) for sid in scenario_ids(f)] == pairs


@pytest.mark.parametrize("bad", ["dv-07", "dv-07/s", "dv-07/sx", "/s2", "dv-07/2", "", "dv-07/s2/extra"])
def test_split_scenario_id_refuses_other_strings(bad):
    with pytest.raises(ValueError):
        split_scenario_id(bad)


# ---------------------------------------------------------------------------
# No leak between splits
# ---------------------------------------------------------------------------
def name_parts(families: list[dict], pool: tuple) -> tuple[set[str], set[str], set[str]]:
    """(first names, last names, full names) used by the families; names are split with the split's pool."""
    first_names, last_names = pool
    full = {p["name"] for family in families for p in family["world"]["people"]}
    firsts = {first for first in first_names for name in full if name.startswith(first + " ")}
    lasts = {last for last in last_names for name in full if name.endswith(" " + last)}
    return firsts, lasts, full


def test_no_name_seed_or_family_id_is_in_two_splits(families, pools):
    parts = {split: name_parts(families[split], pools[split]) for split in SPLITS}
    seeds = {split: {f["world_seed"] for f in families[split]} for split in SPLITS}
    ids = {split: {f["family_id"] for f in families[split]} for split in SPLITS}
    for i, a in enumerate(SPLITS):
        assert len(parts[a][0]) > 20 and len(parts[a][1]) > 20 and len(parts[a][2]) > 100   # not empty
        for b in SPLITS[i + 1:]:
            assert not parts[a][0] & parts[b][0], f"first names shared by {a} and {b}"
            assert not parts[a][1] & parts[b][1], f"last names shared by {a} and {b}"
            assert not parts[a][2] & parts[b][2], f"full names shared by {a} and {b}"
            assert not seeds[a] & seeds[b], f"world seeds shared by {a} and {b}"
            assert not ids[a] & ids[b], f"family ids shared by {a} and {b}"


def test_no_name_of_another_split_appears_in_any_text_of_a_family(families, pools):
    """Stronger than the pool check: search the names of every family for the names of the other splits."""
    for split in SPLITS:
        foreign = [name for other in SPLITS if other != split for kind in (0, 1)
                   for name in pools[other][kind]]
        own = set(pools[split][0]) | set(pools[split][1])
        foreign = [name for name in foreign if name not in own]       # a first name may also be a last name
        pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in foreign) + r")\b")
        for family in families[split]:
            for person in family["world"]["people"]:
                assert not pattern.search(person["name"]), (family["family_id"], person["name"])
            assert not pattern.search(family["request"]["person_name"])


def test_the_leak_check_can_fail(families, pools):
    """Put one dev name into a train family and see that the checks above would catch it."""
    leaked = copy.deepcopy(families["train"][:5])
    dev_person = families["dev"][0]["world"]["people"][0]["name"]
    leaked[0]["world"]["people"][0]["name"] = dev_person
    train_full = name_parts(leaked, pools["train"])[2]
    dev_full = name_parts(families["dev"], pools["dev"])[2]
    assert train_full & dev_full == {dev_person}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_same_inputs_give_byte_identical_families_in_one_process(families, text, pools, cfg):
    again = make_all_splits(text, pools, cfg)
    for split in SPLITS:
        assert to_bytes(again[split]) == to_bytes(families[split])


def test_a_family_does_not_depend_on_the_families_made_before_it(families, plain_families, text, pools,
                                                                 cfg, plain_cfg):
    for built, config in ((plain_families, plain_cfg), (families, cfg)):
        for split, index in (("train", 137), ("dev", 0), ("test", 47), ("calib", 5)):
            stored = built[split][index]
            # What a family takes from its split: where the person meant stands beside a homonym, and in
            # the mixed design its three rule facts (None with the switch off).
            outcomes = split_outcomes(split, config)
            homonyms = split_homonyms(split, outcomes, config)
            place = split_homonym_places(split, outcomes, homonyms, config)[index]
            facts = split_facts(split, outcomes, homonyms, config)[index]
            assert (facts is None) is (config is plain_cfg)
            alone = make_family(split, index, stored["outcome"], stored["homonym"], text, pools, config,
                                target_first=place, facts=facts)
            assert json.dumps(alone) == json.dumps(stored)


def test_seeds_in_the_config_change_the_right_splits(families, text, pools, cfg):
    other_data = make_all_splits(text, pools, dataclasses.replace(cfg, data_seed=cfg.data_seed + 100))
    other_calib = make_all_splits(text, pools, dataclasses.replace(cfg, calib_seed=cfg.calib_seed + 100))
    for split in ("train", "dev", "test"):
        assert to_bytes(other_data[split]) != to_bytes(families[split])
        assert to_bytes(other_calib[split]) == to_bytes(families[split])
    assert to_bytes(other_data["calib"]) == to_bytes(families["calib"])
    assert to_bytes(other_calib["calib"]) != to_bytes(families["calib"])


def test_make_family_checks_its_arguments(text, pools, cfg):
    with pytest.raises(ValueError):
        make_family("dev", 0, "maybe", False, text, pools, cfg)
    with pytest.raises(ValueError):
        make_family("dev", 0, "grant", True, text, pools, dataclasses.replace(cfg, n_people=2))
    with pytest.raises(ValueError):
        make_family("dev", 0, "grant", False, text, pools, dataclasses.replace(cfg, n_departments=1))


def test_generated_text_comes_from_the_text_bundle(marker_text, pools, cfg):
    """With the marker bundle, every department, role name, id prefix and wording is a marker."""
    marker = re.compile(MARKER_PATTERN)
    for family in make_split("dev", marker_text, pools, cfg):
        world = family["world"]
        assert all(marker.fullmatch(p["department"]) for p in world["people"])
        assert all(marker.fullmatch(r["name"]) for r in world["roles"])
        assert all(marker.match(p["person_id"]) for p in world["people"])
        assert all(marker.match(r["role_id"]) for r in world["roles"])
        request = family["request"]
        for wording in family["wordings"]:
            wanted = " ".join([MARKER_PATTERN, re.escape(request["person_name"]), request["department"],
                               request["role_name"]])
            assert re.fullmatch(wanted, wording["text"])


# ---------------------------------------------------------------------------
# Family files
# ---------------------------------------------------------------------------
def test_save_and_load_families(tmp_path, families):
    path = tmp_path / "dev.jsonl"
    save_families(path, families["dev"])
    raw = path.read_bytes()
    assert raw == to_bytes(families["dev"])
    assert raw.count(b"\n") == 48 and raw.endswith(b"}\n") and b"\r" not in raw
    loaded = load_families(path)
    assert loaded == families["dev"]
    assert [list(family) for family in loaded] == [list(family) for family in families["dev"]]   # key order


def test_family_files_keep_non_ascii_text_readable(tmp_path):
    path = tmp_path / "x.jsonl"
    save_families(path, [{"family_id": "x", "name": "Zoë Müller"}])
    assert "Zoë Müller".encode("utf-8") in path.read_bytes()
    assert load_families(path) == [{"family_id": "x", "name": "Zoë Müller"}]


# ---------------------------------------------------------------------------
# The fixed dev subsets
# ---------------------------------------------------------------------------
def test_make_sets_shape_and_sizes(families, cfg, text):
    sets = make_sets(families["dev"], cfg, text.policy_shift_variant)
    by_id = {family["family_id"]: family for family in families["dev"]}
    assert list(sets) == ["curve_subset", "policy_shift", "extra_tool"]
    assert list(sets["policy_shift"]) == ["variant", "differ", "control"]
    assert sets["policy_shift"]["variant"] == "all_grants_need_approval"
    lists = {"curve_subset": sets["curve_subset"], "extra_tool": sets["extra_tool"],
             "differ": sets["policy_shift"]["differ"], "control": sets["policy_shift"]["control"]}
    sizes = {name: len(ids) for name, ids in lists.items()}
    assert sizes == {"curve_subset": 20, "extra_tool": 16, "differ": 11, "control": 5}
    for name, ids in lists.items():
        assert ids == sorted(set(ids)), name
        assert set(ids) <= set(by_id), name
    assert Counter(by_id[i]["outcome"] for i in sets["curve_subset"]) == {outcome: 5 for outcome in OUTCOMES}
    assert Counter(by_id[i]["outcome"] for i in sets["extra_tool"]) == {outcome: 4 for outcome in OUTCOMES}
    assert json.loads(json.dumps(sets)) == sets


def test_policy_shift_sets_differ_and_control(families, cfg):
    by_id = {family["family_id"]: family for family in families["dev"]}
    # Rule set -> the outcome whose families differ, and the outcomes of the 5 controls. Under
    # all_grants_need_approval no control is an approval family (the constant policy "always ask for
    # approval" passes those); under sensitive_denied none is an inactive family ("always deny").
    cases = (("all_grants_need_approval", "grant", {"already": 3, "inactive": 2}),
             ("sensitive_denied", "approval", {"grant": 3, "already": 2}))
    for variant, differing_outcome, controls in cases:
        shift = make_sets(families["dev"], cfg, variant)["policy_shift"]
        assert shift["variant"] == variant
        for family_id in shift["differ"]:
            family = by_id[family_id]
            assert family["expected"][variant] != family["expected"]["default"]
            assert family["outcome"] == differing_outcome
        for family_id in shift["control"]:
            family = by_id[family_id]
            assert family["expected"][variant] == family["expected"]["default"]
            # The constant policy that fits the changed rule closes with the new resolution of the
            # "differ" families. A control has another correct resolution, so that policy fails on it.
            new_resolution = by_id[shift["differ"][0]]["expected"][variant]["request"]["resolution"]
            assert family["expected"]["default"]["request"]["resolution"] != new_resolution
        assert Counter(by_id[i]["outcome"] for i in shift["control"]) == controls       # 3 and 2
        assert not set(shift["differ"]) & set(shift["control"])


def test_make_sets_default_variant_is_the_placeholder_s(families, cfg, text):
    assert make_sets(families["dev"], cfg) == make_sets(families["dev"], cfg, text.policy_shift_variant)


def test_make_sets_is_seeded(families, cfg):
    first = make_sets(families["dev"], cfg)
    assert make_sets(list(families["dev"]), cfg) == first
    other = make_sets(families["dev"], dataclasses.replace(cfg, data_seed=cfg.data_seed + 1))
    assert other["curve_subset"] != first["curve_subset"]
    assert other["extra_tool"] != first["extra_tool"]
    assert first["curve_subset"][:5] != [f"dv-{i:02d}" for i in range(5)]     # not simply the first families


def test_make_sets_refuses_bad_input(families, cfg):
    with pytest.raises(ValueError):
        make_sets(families["dev"], cfg, "default")
    with pytest.raises(ValueError):
        make_sets(families["dev"], cfg, "no_such_rule_set")
    grants = [family for family in families["dev"] if family["outcome"] == "grant"]
    others = [family for family in families["dev"] if family["outcome"] != "grant"]
    with pytest.raises(ValueError, match="policy_shift_differ"):
        make_sets(grants[:10] + others, cfg)                                   # only 10 families differ
    with pytest.raises(ValueError, match="curve_subset"):
        make_sets(families["dev"][:8], cfg)                                    # fewer than 5 of some outcome


# ---------------------------------------------------------------------------
# scripts/build_scenarios.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built(tmp_path_factory, text, cfg):
    """One build with made-up names (an empty names folder), in this process."""
    module = load_build_script()
    out_dir = tmp_path_factory.mktemp("built")
    names_dir = tmp_path_factory.mktemp("no_names")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = module.build(text, names_dir, out_dir, cfg)
    return out_dir, summary


def test_build_writes_the_six_files(built):
    out_dir, _ = built
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(BUILT_FILES)


def test_build_summary(built, text, cfg):
    out_dir, summary = built
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) == summary
    assert summary["text_file"] == "placeholder_text.yaml"
    assert summary["text_sha256"] == text.sha256 and summary["text_is_placeholder"] is True
    assert summary["name_files_sha256"] == {"first_names.txt": None, "last_names.txt": None}
    assert summary["config"] == cfg.to_dict() and summary["config_sha256"] == cfg.sha256()
    for split in SPLITS:
        counts = dict(summary["splits"][split])
        first = counts.pop("homonyms_person_meant_first")       # tested with the homonym places below
        # The default design since the freeze day: the mixed design, so the summary counts the kinds
        # (test_the_build_options_set_the_config_of_one_build pins them), and a homonym in half of the
        # families of each outcome.
        kinds, kinds_with_homonym = counts.pop("kinds"), counts.pop("kinds_with_homonym")
        assert counts == {
            "families": SIZES[split], "scenarios": SCENARIOS[split],
            "outcomes": {outcome: SIZES[split] // 4 for outcome in OUTCOMES},
            "homonyms": {outcome: SIZES[split] // 8 for outcome in OUTCOMES},
        }
        assert list(first) == list(OUTCOMES) == list(kinds) == list(kinds_with_homonym)
        for outcome in OUTCOMES:
            assert sum(kinds[outcome].values()) == SIZES[split] // 4
            assert sum(kinds_with_homonym[outcome].values()) == SIZES[split] // 8
    assert sorted(summary["files_sha256"]) == sorted(set(BUILT_FILES) - {"summary.json"})
    for name, digest in summary["files_sha256"].items():
        assert digest == hashlib.sha256((out_dir / name).read_bytes()).hexdigest()


def test_build_summary_names_no_path_of_this_machine(built):
    out_dir, _ = built
    raw = (out_dir / "summary.json").read_text(encoding="utf-8")
    assert str(ROOT) not in raw and str(out_dir) not in raw and "/home/" not in raw and "/tmp" not in raw


def test_build_hashes_real_name_files(tmp_path, text, cfg):
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    write_name_files(names_dir)
    summary = load_build_script().build(text, names_dir, tmp_path / "out", cfg)
    for name in ("first_names.txt", "last_names.txt"):
        digest = hashlib.sha256((names_dir / name).read_bytes()).hexdigest()
        assert summary["name_files_sha256"][name] == digest
    people = load_families(tmp_path / "out" / "dev.jsonl")[0]["world"]["people"]
    first_names, last_names = load_name_pools(names_dir)["dev"]
    assert len(people) == 8                                         # names of the given files, dev pool only
    names = [p["name"].split(" ") for p in people]
    assert all(first in first_names and last in last_names for first, last in names)


def test_built_files_equal_the_families_made_in_memory(built, text, cfg):
    out_dir, _ = built
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pools = load_name_pools(out_dir / "no_such_folder")
    families = make_all_splits(text, pools, cfg)
    for split in SPLITS:
        assert load_families(out_dir / f"{split}.jsonl") == families[split]
    sets = json.loads((out_dir / "sets.json").read_text(encoding="utf-8"))
    assert sets == make_sets(families["dev"], cfg, text.policy_shift_variant)


def test_building_twice_in_one_process_gives_byte_identical_files(built, tmp_path, text, cfg):
    out_dir, _ = built
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_build_script().build(text, tmp_path / "no_names", tmp_path / "again", cfg)
    for name in BUILT_FILES:
        assert (tmp_path / "again" / name).read_bytes() == (out_dir / name).read_bytes(), name


def run_script(out_dir: Path, names_dir: Path, hash_seed: str, *extra: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    command = [sys.executable, str(BUILD_SCRIPT), "--out", str(out_dir), "--names", str(names_dir), *extra]
    return subprocess.run(command, env=env, capture_output=True, text=True, cwd=str(out_dir.parent),
                          timeout=120)


def test_two_processes_with_different_hash_seeds_give_byte_identical_files(built, tmp_path):
    out_dir, _ = built
    names_dir = tmp_path / "no_names"
    names_dir.mkdir()
    for hash_seed in ("1", "2"):
        result = run_script(tmp_path / f"run{hash_seed}", names_dir, hash_seed, "--allow-made-up-names")
        assert result.returncode == 0, result.stderr
    for name in BUILT_FILES:
        first = (tmp_path / "run1" / name).read_bytes()
        assert first == (tmp_path / "run2" / name).read_bytes(), name
        assert first == (out_dir / name).read_bytes(), name                # and equal to the in-process build


def test_build_script_stops_if_a_name_file_is_missing(tmp_path):
    names_dir = tmp_path / "no_names"
    names_dir.mkdir()
    result = run_script(tmp_path / "out", names_dir, "0")
    assert result.returncode == 2
    assert "first_names.txt" in result.stderr and "--allow-made-up-names" in result.stderr
    assert not (tmp_path / "out").exists()


def test_build_script_default_out_dir_follows_the_text_file():
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert '"scenarios_placeholder" if text.is_placeholder else "scenarios"' in source


def test_scenario_files_in_the_repository_are_up_to_date(tmp_path, text, cfg):
    """data/scenarios_placeholder/ must be what the build script writes now from the text and name files."""
    stored = ROOT / "data" / "scenarios_placeholder"
    if not stored.is_dir():
        pytest.skip("data/scenarios_placeholder/ has not been built yet")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_build_script().build(text, ROOT / "data" / "names", tmp_path / "fresh", cfg)
    for name in BUILT_FILES:
        assert (stored / name).read_bytes() == (tmp_path / "fresh" / name).read_bytes(), (
            f"data/scenarios_placeholder/{name} is stale: run scripts/build_scenarios.py again")


# ---------------------------------------------------------------------------
# Where the person meant stands beside a homonym (finding of 2026-11-17)
# ---------------------------------------------------------------------------
def person_meant_is_the_first_match(family: dict) -> bool:
    """True if, of the people with the name of the request, the first in the world's list is the one meant.
    `lookup_person` returns the matches in this order."""
    name = family["request"]["person_name"]
    matches = [person for person in family["world"]["people"] if person["name"] == name]
    return matches[0]["person_id"] == family["truth"]["person_id"]


def assert_places_are_balanced(families_of_split: list[dict]) -> None:
    homonym_families = [family for family in families_of_split if family["homonym"]]
    for outcome in OUTCOMES:
        group = [family for family in homonym_families if family["outcome"] == outcome]
        n_first = sum(person_meant_is_the_first_match(family) for family in group)
        assert n_first in (len(group) // 2, (len(group) + 1) // 2), (outcome, n_first, len(group))
    n_first = sum(person_meant_is_the_first_match(family) for family in homonym_families)
    assert abs(2 * n_first - len(homonym_families)) <= 1, (n_first, len(homonym_families))


def test_the_person_meant_is_the_first_match_in_half_of_the_homonym_families(families, plain_families):
    """Left to the shuffle, the person meant was the first match in 6 of the 8 homonym families of dev and
    in 1 of 8 of test (data seed 1): a model with the habit "take the first match" would have had a gap
    between dev and test that no training made. The place is balanced per outcome and per split."""
    for split in SPLITS:
        assert_places_are_balanced(plain_families[split])
        assert_places_are_balanced(families[split])
    # The share 0.2. dev and test: 2 homonym families per outcome, 1 with the person meant first.
    for split in ("dev", "test"):
        assert sum(person_meant_is_the_first_match(f) for f in plain_families[split] if f["homonym"]) == 4
    assert sum(person_meant_is_the_first_match(f) for f in plain_families["train"] if f["homonym"]) == 20
    assert sum(person_meant_is_the_first_match(f) for f in plain_families["calib"] if f["homonym"]) == 2
    # The share 0.5, the default since the freeze day: the numbers of FIRST_AT_SHARE.
    for split in SPLITS:
        assert sum(person_meant_is_the_first_match(f) for f in families[split] if f["homonym"]) == \
            FIRST_AT_SHARE[0.5][split]


@pytest.mark.parametrize("seed", [5, 6, 7, 8])
def test_the_places_are_balanced_for_other_seeds_and_odd_numbers_of_homonym_families(text, pools, cfg, seed):
    # 20 dev families: 5 per outcome, 1 homonym family each (odd). 28 test families: 7 per outcome,
    # round(1.4) = 1 each. 24 with a share of 0.5: 3 homonym families per outcome (odd, more than one).
    for changes in ({"n_dev": 20, "n_test": 28}, {"n_dev": 24, "n_test": 24, "homonym_share": 0.5}):
        other = dataclasses.replace(cfg, data_seed=seed, calib_seed=seed + 100, **changes)
        for split in ("dev", "test", "calib"):
            assert_places_are_balanced(make_split(split, text, pools, other))


def test_the_place_is_given_to_homonym_families_only_and_moves_nothing_else(text, pools, cfg):
    outcomes = split_outcomes("train", cfg)
    homonyms = split_homonyms("train", outcomes, cfg)
    places = split_homonym_places("train", outcomes, homonyms, cfg)
    assert [place is not None for place in places] == homonyms
    assert all(isinstance(place, bool) for place in places if place is not None)
    n_swapped = 0
    for index in [i for i, flag in enumerate(homonyms) if flag][:12]:
        made = {place: make_family("train", index, outcomes[index], True, text, pools, cfg,
                                   target_first=place) for place in (True, False, None)}
        assert person_meant_is_the_first_match(made[True])
        assert not person_meant_is_the_first_match(made[False])
        # The shuffle alone gives one of the two; the other differs from it only in that the two people
        # with the same name have changed places.
        same = True if json.dumps(made[None]) == json.dumps(made[True]) else False
        assert json.dumps(made[None]) == json.dumps(made[same])
        people, moved = made[same]["world"]["people"], made[not same]["world"]["people"]
        differ = [i for i, (a, b) in enumerate(zip(people, moved)) if a != b]
        assert len(differ) == 2
        assert people[differ[0]] == moved[differ[1]] and people[differ[1]] == moved[differ[0]]
        assert people[differ[0]]["name"] == people[differ[1]]["name"] == made[same]["request"]["person_name"]
        for key in made[same]:
            if key != "world":
                assert made[same][key] == made[not same][key]
        assert made[same]["world"]["roles"] == made[not same]["world"]["roles"]
        assert made[same]["world"]["access"] == made[not same]["world"]["access"]
        n_swapped += same is not places[index]
    assert 0 < n_swapped < 12                 # the balanced places really changed some families, not all
    # A family without a homonym does not look at the argument.
    index = homonyms.index(False)
    plain = [json.dumps(make_family("train", index, outcomes[index], False, text, pools, cfg, target_first=p))
             for p in (True, False, None)]
    assert len(set(plain)) == 1


def test_the_build_summary_counts_the_homonym_families_with_the_person_meant_first(built):
    out_dir, summary = built
    for split in SPLITS:
        families_of_split = load_families(out_dir / f"{split}.jsonl")
        counted = summary["splits"][split]["homonyms_person_meant_first"]
        assert counted == {outcome: sum(person_meant_is_the_first_match(f) for f in families_of_split
                                        if f["homonym"] and f["outcome"] == outcome) for outcome in OUTCOMES}
        assert abs(2 * sum(counted.values()) - sum(summary["splits"][split]["homonyms"].values())) <= 1


# ---------------------------------------------------------------------------
# The final build needs a fresh seed (finding of 2026-11-17)
# ---------------------------------------------------------------------------
def test_the_final_build_refuses_families_whose_world_seeds_were_already_used(tmp_path, text, cfg):
    """The calibration folders (data/scenarios_calib1, data/scenarios_calib2_*) were built with the data seed
    that the config has until the freeze, read during the work and rolled out. The final folder
    (`scenarios`) must not hold their train, dev or test worlds again; before the repair nothing noticed a
    final build with the same seed."""
    module = load_build_script()
    names_dir = ROOT / "data" / "names"
    module.build(text, names_dir, tmp_path / "scenarios_calib1", cfg)          # an earlier folder
    with pytest.raises(ValueError, match=r"not new: 296 of the 296 world seeds .*scenarios_calib1"):
        module.build(text, names_dir, tmp_path / "scenarios", cfg)
    assert not (tmp_path / "scenarios").exists()                               # nothing was written
    # With a fresh data seed the final build goes through (the calibration seed may stay).
    fresh = dataclasses.replace(cfg, data_seed=cfg.data_seed + 1)
    summary = module.build(text, names_dir, tmp_path / "scenarios", fresh)
    assert summary["config"]["data_seed"] == cfg.data_seed + 1
    assert sorted(path.name for path in (tmp_path / "scenarios").iterdir()) == sorted(BUILT_FILES)
    # The rule is for the final folder only: a folder with another name is built whatever is beside it.
    module.build(text, names_dir, tmp_path / "scenarios_placeholder", cfg)
    assert module.earlier_dirs(tmp_path / "scenarios_placeholder") == []
    assert [path.name for path in module.earlier_dirs(tmp_path / "scenarios")] == \
        ["scenarios_calib1", "scenarios_placeholder"]


def test_the_build_script_stops_with_code_2_when_the_final_families_are_not_new(tmp_path, capsys):
    module = load_build_script()
    assert module.main(["--out", str(tmp_path / "scenarios_placeholder")]) == 0
    capsys.readouterr()
    assert module.main(["--out", str(tmp_path / "scenarios")]) == 2
    assert "Change data_seed in accessdesk/config.py" in capsys.readouterr().err
    assert not (tmp_path / "scenarios").exists()


# ---------------------------------------------------------------------------
# Calibration round 2: the options of the build script, the placeholder seeds, the unchanged default design
# ---------------------------------------------------------------------------
def summary_of(folder: Path) -> dict:
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"))


def real_text_file(folder: Path) -> Path:
    """A text file that is not the placeholder: the placeholder's content with `meta.placeholder: false`."""
    data = yaml.safe_load(placeholder_path().read_text(encoding="utf-8"))
    data["meta"]["placeholder"] = False
    path = folder / "not_the_placeholder.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_the_build_options_set_the_config_of_one_build(tmp_path, capsys):
    """The command of data/scenarios_calib2_mixed, with the placeholder text."""
    module = load_build_script()
    out_dir = tmp_path / "scenarios_mixed"
    options = ["--mix-facts", "--homonym-share", "0.5", "--n-calib", "48"]
    assert module.main(["--out", str(out_dir), *options]) == 0
    assert "mix_facts True" in capsys.readouterr().out
    summary = summary_of(out_dir)
    wanted = dataclasses.replace(RunConfig(), mix_facts=True, homonym_share=0.5, n_calib=48,
                                 data_seed=9001, calib_seed=9002)
    assert summary["config"] == wanted.to_dict() and summary["config_sha256"] == wanted.sha256()
    assert list(summary["config"]) == [field.name for field in dataclasses.fields(RunConfig)]   # every field
    calib = summary["splits"]["calib"]
    assert (calib["families"], calib["scenarios"]) == (48, 192)
    assert calib["outcomes"] == {outcome: 12 for outcome in OUTCOMES}
    assert calib["homonyms"] == {outcome: 6 for outcome in OUTCOMES}
    assert calib["homonyms_person_meant_first"] == {outcome: 3 for outcome in OUTCOMES}
    assert calib["kinds"] == {"grant": {"plain": 12}, "approval": {"plain": 12},
                              "already": {"plain": 6, "sensitive": 6},
                              "inactive": {"plain": 4, "sensitive": 4, "held": 2, "held+sensitive": 2}}
    assert calib["kinds_with_homonym"] == {"grant": {"plain": 6}, "approval": {"plain": 6},
                                           "already": {"plain": 3, "sensitive": 3},
                                           "inactive": {"plain": 2, "sensitive": 2, "held": 1,
                                                        "held+sensitive": 1}}
    assert summary["splits"]["train"]["kinds"]["inactive"] == {"plain": 17, "sensitive": 17, "held": 8,
                                                               "held+sensitive": 8}
    families_of_calib = load_families(out_dir / "calib.jsonl")
    assert len(families_of_calib) == 48 and all("facts" in family for family in families_of_calib)
    assert families_of_calib[47]["family_id"] == "ca-47"
    # The summary's kinds are counted from the facts of the stored worlds.
    for family in families_of_calib:
        target, role = target_of(family), requested_role_of(family)
        facts = {"active": target["status"] == "active",
                 "holds_role": [target["person_id"], role["role_id"]] in family["world"]["access"],
                 "sensitive": role["sensitive"]}
        assert family["facts"] == facts
        assert first_rule_that_applies(*facts.values()) == family["outcome"]


def test_an_option_wins_over_the_config_and_a_missing_option_leaves_it(tmp_path, capsys, monkeypatch):
    """The commands of round 2 name every option, so that they give the same folders after the defaults
    of accessdesk/config.py change. Shown with a config whose default is the mixed design."""
    module = load_build_script()
    monkeypatch.setattr(module, "RunConfig", lambda: RunConfig(mix_facts=True, homonym_share=0.5))
    for options, mix_facts, share in (([], True, 0.5), (["--mix-facts"], True, 0.5),
                                      (["--no-mix-facts", "--homonym-share", "0.2"], False, 0.2)):
        out_dir = tmp_path / f"scenarios_{len(options)}"
        assert module.main(["--out", str(out_dir), *options]) == 0
        config = summary_of(out_dir)["config"]
        assert (config["mix_facts"], config["homonym_share"]) == (mix_facts, share), options
        assert ("facts" in load_families(out_dir / "calib.jsonl")[0]) is mix_facts
    capsys.readouterr()


def test_a_build_without_options_is_the_default_design(built, tmp_path, text, plain_cfg):
    """The default design since the freeze day is the final one: rule facts may be true together, and half
    of the families hold two people of one name. Every family has "facts", and the summary counts the
    kinds (test_build_summary pins the summary's counts).
    The design before that day, with its two values set by hand: no "facts" in a family, no "kinds" in the
    summary."""
    out_dir, summary = built
    assert summary["config"]["mix_facts"] is True and summary["config"]["homonym_share"] == 0.5
    for split in SPLITS:
        assert {"kinds", "kinds_with_homonym"} <= set(summary["splits"][split])
        assert all("facts" in family for family in load_families(out_dir / f"{split}.jsonl"))
    (tmp_path / "no_names").mkdir()                 # made-up names, as in the fixture `built`
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = load_build_script().build(text, tmp_path / "no_names", tmp_path / "plain", plain_cfg)
    assert plain["config"]["mix_facts"] is False and plain["config"]["homonym_share"] == 0.2
    for split in SPLITS:
        assert not {"kinds", "kinds_with_homonym"} & set(plain["splits"][split])
        assert not any("facts" in family for family in load_families(tmp_path / "plain" / f"{split}.jsonl"))


@pytest.mark.parametrize("options, wanted", [
    (["--n-calib", "50"], "multiple of 4"),
    (["--n-calib", "104"], "100 at most"),
    (["--n-calib", "-4"], "multiple of 4"),
    (["--homonym-share", "1.5"], "between 0 and 1"),
])
def test_the_build_script_refuses_options_that_the_generator_cannot_use(tmp_path, capsys, options, wanted):
    assert load_build_script().main(["--out", str(tmp_path / "x"), *options]) == 2
    assert wanted in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_placeholder_builds_have_seeds_of_their_own(tmp_path, capsys):
    """A build from the placeholder text takes the placeholder seeds; a build from another text file takes
    data_seed and calib_seed. So no world of the unit-test data is a world of a build from the real text:
    the two builds share no world seed, no id and no family."""
    module = load_build_script()
    assert module.main(["--out", str(tmp_path / "from_placeholder")]) == 0
    real_text = str(real_text_file(tmp_path))
    assert module.main(["--out", str(tmp_path / "from_real_text"), "--text", real_text]) == 0
    capsys.readouterr()
    placeholder, real = summary_of(tmp_path / "from_placeholder"), summary_of(tmp_path / "from_real_text")
    assert (placeholder["text_is_placeholder"], real["text_is_placeholder"]) == (True, False)
    assert (placeholder["config"]["data_seed"], placeholder["config"]["calib_seed"]) == (9001, 9002)
    assert (real["config"]["data_seed"], real["config"]["calib_seed"]) == (20261118, 20261119)
    assert real["config"] == RunConfig().to_dict()
    assert placeholder["config"] == RunConfig().with_seeds_for(True).to_dict()
    seen: dict[str, set] = {"from_placeholder": set(), "from_real_text": set()}
    for folder, values in seen.items():
        for split in SPLITS:
            for family in load_families(tmp_path / folder / f"{split}.jsonl"):
                values.add(family["world_seed"])
                values.update(person["person_id"] for person in family["world"]["people"])
                values.update(role["role_id"] for role in family["world"]["roles"])
    assert len(seen["from_placeholder"]) == len(seen["from_real_text"]) == 320 * (1 + 8 + 6)
    assert not seen["from_placeholder"] & seen["from_real_text"]
    # A seed option wins over the placeholder seeds: the placeholder text with the two seeds of a real
    # text file as options gives the split files of the real text file.
    assert module.main(["--out", str(tmp_path / "real_seeds"), "--data-seed", "20261118",
                        "--calib-seed", "20261119"]) == 0
    given = summary_of(tmp_path / "real_seeds")
    assert (given["config"]["data_seed"], given["config"]["calib_seed"]) == (20261118, 20261119)
    for split in SPLITS:
        assert given["files_sha256"][f"{split}.jsonl"] == real["files_sha256"][f"{split}.jsonl"]


def test_the_test_fixtures_use_the_placeholder_seeds(cfg, text, families):
    """tests/conftest.py: with the placeholder text the `cfg` fixture carries the placeholder seeds, and the
    `families` fixture is built from them."""
    if not text.is_placeholder:
        pytest.skip("the tests run on another text file (ACCESSDESK_TEST_TEXT)")
    assert (cfg.data_seed, cfg.calib_seed) == (9001, 9002) and cfg == RunConfig().with_seeds_for(True)
    assert families["dev"][0]["world_seed"] == derive_seed(9001, "dev", 0)
    assert families["calib"][0]["world_seed"] == derive_seed(9002, "calib", 0)
    stored = ROOT / "data" / "scenarios_placeholder"
    if stored.is_dir():
        assert summary_of(stored)["config"] == cfg.to_dict()


# The four split files of data/scenarios_placeholder as they were committed BEFORE the switches of round 2
# existed (placeholder text, data seed 1, calibration seed 2; the hashes were taken from the committed files
# on 2026-11-17, before the first change). That design is the plain design. It is not the default now: the
# test below sets it by hand (--no-mix-facts, --homonym-share 0.2, the seeds 1 and 2). It must still give
# exactly these bytes.
TEXT_AND_NAMES_OF_THE_PINNED_FILES = {
    "placeholder_text.yaml": "779be83e32c9e03ee3c2e18c21f6ae7f014a2bd6e21c0344e963b40b3328fc75",
    "first_names.txt": "be99b057c35573b35dd4c5f352846490c22c7417abdf06fe447d1793bd05bd9c",
    "last_names.txt": "785f1a001034646c828027893ac29575224eb588d4734403224dc421e8f4c362",
}
PLAIN_DESIGN_FILES_WITH_SEEDS_1_AND_2 = {
    "train.jsonl": "b371a9d947b937f3a263d148b5341892d16483c93497a8661c33a09a5348446e",
    "dev.jsonl": "4cb88f98d9bae07d6957734aa153b9fd0616858f9ce679a63e0c503d153d65eb",
    "test.jsonl": "4022f37e05b63734ca87bb6f6d68da26da3d14f4a22ffef78a5806c47d20c1db",
    "calib.jsonl": "3321f33f0572800b9931e20ad236c2159b1dab9ce8fcc35158544d75372fbada",
}


def test_with_the_switches_off_the_generator_gives_the_bytes_it_gave_before_round_2(tmp_path, capsys):
    """THE DESIGN WITH THE SWITCHES OFF MUST NOT CHANGE. Built with the old seeds and the old design through
    today's code, the four split files have the SHA-256 of the files that were in the repository before
    mix_facts, the placeholder seeds and the options of the build script existed. (sets.json is not pinned:
    the rule for the policy-shift controls changed on purpose.)
    Every old value is given as an option (seeds 1 and 2, no mixed facts, share 0.2): since the freeze day
    the defaults of accessdesk/config.py are the final design and the final seeds."""
    inputs = {"placeholder_text.yaml": placeholder_path(),
              "first_names.txt": ROOT / "data" / "names" / "first_names.txt",
              "last_names.txt": ROOT / "data" / "names" / "last_names.txt"}
    for name, path in inputs.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != \
                TEXT_AND_NAMES_OF_THE_PINNED_FILES[name]:
            pytest.skip(f"{name} is not the file that the pinned hashes were made from")
    module = load_build_script()
    assert module.main(["--out", str(tmp_path / "old_seeds"), "--data-seed", "1", "--calib-seed", "2",
                        "--no-mix-facts", "--homonym-share", "0.2"]) == 0
    capsys.readouterr()
    hashes = {name: hashlib.sha256((tmp_path / "old_seeds" / name).read_bytes()).hexdigest()
              for name in PLAIN_DESIGN_FILES_WITH_SEEDS_1_AND_2}
    assert hashes == PLAIN_DESIGN_FILES_WITH_SEEDS_1_AND_2


def test_the_final_folder_of_this_repository_is_refused_with_the_seed_of_the_calibration_folders(text, pools):
    """data/scenarios_calib1 (and the later calibration folders) hold the train, dev and test worlds of the
    data seed 1, the default until the freeze day. A build of data/scenarios with that seed must stop; the
    placeholder folder, which has seeds of its own, is not the reason. With the config as it is today (the
    final data seed) no world seed is in an earlier folder, so the final build is not stopped."""
    if not (ROOT / "data" / "scenarios_calib1").is_dir():
        pytest.skip("data/scenarios_calib1 is not in this copy of the repository")
    module = load_build_script()
    earlier = module.earlier_dirs(ROOT / "data" / "scenarios")
    assert {"scenarios_calib1", "scenarios_placeholder"} <= {path.name for path in earlier}
    with_old_seeds = dataclasses.replace(RunConfig(), data_seed=1, calib_seed=2)     # set by hand
    reused = module.reused_world_seeds(make_all_splits(text, pools, with_old_seeds), earlier)
    assert any(line.startswith("296 of the 296") and line.endswith("scenarios_calib1") for line in reused)
    assert not any(line.endswith("scenarios_placeholder") for line in reused)
    families_of_final = make_all_splits(text, pools, RunConfig())            # the config as it is today
    assert (RunConfig().data_seed, RunConfig().calib_seed) == (20261118, 20261119)
    assert module.reused_world_seeds(families_of_final, earlier) == []
    # With the placeholder seeds as the data seed, the placeholder folder is the reason.
    as_placeholder = make_all_splits(text, pools, RunConfig().with_seeds_for(True))
    reused = module.reused_world_seeds(as_placeholder, earlier)
    assert [line.endswith("scenarios_placeholder") for line in reused] == [True]
