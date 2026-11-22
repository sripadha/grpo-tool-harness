"""Tests of accessdesk/eval/tables.py on hand-made episode records.

The small example below has two arms on four families. Every number of its table is worked out by hand here.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from accessdesk import impl
from accessdesk.config import DEFAULT
from accessdesk.eval import bootstrap as main_bootstrap
from accessdesk.eval import metrics, tables
from tests.helpers.make_records import make_record, scenario_records

P, F = 100, 0


def small_example() -> dict[str, list[dict]]:
    """Two arms, four families (one per outcome), two styles (1 = seen, 4 = held out), two samples each.

    Family pass rates:      base                                  sft                             sft - base
        dv-000 grant        s1 P F, s4 F F  -> 1/4 = 0.25         s1 P P, s4 P F -> 0.75          +0.50
        dv-001 approval     s1 F F, s4 F F  -> 0.00               s1 P F, s4 F F -> 0.25          +0.25
        dv-002 already      s1 P 95, s4 P P -> 1.00               s1 P P, s4 P P -> 1.00           0.00
        dv-003 inactive     s1 P F, s4 P F  -> 0.50               s1 P P, s4 F F -> 0.50           0.00
    Pass rate:              (0.25 + 0 + 1 + 0.5) / 4 = 0.4375     (0.75 + 0.25 + 1 + 0.5) / 4 = 0.625
    Difference: 0.625 - 0.4375 = 0.1875.
    """
    base = (scenario_records("dv-000", 1, [P, F], outcome="grant", arm="base")
            + scenario_records("dv-000", 4, [F, F], outcome="grant", arm="base")
            + scenario_records("dv-001", 1, [F, F], outcome="approval", arm="base")
            + scenario_records("dv-001", 4, [F, F], outcome="approval", arm="base")
            + scenario_records("dv-002", 1, [P, 95], outcome="already", arm="base")
            + scenario_records("dv-002", 4, [P, P], outcome="already", arm="base")
            + scenario_records("dv-003", 1, [P, F], outcome="inactive", arm="base")
            + scenario_records("dv-003", 4, [P, F], outcome="inactive", arm="base"))
    sft = (scenario_records("dv-000", 1, [P, P], outcome="grant", arm="sft")
           + scenario_records("dv-000", 4, [P, F], outcome="grant", arm="sft")
           + scenario_records("dv-001", 1, [P, F], outcome="approval", arm="sft")
           + scenario_records("dv-001", 4, [F, F], outcome="approval", arm="sft")
           + scenario_records("dv-002", 1, [P, P], outcome="already", arm="sft")
           + scenario_records("dv-002", 4, [P, P], outcome="already", arm="sft")
           + scenario_records("dv-003", 1, [P, P], outcome="inactive", arm="sft")
           + scenario_records("dv-003", 4, [F, F], outcome="inactive", arm="sft"))
    return {"base": base, "sft": sft}


BASE_RATES = [0.25, 0.0, 1.0, 0.5]     # per family, in family order
SFT_RATES = [0.75, 0.25, 1.0, 0.5]


HASH_A, HASH_B, HASH_C = "a" * 64, "b" * 64, "c" * 64       # stand-ins for the SHA-256 of a scenario file


def with_data_hash(records: list[dict], data_hash: str) -> list[dict]:
    """Copies of the records that carry `data_sha256`, as records of scripts/run_scripted.py do."""
    return [{**record, "data_sha256": data_hash} for record in records]


@pytest.fixture
def main_mode(monkeypatch):
    """The paired bootstrap is the main version, whatever ACCESSDESK_IMPL says outside the test."""
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")


def test_table_header_and_counts(main_mode):
    table = tables.compare_arms(small_example(), "base", [("base", "sft")], n_draws=2000, seed=3)
    assert list(table) == ["level", "n_draws", "seed", "baseline_arm", "paired_family_bootstrap",
                           "text_sha256", "text_is_placeholder", "splits",
                           "n_families", "n_scenarios", "n_episodes_per_arm", "arms", "pairs", "notes"]
    assert (table["level"], table["n_draws"], table["seed"], table["baseline_arm"]) == (0.95, 2000, 3, "base")
    assert table["paired_family_bootstrap"] == "main"
    # 4 families, 2 styles each = 8 scenarios, 2 samples each = 16 episodes per arm.
    assert (table["n_families"], table["n_scenarios"], table["n_episodes_per_arm"]) == (4, 8, 16)
    assert [row["arm"] for row in table["arms"]] == ["base", "sft"]


def test_arm_row_by_hand(main_mode):
    table = tables.compare_arms(small_example(), "base", n_draws=2000, seed=3)
    base = table["arms"][0]
    assert (base["n_families"], base["n_episodes"], base["n_harness_errors"]) == (4, 16, 0)
    assert base["pass_rate"] == 0.4375
    assert base["pass_rate_by_outcome"] == {"grant": 0.25, "approval": 0.0, "already": 1.0, "inactive": 0.5}
    # Seen style 1: family rates 0.5, 0, 1, 0.5 -> 0.5. Held-out style 4: 0, 0, 1, 0.5 -> 0.375.
    assert (base["pass_rate_seen"], base["pass_rate_held_out"]) == (0.5, 0.375)
    # Scenarios with equal rewards: dv-000/s4 (F F), dv-001/s1, dv-001/s4, dv-002/s4 (P P) -> 4 of 8.
    # Equal for pass-or-fail: also dv-002/s1 (100 and 95 are both passes) -> 5 of 8.
    assert (base["zero_variance_equal_reward"], base["zero_variance_equal_passfail"]) == (0.5, 0.625)
    # Messages: grant and approval episodes have 4 turns, already and inactive 3; the reward-95 episode has
    # one malformed message more: 4 * 4 + 4 * 4 + (3 * 3 + 4) + 4 * 3 = 57. Malformed: 1.
    assert base["invalid_call_rate"] == pytest.approx(1 / 57)
    assert table["arms"][1]["pass_rate"] == 0.625
    assert table["arms"][1]["invalid_call_rate"] == 0.0


def test_arm_interval_is_the_family_bootstrap_of_its_family_rates(main_mode):
    table = tables.compare_arms(small_example(), "base", n_draws=2000, seed=3, level=0.9)
    expected = metrics.family_bootstrap_ci(BASE_RATES, 2000, 3, 0.9)
    base = table["arms"][0]
    assert (base["pass_rate_lo"], base["pass_rate_hi"]) == (expected["lo"], expected["hi"])
    assert base["pass_rate_lo"] < base["pass_rate"] < base["pass_rate_hi"]


def test_seed_and_number_of_draws_reach_the_interval_of_an_arm(main_mode):
    # Nine families with the pass rates 0/8, 1/8, ..., 8/8: enough different values for the seed to matter.
    rates = [n_passed / 8 for n_passed in range(9)]
    records = []
    for n_passed in range(9):
        records += scenario_records(f"dv-{n_passed:03d}", 1, [P] * n_passed + [F] * (8 - n_passed))
    seed_0 = metrics.family_bootstrap_ci(rates, 300, 0, 0.95)
    seed_5 = metrics.family_bootstrap_ci(rates, 300, 5, 0.95)
    assert (seed_0["lo"], seed_0["hi"]) != (seed_5["lo"], seed_5["hi"])     # the two seeds can be told apart
    arm = tables.compare_arms({"only": records}, "only", n_draws=300, seed=5)["arms"][0]
    assert arm["pass_rate"] == 0.5
    assert (arm["pass_rate_lo"], arm["pass_rate_hi"]) == (seed_5["lo"], seed_5["hi"])


def test_pair_row_by_hand(main_mode):
    table = tables.compare_arms(small_example(), "base", [("base", "sft")], n_draws=2000, seed=3)
    assert len(table["pairs"]) == 1
    pair = table["pairs"][0]
    assert (pair["a"], pair["b"], pair["n_families"]) == ("base", "sft", 4)
    assert pair["diff"] == pytest.approx(0.1875, abs=1e-12)
    # The interval is the paired bootstrap of the two lists of family rates, written out by hand above.
    expected = main_bootstrap.paired_family_bootstrap(np.array(BASE_RATES), np.array(SFT_RATES),
                                                      2000, 3, 0.95)
    assert (pair["lo"], pair["hi"]) == (expected["lo"], expected["hi"])
    # Per-family differences 0.5, 0.25, 0, 0: every draw of 4 families has a mean between 0 and 0.5.
    assert 0.0 <= pair["lo"] < 0.1875 < pair["hi"] <= 0.5
    # t-interval of the differences: mean 0.1875; deviations 0.3125, 0.0625, -0.1875, -0.1875; sum of squares
    # 0.171875; variance 0.171875 / 3 = 0.057292; standard deviation 0.239357; se = 0.239357 / 2 = 0.119678;
    # t(0.975, 3) = 3.1824; half width 0.38087. Interval -0.1934 to 0.5684.
    assert pair["t_lo"] == pytest.approx(-0.1934, abs=5e-4)
    assert pair["t_hi"] == pytest.approx(0.5684, abs=5e-4)


def test_pair_direction_is_b_minus_a(main_mode):
    table = tables.compare_arms(small_example(), "base", [("sft", "base")], n_draws=500)
    pair = table["pairs"][0]
    assert (pair["a"], pair["b"]) == ("sft", "base")
    assert pair["diff"] == pytest.approx(-0.1875, abs=1e-12)
    assert pair["hi"] <= 0.0


def test_default_pairs_are_every_other_arm_against_the_baseline(main_mode):
    arms = small_example()
    arms["again"] = [dict(record, arm="again") for record in arms["sft"]]
    table = tables.compare_arms(arms, "sft", n_draws=200)
    assert [row["arm"] for row in table["arms"]] == ["sft", "base", "again"]       # the baseline comes first
    assert [(pair["a"], pair["b"]) for pair in table["pairs"]] == [("sft", "base"), ("sft", "again")]
    assert table["pairs"][1]["diff"] == 0.0
    assert (table["pairs"][1]["lo"], table["pairs"][1]["hi"]) == (0.0, 0.0)


def test_paired_bootstrap_is_reached_through_impl_get(monkeypatch):
    calls = []

    def fake_bootstrap(a, b, n_draws, seed, level):
        calls.append((a.tolist(), b.tolist(), n_draws, seed, level))
        return {"diff": 0.111, "lo": -0.222, "hi": 0.333}

    def fake_get(name):
        assert name == "paired_family_bootstrap"
        return fake_bootstrap

    monkeypatch.setattr(impl, "get", fake_get)
    monkeypatch.setattr(impl, "which", lambda name: "second")
    table = tables.compare_arms(small_example(), "base", [("base", "sft")], n_draws=123, seed=9, level=0.8)
    # The function from impl.get received the family rates of both arms in the same family order.
    assert calls == [(BASE_RATES, SFT_RATES, 123, 9, 0.8)]
    pair = table["pairs"][0]
    assert (pair["diff"], pair["lo"], pair["hi"]) == (0.111, -0.222, 0.333)
    assert table["paired_family_bootstrap"] == "second"


def test_tables_module_does_not_import_a_bootstrap_version_directly():
    source = Path(tables.__file__).read_text(encoding="utf-8")
    assert "_second" not in source
    assert "eval.bootstrap" not in source and "eval import bootstrap" not in source
    assert 'impl.get("paired_family_bootstrap")' in source


# ---------------------------------------------------------------------------
# Coverage: all arms must be run on the same scenarios with the same number of samples
# ---------------------------------------------------------------------------
def test_missing_scenario_in_one_arm_is_refused(main_mode):
    arms = small_example()
    arms["sft"] = [r for r in arms["sft"] if not (r["family_id"] == "dv-003" and r["style"] == 4)]
    with pytest.raises(ValueError, match="does not cover the same scenarios"):
        tables.compare_arms(arms, "base", n_draws=100)


def test_extra_family_in_one_arm_is_refused(main_mode):
    arms = small_example()
    arms["sft"] = arms["sft"] + scenario_records("dv-004", 1, [P, P], arm="sft")
    with pytest.raises(ValueError, match="does not cover the same scenarios"):
        tables.compare_arms(arms, "base", n_draws=100)


def test_different_sample_count_is_refused(main_mode):
    arms = small_example()
    arms["sft"] = arms["sft"] + [make_record("dv-000", 1, 2, arm="sft")]     # a third sample of one scenario
    with pytest.raises(ValueError, match="does not cover the same scenarios"):
        tables.compare_arms(arms, "base", n_draws=100)


def test_duplicate_sample_is_refused(main_mode):
    arms = small_example()
    arms["base"] = arms["base"][:-1] + [arms["base"][0]]     # same number of records, one sample twice
    with pytest.raises(ValueError, match="more than once"):
        tables.compare_arms(arms, "base", n_draws=100)


def test_mixed_conditions_inside_one_arm_are_refused(main_mode):
    for key, value in (("ruleset", "all_grants_need_approval"), ("toolset", "six"), ("prompt", "plain")):
        arms = small_example()
        arms["sft"][0] = dict(arms["sft"][0], **{key: value})
        with pytest.raises(ValueError, match="exactly one"):
            tables.compare_arms(arms, "base", n_draws=100)


def test_unknown_arm_names_are_refused(main_mode):
    with pytest.raises(ValueError):
        tables.compare_arms(small_example(), "nobody", n_draws=100)
    with pytest.raises(ValueError):
        tables.compare_arms(small_example(), "base", [("base", "nobody")], n_draws=100)
    with pytest.raises(ValueError):
        tables.compare_arms(small_example(), "base", [("base", "base")], n_draws=100)


def test_harness_errors_keep_the_coverage_and_are_counted(main_mode):
    arms = small_example()
    # The first sft record (dv-000/s1, a pass) becomes a harness error: the coverage is the same, and
    # 15 episodes count.
    arms["sft"][0] = dict(arms["sft"][0], harness_error="made-up error")
    table = tables.compare_arms(arms, "base", n_draws=200)
    sft = table["arms"][1]
    assert (sft["n_episodes"], sft["n_harness_errors"]) == (15, 1)
    # dv-000 under sft is now P (s1), P F (s4): 2 of 3. Pass rate (2/3 + 0.25 + 1 + 0.5) / 4 = 0.604167.
    assert sft["pass_rate"] == pytest.approx(0.604167, abs=1e-6)
    assert table["pairs"][0]["diff"] == pytest.approx(0.604167 - 0.4375, abs=1e-6)


def test_family_with_harness_errors_only_cannot_be_paired(main_mode):
    arms = small_example()
    arms["sft"] = [dict(r, harness_error="made-up error") if r["family_id"] == "dv-001" else r
                   for r in arms["sft"]]
    with pytest.raises(ValueError, match="cannot be paired"):
        tables.compare_arms(arms, "base", n_draws=100)


# ---------------------------------------------------------------------------
# The scenario file: all arms must be run on the same data
# ---------------------------------------------------------------------------
def test_arms_run_on_different_scenario_files_are_refused(main_mode):
    arms = small_example()
    arms = {"base": with_data_hash(arms["base"], HASH_A), "sft": with_data_hash(arms["sft"], HASH_B)}
    with pytest.raises(ValueError, match="same scenario file") as refused:
        tables.compare_arms(arms, "base", n_draws=100)
    # The message names both arms and the start of both hashes.
    message = str(refused.value)
    assert "'base': ['aaaaaaaaaaaa']" in message and "'sft': ['bbbbbbbbbbbb']" in message


def test_arms_run_on_the_same_scenario_file_are_accepted_without_a_note(main_mode):
    arms = small_example()
    arms = {"base": with_data_hash(arms["base"], HASH_A), "sft": with_data_hash(arms["sft"], HASH_A)}
    table = tables.compare_arms(arms, "base", n_draws=100)
    assert table["notes"] == []
    assert table["pairs"][0]["diff"] == pytest.approx(0.1875, abs=1e-12)
    assert "Notes" not in tables.to_markdown(table)


def test_one_other_record_from_another_scenario_file_is_enough_to_refuse(main_mode):
    arms = small_example()
    arms = {"base": with_data_hash(arms["base"], HASH_A), "sft": with_data_hash(arms["sft"], HASH_A)}
    arms["sft"][5] = {**arms["sft"][5], "data_sha256": HASH_B}
    with pytest.raises(ValueError, match="same scenario file"):
        tables.compare_arms(arms, "base", n_draws=100)


def test_arms_that_pool_the_same_two_scenario_files_are_accepted(main_mode):
    # Each arm was run on two split files (for example dev and test): two values, the same two in both.
    arms = small_example()
    first_half = {"dv-000", "dv-001"}
    pooled = {arm: [{**r, "data_sha256": HASH_A if r["family_id"] in first_half else HASH_B} for r in records]
              for arm, records in arms.items()}
    assert tables.compare_arms(pooled, "base", n_draws=100)["notes"] == []
    # The same arm with a third file in place of the second is refused.
    pooled["sft"] = [{**r, "data_sha256": HASH_C} if r["data_sha256"] == HASH_B else r for r in pooled["sft"]]
    with pytest.raises(ValueError, match="same scenario file"):
        tables.compare_arms(pooled, "base", n_draws=100)


def test_records_without_a_data_hash_are_accepted_with_a_note(main_mode):
    arms = small_example()                                    # hand-made records: none carries the key
    table = tables.compare_arms(arms, "base", n_draws=100)
    assert table["notes"] == [
        "Arm base: 16 of 16 records carry no data_sha256, so it was not checked that this arm was run on "
        "the same scenario file as the others.",
        "Arm sft: 16 of 16 records carry no data_sha256, so it was not checked that this arm was run on "
        "the same scenario file as the others."]
    # One arm with the key and one without: accepted, and the note names the arm without it.
    mixed = {"base": with_data_hash(arms["base"], HASH_A), "sft": arms["sft"]}
    (note,) = tables.compare_arms(mixed, "base", n_draws=100)["notes"]
    assert note.startswith("Arm sft: 16 of 16 records carry no data_sha256")
    # Some records of an arm without the key (None counts as "without"): the note counts them.
    partly = {"base": with_data_hash(arms["base"], HASH_A),
              "sft": (with_data_hash(arms["sft"][:13], HASH_A)
                      + [{**r, "data_sha256": None} for r in arms["sft"][13:]])}
    (note,) = tables.compare_arms(partly, "base", n_draws=100)["notes"]
    assert note.startswith("Arm sft: 3 of 16 records carry no data_sha256")
    # The records that do carry it are still compared.
    partly["sft"][0] = {**partly["sft"][0], "data_sha256": HASH_B}
    with pytest.raises(ValueError, match="same scenario file"):
        tables.compare_arms(partly, "base", n_draws=100)


def test_the_notes_are_printed_below_the_markdown_tables(main_mode):
    arms = small_example()
    mixed = {"base": with_data_hash(arms["base"], HASH_A), "sft": arms["sft"]}
    lines = tables.to_markdown(tables.compare_arms(mixed, "base", n_draws=100)).splitlines()
    assert lines[10].startswith("| sft minus base | 4 | +18.8 | ")
    assert lines[-3:] == ["", "Notes:",
                          "- Arm sft: 16 of 16 records carry no data_sha256, so it was not checked that this "
                          "arm was run on the same scenario file as the others."]


# ---------------------------------------------------------------------------
# The interval of the table is a family-level interval
# ---------------------------------------------------------------------------
def test_table_interval_is_clearly_wider_than_an_episode_level_interval_when_the_gain_differs_by_family(
        main_mode):
    # 24 families, 4 styles, 8 samples: 32 episodes per family, 768 per arm.
    # Arm "a": 2 passes of 8 in every scenario (0.25 everywhere).
    # Arm "b": 7 of 8 in every scenario of the first 12 families (gain 0.625), and 2 of 8 in the other 12
    # (gain 0).
    arm_a, arm_b = [], []
    for family in range(24):
        for style in (1, 2, 3, 4):
            family_id = f"dv-{family:03d}"
            arm_a += scenario_records(family_id, style, [P] * 2 + [F] * 6, arm="a")
            n_passed = 7 if family < 12 else 2
            arm_b += scenario_records(family_id, style, [P] * n_passed + [F] * (8 - n_passed), arm="b")
    table = tables.compare_arms({"a": arm_a, "b": arm_b}, "a", n_draws=4000, seed=0)
    pair = table["pairs"][0]
    # Mean gain (12 * 0.625 + 12 * 0) / 24 = 0.3125.
    assert pair["diff"] == pytest.approx(0.3125, abs=1e-12)
    # Family level: the gains are 0.625 or 0: standard deviation 0.3125, standard error 0.3125 / sqrt(24)
    # = 0.0638, width of the 95% interval about 3.92 * 0.0638 = 0.25.
    family_width = pair["hi"] - pair["lo"]
    assert family_width == pytest.approx(0.25, abs=0.03)

    # Episode level (the wrong unit): resample the 768 episodes of each arm.
    # Pooled pass rates 0.25 and 0.5625: standard error sqrt((0.1875 + 0.2461) / 768) = 0.0238, width 0.093.
    passed_a = np.array([r["passed"] for r in arm_a], dtype=float)
    passed_b = np.array([r["passed"] for r in arm_b], dtype=float)
    rng = np.random.default_rng(0)
    draws = (passed_b[rng.integers(0, 768, size=(1000, 768))].mean(axis=1)
             - passed_a[rng.integers(0, 768, size=(1000, 768))].mean(axis=1))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    assert float(hi - lo) == pytest.approx(0.093, abs=0.012)
    assert family_width > 2.0 * float(hi - lo)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def test_table_is_the_same_for_any_order_of_the_records_and_is_plain_json(main_mode):
    arms = small_example()
    first = tables.compare_arms(arms, "base", n_draws=500, seed=1)
    backwards = {name: list(reversed(records)) for name, records in arms.items()}
    second = tables.compare_arms(backwards, "base", n_draws=500, seed=1)
    assert json.dumps(first, allow_nan=False) == json.dumps(second, allow_nan=False)
    assert json.loads(json.dumps(first)) == first


def test_markdown_rows_by_hand(main_mode):
    arms = {arm: with_data_hash(records, HASH_A) for arm, records in small_example().items()}
    table = tables.compare_arms(arms, "base", [("base", "sft")], n_draws=2000, seed=3)
    text = tables.to_markdown(table)
    lines = text.splitlines()
    assert lines[0] == "Families: 4. Scenarios: 8. Episodes per arm: 16."
    assert lines[1] == ("Intervals: 95%, by resampling families, 2000 draws, seed 3. "
                        "Paired bootstrap: the main version of paired_family_bootstrap.")
    base, pair = table["arms"][0], table["pairs"][0]
    interval = f"{100 * base['pass_rate_lo']:.1f} to {100 * base['pass_rate_hi']:.1f}"
    # Pass rate 43.75% is printed as 43.8; outcomes 25, 0, 100, 50; seen 50; held out 37.5; equal rewards 50;
    # equal pass or fail 62.5; invalid calls 1 / 57 = 1.8%.
    assert lines[5] == (f"| base | 16 | 0 | 43.8 | {interval} | 25.0 | 0.0 | 100.0 | 50.0 | 50.0 | 37.5 "
                        f"| 50.0 | 62.5 | 1.8 |")
    assert lines[6].startswith("| sft | 16 | 0 | 62.5 | ")
    boot = f"{100 * pair['lo']:+.1f} to {100 * pair['hi']:+.1f}"
    # Difference +18.75 points is printed as +18.8; t-interval -19.3 to +56.8.
    assert lines[10] == f"| sft minus base | 4 | +18.8 | {boot} | -19.3 to +56.8 |"
    # Every table row has as many cells as its header: the arms, the comparisons, the conditions of the
    # arms (lines 12 to 15), their reliability and behaviour numbers (lines 19 to 22).
    assert [lines[2], lines[7], lines[11], lines[16], lines[18]] == [""] * 5
    for header, rows in ((lines[3], lines[5:7]), (lines[8], lines[10:11]), (lines[12], lines[14:16]),
                         (lines[19], lines[21:23])):
        assert all(row.count("|") == header.count("|") for row in rows)
    for header in (3, 8, 12, 19):
        assert lines[header + 1].count("|") == lines[header].count("|")
        assert set(lines[header + 1]) == {"|", "-"}
    assert text.endswith("\n")


def test_markdown_shows_not_available_for_a_share_that_is_not_defined(main_mode):
    # Arms with one sample per scenario (a scripted policy): no group has 2 samples, so the zero-variance
    # shares are not defined. The table says so; it does not print 0.
    arms = {"oracle": scenario_records("dv-000", 1, [P], arm="oracle")
            + scenario_records("dv-001", 1, [P], arm="oracle"),
            "constant": scenario_records("dv-000", 1, [P], arm="constant")
            + scenario_records("dv-001", 1, [F], arm="constant")}
    table = tables.compare_arms(arms, "oracle", n_draws=200)
    assert table["arms"][0]["zero_variance_equal_reward"] is None
    oracle_row = tables.to_markdown(table).splitlines()[5]
    # Both families have the outcome "grant" and style 1 only: the other outcomes and the held-out style
    # have no record, and are not available either.
    assert oracle_row == ("| oracle | 2 | 0 | 100.0 | 100.0 to 100.0 | 100.0 | n/a | n/a | n/a | 100.0 | n/a "
                          "| n/a | n/a | 0.0 |")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_main_reads_record_files_and_writes_json_and_markdown(tmp_path, main_mode):
    arms = small_example()
    _write_jsonl(tmp_path / "base.jsonl", arms["base"])
    _write_jsonl(tmp_path / "sft.jsonl", arms["sft"])
    arguments = ["--arm", f"base={tmp_path / 'base.jsonl'}", "--arm", f"sft={tmp_path / 'sft.jsonl'}",
                 "--baseline", "base", "--pair", "base:sft", "--draws", "300", "--seed", "5"]
    tables.main(arguments + ["--out-json", str(tmp_path / "t1.json"), "--out-md", str(tmp_path / "t1.md")])
    tables.main(arguments + ["--out-json", str(tmp_path / "t2.json"), "--out-md", str(tmp_path / "t2.md")])
    # Twice the same bytes.
    assert (tmp_path / "t1.json").read_bytes() == (tmp_path / "t2.json").read_bytes()
    assert (tmp_path / "t1.md").read_bytes() == (tmp_path / "t2.md").read_bytes()
    # The files hold what compare_arms and to_markdown return for the same records.
    expected = tables.compare_arms(arms, "base", [("base", "sft")], n_draws=300, seed=5)
    assert json.loads((tmp_path / "t1.json").read_text(encoding="utf-8")) == expected
    assert (tmp_path / "t1.md").read_text(encoding="utf-8") == tables.to_markdown(expected)
    assert expected["pairs"][0]["diff"] == pytest.approx(0.1875, abs=1e-12)


def test_main_refuses_malformed_arguments(tmp_path, main_mode):
    _write_jsonl(tmp_path / "base.jsonl", small_example()["base"])
    outputs = ["--out-json", str(tmp_path / "t.json"), "--out-md", str(tmp_path / "t.md")]
    with pytest.raises(SystemExit):
        tables.main(["--arm", "base", "--baseline", "base"] + outputs)                    # no "=path"
    with pytest.raises(SystemExit):
        tables.main(["--arm", f"base={tmp_path / 'base.jsonl'}", "--arm", f"base={tmp_path / 'base.jsonl'}",
                     "--baseline", "base"] + outputs)                                     # the same arm twice
    assert not (tmp_path / "t.json").exists()


# ---------------------------------------------------------------------------
# What the table refuses beyond coverage, text and data (the findings of 2026-11-17)
# ---------------------------------------------------------------------------
def hashed_example() -> dict[str, list[dict]]:
    """The small example with the hash of one scenario file in every record: nothing is left to note."""
    return {arm: with_data_hash(records, HASH_A) for arm, records in small_example().items()}


def with_settings(records: list[dict], **changes) -> list[dict]:
    """Copies of the records whose `settings` differ from the default config's in `changes`."""
    return [{**record, "settings": {**DEFAULT.compared_settings(), **changes}} for record in records]


def without_settings(records: list[dict], config_sha256: str) -> list[dict]:
    """Copies of the records as an older run wrote them: no `settings`, only the hash of the whole config."""
    return [{key: value for key, value in {**record, "config_sha256": config_sha256}.items()
             if key != "settings"} for record in records]


@pytest.mark.parametrize("name, value", [("max_new_tokens", 96), ("turn_limit", 4), ("parser_strict", False),
                                         ("temperature", 0.7), ("top_k", 50), ("top_p", 0.9),
                                         ("pass_reward", 50), ("malformed_penalty", 0), ("malformed_cap", 0),
                                         ("turn_penalty", 0), ("turn_cap", 0),
                                         ("penalise_malformed_on_fail", True)])
def test_arms_run_with_different_settings_are_refused(main_mode, name, value):
    arms = small_example()
    changed = {"base": arms["base"], "sft": with_settings(arms["sft"], **{name: value})}
    with pytest.raises(ValueError, match=f"same settings.*{name}"):
        tables.compare_arms(changed, "base", n_draws=100)
    # One record of the arm is enough.
    one = {"base": arms["base"], "sft": with_settings(arms["sft"][:1], **{name: value}) + arms["sft"][1:]}
    with pytest.raises(ValueError, match="same settings"):
        tables.compare_arms(one, "base", n_draws=100)


def test_the_settings_are_compared_as_values_and_the_chunk_size_is_not_one_of_them(main_mode):
    arms = hashed_example()
    # 1 and 1.0 are the same temperature (JSON may give either back).
    same = {"base": arms["base"], "sft": with_settings(arms["sft"], temperature=1, top_p=1)}
    assert tables.compare_arms(same, "base", n_draws=100)["notes"] == []
    # Another chunk size gives another config hash and the same settings: accepted, and nothing to note.
    other = dataclasses.replace(DEFAULT, chunk_size=12)
    assert other.sha256() != DEFAULT.sha256() and other.compared_settings() == DEFAULT.compared_settings()
    sft = [{**record, "config_sha256": other.sha256()} for record in arms["sft"]]
    table = tables.compare_arms({"base": arms["base"], "sft": sft}, "base", n_draws=100)
    assert table["notes"] == []
    assert [row["config_sha256"] for row in table["arms"]] == [[DEFAULT.sha256()], [other.sha256()]]


def test_older_records_without_settings_get_a_note_when_the_config_hashes_differ(main_mode):
    arms = hashed_example()
    # Every record names the same config hash: the settings are the same, with or without the key.
    same = {"base": without_settings(arms["base"], DEFAULT.sha256()), "sft": arms["sft"]}
    assert tables.compare_arms(same, "base", n_draws=100)["notes"] == []
    # Another hash and no settings: it cannot be seen what differs, and the table says so.
    other = {"base": arms["base"], "sft": without_settings(arms["sft"], "2" * 64)}
    (note,) = tables.compare_arms(other, "base", n_draws=100)["notes"]
    assert note.startswith("Arm sft: 16 of 16 records carry no settings, and the records do not all name "
                           "the same config_sha256")


def test_records_scored_by_two_versions_of_verify_or_reward_are_refused(main_mode):
    arms = small_example()
    second = [{**record, "impl": {"verify": "second", "reward": "second"}} for record in arms["sft"]]
    with pytest.raises(ValueError, match="same versions of verify and reward"):
        tables.compare_arms({"base": arms["base"], "sft": second}, "base", n_draws=100)
    mixed = [record if index % 2 else {**record, "impl": {"verify": "second", "reward": "main"}}
             for index, record in enumerate(arms["sft"])]
    with pytest.raises(ValueError, match="same versions of verify and reward"):
        tables.compare_arms({"base": arms["base"], "sft": mixed}, "base", n_draws=100)
    # One version for every record of every arm is what the table needs, whichever version it is.
    base_second = [{**record, "impl": {"verify": "second", "reward": "second"}} for record in arms["base"]]
    assert tables.compare_arms({"base": base_second, "sft": second}, "base", n_draws=100)["pairs"]


def test_records_of_another_format_are_refused(main_mode):
    arms = small_example()
    newer = [{**record, "schema": 2} for record in arms["sft"]]
    with pytest.raises(ValueError, match="schema 1"):
        tables.compare_arms({"base": arms["base"], "sft": newer}, "base", n_draws=100)
    with pytest.raises(ValueError, match="schema 1"):
        tables.compare_arms({"base": arms["base"], "sft": newer[:1] + arms["sft"][1:]}, "base", n_draws=100)


def test_one_episode_saved_under_two_sample_numbers_is_refused(main_mode):
    arms = small_example()
    first, second = arms["sft"][-2:]                       # the two samples of dv-003/s4
    assert (first["scenario_id"], first["sample"], second["sample"]) == ("dv-003/s4", 0, 1)
    copy = {**first, "sample": 1}                          # same seed entry, same turns, another number
    with pytest.raises(ValueError, match="same seed"):
        tables.compare_arms({"base": arms["base"], "sft": arms["sft"][:-1] + [copy]}, "base", n_draws=100)
    # A record without a seed entry (None) cannot be checked this way and is accepted.
    no_seed = [{**record, "seed": None} for record in arms["sft"]]
    assert tables.compare_arms({"base": arms["base"], "sft": no_seed}, "base", n_draws=100)["pairs"]


def test_the_same_records_under_two_arm_names_are_refused(main_mode):
    base = small_example()["base"]
    with pytest.raises(ValueError, match="exactly the same records"):
        tables.compare_arms({"base": base, "sft": [dict(record) for record in base]}, "base", n_draws=100)


def test_a_wording_lost_to_harness_errors_in_one_arm_only_is_refused(main_mode):
    """dv-000 keeps valid episodes in both arms (style 1), so the families can be paired; but in `sft` its
    style 4 is gone, and its pass rate would be over one wording against two in `base`."""
    arms = small_example()

    def lose(records: list[dict]) -> list[dict]:
        return [{**record, "harness_error": "out of memory"} if record["scenario_id"] == "dv-000/s4"
                else record for record in records]

    with pytest.raises(ValueError, match=r"scenarios have valid episodes in only one of them.*dv-000.*4"):
        tables.compare_arms({"base": arms["base"], "sft": lose(arms["sft"])}, "base", n_draws=100)
    # Lost in both arms: the same wordings are compared, and the errors are counted.
    table = tables.compare_arms({"base": lose(arms["base"]), "sft": lose(arms["sft"])}, "base", n_draws=100)
    assert [row["n_harness_errors"] for row in table["arms"]] == [2, 2]


# ---------------------------------------------------------------------------
# What the table says about its arms: conditions, reliability, behaviour
# ---------------------------------------------------------------------------
def test_the_table_names_the_conditions_of_every_arm(main_mode):
    table = tables.compare_arms(hashed_example(), "base", n_draws=100)
    assert (table["text_sha256"], table["text_is_placeholder"], table["splits"]) == ("0" * 64, True, ["dev"])
    for row in table["arms"]:
        assert (row["splits"], row["ruleset"], row["toolset"], row["prompt"]) == \
            (["dev"], "default", "five", "final")
        assert row["text_is_placeholder"] is True and row["config_sha256"] == [DEFAULT.sha256()]
    assert table["notes"] == []
    lines = tables.to_markdown(table).splitlines()
    assert lines[12] == "| Arm | Split | Rule set | Tools | Prompt | Text | Config (start of its SHA-256) |"
    assert lines[14] == f"| base | dev | default | five | final | PLACEHOLDER | {DEFAULT.sha256()[:12]} |"
    assert lines[17].startswith("**PLACEHOLDER TEXT:") and "not a result" in lines[17]
    # With the real text the hash of the text file stands in the table, and the warning is gone.
    real = {arm: [{**record, "text_is_placeholder": False, "text_sha256": "ab" * 32} for record in records]
            for arm, records in small_example().items()}
    text = tables.to_markdown(tables.compare_arms(real, "base", n_draws=100))
    assert "| base | dev | default | five | final | abababababab |" in text and "PLACEHOLDER" not in text


def test_an_arm_under_another_condition_and_records_of_two_splits_get_a_note(main_mode):
    arms = hashed_example()
    shifted = [{**record, "ruleset": "all_grants_need_approval", "prompt": "policy_shift"}
               for record in arms["sft"]]
    table = tables.compare_arms({"base": arms["base"], "sft": shifted}, "base", n_draws=100)
    assert table["notes"] == [
        "Arm sft was run under another rule set, tool set or prompt ('all_grants_need_approval', 'five', "
        "'policy_shift') than the baseline arm base ('default', 'five', 'final'): a difference between them "
        "compares two conditions."]
    assert (table["arms"][1]["ruleset"], table["arms"][1]["prompt"]) == \
        ("all_grants_need_approval", "policy_shift")
    pooled = {arm: with_data_hash(scenario_records("dv-000", 1, rewards, arm=arm)
                                  + scenario_records("ca-000", 1, rewards, arm=arm), HASH_A)
              for arm, rewards in (("base", [P, F]), ("sft", [P, P]))}
    table = tables.compare_arms(pooled, "base", n_draws=100)
    assert table["splits"] == ["calib", "dev"] and table["arms"][0]["splits"] == ["calib", "dev"]
    assert table["notes"] == ["The records are of more than one split (calib, dev): every pass rate here is "
                              "over the families of all of them together."]


def test_arm_row_reliability_and_behaviour_numbers_by_hand(main_mode):
    arms = small_example()
    table = tables.compare_arms(arms, "base", n_draws=100, seed=3)
    base = table["arms"][0]
    # Two samples per scenario: pass@8, pass^4 and pass^8 have no scenario to use, and the row says so.
    assert base["pass_at_8"] is None
    assert base["n_scenarios_left_out"] == {"1": 0, "2": 0, "4": 8, "8": 8}
    # pass^1 is the pass rate. pass^2 of a scenario is 1 only if both samples passed: dv-002 in both styles.
    # Families: 0, 0, 1, 0 -> 0.25.
    assert base["pass_hat_k"] == {"1": 0.4375, "2": 0.25, "4": None, "8": None}
    # All wordings pass: style rates multiplied. dv-000 0.5 * 0, dv-001 0 * 0, dv-002 1 * 1,
    # dv-003 0.5 * 0.5 -> (0 + 0 + 1 + 0.25) / 4.
    assert base["all_wordings_pass"] == 0.3125
    # Rewards of the 16 episodes: six clean passes, the one at 95, nine fails.
    assert base["reward_distribution"] == {"0": 9, "95": 1, "100": 6}
    assert sum(base["reward_distribution"].values()) == base["n_episodes"]
    # The other numbers are the ones of the metric functions, which have their own hand-made tests.
    assert base["variance_shares"] == metrics.variance_shares(arms["base"])
    assert base["turn_stats"] == metrics.turn_stats(arms["base"])
    assert base["outcome_labels"] == metrics.outcome_labels(arms["base"])
    # 7 closed correctly, 9 closed wrongly (the hand-made fails close the request).
    assert base["outcome_labels"]["closed_correctly"] == 7 / 16
    # The smallest detectable effect of each arm comes from its own records, with the table's seed.
    assert base["smallest_detectable_effect"] == metrics.two_halves_mde(arms["base"], 3)
    assert table["arms"][1]["smallest_detectable_effect"] == metrics.two_halves_mde(arms["sft"], 3)
    assert json.loads(json.dumps(table, allow_nan=False)) == table


def test_pass_at_8_and_pass_hat_8_reach_the_table(main_mode):
    def arm(name: str, first: str, second: str) -> list[dict]:
        rewards = {"P": P, "F": F}
        return (scenario_records("dv-000", 1, [rewards[x] for x in first], arm=name)
                + scenario_records("dv-001", 1, [rewards[x] for x in second], arm=name))

    table = tables.compare_arms({"base": arm("base", "FFFFFFFP", "FFFFFFFF"),
                                 "sft": arm("sft", "PPPPPPPP", "PPPPFPPP")}, "base", n_draws=100)
    base, sft = table["arms"]
    # base: one of 8 passes in dv-000 -> pass@8 1; none in dv-001 -> 0. pass^8: 0 and 0.
    assert (base["pass_at_8"], base["pass_hat_k"]["8"]) == (0.5, 0.0)
    # sft: pass@8 1 and 1. pass^8: 1 (all 8 pass) and 0 (one fail). pass^1 = (1 + 7/8) / 2.
    assert (sft["pass_at_8"], sft["pass_hat_k"]["8"], sft["pass_hat_k"]["1"]) == (1.0, 0.5, 0.9375)
    assert base["n_scenarios_left_out"] == {"1": 0, "2": 0, "4": 0, "8": 0}
    lines = tables.to_markdown(table).splitlines()
    header = next(line for line in lines if line.startswith("| Arm | Pass@8 |"))
    row = lines[lines.index(header) + 3]
    assert header.startswith("| Arm | Pass@8 | Pass^1 | Pass^2 | Pass^4 | Pass^8 | All wordings pass |")
    # sft: pass@8 100, pass^1 93.8, pass^2 = (1 + 21/28) / 2 = 87.5, pass^4 = (1 + 35/70) / 2 = 75,
    # pass^8 50, all wordings (one wording each) 93.8.
    assert row.startswith("| sft | 100.0 | 93.8 | 87.5 | 75.0 | 50.0 | 93.8 |")
    assert row.count("|") == header.count("|")
    effect = sft["smallest_detectable_effect"]
    lowest, highest = 100 * effect["mde_lowest_split"], 100 * effect["mde_highest_split"]
    assert row.endswith(f"| {100 * effect['mde']:.1f} ({effect['n_splits']} splits; one split alone "
                        f"{lowest:.1f} to {highest:.1f}) |")
    assert "- sft: reward 0: 1, reward 100: 15" in lines


def test_the_column_of_malformed_messages_says_what_it_counts(main_mode):
    lines = tables.to_markdown(tables.compare_arms(small_example(), "base", n_draws=100)).splitlines()
    assert lines[3].endswith("| Malformed messages (% of assistant messages) |")
    assert "Invalid calls" not in "\n".join(lines)
    assert any(line.startswith("Malformed messages: assistant messages that are not exactly one valid tool "
                               "call") and "tool refused" in line for line in lines)
