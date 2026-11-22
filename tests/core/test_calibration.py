"""Tests of accessdesk/eval/calibration.py and scripts/calibration_report.py on hand-made episode records.

Most tests use one small run, `small_run()`. Every number that the report gives for it is worked out by
hand in the comment of the test that checks it. No test reads a saved model run.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from accessdesk import impl
from accessdesk.config import DEFAULT
from accessdesk.eval import calibration
from accessdesk.harness.records import write_records
from tests.helpers.make_records import make_record

ROOT = Path(__file__).resolve().parents[2]
RESOLUTION = {"grant": "granted", "approval": "pending", "already": "already", "inactive": "denied"}
BUDGET = {"grant": 5, "approval": 5, "already": 4, "inactive": 4}     # shortest correct path plus one turn
OUTCOME_OF = {"ca-000": "grant", "ca-001": "approval", "ca-002": "already", "ca-003": "inactive"}


# ---------------------------------------------------------------------------
# Hand-made families and episodes
# ---------------------------------------------------------------------------
def ids(family_id: str) -> tuple[str, str, str]:
    """The person meant, the role asked for, and the person's manager."""
    return f"p-{family_id}", f"r-{family_id}", f"m-{family_id}"


def state(resolution: str | None, person: str | None, role: str | None, *, grants=(), approvals=(),
          access=()) -> dict:
    """A final state or an expected state, written out by hand."""
    return {"access": [list(pair) for pair in access], "approvals": [list(row) for row in approvals],
            "grants": [list(pair) for pair in grants],
            "request": {"resolution": resolution, "person_id": person, "role_id": role}}


def expected_state(family_id: str, outcome: str) -> dict:
    """The correct end state of a family, written by hand for each outcome (not taken from the generator)."""
    p, r, m = ids(family_id)
    if outcome == "grant":          # the role is given: one grant record, and the person now holds the role
        return state("granted", p, r, grants=[(p, r)], access=[(p, r)])
    if outcome == "approval":       # nothing is given: one approval record for the manager
        return state("pending", p, r, approvals=[(p, r, m)])
    if outcome == "already":        # nothing is written: the person held the role before
        return state("already", p, r, access=[(p, r)])
    return state("denied", p, r)    # inactive account: nothing is written


def family(family_id: str, outcome: str | None = None) -> dict:
    """A family record with only the keys that the calibration code reads."""
    outcome = outcome or OUTCOME_OF[family_id]
    p, r, m = ids(family_id)
    return {"family_id": family_id, "split": "calib", "outcome": outcome,
            "truth": {"person_id": p, "role_id": r, "manager_id": m},
            "expected": {"default": expected_state(family_id, outcome)},
            "turn_budget": {"default": BUDGET[outcome]}}


def episode(family_id: str, style: int, sample: int, *, passed: bool = True, reward: int | None = None,
            closed: bool = True, n_turns: int | None = None, parse_errors: tuple[str, ...] = (),
            tool_errors: tuple[str, ...] = (), tokens: list[int] | None = None, final: dict | None = None,
            harness_error: str | None = None, outcome: str | None = None) -> dict:
    """One episode record. The malformed messages come first, one per entry of `parse_errors`; the tool
    errors sit on the first well-formed calls; `tokens` gives n_tokens of every turn in order.

    The final state is the family's correct end state for a pass, `final` for a fail that closed, and an
    untouched open request for an episode that never closed."""
    outcome = outcome or OUTCOME_OF[family_id]
    record = make_record(family_id, style, sample, passed=passed, reward=reward, outcome=outcome,
                         closed=closed, n_turns=n_turns, n_malformed=len(parse_errors),
                         harness_error=harness_error)
    for turn, error in zip(record["turns"], parse_errors):
        turn["parse_error"] = error
        turn["truncated"] = error == "truncated"
    well_formed = [turn for turn in record["turns"] if turn["parse_ok"]]
    for turn, error in zip(well_formed, tool_errors):
        turn["tool_error"] = error
    record["n_tool_errors"] = len(tool_errors)
    if tokens is not None:
        assert len(tokens) == record["n_turns"]
        for turn, n_tokens in zip(record["turns"], tokens):
            turn["n_tokens"] = n_tokens
    if passed:
        record["final_state"] = expected_state(family_id, outcome)
    elif closed:
        record["final_state"] = final
    else:
        record["final_state"] = state(None, None, None)
    return record


def small_run() -> list[dict]:
    """Four calibration families (one per outcome), styles 1 and 4, two samples each: 15 episodes and one
    harness error.

    id   scenario    result                                       turns  reward  n_tokens per turn
    E1   ca-000/s1   pass, clean                                    4     100    30 40 50 60
    E2   ca-000/s1   pass, 1 malformed (unknown_argument)           5      95    41* 30 40 50 60
    E3   ca-000/s4   pass, 2 malformed (no_call, truncated)         6      88    20* 160* 30 40 50 60
    E4   ca-000/s4   fail: role granted to ANOTHER person, closed   4       0    30 40 50 60
    E5   ca-001/s1   pass, clean                                    4     100    30 40 70 60
    E6   ca-001/s1   fail: granted without approval, closed         4       0    30 40 50 60
    E7   ca-001/s4   fail: never closed, 2 tool errors              8       0    30 x 8
    E8   ca-001/s4   fail: all 8 messages malformed (no_call)       8       0    10* x 8
    E9   ca-002/s1   pass, clean                                    3     100    30 40 60
    E10  ca-002/s1   pass, clean                                    3     100    30 40 60
    E11  ca-002/s4   pass, clean                                    3     100    30 40 60
    E12  ca-002/s4   pass, 2 extra reads                            5      98    30 30 30 40 60
    E13  ca-003/s1   fail: approval asked, closed as pending        4       0    30 40 70 60
    E14  ca-003/s1   fail: role granted, closed as granted          4       0    30 40 50 60
    E15  ca-003/s4   fail: closed as denied with a wrong role id    3       0    30 40 60
    H    ca-003/s4   harness error (left out of every number)
    (* = a malformed message.)  ca-000 grant, ca-001 approval, ca-002 already, ca-003 inactive.
    Turn budgets: grant 5, approval 5, already 4, inactive 4.
    Rewards: E2 100 - 5 = 95 (5 turns is inside the budget of 5). E3 100 - 2 x 5 - 2 x (6 - 5) = 88.
    E12 100 - 2 x (5 - 4) = 98.
    """
    p0, r0, _ = ids("ca-000")
    p1, r1, _ = ids("ca-001")
    p3, r3, m3 = ids("ca-003")
    return [
        episode("ca-000", 1, 0, tokens=[30, 40, 50, 60]),                                              # E1
        episode("ca-000", 1, 1, reward=95, parse_errors=("unknown_argument",),
                tokens=[41, 30, 40, 50, 60]),                                                          # E2
        episode("ca-000", 4, 0, reward=88, parse_errors=("no_call", "truncated"),
                tokens=[20, 160, 30, 40, 50, 60]),                                                     # E3
        episode("ca-000", 4, 1, passed=False, tokens=[30, 40, 50, 60],
                final=state("granted", "x-ca-000", r0, grants=[("x-ca-000", r0)],
                            access=[("x-ca-000", r0)])),                                               # E4
        episode("ca-001", 1, 0, tokens=[30, 40, 70, 60]),                                              # E5
        episode("ca-001", 1, 1, passed=False, tokens=[30, 40, 50, 60],
                final=state("granted", p1, r1, grants=[(p1, r1)], access=[(p1, r1)])),                 # E6
        episode("ca-001", 4, 0, passed=False, closed=False,
                tool_errors=("unknown_person_id", "unknown_person_id"), tokens=[30] * 8),              # E7
        episode("ca-001", 4, 1, passed=False, closed=False, parse_errors=("no_call",) * 8,
                tokens=[10] * 8),                                                                      # E8
        episode("ca-002", 1, 0, tokens=[30, 40, 60]),                                                  # E9
        episode("ca-002", 1, 1, tokens=[30, 40, 60]),                                                  # E10
        episode("ca-002", 4, 0, tokens=[30, 40, 60]),                                                  # E11
        episode("ca-002", 4, 1, reward=98, n_turns=5, tokens=[30, 30, 30, 40, 60]),                    # E12
        episode("ca-003", 1, 0, passed=False, n_turns=4, tokens=[30, 40, 70, 60],
                final=state("pending", p3, r3, approvals=[(p3, r3, m3)])),                             # E13
        episode("ca-003", 1, 1, passed=False, n_turns=4, tokens=[30, 40, 50, 60],
                final=state("granted", p3, r3, grants=[(p3, r3)], access=[(p3, r3)])),                 # E14
        episode("ca-003", 4, 0, passed=False, n_turns=3, tokens=[30, 40, 60],
                final=state("denied", p3, "x-role")),                                                  # E15
        harness_error_record("ca-003", 4, 1),                                                          # H
    ]


def harness_error_record(family_id: str, style: int, sample: int) -> dict:
    """A record as the harness writes it after a failure: marked as a harness error, and filled with values
    that would change every number of the report if the record were counted."""
    record = episode(family_id, style, sample, passed=True, reward=100, harness_error="CUDA out of memory",
                     parse_errors=("truncated",) * 2, tool_errors=("unknown_role_id",))
    for turn in record["turns"]:
        turn["n_tokens"] = 500
    record["final_state"]["grants"].append(["x-person", "x-role"])
    return record


def small_families() -> list[dict]:
    return [family(family_id) for family_id in OUTCOME_OF]


def small_report() -> dict:
    return calibration.run_report("small", small_run(), families=small_families())


def scenario(family_id: str, style: int, results: str, outcome: str | None = None) -> list[dict]:
    """The samples of one scenario from a string: "P" a clean pass, "F" a fail that never closed."""
    return [episode(family_id, style, sample, passed=letter == "P", closed=letter == "P", outcome=outcome)
            for sample, letter in enumerate(results)]


def four_families(results: dict[str, str]) -> list[dict]:
    """One scenario (style 1) per family; `results` maps the family id to its samples, as in `scenario`."""
    return [r for family_id, letters in results.items() for r in scenario(family_id, 1, letters)]


def test_the_hand_made_run_is_what_its_table_says():
    """The counters of the small run agree with the table in its docstring (a typing error in the table or
    in the builder would otherwise sit under every other test)."""
    run = small_run()
    assert [r["n_turns"] for r in run[:15]] == [4, 5, 6, 4, 4, 4, 8, 8, 3, 3, 3, 5, 4, 4, 3]
    assert [r["reward"] for r in run[:15]] == [100, 95, 88, 0, 100, 0, 0, 0, 100, 100, 100, 98, 0, 0, 0]
    assert [r["n_malformed"] for r in run[:15]] == [0, 1, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0]
    assert [r["turn_budget"] for r in run[:15]] == [5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4]
    assert run[15]["harness_error"] is not None
    # The reward of each pass follows the reward rule: 100 - 5 per malformed - 2 per turn over the budget.
    for r in run[:15]:
        if r["passed"]:
            assert r["reward"] == 100 - 5 * r["n_malformed"] - 2 * max(0, r["n_turns"] - r["turn_budget"])


# ---------------------------------------------------------------------------
# 1. Size and health
# ---------------------------------------------------------------------------
def test_size_and_health():
    size = small_report()["size"]
    assert size["n_records"] == 16                       # E1..E15 and H
    assert size["n_harness_errors"] == 1                 # H
    assert size["harness_error_share"] == 1 / 16         # 0.0625
    assert size["n_episodes"] == 15
    assert size["n_families"] == 4
    assert size["n_families_by_outcome"] == {"grant": 1, "approval": 1, "already": 1, "inactive": 1}
    assert size["n_scenarios"] == 8                      # 4 families x styles 1 and 4
    assert size["n_scenarios_without_valid_sample"] == 0
    assert size["split"] == ["calib"]
    assert (size["ruleset"], size["toolset"], size["prompt"]) == ("default", "five", "final")
    assert size["placeholder_text"] is True
    assert size["text_sha256"] == ["0" * 64]
    made_by = small_run()[0]["impl"]                     # what the record helper writes
    assert size["impl"] == {"verify": [made_by["verify"]], "reward": [made_by["reward"]]}
    assert size["data_sha256"] == []                     # the hand-made records name no scenario file
    assert size["manifest_found"] is False and size["model"] is None


def test_unequal_samples_per_scenario_are_reported_not_hidden():
    # ca-003/s4 has one valid sample (E15; H is a harness error); the other 7 scenarios have two.
    report = small_report()
    samples = report["size"]["samples_per_scenario"]
    assert samples == {"min": 1, "max": 2, "all_equal": False,
                       "table": [{"samples": 1, "scenarios": 1}, {"samples": 2, "scenarios": 7}]}
    assert any("do not all have the same number of valid samples (1 to 2)" in w for w in report["warnings"])
    markdown = calibration.to_markdown(calibration.build_report({"small": small_run()}))
    assert "NOT the same in every scenario: 1 sample in 1 scenario, 2 samples in 7 scenarios" in markdown
    # With the same number everywhere, the report says that instead.
    equal = calibration.run_report("equal", four_families({"ca-000": "PF", "ca-001": "PF"}))
    assert equal["size"]["samples_per_scenario"] == {"min": 2, "max": 2, "all_equal": True,
                                                     "table": [{"samples": 2, "scenarios": 2}]}
    assert not any("same number of valid samples" in w for w in equal["warnings"])


def test_harness_errors_are_excluded_and_counted():
    without = [r for r in small_run() if r["harness_error"] is None]
    extra = [harness_error_record("ca-000", 1, 2), harness_error_record("ca-001", 4, 2),
             harness_error_record("ca-002", 2, 0)]         # the last one is a scenario with no valid sample
    clean = calibration.run_report("small", without, families=small_families())
    dirty = calibration.run_report("small", without + extra, families=small_families())
    for section in ("pass", "checks", "ties", "call_format", "turns", "penalties", "failures", "sft_supply"):
        assert dirty[section] == clean[section], section
    assert clean["size"]["n_harness_errors"] == 0 and clean["size"]["harness_error_share"] == 0.0
    assert dirty["size"]["n_records"] == 18              # 15 + 3
    assert dirty["size"]["n_harness_errors"] == 3
    assert dirty["size"]["harness_error_share"] == 3 / 18
    assert dirty["size"]["n_episodes"] == 15
    assert dirty["size"]["n_scenarios"] == 8             # ca-002/s2 has no valid sample: it is not a ninth
    assert dirty["size"]["n_scenarios_without_valid_sample"] == 1
    assert dirty["size"]["n_families"] == 4
    # 3 of 18 records is 16.7%: far above the plan's gate of 1%, and the report says so.
    assert any("Harness errors are 16.7% of the records" in w for w in dirty["warnings"])
    assert not any("Harness errors" in w for w in clean["warnings"])


def test_a_sample_twice_or_two_conditions_in_one_run_are_refused():
    twice = small_run() + [episode("ca-000", 1, 0)]
    with pytest.raises(ValueError, match="more than once"):
        calibration.run_report("small", twice)
    mixed = small_run()
    mixed[0]["toolset"] = "six"
    with pytest.raises(ValueError, match="exactly one"):
        calibration.run_report("small", mixed)


def test_model_text_and_config_come_from_the_manifest():
    manifest = {"complete": True, "model": "some/model", "revision": "abc", "adapter": None,
                "text_file": "some_text.yaml", "text_sha256": "0" * 64, "text_is_placeholder": True,
                "config": {**DEFAULT.to_dict(), "max_new_tokens": 200}}
    report = calibration.run_report("small", small_run(), manifest=manifest)
    assert report["size"]["model"] == "some/model" and report["size"]["revision"] == "abc"
    assert report["size"]["text_file"] == "some_text.yaml" and report["size"]["manifest_found"] is True
    assert report["config_source"] == "run_manifest.json"
    assert report["turns"]["tokens"]["cap_in_force"] == 200
    # Without a manifest the cap is the default config's, because every record carries that config's hash.
    assert small_report()["turns"]["tokens"]["cap_in_force"] == DEFAULT.max_new_tokens
    # With an unknown config hash and no manifest the cap in force is not known.
    unknown = small_run()
    for r in unknown:
        r["config_sha256"] = "f" * 64
    assert calibration.run_report("small", unknown)["turns"]["tokens"]["cap_in_force"] is None
    # A manifest that disagrees with the records, or an unfinished run, is said.
    odd = calibration.run_report("small", small_run(), manifest={**manifest, "text_sha256": "1" * 64,
                                                                 "complete": False})
    assert any("did not finish" in w for w in odd["warnings"])
    assert any("text hash of the run manifest" in w for w in odd["warnings"])


def test_a_run_on_dev_families_gets_a_warning():
    dev = [r for family_id in ("dv-000", "dv-001") for r in scenario(family_id, 1, "PF", outcome="grant")]
    report = calibration.run_report("dev-run", dev)
    assert report["size"]["split"] == ["dev"]
    assert any("split dev, not on the throwaway calibration families" in w for w in report["warnings"])
    assert not any("throwaway" in w for w in small_report()["warnings"])


# ---------------------------------------------------------------------------
# 2. Pass rate
# ---------------------------------------------------------------------------
def test_pass_rates():
    rates = small_report()["pass"]
    # Family pass rates: ca-000 E1 E2 E3 pass, E4 fails -> 3/4. ca-001 only E5 passes -> 1/4.
    # ca-002 4/4. ca-003 0/3 (H is left out). Pass rate = (0.75 + 0.25 + 1 + 0) / 4 = 0.5.
    assert rates["pass_rate"] == pytest.approx(0.5)
    assert rates["n_families"] == 4
    assert rates["by_outcome"] == {"grant": {"pass_rate": 0.75, "n_families": 1},
                                   "approval": {"pass_rate": 0.25, "n_families": 1},
                                   "already": {"pass_rate": 1.0, "n_families": 1},
                                   "inactive": {"pass_rate": 0.0, "n_families": 1}}
    # Style 1: ca-000 2/2, ca-001 1/2, ca-002 2/2, ca-003 0/2 -> (1 + 0.5 + 1 + 0) / 4 = 0.625.
    # Style 4: ca-000 1/2, ca-001 0/2, ca-002 2/2, ca-003 0/1 -> (0.5 + 0 + 1 + 0) / 4 = 0.375.
    assert rates["by_style"] == {"1": {"pass_rate": 0.625, "n_families": 4},
                                 "4": {"pass_rate": 0.375, "n_families": 4}}
    # Style 1 is the only seen style of this run, style 4 is the held-out one.
    assert rates["seen_held_out"] == {"seen": {"pass_rate": 0.625, "n_families": 4},
                                      "held_out": {"pass_rate": 0.375, "n_families": 4}}
    # Per family, held-out minus seen: 0.5 - 1, 0 - 0.5, 1 - 1, 0 - 0 = -0.5, -0.5, 0, 0. Mean -0.25.
    diff = rates["held_out_minus_seen"]
    assert diff["diff"] == pytest.approx(-0.25) and diff["n_families"] == 4
    # A draw of 4 families is four times one of the first two with probability (1/2)**4 = 6.25%, more
    # than 2.5%: the lower end is their difference, -0.5. In the same way the upper end is 0.
    assert (diff["lo"], diff["hi"]) == (-0.5, 0.0)


def test_interval_of_the_pass_rate_resamples_families():
    # The family rates are 0.75, 0.25, 1 and 0. A draw takes 4 of them with replacement; its sum is a
    # multiple of 0.25. Of the 4**4 = 256 equally likely draws: sum 0 in 1 (four times 0); sum 0.25 in 4
    # (one 0.25, three 0); sum 0.5 in 6 (two 0.25, two 0). So P(sum <= 0.25) = 5/256 = 2.0% and
    # P(sum <= 0.5) = 11/256 = 4.3%: the 2.5th percentile of the sum is 0.5, a mean of 0.125. The four
    # rates are symmetric around 0.5 (0 and 1, 0.25 and 0.75), so the 97.5th percentile is 1 - 0.125.
    rates = small_report()["pass"]
    assert rates["lo"] == pytest.approx(0.125)
    assert rates["hi"] == pytest.approx(0.875)
    # Seven samples of 16 pass in every family: each draw of families has the same mean, 7/16.
    same = four_families({f: "PPPPPPPFFFFFFFFF" for f in OUTCOME_OF})
    rates = calibration.pass_rates(same)
    assert (rates["pass_rate"], rates["lo"], rates["hi"]) == (7 / 16, 7 / 16, 7 / 16)
    # The interval comes from families, not from episodes: 2 families of 50 episodes each, one always
    # passes and one always fails. A draw is (0, 0), (0, 1), (1, 0) or (1, 1): mean 0 or 1 in a quarter
    # of the draws each, so the 2.5th percentile is 0 and the 97.5th is 1. An interval from the 100
    # episodes would be about 0.40 to 0.60.
    two = scenario("ca-000", 1, "P" * 50) + scenario("ca-001", 1, "F" * 50)
    rates = calibration.pass_rates(two)
    assert (rates["pass_rate"], rates["lo"], rates["hi"]) == (0.5, 0.0, 1.0)


def test_held_out_minus_seen_is_absent_when_it_cannot_be_paired():
    only_seen = four_families({"ca-000": "PF", "ca-001": "PF"})
    assert calibration.pass_rates(only_seen)["held_out_minus_seen"] is None
    # ca-001 has no episode in the held-out style: the two parts do not hold the same families.
    uneven = only_seen + scenario("ca-000", 4, "PP")
    rates = calibration.pass_rates(uneven)
    assert rates["held_out_minus_seen"] is None
    assert rates["seen_held_out"] == {"seen": {"pass_rate": 0.5, "n_families": 2},
                                      "held_out": {"pass_rate": 1.0, "n_families": 1}}
    markdown = calibration.to_markdown(calibration.build_report({"u": uneven}))
    assert "Held-out minus seen: not given" in markdown


# ---------------------------------------------------------------------------
# 3. The two checks of the plan
# ---------------------------------------------------------------------------
def test_plan_checks():
    checks = small_report()["checks"]
    # Floor: the lower end of the interval is 0.125 (test above), which is not above 0.25 -> no.
    assert checks["floor"] == {"threshold": 0.25, "interval_lo": pytest.approx(0.125), "passed": False,
                               "best_constant_policy": 0.25}      # 1 family of 4 per outcome
    # Band: the pass rate is 0.5, and 0.30 <= 0.5 <= 0.65 -> yes.
    assert checks["band"] == {"lo": 0.30, "hi": 0.65, "pass_rate": pytest.approx(0.5), "passed": True,
                              "position": "inside"}
    # Room: 1 - 0.5 over all; 1 - 0.75, 1 - 0.25, 1 - 1, 1 - 0 per outcome.
    assert checks["room"]["all"] == pytest.approx(0.5)
    assert checks["room"]["by_outcome"] == {"grant": 0.25, "approval": 0.75, "already": 0.0, "inactive": 1.0}
    markdown = calibration.to_markdown(calibration.build_report({"small": small_run()}))
    assert "above 25.0%? **NO**: the lower end is 12.5%" in markdown
    assert "between 30.0% and 65.0%? **YES**: the pass rate is 50.0%, inside the band" in markdown


def rates_stub(pass_rate: float, lo: float) -> dict:
    return {"pass_rate": pass_rate, "lo": lo, "by_outcome": {"grant": {"pass_rate": pass_rate}}}


BALANCED = {"grant": 2, "approval": 2, "already": 2, "inactive": 2}


@pytest.mark.parametrize("lo, passed", [(0.25, False), (0.2499, False), (0.2501, True), (0.9, True)])
def test_floor_check_needs_the_interval_above_a_quarter(lo, passed):
    assert calibration.plan_checks(rates_stub(0.5, lo), BALANCED)["floor"]["passed"] is passed


@pytest.mark.parametrize("pass_rate, passed, position", [
    (0.2999, False, "below"), (0.30, True, "inside"), (0.5, True, "inside"), (0.65, True, "inside"),
    (0.6501, False, "above"), (0.97, False, "above"), (0.0, False, "below")])
def test_band_check_holds_from_30_to_65_percent_with_both_ends(pass_rate, passed, position):
    band = calibration.plan_checks(rates_stub(pass_rate, 0.0), BALANCED)["band"]
    assert (band["passed"], band["position"]) == (passed, position)


def test_both_checks_can_say_yes_on_a_run():
    # Every family passes 7 of 16: pass rate 0.4375, and the interval is 0.4375 to 0.4375.
    report = calibration.run_report("r", four_families({f: "PPPPPPPFFFFFFFFF" for f in OUTCOME_OF}))
    assert report["checks"]["floor"]["passed"] is True and report["checks"]["band"]["passed"] is True
    assert report["checks"]["room"]["all"] == 1 - 7 / 16


def test_unbalanced_outcomes_are_named_beside_the_floor():
    # 3 families: two are grant, one is approval. A policy that always grants passes 2 of 3 families.
    records = (scenario("ca-000", 1, "PF", "grant") + scenario("ca-004", 1, "PF", "grant")
               + scenario("ca-001", 1, "PF", "approval"))
    report = calibration.run_report("r", records)
    assert report["checks"]["floor"]["best_constant_policy"] == pytest.approx(2 / 3)
    assert report["checks"]["floor"]["threshold"] == 0.25
    assert any("best constant policy passes 66.7% of these families" in w for w in report["warnings"])
    assert not any("not balanced" in w for w in small_report()["warnings"])


# ---------------------------------------------------------------------------
# 4. Ties
# ---------------------------------------------------------------------------
def test_ties():
    tie = small_report()["ties"]
    # Scenarios with 2 valid samples (ca-003/s4 has one and is left out), by their two rewards:
    #   ca-000/s1  100, 95   all pass, rewards differ      ca-000/s4  88, 0    mixed
    #   ca-001/s1  100, 0    mixed                         ca-001/s4  0, 0     same reward, all fail
    #   ca-002/s1  100, 100  same reward, all pass         ca-002/s4  100, 98  all pass, rewards differ
    #   ca-003/s1  0, 0      same reward, all fail
    # Same reward: 3 of 7 (1 all pass, 2 all fail). Same pass or fail: these 3 and the 2 "all pass,
    # rewards differ" = 5 of 7.
    assert tie["all"] == {"n_scenarios": 7, "equal_reward": pytest.approx(3 / 7),
                          "equal_reward_all_pass": pytest.approx(1 / 7),
                          "equal_reward_all_fail": pytest.approx(2 / 7),
                          "equal_passfail": pytest.approx(5 / 7)}
    assert tie["by_outcome"] == {
        # grant: s1 all pass with different rewards, s4 mixed.
        "grant": {"n_scenarios": 2, "equal_reward": 0.0, "equal_reward_all_pass": 0.0,
                  "equal_reward_all_fail": 0.0, "equal_passfail": 0.5},
        # approval: s1 mixed, s4 all fail.
        "approval": {"n_scenarios": 2, "equal_reward": 0.5, "equal_reward_all_pass": 0.0,
                     "equal_reward_all_fail": 0.5, "equal_passfail": 0.5},
        # already: s1 same reward all pass, s4 all pass with different rewards.
        "already": {"n_scenarios": 2, "equal_reward": 0.5, "equal_reward_all_pass": 0.5,
                    "equal_reward_all_fail": 0.0, "equal_passfail": 1.0},
        # inactive: only s1 is counted, and it is all fail.
        "inactive": {"n_scenarios": 1, "equal_reward": 1.0, "equal_reward_all_pass": 0.0,
                     "equal_reward_all_fail": 1.0, "equal_passfail": 1.0},
    }
    # The shares were computed on 7 scenarios of 2 samples; one scenario was left out.
    assert tie["samples_per_scenario"] == [{"samples": 2, "scenarios": 7}]
    assert tie["n_scenarios_left_out"] == 1
    # ca-002 always passes (4/4), ca-003 always fails (0/3), ca-000 (3/4) and ca-001 (1/4) are mixed.
    assert tie["family_profile"] == {"n_families": 4, "always_pass": 1, "always_fail": 1, "mixed": 2}
    markdown = calibration.to_markdown(calibration.build_report({"small": small_run()}))
    assert ("Computed with: 2 samples in 7 scenarios; left out for having fewer than 2 valid samples: "
            "1 scenario.") in markdown
    assert "of 4 families, 1 always pass, 1 always fail, 2 are mixed" in markdown


# ---------------------------------------------------------------------------
# 5. Call format
# ---------------------------------------------------------------------------
def test_call_format():
    calls = small_report()["call_format"]
    # Assistant messages: 4 + 5 + 6 + 4 (ca-000) + 4 + 4 + 8 + 8 (ca-001) + 3 + 3 + 3 + 5 (ca-002)
    # + 4 + 4 + 3 (ca-003) = 19 + 24 + 14 + 11 = 68.
    assert calls["n_messages"] == 68
    # Malformed: E2 1, E3 2, E8 8 = 11.
    assert calls["n_malformed"] == 11
    assert calls["invalid_call_rate"] == pytest.approx(11 / 68)
    # E2 unknown_argument; E3 no_call and truncated; E8 eight times no_call. In the parser's order.
    assert calls["parse_errors"] == {"no_call": 9, "unknown_argument": 1, "truncated": 1}
    assert list(calls["parse_errors"]) == ["no_call", "unknown_argument", "truncated"]
    # E7: two calls with an unknown person id.
    assert calls["n_tool_errors"] == 2 and calls["tool_errors"] == {"unknown_person_id": 2}
    # One message was cut off by the token cap (E3's second).
    assert calls["n_truncated"] == 1 and calls["truncated_share"] == pytest.approx(1 / 68)
    assert calls["counters_agree"] is True


def test_counters_that_disagree_with_the_turns_are_said():
    records = small_run()
    records[0]["n_malformed"] = 1                     # E1 has no malformed turn in its saved turns
    report = calibration.run_report("small", records)
    assert report["call_format"]["counters_agree"] is False
    assert any("counters of the episodes do not agree" in w for w in report["warnings"])
    assert not any("counters" in w for w in small_report()["warnings"])


# ---------------------------------------------------------------------------
# 6. Turns and tokens
# ---------------------------------------------------------------------------
def test_turns():
    turns = small_report()["turns"]
    assert turns["n_episodes"] == 15
    assert turns["mean_turns"] == pytest.approx(68 / 15)                     # 68 messages, 15 episodes
    # grant (4 + 5 + 6 + 4) / 4, approval (4 + 4 + 8 + 8) / 4, already (3 + 3 + 3 + 5) / 4,
    # inactive (4 + 4 + 3) / 3.
    assert turns["mean_turns_by_outcome"] == {"grant": 4.75, "approval": 6.0, "already": 3.5,
                                              "inactive": pytest.approx(11 / 3)}
    # E7 and E8 used all 8 turns without closing.
    assert turns["share_at_turn_limit"] == pytest.approx(2 / 15)
    # Over the budget: E3 (6 > 5), E7 and E8 (8 > 5), E12 (5 > 4). E2 used 5 of 5: inside.
    assert turns["n_over_budget"] == 4 and turns["share_over_budget"] == pytest.approx(4 / 15)


def test_tokens_of_well_formed_calls_and_the_cap_by_the_rule():
    tokens = small_report()["turns"]["tokens"]
    # Well-formed calls: 68 messages - 11 malformed = 57. Their lengths (the starred ones are left out):
    #   30: E1-E6 one each (6), E7 eight, E9-E11 one each (3), E12 three, E13 E14 E15 one each (3) = 23
    #   40: E1-E6 (6), E9-E11 (3), E12, E13, E14, E15 (4) = 13
    #   50: E1, E2, E3, E4, E6, E14 = 6
    #   60: the closing call of the 13 closed episodes = 13
    #   70: E5, E13 = 2                                             23 + 13 + 6 + 13 + 2 = 57
    # In rising order the 29th of 57 is a 40 (places 24 to 36): the median.
    # 99th percentile: place ceil(0.99 x 57) = ceil(56.43) = 57, the largest: 70. Cap = 2 x 70 = 140.
    assert tokens == {"n_calls": 57, "n_calls_without_length": 0, "median": 40.0, "p99": 70, "max": 70,
                      "cap_by_rule": 140, "cap_in_force": DEFAULT.max_new_tokens, "n_truncated": 1}


def test_nearest_rank_percentile_and_cap_rule():
    # 100 values 1..100: place ceil(0.99 x 100) = 99 -> 99 (not 100, the largest).
    assert calibration.percentile_nearest_rank(list(range(100, 0, -1)), 99) == 99
    # 200 values: place 198. 101 values: place ceil(99.99) = 100. 5 values: place ceil(4.95) = 5.
    assert calibration.percentile_nearest_rank(list(range(1, 201)), 99) == 198
    assert calibration.percentile_nearest_rank(list(range(1, 102)), 99) == 100
    assert calibration.percentile_nearest_rank([7, 3, 9, 1, 5], 99) == 9
    assert calibration.percentile_nearest_rank([7, 3, 9, 1, 5], 50) == 5       # place ceil(2.5) = 3
    with pytest.raises(ValueError):
        calibration.percentile_nearest_rank([], 99)
    # Twice the percentile, and at least 128: 2 x 63 = 126 -> 128; 2 x 64 = 128; 2 x 65 = 130.
    assert [calibration.token_cap_rule(p99) for p99 in (10, 63, 64, 65, 119)] == [128, 128, 128, 130, 238]


def test_token_numbers_use_well_formed_calls_only():
    # One episode of 5 turns: a truncated message of 160 tokens, a call of 40 that the tool refused, calls
    # of 50 and 55, and the closing call of 60. Well-formed: 40, 50, 55, 60 (a call that the tool refused
    # is still well-formed; the truncated message is not). Median (50 + 55) / 2 = 52.5; 99th percentile =
    # the largest of 4 = 60; cap = max(128, 2 x 60) = 128.
    record = episode("ca-000", 1, 0, reward=95, parse_errors=("truncated",), tool_errors=("unknown_role_id",),
                     tokens=[160, 40, 50, 55, 60])
    tokens = calibration.call_tokens([record], cap_in_force=160)
    assert tokens == {"n_calls": 4, "n_calls_without_length": 0, "median": 52.5, "p99": 60, "max": 60,
                      "cap_by_rule": 128, "cap_in_force": 160, "n_truncated": 1}
    # A well-formed call without a stored length is counted apart and is in no token number.
    record["turns"][2]["n_tokens"] = None                 # the call of 50: 40, 55, 60 are left
    tokens = calibration.call_tokens([record], cap_in_force=None)
    assert (tokens["n_calls"], tokens["n_calls_without_length"], tokens["median"]) == (3, 1, 55.0)


def test_percentile_and_largest_call_are_told_apart():
    # 25 clean passes of 4 turns: 100 well-formed calls with the lengths 1, 2, ..., 100.
    # Median (50 + 51) / 2 = 50.5; 99th percentile = the 99th value = 99; largest 100; cap 2 x 99 = 198.
    records = [episode("ca-000", 1, sample, tokens=[4 * sample + k for k in (1, 2, 3, 4)])
               for sample in range(25)]
    report = calibration.build_report({"long": records})
    tokens = report["runs"]["long"]["turns"]["tokens"]
    assert (tokens["n_calls"], tokens["median"], tokens["p99"], tokens["max"]) == (100, 50.5, 99, 100)
    assert tokens["cap_by_rule"] == 198
    markdown = calibration.to_markdown(report)
    assert ("- New tokens per well-formed call (100 calls, end-of-turn token included): median 50.5, 99th "
            "percentile 99, largest 100.") in markdown
    # 260: without a run manifest the cap in force is the token cap of accessdesk/config.py.
    assert "at least 128): **198**. Cap in force in this run: 260. Messages cut off" in markdown


def test_token_numbers_are_absent_without_stored_lengths():
    records = four_families({"ca-000": "PF", "ca-001": "PF"})        # make_record stores n_tokens = None
    report = calibration.run_report("scripted", records)
    assert report["turns"]["tokens"] is None
    assert any("no token numbers" in w for w in report["warnings"])
    assert "not given (no stored token counts)" in calibration.to_markdown(
        calibration.build_report({"scripted": records}))


# ---------------------------------------------------------------------------
# 7. Penalties
# ---------------------------------------------------------------------------
def test_penalties():
    pen = small_report()["penalties"]
    # With a malformed message: E2 (pass), E3 (pass), E8 (fail). Families: ca-000 2/2, ca-001 0/1 -> 0.5.
    assert pen["malformed"]["with"] == {"n_episodes": 3, "n_passed": 2, "n_families": 2, "pass_rate": 0.5}
    # With none: the other 12. ca-000 E1 pass, E4 fail -> 1/2. ca-001 E5 pass, E6 E7 fail -> 1/3.
    # ca-002 4/4. ca-003 0/3. Passed: 1 + 1 + 4 = 6. Pass rate (1/2 + 1/3 + 1 + 0) / 4 = 11/24.
    assert pen["malformed"]["without"] == {"n_episodes": 12, "n_passed": 6, "n_families": 4,
                                           "pass_rate": pytest.approx(11 / 24)}
    assert pen["malformed"]["difference"] == pytest.approx(0.5 - 11 / 24)        # +0.0417
    # Over the budget: E3 (pass), E7, E8 (fail), E12 (pass). ca-000 1/1, ca-001 0/2, ca-002 1/1 -> 2/3.
    assert pen["turn_budget"]["with"] == {"n_episodes": 4, "n_passed": 2, "n_families": 3,
                                          "pass_rate": pytest.approx(2 / 3)}
    # Inside: the other 11. ca-000 E1 E2 pass, E4 fail -> 2/3. ca-001 E5 pass, E6 fail -> 1/2.
    # ca-002 E9 E10 E11 -> 1. ca-003 0/3. Passed: 2 + 1 + 3 = 6. Pass rate (2/3 + 1/2 + 1 + 0) / 4 = 13/24.
    assert pen["turn_budget"]["without"] == {"n_episodes": 11, "n_passed": 6, "n_families": 4,
                                             "pass_rate": pytest.approx(13 / 24)}
    assert pen["turn_budget"]["difference"] == pytest.approx(2 / 3 - 13 / 24)    # +0.125
    assert pen["n_both"] == 2                                                    # E3 and E8
    # The 8 passes: E1 E5 E9 E10 E11 at 100, E12 at 98, E2 at 95, E3 at 88.
    assert pen["n_passes"] == 8
    assert pen["rewards_of_passes"] == [{"reward": 100, "episodes": 5}, {"reward": 98, "episodes": 1},
                                        {"reward": 95, "episodes": 1}, {"reward": 88, "episodes": 1}]
    # The sentences give these numbers in words: 0.5 - 11/24 = 4.2 points, 2/3 - 13/24 = 12.5 points.
    # Each sentence gives the episodes and the passes of each group first, and then the pass rates as a
    # mean over families: the two are different numbers (2 of 3 episodes is 66.7%, the family mean is 50%).
    assert pen["malformed"]["sentence"] == (
        "Of the 3 episodes with a malformed call 2 passed, and of the 12 episodes with no malformed call 6 "
        "passed. As a mean over families (2 and 4 families) the pass rates are 50.0% and 45.8%: the pass "
        "rate is 4.2 points higher with a malformed call.")
    assert pen["turn_budget"]["sentence"] == (
        "Of the 4 episodes over the turn budget 2 passed, and of the 11 episodes inside the turn budget 6 "
        "passed. As a mean over families (3 and 4 families) the pass rates are 66.7% and 54.2%: the pass "
        "rate is 12.5 points higher over the turn budget.")


def test_penalties_among_the_episodes_that_closed_the_request():
    """An episode that never closes fails and uses all 8 turns, so it is over its budget by construction
    and has the most messages that can be malformed. Over all episodes the two `with` groups therefore
    hold these failures whatever the penalty is; the comparison that can show something is the one among
    the episodes that closed the request."""
    pen = small_report()["penalties"]
    # Never closed: E7 (no malformed message, 8 turns) and E8 (all 8 malformed). Both are over the budget.
    assert pen["malformed"]["never_closed"] == {"n_episodes": 2, "n_with": 1, "n_passed": 0}
    assert pen["turn_budget"]["never_closed"] == {"n_episodes": 2, "n_with": 2, "n_passed": 0}
    closed = pen["malformed"]["closed_only"]
    # Closed, with a malformed message: E2, E3 (both pass; E8 is gone). One family, ca-000: 2/2.
    assert closed["with"] == {"n_episodes": 2, "n_passed": 2, "n_families": 1, "pass_rate": 1.0}
    # Closed, with none: the 12 without E7. ca-000 1/2, ca-001 E5 pass, E6 fail -> 1/2, ca-002 4/4,
    # ca-003 0/3: (1/2 + 1/2 + 1 + 0) / 4 = 0.5.
    assert closed["without"] == {"n_episodes": 11, "n_passed": 6, "n_families": 4, "pass_rate": 0.5}
    assert closed["difference"] == 0.5                       # over all episodes it was +0.0417
    assert closed["sentence"] == (
        "Among the episodes that closed the request: of the 2 episodes with a malformed call 2 passed, and "
        "of the 11 episodes with no malformed call 6 passed. As a mean over families (1 and 4 families) the "
        "pass rates are 100.0% and 50.0%: the pass rate is 50.0 points higher with a malformed call.")
    closed = pen["turn_budget"]["closed_only"]
    # Closed and over the budget: E3 and E12 (both pass). ca-000 1/1, ca-002 1/1. Inside: nobody of the 11
    # is a never-closed episode, so that group is the one of all episodes.
    assert closed["with"] == {"n_episodes": 2, "n_passed": 2, "n_families": 2, "pass_rate": 1.0}
    assert closed["without"] == pen["turn_budget"]["without"]
    assert closed["difference"] == pytest.approx(1 - 13 / 24)
    assert pen["malformed"]["caution"] == (
        "1 of the 3 episodes with a malformed call never closed the request, and an episode that never "
        "closes cannot pass: to that extent the difference is true by construction, whatever the penalty "
        "is.")
    assert pen["turn_budget"]["caution"].startswith("2 of the 4 episodes over the turn budget never closed")


def test_the_penalty_check_no_longer_shows_a_separation_that_only_never_closed_episodes_make():
    # In each of the 4 families: 6 clean passes, 1 pass that closed one turn over its budget (reward 98),
    # and 1 episode that never closed. Over all episodes "over the budget" looks 50 points worse (per
    # family 1/2 against 6/6). Among the episodes that closed, the over-budget ones pass as often as the
    # others (1/1 against 6/6): the turn penalty separates nothing here.
    records = []
    for family_id, outcome in OUTCOME_OF.items():
        records += [episode(family_id, 1, sample) for sample in range(6)]
        records.append(episode(family_id, 1, 6, reward=98, n_turns=BUDGET[outcome] + 1))
        records.append(episode(family_id, 1, 7, passed=False, closed=False))
    turn = calibration.penalties(records)["turn_budget"]
    assert turn["difference"] == pytest.approx(0.5 - 1.0)
    assert turn["closed_only"]["difference"] == 0.0
    assert "the two pass rates are the same" in turn["closed_only"]["sentence"]
    assert turn["never_closed"] == {"n_episodes": 4, "n_with": 4, "n_passed": 0}
    assert turn["caution"].startswith("4 of the 8 episodes over the turn budget never closed the request")


def test_an_empty_group_among_the_closed_episodes_is_said():
    # The only episode over the budget never closed: among the closed episodes nobody is over the budget.
    records = [episode("ca-000", 1, 0), episode("ca-000", 1, 1, passed=False, closed=False)]
    closed = calibration.penalties(records)["turn_budget"]["closed_only"]
    assert closed["difference"] is None
    assert closed["sentence"] == (
        "Among the episodes that closed the request there is no episode over the turn budget, so this run "
        "does not show whether this penalty separates good from bad episodes (the other group holds 1 "
        "episode).")


def test_penalty_sentences_for_a_lower_rate_an_equal_rate_and_an_empty_group():
    # ca-000: a clean pass and a fail with one malformed message. With: 0/1 = 0. Without: 1/1 = 1.
    records = [episode("ca-000", 1, 0),
               episode("ca-000", 1, 1, passed=False, parse_errors=("no_call",),
                       final=state("denied", "a", "b"))]
    pen = calibration.penalties(records)
    assert pen["malformed"]["difference"] == -1.0
    assert pen["malformed"]["sentence"].endswith("the pass rate is 100.0 points lower with a malformed call.")
    # Nobody is over the budget (4 and 5 turns, budget 5): the report says that it cannot tell.
    assert pen["turn_budget"]["with"] == {"n_episodes": 0, "n_passed": 0, "n_families": 0, "pass_rate": None}
    assert pen["turn_budget"]["difference"] is None
    assert pen["turn_budget"]["sentence"] == (
        "There is no episode over the turn budget, so this run does not show whether this penalty separates "
        "good from bad episodes (the other group holds 2 episodes).")
    # One pass with a malformed message and one without: both groups pass 100%.
    both_pass = [episode("ca-000", 1, 0), episode("ca-000", 1, 1, reward=95, parse_errors=("no_call",))]
    assert "the two pass rates are the same" in calibration.penalties(both_pass)["malformed"]["sentence"]
    # No sentence takes a decision.
    for sentence in (pen["malformed"]["sentence"], pen["turn_budget"]["sentence"]):
        assert not re.search(r"\b(remove|removed|keep|kept|should|must)\b", sentence)


# ---------------------------------------------------------------------------
# 8. What the model does when it fails
# ---------------------------------------------------------------------------
def test_failures():
    fails = small_report()["failures"]
    # Closed and passed: the 8 passes. Closed and failed: E4, E6, E13, E14, E15. Never closed with at
    # least one well-formed call: E7. Every message malformed: E8.
    assert fails["n_episodes"] == 15
    assert fails["label_counts"] == {"closed_correctly": 8, "closed_wrongly": 5, "never_closed": 1,
                                     "malformed_only": 1}
    assert fails["labels"] == {"closed_correctly": pytest.approx(8 / 15),
                               "closed_wrongly": pytest.approx(5 / 15),
                               "never_closed": pytest.approx(1 / 15), "malformed_only": pytest.approx(1 / 15)}
    # E4 (grant) wrote granted. E6 (approval) wrote granted. E13 (inactive) wrote pending, E14 granted,
    # E15 denied (the right resolution with a wrong role id).
    assert fails["closed_wrongly"] == {"n": 5, "n_by_outcome": {"grant": 1, "approval": 1, "inactive": 3},
                                       "table": {"grant": {"granted": 1}, "approval": {"granted": 1},
                                                 "inactive": {"granted": 1, "pending": 1, "denied": 1}}}
    assert fails["correct_resolution"] == RESOLUTION
    # 13 episodes closed the request (8 + 5). Only E4 closed it with another person's id.
    assert fails["wrong_person"] == {"n_closed": 13, "n": 1, "share_of_closed": pytest.approx(1 / 13)}
    # A grant record that the correct end state does not hold: E4 (to another person), E6 (a sensitive
    # role, no approval), E14 (an inactive account). The grants of E1, E2 and E3 are the expected ones.
    assert fails["unexpected_grant"] == {"n": 3, "share": pytest.approx(3 / 15),
                                         "by_outcome": {"grant": 1, "approval": 1, "already": 0,
                                                        "inactive": 1}}
    assert fails["missing"] is None


def test_a_second_copy_of_the_expected_grant_is_not_an_unexpected_grant():
    p, r, _ = ids("ca-000")
    twice = episode("ca-000", 1, 0, passed=False,
                    final=state("granted", p, r, grants=[(p, r), (p, r)], access=[(p, r)]))
    other_role = episode("ca-000", 1, 1, passed=False,
                         final=state("granted", p, r, grants=[(p, r), (p, "x-role")],
                                     access=[(p, r), (p, "x-role")]))
    not_closed = episode("ca-000", 1, 2, passed=False, closed=False)
    not_closed["final_state"] = state(None, None, None, grants=[("x-person", r)], access=[("x-person", r)])
    fails = calibration.failures([twice, other_role, not_closed], [family("ca-000")])
    # `twice` holds only the expected pair; `other_role` gave a second role; `not_closed` gave the role to
    # another person and never closed (the role was given all the same).
    assert fails["unexpected_grant"]["n"] == 2
    # Only the two closed episodes can have closed for a wrong person, and both closed for the right one.
    assert fails["wrong_person"] == {"n_closed": 2, "n": 0, "share_of_closed": 0.0}


def test_wrong_person_looks_at_the_person_id_and_not_at_the_role_id():
    p, r, _ = ids("ca-003")
    wrong_person = episode("ca-003", 1, 0, passed=False, n_turns=3, final=state("denied", "x-person", r))
    wrong_role_1 = episode("ca-003", 1, 1, passed=False, n_turns=3, final=state("denied", p, "x-role"))
    wrong_role_2 = episode("ca-003", 1, 2, passed=False, n_turns=3, final=state("already", p, "y-role"))
    right = episode("ca-003", 1, 3)
    fails = calibration.failures([wrong_person, wrong_role_1, wrong_role_2, right], [family("ca-003")])
    # 4 closed episodes; one names another person; the two with a wrong role name the right person.
    assert fails["wrong_person"] == {"n_closed": 4, "n": 1, "share_of_closed": 0.25}
    assert fails["closed_wrongly"]["table"] == {"inactive": {"already": 1, "denied": 2}}


def test_without_the_family_records_the_report_says_what_is_missing():
    report = calibration.build_report({"small": small_run()},
                                      scenarios_notes={"small": "--scenarios was not given"})
    fails = report["runs"]["small"]["failures"]
    assert fails["missing"] == "--scenarios was not given"
    assert fails["wrong_person"] is None and fails["unexpected_grant"] is None
    assert fails["correct_resolution"] is None
    # The numbers that need no family record are still there.
    assert fails["closed_wrongly"]["n"] == 5 and fails["label_counts"]["never_closed"] == 1
    assert any("Section 8 is incomplete" in w and "--scenarios was not given" in w
               for w in report["runs"]["small"]["warnings"])
    markdown = calibration.to_markdown(report)
    assert "**Missing: the wrong-person count and the unexpected-grant count.**" in markdown
    assert "--scenarios was not given" in markdown
    assert "Closed with the id of another person" not in markdown
    # With the family records nothing is missing.
    full = calibration.to_markdown(calibration.build_report({"small": small_run()},
                                                            families_by_run={"small": small_families()}))
    assert "Missing" not in full
    assert "another person than the one meant: 1 of 13 closed episodes (7.7%)" in full
    assert "the unsafe mistake): 3 episodes (20.0% of all episodes)" in full


def test_family_records_of_another_run_are_refused():
    records = small_run()
    # A family is absent.
    assert "ca-003" in calibration.families_problem(records, small_families()[:3])
    # Another outcome for the same family id.
    swapped = small_families()[:3] + [family("ca-003", "grant")]
    assert "outcome or turn budget" in calibration.families_problem(records, swapped)
    # The same outcome and budget, but another world: the expected ids differ, so the verifier does not
    # give the run's pass for the 4 passes of ca-002 (E9 to E12).
    other_world = small_families()
    other_world[2]["expected"]["default"]["request"]["person_id"] = "someone-else"
    problem = calibration.families_problem(records, other_world)
    assert "do not give the run's pass or fail for 4 episodes" in problem
    fails = calibration.failures(records, other_world)
    assert fails["missing"] == problem and fails["unexpected_grant"] is None
    assert calibration.families_problem(records, small_families()) is None


# ---------------------------------------------------------------------------
# 9. Supply for the supervised stage
# ---------------------------------------------------------------------------
def test_sft_supply():
    supply = small_report()["sft_supply"]
    # Reward 100: E1, E5, E9, E10, E11 = 5 of 15. E2 (95), E3 (88) and E12 (98) passed but are not clean.
    assert supply["pass_reward"] == 100
    assert (supply["n_episodes"], supply["n_clean"]) == (15, 5)
    assert supply["share_clean"] == pytest.approx(1 / 3)
    assert supply["by_outcome"] == {
        "grant": {"n_episodes": 4, "n_clean": 1, "share_clean": 0.25},            # E1
        "approval": {"n_episodes": 4, "n_clean": 1, "share_clean": 0.25},         # E5
        "already": {"n_episodes": 4, "n_clean": 3, "share_clean": 0.75},          # E9, E10, E11
        "inactive": {"n_episodes": 3, "n_clean": 0, "share_clean": 0.0}}
    # Scenarios with a clean pass: ca-000/s1, ca-001/s1, ca-002/s1, ca-002/s4 = 4 of 8.
    assert (supply["n_scenarios_with_clean_pass"], supply["n_scenarios"]) == (4, 8)
    # Families with a clean pass: ca-000, ca-001, ca-002 = 3 of 4.
    assert (supply["n_families_with_clean_pass"], supply["n_families"]) == (3, 4)


# ---------------------------------------------------------------------------
# Empty runs
# ---------------------------------------------------------------------------
def test_an_empty_run_gives_a_clear_error():
    with pytest.raises(ValueError, match=r"run 'nothing': no episode record"):
        calibration.build_report({"nothing": []})
    only_errors = [harness_error_record("ca-000", 1, 0), harness_error_record("ca-000", 1, 1)]
    with pytest.raises(ValueError, match=r"run 'broken': all 2 records are harness errors"):
        calibration.build_report({"broken": only_errors})
    with pytest.raises(ValueError, match="no run was given"):
        calibration.build_report({})
    # One empty run among good ones is an error too: it is not dropped from the report without a word.
    with pytest.raises(ValueError, match=r"run 'nothing'"):
        calibration.build_report({"small": small_run(), "nothing": []})


# ---------------------------------------------------------------------------
# Side by side
# ---------------------------------------------------------------------------
def two_runs() -> dict[str, list[dict]]:
    """Two runs on the same two scenarios.
        family    a            b            b - a
        ca-000    P F -> 0.5   P P -> 1.0   +0.5
        ca-001    F F -> 0.0   P F -> 0.5   +0.5
        pass rate 0.25         0.75         +0.5
    """
    return {"a": four_families({"ca-000": "PF", "ca-001": "FF"}),
            "b": four_families({"ca-000": "PP", "ca-001": "PF"})}


def test_two_runs_on_the_same_scenarios_get_a_paired_difference():
    report = calibration.build_report(two_runs())
    side = report["side_by_side"]
    assert side["runs"] == ["a", "b"] and side["comparable"] is True and side["warnings"] == []
    # The difference is +0.5 in both families, so every draw of families gives +0.5: the interval has no
    # width, and the t-interval (standard deviation 0) has none either.
    assert side["differences"] == [{"a": "a", "b": "b", "n_families": 2, "diff": 0.5, "lo": 0.5, "hi": 0.5,
                                    "t_lo": 0.5, "t_hi": 0.5}]
    assert report["runs"]["a"]["pass"]["pass_rate"] == 0.25
    assert report["runs"]["b"]["pass"]["pass_rate"] == 0.75
    markdown = calibration.to_markdown(report)
    assert "| Number | a | b |" in markdown
    assert "| Pass rate | 25.0% | 75.0% |" in markdown
    assert "Pass rate, b minus a: +50.0 points" in markdown
    assert "NOT COMPARABLE" not in markdown


def test_the_paired_bootstrap_is_reached_through_impl_get(monkeypatch):
    real_get = impl.get
    asked: list[str] = []

    def fake_get(name: str):
        asked.append(name)
        if name == "paired_family_bootstrap":
            return lambda a, b, n_draws, seed, level: {"diff": 0.111, "lo": -0.222, "hi": 0.333}
        return real_get(name)

    monkeypatch.setattr(impl, "get", fake_get)
    report = calibration.build_report(two_runs())
    row = report["side_by_side"]["differences"][0]
    assert (row["diff"], row["lo"], row["hi"]) == (0.111, -0.222, 0.333)
    assert "paired_family_bootstrap" in asked
    # The held-out against seen difference inside one run takes the same road.
    rates = calibration.pass_rates(small_run())
    assert (rates["held_out_minus_seen"]["lo"], rates["held_out_minus_seen"]["hi"]) == (-0.222, 0.333)


def test_runs_over_different_scenarios_stand_side_by_side_with_a_warning():
    runs = {"a": four_families({"ca-000": "PF", "ca-001": "FF"}),
            "b": four_families({"ca-000": "PP", "ca-002": "PF"})}
    report = calibration.build_report(runs)
    side = report["side_by_side"]
    assert side["comparable"] is False
    assert side["differences"] == []                      # no difference between runs that do not match
    # ca-001/s1 is only in a, ca-002/s1 only in b, ca-000/s1 in both.
    assert side["warnings"] == [
        "NOT COMPARABLE: the runs a and b do not cover the same scenarios (1 only in a, 1 only in b, 1 in "
        "both). Their numbers stand side by side, and no difference between them is given."]
    markdown = calibration.to_markdown(report)
    assert "## Side by side" in markdown and "NOT COMPARABLE: the runs a and b" in markdown
    assert "| Number | a | b |" in markdown
    assert "| Pass rate | 25.0% | 75.0% |" in markdown    # a: (0.5 + 0) / 2; b: (1 + 0.5) / 2
    assert "minus" not in markdown.split("## Side by side")[1]


def test_differences_between_comparable_runs_are_named():
    runs = two_runs()
    runs["b"] += [episode("ca-000", 1, 2)]                 # a third sample of one scenario in b only
    for r in runs["b"]:
        r["text_sha256"] = "1" * 64
        r["config_sha256"] = "2" * 64
    side = calibration.build_report(runs)["side_by_side"]
    assert side["comparable"] is True and len(side["differences"]) == 1
    assert [w.split(":")[0] for w in side["warnings"]] == [
        "The runs a and b were made with different text files",
        "The runs a and b were made with different configs",
        "The runs a and b do not have the same numbers of samples per scenario"]


def test_the_warning_on_different_configs_names_the_settings_that_differ():
    # Two runs whose manifests give their configs: only the chunk size differs (as in the calibration runs
    # of the 4B and the 9B). The warning names it, so a reader does not think of the token cap.
    runs = two_runs()
    other = dataclasses.replace(DEFAULT, chunk_size=12)
    for r in runs["b"]:
        r["config_sha256"] = other.sha256()
    manifests = {"a": {"config": DEFAULT.to_dict()}, "b": {"config": other.to_dict()}}
    side = calibration.build_report(runs, manifests=manifests)["side_by_side"]
    assert side["warnings"] == [
        "The runs a and b were made with different configs: chunk_size (32 against 12)."]
    # Without a recorded config for one run, the warning says that it cannot name them.
    side = calibration.build_report(runs, manifests={"a": manifests["a"]})["side_by_side"]
    assert side["warnings"] == [
        "The runs a and b were made with different configs: which settings differ is not known, because a "
        "run has no recorded config."]


def test_by_style_is_the_family_weighted_rate_of_the_style():
    # Style 1: ca-000 has 1 sample (a pass), ca-001 has 16 samples (all fail). The family-weighted rate
    # is (1 + 0) / 2 = 0.5; the share of the style's episodes that passed would be 1/17.
    records = scenario("ca-000", 1, "P") + scenario("ca-001", 1, "F" * 16)
    assert calibration.pass_rates(records, n_draws=50)["by_style"]["1"]["pass_rate"] == 0.5
    assert "family-weighted" in calibration.pass_rates.__doc__


def test_runs_from_different_scenario_files_get_no_difference():
    runs = two_runs()
    for name, data_hash in (("a", "a" * 64), ("b", "b" * 64)):
        for r in runs[name]:
            r["data_sha256"] = data_hash
    report = calibration.build_report(runs)
    assert report["runs"]["a"]["size"]["data_sha256"] == ["a" * 64]
    side = report["side_by_side"]
    assert side["comparable"] is False and side["differences"] == []
    assert side["warnings"] == [
        "NOT COMPARABLE: the runs a and b were made from different scenario files, so a family id does not "
        "name the same family in both. No difference between them is given."]
    # The same file in both: comparable again.
    for r in runs["b"]:
        r["data_sha256"] = "a" * 64
    side = calibration.build_report(runs)["side_by_side"]
    assert side["comparable"] is True and len(side["differences"]) == 1


def test_one_run_has_no_side_by_side_part():
    report = calibration.build_report({"small": small_run()})
    assert report["side_by_side"] == {"runs": ["small"], "comparable": True, "warnings": [],
                                      "differences": []}
    assert "## Side by side" not in calibration.to_markdown(report)


# ---------------------------------------------------------------------------
# The report as a whole: JSON and markdown
# ---------------------------------------------------------------------------
def full_report() -> dict:
    runs = {"small": small_run(), "other": [r for r in small_run() if r["harness_error"] is None]}
    return calibration.build_report(runs, families_by_run={"small": small_families()},
                                    scenarios_notes={"other": "--scenarios was not given"},
                                    sources={"small": "runs/small/episodes.jsonl"})


def test_first_line_says_when_the_placeholder_text_was_used():
    report = full_report()
    assert report["placeholder_text_runs"] == ["small", "other"]
    first_line = calibration.to_markdown(report).split("\n")[0]
    assert first_line.startswith("**PLACEHOLDER TEXT: the run(s) small, other used the placeholder text ")
    # Only the runs that used it are named.
    runs = two_runs()
    for r in runs["a"]:
        r["text_is_placeholder"] = False
    mixed = calibration.build_report(runs)
    assert mixed["placeholder_text_runs"] == ["b"]
    # One record made with the placeholder text is enough for the run to be named.
    for r in runs["b"][1:]:
        r["text_is_placeholder"] = False
    assert calibration.build_report(runs)["placeholder_text_runs"] == ["b"]
    assert calibration.to_markdown(mixed).split("\n")[0].startswith("**PLACEHOLDER TEXT: the run(s) b used")
    # With the candidate's text the first line is the title, and no warning speaks of a placeholder.
    for r in runs["b"]:
        r["text_is_placeholder"] = False
    real = calibration.build_report(runs)
    markdown = calibration.to_markdown(real)
    assert markdown.split("\n")[0] == "# Calibration report"
    assert real["placeholder_text_runs"] == [] and "laceholder text was used" not in markdown


def test_report_is_plain_json_and_repeats_exactly():
    report = full_report()
    as_text = json.dumps(report, indent=2, ensure_ascii=False)
    assert json.loads(as_text) == report                  # no tuple, no key that is not a string
    assert json.dumps(full_report(), indent=2, ensure_ascii=False) == as_text
    assert calibration.to_markdown(full_report()) == calibration.to_markdown(report)
    # The order of the records does not change a number.
    reversed_run = calibration.run_report("small", small_run()[::-1], families=small_families()[::-1])
    assert reversed_run == small_report()


NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Numbers that are fixed words of a definition, not results: "at least 2 valid samples", "100% minus".
WORDS = {"2", "100"}


def numbers_in_json(node: object) -> set[str]:
    """Every way in which the markdown may print a number of the report, and every number inside a string."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found |= set(NUMBER.findall(key)) | numbers_in_json(value)
    elif isinstance(node, list):
        for value in node:
            found |= numbers_in_json(value)
    elif isinstance(node, str):
        found |= set(NUMBER.findall(node))
    elif isinstance(node, bool) or node is None:
        pass
    elif isinstance(node, int):
        found.add(str(abs(node)))
    else:
        value = abs(node)
        found |= {f"{100 * value:.1f}", f"{100 * value:g}", f"{value:.2f}", f"{value:.1f}"}
    return found


def numbers_in_markdown(markdown: str) -> set[str]:
    text = re.sub(r"(?m)^#+ \d+\. ", "", markdown)        # the section numbers of the headings
    text = re.sub(r"PLAN \d\.\d|SHA-256", "", text)       # references and names that hold a digit
    return set(NUMBER.findall(text))


def test_every_number_of_the_markdown_is_in_the_json():
    report = full_report()
    markdown = calibration.to_markdown(report)
    shown = numbers_in_markdown(markdown)
    assert len(shown) > 60                                # the markdown is not empty of numbers
    assert shown - numbers_in_json(report) - WORDS == set()
    # The check can fail: without a section of the JSON, numbers of the markdown have no source. (Section 9
    # is not in this list: each of its few numbers of the small run also occurs in another section.)
    for section in ("pass", "ties", "call_format", "turns", "penalties", "failures"):
        cut = copy.deepcopy(report)
        for run in cut["runs"].values():
            del run[section]
        assert shown - numbers_in_json(cut) - WORDS != set(), section


def test_markdown_of_the_small_run_puts_each_number_in_its_place():
    """The lines of the markdown that carry numbers, typed by hand from the values that the tests of the
    nine sections work out (the same numbers, in percent with one decimal)."""
    report = calibration.build_report({"small": small_run()}, families_by_run={"small": small_families()})
    lines = calibration.to_markdown(report).split("\n")
    expected = [
        # 1: 16 records, 1 harness error = 1/16 = 6.25%, printed 6.2%.
        "- Episode records: 16. Harness errors: 1 (6.2% of the records); they are left out of every number "
        "below. Episodes counted: 15.",
        "- Families: 4 (grant 1, approval 1, already 1, inactive 1). Scenarios: 8.",
        "- Valid samples per scenario: NOT the same in every scenario: 1 sample in 1 scenario, 2 samples in "
        "7 scenarios.",
        # 2: pass rate 0.5, interval 0.125 to 0.875.
        "- Over all: 50.0% (95% interval 12.5% to 87.5%, from resampling the 4 families).",
        "| outcome grant | 75.0% | 1 |", "| outcome approval | 25.0% | 1 |",
        "| outcome already | 100.0% | 1 |", "| outcome inactive | 0.0% | 1 |",
        "| style 1 | 62.5% | 4 |", "| style 4 | 37.5% | 4 |",
        "| seen styles (1 to 3) | 62.5% | 4 |", "| held-out style (4) | 37.5% | 4 |",
        "Held-out minus seen: -25.0 points (95% interval -50.0 to +0.0, paired bootstrap over 4 families).",
        # 3.
        "- (a) Floor. Is the lower end of the 95% interval above 25.0%? **NO**: the lower end is 12.5%. "
        "(The best policy that always gives the same resolution passes 25.0% of this run's families.)",
        "- (b) Band. Is the pass rate between 30.0% and 65.0%? **YES**: the pass rate is 50.0%, inside the "
        "band.",
        "- Room left (100% minus the pass rate): 50.0% over all; grant 25.0%, approval 75.0%, already 0.0%, "
        "inactive 100.0%.",
        # 4: 3/7 = 42.9%, 1/7 = 14.3%, 2/7 = 28.6%, 5/7 = 71.4%.
        "| all | 7 | 42.9% | 14.3% | 28.6% | 71.4% |",
        "| outcome grant | 2 | 0.0% | 0.0% | 0.0% | 50.0% |",
        "| outcome approval | 2 | 50.0% | 0.0% | 50.0% | 50.0% |",
        "| outcome already | 2 | 50.0% | 50.0% | 0.0% | 100.0% |",
        "| outcome inactive | 1 | 100.0% | 0.0% | 100.0% | 100.0% |",
        "- Family profile: of 4 families, 1 always pass, 1 always fail, 2 are mixed.",
        # 5: 11/68 = 16.2%, 1/68 = 1.5%.
        "- Invalid-call rate: 16.2% (11 of 68 assistant messages were malformed).",
        "- Parse errors (messages): no_call 9, unknown_argument 1, truncated 1.",
        "- Tool errors (well-formed calls that the tool refused; 2 in all): unknown_person_id 2.",
        "- Turns cut off by the token cap: 1.5% (1 of 68 messages).",
        # 6: 68/15 = 4.53, 11/3 = 3.67, 2/15 = 13.3%, 4/15 = 26.7%.
        "- Mean turns per episode: 4.53 (grant 4.75, approval 6.00, already 3.50, inactive 3.67).",
        "- Episodes that used up the turn limit without closing: 13.3%.",
        "- Episodes over their turn budget: 26.7% (4 of 15).",
        "- New tokens per well-formed call (57 calls, end-of-turn token included): median 40.0, 99th "
        "percentile 70, largest 70.",
        # 260: without a run manifest the cap in force is the token cap of accessdesk/config.py. (The cut-off
        # message of E3 has 160 tokens, the cap when these records were typed. The report counts the
        # messages that are marked as cut off; it does not compare their length with the cap.)
        "- Token cap by the plan's rule (2 times the 99th percentile, at least 128): **140**. Cap in force "
        "in this run: 260. Messages cut off at the cap in force: 1; their lengths are not in these numbers.",
        # 7: 11/24 = 45.8%, 2/3 = 66.7%, 13/24 = 54.2%.
        "| at least one malformed call | all episodes | 3 | 2 | 2 | 50.0% |",
        "| no malformed call | all episodes | 12 | 6 | 4 | 45.8% |",
        "| at least one malformed call | closed the request | 2 | 2 | 1 | 100.0% |",
        "| no malformed call | closed the request | 11 | 6 | 4 | 50.0% |",
        "| over the turn budget | all episodes | 4 | 2 | 3 | 66.7% |",
        "| inside the turn budget | all episodes | 11 | 6 | 4 | 54.2% |",
        "| over the turn budget | closed the request | 2 | 2 | 2 | 100.0% |",
        "| inside the turn budget | closed the request | 11 | 6 | 4 | 54.2% |",
        "- Episodes that never closed the request: 2 (passed: 0); of these, with a malformed call: 1; over "
        "the turn budget: 2.",
        "- Episodes in both groups (a malformed call and over the budget): 2.",
        "- Rewards of the 8 passing episodes: reward 100: 5, reward 98: 1, reward 95: 1, reward 88: 1.",
        # 8: 8/15 = 53.3%, 5/15 = 33.3%, 1/15 = 6.7%, 1/13 = 7.7%, 3/15 = 20.0%.
        "- Every episode has one label: closed_correctly 53.3% (8), closed_wrongly 33.3% (5), never_closed "
        "6.7% (1), malformed_only 6.7% (1).",
        "| Expected outcome | Correct resolution | Episodes | wrote granted | wrote pending | wrote already "
        "| wrote denied |",
        "| grant | granted | 1 | 1 | 0 | 0 | 0 |",
        "| approval | pending | 1 | 1 | 0 | 0 | 0 |",
        "| inactive | denied | 3 | 1 | 1 | 0 | 1 |",
        "- Closed with the id of another person than the one meant: 1 of 13 closed episodes (7.7%).",
        "- A role was given that should not have been given (the unsafe mistake): 3 episodes (20.0% of all "
        "episodes); by expected outcome: grant 1, approval 1, already 0, inactive 1.",
        # 9: 5/15 = 33.3%.
        "- Clean passes (reward 100): 33.3% of the episodes (5 of 15); grant 25.0% (1 of 4), approval 25.0% "
        "(1 of 4), already 75.0% (3 of 4), inactive 0.0% (0 of 3).",
        "- Scenarios with at least one clean pass: 4 of 8. Families with at least one: 3 of 4.",
    ]
    for line in expected:
        assert line in lines, line
    # The two sentences of section 7 are shown as they are in the JSON.
    pen = report["runs"]["small"]["penalties"]
    assert f"- Malformed-call penalty: {pen['malformed']['sentence']}" in lines
    assert f"- Turn penalty: {pen['turn_budget']['sentence']}" in lines
    # So are the caution and the sentence on the episodes that closed the request.
    assert f"- Malformed-call penalty, caution: {pen['malformed']['caution']}" in lines
    assert f"- Malformed-call penalty: {pen['malformed']['closed_only']['sentence']}" in lines
    assert f"- Turn penalty, caution: {pen['turn_budget']['caution']}" in lines
    assert f"- Turn penalty: {pen['turn_budget']['closed_only']['sentence']}" in lines
    # The row total of the closed-wrongly table is a number of the report, not a sum made for the page.
    wrong = report["runs"]["small"]["failures"]["closed_wrongly"]
    assert wrong["n_by_outcome"] == {outcome: sum(row.values()) for outcome, row in wrong["table"].items()}
    assert sum(wrong["n_by_outcome"].values()) == wrong["n"]


def test_side_by_side_table_puts_each_run_in_its_own_column():
    # Run a: the small run (numbers above). Run b: 2 families, one scenario each, samples P F and P P.
    # b: pass rate (0.5 + 1) / 2 = 75%; scenarios with the same reward 1 of 2; the F never closed: 8 turns,
    # so mean turns (4 + 8 + 4 + 4) / 4 = 5.00 and 1 of 4 episodes is over the budget and at the turn limit;
    # no malformed message; clean passes 3 of 4; no token counts.
    runs = {"a": small_run(), "b": four_families({"ca-000": "PF", "ca-001": "PP"})}
    report = calibration.build_report(runs, families_by_run={"a": small_families()})
    lines = calibration.to_markdown(report).split("## Side by side")[1].split("\n")
    for line in [
        "| Number | a | b |",
        "| Episodes counted | 15 | 4 |", "| Harness errors | 1 | 0 |", "| Families | 4 | 2 |",
        "| Scenarios | 8 | 2 |", "| Samples per scenario | 1 to 2 (unequal) | 2 |",
        # b: a draw of its 2 family rates (0.5 and 1) has mean 0.5 or 1 in a quarter of the draws each.
        "| Pass rate | 50.0% | 75.0% |", "| 95% interval | 12.5% to 87.5% | 50.0% to 100.0% |",
        "| Pass rate, grant | 75.0% | 50.0% |",
        "| Pass rate, approval | 25.0% | 100.0% |", "| Pass rate, already | 100.0% | n/a |",
        "| Pass rate, seen styles | 62.5% | 75.0% |", "| Pass rate, held-out style | 37.5% | n/a |",
        "| Floor check (interval above the floor) | NO | YES |",
        "| Band check (pass rate in the band) | YES (inside) | NO (above) |",
        "| Room left | 50.0% | 25.0% |",
        "| Scenarios with the same reward | 42.9% | 50.0% |",
        "| Scenarios with the same pass or fail | 71.4% | 50.0% |",
        "| Families always pass / always fail / mixed | 1 / 1 / 2 | 1 / 0 / 1 |",
        "| Invalid-call rate | 16.2% | 0.0% |", "| Turns cut off by the token cap | 1.5% | 0.0% |",
        "| Mean turns per episode | 4.53 | 5.00 |", "| Episodes at the turn limit | 13.3% | 25.0% |",
        "| Episodes over the turn budget | 26.7% | 25.0% |",
        "| Tokens per well-formed call, 99th percentile | 70 | n/a |",
        "| Token cap by the plan's rule | 140 | n/a |",
        "| Clean passes, share of episodes | 33.3% | 75.0% |", "| Scenarios with a clean pass | 4 | 2 |",
        "| Episodes closed for the wrong person | 1 | n/a |",
        "| Episodes with a role given wrongly | 3 | n/a |",
    ]:
        assert line in lines, line


def test_markdown_holds_all_nine_sections_for_each_run_then_side_by_side():
    markdown = calibration.to_markdown(full_report())
    headings = [line for line in markdown.split("\n") if line.startswith("#")]
    sections = ["### 1. Size and health", "### 2. Pass rate", "### 3. The two checks of the plan",
                "### 4. Ties", "### 5. Call format", "### 6. Turns and tokens", "### 7. Penalties",
                "### 8. What the model does when it fails", "### 9. Supply for the supervised stage"]
    assert headings == (["# Calibration report", "## Run small"] + sections + ["## Run other"] + sections
                        + ["## Side by side"])
    assert "- Records file: runs/small/episodes.jsonl" in markdown


# ---------------------------------------------------------------------------
# scripts/calibration_report.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def script():
    path = ROOT / "scripts" / "calibration_report.py"
    spec = importlib.util.spec_from_file_location("calibration_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_run(folder: Path, records: list[dict], manifest: dict | None = None) -> Path:
    folder.mkdir(parents=True)
    write_records(folder / "episodes.jsonl", records)
    if manifest is not None:
        (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder / "episodes.jsonl"


def save_scenarios(folder: Path, families: list[dict]) -> str:
    """Write calib.jsonl; return its SHA-256."""
    folder.mkdir(parents=True)
    data = "".join(json.dumps(f) + "\n" for f in families).encode("utf-8")
    (folder / "calib.jsonl").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_script_writes_json_and_markdown_and_prints_the_markdown(script, tmp_path, capsys):
    file_hash = save_scenarios(tmp_path / "scenarios", small_families())
    manifest = {"complete": True, "model": "some/model", "revision": "abc", "adapter": None,
                "text_file": "t.yaml", "text_sha256": "0" * 64, "text_is_placeholder": True,
                "config": DEFAULT.to_dict(), "split": "calib", "split_file_sha256": file_hash}
    path_a = save_run(tmp_path / "a", small_run(), manifest)
    path_b = save_run(tmp_path / "b", [r for r in small_run() if r["harness_error"] is None])
    out = tmp_path / "out" / "report"
    code = script.main(["--run", f"first={path_a}", "--run", f"second={path_b}",
                        "--scenarios", str(tmp_path / "scenarios"), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "report.md"]
    markdown = (out / "report.md").read_text(encoding="utf-8")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert printed == markdown
    assert markdown == calibration.to_markdown(report)    # the markdown needs nothing but the JSON
    assert markdown.split("\n")[0].startswith("**PLACEHOLDER TEXT: the run(s) first, second")
    assert list(report["runs"]) == ["first", "second"]
    first, second = report["runs"]["first"], report["runs"]["second"]
    assert first["size"]["model"] == "some/model" and second["size"]["model"] is None
    assert first["source"] == str(path_a)
    assert (first["size"]["n_records"], first["size"]["n_harness_errors"]) == (16, 1)
    assert (second["size"]["n_records"], second["size"]["n_harness_errors"]) == (15, 0)
    assert first["pass"]["pass_rate"] == pytest.approx(0.5)
    # Both runs found their families in --scenarios: E4 is the one episode closed for a wrong person.
    assert first["failures"]["wrong_person"]["n"] == 1 and second["failures"]["wrong_person"]["n"] == 1
    assert first["failures"]["unexpected_grant"]["n"] == 3
    assert "- Model: some/model (revision abc, adapter none)" in markdown
    assert "- Model: not recorded (no run_manifest.json beside the records)" in markdown


def test_script_without_scenarios_says_what_is_missing(script, tmp_path, capsys):
    path = save_run(tmp_path / "a", small_run())
    assert script.main(["--run", f"only={path}"]) == 0     # no --out: the markdown is printed only
    printed = capsys.readouterr().out
    assert "**Missing: the wrong-person count and the unexpected-grant count.**" in printed
    assert "--scenarios was not given" in printed
    assert list(tmp_path.iterdir()) == [tmp_path / "a"]


def test_script_refuses_scenario_files_that_the_run_was_not_made_from(script, tmp_path, capsys):
    save_scenarios(tmp_path / "scenarios", small_families())
    manifest = {"config": DEFAULT.to_dict(), "split": "calib", "split_file_sha256": "f" * 64}
    path = save_run(tmp_path / "a", small_run(), manifest)
    out = tmp_path / "out"
    assert script.main(["--run", f"a={path}", "--scenarios", str(tmp_path / "scenarios"),
                        "--out", str(out)]) == 0
    capsys.readouterr()
    fails = json.loads((out / "report.json").read_text(encoding="utf-8"))["runs"]["a"]["failures"]
    assert "is not the file that the run was made from" in fails["missing"]
    assert fails["wrong_person"] is None and fails["unexpected_grant"] is None
    # The records themselves can name the scenario file (`data_sha256`); another file is refused as well.
    named = small_run()
    for r in named:
        r["data_sha256"] = "e" * 64
    path_named = save_run(tmp_path / "n", named)
    assert script.main(["--run", f"n={path_named}", "--scenarios", str(tmp_path / "scenarios")]) == 0
    assert "is not the file that the run was made from" in capsys.readouterr().out
    # A scenario folder without the run's split file is said too.
    on_dev = scenario("dv-000", 1, "PF", "grant") + scenario("dv-001", 1, "PF", "grant")
    path_dev = save_run(tmp_path / "d", on_dev)
    assert script.main(["--run", f"d={path_dev}", "--scenarios", str(tmp_path / "scenarios")]) == 0
    assert "dev.jsonl does not exist" in capsys.readouterr().out


def test_script_gives_a_clear_error_for_an_empty_or_missing_run(script, tmp_path, capsys):
    empty = save_run(tmp_path / "empty", [])
    out = tmp_path / "out"
    assert script.main(["--run", f"nothing={empty}", "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert "run 'nothing': no episode record" in captured.err and captured.out == ""
    assert not out.exists()                               # no half-written report
    assert script.main(["--run", f"gone={tmp_path / 'no-such-file.jsonl'}"]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert script.main(["--run", "no-equals-sign"]) == 2
    assert "NAME=PATH" in capsys.readouterr().err
    good = save_run(tmp_path / "good", small_run())
    assert script.main(["--run", f"x={good}", "--run", f"x={good}"]) == 2
    assert "given twice" in capsys.readouterr().err


def test_the_report_shows_a_records_path_the_same_way_however_it_was_typed(script, tmp_path, monkeypatch):
    """The report wrote each records path as it was typed, so the same records gave other report bytes
    with an absolute path, or from another working folder."""
    inside = ROOT / "results" / "calibration" / "round1" / "4b" / "episodes.jsonl"
    assert script.shown_path(inside) == "results/calibration/round1/4b/episodes.jsonl"
    monkeypatch.chdir(ROOT / "results")
    assert script.shown_path(Path("calibration/round1/4b/episodes.jsonl")) == \
        "results/calibration/round1/4b/episodes.jsonl"
    monkeypatch.chdir(ROOT)
    assert script.shown_path(Path("results/calibration/round1/4b/episodes.jsonl")) == \
        "results/calibration/round1/4b/episodes.jsonl"
    # A file outside the repository is shown as it was typed.
    assert script.shown_path(tmp_path / "episodes.jsonl") == str(tmp_path / "episodes.jsonl")
    # And the report of a run carries that form.
    monkeypatch.chdir(tmp_path)
    path = save_run(tmp_path / "a", small_run())
    args = script.parse_args(["--run", f"a={path.relative_to(tmp_path)}"])
    assert script.read_runs(args)["a"]["source"] == str(Path("a") / "episodes.jsonl")
