"""Tests of accessdesk/eval/metrics.py on hand-made episode records.

Every expected number is worked out by hand in a comment beside the records. Where a wrong unit of analysis
would give another number (records instead of families, episodes instead of messages), the test states that
other number too, so that the two cannot be confused.

In `scenario_records(family, style, rewards)` a reward of 100 is a clean pass, 95 a pass with one malformed
message, and 0 a fail. A reward below 0 (-5, -10, -15) is a fail with malformed messages under a config whose
switch for penalties on failing episodes is on.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from accessdesk.eval import metrics
from tests.helpers.make_records import make_record, scenario_records


# ---------------------------------------------------------------------------
# The record helper itself
# ---------------------------------------------------------------------------
def test_hand_made_record_has_the_keys_of_the_record_format_in_order():
    record = make_record("dv-007", 2, 3, passed=True, outcome="approval")
    assert list(record) == [
        "schema", "run", "arm", "split", "family_id", "style", "scenario_id", "outcome", "ruleset", "toolset",
        "prompt", "sample", "seed", "turns", "n_turns", "n_malformed", "n_tool_errors", "closed", "ended_by",
        "final_state", "passed", "reward", "turn_budget", "impl", "text_sha256", "text_is_placeholder",
        "config_sha256", "settings", "harness_error"]
    assert list(record["turns"][0]) == ["turn", "assistant_text", "truncated", "n_tokens", "parse_ok",
                                        "parse_error", "tool_key", "args", "tool_error", "tool_response"]
    assert (record["split"], record["scenario_id"], record["sample"]) == ("dev", "dv-007/s2", 3)
    assert (record["n_turns"], record["turn_budget"], record["reward"]) == (4, 5, 100)
    assert record["ended_by"] == "close"
    assert json.loads(json.dumps(record)) == record


def test_hand_made_record_counters_agree_with_its_turns():
    record = make_record(passed=True, reward=88, n_malformed=2)
    assert record["n_turns"] == len(record["turns"]) == 6
    assert sum(1 for turn in record["turns"] if not turn["parse_ok"]) == record["n_malformed"] == 2
    assert record["turns"][-1]["tool_key"] == "close_request"
    never_closed = make_record(passed=False, closed=False, n_malformed=3)
    assert (never_closed["n_turns"], never_closed["ended_by"]) == (8, "turn_limit")
    assert never_closed["closed"] is False
    assert never_closed["final_state"]["request"]["resolution"] is None


def test_hand_made_record_refuses_impossible_combinations():
    with pytest.raises(ValueError):
        make_record(passed=True, closed=False)              # a pass needs a close
    with pytest.raises(ValueError):
        make_record(passed=True, reward=0)                  # a pass scores above 0
    with pytest.raises(ValueError):
        make_record(passed=False, reward=95)                # a fail scores 0 or less
    with pytest.raises(ValueError):
        make_record(passed=True, reward=-5)                 # a pass never scores below 0
    with pytest.raises(ValueError):
        make_record(passed=False, closed=False, n_turns=5)  # not closed means the turn limit was used up
    with pytest.raises(ValueError):
        scenario_records("dv-000", 1, [100, 90])            # 90 is not a reward the default config can give


# ---------------------------------------------------------------------------
# Pass rates: the family is the unit
# ---------------------------------------------------------------------------
def test_family_weighted_pass_rate_differs_from_record_weighted():
    # Family dv-000 has 2 records, both pass: family pass rate 1.0.
    # Family dv-001 has 6 records, all fail: family pass rate 0.0.
    records = scenario_records("dv-000", 1, [100, 100]) + scenario_records("dv-001", 1, [0, 0, 0, 0, 0, 0])
    assert metrics.family_pass_rates(records) == {"dv-000": 1.0, "dv-001": 0.0}
    # By family: (1.0 + 0.0) / 2 = 0.5. By record it would be 2 / 8 = 0.25.
    assert metrics.pass_rate(records) == 0.5
    record_weighted = sum(r["passed"] for r in records) / len(records)
    assert record_weighted == 0.25


def test_family_pass_rate_pools_all_wordings_and_samples_of_the_family():
    # dv-000: style 1 has 1 pass of 1, style 2 has 1 pass of 3. All records of the family: 2 of 4 = 0.5.
    # (The mean of the two style rates would be (1 + 1/3) / 2 = 0.667.)
    records = scenario_records("dv-000", 1, [100]) + scenario_records("dv-000", 2, [100, 0, 0])
    assert metrics.family_pass_rates(records) == {"dv-000": 0.5}


def test_a_penalised_pass_counts_as_a_pass():
    # Rewards 95 and 88 are passes with malformed messages: pass rate 2 of 3.
    records = scenario_records("dv-000", 1, [95, 88, 0])
    assert metrics.pass_rate(records) == pytest.approx(2 / 3)


def test_harness_errors_are_left_out_and_counted():
    # dv-000: pass, fail, and one record with a harness error that is marked as passed.
    # Without the harness error: 1 of 2 = 0.5. With it counted as a pass it would be 2 of 3.
    records = scenario_records("dv-000", 1, [100, 0])
    records.append(make_record("dv-000", 1, 2, passed=True, harness_error="made-up error"))
    assert metrics.harness_error_count(records) == 1
    assert len(metrics.valid_records(records)) == 2
    assert metrics.pass_rate(records) == 0.5
    assert metrics.reward_distribution(records) == {0: 1, 100: 1}
    # A family with harness errors only is in no number.
    only_errors = [make_record("dv-009", 1, 0, harness_error="made-up error")]
    assert metrics.family_pass_rates(records + only_errors) == {"dv-000": 0.5}


def test_no_valid_record_is_an_error_and_not_a_zero():
    with pytest.raises(ValueError):
        metrics.pass_rate([])
    with pytest.raises(ValueError):
        metrics.pass_rate([make_record(harness_error="made-up error")])


def test_results_do_not_depend_on_the_order_of_the_records():
    records = (scenario_records("dv-001", 2, [100, 0, 95]) + scenario_records("dv-000", 1, [0, 100])
               + scenario_records("dv-001", 1, [0, 0, 100]))
    backwards = list(reversed(records))
    for function in (metrics.family_pass_rates, metrics.reward_distribution, metrics.zero_variance_share,
                     metrics.turn_stats, metrics.outcome_labels, metrics.variance_shares):
        assert json.dumps(function(records)) == json.dumps(function(backwards))
    assert metrics.pass_at_k(records, 2) == metrics.pass_at_k(backwards, 2)
    assert metrics.two_halves_mde(records, 5) == metrics.two_halves_mde(backwards, 5)
    assert list(metrics.family_pass_rates(records)) == ["dv-000", "dv-001"]


def test_pass_rate_by_outcome_is_family_weighted_inside_each_outcome():
    records = (scenario_records("dv-000", 1, [100, 100], outcome="grant")            # family rate 1.0
               + scenario_records("dv-001", 1, [100, 0, 0, 0], outcome="grant")      # family rate 0.25
               + scenario_records("dv-002", 1, [0, 0], outcome="approval")           # family rate 0.0
               + scenario_records("dv-003", 1, [100], outcome="inactive"))           # family rate 1.0
    # grant: (1.0 + 0.25) / 2 = 0.625 (by record it would be 3 / 6 = 0.5). approval: 0.0. inactive: 1.0.
    by_outcome = metrics.pass_rate_by(records, "outcome")
    assert by_outcome == {"grant": 0.625, "approval": 0.0, "inactive": 1.0}
    assert list(by_outcome) == ["grant", "approval", "inactive"]       # the order of types.OUTCOMES


def test_pass_rate_by_style_and_seen_against_held_out():
    records = (scenario_records("dv-000", 1, [100, 100]) + scenario_records("dv-000", 2, [100, 0])
               + scenario_records("dv-000", 3, [0, 0]) + scenario_records("dv-000", 4, [0, 100])
               + scenario_records("dv-001", 1, [100]) + scenario_records("dv-001", 4, [0, 0, 0, 0]))
    # Style 1: dv-000 1.0, dv-001 1.0 -> 1.0.   Style 2: dv-000 only, 0.5.   Style 3: dv-000 only, 0.0.
    # Style 4: dv-000 0.5, dv-001 0.0 -> 0.25 (by record it would be 1 / 6).
    assert metrics.pass_rate_by(records, "style") == {1: 1.0, 2: 0.5, 3: 0.0, 4: 0.25}
    # Seen styles (1 to 3): dv-000 has 3 passes in 6 records = 0.5; dv-001 has 1 of 1 = 1.0 -> 0.75.
    # Held-out style (4): 0.25, as above.
    assert metrics.pass_rate_by(records, "seen_held_out") == {"seen": 0.75, "held_out": 0.25}


def test_pass_rate_by_refuses_an_unknown_key():
    with pytest.raises(ValueError):
        metrics.pass_rate_by(scenario_records("dv-000", 1, [100]), "family_id")


def test_reward_distribution_counts_episodes():
    records = scenario_records("dv-000", 1, [100, 95, 0]) + scenario_records("dv-001", 1, [0, 100, 0])
    distribution = metrics.reward_distribution(records)
    assert distribution == {0: 3, 95: 1, 100: 2}
    assert list(distribution) == [0, 95, 100]


def test_a_fail_below_zero_is_a_fail_in_every_number():
    # With the switch for penalties on failing episodes on, a fail scores 0, -5, -10 or -15.
    # dv-000: 100, -5, -15, 0 -> 1 pass of 4. dv-001: -5, -5 -> 0 of 2.
    records = scenario_records("dv-000", 1, [100, -5, -15, 0]) + scenario_records("dv-001", 1, [-5, -5])
    assert [r["passed"] for r in records] == [True, False, False, False, False, False]
    assert [r["n_malformed"] for r in records] == [0, 1, 3, 0, 1, 1]
    assert metrics.family_pass_rates(records) == {"dv-000": 0.25, "dv-001": 0.0}
    assert metrics.pass_rate(records) == 0.125                      # (0.25 + 0) / 2
    distribution = metrics.reward_distribution(records)
    assert distribution == {-15: 1, -5: 3, 0: 1, 100: 1}
    assert list(distribution) == [-15, -5, 0, 100]                  # in rising order, the lowest first
    # Malformed messages: 1 + 3 + 1 + 1 = 6. All messages (4 for the shortest path, plus the malformed
    # ones): dv-000 4, 5, 7, 4 and dv-001 5, 5 -> 30.
    assert metrics.invalid_call_rate(records) == pytest.approx(6 / 30)
    labels = metrics.outcome_labels(records)
    assert labels["closed_correctly"] == pytest.approx(1 / 6)
    assert labels["closed_wrongly"] == pytest.approx(5 / 6)


# ---------------------------------------------------------------------------
# Zero-variance share: exact reward equality against pass-or-fail only
# ---------------------------------------------------------------------------
def test_zero_variance_share_exact_equality_against_pass_or_fail_only():
    records = (
        scenario_records("dv-000", 1, [100, 95, 100])    # rewards differ; all pass
        + scenario_records("dv-000", 2, [100, 100])      # equal rewards; all pass
        + scenario_records("dv-001", 1, [0, 0, 0])       # equal rewards; all fail
        + scenario_records("dv-001", 2, [100, 0])        # a pass and a fail
        + scenario_records("dv-001", 3, [0, 0])          # equal rewards; all fail
        + scenario_records("dv-002", 1, [100])           # one sample only: not counted
    )
    # Five scenarios have at least 2 samples.
    # Equal rewards: dv-000/s2, dv-001/s1 and dv-001/s3 -> 3 / 5. Of these, one is all-pass (1 / 5) and two
    # are all-fail (2 / 5).
    # Equal for pass-or-fail: dv-000/s1 too (100, 95, 100 are all passes) -> 4 / 5.
    assert metrics.zero_variance_share(records) == {
        "n_scenarios": 5, "equal_reward": 0.6, "equal_reward_all_pass": 0.2, "equal_reward_all_fail": 0.4,
        "equal_passfail": 0.8}


def test_zero_variance_share_of_one_scenario_with_a_penalised_pass():
    # The single case of the specification: rewards 100, 95, 100, 100.
    # Not equal as rewards (share 0), equal for pass-or-fail (share 1).
    share = metrics.zero_variance_share(scenario_records("tr-000", 1, [100, 95, 100, 100]))
    assert (share["n_scenarios"], share["equal_reward"], share["equal_passfail"]) == (1, 0.0, 1.0)


def test_zero_variance_share_is_by_scenario_and_not_by_family():
    # One family, two scenarios: style 1 all pass, style 2 all fail. The family has passes and fails, but
    # each group that GRPO would see is a tie: both shares are 1.
    records = scenario_records("tr-000", 1, [100, 100]) + scenario_records("tr-000", 2, [0, 0])
    share = metrics.zero_variance_share(records)
    assert (share["n_scenarios"], share["equal_reward"], share["equal_passfail"]) == (2, 1.0, 1.0)
    assert (share["equal_reward_all_pass"], share["equal_reward_all_fail"]) == (0.5, 0.5)


def test_zero_variance_share_needs_two_valid_samples():
    # A scenario whose second sample is a harness error has one valid sample: it is not counted.
    records = [make_record("dv-000", 1, 0), make_record("dv-000", 1, 1, harness_error="made-up error")]
    assert metrics.zero_variance_share(records) == {
        "n_scenarios": 0, "equal_reward": None, "equal_reward_all_pass": None, "equal_reward_all_fail": None,
        "equal_passfail": None}


def test_zero_variance_share_with_fails_below_zero():
    records = (
        scenario_records("dv-000", 1, [0, -5, 0])        # all fail, but the rewards differ
        + scenario_records("dv-000", 2, [-5, -5])        # all fail with equal rewards
        + scenario_records("dv-001", 1, [0, 0])          # all fail with equal rewards
        + scenario_records("dv-001", 2, [100, -15])      # a pass and a fail
    )
    # Four scenarios. Equal rewards: dv-000/s2 and dv-001/s1 -> 2 / 4, both all-fail.
    # Equal for pass-or-fail: dv-000/s1 too (0, -5, 0 are all fails) -> 3 / 4.
    # So with the switch on, a group in which every sample failed can still carry a signal.
    assert metrics.zero_variance_share(records) == {
        "n_scenarios": 4, "equal_reward": 0.5, "equal_reward_all_pass": 0.0, "equal_reward_all_fail": 0.5,
        "equal_passfail": 0.75}


def test_zero_variance_share_refuses_equal_rewards_for_a_pass_and_a_fail():
    # A pass with reward 0 cannot come from the reward function. If records ever say so, stop.
    passed = make_record("dv-000", 1, 0, passed=True)
    passed["reward"] = 0
    with pytest.raises(ValueError):
        metrics.zero_variance_share([passed, make_record("dv-000", 1, 1, passed=False)])


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------
def test_invalid_call_rate_is_per_message_and_not_per_episode():
    records = [
        make_record("dv-000", 1, 0, passed=True, reward=95, n_malformed=1),       # 5 messages, 1 malformed
        make_record("dv-000", 1, 1, passed=False, closed=False, n_malformed=8),   # 8 messages, 8 malformed
        make_record("dv-000", 1, 2, passed=True, outcome="already"),              # 3 messages, 0 malformed
    ]
    # Messages: 5 + 8 + 3 = 16. Malformed: 1 + 8 + 0 = 9. Rate 9 / 16 = 0.5625.
    # (The mean of the three per-episode rates would be (0.2 + 1 + 0) / 3 = 0.4.)
    assert [r["n_turns"] for r in records] == [5, 8, 3]
    assert metrics.invalid_call_rate(records) == 0.5625


def test_invalid_call_rate_leaves_out_harness_errors():
    # One clean episode (4 messages, 0 malformed) and one harness error with 8 malformed messages.
    records = [make_record("dv-000", 1, 0),
               make_record("dv-000", 1, 1, passed=False, closed=False, n_malformed=8,
                           harness_error="made-up error")]
    assert metrics.invalid_call_rate(records) == 0.0


def test_turn_stats():
    records = [
        make_record("dv-000", 1, 0, passed=True),                                  # 4 turns, closed
        make_record("dv-000", 1, 1, passed=False, closed=False),                   # 8 turns, turn limit
        make_record("dv-000", 1, 2, passed=False, closed=True, n_turns=8),         # 8 turns, closed on turn 8
        make_record("dv-000", 1, 3, passed=True, outcome="inactive"),              # 3 turns, closed
    ]
    # Mean turns: (4 + 8 + 8 + 3) / 4 = 5.75. Ended by the turn limit: 1 of 4. Closed: 3 of 4.
    # The third episode used 8 turns but closed: it is not "at the turn limit".
    assert metrics.turn_stats(records) == {"n_episodes": 4, "mean_turns": 5.75, "share_at_turn_limit": 0.25,
                                           "share_closed": 0.75}


def test_outcome_labels_shares():
    records = [
        make_record("dv-000", 1, 0, passed=True),                                  # closed_correctly
        make_record("dv-000", 1, 1, passed=True, reward=95, n_malformed=1),        # closed_correctly
        make_record("dv-000", 1, 2, passed=False, closed=True),                    # closed_wrongly
        make_record("dv-000", 1, 3, passed=False, closed=False, n_malformed=3),    # never_closed (5 calls)
        make_record("dv-000", 1, 4, passed=False, closed=False, n_malformed=8),    # malformed_only
    ]
    assert [metrics.episode_label(r) for r in records] == [
        "closed_correctly", "closed_correctly", "closed_wrongly", "never_closed", "malformed_only"]
    assert metrics.outcome_labels(records) == {"closed_correctly": 0.4, "closed_wrongly": 0.2,
                                               "never_closed": 0.2, "malformed_only": 0.2}


def test_outcome_labels_are_exhaustive_and_exclusive():
    # Every combination that the harness can produce gets exactly one of the four labels, and the label is
    # decided by: closed or not, passed or not, and whether all messages were malformed.
    seen = set()
    for closed, passed, n_malformed in itertools.product((True, False), (True, False), (0, 1, 3, 8)):
        if passed and not closed:
            continue                                      # cannot happen: a pass needs a close
        if closed and n_malformed == 8:
            continue                                      # cannot happen: the close is a well-formed call
        record = make_record(passed=passed, closed=closed, n_malformed=n_malformed,
                             reward={0: 100, 1: 95, 3: 81}[n_malformed] if passed else 0)
        label = metrics.episode_label(record)
        assert label in metrics.OUTCOME_LABELS
        expected = ("closed_correctly" if closed and passed else "closed_wrongly" if closed
                    else "malformed_only" if n_malformed == 8 else "never_closed")
        assert label == expected
        seen.add(label)
        shares = metrics.outcome_labels([record])
        assert sorted(shares.values()) == [0.0, 0.0, 0.0, 1.0] and shares[label] == 1.0
    assert seen == set(metrics.OUTCOME_LABELS)


def test_outcome_labels_add_up_to_one():
    records = (scenario_records("dv-000", 1, [100, 0, 95]) + scenario_records("dv-001", 1, [0, 0])
               + [make_record("dv-001", 2, 0, passed=False, closed=False, n_malformed=8),
                  make_record("dv-001", 2, 1, passed=False, closed=False, n_malformed=2)])
    shares = metrics.outcome_labels(records)
    assert list(shares) == list(metrics.OUTCOME_LABELS)
    assert sum(shares.values()) == pytest.approx(1.0)
    # 7 episodes: 2 closed correctly, 3 closed wrongly, 1 never closed, 1 malformed only.
    assert shares == pytest.approx({"closed_correctly": 2 / 7, "closed_wrongly": 3 / 7, "never_closed": 1 / 7,
                                    "malformed_only": 1 / 7})


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
def _passes(pattern: str) -> list[int]:
    """"PFFP" -> [100, 0, 0, 100]."""
    return [100 if letter == "P" else 0 for letter in pattern]


def test_pass_at_k_by_hand():
    records = (scenario_records("dv-000", 1, _passes("FFPF")) + scenario_records("dv-000", 2, _passes("PFFF"))
               + scenario_records("dv-001", 1, _passes("FFFF")))
    # k = 2 (the first two samples): dv-000/s1 F F -> 0; dv-000/s2 P F -> 1; dv-001/s1 F F -> 0.
    # dv-000: (0 + 1) / 2 = 0.5. dv-001: 0. Mean over families: 0.25.
    # (Over the 3 scenarios it would be 1 / 3.)
    assert metrics.pass_at_k(records, 2) == 0.25
    # k = 4: dv-000/s1 has a pass at sample 2 -> 1; dv-000/s2 -> 1; dv-001/s1 -> 0.
    # Families: 1.0 and 0.0 -> 0.5.
    assert metrics.pass_at_k(records, 4) == 0.5
    # k = 1: only the first sample: 0, 1, 0 -> families 0.5 and 0 -> 0.25.
    assert metrics.pass_at_k(records, 1) == 0.25


def test_pass_at_k_uses_the_sample_index_and_not_the_position_in_the_list():
    # The pass is sample 3. In reversed order it is first in the list, but it is still not one of
    # "the first 2".
    records = list(reversed(scenario_records("dv-000", 1, _passes("FFFP"))))
    assert records[0]["sample"] == 3 and records[0]["passed"]
    assert metrics.pass_at_k(records, 2) == 0.0
    assert metrics.pass_at_k(records, 4) == 1.0


def test_pass_at_k_leaves_out_scenarios_with_fewer_than_k_samples():
    records = scenario_records("dv-000", 1, _passes("FFFF")) + scenario_records("dv-001", 1, _passes("PF"))
    # k = 4: only dv-000/s1 has 4 samples -> 0.0, and one scenario is left out.
    assert metrics.pass_at_k(records, 4) == 0.0
    assert metrics.n_scenarios_below(records, 4) == 1
    assert metrics.n_scenarios_below(records, 2) == 0
    # k = 2: both count: families 0.0 and 1.0 -> 0.5.
    assert metrics.pass_at_k(records, 2) == 0.5
    with pytest.raises(ValueError):
        metrics.pass_at_k(records, 8)                    # no scenario has 8 samples
    with pytest.raises(ValueError):
        metrics.pass_at_k(records, 0)


def test_pass_hat_k_by_hand():
    # One scenario, 4 samples, 3 passes.
    records = scenario_records("dv-000", 1, _passes("PPFP"))
    # k = 1: C(3,1) / C(4,1) = 3 / 4.
    assert metrics.pass_hat_k(records, 1) == 0.75
    # k = 2: C(3,2) / C(4,2) = 3 / 6 = 0.5. (The plain (3/4)**2 = 0.5625 is too high.)
    assert metrics.pass_hat_k(records, 2) == 0.5
    # k = 3: C(3,3) / C(4,3) = 1 / 4.   k = 4: C(3,4) / C(4,4) = 0 / 1 = 0.
    assert metrics.pass_hat_k(records, 3) == 0.25
    assert metrics.pass_hat_k(records, 4) == 0.0


def test_pass_hat_k_is_family_weighted():
    records = (scenario_records("dv-000", 1, _passes("PPPP")) + scenario_records("dv-000", 2, _passes("PPFF"))
               + scenario_records("dv-001", 1, _passes("PFFF")))
    # k = 2: dv-000/s1: C(4,2)/C(4,2) = 1. dv-000/s2: C(2,2)/C(4,2) = 1/6. dv-001/s1: C(1,2)/C(4,2) = 0.
    # dv-000: (1 + 1/6) / 2 = 7/12. dv-001: 0. Mean over families: 7/24.
    # (Over the 3 scenarios it would be (1 + 1/6 + 0) / 3 = 7/18.)
    assert metrics.pass_hat_k(records, 2) == pytest.approx(7 / 24)


def _brute_force_pass_hat_k(records: list[dict], k: int) -> float:
    """The definition, spelled out: per scenario, the share of all k-subsets of its samples in which every
    sample passed; then the mean over the scenarios of a family; then the mean over families."""
    scenarios: dict[tuple[str, int], list[bool]] = {}
    for r in records:
        scenarios.setdefault((r["family_id"], r["style"]), []).append(r["passed"])
    families: dict[str, list[float]] = {}
    for (family_id, _style), passes in sorted(scenarios.items()):
        subsets = list(itertools.combinations(passes, k))
        families.setdefault(family_id, []).append(sum(all(subset) for subset in subsets) / len(subsets))
    return sum(sum(values) / len(values) for values in families.values()) / len(families)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_pass_hat_k_equals_brute_force_over_all_k_subsets(k):
    records = (scenario_records("dv-000", 1, _passes("PPPPPP"))
               + scenario_records("dv-000", 2, _passes("PFPPFP"))
               + scenario_records("dv-000", 3, _passes("FFFFFF"))
               + scenario_records("dv-001", 1, _passes("PPPFPP"))
               + scenario_records("dv-001", 2, _passes("FPFFFP"))
               + scenario_records("dv-002", 4, _passes("PPFPPPP")))
    assert metrics.pass_hat_k(records, k) == pytest.approx(_brute_force_pass_hat_k(records, k), abs=1e-12)


def test_pass_hat_k_with_eight_samples_equals_brute_force():
    # The real design: 8 samples per scenario; k = 1, 2, 4, 8. Every number of passes from 0 to 8.
    records = []
    for n_passed in range(9):
        records += scenario_records(f"dv-{n_passed:03d}", 1, _passes("P" * n_passed + "F" * (8 - n_passed)))
    for k in (1, 2, 4, 8):
        assert metrics.pass_hat_k(records, k) == pytest.approx(_brute_force_pass_hat_k(records, k), abs=1e-12)
    # k = 8 by hand: only the scenario with 8 passes of 8 counts 1: 1 of 9 families.
    assert metrics.pass_hat_k(records, 8) == pytest.approx(1 / 9)
    # k = 1 is the pass rate: the mean of 0/8, 1/8, ..., 8/8 = 0.5.
    assert metrics.pass_hat_k(records, 1) == pytest.approx(0.5)
    assert metrics.pass_hat_k(records, 1) == pytest.approx(metrics.pass_rate(records))


def test_pass_hat_k_leaves_out_short_scenarios_and_refuses_k_below_one():
    records = scenario_records("dv-000", 1, _passes("PP")) + scenario_records("dv-001", 1, _passes("PFPF"))
    # k = 4: dv-000/s1 has only 2 samples and is left out; dv-001/s1: C(2,4) / C(4,4) = 0.
    assert metrics.pass_hat_k(records, 4) == 0.0
    assert metrics.n_scenarios_below(records, 4) == 1
    with pytest.raises(ValueError):
        metrics.pass_hat_k(records, 0)
    with pytest.raises(ValueError):
        metrics.pass_hat_k(records, 5)


def test_all_wordings_pass_by_hand():
    records = (scenario_records("dv-000", 1, _passes("PP")) + scenario_records("dv-000", 2, _passes("PF"))
               + scenario_records("dv-001", 1, _passes("PF")) + scenario_records("dv-001", 2, _passes("PF"))
               + scenario_records("dv-001", 3, _passes("PP")))
    # dv-000: style rates 1.0 and 0.5 -> product 0.5. dv-001: 0.5 * 0.5 * 1.0 = 0.25. Mean: 0.375.
    assert metrics.all_wordings_pass(records) == 0.375
    # One wording that always fails makes the family 0, whatever the others do.
    records += scenario_records("dv-000", 3, _passes("FF"))
    assert metrics.all_wordings_pass(records) == 0.125     # (0 + 0.25) / 2


def test_variance_shares_by_hand():
    wording_only = scenario_records("dv-000", 1, _passes("PP")) + scenario_records("dv-000", 2, _passes("FF"))
    # dv-000: family mean 0.5; style means 1 and 0. Between wordings: 2 * 0.25 + 2 * 0.25 = 1. Within: 0.
    assert metrics.variance_shares(wording_only) == {"wording": 1.0, "sampling": 0.0,
                                                     "wording_if_sampling_only": pytest.approx(1 / 3)}
    sampling_only = (scenario_records("dv-001", 1, _passes("PF"))
                     + scenario_records("dv-001", 2, _passes("PF")))
    # dv-001: family mean 0.5; style means 0.5 and 0.5. Between: 0. Within: 4 * 0.25 = 1.
    assert metrics.variance_shares(sampling_only) == {"wording": 0.0, "sampling": 1.0,
                                                      "wording_if_sampling_only": pytest.approx(1 / 3)}
    # Both families together: between 1, within 1 -> 0.5 and 0.5.
    # Share to expect from sampling alone: (styles - 1) summed = 1 + 1 = 2; (samples - 1) summed = 3 + 3 = 6.
    both = metrics.variance_shares(wording_only + sampling_only)
    assert both == {"wording": 0.5, "sampling": 0.5, "wording_if_sampling_only": pytest.approx(2 / 6)}


def test_variance_shares_unequal_split_by_hand():
    # dv-000: style 1 P P P F (mean 0.75), style 2 F F F F (mean 0). Family mean 3/8.
    # Between: 4 * (0.75 - 0.375)**2 + 4 * (0 - 0.375)**2 = 0.5625 + 0.5625 = 1.125.
    # Within: style 1: 3 * 0.25**2 + 0.75**2 = 0.1875 + 0.5625 = 0.75; style 2: 0. Total 1.875.
    # wording = 1.125 / 1.875 = 0.6; sampling = 0.75 / 1.875 = 0.4.
    records = scenario_records("dv-000", 1, _passes("PPPF")) + scenario_records("dv-000", 2, _passes("FFFF"))
    shares = metrics.variance_shares(records)
    assert shares["wording"] == pytest.approx(0.6)
    assert shares["sampling"] == pytest.approx(0.4)
    assert shares["wording"] + shares["sampling"] == pytest.approx(1.0)
    assert shares["wording_if_sampling_only"] == pytest.approx(1 / 7)       # (2 - 1) / (8 - 1)


def test_variance_shares_ignore_differences_between_families():
    # One family always passes and one always fails: all variation is between families, none inside.
    records = scenario_records("dv-000", 1, _passes("PP")) + scenario_records("dv-000", 2, _passes("PP")) \
        + scenario_records("dv-001", 1, _passes("FF")) + scenario_records("dv-001", 2, _passes("FF"))
    shares = metrics.variance_shares(records)
    assert (shares["wording"], shares["sampling"]) == (0.0, 0.0)
    # Adding such families to others does not change the shares of the others.
    mixed = scenario_records("dv-002", 1, _passes("PPPF")) + scenario_records("dv-002", 2, _passes("FFFF"))
    assert metrics.variance_shares(records + mixed)["wording"] == pytest.approx(0.6)


def test_wording_share_from_sampling_alone_is_3_of_31_for_the_real_design():
    # 4 styles of 8 samples in every family: (4 - 1) / (32 - 1) = 3 / 31 = 0.097.
    records = []
    for family in ("dv-000", "dv-001", "dv-002"):
        for style in (1, 2, 3, 4):
            records += scenario_records(family, style, _passes("PFPFPFPF"))
    assert metrics.variance_shares(records)["wording_if_sampling_only"] == pytest.approx(3 / 31)
    # Simulated check of that number: 1,000 families whose every sample passes with the same chance 0.5.
    rng = np.random.default_rng(0)
    records = []
    for family in range(1000):
        for style in (1, 2, 3, 4):
            passes = ["P" if value else "F" for value in rng.random(8) < 0.5]
            records += scenario_records(f"dv-{family:04d}", style, _passes("".join(passes)))
    shares = metrics.variance_shares(records)
    assert shares["wording"] == pytest.approx(3 / 31, abs=0.01)


# ---------------------------------------------------------------------------
# Smallest detectable effect
# ---------------------------------------------------------------------------
def test_two_halves_mde_by_hand():
    # Three families, one scenario each, two samples each: each half has one sample per scenario.
    # dv-000: P F -> the halves differ by +1 or -1, whichever way the seed splits.
    # dv-001: P P -> difference 0. dv-002: F F -> difference 0.
    # Differences (s, 0, 0) with s = +1 or -1: mean s/3; squared deviations 4/9 + 1/9 + 1/9 = 2/3;
    # variance (n - 1 = 2 in the divisor) = 1/3; standard deviation 0.57735.
    # se_halves = 0.57735 / sqrt(3) = 1/3. se = (1/3) / sqrt(2) = 0.235702. mde = 2.8 * 0.235702 = 0.659966.
    records = (scenario_records("dv-000", 1, _passes("PF")) + scenario_records("dv-001", 1, _passes("PP"))
               + scenario_records("dv-002", 1, _passes("FF")))
    for seed in range(6):
        result = metrics.two_halves_mde(records, seed)
        assert result["n_families"] == 3
        assert result["se_halves"] == pytest.approx(1 / 3, abs=1e-12)
        assert result["se"] == pytest.approx(0.235702, abs=1e-6)
        assert result["mde"] == pytest.approx(0.659966, abs=1e-6)
        assert result["mde"] == pytest.approx(2.8 * result["se_halves"] / math.sqrt(2), abs=1e-12)


def test_two_halves_mde_split_is_inside_each_scenario():
    # dv-000 has style 1 all pass and style 2 all fail. Each half gets half of EACH scenario, so both halves
    # of dv-000 have the pass rate 0.5 and the difference is 0, whatever the seed. The same holds for dv-001.
    # (A split of the family's 8 records without regard to the scenario would give differences other than 0.)
    records = (scenario_records("dv-000", 1, _passes("PPPP"))
               + scenario_records("dv-000", 2, _passes("FFFF"))
               + scenario_records("dv-001", 1, _passes("FFFF"))
               + scenario_records("dv-001", 2, _passes("PPPP")))
    for seed in range(6):
        assert metrics.two_halves_mde(records, seed)["se_halves"] == 0.0


def test_two_halves_mde_depends_on_the_seed_only_through_the_split():
    records = []
    for family in range(6):
        records += scenario_records(f"dv-{family:03d}", 1, _passes("PPFFPFPF"))
        records += scenario_records(f"dv-{family:03d}", 2, _passes("PFFFFFFP"))
    assert metrics.two_halves_mde(records, 3) == metrics.two_halves_mde(records, 3)
    values = {metrics.two_halves_mde(records, seed)["se_halves"] for seed in range(10)}
    assert len(values) > 1


def test_two_halves_mde_needs_two_families_and_two_samples():
    with pytest.raises(ValueError):
        metrics.two_halves_mde(scenario_records("dv-000", 1, _passes("PF")), 0)
    with pytest.raises(ValueError):
        metrics.two_halves_mde(scenario_records("dv-000", 1, _passes("P"))
                               + scenario_records("dv-001", 1, _passes("F")), 0)


def test_two_halves_mde_estimates_the_noise_of_a_comparison_of_two_full_evaluations():
    # 400 families, 4 styles, 8 samples; every sample passes with chance 0.5 (sampling noise only).
    # A family's pass rate from 32 samples has variance 0.25 / 32. The difference of two models evaluated
    # like this has variance 2 * 0.25 / 32 = 1/64 per family, so the standard error of the mean difference
    # over 400 families is sqrt(1/64 / 400) = 0.00625. This is what `se` must estimate.
    # (Without the division by sqrt(2) it would be 0.00884; with a multiplication, 0.0125.)
    rng = np.random.default_rng(1)
    records = []
    for family in range(400):
        for style in (1, 2, 3, 4):
            passes = "".join("P" if value else "F" for value in rng.random(8) < 0.5)
            records += scenario_records(f"tr-{family:03d}", style, _passes(passes))
    result = metrics.two_halves_mde(records, 0)
    assert result["n_families"] == 400
    assert result["se"] == pytest.approx(0.00625, rel=0.12)
    assert result["mde"] == pytest.approx(2.8 * 0.00625, rel=0.12)


def test_two_halves_mde_hardly_moves_with_the_seed_and_agrees_with_the_exact_expectation():
    """One random split of each scenario's samples is a noisy estimate: on real calibration episodes the
    number moved by a factor of 2.7 with the seed. The function averages the squared standard error over
    200 splits. Checked here: (1) seeds agree to within 10%; (2) the result is the exact expectation over
    all splits, worked out from the pass counts; (3) one split alone still moves a lot (what was repaired).

    The exact expectation. A scenario has n = 8 samples and c passes; a half holds h = 4 of them, and its
    number of passes X has the variance h * (c/n) * (1 - c/n) * (n - h) / (n - 1) (drawing without
    replacement). A family has 4 scenarios, so each half has 16 episodes, and the difference of the two
    halves' pass rates is sum(2 X - c) / 16, with variance 4 * sum(var X) / 256. The differences of the
    families have mean 0 and are independent, so the expected sample variance is the mean of these
    variances, and the squared standard error is that mean divided by the number of families."""
    rng = np.random.default_rng(7)
    records, family_variances = [], []
    for family in range(24):
        chance = 0.1 + 0.8 * family / 23
        variance = 0.0
        for style in (1, 2, 3, 4):
            passed = rng.random(8) < chance
            records += scenario_records(f"ca-{family:03d}", style, [100 if x else 0 for x in passed])
            share = passed.sum() / 8
            variance += 4 * (4 * share * (1 - share) * (8 - 4) / (8 - 1))
        family_variances.append(variance / 16 ** 2)
    exact_se_halves = math.sqrt(sum(family_variances) / 24 / 24)

    results = [metrics.two_halves_mde(records, seed) for seed in range(6)]
    effects = [result["mde"] for result in results]
    assert max(effects) / min(effects) < 1.10
    for result in results:
        assert result["n_splits"] == 200 and result["n_families"] == 24
        assert result["se_halves"] == pytest.approx(exact_se_halves, rel=0.06)
        assert result["mde"] == pytest.approx(2.8 * result["se_halves"] / math.sqrt(2), abs=1e-12)
        assert result["mde_lowest_split"] < result["mde"] < result["mde_highest_split"]
    # One split alone (what the function did before): the same records, other seeds, other numbers.
    single = [metrics.two_halves_mde(records, seed, n_splits=1)["mde"] for seed in range(40)]
    assert max(single) / min(single) > 1.3
    one = metrics.two_halves_mde(records, 0, n_splits=1)
    assert one["mde_lowest_split"] == one["mde_highest_split"] == pytest.approx(one["mde"])
    with pytest.raises(ValueError, match="n_splits"):
        metrics.two_halves_mde(records, 0, n_splits=0)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------
def test_family_bootstrap_ci_known_answers():
    # All families equal: every draw has the same mean, so the interval has width 0.
    assert metrics.family_bootstrap_ci([0.25] * 12, 500, 0, 0.95) == {"mean": 0.25, "lo": 0.25, "hi": 0.25}
    # 50 families at 1 and 50 at 0: a draw is Binomial(100, 0.5) / 100. P(X <= 39) = 0.0176 and
    # P(X <= 40) = 0.0284, so the 2.5th percentile is 0.40 and by symmetry the 97.5th is 0.60.
    result = metrics.family_bootstrap_ci([1.0] * 50 + [0.0] * 50, 10_000, 0, 0.95)
    assert result["mean"] == 0.5
    assert result["lo"] == pytest.approx(0.40, abs=0.0101)
    assert result["hi"] == pytest.approx(0.60, abs=0.0101)


def test_family_bootstrap_ci_two_families():
    # Values 0 and 1: a draw of two families has mean 0, 0.5 or 1 with chances 1/4, 1/2, 1/4.
    # 95%: lo 0, hi 1. Level 0.4 (30th and 70th percentile): both 0.5.
    assert metrics.family_bootstrap_ci([0.0, 1.0], 4000, 0, 0.95) == {"mean": 0.5, "lo": 0.0, "hi": 1.0}
    assert metrics.family_bootstrap_ci([0.0, 1.0], 4000, 0, 0.4) == {"mean": 0.5, "lo": 0.5, "hi": 0.5}


def test_family_bootstrap_ci_seed_and_input_checks():
    values = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.125, 0.875]
    first = metrics.family_bootstrap_ci(values, 1000, 4, 0.95)
    assert first == metrics.family_bootstrap_ci(values, 1000, 4, 0.95)
    assert first["lo"] < first["mean"] < first["hi"]
    limits = {(r["lo"], r["hi"]) for r in (metrics.family_bootstrap_ci(values, 1000, seed, 0.95)
                                           for seed in range(10))}
    assert len(limits) > 1
    for bad in ([], [[0.5, 0.5]]):
        with pytest.raises(ValueError):
            metrics.family_bootstrap_ci(bad, 100, 0, 0.95)
    with pytest.raises(ValueError):
        metrics.family_bootstrap_ci(values, 100, 0, 95)


def test_family_bootstrap_ci_covers_the_true_mean_at_about_the_stated_rate():
    # 400 data sets of 40 families with true mean 0.5: a 90% interval must contain 0.5 about 9 times in 10.
    rng = np.random.default_rng(7)
    hits = 0
    for repeat in range(400):
        values = rng.normal(0.5, 0.2, 40)
        result = metrics.family_bootstrap_ci(values, 800, repeat, 0.90)
        hits += result["lo"] <= 0.5 <= result["hi"]
    assert 0.84 <= hits / 400 <= 0.94


@pytest.mark.parametrize("p, df, expected", [
    # Values of a printed t table (two-sided 5%: p = 0.975).
    (0.975, 1, 12.706), (0.975, 2, 4.303), (0.975, 5, 2.571), (0.975, 10, 2.228), (0.975, 30, 2.042),
    (0.975, 47, 2.012), (0.975, 120, 1.980), (0.95, 10, 1.812), (0.995, 10, 3.169), (0.9, 4, 1.533),
])
def test_t_quantile_against_a_printed_table(p, df, expected):
    assert metrics.t_quantile(p, df) == pytest.approx(expected, abs=0.0006)


def test_t_quantile_approaches_the_normal_value_and_refuses_bad_input():
    assert metrics.t_quantile(0.975, 100_000) == pytest.approx(1.960, abs=0.001)
    for p, df in ((0.5, 10), (1.0, 10), (0.975, 0)):
        with pytest.raises(ValueError):
            metrics.t_quantile(p, df)


def test_t_interval_by_hand():
    # Differences 1, 2, 3: mean 2; standard deviation 1; se = 1 / sqrt(3) = 0.57735; df = 2.
    # t(0.975, 2) = 4.3027. Half width = 4.3027 * 0.57735 = 2.4841. Interval -0.4841 to 4.4841.
    result = metrics.t_interval([1.0, 2.0, 3.0], 0.95)
    assert result["mean"] == 2.0
    assert result["df"] == 2
    assert result["se"] == pytest.approx(0.57735, abs=1e-5)
    assert result["lo"] == pytest.approx(-0.4841, abs=5e-4)
    assert result["hi"] == pytest.approx(4.4841, abs=5e-4)


def test_t_interval_is_wider_than_the_normal_interval_and_needs_two_families():
    # 48 families: t(0.975, 47) = 2.012 against 1.960, so the t-interval is 2.6% wider than mean -/+ 1.96 se.
    diffs = np.linspace(-0.2, 0.4, 48)
    result = metrics.t_interval(diffs, 0.95)
    normal_width = 2 * 1.960 * result["se"]
    assert (result["hi"] - result["lo"]) / normal_width == pytest.approx(2.012 / 1.960, abs=0.001)
    # All differences equal: no spread, an interval of width 0.
    assert metrics.t_interval([0.25, 0.25, 0.25], 0.95)["lo"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        metrics.t_interval([0.5], 0.95)


def test_metric_outputs_are_plain_json():
    records = (scenario_records("dv-000", 1, [100, 95, 0, 0])
               + scenario_records("dv-000", 4, [0, 0, 100, 100])
               + scenario_records("dv-001", 1, [100, 100, 100, 0])
               + scenario_records("dv-001", 4, [0, 0, 0, 0]))
    family_rates = list(metrics.family_pass_rates(records).values())
    outputs = [metrics.family_pass_rates(records), metrics.pass_rate_by(records, "seen_held_out"),
               metrics.zero_variance_share(records), metrics.turn_stats(records),
               metrics.outcome_labels(records), metrics.variance_shares(records),
               metrics.two_halves_mde(records, 0), metrics.family_bootstrap_ci(family_rates, 200, 0, 0.95),
               metrics.t_interval(family_rates, 0.95)]
    for output in outputs:
        assert json.loads(json.dumps(output, allow_nan=False)) == output
