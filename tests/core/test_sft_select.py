"""Tests of select_sft_records (accessdesk/train/sft.py): which attempts become training data of SFT.

The records are real episode records of train families, played by scripted policies. They carry no token
data, because the selection reads none.

The first part proves rule 3: an attempt with a refused call is never chosen. A refused call is a
well-formed call that the tool answered with an error, for example a close with a made-up id. The attempt
can pass with the full reward after it, so without the rule it would be training data.
The second part holds the other rules in place (they had no test of their own before).
"""
from __future__ import annotations

import pytest

from accessdesk.eval import monitors
from accessdesk.harness import policies as P
from accessdesk.train.sft import action_sequence, select_sft_records
from tests.core.test_monitors import (already_without_lookup, pick, play, refused_close,
                                      refused_close_then_correct)


def refused_close_then_one_malformed_message(text) -> P.Policy:
    """A refused close, one empty message, then the correct solution: a pass with a penalty."""
    def first(desk: P.Desk) -> None:
        refused_close(desk)
        desk.say("")

    return P.oracle_after(text, first)


@pytest.fixture(scope="module")
def train(families) -> list[dict]:
    """Four train families whose role is to be granted (the correct path has 4 turns, the budget is 5)."""
    chosen = [family for family in families["train"] if family["outcome"] == "grant"][:4]
    assert len(chosen) == 4 and all(family["split"] == "train" for family in chosen)
    return chosen


# ---------------------------------------------------------------------------
# Rule 3: no attempt with a refused call
# ---------------------------------------------------------------------------
def test_a_clean_pass_with_a_refused_call_is_not_chosen(train, text, cfg):
    record = play(train[0], text, cfg, refused_close_then_correct(text))
    assert (record["passed"], record["reward"], record["n_tool_errors"]) == (True, cfg.pass_reward, 1)
    chosen, report = select_sft_records([record], cfg)
    assert chosen == []
    assert (report["n_passed"], report["n_clean_passes"], report["n_with_refused_call"]) == (1, 1, 1)
    assert (report["n_candidates"], report["n_chosen"], report["n_families_covered"]) == (0, 0, 0)


def test_the_same_attempt_without_the_refused_call_is_chosen(train, text, cfg):
    with_refused = play(train[0], text, cfg, refused_close_then_correct(text))
    without = play(train[0], text, cfg, P.oracle(text))
    # The two attempts differ in the refused call only: the first turn of the one, absent in the other.
    calls = [(turn["tool_key"], turn["args"]) for turn in with_refused["turns"]]
    assert calls[1:] == [(turn["tool_key"], turn["args"]) for turn in without["turns"]]
    assert (without["reward"], without["n_tool_errors"]) == (with_refused["reward"], 0)
    chosen, report = select_sft_records([without], cfg)
    assert chosen == [without]
    assert (report["n_with_refused_call"], report["n_candidates"], report["n_chosen"]) == (0, 1, 1)


def test_the_rule_decides_which_attempt_of_a_scenario_is_chosen(train, text, cfg):
    # Two clean passes of one scenario, both 5 turns long. Sample 0 has a refused call, sample 1 an extra
    # read. Without rule 3 the lower sample number would win the tie.
    refused = play(train[0], text, cfg, refused_close_then_correct(text), sample=0)
    extra_read = play(train[0], text, cfg, P.oracle(text, extra_reads=1), sample=1)
    assert (refused["reward"], refused["n_turns"]) == (100, 5)
    assert (extra_read["reward"], extra_read["n_turns"]) == (100, 5)
    assert action_sequence(refused) != action_sequence(extra_read)
    chosen, report = select_sft_records([refused, extra_read], cfg)
    assert chosen == [extra_read]
    assert (report["n_clean_passes"], report["n_with_refused_call"], report["n_candidates"]) == (2, 1, 1)


def test_the_report_counts_the_passes_that_the_rule_removed_and_no_other_episode(train, text, cfg):
    records = [
        play(train[0], text, cfg, refused_close_then_correct(text)),                 # clean pass, refused
        play(train[1], text, cfg, P.oracle(text)),                                   # clean pass
        play(train[2], text, cfg, P.close_at_once(text, "granted")),                 # fail, 8 refused calls
        play(train[3], text, cfg, refused_close_then_one_malformed_message(text)),   # penalised pass, refused
    ]
    assert [(r["passed"], r["reward"], r["n_tool_errors"]) for r in records] == [
        (True, 100, 1), (True, 100, 0), (False, 0, 8), (True, 93, 1)]
    chosen, report = select_sft_records(records, cfg)
    assert chosen == [records[1]]
    # Rule 2 keeps the 2 clean passes; rule 3 removes 1 of them. The fail and the penalised pass were
    # never candidates, so the rule did not remove them.
    assert (report["n_passed"], report["n_clean_passes"], report["n_with_refused_call"]) == (3, 2, 1)
    assert report["n_candidates"] + report["n_with_refused_call"] == report["n_clean_passes"]
    # With penalised passes allowed, rule 2 keeps the 3 passes and rule 3 removes 2 of them.
    chosen, report = select_sft_records(records, cfg, allow_penalised=True)
    assert chosen == [records[1]]
    assert (report["n_passed"], report["n_with_refused_call"], report["n_candidates"]) == (3, 2, 1)
    assert report["n_candidates"] + report["n_with_refused_call"] == report["n_passed"]


def test_allow_penalised_does_not_bring_a_refused_call_back(train, text, cfg):
    clean_refused = play(train[0], text, cfg, refused_close_then_correct(text))
    penalised_refused = play(train[1], text, cfg, refused_close_then_one_malformed_message(text))
    penalised = play(train[2], text, cfg, P.stumbling_oracle(text, 1))
    assert (penalised["passed"], penalised["reward"], penalised["n_tool_errors"]) == (True, 95, 0)
    records = [clean_refused, penalised_refused, penalised]
    chosen, report = select_sft_records(records, cfg, allow_penalised=True)
    assert chosen == [penalised]                      # the penalty is allowed, the refused call is not
    assert (report["n_with_refused_call"], report["n_chosen"], report["n_chosen_with_penalty"]) == (2, 1, 1)
    assert all(record["n_tool_errors"] == 0 for record in chosen)
    assert select_sft_records(records, cfg)[0] == []  # and without the switch nothing is left


def test_rule_3_does_not_remove_a_guessed_id_that_happens_to_exist(families, text, cfg):
    # The limit of rule 3, held in place so that nobody reads more into it. The policy never looks the
    # person up: its first call holds the right person id, as a lucky guess would. The tool serves every
    # call, so the attempt has no refused call, and it is chosen.
    family = pick(families, "already", split="train")
    record = play(family, text, cfg, already_without_lookup(text, family))
    assert (record["passed"], record["reward"], record["n_tool_errors"]) == (True, cfg.pass_reward, 0)
    assert monitors.id_not_from_a_result(record, text)["n_call_served"] == 2    # the monitor m2 sees it
    chosen, report = select_sft_records([record], cfg)
    assert chosen == [record]
    assert (report["n_with_refused_call"], report["n_candidates"]) == (0, 1)


# ---------------------------------------------------------------------------
# The other rules
# ---------------------------------------------------------------------------
def test_harness_errors_fails_and_penalised_passes_are_not_chosen(train, text, cfg):
    clean = play(train[0], text, cfg, P.oracle(text))
    broken = {**play(train[1], text, cfg, P.oracle(text)), "harness_error": "the sampler stopped"}
    failed = play(train[2], text, cfg, P.close_without_write(text, "granted"))
    penalised = play(train[3], text, cfg, P.stumbling_oracle(text, 1))
    chosen, report = select_sft_records([clean, broken, failed, penalised], cfg)
    assert chosen == [clean]
    assert (report["n_records"], report["n_harness_errors"], report["n_passed"], report["n_clean_passes"],
            report["n_with_refused_call"], report["n_chosen"]) == (4, 1, 2, 1, 0, 1)
    chosen, report = select_sft_records([clean, broken, failed, penalised], cfg, allow_penalised=True)
    assert chosen == [clean, penalised] and report["n_chosen_with_penalty"] == 1


def test_one_attempt_per_scenario_the_shortest_then_the_lowest_sample(train, text, cfg):
    long = play(train[0], text, cfg, P.oracle(text, extra_reads=1), sample=0)      # 5 turns
    short_late = play(train[0], text, cfg, P.oracle(text), sample=2)               # 4 turns
    short_early = play(train[0], text, cfg, P.oracle(text), sample=1)              # 4 turns, the same calls
    other = play(train[1], text, cfg, P.oracle(text), sample=3)
    records = [long, short_late, short_early, other]
    chosen, report = select_sft_records(records, cfg)
    assert chosen == sorted([short_early, other], key=lambda r: r["scenario_id"])
    # 4 clean passes; the two short ones made the same calls, so 3 are distinct; 2 scenarios.
    assert (report["n_clean_passes"], report["n_distinct"], report["n_chosen"], report["n_scenarios"],
            report["n_families_covered"]) == (4, 3, 2, 2, 2)
    assert select_sft_records(records[::-1], cfg)[0] == chosen       # the input order does not matter


def test_only_train_records_of_one_set_up_are_accepted(train, families, text, cfg):
    dev = play(pick(families, "grant", split="dev"), text, cfg, P.oracle(text))
    with pytest.raises(ValueError, match="train split only"):
        select_sft_records([dev], cfg)
    record = play(train[0], text, cfg, P.oracle(text))
    with pytest.raises(ValueError, match="different set-ups"):
        select_sft_records([record, {**record, "prompt": "plain"}], cfg)
