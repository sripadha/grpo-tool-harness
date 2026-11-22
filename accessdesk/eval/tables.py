"""The results table: one row per arm, one row per comparison of two arms.

An arm is one model under one prompt, one rule set and one tool set (for example "base", "sft", "grpo-seed1").
Every number comes from saved episode records through `accessdesk/eval/metrics.py`. The interval of a
comparison comes from the paired bootstrap by family, reached only through
`accessdesk.impl.get("paired_family_bootstrap")`; the table says which version (main or second) computed it.

Command line (one line):
    python -m accessdesk.eval.tables --arm base=results/base.jsonl --arm sft=results/sft.jsonl
        --baseline base --pair base:sft --out-json table.json --out-md table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from accessdesk import impl
from accessdesk.config import COMPARED_FIELDS
from accessdesk.eval import metrics
from accessdesk.harness.episode import SCHEMA
from accessdesk.types import OUTCOMES

PASS_AT_K = 8                   # pass@8: at least one of 8 attempts passes
PASS_HAT_KS = (1, 2, 4, 8)      # pass^k: k attempts all pass


def scenario_sample_counts(arm: str, records: list[dict]) -> dict[tuple[str, int], int]:
    """(family_id, style) -> number of records, harness errors included: what the arm was run on.

    Raises ValueError if the records mix rule sets, tool sets or prompts, if one sample of a scenario
    appears twice (two runs pooled by mistake), or if two records of a scenario carry the same seed entry
    (`seed`: run seed, pass, chunk, position): that is one episode saved under two sample numbers."""
    conditions = sorted({(r["ruleset"], r["toolset"], r["prompt"]) for r in records})
    if len(conditions) != 1:
        raise ValueError(f"arm {arm!r}: needs records of exactly one (ruleset, toolset, prompt); "
                         f"found {conditions}")
    seen: set[tuple[str, int, int]] = set()
    seen_seeds: set[tuple[str, int, str]] = set()
    counts: dict[tuple[str, int], int] = {}
    for r in records:
        sample_key = (r["family_id"], r["style"], r["sample"])
        if sample_key in seen:
            raise ValueError(f"arm {arm!r}: sample {sample_key} appears more than once")
        seen.add(sample_key)
        if r.get("seed") is not None:
            seed_key = (r["family_id"], r["style"], json.dumps(r["seed"], sort_keys=True))
            if seed_key in seen_seeds:
                raise ValueError(f"arm {arm!r}: two records of scenario {sample_key[:2]} carry the same seed "
                                 f"entry {r['seed']}: one episode is there under two sample numbers")
            seen_seeds.add(seed_key)
        counts[sample_key[:2]] = counts.get(sample_key[:2], 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def check_same_coverage(records_by_arm: dict[str, list[dict]],
                        baseline_arm: str) -> dict[tuple[str, int], int]:
    """Raise ValueError unless every arm has exactly the scenarios of the baseline arm, with the same number
    of samples for each scenario. Returns the baseline's (family_id, style) -> sample count."""
    expected = scenario_sample_counts(baseline_arm, records_by_arm[baseline_arm])
    for arm, records in records_by_arm.items():
        found = scenario_sample_counts(arm, records)
        if found != expected:
            keys = set(found) | set(expected)
            different = sorted(key for key in keys if found.get(key) != expected.get(key))
            raise ValueError(f"arm {arm!r} does not cover the same scenarios and sample counts as "
                             f"{baseline_arm!r}: {len(different)} scenarios differ, the first is "
                             f"{different[0]}")
    return expected


def check_same_text(records_by_arm: dict[str, list[dict]]) -> None:
    """Raise ValueError unless every record of every arm was made with the same text file (`text_sha256`).

    A family id names the same scenario only under one text file: the wordings, the tool names and the
    prompts come from it. Scenario files built from two text files have the same family ids."""
    by_arm = {arm: sorted({r["text_sha256"] for r in records}) for arm, records in records_by_arm.items()}
    if len({text_hash for hashes in by_arm.values() for text_hash in hashes}) > 1:
        short = {arm: [text_hash[:12] for text_hash in hashes] for arm, hashes in by_arm.items()}
        raise ValueError(f"the arms were not all run with the same text file (text_sha256 per arm): {short}")


def check_same_data(records_by_arm: dict[str, list[dict]]) -> list[str]:
    """Raise ValueError if the arms were run on different scenario files; return notes on what was not
    checked.

    A record may carry `data_sha256`: the SHA-256 of the scenario file that its family was loaded from. A
    family id names the same family only inside one scenario file: files built with another seed or another
    config have the same family ids and other worlds. So every arm whose records carry the key must carry
    the same set of values (one value per split file that the arm was run on).
    A record without the key (an older run, a hand-made record) cannot be checked. It is accepted, and the
    returned notes say for each arm how many such records it has. The notes go into the table."""
    by_arm = {arm: sorted({r["data_sha256"] for r in records if r.get("data_sha256") is not None})
              for arm, records in records_by_arm.items()}
    carried = {arm: hashes for arm, hashes in by_arm.items() if hashes}
    if len({tuple(hashes) for hashes in carried.values()}) > 1:
        short = {arm: [data_hash[:12] for data_hash in hashes] for arm, hashes in carried.items()}
        raise ValueError("the arms were not all run on the same scenario file (data_sha256 per arm): "
                         f"{short}")
    notes = []
    for arm, records in records_by_arm.items():
        n_without = sum(1 for r in records if r.get("data_sha256") is None)
        if n_without:
            notes.append(f"Arm {arm}: {n_without} of {len(records)} records carry no data_sha256, so it was "
                         f"not checked that this arm was run on the same scenario file as the others.")
    return notes


def _distinct(values: list) -> list:
    """The values without repeats, compared with ==, in the order of their first appearance."""
    found: list = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def check_same_grader(records_by_arm: dict[str, list[dict]]) -> None:
    """Raise ValueError unless every record of every arm has the record format of this code (`schema`) and
    names the same versions of verify and reward (`impl`).

    One table is one grader: episodes scored by the main versions and episodes scored by the second
    versions never stand in the same table, and neither do records of another format."""
    schemas = _distinct([r.get("schema") for records in records_by_arm.values() for r in records])
    if schemas != [SCHEMA]:
        raise ValueError(f"every record must have schema {SCHEMA}; found {schemas}")
    by_arm = {arm: _distinct([r.get("impl") for r in records]) for arm, records in records_by_arm.items()}
    if len(_distinct([value for values in by_arm.values() for value in values])) > 1:
        raise ValueError("the records were not all scored by the same versions of verify and reward "
                         f"(impl per arm): {by_arm}")


def check_same_settings(records_by_arm: dict[str, list[dict]]) -> list[str]:
    """Raise ValueError if the arms were run with different settings; return notes on what was not checked.

    A record carries `settings`: the turn limit, the token cap, the parser mode, the sampling settings and
    the reward numbers of its run (`RunConfig.compared_settings`). They decide what an episode is and how it
    is scored, so every record of every arm must carry the same values. The chunk size is not among them.
    A record without the key (an older run) names only `config_sha256`, the hash of its whole config. If all
    records of all arms name the same hash, the settings are the same. If not, the difference may be the
    chunk size only, and this cannot be seen from a hash: the arm is accepted, and the returned notes say
    so. The notes go into the table."""
    by_arm = {arm: _distinct([r["settings"] for r in records if r.get("settings") is not None])
              for arm, records in records_by_arm.items()}
    values = _distinct([value for found in by_arm.values() for value in found])
    if len(values) > 1:
        different = [name for name in COMPARED_FIELDS
                     if len(_distinct([value.get(name) for value in values])) > 1]
        shown = {arm: [{name: value.get(name) for name in different} for value in found]
                 for arm, found in by_arm.items()}
        raise ValueError(f"the arms were not all run with the same settings; {different} differ "
                         f"(per arm): {shown}")
    hashes = {r["config_sha256"] for records in records_by_arm.values() for r in records}
    notes = []
    for arm, records in records_by_arm.items():
        n_without = sum(1 for r in records if r.get("settings") is None)
        if n_without and len(hashes) > 1:
            notes.append(f"Arm {arm}: {n_without} of {len(records)} records carry no settings, and the "
                         f"records do not all name the same config_sha256, so it was not checked that this "
                         f"arm was run with the same turn limit, token cap, parser mode, sampling settings "
                         f"and reward numbers as the others.")
    return notes


def check_arms_differ(records_by_arm: dict[str, list[dict]]) -> None:
    """Raise ValueError if two arms hold exactly the same records: one file given under two arm names."""
    names = list(records_by_arm)
    for i, arm_a in enumerate(names):
        for arm_b in names[i + 1:]:
            if records_by_arm[arm_a] == records_by_arm[arm_b]:
                raise ValueError(f"the arms {arm_a!r} and {arm_b!r} hold exactly the same records: the same "
                                 f"file was given under two names")


def _valid_scenarios(records: list[dict]) -> set[tuple[str, int]]:
    return {(r["family_id"], r["style"]) for r in metrics.valid_records(records)}


def check_same_valid_scenarios(arm_a: str, arm_b: str, records_a: list[dict], records_b: list[dict]) -> None:
    """Raise ValueError unless the two arms have a valid episode in exactly the same scenarios.

    `check_same_coverage` counts harness errors too. If harness errors take every sample of one wording of
    a family in one arm only, the family's pass rate is over fewer wordings in that arm, and the paired
    difference would compare different wordings."""
    only_one = sorted(_valid_scenarios(records_a) ^ _valid_scenarios(records_b))
    if only_one:
        raise ValueError(f"{arm_a!r} and {arm_b!r} cannot be paired: these scenarios have valid episodes in "
                         f"only one of them: {only_one[:5]} ({len(only_one)} in all)")


def condition_notes(records_by_arm: dict[str, list[dict]], baseline_arm: str) -> list[str]:
    """What a reader must know about the conditions of the arms: an arm under another rule set, tool set
    or prompt than the baseline, and records of more than one split."""
    def condition(records: list[dict]) -> tuple[str, str, str]:
        return records[0]["ruleset"], records[0]["toolset"], records[0]["prompt"]

    notes = []
    base = condition(records_by_arm[baseline_arm])
    for arm, records in records_by_arm.items():
        if condition(records) != base:
            notes.append(f"Arm {arm} was run under another rule set, tool set or prompt {condition(records)} "
                         f"than the baseline arm {baseline_arm} {base}: a difference between them compares "
                         f"two conditions.")
    splits = sorted({r["split"] for records in records_by_arm.values() for r in records})
    if len(splits) > 1:
        notes.append(f"The records are of more than one split ({', '.join(splits)}): every pass rate here "
                     f"is over the families of all of them together.")
    return notes


def _for_k(function, records: list[dict], k: int) -> float | None:
    """function(records, k) for pass@k or pass^k; None if no scenario has k valid samples."""
    if metrics.n_scenarios_below(records, k) == len(_valid_scenarios(records)):
        return None
    return function(records, k)


def _smallest_detectable_effect(records: list[dict], seed: int) -> dict | None:
    """`metrics.two_halves_mde` of the arm's own records; None if it has fewer than 2 families with a
    scenario of 2 valid samples (a scripted policy has one sample per scenario)."""
    try:
        return metrics.two_halves_mde(records, seed)
    except ValueError:
        return None


def arm_row(arm: str, records: list[dict], n_draws: int, seed: int, level: float) -> dict:
    """The numbers of one arm. The interval of the pass rate resamples families (not paired).

    The conditions of the arm are part of the row: `splits`, `ruleset`, `toolset`, `prompt`,
    `text_is_placeholder` and the `config_sha256` values of its records.
    The reliability and behaviour numbers are those of `accessdesk/eval/metrics.py`, where each is defined:
    `pass_at_8`, `pass_hat_k` (k = 1, 2, 4, 8; None where no scenario has k valid samples, and
    `n_scenarios_left_out` says how many scenarios have fewer), `all_wordings_pass`, `variance_shares`,
    `turn_stats`, `outcome_labels`, `reward_distribution` (reward value -> episodes, all valid episodes) and
    `smallest_detectable_effect` (`two_halves_mde` of this arm's own records, with the table's seed)."""
    rates = metrics.family_pass_rates(records)
    interval = metrics.family_bootstrap_ci(list(rates.values()), n_draws, seed, level)
    by_outcome = metrics.pass_rate_by(records, "outcome")
    by_style = metrics.pass_rate_by(records, "seen_held_out")
    zero_variance = metrics.zero_variance_share(records)
    return {
        "arm": arm,
        "splits": sorted({r["split"] for r in records}),
        "ruleset": records[0]["ruleset"], "toolset": records[0]["toolset"], "prompt": records[0]["prompt"],
        "text_is_placeholder": any(r["text_is_placeholder"] for r in records),
        "config_sha256": sorted({r["config_sha256"] for r in records}),
        "n_families": len(rates),
        "n_episodes": len(metrics.valid_records(records)),
        "n_harness_errors": metrics.harness_error_count(records),
        "pass_rate": metrics.pass_rate(records),
        "pass_rate_lo": interval["lo"],
        "pass_rate_hi": interval["hi"],
        "pass_rate_by_outcome": {outcome: by_outcome.get(outcome) for outcome in OUTCOMES},
        "pass_rate_seen": by_style.get("seen"),
        "pass_rate_held_out": by_style.get("held_out"),
        "zero_variance_equal_reward": zero_variance["equal_reward"],
        "zero_variance_equal_passfail": zero_variance["equal_passfail"],
        "invalid_call_rate": metrics.invalid_call_rate(records),
        "pass_at_8": _for_k(metrics.pass_at_k, records, PASS_AT_K),
        "pass_hat_k": {str(k): _for_k(metrics.pass_hat_k, records, k) for k in PASS_HAT_KS},
        "n_scenarios_left_out": {str(k): metrics.n_scenarios_below(records, k) for k in PASS_HAT_KS},
        "all_wordings_pass": metrics.all_wordings_pass(records),
        "variance_shares": metrics.variance_shares(records),
        "turn_stats": metrics.turn_stats(records),
        "outcome_labels": metrics.outcome_labels(records),
        "reward_distribution": {str(reward): count
                                for reward, count in metrics.reward_distribution(records).items()},
        "smallest_detectable_effect": _smallest_detectable_effect(records, seed),
    }


def pair_row(arm_a: str, arm_b: str, records_a: list[dict], records_b: list[dict],
             n_draws: int, seed: int, level: float) -> dict:
    """The comparison "b minus a": the difference of the two pass rates, family by family.

    `lo`, `hi`: paired bootstrap by family (percentile interval). `t_lo`, `t_hi`: t-interval of the same
    per-family differences, shown beside it."""
    rates_a = metrics.family_pass_rates(records_a)
    rates_b = metrics.family_pass_rates(records_b)
    if list(rates_a) != list(rates_b):
        only_one = sorted(set(rates_a) ^ set(rates_b))
        raise ValueError(f"{arm_a!r} and {arm_b!r} cannot be paired: these families have valid episodes in "
                         f"only one of them: {only_one}")
    a = np.array(list(rates_a.values()), dtype=float)
    b = np.array(list(rates_b.values()), dtype=float)
    boot = impl.get("paired_family_bootstrap")(a, b, n_draws, seed, level)
    t = metrics.t_interval(b - a, level)
    return {"a": arm_a, "b": arm_b, "n_families": int(a.size), "diff": float(boot["diff"]),
            "lo": float(boot["lo"]), "hi": float(boot["hi"]), "t_lo": t["lo"], "t_hi": t["hi"]}


def compare_arms(records_by_arm: dict[str, list[dict]], baseline_arm: str,
                 pairs: list[tuple[str, str]] | None = None, *, n_draws: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> dict:
    """Build the results table.

    records_by_arm   arm name -> all episode records of that arm (harness errors included)
    baseline_arm     the arm that is listed first and that every other arm must match in coverage
    pairs            the comparisons, as (a, b) for "b minus a"; default: every other arm against the baseline

    Raises ValueError unless all arms cover exactly the same scenarios with the same sample counts, and were
    run with the same text file, on the same scenario file (`data_sha256`; records that do not carry it
    are accepted, and the table's `notes` say so), with the same settings (`check_same_settings`) and
    scored by the same versions of verify and reward (`check_same_grader`). It also raises if two arms
    hold exactly the same records, and if the two arms of a pair do not have valid episodes in the same
    scenarios.
    The table names the conditions: `text_sha256`, `text_is_placeholder`, `splits`, and in each arm row the
    rule set, tool set, prompt and config hash; `notes` say where arms differ in condition.
    The same seed is used for every interval, so every arm and every pair resamples the same families."""
    if baseline_arm not in records_by_arm:
        raise ValueError(f"baseline arm {baseline_arm!r} is not in {list(records_by_arm)}")
    check_same_text(records_by_arm)
    check_same_grader(records_by_arm)
    arm_names = [baseline_arm] + [arm for arm in records_by_arm if arm != baseline_arm]
    in_order = {arm: records_by_arm[arm] for arm in arm_names}
    notes = check_same_data(in_order) + check_same_settings(in_order)
    coverage = check_same_coverage(records_by_arm, baseline_arm)
    check_arms_differ(in_order)
    if pairs is None:
        pairs = [(baseline_arm, arm) for arm in arm_names[1:]]
    for arm_a, arm_b in pairs:
        if arm_a not in records_by_arm or arm_b not in records_by_arm or arm_a == arm_b:
            raise ValueError(f"pair ({arm_a!r}, {arm_b!r}) needs two different arms from {arm_names}")
        check_same_valid_scenarios(arm_a, arm_b, records_by_arm[arm_a], records_by_arm[arm_b])
    notes += condition_notes(in_order, baseline_arm)
    baseline = records_by_arm[baseline_arm]
    return {
        "level": level, "n_draws": n_draws, "seed": seed, "baseline_arm": baseline_arm,
        "paired_family_bootstrap": impl.which("paired_family_bootstrap"),
        "text_sha256": baseline[0]["text_sha256"],
        "text_is_placeholder": any(r["text_is_placeholder"] for records in in_order.values()
                                   for r in records),
        "splits": sorted({r["split"] for r in baseline}),
        "n_families": len({family_id for family_id, _style in coverage}),
        "n_scenarios": len(coverage),
        "n_episodes_per_arm": sum(coverage.values()),
        "arms": [arm_row(arm, records_by_arm[arm], n_draws, seed, level) for arm in arm_names],
        "pairs": [pair_row(arm_a, arm_b, records_by_arm[arm_a], records_by_arm[arm_b], n_draws, seed, level)
                  for arm_a, arm_b in pairs],
        "notes": notes,
    }


def _percent(value: float | None) -> str:
    """0.4583 -> "45.8"; None -> "n/a"."""
    return "n/a" if value is None else f"{100.0 * value:.1f}"


def _signed_points(value: float) -> str:
    """0.125 -> "+12.5"; -0.03 -> "-3.0"."""
    return f"{100.0 * value:+.1f}"


def _effect_cell(effect: dict | None) -> str:
    """"3.7 (200 splits; one split alone 2.1 to 6.0)", in points; "n/a" without an estimate."""
    if effect is None:
        return "n/a"
    return (f"{_percent(effect['mde'])} ({effect['n_splits']} splits; one split alone "
            f"{_percent(effect['mde_lowest_split'])} to {_percent(effect['mde_highest_split'])})")


def _reliability_lines(table: dict) -> list[str]:
    """The conditions of the arms, then their reliability and behaviour numbers, then their rewards."""
    lines = ["", "| Arm | Split | Rule set | Tools | Prompt | Text | Config (start of its SHA-256) |",
             "|---|---|---|---|---|---|---|"]
    for row in table["arms"]:
        cells = [row["arm"], ", ".join(row["splits"]), row["ruleset"], row["toolset"], row["prompt"],
                 "PLACEHOLDER" if row["text_is_placeholder"] else table["text_sha256"][:12],
                 ", ".join(value[:12] for value in row["config_sha256"])]
        lines.append("| " + " | ".join(cells) + " |")
    if table["text_is_placeholder"]:
        lines += ["", "**PLACEHOLDER TEXT: these episodes were made with the placeholder text file. This "
                      "table is a trial of the code, not a result.**"]
    labels = list(metrics.OUTCOME_LABELS)
    lines += ["",
              f"| Arm | Pass@{PASS_AT_K} | " + " | ".join(f"Pass^{k}" for k in PASS_HAT_KS)
              + " | All wordings pass | Variance inside families due to wording (if sampling only) "
                "| Mean turns | Ended at the turn limit | " + " | ".join(labels)
              + " | Smallest detectable effect, points |",
              "|---|---|" + "---|" * len(PASS_HAT_KS) + "---|---|---|---|" + "---|" * len(labels) + "---|"]
    for row in table["arms"]:
        shares = row["variance_shares"]
        cells = [row["arm"], _percent(row["pass_at_8"])]
        cells += [_percent(row["pass_hat_k"][str(k)]) for k in PASS_HAT_KS]
        cells += [_percent(row["all_wordings_pass"]),
                  f"{_percent(shares['wording'])} ({_percent(shares['wording_if_sampling_only'])})",
                  f"{row['turn_stats']['mean_turns']:.2f}",
                  _percent(row["turn_stats"]["share_at_turn_limit"])]
        cells += [_percent(row["outcome_labels"][label]) for label in labels]
        cells.append(_effect_cell(row["smallest_detectable_effect"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Episodes per reward value (all valid episodes):"]
    for row in table["arms"]:
        counts = ", ".join(f"reward {reward}: {count}"
                           for reward, count in row["reward_distribution"].items())
        lines.append(f"- {row['arm']}: {counts}")
    lines += ["",
              "Malformed messages: assistant messages that are not exactly one valid tool call, as a share "
              "of all assistant messages. A well-formed call that a tool refused (for example an unknown "
              "id) is not counted.",
              "Smallest detectable effect: from the arm's own episodes (sampling noise only); a scenario "
              "needs 2 samples or more."]
    return lines


def to_markdown(table: dict) -> str:
    """The table of `compare_arms` as markdown: the arms, the comparisons, the conditions of the arms,
    their reliability and behaviour numbers, their rewards, and the notes if there are any.
    It computes nothing. All numbers are in percent or percentage points, but the mean turns."""
    level = _percent(table["level"]).removesuffix(".0")
    lines = [
        f"Families: {table['n_families']}. Scenarios: {table['n_scenarios']}. "
        f"Episodes per arm: {table['n_episodes_per_arm']}.",
        f"Intervals: {level}%, by resampling families, {table['n_draws']} draws, seed {table['seed']}. "
        f"Paired bootstrap: the {table['paired_family_bootstrap']} version of paired_family_bootstrap.",
        "",
        "| Arm | Episodes | Harness errors | Pass rate | Interval | " + " | ".join(OUTCOMES)
        + " | Seen styles | Held-out style | Groups with equal rewards | Groups with equal pass or fail "
          "| Malformed messages (% of assistant messages) |",
        "|---|---|---|---|---|" + "---|" * len(OUTCOMES) + "---|---|---|---|---|",
    ]
    for row in table["arms"]:
        cells = [row["arm"], str(row["n_episodes"]), str(row["n_harness_errors"]), _percent(row["pass_rate"]),
                 f"{_percent(row['pass_rate_lo'])} to {_percent(row['pass_rate_hi'])}"]
        cells += [_percent(row["pass_rate_by_outcome"][outcome]) for outcome in OUTCOMES]
        cells += [_percent(row["pass_rate_seen"]), _percent(row["pass_rate_held_out"]),
                  _percent(row["zero_variance_equal_reward"]), _percent(row["zero_variance_equal_passfail"]),
                  _percent(row["invalid_call_rate"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["",
              "| Comparison | Families | Difference in points | Paired bootstrap by family | t-interval |",
              "|---|---|---|---|---|"]
    for row in table["pairs"]:
        cells = [f"{row['b']} minus {row['a']}", str(row["n_families"]), _signed_points(row["diff"]),
                 f"{_signed_points(row['lo'])} to {_signed_points(row['hi'])}",
                 f"{_signed_points(row['t_lo'])} to {_signed_points(row['t_hi'])}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += _reliability_lines(table)
    if table["notes"]:
        lines += ["", "Notes:"] + [f"- {note}" for note in table["notes"]]
    return "\n".join(lines) + "\n"


def _read_jsonl(path: str) -> list[dict]:
    """The episode records of one file: one JSON object per line; empty lines are skipped."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _name_and_value(text: str, separator: str, what: str) -> tuple[str, str]:
    name, found, value = text.partition(separator)
    if not found or not name or not value:
        raise SystemExit(f"{what} must look like NAME{separator}VALUE, got {text!r}")
    return name, value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the results table from saved episode records.")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                        help="an arm and its records file (JSON lines); give it once per arm")
    parser.add_argument("--baseline", required=True, help="name of the baseline arm")
    parser.add_argument("--pair", action="append", default=None, metavar="A:B",
                        help="a comparison 'B minus A'; default: every other arm against the baseline")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.95)
    args = parser.parse_args(argv)

    records_by_arm: dict[str, list[dict]] = {}
    for item in args.arm:
        name, path = _name_and_value(item, "=", "--arm")
        if name in records_by_arm:
            raise SystemExit(f"--arm {name!r} is given twice")
        records_by_arm[name] = _read_jsonl(path)
    pairs = None if args.pair is None else [_name_and_value(item, ":", "--pair") for item in args.pair]
    table = compare_arms(records_by_arm, args.baseline, pairs, n_draws=args.draws, seed=args.seed,
                         level=args.level)
    Path(args.out_json).write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(to_markdown(table), encoding="utf-8")


if __name__ == "__main__":
    main()
