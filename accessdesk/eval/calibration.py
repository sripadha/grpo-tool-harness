"""The calibration report: what an untrained model does on throwaway families, before the freeze.

PLAN 5.3 has a calibration step: the untrained models are rolled out, and the numbers decide the prompt
wording, the world size, the token cap, which penalties stay, and which model is trained. This file computes
those numbers from saved episode records. It reports and never decides.

Rules that hold for every function in this file:
- The input is a list of the episode records of ONE run (BUILD_SPEC section 8). No function reads a file;
  `scripts/calibration_report.py` does the reading and the writing.
- A record whose `harness_error` is not null is left out of every number and is counted in section 1.
- Every metric that exists in `accessdesk/eval/metrics.py` is taken from there. The unit of a pass rate is
  the family: a family's pass rate is the mean over its records, and the pass rate of a set of records is
  the mean over its families. Where a number has another unit (the scenario, the episode, the assistant
  message), its docstring says so.
- Every share is a fraction from 0 to 1. The markdown shows shares in percent.
- A returned dictionary is plain JSON: string keys in a fixed order, no tuple, no numpy value.
- A section function needs at least one valid record; `run_report` checks this first.

Calibration numbers are not results: they are in no results table (PLAN 5.3).
"""
from __future__ import annotations

import numpy as np

from accessdesk import impl
from accessdesk.config import DEFAULT
from accessdesk.eval import metrics, tables
from accessdesk.types import OUTCOMES, PARSE_ERRORS, RESOLUTIONS

FLOOR = 0.25                  # PLAN 5.2: a policy that ignores the request's facts passes one outcome in four
BAND = (0.30, 0.65)           # PLAN 5.3: the target for the pass rate of the untrained model that is trained
TOKEN_CAP_FACTOR = 2          # PLAN 5.4: the cap is twice the 99th percentile of well-formed call lengths
TOKEN_CAP_MINIMUM = 128       # PLAN 5.4: and at least 128
TOKEN_PERCENTILE = 99
HARNESS_ERROR_GATE = 0.01     # PLAN 5.4: harness errors must stay below 1% of the episodes
CALIB_SPLIT = "calib"         # the throwaway families
EPS = 1e-9                    # two shares closer than this are equal (rounding of floating-point means)
SEEN, HELD_OUT = "seen", "held_out"
NOT_COMPARABLE = "NOT COMPARABLE"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def check_not_empty(name: str, records: list[dict]) -> None:
    """Raise ValueError if the run has no record, or no record without a harness error."""
    if not records:
        raise ValueError(f"run {name!r}: no episode record (the records file is empty)")
    if not metrics.valid_records(records):
        raise ValueError(f"run {name!r}: all {len(records)} records are harness errors, "
                         f"so there is no episode to report on")


def _share(count: int, total: int) -> float | None:
    """count / total, or None if total is 0."""
    return count / total if total else None


def _n(count: int, noun: str) -> str:
    """(1, "episode") -> "1 episode"; (3, "family") -> "3 families"."""
    if count == 1:
        return f"1 {noun}"
    return f"{count} {noun[:-1]}ies" if noun.endswith("y") else f"{count} {noun}s"


def _by_outcome(records: list[dict]) -> dict[str, list[dict]]:
    """outcome -> its valid records; the four outcomes in their fixed order, only those that are present."""
    groups: dict[str, list[dict]] = {}
    for r in metrics.valid_records(records):
        groups.setdefault(r["outcome"], []).append(r)
    order = [o for o in OUTCOMES if o in groups] + sorted(o for o in groups if o not in OUTCOMES)
    return {outcome: groups[outcome] for outcome in order}


def _samples_per_scenario(records: list[dict]) -> dict[tuple[str, int], int]:
    """(family_id, style) -> number of valid records. Scenarios in sorted order."""
    counts: dict[tuple[str, int], int] = {}
    for r in metrics.valid_records(records):
        key = (r["family_id"], r["style"])
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _size_table(sizes: list[int]) -> list[dict]:
    """[4, 4, 3] -> [{"samples": 3, "scenarios": 1}, {"samples": 4, "scenarios": 2}]."""
    return [{"samples": n, "scenarios": sizes.count(n)} for n in sorted(set(sizes))]


def _families(records: list[dict]) -> int:
    """Number of families with at least one valid record."""
    return len(metrics.family_pass_rates(records))


def resolve_config(records: list[dict], manifest: dict | None) -> tuple[dict | None, str]:
    """The run's config and where it was found: the run manifest, or the default config if every record
    carries the default config's hash, or (None, reason)."""
    if manifest and isinstance(manifest.get("config"), dict):
        return manifest["config"], "run_manifest.json"
    if {r["config_sha256"] for r in records} == {DEFAULT.sha256()}:
        return DEFAULT.to_dict(), "the default config (its hash is in every record)"
    return None, "not known: no run manifest, and the records do not carry the default config's hash"


# ---------------------------------------------------------------------------
# 1. Size and health
# ---------------------------------------------------------------------------
def size_and_health(name: str, records: list[dict], manifest: dict | None = None) -> dict:
    """How large the run is, what it was made with, and how many records are harness errors.

        n_records             all episode records of the run, harness errors included (unit: the episode)
        n_harness_errors      records with a harness error; they are in no other number of the report
        harness_error_share   n_harness_errors / n_records
        n_episodes            n_records - n_harness_errors: the episodes that every other number uses
        n_families            families with at least one valid episode
        n_families_by_outcome the same, per outcome
        n_scenarios           scenarios (family in one wording style) with at least one valid episode
        n_scenarios_without_valid_sample   scenarios of which every record is a harness error
        samples_per_scenario  valid episodes per scenario: `min`, `max`, `all_equal` (True if every scenario
                              has the same number) and `table` (for each number of samples: how many
                              scenarios have it)
        model, revision, adapter, text_file   from run_manifest.json; None if there is no manifest
        text_sha256           the hashes of the text file that the records carry (one, in a sound run)
        data_sha256           the hashes of the scenario files that the records carry (`data_sha256`; a
                              record may lack it, and then the list is empty)
        placeholder_text      True if any record, or the manifest, says that the placeholder text was used
        split, ruleset, toolset, prompt, config_sha256, impl   what the records carry; lists where a run
                              could hold more than one value

    Raises ValueError if the records mix rule sets, tool sets or prompts, or hold one sample twice."""
    tables.scenario_sample_counts(name, records)          # one condition, and no sample twice
    manifest = manifest or {}
    sizes = _samples_per_scenario(records)
    counts = list(sizes.values())
    run_scenarios = {(r["family_id"], r["style"]) for r in records}
    n_errors = metrics.harness_error_count(records)
    first = records[0]
    return {
        "n_records": len(records),
        "n_harness_errors": n_errors,
        "harness_error_share": n_errors / len(records),
        "n_episodes": len(records) - n_errors,
        "n_families": _families(records),
        "n_families_by_outcome": {outcome: _families(group)
                                  for outcome, group in _by_outcome(records).items()},
        "n_scenarios": len(sizes),
        "n_scenarios_without_valid_sample": len(run_scenarios) - len(sizes),
        "samples_per_scenario": {"min": min(counts), "max": max(counts),
                                 "all_equal": min(counts) == max(counts), "table": _size_table(counts)},
        "manifest_found": bool(manifest),
        "model": manifest.get("model"),
        "revision": manifest.get("revision"),
        "adapter": manifest.get("adapter"),
        "text_file": manifest.get("text_file"),
        "text_sha256": sorted({r["text_sha256"] for r in records}),
        "data_sha256": sorted({r["data_sha256"] for r in records if r.get("data_sha256")}),
        "placeholder_text": (any(r["text_is_placeholder"] for r in records)
                             or bool(manifest.get("text_is_placeholder"))),
        "split": sorted({r["split"] for r in records}),
        "ruleset": first["ruleset"], "toolset": first["toolset"], "prompt": first["prompt"],
        "config_sha256": sorted({r["config_sha256"] for r in records}),
        "impl": {key: sorted({r["impl"][key] for r in records}) for key in ("verify", "reward")},
    }


# ---------------------------------------------------------------------------
# 2. Pass rate
# ---------------------------------------------------------------------------
def _style_part(record: dict) -> str:
    return HELD_OUT if record["style"] == metrics.HELD_OUT_STYLE else SEEN


def _rates_by(records: list[dict], key: str) -> dict[str, dict]:
    """part -> {"pass_rate": metrics.pass_rate_by, "n_families": families with a valid episode in it}."""
    families: dict[str, set[str]] = {}
    for r in metrics.valid_records(records):
        part = _style_part(r) if key == "seen_held_out" else r[key]
        families.setdefault(str(part), set()).add(r["family_id"])
    return {str(part): {"pass_rate": rate, "n_families": len(families[str(part)])}
            for part, rate in metrics.pass_rate_by(records, key).items()}


def held_out_minus_seen(records: list[dict], n_draws: int = 10_000, seed: int = 0,
                        level: float = 0.95) -> dict | None:
    """The pass rate in the held-out style (4) minus the pass rate in the seen styles (1 to 3), by family.

    For each family: its pass rate over its held-out episodes minus its pass rate over its seen episodes;
    `diff` is the mean over families, and `lo`, `hi` are the paired bootstrap by family (the same
    families are drawn for both parts). Unit: the family. None if the two parts do not hold the same
    families, or if there are fewer than 2 families."""
    valid = metrics.valid_records(records)
    seen = [r for r in valid if _style_part(r) == SEEN]
    held = [r for r in valid if _style_part(r) == HELD_OUT]
    if not seen or not held:
        return None
    try:
        row = tables.pair_row(SEEN, HELD_OUT, seen, held, n_draws, seed, level)
    except ValueError:
        return None
    return {"diff": row["diff"], "lo": row["lo"], "hi": row["hi"], "n_families": row["n_families"]}


def pass_rates(records: list[dict], n_draws: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """The pass rate of the run, and of each part of it. Unit: the family, also inside each part.

        pass_rate       the mean over families of the family's share of passed episodes
        lo, hi          the ends of the `level` interval of pass_rate: families are drawn again with
                        replacement `n_draws` times, and lo and hi are the percentiles of the draws' means
        n_families      families behind pass_rate
        by_outcome      per outcome (grant, approval, already, inactive): pass rate over its families
        by_style        per wording style number: the family-weighted pass rate of that style (for each
                        family its share of passed episodes among its episodes of the style, then the mean
                        over families; not the share of the style's episodes that passed)
        seen_held_out   "seen": the styles 1 to 3 together; "held_out": style 4
        held_out_minus_seen   see `held_out_minus_seen`
    Each part also gives `n_families`, the families that have a valid episode in it."""
    rates = metrics.family_pass_rates(records)
    interval = metrics.family_bootstrap_ci(list(rates.values()), n_draws, seed, level)
    return {
        "pass_rate": metrics.pass_rate(records), "lo": interval["lo"], "hi": interval["hi"],
        "n_families": len(rates),
        "by_outcome": _rates_by(records, "outcome"),
        "by_style": _rates_by(records, "style"),
        "seen_held_out": _rates_by(records, "seen_held_out"),
        "held_out_minus_seen": held_out_minus_seen(records, n_draws, seed, level),
    }


# ---------------------------------------------------------------------------
# 3. The two checks of the plan
# ---------------------------------------------------------------------------
def plan_checks(rates: dict, n_families_by_outcome: dict[str, int]) -> dict:
    """The floor check and the band check of PLAN 5.3, from the output of `pass_rates`, and the room left.

        floor.threshold      0.25: the pass rate of the best policy that ignores the request's facts, when
                             the four outcomes have the same number of families (PLAN 5.2)
        floor.interval_lo    the lower end of the interval of the pass rate (`pass_rates`)
        floor.passed         True if interval_lo is above the threshold (equal is not above)
        floor.best_constant_policy   the largest share of this run's families that have one outcome: what
                             the best policy that always gives the same resolution passes on THESE families.
                             It is 0.25 only if the outcomes are balanced
        band.lo, band.hi     0.30 and 0.65
        band.passed          True if band.lo <= pass_rate <= band.hi (the ends are inside)
        band.position        "below", "inside" or "above"
        room.all             1 - pass_rate: the most that training can add, in pass rate
        room.by_outcome      1 - the outcome's pass rate
    Unit: the family."""
    rate, lo = rates["pass_rate"], rates["lo"]
    position = "below" if rate < BAND[0] - EPS else "above" if rate > BAND[1] + EPS else "inside"
    return {
        "floor": {"threshold": FLOOR, "interval_lo": lo, "passed": bool(lo > FLOOR + EPS),
                  "best_constant_policy": max(n_families_by_outcome.values())
                  / sum(n_families_by_outcome.values())},
        "band": {"lo": BAND[0], "hi": BAND[1], "pass_rate": rate, "passed": position == "inside",
                 "position": position},
        "room": {"all": 1.0 - rate,
                 "by_outcome": {outcome: 1.0 - part["pass_rate"]
                                for outcome, part in rates["by_outcome"].items()}},
    }


# ---------------------------------------------------------------------------
# 4. Ties
# ---------------------------------------------------------------------------
def ties(records: list[dict]) -> dict:
    """How many scenarios give GRPO no signal, and the pass-rate profile of the families.

        all, by_outcome     `metrics.zero_variance_share`, over all scenarios and per outcome. Unit: the
                            scenario; only scenarios with at least 2 valid samples are counted.
                              equal_reward           share in which every sample has exactly the same reward
                              equal_reward_all_pass  ... and every sample passed
                              equal_reward_all_fail  ... and every sample failed
                              equal_passfail         share in which all passed or all failed (the rewards
                                                     of passes may differ by penalties)
        samples_per_scenario  for each number of valid samples: how many of the COUNTED scenarios have it.
                            The shares hold for these numbers of samples only: with more samples per
                            scenario a tie is less likely.
        n_scenarios_left_out  scenarios with fewer than 2 valid samples
        family_profile      `always_pass`: families whose every valid episode passed; `always_fail`: none
                            passed; `mixed`: the rest. Unit: the family; a count."""
    sizes = list(_samples_per_scenario(records).values())
    rates = list(metrics.family_pass_rates(records).values())
    always_pass = sum(1 for rate in rates if rate == 1.0)
    always_fail = sum(1 for rate in rates if rate == 0.0)
    return {
        "all": metrics.zero_variance_share(records),
        "by_outcome": {outcome: metrics.zero_variance_share(group)
                       for outcome, group in _by_outcome(records).items()},
        "samples_per_scenario": _size_table([n for n in sizes if n >= 2]),
        "n_scenarios_left_out": sum(1 for n in sizes if n < 2),
        "family_profile": {"n_families": len(rates), "always_pass": always_pass, "always_fail": always_fail,
                           "mixed": len(rates) - always_pass - always_fail},
    }


# ---------------------------------------------------------------------------
# 5. Call format
# ---------------------------------------------------------------------------
def _counts(values: list[str], known: tuple[str, ...] = ()) -> dict[str, int]:
    """value -> how often it occurs; the known values first in their order, then the others sorted."""
    found: dict[str, int] = {}
    for value in values:
        found[value] = found.get(value, 0) + 1
    order = [v for v in known if v in found] + sorted(v for v in found if v not in known)
    return {value: found[value] for value in order}


def call_format(records: list[dict]) -> dict:
    """How well the model keeps the tool-call format. Unit: the assistant message (one turn).

        n_messages         all assistant messages of the valid episodes
        n_malformed        messages that were not exactly one valid tool call (the episodes' counters)
        invalid_call_rate  n_malformed / n_messages (`metrics.invalid_call_rate`)
        parse_errors       for each reason of the parser: the number of messages rejected for it
        n_tool_errors      well-formed calls that the tool answered with an error (not malformed)
        tool_errors        for each tool error: the number of such calls
        n_truncated        messages that were cut off by the token cap (they are also malformed)
        truncated_share    n_truncated / n_messages
        counters_agree     True if the episodes' counters (n_turns, n_malformed, n_tool_errors) equal the
                           counts over the saved turns"""
    valid = metrics.valid_records(records)
    turns = [turn for r in valid for turn in r["turns"]]
    parse_errors = _counts([t["parse_error"] for t in turns if not t["parse_ok"]], PARSE_ERRORS)
    tool_errors = _counts([t["tool_error"] for t in turns if t["tool_error"] is not None])
    n_messages = sum(r["n_turns"] for r in valid)
    n_malformed = sum(r["n_malformed"] for r in valid)
    n_tool_errors = sum(r["n_tool_errors"] for r in valid)
    n_truncated = sum(1 for t in turns if t["truncated"])
    return {
        "n_messages": n_messages,
        "n_malformed": n_malformed,
        "invalid_call_rate": metrics.invalid_call_rate(records),
        "parse_errors": parse_errors,
        "n_tool_errors": n_tool_errors,
        "tool_errors": tool_errors,
        "n_truncated": n_truncated,
        "truncated_share": n_truncated / n_messages,
        "counters_agree": (len(turns) == n_messages and sum(parse_errors.values()) == n_malformed
                           and sum(tool_errors.values()) == n_tool_errors),
    }


# ---------------------------------------------------------------------------
# 6. Turns and tokens
# ---------------------------------------------------------------------------
def percentile_nearest_rank(values: list[int], percent: int) -> int:
    """The smallest of the values that is at least as large as `percent`% of them (nearest-rank percentile).

    With n values in rising order it is value number ceil(percent * n / 100), counted from 1: the 99th of
    100 values, the 198th of 200, and the largest value when n is below 100. It is always one of the
    values, never a number between two of them."""
    if not values or not 0 < percent <= 100:
        raise ValueError("need at least one value and 0 < percent <= 100")
    rank = (percent * len(values) + 99) // 100
    return sorted(values)[rank - 1]


def token_cap_rule(p99: int) -> int:
    """PLAN 5.4: the token cap per turn is twice the 99th percentile of well-formed call lengths, and at
    least 128. Unit: new tokens of one assistant message."""
    return max(TOKEN_CAP_MINIMUM, TOKEN_CAP_FACTOR * p99)


def call_tokens(records: list[dict], cap_in_force: int | None = None) -> dict | None:
    """The length of the well-formed calls, in new tokens, and the token cap that the plan's rule gives.

    A well-formed call is a turn that the parser accepted (`parse_ok`), whether or not the tool then
    answered with an error. Its length is the turn's `n_tokens` as the harness stored it: every token that
    the model wrote in that message, the end-of-turn token included; the cap counts the same tokens.
    Unit: the assistant message.
        n_calls        well-formed calls with a stored length
        n_calls_without_length   well-formed calls whose n_tokens is null (scripted runs)
        median         the middle length (the mean of the two middle ones for an even number)
        p99            the 99th percentile, nearest rank (`percentile_nearest_rank`)
        max            the longest well-formed call
        cap_by_rule    `token_cap_rule(p99)`
        cap_in_force   the cap of this run (config `max_new_tokens`), or None if it is not known
        n_truncated    messages cut off at the cap in force. They are not well-formed, so their lengths are
                       in none of the numbers above: a call that would have been longer than the cap in
                       force cannot be seen here.
    None if no well-formed call has a stored length."""
    valid = metrics.valid_records(records)
    calls = [t for r in valid for t in r["turns"] if t["parse_ok"]]
    lengths = [t["n_tokens"] for t in calls if t["n_tokens"] is not None]
    if not lengths:
        return None
    p99 = percentile_nearest_rank(lengths, TOKEN_PERCENTILE)
    return {
        "n_calls": len(lengths),
        "n_calls_without_length": len(calls) - len(lengths),
        "median": float(np.median(lengths)),
        "p99": p99,
        "max": max(lengths),
        "cap_by_rule": token_cap_rule(p99),
        "cap_in_force": cap_in_force,
        "n_truncated": sum(1 for r in valid for t in r["turns"] if t["truncated"]),
    }


def over_budget(record: dict) -> bool:
    """True if the episode used more turns than its turn budget (the shortest correct path plus one)."""
    return record["n_turns"] > record["turn_budget"]


def turns_and_tokens(records: list[dict], cap_in_force: int | None = None) -> dict:
    """Turns per episode, and the tokens of the well-formed calls. Unit: the episode, except `tokens`.

        mean_turns             mean number of assistant messages per episode (`metrics.turn_stats`)
        mean_turns_by_outcome  the same, over the episodes of each outcome
        share_at_turn_limit    share of episodes that used up the turn limit without closing the request
                               (`metrics.turn_stats`)
        n_over_budget, share_over_budget   episodes with more turns than their turn budget
        tokens                 see `call_tokens`"""
    valid = metrics.valid_records(records)
    stats = metrics.turn_stats(records)
    n_over = sum(1 for r in valid if over_budget(r))
    return {
        "n_episodes": stats["n_episodes"],
        "mean_turns": stats["mean_turns"],
        "mean_turns_by_outcome": {outcome: metrics.turn_stats(group)["mean_turns"]
                                  for outcome, group in _by_outcome(records).items()},
        "share_at_turn_limit": stats["share_at_turn_limit"],
        "n_over_budget": n_over,
        "share_over_budget": n_over / len(valid),
        "tokens": call_tokens(records, cap_in_force),
    }


# ---------------------------------------------------------------------------
# 7. Penalties
# ---------------------------------------------------------------------------
def _group(records: list[dict]) -> dict:
    """n_episodes, n_passed, n_families and the pass rate (mean over the group's families) of episodes."""
    return {"n_episodes": len(records),
            "n_passed": sum(1 for r in records if r["passed"]),
            "n_families": _families(records) if records else 0,
            "pass_rate": metrics.pass_rate(records) if records else None}


CLOSED_ONLY = "Among the episodes that closed the request"


def _penalty_sentence(with_text: str, without_text: str, with_group: dict, without_group: dict,
                      difference: float | None, closed_only: bool = False) -> str:
    """One plain sentence on what the two groups show. It states the numbers and does not decide.

    It gives the two units apart: first the episodes of each group and how many of them passed, then the
    pass rates as a mean over families (the unit of `pass_rate` and of `difference`). The two differ when
    the families do not have the same number of episodes in a group."""
    if difference is None:
        missing, other = (with_text, without_group) if with_group["n_episodes"] == 0 else \
            (without_text, with_group)
        opening = f"{CLOSED_ONLY} there is" if closed_only else "There is"
        return (f"{opening} no episode {missing}, so this run does not show whether this penalty separates "
                f"good from bad episodes (the other group holds {_n(other['n_episodes'], 'episode')}).")
    direction = "lower" if difference < 0 else "higher"
    if abs(difference) < EPS:
        compare = "the two pass rates are the same"
    else:
        compare = f"the pass rate is {100.0 * abs(difference):.1f} points {direction} {with_text}"
    opening = f"{CLOSED_ONLY}: of" if closed_only else "Of"
    return (f"{opening} the {_n(with_group['n_episodes'], 'episode')} {with_text} {with_group['n_passed']} "
            f"passed, and of the {_n(without_group['n_episodes'], 'episode')} {without_text} "
            f"{without_group['n_passed']} passed. As a mean over families ({with_group['n_families']} and "
            f"{without_group['n_families']} families) the pass rates are "
            f"{100.0 * with_group['pass_rate']:.1f}% and {100.0 * without_group['pass_rate']:.1f}%: "
            f"{compare}.")


def _two_groups(records: list[dict], carries, with_text: str, without_text: str,
                closed_only: bool = False) -> dict:
    with_group = _group([r for r in records if carries(r)])
    without_group = _group([r for r in records if not carries(r)])
    difference = None
    if with_group["n_episodes"] and without_group["n_episodes"]:
        difference = with_group["pass_rate"] - without_group["pass_rate"]
    return {"with": with_group, "without": without_group, "difference": difference,
            "sentence": _penalty_sentence(with_text, without_text, with_group, without_group, difference,
                                          closed_only)}


def _penalty(valid: list[dict], carries, with_text: str, without_text: str) -> dict:
    """The comparison over all episodes, the same comparison over the episodes that closed the request, and
    the episodes that never closed it (see `penalties`)."""
    result = _two_groups(valid, carries, with_text, without_text)
    never = [r for r in valid if not r["closed"]]
    n_with = sum(1 for r in never if carries(r))
    result["closed_only"] = _two_groups([r for r in valid if r["closed"]], carries, with_text, without_text,
                                        closed_only=True)
    result["never_closed"] = {"n_episodes": len(never), "n_with": n_with,
                              "n_passed": sum(1 for r in never if r["passed"])}
    result["caution"] = None
    if n_with:
        result["caution"] = (f"{n_with} of the {_n(result['with']['n_episodes'], 'episode')} {with_text} "
                             f"never closed the request, and an episode that never closes cannot pass: to "
                             f"that extent the difference is true by construction, whatever the penalty "
                             f"is.")
    return result


def penalties(records: list[dict]) -> dict:
    """Each penalty of the reward against pass or fail (PLAN 5.2). Nothing is decided here.

    `malformed`: the episodes with at least one malformed message (`with`) against the episodes with none
    (`without`). `turn_budget`: the episodes with more turns than their budget (`with`) against the
    episodes inside it (`without`). For each group:
        n_episodes   episodes in the group (unit: the episode)
        n_passed     of these, the episodes that passed: in the `with` group these are the episodes whose
                     reward the penalty lowered
        n_families   families with at least one episode in the group
        pass_rate    the mean over those families of the family's share of passed episodes in the group
                     (unit: the family); None for an empty group. The two groups need not hold the same
                     families, so part of a difference can come from which families are in each.
    `difference` = pass_rate of `with` minus pass_rate of `without`; None if a group is empty.
    `sentence` says in plain words what these numbers show.

    An episode that never closes the request uses up the turn limit and fails. So it is over its turn
    budget whenever the limit is above the budget, and it has the most turns in which a message can be
    malformed: over all episodes, both `with` groups hold these failures by construction. Therefore:
        closed_only    the same four entries (`with`, `without`, `difference`, `sentence`) over the episodes
                       that closed the request: here a penalty can be set against a real pass or fail
        never_closed   `n_episodes`: episodes that never closed the request; `n_with`: of these, the ones
                       in the `with` group; `n_passed`: of these, the ones that passed (0 in a sound run)
        caution        a sentence that says how many episodes of the `with` group never closed; None if
                       there is none

        n_both                 episodes that are in both `with` groups (a malformed message also uses a turn)
        rewards_of_passes      among the passing episodes: for each reward value, the number of episodes
        n_passes               the passing episodes"""
    valid = metrics.valid_records(records)
    passes = [r for r in valid if r["passed"]]
    return {
        "malformed": _penalty(valid, lambda r: r["n_malformed"] >= 1,
                              "with a malformed call", "with no malformed call"),
        "turn_budget": _penalty(valid, over_budget, "over the turn budget", "inside the turn budget"),
        "n_both": sum(1 for r in valid if r["n_malformed"] >= 1 and over_budget(r)),
        "n_passes": len(passes),
        "rewards_of_passes": [{"reward": reward, "episodes": count}
                              for reward, count in sorted(metrics.reward_distribution(passes).items(),
                                                          reverse=True)],
    }


# ---------------------------------------------------------------------------
# 8. What the model does when it fails
# ---------------------------------------------------------------------------
def families_problem(records: list[dict], families: list[dict]) -> str | None:
    """Why these family records cannot be the ones the run was made from, or None if they fit.

    They fit if every valid record's family is there with the same outcome and the same turn budget, and if
    the verifier, given the record's final state and the family's expected state, returns the record's
    `passed`. The verifier is reached through `accessdesk.impl.get("verify")`."""
    by_id = {family["family_id"]: family for family in families}
    verify = impl.get("verify")
    n_disagree = 0
    for r in metrics.valid_records(records):
        family = by_id.get(r["family_id"])
        if family is None:
            return f"family {r['family_id']} of the run is not in the scenario files"
        if r["ruleset"] not in family["expected"]:
            return f"family {r['family_id']} has no expected state for the rule set {r['ruleset']!r}"
        if family["outcome"] != r["outcome"] or family["turn_budget"][r["ruleset"]] != r["turn_budget"]:
            return (f"family {r['family_id']} differs between the run and the scenario files "
                    f"(outcome or turn budget)")
        if bool(verify(r["final_state"], family["expected"][r["ruleset"]])) != r["passed"]:
            n_disagree += 1
    if n_disagree:
        return (f"the expected states of the scenario files do not give the run's pass or fail for "
                f"{n_disagree} episodes, so they are not the families (or not the verifier) of the run")
    return None


def _pairs(rows: list) -> set[tuple]:
    return {tuple(row) for row in rows}


def has_unexpected_grant(record: dict, expected: dict) -> bool:
    """True if the episode's final state holds a grant record (person, role) that is not one of the grant
    records of the expected state: a role was given that should not have been given. A second copy of an
    expected grant is not counted here (it fails the episode, but gives nothing new)."""
    return bool(_pairs(record["final_state"]["grants"]) - _pairs(expected["grants"]))


def failures(records: list[dict], families: list[dict] | None = None) -> dict:
    """What the episodes did, and for the failed ones what went wrong. Unit: the episode.

        labels, label_counts   each episode has exactly one label (`metrics.episode_label`): closed_correctly,
                               closed_wrongly, never_closed, malformed_only. `labels` are shares of the valid
                               episodes (`metrics.outcome_labels`), `label_counts` the numbers of episodes.
        closed_wrongly         for the episodes that closed the request and failed: `n`, `n_by_outcome`
                               (expected outcome of the family -> number of these episodes) and `table`:
                               expected outcome of the family -> resolution that the episode wrote on the
                               request -> number of episodes
    With the family records (they hold the correct end state); else these are None and `missing` says why:
        correct_resolution     outcome -> the resolution of the correct end state
        wrong_person           `n_closed`: episodes that closed the request; `n`: of these, the episodes
                               that closed it with the id of another person than the one meant;
                               `share_of_closed` = n / n_closed
        unexpected_grant       `n`: episodes with at least one grant record that the expected state does
                               not hold (`has_unexpected_grant`): the unsafe mistake;
                               `share` = n / all valid episodes; `by_outcome`: n per expected outcome"""
    valid = metrics.valid_records(records)
    labels = [metrics.episode_label(r) for r in valid]
    wrong = [r for r, label in zip(valid, labels) if label == "closed_wrongly"]
    table: dict[str, dict[str, int]] = {}
    n_by_outcome: dict[str, int] = {}
    for outcome, group in _by_outcome(wrong).items():
        table[outcome] = _counts([str(r["final_state"]["request"]["resolution"]) for r in group], RESOLUTIONS)
        n_by_outcome[outcome] = len(group)
    result = {
        "n_episodes": len(valid),
        "labels": metrics.outcome_labels(records),
        "label_counts": {label: labels.count(label) for label in metrics.OUTCOME_LABELS},
        "closed_wrongly": {"n": len(wrong), "n_by_outcome": n_by_outcome, "table": table},
        "correct_resolution": None, "wrong_person": None, "unexpected_grant": None, "missing": None,
    }
    if families is None:
        result["missing"] = "the family records were not given"
        return result
    problem = families_problem(records, families)
    if problem is not None:
        result["missing"] = problem
        return result
    by_id = {family["family_id"]: family for family in families}
    expected = [by_id[r["family_id"]]["expected"][r["ruleset"]] for r in valid]
    closed = [(r, state) for r, state in zip(valid, expected) if r["closed"]]
    n_wrong_person = sum(1 for r, state in closed
                         if r["final_state"]["request"]["person_id"] != state["request"]["person_id"])
    unsafe = [r for r, state in zip(valid, expected) if has_unexpected_grant(r, state)]
    correct = {r["outcome"]: state["request"]["resolution"] for r, state in zip(valid, expected)}
    result["correct_resolution"] = {outcome: correct[outcome] for outcome in _by_outcome(records)}
    result["wrong_person"] = {"n_closed": len(closed), "n": n_wrong_person,
                              "share_of_closed": _share(n_wrong_person, len(closed))}
    result["unexpected_grant"] = {"n": len(unsafe), "share": len(unsafe) / len(valid),
                                  "by_outcome": {outcome: sum(1 for r in unsafe if r["outcome"] == outcome)
                                                 for outcome in _by_outcome(records)}}
    return result


# ---------------------------------------------------------------------------
# 9. Supply for the supervised stage
# ---------------------------------------------------------------------------
def sft_supply(records: list[dict], pass_reward: int = DEFAULT.pass_reward) -> dict:
    """How many clean passes the run gives: an upper bound on the episodes that the supervised stage can
    keep (PLAN 5.5).

    A clean pass is an episode with the full reward `pass_reward` (100): it passed with no malformed
    message and inside its turn budget.
    It is an upper bound, not the supply itself. The selection (`select_sft_records` in
    accessdesk/train/sft.py) also drops every attempt with a refused call: a well-formed call that the
    tool answered with an error. Such an attempt can still have the full reward, and this function counts
    it. So what the selection keeps without `allow_penalised` (its `n_candidates`, `n_chosen` and
    `n_families_covered`) is equal to `n_clean`, `n_scenarios_with_clean_pass` and
    `n_families_with_clean_pass` here, or smaller.
        n_episodes, n_clean, share_clean   valid episodes, clean passes among them, n_clean / n_episodes
                                           (unit: the episode; every episode counts the same)
        by_outcome                         the same three numbers over the episodes of each outcome
        n_scenarios, n_scenarios_with_clean_pass   scenarios with a valid episode, and those of them with
                                           at least one clean pass (unit: the scenario)
        n_families, n_families_with_clean_pass     the same for families"""
    valid = metrics.valid_records(records)
    clean = [r for r in valid if r["reward"] == pass_reward]

    def part(episodes: list[dict]) -> dict:
        n_clean = sum(1 for r in episodes if r["reward"] == pass_reward)
        return {"n_episodes": len(episodes), "n_clean": n_clean, "share_clean": n_clean / len(episodes)}

    return {
        "pass_reward": pass_reward,
        **part(valid),
        "by_outcome": {outcome: part(group) for outcome, group in _by_outcome(records).items()},
        "n_scenarios": len(_samples_per_scenario(valid)),
        "n_scenarios_with_clean_pass": len(_samples_per_scenario(clean)),
        "n_families": _families(valid),
        "n_families_with_clean_pass": len({r["family_id"] for r in clean}),
    }


# ---------------------------------------------------------------------------
# One run, and several runs side by side
# ---------------------------------------------------------------------------
def _warnings(report: dict, manifest: dict | None) -> list[str]:
    """What a reader must know before trusting the numbers of one run. Plain sentences, in a fixed order."""
    size, calls, floor = report["size"], report["call_format"], report["checks"]["floor"]
    found: list[str] = []
    if size["placeholder_text"]:
        found.append("The placeholder text was used: these numbers say nothing about the final text.")
    if size["split"] != [CALIB_SPLIT]:
        found.append(f"The run is on the split {', '.join(size['split'])}, not on the throwaway "
                     f"calibration families: nothing may be tuned on dev or test families.")
    if manifest and manifest.get("complete") is False:
        found.append("The run manifest says that the run did not finish.")
    if len(size["text_sha256"]) > 1 or len(size["config_sha256"]) > 1:
        found.append("The records were made with more than one text file or config.")
    if manifest and manifest.get("text_sha256") not in (None, *size["text_sha256"]):
        found.append("The text hash of the run manifest is not the one in the records.")
    if size["harness_error_share"] >= HARNESS_ERROR_GATE:
        found.append(f"Harness errors are {100.0 * size['harness_error_share']:.1f}% of the records; the "
                     f"plan's gate before any training is below {100.0 * HARNESS_ERROR_GATE:.0f}%.")
    if not size["samples_per_scenario"]["all_equal"]:
        found.append("The scenarios do not all have the same number of valid samples "
                     f"({size['samples_per_scenario']['min']} to {size['samples_per_scenario']['max']}).")
    if abs(floor["best_constant_policy"] - FLOOR) > EPS:
        found.append(f"The outcomes are not balanced: the best constant policy passes "
                     f"{100.0 * floor['best_constant_policy']:.1f}% of these families, so the floor of "
                     f"{100.0 * FLOOR:.0f}% is too low for this run.")
    if not calls["counters_agree"]:
        found.append("The counters of the episodes do not agree with their saved turns.")
    if report["failures"]["missing"] is not None:
        found.append(f"Section 8 is incomplete (no wrong-person count, no unexpected-grant count): "
                     f"{report['failures']['missing']}.")
    if report["turns"]["tokens"] is None:
        found.append("No well-formed call has a stored token count: section 6 has no token numbers.")
    return found


def run_report(name: str, records: list[dict], *, families: list[dict] | None = None,
               manifest: dict | None = None, scenarios_note: str | None = None, source: str | None = None,
               n_draws: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """The nine sections for one run.

    records         every episode record of the run, harness errors included
    families        the family records of the run's split (they hold the correct end state), or None
    manifest        the content of run_manifest.json, or None
    scenarios_note  why `families` is None (shown in the report)
    source          where the records were read from (shown in the report)
    Raises ValueError for a run with no valid episode."""
    check_not_empty(name, records)
    config, config_source = resolve_config(records, manifest)
    size = size_and_health(name, records, manifest)
    rates = pass_rates(records, n_draws, seed, level)
    fails = failures(records, families)
    if families is None and scenarios_note:
        fails["missing"] = scenarios_note
    report = {
        "name": name, "source": source,
        "config": config, "config_source": config_source,
        "size": size,
        "pass": rates,
        "checks": plan_checks(rates, size["n_families_by_outcome"]),
        "ties": ties(records),
        "call_format": call_format(records),
        "turns": turns_and_tokens(records, config["max_new_tokens"] if config else None),
        "penalties": penalties(records),
        "failures": fails,
        "sft_supply": sft_supply(records, config["pass_reward"] if config else DEFAULT.pass_reward),
    }
    report["warnings"] = _warnings(report, manifest)
    return report


def _config_difference(first: dict | None, other: dict | None) -> str:
    """Which settings differ between two configs, with both values; or that this is not known."""
    if first is None or other is None:
        return "which settings differ is not known, because a run has no recorded config"
    names = [name for name in sorted(set(first) | set(other)) if first.get(name) != other.get(name)]
    if not names:
        return "the recorded configs show no setting that differs"
    return ", ".join(f"{name} ({first.get(name)} against {other.get(name)})" for name in names)


def _scenario_set(records: list[dict]) -> set[tuple[str, int]]:
    return {(r["family_id"], r["style"]) for r in records}


def _compare(first: str, other: str, records_by_run: dict[str, list[dict]], reports: dict[str, dict],
             n_draws: int, seed: int, level: float) -> tuple[dict | None, list[str]]:
    """The paired difference `other` minus `first` if both cover the same scenarios, and the warnings."""
    a, b = reports[first]["size"], reports[other]["size"]
    set_a, set_b = _scenario_set(records_by_run[first]), _scenario_set(records_by_run[other])
    notes: list[str] = []
    if set_a != set_b:
        notes.append(f"{NOT_COMPARABLE}: the runs {first} and {other} do not cover the same scenarios "
                     f"({len(set_a - set_b)} only in {first}, {len(set_b - set_a)} only in {other}, "
                     f"{len(set_a & set_b)} in both). Their numbers stand side by side, and no difference "
                     f"between them is given.")
    different_data = bool(a["data_sha256"] and b["data_sha256"] and a["data_sha256"] != b["data_sha256"])
    if different_data:
        notes.append(f"{NOT_COMPARABLE}: the runs {first} and {other} were made from different scenario "
                     f"files, so a family id does not name the same family in both. No difference between "
                     f"them is given.")
    if a["text_sha256"] != b["text_sha256"]:
        notes.append(f"The runs {first} and {other} were made with different text files: a family id "
                     f"then names the same world seed and outcome, but not the same words.")
    if (a["ruleset"], a["toolset"], a["prompt"]) != (b["ruleset"], b["toolset"], b["prompt"]):
        notes.append(f"The runs {first} and {other} differ in rule set, tool set or prompt.")
    if a["config_sha256"] != b["config_sha256"]:
        notes.append(f"The runs {first} and {other} were made with different configs: "
                     f"{_config_difference(reports[first]['config'], reports[other]['config'])}.")
    if a["samples_per_scenario"]["table"] != b["samples_per_scenario"]["table"]:
        notes.append(f"The runs {first} and {other} do not have the same numbers of samples per scenario: "
                     f"their tie shares cannot be compared.")
    if set_a != set_b or different_data:
        return None, notes
    try:
        row = tables.pair_row(first, other, records_by_run[first], records_by_run[other], n_draws, seed,
                              level)
    except ValueError as error:
        return None, notes + [f"No paired difference for {other} minus {first}: {error}."]
    return row, notes


def side_by_side(reports: dict[str, dict], records_by_run: dict[str, list[dict]], n_draws: int = 10_000,
                 seed: int = 0, level: float = 0.95) -> dict:
    """The runs next to each other, each of the other runs against the first one.

        runs          the run names, in the given order; the numbers of each are in its own report
        comparable    True if every run covers exactly the scenarios (family and style) of the first run,
                      and no two runs name different scenario files in their records
        warnings      why the runs are not comparable, or what differs between them
        differences   for each other run that covers the same scenarios as the first: the pass rate of
                      the other run minus that of the first, family by family (`tables.pair_row`): `diff`,
                      `lo` and `hi` from the paired bootstrap by family, `t_lo` and `t_hi` from the
                      t-interval of the per-family differences, `n_families`. Unit: the family."""
    names = list(reports)
    warnings: list[str] = []
    differences: list[dict] = []
    for other in names[1:]:
        row, notes = _compare(names[0], other, records_by_run, reports, n_draws, seed, level)
        warnings += notes
        if row is not None:
            differences.append(row)
    comparable = not any(warning.startswith(NOT_COMPARABLE) for warning in warnings)
    return {"runs": names, "comparable": comparable, "warnings": warnings, "differences": differences}


def build_report(records_by_run: dict[str, list[dict]], *,
                 families_by_run: dict[str, list[dict]] | None = None,
                 manifests: dict[str, dict] | None = None, scenarios_notes: dict[str, str] | None = None,
                 sources: dict[str, str] | None = None, n_draws: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> dict:
    """The whole report: every run, then the runs side by side. Everything `to_markdown` shows is in it.

    records_by_run   run name -> all its episode records; the first run is the one the others are set against
    families_by_run  run name -> the family records of its split; a run that is absent gets no number that
                     needs the correct end state, and `scenarios_notes[name]` says why
    manifests        run name -> the content of its run_manifest.json
    Raises ValueError if there is no run, or a run has no valid episode."""
    if not records_by_run:
        raise ValueError("no run was given")
    families_by_run, manifests = families_by_run or {}, manifests or {}
    scenarios_notes, sources = scenarios_notes or {}, sources or {}
    reports = {name: run_report(name, records, families=families_by_run.get(name),
                                manifest=manifests.get(name), scenarios_note=scenarios_notes.get(name),
                                source=sources.get(name), n_draws=n_draws, seed=seed, level=level)
               for name, records in records_by_run.items()}
    return {
        "title": "Calibration report",
        "placeholder_text_runs": [name for name, report in reports.items()
                                  if report["size"]["placeholder_text"]],
        "level": level, "n_draws": n_draws, "seed": seed,
        "paired_family_bootstrap": impl.which("paired_family_bootstrap"),
        "seen_styles": [1, 2, 3], "held_out_style": metrics.HELD_OUT_STYLE,
        "token_cap_rule": {"percentile": TOKEN_PERCENTILE, "factor": TOKEN_CAP_FACTOR,
                           "minimum": TOKEN_CAP_MINIMUM},
        "runs": reports,
        "side_by_side": side_by_side(reports, records_by_run, n_draws, seed, level),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    """0.4583 -> "45.8%"; None -> "n/a"."""
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _points(value: float | None) -> str:
    """0.125 -> "+12.5"; None -> "n/a"."""
    return "n/a" if value is None else f"{100.0 * value:+.1f}"


def _level(report: dict) -> str:
    return f"{100.0 * report['level']:g}%"


def _yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows] + [""]


def _sizes_text(table: list[dict]) -> str:
    """"4 samples in 32 scenarios", joined with commas."""
    return ", ".join(f"{_n(row['samples'], 'sample')} in {_n(row['scenarios'], 'scenario')}" for row in table)


def _named(counts: dict[str, int]) -> str:
    """{"no_call": 3, "truncated": 1} -> "no_call 3, truncated 1"; "none" if empty."""
    return ", ".join(f"{key} {count}" for key, count in counts.items()) or "none"


def _md_size(run: dict) -> list[str]:
    size = run["size"]
    samples = size["samples_per_scenario"]
    if size["manifest_found"]:
        model = f"{size['model']} (revision {size['revision']}, adapter {size['adapter'] or 'none'})"
        text_file = size["text_file"]
    else:
        model = text_file = "not recorded (no run_manifest.json beside the records)"
    equal = (f"{samples['min']} in every scenario" if samples["all_equal"] else
             f"NOT the same in every scenario: {_sizes_text(samples['table'])}")
    return [
        "### 1. Size and health", "",
        f"- Records file: {run['source'] or 'not given'}",
        f"- Model: {model}",
        f"- Text file: {text_file}; SHA-256 {', '.join(size['text_sha256'])}; "
        f"placeholder text: {'yes' if size['placeholder_text'] else 'no'}",
        f"- Split: {', '.join(size['split'])}. Rule set {size['ruleset']}, tool set {size['toolset']}, "
        f"prompt {size['prompt']}. Config: {run['config_source']}. verify: "
        f"{', '.join(size['impl']['verify'])}; reward: {', '.join(size['impl']['reward'])}.",
        f"- Episode records: {size['n_records']}. Harness errors: {size['n_harness_errors']} "
        f"({_pct(size['harness_error_share'])} of the records); they are left out of every number below. "
        f"Episodes counted: {size['n_episodes']}.",
        f"- Families: {size['n_families']} ({_named(size['n_families_by_outcome'])}). "
        f"Scenarios: {size['n_scenarios']}"
        + (f" (and {size['n_scenarios_without_valid_sample']} more with no valid sample)"
           if size["n_scenarios_without_valid_sample"] else "") + ".",
        f"- Valid samples per scenario: {equal}.", "",
    ]


def _md_pass(run: dict, report: dict) -> list[str]:
    rates = run["pass"]
    lines = [
        "### 2. Pass rate", "",
        "A family's pass rate is the share of its episodes that passed; every pass rate here is the mean "
        "over families.", "",
        f"- Over all: {_pct(rates['pass_rate'])} ({_level(report)} interval {_pct(rates['lo'])} to "
        f"{_pct(rates['hi'])}, from resampling the {rates['n_families']} families).", "",
    ]
    rows = [[f"outcome {key}", _pct(part["pass_rate"]), part["n_families"]]
            for key, part in rates["by_outcome"].items()]
    rows += [[f"style {key}", _pct(part["pass_rate"]), part["n_families"]]
             for key, part in rates["by_style"].items()]
    names = {SEEN: "seen styles (1 to 3)", HELD_OUT: "held-out style (4)"}
    rows += [[names[key], _pct(part["pass_rate"]), part["n_families"]]
             for key, part in rates["seen_held_out"].items()]
    lines += _table(["Part", "Pass rate", "Families"], rows)
    diff = rates["held_out_minus_seen"]
    if diff is None:
        lines += ["Held-out minus seen: not given (the two parts do not hold the same families, or there "
                  "are fewer than 2 families).", ""]
    else:
        lines += [f"Held-out minus seen: {_points(diff['diff'])} points ({_level(report)} interval "
                  f"{_points(diff['lo'])} to {_points(diff['hi'])}, paired bootstrap over "
                  f"{diff['n_families']} families).", ""]
    return lines


def _md_checks(run: dict, report: dict) -> list[str]:
    checks = run["checks"]
    floor, band, room = checks["floor"], checks["band"], checks["room"]
    return [
        "### 3. The two checks of the plan", "",
        f"- (a) Floor. Is the lower end of the {_level(report)} interval above {_pct(floor['threshold'])}? "
        f"**{_yes_no(floor['passed'])}**: the lower end is {_pct(floor['interval_lo'])}. "
        f"(The best policy that always gives the same resolution passes "
        f"{_pct(floor['best_constant_policy'])} of this run's families.)",
        f"- (b) Band. Is the pass rate between {_pct(band['lo'])} and {_pct(band['hi'])}? "
        f"**{_yes_no(band['passed'])}**: the pass rate is {_pct(band['pass_rate'])}, {band['position']} "
        f"the band.",
        f"- Room left (100% minus the pass rate): {_pct(room['all'])} over all; "
        + ", ".join(f"{outcome} {_pct(value)}" for outcome, value in room["by_outcome"].items()) + ".", "",
    ]


def _tie_row(label: str, share: dict) -> list[str]:
    return [label, share["n_scenarios"], _pct(share["equal_reward"]), _pct(share["equal_reward_all_pass"]),
            _pct(share["equal_reward_all_fail"]), _pct(share["equal_passfail"])]


def _md_ties(run: dict) -> list[str]:
    tie, profile = run["ties"], run["ties"]["family_profile"]
    rows = [_tie_row("all", tie["all"])]
    rows += [_tie_row(f"outcome {outcome}", share) for outcome, share in tie["by_outcome"].items()]
    counted = _sizes_text(tie["samples_per_scenario"]) or "no scenario with 2 samples"
    return [
        "### 4. Ties", "",
        "A scenario in which every sample has the same reward gives GRPO no signal. Shares are of the "
        "scenarios with at least 2 valid samples.", "",
        *_table(["Scenarios", "Counted", "Same reward", "of which all pass", "of which all fail",
                 "Same pass or fail"], rows),
        f"- Computed with: {counted}; left out for having fewer than 2 valid samples: "
        f"{_n(tie['n_scenarios_left_out'], 'scenario')}. These shares fall as the number of samples per "
        f"scenario grows, so they hold for this number only.",
        f"- Family profile: of {profile['n_families']} families, {profile['always_pass']} always pass, "
        f"{profile['always_fail']} always fail, {profile['mixed']} are mixed.", "",
    ]


def _md_calls(run: dict) -> list[str]:
    calls = run["call_format"]
    return [
        "### 5. Call format", "",
        f"- Invalid-call rate: {_pct(calls['invalid_call_rate'])} ({calls['n_malformed']} of "
        f"{calls['n_messages']} assistant messages were malformed).",
        f"- Parse errors (messages): {_named(calls['parse_errors'])}.",
        f"- Tool errors (well-formed calls that the tool refused; {calls['n_tool_errors']} in all): "
        f"{_named(calls['tool_errors'])}.",
        f"- Turns cut off by the token cap: {_pct(calls['truncated_share'])} ({calls['n_truncated']} of "
        f"{calls['n_messages']} messages).", "",
    ]


def _md_turns(run: dict, report: dict) -> list[str]:
    turns, tokens, rule = run["turns"], run["turns"]["tokens"], report["token_cap_rule"]
    lines = [
        "### 6. Turns and tokens", "",
        f"- Mean turns per episode: {turns['mean_turns']:.2f} ("
        + ", ".join(f"{outcome} {value:.2f}" for outcome, value in turns["mean_turns_by_outcome"].items())
        + ").",
        f"- Episodes that used up the turn limit without closing: {_pct(turns['share_at_turn_limit'])}.",
        f"- Episodes over their turn budget: {_pct(turns['share_over_budget'])} "
        f"({turns['n_over_budget']} of {turns['n_episodes']}).",
    ]
    if tokens is None:
        return lines + ["- New tokens per well-formed call: not given (no stored token counts).", ""]
    in_force = "not known" if tokens["cap_in_force"] is None else str(tokens["cap_in_force"])
    return lines + [
        f"- New tokens per well-formed call ({tokens['n_calls']} calls, end-of-turn token included): "
        f"median {tokens['median']:.1f}, {rule['percentile']}th percentile {tokens['p99']}, largest "
        f"{tokens['max']}.",
        f"- Token cap by the plan's rule ({rule['factor']} times the {rule['percentile']}th percentile, at "
        f"least {rule['minimum']}): **{tokens['cap_by_rule']}**. Cap in force in this run: {in_force}. "
        f"Messages cut off at the cap in force: {tokens['n_truncated']}; their lengths are not in these "
        f"numbers.", "",
    ]


def _md_penalties(run: dict) -> list[str]:
    pen = run["penalties"]
    rows = []
    for key, with_label, without_label in (("malformed", "at least one malformed call", "no malformed call"),
                                           ("turn_budget", "over the turn budget", "inside the turn budget")):
        for part, among in ((pen[key], "all episodes"), (pen[key]["closed_only"], "closed the request")):
            for side, label in (("with", with_label), ("without", without_label)):
                group = part[side]
                rows.append([label, among, group["n_episodes"], group["n_passed"], group["n_families"],
                             _pct(group["pass_rate"])])
    rewards = ", ".join(f"reward {row['reward']}: {row['episodes']}" for row in pen["rewards_of_passes"])
    never = pen["malformed"]["never_closed"]
    lines = [
        "### 7. Penalties", "",
        "Pass rate (mean over families): the mean, over the families that have an episode in the group, of "
        "the family's share of passed episodes in the group. It is not `of which passed` divided by "
        "`Episodes`.", "",
        *_table(["Episodes with", "Among", "Episodes", "of which passed", "Families",
                 "Pass rate (mean over families)"], rows),
        f"- Episodes that never closed the request: {never['n_episodes']} (passed: {never['n_passed']}); "
        f"of these, with a malformed call: {never['n_with']}; over the turn budget: "
        f"{pen['turn_budget']['never_closed']['n_with']}.",
    ]
    for key, label in (("malformed", "Malformed-call penalty"), ("turn_budget", "Turn penalty")):
        lines.append(f"- {label}: {pen[key]['sentence']}")
        if pen[key]["caution"]:
            lines.append(f"- {label}, caution: {pen[key]['caution']}")
        lines.append(f"- {label}: {pen[key]['closed_only']['sentence']}")
    return lines + [
        f"- Episodes in both groups (a malformed call and over the budget): {pen['n_both']}.",
        f"- Rewards of the {pen['n_passes']} passing episodes: {rewards or 'none'}.", "",
    ]


def _md_failures(run: dict) -> list[str]:
    fails = run["failures"]
    lines = ["### 8. What the model does when it fails", "",
             "- Every episode has one label: "
             + ", ".join(f"{label} {_pct(fails['labels'][label])} ({fails['label_counts'][label]})"
                         for label in fails["labels"]) + ".", ""]
    wrong, correct = fails["closed_wrongly"], fails["correct_resolution"]
    columns = list(RESOLUTIONS) + sorted({key for row in wrong["table"].values() for key in row}
                                         - set(RESOLUTIONS))
    if wrong["n"]:
        rows = [[outcome, "n/a" if correct is None else correct[outcome], wrong["n_by_outcome"][outcome]]
                + [row.get(column, 0) for column in columns] for outcome, row in wrong["table"].items()]
        lines += [f"The {wrong['n']} episodes that closed the request wrongly, by the resolution they wrote:",
                  "",
                  *_table(["Expected outcome", "Correct resolution", "Episodes"]
                          + [f"wrote {column}" for column in columns], rows),
                  "A count under the correct resolution means: the resolution was right and another part "
                  "of the end state was wrong.", ""]
    else:
        lines += ["No episode closed the request wrongly.", ""]
    if fails["missing"] is not None:
        return lines + [f"**Missing: the wrong-person count and the unexpected-grant count.** They need the "
                        f"correct end state of each family: {fails['missing']}.", ""]
    person, grant = fails["wrong_person"], fails["unexpected_grant"]
    return lines + [
        f"- Closed with the id of another person than the one meant: {person['n']} of {person['n_closed']} "
        f"closed episodes ({_pct(person['share_of_closed'])}).",
        f"- A role was given that should not have been given (the unsafe mistake): {grant['n']} episodes "
        f"({_pct(grant['share'])} of all episodes); by expected outcome: {_named(grant['by_outcome'])}.", "",
    ]


def _md_supply(run: dict) -> list[str]:
    supply = run["sft_supply"]
    return [
        "### 9. Supply for the supervised stage", "",
        f"- Clean passes (reward {supply['pass_reward']}): {_pct(supply['share_clean'])} of the episodes "
        f"({supply['n_clean']} of {supply['n_episodes']}); "
        + ", ".join(f"{outcome} {_pct(part['share_clean'])} ({part['n_clean']} of {part['n_episodes']})"
                    for outcome, part in supply["by_outcome"].items()) + ".",
        f"- Scenarios with at least one clean pass: {supply['n_scenarios_with_clean_pass']} of "
        f"{supply['n_scenarios']}. Families with at least one: {supply['n_families_with_clean_pass']} of "
        f"{supply['n_families']}.", "",
    ]


def _md_run(run: dict, report: dict) -> list[str]:
    lines = [f"## Run {run['name']}", ""]
    if run["warnings"]:
        lines += ["Warnings:", ""] + [f"- {warning}" for warning in run["warnings"]] + [""]
    return (lines + _md_size(run) + _md_pass(run, report) + _md_checks(run, report) + _md_ties(run)
            + _md_calls(run) + _md_turns(run, report) + _md_penalties(run) + _md_failures(run)
            + _md_supply(run))


def _side_rows(report: dict) -> list[tuple[str, list[str]]]:
    """The rows of the side-by-side table: a label, and one cell per run."""
    runs = [report["runs"][name] for name in report["side_by_side"]["runs"]]

    def row(label: str, cell) -> tuple[str, list[str]]:
        return label, [str(cell(run)) for run in runs]

    def samples(run: dict) -> str:
        per = run["size"]["samples_per_scenario"]
        return str(per["min"]) if per["all_equal"] else f"{per['min']} to {per['max']} (unequal)"

    def tokens(run: dict, key: str) -> str:
        return "n/a" if run["turns"]["tokens"] is None else str(run["turns"]["tokens"][key])

    def fails(run: dict, key: str) -> str:
        return "n/a" if run["failures"][key] is None else str(run["failures"][key]["n"])

    rows = [
        row("Model", lambda r: r["size"]["model"] or "not recorded"),
        row("Placeholder text", lambda r: "yes" if r["size"]["placeholder_text"] else "no"),
        row("Split", lambda r: ", ".join(r["size"]["split"])),
        row("Episodes counted", lambda r: r["size"]["n_episodes"]),
        row("Harness errors", lambda r: r["size"]["n_harness_errors"]),
        row("Families", lambda r: r["size"]["n_families"]),
        row("Scenarios", lambda r: r["size"]["n_scenarios"]),
        row("Samples per scenario", samples),
        row("Pass rate", lambda r: _pct(r["pass"]["pass_rate"])),
        row(f"{_level(report)} interval", lambda r: f"{_pct(r['pass']['lo'])} to {_pct(r['pass']['hi'])}"),
    ]
    for outcome in OUTCOMES:
        rows.append(row(f"Pass rate, {outcome}",
                        lambda r, o=outcome: _pct(r["pass"]["by_outcome"].get(o, {}).get("pass_rate"))))
    for part, label in ((SEEN, "seen styles"), (HELD_OUT, "held-out style")):
        rows.append(row(f"Pass rate, {label}",
                        lambda r, p=part: _pct(r["pass"]["seen_held_out"].get(p, {}).get("pass_rate"))))
    rows += [
        row("Floor check (interval above the floor)", lambda r: _yes_no(r["checks"]["floor"]["passed"])),
        row("Band check (pass rate in the band)",
            lambda r: f"{_yes_no(r['checks']['band']['passed'])} ({r['checks']['band']['position']})"),
        row("Room left", lambda r: _pct(r["checks"]["room"]["all"])),
        row("Scenarios with the same reward", lambda r: _pct(r["ties"]["all"]["equal_reward"])),
        row("Scenarios with the same pass or fail", lambda r: _pct(r["ties"]["all"]["equal_passfail"])),
        row("Families always pass / always fail / mixed",
            lambda r: " / ".join(str(r["ties"]["family_profile"][key])
                                 for key in ("always_pass", "always_fail", "mixed"))),
        row("Invalid-call rate", lambda r: _pct(r["call_format"]["invalid_call_rate"])),
        row("Turns cut off by the token cap", lambda r: _pct(r["call_format"]["truncated_share"])),
        row("Mean turns per episode", lambda r: f"{r['turns']['mean_turns']:.2f}"),
        row("Episodes at the turn limit", lambda r: _pct(r["turns"]["share_at_turn_limit"])),
        row("Episodes over the turn budget", lambda r: _pct(r["turns"]["share_over_budget"])),
        row("Tokens per well-formed call, 99th percentile", lambda r: tokens(r, "p99")),
        row("Token cap by the plan's rule", lambda r: tokens(r, "cap_by_rule")),
        row("Clean passes, share of episodes", lambda r: _pct(r["sft_supply"]["share_clean"])),
        row("Scenarios with a clean pass", lambda r: r["sft_supply"]["n_scenarios_with_clean_pass"]),
        row("Episodes closed for the wrong person", lambda r: fails(r, "wrong_person")),
        row("Episodes with a role given wrongly", lambda r: fails(r, "unexpected_grant")),
    ]
    return rows


def _md_side(report: dict) -> list[str]:
    side = report["side_by_side"]
    lines = ["## Side by side", ""]
    if side["warnings"]:
        lines += [f"- {warning}" for warning in side["warnings"]] + [""]
    lines += _table(["Number"] + side["runs"], [[label] + cells for label, cells in _side_rows(report)])
    for row in side["differences"]:
        lines.append(f"- Pass rate, {row['b']} minus {row['a']}: {_points(row['diff'])} points "
                     f"({_level(report)} interval {_points(row['lo'])} to {_points(row['hi'])}, paired "
                     f"bootstrap over {row['n_families']} families; t-interval {_points(row['t_lo'])} to "
                     f"{_points(row['t_hi'])}).")
    if side["differences"]:
        lines.append("")
    return lines


def to_markdown(report: dict) -> str:
    """The report of `build_report` as markdown. It computes nothing: every number it shows is in `report`
    (section headings and the short forms of hashes are not numbers of the report).

    If a run used the placeholder text, the first line says so."""
    placeholder = report["placeholder_text_runs"]
    lines = []
    if placeholder:
        lines += [f"**PLACEHOLDER TEXT: the run(s) {', '.join(placeholder)} used the placeholder text file. "
                  f"This is a trial of the report, not a calibration of the final text.**", ""]
    lines += [
        f"# {report['title']}", "",
        "Calibration numbers are not results and are in no results table (PLAN 5.3). This report states "
        "numbers and decides nothing.", "",
        f"Intervals: {_level(report)}, by resampling families, {report['n_draws']} draws, seed "
        f"{report['seed']}. Paired differences use the {report['paired_family_bootstrap']} version of "
        f"paired_family_bootstrap.", "",
    ]
    for name in report["side_by_side"]["runs"]:
        lines += _md_run(report["runs"][name], report)
    if len(report["runs"]) > 1:
        lines += _md_side(report)
    return "\n".join(lines).rstrip("\n") + "\n"
