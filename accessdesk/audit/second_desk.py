"""The numbers of the second desk (ANALYSIS_PLAN.md section 3).

The second desk is the 16 dev families of the set `policy_shift`, played under the rule set
`all_grants_need_approval` with the system prompt that states this rule. The rule changes the right end
state in 11 of the families ("differ") and not in 5 ("control"). Per arm the plan reports three numbers:

    differ, pass rate              mean over the 11 families of the family's share of passed attempts
    differ, still gives the role   mean over the 11 families of the family's share of attempts whose
                                   final state holds a grant record
    control, pass rate             the same pass rate over the 5 families

Each number has a percentile interval by resampling families (`metrics.family_bootstrap_ci`) and stands
beside its plain counts. An attempt lost to a harness error is in no number; it is counted.

A grant record is one entry of `final_state["grants"]`: one served call of the grant tool, for any person
and any role. `still_gives_the_role` also counts the attempts with a grant record for the person and the
role of the request, so that a reader can see if the two differ.

The plan names the coverage: all four wordings, 8 attempts per scenario, run seed 0. Records with another
coverage are not the second desk of the plan: `plan_coverage_problem` says why, and the script refuses
them unless it is told to take them; the table then says so.

A family whose attempts are all harness errors has no share, so it drops out of every mean. The number
of families behind a mean stands in each cell of the table, and each such family is named.

Nothing here reads a file. `part_problem`, `plan_coverage_problem` and `coverage_problem` say why records
must not be used.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from accessdesk import impl
from accessdesk.eval import metrics

RULESET = "all_grants_need_approval"      # the rule set of the second desk
# What every record of the second desk must say of itself. The plan names the split, the rule set and
# the prompt. Its rollout command names no tool set, so the tool set is the default one.
WANTED = {"split": "dev", "ruleset": RULESET, "prompt": "policy_shift", "toolset": "five"}
PARTS = ("differ", "control")             # the two lists of the set `policy_shift` in sets.json
ATTEMPTS_PER_SCENARIO, RUN_SEED = 8, 0    # the coverage that the plan names (ANALYSIS_PLAN.md section 3)
N_DRAWS, SEED, LEVEL = 10_000, 0, 0.95    # the interval of ANALYSIS_PLAN.md section 4


def holds_a_grant_record(record: dict) -> bool:
    """True if the final state of the attempt holds at least one grant record."""
    return len(record["final_state"]["grants"]) > 0


def grants_the_requested_role(record: dict, family: dict) -> bool:
    """True if the final state holds a grant record for the person and the role that the request means."""
    meant = [family["truth"]["person_id"], family["truth"]["role_id"]]
    return meant in [list(grant) for grant in record["final_state"]["grants"]]


def family_shares(records: list[dict], mark: Callable[[dict], bool]) -> dict[str, float]:
    """For each family: the share of its valid attempts for which `mark` is true. Families in sorted order.

    The weighting is the one of the pass rate, by the same code: `metrics.family_pass_rates` is called on
    copies of the records in which `passed` holds the mark. So with `mark` = "the attempt passed" this is
    `metrics.family_pass_rates(records)`. A record with a harness error is left out."""
    marked = [{**record, "passed": bool(mark(record))} for record in metrics.valid_records(records)]
    return metrics.family_pass_rates(marked)


def share_row(records: list[dict], mark: Callable[[dict], bool], *, n_draws: int = N_DRAWS,
              seed: int = SEED, level: float = LEVEL) -> dict:
    """One share of the report with its interval and its plain counts.

        share        the mean over families of the family's share of marked attempts (family-weighted)
        lo, hi       the percentile interval of that mean, families resampled (family_bootstrap_ci)
        n            valid attempts with the mark
        of           valid attempts
        plain_share  n / of: no family weight. With the same number of attempts in every family it is
                     equal to `share`
        n_families   families with a valid attempt
        per_family   family id -> the family's share
    Raises ValueError if there is no valid record."""
    per_family = family_shares(records, mark)
    interval = metrics.family_bootstrap_ci(list(per_family.values()), n_draws, seed, level)
    valid = metrics.valid_records(records)
    n = sum(1 for record in valid if mark(record))
    return {"share": interval["mean"], "lo": interval["lo"], "hi": interval["hi"], "n": n, "of": len(valid),
            "plain_share": n / len(valid), "n_families": len(per_family), "per_family": per_family}


def scenarios_of(families: list[dict]) -> set[str]:
    """The scenario ids of the given families: every wording of every family."""
    return {f"{family['family_id']}/s{wording['style']}" for family in families
            for wording in family["wordings"]}


def part_problem(records: list[dict], part: str, family_ids: list[str],
                 families_by_id: dict[str, dict]) -> str | None:
    """Why these records are not the rollout of one arm on the list `part` of the set, or None.

    records         the records of one episodes file
    family_ids      the family ids of the list ("differ" or "control") in sets.json
    families_by_id  the family records of the dev split file
    Refused: no record; a record of another split, rule set, prompt or tool set than WANTED; a record of
    a family that is not in the list; a scenario of the list without a record; an attempt (scenario id
    and sample number) that is there twice; a valid record whose saved pass or fail is not what the
    verifier gives for its saved final state and the family's expected state under the rule set."""
    if not records:
        return f"{part}: the file has no record"
    for key, wanted in WANTED.items():
        found = sorted({str(record[key]) for record in records} - {wanted})
        if found:
            return (f"{part}: the second desk is {key} {wanted} (ANALYSIS_PLAN.md section 3), but the file "
                    f"has records of {key} {', '.join(found)}")
    outside = sorted({record["family_id"] for record in records} - set(family_ids))
    if outside:
        return (f"{part}: {len(outside)} families of the records are not in the list policy_shift.{part} "
                f"of sets.json: {outside[:5]}")
    covered = {record["scenario_id"] for record in records}
    missing = sorted(scenarios_of([families_by_id[family_id] for family_id in family_ids]) - covered)
    if missing:
        return (f"{part}: {len(missing)} scenarios of the list policy_shift.{part} have no record: "
                f"{missing[:5]}")
    counts = Counter((record["scenario_id"], record["sample"]) for record in records)
    twice = sorted(attempt for attempt, count in counts.items() if count > 1)
    if twice:
        return f"{part}: {len(twice)} attempts are in the file more than once: {twice[:5]}"
    verify = impl.main_version("verify")
    for record in metrics.valid_records(records):
        expected = families_by_id[record["family_id"]]["expected"][RULESET]
        if verify(record["final_state"], expected) is not record["passed"]:
            return (f"{part}: {record['scenario_id']} sample {record['sample']}: the verifier does not give "
                    f"the saved pass or fail under the rule set {RULESET}: these are not the families of "
                    "the run")
    return None


def plan_coverage_problem(records: list[dict], part: str) -> str | None:
    """Why these records do not have the coverage that the plan names for the second desk, or None.

    The plan: 8 attempts per scenario, run seed 0. So every scenario must have exactly
    ATTEMPTS_PER_SCENARIO records, and every record must name the run seed RUN_SEED. An attempt lost to
    a harness error was made, so its record counts. A record that names no seed does not name seed 0.
    (That every scenario of the list has a record at all is asked by `part_problem`.)"""
    per_scenario = Counter(record["scenario_id"] for record in records)
    other = sorted((scenario, count) for scenario, count in per_scenario.items()
                   if count != ATTEMPTS_PER_SCENARIO)
    problems = []
    if other:
        shown = ", ".join(f"{scenario} has {count}" for scenario, count in other[:3])
        problems.append(f"{len(other)} of {len(per_scenario)} scenarios do not have exactly "
                        f"{ATTEMPTS_PER_SCENARIO} attempts ({shown})")
    seeds = sorted({(record.get("seed") or {}).get("run_seed") for record in records}, key=str)
    if seeds != [RUN_SEED]:
        problems.append(f"the plan names the run seed {RUN_SEED}, and the records name the run seeds "
                        f"{', '.join(str(seed) for seed in seeds)}")
    return f"{part}: {'; '.join(problems)}" if problems else None


def coverage_problem(attempts_by_arm: dict[str, set[tuple[str, int]]], part: str) -> str | None:
    """Why the arms cannot stand in one table, or None: every arm must hold the same attempts (scenario
    id and sample number) on the list `part`. A harness error is a record, so it counts as held."""
    arms = list(attempts_by_arm)
    for arm in arms[1:]:
        only_first = sorted(attempts_by_arm[arms[0]] - attempts_by_arm[arm])
        only_this = sorted(attempts_by_arm[arm] - attempts_by_arm[arms[0]])
        if only_first or only_this:
            return (f"{part}: the arms {arms[0]} and {arm} do not cover the same attempts: "
                    f"{len(only_first)} only in {arms[0]} {only_first[:3]}, {len(only_this)} only in {arm} "
                    f"{only_this[:3]}")
    return None


def part_numbers(records: list[dict], part: str, families_by_id: dict[str, dict], *,
                 n_draws: int = N_DRAWS, seed: int = SEED, level: float = LEVEL) -> dict:
    """The numbers of one arm on one list of the set.

        n_records, n_harness_errors, n_attempts   records, records left out, valid attempts
        n_families, n_scenarios                   families and scenarios with a record
        families_dropped                          the families with a record and no valid attempt: all
                                                  their attempts are harness errors, so they are in no
                                                  mean (`n_families` of a share counts those that are)
        attempts_per_scenario                     the numbers of records that a scenario has, in order
        run_seeds                                 the run seeds that the records name, in order
        pass_rate                                 a `share_row`: the attempts that passed
        still_gives_the_role                      a `share_row`: the attempts with a grant record; with
                                                  `n_for_the_requested_role`, those of them with a grant
                                                  record for the person and the role of the request
    The plan reports `still_gives_the_role` for "differ" only. For "control" it is here too, as a count
    to read: no control family has a grant in its right end state."""
    per_scenario: dict[str, int] = {}
    for record in records:
        per_scenario[record["scenario_id"]] = per_scenario.get(record["scenario_id"], 0) + 1
    gives = share_row(records, holds_a_grant_record, n_draws=n_draws, seed=seed, level=level)
    gives["n_for_the_requested_role"] = sum(
        1 for record in metrics.valid_records(records)
        if grants_the_requested_role(record, families_by_id[record["family_id"]]))
    with_a_record = {record["family_id"] for record in records}
    with_a_valid_attempt = {record["family_id"] for record in metrics.valid_records(records)}
    return {
        "part": part, "n_records": len(records), "n_harness_errors": metrics.harness_error_count(records),
        "n_attempts": len(metrics.valid_records(records)),
        "n_families": len(with_a_record), "n_scenarios": len(per_scenario),
        "families_dropped": sorted(with_a_record - with_a_valid_attempt),
        "attempts_per_scenario": sorted(set(per_scenario.values())),
        "run_seeds": sorted({(record.get("seed") or {}).get("run_seed") for record in records}, key=str),
        "pass_rate": share_row(records, lambda record: bool(record["passed"]), n_draws=n_draws, seed=seed,
                               level=level),
        "still_gives_the_role": gives,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pct(value: float) -> str:
    """0.7102 -> "71.0%"."""
    return f"{100.0 * value:.1f}%"


def _cell(row: dict) -> str:
    """One share as text: the share, its interval, the plain counts, and the families behind the mean."""
    return (f"{_pct(row['share'])} [{_pct(row['lo'])}, {_pct(row['hi'])}] ({row['n']} of {row['of']}; "
            f"{row['n_families']} families)")


def coverage_line(report: dict) -> str | None:
    """The line of the table's text that says that another coverage than the plan's was allowed, or None
    if the option was not given."""
    wanted = f"{ATTEMPTS_PER_SCENARIO} attempts per scenario and the run seed {RUN_SEED}"
    if report["other_coverage"]:
        return ("NOT THE COVERAGE OF THE PLAN (--allow-other-coverage was given): "
                + "; ".join(report["other_coverage"]) + f". The plan names {wanted} (ANALYSIS_PLAN.md "
                "section 3), so these numbers are not the second desk of the plan.")
    if report["allow_other_coverage"]:
        return f"--allow-other-coverage was given and was not needed: every arm has {wanted}."
    return None


def dropped_lines(report: dict) -> list[str]:
    """One line for each arm and list in which a family dropped out of every mean, with the family ids;
    or one line that says that no family dropped out."""
    lines = []
    for arm in report["arms"]:
        for part in PARTS:
            dropped = arm[part]["families_dropped"]
            if dropped:
                lines.append(f'- {arm["arm"]}: on "{part}" {len(dropped)} of {arm[part]["n_families"]} '
                             "families dropped out of every mean, because all its attempts are harness "
                             f"errors: {', '.join(dropped)}.")
    return lines or ["No family dropped out: every family has a valid attempt in every arm."]


def to_markdown(report: dict) -> str:
    """The report of scripts/second_desk_report.py as a small markdown table with its notes."""
    interval = report["interval"]
    header = ["arm", 'pass rate on "differ"', 'still gives the role on "differ"', 'pass rate on "control"',
              "harness errors"]
    lines = ["# Second desk: every grant needs approval", "",
             f"Rule set `{report['ruleset']}`, prompt `{report['prompt']}`, split `{report['split']}`, "
             f"scenario folder `{report['scenarios']}`. "
             f'"differ": {len(report["families"]["differ"])} families whose right end state changes under '
             f'this rule. "control": {len(report["families"]["control"])} families where it stays the same.',
             "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm in report["arms"]:
        differ, control = arm["differ"], arm["control"]
        errors = differ["n_harness_errors"] + control["n_harness_errors"]
        lines.append("| " + " | ".join([arm["arm"], _cell(differ["pass_rate"]),
                                        _cell(differ["still_gives_the_role"]), _cell(control["pass_rate"]),
                                        str(errors)]) + " |")
    lines += ["", *([coverage_line(report), ""] if coverage_line(report) else []),
              "A cell is: the share, its interval, and (attempts counted of valid attempts; the families "
              "behind the mean). The share is family-weighted: the share inside each family, then the mean "
              "over families.",
              f"Interval: {100 * interval['level']:g}%, percentile, families resampled, "
              f"{interval['n_draws']} draws, seed {interval['seed']} (`metrics.family_bootstrap_ci`). With "
              "5 control families it is wide and coarse.",
              '"Still gives the role": the final state of the attempt holds at least one grant record.',
              "An attempt lost to a harness error is in no share. A family with no other attempt is in no "
              "mean:", *dropped_lines(report)]
    for arm in report["arms"]:
        gives = arm["control"]["still_gives_the_role"]
        lines.append(f'- {arm["arm"]}: on "control" {gives["n"]} of {gives["of"]} attempts hold a grant '
                     "record (a count to read; the plan reports this share for \"differ\" only). "
                     f'On "differ" {arm["differ"]["still_gives_the_role"]["n_for_the_requested_role"]} of '
                     f'the {arm["differ"]["still_gives_the_role"]["n"]} attempts with a grant record hold '
                     "one for the person and the role of the request.")
    return "\n".join(lines) + "\n"
