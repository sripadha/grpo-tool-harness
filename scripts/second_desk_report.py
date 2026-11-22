"""The second desk (ANALYSIS_PLAN.md section 3): per arm, the pass rate under the changed rule and the share
of attempts that still give the role. Written after the freeze; it only reads saved attempts.

    .venv/bin/python scripts/second_desk_report.py --scenarios data/scenarios \\
        --arm base results/main/base_shift_differ/episodes.jsonl \\
                   results/main/base_shift_control/episodes.jsonl \\
        --arm sft  results/main/sft_shift_differ/episodes.jsonl \\
                   results/main/sft_shift_control/episodes.jsonl \\
        --out results/main/second_desk

--arm NAME DIFFER CONTROL   one arm: its name in the table, the episodes file of its rollout on the set
                            policy_shift_differ, and the one on policy_shift_control. Give it once per arm.
--scenarios  the scenario folder of the rollouts (its dev.jsonl and sets.json are read)
--text       the text file of the rollouts (default: text/model_text.yaml)
--out        the folder for second_desk.json and second_desk.md (it is made if it is missing)
--draws, --seed   the bootstrap (default: 10,000 draws, seed 0, as in section 4 of the plan)
--allow-other-coverage   take files that do not have the coverage of the plan (8 attempts per scenario,
             run seed 0). The table then says so in a line of its own, and names what differs.

The numbers and what each one means are in accessdesk/audit/second_desk.py.

The script writes nothing and stops with exit code 2, with the reason on stderr, if
  - the scenario folder was not built from the text file, or its dev.jsonl or sets.json is not the file
    that the build wrote, or the set policy_shift of sets.json is for another rule set;
  - a file has a record of another split, rule set, prompt or tool set than the plan names (dev,
    all_grants_need_approval, policy_shift, five), or of another text file or scenario file;
  - a record is of a family that is not in the list of the set, or a scenario of the list has no record,
    or an attempt is in a file twice;
  - the verifier does not give the saved pass or fail of a record under the changed rule;
  - a scenario of an arm does not have exactly 8 attempts, or a record names another run seed than 0
    (the coverage that the plan names), unless --allow-other-coverage is given;
  - two arms do not hold the same attempts (scenario id and sample number).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.audit import load_script, second_desk  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402
from accessdesk.text import TextError, load_text, model_text_path  # noqa: E402

SPLIT = second_desk.WANTED["split"]
JSON_NAME, MARKDOWN_NAME = "second_desk.json", "second_desk.md"


def set_lists(scenarios: Path, text_path: Path) -> dict[str, list[str]]:
    """The two lists of the set policy_shift: {"differ": [family ids], "control": [family ids]}.

    They are read with the function of scripts/gpu/rollout.py that the rollouts used. Raises ValueError if
    the folder cannot be used (see the module text)."""
    rollout = load_script("gpu/rollout")
    try:
        text = load_text(text_path)
    except (OSError, TextError) as error:
        raise ValueError(f"the text file cannot be used: {error}") from None
    problem = rollout.check_scenarios(scenarios, text, (f"{SPLIT}.jsonl", rollout.SETS_FILE))
    if problem:
        raise ValueError(problem)
    sets = json.loads((scenarios / rollout.SETS_FILE).read_text(encoding="utf-8"))
    variant = sets["policy_shift"]["variant"]
    if variant != second_desk.RULESET:
        raise ValueError(f"the set policy_shift of {scenarios / rollout.SETS_FILE} is for the rule set "
                         f"{variant}, not for {second_desk.RULESET}")
    return {part: rollout.set_family_ids(scenarios, rollout.POLICY_SHIFT_PREFIX + part)
            for part in second_desk.PARTS}


def records_of_part(episodes: Path, part: str, family_ids: list[str], scenarios: Path,
                    text_path: Path) -> tuple[list[dict], dict[str, dict]]:
    """(the records of one episodes file, the family records of the dev split by id), after every check
    of one file. Raises ValueError with the reason if the file must not be used."""
    monitor_report = load_script("monitor_report")
    if not episodes.is_file():
        raise ValueError(f"{episodes} does not exist")
    records = read_records(episodes)
    if not records:
        raise ValueError(f"{episodes} has no record")
    monitor_report.text_of_run(text_path, records)
    families, _ = monitor_report.families_of_run(scenarios, SPLIT, records)
    by_id = {family["family_id"]: family for family in families}
    problem = second_desk.part_problem(records, part, family_ids, by_id)
    if problem:
        raise ValueError(f"{episodes}: {problem}")
    return records, by_id


def build_report(arms: list[tuple[str, Path, Path]], scenarios: Path, text_path: Path, *,
                 n_draws: int = second_desk.N_DRAWS, seed: int = second_desk.SEED,
                 allow_other_coverage: bool = False) -> dict:
    """The whole report as a dict (the content of second_desk.json). `arms`: (name, the episodes file on
    "differ", the one on "control"). Raises ValueError for everything that the module text names.

    `other_coverage` of the report lists, for each arm and list, why its records do not have the
    coverage of the plan (`second_desk.plan_coverage_problem`). Without `allow_other_coverage` such a
    list is refused."""
    names = [name for name, _, _ in arms]
    if not arms or len(set(names)) != len(names):
        raise ValueError(f"give each arm once, with its own name: {names}")
    monitor_report = load_script("monitor_report")
    lists = set_lists(scenarios, text_path)
    rows = [{"arm": name} for name in names]
    other_coverage = []
    for position, part in enumerate(second_desk.PARTS):
        attempts: dict[str, set[tuple[str, int]]] = {}
        for row, arm in zip(rows, arms):
            episodes = arm[1 + position]
            records, by_id = records_of_part(episodes, part, lists[part], scenarios, text_path)
            attempts[row["arm"]] = {(record["scenario_id"], record["sample"]) for record in records}
            problem = second_desk.plan_coverage_problem(records, part)
            if problem:
                other_coverage.append(f"{row['arm']}, {problem}")
            row[part] = {"episodes": monitor_report.shown_path(episodes),
                         "episodes_sha256": monitor_report.sha256_of_file(episodes),
                         **second_desk.part_numbers(records, part, by_id, n_draws=n_draws, seed=seed)}
        problem = second_desk.coverage_problem(attempts, part)
        if problem:
            raise ValueError(problem)
    if other_coverage and not allow_other_coverage:
        raise ValueError("; ".join(other_coverage) + ". ANALYSIS_PLAN.md section 3 names "
                         f"{second_desk.ATTEMPTS_PER_SCENARIO} attempts per scenario and the run seed "
                         f"{second_desk.RUN_SEED}. To report these files all the same, give "
                         "--allow-other-coverage: the table then says so")
    return {
        "what": "second desk, ANALYSIS_PLAN.md section 3", **second_desk.WANTED,
        "plan_coverage": {"attempts_per_scenario": second_desk.ATTEMPTS_PER_SCENARIO,
                          "run_seed": second_desk.RUN_SEED},
        "allow_other_coverage": allow_other_coverage, "other_coverage": other_coverage,
        "scenarios": monitor_report.shown_path(scenarios),
        "split_file_sha256": monitor_report.sha256_of_file(scenarios / f"{SPLIT}.jsonl"),
        "sets_file_sha256": monitor_report.sha256_of_file(scenarios / "sets.json"),
        "text_file": monitor_report.shown_path(text_path),
        "text_sha256": monitor_report.sha256_of_file(text_path),
        "families": lists,
        "interval": {"level": second_desk.LEVEL, "n_draws": n_draws, "seed": seed,
                     "method": "percentile, families resampled (metrics.family_bootstrap_ci)"},
        "arms": rows,
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--arm", nargs=3, action="append", required=True,
                        metavar=("NAME", "DIFFER", "CONTROL"),
                        help="an arm: its name, its episodes file on policy_shift_differ, and the one on "
                             "policy_shift_control")
    parser.add_argument("--scenarios", type=Path, required=True, help="the scenario folder of the rollouts")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file of the rollouts")
    parser.add_argument("--out", type=Path, required=True, help=f"folder for {JSON_NAME} and {MARKDOWN_NAME}")
    parser.add_argument("--draws", type=int, default=second_desk.N_DRAWS, help="bootstrap draws")
    parser.add_argument("--seed", type=int, default=second_desk.SEED, help="bootstrap seed")
    parser.add_argument("--allow-other-coverage", action="store_true",
                        help="take files that do not have 8 attempts per scenario and run seed 0; the "
                             "table then says so")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the two files and print the table. Returns the exit code: 0, or 2 if nothing was written."""
    args = parse_args(argv)
    arms = [(name, Path(differ), Path(control)) for name, differ, control in args.arm]
    try:
        report = build_report(arms, args.scenarios, args.text, n_draws=args.draws, seed=args.seed,
                              allow_other_coverage=args.allow_other_coverage)
    except ValueError as error:
        print(f"second_desk_report: {error}", file=sys.stderr)
        return 2
    markdown = second_desk.to_markdown(report)
    args.out.mkdir(parents=True, exist_ok=True)
    as_json = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (args.out / JSON_NAME).write_text(as_json, encoding="utf-8")
    (args.out / MARKDOWN_NAME).write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"wrote {args.out / JSON_NAME} and {args.out / MARKDOWN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
