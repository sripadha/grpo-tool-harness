"""Check a file of saved episodes before a number is computed from it: replay every record, grade it again.

    .venv/bin/python scripts/replay_check.py --episodes results/base_dev/episodes.jsonl \\
        --scenarios data/scenarios --split dev

For every record of the file (a record with a harness error is counted and skipped):
  1. The record must be of these files: its family is in the split file, and it names the hash of the
     text file and of the split file.
  2. Its labels (split, outcome, style, scenario id, turn budget) must be those of the family file, and the
     expected state in the family file must be what accessdesk/env/rules.py gives.
  3. The saved calls are run again on a fresh world (accessdesk/harness/records.py, replay). Every tool
     result must be the saved one, and the final state must be the saved final state.
  4. The saved assistant texts are given to a fresh episode of the harness. Every turn that the harness
     makes must be the saved turn: the same parse result, the same call, the same tool result.
  5. The counters of the record (turns, malformed messages, tool errors, closed) must be those of its turns.
  6. The replayed state is graded again with the MAIN and with the SECOND version of verify and of reward
     (accessdesk/impl.py). All four answers must be the saved `passed` and `reward`.
  7. The record names the hash of the config that made it (config_sha256). The config that this check
     grades by must have that hash. So a reward number in `settings` cannot be changed together with the
     reward. The chunk size of the run is not a setting of a record: the hash is tried with every size.

A record is graded as it says it was made: under its own rule set and tool set, and with the config of
accessdesk/config.py in which the record's `settings` (turn limit, token cap, parser mode, sampling, the
six reward numbers) replace the defaults. --ruleset NAME adds one demand: every record is of that rule set.

The text file must be the one of the run (default: text/model_text.yaml), and the scenario folder must
still be the folder that the build wrote: the hashes in its summary.json are compared first.

What this check cannot see:
  - A record that was taken out of the file, and a failing record that was relabelled as a harness error
    (those are counted and not checked). The line "scenarios" of the report shows whether every scenario
    has the same number of records. Compare the counts with run_manifest.json of the run; the results
    table compares the arms (accessdesk/eval/tables.py).
  - The flags `truncated` and `n_tokens` of a turn. They are taken from the record as they are; only
    tokens.jsonl of the run can confirm them. accessdesk/eval/monitors.py reads these two fields.
  - A malformed turn that was removed, if the counters and the reward were changed to fit.

Printed: the counts, the settings that the records name (compare them with run_config in MANIFEST.json),
the differences by kind, and the first ten differences.
Exit code 0: no difference. 1: at least one difference. 2: the check could not be made (a wrong file, or
no record to check). No file is written or changed.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env import rules  # noqa: E402
from accessdesk.env.generator import SPLITS  # noqa: E402
from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.harness.episode import SCHEMA, Episode  # noqa: E402
from accessdesk.harness.records import read_records, replay  # noqa: E402
from accessdesk.text import TextBundle, load_text, model_text_path  # noqa: E402

SHOWN = 10                        # differences that are printed
VERSIONS = ("main", "second")     # the two versions of verify and of reward
MAX_CHUNK_SIZE = 4096             # the largest chunk size that the config hash of a record is tried with
Difference = tuple[str, str]      # (kind, what differs)


def sha256_of_file(path: Path) -> str:
    """The SHA-256 of the file's bytes, as 64 hexadecimal digits."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folder_problem(scenarios_dir: Path, split: str, text: TextBundle) -> str | None:
    """Why this scenario folder cannot be used with this text file, or None if it can. The meaning is
    that of scripts/gpu/rollout.py (check_scenarios): summary.json names the hash of the text file, and
    the split file is still the file that the build wrote."""
    summary_file = scenarios_dir / "summary.json"
    split_file = scenarios_dir / f"{split}.jsonl"
    if not summary_file.is_file():
        return f"{summary_file} is missing"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary["text_sha256"] != text.sha256:
        return (f"{scenarios_dir} was built from another text file than {text.path} (summary.json has "
                "another text_sha256); give the text file of the run with --text")
    if not split_file.is_file():
        return f"{split_file} is missing"
    if sha256_of_file(split_file) != summary.get("files_sha256", {}).get(split_file.name):
        return f"{split_file} is not the file that the build wrote (summary.json names another SHA-256)"
    return None


def second_versions_problem() -> str | None:
    """Why the second versions cannot grade, or None: both must be written (accessdesk/_second)."""
    missing = [name for name in ("verify", "reward") if not impl.second_is_written(name)]
    return f"the second version of {' and '.join(missing)} is not written" if missing else None


@functools.lru_cache(maxsize=64)
def possible_config_hashes(cfg: RunConfig, text_is_placeholder: bool) -> frozenset[str]:
    """The hash of every config that a run script can make from `cfg` without a change of a setting.

    Two values of a run are not among the settings of its records, so they are tried:
      - the chunk size (scripts/gpu/rollout.py --chunk-size): how many episodes one generate call holds.
        It changes no reward. Every size from 1 to MAX_CHUNK_SIZE is tried.
      - for a placeholder text, the placeholder seeds (scripts/run_scripted.py makes its config with them).
    """
    bases = {cfg, cfg.with_seeds_for(text_is_placeholder)}
    return frozenset(dataclasses.replace(base, chunk_size=size).sha256()
                     for base in bases for size in range(1, MAX_CHUNK_SIZE + 1))


def config_hash_differences(record: dict, cfg: RunConfig, text: TextBundle) -> list[Difference]:
    """The record names the hash of the config that made it. The config that this check grades by (`cfg`:
    accessdesk/config.py with the record's settings) must have that hash, with any chunk size (see
    possible_config_hashes).

    A difference means: a setting of the record was changed after the run, or accessdesk/config.py is
    not the config of the run any more (for example the data seed was changed after the run)."""
    if record["config_sha256"] in possible_config_hashes(cfg, text.is_placeholder):
        return []
    return [("settings", f"the record names the config hash {str(record['config_sha256'])[:12]}; "
                         f"accessdesk/config.py with the settings of the record has {cfg.sha256()[:12]} "
                         "(tried with every chunk size)")]


def config_of(record: dict) -> RunConfig:
    """The config that the record says it was made with: the config of accessdesk/config.py, with the
    record's `settings` in place of the defaults. A record without `settings` gets the defaults.
    Raises TypeError if `settings` names a field that the config does not have."""
    return dataclasses.replace(RunConfig(), **(record.get("settings") or {}))


# ---------------------------------------------------------------------------
# The checks of one record. Each returns its differences; an empty list means: nothing differs.
# ---------------------------------------------------------------------------
def source_differences(record: dict, families: dict[str, dict], text: TextBundle, data_sha256: str,
                       ruleset: str | None) -> list[Difference]:
    """Is the record of these files? Its family, its text hash, its data hash (if it has one) and its rule
    set (a known one, and the one asked for with --ruleset)."""
    found = []
    if record["family_id"] not in families:
        found.append(("family", f"family {record['family_id']} is not in the split file"))
    if record["text_sha256"] != text.sha256:
        found.append(("text", f"the record names the text hash {record['text_sha256'][:12]}, the text "
                              f"file has {text.sha256[:12]}"))
    if record.get("data_sha256", data_sha256) != data_sha256:
        found.append(("data", f"the record names the data hash {record['data_sha256'][:12]}, the split "
                              f"file has {data_sha256[:12]}"))
    if record["ruleset"] not in rules.RULESETS:
        found.append(("ruleset", f"unknown rule set {record['ruleset']!r}"))
    elif ruleset is not None and record["ruleset"] != ruleset:
        found.append(("ruleset", f"the record is of the rule set {record['ruleset']}, not of {ruleset}"))
    return found


def label_differences(record: dict, family: dict, text: TextBundle) -> list[Difference]:
    """The labels that the metrics and the tables group by, and the turn budget that the reward uses: each
    must be what the family file and the text file say."""
    ruleset, style = record["ruleset"], record["style"]
    wanted = {"split": family["split"], "outcome": family["outcome"],
              "scenario_id": f"{family['family_id']}/s{style}", "turn_budget": family["turn_budget"][ruleset],
              "text_is_placeholder": text.is_placeholder}
    wrong = [f"{key}: saved {record[key]!r}, the files give {value!r}"
             for key, value in wanted.items() if record[key] != value]
    if [wording["style"] for wording in family["wordings"]].count(style) != 1:
        wrong.append(f"style: the family has no wording in style {style!r}")
    return [("labels", "; ".join(wrong))] if wrong else []


def expected_differences(family: dict, ruleset: str) -> list[Difference]:
    """The expected state and the turn budget in the family file must be what the rules give now."""
    same = (rules.expected_state(family, ruleset) == family["expected"][ruleset]
            and rules.turn_budget(family, ruleset) == family["turn_budget"][ruleset])
    if same:
        return []
    return [("expected", f"the expected state or the turn budget of {family['family_id']} in the split "
                         f"file is not what accessdesk/env/rules.py gives for the rule set {ruleset}")]


def turn_differences(record: dict, family: dict, text: TextBundle, cfg: RunConfig) -> list[Difference]:
    """The saved assistant texts, given one by one to a fresh episode of the harness. Each turn that the
    harness makes must be the saved turn in every field, and the texts must end the episode."""
    try:
        episode = Episode(family, record["style"], text, cfg, ruleset=record["ruleset"],
                          toolset=record["toolset"], prompt=record["prompt"])
    except ValueError as error:
        return [("labels", f"the harness makes no episode for this record: {error}")]
    for saved in record["turns"]:
        if episode.done:
            return [("turns", f"turn {saved['turn']} comes after the end of the episode")]
        made = episode.step(saved["assistant_text"], truncated=saved["truncated"], n_tokens=saved["n_tokens"])
        fields = [key for key in sorted(set(made) | set(saved)) if made.get(key) != saved.get(key)]
        if fields or list(made) != list(saved):
            return [("turns", f"turn {made['turn']}: for the saved text the harness gives another "
                              f"{', '.join(fields) or 'order of the fields'}")]
    if not episode.done:
        return [("turns", f"the {len(record['turns'])} saved turns do not end the episode: no close, "
                          f"and the turn limit is {cfg.turn_limit}")]
    return []


def counter_differences(record: dict, state: dict) -> list[Difference]:
    """The counters that the reward and the metrics read must be those of the saved turns and of the
    replayed state."""
    turns = record["turns"]
    closed = state["request"]["resolution"] is not None
    wanted = {"n_turns": len(turns),
              "n_malformed": sum(1 for turn in turns if not turn["parse_ok"]),
              "n_tool_errors": sum(1 for turn in turns if turn["tool_error"] is not None),
              "closed": closed, "ended_by": "close" if closed else "turn_limit"}
    wrong = [f"{key}: saved {record[key]!r}, the turns give {value!r}"
             for key, value in wanted.items() if record[key] != value or type(record[key]) is not type(value)]
    return [("counters", "; ".join(wrong))] if wrong else []


def grade(state: dict, family: dict, record: dict, cfg: RunConfig) -> dict[str, tuple[bool, int]]:
    """version -> (passed, reward) for the replayed state: verify and reward of that version. The counts
    that the reward gets are taken from the saved turns, not from the saved counters."""
    ruleset = record["ruleset"]
    n_malformed = sum(1 for turn in record["turns"] if not turn["parse_ok"])
    graded = {}
    for version in VERSIONS:
        get = impl.main_version if version == "main" else impl.second_version
        passed = get("verify")(state, family["expected"][ruleset])
        graded[version] = (passed, get("reward")(passed, n_malformed, len(record["turns"]),
                                                 family["turn_budget"][ruleset], cfg))
    return graded


def grade_differences(record: dict, graded: dict[str, tuple[bool, int]]) -> list[Difference]:
    """The saved `passed` and `reward` must be the answers of both versions, in value and in type."""
    found = []
    for position, key in enumerate(("passed", "reward")):
        answers = {version: graded[version][position] for version in VERSIONS}
        if any(answer != record[key] or type(answer) is not type(record[key]) for answer in answers.values()):
            again = ", ".join(f"{version} {answer!r}" for version, answer in answers.items())
            found.append((key, f"saved {record[key]!r}; graded again: {again}"))
    return found


def check_record(record: dict, families: dict[str, dict], text: TextBundle, data_sha256: str,
                 ruleset: str | None = None) -> list[Difference]:
    """Every difference of one record (not a harness-error record): a list of (kind, what differs).

    Kinds: family, text, data, ruleset (the record is not of these files; nothing else is checked then),
    labels, expected, settings, replay, final_state, turns, counters, passed, reward, and record for a
    record of another schema or one that lacks a field. After a failed replay there is no state, so the
    checks that need one are left out."""
    try:
        if record["schema"] != SCHEMA:
            return [("record", f"the record has the schema {record['schema']!r}; this check knows {SCHEMA}")]
        found = source_differences(record, families, text, data_sha256, ruleset)
        if found:
            return found
        family = families[record["family_id"]]
        found = label_differences(record, family, text) + expected_differences(family, record["ruleset"])
        try:
            cfg = config_of(record)
        except TypeError as error:
            return found + [("settings", f"the settings of the record are not fields of the config: {error}")]
        found += config_hash_differences(record, cfg, text)
        try:
            state = replay(record, family, text, cfg)
        except ValueError as error:
            return found + [("replay", str(error))]
        if state != record["final_state"]:
            parts = [part for part in state if state[part] != record["final_state"].get(part)]
            found.append(("final_state",
                          f"the replayed state differs from the saved one in: {', '.join(parts)}"))
        found += turn_differences(record, family, text, cfg)
        found += counter_differences(record, state)
        return found + grade_differences(record, grade(state, family, record, cfg))
    except (KeyError, TypeError, AttributeError, RuntimeError, ValueError) as error:
        return [("record", f"the record cannot be read: {type(error).__name__} {error}")]


# ---------------------------------------------------------------------------
# The whole file
# ---------------------------------------------------------------------------
def check_records(records: list[dict], families: list[dict], text: TextBundle, data_sha256: str,
                  ruleset: str | None = None) -> dict:
    """The report of a list of records, in file order:

        records            how many records the file has
        harness_errors     records with a harness error: counted here, and not checked
        checked            records - harness_errors
        with_difference    checked records with at least one difference
        by_kind            kind -> number of differences of that kind
        settings           the `settings` of the checked records, as JSON text -> number of records
        per_scenario       the numbers of records that a scenario has, in order (one number: every
                           scenario has as many records as every other); harness errors are counted in
        differences        every difference: {"line", "scenario_id", "sample", "kind", "detail"};
                           `line` is the line of the record in the file, from 1
    """
    by_id = {family["family_id"]: family for family in families}
    per_scenario = Counter(record.get("scenario_id") for record in records)
    report: dict = {"records": len(records), "harness_errors": 0, "checked": 0, "with_difference": 0,
                    "by_kind": {}, "settings": {}, "per_scenario": sorted(set(per_scenario.values())),
                    "n_scenarios": len(per_scenario), "differences": []}
    for line, record in enumerate(records, start=1):
        if record.get("harness_error") is not None:
            report["harness_errors"] += 1
            continue
        found = check_record(record, by_id, text, data_sha256, ruleset)
        report["checked"] += 1
        report["with_difference"] += bool(found)
        named = json.dumps(record.get("settings"), sort_keys=True)
        report["settings"][named] = report["settings"].get(named, 0) + 1
        for kind, detail in found:
            report["by_kind"][kind] = report["by_kind"].get(kind, 0) + 1
            report["differences"].append({"line": line, "scenario_id": record.get("scenario_id"),
                                          "sample": record.get("sample"), "kind": kind, "detail": detail})
    return report


def coverage_lines(report: dict) -> list[str]:
    """One line: how many scenarios the file has and how many records each one has. This is information,
    not a difference: a run may cover a part of a split. An uneven number can mean a removed record."""
    sizes = report["per_scenario"]
    if len(sizes) <= 1:
        each = f"each with {sizes[0]} record(s)" if sizes else "no record"
        return [f"scenarios          {report['n_scenarios']}, {each}"]
    return [f"scenarios          {report['n_scenarios']}, with {sizes[0]} to {sizes[-1]} records each: "
            "NOT EVEN, records may be missing"]


def settings_lines(report: dict) -> list[str]:
    """The settings that the checked records name, each with its number of records."""
    lines = [f"settings           {len(report['settings'])} kind(s) among the checked records"]
    return lines + [f"  {count} record(s): {named}" for named, count in report["settings"].items()]


def report_lines(report: dict, shown: int = SHOWN) -> list[str]:
    """The report as lines of text: the counts, then the first `shown` differences."""
    lines = [f"records            {report['records']}",
             f"harness errors     {report['harness_errors']}   (counted, not checked)",
             f"checked            {report['checked']}",
             f"with a difference  {report['with_difference']}"]
    lines += coverage_lines(report) + settings_lines(report)
    differences = report["differences"]
    if not differences:
        return lines + ["no difference: each checked record has the final state, the turns, the pass and "
                        "the reward that its saved calls and texts give (verify and reward in both versions)"]
    kinds = ", ".join(f"{kind} {count}" for kind, count in report["by_kind"].items())
    lines.append(f"differences        {len(differences)}   by kind: {kinds}")
    lines.append(f"the first {min(shown, len(differences))} of {len(differences)} differences:")
    lines += [f"  line {d['line']}  {d['scenario_id']} sample {d['sample']}  {d['kind']}: {d['detail']}"
              for d in differences[:shown]]
    return lines


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--episodes", type=Path, required=True, help="the file of episode records")
    parser.add_argument("--scenarios", type=Path, required=True, help="the scenario folder of the run")
    parser.add_argument("--split", choices=SPLITS, required=True, help="the split of the run")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file of the run")
    parser.add_argument("--ruleset", choices=rules.RULESETS, default=None,
                        help="the rule set that every record must be of (default: each record's own)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check one file of episodes and print the report. Returns the exit code (see the module text)."""
    args = parse_args(argv)
    text = load_text(args.text)
    problem = folder_problem(args.scenarios, args.split, text) or second_versions_problem()
    if problem:
        print(f"replay_check: {problem}", file=sys.stderr)
        return 2
    split_file = args.scenarios / f"{args.split}.jsonl"
    report = check_records(read_records(args.episodes), load_families(split_file), text,
                           sha256_of_file(split_file), args.ruleset)
    print(f"replay check of {args.episodes}")
    print(f"scenarios {args.scenarios}, split {args.split}; text {args.text}")
    print("\n".join(report_lines(report)))
    if report["checked"] == 0:
        print("replay_check: no record was checked", file=sys.stderr)
        return 2
    return 1 if report["differences"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
