"""The audit by hand (ANALYSIS_PLAN.md section 8, lines 1 and 3): make the blind reading sheet, the labels
form and the key, and check the filled form. Written after the freeze; it only reads saved attempts.

    .venv/bin/python scripts/audit_sheet.py make --scenarios data/scenarios --split dev \\
        --arm grpo_seed1 results/main/grpo_seed1_dev/episodes.jsonl \\
        --arm grpo_seed2 results/main/grpo_seed2_dev/episodes.jsonl \\
        --seed 0 --out results/main/audit
    .venv/bin/python scripts/audit_sheet.py check --form results/main/audit/audit_labels.yaml

make. From the dev episodes of the arms (pooled) it takes two samples, by the rules of
accessdesk/audit/sample.py: 5 top-reward attempts of each expected outcome and every top-reward attempt
that m1, m2 or m3 flags; and 3 failed attempts of each expected outcome. Both samples stand in ONE seeded
order under neutral ids (A001, A002, ...). The draw depends on the seed and on the identity of an attempt
(arm name, scenario id, sample number), so the arm names are part of it. The arms must be of one rule
set: the sheet prints the rules of the desk once. It writes into --out:
    audit_sheet.md       the reading sheet: the rules, then each attempt with its request, the facts of
                         the world, the messages with the tools' answers, and the final state. It shows
                         no verdict, no reward, no monitor and no outcome, and names no arm or scenario
    audit_labels.yaml    the form that a person fills: for every attempt `end_state_right`,
                         `path_acceptable` (yes or no) and a note
    audit_key.json       which attempt each id is, its kind, why it is in the sheet, the verdict, the
                         reward, the flags of the monitors and the right end state; and how many attempts
                         there were to take from. The person does not open it before the form is filled
A labels form that is already there is never written over: it may hold answers. What `make` prints tells
only the number of attempts.

check. It reads the filled form and the key beside it (the form names the key file and its SHA-256),
finds the files that the key names, makes the draw again, and prints: how often `end_state_right` agrees
with the verifier, for the top-reward sample and the reverse sample apart; every disagreement with its
attempt; the paths that the person does not accept among the top-reward attempts, beside the flags of
the monitors; and for each attempt the verdict of the second verifier and of the replay
(scripts/replay_check.py). --json FILE also writes the report as JSON.
While a label is empty, check prints only which labels are empty: the report names the verdict of every
attempt, and the person must not read it before they have judged. Such a check is no run. Each run on a
full form is added to a log beside the form (FORM.checks.log, by `record_run` of scripts/hand_states.py)
with its number, the time and the SHA-256 of the form. The first run is the one that counts: a label
that is changed after the key was read is not a blind label, and the log shows that the form changed.

Exit code of check: 0 if every label is answered and the automatic checks give the saved verdict of
every sampled attempt; 1 if a label is empty or an automatic check differs; 2 if the check could not be
made (the form or the key cannot be read, the key is not the key of the form, a file changed since the
sheet was made, or the attempts of the key are not the sample that its seed draws). A disagreement of
the person with the verifier is a finding: it is listed, and the code stays 0. make ends with 0, or with
2 if it wrote nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.audit import Run, load_run, load_script, sample, sheet  # noqa: E402
from accessdesk.text import model_text_path  # noqa: E402

SHEET_NAME, FORM_FILE, KEY_NAME = "audit_sheet.md", "audit_labels.yaml", "audit_key.json"


def load_arms(arms: list[tuple[str, Path]], scenarios: Path, split: str,
              text_path: Path) -> tuple[dict[str, Run], list[sample.Attempt], str]:
    """(arm name -> its run, the attempts of all arms, the one rule set of the attempts). Raises
    ValueError if a name is given twice, if the arms hold no valid attempt or attempts of two rule sets,
    or for what `load_run` and `sample.attempts_of` refuse."""
    names = [name for name, _ in arms]
    if not arms or len(set(names)) != len(names):
        raise ValueError(f"give each arm once, with its own name: {names}")
    runs = {name: load_run(episodes, scenarios, split, text_path) for name, episodes in arms}
    attempts = [attempt for name, run in runs.items()
                for attempt in sample.attempts_of(name, run.records, run.families, run.text)]
    rulesets = sorted({attempt.record["ruleset"] for attempt in attempts})
    if not rulesets:
        raise ValueError("the arms hold no valid attempt")
    if len(rulesets) > 1:
        raise ValueError("the sheet prints the rules of the desk once, so the arms must be of one rule "
                         f"set, not of {', '.join(rulesets)}")
    return runs, attempts, rulesets[0]


def short_outcomes(population: dict) -> list[str]:
    """One line for each outcome that has fewer attempts than a draw takes: the sample is then smaller
    than 20 or 12, and the reader of the numbers must know it."""
    lines = []
    for outcome, counts in population.items():
        for key, wanted in (("top_reward", sample.TOP_PER_OUTCOME), ("failed", sample.FAILED_PER_OUTCOME)):
            if counts[key] < wanted:
                lines.append(f"NOTE: the outcome {outcome} has only {counts[key]} {key} attempts; the draw "
                             f"takes {wanted}, so all {counts[key]} are in the sample")
    return lines


def meta_of(runs: dict[str, Run], attempts: list[sample.Attempt], ruleset: str, scenarios: Path, split: str,
            text_path: Path, seed: int) -> dict:
    """What the key remembers beside the attempts: the files with their hashes, so that the check finds
    them again and sees a change; the seed and the rule set; and what the samples were drawn from."""
    shown = load_script("monitor_report").shown_path
    first = next(iter(runs.values()))
    population = sample.population(attempts)
    return {"seed": seed, "split": split, "ruleset": ruleset, "scenarios": shown(scenarios),
            "split_file_sha256": first.data_sha256, "text_file": shown(text_path),
            "text_sha256": first.text.sha256, "sheet_name": SHEET_NAME, "form_name": FORM_FILE,
            "arms": [{"name": name, "episodes": shown(run.episodes), "episodes_sha256": run.episodes_sha256}
                     for name, run in runs.items()],
            "per_outcome": {"top_reward_drawn": sample.TOP_PER_OUTCOME,
                            "failed_drawn": sample.FAILED_PER_OUTCOME},
            "flag_monitors": list(sample.FLAG_MONITORS),
            "population": population, "notes": short_outcomes(population)}


def make(args: argparse.Namespace) -> int:
    """Write the sheet, the form and the key. Returns the exit code."""
    form_file = args.out / FORM_FILE
    if form_file.exists():
        print(f"audit_sheet: {form_file} is there already and may hold answers; choose another --out",
              file=sys.stderr)
        return 2
    try:
        runs, attempts, ruleset = load_arms([(name, Path(episodes)) for name, episodes in args.arm],
                                            args.scenarios, args.split, args.text)
        chosen = sample.choose(attempts, args.seed)
        key = sheet.key_of(chosen, meta_of(runs, attempts, ruleset, args.scenarios, args.split, args.text,
                                           args.seed))
        key_bytes = sheet.key_text(key).encode("utf-8")
        shown = load_script("monitor_report").shown_path
        names = {"sheet_name": SHEET_NAME, "form_name": FORM_FILE, "key_name": KEY_NAME, "ruleset": ruleset,
                 "key_sha256": hashlib.sha256(key_bytes).hexdigest(),
                 "check_command": f".venv/bin/python scripts/audit_sheet.py check --form {shown(form_file)}"}
        sheet_text, form_text = sheet.sheet_markdown(chosen, names), sheet.form_text(chosen, names)
    except ValueError as error:
        print(f"audit_sheet: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / KEY_NAME).write_bytes(key_bytes)
    (args.out / SHEET_NAME).write_text(sheet_text, encoding="utf-8")
    form_file.write_text(form_text, encoding="utf-8")
    print(f"{len(chosen)} attempts are in the sheet, in one seeded order (seed {args.seed}) under neutral "
          "ids")
    print(f"wrote {args.out / SHEET_NAME}, {form_file} and {args.out / KEY_NAME}")
    print(f"Do not open {KEY_NAME} before the form is filled: it says which attempt each id is.")
    return 0


def found_path(named: str) -> Path:
    """A path of the key as a file path: as it is if it is absolute or exists from here, else from the
    repository root (the key holds paths from the root for files inside the repository)."""
    path = Path(named)
    return path if path.is_absolute() or path.exists() else ROOT / path


def key_of_form(form: dict, form_file: Path, key_file: Path | None) -> dict:
    """The key that the form names: the file beside the form (or `key_file`), with the SHA-256 that the
    form holds. Raises ValueError for another file, OSError if it cannot be read."""
    path = key_file if key_file is not None else form_file.with_name(str(form["key_file"]))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != form["key_sha256"]:
        raise ValueError(f"{path} is not the key of this form: its SHA-256 is not the one that the form "
                         "names (the key or the form was changed after `make`)")
    return json.loads(raw.decode("utf-8"))


def runs_of_key(key: dict) -> tuple[dict[str, Run], list[sample.Attempt]]:
    """The runs and the attempts of the files that the key names. Raises ValueError if a file is not
    the file that the sheet was made from (its SHA-256 differs)."""
    arms = [(arm["name"], found_path(arm["episodes"])) for arm in key["arms"]]
    runs, attempts, _ = load_arms(arms, found_path(key["scenarios"]), key["split"],
                                  found_path(key["text_file"]))
    for arm in key["arms"]:
        run = runs[arm["name"]]
        hashes = (("the episodes file", run.episodes_sha256, arm["episodes_sha256"]),
                  ("the split file", run.data_sha256, key["split_file_sha256"]),
                  ("the text file", run.text.sha256, key["text_sha256"]))
        changed = [what for what, now, then in hashes if now != then]
        if changed:
            raise ValueError(f"arm {arm['name']}: {' and '.join(changed)} changed since the sheet was made")
    return runs, attempts


def sample_lines(key: dict) -> list[str]:
    """What the key says of the sample as a whole: its size by kind and by reason, and the notes on
    outcomes that had fewer attempts than a draw takes. Printed by `check`, after the form is filled."""
    entries = key["attempts"]
    why = [entry["why"] for entry in entries if entry["kind"] == sample.TOP]
    n_failed = sum(1 for entry in entries if entry["kind"] == sample.FAILED)
    return [f"the sample      {len(entries)} attempts in one seeded order (seed {key['seed']}): {len(why)} "
            f"top-reward ({why.count(sample.DRAWN)} drawn, {why.count(sample.BOTH)} drawn and flagged, "
            f"{why.count(sample.FLAGGED)} flagged only) and {n_failed} failed",
            f"drawn from      {json.dumps(key['population'])}", *key["notes"]]


def check(args: argparse.Namespace) -> int:
    """Check the filled form against its key and print the report. Returns the exit code."""
    try:
        problem = load_script("replay_check").second_versions_problem()
        if problem:
            raise ValueError(problem)
        form_bytes = args.form.read_bytes()
        form = sheet.read_form(form_bytes.decode("utf-8"))
        key = key_of_form(form, args.form, args.key)
        runs, attempts = runs_of_key(key)
        chosen = sample.choose(attempts, key["seed"])
        if json.loads(json.dumps([sheet.key_entry(sampled) for sampled in chosen])) != key["attempts"]:
            raise ValueError(f"the key does not hold the sample that its seed draws from these files "
                             f"(seed {key['seed']}): it was changed, or the files are not those of `make`")
        report = sheet.check_form(form, chosen, runs)
    except (OSError, KeyError, ValueError) as error:
        print(f"audit_sheet: the check could not be made: {error}", file=sys.stderr)
        return 2
    print(f"audit check of {args.form}")
    if report["n_blank"]:
        # No word of the key is printed, no report file is written, and it is no run in the log.
        print("\n".join(sheet.blank_lines(report)))
        return 1
    report["sample"] = {"seed": key["seed"], "n_sampled": key["n_sampled"], "population": key["population"],
                        "notes": key["notes"]}
    lines = sample_lines(key) + sheet.report_lines(report)
    form_sha256 = hashlib.sha256(form_bytes).hexdigest()          # the form as it was checked
    print(load_script("hand_states").record_run(args.form, form_sha256, lines))
    print("\n".join(lines))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return sheet.exit_code(report)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("make", help="write the reading sheet, the empty labels form and the key")
    making.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "EPISODES"),
                        help="an arm: its name and its episodes file; give it once per arm")
    making.add_argument("--scenarios", type=Path, required=True, help="the scenario folder of the runs")
    making.add_argument("--split", required=True, help="the split of the runs, for example dev")
    making.add_argument("--text", type=Path, default=model_text_path(), help="the text file of the runs")
    making.add_argument("--seed", type=int, default=0, help="the seed of the two draws and of the order")
    making.add_argument("--out", type=Path, required=True, help="the folder for the three files")
    checking = commands.add_parser("check", help="check the filled labels form against its key")
    checking.add_argument("--form", type=Path, required=True, help="the filled labels form")
    checking.add_argument("--key", type=Path, default=None,
                          help="the key file (default: the file beside the form that the form names)")
    checking.add_argument("--json", type=Path, default=None, help="also write the report to this file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return make(args) if args.command == "make" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
