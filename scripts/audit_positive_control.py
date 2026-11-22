"""The positive control of the audit (ANALYSIS_PLAN.md section 8, line 5): three verifiers that are broken
on purpose must each be flagged, and the real verifier must not. Written after the freeze; it only reads
saved attempts, and it plays scripted probe attempts in memory.

    .venv/bin/python scripts/audit_positive_control.py \\
        --episodes results/main/base_dev/episodes.jsonl results/main/sft_dev/episodes.jsonl \\
        --scenarios data/scenarios --split dev --out results/main/audit/positive_control

Two sets of attempts are graded again with each verifier of accessdesk/audit/broken.py and with the real
one. Then the audit's automatic comparison (the replay and the second verifier of
scripts/replay_check.py) runs against those verdicts. What "flagged" means is in that module.
  probes   every scripted policy of accessdesk/harness/policies.py (`named_policies`) on the first family
           of each expected outcome of the split, in file order. They are made in memory, under the rule
           set, the tool set, the prompt and the settings of the first episodes file. So the control does
           not depend on what a model happened to do.
  saved    the saved attempts of --episodes: one file or several of one scenario folder, split and text
           file.

The control passes if each broken verifier is flagged on the probes and the real verifier is flagged
nowhere (on no probe and on no saved attempt). How often each broken verifier is flagged on the saved
attempts is reported apart; that number may be 0 and is not a failure.

Written into --out: positive_control.json and positive_control.md. The markdown is also printed.
Exit code 0: the control passes. 1: it fails (a broken verifier was not flagged on the probes, or the
real verifier was flagged). 2: the control could not be made (the files do not belong together, no valid
attempt, or the probes cannot be played).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.audit import broken, load_run, load_script  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.text import model_text_path  # noqa: E402

JSON_NAME, MARKDOWN_NAME = "positive_control.json", "positive_control.md"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--episodes", type=Path, required=True, nargs="+",
                        help="the episodes file of one run; several files are taken together")
    parser.add_argument("--scenarios", type=Path, required=True, help="the scenario folder of the run")
    parser.add_argument("--split", required=True, help="the split of the run, for example dev")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file of the run")
    parser.add_argument("--out", type=Path, required=True, help=f"folder for {JSON_NAME} and {MARKDOWN_NAME}")
    return parser.parse_args(argv)


def file_lines(files: list[dict]) -> list[str]:
    """One markdown line per file: how many attempts each verifier got flagged in it."""
    lines = []
    for one in files:
        flagged = ", ".join(f"{name} {row['n_flagged']}" for name, row in one["broken"].items())
        lines.append(f"- `{one['episodes']}`: {one['real']['n_attempts']} valid attempts; flagged: the real "
                     f"verifier {one['real']['n_flagged']}, {flagged}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run the control on the files, write and print the report. Returns the exit code."""
    args = parse_args(argv)
    replay_check = load_script("replay_check")
    shown = load_script("monitor_report").shown_path
    files, probes = [], None
    try:
        problem = replay_check.second_versions_problem()
        if problem or len(set(args.episodes)) != len(args.episodes):
            raise ValueError(problem or "an episodes file is given twice")
        for episodes in args.episodes:
            run = load_run(episodes, args.scenarios, args.split, args.text)
            if not metrics.valid_records(run.records):
                raise ValueError(f"{episodes} has no valid attempt")
            if probes is None:              # the probes are played under the set-up of the first file
                probes = broken.probe_comparisons(run)
            files.append({"episodes": shown(episodes), "episodes_sha256": run.episodes_sha256,
                          "split_file_sha256": run.data_sha256, "text_sha256": run.text.sha256,
                          **broken.verifier_comparisons(run.records, run)})
    except (ValueError, RuntimeError) as error:     # RuntimeError: a scripted policy could not be played
        print(f"audit_positive_control: {error}", file=sys.stderr)
        return 2
    report = {"what": "positive control, ANALYSIS_PLAN.md section 8, line 5",
              "scenarios": shown(args.scenarios), "split": args.split,
              **broken.positive_control(files, probes)}
    source = "`, `".join(one["episodes"] for one in files)
    markdown = broken.to_markdown(report, source) + "\n".join(["", "By file:", *file_lines(files)]) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    as_json = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (args.out / JSON_NAME).write_text(as_json, encoding="utf-8")
    (args.out / MARKDOWN_NAME).write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"wrote {args.out / JSON_NAME} and {args.out / MARKDOWN_NAME}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
