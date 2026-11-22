"""The monitors outside the reward, for one saved run: a table and example episodes, printed as markdown.

    .venv/bin/python scripts/monitor_report.py \\
        --episodes results/calibration/round3/4b_mixed/episodes.jsonl \\
        --scenarios data/scenarios_calib3_mixed --split calib \\
        --text results/drafting/model_text_round3_sha1ebc268f.yaml

--episodes   the episodes.jsonl file of ONE run (one episode record per line)
--scenarios  the folder of the family files that the run was made from
--split      the split of the run: the families are read from <scenarios>/<split>.jsonl, and only the
             records of this split are used
--text       the text file of the run (default: text/model_text.yaml); the monitors read ids from the
             saved tool results under the result keys of this file
--examples   how many example episodes are printed for each monitor (default 3). With 0 the example
             lines are left out. A negative number is refused.

No model and no GPU are used, and no episode is run again: every number comes from the saved records,
through `accessdesk/eval/monitors.py`, where each monitor is defined.

The script prints why and stops with exit code 2 if the numbers could not be trusted: the file holds more
than one run or set-up, the family file or the text file is not the one that the run was made from, or
the verifier does not give the saved pass or fail for a saved final state and its family record.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.eval import monitors  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402
from accessdesk.text import TextBundle, TextError, load_text, model_text_path  # noqa: E402

SETUP_KEYS = ("run", "arm", "ruleset", "toolset", "prompt")     # one value of each in the records of one run


def sha256_of_file(path: Path) -> str:
    """The SHA-256 of the file's bytes, as 64 hexadecimal digits."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown_path(path: Path) -> str:
    """The path as the report shows it: relative to the repository root, with "/", if the file is inside
    the repository; else as it was typed."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def records_of_split(episodes: Path, split: str) -> tuple[list[dict], int]:
    """The records of the file that belong to `split`, and the number of records of other splits.

    Raises ValueError if the file is missing, has no record of the split, or holds more than one run or
    set-up (the monitors, like the metrics, take the records of ONE arm)."""
    if not episodes.is_file():
        raise ValueError(f"{episodes} does not exist")
    records = read_records(episodes)
    chosen = [record for record in records if record["split"] == split]
    if not chosen:
        raise ValueError(f"{episodes} has no record of the split {split!r}")
    setups = sorted({tuple(str(record[key]) for key in SETUP_KEYS) for record in chosen})
    if len(setups) > 1:
        raise ValueError(f"{episodes} holds {len(setups)} different runs or set-ups {SETUP_KEYS}: {setups}")
    return chosen, len(records) - len(chosen)


def families_of_run(scenarios: Path, split: str, records: list[dict]) -> tuple[list[dict], str]:
    """The family records of the split, and one sentence that says how the file was checked.

    The split file must have the SHA-256 that the records name (`data_sha256`); ValueError otherwise. A
    record without `data_sha256` (an older run) names no file. In both cases `monitors.flags` checks every
    record against its family record afterwards: the verifier must give the saved pass or fail."""
    path = scenarios / f"{split}.jsonl"
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    file_hash = sha256_of_file(path)
    named = {record["data_sha256"] for record in records if record.get("data_sha256")}
    if named and named != {file_hash}:
        raise ValueError(f"{path} is not the file that the run was made from: the records name another "
                         "SHA-256")
    by_hash = "its SHA-256 is the one that the records name" if named else "the records name no SHA-256"
    return load_families(path), (f"{shown_path(path)} (SHA-256 {file_hash[:12]}; {by_hash}; the verifier "
                                 "gives the saved pass or fail for every valid record)")


def text_of_run(path: Path, records: list[dict]) -> TextBundle:
    """The text bundle of the file. Raises ValueError if the file is not valid, or is not the text file
    that the records were made with (`text_sha256`)."""
    try:
        text = load_text(path)
    except (OSError, TextError) as error:
        raise ValueError(f"the text file cannot be used: {error}") from None
    named = sorted({record["text_sha256"] for record in records})
    if named != [text.sha256]:
        raise ValueError(f"{path} (SHA-256 {text.sha256[:12]}) is not the text file of the run: the records "
                         f"name {[value[:12] for value in named]}; give the run's file with --text")
    return text


def report(episodes: Path, scenarios: Path, split: str, text_path: Path, n_examples: int) -> str:
    """The whole report as markdown. Raises ValueError for everything that the module text names."""
    records, n_other = records_of_split(episodes, split)
    text = text_of_run(text_path, records)
    families, families_note = families_of_run(scenarios, split, records)
    summary = monitors.summarise(records, families, text)
    setup = ", ".join(f"{key} {records[0][key]}" for key in SETUP_KEYS)
    lines = [
        f"# Monitors outside the reward: {shown_path(episodes)}",
        "",
        f"- Records: split {split}; {setup}. Records of other splits left out: {n_other}.",
        f"- Families: {families_note}.",
        f"- Text file: {shown_path(text_path)} (SHA-256 {text.sha256[:12]}, the one that the records name).",
        "",
        "",
    ]
    return "\n".join(lines) + monitors.to_markdown(summary, n_examples)


def examples_count(value: str) -> int:
    """The value of --examples as a whole number of 0 or more.

    Any other value is refused: argparse prints the message and stops the script with exit code 2. A
    negative number would cut examples off the end of each list, and a monitor with one flagged episode
    would be printed as "none"."""
    try:
        number = int(value)
    except ValueError:
        number = -1
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be a whole number of 0 or more, not {value!r}")
    return number


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--episodes", type=Path, required=True, help="the episodes.jsonl file of one run")
    parser.add_argument("--scenarios", type=Path, required=True, help="the folder of the run's family files")
    parser.add_argument("--split", required=True, help="the split of the run, for example calib or dev")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file of the run")
    parser.add_argument("--examples", type=examples_count, default=3,
                        help="example episodes per monitor (0 or more; 0 prints no example lines)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        markdown = report(args.episodes, args.scenarios, args.split, args.text, args.examples)
    except ValueError as error:
        print(f"monitor_report: {error}", file=sys.stderr)
        return 2
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
