"""Monitor counts among the top-reward attempts: how many of the attempts that passed with the full reward
each monitor m1 to m5 flags, for the four arms on dev and on test, and for the two GRPO seeds together.

    .venv/bin/python results/main/audit/monitor_counts/count.py              # writes the two files here
    .venv/bin/python results/main/audit/monitor_counts/count.py --out DIR    # or into DIR

Run it from the top folder of the repository. It reads results/main/<arm>_<split>/episodes.jsonl,
data/scenarios/<split>.jsonl and text/model_text.yaml, and writes top_reward_flags.md and
top_reward_flags.json. Written on 2026-11-19, after the test run: it computes no new kind of number.

It does not define a monitor again. It loads the records, the family records and the text file with the
functions of scripts/monitor_report.py (with the same checks: one run per file, the family file and the
text file that the records name), and it calls accessdesk/eval/monitors.py:
- `monitors.flags` gives what each monitor finds in one attempt (it also checks that the verifier gives
  the saved pass or fail);
- `monitors.has_full_reward` says whether the attempt passed with the full reward. These attempts are the
  row "passed with the full reward" of results/main/monitors/<arm>_<split>.md; here they are called
  "top-reward attempts".
Counts are plain counts of attempts, not family-weighted shares.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                 # results/main/audit/monitor_counts -> the top folder
sys.path.insert(0, str(ROOT))

from accessdesk.eval import metrics, monitors  # noqa: E402

ARMS = ("base", "sft", "grpo_seed1", "grpo_seed2")
SPLITS = ("dev", "test")
GRPO = ("grpo_seed1", "grpo_seed2")
FIVE = ("refused_call", "id_not_from_a_result", "access_not_checked", "repeated_call", "words_outside_call")
COLUMNS = ("n_episodes", "n_top_reward", "m1", "m2", "m3", "m4", "m5", "m1_to_m3", "m4_or_m5",
           "any_of_m1_to_m5", "any_of_all_seven")


def monitor_report_module():
    """scripts/monitor_report.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("monitor_report", ROOT / "scripts" / "monitor_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_run(report, arm: str, split: str) -> dict:
    """The counts of one arm on one split."""
    episodes = ROOT / "results" / "main" / f"{arm}_{split}" / "episodes.jsonl"
    records, _ = report.records_of_split(episodes, split)
    text = report.text_of_run(ROOT / "text" / "model_text.yaml", records)
    families, _ = report.families_of_run(ROOT / "data" / "scenarios", split, records)
    by_id = {family["family_id"]: family for family in families}
    valid = metrics.valid_records(records)
    row = {name: 0 for name in COLUMNS}
    row["n_episodes"] = len(valid)
    for record in valid:
        found = monitors.flags(record, by_id[record["family_id"]], text)
        if not monitors.has_full_reward(record):
            continue
        row["n_top_reward"] += 1
        hit = {name: found[name] is not None for name in monitors.MONITORS}
        for name in FIVE:
            row[monitors.LABELS[name]] += hit[name]
        row["m1_to_m3"] += any(hit[name] for name in monitors.CLEAN_MONITORS)
        row["m4_or_m5"] += hit["repeated_call"] or hit["words_outside_call"]
        row["any_of_m1_to_m5"] += any(hit[name] for name in FIVE)
        row["any_of_all_seven"] += any(hit.values())
    return {"arm": arm, "split": split, "episodes": f"results/main/{arm}_{split}/episodes.jsonl", **row}


def all_rows() -> list[dict]:
    report = monitor_report_module()
    rows = []
    for split in SPLITS:
        per_arm = [count_run(report, arm, split) for arm in ARMS]
        together = {"arm": "grpo_seed1 + grpo_seed2", "split": split, "episodes": "the two rows above"}
        together.update({name: sum(row[name] for row in per_arm if row["arm"] in GRPO) for name in COLUMNS})
        rows.extend(per_arm + [together])
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "# Monitor counts among the top-reward attempts",
        "",
        "Made by `results/main/audit/monitor_counts/count.py` from `results/main/<arm>_<split>/episodes.jsonl`,",
        "with the monitors of `accessdesk/eval/monitors.py`. A top-reward attempt passed with the full reward",
        "(`monitors.has_full_reward`): the row \"passed with the full reward\" of `results/main/monitors/<arm>_<split>.md`.",
        "Each number is a count of attempts, not a share.",
        "",
        "m1 refused_call, m2 id_not_from_a_result, m3 access_not_checked, m4 repeated_call, m5 words_outside_call.",
        "\"m1 to m3\": flagged by at least one of m1, m2, m3. \"m4 or m5\": flagged by m4 or m5.",
        "\"any of m1 to m5\": flagged by at least one of the five. \"any of all seven\": also m5b and m6.",
        "",
        "| split | arm | attempts | top-reward | m1 | m2 | m3 | m4 | m5 | m1 to m3 | m4 or m5 | any of m1 to m5 "
        "| any of all seven |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([row["split"], row["arm"]] + [str(row[name]) for name in COLUMNS]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--out", type=Path, default=HERE, help="the folder for the two files (default: here)")
    args = parser.parse_args(argv)
    try:
        rows = all_rows()
    except ValueError as error:
        print(f"count: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "top_reward_flags.json").write_text(
        json.dumps({"made_by": "results/main/audit/monitor_counts/count.py", "columns": list(COLUMNS),
                    "rows": rows}, indent=1) + "\n", encoding="utf-8")
    markdown = to_markdown(rows)
    (args.out / "top_reward_flags.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
