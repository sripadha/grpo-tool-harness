"""Choose the two training families of the overfit check (PLAN.md section 5.6, "Checks before the seeds").

    python scripts/pick_overfit_families.py --episodes results/main/base_train/episodes.jsonl \\
        --out results/main/checks/overfit_families.txt

It needs no GPU and no torch. The overfit check trains GRPO on two scenarios only and expects their pass
rate to go near 1 in about 30 steps. It can show that only on scenarios with mixed outcomes at the start:
some attempts pass and some fail. A scenario that always fails gives only tied groups and stays at 0, and
one that always passes is at 1 from the start.

--episodes is the episodes.jsonl of a rollout on the train split with several attempts per scenario. The
script looks at one wording style (--style, default 1), because the check runs with `--styles 1`: one
scenario per family. For each family it counts the passes among the attempts of that scenario.

The rule is fixed:
  - a family can be chosen if all its attempts in that style are valid (no harness error), and some passed
    and some failed: 0 < passes < attempts;
  - the families whose pass count is nearest to half of the attempts come first (with 8 attempts: nearest
    to 4); ties go to the smaller family id;
  - the first two (--count) are taken.

Written: --out, the family ids, one a line, as `scripts/gpu/train_grpo.py --overfit` reads them; and a file
with the same name plus ".json": the rule, the chosen families with their pass counts, how many families
could be chosen, and the SHA-256 of the episodes file and of this script. The pass counts are printed.

Exit code 1 with a message if fewer than --count families can be chosen (nothing is written). Exit code 2
if the input is refused (nothing is written): a missing or empty file, a record of another split than
train, records of more than one run, no record of the style, or a rollout that is not finished. The run
manifest of the rollout (run_manifest.json beside the episodes file) says whether it is finished: it must be
there, say "complete": true, and name the number of scenarios and attempts that the file holds. A rollout
that is still running has only its first passes on disk, so its pass counts are of fewer attempts.

The counts say how the model that sampled the episodes behaves. The check itself starts from the SFT
adapter. If the SFT model was sampled on train (scripts/gpu/chain_stage1.sh, step sft_train), its
episodes.jsonl is the closer input: it shows the outcomes "at the start" of the check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for sft_profile.py, next to this file

from accessdesk.eval import metrics  # noqa: E402
from sft_profile import Refused, file_entry, read_train_records, rollout_manifest  # noqa: E402

SCRIPT = "scripts/pick_overfit_families.py"
STYLE = 1                    # the wording style of the overfit check
COUNT = 2                    # PLAN 5.6: "two scenarios with mixed outcomes at the start"
RULE = ("families whose attempts in the style are all valid, with 0 < passes < attempts; ordered by the "
        "distance of the pass count from half of the attempts, then by family id; the first ones are taken")


def pass_counts(records: list[dict], style: int) -> dict[str, dict]:
    """family id -> {"passes", "attempts", "harness_errors"} over the family's records in `style`.
    `attempts` counts the valid records. Families in sorted order."""
    counts: dict[str, dict] = {}
    for record in records:
        if record["style"] != style:
            continue
        row = counts.setdefault(record["family_id"], {"passes": 0, "attempts": 0, "harness_errors": 0})
        if record["harness_error"] is not None:
            row["harness_errors"] += 1
        else:
            row["attempts"] += 1
            row["passes"] += bool(record["passed"])
    return {family_id: counts[family_id] for family_id in sorted(counts)}


def is_mixed(row: dict) -> bool:
    """Every attempt is valid, and some passed and some failed."""
    return row["harness_errors"] == 0 and 0 < row["passes"] < row["attempts"]


def choose(counts: dict[str, dict], count: int = COUNT) -> list[str]:
    """The ids of the `count` mixed families whose pass count is nearest to half of their attempts; ties
    go to the smaller family id. Fewer ids if fewer families are mixed."""
    mixed = [family_id for family_id, row in counts.items() if is_mixed(row)]
    # 2 * passes - attempts is twice the distance from half of the attempts: whole numbers, no rounding.
    mixed.sort(key=lambda family_id: (abs(2 * counts[family_id]["passes"] - counts[family_id]["attempts"]),
                                      family_id))
    return mixed[:count]


def source_model(manifest: dict) -> dict:
    """What the rollout's run manifest (as `sft_profile.rollout_manifest` returns it) says about the model."""
    return {"run_manifest": manifest["entry"],
            **{key: manifest.get(key) for key in ("complete", "model", "revision", "adapter", "samples",
                                                  "run_seed")}}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--episodes", type=Path, required=True,
                        help="episodes.jsonl of a rollout on the train split")
    parser.add_argument("--out", type=Path, required=True,
                        help="the families file to write (a second file with .json added holds the counts)")
    parser.add_argument("--style", type=int, default=STYLE, help="the wording style that is counted")
    parser.add_argument("--count", type=int, default=COUNT, help="families to choose")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.count < 1:
        raise Refused(f"--count must be 1 or more, not {args.count}")
    if args.style == metrics.HELD_OUT_STYLE:
        raise Refused(f"style {metrics.HELD_OUT_STYLE} is the held-out style: it is never trained on")
    records = read_train_records(args.episodes)
    # Refused if the rollout is not finished: its manifest is missing, is not complete, or names another
    # number of scenarios and attempts than the file holds.
    manifest = rollout_manifest(args.episodes, records)
    counts = pass_counts(records, args.style)
    if not counts:
        raise Refused(f"{args.episodes} holds no record in wording style {args.style}")
    chosen = choose(counts, args.count)
    numbers = {
        "families": len(counts), "mixed": sum(1 for row in counts.values() if is_mixed(row)),
        "always_pass": sum(1 for row in counts.values()
                           if row["harness_errors"] == 0 and row["passes"] == row["attempts"]),
        "always_fail": sum(1 for row in counts.values() if row["harness_errors"] == 0 and row["passes"] == 0),
        "with_harness_error": sum(1 for row in counts.values() if row["harness_errors"] > 0)}
    if len(chosen) < args.count:
        print(f"pick_overfit_families: only {len(chosen)} of the {numbers['families']} families in "
              f"{args.episodes} have mixed outcomes in style {args.style} (some attempts pass, some fail); "
              f"{args.count} are needed. Counts: {numbers}. Nothing was written.", file=sys.stderr)
        return 1
    report = {
        "rule": RULE, "style": args.style, "count": args.count,
        "chosen": [{"family_id": family_id, **counts[family_id]} for family_id in chosen],
        "counts": numbers,
        "source": {"episodes": file_entry(args.episodes), "run": records[0]["run"], "arm": records[0]["arm"],
                   "model": source_model(manifest)},
        "script": {**file_entry(Path(__file__).resolve()), "path": SCRIPT},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes("".join(f"{family_id}\n" for family_id in chosen).encode("utf-8"))
    report_file = args.out.with_name(args.out.name + ".json")
    report_file.write_bytes((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    for row in report["chosen"]:
        print(f"{row['family_id']}: {row['passes']} of {row['attempts']} attempts passed in style "
              f"{args.style}")
    print(f"{numbers['mixed']} of {numbers['families']} families are mixed in style {args.style} "
          f"({numbers['always_pass']} always pass, {numbers['always_fail']} always fail, "
          f"{numbers['with_harness_error']} have a harness error); wrote {args.out} and {report_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Refused as refusal:
        print(f"pick_overfit_families: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
