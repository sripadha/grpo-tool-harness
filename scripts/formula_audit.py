"""The formula audit of a GRPO run folder (ANALYSIS_PLAN.md section 8, line 4). Written after the freeze;
it only reads the saved training rollouts and the train log.

    .venv/bin/python scripts/formula_audit.py --run results/main/grpo_seed1 \\
        --out results/main/audit/formula_grpo_seed1.json

For every saved rollout of the run (rollouts/step_*.episodes.jsonl, and rollouts/start_<n>/ after a
restart) it asks the four questions of accessdesk/audit/formula.py:
  1. no failing attempt has a positive reward;
  2. every saved reward is the reward that the main and the second version of `reward` give;
  3. the advantages can be made again from the saved rewards, and the numbers of the train log that
     follow from them agree (the log does not hold the advantages themselves);
  4. every kept group has unequal rewards, and every dropped group equal ones (or it lost more than half
     of its attempts).

Printed, and written to --out if given: a JSON with the counts, the differences by kind, the first ten
differences, and for each step whose loss was compared the two losses and the bound (`loss_checks`).
After the JSON a short summary is printed in plain words. It ends with one paragraph on what each of the
four checks can see and cannot see; the same paragraph is in the JSON (`what_the_checks_see`).

The loss comparison is a weak check. The trainer's advantages are saved nowhere, so the logged loss is
the only witness of the advantages that the trainer used, and the bound of that comparison grows with
the gap between the sampler's and the trainer's log-probabilities. Read the bound of each step before
you rest a claim on it.
Exit code 0: no difference. 1: at least one difference. 2: the audit could not be made (no run folder,
no saved rollout, no GRPO config).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.audit import formula, load_script  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="the folder of one GRPO run")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON to this file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Audit one run folder and print the JSON. Returns the exit code (see the module text)."""
    args = parse_args(argv)
    try:
        problem = load_script("replay_check").second_versions_problem()
        if problem or impl.second_is_written("group_advantages") is False:
            raise ValueError(problem or "the second version of group_advantages is not written")
        report = formula.audit_run(args.run)
    except (OSError, ValueError) as error:
        print(f"formula_audit: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    print("\n".join(formula.summary_lines(report)))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 1 if report["n_differences"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
