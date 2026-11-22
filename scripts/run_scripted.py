"""Run one scripted policy over one split of a scenario folder, save the episodes, print the pass rates.

    .venv/bin/python scripts/run_scripted.py --scenarios data/scenarios_placeholder --split dev \\
        --policy oracle --out results/_smoke/scripted_oracle_dev.jsonl
    .venv/bin/python scripts/run_scripted.py --list          (the names of the policies)

Every scenario of the split (each family in each of its wording styles) is played once. No model and no GPU
are used. The two scripted rows of the results table come from here: `oracle` must pass every scenario, and
each `constant_*` policy must pass exactly one outcome in four.

The text file must be the one that the scenarios were built from (its hash is in the folder's summary.json).
Without --text the placeholder text is used.
Every record carries `data_sha256`, the SHA-256 of the split file that was loaded, so that the results table
can refuse to compare runs on different scenario files. The split file must have the SHA-256 that the
folder's summary.json names: a file that was changed after the build stops the run.

Beside the records file the script writes <the records file>.run_manifest.json: the command, the policy, the
split, the config, the hashes of the text file, of the split file and of the records file, and the code
that made the run (`code_sha256` and `frozen`, from accessdesk/freeze.py).

The freeze. Before the freeze (there is no MANIFEST.json) the script runs as always. After it, a tree that
differs from MANIFEST.json stops the run (exit code 2, the differences are listed, nothing is written).
--allow-changed-tree lets the run go on; the run manifest then holds `"allow_changed_tree": true` and the
differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.generator import SPLITS  # noqa: E402
from accessdesk.env.rules import RULESETS  # noqa: E402
from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.freeze import frozen_state, run_problem  # noqa: E402
from accessdesk.harness.episode import Episode  # noqa: E402
from accessdesk.harness.policies import Policy, named_policies, oracle, run_policy  # noqa: E402
from accessdesk.harness.records import write_records  # noqa: E402
from accessdesk.text import TOOLSETS, TextBundle, load_text, placeholder_path  # noqa: E402

SCRIPT = "scripts/run_scripted.py"
ORACLE_OF_RULESET = "oracle_of_ruleset"      # the correct policy for the rule set given with --ruleset
MANIFEST_SUFFIX = ".run_manifest.json"       # the run manifest is <the records file> with this ending


def policies_by_name(text: TextBundle, seed: int, ruleset: str) -> dict[str, Policy]:
    """The named policies (they follow the default rules), and the correct policy for `ruleset`."""
    return {**named_policies(text, seed), ORACLE_OF_RULESET: oracle(text, ruleset)}


def scenarios_problem(scenarios_dir: Path, text: TextBundle, cfg: RunConfig,
                      split: str | None = None) -> str | None:
    """Why this folder cannot be used with this text file and config, or None if it can.

    With `split`, the split's file must still be the file that the build wrote: its SHA-256 is in
    summary.json (`files_sha256`). The expected end states are in that file, so a changed file stops the
    run."""
    summary_path = scenarios_dir / "summary.json"
    if not summary_path.is_file():
        return f"{summary_path} is missing: build the scenarios with scripts/build_scenarios.py"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["text_sha256"] != text.sha256:
        return (f"the scenarios in {scenarios_dir} were built from another text file "
                f"({summary['text_file']}), not from {text.path}")
    if summary["config_sha256"] != cfg.sha256():
        return f"the scenarios in {scenarios_dir} were built with another config"
    if split is not None:
        split_file = scenarios_dir / f"{split}.jsonl"
        if not split_file.is_file():
            return f"{split_file} is missing"
        if sha256_of_file(split_file) != summary.get("files_sha256", {}).get(split_file.name):
            return (f"{split_file} is not the file that the build wrote (summary.json names another "
                    f"SHA-256): build the scenarios again")
    return None


def sha256_of_file(path: Path) -> str:
    """The SHA-256 of the file's bytes, as 64 hexadecimal digits."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_split(families: list[dict], policy: Policy, name: str, text: TextBundle, cfg: RunConfig, *,
              ruleset: str, toolset: str, seed: int, data_sha256: str | None = None) -> list[dict]:
    """One episode record per scenario, in file order: every family in every one of its wording styles.

    `data_sha256` is the hash of the file that `families` were loaded from; every record carries it."""
    records = []
    for family in families:
        for wording in family["wordings"]:
            episode = Episode(family, wording["style"], text, cfg, ruleset=ruleset, toolset=toolset)
            run_policy(episode, policy)
            position = {"run_seed": seed, "pass": 0, "chunk": 0, "pos": len(records)}
            records.append(episode.record(run="scripted", arm=name, sample=0, seed=position,
                                          data_sha256=data_sha256))
    return records


def report(records: list[dict]) -> str:
    """The pass rates per outcome and in total, by family, as lines of text."""
    by_outcome = metrics.pass_rate_by(records, "outcome")
    n_families = len({record["family_id"] for record in records})
    lines = [f"  {outcome:9s} {100 * rate:5.1f}%" for outcome, rate in by_outcome.items()]
    lines.append(f"  {'all':9s} {100 * metrics.pass_rate(records):5.1f}%   "
                 f"({len(records)} episodes, {n_families} families)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scenarios", type=Path, help="the scenario folder (of scripts/build_scenarios.py)")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--policy", help="the name of a scripted policy (see --list)")
    parser.add_argument("--out", type=Path, help="the file for the episode records (JSON lines)")
    parser.add_argument("--text", type=Path, default=placeholder_path(), help="the text file (YAML)")
    parser.add_argument("--ruleset", choices=RULESETS, default="default",
                        help="the rule set of the expected states")
    parser.add_argument("--toolset", choices=TOOLSETS, default="five")
    parser.add_argument("--seed", type=int, default=0, help="for random_valid_calls and the made-up ids")
    parser.add_argument("--list", action="store_true", help="print the policy names and stop")
    parser.add_argument("--allow-changed-tree", action="store_true",
                        help="after the freeze: run also on a tree that differs from MANIFEST.json (the run "
                             "manifest then says so)")
    args = parser.parse_args(argv)

    text = load_text(args.text)
    cfg = RunConfig().with_seeds_for(text.is_placeholder)      # the config that the build script used
    policies = policies_by_name(text, args.seed, args.ruleset)
    if args.list:
        print("\n".join(policies))
        return 0
    if args.scenarios is None or args.policy is None or args.out is None:
        parser.error("--scenarios, --policy and --out are needed")
    if args.policy not in policies:
        parser.error(f"unknown policy {args.policy!r}; run with --list to see the names")
    problem = scenarios_problem(args.scenarios, text, cfg, args.split)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    # The code that makes this run, and whether the tree is as frozen (accessdesk/freeze.py).
    frozen = frozen_state(ROOT)
    problem = run_problem(frozen, args.allow_changed_tree)
    if problem:
        print(problem, file=sys.stderr)
        return 2

    split_file = args.scenarios / f"{args.split}.jsonl"
    data_sha256 = sha256_of_file(split_file)
    families = load_families(split_file)
    records = run_split(families, policies[args.policy], args.policy, text, cfg, ruleset=args.ruleset,
                        toolset=args.toolset, seed=args.seed, data_sha256=data_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_records(args.out, records)
    manifest = {
        "script": SCRIPT, "command": list(sys.argv) if argv is None else [SCRIPT, *argv],
        "policy": args.policy, "split": args.split, "ruleset": args.ruleset, "toolset": args.toolset,
        "seed": args.seed,
        "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "text_file": Path(text.path).name, "text_sha256": text.sha256,
        "text_is_placeholder": text.is_placeholder,
        "impl": impl.status(),
        "code_sha256": frozen["code_sha256"], "frozen": frozen,
        "allow_changed_tree": args.allow_changed_tree,
        "split_file_sha256": data_sha256, "families": len(families), "episodes": len(records),
        "records": {"path": str(args.out), "sha256": sha256_of_file(args.out),
                    "bytes": args.out.stat().st_size},
    }
    manifest_file = args.out.with_name(args.out.name + MANIFEST_SUFFIX)
    manifest_file.write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    print(f"policy {args.policy}, split {args.split}, rule set {args.ruleset}, tools {args.toolset}")
    print("pass rate by family:")
    print(report(records))
    print(f"wrote {args.out} and {manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
