"""Roll a model out on one split of the scenarios and save every episode. Runs on the rented GPU.

    python scripts/gpu/rollout.py --model Qwen/Qwen3.5-4B --text text/model_text.yaml \\
        --scenarios data/scenarios --split dev --samples 8 --run-seed 0 --out results/base_dev

Each scenario (a family in one wording) is attempted `--samples` times. The work is done in passes: pass s
attempts every scenario once and gives the episodes the sample number s. Its random state comes from the run
seed, the pass number, the chunk number and the turn number, so the same command on the same machine gives the
same episodes. The files are extended after every pass.

Written into --out:
    episodes.jsonl      one episode record per line (the format of the scripted runs, plus the seed)
    tokens.jsonl        one line per episode, in the same order: scenario_id, sample, seed, harness_error and
                        the token ids, segments and log-probabilities of the sampled tokens
    run_manifest.json   the command, config, text hash, model, package versions, kernels, sampler settings,
                        the code that made the run (`code_sha256` and `frozen`, from accessdesk/freeze.py)
                        and the summary. `"complete": true` is written only after the last pass.

--set NAME takes the families of one fixed subset of the dev split from the folder's sets.json (curve_subset,
policy_shift_differ, policy_shift_control, extra_tool) in place of a hand-made file of ids (--families).

The split file (and sets.json, if a set is asked for) must have the SHA-256 that the folder's summary.json
names: a file that was changed after the build stops the run. Every record carries `data_sha256`, the hash
of the split file, so that the results table can refuse to compare runs on different scenario files.

With the placeholder text the script needs --allow-placeholder, and then writes only under results/_smoke/.

The freeze. Before the freeze (there is no MANIFEST.json) the script runs as always. After it, a tree that
differs from MANIFEST.json stops the run before the model loads (exit code 2, the differences are listed,
nothing is written). --allow-changed-tree lets the run go on; the run manifest then holds
`"allow_changed_tree": true` and the differences (`frozen`).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accessdesk import impl  # noqa: E402
from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.rules import RULESETS  # noqa: E402
from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.freeze import frozen_state, run_problem  # noqa: E402
from accessdesk.text import TextBundle, load_text, model_text_path  # noqa: E402

SMOKE_DIR = ROOT / "results" / "_smoke"
SPLITS = ("train", "dev", "test", "calib")
SETS_FILE = "sets.json"
SETS_SPLIT = "dev"            # every set of sets.json is a subset of the dev split
POLICY_SHIFT_PREFIX = "policy_shift_"
SETS = ("curve_subset", POLICY_SHIFT_PREFIX + "differ", POLICY_SHIFT_PREFIX + "control", "extra_tool")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", required=True, help="model id on the Hub, or a local folder")
    parser.add_argument("--revision", default=None, help="model revision (default: the pinned one, if known)")
    parser.add_argument("--adapter", default=None, help="folder of a saved LoRA adapter to load on top")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file (YAML)")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="accept the placeholder text; outputs must then be under results/_smoke/")
    parser.add_argument("--scenarios", type=Path, required=True, help="folder with the split files")
    parser.add_argument("--split", choices=SPLITS, required=True)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--families", type=Path, default=None, help="file of family ids to keep, one a line")
    which.add_argument("--set", choices=SETS, default=None,
                       help=f"keep the families of this set of the folder's {SETS_FILE} (needs --split dev)")
    parser.add_argument("--styles", default=None, help="wording styles to keep, such as 1,2,3 (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="keep only the first N families")
    parser.add_argument("--samples", type=int, default=8, help="attempts per scenario")
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--ruleset", choices=RULESETS, default="default")
    parser.add_argument("--toolset", choices=("five", "six"), default="five")
    parser.add_argument("--prompt", choices=("final", "plain", "policy_shift"), default="final")
    parser.add_argument("--chunk-size", type=int, default=None, help="episodes per generate call")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="token cap per turn")
    parser.add_argument("--run", default=None, help="run name in the records (default: the name of --out)")
    parser.add_argument("--arm", default="model", help="arm name in the records")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-changed-tree", action="store_true",
                        help="after the freeze: run also on a tree that differs from MANIFEST.json (the run "
                             "manifest then says so)")
    parser.add_argument("--out", type=Path, required=True, help="output folder; must not hold episodes yet")
    return parser.parse_args(argv)


def run_config(args: argparse.Namespace) -> RunConfig:
    """The default config, with the two values that the command line may change (default: the config's)."""
    asked = {"chunk_size": args.chunk_size, "max_new_tokens": args.max_new_tokens}
    given = {key: value for key, value in asked.items() if value is not None}
    return dataclasses.replace(RunConfig(), **given)


def check_outputs(text: TextBundle, allow_placeholder: bool, out_dir: Path) -> str | None:
    """Why this run must not start, or None."""
    in_smoke = SMOKE_DIR.resolve() in out_dir.resolve().parents
    if text.is_placeholder and not allow_placeholder:
        return "the text file is the placeholder; pass --allow-placeholder for a smoke run"
    if text.is_placeholder and not in_smoke:
        return f"with the placeholder text the output folder must be under {SMOKE_DIR}"
    if (out_dir / "episodes.jsonl").exists() or (out_dir / "tokens.jsonl").exists():
        return f"{out_dir} already holds episodes; choose a new output folder"
    return None


def check_scenarios(scenario_dir: Path, text: TextBundle, file_names: tuple[str, ...] = ()) -> str | None:
    """The scenario files must have been built from this text file (the wordings are in the files), and each
    file of `file_names` that the run reads must still be the file that the build wrote: its SHA-256 is in
    summary.json (`files_sha256`). The expected end states are in these files, so a changed file stops the
    run."""
    summary_file = scenario_dir / "summary.json"
    if not summary_file.is_file():
        return f"{summary_file} is missing; build the scenarios with scripts/build_scenarios.py"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary["text_sha256"] != text.sha256:
        return f"{scenario_dir} was built from another text file (summary.json has another text_sha256)"
    for name in file_names:
        path = scenario_dir / name
        if not path.is_file():
            return f"{path} is missing"
        if sha256_of_file(path) != summary.get("files_sha256", {}).get(name):
            return (f"{path} is not the file that the build wrote (summary.json names another SHA-256): "
                    f"build the scenarios again")
    return None


def check_prompt(text: TextBundle, prompt: str, ruleset: str) -> str | None:
    """The policy-shift prompt goes with its own rule set, and the plain prompt must exist."""
    if prompt == "policy_shift" and ruleset != text.policy_shift_variant:
        return f"--prompt policy_shift needs --ruleset {text.policy_shift_variant}"
    if prompt == "plain" and not text.plain_system_prompt:
        return "the text file has no plain system prompt"
    return None


def set_family_ids(scenario_dir: Path, name: str) -> list[str]:
    """The family ids of one named set of the folder's sets.json. The two lists of the policy-shift set are
    named policy_shift_differ and policy_shift_control."""
    sets = json.loads((scenario_dir / SETS_FILE).read_text(encoding="utf-8"))
    if name.startswith(POLICY_SHIFT_PREFIX):
        return list(sets["policy_shift"][name.removeprefix(POLICY_SHIFT_PREFIX)])
    return list(sets[name])


def wanted_family_ids(args: argparse.Namespace) -> tuple[list[str] | None, str]:
    """(the family ids to keep, where they come from); (None, "") if neither --families nor --set is given."""
    if args.families is not None:
        lines = args.families.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()], str(args.families)
    if args.set is not None:
        return set_family_ids(args.scenarios, args.set), f"the set {args.set} of {SETS_FILE}"
    return None, ""


def select_families(families: list[dict], ids: list[str] | None, limit: int | None,
                    source: str = "the list") -> list[dict]:
    """The families named in `ids` (all if None), in file order of the split, then the first `limit`."""
    if ids is not None:
        wanted = set(ids)
        unknown = sorted(wanted - {family["family_id"] for family in families})
        if unknown:
            raise SystemExit(f"family ids of {source} that are not in the split: {unknown[:5]}")
        families = [family for family in families if family["family_id"] in wanted]
    return families if limit is None else families[:limit]


def scenario_list(families: list[dict], styles: set[int] | None) -> list[tuple[dict, int]]:
    """(family, style) for every wording of every family, in file order; only `styles` if given."""
    return [(family, wording["style"]) for family in families for wording in family["wordings"]
            if styles is None or wording["style"] in styles]


def token_line(record: dict) -> dict:
    """The line of tokens.jsonl for one record: the keys that join it to its episode, then the token data."""
    keys = ("scenario_id", "sample", "seed", "harness_error")
    return {**{key: record[key] for key in keys}, **record["tokens"]}


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("ab") as handle:
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8"))


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_gpu_memory_gb() -> dict | None:
    """The largest GPU memory of this process so far, in GiB: `allocated` is what tensors used (the model's
    weights included), `reserved` is what torch held from the driver. None without a CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return None
    return {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def summarise(records: list[dict]) -> dict:
    """The numbers printed at the end. Every one is computed by accessdesk.eval.metrics from the records."""
    n_errors = metrics.harness_error_count(records)
    summary: dict = {"episodes": len(records), "harness_errors": n_errors,
                     "harness_error_rate": n_errors / len(records) if records else None}
    if n_errors < len(records):
        turns = [turn for record in metrics.valid_records(records) for turn in record["turns"]]
        summary.update({
            "pass_rate": metrics.pass_rate(records),
            "pass_rate_by_outcome": metrics.pass_rate_by(records, "outcome"),
            "invalid_call_rate": metrics.invalid_call_rate(records),
            "truncated_turn_rate": sum(bool(turn["truncated"]) for turn in turns) / len(turns),
            "tokens_per_turn": sum(turn["n_tokens"] for turn in turns) / len(turns),
            "max_tokens_per_turn": max(turn["n_tokens"] for turn in turns),
            **metrics.turn_stats(records),
        })
    return summary


def print_summary(summary: dict) -> None:
    print(f"episodes {summary['episodes']}   harness errors {summary['harness_errors']}")
    if "pass_rate" not in summary:
        print("no valid episode")
        return
    print(f"pass rate (mean over families)  {summary['pass_rate']:.3f}")
    for outcome, rate in summary["pass_rate_by_outcome"].items():
        print(f"  {outcome:10s} {rate:.3f}")
    print(f"invalid-call rate (malformed messages / all messages)  {summary['invalid_call_rate']:.3f}")
    print(f"turns cut off by the token cap  {summary['truncated_turn_rate']:.3f}")
    print(f"tokens per turn  {summary['tokens_per_turn']:.1f}  (largest {summary['max_tokens_per_turn']})")
    print(f"turns per episode  {summary['mean_turns']:.2f}")
    print(f"episodes that used up the turn limit  {summary['share_at_turn_limit']:.3f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = load_text(args.text)
    cfg = run_config(args)
    if args.set is not None and args.split != SETS_SPLIT:
        print(f"--set needs --split {SETS_SPLIT}: the sets of {SETS_FILE} are subsets of that split",
              file=sys.stderr)
        return 2
    read_files = (f"{args.split}.jsonl",) + ((SETS_FILE,) if args.set is not None else ())
    problem = (check_outputs(text, args.allow_placeholder, args.out)
               or check_scenarios(args.scenarios, text, read_files)
               or check_prompt(text, args.prompt, args.ruleset))
    if problem:
        print(problem, file=sys.stderr)
        return 2
    # The code that makes this run, and whether the tree is as frozen (accessdesk/freeze.py). It is asked
    # before the model loads.
    frozen = frozen_state(ROOT)
    problem = run_problem(frozen, args.allow_changed_tree)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    split_file = args.scenarios / f"{args.split}.jsonl"
    data_sha256 = sha256_of_file(split_file)
    family_ids, ids_source = wanted_family_ids(args)
    families = select_families(load_families(split_file), family_ids, args.limit, ids_source)
    styles = {int(style) for style in args.styles.split(",")} if args.styles else None
    scenarios = scenario_list(families, styles)
    if not scenarios:
        print("no scenario is left after --families, --styles and --limit", file=sys.stderr)
        return 2

    # The GPU part is imported here, so that a wrong argument is reported before torch and the model load.
    from accessdesk.harness.episode import Episode
    from accessdesk.harness.hf_backend import (Sampler, environment_report, kernel_report, load_model,
                                               resolve_revision, rollout)
    from accessdesk.harness.tokens import TokenLayout

    model, tokenizer = load_model(args.model, args.revision, adapter_dir=args.adapter, device=args.device)
    sampler = Sampler(model, tokenizer, cfg)
    layouts = {args.toolset: TokenLayout(tokenizer, text.tool_specs(args.toolset))}
    manifest = {
        "complete": False,
        "command": sys.argv,
        "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "text_file": Path(text.path).name, "text_sha256": text.sha256,
        "text_is_placeholder": text.is_placeholder,
        "impl": impl.status(),
        "code_sha256": frozen["code_sha256"], "frozen": frozen,
        "allow_changed_tree": args.allow_changed_tree,
        "model": args.model, "revision": resolve_revision(args.model, args.revision), "adapter": args.adapter,
        "split": args.split, "split_file_sha256": data_sha256, "family_set": args.set,
        "families": len(families), "scenarios": len(scenarios), "samples": args.samples,
        "run_seed": args.run_seed, "ruleset": args.ruleset, "toolset": args.toolset, "prompt": args.prompt,
        "environment": environment_report(),
        "kernels": kernel_report(model),
        "sampler": sampler.settings(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "run_manifest.json", manifest)

    meta = {"run": args.run or args.out.name, "arm": args.arm, "data_sha256": data_sha256}
    records: list[dict] = []
    started = time.perf_counter()
    for sample in range(args.samples):
        options = {"ruleset": args.ruleset, "toolset": args.toolset, "prompt": args.prompt}
        episodes = [Episode(family, style, text, cfg, **options) for family, style in scenarios]
        new = rollout(sampler, layouts, episodes, run_seed=args.run_seed, pass_index=sample,
                      meta={**meta, "first_sample": sample})
        without_tokens = [{key: value for key, value in record.items() if key != "tokens"} for record in new]
        append_jsonl(args.out / "tokens.jsonl", [token_line(record) for record in new])
        append_jsonl(args.out / "episodes.jsonl", without_tokens)
        records.extend(without_tokens)
        seconds = time.perf_counter() - started
        print(f"pass {sample + 1}/{args.samples}: {len(records)} episodes, {seconds:.0f} s, "
              f"{metrics.harness_error_count(records)} harness errors", flush=True)

    seconds = time.perf_counter() - started
    summary = summarise(records)
    manifest.update({"complete": True, "seconds": round(seconds, 1),
                     "episodes_per_hour": round(len(records) / seconds * 3600),
                     "peak_gpu_memory_gb": peak_gpu_memory_gb(), "summary": summary})
    write_json(args.out / "run_manifest.json", manifest)
    print_summary(summary)
    print(f"{manifest['episodes_per_hour']} episodes per hour in {manifest['seconds']} s; "
          f"peak GPU memory {manifest['peak_gpu_memory_gb']} GiB; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
