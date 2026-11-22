"""The family profile after SFT, and the rule on the GRPO step that reads it (PLAN.md section 5.5).

It needs no GPU and no torch. It has two uses, in this order:

    python scripts/sft_profile.py --scenarios data/scenarios --pick results/main/sft_profile/families.txt
    python scripts/sft_profile.py --scenarios data/scenarios \\
        --episodes results/main/sft_train/episodes.jsonl \\
        --families results/main/sft_profile/families.txt --out results/main/sft_profile

--pick FILE   writes the families that the profile is made on: the first `--per-outcome` (25) families of
              each expected outcome, in the order of train.jsonl; 100 family ids, one a line. Then
              scripts/gpu/rollout.py samples the SFT model on them (`--split train --families FILE`).
--episodes    the episodes.jsonl of that rollout. The script computes the pass rate of each family
              (`accessdesk.eval.metrics.family_pass_rates`: the share of the family's episodes that passed,
              over its wordings and attempts, harness errors left out), the median of these numbers, and how
              many families always pass, always fail, or are mixed. Then it applies the rule.

The rule was set before any run (PLAN 5.5). No family is removed. If the median family pass rate is above
0.8, a GRPO step uses groups of 16 attempts on 8 scenarios; else groups of 8 on 16 scenarios. A median of
exactly 0.8 is not above 0.8, so it gives groups of 8. The median is computed with whole fractions, so that
"exactly 0.8" does not depend on rounding.

Written into --out:
    decision.json   median_family_pass_rate, rule, group_size, scenarios_per_step, n_families, the counts
                    n_always_pass, n_always_fail and n_mixed, the same counts per outcome, every family
                    with its numbers, and where the numbers come from: the SHA-256 of the episodes file,
                    of the rollout's run manifest, of the families file, of the adapter and of this script.
                    It holds no time, so the same input gives the same bytes.
    profile.md      the same as a small table
scripts/gpu/chain_stage2.sh reads group_size and scenarios_per_step from decision.json.

The run is refused (exit code 2, nothing is written) if
  - a record is not of the train split, or the records are of more than one run or arm;
  - run_manifest.json of the rollout is missing beside the episodes file, says that the rollout is not
    complete, or names no adapter: the profile is the one of the SFT model, and only the manifest says
    whether the model had an adapter;
  - the records or the manifest name another scenario file than train.jsonl of --scenarios, or that file is
    not the one that the build wrote;
  - the scenarios do not all have the number of attempts that the manifest names;
  - --families is given and the records are not of exactly these families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.eval import metrics  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402
from accessdesk.train.sft import TRAIN_SPLIT  # noqa: E402
from accessdesk.types import OUTCOMES  # noqa: E402

SCRIPT = "scripts/sft_profile.py"
MANIFEST_NAME = "run_manifest.json"           # written by scripts/gpu/rollout.py beside episodes.jsonl
DECISION_NAME = "decision.json"
TABLE_NAME = "profile.md"
TRAIN_FILE = f"{TRAIN_SPLIT}.jsonl"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
PER_OUTCOME = 25                              # PLAN 5.5: the profile is made on 100 training families
THRESHOLD = Fraction(4, 5)                    # PLAN 5.5: "if the median family pass rate is above 0.8"
RULE = "median above 0.8 -> groups of 16 on 8 scenarios per step; else groups of 8 on 16"
# (attempts per scenario, scenarios per step). Both give 128 episodes in a step.
SMALL_GROUPS = (8, 16)                        # the default step of PLAN 5.6
LARGE_GROUPS = (16, 8)                        # "groups of 16 on half as many scenarios per step"
SETUP_KEYS = ("run", "arm", "ruleset", "toolset", "prompt")     # one value of each in the records of one run
PROFILES = ("always_pass", "always_fail", "mixed")


class Refused(Exception):
    """The script must not write a result. The message says why."""


# ---------------------------------------------------------------------------
# Reading and checking the inputs
# ---------------------------------------------------------------------------
def file_entry(path: Path) -> dict:
    """What a result says about one file: its path as given, its SHA-256 and its size in bytes."""
    raw = Path(path).read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def train_file(scenario_dir: Path) -> dict:
    """train.jsonl of the scenario folder, as `file_entry`. Refused if the folder has no summary.json, or if
    the file is not the one that the build wrote (its SHA-256 is in summary.json, `files_sha256`). This is
    the check of scripts/gpu/rollout.py."""
    summary_file, path = scenario_dir / "summary.json", scenario_dir / TRAIN_FILE
    if not summary_file.is_file() or not path.is_file():
        raise Refused(f"{summary_file} or {path} is missing")
    entry = file_entry(path)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if entry["sha256"] != summary.get("files_sha256", {}).get(TRAIN_FILE):
        raise Refused(f"{path} is not the file that the build wrote (summary.json names another SHA-256)")
    return entry


def read_ids(path: Path) -> list[str]:
    """The family ids of a families file: one a line, empty lines left out."""
    if not path.is_file():
        raise Refused(f"{path} is missing")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_train_records(path: Path) -> list[dict]:
    """The episode records of a rollout on the train split.

    Refused if the file is missing or empty, if a record is of another split, or if the records are of more
    than one run or set-up (the metrics take the records of ONE arm)."""
    if not path.is_file():
        raise Refused(f"{path} is missing")
    try:
        records = read_records(path)
    except ValueError as error:      # a line that is not JSON, for example a line that was cut off
        raise Refused(f"{path} is not a file of JSON lines ({error})") from error
    if not records:
        raise Refused(f"{path} holds no record")
    other = sorted({str(record.get("split")) for record in records} - {TRAIN_SPLIT})
    if other:
        raise Refused(f"{path} holds records of the split {', '.join(other)}: only records of the "
                      f"{TRAIN_SPLIT} split are read here (dev and test are never used for a training "
                      "setting)")
    setups = sorted({tuple(str(record.get(key)) for key in SETUP_KEYS) for record in records})
    if len(setups) != 1:
        raise Refused(f"{path} holds records of {len(setups)} runs or set-ups {SETUP_KEYS}: {setups[:3]}")
    return records


def rollout_manifest(episodes_file: Path, records: list[dict]) -> dict:
    """run_manifest.json of the rollout that wrote the episodes file (it lies beside it), with `entry`
    (path, SHA-256, size). Refused if it is missing, not complete, of another split, or if it names another
    number of episodes than the file holds."""
    path = episodes_file.parent / MANIFEST_NAME
    if not path.is_file():
        raise Refused(f"{path} is missing: without the run manifest of the rollout it is not known which "
                      "model sampled the episodes")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise Refused(f"{path} says that the rollout is not complete")
    if manifest.get("split") != TRAIN_SPLIT:
        raise Refused(f"{path} is the manifest of a rollout on the split {manifest.get('split')!r}, not on "
                      f"{TRAIN_SPLIT}")
    scenarios, samples = manifest.get("scenarios"), manifest.get("samples")
    if not isinstance(scenarios, int) or not isinstance(samples, int) or scenarios * samples != len(records):
        raise Refused(f"{episodes_file} holds {len(records)} records, but {path} names {scenarios} "
                      f"scenarios with {samples} attempts each")
    per_scenario: dict[str, int] = {}
    for record in records:
        per_scenario[record["scenario_id"]] = per_scenario.get(record["scenario_id"], 0) + 1
    uneven = sorted(sid for sid, n in per_scenario.items() if n != samples)
    if len(per_scenario) != scenarios or uneven:
        raise Refused(f"{episodes_file} does not hold {samples} attempts of each of {scenarios} scenarios "
                      f"({len(per_scenario)} scenarios, {len(uneven)} with another number of attempts)")
    return {**manifest, "entry": file_entry(path)}


def check_adapter(manifest: dict) -> None:
    """The profile is the one of a model WITH an adapter (the SFT model). The records do not say whether the
    model had one; the run manifest of the rollout does (`adapter`: the folder, or null)."""
    if not manifest.get("adapter"):
        raise Refused(f"{manifest['entry']['path']} names no adapter: these are episodes of the untrained "
                      "model. The profile after SFT needs a rollout of the model with the SFT adapter")


def check_source(records: list[dict], manifest: dict, train: dict, families: list[dict],
                 wanted_ids: list[str] | None) -> None:
    """The records must be episodes of train.jsonl of the scenario folder, and of the wanted families."""
    named = {str(record.get("data_sha256")) for record in records} | {str(manifest.get("split_file_sha256"))}
    if named != {train["sha256"]}:
        raise Refused(f"the records or their run manifest name another scenario file than {train['path']} "
                      f"(they name {sorted(name[:12] for name in named)}, the file has "
                      f"{train['sha256'][:12]})")
    found = {record["family_id"] for record in records}
    unknown = sorted(found - {family["family_id"] for family in families})
    if unknown:
        raise Refused(f"{len(unknown)} families of the records are not in {train['path']}, for example "
                      f"{unknown[0]}")
    if wanted_ids is not None and found != set(wanted_ids):
        missing, more = sorted(set(wanted_ids) - found), sorted(found - set(wanted_ids))
        raise Refused(f"the records are not of the families of --families: {len(missing)} families of the "
                      f"file have no record {missing[:3]}, and {len(more)} families of the records are "
                      f"not in the file {more[:3]}")


def adapter_hashes(folder: str) -> dict | None:
    """The SHA-256 of the adapter's two files, or None if the folder is not on this machine."""
    paths = [Path(folder) / name for name in ADAPTER_FILES]
    if not all(path.is_file() for path in paths):
        return None
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


# ---------------------------------------------------------------------------
# The pick, the profile and the rule
# ---------------------------------------------------------------------------
def pick_families(families: list[dict], per_outcome: int = PER_OUTCOME) -> list[str]:
    """The ids of the first `per_outcome` families of each outcome, in the order of the given list (the
    order of train.jsonl). Refused if an outcome has fewer families."""
    if per_outcome < 1:
        raise Refused(f"--per-outcome must be 1 or more, not {per_outcome}")
    taken = {outcome: 0 for outcome in OUTCOMES}
    ids = []
    for family in families:
        outcome = family["outcome"]
        if outcome in taken and taken[outcome] < per_outcome:
            taken[outcome] += 1
            ids.append(family["family_id"])
    short = {outcome: n for outcome, n in taken.items() if n < per_outcome}
    if short:
        raise Refused(f"the split has fewer than {per_outcome} families of an outcome: {short}")
    return ids


def exact_median(values: list[Fraction]) -> Fraction:
    """The median: the middle value, or the mean of the two middle values for an even number of values."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def step_for(median: Fraction) -> tuple[int, int]:
    """(attempts per scenario, scenarios per step) by the rule of PLAN 5.5. Exactly 0.8 is not "above"."""
    return LARGE_GROUPS if median > THRESHOLD else SMALL_GROUPS


def family_profile(records: list[dict]) -> dict[str, dict]:
    """For each family with a valid record: `passed`, `episodes` (valid records), `pass_rate` (the value of
    `metrics.family_pass_rates`) and `profile` (always_pass, always_fail or mixed), and `outcome`.

    The counts are taken here and the rate by the library; the two must agree to the last bit, or this
    script and the library do not count the same episodes."""
    rates = metrics.family_pass_rates(records)
    counts: dict[str, list] = {}
    for record in metrics.valid_records(records):
        row = counts.setdefault(record["family_id"], [0, 0, record["outcome"]])
        row[0] += bool(record["passed"])
        row[1] += 1
    if set(counts) != set(rates) or any(rates[fid] != passed / n for fid, (passed, n, _) in counts.items()):
        raise RuntimeError("the counts of this script do not give the family pass rates of the library")
    profile = {}
    for family_id in sorted(counts):
        passed, n, outcome = counts[family_id]
        kind = "always_pass" if passed == n else "always_fail" if passed == 0 else "mixed"
        profile[family_id] = {"outcome": outcome, "passed": passed, "episodes": n,
                              "pass_rate": rates[family_id], "profile": kind}
    return profile


def median_of(rows: list[dict]) -> Fraction:
    """The median family pass rate of a set of family rows, as a whole fraction."""
    return exact_median([Fraction(row["passed"], row["episodes"]) for row in rows])


def summary_of(rows: list[dict]) -> dict:
    """The numbers of a set of family rows: how many, the median of their pass rates, and the three counts."""
    median = median_of(rows)
    return {"n_families": len(rows), "median_family_pass_rate": float(median),
            "median_as_fraction": f"{median.numerator}/{median.denominator}",
            **{f"n_{kind}": sum(1 for row in rows if row["profile"] == kind) for kind in PROFILES}}


def decide(records: list[dict]) -> dict:
    """The first part of decision.json: the profile of the records and the step that the rule gives."""
    profile = family_profile(records)
    if not profile:
        raise Refused("no record is valid (every episode is a harness error): there is no profile")
    rows = list(profile.values())
    group_size, scenarios_per_step = step_for(median_of(rows))
    by_outcome = {}
    for outcome in [*OUTCOMES, *sorted({row["outcome"] for row in rows} - set(OUTCOMES))]:
        of_outcome = [row for row in rows if row["outcome"] == outcome]
        if of_outcome:
            by_outcome[outcome] = summary_of(of_outcome)
    per_scenario: dict[str, int] = {}
    for record in records:
        per_scenario[record["scenario_id"]] = per_scenario.get(record["scenario_id"], 0) + 1
    attempts: dict[str, int] = {}
    for n in sorted(per_scenario.values()):
        attempts[str(n)] = attempts.get(str(n), 0) + 1
    whole = summary_of(rows)
    return {
        "median_family_pass_rate": whole["median_family_pass_rate"],
        "median_as_fraction": whole["median_as_fraction"],
        "rule": RULE, "threshold": float(THRESHOLD),
        "group_size": group_size, "scenarios_per_step": scenarios_per_step,
        "n_families": whole["n_families"],
        **{f"n_{kind}": whole[f"n_{kind}"] for kind in PROFILES},
        "pass_rate": metrics.pass_rate(records),
        "episodes": len(records), "harness_errors": metrics.harness_error_count(records),
        "attempts_per_scenario": attempts,
        "by_outcome": by_outcome,
        "families": profile,
    }


def to_markdown(decision: dict) -> str:
    """profile.md: the rule and its result, then one table row per outcome and one row for all."""
    source = decision["source"]
    lines = [
        "# The family profile after SFT", "",
        f"Input: `{source['episodes']['path']}`, {decision['episodes']} episodes of the arm "
        f"`{source['arm']}` on {decision['n_families']} training families "
        f"({decision['harness_errors']} harness errors, left out). Adapter: `{source['adapter']}`.", "",
        "A family's pass rate is the share of its episodes that passed, over its wordings and attempts.", "",
        "| Outcome | Families | Always pass | Always fail | Mixed | Median family pass rate |",
        "|---|---|---|---|---|---|"]
    rows = [*decision["by_outcome"].items(), ("all", decision)]
    for name, row in rows:
        lines.append(f"| {name} | {row['n_families']} | {row['n_always_pass']} | {row['n_always_fail']} | "
                     f"{row['n_mixed']} | {row['median_family_pass_rate']:.3f} |")
    lines += [
        "", f"Rule, set before the runs (PLAN 5.5): {decision['rule']}.", "",
        f"Median family pass rate: {decision['median_family_pass_rate']:.4f} "
        f"({decision['median_as_fraction']}). Result: groups of {decision['group_size']} attempts on "
        f"{decision['scenarios_per_step']} scenarios per step.", "",
        f"Mean family pass rate (`metrics.pass_rate`): {decision['pass_rate']:.4f}. "
        f"Records per scenario (records: scenarios): {decision['attempts_per_scenario']}.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scenarios", type=Path, required=True,
                        help="the scenario folder (summary.json and train.jsonl)")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--pick", type=Path, default=None, metavar="FILE",
                      help="write the family ids for the profile into FILE, and do nothing else")
    what.add_argument("--episodes", type=Path, default=None,
                      help="episodes.jsonl of the rollout of the SFT model on these families")
    parser.add_argument("--per-outcome", type=int, default=PER_OUTCOME,
                        help="with --pick: families of each outcome")
    parser.add_argument("--families", type=Path, default=None,
                        help="with --episodes: the file that --pick wrote; the records must be of exactly "
                             "these families")
    parser.add_argument("--out", type=Path, default=None, help="with --episodes: the output folder")
    return parser.parse_args(argv)


def run_pick(args: argparse.Namespace) -> int:
    train = train_file(args.scenarios)
    ids = pick_families(load_families(train["path"]), args.per_outcome)
    args.pick.parent.mkdir(parents=True, exist_ok=True)
    args.pick.write_bytes("".join(f"{family_id}\n" for family_id in ids).encode("utf-8"))
    print(f"wrote {len(ids)} family ids ({args.per_outcome} of each outcome, the first ones of "
          f"{train['path']}) to {args.pick}")
    return 0


def run_profile(args: argparse.Namespace) -> int:
    if args.out is None:
        raise Refused("--episodes needs --out, the output folder")
    records = read_train_records(args.episodes)
    manifest = rollout_manifest(args.episodes, records)
    check_adapter(manifest)
    train = train_file(args.scenarios)
    wanted_ids = None if args.families is None else read_ids(args.families)
    check_source(records, manifest, train, load_families(train["path"]), wanted_ids)
    decision = decide(records)
    decision["source"] = {
        "episodes": file_entry(args.episodes), "run_manifest": manifest["entry"],
        "run": records[0]["run"], "arm": records[0]["arm"],
        **{key: manifest.get(key) for key in ("model", "revision", "adapter", "split", "samples", "run_seed",
                                              "scenarios", "split_file_sha256", "text_sha256",
                                              "config_sha256")},
        "max_new_tokens": (manifest.get("config") or {}).get("max_new_tokens"),
        "adapter_sha256": adapter_hashes(manifest["adapter"]),
        "train_file": train,
        "families_file": None if args.families is None else file_entry(args.families),
    }
    decision["script"] = file_entry(Path(__file__).resolve())
    decision["script"]["path"] = SCRIPT
    args.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(decision, indent=2, ensure_ascii=False) + "\n"
    (args.out / DECISION_NAME).write_bytes(text.encode("utf-8"))
    (args.out / TABLE_NAME).write_bytes(to_markdown(decision).encode("utf-8"))
    print(to_markdown(decision))
    print(f"wrote {args.out / DECISION_NAME} and {args.out / TABLE_NAME}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_pick(args) if args.pick is not None else run_profile(args)
    except Refused as refusal:
        print(f"sft_profile: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
