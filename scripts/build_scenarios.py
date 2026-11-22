"""Generate the scenario files from a text file, the name files and the config.

    .venv/bin/python scripts/build_scenarios.py                                -> data/scenarios_placeholder/
    .venv/bin/python scripts/build_scenarios.py --text text/model_text.yaml    -> data/scenarios/

Without --text the placeholder text is used. Writes train.jsonl, dev.jsonl, test.jsonl, calib.jsonl (one
family per line), sets.json (the fixed dev subsets) and summary.json (counts, the hashes of the inputs and of
every written file, and every field of the config that the build used).
The output depends only on the text file, the two name files and the config: running the script twice gives
byte-identical files. No path, date or machine name is written.

The config of a build is accessdesk/config.py, changed for this one build by the options:
    --mix-facts            the mixed design: rule facts may be true together (generator.MIXED_KINDS)
    --no-mix-facts         the default design, whatever the config says
    --homonym-share X      the share of families with two people of the requested name
    --n-calib N            the number of calibration families
    --data-seed N, --calib-seed N    the seeds of train, dev and test, and of the calibration split
A build from the placeholder text takes the two placeholder seeds of the config in place of data_seed and
calib_seed (RunConfig.with_seeds_for), unless a seed option is given: so the worlds of the unit-test data
are never the worlds of a build from the real text.

The two calibration folders of round 2 (for calibration only; a folder holds the wordings, so build both
again whenever text/model_text.yaml changes). Every option is written out, so that the commands give the
same folders after a change of the defaults in accessdesk/config.py:
    .venv/bin/python scripts/build_scenarios.py --text text/model_text.yaml \
        --out data/scenarios_calib2_plain --no-mix-facts --homonym-share 0.2 --n-calib 24 \
        --data-seed 1 --calib-seed 2
    .venv/bin/python scripts/build_scenarios.py --text text/model_text.yaml \
        --out data/scenarios_calib2_mixed --mix-facts --homonym-share 0.5 --n-calib 48 \
        --data-seed 1 --calib-seed 2

The final folder is the one named `scenarios`. Its train, dev and test families must be new: the build stops
if one of their world seeds is in a scenario folder beside it (`scenarios_*`: the placeholder files and the
calibration files, which were read and rolled out during the work). So the data seed of
accessdesk/config.py must be changed before the final build (PLAN 5.3: "a fresh seed").
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.generator import (FACT_KEYS, MIXED_KINDS, SPLITS, kind_name,  # noqa: E402
                                      load_name_pools, make_all_splits)
from accessdesk.env.scenarios import load_families, make_sets, save_families, scenario_ids  # noqa: E402
from accessdesk.text import TextBundle, load_text, normalise, placeholder_path  # noqa: E402
from accessdesk.types import OUTCOMES  # noqa: E402

NAME_FILES = ("first_names.txt", "last_names.txt")
FINAL_DIR_NAME = "scenarios"                  # the folder of the final families
EARLIER_DIRS_PATTERN = "scenarios_*"          # beside it: folders of families that were already used
FRESH_SPLITS = ("train", "dev", "test")       # the splits of the final build that must be new


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def target_is_first_match(family: dict) -> bool:
    """True if the person meant is the first person of the world's list with the name of the request (the
    first match that `lookup_person` returns)."""
    name = normalise(family["request"]["person_name"])
    first = next(p for p in family["world"]["people"] if normalise(p["name"]) == name)
    return first["person_id"] == family["truth"]["person_id"]


def kind_counts(families: list[dict]) -> dict:
    """Mixed design: outcome -> kind (generator.kind_name) -> number of families, every kind of
    MIXED_KINDS in its order, also a kind that no family has."""
    counts = {outcome: {kind_name(outcome, kind): 0 for kind, _ in MIXED_KINDS[outcome]}
              for outcome in OUTCOMES}
    for family in families:
        facts = tuple(family["facts"][key] for key in FACT_KEYS)
        counts[family["outcome"]][kind_name(family["outcome"], facts)] += 1
    return counts


def split_summary(families: list[dict]) -> dict:
    """Counts of one split: families, scenarios, families per outcome, families with a homonym per outcome,
    and of the families with a homonym those in which the person meant is the first match. For the mixed
    design also the families of each kind per outcome, all of them and those with a homonym."""
    summary = {
        "families": len(families),
        "scenarios": sum(len(scenario_ids(family)) for family in families),
        "outcomes": {o: sum(f["outcome"] == o for f in families) for o in OUTCOMES},
        "homonyms": {o: sum(f["outcome"] == o and f["homonym"] for f in families) for o in OUTCOMES},
        "homonyms_person_meant_first": {o: sum(f["outcome"] == o and f["homonym"] and target_is_first_match(f)
                                               for f in families) for o in OUTCOMES},
    }
    if any("facts" in family for family in families):
        summary["kinds"] = kind_counts(families)
        summary["kinds_with_homonym"] = kind_counts([family for family in families if family["homonym"]])
    return summary


def names_summary(names_dir: Path) -> dict:
    """The hash of each name file, or null for a file that is missing (made-up tokens are then used)."""
    return {name: sha256_of_file(names_dir / name) if (names_dir / name).is_file() else None
            for name in NAME_FILES}


def earlier_dirs(out_dir: Path) -> list[Path]:
    """For the final folder: the scenario folders beside it whose families were already used. Else none."""
    if out_dir.name != FINAL_DIR_NAME:
        return []
    return sorted(path for path in out_dir.resolve().parent.glob(EARLIER_DIRS_PATTERN) if path.is_dir())


def reused_world_seeds(families: dict[str, list[dict]], earlier: list[Path]) -> list[str]:
    """One line for each earlier folder that holds a world seed of the new train, dev or test families."""
    new_seeds = {family["world_seed"] for split in FRESH_SPLITS for family in families[split]}
    found = []
    for folder in earlier:
        old_seeds = {family["world_seed"] for path in sorted(folder.glob("*.jsonl"))
                     for family in load_families(path)}
        if new_seeds & old_seeds:
            found.append(f"{len(new_seeds & old_seeds)} of the {len(new_seeds)} world seeds of the new "
                         f"train, dev and test families are in {folder}")
    return found


def build(text: TextBundle, names_dir: Path, out_dir: Path, cfg: RunConfig,
          earlier: list[Path] | None = None) -> dict:
    """Generate every split, write the six files into out_dir, and return the summary.

    `earlier`: folders of scenario files that were already used (default: `earlier_dirs(out_dir)`). Raises
    ValueError, before anything is written, if a world seed of the new train, dev or test families is in
    one of them."""
    pools = load_name_pools(names_dir)
    families = make_all_splits(text, pools, cfg)
    reused = reused_world_seeds(families, earlier_dirs(out_dir) if earlier is None else earlier)
    if reused:
        raise ValueError("the families of this build are not new: " + "; ".join(reused)
                         + ". Change data_seed in accessdesk/config.py and build again.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        save_families(out_dir / f"{split}.jsonl", families[split])
    write_json(out_dir / "sets.json", make_sets(families["dev"], cfg, text.policy_shift_variant))
    written = [f"{split}.jsonl" for split in SPLITS] + ["sets.json"]
    summary = {
        "text_file": Path(text.path).name,
        "text_sha256": text.sha256,
        "text_is_placeholder": text.is_placeholder,
        "name_files_sha256": names_summary(names_dir),
        "config": cfg.to_dict(),
        "config_sha256": cfg.sha256(),
        "splits": {split: split_summary(families[split]) for split in SPLITS},
        "files_sha256": {name: sha256_of_file(out_dir / name) for name in written},
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def build_config(args: argparse.Namespace, text_is_placeholder: bool) -> RunConfig:
    """The config of this build: the defaults of accessdesk/config.py, the placeholder seeds if the text
    is the placeholder, and then every option that was given on the command line."""
    cfg = RunConfig().with_seeds_for(text_is_placeholder)
    asked = {"mix_facts": args.mix_facts, "homonym_share": args.homonym_share,
             "n_calib": args.n_calib, "data_seed": args.data_seed, "calib_seed": args.calib_seed}
    return dataclasses.replace(cfg, **{name: value for name, value in asked.items() if value is not None})


def config_problem(cfg: RunConfig) -> str | None:
    """Why the generator cannot build with this config, or None."""
    if not 0.0 <= cfg.homonym_share <= 1.0:
        return f"--homonym-share must be between 0 and 1, not {cfg.homonym_share}"
    if cfg.n_calib < 0 or cfg.n_calib % len(OUTCOMES) != 0:
        return f"--n-calib must be a multiple of {len(OUTCOMES)}, not {cfg.n_calib}"
    if cfg.n_calib > 100:
        return f"--n-calib can be 100 at most (family ids ca-00 to ca-99), not {cfg.n_calib}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--text", type=Path, default=placeholder_path(), help="the text file (YAML)")
    parser.add_argument("--out", type=Path, default=None, help="the output folder")
    parser.add_argument("--names", type=Path, default=ROOT / "data" / "names",
                        help="the folder of the name files")
    parser.add_argument("--allow-made-up-names", action="store_true",
                        help="go on with made-up name tokens if a name file is missing")
    parser.add_argument("--mix-facts", action=argparse.BooleanOptionalAction, default=None,
                        help="the mixed design: rule facts may be true together (default: the config's)")
    parser.add_argument("--homonym-share", type=float, default=None,
                        help="share of the families with two people of the requested name")
    parser.add_argument("--n-calib", type=int, default=None, help="number of calibration families")
    parser.add_argument("--data-seed", type=int, default=None, help="seed of train, dev and test")
    parser.add_argument("--calib-seed", type=int, default=None, help="seed of the calibration split")
    args = parser.parse_args(argv)

    text = load_text(args.text)
    cfg = build_config(args, text.is_placeholder)
    problem = config_problem(cfg)
    if problem:
        print(f"build_scenarios: {problem}", file=sys.stderr)
        return 2
    out_dir = args.out or ROOT / "data" / ("scenarios_placeholder" if text.is_placeholder else "scenarios")
    missing = [name for name in NAME_FILES if not (args.names / name).is_file()]
    if missing and not args.allow_made_up_names:
        print(f"missing in {args.names}: {missing}. Fetch the name files, or pass --allow-made-up-names.",
              file=sys.stderr)
        return 2
    try:
        summary = build(text, args.names, out_dir, cfg)
    except ValueError as error:
        print(f"build_scenarios: {error}", file=sys.stderr)
        return 2

    print(f"wrote {out_dir}   (mix_facts {cfg.mix_facts}, homonym_share {cfg.homonym_share}, "
          f"data_seed {cfg.data_seed}, calib_seed {cfg.calib_seed})")
    for split, counts in summary["splits"].items():
        print(f"  {split:5s} families {counts['families']:3d}  scenarios {counts['scenarios']:3d}  "
              f"outcomes {counts['outcomes']}  homonyms {counts['homonyms']}")
        if "kinds" in counts:
            print(f"        kinds {counts['kinds']}")
    for name, digest in summary["files_sha256"].items():
        print(f"  {digest}  {name}")
    if missing:
        print("  NOTE: made-up name tokens were used, because a name file is missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
