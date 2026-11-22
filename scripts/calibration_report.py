"""The calibration report of one or more saved runs: report.json, report.md, and the markdown on the screen.

    .venv/bin/python scripts/calibration_report.py --run 4b=results/calibration/round1_4b/episodes.jsonl \\
        --run 2b=results/calibration/round1_2b/episodes.jsonl --scenarios data/scenarios \\
        --out results/calibration/round1_report

Each --run is NAME=path to an episodes.jsonl file (one episode record per line). A run_manifest.json in the
same folder is read if it is there: it gives the model and the config. The first run is the one that the
others are set against in the side-by-side part.

--scenarios is the folder of the family files (train.jsonl, dev.jsonl, test.jsonl, calib.jsonl). The family
records hold the correct end state, which two numbers of section 8 need. Without it the report says that
these numbers are missing. No model and no GPU are used: every number comes from the saved records, through
`accessdesk/eval/calibration.py`, where each number is defined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.eval import calibration  # noqa: E402
from accessdesk.harness.records import read_records  # noqa: E402

MANIFEST_NAME = "run_manifest.json"


def shown_path(path: Path) -> str:
    """The path as the report shows it: relative to the repository root, with "/", if the file is inside
    the repository, so that the report does not depend on the folder the script is started from or on how
    the path was typed. A file outside the repository is shown as it was typed."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_manifest(records_path: Path) -> dict | None:
    """The run manifest beside the records file, or None if there is none."""
    path = records_path.parent / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_run_families(scenarios_dir: Path, records: list[dict],
                      manifest: dict | None) -> tuple[list[dict] | None, str | None]:
    """The family records of the run's splits, or (None, why they cannot be used).

    They cannot be used if a split's file is missing, if the file's hash is not the one that the run
    manifest (`split_file_sha256`) or the records (`data_sha256`) name, or if the folder was built from
    another text file than the run's."""
    from accessdesk.env.scenarios import load_families

    families: list[dict] = []
    for split in sorted({r["split"] for r in records}):
        path = scenarios_dir / f"{split}.jsonl"
        if not path.is_file():
            return None, f"{path} does not exist"
        named = {r["data_sha256"] for r in records if r["split"] == split and r.get("data_sha256")}
        if manifest and manifest.get("split") == split and manifest.get("split_file_sha256"):
            named.add(manifest["split_file_sha256"])
        if named and named != {hashlib.sha256(path.read_bytes()).hexdigest()}:
            return None, (f"{path} is not the file that the run was made from (the run manifest or the "
                          f"records name another SHA-256)")
        families += load_families(path)
    summary_path = scenarios_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("text_sha256") not in {r["text_sha256"] for r in records}:
            return None, f"the scenarios in {scenarios_dir} were built from another text file than the run's"
    return families, None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration report from saved episode records.")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                        help="a run and its episodes.jsonl; give it once per run; the first is the reference")
    parser.add_argument("--scenarios", type=Path, default=None,
                        help="folder with the family files of the runs (for the correct end states)")
    parser.add_argument("--out", type=Path, default=None, help="folder for report.json and report.md")
    parser.add_argument("--draws", type=int, default=10_000, help="bootstrap draws")
    parser.add_argument("--seed", type=int, default=0, help="seed of the bootstrap")
    parser.add_argument("--level", type=float, default=0.95, help="level of the intervals")
    return parser.parse_args(argv)


def read_runs(args: argparse.Namespace) -> dict[str, dict]:
    """For each run: records, manifest, families, scenarios note, source. ValueError for a bad --run."""
    runs: dict[str, dict] = {}
    for item in args.run:
        name, found, path_text = item.partition("=")
        if not found or not name or not path_text:
            raise ValueError(f"--run must look like NAME=PATH, got {item!r}")
        if name in runs:
            raise ValueError(f"--run {name!r} is given twice")
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"run {name!r}: {path} does not exist")
        records = read_records(path)
        manifest = read_manifest(path)
        families, note = None, "--scenarios was not given"
        if args.scenarios is not None and records:
            families, note = load_run_families(args.scenarios, records, manifest)
        runs[name] = {"records": records, "manifest": manifest, "families": families, "note": note,
                      "source": shown_path(path)}
    return runs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runs = read_runs(args)
        report = calibration.build_report(
            {name: run["records"] for name, run in runs.items()},
            families_by_run={name: run["families"] for name, run in runs.items()
                             if run["families"] is not None},
            manifests={name: run["manifest"] for name, run in runs.items() if run["manifest"] is not None},
            scenarios_notes={name: run["note"] for name, run in runs.items() if run["note"] is not None},
            sources={name: run["source"] for name, run in runs.items()},
            n_draws=args.draws, seed=args.seed, level=args.level)
    except ValueError as error:
        print(f"calibration_report: {error}", file=sys.stderr)
        return 2
    markdown = calibration.to_markdown(report)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
        (args.out / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
