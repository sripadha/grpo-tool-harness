"""Freeze the project, check the freeze, record a change after it, or show the frozen groups.

    .venv/bin/python scripts/freeze.py show
    .venv/bin/python scripts/freeze.py write --signed-by "NAME" [--note TEXT]
    .venv/bin/python scripts/freeze.py check
    .venv/bin/python scripts/freeze.py amend --reason TEXT

show    A short table: group, number of files, group hash. From MANIFEST.json if it is there, else from
        the tree as it is now (what a freeze would write).
write   Writes MANIFEST.json with the hashes of every frozen file (accessdesk/freeze.py, FROZEN_GROUPS).
        It refuses, with the list of reasons, while the tree must not be frozen
        (problems_before_freeze), and it never replaces a MANIFEST.json that is there.
check   Compares the tree with MANIFEST.json. Exit code 0: nothing changed. Exit code 1: the list of
        differences is printed. Exit code 2: there is no MANIFEST.json to compare with.
amend   For a frozen file that had to change after the freeze: writes the new hashes into MANIFEST.json
        and adds one entry to its "amendments": the time, the reason, and for each changed file the old
        and the new hash. It refuses an empty reason, a tree without a difference, and a tree that has a
        problem (tree_problems). Results from before and after an amendment are separate experiments
        (PLAN 5.3).

Every time is UTC, in ISO form (2026-11-18T10:54:40Z).

The tree is the repository of this script. For tests, another folder can be given with --root (before
the command) or with the environment variable ACCESSDESK_FREEZE_ROOT; --root wins.
No command changes a frozen file. `write` and `amend` write MANIFEST.json and nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk import freeze  # noqa: E402

ROOT_VARIABLE = "ACCESSDESK_FREEZE_ROOT"


def utc_now() -> str:
    """The time now in UTC, in ISO form with seconds: 2026-11-18T10:54:40Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tree_root(given: Path | None) -> Path:
    """The folder to work on: --root, else the environment variable, else the repository of this script."""
    if given is not None:
        return given
    named = os.environ.get(ROOT_VARIABLE, "")
    return Path(named) if named else ROOT


def manifest_bytes(manifest: dict) -> bytes:
    """The bytes of MANIFEST.json: indented JSON, keys in the order of the dict, UTF-8, "\\n" at the end."""
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_manifest(root: Path) -> dict | None:
    """The manifest of the tree, or None if MANIFEST.json is not there."""
    path = root / freeze.MANIFEST_NAME
    return json.loads(path.read_bytes().decode("utf-8")) if path.is_file() else None


def print_list(title: str, lines: list[str]) -> None:
    """A title and one "- " line per entry, on stderr."""
    print(title, file=sys.stderr)
    for line in lines:
        print(f"- {line}", file=sys.stderr)


def group_table(groups: dict[str, dict]) -> list[str]:
    """The lines of the table: group, number of files, group hash."""
    width = max(len(group) for group in [*groups, "group"])
    lines = [f"{'group':{width}s}  files  group hash (SHA-256)"]
    lines += [f"{group:{width}s}  {len(entry['files']):5d}  {entry['sha256']}"
              for group, entry in groups.items()]
    return lines


def write(root: Path, signed_by: str, note: str, now: str) -> int:
    """The command `write`. 0: MANIFEST.json was written. 1: refused, and nothing was written."""
    problems = freeze.problems_before_freeze(root, signed_by=signed_by)
    if problems:
        print_list(f"not frozen: {len(problems)} problem(s)", problems)
        return 1
    manifest = freeze.build_manifest(root, signed_by=signed_by, frozen_at=now, note=note)
    # Other work may change a file while the hashes are taken. So the tree is asked once more, after the
    # manifest is built: the manifest that is written describes a tree without a problem.
    moved = freeze.problems_before_freeze(root, signed_by=signed_by) + freeze.check_manifest(root, manifest)
    if moved:
        print_list("not frozen: the tree changed while it was read; run the command again", moved)
        return 1
    try:
        with (root / freeze.MANIFEST_NAME).open("xb") as handle:      # "x": never replace a manifest
            handle.write(manifest_bytes(manifest))
    except FileExistsError:
        print(f"not frozen: {freeze.MANIFEST_NAME} exists already", file=sys.stderr)
        return 1
    n_files = sum(len(entry["files"]) for entry in manifest["groups"].values())
    print(f"wrote {root / freeze.MANIFEST_NAME}: {n_files} files in {len(manifest['groups'])} groups, "
          f"frozen at {now}, signed by {manifest['signed_by']}")
    print(f"code_sha256 {manifest['code_sha256']}")
    uncommitted = manifest["git"]["uncommitted_frozen_files"]
    if uncommitted:
        print(f"note: {len(uncommitted)} frozen file(s) are not in the commit {manifest['git']['commit']} "
              f"as they are now. Commit them together with {freeze.MANIFEST_NAME}.")
    return 0


def check(root: Path) -> int:
    """The command `check`. 0: nothing changed. 1: differences (printed). 2: no MANIFEST.json."""
    manifest = read_manifest(root)
    if manifest is None:
        print(f"{root / freeze.MANIFEST_NAME} is missing: the tree is not frozen", file=sys.stderr)
        return 2
    differences = freeze.check_manifest(root, manifest)
    if differences:
        print_list(f"{len(differences)} difference(s) between the tree and {freeze.MANIFEST_NAME}",
                   differences)
        return 1
    n_files = sum(len(entry["files"]) for entry in manifest["groups"].values())
    print(f"nothing changed: {n_files} files in {len(manifest['groups'])} groups are as in "
          f"{freeze.MANIFEST_NAME} (code_sha256 {manifest['code_sha256']})")
    return 0


def amend(root: Path, reason: str, now: str) -> int:
    """The command `amend`. 0: MANIFEST.json holds the new hashes and one more amendment. 1: refused,
    and the file is as it was. 2: no MANIFEST.json, or no reason."""
    manifest = read_manifest(root)
    if manifest is None:
        print(f"{root / freeze.MANIFEST_NAME} is missing: there is nothing to amend", file=sys.stderr)
        return 2
    if not reason.strip():
        print("refused: an amendment needs a reason (--reason)", file=sys.stderr)
        return 2
    problems = freeze.tree_problems(root)
    if problems:
        print_list(f"not amended: {len(problems)} problem(s)", problems)
        return 1
    try:
        amended = freeze.amend_manifest(root, manifest, reason=reason, amended_at=now)
    except ValueError as error:
        print(f"not amended: {error}", file=sys.stderr)
        return 1
    # Written beside the file and then moved over it, so that MANIFEST.json is never half written.
    path = root / freeze.MANIFEST_NAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(manifest_bytes(amended))
    os.replace(temporary, path)
    entry = amended["amendments"][-1]
    print(f"amended {path} at {now}: amendment {len(amended['amendments'])}, {len(entry['changes'])} file(s)")
    for change in entry["changes"]:
        print(f"  {change['kind']:8s} {change['path']}  (group {change['group']})")
    print(f"code_sha256 {entry['before']['code_sha256']} -> {entry['after']['code_sha256']}")
    return 0


def show(root: Path) -> int:
    """The command `show`: the table of groups, from MANIFEST.json or, without it, from the tree."""
    manifest = read_manifest(root)
    if manifest is None:
        print(f"{freeze.MANIFEST_NAME} is not there. The tree as it is now (not frozen):")
        groups = freeze.hash_groups(root)
        print(f"code_sha256        {freeze.code_sha256(root)}")
    else:
        groups = manifest["groups"]
        seeds, design = manifest["data"]["seeds"], manifest["data"]["design"]
        print(f"{freeze.MANIFEST_NAME}: frozen at {manifest['frozen_at']}, signed by {manifest['signed_by']}")
        print(f"git commit         {manifest['git']['commit']}")
        print(f"code_sha256        {manifest['code_sha256']}")
        print(f"text_sha256        {manifest['text_sha256']}")
        print(f"run_config_sha256  {manifest['run_config_sha256']}")
        named = ", ".join(f"{key} {value}" for key, value in {**seeds, **design}.items())
        print(f"data               {named}")
        print(f"amendments         {len(manifest['amendments'])}")
        for number, entry in enumerate(manifest["amendments"], start=1):
            print(f"  {number}. {entry['amended_at']}  {len(entry['changes'])} file(s): {entry['reason']}")
    print()
    print("\n".join(group_table(groups)))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--root", type=Path, default=None,
                        help=f"the tree to work on (default: ${ROOT_VARIABLE}, else this repository)")
    commands = parser.add_subparsers(dest="command", required=True)
    writing = commands.add_parser("write", help="write MANIFEST.json")
    writing.add_argument("--signed-by", required=True, help="the name of the person who freezes")
    writing.add_argument("--note", default="", help="a free text that is kept in the manifest")
    commands.add_parser("check", help="compare the tree with MANIFEST.json")
    amending = commands.add_parser("amend", help="record a change of a frozen file")
    amending.add_argument("--reason", required=True, help="why a frozen file had to change")
    commands.add_parser("show", help="print the table of groups")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, now: str | None = None) -> int:
    """Run one command. `now` replaces the clock (for tests): a UTC time in ISO form."""
    args = parse_args(argv)
    root = tree_root(args.root)
    if args.command == "write":
        return write(root, args.signed_by, args.note, now or utc_now())
    if args.command == "check":
        return check(root)
    if args.command == "amend":
        return amend(root, args.reason, now or utc_now())
    return show(root)


if __name__ == "__main__":
    raise SystemExit(main())
