"""Put the name of the person who read the text file into its header.

    .venv/bin/python scripts/approve_text.py --name "NAME"            (text/model_text.yaml)
    .venv/bin/python scripts/approve_text.py --name "NAME" --text FILE

The header of the text file ends with the line

    # Read and approved by: (not yet)

This script replaces "(not yet)" in that line with the name and the date of today in UTC:

    # Read and approved by: NAME, 2026-11-18 (UTC)

and changes no other byte of the file. Run it only after you have read the whole file: the freeze
(scripts/freeze.py) refuses a text file that still says "(not yet)".

It refuses, and leaves the file as it is, if
  - the name is empty, has more than one line (every Unicode line end counts), cannot be written as
    UTF-8, or holds "(not yet)";
  - the header does not hold a line "# Read and approved by:" exactly once, or that line is not exactly
    the line above (for example because a name is already there);
  - the file is not a valid text file, or it is a placeholder;
  - the new file is not a valid text file, or it does not give the same values as the old one when
    accessdesk.text loads both.

How the file is written: the new bytes go into a file beside the text file. That file is loaded and
compared, value by value, with the old one. Only then is it moved over the text file.
The scenario files name the hash of the text file. So approve the text first, and build the final
scenarios after that.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.freeze import APPROVAL_PREFIX, NOT_APPROVED, header_lines  # noqa: E402
from accessdesk.text import TextBundle, TextError, load_text, model_text_path  # noqa: E402

# The header line as the drafting program writes it, and the fields of a loaded text file that are not
# values of the text: the path of the file and the hash of its bytes.
OPEN_LINE = f"{APPROVAL_PREFIX} {NOT_APPROVED}".encode("utf-8")
NOT_COMPARED = ("path", "sha256")


class Refused(Exception):
    """The file was not changed. The message says why."""


def utc_today() -> str:
    """The date of today in UTC, in ISO form: 2026-11-18."""
    return datetime.now(timezone.utc).date().isoformat()


def checked_name(name: str) -> str:
    """The name without the white space around it. Raises Refused for a name that cannot be written."""
    name = name.strip()
    if not name:
        raise Refused("the name is empty")
    if len(name.splitlines()) != 1:           # splitlines knows every line end, also U+0085 and U+2028
        raise Refused("the name must be one line: it holds a line end")
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Refused("the name cannot be written as UTF-8") from error
    if NOT_APPROVED in name:
        raise Refused(f"the name must not hold {NOT_APPROVED!r}")
    return name


def approval_line_index(raw: bytes) -> int:
    """The index of the header line "# Read and approved by: (not yet)" among the lines of the file
    (split at "\\n"). Raises Refused unless the header holds exactly one line that starts with
    "# Read and approved by:", and that line is exactly this one."""
    header = header_lines(raw)
    found = [index for index, line in enumerate(header) if line.startswith(APPROVAL_PREFIX)]
    if len(found) != 1:
        raise Refused(f"the header must hold a line '{APPROVAL_PREFIX}' exactly once, not {len(found)} times")
    line = raw.split(b"\n")[found[0]]
    if line != OPEN_LINE:                     # shown as bytes: a "\r" or a space at the end can be seen
        raise Refused(f"the header line is not exactly {OPEN_LINE!r}: it is {line!r}")
    return found[0]


def approved_bytes(raw: bytes, name: str, today: str) -> bytes:
    """The bytes of the file with "(not yet)" in the approval line replaced by "NAME, DATE (UTC)".
    Every other byte is the same. Raises Refused (see approval_line_index, checked_name)."""
    lines = raw.split(b"\n")
    index = approval_line_index(raw)
    value = f"{checked_name(name)}, {today} (UTC)".encode("utf-8")
    lines[index] = lines[index].replace(NOT_APPROVED.encode("utf-8"), value)
    return b"\n".join(lines)


def values_of(bundle: TextBundle) -> dict:
    """Every value of a loaded text file, by field name, without the path of the file and its hash."""
    return {key: value for key, value in dataclasses.asdict(bundle).items() if key not in NOT_COMPARED}


def approve(path: Path, name: str, today: str) -> str:
    """Write the approval into the text file at `path` and return the new header line.

    Raises Refused, with the file unchanged, in every case that the module text lists."""
    raw = path.read_bytes()
    try:
        before = load_text(path)
    except TextError as error:
        raise Refused(f"the file is not a valid text file: {str(error).splitlines()[0]}") from error
    if before.is_placeholder:
        raise Refused("the file is a placeholder (meta.placeholder is true): there is nothing to approve")
    index = approval_line_index(raw)
    new_raw = approved_bytes(raw, name, today)
    temporary = path.with_name(path.name + ".approving")
    temporary.write_bytes(new_raw)
    try:
        if values_of(load_text(temporary)) != values_of(before):
            raise Refused("the new file does not give the same values as the old one")
        if path.read_bytes() != raw:
            raise Refused("the file was changed by another program while this script ran")
        os.replace(temporary, path)
    except TextError as error:
        raise Refused(f"the new file is not a valid text file: {str(error).splitlines()[0]}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return new_raw.split(b"\n")[index].decode("utf-8")


def main(argv: list[str] | None = None, *, today: str | None = None) -> int:
    """Approve the text file. `today` replaces the clock (for tests): a UTC date in ISO form."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--name", required=True, help="the name of the person who read the text")
    parser.add_argument("--text", type=Path, default=model_text_path(), help="the text file (YAML)")
    args = parser.parse_args(argv)
    try:
        line = approve(args.text, args.name, today or utc_today())
    except Refused as refusal:
        print(f"approve_text: refused, {args.text} is unchanged: {refusal}", file=sys.stderr)
        return 1
    print(f"{args.text}: {line}")
    print("The hash of the file is new. Build the scenarios from this file (scripts/build_scenarios.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
