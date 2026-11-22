#!/usr/bin/env python3
"""Rebuild first_names.txt and last_names.txt from three raw US Census Bureau files.

Standard library only. Reads only the three raw files in .cache/names_raw/.
The same raw files always give byte-identical lists.

    python3 -I data/names/make_name_lists.py

Nothing is written unless every check passes. A short report is printed.
The sources and the selection rules are written out in PROVENANCE.md.
"""
import csv
import hashlib
import io
import json
import string
import sys
import zipfile
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / ".cache" / "names_raw"
# First names, 1990 census. Each line: name, frequency in percent, cumulative frequency, rank.
# https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first
# https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first
FIRST_NAME_FILES = {
    "female": RAW / "census_1990_dist.female.first",
    "male": RAW / "census_1990_dist.male.first",
}
# Last names, 2010 census. Columns: name, rank, count, ...
# https://www2.census.gov/topics/genealogy/2010surnames/names.zip
SURNAME_ZIP = RAW / "census_2010_surnames_names.zip"
SURNAME_MEMBER = "Names_2010Census.csv"
PER_SEX = 100
N_LAST = 200
CAPITALS = set(string.ascii_uppercase)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def capitalise(name):
    return name[0] + name[1:].lower()  # MARY -> Mary, SMITH -> Smith


def read_first_names(text):
    """Return [(name, frequency in percent, rank), ...] in rank order."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        name, percent, _cumulative, rank = line.split()
        if rows and int(rank) < rows[-1][2]:
            raise ValueError("file is not in rank order at %s" % name)
        rows.append((name, Decimal(percent), int(rank)))
    if len({name for name, _, _ in rows}) != len(rows):
        raise ValueError("a name is listed twice in one file")
    return rows


def pick_first_names(rows):
    """Pick 100 female and 100 male names from {"female": rows, "male": rows}.

    Returns (picked, skipped, last_rank). picked[sex] holds the names as spelled in
    the file. skipped[sex][reason] holds the names passed over before the 100th
    name was taken. last_rank[sex] is the rank of that 100th name.
    """
    percent_of = {sex: {name: percent for name, percent, _ in rows[sex]} for sex in rows}
    picked, skipped, last_rank = {}, {}, {}
    for sex, other in (("female", "male"), ("male", "female")):
        picked[sex] = []
        skipped[sex] = {
            "fewer than 3 letters": [],
            "character outside A-Z": [],
            "in both files, belongs to the other sex": [],
        }
        for name, percent, rank in rows[sex]:
            other_percent = percent_of[other].get(name)
            if len(name) < 3:
                skipped[sex]["fewer than 3 letters"].append(name)
            elif not set(name) <= CAPITALS:
                skipped[sex]["character outside A-Z"].append(name)
            # A name in both files is considered only under the sex where its
            # frequency in percent is higher (equal: female).
            elif other_percent is not None and (
                other_percent > percent or (other_percent == percent and sex == "male")
            ):
                skipped[sex]["in both files, belongs to the other sex"].append(name)
            else:
                picked[sex].append(name)
                if len(picked[sex]) == PER_SEX:
                    last_rank[sex] = rank
                    break
        else:
            raise ValueError("fewer than %d usable %s names" % (PER_SEX, sex))
    return picked, skipped, last_rank


def pick_last_names(text, first_names):
    """Pick the first 200 usable surnames, in rank order, from the 2010 CSV text.

    Returns (surnames, skipped, rank of the last surname taken).
    """
    taken = {name.lower() for name in first_names}
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader)
    if header[:3] != ["name", "rank", "count"]:
        raise ValueError("unexpected header: %r" % header)
    picked = []
    skipped = {
        "the row ALL OTHER NAMES": [],
        "starts with MC": [],
        "character outside A-Z": [],
        "same as a chosen first name": [],
    }
    previous_rank = 0
    for row in reader:
        name, rank = row[0], int(row[1])
        if name == "ALL OTHER NAMES":
            skipped["the row ALL OTHER NAMES"].append(name)
            continue
        if rank < previous_rank:
            raise ValueError("file is not in rank order at %s" % name)
        previous_rank = rank
        if name.startswith("MC"):
            skipped["starts with MC"].append(name)
        elif not name or not set(name) <= CAPITALS:
            skipped["character outside A-Z"].append(name)
        elif name.lower() in taken:
            skipped["same as a chosen first name"].append(name)
        else:
            picked.append(capitalise(name))
            if len(picked) == N_LAST:
                return picked, skipped, rank
    raise ValueError("fewer than %d usable surnames" % N_LAST)


def find_problems(first, last):
    """Return a list of broken promises about the two lists (empty when all hold)."""
    problems = []
    for label, names, wanted in (("first", first, 2 * PER_SEX), ("last", last, N_LAST)):
        if len(names) != wanted:
            problems.append("%s names: %d lines, wanted %d" % (label, len(names), wanted))
        if len({name.lower() for name in names}) != len(names):
            problems.append("%s names: a name appears twice" % label)
    both = sorted({name.lower() for name in first} & {name.lower() for name in last})
    if both:
        problems.append("in both files: %s" % ", ".join(both))
    return problems


def main():
    raw_files = list(FIRST_NAME_FILES.values()) + [SURNAME_ZIP]
    missing = [str(path) for path in raw_files if not path.is_file()]
    if missing:
        sys.exit("Nothing written. Raw file not found:\n  " + "\n  ".join(missing))

    report = {"raw_files": {}, "derived_files": {}}
    rows = {}
    for sex, path in FIRST_NAME_FILES.items():
        data = path.read_bytes()
        report["raw_files"][path.name] = {"sha256": sha256(data)}
        rows[sex] = read_first_names(data.decode("ascii"))
    with zipfile.ZipFile(SURNAME_ZIP) as archive:
        surname_data = archive.read(SURNAME_MEMBER)
    report["raw_files"][SURNAME_ZIP.name] = {
        "sha256": sha256(SURNAME_ZIP.read_bytes()),
        "file_used": SURNAME_MEMBER,
        "file_used_sha256": sha256(surname_data),
    }

    picked, skipped_first, last_rank = pick_first_names(rows)
    # Interleave by rank: female 1, male 1, female 2, male 2, ...
    first = [capitalise(name) for pair in zip(picked["female"], picked["male"]) for name in pair]
    last, skipped_last, last_surname_rank = pick_last_names(surname_data.decode("ascii"), first)

    problems = find_problems(first, last)
    if problems:
        sys.exit("Nothing written. Check failed:\n  " + "\n  ".join(problems))

    for filename, names in (("first_names.txt", first), ("last_names.txt", last)):
        data = ("\n".join(names) + "\n").encode("ascii")
        (HERE / filename).write_bytes(data)
        report["derived_files"][filename] = {"lines": len(names), "sha256": sha256(data)}
    report["skipped_first_names"] = skipped_first
    report["skipped_last_names"] = skipped_last
    report["checks"] = {
        "name_twice_in_a_file": False,
        "name_in_both_files": False,
        "rank_of_last_female_name_taken": last_rank["female"],
        "rank_of_last_male_name_taken": last_rank["male"],
        "rank_of_last_surname_taken": last_surname_rank,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
