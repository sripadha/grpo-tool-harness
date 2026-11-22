"""Count the word 5-grams that the request wordings of dev, test and calib share with those of train.

    .venv/bin/python scripts/check_wordings.py --scenarios data/scenarios_placeholder
    .venv/bin/python scripts/check_wordings.py --scenarios data/scenarios --out results/wordings.json

PLAN 5.3: "A script counts shared word 5-grams between train and dev or test wordings, per style, after names
and roles are masked."

Masking. In every wording the person name, the department and the role name of its own request are replaced
by <person>, <department> and <role>. The text is then put in lower case and cut at white space; a 5-gram is
five words in a row.

One row per evaluation split and wording style:
    wordings            distinct masked wordings of that split and style
    five_grams          distinct 5-grams in them
    shared_same_style   of these, the 5-grams that are also in a train wording of the same style
    shared_any_style    of these, the 5-grams that are also in a train wording of any style

What to expect. The seen styles (1 to 3, the generator's TRAIN_STYLES) use the same templates in every
split, so their 5-grams are shared: that is the design, the worlds and the names are what differs. The
held-out style (4) must be in no train family, and it must share no 5-gram with train: a shared one means
that a held-out wording, or part of a held-out template, is also in the training data. The script ends with
code 1 if a train family has a held-out style, or if a held-out row shares a 5-gram with train.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.env.generator import TRAIN_STYLES  # noqa: E402
from accessdesk.env.scenarios import load_families  # noqa: E402

TRAIN_SPLIT = "train"
EVAL_SPLITS = ("dev", "test", "calib")
MASKS = (("person_name", "<person>"), ("department", "<department>"), ("role_name", "<role>"))
N = 5


def masked_words(text: str, request: dict) -> list[str]:
    """The words of one wording, in lower case, with the three values of its request replaced by their
    masks. The longest value is replaced first, so a value that is part of another one does no harm."""
    for key, mask in sorted(MASKS, key=lambda item: -len(request[item[0]])):
        text = text.replace(request[key], f" {mask} ")
    return text.lower().split()


def n_grams(words: list[str], n: int = N) -> set[tuple[str, ...]]:
    """Every run of n words in a row. A text with fewer than n words has none."""
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def by_style(families: list[dict]) -> dict[int, dict]:
    """style -> {"wordings": the distinct masked wordings, "grams": their 5-grams}, styles in rising order."""
    found: dict[int, dict] = {}
    for family in families:
        for wording in family["wordings"]:
            words = masked_words(wording["text"], family["request"])
            entry = found.setdefault(wording["style"], {"wordings": set(), "grams": set()})
            entry["wordings"].add(" ".join(words))
            entry["grams"] |= n_grams(words)
    return {style: found[style] for style in sorted(found)}


def shared_table(families_by_split: dict[str, list[dict]]) -> list[dict]:
    """The rows described at the top of this file, for every split other than train, in the given order."""
    train = by_style(families_by_split[TRAIN_SPLIT])
    train_any = set().union(*(entry["grams"] for entry in train.values())) if train else set()
    rows = []
    for split, families in families_by_split.items():
        if split == TRAIN_SPLIT:
            continue
        for style, entry in by_style(families).items():
            same_style = train.get(style, {"grams": set()})["grams"]
            rows.append({"split": split, "style": style, "held_out": style not in TRAIN_STYLES,
                         "wordings": len(entry["wordings"]), "five_grams": len(entry["grams"]),
                         "shared_same_style": len(entry["grams"] & same_style),
                         "shared_any_style": len(entry["grams"] & train_any)})
    return rows


def problems(rows: list[dict], train_families: list[dict]) -> list[str]:
    """One line if train families have a wording in a held-out style, and one line for each row of a
    held-out style that shares a 5-gram with train."""
    in_train = sorted({wording["style"] for family in train_families for wording in family["wordings"]}
                      - set(TRAIN_STYLES))
    found = [f"train has wordings in the held-out style {style}" for style in in_train]
    return found + [f"{row['split']}, style {row['style']} (held out): {row['shared_any_style']} of "
                    f"{row['five_grams']} 5-grams are also in a train wording"
                    for row in rows if row["held_out"] and row["shared_any_style"] > 0]


def to_text(rows: list[dict]) -> str:
    lines = ["split  style  held out  wordings  5-grams  shared, same style  shared, any style"]
    for row in rows:
        lines.append(f"{row['split']:5s}  {row['style']:5d}  {'yes' if row['held_out'] else 'no':8s}  "
                     f"{row['wordings']:8d}  {row['five_grams']:7d}  {row['shared_same_style']:18d}  "
                     f"{row['shared_any_style']:17d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scenarios", type=Path, required=True, help="folder with the split files")
    parser.add_argument("--out", type=Path, default=None, help="file for the table as JSON")
    args = parser.parse_args(argv)

    splits = (TRAIN_SPLIT,) + EVAL_SPLITS
    families_by_split = {split: load_families(args.scenarios / f"{split}.jsonl") for split in splits}
    rows = shared_table(families_by_split)
    found = problems(rows, families_by_split[TRAIN_SPLIT])
    print(to_text(rows))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table = {"n": N, "rows": rows, "problems": found}
        args.out.write_bytes((json.dumps(table, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    for line in found:
        print(f"PROBLEM: {line}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
