"""The final scenario files: no leak between the splits, and SFT data from the train split only.

tests/core/test_generator.py checks the generator on families built in memory from the placeholder text.
This file reads the saved files that every run used: data/scenarios/{train,dev,test,calib}.jsonl and
results/main/sft_data/sft_records.jsonl. It needs no torch and no model, and runs in a few seconds:

    .venv/bin/python -m pytest tests/core/test_final_splits.py

What identifies a world: its world_seed (accessdesk/env/generator.py: "Everything random in a family comes
from Rng(world_seed)"), and its people (their person ids and their names).
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest

from accessdesk.env.generator import SPLITS, TRAIN_STYLES
from accessdesk.env.scenarios import load_families

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "data" / "scenarios"
SFT_RECORDS = ROOT / "results" / "main" / "sft_data" / "sft_records.jsonl"
SFT_REPORT = ROOT / "results" / "main" / "sft_data" / "sft_report.json"
HELD_OUT_STYLE = 4


@pytest.fixture(scope="module")
def final() -> dict[str, list[dict]]:
    return {split: load_families(SCENARIOS / f"{split}.jsonl") for split in SPLITS}


def test_the_files_have_the_counts_of_summary_json(final):
    summary = json.loads((SCENARIOS / "summary.json").read_text(encoding="utf-8"))
    for split in SPLITS:
        assert len(final[split]) == summary["splits"][split]["families"], split
        assert all(family["split"] == split for family in final[split]), split
        data = (SCENARIOS / f"{split}.jsonl").read_bytes()
        assert hashlib.sha256(data).hexdigest() == summary["files_sha256"][f"{split}.jsonl"], split


def _keys(families: list[dict]) -> dict[str, set]:
    """Everything that must not be in two splits, by kind."""
    return {
        "family id": {f["family_id"] for f in families},
        "world seed": {f["world_seed"] for f in families},
        "world": {json.dumps(f["world"], sort_keys=True) for f in families},
        "person id": {p["person_id"] for f in families for p in f["world"]["people"]},
        "person name": {p["name"] for f in families for p in f["world"]["people"]},
        "request text": {w["text"] for f in families for w in f["wordings"]},
    }


def test_no_family_world_person_or_request_text_is_in_two_splits(final):
    keys = {split: _keys(final[split]) for split in SPLITS}
    for split in SPLITS:
        assert all(keys[split][kind] for kind in keys[split]), split          # nothing is empty
    for a, b in itertools.combinations(SPLITS, 2):
        for kind in keys[a]:
            shared = keys[a][kind] & keys[b][kind]
            assert not shared, f"{kind} in {a} and {b}: {sorted(map(str, shared))[:3]}"


def test_no_training_scenario_has_the_held_out_style(final):
    styles = {w["style"] for f in final["train"] for w in f["wordings"]}
    assert HELD_OUT_STYLE not in TRAIN_STYLES
    assert styles == set(TRAIN_STYLES)
    for split in ("dev", "test"):          # the held-out style is there, so the check above can tell
        assert HELD_OUT_STYLE in {w["style"] for f in final[split] for w in f["wordings"]}, split


def test_sft_records_come_only_from_train_families(final):
    train_ids = {f["family_id"] for f in final["train"]}
    train_sha = hashlib.sha256((SCENARIOS / "train.jsonl").read_bytes()).hexdigest()
    n = 0
    with SFT_RECORDS.open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            n += 1
            assert record["split"] == "train", record["scenario_id"]
            assert record["family_id"] in train_ids, record["scenario_id"]
            assert record["style"] in TRAIN_STYLES, record["scenario_id"]
            assert record["scenario_id"] == f"{record['family_id']}/s{record['style']}"
            assert record["data_sha256"] == train_sha, record["scenario_id"]
    report = json.loads(SFT_REPORT.read_text(encoding="utf-8"))
    assert n == report["selection"]["n_chosen"]
