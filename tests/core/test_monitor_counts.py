"""The monitor counts among the top-reward attempts (results/main/audit/monitor_counts/count.py): a fresh run
gives the saved files, and its top-reward totals are the row "passed with the full reward" of the monitor
reports. Reads the saved episodes of the four arms on dev and test."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "results" / "main" / "audit" / "monitor_counts"
NAMES = ("top_reward_flags.md", "top_reward_flags.json")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("monitor_counts")
    result = subprocess.run([sys.executable, str(FOLDER / "count.py"), "--out", str(out)], cwd=ROOT,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def test_a_fresh_run_gives_the_saved_files(fresh):
    for name in NAMES:
        assert (fresh / name).read_bytes() == (FOLDER / name).read_bytes(), name
        assert str(ROOT) not in (fresh / name).read_text(encoding="utf-8")       # no absolute path


def test_the_top_reward_totals_are_those_of_the_monitor_reports(fresh):
    rows = json.loads((fresh / "top_reward_flags.json").read_text(encoding="utf-8"))["rows"]
    single = [row for row in rows if "+" not in row["arm"]]
    assert len(single) == 8
    for row in single:
        report = (ROOT / "results" / "main" / "monitors" / f"{row['arm']}_{row['split']}.md").read_text(
            encoding="utf-8")
        saved = re.search(r"^\| passed with the full reward \| (\d+) \|", report, re.MULTILINE)
        assert saved is not None and int(saved.group(1)) == row["n_top_reward"], row
    for split in ("dev", "test"):
        together = [row for row in rows if row["split"] == split and "+" in row["arm"]]
        seeds = [row for row in single if row["split"] == split and row["arm"].startswith("grpo_seed")]
        assert len(together) == 1 and len(seeds) == 2
        assert together[0]["n_top_reward"] == sum(row["n_top_reward"] for row in seeds)
