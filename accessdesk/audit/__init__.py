"""The audit of ANALYSIS_PLAN.md section 8, and the numbers of the second desk (section 3).

Written after the freeze. Everything here only reads saved attempts: no model is used, no attempt is made
again by a model, and no saved file is changed. The one thing that is made new is scripted: the probe
attempts of the positive control are played in memory by the scripted policies, and nothing of them is
written.

    second_desk.py   section 3: the pass rate under the changed rule, and the share of attempts that still
                     give the role
    sample.py        section 8, lines 1 and 3: which attempts a person reads (a seeded draw, one seeded
                     order, neutral ids)
    sheet.py         the blind reading sheet, the labels form, the key, and the check of the filled form
    formula.py       section 8, line 4: the formula audit of the saved GRPO training rollouts
    broken.py        section 8, line 5: three verifiers that are broken on purpose, the scripted probe
                     attempts, and the positive control

The scripts in scripts/ (second_desk_report.py, audit_sheet.py, formula_audit.py,
audit_positive_control.py) read the files and print; the modules here hold the rules.

Two helpers are shared. `load_script` gives the module of a script of scripts/, so that the audit uses the
functions of scripts/replay_check.py and scripts/monitor_report.py and does not write them a second time.
`load_run` reads one episodes file with its families and its text file, through those functions.
"""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from accessdesk.text import TextBundle

ROOT = Path(__file__).resolve().parents[2]
# The kinds of difference of scripts/replay_check.py that are about the verdict of an attempt.
#   replay, final_state   the saved calls do not give the saved tool results or the saved final state
#   passed, reward        the main or the second version of verify or reward grades the state otherwise
#   record, family, text, data, ruleset   the record could not be checked at all against these files
# Every other kind (labels, expected, settings, turns, counters) is a difference of the record that leaves
# the verdict as it is. The audit reports those apart.
VERDICT_KINDS = ("replay", "final_state", "passed", "reward", "record", "family", "text", "data", "ruleset")


@functools.lru_cache(maxsize=None)
def load_script(name: str) -> ModuleType:
    """The module of scripts/<name>.py, loaded once ("gpu/rollout" is scripts/gpu/rollout.py).

    The scripts are files, not a package, so the module is loaded by its path. Loading runs no `main`."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("accessdesk_script_" + name.replace("/", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} cannot be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Run(NamedTuple):
    """One episodes file with the files that it was made from."""
    episodes: Path
    records: list[dict]             # the records of the split, in file order
    families: dict[str, dict]       # family id -> family record of the split file
    text: TextBundle
    data_sha256: str                # the SHA-256 of the split file
    episodes_sha256: str            # the SHA-256 of the episodes file


def load_run(episodes: Path, scenarios: Path, split: str, text_path: Path) -> Run:
    """The records of one episodes file, its families and its text file.

    The checks are those of scripts/monitor_report.py and scripts/replay_check.py: the file holds one run
    and one set-up; the text file is the one that the records name; the scenario folder was built from
    that text file and its split file is the file that the build wrote, with the hash that the records
    name. Raises ValueError with the reason if one of them does not hold."""
    monitor_report, replay_check = load_script("monitor_report"), load_script("replay_check")
    records, _ = monitor_report.records_of_split(episodes, split)
    text = monitor_report.text_of_run(text_path, records)
    problem = replay_check.folder_problem(scenarios, split, text)
    if problem:
        raise ValueError(problem)
    families, _ = monitor_report.families_of_run(scenarios, split, records)
    return Run(episodes=episodes, records=records,
               families={family["family_id"]: family for family in families}, text=text,
               data_sha256=replay_check.sha256_of_file(scenarios / f"{split}.jsonl"),
               episodes_sha256=replay_check.sha256_of_file(episodes))


def record_differences(record: dict, run: Run) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """What scripts/replay_check.py (`check_record`) finds in one record, in two lists of (kind, detail).

    The first list holds the differences about the verdict (VERDICT_KINDS). If it is empty, the saved
    calls give the saved final state, and the main and the second version of verify and of reward give
    the saved pass and the saved reward. The second list holds every other difference."""
    found = load_script("replay_check").check_record(record, run.families, run.text, run.data_sha256)
    about_verdict = [(kind, detail) for kind, detail in found if kind in VERDICT_KINDS]
    return about_verdict, [(kind, detail) for kind, detail in found if kind not in VERDICT_KINDS]
