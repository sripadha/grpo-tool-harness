"""Shared fixtures, and the helper that gives a test the main or the second version of a core function.

Fixtures (all session-scoped; treat what they return as read-only, never change it in a test):
    text         the placeholder text bundle (text/placeholder_text.yaml); see "Another text file" below
    cfg          RunConfig with its defaults; with the placeholder text its data seed and calibration seed
                 are the two placeholder seeds of the config (RunConfig.with_seeds_for), so the worlds of
                 the tests are never the worlds of a build from the real text
    pools        the name pools per split (data/names/, or made-up tokens with a warning if a file is missing)
    families     {"train": [...], "dev": [...], "test": [...], "calib": [...]}: the family records of every
                 split, built in memory at the full default sizes from the placeholder text
    marker_text  a text bundle in which every model-visible string is a unique marker of the form Zq000zQ;
                 use it to prove that a piece of code takes all visible text from the bundle
    tokenizer    the Qwen3.5-4B tokenizer from .cache/hf (the test is skipped if the folder is missing)

Another text file. The environment variable ACCESSDESK_TEST_TEXT makes the `text` fixture (and so `families`)
load that file in place of the placeholder. It exists to run the validation suite and the cross-module tests
on the final text (text/model_text.yaml) before the freeze:

    ACCESSDESK_TEST_TEXT=text/model_text.yaml .venv/bin/python -m pytest \
        tests/core/test_validation_suite.py tests/core/test_integration.py

Only these two files are written to hold for any valid text file. Other test files type the placeholder's
words by hand and fail with another text; a plain `pytest` never sets the variable.

The two versions of the six core functions. Each function of accessdesk.impl.NAMES has a main version and an
independent second version (accessdesk/_second). The hand-made tests of a function run on both: the test
file has a fixture with the function's name and the parameters VERSIONS, which returns
`function_under_test(name, version)`. While a second version is not written, its tests are skipped.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Callable, Iterator

import pytest
import yaml

from accessdesk import impl
from accessdesk.config import RunConfig
from accessdesk.text import TextBundle, load_text, placeholder_path

ROOT = Path(__file__).resolve().parent.parent
TEXT_FILE_VARIABLE = "ACCESSDESK_TEST_TEXT"
VERSIONS = ("main", "second")       # the two versions of each core function, as parameters of a fixture


# ---------------------------------------------------------------------------
# The two versions of a core function
# ---------------------------------------------------------------------------
def skip_unless_second_is_written(name: str) -> None:
    """Skip the running test, with the reason, while the second version of `name` is a stub."""
    if not impl.second_is_written(name):
        pytest.skip(f"the second version of {name}() is not written yet (accessdesk/_second)")


def function_under_test(name: str, version: str) -> Callable:
    """The main or the second version of a core function, whatever ACCESSDESK_IMPL says.

    For "second" the running test is skipped while that version is not written."""
    if version == "main":
        return impl.main_version(name)
    if version == "second":
        skip_unless_second_is_written(name)
        return impl.second_version(name)
    raise ValueError(f"version must be one of {VERSIONS}, not {version!r}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def text_file() -> Path:
    """The file that the `text` fixture loads: the placeholder, or the file named in ACCESSDESK_TEST_TEXT
    (a path; a relative one is taken from the repository root)."""
    named = os.environ.get(TEXT_FILE_VARIABLE, "")
    return ROOT / named if named else placeholder_path()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Say it at the end of the run when the tests did not use the placeholder text."""
    if text_file() != placeholder_path():
        terminalreporter.write_line(
            f"{TEXT_FILE_VARIABLE} is set: the `text` and `families` fixtures used {text_file()}")


@pytest.fixture(scope="session")
def text() -> TextBundle:
    return load_text(text_file())


@pytest.fixture(scope="session")
def cfg(text: TextBundle) -> RunConfig:
    return RunConfig().with_seeds_for(text.is_placeholder)


@pytest.fixture(scope="session")
def pools() -> dict:
    from accessdesk.env.generator import load_name_pools
    return load_name_pools(ROOT / "data" / "names")


@pytest.fixture(scope="session")
def families(text: TextBundle, cfg: RunConfig, pools: dict) -> Iterator[dict[str, list[dict]]]:
    from accessdesk.env.generator import make_all_splits
    built = make_all_splits(text, pools, cfg)
    snapshot = json.dumps(built, ensure_ascii=False)
    yield built
    # The families are shared by every test of the session. A test that changes them would make other tests
    # depend on the test order, so say it loudly.
    assert json.dumps(built, ensure_ascii=False) == snapshot, "a test changed the shared `families` fixture"


MARKER_PATTERN = r"Zq\d{3}zQ"
_NOT_VISIBLE = (("meta", "written_by"), ("policy_shift", "variant"))


def _with_markers(node: object, path: tuple[str, ...], counter: itertools.count) -> object:
    """A copy of the parsed text file in which every model-visible string is replaced by a unique marker."""
    if isinstance(node, dict):
        return {key: _with_markers(value, path + (key,), counter) for key, value in node.items()}
    if isinstance(node, list):
        return [_with_markers(value, path, counter) for value in node]
    if not isinstance(node, str) or path in _NOT_VISIBLE:
        return node
    marker = f"Zq{next(counter):03d}zQ"
    if path[0] == "request_templates":
        return marker + " {person_name} {department} {role_name}"
    if path[0] == "errors":
        return marker + " {value}"
    return marker


@pytest.fixture(scope="session")
def marker_text(tmp_path_factory: pytest.TempPathFactory) -> TextBundle:
    data = yaml.safe_load(placeholder_path().read_text(encoding="utf-8"))
    marked = _with_markers(data, (), itertools.count())
    path = tmp_path_factory.mktemp("marker_text") / "marker_text.yaml"
    path.write_text(yaml.safe_dump(marked, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_text(path)


@pytest.fixture(scope="session")
def tokenizer():
    folder = ROOT / ".cache" / "hf" / "Qwen3.5-4B"
    if not folder.is_dir():
        pytest.skip(f"tokenizer files are missing: {folder}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(folder))
