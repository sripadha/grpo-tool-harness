"""Fixtures for the tests that need torch and a GPU, and the rule that leaves this folder out everywhere else.

When torch is not installed or no CUDA device is there, the test files of this folder are not collected (they
are not even imported), so a plain `pytest` on a laptop stays green. On the rented machine:

    python -m pytest tests/gpu -q -s            (-s shows the numbers that the parity tests print)

The model is loaded once per session. Its name comes from the environment variable ACCESSDESK_TEST_MODEL
(default "Qwen/Qwen3.5-2B", the debugging model). Tests must leave the model as they found it: eval mode,
gradient checkpointing off, no parameter changed.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

GPU_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"


CUDA_PROBE = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"


def _skip_reason() -> str | None:
    """Why the tests of this folder cannot run here, or None if they can.

    torch is NOT imported into this process. pytest imports this file when it collects, before any test of
    any folder runs, and tests elsewhere check that a script leaves torch out of `sys.modules`. (On the first
    run on the rented machine, 2026-11-17, an `import torch` here made two tests of tests/core/test_drafting.py
    fail there and nowhere else.) So "does torch see a CUDA device" is asked in a child process; it costs a
    few seconds per pytest run, and only where torch is installed.
    """
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    probe = subprocess.run([sys.executable, "-c", CUDA_PROBE], capture_output=True, text=True)
    if probe.returncode == 3:
        return "no CUDA device"
    if probe.returncode != 0:
        return f"torch cannot be imported: {probe.stderr.strip().splitlines()[-1][:200] if probe.stderr.strip() else probe.returncode}"
    return None


SKIP_REASON = _skip_reason()
# pytest reads this name: files of this folder that match a pattern are not collected.
collect_ignore_glob = ["test_*.py"] if SKIP_REASON is not None else []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Give every test of this folder the marker `gpu`. A test file that was named on the command line is
    collected in spite of the rule above; its tests are then skipped with the reason."""
    for item in items:
        if GPU_DIR in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.gpu)
            if SKIP_REASON is not None:
                item.add_marker(pytest.mark.skip(reason=f"{SKIP_REASON}: run these on the rented machine"))


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from accessdesk.harness.hf_backend import load_model

    return load_model(os.environ.get("ACCESSDESK_TEST_MODEL", DEFAULT_MODEL))


@pytest.fixture(scope="session")
def model(model_and_tokenizer):
    return model_and_tokenizer[0]


@pytest.fixture(scope="session")
def tokenizer(model_and_tokenizer):
    """The tokenizer of the loaded model. It replaces the fixture of tests/conftest.py, which reads the
    tokenizer files from .cache/hf; that folder is not copied to the rented machine."""
    return model_and_tokenizer[1]


@pytest.fixture(scope="session")
def layout(tokenizer, text):
    """The token layout of the five-tool prompt with the placeholder text."""
    from accessdesk.harness.tokens import TokenLayout

    return TokenLayout(tokenizer, text.tool_specs("five"))


@pytest.fixture(scope="session")
def prompts(layout, text, families) -> list[list[int]]:
    """Four real first prompts of different lengths (dev families, two wording styles)."""
    dev = families["dev"]
    chosen = [(dev[0], 0), (dev[1], 1), (dev[2], 0), (dev[3], 3)]            # (family, index of the wording)
    result = [layout.first_prompt(text.system_prompt, family["wordings"][w]["text"]) for family, w in chosen]
    assert len({len(prompt) for prompt in result}) > 1, "the prompts must differ in length"
    return result
