"""Which version of the six core functions is in use.

Each of the six functions (NAMES below) exists in two versions:
    main     in the module where the function belongs (for example `accessdesk/env/verifier.py`).
             Every run uses it.
    second   in `accessdesk/_second/`: an independent version, written by another model from the contract in
             the docstring only. Tests compare it with the main version; nothing else uses it.

All other code reaches these functions only through `get(name)`, and every saved episode records which
version was used.

The environment variable ACCESSDESK_IMPL chooses the mode:
    main     (the default) `get(name)` returns the main version
    second   `get(name)` returns the second version, whether it is written or not: a second version that is
             not written yet raises NotImplementedError when it is called
Any other value is a ValueError. The variable is read again on every call.
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Callable

NAMES = ("verify", "reward", "build_loss_mask", "group_advantages", "policy_loss", "paired_family_bootstrap")
MODES = ("main", "second")

# name -> (module of the main version, module of the second version)
_MODULES: dict[str, tuple[str, str]] = {
    "verify": ("accessdesk.env.verifier", "accessdesk._second.verifier"),
    "reward": ("accessdesk.env.reward", "accessdesk._second.reward"),
    "build_loss_mask": ("accessdesk.train.loss_mask", "accessdesk._second.loss_mask"),
    "group_advantages": ("accessdesk.train.grpo_math", "accessdesk._second.grpo_math"),
    "policy_loss": ("accessdesk.train.grpo_math", "accessdesk._second.grpo_math"),
    "paired_family_bootstrap": ("accessdesk.eval.bootstrap", "accessdesk._second.bootstrap"),
}


def _probe_args(name: str) -> tuple[Any, ...]:
    """A tiny valid input for each function, used only to see if its second version is written."""
    if name == "verify":
        state = {"access": [], "approvals": [], "grants": [],
                 "request": {"resolution": None, "person_id": None, "role_id": None}}
        return (state, state)
    if name == "reward":
        from accessdesk.config import DEFAULT
        return (True, 0, 3, 4, DEFAULT)
    if name == "build_loss_mask":
        from accessdesk.types import Segment
        return (2, [Segment("prompt", 0, 1, -1), Segment("sampled", 1, 2, 0)])
    import numpy as np
    if name == "group_advantages":
        return (np.array([[1.0, 0.0]]),)
    if name == "policy_loss":
        return (np.zeros((1, 2)), np.ones((1, 2)), np.zeros(1), 1.0)
    if name == "paired_family_bootstrap":
        return (np.array([0.0, 1.0]), np.array([1.0, 1.0]), 10, 0, 0.95)
    raise KeyError(name)


def mode() -> str:
    """The mode: "main" or "second", from the environment variable ACCESSDESK_IMPL (default: "main")."""
    value = os.environ.get("ACCESSDESK_IMPL", "main")
    if value not in MODES:
        raise ValueError(f"ACCESSDESK_IMPL must be one of {MODES}, not {value!r}")
    return value


def _function(name: str, version: str) -> Callable:
    """The main or the second version of `name`. The module is imported here, on first use."""
    if name not in _MODULES:
        raise KeyError(f"unknown function {name!r}; known: {NAMES}")
    module = _MODULES[name][MODES.index(version)]
    return getattr(importlib.import_module(module), name)


def main_version(name: str) -> Callable:
    """The main version of `name`, whatever the mode is."""
    return _function(name, "main")


def second_version(name: str) -> Callable:
    """The second version of `name`, whatever the mode is. It may be a stub that is not written yet."""
    return _function(name, "second")


def second_is_written(name: str) -> bool:
    """False if the second version of `name` raises NotImplementedError on the probe input, True otherwise.

    Any other error of the call counts as "written": a bug in a second version must show up in the tests
    that compare the two versions, and must not look like a version that is still missing.
    A second version whose module cannot be imported, or that has no function of this name, is an error
    here (ImportError, AttributeError): it is not reported as written and not as missing.
    """
    function = second_version(name)
    try:
        function(*_probe_args(name))
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


def which(name: str) -> str:
    """The version that get(name) returns now: "main" or "second"."""
    if name not in _MODULES:
        raise KeyError(f"unknown function {name!r}; known: {NAMES}")
    return mode()


def get(name: str) -> Callable:
    """The function to call for `name`: the main version, or the second one in mode "second"."""
    return _function(name, which(name))


def status() -> dict[str, str]:
    """For each of the six functions: "main" or "second"."""
    return {name: which(name) for name in NAMES}
