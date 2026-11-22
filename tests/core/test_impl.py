"""accessdesk/impl.py: which version of the six core functions is in use; the stubs of accessdesk/_second."""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from accessdesk import impl
from accessdesk.config import DEFAULT

ROOT = Path(__file__).resolve().parents[2]
STATE = {"access": [], "approvals": [], "grants": [],
         "request": {"resolution": None, "person_id": None, "role_id": None}}
# Typed by hand: where the two versions of each function live.
MODULES = {
    "verify": ("accessdesk.env.verifier", "accessdesk._second.verifier"),
    "reward": ("accessdesk.env.reward", "accessdesk._second.reward"),
    "build_loss_mask": ("accessdesk.train.loss_mask", "accessdesk._second.loss_mask"),
    "group_advantages": ("accessdesk.train.grpo_math", "accessdesk._second.grpo_math"),
    "policy_loss": ("accessdesk.train.grpo_math", "accessdesk._second.grpo_math"),
    "paired_family_bootstrap": ("accessdesk.eval.bootstrap", "accessdesk._second.bootstrap"),
}
NAMES = tuple(MODULES)
STUB_MESSAGE = "the second version is not written yet"


def main_function(name: str):
    return getattr(importlib.import_module(MODULES[name][0]), name)


def second_function(name: str):
    return getattr(importlib.import_module(MODULES[name][1]), name)


def test_names_and_modes():
    assert impl.NAMES == ("verify", "reward", "build_loss_mask", "group_advantages", "policy_loss",
                          "paired_family_bootstrap")
    assert impl.NAMES == NAMES
    assert impl.MODES == ("main", "second")


# ---------------------------------------------------------------------------
# The mode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_the_default_mode_gives_the_main_version(monkeypatch, name):
    monkeypatch.delenv("ACCESSDESK_IMPL", raising=False)
    assert impl.mode() == "main"
    assert impl.which(name) == "main"
    assert impl.get(name) is main_function(name)
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    assert impl.which(name) == "main"
    assert impl.get(name) is main_function(name)


@pytest.mark.parametrize("name", NAMES)
def test_mode_second_gives_the_second_version_written_or_not(monkeypatch, name):
    monkeypatch.setenv("ACCESSDESK_IMPL", "second")
    assert impl.mode() == "second"
    assert impl.which(name) == "second"
    assert impl.get(name) is second_function(name)
    assert impl.get(name) is not main_function(name)


@pytest.mark.parametrize("name", NAMES)
def test_main_version_and_second_version_do_not_read_the_mode(monkeypatch, name):
    for value in ("main", "second"):
        monkeypatch.setenv("ACCESSDESK_IMPL", value)
        assert impl.main_version(name) is main_function(name)
        assert impl.second_version(name) is second_function(name)


def test_the_mode_is_read_again_on_every_call(monkeypatch):
    monkeypatch.setenv("ACCESSDESK_IMPL", "second")
    assert impl.which("verify") == "second"
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    assert impl.which("verify") == "main"
    monkeypatch.delenv("ACCESSDESK_IMPL")
    assert impl.which("verify") == "main"


@pytest.mark.parametrize("value", ["auto", "user", "reference", "Main", "", "mine"])
def test_any_other_mode_is_refused(monkeypatch, value):
    monkeypatch.setenv("ACCESSDESK_IMPL", value)
    for call in (impl.mode, impl.status, lambda: impl.which("verify"), lambda: impl.get("verify")):
        with pytest.raises(ValueError, match="ACCESSDESK_IMPL"):
            call()


def test_an_unknown_name_is_refused(monkeypatch):
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    for call in (impl.which, impl.get, impl.main_version, impl.second_version, impl.second_is_written):
        with pytest.raises(KeyError):
            call("verifier")


def test_status_covers_all_six_functions(monkeypatch):
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    assert impl.status() == {name: "main" for name in NAMES}
    monkeypatch.setenv("ACCESSDESK_IMPL", "second")
    assert impl.status() == {name: "second" for name in NAMES}
    assert list(impl.status()) == list(NAMES)


# ---------------------------------------------------------------------------
# The probe inputs
# ---------------------------------------------------------------------------
def test_the_probe_inputs_are_the_ones_of_the_specification():
    assert impl._probe_args("verify") == (STATE, STATE)
    assert impl._probe_args("reward") == (True, 0, 3, 4, DEFAULT)
    n_tokens, segments = impl._probe_args("build_loss_mask")
    assert (n_tokens, [tuple(vars(s).values()) for s in segments]) == \
        (2, [("prompt", 0, 1, -1), ("sampled", 1, 2, 0)])
    assert [np.asarray(part).tolist() for part in impl._probe_args("group_advantages")] == [[[1.0, 0.0]]]
    assert [np.asarray(part).tolist() for part in impl._probe_args("policy_loss")] == \
        [[[0.0, 0.0]], [[1.0, 1.0]], [0.0], 1.0]
    assert [np.asarray(part).tolist() for part in impl._probe_args("paired_family_bootstrap")] == \
        [[0.0, 1.0], [1.0, 1.0], 10, 0, 0.95]


def test_the_probe_inputs_are_valid_inputs_of_the_main_versions():
    assert impl.main_version("verify")(*impl._probe_args("verify")) is True
    assert impl.main_version("reward")(*impl._probe_args("reward")) == 100
    assert impl.main_version("build_loss_mask")(*impl._probe_args("build_loss_mask")) == [0, 1]
    advantages, keep = impl.main_version("group_advantages")(*impl._probe_args("group_advantages"))
    assert (advantages.tolist(), keep.tolist()) == ([[0.5, -0.5]], [True])
    assert float(impl.main_version("policy_loss")(*impl._probe_args("policy_loss"))) == 0.0
    result = impl.main_version("paired_family_bootstrap")(*impl._probe_args("paired_family_bootstrap"))
    assert result["diff"] == 0.5 and 0.0 <= result["lo"] <= result["hi"] <= 1.0


# ---------------------------------------------------------------------------
# second_is_written
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_a_second_version_that_raises_not_implemented_is_not_written(monkeypatch, name):
    def stub(*args, **kwargs):
        raise NotImplementedError(STUB_MESSAGE)
    monkeypatch.setattr(importlib.import_module(MODULES[name][1]), name, stub)
    for value in ("main", "second"):                       # the answer does not depend on the mode
        monkeypatch.setenv("ACCESSDESK_IMPL", value)
        assert impl.second_is_written(name) is False


@pytest.mark.parametrize("name", NAMES)
def test_a_second_version_that_returns_is_written(monkeypatch, name):
    seen = []
    monkeypatch.setattr(importlib.import_module(MODULES[name][1]), name,
                        lambda *args, **kwargs: seen.append(args) or "an answer")
    assert impl.second_is_written(name) is True
    assert len(seen) == 1 and len(seen[0]) == len(impl._probe_args(name))        # called on the probe input


@pytest.mark.parametrize("name", NAMES)
def test_a_second_version_with_a_bug_counts_as_written(monkeypatch, name):
    def buggy(*args, **kwargs):
        raise TypeError("a bug in the second version")
    monkeypatch.setattr(importlib.import_module(MODULES[name][1]), name, buggy)
    assert impl.second_is_written(name) is True            # so the comparison runs and shows the bug


def test_a_second_module_that_cannot_be_imported_is_an_error(monkeypatch):
    monkeypatch.setitem(impl._MODULES, "verify",
                        ("accessdesk.env.verifier", "accessdesk._second.no_such_module"))
    with pytest.raises(ImportError):
        impl.second_is_written("verify")
    monkeypatch.setenv("ACCESSDESK_IMPL", "second")
    with pytest.raises(ImportError):
        impl.get("verify")
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    assert impl.get("verify") is main_function("verify")          # the main version does not need it


def test_a_second_module_without_the_function_is_an_error(monkeypatch):
    monkeypatch.delattr(importlib.import_module(MODULES["reward"][1]), "reward")
    with pytest.raises(AttributeError):
        impl.second_is_written("reward")


def test_the_main_module_is_not_imported_to_answer_for_the_second(monkeypatch):
    monkeypatch.setitem(impl._MODULES, "verify", ("accessdesk.no_such_module", "accessdesk._second.verifier"))
    monkeypatch.setenv("ACCESSDESK_IMPL", "second")
    assert impl.get("verify") is second_function("verify")
    assert impl.second_is_written("verify") in (True, False)


# ---------------------------------------------------------------------------
# The two versions side by side: one contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_both_versions_have_the_same_signature_and_the_same_contract(name):
    main, second = main_function(name), second_function(name)
    assert inspect.signature(main) == inspect.signature(second)
    assert main.__doc__ == second.__doc__
    assert len(main.__doc__) > 400                          # the contract is in the docstring
    assert main is not second and main.__module__ != second.__module__


def function_body(module_name: str, name: str) -> list[ast.stmt]:
    """The statements of a function after its docstring, parsed from the module's source file."""
    source = Path(importlib.import_module(module_name).__file__).read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    assert ast.get_docstring(function) is not None
    return function.body[1:]


@pytest.mark.parametrize("name", NAMES)
def test_the_main_version_is_written(name):
    body = function_body(MODULES[name][0], name)
    assert len(body) >= 3
    assert not any(isinstance(node, ast.Raise) and "NotImplementedError" in ast.unparse(node)
                   for node in ast.walk(ast.Module(body=body, type_ignores=[])))


@pytest.mark.parametrize("name", NAMES)
def test_a_second_version_that_is_not_written_is_a_bare_stub(name):
    """While a second version is not written its file must hold no implementation at all: the model that
    writes it works from the contract only."""
    if impl.second_is_written(name):
        pytest.skip(f"the second version of {name}() is written")
    body = function_body(MODULES[name][1], name)
    assert [ast.unparse(node) for node in body] == [f"raise NotImplementedError('{STUB_MESSAGE}')"]
    with pytest.raises(NotImplementedError, match=STUB_MESSAGE):
        second_function(name)(*impl._probe_args(name))


def test_the_second_package_says_what_it_is():
    import accessdesk._second as package
    doc = " ".join(package.__doc__.split())
    assert "ndependent second versions" in doc and "another model" in doc and "contract only" in doc
    assert "tests" in doc
    files = sorted(path.name for path in Path(package.__file__).parent.glob("*.py"))
    assert files == ["__init__.py", "bootstrap.py", "grpo_math.py", "loss_mask.py", "reward.py",
                     "verifier.py"]


def test_the_old_modes_and_words_are_gone():
    source = (ROOT / "accessdesk" / "impl.py").read_text(encoding="utf-8").lower()
    for word in ("candidate", "user", "reference", "auto"):
        assert word not in source, f"impl.py still says {word!r}"
    assert not (ROOT / "accessdesk" / "_reference").exists()
    assert not (ROOT / "tests" / "user_functions").exists()
