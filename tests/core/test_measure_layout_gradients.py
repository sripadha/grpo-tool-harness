"""scripts/gpu/measure_layout_gradients.py, without torch and without a model.

What is tested here, and what is not:
    the splits into two halves   seeded, no half empty, the same split for the same seed, no split twice
    cosine, relative difference  by hand, on plain lists and on numpy arrays, computed in float64
    the noise from two halves    |a - b| / |a + b| by hand; near 1 for pure noise and 0 for no noise, on
                                 made-up gradients; how the halves of each loss make up the whole
    the sets of weights          where the first, a middle and the last layer lie in the gradient vector
    the table of the result      by hand, from small gradients
    the argument parser          the command of the task; the settings of the step come from the run
    the step                     which episodes each loss is taken on, on hand-made records
    what describes the machine   a read that fails gives its error text; the TF32 switches on a stand-in
                                 for torch; the lines and counts read from the source files of a package
    the three stages             the script's main function with the GPU part replaced by a stand-in that
                                 gives gradients a test can compute again: what is written, the numbers
                                 of the result, the refusals
    saving first                 every way is on disk before the next one starts and before the machine is
                                 described; a start after a failure keeps what was saved and measures the rest
The GPU part itself (the class `ModelBackend`: loading the model, the library's gradient functions, reading
`.grad`) is NOT tested here. It has not run yet; it runs on the rented machine only.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from accessdesk.harness.records import write_records
from tests.helpers.make_records import make_record, scenario_records

ROOT = Path(__file__).resolve().parents[2]
GPU_SCRIPTS = ROOT / "scripts" / "gpu"
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"          # the 2B, typed by hand from ANALYSIS_PLAN.md
# The hand-made token rows: every episode has 5 prompt tokens, and each turn has 3 sampled tokens and, but
# for the last turn, 2 tokens of tool result. A hand-made episode has 4 turns: 12 sampled tokens.
PROMPT_TOKENS, TURN_TOKENS, SUFFIX_TOKENS = 5, 3, 2
SAMPLED_TOKENS = 12
# The LoRA weights of the stand-in model, in the order of the gradient vector: three layers of 6 entries.
WEIGHTS = [("model.layers.0.mlp.up_proj.lora_A.default.weight", 4),
           ("model.layers.0.mlp.up_proj.lora_B.default.weight", 2),
           ("model.layers.1.self_attn.q_proj.lora_A.default.weight", 4),
           ("model.layers.1.self_attn.q_proj.lora_B.default.weight", 2),
           ("model.layers.2.mlp.up_proj.lora_A.default.weight", 4),
           ("model.layers.2.mlp.up_proj.lora_B.default.weight", 2)]
N_WEIGHTS = 18
SETS = {"all LoRA weights": [[0, 18]], "first layer (0)": [[0, 6]], "middle layer (1)": [[6, 12]],
        "last layer (2)": [[12, 18]]}
TRAINER, SINGLE, REFERENCE = "bfloat16_trainer_layout", "bfloat16_one_episode", "float32_one_episode"


@pytest.fixture(scope="module")
def script():
    """The script as a module, imported by name from scripts/gpu as the other scripts there import each
    other."""
    if str(GPU_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GPU_SCRIPTS))
    return importlib.import_module("measure_layout_gradients")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_hand_cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))


def same_split(halves) -> frozenset:
    """A split without the order of its two halves."""
    return frozenset(tuple(half) for half in halves)


# ---------------------------------------------------------------------------
# The splits into two halves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_items", [2, 3, 4, 5, 16, 128])
def test_a_split_has_two_halves_that_are_not_empty_and_hold_every_position_once(script, n_items):
    for index in range(10):
        half_a, half_b = script.split_in_halves(n_items, seed=0, index=index)
        assert len(half_a) == n_items // 2 and len(half_b) == n_items - n_items // 2
        assert half_a and half_b
        assert sorted(half_a + half_b) == list(range(n_items))
        assert half_a == sorted(half_a) and half_b == sorted(half_b)


def test_the_same_seed_and_index_give_the_same_split_and_other_ones_another_split(script):
    first = script.split_in_halves(128, seed=7, index=3)
    assert script.split_in_halves(128, seed=7, index=3) == first
    assert script.split_in_halves(128, seed=8, index=3) != first
    assert script.split_in_halves(128, seed=7, index=4) != first


def test_fewer_than_two_positions_cannot_be_split(script):
    for n_items in (0, 1):
        with pytest.raises(ValueError, match="at least 2"):
            script.split_in_halves(n_items, seed=0, index=0)
        assert script.different_splits(n_items, 20, seed=0) == []
        assert script.n_possible_splits(n_items) == 0


def test_the_number_of_possible_splits_is_counted_without_the_order_of_the_halves(script):
    # 2: 0|1. 4: 01|23, 02|13, 03|12. 5: one of the 10 pairs against the other three. 6: 20 triples, each
    # split counted from both sides, so 10. 16: 12,870 / 2.
    assert [script.n_possible_splits(n) for n in (2, 3, 4, 5, 6, 16)] == [1, 3, 3, 10, 10, 6435]


def test_when_fewer_splits_exist_than_are_asked_for_all_of_them_are_returned(script):
    splits = script.different_splits(4, 20, seed=0)
    assert {same_split(halves) for halves in splits} == {
        frozenset({(0, 1), (2, 3)}), frozenset({(0, 2), (1, 3)}), frozenset({(0, 3), (1, 2)})}
    assert len(splits) == 3
    assert len(script.different_splits(2, 20, seed=0)) == 1
    assert len({same_split(halves) for halves in script.different_splits(5, 20, seed=0)}) == 10


def test_no_split_is_returned_twice_and_the_same_seed_gives_the_same_splits(script):
    splits = script.different_splits(16, 20, seed=0)
    assert len(splits) == 20 and len({same_split(halves) for halves in splits}) == 20
    assert all(len(half_a) == 8 and len(half_b) == 8 for half_a, half_b in splits)
    assert script.different_splits(16, 20, seed=0) == splits
    assert script.different_splits(16, 20, seed=1) != splits
    assert script.different_splits(16, 5, seed=0) == splits[:5]


# ---------------------------------------------------------------------------
# Cosine and relative difference
# ---------------------------------------------------------------------------
def test_the_cosine_of_two_vectors_by_hand(script):
    assert script.cosine([1, 0], [0, 1]) == 0.0                        # at a right angle
    assert script.cosine([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0, abs=1e-15)      # the same direction
    assert script.cosine([1, 0], [-5, 0]) == -1.0                      # the opposite direction
    # (3 * 4 + 4 * 3) / (5 * 5) = 24 / 25
    assert script.cosine([3, 4], [4, 3]) == pytest.approx(0.96, abs=1e-15)
    assert script.cosine(np.array([3, 4], dtype=np.float32), np.array([[4], [3]])) == pytest.approx(0.96)


def test_the_relative_difference_by_hand_and_which_vector_is_the_reference(script):
    assert script.relative_difference([3, 4], [3, 4]) == 0.0
    # |(3, 4) - (0, 4)| = 3. The reference (0, 4) has the length 4, the reference (3, 4) the length 5.
    assert script.relative_difference([3, 4], [0, 4]) == pytest.approx(0.75, abs=1e-15)
    assert script.relative_difference([0, 4], [3, 4]) == pytest.approx(0.6, abs=1e-15)
    assert script.relative_difference(np.array([3.0, 4.0]), np.array([0.0, 4.0], dtype=np.float32)) == 0.75


def test_a_vector_of_zeros_has_no_cosine_and_cannot_be_the_reference(script):
    assert script.cosine([0, 0], [1, 2]) is None and script.cosine([1, 2], [0.0, 0.0]) is None
    assert script.relative_difference([1, 2], [0, 0]) is None
    assert script.relative_difference([0, 0], [3, 4]) == 1.0           # zeros may be the other vector
    assert script.norm([3, 4]) == 5.0 and script.norm([]) == 0.0


def test_vectors_of_different_lengths_and_values_that_are_not_finite_are_errors(script):
    for function in (script.cosine, script.relative_difference):
        with pytest.raises(ValueError, match="entries"):
            function([1, 2, 3], [1, 2])
        with pytest.raises(ValueError, match="finite"):
            function([1, float("nan")], [1, 2])
        with pytest.raises(ValueError, match="finite"):
            function([1, 2], [float("inf"), 2])


def test_float32_gradients_are_compared_in_float64(script):
    """4096 squared is 2**24, and 2**24 + 1 is not a float32 number. So in float32 the two vectors below
    would have the cosine 1.0 exactly. In float64 it is 1 / sqrt(1 + 2**-24), about 3e-08 below 1."""
    a = np.array([4096, 1], dtype=np.float32)
    b = np.array([4096, 0], dtype=np.float32)
    assert float(np.dot(a, a)) == float(np.dot(b, b))                  # what float32 arithmetic gives
    value = script.cosine(a, b)
    assert value < 1.0 and value == pytest.approx(1 / math.sqrt(1 + 2 ** -24), abs=1e-15)
    assert script.relative_difference(a, b) == pytest.approx(1 / 4096, abs=1e-18)


# ---------------------------------------------------------------------------
# The sampling noise of a whole step, from its two halves
# ---------------------------------------------------------------------------
def test_the_relative_noise_of_two_parts_by_hand(script):
    assert script.relative_noise([3, 4], [3, 4]) == 0.0                # equal parts: no noise
    # (1, 0) and (0, 1): the difference (1, -1) and the sum (1, 1) are equally long.
    assert script.relative_noise([1, 0], [0, 1]) == pytest.approx(1.0, abs=1e-15)
    # (3, 0) and (1, 0): the difference has the length 2 and the sum the length 4.
    assert script.relative_noise([3, 0], [1, 0]) == 0.5
    assert script.relative_noise([1, 0], [3, 0]) == 0.5                # the order of the parts plays no part
    assert script.relative_noise([1, 2], [-1, -2]) is None             # the sum is all zeros
    assert script.relative_noise([0, 0], [0, 0]) is None
    with pytest.raises(ValueError, match="entries"):
        script.relative_noise([1, 2, 3], [1, 2])
    # In float64. The sum (8192, 1) has the length sqrt(8192 squared + 1); float32 would give 8192.
    a = np.array([4096, 1], dtype=np.float32)
    b = np.array([4096, 0], dtype=np.float32)
    assert script.relative_noise(a, b) == pytest.approx(1 / math.sqrt(8192 ** 2 + 1), abs=1e-18)
    assert script.relative_noise(a, b) < 1 / 8192


def test_the_noise_of_a_mean_is_right_for_halves_of_unequal_size(script):
    """Token gradients are a true gradient plus independent noise. The yardstick must give the noise of the
    mean over all tokens, also when one half has twice the tokens of the other. The simpler form (each half
    times its share, then `relative_noise`) counts a part of the true gradient as noise there."""
    rng = np.random.default_rng(0)
    n_weights = 20_000
    true_gradient = rng.normal(size=n_weights)
    for n_a, n_b in ((15, 15), (12, 18), (10, 20)):
        tokens_a = true_gradient + rng.normal(scale=1.5, size=(n_a, n_weights))
        tokens_b = true_gradient + rng.normal(scale=1.5, size=(n_b, n_weights))
        mean_a, mean_b = tokens_a.mean(axis=0), tokens_b.mean(axis=0)
        whole = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
        true_noise = np.linalg.norm(whole - true_gradient) / np.linalg.norm(whole)
        assert script.relative_noise_of_a_mean(mean_a, mean_b, n_a, n_b) == pytest.approx(true_noise, rel=0.05)
        share_a, share_b = script.shares_of_the_whole("sft_style", [n_a, n_b])
        simple = script.relative_noise(share_a * mean_a, share_b * mean_b)
        if n_a == n_b:
            assert simple == pytest.approx(script.relative_noise_of_a_mean(mean_a, mean_b, n_a, n_b), rel=1e-9)
        else:
            assert simple > 1.15 * true_noise          # the simpler form is too large for unequal halves
    assert script.relative_noise_of_a_mean([1.0, 2.0], [1.0, 2.0], 3, 5) == 0.0     # equal means: no noise
    assert script.relative_noise_of_a_mean([1.0, 2.0], [3.0, 4.0], 0, 5) is None    # a half with no unit
    assert script.relative_noise_of_a_mean([1.0, 0.0], [-1.0, 0.0], 4, 4) is None   # the whole is all zeros
    with pytest.raises(ValueError):
        script.relative_noise_of_a_mean([1.0, 2.0, 3.0], [1.0, 2.0], 2, 2)


def test_pure_noise_gives_a_relative_noise_near_1_and_a_cosine_near_0_and_no_noise_gives_0_and_1(script):
    """What the result says about its two numbers, tried on made-up gradients of 20,000 weights. The two
    halves are signal + noise_a and signal + noise_b. So the whole is 2 signal + noise_a + noise_b, and
    the noise in the whole is noise_a + noise_b."""
    signal, noise_a, noise_b = np.random.default_rng(0).normal(size=(3, 20000))
    assert script.relative_noise(noise_a, noise_b) == pytest.approx(1.0, abs=0.03)      # only noise
    assert script.cosine(noise_a, noise_b) == pytest.approx(0.0, abs=0.03)
    assert script.relative_noise(signal, signal) == 0.0                                  # no noise
    assert script.cosine(signal, signal) == pytest.approx(1.0, abs=1e-12)
    # Signal and noise of four sizes: the number is the true relative noise of the whole, within 3% of it.
    # The cosine of the same halves is another kind of number: with the smallest noise here, 0.07 of the
    # whole, it is above 0.98.
    for size in (3.0, 1.0, 0.3, 0.1):
        half_a, half_b = signal + size * noise_a, signal + size * noise_b
        true_noise = float(np.linalg.norm(size * (noise_a + noise_b)) / np.linalg.norm(half_a + half_b))
        assert script.relative_noise(half_a, half_b) == pytest.approx(true_noise, rel=0.03)
    assert true_noise == pytest.approx(0.07, abs=0.01) and script.cosine(half_a, half_b) > 0.98


def test_the_halves_are_parts_of_the_whole_as_they_are_for_grpo_and_by_their_tokens_for_the_sft_style_loss(
        script):
    # GRPO: both halves are divided by the constant of the whole step, so the whole is their sum.
    assert script.shares_of_the_whole("grpo", [10, 30]) == [1.0, 1.0]
    # SFT style: each half is a mean over its own tokens. The mean over all 40 tokens is
    # (10 * mean_a + 30 * mean_b) / 40.
    assert script.shares_of_the_whole("sft_style", [10, 30]) == [0.25, 0.75]
    mean_a, mean_b = np.array([4.0, 0.0]), np.array([0.0, 8.0])
    share_a, share_b = script.shares_of_the_whole("sft_style", [10, 30])
    assert (share_a * mean_a + share_b * mean_b).tolist() == [1.0, 6.0]


# ---------------------------------------------------------------------------
# What only describes the machine: nothing here may stop a stage
# ---------------------------------------------------------------------------
def test_a_read_that_fails_gives_its_error_text_and_not_an_exception(script):
    assert script.read_or_error(lambda: {"a": 1}) == {"a": 1}
    assert script.read_or_error(lambda: None) is None

    def fails():
        raise RuntimeError("no such switch " + "x" * 400)

    result = script.read_or_error(fails)
    assert set(result) == {"error"} and result["error"].startswith("RuntimeError: no such switch xxx")
    assert len(result["error"]) == 300


class BrokenSwitch:
    """A switch object of torch that can be neither read nor set."""

    @property
    def allow_tf32(self):
        raise RuntimeError("a mix of the legacy and new APIs")

    @allow_tf32.setter
    def allow_tf32(self, value):
        raise RuntimeError("a mix of the legacy and new APIs")


def stand_in_torch() -> SimpleNamespace:
    """In place of the torch module: only the switches that the script reads and sets, all on TF32."""
    cudnn = SimpleNamespace(allow_tf32=True, fp32_precision="tf32",
                            conv=SimpleNamespace(fp32_precision="tf32"),
                            rnn=SimpleNamespace(fp32_precision="tf32"))
    cuda = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True, fp32_precision="tf32"))
    return SimpleNamespace(backends=SimpleNamespace(cuda=cuda, cudnn=cudnn, fp32_precision="none"),
                           get_float32_matmul_precision=lambda: "highest")


def test_tf32_is_switched_off_and_a_switch_that_cannot_be_set_is_recorded_and_does_not_stop(script):
    torch = stand_in_torch()
    assert script.switch_tf32_off(torch) == {"torch.backends.cuda.matmul.allow_tf32": "set to False",
                                             "torch.backends.cudnn.allow_tf32": "set to False"}
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
    torch = stand_in_torch()
    torch.backends.cuda.matmul = BrokenSwitch()
    done = script.switch_tf32_off(torch)
    assert done["torch.backends.cuda.matmul.allow_tf32"] == {
        "error": "RuntimeError: a mix of the legacy and new APIs"}
    assert done["torch.backends.cudnn.allow_tf32"] == "set to False"   # the other switch is still set
    assert torch.backends.cudnn.allow_tf32 is False


def test_every_tf32_switch_is_read_by_itself_and_one_that_cannot_be_read_gives_its_error(script):
    switches = script.tf32_switches(stand_in_torch())
    assert switches == {"torch.backends.cuda.matmul.allow_tf32": True,
                        "torch.backends.cudnn.allow_tf32": True,
                        "torch.get_float32_matmul_precision()": "highest",
                        "torch.backends.fp32_precision": "none",
                        "torch.backends.cuda.matmul.fp32_precision": "tf32",
                        "torch.backends.cudnn.fp32_precision": "tf32",
                        "torch.backends.cudnn.conv.fp32_precision": "tf32",
                        "torch.backends.cudnn.rnn.fp32_precision": "tf32"}
    # A torch without the newer switches, and with an old one that raises when it is read.
    older = stand_in_torch()
    del older.backends.fp32_precision, older.backends.cudnn.conv, older.backends.cudnn.rnn
    older.backends.cuda.matmul = BrokenSwitch()
    switches = script.tf32_switches(older)
    assert switches["torch.backends.cuda.matmul.allow_tf32"] == {
        "error": "RuntimeError: a mix of the legacy and new APIs"}
    assert switches["torch.backends.cudnn.conv.fp32_precision"]["error"].startswith("AttributeError")
    assert switches["torch.backends.fp32_precision"]["error"].startswith("AttributeError")
    assert switches["torch.backends.cudnn.allow_tf32"] is True         # the others are still read
    assert switches["torch.get_float32_matmul_precision()"] == "highest"


def made_up_packages(folder: Path) -> None:
    """Two made-up packages in `folder`: `fla` with three calls of tl.dot in its ops, one of them with
    allow_tf32 and one with input_precision, and `triton` with one line about TF32 in its language."""
    ops = folder / "fla" / "ops"
    (ops / "gated_delta_rule").mkdir(parents=True)
    (ops / "other").mkdir()
    (ops / "gated_delta_rule" / "chunk.py").write_text(
        "a = tl.dot(x, y)\nb = tl.dot(x, y, allow_tf32=False)\n", encoding="utf-8")
    (ops / "other" / "kernel.py").write_text("c = tl.dot(x, y, input_precision='ieee')\n", encoding="utf-8")
    (ops / "other" / "plain.py").write_text("d = 1\n", encoding="utf-8")
    language = folder / "triton" / "language"
    language.mkdir(parents=True)
    (language / "core.py").write_text(
        "x = 1\n    default_precision = 'TF32' if supports_tf32 else 'ieee'\ny = 2\n", encoding="utf-8")


def test_the_dot_calls_and_the_lines_about_tf32_are_read_from_the_source_files_of_a_package(script, tmp_path):
    made_up_packages(tmp_path)
    ops = tmp_path / "fla" / "ops"
    words = ("tl.dot(", "allow_tf32", "input_precision")
    # The totals are over every file; a file is listed only if it lies in a named sub-folder.
    assert script.count_words(ops, words, ("gated_delta_rule",)) == {
        "folder": str(ops), "n_files": 3, "total": {"tl.dot(": 3, "allow_tf32": 1, "input_precision": 1},
        "files": {"gated_delta_rule/chunk.py": {"tl.dot(": 2, "allow_tf32": 1, "input_precision": 0}}}
    language = tmp_path / "triton" / "language"
    assert script.lines_with(language, ("tf32",)) == {
        "folder": str(language), "n_lines": 1,
        "lines": ["core.py:2: default_precision = 'TF32' if supports_tf32 else 'ieee'"]}
    assert script.lines_with(ops, ("tl.dot(",), limit=2)["n_lines"] == 3
    assert len(script.lines_with(ops, ("tl.dot(",), limit=2)["lines"]) == 2
    for function in (lambda: script.count_words(tmp_path / "nothing", words, ()),
                     lambda: script.lines_with(tmp_path / "nothing", words)):
        with pytest.raises(FileNotFoundError, match="is not a folder"):
            function()


def test_the_report_on_the_kernel_sources_gives_what_it_can_read_and_an_error_text_for_the_rest(
        script, tmp_path, monkeypatch):
    made_up_packages(tmp_path)
    monkeypatch.setattr(script, "package_folder", lambda name: tmp_path / name)
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    report = script.kernel_sources_report()
    assert set(report) == {"triton_version", "TRITON_F32_DEFAULT", "tf32_in_triton_language",
                           "dot_calls_in_fla_ops"}
    assert report["TRITON_F32_DEFAULT"] == "ieee"
    assert report["tf32_in_triton_language"]["n_lines"] == 1
    assert report["dot_calls_in_fla_ops"]["total"] == {"tl.dot(": 3, "allow_tf32": 1, "input_precision": 1}
    assert list(report["dot_calls_in_fla_ops"]["files"]) == ["gated_delta_rule/chunk.py"]
    # The version is a text where triton is installed, and an error text where it is not (this laptop).
    assert isinstance(report["triton_version"], str) or set(report["triton_version"]) == {"error"}
    # Packages that are not installed: an error text for each, and no exception.
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError, match="is not installed"):
        script.package_folder("a_package_that_is_not_installed")

    def not_installed(name):
        raise FileNotFoundError(f"the package {name} is not installed")

    monkeypatch.setattr(script, "package_folder", not_installed)
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    report = script.kernel_sources_report()
    assert report["TRITON_F32_DEFAULT"] is None
    assert report["dot_calls_in_fla_ops"] == {"error": "FileNotFoundError: the package fla is not installed"}
    assert report["tf32_in_triton_language"] == {
        "error": "FileNotFoundError: the package triton is not installed"}


def test_a_json_file_is_replaced_whole_or_not_at_all(script, tmp_path, monkeypatch):
    path = tmp_path / "numbers.json"
    script.write_json_whole(path, {"splits": 1})
    script.write_json_whole(path, {"splits": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"splits": 2}
    assert [found.name for found in tmp_path.iterdir()] == ["numbers.json"]

    def fails(target, obj):
        target.write_text('{"spl', encoding="utf-8")                   # half a file, then the failure
        raise OSError("the disk is full")

    monkeypatch.setattr(script.rollout_script, "write_json", fails)
    with pytest.raises(OSError, match="the disk is full"):
        script.write_json_whole(path, {"splits": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"splits": 2}          # the old file is whole


# ---------------------------------------------------------------------------
# The sets of weights
# ---------------------------------------------------------------------------
def test_the_sets_are_all_weights_and_the_first_a_middle_and_the_last_layer(script):
    assert script.weight_sets(WEIGHTS, 3) == {label: [tuple(r) for r in ranges]
                                              for label, ranges in SETS.items()}
    # 24 layers, as the 2B has: the layers 0, 12 and 23 (the layers of the control tests). Each layer has
    # two weights here, of 3 and 2 entries, so layer k starts at 5 k.
    names = [(f"model.layers.{layer}.mlp.down_proj.lora_{side}.default.weight", size)
             for layer in range(24) for side, size in (("A", 3), ("B", 2))]
    assert script.weight_sets(names, 24) == {
        "all LoRA weights": [(0, 120)], "first layer (0)": [(0, 5)], "middle layer (12)": [(60, 65)],
        "last layer (23)": [(115, 120)]}


def test_a_layer_whose_weights_do_not_lie_side_by_side_gets_two_ranges(script):
    names = [("model.layers.0.mlp.up_proj.lora_A.default.weight", 2),
             ("model.layers.1.mlp.up_proj.lora_A.default.weight", 3),
             ("model.layers.0.mlp.up_proj.lora_B.default.weight", 1),
             ("model.layers.2.mlp.up_proj.lora_B.default.weight", 4)]
    sets = script.weight_sets(names, 3)
    assert sets["first layer (0)"] == [(0, 2), (5, 6)] and sets["all LoRA weights"] == [(0, 10)]
    vector = np.arange(10.0)
    assert script.take(vector, sets["first layer (0)"]).tolist() == [0.0, 1.0, 5.0]
    assert script.take(vector, sets["last layer (2)"]).tolist() == [6.0, 7.0, 8.0, 9.0]
    assert script.n_in(sets["first layer (0)"]) == 3


def test_a_weight_outside_the_decoder_layers_and_a_layer_without_a_weight_are_errors(script):
    with pytest.raises(ValueError, match="not a weight of a decoder layer"):
        script.weight_sets([("lm_head.lora_A.default.weight", 4)], 3)
    with pytest.raises(ValueError, match="no LoRA weight in"):
        script.weight_sets(WEIGHTS[:4], 3)                             # nothing in the last layer


# ---------------------------------------------------------------------------
# The comparison of gradients and the table
# ---------------------------------------------------------------------------
def small_comparison(script) -> list[dict]:
    """Three ways with four weights each, two per layer; by hand below."""
    gradients = {TRAINER: [1, 0, 3, 4], SINGLE: [1, 0, 4, 3], REFERENCE: [0, 1, 3, 4]}
    sets = {"all LoRA weights": [(0, 4)], "first layer (0)": [(0, 2)], "last layer (1)": [(2, 4)]}
    return script.compare_gradients(gradients, sets)


def test_every_pair_of_ways_is_compared_on_every_set_of_weights_by_hand(script):
    rows = small_comparison(script)
    assert [(row["other"], row["reference"]) for row in rows] == [
        (TRAINER, REFERENCE), (SINGLE, REFERENCE), (TRAINER, SINGLE)]
    first = rows[0]["weights"]                    # (1, 0, 3, 4) against the reference (0, 1, 3, 4)
    # All: dot 25, both lengths sqrt(26); the difference (1, -1, 0, 0) has the length sqrt(2).
    assert first["all LoRA weights"] == {
        "n_weights": 4, "cosine": pytest.approx(25 / 26, abs=1e-15),
        "relative_difference": pytest.approx(math.sqrt(2 / 26), abs=1e-15),
        "norm_other": pytest.approx(math.sqrt(26)), "norm_reference": pytest.approx(math.sqrt(26))}
    assert first["first layer (0)"] == {"n_weights": 2, "cosine": 0.0,
                                        "relative_difference": pytest.approx(math.sqrt(2)),
                                        "norm_other": 1.0, "norm_reference": 1.0}
    assert first["last layer (1)"] == {"n_weights": 2, "cosine": pytest.approx(1.0, abs=1e-15),
                                       "relative_difference": 0.0, "norm_other": 5.0, "norm_reference": 5.0}
    third = rows[2]["weights"]                    # (1, 0, 3, 4) against (1, 0, 4, 3)
    assert third["first layer (0)"]["cosine"] == 1.0
    assert third["first layer (0)"]["relative_difference"] == 0.0
    assert third["last layer (1)"]["cosine"] == pytest.approx(0.96, abs=1e-15)          # 24 / 25
    assert third["all LoRA weights"]["cosine"] == pytest.approx(25 / 26, abs=1e-15)     # (1 + 24) / 26


def test_a_gradient_of_zeros_gives_no_cosine_but_does_not_stop_the_comparison(script):
    rows = script.compare_gradients({TRAINER: [0, 0], SINGLE: [0, 0], REFERENCE: [0, 0]},
                                    {"all LoRA weights": [(0, 2)]})
    assert all(row["weights"]["all LoRA weights"]["cosine"] is None
               and row["weights"]["all LoRA weights"]["relative_difference"] is None for row in rows)


def test_the_summary_of_the_halves_leaves_out_what_is_not_defined(script):
    assert script.summary_of([0.1, 0.4, None, 0.3, 0.2]) == {
        "n": 4, "n_not_defined": 1, "mean": pytest.approx(0.25), "median": pytest.approx(0.25), "min": 0.1,
        "max": 0.4}
    assert script.summary_of([0.5, 0.1, 0.3])["median"] == 0.3
    assert script.summary_of([]) == {"n": 0, "n_not_defined": 0, "mean": None, "median": None, "min": None,
                                     "max": None}


def test_the_table_has_one_row_per_loss_and_set_with_the_pairs_beside_the_halves(script):
    comparison = small_comparison(script)
    labels = ("all LoRA weights", "first layer (0)", "last layer (1)")
    halves = {loss: {"cosine_summary": {label: script.summary_of([0.1 * k, 0.3 * k]) for label in labels},
                     "relative_noise_summary": {label: script.summary_of([0.25 * k, 0.5 * k, 1.0 * k])
                                                for label in labels}}
              for k, loss in enumerate(("grpo", "sft_style"), start=1)}
    rows = script.result_table({"grpo": comparison, "sft_style": comparison}, halves)
    assert [(row["loss"], row["weights"]) for row in rows] == [
        (loss, label) for loss in ("grpo", "sft_style") for label in labels]
    row = rows[5]                                 # sft_style, the last layer
    assert row["n_weights"] == 2
    assert row["cosine"] == {f"{TRAINER} against {REFERENCE}": pytest.approx(1.0),
                             f"{SINGLE} against {REFERENCE}": pytest.approx(0.96),
                             f"{TRAINER} against {SINGLE}": pytest.approx(0.96)}
    assert row["relative_difference"][f"{TRAINER} against {REFERENCE}"] == 0.0
    assert row["halves_cosine"]["mean"] == pytest.approx(0.4) and row["halves_cosine"]["n"] == 2
    # The relative noise of the halves, as median and range: 0.5, 1.0 and 2.0 for the second loss.
    assert (row["halves_relative_noise"]["median"], row["halves_relative_noise"]["min"],
            row["halves_relative_noise"]["max"]) == (1.0, 0.5, 2.0)
    text = script.format_table(rows)
    assert "sft_style, last layer (1) (2 weights)" in text and "0.960000" in text
    assert "cosine of the two halves, 2 splits: mean 0.400000" in text
    assert ("relative noise of the whole step, from the two halves, 3 splits: median 1.000000, "
            "smallest 0.500000, largest 2.000000") in text


def test_the_differences_of_the_log_probabilities_by_hand(script):
    logprobs = {TRAINER: [-0.5, -1.0, -2.0], SINGLE: [-0.5, -1.25, -2.0], REFERENCE: [-0.75, -1.0, -1.0]}
    rows = script.logprob_differences(logprobs)
    # Against the reference the differences are 0.25, 0 and 1: their mean is 1.25 / 3.
    assert rows[0] == {"other": TRAINER, "reference": REFERENCE, "n": 3, "mean": pytest.approx(1.25 / 3),
                       "p99": 1.0, "max": 1.0}
    assert (rows[2]["mean"], rows[2]["max"]) == (pytest.approx(0.25 / 3), 0.25)
    empty = script.logprob_differences({TRAINER: [], SINGLE: [], REFERENCE: []})
    assert empty[0] == {"other": TRAINER, "reference": REFERENCE, "n": 0, "mean": None, "p99": None,
                        "max": None}


# ---------------------------------------------------------------------------
# The argument parser and the settings of the step
# ---------------------------------------------------------------------------
def task_command(folder: Path = Path("results/main/checks/pilot"), out: Path = Path("out")) -> list[str]:
    """The command of the task for step 0 of a run folder, without the options that have a default."""
    return ["--model", MODEL, "--adapter", str(folder.parent / "adapter"),
            "--episodes", str(folder / "rollouts" / "step_00000.episodes.jsonl"),
            "--tokens", str(folder / "rollouts" / "step_00000.tokens.jsonl"), "--out", str(out)]


def test_the_parser_takes_the_command_of_the_task_and_leaves_the_settings_of_the_step_to_the_run(script):
    args = script.parse_args(task_command())
    assert (args.model, args.revision, args.device) == (MODEL, None, "cuda")
    assert args.episodes == Path("results/main/checks/pilot/rollouts/step_00000.episodes.jsonl")
    assert args.tokens == Path("results/main/checks/pilot/rollouts/step_00000.tokens.jsonl")
    assert (args.scenarios_per_step, args.group_size, args.reference_tokens, args.micro_batch,
            args.tie_tol) == (None, None, None, None, None)
    assert (args.splits, args.split_seed, args.stage) == (20, 0, "all")
    assert not args.no_gradient_checkpointing and not args.allow_other_settings


def test_every_option_can_be_given(script):
    args = script.parse_args(task_command() + [
        "--revision", "abc", "--scenarios-per-step", "8", "--group-size", "16", "--reference-tokens", "200",
        "--micro-batch", "2", "--tie-tol", "-1", "--no-gradient-checkpointing", "--splits", "5",
        "--split-seed", "3", "--stage", "compare", "--allow-other-settings", "--device", "cuda:1"])
    assert (args.revision, args.scenarios_per_step, args.group_size, args.reference_tokens, args.micro_batch,
            args.tie_tol, args.splits, args.split_seed, args.stage, args.device) == (
        "abc", 8, 16, 200, 2, -1.0, 5, 3, "compare", "cuda:1")
    assert args.no_gradient_checkpointing and args.allow_other_settings


@pytest.mark.parametrize("missing", ["--model", "--adapter", "--episodes", "--tokens", "--out"])
def test_the_five_arguments_of_the_task_are_required(script, missing, capsys):
    command = task_command()
    at = command.index(missing)
    with pytest.raises(SystemExit):
        script.parse_args(command[:at] + command[at + 2:])
    assert missing in capsys.readouterr().err


def test_a_stage_command_runs_that_stage_whatever_stage_the_arguments_name(script):
    argv = task_command() + ["--stage", "all"]
    command = script.stage_command(argv, "float32")
    assert command == [sys.executable, str(GPU_SCRIPTS / "measure_layout_gradients.py"), *argv,
                       "--stage", "float32"]
    assert script.parse_args(command[2:]).stage == "float32"
    assert script.parse_args(command[2:]).episodes == script.parse_args(argv).episodes


def test_without_a_run_manifest_the_settings_are_those_of_the_plan(script):
    gcfg, sources = script.step_settings(script.parse_args(task_command()), None)
    # ANALYSIS_PLAN.md: 16 scenarios x 8 attempts, the loss constant with 200 tokens; micro-batch 4 and no
    # tie tolerance are the trainer's defaults. 16 * 8 * 200 = 25,600.
    assert (gcfg.n_scenarios, gcfg.group_size, gcfg.reference_tokens, gcfg.micro_batch, gcfg.tie_tol,
            gcfg.gradient_checkpointing) == (16, 8, 200, 4, 0.0, True)
    assert gcfg.norm_const == 25600.0
    assert set(sources.values()) == {"the default"}


def test_the_settings_of_the_run_are_used_and_an_option_that_differs_from_them_is_refused(script):
    of_run = {"n_scenarios": 8, "group_size": 16, "reference_tokens": 200, "micro_batch": 4, "tie_tol": 0.0,
              "gradient_checkpointing": True, "lr": 1e-5, "steps": 15}
    gcfg, sources = script.step_settings(script.parse_args(task_command()), of_run)
    assert (gcfg.n_scenarios, gcfg.group_size, gcfg.norm_const) == (8, 16, 25600.0)
    assert set(sources.values()) == {"run_manifest.json of the run"}
    same = script.parse_args(task_command() + ["--scenarios-per-step", "8", "--group-size", "16"])
    assert script.step_settings(same, of_run)[1]["group_size"] == "the command line"
    other = task_command() + ["--scenarios-per-step", "16", "--group-size", "8"]
    with pytest.raises(script.Refused, match="--group-size.*8, 16"):
        script.step_settings(script.parse_args(other), of_run)
    gcfg, _ = script.step_settings(script.parse_args(other + ["--allow-other-settings"]), of_run)
    assert (gcfg.n_scenarios, gcfg.group_size) == (16, 8)
    off = script.parse_args(task_command() + ["--no-gradient-checkpointing"])
    assert script.step_settings(off, of_run)[0].gradient_checkpointing is False
    with pytest.raises(script.Refused, match="at least 1"):
        script.step_settings(script.parse_args(task_command() + ["--micro-batch", "0"]), None)


# ---------------------------------------------------------------------------
# A hand-made step: 4 scenarios, 2 attempts each
# ---------------------------------------------------------------------------
def token_row(record: dict) -> dict:
    """A hand-made line of the token file for an episode record (the layout is in the comment at the top).
    The line of a harness error holds no token data."""
    row = {"scenario_id": record["scenario_id"], "sample": record["sample"], "seed": record["seed"],
           "harness_error": record["harness_error"]}
    if record["harness_error"] is not None:
        return {**row, "ids": [], "segments": [], "logprobs": []}
    ids = list(range(1, PROMPT_TOKENS + 1))
    segments = [["prompt", 0, PROMPT_TOKENS, -1]]
    for turn in range(record["n_turns"]):
        segments.append(["sampled", len(ids), len(ids) + TURN_TOKENS, turn])
        ids += [100 + turn] * TURN_TOKENS
        if turn + 1 < record["n_turns"]:
            segments.append(["suffix", len(ids), len(ids) + SUFFIX_TOKENS, turn])
            ids += [200 + turn] * SUFFIX_TOKENS
    return {**row, "ids": ids, "segments": segments, "logprobs": [-0.5] * (TURN_TOKENS * record["n_turns"])}


def step_records() -> list[dict]:
    """The episodes of a step of 4 scenarios x 2 attempts, in the trainer's order (rewards in brackets):

        group 0   tr-001/s1   [100, 0]             kept: advantages +0.5 and -0.5
        group 1   tr-002/s1   [100, 100]           dropped: equal rewards
        group 2   tr-003/s1   [0, 100]             kept: advantages -0.5 and +0.5
        group 3   tr-004/s1   [100, harness error] dropped: one valid attempt has nothing to differ from
    So the GRPO loss is taken on the episodes 0, 1, 4, 5 and the SFT-style loss on the seven valid ones.
    """
    records = (scenario_records("tr-001", 1, [100, 0]) + scenario_records("tr-002", 1, [100, 100])
               + scenario_records("tr-003", 1, [0, 100]) + scenario_records("tr-004", 1, [100]))
    records.append(make_record("tr-004", 1, 1, harness_error="the sample call failed"))
    return [{**record, "arm": "grpo_train"} for record in records]


def with_tokens(records: list[dict]) -> list[dict]:
    return [{**record, "tokens": {key: token_row(record)[key] for key in ("ids", "segments", "logprobs")}}
            for record in records]


def settings_of_the_step(script, **changes):
    values = {"n_scenarios": 4, "group_size": 2, "reference_tokens": 200, "micro_batch": 2, "tie_tol": 0.0,
              **changes}
    return script.GRPOConfig(**values)


def test_the_grpo_loss_is_taken_on_the_kept_groups_and_the_sft_style_loss_on_every_valid_episode(script):
    records = with_tokens(step_records())
    gcfg = settings_of_the_step(script)
    plan = script.step_plan(records, gcfg)
    assert plan["kept_groups"] == [0, 2]
    assert plan["scenarios"] == ["tr-001/s1", "tr-002/s1", "tr-003/s1", "tr-004/s1"]
    assert plan["episodes"] == {"grpo": [0, 1, 4, 5], "sft_style": [0, 1, 2, 3, 4, 5, 6]}
    assert plan["advantages"].tolist() == [[0.5, -0.5], [0.0, 0.0], [-0.5, 0.5], [0.0, 0.0]]
    assert sorted(plan["weights"]) == [0, 1, 2, 3, 4, 5, 6]            # the harness error has no weights
    assert all(sum(weights) == SAMPLED_TOKENS for weights in plan["weights"].values())
    assert plan["weights"][0][:PROMPT_TOKENS + TURN_TOKENS + 1] == [0, 0, 0, 0, 0, 1, 1, 1, 0]
    assert plan["examples"][4] == {"ids": records[4]["tokens"]["ids"], "weights": plan["weights"][4]}
    # 4 * 2 * 200 = 1,600; 4 episodes of 12 sampled tokens are 48 loss tokens, 7 episodes are 84.
    assert script.describe_step(records, plan, gcfg) == {
        "n_episodes": 8, "n_harness_errors": 1, "n_groups": 4, "group_size": 2, "n_kept_groups": 2,
        "kept_groups": [{"group": 0, "scenario_id": "tr-001/s1"}, {"group": 2, "scenario_id": "tr-003/s1"}],
        "norm_const": 1600.0, "grpo": {"n_episodes": 4, "n_loss_tokens": 48},
        "sft_style": {"n_episodes": 7, "n_loss_tokens": 84}}


def test_a_half_of_the_kept_groups_holds_the_valid_episodes_of_its_groups(script):
    records = with_tokens(step_records())
    gcfg = settings_of_the_step(script)
    prepared = {"plan": script.step_plan(records, gcfg), "gcfg": gcfg}
    # Positions into the kept groups [0, 2]: position 1 is group 2, the episodes 4 and 5.
    assert script.half_episodes("grpo", [1], prepared) == [4, 5]
    assert script.half_episodes("grpo", [0, 1], prepared) == [0, 1, 4, 5]
    # Positions into the valid episodes.
    assert script.half_episodes("sft_style", [0, 3, 6], prepared) == [0, 3, 6]


def test_records_that_are_not_the_groups_of_the_settings_are_refused(script):
    records = with_tokens(step_records())
    # The same 8 episodes read as 2 groups of 4: a group would hold two scenarios.
    with pytest.raises(script.Refused, match="not 2 groups of 4"):
        script.step_plan(records, settings_of_the_step(script, n_scenarios=2, group_size=4))
    with pytest.raises(script.Refused, match="not 16 groups of 8"):
        script.step_plan(records, settings_of_the_step(script, n_scenarios=16, group_size=8))
    # 2 scenarios x 4 attempts read as 4 groups of 2: every group holds one scenario, but a scenario is
    # there twice. The library does not see this; the advantages would be taken inside half a group.
    twice = with_tokens(scenario_records("tr-001", 1, [100, 0, 100, 0])
                        + scenario_records("tr-002", 1, [0, 100, 100, 100]))
    with pytest.raises(script.Refused, match="hold 2 scenarios, not 4"):
        script.step_plan(twice, settings_of_the_step(script))


def test_an_episode_whose_log_probabilities_do_not_fit_its_sampled_tokens_is_refused(script):
    records = with_tokens(step_records())
    records[2]["tokens"]["logprobs"] = records[2]["tokens"]["logprobs"][:-1]
    with pytest.raises(script.Refused, match="episode 2 has 12 sampled tokens but 11"):
        script.step_plan(records, settings_of_the_step(script))


# ---------------------------------------------------------------------------
# The three stages, with a stand-in for the GPU part
# ---------------------------------------------------------------------------
ERROR_OF_WAY = {("bfloat16", 2): 0.05, ("bfloat16", 1): 0.02, ("float32", 1): 0.0}   # (dtype, micro-batch)
LOGPROB_SHIFT = {("bfloat16", 2): 0.004, ("bfloat16", 1): 0.001, ("float32", 1): 0.0}
FINGERPRINT = 62299.0
ADVANTAGES = [0.5, -0.5, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0]               # of the hand-made step, by episode
WHOLE = {"grpo": [0, 1, 4, 5], "sft_style": [0, 1, 2, 3, 4, 5, 6]}    # the episodes of each loss
# The peak GPU memory that the stand-in reports for each kind of pass, in GiB.
GRADIENT_PEAK = {"allocated": 9.5, "reserved": 14.0}
LOGPROB_PEAK = {"allocated": 5.25, "reserved": 14.0}
# The files that the bfloat16 stage writes for one way, as soon as that way is measured.
BF16_WAY_FILES = [f"stage_bfloat16.{loss}.{way}.{kind}" for loss in ("grpo", "sft_style")
                  for way in ("bfloat16_trainer_layout", "bfloat16_one_episode") for kind in ("json", "npz")]
FP32_WAY_FILES = [f"stage_float32.{loss}.float32_one_episode.{kind}" for loss in ("grpo", "sft_style")
                  for kind in ("json", "npz")]
HALVES_FILES = ["stage_bfloat16.grpo.halves.json", "stage_bfloat16.sft_style.halves.json"]


def episode_vector(index: int) -> np.ndarray:
    """The made-up gradient of one episode: fixed numbers that depend on the episode only."""
    return np.random.default_rng(index).normal(size=N_WEIGHTS)


def stand_in_gradient(loss: str, episodes: list[int], dtype: str, micro_batch: int,
                      advantages: list[float]) -> np.ndarray:
    """What the stand-in gives as the gradient: the sum over the episodes of factor x `episode_vector`,
    with the advantage as the factor for "grpo" and 1 / number of episodes for "sft_style"; plus an error
    that depends on the dtype and the micro-batch only. float32, as `.grad` of a LoRA weight is."""
    total = np.zeros(N_WEIGHTS)
    for index in episodes:
        total += (advantages[index] if loss == "grpo" else 1 / len(episodes)) * episode_vector(index)
    noise = np.random.default_rng(1000 + micro_batch).normal(size=N_WEIGHTS)
    return (total + ERROR_OF_WAY[(dtype, micro_batch)] * noise).astype(np.float32)


class StandInBackend:
    """In place of `ModelBackend`: no model. Its gradients are those of `stand_in_gradient`, and its
    log-probabilities are the sampler's plus a shift that depends on the dtype and the micro-batch."""

    n_layers = 3
    change_weights = False            # a test sets it to make the fingerprint move
    fail_when = None                  # a test sets a function (dtype, loss, episodes, micro-batch) -> bool:
    #                                   the gradient call for which it says True fails
    describe_fails = False            # a test sets it to make the description of the machine fail
    calls: list = []                  # (dtype, loss, episodes, micro-batch) of every gradient call
    files_at_describe: list = []      # the names in the output folder when `describe` was called last

    def __init__(self, args, dtype: str, prepared: dict) -> None:
        self.dtype, self.prepared, self.out = dtype, prepared, args.out
        self.advantages = prepared["plan"]["advantages"].reshape(-1).tolist()
        self.asked = 0

    def weights(self):
        return list(WEIGHTS)

    def fingerprint(self) -> float:
        self.asked += 1
        return FINGERPRINT + (0.5 if self.change_weights and self.asked > 1 else 0.0)

    def gradient(self, loss: str, episodes: list[int], micro_batch: int) -> dict:
        fail_when = type(self).fail_when
        if fail_when is not None and fail_when(self.dtype, loss, list(episodes), micro_batch):
            raise RuntimeError("the stand-in was told to fail here")
        type(self).calls.append((self.dtype, loss, list(episodes), micro_batch))
        weights = self.prepared["plan"]["weights"]
        return {"gradient": stand_in_gradient(loss, episodes, self.dtype, micro_batch, self.advantages),
                "loss": -0.001, "n_episodes": len(episodes),
                "n_loss_tokens": sum(sum(weights[index]) for index in episodes),
                "logprob_gap_to_sampler": {"n": 48, "mean": 0.002, "p99": 0.01, "max": 0.02} if loss == "grpo"
                else None,
                "seconds": 1.5, "peak_gpu_memory_gb": dict(GRADIENT_PEAK)}

    def logprobs(self, episodes: list[int], micro_batch: int) -> dict:
        recorded = [value for index in episodes
                    for value in self.prepared["records"][index]["tokens"]["logprobs"]]
        return {"logprobs": np.asarray(recorded) + LOGPROB_SHIFT[(self.dtype, micro_batch)],
                "seconds": 0.25, "peak_gpu_memory_gb": dict(LOGPROB_PEAK)}

    def largest_peak(self):
        return {"allocated": 9.5 if self.dtype == "bfloat16" else 12.25, "reserved": 14.0}

    def describe(self) -> dict:
        type(self).files_at_describe = sorted(path.name for path in self.out.iterdir())
        if type(self).describe_fails:
            raise RuntimeError("the machine cannot be read")
        return {"base_weight_dtypes": {f"torch.{self.dtype}": 2000}, "pad_id": 7,
                "tf32": {"switched_off_by_this_stage": {}, "switches": {"torch.backends.cudnn.allow_tf32":
                                                                         self.dtype == "bfloat16"}},
                "environment": {"gpu_name": "a stand-in"}, "kernels": {"fast_path": True},
                "kernel_sources": {"triton_version": "9.9"}}


@pytest.fixture
def run_folder(tmp_path) -> Path:
    """A run folder as scripts/gpu/train_grpo.py leaves it after one step with --save-tokens, for the
    hand-made step, and beside it the adapter that the run started from (two made-up files)."""
    records = step_records()
    folder = tmp_path / "pilot"
    (folder / "rollouts").mkdir(parents=True)
    write_records(folder / "rollouts" / "step_00000.episodes.jsonl", records)
    write_records(folder / "rollouts" / "step_00000.tokens.jsonl", [token_row(record) for record in records])
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"made-up adapter weights")
    hashes = {path.name: {"sha256": sha256_of(path)} for path in adapter.iterdir()}
    (folder / "run_manifest.json").write_text(json.dumps({
        "complete": True, "model": MODEL, "revision": REVISION, "seed": 100,
        "grpo_config": {"n_scenarios": 4, "group_size": 2, "reference_tokens": 200, "micro_batch": 2,
                        "tie_tol": 0.0, "gradient_checkpointing": True, "lr": 1e-5},
        "start_adapter": {"folder": str(adapter), "files": hashes}}), encoding="utf-8")
    log = {"step": 0, "loss": -0.001, "grad_norm": 0.07, "kept_groups": 2, "n_trained_episodes": 4,
           "n_loss_tokens": 48, "norm_const": 1600.0, "logprob_gap": {"n": 48, "mean": 0.002}, "fault": None}
    (folder / "train_log.jsonl").write_text(json.dumps(log) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def stand_in(script, monkeypatch):
    """The script with the GPU part replaced: `open_backend` gives a `StandInBackend`, and a stage of
    --stage all runs in this process (there is no torch here to start a model in a new one)."""
    StandInBackend.calls = []
    StandInBackend.files_at_describe = []
    monkeypatch.setattr(StandInBackend, "change_weights", False)
    monkeypatch.setattr(StandInBackend, "fail_when", None)
    monkeypatch.setattr(StandInBackend, "describe_fails", False)
    monkeypatch.setattr(script, "open_backend", StandInBackend)
    monkeypatch.setattr(script, "run_in_new_process",
                        lambda argv, stage: script.main([*argv, "--stage", stage]))
    return script


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_stages_write_the_result_and_its_numbers_are_those_of_the_gradients(stand_in, run_folder,
                                                                                tmp_path):
    out = tmp_path / "out"
    assert stand_in.main(task_command(run_folder, out)) == 0
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["layout_gradients.json", "stage_bfloat16.json", "stage_float32.json"]
        + BF16_WAY_FILES + FP32_WAY_FILES + HALVES_FILES)
    result = read_json(out / "layout_gradients.json")
    assert result["complete"] is True and (result["model"], result["revision"]) == (MODEL, REVISION)
    assert result["settings"] == {"n_scenarios": 4, "group_size": 2, "reference_tokens": 200,
                                  "micro_batch": 2, "tie_tol": 0.0, "gradient_checkpointing": True}
    assert set(result["settings_from"].values()) == {"run_manifest.json of the run"}
    assert result["step"]["n_kept_groups"] == 2 and result["step"]["norm_const"] == 1600.0
    assert result["weights"]["sets"] == {label: {"n_weights": ranges[0][1] - ranges[0][0], "ranges": ranges}
                                         for label, ranges in SETS.items()}
    assert (result["weights"]["n_weights"], result["weights"]["n_layers"]) == (N_WEIGHTS, 3)

    # The ways: dtype, micro-batch, episodes and loss tokens of each.
    advantages = ADVANTAGES
    episodes = WHOLE
    layouts = {TRAINER: ("bfloat16", 2), SINGLE: ("bfloat16", 1), REFERENCE: ("float32", 1)}
    for loss, indices in episodes.items():
        assert list(result["ways"][loss]) == [TRAINER, SINGLE, REFERENCE]
        for way, (dtype, micro_batch) in layouts.items():
            numbers = result["ways"][loss][way]
            gradient = stand_in_gradient(loss, indices, dtype, micro_batch, advantages)
            assert (numbers["dtype"], numbers["micro_batch"]) == (dtype, micro_batch)
            assert (numbers["n_episodes"], numbers["n_loss_tokens"]) == (
                len(indices), SAMPLED_TOKENS * len(indices))
            assert numbers["gradient_norm"] == pytest.approx(
                float(np.linalg.norm(gradient.astype(np.float64))))
            assert numbers["logprob_gap_to_sampler_second_pass"]["mean"] == pytest.approx(
                LOGPROB_SHIFT[(dtype, micro_batch)], abs=1e-12)
            # Seconds and peak GPU memory: one number for the gradient pass and one for the pass that
            # computes the log-probabilities, each under a name that says which pass it covers.
            assert (numbers["seconds_gradient_pass"], numbers["seconds_logprob_pass"]) == (1.5, 0.25)
            assert numbers["peak_gpu_memory_gb_gradient_pass"] == GRADIENT_PEAK
            assert numbers["peak_gpu_memory_gb_logprob_pass"] == LOGPROB_PEAK
            assert "seconds" not in numbers and "peak_gpu_memory_gb" not in numbers
            # The arrays of the way: its own file, with its hash; and the LoRA weights after the way.
            assert numbers["arrays"]["path"] == str(out / f"stage_{dtype}.{loss}.{way}.npz")
            assert numbers["arrays"]["sha256"] == sha256_of(out / f"stage_{dtype}.{loss}.{way}.npz")
            assert numbers["lora_fingerprint_after"] == FINGERPRINT

    # The comparisons: every pair, on every set of weights, against the same formula written out here.
    for loss, indices in episodes.items():
        made = {way: stand_in_gradient(loss, indices, *layout, advantages).astype(np.float64)
                for way, layout in layouts.items()}
        pairs = result["comparisons"][loss]
        assert [(pair["other"], pair["reference"]) for pair in pairs] == [
            (TRAINER, REFERENCE), (SINGLE, REFERENCE), (TRAINER, SINGLE)]
        for pair in pairs:
            for label, ((start, stop),) in SETS.items():
                a, b = made[pair["other"]][start:stop], made[pair["reference"]][start:stop]
                cell = pair["weights"][label]
                assert cell["cosine"] == pytest.approx(by_hand_cosine(a, b), abs=1e-12)
                assert cell["relative_difference"] == pytest.approx(
                    float(np.linalg.norm(a - b) / np.linalg.norm(b)), abs=1e-12)
        # The stand-in's error is larger in the trainer's layout, so that way is further from the reference.
        whole = [pair["weights"]["all LoRA weights"]["relative_difference"] for pair in pairs]
        assert whole[0] > whole[1] > 0

    # The log-probabilities: the shifts of the stand-in, one against the other.
    for loss in episodes:
        rows = result["logprob_differences"][loss]
        assert [row["n"] for row in rows] == [SAMPLED_TOKENS * len(episodes[loss])] * 3
        shifts = [pytest.approx(0.004), pytest.approx(0.001), pytest.approx(0.003)]
        assert [row["mean"] for row in rows] == shifts and [row["max"] for row in rows] == shifts

    # The table: one row per loss and set, with the same cosines and the summary of the halves.
    assert [(row["loss"], row["weights"]) for row in result["table"]] == [
        (loss, label) for loss in ("grpo", "sft_style") for label in SETS]
    first = result["table"][0]
    assert first["cosine"][f"{TRAINER} against {REFERENCE}"] == \
        result["comparisons"]["grpo"][0]["weights"]["all LoRA weights"]["cosine"]
    assert first["halves_cosine"] == result["halves"]["grpo"]["cosine_summary"]["all LoRA weights"]
    assert first["halves_relative_noise"] == \
        result["halves"]["grpo"]["relative_noise_summary"]["all LoRA weights"]

    # What a reader needs to trace the numbers.
    stages = result["stages"]
    assert (stages["bfloat16"]["dtype"], stages["float32"]["dtype"]) == ("bfloat16", "float32")
    assert stages["bfloat16"]["peak_gpu_memory_gb"] == {"allocated": 9.5, "reserved": 14.0}
    assert stages["float32"]["peak_gpu_memory_gb"]["allocated"] == 12.25
    assert stages["float32"]["tf32"]["switches"] == {"torch.backends.cudnn.allow_tf32": False}
    assert stages["bfloat16"]["kernel_sources"] == {"triton_version": "9.9"}
    assert stages["float32"]["base_weight_dtypes"] == {"torch.float32": 2000}
    assert stages["bfloat16"]["command"][-2:] == ["--stage", "bfloat16"]
    npz_files = {"bfloat16": [name for name in BF16_WAY_FILES if name.endswith(".npz")],
                 "float32": [name for name in FP32_WAY_FILES if name.endswith(".npz")]}
    for stage in ("bfloat16", "float32"):
        arrays = stages[stage]["arrays"]
        assert sorted(arrays) == sorted(npz_files[stage])
        assert all(entry["sha256"] == sha256_of(out / name) for name, entry in arrays.items())
        assert stages[stage]["kept_from_an_earlier_start"] == []
        assert stages[stage]["description_error"] is None
    assert result["code"]["this_script"]["path"].endswith("scripts/gpu/measure_layout_gradients.py")
    assert set(result["code"]) >= {"code_sha256", "frozen", "training_code", "impl", "this_script"}
    assert result["inputs"]["episodes"]["path"] == str(run_folder / "rollouts" / "step_00000.episodes.jsonl")


def test_the_halves_split_the_kept_groups_for_grpo_and_the_episodes_for_the_sft_style_loss(
        stand_in, run_folder, tmp_path):
    out = tmp_path / "out"
    assert stand_in.main(task_command(run_folder, out)) == 0
    halves = read_json(out / "layout_gradients.json")["halves"]
    advantages = ADVANTAGES

    def length(vector) -> float:
        return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))

    # GRPO: two kept groups (0 and 2) can be split in one way only: one group in each half.
    grpo = halves["grpo"]
    assert (grpo["unit"], grpo["n_units"], grpo["n_splits_asked"], grpo["n_splits"], grpo["micro_batch"]) == (
        "kept group", 2, 20, 1, 2)
    assert grpo["n_splits_planned"] == 1
    (split,) = grpo["splits"]
    assert sorted([split["half_a"], split["half_b"]]) == [[0], [2]] and split["n_episodes"] == [2, 2]
    # Each half is its part of the whole as it is: both are divided by the constant of the whole step.
    assert (split["n_loss_tokens"], split["share_of_the_whole"]) == ([24, 24], [1.0, 1.0])
    of_group = {0: [0, 1], 2: [4, 5]}
    a = stand_in_gradient("grpo", of_group[split["half_a"][0]], "bfloat16", 2, advantages).astype(np.float64)
    b = stand_in_gradient("grpo", of_group[split["half_b"][0]], "bfloat16", 2, advantages).astype(np.float64)
    for label, ((start, stop),) in SETS.items():
        assert split["cosine"][label] == pytest.approx(by_hand_cosine(a[start:stop], b[start:stop]),
                                                       abs=1e-12)
        assert grpo["cosine_summary"][label]["n"] == 1
        assert grpo["cosine_summary"][label]["mean"] == split["cosine"][label]
        # The new yardstick: |g_A - g_B| / |g_A + g_B|.
        by_hand = length(a[start:stop] - b[start:stop]) / length(a[start:stop] + b[start:stop])
        assert split["relative_noise"][label] == pytest.approx(by_hand, abs=1e-12)
        assert grpo["relative_noise_summary"][label]["n"] == 1
        assert grpo["relative_noise_summary"][label]["median"] == split["relative_noise"][label]

    # SFT style: the 7 valid episodes, 3 in one half and 4 in the other, in 20 different splits (35 exist).
    sft = halves["sft_style"]
    assert (sft["unit"], sft["n_units"], sft["n_splits"], sft["seed"]) == ("valid episode", 7, 20, 0)
    assert sft["n_splits_planned"] == 20
    assert len({frozenset((tuple(s["half_a"]), tuple(s["half_b"]))) for s in sft["splits"]}) == 20
    whole_gradient = stand_in_gradient("sft_style", WHOLE["sft_style"], "bfloat16", 2, advantages)
    for split in sft["splits"]:
        assert sorted(split["half_a"] + split["half_b"]) == [0, 1, 2, 3, 4, 5, 6]     # no harness error
        assert split["n_episodes"] == [3, 4]
        a = stand_in_gradient("sft_style", split["half_a"], "bfloat16", 2, advantages).astype(np.float64)
        b = stand_in_gradient("sft_style", split["half_b"], "bfloat16", 2, advantages).astype(np.float64)
        assert split["cosine"]["last layer (2)"] == pytest.approx(by_hand_cosine(a[12:], b[12:]), abs=1e-12)
        # Each half is a mean over its own tokens: 3 and 4 episodes of 12 sampled tokens. Its part of the
        # whole is the half times its share of the 84 tokens, and the two parts add up to the whole.
        assert split["n_loss_tokens"] == [36, 48]
        assert split["share_of_the_whole"] == [pytest.approx(3 / 7), pytest.approx(4 / 7)]
        part_a, part_b = 3 / 7 * a, 4 / 7 * b
        assert length(part_a + part_b - whole_gradient) < 1e-6 * length(whole_gradient)
        # The noise of a mean from halves of unequal size (36 and 48 tokens): the factor
        # sqrt(36 * 48) / 84 times the difference of the two half means, over the length of the whole.
        factor = math.sqrt(36 * 48) / 84
        assert split["relative_noise"]["all LoRA weights"] == pytest.approx(
            factor * length(a - b) / length(part_a + part_b), abs=1e-12)
        assert split["relative_noise"]["last layer (2)"] == pytest.approx(
            factor * length(a[12:] - b[12:]) / length(part_a[12:] + part_b[12:]), abs=1e-12)
    cosines = [split["cosine"]["all LoRA weights"] for split in sft["splits"]]
    summary = sft["cosine_summary"]["all LoRA weights"]
    assert summary["mean"] == pytest.approx(sum(cosines) / 20) and summary["min"] == min(cosines)
    assert summary["max"] == max(cosines) and summary["n"] == 20
    # The yardstick over the 20 splits: the median and the range.
    noise = [split["relative_noise"]["all LoRA weights"] for split in sft["splits"]]
    summary = sft["relative_noise_summary"]["all LoRA weights"]
    assert summary["median"] == pytest.approx(float(np.median(noise))) and summary["n"] == 20
    assert (summary["min"], summary["max"]) == (min(noise), max(noise))

    # The result says what the two numbers mean, what pure noise gives, and how the halves make the whole.
    for loss in ("grpo", "sft_style"):
        meaning = halves[loss]["meaning"]
        assert "near 0" in meaning["cosine"] and "near 1" in meaning["relative_noise"]
    assert "g_A + g_B" in halves["grpo"]["meaning"]["halves_and_whole"]
    assert "n_A / (n_A + n_B)" in halves["sft_style"]["meaning"]["halves_and_whole"]

    # Every half was asked for in the trainer's layout and in bfloat16, and never in float32.
    whole = WHOLE
    half_calls = [call for call in StandInBackend.calls if call[2] != whole[call[1]]]
    assert len(half_calls) == 2 * (1 + 20)
    assert {(dtype, micro_batch) for dtype, _, _, micro_batch in half_calls} == {("bfloat16", 2)}
    assert [call for call in StandInBackend.calls if call[0] == "float32"] == [
        ("float32", "grpo", whole["grpo"], 1), ("float32", "sft_style", whole["sft_style"], 1)]


def test_the_result_holds_the_trainers_own_numbers_of_the_step_beside_the_scripts(stand_in, run_folder,
                                                                                  tmp_path):
    out = tmp_path / "out"
    assert stand_in.main(task_command(run_folder, out)) == 0
    result = read_json(out / "layout_gradients.json")
    checks = result["checks"]
    assert checks["adapter_files_unchanged"] is True and checks["lora_weights_unchanged"] is True
    assert checks["same_lora_weights_in_both_stages"] is True
    assert checks["lora_fingerprint"]["bfloat16"] == {"before": FINGERPRINT, "after": FINGERPRINT,
                                                      "unchanged": True}
    log = checks["against_the_training_log"]
    assert log["adapter_holds_the_weights_of_the_step"] is True
    assert log["logged"] == {"loss": -0.001, "grad_norm": 0.07, "kept_groups": 2, "n_trained_episodes": 4,
                             "n_loss_tokens": 48, "norm_const": 1600.0,
                             "logprob_gap": {"n": 48, "mean": 0.002}}
    mine = log["this_script"]
    assert (mine["kept_groups"], mine["n_trained_episodes"], mine["n_loss_tokens"], mine["norm_const"]) == (
        2, 4, 48, 1600.0)
    assert mine["grad_norm"] == result["ways"]["grpo"][TRAINER]["gradient_norm"]
    assert (result["run"]["folder"], result["run"]["step"]) == (str(run_folder), 0)


def test_the_adapter_is_only_said_to_hold_the_weights_of_step_0_of_the_run_it_started(script):
    adapter = {"adapter_config.json": {"sha256": "aa"}, "adapter_model.safetensors": {"sha256": "bb"}}
    manifest = {"start_adapter": {"files": {"adapter_config.json": {"sha256": "aa"},
                                            "adapter_model.safetensors": {"sha256": "bb"}}}}
    assert script.weights_of_the_step(0, adapter, manifest) is True
    other = {"start_adapter": {"files": {"adapter_config.json": {"sha256": "aa"},
                                         "adapter_model.safetensors": {"sha256": "cc"}}}}
    assert script.weights_of_the_step(0, adapter, other) is False
    assert script.weights_of_the_step(3, adapter, manifest) is None   # the weights of step 3 are not saved
    assert script.weights_of_the_step(0, adapter, None) is None
    assert script.weights_of_the_step(0, adapter, {}) is None


def test_the_stages_can_be_run_one_by_one_and_compare_can_be_run_again_from_the_saved_files(
        stand_in, run_folder, tmp_path):
    out = tmp_path / "out"
    command = task_command(run_folder, out)
    assert stand_in.main(command + ["--stage", "bfloat16"]) == 0
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["stage_bfloat16.json"] + BF16_WAY_FILES + HALVES_FILES)
    assert stand_in.main(command + ["--stage", "float32"]) == 0
    calls = len(StandInBackend.calls)
    assert stand_in.main(command + ["--stage", "compare"]) == 0
    first = read_json(out / "layout_gradients.json")
    (out / "layout_gradients.json").unlink()
    assert stand_in.main(command + ["--stage", "compare"]) == 0
    again = read_json(out / "layout_gradients.json")
    assert again["comparisons"] == first["comparisons"] and again["table"] == first["table"]
    assert len(StandInBackend.calls) == calls                          # compare asks for no gradient
    # Each way has its own file of arrays: its gradient and its log-probabilities.
    with np.load(out / f"stage_bfloat16.grpo.{TRAINER}.npz") as saved:
        assert sorted(saved.files) == [f"grpo__{TRAINER}__gradient", f"grpo__{TRAINER}__logprobs"]
        assert saved[f"grpo__{TRAINER}__gradient"].dtype == np.float32
        assert saved[f"grpo__{TRAINER}__gradient"].shape == (N_WEIGHTS,)
        assert saved[f"grpo__{TRAINER}__logprobs"].shape == (SAMPLED_TOKENS * 4,)


# ---------------------------------------------------------------------------
# Saving first: what is measured is on disk before the next thing starts
# ---------------------------------------------------------------------------
def late_failure(dtype: str, loss: str, episodes: list[int], micro_batch: int) -> bool:
    """For `StandInBackend.fail_when`: the last way of the bfloat16 stage fails, the SFT-style loss with
    one episode per micro-batch. Everything before it is measured."""
    return (loss, episodes, micro_batch) == ("sft_style", WHOLE["sft_style"], 1)


SAVED_BEFORE_THE_LATE_FAILURE = sorted(
    [name for name in BF16_WAY_FILES if name.startswith("stage_bfloat16.grpo.")]
    + ["stage_bfloat16.grpo.halves.json", f"stage_bfloat16.sft_style.{TRAINER}.json",
       f"stage_bfloat16.sft_style.{TRAINER}.npz"])


def test_a_failure_in_a_later_way_keeps_the_earlier_ways_and_a_second_start_measures_only_the_rest(
        stand_in, run_folder, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    command = task_command(run_folder, out) + ["--stage", "bfloat16"]
    monkeypatch.setattr(StandInBackend, "fail_when", late_failure)
    assert stand_in.main(command) == 1
    assert "the stand-in was told to fail here" in capsys.readouterr().err
    # On disk: the three ways that were measured, each with its arrays, and the halves of the GRPO loss.
    # There is no stage_bfloat16.json: the stage is not complete.
    assert sorted(path.name for path in out.iterdir()) == SAVED_BEFORE_THE_LATE_FAILURE
    before = {name: sha256_of(out / name) for name in SAVED_BEFORE_THE_LATE_FAILURE}
    way = read_json(out / f"stage_bfloat16.grpo.{TRAINER}.json")
    assert way["complete"] is True
    assert (way["stage"], way["loss"], way["way"]) == ("bfloat16", "grpo", TRAINER)
    assert way["command"][-2:] == ["--stage", "bfloat16"]
    assert way["numbers"]["arrays"]["sha256"] == before[f"stage_bfloat16.grpo.{TRAINER}.npz"]
    with np.load(out / f"stage_bfloat16.grpo.{TRAINER}.npz") as saved:
        np.testing.assert_array_equal(
            saved[f"grpo__{TRAINER}__gradient"],
            stand_in_gradient("grpo", WHOLE["grpo"], "bfloat16", 2, ADVANTAGES))

    # The second start of the same command, with nothing failing: it measures only what is missing.
    monkeypatch.setattr(StandInBackend, "fail_when", None)
    StandInBackend.calls = []
    assert stand_in.main(command) == 0
    measured = [(loss, episodes, micro_batch) for _, loss, episodes, micro_batch in StandInBackend.calls]
    assert measured[0] == ("sft_style", WHOLE["sft_style"], 1)         # the way that had failed
    assert len(measured) == 1 + 2 * 20                                 # and the 20 splits after it
    assert all(loss == "sft_style" for loss, _, _ in measured)
    assert {name: sha256_of(out / name) for name in SAVED_BEFORE_THE_LATE_FAILURE} == before
    stage = read_json(out / "stage_bfloat16.json")
    assert stage["complete"] is True
    assert stage["kept_from_an_earlier_start"] == [
        f"stage_bfloat16.grpo.{TRAINER}.json", f"stage_bfloat16.grpo.{SINGLE}.json",
        "stage_bfloat16.grpo.halves.json", f"stage_bfloat16.sft_style.{TRAINER}.json"]
    assert "kept from an earlier start" in capsys.readouterr().out

    # The result is the result of a run in which nothing failed.
    assert stand_in.main(command[:-1] + ["float32"]) == 0
    assert stand_in.main(command[:-1] + ["compare"]) == 0
    clean = tmp_path / "clean"
    assert stand_in.main(task_command(run_folder, clean)) == 0
    result, wanted = read_json(out / "layout_gradients.json"), read_json(clean / "layout_gradients.json")
    assert result["complete"] is True and result["comparisons"] == wanted["comparisons"]
    for loss in ("grpo", "sft_style"):
        for key in ("cosine_summary", "relative_noise_summary"):
            assert result["halves"][loss][key] == wanted["halves"][loss][key]
    assert result["stages"]["bfloat16"]["kept_from_an_earlier_start"] == stage["kept_from_an_earlier_start"]


def test_after_a_failure_the_same_command_goes_on_or_says_which_stages_to_run(stand_in, run_folder,
                                                                              tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    command = task_command(run_folder, out)                            # --stage all
    # The first stage fails late. The same command again keeps what it saved and runs all three stages.
    monkeypatch.setattr(StandInBackend, "fail_when", late_failure)
    assert stand_in.main(command) == 1
    assert "keeps them and measures the rest" in capsys.readouterr().err
    monkeypatch.setattr(StandInBackend, "fail_when", None)
    assert stand_in.main(command) == 0
    result = read_json(out / "layout_gradients.json")
    assert result["complete"] is True and len(result["stages"]["bfloat16"]["kept_from_an_earlier_start"]) == 4
    assert result["stages"]["float32"]["kept_from_an_earlier_start"] == []

    # The second stage fails at its second way. The first stage is complete, so the same command is
    # refused and says what to do: the stages that are missing, one by one.
    out = tmp_path / "second"
    command = task_command(run_folder, out)
    monkeypatch.setattr(StandInBackend, "fail_when",
                        lambda dtype, loss, episodes, micro_batch: (dtype, loss) == ("float32", "sft_style"))
    assert stand_in.main(command) == 1
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["stage_bfloat16.json"] + BF16_WAY_FILES + HALVES_FILES
        + [f"stage_float32.grpo.{REFERENCE}.json", f"stage_float32.grpo.{REFERENCE}.npz"])
    monkeypatch.setattr(StandInBackend, "fail_when", None)
    capsys.readouterr()
    assert stand_in.main(command) == 2
    assert "one by one with --stage" in capsys.readouterr().err
    StandInBackend.calls = []
    assert stand_in.main(command + ["--stage", "float32"]) == 0
    assert StandInBackend.calls == [("float32", "sft_style", WHOLE["sft_style"], 1)]     # the missing way
    assert stand_in.main(command + ["--stage", "compare"]) == 0
    assert read_json(out / "layout_gradients.json")["stages"]["float32"]["kept_from_an_earlier_start"] == [
        f"stage_float32.grpo.{REFERENCE}.json"]


def test_a_failure_in_a_later_split_keeps_the_earlier_splits(stand_in, run_folder, tmp_path, monkeypatch):
    out = tmp_path / "out"
    command = task_command(run_folder, out) + ["--stage", "bfloat16"]
    seen = []

    def fourth_split(dtype: str, loss: str, episodes: list[int], micro_batch: int) -> bool:
        """True at the seventh half of the SFT-style loss: the first half of its fourth split."""
        if loss == "sft_style" and episodes != WHOLE["sft_style"]:
            seen.append(episodes)
        return len(seen) == 7

    monkeypatch.setattr(StandInBackend, "fail_when", fourth_split)
    assert stand_in.main(command) == 1
    saved = read_json(out / "stage_bfloat16.sft_style.halves.json")
    assert saved["complete"] is False
    assert (saved["halves"]["n_splits"], saved["halves"]["n_splits_planned"]) == (3, 20)
    assert len(saved["halves"]["splits"]) == 3
    assert saved["halves"]["relative_noise_summary"]["all LoRA weights"]["n"] == 3

    monkeypatch.setattr(StandInBackend, "fail_when", None)
    StandInBackend.calls = []
    assert stand_in.main(command) == 0
    assert len(StandInBackend.calls) == 2 * 17                         # only the 17 splits that were missing
    halves = read_json(out / "stage_bfloat16.json")["halves"]["sft_style"]
    assert halves["n_splits"] == 20 and halves["splits"][:3] == saved["halves"]["splits"]
    assert len({frozenset((tuple(s["half_a"]), tuple(s["half_b"]))) for s in halves["splits"]}) == 20
    assert read_json(out / "stage_bfloat16.sft_style.halves.json")["complete"] is True


def test_every_way_is_on_disk_before_the_machine_is_described_and_a_description_that_fails_loses_nothing(
        stand_in, run_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(StandInBackend, "describe_fails", True)
    out = tmp_path / "out"
    assert stand_in.main(task_command(run_folder, out) + ["--stage", "bfloat16"]) == 0
    # When the stand-in was asked to describe the machine, every way and every split was saved already.
    assert StandInBackend.files_at_describe == sorted(BF16_WAY_FILES + HALVES_FILES)
    stage = read_json(out / "stage_bfloat16.json")
    assert stage["complete"] is True
    assert stage["description_error"] == "RuntimeError: the machine cannot be read"
    assert "environment" not in stage and sorted(stage["arrays"]) == sorted(
        name for name in BF16_WAY_FILES if name.endswith(".npz"))
    # The other stages go on, and the result names the failure.
    assert stand_in.main(task_command(run_folder, out) + ["--stage", "float32"]) == 0
    assert stand_in.main(task_command(run_folder, out) + ["--stage", "compare"]) == 0
    result = read_json(out / "layout_gradients.json")
    assert result["complete"] is True
    assert result["stages"]["bfloat16"]["description_error"].startswith("RuntimeError")
    assert result["stages"]["bfloat16"]["environment"] is None


def test_what_an_earlier_start_saved_with_other_settings_or_other_code_is_measured_again(
        stand_in, run_folder, tmp_path, monkeypatch, capsys):
    for case in ("other settings", "other code"):
        out = tmp_path / case
        command = task_command(run_folder, out) + ["--stage", "bfloat16"]
        monkeypatch.setattr(StandInBackend, "fail_when", late_failure)
        assert stand_in.main(command) == 1
        monkeypatch.setattr(StandInBackend, "fail_when", None)
        StandInBackend.calls = []
        if case == "other settings":
            command += ["--split-seed", "5"]
        else:
            code_state = stand_in.code_state
            monkeypatch.setattr(stand_in, "code_state", lambda: {**code_state(), "this_script": "changed"})
        capsys.readouterr()
        assert stand_in.main(command) == 0
        assert "is not kept" in capsys.readouterr().out
        measured = [(loss, episodes, micro_batch) for _, loss, episodes, micro_batch in StandInBackend.calls]
        assert measured[0] == ("grpo", WHOLE["grpo"], 2)               # from the first way on
        assert len(measured) == 4 + 2 * (1 + 20)
        assert read_json(out / "stage_bfloat16.json")["kept_from_an_earlier_start"] == []


def test_the_result_says_beside_the_float32_numbers_what_the_reference_can_resolve(stand_in, run_folder,
                                                                                   tmp_path, capsys):
    out = tmp_path / "out"
    assert stand_in.main(task_command(run_folder, out)) == 0
    result = read_json(out / "layout_gradients.json")
    limit = result["float32_reference"]
    assert limit["resolves_relative_differences_above"] == 0.01
    assert "gradient_controls_result.md" in limit["measured_in"]
    for loss in ("grpo", "sft_style"):
        against_float32, against_float32_too, between_bfloat16 = result["comparisons"][loss]
        assert against_float32["reference_resolves_relative_differences_above"] == 0.01
        assert against_float32_too["reference_resolves_relative_differences_above"] == 0.01
        assert "reference_resolves_relative_differences_above" not in between_bfloat16
    assert read_json(out / "stage_float32.json")["float32_reference"] == limit
    assert "float32_reference" not in read_json(out / "stage_bfloat16.json")
    printed = capsys.readouterr().out
    assert "relative noise of the whole step" in printed and "below about 0.01" in printed


def test_compare_refuses_a_missing_stage_and_a_stage_that_was_run_on_other_settings(stand_in, run_folder,
                                                                                    tmp_path, capsys):
    out = tmp_path / "out"
    command = task_command(run_folder, out)
    assert stand_in.main(command + ["--stage", "bfloat16"]) == 0
    assert stand_in.main(command + ["--stage", "compare"]) == 2
    assert "holds no complete stage float32" in capsys.readouterr().err
    assert stand_in.main(command + ["--stage", "float32", "--split-seed", "5"]) == 0
    assert stand_in.main(command + ["--stage", "compare"]) == 2
    assert "the stage float32" in capsys.readouterr().err and not (out / "layout_gradients.json").exists()


def test_a_folder_that_holds_a_result_is_not_written_over(stand_in, run_folder, tmp_path, capsys):
    out = tmp_path / "out"
    command = task_command(run_folder, out)
    assert stand_in.main(command) == 0
    before = (out / "layout_gradients.json").read_bytes()
    for stage in ("all", "bfloat16", "float32", "compare"):
        assert stand_in.main(command + ["--stage", stage]) == 2
        assert "already holds" in capsys.readouterr().err
    assert (out / "layout_gradients.json").read_bytes() == before


def test_a_stage_fails_when_the_lora_weights_changed_and_the_later_stages_are_not_run(
        stand_in, run_folder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(StandInBackend, "change_weights", True)
    out = tmp_path / "out"
    assert stand_in.main(task_command(run_folder, out)) == 1
    err = capsys.readouterr().err
    assert "THE LORA WEIGHTS CHANGED" in err and "the stages after it were not run" in err
    assert read_json(out / "stage_bfloat16.json")["lora_fingerprint"]["unchanged"] is False
    assert not (out / "stage_float32.json").exists() and not (out / "layout_gradients.json").exists()


def test_inputs_that_are_missing_or_of_another_model_are_refused_before_a_model_loads(stand_in, run_folder,
                                                                                    tmp_path, capsys):
    out = tmp_path / "out"
    command = task_command(run_folder, out)
    cases = [(command[:5] + [str(tmp_path / "nothing.jsonl")] + command[6:], "--episodes"),
             (command[:3] + [str(tmp_path / "no_adapter")] + command[4:], "is not a saved adapter"),
             (["--model", "Qwen/Qwen3.5-4B"] + command[2:], "the run was made with"),
             (command + ["--group-size", "4"], "differ from the run's manifest"),
             (command + ["--splits", "0"], "--splits must be at least 1")]
    for argv, message in cases:
        assert stand_in.main(argv) == 2
        assert message in capsys.readouterr().err
    assert StandInBackend.calls == [] and not out.exists()


# ---------------------------------------------------------------------------
# The glue of the GPU side, with stand-ins for torch, the model and the library call. `ModelBackend` as a
# whole needs torch and a model and is not tested: an object of it is made here without loading anything.
# ---------------------------------------------------------------------------
def unloaded_backend(script, cuda: SimpleNamespace):
    """A `ModelBackend` on which `__init__` did not run: no torch, no model. Only what a test sets is
    there."""
    backend = object.__new__(script.ModelBackend)
    backend.torch = SimpleNamespace(cuda=cuda)
    backend.timing_and_memory_errors = {}
    backend.peaks = []
    return backend


def test_a_call_that_only_serves_the_seconds_or_the_peak_memory_cannot_cost_a_measurement(script):
    calls = []
    cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: calls.append("synchronize"),
                           reset_peak_memory_stats=lambda: calls.append("reset"))
    backend = unloaded_backend(script, cuda)
    backend._synchronize()
    backend._reset_peak()
    assert calls == ["synchronize", "reset"] and backend.timing_and_memory_errors == {}

    def fails():
        raise RuntimeError("CUDA error: it came late")

    cuda.synchronize = cuda.reset_peak_memory_stats = fails
    backend._synchronize()                                             # no exception: the stage goes on
    backend._reset_peak()
    assert backend.timing_and_memory_errors == {
        "torch.cuda.synchronize": "RuntimeError: CUDA error: it came late",
        "torch.cuda.reset_peak_memory_stats": "RuntimeError: CUDA error: it came late"}
    # Without a CUDA device neither call is made.
    cuda.is_available = lambda: False
    backend.timing_and_memory_errors = {}
    backend._synchronize()
    backend._reset_peak()
    assert backend.timing_and_memory_errors == {}


def test_the_pass_for_the_log_probabilities_gives_its_own_seconds_and_its_own_peak_memory(script,
                                                                                          monkeypatch):
    import accessdesk.train.forward as forward

    class Tensor:
        """In place of a torch tensor: only the four calls that turn it into a numpy array."""

        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def double(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.values, dtype=np.float64)

    modes, events, seen = [], [], {}

    def token_logprobs(model, seqs, weights, *, pad_id, micro_batch, with_grad):
        seen.update(seqs=seqs, weights=weights, pad_id=pad_id, micro_batch=micro_batch, with_grad=with_grad,
                    mode=list(modes))
        return [Tensor([-0.5, -1.0]), Tensor([-2.0])]

    monkeypatch.setattr(forward, "token_logprobs", token_logprobs)
    monkeypatch.setattr(script.train_sft_script, "peak_gpu_memory_gb", lambda: dict(LOGPROB_PEAK))
    backend = unloaded_backend(script, SimpleNamespace(
        is_available=lambda: True, synchronize=lambda: events.append("synchronize"),
        reset_peak_memory_stats=lambda: events.append("reset")))
    backend.model = SimpleNamespace(eval=lambda: modes.append("eval"), train=lambda: modes.append("train"))
    backend.pad_id = 7
    backend.records = [{"tokens": {"ids": [1, 2, 3]}}, {"tokens": {"ids": [4, 5]}}]
    backend.plan = {"weights": {0: [0, 1, 1], 1: [0, 1]}}

    result = backend.logprobs([0, 1], 2)
    assert result["logprobs"].tolist() == [-0.5, -1.0, -2.0] and result["logprobs"].dtype == np.float64
    assert result["peak_gpu_memory_gb"] == LOGPROB_PEAK and result["seconds"] >= 0
    assert events == ["reset", "synchronize"]          # the peak starts anew before the pass
    assert backend.peaks == [LOGPROB_PEAK]             # and counts for the largest peak of the stage
    assert backend.largest_peak() == LOGPROB_PEAK
    # The library call: the stored ids and weights, no gradients, in eval mode; train mode after it.
    assert seen == {"seqs": [[1, 2, 3], [4, 5]], "weights": [[0, 1, 1], [0, 1]], "pad_id": 7,
                    "micro_batch": 2, "with_grad": False, "mode": ["eval"]}
    assert modes == ["eval", "train"]
    nothing = backend.logprobs([], 2)
    assert nothing["logprobs"].shape == (0,) and nothing["peak_gpu_memory_gb"] is None
    assert backend.peaks == [LOGPROB_PEAK]


def test_the_description_of_the_machine_comes_back_whole_when_its_reads_fail(script):
    """`ModelBackend.describe` on a stand-in. The model here is not a model, so the library's kernel
    report fails; on this laptop there is no torch, so its environment report fails too. Each failure
    must come back as an error text, beside what could be read."""
    backend = unloaded_backend(script, SimpleNamespace())
    backend.torch = stand_in_torch()
    backend.model = SimpleNamespace(eval=lambda: None)
    backend.base_dtypes, backend.pad_id = {"torch.float32": 10}, 7
    backend.tf32_switched_off = {"torch.backends.cudnn.allow_tf32": "set to False"}
    backend.timing_and_memory_errors = {"torch.cuda.synchronize": "RuntimeError: late"}
    description = backend.describe()
    assert set(description) == {"base_weight_dtypes", "pad_id", "timing_and_memory_errors", "tf32",
                                "environment", "kernels", "kernel_sources"}
    assert (description["base_weight_dtypes"], description["pad_id"]) == ({"torch.float32": 10}, 7)
    assert description["timing_and_memory_errors"] == {"torch.cuda.synchronize": "RuntimeError: late"}
    assert description["tf32"]["switched_off_by_this_stage"] == {
        "torch.backends.cudnn.allow_tf32": "set to False"}
    assert description["tf32"]["switches"]["torch.backends.cudnn.conv.fp32_precision"] == "tf32"
    assert description["kernels"]["error"].startswith("TypeError: expected the text-only class")
    assert isinstance(description["environment"], dict)        # the report, or {"error": ...} without torch
    assert set(description["kernel_sources"]) == {"triton_version", "TRITON_F32_DEFAULT",
                                                  "tf32_in_triton_language", "dot_calls_in_fla_ops"}
    json.dumps(description)                                    # everything in it can go into a JSON file


def test_the_script_can_be_imported_without_torch():
    """--stage compare needs numpy only, so the saved gradients can be compared again on a laptop."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import measure_layout_gradients; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code, str(GPU_SCRIPTS)], capture_output=True).returncode == 0
