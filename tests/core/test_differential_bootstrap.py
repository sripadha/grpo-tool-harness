"""The second version of `paired_family_bootstrap` against the main version on many generated inputs.

The comparison at the top is skipped while the second version (accessdesk/_second/bootstrap.py) is not
written. So that a skipped test is not an untested test, the rest of the file runs today: the comparison
passes for correct versions and fails for each of several wrong versions.

The contract fixes how the random indices are drawn (one generator from the seed, one call for all indices,
one row per draw) and how the limits are taken (numpy's default quantile). So two correct versions give the
same numbers, and the comparison asks for `diff`, `lo` and `hi` equal to 1e-12. That small slack is for the
order in which a version adds up the same numbers; a version that draws other indices is far outside it.

Further down, added by the comparison of the two versions: edge inputs. Every combination of 12 numbers of
families (from 1 up to 257), 5 levels (0.5 to 0.999) and 6 numbers of draws (from 1 up to 1000), each with
its own seed and with one of seven kinds of data (equal arrays, a constant gain, all zeros, 0-and-1 ints,
bools, steps of 1/32, any floats); and inputs that must be refused. Where the answer is known without any
version it is asserted: equal arrays give three zeros, one family gives lo = hi = diff.
"""
from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest

from accessdesk import impl
from accessdesk.eval import bootstrap as main
from tests.conftest import VERSIONS, function_under_test, skip_unless_second_is_written

N_DRAWS = 4000
TOLERANCE = 1e-12


def bootstrap_inputs(n_cases: int = 40):
    """Generated (a, b, seed, level): family pass rates in steps of 1/32, with a shared family difficulty."""
    rng = np.random.default_rng(3)
    for case in range(n_cases):
        n_families = int(rng.integers(8, 61))
        difficulty = rng.random(n_families)
        a = np.round(32 * np.clip(difficulty + rng.normal(0.0, 0.05, n_families), 0.0, 1.0)) / 32
        gain = rng.choice([0.0, 0.1, 0.5], size=n_families)
        b = np.round(32 * np.clip(difficulty + gain + rng.normal(0.0, 0.05, n_families), 0.0, 1.0)) / 32
        if case % 10 == 0:
            b = a + 0.125                      # the same gain in every family: an interval of width 0
        yield a, b, case, float(rng.choice([0.8, 0.9, 0.95]))


def compare_bootstrap(other_function, main_function) -> None:
    """Assert that the two functions give the same three numbers on every generated input."""
    for a, b, seed, level in bootstrap_inputs():
        expected = main_function(a.copy(), b.copy(), N_DRAWS, seed, level)
        found = other_function(a.copy(), b.copy(), N_DRAWS, seed, level)
        message = f"a={a.tolist()} b={b.tolist()} seed={seed} level={level}"
        assert sorted(found) == ["diff", "hi", "lo"], message
        for key in ("diff", "lo", "hi"):
            assert type(found[key]) is float, f"{key} is not a Python float; {message}"
            assert abs(found[key] - expected[key]) <= TOLERANCE, f"{key}: {message}"


# ---------------------------------------------------------------------------
# The second version (skipped until it is written)
# ---------------------------------------------------------------------------
def test_the_second_paired_family_bootstrap_matches_the_main_version():
    skip_unless_second_is_written("paired_family_bootstrap")
    compare_bootstrap(impl.second_version("paired_family_bootstrap"), main.paired_family_bootstrap)


# ---------------------------------------------------------------------------
# The comparison itself: it passes for correct versions and fails for wrong ones
# ---------------------------------------------------------------------------
def _percentile_interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return float(lo), float(hi)


def _correct_with_a_loop(a, b, n_draws, seed, level):
    """A second correct version, written differently: the contract's procedure as a plain loop over the
    draws, with the statistic taken on the per-family differences b - a."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_draws, a.size))
    d = b - a
    draws = np.array([sum(d[i] for i in row) / a.size for row in idx])
    lo, hi = _percentile_interval(draws, level)
    return {"diff": float(np.mean(b) - np.mean(a)), "lo": lo, "hi": hi}


def _wrong_other_random_numbers(a, b, n_draws, seed, level):
    """A sound bootstrap that is not the contract's: it draws its indices from another stream."""
    rng = np.random.default_rng(seed + 1000)
    d = b - a
    draws = d[rng.integers(0, d.size, (n_draws, d.size))].mean(axis=1)
    lo, hi = _percentile_interval(draws, level)
    return {"diff": float(b.mean() - a.mean()), "lo": lo, "hi": hi}


def _wrong_one_call_per_draw(a, b, n_draws, seed, level):
    """Not the contract's draws: one call of the generator for each draw, each for both models at once
    (two rows per call, of which the first is used)."""
    rng = np.random.default_rng(seed)
    d = b - a
    draws = np.array([d[rng.integers(0, d.size, size=(2, d.size))[0]].mean() for _ in range(n_draws)])
    lo, hi = _percentile_interval(draws, level)
    return {"diff": float(b.mean() - a.mean()), "lo": lo, "hi": hi}


def _wrong_quantile_method(a, b, n_draws, seed, level):
    """The contract's draws, but the limits are taken without interpolation (the nearest lower draw)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_draws, a.size))
    draws = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail], method="lower")
    return {"diff": float(b.mean() - a.mean()), "lo": float(lo), "hi": float(hi)}


def _wrong_unpaired(a, b, n_draws, seed, level):
    rng = np.random.default_rng(seed)
    draws = b[rng.integers(0, b.size, (n_draws, b.size))].mean(axis=1) \
        - a[rng.integers(0, a.size, (n_draws, a.size))].mean(axis=1)
    lo, hi = _percentile_interval(draws, level)
    return {"diff": float(b.mean() - a.mean()), "lo": lo, "hi": hi}


def _wrong_a_minus_b(a, b, n_draws, seed, level):
    return main.paired_family_bootstrap(b, a, n_draws, seed, level)


def _wrong_one_sided_tails(a, b, n_draws, seed, level):
    """Uses the quantiles 1 - level and level (5% and 95% for a 95% interval): too narrow."""
    rng = np.random.default_rng(seed)
    d = b - a
    draws = d[rng.integers(0, d.size, (n_draws, d.size))].mean(axis=1)
    lo, hi = np.quantile(draws, [1.0 - level, level])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def _wrong_resamples_without_replacement(a, b, n_draws, seed, level):
    """Every draw is a reshuffle of all families: every draw equals the plain difference."""
    d = float(b.mean() - a.mean())
    return {"diff": d, "lo": d, "hi": d}


def _wrong_standard_deviation_of_families(a, b, n_draws, seed, level):
    """An interval of the families themselves and not of their mean (forgets the division by sqrt(n))."""
    d = b - a
    lo, hi = _percentile_interval(d, level)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi}


def _wrong_numpy_floats(a, b, n_draws, seed, level):
    """The right numbers, but as numpy scalars: the table could not be saved as JSON."""
    result = main.paired_family_bootstrap(a, b, n_draws, seed, level)
    return {key: np.float64(value) for key, value in result.items()}


WRONG_BOOTSTRAPS = [_wrong_other_random_numbers, _wrong_one_call_per_draw, _wrong_quantile_method,
                    _wrong_unpaired, _wrong_a_minus_b, _wrong_one_sided_tails,
                    _wrong_resamples_without_replacement, _wrong_standard_deviation_of_families,
                    _wrong_numpy_floats]


def test_generated_inputs_have_wide_and_zero_width_cases():
    widths = []
    for a, b, seed, level in bootstrap_inputs():
        result = main.paired_family_bootstrap(a, b, N_DRAWS, seed, level)
        widths.append(result["hi"] - result["lo"])
    assert len(widths) == 40
    assert sum(1 for width in widths if width < 1e-12) == 4
    assert sum(1 for width in widths if width > 0.05) >= 30


def test_comparison_passes_for_the_main_version_against_itself():
    compare_bootstrap(main.paired_family_bootstrap, main.paired_family_bootstrap)


def test_comparison_passes_for_a_correct_version_that_is_written_differently():
    compare_bootstrap(_correct_with_a_loop, main.paired_family_bootstrap)


@pytest.mark.parametrize("wrong", WRONG_BOOTSTRAPS, ids=lambda function: function.__name__.lstrip("_"))
def test_comparison_fails_for_a_wrong_bootstrap(wrong):
    with pytest.raises(AssertionError):
        compare_bootstrap(wrong, main.paired_family_bootstrap)


# ---------------------------------------------------------------------------
# Edge inputs: few families, few draws, other levels and seeds, special data
# ---------------------------------------------------------------------------
EDGE_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 100, 257)
EDGE_LEVELS = (0.5, 0.8, 0.95, 0.99, 0.999)
EDGE_DRAWS = (1, 2, 3, 10, 101, 1000)
EDGE_SEEDS = (0, 1, 7, 12345, 2**32 + 7, 2**63 - 1)
DATA_KINDS = ("floats", "steps", "equal", "constant_gain", "zeros", "ints", "bools")


@functools.lru_cache(maxsize=None)
def edge_bootstrap_inputs() -> list[tuple]:
    """(a, b, n_draws, seed, level, kind of data) for every combination of size, level and number of draws.
    The kind of data and the seed go round, so every kind meets every size and every level."""
    rng = np.random.default_rng(8)
    inputs = []
    combinations = itertools.product(EDGE_SIZES, EDGE_LEVELS, EDGE_DRAWS)
    for number, (n_families, level, n_draws) in enumerate(combinations):
        kind = DATA_KINDS[(number + number // len(DATA_KINDS)) % len(DATA_KINDS)]
        a, b = rng.random(n_families), rng.random(n_families)
        if kind == "steps":
            a, b = np.round(32 * a) / 32, np.round(32 * b) / 32
        elif kind == "equal":
            b = a.copy()
        elif kind == "constant_gain":
            a = np.round(32 * a) / 64
            b = a + 0.25                               # exact in floats: the interval has width 0
        elif kind == "zeros":
            a, b = np.zeros(n_families), np.zeros(n_families)
        elif kind == "ints":
            a, b = (a < 0.5).astype(int), (b < 0.7).astype(int)
        elif kind == "bools":
            a, b = a < 0.5, b < 0.7
        inputs.append((a, b, n_draws, EDGE_SEEDS[number % len(EDGE_SEEDS)], level, kind))
    return inputs


def compare_bootstrap_on_the_edge_inputs(other_function, main_function, kinds=DATA_KINDS,
                                         largest: int = 257) -> None:
    """Assert that the two functions give the same three Python floats on every edge input of the named
    kinds with at most `largest` families, and the answers that are known without any version."""
    for a, b, n_draws, seed, level, kind in edge_bootstrap_inputs():
        if kind not in kinds or a.size > largest:
            continue
        expected = main_function(a.copy(), b.copy(), n_draws, seed, level)
        found = other_function(a.copy(), b.copy(), n_draws, seed, level)
        message = f"{kind}: a={a.tolist()} b={b.tolist()} n_draws={n_draws} seed={seed} level={level}"
        assert sorted(found) == ["diff", "hi", "lo"], message
        for key in ("diff", "lo", "hi"):
            assert type(found[key]) is float, f"{key} is not a Python float; {message}"
            assert abs(found[key] - expected[key]) <= TOLERANCE, f"{key}: {message}"
        assert found["lo"] <= found["hi"], message
        if kind in ("equal", "zeros"):
            assert found == {"diff": 0.0, "lo": 0.0, "hi": 0.0}, message
        if kind == "constant_gain":             # 0.25 in every family; a mean may round in its last bit
            assert all(abs(found[key] - 0.25) <= TOLERANCE for key in ("diff", "lo", "hi")), message
        if a.size == 1:
            assert found["lo"] == found["hi"] == found["diff"] == float(b[0]) - float(a[0]), message
        if n_draws == 1:
            assert found["lo"] == found["hi"], message


def refused_bootstrap_inputs() -> list[tuple]:
    """(a, b, n_draws, seed, level) that the contract refuses."""
    a, b = np.array([0.0, 0.25, 0.5, 1.0]), np.array([0.5, 0.25, 1.0, 1.0])
    return [(a, b, 0, 0, 0.95), (a, b, -1, 0, 0.95), (a, b, -1000, 0, 0.95),
            (a, b, 50, 0, 0.0), (a, b, 50, 0, 1.0), (a, b, 50, 0, -0.1), (a, b, 50, 0, 1.5),
            (a, b, 50, 0, float("nan")), (a, b, 50, 0, float("inf")), (a, b, 50, 0, 95.0),
            (a, b[:3], 50, 0, 0.95), (a[:1], b, 50, 0, 0.95), (a, np.array([]), 50, 0, 0.95),
            (np.array([]), np.array([]), 50, 0, 0.95), ([], [], 50, 0, 0.95),
            (a.reshape(2, 2), b.reshape(2, 2), 50, 0, 0.95),
            (a.reshape(4, 1), b.reshape(4, 1), 50, 0, 0.95),
            (a, b.reshape(4, 1), 50, 0, 0.95), (a.reshape(1, 4), b, 50, 0, 0.95),
            (np.array(0.5), np.array(0.7), 50, 0, 0.95), (0.5, 0.7, 50, 0, 0.95)]


def test_the_second_paired_family_bootstrap_matches_the_main_version_on_the_edge_inputs():
    skip_unless_second_is_written("paired_family_bootstrap")
    compare_bootstrap_on_the_edge_inputs(impl.second_version("paired_family_bootstrap"),
                                         main.paired_family_bootstrap)


@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_matches_the_plain_loop_on_the_small_edge_inputs(version):
    # The plain loop adds up one Python float at a time, so it is slow: sizes up to 17, float data only
    # (it subtracts the arrays, which numpy refuses for bools).
    kinds = ("floats", "steps", "equal", "constant_gain", "zeros")
    compare_bootstrap_on_the_edge_inputs(function_under_test("paired_family_bootstrap", version),
                                         _correct_with_a_loop, kinds=kinds, largest=17)


@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_refuses_what_the_contract_refuses(version):
    function = function_under_test("paired_family_bootstrap", version)
    for args in refused_bootstrap_inputs():
        with pytest.raises(ValueError):
            function(*args)


@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_reads_lists_ints_and_bools_like_float_arrays(version):
    function = function_under_test("paired_family_bootstrap", version)
    a, b = [0, 1, 1, 0, 1], [1, 1, 0, 1, 1]                 # pass rates 0.6 and 0.8
    as_floats = function(np.array(a, dtype=float), np.array(b, dtype=float), 200, 3, 0.9)
    assert function(a, b, 200, 3, 0.9) == as_floats
    assert function(tuple(float(x) for x in a), tuple(float(x) for x in b), 200, 3, 0.9) == as_floats
    assert function([bool(x) for x in a], [bool(x) for x in b], 200, 3, 0.9) == as_floats
    assert function(np.array(a), np.array(b), 200, 3, 0.9) == as_floats
    assert as_floats["diff"] == pytest.approx(0.8 - 0.6, abs=1e-12)


def test_the_edge_inputs_hold_what_they_claim():
    inputs = edge_bootstrap_inputs()
    assert len(inputs) == 12 * 5 * 6
    every = itertools.product
    assert {(a.size, kind) for a, _, _, _, _, kind in inputs} == set(every(EDGE_SIZES, DATA_KINDS))
    assert {(level, kind) for _, _, _, _, level, kind in inputs} == set(every(EDGE_LEVELS, DATA_KINDS))
    assert {(a.size, n_draws) for a, _, n_draws, _, _, _ in inputs} == set(every(EDGE_SIZES, EDGE_DRAWS))
    assert {seed for _, _, _, seed, _, _ in inputs} == set(EDGE_SEEDS)
    widths = []
    for a, b, n_draws, seed, level, _ in inputs:
        result = main.paired_family_bootstrap(a, b, n_draws, seed, level)
        widths.append(result["hi"] - result["lo"])
    assert sum(1 for width in widths if width > 0.05) > 100
    assert sum(1 for width in widths if width == 0.0) > 100


@pytest.mark.parametrize("wrong", [w for w in WRONG_BOOTSTRAPS if w.__name__ != "_wrong_numpy_floats"],
                         ids=lambda function: function.__name__.lstrip("_"))
def test_the_edge_comparison_fails_for_a_wrong_bootstrap(wrong):
    with pytest.raises(AssertionError):
        compare_bootstrap_on_the_edge_inputs(wrong, main.paired_family_bootstrap,
                                             kinds=("floats", "steps"))
