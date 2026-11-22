"""accessdesk/rng.py: derive_seed and Rng."""
from __future__ import annotations

import ast
import hashlib
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from accessdesk.rng import TOKEN_ALPHABET, Rng, derive_seed

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# derive_seed
# ---------------------------------------------------------------------------
def test_derive_seed_is_the_first_8_bytes_of_sha256_of_the_joined_parts():
    assert derive_seed(1, "train", 0) == int.from_bytes(hashlib.sha256(b"1|train|0").digest()[:8], "big")


def test_derive_seed_fixed_value():
    # Computed once on 2026-11-16. If this number changes, every generated scenario changes.
    assert derive_seed(1, "train", 0) == 16688891439445519583


def test_derive_seed_keeps_the_parts_apart():
    assert derive_seed(1, 23) != derive_seed(12, 3)
    assert derive_seed("a", "b") != derive_seed("ab")
    assert derive_seed(1, "train", 0) != derive_seed(1, "train", 1)
    assert derive_seed(1, "train", 0) != derive_seed(2, "train", 0)
    assert derive_seed(1, "train", 0) != derive_seed(1, "dev", 0)


def test_derive_seed_fits_in_64_bits():
    assert all(0 <= derive_seed("x", i) < 2 ** 64 for i in range(200))


# ---------------------------------------------------------------------------
# Rng
# ---------------------------------------------------------------------------
def test_fixed_stream():
    # Pins the algorithm: int(random.Random(seed).random() * n). Computed once on 2026-11-16.
    rng = Rng(123)
    assert [rng.below(10) for _ in range(12)] == [0, 0, 4, 1, 9, 0, 5, 3, 8, 1, 3, 3]


def test_fixed_values_of_every_method():
    rng = Rng(7)
    assert rng.shuffle(list(range(8))) == [5, 4, 6, 7, 0, 3, 1, 2]
    assert rng.sample("abcdefgh", 3) == ["d", "f", "b"]
    assert rng.token(6) == "dgw8up"
    assert rng.chance(0.5) is False
    assert rng.choice("xyz") == "x"


def test_same_seed_same_stream_and_other_seed_other_stream():
    a, b, c = Rng(5), Rng(5), Rng(6)
    stream_a = [a.below(1000) for _ in range(50)]
    assert stream_a == [b.below(1000) for _ in range(50)]
    assert stream_a != [c.below(1000) for _ in range(50)]


def test_the_global_random_module_has_no_effect():
    a = Rng(11)
    first = [a.below(1000) for _ in range(20)]
    b = Rng(11)
    second = []
    for _ in range(20):
        random.seed(0)          # disturb the module-level generator between draws
        random.random()
        second.append(b.below(1000))
    assert first == second


def test_seed_must_be_an_int():
    for bad in ("1", 1.0, None, True):
        with pytest.raises(TypeError):
            Rng(bad)


def test_below_is_in_range_and_close_to_uniform():
    rng = Rng(1)
    counts = Counter(rng.below(6) for _ in range(12000))
    assert sorted(counts) == [0, 1, 2, 3, 4, 5]
    assert all(abs(count - 2000) < 200 for count in counts.values()), counts   # sd is about 41


def test_below_needs_a_positive_n():
    assert Rng(1).below(1) == 0
    for bad in (0, -1):
        with pytest.raises(ValueError):
            Rng(1).below(bad)


def test_chance():
    rng = Rng(2)
    assert not any(rng.chance(0.0) for _ in range(500))
    assert all(rng.chance(1.0) for _ in range(500))
    hits = sum(rng.chance(0.25) for _ in range(8000))
    assert abs(hits - 2000) < 200, hits                                        # sd is about 39


def test_chance_always_uses_one_draw():
    a, b = Rng(3), Rng(3)
    a.chance(0.0), a.chance(1.0), a.chance(0.5)
    b.below(2), b.below(2), b.below(2)
    assert a.below(10 ** 6) == b.below(10 ** 6)


def test_choice_returns_elements_and_reaches_all_of_them():
    rng = Rng(4)
    seen = {rng.choice(("a", "b", "c")) for _ in range(200)}
    assert seen == {"a", "b", "c"}
    with pytest.raises(ValueError):
        rng.choice(())


def test_shuffle_returns_a_new_permutation_and_leaves_the_input_alone():
    original = list(range(20))
    given = list(original)
    result = Rng(8).shuffle(given)
    assert given == original
    assert result is not given
    assert sorted(result) == original
    assert result != original
    assert Rng(8).shuffle(()) == []
    assert Rng(8).shuffle((1,)) == [1]


def test_shuffle_gives_every_order_equally_often():
    # A wrong shuffle (swap with any position at every step) gives 4000 or 5000 here instead of 4500.
    rng = Rng(9)
    counts = Counter(tuple(rng.shuffle("abc")) for _ in range(27000))
    assert len(counts) == 6
    assert all(abs(count - 4500) < 250 for count in counts.values()), counts   # sd is about 61


def test_sample():
    rng = Rng(10)
    items = list("abcdefghij")
    for k in range(len(items) + 1):
        picked = rng.sample(items, k)
        assert len(picked) == k == len(set(picked))
        assert set(picked) <= set(items)
    assert sorted(rng.sample(items, len(items))) == items
    with pytest.raises(ValueError):
        rng.sample(items, len(items) + 1)
    with pytest.raises(ValueError):
        rng.sample(items, -1)


def test_sample_keeps_equal_elements_at_different_positions():
    assert sorted(Rng(1).sample(["x", "x", "y"], 3)) == ["x", "x", "y"]


def test_token_alphabet_and_length():
    assert TOKEN_ALPHABET == "abcdefghjkmnpqrstuvwxyz23456789"
    assert len(TOKEN_ALPHABET) == 31 == len(set(TOKEN_ALPHABET))
    assert not set("ilo01") & set(TOKEN_ALPHABET)
    rng = Rng(12)
    tokens = [rng.token(6) for _ in range(500)]
    assert all(len(token) == 6 and set(token) <= set(TOKEN_ALPHABET) for token in tokens)
    assert {char for token in tokens for char in token} == set(TOKEN_ALPHABET)
    assert rng.token(0) == ""


# ---------------------------------------------------------------------------
# The determinism rule for the data code: no other source of randomness or time
# ---------------------------------------------------------------------------
DATA_CODE = ["accessdesk/rng.py", "accessdesk/config.py", "accessdesk/text.py", "accessdesk/env/world.py",
             "accessdesk/env/tools.py", "accessdesk/env/rules.py", "accessdesk/env/generator.py",
             "accessdesk/env/scenarios.py", "accessdesk/env/verifier.py", "accessdesk/env/reward.py",
             "accessdesk/_second/verifier.py", "accessdesk/_second/reward.py",
             "scripts/build_scenarios.py"]


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize("relative", DATA_CODE)
def test_data_code_has_no_other_source_of_randomness(relative):
    path = ROOT / relative
    modules = _imported_modules(path)
    forbidden = {"time", "datetime", "uuid", "secrets"}
    if relative != "accessdesk/rng.py":
        forbidden.add("random")
    assert not modules & forbidden, (relative, sorted(modules & forbidden))
    source = path.read_text(encoding="utf-8")
    assert "urandom" not in source
    assert re.search(r"(?<![\w.])hash\(", source) is None, "the built-in hash() depends on PYTHONHASHSEED"


def test_the_import_scan_sees_imports():
    assert "random" in _imported_modules(ROOT / "accessdesk/rng.py")
    assert "hashlib" in _imported_modules(ROOT / "accessdesk/rng.py")
