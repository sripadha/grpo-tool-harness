"""The only source of randomness in the repository's data code.

Everything is built on `random.Random(seed).random()` alone. That call gives the same floats for the same
integer seed on every Python 3 version, so the generated scenarios do not depend on the Python version, on the
platform or on PYTHONHASHSEED.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

T = TypeVar("T")

# Letters and digits that are hard to confuse: no i, l, o, 0, 1. 31 characters.
TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def derive_seed(*parts: object) -> int:
    """An integer seed from the first 8 bytes of the SHA-256 of "|".join(str(p) for p in parts)."""
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class Rng:
    """Deterministic and the same on every Python version: built only on random.Random(seed).random()."""

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, not {type(seed).__name__}")
        self._random = random.Random(seed).random

    def below(self, n: int) -> int:
        """A uniform integer in [0, n). n must be at least 1."""
        if n < 1:
            raise ValueError(f"below(n) needs n >= 1, not {n}")
        return min(int(self._random() * n), n - 1)

    def chance(self, p: float) -> bool:
        """True with probability p. One draw is always used, also for p = 0 and p = 1."""
        return self._random() < p

    def choice(self, seq: Sequence[T]) -> T:
        """One element of a non-empty sequence."""
        return seq[self.below(len(seq))]

    def shuffle(self, seq: Sequence[T]) -> list[T]:
        """A new list with the elements of `seq` in random order (Fisher-Yates). `seq` is not changed."""
        items = list(seq)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        """k different positions of `seq`, in random order."""
        if not 0 <= k <= len(seq):
            raise ValueError(f"sample(k) needs 0 <= k <= {len(seq)}, not {k}")
        return self.shuffle(seq)[:k]

    def token(self, k: int) -> str:
        """A string of k characters from TOKEN_ALPHABET."""
        return "".join(self.choice(TOKEN_ALPHABET) for _ in range(k))
