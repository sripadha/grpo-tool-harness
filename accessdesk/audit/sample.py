"""Which attempts a person reads (ANALYSIS_PLAN.md section 8, lines 1 and 3).

    top-reward sample   5 top-reward attempts of each expected outcome, by a seeded draw (20 in all), and
                        EVERY top-reward attempt that the monitor m1, m2 or m3 flags
    reverse sample      3 failed attempts of each expected outcome, by a seeded draw (12 in all)

Words. An attempt is one valid episode record of one arm. Its identity is (arm, scenario id, sample
number). Top reward: it passed with the full reward (`monitors.has_full_reward`). Failed: `passed` is
false. Flagged: m1 `refused_call`, m2 `id_not_from_a_result` or m3 `access_not_checked` found something
(`monitors.CLEAN_MONITORS`). The expected outcome is the `outcome` of the family: grant, approval, already
or inactive. An attempt lost to a harness error is in no sample.

The draw. Each attempt gets a rank: `derive_seed(seed, "audit sample", kind, arm, scenario id, sample
number)`. A draw of k takes the k attempts with the lowest rank. So the draw depends on the seed and on
the identity of the attempts only: not on the order of a file, and not on the order of the arms. An
attempt that is in the draw stays in it when other attempts are added, unless k of them rank lower.

The arms are pooled: 5 per outcome over all arms together, not 5 per arm.

One order, neutral ids. The person who reads must not know which sample an attempt is from. So both
samples stand in ONE list, in a seeded order of its own (`in_one_order`), and each attempt gets a neutral
id: A001, A002, ... The order rank is `derive_seed(seed, "audit sample", "order", arm, scenario id, sample
number)`: the kind of the attempt is not part of it, so neither the place nor the id tells the kind, the
outcome, or whether the attempt was drawn or flagged. Which id is which attempt stands in the key file
only (accessdesk/audit/sheet.py).
"""
from __future__ import annotations

from typing import NamedTuple

from accessdesk.eval import metrics, monitors
from accessdesk.rng import derive_seed
from accessdesk.text import TextBundle
from accessdesk.types import OUTCOMES

TOP_PER_OUTCOME = 5                         # drawn top-reward attempts of each outcome: 20 in all
FAILED_PER_OUTCOME = 3                      # drawn failed attempts of each outcome: 12 in all
FLAG_MONITORS = monitors.CLEAN_MONITORS     # m1, m2, m3: a flagged top-reward attempt is always read
TOP, FAILED = "top_reward", "failed"        # the two kinds of sampled attempt
ORDER = "order"                             # the name of the rank that orders the sheet (no kind is in it)
ID_FORM = "A{:03d}"                         # A001, A002, ...: one row of ids for both kinds
SEED_LABEL = "audit sample"
DRAWN, FLAGGED, BOTH = "drawn", "flagged", "drawn and flagged"


class Attempt(NamedTuple):
    """One valid episode record of one arm, with its family record and what the monitors found."""
    arm: str
    record: dict
    family: dict
    found: dict[str, dict | None]           # monitor name -> its dict, or None (`monitors.flags`)

    @property
    def key(self) -> tuple[str, str, int]:
        """The identity of the attempt: (arm, scenario id, sample number)."""
        return (self.arm, self.record["scenario_id"], self.record["sample"])


class Picked(NamedTuple):
    """One attempt that a sample holds: its kind, and why it is there. It has no id yet."""
    kind: str                               # TOP or FAILED
    why: str                                # DRAWN, FLAGGED or BOTH (a failed attempt is always DRAWN)
    attempt: Attempt


class Sampled(NamedTuple):
    """One attempt of the sheet: its neutral id, and what `Picked` holds. The sheet and the form show the
    id only; the kind and the reason go into the key file."""
    id: str                                 # "A007"
    kind: str                               # TOP or FAILED
    why: str                                # DRAWN, FLAGGED or BOTH
    attempt: Attempt


def attempts_of(arm: str, records: list[dict], families_by_id: dict[str, dict],
                text: TextBundle) -> list[Attempt]:
    """The attempts of one arm: one per valid record, in the given order. A harness error is left out.

    Raises ValueError for what `monitors.flags` refuses: a record of another text file, or a family record
    for which the verifier does not give the saved pass or fail."""
    return [Attempt(arm, record, families_by_id[record["family_id"]],
                    monitors.flags(record, families_by_id[record["family_id"]], text))
            for record in metrics.valid_records(records)]


def is_top_reward(attempt: Attempt) -> bool:
    """True if the attempt passed with the full reward of its run."""
    return monitors.has_full_reward(attempt.record)


def is_failed(attempt: Attempt) -> bool:
    """True if the verifier failed the attempt."""
    return not attempt.record["passed"]


def flag_names(attempt: Attempt) -> list[str]:
    """The monitors of FLAG_MONITORS that flag the attempt, in that order."""
    return [name for name in FLAG_MONITORS if attempt.found[name] is not None]


def rank(seed: int, kind: str, key: tuple[str, str, int]) -> int:
    """The rank of an attempt in the draw of `kind`. It depends on the seed, the kind and the identity."""
    return derive_seed(seed, SEED_LABEL, kind, *key)


def draw(attempts: list[Attempt], k: int, seed: int, kind: str) -> list[Attempt]:
    """The k attempts with the lowest rank, in rank order; all of them if there are fewer than k.

    The result does not depend on the order of `attempts`. Raises ValueError if two attempts have one
    identity: a draw over them would not be defined by identity."""
    keys = [attempt.key for attempt in attempts]
    if len(set(keys)) != len(keys):
        twice = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"{len(twice)} attempts are there more than once (arm, scenario id, sample "
                         f"number): {twice[:3]}")
    return sorted(attempts, key=lambda attempt: (rank(seed, kind, attempt.key), attempt.key))[:k]


def _of_outcome(attempts: list[Attempt], outcome: str) -> list[Attempt]:
    return [attempt for attempt in attempts if attempt.record["outcome"] == outcome]


def in_one_order(picked: list[Picked], seed: int) -> list[Sampled]:
    """The picked attempts of both samples as ONE list in a seeded order, with the ids A001, A002, ...

    The order has a rank of its own (`rank` with ORDER in the place of the kind), made from the seed and
    the identity of the attempt. The kind is not in it. So the place and the id of an attempt tell
    neither its kind nor its outcome, nor whether it was drawn or flagged: by the rank of the draw, the
    attempts that are only flagged would nearly always come last."""
    ordered = sorted(picked, key=lambda item: (rank(seed, ORDER, item.attempt.key), item.attempt.key))
    return [Sampled(ID_FORM.format(position), item.kind, item.why, item.attempt)
            for position, item in enumerate(ordered, start=1)]


def top_reward_sample(attempts: list[Attempt], seed: int,
                      per_outcome: int = TOP_PER_OUTCOME) -> list[Picked]:
    """The top-reward sample: `per_outcome` drawn top-reward attempts of each outcome, and every flagged
    top-reward attempt. An attempt is in the list once; `why` says if it was drawn, flagged, or both.
    The list is in the order of `attempts`; `in_one_order` gives the order of the sheet."""
    top = [attempt for attempt in attempts if is_top_reward(attempt)]
    drawn = {attempt.key for outcome in OUTCOMES
             for attempt in draw(_of_outcome(top, outcome), per_outcome, seed, TOP)}
    chosen = []
    for attempt in top:
        flagged = bool(flag_names(attempt))
        if attempt.key in drawn:
            chosen.append(Picked(TOP, BOTH if flagged else DRAWN, attempt))
        elif flagged:
            chosen.append(Picked(TOP, FLAGGED, attempt))
    return chosen


def failed_sample(attempts: list[Attempt], seed: int, per_outcome: int = FAILED_PER_OUTCOME) -> list[Picked]:
    """The reverse sample: `per_outcome` drawn failed attempts of each outcome."""
    failed = [attempt for attempt in attempts if is_failed(attempt)]
    return [Picked(FAILED, DRAWN, attempt) for outcome in OUTCOMES
            for attempt in draw(_of_outcome(failed, outcome), per_outcome, seed, FAILED)]


def population(attempts: list[Attempt]) -> dict:
    """What the samples are drawn from, per outcome: the numbers of attempts, of top-reward attempts, of
    flagged top-reward attempts and of failed attempts."""
    counts = {}
    for outcome in OUTCOMES:
        group = _of_outcome(attempts, outcome)
        top = [attempt for attempt in group if is_top_reward(attempt)]
        counts[outcome] = {"attempts": len(group), "top_reward": len(top),
                           "top_reward_flagged": sum(1 for attempt in top if flag_names(attempt)),
                           "failed": sum(1 for attempt in group if is_failed(attempt))}
    return counts


def choose(attempts: list[Attempt], seed: int, top_per_outcome: int = TOP_PER_OUTCOME,
           failed_per_outcome: int = FAILED_PER_OUTCOME) -> list[Sampled]:
    """Both samples as one list in one seeded order, under the ids A001, A002, ... (`in_one_order`).

    Raises ValueError if an attempt has an outcome that is not one of the four: it would be in no draw."""
    unknown = sorted({attempt.record["outcome"] for attempt in attempts} - set(OUTCOMES))
    if unknown:
        raise ValueError(f"attempts of the outcomes {unknown} are in no draw; known: {OUTCOMES}")
    return in_one_order(top_reward_sample(attempts, seed, top_per_outcome)
                        + failed_sample(attempts, seed, failed_per_outcome), seed)
