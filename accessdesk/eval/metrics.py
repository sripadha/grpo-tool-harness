"""Every number of the evaluation, computed from saved episode records.

Rules that hold for every function in this file:
- The input is a list of episode records (the JSON lines that the harness saves). No function reads a file.
- The caller passes the records of ONE arm: one model under one prompt, one rule set and one tool set.
  `accessdesk/eval/tables.py` checks this before it calls a function here.
- A record whose `harness_error` is not null is left out of every number. `harness_error_count` counts them.
- The unit of analysis is the family, unless a docstring names another unit. A family's pass rate is the mean
  of `passed` over all its records (all its wordings and samples). The pass rate of a set of records is the
  mean of the family pass rates, so every family has the same weight.
- A scenario is one family in one wording style: the pair (family_id, style).
- Results do not depend on the order of the records. The keys of a returned dictionary are in a fixed order.
"""
from __future__ import annotations

import math

import numpy as np

from accessdesk.types import OUTCOMES

HELD_OUT_STYLE = 4              # the wording style that no training scenario uses
PASS_RATE_KEYS = ("outcome", "style", "seen_held_out")
OUTCOME_LABELS = ("closed_correctly", "closed_wrongly", "never_closed", "malformed_only")
MDE_STANDARD_ERRORS = 2.8       # 1.96 (5% two-sided test) + 0.84 (80% power)
MDE_SPLITS = 200                # random half-and-half splits that two_halves_mde averages over


# ---------------------------------------------------------------------------
# Selecting and grouping records
# ---------------------------------------------------------------------------
def valid_records(records: list[dict]) -> list[dict]:
    """The records without a harness error, in the given order."""
    return [r for r in records if r["harness_error"] is None]


def harness_error_count(records: list[dict]) -> int:
    """Number of records with a harness error (episodes that are in no other number)."""
    return sum(1 for r in records if r["harness_error"] is not None)


def _mean(values: list) -> float:
    """Arithmetic mean of a list that is not empty (booleans count as 0 and 1)."""
    if len(values) == 0:
        raise ValueError("nothing to average: no valid record, or no scenario with enough samples")
    return float(sum(values)) / len(values)


def _by_family(records: list[dict]) -> dict[str, list[dict]]:
    """family_id -> its valid records. Families in sorted order."""
    groups: dict[str, list[dict]] = {}
    for r in valid_records(records):
        groups.setdefault(r["family_id"], []).append(r)
    return {family_id: groups[family_id] for family_id in sorted(groups)}


def _by_scenario(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """(family_id, style) -> its valid records, ordered by sample index. Scenarios in sorted order."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in valid_records(records):
        groups.setdefault((r["family_id"], r["style"]), []).append(r)
    return {key: sorted(groups[key], key=lambda r: r["sample"]) for key in sorted(groups)}


def _family_weighted_mean(scenario_values: dict[tuple[str, int], float]) -> float:
    """Mean over families of (the mean of the family's scenario values)."""
    per_family: dict[str, list[float]] = {}
    for (family_id, _style), value in scenario_values.items():
        per_family.setdefault(family_id, []).append(value)
    return _mean([_mean(values) for values in per_family.values()])


# ---------------------------------------------------------------------------
# Pass rates
# ---------------------------------------------------------------------------
def family_pass_rates(records: list[dict]) -> dict[str, float]:
    """For each family: the share of its episodes that passed, over all its wordings and samples.

    Unit: one number per family. A family with no valid record is absent."""
    return {family_id: _mean([r["passed"] for r in group])
            for family_id, group in _by_family(records).items()}


def pass_rate(records: list[dict]) -> float:
    """The mean of the family pass rates: every family has the same weight, however many records it has.

    Unit: the family. This is the main number of the evaluation. Raises ValueError if there is no valid
    record."""
    return _mean(list(family_pass_rates(records).values()))


def pass_rate_by(records: list[dict], key: str) -> dict:
    """The pass rate (family-weighted, as in `pass_rate`) of each part of the records.

    key "outcome": one entry per outcome (grant, approval, already, inactive). key "style": one entry per
    wording style number. key "seen_held_out": "seen" for the styles 1 to 3 together and "held_out" for
    style 4. Unit: the family, inside each part. A part with no valid record is absent."""
    if key not in PASS_RATE_KEYS:
        raise ValueError(f"key must be one of {PASS_RATE_KEYS}, not {key!r}")
    parts: dict = {}
    for r in valid_records(records):
        if key == "seen_held_out":
            part = "held_out" if r["style"] == HELD_OUT_STYLE else "seen"
        else:
            part = r[key]
        parts.setdefault(part, []).append(r)
    if key == "outcome":
        order = [o for o in OUTCOMES if o in parts] + sorted(p for p in parts if p not in OUTCOMES)
    elif key == "style":
        order = sorted(parts)
    else:
        order = [p for p in ("seen", "held_out") if p in parts]
    return {part: pass_rate(parts[part]) for part in order}


def reward_distribution(records: list[dict]) -> dict[int, int]:
    """How many episodes got each reward value (whole hundredths: 0 to 100, and down to -15 for a failing
    episode when the switch `penalise_malformed_on_fail` is on). Unit: the episode. Keys in rising order."""
    counts: dict[int, int] = {}
    for r in valid_records(records):
        counts[r["reward"]] = counts.get(r["reward"], 0) + 1
    return {reward: counts[reward] for reward in sorted(counts)}


# ---------------------------------------------------------------------------
# Groups without a learning signal
# ---------------------------------------------------------------------------
def zero_variance_share(records: list[dict]) -> dict:
    """The share of scenarios whose samples all got the same result. Such a group teaches GRPO nothing.

    Unit: the scenario (not the family). Only scenarios with at least 2 valid samples are counted.
        n_scenarios             number of scenarios counted
        equal_reward            share in which every sample has exactly the same reward
        equal_reward_all_pass   share in which every sample has the same reward and every sample passed
        equal_reward_all_fail   share in which every sample has the same reward and every sample failed
        equal_passfail          share in which every sample passed or every sample failed (rewards may differ
                                by penalties, for example 100 and 95; with the switch
                                `penalise_malformed_on_fail` on also among fails, for example 0 and -5)
    equal_reward = equal_reward_all_pass + equal_reward_all_fail, because a fail scores 0 or less and a pass
    scores more than 0: equal rewards with a mix of pass and fail cannot happen, and ValueError is raised if
    they do.
    If no scenario has 2 valid samples, the four shares are None."""
    n_scenarios = n_equal_pass = n_equal_fail = n_equal_passfail = 0
    for key, group in _by_scenario(records).items():
        if len(group) < 2:
            continue
        n_scenarios += 1
        n_passed = sum(1 for r in group if r["passed"])
        all_pass = n_passed == len(group)
        all_fail = n_passed == 0
        same_reward = all(r["reward"] == group[0]["reward"] for r in group)
        if same_reward and not (all_pass or all_fail):
            raise ValueError(f"scenario {key}: equal rewards for a mix of passed and failed episodes")
        if all_pass or all_fail:
            n_equal_passfail += 1
        if same_reward and all_pass:
            n_equal_pass += 1
        if same_reward and all_fail:
            n_equal_fail += 1
    if n_scenarios == 0:
        return {"n_scenarios": 0, "equal_reward": None, "equal_reward_all_pass": None,
                "equal_reward_all_fail": None, "equal_passfail": None}
    return {"n_scenarios": n_scenarios,
            "equal_reward": (n_equal_pass + n_equal_fail) / n_scenarios,
            "equal_reward_all_pass": n_equal_pass / n_scenarios,
            "equal_reward_all_fail": n_equal_fail / n_scenarios,
            "equal_passfail": n_equal_passfail / n_scenarios}


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------
def invalid_call_rate(records: list[dict]) -> float:
    """Malformed assistant messages divided by all assistant messages, pooled over all episodes.

    Unit: the assistant message (one turn). It uses the counters `n_malformed` and `n_turns` of each record,
    the same counters that the reward uses. Raises ValueError if there is no assistant message."""
    valid = valid_records(records)
    n_messages = sum(r["n_turns"] for r in valid)
    if n_messages == 0:
        raise ValueError("no assistant message in the valid records")
    return sum(r["n_malformed"] for r in valid) / n_messages


def turn_stats(records: list[dict]) -> dict:
    """Turns per episode, and how episodes ended. Unit: the episode.

        n_episodes            number of valid episodes
        mean_turns            mean number of assistant messages per episode
        share_at_turn_limit   share of episodes that ended because the turn limit was used up without a close
                              (`ended_by` is "turn_limit"); an episode that closes on its last allowed turn
                              is not in this share
        share_closed          share of episodes that ended with a successful close_request"""
    valid = valid_records(records)
    return {"n_episodes": len(valid),
            "mean_turns": _mean([r["n_turns"] for r in valid]),
            "share_at_turn_limit": _mean([r["ended_by"] == "turn_limit" for r in valid]),
            "share_closed": _mean([bool(r["closed"]) for r in valid])}


def episode_label(record: dict) -> str:
    """Exactly one label for an episode.

        closed_correctly   the request was closed and the episode passed
        closed_wrongly     the request was closed and the episode failed
        malformed_only     the request was never closed and every assistant message was malformed
        never_closed       the request was never closed and at least one message was a well-formed call"""
    if record["closed"]:
        return "closed_correctly" if record["passed"] else "closed_wrongly"
    if record["n_malformed"] == record["n_turns"]:
        return "malformed_only"
    return "never_closed"


def outcome_labels(records: list[dict]) -> dict[str, float]:
    """The share of episodes with each label of `episode_label`. Unit: the episode. The shares add to 1."""
    labels = [episode_label(r) for r in valid_records(records)]
    return {label: _mean([found == label for found in labels]) for label in OUTCOME_LABELS}


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
def pass_at_k(records: list[dict], k: int = 8) -> float:
    """Pass@k: how often at least one of k attempts at a scenario passes.

    For each scenario: 1 if at least one of its first k valid samples (lowest sample index first) passed,
    else 0. Then the mean over the scenarios of a family, then the mean over families. Unit: the family.
    A scenario with fewer than k valid samples is left out (`n_scenarios_below` counts them). Raises
    ValueError if every scenario is left out."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    values = {key: float(any(r["passed"] for r in group[:k]))
              for key, group in _by_scenario(records).items() if len(group) >= k}
    return _family_weighted_mean(values)


def pass_hat_k(records: list[dict], k: int) -> float:
    """Pass^k: the chance that k independent attempts at a scenario ALL pass.

    For each scenario with n valid samples of which c passed: C(c, k) / C(n, k), the share of the k-subsets of
    its samples in which every sample passed. This is the unbiased estimate of p**k (the plain (c / n)**k is
    too high). Then the mean over the scenarios of a family, then the mean over families. Unit: the family.
    For k = 1 this is the pass rate. A scenario with fewer than k valid samples is left out
    (`n_scenarios_below` counts them). Raises ValueError if every scenario is left out."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    values = {}
    for key, group in _by_scenario(records).items():
        if len(group) >= k:
            n_passed = sum(1 for r in group if r["passed"])
            values[key] = math.comb(n_passed, k) / math.comb(len(group), k)
    return _family_weighted_mean(values)


def n_scenarios_below(records: list[dict], k: int) -> int:
    """Number of scenarios with fewer than k valid samples: the ones that pass_at_k and pass_hat_k leave out.
    A scenario with no valid sample at all is not seen here."""
    return sum(1 for group in _by_scenario(records).values() if len(group) < k)


def all_wordings_pass(records: list[dict]) -> float:
    """The chance that a family passes in every one of its wordings, with one attempt per wording.

    For each family: the product, over its wording styles, of the style's pass rate (the mean of `passed` over
    the style's samples). Then the mean over families. Unit: the family."""
    products: dict[str, float] = {}
    for (family_id, _style), group in _by_scenario(records).items():
        products[family_id] = products.get(family_id, 1.0) * _mean([r["passed"] for r in group])
    return _mean(list(products.values()))


def variance_shares(records: list[dict]) -> dict[str, float]:
    """How the variation of pass or fail INSIDE families splits between wordings and repeated sampling.

    For each family, with x = 1 for a pass and 0 for a fail:
        between wordings  sum over its styles of (number of samples) * (style mean - family mean)**2
        within wordings   sum over its samples of (x - mean of the sample's style)**2
    Both sums are added up over all families. "wording" is the between sum divided by the total, "sampling" is
    the within sum divided by the total. They add to 1; if the total is 0 (no family has both a pass and a
    fail) both are 0. Differences BETWEEN families are in neither part. Unit: the episode inside its family.

    Read "wording" against "wording_if_sampling_only": these are raw sums of squares, so sampling noise alone
    also makes the style means differ. The third key is the share to expect if wording had no effect at all:
    sum over families of (styles - 1), divided by the sum over families of (samples - 1). It is 3/31 for
    4 styles of 8 samples. (Exact as a ratio of expected sums when every family has the same design.)"""
    between = within = 0.0
    n_between = n_total = 0
    for family_id, group in _by_family(records).items():
        family_mean = _mean([r["passed"] for r in group])
        styles = sorted({r["style"] for r in group})
        for style in styles:
            x = [float(r["passed"]) for r in group if r["style"] == style]
            style_mean = _mean(x)
            between += len(x) * (style_mean - family_mean) ** 2
            within += sum((value - style_mean) ** 2 for value in x)
        n_between += len(styles) - 1
        n_total += len(group) - 1
    total = between + within
    by_chance = n_between / n_total if n_total > 0 else 0.0
    if total == 0.0:
        return {"wording": 0.0, "sampling": 0.0, "wording_if_sampling_only": by_chance}
    return {"wording": between / total, "sampling": within / total, "wording_if_sampling_only": by_chance}


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def two_halves_mde(records: list[dict], seed: int = 0, n_splits: int = MDE_SPLITS) -> dict:
    """The smallest difference between two models that this evaluation can detect, from one model's records.

    One split: the samples of each scenario are split at random into two halves of equal size; with an odd
    number of samples one is left out. Each half gives a pass rate per family. The two halves are the same
    model, so their difference per family is sampling noise only. The squared standard error of the mean
    per-family difference is: variance of the differences (n - 1 in the divisor) / n_families.
    One split alone is a noisy estimate: on 24 families it moved by a factor of more than two with the seed
    of the split. So the squared standard error is averaged over `n_splits` splits, all drawn from `seed`.
        n_families   number of families used
        n_splits     number of random splits that were averaged
        se_halves    square root of the mean, over the splits, of the squared standard error of the mean
                     per-family difference between the halves
        se           se_halves / sqrt(2): a real comparison has twice as many samples per model as one half
        mde          2.8 * se: the difference that a 5% two-sided test finds 8 times out of 10
        mde_lowest_split, mde_highest_split   the lowest and the highest value that one split alone gives
    Unit: the family. This counts sampling noise only: when the gain differs between families the real
    interval is wider. With n_splits = 1 this is the number of one split. Raises ValueError with fewer than
    2 families, or with n_splits below 1."""
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}")
    groups = [(family_id, [bool(r["passed"]) for r in group])
              for (family_id, _style), group in _by_scenario(records).items() if len(group) >= 2]
    families = sorted({family_id for family_id, _passed in groups})
    if len(families) < 2:
        raise ValueError("two_halves_mde needs at least 2 families that have a scenario with 2 valid samples")
    rng = np.random.default_rng(seed)
    squared = np.array([_squared_se_of_one_split(groups, families, rng) for _ in range(n_splits)])
    se_halves = math.sqrt(float(squared.mean()))
    se = se_halves / math.sqrt(2.0)
    scale = MDE_STANDARD_ERRORS / math.sqrt(2.0)
    return {"n_families": len(families), "n_splits": n_splits, "se_halves": se_halves, "se": se,
            "mde": MDE_STANDARD_ERRORS * se,
            "mde_lowest_split": scale * math.sqrt(float(squared.min())),
            "mde_highest_split": scale * math.sqrt(float(squared.max()))}


def _squared_se_of_one_split(groups: list[tuple[str, list[bool]]], families: list[str],
                             rng: np.random.Generator) -> float:
    """One random split of every scenario's samples into two halves: the squared standard error of the mean
    per-family difference of the two halves' pass rates. `groups`: (family_id, pass or fail per sample)."""
    halves: tuple[dict[str, list], dict[str, list]] = ({}, {})
    for family_id, passed in groups:
        half = len(passed) // 2
        order = rng.permutation(len(passed))
        for which, positions in ((0, order[:half]), (1, order[half:2 * half])):
            halves[which].setdefault(family_id, []).extend(passed[int(i)] for i in positions)
    diffs = np.array([_mean(halves[0][f]) - _mean(halves[1][f]) for f in families], dtype=float)
    return float(diffs.var(ddof=1)) / diffs.size


def family_bootstrap_ci(values, n_draws: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile interval of one mean, by resampling families.

    values: one number per family (for example the family pass rates of one model). Each draw takes
    len(values) families with replacement and their mean. Returns {"mean", "lo", "hi"}; lo and hi are the
    (1 - level) / 2 and 1 - (1 - level) / 2 quantiles of the draws. Unit: the family.
    Uses np.random.default_rng(seed)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"values must be a 1-D array with at least one family; got shape {values.shape}")
    if n_draws < 1 or not 0.0 < level < 1.0:
        raise ValueError(f"need n_draws >= 1 and 0 < level < 1; got {n_draws} and {level}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_draws, values.size))
    draws = values[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def _t_cdf(t: float, df: float, n_intervals: int = 20_000) -> float:
    """P(T <= t) for Student's t with df degrees of freedom, t >= 0: 0.5 + Simpson's rule on the density."""
    x = np.linspace(0.0, t, n_intervals + 1)
    log_const = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    density = np.exp(log_const - (df + 1.0) / 2.0 * np.log1p(x * x / df))
    step = t / n_intervals
    inner = 4.0 * density[1:-1:2].sum() + 2.0 * density[2:-1:2].sum()
    return 0.5 + step / 3.0 * float(density[0] + density[-1] + inner)


def t_quantile(p: float, df: float) -> float:
    """The value t with P(T <= t) = p for Student's t with df degrees of freedom; 0.5 < p < 1.

    Found by bisection on the numerically integrated density. No table and no scipy."""
    if not 0.5 < p < 1.0 or not df > 0:
        raise ValueError(f"need 0.5 < p < 1 and df > 0; got p={p}, df={df}")
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        lo, hi = hi, 2.0 * hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def t_interval(diffs, level: float = 0.95) -> dict:
    """The classical t-interval of the mean of per-family differences.

    diffs: one difference per family. Returns {"mean", "lo", "hi", "se", "df"} with
    se = standard deviation (n - 1 in the divisor) / sqrt(n), df = n - 1, and
    lo, hi = mean -/+ t_quantile(1 - (1 - level) / 2, df) * se. Unit: the family. Needs 2 families or more."""
    diffs = np.asarray(diffs, dtype=float)
    if diffs.ndim != 1 or diffs.size < 2:
        raise ValueError(f"diffs must be a 1-D array with at least 2 families; got shape {diffs.shape}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level}")
    df = int(diffs.size) - 1
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1)) / math.sqrt(diffs.size)
    half_width = t_quantile(1.0 - (1.0 - level) / 2.0, df) * se
    return {"mean": mean, "lo": mean - half_width, "hi": mean + half_width, "se": se, "df": df}
