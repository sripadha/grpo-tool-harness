#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rederive.py - a second, separately written calculation of the two dev result tables.

Reads (and nothing else):
  results/main/base_dev/episodes.jsonl
  results/main/sft_dev/episodes.jsonl
  results/main/grpo_seed1_dev/episodes.jsonl
  results/main/grpo_seed2_dev/episodes.jsonl
  data/scenarios/dev.jsonl
  results/main/tables/base_sft_grpo_seed1.json and .md      (only to compare)
  results/main/tables/base_sft_grpo_seed2.json and .md      (only to compare)

Writes (only inside its own folder, or into the folder given by --out):
  report.md          the comparison, every number filled in by this script
  my_numbers.json    every number that this script computed

It imports nothing from the package accessdesk and copies no code from it.
Standard library and numpy only. The meaning of each metric is taken from the words
of ANALYSIS_PLAN.md sections 4, 5 and 6 and from the column names of the .md tables.
The files accessdesk/eval/metrics.py, bootstrap.py and tables.py were not opened.

What it does, in order:
  1. reads the 48 dev families and the 4 x 1536 saved attempts, and checks that the design is complete;
  2. computes every per-arm metric as an exact fraction where the metric is a ratio of counts;
  3. bootstrap intervals with my own seed, the exact bootstrap distribution (no random draws),
     the spread of the interval ends over 200 further seeds, and the t-intervals;
  4. the same intervals and the two-halves standard error once more with the seed that the tables
     name (0), to see whether the table's numbers come out exactly;
  5. sign test, sign-flip permutation test, leave-one-family-out (the questions b to e);
  6. compares every entry of the two .json tables and every printed number of the two .md tables
     with these values, and writes report.md and my_numbers.json.

Beside it in this folder: seed0_trials.py (which way of drawing with seed 0 gives the table) and
selftest_compare.py (the comparison run against copies of the tables with errors planted on purpose).

Run from the top folder of the repository (about 15 seconds):
      .venv/bin/python results/main/audit/recount/rederive.py [--out DIR]

Moved on 2026-11-19 from the working folder of the build into results/main/audit/recount/; changed then:
the top folder of the repository (four folders up, not three), the option --out, the commands and the
script's path in the report, and the folder in the last line printed (relative to the current folder).

Changed on 2026-11-19 (after the test results were opened), and nothing else: the option --tables DIR
(compare against copies of the two table files in DIR) and the exit code (1 when any compared number
differs, an input check fails or a part of the .md is not covered; then one line goes to standard error;
0 otherwise). The printed lines and the files written by a clean run are unchanged.
The test table and the dev table base_vs_sft: rederive_table.py beside this script.
"""

import collections
import hashlib
import json
import math
import os
import re
import sys
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

ARMS = ["base", "sft", "grpo_seed1", "grpo_seed2"]
EPISODES = {a: os.path.join(REPO, "results", "main", a + "_dev", "episodes.jsonl") for a in ARMS}
DEV = os.path.join(REPO, "data", "scenarios", "dev.jsonl")
TABLES = {
    "seed1": os.path.join(REPO, "results", "main", "tables", "base_sft_grpo_seed1"),
    "seed2": os.path.join(REPO, "results", "main", "tables", "base_sft_grpo_seed2"),
}
TABLE_ARMS = {"seed1": ["base", "sft", "grpo_seed1"], "seed2": ["base", "sft", "grpo_seed2"]}
# (a, b) means "b minus a", the order in which the tables store a pair
PAIRS = [("base", "sft"), ("sft", "grpo_seed1"), ("base", "grpo_seed1"),
         ("sft", "grpo_seed2"), ("base", "grpo_seed2")]
OUTCOMES = ["grant", "approval", "already", "inactive"]
SEEN_STYLES = (1, 2, 3)
HELD_OUT_STYLE = 4
KS = (1, 2, 4, 8)

LEVEL = 0.95
N_DRAWS = 10000
MY_SEED = 7919                      # my own seed for the bootstrap (the tables used seed 0)
NOISE_SEEDS = list(range(1000, 1200))   # 200 further seeds, only to measure Monte Carlo noise
SPLIT_SEED = 7920                   # my own seed for the two-halves splits (extra)
PERM_SEED = 7921                    # my own seed for the Monte Carlo cross-check of the sign-flip test
MANY_SPLITS_SEED = 7922             # my own seed for 20,000 two-halves splits (does the pooled value settle?)

TABLE_SEED = 0                      # the seed that the tables name ("seed 0"); used only for the seed-0 check
N_SPLITS = 200                      # the number of two-halves splits that the tables name
TOL_JSON = 1e-12                    # absolute tolerance against a .json number
Z95 = 1.959963984540054             # normal quantile for a two-sided 95% band
Z999 = 3.290526731491926            # normal quantile for a two-sided 99.9% band


# ----------------------------------------------------------------------------------------------
# 1. Reading
# ----------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_families(path):
    fams, order = {}, []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fid = d["family_id"]
            if fid in fams:
                raise SystemExit("family twice in dev.jsonl: %s" % fid)
            fams[fid] = {
                "outcome": d["outcome"],
                "homonym": d["homonym"],
                "facts": d["facts"],
                "styles": sorted(w["style"] for w in d["wordings"]),
                "expected_default": d["expected"]["default"],
                "turn_budget_default": d["turn_budget"]["default"],
            }
            order.append(fid)
    return fams, order


def same_records(a, b, as_set):
    ta = [tuple(x) for x in a]
    tb = [tuple(x) for x in b]
    if as_set:
        return set(ta) == set(tb)
    return sorted(ta) == sorted(tb)


def pass_by_the_words(final_state, expected, as_set):
    """Section 6 of the plan, in my own code: 'the final state has the expected roles, approval
    records and grant records, and the request is closed with the right resolution for the right
    person and role'. as_set=True ignores a record that is there twice; False does not."""
    if not isinstance(final_state, dict):
        return None
    for key in ("access", "approvals", "grants"):
        if not same_records(final_state[key], expected[key], as_set):
            return False
    req, exp = final_state["request"], expected["request"]
    if not isinstance(req, dict):
        return False
    return all(req.get(k) == exp.get(k) for k in ("resolution", "person_id", "role_id")) and \
        exp.get("resolution") is not None


def read_attempts(path, fams):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            turns = r["turns"] or []
            fam = fams.get(r["family_id"])
            exp = fam["expected_default"] if fam else None
            out.append({
                "arm": r["arm"], "split": r["split"],
                "family": r["family_id"], "style": r["style"], "sample": r["sample"],
                "scenario": r["scenario_id"], "outcome": r["outcome"],
                "passed": r["passed"], "reward": r["reward"],
                "n_msgs": len(turns),                                            # assistant messages
                "n_bad": sum(1 for t in turns if t.get("parse_ok") is not True),  # malformed ones
                "n_turns_field": r["n_turns"], "n_malformed_field": r["n_malformed"],
                "closed": r["closed"], "ended_by": r["ended_by"],
                "harness_error": r["harness_error"],
                "turn_budget": r["turn_budget"],
                "ruleset": r["ruleset"], "toolset": r["toolset"], "prompt": r["prompt"],
                "text_sha256": r["text_sha256"], "text_is_placeholder": r["text_is_placeholder"],
                "config_sha256": r["config_sha256"],
                "data_sha256": r.get("data_sha256"),
                "settings": json.dumps(r["settings"], sort_keys=True),
                "doubled_record": isinstance(r["final_state"], dict) and any(
                    len(r["final_state"][k]) != len(set(tuple(x) for x in r["final_state"][k])) for k in ("approvals", "grants")),
                "pass_words_multiset": pass_by_the_words(r["final_state"], exp, False) if exp else None,
                "pass_words_set": pass_by_the_words(r["final_state"], exp, True) if exp else None,
            })
            del r
    return out


# ----------------------------------------------------------------------------------------------
# 2. Small helpers
# ----------------------------------------------------------------------------------------------

def mean_fr(values):
    values = list(values)
    return sum(values, Fr(0)) / len(values)


def family_then_mean(scenario_value):
    """scenario_value: {(family, style): Fraction}. Mean inside each family, then over families."""
    per = collections.defaultdict(list)
    for (fam, _style), v in scenario_value.items():
        per[fam].append(v)
    return mean_fr(mean_fr(v) for v in per.values())


def fr_json(x):
    return {"fraction": "%d/%d" % (x.numerator, x.denominator), "value": float(x)}


# ----------------------------------------------------------------------------------------------
# 3. Per-arm metrics (exact fractions wherever the metric is a count ratio)
# ----------------------------------------------------------------------------------------------

def arm_metrics(recs, fams, fam_order):
    valid = [r for r in recs if r["harness_error"] is None]
    m = {"n_records": len(recs), "n_harness_errors": len(recs) - len(valid), "n_episodes": len(valid)}

    fam_pass = collections.Counter()
    fam_n = collections.Counter()
    scen_pass = collections.Counter()
    scen_n = collections.Counter()
    scen_rewards = collections.defaultdict(list)
    for r in valid:
        f, s = r["family"], r["style"]
        fam_n[f] += 1
        scen_n[(f, s)] += 1
        scen_rewards[(f, s)].append(r["reward"])
        if r["passed"]:
            fam_pass[f] += 1
            scen_pass[(f, s)] += 1
    families = [f for f in fam_order if fam_n[f] > 0]
    scenarios = sorted(scen_n)
    m["families"] = families
    m["n_families"] = len(families)
    m["n_scenarios"] = len(scenarios)
    m["fam_pass"] = {f: fam_pass[f] for f in families}
    m["fam_n"] = {f: fam_n[f] for f in families}
    m["scen_pass"] = {k: scen_pass[k] for k in scenarios}
    m["scen_n"] = {k: scen_n[k] for k in scenarios}

    # pass rate: the share of passed attempts inside each family, then the mean over families
    fam_rate = {f: Fr(fam_pass[f], fam_n[f]) for f in families}
    m["fam_rate"] = fam_rate
    m["pass_rate"] = mean_fr(fam_rate.values())
    # (a) the plain share of passed attempts, for the question whether the two are the same
    m["plain_pass_share"] = Fr(sum(fam_pass.values()), len(valid))
    m["n_passed"] = sum(fam_pass.values())

    # per expected outcome (the outcome of the family in dev.jsonl)
    m["pass_rate_by_outcome"] = {}
    m["by_outcome_counts"] = {}
    for o in OUTCOMES:
        fs = [f for f in families if fams[f]["outcome"] == o]
        m["pass_rate_by_outcome"][o] = mean_fr(fam_rate[f] for f in fs)
        m["by_outcome_counts"][o] = (sum(fam_pass[f] for f in fs), sum(fam_n[f] for f in fs), len(fs))

    # styles 1 to 3, and style 4: again per family first, then the mean over families
    def styles_rate(styles):
        rates, c_all, n_all = [], 0, 0
        for f in families:
            c = sum(scen_pass[(f, s)] for s in styles)
            n = sum(scen_n[(f, s)] for s in styles)
            if n:
                rates.append(Fr(c, n))
                c_all += c
                n_all += n
        return mean_fr(rates), (c_all, n_all)
    m["pass_rate_seen"], m["seen_counts"] = styles_rate(SEEN_STYLES)
    m["pass_rate_held_out"], m["held_out_counts"] = styles_rate((HELD_OUT_STYLE,))

    # scenarios whose attempts all have exactly the same reward; all pass or all fail
    n_eq_reward = sum(1 for k in scenarios if len(set(scen_rewards[k])) == 1)
    n_all_pass = sum(1 for k in scenarios if scen_pass[k] == scen_n[k])
    n_all_fail = sum(1 for k in scenarios if scen_pass[k] == 0)
    m["n_scen_equal_reward"] = n_eq_reward
    m["n_scen_all_pass"] = n_all_pass
    m["n_scen_all_fail"] = n_all_fail
    m["zero_variance_equal_reward"] = Fr(n_eq_reward, len(scenarios))
    m["zero_variance_equal_passfail"] = Fr(n_all_pass + n_all_fail, len(scenarios))

    # malformed messages / all assistant messages (pooled over all attempts)
    n_msgs = sum(r["n_msgs"] for r in valid)
    n_bad = sum(r["n_bad"] for r in valid)
    m["n_msgs"], m["n_bad"] = n_msgs, n_bad
    m["invalid_call_rate"] = Fr(n_bad, n_msgs)
    # the other reading (family-weighted mean of per-family ratios), to show which one the table has
    fam_msgs, fam_bad = collections.Counter(), collections.Counter()
    for r in valid:
        fam_msgs[r["family"]] += r["n_msgs"]
        fam_bad[r["family"]] += r["n_bad"]
    m["invalid_call_rate_family_weighted"] = mean_fr(Fr(fam_bad[f], fam_msgs[f]) for f in families)

    # Pass@8: per scenario 1 if at least one attempt passed; family-weighted
    m["pass_at_8"] = family_then_mean({k: Fr(1 if scen_pass[k] >= 1 else 0) for k in scenarios})
    m["n_scen_any_pass"] = sum(1 for k in scenarios if scen_pass[k] >= 1)
    m["scen_n_values"] = sorted(set(scen_n.values()))

    # Pass^k: per scenario C(c, k) / C(n, k); family-weighted; a scenario with n < k is left out
    m["pass_hat_k"], m["n_scenarios_left_out"] = {}, {}
    for k in KS:
        vals = {s: Fr(math.comb(scen_pass[s], k), math.comb(scen_n[s], k))
                for s in scenarios if scen_n[s] >= k}
        m["n_scenarios_left_out"][k] = len(scenarios) - len(vals)
        m["pass_hat_k"][k] = family_then_mean(vals)

    # all wordings pass: per family the product of the pass rates of its styles; mean over families
    prods = []
    for f in families:
        p = Fr(1)
        for s in sorted(st for (ff, st) in scenarios if ff == f):
            p *= Fr(scen_pass[(f, s)], scen_n[(f, s)])
        prods.append(p)
    m["all_wordings_pass"] = mean_fr(prods)

    # turns: assistant messages per attempt; ended by the turn limit without a close; closed
    m["mean_turns"] = Fr(n_msgs, len(valid))
    m["n_at_turn_limit"] = sum(1 for r in valid if r["ended_by"] == "turn_limit" and not r["closed"])
    m["n_closed"] = sum(1 for r in valid if r["closed"])
    m["share_at_turn_limit"] = Fr(m["n_at_turn_limit"], len(valid))
    m["share_closed"] = Fr(m["n_closed"], len(valid))

    # end labels: every attempt gets exactly one
    lab = collections.Counter()
    for r in valid:
        if r["closed"]:
            lab["closed_correctly" if r["passed"] else "closed_wrongly"] += 1
        elif r["n_msgs"] > 0 and r["n_bad"] == r["n_msgs"]:
            lab["malformed_only"] += 1
        else:
            lab["never_closed"] += 1
    m["label_counts"] = {k: lab[k] for k in ("closed_correctly", "closed_wrongly", "never_closed", "malformed_only")}
    m["outcome_labels"] = {k: Fr(v, len(valid)) for k, v in m["label_counts"].items()}
    m["n_passed_not_closed"] = sum(1 for r in valid if r["passed"] and not r["closed"])
    m["n_all_messages_malformed"] = sum(1 for r in valid if r["n_msgs"] > 0 and r["n_bad"] == r["n_msgs"])

    # attempts per reward value
    m["reward_distribution"] = dict(sorted(collections.Counter(r["reward"] for r in valid).items()))

    # ---- extras: in the tables, not on the task list ----
    # variance inside families: wording part = between-style sum of squares / total sum of squares,
    # both summed over families (my reading of the column name; the plan does not define it)
    ss_total, ss_within = Fr(0), Fr(0)
    ratios = []
    for f in families:
        c, n = fam_pass[f], fam_n[f]
        tot = Fr(c) * (1 - Fr(c, n))
        wit = sum((Fr(scen_pass[(f, s)]) * (1 - Fr(scen_pass[(f, s)], scen_n[(f, s)]))
                   for s in sorted(st for (ff, st) in scenarios if ff == f)), Fr(0))
        ss_total += tot
        ss_within += wit
        if tot > 0:
            ratios.append((tot - wit) / tot)
    m["variance_shares"] = {
        "wording": (ss_total - ss_within) / ss_total if ss_total else None,
        "sampling": ss_within / ss_total if ss_total else None,
        "wording_if_sampling_only": Fr(sum(len([1 for (ff, st) in scenarios if ff == f]) - 1 for f in families),
                                       sum(fam_n[f] - 1 for f in families)),
        "wording_mean_of_family_ratios": mean_fr(ratios) if ratios else None,
        "n_families_with_variance": len(ratios),
    }

    # smallest detectable effect, the part that needs no random numbers: the exact variance, over
    # all ways to split each scenario into two halves, of (pass rate of half A - pass rate of half B)
    var_d = Fr(0)
    for f in families:
        scs = [s for (ff, s) in scenarios if ff == f]
        for s in scs:
            c, n = scen_pass[(f, s)], scen_n[(f, s)]
            if n >= 2 and n % 2 == 0:
                p = Fr(c, n)
                var_d += Fr(4) * p * (1 - p) / (n - 1) / (len(scs) ** 2) / (len(families) ** 2)
    m["sde_exact"] = {"se_halves": math.sqrt(var_d), "se": math.sqrt(var_d / 2),
                      "mde": 2.8 * math.sqrt(var_d / 2)}

    # identity fields
    for key in ("split", "ruleset", "toolset", "prompt", "text_sha256", "text_is_placeholder",
                "config_sha256", "settings", "arm"):
        m["values_" + key] = sorted(set(r[key] for r in recs), key=str)
    return m


# ----------------------------------------------------------------------------------------------
# 4. Intervals
# ----------------------------------------------------------------------------------------------

def reg_inc_beta(x, a, b):
    """Regularised incomplete beta function I_x(a, b) by its continued fraction (modified Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - reg_inc_beta(1.0 - x, b, a)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta) / a
    tiny = 1e-300
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 2000):
        k = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k))
        else:
            num = -((a + k) * (a + b + k) * x) / ((a + 2 * k) * (a + 2 * k + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        f *= c * d
        if abs(1.0 - c * d) < 1e-16:
            break
    return front * (f - 1.0)


def t_cdf_beta(t, df):
    if t == 0:
        return 0.5
    tail = 0.5 * reg_inc_beta(df / (df + t * t), df / 2.0, 0.5)
    return 1.0 - tail if t > 0 else tail


def t_cdf_simpson(t, df, n=200000):
    """A second route to the same number: Simpson's rule on the density of Student's t."""
    logc = math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    u = np.linspace(0.0, abs(t), n + 1)
    y = np.exp(logc - (df + 1) / 2.0 * np.log1p(u * u / df))
    h = abs(t) / n
    area = h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())
    return 0.5 + area if t > 0 else 0.5 - area


def t_quantile(p, df, cdf=t_cdf_beta):
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_interval(diffs, level=LEVEL):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    tq = t_quantile(0.5 + level / 2.0, n - 1)
    half = tq * sd / math.sqrt(n)
    return mean - half, mean + half, sd, tq


def percentile_interval(means, level=LEVEL):
    a = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100.0 * a, 100.0 * (1.0 - a)])
    return float(lo), float(hi)


def exact_bootstrap(int_values, denom):
    """The bootstrap distribution with no Monte Carlo: the family values are whole numbers of
    attempts (value = int / denom), so the mean of F families drawn with replacement is the sum of
    F independent draws from the list, divided by F * denom. The distribution of that sum is the
    F-fold convolution of the distribution of one draw. Returns (support of the mean, cdf)."""
    v = np.asarray(int_values, dtype=np.int64)
    shift = int(v.min())
    one = np.bincount(v - shift).astype(np.float64) / len(v)
    pmf = np.array([1.0])
    for _ in range(len(v)):
        pmf = np.convolve(pmf, one)
    support = (np.arange(len(pmf)) + len(v) * shift) / float(len(v) * denom)
    return support, np.cumsum(pmf)


def exact_quantile(support, cdf, p):
    i = int(np.argmax(cdf >= p - 1e-12))
    return float(support[i])


def exact_interval_and_band(int_values, denom, level=LEVEL, n_draws=N_DRAWS):
    """Exact percentile ends, and for each end the band in which a run with n_draws draws is
    expected to land about 95% of the time: the number of draws below a point is binomial, so the
    empirical level of the end is p +- 1.96 * sqrt(p (1 - p) / n_draws)."""
    support, cdf = exact_bootstrap(int_values, denom)
    a = (1.0 - level) / 2.0
    se_p = math.sqrt(a * (1.0 - a) / n_draws)
    out = {"step": 1.0 / (len(int_values) * denom), "se_level": se_p}
    for name, p in (("lo", a), ("hi", 1.0 - a)):
        out[name] = exact_quantile(support, cdf, p)
        out[name + "_band"] = (exact_quantile(support, cdf, p - Z95 * se_p),
                               exact_quantile(support, cdf, p + Z95 * se_p))
        out[name + "_band_wide"] = (exact_quantile(support, cdf, p - Z999 * se_p),
                                    exact_quantile(support, cdf, p + Z999 * se_p))
    return out


def passes_array(recs, fam_order):
    """Pass marks as a (families, styles, attempts) array of 0 and 1; attempts by their number."""
    pos = {f: i for i, f in enumerate(fam_order)}
    y = np.zeros((len(fam_order), 4, 8), dtype=np.float64)
    for r in recs:
        if r["harness_error"] is None:
            y[pos[r["family"]], r["style"] - 1, r["sample"]] = 1.0 if r["passed"] else 0.0
    return y


def two_halves(y, seed, n_splits):
    """Smallest detectable effect, my reading of the plan's words plus what the table's own numbers
    show (section 8.2 of the report says how I got there): for each split, every scenario's attempts
    are put in a random order (one rng.permutation per scenario, families in the order of dev.jsonl,
    styles 1 to 4) and cut into a first and a second half. Per family: pass rate of the first half
    minus pass rate of the second half. One split alone gives a standard error: the standard
    deviation of the family differences (n - 1) / sqrt(families). se_halves: the root mean square of
    that over the splits. se = se_halves / sqrt(2). mde = 2.8 * se."""
    rng = np.random.default_rng(seed)
    n_f, n_s, n_a = y.shape
    half = n_a // 2
    se = []
    for _ in range(n_splits):
        a = np.zeros((n_f, n_s))
        b = np.zeros((n_f, n_s))
        for f in range(n_f):
            for st in range(n_s):
                order = rng.permutation(n_a)
                a[f, st] = y[f, st, order[:half]].mean()
                b[f, st] = y[f, st, order[half:]].mean()
        d = a.mean(axis=1) - b.mean(axis=1)
        se.append(d.std(ddof=1) / math.sqrt(n_f))
    se = np.array(se)
    se_halves = math.sqrt(float((se ** 2).mean()))
    return {"se_halves": se_halves, "se": se_halves / math.sqrt(2), "mde": 2.8 * se_halves / math.sqrt(2),
            "mde_lowest_split": 2.8 * float(se.min()) / math.sqrt(2),
            "mde_highest_split": 2.8 * float(se.max()) / math.sqrt(2),
            "mean_of_split_se": float(se.mean()), "median_of_split_se": float(np.median(se))}


# ----------------------------------------------------------------------------------------------
# 5. Tests on the family differences
# ----------------------------------------------------------------------------------------------

def sign_test(int_diffs):
    pos = sum(1 for d in int_diffs if d > 0)
    neg = sum(1 for d in int_diffs if d < 0)
    zero = sum(1 for d in int_diffs if d == 0)
    n = pos + neg
    if n == 0:
        return {"pos": pos, "neg": neg, "zero": zero, "n": 0, "p_two_sided": Fr(1), "p_one_sided_gain": Fr(1)}
    k = max(pos, neg)
    tail = sum(math.comb(n, j) for j in range(k, n + 1))
    p_two = min(Fr(1), Fr(2 * tail, 2 ** n))
    p_one = Fr(sum(math.comb(n, j) for j in range(pos, n + 1)), 2 ** n)
    return {"pos": pos, "neg": neg, "zero": zero, "n": n, "p_two_sided": p_two, "p_one_sided_gain": p_one}


def sign_flip_test_exact(int_diffs):
    """Paired permutation test: under 'no difference between the two arms' the two labels inside a
    family can be swapped, which flips the sign of that family's difference. All 2^n sign patterns
    of the n non-zero differences are equally likely. Statistic: the sum of the differences (the
    mean times a constant). Counted exactly by dynamic programming over whole-number sums."""
    nz = [abs(d) for d in int_diffs if d != 0]
    total = sum(nz)
    ways = [0] * (2 * total + 1)
    ways[total] = 1
    for v in nz:
        new = [0] * (2 * total + 1)
        for i, w in enumerate(ways):
            if w:
                new[i + v] += w
                new[i - v] += w
        ways = new
    obs = sum(int_diffs)
    n_abs = sum(w for i, w in enumerate(ways) if abs(i - total) >= abs(obs))
    n_ge = sum(w for i, w in enumerate(ways) if (i - total) >= obs)
    assert sum(ways) == 2 ** len(nz)
    return {"n_nonzero": len(nz), "n_patterns": 2 ** len(nz), "observed_sum": obs,
            "n_patterns_abs_ge": n_abs, "n_patterns_ge": n_ge,
            "p_two_sided": Fr(n_abs, 2 ** len(nz)), "p_one_sided_gain": Fr(n_ge, 2 ** len(nz))}


def sign_flip_test_mc(int_diffs, rng, n=1000000, chunk=100000):
    d = np.asarray(int_diffs, dtype=np.int64)
    obs = abs(int(d.sum()))
    hits = 0
    for _ in range(n // chunk):
        signs = rng.integers(0, 2, size=(chunk, len(d)), dtype=np.int8) * 2 - 1
        hits += int((np.abs(signs.astype(np.int64) @ d) >= obs).sum())
    return hits, n


# ----------------------------------------------------------------------------------------------
# 6. Reading the two tables
# ----------------------------------------------------------------------------------------------

def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, prefix + "." + str(k) if prefix else str(k)))
    elif isinstance(obj, list) and any(isinstance(x, (dict, list)) for x in obj):
        for i, v in enumerate(obj):
            out.update(flatten(v, "%s[%d]" % (prefix, i)))
    else:
        out[prefix] = obj
    return out


def parse_md(path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    tables, cur = [], None
    for ln in lines:
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if cur is None:
                cur = {"header": cells, "rows": []}
            elif set("".join(cells)) <= set("-: "):
                continue
            else:
                cur["rows"].append(dict(zip(cur["header"], cells)))
        elif cur is not None:
            tables.append(cur)
            cur = None
    if cur is not None:
        tables.append(cur)
    rewards = {}
    for ln in lines:
        mm = re.match(r"^- (\w+): (reward .*)$", ln)
        if mm:
            rewards[mm.group(1)] = {int(a): int(b) for a, b in re.findall(r"reward (-?\d+): (\d+)", mm.group(2))}
    return lines, tables, rewards


NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def md_numbers(cell):
    return NUM.findall(cell)


def md_equal(text, my_value):
    """Is the printed number the rounding of my value? Tolerance: half a unit of the last printed digit."""
    decimals = len(text.split(".")[1]) if "." in text else 0
    return abs(float(text) - my_value) <= 0.5 * 10 ** (-decimals) + 1e-9


# ----------------------------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------------------------

def main(tables=None, out_dir=None, quiet=False):
    """tables and out_dir can be replaced (selftest_compare.py does that to plant errors on purpose)."""
    tables = tables or TABLES
    out_dir = out_dir or HERE
    fams, fam_order = read_families(DEV)
    F = len(fam_order)

    # ---------------- read and check ----------------
    hashes = collections.OrderedDict()
    hashes[os.path.relpath(DEV, REPO)] = sha256_file(DEV)
    for arm in ARMS:
        hashes[os.path.relpath(EPISODES[arm], REPO)] = sha256_file(EPISODES[arm])
    for name in ("seed1", "seed2"):
        for ext in (".json", ".md"):
            hashes[os.path.relpath(tables[name] + ext, REPO)] = sha256_file(tables[name] + ext)
    dev_sha = hashes[os.path.relpath(DEV, REPO)]
    data, checks = {}, []
    for arm in ARMS:
        recs = read_attempts(EPISODES[arm], fams)
        data[arm] = recs
        keys = [(r["family"], r["style"], r["sample"]) for r in recs]
        want = set((f, s, i) for f in fam_order for s in (1, 2, 3, 4) for i in range(8))
        checks.append((arm, "records in the file", len(recs), 1536))
        checks.append((arm, "distinct (family, style, attempt number)", len(set(keys)), 1536))
        checks.append((arm, "missing or extra against 48 families x 4 styles x 8 attempts",
                       len(want.symmetric_difference(set(keys))), 0))
        checks.append((arm, "records whose arm field is not the arm", sum(1 for r in recs if r["arm"] != arm), 0))
        checks.append((arm, "records whose split is not dev", sum(1 for r in recs if r["split"] != "dev"), 0))
        checks.append((arm, "records whose outcome differs from dev.jsonl",
                       sum(1 for r in recs if fams[r["family"]]["outcome"] != r["outcome"]), 0))
        checks.append((arm, "records whose scenario_id is not family/s<style>",
                       sum(1 for r in recs if r["scenario"] != "%s/s%d" % (r["family"], r["style"])), 0))
        checks.append((arm, "records whose n_turns differs from the number of saved turns",
                       sum(1 for r in recs if r["n_turns_field"] != r["n_msgs"]), 0))
        checks.append((arm, "records whose n_malformed differs from the turns with parse_ok false",
                       sum(1 for r in recs if r["n_malformed_field"] != r["n_bad"]), 0))
        checks.append((arm, "records whose turn_budget differs from dev.jsonl (default rule set)",
                       sum(1 for r in recs if fams[r["family"]]["turn_budget_default"] != r["turn_budget"]), 0))
        checks.append((arm, "records where closed and ended_by disagree",
                       sum(1 for r in recs if r["closed"] != (r["ended_by"] == "close")), 0))
        checks.append((arm, "records whose data_sha256 is not the SHA-256 of data/scenarios/dev.jsonl",
                       sum(1 for r in recs if r["data_sha256"] != dev_sha), 0))
    settings_all = sorted(set(r["settings"] for arm in ARMS for r in data[arm]))
    checks.append(("all four", "distinct settings blocks over all records", len(settings_all), 1))
    bad_checks = [c for c in checks if c[2] != c[3]]

    M = {arm: arm_metrics(data[arm], fams, fam_order) for arm in ARMS}

    # family vectors, in the order of dev.jsonl
    for arm in ARMS:
        if M[arm]["families"] != fam_order:
            raise SystemExit("arm %s does not cover the 48 families" % arm)
    same_n = all(set(M[arm]["fam_n"].values()) == {32} for arm in ARMS)
    C = {arm: np.array([M[arm]["fam_pass"][f] for f in fam_order], dtype=np.int64) for arm in ARMS}
    Nf = {arm: np.array([M[arm]["fam_n"][f] for f in fam_order], dtype=np.int64) for arm in ARMS}
    R = {arm: C[arm] / Nf[arm] for arm in ARMS}        # family pass rates as floats
    if not same_n:
        raise SystemExit("a family does not have 32 valid attempts; the exact bootstrap below assumes it")
    DEN = 32

    # ---------------- bootstrap, my seed ----------------
    rng = np.random.default_rng(MY_SEED)
    boot = {}
    for arm in ARMS:
        idx = rng.integers(0, F, size=(N_DRAWS, F))
        boot[arm] = percentile_interval(R[arm][idx].mean(axis=1))
    pair = {}
    for a, b in PAIRS:
        idx = rng.integers(0, F, size=(N_DRAWS, F))      # the same resampled families for both arms
        means = R[b][idx].mean(axis=1) - R[a][idx].mean(axis=1)
        d_int = (C[b] - C[a]).tolist()
        d = R[b] - R[a]
        t_lo, t_hi, sd, tq = t_interval(d)
        pair[(a, b)] = {
            "diff": Fr(int(C[b].sum() - C[a].sum()), F * DEN),
            "boot": percentile_interval(means),
            "t": (t_lo, t_hi), "sd_of_family_differences": sd, "t_quantile": tq,
            "d_int": d_int,
        }
        assert abs(float(pair[(a, b)]["diff"]) - d.mean()) < 1e-12

    # ---------------- the same with the table's seed (0): can my code give the table's ends exactly? ----
    # One fresh generator per interval, numpy's default generator, whole index matrix in one call.
    # (seed0_trials.py shows the other ways to draw that I tried; this is the one that gives the table.)
    boot0 = {}
    for arm in ARMS:
        idx = np.random.default_rng(TABLE_SEED).integers(0, F, size=(N_DRAWS, F))
        boot0[arm] = percentile_interval(R[arm][idx].mean(axis=1))
    for a, b in PAIRS:
        idx = np.random.default_rng(TABLE_SEED).integers(0, F, size=(N_DRAWS, F))
        boot0[(a, b)] = percentile_interval(R[b][idx].mean(axis=1) - R[a][idx].mean(axis=1))

    # ---------------- exact bootstrap and the expected Monte Carlo band ----------------
    exact = {arm: exact_interval_and_band(C[arm].tolist(), DEN) for arm in ARMS}
    for a, b in PAIRS:
        exact[(a, b)] = exact_interval_and_band(pair[(a, b)]["d_int"], DEN)

    # ---------------- 200 further seeds: how much do the ends move? ----------------
    quantities = list(ARMS) + list(PAIRS)
    V = np.stack([R[q] if isinstance(q, str) else (R[q[1]] - R[q[0]]) for q in quantities], axis=1)  # (F, 9)
    ends = np.zeros((len(NOISE_SEEDS), 2, len(quantities)))
    for si, seed in enumerate(NOISE_SEEDS):
        g = np.random.default_rng(seed)
        idx = g.integers(0, F, size=(N_DRAWS, F))
        means = V[idx].mean(axis=1)                       # (N_DRAWS, 9)
        ends[si] = np.percentile(means, [2.5, 97.5], axis=0)
    noise = {}
    for qi, qn in enumerate(quantities):
        noise[qn] = {"lo_min": float(ends[:, 0, qi].min()), "lo_max": float(ends[:, 0, qi].max()),
                     "lo_sd": float(ends[:, 0, qi].std(ddof=1)), "lo_mean": float(ends[:, 0, qi].mean()),
                     "hi_min": float(ends[:, 1, qi].min()), "hi_max": float(ends[:, 1, qi].max()),
                     "hi_sd": float(ends[:, 1, qi].std(ddof=1)), "hi_mean": float(ends[:, 1, qi].mean()),
                     "lo_all": ends[:, 0, qi].copy(), "hi_all": ends[:, 1, qi].copy()}

    # ---------------- t quantile, two routes ----------------
    tq_beta = t_quantile(0.975, F - 1, t_cdf_beta)
    tq_simpson = t_quantile(0.975, F - 1, t_cdf_simpson)

    # ---------------- questions (b) to (e) ----------------
    q = {}
    prng = np.random.default_rng(PERM_SEED)
    for a, b in PAIRS:
        d_int = pair[(a, b)]["d_int"]
        st = sign_test(d_int)
        pt = sign_flip_test_exact(d_int)
        hits, n_mc = sign_flip_test_mc(d_int, prng)
        q[(a, b)] = {"sign": st, "perm": pt, "perm_mc": (hits, n_mc),
                     "n_zero": st["zero"], "n_pos": st["pos"], "n_neg": st["neg"]}
        # (b) more than 10 points in either direction: |d| / 32 > 0.10
        moved = [(f, C[a][i], C[b][i]) for i, f in enumerate(fam_order) if abs(d_int[i]) * 10 > DEN]
        q[(a, b)]["moved"] = [(f, int(x), int(y)) for f, x, y in moved]
        # (c) leave one family out
        loo = {}
        for name, pick in (("largest_gain", max(d_int)), ("largest_loss", min(d_int))):
            ties = [fam_order[i] for i, v in enumerate(d_int) if v == pick]
            i_out = d_int.index(pick)
            rest = [v for i, v in enumerate(d_int) if i != i_out]
            ex = exact_interval_and_band(rest, DEN)
            tl, th, _sd, tqq = t_interval(np.array(rest) / DEN)
            loo[name] = {"family": fam_order[i_out], "ties": ties, "d_int": pick,
                         "diff": Fr(sum(rest), len(rest) * DEN), "n_families": len(rest),
                         "exact_boot": (ex["lo"], ex["hi"]), "t": (tl, th),
                         "sign": sign_test(rest), "perm": sign_flip_test_exact(rest)}
        q[(a, b)]["loo"] = loo

    # ---------------- extra: two-halves splits with my own seed ----------------
    srng = np.random.default_rng(SPLIT_SEED)
    sde = {}
    for arm in ARMS:
        # passes as a (families, styles, attempts) 0/1 array, attempts ordered by their number
        y = np.zeros((F, 4, 8), dtype=np.float64)
        for r in data[arm]:
            y[fam_order.index(r["family"]), r["style"] - 1, r["sample"]] = 1.0 if r["passed"] else 0.0
        D, se_family = [], []
        for _ in range(200):
            perm = np.argsort(srng.random((F, 4, 8)), axis=2)
            ys = np.take_along_axis(y, perm, axis=2)
            d_f = ys[:, :, :4].mean(axis=(1, 2)) - ys[:, :, 4:].mean(axis=(1, 2))   # per family
            D.append(d_f.mean())
            se_family.append(d_f.std(ddof=1) / math.sqrt(F))
        D, se_family = np.array(D), np.array(se_family)
        sde[arm] = {
            "rms_of_half_difference": float(np.sqrt((D ** 2).mean())),
            "sd_of_half_difference": float(D.std(ddof=1)),
            "mean_of_family_se": float(se_family.mean()),
            "rms_of_family_se": float(np.sqrt((se_family ** 2).mean())),
            "min_family_se": float(se_family.min()), "max_family_se": float(se_family.max()),
        }

    # ---------------- extra: two-halves splits with the table's seed (0) ----------------
    sde0 = {arm: two_halves(passes_array(data[arm], fam_order), TABLE_SEED, N_SPLITS) for arm in ARMS}

    # ---------------- extra: 20,000 splits of my own: where does the pooled value settle? ----------------
    many = {}
    if not quiet:
        mrng = np.random.default_rng(MANY_SPLITS_SEED)
        for arm in ARMS:
            y = passes_array(data[arm], fam_order)
            sq = []
            for _ in range(10):                                   # 10 x 2000 splits, to keep memory small
                order = np.argsort(mrng.random((2000,) + y.shape), axis=3)
                ys = np.take_along_axis(np.broadcast_to(y, (2000,) + y.shape), order, axis=3)
                d = ys[..., :4].mean(axis=(2, 3)) - ys[..., 4:].mean(axis=(2, 3))      # (2000, families)
                sq.append(d.std(axis=1, ddof=1) ** 2 / F)
                del order, ys, d
            many[arm] = math.sqrt(float(np.concatenate(sq).mean()))

    # ---------------- extra: pass and reward by the words of section 6 ----------------
    words = {}
    for arm in ARMS:
        recs = data[arm]
        w = {"n": len(recs)}
        w["pass_multiset_disagree"] = sum(1 for r in recs if r["pass_words_multiset"] != r["passed"])
        w["pass_set_disagree"] = sum(1 for r in recs if r["pass_words_set"] != r["passed"])
        w["set_vs_multiset_differ"] = sum(1 for r in recs if r["pass_words_set"] != r["pass_words_multiset"])
        w["doubled_record"] = sum(1 for r in recs if r["doubled_record"])
        w["doubled_record_and_passed"] = sum(1 for r in recs if r["doubled_record"] and r["passed"])

        def reward_words(r):
            if not r["passed"]:
                return 0
            return 100 - min(15, 5 * r["n_bad"]) - min(10, 2 * max(0, r["n_msgs"] - fams[r["family"]]["turn_budget_default"]))
        w["reward_disagree"] = sum(1 for r in recs if reward_words(r) != r["reward"])
        w["fail_with_reward"] = sum(1 for r in recs if (not r["passed"]) and r["reward"] != 0)
        words[arm] = w

    # ==============================================================================================
    # compare with the tables
    # ==============================================================================================
    T, MD = {}, {}
    for name, stem in tables.items():
        with open(stem + ".json", "r", encoding="utf-8") as fh:
            T[name] = json.load(fh)
        MD[name] = parse_md(stem + ".md")

    rows = []          # one comparison each
    not_remade = []    # table leaves that I did not make again

    def add(table, kind, metric, arm, tval, mdval, mine, equal, note=""):
        rows.append({"table": table, "kind": kind, "metric": metric, "arm": arm, "table_value": tval,
                     "md_value": mdval, "mine": mine, "equal": equal, "note": note})

    def my_arm_leaves(arm):
        m = M[arm]
        L = {}
        L["arm"] = ("identity", arm)
        L["splits"] = ("identity", m["values_split"])
        L["ruleset"] = ("identity", m["values_ruleset"][0] if len(m["values_ruleset"]) == 1 else m["values_ruleset"])
        L["toolset"] = ("identity", m["values_toolset"][0] if len(m["values_toolset"]) == 1 else m["values_toolset"])
        L["prompt"] = ("identity", m["values_prompt"][0] if len(m["values_prompt"]) == 1 else m["values_prompt"])
        L["text_is_placeholder"] = ("identity", m["values_text_is_placeholder"][0]
                                    if len(m["values_text_is_placeholder"]) == 1 else m["values_text_is_placeholder"])
        L["config_sha256"] = ("identity", m["values_config_sha256"])
        L["n_families"] = ("exact", m["n_families"])
        L["n_episodes"] = ("exact", m["n_episodes"])
        L["n_harness_errors"] = ("exact", m["n_harness_errors"])
        L["pass_rate"] = ("exact", m["pass_rate"])
        L["pass_rate_lo"] = ("mc", boot[arm][0])
        L["pass_rate_hi"] = ("mc", boot[arm][1])
        for o in OUTCOMES:
            L["pass_rate_by_outcome." + o] = ("exact", m["pass_rate_by_outcome"][o])
        L["pass_rate_seen"] = ("exact", m["pass_rate_seen"])
        L["pass_rate_held_out"] = ("exact", m["pass_rate_held_out"])
        L["zero_variance_equal_reward"] = ("exact", m["zero_variance_equal_reward"])
        L["zero_variance_equal_passfail"] = ("exact", m["zero_variance_equal_passfail"])
        L["invalid_call_rate"] = ("exact", m["invalid_call_rate"])
        L["pass_at_8"] = ("exact", m["pass_at_8"])
        for k in KS:
            L["pass_hat_k.%d" % k] = ("exact", m["pass_hat_k"][k])
            L["n_scenarios_left_out.%d" % k] = ("exact", m["n_scenarios_left_out"][k])
        L["all_wordings_pass"] = ("exact", m["all_wordings_pass"])
        L["variance_shares.wording"] = ("extra_exact", m["variance_shares"]["wording"])
        L["variance_shares.sampling"] = ("extra_exact", m["variance_shares"]["sampling"])
        L["variance_shares.wording_if_sampling_only"] = ("extra_exact", m["variance_shares"]["wording_if_sampling_only"])
        L["turn_stats.n_episodes"] = ("exact", m["n_episodes"])
        L["turn_stats.mean_turns"] = ("exact", m["mean_turns"])
        L["turn_stats.share_at_turn_limit"] = ("exact", m["share_at_turn_limit"])
        L["turn_stats.share_closed"] = ("exact", m["share_closed"])
        for k2 in ("closed_correctly", "closed_wrongly", "never_closed", "malformed_only"):
            L["outcome_labels." + k2] = ("exact", m["outcome_labels"][k2])
        for rv, cnt in m["reward_distribution"].items():
            L["reward_distribution.%d" % rv] = ("exact", cnt)
        L["smallest_detectable_effect.n_families"] = ("exact", m["n_families"])
        L["smallest_detectable_effect.n_splits"] = ("setting", 200)
        for k2 in ("se_halves", "se", "mde", "mde_lowest_split", "mde_highest_split"):
            L["smallest_detectable_effect." + k2] = ("seed0", sde0[arm][k2])
        return L

    mc_rows = []       # interval ends: table, mine, exact, band, 200 seeds

    def mc_row(table, key, label, end, tval, mval):
        ex, nz = exact[key], noise[key]
        band, wide = ex[end + "_band"], ex[end + "_band_wide"]
        seeds = nz[end + "_all"]
        mine0 = boot0[key][0 if end == "lo" else 1]
        return {"table": table, "what": label, "end": end, "table_value": tval, "mine": mval,
                "mine0": mine0, "equal0": abs(tval - mine0) <= TOL_JSON,
                "exact": ex[end], "band": band, "band_wide": wide,
                "seed_min": nz[end + "_min"], "seed_max": nz[end + "_max"], "seed_sd": nz[end + "_sd"],
                "inside_band": band[0] - 1e-12 <= tval <= band[1] + 1e-12,
                "inside_wide": wide[0] - 1e-12 <= tval <= wide[1] + 1e-12,
                "inside_seeds": nz[end + "_min"] - 1e-12 <= tval <= nz[end + "_max"] + 1e-12,
                "mine_inside_band": band[0] - 1e-12 <= mval <= band[1] + 1e-12,
                "seeds_inside_band": int(((seeds >= band[0] - 1e-12) & (seeds <= band[1] + 1e-12)).sum()),
                "n_seeds": int(len(seeds)), "step": ex["step"]}
    for name in ("seed1", "seed2"):
        tab = T[name]
        # ---- top level ----
        top_mine = {
            "n_families": F,
            "n_scenarios": M[TABLE_ARMS[name][0]]["n_scenarios"],
            "n_episodes_per_arm": M[TABLE_ARMS[name][0]]["n_episodes"],
            "splits": sorted(set(s for a in TABLE_ARMS[name] for s in M[a]["values_split"])),
            "text_sha256": M[TABLE_ARMS[name][0]]["values_text_sha256"][0],
            "text_is_placeholder": M[TABLE_ARMS[name][0]]["values_text_is_placeholder"][0],
        }
        for key, val in tab.items():
            if key in ("arms", "pairs"):
                continue
            if key in top_mine:
                add(name, "identity" if not isinstance(val, (int, float)) or isinstance(val, bool) else "exact",
                    "top." + key, "-", val, None, top_mine[key], val == top_mine[key])
            elif key == "notes":
                not_remade.append((name, "top." + key, val, "a list of notes of the table code; it is empty"))
            else:
                not_remade.append((name, "top." + key, val, "a setting or label of the table code, not a result counted from attempts"))
        # ---- arms ----
        t_arms = {e["arm"]: e for e in tab["arms"]}
        if sorted(t_arms) != sorted(TABLE_ARMS[name]):
            raise SystemExit("table %s holds arms %s" % (name, sorted(t_arms)))
        for arm in TABLE_ARMS[name]:
            leaves = flatten(t_arms[arm])
            mine = my_arm_leaves(arm)
            for path, tval in leaves.items():
                if path not in mine:
                    not_remade.append((name, "arms[%s].%s" % (arm, path), tval, "no value of mine for this entry"))
                    continue
                kind, mval = mine[path]
                if kind == "setting":
                    not_remade.append((name, "arms[%s].%s" % (arm, path), tval, "a setting of the table code, not a result"))
                elif kind == "identity":
                    add(name, kind, path, arm, tval, None, mval, tval == mval)
                elif kind in ("exact", "extra_exact"):
                    ok = mval is not None and abs(float(tval) - float(mval)) <= TOL_JSON
                    add(name, kind, path, arm, tval, None, mval, ok)
                elif kind == "mc":
                    end = "lo" if path.endswith("_lo") else "hi"
                    mc_rows.append(mc_row(name, arm, arm, end, tval, mval))
                    add(name, kind, path, arm, tval, None, mval, mc_rows[-1]["inside_wide"],
                        "Monte Carlo: not expected to be equal; 'yes' means inside the 99.9% noise band")
                    add(name, "seed0", path + " (my code, seed 0)", arm, tval, None, mc_rows[-1]["mine0"],
                        mc_rows[-1]["equal0"], "10,000 draws; mine drawn with the table's seed 0")
                elif kind == "seed0":
                    add(name, kind, path, arm, tval, None, mval, abs(float(tval) - float(mval)) <= TOL_JSON,
                        "200 random splits; mine drawn with the table's seed 0")
            # anything of mine that the table does not hold (a reward value with a different count shows here)
            for path in mine:
                if path not in leaves:
                    add(name, mine[path][0], path, arm, None, None, mine[path][1], False, "in my numbers, not in the table")
        # ---- pairs ----
        for e in tab["pairs"]:
            key = (e["a"], e["b"])
            if key not in pair:
                not_remade.append((name, "pairs[%s minus %s]" % (e["b"], e["a"]), None, "pair not made"))
                continue
            p = pair[key]
            label = "%s minus %s" % (e["b"], e["a"])
            for k2, tval in e.items():
                if k2 in ("a", "b"):
                    continue
                if k2 == "n_families":
                    add(name, "exact", "pair.n_families", label, tval, None, F, tval == F)
                elif k2 == "diff":
                    add(name, "exact", "pair.diff", label, tval, None, p["diff"], abs(tval - float(p["diff"])) <= TOL_JSON)
                elif k2 in ("t_lo", "t_hi"):
                    mval = p["t"][0 if k2 == "t_lo" else 1]
                    add(name, "exact", "pair." + k2, label, tval, None, mval, abs(tval - mval) <= 1e-9,
                        "tolerance 1e-9 (my own t quantile)")
                elif k2 in ("lo", "hi"):
                    mval = p["boot"][0 if k2 == "lo" else 1]
                    mc_rows.append(mc_row(name, key, label, k2, tval, mval))
                    add(name, "mc", "pair." + k2, label, tval, None, mval, mc_rows[-1]["inside_wide"],
                        "Monte Carlo: not expected to be equal; 'yes' means inside the 99.9% noise band")
                    add(name, "seed0", "pair." + k2 + " (my code, seed 0)", label, tval, None, mc_rows[-1]["mine0"],
                        mc_rows[-1]["equal0"], "10,000 draws; mine drawn with the table's seed 0")
                else:
                    not_remade.append((name, "pairs[%s].%s" % (label, k2), tval, "no value of mine for this entry"))

    # ---- the .md files: every printed number against my value, and against the .json beside it ----
    md_rows = []

    def md_add(table, metric, arm, text, my_value, kind="exact", json_value=None):
        """text: one printed number. my_value and json_value in the printed unit."""
        eq_mine = md_equal(text, my_value) if my_value is not None else None
        eq_json = md_equal(text, json_value) if json_value is not None else None
        md_rows.append({"table": table, "metric": metric, "arm": arm, "printed": text, "mine": my_value,
                        "kind": kind, "equal_mine": eq_mine, "json_value": json_value, "equal_json": eq_json})

    md_unhandled = []     # anything in the .md that the comparison below would not cover
    MD_COLUMNS = [
        ["Arm", "Episodes", "Harness errors", "Pass rate", "Interval", "grant", "approval", "already", "inactive", "Seen styles",
         "Held-out style", "Groups with equal rewards", "Groups with equal pass or fail", "Malformed messages*"],
        ["Comparison", "Families", "Difference in points", "Paired bootstrap by family", "t-interval"],
        ["Arm", "Split", "Rule set", "Tools", "Prompt", "Text", "Config*"],
        ["Arm", "Pass@8", "Pass^1", "Pass^2", "Pass^4", "Pass^8", "All wordings pass", "Variance inside families*", "Mean turns",
         "Ended at the turn limit", "closed_correctly", "closed_wrongly", "never_closed", "malformed_only", "Smallest detectable effect*"],
    ]
    for name in ("seed1", "seed2"):
        lines, tables, rewards = MD[name]
        tab = T[name]
        t_arms = {e["arm"]: e for e in tab["arms"]}
        t_pairs = {(e["a"], e["b"]): e for e in tab["pairs"]}
        if sorted(rewards) != sorted(TABLE_ARMS[name]):
            md_unhandled.append((name, "reward lines found for the arms %s" % sorted(rewards)))
        head = re.match(r"^Families: (\d+)\. Scenarios: (\d+)\. Episodes per arm: (\d+)\.", lines[0])
        if head:
            md_add(name, "Families (first line)", "-", head.group(1), F, json_value=tab["n_families"])
            md_add(name, "Scenarios (first line)", "-", head.group(2), M[TABLE_ARMS[name][0]]["n_scenarios"], json_value=tab["n_scenarios"])
            md_add(name, "Episodes per arm (first line)", "-", head.group(3), M[TABLE_ARMS[name][0]]["n_episodes"], json_value=tab["n_episodes_per_arm"])
        line2 = re.match(r"^Intervals: (\d+)%, by resampling families, (\d+) draws, seed (\d+)\.", lines[1]) if len(lines) > 1 else None
        if line2:
            md_add(name, "Level in percent (second line)", "-", line2.group(1), None, "setting", 100 * tab["level"])
            md_add(name, "Draws (second line)", "-", line2.group(2), None, "setting", tab["n_draws"])
            md_add(name, "Seed (second line)", "-", line2.group(3), None, "setting", tab["seed"])
        if not head or not line2:
            md_unhandled.append((name, "the first two lines do not have the form that I expect"))
        if len(tables) != 4:
            md_unhandled.append((name, "%d tables found in the .md, 4 expected" % len(tables)))
        for tb in tables:
            h = tb["header"]
            known = None
            for kn in MD_COLUMNS:
                if h[:2] == kn[:2]:
                    known = kn
            if known is None:
                md_unhandled.append((name, "a table that I do not know, with the columns %s" % h))
            else:
                for c in h:
                    if not any(c == k or (k.endswith("*") and c.startswith(k[:-1])) for k in known):
                        md_unhandled.append((name, "column %r is not compared" % c))
                for k in known:
                    if not any(c == k or (k.endswith("*") and c.startswith(k[:-1])) for c in h):
                        md_unhandled.append((name, "column %r is missing in the .md" % k))
            if h[0] == "Arm" and "Episodes" in h:
                for row in tb["rows"]:
                    arm = row["Arm"]
                    m, tj = M[arm], t_arms[arm]
                    md_add(name, "Episodes", arm, row["Episodes"], m["n_episodes"], json_value=tj["n_episodes"])
                    md_add(name, "Harness errors", arm, row["Harness errors"], m["n_harness_errors"], json_value=tj["n_harness_errors"])
                    md_add(name, "Pass rate", arm, row["Pass rate"], 100 * float(m["pass_rate"]), json_value=100 * tj["pass_rate"])
                    lo, hi = md_numbers(row["Interval"])
                    md_add(name, "Interval, lower end", arm, lo, 100 * boot[arm][0], "mc", 100 * tj["pass_rate_lo"])
                    md_add(name, "Interval, upper end", arm, hi, 100 * boot[arm][1], "mc", 100 * tj["pass_rate_hi"])
                    md_add(name, "Interval, lower end (my code, seed 0)", arm, lo, 100 * boot0[arm][0], "seed0")
                    md_add(name, "Interval, upper end (my code, seed 0)", arm, hi, 100 * boot0[arm][1], "seed0")
                    for o in OUTCOMES:
                        md_add(name, o, arm, row[o], 100 * float(m["pass_rate_by_outcome"][o]), json_value=100 * tj["pass_rate_by_outcome"][o])
                    md_add(name, "Seen styles", arm, row["Seen styles"], 100 * float(m["pass_rate_seen"]), json_value=100 * tj["pass_rate_seen"])
                    md_add(name, "Held-out style", arm, row["Held-out style"], 100 * float(m["pass_rate_held_out"]), json_value=100 * tj["pass_rate_held_out"])
                    md_add(name, "Groups with equal rewards", arm, row["Groups with equal rewards"], 100 * float(m["zero_variance_equal_reward"]), json_value=100 * tj["zero_variance_equal_reward"])
                    md_add(name, "Groups with equal pass or fail", arm, row["Groups with equal pass or fail"], 100 * float(m["zero_variance_equal_passfail"]), json_value=100 * tj["zero_variance_equal_passfail"])
                    col = [c for c in h if c.startswith("Malformed messages")][0]
                    md_add(name, "Malformed messages", arm, row[col], 100 * float(m["invalid_call_rate"]), json_value=100 * tj["invalid_call_rate"])
            elif h[0] == "Comparison":
                for row in tb["rows"]:
                    mm = re.match(r"^(\w+) minus (\w+)$", row["Comparison"])
                    key = (mm.group(2), mm.group(1))
                    p, tj = pair[key], t_pairs[key]
                    md_add(name, "Families", row["Comparison"], row["Families"], F, json_value=tj["n_families"])
                    md_add(name, "Difference in points", row["Comparison"], row["Difference in points"], 100 * float(p["diff"]), json_value=100 * tj["diff"])
                    lo, hi = md_numbers(row["Paired bootstrap by family"])
                    md_add(name, "Paired bootstrap, lower end", row["Comparison"], lo, 100 * p["boot"][0], "mc", 100 * tj["lo"])
                    md_add(name, "Paired bootstrap, upper end", row["Comparison"], hi, 100 * p["boot"][1], "mc", 100 * tj["hi"])
                    md_add(name, "Paired bootstrap, lower end (my code, seed 0)", row["Comparison"], lo, 100 * boot0[key][0], "seed0")
                    md_add(name, "Paired bootstrap, upper end (my code, seed 0)", row["Comparison"], hi, 100 * boot0[key][1], "seed0")
                    lo, hi = md_numbers(row["t-interval"])
                    md_add(name, "t-interval, lower end", row["Comparison"], lo, 100 * p["t"][0], json_value=100 * tj["t_lo"])
                    md_add(name, "t-interval, upper end", row["Comparison"], hi, 100 * p["t"][1], json_value=100 * tj["t_hi"])
            elif h[0] == "Arm" and "Split" in h:
                for row in tb["rows"]:
                    arm = row["Arm"]
                    m = M[arm]
                    col = [c for c in h if c.startswith("Config")][0]
                    ident = [("Split", ",".join(m["values_split"])), ("Rule set", ",".join(m["values_ruleset"])),
                             ("Tools", ",".join(m["values_toolset"])), ("Prompt", ",".join(m["values_prompt"])),
                             ("Text", ",".join(v[:12] for v in m["values_text_sha256"])),
                             (col, ",".join(v[:12] for v in m["values_config_sha256"]))]
                    for cname, mine_text in ident:
                        md_rows.append({"table": name, "metric": cname + " (text)", "arm": arm, "printed": row[cname],
                                        "mine": mine_text, "kind": "identity", "equal_mine": row[cname] == mine_text,
                                        "json_value": None, "equal_json": None})
            elif h[0] == "Arm" and "Pass@8" in h:
                for row in tb["rows"]:
                    arm = row["Arm"]
                    m, tj = M[arm], t_arms[arm]
                    md_add(name, "Pass@8", arm, row["Pass@8"], 100 * float(m["pass_at_8"]), json_value=100 * tj["pass_at_8"])
                    for k in KS:
                        md_add(name, "Pass^%d" % k, arm, row["Pass^%d" % k], 100 * float(m["pass_hat_k"][k]), json_value=100 * tj["pass_hat_k"][str(k)])
                    md_add(name, "All wordings pass", arm, row["All wordings pass"], 100 * float(m["all_wordings_pass"]), json_value=100 * tj["all_wordings_pass"])
                    col = [c for c in h if c.startswith("Variance inside families")][0]
                    v1, v2 = md_numbers(row[col])
                    md_add(name, "Variance due to wording (extra)", arm, v1, 100 * float(m["variance_shares"]["wording"]), "extra_exact", 100 * tj["variance_shares"]["wording"])
                    md_add(name, "Variance due to wording if sampling only (extra)", arm, v2, 100 * float(m["variance_shares"]["wording_if_sampling_only"]), "extra_exact", 100 * tj["variance_shares"]["wording_if_sampling_only"])
                    md_add(name, "Mean turns", arm, row["Mean turns"], float(m["mean_turns"]), json_value=tj["turn_stats"]["mean_turns"])
                    md_add(name, "Ended at the turn limit", arm, row["Ended at the turn limit"], 100 * float(m["share_at_turn_limit"]), json_value=100 * tj["turn_stats"]["share_at_turn_limit"])
                    for k2 in ("closed_correctly", "closed_wrongly", "never_closed", "malformed_only"):
                        md_add(name, k2, arm, row[k2], 100 * float(m["outcome_labels"][k2]), json_value=100 * tj["outcome_labels"][k2])
                    col = [c for c in h if c.startswith("Smallest detectable effect")][0]
                    nums = md_numbers(row[col])
                    sj = tj["smallest_detectable_effect"]
                    md_add(name, "Smallest detectable effect (extra)", arm, nums[0], 100 * sde0[arm]["mde"], "seed0", 100 * sj["mde"])
                    md_add(name, "Smallest detectable effect: splits (extra)", arm, nums[1], None, "setting", sj["n_splits"])
                    md_add(name, "Smallest detectable effect: lowest split (extra)", arm, nums[2], 100 * sde0[arm]["mde_lowest_split"], "seed0", 100 * sj["mde_lowest_split"])
                    md_add(name, "Smallest detectable effect: highest split (extra)", arm, nums[3], 100 * sde0[arm]["mde_highest_split"], "seed0", 100 * sj["mde_highest_split"])
        for arm, dist in rewards.items():
            mine_d = M[arm]["reward_distribution"]
            json_d = {int(k): v for k, v in t_arms[arm]["reward_distribution"].items()}
            for rv in sorted(set(dist) | set(mine_d)):
                md_rows.append({"table": name, "metric": "attempts with reward %d" % rv, "arm": arm,
                                "printed": str(dist.get(rv)), "mine": mine_d.get(rv), "kind": "exact",
                                "equal_mine": dist.get(rv) == mine_d.get(rv),
                                "json_value": json_d.get(rv), "equal_json": dist.get(rv) == json_d.get(rv)})

    # are the base and sft entries of the two tables the same?
    a1 = {e["arm"]: e for e in T["seed1"]["arms"]}
    a2 = {e["arm"]: e for e in T["seed2"]["arms"]}
    shared_same = {arm: a1[arm] == a2[arm] for arm in ("base", "sft")}
    p1 = {(e["a"], e["b"]): e for e in T["seed1"]["pairs"]}
    p2 = {(e["a"], e["b"]): e for e in T["seed2"]["pairs"]}
    shared_same["sft minus base"] = p1[("base", "sft")] == p2[("base", "sft")]

    # ==============================================================================================
    # verdicts
    # ==============================================================================================
    exact_rows = [r for r in rows if r["kind"] in ("exact", "identity")]
    exact_bad = [r for r in exact_rows if not r["equal"]]
    extra_exact_rows = [r for r in rows if r["kind"] == "extra_exact"]
    extra_exact_bad = [r for r in extra_exact_rows if not r["equal"]]
    seed0_rows = [r for r in rows if r["kind"] == "seed0"]
    seed0_bad = [r for r in seed0_rows if not r["equal"]]
    md_seed0 = [r for r in md_rows if r["kind"] == "seed0"]
    md_seed0_bad = [r for r in md_seed0 if not r["equal_mine"]]
    mc_bad = [r for r in mc_rows if not r["inside_wide"]]          # outside the 99.9% band
    mc_out95 = [r for r in mc_rows if not r["inside_band"]]        # outside the 95% band (some are expected)
    mc_out_seeds = [r for r in mc_rows if not r["inside_seeds"]]   # outside the range of my 200 seeds
    # each distinct end once (base, sft and sft minus base are in both table files)
    mc_unique = {}
    for r in mc_rows:
        mc_unique.setdefault((r["what"], r["end"], repr(r["table_value"])), r)
    mc_unique = list(mc_unique.values())
    md_exact = [r for r in md_rows if r["kind"] in ("exact", "identity", "extra_exact")]
    md_bad = [r for r in md_exact if not r["equal_mine"]]
    md_json_bad = [r for r in md_rows if r["equal_json"] is False]

    def is_tie(r):
        """A printed number whose exact value sits on a rounding tie (for example 31.25 printed with one decimal)."""
        if r["kind"] not in ("exact", "extra_exact") or not isinstance(r["mine"], (int, float)) or "." not in str(r["printed"]):
            return False
        decimals = len(str(r["printed"]).split(".")[1])
        return abs(abs(float(r["printed"]) - float(r["mine"])) - 0.5 * 10 ** (-decimals)) < 1e-9
    md_ties = {}
    for r in md_rows:
        if is_tie(r):
            md_ties.setdefault((r["arm"], r["metric"], r["printed"], r["mine"]), []).append(r["table"])

    # ==============================================================================================
    # my_numbers.json
    # ==============================================================================================
    def clean(o):
        if isinstance(o, Fr):
            return fr_json(o)
        if isinstance(o, dict):
            return {("%s minus %s" % (k[1], k[0]) if isinstance(k, tuple) and len(k) == 2 and k in pair else
                     ("/s".join(map(str, k)) if isinstance(k, tuple) else str(k))): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    dump = {
        "inputs": {"episodes": {a: os.path.relpath(EPISODES[a], REPO) for a in ARMS}, "families": os.path.relpath(DEV, REPO)},
        "sha256_of_files_read": dict(hashes),
        "seeds": {"bootstrap": MY_SEED, "noise_seeds": [NOISE_SEEDS[0], NOISE_SEEDS[-1]], "splits": SPLIT_SEED, "permutation_mc": PERM_SEED,
                  "many_splits": MANY_SPLITS_SEED, "table_seed": TABLE_SEED},
        "t_quantile_0.975_df47": {"beta_continued_fraction": tq_beta, "simpson": tq_simpson},
        "arms": {}, "pairs": {}, "checks": [list(c) for c in checks],
    }
    for arm in ARMS:
        m = dict(M[arm])
        for drop in ("families", "scen_pass", "scen_n", "fam_rate"):
            m.pop(drop)
        m["interval_my_seed"] = boot[arm]
        m["interval_table_seed_0"] = boot0[arm]
        m["interval_exact_bootstrap"] = {k: v for k, v in exact[arm].items()}
        m["interval_200_seeds"] = {k: v for k, v in noise[arm].items() if not k.endswith("_all")}
        m["two_halves_my_seed"] = sde[arm]
        m["two_halves_table_seed_0"] = sde0[arm]
        m["two_halves_20000_splits_my_seed_se_halves"] = many.get(arm)
        m["section6_recheck"] = words[arm]
        dump["arms"][arm] = clean(m)
    for key in PAIRS:
        p = dict(pair[key])
        p["boot_table_seed_0"] = boot0[key]
        p["exact_bootstrap"] = exact[key]
        p["interval_200_seeds"] = {k: v for k, v in noise[key].items() if not k.endswith("_all")}
        qq = dict(q[key])
        dump["pairs"]["%s minus %s" % (key[1], key[0])] = clean({**p, **qq})
    dump["family_passes_of_32"] = {f: {arm: int(M[arm]["fam_pass"][f]) for arm in ARMS} for f in fam_order}
    with open(os.path.join(out_dir, "my_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=1, sort_keys=False)

    # ==============================================================================================
    # report.md
    # ==============================================================================================
    W = []
    w = W.append

    def pts(x, digits=2, sign=False):
        return ("%+." + str(digits) + "f" if sign else "%." + str(digits) + "f") % (100.0 * float(x))

    def show_mine(x):
        if isinstance(x, Fr):
            if x.denominator == 1:
                return "%d" % x.numerator
            if x.denominator <= 20000:
                return "%s (= %d/%d)" % (repr(float(x)), x.numerator, x.denominator)
            return repr(float(x))
        if isinstance(x, float):
            return repr(x)
        if isinstance(x, list):
            return ", ".join(str(v)[:12] + ("..." if isinstance(v, str) and len(v) > 12 else "") for v in x)
        return str(x)

    def show_table(x):
        if isinstance(x, float):
            return repr(x)
        if isinstance(x, list):
            return ", ".join(str(v)[:12] + ("..." if isinstance(v, str) and len(v) > 12 else "") for v in x)
        return str(x)

    def yesno(b):
        return "yes" if b else "NO"

    def signed(v):
        return "0" if v == 0 else "%+d" % v

    n_exact = len(exact_rows)
    n_md_exact = len(md_exact)
    all_exact_ok = not exact_bad and not md_bad and not bad_checks
    w("# The two dev result tables, made again with separate code")
    w("")
    w("Made by `results/main/audit/recount/rederive.py` (standard library and numpy; nothing imported from `accessdesk`). "
      "Every number in the column \"mine\" was counted by that script from the four files `results/main/<arm>_dev/episodes.jsonl` "
      "(`base_dev`, `sft_dev`, `grpo_seed1_dev`, `grpo_seed2_dev`) and from `data/scenarios/dev.jsonl`. Every \"table\" number is read from "
      "`results/main/tables/base_sft_grpo_seed1.json` / `.md` and `base_sft_grpo_seed2.json` / `.md`. All my numbers are also in `my_numbers.json` in this folder.")
    w("")
    w("Files in this folder: `rederive.py` (the calculation and this report), `report.md` (this file), `my_numbers.json` (every number of mine), `rederive_console.txt` (the short summary that the script prints), "
      "`seed0_trials.py` and `seed0_trials.txt` (which way of drawing with seed 0 gives the table's intervals), `selftest_compare.py` and `selftest_compare.txt` (the comparison run against tables with planted errors).")
    w("")
    w("Units: a family is one world with one correct end state (48 on dev). A scenario is one family in one wording style (192). An attempt is one conversation (1536 per arm). \"Points\" are percentage points.")
    w("")
    w("## 1. In short")
    w("")
    def num_gap(r):
        try:
            return abs(float(r["table_value"]) - float(r["mine"]))
        except (TypeError, ValueError):
            return 0.0
    numeric_exact = [r for r in exact_rows + extra_exact_rows if r["kind"] != "identity" and r["table_value"] is not None]
    max_gap_exact = max([num_gap(r) for r in numeric_exact] or [0.0])
    max_gap_seed0 = max([num_gap(r) for r in seed0_rows if r["table_value"] is not None] or [0.0])
    n_json_all = n_exact + len(extra_exact_rows)
    w("The two table files hold `base` and `sft` twice (once in each file) and one GRPO seed each. Counts of comparisons below count each file's entries.")
    w("")
    if all_exact_ok and not extra_exact_bad:
        n_json_text = len([r for r in exact_rows if r["kind"] == "identity"])
        n_md_text = len([r for r in md_exact if r["kind"] == "identity"])
        w("- **Every entry without a random part is equal.** %d entries of the two `.json` files (%d numbers and %d labels such as the split, the rule set and the config hash) "
          "and %d printed entries of the two `.md` files (%d numbers and %d text cells) were compared with my values. Mismatches: %d and %d. "
          "The largest absolute difference to a `.json` number is %.1e (rounding in the last digits of a float)."
          % (n_json_all, n_json_all - n_json_text, n_json_text, n_md_exact, n_md_exact - n_md_text, n_md_text,
             len(exact_bad) + len(extra_exact_bad), len(md_bad), max_gap_exact))
    else:
        w("- **There are mismatches** among the numbers without a random part: %d of %d `.json` entries and %d of %d printed `.md` numbers differ from my values (listed in section 9); input checks that failed: %d."
          % (len(exact_bad) + len(extra_exact_bad), n_json_all, len(md_bad), n_md_exact, len(bad_checks)))
    w("- **Bootstrap intervals, my own seed (%d).** They depend on random draws, so equality is not expected. %d distinct interval ends (4 arms and 5 comparisons, two ends each): "
      "inside the 95%% band that 10,000 draws leave: %d; inside the 99.9%% band: %d; inside the range of 200 further seeds of mine: %d. The largest gap between a table end and my end is %.3f points; "
      "between a table end and the exact bootstrap end (no random draws) %.3f points. One standard deviation of an end over seeds is at most %.3f points (section 5.2)."
      % (MY_SEED, len(mc_unique), len([r for r in mc_unique if r["inside_band"]]), len([r for r in mc_unique if r["inside_wide"]]),
         len([r for r in mc_unique if r["inside_seeds"]]), 100 * max(abs(r["table_value"] - r["mine"]) for r in mc_unique),
         100 * max(abs(r["table_value"] - r["exact"]) for r in mc_unique), 100 * max(r["seed_sd"] for r in mc_unique)))
    w("- **The same intervals and the smallest detectable effect, with the table's seed (0).** With seed 0 my code gives %d of %d of these `.json` numbers exactly (largest absolute difference %.1e), and %d of %d printed `.md` numbers. "
      "So the difference between the table's intervals and mine is the seed and nothing else (sections 5.2 and 8.2)."
      % (len(seed0_rows) - len(seed0_bad), len(seed0_rows), max_gap_seed0, len(md_seed0) - len(md_seed0_bad), len(md_seed0)))
    w("- **The headline numbers as counts** (mine, equal to the tables): passed attempts of 1536: %s. Differences: %s."
      % ("; ".join("%s %d (%.2f%%)" % (a2, M[a2]["n_passed"], 100 * float(M[a2]["pass_rate"])) for a2 in ARMS),
         "; ".join("%s minus %s %+d attempts of 1536 = %s points (paired bootstrap, table: %s to %s; t-interval: %s to %s)"
                   % (k[1], k[0], pair[k]["diff"] * 1536, pts(pair[k]["diff"], 2, True), pts(boot0[k][0], 1, True), pts(boot0[k][1], 1, True),
                      pts(pair[k]["t"][0], 1, True), pts(pair[k]["t"][1], 1, True)) for k in PAIRS)))
    w("- **Tolerances.** `.json` numbers: absolute difference at most 1e-12 (my values are exact fractions turned into floats); t-interval ends: at most 1e-9 (I compute the t quantile myself); "
      "`.md` numbers: the printed number must be my value rounded to the printed digits (half a unit of the last digit).")
    w("- **Not made again:** only settings and labels of the table code (the level 0.95, 10,000 draws, seed 0, 200 splits, the labels `baseline_arm` and `paired_family_bootstrap`, and an empty list of notes; section 9). "
      "Parts of the `.md` files that the comparison does not cover: %d." % len(md_unhandled))
    w("- **The table code was not opened.** I did not read `accessdesk/eval/metrics.py`, `bootstrap.py` or `tables.py` at any point: no difference was left to explain.")
    w("- **The comparison itself was tested** with errors planted on purpose in copies of the tables: `selftest_compare.py` in this folder (result in `selftest_compare.txt`).")
    w("")
    w("Answers to the five questions, in one line each (details in section 6):")
    w("")
    w("- (a) Family-weighted pass rate = plain share of passed attempts, exactly, in all four arms, because every family has 32 valid attempts.")
    for a, b in (("sft", "grpo_seed1"), ("sft", "grpo_seed2")):
        qq = q[(a, b)]
        up = [(f, x, y) for f, x, y in qq["moved"] if y > x]
        down = [(f, x, y) for f, x, y in qq["moved"] if y < x]
        w("- (b) %s minus sft: %d of 48 families moved by more than 10 points (passes of 32, sft then %s): %d up (%s; expected outcome of these families: %s), %d down (%s)."
          % (b, len(qq["moved"]), b, len(up), ", ".join("%s %d then %d" % t for t in up),
             ", ".join(sorted(set(fams[f]["outcome"] for f, _x, _y in up))), len(down),
             ", ".join("%s %d then %d, expected outcome %s" % (f, x, y, fams[f]["outcome"]) for f, x, y in down)))
    for a, b in (("sft", "grpo_seed1"), ("sft", "grpo_seed2")):
        qq, p = q[(a, b)], pair[(a, b)]
        g, l_ = qq["loo"]["largest_gain"], qq["loo"]["largest_loss"]
        w("- (c) %s minus sft: all 48 families %s points; without the largest gain (%s) %s points; without the largest loss (%s) %s points."
          % (b, pts(p["diff"], 2, True), g["family"], pts(g["diff"], 2, True), l_["family"], pts(l_["diff"], 2, True)))
    for a, b in (("sft", "grpo_seed1"), ("sft", "grpo_seed2")):
        qq = q[(a, b)]
        w("- (d) %s minus sft: exact sign test two-sided p = %.2g (%d up, %d down); exact sign-flip permutation test two-sided p = %.2g."
          % (b, float(qq["sign"]["p_two_sided"]), qq["n_pos"], qq["n_neg"], float(qq["perm"]["p_two_sided"])))
    w("- (e) Families with a difference of exactly zero: %s."
      % "; ".join("%s minus sft %d of 48" % (b, q[("sft", b)]["n_zero"]) for b in ("grpo_seed1", "grpo_seed2")))
    w("")

    # ---- inputs ----
    w("## 2. Inputs and checks on them")
    w("")
    w("Counted from the four `episodes.jsonl` files and `dev.jsonl`. \"Want\" is what the design says (48 families x 4 styles x 8 attempts = 1536).")
    w("")
    w("| Check | base | sft | grpo_seed1 | grpo_seed2 | Want | OK |")
    w("|---|---|---|---|---|---|---|")
    order = []
    for arm, what, got, want in checks:
        if arm in ARMS and what not in order:
            order.append(what)
    for what in order:
        got = {arm: [g for a2, w2, g, _wt in checks if a2 == arm and w2 == what][0] for arm in ARMS}
        want = [wt for a2, w2, _g, wt in checks if w2 == what][0]
        w("| %s | %s | %s | %s | %s | %s | %s |" % (what, got["base"], got["sft"], got["grpo_seed1"], got["grpo_seed2"], want,
                                                yesno(all(g == want for g in got.values()))))
    for arm, what, got, want in checks:
        if arm not in ARMS:
            w("| %s (%s) | %s | | | | %s | %s |" % (what, arm, got, want, yesno(got == want)))
    w("")
    w("Every family has 32 valid attempts in every arm: %s. Harness errors (records with `harness_error` not null): %s."
      % (yesno(same_n), ", ".join("%s %d" % (a, M[a]["n_harness_errors"]) for a in ARMS)))
    w("")
    w("The files that were read, with their SHA-256, so that a later change of a file can be seen:")
    w("")
    w("| File | SHA-256 |")
    w("|---|---|")
    for path, digest in hashes.items():
        w("| `%s` | `%s` |" % (path, digest))
    w("")

    # ---- how counted ----
    w("## 3. How I counted each metric (from the words of ANALYSIS_PLAN.md sections 4 to 6 and the column names)")
    w("")
    w("- **Valid attempt**: a record with `harness_error` null. All metrics use valid attempts only.")
    w("- **Pass rate**: per family, passed attempts / valid attempts (32 here); then the mean over the 48 families.")
    w("- **Per outcome**: the same, over the 12 families whose `outcome` in `dev.jsonl` is that outcome.")
    w("- **Seen styles / held-out style**: per family, passes / attempts over styles 1 to 3 (24 attempts) or over style 4 (8 attempts); then the mean over families.")
    w("- **Groups with equal rewards**: scenarios (family x style, 192) whose 8 `reward` values are all the same number / 192. **Equal pass or fail**: scenarios with 8 passes or 0 passes / 192.")
    w("- **Malformed messages**: saved turns with `parse_ok` not true, summed over all attempts / all saved turns (one saved turn = one assistant message).")
    w("- **Pass@8**: per scenario 1 if at least one of the 8 attempts passed; mean inside the family (4 scenarios), then over families.")
    w("- **Pass^k**: per scenario with c passes of n attempts, C(c, k) / C(n, k); mean inside the family, then over families. A scenario with fewer than k attempts would be left out (none is). "
      "Example: a scenario with 6 passes of 8 has Pass^2 = C(6, 2) / C(8, 2) = 15/28 = 0.536: the chance that 2 attempts drawn from its 8 both passed.")
    w("- **All wordings pass**: per family the product of the 4 per-style pass rates; then the mean over families.")
    w("- **Mean turns**: saved turns / attempts. **Ended at the turn limit**: attempts with `ended_by` = turn_limit and `closed` false / attempts. **share_closed**: attempts with `closed` true / attempts.")
    w("- **End labels** (each attempt gets one): closed_correctly = closed and passed; closed_wrongly = closed and not passed; malformed_only = not closed and every message malformed; never_closed = not closed otherwise.")
    w("- **Reward values**: the number of attempts at each value of `reward`.")
    w("- **Interval of a pass rate**: draw 48 families with replacement, take the mean of their pass rates; 10,000 draws; the 2.5th and 97.5th percentile (numpy, linear interpolation). My seed: %d (`numpy.random.default_rng`)." % MY_SEED)
    w("- **Comparison b minus a**: the mean over families of (pass rate in b - pass rate in a). **Paired bootstrap**: the same drawn families for both arms, 10,000 draws, percentiles as above. "
      "**t-interval**: mean +- t(0.975, 47 degrees of freedom) x standard deviation of the 48 differences (n - 1 in the denominator) / sqrt(48).")
    w("- My t quantile: %.12f by my own incomplete beta function, %.12f by Simpson's rule on the t density (two routes, difference %.1e)." % (tq_beta, tq_simpson, abs(tq_beta - tq_simpson)))
    w("")

    # ---- per-arm tables ----
    w("## 4. Per-arm numbers: table against mine")
    w("")
    w("The `base` and `sft` entries are in both table files. Are they identical in the two `.json` files? base: %s; sft: %s; the pair sft minus base: %s. "
      "Below, `base` and `sft` are shown once (compared against both files; the column \"equal\" covers both)."
      % (yesno(shared_same["base"]), yesno(shared_same["sft"]), yesno(shared_same["sft minus base"])))
    w("")
    w("Column \"table .md\": the number as printed in the `.md` file (percent, or turns). Column \"equal\": the `.json` value equals mine within 1e-12 and the `.md` number is my value rounded. "
      "Interval rows are in section 5. Rows marked (extra) are in the tables but not on my task list; see section 8.")
    w("")

    def md_lookup(name, metric, arm):
        for r in md_rows:
            if r["table"] == name and r["metric"] == metric and r["arm"] == arm:
                return r
        return None

    arm_rows_spec = [
        ("Episodes", "n_episodes", "Episodes"),
        ("Harness errors", "n_harness_errors", "Harness errors"),
        ("Families", "n_families", None),
        ("Pass rate (family-weighted)", "pass_rate", "Pass rate"),
        ("Pass rate, outcome grant", "pass_rate_by_outcome.grant", "grant"),
        ("Pass rate, outcome approval", "pass_rate_by_outcome.approval", "approval"),
        ("Pass rate, outcome already", "pass_rate_by_outcome.already", "already"),
        ("Pass rate, outcome inactive", "pass_rate_by_outcome.inactive", "inactive"),
        ("Pass rate, styles 1 to 3", "pass_rate_seen", "Seen styles"),
        ("Pass rate, style 4", "pass_rate_held_out", "Held-out style"),
        ("Scenarios with 8 equal rewards", "zero_variance_equal_reward", "Groups with equal rewards"),
        ("Scenarios all pass or all fail", "zero_variance_equal_passfail", "Groups with equal pass or fail"),
        ("Malformed / all assistant messages", "invalid_call_rate", "Malformed messages"),
        ("Pass@8", "pass_at_8", "Pass@8"),
        ("Pass^1", "pass_hat_k.1", "Pass^1"),
        ("Pass^2", "pass_hat_k.2", "Pass^2"),
        ("Pass^4", "pass_hat_k.4", "Pass^4"),
        ("Pass^8", "pass_hat_k.8", "Pass^8"),
        ("Scenarios left out of Pass^1 / ^2 / ^4 / ^8", "n_scenarios_left_out", None),
        ("All wordings pass", "all_wordings_pass", "All wordings pass"),
        ("Mean turns", "turn_stats.mean_turns", "Mean turns"),
        ("Ended at the turn limit", "turn_stats.share_at_turn_limit", "Ended at the turn limit"),
        ("Closed the request (share_closed)", "turn_stats.share_closed", None),
        ("closed_correctly", "outcome_labels.closed_correctly", "closed_correctly"),
        ("closed_wrongly", "outcome_labels.closed_wrongly", "closed_wrongly"),
        ("never_closed", "outcome_labels.never_closed", "never_closed"),
        ("malformed_only", "outcome_labels.malformed_only", "malformed_only"),
        ("(extra) variance due to wording", "variance_shares.wording", "Variance due to wording (extra)"),
        ("(extra) variance due to sampling", "variance_shares.sampling", None),
        ("(extra) wording share if sampling only", "variance_shares.wording_if_sampling_only", "Variance due to wording if sampling only (extra)"),
    ]

    def rows_for(arm, path):
        return [r for r in rows if r["arm"] == arm and r["metric"] == path]

    for arm in ARMS:
        names = [n for n in ("seed1", "seed2") if arm in TABLE_ARMS[n]]
        w("### %s (in table file%s: %s)" % (arm, "s" if len(names) > 1 else "", ", ".join("base_sft_grpo_" + n for n in names)))
        w("")
        w("| Metric | Arm | Table .json | Table .md | Mine | Equal |")
        w("|---|---|---|---|---|---|")
        for label, path, mdmetric in arm_rows_spec:
            if path == "n_scenarios_left_out":
                rs = [r for k in KS for r in rows_for(arm, "n_scenarios_left_out.%d" % k)]
                tv = " / ".join(str(r["table_value"]) for r in rs[:4])
                mv = " / ".join(str(M[arm]["n_scenarios_left_out"][k]) for k in KS)
                w("| %s | %s | %s | - | %s | %s |" % (label, arm, tv, mv, yesno(all(r["equal"] for r in rs))))
                continue
            rs = rows_for(arm, path)
            if not rs:
                w("| %s | %s | (not in the table) | - | - | NO |" % (label, arm))
                continue
            tvals = sorted(set(show_table(r["table_value"]) for r in rs))
            mdr = [md_lookup(n, mdmetric, arm) for n in names] if mdmetric else []
            mdr = [x for x in mdr if x is not None]
            mdtxt = " / ".join(sorted(set(x["printed"] for x in mdr))) if mdr else "-"
            ok = all(r["equal"] for r in rs) and all(x["equal_mine"] for x in mdr)
            w("| %s | %s | %s | %s | %s | %s |" % (label, arm, " / ".join(tvals), mdtxt, show_mine(rs[0]["mine"]), yesno(ok)))
        # reward values
        rd = M[arm]["reward_distribution"]
        rs = [r for r in rows if r["arm"] == arm and r["metric"].startswith("reward_distribution.")]
        mdr = [r for r in md_rows if r["arm"] == arm and r["metric"].startswith("attempts with reward")]
        tdist = {}
        for r in rs:
            if r["table_value"] is not None:
                tdist[int(r["metric"].split(".")[1])] = r["table_value"]
        ok = all(r["equal"] for r in rs) and all(x["equal_mine"] for x in mdr) and len(mdr) == len(rd) * len(names)
        w("| Attempts per reward value | %s | %s | (the same list, %d printed numbers checked) | %s | %s |"
          % (arm, ", ".join("%d: %d" % (k, v) for k, v in sorted(tdist.items())), len(mdr),
             ", ".join("%d: %d" % (k, v) for k, v in rd.items()), yesno(ok)))
        # identity
        rs = [r for r in rows if r["arm"] == arm and r["kind"] == "identity"]
        mdr = [r for r in md_rows if r["arm"] == arm and r["kind"] == "identity"]
        w("| Split, rule set, tools, prompt, text and config hash | %s | %s | %s | the same values, read from the `split`, `ruleset`, `toolset`, `prompt`, `text_sha256`, `config_sha256` fields of all records | %s |"
          % (arm, "; ".join(sorted(set("%s=%s" % (r["metric"], show_table(r["table_value"])) for r in rs))),
             "; ".join(sorted(set(r["printed"] for r in mdr))), yesno(all(r["equal"] for r in rs) and all(r["equal_mine"] for r in mdr))))
        w("")
        m = M[arm]
        w("The counts behind the main rows for `%s` (from `results/main/%s_dev/episodes.jsonl`): passed attempts %d of %d; per outcome %s; styles 1 to 3: %d of %d; style 4: %d of %d; "
          "scenarios with 8 equal rewards %d of %d; all pass %d, all fail %d of %d; malformed messages %d of %d assistant messages; scenarios with at least one pass %d of %d; "
          "attempts ended at the turn limit %d; closed %d; labels %s."
          % (arm, arm, m["n_passed"], m["n_episodes"],
             ", ".join("%s %d of %d" % (o, m["by_outcome_counts"][o][0], m["by_outcome_counts"][o][1]) for o in OUTCOMES),
             m["seen_counts"][0], m["seen_counts"][1], m["held_out_counts"][0], m["held_out_counts"][1],
             m["n_scen_equal_reward"], m["n_scenarios"], m["n_scen_all_pass"], m["n_scen_all_fail"], m["n_scenarios"],
             m["n_bad"], m["n_msgs"], m["n_scen_any_pass"], m["n_scenarios"], m["n_at_turn_limit"], m["n_closed"],
             ", ".join("%s %d" % (k, v) for k, v in m["label_counts"].items())))
        w("")

    w("Notes on the readings, and on what this data can and cannot tell apart:")
    w("")
    w("- Shares over attempts or scenarios (equal rewards, turn limit, end labels, mean turns) are the same whether taken plainly or family-weighted, because every family has 4 scenarios x 8 valid attempts.")
    w("- The malformed-message share is different in the two readings. Pooled (what I used, what the plan's words say): %s. Mean of per-family ratios: %s. The tables hold the pooled one."
      % (", ".join("%s %.4f%%" % (a, 100 * float(M[a]["invalid_call_rate"])) for a in ARMS),
         ", ".join("%s %.4f%%" % (a, 100 * float(M[a]["invalid_call_rate_family_weighted"])) for a in ARMS)))
    w("- malformed_only is 0 in every arm, and no attempt has all its messages malformed (%s). So my definition of that label is not tested by this data: any definition that needs \"all messages malformed\" gives 0."
      % ", ".join("%s %d" % (a, M[a]["n_all_messages_malformed"]) for a in ARMS))
    w("- closed_correctly equals the pass rate because no passed attempt is unclosed (%s) and the family sizes are equal."
      % ", ".join("%s %d" % (a, M[a]["n_passed_not_closed"]) for a in ARMS))
    if md_ties:
        w("- Printed numbers that sit exactly on a rounding tie (I count them as equal; the `.md` shows the tie rounded to the even digit, which is how Python formats a float): %s."
          % "; ".join("%s, %s: exact value %s, printed %s" % (a2, mt, ("%.4f" % mv).rstrip("0"), pr) for (a2, mt, pr, mv) in md_ties))
    else:
        w("- No printed number sits exactly on a rounding tie.")
    w("")

    # ---- comparisons ----
    w("## 5. Comparisons and intervals")
    w("")
    w("### 5.1 Differences and t-intervals (no random draws)")
    w("")
    w("In points (percent). \"Table\" is the `.json` value x 100; the `.md` print is beside it. Tolerance for the t ends: 1e-9 on the fraction scale.")
    w("")
    w("| Comparison | Table file | Difference: table (.md) | Difference: mine | Equal | t-interval: table (.md) | t-interval: mine | Equal |")
    w("|---|---|---|---|---|---|---|---|")
    for name in ("seed1", "seed2"):
        for e in T[name]["pairs"]:
            key = (e["a"], e["b"])
            label = "%s minus %s" % (e["b"], e["a"])
            p = pair[key]
            r_d = [r for r in rows if r["table"] == name and r["arm"] == label and r["metric"] == "pair.diff"][0]
            r_t = [r for r in rows if r["table"] == name and r["arm"] == label and r["metric"] in ("pair.t_lo", "pair.t_hi")]
            md_d = md_lookup(name, "Difference in points", label)
            md_t = [md_lookup(name, "t-interval, lower end", label), md_lookup(name, "t-interval, upper end", label)]
            w("| %s | %s | %+.10f (%s) | %+.10f (= %+d/%d) | %s | %+.6f to %+.6f (%s to %s) | %+.6f to %+.6f | %s |"
              % (label, name, 100 * e["diff"], md_d["printed"], 100 * float(p["diff"]), p["diff"].numerator, p["diff"].denominator,
                 yesno(r_d["equal"] and md_d["equal_mine"]),
                 100 * e["t_lo"], 100 * e["t_hi"], md_t[0]["printed"], md_t[1]["printed"], 100 * p["t"][0], 100 * p["t"][1],
                 yesno(all(r["equal"] for r in r_t) and all(x["equal_mine"] for x in md_t))))
    w("")
    w("### 5.2 Bootstrap intervals (random draws: the table used seed 0, I used seed %d)" % MY_SEED)
    w("")
    w("The table code and my code draw different random families, so with my own seed the ends are not expected to be equal. Two questions: is the gap inside the noise that 10,000 draws leave, and does my code give the table's ends when it uses the table's seed?")
    w("")
    w("**Table A: the ends.** In points. `base`, `sft` and `sft minus base` have the same values in both table files (checked), so each end is shown once: %d ends. \"Mine, seed 0\": my code with "
      "`numpy.random.default_rng(0).integers(0, 48, size=(10000, 48))`, a fresh generator for every interval, then the 2.5th and 97.5th percentile." % len(mc_unique))
    w("")
    w("| Quantity | End | Table (seed 0) | Mine, my seed %d | Table minus mine | Mine, seed 0 | Mine with seed 0 equals the table (1e-12) |" % MY_SEED)
    w("|---|---|---|---|---|---|---|")
    for r in mc_unique:
        w("| %s | %s | %.4f | %.4f | %+.4f | %.4f | %s |"
          % (r["what"], "lower" if r["end"] == "lo" else "upper", 100 * r["table_value"], 100 * r["mine"],
             100 * (r["table_value"] - r["mine"]), 100 * r["mine0"], yesno(r["equal0"])))
    w("")
    n_u = len(mc_unique)
    n_eq0 = len([r for r in mc_unique if r["equal0"]])
    w("With the table's seed my code gives %d of %d ends exactly (largest absolute difference %.1e on the fraction scale). I found the way to draw by trial: `seed0_trials.py` in this folder tries seven ways to draw 48 families with seed 0 "
      "(numpy's default generator with `integers` or `choice`; the old `RandomState` with `randint` or `choice`; Python's `random` with `randrange`, `choices` or `random()`); "
      "the two with numpy's default generator give all 18 ends, the other five give 4 to 7 of the 18 (single ends can agree by chance, because the ends sit on a grid of 1/1536; output in `seed0_trials.txt`)."
      % (n_eq0, n_u, max(abs(r["table_value"] - r["mine0"]) for r in mc_unique)))
    w("")
    w("The cause that I give for the gap between the table's ends and mine with seed %d is the seed. Evidence: the same code of mine with seed 0 gives %d of %d table ends exactly. "
      "What would refute it: one end that still differed with seed 0%s"
      % (MY_SEED, n_eq0, n_u, "; there is none." if n_eq0 == n_u else "; %d ends do differ, so the seed is not the whole explanation." % (n_u - n_eq0)))
    w("")
    w("**Table B: how much noise 10,000 draws leave.** Three measures, all mine:")
    w("")
    w("1. **Exact bootstrap.** Every family pass rate is a whole number of passes out of 32, so the mean of 48 drawn families is a whole number of passes out of 1536, and its distribution can be computed exactly "
      "(the 48-fold convolution of the distribution of one draw). This gives the percentile ends that an endless number of draws would give. One grid step is 1/1536 = %.4f points." % (100.0 / 1536))
    w("2. **Expected band for 10,000 draws.** The number of draws below a point is binomial, so the 2.5%% end of a 10,000-draw run sits at an exact level of 2.5%% +- 1.96 x %.3f%% (2.19%% to 2.81%%); "
      "I give the exact quantiles at those two levels. About 95%% of seeds should land inside this band, so about 1 end in 20 is expected outside it by chance alone. "
      "A wider band (+- 3.29 standard errors, 99.9%%) is my line for \"outside the Monte Carlo noise\"." % (100 * exact[ARMS[0]]["se_level"]))
    w("3. **200 further seeds** (seeds %d to %d, 10,000 draws each): the smallest and largest end that I saw, and the standard deviation." % (NOISE_SEEDS[0], NOISE_SEEDS[-1]))
    w("")
    w("| Quantity | End | Table (seed 0) | Exact bootstrap end | Table minus exact, grid steps | 95% band for 10,000 draws | 99.9% band | 200 seeds: min to max | SD over 200 seeds | Table inside the 95% band | inside the 99.9% band | inside the range of 200 seeds |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in mc_unique:
        steps = (r["table_value"] - r["exact"]) / r["step"]
        w("| %s | %s | %.4f | %.4f | %+.2f | %.4f to %.4f | %.4f to %.4f | %.4f to %.4f | %.4f | %s | %s | %s |"
          % (r["what"], "lower" if r["end"] == "lo" else "upper", 100 * r["table_value"], 100 * r["exact"], steps,
             100 * r["band"][0], 100 * r["band"][1], 100 * r["band_wide"][0], 100 * r["band_wide"][1],
             100 * r["seed_min"], 100 * r["seed_max"], 100 * r["seed_sd"],
             yesno(r["inside_band"]), yesno(r["inside_wide"]), yesno(r["inside_seeds"])))
    w("")
    max_gap = max(abs(r["table_value"] - r["mine"]) for r in mc_unique)
    max_sd = max(r["seed_sd"] for r in mc_unique)
    max_exact_gap = max(abs(r["table_value"] - r["exact"]) for r in mc_unique)
    u_out95 = [r for r in mc_unique if not r["inside_band"]]
    u_outwide = [r for r in mc_unique if not r["inside_wide"]]
    u_outseeds = [r for r in mc_unique if not r["inside_seeds"]]
    mine_out95 = [r for r in mc_unique if not r["mine_inside_band"]]

    def name_ends(rs):
        return "; ".join("%s, %s end" % (r["what"], "lower" if r["end"] == "lo" else "upper") for r in rs)
    exr = [r for r in mc_unique if r["what"] == "grpo_seed2 minus sft" and r["end"] == "lo"][0]
    w("Example, the lower end of grpo_seed2 minus sft (the end that is nearest to zero): the table has %+.2f; the exact bootstrap gives %+.2f; my seed gives %+.2f; over my 200 further seeds it ran from %+.2f to %+.2f."
      % (100 * exr["table_value"], 100 * exr["exact"], 100 * exr["mine"], 100 * exr["seed_min"], 100 * exr["seed_max"]))
    w("")
    w("Reading:")
    w("")
    w("- **Noise to expect:** one standard deviation of an end over seeds is %.3f to %.3f points (0 where every seed gives the same grid point); the 95%% band is at most %.2f points wide. So two runs with different seeds can differ by a few tenths of a point in an end, and the first decimal of an end is not stable."
      % (100 * min(r["seed_sd"] for r in mc_unique), 100 * max_sd, 100 * max(r["band"][1] - r["band"][0] for r in mc_unique)))
    w("- **Gap found:** the largest gap between a table end and my end (seed %d) is %.3f points; between a table end and the exact bootstrap end %.3f points (%.1f grid steps)."
      % (MY_SEED, 100 * max_gap, 100 * max_exact_gap, max_exact_gap * 1536))
    w("- Table ends inside the 95%% band: %d of %d%s" % (n_u - len(u_out95), n_u, "." if not u_out95 else ". Outside: " + name_ends(u_out95) + "."))
    w("- Table ends inside the 99.9%% band: %d of %d%s" % (n_u - len(u_outwide), n_u, "." if not u_outwide else ". Outside: " + name_ends(u_outwide) + "."))
    w("- Table ends inside the range of my 200 seeds: %d of %d%s" % (n_u - len(u_outseeds), n_u, "." if not u_outseeds else ". Outside: " + name_ends(u_outseeds) + "."))
    w("- My own ends (seed %d) inside the 95%% band: %d of %d%s" % (MY_SEED, n_u - len(mine_out95), n_u, "." if not mine_out95 else ". Outside: " + name_ends(mine_out95) + " (about 1 in 20 is expected outside)."))
    w("- A check of the band itself: of my 200 x %d seed-ends, %d (%.1f%%) fall inside their 95%% band. The band is built to hold about 95%%, and holds more where the distribution has few grid points near the end."
      % (n_u, sum(r["seeds_inside_band"] for r in mc_unique), 100.0 * sum(r["seeds_inside_band"] for r in mc_unique) / (200.0 * n_u)))
    lo_rows = [r for r in mc_unique if r["end"] == "lo" and " minus " in r["what"]]
    flips = [r for r in lo_rows if (r["table_value"] > 0) != (r["exact"] > 0)]
    w("- The exact bootstrap ends are a seed-free version of the table's intervals. The widest shift against the table is %.2f points. Lower ends of the %d comparisons that are above zero in the table: %d; above zero in the exact bootstrap: %d; comparisons where the two disagree about the sign of the lower end: %d."
      % (100 * max_exact_gap, len(lo_rows), len([r for r in lo_rows if r["table_value"] > 0]), len([r for r in lo_rows if r["exact"] > 0]), len(flips)))
    w("")
    w("The `.md` files print these ends with one decimal. Printed number against the `.json` value beside it (a check of the table files themselves): %d printed numbers checked in all, %d do not match their `.json` value."
      % (len([r for r in md_rows if r["equal_json"] is not None]), len(md_json_bad)))
    w("")
    w("Exact bootstrap intervals (no random draws) beside the table's, in points:")
    w("")
    w("| Quantity | Table interval (seed 0) | Exact bootstrap interval |")
    w("|---|---|---|")
    for qn in quantities:
        label = qn if isinstance(qn, str) else "%s minus %s" % (qn[1], qn[0])
        tl = [r for r in mc_unique if r["what"] == label and r["end"] == "lo"][0]["table_value"]
        th = [r for r in mc_unique if r["what"] == label and r["end"] == "hi"][0]["table_value"]
        fmt = "| %s | %.2f to %.2f | %.2f to %.2f |" if isinstance(qn, str) else "| %s | %+.2f to %+.2f | %+.2f to %+.2f |"
        w(fmt % (label, 100 * tl, 100 * th, 100 * exact[qn]["lo"], 100 * exact[qn]["hi"]))
    w("")

    # ---- questions ----
    w("## 6. The questions (a) to (e)")
    w("")
    w("### (a) Is the family-weighted pass rate equal to the plain share of passed attempts here?")
    w("")
    eq_a = all(M[a]["pass_rate"] == M[a]["plain_pass_share"] for a in ARMS)
    w("%s, exactly (compared as fractions): %s." % ("Yes" if eq_a else "No",
      "; ".join("%s: family-weighted %d/%d, plain %d/%d" % (a, M[a]["pass_rate"].numerator, M[a]["pass_rate"].denominator,
                                                          M[a]["plain_pass_share"].numerator, M[a]["plain_pass_share"].denominator) for a in ARMS)))
    w("(Fractions are shown reduced; unreduced they are %s passed attempts of 1536.)" % ", ".join("%s %d" % (a, M[a]["n_passed"]) for a in ARMS))
    w("")
    w("Why: every family has the same number of valid attempts (32 = 4 styles x 8 attempts; no harness error in any arm). The mean of 48 shares that all have the denominator 32 is the total number of passes divided by 48 x 32 = 1536. "
      "Example: two families with 8/32 and 24/32 give (0.25 + 0.75) / 2 = 0.5, and 32 passes of 64 attempts is also 0.5. "
      "The two would differ as soon as families had different numbers of valid attempts (a harness error, or a family with fewer styles). "
      "The same holds for the per-outcome and per-style rates here (12 families x 32, 48 x 24, 48 x 8 attempts). "
      "What stays different is the unit of the interval: the bootstrap and the t-interval treat the 48 families as the units, not the 1536 attempts. "
      "What would refute this explanation: a family with a number of valid attempts other than 32; the check in section 2 found none (families with exactly 32 valid attempts: %s)."
      % ", ".join("%s %d of 48" % (a2, sum(1 for f in fam_order if M[a2]["fam_n"][f] == 32)) for a2 in ARMS))
    w("")

    for a, b in (("sft", "grpo_seed1"), ("sft", "grpo_seed2")):
        key = (a, b)
        qq = q[key]
        p = pair[key]
        w("### %s minus sft" % b)
        w("")
        w("Counted from `results/main/sft_dev/episodes.jsonl` and `results/main/%s_dev/episodes.jsonl`: passes of 32 per family in each arm." % b)
        w("")
        w("**(b) Families whose pass rate changed by more than 10 points** (more than 3.2 passes of 32, so 4 or more passes): %d of 48 (%d up, %d down)."
          % (len(qq["moved"]), sum(1 for _f, x, y in qq["moved"] if y > x), sum(1 for _f, x, y in qq["moved"] if y < x)))
        w("")
        if qq["moved"]:
            w("| Family | Expected outcome | Facts in dev.jsonl | Two people with one name | sft passes of 32 | %s passes of 32 | Change in points | sft passes of 8 by style 1/2/3/4 | %s passes of 8 by style 1/2/3/4 |" % (b, b))
            w("|---|---|---|---|---|---|---|---|---|")
            for f, x, y in sorted(qq["moved"], key=lambda t: (-(t[2] - t[1]), t[0])):
                fa = fams[f]["facts"]
                facts = ", ".join("%s=%s" % (k, str(v).lower()) for k, v in sorted(fa.items()))
                sa = "/".join(str(M[a]["scen_pass"][(f, s)]) for s in (1, 2, 3, 4))
                sb = "/".join(str(M[b]["scen_pass"][(f, s)]) for s in (1, 2, 3, 4))
                w("| %s | %s | %s | %s | %d | %d | %+.1f | %s | %s |" % (f, fams[f]["outcome"], facts, "yes" if fams[f]["homonym"] else "no", x, y, 100.0 * (y - x) / 32, sa, sb))
            w("")
        w("**(e) Families with a difference of exactly zero**: %d of 48. Up: %d. Down: %d." % (qq["n_zero"], qq["n_pos"], qq["n_neg"]))
        w("")
        w("**(c) Leaving one family out.** All 48 families: %s points (= %+d/%d)." % (pts(p["diff"], 2, True), p["diff"].numerator, p["diff"].denominator))
        w("")
        w("| Left out | Family | Its change (passes of 32) | Difference over the other 47 families, points | Exact bootstrap interval (47 families) | t-interval (46 degrees of freedom) | Exact sign test, two-sided | Sign-flip test, two-sided |")
        w("|---|---|---|---|---|---|---|---|")
        for nm, lab in (("largest_gain", "the largest gain"), ("largest_loss", "the largest loss")):
            lo_ = qq["loo"][nm]
            tie_txt = "" if len(lo_["ties"]) == 1 else " (tied with %s; leaving out any one of them gives the same number)" % ", ".join(t for t in lo_["ties"] if t != lo_["family"])
            if nm == "largest_loss" and lo_["d_int"] >= 0:
                lab = "the smallest change (no family lost)"
            w("| %s | %s%s | %+d | %s (= %+d/%d) | %s to %s | %s to %s | %.3g (%d up, %d down, %d zero) | %.3g |"
              % (lab, lo_["family"], tie_txt, lo_["d_int"], pts(lo_["diff"], 2, True), lo_["diff"].numerator, lo_["diff"].denominator,
                 pts(lo_["exact_boot"][0], 2, True), pts(lo_["exact_boot"][1], 2, True), pts(lo_["t"][0], 2, True), pts(lo_["t"][1], 2, True),
                 float(lo_["sign"]["p_two_sided"]), lo_["sign"]["pos"], lo_["sign"]["neg"], lo_["sign"]["zero"], float(lo_["perm"]["p_two_sided"])))
        w("")
    w("### Where the change sits, by expected outcome (a reading aid, not a comparison of the plan)")
    w("")
    w("Passed attempts of 384 per expected outcome (12 families x 32 attempts), from the same four files. The per-outcome pass rates are columns of the tables; the differences below are plain subtractions of those columns, with no interval.")
    w("")
    w("| Expected outcome | Families | sft | grpo_seed1 | grpo_seed1 minus sft | grpo_seed2 | grpo_seed2 minus sft |")
    w("|---|---|---|---|---|---|---|")
    for o in OUTCOMES:
        cs = [M[a2]["by_outcome_counts"][o][0] for a2 in ("sft", "grpo_seed1", "grpo_seed2")]
        w("| %s | %d | %d | %d | %s | %d | %s |" % (o, M["sft"]["by_outcome_counts"][o][2], cs[0], cs[1], signed(cs[1] - cs[0]), cs[2], signed(cs[2] - cs[0])))
    tot = [M[a2]["n_passed"] for a2 in ("sft", "grpo_seed1", "grpo_seed2")]
    w("| all (of 1536) | 48 | %d | %d | %s | %d | %s |" % (tot[0], tot[1], signed(tot[1] - tot[0]), tot[2], signed(tot[2] - tot[0])))
    ina = [M[a2]["by_outcome_counts"]["inactive"][0] for a2 in ("sft", "grpo_seed1", "grpo_seed2")]
    rest = [tot[i] - ina[i] for i in range(3)]
    w("| all but inactive (of 1152) | 36 | %d | %d | %s | %d | %s |" % (rest[0], rest[1], signed(rest[1] - rest[0]), rest[2], signed(rest[2] - rest[0])))
    w("")
    w("In words: grpo_seed1 has %s passed attempts against sft: %s in the 12 families with the expected outcome inactive and %s in the other 36 families (%s points over those 36). "
      "grpo_seed2 has %s against sft: %s in the 12 inactive families and %s in the other 36 (%s points over those 36)."
      % (signed(tot[1] - tot[0]), signed(ina[1] - ina[0]), signed(rest[1] - rest[0]), "%+.2f" % (100.0 * (rest[1] - rest[0]) / 1152),
         signed(tot[2] - tot[0]), signed(ina[2] - ina[0]), signed(rest[2] - rest[0]), "%+.2f" % (100.0 * (rest[2] - rest[0]) / 1152)))
    w("")
    w("### (d) A second view beside the bootstrap: sign test and paired permutation test")
    w("")
    w("What I computed, on the 48 family differences d = passes of 32 in arm b minus passes of 32 in arm a:")
    w("")
    w("- **Exact sign test.** Families with d = 0 are dropped. Of the n families left, k are up. Under \"up and down are equally likely\" the number of ups is binomial(n, 1/2). "
      "Two-sided p = 2 x P(X >= max(up, down)), capped at 1; one-sided p (for a gain) = P(X >= up). Computed with whole-number binomial coefficients.")
    w("- **Paired permutation (sign-flip) test.** Under \"the two arms do not differ\" the two labels inside a family can be swapped, which turns d into -d. All 2^n sign patterns of the n non-zero differences are equally likely. "
      "Statistic: the sum of the d (the mean difference times 1536). Two-sided p = the share of the 2^n patterns whose absolute sum is at least the observed absolute sum (the observed pattern counts); one-sided = the share whose sum is at least the observed sum. "
      "Counted exactly over all 2^n patterns by dynamic programming on whole-number sums, and checked with 1,000,000 random patterns (seed %d)." % PERM_SEED)
    w("")
    w("| Comparison | Up | Down | Zero | Sign test p, two-sided | Sign test p, one-sided | Sign-flip p, two-sided (exact) | Sign-flip p, one-sided (exact) | Patterns at least as extreme / all | Monte Carlo check, two-sided |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for key in PAIRS:
        qq = q[key]
        st, pt = qq["sign"], qq["perm"]
        w("| %s minus %s | %d | %d | %d | %.3g | %.3g | %.3g | %.3g | %d / 2^%d | %d of %d = %.3g |"
          % (key[1], key[0], st["pos"], st["neg"], st["zero"], float(st["p_two_sided"]), float(st["p_one_sided_gain"]),
             float(pt["p_two_sided"]), float(pt["p_one_sided_gain"]), pt["n_patterns_abs_ge"], pt["n_nonzero"],
             qq["perm_mc"][0], qq["perm_mc"][1], qq["perm_mc"][0] / qq["perm_mc"][1]))
    w("")
    ex = q[("sft", "grpo_seed1")]["perm"]
    w("Example, grpo_seed1 minus sft: %d families have a non-zero difference, so there are 2^%d = %d sign patterns; %d of them give a sum at least as far from zero as the observed %+d passes; %d / %d = %.5f."
      % (ex["n_nonzero"], ex["n_nonzero"], ex["n_patterns"], ex["n_patterns_abs_ge"], ex["observed_sum"], ex["n_patterns_abs_ge"], ex["n_patterns"],
         ex["n_patterns_abs_ge"] / ex["n_patterns"]))
    w("")
    w("Limits of these two tests: they ask only whether the family differences are balanced around zero. They say nothing about the size of the gain. The sign test ignores how large each change is; the sign-flip test is driven by the few families with large changes. "
      "No correction is made for running five comparisons. The 48 families are treated as independent draws; attempts inside a family are not treated as independent.")
    w("")

    # ---- family table ----
    w("## 7. Passes of 32 per family, all four arms")
    w("")
    w("From the four `episodes.jsonl` files (`passed` true, per `family_id`); outcome from `data/scenarios/dev.jsonl`.")
    w("")
    w("| Family | Expected outcome | base | sft | grpo_seed1 | grpo_seed2 | seed1 minus sft | seed2 minus sft |")
    w("|---|---|---|---|---|---|---|---|")
    for i, f in enumerate(fam_order):
        w("| %s | %s | %d | %d | %d | %d | %s | %s |" % (f, fams[f]["outcome"], C["base"][i], C["sft"][i], C["grpo_seed1"][i], C["grpo_seed2"][i],
                                                         signed(C["grpo_seed1"][i] - C["sft"][i]), signed(C["grpo_seed2"][i] - C["sft"][i])))
    w("| Sum of 1536 | | %d | %d | %d | %d | %+d | %+d |" % (C["base"].sum(), C["sft"].sum(), C["grpo_seed1"].sum(), C["grpo_seed2"].sum(),
                                                             C["grpo_seed1"].sum() - C["sft"].sum(), C["grpo_seed2"].sum() - C["sft"].sum()))
    w("")

    # ---- extras ----
    w("## 8. Extras: numbers that are in the tables but not on my task list")
    w("")
    w("### 8.1 Variance inside families due to wording")
    w("")
    w("The plan does not define this column. My reading of its name: inside each family, split the sum of squares of the 32 pass marks (0 or 1) into the part between the 4 styles and the part inside the styles; add both over the 48 families; "
      "wording share = between / total. \"If sampling only\" = (4 - 1) / (32 - 1) = 3/31, the share that chance alone gives.")
    w("")
    w("| Arm | Table: wording | Mine: wording | Equal (1e-12) | Table: if sampling only | Mine | Equal | Another reading (mean of the per-family ratios, families with some variance) |")
    w("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        names = [n for n in ("seed1", "seed2") if arm in TABLE_ARMS[n]]
        tv = {e["arm"]: e for e in T[names[0]]["arms"]}[arm]["variance_shares"]
        mv = M[arm]["variance_shares"]
        w("| %s | %s | %s | %s | %s | %s | %s | %.6f (%d families) |"
          % (arm, repr(tv["wording"]), repr(float(mv["wording"])), yesno(abs(tv["wording"] - float(mv["wording"])) <= TOL_JSON),
             repr(tv["wording_if_sampling_only"]), repr(float(mv["wording_if_sampling_only"])),
             yesno(abs(tv["wording_if_sampling_only"] - float(mv["wording_if_sampling_only"])) <= TOL_JSON),
             float(mv["wording_mean_of_family_ratios"]), mv["n_families_with_variance"]))
    w("")
    w("### 8.2 Smallest detectable effect")
    w("")
    same_extremes = []
    for arm in ARMS:
        names = [n for n in ("seed1", "seed2") if arm in TABLE_ARMS[n]]
        sj = {e["arm"]: e for e in T[names[0]]["arms"]}[arm]["smallest_detectable_effect"]
        if abs(sj["mde_lowest_split"] - 2.8 * sde[arm]["min_family_se"] / math.sqrt(2)) <= TOL_JSON and \
                abs(sj["mde_highest_split"] - 2.8 * sde[arm]["max_family_se"] / math.sqrt(2)) <= TOL_JSON:
            same_extremes.append(arm)
    w("Plan, section 5: \"2.8 standard errors of a difference of two pass rates, from one arm's attempts, split at random into two halves 200 times\". The `.md` footnote adds: \"a scenario needs 2 samples or more\". "
      "These words leave open how the halves are drawn and how 200 splits become one standard error, and the result depends on random splits. Three checks, in the order in which I did them:")
    w("")
    w("1. **No random numbers.** Split the 8 attempts of every scenario into two halves of 4. Over all possible splits, the variance of (pass rate of half A - pass rate of half B) is exactly the sum over scenarios of 4 p (1 - p) / 7, divided by 192 squared "
      "(p = the scenario's share of passes; the number of passes in a half is hypergeometric). Its square root is the standard error for two half-size arms; divided by sqrt(2) it is the one for two full-size arms; times 2.8 it is the smallest detectable effect.")
    w("2. **200 random splits with my own seed (%d)**, turned into a standard error in two ways: (i) the root mean square over the splits of the half difference itself; (ii) per split, the standard deviation over the 48 families of the family's half difference / sqrt(48), which gives one value per split. "
      "With my seed, the lowest and the highest single split of (ii) are equal to the table's `mde_lowest_split` and `mde_highest_split` within 1e-12 in these arms: %s "
      "(arms with few scenarios of mixed results, where few values are possible: scenarios with both passes and fails: %s). That told me that the table's single-split number is (ii)."
      % (SPLIT_SEED, ", ".join(same_extremes) if same_extremes else "none",
         ", ".join("%s %d" % (a2, M[a2]["n_scenarios"] - M[a2]["n_scen_all_pass"] - M[a2]["n_scen_all_fail"]) for a2 in ARMS)))
    w("3. **200 splits with the table's seed (0).** `numpy.random.default_rng(0)`; for each split, for each family in the order of `dev.jsonl` and each style 1 to 4, one `rng.permutation(8)` over the attempts in the order of their number; the first 4 are half A. "
      "Per split: the standard deviation (n - 1) over the 48 families of (family pass rate in A - in B), divided by sqrt(48). `se_halves`: the root mean square of that over the 200 splits. `se` = `se_halves` / sqrt(2). `mde` = 2.8 x `se`. "
      "Lowest and highest split: 2.8 x that split's value / sqrt(2). I found the way to draw and to pool by trial (`seed0_trials.py`: seven ways to write the draw of the halves, of which four give identical values and the table and three do not; "
      "three ways to pool, of which only the root mean square gives the table; output in `seed0_trials.txt`).")
    w("")
    w("Check 3: table against my code with seed 0 (tolerance 1e-12):")
    w("")
    w("| Arm | Number | Table .json | Mine, seed 0 | Equal |")
    w("|---|---|---|---|---|")
    for arm in ARMS:
        names = [n for n in ("seed1", "seed2") if arm in TABLE_ARMS[n]]
        for k2 in ("se_halves", "se", "mde", "mde_lowest_split", "mde_highest_split"):
            rs = [r for r in rows if r["arm"] == arm and r["metric"] == "smallest_detectable_effect." + k2]
            tvals = sorted(set(repr(r["table_value"]) for r in rs))
            w("| %s | %s | %s | %s | %s |" % (arm, k2, " / ".join(tvals), repr(sde0[arm][k2]), yesno(bool(rs) and all(r["equal"] for r in rs))))
    w("")
    w("Checks 1 and 2: how far the table's value is from a value that needs no random numbers, and what another seed gives (`se_halves`; `mde` in points):")
    w("")
    w("| Arm | Table `se_halves` | Exact value over all splits (check 1) | Table / exact | Mine, seed %d, (ii) pooled by root mean square as in the table | Mine, seed %d, (i) | Table `mde`, points | 2.8 x exact / sqrt(2), points | Table: one split alone, points | Mine, seed %d, (ii): one split alone, points |" % (SPLIT_SEED, SPLIT_SEED, SPLIT_SEED))
    w("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        names = [n for n in ("seed1", "seed2") if arm in TABLE_ARMS[n]]
        sj = {e["arm"]: e for e in T[names[0]]["arms"]}[arm]["smallest_detectable_effect"]
        se_x = M[arm]["sde_exact"]["se_halves"]
        w("| %s | %.6f | %.6f | %.3f | %.6f | %.6f | %.3f | %.3f | %.2f to %.2f | %.2f to %.2f |"
          % (arm, sj["se_halves"], se_x, sj["se_halves"] / se_x, sde[arm]["rms_of_family_se"], sde[arm]["rms_of_half_difference"],
             100 * sj["mde"], 100 * M[arm]["sde_exact"]["mde"], 100 * sj["mde_lowest_split"], 100 * sj["mde_highest_split"],
             100 * 2.8 * sde[arm]["min_family_se"] / math.sqrt(2), 100 * 2.8 * sde[arm]["max_family_se"] / math.sqrt(2)))
    w("")
    ratios = []
    for arm in ARMS:
        names = [n for n in ("seed1", "seed2") if arm in TABLE_ARMS[n]]
        sj = {e["arm"]: e for e in T[names[0]]["arms"]}[arm]["smallest_detectable_effect"]
        ratios.append(sj["se_halves"] / M[arm]["sde_exact"]["se_halves"])
    if many:
        w("Does the pooled value of check 3 aim at the exact value of check 1? With 20,000 splits of my own (seed %d), pooled as in the table, `se_halves` comes out as: %s. So the exact value is where the table's way of pooling settles when the number of splits grows."
          % (MANY_SPLITS_SEED, "; ".join("%s %.6f against the exact %.6f (ratio %.4f)" % (a2, many[a2], M[a2]["sde_exact"]["se_halves"], many[a2] / M[a2]["sde_exact"]["se_halves"]) for a2 in ARMS)))
        w("")
    w("Reading: the table's value is within %.1f%% of the value that needs no random numbers in every arm (ratios %s). So the printed smallest detectable effects (%s points) do not hang on the seed of the splits. "
      "What the number means is unchanged by this check: sampling noise only, as the plan says. For scale, the half-width of the family-level t-interval is %s points."
      % (100 * max(abs(x - 1) for x in ratios), ", ".join("%.3f" % x for x in ratios),
         ", ".join("%s %.1f" % (arm, 100 * sde0[arm]["mde"]) for arm in ARMS),
         "; ".join("%s minus %s %.1f" % (k[1], k[0], 100 * (pair[k]["t"][1] - pair[k]["t"][0]) / 2) for k in PAIRS)))
    w("")
    w("### 8.3 Pass and reward, checked against the words of section 6 (not a table number)")
    w("")
    w("For every attempt I compared the saved `final_state` with the family's expected end state for the default rule set in `dev.jsonl` (roles, approval records, grant records, and the closed request with its resolution, person and role), "
      "and I computed the reward from the formula in section 6 (0 for a fail; 100 minus 5 per malformed message, at most 15, minus 2 per turn over the budget, at most 10) with my own count of malformed messages and turns and the budget from `dev.jsonl`.")
    w("")
    w("| Arm | Attempts | `passed` differs from my comparison (records as lists) | `passed` differs (records as sets, a doubled record ignored) | Attempts where the two readings differ | Attempts with a doubled approval or grant record (of them passed) | `reward` differs from the formula | Failed attempts with a reward other than 0 |")
    w("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        x = words[arm]
        w("| %s | %d | %d | %d | %d | %d (%d) | %d | %d |" % (arm, x["n"], x["pass_multiset_disagree"], x["pass_set_disagree"], x["set_vs_multiset_differ"],
                                                          x["doubled_record"], x["doubled_record_and_passed"], x["reward_disagree"], x["fail_with_reward"]))
    w("")
    w("This is a plain comparison of saved end states; it does not replay the tool calls (the audit's replay check does that). "
      "Whether a doubled record should fail an attempt is not settled by this data: the attempts with a doubled record fail under both readings (the column \"two readings differ\" is 0).")
    w("")

    # ---- mismatches ----
    w("## 9. Every mismatch")
    w("")
    any_mis = False
    if bad_checks:
        any_mis = True
        w("Input checks that failed:")
        w("")
        for c in bad_checks:
            w("- %s: %s: found %s, want %s" % c)
        w("")
    if exact_bad or extra_exact_bad:
        any_mis = True
        w("`.json` entries without a random part that differ from my value:")
        w("")
        for r in exact_bad + extra_exact_bad:
            w("- table %s, arm %s, `%s`: table %s, mine %s %s" % (r["table"], r["arm"], r["metric"], show_table(r["table_value"]), show_mine(r["mine"]), r["note"]))
        w("")
    if md_bad:
        any_mis = True
        w("`.md` numbers without a random part that are not my value rounded:")
        w("")
        for r in md_bad:
            w("- table %s, arm %s, %s: printed %s, mine %s" % (r["table"], r["arm"], r["metric"], r["printed"], r["mine"]))
        w("")
    if md_json_bad:
        any_mis = True
        w("`.md` numbers that are not the rounding of the `.json` value beside them:")
        w("")
        for r in md_json_bad:
            w("- table %s, arm %s, %s: printed %s, .json %s" % (r["table"], r["arm"], r["metric"], r["printed"], r["json_value"]))
        w("")
    if mc_bad:
        any_mis = True
        w("Interval ends outside the 99.9% Monte Carlo band (my line for \"outside the noise\"):")
        w("")
        for r in mc_bad:
            w("- table %s, %s, %s end: table %.4f, exact %.4f, 99.9%% band %.4f to %.4f, 200 seeds %.4f to %.4f (points)"
              % (r["table"], r["what"], r["end"], 100 * r["table_value"], 100 * r["exact"], 100 * r["band_wide"][0], 100 * r["band_wide"][1], 100 * r["seed_min"], 100 * r["seed_max"]))
        w("")
    if seed0_bad or md_seed0_bad:
        any_mis = True
        w("Numbers with a random part that my code does not give exactly with the table's seed 0:")
        w("")
        for r in seed0_bad:
            w("- table %s, %s, `%s`: table %s, mine with seed 0 %s" % (r["table"], r["arm"], r["metric"], show_table(r["table_value"]), show_mine(r["mine"])))
        for r in md_seed0_bad:
            w("- table %s, %s, %s (.md): printed %s, mine with seed 0 %s" % (r["table"], r["arm"], r["metric"], r["printed"], r["mine"]))
        w("")
    if md_unhandled:
        any_mis = True
        w("Parts of the `.md` files that the comparison does not cover:")
        w("")
        for name, what in md_unhandled:
            w("- table %s: %s" % (name, what))
        w("")
    if not any_mis:
        w("None.")
        w("")
        w("- No `.json` entry without a random part and no printed `.md` entry differs from my value beyond the tolerance (%d and %d compared)." % (n_json_all, n_md_exact))
        w("- Every printed `.md` number is the rounding of the `.json` value beside it (%d compared)." % len([r for r in md_rows if r["equal_json"] is not None]))
        w("- Every bootstrap end lies inside the Monte Carlo noise when I use my own seed, and is equal to mine when I use the table's seed 0 (%d `.json` numbers with a random part compared with seed 0, intervals and smallest detectable effect; %d printed `.md` numbers)."
          % (len(seed0_rows), len(md_seed0)))
        w("")
    w("Table entries that I did not make again, and why (settings and labels, not results):")
    w("")
    grouped = collections.OrderedDict()
    for name, path, val, why in not_remade:
        short = re.sub(r"^arms\[[^\]]+\]\.", "arms[each arm].", path)
        grouped.setdefault((short, show_table(val), why), set()).add(name)
    for (short, val, why), names in grouped.items():
        w("- `%s` = %s (%s): %s" % (short, val if val != "" else "an empty list", "both table files" if len(names) > 1 else "table " + sorted(names)[0], why))
    w("")

    # ---- what it proves ----
    w("## 10. What this does and does not prove")
    w("")
    if not any_mis:
        w("What it shows:")
        w("")
        w("- **The numbers in the two tables follow from the saved attempts.** A second program, written from the plan's words without opening the table code, gives the same value for every number without a random part, "
          "and the same value for every number with a random part once it uses the table's seed.")
        w("- So between `episodes.jsonl` and the tables there is no arithmetic slip, no wrong denominator, no swapped arm or column, no stale file and no rounding slip in the `.md` files, for every number in the two tables.")
        w("- The readings that the plan's words leave open are settled by the match: the malformed share is pooled over all messages (not a mean of per-family ratios); the t-interval uses the standard deviation with n - 1; "
          "the percentile ends use numpy's linear rule; the variance split uses sums of squares added over families; the smallest detectable effect pools the splits by a root mean square.")
        w("- The comparison can fail: with errors planted in copies of the tables, `selftest_compare.py` checks that each planted error is reported and nothing else (see `selftest_compare.txt`).")
    else:
        w("There are mismatches (section 9), so the tables are not confirmed as they stand.")
    w("")
    w("What it does not show:")
    w("")
    w("- **Nothing about the saved attempts themselves.** I took `passed`, `reward`, `turns`, `closed` and `ended_by` as they are saved. Whether the harness ran the model correctly, whether the verifier is right, and whether the adapters are the ones named, is outside this check. "
      "(Section 8.3 only shows that the saved `passed` and `reward` agree with the saved end states and the reward formula; it replays nothing.)")
    w("- **Both programs follow the same plan.** If the plan's definition of a metric is a poor choice, both programs share it. Agreement shows that the table code does what the plan says, not that the plan measures the right thing. "
      "One thing to keep in mind when reading the GRPO rows (the counts are measured here; the reading is my judgement): %s. "
      "So the interval of a GRPO arm mostly says how often a resample of 48 families draws that one family."
      % "; ".join("in `%s` %d of 48 families are below 32 of 32, and one of them, %s, holds %d of the %d failed attempts"
                  % (a2, sum(1 for f in fam_order if M[a2]["fam_pass"][f] < 32),
                     min(fam_order, key=lambda f: M[a2]["fam_pass"][f]),
                     32 - min(M[a2]["fam_pass"][f] for f in fam_order),
                     M[a2]["n_episodes"] - M[a2]["n_passed"]) for a2 in ("grpo_seed1", "grpo_seed2")))
    w("- **Where the data cannot tell two definitions apart, agreement proves little**: malformed_only (no attempt qualifies under any reading), harness errors (none exist, so \"left out of every metric\" was never exercised), "
      "scenarios left out of Pass^k (none), `Episodes` (all records or valid records: the same number here), and plain against family-weighted shares (equal family sizes).")
    w("- **The intervals are only about the 48 dev families as a sample of families.** They do not cover the choice of training seed (two seeds describe two checkpoints), the run seed of the rollouts (one run seed, 0), or the fact that dev was watched during the work. The test set was not run and is not part of this.")
    w("- **The way to draw with seed 0 was found by trial**, not derived from the plan (the plan names the seed, not the generator). The trial is small (seven ways for the bootstrap, seven ways times three poolings for the splits) and a full hit means all 18 ends, or all 12 split numbers (`se_halves`, lowest and highest split, 4 arms), equal to 1e-12, so in my judgement a chance hit is not a worry; but it means that \"exactly equal\" for these numbers was reached with knowledge of the table's numbers.")
    w("")

    w("## 11. How to run this again")
    w("")
    w("From the repository folder, one after the other (each is a single light process; no model, no torch):")
    w("")
    w("- `.venv/bin/python results/main/audit/recount/rederive.py` writes `report.md` and `my_numbers.json` into this folder (or into `--out DIR`) (about 20 seconds; about 125 MB of memory at the peak when I measured it).")
    w("- `.venv/bin/python results/main/audit/recount/seed0_trials.py > results/main/audit/recount/seed0_trials.txt` (about 1 minute).")
    w("- `.venv/bin/python results/main/audit/recount/selftest_compare.py > results/main/audit/recount/selftest_compare.txt` (about 3 minutes; it works on copies in a new temporary folder and removes them at the end).")
    w("")
    w("All seeds are fixed in the script (bootstrap %d; noise study %d to %d; splits %d; sign-flip Monte Carlo check %d; 20,000 splits %d; the table's seed 0), so a second run gives the same report."
      % (MY_SEED, NOISE_SEEDS[0], NOISE_SEEDS[-1], SPLIT_SEED, PERM_SEED, MANY_SPLITS_SEED))
    w("")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(W) + "\n")

    # ---------------- console summary ----------------
    result = {
        "bad_checks": bad_checks,
        "json_mismatch": [(r["table"], r["arm"], r["metric"]) for r in exact_bad + extra_exact_bad],
        "md_mismatch": [(r["table"], r["arm"], r["metric"]) for r in md_bad],
        "md_vs_json_mismatch": [(r["table"], r["arm"], r["metric"]) for r in md_json_bad],
        "outside_noise": [(r["table"], r["what"], r["end"]) for r in mc_bad],
        "outside_95_band": [(r["table"], r["what"], r["end"]) for r in mc_out95],
        "outside_200_seeds": [(r["table"], r["what"], r["end"]) for r in mc_out_seeds],
        "seed0_mismatch": [(r["table"], r["arm"], r["metric"]) for r in seed0_bad],
        "md_seed0_mismatch": [(r["table"], r["arm"], r["metric"]) for r in md_seed0_bad],
        "n_json": n_json_all, "n_md": n_md_exact, "n_seed0": len(seed0_rows), "n_md_seed0": len(md_seed0),
        "n_interval_rows": len(mc_rows), "n_not_remade": len(not_remade), "md_unhandled": md_unhandled,
        "max_gap_exact": max_gap_exact, "max_gap_seed0": max_gap_seed0,
    }
    if quiet:
        return result
    print("families %d; records per arm %s" % (F, {a: M[a]["n_records"] for a in ARMS}))
    print("input checks failed: %d" % len(bad_checks))
    for arm in ARMS:
        m = M[arm]
        print("%-11s pass %d/%d = %.10f  interval: my seed %.4f..%.4f  seed 0 %.4f..%.4f  exact %.4f..%.4f"
              % (arm, m["n_passed"], m["n_episodes"], float(m["pass_rate"]), boot[arm][0], boot[arm][1],
                 boot0[arm][0], boot0[arm][1], exact[arm]["lo"], exact[arm]["hi"]))
    for key in PAIRS:
        p = pair[key]
        print("%-11s minus %-5s diff %+.10f  boot: my seed %+.4f..%+.4f  seed 0 %+.4f..%+.4f  exact %+.4f..%+.4f  t %+.6f..%+.6f"
              % (key[1], key[0], float(p["diff"]), p["boot"][0], p["boot"][1], boot0[key][0], boot0[key][1],
                 exact[key]["lo"], exact[key]["hi"], p["t"][0], p["t"][1]))
    print("t quantile 0.975, df 47: %.12f (beta) %.12f (simpson)" % (tq_beta, tq_simpson))
    print(".json numbers without a random part: %d compared, mismatches %d, largest absolute difference %.2e"
          % (n_json_all, len(exact_bad) + len(extra_exact_bad), max_gap_exact))
    print(".md printed numbers without a random part: %d compared, mismatches %d" % (n_md_exact, len(md_bad)))
    print(".md printed numbers against their .json value: %d compared, mismatches %d"
          % (len([r for r in md_rows if r["equal_json"] is not None]), len(md_json_bad)))
    print("interval ends (rows in both files): %d; outside the 95%% band %d; outside the 99.9%% band %d; outside the range of 200 seeds %d"
          % (len(mc_rows), len(mc_out95), len(mc_bad), len(mc_out_seeds)))
    print("numbers with a random part, my code with seed 0: .json %d compared, mismatches %d, largest absolute difference %.2e; .md %d compared, mismatches %d"
          % (len(seed0_rows), len(seed0_bad), max_gap_seed0, len(md_seed0), len(md_seed0_bad)))
    print("table entries not made again (settings and labels): %d; parts of the .md not covered: %d" % (len(not_remade), len(md_unhandled)))
    worst = sorted(numeric_exact, key=num_gap, reverse=True)[:3]
    print("the three largest gaps among the numbers without a random part: %s"
          % "; ".join("%s %s %s %.2e" % (r["table"], r["arm"], r["metric"], num_gap(r)) for r in worst))
    for key2 in ("json_mismatch", "md_mismatch", "md_vs_json_mismatch", "outside_noise", "seed0_mismatch", "md_seed0_mismatch"):
        for item in result[key2]:
            print("  %s: %s" % (key2.upper(), " | ".join(str(x) for x in item)))
    shown = os.path.relpath(out_dir)
    print("wrote report.md and my_numbers.json in", out_dir if shown.startswith("..") else shown)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A second calculation of the two dev result tables.")
    parser.add_argument("--out", help="the folder for report.md and my_numbers.json (default: this script's folder)")
    # added on 2026-11-19: --tables and the exit code; the printed lines of a clean run are unchanged
    parser.add_argument("--tables", help="a folder with copies of the two table files (same names), to compare against them instead")
    args = parser.parse_args()
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    tables_arg = {k: os.path.join(args.tables, os.path.basename(v)) for k, v in TABLES.items()} if args.tables else None
    res = main(tables=tables_arg, out_dir=args.out)
    fail_keys = ("bad_checks", "json_mismatch", "md_mismatch", "md_vs_json_mismatch", "outside_noise",
                 "seed0_mismatch", "md_seed0_mismatch", "md_unhandled")
    if any(res[k] for k in fail_keys):
        sys.stderr.write("exit code 1: %s\n" % "; ".join("%s %d" % (k, len(res[k])) for k in fail_keys if res[k]))
        sys.exit(1)
    sys.exit(0)
