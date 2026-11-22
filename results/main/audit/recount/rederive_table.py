#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rederive_table.py - the recount of rederive.py, for one more result table.

  --table test          (the default) results/main/tables/test_base_sft_grpo_seed1_seed2.json and .md,
                        from results/main/<arm>_test/episodes.jsonl (base, sft, grpo_seed1, grpo_seed2)
                        and data/scenarios/test.jsonl.
                        Writes report_test.md, my_numbers_test.json.
  --table base_vs_sft   results/main/tables/base_vs_sft.json and .md (a dev table),
                        from results/main/<arm>_dev/episodes.jsonl (base, sft) and data/scenarios/dev.jsonl.
                        Writes report_base_vs_sft.md, my_numbers_base_vs_sft.json.

It uses the functions of rederive.py in this folder (reading, the metrics, the intervals, the tests,
the .md reader) and repeats its comparison for one table. It imports nothing from the package
accessdesk; standard library and numpy only.

Written on 2026-11-19, after the test results were opened (commit 7fad426). It changes no number of any
table: it only counts again from the saved attempts and compares.

Exit code: 0 when every compared number agrees; 1 when any compared number differs (or an input check
fails, or a part of the .md is not covered). The printed lines of a clean run do not change; on a
failure one more line goes to standard error.

Run from the top folder of the repository:
      .venv/bin/python results/main/audit/recount/rederive_table.py [--table test|base_vs_sft] [--out DIR] [--tables DIR]
  --out DIR     write the report and the numbers into DIR (default: this script's folder)
  --tables DIR  compare against copies of the table files in DIR (same file names); used to show the exit code
"""

import collections
import json
import math
import os
import re
import sys
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rederive as rd  # noqa: E402  (my own script in this folder)

REPO = rd.REPO
TABLES_DIR = os.path.join(REPO, "results", "main", "tables")

CHOICES = {
    "test": {
        "stem": os.path.join(TABLES_DIR, "test_base_sft_grpo_seed1_seed2"),
        "split": "test",
        "scenarios": os.path.join(REPO, "data", "scenarios", "test.jsonl"),
        "arms": ["base", "sft", "grpo_seed1", "grpo_seed2"],
        # (a, b) means "b minus a"; the test comparisons of ANALYSIS_PLAN.md section 4
        "pairs": [("base", "sft"), ("sft", "grpo_seed1"), ("sft", "grpo_seed2")],
        "suffix": "test",
    },
    "base_vs_sft": {
        "stem": os.path.join(TABLES_DIR, "base_vs_sft"),
        "split": "dev",
        "scenarios": os.path.join(REPO, "data", "scenarios", "dev.jsonl"),
        "arms": ["base", "sft"],
        "pairs": [("base", "sft")],
        "suffix": "base_vs_sft",
    },
}

FAIL_KEYS = ("bad_checks", "structure", "json_mismatch", "md_mismatch", "md_vs_json_mismatch", "outside_noise",
             "seed0_mismatch", "md_seed0_mismatch", "md_unhandled")


def episodes_path(arm, split):
    return os.path.join(REPO, "results", "main", "%s_%s" % (arm, split), "episodes.jsonl")


def main(which="test", stem=None, out_dir=None, quiet=False):
    """stem and out_dir can be replaced (selftest_table.py does that to plant errors on purpose)."""
    cfg = CHOICES[which]
    stem = stem or cfg["stem"]
    out_dir = out_dir or HERE
    split, ARMS, PAIRS = cfg["split"], cfg["arms"], cfg["pairs"]
    NAME = which
    EPI = {a: episodes_path(a, split) for a in ARMS}
    fams, fam_order = rd.read_families(cfg["scenarios"])
    F = len(fam_order)

    # ---------------- read and check ----------------
    hashes = collections.OrderedDict()
    hashes[os.path.relpath(cfg["scenarios"], REPO)] = rd.sha256_file(cfg["scenarios"])
    for arm in ARMS:
        hashes[os.path.relpath(EPI[arm], REPO)] = rd.sha256_file(EPI[arm])
    for ext in (".json", ".md"):
        hashes[os.path.relpath(stem + ext, REPO)] = rd.sha256_file(stem + ext)
    data_sha = hashes[os.path.relpath(cfg["scenarios"], REPO)]
    data, checks = {}, []
    want = set((f, s, i) for f in fam_order for s in (1, 2, 3, 4) for i in range(8))
    for arm in ARMS:
        recs = rd.read_attempts(EPI[arm], fams)
        data[arm] = recs
        keys = [(r["family"], r["style"], r["sample"]) for r in recs]
        checks.append((arm, "records in the file", len(recs), len(want)))
        checks.append((arm, "distinct (family, style, attempt number)", len(set(keys)), len(want)))
        checks.append((arm, "missing or extra against %d families x 4 styles x 8 attempts" % F,
                       len(want.symmetric_difference(set(keys))), 0))
        checks.append((arm, "records whose family_id is not in %s" % os.path.basename(cfg["scenarios"]),
                       sum(1 for r in recs if r["family"] not in fams), 0))
        checks.append((arm, "records whose arm field is not the arm", sum(1 for r in recs if r["arm"] != arm), 0))
        checks.append((arm, "records whose split is not %s" % split, sum(1 for r in recs if r["split"] != split), 0))
        checks.append((arm, "records whose outcome differs from %s" % os.path.basename(cfg["scenarios"]),
                       sum(1 for r in recs if r["family"] in fams and fams[r["family"]]["outcome"] != r["outcome"]), 0))
        checks.append((arm, "records whose scenario_id is not family/s<style>",
                       sum(1 for r in recs if r["scenario"] != "%s/s%d" % (r["family"], r["style"])), 0))
        checks.append((arm, "records whose n_turns differs from the number of saved turns",
                       sum(1 for r in recs if r["n_turns_field"] != r["n_msgs"]), 0))
        checks.append((arm, "records whose n_malformed differs from the turns with parse_ok false",
                       sum(1 for r in recs if r["n_malformed_field"] != r["n_bad"]), 0))
        checks.append((arm, "records whose turn_budget differs from the scenario file (default rule set)",
                       sum(1 for r in recs if r["family"] in fams and fams[r["family"]]["turn_budget_default"] != r["turn_budget"]), 0))
        checks.append((arm, "records where closed and ended_by disagree",
                       sum(1 for r in recs if r["closed"] != (r["ended_by"] == "close")), 0))
        checks.append((arm, "records whose data_sha256 is not the SHA-256 of the scenario file",
                       sum(1 for r in recs if r["data_sha256"] != data_sha), 0))
    settings_all = sorted(set(r["settings"] for arm in ARMS for r in data[arm]))
    checks.append(("all arms", "distinct settings blocks over all records", len(settings_all), 1))
    bad_checks = [c for c in checks if c[2] != c[3]]
    if any(c[1].startswith("records whose family_id") for c in bad_checks):
        raise SystemExit("a record names a family that is not in the scenario file")

    M = {arm: rd.arm_metrics(data[arm], fams, fam_order) for arm in ARMS}
    for arm in ARMS:
        if M[arm]["families"] != fam_order:
            raise SystemExit("arm %s does not cover the %d families" % (arm, F))
    if not all(set(M[arm]["fam_n"].values()) == {32} for arm in ARMS):
        raise SystemExit("a family does not have 32 valid attempts; the exact bootstrap below assumes it")
    DEN = 32
    C = {arm: np.array([M[arm]["fam_pass"][f] for f in fam_order], dtype=np.int64) for arm in ARMS}
    R = {arm: C[arm] / float(DEN) for arm in ARMS}

    # ---------------- bootstrap with my own seed ----------------
    rng = np.random.default_rng(rd.MY_SEED)
    boot = {}
    for arm in ARMS:
        idx = rng.integers(0, F, size=(rd.N_DRAWS, F))
        boot[arm] = rd.percentile_interval(R[arm][idx].mean(axis=1))
    pair = {}
    for a, b in PAIRS:
        idx = rng.integers(0, F, size=(rd.N_DRAWS, F))      # the same resampled families for both arms
        d = R[b] - R[a]
        t_lo, t_hi, sd, tq = rd.t_interval(d)
        pair[(a, b)] = {"diff": Fr(int(C[b].sum() - C[a].sum()), F * DEN),
                        "boot": rd.percentile_interval(R[b][idx].mean(axis=1) - R[a][idx].mean(axis=1)),
                        "t": (t_lo, t_hi), "sd_of_family_differences": sd, "t_quantile": tq,
                        "d_int": (C[b] - C[a]).tolist()}
        assert abs(float(pair[(a, b)]["diff"]) - d.mean()) < 1e-12

    # ---------------- the same with the table's seed (0), drawn as in rederive.py ----------------
    boot0 = {}
    for arm in ARMS:
        idx = np.random.default_rng(rd.TABLE_SEED).integers(0, F, size=(rd.N_DRAWS, F))
        boot0[arm] = rd.percentile_interval(R[arm][idx].mean(axis=1))
    for a, b in PAIRS:
        idx = np.random.default_rng(rd.TABLE_SEED).integers(0, F, size=(rd.N_DRAWS, F))
        boot0[(a, b)] = rd.percentile_interval(R[b][idx].mean(axis=1) - R[a][idx].mean(axis=1))

    # ---------------- exact bootstrap and the Monte Carlo band; 200 further seeds ----------------
    exact = {arm: rd.exact_interval_and_band(C[arm].tolist(), DEN) for arm in ARMS}
    for key in PAIRS:
        exact[key] = rd.exact_interval_and_band(pair[key]["d_int"], DEN)
    quantities = list(ARMS) + list(PAIRS)
    V = np.stack([R[qn] if isinstance(qn, str) else (R[qn[1]] - R[qn[0]]) for qn in quantities], axis=1)
    ends = np.zeros((len(rd.NOISE_SEEDS), 2, len(quantities)))
    for si, seed in enumerate(rd.NOISE_SEEDS):
        idx = np.random.default_rng(seed).integers(0, F, size=(rd.N_DRAWS, F))
        ends[si] = np.percentile(V[idx].mean(axis=1), [2.5, 97.5], axis=0)
    noise = {}
    for qi, qn in enumerate(quantities):
        noise[qn] = {"lo_min": float(ends[:, 0, qi].min()), "lo_max": float(ends[:, 0, qi].max()),
                     "lo_sd": float(ends[:, 0, qi].std(ddof=1)), "hi_min": float(ends[:, 1, qi].min()),
                     "hi_max": float(ends[:, 1, qi].max()), "hi_sd": float(ends[:, 1, qi].std(ddof=1)),
                     "lo_all": ends[:, 0, qi].copy(), "hi_all": ends[:, 1, qi].copy()}
    del ends
    tq_beta = rd.t_quantile(0.975, F - 1, rd.t_cdf_beta)
    tq_simpson = rd.t_quantile(0.975, F - 1, rd.t_cdf_simpson)

    # ---------------- smallest detectable effect with the table's seed (0) ----------------
    sde0 = {arm: rd.two_halves(rd.passes_array(data[arm], fam_order), rd.TABLE_SEED, rd.N_SPLITS) for arm in ARMS}

    # ---------------- sign test and exact sign-flip test (a second view, not a table number) ----------------
    tests = {key: {"sign": rd.sign_test(pair[key]["d_int"]), "perm": rd.sign_flip_test_exact(pair[key]["d_int"])} for key in PAIRS}

    # ---------------- pass and reward by the words of section 6 ----------------
    words = {}
    for arm in ARMS:
        recs = data[arm]
        w_ = {"n": len(recs)}
        w_["pass_multiset_disagree"] = sum(1 for r in recs if r["pass_words_multiset"] != r["passed"])
        w_["pass_set_disagree"] = sum(1 for r in recs if r["pass_words_set"] != r["passed"])
        w_["doubled_record"] = sum(1 for r in recs if r["doubled_record"])

        def reward_words(r):
            if not r["passed"]:
                return 0
            return 100 - min(15, 5 * r["n_bad"]) - min(10, 2 * max(0, r["n_msgs"] - fams[r["family"]]["turn_budget_default"]))
        w_["reward_disagree"] = sum(1 for r in recs if reward_words(r) != r["reward"])
        words[arm] = w_

    # ==============================================================================================
    # compare with the table
    # ==============================================================================================
    with open(stem + ".json", "r", encoding="utf-8") as fh:
        tab = json.load(fh)
    md_lines, md_tables, md_rewards = rd.parse_md(stem + ".md")

    rows, not_remade, structure = [], [], []

    def add(kind, metric, arm, tval, mine, equal, note=""):
        rows.append({"table": NAME, "kind": kind, "metric": metric, "arm": arm, "table_value": tval,
                     "mine": mine, "equal": equal, "note": note})

    def my_arm_leaves(arm):
        m = M[arm]

        def one(key):
            v = m["values_" + key]
            return v[0] if len(v) == 1 else v
        L = {"arm": ("identity", arm), "splits": ("identity", m["values_split"]), "ruleset": ("identity", one("ruleset")),
             "toolset": ("identity", one("toolset")), "prompt": ("identity", one("prompt")),
             "text_is_placeholder": ("identity", one("text_is_placeholder")),
             "config_sha256": ("identity", m["values_config_sha256"]),
             "n_families": ("exact", m["n_families"]), "n_episodes": ("exact", m["n_episodes"]),
             "n_harness_errors": ("exact", m["n_harness_errors"]), "pass_rate": ("exact", m["pass_rate"]),
             "pass_rate_lo": ("mc", boot[arm][0]), "pass_rate_hi": ("mc", boot[arm][1])}
        for o in rd.OUTCOMES:
            L["pass_rate_by_outcome." + o] = ("exact", m["pass_rate_by_outcome"][o])
        for k in ("pass_rate_seen", "pass_rate_held_out", "zero_variance_equal_reward", "zero_variance_equal_passfail",
                  "invalid_call_rate", "pass_at_8", "all_wordings_pass"):
            L[k] = ("exact", m[k])
        for k in rd.KS:
            L["pass_hat_k.%d" % k] = ("exact", m["pass_hat_k"][k])
            L["n_scenarios_left_out.%d" % k] = ("exact", m["n_scenarios_left_out"][k])
        for k in ("wording", "sampling", "wording_if_sampling_only"):
            L["variance_shares." + k] = ("extra_exact", m["variance_shares"][k])
        L["turn_stats.n_episodes"] = ("exact", m["n_episodes"])
        for k in ("mean_turns", "share_at_turn_limit", "share_closed"):
            L["turn_stats." + k] = ("exact", m[k])
        for k in ("closed_correctly", "closed_wrongly", "never_closed", "malformed_only"):
            L["outcome_labels." + k] = ("exact", m["outcome_labels"][k])
        for rv, cnt in m["reward_distribution"].items():
            L["reward_distribution.%d" % rv] = ("exact", cnt)
        L["smallest_detectable_effect.n_families"] = ("exact", m["n_families"])
        L["smallest_detectable_effect.n_splits"] = ("setting", rd.N_SPLITS)
        for k in ("se_halves", "se", "mde", "mde_lowest_split", "mde_highest_split"):
            L["smallest_detectable_effect." + k] = ("seed0", sde0[arm][k])
        return L

    mc_rows = []

    def mc_row(key, label, end, tval, mval):
        ex, nz = exact[key], noise[key]
        band, wide = ex[end + "_band"], ex[end + "_band_wide"]
        mine0 = boot0[key][0 if end == "lo" else 1]
        return {"what": label, "end": end, "table_value": tval, "mine": mval, "mine0": mine0,
                "equal0": abs(tval - mine0) <= rd.TOL_JSON, "exact": ex[end], "band": band, "band_wide": wide,
                "seed_min": nz[end + "_min"], "seed_max": nz[end + "_max"], "seed_sd": nz[end + "_sd"],
                "inside_band": band[0] - 1e-12 <= tval <= band[1] + 1e-12,
                "inside_wide": wide[0] - 1e-12 <= tval <= wide[1] + 1e-12,
                "inside_seeds": nz[end + "_min"] - 1e-12 <= tval <= nz[end + "_max"] + 1e-12,
                "step": ex["step"]}

    # ---- top level ----
    top_mine = {"n_families": F, "n_scenarios": M[ARMS[0]]["n_scenarios"], "n_episodes_per_arm": M[ARMS[0]]["n_episodes"],
                "splits": sorted(set(s for a in ARMS for s in M[a]["values_split"])),
                "text_sha256": M[ARMS[0]]["values_text_sha256"][0],
                "text_is_placeholder": M[ARMS[0]]["values_text_is_placeholder"][0]}
    for key, val in tab.items():
        if key in ("arms", "pairs"):
            continue
        if key in top_mine:
            add("identity" if not isinstance(val, (int, float)) or isinstance(val, bool) else "exact",
                "top." + key, "-", val, top_mine[key], val == top_mine[key])
        elif key == "notes":
            not_remade.append(("top." + key, val, "a list of notes of the table code; it is empty" if val == [] else "a list of notes of the table code"))
        else:
            not_remade.append(("top." + key, val, "a setting or label of the table code, not a result counted from attempts"))
    for key in top_mine:
        if key not in tab:
            add("exact", "top." + key, "-", None, top_mine[key], False, "in my numbers, not in the table")

    # ---- arms ----
    t_arms = {e["arm"]: e for e in tab["arms"]}
    if sorted(t_arms) != sorted(ARMS) or len(tab["arms"]) != len(ARMS):
        structure.append(("arms", "the table holds the arms %s, I expect %s" % ([e["arm"] for e in tab["arms"]], ARMS)))
    for arm in ARMS:
        if arm not in t_arms:
            continue
        leaves = rd.flatten(t_arms[arm])
        mine = my_arm_leaves(arm)
        for path, tval in leaves.items():
            if path not in mine:
                not_remade.append(("arms[%s].%s" % (arm, path), tval, "no value of mine for this entry"))
                continue
            kind, mval = mine[path]
            if kind == "setting":
                not_remade.append(("arms[%s].%s" % (arm, path), tval, "a setting of the table code, not a result"))
            elif kind == "identity":
                add(kind, path, arm, tval, mval, tval == mval)
            elif kind in ("exact", "extra_exact"):
                ok = mval is not None and tval is not None and abs(float(tval) - float(mval)) <= rd.TOL_JSON
                add(kind, path, arm, tval, mval, ok)
            elif kind == "mc":
                end = "lo" if path.endswith("_lo") else "hi"
                mc_rows.append(mc_row(arm, arm, end, tval, mval))
                add(kind, path, arm, tval, mval, mc_rows[-1]["inside_wide"],
                    "Monte Carlo: not expected to be equal; 'yes' means inside the 99.9% noise band")
                add("seed0", path + " (my code, seed 0)", arm, tval, mc_rows[-1]["mine0"], mc_rows[-1]["equal0"],
                    "10,000 draws; mine drawn with the table's seed 0")
            elif kind == "seed0":
                add(kind, path, arm, tval, mval, abs(float(tval) - float(mval)) <= rd.TOL_JSON,
                    "200 random splits; mine drawn with the table's seed 0")
        for path in mine:
            if path not in leaves:
                add(mine[path][0], path, arm, None, mine[path][1], False, "in my numbers, not in the table")

    # ---- pairs ----
    t_pairs = {}
    for e in tab["pairs"]:
        key = (e["a"], e["b"])
        label = "%s minus %s" % (e["b"], e["a"])
        if key not in pair:
            structure.append(("pairs", "the table holds the comparison %s, which I do not expect" % label))
            continue
        t_pairs[key] = e
        p = pair[key]
        for k2, tval in e.items():
            if k2 in ("a", "b"):
                continue
            if k2 == "n_families":
                add("exact", "pair.n_families", label, tval, F, tval == F)
            elif k2 == "diff":
                add("exact", "pair.diff", label, tval, p["diff"], abs(tval - float(p["diff"])) <= rd.TOL_JSON)
            elif k2 in ("t_lo", "t_hi"):
                mval = p["t"][0 if k2 == "t_lo" else 1]
                add("exact", "pair." + k2, label, tval, mval, abs(tval - mval) <= 1e-9, "tolerance 1e-9 (my own t quantile)")
            elif k2 in ("lo", "hi"):
                mval = p["boot"][0 if k2 == "lo" else 1]
                mc_rows.append(mc_row(key, label, k2, tval, mval))
                add("mc", "pair." + k2, label, tval, mval, mc_rows[-1]["inside_wide"],
                    "Monte Carlo: not expected to be equal; 'yes' means inside the 99.9% noise band")
                add("seed0", "pair." + k2 + " (my code, seed 0)", label, tval, mc_rows[-1]["mine0"], mc_rows[-1]["equal0"],
                    "10,000 draws; mine drawn with the table's seed 0")
            else:
                not_remade.append(("pairs[%s].%s" % (label, k2), tval, "no value of mine for this entry"))
    for key in PAIRS:
        if key not in t_pairs:
            structure.append(("pairs", "the comparison %s minus %s is missing in the table" % (key[1], key[0])))

    # ---- the .md file: every printed number against my value, and against the .json beside it ----
    md_rows, md_unhandled = [], []

    def md_add(metric, arm, text, my_value, kind="exact", json_value=None):
        md_rows.append({"table": NAME, "metric": metric, "arm": arm, "printed": text, "mine": my_value, "kind": kind,
                        "equal_mine": rd.md_equal(text, my_value) if my_value is not None else None,
                        "json_value": json_value,
                        "equal_json": rd.md_equal(text, json_value) if json_value is not None else None})

    MD_COLUMNS = [
        ["Arm", "Episodes", "Harness errors", "Pass rate", "Interval", "grant", "approval", "already", "inactive", "Seen styles",
         "Held-out style", "Groups with equal rewards", "Groups with equal pass or fail", "Malformed messages*"],
        ["Comparison", "Families", "Difference in points", "Paired bootstrap by family", "t-interval"],
        ["Arm", "Split", "Rule set", "Tools", "Prompt", "Text", "Config*"],
        ["Arm", "Pass@8", "Pass^1", "Pass^2", "Pass^4", "Pass^8", "All wordings pass", "Variance inside families*", "Mean turns",
         "Ended at the turn limit", "closed_correctly", "closed_wrongly", "never_closed", "malformed_only", "Smallest detectable effect*"],
    ]
    if sorted(md_rewards) != sorted(ARMS):
        md_unhandled.append("reward lines found for the arms %s" % sorted(md_rewards))
    head = re.match(r"^Families: (\d+)\. Scenarios: (\d+)\. Episodes per arm: (\d+)\.", md_lines[0]) if md_lines else None
    if head:
        md_add("Families (first line)", "-", head.group(1), F, json_value=tab.get("n_families"))
        md_add("Scenarios (first line)", "-", head.group(2), M[ARMS[0]]["n_scenarios"], json_value=tab.get("n_scenarios"))
        md_add("Episodes per arm (first line)", "-", head.group(3), M[ARMS[0]]["n_episodes"], json_value=tab.get("n_episodes_per_arm"))
    line2 = re.match(r"^Intervals: (\d+)%, by resampling families, (\d+) draws, seed (\d+)\.", md_lines[1]) if len(md_lines) > 1 else None
    if line2:
        md_add("Level in percent (second line)", "-", line2.group(1), None, "setting", 100 * tab["level"])
        md_add("Draws (second line)", "-", line2.group(2), None, "setting", tab["n_draws"])
        md_add("Seed (second line)", "-", line2.group(3), None, "setting", tab["seed"])
    if not head or not line2:
        md_unhandled.append("the first two lines do not have the form that I expect")
    if len(md_tables) != 4:
        md_unhandled.append("%d tables found in the .md, 4 expected" % len(md_tables))
    seen_rows = collections.Counter()
    for tb in md_tables:
        h = tb["header"]
        known = None
        for kn in MD_COLUMNS:
            if h[:2] == kn[:2]:
                known = kn
        if known is None:
            md_unhandled.append("a table that I do not know, with the columns %s" % h)
            continue
        for c in h:
            if not any(c == k or (k.endswith("*") and c.startswith(k[:-1])) for k in known):
                md_unhandled.append("column %r is not compared" % c)
        for k in known:
            if not any(c == k or (k.endswith("*") and c.startswith(k[:-1])) for c in h):
                md_unhandled.append("column %r is missing in the .md" % k)
                known = None
        if known is None:
            continue
        for row in tb["rows"]:
            if h[0] == "Arm" and row["Arm"] not in ARMS:
                md_unhandled.append("a row for the arm %r, which I do not expect" % row["Arm"])
                continue
            if h[0] == "Arm" and "Episodes" in h:
                arm = row["Arm"]
                seen_rows[("main", arm)] += 1
                m, tj = M[arm], t_arms.get(arm, {})
                md_add("Episodes", arm, row["Episodes"], m["n_episodes"], json_value=tj.get("n_episodes"))
                md_add("Harness errors", arm, row["Harness errors"], m["n_harness_errors"], json_value=tj.get("n_harness_errors"))
                md_add("Pass rate", arm, row["Pass rate"], 100 * float(m["pass_rate"]), json_value=100 * tj["pass_rate"] if tj else None)
                lo, hi = rd.md_numbers(row["Interval"])
                md_add("Interval, lower end", arm, lo, 100 * boot[arm][0], "mc", 100 * tj["pass_rate_lo"] if tj else None)
                md_add("Interval, upper end", arm, hi, 100 * boot[arm][1], "mc", 100 * tj["pass_rate_hi"] if tj else None)
                md_add("Interval, lower end (my code, seed 0)", arm, lo, 100 * boot0[arm][0], "seed0")
                md_add("Interval, upper end (my code, seed 0)", arm, hi, 100 * boot0[arm][1], "seed0")
                for o in rd.OUTCOMES:
                    md_add(o, arm, row[o], 100 * float(m["pass_rate_by_outcome"][o]), json_value=100 * tj["pass_rate_by_outcome"][o] if tj else None)
                md_add("Seen styles", arm, row["Seen styles"], 100 * float(m["pass_rate_seen"]), json_value=100 * tj["pass_rate_seen"] if tj else None)
                md_add("Held-out style", arm, row["Held-out style"], 100 * float(m["pass_rate_held_out"]), json_value=100 * tj["pass_rate_held_out"] if tj else None)
                md_add("Groups with equal rewards", arm, row["Groups with equal rewards"], 100 * float(m["zero_variance_equal_reward"]),
                       json_value=100 * tj["zero_variance_equal_reward"] if tj else None)
                md_add("Groups with equal pass or fail", arm, row["Groups with equal pass or fail"], 100 * float(m["zero_variance_equal_passfail"]),
                       json_value=100 * tj["zero_variance_equal_passfail"] if tj else None)
                col = [c for c in h if c.startswith("Malformed messages")][0]
                md_add("Malformed messages", arm, row[col], 100 * float(m["invalid_call_rate"]), json_value=100 * tj["invalid_call_rate"] if tj else None)
            elif h[0] == "Comparison":
                mm = re.match(r"^(\w+) minus (\w+)$", row["Comparison"])
                key = (mm.group(2), mm.group(1)) if mm else None
                if key not in pair:
                    md_unhandled.append("a comparison row %r that I do not expect" % row["Comparison"])
                    continue
                seen_rows[("pair", key)] += 1
                p, tj = pair[key], t_pairs.get(key)
                md_add("Families", row["Comparison"], row["Families"], F, json_value=tj["n_families"] if tj else None)
                md_add("Difference in points", row["Comparison"], row["Difference in points"], 100 * float(p["diff"]), json_value=100 * tj["diff"] if tj else None)
                lo, hi = rd.md_numbers(row["Paired bootstrap by family"])
                md_add("Paired bootstrap, lower end", row["Comparison"], lo, 100 * p["boot"][0], "mc", 100 * tj["lo"] if tj else None)
                md_add("Paired bootstrap, upper end", row["Comparison"], hi, 100 * p["boot"][1], "mc", 100 * tj["hi"] if tj else None)
                md_add("Paired bootstrap, lower end (my code, seed 0)", row["Comparison"], lo, 100 * boot0[key][0], "seed0")
                md_add("Paired bootstrap, upper end (my code, seed 0)", row["Comparison"], hi, 100 * boot0[key][1], "seed0")
                lo, hi = rd.md_numbers(row["t-interval"])
                md_add("t-interval, lower end", row["Comparison"], lo, 100 * p["t"][0], json_value=100 * tj["t_lo"] if tj else None)
                md_add("t-interval, upper end", row["Comparison"], hi, 100 * p["t"][1], json_value=100 * tj["t_hi"] if tj else None)
            elif h[0] == "Arm" and "Split" in h:
                arm = row["Arm"]
                seen_rows[("ident", arm)] += 1
                m = M[arm]
                col = [c for c in h if c.startswith("Config")][0]
                ident = [("Split", ",".join(m["values_split"])), ("Rule set", ",".join(m["values_ruleset"])),
                         ("Tools", ",".join(m["values_toolset"])), ("Prompt", ",".join(m["values_prompt"])),
                         ("Text", ",".join(v[:12] for v in m["values_text_sha256"])),
                         (col, ",".join(v[:12] for v in m["values_config_sha256"]))]
                for cname, mine_text in ident:
                    md_rows.append({"table": NAME, "metric": cname + " (text)", "arm": arm, "printed": row[cname], "mine": mine_text,
                                    "kind": "identity", "equal_mine": row[cname] == mine_text, "json_value": None, "equal_json": None})
            elif h[0] == "Arm" and "Pass@8" in h:
                arm = row["Arm"]
                seen_rows[("more", arm)] += 1
                m, tj = M[arm], t_arms.get(arm)
                md_add("Pass@8", arm, row["Pass@8"], 100 * float(m["pass_at_8"]), json_value=100 * tj["pass_at_8"] if tj else None)
                for k in rd.KS:
                    md_add("Pass^%d" % k, arm, row["Pass^%d" % k], 100 * float(m["pass_hat_k"][k]), json_value=100 * tj["pass_hat_k"][str(k)] if tj else None)
                md_add("All wordings pass", arm, row["All wordings pass"], 100 * float(m["all_wordings_pass"]), json_value=100 * tj["all_wordings_pass"] if tj else None)
                col = [c for c in h if c.startswith("Variance inside families")][0]
                v1, v2 = rd.md_numbers(row[col])
                md_add("Variance due to wording (extra)", arm, v1, 100 * float(m["variance_shares"]["wording"]), "extra_exact",
                       100 * tj["variance_shares"]["wording"] if tj else None)
                md_add("Variance due to wording if sampling only (extra)", arm, v2, 100 * float(m["variance_shares"]["wording_if_sampling_only"]), "extra_exact",
                       100 * tj["variance_shares"]["wording_if_sampling_only"] if tj else None)
                md_add("Mean turns", arm, row["Mean turns"], float(m["mean_turns"]), json_value=tj["turn_stats"]["mean_turns"] if tj else None)
                md_add("Ended at the turn limit", arm, row["Ended at the turn limit"], 100 * float(m["share_at_turn_limit"]),
                       json_value=100 * tj["turn_stats"]["share_at_turn_limit"] if tj else None)
                for k2 in ("closed_correctly", "closed_wrongly", "never_closed", "malformed_only"):
                    md_add(k2, arm, row[k2], 100 * float(m["outcome_labels"][k2]), json_value=100 * tj["outcome_labels"][k2] if tj else None)
                col = [c for c in h if c.startswith("Smallest detectable effect")][0]
                nums = rd.md_numbers(row[col])
                sj = tj["smallest_detectable_effect"] if tj else {}
                md_add("Smallest detectable effect (extra)", arm, nums[0], 100 * sde0[arm]["mde"], "seed0", 100 * sj["mde"] if sj else None)
                md_add("Smallest detectable effect: splits (extra)", arm, nums[1], None, "setting", sj.get("n_splits"))
                md_add("Smallest detectable effect: lowest split (extra)", arm, nums[2], 100 * sde0[arm]["mde_lowest_split"], "seed0",
                       100 * sj["mde_lowest_split"] if sj else None)
                md_add("Smallest detectable effect: highest split (extra)", arm, nums[3], 100 * sde0[arm]["mde_highest_split"], "seed0",
                       100 * sj["mde_highest_split"] if sj else None)
    for kind_ in ("main", "ident", "more"):
        for arm in ARMS:
            if seen_rows[(kind_, arm)] != 1:
                md_unhandled.append("the %s table of the .md has %d rows for %s, 1 expected" % (kind_, seen_rows[(kind_, arm)], arm))
    for key in PAIRS:
        if seen_rows[("pair", key)] != 1:
            md_unhandled.append("the comparison table of the .md has %d rows for %s minus %s, 1 expected" % (seen_rows[("pair", key)], key[1], key[0]))
    for arm, dist in md_rewards.items():
        if arm not in ARMS:
            continue
        mine_d = M[arm]["reward_distribution"]
        json_d = {int(k): v for k, v in t_arms[arm]["reward_distribution"].items()} if arm in t_arms else {}
        for rv in sorted(set(dist) | set(mine_d)):
            md_rows.append({"table": NAME, "metric": "attempts with reward %d" % rv, "arm": arm, "printed": str(dist.get(rv)),
                            "mine": mine_d.get(rv), "kind": "exact", "equal_mine": dist.get(rv) == mine_d.get(rv),
                            "json_value": json_d.get(rv), "equal_json": dist.get(rv) == json_d.get(rv)})

    # ==============================================================================================
    # verdicts
    # ==============================================================================================
    exact_rows = [r for r in rows if r["kind"] in ("exact", "identity")]
    extra_rows = [r for r in rows if r["kind"] == "extra_exact"]
    json_bad = [r for r in exact_rows + extra_rows if not r["equal"]]
    seed0_rows = [r for r in rows if r["kind"] == "seed0"]
    seed0_bad = [r for r in seed0_rows if not r["equal"]]
    md_exact = [r for r in md_rows if r["kind"] in ("exact", "identity", "extra_exact")]
    md_bad = [r for r in md_exact if not r["equal_mine"]]
    md_seed0 = [r for r in md_rows if r["kind"] == "seed0"]
    md_seed0_bad = [r for r in md_seed0 if not r["equal_mine"]]
    md_json = [r for r in md_rows if r["equal_json"] is not None]
    md_json_bad = [r for r in md_rows if r["equal_json"] is False or (r["kind"] != "identity" and r["json_value"] is None
                                                                       and r["kind"] != "seed0")]
    mc_bad = [r for r in mc_rows if not r["inside_wide"]]
    mc_out95 = [r for r in mc_rows if not r["inside_band"]]
    mc_out_seeds = [r for r in mc_rows if not r["inside_seeds"]]

    def num_gap(r):
        try:
            return abs(float(r["table_value"]) - float(r["mine"]))
        except (TypeError, ValueError):
            return 0.0
    numeric = [r for r in exact_rows + extra_rows if r["kind"] != "identity" and r["table_value"] is not None]
    max_gap_exact = max([num_gap(r) for r in numeric] or [0.0])
    max_gap_seed0 = max([num_gap(r) for r in seed0_rows if r["table_value"] is not None] or [0.0])
    n_compared = len(exact_rows) + len(extra_rows) + len(md_exact) + len(md_json) + len(seed0_rows) + len(md_seed0) + len(mc_rows)
    n_differ = len(json_bad) + len(md_bad) + len(md_json_bad) + len(seed0_bad) + len(md_seed0_bad) + len(mc_bad)

    result = {
        "bad_checks": bad_checks, "structure": structure,
        "json_mismatch": [(r["table"], r["arm"], r["metric"]) for r in json_bad],
        "md_mismatch": [(r["table"], r["arm"], r["metric"]) for r in md_bad],
        "md_vs_json_mismatch": [(r["table"], r["arm"], r["metric"]) for r in md_json_bad],
        "outside_noise": [(NAME, r["what"], r["end"]) for r in mc_bad],
        "outside_95_band": [(NAME, r["what"], r["end"]) for r in mc_out95],
        "outside_200_seeds": [(NAME, r["what"], r["end"]) for r in mc_out_seeds],
        "seed0_mismatch": [(r["table"], r["arm"], r["metric"]) for r in seed0_bad],
        "md_seed0_mismatch": [(r["table"], r["arm"], r["metric"]) for r in md_seed0_bad],
        "md_unhandled": md_unhandled,
        "n_json": len(exact_rows) + len(extra_rows), "n_md": len(md_exact), "n_md_json": len(md_json),
        "n_seed0": len(seed0_rows), "n_md_seed0": len(md_seed0), "n_interval_rows": len(mc_rows),
        "n_compared": n_compared, "n_differ": n_differ, "n_not_remade": len(not_remade),
    }
    result["failed"] = any(result[k] for k in FAIL_KEYS)

    # ==============================================================================================
    # my_numbers_<suffix>.json
    # ==============================================================================================
    def clean(o):
        if isinstance(o, Fr):
            return rd.fr_json(o)
        if isinstance(o, dict):
            return {("%s minus %s" % (k[1], k[0]) if isinstance(k, tuple) and len(k) == 2 and k in pair else
                     ("/s".join(map(str, k)) if isinstance(k, tuple) else str(k))): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o
    dump = {
        "table": os.path.relpath(cfg["stem"], REPO),
        "inputs": {"episodes": {a: os.path.relpath(EPI[a], REPO) for a in ARMS}, "families": os.path.relpath(cfg["scenarios"], REPO)},
        "sha256_of_files_read": dict(hashes),
        "seeds": {"bootstrap": rd.MY_SEED, "noise_seeds": [rd.NOISE_SEEDS[0], rd.NOISE_SEEDS[-1]], "table_seed": rd.TABLE_SEED},
        "t_quantile_0.975_df%d" % (F - 1): {"beta_continued_fraction": tq_beta, "simpson": tq_simpson},
        "arms": {}, "pairs": {}, "checks": [list(c) for c in checks],
    }
    for arm in ARMS:
        m = dict(M[arm])
        for drop in ("families", "scen_pass", "scen_n", "fam_rate"):
            m.pop(drop)
        m["interval_my_seed"] = boot[arm]
        m["interval_table_seed_0"] = boot0[arm]
        m["interval_exact_bootstrap"] = dict(exact[arm])
        m["interval_200_seeds"] = {k: v for k, v in noise[arm].items() if not k.endswith("_all")}
        m["two_halves_table_seed_0"] = sde0[arm]
        m["section6_recheck"] = words[arm]
        dump["arms"][arm] = clean(m)
    for key in PAIRS:
        p = dict(pair[key])
        p["boot_table_seed_0"] = boot0[key]
        p["exact_bootstrap"] = exact[key]
        p["interval_200_seeds"] = {k: v for k, v in noise[key].items() if not k.endswith("_all")}
        p.update(tests[key])
        dump["pairs"]["%s minus %s" % (key[1], key[0])] = clean(p)
    dump["family_passes_of_32"] = {f: {arm: int(M[arm]["fam_pass"][f]) for arm in ARMS} for f in fam_order}
    numbers_name = "my_numbers_%s.json" % cfg["suffix"]
    report_name = "report_%s.md" % cfg["suffix"]
    with open(os.path.join(out_dir, numbers_name), "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=1, sort_keys=False)

    # ==============================================================================================
    # report_<suffix>.md
    # ==============================================================================================
    W = []
    w = W.append

    def pts(x, digits=2, sign=False):
        return ("%+." + str(digits) + "f" if sign else "%." + str(digits) + "f") % (100.0 * float(x))

    def show(x):
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

    def yesno(b):
        return "yes" if b else "NO"

    def signed(v):
        return "0" if v == 0 else "%+d" % v

    rel_stem = os.path.relpath(cfg["stem"], REPO)
    n_att = M[ARMS[0]]["n_episodes"]
    w("# The table `%s`, made again with separate code" % os.path.basename(rel_stem))
    w("")
    w("Made by `results/main/audit/recount/rederive_table.py --table %s`. It uses the functions of `rederive.py` in this folder (standard library and numpy; nothing imported from `accessdesk`). "
      "Every number in the column \"mine\" was counted from the files `results/main/<arm>_%s/episodes.jsonl` (%s) and from `%s`. "
      "Every \"table\" number is read from `%s.json` and `.md`. All my numbers are also in `%s` in this folder."
      % (which, split, ", ".join("`%s`" % a for a in ARMS), os.path.relpath(cfg["scenarios"], REPO), rel_stem, numbers_name))
    w("")
    if which == "test":
        w("When: written on 2026-11-19, after the test results were opened (commit 7fad426). It changes no number of the table; it only counts again from the saved attempts and compares. "
          "The definitions are the ones of `rederive.py` (written on 2026-11-18 for the dev tables, from `ANALYSIS_PLAN.md` and the column names); nothing was changed to fit the test numbers.")
    else:
        w("When: written on 2026-11-19. It changes no number of the table; it only counts again from the saved attempts and compares. The definitions are the ones of `rederive.py`.")
    w("")
    w("Units: a family is one world with one correct end state (%d here). A scenario is one family in one wording style (%d). An attempt is one conversation (%d per arm). \"Points\" are percentage points."
      % (F, M[ARMS[0]]["n_scenarios"], n_att))
    w("")
    w("## 1. In short")
    w("")
    if not result["failed"]:
        w("- **Every number without a random part is equal.** `.json` entries compared: %d (largest absolute difference %.1e). Printed `.md` entries compared with my values: %d. Mismatches: 0 and 0."
          % (result["n_json"], max_gap_exact, result["n_md"]))
    else:
        w("- **There are differences** (all listed in section 7): `.json` entries %d of %d; printed `.md` entries %d of %d; `.md` against `.json` %d of %d; input checks that failed %d; structure problems %d; parts of the `.md` not covered %d."
          % (len(json_bad), result["n_json"], len(md_bad), result["n_md"], len(md_json_bad), len(md_json), len(bad_checks), len(structure), len(md_unhandled)))
    w("- **Numbers with a random part** (bootstrap ends, smallest detectable effect), treated as in `rederive.py`: (1) with my own seed %d each end must lie inside the 99.9%% band that 10,000 draws leave around the exact bootstrap end: %d of %d inside; "
      "(2) with the table's seed 0, drawn as `numpy.random.default_rng(0).integers(0, %d, size=(10000, %d))`, a fresh generator for every interval, my code must give the table's number within 1e-12: %d of %d `.json` numbers and %d of %d printed `.md` numbers."
      % (rd.MY_SEED, len(mc_rows) - len(mc_bad), len(mc_rows), F, F, len(seed0_rows) - len(seed0_bad), len(seed0_rows), len(md_seed0) - len(md_seed0_bad), len(md_seed0)))
    w("- **The `.md` against its own `.json`:** %d printed numbers checked, %d are not the rounding of the `.json` value beside them." % (len(md_json), len(md_json_bad)))
    w("- **Passed attempts** (of %d): %s." % (n_att, "; ".join("%s %d (%.2f%%)" % (a, M[a]["n_passed"], 100 * float(M[a]["pass_rate"])) for a in ARMS)))
    w("- **Differences:** %s." % "; ".join(
        "%s minus %s %+d attempts of %d = %s points (paired bootstrap with seed 0: %s to %s; t-interval: %s to %s)"
        % (k[1], k[0], pair[k]["diff"] * (F * DEN), F * DEN, pts(pair[k]["diff"], 2, True), pts(boot0[k][0], 1, True), pts(boot0[k][1], 1, True),
           pts(pair[k]["t"][0], 1, True), pts(pair[k]["t"][1], 1, True)) for k in PAIRS))
    w("- **Tolerances:** `.json` numbers 1e-12 (my values are exact fractions); t-interval ends 1e-9 (my own t quantile); `.md` numbers: my value rounded to the printed digits (half a unit of the last digit).")
    w("- **Not made again:** %d settings and labels of the table code (section 7)." % len(not_remade))
    w("")

    w("## 2. Inputs and checks on them")
    w("")
    w("\"Want\" is what the design says (%d families x 4 styles x 8 attempts = %d)." % (F, n_att))
    w("")
    w("| Check | %s | Want | OK |" % " | ".join(ARMS))
    w("|---|%s---|---|" % ("---|" * len(ARMS)))
    order = []
    for arm, what, _g, _wt in checks:
        if arm in ARMS and what not in order:
            order.append(what)
    for what in order:
        got = {arm: [g for a2, w2, g, _wt in checks if a2 == arm and w2 == what][0] for arm in ARMS}
        want_ = [wt for a2, w2, _g, wt in checks if w2 == what][0]
        w("| %s | %s | %s | %s |" % (what, " | ".join(str(got[a]) for a in ARMS), want_, yesno(all(g == want_ for g in got.values()))))
    for arm, what, got, want_ in checks:
        if arm not in ARMS:
            w("| %s (%s) | %s%s | %s | %s |" % (what, arm, got, " |" * (len(ARMS) - 1), want_, yesno(got == want_)))
    w("")
    w("Harness errors (records with `harness_error` not null): %s. Every family has 32 valid attempts in every arm." % ", ".join("%s %d" % (a, M[a]["n_harness_errors"]) for a in ARMS))
    w("")
    w("| File read | SHA-256 |")
    w("|---|---|")
    for path, digest in hashes.items():
        w("| `%s` | `%s` |" % (path, digest))
    w("")
    w("How each number is counted: as in `report.md` section 3 in this folder (the same functions of `rederive.py`). "
      "The t quantile for %d degrees of freedom: %.12f (incomplete beta) and %.12f (Simpson's rule)." % (F - 1, tq_beta, tq_simpson))
    w("")

    w("## 3. Per-arm numbers: table against mine")
    w("")
    w("Column \"equal\": the `.json` value equals mine within 1e-12 and the printed `.md` number is my value rounded. Interval rows are in section 4.")
    w("")
    spec = [("Episodes", "n_episodes", "Episodes"), ("Harness errors", "n_harness_errors", "Harness errors"), ("Families", "n_families", None),
            ("Pass rate (family-weighted)", "pass_rate", "Pass rate")] + \
        [("Pass rate, outcome " + o, "pass_rate_by_outcome." + o, o) for o in rd.OUTCOMES] + \
        [("Pass rate, styles 1 to 3", "pass_rate_seen", "Seen styles"), ("Pass rate, style 4", "pass_rate_held_out", "Held-out style"),
         ("Scenarios with 8 equal rewards", "zero_variance_equal_reward", "Groups with equal rewards"),
         ("Scenarios all pass or all fail", "zero_variance_equal_passfail", "Groups with equal pass or fail"),
         ("Malformed / all assistant messages", "invalid_call_rate", "Malformed messages"), ("Pass@8", "pass_at_8", "Pass@8")] + \
        [("Pass^%d" % k, "pass_hat_k.%d" % k, "Pass^%d" % k) for k in rd.KS] + \
        [("Scenarios left out of Pass^%d" % k, "n_scenarios_left_out.%d" % k, None) for k in rd.KS] + \
        [("All wordings pass", "all_wordings_pass", "All wordings pass"), ("Mean turns", "turn_stats.mean_turns", "Mean turns"),
         ("Episodes (turn statistics)", "turn_stats.n_episodes", None),
         ("Ended at the turn limit", "turn_stats.share_at_turn_limit", "Ended at the turn limit"),
         ("Closed the request (share_closed)", "turn_stats.share_closed", None)] + \
        [(k, "outcome_labels." + k, k) for k in ("closed_correctly", "closed_wrongly", "never_closed", "malformed_only")] + \
        [("(extra) variance due to wording", "variance_shares.wording", "Variance due to wording (extra)"),
         ("(extra) variance due to sampling", "variance_shares.sampling", None),
         ("(extra) wording share if sampling only", "variance_shares.wording_if_sampling_only", "Variance due to wording if sampling only (extra)"),
         ("(extra) smallest detectable effect: families", "smallest_detectable_effect.n_families", None)]

    def md_lookup(metric, arm):
        for r in md_rows:
            if r["metric"] == metric and r["arm"] == arm:
                return r
        return None
    for arm in ARMS:
        w("### %s" % arm)
        w("")
        w("| Metric | Table .json | Table .md | Mine | Equal |")
        w("|---|---|---|---|---|")
        for label, path, mdm in spec:
            rs = [r for r in rows if r["arm"] == arm and r["metric"] == path]
            if not rs:
                w("| %s | (not in the table) | - | - | NO |" % label)
                continue
            mdr = md_lookup(mdm, arm) if mdm else None
            ok = all(r["equal"] for r in rs) and (mdr is None or mdr["equal_mine"])
            w("| %s | %s | %s | %s | %s |" % (label, show(rs[0]["table_value"]), mdr["printed"] if mdr else "-", show(rs[0]["mine"]), yesno(ok)))
        rs = [r for r in rows if r["arm"] == arm and r["metric"].startswith("reward_distribution.")]
        mdr = [r for r in md_rows if r["arm"] == arm and r["metric"].startswith("attempts with reward")]
        tdist = {int(r["metric"].split(".")[1]): r["table_value"] for r in rs if r["table_value"] is not None}
        ok = all(r["equal"] for r in rs) and all(x["equal_mine"] for x in mdr) and len(mdr) == len(M[arm]["reward_distribution"])
        w("| Attempts per reward value | %s | (the same list, %d printed numbers checked) | %s | %s |"
          % (", ".join("%d: %d" % kv for kv in sorted(tdist.items())), len(mdr), ", ".join("%d: %d" % kv for kv in M[arm]["reward_distribution"].items()), yesno(ok)))
        rs = [r for r in rows if r["arm"] == arm and r["kind"] == "identity"]
        mdr = [r for r in md_rows if r["arm"] == arm and r["kind"] == "identity"]
        w("| Split, rule set, tools, prompt, text and config hash | %s | %s | the same values, read from all records | %s |"
          % ("; ".join(sorted(set("%s=%s" % (r["metric"], show(r["table_value"])) for r in rs))), "; ".join(r["printed"] for r in mdr),
             yesno(all(r["equal"] for r in rs) and all(r["equal_mine"] for r in mdr))))
        w("")
        m = M[arm]
        w("Counts behind the rows for `%s`: passed %d of %d; per outcome %s; styles 1 to 3: %d of %d; style 4: %d of %d; scenarios with 8 equal rewards %d of %d; "
          "all pass %d, all fail %d; malformed messages %d of %d; ended at the turn limit %d; closed %d; labels %s."
          % (arm, m["n_passed"], m["n_episodes"], ", ".join("%s %d of %d" % (o, m["by_outcome_counts"][o][0], m["by_outcome_counts"][o][1]) for o in rd.OUTCOMES),
             m["seen_counts"][0], m["seen_counts"][1], m["held_out_counts"][0], m["held_out_counts"][1], m["n_scen_equal_reward"], m["n_scenarios"],
             m["n_scen_all_pass"], m["n_scen_all_fail"], m["n_bad"], m["n_msgs"], m["n_at_turn_limit"], m["n_closed"],
             ", ".join("%s %d" % kv for kv in m["label_counts"].items())))
        w("")
    rows_top = [r for r in rows if r["arm"] == "-"]
    w("Top-level entries of the `.json`: %s." % "; ".join("`%s` table %s, mine %s, %s" % (r["metric"][4:], show(r["table_value"]), show(r["mine"]), yesno(r["equal"])) for r in rows_top))
    w("")

    w("## 4. Comparisons and intervals")
    w("")
    w("### 4.1 Differences and t-intervals (no random draws)")
    w("")
    w("| Comparison | Difference: table (.md) | Difference: mine | Equal | t-interval: table (.md) | t-interval: mine | Equal |")
    w("|---|---|---|---|---|---|---|")
    for key in PAIRS:
        if key not in t_pairs:
            continue
        e, p = t_pairs[key], pair[key]
        label = "%s minus %s" % (key[1], key[0])
        r_d = [r for r in rows if r["arm"] == label and r["metric"] == "pair.diff"]
        r_t = [r for r in rows if r["arm"] == label and r["metric"] in ("pair.t_lo", "pair.t_hi")]
        md_d = md_lookup("Difference in points", label)
        md_t = [md_lookup("t-interval, lower end", label), md_lookup("t-interval, upper end", label)]
        md_ok = md_d is not None and all(x is not None for x in md_t)
        w("| %s | %+.10f (%s) | %+.10f (= %+d/%d) | %s | %+.6f to %+.6f (%s to %s) | %+.6f to %+.6f | %s |"
          % (label, 100 * e["diff"], md_d["printed"] if md_d else "-", 100 * float(p["diff"]), p["diff"].numerator, p["diff"].denominator,
             yesno(all(r["equal"] for r in r_d) and md_d is not None and md_d["equal_mine"]),
             100 * e["t_lo"], 100 * e["t_hi"], md_t[0]["printed"] if md_ok else "-", md_t[1]["printed"] if md_ok else "-",
             100 * p["t"][0], 100 * p["t"][1], yesno(all(r["equal"] for r in r_t) and md_ok and all(x["equal_mine"] for x in md_t))))
    w("")
    w("### 4.2 Bootstrap ends (random draws: the table names seed 0; I used seed %d and also seed 0)" % rd.MY_SEED)
    w("")
    w("In points. \"Exact\": the exact bootstrap end (no random draws; every family value is a whole number of passes of 32, so the distribution of the mean is a convolution). "
      "\"99.9%% band\": where a 10,000-draw run is expected to put the end (+- 3.29 standard errors of the level). One grid step is 1/%d = %.4f points." % (F * DEN, 100.0 / (F * DEN)))
    w("")
    w("| Quantity | End | Table | Mine, seed %d | Mine, seed 0 | Seed 0 equals the table (1e-12) | Exact | 95%% band | 99.9%% band | 200 seeds: min to max | Table inside the 99.9%% band |" % rd.MY_SEED)
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in mc_rows:
        w("| %s | %s | %.4f | %.4f | %.4f | %s | %.4f | %.4f to %.4f | %.4f to %.4f | %.4f to %.4f | %s |"
          % (r["what"], "lower" if r["end"] == "lo" else "upper", 100 * r["table_value"], 100 * r["mine"], 100 * r["mine0"], yesno(r["equal0"]),
             100 * r["exact"], 100 * r["band"][0], 100 * r["band"][1], 100 * r["band_wide"][0], 100 * r["band_wide"][1],
             100 * r["seed_min"], 100 * r["seed_max"], yesno(r["inside_wide"])))
    w("")
    w("Ends of the table inside the 95%% band: %d of %d; inside the 99.9%% band: %d of %d; inside the range of my 200 seeds (%d to %d): %d of %d. "
      "About 1 end in 20 is expected outside the 95%% band by chance alone; only the 99.9%% band counts as a difference."
      % (len(mc_rows) - len(mc_out95), len(mc_rows), len(mc_rows) - len(mc_bad), len(mc_rows), rd.NOISE_SEEDS[0], rd.NOISE_SEEDS[-1],
         len(mc_rows) - len(mc_out_seeds), len(mc_rows)))
    w("")
    w("Exact bootstrap intervals beside the table's, in points:")
    w("")
    w("| Quantity | Table interval (seed 0) | Exact bootstrap interval |")
    w("|---|---|---|")
    for qn in quantities:
        label = qn if isinstance(qn, str) else "%s minus %s" % (qn[1], qn[0])
        tl = [r for r in mc_rows if r["what"] == label and r["end"] == "lo"]
        th = [r for r in mc_rows if r["what"] == label and r["end"] == "hi"]
        if not tl or not th:
            continue
        fmt = "| %s | %.2f to %.2f | %.2f to %.2f |" if isinstance(qn, str) else "| %s | %+.2f to %+.2f | %+.2f to %+.2f |"
        w(fmt % (label, 100 * tl[0]["table_value"], 100 * th[0]["table_value"], 100 * exact[qn]["lo"], 100 * exact[qn]["hi"]))
    w("")
    w("### 4.3 Smallest detectable effect (200 random splits; my code with the table's seed 0, as in `report.md` section 8.2)")
    w("")
    w("| Arm | Number | Table .json | Mine, seed 0 | Equal (1e-12) |")
    w("|---|---|---|---|---|")
    for arm in ARMS:
        for k2 in ("se_halves", "se", "mde", "mde_lowest_split", "mde_highest_split"):
            rs = [r for r in rows if r["arm"] == arm and r["metric"] == "smallest_detectable_effect." + k2]
            w("| %s | %s | %s | %s | %s |" % (arm, k2, show(rs[0]["table_value"]) if rs else "(not in the table)", repr(sde0[arm][k2]), yesno(bool(rs) and all(r["equal"] for r in rs))))
    w("")
    w("### 4.4 A second view beside the bootstrap (not a table number): exact sign test and exact sign-flip test on the %d family differences" % F)
    w("")
    w("| Comparison | Families up | down | zero | Sign test p, two-sided | Sign-flip p, two-sided |")
    w("|---|---|---|---|---|---|")
    for key in PAIRS:
        st, pt = tests[key]["sign"], tests[key]["perm"]
        w("| %s minus %s | %d | %d | %d | %.3g | %.3g |" % (key[1], key[0], st["pos"], st["neg"], st["zero"], float(st["p_two_sided"]), float(pt["p_two_sided"])))
    w("")
    w("No correction for the number of comparisons. The families are treated as independent draws.")
    w("")

    w("## 5. Passes of 32 per family")
    w("")
    w("From the `episodes.jsonl` files (`passed` true, per `family_id`); outcome from `%s`." % os.path.relpath(cfg["scenarios"], REPO))
    w("")
    diff_pairs = [k for k in PAIRS if k[0] != "base"] or PAIRS
    w("| Family | Expected outcome | %s | %s |" % (" | ".join(ARMS), " | ".join("%s minus %s" % (k[1], k[0]) for k in diff_pairs)))
    w("|---|---|%s%s" % ("---|" * len(ARMS), "---|" * len(diff_pairs)))
    for i, f in enumerate(fam_order):
        w("| %s | %s | %s | %s |" % (f, fams[f]["outcome"], " | ".join(str(int(C[a][i])) for a in ARMS), " | ".join(signed(int(C[k[1]][i] - C[k[0]][i])) for k in diff_pairs)))
    w("| Sum of %d | | %s | %s |" % (F * DEN, " | ".join(str(int(C[a].sum())) for a in ARMS), " | ".join(signed(int(C[k[1]].sum() - C[k[0]].sum())) for k in diff_pairs)))
    w("")

    w("## 6. Pass and reward, checked against the words of section 6 of the plan (not a table number)")
    w("")
    w("The saved `final_state` against the family's expected end state (default rule set) in the scenario file, and the reward formula (0 for a fail; 100 minus 5 per malformed message, at most 15, minus 2 per turn over the budget, at most 10).")
    w("")
    w("| Arm | Attempts | `passed` differs (records as lists) | `passed` differs (records as sets) | Attempts with a doubled approval or grant record | `reward` differs from the formula |")
    w("|---|---|---|---|---|---|")
    for arm in ARMS:
        x = words[arm]
        w("| %s | %d | %d | %d | %d | %d |" % (arm, x["n"], x["pass_multiset_disagree"], x["pass_set_disagree"], x["doubled_record"], x["reward_disagree"]))
    w("")
    w("This compares saved end states only; it replays no tool call.")
    w("")

    w("## 7. Every difference")
    w("")
    any_mis = False
    for title, items, fmt in (
            ("Input checks that failed", bad_checks, lambda c: "%s: %s: found %s, want %s" % c),
            ("Structure of the table", structure, lambda s: "%s: %s" % s),
            ("`.json` entries without a random part that differ from my value", json_bad,
             lambda r: "arm %s, `%s`: table %s, mine %s %s" % (r["arm"], r["metric"], show(r["table_value"]), show(r["mine"]), r["note"])),
            ("`.md` numbers without a random part that are not my value rounded", md_bad,
             lambda r: "arm %s, %s: printed %s, mine %s" % (r["arm"], r["metric"], r["printed"], r["mine"])),
            ("`.md` numbers that are not the rounding of the `.json` value beside them", md_json_bad,
             lambda r: "arm %s, %s: printed %s, .json %s" % (r["arm"], r["metric"], r["printed"], r["json_value"])),
            ("Interval ends outside the 99.9% Monte Carlo band", mc_bad,
             lambda r: "%s, %s end: table %.4f, exact %.4f, 99.9%% band %.4f to %.4f (points)" % (r["what"], r["end"], 100 * r["table_value"], 100 * r["exact"], 100 * r["band_wide"][0], 100 * r["band_wide"][1])),
            ("Numbers with a random part that my code does not give exactly with seed 0", seed0_bad + md_seed0_bad,
             lambda r: "arm %s, %s: table %s, mine with seed 0 %s" % (r["arm"], r["metric"], show(r.get("table_value", r.get("printed"))), show(r["mine"]))),
            ("Parts of the `.md` file that the comparison does not cover", md_unhandled, lambda s: s)):
        if items:
            any_mis = True
            w("%s:" % title)
            w("")
            for it in items:
                w("- " + fmt(it))
            w("")
    if not any_mis:
        w("None.")
        w("")
    w("Table entries that I did not make again, and why (settings and labels, not results):")
    w("")
    grouped = collections.OrderedDict()
    for path, val, why in not_remade:
        short = re.sub(r"^arms\[[^\]]+\]\.", "arms[each arm].", path)
        grouped.setdefault((short, show(val), why), 0)
        grouped[(short, show(val), why)] += 1
    for (short, val, why), n in grouped.items():
        w("- `%s` = %s (%d entr%s): %s" % (short, val if val != "" else "an empty list", n, "y" if n == 1 else "ies", why))
    w("")

    w("## 8. What this does and does not show")
    w("")
    if not result["failed"]:
        w("- It shows that every number of this table follows from the saved attempts by the definitions of `rederive.py`: equal for every number without a random part, and equal for every number with a random part once my code uses the table's seed 0.")
    else:
        w("- There are differences (section 7), so the table is not confirmed as it stands.")
    w("- It shows nothing about the saved attempts themselves (whether the harness ran the model correctly, whether the verifier is right). Section 6 only checks that `passed` and `reward` agree with the saved end states and the reward formula.")
    w("- Both programs follow the same plan. Agreement shows that the table code does what the plan says, not that the plan measures the right thing.")
    w("- The intervals treat the %d families as a sample of families. They do not cover the choice of training seed or the run seed of the rollouts." % F)
    w("- The way to draw with seed 0 was found by trial on the dev tables (`seed0_trials.py`); here it is used unchanged.")
    w("- The claim that the code was written without reading the evaluation code (`accessdesk/eval/`, `accessdesk/audit/`) is the statement of the agents that wrote it; nobody checked it from outside.")
    w("")
    w("## 9. How to run this again")
    w("")
    w("From the top folder of the repository (no model, no torch; about %s):" % ("30 seconds" if len(ARMS) > 2 else "15 seconds"))
    w("")
    w("- `.venv/bin/python results/main/audit/recount/rederive_table.py --table %s` writes `%s` and `%s` into this folder (or into `--out DIR`); exit code 0 when nothing differs, 1 otherwise."
      % (which, report_name, numbers_name))
    w("")
    w("All seeds are fixed (bootstrap %d; noise study %d to %d; the table's seed 0), so a second run gives the same report." % (rd.MY_SEED, rd.NOISE_SEEDS[0], rd.NOISE_SEEDS[-1]))
    with open(os.path.join(out_dir, report_name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(W) + "\n")

    if quiet:
        return result
    # ---------------- console summary ----------------
    print("table %s; families %d; records per arm %s" % (rel_stem, F, {a: M[a]["n_records"] for a in ARMS}))
    print("input checks failed: %d; structure problems: %d" % (len(bad_checks), len(structure)))
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
    print(".json numbers without a random part: %d compared, mismatches %d, largest absolute difference %.2e" % (result["n_json"], len(json_bad), max_gap_exact))
    print(".md printed numbers without a random part: %d compared, mismatches %d" % (len(md_exact), len(md_bad)))
    print(".md printed numbers against their .json value: %d compared, mismatches %d" % (len(md_json), len(md_json_bad)))
    print("interval ends: %d; outside the 95%% band %d; outside the 99.9%% band %d; outside the range of 200 seeds %d"
          % (len(mc_rows), len(mc_out95), len(mc_bad), len(mc_out_seeds)))
    print("numbers with a random part, my code with seed 0: .json %d compared, mismatches %d, largest absolute difference %.2e; .md %d compared, mismatches %d"
          % (len(seed0_rows), len(seed0_bad), max_gap_seed0, len(md_seed0), len(md_seed0_bad)))
    print("table entries not made again (settings and labels): %d; parts of the .md not covered: %d" % (len(not_remade), len(md_unhandled)))
    for key2 in ("structure", "json_mismatch", "md_mismatch", "md_vs_json_mismatch", "outside_noise", "seed0_mismatch", "md_seed0_mismatch"):
        for item in result[key2]:
            print("  %s: %s" % (key2.upper(), " | ".join(str(x) for x in item)))
    print("in all: %d comparisons (the lines above added up), %d differ" % (n_compared, n_differ))
    shown = os.path.relpath(out_dir)
    print("wrote %s and %s in %s" % (report_name, numbers_name, out_dir if shown.startswith("..") else shown))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The recount of rederive.py for the test table or the dev table base_vs_sft.")
    parser.add_argument("--table", choices=sorted(CHOICES), default="test", help="which table (default: test)")
    parser.add_argument("--out", help="the folder for the report and the numbers (default: this script's folder)")
    parser.add_argument("--tables", help="a folder with copies of the table files (same names), to compare against them instead")
    args = parser.parse_args()
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    stem_arg = os.path.join(args.tables, os.path.basename(CHOICES[args.table]["stem"])) if args.tables else None
    res = main(which=args.table, stem=stem_arg, out_dir=args.out)
    if res["failed"]:
        sys.stderr.write("exit code 1: %d of %d comparisons differ; input checks failed %d; structure problems %d; parts of the .md not covered %d\n"
                         % (res["n_differ"], res["n_compared"], len(res["bad_checks"]), len(res["structure"]), len(res["md_unhandled"])))
        sys.exit(1)
    sys.exit(0)
