#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest_compare.py - does the comparison in rederive.py notice a wrong table?

It copies the two result tables (.json and .md) into a new temporary folder, plants
one error on purpose in the copy, runs the comparison of rederive.py against the copy, and checks
that exactly the planted error is reported and nothing else. The real tables are only read.
Test 0 runs the unchanged copies: nothing may be reported.

Run from the top folder of the repository:
      .venv/bin/python results/main/audit/recount/selftest_compare.py > results/main/audit/recount/selftest_compare.txt
(Moved on 2026-11-19 from the working folder of the build; changed then: this line, and the copies go into a
new temporary folder, not into _selftest/ inside this folder.)
"""

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rederive as rd  # noqa: E402  (my own script in this folder)

WORK = tempfile.mkdtemp(prefix="recount_selftest_")
KEYS = ("json_mismatch", "md_mismatch", "md_vs_json_mismatch", "outside_noise", "seed0_mismatch", "md_seed0_mismatch")


def fresh_copies():
    os.makedirs(os.path.join(WORK, "tables"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "out"), exist_ok=True)
    stems = {}
    for name, stem in rd.TABLES.items():
        new = os.path.join(WORK, "tables", os.path.basename(stem))
        for ext in (".json", ".md"):
            shutil.copyfile(stem + ext, new + ext)
        stems[name] = new
    return stems


def edit_json(stem, fn):
    with open(stem + ".json", "r", encoding="utf-8") as fh:
        d = json.load(fh)
    fn(d)
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)


def edit_md(stem, old, new):
    with open(stem + ".md", "r", encoding="utf-8") as fh:
        text = fh.read()
    if text.count(old) != 1:
        raise SystemExit("selftest: the text to change must occur once in %s.md, found %d times: %r" % (stem, text.count(old), old))
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(text.replace(old, new))


def arm(d, name):
    return [e for e in d["arms"] if e["arm"] == name][0]


def pair(d, a, b):
    return [e for e in d["pairs"] if e["a"] == a and e["b"] == b][0]


def t1(s):
    edit_json(s["seed1"], lambda d: arm(d, "sft").__setitem__("pass_rate_seen", arm(d, "sft")["pass_rate_seen"] + 1e-6))


def t2(s):
    edit_json(s["seed1"], lambda d: arm(d, "grpo_seed1")["reward_distribution"].__setitem__("100", 1511))


def t3(s):
    edit_json(s["seed2"], lambda d: pair(d, "sft", "grpo_seed2").__setitem__("t_lo", pair(d, "sft", "grpo_seed2")["t_lo"] + 1e-6))


def t4(s):
    edit_json(s["seed2"], lambda d: arm(d, "grpo_seed2").__setitem__("pass_rate_lo", arm(d, "grpo_seed2")["pass_rate_lo"] - 0.01))


def t5(s):
    edit_md(s["seed1"], "| base | 1536 | 0 | 39.8 |", "| base | 1536 | 0 | 39.9 |")


def t6(s):
    edit_md(s["seed2"], "| 15.8 (9.7) | 3.59 |", "| 15.8 (9.7) | 3.60 |")


def t7(s):
    def swap(d):
        lab = arm(d, "base")["outcome_labels"]
        lab["closed_wrongly"], lab["never_closed"] = lab["never_closed"], lab["closed_wrongly"]
    edit_json(s["seed1"], swap)


def t8(s):
    def f(d):
        arm(d, "sft")["smallest_detectable_effect"]["mde"] *= 1.01
    edit_json(s["seed2"], f)


def t9(s):
    def f(d):
        arm(d, "grpo_seed2")["variance_shares"]["wording"] += 1e-6
    edit_json(s["seed2"], f)


def t10(s):
    def f(d):
        a, b = arm(d, "sft"), arm(d, "grpo_seed1")
        a["arm"], b["arm"] = "grpo_seed1", "sft"
    edit_json(s["seed1"], f)


def t11(s):
    def f(d):
        pair(d, "sft", "grpo_seed1")["diff"] *= -1
    edit_json(s["seed1"], f)


def t12(s):
    edit_md(s["seed1"], "| +9.2 | +3.1 to +16.7 |", "| +9.2 | +3.1 to +17.7 |")


def t13(s):
    def f(d):
        arm(d, "base")["pass_rate_by_outcome"]["grant"], arm(d, "base")["pass_rate_by_outcome"]["approval"] = \
            arm(d, "base")["pass_rate_by_outcome"]["approval"], arm(d, "base")["pass_rate_by_outcome"]["grant"]
    edit_json(s["seed2"], f)


def t14(s):
    def f(d):
        pair(d, "base", "grpo_seed2")["hi"] += 1.0 / 1536      # one grid step: inside the noise, but not what seed 0 gives
    edit_json(s["seed2"], f)


TESTS = [
    ("0  unchanged copies", None, {}),
    ("1  .json seed1: sft pass_rate_seen + 1e-6", t1,
     {"json_mismatch": {("seed1", "sft", "pass_rate_seen")}}),
    ("2  .json seed1: grpo_seed1 attempts with reward 100: 1512 -> 1511", t2,
     {"json_mismatch": {("seed1", "grpo_seed1", "reward_distribution.100")},
      "md_vs_json_mismatch": {("seed1", "grpo_seed1", "attempts with reward 100")}}),
    ("3  .json seed2: t_lo of grpo_seed2 minus sft + 1e-6", t3,
     {"json_mismatch": {("seed2", "grpo_seed2 minus sft", "pair.t_lo")}}),
    ("4  .json seed2: grpo_seed2 pass_rate_lo - 0.01 (one point)", t4,
     {"outside_noise": {("seed2", "grpo_seed2", "lo")},
      "seed0_mismatch": {("seed2", "grpo_seed2", "pass_rate_lo (my code, seed 0)")},
      "md_vs_json_mismatch": {("seed2", "grpo_seed2", "Interval, lower end")}}),
    ("5  .md seed1: base pass rate 39.8 -> 39.9", t5,
     {"md_mismatch": {("seed1", "base", "Pass rate")}, "md_vs_json_mismatch": {("seed1", "base", "Pass rate")}}),
    ("6  .md seed2: sft mean turns 3.59 -> 3.60", t6,
     {"md_mismatch": {("seed2", "sft", "Mean turns")}, "md_vs_json_mismatch": {("seed2", "sft", "Mean turns")}}),
    ("7  .json seed1: base closed_wrongly and never_closed swapped", t7,
     {"json_mismatch": {("seed1", "base", "outcome_labels.closed_wrongly"), ("seed1", "base", "outcome_labels.never_closed")},
      "md_vs_json_mismatch": {("seed1", "base", "closed_wrongly"), ("seed1", "base", "never_closed")}}),
    ("8  .json seed2: sft smallest detectable effect mde x 1.01", t8,
     {"seed0_mismatch": {("seed2", "sft", "smallest_detectable_effect.mde")}}),
    ("9  .json seed2: grpo_seed2 variance share of wording + 1e-6", t9,
     {"json_mismatch": {("seed2", "grpo_seed2", "variance_shares.wording")}}),
    ("10 .json seed1: the entries of sft and grpo_seed1 carry each other's name", t10, "many"),
    ("11 .json seed1: sign of the difference grpo_seed1 minus sft turned", t11,
     {"json_mismatch": {("seed1", "grpo_seed1 minus sft", "pair.diff")},
      "md_vs_json_mismatch": {("seed1", "grpo_seed1 minus sft", "Difference in points")}}),
    ("12 .md seed1: upper end of the paired bootstrap of grpo_seed1 minus sft +16.7 -> +17.7", t12,
     {"md_vs_json_mismatch": {("seed1", "grpo_seed1 minus sft", "Paired bootstrap, upper end")},
      "md_seed0_mismatch": {("seed1", "grpo_seed1 minus sft", "Paired bootstrap, upper end (my code, seed 0)")}}),
    ("13 .json seed2: base pass rates of grant and approval swapped", t13,
     {"json_mismatch": {("seed2", "base", "pass_rate_by_outcome.grant"), ("seed2", "base", "pass_rate_by_outcome.approval")},
      "md_vs_json_mismatch": {("seed2", "base", "grant"), ("seed2", "base", "approval")}}),
    ("14 .json seed2: upper bootstrap end of grpo_seed2 minus base + one grid step (1/1536)", t14,
     {"seed0_mismatch": {("seed2", "grpo_seed2 minus base", "pair.hi (my code, seed 0)")}}),
]


def main():
    n_ok = 0
    for label, plant, expect in TESTS:
        stems = fresh_copies()
        if plant is not None:
            plant(stems)
        res = rd.main(tables=stems, out_dir=os.path.join(WORK, "out"), quiet=True)
        got = {k: set(tuple(x) for x in res[k]) for k in KEYS}
        if expect == "many":
            n_flags = sum(len(v) for v in got.values())
            ok = n_flags >= 40 and ("seed1", "sft", "pass_rate") in got["json_mismatch"] and \
                ("seed1", "grpo_seed1", "pass_rate") in got["json_mismatch"]
            detail = "%d entries reported (at least 40 wanted, with the pass rate of both arms among them)" % n_flags
        else:
            want = {k: set(expect.get(k, set())) for k in KEYS}
            ok = got == want
            parts = []
            for k in KEYS:
                if got[k] or want[k]:
                    parts.append("%s: reported %s%s" % (k, sorted(got[k]), "" if got[k] == want[k] else "  BUT WANTED %s" % sorted(want[k])))
            detail = "; ".join(parts) if parts else "nothing reported"
        n_ok += ok
        print("%s  test %s" % ("PASS" if ok else "FAIL", label))
        print("      %s" % detail)
    print()
    print("%d of %d tests pass" % (n_ok, len(TESTS)))
    shutil.rmtree(WORK, ignore_errors=True)
    return 0 if n_ok == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
