#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest_table.py - does the comparison in rederive_table.py notice a wrong test table?

Like selftest_compare.py, for the test table: it copies
results/main/tables/test_base_sft_grpo_seed1_seed2.json and .md into a new temporary folder, plants one
error on purpose in the copy, runs the comparison of rederive_table.py against the copy, and checks that
exactly the planted error is reported and nothing else. The real table is only read.
Test 0 runs the unchanged copy: nothing may be reported. At the end it runs rederive_table.py once more
as a command on a copy with one planted change and checks the exit code (1), and on the unchanged copy (0).

Written on 2026-11-19, after the test results were opened. Run from the top folder of the repository
(about 5 minutes; about 100 MB of memory):
      .venv/bin/python results/main/audit/recount/selftest_table.py > results/main/audit/recount/selftest_table.txt
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rederive_table as rt  # noqa: E402  (my own script in this folder)

WORK = tempfile.mkdtemp(prefix="recount_selftest_table_")
STEM = rt.CHOICES["test"]["stem"]
KEYS = ("bad_checks", "structure", "json_mismatch", "md_mismatch", "md_vs_json_mismatch", "outside_noise",
        "seed0_mismatch", "md_seed0_mismatch", "md_unhandled")
T = "test"


def fresh_copy():
    os.makedirs(os.path.join(WORK, "tables"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "out"), exist_ok=True)
    new = os.path.join(WORK, "tables", os.path.basename(STEM))
    for ext in (".json", ".md"):
        shutil.copyfile(STEM + ext, new + ext)
    return new


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


def t1(s):    # a pass rate in the .json
    edit_json(s, lambda d: arm(d, "grpo_seed1").__setitem__("pass_rate", arm(d, "grpo_seed1")["pass_rate"] + 1e-6))


def t2(s):    # a pass rate in the .md
    edit_md(s, "| sft | 1536 | 0 | 91.3 |", "| sft | 1536 | 0 | 91.4 |")


def t3(s):    # an interval end in the .json
    edit_json(s, lambda d: pair(d, "sft", "grpo_seed1").__setitem__("lo", pair(d, "sft", "grpo_seed1")["lo"] - 0.01))


def t4(s):    # an interval end in the .md
    edit_md(s, "| +3.9 to +13.6 |", "| +3.9 to +14.6 |")


def t5(s):    # an interval end in the .json, by one grid step only (inside the noise, but not what seed 0 gives)
    edit_json(s, lambda d: arm(d, "base").__setitem__("pass_rate_hi", arm(d, "base")["pass_rate_hi"] + 1.0 / 1536))


def t6(s):    # a count in the .json
    edit_json(s, lambda d: arm(d, "grpo_seed2")["reward_distribution"].__setitem__("100", 1523))


def t7(s):    # a count in the .md
    edit_md(s, "- grpo_seed1: reward 0: 7,", "- grpo_seed1: reward 0: 8,")


def t8(s):    # a count in the .json: episodes of base
    edit_json(s, lambda d: arm(d, "base").__setitem__("n_episodes", 1535))


def t9(s):    # the sign of a difference
    def f(d):
        pair(d, "sft", "grpo_seed2")["diff"] *= -1
    edit_json(s, f)


def t10(s):   # an end of a t-interval
    edit_json(s, lambda d: pair(d, "base", "sft").__setitem__("t_hi", pair(d, "base", "sft")["t_hi"] + 1e-6))


def t11(s):   # the smallest detectable effect
    def f(d):
        arm(d, "sft")["smallest_detectable_effect"]["mde"] *= 1.01
    edit_json(s, f)


def t12(s):   # a comparison left out of the .json
    def f(d):
        d["pairs"] = [e for e in d["pairs"] if not (e["a"] == "sft" and e["b"] == "grpo_seed2")]
    edit_json(s, f)


TESTS = [
    ("0  unchanged copy", None, {}),
    ("1  .json: grpo_seed1 pass_rate + 1e-6", t1,
     {"json_mismatch": {(T, "grpo_seed1", "pass_rate")}}),
    ("2  .md: sft pass rate 91.3 -> 91.4", t2,
     {"md_mismatch": {(T, "sft", "Pass rate")}, "md_vs_json_mismatch": {(T, "sft", "Pass rate")}}),
    ("3  .json: lower bootstrap end of grpo_seed1 minus sft - 0.01 (one point)", t3,
     {"outside_noise": {(T, "grpo_seed1 minus sft", "lo")},
      "seed0_mismatch": {(T, "grpo_seed1 minus sft", "pair.lo (my code, seed 0)")},
      "md_vs_json_mismatch": {(T, "grpo_seed1 minus sft", "Paired bootstrap, lower end")}}),
    ("4  .md: upper bootstrap end of grpo_seed1 minus sft +13.6 -> +14.6", t4,
     {"md_vs_json_mismatch": {(T, "grpo_seed1 minus sft", "Paired bootstrap, upper end")},
      "md_seed0_mismatch": {(T, "grpo_seed1 minus sft", "Paired bootstrap, upper end (my code, seed 0)")}}),
    ("5  .json: upper interval end of base + one grid step (1/1536)", t5,
     {"seed0_mismatch": {(T, "base", "pass_rate_hi (my code, seed 0)")}}),
    ("6  .json: grpo_seed2 attempts with reward 100: 1524 -> 1523", t6,
     {"json_mismatch": {(T, "grpo_seed2", "reward_distribution.100")},
      "md_vs_json_mismatch": {(T, "grpo_seed2", "attempts with reward 100")}}),
    ("7  .md: grpo_seed1 attempts with reward 0: 7 -> 8", t7,
     {"md_mismatch": {(T, "grpo_seed1", "attempts with reward 0")},
      "md_vs_json_mismatch": {(T, "grpo_seed1", "attempts with reward 0")}}),
    ("8  .json: base n_episodes 1536 -> 1535", t8,
     {"json_mismatch": {(T, "base", "n_episodes")}, "md_vs_json_mismatch": {(T, "base", "Episodes")}}),
    ("9  .json: sign of the difference grpo_seed2 minus sft turned", t9,
     {"json_mismatch": {(T, "grpo_seed2 minus sft", "pair.diff")},
      "md_vs_json_mismatch": {(T, "grpo_seed2 minus sft", "Difference in points")}}),
    ("10 .json: t_hi of sft minus base + 1e-6", t10,
     {"json_mismatch": {(T, "sft minus base", "pair.t_hi")}}),
    ("11 .json: sft smallest detectable effect mde x 1.01", t11,
     {"seed0_mismatch": {(T, "sft", "smallest_detectable_effect.mde")}}),
    ("12 .json: the comparison grpo_seed2 minus sft removed", t12, "structure"),
]


def main():
    n_ok = 0
    for label, plant, expect in TESTS:
        stem = fresh_copy()
        if plant is not None:
            plant(stem)
        res = rt.main(which="test", stem=stem, out_dir=os.path.join(WORK, "out"), quiet=True)
        got = {k: set(tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in res[k]) for k in KEYS}
        if expect == "structure":
            ok = any("missing in the table" in s[1] for s in got["structure"]) and res["failed"]
            detail = "structure: reported %s; and the .md rows of this comparison have no .json value: %d reported" % (
                sorted(got["structure"]), len(got["md_vs_json_mismatch"]))
        else:
            want = {k: set(expect.get(k, set())) for k in KEYS}
            ok = got == want and res["failed"] == (plant is not None)
            parts = []
            for k in KEYS:
                if got[k] or want[k]:
                    parts.append("%s: reported %s%s" % (k, sorted(got[k]), "" if got[k] == want[k] else "  BUT WANTED %s" % sorted(want[k])))
            detail = "; ".join(parts) if parts else "nothing reported"
        n_ok += ok
        print("%s  test %s" % ("PASS" if ok else "FAIL", label))
        print("      %s" % detail)
    # the exit code of the command itself
    script = os.path.join(HERE, "rederive_table.py")
    codes = {}
    for label, plant in (("unchanged copy", None), ("copy with the change of test 2", t2)):
        fresh_copy()
        if plant is not None:
            plant(os.path.join(WORK, "tables", os.path.basename(STEM)))
        p = subprocess.run([sys.executable, script, "--tables", os.path.join(WORK, "tables"), "--out", os.path.join(WORK, "out")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, universal_newlines=True)
        codes[label] = p.returncode
        print("command on the %s: exit code %d%s" % (label, p.returncode, ("; standard error: " + p.stderr.strip()) if p.stderr.strip() else ""))
    ok = codes == {"unchanged copy": 0, "copy with the change of test 2": 1}
    n_ok += ok
    print("%s  test 13 exit codes (0 wanted for the unchanged copy, 1 for the changed copy)" % ("PASS" if ok else "FAIL"))
    print()
    print("%d of %d tests pass" % (n_ok, len(TESTS) + 1))
    shutil.rmtree(WORK, ignore_errors=True)
    return 0 if n_ok == len(TESTS) + 1 else 1


if __name__ == "__main__":
    sys.exit(main())
