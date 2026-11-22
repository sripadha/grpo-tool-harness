#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed0_trials.py - which way of drawing random numbers with seed 0 gives the table's numbers?

The tables say "seed 0" for the bootstrap, and the plan says "split at random into two halves
200 times" for the smallest detectable effect. Neither says which random generator is used or how
it is called. This script tries several natural ways, all with seed 0, and prints for each one how
many of the table's numbers come out equal (tolerance 1e-12). It reads the same inputs as
rederive.py and uses its readers. It imports nothing from the package accessdesk.

Run from the top folder of the repository:
      .venv/bin/python results/main/audit/recount/seed0_trials.py > results/main/audit/recount/seed0_trials.txt
(Moved on 2026-11-19 from the working folder of the build; only this line changed.)
"""

import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rederive as rd  # noqa: E402  (my own script in this folder)

TOL = 1e-12
B = rd.N_DRAWS


def main():
    fams, fam_order = rd.read_families(rd.DEV)
    F = len(fam_order)
    data = {arm: rd.read_attempts(rd.EPISODES[arm], fams) for arm in rd.ARMS}
    Y = {arm: rd.passes_array(data[arm], fam_order) for arm in rd.ARMS}          # (48, 4, 8)
    R = {arm: Y[arm].mean(axis=(1, 2)) for arm in rd.ARMS}                        # family pass rates

    T = {}
    for name, stem in rd.TABLES.items():
        with open(stem + ".json", "r", encoding="utf-8") as fh:
            T[name] = json.load(fh)
    ends, sde_t = {}, {}
    for name in T:
        for e in T[name]["arms"]:
            ends[e["arm"]] = (e["pass_rate_lo"], e["pass_rate_hi"])
            sde_t[e["arm"]] = e["smallest_detectable_effect"]
        for e in T[name]["pairs"]:
            ends[(e["a"], e["b"])] = (e["lo"], e["hi"])

    def vec(q):
        return R[q] if isinstance(q, str) else R[q[1]] - R[q[0]]

    # ------------------------------------------------------------------------------------------
    print("PART 1. Bootstrap interval ends: %d ends (4 arms and 5 comparisons, two ends each)." % (2 * len(ends)))
    print("Each way draws a (10000, 48) table of family positions with seed 0, fresh for every interval;")
    print("then the mean of the drawn families per row, then numpy's 2.5th and 97.5th percentile.")
    print()

    def py_randrange(_q):
        rnd = random.Random(0)
        return np.array([[rnd.randrange(F) for _ in range(F)] for _ in range(B)])

    def py_choices(_q):
        rnd = random.Random(0)
        return np.array([rnd.choices(range(F), k=F) for _ in range(B)])

    def py_random(_q):
        rnd = random.Random(0)
        return np.array([[int(rnd.random() * F) for _ in range(F)] for _ in range(B)])

    ways = [
        ("numpy default_rng(0).integers(0, 48, size=(10000, 48))", lambda _q: np.random.default_rng(0).integers(0, F, size=(B, F))),
        ("numpy default_rng(0).choice(48, size=(10000, 48))", lambda _q: np.random.default_rng(0).choice(F, size=(B, F))),
        ("numpy RandomState(0).randint(0, 48, size=(10000, 48))", lambda _q: np.random.RandomState(0).randint(0, F, size=(B, F))),
        ("numpy RandomState(0).choice(48, size=(10000, 48))", lambda _q: np.random.RandomState(0).choice(F, size=(B, F))),
        ("python random.Random(0).randrange(48), one by one", py_randrange),
        ("python random.Random(0).choices(range(48), k=48), row by row", py_choices),
        ("python int(random.Random(0).random() * 48), one by one", py_random),
    ]
    print("%-62s %-22s %s" % ("way to draw", "ends equal to the table", "largest gap (fraction scale)"))
    for label, draw in ways:
        equal, worst = 0, 0.0
        for q, (lo, hi) in ends.items():
            idx = draw(q)
            m = vec(q)[idx].mean(axis=1)
            mylo, myhi = np.percentile(m, [2.5, 97.5])
            for mine, tab in ((mylo, lo), (myhi, hi)):
                gap = abs(float(mine) - tab)
                worst = max(worst, gap)
                equal += gap <= TOL
        print("%-62s %2d of %-16d %.3e" % (label, equal, 2 * len(ends), worst))
    print()

    # ------------------------------------------------------------------------------------------
    print("PART 2. Smallest detectable effect: se_halves, mde_lowest_split, mde_highest_split of the 4 arms.")
    print("Each way makes 200 splits with numpy default_rng(0). Per split: family pass rate of half A minus")
    print("half B; the standard deviation of that over the 48 families (n - 1) / sqrt(48) is the split's value.")
    print("Three ways to pool the 200 values into se_halves: mean, root mean square, median.")
    print()

    def split_values(y, how):
        rng = np.random.default_rng(0)
        n_f, n_s, n_a = y.shape
        out = []
        for _ in range(200):
            a = np.zeros((n_f, n_s))
            b = np.zeros((n_f, n_s))
            if how == "one rng.permuted call per split on a (192, 8) table":
                pos = rng.permuted(np.tile(np.arange(n_a), (n_f * n_s, 1)), axis=1).reshape(n_f, n_s, n_a)
            for f, st in ((f, st) for f in range(n_f) for st in range(n_s)) if how != "styles outside, families inside; rng.permutation(8)" \
                    else ((f, st) for st in range(n_s) for f in range(n_f)):
                if how in ("rng.permutation(8) per scenario; families outside, styles inside",
                           "styles outside, families inside; rng.permutation(8)"):
                    order = rng.permutation(n_a)
                elif how == "rng.shuffle of the positions 0..7 per scenario":
                    order = np.arange(n_a)
                    rng.shuffle(order)
                elif how == "rng.permutation of the 8 pass marks themselves":
                    v = rng.permutation(y[f, st])
                    a[f, st], b[f, st] = v[:4].mean(), v[4:].mean()
                    continue
                elif how == "rng.choice(8, 4, replace=False) for half A":
                    first = rng.choice(n_a, size=4, replace=False)
                    order = np.concatenate([first, np.setdiff1d(np.arange(n_a), first)])
                elif how == "argsort of rng.random(8) per scenario":
                    order = np.argsort(rng.random(n_a))
                else:
                    order = pos[f, st]
                a[f, st], b[f, st] = y[f, st, order[:4]].mean(), y[f, st, order[4:]].mean()
            d = a.mean(axis=1) - b.mean(axis=1)
            out.append(d.std(ddof=1) / math.sqrt(n_f))
        return np.array(out)

    hows = [
        "rng.permutation(8) per scenario; families outside, styles inside",
        "rng.shuffle of the positions 0..7 per scenario",
        "rng.permutation of the 8 pass marks themselves",
        "rng.choice(8, 4, replace=False) for half A",
        "argsort of rng.random(8) per scenario",
        "one rng.permuted call per split on a (192, 8) table",
        "styles outside, families inside; rng.permutation(8)",
    ]
    print("%-66s %-10s %-10s %-10s %-10s %-10s" % ("way to draw the halves (arms equal to the table, of 4)", "mean", "rms", "median", "lowest", "highest"))
    for how in hows:
        hit = {"mean": 0, "rms": 0, "median": 0, "lowest": 0, "highest": 0}
        for arm in rd.ARMS:
            se = split_values(Y[arm], how)
            t = sde_t[arm]
            hit["mean"] += abs(se.mean() - t["se_halves"]) <= TOL
            hit["rms"] += abs(math.sqrt((se ** 2).mean()) - t["se_halves"]) <= TOL
            hit["median"] += abs(float(np.median(se)) - t["se_halves"]) <= TOL
            hit["lowest"] += abs(2.8 * se.min() / math.sqrt(2) - t["mde_lowest_split"]) <= TOL
            hit["highest"] += abs(2.8 * se.max() / math.sqrt(2) - t["mde_highest_split"]) <= TOL
        print("%-66s %-10d %-10d %-10d %-10d %-10d" % (how, hit["mean"], hit["rms"], hit["median"], hit["lowest"], hit["highest"]))
    print()
    print("Note: ways 1, 2, 3 and 6 are one way written four times if they give the same row above: a random")
    print("order of the 8 attempts of each scenario, scenarios taken family by family and style by style.")
    print("The lowest and highest split can be equal to the table by chance in the GRPO arms even with other")
    print("random numbers, because few values are possible there (few scenarios have both passes and fails).")
    print("In part 1 a single end can also be equal by chance: the ends sit on a grid of 1/1536.")


if __name__ == "__main__":
    main()
