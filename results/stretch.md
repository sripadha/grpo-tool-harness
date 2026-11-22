# Stretch: a bonus when rewordings give the same correct action

**The experiment was not run.** This file analyses the saved runs and gives a design. `scripts/stretch_numbers.py` makes every number from saved files; `results/main/report/stretch/stretch_numbers.md` gives the rule and source of each (tables T1 to T13).

## 1. What the stretch asks

The brief: add a bonus when reworded versions of one scenario produce the same correct action, and test whether it helps beyond a reward for correctness alone.

A *family* is one world with one correct end state. A *scenario* is one family in one wording style. An *attempt* is one conversation. The verifier accepts one end state per family, so "the same correct action" means: the attempt passes, and so do the attempts on the family's other wordings. The measure is *all wordings pass*: per family the product of the pass rates of its wording styles, then the mean over families.

## 2. What the bonus can do in groups of one wording

A GRPO *group* here is the 16 attempts of one scenario in one step. The *advantage* of an attempt is its reward minus the mean reward of its group. A saved pass scores 77 to 100 points: 100 minus small penalties for malformed calls and extra turns (T2). A fail scores 0. A bonus for agreement with the other wordings is one number b for the whole group, because every attempt of the group has the same other wordings. Add b to every pass of a group with k passes of n attempts. The mean of the group rises by b k/n, so every pass gains b (1 - k/n) and every fail loses b k/n.

If every pass scores 100, this multiplies each advantage by (100 + b)/100. Example (T1): 12 passes, 4 fails, b = 20. A pass goes from +25 to +30 and a fail from -75 to -90: 1.2 times.

With 20 points on every pass, the saved training (T2) keeps the same groups (91 of 640 in grpo_seed1 and 95 of 640 in grpo_seed2), and no kept group changes the order of its attempts. In 36 and 45 of the kept groups every attempt passes, and the bonus changes nothing.

So in a group of one wording the bonus cannot make the training prefer one passing attempt over another: it adds b to each pass-fail gap and leaves the gaps between passes as they are. Where every pass scores 100 (47 and 45 kept groups with passes and fails), that only changes the weight of the group's step, as b differs between groups; in 8 and 5 such groups a pass scores less (T2). For the bonus to differ between passes, a group must mix the wordings of one family.

## 3. How much there is to gain

Dev: 48 families x 4 wording styles x 8 attempts = 1536 attempts per arm (T3 to T7).

| Arm | Pass rate (passed attempts) | All wordings pass | All wordings pass, styles 1, 2 and 3 only | Families whose wordings disagree | Variance due to wording |
|---|---|---|---|---|---|
| base | 39.8% (611 of 1536) | 13.4% | 19.3% | 14 | 14.7% |
| sft | 89.4% (1373 of 1536) | 80.3% | 84.1% | 1 | 15.8% |
| grpo_seed1 | 98.6% (1515 of 1536) | 96.4% | 97.2% | 1 | 54.9% |
| grpo_seed2 | 97.7% (1501 of 1536) | 96.4% | 97.1% | 0 | 18.2% |

Wordings disagree (our rule): one style passes 5 or more of its 8 attempts and another style 3 or fewer. A *held-out* style is a wording that no training scenario uses: here style 4.

**Room after GRPO** (100% minus all wordings pass): 3.6 points in each GRPO arm, in 5 of 48 families. One family, dv-46, holds 2.1 of these points. In every GRPO arm dv-46 passes 0 of 8 in its held-out style 4, so its product stays 0: a bonus on the training styles reaches these points only through a gain on a held-out wording.

**Variance due to wording** (`variance_shares`, fixed before the runs; T7): the share of the pass/fail variation inside families that lies between the 4 styles, not within a style. Chance alone gives about 9.7%. It is a share of what is left: after GRPO, 21 and 35 of 1536 attempts fail. Without dv-46 it is 6.7% and 12.1%.

After training only **dv-46** meets the rule (in sft and grpo_seed1; the 14 families of base are in T5). It is also the only family with fewer passes than under sft in every GRPO arm (T6):

| dv-46: passes of 8 | Style 1 | Style 2 | Style 3 | Style 4 (held out) |
|---|---|---|---|---|
| base | 2 | 5 | 2 | 0 |
| sft | 8 | 8 | 7 | 1 |
| grpo_seed1 | 8 | 8 | 1 | 0 |
| grpo_seed2 | 3 | 0 | 0 | 0 |

grpo_seed1 fails in styles 3 and 4. grpo_seed2 fails in all 4 styles (3, 0, 0 and 0 of 8): by the rule of this note its wordings do not disagree. Every failed attempt on dv-46 in these arms closes the request as denied (15 of 15 and 29 of 29); the correct close is granted. Agreement must be read beside the pass rate.

**Test set** (T13; 48 families): all wordings pass is 15.8% (base), 79.5% (sft), 98.3% (grpo_seed1) and 98.0% (grpo_seed2); room after GRPO: 1.7 points in 4 families (grpo_seed1) and 2.0 points in 6 families (grpo_seed2).

## 4. The design that would test it

Two new arms start from the saved SFT weights and differ only in the bonus (T8). The saved runs cannot be arm 1: their groups differ. The saved code builds each group from one scenario: the design needs new training code.

| Part | Value | Origin |
|---|---|---|
| Group | one family: its 3 training wordings x 4 attempts = 12; advantage = reward minus the group mean | 4: judgement |
| Step | 10 families = 120 attempts (saved runs: 8 x 16 = 128); 80 steps as saved | 10: judgement |
| Arm 1 | the saved reward (pass or fail, with the penalties of part 2) | the saved runs |
| Arm 2 | arm 1; a pass also gets 20 x (q - q_mean) points | 20: judgement |
| Runs | 3 seeds per arm | judgement |
| Other settings | as in `grpo_config` of `results/main/grpo_seed1/run_manifest.json` (learning rate 1e-05, tie_tol 0, start `results/main/sft/adapter`) | the saved runs |

q is the share of passes among the family's valid attempts in the other wordings (a pass with no such attempt gets no bonus); q_mean is its mean over the family's passes. The bonuses of a group add up to zero: fails keep their advantage, passes are ranked again. Example: wordings A, B and C pass 4, 2 and 0 of 4. Arm 1: a pass +50, a fail -50. Arm 2: a pass in A +48.3 and a pass in B +53.3: the weaker wording counts more.

Measures on dev: all wordings pass (over all styles, and over the training styles only) and the variance due to wording; the pass rate must not fall. Rule (judgement): the bonus helps if every run of arm 2 is above every run of arm 1 on all wordings pass. By chance alone that happens in 1 of 20 cases (own calculation).

## 5. What size of effect it could see

This is no power calculation: two saved seeds do not show the spread between runs (T10, T11).

- **Interval.** The plan's paired bootstrap (48 families drawn again 10,000 times, 95%) gives the saved comparisons with sft intervals 13.6 and 14.6 points wide on the pass rate and 17.3 and 16.9 on all wordings pass.
- **Families.** A gain in k families is absent from (1 - k/48)^48 of the draws: 36.4% for k = 1, 4.5% for k = 3 and 1.5% for k = 4. So a 95% interval is above zero only if at least 4 families gain. If arm 1 ends where the saved runs did, 4 of the 5 families with room would have to gain and no other family lose. Yet two runs of one recipe already differ in 4 of 48 families on all wordings pass. So a comparison on dev alone is unlikely to give an interval above zero (our judgement).

## 6. What the experiment would cost

The experiment was not run. The frozen `ANALYSIS_PLAN.md` does not list it among its arms or its planned extras (section 3).

In run time (`seconds` of the run manifests), not rented time (T12), one design run counts as a saved GRPO run plus its dev run: grpo_seed1 8903.7 + 1141.7 = 10045.4 s and grpo_seed2 9427.7 + 1208.3 = 10636.0 s; mean 10340.7 s = 2.87 hours. 2 arms x 3 seeds = 6 runs: 6 x 2.87 = about 17.2 GPU hours. At 0.47 to 0.5495 USD per GPU hour (rentals up to the end of the last saved GRPO run): 8.1 to 9.5 USD.
