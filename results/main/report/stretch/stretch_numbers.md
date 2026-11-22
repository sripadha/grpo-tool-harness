# Numbers for the stretch item

**The stretch experiment was not run.** These tables hold what the saved runs show, and the arithmetic of a design. `scripts/stretch_numbers.py` made this file; `results/stretch.md` quotes it. No model was run to make it.

Words. A family is one world with one correct end state. A scenario is one family in one wording style. An attempt is one conversation. A group is the attempts of one scenario in one training step. The advantage of an attempt is its reward minus the mean reward of its group. A point is a hundredth of a pass: a pass with no penalty has a reward of 100 points, a fail has 0.

## T1. A bonus inside a group of one wording (a made-up example)

Rule: `group_advantages` (the function of the training, `accessdesk/train/grpo_math.py`) on one group of 16 attempts with 12 passes and 4 fails, as it is and with 20 points added to every pass. Unit: points.

|  | Reward of a pass | Reward of a fail | Mean of the group | Advantage of a pass | Advantage of a fail |
|---|---|---|---|---|---|
| Without the bonus | 100 | 0 | 75 | +25 | -75 |
| With 20 points on every pass | 120 | 0 | 90 | +30 | -90 |

Each advantage with the bonus is 1.2 times the advantage without it (the formula gives (100 + 20) / 100 = 1.2).

Why every pass of the group gets the same bonus: the bonus looks at the attempts on the family's other wordings, and these are the same for every attempt of this group. A bonus that paired each attempt with one single attempt of another wording would differ between the passes, but only by chance: the pairing is arbitrary, so its expected value is the same for every pass of the group. It would add noise to the step, not a preference between attempts.

## T2. The same bonus on the saved training groups

Rule: for every step of a saved GRPO run, the saved attempts give the reward matrix of the training (`reward_matrix`), and `group_advantages` is run on it twice: as it is, and with 20 points added to the reward of every passing attempt. Nothing is trained. Unit: the group. Source: `rollouts/step_*.episodes.jsonl`, `train_log.jsonl` and `run_manifest.json` of each run (`results/main/grpo_seed1`, `results/main/grpo_seed2`).

| Count of groups | grpo_seed1 | grpo_seed2 |
|---|---|---|
| All groups (80 steps x 8 scenarios; 16 attempts each) | 640 | 640 |
| Kept by the rule of the training (the rewards are not all equal) | 91 | 95 |
| Kept with the bonus | 91 | 95 |
| Groups that the bonus moves between kept and dropped | 0 | 0 |
| Kept groups in which the bonus changes the order of the attempts | 0 | 0 |
| Kept, with passes and fails, every pass at 100 points: the bonus multiplies every advantage by 1.2 | 47 | 45 |
| Kept, every attempt passes (the rewards differ by penalties): the bonus changes no advantage | 36 | 45 |
| Kept, with passes and fails, a pass below 100 points: the bonus moves all passes by one amount and all fails by another | 8 | 5 |
| Groups with a pass and a fail | 55 | 50 |
| Groups with another wording of their family in the same step | 14 | 12 |
| Attempts that the harness lost | 0 | 0 |
| Passing attempts (not groups) | 10140 | 10119 |
| Passing attempts below 100 points (penalties) | 44 | 52 |
| Failing attempts with a reward other than 0 | 0 | 0 |
| Lowest reward of a pass, points | 81 | 77 |

The number of kept groups is also the one of the training log (the script stops if they differ). The training drops a group only if its rewards are all equal (tie_tol 0). A bonus on every pass keeps equal rewards equal and unequal rewards unequal when every fail is below every pass, so the same groups are kept. With every pass at the full reward the advantages are multiplied; in general every pass gains b (1 - k/n) and every fail loses b k/n (k passes of n attempts). The saved training code builds every group from one scenario: its reward matrix refuses a group that holds two scenarios (`reward_matrix` in `accessdesk/train/grpo.py`).

## T3. Per arm, on dev

48 families x 4 wording styles x 8 attempts = 1536 attempts per arm. Source: `results/main/base_dev/episodes.jsonl`, `results/main/sft_dev/episodes.jsonl`, `results/main/grpo_seed1_dev/episodes.jsonl`, `results/main/grpo_seed2_dev/episodes.jsonl`.

| Arm | Pass rate (passed attempts) | All wordings pass | All wordings pass, styles 1, 2 and 3 only | Room, points | Families with a failed attempt | Families whose wordings disagree | Variance due to wording |
|---|---|---|---|---|---|---|---|
| base | 39.8% (611 of 1536) | 13.4% | 19.3% | 86.6 | 47 of 48 | 14 | 14.7% |
| sft | 89.4% (1373 of 1536) | 80.3% | 84.1% | 19.7 | 23 of 48 | 1 | 15.8% |
| grpo_seed1 | 98.6% (1515 of 1536) | 96.4% | 97.2% | 3.6 | 5 of 48 | 1 | 54.9% |
| grpo_seed2 | 97.7% (1501 of 1536) | 96.4% | 97.1% | 3.6 | 5 of 48 | 0 | 18.2% |

- Pass rate: the mean over families of the family's share of passed attempts (`pass_rate`). In brackets: the passed attempts of all attempts that the harness did not lose.
- All wordings pass: per family the product of the pass rates of its wording styles, then the mean over families (`all_wordings_pass`). It is the chance that one attempt per wording passes in every wording.
- All wordings pass, training styles only: the same frozen function on the attempts of the styles that training uses (a held-out style is one that no training scenario uses).
- Room: 100% minus all wordings pass, in points.
- Families whose wordings disagree (our rule): one style passes 5 or more of its 8 attempts and another style 3 or fewer.
- Variance due to wording (`variance_shares`): of the variation of pass or fail inside families, the share that is between wordings. The variance table below says exactly what it is.

Intervals (95%, families drawn again; `family_bootstrap_ci`): base: pass rate 31.4% to 48.4%, all wordings pass 6.8% to 20.9%; sft: pass rate 82.1% to 95.4%, all wordings pass 70.8% to 88.7%; grpo_seed1: pass rate 96.5% to 99.9%, all wordings pass 91.5% to 99.5%; grpo_seed2: pass rate 93.8% to 99.9%, all wordings pass 91.4% to 99.5%.

## T4. Where the room sits after GRPO, on dev

Rule: a family's part of the room is (1 minus its product of style pass rates) divided by the 48 families. One family is at most 2.1 points. Unit: points of all wordings pass. The document calls the room "little" when it is below 5.0 points in every GRPO arm (a judgement).

grpo_seed1: 3.6 points of room, in 5 of 48 families.

| Family | Expected outcome | Style 1 | Style 2 | Style 3 | Style 4 (in no training scenario) | Part of the room, points |
|---|---|---|---|---|---|---|
| dv-46 | grant | 8 of 8 | 8 of 8 | 1 of 8 | 0 of 8 | 2.08 |
| dv-02 | approval | 7 of 8 | 8 of 8 | 7 of 8 | 7 of 8 | 0.69 |
| dv-08 | approval | 7 of 8 | 8 of 8 | 8 of 8 | 8 of 8 | 0.26 |
| dv-31 | already | 8 of 8 | 8 of 8 | 8 of 8 | 7 of 8 | 0.26 |
| dv-37 | approval | 8 of 8 | 8 of 8 | 7 of 8 | 8 of 8 | 0.26 |

grpo_seed2: 3.6 points of room, in 5 of 48 families.

| Family | Expected outcome | Style 1 | Style 2 | Style 3 | Style 4 (in no training scenario) | Part of the room, points |
|---|---|---|---|---|---|---|
| dv-46 | grant | 3 of 8 | 0 of 8 | 0 of 8 | 0 of 8 | 2.08 |
| dv-31 | already | 8 of 8 | 8 of 8 | 8 of 8 | 6 of 8 | 0.52 |
| dv-02 | approval | 8 of 8 | 8 of 8 | 7 of 8 | 7 of 8 | 0.49 |
| dv-09 | inactive | 8 of 8 | 7 of 8 | 8 of 8 | 8 of 8 | 0.26 |
| dv-37 | approval | 8 of 8 | 8 of 8 | 7 of 8 | 8 of 8 | 0.26 |

## T5. Families whose wordings disagree, on dev

Rule (ours): one style passes 5 or more of its 8 attempts and another style 3 or fewer. Each cell: passes of attempts, per wording style in rising order. A row is in the table if the family meets the rule in at least one arm; the last column names those arms.

| Family | Expected outcome | base | sft | grpo_seed1 | grpo_seed2 | Meets the rule in |
|---|---|---|---|---|---|---|
| dv-06 | grant | 7 / 6 / 3 / 1 | 8 / 7 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-15 | grant | 6 / 7 / 5 / 1 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-17 | grant | 5 / 6 / 3 / 4 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-19 | grant | 6 / 5 / 5 / 3 | 8 / 8 / 8 / 7 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-21 | grant | 4 / 5 / 4 / 1 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-24 | approval | 8 / 3 / 3 / 8 | 8 / 8 / 7 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-26 | approval | 7 / 7 / 6 / 3 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-27 | grant | 3 / 6 / 5 / 4 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-31 | already | 5 / 1 / 5 / 4 | 8 / 8 / 8 / 7 | 8 / 8 / 8 / 7 | 8 / 8 / 8 / 6 | base |
| dv-43 | already | 2 / 5 / 4 / 2 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-44 | inactive | 7 / 2 / 5 / 4 | 7 / 7 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-45 | grant | 3 / 5 / 4 / 4 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| dv-46 | grant | 2 / 5 / 2 / 0 | 8 / 8 / 7 / 1 | 8 / 8 / 1 / 0 | 3 / 0 / 0 / 0 | base, sft, grpo_seed1 |
| dv-47 | already | 1 / 1 / 2 / 5 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |

Count per arm: base 14; sft 1; grpo_seed1 1; grpo_seed2 0 (of 48 families).

## T6. Families with fewer passes than under sft, on dev

Rule: the family's share of passed attempts, over all its wording styles, is lower in the GRPO arm than in sft. Source: the same files.

- grpo_seed1: dv-46.
- grpo_seed2: dv-31 and dv-46.
- In every GRPO arm: dv-46.

| dv-46 | Style 1 | Style 2 | Style 3 | Style 4 (in no training scenario) | All styles |
|---|---|---|---|---|---|
| base | 2 of 8 | 5 of 8 | 2 of 8 | 0 of 8 | 9 of 32 |
| sft | 8 of 8 | 8 of 8 | 7 of 8 | 1 of 8 | 24 of 32 |
| grpo_seed1 | 8 of 8 | 8 of 8 | 1 of 8 | 0 of 8 | 17 of 32 |
| grpo_seed2 | 3 of 8 | 0 of 8 | 0 of 8 | 0 of 8 | 3 of 32 |

- grpo_seed1, failed attempts of dv-46: 15 of 15 close the request as denied. 8 of 15 name the person of the request in the close.
- grpo_seed2, failed attempts of dv-46: 29 of 29 close the request as denied. 22 of 29 name the person of the request in the close.
- Scenario file: the expected close of dv-46 is granted. 2 people of its world have the name of the request; the status of the other one is inactive.

## T7. Variance inside families, on dev

Rule (`variance_shares`, frozen): score a pass 1 and a fail 0. Inside each family, the squared distances of its attempts from the family's pass rate split into two sums: between the pass rates of its wording styles (each style counted with its number of attempts), and between the attempts of one style. Both sums are added up over all families. "Wording" is the first sum as a share of both, "sampling" is the second. Differences between families are in neither. "If sampling only" is the share to expect when wording has no effect: the style rates still differ by chance. The share is of the variation that is left, so it says nothing about how much is left: the column of failed attempts does.

| Arm | Failed attempts | Wording | Sampling | If sampling only | Wording, without dv-46 | Failed attempts, without dv-46 |
|---|---|---|---|---|---|---|
| base | 925 of 1536 | 14.7% | 85.3% | 9.7% | 14.4% | 902 of 1504 |
| sft | 163 of 1536 | 15.8% | 84.2% | 9.7% | 9.5% | 155 of 1504 |
| grpo_seed1 | 21 of 1536 | 54.9% | 45.1% | 9.7% | 6.7% | 6 of 1504 |
| grpo_seed2 | 35 of 1536 | 18.2% | 81.8% | 9.7% | 12.1% | 6 of 1504 |

## T8. The design (not run)

Every value is a choice. The column Origin says where it comes from: the brief, the saved runs, our own calculation from other values, or a judgement (no run and no source chose it).

| Part | Value | Origin |
|---|---|---|
| arms | 2 | the brief: with the bonus and without it |
| wordings in a family group | 3 | the saved runs: the wording styles that training uses |
| attempts per wording | 4 | judgement |
| attempts per family group | 12 | own calculation |
| family groups per step | 10 | judgement |
| attempts per step | 120 | own calculation |
| steps | 80 | the saved runs |
| learning rate | 0.00001 | the saved runs |
| largest bonus of a pass, in points | 20 | judgement |
| training runs (seeds) per arm | 3 | judgement |
| start of both arms | the adapter that the saved GRPO runs started from (results/main/sft/adapter) | the saved runs |
| group | the attempts of one family in all its training wordings | judgement |
| advantage, both arms | the reward minus the mean reward of the family group; a group with equal rewards is dropped, as in the saved runs | judgement |
| reward of arm 1 | the reward of the saved runs: pass or fail, with small penalties for malformed calls and extra turns | the saved runs |
| other settings | as in grpo_config of the run manifest of the saved GRPO runs | the saved runs |
| bonus when another wording has no valid attempt | a pass with no valid attempt in another wording of its group gets no bonus | judgement |
| reward of arm 2 | the reward of arm 1, and the bonus below for a passing attempt | judgement |
| measures, on dev | all wordings pass, and the variance due to wording; the pass rate must not fall | judgement |
| rule for the result | the bonus helps if every run of arm 2 is above every run of arm 1 on all wordings pass | judgement |

The bonus of arm 2: a passing attempt gets 20 x (q - q_mean) points. q is the share of passes among the family's attempts in the other wordings. q_mean is the mean of q over the family's passing attempts. So the bonuses of a group add up to zero, the mean reward of the group stays, and so does the advantage of every fail.

A made-up family group (12 attempts). Unit: points. Source: `family_group` of the script, which calls `group_advantages`.

| Wording | Passes | q of a pass | Bonus of a pass | Arm 1: advantage of a pass | Arm 2: advantage of a pass | Advantage of a fail, both arms |
|---|---|---|---|---|---|---|
| A | 4 of 4 | 0.25 | -1.67 | +50.00 | +48.33 | no fail |
| B | 2 of 4 | 0.5 | +3.33 | +50.00 | +53.33 | -50.00 |
| C | 0 of 4 | no pass | no pass | no pass | no pass | -50.00 |

q_mean is 0.3333; the bonuses of the group add up to 0.

The rule for the result (a judgement): the bonus helps if every run of arm 2 is above every run of arm 1. With no effect of the bonus, and no ties, this happens by chance in 1 of C(2m, m) cases for m runs per arm (own calculation): m = 2: 1 of 6 (16.7%); m = 3: 1 of 20 (5.0%).

## T9. The bonus of the design on saved attempts

Rule: the saved attempts of the SFT model on training families are cut into family groups of the design (3 wordings x 4 attempts; draw 1 takes the lowest sample numbers of each wording, draw 2 the next ones), and the arithmetic of both arms is run on each group. Nothing is trained: this shows the size of the bonus at the start of both arms. Unit: the family group. Source: `results/main/sft_train/episodes.jsonl` (2400 attempts, 0 lost by the harness).

|  | Draw 1 (attempts 1 to 4 of each wording) | Draw 2 (attempts 5 to 8 of each wording) |
|---|---|---|
| Family groups | 100 | 100 |
| Groups with equal rewards (dropped in both arms) | 68 | 79 |
| Groups that one arm keeps and the other drops | 0 | 0 |
| Groups with a pass and a fail | 23 | 13 |
| Groups in which the bonus is not zero | 20 | 10 |
| Groups in which the bonus changes the order of the attempts | 20 | 10 |
| Passing attempts in the groups with a bonus | 214 | 102 |
| Mean size of the bonus of those passes, points | 1.07 | 1.21 |
| Largest bonus, points | 5.62 | 5.62 |
| Mean size of the arm-1 advantage in those groups, points | 18.45 | 23.6 |

## T10. Intervals of the saved comparisons (dev)

Rule: the difference "b minus a" of two arms, family by family, with the paired bootstrap over families of the plan (`paired_family_bootstrap`, main version; 95%, 10000 draws, seed 0; percentile form). Pass rate: the row of the frozen table program. All wordings pass: the same function on the family values of that metric (ours; not in the frozen tables). Unit: points; the family is the unit that is drawn again. A comparison of two GRPO arms is not a comparison of the plan: it shows two runs of one recipe. Source: `results/main/base_dev/episodes.jsonl`, `results/main/sft_dev/episodes.jsonl`, `results/main/grpo_seed1_dev/episodes.jsonl`, `results/main/grpo_seed2_dev/episodes.jsonl`.

| Comparison | Families | Pass rate: difference (interval) | Width | All wordings pass: difference (interval) | Width | Families that differ on all wordings pass | In the plan |
|---|---|---|---|---|---|---|---|
| sft minus base | 48 | +49.6 (+41.1 to +57.9) | 16.9 | +66.8 (+56.4 to +76.8) | 20.5 | 45 | yes |
| grpo_seed1 minus sft | 48 | +9.2 (+3.1 to +16.7) | 13.6 | +16.2 (+8.1 to +25.4) | 17.3 | 20 | yes |
| grpo_seed2 minus sft | 48 | +8.3 (+1.5 to +16.1) | 14.6 | +16.1 (+8.2 to +25.1) | 16.9 | 22 | yes |
| grpo_seed1 minus base | 48 | +58.9 (+49.9 to +67.6) | 17.8 | +83.0 (+74.2 to +91.0) | 16.8 | 46 | yes |
| grpo_seed2 minus base | 48 | +57.9 (+48.5 to +67.1) | 18.6 | +82.9 (+74.4 to +90.7) | 16.3 | 46 | yes |
| grpo_seed2 minus grpo_seed1 | 48 | -0.9 (-2.9 to +0.2) | 3.1 | -0.1 (-1.0 to +0.9) | 1.9 | 4 | no |

Smallest detectable effect of the frozen tables (`two_halves_mde`: a difference of two pass rates, sampling noise only, from one arm's own attempts): base 3.9 points; sft 1.9 points; grpo_seed1 0.7 points; grpo_seed2 0.7 points.

## T11. How many families must gain before the interval can be above zero

Rule: one draw of the bootstrap takes 48 families with replacement. If a gain sits in k families and every other family is equal in both arms, a draw without those k families has a difference of exactly zero. The share of such draws is (1 - k / 48) ^ 48. The lower end of a 95% interval is the 2.5% quantile of the draws, so it is zero while that share is above 2.5%. The last column runs the frozen bootstrap on made-up family values (a gain in the first k families, zero elsewhere). This is a limit of the method, not a power calculation.

| Families that gain (k) | Draws without them | Lower end above zero, by the formula | Lower end above zero, by the frozen bootstrap |
|---|---|---|---|
| 1 | 36.4% | no | no |
| 2 | 13.0% | no | no |
| 3 | 4.5% | no | no |
| 4 | 1.5% | yes | yes |
| 5 | 0.5% | yes | yes |
| 6 | 0.2% | yes | yes |
| 7 | 0.1% | yes | yes |
| 8 | 0.0% | yes | yes |

The 2 GRPO arms are runs of one recipe with other seeds. Largest minus smallest value on dev: pass rate 0.9 points; all wordings pass 0.1 points; share of the variance due to wording 36.7 points.

## T12. Cost

Rule: one GPU ran each saved run, so its GPU hours are the wall-clock seconds of its run manifest (`seconds`) divided by 3600: run time, not rented time. One run of the design is priced as a saved GRPO run plus the dev run of its last checkpoint (mean over the saved GRPO arms): an estimate, because a step of the design has 120 attempts and a saved step has 128. Source: `run_manifest.json` of each run; `results/spend_ledger.csv` for the prices.

| Saved run | GPU | GPU hours of the training | GPU hours of the dev run |
|---|---|---|---|
| grpo_seed1 | NVIDIA GeForce RTX 4090 | 2.47 | 0.32 |
| grpo_seed2 | NVIDIA GeForce RTX 4090 | 2.62 | 0.34 |

| Design | Training runs | GPU hours (estimate) | USD (estimate) |
|---|---|---|---|
| 2 arms x 2 seeds | 4 | 11.5 | 5.4 to 6.3 |
| 2 arms x 3 seeds | 6 | 17.2 | 8.1 to 9.5 |

Prices: the lowest and the highest hourly price of the machines rented up to the end of the last saved GRPO training run (2026-11-19T06:08:48Z, `ended_utc` of its run manifest): 0.47 and 0.5495 USD per hour (rentals 61045163, 61047327, 61098966, 61182932, 61285391, 61312835, 61313315, 61368330). A machine rented later does not change them. One design run: 2.87 GPU hours. The ledger bills a rental from rent to stop, with start-up and downloads; these amounts leave that time out.

## T13. The test set

The same tables as T3 to T7, on the test set. The test set was not used for any choice.

### Per arm, on test

48 families x 4 wording styles x 8 attempts = 1536 attempts per arm. Source: `results/main/base_test/episodes.jsonl`, `results/main/sft_test/episodes.jsonl`, `results/main/grpo_seed1_test/episodes.jsonl`, `results/main/grpo_seed2_test/episodes.jsonl`.

| Arm | Pass rate (passed attempts) | All wordings pass | All wordings pass, styles 1, 2 and 3 only | Room, points | Families with a failed attempt | Families whose wordings disagree | Variance due to wording |
|---|---|---|---|---|---|---|---|
| base | 41.3% (634 of 1536) | 15.8% | 20.8% | 84.2 | 46 of 48 | 13 | 9.7% |
| sft | 91.3% (1403 of 1536) | 79.5% | 81.6% | 20.5 | 29 of 48 | 2 | 14.5% |
| grpo_seed1 | 99.5% (1529 of 1536) | 98.3% | 98.7% | 1.7 | 4 of 48 | 0 | 12.2% |
| grpo_seed2 | 99.5% (1528 of 1536) | 98.0% | 98.3% | 2.0 | 6 of 48 | 0 | 7.4% |

- Pass rate: the mean over families of the family's share of passed attempts (`pass_rate`). In brackets: the passed attempts of all attempts that the harness did not lose.
- All wordings pass: per family the product of the pass rates of its wording styles, then the mean over families (`all_wordings_pass`). It is the chance that one attempt per wording passes in every wording.
- All wordings pass, training styles only: the same frozen function on the attempts of the styles that training uses (a held-out style is one that no training scenario uses).
- Room: 100% minus all wordings pass, in points.
- Families whose wordings disagree (our rule): one style passes 5 or more of its 8 attempts and another style 3 or fewer.
- Variance due to wording (`variance_shares`): of the variation of pass or fail inside families, the share that is between wordings. The variance table below says exactly what it is.

Intervals (95%, families drawn again; `family_bootstrap_ci`): base: pass rate 32.2% to 50.1%, all wordings pass 8.4% to 24.0%; sft: pass rate 85.9% to 95.8%, all wordings pass 70.5% to 87.4%; grpo_seed1: pass rate 99.0% to 99.9%, all wordings pass 96.1% to 99.7%; grpo_seed2: pass rate 99.0% to 99.9%, all wordings pass 96.1% to 99.5%.

### Where the room sits after GRPO, on test

Rule: a family's part of the room is (1 minus its product of style pass rates) divided by the 48 families. One family is at most 2.1 points. Unit: points of all wordings pass. The document calls the room "little" when it is below 5.0 points in every GRPO arm (a judgement).

grpo_seed1: 1.7 points of room, in 4 of 48 families.

| Family | Expected outcome | Style 1 | Style 2 | Style 3 | Style 4 (in no training scenario) | Part of the room, points |
|---|---|---|---|---|---|---|
| te-08 | grant | 8 of 8 | 6 of 8 | 8 of 8 | 6 of 8 | 0.91 |
| te-07 | already | 8 of 8 | 8 of 8 | 7 of 8 | 8 of 8 | 0.26 |
| te-12 | grant | 8 of 8 | 7 of 8 | 8 of 8 | 8 of 8 | 0.26 |
| te-31 | grant | 8 of 8 | 7 of 8 | 8 of 8 | 8 of 8 | 0.26 |

grpo_seed2: 2.0 points of room, in 6 of 48 families.

| Family | Expected outcome | Style 1 | Style 2 | Style 3 | Style 4 (in no training scenario) | Part of the room, points |
|---|---|---|---|---|---|---|
| te-23 | already | 7 of 8 | 7 of 8 | 7 of 8 | 8 of 8 | 0.69 |
| te-03 | grant | 8 of 8 | 8 of 8 | 7 of 8 | 8 of 8 | 0.26 |
| te-08 | grant | 8 of 8 | 8 of 8 | 8 of 8 | 7 of 8 | 0.26 |
| te-12 | grant | 8 of 8 | 7 of 8 | 8 of 8 | 8 of 8 | 0.26 |
| te-14 | grant | 8 of 8 | 7 of 8 | 8 of 8 | 8 of 8 | 0.26 |
| te-31 | grant | 8 of 8 | 7 of 8 | 8 of 8 | 8 of 8 | 0.26 |

### Families whose wordings disagree, on test

Rule (ours): one style passes 5 or more of its 8 attempts and another style 3 or fewer. Each cell: passes of attempts, per wording style in rising order. A row is in the table if the family meets the rule in at least one arm; the last column names those arms.

| Family | Expected outcome | base | sft | grpo_seed1 | grpo_seed2 | Meets the rule in |
|---|---|---|---|---|---|---|
| te-02 | already | 5 / 3 / 2 / 5 | 8 / 7 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-06 | grant | 6 / 7 / 3 / 2 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-07 | already | 5 / 4 / 3 / 4 | 8 / 8 / 7 / 8 | 8 / 8 / 7 / 8 | 8 / 8 / 8 / 8 | base |
| te-09 | grant | 5 / 5 / 4 / 2 | 7 / 7 / 7 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-11 | grant | 3 / 7 / 7 / 4 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-12 | grant | 4 / 4 / 5 / 3 | 8 / 7 / 8 / 8 | 8 / 7 / 8 / 8 | 8 / 7 / 8 / 8 | base |
| te-18 | grant | 5 / 4 / 6 / 2 | 8 / 7 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-25 | inactive | 0 / 1 / 0 / 0 | 0 / 7 / 7 / 6 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | sft |
| te-27 | inactive | 5 / 1 / 1 / 0 | 7 / 6 / 7 / 7 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-29 | inactive | 6 / 3 / 4 / 6 | 8 / 8 / 8 / 7 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-30 | approval | 5 / 1 / 3 / 0 | 8 / 8 / 8 / 7 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-31 | grant | 4 / 4 / 5 / 3 | 8 / 7 / 8 / 8 | 8 / 7 / 8 / 8 | 8 / 7 / 8 / 8 | base |
| te-34 | already | 5 / 4 / 5 / 2 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-36 | grant | 3 / 6 / 3 / 4 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | base |
| te-42 | inactive | 1 / 2 / 1 / 0 | 2 / 6 / 8 / 5 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | sft |

Count per arm: base 13; sft 2; grpo_seed1 0; grpo_seed2 0 (of 48 families).

### Families with fewer passes than under sft, on test

Rule: the family's share of passed attempts, over all its wording styles, is lower in the GRPO arm than in sft. Source: the same files.

- grpo_seed1: no family.
- grpo_seed2: te-23.
- In every GRPO arm: no family.

### Variance inside families, on test

Rule (`variance_shares`, frozen): score a pass 1 and a fail 0. Inside each family, the squared distances of its attempts from the family's pass rate split into two sums: between the pass rates of its wording styles (each style counted with its number of attempts), and between the attempts of one style. Both sums are added up over all families. "Wording" is the first sum as a share of both, "sampling" is the second. Differences between families are in neither. "If sampling only" is the share to expect when wording has no effect: the style rates still differ by chance. The share is of the variation that is left, so it says nothing about how much is left: the column of failed attempts does.

| Arm | Failed attempts | Wording | Sampling | If sampling only |
|---|---|---|---|---|
| base | 902 of 1536 | 9.7% | 90.3% | 9.7% |
| sft | 133 of 1536 | 14.5% | 85.5% | 9.7% |
| grpo_seed1 | 7 of 1536 | 12.2% | 87.8% | 9.7% |
| grpo_seed2 | 8 of 1536 | 7.4% | 92.6% | 9.7% |

## Files

| What | File | Start of its SHA-256 |
|---|---|---|
| attempts of base on dev | `results/main/base_dev/episodes.jsonl` | e074b3071398 |
| attempts of sft on dev | `results/main/sft_dev/episodes.jsonl` | 748363819924 |
| attempts of grpo_seed1 on dev | `results/main/grpo_seed1_dev/episodes.jsonl` | 7cd90a353bce |
| attempts of grpo_seed2 on dev | `results/main/grpo_seed2_dev/episodes.jsonl` | a073d31f54ea |
| training log of grpo_seed1 | `results/main/grpo_seed1/train_log.jsonl` | b6a080196a1d |
| training log of grpo_seed2 | `results/main/grpo_seed2/train_log.jsonl` | 0c02ece824fc |
| 80 step files of grpo_seed1 (hash of their `sha256sum` lines) | `results/main/grpo_seed1/rollouts/` | fe6d4dee8728 |
| 80 step files of grpo_seed2 (hash of their `sha256sum` lines) | `results/main/grpo_seed2/rollouts/` | 3486eba580e5 |
| attempts of the SFT model on training families | `results/main/sft_train/episodes.jsonl` | 87b4b6c7e0a9 |
| frozen table (33 values compared, all equal) | `results/main/tables/base_sft_grpo_seed1.json` | ee6d0a400b92 |
| frozen table (33 values compared, all equal) | `results/main/tables/base_sft_grpo_seed2.json` | 3f9963868c98 |
| frozen table (19 values compared, all equal) | `results/main/tables/base_vs_sft.json` | 4e6253cf843c |
| frozen table (41 values compared, all equal) | `results/main/tables/test_base_sft_grpo_seed1_seed2.json` | fc7f2aa98a29 |
| scenario file (dev) | `data/scenarios/dev.jsonl` | fea3e07efa22 |
| code that computes the numbers | `scripts/stretch_numbers.py` | 8201f2d97146 |
| code that computes the numbers | `accessdesk/eval/metrics.py` | 3bce7f5f3b49 |
| code that computes the numbers | `accessdesk/eval/tables.py` | a4aaad48d9ff |
| code that computes the numbers | `accessdesk/eval/bootstrap.py` | f52f146b1e50 |
| code that computes the numbers | `accessdesk/impl.py` | 3e631d987ee1 |
| code that computes the numbers | `accessdesk/train/grpo_math.py` | 8d203e9e3dae |
| code that computes the numbers | `accessdesk/train/grpo.py` | cf18a0e50f2d |
