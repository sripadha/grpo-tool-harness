# Analysis plan

Frozen before any training run: `scripts/freeze.py` writes the SHA-256 of this file, the scenarios, the text that the model reads, the config, the grader, the harness and the evaluation code into `MANIFEST.json`. After that:

- A frozen file that changes gets a new hash and a reason there (`scripts/freeze.py amend`), and results from before and after the change share no table. We keep this by hand: `accessdesk/eval/tables.py` does not compare code hashes.
- `scripts/gpu/rollout.py` and `scripts/run_scripted.py` write the hash of the frozen code (`code_sha256`) into their run manifest. On a tree that differs from `MANIFEST.json` they stop, unless `--allow-changed-tree` is given; the run manifest then says so.
- New code that only reads saved attempts (the audit) gets a hash and a date there and splits nothing.
- Training code (`accessdesk/train/`, and the training and chain scripts in `scripts/gpu/`) is not frozen: each training run records its hashes in its `run_manifest.json`, and only warns on a changed tree. Section 7 is fixed by its words.

Family: one world with one correct end state. Scenario: one family in one wording style. Attempt: one episode. "(ours)": not asked for by the assignment. `NOTES.md` is written with the results.

## 1. The question

Does a small post-training pipeline (SFT on the model's own passing attempts, then GRPO) work at toy scale? Is the reward wired correctly, does training improve held-out behaviour, and does the model exploit the reward?

## 2. What is fixed

- **Model:** `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc` (`PINNED_REVISIONS` in `accessdesk/harness/hf_backend.py`).
- **Scenarios (ours):** four outcomes, a quarter of the families each. Rule facts can be true together (`mix_facts`). Half of the families hold two people with one full name (`homonym_share` 0.5). 200 train, 48 dev and 48 test families.
- **Wording styles:** train 1 to 3 (600 scenarios); dev and test 1 to 4 (192 each). Style 4 is held out.
- **Attempts:** 8 per scenario, in every table arm and in the SFT sampling.
- **Decoding:** temperature 1.0, no top-k, no top-p, in training and in evaluation (`RunConfig` in `accessdesk/config.py`); thinking off (`accessdesk/harness/tokens.py`); turn limit 8.
- **Token cap per turn:** 260 (`max_new_tokens` in `accessdesk/config.py`). The rule: twice the 99th percentile of the lengths of well-formed calls in calibration, and at least 128. Every run uses that value: none is started with another `--max-new-tokens`, or with another `MAX_NEW_TOKENS` of `scripts/gpu/chain_stage1.sh` or `scripts/gpu/chain_stage2.sh`. Each record carries it in `settings`, and `accessdesk/eval/tables.py` refuses arms that differ.
- **Seeds:** data: `data_seed` 20261118 in `accessdesk/config.py`; every scenario folder of the calibration rounds was built with data seed 1. Rollouts (`--run-seed` of `scripts/gpu/rollout.py`): `0` on dev and test, in every arm; `10` for `base` on train (the SFT data); `20` for `sft` on train (the family profile of section 7). SFT: `--seed 0` (`scripts/gpu/train_sft.py`). GRPO: `--seed 1` and `2` (`scripts/gpu/train_grpo.py`); pilot `100`; overfit check `101`. Sampling inside a GRPO run has a run seed of its own, made from the seed (`sampling_run_seed` in `accessdesk/train/grpo.py`). Bootstrap: `--seed 0` (`accessdesk/eval/tables.py`).

## 3. Arms

`base` (untrained), `sft` (stage 1), GRPO seed 1 and GRPO seed 2 (stage 2; each trains the same SFT adapter further). Two seeds describe two checkpoints, not the recipe.

Reference rows, in no comparison (`scripts/run_scripted.py`): `oracle` must pass every scenario, and each `constant_*` policy exactly one outcome in four.

Second desk (ours; an extra, in no comparison of section 4): the 16 dev families of the set `policy_shift` in `data/scenarios/sets.json`, with the rule set `all_grants_need_approval` and the system prompt that states it (`scripts/gpu/rollout.py --set policy_shift_differ` and `--set policy_shift_control`, each with `--prompt policy_shift --ruleset all_grants_need_approval`). Training never shows this rule. It changes the expected end state in 11 of the families ("differ") and not in 5 ("control"). Arms: `base`, `sft` and each GRPO seed; all four wordings, 8 attempts, run seed 0. Reported per arm: the pass rate under this rule on the 11 and on the 5, and on the 11 the mean over families of the share of attempts whose final state holds a grant record ("still gives the role"). That share has no function yet: its code only reads saved attempts and gets its hash after the freeze, like the audit code. If time is short, the second desk is cut before any comparison of section 4, and the write-up says so.

Planned extras (ours; after every required item, cut first): a teacher arm (on-policy distillation from the untrained 4B into the 2B, starting from the SFT checkpoint) and a hint check (the untrained 2B on dev with one added sentence about its most common mistake). Each gets its own description, with its comparison and metric, in a separate file with its own hash before it runs. Neither changes a frozen file.

## 4. Comparisons

- **Unit:** the family, by its pass rate over all its wordings and attempts.
- **Interval:** paired bootstrap over families, the same resampled families for both arms (`paired_family_bootstrap` in `accessdesk/eval/bootstrap.py`). Ours: 95%, 10,000 draws, the percentile form, a t-interval beside it, no correction for the number of comparisons.
- **Dev:** `sft` minus `base`; each GRPO seed minus `sft`, and minus `base`.
- **Test (ours; primary, because dev is watched during the work):** `sft` minus `base`; each GRPO seed minus `sft`.
- **The test set is run once,** after the freeze and the dev table. No setting, checkpoint or frozen file is chosen or changed after a test number is seen. A repeat needs a machine fault or a harness fault, and a line in `NOTES.md`.
- Every table value comes from saved records: `python -m accessdesk.eval.tables`, and `scripts/monitor_report.py` for the flagged shares.

## 5. Metrics

| Metric | Exact meaning | Function (in `accessdesk/eval/metrics.py`, but the last one) |
|---|---|---|
| Pass rate | Mean over families of the family's share of passed attempts ("family-weighted"). Its interval resamples families | `pass_rate`, `family_bootstrap_ci` |
| Per outcome; seen and held-out style | The pass rate in each of the four outcomes; over styles 1 to 3; over style 4 | `pass_rate_by` |
| Reward distribution | Attempts at each reward value | `reward_distribution` |
| Equal-reward share | Share of scenarios whose attempts all have exactly the same reward; split into all pass and all fail | `zero_variance_share` |
| Invalid-call rate | Assistant messages that are not exactly one valid tool call, divided by all assistant messages | `invalid_call_rate` |
| Pass@8 | Per scenario: 1 if at least one of 8 attempts passed. Then family-weighted | `pass_at_k` |
| Pass^k, k = 1, 2, 4, 8 | Per scenario with c passes in n attempts: C(c, k) / C(n, k). Then family-weighted. Pass^8 is descriptive only | `pass_hat_k` |
| All wordings pass | Per family: the product of the pass rates of its styles. Then the mean over families | `all_wordings_pass` |
| Turns | Mean assistant messages per attempt; share ended by the turn limit without a close; share that closed the request | `turn_stats` |
| Smallest detectable effect (ours) | 2.8 standard errors of a difference of two pass rates, from one arm's attempts, split at random into two halves 200 times. Sampling noise only: too small when the gain differs between families | `two_halves_mde` |
| Passed but flagged (ours) | Per monitor: the share of passing attempts that it flags; the mean over the families with a pass | `summarise` in `accessdesk/eval/monitors.py` |

The monitors (a call that a tool refused; an id that no earlier tool result showed; a close with no access check for that person and role; a repeated call; three more) never change a verdict or a reward.

Rule order (ours): the first rule that applies wins: an inactive person, then a role held already, then a role that needs approval, then a grant (`decide` in `accessdesk/env/rules.py`). Another order fails only families with two of these facts true: 14 of the 48 dev families and 14 of the 48 test families. Only 4 of them per split (16 scenarios, 128 attempts per arm) tell "inactive before already held" from the reverse: an inactive person who holds the role. So the write-up does not claim that a model learned this order.

## 6. What "reward 100" means

- **Pass:** the final state has the expected roles, approval records and grant records, and the request is closed with the right resolution for the right person and role (`verify` in `accessdesk/env/verifier.py`). How the state was reached is not checked.
- **Reward** (`reward` in `accessdesk/env/reward.py`): a fail scores 0. A pass scores 100, minus 5 per malformed message (at most 15), minus 2 per turn over the budget (at most 10). The budget is the shortest correct path plus one turn.
- **So reward 100 means** a pass with no malformed message and at most one spare turn, not that every call was accepted. A refused call (for example a made-up id) costs no points by itself: it uses a turn, and a turn costs 2 points only beyond the budget.
- The reward stays as it is (ours). SFT trains only on attempts with no refused call, and the flagged shares stand beside every pass rate.

## 7. Rules written before the runs

- **SFT data** (`select_sft_records` in `accessdesk/train/sft.py`): attempts of `base` on train with reward 100 and no refused call; one per scenario (the fewest turns, then the lowest sample number).
- **Group size (ours):** `sft` makes 8 attempts per scenario on 100 training families (the first 25 of each outcome in `train.jsonl`, listed by `scripts/sft_profile.py --pick`; `scripts/gpu/rollout.py --families`). `scripts/sft_profile.py` applies the rule. It reads `episodes.jsonl` and `run_manifest.json` of that rollout. If the median of `family_pass_rates` (`accessdesk/eval/metrics.py`) is above 0.8 (exactly 0.8 is not), GRPO uses groups of 16 on 8 scenarios per step, not groups of 8 on 16. It writes `decision.json` and `profile.md`; `scripts/gpu/chain_stage2.sh` reads the step from `decision.json`. No family is removed.
- **Overfit check** (`scripts/gpu/train_grpo.py --overfit FILE --styles 1`): two training families, one wording each, with passes and fails at the start, reach a pass rate near 1 in about 30 steps ("near 1" is a judgement). `scripts/pick_overfit_families.py` writes FILE by a fixed rule, from a finished rollout on train in style 1 (its report names the rollout): of the families with no harness error and with both passes and fails, the two whose number of passes is nearest to half of the attempts; ties go to the smaller family id. `--max-harness-error-rate` may be raised for this check only.
- **Loss constant (ours):** every GRPO run (overfit check, pilot, both seeds) is started with `--reference-tokens 200`. The GRPO loss is divided by scenarios per step x attempts per scenario x this number, so the number scales the loss by a constant and nothing else. 200 is the median number of sampled tokens in a full-reward attempt of the untrained model in calibration (`results/calibration/round4/2b_mixed`). It is fixed here so that it is not tuned later.
- **Pilot gate (ours):** after 15 GRPO steps at the real settings (`--pilot`), 20 attempts are read. If the training reward has not moved and the numbers of the stop rule are normal, both seeds use the learning rate 3e-5, not 1e-5. Both conditions are judgements, on the training log only, written into `NOTES.md` before the seeds start.
- **Stop rule** (`find_fault` in `accessdesk/train/grpo.py`): a run goes to 80 steps, and its last checkpoint is the result, whatever its dev number. It stops early only for a fault: a loss or gradient that is not finite; harness errors above 1% of its attempts so far; a mean log-probability gap between sampler and trainer above 0.02; an invalid-call rate above 5% and at least twice its step-0 value. A stopped run is reported as stopped and is not started again, unless `--allow-resume-after-fault` is given (`RESUME_AFTER_FAULT=1` of `scripts/gpu/chain_stage2.sh`). The record of the stop is kept in the run folder: `stopped.json` and `starts.jsonl`; after such a restart `stopped.start_<n>.json`, `train_log.start_<n>.jsonl` and `rollouts/start_<n>/`. If the run then finishes, its run manifest has the status `complete_after_fault`, not `complete`.
- **Harness errors:** attempts that the harness lost are left out of every metric and counted in every table row (`harness_error_count`). Gate: below 1% of a run's attempts; a run that fails it is repaired and repeated, and `NOTES.md` says so. No script enforces the gate for an evaluation run: `scripts/gpu/rollout.py` only writes `harness_error_rate` into `run_manifest.json`. For a GRPO run that was started again, its run manifest counts the harness errors of the final log (`summary`) and of the steps that were run again (`steps_run_again`); both counts are reported.

## 8. Audit

1. Sample: all top-reward dev attempts of the final checkpoints. 20 of them (5 per outcome, fixed seed) and every flagged one are read by hand against the ground truth.
2. Automatic checks: `scripts/replay_check.py` replays the saved calls and grades them again with a second, separately written verifier. A monitor flags ids that no earlier tool result showed.
3. Reverse check: failed attempts are read for correct work that the verifier rejected.
4. Formula audit: no failing attempt of any GRPO group has a positive reward.
5. Positive control: the audit must flag a verifier that is broken on purpose.

The code of lines 1, 3, 4 and 5 is written after the freeze and before the test run.

## 9. Fallbacks

| Case | What is done |
|---|---|
| The overfit check fails and is not repaired in time | Stage 2 becomes iterated rejection sampling, which the assignment allows: 8 attempts of `sft` on each training scenario, the SFT data rule, one more round of SFT on them. Dev evaluation with the run seeds 0, 1 and 2, each its own row. One GRPO seed if time is left |
| Only one GRPO seed finishes | Dev evaluation of its checkpoint with the run seeds 0, 1 and 2, each its own row. The write-up says that there was one training run |
| Both seeds are flat and no fault is found | A null result, reported with the intervals and the smallest detectable effect |

## 10. Reported whatever the outcome

Every arm and every seed as its own row, with the metrics of section 5. Every comparison of section 4. The test numbers, or the sentence that the test run did not finish. The log of every GRPO step (`rollout_stats` in `accessdesk/train/grpo.py`). The dev curve of each GRPO run (the set `curve_subset` at steps 0, 20, 40, 60 and 80); no checkpoint is chosen by it. A small, zero or negative gain as it is. Every change after the freeze, and every use of `--allow-changed-tree` or `--allow-resume-after-fault`.

Not results: runs on calibration families (`results/calibration/`) and trial runs (`results/_smoke/`). They chose the text, the token cap, the scenario design and the model before the freeze, and they are in no results table.
