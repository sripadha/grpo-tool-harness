# NOTES

Here "the assistant" is the coding assistant Claude Code, a closed API model, and "agents" are the helper agents that it started. "The freeze" is when the hashes of the scenarios, splits, graders and evaluation code went into `MANIFEST.json`. To keep the package small, the working log, the working notes and the working history were left out. Where a statement rests on them, this file says so.

## 1. Tools and models

The assistant ran on closed models. Every model in a training run or a reported dataset is open-weight. Only one text set was written by the assistant, a placeholder set. It was used by the tests and two smoke rollouts, never in a training run or a reported number (section 7).

| Tool or model | What it did | Licence |
|---|---|---|
| Claude Code | The assistant: code, tests, analysis plan, text of the documents | Closed |
| Claude Fable 5.2, Claude Opus 5.5 | Models of the assistant and its agents (Opus 5.5 also for checker agents) | Closed |
| Qwen3.5-9B | Wrote the scenario text. One calibration rollout | Open (apache-2.0) |
| Qwen3.5-2B | Trained and evaluated | Open (apache-2.0) |
| Qwen3.5-4B | Calibration. Teacher in the rehearsals of the teacher arm (no real run, section 8) | Open (apache-2.0) |
| torch, transformers, peft | The project's SFT and GRPO loops | Open |
| TRL 1.18.0 | Only in `tests/gpu/test_trl_advantages.py`, to compare advantages | Open |
| accelerate, flash-linear-attention, fla-core, matplotlib, pytest, uv, markdown, xhtml2pdf | GPU stack, figure, tests, `WRITEUP.pdf` (versions: README sections 1 and 3) | Open |
| RTX 4090 on Vast.ai | All GPU work, 10 rentals | Hardware |

Licences as read on the Hugging Face model pages on 2026-11-19.

## 2. Who did what

The assistant and its agents wrote all code and all tests, the analysis plan and the text of the documents. Sri made the decisions below, wrote six end states by hand, labelled the audit sheet by hand, read the results, and asked for changes to the documents.

- **Sri decided:** Census names, two separate versions of the six core functions, the SFT filter, the rule that chose the model, the freeze, two seeds on two machines, the test run at night.
- **Left to the assistant:** the seeds, chosen without a rule, and the GRPO step of 8 scenarios x 16 attempts. The assistant also set the token cap, the loss constant and the learning rate, each by a written rule (section 6).

## 3. What an agent got wrong, and how it was noticed

A working log of 64 entries was kept during the work. Six cases from it follow. The log records no wrong loss mask.

| Name | What was wrong | How noticed | What was done |
|---|---|---|---|
| Wrong token | End of turn taken as id 248044 (padding), not 248046 | Special ids printed | Fixed and tested (`tests/core/test_tokens.py`) |
| Split by name popularity | Names went to splits in file order, and the files run from common to rare names. So train would get the common names, and dev and test the rare ones. The log said this was fixed before it was | The assistant read the generator next to the name files. Later an agent compared the log with the code | Names now go to splits by a seeded shuffle. A test checks that every split gets common and less common names (`tests/core/test_generator.py`) |
| Refused call | A refused made-up id, then correct work, scores 100 | Scripted attack by an agent | The reward was kept (`ANALYSIS_PLAN.md` section 6). SFT data leave such attempts out. Monitor m1 counts them |
| Rule order not tested | Two rule facts were never true together, so the order of the four rules was never tested: a worker that tried them in reverse order passed every request. Also, only 8 of 48 dev families had two people with one name, so a worker that never read the department still passed about 90% | Two attack agents on two models found it, by scoring simple rules | Before the freeze the harder design was chosen (section 6): rule facts can be true together, and half the families have two people with one name |
| First-match shortcut | In an early draw of the data, before the freeze, the person meant was the first of two people with the same name in 6 of the 8 such dev families. A worker that always took the first one scored 95.8% on that dev split | An attack agent scored the rule "take the first match" on each split | Fixed: in every split and outcome, the person meant comes first in half of these families and last in the other half. A test checks this (`tests/core/test_generator.py`) |
| Rules prompt lost "first rule wins" | In the second text round the open-weight model wrote the rules prompt again, and the sentence that the first rule that fits wins was gone. No check looked for it | Calibration round 2: on the 14 families where the rule order matters the 4B passed 6.9%, on the others 90.9% (`accessdesk/drafting.py`, note on check C6). An agent and the assistant read the numbers | Three checks were added in `accessdesk/drafting.py`: the order sentence (C6), where the account state is read (C7), and the words between role and department (C8). The fields were drawn again |

## 4. How the key parts were verified

| Part | Check | Result |
|---|---|---|
| Six core functions | Main version (Fable agent) against a second version from the docstrings only (Opus agent): `tests/core/test_differential_*.py` | Defects in the main versions fixed. None in the loss mask |
| CPU tests | `pytest tests/core`, fresh copy of the package, 2026-11-21 | 4,272 tests: 4,178 passed, 94 skipped, 0 failed, 13 minutes |
| GPU code | 110 tests on the 2B, among them: a masked token gets no gradient, planted unmasked pads are seen | 110 passed, 2026-11-18. Planted mistake: 0.82 against 0.047 (`results/gpu_logs_2026-11-18/`) |
| Sampler against trainer | Log-probability gap | At most 0.0049 (seed 1) and 0.0046 (seed 2) per step. The stop rule is 0.02 (`results/main/report/training/steps_seed{1,2}.csv`, column `logprob_gap_mean`) |
| GRPO loop | Overfit check, 2 families | Pass rate 0.594 at step 0, 1.0 at step 2 |
| Grades | Saved calls replayed, both verifiers | 0 differences in 45,696 attempts, test included |
| Training maths | Advantages made again (`results/main/audit/formula/`) | The counts are equal. The loss check is weak, because its bound can be larger than the loss. The trainer does not save its own advantages, so this does not prove that the trainer used them |
| Tables | Separate recount from saved attempts (`results/main/audit/recount/`). Test part written after the test results were opened | Test: 651 comparisons, 0 differ. Dev: 0 differ |
| Audit power | 3 verifiers broken on purpose | 3 of 3 caught on 128 probes |
| By hand | Sri's six end states. Sri's blind audit of 31 dev GRPO attempts | End states: 5 of 6 at first, because one entry had a role id where a person id belongs. 6 of 6 after that entry was corrected (`hand_states/form.yaml.checks.log`). A note of the assistant had named the outcome of a set that holds one of the six families. An agent found this, and the lines were moved before Sri filled the form. Audit: the form had two labels, "end state right" and "path acceptable". On "end state right" the sheet agrees with the verifier in 19 of 31 attempts, or 19 of 28 without the 3 labels of dv-46. Those 3 were not blind: the assistant had told Sri before the audit that dv-46 fails. All 10 failed attempts have the label "path acceptable: no". Not an independent check (section 7) |
| Rebuild | `scripts/rebuild_results.sh`, fresh copy, 2026-11-21, about 4 minutes. Makes every table again from saved attempts, runs no model | 73 equal, 17 equal by rule, 0 different, 14 new (readable reports with no saved copy) |

## 5. The freeze and what changed after it

The freeze: 2026-11-18T21:48:10Z, signed by Sri, 61 files in 11 groups, before the first step of a reported training run (SFT data chosen 2026-11-19T00:17:15Z). Calibration and trial steps came before it (section 6). `code_sha256` `a157f321...` never changed. All 44 run manifests in `results/main`, test runs included, name it. Changes are amendments in `MANIFEST.json`, hashes per state in `results/manifest_history.csv`. Only its last row can be checked from the package (`sha256sum MANIFEST.json`). The earlier rows come from the working history, which is not published.

- **Amendments 1 and 2** (2026-11-19, 06:53 and 18:06 UTC): new groups `audit` (10 files) and `report` (6 files), written after the freeze. They only read saved runs.
- **Amendment 3** (2026-11-21): `PREDICTIONS.md` left the group `plan` and the package. The brief does not ask for it, and Sri chose a smaller package. It is not code, scenario, split, grader or evaluation, and no reported number depends on it. `code_sha256`, `text_sha256` and `run_config_sha256` are unchanged. Sri kept the file and can show it. Sri signed its 14 lines before any baseline. Before Sri answered, the assistant gave two guesses of its own and said that one of Sri's reasons was not right. Its SHA-256 at the freeze is the field `old_sha256` of amendment 3 in `MANIFEST.json`, so a reader can check that it is the frozen file.
- **Now:** `scripts/freeze.py check` reports 76 files in 13 groups, nothing changed.
- **Written after the freeze, not frozen, among others:** `scripts/gpu/chain_test.sh` (SHA-256 `48ea41c8...`, as in `results/main/test_run_record.txt`), a gradient measurement script, `scripts/render_writeup.py`, `scripts/make_package.sh`, `tests/core/test_final_splits.py`, and the recount and count scripts under `results/main/audit/`. Changed: `accessdesk/freeze.py` and its test `tests/core/test_freeze.py` (for the new groups and for amendment 3), `.gitignore`.
- **Training code** is not frozen. Each training run records the SHA-256 of its 14 files (`training_code` in its run manifest). The files in the package equal the records of `sft_data`, `sft`, `grpo_seed1` and `grpo_seed2` (own check, 2026-11-21).

## 6. Choices made after a result, and the held-out data

No choice used a test result. Five calibration rounds ran before the freeze (18 rollouts, 17,024 attempts). They chose the text, the model, the design and the first three settings below, and no number of them is in a results table. Two used the dev split of an earlier draw (`data/scenarios_calib5_mixed`), not the frozen dev. It shares no world, person id or request text with the frozen dev, though some names recur.

| Choice | Rule | Data | When |
|---|---|---|---|
| 2B, harder design | Untrained 2B between 30% and 80% (it was 44.8%) | Calibration | Before freeze |
| Token cap 260 | Twice the 99th percentile call length | Calibration | Before freeze |
| Loss constant 200 | Median tokens of a full-reward attempt | Calibration | Before freeze |
| 8 scenarios x 16 attempts | SFT median family pass rate above 0.8 | Train | After SFT |
| Learning rate 1e-5 | Pilot gate | Train | Before seeds |

**Pilot gate** (`ANALYSIS_PLAN.md` section 7). The plan asks for these judgements in `NOTES.md`. They were written in a working note instead. At 02:10 UTC on 2026-11-19, 3 minutes after the pilot started and before any pilot number was seen, the assistant wrote the two judgements down. "The reward has moved": on the scenarios of pilot steps 8 to 14 that the SFT model did not always pass in its profile rollout (`results/main/sft_train`), the pilot passes at least 10 points more often. With fewer than 5 such scenarios the overfit check decides (pass rate 0.59 to 1.0 at 1e-5). "The stop-rule numbers are normal": no step has a fault and the mean log-probability gap stays below 0.02. At 02:55 UTC, after the pilot: no fault, largest gap 0.0051, and only 4 such scenarios. On them the SFT model passed 50.0% of its attempts and the pilot passed 98.4%. So the overfit check decided, and both seeds used 1e-5. The gate would have chosen 3e-5 only if the reward had not moved. The two times come from the note's own headings: the note entered the working history at 19:21 UTC that day, after the seeds had started (03:32 UTC). The pilot's files are in `results/main/checks/pilot/`.

**Trial runs** (`results/_smoke/`) are not results. Rehearsals after the freeze record the frozen code hash. Runs before the freeze record older code hashes (`9004c625...` and `c16a40e9...`), because the code still changed before the freeze. Twelve of the run manifests (`results/_smoke/extras_distill/`, `extras_rs2/`) are rehearsals of the teacher arm and of one more round of rejection sampling (section 8). Neither had a real run. Their code is not in the package. The cost report counts their GPU seconds (`results/main/report/training/training_report.json`, field `cost`).

**Test day.** The test set ran once, 2026-11-19, 09:31 to 11:14 UTC, 0 harness errors (`results/main/test_run_record.txt`), and stayed closed. Its 33 file hashes were written at 11:17 UTC (`results/main/test_run_sealed.sha256`). Sri's audit form was checked at 18:33 UTC (`results/main/audit/audit_labels.yaml.checks.log`). The results were opened at 18:38 UTC, 33 of 33 hashes OK. That last time is from the working history. So the test set ran before Sri's hand audit, although audit first is the usual order. The frozen plan asks only that the test run comes after the freeze and the dev table.

**Test inputs.** At 07:10 UTC on 2026-11-19, after dev and before the test run, the assistant read the test scenario file (inputs only) and named te-08 and te-31 to compare with dv-46. That time is from the working notes. Nothing frozen, no setting and no checkpoint changed after that.

**Audit key.** It was in the repository before the form was filled. Sri states that it was not opened before the labels were finished.

**Not run.** The plan's two extras (`ANALYSIS_PLAN.md` section 3), the teacher arm and the hint check, did not run. Sri decided against them. A third extra, one more round of rejection sampling, was designed after the dev table. It did not run either. Only the teacher arm and that round had rehearsals (section 8). The stretch item (design: `results/stretch.md`) did not run: it was not in the frozen plan, and the time went to the required parts. It needs new training code and about 17 GPU hours, for little gain (3.6 points in 5 of 48 dev families). Sri decided not to run it.

**Second desk** (`results/main/audit/second_desk/`): 8 rollouts outside the chain scripts. No adapter hash was recorded.

**Steps where every group was dropped:** 30 of the 80 GRPO steps of seed 1 and 27 of seed 2. The trainer still makes the optimiser step there, so the weights can move a little from the momentum of earlier steps. The size was not measured.

## 7. Data and provenance

Only public and synthetic data were used: Census name files (`data/names/PROVENANCE.md`), a seeded generator, and 452 passing attempts of the untrained 2B as SFT data.

The open-weight Qwen3.5-9B wrote every value of `text/model_text.yaml`, except five tool names (from the brief's example list) and repeated parameter names. A program took the first reply that passed fixed checks and cleaned it by fixed rules. It added no word. Sri approved the text on 2026-11-18.

**Closed model and data.** The assistant chose the 9B, wrote its instructions and the checks, read each draft and asked for new draws of fields it judged wrong. It wrote and edited no value and chose no candidate. A candidate that repeated 8 or more words of its instructions was refused. The longest run of the assistant's words in a kept value is 6. The assistant also wrote `text/placeholder_text.yaml` (its first line says so). Only the tests and two smoke rollouts (`results/_smoke/first_rollout_*`) used it, no training run or reported number.

**Closed model and evaluation.** No closed model wrote a name, was sampled as the policy, or gave a grade in a reported number. Code, written by the assistant, gives every grade. The assistant did read attempts: the 12 hand-audit disagreements with the answer key open, judging the verifier right in all 12 (WRITEUP.md section 6), and 20 pilot attempts. Its reading of the 12 is a working note. The 12 are listed in `results/main/audit/hand_audit_report.json`, field `disagreements`. So the hand audit is not an independent check.

**Employer data.** No file names any. Sri confirms that no data of a current or past employer was used.

## 8. Time and cost

- **Sri's time:** about 20 hours (Sri's estimate): reading, decisions and checks, hand labels, practice for the call. The working time of the assistant and its agents is not counted.
- **GPU:** 36.95 rented GPU hours over 10 rentals, 19.58 before the freeze and 17.36 after. Of the hours after the freeze, the run manifests record 13.07 as run time (README section 5). SFT 295 s (`results/main/sft/run_manifest.json`), GRPO 2.47 and 2.62 hours, test run 1.69 hours (`results/main/report/training/training_report.json`).
- **One rental for code that is not in the package:** rental 61480327 (2026-11-19, 16:36 to 18:11 UTC, `results/rentals.csv`) ran GPU tests and rehearsals of two extras that never had a real run: the teacher arm of the plan, and one more round of rejection sampling that was added after the freeze. Their code is not in the package, and no number in the package comes from them. The rental is counted in the hours and the money.
- **Money:** ledger estimate 19.44 USD (`results/spend_ledger.csv`). Vast.ai charged 18.955 USD: GPU 17.732, disk 0.808, bandwidth 0.415, read by the assistant from the Vast.ai API after the last machine was destroyed. Most of the 0.49 USD gap is on two machines that never started. The invoice is the authority. The billing export is sent beside the package.
