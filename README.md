# A small GRPO run on a tiny model: the access desk

## 1. What this is

A small IT access-request desk (an in-memory environment with five tools), a harness that runs a model on it, and the training and evaluation code. The model is Qwen/Qwen3.5-2B with a LoRA adapter (a small set of added weights), trained by SFT and then by GRPO with two training seeds.

- `WRITEUP.md`: the question, the approach, the results, what did not work, what to do next. Its two pages are counted on `WRITEUP.pdf`, made by `python scripts/render_writeup.py` (A4, 2 cm margins, 10 pt text, 8 pt tables; needs markdown 3.11.3 and xhtml2pdf 0.2.28).
- `NOTES.md`: the tools and models used, the mistakes of the coding agents, how each part was checked.
- `ANALYSIS_PLAN.md`: the analysis plan, frozen before training. Its section 3 also names two planned extras, a teacher arm and a hint check. They were not run.
- `MANIFEST.json`: the freeze record: the SHA-256 of the frozen files (76 files in 13 groups). The record has three amendments, and NOTES.md lists them.

Read `WRITEUP.md` first, then `NOTES.md`. For a check without a GPU, set up section 3 and run the three commands at the top of section 4 (about four minutes).

## 2. Layout

**Code to read first.** The core is 13 files of `accessdesk/` (`wc -l` lines):

- desk and rules: `env/world.py` (35), `env/tools.py` (153), `env/rules.py` (109);
- verifier and reward: `env/verifier.py` (68), `env/reward.py` (67);
- runner: `harness/episode.py` (207), `harness/hf_backend.py` (588);
- loss mask: `train/loss_mask.py` (80); SFT step: `train/sft.py` (332);
- GRPO step: `train/grpo.py` (895), its advantage formula: `train/grpo_math.py` (136);
- metrics and bootstrap: `eval/metrics.py` (423), `eval/bootstrap.py` (72).

Together 3,165 lines; `accessdesk/` holds 14,662 and `tests/` 44,960. The rest is tests, checks, reports and run scripts.

| Folder | What it holds |
|---|---|
| `accessdesk/` | The Python package: the core above, `audit`, and `_second` (second versions of the core functions). |
| `scripts/` | CPU scripts. `scripts/gpu/`: GPU scripts and chain scripts. `scripts/remote/`: machine set-up. |
| `tests/` | `tests/core`: the CPU tests. `tests/gpu`: tests that need torch and a GPU. |
| `data/` | `data/scenarios`: the frozen scenarios. `data/names`: public name lists. |
| `text/` | `model_text.yaml`: every word that the model reads. |
| `results/` | `main/`: every frozen run. `calibration/`: runs before the freeze (not results). `_smoke/`: run manifests of trial runs and rehearsals (not results). `drafting/`: logs and text files of the rounds in which Qwen/Qwen3.5-9B drafted `text/model_text.yaml`. The cost ledger (`spend_ledger.csv`, `rentals.csv`), `manifest_history.csv` (every state of `MANIFEST.json`, read by the rebuild), `stretch.md` and the GPU logs. |
| `hand_states/` | Six end states written by hand, and their check log. |

## 3. Environment

| | Laptop (no GPU) | GPU machine (every reported run) |
|---|---|---|
| Python | 3.11.15 | 3.11.17 |
| torch; peft; accelerate | not installed | 2.15.0; 0.21.3; 1.16.0 |
| transformers; tokenizers; safetensors; numpy | 5.23.0; 0.23.3; 0.8.0; 2.4.6 | the same |
| jinja2; huggingface-hub | 3.1.6; 2.4.0 | the same |
| flash-linear-attention; fla-core; causal-conv1d; kernels | not installed | 0.6.0; 0.6.0; 1.7.0; 0.18.3 |
| trl (one GPU test only); triton | not installed | 1.18.0; 3.9.0 |
| pyyaml; pytest | 6.0.3; 9.2.0 | the same |
| matplotlib (figure only) | 3.11.2 | not used |
| CUDA; GPU | none | 13.2 (from the torch wheel); one NVIDIA GeForce RTX 4090, 23.5 or 23.6 GB (`gpu_memory_gb` in the run manifests) |

Of the 44 run manifests in `results/main`, 33 are of GPU runs (the other 11 ran on CPU: 10 reference rows and the choice of SFT data). All 33 record Python, CUDA, the GPU and the versions of the GPU column, except jinja2, huggingface-hub, trl, triton, pyyaml and pytest. Those six come from `results/gpu_logs_2026-11-17/env_freeze.txt`.
Model: `Qwen/Qwen3.5-2B` at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, in bfloat16. Its 187 linear modules hold 1,881,276,416 weights (own sum of the shapes in `results/main/sft/lora_coverage.json`, `lm_head` included). Licence: apache-2.0 (NOTES.md section 1).

**Local environment.** These commands need uv (https://docs.astral.sh/uv/). In a fresh copy of the repository on 2026-11-19 they (uv 0.11.19) made, in 12 s, an environment with the versions of the laptop column:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -c requirements-gpu.txt -e '.[dev]'
uv pip install --python .venv/bin/python -r scripts/requirements-report.txt   # matplotlib, for the figure
```

Without uv, this pip form (with CPython 3.11.15) made the same versions in a fresh copy of the repository on 2026-11-19:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -c requirements-gpu.txt -e ".[dev]" -r scripts/requirements-report.txt
```

**GPU machine.** Container image `vastai/pytorch:cuda-13.0.3-auto` (digest read for this tag on 2026-11-17: `sha256:e052540e87b18c5b9f8048826ccd3c9c9f0bf6ae6f9245bd6ba684fbd362ef27`); the machines were rented by tag, so a later pull can give another image. `results/gpu_logs_2026-11-17/` holds, for the first machine only, the package list (`env_freeze.txt`, 80 lines), the driver (615.71.09, `setup.log`) and the set-up summary (step 9: "the pinned versions are the installed ones"). The later machines were set up by the same script from the same pinned files. `requirements-gpu-optional.txt` lists optional packages that `scripts/remote/setup_box.sh` installs one at a time.

## 4. Path A: no GPU

Run every command from the top folder, after section 3. Scripts start with `bash`, so an unpacker that drops the executable bit does no harm. The first three commands give a result in about four minutes:

```bash
.venv/bin/python scripts/freeze.py check
(cd results/main && sha256sum --ignore-missing -c test_run_sealed.sha256)
bash scripts/rebuild_results.sh                 # writes into a new temporary folder; or --out DIR (new or empty)
```

1. The freeze check must print: `nothing changed: 76 files in 13 groups are as in MANIFEST.json (code_sha256 a157f32175cc4a24092f916a8dcf2a382a0ac4764b81d60d94f28a02963202f3)`.
2. The hash check compares the files of the test run with the hashes in `test_run_sealed.sha256`. Its header says they were written before anybody opened a test result. Only the working history, which is not published, shows that order. It must print `OK` for 29 files. The other 4 of the 33 are the `tokens.jsonl` files of the test runs, not in git (section 7); without `--ignore-missing` they give `FAILED open or read` and exit code 1.
3. The rebuild makes every table value and the figure again from the saved attempts and compares each made file with the saved one. "Equal by rule": the file differs only in the path of the fresh folder, in a time stamp, or (for a run manifest) in the hash of `MANIFEST.json` with the same code hash. In a fresh copy of the package on 2026-11-21 it took about 4 minutes, ended with exit code 0 and printed:

```
summary: 73 equal, 17 equal by rule, 0 DIFFERENT, 14 new (no saved file), 0 saved but not made; 68 commands ran, 0 failed; 0 missing input(s)
```

A file marked `NEW` is a readable report that the rebuild makes but whose saved copy is not in the package. It counts as "new (no saved file)" and does not change the exit code. `DIFFERENT` must stay 0.

Then two more checks: a recount by separate code, and a count of the monitor flags among the top-reward attempts (this one reuses the package's monitors in `accessdesk/eval/monitors.py`, so it is not an independent check of them):

```bash
.venv/bin/python results/main/audit/recount/rederive.py --out /tmp/recount
.venv/bin/python results/main/audit/monitor_counts/count.py --out /tmp/monitor_counts
cmp /tmp/monitor_counts/top_reward_flags.md results/main/audit/monitor_counts/top_reward_flags.md   # prints nothing when equal
```

The recount imports nothing from `accessdesk`. The headers of `rederive.py` and `rederive_table.py` list the tables it covers and its commands. `rederive_console.txt`, `rederive_test_console.txt` and `rederive_base_vs_sft_console.txt` show what they print.

Last, the whole test suite. It took 13 minutes in a fresh copy (see below) and can take longer on a slower laptop:

```bash
.venv/bin/python -m pytest tests/core
```

It must end with 0 failed. In a fresh copy of the package on 2026-11-21 (no `.cache/`, no `_dev/`) it collected 4,272 tests: 4,178 passed, 94 skipped, in 13 minutes. Most of the tests with the marker `tokenizer` are skipped without the tokenizer files of Qwen/Qwen3.5-4B and Qwen/Qwen3.5-2B in `.cache/hf/` (not in git). These commands get them (no weights; the revisions of `scripts/remote/setup_box.sh`); on 2026-11-19 each fetched six files with the SHA-256 of the laptop's copies. With them, `.venv/bin/python -m pytest -m tokenizer tests/core` runs these tests:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', allow_patterns=['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja','config.json'], local_dir='.cache/hf/Qwen3.5-4B')"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B', revision='15852e8c16360a2fea060d615a32b45270f8a8fc', allow_patterns=['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja','config.json'], local_dir='.cache/hf/Qwen3.5-2B')"
```

**Single commands. They write over the saved files.** In a git clone, `git diff` then shows any change. The unpacked zip has no git: copy the folder first, or use `bash scripts/rebuild_results.sh --only <step>`, which compares for you.

```bash
# one table: base against SFT on dev (bootstrap: 10,000 draws, seed 0, level 0.95 by default)
.venv/bin/python -m accessdesk.eval.tables --arm base=results/main/base_dev/episodes.jsonl --arm sft=results/main/sft_dev/episodes.jsonl --baseline base --pair base:sft --out-json results/main/tables/base_vs_sft.json --out-md results/main/tables/base_vs_sft.md
# one monitor report (arm: base, sft, grpo_seed1, grpo_seed2; split: dev or test)
.venv/bin/python scripts/monitor_report.py --episodes results/main/sft_dev/episodes.jsonl --scenarios data/scenarios --split dev --text text/model_text.yaml > results/main/monitors/sft_dev.md
# the figure
.venv/bin/python scripts/make_figure.py --run results/main/grpo_seed1 --run results/main/grpo_seed2 --scenarios data/scenarios --out results/main/report/figure
```

## 5. Path B: with a GPU

**Machine set-up.** Every GPU run used one rented Vast.ai machine (Linux x86_64, `requirements-gpu.txt` line 1) with one RTX 4090. `scripts/remote/rent.py` rents, copies, runs, fetches and destroys a machine (it needs a Vast.ai account and key). On the machine, in the copied repository, `bash scripts/remote/setup_box.sh` makes `/workspace/venv`, installs the pinned packages, downloads the pinned model snapshots and writes `/workspace/env.sh` for the chain scripts. With `REQUIRE_FROZEN=1` a chain starts only if the tree is as `MANIFEST.json` says.

The GPU tests of the training code are the 8 files below. On 2026-11-18, before the freeze, they gave 110 passed in 284 s (`results/gpu_logs_2026-11-18/trial_gpu_tests.log`). The log holds no command line, so this command is a reconstruction:

```bash
ACCESSDESK_TEST_MODEL=Qwen/Qwen3.5-2B python -m pytest tests/gpu/test_train_fixes_gpu.py tests/gpu/test_forward.py tests/gpu/test_grpo_step.py tests/gpu/test_lora.py tests/gpu/test_policy_loss_grad.py tests/gpu/test_trl_advantages.py tests/gpu/test_sampler.py tests/gpu/test_rollout.py -q -rA
```

The fields `command` and `seconds` of each run manifest hold the exact command and the GPU seconds (without the loading of the model).

```bash
REQUIRE_FROZEN=1 bash scripts/gpu/chain_stage1.sh                      # stage 1
export GRPO_OPTIONS="--reference-tokens 200" REQUIRE_FROZEN=1
CHECK=overfit OVERFIT_EPISODES=results/main/sft_train/episodes.jsonl bash scripts/gpu/chain_stage2.sh; CHECK=pilot bash scripts/gpu/chain_stage2.sh   # checks, not results
SEED=1 bash scripts/gpu/chain_stage2.sh                                # GRPO seed 1; SEED=2 ran at the same time on a second machine
bash scripts/gpu/chain_test.sh                                         # the test run, once
```

Stage 2 reads, from stage 1 under `OUT`: `sft/`, `sft_profile/decision.json` (the GRPO step: 8 scenarios x 16 attempts), and `base_dev`, `sft_dev` (for the table of three arms); the machine of seed 2 got copies of these four. Learning rates: code defaults (`accessdesk/train/sft.py` 1e-4, `accessdesk/train/grpo.py` 1e-5), in the run manifests (`sft_config.lr`, `grpo_config.lr`).

**Running again.** A chain skips each step whose folder under `OUT` (default `results/main`) is complete and fits the chain, so in a fresh copy these commands run no rollout and no training (on a machine without `python` on the path, set `PYTHON=.venv/bin/python`). Run in a fresh copy on 2026-11-21: `chain_stage2.sh` with the options above skipped its training and rollout steps, made the table and monitor files again with the same bytes, and ended with `CHAIN_DONE`. `chain_stage1.sh` skipped `base_dev` and `base_train` and then stopped with `CHAIN_FAILED sft_data` (exit code 1), because that step checks the SHA-256 of `results/main/base_train/tokens.jsonl`, which is not in git (section 7). With a new folder: `OUT=results/try bash scripts/gpu/chain_stage1.sh`, then `OUT=results/try SEED=1 GRPO_OPTIONS="--reference-tokens 200" bash scripts/gpu/chain_stage2.sh`. `chain_test.sh`, the record of the one test run, writes to `results/main` (with `DRESS=1`, to `results/_smoke/test_dress`), finds its four test folders complete and runs no rollout. It still adds a record of the run to `results/main/test_run_record.txt` and writes the five reference run manifests of the test split again, so the hash check of section 4 then reports these six files. Run it only in a copy. For new test rollouts, run the field `command` of `results/main/<arm>_test/run_manifest.json` with another `--out`.

| Stage | Run | Seed | Attempts | GPU s | Folder in `results/main` |
|---|---|---|---|---|---|
| 1 | base on dev | run seed 0 | 1,536 | 1,725.1 | `base_dev` |
| 1 | base on train | run seed 10 | 4,800 | 5,399.2 | `base_train` |
| 1 | choice of SFT data: rejection sampling (no GPU) | none | 452 examples | 2.09 | `sft_data` |
| 1 | SFT, 2 epochs | 0 | | 295.1 | `sft` |
| 1 | sft on dev | run seed 0 | 1,536 | 1,751.0 | `sft_dev` |
| 1 | sft on train (family profile) | run seed 20 | 2,400 | 3,093.7 | `sft_train`, `sft_profile` |
| checks | overfit check, 30 steps | 101 | 960 | 1,141.9 | `checks/overfit` |
| checks | pilot, 15 steps | 100 | 1,920 | 2,336.2 | `checks/pilot` |
| 2 | GRPO, 80 steps | 1 | 10,240 | 8,903.7 | `grpo_seed1` |
| 2 | GRPO, 80 steps | 2 | 10,240 | 9,427.7 | `grpo_seed2` |
| 2 | step-80 adapter on dev | run seed 0 | 1,536 each | 1,141.7; 1,208.3 | `grpo_seed1_dev`, `grpo_seed2_dev` |
| 2 | curve: SFT adapter and steps 20 to 80 | run seed 0 | 80 per point | 471.5; 433.2 | `grpo_seedN_curve_000` to `_080` |
| test | base, sft, grpo_seed1, grpo_seed2 | run seed 0 | 1,536 each | 1,821.5; 1,957.3; 1,166.9; 1,140.1 | `<arm>_test` |

The curve uses 20 dev families in one wording style, with 4 attempts per scenario. The test run ran on 2026-11-19 on a third machine (`results/main/test_run_record.txt`), after two rehearsals on calibration families (`DRESS=1 bash scripts/gpu/chain_test.sh`).

The second desk (a changed rule in the prompt) and the layout-gradient check ran outside the chain scripts; their commands are in the field `command` of the run manifests of `results/main/<arm>_shift_differ` and `<arm>_shift_control`, and of `results/main/checks/layout_gradients/layout_gradients.json`. The eight second-desk rollouts took 2,405.6 GPU s, the layout check 1,213.1 s.

The training code in the package is the code that the runs recorded: each training run manifest holds the SHA-256 of its 14 files (field `training_code`), and the files here equal them (NOTES.md section 5).

**Totals.** By `results/main/report/training/training_report.json` (fields `cost.hours_in_run_manifests_after_freeze` and `cost.stages`): the run manifests after the freeze hold 13.07 GPU hours, the test run 1.69 of them. Rented hours and money: NOTES.md.

## 6. Seeds

| What | Seed | Recorded in |
|---|---|---|
| Text that the model reads (written by Qwen/Qwen3.5-9B) | run seed 13 | header of `text/model_text.yaml` |
| Scenarios (train, dev, test) | data seed 20261118 | `accessdesk/config.py`, `data/scenarios/summary.json` |
| Calibration split of `data/scenarios` | 20261119 | `accessdesk/config.py` |
| Calibration folders `data/scenarios_calib*` | data seed 1, calibration seed 2 | `data/scenarios_calib*/summary.json` (field `config`); `ANALYSIS_PLAN.md` section 2 (data seed) |
| Rollouts on dev and test (every arm, curve, second desk) | run seed 0 | each run manifest |
| Base on train (SFT data); sft on train (profile) | run seed 10; 20 | `base_train`, `sft_train` run manifests |
| SFT training (LoRA start values, example order) | 0 | `results/main/sft/run_manifest.json` |
| GRPO training | 1 and 2 | `results/main/grpo_seed{1,2}/run_manifest.json` |
| Sampling inside GRPO (made from the training seed) | 3138651447157350 (seed 1), 8888994274034250 (seed 2) | field `sampling_run_seed` |
| Pilot; overfit check | 100; 101 | `results/main/checks/*/run_manifest.json` |
| Layout-gradient check (random half splits) | 0 | `results/main/checks/layout_gradients/layout_gradients.json` |
| Bootstrap (10,000 draws, level 0.95) of every table, the second-desk report, panel (d) of the figure and the stretch numbers | 0 | each table JSON; `results/main/audit/second_desk/second_desk.json`; `scripts/make_figure.py`; `scripts/stretch_numbers.py` |
| Smallest detectable effect (200 random half splits) | 0 | `accessdesk/eval/metrics.py` |
| Hand-audit sample; reference rows; positive-control probes | 0 | `scripts/audit_sheet.py`; reference run manifests; `results/main/audit/positive_control/` |
| Choice of the six hand-written end states | 0 | `hand_states/form.yaml.checks.log` |

## 7. What is not in the repository

`.gitignore` keeps these files out of git. In `results/main`:

- 19 adapter weight files (`*.safetensors`, 1,188,672,072 bytes);
- 101 files under `checkpoints/` folders (3,820,023,557 bytes, adapters included);
- 28 `tokens.jsonl` files of the rollouts (291,890,138 bytes): the token ids and log-probabilities of each attempt;
- 6 raw gradient dumps of the layout check (`*.npz`, 375,982,336 bytes); the JSON summary is kept.

Also ignored: `results/_smoke` except its run manifests and the two rehearsal records; the token files of `results/calibration`; `.venv/`, `.cache/`, `dist/` and the working notes (`_dev/`, and Sri's own `research_notes/` outside the repository, named in some code comments; not needed).

Working documents are not part of the package: the predictions, the working notes, the designs of the extras that were not run, the assistant's reading of the hand-audit disagreements, and the readable .md copies of some reports. The rebuild (section 4) makes most of these reports again and prints them as NEW. Some it does not make again: the SFT data report (it needs the tokenizer), the recount reports (`results/main/audit/recount/rederive.py` writes them), the calibration reports and a few README files. Their numbers are in the .json files beside them. Some code comments, run manifests, `results/rentals.csv` and the frozen `ANALYSIS_PLAN.md` still name such files, or commits of the working history, which is not published.

The three final adapters are in git and in the zip: `results/main/sft/adapter` and `results/main/grpo_seed{1,2}/checkpoints/step_00080/adapter`. Path B makes the other files again, not always with the same bytes; the run manifests and `test_run_record.txt` hold the SHA-256 of the adapter files. git keeps the token files of the training steps (205 files `rollouts/step_*.tokens.jsonl`, 287,949,466 bytes), so that the loss of each step can be computed again.

**The zip.** `bash scripts/make_package.sh` zips the last commit (`git archive`) into `dist/<folder name>_<short commit>.zip`, with its SHA-256 and a summary beside it. It refuses a tree with uncommitted changes (`--allow-dirty` packs the commit anyway and names the zip `_dirty`). It scans the zip for paths of the working folders, `*.env` files, keys, Hugging Face tokens, e-mail addresses, paths under `/home/` and the words of a private words file (not in the package: a reader gives `--no-words`); a finding gives exit code 1 (the script accepts listed harmless lines; its header gives the scan's limits). On 2026-11-21 the zip of this package held 1,027 files, about 755 MB unpacked (about 193 MB zipped).

## 8. What a second run can give again

Without a GPU, every number made from the saved attempts comes out the same (section 4).
On one machine, the rollout code sets the random state from the run seed, the pass, the chunk and the turn, and a small GPU test gave the same episodes twice (`tests/gpu/test_rollout.py::test_same_seed_gives_the_same_episodes`). No frozen or training run was repeated: whether they repeat byte for byte is not known.
On two machines, the same curve point (SFT adapter, 80 attempts, run seed 0: `results/main/grpo_seed1_curve_000` against `grpo_seed2_curve_000`) gave identical turns in 78 of 80 attempts and the same final state, verdict and reward in 80 of 80. This one small repeat cannot say whether the 2 differences come from the machine or from noise.

## 9. Where each number comes from

"Step" is `bash scripts/rebuild_results.sh --only <step>`. It writes into a fresh folder and compares.

| Number in WRITEUP.md | File | Step |
|---|---|---|
| Test pass rates (41.3, 91.3, 99.5, 99.5), differences (+50.1, +8.2, +8.1), intervals, by outcome | `results/main/tables/test_base_sft_grpo_seed1_seed2.json` and `.md` | `tables` |
| Test: failed inactive-person attempts after SFT (105); passes added by GRPO (126 and 125) | `results/main/report/failures/test/failure_report.json`, fields `kinds.arms.<arm>.failed_by_outcome` and `n_passed` (added passes: own subtraction) | `failures` |
| Dev pass rates, differences, intervals, zero-variance share | `results/main/tables/base_vs_sft`, `base_sft_grpo_seed1`, `base_sft_grpo_seed2` (`.json`, `.md`) | `tables` |
| Dev rewards of the untrained model (925 at 0, 553 at 100, 58 from 79 to 98) | `results/main/tables/base_vs_sft.json`, `arms[0].reward_distribution` | `tables` |
| 452 SFT examples | `results/main/sft_data/sft_report.json`, field `selection.n_chosen` | none (needs the tokenizer) |
| Passes per family: dv-46 on dev (24, 17 and 3 of 32 for sft, seed 1, seed 2), te-08 and te-31 on test | `results/main/report/failures/failure_report.json` and `report/failures/test/failure_report.json`, field `flips.families` | `failures` |
| Kinds of failure; LoRA targets | `results/main/report/failures/failure_report.json`, fields `kinds` and `lora`, and `report/failures/test/failure_report.json`, field `kinds` | `failures` |
| Kept groups (91 and 95 of 640), per-step numbers | `results/main/report/training/training_report.json`, fields `runs.seed1.totals.kept_groups` and `runs.seed2.totals.kept_groups`, and `steps_seed1.csv`, `steps_seed2.csv` | `training` |
| Learning curves (reward, length, zero-variance share, invalid calls) | `results/main/report/figure/` | `figure` |
| Monitor shares | `results/main/monitors/<arm>_<split>.md` | `monitors` |
| Monitor counts among top-reward GRPO attempts (3,010 dev, 3,049 test; m1 to m3: 1, 0; m4 or m5: 32, 28) | `results/main/audit/monitor_counts/top_reward_flags.md`, rows `grpo_seed1 + grpo_seed2` | none (section 4) |
| Hand audit (the label "end state right" agrees with the verifier in 19 of 31 attempts. All 10 failed attempts have "path acceptable: no") | `results/main/audit/hand_audit_check_run1.txt`, `hand_audit_report.json` (fields `end_state` and `path`. The 12 differences: field `disagreements`. The assistant's reading of them is a working note, not in the package. Not an independent check: WRITEUP.md section 6) | none (see below) |
| Replay; formula audit; positive control; second desk | `results/main/audit/replay/`, `formula/`, `positive_control/`, `second_desk/` | `replay`, `formula`, `positive_control`, `second_desk` |
| 45,696 replayed attempts; 20,480 GRPO training attempts | 22,336 (`checked` in the 28 `audit/replay/<run>.txt`) + 23,360 (the 205 step files of `replay/training_rollouts.txt`: `counts.attempts` of the four `audit/formula/*.json`); 20,480: that field of `formula/grpo_seed1.json` and `grpo_seed2.json` | `replay`, `formula` |
| Six hand-written end states: 6 of 6 match the rules (at first one could not be read) | `hand_states/form.yaml.checks.log` | none (a saved log) |
| Reference rows (oracle and constant policies) | `results/main/reference/` | `reference` |
| Tables counted again by separate code | `results/main/audit/recount/` | none |
| Stretch | `results/stretch.md`, `results/main/report/stretch/` | `stretch` |
| Scenario counts; no leak between the splits | `data/scenarios/summary.json`; `tests/core/test_final_splits.py` | none |
| Cost (estimates) | `results/main/report/training/training_report.json`, field `cost`, from `results/spend_ledger.csv` | `training` |
| Charged by Vast.ai (18.96 USD) | The Vast.ai billing export, sent beside the repository (not in the package); its parts are in NOTES.md section 8 | none |

The hand-audit numbers come from the first check of the filled form (run 1 in `results/main/audit/audit_labels.yaml.checks.log`). Each check adds a run to that log, so check a copy:

```bash
mkdir -p /tmp/audit && cp results/main/audit/audit_labels.yaml /tmp/audit/
.venv/bin/python scripts/audit_sheet.py check --form /tmp/audit/audit_labels.yaml --key results/main/audit/audit_key.json --json /tmp/audit/report.json
cmp /tmp/audit/report.json results/main/audit/hand_audit_report.json   # prints nothing when the two are equal
```

The step `audit_sheet` of the rebuild compares only the SHA-256 of the sheet and the key.
