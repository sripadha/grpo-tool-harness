"""scripts/gpu/sft_data.py, train_sft.py and train_grpo.py, without torch and without a model.

What is tested here, and what is not:
    the argument parsers        their defaults are the values of PLAN.md (typed here by hand), and every
                                setting of SFTConfig and GRPOConfig can be given on the command line
    sft_data.py end to end      on episodes of scripted policies with hand-made token rows; the numbers of
                                its report are computed by hand in the comments; every refusal
    train_sft.py, train_grpo.py their main functions with the GPU part replaced by a stand-in: the checks
                                before the model loads, what is passed on to the library, the run manifest,
                                the exit codes, the restart. No training runs here. The training code is
                                tested on the rented machine (tests/gpu).
The two chain scripts that start these scripts (scripts/gpu/chain_stage1.sh, chain_stage2.sh) are tested in
tests/core/test_chain_scripts.py.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from accessdesk.config import RunConfig
from accessdesk.env.scenarios import save_families, scenario_ids
from accessdesk.harness.episode import Episode
from accessdesk.harness.hf_backend import PINNED_REVISIONS
from accessdesk.harness.policies import constant, oracle, run_policy, stumbling_oracle
from accessdesk.harness.records import read_records, write_records
from accessdesk.train.grpo import GRPOConfig, step_scenarios
from accessdesk.train.sft import SFTConfig
from accessdesk.types import OUTCOMES

ROOT = Path(__file__).resolve().parents[2]
GPU_SCRIPTS = ROOT / "scripts" / "gpu"
PLACEHOLDER_SCENARIOS = ROOT / "data" / "scenarios_placeholder"
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"          # the 2B, typed by hand from the task
# The hand-made token rows: every episode has 5 prompt tokens, and each turn has 3 sampled tokens and, but
# for the last turn, 2 tokens of tool result.
PROMPT_TOKENS, TURN_TOKENS, SUFFIX_TOKENS = 5, 3, 2
SAMPLES = 3
# A constant policy that is wrong for the outcome: its episodes fail.
WRONG_RESOLUTION = {"grant": "denied", "approval": "denied", "already": "denied", "inactive": "already"}
# What `environment_report` of the library returns on the rented machine (results/_smoke/dryrun_2b).
ENVIRONMENT = {"python": "3.11.17",
               "packages": {"torch": "2.15.0", "transformers": "5.23.0", "peft": "0.21.3"},
               "cuda_version": "13.2", "gpu_name": "NVIDIA GeForce RTX 4090", "gpu_memory_gb": 23.6}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")              # PLAN section 2, typed by hand


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def every_training_file() -> set[str]:
    """What the run manifest of a training run must name under `training_code`: every file of
    accessdesk/train/ as the folder is now, and, typed by hand, the two training parts of
    accessdesk/_second/, the three training scripts and the two chain scripts."""
    library = {path.relative_to(ROOT).as_posix() for path in (ROOT / "accessdesk" / "train").rglob("*")
               if path.is_file() and "__pycache__" not in path.parts}
    assert {"accessdesk/train/sft.py", "accessdesk/train/grpo.py", "accessdesk/train/forward.py",
            "accessdesk/train/lora.py"} <= library
    return library | {"accessdesk/_second/grpo_math.py", "accessdesk/_second/loss_mask.py",
                      "scripts/gpu/sft_data.py", "scripts/gpu/train_sft.py", "scripts/gpu/train_grpo.py",
                      "scripts/gpu/chain_stage1.sh", "scripts/gpu/chain_stage2.sh"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def scripts() -> SimpleNamespace:
    """The three Python scripts as modules. They are imported by name from scripts/gpu, as they import each
    other, so that there is one module object of each."""
    if str(GPU_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GPU_SCRIPTS))
    return SimpleNamespace(sft_data=importlib.import_module("sft_data"),
                           train_sft=importlib.import_module("train_sft"),
                           train_grpo=importlib.import_module("train_grpo"))


@pytest.fixture(autouse=True)
def smoke_dir_is_the_test_folder(scripts, monkeypatch, tmp_path):
    """The tests use the placeholder text, and with it the scripts write under results/_smoke/ only. For
    the tests that folder is the test's own temporary folder."""
    monkeypatch.setattr(scripts.sft_data.rollout_script, "SMOKE_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Hand-made inputs: a scenario folder, and a rollout of scripted policies on its train split
# ---------------------------------------------------------------------------
def token_row(record: dict) -> dict:
    """A hand-made line of tokens.jsonl for an episode record (the layout is in the comment at the top)."""
    ids = list(range(1, PROMPT_TOKENS + 1))
    segments = [["prompt", 0, PROMPT_TOKENS, -1]]
    for turn in range(record["n_turns"]):
        segments.append(["sampled", len(ids), len(ids) + TURN_TOKENS, turn])
        ids += [100 + turn] * TURN_TOKENS
        if turn + 1 < record["n_turns"]:
            segments.append(["suffix", len(ids), len(ids) + SUFFIX_TOKENS, turn])
            ids += [200 + turn] * SUFFIX_TOKENS
    return {"scenario_id": record["scenario_id"], "sample": record["sample"], "seed": record["seed"],
            "harness_error": record["harness_error"], "ids": ids, "segments": segments,
            "logprobs": [-0.5] * (TURN_TOKENS * record["n_turns"])}


def write_rollout(folder: Path, records: list[dict], cfg: RunConfig, text, data_sha256: str, *,
                  split: str = "train") -> None:
    """episodes.jsonl, tokens.jsonl and run_manifest.json, as scripts/gpu/rollout.py leaves them."""
    folder.mkdir(parents=True, exist_ok=True)
    write_records(folder / "episodes.jsonl", records)
    write_records(folder / "tokens.jsonl", [token_row(record) for record in records])
    n_scenarios = len({record["scenario_id"] for record in records})
    write_json(folder / "run_manifest.json", {
        "complete": True, "config": cfg.to_dict(), "config_sha256": cfg.sha256(), "model": MODEL,
        "revision": REVISION, "adapter": None, "split": split, "samples": len(records) // n_scenarios,
        "run_seed": 0, "families": len({record["family_id"] for record in records}), "scenarios": n_scenarios,
        "text_sha256": text.sha256, "split_file_sha256": data_sha256})


@pytest.fixture(scope="module")
def template(tmp_path_factory, text, cfg, families) -> Path:
    """A folder that every test copies:

        scenarios/     train.jsonl with the 200 train families, and a summary.json with the keys that the
                       scripts read (text_sha256, text_is_placeholder, config_sha256, files_sha256)
        base_train/    a rollout of scripted policies on 8 of those families (2 per outcome), in their 3
                       wordings, 3 attempts each: 72 episodes

    The attempts, by sample number (0, 1, 2):
        the first family of each outcome                  fail, clean pass, clean pass (the same calls)
        the second family of "grant" and of "approval"    pass with one malformed message (reward 95),
                                                          fail, fail
        the second family of "already" and of "inactive"  fail, fail, fail
    """
    root = tmp_path_factory.mktemp("train_scripts")
    scenarios = root / "scenarios"
    scenarios.mkdir()
    save_families(scenarios / "train.jsonl", families["train"])
    data_sha256 = sha256_of(scenarios / "train.jsonl")
    write_json(scenarios / "summary.json", {
        "text_file": Path(text.path).name, "text_sha256": text.sha256,
        "text_is_placeholder": text.is_placeholder, "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "files_sha256": {"train.jsonl": data_sha256}})

    plans: list[tuple[dict, list]] = []
    for outcome in OUTCOMES:
        first, second = [family for family in families["train"] if family["outcome"] == outcome][:2]
        fail = constant(text, WRONG_RESOLUTION[outcome])
        plans.append((first, [fail, oracle(text), oracle(text)]))
        stumble = stumbling_oracle(text, 1) if outcome in ("grant", "approval") else fail
        plans.append((second, [stumble, fail, fail]))
    records = []
    for sample in range(SAMPLES):                    # pass by pass, as rollout.py writes them
        for family, policies in plans:
            for wording in family["wordings"]:
                episode = run_policy(Episode(family, wording["style"], text, cfg), policies[sample])
                seed = {"run_seed": 0, "pass": sample, "chunk": 0, "pos": len(records)}
                records.append(episode.record(run="base_train", arm="base_train", sample=sample, seed=seed,
                                              data_sha256=data_sha256))
    # The scripted episodes are what the docstring says they are.
    by_reward: dict[int, int] = {}
    for record in records:
        by_reward[record["reward"]] = by_reward.get(record["reward"], 0) + 1
    assert by_reward == {100: 24, 95: 6, 0: 42}
    write_rollout(root / "base_train", records, cfg, text, data_sha256)
    return root


@pytest.fixture
def work(template, tmp_path) -> Path:
    """A copy of the template that the test may change."""
    folder = tmp_path / "work"
    shutil.copytree(template, folder)
    return folder


def sft_data_argv(work: Path, *more: str) -> list[str]:
    return ["--episodes", str(work / "base_train" / "episodes.jsonl"),
            "--tokens", str(work / "base_train" / "tokens.jsonl"),
            "--scenarios", str(work / "scenarios"), "--out", str(work / "sft_data"), *more]


# ---------------------------------------------------------------------------
# The parsers: the defaults are the plan's values, and every setting can be given
# ---------------------------------------------------------------------------
def test_importing_the_scripts_loads_no_torch():
    code = ("import sys; sys.path.insert(0, 'scripts/gpu'); import sft_data, train_sft, train_grpo; "
            "bad = [name for name in ('torch', 'peft') if name in sys.modules]; sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True).returncode == 0


def test_the_defaults_of_train_sft_are_the_values_of_the_plan(scripts):
    args = scripts.train_sft.parse_args(["--model", MODEL, "--data", "d", "--out", "o", "--seed", "7"])
    config = scripts.train_sft.sft_config(args)
    # PLAN 5.5: learning rate 1e-4, 2 epochs, micro-batches of 1 or 2 episodes, gradient checkpointing.
    assert (config.lr, config.epochs, config.micro_batch, config.gradient_checkpointing) == (1e-4, 2, 2, True)
    # _dev/GPU_SPEC.md: an effective batch of 8 episodes. The clip value 1.0 is the plan's for GRPO (5.6);
    # for SFT it is the library's default.
    assert (config.batch_episodes, config.max_grad_norm) == (8, 1.0)
    assert config.seed == 7 and config == SFTConfig(seed=7)
    # PLAN section 2: rank 16, alpha 32, the ten projections; the two gate projections are left out.
    assert (args.rank, args.alpha, args.revision, args.device) == (16, 32, None, "cuda")
    assert scripts.train_sft.lora_targets(args) == LORA_TARGETS
    with_gates = scripts.train_sft.parse_args(["--model", MODEL, "--data", "d", "--out", "o", "--seed", "7",
                                               "--gate-projections"])
    assert scripts.train_sft.lora_targets(with_gates) == LORA_TARGETS + ("in_proj_a", "in_proj_b")


def test_every_setting_of_the_sft_config_can_be_given(scripts):
    args = scripts.train_sft.parse_args([
        "--model", MODEL, "--data", "d", "--out", "o", "--seed", "3", "--epochs", "1", "--lr", "3e-5",
        "--batch-episodes", "4", "--micro-batch", "1", "--max-grad-norm", "0.5",
        "--no-gradient-checkpointing"])
    wanted = SFTConfig(lr=3e-5, epochs=1, batch_episodes=4, micro_batch=1, max_grad_norm=0.5,
                       gradient_checkpointing=False, seed=3)
    assert scripts.train_sft.sft_config(args) == wanted
    assert all(getattr(wanted, field.name) != getattr(SFTConfig(), field.name)
               for field in dataclasses.fields(SFTConfig)), "the test must change every field"


GRPO_REQUIRED = ["--model", MODEL, "--adapter", "a", "--scenarios", "s", "--out", "o", "--seed", "5"]


def test_the_defaults_of_train_grpo_are_the_values_of_the_plan(scripts):
    args = scripts.train_grpo.parse_args(GRPO_REQUIRED)
    config = scripts.train_grpo.grpo_config(args, 600)
    # PLAN 5.6 and section 2: 80 steps of 16 scenarios with 8 samples each; learning rate 1e-5; gradient
    # norm clipped at 1.0; only exact ties are dropped.
    assert (config.steps, config.n_scenarios, config.group_size) == (80, 16, 8)
    assert (config.lr, config.max_grad_norm, config.tie_tol) == (1e-5, 1.0, 0.0)
    # The stop rule of PLAN 5.6: harness errors above 1%; mean log-probability gap above 0.02; invalid-call
    # rate above 5% (and twice its step-0 value: that part is in the library).
    assert (config.max_harness_error_rate, config.max_logprob_gap, config.max_invalid_call_rate) == \
        (0.01, 0.02, 0.05)
    # The constant of the loss does not depend on the batch: 16 * 8 * 160.
    assert config.norm_const == 20480.0
    assert config.seed == 5 and config == GRPOConfig(seed=5)
    assert scripts.train_grpo.run_mode(args) == "seed"
    # The three training styles; style 4 is the held-out one.
    assert (args.styles, args.resume, args.pilot, args.overfit, args.save_tokens) == \
        ("1,2,3", False, False, None, False)
    assert (args.max_new_tokens, args.chunk_size) == (None, None)        # None: the value of the config


def test_every_setting_of_the_grpo_config_can_be_given(scripts):
    args = scripts.train_grpo.parse_args([
        *GRPO_REQUIRED, "--steps", "40", "--scenarios-per-step", "8", "--group-size", "16", "--lr", "3e-5",
        "--max-grad-norm", "0.5", "--micro-batch", "2", "--reference-tokens", "250", "--tie-tol", "0.01",
        "--save-every", "5", "--no-gradient-checkpointing", "--max-harness-error-rate", "0.25",
        "--max-logprob-gap", "0.05", "--max-invalid-call-rate", "0.1", "--usd-per-hour", "0.4"])
    wanted = GRPOConfig(n_scenarios=8, group_size=16, reference_tokens=250, steps=40, lr=3e-5,
                        max_grad_norm=0.5, micro_batch=2, gradient_checkpointing=False, tie_tol=0.01,
                        save_every=5, seed=5, max_harness_error_rate=0.25, max_logprob_gap=0.05,
                        max_invalid_call_rate=0.1, usd_per_hour=0.4)
    assert scripts.train_grpo.grpo_config(args, 600) == wanted
    assert all(getattr(wanted, field.name) != getattr(GRPOConfig(), field.name)
               for field in dataclasses.fields(GRPOConfig)), "the test must change every field"


def test_the_pilot_is_15_steps_at_the_real_settings(scripts):
    args = scripts.train_grpo.parse_args([*GRPO_REQUIRED, "--pilot"])
    assert scripts.train_grpo.run_mode(args) == "pilot"
    assert scripts.train_grpo.grpo_config(args, 600) == GRPOConfig(seed=5, steps=15)       # PLAN 5.6
    for more in (["--steps", "20"], ["--overfit", "ids.txt"]):
        args = scripts.train_grpo.parse_args([*GRPO_REQUIRED, "--pilot", *more])
        with pytest.raises(scripts.sft_data.Refused):
            scripts.train_grpo.grpo_config(args, 600)


def test_the_overfit_check_uses_all_its_scenarios_in_every_step(scripts):
    args = scripts.train_grpo.parse_args([*GRPO_REQUIRED, "--overfit", "ids.txt"])
    assert scripts.train_grpo.run_mode(args) == "overfit"
    # PLAN 5.6: two scenarios overfit "in about 30 steps".
    assert scripts.train_grpo.grpo_config(args, 2) == GRPOConfig(seed=5, steps=30, n_scenarios=2)
    assert scripts.train_grpo.grpo_config(args, 40).n_scenarios == 16
    given = scripts.train_grpo.parse_args([*GRPO_REQUIRED, "--overfit", "ids.txt", "--steps", "12",
                                           "--scenarios-per-step", "3"])
    assert scripts.train_grpo.grpo_config(given, 6) == GRPOConfig(seed=5, steps=12, n_scenarios=3)


def test_sft_data_needs_its_four_paths(scripts, capsys):
    args = scripts.sft_data.parse_args(["--episodes", "e", "--tokens", "t", "--scenarios", "s", "--out", "o"])
    assert args.allow_penalised is False
    with pytest.raises(SystemExit):
        scripts.sft_data.parse_args(["--episodes", "e", "--tokens", "t", "--out", "o"])
    capsys.readouterr()


# ---------------------------------------------------------------------------
# sft_data.py end to end
# ---------------------------------------------------------------------------
def test_sft_data_keeps_one_clean_pass_per_scenario(scripts, work, cfg, text, capsys):
    assert scripts.sft_data.main(sft_data_argv(work)) == 0
    capsys.readouterr()
    out = work / "sft_data"
    assert sorted(path.name for path in out.iterdir()) == ["run_manifest.json", "sft_records.jsonl",
                                                           "sft_report.json", "sft_report.md"]
    # The chosen records: the first family of each outcome, its 3 wordings, the clean pass with the lowest
    # sample number (1; sample 0 failed and sample 2 made the same calls). 4 * 3 = 12, in scenario order.
    chosen = read_records(out / "sft_records.jsonl")
    assert len(chosen) == 12
    assert [r["scenario_id"] for r in chosen] == sorted(r["scenario_id"] for r in chosen)
    assert all(r["sample"] == 1 and r["passed"] and r["reward"] == 100 for r in chosen)
    assert {r["split"] for r in chosen} == {"train"}
    assert sorted({r["outcome"] for r in chosen}) == sorted(OUTCOMES)
    # Each one has the token row of its own line of tokens.jsonl.
    rows = {(row["scenario_id"], row["sample"]): row
            for row in read_records(work / "base_train" / "tokens.jsonl")}
    for record in chosen:
        row = rows[(record["scenario_id"], record["sample"])]
        assert record["tokens"] == {key: row[key] for key in ("ids", "segments", "logprobs")}

    report = read_json(out / "sft_report.json")
    selection = report["selection"]
    # 72 records. Passes: 4 families * 3 wordings * 2 clean + 2 families * 3 wordings * 1 with a penalty.
    assert (selection["n_records"], selection["n_harness_errors"], selection["n_passed"]) == (72, 0, 30)
    assert (selection["n_clean_passes"], selection["n_chosen"]) == (24, 12)
    assert selection["n_chosen_with_penalty"] == 0
    assert (selection["n_scenarios"], selection["n_families"], selection["n_families_covered"]) == (24, 8, 4)
    assert selection["by_outcome"] == {outcome: {"families": 2, "families_covered": 1, "scenarios_chosen": 3}
                                       for outcome in OUTCOMES}
    # Pass rate of the input, by family: the first family passes 2 of 3 attempts, the second 1 of 3 (grant,
    # approval) or none (already, inactive). The mean over the two families of an outcome: 1/2 or 1/3.
    assert selection["pass_rate_by_outcome"] == pytest.approx({"grant": 0.5, "approval": 0.5,
                                                               "already": 1 / 3, "inactive": 1 / 3})
    assert report["pass_rate"] == pytest.approx(10 / 24)        # (4 * 2/3 + 2 * 1/3 + 2 * 0) / 8 families
    assert report["attempts_per_scenario"] == {"3": 24}
    # Turns of the correct solution: 4 (grant, approval) and 3 (already, inactive). Tokens in the loss:
    # 3 per turn, so (4 + 4 + 3 + 3) * 3 tokens * 3 wordings = 126. All tokens of an episode with n turns:
    # 5 + 3n + 2(n - 1), so 23 or 18, and (23 + 23 + 18 + 18) * 3 = 246.
    assert (report["n_loss_tokens"], report["n_chosen_tokens"]) == (126, 246)

    table = (out / "sft_report.md").read_text(encoding="utf-8")
    assert ("| Outcome | Families in the input | Families covered | Scenarios chosen | "
            "Pass rate of the input |") in table.splitlines()
    for line in ("| grant | 2 | 1 | 3 | 50.0% |", "| approval | 2 | 1 | 3 | 50.0% |",
                 "| already | 2 | 1 | 3 | 33.3% |", "| inactive | 2 | 1 | 3 | 33.3% |",
                 "| all | 8 | 4 | 12 | 41.7% |", "- n_clean_passes: 24"):
        assert line in table.splitlines(), line


def test_the_manifest_of_sft_data_names_every_file_by_its_hash(scripts, work, cfg, text, capsys):
    argv = sft_data_argv(work)
    assert scripts.sft_data.main(argv) == 0
    capsys.readouterr()
    out = work / "sft_data"
    manifest = read_json(out / "run_manifest.json")
    assert manifest["complete"] is True and manifest["script"] == "scripts/gpu/sft_data.py"
    assert manifest["command"] == ["scripts/gpu/sft_data.py", *argv]
    assert manifest["allow_penalised"] is False and manifest["split"] == "train"
    # Every input and every output file, with the hash of the file as it is on disk.
    inputs = {"episodes": work / "base_train" / "episodes.jsonl",
              "tokens": work / "base_train" / "tokens.jsonl",
              "summary": work / "scenarios" / "summary.json",
              "train_file": work / "scenarios" / "train.jsonl"}
    assert {key: entry["sha256"] for key, entry in manifest["inputs"].items()} == \
        {key: sha256_of(path) for key, path in inputs.items()}
    assert {name: entry["sha256"] for name, entry in manifest["outputs"].items()} == \
        {name: sha256_of(out / name) for name in ("sft_records.jsonl", "sft_report.json", "sft_report.md")}
    # The config is the one of the rollout (its run manifest lies beside the episodes), with its hash.
    assert manifest["config"] == cfg.to_dict() and manifest["config_sha256"] == cfg.sha256()
    assert manifest["config_source"] == "run_manifest.json of the rollout"
    assert manifest["records_config_sha256"] == cfg.sha256()
    assert manifest["records_settings"] == cfg.compared_settings()
    assert manifest["text_sha256"] == text.sha256
    assert manifest["data_sha256"] == sha256_of(inputs["train_file"])
    source = manifest["source_run"]
    assert (source["model"], source["revision"]) == (MODEL, REVISION)
    assert (source["adapter"], source["samples"], source["scenarios"]) == (None, 3, 24)
    assert source["manifest"]["sha256"] == sha256_of(work / "base_train" / "run_manifest.json")
    assert manifest["counts"] == {"records": 72, "chosen": 12, "loss_tokens": 126}
    assert {"code_sha256", "frozen", "frozen_note"} <= set(manifest)
    assert manifest["started_utc"].endswith("Z") and manifest["ended_utc"].endswith("Z")
    # The training code is named file by file; the one hash over it follows the rule of sha256sum.
    code = manifest["training_code"]
    assert set(code["files"]) == every_training_file()
    assert all(digest == sha256_of(ROOT / path) for path, digest in code["files"].items())
    lines = "".join(f"{digest}  {path}\n" for path, digest in code["files"].items())
    assert list(code["files"]) == sorted(code["files"])
    assert code["sha256"] == hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_sft_data_takes_passes_with_a_penalty_only_when_asked(scripts, work, capsys):
    assert scripts.sft_data.main(sft_data_argv(work, "--allow-penalised")) == 0
    capsys.readouterr()
    out = work / "sft_data"
    chosen = read_records(out / "sft_records.jsonl")
    # 12 clean passes, and the 2 * 3 scenarios whose only pass has one malformed message (reward 95).
    assert len(chosen) == 18 and sorted({r["reward"] for r in chosen}) == [95, 100]
    report = read_json(out / "sft_report.json")
    selection = report["selection"]
    assert (selection["n_chosen"], selection["n_chosen_with_penalty"], selection["n_families_covered"]) == \
        (18, 6, 6)
    assert selection["by_outcome"]["grant"] == {"families": 2, "families_covered": 2, "scenarios_chosen": 6}
    assert selection["by_outcome"]["already"] == {"families": 2, "families_covered": 1, "scenarios_chosen": 3}
    # A penalised pass has 5 turns: the malformed one and the 4 of the solution. Its malformed turn is left
    # out of the loss: 4 * 3 = 12 tokens each. 126 + 6 * 12 = 198.
    assert report["n_loss_tokens"] == 198
    assert read_json(out / "run_manifest.json")["allow_penalised"] is True
    assert "| grant | 2 | 2 | 6 | 50.0% |" in (out / "sft_report.md").read_text(encoding="utf-8").splitlines()


def test_sft_data_never_takes_an_attempt_with_a_refused_call(scripts, work, capsys):
    """The rule is the library's (`select_sft_records`); the script must pass it through and report it."""
    episodes = work / "base_train" / "episodes.jsonl"
    marked = next(r for r in read_records(episodes) if r["reward"] == 100 and r["sample"] == 1)

    def one_refused_call(records: list[dict]) -> None:
        for record in records:
            if (record["scenario_id"], record["sample"]) == (marked["scenario_id"], 1):
                record["n_tool_errors"] = 1

    rewrite(episodes, one_refused_call)
    assert scripts.sft_data.main(sft_data_argv(work)) == 0
    capsys.readouterr()
    chosen = {r["scenario_id"]: r["sample"] for r in read_records(work / "sft_data" / "sft_records.jsonl")}
    # The scenario is still covered, by its other clean pass (sample 2); the 11 others keep sample 1.
    assert len(chosen) == 12 and chosen[marked["scenario_id"]] == 2
    assert sorted(chosen.values()) == [1] * 11 + [2]
    assert read_json(work / "sft_data" / "sft_report.json")["selection"]["n_with_refused_call"] == 1
    table = (work / "sft_data" / "sft_report.md").read_text(encoding="utf-8")
    assert "- n_with_refused_call: 1" in table.splitlines()


def test_sft_data_without_the_rollouts_manifest_uses_the_default_config_and_says_so(scripts, work, capsys):
    (work / "base_train" / "run_manifest.json").unlink()
    assert scripts.sft_data.main(sft_data_argv(work)) == 0
    capsys.readouterr()
    manifest = read_json(work / "sft_data" / "run_manifest.json")
    assert manifest["source_run"] is None and manifest["config_source"].startswith("the default config")
    assert manifest["config_sha256"] == RunConfig().sha256()
    assert manifest["counts"]["chosen"] == 12


def refused(scripts, capsys, argv: list[str], out: Path) -> str:
    """Run sft_data.py; it must refuse (exit code 2) and write nothing. Returns its message."""
    assert scripts.sft_data.main(argv) == 2
    assert not out.exists()
    return capsys.readouterr().err


def rewrite(path: Path, change) -> None:
    """Apply `change` to the list of records of a JSON-lines file and write it back."""
    records = read_records(path)
    write_records(path, change(records) or records)


def test_sft_data_refuses_an_output_folder_that_holds_data(scripts, work, capsys):
    assert scripts.sft_data.main(sft_data_argv(work)) == 0
    before = {path.name: sha256_of(path) for path in (work / "sft_data").iterdir()}
    assert scripts.sft_data.main(sft_data_argv(work)) == 2
    assert "already holds" in capsys.readouterr().err
    assert {path.name: sha256_of(path) for path in (work / "sft_data").iterdir()} == before
    # The folder of the rollout is refused too: the two manifests have the same file name.
    assert scripts.sft_data.main([*sft_data_argv(work)[:6], "--out", str(work / "base_train")]) == 2
    assert "already holds run_manifest.json" in capsys.readouterr().err


def test_sft_data_refuses_records_of_another_split(scripts, work, cfg, text, families, capsys):
    """Nothing from dev may become training data. The library refuses it, and the script passes that on."""
    dev_file = work / "scenarios" / "dev.jsonl"
    save_families(dev_file, families["dev"][:2])
    records = []
    for family in families["dev"][:2]:
        episode = run_policy(Episode(family, 1, text, cfg), oracle(text))
        seed = {"run_seed": 0, "pass": 0, "chunk": 0, "pos": len(records)}
        records.append(episode.record(run="dev", arm="base", sample=0, seed=seed,
                                      data_sha256=sha256_of(dev_file)))
    write_rollout(work / "base_dev", records, cfg, text, sha256_of(dev_file), split="dev")
    argv = ["--episodes", str(work / "base_dev" / "episodes.jsonl"),
            "--tokens", str(work / "base_dev" / "tokens.jsonl"),
            "--scenarios", str(work / "scenarios"), "--out", str(work / "sft_data")]
    assert "train split only" in refused(scripts, capsys, argv, work / "sft_data")
    # One dev record among the train records is enough.
    rewrite(work / "base_train" / "episodes.jsonl", lambda rs: rs[0].update(split="dev"))
    assert "train split only" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")


def test_sft_data_refuses_a_scenario_file_that_was_changed_after_the_build(scripts, work, capsys):
    train_file = work / "scenarios" / "train.jsonl"
    train_file.write_bytes(train_file.read_bytes() + b"\n")
    message = refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    assert "not the file that the build wrote" in message
    (work / "scenarios" / "summary.json").unlink()
    assert "summary.json is missing" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")


def test_sft_data_refuses_records_of_another_scenario_file(scripts, work, families, capsys):
    """Another build has the same family ids and other worlds: the folder is whole, but it is not the one
    that the rollout read."""
    train_file = work / "scenarios" / "train.jsonl"
    save_families(train_file, list(reversed(families["train"])))
    summary = read_json(work / "scenarios" / "summary.json")
    summary["files_sha256"]["train.jsonl"] = sha256_of(train_file)
    write_json(work / "scenarios" / "summary.json", summary)
    message = refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    assert "72 of 72 records name another scenario file" in message


def test_sft_data_refuses_a_record_that_names_no_scenario_file(scripts, work, capsys):
    """A record without the hash cannot be checked, so it is refused as well."""
    rewrite(work / "base_train" / "episodes.jsonl", lambda rs: rs[5].pop("data_sha256") and None)
    assert "1 of 72 records name another scenario file" in refused(scripts, capsys, sft_data_argv(work),
                                                                  work / "sft_data")


def test_sft_data_refuses_records_that_were_made_with_another_text_file(scripts, work, capsys):
    summary = read_json(work / "scenarios" / "summary.json")
    summary["text_sha256"] = "0" * 64
    write_json(work / "scenarios" / "summary.json", summary)
    assert "another text file" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")


def test_sft_data_refuses_token_rows_that_do_not_belong_to_the_episodes(scripts, work, capsys):
    tokens = work / "base_train" / "tokens.jsonl"
    original = tokens.read_bytes()
    rewrite(tokens, lambda rows: rows[:-1])                                  # one line is missing
    assert "do not belong together" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    tokens.write_bytes(original)
    rewrite(tokens, lambda rows: [rows[1], rows[0], *rows[2:]])              # two lines changed places
    assert "do not belong together" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    tokens.write_bytes(original[:-40])                                       # the last line is cut off
    assert "not a file of JSON lines" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    tokens.unlink()
    assert "is missing" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")


def test_sft_data_refuses_a_rollout_that_is_not_complete(scripts, work, capsys):
    manifest_file = work / "base_train" / "run_manifest.json"
    manifest = read_json(manifest_file)
    write_json(manifest_file, {**manifest, "complete": False})
    assert "not complete" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    write_json(manifest_file, {**manifest, "samples": 8})                    # 24 scenarios * 8 is not 72
    assert "holds 72 records" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    write_json(manifest_file, {**manifest, "config": {**manifest["config"], "max_new_tokens": 999}})
    assert "does not have the hash" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")


def test_sft_data_refuses_records_that_were_scored_with_other_reward_numbers(scripts, work, capsys):
    """A "clean pass" is a reward equal to the config's pass reward. Records of a run with another reward
    scale would be read wrongly."""
    (work / "base_train" / "run_manifest.json").unlink()
    episodes = work / "base_train" / "episodes.jsonl"
    original = episodes.read_bytes()

    def other_reward(records: list[dict]) -> None:
        for record in records:
            record["settings"] = {**record["settings"], "malformed_penalty": 1}

    rewrite(episodes, other_reward)
    assert "other reward numbers" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    episodes.write_bytes(original)
    rewrite(episodes, lambda rs: rs[0].update(config_sha256="0" * 64))       # two runs in one file
    assert "not all made with one config" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")
    episodes.write_bytes(original)
    rewrite(episodes, lambda rs: rs[0].pop("settings") and None)
    assert "carries no `settings`" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")


def test_sft_data_with_no_pass_at_all_writes_nothing_and_ends_with_code_1(scripts, work, capsys):
    def all_fail(records: list[dict]) -> None:
        for record in records:
            record.update(passed=False, reward=0)

    rewrite(work / "base_train" / "episodes.jsonl", all_fail)
    assert scripts.sft_data.main(sft_data_argv(work)) == 1
    captured = capsys.readouterr()
    assert "no episode was chosen" in captured.err and '"n_passed": 0' in captured.out
    assert not (work / "sft_data").exists()


def test_data_of_the_placeholder_text_is_written_under_the_smoke_folder_only(scripts, work, monkeypatch,
                                                                             capsys):
    monkeypatch.setattr(scripts.sft_data.rollout_script, "SMOKE_DIR", work / "elsewhere")
    assert "placeholder text" in refused(scripts, capsys, sft_data_argv(work), work / "sft_data")


def test_the_real_scenario_folders_have_the_keys_that_sft_data_reads(scripts):
    """The hand-made summary.json of these tests has the layout of a folder that the build wrote."""
    summary = read_json(PLACEHOLDER_SCENARIOS / "summary.json")
    assert {"text_sha256", "text_is_placeholder", "config_sha256", "files_sha256"} <= set(summary)
    assert summary["files_sha256"]["train.jsonl"] == sha256_of(PLACEHOLDER_SCENARIOS / "train.jsonl")
    folder = scripts.sft_data.check_scenario_folder(PLACEHOLDER_SCENARIOS)
    assert len(folder["scenario_ids"]) == 600                   # 200 families in the 3 training styles


# ---------------------------------------------------------------------------
# train_sft.py: main() with a stand-in for the GPU part
# ---------------------------------------------------------------------------
def sft_stand_ins(scripts, monkeypatch, *, change: dict | None = None,
                  error: Exception | None = None) -> dict:
    """Replace the two GPU functions of train_sft.py. The stand-in for the training writes the files that
    the library writes (sft_log.jsonl and the adapter). Returns a dict that collects what they were given."""
    seen: dict = {}

    def setup_model(args, cfg, targets):
        seen["setup"] = {"model": args.model, "revision": args.revision, "cfg": cfg, "targets": targets}
        lora = {"rank": args.rank, "alpha": args.alpha, "dropout": 0.0, "targets": list(targets),
                "seed": args.seed, "wrapped_modules": 200, "trainable_parameters": 1, "coverage": []}
        return {"model": "the model", "tokenizer": None, "pad_id": 0, "initial_lora": {},
                "coverage_table": [{"path": "model.layers.0.mlp.up_proj", "wrapped": True}], "lora": lora,
                "environment": ENVIRONMENT, "kernels": {"fast_path": True},
                "sampler": {"max_new_tokens": cfg.max_new_tokens}}

    def run_training(setup, examples, sft_cfg, out_dir):
        seen["training"] = {"examples": examples, "sft_cfg": sft_cfg, "pad_id": setup["pad_id"]}
        if error is not None:
            raise error
        logs = [{"step": step, "epoch": step // 2, "loss": 1.0 / (step + 1), "n_tokens": 10 * (step + 1),
                 "n_episodes": 8, "grad_norm": 2.0 - step, "lr": sft_cfg.lr, "seconds": 2.5}
                for step in range(4)]
        write_records(out_dir / "sft_log.jsonl", logs)
        (out_dir / "adapter").mkdir()
        (out_dir / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (out_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
        return logs, change or {"by_block": {"linear_attn": 0.5, "mlp": 0.7, "self_attn": 0.2}, "total": 0.9}

    monkeypatch.setattr(scripts.train_sft, "setup_model", setup_model)
    monkeypatch.setattr(scripts.train_sft, "run_training", run_training)
    monkeypatch.setattr(scripts.train_sft, "peak_gpu_memory_gb", lambda: {"allocated": 1.0, "reserved": 2.0})
    return seen


@pytest.fixture
def sft_data_done(scripts, work, capsys) -> Path:
    """The work folder after sft_data.py has run."""
    assert scripts.sft_data.main(sft_data_argv(work)) == 0
    capsys.readouterr()
    return work


def train_sft_argv(work: Path, *more: str) -> list[str]:
    return ["--model", MODEL, "--data", str(work / "sft_data"), "--out", str(work / "sft"), "--seed", "0",
            *more]


def test_train_sft_gives_the_library_the_chosen_episodes_and_writes_the_manifest(scripts, sft_data_done,
                                                                                 monkeypatch, cfg, capsys):
    work = sft_data_done
    seen = sft_stand_ins(scripts, monkeypatch)
    argv = train_sft_argv(work)
    assert scripts.train_sft.main(argv) == 0
    capsys.readouterr()
    # The model is set up with the config of the rollout and with the plan's ten projections.
    assert seen["setup"] == {"model": MODEL, "revision": None, "cfg": cfg, "targets": LORA_TARGETS}
    # The training gets one example per chosen episode: the stored ids, and weight 1 on the 3 sampled tokens
    # of every turn only (126 in all, the number of sft_data.py's report).
    examples = seen["training"]["examples"]
    records = read_records(work / "sft_data" / "sft_records.jsonl")
    assert [example["ids"] for example in examples] == [record["tokens"]["ids"] for record in records]
    assert sum(sum(example["weights"]) for example in examples) == 126
    first = examples[0]["weights"]
    assert first[:PROMPT_TOKENS + TURN_TOKENS + SUFFIX_TOKENS] == [0] * 5 + [1] * 3 + [0] * 2
    assert seen["training"]["sft_cfg"] == SFTConfig(seed=0)

    out = work / "sft"
    manifest = read_json(out / "run_manifest.json")
    assert manifest["complete"] is True and manifest["status"] == "complete" and manifest["error"] is None
    assert manifest["script"] == "scripts/gpu/train_sft.py"
    assert manifest["command"] == ["scripts/gpu/train_sft.py", *argv]
    assert (manifest["model"], manifest["revision"], manifest["seed"]) == (MODEL, REVISION, 0)
    assert REVISION == PINNED_REVISIONS[MODEL]
    # Every setting of the two configs, each with its hash.
    assert manifest["sft_config"] == SFTConfig(seed=0).to_dict()
    assert manifest["sft_config_sha256"] == hashlib.sha256(
        json.dumps(SFTConfig(seed=0).to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    assert manifest["config"] == cfg.to_dict() and manifest["config_sha256"] == cfg.sha256()
    assert manifest["lora"]["rank"] == 16 and manifest["lora"]["alpha"] == 32
    assert manifest["lora"]["targets"] == list(LORA_TARGETS) and manifest["lora"]["seed"] == 0
    # The data files by their hashes, and where the data came from.
    data = manifest["data"]
    assert data["records"]["sha256"] == sha256_of(work / "sft_data" / "sft_records.jsonl")
    assert data["manifest"]["sha256"] == sha256_of(work / "sft_data" / "run_manifest.json")
    assert (data["examples"], data["loss_tokens"], data["source_note"]) == (12, 126, None)
    assert data["source_run"]["model"] == MODEL and data["source_run"]["adapter"] is None
    # Versions, the GPU, the times, the code.
    assert manifest["environment"]["python"] == "3.11.17" and manifest["environment"]["gpu_name"]
    assert {"torch", "transformers", "peft"} <= set(manifest["environment"]["packages"])
    assert manifest["started_utc"] <= manifest["ended_utc"] and manifest["seconds"] >= 0
    assert {"code_sha256", "frozen", "frozen_note"} <= set(manifest)
    assert set(manifest["training_code"]["files"]) == every_training_file()
    assert manifest["training_code"] == scripts.sft_data.training_code()
    # After the training: the log's numbers, the adapter's hashes, the weight change.
    summary = manifest["summary"]
    assert (summary["steps"], summary["first_loss"], summary["last_loss"]) == (4, 1.0, 0.25)
    # Epoch 0: steps 0 and 1, losses 1 and 1/2 on 10 and 20 tokens: (10 + 10) / 30. Epoch 1: losses 1/3 and
    # 1/4 on 30 and 40 tokens: (10 + 10) / 70.
    assert summary["mean_loss_by_epoch"] == pytest.approx([20 / 30, 20 / 70])
    assert (summary["max_grad_norm_seen"], summary["clipped_steps"]) == (2.0, 1)
    assert summary["seconds_in_steps"] == 10.0                           # 4 steps of 2.5 s
    assert manifest["outputs"]["adapter"]["adapter_model.safetensors"]["sha256"] == \
        hashlib.sha256(b"weights").hexdigest()
    assert manifest["outputs"]["log"]["sha256"] == sha256_of(out / "sft_log.jsonl")
    assert manifest["weight_change"]["by_block"]["mlp"] == 0.7
    assert read_json(out / "lora_coverage.json")["rows"][0]["wrapped"] is True


def test_train_sft_refuses_before_the_model_loads(scripts, sft_data_done, monkeypatch, capsys):
    work = sft_data_done
    seen = sft_stand_ins(scripts, monkeypatch)

    def message(argv: list[str]) -> str:
        assert scripts.train_sft.main(argv) == 2
        return capsys.readouterr().err

    # The episodes were sampled by the 2B: another model must not be trained on them without a word.
    other = ["--model", "Qwen/Qwen3.5-4B", *train_sft_argv(work)[2:]]
    assert "sampled by" in message(other)
    assert "sampled by" in message([*train_sft_argv(work), "--revision", "0" * 40])
    # A data folder that sft_data.py did not finish, or whose record file was changed afterwards.
    records_file = work / "sft_data" / "sft_records.jsonl"
    original = records_file.read_bytes()
    records_file.write_bytes(original + b"\n")
    assert "not the file that sft_data.py wrote" in message(train_sft_argv(work))
    records_file.write_bytes(original)
    manifest_file = work / "sft_data" / "run_manifest.json"
    manifest = read_json(manifest_file)
    write_json(manifest_file, {**manifest, "complete": False})
    assert "not the manifest of a complete run" in message(train_sft_argv(work))
    write_json(manifest_file, {**manifest, "config": {**manifest["config"], "turn_limit": 3}})
    assert "does not give its own hash again" in message(train_sft_argv(work))
    manifest_file.unlink()
    assert "is not an output folder" in message(train_sft_argv(work))
    assert seen == {} and not (work / "sft").exists()


def test_train_sft_accepts_data_of_another_model_only_when_told(scripts, sft_data_done, monkeypatch, capsys):
    work = sft_data_done
    sft_stand_ins(scripts, monkeypatch)
    argv = ["--model", "Qwen/Qwen3.5-4B", *train_sft_argv(work)[2:], "--allow-other-source"]
    assert scripts.train_sft.main(argv) == 0
    capsys.readouterr()
    note = read_json(work / "sft" / "run_manifest.json")["data"]["source_note"]
    assert "accepted with --allow-other-source" in note and MODEL in note
    # A second run into the same folder is refused.
    assert scripts.train_sft.main(argv) == 2
    assert "already holds" in capsys.readouterr().err


def test_a_failed_sft_run_is_not_complete(scripts, sft_data_done, monkeypatch, capsys):
    work = sft_data_done
    sft_stand_ins(scripts, monkeypatch, error=FloatingPointError("the gradient norm is nan"))
    assert scripts.train_sft.main(train_sft_argv(work)) == 1
    assert "SFT FAILED" in capsys.readouterr().err
    manifest = read_json(work / "sft" / "run_manifest.json")
    assert manifest["complete"] is False and manifest["status"] == "failed"
    assert "the gradient norm is nan" in manifest["error"] and manifest["summary"] is None


def test_an_sft_run_whose_adapters_did_not_move_in_one_block_is_not_complete(scripts, sft_data_done,
                                                                             monkeypatch, capsys):
    work = sft_data_done
    still = {"by_block": {"linear_attn": 0.0, "mlp": 0.7, "self_attn": 0.2}, "total": 0.7}
    sft_stand_ins(scripts, monkeypatch, change=still)
    assert scripts.train_sft.main(train_sft_argv(work)) == 1
    assert "did not move" in capsys.readouterr().err
    manifest = read_json(work / "sft" / "run_manifest.json")
    assert manifest["complete"] is False and "linear_attn" in manifest["error"]


# ---------------------------------------------------------------------------
# train_grpo.py: main() with a stand-in for the GPU part
# ---------------------------------------------------------------------------
def step_log(step: int, scenarios: list[str], fault: str | None = None) -> dict:
    """One line of train_log.jsonl with the keys that the script reads."""
    return {"step": step, "scenarios": scenarios, "n_episodes": 8 * len(scenarios), "n_harness_errors": 0,
            "mean_reward": 0.5 + step / 1000, "pass_rate": 0.5, "kept_groups": len(scenarios) - 1,
            "zero_variance_exact": 0.25, "zero_variance_pass_fail": 0.25, "invalid_call_rate": 0.01,
            "n_trained_episodes": 8 * (len(scenarios) - 1), "n_loss_tokens": 1000,
            "logprob_gap": {"n": 1000, "mean": 0.003, "p99": 0.02, "max": 0.1}, "grad_norm": 0.4,
            "fault": fault, "rollout_seconds": 100.0, "seconds": 150.0, "cost_usd": None}


def grpo_stand_ins(scripts, monkeypatch, *, fault_at: int | None = None, crash_after: int | None = None,
                   replay_step: int | None = None) -> dict:
    """Replace the GPU functions of train_grpo.py. The stand-in for the training writes what the library
    writes: the log, and a checkpoint with an adapter every `save_every` steps and at the end.

        fault_at      the stop rule fires at this step (its log line has a fault; the run ends there)
        crash_after   the process dies after this many steps (an exception)
        replay_step   this step uses the scenarios of the step before it (a faulty restart)
    """
    seen: dict = {"starts": []}

    def setup_model(args, cfg, adapter_dir):
        seen["starts"].append({"adapter_dir": adapter_dir, "cfg": cfg})
        return {"model": "the model", "tokenizer": "the tokenizer", "environment": ENVIRONMENT,
                "lora": {"loaded_from": str(adapter_dir)}, "kernels": {}, "sampler": {}}

    def save(out_dir: Path, steps: int) -> None:
        name = f"checkpoints/step_{steps:05d}"
        (out_dir / name / "adapter").mkdir(parents=True, exist_ok=True)
        (out_dir / name / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (out_dir / name / "adapter" / "adapter_model.safetensors").write_bytes(f"step {steps}".encode())
        write_json(out_dir / "latest.json", {"completed_steps": steps, "checkpoint": name})

    def run_training(setup, families, text, cfg, gcfg, args):
        seen.update({"families": families, "gcfg": gcfg, "cfg": cfg, "resume": args.resume, "text": text})
        ids = sorted(scenario_id for family in families for scenario_id in scenario_ids(family))
        log_file = args.out / "train_log.jsonl"
        first = read_json(args.out / "latest.json")["completed_steps"] if args.resume else 0
        logs = [log for log in (read_records(log_file) if args.resume else []) if log["step"] < first]
        for step in range(first, gcfg.steps):
            if crash_after is not None and step == crash_after:
                write_records(log_file, logs)
                raise RuntimeError("the machine went away")
            used = step - 1 if step == replay_step else step
            log = step_log(step, step_scenarios(ids, gcfg.n_scenarios, gcfg.seed, used),
                           "harness errors: 3 of 128 episodes so far" if step == fault_at else None)
            logs.append(log)
            if log["fault"] is not None:
                write_json(args.out / "stopped.json", {"step": step, "reason": log["fault"]})
                break
            if (step + 1) % gcfg.save_every == 0 or step + 1 == gcfg.steps:
                save(args.out, step + 1)
        write_records(log_file, logs)
        return logs

    monkeypatch.setattr(scripts.train_grpo, "setup_model", setup_model)
    monkeypatch.setattr(scripts.train_grpo, "run_training", run_training)
    monkeypatch.setattr(scripts.train_grpo, "peak_gpu_memory_gb", lambda: None)
    return seen


@pytest.fixture
def grpo_work(work, cfg) -> Path:
    """The work folder with an SFT adapter and the manifest of the run that trained it."""
    adapter = work / "sft" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"sft weights")
    of_sft = dataclasses.replace(RunConfig(), max_new_tokens=256)        # the token cap of the chain
    write_json(work / "sft" / "run_manifest.json", {
        "complete": True, "script": "scripts/gpu/train_sft.py", "model": MODEL, "revision": REVISION,
        "seed": 0, "config": of_sft.to_dict(), "config_sha256": of_sft.sha256(),
        "sft_config_sha256": "1" * 64})
    return work


def train_grpo_argv(work: Path, text, *more: str, out: str = "grpo_seed1", cap: str = "256") -> list[str]:
    return ["--model", MODEL, "--adapter", str(work / "sft" / "adapter"), "--text", str(text.path),
            "--allow-placeholder", "--scenarios", str(work / "scenarios"), "--out", str(work / out),
            "--max-new-tokens", cap, "--seed", "1", *more]


def test_train_grpo_trains_the_sft_adapter_on_the_600_training_scenarios(scripts, grpo_work, monkeypatch,
                                                                         text, capsys):
    work = grpo_work
    seen = grpo_stand_ins(scripts, monkeypatch)
    argv = train_grpo_argv(work, text)
    assert scripts.train_grpo.main(argv) == 0
    capsys.readouterr()
    # The model is loaded with the SFT adapter, and the library gets the 200 train families in their three
    # training wordings (600 scenarios), the plan's settings, and the config with the given token cap.
    assert seen["starts"][0]["adapter_dir"] == work / "sft" / "adapter"
    families = seen["families"]
    assert len(families) == 200 and {family["split"] for family in families} == {"train"}
    assert sum(len(family["wordings"]) for family in families) == 600
    assert {wording["style"] for family in families for wording in family["wordings"]} == {1, 2, 3}
    assert seen["gcfg"] == GRPOConfig(seed=1) and seen["resume"] is False
    assert seen["cfg"] == dataclasses.replace(RunConfig(), max_new_tokens=256) == seen["starts"][0]["cfg"]

    out = work / "grpo_seed1"
    manifest = read_json(out / "run_manifest.json")
    assert manifest["complete"] is True and manifest["status"] == "complete" and manifest["mode"] == "seed"
    assert manifest["command"] == ["scripts/gpu/train_grpo.py", *argv]
    assert (manifest["model"], manifest["revision"], manifest["seed"]) == (MODEL, REVISION, 1)
    assert manifest["grpo_config"] == GRPOConfig(seed=1).to_dict() and manifest["norm_const"] == 20480.0
    assert manifest["grpo_config_sha256"] == hashlib.sha256(
        json.dumps(GRPOConfig(seed=1).to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    assert manifest["config"]["max_new_tokens"] == 256
    assert manifest["config_sha256"] == dataclasses.replace(RunConfig(), max_new_tokens=256).sha256()
    assert manifest["text_sha256"] == text.sha256
    assert manifest["split_file_sha256"] == sha256_of(work / "scenarios" / "train.jsonl")
    assert (manifest["families"], manifest["scenarios"], manifest["styles"], manifest["overfit"]) == \
        (200, 600, [1, 2, 3], None)
    start = manifest["start_adapter"]
    assert start["files"]["adapter_model.safetensors"]["sha256"] == hashlib.sha256(b"sft weights").hexdigest()
    assert start["trained_by"]["manifest"]["sha256"] == sha256_of(work / "sft" / "run_manifest.json")
    assert manifest["environment"]["gpu_name"] and manifest["environment"]["packages"]["torch"]
    assert {"code_sha256", "frozen", "frozen_note"} <= set(manifest)
    assert set(manifest["training_code"]["files"]) == every_training_file()
    assert manifest["training_code"] == scripts.sft_data.training_code()
    assert len(manifest["starts"]) == 1 and manifest["starts"][0]["resumed_after_step"] is None
    assert manifest["starts"][0]["result"] == "complete" and manifest["seconds"] >= 0
    # The adapter of the last checkpoint is the result of the run.
    assert manifest["final_adapter"]["folder"] == "checkpoints/step_00080/adapter"
    assert manifest["final_adapter"]["files"]["adapter_model.safetensors"]["sha256"] == \
        hashlib.sha256(b"step 80").hexdigest()
    # The summary adds up the log lines of the stand-in: 80 steps of 16 groups, 15 kept in each.
    summary = manifest["summary"]
    assert (summary["steps"], summary["groups"], summary["kept_groups"]) == (80, 1280, 1200)
    assert (summary["episodes"], summary["trained_episodes"], summary["harness_errors"]) == (10240, 9600, 0)
    assert (summary["seconds_in_steps"], summary["cost_usd"]) == (12000.0, None)
    assert summary["first_step"]["mean_reward"] == 0.5 and summary["last_step"]["step"] == 79
    assert summary["last_step"]["logprob_gap_mean"] == 0.003


def test_a_fault_of_the_stop_rule_ends_with_exit_code_3(scripts, grpo_work, monkeypatch, text, capsys):
    work = grpo_work
    grpo_stand_ins(scripts, monkeypatch, fault_at=12)
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 3
    message = capsys.readouterr().err
    assert "STOPPED BY THE STOP RULE at step 12: harness errors: 3 of 128 episodes so far" in message
    assert "checkpoints/step_00010" in message
    manifest = read_json(work / "grpo_seed1" / "run_manifest.json")
    assert manifest["complete"] is False and manifest["status"] == "stopped_by_rule"
    assert manifest["fault"] == {"step": 12, "reason": "harness errors: 3 of 128 episodes so far"}
    assert manifest["final_adapter"] is None and manifest["summary"]["steps"] == 13
    assert manifest["last_checkpoint"]["completed_steps"] == 10
    assert manifest["starts"][0]["result"] == "stopped_by_rule"


def test_a_run_that_lost_its_machine_is_continued_from_the_last_checkpoint(scripts, grpo_work, monkeypatch,
                                                                           text, capsys):
    work = grpo_work
    grpo_stand_ins(scripts, monkeypatch, crash_after=23)
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 1
    assert "GRPO FAILED: RuntimeError: the machine went away" in capsys.readouterr().err
    out = work / "grpo_seed1"
    manifest = read_json(out / "run_manifest.json")
    assert manifest["complete"] is False and manifest["status"] == "failed"
    assert manifest["last_checkpoint"]["completed_steps"] == 20

    # A second start without --resume is refused; so is a resume with another seed or learning rate.
    seen = grpo_stand_ins(scripts, monkeypatch)
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 2
    assert "pass --resume" in capsys.readouterr().err
    other_seed = [arg if arg != "1" else "2" for arg in train_grpo_argv(work, text)] + ["--resume"]
    assert scripts.train_grpo.main(other_seed) == 2
    message = capsys.readouterr().err
    assert "differs from the run" in message and "grpo_config" in message
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume", "--lr", "3e-5")) == 2
    assert "grpo_config" in capsys.readouterr().err
    assert seen["starts"] == []

    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 0
    capsys.readouterr()
    # The model is loaded with the adapter of the checkpoint after step 20, not with the SFT adapter.
    assert seen["starts"][0]["adapter_dir"] == out / "checkpoints" / "step_00020" / "adapter"
    assert seen["resume"] is True
    manifest = read_json(out / "run_manifest.json")
    assert manifest["complete"] is True
    assert manifest["final_adapter"]["folder"] == "checkpoints/step_00080/adapter"
    assert [(start["resumed_after_step"], start["result"]) for start in manifest["starts"]] == \
        [(None, "failed"), (20, "complete")]
    assert manifest["started_utc"] == manifest["starts"][0]["started_utc"]
    # The finished log holds every step once, each with the scenarios of the seeded order.
    logs = read_records(out / "train_log.jsonl")
    assert [log["step"] for log in logs] == list(range(80))
    # A third start has nothing to do and loads no model.
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 0
    assert "already complete" in capsys.readouterr().out and len(seen["starts"]) == 1


def test_a_resume_needs_a_checkpoint(scripts, grpo_work, monkeypatch, text, capsys):
    work = grpo_work
    grpo_stand_ins(scripts, monkeypatch, crash_after=4)              # before the first checkpoint
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 1
    capsys.readouterr()
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 2
    assert "no checkpoint" in capsys.readouterr().err
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume", out="nothing_here")) == 2
    capsys.readouterr()


def test_a_restart_that_replays_a_step_is_found_in_the_log(scripts, grpo_work, monkeypatch, text, capsys):
    """PLAN 5.6: after a restart no scenario is replayed. The script checks its own finished log against
    the library's `step_scenarios`."""
    work = grpo_work
    grpo_stand_ins(scripts, monkeypatch, replay_step=21)
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 1
    assert "did not use the scenarios of the seeded order, first: 21" in capsys.readouterr().err
    manifest = read_json(work / "grpo_seed1" / "run_manifest.json")
    assert manifest["complete"] is False and manifest["status"] == "failed"
    assert manifest["final_adapter"] is None


def test_log_problems_and_the_scenario_order_of_a_pass(scripts, families):
    ids = sorted(scenario_id for family in families["train"] for scenario_id in scenario_ids(family))
    gcfg = GRPOConfig(seed=4, steps=40)
    logs = [step_log(step, step_scenarios(ids, 16, 4, step)) for step in range(40)]
    saved = {"completed_steps": 40, "checkpoint": "checkpoints/step_00040"}
    assert scripts.train_grpo.log_problems(logs, ids, gcfg, saved) == []
    # 600 scenarios in steps of 16: one pass is 37 steps, and inside it no scenario is used twice.
    used = [scenario for log in logs[:37] for scenario in log["scenarios"]]
    assert len(used) == 592 == len(set(used))
    assert len(scripts.train_grpo.log_problems(logs[:30], ids, gcfg, saved)) == 1          # steps are missing
    assert len(scripts.train_grpo.log_problems(logs, ids, gcfg, {**saved, "completed_steps": 30})) == 1
    assert len(scripts.train_grpo.log_problems(logs, ids, gcfg, None)) == 1
    twice = [*logs[:20], {**logs[20], "scenarios": logs[19]["scenarios"]}, *logs[21:]]
    assert "first: 20" in scripts.train_grpo.log_problems(twice, ids, gcfg, saved)[0]


def test_the_overfit_check_trains_on_the_named_families_only(scripts, grpo_work, monkeypatch, text, families,
                                                             capsys):
    work = grpo_work
    seen = grpo_stand_ins(scripts, monkeypatch)
    wanted = [families["train"][3]["family_id"], families["train"][40]["family_id"]]
    ids_file = work / "overfit.txt"
    ids_file.write_text("\n".join(wanted) + "\n\n", encoding="utf-8")
    argv = train_grpo_argv(work, text, "--overfit", str(ids_file), "--styles", "1", out="overfit")
    assert scripts.train_grpo.main(argv) == 0
    capsys.readouterr()
    assert [family["family_id"] for family in seen["families"]] == wanted
    assert all([wording["style"] for wording in family["wordings"]] == [1] for family in seen["families"])
    assert seen["gcfg"] == GRPOConfig(seed=1, steps=30, n_scenarios=2)
    manifest = read_json(work / "overfit" / "run_manifest.json")
    assert (manifest["mode"], manifest["families"], manifest["scenarios"], manifest["styles"]) == \
        ("overfit", 2, 2, [1])
    assert manifest["overfit"]["family_ids"] == wanted
    assert manifest["overfit"]["file"]["sha256"] == sha256_of(ids_file)
    assert manifest["final_adapter"]["folder"] == "checkpoints/step_00030/adapter"
    # The families of the scenario file are not changed by the selection of one wording.
    assert all(len(family["wordings"]) == 3 for family in families["train"])


def test_train_grpo_refuses_before_the_model_loads(scripts, grpo_work, monkeypatch, text, families, capsys):
    work = grpo_work
    seen = grpo_stand_ins(scripts, monkeypatch)

    def message(argv: list[str]) -> str:
        assert scripts.train_grpo.main(argv) == 2
        return capsys.readouterr().err

    # The held-out style is never trained on; a family of another split cannot be named.
    assert "held-out style" in message(train_grpo_argv(work, text, "--styles", "1,2,3,4"))
    ids_file = work / "ids.txt"
    ids_file.write_text(families["dev"][0]["family_id"] + "\n", encoding="utf-8")
    overfit = train_grpo_argv(work, text, "--overfit", str(ids_file))
    assert "must name families of the train split" in message(overfit)
    # Too few scenarios for a step.
    ids_file.write_text(families["train"][0]["family_id"] + "\n", encoding="utf-8")
    assert "not enough" in message(train_grpo_argv(work, text, "--overfit", str(ids_file),
                                                   "--scenarios-per-step", "16"))
    # The placeholder text needs its switch, and then the smoke folder.
    without = [arg for arg in train_grpo_argv(work, text) if arg != "--allow-placeholder"]
    assert "placeholder" in message(without)
    monkeypatch.setattr(scripts.sft_data.rollout_script, "SMOKE_DIR", work / "elsewhere")
    assert "must be under" in message(train_grpo_argv(work, text))
    monkeypatch.setattr(scripts.sft_data.rollout_script, "SMOKE_DIR", work.parent)
    # The SFT run used another token cap (256) than this run: training and evaluation would differ.
    assert "other settings" in message(train_grpo_argv(work, text, cap="128"))
    # The adapter was trained on another model, or its run is not complete, or the folder has no adapter.
    assert "was trained on" in message(["--model", "Qwen/Qwen3.5-4B", *train_grpo_argv(work, text)[2:]])
    manifest_file = work / "sft" / "run_manifest.json"
    write_json(manifest_file, {**read_json(manifest_file), "complete": False})
    assert "not complete" in message(train_grpo_argv(work, text))
    (work / "sft" / "adapter" / "adapter_model.safetensors").unlink()
    assert "not a saved adapter" in message(train_grpo_argv(work, text))
    # A scenario file that was changed after the build.
    train_file = work / "scenarios" / "train.jsonl"
    train_file.write_bytes(train_file.read_bytes() + b"\n")
    assert "not the file that the build wrote" in message(train_grpo_argv(work, text))
    assert seen["starts"] == [] and not (work / "grpo_seed1").exists()


def test_train_grpo_accepts_other_settings_of_the_sft_run_only_when_told(scripts, grpo_work, monkeypatch,
                                                                         text, capsys):
    work = grpo_work
    grpo_stand_ins(scripts, monkeypatch)
    argv = train_grpo_argv(work, text, "--allow-other-settings", "--pilot", cap="128")
    assert scripts.train_grpo.main(argv) == 0
    capsys.readouterr()
    manifest = read_json(work / "grpo_seed1" / "run_manifest.json")
    assert "max_new_tokens" in manifest["start_adapter"]["note"]
    assert manifest["mode"] == "pilot" and manifest["grpo_config"]["steps"] == 15
    assert manifest["final_adapter"]["folder"] == "checkpoints/step_00015/adapter"
    # Without a manifest beside the adapter the run starts, and its manifest says what is not known.
    (work / "sft" / "run_manifest.json").unlink()
    assert scripts.train_grpo.main(train_grpo_argv(work, text, out="second")) == 0
    capsys.readouterr()
    assert "not known" in read_json(work / "second" / "run_manifest.json")["start_adapter"]["note"]


def test_the_fields_on_the_frozen_code_never_stop_a_run(scripts, monkeypatch, capsys):
    """Whatever accessdesk/freeze.py is at the moment, a run manifest gets its three fields."""
    assert set(scripts.sft_data.frozen_code()) == {"code_sha256", "frozen", "frozen_note"}

    def broken():
        raise SyntaxError("half-written")

    monkeypatch.setattr(scripts.sft_data, "_freeze_module", broken)
    fields = scripts.sft_data.frozen_code()
    assert fields["code_sha256"] is None and fields["frozen"] is None
    assert "half-written" in fields["frozen_note"]
    module = SimpleNamespace(code_sha256=lambda: "ab" * 32)
    monkeypatch.setattr(scripts.sft_data, "_freeze_module", lambda: module)
    fields = scripts.sft_data.frozen_code()
    assert fields["code_sha256"] == "ab" * 32 and fields["frozen"] is None
    assert fields["frozen_note"] == "accessdesk.freeze has no frozen_state"
    # With the two functions of the freeze module, the fields are what they return.
    state = {"manifest_sha256": "cd" * 32, "code_sha256": "ab" * 32, "matches": False,
             "differences": ["changed: accessdesk/config.py (group config)"]}
    module = SimpleNamespace(code_sha256=lambda root=None: "ab" * 32, frozen_state=lambda root=None: state)
    monkeypatch.setattr(scripts.sft_data, "_freeze_module", lambda: module)
    fields = scripts.sft_data.frozen_code()
    assert fields == {"code_sha256": "ab" * 32, "frozen": state, "frozen_note": None}
    # A frozen tree that differs from its manifest is said aloud; a tree that is not frozen, or is as
    # frozen, is not.
    scripts.sft_data.warn_if_not_as_frozen(fields)
    assert "differs from MANIFEST.json in 1 places" in capsys.readouterr().err
    for quiet in ({**state, "matches": True, "differences": []}, {**state, "manifest_sha256": None}, None):
        scripts.sft_data.warn_if_not_as_frozen({**fields, "frozen": quiet})
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# scripts/gpu/dryrun.py: how its GRPO stage calls the training loop (no torch, no model)
# ---------------------------------------------------------------------------
def test_the_dry_run_calls_the_grpo_loop_with_arguments_that_it_takes(scripts, monkeypatch, tmp_path,
                                                                      families, text, cfg):
    """The dry run itself runs on the rented machine only. Here its GRPO stage runs with a stand-in for
    `train_grpo` that binds every call to the signature of the real function: a call that the real
    function would not take raises TypeError. Both calls (the start and the restart) must name the hash of
    the train families and the hashes of the adapter that the SFT stage saved."""
    import argparse
    import inspect

    from accessdesk.env.scenarios import load_families
    from accessdesk.train import grpo

    dryrun = importlib.import_module("dryrun")
    assert "torch" not in sys.modules
    signature = inspect.signature(grpo.train_grpo)
    calls: list[dict] = []

    def fake_train_grpo(*args, **kwargs) -> list[dict]:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        # A line of the real log also has `loss` (accessdesk/train/grpo.py, `grpo_step`).
        return [{**step_log(step, [f"scenario {step}a", f"scenario {step}b"]), "loss": 0.25}
                for step in range(bound.arguments["gcfg"].steps)]

    monkeypatch.setattr(grpo, "train_grpo", fake_train_grpo)
    monkeypatch.setattr(grpo, "latest_checkpoint",
                        lambda out_dir: {"completed_steps": 2, "checkpoint": "checkpoints/step_00002"})
    monkeypatch.setattr(dryrun, "reload_from_checkpoint", lambda ctx, adapter_dir: {"adapter": str(adapter_dir)})
    out = tmp_path / "name" / "dryrun"
    for name, content in (("adapter_config.json", b"{}"), ("adapter_model.safetensors", b"sft weights")):
        (out / "sft" / "adapter").mkdir(parents=True, exist_ok=True)       # what the SFT stage leaves
        (out / "sft" / "adapter" / name).write_bytes(content)
    ctx = {"args": argparse.Namespace(grpo_steps=3), "out_dir": out, "families": families, "text": text,
           "cfg": cfg, "model": "the model", "tokenizer": "the tokenizer"}
    numbers = dryrun.stage_grpo(ctx)
    assert numbers["steps"] == 3 and numbers["restart"] == {"adapter": str(out / "grpo" / "checkpoints"
                                                                           / "step_00002" / "adapter")}
    # Two calls: two steps, then the restart for the third. Everything else is the same in both.
    first, second = calls
    assert (first["resume"], first["gcfg"].steps) == (False, 2)
    assert (second["resume"], second["gcfg"].steps) == (True, 3)
    for call in calls:
        assert call["families"] is families["train"] and call["run_cfg"] is cfg and call["text"] is text
        assert (call["out_dir"], call["run"], call["save_tokens"]) == (out / "grpo", "name", True)
        assert call["allow_resume_after_fault"] is False
    # The data hash is the hash of a file that holds the train families as the build writes them.
    train_file = out / "grpo_train.jsonl"
    save_families(tmp_path / "expected.jsonl", families["train"])
    assert train_file.read_bytes() == (tmp_path / "expected.jsonl").read_bytes()
    assert len(load_families(train_file)) == len(families["train"]) == 200
    assert first["data_sha256"] == second["data_sha256"] == sha256_of(train_file)
    # The start adapter is named by the hashes of its two files.
    assert first["start_adapter"] == second["start_adapter"] == {
        "adapter_config.json": hashlib.sha256(b"{}").hexdigest(),
        "adapter_model.safetensors": hashlib.sha256(b"sft weights").hexdigest()}
    # The library's own rule for a restart takes the second call after the first, and would refuse it
    # with another adapter.
    state = {"start_adapter": {"files_sha256": first["start_adapter"]}, "data_sha256": first["data_sha256"]}
    assert grpo.origin_problem(state, second["start_adapter"], second["data_sha256"]) is None
    assert grpo.origin_problem(state, {**second["start_adapter"], "adapter_model.safetensors": "0" * 64},
                               second["data_sha256"]) is not None
