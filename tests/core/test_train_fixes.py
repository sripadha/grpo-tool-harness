"""Tests of the repairs of the training code after the review of the training maths (2026-11-18). No torch.

    a restart never erases the record of a stop              accessdesk/train/grpo.py (`begin_start`)
    the pieces of the output layer                           accessdesk/train/forward.py (`row_pieces`)
    only train scenarios in the training styles are used     grpo.py (`training_scenarios`), sft.py
    the training records name their data and their adapter   grpo.py, scripts/gpu/train_grpo.py, train_sft.py
    the GRPO path has sampling seeds of its own              grpo.py (`sampling_run_seed`)

Two kinds of tests of the loop:
  - the file steps of a start (`begin_start`, `record_step`, `stop_of_last_start`) on hand-made folders;
  - the real `train_grpo` with stand-ins for everything that needs a model (the fixture `loop`): the stand-in
    for the rollout plays scripted policies, and the stand-ins for the gradient and the optimiser step only
    count. So these tests show what the loop writes, refuses and passes on. They do not show that training
    works: that is tested on the rented machine (tests/gpu).

The tests of scripts/gpu/train_grpo.py and train_sft.py use the stand-ins and the fixtures of
tests/core/test_train_scripts.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from accessdesk.env.generator import TRAIN_STYLES
from accessdesk.env.scenarios import scenario_ids
from accessdesk.eval import metrics
from accessdesk.harness.episode import Episode
from accessdesk.harness.policies import constant, oracle, run_policy
from accessdesk.harness.records import read_records
from accessdesk.rng import derive_seed
from accessdesk.train import grpo
from accessdesk.train.forward import HEAD_ROWS, row_pieces
from accessdesk.train.grpo import (GRPOConfig, StoppedRunError, begin_start, origin_problem, read_starts,
                                   record_step, sampling_run_seed, step_scenarios, stop_of_last_start,
                                   train_grpo, training_scenarios)
from accessdesk.train.sft import attach_tokens, select_sft_records, training_example
from tests.core.test_train_scripts import (WRONG_RESOLUTION, grpo_stand_ins, grpo_work,  # noqa: F401
                                           read_json, scripts, sft_data_done, sft_stand_ins, sha256_of,
                                           smoke_dir_is_the_test_folder, template, token_row,
                                           train_grpo_argv, train_sft_argv, work)

DATA = "d" * 64                                  # a made-up SHA-256 of a scenario file
ADAPTER = {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}
FAULT = "mean log-probability gap 0.0300 is above 0.02"


def snapshot(folder: Path) -> dict[str, bytes]:
    """Every file under `folder`: its path from the folder -> its bytes."""
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def lines_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# The file steps of a start, on a hand-made folder
# ---------------------------------------------------------------------------
def hand_made_record(step: int, pos: int) -> dict:
    """The fields of an episode record that `record_step` reads, with made-up token data."""
    return {"scenario_id": f"tr-{step:03d}/s1", "sample": pos, "harness_error": None,
            "reward": 100 * (pos % 2),
            "seed": {"run_seed": 7, "pass": step, "chunk": 0, "pos": pos},
            "tokens": {"ids": [step, pos, 5], "segments": [["prompt", 0, 1, -1], ["sampled", 1, 3, 0]],
                       "logprobs": [-0.5, -0.25]}}


def run_steps(out: Path, start: int, steps: range, fault_at: int | None = None) -> None:
    """What the loop writes for these steps: `record_step` for each, with a fault at `fault_at`."""
    for step in steps:
        log = {"step": step, "scenarios": [f"tr-{step:03d}/s1"], "fault": FAULT if step == fault_at else None}
        stopped = record_step(out, start, log, [hand_made_record(step, pos) for pos in range(2)], True)
        assert stopped is (step == fault_at)
        if stopped:
            break


@pytest.fixture
def stopped_run(tmp_path) -> Path:
    """The folder of the review's case: start 1 ran the steps 0 to 45, the stop rule fired at step 45, and
    the last checkpoint is the one after 40 steps (the caller of `begin_start` gives that number)."""
    out = tmp_path / "run"
    out.mkdir()
    assert begin_start(out, resume=False) == (1, [])
    run_steps(out, 1, range(0, 80), fault_at=45)
    return out


def test_without_the_flag_a_restart_of_a_stopped_run_is_refused_and_touches_no_file(stopped_run):
    before = snapshot(stopped_run)
    assert json.loads(before["stopped.json"]) == {"step": 45, "reason": FAULT, "start": 1}
    with pytest.raises(StoppedRunError, match="step 45") as refusal:
        begin_start(stopped_run, resume=True, completed_steps=40)
    assert FAULT in str(refusal.value) and "allow_resume_after_fault" in str(refusal.value)
    assert snapshot(stopped_run) == before


def test_an_allowed_restart_keeps_the_fault_on_disk(stopped_run):
    before = snapshot(stopped_run)
    old_log = before["train_log.jsonl"].decode("utf-8").splitlines(keepends=True)
    assert len(old_log) == 46 and json.loads(old_log[45])["fault"] == FAULT

    start, kept = begin_start(stopped_run, resume=True, completed_steps=40, allow_resume_after_fault=True,
                              details={"seed": 3})
    after = snapshot(stopped_run)
    assert start == 2 and [log["step"] for log in kept] == list(range(40))
    # The log keeps the lines of the steps 0 to 39, byte for byte. The six lines that were cut (the steps
    # 40 to 45, which the new start runs again) are in the file of start 1, byte for byte, with the fault.
    assert after["train_log.jsonl"] == "".join(old_log[:40]).encode("utf-8")
    assert after["train_log.start_1.jsonl"] == "".join(old_log[40:]).encode("utf-8")
    cut = lines_of(stopped_run / "train_log.start_1.jsonl")
    assert [log["step"] for log in cut] == [40, 41, 42, 43, 44, 45]
    assert [log["fault"] for log in cut] == [None] * 5 + [FAULT] and all(log["start"] == 1 for log in cut)
    # stopped.json of start 1 has the name of start 1 now, with the same bytes.
    assert "stopped.json" not in after and after["stopped.start_1.json"] == before["stopped.json"]
    # The episodes and the token data of the steps 40 to 45 are under the name of start 1; those of the
    # steps before the checkpoint stay where they are.
    for step in range(46):
        for kind in ("episodes", "tokens"):
            name = f"step_{step:05d}.{kind}.jsonl"
            new_place = f"rollouts/start_1/{name}" if step >= 40 else f"rollouts/{name}"
            assert after[new_place] == before[f"rollouts/{name}"], name
            assert (f"rollouts/{name}" in after) is (step < 40), name
    # Nothing that was on disk is gone: the files are the old ones, under the old or the new name, and
    # starts.jsonl has one more line.
    assert len(after) == len(before) + 1                    # the log is two files now
    assert after["starts.jsonl"].startswith(before["starts.jsonl"])
    events = read_starts(stopped_run)
    assert [(event["event"], event["start"]) for event in events] == \
        [("start", 1), ("fault", 1), ("start", 2)]
    assert (events[1]["step"], events[1]["reason"]) == (45, FAULT)
    assert events[0]["resume"] is False and events[0]["resumed_after_fault"] is None
    restart = events[2]
    assert (restart["resume"], restart["resumed_after_step"], restart["allow_resume_after_fault"]) == \
        (True, 40, True)
    assert restart["resumed_after_fault"] == {"step": 45, "reason": FAULT, "start": 1}
    assert restart["moved"] == {"log_lines": "train_log.start_1.jsonl", "n_log_lines": 6,
                                "rollout_files": "rollouts/start_1", "n_rollout_files": 12,
                                "stopped": "stopped.start_1.json"}
    assert restart["seed"] == 3                             # the caller's details are in the line


def test_after_an_allowed_restart_the_flag_is_needed_again_only_if_the_rule_fires_again(stopped_run):
    begin_start(stopped_run, resume=True, completed_steps=40, allow_resume_after_fault=True)
    assert stop_of_last_start(stopped_run) is None
    # Start 2 runs the steps 40 to 43 and loses its machine: start 3 needs no flag.
    run_steps(stopped_run, 2, range(40, 44))
    start, kept = begin_start(stopped_run, resume=True, completed_steps=40)
    assert start == 3 and len(kept) == 40
    assert [log["step"] for log in lines_of(stopped_run / "train_log.start_2.jsonl")] == [40, 41, 42, 43]
    # Start 3 runs into the rule at step 41: the next restart is refused again, and an allowed one keeps
    # the records of both stops.
    run_steps(stopped_run, 3, range(40, 80), fault_at=41)
    assert stop_of_last_start(stopped_run) == {"step": 41, "reason": FAULT, "start": 3}
    with pytest.raises(StoppedRunError, match="step 41"):
        begin_start(stopped_run, resume=True, completed_steps=40)
    start, _ = begin_start(stopped_run, resume=True, completed_steps=40, allow_resume_after_fault=True)
    assert start == 4
    assert read_json(stopped_run / "stopped.start_1.json")["step"] == 45
    assert read_json(stopped_run / "stopped.start_3.json") == {"step": 41, "reason": FAULT, "start": 3}
    faults = [log["step"] for name in ("train_log.start_1.jsonl", "train_log.start_3.jsonl")
              for log in lines_of(stopped_run / name) if log["fault"] is not None]
    assert faults == [45, 41]
    assert sorted(path.name for path in (stopped_run / "rollouts").iterdir() if path.is_dir()) == \
        ["start_1", "start_2", "start_3"]


def test_a_restart_after_a_lost_machine_needs_no_flag_and_keeps_what_it_cuts(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    begin_start(out, resume=False)
    run_steps(out, 1, range(0, 24))                         # then the machine went away; checkpoint at 20
    before = snapshot(out)
    assert stop_of_last_start(out) is None
    start, kept = begin_start(out, resume=True, completed_steps=20)
    assert start == 2 and [log["step"] for log in kept] == list(range(20))
    assert [log["step"] for log in lines_of(out / "train_log.start_1.jsonl")] == [20, 21, 22, 23]
    after = snapshot(out)
    assert after["rollouts/start_1/step_00023.tokens.jsonl"] == before["rollouts/step_00023.tokens.jsonl"]
    assert not any(name.startswith("stopped") for name in after)
    restart = read_starts(out)[-1]
    assert (restart["allow_resume_after_fault"], restart["resumed_after_fault"]) == (False, None)
    # A restart right at a checkpoint cuts nothing and writes no file but its start line.
    run_steps(out, 2, range(20, 30))
    start, kept = begin_start(out, resume=True, completed_steps=30)
    assert start == 3 and len(kept) == 30 and not (out / "train_log.start_2.jsonl").exists()
    assert read_starts(out)[-1]["moved"]["n_log_lines"] == 0


def test_a_cut_off_line_of_the_log_is_kept_as_it_is_and_does_not_block_the_restart(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    begin_start(out, resume=False)
    run_steps(out, 1, range(0, 12))
    with (out / "train_log.jsonl").open("ab") as handle:
        handle.write(b'{"step": 12, "scenar')                # the machine went away in the middle of a line
    start, kept = begin_start(out, resume=True, completed_steps=10)
    assert start == 2 and [log["step"] for log in kept] == list(range(10))
    cut = (out / "train_log.start_1.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in cut[:2]] == [10, 11] and cut[2] == '{"step": 12, "scenar'
    assert [log["step"] for log in lines_of(out / "train_log.jsonl")] == list(range(10))


def test_a_stop_is_found_in_each_of_the_three_places(stopped_run, tmp_path):
    wanted = {"step": 45, "reason": FAULT, "start": 1}
    assert stop_of_last_start(stopped_run) == wanted
    # Only the fault line of starts.jsonl is left (the other two were taken away by hand).
    only_starts = tmp_path / "only_starts"
    only_starts.mkdir()
    (only_starts / "starts.jsonl").write_bytes((stopped_run / "starts.jsonl").read_bytes())
    assert stop_of_last_start(only_starts) == wanted
    # Only stopped.json is left (a run of older code has no starts.jsonl, and no `start` in the file).
    only_file = tmp_path / "only_file"
    only_file.mkdir()
    (only_file / "stopped.json").write_text(json.dumps({"step": 45, "reason": FAULT}), encoding="utf-8")
    assert stop_of_last_start(only_file) == {**wanted, "start": None}
    # Only the fault in the log is left (the process died before stopped.json was written).
    only_log = tmp_path / "only_log"
    only_log.mkdir()
    (only_log / "train_log.jsonl").write_bytes((stopped_run / "train_log.jsonl").read_bytes())
    assert stop_of_last_start(only_log) == wanted
    for folder in (only_starts, only_file, only_log):
        before = snapshot(folder)
        with pytest.raises(StoppedRunError):
            begin_start(folder, resume=True, completed_steps=40)
        assert snapshot(folder) == before
    # A run of older code is counted as one earlier start: its files get the name of start 1.
    start, _ = begin_start(only_file, resume=True, completed_steps=40, allow_resume_after_fault=True)
    assert start == 2 and read_json(only_file / "stopped.start_1.json") == {"step": 45, "reason": FAULT}
    assert stop_of_last_start(tmp_path / "nothing_here") is None


def test_a_file_that_was_kept_before_is_never_replaced(stopped_run):
    (stopped_run / "stopped.start_1.json").write_text("an earlier record", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Nothing was deleted"):
        begin_start(stopped_run, resume=True, completed_steps=40, allow_resume_after_fault=True)
    assert (stopped_run / "stopped.start_1.json").read_text(encoding="utf-8") == "an earlier record"
    assert read_json(stopped_run / "stopped.json")["step"] == 45          # and the new one is still there


def test_a_new_run_does_not_start_in_a_folder_that_holds_a_run(stopped_run, tmp_path, families, text, cfg):
    for name in ("train_log.jsonl", "starts.jsonl", "stopped.json", "rollouts/step_00003.episodes.jsonl"):
        folder = tmp_path / name.replace("/", "_")
        (folder / name).parent.mkdir(parents=True)
        (folder / name).write_bytes((stopped_run / name).read_bytes())
        before = snapshot(folder)
        with pytest.raises(FileExistsError, match="choose a new folder"):
            begin_start(folder, resume=False)
        # The loop itself refuses too, before it touches the model (None here).
        with pytest.raises(FileExistsError, match="choose a new folder"):
            train_grpo(None, None, families["train"][:8], text, cfg, GRPOConfig(), out_dir=folder)
        assert snapshot(folder) == before


# ---------------------------------------------------------------------------
# The real loop, with stand-ins for everything that needs a model
# ---------------------------------------------------------------------------
class FakeModel:
    """Stands in for the model: `fingerprint` plays the sum of the LoRA weights. The stand-in for the
    optimiser step adds 1 to it, so every checkpoint has another fingerprint."""

    def __init__(self, fingerprint: float = 10.0) -> None:
        self.fingerprint = fingerprint
        self.mode = "eval"

    def train(self) -> None:
        self.mode = "train"

    def eval(self) -> None:
        self.mode = "eval"


class FakeOptimizer:
    def __init__(self) -> None:
        self.steps = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass


@pytest.fixture
def loop(monkeypatch, text) -> SimpleNamespace:
    """Replace what `train_grpo` needs a model for. Returns an object with
        calls            one entry per rollout call: run_seed, pass_index, meta, the scenario ids
        harness_errors   {step: n}: the first n episodes of that step are lost to a harness error

    The stand-in for the rollout plays every episode with a scripted policy, in the order of the real one:
    attempt 0, 2, ... of a scenario with the correct solution, attempt 1, 3, ... with a constant policy that
    is wrong for the family. So every group has two rewards of 100 and two of 0, and is kept.
    """
    seen = SimpleNamespace(calls=[], harness_errors={})

    def fake_rollout(sampler, layouts, episodes, *, run_seed, pass_index, meta):
        seen.calls.append({"run_seed": run_seed, "pass_index": pass_index, "meta": dict(meta),
                           "scenarios": sorted({f"{e.family['family_id']}/s{e.style}" for e in episodes})})
        records, attempts = [], {}
        for pos, episode in enumerate(episodes):
            scenario = f"{episode.family['family_id']}/s{episode.style}"
            sample = attempts.get(scenario, 0)
            attempts[scenario] = sample + 1
            wrong = constant(text, WRONG_RESOLUTION[episode.family["outcome"]])
            policy = oracle(text) if sample % 2 == 0 else wrong
            while not episode.done:
                episode.step(policy(episode.messages, episode.tool_specs), n_tokens=3)
            seed = {"run_seed": run_seed, "pass": pass_index, "chunk": 0, "pos": pos}
            record = episode.record(**meta, sample=sample, seed=seed)
            if pos < seen.harness_errors.get(pass_index, 0):
                record["harness_error"] = "made up for the test"
            record["tokens"] = {"ids": [1, 2, 3], "segments": [["prompt", 0, 1, -1], ["sampled", 1, 3, 0]],
                                "logprobs": [-0.5, -0.25]}
            records.append(record)
        return records

    def fake_gradient(model, records, advantages, selected, *, norm_const, pad_id, micro_batch):
        return {"loss": 0.125, "n_episodes": len(selected), "n_tokens": 2 * len(selected),
                "logprob_gap": {"n": 2 * len(selected), "mean": 0.001, "p99": 0.002, "max": 0.003}}

    def fake_step(model, optimizer, max_grad_norm):
        model.fingerprint += 1.0
        optimizer.steps += 1
        return 0.5

    def fake_save_adapter(model, folder):
        Path(folder).mkdir(parents=True)
        (Path(folder) / "adapter_model.safetensors").write_text(str(model.fingerprint), encoding="utf-8")

    def fake_load_optimizer(optimizer, path, model):
        optimizer.steps = json.loads(Path(path).read_text(encoding="utf-8"))["steps"]

    monkeypatch.setattr(grpo, "rollout", fake_rollout)
    monkeypatch.setattr(grpo, "make_optimizer", lambda model, lr: FakeOptimizer())
    monkeypatch.setattr(grpo, "lora_fingerprint", lambda model: model.fingerprint)
    monkeypatch.setattr(grpo, "set_gradient_checkpointing", lambda model, enabled: None)
    monkeypatch.setattr(grpo, "Sampler", lambda model, tokenizer, cfg: SimpleNamespace(cfg=cfg, pad_id=0))
    monkeypatch.setattr(grpo, "TokenLayout", lambda tokenizer, tool_specs: None)
    monkeypatch.setattr(grpo, "zero_gradients", lambda model: None)
    monkeypatch.setattr(grpo, "accumulate_policy_gradient", fake_gradient)
    monkeypatch.setattr(grpo, "clip_and_step", fake_step)
    monkeypatch.setattr(grpo, "save_adapter", fake_save_adapter)
    monkeypatch.setattr(grpo, "_save_optimizer", lambda optimizer, path: Path(path).write_text(
        json.dumps({"steps": optimizer.steps}), encoding="utf-8"))
    monkeypatch.setattr(grpo, "_load_optimizer", fake_load_optimizer)
    return seen


# Small steps: 2 scenarios x 4 attempts = 8 episodes, 8 steps, a checkpoint every 2 steps.
SMALL = GRPOConfig(n_scenarios=2, group_size=4, steps=8, save_every=2, seed=1)


def small_run(loop, families, text, cfg, out: Path, **more) -> tuple[FakeModel, list[dict]]:
    """A new run of `train_grpo` on 6 train families (18 scenarios) with the settings of SMALL."""
    model = FakeModel()
    options = {"out_dir": out, "run": "test_run", "save_tokens": True, "data_sha256": DATA,
               "start_adapter": ADAPTER, **more}
    return model, train_grpo(model, None, families["train"][:6], text, cfg, SMALL, **options)


def test_the_loop_runs_every_step_once_with_the_scenarios_and_the_seeds_of_its_seed(loop, families, text, cfg,
                                                                                    tmp_path):
    out = tmp_path / "run"
    model, logs = small_run(loop, families, text, cfg, out)
    ids = sorted(sid for family in families["train"][:6] for sid in scenario_ids(family))
    assert len(ids) == 18 and model.mode == "eval"
    assert [log["step"] for log in logs] == list(range(8)) and all(log["fault"] is None for log in logs)
    assert logs == lines_of(out / "train_log.jsonl")
    # Every group is two passes and two fails: kept, mean reward 0.5.
    assert all((log["kept_groups"], log["mean_reward"], log["n_trained_episodes"]) == (2, 0.5, 8)
               for log in logs)
    assert all(log["start"] == 1 for log in logs)
    # The scenarios of step k are the ones of the seeded order, and the rollout of step k got the run seed
    # of the GRPO path and k as its pass number.
    for step, (log, call) in enumerate(zip(logs, loop.calls)):
        assert log["scenarios"] == step_scenarios(ids, 2, 1, step)
        assert call["scenarios"] == sorted(log["scenarios"])
        assert (call["run_seed"], call["pass_index"]) == (sampling_run_seed(1), step)
        assert call["meta"] == {"run": "test_run", "arm": "grpo_train", "data_sha256": DATA}
    # Every saved episode names its data file, and its seed is the one that the harness was given.
    for step in range(8):
        records = read_records(out / "rollouts" / f"step_{step:05d}.episodes.jsonl")
        assert len(records) == 8 and all(record["data_sha256"] == DATA for record in records)
        assert all(record["seed"]["run_seed"] == sampling_run_seed(1) and record["seed"]["pass"] == step
                   for record in records)
        assert all("tokens" not in record for record in records)
        tokens = read_records(out / "rollouts" / f"step_{step:05d}.tokens.jsonl")
        assert [(row["scenario_id"], row["sample"], row["seed"]) for row in tokens] == \
            [(record["scenario_id"], record["sample"], record["seed"]) for record in records]
        assert all(row["ids"] == [1, 2, 3] and row["logprobs"] == [-0.5, -0.25] for row in tokens)
        assert attach_tokens(records, tokens)[0]["tokens"]["segments"] == [["prompt", 0, 1, -1],
                                                                           ["sampled", 1, 3, 0]]
    # Every checkpoint says which adapter the run started from: the hashes that the caller gave, and the
    # fingerprint of the weights before step 0 (10.0 in the stand-in; each step adds 1).
    origin = {"lora_fingerprint": 10.0, "files_sha256": ADAPTER}
    for steps in (2, 4, 6, 8):
        state = read_json(out / "checkpoints" / f"step_{steps:05d}" / "state.json")
        assert (state["completed_steps"], state["lora_fingerprint"]) == (steps, 10.0 + steps)
        assert state["start_adapter"] == origin and state["data_sha256"] == DATA
        assert state["sampling_run_seed"] == sampling_run_seed(1)
    assert read_json(out / "latest.json")["checkpoint"] == "checkpoints/step_00008"
    (first,) = read_starts(out)
    assert (first["event"], first["start"], first["resume"], first["seed"], first["steps"]) == \
        ("start", 1, False, 1, 8)
    assert first["start_adapter"] == origin and first["data_sha256"] == DATA
    assert (first["sampling_run_seed"], first["save_tokens"], first["lora_fingerprint"]) == \
        (sampling_run_seed(1), True, 10.0)
    assert not (out / "stopped.json").exists()


def test_without_save_tokens_no_token_file_is_written_and_without_a_data_hash_no_record_has_one(
        loop, families, text, cfg, tmp_path):
    out = tmp_path / "run"
    small_run(loop, families, text, cfg, out, save_tokens=False, data_sha256=None, start_adapter=None)
    assert len(list((out / "rollouts").glob("step_*.episodes.jsonl"))) == 8
    assert list((out / "rollouts").glob("*.tokens.jsonl")) == []
    assert all("data_sha256" not in record
               for record in read_records(out / "rollouts" / "step_00000.episodes.jsonl"))
    assert loop.calls[0]["meta"] == {"run": "test_run", "arm": "grpo_train"}
    state = read_json(out / "latest.json")
    assert state["start_adapter"] == {"lora_fingerprint": 10.0, "files_sha256": None}
    assert state["data_sha256"] is None


def test_the_loop_stops_at_a_fault_and_a_restart_keeps_every_record_of_it(loop, families, text, cfg,
                                                                          tmp_path, scripts):
    out = tmp_path / "run"
    train = families["train"][:6]
    ids = sorted(sid for family in train for sid in scenario_ids(family))
    loop.harness_errors = {5: 1}
    model, logs = small_run(loop, families, text, cfg, out)
    # Step 5 loses 1 episode: 1 harness error in 6 steps x 8 episodes = 48 is 2.1 %, above the limit of 1 %.
    assert [log["step"] for log in logs] == [0, 1, 2, 3, 4, 5]
    assert [log["fault"] is None for log in logs] == [True] * 5 + [False]
    reason = logs[5]["fault"]
    assert reason.startswith("harness errors: 1 of 48 episodes so far")
    assert logs[5]["grad_norm"] is None and logs[5]["n_trained_episodes"] == 0     # no update at the fault
    assert read_json(out / "stopped.json") == {"step": 5, "reason": reason, "start": 1}
    latest = read_json(out / "latest.json")
    assert (latest["completed_steps"], latest["lora_fingerprint"]) == (4, 14.0)
    assert model.fingerprint == 15.0                         # step 4 was done after the last checkpoint
    stopped = snapshot(out)

    # A restart is refused before the optimiser is made, and no file changes.
    restarted = FakeModel(fingerprint=14.0)                  # "the adapter of the last checkpoint is loaded"
    common = {"out_dir": out, "run": "test_run", "save_tokens": True, "resume": True}
    calls_before = len(loop.calls)
    with pytest.raises(StoppedRunError, match="step 5"):
        train_grpo(restarted, None, train, text, cfg, SMALL, data_sha256=DATA, start_adapter=ADAPTER,
                   **common)
    assert snapshot(out) == stopped and len(loop.calls) == calls_before

    # With the flag, the run must still be the same run: another scenario file or another start adapter is
    # refused, and so is a model that does not hold the adapter of the checkpoint. No file changes.
    allowed = {**common, "allow_resume_after_fault": True}
    with pytest.raises(ValueError, match="scenario file differs"):
        train_grpo(restarted, None, train, text, cfg, SMALL, data_sha256="e" * 64, start_adapter=ADAPTER,
                   **allowed)
    with pytest.raises(ValueError, match="start adapter differs"):
        train_grpo(restarted, None, train, text, cfg, SMALL, data_sha256=DATA,
                   start_adapter={**ADAPTER, "adapter_model.safetensors": "c" * 64}, **allowed)
    with pytest.raises(RuntimeError, match="does not hold the adapter"):
        train_grpo(FakeModel(fingerprint=15.0), None, train, text, cfg, SMALL, data_sha256=DATA,
                   start_adapter=ADAPTER, **allowed)
    assert snapshot(out) == stopped and len(loop.calls) == calls_before

    # The allowed restart goes on after step 3 (4 steps are in the checkpoint) and ends the run.
    loop.harness_errors = {}
    logs = train_grpo(restarted, None, train, text, cfg, SMALL, data_sha256=DATA, start_adapter=ADAPTER,
                      **allowed)
    after = snapshot(out)
    assert [log["step"] for log in logs] == list(range(8)) and all(log["fault"] is None for log in logs)
    assert logs == lines_of(out / "train_log.jsonl")
    assert [log["start"] for log in logs] == [1, 1, 1, 1, 2, 2, 2, 2]
    # No scenario is replayed or skipped: every step has the scenarios of the seeded order, and the steps
    # 4 and 5, which ran twice, were sampled with the same run seed and pass number both times.
    assert all(log["scenarios"] == step_scenarios(ids, 2, 1, log["step"]) for log in logs)
    assert [call["pass_index"] for call in loop.calls] == [0, 1, 2, 3, 4, 5, 4, 5, 6, 7]
    assert {call["run_seed"] for call in loop.calls} == {sampling_run_seed(1)}
    assert loop.calls[4]["scenarios"] == loop.calls[6]["scenarios"]
    # The record of the stop is still on disk: the log lines of the steps 4 and 5 of start 1, stopped.json
    # of start 1, and the episodes of the step with the harness error.
    old_lines = stopped["train_log.jsonl"].splitlines(keepends=True)
    assert after["train_log.start_1.jsonl"] == b"".join(old_lines[4:])
    assert after["train_log.jsonl"].startswith(b"".join(old_lines[:4]))
    cut = lines_of(out / "train_log.start_1.jsonl")
    assert [(log["step"], log["fault"]) for log in cut] == [(4, None), (5, reason)]
    assert after["stopped.start_1.json"] == stopped["stopped.json"] and "stopped.json" not in after
    for name in ("step_00004.episodes.jsonl", "step_00004.tokens.jsonl", "step_00005.episodes.jsonl",
                 "step_00005.tokens.jsonl"):
        assert after[f"rollouts/start_1/{name}"] == stopped[f"rollouts/{name}"]
    lost = read_records(out / "rollouts" / "start_1" / "step_00005.episodes.jsonl")
    assert [record["harness_error"] is not None for record in lost] == [True] + [False] * 7
    assert all(record["harness_error"] is None
               for record in read_records(out / "rollouts" / "step_00005.episodes.jsonl"))
    # Every file of the stopped run is still there with its bytes, under its name or under the name of
    # start 1. Three files grew or were replaced by a newer state: the log, starts.jsonl and latest.json.
    moved = {"stopped.json": "stopped.start_1.json",
             **{f"rollouts/step_0000{step}.{kind}.jsonl": f"rollouts/start_1/step_0000{step}.{kind}.jsonl"
                for step in (4, 5) for kind in ("episodes", "tokens")}}
    for name, content in stopped.items():
        if name not in ("train_log.jsonl", "starts.jsonl", "latest.json"):
            assert after[moved.get(name, name)] == content, name
    assert after["starts.jsonl"].startswith(stopped["starts.jsonl"])
    events = read_starts(out)
    assert [(event["event"], event["start"]) for event in events] == \
        [("start", 1), ("fault", 1), ("start", 2)]
    assert (events[1]["step"], events[1]["reason"]) == (5, reason)
    restart = events[2]
    assert (restart["resumed_after_step"], restart["allow_resume_after_fault"]) == (4, True)
    assert restart["resumed_after_fault"] == {"step": 5, "reason": reason, "start": 1}
    assert restart["lora_fingerprint"] == 14.0               # of the checkpoint that the restart began with
    # The origin of the run is the one of the first start, in every checkpoint of the restart too.
    final = read_json(out / "latest.json")
    assert (final["completed_steps"], final["lora_fingerprint"]) == (8, 18.0)
    assert final["start_adapter"] == {"lora_fingerprint": 10.0, "files_sha256": ADAPTER}
    assert final["data_sha256"] == DATA == restart["data_sha256"]
    # The script's list of the history files names all of them, each with its hash: starts.jsonl, the log
    # lines and the stopped.json of start 1, and the episode and token files of the steps that were run
    # again. The names are the paths from the run folder.
    history = scripts.train_grpo.history_files(out)
    assert sorted(history) == [
        "rollouts/start_1/step_00004.episodes.jsonl", "rollouts/start_1/step_00004.tokens.jsonl",
        "rollouts/start_1/step_00005.episodes.jsonl", "rollouts/start_1/step_00005.tokens.jsonl",
        "starts.jsonl", "stopped.start_1.json", "train_log.start_1.jsonl"]
    assert all(entry["sha256"] == sha256_of(out / name) for name, entry in history.items())
    assert all(entry["sha256"] == hashlib.sha256(after[name]).hexdigest() for name, entry in history.items())


def test_a_restart_that_gives_no_origin_keeps_the_one_of_the_first_start():
    recorded = {"start_adapter": {"lora_fingerprint": 10.0, "files_sha256": ADAPTER}, "data_sha256": DATA}
    assert origin_problem(recorded, None, None) is None
    assert origin_problem(recorded, dict(ADAPTER), DATA) is None
    assert "scenario file differs" in origin_problem(recorded, None, "e" * 64)
    assert "start adapter differs" in origin_problem(recorded, {"adapter_config.json": "a" * 64}, None)
    # A checkpoint of a first start that gave none of the two: a restart cannot add one later.
    nothing = {"start_adapter": {"lora_fingerprint": 10.0, "files_sha256": None}, "data_sha256": None}
    assert origin_problem(nothing, None, None) is None
    assert origin_problem(nothing, None, DATA) is not None
    assert origin_problem(nothing, ADAPTER, None) is not None


# ---------------------------------------------------------------------------
# Only train scenarios in the training styles
# ---------------------------------------------------------------------------
def with_held_out_wording(family: dict) -> dict:
    """A copy of a train family that also carries a wording in the held-out style, as a file that was
    changed by hand would."""
    extra = {**family["wordings"][0], "style": metrics.HELD_OUT_STYLE}
    return {**family, "wordings": [*family["wordings"], extra]}


def test_grpo_trains_on_train_families_in_the_training_styles_only(families, text, cfg, tmp_path):
    train = families["train"]
    assert metrics.HELD_OUT_STYLE not in TRAIN_STYLES
    scenarios = training_scenarios(train)
    assert len(scenarios) == 600 == len(set(scenarios)) and scenarios == sorted(scenarios)
    assert {sid.rpartition("/s")[2] for sid in scenarios} == {str(style) for style in TRAIN_STYLES}
    with pytest.raises(ValueError, match="train split only"):
        training_scenarios([*train[:3], families["dev"][0]])
    changed = [*train[:3], with_held_out_wording(train[3])]
    held_out = f"{train[3]['family_id']}/s{metrics.HELD_OUT_STYLE}"
    with pytest.raises(ValueError, match="training styles") as refusal:
        training_scenarios(changed)
    assert held_out in str(refusal.value) and "found 1 scenarios" in str(refusal.value)
    # The loop refuses both before it touches the model (None here) or the folder.
    for bad in (changed, [*train[:3], families["dev"][0]]):
        with pytest.raises(ValueError, match="training styles|train split only"):
            train_grpo(None, None, bad, text, cfg, GRPOConfig(), out_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def played(family: dict, style: int, text, cfg) -> dict:
    """The record of the correct solution on one scenario, with hand-made token data (the layout of
    tests/core/test_train_scripts.py: 5 prompt tokens, then 3 sampled tokens per turn)."""
    record = run_policy(Episode(family, style, text, cfg), oracle(text)).record(run="test", arm="test")
    return attach_tokens([record], [token_row(record)])[0]


def test_an_sft_example_is_made_from_a_train_record_in_a_training_style_only(families, text, cfg):
    family = families["train"][0]
    record = played(family, TRAIN_STYLES[0], text, cfg)
    example = training_example(record)
    assert sum(example["weights"]) == 3 * record["n_turns"] and example["ids"] == record["tokens"]["ids"]
    # The same episode with another split written on it, as a record of dev, test or calibration has.
    for split in ("dev", "test", "calib"):
        with pytest.raises(ValueError, match="train split only"):
            training_example({**record, "split": split})
    # A real dev episode, and a dev episode in the held-out style.
    dev = families["dev"][0]
    for style in (TRAIN_STYLES[0], metrics.HELD_OUT_STYLE):
        with pytest.raises(ValueError, match="train split only"):
            training_example(played(dev, style, text, cfg))
    # A train record in the held-out style can only come from a file that was changed by hand.
    held_out = {**record, "style": metrics.HELD_OUT_STYLE,
                "scenario_id": f"{family['family_id']}/s{metrics.HELD_OUT_STYLE}"}
    with pytest.raises(ValueError, match="training styles"):
        training_example(held_out)
    with pytest.raises(ValueError, match="training styles"):
        select_sft_records([record, held_out], cfg)
    assert select_sft_records([record], cfg)[0] == [record]


# ---------------------------------------------------------------------------
# The pieces of the output layer
# ---------------------------------------------------------------------------
def test_row_pieces_by_hand():
    assert row_pieces(0, 1024) == []
    assert row_pieces(1, 1024) == [(0, 1)]
    assert row_pieces(1024, 1024) == [(0, 1024)]
    assert row_pieces(1025, 1024) == [(0, 1024), (1024, 1025)]
    assert row_pieces(2500, 1024) == [(0, 1024), (1024, 2048), (2048, 2500)]
    assert row_pieces(7, 3) == [(0, 3), (3, 6), (6, 7)]
    assert row_pieces(3237) == [(0, 1024), (1024, 2048), (2048, 3072), (3072, 3237)]     # the default size
    for bad in ((5, 0), (5, -1), (-1, 4)):
        with pytest.raises(ValueError):
            row_pieces(*bad)


def test_row_pieces_hold_every_row_once_and_in_order():
    for n_rows in range(0, 40):
        for piece in range(1, 9):
            pieces = row_pieces(n_rows, piece)
            assert [row for start, stop in pieces for row in range(start, stop)] == list(range(n_rows))
            assert all(0 < stop - start <= piece for start, stop in pieces)
            assert len(pieces) == -(-n_rows // piece)        # the smallest number of pieces


def test_the_memory_numbers_in_the_docstring_are_those_of_the_piece_size():
    """The docstring of `_micro_batch_logprobs` gives the memory of the output layer for pieces of 1,024
    rows. The arithmetic, with the vocabulary of the 2B (248,320) and the bytes per entry of the docstring:
    (4 R + 10 P) V with pieces and 14 R V without. Change the docstring if the piece size changes."""
    vocabulary, gib = 248_320, 2 ** 30
    assert HEAD_ROWS == 1024
    assert round((4 * 3237 + 10 * HEAD_ROWS) * vocabulary / gib, 1) == 5.4
    assert round(14 * 3237 * vocabulary / gib, 1) == 10.5
    assert round((4 * 8192 + 10 * HEAD_ROWS) * vocabulary / gib, 1) == 9.9
    assert round(14 * 8192 * vocabulary / gib, 1) == 26.5


# ---------------------------------------------------------------------------
# The sampling seeds of the GRPO path
# ---------------------------------------------------------------------------
def test_the_run_seed_of_the_grpo_path_is_derived_from_the_seed_with_a_label():
    # The formula, written out without the repository's helper: the first 8 bytes of
    # SHA-256("1|grpo_sampling") as a whole number, and of that number the lowest 53 bits.
    by_hand = int.from_bytes(hashlib.sha256(b"1|grpo_sampling").digest()[:8], "big") % 2 ** 53
    assert sampling_run_seed(1) == by_hand == 3138651447157350
    assert sampling_run_seed(2) == 8888994274034250
    assert sampling_run_seed(1) == sampling_run_seed(1) != sampling_run_seed(2)


def test_every_reader_of_the_records_gets_the_run_seed_of_the_grpo_path_exactly():
    """The episode records of the training hold this number (`seed.run_seed`). The label alone gives a
    64-bit number: for seed 2 it is above 2**63, which a signed 64-bit integer cannot hold, and a reader that
    keeps numbers as doubles rounds every whole number above 2**53."""
    assert derive_seed(2, "grpo_sampling") >= 2 ** 63        # the number before the cut: the fault
    for seed in (0, 1, 2, 100, 101):
        run_seed = sampling_run_seed(seed)
        assert 0 <= run_seed < 2 ** 53
        as_double = json.loads(json.dumps(run_seed), parse_int=float)       # a reader with doubles only
        assert as_double == run_seed and int(as_double) == run_seed
        assert np.array([run_seed]).dtype == np.int64                       # numpy needs no unsigned type


def test_a_restart_is_refused_if_the_checkpoint_was_sampled_with_another_run_seed(tmp_path, cfg, monkeypatch):
    """A checkpoint that the code before the cut to 53 bits wrote names the uncut number. A restart from it
    would go on with other sampling seeds, so it is refused before the model or the optimiser is touched."""
    out = tmp_path / "run"
    out.mkdir()
    gcfg = GRPOConfig(seed=2)
    saved = {"completed_steps": 10, "n_episodes": 1280, "n_harness_errors": 0, "invalid_call_rate_step0": 0.0,
             "checkpoint": "checkpoints/step_00010", "lora_fingerprint": 14.0,
             "grpo_config": gcfg.to_dict(), "run_config_sha256": cfg.sha256(),
             "sampling_run_seed": derive_seed(2, "grpo_sampling")}
    (out / "latest.json").write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="run seed 18167402691831874122.*run seed 8888994274034250"):
        grpo.resume_state(out, None, None, gcfg, cfg)        # None: the model is not reached
    # With the run seed of this code the check passes, and the checks after it run.
    loaded = []
    monkeypatch.setattr(grpo, "lora_fingerprint", lambda model: 14.0)
    monkeypatch.setattr(grpo, "_load_optimizer", lambda optimizer, path, model: loaded.append(path))
    (out / "latest.json").write_text(json.dumps({**saved, "sampling_run_seed": sampling_run_seed(2)}),
                                     encoding="utf-8")
    state = grpo.resume_state(out, None, None, gcfg, cfg)
    assert state["completed_steps"] == 10 and loaded == [out / "checkpoints/step_00010" / "optimizer.pt"]


def test_a_grpo_run_shares_no_torch_seed_with_a_rollout():
    def stream(run_seed: int) -> set[int]:
        """The torch seeds that the harness makes from one run seed: 80 passes or steps, 8 chunks, 8 turns."""
        return {derive_seed(run_seed, number, chunk, turn)
                for number in range(80) for chunk in range(8) for turn in range(8)}

    # The fault that was repaired: the GRPO path gave its plain seed to the harness. Step s of a GRPO run
    # with seed 1 then had the torch seeds of pass s of a rollout with run seed 1: the whole of stream(1).
    # Now a GRPO run with seed 1 gives the harness another run seed, and its stream shares nothing with the
    # stream of a rollout that has the run seed 1, or any other run seed that is in use.
    # Rollouts: 0 (dev and test), 10 (the train rollout for SFT), and the numbers of the GRPO runs
    # themselves, in case a rollout is given one. GRPO: 1 and 2 (the two seeds), 100 (pilot), 101 (overfit
    # check).
    rollouts = {seed: stream(seed) for seed in (0, 1, 2, 10, 100, 101)}
    training = {seed: stream(sampling_run_seed(seed)) for seed in (1, 2, 100, 101)}
    for seed, seeds in training.items():
        assert len(seeds) == 80 * 8 * 8                      # 5,120 different torch seeds
        assert all(not seeds & other for other in rollouts.values()), seed
        assert all(not seeds & other for name, other in training.items() if name != seed), seed


# ---------------------------------------------------------------------------
# scripts/gpu/train_grpo.py and train_sft.py, with the stand-ins of tests/core/test_train_scripts.py
# ---------------------------------------------------------------------------
def with_origin_and_fingerprint(scripts, monkeypatch, seen: dict, fingerprints: list[float]) -> None:
    """Wrap the two stand-ins of `grpo_stand_ins`: the set-up gives each start the next fingerprint of the
    list (as `setup_model` does with the real model), and the training notes `setup["origin"]`."""
    inner_setup, inner_training = scripts.train_grpo.setup_model, scripts.train_grpo.run_training

    def setup_model(args, cfg, adapter_dir):
        setup = inner_setup(args, cfg, adapter_dir)
        setup["lora"]["fingerprint_at_start"] = fingerprints.pop(0)
        return setup

    def run_training(setup, families, text, cfg, gcfg, args):
        seen.setdefault("origins", []).append(setup.get("origin"))
        return inner_training(setup, families, text, cfg, gcfg, args)

    monkeypatch.setattr(scripts.train_grpo, "setup_model", setup_model)
    monkeypatch.setattr(scripts.train_grpo, "run_training", run_training)


def test_train_grpo_script_records_the_start_adapter_the_data_file_and_the_sampling_seed(
        scripts, grpo_work, monkeypatch, text, capsys):
    work = grpo_work
    seen = grpo_stand_ins(scripts, monkeypatch, crash_after=23)
    with_origin_and_fingerprint(scripts, monkeypatch, seen, [12.5, 99.0])
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 1          # the machine went away
    adapter_hashes = {"adapter_config.json": hashlib.sha256(b"{}").hexdigest(),
                      "adapter_model.safetensors": hashlib.sha256(b"sft weights").hexdigest()}
    origin = {"data_sha256": sha256_of(work / "scenarios" / "train.jsonl"), "start_adapter": adapter_hashes}
    assert seen["origins"] == [origin]
    manifest = read_json(work / "grpo_seed1" / "run_manifest.json")
    assert (manifest["seed"], manifest["sampling_run_seed"], manifest["save_tokens"]) == \
        (1, sampling_run_seed(1), False)
    assert manifest["start_adapter"]["lora_fingerprint_step0"] == 12.5
    assert manifest["starts"][0]["lora_fingerprint"] == 12.5
    assert manifest["resumed_after_fault"] == []

    seen = grpo_stand_ins(scripts, monkeypatch)
    with_origin_and_fingerprint(scripts, monkeypatch, seen, [99.0])
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 0
    assert "went on after the stop rule" not in capsys.readouterr().out
    # The restart gives the library the same origin, though the model was loaded from the checkpoint.
    assert seen["origins"] == [origin]
    manifest = read_json(work / "grpo_seed1" / "run_manifest.json")
    # The fingerprint before step 0 is still the one of the first start; each start has its own.
    assert manifest["start_adapter"]["lora_fingerprint_step0"] == 12.5
    assert [start["lora_fingerprint"] for start in manifest["starts"]] == [12.5, 99.0]
    assert [start["allow_resume_after_fault"] for start in manifest["starts"]] == [False, False]
    assert manifest["resumed_after_fault"] == [] and manifest["complete"] is True
    # A restart after a lost machine is not a restart after a stop of the stop rule: the plain status.
    assert manifest["status"] == "complete" and manifest["starts"][1]["result"] == "complete"


def test_run_training_gives_the_library_the_origin_and_the_two_switches(scripts, monkeypatch, tmp_path):
    got: dict = {}

    def library(model, tokenizer, families, text, cfg, gcfg, **options):
        got.update(options, positional=(model, tokenizer, families, text, cfg, gcfg))
        return ["the log lines"]

    monkeypatch.setattr(grpo, "train_grpo", library)
    argv = ["--model", "m", "--adapter", "a", "--scenarios", "s", "--out", str(tmp_path / "grpo_seed2"),
            "--seed", "2", "--save-tokens", "--resume", "--allow-resume-after-fault"]
    args = scripts.train_grpo.parse_args(argv)
    setup = {"model": "the model", "tokenizer": "the tokenizer",
             "origin": {"data_sha256": DATA, "start_adapter": ADAPTER}}
    gcfg = GRPOConfig(seed=2)
    logs = scripts.train_grpo.run_training(setup, ["families"], "text", "cfg", gcfg, args)
    assert logs == ["the log lines"]
    assert got == {"positional": ("the model", "the tokenizer", ["families"], "text", "cfg", gcfg),
                   "out_dir": tmp_path / "grpo_seed2", "run": "grpo_seed2", "resume": True,
                   "save_tokens": True, "allow_resume_after_fault": True, "data_sha256": DATA,
                   "start_adapter": ADAPTER}
    # The defaults: no token data, no restart, and a stopped run stays stopped.
    plain = scripts.train_grpo.parse_args(argv[:10])
    assert (plain.save_tokens, plain.resume, plain.allow_resume_after_fault) == (False, False, False)
    scripts.train_grpo.run_training(setup, [], "text", "cfg", gcfg, plain)
    assert (got["save_tokens"], got["resume"], got["allow_resume_after_fault"]) == (False, False, False)


def test_train_grpo_script_continues_a_stopped_run_only_when_told_and_says_so(scripts, grpo_work, monkeypatch,
                                                                              text, capsys):
    work = grpo_work
    out = work / "grpo_seed1"
    grpo_stand_ins(scripts, monkeypatch, fault_at=12)
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 3
    capsys.readouterr()
    stopped = read_json(out / "run_manifest.json")
    assert stopped["status"] == "stopped_by_rule" and stopped["resumed_after_fault"] == []
    reason = "harness errors: 3 of 128 episodes so far"

    # --resume alone is refused before the model loads, and the manifest still says "stopped".
    seen = grpo_stand_ins(scripts, monkeypatch)
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 2
    message = capsys.readouterr().err
    assert "the stop rule ended the run" in message and "step 12" in message and reason in message
    assert "--allow-resume-after-fault" in message
    assert seen["starts"] == [] and read_json(out / "run_manifest.json") == stopped
    # The switch without --resume is refused, in a new folder too.
    for folder in ("grpo_seed1", "another"):
        argv = train_grpo_argv(work, text, "--allow-resume-after-fault", out=folder)
        assert scripts.train_grpo.main(argv) == 2
        assert "option of --resume" in capsys.readouterr().err
    assert seen["starts"] == [] and not (work / "another").exists()
    # The same holds when stopped.json was taken away by hand: the fault line of the log still tells of
    # the stop. And when the folder tells nothing at all, the status of the earlier manifest does.
    (out / "stopped.json").rename(out / "taken_away.json")
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 2
    assert "the stop rule ended the run" in capsys.readouterr().err
    (out / "taken_away.json").rename(out / "stopped.json")
    told_by_manifest = {"status": "stopped_by_rule", "fault": {"step": 3, "reason": "a reason"}}
    with pytest.raises(scripts.sft_data.Refused, match="step 3: a reason"):
        scripts.train_grpo.stop_before(work / "empty_folder", told_by_manifest, False)
    assert scripts.train_grpo.stop_before(work / "empty_folder", told_by_manifest, True) == \
        {"step": 3, "reason": "a reason", "start": None}
    assert scripts.train_grpo.stop_before(work / "empty_folder", {"status": "failed"}, False) is None

    # With the switch the run goes on from the checkpoint after step 10, and the manifest says so.
    argv = train_grpo_argv(work, text, "--resume", "--allow-resume-after-fault")
    assert scripts.train_grpo.main(argv) == 0
    printed = capsys.readouterr().out
    assert "NOTE: this run went on after the stop rule had ended it at step 12" in printed
    assert seen["starts"][0]["adapter_dir"] == out / "checkpoints" / "step_00010" / "adapter"
    manifest = read_json(out / "run_manifest.json")
    # All steps are done, so the chain may go on ("complete": true). But the status is not the plain
    # "complete": a reader who looks at the status alone sees that the run went on after a stop.
    assert manifest["complete"] is True and manifest["status"] == "complete_after_fault"
    assert manifest["final_adapter"]["folder"] == "checkpoints/step_00080/adapter"
    assert manifest["resumed_after_fault"] == [{"step": 12, "reason": reason, "start": None}]
    first, second = manifest["starts"]
    assert (first["result"], first["fault"]) == ("stopped_by_rule", {"step": 12, "reason": reason})
    assert (first["allow_resume_after_fault"], first["resumed_after_fault"]) == (False, None)
    assert (second["result"], second["resumed_after_step"], second["allow_resume_after_fault"]) == \
        ("complete_after_fault", 10, True)
    assert second["resumed_after_fault"] == {"step": 12, "reason": reason, "start": None}
    assert "--allow-resume-after-fault" in second["command"]


def test_a_run_that_went_on_after_a_stop_keeps_that_status_through_a_later_plain_resume(
        scripts, grpo_work, monkeypatch, text, capsys):
    work = grpo_work
    out = work / "grpo_seed1"
    grpo_stand_ins(scripts, monkeypatch, fault_at=12)
    assert scripts.train_grpo.main(train_grpo_argv(work, text)) == 3
    # The allowed restart loses its machine after step 22.
    grpo_stand_ins(scripts, monkeypatch, crash_after=23)
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume", "--allow-resume-after-fault")) == 1
    assert read_json(out / "run_manifest.json")["status"] == "failed"
    # At an allowed restart the library gives stopped.json the name of its start. The stand-in for the
    # training does not, so the test does it.
    (out / "stopped.json").rename(out / "stopped.start_1.json")
    grpo_stand_ins(scripts, monkeypatch)
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 0       # no flag is needed now
    assert "NOTE: this run went on after the stop rule had ended it at step 12" in capsys.readouterr().out
    manifest = read_json(out / "run_manifest.json")
    assert (manifest["complete"], manifest["status"]) == (True, "complete_after_fault")
    assert [start["result"] for start in manifest["starts"]] == \
        ["stopped_by_rule", "failed", "complete_after_fault"]
    assert [start["allow_resume_after_fault"] for start in manifest["starts"]] == [False, True, False]
    assert manifest["final_adapter"] is not None


def test_a_resume_must_save_tokens_as_the_first_start_did(scripts, grpo_work, monkeypatch, text, capsys):
    work = grpo_work
    grpo_stand_ins(scripts, monkeypatch, crash_after=23)
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--save-tokens")) == 1
    capsys.readouterr()
    manifest = read_json(work / "grpo_seed1" / "run_manifest.json")
    assert manifest["save_tokens"] is True and manifest["identity"]["save_tokens"] is True
    seen = grpo_stand_ins(scripts, monkeypatch)
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume")) == 2
    assert "save_tokens" in capsys.readouterr().err and seen["starts"] == []
    assert scripts.train_grpo.main(train_grpo_argv(work, text, "--resume", "--save-tokens")) == 0
    capsys.readouterr()
    assert read_json(work / "grpo_seed1" / "run_manifest.json")["save_tokens"] is True


def test_the_manifest_of_train_sft_names_both_seeds_and_the_option_that_sets_each(scripts, sft_data_done,
                                                                                  monkeypatch, capsys):
    work = sft_data_done
    seen = sft_stand_ins(scripts, monkeypatch)
    argv = train_sft_argv(work)
    argv[argv.index("--seed") + 1] = "7"
    assert scripts.train_sft.main(argv) == 0
    capsys.readouterr()
    manifest = read_json(work / "sft" / "run_manifest.json")
    seeds = manifest["seeds"]
    assert sorted(seeds) == ["example_order", "lora_start_values"]
    assert (seeds["lora_start_values"]["value"], seeds["lora_start_values"]["set_by"]) == (7, "--seed")
    assert (seeds["example_order"]["value"], seeds["example_order"]["set_by"]) == (7, "--seed")
    assert "add_lora" in seeds["lora_start_values"]["used_for"]
    assert "SFTConfig.seed" in seeds["example_order"]["used_for"]
    # The two values are the ones that reached the two places: the adapter set-up and the training config.
    assert manifest["lora"]["seed"] == 7 and seen["training"]["sft_cfg"].seed == 7
    assert manifest["sft_config"]["seed"] == 7


# ---------------------------------------------------------------------------
# After the second review (2026-11-18): the steps that a restart ran again, and the sentence of sft_report.md
# ---------------------------------------------------------------------------
def test_the_script_counts_the_log_lines_of_the_steps_that_a_restart_ran_again(scripts, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    nothing = {"lines": 0, "steps": [], "episodes": 0, "harness_errors": 0, "unreadable_lines": 0}
    assert scripts.train_grpo.steps_run_again(out) == nothing
    assert begin_start(out, resume=False) == (1, [])
    for step in range(46):              # start 1: the steps 0 to 45, a fault at 45, the checkpoint after 40
        log = {"step": step, "scenarios": [f"tr-{step:03d}/s1"], "n_episodes": 2,
               "n_harness_errors": 1 if step in (3, 44) else 0, "fault": FAULT if step == 45 else None}
        record_step(out, 1, log, [hand_made_record(step, pos) for pos in range(2)], False)
    assert scripts.train_grpo.steps_run_again(out) == nothing            # no restart yet
    begin_start(out, resume=True, completed_steps=40, allow_resume_after_fault=True)
    # The steps 40 to 45 are run again. Their old lines hold 6 x 2 episodes and one harness error (step 44).
    # The harness error of step 3 is in the log that stays, so it is not counted here.
    assert scripts.train_grpo.steps_run_again(out) == {
        "lines": 6, "steps": [40, 41, 42, 43, 44, 45], "episodes": 12, "harness_errors": 1,
        "unreadable_lines": 0}
    # A restart that died in the middle of its file steps can leave these lines in the kept file twice
    # (`begin_start`). A line is counted once.
    kept_file = out / "train_log.start_1.jsonl"
    kept_file.write_bytes(kept_file.read_bytes() * 2)
    assert len(kept_file.read_text(encoding="utf-8").splitlines()) == 12
    assert scripts.train_grpo.steps_run_again(out) == {
        "lines": 6, "steps": [40, 41, 42, 43, 44, 45], "episodes": 12, "harness_errors": 1,
        "unreadable_lines": 0}
    # A line that was cut off in the middle of a write is kept as it is. It is counted as unreadable.
    with kept_file.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 46, "n_epis')
    assert scripts.train_grpo.steps_run_again(out) == {
        "lines": 6, "steps": [40, 41, 42, 43, 44, 45], "episodes": 12, "harness_errors": 1,
        "unreadable_lines": 1}


def test_the_manifest_of_a_run_without_a_restart_says_that_no_step_was_run_again(scripts, grpo_work,
                                                                                 monkeypatch, text, capsys):
    grpo_stand_ins(scripts, monkeypatch)
    assert scripts.train_grpo.main(train_grpo_argv(grpo_work, text)) == 0
    capsys.readouterr()
    manifest = read_json(grpo_work / "grpo_seed1" / "run_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["steps_run_again"] == {"lines": 0, "steps": [], "episodes": 0, "harness_errors": 0,
                                           "unreadable_lines": 0}


@pytest.mark.parametrize("allow_penalised", [False, True])
def test_sft_report_md_says_that_an_attempt_with_a_refused_call_is_not_kept(scripts, families, text, cfg,
                                                                            allow_penalised):
    """The sentence of sft_report.md about what is kept names the rule "no attempt with a refused call",
    with and without --allow-penalised. Before the repair of 2026-11-18 it said "passes without a penalty"
    and, with the option, "every pass", which was false."""
    def solved(family: dict) -> dict:
        return run_policy(Episode(family, TRAIN_STYLES[0], text, cfg), oracle(text)).record(run="test",
                                                                                           arm="test")

    first, second = families["train"][:2]
    clean = solved(first)
    # A pass with the full reward and one refused call. The count is written by hand here; real attempts of
    # this kind are played in tests/core/test_sft_select.py.
    with_refused_call = {**solved(second), "n_tool_errors": 1}
    chosen, selection = select_sft_records([clean, with_refused_call], cfg, allow_penalised=allow_penalised)
    assert chosen == [clean] and (selection["n_passed"], selection["n_with_refused_call"]) == (2, 1)
    report = {"selection": selection, "pass_rate": 1.0, "attempts_per_scenario": {"1": 2},
              "n_chosen_tokens": 0, "n_loss_tokens": 0}
    lines = scripts.sft_data.to_markdown(report).splitlines()
    sentence = next(line for line in lines if line.startswith("Input:"))
    assert "- n_with_refused_call: 1" in lines               # the count is in the file
    # A pass was left out for its refused call, so the sentence about what is kept must name that rule.
    assert "without a refused call" in sentence
    # With the option a penalty is allowed, and the sentence says so; without it, it says the opposite.
    assert ("a penalty is allowed" in sentence) == allow_penalised
    assert ("without a penalty" in sentence) == (not allow_penalised)
