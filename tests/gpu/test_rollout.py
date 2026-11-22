"""Tests of `rollout`: the glue between the sampler, the episodes and the token buffers.

The first tests replace the model by `ScriptedSampler`, which answers every prompt with the next message of a
fixed script, as token ids. They use the real tokenizer, episodes and token buffers, and show exactly which
record follows from which sampled tokens. The last tests use the real model.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from accessdesk import impl
from accessdesk.config import RunConfig
from accessdesk.harness.episode import Episode
from accessdesk.harness.hf_backend import SampledTurn, Sampler, rollout
from accessdesk.harness.policies import oracle, run_policy
from accessdesk.harness.records import replay
from accessdesk.rng import derive_seed
from accessdesk.types import Segment

META = {"run": "test", "arm": "scripted"}


def oracle_texts(family: dict, style: int, text, cfg: RunConfig) -> list[str]:
    """The assistant messages of the scripted oracle on one scenario."""
    episode = run_policy(Episode(family, style, text, cfg), oracle(text))
    return [turn["assistant_text"] for turn in episode.turns]


class ScriptedSampler:
    """Stands in for the model. A prompt is recognised by its first prompt; the turn number is read from the
    prompt itself: every finished turn adds two end-of-turn tokens (the assistant's and the tool result's)."""

    def __init__(self, tokenizer, cfg: RunConfig, layout, cut_off_first_turn: bool = False) -> None:
        self.tokenizer, self.cfg = tokenizer, cfg
        self.stop_id, self.pad_id = layout.im_end_id, layout.pad_id
        self.scripts: dict[tuple, list[str]] = {}
        self.cut_off_first_turn = cut_off_first_turn
        self.seeds: list[int] = []
        self.batch_sizes: list[int] = []

    def add_script(self, first_prompt: list[int], texts: list[str]) -> None:
        self.scripts[tuple(first_prompt)] = texts

    def sample(self, prompts: list[list[int]], seed: int) -> list[SampledTurn]:
        self.seeds.append(seed)
        self.batch_sizes.append(len(prompts))
        return [self._turn(prompt) for prompt in prompts]

    def _turn(self, prompt: list[int]) -> SampledTurn:
        first = next(key for key in self.scripts if tuple(prompt[:len(key)]) == key)
        turn = prompt[len(first):].count(self.stop_id) // 2
        if self.cut_off_first_turn and turn == 0:             # a message that hit the token cap: no end token
            ids = self.tokenizer.encode(self.scripts[first][0], add_special_tokens=False)[:8]
            return SampledTurn(ids=ids, logprobs=[-9.0] * len(ids), truncated=True)
        text = self.scripts[first][turn - 1 if self.cut_off_first_turn else turn]
        ids = self.tokenizer.encode(text, add_special_tokens=False) + [self.stop_id]
        return SampledTurn(ids=ids, logprobs=[-(turn + j / 1000) for j in range(len(ids))], truncated=False)


def scripted_setup(tokenizer, layout, text, families, cfg, scenarios, **sampler_options):
    """Fresh episodes for `scenarios` = [(index of a dev family, style), ...]; and their sampler."""
    sampler = ScriptedSampler(tokenizer, cfg, layout, **sampler_options)
    episodes = []
    for index, style in scenarios:
        family = families["dev"][index]
        episode = Episode(family, style, text, cfg)
        sampler.add_script(layout.first_prompt(episode.system_prompt, episode.request_text),
                           oracle_texts(family, style, text, cfg))
        episodes.append(episode)
    return sampler, episodes


def check_tokens(record: dict, stop_id: int) -> None:
    """What must hold for the token data of every episode without a harness error."""
    tokens = record["tokens"]
    segments = [Segment(*segment) for segment in tokens["segments"]]
    assert segments[0].kind == "prompt" and segments[0].start == 0
    assert all(a.end == b.start for a, b in zip(segments, segments[1:]))
    assert segments[-1].end == len(tokens["ids"])
    sampled = [s for s in segments if s.kind == "sampled"]
    assert len(sampled) == record["n_turns"] == len(record["turns"])
    assert sum(s.end - s.start for s in sampled) == len(tokens["logprobs"])
    assert [s.end - s.start for s in sampled] == [turn["n_tokens"] for turn in record["turns"]]
    assert tokens["ids"][-1] == stop_id                                   # the episode ends with an end token
    mask = impl.get("build_loss_mask")(len(tokens["ids"]), segments)
    assert sum(mask) == len(tokens["logprobs"])                    # one log-probability per trained token


# ---------------------------------------------------------------------------
# With the scripted sampler (no model)
# ---------------------------------------------------------------------------
def test_scripted_rollout_gives_the_oracles_episodes(tokenizer, layout, text, families):
    cfg = dataclasses.replace(RunConfig(), chunk_size=3)
    scenarios = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 1), (5, 2), (6, 3)]              # three chunks: 3, 3, 1
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, scenarios)
    records = rollout(sampler, {"five": layout}, episodes, run_seed=7, pass_index=2, meta=META)
    assert len(records) == len(scenarios)
    for position, (record, (index, style)) in enumerate(zip(records, scenarios)):
        family = families["dev"][index]
        script = oracle_texts(family, style, text, cfg)
        assert record["scenario_id"] == f"{family['family_id']}/s{style}"
        assert (record["run"], record["arm"], record["sample"]) == ("test", "scripted", 0)
        assert record["seed"] == {"run_seed": 7, "pass": 2, "chunk": position // 3, "pos": position % 3}
        assert record["harness_error"] is None
        assert [turn["assistant_text"] for turn in record["turns"]] == script     # decode gives the text back
        assert (record["passed"], record["reward"], record["n_malformed"]) == (True, 100, 0)
        assert replay(record, family, text, cfg) == record["final_state"]
        check_tokens(record, layout.im_end_id)
        # The log-probabilities are in sampling order: turn k, token j was given -(k + j / 1000).
        lengths = [turn["n_tokens"] for turn in record["turns"]]
        expected = [-(k + j / 1000) for k, length in enumerate(lengths) for j in range(length)]
        assert record["tokens"]["logprobs"] == expected


def test_one_sample_call_per_chunk_and_turn_with_the_derived_seed(tokenizer, layout, text, families):
    cfg = dataclasses.replace(RunConfig(), chunk_size=2)
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, [(0, 1), (1, 1), (2, 1)])
    records = rollout(sampler, {"five": layout}, episodes, run_seed=3, pass_index=5, meta=META)
    turns = [record["n_turns"] for record in records]
    expected_seeds, expected_sizes = [], []
    for chunk, members in enumerate([turns[:2], turns[2:]]):
        for turn in range(max(members)):
            expected_seeds.append(derive_seed(3, 5, chunk, turn))
            expected_sizes.append(sum(n > turn for n in members))                 # finished episodes drop out
    assert sampler.seeds == expected_seeds
    assert sampler.batch_sizes == expected_sizes


def test_samples_of_one_scenario_are_numbered_in_order(tokenizer, layout, text, families):
    cfg = dataclasses.replace(RunConfig(), chunk_size=4)
    scenarios = [(0, 1), (1, 1), (0, 1), (0, 1), (1, 1), (0, 2)]
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, scenarios)
    records = rollout(sampler, {"five": layout}, episodes, run_seed=0, pass_index=0, meta=META)
    assert [record["sample"] for record in records] == [0, 0, 1, 2, 1, 0]
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, scenarios)
    records = rollout(sampler, {"five": layout}, episodes, run_seed=0, pass_index=0,
                      meta={**META, "first_sample": 4})
    assert [record["sample"] for record in records] == [4, 4, 5, 6, 5, 4]
    assert all("first_sample" not in record for record in records)


def test_a_turn_cut_off_by_the_token_cap_is_malformed_and_gets_a_forced_end_token(tokenizer, layout, text,
                                                                                    families):
    cfg = RunConfig()
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, [(0, 1)],
                                       cut_off_first_turn=True)
    (record,) = rollout(sampler, {"five": layout}, episodes, run_seed=0, pass_index=0, meta=META)
    first = record["turns"][0]
    assert (first["truncated"], first["parse_ok"], first["parse_error"]) == (True, False, "truncated")
    assert first["n_tokens"] == 8
    assert record["harness_error"] is None and record["n_malformed"] == 1
    assert (record["passed"], record["reward"]) == (True, 100 - cfg.malformed_penalty)
    kinds = [segment[0] for segment in record["tokens"]["segments"]]
    assert kinds[:4] == ["prompt", "sampled", "forced_eos", "suffix"] and kinds.count("forced_eos") == 1
    forced = record["tokens"]["segments"][2]
    assert forced[2] - forced[1] == 1 and record["tokens"]["ids"][forced[1]] == layout.im_end_id
    check_tokens(record, layout.im_end_id)
    assert record["tokens"]["logprobs"][:8] == [-9.0] * 8


def test_an_exception_in_one_episode_is_a_harness_error_and_the_others_go_on(tokenizer, layout, text,
                                                                              families):
    cfg = dataclasses.replace(RunConfig(), chunk_size=3)
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, [(0, 1), (1, 1), (2, 1)])
    broken = episodes[1]
    original_step = broken.step

    def step_that_fails_once(*args, **kwargs):
        broken.step = original_step
        raise RuntimeError("made-up failure")

    broken.step = step_that_fails_once
    records = rollout(sampler, {"five": layout}, episodes, run_seed=0, pass_index=0, meta=META)
    assert [record["harness_error"] is None for record in records] == [True, False, True]
    assert "turn 0" in records[1]["harness_error"] and "made-up failure" in records[1]["harness_error"]
    assert records[1]["scenario_id"] == f"{families['dev'][1]['family_id']}/s1"      # still a whole record
    assert all(record["passed"] and record["reward"] == 100 for record in (records[0], records[2]))


def test_a_failed_sample_call_fails_its_chunk_only(tokenizer, layout, text, families):
    cfg = dataclasses.replace(RunConfig(), chunk_size=2)
    scenarios = [(0, 1), (1, 1), (2, 1), (3, 1)]
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, scenarios)
    good_sample = sampler.sample

    def sample_that_fails_in_chunk_0(prompts, seed):
        if seed == derive_seed(0, 0, 0, 1):
            raise RuntimeError("made-up out of memory")
        return good_sample(prompts, seed)

    sampler.sample = sample_that_fails_in_chunk_0
    records = rollout(sampler, {"five": layout}, episodes, run_seed=0, pass_index=0, meta=META)
    assert [record["harness_error"] is None for record in records] == [False, False, True, True]
    assert all("sample call" in record["harness_error"] for record in records[:2])
    assert all(record["passed"] for record in records[2:])


def test_rollout_refuses_a_layout_with_another_stop_token(tokenizer, layout, text, families):
    cfg = RunConfig()
    sampler, episodes = scripted_setup(tokenizer, layout, text, families, cfg, [(0, 1)])
    sampler.stop_id = layout.pad_id
    with pytest.raises(ValueError):
        rollout(sampler, {"five": layout}, episodes, run_seed=0, pass_index=0, meta=META)


# ---------------------------------------------------------------------------
# With the model
# ---------------------------------------------------------------------------
def _model_episodes(text, families, cfg) -> list[Episode]:
    """Ten episodes: three dev families in every wording style, with one scenario twice."""
    chosen = [(family, wording["style"]) for family in families["dev"][:3] for wording in family["wordings"]]
    chosen = chosen[:9]
    chosen.append(chosen[0])
    return [Episode(family, style, text, cfg) for family, style in chosen]


def _without_logprobs(records: list[dict]) -> str:
    slim = [{**record, "tokens": {key: value for key, value in record["tokens"].items() if key != "logprobs"}}
            for record in records]
    return json.dumps(slim, ensure_ascii=False)


def test_rollout_with_the_model(model, tokenizer, layout, text, families):
    from accessdesk.train.forward import token_logprobs

    cfg = dataclasses.replace(RunConfig(), chunk_size=4, max_new_tokens=96)
    sampler = Sampler(model, tokenizer, cfg)
    family_of = {family["family_id"]: family for family in families["dev"]}
    records = rollout(sampler, {"five": layout}, _model_episodes(text, families, cfg), run_seed=1,
                      pass_index=0, meta={"run": "test", "arm": "base"})
    assert len(records) == 10
    assert [record["seed"]["chunk"] for record in records] == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert records[9]["scenario_id"] == records[0]["scenario_id"]
    assert (records[0]["sample"], records[9]["sample"]) == (0, 1)
    differences = []
    for record in records:
        assert record["harness_error"] is None
        assert 1 <= record["n_turns"] <= cfg.turn_limit
        check_tokens(record, layout.im_end_id)
        assert replay(record, family_of[record["family_id"]], text, cfg) == record["final_state"]
        for turn in record["turns"]:                                    # a truncated turn has exactly the cap
            at_cap_without_end = turn["n_tokens"] == cfg.max_new_tokens and turn["parse_error"] == "truncated"
            assert turn["truncated"] == at_cap_without_end
        # Parity on whole episodes: the stored ids with the loss mask that training uses, against the sampler.
        tokens = record["tokens"]
        segments = [Segment(*segment) for segment in tokens["segments"]]
        mask = impl.get("build_loss_mask")(len(tokens["ids"]), segments)
        (trained,) = token_logprobs(model, [tokens["ids"]], [mask], pad_id=sampler.pad_id, micro_batch=1,
                                    with_grad=False)
        differences.extend(a - b for a, b in zip(trained.tolist(), tokens["logprobs"]))
    values = sorted(abs(d) for d in differences)
    stats = {"n": len(values), "mean": sum(values) / len(values),
             "p99": values[int(0.99 * (len(values) - 1))], "max": values[-1]}
    passed = sum(record["passed"] for record in records)
    n_messages = sum(record["n_turns"] for record in records)
    malformed = sum(record["n_malformed"] for record in records) / n_messages
    print(f"\nwhole episodes, sampler against trainer: {stats}")
    print(f"passed {passed}/10, malformed messages {malformed:.2f}")
    # Measured on 2026-11-17 (RTX 4090), ten whole episodes, sampled in chunks of 4, scored one by one:
    #   2B: 2073 tokens, mean 0.0017, 99th percentile 0.046, largest 0.24
    #   4B: 1965 tokens, mean 0.0015, 99th percentile 0.042, largest 0.37
    assert stats["mean"] < 0.02 and stats["max"] < 1.0


def test_same_seed_gives_the_same_episodes(model, tokenizer, layout, text, families):
    cfg = dataclasses.replace(RunConfig(), chunk_size=4, max_new_tokens=96)
    sampler = Sampler(model, tokenizer, cfg)

    def run(run_seed: int) -> str:
        records = rollout(sampler, {"five": layout}, _model_episodes(text, families, cfg), run_seed=run_seed,
                          pass_index=0, meta={"run": "test", "arm": "base"})
        return _without_logprobs(records)

    assert run(1) == run(1)


# ---------------------------------------------------------------------------
# scripts/gpu/rollout.py with the model: the run manifest names the code that made the run
# ---------------------------------------------------------------------------
def test_the_run_manifest_of_the_rollout_script_names_the_code_that_made_it(model_and_tokenizer, monkeypatch,
                                                                            tmp_path):
    """Written on 2026-11-18 on the laptop and not run there (no GPU). One real pass of the script's main():
    one calibration family in one wording, one attempt. The script's own load of the model is replaced by
    the model of this session, so that the GPU holds one model; everything else is the real code.

    The run is started with --allow-changed-tree, so that the test runs in every state of the tree: not
    frozen, as frozen, and frozen with a change. The stop without that option is tested without a model,
    in tests/core/test_scripts.py."""
    import importlib.util
    import os
    import re
    from pathlib import Path

    from accessdesk import freeze
    from accessdesk.harness import hf_backend
    from tests.gpu.conftest import DEFAULT_MODEL

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("rollout_script_of_the_gpu_test",
                                                  root / "scripts" / "gpu" / "rollout.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(hf_backend, "load_model", lambda *args, **kwargs: model_and_tokenizer)
    monkeypatch.setattr(script, "SMOKE_DIR", tmp_path)           # the placeholder text may write here
    out = tmp_path / "run"
    code = script.main(["--model", os.environ.get("ACCESSDESK_TEST_MODEL", DEFAULT_MODEL),
                        "--text", str(root / "text" / "placeholder_text.yaml"), "--allow-placeholder",
                        "--scenarios", str(root / "data" / "scenarios_placeholder"), "--split", "calib",
                        "--limit", "1", "--styles", "1", "--samples", "1", "--max-new-tokens", "96",
                        "--allow-changed-tree", "--out", str(out)])
    assert code == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True and (manifest["scenarios"], manifest["samples"]) == (1, 1)
    # The code hash: 64 hexadecimal digits, the hash that the freeze module gives for this tree.
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["code_sha256"])
    assert manifest["code_sha256"] == freeze.code_sha256() == manifest["frozen"]["code_sha256"]
    # The whole state of the freeze, as the freeze module gives it now, and the option that was used.
    assert manifest["frozen"] == freeze.frozen_state()
    assert manifest["allow_changed_tree"] is True
