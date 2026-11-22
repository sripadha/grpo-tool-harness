"""The two chain scripts (scripts/gpu/chain_stage1.sh, chain_stage2.sh) and the two small scripts that they
call (scripts/sft_profile.py, scripts/pick_overfit_families.py), without torch and without a model.

What is tested here, and what is not:
    sft_profile.py              the pick of the 100 families; the profile and the rule on hand-made records on
                                both sides of 0.8 and at exactly 0.8, with the numbers worked out in the
                                comments; every refusal
    pick_overfit_families.py    the rule on hand-made records; too few mixed families; the refusals
    the chain scripts           `bash -n`; the decided values at their top; they name only options that the
                                Python scripts have; and runs of each chain with a stand-in for `python`:
                                the order of the steps, what each step is given, a second start that skips
                                the complete steps, a second start with another value that is refused, a
                                failed step, a run that the stop rule ended, the two checks before the seeds
The stand-in runs the chain's own snippets and the two small scripts for real. It replaces every script
that needs a model, the table and the monitor report: it writes what the real script leaves for the chain
(a run manifest with the real key names, episodes, adapters, checkpoints). So these tests show what the
chains start and in which order. They do not show that a rollout or a training works.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from accessdesk.config import RunConfig
from accessdesk.env.scenarios import save_families
from accessdesk.harness.records import read_records, write_records
from accessdesk.train.grpo import GRPOConfig
from accessdesk.types import OUTCOMES

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
GPU_SCRIPTS = SCRIPTS / "gpu"
CHAINS = ("chain_stage1.sh", "chain_stage2.sh")
PLACEHOLDER_SCENARIOS = ROOT / "data" / "scenarios_placeholder"
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"          # the 2B, typed by hand from the task
# The words of the rule, as the task gives them.
RULE = "median above 0.8 -> groups of 16 on 8 scenarios per step; else groups of 8 on 16"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scripts() -> SimpleNamespace:
    """The two small scripts as modules, imported by name from scripts/ (pick_overfit_families.py imports
    sft_profile.py in the same way)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return SimpleNamespace(profile=importlib.import_module("sft_profile"),
                           overfit=importlib.import_module("pick_overfit_families"))


# ---------------------------------------------------------------------------
# Hand-made inputs: a scenario folder with the train split, and a rollout on some of its families
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scenario_dir(tmp_path_factory, families) -> Path:
    """A folder with train.jsonl (the 200 train families of the tests) and a summary.json that names its
    SHA-256, as scripts/build_scenarios.py leaves them."""
    folder = tmp_path_factory.mktemp("chain_scripts") / "scenarios"
    folder.mkdir()
    save_families(folder / "train.jsonl", families["train"])
    (folder / "summary.json").write_text(
        json.dumps({"files_sha256": {"train.jsonl": sha256_of(folder / "train.jsonl")}}), encoding="utf-8")
    return folder


def attempts(family: dict, style: int, passes: int, n: int, *, data_sha256: str, arm: str = "sft_train",
             split: str = "train", errors: int = 0) -> list[dict]:
    """`n` episode records of one scenario, with the keys that the two scripts read. The first `passes`
    attempts pass, the others fail; the last `errors` attempts are harness errors."""
    records = []
    for sample in range(n):
        records.append({
            "run": arm, "arm": arm, "split": split, "family_id": family["family_id"], "style": style,
            "scenario_id": f"{family['family_id']}/s{style}", "outcome": family["outcome"],
            "ruleset": "default", "toolset": "five", "prompt": "final", "sample": sample,
            "passed": sample < passes, "harness_error": "made up" if sample >= n - errors else None,
            "data_sha256": data_sha256})
    return records


def write_rollout(folder: Path, records: list[dict], *, adapter: str | None = "results/x/sft/adapter",
                  complete: bool = True, split: str = "train", data_sha256: str = "") -> Path:
    """episodes.jsonl and run_manifest.json as scripts/gpu/rollout.py leaves them (the keys that are read).
    Returns the episodes file."""
    folder.mkdir(parents=True, exist_ok=True)
    write_records(folder / "episodes.jsonl", records)
    scenarios = {record["scenario_id"] for record in records}
    (folder / "run_manifest.json").write_text(json.dumps({
        "complete": complete, "model": MODEL, "revision": REVISION, "adapter": adapter, "split": split,
        "samples": len(records) // len(scenarios), "scenarios": len(scenarios), "run_seed": 20,
        "split_file_sha256": data_sha256, "text_sha256": "t" * 64, "config_sha256": "c" * 64,
        "config": {"max_new_tokens": 160}}), encoding="utf-8")
    return folder / "episodes.jsonl"


def profile_rollout(tmp_path: Path, scenario_dir: Path, families: dict, pass_counts: list[int],
                    n: int = 10, **manifest) -> Path:
    """A rollout of `n` attempts in style 1 on the first families of the train split: family i passes
    `pass_counts[i]` of them. So the pass rate of family i is pass_counts[i] / n."""
    data_sha256 = sha256_of(scenario_dir / "train.jsonl")
    records = [record for family, passes in zip(families["train"], pass_counts)
               for record in attempts(family, 1, passes, n, data_sha256=data_sha256)]
    return write_rollout(tmp_path / "sft_train", records, data_sha256=data_sha256, **manifest)


# ---------------------------------------------------------------------------
# sft_profile.py: the pick
# ---------------------------------------------------------------------------
def test_importing_the_two_scripts_loads_no_torch(scripts):
    assert "torch" not in sys.modules


def test_the_profile_families_are_the_first_25_of_each_outcome(scripts, scenario_dir, families, tmp_path,
                                                               capsys):
    target = tmp_path / "sft_profile" / "families.txt"
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--pick", str(target)]) == 0
    ids = target.read_text(encoding="utf-8").split()
    train = families["train"]
    # Counted here in another way: per outcome, the first 25 families of the file.
    wanted = {family["family_id"] for outcome in OUTCOMES
              for family in [f for f in train if f["outcome"] == outcome][:25]}
    assert len(ids) == 100 and set(ids) == wanted
    assert ids == [family["family_id"] for family in train if family["family_id"] in wanted]   # file order
    assert target.read_bytes().endswith(b"\n") and "wrote 100 family ids" in capsys.readouterr().out
    # The pick writes nothing else, and the same call gives the same file.
    assert [path.name for path in target.parent.iterdir()] == ["families.txt"]
    before = target.read_bytes()
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--pick", str(target)]) == 0
    assert target.read_bytes() == before


def test_the_pick_takes_the_number_per_outcome_that_is_asked_for(scripts, scenario_dir, families, tmp_path,
                                                                 capsys):
    target = tmp_path / "families.txt"
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--per-outcome", "2",
                                 "--pick", str(target)]) == 0
    ids = target.read_text(encoding="utf-8").split()
    by_id = {family["family_id"]: family["outcome"] for family in families["train"]}
    assert len(ids) == 8 and sorted(by_id[i] for i in ids) == sorted(list(OUTCOMES) * 2)
    # The split has 50 families of each outcome: 51 cannot be picked, and no file is written.
    other = tmp_path / "too_many.txt"
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--per-outcome", "51",
                                 "--pick", str(other)]) == 2
    assert "fewer than 51 families" in capsys.readouterr().err and not other.exists()


def test_the_pick_refuses_a_train_file_that_was_changed_after_the_build(scripts, scenario_dir, tmp_path,
                                                                        capsys):
    changed = tmp_path / "scenarios"
    shutil.copytree(scenario_dir, changed)
    with (changed / "train.jsonl").open("ab") as handle:
        handle.write(b"\n")
    assert scripts.profile.main(["--scenarios", str(changed), "--pick", str(tmp_path / "f.txt")]) == 2
    assert "not the file that the build wrote" in capsys.readouterr().err
    assert not (tmp_path / "f.txt").exists()


# ---------------------------------------------------------------------------
# sft_profile.py: the profile and the rule
# ---------------------------------------------------------------------------
# Five families with 10 attempts each, so a family's pass rate is its pass count / 10. The median of five
# numbers is the third one in sorted order.
#   pass counts          sorted rates              median   above 0.8?   step (scenarios x attempts)
#   10, 9, 9, 0, 5       0.0 0.5 0.9 0.9 1.0       0.9      yes          8 x 16
#   10, 9, 8, 0, 5       0.0 0.5 0.8 0.9 1.0       0.8      no (equal)   16 x 8
#   10, 9, 7, 0, 5       0.0 0.5 0.7 0.9 1.0       0.7      no           16 x 8
# In all three rows one family always passes (10), one always fails (0) and three are mixed.
@pytest.mark.parametrize("pass_counts, median, scenarios_per_step, group_size", [
    ([10, 9, 9, 0, 5], 0.9, 8, 16),
    ([10, 9, 8, 0, 5], 0.8, 16, 8),
    ([10, 9, 7, 0, 5], 0.7, 16, 8),
])
def test_the_rule_on_both_sides_of_0_8(scripts, scenario_dir, families, tmp_path, capsys, pass_counts,
                                       median, scenarios_per_step, group_size):
    episodes = profile_rollout(tmp_path, scenario_dir, families, pass_counts)
    out = tmp_path / "sft_profile"
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--episodes", str(episodes),
                                 "--out", str(out)]) == 0
    decision = read_json(out / "decision.json")
    assert decision["median_family_pass_rate"] == median
    assert decision["rule"] == RULE
    assert (decision["scenarios_per_step"], decision["group_size"]) == (scenarios_per_step, group_size)
    assert decision["n_families"] == 5
    assert (decision["n_always_pass"], decision["n_always_fail"], decision["n_mixed"]) == (1, 1, 3)
    assert decision["episodes"] == 50 and decision["harness_errors"] == 0
    assert decision["attempts_per_scenario"] == {"10": 5}
    # Every family is in the file with its own numbers, so that the median can be counted again.
    rows = decision["families"]
    assert [rows[family["family_id"]]["passed"] for family in families["train"][:5]] == pass_counts
    assert sorted(row["pass_rate"] for row in rows.values())[2] == median
    assert {row["episodes"] for row in rows.values()} == {10}
    # Where the numbers come from.
    source = decision["source"]
    assert source["episodes"]["sha256"] == sha256_of(episodes)
    assert source["run_manifest"]["sha256"] == sha256_of(episodes.parent / "run_manifest.json")
    assert (source["arm"], source["run_seed"]) == ("sft_train", 20)
    assert source["adapter"] == "results/x/sft/adapter"
    assert source["adapter_sha256"] is None            # that adapter folder is not on this machine
    assert decision["script"] == {"path": "scripts/sft_profile.py",
                                  "sha256": sha256_of(SCRIPTS / "sft_profile.py"),
                                  "bytes": (SCRIPTS / "sft_profile.py").stat().st_size}
    table = (out / "profile.md").read_text(encoding="utf-8")
    assert f"groups of {group_size} attempts on {scenarios_per_step} scenarios per step" in table
    assert "| all | 5 | 1 | 1 | 3 |" in table and table in capsys.readouterr().out
    # The same input gives the same bytes: the file holds no time.
    before = (out / "decision.json").read_bytes()
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--episodes", str(episodes),
                                 "--out", str(out)]) == 0
    assert (out / "decision.json").read_bytes() == before


def test_a_median_of_exactly_0_8_between_two_families_gives_groups_of_8(scripts, scenario_dir, families,
                                                                        tmp_path, capsys):
    """Four families with 20 attempts: 0, 15, 17 and 20 passes. The median of four numbers is the mean of
    the two in the middle: (15/20 + 17/20) / 2 = 32/40 = 4/5, exactly 0.8, which is not above 0.8. With
    floating-point numbers 0.75 + 0.85 need not be 1.6; the script counts with whole fractions."""
    episodes = profile_rollout(tmp_path, scenario_dir, families, [0, 15, 17, 20], n=20)
    out = tmp_path / "sft_profile"
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--episodes", str(episodes),
                                 "--out", str(out)]) == 0
    decision = read_json(out / "decision.json")
    assert decision["median_as_fraction"] == "4/5" and decision["median_family_pass_rate"] == 0.8
    assert (decision["scenarios_per_step"], decision["group_size"]) == (16, 8)
    # One pass more in one of the two middle families, and the median is above 0.8: 33/40.
    episodes = profile_rollout(tmp_path / "more", scenario_dir, families, [0, 16, 17, 20], n=20)
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--episodes", str(episodes),
                                 "--out", str(out)]) == 0
    decision = read_json(out / "decision.json")
    assert decision["median_as_fraction"] == "33/40"
    assert (decision["scenarios_per_step"], decision["group_size"]) == (8, 16)


def test_a_family_rate_is_taken_over_its_wordings_and_attempts_without_harness_errors(scripts, scenario_dir,
                                                                                      families, tmp_path,
                                                                                      capsys):
    """Three families in the three training wordings, 8 attempts each (24 episodes a family):
        family 0   8, 8 and 4 passes: 20 of 24 = 0.8333, mixed (it always passes in two wordings)
        family 1   8, 8 and 8 passes, but 2 attempts of the last wording are harness errors: 22 valid
                   episodes, all passed: 1.0, always pass
        family 2   0, 0 and 0 passes: 0.0, always fail
    The median of 0.0, 0.8333 and 1.0 is 20/24 = 5/6, above 0.8."""
    data_sha256 = sha256_of(scenario_dir / "train.jsonl")
    first, second, third = families["train"][:3]
    records = []
    for family, per_style, errors in ((first, (8, 8, 4), 0), (second, (8, 8, 8), 2), (third, (0, 0, 0), 0)):
        for style, passes in zip((1, 2, 3), per_style):
            records += attempts(family, style, passes, 8, data_sha256=data_sha256,
                                errors=errors if style == 3 else 0)
    episodes = write_rollout(tmp_path / "sft_train", records, data_sha256=data_sha256)
    out = tmp_path / "sft_profile"
    assert scripts.profile.main(["--scenarios", str(scenario_dir), "--episodes", str(episodes),
                                 "--out", str(out)]) == 0
    decision = read_json(out / "decision.json")
    rows = decision["families"]
    assert (rows[first["family_id"]]["passed"], rows[first["family_id"]]["episodes"]) == (20, 24)
    assert rows[first["family_id"]]["profile"] == "mixed"
    assert (rows[second["family_id"]]["passed"], rows[second["family_id"]]["episodes"]) == (22, 22)
    assert rows[second["family_id"]]["profile"] == "always_pass"
    assert rows[third["family_id"]]["profile"] == "always_fail"
    assert decision["median_as_fraction"] == "5/6" and decision["median_family_pass_rate"] == 20 / 24
    assert (decision["episodes"], decision["harness_errors"]) == (72, 2)
    assert (decision["scenarios_per_step"], decision["group_size"]) == (8, 16)
    # The counts per outcome add up to the counts of all families.
    assert sum(row["n_families"] for row in decision["by_outcome"].values()) == 3
    assert decision["pass_rate"] == pytest.approx((20 / 24 + 1.0 + 0.0) / 3)


def test_the_two_steps_of_the_rule_are_the_plans_step_and_its_other_form(scripts):
    """The default step of the plan is GRPOConfig's: 16 scenarios x 8 attempts. The rule's other step has
    groups twice as large on half as many scenarios, so both sample 128 episodes."""
    defaults = GRPOConfig()
    assert scripts.profile.SMALL_GROUPS == (defaults.group_size, defaults.n_scenarios) == (8, 16)
    assert scripts.profile.LARGE_GROUPS == (2 * defaults.group_size, defaults.n_scenarios // 2) == (16, 8)
    assert scripts.profile.RULE == RULE and scripts.profile.THRESHOLD == Fraction(8, 10)


def refused(scripts, capsys, argv: list[str], out: Path) -> str:
    """The call ends with exit code 2 and writes nothing. Returns its message."""
    assert scripts.profile.main(argv) == 2
    assert not out.exists()
    return capsys.readouterr().err


def test_the_profile_refuses_episodes_of_the_untrained_model(scripts, scenario_dir, families, tmp_path,
                                                             capsys):
    """The records do not say whether the model had an adapter; the run manifest of the rollout does."""
    episodes = profile_rollout(tmp_path, scenario_dir, families, [10, 9, 9, 0, 5], adapter=None)
    out = tmp_path / "sft_profile"
    message = refused(scripts, capsys, ["--scenarios", str(scenario_dir), "--episodes", str(episodes),
                                        "--out", str(out)], out)
    assert "names no adapter" in message
    (episodes.parent / "run_manifest.json").unlink()
    message = refused(scripts, capsys, ["--scenarios", str(scenario_dir), "--episodes", str(episodes),
                                        "--out", str(out)], out)
    assert "run_manifest.json is missing" in message


def test_the_profile_refuses_records_that_are_not_of_the_train_split(scripts, scenario_dir, families,
                                                                     tmp_path, capsys):
    data_sha256 = sha256_of(scenario_dir / "train.jsonl")
    out = tmp_path / "sft_profile"
    argv = ["--scenarios", str(scenario_dir), "--out", str(out), "--episodes"]
    # Every record is of dev.
    dev = [record for family in families["dev"][:2]
           for record in attempts(family, 1, 5, 8, data_sha256=data_sha256, split="dev")]
    episodes = write_rollout(tmp_path / "dev", dev, split="dev", data_sha256=data_sha256)
    assert "records of the split dev" in refused(scripts, capsys, [*argv, str(episodes)], out)
    # One dev record among train records, with a manifest that says train.
    train = [record for family in families["train"][:2]
             for record in attempts(family, 1, 5, 8, data_sha256=data_sha256)]
    episodes = write_rollout(tmp_path / "one_dev", train[:-1] + [{**train[-1], "split": "dev"}],
                             data_sha256=data_sha256)
    assert "records of the split dev" in refused(scripts, capsys, [*argv, str(episodes)], out)


def test_the_profile_refuses_inputs_that_do_not_belong_together(scripts, scenario_dir, families, tmp_path,
                                                                capsys):
    data_sha256 = sha256_of(scenario_dir / "train.jsonl")
    out = tmp_path / "sft_profile"
    argv = ["--scenarios", str(scenario_dir), "--out", str(out), "--episodes"]
    records = [record for family in families["train"][:3]
               for record in attempts(family, 1, 5, 8, data_sha256=data_sha256)]
    # A rollout that is not complete.
    episodes = write_rollout(tmp_path / "a", records, complete=False, data_sha256=data_sha256)
    assert "not complete" in refused(scripts, capsys, [*argv, str(episodes)], out)
    # Records of two arms in one file.
    two_arms = records[:-8] + [{**record, "arm": "base_train"} for record in records[-8:]]
    episodes = write_rollout(tmp_path / "b", two_arms, data_sha256=data_sha256)
    assert "2 runs or set-ups" in refused(scripts, capsys, [*argv, str(episodes)], out)
    # Records that name another scenario file.
    episodes = write_rollout(tmp_path / "c", [{**r, "data_sha256": "0" * 64} for r in records],
                             data_sha256=data_sha256)
    assert "another scenario file" in refused(scripts, capsys, [*argv, str(episodes)], out)
    # A file that lost its last record: the manifest names 3 scenarios with 8 attempts.
    episodes = write_rollout(tmp_path / "d", records, data_sha256=data_sha256)
    write_records(episodes, records[:-1])
    assert "holds 23 records" in refused(scripts, capsys, [*argv, str(episodes)], out)
    # Records that are not of the families of the families file: one family too few, then one too many.
    episodes = write_rollout(tmp_path / "e", records, data_sha256=data_sha256)
    ids = [family["family_id"] for family in families["train"][:4]]
    wanted = tmp_path / "families.txt"
    wanted.write_text("\n".join(ids) + "\n", encoding="utf-8")
    message = refused(scripts, capsys, [*argv, str(episodes), "--families", str(wanted)], out)
    assert "1 families of the file have no record" in message
    wanted.write_text("\n".join(ids[:2]) + "\n", encoding="utf-8")
    message = refused(scripts, capsys, [*argv, str(episodes), "--families", str(wanted)], out)
    assert "1 families of the records are not in the file" in message
    # With the right file the same records are taken.
    wanted.write_text("\n".join(ids[:3]) + "\n", encoding="utf-8")
    assert scripts.profile.main([*argv, str(episodes), "--families", str(wanted)]) == 0
    assert read_json(out / "decision.json")["source"]["families_file"]["sha256"] == sha256_of(wanted)


# ---------------------------------------------------------------------------
# pick_overfit_families.py
# ---------------------------------------------------------------------------
def overfit_rollout(tmp_path: Path, scenario_dir: Path, families: dict, style_1: list[int],
                    errors: dict[int, int] | None = None) -> Path:
    """A rollout of the untrained model, 8 attempts, on the first families of the train split: family i
    passes `style_1[i]` attempts in style 1 and all 8 in style 2. `errors`: family index -> how many of its
    style-1 attempts are harness errors."""
    data_sha256 = sha256_of(scenario_dir / "train.jsonl")
    records = []
    for index, (family, passes) in enumerate(zip(families["train"], style_1)):
        records += attempts(family, 1, passes, 8, data_sha256=data_sha256, arm="base_train",
                            errors=(errors or {}).get(index, 0))
        records += attempts(family, 2, 8, 8, data_sha256=data_sha256, arm="base_train")
    return write_rollout(tmp_path / "base_train", records, adapter=None, data_sha256=data_sha256)


def test_the_overfit_families_are_the_two_mixed_ones_nearest_to_4_passes(scripts, scenario_dir, families,
                                                                         tmp_path, capsys):
    """Passes of 8 in style 1, by family: 8, 0, 3, 5, 6, 4, 4 and 7. The first always passes and the second
    always fails, so they are out. Distance from 4: family 2 -> 1, 3 -> 1, 4 -> 2, 5 -> 0, 6 -> 0, 7 -> 3.
    Chosen: families 5 and 6. Style 2 always passes and is not read."""
    ids = [family["family_id"] for family in families["train"][:8]]
    episodes = overfit_rollout(tmp_path, scenario_dir, families, [8, 0, 3, 5, 6, 4, 4, 7])
    target = tmp_path / "checks" / "overfit_families.txt"
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == f"{ids[5]}\n{ids[6]}\n"
    printed = capsys.readouterr().out
    assert f"{ids[5]}: 4 of 8 attempts passed in style 1" in printed
    assert f"{ids[6]}: 4 of 8 attempts passed in style 1" in printed
    report = read_json(tmp_path / "checks" / "overfit_families.txt.json")
    assert [row["family_id"] for row in report["chosen"]] == [ids[5], ids[6]]
    assert report["counts"] == {"families": 8, "mixed": 6, "always_pass": 1, "always_fail": 1,
                                "with_harness_error": 0}
    assert report["source"]["episodes"]["sha256"] == sha256_of(episodes)
    assert report["source"]["model"]["adapter"] is None and report["source"]["arm"] == "base_train"
    assert report["script"]["sha256"] == sha256_of(SCRIPTS / "pick_overfit_families.py")


def test_ties_of_the_overfit_pick_go_to_the_smaller_family_id(scripts, scenario_dir, families, tmp_path,
                                                              capsys):
    """Passes 5, 3, 3, 5: all four are 1 away from 4. The two smallest family ids are taken. Then the
    first family gets 4 passes: it comes first, and the tie for the second place goes to the next id."""
    ids = sorted(family["family_id"] for family in families["train"][:4])
    assert ids == [family["family_id"] for family in families["train"][:4]]        # the file is in id order
    target = tmp_path / "families.txt"
    episodes = overfit_rollout(tmp_path / "a", scenario_dir, families, [5, 3, 3, 5])
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target)]) == 0
    assert target.read_text(encoding="utf-8").split() == ids[:2]
    episodes = overfit_rollout(tmp_path / "b", scenario_dir, families, [5, 3, 3, 4])
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target)]) == 0
    assert target.read_text(encoding="utf-8").split() == [ids[3], ids[0]]


def test_a_family_with_a_harness_error_is_not_an_overfit_family(scripts, scenario_dir, families, tmp_path,
                                                                capsys):
    """Family 0 has 4 passes but one of its 8 attempts is a harness error, so it is left out. Families 1
    and 2 (2 and 6 passes, both 2 away from 4) are taken."""
    ids = [family["family_id"] for family in families["train"][:3]]
    episodes = overfit_rollout(tmp_path, scenario_dir, families, [4, 2, 6], errors={0: 1})
    target = tmp_path / "families.txt"
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target)]) == 0
    assert target.read_text(encoding="utf-8").split() == ids[1:]
    assert read_json(tmp_path / "families.txt.json")["counts"]["with_harness_error"] == 1


def test_fewer_than_two_mixed_families_end_the_overfit_pick_with_code_1(scripts, scenario_dir, families,
                                                                        tmp_path, capsys):
    """Passes 8, 0, 4, 8, 0: only one family is mixed."""
    episodes = overfit_rollout(tmp_path, scenario_dir, families, [8, 0, 4, 8, 0])
    target = tmp_path / "families.txt"
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target)]) == 1
    message = capsys.readouterr().err
    assert "only 1 of the 5 families" in message and "2 are needed" in message
    assert "Nothing was written" in message
    assert not target.exists() and not (tmp_path / "families.txt.json").exists()


def test_the_overfit_pick_refuses_what_it_must_not_read(scripts, scenario_dir, families, tmp_path, capsys):
    data_sha256 = sha256_of(scenario_dir / "train.jsonl")
    target = tmp_path / "families.txt"
    dev = [record for family in families["dev"][:3]
           for record in attempts(family, 1, 4, 8, data_sha256=data_sha256, split="dev", arm="base")]
    episodes = write_rollout(tmp_path / "dev", dev, adapter=None, split="dev", data_sha256=data_sha256)
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target)]) == 2
    assert "records of the split dev" in capsys.readouterr().err
    episodes = overfit_rollout(tmp_path, scenario_dir, families, [4, 4, 4])
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target), "--style", "4"]) == 2
    assert "held-out style" in capsys.readouterr().err
    assert scripts.overfit.main(["--episodes", str(episodes), "--out", str(target), "--style", "3"]) == 2
    assert "no record in wording style 3" in capsys.readouterr().err
    assert scripts.overfit.main(["--episodes", str(tmp_path / "none.jsonl"), "--out", str(target)]) == 2
    assert "is missing" in capsys.readouterr().err
    assert not target.exists()


def test_the_overfit_pick_refuses_a_rollout_that_is_not_finished(scripts, scenario_dir, families, tmp_path,
                                                                 capsys):
    """A rollout that still runs has only its first passes on disk, and its run manifest says "complete":
    false. Here the file holds 5 of the 8 attempts of three families. Two of them look mixed (4 of 5 and 3
    of 5 passes), so a pick that did not look at the manifest would write them."""
    episodes = overfit_rollout(tmp_path, scenario_dir, families, [4, 3, 5])
    records = read_records(episodes)
    manifest_file = episodes.parent / "run_manifest.json"
    manifest = read_json(manifest_file)
    target = tmp_path / "families.txt"
    argv = ["--episodes", str(episodes), "--out", str(target)]
    write_records(episodes, [record for record in records if record["sample"] < 5])
    manifest_file.write_text(json.dumps({**manifest, "complete": False}), encoding="utf-8")
    assert scripts.overfit.main(argv) == 2
    assert "says that the rollout is not complete" in capsys.readouterr().err
    # A manifest that says complete beside a file that lost records: 3 families x 2 styles x 5 = 30.
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    assert scripts.overfit.main(argv) == 2
    assert "holds 30 records" in capsys.readouterr().err
    # No manifest: it is not known whether the rollout ended.
    manifest_file.unlink()
    assert scripts.overfit.main(argv) == 2
    assert "run_manifest.json is missing" in capsys.readouterr().err
    assert not target.exists() and not (tmp_path / "families.txt.json").exists()
    # The whole rollout with its manifest is read: all three families are mixed, the first two are taken.
    write_records(episodes, records)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    assert scripts.overfit.main(argv) == 0
    assert target.read_text(encoding="utf-8").split() == \
        [family["family_id"] for family in families["train"][:2]]
    assert read_json(tmp_path / "families.txt.json")["source"]["model"]["complete"] is True


# ---------------------------------------------------------------------------
# The chain scripts: what can be read from their text
# ---------------------------------------------------------------------------
OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
SOURCES = {"rollout": GPU_SCRIPTS / "rollout.py", "sft_data": GPU_SCRIPTS / "sft_data.py",
           "train_sft": GPU_SCRIPTS / "train_sft.py", "train_grpo": GPU_SCRIPTS / "train_grpo.py",
           "tables": ROOT / "accessdesk" / "eval" / "tables.py", "sft_profile": SCRIPTS / "sft_profile.py",
           "pick_overfit_families": SCRIPTS / "pick_overfit_families.py",
           "monitor_report": SCRIPTS / "monitor_report.py"}
# The scripts that each chain starts.
CALLED = {"chain_stage1.sh": ("rollout", "sft_data", "train_sft", "tables", "sft_profile"),
          "chain_stage2.sh": ("train_grpo", "rollout", "tables", "monitor_report", "pick_overfit_families")}
RESUME_AFTER_FAULT = "--allow-resume-after-fault"     # stage 2 passes it only with RESUME_AFTER_FAULT=1


def options_of(script: str) -> set[str]:
    """The options that a Python script has, read from its source (the first argument of add_argument)."""
    source = SOURCES[script].read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\(\s*\"(--[a-z0-9-]+)\"", source))


def chain_text(chain: str) -> str:
    return (GPU_SCRIPTS / chain).read_text(encoding="utf-8")


@pytest.mark.parametrize("chain", CHAINS)
def test_a_chain_script_is_valid_bash(chain):
    result = subprocess.run(["bash", "-n", str(GPU_SCRIPTS / chain)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("chain", CHAINS)
def test_a_chain_names_only_scripts_that_exist_and_options_that_they_have(chain):
    """Every script path in the chain is a file, and every double-dash word outside the comments is an
    option of a script that the chain starts. What each single call is given is checked on the real calls,
    in every run of a chain below (`Chain.run`)."""
    body = "\n".join(line for line in chain_text(chain).splitlines() if not line.lstrip().startswith("#"))
    named = set(re.findall(r"scripts/(?:gpu/)?\w+\.py", body))
    assert named and all((ROOT / path).is_file() for path in named), named
    stems = {Path(path).stem for path in named}
    if "-m accessdesk.eval.tables" in body:
        stems.add("tables")
    assert stems == set(CALLED[chain])
    known = set().union(*(options_of(script) for script in CALLED[chain]))
    assert set(OPTION.findall(body)) <= known, sorted(set(OPTION.findall(body)) - known)
    assert len(known) > 20
    if chain == "chain_stage2.sh":
        assert {"--resume", RESUME_AFTER_FAULT, "--save-tokens"} <= set(OPTION.findall(body))


# What a chain compares in a run manifest before it skips a step, by the script that writes the manifest
# ("/" goes into a part of the manifest). Typed by hand from the two chains.
COMPARED = {
    "rollout": ("model", "revision", "adapter", "split", "samples", "run_seed", "config/max_new_tokens",
                "text_sha256", "split_file_sha256", "family_set"),
    "sft_data": ("split", "text_sha256", "data_sha256", "config/max_new_tokens", "allow_penalised",
                 "source_run/model", "source_run/revision", "source_run/adapter", "source_run/run_seed",
                 "source_run/samples", "inputs/episodes/sha256", "inputs/tokens/sha256"),
    "train_sft": ("model", "revision", "seed", "config/max_new_tokens", "data/text_sha256",
                  "data/data_sha256", "data/records/sha256"),
    "train_grpo": ("model", "revision", "seed", "mode", "config/max_new_tokens", "text_sha256",
                   "split_file_sha256", "scenario_folder", "grpo_config/n_scenarios",
                   "grpo_config/group_size", "start_adapter/files/adapter_model.safetensors/sha256",
                   "overfit/file/sha256"),
}


def test_what_a_chain_compares_before_a_skip_is_what_the_scripts_write():
    """A value that a run manifest does not name is not compared (the chain only says so). So a name that
    is typed wrong in a chain would switch a check off without a fault. Two guards: the chains compare
    exactly the names of COMPARED, and each name is in the code that writes the manifest. The first part
    of a name must be a quoted word of the script itself; a later part may also be a field of the two
    config classes, or a quoted word of one of the four scripts (they share `file_entry`)."""
    # An item is a quoted "name=value" with a name in small letters. `--arm "base=..."` of the table is not
    # one, and an item that starts with "chain:" is the chain's own note, not a name of a manifest.
    item = re.compile(r'(?<!--arm )"((?!chain:)[a-z_0-9./]+)=')
    in_chains = set()
    for chain in CHAINS:
        body = "\n".join(line for line in chain_text(chain).splitlines() if not line.lstrip().startswith("#"))
        in_chains |= set(item.findall(body))
    assert in_chains == {name for names in COMPARED.values() for name in names}
    sources = {script: SOURCES[script].read_text(encoding="utf-8") for script in COMPARED}
    fields = set(RunConfig().to_dict()) | set(GRPOConfig().to_dict())
    for script, names in COMPARED.items():
        for name in names:
            first, *others = name.split("/")
            assert f'"{first}"' in sources[script], (script, name)
            for part in others:
                assert part in fields or any(f'"{part}"' in source for source in sources.values()), \
                    (script, name, part)


def test_the_real_run_manifest_of_a_rollout_names_what_the_chains_compare():
    """One run manifest that scripts/gpu/rollout.py wrote on the rented machine is in the repository (the
    2B of calibration round 3). It must hold every name that the chains compare for a rollout."""
    path = ROOT / "results" / "calibration" / "round3" / "2b_mixed" / "run_manifest.json"
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    manifest = read_json(path)
    for name in COMPARED["rollout"]:
        value = manifest
        for part in name.split("/"):
            assert part in value, name
            value = value[part]
    assert (manifest["model"], manifest["revision"], manifest["adapter"]) == (MODEL, REVISION, None)


def default_of(chain: str, name: str) -> str:
    """The default of a variable at the top of a chain: NAME="${NAME:-default}" at the start of a line."""
    match = re.search(rf'^{name}="\$\{{{name}:-([^"]*)\}}"', chain_text(chain), flags=re.MULTILINE)
    assert match, (chain, name)
    return match.group(1)


def test_the_chains_have_the_decided_values_at_their_top():
    """The values of the task, typed by hand: the 2B at its pinned revision; dev rollouts with run seed 0;
    the train rollout that feeds SFT with run seed 10; SFT seed 0; the profile rollout with run seed 20;
    GRPO seed 1 (2 for the second job); pilot 100; overfit check 101; 8 attempts; the curve at steps 0, 20,
    40, 60 and 80 in one seen wording with 4 attempts."""
    for chain in CHAINS:
        assert default_of(chain, "MODEL") == MODEL and default_of(chain, "REVISION") == REVISION
        assert default_of(chain, "SCENARIOS") == "data/scenarios"
        assert default_of(chain, "OUT") == "results/main"
        assert default_of(chain, "EVAL_SEED") == "0" and default_of(chain, "SAMPLES") == "8"
        assert default_of(chain, "MAX_NEW_TOKENS") == ""          # the cap comes from accessdesk/config.py
    assert default_of("chain_stage1.sh", "SEED") == "0"
    assert default_of("chain_stage1.sh", "TRAIN_ROLLOUT_SEED") == "10"
    assert default_of("chain_stage1.sh", "PROFILE_SEED") == "20"
    assert default_of("chain_stage1.sh", "PROFILE_PER_OUTCOME") == "25"
    assert default_of("chain_stage1.sh", "TRAIN_STYLES") == "1,2,3"
    assert default_of("chain_stage2.sh", "SEED") == "1"
    assert default_of("chain_stage2.sh", "PILOT_SEED") == "100"
    assert default_of("chain_stage2.sh", "OVERFIT_SEED") == "101"
    assert default_of("chain_stage2.sh", "CURVE_STEPS") == "0 20 40 60 80"
    assert (default_of("chain_stage2.sh", "CURVE_STYLE"), default_of("chain_stage2.sh", "CURVE_SAMPLES")) == \
        ("1", "4")
    assert default_of("chain_stage2.sh", "SCENARIOS_PER_STEP") == ""
    assert default_of("chain_stage2.sh", "GROUP_SIZE") == ""
    assert default_of("chain_stage2.sh", "ALLOW_DEFAULT_STEP") == "0"
    assert default_of("chain_stage2.sh", "DECISION") == "$OUT/sft_profile/decision.json"
    assert default_of("chain_stage2.sh", "OVERFIT_EPISODES") == "$OUT/base_train/episodes.jsonl"
    assert default_of("chain_stage2.sh", "RESUME_AFTER_FAULT") == "0"


def test_the_curve_set_is_a_set_of_the_rollout_script_and_of_the_scenario_folders():
    """The chain asks for `--set curve_subset`. That name must be one that scripts/gpu/rollout.py takes,
    and a list of 20 dev families in sets.json of every scenario folder that is on this machine."""
    assert "--set curve_subset --styles" in chain_text("chain_stage2.sh")
    assert re.search(r'SETS = \("curve_subset",', SOURCES["rollout"].read_text(encoding="utf-8"))
    folders = sorted(path.parent for path in (ROOT / "data").glob("scenarios*/sets.json"))
    assert PLACEHOLDER_SCENARIOS in folders
    for folder in folders:
        assert len(read_json(folder / "sets.json")["curve_subset"]) == 20, folder


def test_the_shared_part_of_the_two_chains_is_the_same_text():
    """Both chains hold the same helper functions between two marker lines. A change in one file only
    would give the two stages different rules for a skip."""
    parts = []
    for chain in CHAINS:
        text = chain_text(chain)
        start, end = text.index("# ---- The part below"), text.index("# ---- end of the shared part")
        assert 0 < start < end
        parts.append(text[start:end])
    assert parts[0] == parts[1] and "fits()" in parts[0] and "rollout_step()" in parts[0]


# ---------------------------------------------------------------------------
# The chain scripts: runs with a stand-in for python
# ---------------------------------------------------------------------------
FAKE_PYTHON = r'''#!@PYTHON@
"""Stands in for `python` in a chain script.

The chain's own snippets (`-` and `-c`) and the two small scripts (sft_profile.py, pick_overfit_families.py)
run for real, in this process. Every other script is replaced: its call is written to the call log, and it
leaves what the chain and the small scripts read afterwards, with the key names of the real script.

FAKE_FAIL = <folder name or script name>:<exit code>[:<status>[:<last checkpoint>]] makes one step fail.
FAKE_PASSES_<arm> (or FAKE_PASSES) = pass counts, with commas: in a rollout, family i of the kept families
passes that many of its attempts in every wording (the list is used again from its start when it ends).
"""
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

args = sys.argv[1:]
sys.path.insert(0, os.getcwd())
if args[0] == "-":
    sys.argv = args
    exec(compile(sys.stdin.read(), "<snippet of the chain>", "exec"), {"__name__": "__main__"})
    raise SystemExit(0)
if args[0] == "-c":
    sys.argv = ["-c", *args[2:]]
    exec(compile(args[1], "<snippet of the chain>", "exec"), {"__name__": "__main__"})
    raise SystemExit(0)
name = "tables" if args[:2] == ["-m", "accessdesk.eval.tables"] else Path(args[0]).stem
with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"script": name, "args": args}) + "\n")
if name in ("sft_profile", "pick_overfit_families"):
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
    raise SystemExit(0)

options = dict(zip(args, args[1:]))
fail = os.environ.get("FAKE_FAIL", "").split(":")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


def save_adapter(folder, content):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    (folder / "adapter_model.safetensors").write_text(content, encoding="utf-8")


if name == "tables":
    if fail[0] == name:
        raise SystemExit(int(fail[1]))
    write(options["--out-json"], {})
    Path(options["--out-md"]).write_text("table", encoding="utf-8")
    raise SystemExit(0)
if name == "monitor_report":
    if fail[0] == name:
        raise SystemExit(int(fail[1]))
    print("monitor report of " + options["--episodes"])
    raise SystemExit(0)

out = Path(options["--out"])
out.mkdir(parents=True, exist_ok=True)
if fail[0] == out.name and name != "train_grpo":
    write(out / "run_manifest.json", {"complete": False, "status": "failed"})
    raise SystemExit(int(fail[1]))

if name == "rollout":
    scenarios, split = Path(options["--scenarios"]), options["--split"]
    lines = (scenarios / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
    families = [json.loads(line) for line in lines if line]
    if "--families" in options:
        wanted = set(Path(options["--families"]).read_text(encoding="utf-8").split())
        families = [family for family in families if family["family_id"] in wanted]
    if "--set" in options:
        wanted = set(read(scenarios / "sets.json")[options["--set"]])
        families = [family for family in families if family["family_id"] in wanted]
    styles = {int(style) for style in options["--styles"].split(",")} if "--styles" in options else None
    samples, arm = int(options["--samples"]), options["--arm"]
    passes = os.environ.get("FAKE_PASSES_" + arm, os.environ.get("FAKE_PASSES", "6")).split(",")
    data_sha256 = sha(scenarios / (split + ".jsonl"))
    records = []
    for sample in range(samples):
        for index, family in enumerate(families):
            for wording in family["wordings"]:
                if styles is None or wording["style"] in styles:
                    records.append({
                        "run": out.name, "arm": arm, "split": split, "family_id": family["family_id"],
                        "style": wording["style"],
                        "scenario_id": family["family_id"] + "/s" + str(wording["style"]),
                        "outcome": family["outcome"], "ruleset": "default", "toolset": "five",
                        "prompt": "final", "sample": sample,
                        "passed": sample < int(passes[index % len(passes)]), "harness_error": None,
                        "data_sha256": data_sha256})
    text = "".join(json.dumps(record) + "\n" for record in records)
    (out / "episodes.jsonl").write_text(text, encoding="utf-8")
    (out / "tokens.jsonl").write_text("tokens of " + out.name + "\n", encoding="utf-8")
    write(out / "run_manifest.json", {
        "complete": True, "command": args, "config": {"max_new_tokens": int(options["--max-new-tokens"])},
        "text_sha256": sha(options["--text"]), "model": options["--model"], "revision": options["--revision"],
        "adapter": options.get("--adapter"), "split": split, "split_file_sha256": data_sha256,
        "family_set": options.get("--set"), "families": len(families),
        "scenarios": len(records) // samples, "samples": samples, "run_seed": int(options["--run-seed"])})
elif name == "sft_data":
    episodes, scenarios = Path(options["--episodes"]), Path(options["--scenarios"])
    source = read(episodes.parent / "run_manifest.json")
    (out / "sft_records.jsonl").write_text("records of " + sha(episodes) + "\n", encoding="utf-8")
    write(out / "run_manifest.json", {
        "complete": True, "script": "scripts/gpu/sft_data.py", "split": "train",
        "text_sha256": read(scenarios / "summary.json")["text_sha256"],
        "data_sha256": sha(scenarios / "train.jsonl"), "config": source["config"],
        "allow_penalised": "--allow-penalised" in args,
        "source_run": {key: source[key] for key in ("model", "revision", "adapter", "run_seed", "samples")},
        "inputs": {"episodes": {"sha256": sha(episodes)}, "tokens": {"sha256": sha(options["--tokens"])}}})
elif name == "train_sft":
    data = read(Path(options["--data"]) / "run_manifest.json")
    save_adapter(out / "adapter", "SFT weights, made at " + str(time.time_ns()))
    write(out / "run_manifest.json", {
        "complete": True, "status": "complete", "model": options["--model"],
        "revision": options["--revision"], "seed": int(options["--seed"]), "config": data["config"],
        "data": {"text_sha256": data["text_sha256"], "data_sha256": data["data_sha256"],
                 "records": {"sha256": sha(Path(options["--data"]) / "sft_records.jsonl")}}})
elif name == "train_grpo":
    mode = "pilot" if "--pilot" in args else "overfit" if "--overfit" in args else "seed"
    steps = {"pilot": 15, "overfit": 30, "seed": 80}[mode]
    failed = fail[0] == out.name
    status = (fail[2] if len(fail) > 2 else "failed") if failed else "complete"
    last = (int(fail[3]) if len(fail) > 3 else 0) if failed else steps
    saved = [step for step in range(10, last + 1, 10)] + ([last] if last % 10 else [])
    for step in saved:
        save_adapter(out / "checkpoints" / ("step_%05d" % step) / "adapter", "weights after step %d" % step)
    if saved:
        write(out / "latest.json", {"checkpoint": "checkpoints/step_%05d" % saved[-1]})
    adapter = Path(options["--adapter"])
    manifest = {
        "complete": status == "complete", "status": status, "mode": mode, "command": args,
        "model": options["--model"], "revision": options["--revision"], "seed": int(options["--seed"]),
        "config": {"max_new_tokens": int(options["--max-new-tokens"])},
        "text_sha256": sha(options["--text"]), "scenario_folder": options["--scenarios"],
        "split_file_sha256": sha(Path(options["--scenarios"]) / "train.jsonl"),
        "grpo_config": {"n_scenarios": int(options.get("--scenarios-per-step", 2)),
                        "group_size": int(options.get("--group-size", 8))},
        "start_adapter": {"files": {"adapter_model.safetensors":
                                    {"sha256": sha(adapter / "adapter_model.safetensors")}}},
        "overfit": {"file": {"sha256": sha(options["--overfit"])}} if mode == "overfit" else None,
        "fault": {"step": last + 3, "reason": "a made-up fault"} if status == "stopped_by_rule" else None,
        "final_adapter": None,
        "summary": {"first_step": {"pass_rate": 0.5, "mean_reward": 48.5},
                    "last_step": {"pass_rate": 0.875, "mean_reward": 86.25}}}
    if status == "stopped_by_rule":
        write(out / "stopped.json", manifest["fault"])
    if status == "complete":
        (out / "stopped.json").unlink(missing_ok=True)
        manifest["final_adapter"] = {"folder": "checkpoints/step_%05d/adapter" % steps}
    write(out / "run_manifest.json", manifest)
    if failed:
        raise SystemExit(int(fail[1]))
else:
    raise SystemExit("the stand-in does not know the script " + name)
'''
# Every variable that a chain reads. A value of the shell that runs the tests must not reach a chain.
CHAIN_VARIABLES = ("MODEL", "REVISION", "SCENARIOS", "OUT", "SEED", "EVAL_SEED", "TRAIN_ROLLOUT_SEED",
                   "PROFILE_SEED", "MAX_NEW_TOKENS", "SAMPLES", "TRAIN_STYLES", "PROFILE_PER_OUTCOME", "TEXT",
                   "SFT_DATA_OPTIONS", "PYTHON", "REQUIRE_FROZEN", "SFT_ADAPTER", "DECISION",
                   "SCENARIOS_PER_STEP", "GROUP_SIZE", "ALLOW_DEFAULT_STEP", "GRPO_OPTIONS", "CURVE_STEPS",
                   "CURVE_STYLE",
                   "CURVE_SAMPLES", "CHECK", "OVERFIT_EPISODES", "OVERFIT_SEED",
                   "OVERFIT_MAX_HARNESS_ERROR_RATE", "PILOT_SEED", "RESUME_AFTER_FAULT", "FAKE_FAIL",
                   "FAKE_PASSES")
STAGE_1 = ["base_dev", "base_train", "sft_data", "sft", "sft_dev", "table", "profile_families", "sft_train",
           "sft_profile"]
STAGE_1_SCRIPTS = ["rollout", "rollout", "sft_data", "train_sft", "rollout", "tables", "sft_profile",
                   "rollout", "sft_profile"]
CURVE = ["grpo_seed1_curve_000", "grpo_seed1_curve_020", "grpo_seed1_curve_040", "grpo_seed1_curve_060",
         "grpo_seed1_curve_080"]
STAGE_2 = ["grpo_seed1", "grpo_seed1_dev", "table", "monitors_base", "monitors_sft", "monitors_grpo_seed1",
           *CURVE]
STAGE_2_SCRIPTS = ["train_grpo", "rollout", "tables", "monitor_report", "monitor_report", "monitor_report",
                   *["rollout"] * 5]


def ran(names: list[str]) -> list[str]:
    """The marker lines of steps that ran one after the other without a fault."""
    return [line for name in names for line in (f"STEP_START {name}", f"STEP_DONE {name} 0")]


class Chain:
    """Runs a chain script with the stand-in for `python`, on the placeholder scenario folder (it is only
    read) and with an output folder of the test."""

    def __init__(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.out = folder / "out"
        self.calls_file = folder / "calls.jsonl"
        self.python = folder / "fake_python"
        self.python.write_text(FAKE_PYTHON.replace("@PYTHON@", sys.executable), encoding="utf-8")
        self.python.chmod(0o755)
        self.output = ""

    def run(self, chain: str, **variables: str) -> tuple[int, list[str], list[dict]]:
        """(exit code, the chain's step lines, the script calls of this run). The CHAIN_NOTE lines are kept
        in `self.notes` and the whole output in `self.output`. Every option of every call must be one
        that the called script has."""
        self.calls_file.write_text("", encoding="utf-8")
        clean = {key: value for key, value in os.environ.items()
                 if key not in CHAIN_VARIABLES and not key.startswith("FAKE_PASSES_")}
        env = {**clean, "PYTHON": str(self.python), "FAKE_CALLS": str(self.calls_file), "OUT": str(self.out),
               "SCENARIOS": "data/scenarios_placeholder", "TEXT": "text/placeholder_text.yaml", **variables}
        result = subprocess.run(["bash", str(GPU_SCRIPTS / chain)], env=env, capture_output=True, text=True)
        self.output = result.stdout + result.stderr
        lines = [re.sub(r"\.unfinished\.\S+", ".unfinished", line) for line in result.stdout.splitlines()]
        self.notes = [line for line in lines if line.startswith("CHAIN_NOTE")]
        markers = [line for line in lines if line.startswith(("STEP_", "CHAIN_")) and line not in self.notes]
        calls = read_records(self.calls_file)
        for call in calls:
            unknown = {arg for arg in call["args"] if arg.startswith("--")} - options_of(call["script"])
            assert not unknown, (call["script"], sorted(unknown))
        return result.returncode, markers, calls

    def note(self, start: str) -> str:
        """The one CHAIN_NOTE line that starts with these words."""
        found = [line for line in self.notes if line.startswith("CHAIN_NOTE " + start)]
        assert len(found) == 1, (start, self.notes)
        return found[0]


def value_of(call: dict, option: str) -> str:
    return call["args"][call["args"].index(option) + 1]


def files_of(folder: Path) -> dict[str, bytes]:
    """Every file under the folder: its path from the folder -> its bytes."""
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def step_size(call: dict) -> tuple[str, str]:
    """(scenarios per step, attempts per scenario) that a call of train_grpo.py is given."""
    return value_of(call, "--scenarios-per-step"), value_of(call, "--group-size")


@pytest.fixture(scope="module")
def stage_1_done(tmp_path_factory) -> Path:
    """The output folder of one whole run of stage 1 (the tests copy it). In the rollout of the SFT model
    on train every family passes 6 of 8 attempts: the median family pass rate is 0.75, not above 0.8, so
    the profile gives the plan's step of 16 scenarios x 8 attempts. The untrained model on train passes 8,
    0, 3, 5, 6, 4, 4 and 7 of 8 attempts in the first eight families, and so on in a circle."""
    chain = Chain(tmp_path_factory.mktemp("stage_1"))
    code, markers, _ = chain.run("chain_stage1.sh", FAKE_PASSES_base_train="8,0,3,5,6,4,4,7")
    assert code == 0 and markers[-1] == "CHAIN_DONE", chain.output
    return chain.out


@pytest.fixture(scope="module")
def stage_1_chain(tmp_path_factory) -> Chain:
    """A chain that ran stage 1 once, with the values of the script. Tests start it again in the same
    folder; they must leave every run folder as it is (a start that is refused changes none)."""
    chain = Chain(tmp_path_factory.mktemp("stage_1_again"))
    code, markers, _ = chain.run("chain_stage1.sh")
    assert code == 0 and markers[-1] == "CHAIN_DONE", chain.output
    return chain


def stage_2(tmp_path: Path, stage_1_done: Path, *, keep: tuple[str, ...] | None = None) -> Chain:
    """A chain whose output folder holds a copy of the results of stage 1 (only the folders of `keep`, if
    given: the second machine gets the SFT run and the decision, not the dev rollouts)."""
    chain = Chain(tmp_path)
    shutil.copytree(stage_1_done, chain.out)
    if keep is not None:
        for path in chain.out.iterdir():
            if path.name not in keep:
                shutil.rmtree(path)
    return chain


def test_chain_stage_1_runs_its_nine_steps_in_order_with_the_decided_seeds(tmp_path):
    chain = Chain(tmp_path)
    code, markers, calls = chain.run("chain_stage1.sh", FAKE_PASSES_sft_train="7")
    assert code == 0, chain.output
    assert markers == ran(STAGE_1) + ["CHAIN_DONE"]
    assert [call["script"] for call in calls] == STAGE_1_SCRIPTS
    # The log says once what the state of the freeze is. It has no note on the token cap: without
    # MAX_NEW_TOKENS the cap is the config's.
    assert chain.note("the tree is ")
    assert not [line for line in chain.notes if line.startswith("CHAIN_NOTE the token cap")]
    base_dev, base_train, sft_data, train_sft, sft_dev, table, pick, sft_train, profile = calls
    out = chain.out
    cap = str(RunConfig().max_new_tokens)
    # Every rollout: the 2B at its revision, the token cap of the config, 8 attempts, the same text file
    # and scenario folder.
    for call in (base_dev, base_train, sft_dev, sft_train):
        assert (value_of(call, "--model"), value_of(call, "--revision")) == (MODEL, REVISION)
        assert (value_of(call, "--max-new-tokens"), value_of(call, "--samples")) == (cap, "8")
        assert value_of(call, "--scenarios") == "data/scenarios_placeholder"
        assert value_of(call, "--text") == "text/placeholder_text.yaml"
    # The baseline on dev: run seed 0, every wording, no adapter.
    assert (value_of(base_dev, "--split"), value_of(base_dev, "--run-seed")) == ("dev", "0")
    assert value_of(base_dev, "--arm") == "base" and value_of(base_dev, "--out") == str(out / "base_dev")
    assert not {"--styles", "--adapter", "--families", "--set"} & set(base_dev["args"])
    # Rejection sampling on train: its own run seed 10, the three training styles, no adapter.
    assert (value_of(base_train, "--split"), value_of(base_train, "--run-seed")) == ("train", "10")
    assert value_of(base_train, "--styles") == "1,2,3" and "--adapter" not in base_train["args"]
    # sft_data.py reads that rollout with its tokens; train_sft.py reads sft_data's folder, with seed 0.
    assert value_of(sft_data, "--episodes") == str(out / "base_train" / "episodes.jsonl")
    assert value_of(sft_data, "--tokens") == str(out / "base_train" / "tokens.jsonl")
    assert (out / "base_train" / "tokens.jsonl").is_file()
    assert value_of(train_sft, "--data") == value_of(sft_data, "--out") == str(out / "sft_data")
    assert value_of(train_sft, "--seed") == "0" and value_of(train_sft, "--out") == str(out / "sft")
    # The SFT model on dev: the SFT adapter, and everything else as the baseline, so the two arms are
    # sampled with the same sampling seeds.
    assert value_of(sft_dev, "--adapter") == str(out / "sft" / "adapter")
    assert value_of(sft_dev, "--arm") == "sft"
    for option in ("--split", "--samples", "--run-seed", "--max-new-tokens"):
        assert value_of(sft_dev, option) == value_of(base_dev, option), option
    assert "--styles" not in sft_dev["args"]
    # The table: base against SFT, from the two dev rollouts.
    assert table["args"][2:] == [
        "--arm", f"base={out}/base_dev/episodes.jsonl", "--arm", f"sft={out}/sft_dev/episodes.jsonl",
        "--baseline", "base", "--pair", "base:sft", "--out-json", f"{out}/tables/base_vs_sft.json",
        "--out-md", f"{out}/tables/base_vs_sft.md"]
    assert (out / "tables" / "base_vs_sft.md").is_file()
    # The profile: 100 training families (25 of each outcome), the SFT adapter, the three training styles,
    # 8 attempts, its own run seed 20.
    families_file = out / "sft_profile" / "families.txt"
    # The pick is written to a second file, which then takes the place of families.txt (there was none).
    assert value_of(pick, "--pick") == f"{families_file}.new" and value_of(pick, "--per-outcome") == "25"
    assert len(families_file.read_text(encoding="utf-8").split()) == 100
    assert sorted(path.name for path in families_file.parent.iterdir()) == \
        ["decision.json", "families.txt", "profile.md"]
    assert (value_of(sft_train, "--split"), value_of(sft_train, "--run-seed")) == ("train", "20")
    assert value_of(sft_train, "--families") == str(families_file)
    assert value_of(sft_train, "--styles") == "1,2,3"
    assert value_of(sft_train, "--adapter") == str(out / "sft" / "adapter")
    assert len(read_records(out / "sft_train" / "episodes.jsonl")) == 100 * 3 * 8
    assert value_of(profile, "--episodes") == str(out / "sft_train" / "episodes.jsonl")
    assert value_of(profile, "--families") == str(families_file)
    # The real sft_profile.py ran on these episodes: every family passes 7 of 8 attempts, so the median is
    # 0.875, above 0.8: groups of 16 on 8 scenarios.
    decision = read_json(out / "sft_profile" / "decision.json")
    assert (decision["median_family_pass_rate"], decision["n_families"]) == (0.875, 100)
    assert (decision["scenarios_per_step"], decision["group_size"]) == (8, 16)
    assert decision["source"]["adapter_sha256"]["adapter_model.safetensors"] == \
        sha256_of(out / "sft" / "adapter" / "adapter_model.safetensors")
    assert "median family pass rate 0.875 on 100 families; a GRPO step is 8 scenarios x 16 attempts" in \
        chain.note("the family profile after SFT")
    # The three run seeds of the rollouts are different, and no one is a GRPO seed (1, 2, 100, 101).
    seeds = {value_of(call, "--run-seed") for call in (base_dev, base_train, sft_train)}
    assert seeds == {"0", "10", "20"}


def test_a_second_start_of_stage_1_skips_the_complete_steps(stage_1_chain):
    chain = stage_1_chain
    code, markers, calls = chain.run("chain_stage1.sh")
    assert code == 0, chain.output
    # The five runs and the profile rollout are not made again. The table, the families file and the
    # profile are built again from what is there.
    assert markers == [f"STEP_SKIPPED {name}" for name in STAGE_1[:5]] + ran(["table", "profile_families"]) \
        + ["STEP_SKIPPED sft_train"] + ran(["sft_profile"]) + ["CHAIN_DONE"]
    assert [call["script"] for call in calls] == ["tables", "sft_profile", "sft_profile"]
    assert "not compared" not in chain.output and not list(chain.out.glob("*.unfinished.*"))


# Each case: a variable with another value at the second start, and the first step that this value is part
# of. The steps before it are skipped; that step is refused; nothing runs after it.
@pytest.mark.parametrize("variable, value, refused_step", [
    ("MODEL", "Qwen/Qwen3.5-4B", "base_dev"),
    ("REVISION", "0" * 40, "base_dev"),
    ("EVAL_SEED", "5", "base_dev"),
    ("SAMPLES", "4", "base_dev"),
    ("MAX_NEW_TOKENS", "300", "base_dev"),
    ("TRAIN_ROLLOUT_SEED", "11", "base_train"),
    ("TRAIN_STYLES", "1,2", "base_train"),
    ("SFT_DATA_OPTIONS", "--allow-penalised", "sft_data"),
    ("SEED", "3", "sft"),
    ("PROFILE_SEED", "21", "sft_train"),
    ("PROFILE_PER_OUTCOME", "20", "sft_train"),
])
def test_stage_1_does_not_skip_a_step_that_other_values_made(stage_1_chain, variable, value, refused_step):
    chain = stage_1_chain
    files = files_of(chain.out)
    code, markers, calls = chain.run("chain_stage1.sh", **{variable: value})
    assert code == 1, chain.output
    assert markers[-1] == f"CHAIN_FAILED {refused_step}" and "CHAIN_DONE" not in markers
    before = STAGE_1[:STAGE_1.index(refused_step)]
    assert [line for line in markers if line.startswith("STEP_SKIPPED")] == \
        [f"STEP_SKIPPED {name}" for name in before if name not in ("table", "profile_families")]
    # No model step ran, and no folder was moved. No file under the output folder is new, gone or changed:
    # not a run manifest, and not the families file of the profile, which the complete sft_train read (the
    # table is built again when the chain gets that far, from the same episodes, so it is the same file).
    assert not {"rollout", "sft_data", "train_sft"} & {call["script"] for call in calls}
    assert not list(chain.out.glob("*.unfinished.*"))
    assert len([name for name in files if name.endswith("/run_manifest.json")]) == 6
    assert "sft_profile/families.txt" in files and "sft_profile/decision.json" in files
    assert files_of(chain.out) == files
    assert f"{chain.out}/{refused_step} is complete, but it is not the step that this chain would run" in \
        chain.output


def test_stage_1_does_not_skip_a_step_whose_text_or_scenario_file_changed(stage_1_chain, tmp_path):
    """The run manifests name the text file and the scenario file by their SHA-256. A second start with
    another text file, or with a scenario folder whose dev file is another, is refused at the first step."""
    chain = stage_1_chain
    other_text = tmp_path / "other_text.yaml"
    other_text.write_bytes((ROOT / "text" / "placeholder_text.yaml").read_bytes() + b"\n")
    code, markers, calls = chain.run("chain_stage1.sh", TEXT=str(other_text))
    assert (code, markers[-1], calls) == (1, "CHAIN_FAILED base_dev", [])
    assert "base_dev: text_sha256 is" in chain.output
    other_data = tmp_path / "other_scenarios"
    shutil.copytree(PLACEHOLDER_SCENARIOS, other_data)
    with (other_data / "dev.jsonl").open("ab") as handle:
        handle.write(b"\n")
    code, markers, calls = chain.run("chain_stage1.sh", SCENARIOS=str(other_data))
    assert (code, markers[-1], calls) == (1, "CHAIN_FAILED base_dev", [])
    assert "base_dev: split_file_sha256 is" in chain.output
    # The same files in a folder with another name: the content is the same, but the chain noted the
    # folder that the step read, and says that it is another.
    same_data = tmp_path / "same_scenarios"
    shutil.copytree(PLACEHOLDER_SCENARIOS, same_data)
    code, markers, calls = chain.run("chain_stage1.sh", SCENARIOS=str(same_data))
    assert (code, markers[-1], calls) == (1, "CHAIN_FAILED base_dev", [])
    assert f"scenarios={same_data}" in chain.output


def test_stage_1_does_not_skip_a_rollout_of_an_adapter_that_was_trained_again(tmp_path):
    """The run manifest of a rollout names its adapter by folder only. The chain notes the SHA-256 of the
    adapter file when the rollout ends. If SFT is trained again into the same folder, the complete dev
    rollout of the old adapter is not taken for the new one."""
    chain = Chain(tmp_path)
    assert chain.run("chain_stage1.sh")[0] == 0
    record = (chain.out / "sft_dev" / "chain_record.txt").read_text(encoding="utf-8")
    old = sha256_of(chain.out / "sft" / "adapter" / "adapter_model.safetensors")
    assert f"adapter_model.safetensors={old}\n" in record
    shutil.rmtree(chain.out / "sft")
    code, markers, calls = chain.run("chain_stage1.sh")
    assert code == 1 and [call["script"] for call in calls] == ["train_sft"]
    assert markers == [f"STEP_SKIPPED {name}" for name in STAGE_1[:3]] + ran(["sft"]) + \
        ["CHAIN_FAILED sft_dev"]
    assert sha256_of(chain.out / "sft" / "adapter" / "adapter_model.safetensors") != old
    assert "what the chain noted" in chain.output and "what this chain has now" in chain.output


def test_stage_1_stops_at_the_first_failed_step_and_can_be_started_again(tmp_path):
    chain = Chain(tmp_path)
    code, markers, calls = chain.run("chain_stage1.sh", FAKE_FAIL="sft_data:2")
    assert code == 2
    assert markers == ran(STAGE_1[:2]) + ["STEP_START sft_data", "STEP_DONE sft_data 2",
                                          "CHAIN_FAILED sft_data"]
    assert [call["script"] for call in calls] == STAGE_1_SCRIPTS[:3] and not (chain.out / "sft").exists()
    code, markers, calls = chain.run("chain_stage1.sh")
    assert code == 0, chain.output
    # The two finished rollouts are not run again. The unfinished folder is kept under another name.
    assert markers == ["STEP_SKIPPED base_dev", "STEP_SKIPPED base_train",
                       f"STEP_MOVED_ASIDE sft_data {chain.out}/sft_data.unfinished"] + ran(STAGE_1[2:]) + \
        ["CHAIN_DONE"]
    assert [call["script"] for call in calls] == STAGE_1_SCRIPTS[2:]
    assert len(list(chain.out.glob("sft_data.unfinished.*"))) == 1


def test_a_failed_table_does_not_cost_the_profile_but_the_chain_ends_as_failed(tmp_path):
    """The table is made from saved episodes and can be made again. The profile rollout needs the GPU that
    is rented now. So the chain goes on after a failed table, and says at its end that the table failed."""
    chain = Chain(tmp_path)
    code, markers, calls = chain.run("chain_stage1.sh", FAKE_FAIL="tables:4")
    assert code == 4
    assert markers == ran(STAGE_1[:5]) + ["STEP_START table", "STEP_DONE table 4"] + ran(STAGE_1[6:]) + \
        ["CHAIN_FAILED table"]
    assert [call["script"] for call in calls] == STAGE_1_SCRIPTS
    assert (chain.out / "sft_profile" / "decision.json").is_file()
    assert not (chain.out / "tables" / "base_vs_sft.md").exists()


def test_stage_1_passes_its_token_cap_and_more_options_on(tmp_path):
    """MAX_NEW_TOKENS reaches every rollout and is said in the log. SFT_DATA_OPTIONS reaches sft_data.py
    only (PLAN 5.5 lets passes with a penalty in if clean passes are too few)."""
    assert RunConfig().max_new_tokens != 300
    chain = Chain(tmp_path)
    code, _, calls = chain.run("chain_stage1.sh", MAX_NEW_TOKENS="300", SFT_DATA_OPTIONS="--allow-penalised")
    assert code == 0, chain.output
    assert "Every step gets --max-new-tokens 300" in chain.note("the token cap of this chain is 300")
    rollouts = [call for call in calls if call["script"] == "rollout"]
    assert len(rollouts) == 4 and {value_of(call, "--max-new-tokens") for call in rollouts} == {"300"}
    assert calls[2]["script"] == "sft_data" and calls[2]["args"][-1] == "--allow-penalised"
    assert all("--allow-penalised" not in call["args"] for call in calls[:2] + calls[3:])


def test_stage_1_can_be_told_to_start_only_on_the_frozen_tree(tmp_path):
    """REQUIRE_FROZEN=1. What the chain must do depends on the repository itself: before the freeze (no
    MANIFEST.json), and when a frozen file has changed, it must not start; on the frozen tree it runs."""
    from accessdesk.freeze import frozen_state
    state = frozen_state()
    chain = Chain(tmp_path)
    code, markers, calls = chain.run("chain_stage1.sh", REQUIRE_FROZEN="1")
    if state["manifest_sha256"] is not None and state["matches"]:
        assert code == 0 and markers[-1] == "CHAIN_DONE"
    else:
        assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], []) and not chain.out.exists()


def test_chain_stage_2_runs_a_seed_then_dev_the_reports_and_the_curve(tmp_path, stage_1_done):
    chain = stage_2(tmp_path, stage_1_done)
    code, markers, calls = chain.run("chain_stage2.sh")
    assert code == 0, chain.output
    assert markers == ran(STAGE_2) + ["CHAIN_DONE"]
    assert [call["script"] for call in calls] == STAGE_2_SCRIPTS
    # As in stage 1: one line on the state of the freeze, and none on the token cap.
    assert chain.note("the tree is ")
    assert not [line for line in chain.notes if line.startswith("CHAIN_NOTE the token cap")]
    train, dev, table = calls[:3]
    monitors, curve = calls[3:6], calls[6:]
    out = chain.out
    cap = str(RunConfig().max_new_tokens)
    # GRPO: the SFT adapter, seed 1, the token data of every step saved, and the step of the profile. In
    # the copied stage 1 every family passes 6 of 8 attempts: median 0.75, so 16 scenarios x 8 attempts.
    assert (value_of(train, "--model"), value_of(train, "--revision"), value_of(train, "--seed")) == \
        (MODEL, REVISION, "1")
    assert value_of(train, "--adapter") == str(out / "sft" / "adapter")
    assert value_of(train, "--out") == str(out / "grpo_seed1") and value_of(train, "--max-new-tokens") == cap
    assert "--save-tokens" in train["args"] and step_size(train) == ("16", "8")
    assert not {"--resume", "--pilot", "--overfit", "--styles"} & set(train["args"])
    note = chain.note("a GRPO step is 16 scenarios x 8 attempts")
    assert "from the family profile after SFT" in note and "median family pass rate 0.75" in note
    assert RULE in note
    # The dev rollout: the adapter of the last checkpoint; the run seed of the dev rollouts of stage 1 (0),
    # not the seed of the training; every wording; the same token cap.
    assert value_of(dev, "--adapter") == str(out / "grpo_seed1" / "checkpoints" / "step_00080" / "adapter")
    assert (value_of(dev, "--split"), value_of(dev, "--samples"), value_of(dev, "--run-seed")) == \
        ("dev", "8", "0")
    assert value_of(dev, "--arm") == "grpo_seed1" and value_of(dev, "--max-new-tokens") == cap
    assert not {"--styles", "--set", "--families"} & set(dev["args"])
    assert read_json(out / "base_dev" / "run_manifest.json")["run_seed"] == \
        read_json(out / "sft_dev" / "run_manifest.json")["run_seed"] == \
        read_json(out / "grpo_seed1_dev" / "run_manifest.json")["run_seed"] == 0
    # The table: the three arms, and the three comparisons of PLAN 5.7 on dev.
    assert table["args"][2:] == [
        "--arm", f"base={out}/base_dev/episodes.jsonl", "--arm", f"sft={out}/sft_dev/episodes.jsonl",
        "--arm", f"grpo_seed1={out}/grpo_seed1_dev/episodes.jsonl", "--baseline", "base",
        "--pair", "base:sft", "--pair", "sft:grpo_seed1", "--pair", "base:grpo_seed1",
        "--out-json", f"{out}/tables/base_sft_grpo_seed1.json",
        "--out-md", f"{out}/tables/base_sft_grpo_seed1.md"]
    assert (out / "tables" / "base_sft_grpo_seed1.md").is_file()
    # The monitor reports: one file per arm, from its dev episodes.
    for call, arm, folder in zip(monitors, ("base", "sft", "grpo_seed1"), ("base_dev", "sft_dev",
                                                                          "grpo_seed1_dev")):
        assert call["args"][1:] == ["--episodes", f"{out}/{folder}/episodes.jsonl", "--scenarios",
                                    "data/scenarios_placeholder", "--split", "dev", "--text",
                                    "text/placeholder_text.yaml"]
        assert (out / "monitors" / f"{arm}_dev.md").read_text(encoding="utf-8") == \
            f"monitor report of {out}/{folder}/episodes.jsonl\n"
    # The curve: the fixed dev subset in one seen wording, 4 attempts, the run seed of dev, for the SFT
    # adapter and the checkpoints after 20, 40, 60 and 80 steps.
    adapters = [out / "sft" / "adapter"] + [out / "grpo_seed1" / "checkpoints" / f"step_000{step}" / "adapter"
                                            for step in (20, 40, 60, 80)]
    assert [value_of(call, "--adapter") for call in curve] == [str(adapter) for adapter in adapters]
    for call, name in zip(curve, CURVE):
        assert (value_of(call, "--set"), value_of(call, "--styles"), value_of(call, "--samples")) == \
            ("curve_subset", "1", "4")
        assert (value_of(call, "--split"), value_of(call, "--run-seed")) == ("dev", "0")
        assert value_of(call, "--out") == str(out / name) and value_of(call, "--arm") == name
        assert len(read_records(out / name / "episodes.jsonl")) == 20 * 1 * 4

    # A second start: nothing with a model runs again; the reports are built again.
    code, markers, calls = chain.run("chain_stage2.sh")
    assert code == 0, chain.output
    assert markers == ["STEP_SKIPPED grpo_seed1", "STEP_SKIPPED grpo_seed1_dev"] + ran(STAGE_2[2:6]) + \
        [f"STEP_SKIPPED {name}" for name in CURVE] + ["CHAIN_DONE"]
    assert [call["script"] for call in calls] == STAGE_2_SCRIPTS[2:6]
    assert "not compared" not in chain.output


def test_stage_2_takes_the_size_of_a_step_from_the_profile_or_from_the_variables(tmp_path, stage_1_done):
    # The profile says groups of 16 (a decision file that the real script wrote for 7 passes of 8). Both
    # stages run with a token cap that is not the config's: every step of stage 2 gets it, and the log
    # says so.
    assert RunConfig().max_new_tokens != 300
    above = Chain(tmp_path / "above")
    assert above.run("chain_stage1.sh", FAKE_PASSES_sft_train="7", MAX_NEW_TOKENS="300")[0] == 0
    code, _, calls = above.run("chain_stage2.sh", CURVE_STEPS="0", MAX_NEW_TOKENS="300")
    assert code == 0 and step_size(calls[0]) == ("8", "16")
    assert "median family pass rate 0.875" in above.note("a GRPO step is 8 scenarios x 16 attempts, from the")
    assert "Every step gets --max-new-tokens 300" in above.note("the token cap of this chain is 300")
    with_cap = [call for call in calls if call["script"] in ("train_grpo", "rollout")]
    assert len(with_cap) == 3 and {value_of(call, "--max-new-tokens") for call in with_cap} == {"300"}
    # The second seed gets the same step from the same file.
    code, _, calls = above.run("chain_stage2.sh", SEED="2", CURVE_STEPS="0", MAX_NEW_TOKENS="300")
    assert code == 0 and step_size(calls[0]) == ("8", "16") and value_of(calls[0], "--seed") == "2"

    # The two variables win over the file, and the log says that the profile is not used.
    by_hand = stage_2(tmp_path / "by_hand", stage_1_done)
    code, _, calls = by_hand.run("chain_stage2.sh", SCENARIOS_PER_STEP="8", GROUP_SIZE="8", CURVE_STEPS="0")
    assert code == 0 and step_size(calls[0]) == ("8", "8")
    note = by_hand.note("a GRPO step is 8 scenarios x 8 attempts, set by hand")
    assert "gives 16 x 8 and is NOT used" in note
    # One of the two alone is refused before any step.
    code, markers, calls = by_hand.run("chain_stage2.sh", SEED="2", GROUP_SIZE="16")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], [])
    # The size of a step is not given through GRPO_OPTIONS: the chain would not know it.
    code, markers, calls = by_hand.run("chain_stage2.sh", SEED="2", GRPO_OPTIONS="--scenarios-per-step 8")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], [])

    # No profile on the machine and no variable: a seed does not start. The other seed may have the
    # profile, and the two seeds must have the same step.
    no_profile = stage_2(tmp_path / "no_profile", stage_1_done, keep=("sft",))
    code, markers, calls = no_profile.run("chain_stage2.sh", CURVE_STEPS="0")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], []), no_profile.output
    assert "The size of a GRPO step is not known" in no_profile.output
    assert not (no_profile.out / "grpo_seed1").exists()
    # ALLOW_DEFAULT_STEP=1: the default of the plan, and the log says that the rule was not applied.
    code, _, calls = no_profile.run("chain_stage2.sh", CURVE_STEPS="0", ALLOW_DEFAULT_STEP="1")
    defaults = GRPOConfig()
    assert code == 0 and step_size(calls[0]) == (str(defaults.n_scenarios), str(defaults.group_size))
    assert step_size(calls[0]) == ("16", "8")
    assert "was NOT applied" in no_profile.note("a GRPO step is 16 scenarios x 8 attempts, the default")
    # A check before the seeds can run while stage 1 still makes the profile: it takes the default without
    # being told to, and says so.
    pilot = stage_2(tmp_path / "pilot", stage_1_done, keep=("sft",))
    code, markers, calls = pilot.run("chain_stage2.sh", CHECK="pilot")
    assert code == 0 and markers == ran(["pilot"]) + ["CHAIN_DONE"], pilot.output
    assert step_size(calls[0]) == ("16", "8")
    assert "was NOT applied" in pilot.note("a GRPO step is 16 scenarios x 8 attempts, the default")


def test_stage_2_refuses_a_profile_of_another_adapter_or_with_an_impossible_step(tmp_path, stage_1_done):
    chain = stage_2(tmp_path, stage_1_done)
    decision_file = chain.out / "sft_profile" / "decision.json"
    decision = read_json(decision_file)
    decision_file.write_text(json.dumps({**decision, "scenarios_per_step": 4}), encoding="utf-8")
    code, markers, calls = chain.run("chain_stage2.sh")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], [])
    assert "can give only 16 x 8 and 8 x 16" in chain.output
    decision["source"]["adapter_sha256"]["adapter_model.safetensors"] = "0" * 64
    decision_file.write_text(json.dumps(decision), encoding="utf-8")
    code, markers, calls = chain.run("chain_stage2.sh")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], [])
    assert "is the profile of another adapter" in chain.output


def test_stage_2_on_a_machine_without_the_dev_rollouts_of_stage_1(tmp_path, stage_1_done):
    """The second machine gets the SFT run and the decision of the profile. Without base_dev and sft_dev
    there is no table; the chain says so, and builds the monitor report of its own arm."""
    chain = stage_2(tmp_path, stage_1_done, keep=("sft", "sft_profile"))
    code, markers, calls = chain.run("chain_stage2.sh", SEED="2")
    assert code == 0, chain.output
    names = ["grpo_seed2", "grpo_seed2_dev", "monitors_grpo_seed2"] + \
        [name.replace("seed1", "seed2") for name in CURVE]
    assert markers == ran(names) + ["CHAIN_DONE"]
    assert [call["script"] for call in calls] == ["train_grpo", "rollout", "monitor_report", *["rollout"] * 5]
    assert value_of(calls[0], "--seed") == "2" and step_size(calls[0]) == ("16", "8")
    assert "are not both complete on this machine" in chain.note("no table, and no monitor report of base")
    assert not (chain.out / "tables" / "base_sft_grpo_seed2.md").exists()


# Each case: a variable with another value at a start of stage 2 after a complete one, and where the chain
# stops. "start": before any step, because the dev rollouts of stage 1 on the machine would not be paired
# with this chain's dev rollout.
@pytest.mark.parametrize("variable, value, refused_step", [
    ("EVAL_SEED", "5", "start"),
    ("MAX_NEW_TOKENS", "300", "start"),
    ("SAMPLES", "4", "start"),
    ("GRPO_OPTIONS", "--lr 3e-5", "grpo_seed1"),
    ("CURVE_SAMPLES", "8", "grpo_seed1_curve_000"),
    ("CURVE_STYLE", "2", "grpo_seed1_curve_000"),
])
def test_stage_2_does_not_skip_a_step_that_other_values_made(tmp_path, stage_1_done, variable, value,
                                                             refused_step):
    chain = stage_2(tmp_path, stage_1_done)
    assert chain.run("chain_stage2.sh")[0] == 0
    code, markers, calls = chain.run("chain_stage2.sh", **{variable: value})
    assert code == 1, chain.output
    assert markers[-1] == f"CHAIN_FAILED {refused_step}"
    assert not {"train_grpo", "rollout"} & {call["script"] for call in calls}
    assert not list(chain.out.glob("*.unfinished.*"))


def test_stage_2_does_not_skip_a_run_with_another_step_size_or_another_start_adapter(tmp_path, stage_1_done):
    chain = stage_2(tmp_path, stage_1_done)
    assert chain.run("chain_stage2.sh")[0] == 0
    code, markers, calls = chain.run("chain_stage2.sh", SCENARIOS_PER_STEP="8", GROUP_SIZE="16")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED grpo_seed1"], [])
    assert "grpo_seed1: grpo_config/n_scenarios is '16'" in chain.output
    # Without the dev rollouts of stage 1 on the machine a changed EVAL_SEED is found at the dev rollout.
    shutil.rmtree(chain.out / "base_dev")
    shutil.rmtree(chain.out / "sft_dev")
    code, markers, calls = chain.run("chain_stage2.sh", EVAL_SEED="5")
    assert (code, markers, calls) == (1, ["STEP_SKIPPED grpo_seed1", "CHAIN_FAILED grpo_seed1_dev"], [])
    # Another SFT adapter in the same folder: the profile names the old one, and the chain stops at once.
    weights = chain.out / "sft" / "adapter" / "adapter_model.safetensors"
    weights.write_text("other weights", encoding="utf-8")
    code, markers, calls = chain.run("chain_stage2.sh")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], [])
    # Even with the step set by hand, the complete GRPO run names the adapter that it started from.
    code, markers, calls = chain.run("chain_stage2.sh", SCENARIOS_PER_STEP="16", GROUP_SIZE="8")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED grpo_seed1"], [])
    assert "start_adapter/files/adapter_model.safetensors/sha256" in chain.output


def test_stage_2_stops_if_sft_dev_is_not_of_the_adapter_that_grpo_starts_from(tmp_path, stage_1_done):
    """The table calls the dev rollout of stage 1 the arm "sft" and compares GRPO with it. That is right
    only if sft_dev was sampled with the adapter that GRPO starts from (SFT_ADAPTER). The step is set by
    hand in every start here: then the profile is not read, so its own check of the adapter cannot be the
    one that stops the chain."""
    by_hand = {"SCENARIOS_PER_STEP": "16", "GROUP_SIZE": "8"}
    # Another adapter folder with other weights, as a second SFT seed would leave it.
    chain = stage_2(tmp_path / "other_folder", stage_1_done)
    other = chain.out / "sft_seed1" / "adapter"
    shutil.copytree(chain.out / "sft" / "adapter", other)
    (other / "adapter_model.safetensors").write_text("weights of another SFT run", encoding="utf-8")
    code, markers, calls = chain.run("chain_stage2.sh", SFT_ADAPTER=str(other), **by_hand)
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], []), chain.output
    assert "sft_dev: what the chain noted" in chain.output
    assert "was not sampled with the adapter that GRPO starts from" in chain.output
    assert not (chain.out / "grpo_seed1").exists()
    assert not (chain.out / "tables" / "base_sft_grpo_seed1.md").exists()
    # The weights decide, not the name of the folder: a copy of the SFT adapter in another folder is the
    # model that sft_dev was sampled with, and the chain runs.
    shutil.copy(chain.out / "sft" / "adapter" / "adapter_model.safetensors", other)
    code, markers, calls = chain.run("chain_stage2.sh", SFT_ADAPTER=str(other), CURVE_STEPS="0", **by_hand)
    assert code == 0 and markers[-1] == "CHAIN_DONE", chain.output
    assert value_of(calls[0], "--adapter") == str(other) and calls[2]["script"] == "tables"
    # The same folder, but SFT was trained again into it after sft_dev was sampled: the folder name is the
    # same, and the chain's note in sft_dev names the SHA-256 of the old weights.
    chain = stage_2(tmp_path / "trained_again", stage_1_done)
    (chain.out / "sft" / "adapter" / "adapter_model.safetensors").write_text("new weights", encoding="utf-8")
    code, markers, calls = chain.run("chain_stage2.sh", **by_hand)
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], []), chain.output
    assert "sft_dev: what the chain noted" in chain.output
    # A check before the seeds builds no table, but it stops in the same way: the folder is not right.
    code, markers, calls = chain.run("chain_stage2.sh", CHECK="pilot", **by_hand)
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], []), chain.output
    # An sft_dev that this chain script did not run has no note. Then only the folder name of its run
    # manifest can be compared (here the folder that stage 1 wrote to, before the results were copied):
    # another folder is refused, and for that folder the log says that the weights were not compared.
    chain = stage_2(tmp_path / "no_note", stage_1_done)
    (chain.out / "sft_dev" / "chain_record.txt").unlink()
    code, markers, calls = chain.run("chain_stage2.sh", **by_hand)
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], []), chain.output
    assert f"sft_dev: adapter is '{stage_1_done}/sft/adapter'" in chain.output
    code, markers, calls = chain.run("chain_stage2.sh", SFT_ADAPTER=f"{stage_1_done}/sft/adapter",
                                     CURVE_STEPS="0", **by_hand)
    assert code == 0 and markers[-1] == "CHAIN_DONE", chain.output
    assert f"sft_dev: {chain.out}/sft_dev has no chain_record.txt" in chain.output
    assert "tables" in [call["script"] for call in calls]


def test_a_run_that_the_stop_rule_ended_still_gets_its_curve_and_is_not_started_again(tmp_path,
                                                                                      stage_1_done):
    """The stand-in ends the GRPO run by the stop rule (exit code 3) with checkpoints after 10, 20, 30 and
    40 steps. The chain runs the curve for the SFT adapter and for the checkpoints 20 and 40, says that 60
    and 80 do not exist, and ends as failed with code 3. There is no dev rollout, table or monitor report."""
    chain = stage_2(tmp_path, stage_1_done)
    code, markers, calls = chain.run("chain_stage2.sh", FAKE_FAIL="grpo_seed1:3:stopped_by_rule:40")
    assert code == 3
    assert markers == ["STEP_START grpo_seed1", "STEP_DONE grpo_seed1 3"] + ran(CURVE[:3]) + \
        ["CHAIN_FAILED grpo_seed1"]
    assert [call["script"] for call in calls] == ["train_grpo", "rollout", "rollout", "rollout"]
    assert [value_of(call, "--adapter") for call in calls[1:]] == [
        str(chain.out / "sft" / "adapter"),
        str(chain.out / "grpo_seed1" / "checkpoints" / "step_00020" / "adapter"),
        str(chain.out / "grpo_seed1" / "checkpoints" / "step_00040" / "adapter")]
    note = chain.note("the stop rule ended grpo_seed1")
    assert "The curve steps run for the checkpoints that exist" in note
    assert "There is no dev rollout, no table and no monitor report of this run." in note
    assert len([line for line in chain.notes if "no curve step" in line]) == 2
    assert "no checkpoint" in chain.note("no curve step 60 of grpo_seed1")
    assert "a made-up fault" in chain.output
    assert not (chain.out / "grpo_seed1_dev").exists()
    assert not (chain.out / "monitors" / "sft_dev.md").exists()

    # A second start does not train again. The curve steps are complete and are skipped.
    code, markers, calls = chain.run("chain_stage2.sh")
    assert (code, calls) == (3, [])
    assert markers == [f"STEP_SKIPPED {name}" for name in CURVE[:3]] + ["CHAIN_FAILED grpo_seed1"]
    assert read_json(chain.out / "grpo_seed1" / "run_manifest.json")["status"] == "stopped_by_rule"
    assert (chain.out / "grpo_seed1" / "stopped.json").is_file()

    # RESUME_AFTER_FAULT=1 is the only way to continue such a run, and the log says that it was used.
    code, markers, calls = chain.run("chain_stage2.sh", RESUME_AFTER_FAULT="1")
    assert code == 0, chain.output
    assert calls[0]["script"] == "train_grpo" and calls[0]["args"][-2:] == ["--resume", RESUME_AFTER_FAULT]
    assert "is continued from its last checkpoint" in chain.note("RESUME_AFTER_FAULT=1")
    assert markers[:2] == ran(["grpo_seed1"]) and markers[-1] == "CHAIN_DONE"


def test_stage_2_continues_a_run_that_has_a_checkpoint_and_restarts_one_that_has_none(tmp_path, stage_1_done):
    chain = stage_2(tmp_path / "a", stage_1_done)
    code, markers, _ = chain.run("chain_stage2.sh", FAKE_FAIL="grpo_seed1:1:failed:30")
    assert code == 1 and markers == ["STEP_START grpo_seed1", "STEP_DONE grpo_seed1 1",
                                     "CHAIN_FAILED grpo_seed1"]
    code, markers, calls = chain.run("chain_stage2.sh", GRPO_OPTIONS="--lr 3e-5")
    assert code == 0, chain.output
    assert [call["script"] for call in calls] == STAGE_2_SCRIPTS
    # The options of GRPO_OPTIONS come after the chain's own, and --resume is last. Only train_grpo.py
    # gets them.
    assert calls[0]["args"][-3:] == ["--lr", "3e-5", "--resume"]
    assert all("--lr" not in call["args"] for call in calls[1:])
    assert not [line for line in markers if line.startswith("STEP_MOVED_ASIDE")]

    other = stage_2(tmp_path / "b", stage_1_done)
    code, _, _ = other.run("chain_stage2.sh", FAKE_FAIL="grpo_seed1:1")
    assert code == 1
    code, markers, calls = other.run("chain_stage2.sh")
    assert code == 0 and "--resume" not in calls[0]["args"]
    assert markers[0] == f"STEP_MOVED_ASIDE grpo_seed1 {other.out}/grpo_seed1.unfinished"


def test_a_failed_report_of_stage_2_does_not_cost_the_curve(tmp_path, stage_1_done):
    chain = stage_2(tmp_path, stage_1_done)
    code, markers, calls = chain.run("chain_stage2.sh", FAKE_FAIL="monitor_report:2")
    assert code == 2
    assert markers == ran(STAGE_2[:3]) + [line for name in STAGE_2[3:6]
                                          for line in (f"STEP_START {name}", f"STEP_DONE {name} 2")] + \
        ran(CURVE) + ["CHAIN_FAILED monitors_base"]
    assert [call["script"] for call in calls] == STAGE_2_SCRIPTS
    assert not list((chain.out / "monitors").iterdir())          # no half-written report is left


def test_the_overfit_check_picks_two_mixed_families_and_trains_on_them(tmp_path, stage_1_done):
    """CHECK=overfit. The untrained model's train rollout of the copied stage 1 has, in the order of the
    families, 8, 0, 3, 5, 6, 4, 4 and 7 passes of 8 attempts, and so on in a circle. The first two families
    with 4 passes are the sixth and the seventh of train.jsonl."""
    chain = stage_2(tmp_path, stage_1_done)
    code, markers, calls = chain.run("chain_stage2.sh", CHECK="overfit")
    assert code == 0, chain.output
    assert markers == ran(["overfit_families", "overfit"]) + ["CHAIN_DONE"]
    pick, train = calls
    assert (pick["script"], train["script"]) == ("pick_overfit_families", "train_grpo")
    families_file = chain.out / "checks" / "overfit_families.txt"
    assert value_of(pick, "--episodes") == str(chain.out / "base_train" / "episodes.jsonl")
    assert (value_of(pick, "--out"), value_of(pick, "--style")) == (str(families_file), "1")
    lines = (PLACEHOLDER_SCENARIOS / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert families_file.read_text(encoding="utf-8").split() == \
        [json.loads(lines[5])["family_id"], json.loads(lines[6])["family_id"]]
    # The check: these families only, one wording, seed 101, its own folder, the looser limit on harness
    # errors, the group size of the seeds, and no number of scenarios (every step takes both).
    assert value_of(train, "--overfit") == str(families_file) and value_of(train, "--styles") == "1"
    assert value_of(train, "--seed") == "101"
    assert value_of(train, "--out") == str(chain.out / "checks" / "overfit")
    assert value_of(train, "--max-harness-error-rate") == "0.1" and value_of(train, "--group-size") == "8"
    assert value_of(train, "--adapter") == str(chain.out / "sft" / "adapter")
    assert "--save-tokens" in train["args"]
    assert not {"--scenarios-per-step", "--pilot", "--resume"} & set(train["args"])
    note = chain.note("overfit, numbers of the training episodes")
    assert "pass rate 0.5 at the first step and 0.875 at the last step; reward 48.5 and 86.25" in note
    # No step of a seed ran.
    assert not (chain.out / "grpo_seed1").exists() and not list(chain.out.glob("*_curve_*"))
    # The log says that the SFT model's own attempts are on the machine and how to choose from them.
    sft_train = chain.out / "sft_train" / "episodes.jsonl"
    assert f"OVERFIT_EPISODES={sft_train} chooses from them" in chain.note("the overfit families are chosen")
    from_sft = stage_2(tmp_path / "from_sft", stage_1_done)
    sft_train = from_sft.out / "sft_train" / "episodes.jsonl"
    code, _, calls = from_sft.run("chain_stage2.sh", CHECK="overfit", OVERFIT_EPISODES=str(sft_train))
    assert code == 0 and value_of(calls[0], "--episodes") == str(sft_train)
    assert not [line for line in from_sft.notes if "the overfit families are chosen" in line]
    # In that rollout every family passes 6 of 8 attempts, so all are 2 away from 4: the two smallest ids
    # of the 100 profile families are taken.
    profile_ids = (from_sft.out / "sft_profile" / "families.txt").read_text(encoding="utf-8").split()
    assert (from_sft.out / "checks" / "overfit_families.txt").read_text(encoding="utf-8").split() == \
        sorted(profile_ids)[:2]

    # A second start skips the check and does not choose the families again.
    code, markers, calls = chain.run("chain_stage2.sh", CHECK="overfit")
    assert (code, markers, calls) == (0, ["STEP_SKIPPED overfit", "CHAIN_DONE"], [])
    # With too few mixed families the pick ends the chain (exit code 1 of the script).
    other = stage_2(tmp_path / "other", stage_1_done)
    episodes = other.out / "base_train" / "episodes.jsonl"
    write_records(episodes, [{**record, "passed": True} for record in read_records(episodes)])
    code, markers, calls = other.run("chain_stage2.sh", CHECK="overfit")
    assert code == 1 and markers == ["STEP_START overfit_families", "STEP_DONE overfit_families 1",
                                     "CHAIN_FAILED overfit_families"]
    assert [call["script"] for call in calls] == ["pick_overfit_families"]


def test_the_families_of_an_overfit_run_that_is_kept_are_not_chosen_again(tmp_path, stage_1_done):
    """checks/overfit_families.txt is what the run in checks/overfit read. Families are chosen only for a
    check that starts from its first step. A run that the stop rule ended, and a run that is continued
    from a checkpoint, keep their file; a start with another OVERFIT_EPISODES is refused and writes
    nothing."""
    def pick_files(chain: Chain) -> dict[str, bytes]:
        return {name: content for name, content in files_of(chain.out / "checks").items()
                if name.startswith("overfit_families.txt")}

    chain = stage_2(tmp_path / "stopped", stage_1_done)
    code, markers, calls = chain.run("chain_stage2.sh", CHECK="overfit",
                                     FAKE_FAIL="overfit:3:stopped_by_rule:10")
    assert code == 3 and markers == ran(["overfit_families"]) + \
        ["STEP_START overfit", "STEP_DONE overfit 3", "CHAIN_FAILED overfit"]
    files = pick_files(chain)
    assert sorted(files) == ["overfit_families.txt", "overfit_families.txt.json"]
    # A second start: the run is reported as stopped again, and no family is chosen.
    code, markers, calls = chain.run("chain_stage2.sh", CHECK="overfit")
    assert (code, markers, calls) == (3, ["CHAIN_FAILED overfit"], []), chain.output
    assert "are not chosen again" in chain.output and pick_files(chain) == files
    # A start that would choose from other episodes (here the SFT model's own, with other families) is
    # refused before any script runs.
    sft_train = chain.out / "sft_train" / "episodes.jsonl"
    code, markers, calls = chain.run("chain_stage2.sh", CHECK="overfit", OVERFIT_EPISODES=str(sft_train))
    assert (code, markers, calls) == (1, ["CHAIN_FAILED overfit"], []), chain.output
    assert f"were chosen from {chain.out}/base_train/episodes.jsonl" in chain.output
    assert pick_files(chain) == files
    # A complete check is refused in the same way.
    done = stage_2(tmp_path / "done", stage_1_done)
    assert done.run("chain_stage2.sh", CHECK="overfit")[0] == 0
    files = pick_files(done)
    sft_train = done.out / "sft_train" / "episodes.jsonl"
    code, markers, calls = done.run("chain_stage2.sh", CHECK="overfit", OVERFIT_EPISODES=str(sft_train))
    assert (code, markers, calls) == (1, ["CHAIN_FAILED overfit"], []), done.output
    assert pick_files(done) == files

    # A run that failed for another reason and has a checkpoint is continued, on its own families.
    resumed = stage_2(tmp_path / "resumed", stage_1_done)
    assert resumed.run("chain_stage2.sh", CHECK="overfit", FAKE_FAIL="overfit:1:failed:20")[0] == 1
    files = pick_files(resumed)
    code, markers, calls = resumed.run("chain_stage2.sh", CHECK="overfit")
    assert code == 0 and markers == ran(["overfit"]) + ["CHAIN_DONE"], resumed.output
    assert [call["script"] for call in calls] == ["train_grpo"] and calls[0]["args"][-1] == "--resume"
    assert pick_files(resumed) == files
    # A run without a checkpoint starts again from its first step: its folder is moved aside, and the
    # families are chosen again.
    again = stage_2(tmp_path / "again", stage_1_done)
    assert again.run("chain_stage2.sh", CHECK="overfit", FAKE_FAIL="overfit:1")[0] == 1
    code, markers, calls = again.run("chain_stage2.sh", CHECK="overfit")
    assert code == 0, again.output
    assert markers == ran(["overfit_families"]) + \
        [f"STEP_MOVED_ASIDE overfit {again.out}/checks/overfit.unfinished"] + ran(["overfit"]) + ["CHAIN_DONE"]
    assert [call["script"] for call in calls] == ["pick_overfit_families", "train_grpo"]


def test_the_pilot_runs_15_steps_at_the_real_settings_in_its_own_folder(tmp_path, stage_1_done):
    chain = stage_2(tmp_path, stage_1_done)
    code, markers, calls = chain.run("chain_stage2.sh", CHECK="pilot", GRPO_OPTIONS="--reference-tokens 200")
    assert code == 0, chain.output
    assert markers == ran(["pilot"]) + ["CHAIN_DONE"]
    (train,) = calls
    assert train["script"] == "train_grpo" and "--pilot" in train["args"] and "--save-tokens" in train["args"]
    assert value_of(train, "--seed") == "100"
    assert value_of(train, "--out") == str(chain.out / "checks" / "pilot")
    # The real settings: the step of the profile, the token cap of the chain, and the options of the seeds.
    assert step_size(train) == ("16", "8") and value_of(train, "--reference-tokens") == "200"
    assert value_of(train, "--max-new-tokens") == str(RunConfig().max_new_tokens)
    assert not {"--overfit", "--styles", "--steps", "--max-harness-error-rate"} & set(train["args"])
    assert read_json(chain.out / "checks" / "pilot" / "run_manifest.json")["mode"] == "pilot"
    assert "pass rate 0.5 at the first step" in chain.note("pilot, numbers of the training episodes")
    # A check that the stop rule ended is reported as failed, with exit code 3.
    other = stage_2(tmp_path / "other", stage_1_done)
    code, markers, _ = other.run("chain_stage2.sh", CHECK="pilot", FAKE_FAIL="pilot:3:stopped_by_rule:10")
    assert code == 3 and markers == ["STEP_START pilot", "STEP_DONE pilot 3", "CHAIN_FAILED pilot"]
    # CHECK takes only its two words.
    code, markers, calls = other.run("chain_stage2.sh", CHECK="both")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], [])


def test_stage_2_needs_the_sft_adapter(tmp_path):
    chain = Chain(tmp_path)
    code, markers, calls = chain.run("chain_stage2.sh")
    assert (code, markers, calls) == (1, ["CHAIN_FAILED start"], [])
    assert f"{chain.out}/sft/adapter/adapter_config.json is missing" in chain.output
