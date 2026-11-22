"""Tests of scripts/stretch_numbers.py: the numbers, the tables and the document of the stretch item.

The saved runs of these tests are made up, in a temporary folder: four arms on six dev families, two small
GRPO training runs, and attempts of the SFT model on four training families. Every expected number below is
worked out by hand in the comments. Nothing is written into the repository, and no test needs results/,
except the last one, which runs the script on the real saved runs and is skipped if they are not there.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from accessdesk import impl
from accessdesk.eval import metrics, tables
from tests.helpers.make_records import make_record

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ("grant", "approval", "already", "inactive")
GRPO = ("grpo_a", "grpo_b")
ARMS = ("base", "sft") + GRPO

# Passes of 8 attempts per wording style (1, 2, 3, 4) of each dev family, per arm.
#   dv-04 is the family that got worse: after training its wordings disagree in sft and grpo_a (8 and 8
#   against 1 or 0), and in grpo_b it fails in every style.
DEV = {
    "base":   {"dv-00": (8, 8, 8, 8), "dv-01": (6, 2, 4, 4), "dv-02": (0, 0, 0, 0), "dv-03": (4, 4, 4, 4),
               "dv-04": (8, 8, 0, 0), "dv-05": (5, 5, 5, 3)},
    "sft":    {"dv-00": (8, 8, 8, 8), "dv-01": (8, 8, 8, 8), "dv-02": (4, 4, 4, 4), "dv-03": (8, 8, 8, 6),
               "dv-04": (8, 8, 7, 1), "dv-05": (8, 8, 8, 8)},
    "grpo_a": {"dv-00": (8, 8, 8, 8), "dv-01": (8, 8, 8, 8), "dv-02": (8, 8, 8, 8), "dv-03": (8, 8, 8, 7),
               "dv-04": (8, 8, 1, 0), "dv-05": (8, 8, 8, 8)},
    "grpo_b": {"dv-00": (8, 8, 8, 8), "dv-01": (8, 8, 8, 8), "dv-02": (8, 8, 8, 8), "dv-03": (8, 8, 8, 8),
               "dv-04": (3, 0, 0, 0), "dv-05": (8, 7, 8, 8)},
}
# The same four arms on a made-up test set of four families (only used when a test asks for it).
TEST_SET = {
    "base":   {"te-00": (4, 4, 4, 4), "te-01": (0, 0, 0, 0), "te-02": (8, 8, 8, 8), "te-03": (2, 2, 2, 2)},
    "sft":    {"te-00": (8, 8, 8, 8), "te-01": (4, 4, 4, 4), "te-02": (8, 8, 8, 8), "te-03": (8, 8, 8, 8)},
    "grpo_a": {"te-00": (8, 8, 8, 8), "te-01": (8, 8, 8, 4), "te-02": (8, 8, 8, 8), "te-03": (8, 8, 8, 8)},
    "grpo_b": {"te-00": (8, 8, 8, 8), "te-01": (8, 8, 8, 8), "te-02": (8, 8, 8, 8), "te-03": (8, 8, 8, 8)},
}
# The rewards of the attempts of each group of each training step: (scenario id, rewards of its 4 attempts).
#   grpo_a   step 0: a pass-and-fail group at the full reward (kept); a group of equal rewards (dropped)
#            step 1: an all-pass group with a penalty (kept); an all-fail group (dropped). Both are wordings
#                    of the family tr-002, so each has another wording of its family in the step
#            step 2: a pass-and-fail group with a pass at 95 (kept); a pass-and-fail group at 100 (kept)
#   grpo_b   step 0: kept (full reward), dropped; step 1: dropped, dropped; step 2: kept (all pass, with
#            penalties), kept (full reward)
TRAINING = {
    "grpo_a": [[("tr-000/s1", (100, 100, 0, 0)), ("tr-001/s2", (100, 100, 100, 100))],
               [("tr-002/s1", (100, 95, 100, 100)), ("tr-002/s3", (0, 0, 0, 0))],
               [("tr-003/s2", (100, 95, 0, 0)), ("tr-004/s1", (100, 0, 0, 0))]],
    "grpo_b": [[("tr-010/s1", (100, 0, 100, 0)), ("tr-011/s1", (100, 100, 100, 100))],
               [("tr-012/s2", (0, 0, 0, 0)), ("tr-013/s3", (100, 100, 100, 100))],
               [("tr-014/s1", (95, 100, 100, 88)), ("tr-015/s2", (100, 100, 100, 0))]],
}
KEPT_PER_STEP = {"grpo_a": [1, 1, 2], "grpo_b": [1, 0, 2]}
TRAINING_SECONDS = {"grpo_a": 3600.0, "grpo_b": 7200.0}      # 1 and 2 GPU hours
# The ends of the training runs: the prices of the cost are those of the machines rented up to the later one.
TRAINING_ENDED = {"grpo_a": "2026-01-01T09:00:00Z", "grpo_b": "2026-01-01T10:00:00Z"}
DEV_RUN_SECONDS = 1800.0                                       # half a GPU hour, for every dev run
# Attempts of the SFT model on four training families: the rewards of the 8 attempts of each of 3 styles.
#   Draw 1 takes the attempts 1 to 4 of each style, draw 2 the attempts 5 to 8.
#   tr-100  every attempt passes                                   -> equal rewards in both draws
#   tr-101  draw 1: style 1 passes 4, style 2 passes 2, style 3 0  -> the example of the document
#           draw 2: every attempt passes                           -> equal rewards
#   tr-102  every attempt fails                                    -> equal rewards in both draws
#   tr-103  draw 1: every style passes 2 of 4                      -> kept, and the bonus is zero
#           draw 2: every attempt passes, one with a penalty (95)  -> kept, and the bonus is zero
P, F = 100, 0
SFT_TRAIN = {
    "tr-100": {1: [P] * 8, 2: [P] * 8, 3: [P] * 8},
    "tr-101": {1: [P] * 8, 2: [P, P, F, F, P, P, P, P], 3: [F, F, F, F, P, P, P, P]},
    "tr-102": {1: [F] * 8, 2: [F] * 8, 3: [F] * 8},
    "tr-103": {1: [P, F, P, F, P, P, P, P], 2: [P, F, P, F, P, P, P, P], 3: [P, F, P, F, 95, P, P, P]},
}
MALFORMED = {100: 0, 95: 1, 88: 2}
# The document must stay short (two pages): at most this many characters, also after the test runs are
# copied in. Then one sentence of 66 characters gives way to the "Test set" sentence; its widest form (every
# value 100.0, 48 families, the names of the real arms) is at most this much longer
# (test_the_widest_test_set_sentence_fits_the_room_kept_for_it checks it).
DOC_CHARACTERS = 8100
MOST_GROWTH_WITH_THE_TEST_SET = 175


# ---------------------------------------------------------------------------
# The script, and the made-up saved runs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("stretch_numbers", ROOT / "scripts" / "stretch_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def main_versions(monkeypatch):
    """The main versions of the core functions, whatever ACCESSDESK_IMPL says outside the test."""
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")


def attempt(family_id: str, style: int, sample: int, reward: int, *, arm: str, run: str,
            outcome: str = "grant", close: str = "denied", person: str = "USRright") -> dict:
    """One attempt record. A fail closes the request as `close` for `person`."""
    record = make_record(family_id, style, sample, passed=reward > 0, reward=reward, outcome=outcome,
                         n_malformed=MALFORMED.get(reward, 0), arm=arm, run=run)
    if reward <= 0:
        record["final_state"]["request"] = {"resolution": close, "person_id": person, "role_id": "ROLEx"}
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rollout_records(patterns: dict[str, tuple], arm: str, run: str) -> list[dict]:
    """8 attempts for each of the 4 wording styles of each family: the first `passes` of them pass.

    In dv-04 a failed attempt in style 4 closes the request for another person; all fails close it as
    denied."""
    records = []
    for index, (family_id, passes) in enumerate(sorted(patterns.items())):
        for style, n_passes in enumerate(passes, start=1):
            for sample in range(8):
                person = "USRother" if (family_id == "dv-04" and style == 4) else "USRright"
                records.append(attempt(family_id, style, sample, P if sample < n_passes else F, arm=arm,
                                       run=run, outcome=OUTCOMES[index % 4], person=person))
    return records


def write_rollout(folder: Path, records: list[dict], seconds: float, *, complete: bool = True) -> None:
    write_jsonl(folder / "episodes.jsonl", records)
    write_json(folder / "run_manifest.json", {"complete": complete, "seconds": seconds, "run_seed": 0,
                                              "adapter": None, "environment": {"gpu_name": "TEST GPU"}})


def write_training_run(folder: Path, arm: str, *, steps=None, kept=None, group_size: int = 4) -> None:
    steps = TRAINING[arm] if steps is None else steps
    kept = KEPT_PER_STEP[arm] if kept is None else kept
    log = []
    for step, groups in enumerate(steps):
        records = []
        for scenario_id, rewards in groups:
            family_id, style = scenario_id.split("/s")
            records += [attempt(family_id, int(style), sample, reward, arm="grpo_train", run=arm)
                        for sample, reward in enumerate(rewards)]
        write_jsonl(folder / "rollouts" / f"step_{step:05d}.episodes.jsonl", records)
        log.append({"step": step, "scenarios": [scenario_id for scenario_id, _ in groups],
                    "kept_groups": kept[step], "seconds": TRAINING_SECONDS[arm] / len(steps)})
    write_jsonl(folder / "train_log.jsonl", log)
    write_json(folder / "run_manifest.json", {
        "complete": True, "status": "complete", "seconds": TRAINING_SECONDS[arm],
        "ended_utc": TRAINING_ENDED[arm],
        "grpo_config": {"n_scenarios": len(steps[0]), "group_size": group_size, "tie_tol": 0.0,
                        "steps": len(steps), "lr": 1e-05, "seed": 1},
        "config": {"pass_reward": 100}, "styles": [1, 2, 3], "environment": {"gpu_name": "TEST GPU"},
        "start_adapter": {"folder": "made-up/sft/adapter"}})


def write_scenarios(folder: Path) -> None:
    """A scenario folder with the dev family that got worse, and a test.jsonl that must never be opened."""
    family = {"family_id": "dv-04", "outcome": "grant",
              "truth": {"person_id": "USRright"}, "request": {"person_name": "Kim Roe"},
              "world": {"people": [{"person_id": "USRright", "name": "Kim Roe", "status": "active"},
                                   {"person_id": "USRother", "name": "Kim Roe", "status": "inactive"},
                                   {"person_id": "USRthird", "name": "Ana Poe", "status": "active"}]},
              "expected": {"default": {"request": {"resolution": "granted"}}}}
    write_jsonl(folder / "dev.jsonl", [family])
    (folder / "test.jsonl").write_text("this file must never be opened\n", encoding="utf-8")


def build_tree(tmp_path: Path, *, dev: dict | None = None, with_test: bool = False) -> dict:
    """Write the made-up saved runs. Returns the paths that the tests give to the script."""
    main = tmp_path / "results" / "main"
    dev = DEV if dev is None else dev
    for arm in ARMS:
        write_rollout(main / f"{arm}_dev", rollout_records(dev[arm], arm, f"{arm}_dev"), DEV_RUN_SECONDS)
        if with_test:
            write_rollout(main / f"{arm}_test", rollout_records(TEST_SET[arm], arm, f"{arm}_test"), 60.0)
    for arm in GRPO:
        write_training_run(main / arm, arm)
    sft_train = [attempt(family_id, style, sample, reward, arm="sft_train", run="sft_train")
                 for family_id, styles in SFT_TRAIN.items() for style, rewards in styles.items()
                 for sample, reward in enumerate(rewards)]
    write_rollout(main / "sft_train", sft_train, 100.0)
    ledger = tmp_path / "results" / "spend_ledger.csv"
    ledger.write_text("time_utc,event,instance_id,gpu,dph_usd,actual_usd\n"
                      "2026-01-01T00:00:00Z,rent,1,TEST GPU,0.5,\n2026-01-01T03:00:00Z,destroy,1,TEST GPU,0.5,1.0\n"
                      "2026-01-01T04:00:00Z,rent,2,TEST GPU,1.0,\n",
                      encoding="utf-8")
    scenarios = tmp_path / "data" / "scenarios"
    write_scenarios(scenarios)
    return {"main": main, "ledger": ledger, "scenarios": scenarios, "out": main / "report" / "stretch",
            "doc": tmp_path / "results" / "stretch.md"}


def arguments(paths: dict, *extra: str) -> list[str]:
    return ["--main", str(paths["main"]), "--grpo", "grpo_a", "--grpo", "grpo_b", "--ledger",
            str(paths["ledger"]), "--scenarios", str(paths["scenarios"]), "--draws", "2000", *extra]


def run(script, paths: dict, *extra: str) -> dict:
    """Run the script on the made-up runs; return its numbers. The exit code must be 0."""
    assert script.main(arguments(paths, *extra)) == 0
    return json.loads((paths["out"] / script.JSON_NAME).read_text(encoding="utf-8"))


def files_of(folder: Path) -> dict[str, str]:
    """Every file below the folder -> the SHA-256 of its bytes."""
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


@pytest.fixture(scope="module")
def made(script, tmp_path_factory) -> dict:
    """The made-up runs, the script run once on them, and what it wrote."""
    tmp_path = tmp_path_factory.mktemp("stretch")
    paths = build_tree(tmp_path)
    before = files_of(tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ACCESSDESK_IMPL", "main")
        numbers = run(script, paths)
    return {"paths": paths, "numbers": numbers, "before": before, "after": files_of(tmp_path),
            "root": tmp_path, "doc": paths["doc"].read_text(encoding="utf-8"),
            "tables": (paths["out"] / script.TABLES_NAME).read_text(encoding="utf-8")}


# ---------------------------------------------------------------------------
# The arithmetic, on hand-made groups
# ---------------------------------------------------------------------------
def test_a_bonus_on_every_pass_of_a_one_wording_group_multiplies_every_advantage(script):
    """12 passes and 4 fails. Without: mean 1200 / 16 = 75, pass +25, fail -75. With 20 points on a pass:
    mean 12 * 120 / 16 = 90, pass +30, fail -90. Each is 1.2 times the value before."""
    example = script.one_wording_example(12, 4, 20)
    assert (example["passes"], example["fails"], example["attempts"]) == (12, 4, 16)
    assert example["without"]["mean_of_the_group"] == pytest.approx(75.0)
    assert example["without"]["advantage_of_a_pass"] == pytest.approx(25.0)
    assert example["without"]["advantage_of_a_fail"] == pytest.approx(-75.0)
    assert example["with"]["reward_of_a_pass"] == 120
    assert example["with"]["mean_of_the_group"] == pytest.approx(90.0)
    assert example["with"]["advantage_of_a_pass"] == pytest.approx(30.0)
    assert example["with"]["advantage_of_a_fail"] == pytest.approx(-90.0)
    assert example["factor"] == pytest.approx(1.2) and example["factor_by_formula"] == pytest.approx(1.2)


@pytest.mark.parametrize("n_passes, n_fails, bonus", [(1, 15, 20), (15, 1, 5), (8, 8, 37.5), (3, 2, 100)])
def test_the_factor_of_the_formula_holds_for_other_groups_and_bonuses(script, n_passes, n_fails, bonus):
    example = script.one_wording_example(n_passes, n_fails, bonus)
    assert example["factor"] == pytest.approx((100 + bonus) / 100)
    k, n = n_passes, n_passes + n_fails
    assert example["with"]["advantage_of_a_pass"] == pytest.approx((100 + bonus) * (1 - k / n))
    assert example["with"]["advantage_of_a_fail"] == pytest.approx(-(100 + bonus) * k / n)


def test_the_family_group_of_the_document_by_hand(script):
    """Wording A passes 4 of 4, B 2 of 4, C 0 of 4; the bonus is 20 points.

    q of a pass in A: the other wordings (B, C) hold 2 passes in 8 attempts = 0.25.
    q of a pass in B: the other wordings (A, C) hold 4 passes in 8 attempts = 0.5.
    q_mean over the 6 passes: (4 * 0.25 + 2 * 0.5) / 6 = 1/3.
    Bonus of a pass in A: 20 * (0.25 - 1/3) = -5/3. In B: 20 * (0.5 - 1/3) = +10/3. Sum: -20/3 + 20/3 = 0.
    Arm 1: mean 600 / 12 = 50, so a pass has +50 and a fail -50.
    Arm 2: the mean stays 50; a pass in A has 50 - 5/3 = 48.33, in B 50 + 10/3 = 53.33, a fail -50."""
    rewards = [100] * 4 + [100, 100, 0, 0] + [0] * 4
    group = script.family_group(rewards, [reward > 0 for reward in rewards], list("AAAABBBBCCCC"), 20)
    assert group["mean_share"] == pytest.approx(1 / 3)
    assert group["bonus"][:4] == pytest.approx([-5 / 3] * 4)
    assert group["bonus"][4:6] == pytest.approx([10 / 3] * 2)
    assert group["bonus"][6:] == [0.0] * 6                      # a fail never gets a bonus
    assert sum(group["bonus"]) == pytest.approx(0.0, abs=1e-12)
    assert group["arm1"] == pytest.approx([50.0] * 6 + [-50.0] * 6)
    assert group["arm2"] == pytest.approx([50 - 5 / 3] * 4 + [50 + 10 / 3] * 2 + [-50.0] * 6)
    assert group["kept_arm1"] and group["kept_arm2"]
    example = script.family_example(20, 4)
    assert example["wordings"]["A"]["arm2_advantage_of_a_pass"] == pytest.approx(50 - 5 / 3)
    assert example["wordings"]["B"]["arm2_advantage_of_a_pass"] == pytest.approx(50 + 10 / 3)
    assert example["wordings"]["B"]["arm2_advantage_of_a_fail"] == pytest.approx(-50.0)
    assert "bonus_of_a_pass" not in example["wordings"]["C"]
    assert example["sum_of_the_bonuses"] == pytest.approx(0.0, abs=1e-12)


def test_the_arm_1_advantage_of_a_family_group_is_the_function_of_the_training(script):
    rewards = [100, 95, 0, 100, 0, 0, 88, 100, 0]
    group = script.family_group(rewards, [reward > 0 for reward in rewards], [1, 1, 1, 2, 2, 2, 3, 3, 3], 20)
    expected, keep = impl.main_version("group_advantages")(np.array([rewards], dtype=float) / 100, 0.0)
    assert group["arm1"] == pytest.approx(list(expected[0] * 100)) and keep[0] and group["kept_arm1"]
    # Arm 2 only moves the passes, and it moves them by amounts that add up to zero.
    moved = [after - before for before, after in zip(group["arm1"], group["arm2"])]
    assert moved == pytest.approx(group["bonus"])
    assert sum(moved) == pytest.approx(0.0, abs=1e-9)
    assert all(amount == 0.0 for amount, reward in zip(group["bonus"], rewards) if reward == 0)


@pytest.mark.parametrize("rewards, wordings", [
    ([100] * 12, list("AAAABBBBCCCC")),                          # every attempt passes: q is 1 for all
    ([0] * 12, list("AAAABBBBCCCC")),                            # no pass, so no bonus
    ([100, 0, 100, 0] * 3, list("AAAABBBBCCCC")),                # every wording passes 2 of 4: equal q
    ([100, 100, 0, 0], list("AAAA")),                            # one wording only: no other wording
    ([100, 0, 0, 100, 100, 0, 100, 0, 0], list("AAABBBCCC")),    # thirds: 1, 2 and 1 of 3
])
def test_the_bonus_is_exactly_zero_when_every_pass_has_the_same_share(script, rewards, wordings):
    group = script.family_group(rewards, [reward > 0 for reward in rewards], wordings, 20)
    if wordings == list("AAABBBCCC"):
        # A and C pass 1 of 3 (q = 3/6), B passes 2 of 3 (q = 2/6): not equal, the bonus is not zero.
        assert max(abs(value) for value in group["bonus"]) > 1.0
        assert sum(group["bonus"]) == pytest.approx(0.0, abs=1e-12)
        return
    assert group["bonus"] == [0.0] * len(rewards)                # exactly, not only nearly
    assert group["arm1"] == group["arm2"]
    assert group["kept_arm1"] == group["kept_arm2"]


def test_a_group_with_equal_rewards_is_dropped_in_both_arms_and_a_lost_attempt_is_left_out(script):
    dropped = script.family_group([100] * 6, [True] * 6, [1, 1, 2, 2, 3, 3], 20)
    assert not dropped["kept_arm1"] and not dropped["kept_arm2"]
    assert dropped["arm1"] == [0.0] * 6 and dropped["arm2"] == [0.0] * 6
    # The harness lost the second attempt (None). Wording 1 then has one valid attempt, a pass.
    #   q of the pass in wording 1: wording 2 holds 1 pass in 2 = 0.5.
    #   q of the pass in wording 2: wording 1 holds 1 pass in 1 valid attempt = 1.0.
    #   q_mean = 0.75; bonus: 20 * (0.5 - 0.75) = -5 and 20 * (1.0 - 0.75) = +5.
    lost = script.family_group([100, None, 100, 0], [True, False, True, False], [1, 1, 2, 2], 20)
    assert lost["valid"] == [True, False, True, True]
    assert lost["bonus"] == pytest.approx([-5.0, 0.0, 5.0, 0.0])
    assert lost["arm1"][1] == 0.0 and lost["arm2"][1] == 0.0     # the lost attempt has no advantage
    # Arm 1: the mean of the valid rewards is 200 / 3.
    assert lost["arm1"] == pytest.approx([100 - 200 / 3, 0.0, 100 - 200 / 3, -200 / 3])


@pytest.mark.parametrize("counts, expected", [
    ({1: [5, 8], 2: [3, 8]}, True),          # 5 of 8 and 3 of 8: the two limits themselves
    ({1: [5, 8], 2: [4, 8]}, False),         # nobody at 3 or fewer
    ({1: [4, 8], 2: [3, 8]}, False),         # nobody at 5 or more
    ({1: [8, 8], 2: [8, 8], 3: [7, 8], 4: [1, 8]}, True),
    ({1: [3, 8], 2: [0, 8], 3: [0, 8], 4: [0, 8]}, False),     # every style fails: they agree
    ({1: [8, 8]}, False),                    # one style cannot disagree with itself
    ({1: [10, 16], 2: [6, 16]}, True),       # the rule is a share: 10 of 16 = 5 of 8, 6 of 16 = 3 of 8
    ({1: [9, 16], 2: [6, 16]}, False),
])
def test_the_rule_for_wordings_that_disagree(script, counts, expected):
    assert script.styles_disagree(counts) is expected


def test_how_many_families_must_gain_for_48_families(script):
    """(1 - k/48)^48: 36.4% for k = 1, 13.0% for 2, 4.5% for 3, 1.5% for 4. The lower end of a 95% interval
    is the 2.5% quantile, so 4 families is the first number below it."""
    needed = script.families_that_must_gain(48, 5, n_draws=10_000, seed=0, level=0.95)
    shares = {line["families_that_gain"]: line["share_of_draws_without_them"] for line in needed["lines"]}
    assert shares[1] == pytest.approx((47 / 48) ** 48) and round(100 * shares[1], 1) == 36.4
    assert round(100 * shares[3], 1) == 4.5 and round(100 * shares[4], 1) == 1.5
    assert needed["families_needed"] == 4 and needed["tail"] == pytest.approx(0.025)
    # The frozen bootstrap function says the same for every line.
    for line in needed["lines"]:
        assert (line["lower_end_above_zero_by_the_formula"]
                == line["lower_end_above_zero_by_the_frozen_bootstrap"])
    assert [line["lower_end_above_zero_by_the_formula"] for line in needed["lines"][:5]] == [
        False, False, False, True, True]


def test_the_chance_that_every_run_of_one_arm_is_above_every_run_of_the_other(script):
    """2 runs per arm: 4 runs, C(4, 2) = 6 ways to pick the two best. 3 per arm: C(6, 3) = 20."""
    rule = script.rank_rule(3)
    assert rule["2"] == {"runs": 4, "ways": 6, "chance": pytest.approx(1 / 6)}
    assert rule["3"] == {"runs": 6, "ways": 20, "chance": pytest.approx(1 / 20)}
    assert math.comb(6, 3) == 20


def test_number_formats(script):
    assert script.pct(0.39779) == "39.8%" and script.pts(0.0361) == "3.6" and script.pts(0.03552) == "3.6"
    assert script.signed(12.5) == "+12.5" and script.signed(-3) == "-3.0" and script.signed(-0.04) == "0.0"
    assert script.signed_short(25.0) == "+25" and script.signed_short(48.333) == "+48.3"
    assert script.signed_short(-90.0) == "-90" and script.signed_short(0.0) == "0"
    assert script.plain(20.0) == "20" and script.plain(5.625) == "5.6" and script.plain(1e-05, 6) == "0.00001"
    assert script.listed(["a"]) == "a" and script.listed(["a", "b", "c"]) == "a, b and c"
    assert script.families_word(1) == "1 family" and script.families_word(0) == "0 families"
    assert script.count_word(2) == "two" and script.count_word(7) == "7"
    assert script.same_or_listed(["3.6", "3.6"], ["x", "y"]) == "3.6 in each GRPO arm"
    assert script.same_or_listed(["3.6", "3.6"], ["x", "y"], short=True) == "3.6"
    assert script.same_or_listed(["3.6", "4.0"], ["x", "y"]) == "3.6 (x) and 4.0 (y)"
    assert script.interval({"diff": 0.162, "lo": 0.081, "hi": 0.254}) == "+16.2 (+8.1 to +25.4)"


# ---------------------------------------------------------------------------
# The script on the made-up saved runs: every number by hand
# ---------------------------------------------------------------------------
def test_the_files_say_in_their_first_lines_that_the_experiment_was_not_run(made):
    first_lines = "\n".join(made["doc"].splitlines()[:3])
    assert "**The experiment was not run.**" in first_lines
    assert "**The stretch experiment was not run.**" in "\n".join(made["tables"].splitlines()[:3])
    assert made["numbers"]["experiment_was_run"] is False


def test_dev_numbers_per_arm_by_hand(made):
    """Six families x 4 styles x 8 attempts = 192 attempts per arm.

    Passed attempts: base 32 + 16 + 0 + 16 + 16 + 18 = 98; sft 32 + 32 + 16 + 30 + 24 + 32 = 166;
    grpo_a 32 + 32 + 32 + 31 + 17 + 32 = 176; grpo_b 32 + 32 + 32 + 32 + 3 + 31 = 162. Every family has 32
    attempts, so the family-weighted pass rate is the plain share.

    All wordings pass (the product of the four style rates per family, then the mean over 6 families):
        base    1, (6/8)(2/8)(4/8)(4/8) = 0.046875, 0, (4/8)^4 = 0.0625, 0, (5/8)^3 (3/8) = 0.091552734375
        sft     1, 1, 0.0625, 6/8, (7/8)(1/8) = 0.109375, 1                         -> 3.921875 / 6
        grpo_a  1, 1, 1, 7/8, 0, 1                                                  -> 4.875 / 6 = 0.8125
        grpo_b  1, 1, 1, 1, 0, 7/8                                                  -> 4.875 / 6 = 0.8125
    """
    dev = made["numbers"]["dev"]
    assert (dev["families"], dev["scenarios"], dev["attempts_per_arm"]) == (6, 24, 192)
    assert dev["attempts_per_scenario"] == [8] and dev["styles"] == [1, 2, 3, 4]
    assert dev["styles_in_no_training_scenario"] == [4]
    assert dev["points_of_one_family"] == pytest.approx(1 / 6)
    passed = {"base": 98, "sft": 166, "grpo_a": 176, "grpo_b": 162}
    all_wordings = {"base": (1 + 0.046875 + 0 + 0.0625 + 0 + 0.091552734375) / 6, "sft": 3.921875 / 6,
                    "grpo_a": 0.8125, "grpo_b": 0.8125}
    with_a_fail = {"base": 5, "sft": 3, "grpo_a": 2, "grpo_b": 2}
    for arm in ARMS:
        row = dev["arms"][arm]
        assert (row["attempts"], row["valid_attempts"], row["harness_errors"]) == (192, 192, 0)
        assert (row["passed_attempts"], row["failed_attempts"]) == (passed[arm], 192 - passed[arm])
        assert row["pass_rate"] == pytest.approx(passed[arm] / 192)
        assert row["all_wordings_pass"] == pytest.approx(all_wordings[arm])
        assert row["room"] == pytest.approx(1 - all_wordings[arm])
        assert row["families_with_a_failed_attempt"] == with_a_fail[arm]
        assert row["families_below_one_on_all_wordings"] == with_a_fail[arm]
        assert row["all_wordings_pass_lo"] <= row["all_wordings_pass"] <= row["all_wordings_pass_hi"]
    # All wordings pass over the training styles 1 to 3 only (the frozen function on those attempts):
    #   base  1, (6/8)(2/8)(4/8) = 0.09375, 0, (1/2)^3 = 0.125, 0, (5/8)^3 = 0.244140625  -> 1.462890625 / 6
    #   sft   1, 1, 0.125, 1, 7/8, 1 -> 5 / 6;  grpo_a 1, 1, 1, 1, 1/8, 1 -> 5.125 / 6;  grpo_b 4.875 / 6
    assert dev["training_styles"] == [1, 2, 3]
    trained = {"base": 1.462890625 / 6, "sft": 5 / 6, "grpo_a": 5.125 / 6, "grpo_b": 4.875 / 6}
    for arm in ARMS:
        assert dev["arms"][arm]["all_wordings_pass_training_styles"] == pytest.approx(trained[arm])
    # The room of grpo_a: dv-04 holds (1 - 0) / 6 and dv-03 holds (1 - 7/8) / 6; the largest part is first.
    assert list(dev["arms"]["grpo_a"]["room_by_family"].items()) == [
        ("dv-04", pytest.approx(1 / 6)), ("dv-03", pytest.approx(0.125 / 6))]
    assert list(dev["arms"]["grpo_b"]["room_by_family"]) == ["dv-04", "dv-05"]


def test_the_metrics_are_the_frozen_functions(made, script):
    """The same numbers from the frozen functions, called here on the same made-up records."""
    for arm in ARMS:
        records = rollout_records(DEV[arm], arm, f"{arm}_dev")
        row = made["numbers"]["dev"]["arms"][arm]
        assert row["pass_rate"] == pytest.approx(metrics.pass_rate(records))
        assert row["all_wordings_pass"] == pytest.approx(metrics.all_wordings_pass(records))
        assert row["variance_shares"] == pytest.approx(metrics.variance_shares(records))
        assert row["variance_shares"]["wording_if_sampling_only"] == pytest.approx(3 / 31)
    # No metric of the plan is defined a second time in the script.
    for name in ("pass_rate", "all_wordings_pass", "variance_shares", "family_pass_rates", "pass_rate_by",
                 "group_advantages", "paired_family_bootstrap", "family_bootstrap_ci", "two_halves_mde",
                 "zero_variance_share", "pass_at_k", "pass_hat_k", "reward", "verify"):
        assert not hasattr(script, name), f"the script defines {name} itself"
    assert script.metrics is metrics and script.tables is tables and script.impl is impl


def test_a_changed_frozen_function_changes_the_numbers(script, tmp_path, monkeypatch):
    """The numbers come through the frozen modules: a stand-in for one of their functions reaches the
    output."""
    paths = build_tree(tmp_path)
    real = metrics.variance_shares
    monkeypatch.setattr(metrics, "variance_shares", lambda records: {**real(records), "wording": 0.123456})
    numbers = run(script, paths)
    assert all(numbers["dev"]["arms"][arm]["variance_shares"]["wording"] == 0.123456 for arm in ARMS)
    calls = []
    real_advantages = impl.main_version("group_advantages")

    def counted(rewards, tie_tol=0.0):
        calls.append(np.asarray(rewards).shape)
        return real_advantages(rewards, tie_tol)

    monkeypatch.setattr(impl, "get", lambda name: counted if name == "group_advantages" else
                        impl.main_version(name))
    run(script, paths)
    # 2 calls for each of the 6 training steps (with and without the bonus), and the family groups.
    assert calls.count((2, 4)) == 12 and (1, 12) in calls and (1, 16) in calls


def test_families_whose_wordings_disagree_and_the_family_that_got_worse(made):
    """Rule: one style at 5 or more of 8 and another at 3 or fewer.
        base    dv-01 (6 and 2), dv-04 (8 and 0), dv-05 (5 and 3)
        sft     dv-04 (8 and 1)          grpo_a  dv-04 (8 and 0)         grpo_b  none (dv-04: 3, 0, 0, 0)
    Fewer passes than under sft: grpo_a dv-04 (17 < 24); grpo_b dv-04 (3 < 24) and dv-05 (31 < 32)."""
    dev = made["numbers"]["dev"]
    found = {arm: dev["arms"][arm]["families_whose_wordings_disagree"] for arm in ARMS}
    assert found == {"base": ["dv-01", "dv-04", "dv-05"], "sft": ["dv-04"], "grpo_a": ["dv-04"], "grpo_b": []}
    assert dev["families_with_fewer_passes_than_sft"] == {"grpo_a": ["dv-04"], "grpo_b": ["dv-04", "dv-05"]}
    assert dev["families_with_fewer_passes_than_sft_in_every_grpo_arm"] == ["dv-04"]
    rows = dev["family_rows"]["dv-04"]["arms"]
    assert rows["sft"]["passes_by_style"] == {"1": [8, 8], "2": [8, 8], "3": [7, 8], "4": [1, 8]}
    assert rows["grpo_a"]["passes_by_style"] == {"1": [8, 8], "2": [8, 8], "3": [1, 8], "4": [0, 8]}
    assert rows["grpo_b"]["passes_by_style"] == {"1": [3, 8], "2": [0, 8], "3": [0, 8], "4": [0, 8]}
    assert [rows[arm]["passed_attempts"] for arm in ARMS] == [16, 24, 17, 3]
    # Every failed attempt closes the request as denied: 15 in grpo_a, 29 in grpo_b. The 8 failed attempts
    # of style 4 name another person: 7 of 15 name the person of the request in grpo_a, 21 of 29 in grpo_b.
    assert rows["grpo_a"]["closes_of_the_failed_attempts"] == {"denied": 15}
    assert rows["grpo_b"]["closes_of_the_failed_attempts"] == {"denied": 29}
    assert rows["grpo_a"]["person_of_the_close_in_failed_attempts"] == {"USRright": 7, "USRother": 8}
    assert rows["grpo_b"]["person_of_the_close_in_failed_attempts"] == {"USRright": 21, "USRother": 8}
    assert made["numbers"]["family_facts"]["dv-04"] == {
        "outcome": "grant", "expected_close": "granted", "person_of_the_request": "USRright",
        "people_with_the_name_of_the_request": 2, "status_of_the_other_people_with_that_name": ["inactive"]}
    # The variance split without that family is the frozen function on the other five families.
    for arm in ARMS:
        rest = [record for record in rollout_records(DEV[arm], arm, f"{arm}_dev")
                if record["family_id"] != "dv-04"]
        without = dev["without_those_families"][arm]
        assert without["variance_shares"] == pytest.approx(metrics.variance_shares(rest))
        assert without["valid_attempts"] == 160
    assert dev["without_those_families"]["grpo_a"]["failed_attempts"] == 1       # dv-03, style 4
    assert dev["without_those_families"]["grpo_b"]["failed_attempts"] == 1       # dv-05, style 2


def test_the_saved_training_groups_by_hand(made):
    """grpo_a: 6 groups of 4 attempts, 24 attempts.
        kept: (100, 100, 0, 0), (100, 95, 100, 100), (100, 95, 0, 0), (100, 0, 0, 0) = 4
        with a pass and a fail, every pass at 100: 2; every attempt passes: 1; a pass below 100: 1
        the two wordings of tr-002 are in one step: 2 groups have another wording of their family
    grpo_b: kept (100, 0, 100, 0), (95, 100, 100, 88), (100, 100, 100, 0) = 3."""
    runs = made["numbers"]["one_wording_group"]["saved_runs"]
    a, b = runs["grpo_a"], runs["grpo_b"]
    assert (a["steps"], a["scenarios_per_step"], a["attempts_per_group"]) == (3, 2, 4)
    assert a["attempts_per_step"] == 8
    assert (a["groups"], a["attempts"], a["harness_errors"]) == (6, 24, 0)
    assert (a["kept"], a["kept_with_bonus"]) == (4, 4)
    assert (a["keep_decision_changes"], a["kept_order_changes"]) == (0, 0)
    assert (a["kept_mixed_full_reward"], a["kept_one_outcome"], a["kept_mixed_other"]) == (2, 1, 1)
    assert a["with_a_pass_and_a_fail"] == 3 and a["with_another_wording_of_the_family_in_the_step"] == 2
    assert (b["groups"], b["kept"], b["kept_with_bonus"]) == (6, 3, 3)
    assert (b["kept_mixed_full_reward"], b["kept_one_outcome"], b["kept_mixed_other"]) == (2, 1, 0)
    assert b["with_a_pass_and_a_fail"] == 2 and b["with_another_wording_of_the_family_in_the_step"] == 0
    assert a["factor"] == pytest.approx(1.2) and a["bonus_points"] == 20
    assert (a["seconds"], b["seconds"]) == (3600.0, 7200.0)
    assert a["seconds_in_steps"] == pytest.approx(3600.0) and a["rollout_files"] == 3
    assert made["numbers"]["one_wording_group"]["saved_code_refuses_a_group_of_two_scenarios"] is True
    # Passing attempts: grpo_a 6 + 4 + 3 = 13, two of them at 95; grpo_b 6 + 4 + 7 = 17, at 95 and 88.
    assert (a["passes"], a["passes_below_full_reward"], a["lowest_pass_reward"]) == (13, 2, 95)
    assert (b["passes"], b["passes_below_full_reward"], b["lowest_pass_reward"]) == (17, 2, 88)
    assert a["fails_not_at_zero"] == 0 and b["fails_not_at_zero"] == 0


def test_the_design_on_the_saved_sft_attempts_by_hand(made):
    """Draw 1 (attempts 1 to 4): tr-100 and tr-102 have equal rewards; tr-101 is the example of the
    document (6 passes; bonus -5/3 for 4 of them and +10/3 for 2: mean size (4 * 5/3 + 2 * 10/3) / 6 =
    20/9, largest 10/3; arm-1 advantages all +50 or -50); tr-103 has 2 passes of 4 in every style, so its
    bonus is zero.
    Draw 2 (attempts 5 to 8): tr-100, tr-101 and tr-102 have equal rewards; tr-103 passes everywhere with
    one reward of 95, so it is kept and its bonus is zero."""
    first, second = made["numbers"]["design"]["on_saved_attempts"]
    assert (first["draw"], first["sample_positions"], first["groups"], first["attempts_per_group"]) == (
        1, [1, 4], 4, 12)
    assert (first["groups_with_equal_rewards"], first["groups_kept"]) == (2, 2)
    assert first["groups_with_a_pass_and_a_fail"] == 2
    assert first["groups_where_the_bonus_is_not_zero"] == 1
    assert first["groups_where_the_order_of_the_attempts_changes"] == 1
    assert first["groups_where_the_arms_keep_differently"] == 0
    assert first["passes_in_those_groups"] == 6
    assert first["mean_size_of_the_bonus_points"] == pytest.approx(20 / 9)
    assert first["largest_bonus_points"] == pytest.approx(10 / 3)
    assert first["mean_size_of_the_arm1_advantage_points"] == pytest.approx(50.0)
    assert (second["draw"], second["sample_positions"]) == (2, [5, 8])
    assert (second["groups_with_equal_rewards"], second["groups_kept"]) == (3, 1)
    assert second["groups_with_a_pass_and_a_fail"] == 0
    assert second["groups_where_the_bonus_is_not_zero"] == 0
    assert second["mean_size_of_the_bonus_points"] is None and second["largest_bonus_points"] is None
    assert first["families_left_out"] == 0


def test_the_numbers_of_the_design_carry_their_origin(made):
    design = made["numbers"]["design"]
    origins = {row["part"]: (row["value"], row["origin"]) for row in design["numbers"]}
    assert origins["attempts per wording"] == (4, "judgement")
    assert origins["family groups per step"] == (10, "judgement")
    assert origins["largest bonus of a pass, in points"] == (20, "judgement")
    assert origins["training runs (seeds) per arm"] == (3, "judgement")
    assert origins["arms"][0] == 2 and origins["arms"][1].startswith("the brief")
    assert origins["wordings in a family group"][0] == 3
    assert "saved runs" in origins["wordings in a family group"][1]
    assert origins["attempts per family group"] == (12, "own calculation")
    assert origins["attempts per step"] == (120, "own calculation")
    assert origins["steps"] == (3, "the saved runs") and origins["learning rate"] == (1e-05, "the saved runs")
    assert all(row["origin"] for row in design["numbers"])
    parts = {row["part"]: row["origin"] for row in design["parts"]}
    assert parts["reward of arm 1"] == "the saved runs" and parts["other settings"] == "the saved runs"
    assert parts["bonus when another wording has no valid attempt"] == "judgement"
    assert design["rank_rule"]["3"]["ways"] == 20 and design["rank_rule"]["2"]["ways"] == 6


def test_the_comparisons_are_the_frozen_bootstrap(made):
    """Pass rate: the pair row of the frozen table program. All wordings pass: the same frozen function on
    the family values (the products) worked out above."""
    comparisons = {(row["a"], row["b"]): row for row in made["numbers"]["dev"]["comparisons"]}
    assert list(comparisons) == [("base", "sft"), ("sft", "grpo_a"), ("sft", "grpo_b"), ("base", "grpo_a"),
                                 ("base", "grpo_b"), ("grpo_a", "grpo_b")]
    records = {arm: rollout_records(DEV[arm], arm, f"{arm}_dev") for arm in ARMS}
    row = comparisons[("sft", "grpo_a")]
    frozen = tables.pair_row("sft", "grpo_a", records["sft"], records["grpo_a"], 2000, 0, 0.95)
    assert row["pass_rate"]["diff"] == pytest.approx(frozen["diff"])
    assert row["pass_rate"]["diff"] == pytest.approx((176 - 166) / 192)
    assert (row["pass_rate"]["lo"], row["pass_rate"]["hi"]) == pytest.approx((frozen["lo"], frozen["hi"]))
    assert row["pass_rate"]["width"] == pytest.approx(frozen["hi"] - frozen["lo"])
    sft = np.array([1, 1, 0.0625, 0.75, 0.109375, 1])
    grpo_a = np.array([1, 1, 1, 0.875, 0, 1])
    on_values = impl.main_version("paired_family_bootstrap")(sft, grpo_a, 2000, 0, 0.95)
    assert row["all_wordings_pass"]["diff"] == pytest.approx(0.8125 - 3.921875 / 6)
    assert (row["all_wordings_pass"]["lo"], row["all_wordings_pass"]["hi"]) == pytest.approx(
        (on_values["lo"], on_values["hi"]))
    assert row["all_wordings_pass"]["families_that_differ"] == 3          # dv-02, dv-03 and dv-04
    assert row["in_the_plan"] is True and comparisons[("grpo_a", "grpo_b")]["in_the_plan"] is False
    # grpo_b minus grpo_a on all wordings pass: dv-03 +1/8, dv-05 -1/8, so the difference is 0.
    assert comparisons[("grpo_a", "grpo_b")]["all_wordings_pass"]["diff"] == pytest.approx(0.0)
    assert comparisons[("grpo_a", "grpo_b")]["all_wordings_pass"]["families_that_differ"] == 2
    assert made["numbers"]["dev"]["interval"] == {"level": 0.95, "n_draws": 2000, "seed": 0,
                                                  "paired_family_bootstrap": "main"}


def test_what_can_be_said_about_an_effect_by_hand(made):
    """6 families: (1 - k/6)^6 is 33.5% for k = 1, 8.8% for k = 2 and 1.6% for k = 3, so 3 families must
    gain. The two GRPO arms differ by (176 - 162) / 192 in pass rate and by 0 in all wordings pass. The
    room is 18.75 points, which is not below the 5 points that the document calls little."""
    effect = made["numbers"]["effect"]
    needed = effect["families_that_must_gain"]
    assert (needed["families"], needed["families_needed"], needed["families_with_room"]) == (6, 3, 2)
    assert [round(100 * line["share_of_draws_without_them"], 1) for line in needed["lines"][:3]] == [
        33.5, 8.8, 1.6]
    assert effect["spread_between_the_grpo_arms"]["pass_rate"] == pytest.approx(14 / 192)
    assert effect["spread_between_the_grpo_arms"]["all_wordings_pass"] == pytest.approx(0.0)
    assert effect["largest_room_after_grpo"] == pytest.approx(0.1875)
    assert effect["little_room_is_below"] == 0.05 and effect["room_is_little"] is False


def test_the_cost_by_hand(made):
    """Training 1 and 2 GPU hours, each dev run 0.5: one run of the design is (1.5 + 2.5) / 2 = 2 hours.
    2 arms x 3 seeds = 6 runs = 12 hours; x 2 seeds = 4 runs = 8 hours. The ledger has two rent lines before
    the end of the later GRPO run (10:00), at 0.5 and 1.0 USD per hour (the destroy line is not a price):
    6 to 12 USD, and 4 to 8 USD."""
    cost = made["numbers"]["cost"]
    assert cost["saved"]["grpo_a"]["training_hours"] == pytest.approx(1.0)
    assert cost["saved"]["grpo_b"]["training_hours"] == pytest.approx(2.0)
    assert cost["saved"]["grpo_a"]["dev_run_hours"] == pytest.approx(0.5)
    assert cost["hours_of_one_design_run"] == pytest.approx(2.0)
    assert cost["designs"]["3"] == {"seeds_per_arm": 3, "runs": 6, "gpu_hours": pytest.approx(12.0),
                                    "usd_low": pytest.approx(6.0), "usd_high": pytest.approx(12.0)}
    assert cost["designs"]["2"] == {"seeds_per_arm": 2, "runs": 4, "gpu_hours": pytest.approx(8.0),
                                    "usd_low": pytest.approx(4.0), "usd_high": pytest.approx(8.0)}
    assert (cost["prices"]["rentals"], cost["prices"]["until_utc"], cost["prices"]["lowest_usd_per_hour"],
            cost["prices"]["highest_usd_per_hour"]) == (["1", "2"], "2026-01-01T10:00:00Z", 0.5, 1.0)
    # The ledger itself is not described: no count of its lines and no hash.
    assert "sha256" not in cost["prices"] and "rent_lines" not in cost["prices"]
    assert cost["gpus"] == ["TEST GPU"]


def test_a_machine_rented_after_the_last_grpo_run_changes_no_output(script, tmp_path):
    """A new open rental (and a closed one) after the end of the last saved GRPO run: the three files keep
    every byte. A rent line before that end changes the prices."""
    paths = build_tree(tmp_path)
    run(script, paths)
    before = files_of(paths["out"]) | {"doc": hashlib.sha256(paths["doc"].read_bytes()).hexdigest()}
    with paths["ledger"].open("a", encoding="utf-8") as handle:
        handle.write("2026-01-01T10:00:01Z,rent,3,TEST GPU,9.0,\n2026-01-02T00:00:00Z,rent,4,TEST GPU,0.1,\n"
                     "2026-01-02T01:00:00Z,destroy,4,TEST GPU,0.1,0.1\n")
    run(script, paths)
    assert files_of(paths["out"]) | {"doc": hashlib.sha256(paths["doc"].read_bytes()).hexdigest()} == before
    with paths["ledger"].open("a", encoding="utf-8") as handle:
        handle.write("2026-01-01T09:30:00Z,rent,5,TEST GPU,2.0,\n")
    numbers = run(script, paths)
    assert numbers["cost"]["prices"]["highest_usd_per_hour"] == 2.0
    assert numbers["cost"]["designs"]["3"]["usd_high"] == pytest.approx(24.0)


def test_without_the_end_of_a_grpo_run_no_price_is_given(script, tmp_path):
    paths = build_tree(tmp_path)
    manifest = paths["main"] / "grpo_b" / "run_manifest.json"
    found = json.loads(manifest.read_text(encoding="utf-8"))
    del found["ended_utc"]
    manifest.write_text(json.dumps(found), encoding="utf-8")
    numbers = run(script, paths)
    assert numbers["cost"]["prices"] is None and numbers["cost"]["designs"]["3"]["usd_low"] is None
    assert "No price: the ledger is missing, or a saved GRPO run has no `ended_utc`." in (
        paths["out"] / script.TABLES_NAME).read_text(encoding="utf-8")


def test_the_document_holds_the_numbers_of_the_made_up_runs(made):
    doc = made["doc"]
    for heading in ("## 1. What the stretch asks",
                    "## 2. What the bonus can do in groups of one wording",
                    "## 3. How much there is to gain", "## 4. The design that would test it",
                    "## 5. What size of effect it could see", "## 6. What the experiment would cost"):
        assert heading in doc
    assert [line for line in doc.splitlines() if line.startswith("## ")][0] == "## 1. What the stretch asks"
    assert "A GRPO *group* here is the 4 attempts of one scenario in one step." in doc
    assert ("12 passes, 4 fails, b = 20. A pass goes from +25 to +30 and a fail from -75 to -90: 1.2 "
            "times.") in doc
    # Passes score below 100 because of penalties; the general shift, not only the factor, is stated.
    assert "A saved pass scores 88 to 100 points: 100 minus small penalties" in doc
    assert "every pass gains b (1 - k/n) and every fail loses b k/n" in doc
    assert "If every pass scores 100, this multiplies each advantage by (100 + b)/100." in doc
    assert ("the saved training (T2) keeps the same groups (4 of 6 in grpo_a and 3 of 6 in grpo_b), and no kept "
            "group changes the order of its attempts. In 1 and 1 of the kept groups every attempt passes") in doc
    # What the bonus does in a group of one wording, said exactly; "only the weight" is exact only where every
    # pass scores 100. grpo_a: 2 kept pass-and-fail groups at 100 (tr-000/s1, tr-004/s1), 1 with a pass at
    # 95 (tr-003/s2). grpo_b: 2 at 100 (tr-010/s1, tr-015/s2), none with a pass below 100.
    assert ("So in a group of one wording the bonus cannot make the training prefer one passing attempt over "
            "another: it adds b to each pass-fail gap and leaves the gaps between passes as they are. Where "
            "every pass scores 100 (2 and 2 kept groups with passes and fails), that only changes the weight "
            "of the group's step, as b differs between groups; in 1 and 0 such groups a pass scores less "
            "(T2). For the bonus to differ between passes, a group must mix the wordings of one family.") in doc
    assert "cannot test" not in doc and "as the brief means it" not in doc
    assert "Dev: 6 families x 4 wording styles x 8 attempts = 192 attempts per arm" in doc
    assert "| All wordings pass, styles 1, 2 and 3 only |" in doc
    assert "| base | 51.0% (98 of 192) | 20.0% | 24.4% | 3 |" in doc
    assert "| grpo_a | 91.7% (176 of 192) | 81.2% | 85.4% | 1 |" in doc
    assert "| grpo_b | 84.4% (162 of 192) | 81.2% | 81.2% | 0 |" in doc
    assert "A *held-out* style is a wording that no training scenario uses: here style 4." in doc
    assert ("In every GRPO arm dv-04 passes 0 of 8 in its held-out style 4, so its product stays 0: a bonus "
            "on the training styles reaches these points only through a gain on a held-out wording.") in doc
    assert "one style passes 5 or more of its 8 attempts and another style 3 or fewer" in doc
    assert ("**Room after GRPO** (100% minus all wordings pass): 18.8 points in each GRPO arm, in 2 of 6 "
            "families. One family, dv-04, holds 16.7 of these points.") in doc
    assert "moved the measure" not in doc                  # the table above gives these numbers
    assert "Correctness alone" not in doc
    assert "after GRPO, 16 and 30 of 192 attempts fail. Without dv-04 it is" in doc
    assert ("After training only **dv-04** meets the rule (in sft and grpo_a; the 3 families of base are in "
            "T5). It is also the only family with fewer passes than under sft in every GRPO arm (T6):") in doc
    assert "| dv-04: passes of 8 | Style 1 | Style 2 | Style 3 | Style 4 (held out) |" in doc
    assert "| grpo_a | 8 | 8 | 1 | 0 |" in doc and "| grpo_b | 3 | 0 | 0 | 0 |" in doc
    assert ("grpo_a fails in styles 3 and 4. grpo_b fails in all 4 styles (3, 0, 0 and 0 of 8): by the rule of "
            "this note its wordings do not disagree.") in doc
    assert ("Every failed attempt on dv-04 in these arms closes the request as denied (15 of 15 and 29 of "
            "29); the correct close is granted.") in doc
    assert "Agreement must be read beside the pass rate." in doc and "the wordings agree" not in doc
    assert "The test set is not in this file" in doc
    assert "10 families = 120 attempts (saved runs: 2 x 4 = 8); 3 steps as saved" in doc
    assert "| Arm 1 | the saved reward (pass or fail, with the penalties of part 2) | the saved runs |" in doc
    assert "| Other settings | as in `grpo_config` of `" in doc and "tie_tol 0, start `made-up/sft/adapter`" in doc
    assert "(a pass with no such attempt gets no bonus)" in doc
    assert "all wordings pass (over all styles, and over the training styles only)" in doc
    assert "Arm 2: a pass in A +48.3 and a pass in B +53.3" in doc
    assert "By chance alone that happens in 1 of 20 cases (own calculation)." in doc
    assert "33.5% for k = 1, 8.8% for k = 2 and 1.6% for k = 3" in doc
    # The room is a ceiling only if arm 1 ends where the saved runs did; the seeds differ in 2 families.
    assert ("only if at least 3 families gain. If arm 1 ends where the saved runs did, only 2 families have "
            "room, fewer than 3: the interval can be above zero only "
            "if other families gain too. Yet two runs of one recipe already differ in 2 of 6 families on all "
            "wordings pass.") in doc
    assert "the bonus can add no more" not in doc
    assert "This is no power calculation: two saved seeds do not show the spread between runs (T10, T11)." in doc
    # Section 6: what the experiment would cost, in GPU hours of run time, with the calculation.
    # The sums are in seconds, as the run manifests save them, so the printed parts add up to the printed
    # total (test_every_printed_sum_of_the_document_adds_up checks every sum).
    assert ("In run time (`seconds` of the run manifests), not rented time (T12), one design run counts as a "
            "saved GRPO run plus its dev run: grpo_a 3600.0 + 1800.0 = 5400.0 s and grpo_b 7200.0 + 1800.0 = "
            "9000.0 s; mean 7200.0 s = 2 hours. 2 arms x 3 seeds = 6 runs: 6 x 2 = about 12 GPU hours.") in doc
    assert ("At 0.5 to 1 USD per GPU hour (rentals up to the end of the last saved GRPO run): 6 to 12 "
            "USD.") in doc
    # No reason for not running it: only a fact that a reader can check in the frozen analysis plan.
    section = doc.split("## 6. ")[1]
    assert section.startswith("What the experiment would cost\n\nThe experiment was not run. The frozen "
                              "`ANALYSIS_PLAN.md` does not list it among its arms or its planned extras "
                              "(section 3).\n")
    assert "no saved file records" not in doc
    assert "required items" not in doc and "time of the take-home" not in doc
    assert "Beyond the cost" not in doc and "why it was not run" not in doc
    for reason in ("because of the cost", "too expensive", "no time", "little room", "leave this room"):
        assert reason not in section, reason
    # The missing training code is a fact of the design (section 4), not a reason in section 6.
    assert ("The saved code builds each group from one scenario: the design needs new training "
            "code.") in doc.split("## 4. ")[1].split("## 5. ")[0]
    assert "training code" not in section
    assert str(made["paths"]["out"] / "stretch_numbers.md") in doc     # where the sources are


# ---------------------------------------------------------------------------
# Every printed sum and product of the document, checked by arithmetic
# ---------------------------------------------------------------------------
_NUMBER = r"\d+(?:\.\d+)?"
_WORDS = r"(?:\s+[A-Za-z][A-Za-z ]*?)?"
_CALCULATION = re.compile(rf"(?<![\w.])({_NUMBER}{_WORDS}(?:\s+[x+]\s+{_NUMBER}{_WORDS})+)\s+=\s+(about\s+)?"
                          rf"({_NUMBER})")


def printed_calculations(text: str) -> list[tuple[str, list[Decimal], bool, Decimal]]:
    """Every "a + b = c" and "a x b = c" of a text (words after a number are allowed: "2 arms x 3 seeds =
    6"): (operator, the parts, whether "about" stands before the result, the result)."""
    found = []
    for match in _CALCULATION.finditer(text):
        left, about, result = match.group(1), bool(match.group(2)), Decimal(match.group(3))
        operators = set(re.findall(r"\s([x+])\s", left))
        assert len(operators) == 1, f"a calculation with two operators: {match.group(0)!r}"
        parts = [Decimal(number) for number in re.findall(_NUMBER, left)]
        found.append((operators.pop(), parts, about, result))
    return found


def assert_sums_add_up(text: str) -> int:
    """A sum must add up as printed: the printed parts add up to the printed total, exactly. A product
    without "about" must be exact; with "about" it may differ from the result by half a unit of the
    result's last digit. Returns the number of calculations checked."""
    calculations = printed_calculations(text)
    for operator, parts, about, result in calculations:
        value = sum(parts) if operator == "+" else math.prod(parts)
        shown = f"{(' ' + operator + ' ').join(map(str, parts))} = {'about ' if about else ''}{result}"
        if operator == "+":
            assert not about and value == result, f"the printed sum does not add up: {shown}"
        elif about:
            half_unit = Decimal(1).scaleb(result.as_tuple().exponent) / 2
            assert abs(value - result) <= half_unit, f"the printed product is wrong: {shown}"
        else:
            assert value == result, f"the printed product is wrong: {shown}"
    return len(calculations)


def test_the_check_of_the_printed_sums_finds_a_wrong_sum():
    assert assert_sums_add_up("one 2.47 + 0.32 = 2.79 hours, and 2 arms x 3 seeds = 6 runs") == 2
    with pytest.raises(AssertionError, match="does not add up"):
        assert_sums_add_up("grpo_seed2 2.62 + 0.34 = 2.95 hours")
    with pytest.raises(AssertionError, match="product is wrong"):
        assert_sums_add_up("6 x 2.87 = about 17.3 GPU hours")


def test_every_printed_sum_of_the_document_adds_up(made, script, tmp_path):
    """The made-up document, without and with the test set: every sum and product, by arithmetic."""
    doc = made["doc"]
    calculations = printed_calculations(doc)
    assert ("+", [Decimal("3600.0"), Decimal("1800.0")], False, Decimal("5400.0")) in calculations
    assert ("x", [Decimal(6), Decimal(4), Decimal(8)], False, Decimal(192)) in calculations
    assert ("x", [Decimal(6), Decimal(2)], True, Decimal(12)) in calculations
    assert assert_sums_add_up(doc) >= 7
    paths = build_tree(tmp_path, with_test=True)
    run(script, paths)
    assert assert_sums_add_up(paths["doc"].read_text(encoding="utf-8")) >= 7


def test_seconds_that_do_not_add_up_when_rounded_to_hours_add_up_as_printed(script, tmp_path):
    """The saved seconds of grpo_seed2: 9427.7 + 1208.3. In hours with two decimals they print as
    2.62 + 0.34, but their sum is 2.95: the document prints the seconds, which add up exactly."""
    paths = build_tree(tmp_path)
    for folder, seconds in (("grpo_b", 9427.7), ("grpo_b_dev", 1208.3)):
        manifest_path = paths["main"] / folder / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["seconds"] = seconds
        write_json(manifest_path, manifest)
    run(script, paths)
    doc = paths["doc"].read_text(encoding="utf-8")
    assert "grpo_b 9427.7 + 1208.3 = 10636.0 s" in doc
    assert assert_sums_add_up(doc) >= 7


def test_with_the_test_set_the_document_grows_by_one_sentence_only(made, script, tmp_path):
    """With the test runs, only the sentence about the test set changes, and the growth stays inside the
    room that the real document keeps for it."""
    paths = build_tree(tmp_path, with_test=True)
    run(script, paths)
    with_test = paths["doc"].read_text(encoding="utf-8").replace(str(tmp_path), "ROOT")
    without = made["doc"].replace(str(made["root"]), "ROOT")
    placeholder = "The test set is not in this file: its runs were not there (T13).\n"
    assert placeholder in without and placeholder not in with_test
    sentence = next(line for line in with_test.splitlines() if line.startswith("**Test set**"))
    assert with_test == without.replace(placeholder, sentence + "\n")
    assert 0 < len(with_test) - len(without) <= MOST_GROWTH_WITH_THE_TEST_SET


def test_the_widest_test_set_sentence_fits_the_room_kept_for_it(made, script):
    """The test-set sentence at its widest: 48 families, every share 100.0%, every room 100.0 points in 48
    families, under the names of the real GRPO arms. It is at most MOST_GROWTH_WITH_THE_TEST_SET characters
    longer than the sentence that it replaces."""
    numbers = json.loads(json.dumps(made["numbers"]))
    names = {"grpo_a": "grpo_seed1", "grpo_b": "grpo_seed2"}
    without = script.render_doc(numbers, "T")
    widest = {"families": 48, "arms": {arm: {"all_wordings_pass": 1.0, "room": 1.0,
                                            "families_below_one_on_all_wordings": 48} for arm in ARMS}}
    numbers["test"] = widest
    with_test = script.render_doc(numbers, "T")
    placeholder = "The test set is not in this file: its runs were not there (T13)."
    sentence = next(line for line in with_test.splitlines() if line.startswith("**Test set**"))
    assert with_test == without.replace(placeholder, sentence)
    assert "100.0% (base)" in sentence and "100.0 points in 48 families (grpo_b)" in sentence
    for short, real in names.items():
        sentence = sentence.replace(f"({short})", f"({real})")
    assert "(grpo_seed2)" in sentence and "grpo_a" not in sentence
    growth = len(sentence) - len(placeholder)
    assert 0 < growth <= MOST_GROWTH_WITH_THE_TEST_SET, growth


def test_a_plan_that_lists_the_experiment_or_no_plan_leaves_the_plan_sentence_out(script, tmp_path):
    paths = build_tree(tmp_path)
    plan = tmp_path / "ANALYSIS_PLAN.md"
    sentence = "The frozen `ANALYSIS_PLAN.md` does not list it among its arms or its planned extras (section 3)."
    plan.write_text("# Plan\n\n## 3. Arms\n\n`base`.\n\nPlanned extras (ours): a teacher arm.\n\n## 4. "
                    "Comparisons\n\nA bonus for the same correct action.\n", encoding="utf-8")
    numbers = run(script, paths, "--plan", str(plan))
    assert numbers["analysis_plan"]["lists_the_experiment"] is False      # "bonus" is in section 4 only
    assert sentence in paths["doc"].read_text(encoding="utf-8")
    plan.write_text("# Plan\n\n## 3. Arms\n\nPlanned extras (ours): a bonus when rewordings agree.\n",
                    encoding="utf-8")
    numbers = run(script, paths, "--plan", str(plan))
    assert numbers["analysis_plan"]["words_found"] == ["bonus", "rewording"]
    doc = paths["doc"].read_text(encoding="utf-8")
    assert sentence not in doc and "## 6. What the experiment would cost\n\nThe experiment was not run.\n" in doc
    numbers = run(script, paths, "--plan", str(tmp_path / "no_such_plan.md"))
    assert numbers["analysis_plan"] is None
    assert sentence not in paths["doc"].read_text(encoding="utf-8")


def test_the_real_analysis_plan_does_not_list_the_experiment(script):
    """The sentence of section 6, checked against the frozen file: section 3 names the arms, the reference
    rows, the second desk and two planned extras (a teacher arm and a hint check), and no stretch."""
    found = script.plan_lists_the_experiment(ROOT / "ANALYSIS_PLAN.md")
    assert found is not None and found["has_planned_extras"] and found["lists_the_experiment"] is False
    text = (ROOT / "ANALYSIS_PLAN.md").read_text(encoding="utf-8")
    section = text.split("\n## 3. ")[1].split("\n## ")[0]
    assert "Planned extras" in section and "a teacher arm" in section and "a hint check" in section


def test_the_table_file_states_rule_unit_and_source_for_every_table(made):
    text = made["tables"]
    titles = [line for line in text.splitlines() if line.startswith("## T")]
    assert [title.split(".")[0] for title in titles] == [f"## T{n}" for n in range(1, 14)]
    for number in (1, 2, 4, 5, 6, 7, 9, 10, 11, 12):
        section = text.split(f"## T{number}. ")[1].split("\n## ")[0]
        assert "Rule" in section, f"T{number} states no rule"
    assert "Source:" in text.split("## T3. ")[1].split("\n## ")[0]
    full = "8 / 8 / 8 / 8"
    assert f"| dv-01 | approval | 6 / 2 / 4 / 4 | {full} | {full} | {full} | base |" in text
    assert ("| dv-04 | grant | 8 / 8 / 0 / 0 | 8 / 8 / 7 / 1 | 8 / 8 / 1 / 0 | 3 / 0 / 0 / 0 | base, sft, "
            "grpo_a |") in text
    assert "Count per arm: base 3; sft 1; grpo_a 1; grpo_b 0 (of 6 families)." in text
    assert "- grpo_b: dv-04 and dv-05." in text and "- In every GRPO arm: dv-04." in text
    assert ("- grpo_a, failed attempts of dv-04: 15 of 15 close the request as denied. 7 of 15 name the "
            "person of the request in the close.") in text
    assert ("2 people of its world have the name of the request; the status of the other one is "
            "inactive.") in text
    assert "| Kept by the rule of the training (the rewards are not all equal) | 4 | 3 |" in text
    assert "| Groups with another wording of their family in the same step | 2 | 0 |" in text
    assert "| 2 arms x 3 seeds | 6 | 12 | 6 to 12 |" in text
    assert "The test set is not in this file" in text.split("## T13. ")[1]
    assert "its reward matrix refuses a group that holds two scenarios" in text


def test_the_script_writes_three_files_and_changes_no_saved_run(made, script):
    new = sorted(set(made["after"]) - set(made["before"]))
    assert new == ["results/main/report/stretch/stretch_numbers.json",
                   "results/main/report/stretch/stretch_numbers.md", "results/stretch.md"]
    assert {path: made["after"][path] for path in made["before"]} == made["before"]      # nothing changed
    assert made["numbers"]["outputs"] == {
        "numbers": str(made["paths"]["out"] / "stretch_numbers.json"),
        "tables": str(made["paths"]["out"] / "stretch_numbers.md"), "document": str(made["paths"]["doc"])}
    assert made["numbers"]["inputs"]["test_runs_not_there"] == [
        str(made["paths"]["main"] / f"{arm}_test") for arm in ARMS]
    assert made["numbers"]["test"] is None


def test_a_second_run_writes_the_same_bytes(made, script):
    assert script.main(arguments(made["paths"])) == 0
    assert files_of(made["root"]) == made["after"]


def test_only_the_dev_scenario_file_is_opened_and_torch_is_not_imported(script, tmp_path, monkeypatch):
    paths = build_tree(tmp_path)
    opened = []
    real_open = Path.open

    def recording_open(self, *args, **kwargs):
        opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    run(script, paths)
    in_scenarios = {path.name for path in opened if paths["scenarios"] in path.parents}
    assert in_scenarios == {"dev.jsonl"}
    assert not [path for path in opened if path.name.startswith("tokens")]
    assert "test.jsonl" not in (ROOT / "scripts" / "stretch_numbers.py").read_text(encoding="utf-8")
    assert "torch" not in sys.modules


# ---------------------------------------------------------------------------
# Other shapes of the saved runs
# ---------------------------------------------------------------------------
def test_with_the_test_runs_the_files_hold_the_test_set_too(script, tmp_path):
    """Test set: 4 families x 4 styles x 8 attempts = 128 attempts per arm. All wordings pass:
    base (0.5^4 + 0 + 1 + 0.25^4) / 4 = 0.26660; sft (1 + 0.0625 + 1 + 1) / 4 = 0.765625;
    grpo_a (1 + 4/8 + 1 + 1) / 4 = 0.875; grpo_b 1."""
    paths = build_tree(tmp_path, with_test=True)
    numbers = run(script, paths)
    test = numbers["test"]
    assert (test["split"], test["families"], test["attempts_per_arm"]) == ("test", 4, 128)
    assert test["arms"]["base"]["all_wordings_pass"] == pytest.approx((0.0625 + 0 + 1 + 0.00390625) / 4)
    assert test["arms"]["sft"]["all_wordings_pass"] == pytest.approx(0.765625)
    assert test["arms"]["grpo_a"]["all_wordings_pass"] == pytest.approx(0.875)
    assert test["arms"]["grpo_b"]["all_wordings_pass"] == pytest.approx(1.0)
    assert test["arms"]["grpo_a"]["families_below_one_on_all_wordings"] == 1
    assert numbers["inputs"]["test_runs_not_there"] == [] and set(numbers["inputs"]["test"]) == set(ARMS)
    assert numbers["dev"]["families"] == 6                       # the dev numbers stay those of dev
    doc = paths["doc"].read_text(encoding="utf-8")
    assert "The test set is not in this file" not in doc
    assert ("**Test set** (T13; 4 families): all wordings pass is 26.7% (base), 76.6% (sft), 87.5% (grpo_a) "
            "and 100.0% (grpo_b); room after GRPO: 12.5 points in 1 family (grpo_a) and 0.0 points in 0 "
            "families (grpo_b).") in doc
    text = (paths["out"] / script.TABLES_NAME).read_text(encoding="utf-8")
    assert "### Per arm, on test" in text.split("## T13. ")[1]
    # te-01 of grpo_a passes 8 of 8 in styles 1 to 3 and 4 of 8 in the held-out style 4: over the training
    # styles only, its product is 1.
    assert test["arms"]["grpo_a"]["all_wordings_pass_training_styles"] == pytest.approx(1.0)
    assert "| grpo_a | 96.9% (124 of 128) | 87.5% | 100.0% | 12.5 | 1 of 4 |" in text.split("## T13. ")[1]


def test_one_unfinished_test_run_keeps_the_test_set_out(script, tmp_path):
    paths = build_tree(tmp_path, with_test=True)
    write_rollout(paths["main"] / "sft_test", rollout_records(TEST_SET["sft"], "sft", "sft_test"), 60.0,
                  complete=False)
    numbers = run(script, paths)
    assert numbers["test"] is None
    assert numbers["inputs"]["test_runs_not_there"] == [str(paths["main"] / "sft_test")]


def test_runs_without_a_family_that_got_worse(script, tmp_path):
    """dv-04 passes everywhere after training. Then no family meets the rule after training, no family has
    fewer passes than sft in every GRPO arm (dv-05 is lower in grpo_b only), and the room is (1/8) / 6 =
    2.1 points in each GRPO arm, in one family each: dv-03 in grpo_a and dv-05 in grpo_b."""
    dev = {arm: dict(families) for arm, families in DEV.items()}
    for arm in ("sft",) + GRPO:
        dev[arm]["dv-04"] = (8, 8, 8, 8)
    paths = build_tree(tmp_path, dev=dev)
    numbers = run(script, paths)
    assert numbers["dev"]["families_with_fewer_passes_than_sft_in_every_grpo_arm"] == []
    assert numbers["dev"]["families_with_fewer_passes_than_sft"] == {"grpo_a": [], "grpo_b": ["dv-05"]}
    assert numbers["dev"]["without_those_families"] == {} and numbers["family_facts"] is None
    assert numbers["effect"]["room_is_little"] is True
    doc = paths["doc"].read_text(encoding="utf-8")
    assert ("**Families (T5, T6).** After training no family meets the rule; the 3 families of base are in "
            "T5. No family has fewer passes than under sft in every GRPO arm.") in doc
    assert ("2.1 points in each GRPO arm, in 1 of 6 families. The largest part: dv-03 with 2.1 points in "
            "grpo_a and dv-05 with 2.1 points in grpo_b.") in doc
    assert "little room" not in doc
    assert ("If arm 1 ends where the saved runs did, only 1 family has room, fewer than 3: the interval can "
            "be above zero only if other families gain too.") in doc
    assert "held-out style 4, so" not in doc
    assert "| dv-0" not in doc                                   # no family table in the document
    assert "Without dv-04" not in doc and "A repair of" not in doc


def test_two_families_that_got_worse(script, tmp_path):
    """dv-03 gets worse too: sft 8, 8, 8, 6 (30 passes); grpo_a 8, 8, 2, 2 (20); grpo_b 8, 1, 8, 8 (25).
    Its wordings then disagree in both GRPO arms. The document shows both families, each with its table."""
    dev = {arm: dict(families) for arm, families in DEV.items()}
    dev["grpo_a"]["dv-03"], dev["grpo_b"]["dv-03"] = (8, 8, 2, 2), (8, 1, 8, 8)
    paths = build_tree(tmp_path, dev=dev)
    numbers = run(script, paths)
    assert numbers["dev"]["families_with_fewer_passes_than_sft_in_every_grpo_arm"] == ["dv-03", "dv-04"]
    assert numbers["dev"]["arms"]["grpo_a"]["families_whose_wordings_disagree"] == ["dv-03", "dv-04"]
    assert numbers["dev"]["arms"]["grpo_b"]["families_whose_wordings_disagree"] == ["dv-03"]
    # Without the two families, 4 families x 32 attempts are left.
    assert numbers["dev"]["without_those_families"]["grpo_a"]["valid_attempts"] == 128
    doc = paths["doc"].read_text(encoding="utf-8")
    assert ("**Families (T5, T6).** After training the rule is met by dv-03 (in grpo_a and grpo_b) and dv-04 "
            "(in sft and grpo_a); the 3 families of base are in T5. Fewer passes than under sft in every "
            "GRPO arm: dv-03 and dv-04.") in doc
    assert "| dv-03: passes of 8 |" in doc and "| dv-04: passes of 8 |" in doc
    assert "| grpo_a | 8 | 8 | 2 | 2 |" in doc and "| grpo_b | 8 | 1 | 8 | 8 |" in doc
    # dv-03: 12 failed attempts in grpo_a (styles 3 and 4) and 7 in grpo_b (style 2). The made-up scenario
    # file does not hold this family, so the document names no correct close for it.
    assert ("grpo_a fails in styles 3 and 4. grpo_b fails in style 2. Every failed attempt on dv-03 in these "
            "arms closes the request as denied (12 of 12 and 7 of 7).") in doc
    assert "Without dv-03 and dv-04 it is" in doc


def test_one_grpo_arm_only(script, tmp_path):
    paths = build_tree(tmp_path)
    assert script.main(["--main", str(paths["main"]), "--grpo", "grpo_a", "--ledger", str(paths["ledger"]),
                        "--scenarios", str(paths["scenarios"]), "--draws", "2000"]) == 0
    numbers = json.loads((paths["out"] / script.JSON_NAME).read_text(encoding="utf-8"))
    assert numbers["grpo_arms"] == ["grpo_a"] and numbers["arms"] == ["base", "sft", "grpo_a"]
    assert numbers["effect"]["spread_between_the_grpo_arms"] == {}
    assert [(row["a"], row["b"]) for row in numbers["dev"]["comparisons"]] == [
        ("base", "sft"), ("sft", "grpo_a"), ("base", "grpo_a")]
    doc = paths["doc"].read_text(encoding="utf-8")
    assert "one saved seed does not show the spread between runs (T10, T11)." in doc
    assert "grpo_a 3600.0 + 1800.0 = 5400.0 s = 2 hours" not in doc and "5400.0 s = 1.5 hours" in doc
    assert "Two runs of one recipe" not in doc
    assert "grpo_b" not in doc
    assert "Every failed attempt on dv-04 in grpo_a closes the request as denied (15 of 15)" in doc
    assert "18.8 points in 2 of 6 families (grpo_a)" in doc


def test_other_numbers_of_the_design_reach_the_files(script, tmp_path):
    """A bonus of 10 points, 2 attempts per wording, 5 families per step, 2 seeds per arm."""
    paths = build_tree(tmp_path)
    numbers = run(script, paths, "--bonus", "10", "--attempts-per-wording", "2", "--families-per-step", "5",
                  "--seeds-per-arm", "2")
    design = numbers["design"]
    assert (design["attempts_per_group"], design["attempts_per_step"], design["seeds_per_arm"]) == (6, 30, 2)
    assert numbers["one_wording_group"]["example"]["factor"] == pytest.approx(1.1)
    assert len(design["on_saved_attempts"]) == 4                 # 8 saved attempts per wording, 2 per draw
    assert numbers["cost"]["designs"]["2"]["runs"] == 4 and list(numbers["cost"]["designs"]) == ["2"]
    doc = paths["doc"].read_text(encoding="utf-8")
    assert "a pass also gets 10 x (q - q_mean) points" in doc
    assert "| Runs | 2 seeds per arm | judgement |" in doc
    assert "By chance alone that happens in 1 of 6 cases" in doc
    # 2 attempts per wording: A passes 2 of 2, B 1 of 2, C 0 of 2. q is 1/4 in A and 2/4 in B, q_mean is
    # (2 * 1/4 + 1/2) / 3 = 1/3. Arm 1: 3 passes of 6, so +50 and -50. Arm 2: a pass in A has
    # 50 + 10 * (1/4 - 1/3) = 49.2 and a pass in B has 50 + 10 * (1/2 - 1/3) = 51.7.
    assert "Example: wordings A, B and C pass 2, 1 and 0 of 2. Arm 1: a pass +50, a fail -50." in doc
    assert "Arm 2: a pass in A +49.2 and a pass in B +51.7: the weaker wording counts more." in doc


def test_the_saved_tables_of_the_frozen_program_are_compared(script, tmp_path):
    """A table that the frozen program makes from the same files agrees. A changed value in it is refused."""
    paths = build_tree(tmp_path)
    folder = paths["main"] / "tables"
    folder.mkdir()
    tables.main(["--arm", f"base={paths['main'] / 'base_dev' / 'episodes.jsonl'}",
                 "--arm", f"sft={paths['main'] / 'sft_dev' / 'episodes.jsonl'}",
                 "--arm", f"grpo_a={paths['main'] / 'grpo_a_dev' / 'episodes.jsonl'}",
                 "--baseline", "base", "--pair", "base:sft", "--pair", "sft:grpo_a", "--draws", "2000",
                 "--out-json", str(folder / "one.json"), "--out-md", str(folder / "one.md")])
    (folder / "other.json").write_text('{"not": "a table"}\n', encoding="utf-8")
    (folder / "list.json").write_text('{"arms": [1, 2], "splits": ["dev"]}\n', encoding="utf-8")
    numbers = run(script, paths)
    compared = numbers["inputs"]["saved_tables"]
    assert [(Path(row["file"]).name, row["agrees"], row["intervals_compared"]) for row in compared] == [
        ("one.json", True, True)]
    # 3 arms x (5 values + 3 variance shares) + 2 comparisons x 3 values = 30.
    assert compared[0]["values_compared"] == 30
    saved = json.loads((folder / "one.json").read_text(encoding="utf-8"))
    saved["arms"][1]["all_wordings_pass"] += 0.01
    (folder / "one.json").write_text(json.dumps(saved), encoding="utf-8")
    assert_refused(script, paths, "gives other values than this script")
    # A table of another split is not compared with the dev numbers.
    saved["splits"] = ["test"]
    (folder / "one.json").write_text(json.dumps(saved), encoding="utf-8")
    assert run(script, paths)["inputs"]["saved_tables"] == []


# ---------------------------------------------------------------------------
# Refusals: exit code 2, the reason on stderr, and nothing written
# ---------------------------------------------------------------------------
def assert_refused(script, paths: dict, phrase: str, *extra: str, capsys=None) -> None:
    for path in (paths["out"] / script.JSON_NAME, paths["out"] / script.TABLES_NAME, paths["doc"]):
        path.unlink(missing_ok=True)
    assert script.main(arguments(paths, *extra)) == 2
    assert not paths["doc"].exists() and not (paths["out"] / script.JSON_NAME).exists()
    assert not (paths["out"] / script.TABLES_NAME).exists()
    if capsys is not None:
        assert phrase in capsys.readouterr().err


def test_a_missing_or_unfinished_run_is_refused(script, tmp_path, capsys):
    paths = build_tree(tmp_path)
    (paths["main"] / "sft_dev" / "episodes.jsonl").rename(paths["main"] / "sft_dev" / "moved.jsonl")
    assert_refused(script, paths, "episodes.jsonl does not exist", capsys=capsys)
    (paths["main"] / "sft_dev" / "moved.jsonl").rename(paths["main"] / "sft_dev" / "episodes.jsonl")
    write_json(paths["main"] / "grpo_a_dev" / "run_manifest.json", {"complete": False, "seconds": 1.0})
    assert_refused(script, paths, "does not say that the run is complete", capsys=capsys)


def test_a_run_of_another_split_or_of_two_runs_is_refused(script, tmp_path, capsys):
    paths = build_tree(tmp_path)
    assert_refused(script, paths, "must hold records of the split 'dev' only",
                   "--folder", f"sft_dev={paths['main'] / 'sft_train'}", capsys=capsys)
    mixed = rollout_records(DEV["sft"], "sft", "sft_dev")
    mixed[0]["run"] = "another_run"
    write_jsonl(paths["main"] / "sft_dev" / "episodes.jsonl", mixed)
    assert_refused(script, paths, "holds 2 runs or set-ups", capsys=capsys)


def test_arms_that_do_not_cover_the_same_scenarios_are_refused(script, tmp_path, capsys):
    """The check is the one of the frozen table program."""
    paths = build_tree(tmp_path)
    fewer = [record for record in rollout_records(DEV["sft"], "sft", "sft_dev")
             if record["family_id"] != "dv-05"]
    write_jsonl(paths["main"] / "sft_dev" / "episodes.jsonl", fewer)
    assert_refused(script, paths, "does not cover the same scenarios", capsys=capsys)


def test_training_files_that_do_not_fit_their_log_are_refused(script, tmp_path, capsys):
    paths = build_tree(tmp_path)
    write_training_run(paths["main"] / "grpo_a", "grpo_a", kept=[1, 1, 1])        # the last step keeps 2
    assert_refused(script, paths, "the saved attempts give 2 kept groups, the log says 1", capsys=capsys)
    write_training_run(paths["main"] / "grpo_a", "grpo_a")
    log_path = paths["main"] / "grpo_a" / "train_log.jsonl"
    log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    log[0]["scenarios"] = ["tr-000/s1", "tr-009/s2"]
    write_jsonl(log_path, log)
    assert_refused(script, paths, "does not hold the scenarios of step 0", capsys=capsys)
    write_training_run(paths["main"] / "grpo_a", "grpo_a")
    (paths["main"] / "grpo_a" / "rollouts" / "step_00002.episodes.jsonl").unlink()
    assert_refused(script, paths, "step_00002.episodes.jsonl does not exist", capsys=capsys)


def test_grpo_runs_with_two_layouts_are_refused(script, tmp_path, capsys):
    paths = build_tree(tmp_path)
    write_training_run(paths["main"] / "grpo_b", "grpo_b", steps=TRAINING["grpo_b"][:2], kept=[1, 0])
    assert_refused(script, paths, "the GRPO runs differ in their layout", capsys=capsys)


def test_a_scenario_file_that_the_runs_do_not_name_is_refused(script, tmp_path, capsys):
    """The dev records name the SHA-256 of their scenario file. The file of the folder must have it."""
    paths = build_tree(tmp_path)
    file_hash = hashlib.sha256((paths["scenarios"] / "dev.jsonl").read_bytes()).hexdigest()
    for arm in ARMS:
        records = [{**record, "data_sha256": file_hash}
                   for record in rollout_records(DEV[arm], arm, f"{arm}_dev")]
        write_jsonl(paths["main"] / f"{arm}_dev" / "episodes.jsonl", records)
    numbers = run(script, paths)
    assert numbers["inputs"]["scenario_file"]["checked_against_the_records"] is True
    assert numbers["inputs"]["scenario_file"]["sha256"] == file_hash
    (paths["scenarios"] / "dev.jsonl").write_text(
        (paths["scenarios"] / "dev.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert_refused(script, paths, "is not the scenario file of the dev runs", capsys=capsys)


def test_without_the_ledger_and_the_scenario_file_the_files_say_less(script, tmp_path):
    paths = build_tree(tmp_path)
    paths["ledger"].unlink()
    (paths["scenarios"] / "dev.jsonl").unlink()
    numbers = run(script, paths)
    assert numbers["cost"]["prices"] is None and numbers["cost"]["designs"]["3"]["usd_low"] is None
    assert numbers["family_facts"] is None and numbers["inputs"]["scenario_file"] is None
    doc = paths["doc"].read_text(encoding="utf-8")
    assert "USD" not in doc and "6 runs: 6 x 2 = about 12 GPU hours." in doc
    assert "the correct close is" not in doc
    assert "Every failed attempt on dv-04 in these arms closes the request as denied (15 of 15 and 29 of 29)." in doc
    assert "for run time only" not in doc


def test_wrong_arguments_are_refused(script, tmp_path, capsys):
    paths = build_tree(tmp_path)
    assert_refused(script, paths, "--folder must look like KEY=PATH", "--folder", "sft_dev", capsys=capsys)
    assert_refused(script, paths, "the numbers of the design must be above zero", "--bonus", "0",
                   capsys=capsys)
    assert script.main(["--main", str(paths["main"]), "--grpo", "sft", "--ledger", str(paths["ledger"]),
                        "--scenarios", str(paths["scenarios"])]) == 2
    assert "give each GRPO arm once" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The real saved runs (skipped if they are not there)
# ---------------------------------------------------------------------------
REAL_MAIN = ROOT / "results" / "main"
REAL_NEEDED = ([REAL_MAIN / f"{arm}_dev" / "episodes.jsonl" for arm in ("base", "sft", "grpo_seed1",
                                                                           "grpo_seed2")]
               + [REAL_MAIN / arm / "train_log.jsonl" for arm in ("grpo_seed1", "grpo_seed2")]
               + [REAL_MAIN / "sft_train" / "episodes.jsonl"])


def test_the_script_on_the_real_saved_runs(script, tmp_path):
    """The real runs give the numbers of the frozen tables, and the files in the repository are what the
    script makes from them now."""
    missing = [path for path in REAL_NEEDED if not path.is_file()]
    if missing:
        pytest.skip(f"the saved runs are not here: {missing[0]}")
    out, doc = tmp_path / "out", tmp_path / "stretch.md"
    assert script.main(["--out", str(out), "--doc", str(doc)]) == 0
    numbers = json.loads((out / script.JSON_NAME).read_text(encoding="utf-8"))
    dev = numbers["dev"]
    assert (dev["families"], dev["scenarios"], dev["attempts_per_arm"]) == (48, 192, 1536)
    # The pass rates of the frozen dev tables, as the task of this piece names them.
    rates = [round(100 * dev["arms"][arm]["pass_rate"], 1) for arm in numbers["arms"]]
    assert rates == [39.8, 89.4, 98.6, 97.7]
    compared = numbers["inputs"]["saved_tables"]
    assert compared and all(row["agrees"] for row in compared)
    for arm in numbers["grpo_arms"]:
        run_numbers = numbers["one_wording_group"]["saved_runs"][arm]
        assert (run_numbers["groups"], run_numbers["attempts"]) == (640, 10240)
        assert run_numbers["kept"] == run_numbers["kept_with_bonus"]
        assert run_numbers["kept"] == (run_numbers["kept_mixed_full_reward"] + run_numbers["kept_one_outcome"]
                                       + run_numbers["kept_mixed_other"])
        assert 0.0 <= dev["arms"][arm]["room"] <= 1.0
    text = doc.read_text(encoding="utf-8")
    assert "**The experiment was not run.**" in "\n".join(text.splitlines()[:3])
    # The document must stay short: two pages. Words are counted without the marks of the tables.
    words = [word for word in text.split() if word.strip("|-:")]
    assert len(words) <= 1500, f"the document has {len(words)} words"
    assert len(text) <= DOC_CHARACTERS, f"the document has {len(text)} characters"
    # The findings of the review, on the real runs.
    for arm, lowest in (("grpo_seed1", 81), ("grpo_seed2", 77)):
        assert numbers["one_wording_group"]["saved_runs"][arm]["lowest_pass_reward"] == lowest
        assert numbers["one_wording_group"]["saved_runs"][arm]["fails_not_at_zero"] == 0
    assert "required items" not in text and "Correctness alone" not in text
    assert "In every GRPO arm dv-46 passes 0 of 8 in its held-out style 4" in text
    assert ("4 of the 5 families with room would have to gain and no other family lose. Yet two runs of one "
            "recipe already differ in 4 of 48 families on all wordings pass.") in text
    assert "## 6. What the experiment would cost" in text and "Beyond the cost" not in text
    assert ("The experiment was not run. The frozen `ANALYSIS_PLAN.md` does not list it among its arms or its "
            "planned extras (section 3).") in text
    assert ("grpo_seed2 fails in all 4 styles (3, 0, 0 and 0 of 8): by the rule of this note its wordings do "
            "not disagree.") in text
    # "Only the weight" is exact where every pass scores 100; T2 counts the kept groups where it is not.
    for arm, exact, other in (("grpo_seed1", 47, 8), ("grpo_seed2", 45, 5)):
        run_numbers = numbers["one_wording_group"]["saved_runs"][arm]
        assert (run_numbers["kept_mixed_full_reward"], run_numbers["kept_mixed_other"]) == (exact, other)
    assert "Where every pass scores 100 (47 and 45 kept groups" in text and "in 8 and 5 such groups" in text
    assert_sums_add_up(text)
    # Room for the test set: the document must stay inside its limit after the test runs are copied in.
    # When the test runs are there, the sentence about the test set is in the document already, and the
    # limit is asked of the document as it is.
    if "**Test set** (T13" in text:
        assert len(text) <= DOC_CHARACTERS, f"the document has {len(text)} characters; the limit is {DOC_CHARACTERS}"
    else:
        assert len(text) + MOST_GROWTH_WITH_THE_TEST_SET <= DOC_CHARACTERS, (
            f"the document has {len(text)} characters; with the test set it could pass {DOC_CHARACTERS}")
    # The files in the repository are the ones that the script writes now (only the paths differ).
    in_repository = {"doc": ROOT / "results" / "stretch.md",
                     "tables": REAL_MAIN / "report" / "stretch" / script.TABLES_NAME,
                     "numbers": REAL_MAIN / "report" / "stretch" / script.JSON_NAME}
    if not all(path.is_file() for path in in_repository.values()):
        pytest.skip("the output files of the script are not in the repository yet")
    shown = {str(out / script.TABLES_NAME): "results/main/report/stretch/stretch_numbers.md",
             str(out / script.JSON_NAME): "results/main/report/stretch/stretch_numbers.json",
             str(doc): "results/stretch.md"}

    def as_in_repository(content: str) -> str:
        for made_here, name in shown.items():
            content = content.replace(made_here, name)
        return content

    # The spend ledger grows with every rented machine, but only the machines rented up to the end of the
    # last saved GRPO run are read: the files are compared byte for byte (after the paths).
    stale = "run .venv/bin/python scripts/stretch_numbers.py again: the saved runs or the script changed"
    assert as_in_repository(text) == in_repository["doc"].read_text(encoding="utf-8"), stale
    fresh_tables = as_in_repository((out / script.TABLES_NAME).read_text(encoding="utf-8"))
    assert fresh_tables == in_repository["tables"].read_text(encoding="utf-8"), stale
    fresh_numbers = as_in_repository((out / script.JSON_NAME).read_text(encoding="utf-8"))
    assert fresh_numbers == in_repository["numbers"].read_text(encoding="utf-8"), stale
