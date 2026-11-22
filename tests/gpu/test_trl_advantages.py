"""The advantages of `group_advantages` against TRL's own computation with reward scaling switched off.

Skipped if `trl` is not installed (it is an optional package of the rented machine).

How TRL computes advantages (trl 1.18.0, trl/trainer/grpo_trainer.py, inside the method
`GRPOTrainer._generate_and_score_completions`, lines 2931-2976), and which setting switches the scaling off:

    mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)            # line 2931
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)     # line 2932
    ...
    advantages = rewards - mean_grouped_rewards                                               # line 2951
    if self.scale_rewards != "none":                                                          # line 2952
        advantages = advantages / (std_rewards + 1e-4)                                        # line 2953
    ...
    advantages = torch.nan_to_num(advantages, nan=0.0)                                        # line 2976

`scale_rewards` accepts "group" (the default), "batch" and "none"; True means "group" and False means "none"
(trl/trainer/grpo_config.py lines 227-236 and 1119). So with `scale_rewards="none"` (or False) the advantage
is the reward minus the mean of its group, NaN rewards are left out of the mean and get advantage 0.

WHICH OF THE TWO WAYS THIS TEST USES: the lines are COPIED into `trl_advantages` below. They cannot be called:
they are in the middle of a method of several hundred lines that needs a trainer with a model, a tokenizer and
a dataset. To keep the copy honest, `test_the_copied_lines_are_in_the_installed_trl` reads the source file of
the installed TRL and fails if one of the copied lines is no longer there, word for word.

Two differences between TRL and this repository are by design and are tested as such:
  - TRL has no "drop a tied group" switch. A tied group gets advantage 0 in both, so the numbers agree; this
    repository in addition reports the group as not kept.
  - TRL keeps a group whatever number of its rewards is NaN. This repository drops a group with fewer than
    half of its samples valid (PLAN 5.6). On such a group the numbers differ, on purpose.
The computation is done in float64 on both sides (TRL itself holds rewards in float32).
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from accessdesk import impl

# The lines of trl/trainer/grpo_trainer.py that `trl_advantages` copies, and the two lines (the `if` and the
# division) that "none" switches off.
COPIED_LINES = (
    "mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)",
    "mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)",
    "advantages = rewards - mean_grouped_rewards",
    'if self.scale_rewards != "none":',
    "advantages = advantages / (std_rewards + 1e-4)",
    "advantages = torch.nan_to_num(advantages, nan=0.0)",
)
# The line of trl/trainer/grpo_config.py that says which values mean "no scaling".
CONFIG_LINE = ('self.scale_rewards = {True: "group", False: "none"}'
               '.get(self.scale_rewards, self.scale_rewards)')
NAN = float("nan")


def _trl_folder() -> Path:
    """The folder of the installed `trl` package; the test is skipped if there is none. TRL is not
    imported."""
    spec = importlib.util.find_spec("trl")
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    return Path(spec.origin).parent


def _source_lines(path: Path) -> list[str]:
    """The lines of a source file without the white space at their ends."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def trl_advantages(rewards: list[list[float]]):
    """TRL's advantages for `scale_rewards="none"`: the lines 2931, 2932, 2951 and 2976 of grpo_trainer.py.

    `rewards` is [groups, num_generations]; TRL holds it flat, one group after the other. Returns
    [groups, num_generations].
    """
    import torch

    num_generations = len(rewards[0])
    rewards = torch.tensor(rewards, dtype=torch.float64).reshape(-1)                         # flat, as in TRL
    mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)            # line 2931
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)     # line 2932
    advantages = rewards - mean_grouped_rewards                                               # line 2951
    # lines 2952-2953 divide by the standard deviation only if scale_rewards != "none": not done here
    advantages = torch.nan_to_num(advantages, nan=0.0)                                        # line 2976
    return advantages.view(-1, num_generations)


def _ours(rewards: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    return impl.get("group_advantages")(np.array(rewards, dtype=float))


def test_the_copied_lines_are_in_the_installed_trl():
    folder = _trl_folder()
    print(f"\ntrl {importlib.metadata.version('trl')}; group_advantages: the "
          f"{impl.which('group_advantages')} version")
    trainer = _source_lines(folder / "trainer" / "grpo_trainer.py")
    missing = [line for line in COPIED_LINES if line not in trainer]
    assert not missing, f"TRL's advantage code has changed; these copied lines are gone: {missing}"
    # The order of the lines is the order of the computation.
    positions = [trainer.index(line) for line in COPIED_LINES]
    assert positions == sorted(positions)
    config = _source_lines(folder / "trainer" / "grpo_config.py")
    assert CONFIG_LINE in config, "TRL's handling of scale_rewards has changed"


def test_equal_on_hand_made_rewards():
    import torch

    _trl_folder()
    rewards = [[1.0, 0.0, 1.0, 0.0],        # mixed: mean 0.5 -> +0.5, -0.5, +0.5, -0.5
               [1.0, 1.0, 1.0, 1.0],        # tied: all 0
               [1.0, 0.95, 0.0, 0.0],       # a penalty: mean 0.4875 -> 0.5125, 0.4625, -0.4875, -0.4875
               [0.0, 0.0, 0.0, 0.0],        # tied at 0: all 0
               [1.0, 0.95, 0.95, 1.0]]      # penalties only: mean 0.975 -> 0.025, -0.025, -0.025, 0.025
    by_hand = [[0.5, -0.5, 0.5, -0.5], [0.0] * 4, [0.5125, 0.4625, -0.4875, -0.4875], [0.0] * 4,
               [0.025, -0.025, -0.025, 0.025]]
    ours, keep = _ours(rewards)
    theirs = trl_advantages(rewards)
    assert torch.allclose(theirs, torch.tensor(by_hand, dtype=torch.float64), atol=1e-12)
    assert np.allclose(ours, theirs.numpy(), atol=1e-12)
    assert keep.tolist() == [True, False, True, False, True]
    # No scaling: the penalty-only group keeps its small advantages (dividing by its standard deviation of
    # about 0.029 would turn 0.025 into about 0.87, as large as a pass against a fail).
    assert np.abs(ours[4]).max() == pytest.approx(0.025, abs=1e-12)


def test_equal_with_lost_samples_when_at_least_half_are_valid():
    _trl_folder()
    rewards = [[1.0, NAN, 0.0, 0.0],        # 3 valid, mean 1/3 -> 2/3, 0, -1/3, -1/3
               [NAN, 1.0, NAN, 0.0],        # 2 valid of 4 (exactly half stays), mean 0.5 -> 0, 0.5, 0, -0.5
               [1.0, 1.0, NAN, 1.0]]        # tied among the valid ones: all 0
    ours, keep = _ours(rewards)
    theirs = trl_advantages(rewards).numpy()
    assert np.allclose(ours, theirs, atol=1e-12)
    assert np.allclose(ours[0], [2 / 3, 0.0, -1 / 3, -1 / 3], atol=1e-12)
    assert np.allclose(ours[1], [0.0, 0.5, 0.0, -0.5], atol=1e-12)
    assert keep.tolist() == [True, True, False]
    assert not np.isnan(ours).any()


def test_the_one_difference_a_group_with_fewer_than_half_valid_samples():
    _trl_folder()
    rewards = [[1.0, NAN, NAN, 0.0, NAN, NAN, NAN, 0.0]]         # 3 valid of 8: fewer than half
    ours, keep = _ours(rewards)
    theirs = trl_advantages(rewards).numpy()
    assert keep.tolist() == [False] and not ours.any()            # this repository drops the group
    assert np.allclose(theirs[0], [2 / 3, 0, 0, -1 / 3, 0, 0, 0, -1 / 3], atol=1e-12)    # TRL keeps it


def test_equal_on_many_generated_groups():
    _trl_folder()
    grid = np.array([0.0, 0.81, 0.88, 0.95, 1.0])                # the reward values that the reward can give
    generator = np.random.default_rng(20261117)
    rewards = grid[generator.integers(0, len(grid), size=(300, 8))]
    rewards[::25] = rewards[::25, :1]                            # every 25th group is tied on purpose
    ours, keep = _ours(rewards.tolist())
    theirs = trl_advantages(rewards.tolist()).numpy()
    assert np.allclose(ours, theirs, atol=1e-12)
    tied = rewards.max(axis=1) == rewards.min(axis=1)
    assert keep.tolist() == (~tied).tolist()
    assert np.allclose(ours.sum(axis=1), 0.0, atol=1e-12)        # advantages of a group add up to 0
