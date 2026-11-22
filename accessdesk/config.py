"""One config object for data generation, data collection, training and evaluation."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass


# The settings that decide what an episode is and how it is scored: the turn limit, the token cap, the parser
# mode, the sampling settings and the reward numbers. Every episode record carries them (`settings`), and the
# results table refuses to compare arms that differ in one of them.
# Not in the list: `chunk_size` (it only changes which random numbers are drawn), and the world sizes, the
# share of same-name families, `mix_facts`, the split sizes and the seeds (they describe the scenario files,
# which a record names with `data_sha256`; the build writes them into the folder's summary.json).
COMPARED_FIELDS = ("turn_limit", "max_new_tokens", "parser_strict", "temperature", "top_k", "top_p",
                   "pass_reward", "malformed_penalty", "malformed_cap", "turn_penalty", "turn_cap",
                   "penalise_malformed_on_fail")


@dataclass(frozen=True)
class RunConfig:
    # Harness and sampling.
    turn_limit: int = 8
    # The token cap per turn. The rule (PLAN 5.4): twice the 99th percentile of the lengths of well-formed
    # calls, and at least 128. Measured in calibration round 4 on Qwen3.5-2B with the final design
    # (results/calibration/round4/2b_mixed, 6510 well-formed calls): the 99th percentile is 130 tokens by the
    # nearest-rank rule of accessdesk/eval/calibration.py, so the cap is 260. The report of that round prints
    # the same value (results/calibration/round4/report_mixed/report.md). The calibration runs used 256.
    max_new_tokens: int = 260
    parser_strict: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    chunk_size: int = 32
    # Reward, in whole hundredths.
    pass_reward: int = 100
    malformed_penalty: int = 5
    malformed_cap: int = 15
    turn_penalty: int = 2
    turn_cap: int = 10
    penalise_malformed_on_fail: bool = False
    # World size.
    n_people: int = 8
    n_roles: int = 6
    n_departments: int = 3
    # The share of families that hold two people with one full name. 0.5 is the final design. A rule that
    # the candidate fixed before the numbers, and calibration round 4, decided it. Until the freeze day
    # (2026-11-18) the default was 0.2.
    homonym_share: float = 0.5
    # False: the outcome of a family fixes its three rule facts, and no two of them are true together (an
    # inactive person never asks for a sensitive role). True: the facts that a later rule would look at
    # may be true as well, so the order of the rules is tested (accessdesk/env/generator.py, MIXED_KINDS).
    # True is the final design, decided together with `homonym_share`. Until the freeze day the default
    # was False.
    mix_facts: bool = True
    # Families per split.
    n_train: int = 200
    n_dev: int = 48
    n_test: int = 48
    n_calib: int = 24
    # The final seeds. `data_seed` is the date of the freeze, 2026-11-18. It was written down before any
    # final family was built. `calib_seed` is the next number; no earlier build used it. Every calibration
    # folder (data/scenarios_calib*) was built with data seed 1 and calibration seed 2, so a final build
    # starts from other seeds than every calibration round.
    data_seed: int = 20261118
    calib_seed: int = 20261119
    # The seeds of data that is built from the placeholder text (the unit-test data and the smoke tests).
    # They are numbers of their own, so that the worlds that the tests read are never the worlds of a
    # build from the real text. See `with_seeds_for`.
    placeholder_data_seed: int = 9001
    placeholder_calib_seed: int = 9002

    def to_dict(self) -> dict:
        """Every field, by name, as plain JSON values."""
        return dataclasses.asdict(self)

    def with_seeds_for(self, text_is_placeholder: bool) -> "RunConfig":
        """The config for data that is built from a text file: this config as it is for a real text file;
        for the placeholder text a copy whose `data_seed` and `calib_seed` are the two placeholder seeds.

        scripts/build_scenarios.py, scripts/run_scripted.py and tests/conftest.py call it, so the generator
        itself reads `data_seed` and `calib_seed` only."""
        if not text_is_placeholder:
            return self
        return dataclasses.replace(self, data_seed=self.placeholder_data_seed,
                                   calib_seed=self.placeholder_calib_seed)

    def compared_settings(self) -> dict:
        """The fields of COMPARED_FIELDS, by name: what must be the same in every arm of one results table."""
        return {name: getattr(self, name) for name in COMPARED_FIELDS}

    def sha256(self) -> str:
        """Hash of json.dumps(to_dict(), sort_keys=True)."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


DEFAULT = RunConfig()
