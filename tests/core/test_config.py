"""accessdesk/config.py: RunConfig."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from accessdesk.config import COMPARED_FIELDS, DEFAULT, RunConfig

# The defaults of section 11 of the build specification, typed here a second time on purpose, with the three
# fields of calibration round 2: `mix_facts` and the two seeds of placeholder data.
# Five values are those of the freeze day (2026-11-18), not of the specification: the token cap 260 (twice
# the 99th percentile of the call lengths in calibration round 4), the final design (`homonym_share` 0.5
# and `mix_facts` True) and the two final seeds (the date of the freeze, and the next number).
SPEC_DEFAULTS = {
    "turn_limit": 8, "max_new_tokens": 260, "parser_strict": True,
    "temperature": 1.0, "top_k": 0, "top_p": 1.0, "chunk_size": 32,
    "pass_reward": 100, "malformed_penalty": 5, "malformed_cap": 15,
    "turn_penalty": 2, "turn_cap": 10, "penalise_malformed_on_fail": False,
    "n_people": 8, "n_roles": 6, "n_departments": 3, "homonym_share": 0.5, "mix_facts": True,
    "n_train": 200, "n_dev": 48, "n_test": 48, "n_calib": 24,
    "data_seed": 20261118, "calib_seed": 20261119,
    "placeholder_data_seed": 9001, "placeholder_calib_seed": 9002,
}


def test_defaults_are_the_specified_ones():
    assert RunConfig().to_dict() == SPEC_DEFAULTS


def test_default_value_types():
    for name, value in RunConfig().to_dict().items():
        assert type(value) is type(SPEC_DEFAULTS[name]), name


def test_default_object():
    assert DEFAULT == RunConfig()


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.turn_limit = 9  # type: ignore[misc]


def test_split_sizes_are_multiples_of_four():
    for name in ("n_train", "n_dev", "n_test", "n_calib"):
        assert getattr(DEFAULT, name) % 4 == 0, name


def test_to_dict_has_every_field_and_is_json():
    as_dict = DEFAULT.to_dict()
    assert list(as_dict) == [field.name for field in dataclasses.fields(RunConfig)]
    assert json.loads(json.dumps(as_dict)) == as_dict


def test_sha256_is_the_hash_of_the_sorted_json():
    wanted = hashlib.sha256(json.dumps(SPEC_DEFAULTS, sort_keys=True).encode("utf-8")).hexdigest()
    assert DEFAULT.sha256() == wanted
    assert RunConfig().sha256() == wanted


@pytest.mark.parametrize("name", list(SPEC_DEFAULTS))
def test_sha256_changes_with_every_field(name):
    old = SPEC_DEFAULTS[name]
    new = (not old) if isinstance(old, bool) else old + 1
    changed = dataclasses.replace(DEFAULT, **{name: new})
    assert changed.to_dict()[name] == new
    assert changed.sha256() != DEFAULT.sha256()


def test_every_config_field_is_compared_between_arms_or_is_known_not_to_matter_for_a_comparison():
    """A new field of the config must be put in one of the three groups by hand: compared between the arms
    of a results table; the chunk size (it only changes which random numbers are drawn); or a field that
    describes the scenario files (a record names its file with `data_sha256`)."""
    compared = {"turn_limit", "max_new_tokens", "parser_strict", "temperature", "top_k", "top_p",
                "pass_reward", "malformed_penalty", "malformed_cap", "turn_penalty", "turn_cap",
                "penalise_malformed_on_fail"}
    data = {"n_people", "n_roles", "n_departments", "homonym_share", "mix_facts", "n_train", "n_dev",
            "n_test", "n_calib", "data_seed", "calib_seed", "placeholder_data_seed", "placeholder_calib_seed"}
    assert set(COMPARED_FIELDS) == compared and len(COMPARED_FIELDS) == len(compared)
    assert set(DEFAULT.to_dict()) == compared | data | {"chunk_size"}
    assert DEFAULT.compared_settings() == {name: getattr(DEFAULT, name) for name in COMPARED_FIELDS}
    assert list(DEFAULT.compared_settings()) == list(COMPARED_FIELDS)
    # The chunk size and the data fields change the hash of the whole config and leave the settings alone.
    other = dataclasses.replace(DEFAULT, chunk_size=12, data_seed=77)
    assert other.sha256() != DEFAULT.sha256() and other.compared_settings() == DEFAULT.compared_settings()
    assert dataclasses.replace(DEFAULT, max_new_tokens=96).compared_settings() != DEFAULT.compared_settings()


def test_placeholder_data_has_seeds_of_its_own():
    """Data built from the placeholder text takes the two placeholder seeds; data from a real text file takes
    the config as it is. The four seeds are four different numbers, so no placeholder world is a real one."""
    real, placeholder = DEFAULT.with_seeds_for(False), DEFAULT.with_seeds_for(True)
    assert real is DEFAULT
    assert (placeholder.data_seed, placeholder.calib_seed) == (9001, 9002)
    assert placeholder == dataclasses.replace(DEFAULT, data_seed=9001, calib_seed=9002)
    seeds = [DEFAULT.data_seed, DEFAULT.calib_seed, DEFAULT.placeholder_data_seed,
             DEFAULT.placeholder_calib_seed]
    assert len(set(seeds)) == 4
    # A changed data seed (the final seed, at the freeze) does not move the placeholder data, and the
    # placeholder seeds do not touch what the episodes are scored with.
    assert dataclasses.replace(DEFAULT, data_seed=77, calib_seed=78).with_seeds_for(True) == placeholder
    assert placeholder.compared_settings() == DEFAULT.compared_settings()
