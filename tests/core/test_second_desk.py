"""The second desk (accessdesk/audit/second_desk.py, scripts/second_desk_report.py): the numbers on both
sides of each rule, and every refusal.

The attempts come from the scripted policies, played under the changed rule on a scenario folder that the
real build script writes into a temporary folder, and from hand-made records. Nothing is written into the
repository.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from accessdesk.audit import second_desk
from accessdesk.env.scenarios import load_families
from accessdesk.eval import metrics
from accessdesk.harness import policies
from accessdesk.harness.episode import Episode
from accessdesk.harness.records import write_records
from tests.helpers.make_records import make_record

ROOT = Path(__file__).resolve().parents[2]
RULESET = second_desk.RULESET
SAMPLES = second_desk.ATTEMPTS_PER_SCENARIO     # attempts per scenario in the played files: 8, as the plan


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report_script():
    return load_script("second_desk_report")


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory, text, cfg) -> Path:
    """A scenario folder as the build script writes it, from the text and the config of the tests."""
    assert text.policy_shift_variant == RULESET
    folder = tmp_path_factory.mktemp("second_desk") / "scenarios_for_second_desk"
    load_script("build_scenarios").build(text, ROOT / "data" / "names", folder, cfg)
    return folder


@pytest.fixture(scope="module")
def dev(scenarios) -> dict[str, dict]:
    """The dev families of the built folder, by id."""
    return {family["family_id"]: family for family in load_families(scenarios / "dev.jsonl")}


@pytest.fixture(scope="module")
def lists(scenarios) -> dict[str, list[str]]:
    """The two lists of the set policy_shift of the built folder."""
    sets = json.loads((scenarios / "sets.json").read_text(encoding="utf-8"))
    return {part: sets["policy_shift"][part] for part in second_desk.PARTS}


@pytest.fixture(scope="module")
def data_sha256(scenarios) -> str:
    return hashlib.sha256((scenarios / "dev.jsonl").read_bytes()).hexdigest()


def play(policy, family_ids, dev, text, cfg, data_sha256, *, ruleset=RULESET, prompt="policy_shift",
         samples=SAMPLES) -> list[dict]:
    """`samples` records per scenario of the named families, played by a scripted policy."""
    records = []
    for family_id in family_ids:
        for wording in dev[family_id]["wordings"]:
            for sample in range(samples):
                episode = Episode(dev[family_id], wording["style"], text, cfg, ruleset=ruleset, prompt=prompt)
                policies.run_policy(episode, policy)
                records.append(episode.record(run="played", arm="played", sample=sample,
                                              seed={"run_seed": 0, "pass": sample, "chunk": 0, "pos": 0},
                                              data_sha256=data_sha256))
    return records


@pytest.fixture(scope="module")
def follows_the_new_rule(dev, lists, text, cfg, data_sha256) -> dict[str, list[dict]]:
    """The oracle of the changed rule on both lists: it asks for approval where the old rule granted."""
    policy = policies.oracle(text, RULESET)
    return {part: play(policy, lists[part], dev, text, cfg, data_sha256) for part in second_desk.PARTS}


@pytest.fixture(scope="module")
def follows_the_old_rule(dev, lists, text, cfg, data_sha256) -> dict[str, list[dict]]:
    """The oracle of the default rules, graded under the changed rule: it still grants."""
    policy = policies.oracle(text)
    return {part: play(policy, lists[part], dev, text, cfg, data_sha256) for part in second_desk.PARTS}


def with_grants(record: dict, grants: list[list[str]]) -> dict:
    """A copy of a hand-made record whose final state holds these grant records."""
    changed = copy.deepcopy(record)
    changed["final_state"]["grants"] = grants
    return changed


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------
def test_a_final_state_with_a_grant_record_counts_and_one_without_does_not():
    without = make_record("dv-000")
    assert second_desk.holds_a_grant_record(without) is False
    assert second_desk.holds_a_grant_record(with_grants(without, [["p", "q"]])) is True


def test_a_grant_for_another_person_is_a_grant_record_but_not_the_requested_role():
    family = {"truth": {"person_id": "p", "role_id": "q"}}
    other = with_grants(make_record("dv-000"), [["somebody", "q"]])
    assert second_desk.holds_a_grant_record(other) is True
    assert second_desk.grants_the_requested_role(other, family) is False
    assert second_desk.grants_the_requested_role(with_grants(other, [["p", "q"]]), family) is True


def test_family_shares_of_the_pass_mark_are_the_family_pass_rates():
    records = [make_record("dv-000", 1, 0), make_record("dv-000", 1, 1, passed=False),
               make_record("dv-001", 2, 0, passed=False), make_record("dv-001", 2, 1, passed=False),
               make_record("dv-002", 1, 0, harness_error="lost")]
    shares = second_desk.family_shares(records, lambda record: bool(record["passed"]))
    assert shares == metrics.family_pass_rates(records) == {"dv-000": 0.5, "dv-001": 0.0}


def test_a_share_is_family_weighted_and_stands_beside_its_plain_counts():
    # dv-000: 1 of 4 attempts holds a grant record. dv-001: 2 of 2. Family-weighted: (0.25 + 1.0) / 2.
    grant = [["p", "q"]]
    records = [with_grants(make_record("dv-000", 1, 0), grant)]
    records += [make_record("dv-000", 1, sample) for sample in (1, 2, 3)]
    records += [with_grants(make_record("dv-001", 1, sample), grant) for sample in (0, 1)]
    row = second_desk.share_row(records, second_desk.holds_a_grant_record)
    assert row["share"] == pytest.approx(0.625)
    assert (row["n"], row["of"], row["n_families"]) == (3, 6, 2)
    assert row["plain_share"] == pytest.approx(0.5)
    assert row["per_family"] == {"dv-000": 0.25, "dv-001": 1.0}


def test_the_interval_of_a_share_is_the_family_bootstrap_of_its_family_values():
    records = [make_record(f"dv-{family:03d}", 1, sample, passed=sample < family)
               for family in range(5) for sample in range(4)]
    row = second_desk.share_row(records, lambda record: bool(record["passed"]), n_draws=500, seed=3)
    values = list(metrics.family_pass_rates(records).values())
    interval = metrics.family_bootstrap_ci(values, 500, 3, second_desk.LEVEL)
    assert (row["share"], row["lo"], row["hi"]) == (interval["mean"], interval["lo"], interval["hi"])
    assert row["lo"] < row["share"] < row["hi"]


def test_an_attempt_lost_to_a_harness_error_is_in_no_share_and_is_counted(dev, lists, follows_the_new_rule):
    records = copy.deepcopy(follows_the_new_rule["differ"])
    records[0]["harness_error"] = "lost"
    numbers = second_desk.part_numbers(records, "differ", dev)
    assert numbers["n_harness_errors"] == 1
    assert numbers["n_attempts"] == numbers["pass_rate"]["of"] == len(records) - 1
    assert numbers["n_records"] == len(records)


def test_a_policy_that_follows_the_new_rule_passes_and_gives_no_role(dev, lists, follows_the_new_rule):
    differ = second_desk.part_numbers(follows_the_new_rule["differ"], "differ", dev)
    control = second_desk.part_numbers(follows_the_new_rule["control"], "control", dev)
    assert differ["pass_rate"]["share"] == 1.0
    assert differ["still_gives_the_role"]["share"] == 0.0
    assert differ["still_gives_the_role"]["n"] == 0
    assert control["pass_rate"]["share"] == 1.0
    assert (differ["n_families"], control["n_families"]) == (len(lists["differ"]), len(lists["control"]))
    assert differ["attempts_per_scenario"] == [SAMPLES]


def test_a_policy_that_follows_the_old_rule_fails_on_differ_and_still_gives_the_role(dev,
                                                                                    follows_the_old_rule):
    differ = second_desk.part_numbers(follows_the_old_rule["differ"], "differ", dev)
    control = second_desk.part_numbers(follows_the_old_rule["control"], "control", dev)
    assert differ["pass_rate"]["share"] == 0.0
    gives = differ["still_gives_the_role"]
    assert gives["share"] == 1.0
    n_differ = len(follows_the_old_rule["differ"])
    assert gives["n"] == gives["of"] == gives["n_for_the_requested_role"] == n_differ
    # On the control families the right end state is the same under both rules.
    assert control["pass_rate"]["share"] == 1.0
    assert control["still_gives_the_role"]["n"] == 0


def test_a_family_whose_attempts_are_all_harness_errors_drops_out_and_is_named(dev, lists,
                                                                              follows_the_new_rule):
    records = copy.deepcopy(follows_the_new_rule["differ"])
    gone = lists["differ"][2]
    for record in records:
        if record["family_id"] == gone:
            record["harness_error"] = "lost"
    numbers = second_desk.part_numbers(records, "differ", dev)
    assert numbers["families_dropped"] == [gone]
    assert numbers["n_families"] == len(lists["differ"])              # families with a record
    assert numbers["pass_rate"]["n_families"] == len(lists["differ"]) - 1       # families behind the mean
    assert numbers["still_gives_the_role"]["n_families"] == len(lists["differ"]) - 1
    assert gone not in numbers["pass_rate"]["per_family"]
    # One lost attempt does not drop a family.
    one_lost = copy.deepcopy(follows_the_new_rule["differ"])
    one_lost[0]["harness_error"] = "lost"
    assert second_desk.part_numbers(one_lost, "differ", dev)["families_dropped"] == []


# ---------------------------------------------------------------------------
# The coverage of the plan: 8 attempts per scenario, run seed 0
# ---------------------------------------------------------------------------
def test_eight_attempts_per_scenario_with_run_seed_zero_is_the_coverage_of_the_plan(follows_the_new_rule):
    assert (second_desk.ATTEMPTS_PER_SCENARIO, second_desk.RUN_SEED) == (8, 0)
    assert second_desk.plan_coverage_problem(follows_the_new_rule["differ"], "differ") is None
    # A harness error is an attempt that was made: it counts.
    lost = copy.deepcopy(follows_the_new_rule["differ"])
    lost[0]["harness_error"] = "lost"
    assert second_desk.plan_coverage_problem(lost, "differ") is None


def test_a_scenario_with_fewer_or_more_than_eight_attempts_is_not_the_coverage_of_the_plan(
        follows_the_new_rule):
    records = follows_the_new_rule["differ"]
    problem = second_desk.plan_coverage_problem(records[1:], "differ")
    assert problem is not None and "differ: 1 of 44 scenarios do not have exactly 8 attempts" in problem
    assert f"{records[0]['scenario_id']} has 7" in problem
    one_more = records + [{**records[0], "sample": 8}]
    assert f"{records[0]['scenario_id']} has 9" in second_desk.plan_coverage_problem(one_more, "differ")
    fewer = [record for record in records if record["sample"] < 2]
    assert "44 of 44 scenarios do not have exactly 8 attempts" in second_desk.plan_coverage_problem(fewer,
                                                                                                    "differ")


def test_a_record_with_another_run_seed_is_not_the_coverage_of_the_plan(follows_the_new_rule):
    records = copy.deepcopy(follows_the_new_rule["control"])
    records[5]["seed"]["run_seed"] = 7
    problem = second_desk.plan_coverage_problem(records, "control")
    assert problem is not None and "control: the plan names the run seed 0" in problem
    assert "run seeds 0, 7" in problem
    for record in records:
        record["seed"] = None                       # a record that names no seed does not name seed 0
    assert "run seeds None" in second_desk.plan_coverage_problem(records, "control")


# ---------------------------------------------------------------------------
# What is refused: one file
# ---------------------------------------------------------------------------
def test_the_records_of_a_rollout_on_its_own_list_are_accepted(dev, lists, follows_the_new_rule):
    for part in second_desk.PARTS:
        assert second_desk.part_problem(follows_the_new_rule[part], part, lists[part], dev) is None


def test_no_record_is_refused(dev, lists):
    assert "no record" in second_desk.part_problem([], "differ", lists["differ"], dev)


@pytest.mark.parametrize("key,value", [("ruleset", "default"), ("prompt", "final"), ("split", "test"),
                                       ("toolset", "six")])
def test_a_record_of_another_split_rule_set_prompt_or_tool_set_is_refused(dev, lists, follows_the_new_rule,
                                                                         key, value):
    records = copy.deepcopy(follows_the_new_rule["differ"])
    records[3][key] = value
    problem = second_desk.part_problem(records, "differ", lists["differ"], dev)
    assert problem is not None and f"{key} {value}" in problem


def test_a_record_of_a_family_outside_the_list_is_refused(dev, lists, follows_the_new_rule):
    mixed = follows_the_new_rule["differ"] + follows_the_new_rule["control"][:1]
    problem = second_desk.part_problem(mixed, "differ", lists["differ"], dev)
    assert problem is not None and "not in the list policy_shift.differ" in problem
    # The same record is at home in the other list.
    assert lists["control"][0] in problem


def test_a_scenario_of_the_list_without_a_record_is_refused(dev, lists, follows_the_new_rule):
    gone = follows_the_new_rule["differ"][0]["scenario_id"]
    records = [record for record in follows_the_new_rule["differ"] if record["scenario_id"] != gone]
    problem = second_desk.part_problem(records, "differ", lists["differ"], dev)
    assert problem is not None and "have no record" in problem and gone in problem


def test_an_attempt_that_is_in_the_file_twice_is_refused(dev, lists, follows_the_new_rule):
    records = follows_the_new_rule["differ"] + follows_the_new_rule["differ"][:1]
    problem = second_desk.part_problem(records, "differ", lists["differ"], dev)
    assert problem is not None and "more than once" in problem


def test_a_saved_pass_that_the_verifier_does_not_give_is_refused(dev, lists, follows_the_old_rule):
    records = copy.deepcopy(follows_the_old_rule["differ"])
    assert records[0]["passed"] is False
    records[0]["passed"] = True
    problem = second_desk.part_problem(records, "differ", lists["differ"], dev)
    assert problem is not None and "does not give the saved pass or fail" in problem


def test_a_harness_error_is_not_graded_again(dev, lists, follows_the_old_rule):
    records = copy.deepcopy(follows_the_old_rule["differ"])
    records[0]["passed"], records[0]["harness_error"] = True, "lost"
    assert second_desk.part_problem(records, "differ", lists["differ"], dev) is None


# ---------------------------------------------------------------------------
# What is refused: the arms together
# ---------------------------------------------------------------------------
def test_arms_with_the_same_attempts_can_stand_in_one_table():
    attempts = {("dv-000/s1", 0), ("dv-000/s1", 1)}
    assert second_desk.coverage_problem({"base": set(attempts), "sft": set(attempts)}, "differ") is None
    assert second_desk.coverage_problem({"base": set(attempts)}, "differ") is None


def test_arms_that_do_not_cover_the_same_attempts_are_refused():
    base = {("dv-000/s1", 0), ("dv-000/s1", 1)}
    problem = second_desk.coverage_problem({"base": base, "sft": {("dv-000/s1", 0)}}, "differ")
    assert problem is not None and "base and sft do not cover the same attempts" in problem
    assert "1 only in base" in problem and "0 only in sft" in problem


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------
def write_arm(folder: Path, name: str, by_part: dict[str, list[dict]]) -> list[str]:
    """Write the two episodes files of one arm; return the three values of --arm."""
    values = [name]
    for part in second_desk.PARTS:
        path = folder / f"{name}_{part}.jsonl"
        write_records(path, by_part[part])
        values.append(str(path))
    return values


def run(report_script, capsys, scenarios, text, out: Path, *arms: list[str],
        options: tuple[str, ...] = ()) -> tuple[int, str, str]:
    """Run the script: (exit code, stdout, stderr)."""
    argv = ["--scenarios", str(scenarios), "--text", str(text.path), "--out", str(out), "--draws", "200",
            *options]
    for arm in arms:
        argv += ["--arm", *arm]
    code = report_script.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_script_writes_the_table_of_two_arms(report_script, capsys, tmp_path, scenarios, text, lists,
                                                 follows_the_new_rule, follows_the_old_rule):
    new = write_arm(tmp_path, "new_rule", follows_the_new_rule)
    old = write_arm(tmp_path, "old_rule", follows_the_old_rule)
    code, out, err = run(report_script, capsys, scenarios, text, tmp_path / "out", new, old)
    assert (code, err) == (0, "")
    report = json.loads((tmp_path / "out" / "second_desk.json").read_text(encoding="utf-8"))
    assert [arm["arm"] for arm in report["arms"]] == ["new_rule", "old_rule"]
    assert report["families"] == lists
    method = report["interval"]["method"]
    assert report["interval"] == {"level": 0.95, "n_draws": 200, "seed": 0, "method": method}
    new_row, old_row = report["arms"]
    assert new_row["differ"]["pass_rate"]["share"] == 1.0
    assert old_row["differ"]["pass_rate"]["share"] == 0.0
    assert old_row["differ"]["still_gives_the_role"]["share"] == 1.0
    assert old_row["control"]["pass_rate"]["share"] == 1.0
    n_differ, n_control = len(follows_the_old_rule["differ"]), len(follows_the_old_rule["control"])
    markdown = (tmp_path / "out" / "second_desk.md").read_text(encoding="utf-8")
    assert markdown in out
    families = f"{len(lists['differ'])} families"
    all_of_them = f"({n_differ} of {n_differ}; {families})"
    assert (f"| old_rule | 0.0% [0.0%, 0.0%] (0 of {n_differ}; {families}) | 100.0% [100.0%, 100.0%] "
            f"{all_of_them} | 100.0% [100.0%, 100.0%] ({n_control} of {n_control}; "
            f"{len(lists['control'])} families) | 0 |") in markdown
    assert "No family dropped out: every family has a valid attempt in every arm." in markdown
    assert "--allow-other-coverage" not in markdown
    assert report["allow_other_coverage"] is False and report["other_coverage"] == []


def refused(report_script, capsys, scenarios, text, out: Path, *arms: list[str],
            options: tuple[str, ...] = ()) -> str:
    """Run the script on input that it must refuse: exit code 2, nothing written; returns stderr."""
    code, _, err = run(report_script, capsys, scenarios, text, out, *arms, options=options)
    assert code == 2
    assert not out.exists()
    return err


def test_the_script_refuses_the_two_files_of_an_arm_in_the_wrong_order(report_script, capsys, tmp_path,
                                                                     scenarios, text, follows_the_new_rule):
    name, differ, control = write_arm(tmp_path, "arm", follows_the_new_rule)
    err = refused(report_script, capsys, scenarios, text, tmp_path / "out", [name, control, differ])
    assert "not in the list policy_shift.differ" in err


def test_the_script_refuses_records_of_the_default_rule_set(report_script, capsys, tmp_path, scenarios, text,
                                                            dev, lists, cfg, data_sha256):
    policy = policies.oracle(text)
    by_part = {part: play(policy, lists[part], dev, text, cfg, data_sha256, ruleset="default", prompt="final")
               for part in second_desk.PARTS}
    arm = write_arm(tmp_path, "arm", by_part)
    err = refused(report_script, capsys, scenarios, text, tmp_path / "out", arm)
    assert "ruleset all_grants_need_approval" in err and "records of ruleset default" in err


def test_the_script_refuses_arms_that_do_not_cover_the_same_attempts(report_script, capsys, tmp_path,
                                                                    scenarios, text, follows_the_new_rule):
    fewer = {part: [record for record in records if record["sample"] == 0]
             for part, records in follows_the_new_rule.items()}
    # Even where another coverage is allowed, the arms of one table must hold the same attempts.
    err = refused(report_script, capsys, scenarios, text, tmp_path / "out",
                  write_arm(tmp_path, "all", follows_the_new_rule), write_arm(tmp_path, "fewer", fewer),
                  options=("--allow-other-coverage",))
    assert "all and fewer do not cover the same attempts" in err


def test_the_script_refuses_an_arm_with_a_missing_record_unless_another_coverage_is_allowed(
        report_script, capsys, tmp_path, scenarios, text, follows_the_new_rule):
    # One arm alone: no other arm is there to compare the attempts with.
    short = {"differ": follows_the_new_rule["differ"][1:], "control": follows_the_new_rule["control"]}
    arm = write_arm(tmp_path, "short", short)
    err = refused(report_script, capsys, scenarios, text, tmp_path / "out", arm)
    assert "short, differ: 1 of 44 scenarios do not have exactly 8 attempts" in err
    assert "--allow-other-coverage" in err
    code, out, err = run(report_script, capsys, scenarios, text, tmp_path / "out", arm,
                         options=("--allow-other-coverage",))
    assert (code, err) == (0, "")
    lines = (tmp_path / "out" / "second_desk.md").read_text(encoding="utf-8").splitlines()
    own_line = [line for line in lines if line.startswith("NOT THE COVERAGE OF THE PLAN")]
    assert len(own_line) == 1 and "--allow-other-coverage was given" in own_line[0]
    assert "short, differ: 1 of 44 scenarios do not have exactly 8 attempts" in own_line[0]
    report = json.loads((tmp_path / "out" / "second_desk.json").read_text(encoding="utf-8"))
    assert report["allow_other_coverage"] is True and len(report["other_coverage"]) == 1
    assert report["arms"][0]["differ"]["attempts_per_scenario"] == [7, 8]


def test_the_script_refuses_another_run_seed_unless_another_coverage_is_allowed(
        report_script, capsys, tmp_path, scenarios, text, follows_the_new_rule):
    other_seed = copy.deepcopy(follows_the_new_rule)
    for record in other_seed["control"]:
        record["seed"]["run_seed"] = 7
    arm = write_arm(tmp_path, "seed7", other_seed)
    err = refused(report_script, capsys, scenarios, text, tmp_path / "out", arm)
    assert "seed7, control: the plan names the run seed 0" in err
    code, out, _ = run(report_script, capsys, scenarios, text, tmp_path / "out", arm,
                       options=("--allow-other-coverage",))
    assert code == 0 and "NOT THE COVERAGE OF THE PLAN" in out and "run seeds 7" in out


def test_the_table_says_it_when_another_coverage_was_allowed_and_not_needed(
        report_script, capsys, tmp_path, scenarios, text, follows_the_new_rule):
    code, out, _ = run(report_script, capsys, scenarios, text, tmp_path / "out",
                       write_arm(tmp_path, "arm", follows_the_new_rule), options=("--allow-other-coverage",))
    assert code == 0 and "NOT THE COVERAGE OF THE PLAN" not in out
    assert ("--allow-other-coverage was given and was not needed: every arm has 8 attempts per scenario "
            "and the run seed 0.") in out.splitlines()


def test_the_table_names_a_family_that_dropped_out_and_shows_the_families_behind_each_mean(
        report_script, capsys, tmp_path, scenarios, text, lists, follows_the_new_rule):
    lost = copy.deepcopy(follows_the_new_rule)
    gone = lists["differ"][0]
    for record in lost["differ"]:
        if record["family_id"] == gone:
            record["harness_error"] = "lost"
    code, out, err = run(report_script, capsys, scenarios, text, tmp_path / "out",
                         write_arm(tmp_path, "lossy", lost))
    assert (code, err) == (0, "")
    n_differ = len(lists["differ"])
    assert f"; {n_differ - 1} families) |" in out and f"; {len(lists['control'])} families) |" in out
    assert (f"- lossy: on \"differ\" 1 of {n_differ} families dropped out of every mean, because all its "
            f"attempts are harness errors: {gone}.") in out.splitlines()
    assert "No family dropped out" not in out


def test_the_script_refuses_an_arm_name_that_is_given_twice(report_script, capsys, tmp_path, scenarios, text,
                                                           follows_the_new_rule):
    arm = write_arm(tmp_path, "arm", follows_the_new_rule)
    assert "give each arm once" in refused(report_script, capsys, scenarios, text, tmp_path / "out", arm, arm)


def test_the_script_refuses_records_of_another_scenario_file(report_script, capsys, tmp_path, scenarios, text,
                                                            follows_the_new_rule):
    changed = copy.deepcopy(follows_the_new_rule)
    for record in changed["differ"]:
        record["data_sha256"] = "0" * 64
    arm = write_arm(tmp_path, "arm", changed)
    err = refused(report_script, capsys, scenarios, text, tmp_path / "out", arm)
    assert "is not the file that the run was made from" in err


def test_the_script_refuses_a_sets_file_that_was_changed_after_the_build(
        report_script, capsys, tmp_path, scenarios, text, follows_the_new_rule):
    copied = tmp_path / "scenarios"
    copied.mkdir()
    for path in scenarios.iterdir():
        (copied / path.name).write_bytes(path.read_bytes())
    sets = json.loads((copied / "sets.json").read_text(encoding="utf-8"))
    sets["policy_shift"]["differ"] = sets["policy_shift"]["differ"][:-1]
    (copied / "sets.json").write_text(json.dumps(sets), encoding="utf-8")
    err = refused(report_script, capsys, copied, text, tmp_path / "out",
                  write_arm(tmp_path, "arm", follows_the_new_rule))
    assert "sets.json is not the file that the build wrote" in err
