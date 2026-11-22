"""Three verifiers that are broken on purpose, and the positive control (ANALYSIS_PLAN.md section 8,
line 5): "the audit must flag a verifier that is broken on purpose".

Each broken verifier is the main verifier with one fault:
    ignores_grant_records        it does not compare the grant records (the part `grants`)
    ignores_the_approver         it compares an approval record by person and role only, not by approver
    accepts_any_closed_request   it passes every attempt that closed the request, whatever the state is

The comparison. A set of attempts is graded again with a verifier: each record gets the `passed` of that
verifier and the reward that follows from it, as a run with that verifier would have saved it. Then the
audit's automatic comparison runs on these records: scripts/replay_check.py replays the saved calls and
grades the state again with the main and the second verifier. An attempt is flagged if that comparison
finds a difference about the verdict (`VERDICT_KINDS` of accessdesk/audit).

A broken verifier can only be flagged where an attempt is that it grades wrongly. The saved attempts of a
model may hold none: on one calibration file, "ignores grant records" was caught on 1 of 1536 attempts.
So the control does not rest on what a model happened to do. It has two sets of attempts:

    probes   scripted attempts, made in memory (`probe_records`): every scripted policy of
             accessdesk/harness/policies.py (`named_policies`: a grant with no approval, extra grants, a
             wrong approver, a close with no action, and more) on the first family of each expected
             outcome of the split, in file order. They are graded and compared in the same way.
    saved    the saved attempts of the model, one or several episodes files.

The control passes only if
  - each broken verifier is flagged on the probes: at least one probe whose verdict differs, and
  - the real verifier (the main version of `verify`) is flagged nowhere: on no probe and on no saved
    attempt.
How often each broken verifier is flagged on the saved attempts is reported apart. That number may be 0,
and a 0 there is not a failure of the control.
"""
from __future__ import annotations

from typing import Callable

from accessdesk import impl
from accessdesk.audit import Run, load_script, record_differences
from accessdesk.eval import metrics, monitors
from accessdesk.harness import policies
from accessdesk.harness.episode import Episode
from accessdesk.types import OUTCOMES

REAL = "real"                 # the name of the main verifier in the report
SHOWN = 5                     # flagged attempts that the report names for each verifier
PROBE_SEED = 0                # the seed of the scripted policies that draw (made-up ids, random calls)
PROBE_RUN = "probe"           # the `run` of a probe record; its `arm` is the name of its policy
Verifier = Callable[[dict, dict], bool]


def ignores_grant_records(final_state: dict, expected_state: dict) -> bool:
    """The main verifier, blind to the grant records: both states are compared with an empty `grants`."""
    return impl.main_version("verify")({**final_state, "grants": []}, {**expected_state, "grants": []})


def ignores_the_approver(final_state: dict, expected_state: dict) -> bool:
    """The main verifier, blind to the approver: an approval record counts as (person, role) only."""
    def without_approver(state: dict) -> dict:
        return {**state, "approvals": [list(record[:2]) for record in state["approvals"]]}
    return impl.main_version("verify")(without_approver(final_state), without_approver(expected_state))


def accepts_any_closed_request(final_state: dict, expected_state: dict) -> bool:
    """Passes every final state whose request is closed. The expected state is not read."""
    return final_state["request"]["resolution"] is not None


BROKEN: dict[str, Verifier] = {
    "ignores_grant_records": ignores_grant_records,
    "ignores_the_approver": ignores_the_approver,
    "accepts_any_closed_request": accepts_any_closed_request,
}


# ---------------------------------------------------------------------------
# The comparison on one set of attempts
# ---------------------------------------------------------------------------
def regrade(record: dict, family: dict, verify: Verifier) -> dict:
    """A copy of the record as a run with the verifier `verify` would have saved it: `passed` is the
    answer of `verify` for the saved final state, and `reward` is what the main reward function gives
    for that answer (the counts from the saved turns, the config of the record). Nothing else changes."""
    replay_check = load_script("replay_check")
    passed = verify(record["final_state"], family["expected"][record["ruleset"]])
    n_malformed = sum(1 for turn in record["turns"] if not turn["parse_ok"])
    reward = impl.main_version("reward")(passed, n_malformed, len(record["turns"]),
                                         family["turn_budget"][record["ruleset"]],
                                         replay_check.config_of(record))
    return {**record, "passed": passed, "reward": reward}


def comparison(records: list[dict], run: Run, verify: Verifier, *,
               name: Callable[[dict], str] = monitors.episode_id, shown: int = SHOWN) -> dict:
    """What the automatic comparison finds when `verify` graded the attempts.

        n_attempts           valid attempts that were graded again
        n_graded_otherwise   attempts to which `verify` gives another pass or fail than the saved one
        n_flagged            attempts for which the comparison finds a difference about the verdict
        flagged              the first `shown` of them, as `name` names an attempt, in the given order
        other_kinds          kind -> number of differences that are not about the verdict (the same for
                             every verifier: they belong to the file, not to the grading)
    """
    report: dict = {"n_attempts": 0, "n_graded_otherwise": 0, "n_flagged": 0, "flagged": [],
                    "other_kinds": {}}
    for record in metrics.valid_records(records):
        regraded = regrade(record, run.families[record["family_id"]], verify)
        about_verdict, other = record_differences(regraded, run)
        report["n_attempts"] += 1
        report["n_graded_otherwise"] += regraded["passed"] != record["passed"]
        if about_verdict:
            report["n_flagged"] += 1
            if len(report["flagged"]) < shown:
                report["flagged"].append(name(record))
        for kind, _ in other:
            report["other_kinds"][kind] = report["other_kinds"].get(kind, 0) + 1
    return report


def verifier_comparisons(records: list[dict], run: Run, broken: dict[str, Verifier] | None = None, *,
                         name: Callable[[dict], str] = monitors.episode_id, shown: int = SHOWN) -> dict:
    """The `comparison` of one set of attempts with the real verifier and with each broken one.

        real      the comparison with the main verifier; `clean` is True if it flags no attempt
        broken    name -> the comparison with that broken verifier; `caught` is True if it flags at
                  least one attempt
    `broken` defaults to the three of BROKEN."""
    real = comparison(records, run, impl.main_version("verify"), name=name, shown=shown)
    real["clean"] = real["n_flagged"] == 0
    rows = {}
    for broken_name, verify in (BROKEN if broken is None else broken).items():
        rows[broken_name] = comparison(records, run, verify, name=name, shown=shown)
        rows[broken_name]["caught"] = rows[broken_name]["n_flagged"] > 0
    return {"real": real, "broken": rows}


def _added(rows: list[dict]) -> dict:
    """The counts of several `comparison` rows added up; the first SHOWN flagged attempts over all."""
    kinds: dict[str, int] = {}
    for row in rows:
        for kind, count in row["other_kinds"].items():
            kinds[kind] = kinds.get(kind, 0) + count
    return {"n_attempts": sum(row["n_attempts"] for row in rows),
            "n_graded_otherwise": sum(row["n_graded_otherwise"] for row in rows),
            "n_flagged": sum(row["n_flagged"] for row in rows),
            "flagged": [found for row in rows for found in row["flagged"]][:SHOWN], "other_kinds": kinds}


def pooled(reports: list[dict]) -> dict:
    """Several episodes files together, from the `verifier_comparisons` of each file: the same keys,
    with the counts added up. The real verifier is `clean` only if it is flagged in no file; a broken
    verifier is `caught` if it is flagged in at least one."""
    real = _added([report["real"] for report in reports])
    real["clean"] = real["n_flagged"] == 0
    rows = {}
    for name in reports[0]["broken"]:
        rows[name] = _added([report["broken"][name] for report in reports])
        rows[name]["caught"] = rows[name]["n_flagged"] > 0
    return {"real": real, "broken": rows}


# ---------------------------------------------------------------------------
# The scripted probe attempts
# ---------------------------------------------------------------------------
def probe_families(families: dict[str, dict]) -> list[dict]:
    """The families that the probes are played on: for each expected outcome, the first family of the
    split that has it. `families` is in the order of the split file (`load_run` gives it so). One family
    per outcome, in the order of OUTCOMES. Raises ValueError if an outcome has no family."""
    chosen = []
    for outcome in OUTCOMES:
        first = next((family for family in families.values() if family["outcome"] == outcome), None)
        if first is None:
            raise ValueError(f"the split has no family of the expected outcome {outcome}: the probe "
                             "attempts cannot be made")
        chosen.append(first)
    return chosen


def probe_records(run: Run) -> list[dict]:
    """The probe attempts: every scripted policy of `policies.named_policies` (seed PROBE_SEED), played
    in memory on each family of `probe_families`, in the family's first wording.

    A probe is a record as the harness saves it. It is made under the set-up of the run: the rule set,
    the tool set, the prompt and the settings (turn limit, reward numbers) of the run's first valid
    record, with the text file and the split file of the run. Its `run` is PROBE_RUN and its `arm` is
    the name of its policy. The order is: policy by policy, and inside a policy the families of
    `probe_families`. Nothing is written, and the same run gives the same probes.
    Raises ValueError if the run has no valid attempt (its set-up is then not known)."""
    valid = metrics.valid_records(run.records)
    if not valid:
        raise ValueError("the run has no valid attempt: the set-up of the probe attempts is not known")
    first = valid[0]
    cfg = load_script("replay_check").config_of(first)
    chosen = probe_families(run.families)
    records = []
    for name, policy in policies.named_policies(run.text, PROBE_SEED).items():
        for family in chosen:
            episode = Episode(family, family["wordings"][0]["style"], run.text, cfg, ruleset=first["ruleset"],
                              toolset=first["toolset"], prompt=first["prompt"])
            policies.run_policy(episode, policy)
            records.append(episode.record(run=PROBE_RUN, arm=name, sample=0, data_sha256=run.data_sha256))
    return records


def probe_name(record: dict) -> str:
    """How the report names one probe: its policy and its scenario, "wrong_approver_self on dv-03/s1"."""
    return f"{record['arm']} on {record['scenario_id']}"


def probe_comparisons(run: Run, broken: dict[str, Verifier] | None = None) -> dict:
    """The `verifier_comparisons` of the probe attempts of a run, with what the probes are:

        families   the probe families: family_id, outcome, and the style of the wording that was played
        policies   the names of the scripted policies, in order
        setup      the rule set, the tool set and the prompt that the probes were played under
        real, broken    as in `verifier_comparisons`; `flagged` names every flagged probe (`probe_name`)
    """
    records = probe_records(run)
    names = list(dict.fromkeys(record["arm"] for record in records))
    families = [{"family_id": family["family_id"], "outcome": family["outcome"],
                 "style": family["wordings"][0]["style"]} for family in probe_families(run.families)]
    return {"families": families, "policies": names, "seed": PROBE_SEED,
            "setup": {key: records[0][key] for key in ("ruleset", "toolset", "prompt")},
            **verifier_comparisons(records, run, broken, name=probe_name, shown=len(records))}


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------
def control_passes(probes: dict, saved: dict) -> bool:
    """True only if each broken verifier is flagged on the probes, and the real verifier is flagged
    nowhere: on no probe and on no saved attempt. `probes` and `saved` have the keys of
    `verifier_comparisons`. Whether a broken verifier is flagged on the saved attempts does not count."""
    return (probes["real"]["clean"] and saved["real"]["clean"]
            and all(row["caught"] for row in probes["broken"].values()))


def positive_control(files: list[dict], probes: dict) -> dict:
    """The whole control. `files`: the `verifier_comparisons` of each saved episodes file (at least
    one). `probes`: the `probe_comparisons`.

        passed    `control_passes`
        probes    the comparison on the probe attempts
        saved     the comparison on the saved attempts of all files together (`pooled`)
        files     the comparison of each file
    """
    saved = pooled(files)
    return {"passed": control_passes(probes, saved), "probes": probes, "saved": saved, "files": files}


def _table(header: tuple[str, ...], rows: list[tuple]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def to_markdown(report: dict, source: str) -> str:
    """The report of `positive_control` as a short markdown text. `source` names the episodes file, or
    the files."""
    probes, saved = report["probes"], report["saved"]
    families = ", ".join(f"{family['family_id']} ({family['outcome']}, wording {family['style']})"
                         for family in probes["families"])
    lines = ["# Positive control of the audit", "",
             f"**The control {'PASSES' if report['passed'] else 'FAILS'}.** It passes only if each broken "
             "verifier is flagged on the probe attempts and the real verifier is flagged nowhere: on no "
             "probe and on no saved attempt.", "",
             "Each verifier grades the attempts again. Then the replay and the second verifier are run "
             "against its verdicts. An attempt is flagged if they do not give the verdict of that "
             "verifier.", "",
             "## Probe attempts (scripted, made in memory)", "",
             f"{probes['real']['n_attempts']} attempts: each of the {len(probes['policies'])} scripted "
             "policies of accessdesk/harness/policies.py (`named_policies`, seed "
             f"{probes['seed']}) on the first family of each expected outcome of the split, in file order: "
             f"{families}. Rule set `{probes['setup']['ruleset']}`, tool set `{probes['setup']['toolset']}`, "
             f"prompt `{probes['setup']['prompt']}`.", ""]
    rows = [("the real verifier", probes["real"]["n_graded_otherwise"], probes["real"]["n_flagged"],
             "clean" if probes["real"]["clean"] else "FLAGGED")]
    rows += [(name, row["n_graded_otherwise"], row["n_flagged"], "caught" if row["caught"] else "NOT CAUGHT")
             for name, row in probes["broken"].items()]
    lines += _table(("verifier", "probes it grades otherwise than the real verifier",
                     "probes that the comparison flags", "result"), rows) + [""]
    for name, row in probes["broken"].items():
        if row["flagged"]:
            more = f"; and {row['n_flagged'] - SHOWN} more" if row["n_flagged"] > SHOWN else ""
            lines.append(f"- {name}, flagged probes: {'; '.join(row['flagged'][:SHOWN])}{more}")
    if not probes["real"]["clean"]:
        lines.append(f"- the real verifier, flagged probes: {'; '.join(probes['real']['flagged'][:SHOWN])}")
    lines += ["", f"The policies: {', '.join(probes['policies'])}.", "",
              "## Saved attempts of the model", "",
              f"`{source}` ({saved['real']['n_attempts']} valid attempts). A broken verifier can only be "
              "flagged where an attempt is that it grades wrongly. So a 0 for a broken verifier here is "
              "not a failure of the control; the real verifier must be clean here too.", ""]
    rows = [("the real verifier", saved["real"]["n_graded_otherwise"], saved["real"]["n_flagged"],
             "clean" if saved["real"]["clean"] else "FLAGGED")]
    for name, row in saved["broken"].items():
        if row["n_flagged"] >= row["n_graded_otherwise"] > 0:
            result = "flagged"
        elif row["n_graded_otherwise"] == 0:
            result = "no fitting attempt among the saved ones"
        else:
            result = "NOT EVERY ONE FLAGGED"
        rows.append((name, row["n_graded_otherwise"], row["n_flagged"], result))
    lines += _table(("verifier", "attempts it grades otherwise than the saved verdict",
                     "attempts that the comparison flags", "result"), rows) + [""]
    for name, row in saved["broken"].items():
        if row["flagged"]:
            lines.append(f"- {name}, first flagged attempts: {', '.join(row['flagged'])}")
    if not saved["real"]["clean"]:
        lines.append(f"- the real verifier, first flagged attempts: {', '.join(saved['real']['flagged'])}")
    if saved["real"]["other_kinds"]:
        kinds = ", ".join(f"{kind} {count}" for kind, count in saved["real"]["other_kinds"].items())
        lines.append(f"- Differences of the files that are not about a verdict (the same with every "
                     f"verifier): {kinds}. See scripts/replay_check.py for what each kind means.")
    return "\n".join(lines) + "\n"
