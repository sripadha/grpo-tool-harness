"""The chain script of the test run (scripts/gpu/chain_test.sh), without torch, without a model and without
the test split.

How the chain is run here. The script has no variable for its folders: it works in the repository that it
lies in. So every run of these tests starts a COPY of the script in a small made-up repository in a
temporary folder (`Tree`): a few hand-made families under data/scenarios/, a made-up text file, made-up
adapters, and the records of results/main with the key names of the real ones. The made-up tree is frozen
with the real accessdesk/freeze.py. The test split of the real repository is never opened, hashed or
counted here, and nothing is written into the real results/.

A stand-in for `python` runs the chain's own snippets and scripts/freeze.py for real (a copy of the real
script, on the made-up tree). It replaces every other script: the call goes into a call log, and the
stand-in leaves what the chain reads afterwards, with the key names of the real script. So these tests show
what the chain starts, in which order and with which options, and where it stops. They do not show that a
rollout, a table or a report works.

What is tested:
    the text of the script    `bash -n`; the scripts and options that it names; the helper functions that it
                              shares with the stage chains; which variables it reads; the names that it
                              compares in a run manifest
    the test run (DRESS=0)    the order of the commands, what each is given, the record of the run
    a rehearsal (DRESS=1)     the same commands on calibration families, and nothing written in results/main
    run once                  a complete folder is skipped and never run again; a folder that is not complete
                              stops the chain; REPEAT_AFTER_FAULT with a reason notes the reason first and
                              then moves the folder aside, and without one it is refused; an arm whose
                              rollout the record names and whose folder is gone needs a reason too
    one chain at a time       a second start stops at the lock while a chain or its rollout runs
    the record                a line before each rollout and after it, the gate, a repeat, a changed adapter;
                              nothing that the record must name happens if its line cannot be written
    the stops                 a tree that is not as frozen; an adapter or a record that is not the one of the
                              dev run, before, right before a rollout, right after it and at the end; a
                              complete folder without the note of the chain; the harness-error gate at 1%;
                              reference rows that are not as the plan says; the first step that fails
    the file of the machine   /workspace/env.sh cannot choose the mode or ask for a repeat
Four tests read results and data of the real repository (never its test split): the command of each dev
run, the real records of results/main with the real adapters, the reference rows of dev, and the
calibration families of a rehearsal. They are skipped where these files are not on the machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from accessdesk import freeze
from accessdesk.config import RunConfig
from accessdesk.eval import tables
from tests.helpers.make_records import scenario_records

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "gpu" / "chain_test.sh"
STAGE_CHAINS = (SCRIPTS / "gpu" / "chain_stage1.sh", SCRIPTS / "gpu" / "chain_stage2.sh")
# The values of ANALYSIS_PLAN.md section 2 and of the task, typed by hand.
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ARMS = ("base", "sft", "grpo_seed1", "grpo_seed2")
ADAPTERS = {"base": None, "sft": "results/main/sft/adapter",
            "grpo_seed1": "results/main/grpo_seed1/checkpoints/step_00080/adapter",
            "grpo_seed2": "results/main/grpo_seed2/checkpoints/step_00080/adapter"}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
POLICIES = ("oracle", "constant_granted", "constant_pending", "constant_already", "constant_denied")
DRESS_OUT = "results/_smoke/test_dress"
DRESS_FAMILIES = ["ca-00", "ca-01", "ca-02", "ca-03"]
TIME = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def put(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def files_of(folder: Path) -> dict[str, bytes]:
    """Every file under the folder: its path from the folder -> its bytes."""
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


# ---------------------------------------------------------------------------
# The text of the script
# ---------------------------------------------------------------------------
OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
SOURCES = {"rollout": SCRIPTS / "gpu" / "rollout.py", "tables": ROOT / "accessdesk" / "eval" / "tables.py",
           "replay_check": SCRIPTS / "replay_check.py", "monitor_report": SCRIPTS / "monitor_report.py",
           "run_scripted": SCRIPTS / "run_scripted.py"}
# The helper functions that the script has from the stage chains, and those that it must not have: they
# move an unfinished folder aside without being told to, or let a chain go on after a failed step.
SHARED = ("json_value", "manifest_value", "is_complete", "file_sha256", "chain_lines", "fits", "timed",
          "run_command", "write_to", "rollout_items")
NOT_TAKEN = ("move_aside", "skip_if_done", "run_step", "run_report", "end_chain")
# Every double-dash word of the script outside its comments. Typed by hand. `rollout_items`, a helper of the
# stage chains, asks whether an option is --set; this chain gives --set to no rollout (the runs below show
# it).
OPTIONS_USED = {"--model", "--revision", "--adapter", "--text", "--scenarios", "--split", "--samples",
                "--run-seed", "--max-new-tokens", "--arm", "--out", "--families",      # rollout.py
                "--set",                                                               # only asked for
                "--episodes",                                                          # the two reports
                "--baseline", "--pair", "--out-json", "--out-md",                      # the table
                "--policy"}                                                            # run_scripted.py


def options_of(script: str) -> set[str]:
    """The options that a Python script has, read from its source (the first argument of add_argument)."""
    source = SOURCES[script].read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\(\s*\"(--[a-z0-9-]+)\"", source))


def chain_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def chain_body() -> str:
    """The script without its comment lines."""
    return "\n".join(line for line in chain_text().splitlines() if not line.lstrip().startswith("#"))


def function_text(path: Path, name: str) -> str:
    """The text of one shell function of a file: from `name() {` to the `}` at the start of a line."""
    match = re.search(rf"^{name}\(\) \{{.*?^\}}$", path.read_text(encoding="utf-8"),
                      flags=re.DOTALL | re.MULTILINE)
    assert match, (path.name, name)
    return match.group(0)


def test_the_script_is_valid_bash():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_script_names_only_scripts_that_exist_and_options_that_they_have():
    """What each single call is given is checked on the calls themselves, in every run below (`Tree.run`)."""
    body = chain_body()
    named = set(re.findall(r"scripts/(?:gpu/)?\w+\.py", body))
    assert named == {"scripts/freeze.py", "scripts/gpu/rollout.py", "scripts/replay_check.py",
                     "scripts/monitor_report.py", "scripts/run_scripted.py"}
    assert all((ROOT / path).is_file() for path in named) and "-m accessdesk.eval.tables" in body
    assert 'commands.add_parser("check"' in (SCRIPTS / "freeze.py").read_text(encoding="utf-8")
    known = set().union(*(options_of(script) for script in SOURCES))
    assert set(OPTION.findall(body)) == OPTIONS_USED and OPTIONS_USED <= known
    # The option that lets a run start on a changed tree is nowhere in the file, not even in a comment.
    assert all("--allow-changed-tree" in options_of(script) for script in ("rollout", "run_scripted"))
    assert "allow-changed-tree" not in chain_text()
    # Both splits of the chain are splits that the scripts take.
    from accessdesk.env.generator import SPLITS
    assert {"test", "calib"} <= set(SPLITS)
    assert 'SPLITS = ("train", "dev", "test", "calib")' in SOURCES["rollout"].read_text(encoding="utf-8")


def test_the_helpers_of_the_stage_chains_have_their_text_and_the_ones_that_repeat_a_step_are_left_out():
    """`is_complete` is the test of completeness of the other chains, and `fits` their rule for a skip. A
    change in one file only would give the test run other rules than the dev runs had."""
    for name in SHARED:
        texts = {function_text(path, name) for path in (SCRIPT, *STAGE_CHAINS)}
        assert len(texts) == 1, name
    for name in NOT_TAKEN:
        assert function_text(STAGE_CHAINS[1], name)                       # the stage chains have it
        assert not re.search(rf"^{name}\(\)", chain_text(), flags=re.MULTILINE), name
    # The command of a rollout is the one of the stage chains, word for word: the same options in the same
    # order. Only the names of four shell variables are other.
    def words(text: str) -> str:
        return " ".join(text.replace("\\\n", " ").split())

    stage = function_text(STAGE_CHAINS[1], "rollout_step")
    command = words(stage[stage.index('"$PYTHON" scripts/gpu/rollout.py'):stage.rindex("}")])
    assert command.endswith('--arm "$arm" --out "$OUT/$name" "$@"')
    for theirs, ours in (('"$OUT/$name"', '"$folder"'), ('"$split"', '"$SPLIT"'),
                         ('"$samples"', '"$SAMPLES"'), ('"$seed"', '"$EVAL_SEED"')):
        assert command.count(theirs) == 1
        command = command.replace(theirs, ours)
    assert command in words(function_text(SCRIPT, "rollout_step"))


def test_only_four_variables_come_from_the_environment_and_every_value_of_the_plan_is_fixed():
    text, body = chain_text(), chain_body()
    read = re.findall(r'^([A-Z_]+)="\$\{\1:-', text, flags=re.MULTILINE)
    assert read == ["DRESS", "REPEAT_AFTER_FAULT", "REPEAT_REASON", "PYTHON"]
    assert set(re.findall(r"\$\{([A-Z_]+):-", body)) == set(read)        # no other default from outside
    for line in ('MODEL="Qwen/Qwen3.5-2B"', f'REVISION="{REVISION}"', 'SCENARIOS="data/scenarios"',
                 'TEXT="text/model_text.yaml"', "SAMPLES=8  ", "EVAL_SEED=0  ", "MAX_NEW_TOKENS=260  ",
                 'MAIN="results/main"  ', "ARMS=(base sft grpo_seed1 grpo_seed2)",
                 "POLICIES=(oracle constant_granted constant_pending constant_already constant_denied)",
                 "DRESS_FAMILIES=(ca-00 ca-01 ca-02 ca-03)"):
        assert re.search(rf"^{re.escape(line)}", text, flags=re.MULTILINE), line
    # The split and the folder of what is written are set in one place, by DRESS.
    assert len(re.findall(r"^\s*SPLIT=", body, flags=re.MULTILINE)) == 2
    assert len(re.findall(r"^\s*OUT=", body, flags=re.MULTILINE)) == 2
    assert 'SPLIT="test"\n        OUT="$MAIN"' in body
    assert f'SPLIT="calib"\n        OUT="{DRESS_OUT}"' in body
    # The fixed values are those of the config and of the pinned revisions.
    from accessdesk.harness.hf_backend import PINNED_REVISIONS
    assert RunConfig().max_new_tokens == 260 and PINNED_REVISIONS[MODEL] == REVISION


# What the script compares in a run manifest beside the names of `rollout_items` (which the stage chains
# compare too, and tests/core/test_chain_scripts.py guards there), and what its gate and its adapter check
# read. Typed by hand from the script.
MORE_COMPARED = ("ruleset", "toolset", "prompt", "code_sha256", "allow_changed_tree")
COMPARED = ("model", "revision", "adapter", "split", "samples", "run_seed", "config/max_new_tokens",
            "text_sha256", "split_file_sha256", "family_set", *MORE_COMPARED)
GATE_READS = ("summary", "harness_error_rate", "episodes", "harness_errors")


def test_what_the_script_reads_in_a_run_manifest_is_what_the_scripts_write():
    """A name that a run manifest does not have is not compared (`fits` only says so), so a name that is
    typed wrong would switch a check off without a fault."""
    items = re.search(r"^MORE_ITEMS=\((.*?)\)$", chain_text(), flags=re.DOTALL | re.MULTILINE).group(1)
    assert tuple(re.findall(r'"([a-z_0-9]+)=', items)) == MORE_COMPARED
    assert '"allow_changed_tree=False"' in items and '"code_sha256=$CODE_SHA256"' in items
    rollout = SOURCES["rollout"].read_text(encoding="utf-8")
    for name in (*MORE_COMPARED, *GATE_READS):
        assert f'"{name}"' in rollout, name
    for name in GATE_READS:
        assert f'"{name}"' in function_text(SCRIPT, "gate") or f"'{name}'" in function_text(SCRIPT, "gate")
    # The two training scripts write the hashes of their adapter under the names that the script reads.
    reads = function_text(SCRIPT, "trained_sha256")
    assert 'outputs adapter "$2" sha256' in reads and 'final_adapter files "$2" sha256' in reads
    train_sft = (SCRIPTS / "gpu" / "train_sft.py").read_text(encoding="utf-8")
    train_grpo = (SCRIPTS / "gpu" / "train_grpo.py").read_text(encoding="utf-8")
    assert '"adapter": adapter_entries(' in train_sft and '"outputs"' in train_sft
    assert 'manifest["final_adapter"] = {"folder": str(final),' in train_grpo and '"files"' in train_grpo


# ---------------------------------------------------------------------------
# The made-up repository
# ---------------------------------------------------------------------------
def made_up_families(prefix: str, split: str, n: int, styles: tuple[int, ...] = (1, 2, 3, 4)) -> list[dict]:
    """`n` family records with the keys that the stand-in for the rollout reads."""
    outcomes = ("inactive", "grant", "approval", "already")
    return [{"family_id": f"{prefix}-{index:02d}", "split": split, "outcome": outcomes[index % 4],
             "wordings": [{"style": style, "text": f"made-up request {index} in style {style}"}
                          for style in styles]} for index in range(n)]


def adapter_entries(root: Path, folder: str) -> dict:
    """What a training script writes about the files of its adapter (`adapter_entries` of train_sft.py)."""
    return {name: {"path": f"{folder}/{name}", "sha256": sha256_of(root / folder / name),
                   "bytes": (root / folder / name).stat().st_size} for name in ADAPTER_FILES}


def dev_manifest(root: Path, arm: str, code_sha256: str) -> dict:
    """The run manifest of an arm's dev run, with the keys of scripts/gpu/rollout.py that the chain reads."""
    adapter = ADAPTERS[arm]
    command = ["scripts/gpu/rollout.py", "--model", MODEL, "--revision", REVISION,
               *(["--adapter", adapter] if adapter else []), "--text", "text/model_text.yaml",
               "--scenarios", "data/scenarios", "--split", "dev", "--samples", "8", "--run-seed", "0",
               "--max-new-tokens", "260", "--arm", arm, "--out", f"results/main/{arm}_dev"]
    return {"complete": True, "command": command, "config": {"max_new_tokens": 260},
            "text_sha256": sha256_of(root / "text" / "model_text.yaml"), "code_sha256": code_sha256,
            "allow_changed_tree": False, "model": MODEL, "revision": REVISION, "adapter": adapter,
            "split": "dev", "split_file_sha256": sha256_of(root / "data" / "scenarios" / "dev.jsonl"),
            "family_set": None, "families": 2, "scenarios": 8, "samples": 8, "run_seed": 0,
            "ruleset": "default", "toolset": "five", "prompt": "final",
            "summary": {"episodes": 64, "harness_errors": 0, "harness_error_rate": 0.0}}


def dev_note(root: Path, arm: str) -> str:
    """chain_record.txt of an arm's dev run, as `run_step` of the stage chains wrote it."""
    lines = ["scenarios=data/scenarios", f"arm={arm}", "options="]
    if ADAPTERS[arm]:
        lines += [f"{name}={sha256_of(root / ADAPTERS[arm] / name)}" for name in ADAPTER_FILES]
    return "\n".join(lines) + "\n"


def make_tree(root: Path, splits: tuple[str, ...] = ("test", "calib")) -> Path:
    """A small repository that the chain can run in.

    data/scenarios   dev.jsonl (2 families; the chain only hashes it), and of `splits`: test.jsonl (5
                     families in 5 wordings: 25 scenarios, so 200 attempts per arm, and 2 harness errors are
                     exactly 1%) and calib.jsonl (6 families in 4 wordings; a rehearsal takes 4 of them)
    MANIFEST.json    the freeze of this tree, made by the real accessdesk/freeze.py: it holds the text file
                     and the scenario files
    results/main     the three adapters, the run manifests of their training runs, and the run manifest
                     and the chain's note of each dev run
    """
    for path in ("scripts/gpu/chain_test.sh", "scripts/freeze.py"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, root / path)
    put(root / "text" / "model_text.yaml", "the text file of a made-up tree\n")
    scenarios = root / "data" / "scenarios"
    made = {"dev": made_up_families("dv", "dev", 2)}
    if "test" in splits:
        made["test"] = made_up_families("te", "test", 5, styles=(1, 2, 3, 4, 5))
    if "calib" in splits:
        made["calib"] = made_up_families("ca", "calib", 6)
    for split, families in made.items():
        put(scenarios / f"{split}.jsonl", "".join(json.dumps(family) + "\n" for family in families))
    listed = {f"{split}.jsonl": sha256_of(scenarios / f"{split}.jsonl") for split in made}
    write_json(scenarios / "summary.json", {"files_sha256": listed})
    manifest = freeze.build_manifest(root, signed_by="Test Person", frozen_at="2026-11-18T12:00:00Z")
    write_json(root / freeze.MANIFEST_NAME, manifest)
    assert freeze.check_manifest(root, read_json(root / freeze.MANIFEST_NAME)) == []
    main = root / "results" / "main"
    for arm, folder in ADAPTERS.items():
        if folder:
            put(root / folder / "adapter_config.json", "{}\n")            # the same file in every adapter
            put(root / folder / "adapter_model.safetensors", f"made-up weights of {arm}\n")
    write_json(main / "sft" / "run_manifest.json",
               {"complete": True, "status": "complete",
                "outputs": {"adapter": adapter_entries(root, ADAPTERS["sft"])}})
    for arm in ("grpo_seed1", "grpo_seed2"):
        write_json(main / arm / "run_manifest.json",
                   {"complete": True, "status": "complete", "mode": "seed",
                    "final_adapter": {"folder": "checkpoints/step_00080/adapter",
                                      "files": adapter_entries(root, ADAPTERS[arm])}})
    for arm in ARMS:
        dev = main / f"{arm}_dev"
        write_json(dev / "run_manifest.json", dev_manifest(root, arm, manifest["code_sha256"]))
        put(dev / "chain_record.txt", dev_note(root, arm))
    return root


FAKE_PYTHON = r'''#!@PYTHON@
"""Stands in for `python` in scripts/gpu/chain_test.sh.

The chain's own snippets (`-`) run for real, in this process, and so does scripts/freeze.py (the copy of the
real script in the tree of the test; FAKE_FREEZE = <exit code> puts a stand-in in its place). Every other
script is replaced: its call is written to the call log, and it leaves what the chain reads afterwards, with
the key names of the real script.

FAKE_FAIL = <name>:<exit code>[:<more>] makes a step fail. <name> is the folder name of a rollout, or the
name of a script. A rollout fails after its first pass: the folder then holds a run manifest that says
"complete": false and the episodes of that pass. With "before" as <more> it fails before it makes the
folder. For a script <more> is a part of its --episodes path: only that call fails.
FAKE_ERRORS_<arm> = <n>: the first n attempts of that arm's rollout are harness errors.
FAKE_CHANGE_ADAPTER_IN = <folder name of a rollout>[:<weights file>]: when that rollout ends, the weights of
its own adapter (or the weights file named) are other bytes.
FAKE_WAIT = <file>: a rollout waits until this file is there. It waits after its look at its folder and
before it makes the folder, where the real script loads the model.
FAKE_BLOCK_NOTE = <folder name of a rollout>: that rollout leaves a folder with the name chain_record.txt
in its run folder, so the chain cannot write its note there.
FAKE_REFERENCE = <policy>: the first record of that scripted policy has the other verdict.

A rollout adds its records to episodes.jsonl and tokens.jsonl (mode "a"), as `append_jsonl` of the real
script does: two rollouts into one folder would leave every episode twice.
"""
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

args = sys.argv[1:]
sys.path.insert(0, "@ROOT@")                 # the real repository: accessdesk/, for scripts/freeze.py
if args[0] == "-":
    sys.argv = args
    exec(compile(sys.stdin.read(), "<snippet of the chain>", "exec"), {"__name__": "__main__"})
    raise SystemExit(0)
name = "tables" if args[:2] == ["-m", "accessdesk.eval.tables"] else Path(args[0]).stem
with open(os.environ["FAKE_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"script": name, "args": args}) + "\n")
options = dict(zip(args, args[1:]))
fail = os.environ.get("FAKE_FAIL", "").split(":")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


def fails(episodes=""):
    """The exit code that FAKE_FAIL asks of this script, or None."""
    if fail[0] != name or (len(fail) > 2 and fail[2] not in episodes):
        return None
    return int(fail[1])


if name == "freeze":
    if os.environ.get("FAKE_FREEZE", ""):
        print("a stand-in for scripts/freeze.py: exit code " + os.environ["FAKE_FREEZE"])
        raise SystemExit(int(os.environ["FAKE_FREEZE"]))
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
    raise SystemExit("scripts/freeze.py did not end with an exit code")

if name == "tables":
    arms = [value for option, value in zip(args, args[1:]) if option == "--arm"]
    pairs = [value for option, value in zip(args, args[1:]) if option == "--pair"]
    if not all(Path(arm.partition("=")[2]).is_file() for arm in arms):
        raise SystemExit("an episodes file of the table is missing")
    if fails() is not None:
        raise SystemExit(fails())
    write(options["--out-json"], {"arms": arms, "pairs": pairs, "baseline": options["--baseline"]})
    Path(options["--out-md"]).write_text("table of " + ", ".join(arms) + "\n", encoding="utf-8")
    raise SystemExit(0)

if name in ("replay_check", "monitor_report"):
    episodes = options["--episodes"]
    split_file = Path(options["--scenarios"]) / (options["--split"] + ".jsonl")
    if not Path(episodes).is_file() or not split_file.is_file():
        raise SystemExit(2)
    if name == "monitor_report":
        if fails(episodes) is not None:
            raise SystemExit(fails(episodes))
        print("monitor report of " + episodes + " on " + options["--split"])
        raise SystemExit(0)
    print("replay check of " + episodes)
    if fails(episodes) is not None:
        print("differences        3   by kind: reward 3")
        raise SystemExit(fails(episodes))
    print("no difference: each checked record has the final state, the turns, the pass and the reward")
    raise SystemExit(0)

if name == "run_scripted":
    if fails() is not None:
        raise SystemExit(fails())
    out = Path(options["--out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    policy, split = options["--policy"], options["--split"]
    # As the real policies on the dev split: the oracle passes every scenario, and a constant policy the
    # scenarios of one outcome.
    passes = {"constant_granted": "grant", "constant_pending": "approval", "constant_already": "already",
              "constant_denied": "inactive"}
    lines = (Path(options["--scenarios"]) / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
    rows = [{"run": "scripted", "arm": policy, "split": split, "family_id": family["family_id"],
             "style": wording["style"], "outcome": family["outcome"], "sample": 0, "harness_error": None,
             "passed": policy == "oracle" or passes[policy] == family["outcome"]}
            for family in map(json.loads, lines) for wording in family["wordings"]]
    if os.environ.get("FAKE_REFERENCE", "") == policy:
        rows[0]["passed"] = not rows[0]["passed"]
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write(str(out) + ".run_manifest.json", {"policy": policy, "split": split, "episodes": len(rows),
                                            "allow_changed_tree": "--allow-changed-tree" in args})
    print("policy " + policy + ", split " + split)
    raise SystemExit(0)

if name != "rollout":
    raise SystemExit("the stand-in does not know the script " + name)

out = Path(options["--out"])
failing = fail[0] == out.name
if failing and fail[2:] == ["before"]:
    print("the stand-in fails before it makes the folder", file=sys.stderr)
    raise SystemExit(int(fail[1]))
if (out / "episodes.jsonl").exists() or (out / "tokens.jsonl").exists():          # as the real script
    print(str(out) + " already holds episodes; choose a new output folder", file=sys.stderr)
    raise SystemExit(2)
if os.environ.get("FAKE_WAIT", ""):          # where the real script loads the model
    for _ in range(600):
        if Path(os.environ["FAKE_WAIT"]).exists():
            break
        time.sleep(0.1)
    else:
        raise SystemExit("the stand-in waited a minute for " + os.environ["FAKE_WAIT"])
scenarios, split = Path(options["--scenarios"]), options["--split"]
lines = (scenarios / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
families = [json.loads(line) for line in lines if line]
if "--families" in options:
    wanted = Path(options["--families"]).read_text(encoding="utf-8").split()
    unknown = sorted(set(wanted) - {family["family_id"] for family in families})
    if unknown:
        raise SystemExit("family ids of " + options["--families"] + " that are not in the split: "
                         + str(unknown))
    families = [family for family in families if family["family_id"] in wanted]
samples, arm = int(options["--samples"]), options["--arm"]
data_sha256 = sha(scenarios / (split + ".jsonl"))
records = [{"run": out.name, "arm": arm, "split": split, "family_id": family["family_id"],
            "style": wording["style"], "scenario_id": family["family_id"] + "/s" + str(wording["style"]),
            "outcome": family["outcome"], "ruleset": "default", "toolset": "five", "prompt": "final",
            "sample": sample, "passed": True, "harness_error": None, "data_sha256": data_sha256}
           for sample in range(samples) for family in families for wording in family["wordings"]]
n_errors = int(os.environ.get("FAKE_ERRORS_" + arm, "0"))
for record in records[:n_errors]:
    record["harness_error"] = "made up"
manifest = {
    "complete": False, "command": args, "config": {"max_new_tokens": int(options["--max-new-tokens"])},
    "text_sha256": sha(options["--text"]),
    "code_sha256": json.loads(Path("MANIFEST.json").read_text(encoding="utf-8"))["code_sha256"],
    "allow_changed_tree": "--allow-changed-tree" in args, "model": options["--model"],
    "revision": options["--revision"], "adapter": options.get("--adapter"), "split": split,
    "split_file_sha256": data_sha256, "family_set": options.get("--set"), "families": len(families),
    "scenarios": len(records) // samples, "samples": samples, "run_seed": int(options["--run-seed"]),
    "ruleset": options.get("--ruleset", "default"), "toolset": options.get("--toolset", "five"),
    "prompt": options.get("--prompt", "final")}
out.mkdir(parents=True, exist_ok=True)       # as the real script: a folder without episodes is used again
write(out / "run_manifest.json", manifest)


def save(rows):
    with (out / "tokens.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("".join("tokens\n" for row in rows))
    with (out / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))


if failing:
    save([record for record in records if record["sample"] == 0])
    raise SystemExit(int(fail[1]))
save(records)
change = os.environ.get("FAKE_CHANGE_ADAPTER_IN", "").split(":")
if change[0] == out.name:
    weights = Path(change[1]) if len(change) > 1 else Path(options["--adapter"]) / "adapter_model.safetensors"
    weights.write_text("changed while the rollout ran\n", encoding="utf-8")
if os.environ.get("FAKE_BLOCK_NOTE", "") == out.name:
    (out / "chain_record.txt").mkdir()
manifest.update({"complete": True, "summary": {"episodes": len(records), "harness_errors": n_errors,
                                               "harness_error_rate": n_errors / len(records)}})
write(out / "run_manifest.json", manifest)
print("episodes " + str(len(records)) + "   harness errors " + str(n_errors))
'''
# Every name that the script reads or sets, and the two of the freeze script and of the stand-in. A value
# of the shell that runs the tests must not reach a chain.
CHAIN_VARIABLES = ("DRESS", "REPEAT_AFTER_FAULT", "REPEAT_REASON", "PYTHON", "MODEL", "REVISION", "SCENARIOS",
                   "TEXT", "SAMPLES", "EVAL_SEED", "MAX_NEW_TOKENS", "MAIN", "ARMS", "POLICIES",
                   "DRESS_FAMILIES", "SPLIT", "OUT", "EXTRA", "RECORD", "REASON", "TABLE", "MORE_ITEMS",
                   "MARKER", "TIME_SHAPE", "RAN", "RUN_LINES", "HOST", "KIND", "ROLLOUTS_RUN", "NOW",
                   "ACCESSDESK_FREEZE_ROOT")


class Tree:
    """A made-up repository with a copy of the chain script, and a stand-in for `python` beside it."""

    def __init__(self, folder: Path, *, source: Path | None = None,
                 splits: tuple[str, ...] = ("test", "calib")) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.root = folder / "repo"
        if source is None:
            make_tree(self.root, splits)
        else:
            shutil.copytree(source, self.root)
        self.main = self.root / "results" / "main"
        self.dress = self.root / DRESS_OUT
        self.calls_file = folder / "calls.jsonl"
        self.python = folder / "fake_python"
        self.python.write_text(FAKE_PYTHON.replace("@PYTHON@", sys.executable).replace("@ROOT@", str(ROOT)),
                               encoding="utf-8")
        self.python.chmod(0o755)
        self.output = ""
        self.notes: list[str] = []

    def run(self, **variables: str) -> tuple[int, list[str], list[dict]]:
        """(exit code, the chain's step lines, the script calls of this run). The CHAIN_NOTE lines are
        kept in `self.notes` and the whole output in `self.output`. The chain is started from another
        folder than its repository. Every option of every call must be one that the called script has."""
        self.calls_file.write_text("", encoding="utf-8")
        clean = {key: value for key, value in os.environ.items()
                 if key not in CHAIN_VARIABLES and not key.startswith("FAKE_")}
        env = {**clean, "PYTHON": str(self.python), "FAKE_CALLS": str(self.calls_file), **variables}
        result = subprocess.run(["bash", str(self.root / "scripts" / "gpu" / "chain_test.sh")], env=env,
                                capture_output=True, text=True, cwd=self.root.parent)
        self.output = result.stdout + result.stderr
        lines = result.stdout.splitlines()
        self.notes = [line for line in lines if line.startswith("CHAIN_NOTE")]
        markers = [line for line in lines if line.startswith(("STEP_", "CHAIN_")) and line not in self.notes]
        calls = [json.loads(line) for line in self.calls_file.read_text(encoding="utf-8").splitlines()]
        for call in calls:
            if call["script"] == "freeze":
                assert call["args"] == ["scripts/freeze.py", "check"]
                continue
            unknown = {arg for arg in call["args"] if arg.startswith("--")} - options_of(call["script"])
            assert not unknown, (call["script"], sorted(unknown))
            assert "--allow-changed-tree" not in call["args"]
        return result.returncode, markers, calls

    def start(self, name: str, **variables: str) -> subprocess.Popen:
        """Start a chain and do not wait for it. Its script calls go into the call log <name>."""
        clean = {key: value for key, value in os.environ.items()
                 if key not in CHAIN_VARIABLES and not key.startswith("FAKE_")}
        env = {**clean, "PYTHON": str(self.python), "FAKE_CALLS": str(self.calls_file.with_name(name)),
               **variables}
        self.calls_file.with_name(name).write_text("", encoding="utf-8")
        return subprocess.Popen(["bash", str(self.root / "scripts" / "gpu" / "chain_test.sh")], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                cwd=self.root.parent)

    def record(self, folder: Path | None = None) -> list[str]:
        """The lines of the record of the run (of results/main, or of the folder given)."""
        return ((folder or self.main) / "test_run_record.txt").read_text(encoding="utf-8").splitlines()

    def ledger(self, folder: Path | None = None) -> list[str]:
        """The lines of the record without their time. Every line must start with a time (UTC)."""
        lines = self.record(folder)
        assert all(re.match(rf"{TIME} ", line) for line in lines), lines
        return [line[21:] for line in lines]


def ran(names: list[str]) -> list[str]:
    """The marker lines of steps that ran one after the other without a fault."""
    return [line for name in names for line in (f"STEP_START {name}", f"STEP_DONE {name} 0")]


def stopped_at(name: str, code: int = 1) -> list[str]:
    """The marker lines of a step that ended the chain."""
    return [f"STEP_START {name}", f"STEP_DONE {name} {code}", f"CHAIN_FAILED {name}"]


def value_of(call: dict, option: str) -> str:
    return call["args"][call["args"].index(option) + 1]


def scripts_of(calls: list[dict]) -> list[str]:
    return [call["script"] for call in calls]


START = ["freeze_check", "adapters", "run_once"]


def rollouts(split: str) -> list[str]:
    return [name for arm in ARMS for name in (f"{arm}_{split}", f"gate_{arm}")]


AFTER = ["adapters_after"]          # the step adapters once more, between the rollouts and the reports
REPORTS = [f"replay_{arm}" for arm in ARMS] + ["table"] + [f"monitors_{arm}" for arm in ARMS] \
    + [f"reference_{policy}" for policy in POLICIES] + ["reference_check"]
# The step reference_check starts no script: it is a snippet of the chain that reads the reference records.
REPORT_SCRIPTS = ["replay_check"] * 4 + ["tables"] + ["monitor_report"] * 4 + ["run_scripted"] * 5


# The lines that the chain adds to the record for one rollout, without their time.
def adapter_part(root: Path, arm: str) -> str:
    """The adapter of an arm as the record names it: its folder and the SHA-256 of its two files."""
    if not ADAPTERS[arm]:
        return "adapter=none"
    hashes = " ".join(f"{name}={sha256_of(root / ADAPTERS[arm] / name)}" for name in ADAPTER_FILES)
    return f"adapter={ADAPTERS[arm]} {hashes}"


def started(root: Path, arm: str, split: str = "test", out: str = "results/main") -> str:
    """A pattern for the line ROLLOUT_START of an arm: the process and the machine are not fixed."""
    return (rf"ROLLOUT_START arm={arm} folder={out}/{arm}_{split} pid=\d+ host=\S+ script={sha256_of(SCRIPT)} "
            + re.escape(adapter_part(root, arm)))


def ended(arm: str, code: int = 0, wrote: str = "complete", split: str = "test",
          out: str = "results/main") -> str:
    return f"ROLLOUT_END arm={arm} folder={out}/{arm}_{split} exit={code} wrote={wrote}"


def gate_line(arm: str, kind: str = "GATE_PASSED", counts: str | None = None, split: str = "test",
              out: str = "results/main") -> str:
    counts = counts or "attempts=200 harness_errors=0 harness_error_rate=0.0"
    return f"{kind} arm={arm} folder={out}/{arm}_{split} {counts}"


def a_rollout(root: Path, arm: str, split: str = "test", out: str = "results/main",
              counts: str | None = None) -> list[str]:
    """Patterns for the three lines of a rollout that ended and passed the gate."""
    return [started(root, arm, split, out), re.escape(ended(arm, split=split, out=out)),
            re.escape(gate_line(arm, counts=counts, split=split, out=out))]


def assert_lines(lines: list[str], patterns: list[str]) -> None:
    """Every line matches the pattern at its place, as a whole."""
    assert len(lines) == len(patterns), lines
    for line, pattern in zip(lines, patterns):
        assert re.fullmatch(pattern, line), (line, pattern)


def block(root: Path, kind: str, rollouts_run: str, split: str = "test", out: str = "results/main",
          dress: str = "0", counts: str | None = None, aside: dict[str, str] | None = None) -> list[str]:
    """Patterns for the six lines that a chain adds to the record when it ends without a fault."""
    counts = counts or "attempts=200 harness_errors=0 harness_error_rate=0.0"
    code_sha256 = read_json(root / "MANIFEST.json")["code_sha256"]
    lines = [f"{kind} DRESS={dress} split={split} out={out} rollouts_run={rollouts_run}",
             f"  MANIFEST.json={sha256_of(root / 'MANIFEST.json')} code_sha256={code_sha256} "
             f"scripts/gpu/chain_test.sh={sha256_of(SCRIPT)}"]
    lines += [f"  arm={arm} {adapter_part(root, arm)} folder={out}/{arm}_{split} {counts} "
              f"moved_aside={(aside or {}).get(arm, 'none')}" for arm in ARMS]
    return [re.escape(line) for line in lines]


# What each call is given, typed by hand from the task. `out` is the folder of everything that is written.
def rollout_args(arm: str, split: str, out: str) -> list[str]:
    adapter = ["--adapter", ADAPTERS[arm]] if ADAPTERS[arm] else []
    return ["scripts/gpu/rollout.py", "--model", MODEL, "--revision", REVISION, *adapter,
            "--text", "text/model_text.yaml", "--scenarios", "data/scenarios", "--split", split,
            "--samples", "8", "--run-seed", "0", "--max-new-tokens", "260", "--arm", arm,
            "--out", f"{out}/{arm}_{split}"]


def report_args(script: str, arm: str, split: str, out: str) -> list[str]:
    return [f"scripts/{script}.py", "--episodes", f"{out}/{arm}_{split}/episodes.jsonl",
            "--scenarios", "data/scenarios", "--split", split, "--text", "text/model_text.yaml"]


def table_args(split: str, out: str) -> list[str]:
    arms = [item for arm in ARMS for item in ("--arm", f"{arm}={out}/{arm}_{split}/episodes.jsonl")]
    name = f"{out}/tables/{split}_base_sft_grpo_seed1_seed2"
    return ["-m", "accessdesk.eval.tables", *arms, "--baseline", "base", "--pair", "base:sft",
            "--pair", "sft:grpo_seed1", "--pair", "sft:grpo_seed2", "--out-json", f"{name}.json",
            "--out-md", f"{name}.md"]


def reference_args(policy: str, split: str, out: str) -> list[str]:
    return ["scripts/run_scripted.py", "--scenarios", "data/scenarios", "--split", split,
            "--text", "text/model_text.yaml", "--policy", policy,
            "--out", f"{out}/reference/{policy}_{split}.jsonl"]


def all_args(split: str, out: str, more: list[str]) -> list[list[str]]:
    """The arguments of every call of a whole run, in order. `more`: more options of each rollout."""
    return [["scripts/freeze.py", "check"]] + [rollout_args(arm, split, out) + more for arm in ARMS] \
        + [report_args("replay_check", arm, split, out) for arm in ARMS] + [table_args(split, out)] \
        + [report_args("monitor_report", arm, split, out) for arm in ARMS] \
        + [reference_args(policy, split, out) for policy in POLICIES]


@pytest.fixture(scope="module")
def real_run(tmp_path_factory) -> SimpleNamespace:
    """One whole run in the mode of the test run (DRESS=0), in a made-up repository that has no calibration
    split. Nothing of it is the real test run. The tests copy its tree; they must not change it."""
    tree = Tree(tmp_path_factory.mktemp("real_run"), splits=("test",))
    before = files_of(tree.root)
    code, markers, calls = tree.run()
    assert code == 0 and markers[-1] == "CHAIN_DONE", tree.output
    return SimpleNamespace(tree=tree, before=before, markers=markers, calls=calls, output=tree.output,
                           notes=tree.notes)


# ---------------------------------------------------------------------------
# The test run and the rehearsal: the order of the commands
# ---------------------------------------------------------------------------
def test_the_test_run_is_the_four_rollouts_then_the_reports_in_this_order(real_run):
    tree, calls = real_run.tree, real_run.calls
    assert real_run.markers == ran(START + rollouts("test") + AFTER + REPORTS) + ["CHAIN_DONE"]
    assert scripts_of(calls) == ["freeze"] + ["rollout"] * 4 + REPORT_SCRIPTS
    assert [call["args"] for call in calls] == all_args("test", "results/main", [])
    # The check of the frozen tree is the first line of the log after the start, before the mode.
    lines = real_run.output.splitlines()
    assert re.fullmatch(rf"chain test starts at {TIME}, PID \d+ on \S+", lines[0])
    assert lines[1].startswith("python: ") and lines[2].startswith("lock: this chain holds the lock")
    assert lines[3] == "STEP_START freeze_check"
    # The made-up tree has four frozen files: the text file, the summary, and the dev and the test file.
    # The number of groups is the one of accessdesk/freeze.py: an amendment that adds a group (the audit
    # code, the report code) must not break this test.
    assert (f"nothing changed: 4 files in {len(freeze.FROZEN_GROUPS)} groups are as in MANIFEST.json"
            in real_run.output)
    assert real_run.notes[0].startswith("CHAIN_NOTE DRESS=0: the TEST RUN of the plan, on the split test")
    # No other split than test, no part of it, and nothing of a rehearsal.
    assert {value_of(call, "--split") for call in calls[1:5] + calls[5:9] + calls[10:]} == {"test"}
    for call in calls[1:5]:
        assert not {"--families", "--limit", "--styles", "--set", "--ruleset", "--toolset", "--prompt",
                    "--chunk-size"} & set(call["args"])
    assert not (tree.root / "results" / "_smoke").exists()
    # Each run folder holds the run and the chain's own note of it, as the dev folders do.
    for arm in ARMS:
        folder = tree.main / f"{arm}_test"
        manifest = read_json(folder / "run_manifest.json")
        assert manifest["complete"] is True and manifest["split"] == "test" and manifest["families"] == 5
        assert manifest["summary"] == {"episodes": 200, "harness_errors": 0, "harness_error_rate": 0.0}
        note = (folder / "chain_record.txt").read_text(encoding="utf-8")
        assert note == dev_note(tree.root, arm)        # the same lines as the note of the dev run
    # The reports: one table of the four arms with the three comparisons of the plan; a file per arm.
    table = read_json(tree.main / "tables" / "test_base_sft_grpo_seed1_seed2.json")
    assert table["pairs"] == ["base:sft", "sft:grpo_seed1", "sft:grpo_seed2"] and table["baseline"] == "base"
    assert [arm.partition("=")[0] for arm in table["arms"]] == list(ARMS)
    assert (tree.main / "tables" / "test_base_sft_grpo_seed1_seed2.md").is_file()
    for arm in ARMS:
        assert (tree.main / "monitors" / f"{arm}_test.md").read_text(encoding="utf-8") == \
            f"monitor report of results/main/{arm}_test/episodes.jsonl on test\n"
    for policy in POLICIES:
        rows = (tree.main / "reference" / f"{policy}_test.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 25 and {json.loads(row)["arm"] for row in rows} == {policy}
        assert (tree.main / "reference" / f"{policy}_test.jsonl.run_manifest.json").is_file()
    # The check of the reference rows says what it found: the oracle passes every scenario, and each
    # constant policy the scenarios of one outcome.
    assert "oracle: 25 records in 4 outcomes; every record passes in ['already', 'approval', 'grant', " \
           "'inactive'], and none passes in []: as the plan says" in real_run.output
    assert "constant_pending: 25 records in 4 outcomes; every record passes in ['approval'], and none " \
           "passes in ['already', 'grant', 'inactive']: as the plan says" in real_run.output
    # Nothing that was there before the run was changed or removed, and the lock left no file.
    after = files_of(tree.root)
    assert {name: after[name] for name in real_run.before} == real_run.before
    new = {name.split("/")[2] for name in after if name not in real_run.before}
    assert new == {f"{arm}_test" for arm in ARMS} | {"tables", "monitors", "reference", "test_run_record.txt"}
    assert not [path for path in tree.root.rglob("*") if "lock" in path.name]
    # Every episode is in its file once: 200 lines per arm.
    for arm in ARMS:
        for name in ("episodes.jsonl", "tokens.jsonl"):
            assert len((tree.main / f"{arm}_test" / name).read_text(encoding="utf-8").splitlines()) == 200


def test_the_gate_and_the_record_of_a_test_run(real_run):
    tree = real_run.tree
    for arm in ARMS:
        assert f"results/main/{arm}_test/run_manifest.json: attempts=200 harness_errors=0 " \
               "harness_error_rate=0.0" in real_run.output
    assert real_run.notes[-1] == \
        "CHAIN_NOTE the record of this run was added to results/main/test_run_record.txt"
    # The record: three lines for each rollout (it was started, it ended, the gate), then the six lines of
    # the end. It holds counts of attempts and of harness errors, and no pass rate.
    record = tree.record()
    lines = tree.ledger()
    assert len(record) == 18 and len({line[:20] for line in record[12:]}) == 1     # one time for the block
    assert_lines(lines[:12], [pattern for arm in ARMS for pattern in a_rollout(tree.root, arm)])
    assert_lines(lines[12:], block(tree.root, "TEST_RUN_DONE", "base,sft,grpo_seed1,grpo_seed2"))
    assert lines[12] == "TEST_RUN_DONE DRESS=0 split=test out=results/main rollouts_run=base,sft,grpo_seed1," \
                        "grpo_seed2"
    code_sha256 = read_json(tree.root / "MANIFEST.json")["code_sha256"]
    assert lines[13] == (f"  MANIFEST.json={sha256_of(tree.root / 'MANIFEST.json')} code_sha256={code_sha256} "
                         f"scripts/gpu/chain_test.sh={sha256_of(SCRIPT)}")
    counts = "attempts=200 harness_errors=0 harness_error_rate=0.0 moved_aside=none"
    assert lines[14] == f"  arm=base adapter=none folder=results/main/base_test {counts}"
    for line, arm in zip(lines[15:], ARMS[1:]):
        hashes = " ".join(f"{name}={sha256_of(tree.root / ADAPTERS[arm] / name)}" for name in ADAPTER_FILES)
        assert line == f"  arm={arm} adapter={ADAPTERS[arm]} {hashes} folder=results/main/{arm}_test {counts}"
    assert "pass" not in "".join(lines).replace("GATE_PASSED", "")
    # The line that a rollout was started names this process and this machine, and it is written before the
    # rollout: in the log it stands above the start of the step.
    assert re.search(rf" pid=\d+ host=\S+ script={sha256_of(SCRIPT)} adapter=none$", lines[0])
    output = real_run.output
    assert output.index("STEP_START base_test") < output.index("STEP_DONE base_test 0") < \
        output.index("STEP_START gate_base")


@pytest.mark.parametrize("arm", ARMS)
def test_the_rollout_of_an_arm_is_the_command_of_its_dev_run_with_another_split_and_folder(real_run, arm):
    """Reads the run manifest of the arm's dev run in the repository: the command that made the dev numbers.
    The made-up repository has the adapters in the places of the real ones, so the two commands can be
    compared word for word."""
    path = ROOT / "results" / "main" / f"{arm}_dev" / "run_manifest.json"
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    dev = read_json(path)
    wanted = list(dev["command"])
    assert (value_of({"args": wanted}, "--split"), value_of({"args": wanted}, "--out")) == \
        ("dev", f"results/main/{arm}_dev")
    wanted[wanted.index("--split") + 1] = "test"
    wanted[wanted.index("--out") + 1] = f"results/main/{arm}_test"
    assert real_run.calls[1 + ARMS.index(arm)]["args"] == wanted
    # What the command does not name is the default of the script, and the dev run says which it was.
    assert (dev["ruleset"], dev["toolset"], dev["prompt"]) == ("default", "five", "final")
    assert (dev["model"], dev["revision"], dev["adapter"]) == (MODEL, REVISION, ADAPTERS[arm])
    assert (dev["samples"], dev["run_seed"], dev["config"]["max_new_tokens"]) == (8, 0, 260)
    assert dev["complete"] is True and dev["allow_changed_tree"] is False
    # Every name that the chain compares is in the real manifest.
    for name in COMPARED:
        value = dev
        for part in name.split("/"):
            assert part in value, name
            value = value[part]


def test_a_rehearsal_runs_the_same_commands_on_calibration_families_and_writes_only_in_its_own_folder(
        tmp_path, real_run):
    """The tree of this test has no test split at all, so a rehearsal cannot have read one."""
    tree = Tree(tmp_path, splits=("calib",))
    assert not (tree.root / "data" / "scenarios" / "test.jsonl").exists()
    before = files_of(tree.root)
    code, markers, calls = tree.run(DRESS="1")
    assert code == 0, tree.output
    assert markers == ran(START + rollouts("calib") + AFTER + REPORTS) + ["CHAIN_DONE"]
    families_file = f"{DRESS_OUT}/dress_families.txt"
    assert [call["args"] for call in calls] == all_args("calib", DRESS_OUT, ["--families", families_file])
    # The same commands in the same order as the test run: the same scripts, and the same options but for
    # the list of families that each rollout of a rehearsal is given.
    assert scripts_of(calls) == scripts_of(real_run.calls)
    for dress, real in zip(calls, real_run.calls):
        options = [arg for arg in dress["args"] if arg.startswith("--")]
        assert [option for option in options if option != "--families"] == \
            [arg for arg in real["args"] if arg.startswith("--")]
    assert tree.notes[0].startswith("CHAIN_NOTE DRESS=1: a REHEARSAL on the calibration families ca-00 ca-01 "
                                    "ca-02 ca-03 (the split calib), into results/_smoke/test_dress.")
    # Four families, each in its four wordings, 8 attempts: 128 attempts per arm.
    assert (tree.root / families_file).read_text(encoding="utf-8").split() == DRESS_FAMILIES
    for arm in ARMS:
        manifest = read_json(tree.dress / f"{arm}_calib" / "run_manifest.json")
        assert (manifest["split"], manifest["families"], manifest["summary"]["episodes"]) == ("calib", 4, 128)
        assert manifest["adapter"] == ADAPTERS[arm]                      # the adapters of results/main
        assert f"options=--families {families_file}\n" in \
            (tree.dress / f"{arm}_calib" / "chain_record.txt").read_text(encoding="utf-8")
    # Nothing is written outside the folder of the rehearsal: results/main is as it was, and so is every
    # other file of the tree.
    after = files_of(tree.root)
    assert {name for name in after if name not in before} == \
        {name for name in after if name.startswith(DRESS_OUT + "/")}
    assert {name: after[name] for name in before} == before
    assert not list(tree.main.glob("*_test*")) and not list(tree.main.glob("*_calib*"))
    assert not (tree.main / "test_run_record.txt").exists()
    # The record of a rehearsal is in its own folder and says what it is.
    lines = tree.ledger(tree.dress)
    counts = "attempts=128 harness_errors=0 harness_error_rate=0.0"
    assert_lines(lines[:12], [pattern for arm in ARMS
                              for pattern in a_rollout(tree.root, arm, "calib", DRESS_OUT, counts)])
    assert_lines(lines[12:], block(tree.root, "TEST_RUN_DONE", "base,sft,grpo_seed1,grpo_seed2", "calib",
                                   DRESS_OUT, "1", counts))
    assert lines[12] == f"TEST_RUN_DONE DRESS=1 split=calib out={DRESS_OUT} rollouts_run=base,sft," \
                        "grpo_seed1,grpo_seed2"
    assert {path.name for path in tree.dress.iterdir()} == {
        "base_calib", "sft_calib", "grpo_seed1_calib", "grpo_seed2_calib", "dress_families.txt", "tables",
        "monitors", "reference", "test_run_record.txt"}
    assert (tree.dress / "tables" / "calib_base_sft_grpo_seed1_seed2.md").is_file()
    assert (tree.dress / "monitors" / "grpo_seed2_calib.md").is_file()
    assert (tree.dress / "reference" / "oracle_calib.jsonl").is_file()


def test_no_variable_of_the_environment_changes_the_split_the_folders_or_the_options(tmp_path, real_run):
    """The names of the stage chains (OUT, SCENARIOS, EVAL_SEED, ...) and the script's own names are given
    other values. The test run is the same, call for call."""
    tree = Tree(tmp_path, splits=("test",))
    elsewhere = str(tmp_path / "elsewhere")
    code, markers, calls = tree.run(
        SPLIT="dev", OUT=elsewhere, MAIN=elsewhere, MODEL="Qwen/Qwen3.5-4B", REVISION="0" * 40,
        SCENARIOS="data/scenarios_placeholder", TEXT="text/placeholder_text.yaml", SAMPLES="1", EVAL_SEED="5",
        MAX_NEW_TOKENS="300", ARMS="base", POLICIES="oracle", DRESS_FAMILIES="ca-05", EXTRA="--limit 1",
        RECORD=elsewhere + "/record.txt", TABLE=elsewhere + "/table", REQUIRE_FROZEN="0",
        MORE_ITEMS="ruleset=other", ACCESSDESK_FREEZE_ROOT=elsewhere)
    assert code == 0, tree.output
    assert markers == real_run.markers and calls == real_run.calls
    assert not Path(elsewhere).exists()
    # DRESS takes only its two values. Nothing but the check of the tree runs before the refusal.
    for value in ("2", "yes", "true", "01"):
        code, markers, calls = tree.run(DRESS=value)
        assert (code, markers, scripts_of(calls)) == (1, ran(["freeze_check"]) + ["CHAIN_FAILED start"],
                                                      ["freeze"]), value
        assert f"DRESS must be 0 (the test run) or 1 (a rehearsal), not '{value}'" in tree.output
    assert not (tree.root / "results" / "_smoke").exists()


def test_the_real_tables_program_takes_the_table_call_of_the_chain(tmp_path, real_run, monkeypatch):
    """The stand-in only writes down what the table is asked for. Here the real accessdesk/eval/tables.py
    gets the arguments of the chain, with four files of hand-made records in the places of the episodes:
    4 families in 2 wordings with 2 attempts each. It makes one table with a row for each arm and the three
    comparisons that the plan names for test, in the plan's order."""
    monkeypatch.setenv("ACCESSDESK_IMPL", "main")
    passed, fail = 100, 0
    # Per arm: the rewards of the two attempts of every scenario of a family, family by family.
    rewards = {"base": [[fail, fail], [fail, fail], [passed, fail], [fail, fail]],
               "sft": [[passed, fail], [passed, fail], [passed, passed], [fail, fail]],
               "grpo_seed1": [[passed, passed], [passed, fail], [passed, passed], [passed, fail]],
               "grpo_seed2": [[passed, passed], [passed, passed], [passed, passed], [fail, fail]]}
    outcomes = ("grant", "approval", "already", "inactive")
    args = list(real_run.calls[9]["args"][2:])
    assert real_run.calls[9]["script"] == "tables"
    for arm in ARMS:
        records = [record for index, per_scenario in enumerate(rewards[arm]) for style in (1, 4)
                   for record in scenario_records(f"te-{index:03d}", style, per_scenario,
                                                  outcome=outcomes[index], arm=arm)]
        put(tmp_path / f"{arm}.jsonl", "".join(json.dumps(record) + "\n" for record in records))
        position = args.index(f"{arm}=results/main/{arm}_test/episodes.jsonl")
        args[position] = f"{arm}={tmp_path / f'{arm}.jsonl'}"
    args[args.index("--out-json") + 1] = str(tmp_path / "table.json")
    args[args.index("--out-md") + 1] = str(tmp_path / "table.md")
    tables.main(args)
    table = read_json(tmp_path / "table.json")
    assert [row["arm"] for row in table["arms"]] == list(ARMS) and table["baseline_arm"] == "base"
    assert (table["n_families"], table["n_scenarios"], table["n_episodes_per_arm"]) == (4, 8, 16)
    # Family pass rates: base 0, 0, 0.5, 0; sft 0.5, 0.5, 1, 0; seed 1: 1, 0.5, 1, 0.5; seed 2: 1, 1, 1, 0.
    assert [row["pass_rate"] for row in table["arms"]] == [0.125, 0.5, 0.75, 0.75]
    assert [(row["a"], row["b"], row["diff"]) for row in table["pairs"]] == \
        [("base", "sft", 0.375), ("sft", "grpo_seed1", 0.25), ("sft", "grpo_seed2", 0.25)]
    markdown = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert re.findall(r"^\| (\w+ minus \w+) \|", markdown, flags=re.MULTILINE) == \
        ["sft minus base", "grpo_seed1 minus sft", "grpo_seed2 minus sft"]


# ---------------------------------------------------------------------------
# Run once
# ---------------------------------------------------------------------------
RUN_FILES = ("episodes.jsonl", "tokens.jsonl", "run_manifest.json", "chain_record.txt")


def run_folders(tree: Tree, split: str = "test") -> dict[str, bytes]:
    """The files of the four run folders of a tree."""
    return {name: content for name, content in files_of(tree.main).items()
            if name.split("/")[0] in {f"{arm}_{split}" for arm in ARMS}}


def test_a_second_start_runs_no_arm_again_and_builds_the_reports_again(tmp_path, real_run):
    tree = Tree(tmp_path, source=real_run.tree.root)
    runs = run_folders(tree)
    assert sorted(runs) == sorted(f"{arm}_test/{name}" for arm in ARMS for name in RUN_FILES)
    code, markers, calls = tree.run()
    assert code == 0, tree.output
    skipped = [line for arm in ARMS for line in (f"STEP_SKIPPED {arm}_test", *ran([f"gate_{arm}"]))]
    assert markers == ran(START) + skipped + ran(AFTER + REPORTS) + ["CHAIN_DONE"]
    assert scripts_of(calls) == ["freeze"] + REPORT_SCRIPTS
    assert "not compared" not in tree.output and run_folders(tree) == runs
    for arm in ARMS:
        assert f"results/main/{arm}_test: a complete run. It is not run again." in tree.output
    # The record only grows: the lines of the first run are still its first lines. The second start ran no
    # rollout, so its lines are not a second TEST_RUN_DONE: they say that the reports were built again.
    record = tree.record()
    assert record[:18] == real_run.tree.record() and len(record) == 24
    assert_lines(tree.ledger()[18:], block(tree.root, "REPORTS_REBUILT", "none"))
    assert [line.split()[1] for line in record].count("TEST_RUN_DONE") == 1


def test_a_complete_folder_is_never_run_again_whatever_the_variables_say(tmp_path, real_run):
    tree = Tree(tmp_path, source=real_run.tree.root)
    files = files_of(tree.root)
    # All four are complete: there is nothing to repeat, and the chain says so and does not start.
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="a reason that fits nothing")
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "REPEAT_AFTER_FAULT=1, and no arm needs a repeat: no folder is unfinished, and the record names " \
           "no rollout whose folder is gone. Start the chain without REPEAT_AFTER_FAULT." in tree.output
    assert files_of(tree.root) == files and not list(tree.main.glob("*.fault_*"))
    # One arm is unfinished and three are complete: only that one is moved aside and run.
    shutil.rmtree(tree.main / "grpo_seed1_test")
    put(tree.main / "grpo_seed1_test" / "run_manifest.json", '{"complete": false}')
    runs = run_folders(tree)
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="the machine was lost")
    assert code == 0, tree.output
    assert scripts_of(calls) == ["freeze", "rollout"] + REPORT_SCRIPTS
    assert value_of(calls[1], "--out") == "results/main/grpo_seed1_test"
    assert [line for line in markers if line.startswith("STEP_SKIPPED")] == \
        ["STEP_SKIPPED base_test", "STEP_SKIPPED sft_test", "STEP_SKIPPED grpo_seed2_test"]
    after = run_folders(tree)
    assert {name: after[name] for name in runs if not name.startswith("grpo_seed1_test/")} == \
        {name: content for name, content in runs.items() if not name.startswith("grpo_seed1_test/")}


@pytest.mark.parametrize("held", [
    {"run_manifest.json": '{"complete": false}', "episodes.jsonl": "a pass\n", "tokens.jsonl": "a pass\n"},
    {"run_manifest.json": '{"complete": false}'},
    {},
    {"run_manifest.json": "{this is not JSON"},
    {"episodes.jsonl": "episodes without a run manifest\n"},
], ids=["a first pass", "only a run manifest", "an empty folder", "a manifest that cannot be read",
        "episodes without a manifest"])
def test_a_folder_that_is_not_complete_stops_the_chain_before_any_rollout(tmp_path, held):
    """The folder of the second arm is there and is not complete. scripts/gpu/rollout.py itself would refuse
    the first and the last of these folders (they hold episodes) and would write into the three others (the
    stand-in does the same). The chain leaves that question to no script: it stops before any rollout, also
    before the one of the first arm, whose folder is not there."""
    tree = Tree(tmp_path, splits=("test",))
    folder = tree.main / "sft_test"
    folder.mkdir()
    for name, content in held.items():
        put(folder / name, content)
    files = files_of(tree.root)
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "results/main/sft_test: there, and NOT complete." in tree.output
    assert "results/main/base_test: not there. The arm runs." in tree.output
    assert "The chain stops, and nothing was deleted, moved or written over. A reason is needed for: sft." \
        in tree.output
    assert "REPEAT_AFTER_FAULT=1 REPEAT_REASON='what the fault was'" in tree.output
    assert "a line in NOTES.md" in tree.output
    assert files_of(tree.root) == files and folder.is_dir()
    assert sorted(path.name for path in folder.iterdir()) == sorted(held)
    assert not list(tree.main.glob("*.fault_*")) and not (tree.main / "base_test").exists()
    assert not (tree.main / "test_run_record.txt").exists()


def test_a_file_or_a_dead_link_in_the_place_of_a_run_folder_stops_the_chain_too(tmp_path):
    tree = Tree(tmp_path, splits=("test",))
    put(tree.main / "base_test", "a file, not a folder\n")
    (tree.main / "grpo_seed2_test").symlink_to(tree.main / "nowhere")
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "A reason is needed for: base grpo_seed2." in tree.output
    assert (tree.main / "base_test").read_text(encoding="utf-8") == "a file, not a folder\n"
    assert (tree.main / "grpo_seed2_test").is_symlink()


def test_repeat_after_fault_notes_the_reason_first_then_moves_the_folder_and_runs_the_arm_again(tmp_path):
    """The rollout of the second arm ends with a fault (exit code 7) after its first pass."""
    tree = Tree(tmp_path, splits=("test",))
    code, markers, calls = tree.run(FAKE_FAIL="sft_test:7")
    assert code == 7
    assert markers == ran(START + ["base_test", "gate_base"]) + stopped_at("sft_test", 7)
    assert scripts_of(calls) == ["freeze", "rollout", "rollout"]
    unfinished = files_of(tree.main / "sft_test")
    assert sorted(unfinished) == ["episodes.jsonl", "run_manifest.json", "tokens.jsonl"]
    assert read_json(tree.main / "sft_test" / "run_manifest.json")["complete"] is False
    # The record names both rollouts. The one of sft was started; it ended with the code 7 and left an
    # unfinished folder. So a test rollout is in the record from the moment before it starts.
    first = a_rollout(tree.root, "base") + [started(tree.root, "sft"), re.escape(ended("sft", 7, "unfinished"))]
    assert_lines(tree.ledger(), first)
    files = files_of(tree.root)

    # A plain second start stops. So does one that asks for the repeat without a reason, or with a
    # reason of white space, and one with another word than 1: nothing is moved, and nothing runs.
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    for variables in ({"REPEAT_AFTER_FAULT": "1"}, {"REPEAT_AFTER_FAULT": "1", "REPEAT_REASON": " \t\n "},
                      {"REPEAT_AFTER_FAULT": "yes", "REPEAT_REASON": "the machine was lost"}):
        code, markers, calls = tree.run(**variables)
        assert (code, markers, scripts_of(calls)) == (1, ran(["freeze_check"]) + ["CHAIN_FAILED start"],
                                                      ["freeze"]), variables
    assert "REPEAT_AFTER_FAULT must be 0 or 1, not 'yes'" in tree.output
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1")
    assert "REPEAT_AFTER_FAULT=1 needs REPEAT_REASON: what the fault was." in tree.output
    assert files_of(tree.root) == files

    # With a reason: the record gets the line, the folder is renamed and kept, and the arm runs again.
    # The complete first arm is not run again.
    reason = "the host stopped the machine in pass 2 of sft;\nsee the log"
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON=reason)
    assert code == 0, tree.output
    assert markers == ran(START[:2]) + ["STEP_START run_once",
                                        "STEP_MOVED_ASIDE sft_test results/main/sft_test.fault_1",
                                        "STEP_DONE run_once 0", "STEP_SKIPPED base_test"] \
        + ran(["gate_base"] + rollouts("test")[2:] + AFTER + REPORTS) + ["CHAIN_DONE"]
    assert scripts_of(calls) == ["freeze"] + ["rollout"] * 3 + REPORT_SCRIPTS
    assert [value_of(call, "--arm") for call in calls[1:4]] == ["sft", "grpo_seed1", "grpo_seed2"]
    assert files_of(tree.main / "sft_test.fault_1") == unfinished
    assert read_json(tree.main / "sft_test" / "run_manifest.json")["complete"] is True
    lines = tree.ledger()
    assert_lines(lines[:5], first)
    assert lines[5] == ("REPEAT_AFTER_FAULT arm=sft folder=results/main/sft_test state=unfinished "
                        "moving_to=results/main/sft_test.fault_1 "
                        "reason=the host stopped the machine in pass 2 of sft; see the log")
    assert_lines(lines[6:15], [pattern for arm in ARMS[1:] for pattern in a_rollout(tree.root, arm)])
    assert_lines(lines[15:], block(tree.root, "TEST_RUN_DONE", "sft,grpo_seed1,grpo_seed2",
                                   aside={"sft": "results/main/sft_test.fault_1"}))
    # A start after that runs nothing again and asks for no reason: the folder beside sft_test is named.
    code, markers, calls = tree.run()
    assert code == 0 and scripts_of(calls) == ["freeze"] + REPORT_SCRIPTS, tree.output
    assert "named by no line" not in tree.output


def test_no_folder_is_moved_and_no_rollout_is_started_without_its_line_in_the_record(tmp_path):
    """The record cannot be written (here: a file without the right to write; on a machine: a full disk).
    The line of a repeat comes before the move, and the line of a start before the rollout, so neither
    happens: the unfinished folder stays where it is, and a plain start after that still stops at it."""
    if os.geteuid() == 0:
        pytest.skip("root writes to a file without the right to write")
    tree = Tree(tmp_path, splits=("test",))
    code, markers, calls = tree.run(FAKE_FAIL="sft_test:7")
    assert code == 7
    record = tree.main / "test_run_record.txt"
    record.chmod(0o444)
    files = files_of(tree.root)
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="the machine stopped")
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "The line of this repeat could not be added to results/main/test_run_record.txt. Nothing was " \
           "moved: results/main/sft_test is as it was." in tree.output
    assert files_of(tree.root) == files and not list(tree.main.glob("*.fault_*"))
    record.chmod(0o644)
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "results/main/sft_test: there, and NOT complete." in tree.output
    assert files_of(tree.root) == files

    # A first start with a record that cannot be written: no rollout is started.
    other = Tree(tmp_path / "other", splits=("test",))
    put(other.main / "test_run_record.txt", "")
    (other.main / "test_run_record.txt").chmod(0o444)
    code, markers, calls = other.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START) + ["CHAIN_FAILED base_test"], ["freeze"])
    assert "The line ROLLOUT_START could not be added to results/main/test_run_record.txt. The rollout of " \
           "base was not started: nothing was sampled." in other.output
    assert not (other.main / "base_test").exists()

    # Something that is not a file in the place of the record: the chain does not start, with or without
    # the wish to repeat.
    third = Tree(tmp_path / "third", splits=("test",))
    put(third.main / "sft_test" / "run_manifest.json", '{"complete": false}')
    (third.main / "test_run_record.txt").mkdir()
    files = files_of(third.root)
    for variables in ({}, {"REPEAT_AFTER_FAULT": "1", "REPEAT_REASON": "the host died"}):
        code, markers, calls = third.run(**variables)
        assert (code, markers, scripts_of(calls)) == (1, ran(["freeze_check"]) + ["CHAIN_FAILED start"],
                                                      ["freeze"])
        assert "results/main/test_run_record.txt is there, and it is not a file that can be read" in \
            third.output
    assert files_of(third.root) == files and (third.main / "sft_test").is_dir()


def test_a_second_fault_of_an_arm_gets_the_next_number_and_the_first_folder_stays(tmp_path):
    """The first arm fails twice. Each repeat has its own line with its own reason, and each line names
    the folder that it moved. So the last start asks for nothing more."""
    tree = Tree(tmp_path, splits=("test",))
    code, _, _ = tree.run(FAKE_FAIL="base_test:7")
    assert code == 7
    code, markers, _ = tree.run(FAKE_FAIL="base_test:7", REPEAT_AFTER_FAULT="1", REPEAT_REASON="a first fault")
    assert code == 7 and "STEP_MOVED_ASIDE base_test results/main/base_test.fault_1" in markers
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="a second fault")
    assert code == 0, tree.output
    assert "STEP_MOVED_ASIDE base_test results/main/base_test.fault_2" in markers
    for n in (1, 2):
        assert read_json(tree.main / f"base_test.fault_{n}" / "run_manifest.json")["complete"] is False
    lines = tree.ledger()
    head = "REPEAT_AFTER_FAULT arm=base folder=results/main/base_test state=unfinished moving_to="
    assert_lines(lines[:8], [
        started(tree.root, "base"), re.escape(ended("base", 7, "unfinished")),
        re.escape(head + "results/main/base_test.fault_1 reason=a first fault"),
        started(tree.root, "base"), re.escape(ended("base", 7, "unfinished")),
        re.escape(head + "results/main/base_test.fault_2 reason=a second fault"),
        started(tree.root, "base"), re.escape(ended("base"))])
    assert not any(" aside=" in line for line in lines)
    assert lines[-4].endswith(" moved_aside=results/main/base_test.fault_1,results/main/base_test.fault_2")


def test_a_rollout_that_the_record_names_and_whose_folder_is_gone_is_not_run_again_without_a_reason(tmp_path):
    """The machine is lost while the second arm runs. On the new machine are the complete folder of the
    first arm and the record, as they were fetched; the unfinished folder was lost with the machine. A
    plain start does not run the second arm as if for the first time: the record says that its rollout
    was started."""
    tree = Tree(tmp_path, splits=("test",))
    code, _, _ = tree.run(FAKE_FAIL="sft_test:7")
    assert code == 7
    record = tree.main / "test_run_record.txt"
    kept = record.read_text(encoding="utf-8").splitlines()
    assert " ROLLOUT_END arm=sft " in kept[-1]
    put(record, "\n".join(kept[:-1]) + "\n")          # the machine was lost before the rollout ended
    shutil.rmtree(tree.main / "sft_test")
    files = files_of(tree.root)
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "results/main/sft_test: NOT there. But results/main/test_run_record.txt says that a rollout of " \
           "this arm was started, and no later line says that it wrote nothing or why the arm is repeated:" \
        in tree.output
    assert "A reason is needed for: sft." in tree.output
    assert "A run folder that is only in another place (another machine, the laptop) is brought back" in \
        tree.output
    assert files_of(tree.root) == files
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="the machine was lost in sft")
    assert code == 0, tree.output
    assert not [line for line in markers if line.startswith("STEP_MOVED_ASIDE")]      # nothing to move
    assert [value_of(call, "--arm") for call in calls if call["script"] == "rollout"] == \
        ["sft", "grpo_seed1", "grpo_seed2"]
    lines = tree.ledger()
    assert lines[4] == ("REPEAT_AFTER_FAULT arm=sft folder=results/main/sft_test state=not_there "
                        "reason=the machine was lost in sft")
    assert_lines(lines[5:8], a_rollout(tree.root, "sft"))
    # The same with the line that the rollout ended and left an unfinished folder: only a rollout that
    # wrote nothing is no test attempt.
    other = Tree(tmp_path / "other", splits=("test",))
    other.run(FAKE_FAIL="sft_test:7")
    shutil.rmtree(other.main / "sft_test")
    code, markers, calls = other.run()
    assert (code, markers[-1], scripts_of(calls)) == (1, "CHAIN_FAILED run_once", ["freeze"])


def test_an_earlier_folder_beside_a_run_folder_that_no_line_names_needs_a_reason_too(tmp_path, real_run):
    """A folder <arm>_test.<name> is an earlier run of the arm. Here the record does not know it (the
    record of the machine that made it was not brought along)."""
    tree = Tree(tmp_path, splits=("test",))
    put(tree.main / "grpo_seed1_test.fault_1" / "run_manifest.json", '{"complete": false}')
    files = files_of(tree.root)
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "results/main/grpo_seed1_test: NOT there. But beside it lies an earlier folder of this arm that " \
           "no line of results/main/test_run_record.txt names: results/main/grpo_seed1_test.fault_1" in \
        tree.output
    assert "A reason is needed for: grpo_seed1." in tree.output and files_of(tree.root) == files
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="lost on the first machine")
    assert code == 0, tree.output
    assert tree.ledger()[0] == ("REPEAT_AFTER_FAULT arm=grpo_seed1 folder=results/main/grpo_seed1_test "
                                "state=not_there aside=results/main/grpo_seed1_test.fault_1 "
                                "reason=lost on the first machine")
    # A name with a space cannot be named in the record: the chain asks for a plain name.
    spaced = Tree(tmp_path / "spaced", splits=("test",))
    (spaced.main / "sft_test.old copy").mkdir()
    code, markers, calls = spaced.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="a copy")
    assert (code, markers[-1], scripts_of(calls)) == (1, "CHAIN_FAILED run_once", ["freeze"])
    assert "A name beside results/main/sft_test holds a space or a comma." in spaced.output
    # Beside a COMPLETE folder such a folder starts no rollout, so the chain goes on; it says what it saw,
    # and the last lines of the record name the folder.
    done = Tree(tmp_path / "done", source=real_run.tree.root)
    (done.main / "sft_test.copy").mkdir()
    code, markers, calls = done.run()
    assert code == 0 and scripts_of(calls) == ["freeze"] + REPORT_SCRIPTS, done.output
    assert "    Beside it, and named by no line of results/main/test_run_record.txt: " \
           "results/main/sft_test.copy" in done.output
    assert done.ledger()[-3].endswith(" moved_aside=results/main/sft_test.copy")


def test_a_run_manifest_that_says_complete_and_cannot_be_read_is_never_moved_aside(tmp_path):
    """`is_complete` reads the run manifest with Python. Here that reading fails for a folder whose manifest
    holds "complete": true (the file is cut off, so it is not JSON), and the folder counts as unfinished. A
    start with REPEAT_AFTER_FAULT does not move it, and it does not move the other unfinished folder."""
    tree = Tree(tmp_path, splits=("test",))
    put(tree.main / "base_test" / "run_manifest.json", '{"complete": false}')
    put(tree.main / "sft_test" / "run_manifest.json", '{\n  "complete": true,\n  "command": [\n')
    files = files_of(tree.root)
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="the machine was lost")
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "results/main/sft_test: its run manifest could not be read as that of a complete run, but it " \
           'holds the words "complete": true. No folder was moved.' in tree.output
    assert files_of(tree.root) == files and not list(tree.main.glob("*.fault_*"))


def test_a_complete_folder_that_another_run_made_stops_the_chain_and_stays(tmp_path, real_run):
    """A complete folder whose run manifest names another value than this chain has is not the test run of
    the plan. The chain does not go on with it, does not run the arm again and does not move the folder."""
    for key, value, shown in (("run_seed", 3, "sft_test: run_seed is '3'"),
                              ("split", "dev", "sft_test: split is 'dev'"),
                              ("code_sha256", "0" * 64, "sft_test: code_sha256 is '0000"),
                              ("allow_changed_tree", True, "sft_test: allow_changed_tree is 'True'"),
                              ("prompt", "plain", "sft_test: prompt is 'plain'")):
        tree = Tree(tmp_path / key, source=real_run.tree.root)
        manifest_file = tree.main / "sft_test" / "run_manifest.json"
        write_json(manifest_file, {**read_json(manifest_file), key: value})
        files = files_of(tree.root)
        code, markers, calls = tree.run(REPEAT_AFTER_FAULT="0")
        assert code == 1 and scripts_of(calls) == ["freeze"], tree.output
        assert markers == ran(START) + ["STEP_SKIPPED base_test"] + ran(["gate_base"]) + \
            ["CHAIN_FAILED sft_test"]
        assert shown in tree.output
        assert "results/main/sft_test is complete, but it is not the run that this chain would make" in \
            tree.output
        assert files_of(tree.root) == files


# ---------------------------------------------------------------------------
# The stops: the frozen tree, the adapters, the gate, a failed step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("change, code, shown", [
    ("a changed frozen file", 1, "changed: data/scenarios/test.jsonl (group scenarios)"),
    ("a new file under a frozen pattern", 1, "new: data/scenarios/more.jsonl (group scenarios)"),
    ("no manifest", 2, "MANIFEST.json is missing: the tree is not frozen"),
])
def test_a_tree_that_is_not_as_frozen_stops_the_chain_before_anything_else(tmp_path, change, code, shown):
    """The real scripts/freeze.py checks the made-up tree, in both modes. Nothing else is started, and
    nothing is written."""
    tree = Tree(tmp_path)
    if change == "a changed frozen file":
        with (tree.root / "data" / "scenarios" / "test.jsonl").open("ab") as handle:
            handle.write(b"\n")
    elif change == "a new file under a frozen pattern":
        put(tree.root / "data" / "scenarios" / "more.jsonl", "{}\n")
    else:
        (tree.root / "MANIFEST.json").unlink()
    files = files_of(tree.root)
    for dress in ("0", "1"):
        found, markers, calls = tree.run(DRESS=dress)
        assert (found, markers, scripts_of(calls)) == (code, stopped_at("freeze_check", code), ["freeze"])
        assert shown in tree.output and tree.notes == []
        assert files_of(tree.root) == files and not (tree.root / "results" / "_smoke").exists()


def another_hash(tree: Tree, arm: str, name: str) -> None:
    """The note of the arm's dev run names another SHA-256 for a file of the adapter."""
    path = tree.main / f"{arm}_dev" / "chain_record.txt"
    old = sha256_of(tree.root / ADAPTERS[arm] / name)
    put(path, path.read_text(encoding="utf-8").replace(f"{name}={old}", f"{name}={'0' * 64}"))


def another_value(tree: Tree, arm: str, key: str, value: object) -> None:
    """The run manifest of the arm's dev run names another value."""
    path = tree.main / f"{arm}_dev" / "run_manifest.json"
    write_json(path, {**read_json(path), key: value})


def another_run(tree: Tree, arm: str, **values: object) -> None:
    """The run manifest of the arm's training run names other values. A key with a dot is a key of
    `final_adapter`."""
    path = tree.main / arm / "run_manifest.json"
    manifest = read_json(path)
    for key, value in values.items():
        if key.startswith("final_"):
            manifest["final_adapter"][key.removeprefix("final_")] = value
        else:
            manifest[key] = value
    write_json(path, manifest)


def another_training(tree: Tree, arm: str, keys: tuple[str, ...]) -> None:
    """The run manifest of the arm's training run names another SHA-256 for the weights of the adapter."""
    path = tree.main / arm / "run_manifest.json"
    manifest = read_json(path)
    entry = manifest
    for key in keys:
        entry = entry[key]
    entry["adapter_model.safetensors"]["sha256"] = "0" * 64
    write_json(path, manifest)


# Each case: what is wrong on the machine, and words of the message. Every case stops at the step adapters.
WRONG_ADAPTERS = {
    "the weights on disk are not those of the dev run and of the training run": (
        lambda tree: put(tree.root / ADAPTERS["grpo_seed1"] / "adapter_model.safetensors", "other weights\n"),
        ["grpo_seed1_dev: what the chain noted in results/main/grpo_seed1_dev/chain_record.txt",
         "grpo_seed1: results/main/grpo_seed1_dev is not the dev run of what this chain would run",
         "and the run manifest of the training run names '"]),
    "the config of the adapter on disk is another": (
        lambda tree: put(tree.root / ADAPTERS["sft"] / "adapter_config.json", '{"r": 1}\n'),
        ["sft_dev: what this chain has now:",
         "sft: results/main/sft/adapter/adapter_config.json has the SHA-256"]),
    "the note of the dev run names other weights": (
        lambda tree: another_hash(tree, "sft", "adapter_model.safetensors"),
        ["sft_dev: what the chain noted in results/main/sft_dev/chain_record.txt",
         "adapter_model.safetensors=" + "0" * 64,
         "the adapter in 'results/main/sft/adapter' is not the one that the dev rollout was started with"]),
    "the training run wrote other weights": (
        lambda tree: another_training(tree, "sft", ("outputs", "adapter")),
        ["sft: results/main/sft/adapter/adapter_model.safetensors has the SHA-256 '",
         "names '" + "0" * 64 + "'"]),
    "the last checkpoint of the GRPO run has other weights": (
        lambda tree: another_training(tree, "grpo_seed2", ("final_adapter", "files")),
        ["grpo_seed2: results/main/grpo_seed2/checkpoints/step_00080/adapter/adapter_model.safetensors has"]),
    "the run manifest of a dev run is missing": (
        lambda tree: (tree.main / "base_dev" / "run_manifest.json").unlink(),
        ["base: the dev run of this arm is not here: results/main/base_dev/run_manifest.json is missing"]),
    "a dev run is not complete": (
        lambda tree: another_value(tree, "grpo_seed2", "complete", False),
        ["grpo_seed2: the dev run of this arm is not here"]),
    "the note of a dev run is missing": (
        lambda tree: (tree.main / "sft_dev" / "chain_record.txt").unlink(),
        ["sft: results/main/sft_dev/chain_record.txt is missing"]),
    "the run manifest of a training run is missing": (
        lambda tree: (tree.main / "grpo_seed1" / "run_manifest.json").unlink(),
        ["the run manifest of the training run names ''"]),
    "the weights are not on the machine": (
        lambda tree: (tree.root / ADAPTERS["grpo_seed2"] / "adapter_model.safetensors").unlink(),
        ["adapter_model.safetensors has the SHA-256 ''"]),
    "the dev run had another run seed": (
        lambda tree: another_value(tree, "base", "run_seed", 1),
        ["base_dev: run_seed is '1' in results/main/base_dev/run_manifest.json; this chain has '0'"]),
    "the dev run loaded another adapter folder": (
        lambda tree: another_value(tree, "grpo_seed1", "adapter", ADAPTERS["grpo_seed1"].replace("80", "70")),
        ["grpo_seed1_dev: adapter is 'results/main/grpo_seed1/checkpoints/step_00070/adapter'"]),
    "the dev run was made by other code": (
        lambda tree: another_value(tree, "sft", "code_sha256", "f" * 64),
        ["sft_dev: code_sha256 is 'ffff"]),
    "the dev run was made on a changed tree": (
        lambda tree: another_value(tree, "sft", "allow_changed_tree", True),
        ["sft_dev: allow_changed_tree is 'True'"]),
    "the dev run had another prompt": (
        lambda tree: another_value(tree, "base", "prompt", "plain"),
        ["base_dev: prompt is 'plain'"]),
    # The training run that wrote the adapter: the SHA-256 of the files are the right ones in each case.
    "the stop rule ended the training run": (
        lambda tree: another_run(tree, "grpo_seed1", complete=False, status="stopped_by_rule"),
        ["grpo_seed1: the training run in results/main/grpo_seed1 is not a complete run: its run manifest "
         "is missing, or it does not say \"complete\": true (its status: 'stopped_by_rule')"]),
    "the training run has another status than complete": (
        lambda tree: another_run(tree, "grpo_seed2", status="failed"),
        ["grpo_seed2: the training run in results/main/grpo_seed2 has the status 'failed', not 'complete'"]),
    "the SFT run has a status that only a GRPO run can have": (
        lambda tree: another_run(tree, "sft", status="complete_after_fault"),
        ["sft: the training run in results/main/sft has the status 'complete_after_fault', not 'complete'"]),
    "the final adapter of the training run is another checkpoint": (
        lambda tree: another_run(tree, "grpo_seed2", final_folder="checkpoints/step_00060/adapter"),
        ["grpo_seed2: the run manifest of the training run names 'checkpoints/step_00060/adapter' as its "
         "final adapter; this chain loads 'results/main/grpo_seed2/checkpoints/step_00080/adapter'"]),
}


# The cases that a rehearsal is given too: the step is the same code in both modes.
ALSO_IN_A_REHEARSAL = ("the weights on disk are not those of the dev run and of the training run",
                       "the note of the dev run names other weights", "the training run wrote other weights",
                       "the run manifest of a dev run is missing")
ADAPTER_RUNS = [(case, "0") for case in WRONG_ADAPTERS] + [(case, "1") for case in ALSO_IN_A_REHEARSAL]


@pytest.mark.parametrize("case, dress", ADAPTER_RUNS,
                         ids=[f"{case} (DRESS={dress})" for case, dress in ADAPTER_RUNS])
def test_an_adapter_or_a_record_that_is_not_the_one_of_the_dev_run_stops_the_chain(tmp_path, case, dress):
    """The chain stops at the step adapters: before any rollout, and without writing anything."""
    spoil, shown = WRONG_ADAPTERS[case]
    tree = Tree(tmp_path)
    spoil(tree)
    files = files_of(tree.root)
    code, markers, calls = tree.run(DRESS=dress)
    assert (code, markers, scripts_of(calls)) == \
        (1, ran(["freeze_check"]) + stopped_at("adapters"), ["freeze"]), tree.output
    for words in shown:
        assert words in tree.output, words
    assert "must be on this machine as they are on the machine that holds the results. Nothing was " \
           "changed." in tree.output
    assert files_of(tree.root) == files and not (tree.root / "results" / "_smoke").exists()


def test_a_grpo_run_that_was_started_again_after_a_stop_is_taken_and_a_note_says_so(tmp_path):
    """The plan, section 7: such a run has the status complete_after_fault, and the write-up must say so."""
    tree = Tree(tmp_path, splits=("test",))
    another_run(tree, "grpo_seed1", status="complete_after_fault")
    code, markers, calls = tree.run()
    assert code == 0, tree.output
    note = ("CHAIN_NOTE the training run in results/main/grpo_seed1 has the status complete_after_fault: it "
            "was started again after the stop rule had ended it. The write-up must say so (the plan, "
            "section 7).")
    assert tree.notes.count(note) == 3          # at the step adapters, before the rollout, and at the end


def skipped(*arms: str) -> list[str]:
    """The marker lines of arms whose complete run was taken, with their gate."""
    return [line for arm in arms for line in (f"STEP_SKIPPED {arm}_test", *ran([f"gate_{arm}"]))]


def test_an_adapter_that_changes_while_its_rollout_runs_marks_that_run_and_stops_the_chain(tmp_path):
    """The weights of the first GRPO seed are other bytes when its rollout ends. The chain hashes the two
    adapter files again right after each rollout. So it stops at once, the record and the run folder say
    what happened, and that folder is never taken as the run of the arm: not after the weights are put
    back, and not with any variable."""
    tree = Tree(tmp_path, splits=("test",))
    weights = tree.root / ADAPTERS["grpo_seed1"] / "adapter_model.safetensors"
    good = weights.read_bytes()
    code, markers, calls = tree.run(FAKE_CHANGE_ADAPTER_IN="grpo_seed1_test")
    assert code == 1
    assert markers == ran(START + rollouts("test")[:5]) + ["CHAIN_FAILED grpo_seed1_test"]
    assert scripts_of(calls) == ["freeze"] + ["rollout"] * 3            # no fourth rollout, and no report
    folder = tree.main / "grpo_seed1_test"
    assert read_json(folder / "run_manifest.json")["complete"] is True
    assert not (folder / "chain_record.txt").exists()                    # the chain did not note this run
    before, after = hashlib.sha256(good).hexdigest(), sha256_of(weights)
    marker = (folder / "adapter_changed.txt").read_text(encoding="utf-8").splitlines()
    assert re.fullmatch(rf"{TIME} the adapter of the arm grpo_seed1 changed while this rollout ran\.", marker[0])
    assert marker[1].endswith(f"adapter_model.safetensors={before}") and marker[1].startswith("before: ")
    assert marker[2].endswith(f"adapter_model.safetensors={after}") and marker[2].startswith("after:  ")
    lines = tree.ledger()
    assert_lines(lines[:8], a_rollout(tree.root, "base") + a_rollout(tree.root, "sft")
                 + [rf"ROLLOUT_START arm=grpo_seed1 .* adapter_model\.safetensors={before}",
                    re.escape(ended("grpo_seed1"))])
    assert re.fullmatch(r"ADAPTER_CHANGED arm=grpo_seed1 folder=results/main/grpo_seed1_test before: "
                        rf"adapter=\S+ adapter_config\.json=\w+ adapter_model\.safetensors={before} after: "
                        rf"adapter=\S+ adapter_config\.json=\w+ adapter_model\.safetensors={after}", lines[8])
    assert len(lines) == 9
    assert "The adapter of grpo_seed1 changed while its rollout ran:" in tree.output

    # A second start with the changed weights stops at the first step adapters.
    code, markers, calls = tree.run()
    assert (code, markers) == (1, ran(["freeze_check"]) + stopped_at("adapters")) and len(calls) == 1
    # The right weights are put back. The run folder is still not taken: it holds the mark.
    weights.write_bytes(good)
    runs = run_folders(tree)
    code, markers, calls = tree.run()
    assert (code, scripts_of(calls)) == (1, ["freeze"])
    assert markers == ran(START) + skipped("base", "sft") + ["CHAIN_FAILED grpo_seed1_test"]
    assert "results/main/grpo_seed1_test is complete, but it holds adapter_changed.txt:" in tree.output
    # Without the mark it is still not taken: it has no note of the chain.
    (folder / "adapter_changed.txt").rename(tmp_path / "kept_mark.txt")
    code, markers, calls = tree.run()
    assert (code, markers[-1], scripts_of(calls)) == (1, "CHAIN_FAILED grpo_seed1_test", ["freeze"])
    assert "results/main/grpo_seed1_test is complete, but it has no chain_record.txt." in tree.output
    (tmp_path / "kept_mark.txt").rename(folder / "adapter_changed.txt")
    # No variable makes the chain move or run a complete folder.
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="the adapter changed")
    assert (code, markers[-1], scripts_of(calls)) == (1, "CHAIN_FAILED run_once", ["freeze"])
    assert run_folders(tree) == runs and not list(tree.main.glob("*.fault_*"))

    # A person moves the folder away by hand. A plain start does not run the arm: the record names its
    # rollout. With the reason the record gets the line, with the folder that lies beside, and the arm runs.
    folder.rename(tree.main / "grpo_seed1_test.fault_1")
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "A reason is needed for: grpo_seed1." in tree.output
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1",
                                    REPEAT_REASON="a copy job wrote into the adapter folder during the run")
    assert code == 0, tree.output
    assert [value_of(call, "--arm") for call in calls if call["script"] == "rollout"] == \
        ["grpo_seed1", "grpo_seed2"]
    lines = tree.ledger()
    assert lines[9] == ("REPEAT_AFTER_FAULT arm=grpo_seed1 folder=results/main/grpo_seed1_test "
                        "state=not_there aside=results/main/grpo_seed1_test.fault_1 "
                        "reason=a copy job wrote into the adapter folder during the run")
    assert_lines(lines[-6:], block(tree.root, "TEST_RUN_DONE", "grpo_seed1,grpo_seed2",
                                   aside={"grpo_seed1": "results/main/grpo_seed1_test.fault_1"}))
    assert (tree.main / "grpo_seed1_test.fault_1" / "adapter_changed.txt").is_file()


def test_a_changed_adapter_of_a_later_arm_stops_the_chain_before_that_arm_is_sampled(tmp_path):
    """The weights of the last arm change while the first arm runs, more than an hour before their own
    rollout on the machine. The chain compares the adapter of an arm with its dev run once more right
    before its rollout: that rollout is not started, and the record names no start of it."""
    tree = Tree(tmp_path, splits=("test",))
    weights = ADAPTERS["grpo_seed2"] + "/adapter_model.safetensors"
    good = (tree.root / weights).read_bytes()
    code, markers, calls = tree.run(FAKE_CHANGE_ADAPTER_IN=f"base_test:{weights}")
    assert code == 1
    assert markers == ran(START + rollouts("test")[:6]) + ["CHAIN_FAILED grpo_seed2_test"]
    assert scripts_of(calls) == ["freeze"] + ["rollout"] * 3
    assert "The rollout of grpo_seed2 was not started: nothing was sampled." in tree.output
    assert not (tree.main / "grpo_seed2_test").exists()
    assert not [line for line in tree.ledger() if " arm=grpo_seed2 " in line]
    # With the right weights back, a plain start runs the last arm: it was never started.
    (tree.root / weights).write_bytes(good)
    code, markers, calls = tree.run()
    assert code == 0, tree.output
    assert [value_of(call, "--arm") for call in calls if call["script"] == "rollout"] == ["grpo_seed2"]
    assert tree.ledger()[-6] == "TEST_RUN_DONE DRESS=0 split=test out=results/main rollouts_run=grpo_seed2"


def test_a_file_that_changes_after_the_rollout_of_its_arm_stops_the_chain_before_the_reports(tmp_path):
    """The step adapters runs once more after the last rollout. Here the weights of the arm sft change
    while the last arm runs. Each rollout was checked right before and right after it ran, so no run is in
    doubt; but no report is built while a file on disk is not the one of the dev runs, and the record says
    that the chain stopped there."""
    tree = Tree(tmp_path, splits=("test",))
    weights = ADAPTERS["sft"] + "/adapter_model.safetensors"
    good = (tree.root / weights).read_bytes()
    code, markers, calls = tree.run(FAKE_CHANGE_ADAPTER_IN=f"grpo_seed2_test:{weights}")
    assert code == 1
    assert markers == ran(START + rollouts("test")) + stopped_at("adapters_after")
    assert scripts_of(calls) == ["freeze"] + ["rollout"] * 4
    assert "sft_dev: what the chain noted in results/main/sft_dev/chain_record.txt" in tree.output
    assert not list((tree.main / "tables").iterdir())
    lines = tree.ledger()
    assert len(lines) == 13 and lines[12] == \
        "ADAPTERS_AFTER_FAILED the adapter files or their records are not those of the dev runs"
    # The note of each run names the weights that were on disk when its rollout was started.
    for arm in ARMS:
        assert (tree.main / f"{arm}_test" / "chain_record.txt").read_text(encoding="utf-8") == \
            (tree.main / f"{arm}_dev" / "chain_record.txt").read_text(encoding="utf-8")
    # A second start stops at once, at the first step adapters. With the right file back the four runs are
    # taken, the reports are built, and the record gets its TEST_RUN_DONE: no start had reached the end.
    code, markers, calls = tree.run()
    assert (code, markers) == (1, ran(["freeze_check"]) + stopped_at("adapters")) and len(calls) == 1
    (tree.root / weights).write_bytes(good)
    code, markers, calls = tree.run()
    assert code == 0 and scripts_of(calls) == ["freeze"] + REPORT_SCRIPTS, tree.output
    assert markers == ran(START) + skipped(*ARMS) + ran(AFTER + REPORTS) + ["CHAIN_DONE"]
    assert_lines(tree.ledger()[13:], block(tree.root, "TEST_RUN_DONE", "none"))
    code, markers, calls = tree.run()
    assert code == 0 and tree.ledger()[19].startswith("REPORTS_REBUILT DRESS=0 split=test ")


def test_the_step_adapters_says_what_it_found_for_each_arm(real_run):
    """Three times for each arm: at the step adapters, right before the rollout of the arm, and at the
    step adapters_after."""
    for arm in ARMS:
        assert real_run.output.count(f"{arm}: results/main/{arm}_dev is the dev run of this arm: its run "
                                     "manifest and its chain_record.txt fit this chain") == 3
    for arm in ARMS[1:]:
        for name in ADAPTER_FILES:
            digest = sha256_of(real_run.tree.root / ADAPTERS[arm] / name)
            assert real_run.output.count(f"{arm}: {ADAPTERS[arm]}/{name} has the SHA-256 that the training "
                                         f"run wrote: {digest}") == 3
    assert "not compared" not in real_run.output
    # The look right before a rollout comes after the gate of the arm before it, and before the start.
    output = real_run.output
    last = output.rindex("sft: results/main/sft_dev is the dev run of this arm", 0,
                         output.index("STEP_START sft_test"))
    assert output.index("STEP_DONE gate_base 0") < last


def test_the_gate_stops_the_chain_at_one_percent_and_the_record_says_so(tmp_path):
    """The made-up test split gives 200 attempts per arm. One harness error is 0.5%: the gate is passed. Two
    are exactly 1%: the plan asks for "below 1%", so the chain stops after that rollout."""
    tree = Tree(tmp_path, splits=("test",))
    code, markers, calls = tree.run(FAKE_ERRORS_base="1", FAKE_ERRORS_sft="2", FAKE_ERRORS_grpo_seed1="0")
    assert code == 1
    assert markers == ran(START + ["base_test", "gate_base", "sft_test"]) + stopped_at("gate_sft")
    assert scripts_of(calls) == ["freeze", "rollout", "rollout"]        # no third rollout, and no report
    base_counts = "attempts=200 harness_errors=1 harness_error_rate=0.005"
    sft_counts = "attempts=200 harness_errors=2 harness_error_rate=0.01"
    for arm, counts in (("base", base_counts), ("sft", sft_counts)):
        assert f"results/main/{arm}_test/run_manifest.json: {counts}" in tree.output
    assert "The gate is not passed: harness_error_rate must be below 0.01" in tree.output
    assert "This chain never runs a complete folder again and never moves one." in tree.output
    assert "it is a decision for the main session" in tree.output
    assert read_json(tree.main / "sft_test" / "run_manifest.json")["complete"] is True
    assert not (tree.main / "grpo_seed1_test").exists()
    # The record: the rollout of sft was started, it ended, and its gate failed, with the counts.
    failed_gate = gate_line("sft", "GATE_FAILED", sft_counts)
    assert_lines(tree.ledger(), a_rollout(tree.root, "base", counts=base_counts)
                 + [started(tree.root, "sft"), re.escape(ended("sft")), re.escape(failed_gate)])
    runs = run_folders(tree)
    # A second start stops at the same gate: the run is complete, so it is not made again, and no variable
    # changes that. The record gets the line of the gate once more.
    code, markers, calls = tree.run()
    assert (code, scripts_of(calls)) == (1, ["freeze"])
    assert markers == ran(START) + ["STEP_SKIPPED base_test"] + ran(["gate_base"]) + \
        ["STEP_SKIPPED sft_test"] + stopped_at("gate_sft")
    assert tree.ledger()[6:] == [failed_gate]
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON="the gate was not passed")
    assert (code, markers[-1], scripts_of(calls)) == (1, "CHAIN_FAILED run_once", ["freeze"])
    assert run_folders(tree) == runs and not list(tree.main.glob("*.fault_*"))
    assert len(tree.ledger()) == 7


@pytest.mark.parametrize("moved", ["beside the run folder", "to another place", "deleted"])
def test_a_complete_run_that_failed_the_gate_and_was_moved_by_hand_is_not_run_again_without_a_reason(
        tmp_path, moved):
    """The chain never moves a complete folder, so after a failed gate a person does it. Wherever the
    folder went, the record still names its rollout: a plain start stops, and the start with a reason
    writes that reason into the record before the arm runs again."""
    tree = Tree(tmp_path, splits=("test",))
    code, markers, calls = tree.run(FAKE_ERRORS_sft="2")
    assert (code, markers[-1]) == (1, "CHAIN_FAILED gate_sft")
    aside = "none"
    if moved == "beside the run folder":
        (tree.main / "sft_test").rename(tree.main / "sft_test.fault_1")
        aside = "results/main/sft_test.fault_1"
    elif moved == "to another place":
        (tree.main / "sft_test").rename(tmp_path / "old_sft_test")
    else:
        shutil.rmtree(tree.main / "sft_test")
    files = files_of(tree.root)
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START[:2]) + stopped_at("run_once"), ["freeze"])
    assert "results/main/sft_test: NOT there. But results/main/test_run_record.txt says that a rollout of " \
           "this arm was started" in tree.output
    assert "A reason is needed for: sft." in tree.output and files_of(tree.root) == files
    reason = "16 harness errors in sft; the main session decided on the repeat"
    code, markers, calls = tree.run(REPEAT_AFTER_FAULT="1", REPEAT_REASON=reason)
    assert code == 0, tree.output
    assert not [line for line in markers if line.startswith("STEP_MOVED_ASIDE")]
    assert [value_of(call, "--arm") for call in calls if call["script"] == "rollout"] == \
        ["sft", "grpo_seed1", "grpo_seed2"]
    lines = tree.ledger()
    named = "" if aside == "none" else f" aside={aside}"
    assert lines[5].startswith("GATE_FAILED arm=sft ")
    assert lines[6] == (f"REPEAT_AFTER_FAULT arm=sft folder=results/main/sft_test state=not_there{named} "
                        f"reason={reason}")
    assert_lines(lines[-6:], block(tree.root, "TEST_RUN_DONE", "sft,grpo_seed1,grpo_seed2",
                                   aside={"sft": aside}))


@pytest.mark.parametrize("summary, passed", [
    ({"episodes": 1536, "harness_errors": 15, "harness_error_rate": 15 / 1536}, True),      # 0.98%
    ({"episodes": 1536, "harness_errors": 16, "harness_error_rate": 16 / 1536}, False),     # 1.04%
    ({"episodes": 1600, "harness_errors": 16, "harness_error_rate": 16 / 1600}, False),     # exactly 1%
    ({"episodes": 1536, "harness_errors": 0, "harness_error_rate": 0}, True),
    ({"episodes": 0, "harness_errors": 0, "harness_error_rate": None}, False),              # no attempt
    (None, False),                                                                          # no summary
])
def test_the_gate_reads_the_rate_that_the_rollout_wrote_into_its_run_manifest(tmp_path, real_run, summary,
                                                                              passed):
    """The numbers of a run of the real size (192 scenarios x 8 attempts = 1,536): 15 harness errors pass,
    16 do not. The summary of the last arm's complete run is replaced by hand."""
    tree = Tree(tmp_path, source=real_run.tree.root)
    path = tree.main / "grpo_seed2_test" / "run_manifest.json"
    manifest = read_json(path)
    if summary is None:
        del manifest["summary"]
    else:
        manifest["summary"] = summary
    write_json(path, manifest)
    code, markers, calls = tree.run()
    if passed:
        assert code == 0 and markers[-1] == "CHAIN_DONE", tree.output
        assert tree.record()[-1].endswith(
            f"attempts={summary['episodes']} harness_errors={summary['harness_errors']} "
            f"harness_error_rate={summary['harness_error_rate']} moved_aside=none")
    else:
        assert (code, scripts_of(calls)) == (1, ["freeze"]), tree.output
        assert markers[-3:] == stopped_at("gate_grpo_seed2")
        lines = tree.ledger()                                            # no second block in the record
        assert len(lines) == 19 and lines[18].startswith(
            "GATE_FAILED arm=grpo_seed2 folder=results/main/grpo_seed2_test ")
        assert lines[18].endswith("the summary could not be read" if summary is None else
                                  f"harness_error_rate={summary['harness_error_rate']}")
        if summary is None or summary["harness_error_rate"] is None:
            shown = "its summary could not be read" if summary is None else "harness_error_rate=None"
            assert shown in tree.output


def test_the_chain_stops_at_the_first_failed_step_with_its_exit_code(tmp_path):
    """One tree, started again after each fault. A report that fails stops the chain too: the four runs are
    saved by then, and the next start builds every report again."""
    tree = Tree(tmp_path, splits=("test",))
    # A rollout that fails before it makes its folder (as the real script does for a wrong argument): the
    # next start runs the arm; nothing was sampled, so there is nothing to repeat.
    code, markers, calls = tree.run(FAKE_FAIL="base_test:2:before")
    assert (code, markers) == (2, ran(START) + stopped_at("base_test", 2))
    assert not (tree.main / "base_test").exists()
    # The record says that this rollout wrote nothing, so the next start needs no reason for the arm.
    assert_lines(tree.ledger(), [started(tree.root, "base"), re.escape(ended("base", 2, "nothing"))])
    # The replay check of the second arm finds differences: no table is built on these runs.
    code, markers, calls = tree.run(FAKE_FAIL="replay_check:1:sft_test")
    assert code == 1
    assert markers == ran(START + rollouts("test") + AFTER + ["replay_base"]) + stopped_at("replay_sft")
    assert scripts_of(calls) == ["freeze"] + ["rollout"] * 4 + ["replay_check"] * 2
    assert not list((tree.main / "tables").iterdir())
    assert_lines(tree.ledger()[2:], [pattern for arm in ARMS for pattern in a_rollout(tree.root, arm)])
    runs = run_folders(tree)
    skipped = [line for arm in ARMS for line in (f"STEP_SKIPPED {arm}_test", *ran([f"gate_{arm}"]))]
    # A replay check that could not be made (exit code 2 of the real script) stops the chain in the same way.
    code, markers, calls = tree.run(FAKE_FAIL="replay_check:2")
    assert (code, markers) == (2, ran(START) + skipped + ran(AFTER) + stopped_at("replay_base", 2))
    # The table fails: no monitor report and no reference row follow.
    code, markers, calls = tree.run(FAKE_FAIL="tables:4")
    assert code == 4
    assert markers == ran(START) + skipped + ran(AFTER + REPORTS[:4]) + stopped_at("table", 4)
    assert scripts_of(calls) == ["freeze"] + ["replay_check"] * 4 + ["tables"]
    # A monitor report fails: no half-written file is left.
    code, markers, calls = tree.run(FAKE_FAIL="monitor_report:2:grpo_seed1_test")
    assert code == 2
    assert markers == ran(START) + skipped + ran(AFTER + REPORTS[:7]) + stopped_at("monitors_grpo_seed1", 2)
    assert sorted(path.name for path in (tree.main / "monitors").iterdir()) == ["base_test.md", "sft_test.md"]
    # A reference row fails.
    code, markers, calls = tree.run(FAKE_FAIL="run_scripted:3")
    assert (code, markers[-3:]) == (3, stopped_at("reference_oracle", 3))
    assert len(tree.ledger()) == 14                  # the lines of the rollouts, and no line of an end yet
    # A start without a fault ends the chain. No arm was run a second time on the way. The record gets
    # its TEST_RUN_DONE now: it is the first start that reached the end.
    code, markers, calls = tree.run()
    assert code == 0 and markers == ran(START) + skipped + ran(AFTER + REPORTS) + ["CHAIN_DONE"]
    assert run_folders(tree) == runs
    assert_lines(tree.ledger()[14:], block(tree.root, "TEST_RUN_DONE", "none"))


def test_a_needed_file_that_is_missing_stops_the_chain_before_the_adapters(tmp_path):
    """The split file of the mode is needed, and the other split is not: a test run needs no calibration
    file, and a rehearsal needs no test file."""
    tree = Tree(tmp_path, splits=("test",))
    stopped = (1, ran(["freeze_check"]) + ["CHAIN_FAILED start"], ["freeze"])
    code, markers, calls = tree.run(DRESS="1")
    assert (code, markers, scripts_of(calls)) == stopped
    assert "data/scenarios/calib.jsonl is missing" in tree.output
    assert not (tree.root / "results" / "_smoke").exists()
    other = Tree(tmp_path / "other", splits=("calib",))
    code, markers, calls = other.run()
    assert (code, markers, scripts_of(calls)) == stopped
    assert "data/scenarios/test.jsonl is missing" in other.output


# ---------------------------------------------------------------------------
# One chain at a time; the note of the chain in a run folder; the file of the machine; the reference rows
# ---------------------------------------------------------------------------
def wait_for(condition, seconds: float = 60.0) -> bool:
    """Ask `condition` until it is true, at most for this long."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def test_a_second_chain_stops_at_the_lock_while_the_first_one_runs(tmp_path):
    """The first chain is inside its first rollout, at the place where the real script loads the model:
    it has looked at its folder and has not made it yet. Without the lock a second start would find no
    folder and sample the test set too. With it the second start does nothing at all: no check of the
    tree, no rollout, no folder moved, whatever its variables are."""
    tree = Tree(tmp_path, splits=("test",))
    go = tmp_path / "go"
    log = tmp_path / "calls_first.jsonl"
    first = tree.start(log.name, FAKE_WAIT=str(go))
    try:
        assert wait_for(lambda: '"rollout"' in log.read_text(encoding="utf-8"))
        assert not (tree.main / "base_test").exists()
        assert_lines(tree.ledger(), [started(tree.root, "base")])
        files = files_of(tree.root)
        for variables in ({}, {"REPEAT_AFTER_FAULT": "1", "REPEAT_REASON": "a second start"}, {"DRESS": "1"}):
            code, markers, calls = tree.run(**variables)
            assert (code, markers, calls) == (1, ["CHAIN_FAILED lock"], []), tree.output
            assert "Another chain_test.sh, or a rollout that one started, runs in this repository folder: " \
                   "the lock is held. This start did nothing" in tree.output
        assert files_of(tree.root) == files
    finally:
        go.write_text("go\n", encoding="utf-8")
        output, _ = first.communicate(timeout=120)
    assert first.returncode == 0 and output.splitlines()[-1] == "CHAIN_DONE", output
    # The test set was sampled once: each arm has one start in the record and each episode once.
    lines = tree.ledger()
    assert [line.split()[0] for line in lines].count("ROLLOUT_START") == 4
    assert [line.split()[0] for line in lines].count("TEST_RUN_DONE") == 1
    for arm in ARMS:
        episodes = (tree.main / f"{arm}_test" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(episodes) == 200 and len(set(episodes)) == 200
    # The lock is gone with the chain: no file is left, and a start after it runs.
    assert not [path for path in tree.root.rglob("*") if "lock" in path.name]
    code, markers, calls = tree.run()
    assert code == 0 and scripts_of(calls) == ["freeze"] + REPORT_SCRIPTS, tree.output


def test_a_rollout_that_outlives_its_chain_keeps_the_lock_and_its_folder_is_not_taken(tmp_path):
    """The chain is stopped (here: its shell is killed) while a rollout runs, and the rollout goes on. It
    still holds the lock, so no second chain starts beside it. When it has ended, its folder is complete
    but has no note of the chain: the next start does not take it as the run of the arm."""
    tree = Tree(tmp_path, splits=("test",))
    go = tmp_path / "go"
    log = tmp_path / "calls_first.jsonl"
    first = tree.start(log.name, FAKE_WAIT=str(go))
    try:
        assert wait_for(lambda: '"rollout"' in log.read_text(encoding="utf-8"))
        first.kill()                                   # the shell of the chain; the rollout is its child
        first.wait(timeout=30)
        code, markers, calls = tree.run()
        assert (code, markers, calls) == (1, ["CHAIN_FAILED lock"], []), tree.output
    finally:
        go.write_text("go\n", encoding="utf-8")
    manifest = tree.main / "base_test" / "run_manifest.json"
    assert wait_for(lambda: manifest.is_file() and '"complete": true' in manifest.read_text(encoding="utf-8"))
    first.communicate(timeout=60)                      # the rollout has ended: the pipe is closed
    assert wait_for(lambda: tree.run()[1] != ["CHAIN_FAILED lock"], 30)
    code, markers, calls = tree.run()
    assert (code, markers, scripts_of(calls)) == (1, ran(START) + ["CHAIN_FAILED base_test"], ["freeze"])
    assert "results/main/base_test is complete, but it has no chain_record.txt." in tree.output
    assert_lines(tree.ledger(), [started(tree.root, "base")])      # a start, and no line of an end


def test_the_lock_is_taken_before_the_first_step_and_writes_no_file():
    body = chain_body()
    lock, first_step = body.index("if ! flock -n 9; then"), body.index("run_command freeze_check")
    assert body.index("exec 9< .") < lock < first_step
    assert body.count("flock -n 9") == 1 and "failed lock" in body[lock:first_step]
    # The lock is on the repository folder itself: nothing is made for it, in neither mode.
    assert "exec 9< .;" in body and ".lock" not in body
    # The first step is the check of the tree: no other step stands before it.
    steps = re.findall(r"^run_command (\S+)", body, flags=re.MULTILINE)
    assert steps[:3] == ["freeze_check", "adapters", "run_once"]
    assert "scripts/freeze.py check" in body[first_step:first_step + 60]


def test_a_complete_folder_without_the_note_of_the_chain_is_not_taken_as_the_run_of_the_arm(tmp_path,
                                                                                            real_run):
    """The chain writes chain_record.txt into a run folder when the rollout that it started has ended and
    the adapter is still the same. A complete folder without it was made by hand, or the chain was stopped
    in between. The chain does not go on with it, does not run the arm again and does not move it."""
    tree = Tree(tmp_path, source=real_run.tree.root)
    (tree.main / "sft_test" / "chain_record.txt").unlink()
    files = files_of(tree.root)
    for variables in ({}, {"REPEAT_AFTER_FAULT": "1", "REPEAT_REASON": "the note is missing"}):
        code, markers, calls = tree.run(**variables)
        assert (code, scripts_of(calls)) == (1, ["freeze"])
        if variables:
            assert markers[-1] == "CHAIN_FAILED run_once"
        else:
            assert markers == ran(START) + ["STEP_SKIPPED base_test"] + ran(["gate_base"]) + \
                ["CHAIN_FAILED sft_test"]
            assert "results/main/sft_test is complete, but it has no chain_record.txt." in tree.output
            assert "It is not taken as the run of the arm." in tree.output
    assert files_of(tree.root) == files

    # The note cannot be written when the rollout has ended: the chain stops there and says so, and the
    # next start stops at the same folder.
    other = Tree(tmp_path / "other", splits=("test",))
    code, markers, calls = other.run(FAKE_BLOCK_NOTE="sft_test")
    assert code == 1
    assert markers == ran(START + ["base_test", "gate_base", "sft_test"]) + ["CHAIN_FAILED sft_test"]
    assert scripts_of(calls) == ["freeze", "rollout", "rollout"]
    assert "The chain's note results/main/sft_test/chain_record.txt could not be written." in other.output
    assert read_json(other.main / "sft_test" / "run_manifest.json")["complete"] is True
    code, markers, calls = other.run()
    assert (code, markers[-1], scripts_of(calls)) == (1, "CHAIN_FAILED sft_test", ["freeze"])
    assert not (other.main / "grpo_seed1_test").exists()


ENV_FILE_PART = re.compile(r"^read_env_file\(\) \{.*?^unset ACCESSDESK_FREEZE_ROOT", re.DOTALL | re.MULTILINE)


def env_file_part(env_file: Path) -> str:
    """The lines of the script that read the file of the machine and ask whether the four values of the
    start are kept, with another file in the place of /workspace/env.sh. Then one line that prints them."""
    part = ENV_FILE_PART.search(chain_text()).group(0)
    assert part.count("/workspace/env.sh") == 1
    return ('DRESS="${DRESS:-0}"; REPEAT_AFTER_FAULT="${REPEAT_AFTER_FAULT:-0}"; '
            'REPEAT_REASON="${REPEAT_REASON:-}"; PYTHON="${PYTHON:-python}"\n'
            + part.replace("/workspace/env.sh", str(env_file))
            + '\necho "values: $DRESS|$REPEAT_AFTER_FAULT|$REPEAT_REASON|$PYTHON|${FROM_FILE-}|${PATH%%:*}"\n')


def test_the_file_of_the_machine_cannot_choose_the_mode_or_ask_for_a_repeat(tmp_path):
    """The script reads /workspace/env.sh, the Python environment of the machine. Here its own lines read
    another file. What the file exports is used; DRESS, REPEAT_AFTER_FAULT, REPEAT_REASON and PYTHON stay
    as they were at the start."""
    body = chain_body()
    assert body.count("/workspace/env.sh") == 1 and "read_env_file /workspace/env.sh\n" in body
    assert len(re.findall(r"(?:^|[;\s])\. ", body, flags=re.MULTILINE)) == 1          # one file is sourced
    assert body.index('PYTHON="${PYTHON:-python}"') < body.index("read_env_file /workspace/env.sh")
    env_file = tmp_path / "env.sh"
    clean = {key: value for key, value in os.environ.items() if key not in CHAIN_VARIABLES}

    def run(content: str | None, **variables: str) -> tuple[int, list[str]]:
        if content is None:
            env_file.unlink(missing_ok=True)
        else:
            env_file.write_text(content, encoding="utf-8")
        result = subprocess.run(["bash", "-c", env_file_part(env_file)], env={**clean, **variables},
                                capture_output=True, text=True)
        return result.returncode, result.stdout.splitlines()

    note = (f"CHAIN_NOTE {env_file} sets DRESS, REPEAT_AFTER_FAULT, REPEAT_REASON or PYTHON. That is not "
            "used: the chain keeps the values of its start.")
    exports = 'export PATH="/somewhere:$PATH"\nexport FROM_FILE=yes\n'
    assert run(None) == (0, [f"values: 0|0||python||{os.environ['PATH'].split(':')[0]}"])
    assert run(exports) == (0, ["values: 0|0||python|yes|/somewhere"])
    # The file tries each of the four, and all of them.
    for line in ("DRESS=1\n", "export REPEAT_AFTER_FAULT=1\n", 'REPEAT_REASON="from the file"\n',
                 "PYTHON=/usr/bin/other\n", "unset DRESS\n",
                 'DRESS=1\nREPEAT_AFTER_FAULT=1\nREPEAT_REASON="from the file"\nPYTHON=/usr/bin/other\n'):
        assert run(exports + line) == (0, [note, "values: 0|0||python|yes|/somewhere"]), line
    # What was given at the start stays, also against the file.
    assert run(exports + "DRESS=0\nREPEAT_AFTER_FAULT=0\n", DRESS="1", REPEAT_AFTER_FAULT="1",
               REPEAT_REASON="typed", PYTHON="/my/python") == \
        (0, [note, "values: 1|1|typed|/my/python|yes|/somewhere"])
    # A file that makes a value fixed for good: the chain does not start.
    code, lines = run("readonly DRESS=1\n")
    assert code == 1 and lines[-1] == "CHAIN_FAILED start" and not any(line.startswith("values:") for line in lines)
    assert "does not let the chain keep DRESS, REPEAT_AFTER_FAULT, REPEAT_REASON and PYTHON" in lines[-2]


def test_the_reference_rows_must_be_as_the_plan_says(tmp_path, real_run):
    """The plan, section 3: the oracle must pass every scenario, and each constant policy exactly one
    outcome in four. In the made-up tree each arm has 25 scenarios in four outcomes. One record with the
    other verdict stops the chain at the step reference_check; the four runs and their reports are saved,
    and the record gets no line of an end."""
    tree = Tree(tmp_path, source=real_run.tree.root)
    for policy, shown in (
            ("oracle", "oracle: 25 records in 4 outcomes; every record passes in ['already', 'approval', "
                       "'grant'], and none passes in []: NOT as the plan says"),
            ("constant_denied", "constant_denied: 25 records in 4 outcomes; every record passes in [], and "
                                "none passes in ['already', 'approval', 'grant']: NOT as the plan says"),
            ("constant_granted", "constant_granted: 25 records in 4 outcomes; every record passes in "
                                 "['grant'], and none passes in ['already', 'approval']: NOT as the plan "
                                 "says")):
        code, markers, calls = tree.run(FAKE_REFERENCE=policy)
        assert code == 1 and markers[-3:] == stopped_at("reference_check"), tree.output
        assert scripts_of(calls) == ["freeze"] + REPORT_SCRIPTS
        assert shown in tree.output
        assert "ANALYSIS_PLAN.md, section 3: `oracle` must pass every scenario" in tree.output
        assert len(tree.record()) == 18
    code, markers, calls = tree.run()
    assert code == 0 and len(tree.record()) == 24, tree.output


def test_the_check_of_the_reference_rows_takes_the_real_dev_rows_of_this_repository():
    """The function of the script itself reads the reference rows of dev that are in results/main (192
    records per policy). They are as the plan says, so the same check on the test rows asks for what the
    dev rows show. Nothing of the test split is read here."""
    reference = ROOT / "results" / "main" / "reference"
    if not all((reference / f"{policy}_dev.jsonl").is_file() for policy in POLICIES):
        pytest.skip(f"the reference rows of dev are not in {reference}")
    script = (function_text(SCRIPT, "check_reference") + "\nset -u\nPOLICIES=(" + " ".join(POLICIES) + ")\n"
              + f'OUT="{ROOT / "results" / "main"}"\nSPLIT=dev\nPYTHON="{sys.executable}"\ncheck_reference\n')
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == ("oracle: 192 records in 4 outcomes; every record passes in ['already', 'approval', "
                        "'grant', 'inactive'], and none passes in []: as the plan says")
    passes = {"constant_granted": "grant", "constant_pending": "approval", "constant_already": "already",
              "constant_denied": "inactive"}
    for line, policy in zip(lines[1:], POLICIES[1:]):
        others = sorted(set(passes.values()) - {passes[policy]})
        assert line == (f"{policy}: 192 records in 4 outcomes; every record passes in ['{passes[policy]}'], "
                        f"and none passes in {others}: as the plan says")
    assert len(lines) == 5


# ---------------------------------------------------------------------------
# The real repository: its records, its adapters and its calibration families (never its test split)
# ---------------------------------------------------------------------------
def test_the_families_of_a_rehearsal_are_calibration_families_of_the_final_folder_one_of_each_outcome():
    path = ROOT / "data" / "scenarios" / "calib.jsonl"
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    lines = path.read_text(encoding="utf-8").splitlines()
    families = {family["family_id"]: family for family in map(json.loads, lines)}
    assert set(DRESS_FAMILIES) <= set(families)
    assert sorted(families[family_id]["outcome"] for family_id in DRESS_FAMILIES) == \
        ["already", "approval", "grant", "inactive"]
    assert all(families[family_id]["split"] == "calib" for family_id in DRESS_FAMILIES)


def test_the_step_adapters_takes_the_real_records_and_the_real_adapters_of_this_repository(tmp_path):
    """A rehearsal in a tree that holds copies of the real records of results/main (the run manifest and
    the chain's note of each dev run, the run manifests of the three training runs), links to the real
    adapter files, and copies of the real MANIFEST.json, text file, dev file and calibration file. The
    tree has no test file. The check of the tree is a stand-in here (the copy is not the frozen tree); the
    step adapters is the script's own, on the real bytes."""
    main = ROOT / "results" / "main"
    records = [f"{arm}_dev/{name}" for arm in ARMS for name in ("run_manifest.json", "chain_record.txt")] \
        + ["sft/run_manifest.json", "grpo_seed1/run_manifest.json", "grpo_seed2/run_manifest.json"]
    adapters = [f"{folder}/{name}" for folder in ADAPTERS.values() if folder for name in ADAPTER_FILES]
    others = ["MANIFEST.json", "text/model_text.yaml", "data/scenarios/summary.json",
              "data/scenarios/dev.jsonl", "data/scenarios/calib.jsonl"]
    needed = [main / name for name in records] + [ROOT / name for name in adapters + others]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        pytest.skip(f"not on this machine: {missing[:3]}")
    tree = Tree(tmp_path, splits=("calib",))
    shutil.rmtree(tree.main)
    for name in records:
        (tree.main / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(main / name, tree.main / name)
    for name in adapters:
        (tree.root / name).parent.mkdir(parents=True, exist_ok=True)
        (tree.root / name).symlink_to(ROOT / name)
    for name in others:
        shutil.copy(ROOT / name, tree.root / name)
    assert not (tree.root / "data" / "scenarios" / "test.jsonl").exists()
    code, markers, calls = tree.run(DRESS="1", FAKE_FREEZE="0")
    assert code == 0, tree.output
    assert markers == ran(START + rollouts("calib") + AFTER + REPORTS) + ["CHAIN_DONE"]
    assert "not compared" not in tree.output
    # The record names the SHA-256 of the real weights, the ones that the dev notes name.
    record = tree.record(tree.dress)
    assert len(record) == 18
    for line, arm in zip(record[15:], ARMS[1:]):
        note = (main / f"{arm}_dev" / "chain_record.txt").read_text(encoding="utf-8").splitlines()
        assert note[3].startswith("adapter_config.json=") and note[4].startswith("adapter_model.safetensors=")
        assert f"arm={arm} adapter={ADAPTERS[arm]} {note[3]} {note[4]} folder={DRESS_OUT}/{arm}_calib" in line
    # The rollouts were given the real adapter folders and the four real calibration families.
    assert [value_of(call, "--adapter") for call in calls[2:5]] == [ADAPTERS[arm] for arm in ARMS[1:]]
    assert read_json(tree.dress / "base_calib" / "run_manifest.json")["families"] == 4
