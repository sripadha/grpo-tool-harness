#!/usr/bin/env bash
# The test run of the frozen plan as ONE job on the rented machine: the four arms on the test split, then the
# replay check, the test table, the monitor reports and the reference rows.
#
# ANALYSIS_PLAN.md, section 4: "The test set is run once, after the freeze and the dev table. No setting,
# checkpoint or frozen file is chosen or changed after a test number is seen. A repeat needs a machine fault
# or a harness fault, and a line in NOTES.md." This chain keeps that rule: see "Run once" below.
# _dev/test_run/README.md says how to rehearse the chain and how to start it on a rented machine.
#
# Start it on the machine, in the repository folder, so that it survives a lost connection:
#     mkdir -p /workspace/logs
#     nohup bash scripts/gpu/chain_test.sh > /workspace/logs/test_run.log 2>&1 < /dev/null &
# From the laptop the same is:  _dev/box.sh bg test_run 'bash scripts/gpu/chain_test.sh'
# The rehearsal comes first:    _dev/box.sh bg test_dress 'DRESS=1 bash scripts/gpu/chain_test.sh'
#
# The four arms, each with the adapter folder that its dev run loaded:
#   base          no adapter: the untrained model
#   sft           results/main/sft/adapter
#   grpo_seed1    results/main/grpo_seed1/checkpoints/step_00080/adapter
#   grpo_seed2    results/main/grpo_seed2/checkpoints/step_00080/adapter
# By the stop rule of the plan (section 7) a GRPO run goes to 80 steps, and its last checkpoint is the result.
#
# The steps, in order. <split> is test, and every folder is under results/main:
#   (the lock)          before any step: only one chain_test.sh runs in this repository folder at a time. A
#                       second start ends at once with CHAIN_FAILED lock (see "One chain at a time" below).
#   freeze_check        scripts/freeze.py check, the first step: the tree must be as MANIFEST.json says.
#                       scripts/gpu/rollout.py and scripts/run_scripted.py ask the same again at each start,
#                       and this chain never gives them the option that lets a changed tree pass.
#   adapters            every arm must be the model of its dev run (see "The same adapters" below)
#   run_once            what the folder of each arm holds now, and what the record says (see "Run once")
#   <arm>_<split>       scripts/gpu/rollout.py with the options of that arm's dev run; only --split and --out
#                       are other. So: the 2B at its pinned revision, text/model_text.yaml, data/scenarios, 8
#                       attempts per scenario, run seed 0, a token cap of 260, and the script's defaults for
#                       the rule set (default), the tools (five) and the prompt (final). Right before the
#                       rollout the adapter of the arm is compared with its dev run once more, and right
#                       after it the two adapter files are hashed again.
#   gate_<arm>          right after the rollout of the arm: the harness-error gate of the plan (section 7)
#   adapters_after      the step adapters once more, after the last rollout: the adapter files and the
#                       records must still be those of the dev runs
#   and then, only if all four runs are complete:
#   replay_<arm>        scripts/replay_check.py on the saved episodes of each run: it must find no difference
#   table               python -m accessdesk.eval.tables: ONE table of the four arms, with the comparisons
#                       that the plan names for test (section 4): sft minus base, and each GRPO seed minus
#                       sft. Into tables/<split>_base_sft_grpo_seed1_seed2.json and .md
#   monitors_<arm>      scripts/monitor_report.py on each run, into monitors/<arm>_<split>.md
#   reference_<policy>  scripts/run_scripted.py: oracle and the four constant policies, as the reference rows
#                       of dev were made, into reference/<policy>_<split>.jsonl
#   reference_check     the plan, section 3: oracle must pass every scenario, and each constant policy
#                       exactly one outcome in four. The chain reads `outcome` and `passed` of the records
#                       of the five steps before and stops if this is not so.
# At the end the chain adds the record of the run to test_run_record.txt (see "The record" below).
# The chain stops at the first step that fails, and its exit code is then not 0.
#
# One chain at a time. The chain takes a lock on the repository folder (flock; no file is written) and keeps
# it until it ends. Every program that it starts holds the lock too, so a rollout that outlives its chain
# still keeps a second chain out. A second start prints CHAIN_FAILED lock and does nothing else: it runs no
# check, starts no rollout and moves no folder. A rehearsal and the test run share the lock.
#
# The same adapters. scripts/gpu/rollout.py writes the folder of the adapter into its run manifest, and no
# hash. So the step "adapters" reads three records of each arm, all under results/main:
#   <arm>_dev/run_manifest.json   It must be there and complete, and it must name the values of this chain:
#                                 the model, the revision, the adapter folder, the attempts, the run seed,
#                                 the token cap, the text file, the dev file, the rule set, the tools, the
#                                 prompt, and the code_sha256 of MANIFEST.json.
#   <arm>_dev/chain_record.txt    The note of the stage chain that made the dev run. It names the SHA-256 of
#                                 the two adapter files as they were when that chain started the dev rollout.
#   the training run manifest     sft/run_manifest.json (outputs/adapter) and grpo_seed<N>/run_manifest.json
#                                 (final_adapter/files, the adapter of the last checkpoint). It names the
#                                 SHA-256 of the two adapter files as the training run wrote them. The run
#                                 must be complete (status complete; for a GRPO run also
#                                 complete_after_fault, with a CHAIN_NOTE), and the final adapter of a GRPO
#                                 run must be the folder checkpoints/step_00080/adapter.
# adapter_config.json and adapter_model.safetensors on disk must have the SHA-256 of both records. If one
# record is missing or names another hash, the chain stops. This is asked four times: at the step adapters;
# for each arm right before its rollout; for each arm right after its rollout (the two files are hashed
# again); and at the step adapters_after.
# What this shows: the adapter files on this machine are, byte for byte, the files that the training run
# wrote and that were in the folder when the dev rollout was started, and they were so right before and right
# after the rollout of their arm. What it cannot show: that a rollout loaded these bytes (no run manifest
# holds a hash of what was loaded), that a file was not changed and put back between two looks, and that
# chain_record.txt, a plain text file that no hash guards, was not changed by hand. The untrained model of
# the arm base is named by its revision only.
#
# Run once.
#   - A folder <arm>_<split> with a complete run (its run_manifest.json says "complete": true) is never run
#     again and never moved, whatever the variables say: STEP_SKIPPED. It is taken as the run of the arm only
#     if it holds the chain's own note (chain_record.txt, written when the rollout had ended and the adapter
#     was still the same), if it holds no adapter_changed.txt, and if its run manifest and the note fit this
#     chain. If not, the chain stops and leaves the folder as it is.
#   - A folder that is there and is NOT complete stops the chain at the step run_once, before any rollout.
#     Nothing is deleted, moved or written over. This does not rest on scripts/gpu/rollout.py: that script
#     refuses a folder that holds episodes, but a folder that holds only the run manifest of a run that
#     ended before its first pass it would use again without a word.
#   - A folder that is NOT there stops the chain at run_once too, if the record says that a rollout of the
#     arm was started and no later line of the record says that this rollout wrote nothing or why the arm
#     is repeated; or if an earlier folder of the arm lies beside it (<folder>.<name>) and no line of the
#     record names it. So an arm whose folder was moved or deleted by hand is not run again without a reason.
#   - REPEAT_AFTER_FAULT=1 with REPEAT_REASON='what the fault was' is the only way on. For each such arm the
#     chain FIRST adds a line with the time, the arm and the reason to test_run_record.txt. Only then it
#     renames an unfinished folder to <folder>.fault_<n> (kept). Then these arms run again. Without a reason
#     the chain does not start, and neither if no arm needs a repeat. If the line cannot be written, nothing
#     is moved. NOTES.md must name the repeat (the plan, section 4).
#   - The gate: harness_error_rate in the summary of the arm's run_manifest.json must be below 0.01 (1% of
#     the run's attempts). If it is 0.01 or more, the chain adds a line GATE_FAILED to the record and stops.
#     The plan (section 7): such a run is repaired and repeated, and NOTES.md says so. This chain never
#     makes that repeat by itself: the folder is complete, so a person must move it away first (to
#     <folder>.fault_<n>), and the next start then needs REPEAT_AFTER_FAULT=1 and the reason.
#   - An adapter that changes while its rollout runs: the chain adds a line ADAPTER_CHANGED to the record,
#     writes adapter_changed.txt into the run folder and stops. That folder is not taken as the run of the
#     arm at any later start.
#
# The rehearsal. DRESS chooses between the two modes, and nothing else does:
#                         split    every folder is under           more options of each rollout
#   DRESS=0 (default)     test     results/main                    none
#   DRESS=1               calib    results/_smoke/test_dress       --families, a file of four families
# A rehearsal runs the same steps in the same order. It reads the adapters and the dev records in
# results/main and writes nothing there. None of its numbers is a result. The split, the folders and every
# value of the list "Fixed" below are set in this file: no variable of the environment changes one of them,
# and neither does /workspace/env.sh (`read_env_file`).
#
# The record. test_run_record.txt (in results/main; for a rehearsal in its own folder) only grows. Every
# line starts with the time (UTC). It holds counts of attempts and of harness errors, and no pass rate:
#   REPEAT_AFTER_FAULT arm=... folder=... state=unfinished|not_there [moving_to=...] [aside=...] reason=...
#                       an arm is run again after a fault. Written BEFORE the folder is moved (moving_to).
#                       aside names a folder beside the run folder that no earlier line named.
#   MOVE_FAILED ...     the folder of the line before could not be renamed; the chain stopped
#   ROLLOUT_START arm=... folder=... pid=... host=... script=... adapter=...
#                       written right before a rollout is started, with the SHA-256 of this script and of
#                       the adapter files
#   ROLLOUT_END arm=... folder=... exit=<code> wrote=complete|unfinished|nothing
#                       the rollout ended with this exit code, and this is what its folder holds
#   ADAPTER_CHANGED ... an adapter file had another SHA-256 after the rollout than before it
#   GATE_PASSED / GATE_FAILED arm=... folder=... attempts=... harness_errors=... harness_error_rate=...
#                       GATE_PASSED is written for a rollout of this start, GATE_FAILED whenever the gate
#                       is not passed
#   ADAPTERS_AFTER_FAILED   the step adapters_after failed
#   TEST_RUN_DONE ... rollouts_run=<arms>   the chain reached its end, for the first time after a rollout
#                       was started. rollouts_run names the rollouts of this start (none: the rollouts were
#                       those of earlier starts, and this one built the reports).
#   REPORTS_REBUILT ... rollouts_run=none   a later start: every arm was complete already, the record held
#                       its TEST_RUN_DONE, and only the reports were built again
#                       Below each of the two: the SHA-256 of MANIFEST.json, the code_sha256, the SHA-256
#                       of this script, and for each arm its adapter with the SHA-256 of its two files, its
#                       folder, the attempts, the harness errors, and the folders beside the run folder
# A ROLLOUT_START without a ROLLOUT_END after it: the chain or the machine stopped while that rollout ran.
#
# Lines to look for in the log:
#   STEP_START <name>               a step starts
#   STEP_DONE <name> <exit code>    it ended
#   STEP_SKIPPED <name>             the folder of the arm holds a complete run that fits this chain
#   STEP_MOVED_ASIDE <name> <dir>   REPEAT_AFTER_FAULT=1: an unfinished folder was renamed, not deleted
#   CHAIN_FAILED <name>             the chain failed at this step ("start": a wrong variable, a missing
#                                   file; "lock": another chain runs; "record": the last lines of the
#                                   record could not be written). <arm>_<split> with STEP_DONE ... 0 before
#                                   it: the rollout ended, and a check right after it failed.
#   CHAIN_NOTE <text>               the mode, and where the record of the run is
#   CHAIN_DONE                      every step is done
# This script stops no machine and copies no results: the machine's timer and the fetch loop do that.

DRESS="${DRESS:-0}"                              # 1: a rehearsal on calibration families
REPEAT_AFTER_FAULT="${REPEAT_AFTER_FAULT:-0}"    # 1: after a fault, note the reason and run those arms again
REPEAT_REASON="${REPEAT_REASON:-}"               # what the fault was; needed with REPEAT_AFTER_FAULT=1
PYTHON="${PYTHON:-python}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || { echo "CHAIN_FAILED start"; exit 1; }

read_env_file() {     # read_env_file <file>: read the file of the machine's Python environment, if it is
                      # there. The four values above are those of the start, and they stay so: a line of
                      # the file that sets one of them changes nothing, and a CHAIN_NOTE says so. So the
                      # file cannot choose the mode or ask for a repeat.
    local chain_kept_dress="$DRESS" chain_kept_repeat="$REPEAT_AFTER_FAULT"
    local chain_kept_reason="$REPEAT_REASON" chain_kept_python="$PYTHON"
    if [ -r "$1" ]; then . "$1"; fi
    if [ "${DRESS-}" != "$chain_kept_dress" ] || [ "${REPEAT_AFTER_FAULT-}" != "$chain_kept_repeat" ] \
            || [ "${REPEAT_REASON-}" != "$chain_kept_reason" ] || [ "${PYTHON-}" != "$chain_kept_python" ]; then
        echo "CHAIN_NOTE $1 sets DRESS, REPEAT_AFTER_FAULT, REPEAT_REASON or PYTHON. That is not used: the" \
             "chain keeps the values of its start."
    fi
    DRESS="$chain_kept_dress"
    REPEAT_AFTER_FAULT="$chain_kept_repeat"
    REPEAT_REASON="$chain_kept_reason"
    PYTHON="$chain_kept_python"
}

AT_START=("$DRESS" "$REPEAT_AFTER_FAULT" "$REPEAT_REASON" "$PYTHON")
read_env_file /workspace/env.sh
# Asked here and not in the function: a value that the file made read-only ends the function early.
if [ "${DRESS-}" != "${AT_START[0]}" ] || [ "${REPEAT_AFTER_FAULT-}" != "${AT_START[1]}" ] \
        || [ "${REPEAT_REASON-}" != "${AT_START[2]}" ] || [ "${PYTHON-}" != "${AT_START[3]}" ]; then
    echo "The file of the machine's environment does not let the chain keep DRESS, REPEAT_AFTER_FAULT," \
         "REPEAT_REASON and PYTHON as they were at its start."
    echo "CHAIN_FAILED start"
    exit 1
fi
unset ACCESSDESK_FREEZE_ROOT      # for tests of scripts/freeze.py: it points that script at another tree
set -u

# Fixed. These are the values of the dev runs; the step "adapters" compares them with the run manifests of
# those runs. A value that the environment gives for one of these names is not read.
MODEL="Qwen/Qwen3.5-2B"
REVISION="15852e8c16360a2fea060d615a32b45270f8a8fc"
SCENARIOS="data/scenarios"
TEXT="text/model_text.yaml"
SAMPLES=8                                        # attempts per scenario
EVAL_SEED=0                                      # run seed of every rollout on dev and on test
MAX_NEW_TOKENS=260                               # token cap per turn
MAIN="results/main"                              # the adapters and the dev runs are here
ARMS=(base sft grpo_seed1 grpo_seed2)
POLICIES=(oracle constant_granted constant_pending constant_already constant_denied)    # the reference rows
DRESS_FAMILIES=(ca-00 ca-01 ca-02 ca-03)         # a rehearsal: one calibration family of each outcome

# ---- The functions below, up to "end of the helpers of the stage chains", have the text that they have in
# chain_stage1.sh and chain_stage2.sh (tests/core/test_chain_test.py compares them). In this file the
# chain's own note of a step is written by `rollout_step`; there is no `run_step`.
RECORD_NAME="chain_record.txt"      # the chain's own note in the folder of a step that it ran

json_value() {        # json_value <file> <key> [<key>...]: a value of a JSON file, or "" (missing, or null)
    "$PYTHON" - "$@" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        value = json.load(handle)
    for key in sys.argv[2:]:
        value = value[key]
except (OSError, ValueError, KeyError, TypeError):
    value = ""
print("" if value is None else value)
PY
}

manifest_value() {    # manifest_value <folder> <key> [<key>...]: a value of the folder's run manifest, or ""
    json_value "$1/run_manifest.json" "${@:2}"
}

is_complete() {       # is_complete <folder>: true if the folder's run manifest says "complete": true
    [ "$(manifest_value "$1" complete)" = "True" ]
}

file_sha256() {       # file_sha256 <file>: the SHA-256 of the file, or "" if it cannot be read
    sha256sum "$1" 2>/dev/null | cut -d' ' -f1
}

chain_lines() {       # chain_lines <item>...: the items "chain:<label>=<value>", one a line, without "chain:"
    local item
    for item in "$@"; do
        case "$item" in chain:*) printf '%s\n' "${item#chain:}" ;; esac
    done
}

fits() {              # fits <name> <folder> <item>...: is the complete step in <folder> the step that this
                      # chain would run now? Every difference is printed. The items:
                      #   <key>[/<key>...]=<value>  the run manifest must hold this value. A key that the
                      #                             manifest does not have is not compared; a line says so.
                      #   chain:<label>=<value>     the chain's own note of the step ($RECORD_NAME, written
                      #                             by `run_step` when the step ended) must hold these lines.
                      #                             It holds what a run manifest does not name, such as the
                      #                             SHA-256 of the adapter that a rollout loaded.
    local name="$1" folder="$2"; shift 2
    local same=0 item of_manifest=()
    for item in "$@"; do
        case "$item" in chain:*) ;; *) of_manifest+=("$item") ;; esac
    done
    "$PYTHON" - "$name" "$folder/run_manifest.json" ${of_manifest[@]+"${of_manifest[@]}"} <<'PY' || same=1
import json
import sys

name, path = sys.argv[1:3]
with open(path, encoding="utf-8") as handle:
    manifest = json.load(handle)
different = 0
for item in sys.argv[3:]:
    keys, _, wanted = item.partition("=")
    value = manifest
    try:
        for key in keys.split("/"):
            value = value[key]
    except (KeyError, TypeError):
        print(f"{name}: the run manifest does not name {keys}: not compared")
        continue
    found = "" if value is None else str(value)
    if found != wanted:
        different += 1
        print(f"{name}: {keys} is {found!r} in {path}; this chain has {wanted!r}")
raise SystemExit(1 if different else 0)
PY
    if [ -z "$(chain_lines "$@")" ]; then
        return "$same"
    fi
    if [ ! -f "$folder/$RECORD_NAME" ]; then
        echo "$name: $folder has no $RECORD_NAME (this chain script did not run the step): the chain's" \
             "own notes are not compared"
    elif [ "$(chain_lines "$@")" != "$(cat "$folder/$RECORD_NAME")" ]; then
        echo "$name: what the chain noted in $folder/$RECORD_NAME when the step ran:"
        sed 's/^/    /' "$folder/$RECORD_NAME"
        echo "$name: what this chain has now:"
        chain_lines "$@" | sed 's/^/    /'
        same=1
    fi
    return "$same"
}

timed() {             # timed <name> <command...>: run it between STEP_START and STEP_DONE; its exit code
    local name="$1"; shift
    echo "STEP_START $name"
    date -u +%FT%TZ
    "$@"
    local code=$?
    date -u +%FT%TZ
    echo "STEP_DONE $name $code"
    return "$code"
}

run_command() {       # run_command <name> <command...>: run it; a failure ends the chain
    local code=0
    timed "$@" || code=$?
    if [ "$code" -ne 0 ]; then
        echo "CHAIN_FAILED $1"
        exit "$code"
    fi
}

write_to() {          # write_to <file> <command...>: the command's output goes into the file
    local target="$1" code=0; shift
    "$@" > "$target.tmp" || code=$?
    if [ "$code" -ne 0 ]; then
        rm -f "$target.tmp"
        return "$code"
    fi
    mv "$target.tmp" "$target"
}

ROLLOUT_ITEMS=()      # set by rollout_items

rollout_items() {     # rollout_items <arm> <split> <run seed> <attempts> <adapter> [<option>...]: what one
                      # rollout must be, as the items of `fits`, into ROLLOUT_ITEMS. <adapter> and the
                      # options are those of `rollout_step`. A rollout is skipped only if it fits these
                      # items. chain_stage2.sh also checks the dev rollouts of stage 1 with them, before it
                      # builds a table on them.
    local arm="$1" split="$2" seed="$3" samples="$4" adapter="$5"; shift 5
    local family_set="" previous="" option
    ROLLOUT_ITEMS=("model=$MODEL" "revision=$REVISION" "adapter=$adapter" "split=$split" "samples=$samples"
                   "run_seed=$seed" "config/max_new_tokens=$MAX_NEW_TOKENS" "text_sha256=$TEXT_SHA256"
                   "split_file_sha256=$(file_sha256 "$SCENARIOS/$split.jsonl")"
                   "chain:scenarios=$SCENARIOS" "chain:arm=$arm" "chain:options=$*")
    if [ -n "$adapter" ]; then
        ROLLOUT_ITEMS+=("chain:adapter_config.json=$(file_sha256 "$adapter/adapter_config.json")"
                        "chain:adapter_model.safetensors=$(file_sha256 "$adapter/adapter_model.safetensors")")
    fi
    for option in "$@"; do
        if [ "$previous" = "--families" ]; then
            ROLLOUT_ITEMS+=("chain:families_file=$(file_sha256 "$option")")
        fi
        if [ "$previous" = "--set" ]; then
            family_set="$option"
            ROLLOUT_ITEMS+=("chain:sets.json=$(file_sha256 "$SCENARIOS/sets.json")")
        fi
        previous="$option"
    done
    ROLLOUT_ITEMS+=("family_set=$family_set")
}
# ---- end of the helpers of the stage chains

MARKER="adapter_changed.txt"        # in a run folder: the adapter of the arm changed while its rollout ran
TIME_SHAPE='????-??-??T??:??:??Z'   # the first word of every line of the record, as a pattern
RAN=()                              # the arms whose rollout this start ran
RUN_LINES=()                        # one line per arm for the record of the run; `gate` adds them

failed() {            # failed <name>: the last line of a chain that stops outside a step
    echo "CHAIN_FAILED $1"
    exit 1
}

note() {              # note <text>: add the line "<time> <text>" to the record of the run, and look that it
                      # is its last line now. If the line cannot be written, this returns 1 and the caller
                      # stops: nothing that the record must name happens without its line.
    local line
    line="$(date -u +%FT%TZ) $1"
    printf '%s\n' "$line" >> "$RECORD" || return 1
    [ "$(tail -n 1 "$RECORD")" = "$line" ] || return 1
    sync "$RECORD" 2> /dev/null || true
}

adapter_of() {        # adapter_of <arm>: the adapter folder of the arm, or "" for base, the untrained model
    case "$1" in
        base) ;;
        sft) echo "$MAIN/sft/adapter" ;;
        *) echo "$MAIN/$1/checkpoints/step_00080/adapter" ;;
    esac
}

adapter_note() {      # adapter_note <arm>: the adapter of the arm as a part of a record line: "adapter=none",
                      # or its folder and the SHA-256 of its two files as they are on disk now
    local adapter
    adapter="$(adapter_of "$1")"
    if [ -z "$adapter" ]; then
        echo "adapter=none"
    else
        echo "adapter=$adapter adapter_config.json=$(file_sha256 "$adapter/adapter_config.json")" \
             "adapter_model.safetensors=$(file_sha256 "$adapter/adapter_model.safetensors")"
    fi
}

trained_sha256() {    # trained_sha256 <arm> <file>: the SHA-256 that the training run of the arm wrote into
                      # its run manifest for a file of its adapter, or "" if the manifest names none.
                      # scripts/gpu/train_sft.py: outputs/adapter. scripts/gpu/train_grpo.py:
                      # final_adapter/files, the adapter of the last checkpoint of a complete run.
    if [ "$1" = "sft" ]; then
        json_value "$MAIN/sft/run_manifest.json" outputs adapter "$2" sha256
    else
        json_value "$MAIN/$1/run_manifest.json" final_adapter files "$2" sha256
    fi
}

same_as_dev() {       # same_as_dev [<arm>...]: the step "adapters": each of these arms (without a name:
                      # every arm) must be the model of its dev run (the top of this file says what is
                      # compared, and what that can and cannot show). Every difference is printed. It only
                      # reads.
    local arms=("$@") arm adapter dev file on_disk trained training status final wrong=0
    if [ "$#" -eq 0 ]; then arms=("${ARMS[@]}"); fi
    for arm in "${arms[@]}"; do
        adapter="$(adapter_of "$arm")"
        dev="$MAIN/${arm}_dev"
        if ! is_complete "$dev"; then
            echo "$arm: the dev run of this arm is not here: $dev/run_manifest.json is missing, or it does" \
                 "not say \"complete\": true"
            wrong=1
            continue
        fi
        if [ ! -f "$dev/$RECORD_NAME" ]; then
            echo "$arm: $dev/$RECORD_NAME is missing: the note of the chain that made the dev run. Only it" \
                 "names the SHA-256 of the adapter files that the dev rollout was started with."
            wrong=1
            continue
        fi
        rollout_items "$arm" dev "$EVAL_SEED" "$SAMPLES" "$adapter"
        if fits "${arm}_dev" "$dev" "${ROLLOUT_ITEMS[@]}" "${MORE_ITEMS[@]}"; then
            echo "$arm: $dev is the dev run of this arm: its run manifest and its $RECORD_NAME fit this chain"
        else
            echo "$arm: $dev is not the dev run of what this chain would run (the lines above). If they" \
                 "name adapter_config.json or adapter_model.safetensors: the adapter in '$adapter' is not" \
                 "the one that the dev rollout was started with."
            wrong=1
        fi
        if [ -z "$adapter" ]; then continue; fi
        # The training run that wrote the adapter: a complete run, and for a GRPO run its last checkpoint.
        training="$MAIN/$arm"
        status="$(manifest_value "$training" status)"
        if ! is_complete "$training"; then
            echo "$arm: the training run in $training is not a complete run: its run manifest is missing, or" \
                 "it does not say \"complete\": true (its status: '$status')"
            wrong=1
        elif [ "$status" = "complete_after_fault" ] && [ "$arm" != "sft" ]; then
            echo "CHAIN_NOTE the training run in $training has the status complete_after_fault: it was" \
                 "started again after the stop rule had ended it. The write-up must say so (the plan," \
                 "section 7)."
        elif [ "$status" != "complete" ]; then
            echo "$arm: the training run in $training has the status '$status', not 'complete'"
            wrong=1
        fi
        if [ "$arm" != "sft" ]; then
            final="$(manifest_value "$training" final_adapter folder)"
            if [ "$training/$final" != "$adapter" ]; then
                echo "$arm: the run manifest of the training run names '$final' as its final adapter; this" \
                     "chain loads '$adapter' (the last checkpoint of a run of 80 steps)"
                wrong=1
            fi
        fi
        for file in adapter_config.json adapter_model.safetensors; do
            on_disk="$(file_sha256 "$adapter/$file")"
            trained="$(trained_sha256 "$arm" "$file")"
            if [ -n "$on_disk" ] && [ "$on_disk" = "$trained" ]; then
                echo "$arm: $adapter/$file has the SHA-256 that the training run wrote: $on_disk"
            else
                echo "$arm: $adapter/$file has the SHA-256 '$on_disk', and the run manifest of the training" \
                     "run names '$trained'. An empty value: the file is missing, or the run manifest is" \
                     "missing or does not name the file."
                wrong=1
            fi
        done
    done
    if [ "$wrong" -ne 0 ]; then
        echo "The adapters ($MAIN/sft/adapter and $MAIN/grpo_seed<N>/checkpoints/step_00080/adapter), the" \
             "run manifests of the training runs and of the dev runs, and $RECORD_NAME of the dev runs must" \
             "be on this machine as they are on the machine that holds the results. Nothing was changed."
    fi
    return "$wrong"
}

after_rollouts() {    # the step "adapters_after": the step adapters once more, after the last rollout
    if same_as_dev; then return 0; fi
    note "ADAPTERS_AFTER_FAILED the adapter files or their records are not those of the dev runs" \
        || echo "(the line ADAPTERS_AFTER_FAILED could not be added to $RECORD)"
    echo "Each rollout of this start was checked right before and right after it ran, and the four runs are" \
         "saved. No report was built. Put the right files back and start the chain again."
    return 1
}

open_start() {        # open_start <arm>: the last line ROLLOUT_START of the arm in the record that no later
                      # line of the arm explains; nothing if there is none. Two lines explain it:
                      # ROLLOUT_END ... wrote=nothing (that rollout ended before it made its folder) and
                      # REPEAT_AFTER_FAULT (the arm is run again after a fault, with a reason).
    local arm="$1" line head open=""
    if [ ! -f "$RECORD" ]; then return 0; fi
    while IFS= read -r line || [ -n "$line" ]; do
        head="${line%% reason=*} "
        case "$head" in
            $TIME_SHAPE" ROLLOUT_START arm=$arm "*) open="$line" ;;
            $TIME_SHAPE" REPEAT_AFTER_FAULT arm=$arm "*) open="" ;;
            $TIME_SHAPE" ROLLOUT_END arm=$arm "*" wrote=nothing "*) open="" ;;
        esac
    done < "$RECORD"
    printf '%s' "$open"
}

is_named() {          # is_named <arm> <path>: true if a line REPEAT_AFTER_FAULT of the arm in the record
                      # names this folder beside the run folder (moving_to=<path> or aside=<path>)
    local arm="$1" path="$2" line head word words
    if [ ! -f "$RECORD" ]; then return 1; fi
    while IFS= read -r line || [ -n "$line" ]; do
        head="${line%% reason=*}"
        case "$head" in
            $TIME_SHAPE" REPEAT_AFTER_FAULT arm=$arm "*) ;;
            *) continue ;;
        esac
        read -ra words <<< "$head"
        for word in "${words[@]}"; do
            if [ "$word" = "moving_to=$path" ] || [ "$word" = "aside=$path" ]; then return 0; fi
        done
    done < "$RECORD"
    return 1
}

beside() {            # beside <folder>: what lies beside a run folder under its name with an ending
                      # (<folder>.fault_<n>, or another <folder>.<name>), one a line
    local path
    for path in "$1".*; do
        if [ -e "$path" ] || [ -L "$path" ]; then printf '%s\n' "$path"; fi
    done
}

not_named() {         # not_named <arm> <folder>: those of `beside` that no line of the record names
    local path
    while IFS= read -r path; do
        if [ -n "$path" ] && ! is_named "$1" "$path"; then printf '%s\n' "$path"; fi
    done <<< "$(beside "$2")"
}

run_once() {          # the step "run_once": the test set is run once (the top of this file has the rules).
                      # It prints what the folder of each arm holds and what the record says of the arm. An
                      # arm that needs a reason stops the chain; with REPEAT_AFTER_FAULT=1 the reason goes
                      # into the record first, then an unfinished folder is renamed, and the arm runs again.
                      # A complete folder is never touched here.
    local arm folder started unnamed path index n aside line repeat=() states=()
    for arm in "${ARMS[@]}"; do
        folder="$OUT/${arm}_$SPLIT"
        unnamed="$(not_named "$arm" "$folder")"
        case "$unnamed" in
            *[[:blank:],]*)
                echo "A name beside $folder holds a space or a comma. Give it a plain name: the record of" \
                     "the run names these folders."
                return 1
                ;;
        esac
        if is_complete "$folder"; then
            echo "$folder: a complete run. It is not run again."
            if [ -n "$unnamed" ]; then
                echo "    Beside it, and named by no line of $RECORD: ${unnamed//$'\n'/ }"
            fi
        elif [ -e "$folder" ] || [ -L "$folder" ]; then
            echo "$folder: there, and NOT complete."
            repeat+=("$arm")
            states+=(unfinished)
        else
            started="$(open_start "$arm")"
            if [ -n "$started" ]; then
                echo "$folder: NOT there. But $RECORD says that a rollout of this arm was started, and no" \
                     "later line says that it wrote nothing or why the arm is repeated:"
                echo "    $started"
                repeat+=("$arm")
                states+=(not_there)
            elif [ -n "$unnamed" ]; then
                echo "$folder: NOT there. But beside it lies an earlier folder of this arm that no line of" \
                     "$RECORD names: ${unnamed//$'\n'/ }"
                repeat+=("$arm")
                states+=(not_there)
            else
                echo "$folder: not there. The arm runs."
            fi
        fi
    done
    if [ "${#repeat[@]}" -eq 0 ]; then
        if [ "$REPEAT_AFTER_FAULT" = "1" ]; then
            echo "REPEAT_AFTER_FAULT=1, and no arm needs a repeat: no folder is unfinished, and the record" \
                 "names no rollout whose folder is gone. Start the chain without REPEAT_AFTER_FAULT."
            return 1
        fi
        return 0
    fi
    if [ "$REPEAT_AFTER_FAULT" != "1" ]; then
        echo "The chain stops, and nothing was deleted, moved or written over. A reason is needed for:" \
             "${repeat[*]}."
        echo "ANALYSIS_PLAN.md, section 4: the test set is run once; a repeat needs a machine fault or a" \
             "harness fault, and a line in NOTES.md."
        echo "Read the log of the run that left this first. A run folder that is only in another place" \
             "(another machine, the laptop) is brought back, and then nothing is repeated."
        echo "If it was such a fault, start the chain again with:" \
             " REPEAT_AFTER_FAULT=1 REPEAT_REASON='what the fault was'"
        echo "Then the time, the arm and the reason are added to $RECORD first, an unfinished folder is" \
             "renamed to <folder>.fault_<n> and kept, and these arms run again."
        if [ "$DRESS" = "1" ]; then
            echo "This is a rehearsal: $OUT holds no result, so the whole folder may also be moved away."
        fi
        return 1
    fi
    # A second look without Python, before any line is written or any folder is moved: no fault of
    # `is_complete` can then move a complete run aside.
    for arm in "${repeat[@]}"; do
        if grep -qs '"complete": *true' "$OUT/${arm}_$SPLIT/run_manifest.json"; then
            echo "$OUT/${arm}_$SPLIT: its run manifest could not be read as that of a complete run, but it" \
                 "holds the words \"complete\": true. No folder was moved. Find out what this folder is."
            return 1
        fi
    done
    for index in "${!repeat[@]}"; do
        arm="${repeat[$index]}"
        folder="$OUT/${arm}_$SPLIT"
        line="REPEAT_AFTER_FAULT arm=$arm folder=$folder state=${states[$index]}"
        aside=""
        if [ "${states[$index]}" = "unfinished" ]; then
            n=1
            while [ -e "$folder.fault_$n" ] || [ -L "$folder.fault_$n" ]; do n=$((n + 1)); done
            aside="$folder.fault_$n"
            line="$line moving_to=$aside"
        fi
        while IFS= read -r path; do
            if [ -n "$path" ]; then line="$line aside=$path"; fi
        done <<< "$(not_named "$arm" "$folder")"
        # The line first, the move after it: no folder is moved without its line in the record.
        if ! note "$line reason=$REASON"; then
            echo "The line of this repeat could not be added to $RECORD. Nothing was moved: $folder is as" \
                 "it was."
            return 1
        fi
        if [ -n "$aside" ]; then
            if ! mv -T "$folder" "$aside"; then
                note "MOVE_FAILED arm=$arm folder=$folder moving_to=$aside" \
                    || echo "(the line MOVE_FAILED could not be added to $RECORD)"
                echo "$folder could not be renamed to $aside. It is as it was."
                return 1
            fi
            echo "STEP_MOVED_ASIDE ${arm}_$SPLIT $aside"
        fi
    done
    echo "The reason is in $RECORD: $REASON"
}

rollout_step() {      # rollout_step <arm> [<option>...]: the one run of an arm on $SPLIT, into
                      # $OUT/<arm>_$SPLIT. The command is the one of `rollout_step` of the stage chains, so
                      # it is the command of the arm's dev run with another --split and --out. The options
                      # are more options of scripts/gpu/rollout.py; only a rehearsal has them.
    local arm="$1"; shift
    local name="${arm}_$SPLIT" folder="$OUT/${arm}_$SPLIT" adapter with_adapter=()
    local code=0 ended=0 wrote before after
    adapter="$(adapter_of "$arm")"
    if [ -n "$adapter" ]; then with_adapter=(--adapter "$adapter"); fi
    rollout_items "$arm" "$SPLIT" "$EVAL_SEED" "$SAMPLES" "$adapter" "$@"
    if is_complete "$folder"; then
        if [ -e "$folder/$MARKER" ]; then
            echo "$folder is complete, but it holds $MARKER:"
            sed 's/^/    /' "$folder/$MARKER"
            echo "It is not taken as the run of the arm. This chain never runs a complete folder again and" \
                 "never moves one: see _dev/test_run/README.md, section 4."
            failed "$name"
        fi
        if [ ! -f "$folder/$RECORD_NAME" ]; then
            echo "$folder is complete, but it has no $RECORD_NAME. This chain writes that note when a" \
                 "rollout that it started has ended and the adapter is still the same. So this folder was" \
                 "made by hand, or the chain was stopped between the end of the rollout and its note."
            echo "It is not taken as the run of the arm. This chain never runs a complete folder again and" \
                 "never moves one: see _dev/test_run/README.md, section 4."
            failed "$name"
        fi
        if ! fits "$name" "$folder" "${ROLLOUT_ITEMS[@]}" "${MORE_ITEMS[@]}"; then
            echo "$folder is complete, but it is not the run that this chain would make (the lines above)."
            echo "This chain never runs a complete folder again and never moves one: find out what made it."
            failed "$name"
        fi
        echo "STEP_SKIPPED $name"
        return 0
    fi
    if [ -e "$folder" ] || [ -L "$folder" ]; then
        echo "$folder is there and is not complete. It was not so when this chain started (the step" \
             "run_once): something else writes to it. This chain changed nothing in it."
        failed "$name"
    fi
    # Right before the rollout: the adapter of this arm must still be the one of its dev run.
    if ! same_as_dev "$arm"; then
        echo "The rollout of $arm was not started: nothing was sampled."
        failed "$name"
    fi
    rollout_items "$arm" "$SPLIT" "$EVAL_SEED" "$SAMPLES" "$adapter" "$@"
    before="$(adapter_note "$arm")"
    if ! note "ROLLOUT_START arm=$arm folder=$folder pid=$$ host=$HOST script=$SCRIPT_SHA256 $before"; then
        echo "The line ROLLOUT_START could not be added to $RECORD. The rollout of $arm was not started:" \
             "nothing was sampled."
        failed "$name"
    fi
    timed "$name" "$PYTHON" scripts/gpu/rollout.py --model "$MODEL" --revision "$REVISION" \
        ${with_adapter[@]+"${with_adapter[@]}"} --text "$TEXT" --scenarios "$SCENARIOS" \
        --split "$SPLIT" --samples "$SAMPLES" --run-seed "$EVAL_SEED" --max-new-tokens "$MAX_NEW_TOKENS" \
        --arm "$arm" --out "$folder" "$@" || code=$?
    if is_complete "$folder"; then
        wrote="complete"
    elif [ -e "$folder" ] || [ -L "$folder" ]; then
        wrote="unfinished"
    else
        wrote="nothing"
    fi
    note "ROLLOUT_END arm=$arm folder=$folder exit=$code wrote=$wrote" || ended=1
    if [ "$ended" -ne 0 ]; then echo "(the line ROLLOUT_END could not be added to $RECORD)"; fi
    if [ "$code" -ne 0 ]; then
        echo "CHAIN_FAILED $name"
        exit "$code"
    fi
    # Right after the rollout: the two adapter files must have the SHA-256 that they had before it.
    after="$(adapter_note "$arm")"
    if [ "$after" != "$before" ]; then
        echo "The adapter of $arm changed while its rollout ran:"
        echo "    before: $before"
        echo "    after:  $after"
        if [ -d "$folder" ]; then
            printf '%s\n' "$(date -u +%FT%TZ) the adapter of the arm $arm changed while this rollout ran." \
                "before: $before" "after:  $after" \
                "So it is not known which weights this run sampled with." > "$folder/$MARKER" \
                || echo "($folder/$MARKER could not be written)"
        fi
        note "ADAPTER_CHANGED arm=$arm folder=$folder before: $before after: $after" \
            || echo "(the line ADAPTER_CHANGED could not be added to $RECORD)"
        echo "This run is not taken as the run of the arm: the chain did not write its $RECORD_NAME, and" \
             "$folder/$MARKER says why. See _dev/test_run/README.md, section 4."
        failed "$name"
    fi
    if [ "$wrote" != "complete" ]; then
        echo "The rollout ended with the exit code 0, but $folder is not complete."
        failed "$name"
    fi
    if ! chain_lines "${ROLLOUT_ITEMS[@]}" > "$folder/$RECORD_NAME"; then
        echo "The chain's note $folder/$RECORD_NAME could not be written. Without it a later start does" \
             "not take this folder as the run of the arm: see _dev/test_run/README.md, section 4."
        failed "$name"
    fi
    if [ "$ended" -ne 0 ]; then failed "$name"; fi
    RAN+=("$arm")
}

ran_now() {           # ran_now <arm>: true if this start ran the rollout of the arm
    local arm
    for arm in ${RAN[@]+"${RAN[@]}"}; do
        if [ "$arm" = "$1" ]; then return 0; fi
    done
    return 1
}

gate() {              # gate <arm>: the step "gate_<arm>", the harness-error gate of the plan (section 7).
                      # scripts/gpu/rollout.py wrote harness_error_rate into the summary of the run manifest:
                      # harness errors / attempts. It must be below 0.01. The gate reads the run manifest
                      # and changes no run folder; its result goes into the record.
    local arm="$1" folder="$OUT/$1_$SPLIT" counts code=0
    counts="$("$PYTHON" - "$folder/run_manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)["summary"]
rate = summary["harness_error_rate"]
print(f"attempts={summary['episodes']} harness_errors={summary['harness_errors']} harness_error_rate={rate}")
raise SystemExit(0 if rate is not None and rate < 0.01 else 1)
PY
)" || code=$?
    echo "$folder/run_manifest.json: ${counts:-its summary could not be read}"
    if [ "$code" -ne 0 ]; then
        note "GATE_FAILED arm=$arm folder=$folder ${counts:-the summary could not be read}" \
            || echo "(the line GATE_FAILED could not be added to $RECORD)"
        echo "The gate is not passed: harness_error_rate must be below 0.01 (1% of the run's attempts)." \
             "No run folder was changed."
        echo "ANALYSIS_PLAN.md, section 7: a run that fails the gate is repaired and repeated, and NOTES.md" \
             "says so. This chain never runs a complete folder again and never moves one. What a repeat" \
             "needs is in _dev/test_run/README.md, section 4: it is a decision for the main session."
        return 1
    fi
    if ran_now "$arm"; then
        if ! note "GATE_PASSED arm=$arm folder=$folder $counts"; then
            echo "The line GATE_PASSED could not be added to $RECORD."
            return 1
        fi
    fi
    RUN_LINES+=("arm=$arm $(adapter_note "$arm") folder=$folder $counts moved_aside=$(moved_aside "$folder")")
}

moved_aside() {       # moved_aside <folder>: the folders beside it that hold an earlier run of the arm
                      # (<folder>.fault_<n>, or another <folder>.<name>), with commas; else "none"
    local found="" aside
    for aside in "$1".*; do
        if [ -e "$aside" ] || [ -L "$aside" ]; then found="${found:+$found,}$aside"; fi
    done
    echo "${found:-none}"
}

ended_before() {      # ended_before: true if the record says that a chain reached its end after the last
                      # rollout was started (TEST_RUN_DONE comes after the last ROLLOUT_START)
    local line last=""
    if [ ! -f "$RECORD" ]; then return 1; fi
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            $TIME_SHAPE" ROLLOUT_START "*) last="start" ;;
            $TIME_SHAPE" TEST_RUN_DONE "*) last="done" ;;
        esac
    done < "$RECORD"
    [ "$last" = "done" ]
}

check_reference() {   # the step "reference_check": the plan, section 3: `oracle` must pass every scenario,
                      # and each `constant_*` policy exactly one outcome in four. It reads `outcome` and
                      # `passed` of the records that the steps reference_<policy> wrote. It counts records
                      # and computes no rate.
    local policy items=()
    for policy in "${POLICIES[@]}"; do
        items+=("$policy=$OUT/reference/${policy}_$SPLIT.jsonl")
    done
    "$PYTHON" - "${items[@]}" <<'PY'
import json
import sys

wrong = 0
for item in sys.argv[1:]:
    policy, _, path = item.partition("=")
    counts = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                passed, total = counts.get(record["outcome"], (0, 0))
                counts[record["outcome"]] = (passed + (record["passed"] is True), total + 1)
    every = sorted(outcome for outcome, (passed, total) in counts.items() if passed == total)
    none = sorted(outcome for outcome, (passed, total) in counts.items() if passed == 0)
    if policy == "oracle":
        fine = len(counts) == 4 and len(every) == 4
    else:
        fine = len(counts) == 4 and len(every) == 1 and len(none) == 3
    print(f"{policy}: {sum(total for _, total in counts.values())} records in {len(counts)} outcomes; every "
          f"record passes in {every}, and none passes in {none}: "
          + ("as the plan says" if fine else "NOT as the plan says"))
    wrong += not fine
if wrong:
    print("ANALYSIS_PLAN.md, section 3: `oracle` must pass every scenario, and each `constant_*` policy "
          "exactly one outcome in four. The four runs and their reports are saved.")
raise SystemExit(1 if wrong else 0)
PY
}

HOST="$(uname -n 2> /dev/null)"
if [ -z "$HOST" ]; then HOST="unknown"; fi
echo "chain test starts at $(date -u +%FT%TZ), PID $$ on $HOST"
echo "python: $(command -v "$PYTHON")"

# One chain at a time: a lock on the repository folder, kept until this chain and everything that it
# started have ended. No file is written for it.
if ! command -v flock > /dev/null; then
    echo "flock is not on this machine (the package util-linux). Without it the chain cannot make sure that" \
         "it runs alone, so it does not start."
    failed lock
fi
if ! exec 9< .; then
    echo "The repository folder could not be opened for the lock."
    failed lock
fi
if ! flock -n 9; then
    echo "Another chain_test.sh, or a rollout that one started, runs in this repository folder: the lock is" \
         "held. This start did nothing: no check, no rollout, no folder moved. Wait for the other chain to" \
         "end. To see it:  ps -ef | grep -e chain_test.sh -e rollout.py"
    failed lock
fi
echo "lock: this chain holds the lock of the repository folder until it ends; a second start stops at once"

# The first step: the tree must be the frozen one.
run_command freeze_check "$PYTHON" scripts/freeze.py check

# The mode. DRESS sets the split, the folder of everything that is written, and the options of a rehearsal.
case "$DRESS" in
    0)
        SPLIT="test"
        OUT="$MAIN"
        EXTRA=()
        echo "CHAIN_NOTE DRESS=0: the TEST RUN of the plan, on the split $SPLIT, into $OUT. It is run once."
        ;;
    1)
        SPLIT="calib"
        OUT="results/_smoke/test_dress"
        EXTRA=(--families "$OUT/dress_families.txt")
        echo "CHAIN_NOTE DRESS=1: a REHEARSAL on the calibration families ${DRESS_FAMILIES[*]} (the split" \
             "$SPLIT), into $OUT. None of its numbers is a result."
        ;;
    *)
        echo "DRESS must be 0 (the test run) or 1 (a rehearsal), not '$DRESS'"
        failed start
        ;;
esac
RECORD="$OUT/test_run_record.txt"
REASON="${REPEAT_REASON//$'\n'/ }"           # the reason as one line of the record
case "$REPEAT_AFTER_FAULT" in
    0) ;;
    1)
        if [ -z "${REASON//[[:space:]]/}" ]; then
            echo "REPEAT_AFTER_FAULT=1 needs REPEAT_REASON: what the fault was. The reason goes into" \
                 "$RECORD, and NOTES.md must name the repeat (ANALYSIS_PLAN.md, section 4)."
            failed start
        fi
        ;;
    *)
        echo "REPEAT_AFTER_FAULT must be 0 or 1, not '$REPEAT_AFTER_FAULT'"
        failed start
        ;;
esac
echo "MODEL=$MODEL REVISION=$REVISION SCENARIOS=$SCENARIOS TEXT=$TEXT"
echo "SPLIT=$SPLIT OUT=$OUT SAMPLES=$SAMPLES EVAL_SEED=$EVAL_SEED MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "REPEAT_AFTER_FAULT=$REPEAT_AFTER_FAULT REPEAT_REASON=$REASON"
for needed in MANIFEST.json "$TEXT" "$SCENARIOS/summary.json" "$SCENARIOS/dev.jsonl" \
              "$SCENARIOS/$SPLIT.jsonl"; do
    if [ ! -f "$needed" ]; then
        echo "$needed is missing"
        failed start
    fi
done
if [ -e "$RECORD" ] || [ -L "$RECORD" ]; then
    if [ ! -f "$RECORD" ] || [ ! -r "$RECORD" ]; then
        echo "$RECORD is there, and it is not a file that can be read: the chain cannot know what was run."
        failed start
    fi
fi
sha256sum MANIFEST.json "$TEXT" "$SCENARIOS/summary.json" "$SCENARIOS/$SPLIT.jsonl" scripts/gpu/chain_test.sh
TEXT_SHA256="$(file_sha256 "$TEXT")"
MANIFEST_SHA256="$(file_sha256 MANIFEST.json)"
SCRIPT_SHA256="$(file_sha256 scripts/gpu/chain_test.sh)"
CODE_SHA256="$(json_value MANIFEST.json code_sha256)"
if [ -z "$CODE_SHA256" ]; then
    echo "MANIFEST.json names no code_sha256"
    failed start
fi
# More values that the run manifest of every run of this chain, and of every dev run, must name: the
# defaults of scripts/gpu/rollout.py for the rule set, the tools and the prompt, the frozen code, and a
# start without the option that lets a changed tree pass.
MORE_ITEMS=("ruleset=default" "toolset=five" "prompt=final" "code_sha256=$CODE_SHA256"
            "allow_changed_tree=False")

run_command adapters same_as_dev
run_command run_once run_once

mkdir -p "$OUT/tables" "$OUT/monitors" "$OUT/reference" || failed start
if [ "$DRESS" = "1" ]; then
    printf '%s\n' "${DRESS_FAMILIES[@]}" > "$OUT/dress_families.txt" || failed start
fi

for arm in "${ARMS[@]}"; do
    rollout_step "$arm" ${EXTRA[@]+"${EXTRA[@]}"}
    run_command "gate_$arm" gate "$arm"
done

# The adapters once more, now that every rollout has loaded its own: the files and their records must still
# be those of the dev runs.
run_command adapters_after after_rollouts

# All four rollouts come first. The reports are made only if all four runs are complete.
for arm in "${ARMS[@]}"; do
    if ! is_complete "$OUT/${arm}_$SPLIT"; then
        echo "$OUT/${arm}_$SPLIT is not complete: no report is made"
        failed "${arm}_$SPLIT"
    fi
done

for arm in "${ARMS[@]}"; do
    run_command "replay_$arm" \
        "$PYTHON" scripts/replay_check.py --episodes "$OUT/${arm}_$SPLIT/episodes.jsonl" \
            --scenarios "$SCENARIOS" --split "$SPLIT" --text "$TEXT"
done

# One table of the four arms. The plan's comparisons on test: sft minus base, each GRPO seed minus sft.
TABLE="$OUT/tables/${SPLIT}_base_sft_grpo_seed1_seed2"
run_command table \
    "$PYTHON" -m accessdesk.eval.tables --arm "base=$OUT/base_$SPLIT/episodes.jsonl" \
        --arm "sft=$OUT/sft_$SPLIT/episodes.jsonl" --arm "grpo_seed1=$OUT/grpo_seed1_$SPLIT/episodes.jsonl" \
        --arm "grpo_seed2=$OUT/grpo_seed2_$SPLIT/episodes.jsonl" --baseline base --pair base:sft \
        --pair sft:grpo_seed1 --pair sft:grpo_seed2 --out-json "$TABLE.json" --out-md "$TABLE.md"

for arm in "${ARMS[@]}"; do
    run_command "monitors_$arm" write_to "$OUT/monitors/${arm}_$SPLIT.md" \
        "$PYTHON" scripts/monitor_report.py --episodes "$OUT/${arm}_$SPLIT/episodes.jsonl" \
            --scenarios "$SCENARIOS" --split "$SPLIT" --text "$TEXT"
done

for policy in "${POLICIES[@]}"; do
    run_command "reference_$policy" \
        "$PYTHON" scripts/run_scripted.py --scenarios "$SCENARIOS" --split "$SPLIT" --text "$TEXT" \
            --policy "$policy" --out "$OUT/reference/${policy}_$SPLIT.jsonl"
done
run_command reference_check check_reference

# The record of the run: lines are added, never taken away. TEST_RUN_DONE is written when a chain reaches
# its end for the first time after a rollout was started. A later start that only builds the reports again
# writes REPORTS_REBUILT, so that nobody counts it as a second test run.
NOW="$(date -u +%FT%TZ)"
KIND="TEST_RUN_DONE"
ROLLOUTS_RUN="none"
if [ "${#RAN[@]}" -gt 0 ]; then
    ROLLOUTS_RUN="$(IFS=,; echo "${RAN[*]}")"
elif ended_before; then
    KIND="REPORTS_REBUILT"
fi
{
    echo "$NOW $KIND DRESS=$DRESS split=$SPLIT out=$OUT rollouts_run=$ROLLOUTS_RUN"
    echo "$NOW   MANIFEST.json=$MANIFEST_SHA256 code_sha256=$CODE_SHA256" \
         "scripts/gpu/chain_test.sh=$SCRIPT_SHA256"
    for line in "${RUN_LINES[@]}"; do echo "$NOW   $line"; done
} >> "$RECORD" || failed record
echo "CHAIN_NOTE the record of this run was added to $RECORD"
echo "chain test ends at $(date -u +%FT%TZ)"
echo "CHAIN_DONE"
exit 0
