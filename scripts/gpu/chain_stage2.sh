#!/usr/bin/env bash
# Stage 2 for ONE seed as one job on the rented machine: GRPO on top of the SFT adapter, the trained model
# on dev, the table and the monitor reports, and the curve. With CHECK it runs one of the two checks before
# the seeds in place of a seed.
#
# Start it on the machine, in the repository folder, so that it survives a lost connection:
#     mkdir -p /workspace/logs
#     SEED=1 nohup bash scripts/gpu/chain_stage2.sh > /workspace/logs/stage2_seed1.log 2>&1 < /dev/null &
# From the laptop the same is:  _dev/box.sh bg stage2_seed1 'SEED=1 bash scripts/gpu/chain_stage2.sh'
# A variable of the list below can be set in front, as SEED is here. Two seeds are two jobs.
# The machine needs $OUT/sft/ (the adapter and its run_manifest.json) and $OUT/sft_profile/decision.json of
# stage 1. With $OUT/base_dev/ and $OUT/sft_dev/ on it too, the chain also builds the table of the three
# arms and their monitor reports.
#
# The steps of a seed, in order, each with its folder under $OUT:
#   grpo_seed<SEED>              scripts/gpu/train_grpo.py: GRPO with the seed $SEED, starting from
#                                $SFT_ADAPTER, with --save-tokens (the token data of every step is kept in
#                                rollouts/, so that the loss of a step can be computed again)
#   grpo_seed<SEED>_dev          the model with the adapter of the last checkpoint on dev: every family in
#                                all four wordings, $SAMPLES attempts each, with the run seed $EVAL_SEED
#   table                        python -m accessdesk.eval.tables: base, SFT and this seed, with SFT minus
#                                base, GRPO minus SFT and GRPO minus base, into
#                                tables/base_sft_grpo_seed<SEED>.json and .md
#   monitors_base, monitors_sft, monitors_grpo_seed<SEED>
#                                scripts/monitor_report.py on the dev episodes of each arm, into monitors/
#   grpo_seed<SEED>_curve_000 to _080
#                                the curve of PLAN 5.6: the fixed dev subset (the set curve_subset of
#                                sets.json: 20 families) in wording style $CURVE_STYLE, $CURVE_SAMPLES
#                                attempts, run seed $EVAL_SEED, for the SFT adapter (step 0) and for the
#                                checkpoints after the steps of $CURVE_STEPS
# The table and the monitor reports of base and SFT are built only if $OUT/base_dev and $OUT/sft_dev are
# complete on this machine. If not, a CHAIN_NOTE says so, and they are built later from the copied episodes.
#
# The size of a GRPO step. The rule of PLAN 5.5 is applied by scripts/sft_profile.py at the end of stage 1,
# and its result is in $DECISION: 16 scenarios x 8 attempts, or 8 scenarios x 16 attempts. This chain reads
# the two numbers and gives them to train_grpo.py. SCENARIOS_PER_STEP and GROUP_SIZE (both) set the step by
# hand in place of the file. Both seeds must get the same step. So a seed without the file and without the
# two variables does not start (CHAIN_FAILED start): copy decision.json to the machine, or set the two
# variables. ALLOW_DEFAULT_STEP=1 lets such a seed take the default of the plan, 16 x 8. A check before the
# seeds takes that default without being told to. A CHAIN_NOTE says which of the three was used.
#
# $EVAL_SEED must be the run seed of the dev rollouts of stage 1, so that base, SFT and GRPO are sampled on
# dev with the same sampling seeds. The chain checks that against $OUT/base_dev and $OUT/sft_dev if they
# are on the machine. It also checks that sft_dev was sampled with $SFT_ADAPTER, the adapter that GRPO
# starts from: the table compares GRPO with the arm sft. If one of the two does not fit, the chain does not
# start (CHAIN_FAILED start). $SEED here is the seed of the training.
#
# Folders are compared as text. Keep OUT, SCENARIOS and SFT_ADAPTER written as they were at the first start
# (results/main, not /workspace/.../results/main), and copy results with the same layout under $OUT.
# For the real runs put REQUIRE_FROZEN=1 in front: then the chain starts only on the frozen tree.
#
# The checks before the seeds (PLAN 5.6). Each is one start of this chain, with its folder under $OUT/checks:
#   CHECK=overfit   overfit_families: scripts/pick_overfit_families.py reads $OVERFIT_EPISODES and writes
#                   checks/overfit_families.txt, two training families with mixed outcomes in style 1.
#                   The default is the rollout of the untrained model, which is there as soon as SFT can
#                   start. The check itself starts from the SFT adapter, so the attempts of the SFT model
#                   ($OUT/sft_train/episodes.jsonl, at the end of stage 1) show better which families are
#                   mixed at its start. The families are chosen only for a check that starts from its first
#                   step. A run in checks/overfit that is complete, that the stop rule ended, or that is
#                   continued from a checkpoint keeps its families file; a start with another
#                   $OVERFIT_EPISODES than that run's is refused (CHAIN_FAILED overfit).
#                   overfit: train_grpo.py --overfit on them, in style 1, seed $OVERFIT_SEED, 30 steps, into
#                   checks/overfit. With 16 episodes in a step one harness error is 6%, above the 1% of
#                   the stop rule, so this check alone runs with the limit $OVERFIT_MAX_HARNESS_ERROR_RATE.
#   CHECK=pilot     pilot: train_grpo.py --pilot, 15 steps at the real settings, seed $PILOT_SEED, into
#                   checks/pilot.
# A check ends with a CHAIN_NOTE that gives the train pass rate and reward of its first and last step.
#
# Lines to look for in the log:
#   STEP_START <name>               a step starts
#   STEP_DONE <name> <exit code>    it ended
#   STEP_SKIPPED <name>             its folder is complete (its run_manifest.json says "complete": true) and
#                                   was made with the values that this chain has
#   STEP_MOVED_ASIDE <name> <dir>   its folder held an unfinished run; the folder was renamed, not deleted
#   CHAIN_FAILED <name>             the chain failed at this step ("start": before any step)
#   CHAIN_NOTE <text>               something to know: the state of the freeze, a token cap that is not the
#                                   config's default, the size of a GRPO step and where it comes from, a
#                                   run that the stop rule ended, a curve step without a checkpoint
#   CHAIN_DONE                      every step is done
# The chain can be started again after a fault: complete steps are skipped, and the table and the monitor
# reports are always built again. Before a step is skipped, its run manifest is compared with this chain (the
# model, the revision, the seed, the attempts, the token cap, the text file, the scenario file, the size of
# a step, the adapter). If one differs the chain stops with CHAIN_FAILED <step>. A GRPO run that has a
# checkpoint is continued from it (--resume); one without a checkpoint is moved aside and started again.
# A GRPO run that the stop rule of the plan ended (exit code 3) is NOT started again: it is reported as
# stopped. It still gets the curve steps for the checkpoints that exist, and then the chain ends with
# CHAIN_FAILED grpo_seed<SEED> and exit code 3, without a dev rollout, a table or a monitor report.
# RESUME_AFTER_FAULT=1 continues such a run from its last checkpoint; use it only after the fault was read.
# The chain stops at the first step that fails. Only the table and the monitor reports are different: they
# are made from saved episodes, so if one fails the chain still runs the curve, and then ends with
# CHAIN_FAILED <that step>.
# This script stops no machine and copies no results: the machine's timer and the fetch loop do that.

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
REVISION="${REVISION:-15852e8c16360a2fea060d615a32b45270f8a8fc}"
SCENARIOS="${SCENARIOS:-data/scenarios}"
OUT="${OUT:-results/main}"
SEED="${SEED:-1}"                         # seed of the GRPO run: scenario order and sampling in training
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"      # token cap per turn; empty: the value of accessdesk/config.py.
                                          # It must be the value of stage 1.
SFT_ADAPTER="${SFT_ADAPTER:-$OUT/sft/adapter}"
EVAL_SEED="${EVAL_SEED:-0}"               # run seed of every rollout on dev; the EVAL_SEED of chain_stage1.sh
SAMPLES="${SAMPLES:-8}"                   # attempts per dev scenario
TEXT="${TEXT:-text/model_text.yaml}"
DECISION="${DECISION:-$OUT/sft_profile/decision.json}"    # the result of the family profile of stage 1
SCENARIOS_PER_STEP="${SCENARIOS_PER_STEP:-}"      # scenarios in a GRPO step; empty: from $DECISION, else 16
GROUP_SIZE="${GROUP_SIZE:-}"              # attempts per scenario in a step; empty: from $DECISION, else 8
ALLOW_DEFAULT_STEP="${ALLOW_DEFAULT_STEP:-0}"     # 1: a seed may take 16 x 8 without $DECISION and without
                                          # the two variables above
GRPO_OPTIONS="${GRPO_OPTIONS:-}"          # more options for train_grpo.py, such as --lr 3e-5 after the pilot.
                                          # Give the same ones at a second start.
CURVE_STEPS="${CURVE_STEPS:-0 20 40 60 80}"       # the curve: 0 is the SFT adapter, the others checkpoints
CURVE_STYLE="${CURVE_STYLE:-1}"           # the one seen wording style of the curve
CURVE_SAMPLES="${CURVE_SAMPLES:-4}"       # attempts per scenario of the curve
CHECK="${CHECK:-}"                        # empty: a seed. overfit or pilot: that check before the seeds
OVERFIT_EPISODES="${OVERFIT_EPISODES:-$OUT/base_train/episodes.jsonl}"   # where the overfit families are
                                          # chosen from; $OUT/sft_train/episodes.jsonl is the SFT model's own
OVERFIT_SEED="${OVERFIT_SEED:-101}"       # seed of the overfit check
OVERFIT_MAX_HARNESS_ERROR_RATE="${OVERFIT_MAX_HARNESS_ERROR_RATE:-0.1}"  # for the overfit check only
PILOT_SEED="${PILOT_SEED:-100}"           # seed of the pilot
RESUME_AFTER_FAULT="${RESUME_AFTER_FAULT:-0}"     # 1: continue a run that the stop rule ended
PYTHON="${PYTHON:-python}"
REQUIRE_FROZEN="${REQUIRE_FROZEN:-0}"    # 1: start only if the tree is as MANIFEST.json says

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || { echo "CHAIN_FAILED start"; exit 1; }
if [ -r /workspace/env.sh ]; then . /workspace/env.sh; fi      # the machine's Python environment
set -u

# ---- The part below, up to "end of the shared part", is the same in chain_stage1.sh and chain_stage2.sh.
RECORD_NAME="chain_record.txt"      # the chain's own note in the folder of a step that it ran
FAILED_REPORT=""                    # the first report step that failed, and its exit code
FAILED_CODE=0

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

move_aside() {        # move_aside <name> <folder>: rename a folder that holds an unfinished run
    local aside
    aside="$2.unfinished.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$2" "$aside" || { echo "CHAIN_FAILED $1"; exit 1; }
    echo "STEP_MOVED_ASIDE $1 $aside"
}

skip_if_done() {      # skip_if_done <name> <folder> <item>...: true if the step is complete and fits this
                      # chain (the items are those of `fits`): the caller skips it. A complete step that
                      # does not fit ends the chain. If the step must run, a folder with an unfinished run
                      # is renamed first.
    local name="$1" folder="$2"; shift 2
    if is_complete "$folder"; then
        if ! fits "$name" "$folder" "$@"; then
            echo "$folder is complete, but it is not the step that this chain would run (the lines above)."
            echo "Give the chain the values of that run, or move the folder away to run the step again."
            echo "CHAIN_FAILED $name"
            exit 1
        fi
        echo "STEP_SKIPPED $name"
        return 0
    fi
    if [ -e "$folder" ]; then move_aside "$name" "$folder"; fi
    return 1
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

run_report() {        # run_report <name> <command...>: run a report that is made from saved episodes, without
                      # a GPU. If it fails, the chain goes on with its next steps and ends as failed
                      # (`end_chain`): no GPU step is lost for a report that can be made again.
    local code=0
    timed "$@" || code=$?
    if [ "$code" -ne 0 ] && [ -z "$FAILED_REPORT" ]; then
        FAILED_REPORT="$1"
        FAILED_CODE="$code"
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

run_step() {          # run_step <name> <folder> [<item>...] -- <command...>: skip a complete step that fits
                      # this chain (the items are those of `fits`); else run it and note the chain's items
    local name="$1" folder="$2"; shift 2
    local items=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do items+=("$1"); shift; done
    shift
    if skip_if_done "$name" "$folder" ${items[@]+"${items[@]}"}; then return 0; fi
    run_command "$name" "$@"
    chain_lines ${items[@]+"${items[@]}"} > "$folder/$RECORD_NAME"
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

rollout_step() {      # rollout_step <name> <arm> <split> <run seed> <attempts> <adapter> [<option>...]
                      # One rollout into $OUT/<name>. <adapter> is the folder of a LoRA adapter, or "" for
                      # the untrained model. Every rollout of a chain gets the same model, revision, text
                      # file, scenario folder and token cap. The options are more options of
                      # scripts/gpu/rollout.py: --styles, --families, --set.
    local name="$1" arm="$2" split="$3" seed="$4" samples="$5" adapter="$6"; shift 6
    local with_adapter=()
    if [ -n "$adapter" ]; then with_adapter=(--adapter "$adapter"); fi
    rollout_items "$arm" "$split" "$seed" "$samples" "$adapter" "$@"
    run_step "$name" "$OUT/$name" "${ROLLOUT_ITEMS[@]}" -- \
        "$PYTHON" scripts/gpu/rollout.py --model "$MODEL" --revision "$REVISION" \
            ${with_adapter[@]+"${with_adapter[@]}"} --text "$TEXT" --scenarios "$SCENARIOS" \
            --split "$split" --samples "$samples" --run-seed "$seed" --max-new-tokens "$MAX_NEW_TOKENS" \
            --arm "$arm" --out "$OUT/$name" "$@"
}

end_chain() {         # end_chain <stage>: the last lines of the log
    echo "chain stage $1 ends at $(date -u +%FT%TZ)"
    if [ -n "$FAILED_REPORT" ]; then
        echo "CHAIN_FAILED $FAILED_REPORT"
        exit "$FAILED_CODE"
    fi
    echo "CHAIN_DONE"
    exit 0
}
# ---- end of the shared part

GRPO_STATE=""         # set by grpo_step: "complete", or "stopped" if the stop rule ended the run

grpo_step() {         # grpo_step <name> <folder> <mode> <seed> [<item>...] -- [<option>...]: one run of
                      # scripts/gpu/train_grpo.py. <mode> is what its run manifest calls the run: seed,
                      # pilot or overfit. The items are those of `fits`, for this run only. The options
                      # are more options of train_grpo.py, for this run only.
    local name="$1" folder="$2" mode="$3" seed="$4"; shift 4
    local items=("model=$MODEL" "revision=$REVISION" "seed=$seed" "mode=$mode"
                 "config/max_new_tokens=$MAX_NEW_TOKENS" "text_sha256=$TEXT_SHA256"
                 "split_file_sha256=$TRAIN_SHA256" "scenario_folder=$SCENARIOS"
                 "start_adapter/files/adapter_model.safetensors/sha256=$ADAPTER_SHA256")
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do items+=("$1"); shift; done
    shift
    items+=("chain:options=$*" "chain:GRPO_OPTIONS=$GRPO_OPTIONS")
    local resume=() code=0
    GRPO_STATE="complete"
    if skip_if_done_grpo "$name" "$folder" "${items[@]}"; then return 0; fi
    if [ -e "$folder" ]; then
        if [ "$(manifest_value "$folder" status)" = "stopped_by_rule" ] || [ -f "$folder/stopped.json" ]; then
            if [ "$RESUME_AFTER_FAULT" = "1" ] && [ -f "$folder/latest.json" ]; then
                resume=(--resume --allow-resume-after-fault)
                echo "CHAIN_NOTE RESUME_AFTER_FAULT=1: the run in $folder, which the stop rule ended, is" \
                     "continued from its last checkpoint. The write-up must say so."
            else
                GRPO_STATE="stopped"
                echo "the stop rule ended the run in $folder: $(manifest_value "$folder" fault reason)"
                echo "It is reported as stopped and is not started again by this chain."
                return 0
            fi
        elif [ -f "$folder/latest.json" ] && [ -f "$folder/run_manifest.json" ]; then
            resume=(--resume)
            echo "$folder holds a checkpoint: the run is continued from it"
        else
            move_aside "$name" "$folder"
        fi
    fi
    # shellcheck disable=SC2086    # $GRPO_OPTIONS is empty or a list of options
    timed "$name" \
        "$PYTHON" scripts/gpu/train_grpo.py --model "$MODEL" --revision "$REVISION" --adapter "$SFT_ADAPTER" \
            --text "$TEXT" --scenarios "$SCENARIOS" --out "$folder" --seed "$seed" \
            --max-new-tokens "$MAX_NEW_TOKENS" --save-tokens "$@" $GRPO_OPTIONS \
            ${resume[@]+"${resume[@]}"} || code=$?
    if [ "$code" -eq 3 ] && [ "$(manifest_value "$folder" status)" = "stopped_by_rule" ]; then
        GRPO_STATE="stopped"        # exit code 3 of train_grpo.py: the stop rule of the plan ended the run
        echo "the stop rule ended the run in $folder: $(manifest_value "$folder" fault reason)"
        return 0
    fi
    if [ "$code" -ne 0 ]; then
        echo "CHAIN_FAILED $name"
        exit "$code"
    fi
    chain_lines "${items[@]}" > "$folder/$RECORD_NAME"
}

skip_if_done_grpo() { # as skip_if_done, but a folder with an unfinished run is left as it is: grpo_step
                      # decides what happens to it
    if is_complete "$2"; then
        skip_if_done "$@"
        return 0
    fi
    return 1
}

keeps_its_run() {     # keeps_its_run <folder>: true if grpo_step will not start the run in <folder> from its
                      # first step: the run is complete, the stop rule ended it, or it has a checkpoint.
                      # These are the cases in which grpo_step does not move the folder aside.
    is_complete "$1" || [ "$(manifest_value "$1" status)" = "stopped_by_rule" ] || [ -f "$1/stopped.json" ] \
        || { [ -f "$1/latest.json" ] && [ -f "$1/run_manifest.json" ]; }
}

curve_steps() {       # the curve of PLAN 5.6: one small rollout on dev for the SFT adapter (step 0) and for
                      # each checkpoint of $CURVE_STEPS that exists, all with the same run seed
    local step label adapter
    for step in $CURVE_STEPS; do
        step="$((10#$step))"
        printf -v label '%03d' "$step"
        if [ "$step" -eq 0 ]; then
            adapter="$SFT_ADAPTER"
        else
            adapter="$GRPO_OUT/$(printf 'checkpoints/step_%05d/adapter' "$step")"
        fi
        if [ ! -f "$adapter/adapter_model.safetensors" ]; then
            echo "CHAIN_NOTE no curve step $step of $GRPO_NAME: there is no checkpoint $adapter"
            continue
        fi
        rollout_step "${GRPO_NAME}_curve_$label" "${GRPO_NAME}_curve_$label" dev "$EVAL_SEED" \
            "$CURVE_SAMPLES" "$adapter" --set curve_subset --styles "$CURVE_STYLE"
    done
}

monitor_step() {      # monitor_step <arm> <episodes file>: the monitor report of one arm on dev
    run_report "monitors_$1" write_to "$OUT/monitors/$1_dev.md" \
        "$PYTHON" scripts/monitor_report.py --episodes "$2" --scenarios "$SCENARIOS" --split dev \
            --text "$TEXT"
}

end_check() {         # end_check <name> <folder>: the last lines of a check before the seeds
    if [ "$GRPO_STATE" = "stopped" ]; then
        echo "chain stage 2 ends at $(date -u +%FT%TZ)"
        echo "CHAIN_FAILED $1"
        exit 3
    fi
    echo "CHAIN_NOTE $1, numbers of the training episodes of one step, from $2/run_manifest.json:" \
         "pass rate $(manifest_value "$2" summary first_step pass_rate) at the first step and" \
         "$(manifest_value "$2" summary last_step pass_rate) at the last step; reward" \
         "$(manifest_value "$2" summary first_step mean_reward) and" \
         "$(manifest_value "$2" summary last_step mean_reward). Every step is in $2/train_log.jsonl."
    end_chain 2
}

GRPO_NAME="grpo_seed$SEED"
GRPO_OUT="$OUT/$GRPO_NAME"
DEV_NAME="${GRPO_NAME}_dev"

echo "chain stage 2 starts at $(date -u +%FT%TZ)"
echo "python: $(command -v "$PYTHON")"
# The token cap. Every step gets it as --max-new-tokens, so that all steps use one value. Without
# MAX_NEW_TOKENS it is the value of the config. train_grpo.py refuses a cap that the SFT run did not use.
CONFIG_CAP="$("$PYTHON" -c 'from accessdesk.config import RunConfig; print(RunConfig().max_new_tokens)')"
if [ -z "$CONFIG_CAP" ]; then
    echo "the token cap of accessdesk/config.py could not be read with $PYTHON"
    echo "CHAIN_FAILED start"
    exit 1
fi
if [ -z "$MAX_NEW_TOKENS" ]; then
    MAX_NEW_TOKENS="$CONFIG_CAP"
elif [ "$MAX_NEW_TOKENS" != "$CONFIG_CAP" ]; then
    echo "CHAIN_NOTE the token cap of this chain is $MAX_NEW_TOKENS; the default of accessdesk/config.py is" \
         "$CONFIG_CAP. Every step gets --max-new-tokens $MAX_NEW_TOKENS."
fi
echo "MODEL=$MODEL REVISION=$REVISION SCENARIOS=$SCENARIOS OUT=$OUT SEED=$SEED EVAL_SEED=$EVAL_SEED"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS SAMPLES=$SAMPLES SFT_ADAPTER=$SFT_ADAPTER TEXT=$TEXT"
echo "DECISION=$DECISION SCENARIOS_PER_STEP=$SCENARIOS_PER_STEP GROUP_SIZE=$GROUP_SIZE" \
     "ALLOW_DEFAULT_STEP=$ALLOW_DEFAULT_STEP"
echo "GRPO_OPTIONS=$GRPO_OPTIONS"
echo "CURVE_STEPS=$CURVE_STEPS CURVE_STYLE=$CURVE_STYLE CURVE_SAMPLES=$CURVE_SAMPLES"
echo "CHECK=$CHECK OVERFIT_EPISODES=$OVERFIT_EPISODES OVERFIT_SEED=$OVERFIT_SEED PILOT_SEED=$PILOT_SEED"
echo "OVERFIT_MAX_HARNESS_ERROR_RATE=$OVERFIT_MAX_HARNESS_ERROR_RATE RESUME_AFTER_FAULT=$RESUME_AFTER_FAULT"
case "$CHECK" in
    ""|overfit|pilot) ;;
    *)
        echo "CHECK must be empty, overfit or pilot, not $CHECK"
        echo "CHAIN_FAILED start"
        exit 1
        ;;
esac
for needed in "$TEXT" "$SCENARIOS/summary.json" "$SCENARIOS/train.jsonl" "$SCENARIOS/dev.jsonl" \
              "$SCENARIOS/sets.json" "$SFT_ADAPTER/adapter_config.json" \
              "$SFT_ADAPTER/adapter_model.safetensors"; do
    if [ ! -f "$needed" ]; then
        echo "$needed is missing"
        echo "CHAIN_FAILED start"
        exit 1
    fi
done
sha256sum "$TEXT" "$SCENARIOS/summary.json" "$SCENARIOS/train.jsonl" "$SCENARIOS/dev.jsonl" \
    "$SCENARIOS/sets.json" "$SFT_ADAPTER/adapter_config.json" "$SFT_ADAPTER/adapter_model.safetensors"
TEXT_SHA256="$(file_sha256 "$TEXT")"
TRAIN_SHA256="$(file_sha256 "$SCENARIOS/train.jsonl")"
ADAPTER_SHA256="$(file_sha256 "$SFT_ADAPTER/adapter_model.safetensors")"
# Is the tree the one that MANIFEST.json describes (accessdesk/freeze.py)? The answer is printed. Each
# Python script writes it into its run manifest too. With REQUIRE_FROZEN=1 the chain starts only on a
# tree that is frozen and has not changed since.
FROZEN="$("$PYTHON" - <<'PY'
try:
    from accessdesk.freeze import frozen_state
    state = frozen_state()
    if state["manifest_sha256"] is None:
        print("not frozen: there is no MANIFEST.json")
    elif state["matches"]:
        print("as frozen: it matches MANIFEST.json " + state["manifest_sha256"][:12])
    else:
        print("NOT as frozen: " + "; ".join(state["differences"][:5]))
except Exception as error:
    print(f"of unknown state ({type(error).__name__}: {error})")
PY
)"
echo "CHAIN_NOTE the tree is $FROZEN"
if [ "$REQUIRE_FROZEN" = "1" ] && [ "${FROZEN#as frozen}" = "$FROZEN" ]; then
    echo "REQUIRE_FROZEN=1: the chain does not start on this tree"
    echo "CHAIN_FAILED start"
    exit 1
fi

# The size of a GRPO step: by hand, or from the family profile of stage 1, or the default of the plan.
case "$GRPO_OPTIONS" in
    *--scenarios-per-step*|*--group-size*)
        echo "GRPO_OPTIONS must not hold --scenarios-per-step or --group-size: set SCENARIOS_PER_STEP and" \
             "GROUP_SIZE, so that the chain knows the size of a step"
        echo "CHAIN_FAILED start"
        exit 1
        ;;
esac
if [ -n "$SCENARIOS_PER_STEP" ] || [ -n "$GROUP_SIZE" ]; then
    if [ -z "$SCENARIOS_PER_STEP" ] || [ -z "$GROUP_SIZE" ]; then
        echo "SCENARIOS_PER_STEP and GROUP_SIZE must be given together"
        echo "CHAIN_FAILED start"
        exit 1
    fi
    STEP_SOURCE="set by hand with SCENARIOS_PER_STEP and GROUP_SIZE"
    if [ -f "$DECISION" ]; then
        STEP_SOURCE="$STEP_SOURCE; the family profile ($DECISION) gives"
        STEP_SOURCE="$STEP_SOURCE $(json_value "$DECISION" scenarios_per_step) x"
        STEP_SOURCE="$STEP_SOURCE $(json_value "$DECISION" group_size) and is NOT used"
    fi
elif [ -f "$DECISION" ]; then
    SCENARIOS_PER_STEP="$(json_value "$DECISION" scenarios_per_step)"
    GROUP_SIZE="$(json_value "$DECISION" group_size)"
    STEP="$SCENARIOS_PER_STEP x $GROUP_SIZE"
    if [ "$STEP" != "16 x 8" ] && [ "$STEP" != "8 x 16" ]; then
        echo "$DECISION gives a step of '$SCENARIOS_PER_STEP' scenarios x '$GROUP_SIZE' attempts: the rule" \
             "of PLAN 5.5 can give only 16 x 8 and 8 x 16"
        echo "CHAIN_FAILED start"
        exit 1
    fi
    PROFILE_ADAPTER="$(json_value "$DECISION" source adapter_sha256 adapter_model.safetensors)"
    if [ -z "$PROFILE_ADAPTER" ]; then
        echo "$DECISION does not name the SHA-256 of the adapter that was profiled: not compared"
    elif [ "$PROFILE_ADAPTER" != "$ADAPTER_SHA256" ]; then
        echo "$DECISION is the profile of another adapter than $SFT_ADAPTER (SHA-256 of" \
             "adapter_model.safetensors: $PROFILE_ADAPTER in the profile, $ADAPTER_SHA256 here)"
        echo "CHAIN_FAILED start"
        exit 1
    fi
    STEP_SOURCE="from the family profile after SFT ($DECISION): median family pass rate"
    STEP_SOURCE="$STEP_SOURCE $(json_value "$DECISION" median_family_pass_rate); rule:"
    STEP_SOURCE="$STEP_SOURCE $(json_value "$DECISION" rule)"
else
    # A seed never takes the default step without being told to: the other seed may run on a machine that
    # has the profile, and then the two seeds would have steps of different sizes.
    if [ -z "$CHECK" ] && [ "$ALLOW_DEFAULT_STEP" != "1" ]; then
        echo "The size of a GRPO step is not known: $DECISION is missing, and SCENARIOS_PER_STEP and" \
             "GROUP_SIZE are not set. Copy decision.json of stage 1 to this machine, or set the two" \
             "variables to the step of the other seed. ALLOW_DEFAULT_STEP=1 takes the default of the" \
             "plan, 16 x 8, without the rule of PLAN 5.5."
        echo "CHAIN_FAILED start"
        exit 1
    fi
    SCENARIOS_PER_STEP=16
    GROUP_SIZE=8
    STEP_SOURCE="the default of PLAN 5.6: there is no family profile ($DECISION is missing), so the rule"
    STEP_SOURCE="$STEP_SOURCE of PLAN 5.5 was NOT applied"
fi
echo "CHAIN_NOTE a GRPO step is $SCENARIOS_PER_STEP scenarios x $GROUP_SIZE attempts, $STEP_SOURCE"

# The dev rollouts of stage 1, if they are on this machine, must be the ones that chain_stage1.sh would make
# with the values of this chain: the same model, attempts, run seed, token cap, text file and dev file. Else
# the arms are not paired. And sft_dev must be of $SFT_ADAPTER: the table calls sft_dev the arm sft and
# compares GRPO with it, so it must be the model that GRPO starts from. Stage 1 noted the SHA-256 of the
# adapter's two files in sft_dev ($RECORD_NAME), and they are compared with the files of $SFT_ADAPTER. Then
# the folder name in the run manifest is not compared: the results of stage 1 may be a copy in another
# place. Without the note only the folder name can be compared, and a line says so.
HAVE_STAGE1=1
for arm_folder in base_dev sft_dev; do
    if ! is_complete "$OUT/$arm_folder"; then
        HAVE_STAGE1=0
        continue
    fi
    if [ "$arm_folder" = "base_dev" ]; then
        rollout_items base dev "$EVAL_SEED" "$SAMPLES" ""
    else
        rollout_items sft dev "$EVAL_SEED" "$SAMPLES" "$SFT_ADAPTER"
        if [ -f "$OUT/sft_dev/$RECORD_NAME" ]; then
            WITHOUT_FOLDER=()
            for item in "${ROLLOUT_ITEMS[@]}"; do
                if [ "$item" != "adapter=$SFT_ADAPTER" ]; then WITHOUT_FOLDER+=("$item"); fi
            done
            ROLLOUT_ITEMS=("${WITHOUT_FOLDER[@]}")
        fi
    fi
    if ! fits "$arm_folder" "$OUT/$arm_folder" "${ROLLOUT_ITEMS[@]}"; then
        echo "$OUT/$arm_folder of stage 1 was not made with the values of this chain (the lines above)."
        echo "EVAL_SEED, SAMPLES, MAX_NEW_TOKENS, TEXT and SCENARIOS must be those of chain_stage1.sh."
        if [ "$arm_folder" = "sft_dev" ]; then
            echo "If the lines above name the adapter: $OUT/sft_dev was not sampled with the adapter that" \
                 "GRPO starts from (SFT_ADAPTER=$SFT_ADAPTER), so a table of base, SFT and GRPO would" \
                 "compare GRPO with another SFT model. Give the adapter of sft_dev as SFT_ADAPTER, or move" \
                 "$OUT/sft_dev away: then this chain builds no table."
        fi
        echo "CHAIN_FAILED start"
        exit 1
    fi
done
mkdir -p "$OUT/tables" "$OUT/monitors" "$OUT/checks" || { echo "CHAIN_FAILED start"; exit 1; }

# The two checks before the seeds. Each ends the chain.
if [ "$CHECK" = "overfit" ]; then
    OVERFIT_FILE="$OUT/checks/overfit_families.txt"
    if keeps_its_run "$OUT/checks/overfit"; then
        # The families file is what the run in the folder read. It is written only for a check that starts
        # from its first step.
        echo "$OUT/checks/overfit holds a run that is not started again from its first step: its families" \
             "($OVERFIT_FILE) are not chosen again"
        CHOSEN_FROM="$(json_value "$OVERFIT_FILE.json" source episodes path)"
        if [ -z "$CHOSEN_FROM" ]; then
            echo "overfit: $OVERFIT_FILE.json does not name the episodes that the families were chosen from:" \
                 "not compared"
        elif [ "$CHOSEN_FROM" != "$OVERFIT_EPISODES" ]; then
            echo "overfit: the families of the run in $OUT/checks/overfit were chosen from $CHOSEN_FROM; this" \
                 "chain has OVERFIT_EPISODES=$OVERFIT_EPISODES"
            echo "Give the chain the value of that run, or move $OUT/checks/overfit away to run the check" \
                 "on other families."
            echo "CHAIN_FAILED overfit"
            exit 1
        fi
    else
        if [ "$OVERFIT_EPISODES" != "$OUT/sft_train/episodes.jsonl" ] && is_complete "$OUT/sft_train"; then
            echo "CHAIN_NOTE the overfit families are chosen from $OVERFIT_EPISODES. The check starts from" \
                 "the SFT adapter, and $OUT/sft_train holds the attempts of that model on 100 training" \
                 "families: OVERFIT_EPISODES=$OUT/sft_train/episodes.jsonl chooses from them."
        fi
        run_command overfit_families \
            "$PYTHON" scripts/pick_overfit_families.py --episodes "$OVERFIT_EPISODES" --style 1 \
                --out "$OVERFIT_FILE"
    fi
    grpo_step overfit "$OUT/checks/overfit" overfit "$OVERFIT_SEED" \
        "grpo_config/group_size=$GROUP_SIZE" "overfit/file/sha256=$(file_sha256 "$OVERFIT_FILE")" -- \
        --overfit "$OVERFIT_FILE" --styles 1 --group-size "$GROUP_SIZE" \
        --max-harness-error-rate "$OVERFIT_MAX_HARNESS_ERROR_RATE"
    end_check overfit "$OUT/checks/overfit"
fi
if [ "$CHECK" = "pilot" ]; then
    grpo_step pilot "$OUT/checks/pilot" pilot "$PILOT_SEED" \
        "grpo_config/n_scenarios=$SCENARIOS_PER_STEP" "grpo_config/group_size=$GROUP_SIZE" -- \
        --pilot --scenarios-per-step "$SCENARIOS_PER_STEP" --group-size "$GROUP_SIZE"
    end_check pilot "$OUT/checks/pilot"
fi

# A seed. What happens to a folder that holds an unfinished run depends on how that run ended (grpo_step).
grpo_step "$GRPO_NAME" "$GRPO_OUT" seed "$SEED" \
    "grpo_config/n_scenarios=$SCENARIOS_PER_STEP" "grpo_config/group_size=$GROUP_SIZE" -- \
    --scenarios-per-step "$SCENARIOS_PER_STEP" --group-size "$GROUP_SIZE"

if [ "$GRPO_STATE" = "stopped" ]; then
    echo "CHAIN_NOTE the stop rule ended $GRPO_NAME. The curve steps run for the checkpoints that exist." \
         "There is no dev rollout, no table and no monitor report of this run."
    curve_steps
    echo "chain stage 2 ends at $(date -u +%FT%TZ)"
    echo "CHAIN_FAILED $GRPO_NAME"
    exit 3
fi

# The adapter of the last checkpoint of the complete run; train_grpo.py wrote its place into the manifest.
FINAL_ADAPTER="$GRPO_OUT/$(manifest_value "$GRPO_OUT" final_adapter folder)"
if [ ! -f "$FINAL_ADAPTER/adapter_model.safetensors" ]; then
    echo "the final adapter of $GRPO_OUT is missing: $FINAL_ADAPTER"
    echo "CHAIN_FAILED $DEV_NAME"
    exit 1
fi

rollout_step "$DEV_NAME" "$GRPO_NAME" dev "$EVAL_SEED" "$SAMPLES" "$FINAL_ADAPTER"

# The reports come before the curve: they take seconds, and they are the result of the night.
if [ "$HAVE_STAGE1" = "1" ]; then
    run_report table \
        "$PYTHON" -m accessdesk.eval.tables --arm "base=$OUT/base_dev/episodes.jsonl" \
            --arm "sft=$OUT/sft_dev/episodes.jsonl" --arm "$GRPO_NAME=$OUT/$DEV_NAME/episodes.jsonl" \
            --baseline base --pair base:sft --pair "sft:$GRPO_NAME" --pair "base:$GRPO_NAME" \
            --out-json "$OUT/tables/base_sft_$GRPO_NAME.json" --out-md "$OUT/tables/base_sft_$GRPO_NAME.md"
    monitor_step base "$OUT/base_dev/episodes.jsonl"
    monitor_step sft "$OUT/sft_dev/episodes.jsonl"
else
    echo "CHAIN_NOTE no table, and no monitor report of base and SFT: $OUT/base_dev and $OUT/sft_dev of" \
         "stage 1 are not both complete on this machine. Build them where the episodes of all three arms are."
fi
monitor_step "$GRPO_NAME" "$OUT/$DEV_NAME/episodes.jsonl"

curve_steps

end_chain 2
