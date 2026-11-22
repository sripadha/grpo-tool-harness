#!/usr/bin/env bash
# Stage 1 as ONE job on the rented machine: the baseline on dev, rejection sampling on train, SFT, the SFT
# model on dev, the table of base against SFT, and the family profile of the SFT model that sets the size
# of a GRPO step.
#
# Start it on the machine, in the repository folder, so that it survives a lost connection:
#     mkdir -p /workspace/logs
#     nohup bash scripts/gpu/chain_stage1.sh > /workspace/logs/stage1.log 2>&1 < /dev/null &
# From the laptop the same is:  _dev/box.sh bg stage1 'bash scripts/gpu/chain_stage1.sh'
# A variable of the list below can be set in front:  OUT=results/try bash scripts/gpu/chain_stage1.sh
#
# The steps, in order, each with its folder under $OUT:
#   base_dev          the untrained model on dev: every family in all four wordings, $SAMPLES attempts each,
#                     run seed $EVAL_SEED
#   base_train        the untrained model on train (the three training wordings), $SAMPLES attempts each,
#                     run seed $TRAIN_ROLLOUT_SEED
#   sft_data          scripts/gpu/sft_data.py chooses the episodes that SFT trains on (no GPU)
#   sft               scripts/gpu/train_sft.py trains the LoRA adapter on them, with the seed $SEED
#   sft_dev           the model with that adapter on dev, with the run seed of base_dev
#   table             python -m accessdesk.eval.tables: base against SFT, into tables/base_vs_sft.json and .md
#   profile_families  scripts/sft_profile.py --pick writes sft_profile/families.txt: the first
#                     $PROFILE_PER_OUTCOME training families of each outcome (100 families)
#   sft_train         the model with the SFT adapter on these families (the three training wordings),
#                     $SAMPLES attempts each, run seed $PROFILE_SEED
#   sft_profile       scripts/sft_profile.py: the pass rate of each family, and the rule of PLAN 5.5 on the
#                     size of a GRPO step, into sft_profile/decision.json and profile.md (no GPU).
#                     scripts/gpu/chain_stage2.sh reads decision.json.
#
# The seeds. base_dev and sft_dev get the same --run-seed ($EVAL_SEED), and so does every later rollout on
# dev. The random state of a rollout comes from the run seed, the pass, the chunk and the turn
# (scripts/gpu/rollout.py), so all arms draw from the same sampling seeds on dev. This does not make
# single episodes of two arms a pair: the seed is set for a batch, and a batch gets smaller as its
# episodes end. The two rollouts on train have their own run seeds, so that they share no sampling seed
# with dev or with each other.
#
# The tokens. Every rollout writes tokens.jsonl beside episodes.jsonl: the token ids and the
# log-probabilities of every episode. sft_data.py needs base_train/tokens.jsonl, so that file must stay. In
# calibration the file had 23,145,856 bytes for 1,536 episodes
# (results/calibration/round3/2b_mixed/tokens.jsonl); at that size per episode the 4,800 episodes of
# base_train give about 72 MB.
#
# Lines to look for in the log:
#   STEP_START <name>               a step starts
#   STEP_DONE <name> <exit code>    it ended
#   STEP_SKIPPED <name>             its folder is complete (its run_manifest.json says "complete": true) and
#                                   was made with the values that this chain has
#   STEP_MOVED_ASIDE <name> <dir>   its folder held an unfinished run; the folder was renamed, not deleted
#   CHAIN_FAILED <name>             the chain failed at this step ("start": before any step)
#   CHAIN_NOTE <text>               something to know: the state of the freeze, a token cap that is not the
#                                   config's default, the result of the family profile
#   CHAIN_DONE                      every step is done
# The chain can be started again after a fault: complete steps are skipped, and the table and the profile
# are always built again. The families of the profile are picked again too, into families.txt.new. That
# file takes the place of sft_profile/families.txt only if there is none, or if it holds other families and
# sft_train is not complete. Before a step is skipped, its run manifest is compared with this chain: the
# model, the revision, the seed, the attempts, the token cap, the text file, the scenario file and what the
# step read. If one differs the chain stops with CHAIN_FAILED <step>: it never goes on with a folder that
# another set of values made, and it changes no file that a step read. The chain stops at the first step
# that fails. Only the table is different: it is made from saved episodes, so if it fails the chain still
# runs the profile steps, and then ends with CHAIN_FAILED table.
# Folders are compared as text. Keep OUT and SCENARIOS written as they were at the first start
# (results/main, not /workspace/.../results/main), and copy results with the same layout under $OUT.
# For the real runs put REQUIRE_FROZEN=1 in front: then the chain starts only on the frozen tree.
# This script stops no machine and copies no results: the machine's timer and the fetch loop do that.

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
REVISION="${REVISION:-15852e8c16360a2fea060d615a32b45270f8a8fc}"
SCENARIOS="${SCENARIOS:-data/scenarios}"
OUT="${OUT:-results/main}"
SEED="${SEED:-0}"                         # seed of the SFT training
EVAL_SEED="${EVAL_SEED:-0}"               # run seed of every rollout on dev, in this chain and in stage 2
TRAIN_ROLLOUT_SEED="${TRAIN_ROLLOUT_SEED:-10}"    # run seed of base_train, the rollout that feeds SFT
PROFILE_SEED="${PROFILE_SEED:-20}"        # run seed of sft_train, the rollout for the family profile
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"      # token cap per turn; empty: the value of accessdesk/config.py
SAMPLES="${SAMPLES:-8}"                   # attempts per scenario
TRAIN_STYLES="${TRAIN_STYLES:-1,2,3}"     # the wording styles of the train split (style 4 is held out)
PROFILE_PER_OUTCOME="${PROFILE_PER_OUTCOME:-25}"  # families of each outcome in the profile: 100 in all
TEXT="${TEXT:-text/model_text.yaml}"
SFT_DATA_OPTIONS="${SFT_DATA_OPTIONS:-}"  # more options for sft_data.py, such as --allow-penalised
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

echo "chain stage 1 starts at $(date -u +%FT%TZ)"
echo "python: $(command -v "$PYTHON")"
# The token cap. Every step gets it as --max-new-tokens, so that all steps use one value. Without
# MAX_NEW_TOKENS it is the value of the config, which is also what scripts/run_scripted.py uses.
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
echo "MODEL=$MODEL REVISION=$REVISION SCENARIOS=$SCENARIOS OUT=$OUT"
echo "SEED=$SEED EVAL_SEED=$EVAL_SEED TRAIN_ROLLOUT_SEED=$TRAIN_ROLLOUT_SEED PROFILE_SEED=$PROFILE_SEED"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS SAMPLES=$SAMPLES TRAIN_STYLES=$TRAIN_STYLES TEXT=$TEXT"
echo "PROFILE_PER_OUTCOME=$PROFILE_PER_OUTCOME SFT_DATA_OPTIONS=$SFT_DATA_OPTIONS"
for needed in "$TEXT" "$SCENARIOS/summary.json" "$SCENARIOS/train.jsonl" "$SCENARIOS/dev.jsonl"; do
    if [ ! -f "$needed" ]; then
        echo "$needed is missing"
        echo "CHAIN_FAILED start"
        exit 1
    fi
done
sha256sum "$TEXT" "$SCENARIOS/summary.json" "$SCENARIOS/train.jsonl" "$SCENARIOS/dev.jsonl"
TEXT_SHA256="$(file_sha256 "$TEXT")"
TRAIN_SHA256="$(file_sha256 "$SCENARIOS/train.jsonl")"
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
mkdir -p "$OUT/tables" || { echo "CHAIN_FAILED start"; exit 1; }

rollout_step base_dev base dev "$EVAL_SEED" "$SAMPLES" ""

rollout_step base_train base_train train "$TRAIN_ROLLOUT_SEED" "$SAMPLES" "" --styles "$TRAIN_STYLES"

case " $SFT_DATA_OPTIONS " in
    *" --allow-penalised "*) ALLOW_PENALISED=True ;;
    *) ALLOW_PENALISED=False ;;
esac
# shellcheck disable=SC2086    # $SFT_DATA_OPTIONS is empty or a list of options
run_step sft_data "$OUT/sft_data" \
    "split=train" "text_sha256=$TEXT_SHA256" "data_sha256=$TRAIN_SHA256" \
    "config/max_new_tokens=$MAX_NEW_TOKENS" "allow_penalised=$ALLOW_PENALISED" \
    "source_run/model=$MODEL" "source_run/revision=$REVISION" "source_run/adapter=" \
    "source_run/run_seed=$TRAIN_ROLLOUT_SEED" "source_run/samples=$SAMPLES" \
    "inputs/episodes/sha256=$(file_sha256 "$OUT/base_train/episodes.jsonl")" \
    "inputs/tokens/sha256=$(file_sha256 "$OUT/base_train/tokens.jsonl")" \
    "chain:scenarios=$SCENARIOS" "chain:SFT_DATA_OPTIONS=$SFT_DATA_OPTIONS" -- \
    "$PYTHON" scripts/gpu/sft_data.py --episodes "$OUT/base_train/episodes.jsonl" \
        --tokens "$OUT/base_train/tokens.jsonl" --scenarios "$SCENARIOS" --out "$OUT/sft_data" \
        $SFT_DATA_OPTIONS

run_step sft "$OUT/sft" \
    "model=$MODEL" "revision=$REVISION" "seed=$SEED" "config/max_new_tokens=$MAX_NEW_TOKENS" \
    "data/text_sha256=$TEXT_SHA256" "data/data_sha256=$TRAIN_SHA256" \
    "data/records/sha256=$(file_sha256 "$OUT/sft_data/sft_records.jsonl")" -- \
    "$PYTHON" scripts/gpu/train_sft.py --model "$MODEL" --revision "$REVISION" --data "$OUT/sft_data" \
        --out "$OUT/sft" --seed "$SEED"

rollout_step sft_dev sft dev "$EVAL_SEED" "$SAMPLES" "$OUT/sft/adapter"

run_report table \
    "$PYTHON" -m accessdesk.eval.tables --arm "base=$OUT/base_dev/episodes.jsonl" \
        --arm "sft=$OUT/sft_dev/episodes.jsonl" --baseline base --pair base:sft \
        --out-json "$OUT/tables/base_vs_sft.json" --out-md "$OUT/tables/base_vs_sft.md"

# The family profile after SFT (PLAN 5.5): the SFT model on 100 training families, and the rule that reads
# the profile. It uses training families only.
# sft_profile/families.txt is what the sft_train rollout reads, so the pick goes into a second file first.
# If that holds the families of the file that is there, the file stays as it is. If it holds other families
# and sft_train is complete, the chain stops and the file stays as it is: a start that is refused changes
# no file that a step read.
PROFILE_FAMILIES="$OUT/sft_profile/families.txt"
run_command profile_families \
    "$PYTHON" scripts/sft_profile.py --scenarios "$SCENARIOS" --per-outcome "$PROFILE_PER_OUTCOME" \
        --pick "$PROFILE_FAMILIES.new"
if [ -f "$PROFILE_FAMILIES" ] \
        && [ "$(file_sha256 "$PROFILE_FAMILIES.new")" = "$(file_sha256 "$PROFILE_FAMILIES")" ]; then
    rm -f "$PROFILE_FAMILIES.new"
elif [ -f "$PROFILE_FAMILIES" ] && is_complete "$OUT/sft_train"; then
    rm -f "$PROFILE_FAMILIES.new"
    echo "sft_train: $PROFILE_FAMILIES holds other families than this chain picks now" \
         "($PROFILE_PER_OUTCOME of each outcome of $SCENARIOS/train.jsonl). The file is left as it is."
    echo "$OUT/sft_train is complete, but it is not the step that this chain would run (the lines above)."
    echo "Give the chain the values of that run, or move the folder away to run the step again."
    echo "CHAIN_FAILED sft_train"
    exit 1
else
    mv -f "$PROFILE_FAMILIES.new" "$PROFILE_FAMILIES" || { echo "CHAIN_FAILED profile_families"; exit 1; }
fi

rollout_step sft_train sft_train train "$PROFILE_SEED" "$SAMPLES" "$OUT/sft/adapter" \
    --families "$PROFILE_FAMILIES" --styles "$TRAIN_STYLES"

run_command sft_profile \
    "$PYTHON" scripts/sft_profile.py --scenarios "$SCENARIOS" --episodes "$OUT/sft_train/episodes.jsonl" \
        --families "$PROFILE_FAMILIES" --out "$OUT/sft_profile"
DECISION="$OUT/sft_profile/decision.json"
echo "CHAIN_NOTE the family profile after SFT: median family pass rate" \
     "$(json_value "$DECISION" median_family_pass_rate) on $(json_value "$DECISION" n_families) families;" \
     "a GRPO step is $(json_value "$DECISION" scenarios_per_step) scenarios x" \
     "$(json_value "$DECISION" group_size) attempts ($DECISION)"

end_chain 1
