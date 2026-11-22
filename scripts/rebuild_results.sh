#!/usr/bin/env bash
# Make every table value and the figure again from the saved attempts and logs, with no GPU, into a fresh
# folder, and compare each made file with the saved one. Written after the freeze (2026-11-19). It only
# reads saved runs: it writes into the fresh folder and nowhere else. It never reads _sealed/.
#
#     scripts/rebuild_results.sh                       # fresh folder: a new temporary folder
#     scripts/rebuild_results.sh --out DIR             # fresh folder: DIR (it must be empty or new)
#     scripts/rebuild_results.sh --only tables,figure  # only these steps (names below)
#     scripts/rebuild_results.sh --list                # print what would be made and compared; run nothing
#
# Options: --results DIR (the saved runs; default results/main), --scenarios DIR (default data/scenarios),
# --text FILE (default text/model_text.yaml), --manifest-history FILE (default results/manifest_history.csv).
# The environment variable PYTHON names the interpreter (default .venv/bin/python). Run it from any folder:
# it works in the top folder of the repository.
#
# results/manifest_history.csv lists every state of MANIFEST.json so far: one row per state (columns sha256,
# state, utc, source). The main session adds a row after each amendment (scripts/freeze.py amend), with the
# SHA-256 of the new MANIFEST.json. If the present MANIFEST.json is not a row of the file, this command says
# so in one line (MANIFEST HISTORY), counts it as a missing input and ends with exit code 1.
#
# The steps (names for --only), with the calls that made the saved files:
#   tables            python -m accessdesk.eval.tables, the calls of scripts/gpu/chain_stage1.sh (base_vs_sft)
#                     and chain_stage2.sh (base_sft_grpo_seed<N>); and the call of chain_test.sh
#                     (test_base_sft_grpo_seed1_seed2) when the four test runs exist
#   monitors          scripts/monitor_report.py on <arm>_dev and, when it exists, <arm>_test of the four arms
#   second_desk       scripts/second_desk_report.py on the eight <arm>_shift_differ and _shift_control runs
#   formula           scripts/formula_audit.py on every run folder with a train_log.jsonl (results and checks)
#   positive_control  scripts/audit_positive_control.py on grpo_seed1_dev and grpo_seed2_dev
#   replay            scripts/replay_check.py on every run folder with episodes.jsonl and a run manifest, with
#                     the split and the rule set (--ruleset) of its run manifest; and with --split train on
#                     every rollouts/step_*.episodes.jsonl of a training run (one summary line per run)
#   reference         scripts/run_scripted.py for every saved reference/<policy>_<split>.jsonl: the call that
#                     its run manifest records ("command"), with only the value of --out changed
#   sft_profile       scripts/sft_profile.py: the pick of the families, and the profile of sft_train
#   audit_sheet       scripts/audit_sheet.py make ... --seed 0 into a folder of its own; only the SHA-256 of
#                     the sheet and of the key are compared with the saved files, and the made files are
#                     deleted at once. Nothing of their content is printed. The labels form is not compared.
#   figure            scripts/make_figure.py
#   training          scripts/training_report.py
#   failures          scripts/failure_report.py (and --split test when the four test runs exist)
#   stretch           scripts/stretch_numbers.py (its document results/stretch.md too)
#
# The fresh folder mirrors the parent of --results: <fresh>/main/... stands for results/main/..., and
# <fresh>/stretch.md for results/stretch.md. The output of each command goes into <fresh>/logs/.
#
# How a made file is compared with the saved file (one line per file):
#   equal               the bytes are the same.
#   equal by rule       the bytes differ, but the lines are the same after the stated rules, and only those
#                       lines differ:
#                       (P) in the made file, the path of the fresh folder is replaced by the parent of
#                           --results (results), since a file that names its own place names the fresh folder.
#                           Both forms of that path are replaced: the absolute path, and, when the fresh folder
#                           lies inside the repository, the path relative to the top folder of the repository
#                           (the scripts write that form for a folder inside the repository).
#                       (T) in both files, each ISO-8601 time (2026-11-19T09:31:00Z and the like) is replaced
#                           by <time>.
#                       The count of lines that a rule changed is printed.
#   equal by rule M     only for a run manifest (*run_manifest.json): the field frozen.manifest_sha256 may
#                       differ from the saved one if and only if all of these hold:
#                         - the saved value is a row of results/manifest_history.csv (a past or present state
#                           of MANIFEST.json);
#                         - the made value is the SHA-256 of the present MANIFEST.json;
#                         - frozen.code_sha256 and code_sha256 are equal in both files.
#                       So the code that made the file did not change; only MANIFEST.json changed after the
#                       saved file was made, from one recorded state to the present one (for example by an
#                       amendment that adds files outside the code hash). Else the file is DIFFERENT. Every
#                       other field is compared as above (rules P and T). The line prints both manifest
#                       hashes and the state of the saved one.
#   DIFFERENT           the lines differ after the rules; the first different lines are printed. A file that
#                       is not text (PNG, PDF) is compared by its bytes only. When the test runs exist, a file
#                       that may hold a number of the test set (a name with "test", and the files of the
#                       report pieces failures and stretch) is reported as equal or DIFFERENT with no line of
#                       its content.
#   NEW                 made, but there is no saved file (for example a run that was added later).
#   SAVED, NOT MADE     a saved file of a folder that this command makes, which the command did not make.
# A replay check with no saved output prints its result: no difference, or DIFFERENT.
#
# Exit code: 0 if every file is equal (or equal by rule) and every command ran; 1 if a file differs, a
# saved file was not made, an input is missing or a command failed; 2 for a wrong option.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
PY="${PYTHON:-.venv/bin/python}"

RESULTS="results/main"
SCENARIOS="data/scenarios"
TEXT="text/model_text.yaml"
HISTORY="results/manifest_history.csv"
OUT=""
ONLY=""
LIST=0
ALL_STEPS="tables monitors second_desk formula positive_control replay reference sft_profile audit_sheet figure training failures stretch"
ARMS=(base sft grpo_seed1 grpo_seed2)

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
    case "$1" in
        --results) RESULTS="${2:?}"; shift 2 ;;
        --scenarios) SCENARIOS="${2:?}"; shift 2 ;;
        --text) TEXT="${2:?}"; shift 2 ;;
        --manifest-history) HISTORY="${2:?}"; shift 2 ;;
        --out) OUT="${2:?}"; shift 2 ;;
        --only) ONLY="${2:?}"; shift 2 ;;
        --list) LIST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1"; usage; exit 2 ;;
    esac
done
RESULTS="${RESULTS%/}"
case "$RESULTS" in
    *_sealed*) echo "this command never reads _sealed/"; exit 2 ;;
esac
[ -d "$RESULTS" ] || { echo "no folder of saved runs: $RESULTS"; exit 2; }
for step in ${ONLY//,/ }; do
    case " $ALL_STEPS " in *" $step "*) ;; *) echo "unknown step for --only: $step"; exit 2 ;; esac
done
want() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0 ;; esac; return 1; }

PARENT="$(dirname "$RESULTS")"           # results: the folder that holds results/main and results/stretch.md
MAIN_NAME="$(basename "$RESULTS")"       # main
ABS_RESULTS="$(cd "$RESULTS" && pwd)"
ABS_PARENT="$(cd "$PARENT" && pwd)"

if [ "$LIST" = "1" ]; then
    FRESH="<fresh>"
elif [ -z "$OUT" ]; then
    FRESH="$(mktemp -d "${TMPDIR:-/tmp}/rebuild_results.XXXXXX")" || exit 2
else
    mkdir -p "$OUT" || exit 2
    if [ -n "$(ls -A "$OUT")" ]; then echo "the fresh folder must be empty: $OUT"; exit 2; fi
    FRESH="$(cd "$OUT" && pwd)"
fi
# Rule P: the fresh folder relative to the top folder of the repository, if it lies inside it (else empty).
FRESH_REL=""
case "$FRESH" in "$ROOT"/*) FRESH_REL="${FRESH#"$ROOT"/}" ;; esac
F="$FRESH/$MAIN_NAME"                    # the fresh copy of results/main
LOGS="$FRESH/logs"
[ "$LIST" = "1" ] || mkdir -p "$F" "$LOGS"

N_EQUAL=0; N_RULE=0; N_DIFF=0; N_NEW=0; N_NOT_MADE=0; N_FAILED=0; N_MISSING=0; N_RAN=0

# ---- the comparison of one file (rules P, T and M above) ----
# Arguments: made, saved, fresh folder, parent of --results; optional: the fresh folder relative to the
# repository (rule P; empty if it lies outside), and "hide" (show no line of the content).
COMPARE_PY='
import difflib, json, re, sys
from pathlib import Path
made, saved, fresh, parent = sys.argv[1:5]
fresh_rel = sys.argv[5] if len(sys.argv) > 5 else ""
hide = len(sys.argv) > 6 and sys.argv[6] == "hide"
# Rule M: the file of the states of MANIFEST.json, and the SHA-256 of the present MANIFEST.json.
history_file = sys.argv[7] if len(sys.argv) > 7 else ""
present_manifest = sys.argv[8] if len(sys.argv) > 8 else ""
TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?")
m, s = Path(made), Path(saved)
if not m.exists():
    print(f"SAVED, NOT MADE  {saved}")
    sys.exit(4)
if not s.exists():
    print(f"NEW  {saved}: made ({m.stat().st_size} bytes), but there is no saved file to compare")
    sys.exit(3)
a, b = m.read_bytes(), s.read_bytes()
if a == b:
    print(f"equal  {saved}")
    sys.exit(0)
try:
    ta, tb = a.decode("utf-8"), b.decode("utf-8")
except UnicodeDecodeError:
    print(f"DIFFERENT  {saved}: the bytes differ ({len(a)} made, {len(b)} saved); not a text file, so no rule applies")
    sys.exit(1)

def rule_p(text):
    # The absolute path first: it ends with the relative one.
    text = text.replace(fresh, parent)
    return text.replace(fresh_rel, parent) if fresh_rel else text

def by_rules(made_text):
    """(made lines before rule P, after rule P, both after rule T, saved after rule T, equal?)"""
    raw, lp = made_text.splitlines(), rule_p(made_text).splitlines()
    na, nb = [TIME.sub("<time>", x) for x in lp], [TIME.sub("<time>", x) for x in tb.splitlines()]
    return raw, lp, na, nb, na == nb and made_text.endswith("\n") == tb.endswith("\n")

la_raw, la, na, nb, same = by_rules(ta)
lb = tb.splitlines()
if same:
    by_p = sum(1 for x, y in zip(la_raw, la) if x != y)
    by_t = sum(1 for x, y in zip(la, lb) if x != y)
    print(f"equal by rule  {saved}: {by_p} line(s) differ only in the path of the fresh folder (rule P), "
          f"{by_t} line(s) only in a time (rule T)")
    sys.exit(0)
# Rule M: a run manifest whose frozen.manifest_sha256 differs while both code hashes are equal.
if saved.endswith("run_manifest.json"):
    try:
        jm, js = json.loads(ta), json.loads(tb)
    except ValueError:
        jm = js = None
    if isinstance(jm, dict) and isinstance(js, dict):
        fm, fs = jm.get("frozen") or {}, js.get("frozen") or {}
        hm, hs = fm.get("manifest_sha256"), fs.get("manifest_sha256")
        code = fm.get("code_sha256")
        old, new = f"\"manifest_sha256\": \"{hm}\"", f"\"manifest_sha256\": \"{hs}\""
        states = {}
        if history_file and Path(history_file).is_file():
            import csv
            with open(history_file, newline="", encoding="utf-8") as handle:
                states = {row["sha256"].strip(): row["state"].strip() for row in csv.DictReader(handle)}
        why = []
        if hs not in states:
            why.append("the saved frozen.manifest_sha256 is not a row of " + (history_file or "the history file"))
        if not present_manifest or hm != present_manifest:
            why.append("the made frozen.manifest_sha256 is not the SHA-256 of the present MANIFEST.json")
        if (isinstance(hm, str) and isinstance(hs, str) and hm != hs and code
                and code == fs.get("code_sha256") and jm.get("code_sha256") == js.get("code_sha256")
                and ta.count(old) == 1):
            raw_m, lp_m, na_m, nb_m, same_m = by_rules(ta.replace(old, new))
            if same_m and not why:
                by_p = sum(1 for x, y in zip(raw_m, lp_m) if x != y)
                by_t = sum(1 for x, y in zip(lp_m, lb) if x != y)
                print(f"equal by rule M  {saved}: frozen.manifest_sha256 is {hs} ({states[hs]}) in the saved "
                      f"file and {hm} (the present MANIFEST.json) in the made one; code_sha256 and "
                      f"frozen.code_sha256 are {code} in both. The code hash is the same, so the code that made "
                      "the file did not change; only MANIFEST.json changed after the saved file was made. "
                      f"Other lines: {by_p} by rule P, {by_t} by rule T")
                sys.exit(0)
            if same_m:
                print(f"rule M does not apply to {saved}: " + "; ".join(why))
diff = [d for d in difflib.unified_diff(nb, na, "saved", "made", n=0, lineterm="")
        if not d.startswith(("---", "+++"))]
changed = sum(1 for d in diff if d[:1] in "+-")
if hide:
    print(f"DIFFERENT  {saved}: {changed} line(s) differ after rules P and T; no line is shown (the file may "
          "hold a number of the test set)")
    sys.exit(1)
print(f"DIFFERENT  {saved}: {changed} line(s) differ after rules P and T (- saved, + made; at most 16 shown)")
for d in diff[:16]:
    print("    " + (d if len(d) <= 220 else d[:217] + "..."))
sys.exit(1)
'

hidden() {            # hidden <saved file>: 0 if no line of its content may be shown (see DIFFERENT above)
    [ "$TEST_ANY" = "1" ] || return 1
    case "$1" in
        *_test*|*test_*|*/test/*|*/report/stretch/*|*/stretch.md|*/report/failures/*) return 0 ;;
    esac
    return 1
}

compare() {           # compare <made file> <saved file>
    if [ "$LIST" = "1" ]; then echo "  would compare  $2"; return; fi
    local code=0 shown_as=""
    hidden "$2" && shown_as="hide"
    "$PY" -c "$COMPARE_PY" "$1" "$2" "$FRESH" "$PARENT" "$FRESH_REL" "$shown_as" "$HISTORY" "$PRESENT_MANIFEST" || code=$?
    case "$code" in
        0) if [ -f "$1" ] && cmp -s "$1" "$2"; then N_EQUAL=$((N_EQUAL+1)); else N_RULE=$((N_RULE+1)); fi ;;
        1) N_DIFF=$((N_DIFF+1)) ;;
        3) N_NEW=$((N_NEW+1)) ;;
        4) N_NOT_MADE=$((N_NOT_MADE+1)) ;;
        *) echo "FAILED  the comparison of $2 (exit $code)"; N_FAILED=$((N_FAILED+1)) ;;
    esac
}

compare_dir() {       # compare_dir <made folder> <saved folder>: every file of either (not of subfolders)
    local names name
    names="$( { [ -d "$1" ] && ls -p "$1" | grep -v /; [ -d "$2" ] && ls -p "$2" | grep -v /; } | sort -u)"
    if [ -z "$names" ]; then echo "  (no file in $2)"; return; fi
    while IFS= read -r name; do compare "$1/$name" "$2/$name"; done <<< "$names"
}

run() {               # run <name> <command...>: the output goes into logs/<name>.log
    local name="$1"; shift
    if [ "$LIST" = "1" ]; then echo "would run  $name: $*"; return 0; fi
    local code=0
    "$@" > "$LOGS/$name.log" 2>&1 || code=$?
    if [ "$code" -ne 0 ]; then
        echo "FAILED  $name (exit $code; the output is in $LOGS/$name.log)"
        N_FAILED=$((N_FAILED+1)); return 1
    fi
    N_RAN=$((N_RAN+1)); return 0
}

run_to() {            # run_to <file> <name> <command...>: the command's output is the file
    local target="$1" name="$2"; shift 2
    if [ "$LIST" = "1" ]; then echo "would run  $name: $* > $target"; return 0; fi
    local code=0
    "$@" > "$target" 2> "$LOGS/$name.log" || code=$?
    if [ "$code" -ne 0 ]; then
        echo "FAILED  $name (exit $code; the output is in $target and $LOGS/$name.log)"
        N_FAILED=$((N_FAILED+1)); return 1
    fi
    N_RAN=$((N_RAN+1)); return 0
}

need() {              # need <file...>: 0 if every file exists; else says which one is missing
    local f ok=0
    for f in "$@"; do
        if [ ! -e "$f" ]; then echo "MISSING input  $f"; N_MISSING=$((N_MISSING+1)); ok=1; fi
    done
    return $ok
}

mk() { [ "$LIST" = "1" ] || mkdir -p "$@"; }
header() { echo; echo "== $1"; }

manifest_value() {    # manifest_value <run_manifest.json> <key>: one value, empty if absent
    "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"
}

TEST_ALL=1; TEST_ANY=0
for arm in "${ARMS[@]}"; do
    if [ -f "$RESULTS/${arm}_test/episodes.jsonl" ]; then TEST_ANY=1; else TEST_ALL=0; fi
done

echo "rebuild of $RESULTS (scenarios $SCENARIOS, text $TEXT), into the fresh folder $FRESH"
echo "started $(date -u +%FT%TZ); python $("$PY" -c 'import sys; print(sys.version.split()[0])')"
# Rule M: the SHA-256 of the present MANIFEST.json must be a row of the file of its states.
PRESENT_MANIFEST="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' MANIFEST.json 2>/dev/null)"
if [ -z "$PRESENT_MANIFEST" ]; then
    echo "MANIFEST HISTORY  there is no MANIFEST.json in $ROOT, so rule M applies to no file"
    N_MISSING=$((N_MISSING+1))
elif [ ! -f "$HISTORY" ]; then
    echo "MANIFEST HISTORY  there is no file $HISTORY, so rule M applies to no file"
    N_MISSING=$((N_MISSING+1))
elif ! cut -d, -f1 "$HISTORY" | grep -qx "$PRESENT_MANIFEST"; then
    echo "MANIFEST HISTORY  the present MANIFEST.json (SHA-256 $PRESENT_MANIFEST) is not a row of $HISTORY: add a row for it (the main session adds one after each amendment)"
    N_MISSING=$((N_MISSING+1))
else
    echo "rule M: the present MANIFEST.json (SHA-256 $PRESENT_MANIFEST) is a row of $HISTORY ($(($(wc -l < "$HISTORY") - 1)) state(s))"
fi
if [ -n "$FRESH_REL" ]; then
    echo "rule P: in a made file, $FRESH and $FRESH_REL (the same folder, relative to the repository) are read as $PARENT"
else
    echo "rule P: in a made file, $FRESH is read as $PARENT"
fi
if [ "$TEST_ALL" = "1" ]; then
    echo "the test runs exist: $RESULTS/{base,sft,grpo_seed1,grpo_seed2}_test; they are made again too"
elif [ "$TEST_ANY" = "1" ]; then
    echo "MISSING input  only some of the four test runs exist in $RESULTS; the test parts that need all four are not made"
    N_MISSING=$((N_MISSING+1))
else
    echo "the test runs do not exist: $RESULTS/base_test, sft_test, grpo_seed1_test and grpo_seed2_test are" \
         "not there, so no test table, test monitor report, test replay or test part of a report is made"
fi

# ---- tables ----
if want tables; then
    header "tables (python -m accessdesk.eval.tables)"
    mk "$F/tables"
    R="$RESULTS"
    if need "$R/base_dev/episodes.jsonl" "$R/sft_dev/episodes.jsonl"; then
        run table_base_vs_sft "$PY" -m accessdesk.eval.tables --arm "base=$R/base_dev/episodes.jsonl" \
            --arm "sft=$R/sft_dev/episodes.jsonl" --baseline base --pair base:sft \
            --out-json "$F/tables/base_vs_sft.json" --out-md "$F/tables/base_vs_sft.md"
        for g in grpo_seed1 grpo_seed2; do
            need "$R/${g}_dev/episodes.jsonl" || continue
            run "table_base_sft_$g" "$PY" -m accessdesk.eval.tables --arm "base=$R/base_dev/episodes.jsonl" \
                --arm "sft=$R/sft_dev/episodes.jsonl" --arm "$g=$R/${g}_dev/episodes.jsonl" \
                --baseline base --pair base:sft --pair "sft:$g" --pair "base:$g" \
                --out-json "$F/tables/base_sft_$g.json" --out-md "$F/tables/base_sft_$g.md"
        done
    fi
    if [ "$TEST_ALL" = "1" ]; then
        T="$F/tables/test_base_sft_grpo_seed1_seed2"
        run table_test "$PY" -m accessdesk.eval.tables --arm "base=$R/base_test/episodes.jsonl" \
            --arm "sft=$R/sft_test/episodes.jsonl" --arm "grpo_seed1=$R/grpo_seed1_test/episodes.jsonl" \
            --arm "grpo_seed2=$R/grpo_seed2_test/episodes.jsonl" --baseline base --pair base:sft \
            --pair sft:grpo_seed1 --pair sft:grpo_seed2 --out-json "$T.json" --out-md "$T.md"
    fi
    compare_dir "$F/tables" "$RESULTS/tables"
fi

# ---- monitors ----
if want monitors; then
    header "monitor reports (scripts/monitor_report.py)"
    mk "$F/monitors"
    for split in dev test; do
        for arm in "${ARMS[@]}"; do
            ep="$RESULTS/${arm}_$split/episodes.jsonl"
            if [ ! -f "$ep" ]; then
                [ "$split" = "dev" ] && need "$ep"
                continue
            fi
            run_to "$F/monitors/${arm}_$split.md" "monitors_${arm}_$split" "$PY" scripts/monitor_report.py \
                --episodes "$ep" --scenarios "$SCENARIOS" --split "$split" --text "$TEXT"
        done
    done
    compare_dir "$F/monitors" "$RESULTS/monitors"
fi

# ---- second desk ----
if want second_desk; then
    header "second desk (scripts/second_desk_report.py)"
    args=(); ok=1
    for arm in "${ARMS[@]}"; do
        d="$RESULTS/${arm}_shift_differ/episodes.jsonl"; c="$RESULTS/${arm}_shift_control/episodes.jsonl"
        need "$d" "$c" || ok=0
        args+=(--arm "$arm" "$d" "$c")
    done
    if [ "$ok" = "1" ]; then
        mk "$F/audit/second_desk"
        run second_desk "$PY" scripts/second_desk_report.py "${args[@]}" --scenarios "$SCENARIOS" \
            --text "$TEXT" --out "$F/audit/second_desk"
        compare_dir "$F/audit/second_desk" "$RESULTS/audit/second_desk"
    fi
fi

# ---- formula audit ----
if want formula; then
    header "formula audit (scripts/formula_audit.py)"
    mk "$F/audit/formula"
    for log in "$RESULTS"/*/train_log.jsonl "$RESULTS"/checks/*/train_log.jsonl; do
        [ -f "$log" ] || continue
        run_dir="$(dirname "$log")"; rel="${run_dir#"$RESULTS"/}"; name="${rel//\//_}"
        run "formula_$name" "$PY" scripts/formula_audit.py --run "$run_dir" --out "$F/audit/formula/$name.json"
    done
    compare_dir "$F/audit/formula" "$RESULTS/audit/formula"
fi

# ---- positive control ----
if want positive_control; then
    header "positive control (scripts/audit_positive_control.py)"
    if need "$RESULTS/grpo_seed1_dev/episodes.jsonl" "$RESULTS/grpo_seed2_dev/episodes.jsonl"; then
        mk "$F/audit/positive_control"
        run positive_control "$PY" scripts/audit_positive_control.py --episodes \
            "$RESULTS/grpo_seed1_dev/episodes.jsonl" "$RESULTS/grpo_seed2_dev/episodes.jsonl" \
            --scenarios "$SCENARIOS" --split dev --text "$TEXT" --out "$F/audit/positive_control"
        compare_dir "$F/audit/positive_control" "$RESULTS/audit/positive_control"
    fi
fi

# ---- replay checks ----
if want replay; then
    header "replay checks (scripts/replay_check.py)"
    mk "$F/audit/replay"
    for man in "$RESULTS"/*/run_manifest.json "$RESULTS"/checks/*/run_manifest.json; do
        run_dir="$(dirname "$man")"
        [ -f "$run_dir/episodes.jsonl" ] || continue
        rel="${run_dir#"$RESULTS"/}"; name="${rel//\//_}"
        split="$(manifest_value "$man" split)"; ruleset="$(manifest_value "$man" ruleset)"
        rs=(); [ -n "$ruleset" ] && rs=(--ruleset "$ruleset")
        made="$F/audit/replay/$name.txt"; saved="$RESULTS/audit/replay/$name.txt"
        run_to "$made" "replay_$name" "$PY" scripts/replay_check.py --episodes "$RESULTS/$rel/episodes.jsonl" \
            --scenarios "$SCENARIOS" --split "$split" --text "$TEXT" "${rs[@]}" || continue
        if [ -f "$saved" ]; then
            compare "$made" "$saved"
        elif [ "$LIST" = "1" ]; then
            echo "  would check  $rel (no saved output: only the result of the check is printed)"
        else
            echo "no difference  $rel (split $split, rule set ${ruleset:-of each record}): the replay check found" \
                 "no difference; no saved output to compare (made: $made)"
        fi
    done
    # The training rollouts: the runs of last night in their order, then any other run, sorted.
    runs=()
    for r in grpo_seed1 grpo_seed2 checks/pilot checks/overfit; do
        [ -d "$RESULTS/$r/rollouts" ] && runs+=("$r")
    done
    for d in "$RESULTS"/*/rollouts "$RESULTS"/checks/*/rollouts; do
        [ -d "$d" ] || continue
        r="$(dirname "$d")"; r="${r#"$RESULTS"/}"
        case " ${runs[*]} " in *" $r "*) ;; *) runs+=("$r") ;; esac
    done
    summary="$F/audit/replay/training_rollouts.txt"
    [ "$LIST" = "1" ] || : > "$summary"
    for r in "${runs[@]}"; do
        n=0; bad=0
        for ep in "$RESULTS/$r"/rollouts/step_*.episodes.jsonl; do
            [ -f "$ep" ] || continue
            n=$((n+1))
            [ "$LIST" = "1" ] && continue
            code=0
            "$PY" scripts/replay_check.py --episodes "$ep" --scenarios "$SCENARIOS" --split train \
                --text "$TEXT" >> "$LOGS/replay_training_${r//\//_}.log" 2>&1 || code=$?
            [ "$code" -ne 0 ] && bad=$((bad+1))
        done
        if [ "$LIST" = "1" ]; then
            echo "would run  replay of $n rollout files of $r (--split train)"
        else
            echo "$r: $n rollout files replayed, $bad with a difference or an error" >> "$summary"
            if [ "$bad" -ne 0 ]; then
                echo "FAILED  replay of the rollouts of $r: $bad of $n files with a difference or an error" \
                     "(see $LOGS/replay_training_${r//\//_}.log)"
                N_FAILED=$((N_FAILED+1))
            else
                N_RAN=$((N_RAN+1))
            fi
        fi
    done
    compare "$summary" "$RESULTS/audit/replay/training_rollouts.txt"
    # A saved replay output that no run made again.
    for saved in "$RESULTS"/audit/replay/*.txt; do
        [ -f "$saved" ] || continue
        [ -f "$F/audit/replay/$(basename "$saved")" ] || [ "$LIST" = "1" ] || compare "$F/audit/replay/$(basename "$saved")" "$saved"
    done
fi

# ---- reference rows ----
if want reference; then
    header "reference rows (scripts/run_scripted.py)"
    mk "$F/reference"
    for rec in "$RESULTS"/reference/*.jsonl; do
        [ -f "$rec" ] || continue
        base="$(basename "$rec")"
        man="$rec.run_manifest.json"
        need "$man" || continue
        # The call of the saved run manifest ("command"), with only the value of --out changed.
        cmd=()
        while IFS= read -r -d '' word; do cmd+=("$word"); done < <("$PY" -c '
import json, sys
command = json.load(open(sys.argv[1]))["command"]
if "--out" not in command or not command[0].endswith("run_scripted.py"):
    sys.exit(1)
command[command.index("--out") + 1] = sys.argv[2]
sys.stdout.write("".join(word + "\0" for word in command))' "$man" "$F/reference/$base")
        if [ "${#cmd[@]}" -eq 0 ]; then
            echo "FAILED  $man holds no call of scripts/run_scripted.py with --out"; N_FAILED=$((N_FAILED+1)); continue
        fi
        run "reference_${base%.jsonl}" "$PY" "${cmd[@]}" || continue
        compare "$F/reference/$base" "$rec"
        compare "$F/reference/$base.run_manifest.json" "$man"
    done
fi

# ---- the family profile after SFT ----
if want sft_profile; then
    header "family profile after SFT (scripts/sft_profile.py)"
    if need "$RESULTS/sft_profile/families.txt" "$RESULTS/sft_train/episodes.jsonl"; then
        mk "$F/sft_profile"
        run sft_profile_pick "$PY" scripts/sft_profile.py --scenarios "$SCENARIOS" --pick "$F/sft_profile/families.txt"
        run sft_profile "$PY" scripts/sft_profile.py --scenarios "$SCENARIOS" \
            --episodes "$RESULTS/sft_train/episodes.jsonl" --families "$RESULTS/sft_profile/families.txt" \
            --out "$F/sft_profile"
        compare_dir "$F/sft_profile" "$RESULTS/sft_profile"
    fi
fi

# ---- the blind audit sheet: hashes only ----
if want audit_sheet; then
    header "blind audit sheet (scripts/audit_sheet.py make; SHA-256 only)"
    if [ ! -f "$RESULTS/audit/audit_sheet.md" ] || [ ! -f "$RESULTS/audit/audit_key.json" ]; then
        echo "no saved audit sheet and key in $RESULTS/audit: nothing to compare"
    elif need "$RESULTS/grpo_seed1_dev/episodes.jsonl" "$RESULTS/grpo_seed2_dev/episodes.jsonl"; then
        SHEET="$FRESH/audit_sheet_made"
        mk "$SHEET"
        if run audit_sheet "$PY" scripts/audit_sheet.py make --scenarios "$SCENARIOS" --split dev \
                --arm grpo_seed1 "$RESULTS/grpo_seed1_dev/episodes.jsonl" \
                --arm grpo_seed2 "$RESULTS/grpo_seed2_dev/episodes.jsonl" --seed 0 --out "$SHEET" \
                && [ "$LIST" = "0" ]; then
            for name in audit_sheet.md audit_key.json; do
                code=0
                # Rule P: the absolute path of the made folder first, then its path relative to the
                # repository (when the fresh folder lies inside it).
                "$PY" -c '
import hashlib, sys
made, saved, fresh, saved_dir, fresh_rel = sys.argv[1:6]
a = open(made, "rb").read().decode("utf-8").replace(fresh, saved_dir)
if fresh_rel:
    a = a.replace(fresh_rel, saved_dir)
b = open(saved, "rb").read()
sys.exit(0 if hashlib.sha256(a.encode("utf-8")).digest() == hashlib.sha256(b).digest() else 1)' \
                    "$SHEET/$name" "$RESULTS/audit/$name" "$SHEET" "$RESULTS/audit" \
                    "${FRESH_REL:+$FRESH_REL/audit_sheet_made}" || code=$?
                if [ "$code" -eq 0 ]; then
                    echo "equal by rule  $RESULTS/audit/$name: the same SHA-256 after rule P; no content is shown"
                    N_RULE=$((N_RULE+1))
                else
                    echo "DIFFERENT  $RESULTS/audit/$name: not the same SHA-256 (after rule P); no content is shown"
                    N_DIFF=$((N_DIFF+1))
                fi
            done
        fi
        [ "$LIST" = "1" ] || rm -rf "$SHEET"
        [ "$LIST" = "1" ] || echo "  the made sheet, form and key were deleted"
    fi
fi

# ---- the four pieces of the report ----
if want figure; then
    header "report piece: figure (scripts/make_figure.py)"
    mk "$F/report/figure"
    run figure "$PY" scripts/make_figure.py --run "$ABS_RESULTS/grpo_seed1" --run "$ABS_RESULTS/grpo_seed2" \
        --scenarios "$SCENARIOS" --out "$F/report/figure"
    compare_dir "$F/report/figure" "$RESULTS/report/figure"
fi
if want training; then
    header "report piece: training (scripts/training_report.py)"
    mk "$F/report/training"
    run training "$PY" scripts/training_report.py --main "$ABS_RESULTS" --calibration "$ABS_PARENT/calibration" \
        --smoke "$ABS_PARENT/_smoke" --ledger "$ABS_PARENT/spend_ledger.csv" --scenarios "$SCENARIOS" \
        --out "$F/report/training"
    compare_dir "$F/report/training" "$RESULTS/report/training"
fi
if want failures; then
    header "report piece: failures (scripts/failure_report.py)"
    mk "$F/report/failures"
    mon="$F/monitors"; want monitors || mon="$ABS_RESULTS/monitors"
    run failures "$PY" scripts/failure_report.py --results "$ABS_RESULTS" --scenarios "$SCENARIOS" \
        --text "$TEXT" --monitors "$mon" --out "$F/report/failures"
    compare_dir "$F/report/failures" "$RESULTS/report/failures"
    if [ "$TEST_ALL" = "1" ]; then
        mk "$F/report/failures/test"
        run failures_test "$PY" scripts/failure_report.py --results "$ABS_RESULTS" --split test \
            --scenarios "$SCENARIOS" --text "$TEXT" --monitors "$mon" --out "$F/report/failures/test"
        compare_dir "$F/report/failures/test" "$RESULTS/report/failures/test"
    fi
fi
if want stretch; then
    header "report piece: stretch (scripts/stretch_numbers.py)"
    mk "$F/report/stretch"
    run stretch "$PY" scripts/stretch_numbers.py --main "$ABS_RESULTS" --ledger "$ABS_PARENT/spend_ledger.csv" \
        --scenarios "$SCENARIOS" --out "$F/report/stretch" --doc "$FRESH/stretch.md"
    compare_dir "$F/report/stretch" "$RESULTS/report/stretch"
    compare "$FRESH/stretch.md" "$PARENT/stretch.md"
fi

# ---- what this command does not make ----
header "not made again by this command"
echo "  the attempts, logs and adapters of the runs (episodes.jsonl, train_log.jsonl, rollouts, checkpoints, sft/):"
echo "    they need the GPU and the model; every number above is made FROM them"
echo "  sft_data (needs the tokenizer of the model), checks/layout_gradients (needs the GPU),"
echo "  the calibration report of results/calibration, the labels form of the audit (a person fills it)"
if [ "$LIST" = "0" ]; then
    other=()
    for entry in "$RESULTS"/*; do
        n="$(basename "$entry")"
        case "$n" in
            audit|tables|monitors|reference|report|sft_profile|checks|sft|sft_data) continue ;;
        esac
        [ -f "$entry/episodes.jsonl" ] || [ -f "$entry/train_log.jsonl" ] && continue
        other+=("$n")
    done
    [ "${#other[@]}" -gt 0 ] && echo "  other entries of $RESULTS that no step reads: ${other[*]}"
fi

echo
if [ "$LIST" = "1" ]; then
    echo "list only: nothing was run"
    exit 0
fi
echo "summary: $N_EQUAL equal, $N_RULE equal by rule, $N_DIFF DIFFERENT, $N_NEW new (no saved file)," \
     "$N_NOT_MADE saved but not made; $N_RAN commands ran, $N_FAILED failed; $N_MISSING missing input(s)"
echo "ended $(date -u +%FT%TZ); the made files are in $FRESH"
if [ "$N_DIFF" -gt 0 ] || [ "$N_NOT_MADE" -gt 0 ] || [ "$N_FAILED" -gt 0 ] || [ "$N_MISSING" -gt 0 ]; then
    exit 1
fi
exit 0
