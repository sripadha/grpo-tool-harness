#!/usr/bin/env bash
# Runs ON the rented machine. Destroys this machine after a time limit, so that a forgotten or hung machine
# cannot keep costing money. Needs only bash, curl and the standard tools (date, sleep, tr, grep, cut, head).
#
#   watchdog.sh <minutes>
#
# Start it detached, for example:
#   nohup bash watchdog.sh 240 >> /workspace/watchdog.out 2>&1 < /dev/null &
#
# What it does:
#   1. Checks its inputs NOW and exits with code 2 on any problem, so a wrong value shows up at the start
#      and not hours later.
#   2. Stops an earlier watchdog if one runs: a second call replaces the first timer.
#   3. Writes an "ARMED" line to /workspace/watchdog.log and waits.
#   4. Writes a "FIRING" line and asks Vast.ai to destroy this machine. If that fails it asks to stop the
#      machine. It repeats both every 60 seconds for as long as the machine is alive.
#
# The API calls are the ones the vastai command line tool 1.11.0 makes (vastai/api/instances.py):
#   show_instance:    GET    {server}/api/v0/instances/{id}/?owner=me
#   destroy_instance: DELETE {server}/api/v0/instances/{id}/    JSON body {}
#   stop_instance:    PUT    {server}/api/v0/instances/{id}/    JSON body {"state": "stopped"}
# each with the header "Authorization: Bearer <key>" (vastai/api/client.py, _build_headers).
# The key is CONTAINER_API_KEY. Vast.ai puts it into every container, and it is valid for this machine only.
# This script never prints the key and does not use "set -x".
#
# Settings for tests only: WATCHDOG_DIR (folder of the log), WATCHDOG_API_URL, WATCHDOG_MINUTE_SECONDS
# (length of a "minute"), WATCHDOG_RETRY_SECONDS, WATCHDOG_SKIP_PREFLIGHT=1 (do not ask the API at the start).
set -u

MINUTES="${1:-}"
DIR="${WATCHDOG_DIR:-/workspace}"
API="${WATCHDOG_API_URL:-https://console.vast.ai/api/v0}"
MINUTE_SECONDS="${WATCHDOG_MINUTE_SECONDS:-60}"
RETRY_SECONDS="${WATCHDOG_RETRY_SECONDS:-60}"
LOG="$DIR/watchdog.log"
PIDFILE="$DIR/watchdog.pid"
RESPONSE="$DIR/watchdog.response"
HTTP_CODE="000"

mkdir -p "$DIR" 2>/dev/null
if ! { : >> "$LOG"; } 2>/dev/null; then
    echo "watchdog: FATAL: cannot write to $LOG" >&2
    exit 2
fi

log() {  # one line with the UTC time, to the log file and to the screen
    printf '%s watchdog[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$*" | tee -a "$LOG" >&2
}

die() {
    log "FATAL: $*"
    exit 2
}

read_var() {  # read_var NAME: from the environment, else from the first process, else from /etc/environment
    local name="$1" value="${!1:-}"
    if [ -z "$value" ] && [ -r /proc/1/environ ]; then
        value="$(tr '\0' '\n' < /proc/1/environ | grep -m1 "^${name}=" | cut -d= -f2-)"
    fi
    if [ -z "$value" ] && [ -r /etc/environment ]; then
        value="$(grep -m1 "^${name}=" /etc/environment | cut -d= -f2- | tr -d "\"'")"
    fi
    printf '%s' "$value"
}

api_call() {  # api_call METHOD URL [JSON_BODY]: sets HTTP_CODE ("000" = no answer); the answer goes to $RESPONSE
    local method="$1" url="$2" body="${3:-}"
    local args=(-sS -m 30 -o "$RESPONSE" -w '%{http_code}' -X "$method" -H "Authorization: Bearer $KEY")
    if [ -n "$body" ]; then
        args+=(-H 'Content-Type: application/json' -d "$body")
    fi
    : > "$RESPONSE"
    HTTP_CODE="$(curl "${args[@]}" "$url" 2>>"$LOG")"
    [[ "$HTTP_CODE" =~ ^[0-9]{3}$ ]] || HTTP_CODE="000"
}

answer_text() {  # the start of the last answer on one line, with the key removed in case the server echoes it
    local text
    text="$(head -c 300 "$RESPONSE" 2>/dev/null | tr '\n\r' '  ')"
    printf '%s' "${text//"$KEY"/***}"
}

accepted() {  # true if the last call got HTTP 200 and the answer says "success": true
    [ "$HTTP_CODE" = "200" ] && grep -q '"success"[[:space:]]*:[[:space:]]*true' "$RESPONSE"
}

# ----------------------------------------------------------------------------------------------------------
# 1. Inputs, checked before anything else
# ----------------------------------------------------------------------------------------------------------
[[ "$MINUTES" =~ ^[1-9][0-9]{0,4}$ ]] || die "usage: watchdog.sh <minutes>   (a whole number, 1 to 99999; got '$MINUTES')"
[[ "$MINUTE_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "WATCHDOG_MINUTE_SECONDS must be a whole number, got '$MINUTE_SECONDS'"
[[ "$RETRY_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "WATCHDOG_RETRY_SECONDS must be a whole number, got '$RETRY_SECONDS'"
command -v curl >/dev/null 2>&1 || die "curl is not installed"

ID="$(read_var CONTAINER_ID)"
if [ -z "$ID" ]; then  # Vast.ai also sets VAST_CONTAINERLABEL, of the form C.<instance id>
    LABEL="$(read_var VAST_CONTAINERLABEL)"
    ID="${LABEL#C.}"
fi
[[ "$ID" =~ ^[1-9][0-9]*$ ]] || die "CONTAINER_ID is missing or is not a number (got '$ID'): is this a Vast.ai container?"

KEY="$(read_var CONTAINER_API_KEY)"
[ -n "$KEY" ] || die "CONTAINER_API_KEY is missing: this machine could not destroy itself"
[[ "$KEY" =~ ^[^[:space:]]{16,}$ ]] || die "CONTAINER_API_KEY does not look like a key (its length is ${#KEY})"

URL="$API/instances/$ID/"

# Ask the API one harmless question, to learn now whether the key and the id are accepted.
if [ "${WATCHDOG_SKIP_PREFLIGHT:-0}" = "1" ]; then
    log "preflight skipped on request (WATCHDOG_SKIP_PREFLIGHT=1)"
else
    for TRY in 1 2 3; do
        api_call GET "$URL?owner=me"
        [ "$HTTP_CODE" = "000" ] || break
        sleep 5
    done
    case "$HTTP_CODE" in
        200) log "preflight OK: Vast.ai accepts the key for instance $ID" ;;
        401|403|404) die "preflight: Vast.ai answered HTTP $HTTP_CODE for instance $ID: wrong key or wrong id ($(answer_text))" ;;
        *) log "WARNING: preflight gave no clear answer (HTTP $HTTP_CODE); the timer starts anyway" ;;
    esac
fi

# ----------------------------------------------------------------------------------------------------------
# 2. Replace an earlier timer (only now, after this call has passed its own checks)
# ----------------------------------------------------------------------------------------------------------
if [ -r "$PIDFILE" ]; then
    OLD="$(head -c 20 "$PIDFILE" 2>/dev/null | tr -d '[:space:]')"
    if [[ "$OLD" =~ ^[0-9]+$ ]] && [ "$OLD" != "$$" ] && [ -r "/proc/$OLD/cmdline" ] \
        && tr '\0' ' ' < "/proc/$OLD/cmdline" | grep -q 'watchdog'; then
        if kill "$OLD" 2>/dev/null; then
            log "stopped the earlier timer (pid $OLD)"
        fi
    fi
fi
echo "$$" > "$PIDFILE" || die "cannot write $PIDFILE"

# ----------------------------------------------------------------------------------------------------------
# 3. Wait
# ----------------------------------------------------------------------------------------------------------
LIMIT=$((MINUTES * MINUTE_SECONDS))
FIRE_AT="$(date -u -d "+${LIMIT} seconds" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
log "ARMED: instance $ID will be destroyed in $MINUTES minutes, at about $FIRE_AT"

SECONDS=0
NEXT_NOTE=1800
while [ "$SECONDS" -lt "$LIMIT" ]; do
    LEFT=$((LIMIT - SECONDS))
    if [ "$LEFT" -gt 20 ]; then sleep 20; else sleep "$LEFT"; fi
    if [ "$SECONDS" -ge "$NEXT_NOTE" ] && [ "$SECONDS" -lt "$LIMIT" ]; then
        log "waiting: about $(((LIMIT - SECONDS) / MINUTE_SECONDS)) minutes left"
        NEXT_NOTE=$((NEXT_NOTE + 1800))
    fi
done

# ----------------------------------------------------------------------------------------------------------
# 4. Fire: destroy, else stop, and again until this machine is gone
# ----------------------------------------------------------------------------------------------------------
log "FIRING: the limit of $MINUTES minutes is reached; destroying instance $ID"
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    api_call DELETE "$URL" '{}'
    if accepted; then
        log "attempt $ATTEMPT: destroy accepted (HTTP 200)"
    else
        log "attempt $ATTEMPT: destroy FAILED (HTTP $HTTP_CODE: $(answer_text)); trying to stop instead"
        api_call PUT "$URL" '{"state": "stopped"}'
        if accepted; then
            log "attempt $ATTEMPT: stop accepted (HTTP 200); a stopped machine still pays for its disk"
        else
            log "attempt $ATTEMPT: stop FAILED (HTTP $HTTP_CODE: $(answer_text))"
        fi
    fi
    sleep "$RETRY_SECONDS"
    log "attempt $ATTEMPT: this machine is still running after $RETRY_SECONDS seconds; trying again"
done
