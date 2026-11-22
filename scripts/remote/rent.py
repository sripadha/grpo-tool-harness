#!/usr/bin/env python3
"""Rent, use and destroy one Vast.ai machine from the local computer.

Standard library only. Calls that need the account go through the `vastai` command line tool (version 1.11.0
was read for this script). The search for offers uses the public endpoint and never touches the key.

    rent.py [--env-file F] [--ledger F] [--ssh-key F] <sub-command> ...

Sub-commands (every one that changes something has --dry-run and prints what it will do first):
    offers                       list the cheapest machines that pass the filter (no key needed)
    balance                      the account's credit
    rent OFFER --max-minutes N   create the machine, start its watchdog, wait for SSH, write the ledger line
    status                       the machines of the account and the ledger total
    sync-up ID                   copy the repository to /workspace/repo on the machine
    run ID COMMAND               run a command in /workspace/repo on the machine
    fetch ID                     copy /workspace/repo/results/ back into results/
    attach-key ID                add the local public key to a machine that already exists
    destroy ID                   destroy the machine and write its real cost to the ledger
    destroy-all                  destroy every machine of this project

Rules that the code keeps:
  * The API key is read in one place (`load_api_key`) and lives in a `Secret`, which prints as "***".
    It is handed to the vastai tool through the environment variable VAST_API_KEY (the tool reads it itself),
    never on a command line, and it is never sent to the rented machine.
  * Everything the vastai tool prints passes through `redact` before it is shown.
  * Only an offer that `offers` listed in the last 15 minutes can be rented (it passed the filter, and its
    price is known), and `rent` refuses if the ledger total plus the projected cost of the new machine is
    above --budget. After the machine exists its real price is read back and checked again.
  * The machine destroys itself after --max-minutes (scripts/remote/watchdog.sh, started at boot). If the
    watchdog cannot be started, `rent` destroys the machine again.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT.parent / "vast.env"        # outside the repository: it can never be committed
LEDGER_NAME = "spend_ledger.csv"
DEFAULT_LEDGER = REPO_ROOT / "results" / LEDGER_NAME
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "vast_takehome_ed25519"
WATCHDOG_SCRIPT = REPO_ROOT / "scripts" / "remote" / "watchdog.sh"
DEFAULT_OFFERS_CACHE = REPO_ROOT / ".cache" / "vast_offers.json"   # .cache/ is in .gitignore
OFFERS_MAX_AGE_MINUTES = 15
# What is kept of each offer. All of it is public market data.
OFFER_FIELDS = ("id", "machine_id", "host_id", "gpu_name", "num_gpus", "gpu_ram", "dph_total", "storage_cost",
                "inet_down_cost", "inet_up_cost", "cuda_max_good", "driver_version", "cpu_arch", "cpu_ram",
                "cpu_cores_effective", "inet_down", "reliability2", "duration", "verification", "rentable",
                "rented", "geolocation")

KEY_NAME = "VAST_API_KEY"
# Public. `vastai search offers` posts its query to this address (vastai/cli/commands/offers.py).
OFFERS_URL = "https://console.vast.ai/api/v0/bundles/"

# Container image. Tag and digest were read from the Docker Hub tags API on 2026-11-17 (pushed 2026-10-24,
# 8.7 GB). Our Python environment is built with uv inside the container, so only the image's system matters.
# The machine is created by TAG. The digest is what that tag pointed to on 2026-11-17; it is written into
# every ledger line, so a later change of the tag can be seen. Creating by digest (IMAGE_BY_DIGEST, with
# --image) was tried once on 2026-11-17: Vast.ai accepted it, but the machine did not answer on SSH within
# 20 minutes and was destroyed (see the ledger). The cause is not known: it may be the digest form or that
# one host. The Vast.ai documentation describes image names with tags only.
IMAGE_TAG = "vastai/pytorch:cuda-13.0.3-auto"
IMAGE_DIGEST = "sha256:e052540e87b18c5b9f8048826ccd3c9c9f0bf6ae6f9245bd6ba684fbd362ef27"
IMAGE_BY_DIGEST = "vastai/pytorch@" + IMAGE_DIGEST

REMOTE_REPO = "/workspace/repo"
LABEL_PREFIX = "accessdesk"
ONSTART_LIMIT = 4000        # the Vast.ai API documents a limit of 4048 characters for the on-start field
MAX_MINUTES_LIMIT = 24 * 60
# From these states a machine never reaches "running" (the vastai tool's own notes, vastai/SKILL.md).
DEAD_STATES = ("exited", "unknown", "offline")

LEDGER_FIELDS = ("time_utc", "event", "instance_id", "offer_id", "gpu", "dph_usd", "max_minutes",
                 "bandwidth_est_usd", "projected_usd", "actual_usd", "note")

# Never copied to the machine. A leading "/" means "at the top of the repository only".
SYNC_EXCLUDES = ("/.venv/", "/.cache/", "/_dev/", "/.git/", "/results/", "__pycache__/", "*.pyc",
                 ".pytest_cache/", "*.egg-info/", "*.env", ".env*", "*.pem", "*_ed25519*")

# No host key checking: every rental is a new machine at a reused address, and nothing secret is sent to it.
SSH_OPTIONS = ("-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
               "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6")

# Shell text that runs on the machine. `$(command -v setsid)` is empty if setsid is missing, which is fine.
START_WATCHDOG = ("$(command -v setsid) nohup bash /root/watchdog.sh {minutes} "
                  ">> /workspace/watchdog.out 2>&1 < /dev/null &")
WATCHDOG_PROBE = ("if [ -r /workspace/watchdog.pid ] && "
                  'kill -0 "$(cat /workspace/watchdog.pid)" 2>/dev/null; then echo WATCHDOG_ALIVE; fi; '
                  "tail -n 6 /workspace/watchdog.log 2>/dev/null")


class RentError(Exception):
    """A problem that stops the command. The message is shown to the user as it is."""


class BudgetError(RentError):
    """The new machine would take the projected total above the budget."""


# ---------------------------------------------------------------------------------------------------------
# The key and the redaction
# ---------------------------------------------------------------------------------------------------------
class Secret:
    """Holds the API key. str() and repr() give "***", so the key cannot reach a message by accident."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def __repr__(self) -> str:
        return "***"

    __str__ = __repr__

    def reveal(self) -> str:
        """The key itself. Called only by `child_env` and `redact`."""
        return self._value


# Shapes in which a key can show up in text, blanked even when the key's value is not known here.
# The last one: on a login error the vastai tool prints a key's last four characters ("ends in ...a3f9").
_KEY_SHAPES = (
    (re.compile(r"(Bearer\s+)[^\s\"']+", re.IGNORECASE), r"\1***"),
    (re.compile(r"(--api-key[=\s]+)[^\s\"']+"), r"\1***"),
    (re.compile(r"((?:VAST_API_KEY|CONTAINER_API_KEY|api_key)[\"']?\s*[=:]\s*[\"']?)[^\s\"',}]+"), r"\1***"),
    (re.compile(r"(ends in \.\.\.)\w{1,4}"), r"\1****"),
)


def redact(text: str, secret: Secret | None = None) -> str:
    """`text` with the key replaced by "***", and with anything that has the shape of a key blanked."""
    if secret is not None and secret.reveal():
        text = text.replace(secret.reveal(), "***")
    for pattern, replacement in _KEY_SHAPES:
        text = pattern.sub(replacement, text)
    return text


def parse_env_text(text: str) -> dict[str, str]:
    """Parse lines of the form NAME=value. Blank lines and lines that start with # are skipped.

    Allowed: a leading "export ", spaces around the name and the value, a value in single or double quotes.
    An unquoted value ends at " #" (a comment). Error messages name the line number and never show the line.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, equals, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not equals or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise RentError(f"line {number} of the key file is not of the form NAME=value")
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end == -1:
                raise RentError(f"line {number} of the key file has a quote that is not closed")
            value = value[1:end]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


def load_api_key(path: Path) -> Secret:
    """The only place where the API key is read. Raises RentError with a clear text if it cannot be found."""
    if not path.is_file():
        raise RentError(
            f"Key file not found: {path}\n"
            f"Create it with one line: the name {KEY_NAME}, an equals sign, and your key from\n"
            "https://cloud.vast.ai/manage-keys/ . The file stays outside the repository, and this script\n"
            "never prints the key.")
    values = parse_env_text(path.read_text(encoding="utf-8"))
    if KEY_NAME not in values:
        raise RentError(f"{path} has no line {KEY_NAME}=...")
    value = values[KEY_NAME]
    if not value:
        raise RentError(f"{KEY_NAME} is empty in {path}")
    if any(ch.isspace() for ch in value):
        raise RentError(f"{KEY_NAME} in {path} contains white space")
    return Secret(value)


def child_env(secret: Secret) -> dict[str, str]:
    """The environment of a vastai process: ours, plus the key, minus the tool's update check."""
    env = dict(os.environ)
    env[KEY_NAME] = secret.reveal()          # vastai/cli/main.py reads it as the default of --api-key
    env["VASTAI_NO_UPDATE_CHECK"] = "1"      # vastai/cli/selfupdate.py: no extra request, no extra output
    return env


# ---------------------------------------------------------------------------------------------------------
# Running programs
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str     # already redacted
    stderr: str     # already redacted


def vastai_bin() -> str:
    path = os.environ.get("VASTAI_BIN") or shutil.which("vastai")
    if not path:
        raise RentError("the `vastai` command was not found; install it with: uv tool install vastai==1.11.0")
    return path


def shown_command(args: Sequence[str]) -> str:
    """The vastai call as it is printed: the key is never part of it."""
    return f"{KEY_NAME}=*** vastai {shlex.join(args)}"


def run_vastai(args: Sequence[str], secret: Secret, *, timeout: float = 300.0) -> Completed:
    """Run `vastai <args>` with the key in its environment. The returned text is already redacted."""
    try:
        done = subprocess.run([vastai_bin(), *args], env=child_env(secret), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        raise RentError(f"no answer within {timeout:.0f} s from: {shown_command(args)}") from None
    return Completed(done.returncode, redact(done.stdout, secret), redact(done.stderr, secret))


def run_local(argv: Sequence[str], *, capture: bool = False, timeout: float | None = None,
              stdin_bytes: bytes | None = None) -> Completed:
    """Run ssh or rsync. With `capture` (or with input) the output is returned, else it is shown line by
    line as it comes. Always redacted."""
    if capture or stdin_bytes is not None:
        feed = {"input": stdin_bytes} if stdin_bytes is not None else {"stdin": subprocess.DEVNULL}
        try:
            done = subprocess.run(list(argv), capture_output=True, timeout=timeout, check=False, **feed)
        except subprocess.TimeoutExpired:
            return Completed(124, "", f"no answer within {timeout:.0f} s")
        return Completed(done.returncode, redact(done.stdout.decode("utf-8", "replace")),
                         redact(done.stderr.decode("utf-8", "replace")))
    process = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace")
    assert process.stdout is not None
    for line in process.stdout:
        print(redact(line.rstrip("\n")), flush=True)
    return Completed(process.wait(), "", "")


def parse_json(text: str) -> object:
    """The JSON value in the tool's output (other lines may come before it). Raises ValueError if none."""
    text = text.strip()
    for start in sorted({i for i in (text.find("{"), text.find("[")) if i != -1}):
        try:
            return json.loads(text[start:])
        except ValueError:
            continue
    raise ValueError("no JSON in the output")


def api_error(done: Completed) -> tuple[int, str] | None:
    """(status code, message) if vastai reported an API error, else None.

    vastai 1.11.0 exits with code 0 even when the API answers with an error (vastai/cli/main.py, run_command).
    With --raw it prints {"error": true, "status_code": .., "msg": ..} on stderr; without --raw it prints the
    line "Failed with error <code>: <text>". So the exit code alone proves nothing.
    """
    for line in done.stderr.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                blob = json.loads(line)
            except ValueError:
                continue
            if isinstance(blob, dict) and blob.get("error"):
                return int(blob.get("status_code") or 0), str(blob.get("msg"))
        match = re.match(r"Failed with error (\d+): (.*)", line)
        if match:
            return int(match.group(1)), match.group(2)
    return None


def vast_json(args: Sequence[str], secret: Secret) -> object:
    """Run a vastai call that ends in --raw and return its JSON. Raises RentError on any error answer."""
    done = run_vastai(args, secret)
    error = api_error(done)
    if error is not None:
        raise RentError(f"Vast.ai answered with error {error[0]}: {error[1]}\n  call: {shown_command(args)}")
    try:
        return parse_json(done.stdout)
    except ValueError:
        raise RentError(f"unexpected output from: {shown_command(args)}\n  exit code {done.returncode}\n"
                        f"  stdout: {done.stdout.strip()[:500]}\n"
                        f"  stderr: {done.stderr.strip()[:500]}") from None


# ---------------------------------------------------------------------------------------------------------
# The vastai calls (flags as in vastai 1.11.0: vastai/cli/commands/instances.py, keys.py, billing.py)
# ---------------------------------------------------------------------------------------------------------
def call_show_instance(instance_id: int) -> list[str]:
    return ["show", "instance", str(instance_id), "--raw"]


def call_show_instances() -> list[str]:
    return ["show", "instances", "--raw"]


def call_show_user() -> list[str]:
    return ["show", "user", "--raw"]


def call_show_ssh_keys() -> list[str]:
    return ["show", "ssh-keys", "--raw"]


def call_create_ssh_key(public_key: str) -> list[str]:
    """The key TEXT, not a file name: `create ssh-key` in 1.11.0 sends its argument to the server as it is."""
    return ["create", "ssh-key", public_key]


def call_attach_ssh(instance_id: int, public_key: str) -> list[str]:
    return ["attach", "ssh", str(instance_id), public_key]


def call_create_instance(offer_id: int, *, image: str, disk_gb: int, label: str, onstart: str) -> list[str]:
    """--cancel-unavail: fail, instead of creating a stopped machine that would pay for its disk.
    --retry 1: a second try after a lost answer could create a machine that this script does not know."""
    return ["create", "instance", str(offer_id), "--image", image, "--disk", str(disk_gb), "--ssh",
            "--direct", "--label", label, "--onstart-cmd", onstart, "--cancel-unavail", "--retry", "1",
            "--raw"]


def call_destroy_instance(instance_id: int) -> list[str]:
    """Without -y the tool waits for a typed confirmation. Without --raw it prints a line on success."""
    return ["destroy", "instance", str(instance_id), "-y"]


# ---------------------------------------------------------------------------------------------------------
# Offers (public endpoint, no key)
# ---------------------------------------------------------------------------------------------------------
def offer_query(*, gpu: str = "RTX 4090", min_cuda: float = 13.2, min_reliability: float = 0.99,
                min_inet_down: int = 500, min_ram_gb: int = 48, disk_gb: int = 60, limit: int = 64) -> dict:
    """The body that `vastai search offers` would post for our filter, sorted by price.

    Equal to what the tool's own parser (vastai/api/query.py, parse_query) builds from
        'gpu_name=RTX_4090 num_gpus=1 verified=true reliability>0.99 cuda_max_good>=13.2 inet_down>=500
         cpu_ram>=48 direct_port_count>=1 rentable=true cpu_arch=amd64'      -o dph_total --storage 60
    The tool sends numbers as text, and host memory in MB (GB times 1000). `allocated_storage` makes the
    price per hour include the disk.
    """
    return {
        "verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True},
        "gpu_name": {"eq": gpu.replace("_", " ")}, "num_gpus": {"eq": "1"},
        "reliability": {"gt": str(min_reliability)}, "cuda_max_good": {"gte": str(min_cuda)},
        "inet_down": {"gte": str(min_inet_down)}, "cpu_ram": {"gte": float(min_ram_gb) * 1000},
        "direct_port_count": {"gte": "1"}, "cpu_arch": {"eq": "amd64"},
        "order": [["dph_total", "asc"]], "type": "on-demand", "limit": int(limit),
        "allocated_storage": float(disk_gb),
    }


def query_from_args(args: argparse.Namespace) -> dict:
    return offer_query(gpu=args.gpu, min_cuda=args.min_cuda, min_reliability=args.min_reliability,
                       min_inet_down=args.min_inet_down, min_ram_gb=args.min_ram_gb, disk_gb=args.disk)


def post_public(url: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST JSON without any credentials. This function has no access to the key."""
    headers = {"Content-Type": "application/json", "User-Agent": "accessdesk-rent"}
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise RentError(f"the public offers endpoint did not answer: {error}") from None


def fetch_offers(query: dict) -> list[dict]:
    """The offers that pass `query`, cheapest first (the endpoint returns at most `limit` of them)."""
    return list(post_public(OFFERS_URL, query).get("offers") or [])


def save_offers(path: Path, offers: Sequence[dict], disk_gb: int, when: datetime) -> None:
    """Keep the offers that `offers` has just listed, so that `rent` knows the price of the one chosen.

    Why a file: the endpoint does not answer a search by offer id, and two equal searches a few seconds
    apart return different sets of ids (seen on 2026-11-17), so `rent` cannot look an offer up again.
    """
    slim = [{name: offer.get(name) for name in OFFER_FIELDS} for offer in offers]
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"time_utc": stamp(when), "disk_gb": disk_gb, "offers": slim}
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")


def load_cached_offer(path: Path, offer_id: int, disk_gb: int, when: datetime) -> dict:
    """The offer with this id from the last `offers` run. Raises RentError if that run is missing, older
    than OFFERS_MAX_AGE_MINUTES, was priced for another disk size, or did not list the id."""
    again = "Run `rent.py offers` and choose an offer from its list."
    if not path.is_file():
        raise RentError(f"no list of offers found. {again}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    listed = datetime.strptime(blob["time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age = (when - listed).total_seconds() / 60.0
    if not 0 <= age <= OFFERS_MAX_AGE_MINUTES:
        raise RentError(f"the list of offers is {age:.0f} minutes old; the limit is "
                        f"{OFFERS_MAX_AGE_MINUTES}. {again}")
    if blob["disk_gb"] != disk_gb:
        raise RentError(f"the offers were priced with {blob['disk_gb']} GB of disk, not {disk_gb} GB. "
                        + again)
    for offer in blob["offers"]:
        if offer.get("id") == offer_id:
            return offer
    raise RentError(f"offer {offer_id} is not in the list of offers. {again}")


def bandwidth_estimate(offer: dict, download_gb: float) -> float:
    """USD for downloading `download_gb` to the machine (inet_down_cost is USD per GB)."""
    return float(offer.get("inet_down_cost") or 0.0) * download_gb


def projected_cost(offer: dict, max_minutes: int, download_gb: float) -> float:
    """USD if the machine runs for the whole limit: price per hour (with disk) times hours, plus download."""
    return float(offer["dph_total"]) * max_minutes / 60.0 + bandwidth_estimate(offer, download_gb)


def check_offer(offer: dict, *, gpu: str, min_cuda: float) -> list[str]:
    """Reasons why this offer must not be rented for this project (an empty list: fine).

    The filter was applied by the server when the offers were listed; this is a second look at the record
    itself before money is spent.
    """
    problems = []
    if offer.get("gpu_name") != gpu.replace("_", " "):
        problems.append(f"GPU is {offer.get('gpu_name')!r}, not {gpu!r}")
    if offer.get("num_gpus") != 1:
        problems.append(f"{offer.get('num_gpus')} GPUs, not 1")
    if float(offer.get("cuda_max_good") or 0) < min_cuda:
        problems.append(f"the driver supports CUDA {offer.get('cuda_max_good')}, below {min_cuda}")
    if offer.get("cpu_arch") not in (None, "amd64"):
        problems.append(f"CPU architecture {offer.get('cpu_arch')!r}, not amd64")
    if offer.get("verification") != "verified":
        problems.append(f"host is {offer.get('verification')!r}, not verified")
    if not offer.get("rentable") or offer.get("rented"):
        problems.append("not rentable at the moment")
    return problems


def offers_table(offers: Sequence[dict], *, hours: float, download_gb: float) -> str:
    """A plain table. "$/h" includes the disk. "$ total" = $/h times `hours` plus the download estimate."""
    lines = [f"{'offer id':>10} {'$/h':>6} {'$ total':>8} {'disk $/GB/mo':>12} {'down $/GB':>9} "
             f"{'up $/GB':>8} {'CUDA':>5} {'VRAM GB':>7} {'RAM GB':>6} {'vCPU':>5} {'Mbps':>6} {'rel.':>6} "
             f"{'days':>6}  place"]
    for o in offers:
        total = o["dph_total"] * hours + bandwidth_estimate(o, download_gb)
        reliability = o.get("reliability2") or o.get("reliability") or 0
        lines.append(
            f"{o['id']:>10} {o['dph_total']:>6.3f} {total:>8.2f} {o.get('storage_cost') or 0:>12.2f} "
            f"{o.get('inet_down_cost') or 0:>9.4f} {o.get('inet_up_cost') or 0:>8.4f} "
            f"{o.get('cuda_max_good') or 0:>5} {(o.get('gpu_ram') or 0) / 1024:>7.0f} "
            f"{(o.get('cpu_ram') or 0) / 1000:>6.0f} {o.get('cpu_cores_effective') or 0:>5.0f} "
            f"{o.get('inet_down') or 0:>6.0f} {reliability:>6.4f} {(o.get('duration') or 0) / 86400:>6.1f}"
            f"  {o.get('geolocation') or '?'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def money(value: float) -> str:
    return f"{value:.4f}"


def ledger_line(row: dict) -> str:
    """One CSV line (no line break) with the fields of LEDGER_FIELDS in order; a missing field is empty."""
    unknown = set(row) - set(LEDGER_FIELDS)
    if unknown:
        raise ValueError(f"unknown ledger fields: {sorted(unknown)}")
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow([redact(str(row.get(name, ""))) for name in LEDGER_FIELDS])
    return buffer.getvalue()


def append_ledger(path: Path, row: dict) -> None:
    """Append one line; the file is created with its header line if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        if new_file:
            handle.write(",".join(LEDGER_FIELDS) + "\n")
        handle.write(ledger_line(row) + "\n")


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ledger_total(rows: Sequence[dict]) -> float:
    """USD counted against the budget.

    For each machine: its real cost if it was destroyed (the "destroy" line), else its projected cost (the
    "rent" line), which is the most it can cost because the watchdog ends it at the time limit.
    """
    cost: dict[str, float] = {}
    for row in rows:
        machine = row["instance_id"]
        if row["event"] == "rent" and row.get("projected_usd"):
            cost.setdefault(machine, float(row["projected_usd"]))
        elif row["event"] == "destroy" and row.get("actual_usd"):
            cost[machine] = float(row["actual_usd"])
    return sum(cost[machine] for machine in sorted(cost))


def check_budget(rows: Sequence[dict], new_cost: float, budget: float) -> float:
    """Return the new projected total, or raise BudgetError if it is above the budget."""
    total = ledger_total(rows) + new_cost
    if total > budget + 1e-9:
        raise BudgetError(
            f"REFUSED: the ledger stands at {ledger_total(rows):.2f} USD and this machine is projected at "
            f"{new_cost:.2f} USD. The sum, {total:.2f} USD, is above the budget of {budget:.2f} USD.")
    return total


def rent_row(rows: Sequence[dict], instance_id: int) -> dict | None:
    """The ledger's "rent" line of this machine, if there is one."""
    for row in rows:
        if row["event"] == "rent" and row["instance_id"] == str(instance_id):
            return row
    return None


# ---------------------------------------------------------------------------------------------------------
# The on-start command, SSH
# ---------------------------------------------------------------------------------------------------------
def strip_comments(script: str) -> str:
    """The shell script without blank lines and without lines that hold only a comment (line 1 is kept)."""
    lines = script.split("\n")
    kept = lines[:1] + [line for line in lines[1:] if line.strip() and not line.strip().startswith("#")]
    return "\n".join(kept) + "\n"


def watchdog_start_command(minutes: int, skip_preflight: bool) -> str:
    """The shell command that starts /root/watchdog.sh in the background with the time limit."""
    prefix = "WATCHDOG_SKIP_PREFLIGHT=1 " if skip_preflight else ""
    return prefix + START_WATCHDOG.format(minutes=minutes)


def build_onstart(watchdog_script: str, minutes: int, *, skip_preflight: bool = False) -> str:
    """The command the machine runs at boot: write the watchdog script and start it with the time limit.

    The script travels as gzip + base64 inside the command (the Vast.ai documentation's own advice for long
    on-start scripts). The machine needs base64, gunzip and bash. Raises RentError above ONSTART_LIMIT.
    """
    if not 1 <= minutes <= MAX_MINUTES_LIMIT:
        raise RentError(f"--max-minutes must be from 1 to {MAX_MINUTES_LIMIT}, not {minutes}")
    packed = gzip.compress(strip_comments(watchdog_script).encode("utf-8"), mtime=0)
    blob = base64.b64encode(packed).decode("ascii")
    command = ("mkdir -p /workspace; touch /root/.no_auto_tmux; "
               f"echo {blob} | base64 -d | gunzip > /root/watchdog.sh; "
               + watchdog_start_command(minutes, skip_preflight))
    if len(command) > ONSTART_LIMIT:
        raise RentError(f"the on-start command has {len(command)} characters; the limit is {ONSTART_LIMIT}. "
                        "Make scripts/remote/watchdog.sh shorter.")
    return command


def read_public_key(private_key: Path) -> str:
    """The text "<type> <key>" from the .pub file of `private_key`. The comment (often user@host) is
    dropped, so it is not sent to the account."""
    public = Path(str(private_key) + ".pub")
    if not public.is_file():
        raise RentError(f"SSH public key not found: {public}\nCreate the pair with:\n"
                        f"    ssh-keygen -t ed25519 -N '' -C vast-takehome -f {private_key}")
    text = public.read_text(encoding="utf-8").strip()
    parts = text.split()
    if "PRIVATE KEY" in text or len(parts) < 2 or not parts[0].startswith(("ssh-", "ecdsa-", "sk-")):
        raise RentError(f"{public} does not look like an SSH public key")
    return f"{parts[0]} {parts[1]}"


@dataclass(frozen=True)
class Conn:
    host: str
    port: int


def ssh_target(instance: dict) -> Conn | None:
    """Where SSH listens, worked out as in vastai/cli/commands/misc.py (_ssh_url): the machine's own address
    and mapped port if port 22 is mapped (direct), else the Vast.ai proxy. None if it is not known yet."""
    try:
        mapped = (instance.get("ports") or {}).get("22/tcp")
        if mapped:
            return Conn(str(instance["public_ipaddr"]).strip(), int(mapped[0]["HostPort"]))
        return Conn(str(instance["ssh_host"]).strip(), int(instance["ssh_port"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def ssh_argv(conn: Conn, private_key: Path, remote_command: str) -> list[str]:
    return ["ssh", "-p", str(conn.port), "-i", str(private_key), *SSH_OPTIONS, f"root@{conn.host}",
            remote_command]


def rsync_argv(conn: Conn, private_key: Path, source: str, target: str, *, excludes: Sequence[str] = (),
               delete: bool = False) -> list[str]:
    """rsync over SSH. With `delete`, files missing in the source are deleted in the target; rsync never
    deletes files that match an exclude pattern."""
    transport = shlex.join(["ssh", "-p", str(conn.port), "-i", str(private_key), *SSH_OPTIONS])
    argv = ["rsync", "-az", "--stats", "-e", transport]
    if delete:
        argv.append("--delete")
    argv += [f"--exclude={pattern}" for pattern in excludes]
    return argv + [source, target]


def remote_run_command(command: str) -> str:
    """`command`, run in the repository on the machine with the environment that setup_box.sh wrote."""
    return f"cd {REMOTE_REPO} && if [ -r /workspace/env.sh ]; then . /workspace/env.sh; fi && {command}"


def remote_detached_command(command: str, log_name: str) -> str:
    """The same, started in the background so that it survives the end of the SSH session."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", log_name):
        raise RentError("--log must be a plain file name (letters, digits, dot, dash, underscore)")
    log = f"/workspace/logs/{log_name}.log"
    inner = shlex.quote(remote_run_command(command))
    # ";" and not "&&" before the job: with "&&" the whole list would go to the background and keep the SSH
    # session open until the job ends.
    return (f"mkdir -p /workspace/logs || exit 1; $(command -v setsid) nohup bash -c {inner} "
            f"> {log} 2>&1 < /dev/null & echo started, log: {log}")


# ---------------------------------------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------------------------------------
def get_instance(instance_id: int, secret: Secret) -> dict | None:
    """The machine's record, or None if Vast.ai does not know it (any more)."""
    done = run_vastai(call_show_instance(instance_id), secret)
    error = api_error(done)
    if error is not None:
        if error[0] == 404 and "session" not in error[1].lower():   # 404 "Session expired" is a login problem
            return None
        raise RentError(f"Vast.ai answered with error {error[0]}: {error[1]}")
    try:
        blob = parse_json(done.stdout)
    except ValueError:
        blob = None
    if blob == {"instances": None}:      # what `show instance --raw` prints for an id that does not exist
        return None
    if isinstance(blob, dict) and "id" in blob:
        return blob
    raise RentError("unexpected output of `show instance`: "
                    f"{done.stdout.strip()[:300]} {done.stderr.strip()[:300]}")


def list_instances(secret: Secret) -> list[dict]:
    blob = vast_json(call_show_instances(), secret)
    if not isinstance(blob, list):
        raise RentError("unexpected output of `show instances`: not a list")
    return blob


def is_ours(instance: dict, rows: Sequence[dict]) -> bool:
    """True if this project created the machine: it has our label, or its id is in the ledger."""
    label = str(instance.get("label") or "")
    return label.startswith(LABEL_PREFIX) or rent_row(rows, int(instance["id"])) is not None


def age_hours(instance: dict) -> float:
    """Hours since the machine's start time (`start_date`, seconds since 1970)."""
    return max(0.0, time.time() - float(instance.get("start_date") or time.time())) / 3600.0


def connection(instance_id: int, secret: Secret) -> Conn:
    instance = get_instance(instance_id, secret)
    if instance is None:
        raise RentError(f"machine {instance_id} does not exist (any more)")
    conn = ssh_target(instance)
    if conn is None:
        raise RentError(f"machine {instance_id} has no SSH address yet "
                        f"(status {instance.get('actual_status')!r})")
    return conn


def wait_for_ssh(instance_id: int, secret: Secret, private_key: Path, wait_minutes: float) -> Conn:
    """Poll until the machine runs and answers on SSH. Raises RentError on a dead state or after the wait."""
    deadline = time.monotonic() + wait_minutes * 60
    last = ""
    while time.monotonic() < deadline:
        try:
            instance = get_instance(instance_id, secret)
        except RentError as problem:        # one failed question must not end the wait
            print(f"  {stamp(now_utc())} the status could not be read ({problem}); asking again", flush=True)
            time.sleep(10)
            continue
        if instance is None:
            raise RentError(f"machine {instance_id} disappeared while it was starting")
        status = str(instance.get("actual_status"))
        note = f"status {status}: {str(instance.get('status_msg') or '').strip()[:120]}"
        if note != last:
            print(f"  {stamp(now_utc())} {note}", flush=True)
            last = note
        if status in DEAD_STATES:
            raise RentError(f"machine {instance_id} is in state {status!r} and will not start")
        conn = ssh_target(instance) if status == "running" else None
        if conn is not None:
            if run_local(ssh_argv(conn, private_key, "true"), capture=True, timeout=40).returncode == 0:
                return conn
        time.sleep(10)
    raise RentError(
        f"machine {instance_id} did not answer on SSH within {wait_minutes:.0f} minutes "
        f"(last {last or 'status unknown'}).\n"
        "If it never left the loading state, this host may have to download the whole image: try another\n"
        "offer, or a longer --wait-minutes. An image given by digest (name@sha256:...) did not start in the\n"
        f"one try of 2026-11-17; the default, {IMAGE_TAG}, is given by tag.")


def watchdog_armed(probe_output: str) -> bool:
    """True if the probe shows a live watchdog process and an ARMED line in its log."""
    return "WATCHDOG_ALIVE" in probe_output and " ARMED: " in probe_output


def ensure_watchdog(conn: Conn, private_key: Path, minutes: int, script: str, skip_preflight: bool) -> bool:
    """Wait for the watchdog that the on-start command started; if there is none, start one over SSH."""
    probe = Completed(1, "", "")
    for attempt in ("boot", "ssh"):
        if attempt == "ssh":
            print("  no watchdog from the on-start command; starting one over SSH", flush=True)
            run_local(ssh_argv(conn, private_key, "mkdir -p /workspace && cat > /root/watchdog.sh"),
                      stdin_bytes=script.encode("utf-8"), timeout=60)
            run_local(ssh_argv(conn, private_key, watchdog_start_command(minutes, skip_preflight)),
                      capture=True, timeout=60)
        for _ in range(15):
            probe = run_local(ssh_argv(conn, private_key, WATCHDOG_PROBE), capture=True, timeout=40)
            if watchdog_armed(probe.stdout):
                print("  watchdog: " + probe.stdout.strip().splitlines()[-1], flush=True)
                return True
            time.sleep(10)
        print("  watchdog log so far:\n" + (probe.stdout.strip() or "(empty)"), flush=True)
    return False


def destroy_machine(instance_id: int, secret: Secret, ledger: Path, *, note: str = "") -> None:
    """Destroy the machine, check that it is gone, and append the "destroy" line with the real cost.

    Real cost = hours since the machine's start time x its price per hour, plus the download estimate from
    its "rent" line. If the machine's record cannot be read, the cost is the projected upper limit of the
    "rent" line. Either way it is this script's estimate; the Vast.ai invoice is the authority.
    """
    rented = rent_row(read_ledger(ledger), instance_id)
    carried = float(rented["bandwidth_est_usd"] or 0) if rented else 0.0
    row = {"time_utc": stamp(now_utc()), "event": "destroy", "instance_id": instance_id,
           "offer_id": rented["offer_id"] if rented else "", "gpu": rented["gpu"] if rented else "",
           "bandwidth_est_usd": money(carried), "actual_usd": rented["projected_usd"] if rented else ""}
    try:
        instance = get_instance(instance_id, secret)
    except RentError as problem:            # a destroy must be tried even if the machine cannot be read
        print(f"machine {instance_id}: its record could not be read ({problem}); destroying it anyway")
        cost_note = "record not readable; cost = the projected upper limit."
    else:
        if instance is None:
            print(f"machine {instance_id} does not exist (any more); nothing to destroy")
            row["note"] = ("already gone (the watchdog?); cost = the projected upper limit. " + note).strip()
            append_ledger(ledger, row)
            return
        hours, price = age_hours(instance), float(instance.get("dph_total") or 0.0)
        print(f"machine {instance_id}: {instance.get('gpu_name')}, {price:.3f} USD/h, {hours:.2f} h old")
        row.update({"gpu": instance.get("gpu_name") or row["gpu"], "dph_usd": money(price),
                    "actual_usd": money(hours * price + carried)})
        cost_note = f"{hours:.3f} h x price + download estimate; an estimate, the invoice is the authority."
    call = call_destroy_instance(instance_id)
    print(f"Destroying machine {instance_id}. All data on it is lost.\n  running: {shown_command(call)}")
    done = run_vastai(call, secret)
    if done.stdout.strip() or done.stderr.strip():
        print("  " + (done.stdout.strip() + " " + done.stderr.strip()).strip())
    accepted = f"destroying instance {instance_id}." in done.stdout     # printed only if the API said success
    gone = False
    for _ in range(6):
        try:
            gone = get_instance(instance_id, secret) is None
        except RentError:
            gone = False
        if gone:
            break
        time.sleep(5)
    state = "" if gone else ("destroy accepted, machine still listed after 30 s. " if accepted
                             else "DESTROY NOT CONFIRMED. ")
    row["note"] = f"{state}{cost_note} {note}".strip()
    append_ledger(ledger, row)
    if not gone and not accepted:
        raise RentError(f"machine {instance_id} was NOT destroyed. Check with `rent.py status` and destroy "
                        "it in the web console (https://cloud.vast.ai/instances/) if needed.")
    result = "gone." if gone else "destroy accepted, but still listed; run `rent.py status` in a minute."
    print(f"machine {instance_id}: {result} Cost by this script's estimate: {row['actual_usd'] or '?'} USD")


# ---------------------------------------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------------------------------------
def cmd_offers(args: argparse.Namespace) -> int:
    min_seconds = args.min_days * 86400       # `duration` is the longest rental the host allows, in seconds
    offers = [o for o in fetch_offers(query_from_args(args)) if (o.get("duration") or 0) >= min_seconds]
    print(f"{len(offers)} offers pass the filter: one {args.gpu}, verified host, reliability > "
          f"{args.min_reliability}, driver for CUDA >= {args.min_cuda}, download >= {args.min_inet_down} "
          f"Mbps, host memory >= {args.min_ram_gb} GB, a direct port, rentable for >= {args.min_days:g} "
          "days.")
    print(f'"$/h" includes {args.disk} GB of disk. "$ total" = $/h x {args.hours:g} h + '
          f"{args.download_gb:g} GB of download at the host's price (which differs a lot between hosts).\n")
    print(offers_table(offers[:args.top], hours=args.hours, download_gb=args.download_gb))
    save_offers(args.offers_cache, offers[:args.top], args.disk, now_utc())
    print(f"\nTo rent one of these within {OFFERS_MAX_AGE_MINUTES} minutes: "
          "rent.py rent <offer id> --max-minutes <N> --dry-run")
    return 0


def cmd_balance(args: argparse.Namespace) -> int:
    user = vast_json(call_show_user(), load_api_key(args.env_file))
    if not isinstance(user, dict):
        raise RentError("unexpected output of `show user`")
    # Only these two fields of the account record are shown.
    print(f"credit:  {user.get('credit')}\nbalance: {user.get('balance')}")
    print(f"ledger total (projected): {ledger_total(read_ledger(args.ledger)):.2f} USD")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rows = read_ledger(args.ledger)
    instances = list_instances(load_api_key(args.env_file))
    print(f"{len(instances)} machine(s) on the account")
    for inst in instances:
        conn = ssh_target(inst)
        where = f"-p {conn.port} root@{conn.host}" if conn else "not yet"
        print(f"  {inst.get('id')}  {inst.get('actual_status')}  {inst.get('gpu_name')}  "
              f"{float(inst.get('dph_total') or 0):.3f} USD/h  {age_hours(inst):.2f} h old  "
              f"label={inst.get('label')!r}  ours={'yes' if is_ours(inst, rows) else 'NO'}  ssh={where}")
    print(f"ledger total (projected): {ledger_total(rows):.2f} USD   ({args.ledger})")
    return 0


def cmd_rent(args: argparse.Namespace) -> int:
    offer = load_cached_offer(args.offers_cache, args.offer_id, args.disk, now_utc())
    problems = check_offer(offer, gpu=args.gpu, min_cuda=args.min_cuda)
    if problems:
        raise RentError(f"offer {args.offer_id} does not fit this project: " + "; ".join(problems))
    cost = projected_cost(offer, args.max_minutes, args.download_gb)
    download = bandwidth_estimate(offer, args.download_gb)
    rows = read_ledger(args.ledger)
    total = check_budget(rows, cost, args.budget)

    secret = load_api_key(args.env_file)
    script = WATCHDOG_SCRIPT.read_text(encoding="utf-8")
    onstart = build_onstart(script, args.max_minutes, skip_preflight=args.skip_watchdog_preflight)
    label = f"{LABEL_PREFIX}-{now_utc().strftime('%Y%m%dT%H%M%SZ')}"
    image_note = f" ({IMAGE_DIGEST} on 2026-11-17)" if args.image == IMAGE_TAG else ""
    create = call_create_instance(args.offer_id, image=args.image, disk_gb=args.disk, label=label,
                                  onstart=onstart)
    row = {"time_utc": stamp(now_utc()), "event": "rent", "offer_id": args.offer_id,
           "gpu": offer.get("gpu_name"), "dph_usd": money(float(offer["dph_total"])),
           "max_minutes": args.max_minutes, "bandwidth_est_usd": money(download),
           "projected_usd": money(cost), "note": f"label {label}; image {args.image}{image_note}"}

    print(f"Plan: rent offer {args.offer_id}: {offer.get('gpu_name')} in {offer.get('geolocation')}, "
          f"{float(offer['dph_total']):.3f} USD/h with {args.disk} GB of disk.")
    print(f"  The machine destroys itself after {args.max_minutes} minutes. Projected cost {cost:.2f} USD "
          f"(of which download {download:.2f}); projected ledger total {total:.2f} of {args.budget:.2f} USD.")
    print(f"  1. {shown_command(call_show_ssh_keys())}")
    print("  2. if the public key is not on the account: "
          + shown_command(call_create_ssh_key("<public key>")))
    print(f"  3. {shown_command(create)}")
    print(f"  4. append to {args.ledger}:\n     {ledger_line({**row, 'instance_id': '<new id>'})}")
    print("  5. wait for SSH, check that the watchdog is armed, print the SSH command")
    if args.dry_run:
        print("DRY RUN: nothing was created and nothing was written.")
        return 0

    public_key = read_public_key(args.ssh_key)
    if public_key.split()[1] not in json.dumps(vast_json(call_show_ssh_keys(), secret)):
        print("Adding the public key to the account.")
        added = run_vastai(call_create_ssh_key(public_key), secret)
        if api_error(added) is not None:
            raise RentError(f"could not add the SSH key: {api_error(added)}")

    done = run_vastai(create, secret)
    try:
        answer = parse_json(done.stdout)
    except ValueError:
        answer = None
    if api_error(done) is not None or not isinstance(answer, dict) or not answer.get("new_contract"):
        raise RentError(
            f"the machine was NOT confirmed as created.\n  stdout: {done.stdout.strip()[:400]}\n"
            f"  stderr: {done.stderr.strip()[:400]}\n"
            f"Run `rent.py status` now: if a machine with the label {label} is listed, destroy it.\n"
            f"If the image name was refused, give another one with --image (the default is {IMAGE_TAG}).")
    instance_id = int(answer["new_contract"])
    try:        # the price that is really charged; if it cannot be read, the listed price stays in the ledger
        real_price = float((get_instance(instance_id, secret) or {}).get("dph_total") or 0.0)
    except RentError:
        real_price = 0.0
    if real_price > 0:
        cost = real_price * args.max_minutes / 60.0 + download
        row.update({"dph_usd": money(real_price), "projected_usd": money(cost)})
    append_ledger(args.ledger, {**row, "instance_id": instance_id})
    print(f"Created machine {instance_id} at {float(row['dph_usd']):.3f} USD/h. The ledger line is written.")

    try:
        check_budget(rows, cost, args.budget)       # again, with the real price
        conn = wait_for_ssh(instance_id, secret, args.ssh_key, args.wait_minutes)
        if not ensure_watchdog(conn, args.ssh_key, args.max_minutes, script, args.skip_watchdog_preflight):
            raise RentError("the watchdog could not be started on the machine (its log is printed above; "
                            "if the reason is its preflight, rent again with --skip-watchdog-preflight)")
    except (RentError, KeyboardInterrupt) as problem:
        print(f"PROBLEM: {str(problem) or 'interrupted'}")
        if args.keep_on_failure:
            print(f"Machine {instance_id} is kept (--keep-on-failure). It may have NO watchdog.")
        else:
            destroy_machine(instance_id, secret, args.ledger, note="destroyed because start-up failed")
        return 1
    print(f"\nMachine {instance_id} is ready. Connect with:\n"
          f"    ssh -p {conn.port} -i {args.ssh_key} root@{conn.host}\n"
          f"Next: rent.py sync-up {instance_id}\n"
          f"      rent.py run {instance_id} 'bash scripts/remote/setup_box.sh'")
    return 0


def cmd_sync_up(args: argparse.Namespace) -> int:
    source, target = f"{REPO_ROOT}/", f"{REMOTE_REPO}/"
    print(f"Copy {source} to machine {args.instance_id}:{target} (without {', '.join(SYNC_EXCLUDES)}). "
          "Files that were deleted here are deleted there, except under the excluded names.")
    if args.dry_run:
        argv = rsync_argv(Conn("<host>", 0), args.ssh_key, source, f"root@<host>:{target}",
                          excludes=SYNC_EXCLUDES, delete=True)
        print("DRY RUN, would run: " + shlex.join(argv))
        return 0
    conn = connection(args.instance_id, load_api_key(args.env_file))
    prepare = (f"mkdir -p {REMOTE_REPO} && (command -v rsync >/dev/null || "
               "(echo 'installing rsync on the machine' && apt-get update -qq && "
               "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync))")
    if run_local(ssh_argv(conn, args.ssh_key, prepare), timeout=600).returncode != 0:
        raise RentError("the machine has no rsync and it could not be installed")
    argv = rsync_argv(conn, args.ssh_key, source, f"root@{conn.host}:{target}", excludes=SYNC_EXCLUDES,
                      delete=True)
    return run_local(argv).returncode


def cmd_run(args: argparse.Namespace) -> int:
    remote = (remote_detached_command(args.remote_command, args.log) if args.detach
              else remote_run_command(args.remote_command))
    print(f"Run on machine {args.instance_id}: {remote}")
    if args.dry_run:
        print("DRY RUN: nothing was run.")
        return 0
    conn = connection(args.instance_id, load_api_key(args.env_file))
    return run_local(ssh_argv(conn, args.ssh_key, remote)).returncode


def cmd_fetch(args: argparse.Namespace) -> int:
    target = REPO_ROOT / "results"
    source = f"{REMOTE_REPO}/results/"
    print(f"Copy machine {args.instance_id}:{source} into {target}/ "
          "(nothing is deleted here; the ledger is never overwritten).")
    keep = sorted({LEDGER_NAME, args.ledger.name})      # a file of this name on the machine is not copied
    if args.dry_run:
        argv = rsync_argv(Conn("<host>", 0), args.ssh_key, f"root@<host>:{source}", f"{target}/",
                          excludes=keep)
        print("DRY RUN, would run: " + shlex.join(argv))
        return 0
    conn = connection(args.instance_id, load_api_key(args.env_file))
    target.mkdir(parents=True, exist_ok=True)
    argv = rsync_argv(conn, args.ssh_key, f"root@{conn.host}:{source}", f"{target}/", excludes=keep)
    return run_local(argv).returncode


def cmd_attach_key(args: argparse.Namespace) -> int:
    shown = shown_command(call_attach_ssh(args.instance_id, "<public key>"))
    print(f"Add the public key of {args.ssh_key} to machine {args.instance_id}:\n  {shown}")
    if args.dry_run:
        print("DRY RUN: nothing was changed.")
        return 0
    secret = load_api_key(args.env_file)
    done = run_vastai(call_attach_ssh(args.instance_id, read_public_key(args.ssh_key)), secret)
    if api_error(done) is not None:
        raise RentError(f"could not attach the key: {api_error(done)}")
    print("  done")
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    secret = load_api_key(args.env_file)
    if args.dry_run:
        print("DRY RUN: nothing is changed. A real run would do this, in order:\n"
              f"  1. {shown_command(call_show_instance(args.instance_id))}    (start time and price)\n"
              f"  2. {shown_command(call_destroy_instance(args.instance_id))}\n"
              f"  3. {shown_command(call_show_instance(args.instance_id))}    (check that it is gone)\n"
              f'  4. append a "destroy" line with the real cost to {args.ledger}')
        return 0
    destroy_machine(args.instance_id, secret, args.ledger)
    return 0


def cmd_destroy_all(args: argparse.Namespace) -> int:
    secret = load_api_key(args.env_file)
    rows = read_ledger(args.ledger)
    instances = list_instances(secret)       # read-only; also done in a dry run, to show the exact calls
    targets = [inst for inst in instances if args.everything or is_ours(inst, rows)]
    print(f"{len(instances)} machine(s) on the account; {len(targets)} to destroy.")
    for inst in instances:
        if inst not in targets:
            print("  left alone (not created by this project; --everything would include it): "
                  f"{inst.get('id')} label={inst.get('label')!r}")
    for inst in targets:
        print(f"  {shown_command(call_destroy_instance(int(inst['id'])))}    "
              f"({inst.get('gpu_name')}, {inst.get('actual_status')}, label={inst.get('label')!r})")
    if args.dry_run:
        print("DRY RUN: nothing was destroyed.")
        return 0
    failed = 0
    for inst in targets:
        try:
            destroy_machine(int(inst["id"]), secret, args.ledger, note="destroy-all")
        except RentError as problem:
            failed += 1
            print(f"PROBLEM: {problem}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    env_default = Path(os.environ.get("ACCESSDESK_VAST_ENV", DEFAULT_ENV_FILE))
    parser.add_argument("--env-file", type=Path, default=env_default,
                        help=f"file with the line {KEY_NAME}=... (default: vast.env next to the repository)")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="the spend ledger (CSV)")
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY,
                        help="private key; its .pub file is registered with the account")
    parser.add_argument("--offers-cache", type=Path, default=DEFAULT_OFFERS_CACHE,
                        help="where `offers` keeps its last list for `rent`")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Callable[[argparse.Namespace], int], text: str, *, changes: bool,
            machine: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=text)
        p.set_defaults(func=func)
        if machine:
            p.add_argument("instance_id", type=int, help="the machine's id (see `status`)")
        if changes:
            p.add_argument("--dry-run", action="store_true", help="print what would be done; change nothing")
        return p

    def add_machine_choice(p: argparse.ArgumentParser) -> None:
        p.add_argument("--gpu", default="RTX 4090")
        p.add_argument("--min-cuda", type=float, default=13.2,
                       help="lowest CUDA version the host driver must support (torch 2.15.0 ships CUDA 13.2)")
        p.add_argument("--disk", type=int, default=60, help="disk in GB; it cannot be changed later")
        p.add_argument("--download-gb", type=float, default=20.0,
                       help="GB the machine will download (models and packages; an estimate), for the cost")

    p = add("offers", cmd_offers, "list the cheapest machines that pass the filter (no key)", changes=False)
    add_machine_choice(p)
    p.add_argument("--min-reliability", type=float, default=0.99)
    p.add_argument("--min-inet-down", type=int, default=500, help="Mbps")
    p.add_argument("--min-ram-gb", type=int, default=48)
    p.add_argument("--min-days", type=float, default=1.0, help="the host must allow this rental time")
    p.add_argument("--hours", type=float, default=8.0, help='hours for the "$ total" column')
    p.add_argument("--top", type=int, default=10)

    add("balance", cmd_balance, "the account's credit", changes=False)
    add("status", cmd_status, "machines of the account and the ledger total", changes=False)

    p = add("rent", cmd_rent, "create a machine from an offer that `offers` listed", changes=True)
    p.add_argument("offer_id", type=int)
    add_machine_choice(p)
    p.add_argument("--max-minutes", type=int, required=True, help="the machine destroys itself after this")
    p.add_argument("--budget", type=float, default=40.0, help="USD; refuse if the projected total is higher")
    p.add_argument("--image", default=IMAGE_TAG, help="container image, as <repository>:<tag>")
    p.add_argument("--wait-minutes", type=float, default=20.0, help="how long to wait for SSH")
    p.add_argument("--keep-on-failure", action="store_true", help="keep the machine if its start-up fails")
    p.add_argument("--skip-watchdog-preflight", action="store_true",
                   help="start the watchdog without its first question to the API")

    add("sync-up", cmd_sync_up, "copy the repository to the machine", changes=True, machine=True)
    p = add("run", cmd_run, "run a command in the repository on the machine", changes=True, machine=True)
    p.add_argument("remote_command", help="one shell command, in quotes")
    p.add_argument("--detach", action="store_true", help="start it in the background and return")
    p.add_argument("--log", default="job", help="with --detach: name of the log under /workspace/logs/")
    add("fetch", cmd_fetch, "copy the machine's results/ back", changes=True, machine=True)
    add("attach-key", cmd_attach_key, "add the local public key to a machine", changes=True, machine=True)
    add("destroy", cmd_destroy, "destroy a machine and write its real cost", changes=True, machine=True)
    p = add("destroy-all", cmd_destroy_all, "destroy every machine of this project", changes=True)
    p.add_argument("--everything", action="store_true", help="also machines that this project did not create")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BudgetError as problem:
        print(redact(str(problem)))
        return 2
    except RentError as problem:
        print("ERROR: " + redact(str(problem)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
