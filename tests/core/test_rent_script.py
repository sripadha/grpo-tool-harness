"""Tests of scripts/remote/rent.py and of the start-up checks of scripts/remote/watchdog.sh.

Nothing here touches the network or the real key file: every test gets its own key file, ledger and offer
list in a temporary folder, and the two functions that could reach Vast.ai are replaced by ones that fail.
The key used here is made up.
"""
from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RENT_PATH = ROOT / "scripts" / "remote" / "rent.py"
WATCHDOG_PATH = ROOT / "scripts" / "remote" / "watchdog.sh"
SETUP_PATH = ROOT / "scripts" / "remote" / "setup_box.sh"

FAKE_KEY = "fk" + "0123456789abcdef" * 4          # 66 characters, made up
NOW = datetime(2026, 11, 17, 8, 0, 0, tzinfo=timezone.utc)

OFFER = {"id": 5001, "machine_id": 77, "host_id": 9, "gpu_name": "RTX 4090", "num_gpus": 1, "gpu_ram": 24564,
         "dph_total": 0.5, "storage_cost": 0.2, "inet_down_cost": 0.01, "inet_up_cost": 0.02,
         "cuda_max_good": 13.2, "driver_version": "595.71.05", "cpu_arch": "amd64", "cpu_ram": 64000,
         "cpu_cores_effective": 16.0, "inet_down": 800.0, "reliability2": 0.995, "duration": 864000.0,
         "verification": "verified", "rentable": True, "rented": False, "geolocation": "Somewhere, XX"}


def _load_rent():
    spec = importlib.util.spec_from_file_location("rent_script", RENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rent_script"] = module        # dataclasses look the module up while it is being loaded
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rent_module():
    return _load_rent()


@pytest.fixture
def rent(rent_module, tmp_path, monkeypatch):
    """The module, with every default path in a temporary folder and every way to Vast.ai closed."""
    def no_network(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    def no_vastai(*args, **kwargs):
        raise AssertionError("a test tried to run the vastai tool")

    monkeypatch.setattr(rent_module, "post_public", no_network)
    monkeypatch.setattr(rent_module, "run_vastai", no_vastai)
    monkeypatch.setattr(rent_module, "DEFAULT_ENV_FILE", tmp_path / "default.env")
    monkeypatch.setattr(rent_module, "DEFAULT_LEDGER", tmp_path / "default_ledger.csv")
    monkeypatch.setattr(rent_module, "DEFAULT_OFFERS_CACHE", tmp_path / "default_offers.json")
    monkeypatch.setattr(rent_module, "now_utc", lambda: NOW)
    monkeypatch.delenv("ACCESSDESK_VAST_ENV", raising=False)
    return rent_module


@pytest.fixture
def files(tmp_path):
    """Paths of a key file (with the made-up key), a ledger and an offer list; only the key file exists."""
    env_file = tmp_path / "vast.env"
    env_file.write_text(f"# the key of the test\nVAST_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    return {"env": env_file, "ledger": tmp_path / "results" / "spend_ledger.csv",
            "offers": tmp_path / "offers.json", "ssh": tmp_path / "id_test"}


def run_main(rent, files, *args: str) -> int:
    return rent.main(["--env-file", str(files["env"]), "--ledger", str(files["ledger"]),
                      "--offers-cache", str(files["offers"]), "--ssh-key", str(files["ssh"]), *args])


# ---------------------------------------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------------------------------------
def test_redact_replaces_the_key_everywhere(rent):
    secret = rent.Secret(FAKE_KEY)
    text = f"first {FAKE_KEY} then again {FAKE_KEY}."
    assert rent.redact(text, secret) == "first *** then again ***."


@pytest.mark.parametrize("text, expected", [
    ("Authorization: Bearer abc123XYZ", "Authorization: Bearer ***"),
    ('-H "Authorization: Bearer abc123XYZ" https://x', '-H "Authorization: Bearer ***" https://x'),
    ("vastai show user --api-key abc123XYZ --raw", "vastai show user --api-key *** --raw"),
    ("vastai --api-key=abc123XYZ show user", "vastai --api-key=*** show user"),
    ("VAST_API_KEY=abc123XYZ vastai show user", "VAST_API_KEY=*** vastai show user"),
    ('{"api_key": "abc123XYZ", "id": 7}', '{"api_key": "***", "id": 7}'),
    ("CONTAINER_API_KEY=abc123XYZ", "CONTAINER_API_KEY=***"),
    ("Sent key from $VAST_API_KEY (ends in ...a3f9). Update",
     "Sent key from $VAST_API_KEY (ends in ...****). Update"),
])
def test_redact_blanks_key_shapes_without_knowing_the_key(rent, text, expected):
    assert rent.redact(text) == expected


def test_redact_leaves_ordinary_text_alone(rent):
    text = '{"id": 123, "dph_total": 0.5, "label": "accessdesk-20261117T080000Z"} destroying instance 123.'
    assert rent.redact(text, rent.Secret(FAKE_KEY)) == text


def test_secret_never_prints_its_value(rent):
    secret = rent.Secret(FAKE_KEY)
    assert str(secret) == "***" and repr(secret) == "***" and f"{secret}" == "***"
    assert FAKE_KEY not in f"{secret!r} {[secret]} { {'k': secret} }"
    assert secret.reveal() == FAKE_KEY


def test_run_vastai_passes_the_key_in_the_environment_and_redacts_the_output(rent_module, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, stdout=f'{{"msg": "{FAKE_KEY}"}}\n',
                                           stderr=f"Sent key (ends in ...{FAKE_KEY[-4:]})\n")

    monkeypatch.setattr(rent_module, "vastai_bin", lambda: "/nowhere/vastai")
    monkeypatch.setattr(rent_module.subprocess, "run", fake_run)
    done = rent_module.run_vastai(["show", "user", "--raw"], rent_module.Secret(FAKE_KEY))
    assert seen["env"]["VAST_API_KEY"] == FAKE_KEY                 # the tool gets the key ...
    assert seen["env"]["VASTAI_NO_UPDATE_CHECK"] == "1"
    assert all(FAKE_KEY not in part for part in seen["argv"])       # ... but never on its command line
    assert FAKE_KEY not in done.stdout and "***" in done.stdout
    assert FAKE_KEY[-4:] not in done.stderr


def test_shown_command_has_no_key(rent):
    shown = rent.shown_command(rent.call_destroy_instance(4242))
    assert shown == "VAST_API_KEY=*** vastai destroy instance 4242 -y"


# ---------------------------------------------------------------------------------------------------------
# The key file
# ---------------------------------------------------------------------------------------------------------
def test_parse_env_text_comments_quotes_and_spaces(rent):
    text = "\n".join([
        "# a comment",
        "",
        "   # an indented comment",
        "PLAIN=abc",
        "  SPACED  =  def  ",
        'DOUBLE="ghi jkl"',
        "SINGLE='mno # not a comment'",
        "export EXPORTED=pqr",
        "TRAILING=stu # a comment after the value",
        "WITH_EQUALS=a=b=c",
        "EMPTY=",
        'QUOTED_THEN_COMMENT="vwx"   # comment',
    ])
    assert rent.parse_env_text(text) == {
        "PLAIN": "abc", "SPACED": "def", "DOUBLE": "ghi jkl", "SINGLE": "mno # not a comment",
        "EXPORTED": "pqr", "TRAILING": "stu", "WITH_EQUALS": "a=b=c", "EMPTY": "",
        "QUOTED_THEN_COMMENT": "vwx"}


@pytest.mark.parametrize("bad_line", [
    FAKE_KEY, "1NAME=x", "NA ME=x", '"NAME"=x', f'VAST_API_KEY="{FAKE_KEY}'])
def test_parse_env_text_errors_name_the_line_and_hide_its_content(rent, bad_line):
    with pytest.raises(rent.RentError) as error:
        rent.parse_env_text("GOOD=1\n# comment\n" + bad_line + "\n")
    assert "line 3" in str(error.value)
    assert FAKE_KEY not in str(error.value)


def test_load_api_key_reads_the_key(rent, files):
    content = f'# comment\n\nOTHER=1\nexport VAST_API_KEY = "{FAKE_KEY}"  # mine\n'
    files["env"].write_text(content, encoding="utf-8")
    secret = rent.load_api_key(files["env"])
    assert isinstance(secret, rent.Secret) and secret.reveal() == FAKE_KEY


def test_load_api_key_missing_file_gives_a_clear_error(rent, tmp_path):
    missing = tmp_path / "nowhere" / "vast.env"
    with pytest.raises(rent.RentError) as error:
        rent.load_api_key(missing)
    message = str(error.value)
    assert str(missing) in message and "VAST_API_KEY" in message and "not found" in message


@pytest.mark.parametrize("content, expected", [
    ("OTHER=1\n", "has no line VAST_API_KEY"),
    ("VAST_API_KEY=\n", "is empty"),
    ('VAST_API_KEY=""\n', "is empty"),
    ('VAST_API_KEY="abcdefgh ijklmnop"\n', "white space"),
])
def test_load_api_key_refuses_a_missing_or_broken_value(rent, files, content, expected):
    files["env"].write_text(content, encoding="utf-8")
    with pytest.raises(rent.RentError) as error:
        rent.load_api_key(files["env"])
    assert expected in str(error.value)
    assert "abcdefgh" not in str(error.value)


def test_main_reports_a_missing_key_file_without_a_traceback(rent, files, capsys):
    rent.save_offers(files["offers"], [OFFER], 60, NOW)
    files["env"].unlink()
    assert run_main(rent, files, "rent", "5001", "--max-minutes", "60", "--dry-run") == 1
    out = capsys.readouterr().out
    assert "ERROR: Key file not found" in out and str(files["env"]) in out


# ---------------------------------------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------------------------------------
def rent_line(rent, instance_id=900, projected="1.2500", **extra) -> dict:
    row = {"time_utc": "2026-11-17T08:00:00Z", "event": "rent", "instance_id": instance_id, "offer_id": 5001,
           "gpu": "RTX 4090", "dph_usd": rent.money(0.5), "max_minutes": 120,
           "bandwidth_est_usd": rent.money(0.25), "projected_usd": projected, "note": "label accessdesk-x"}
    row.update(extra)
    return row


def test_ledger_line_format(rent):
    assert rent.LEDGER_FIELDS == (
        "time_utc", "event", "instance_id", "offer_id", "gpu", "dph_usd", "max_minutes", "bandwidth_est_usd",
        "projected_usd", "actual_usd", "note")
    line = rent.ledger_line(rent_line(rent))
    assert line == "2026-11-17T08:00:00Z,rent,900,5001,RTX 4090,0.5000,120,0.2500,1.2500,,label accessdesk-x"


def test_ledger_line_quotes_commas_and_refuses_unknown_fields(rent):
    line = rent.ledger_line(rent_line(rent, note='a, b and "c"'))
    assert line.endswith(',"a, b and ""c"""')
    with pytest.raises(ValueError):
        rent.ledger_line({"event": "rent", "price": 1})


def test_ledger_line_never_holds_a_key(rent):
    line = rent.ledger_line(rent_line(rent, note=f"VAST_API_KEY={FAKE_KEY}"))
    assert FAKE_KEY not in line


def test_append_ledger_writes_the_header_once(rent, files):
    rent.append_ledger(files["ledger"], rent_line(rent, 900))
    rent.append_ledger(files["ledger"], rent_line(rent, 901))
    lines = files["ledger"].read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(rent.LEDGER_FIELDS)
    assert len(lines) == 3 and lines[1].split(",")[2] == "900" and lines[2].split(",")[2] == "901"
    rows = rent.read_ledger(files["ledger"])
    assert [row["instance_id"] for row in rows] == ["900", "901"]
    assert rows[0]["note"] == "label accessdesk-x" and rows[0]["actual_usd"] == ""


def test_ledger_total_uses_real_cost_after_destroy_and_projected_cost_before(rent):
    rows = [
        {"event": "rent", "instance_id": "1", "projected_usd": "2.0000", "actual_usd": ""},
        {"event": "rent", "instance_id": "2", "projected_usd": "3.0000", "actual_usd": ""},
        {"event": "destroy", "instance_id": "1", "projected_usd": "", "actual_usd": "0.7500"},
        {"event": "destroy", "instance_id": "3", "projected_usd": "", "actual_usd": "0.1000"},  # no rent line
        {"event": "destroy", "instance_id": "2", "projected_usd": "", "actual_usd": ""},        # cost unknown
    ]
    # machine 1: 0.75 (real), machine 2: 3.00 (still projected), machine 3: 0.10
    assert rent.ledger_total(rows) == pytest.approx(3.85)
    assert rent.ledger_total([]) == 0.0
    assert rent.ledger_total(rows[:2]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------------------------------------
def test_check_budget_refuses_above_and_allows_up_to_the_budget(rent):
    rows = [{"event": "rent", "instance_id": "1", "projected_usd": "30.0000", "actual_usd": ""}]
    assert rent.check_budget(rows, 10.0, 40.0) == pytest.approx(40.0)       # exactly the budget: allowed
    with pytest.raises(rent.BudgetError) as error:
        rent.check_budget(rows, 10.01, 40.0)
    assert "REFUSED" in str(error.value) and "40.00" in str(error.value) and "30.00" in str(error.value)


def test_projected_cost_by_hand(rent):
    # 0.50 USD/h for 90 minutes = 0.75; 20 GB at 0.01 USD/GB = 0.20
    assert rent.bandwidth_estimate(OFFER, 20.0) == pytest.approx(0.20)
    assert rent.projected_cost(OFFER, 90, 20.0) == pytest.approx(0.95)


def test_rent_is_refused_when_the_ledger_plus_the_new_machine_pass_the_budget(rent, files, capsys):
    rent.save_offers(files["offers"], [OFFER], 60, NOW)
    rent.append_ledger(files["ledger"], rent_line(rent, 900, projected="39.5000"))
    before = files["ledger"].read_text(encoding="utf-8")
    # the new machine: 0.5 USD/h x 1 h + 20 GB x 0.01 = 0.70 USD; 39.50 + 0.70 > 40
    code = run_main(rent, files, "rent", "5001", "--max-minutes", "60")       # not a dry run
    out = capsys.readouterr().out
    assert code == 2 and "REFUSED" in out and "40.20" in out
    assert files["ledger"].read_text(encoding="utf-8") == before               # nothing was written
    # (run_vastai is replaced by a function that fails the test, so nothing was created either)


def test_rent_inside_the_budget_reaches_the_plan(rent, files, capsys):
    rent.save_offers(files["offers"], [OFFER], 60, NOW)
    rent.append_ledger(files["ledger"], rent_line(rent, 900, projected="39.3000"))
    assert run_main(rent, files, "rent", "5001", "--max-minutes", "60", "--dry-run") == 0
    assert "projected ledger total 40.00 of 40.00 USD" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------
# Dry runs: the exact call is shown, nothing happens, and the key is nowhere
# ---------------------------------------------------------------------------------------------------------
def test_rent_dry_run_prints_the_exact_create_call_and_no_key(rent, files, capsys):
    rent.save_offers(files["offers"], [OFFER], 60, NOW)
    code = run_main(rent, files, "rent", "5001", "--max-minutes", "90", "--dry-run")
    out = capsys.readouterr().out
    assert code == 0
    assert FAKE_KEY not in out and FAKE_KEY[-4:] not in out
    assert "DRY RUN: nothing was created and nothing was written." in out
    assert not files["ledger"].exists()

    create_line = next(line for line in out.splitlines() if line.startswith("  3. "))
    words = shlex.split(create_line[len("  3. "):])
    onstart = words[words.index("--onstart-cmd") + 1]
    expected = rent.call_create_instance(5001, image=rent.IMAGE_TAG, disk_gb=60,
                                         label="accessdesk-20261117T080000Z", onstart=onstart)
    assert words == ["VAST_API_KEY=***", "vastai", *expected]
    assert words[2:5] == ["create", "instance", "5001"]
    for flag in ("--ssh", "--direct", "--cancel-unavail", "--raw"):
        assert flag in words
    assert words[words.index("--disk") + 1] == "60"
    assert words[words.index("--image") + 1] == "vastai/pytorch:cuda-13.0.3-auto"
    assert words[words.index("--retry") + 1] == "1"

    # The on-start command carries the real watchdog script and starts it with the time limit.
    assert onstart.endswith("bash /root/watchdog.sh 90 >> /workspace/watchdog.out 2>&1 < /dev/null &")
    blob = re.search(r"echo (\S+) \| base64 -d \| gunzip > /root/watchdog.sh", onstart).group(1)
    script = gzip.decompress(base64.b64decode(blob)).decode("utf-8")
    assert script == rent.strip_comments(WATCHDOG_PATH.read_text(encoding="utf-8"))

    # The ledger line that would be written: 0.5 x 1.5 h + 0.20 download = 0.95
    ledger_line = out.splitlines()[out.splitlines().index(f"  4. append to {files['ledger']}:") + 1].strip()
    assert ledger_line.startswith(
        "2026-11-17T08:00:00Z,rent,<new id>,5001,RTX 4090,0.5000,90,0.2000,0.9500,,label accessdesk-")
    assert ledger_line.endswith(f"image vastai/pytorch:cuda-13.0.3-auto ({rent.IMAGE_DIGEST} on 2026-11-17)")


def test_destroy_dry_run_prints_the_call_and_does_nothing(rent, files, capsys):
    rent.append_ledger(files["ledger"], rent_line(rent, 4242))
    before = files["ledger"].read_text(encoding="utf-8")
    assert run_main(rent, files, "destroy", "4242", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "VAST_API_KEY=*** vastai destroy instance 4242 -y" in out
    assert "VAST_API_KEY=*** vastai show instance 4242 --raw" in out
    assert "DRY RUN" in out and FAKE_KEY not in out
    assert files["ledger"].read_text(encoding="utf-8") == before


INSTANCES = [
    {"id": 4242, "label": "accessdesk-20261117T070000Z", "gpu_name": "RTX 4090", "actual_status": "running",
     "dph_total": 0.5},
    {"id": 4343, "label": None, "gpu_name": "RTX 4090", "actual_status": "running", "dph_total": 0.5},
    {"id": 4444, "label": "someone-else", "gpu_name": "RTX 3090", "actual_status": "running",
     "dph_total": 0.2},
]


def fake_vastai(rent, answers: dict, calls: list):
    """A stand-in for run_vastai: `answers` maps the words of a call (joined by spaces) to the output."""
    def run(args, secret, *, timeout=300.0):
        assert isinstance(secret, rent.Secret)
        calls.append(list(args))
        answer = answers[" ".join(args)]
        answer = answer.pop(0) if isinstance(answer, list) else answer
        return answer if isinstance(answer, rent.Completed) else rent.Completed(0, answer, "")
    return run


def test_destroy_all_dry_run_lists_our_machines_only_and_destroys_nothing(rent, files, capsys, monkeypatch):
    rent.append_ledger(files["ledger"], rent_line(rent, 4343))            # 4343 is ours by the ledger
    calls: list = []
    answers = {"show instances --raw": json.dumps(INSTANCES)}
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, answers, calls))
    assert run_main(rent, files, "destroy-all", "--dry-run") == 0
    out = capsys.readouterr().out
    assert calls == [["show", "instances", "--raw"]]                       # only the read; no destroy call
    assert "VAST_API_KEY=*** vastai destroy instance 4242 -y" in out     # ours by its label
    assert "VAST_API_KEY=*** vastai destroy instance 4343 -y" in out     # ours by the ledger
    assert "destroy instance 4444" not in out
    assert "left alone" in out and "4444" in out
    assert "DRY RUN: nothing was destroyed." in out and FAKE_KEY not in out


def test_destroy_all_everything_includes_foreign_machines(rent, files, capsys, monkeypatch):
    calls: list = []
    answers = {"show instances --raw": json.dumps(INSTANCES)}
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, answers, calls))
    assert run_main(rent, files, "destroy-all", "--everything", "--dry-run") == 0
    out = capsys.readouterr().out
    assert all(f"vastai destroy instance {number} -y" in out for number in (4242, 4343, 4444))
    assert calls == [["show", "instances", "--raw"]]


def test_destroy_writes_the_real_cost_to_the_ledger(rent, files, capsys, monkeypatch):
    rent.append_ledger(files["ledger"], rent_line(rent, 4242))            # download estimate 0.25 USD
    record = {"id": 4242, "gpu_name": "RTX 4090", "dph_total": 0.5, "start_date": time.time() - 2 * 3600,
              "actual_status": "running"}
    answers = {"show instance 4242 --raw": [json.dumps(record), json.dumps({"instances": None})],
               "destroy instance 4242 -y": "destroying instance 4242.\n"}
    calls: list = []
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, answers, calls))
    assert run_main(rent, files, "destroy", "4242") == 0
    assert calls == [["show", "instance", "4242", "--raw"], ["destroy", "instance", "4242", "-y"],
                     ["show", "instance", "4242", "--raw"]]
    rows = rent.read_ledger(files["ledger"])
    assert [row["event"] for row in rows] == ["rent", "destroy"]
    destroyed = rows[1]
    assert destroyed["instance_id"] == "4242" and destroyed["time_utc"] == "2026-11-17T08:00:00Z"
    # 2 hours x 0.5 USD/h = 1.00, plus the 0.25 download estimate of the rent line
    assert float(destroyed["actual_usd"]) == pytest.approx(1.25, abs=0.01)
    assert rent.ledger_total(rows) == pytest.approx(1.25, abs=0.01)       # real cost, not the projection
    assert "gone" in capsys.readouterr().out


def test_destroy_of_a_machine_that_is_already_gone_keeps_the_projected_cost(rent, files, monkeypatch):
    rent.append_ledger(files["ledger"], rent_line(rent, 4242, projected="1.2500"))
    answers = {"show instance 4242 --raw": json.dumps({"instances": None})}
    calls: list = []
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, answers, calls))
    assert run_main(rent, files, "destroy", "4242") == 0
    assert calls == [["show", "instance", "4242", "--raw"]]               # no destroy call: nothing is there
    rows = rent.read_ledger(files["ledger"])
    assert rows[1]["event"] == "destroy" and rows[1]["actual_usd"] == "1.2500"
    assert "already gone" in rows[1]["note"]


def test_destroy_is_tried_even_if_the_machine_cannot_be_read(rent, files, monkeypatch):
    rent.append_ledger(files["ledger"], rent_line(rent, 4242, projected="1.2500"))
    broken = rent.Completed(0, "", json.dumps({"error": True, "status_code": 502, "msg": "bad gateway"}))
    answers = {"show instance 4242 --raw": [broken, json.dumps({"instances": None})],
               "destroy instance 4242 -y": "destroying instance 4242.\n"}
    calls: list = []
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, answers, calls))
    assert run_main(rent, files, "destroy", "4242") == 0
    assert ["destroy", "instance", "4242", "-y"] in calls
    destroyed = rent.read_ledger(files["ledger"])[1]
    assert destroyed["actual_usd"] == "1.2500" and "record not readable" in destroyed["note"]


def test_destroy_that_is_not_confirmed_is_an_error_and_is_recorded(rent, files, capsys, monkeypatch):
    record = {"id": 4242, "gpu_name": "RTX 4090", "dph_total": 0.5, "start_date": time.time() - 3600}
    answers = {"show instance 4242 --raw": json.dumps(record), "destroy instance 4242 -y": "no such thing\n"}
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, answers, []))
    monkeypatch.setattr(rent.time, "sleep", lambda seconds: None)
    assert run_main(rent, files, "destroy", "4242") == 1
    assert "was NOT destroyed" in capsys.readouterr().out
    assert "DESTROY NOT CONFIRMED" in rent.read_ledger(files["ledger"])[0]["note"]


# ---------------------------------------------------------------------------------------------------------
# The real `rent` path, with a stand-in for the vastai tool and for SSH
# ---------------------------------------------------------------------------------------------------------
RUNNING = {"id": 7001, "actual_status": "running", "status_msg": "ok", "gpu_name": "RTX 4090",
           "dph_total": 0.6, "start_date": 0.0, "public_ipaddr": "203.0.113.9",
           "ports": {"22/tcp": [{"HostPort": "40123"}]}}
ARMED_PROBE = "WATCHDOG_ALIVE\n2026-11-17T08:01:00Z watchdog[9] ARMED: instance 7001 will be destroyed"


class FakeCloud:
    """Answers the calls of `rent` like Vast.ai and the machine would, and remembers them."""

    def __init__(self, rent, *, create_ok: bool = True, probe: str = ARMED_PROBE, price: float = 0.6,
                 status: str = "running"):
        self.rent, self.create_ok, self.probe, self.price, self.status = rent, create_ok, probe, price, status
        self.vastai_calls: list[list[str]] = []
        self.ssh_commands: list[str] = []
        self.destroyed = False

    def run_vastai(self, args, secret, *, timeout=300.0):
        assert isinstance(secret, self.rent.Secret)
        self.vastai_calls.append(list(args))
        kind = tuple(args[:2])
        if kind == ("show", "ssh-keys"):
            return self.rent.Completed(0, json.dumps([{"id": 1, "public_key": "ssh-ed25519 OTHERKEY"}]), "")
        if kind == ("create", "ssh-key"):
            return self.rent.Completed(0, "ssh-key created {'success': True}\n", "")
        if kind == ("create", "instance"):
            if not self.create_ok:
                error = {"error": True, "status_code": 400, "msg": "no_such_ask"}
                return self.rent.Completed(0, "", json.dumps(error))
            return self.rent.Completed(0, json.dumps({"success": True, "new_contract": 7001}), "")
        if kind == ("show", "instance"):
            if self.destroyed:
                return self.rent.Completed(0, json.dumps({"instances": None}), "")
            record = dict(RUNNING, dph_total=self.price, start_date=time.time() - 1800,
                          actual_status=self.status)
            return self.rent.Completed(0, json.dumps(record), "")
        if kind == ("destroy", "instance"):
            self.destroyed = True
            return self.rent.Completed(0, "destroying instance 7001.\n", "")
        raise AssertionError(f"unexpected vastai call: {args[:3]}")

    def run_local(self, argv, *, capture=False, timeout=None, stdin_bytes=None):
        assert argv[0] == "ssh" and FAKE_KEY not in " ".join(argv)
        self.ssh_commands.append(argv[-1])
        if argv[-1] == self.rent.WATCHDOG_PROBE:
            return self.rent.Completed(0, self.probe, "")
        return self.rent.Completed(0, "", "")


def start_rent(rent, files, monkeypatch, cloud: FakeCloud, *extra: str) -> int:
    rent.save_offers(files["offers"], [OFFER], 60, NOW)
    Path(str(files["ssh"]) + ".pub").write_text("ssh-ed25519 AAAATESTKEY me@laptop\n", encoding="utf-8")
    monkeypatch.setattr(rent, "run_vastai", cloud.run_vastai)
    monkeypatch.setattr(rent, "run_local", cloud.run_local)
    monkeypatch.setattr(rent.time, "sleep", lambda seconds: None)
    # a short wait for SSH: with the sleeps switched off, a broken status check must end the test quickly
    return run_main(rent, files, "rent", "5001", "--max-minutes", "90", "--wait-minutes", "0.02", *extra)


def test_rent_creates_the_machine_and_writes_the_real_price(rent, files, capsys, monkeypatch):
    cloud = FakeCloud(rent)
    assert start_rent(rent, files, monkeypatch, cloud) == 0
    out = capsys.readouterr().out
    kinds = [tuple(call[:2]) for call in cloud.vastai_calls]
    # the key is registered before the machine is created
    assert kinds[:3] == [("show", "ssh-keys"), ("create", "ssh-key"), ("create", "instance")]
    assert cloud.vastai_calls[1] == ["create", "ssh-key", "ssh-ed25519 AAAATESTKEY"]   # no "me@laptop"
    assert ("destroy", "instance") not in kinds
    rows = rent.read_ledger(files["ledger"])
    assert len(rows) == 1 and rows[0]["event"] == "rent" and rows[0]["instance_id"] == "7001"
    # listed at 0.50 USD/h, really 0.60: 0.60 x 1.5 h + 0.20 download = 1.10
    assert rows[0]["dph_usd"] == "0.6000" and rows[0]["projected_usd"] == "1.1000"
    assert f"ssh -p 40123 -i {files['ssh']} root@203.0.113.9" in out
    assert "true" in cloud.ssh_commands and rent.WATCHDOG_PROBE in cloud.ssh_commands
    assert FAKE_KEY not in out


def test_rent_destroys_the_machine_if_the_watchdog_does_not_start(rent, files, capsys, monkeypatch):
    cloud = FakeCloud(rent, probe="2026-11-17T08:01:00Z watchdog[9] FATAL: CONTAINER_API_KEY is missing")
    assert start_rent(rent, files, monkeypatch, cloud) == 1
    out = capsys.readouterr().out
    assert "the watchdog could not be started" in out and cloud.destroyed
    assert any("cat > /root/watchdog.sh" in command for command in cloud.ssh_commands)    # the second try
    assert [row["event"] for row in rent.read_ledger(files["ledger"])] == ["rent", "destroy"]


def test_rent_passes_the_preflight_switch_to_both_ways_of_starting_the_watchdog(rent, files, monkeypatch):
    cloud = FakeCloud(rent, probe="")
    assert start_rent(rent, files, monkeypatch, cloud, "--skip-watchdog-preflight") == 1
    create = next(call for call in cloud.vastai_calls if call[:2] == ["create", "instance"])
    wanted = "WATCHDOG_SKIP_PREFLIGHT=1 $(command -v setsid) nohup bash /root/watchdog.sh 90 "
    assert wanted in create[create.index("--onstart-cmd") + 1]
    assert any(command.startswith(wanted) for command in cloud.ssh_commands)


@pytest.mark.parametrize("status, expected", [
    ("loading", "did not answer on SSH"),          # never starts: the wait ends, with a hint about the image
    ("exited", "will not start"),                  # a state from which a machine never reaches "running"
])
def test_rent_destroys_a_machine_that_does_not_start(rent, files, capsys, monkeypatch, status, expected):
    cloud = FakeCloud(rent, status=status)
    assert start_rent(rent, files, monkeypatch, cloud) == 1
    out = capsys.readouterr().out
    assert expected in out and cloud.destroyed and "true" not in cloud.ssh_commands
    assert ("this host may have to download the whole image" in out) == (status == "loading")
    assert [row["event"] for row in rent.read_ledger(files["ledger"])] == ["rent", "destroy"]


def test_rent_keeps_the_machine_on_request_when_the_start_fails(rent, files, capsys, monkeypatch):
    cloud = FakeCloud(rent, probe="")
    assert start_rent(rent, files, monkeypatch, cloud, "--keep-on-failure") == 1
    assert not cloud.destroyed and "NO watchdog" in capsys.readouterr().out
    assert [row["event"] for row in rent.read_ledger(files["ledger"])] == ["rent"]


def test_rent_destroys_the_machine_if_the_real_price_breaks_the_budget(rent, files, capsys, monkeypatch):
    # listed: 0.5 x 1.5 + 0.2 = 0.95 (inside 1.00); real price 0.9: 0.9 x 1.5 + 0.2 = 1.55 (outside)
    cloud = FakeCloud(rent, price=0.9)
    assert start_rent(rent, files, monkeypatch, cloud, "--budget", "1.0") == 1
    assert "REFUSED" in capsys.readouterr().out and cloud.destroyed
    rows = rent.read_ledger(files["ledger"])
    assert [row["event"] for row in rows] == ["rent", "destroy"] and rows[0]["projected_usd"] == "1.5500"


def test_rent_reports_a_machine_that_was_not_created(rent, files, capsys, monkeypatch):
    cloud = FakeCloud(rent, create_ok=False)
    assert start_rent(rent, files, monkeypatch, cloud) == 1
    out = capsys.readouterr().out
    assert "NOT confirmed as created" in out and "no_such_ask" in out
    assert "accessdesk-20261117T080000Z" in out and "--image" in out
    assert not files["ledger"].exists()


def test_balance_shows_only_the_money(rent, files, capsys, monkeypatch):
    user = {"credit": 37.5, "balance": 0.0, "email": "someone@example.com", "fullname": "Some One"}
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, {"show user --raw": json.dumps(user)}, []))
    assert run_main(rent, files, "balance") == 0
    out = capsys.readouterr().out
    assert "credit:  37.5" in out and "example.com" not in out and "Some One" not in out


def test_status_lists_machines_and_the_ledger_total(rent, files, capsys, monkeypatch):
    rent.append_ledger(files["ledger"], rent_line(rent, 4343, projected="2.5000"))
    listed = [dict(inst, start_date=time.time() - 3600) for inst in INSTANCES]
    answers = {"show instances --raw": json.dumps(listed)}
    monkeypatch.setattr(rent, "run_vastai", fake_vastai(rent, answers, []))
    assert run_main(rent, files, "status") == 0
    out = capsys.readouterr().out
    assert "3 machine(s) on the account" in out and "ledger total (projected): 2.50 USD" in out
    assert out.count("ours=yes") == 2 and out.count("ours=NO") == 1


def test_sync_up_and_fetch_dry_runs(rent, files, capsys):
    assert run_main(rent, files, "sync-up", "4242", "--dry-run") == 0
    up = capsys.readouterr().out
    assert "--delete" in up
    for pattern in ("/.venv/", "/.cache/", "/_dev/", "/.git/", "/results/", "*.env"):
        assert f"--exclude={pattern}" in up or f"'--exclude={pattern}'" in up
    assert "root@<host>:/workspace/repo/" in up

    assert run_main(rent, files, "fetch", "4242", "--dry-run") == 0
    down = capsys.readouterr().out
    assert "--delete" not in down                              # fetching never deletes local results
    assert "--exclude=spend_ledger.csv" in down                # and never overwrites the ledger
    assert "root@<host>:/workspace/repo/results/" in down


def test_run_dry_run_shows_the_remote_command(rent, files, capsys):
    assert run_main(rent, files, "run", "4242", "python scripts/gpu/smoke.py", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "cd /workspace/repo && if [ -r /workspace/env.sh ]; then . /workspace/env.sh; fi && " \
           "python scripts/gpu/smoke.py" in out
    assert run_main(rent, files, "run", "4242", "echo 'a b'", "--detach", "--log", "job1", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "/workspace/logs/job1.log" in out and "nohup bash -c" in out
    assert "mkdir -p /workspace/logs || exit 1; " in out and "logs && " not in out
    assert run_main(rent, files, "run", "4242", "true", "--detach", "--log", "../x", "--dry-run") == 1


# ---------------------------------------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------------------------------------
def test_offer_query_is_what_the_vastai_parser_builds(rent):
    """The expected value was produced on 2026-11-17 by vastai 1.11.0's own parse_query (vastai/api/query.py)
    from 'gpu_name=RTX_4090 num_gpus=1 verified=true reliability>0.99 cuda_max_good>=13.2 inet_down>=500
    cpu_ram>=48 direct_port_count>=1 rentable=true cpu_arch=amd64' on top of the tool's default filter."""
    from_the_tool = {
        "verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True},
        "gpu_name": {"eq": "RTX 4090"}, "num_gpus": {"eq": "1"}, "reliability": {"gt": "0.99"},
        "cuda_max_good": {"gte": "13.2"}, "inet_down": {"gte": "500"}, "cpu_ram": {"gte": 48000.0},
        "direct_port_count": {"gte": "1"}, "cpu_arch": {"eq": "amd64"}}
    query = rent.offer_query()
    assert {name: query[name] for name in from_the_tool} == from_the_tool
    assert query["order"] == [["dph_total", "asc"]] and query["type"] == "on-demand"
    assert query["allocated_storage"] == 60.0 and query["limit"] == 64
    assert set(query) == set(from_the_tool) | {"order", "type", "limit", "allocated_storage"}
    assert rent.offer_query(gpu="RTX_3090", min_cuda=13.0, disk_gb=80)["gpu_name"] == {"eq": "RTX 3090"}


def test_check_offer(rent):
    assert rent.check_offer(OFFER, gpu="RTX 4090", min_cuda=13.2) == []
    assert rent.check_offer(OFFER, gpu="RTX_4090", min_cuda=13.2) == []
    bad = dict(OFFER, gpu_name="RTX 5090", num_gpus=2, cuda_max_good=13.0, cpu_arch="arm64",
               verification="unverified", rented=True)
    assert len(rent.check_offer(bad, gpu="RTX 4090", min_cuda=13.2)) == 6
    assert len(rent.check_offer(dict(OFFER, rentable=False), gpu="RTX 4090", min_cuda=13.2)) == 1


def test_offer_list_round_trip_and_its_limits(rent, files):
    rent.save_offers(files["offers"], [dict(OFFER, secret_field="dropped"), dict(OFFER, id=5002)], 60, NOW)
    offer = rent.load_cached_offer(files["offers"], 5001, 60, NOW + timedelta(minutes=14))
    assert offer == OFFER                                                 # only the fields of OFFER_FIELDS
    assert set(OFFER) == set(rent.OFFER_FIELDS)
    for when, disk, number, expected in [
        (NOW + timedelta(minutes=16), 60, 5001, "minutes old"),
        (NOW - timedelta(minutes=1), 60, 5001, "minutes old"),            # a list from the future: refused
        (NOW, 80, 5001, "priced with 60 GB"),
        (NOW, 60, 9999, "not in the list"),
    ]:
        with pytest.raises(rent.RentError) as error:
            rent.load_cached_offer(files["offers"], number, disk, when)
        assert expected in str(error.value)
    with pytest.raises(rent.RentError) as error:
        rent.load_cached_offer(files["offers"].with_name("missing.json"), 5001, 60, NOW)
    assert "no list of offers" in str(error.value)


def test_rent_refuses_an_offer_that_does_not_fit(rent, files, capsys):
    rent.save_offers(files["offers"], [dict(OFFER, cuda_max_good=13.0)], 60, NOW)
    assert run_main(rent, files, "rent", "5001", "--max-minutes", "60", "--dry-run") == 1
    assert "does not fit this project" in capsys.readouterr().out


def test_offers_table_has_one_line_per_offer(rent):
    table = rent.offers_table([OFFER, dict(OFFER, id=5002, dph_total=0.75)], hours=8.0, download_gb=20.0)
    lines = table.splitlines()
    assert len(lines) == 3 and lines[1].split()[0] == "5001" and lines[2].split()[0] == "5002"
    assert lines[1].split()[1] == "0.500" and lines[1].split()[2] == "4.20"      # 8 x 0.5 + 20 x 0.01


def test_offers_command_saves_exactly_the_offers_it_shows(rent, files, capsys, monkeypatch):
    many = [dict(OFFER, id=5000 + number, dph_total=0.4 + number / 100) for number in range(5)]
    many.append(dict(OFFER, id=5900, duration=3600.0))                    # rentable for one hour only
    monkeypatch.setattr(rent, "post_public", lambda url, payload, timeout=60.0: {"offers": many})
    assert run_main(rent, files, "offers", "--top", "3") == 0
    out = capsys.readouterr().out
    saved = json.loads(files["offers"].read_text(encoding="utf-8"))
    assert [offer["id"] for offer in saved["offers"]] == [5000, 5001, 5002]
    assert saved["disk_gb"] == 60 and saved["time_utc"] == "2026-11-17T08:00:00Z"
    assert "5 offers pass the filter" in out and "5900" not in out


# ---------------------------------------------------------------------------------------------------------
# Answers of the vastai tool
# ---------------------------------------------------------------------------------------------------------
def test_api_error_reads_both_shapes_of_an_error(rent):
    raw = rent.Completed(0, "", '{"error": true, "status_code": 401, "msg": "Invalid user key"}\n')
    plain = rent.Completed(0, "", "Failed with error 404: Instance not found\n  Sent key from ...\n")
    assert rent.api_error(raw) == (401, "Invalid user key")
    assert rent.api_error(plain) == (404, "Instance not found")
    assert rent.api_error(rent.Completed(0, '{"success": true}', "a warning\n")) is None


def test_parse_json_skips_leading_lines(rent):
    assert rent.parse_json('{"success": true, "new_contract": 7}') == {"success": True, "new_contract": 7}
    assert rent.parse_json('some notice\n[\n {"id": 1}\n]\n') == [{"id": 1}]
    with pytest.raises(ValueError):
        rent.parse_json("Started. nothing useful")


def test_get_instance_tells_gone_from_broken(rent, monkeypatch):
    secret = rent.Secret(FAKE_KEY)

    def error(code: int, text: str):
        return rent.Completed(0, "", json.dumps({"error": True, "status_code": code, "msg": text}))

    record = {"id": 7, "actual_status": "running"}
    cases = {
        "found": (rent.Completed(0, json.dumps(record), ""), record),
        "gone": (rent.Completed(0, '{"instances": null}', ""), None),
        "gone 404": (error(404, "Instance not found"), None),
    }
    for name, (done, expected) in cases.items():
        monkeypatch.setattr(rent, "run_vastai", lambda args, secret, done=done, **kwargs: done)
        assert rent.get_instance(7, secret) == expected, name
    for done in (error(401, "Invalid user key"), error(404, "Session expired. Please log in again."),
                 rent.Completed(0, "garbage", ""), rent.Completed(0, '{"unexpected": 1}', "")):
        monkeypatch.setattr(rent, "run_vastai", lambda args, secret, done=done, **kwargs: done)
        with pytest.raises(rent.RentError):
            rent.get_instance(7, secret)


def test_ssh_target_direct_then_proxy(rent):
    direct = {"ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40123"}]},
              "public_ipaddr": " 203.0.113.9\n", "ssh_host": "ssh5.vast.ai", "ssh_port": 31234}
    assert rent.ssh_target(direct) == rent.Conn("203.0.113.9", 40123)
    proxy = {"ports": {}, "ssh_host": "ssh5.vast.ai", "ssh_port": 31234}
    assert rent.ssh_target(proxy) == rent.Conn("ssh5.vast.ai", 31234)
    assert rent.ssh_target({"ports": None, "ssh_host": None, "ssh_port": None}) is None
    assert rent.ssh_target({}) is None


def test_public_key_is_sent_without_its_comment(rent, tmp_path):
    private = tmp_path / "id_test"
    Path(str(private) + ".pub").write_text("ssh-ed25519 AAAAC3Nza+abc/def user@laptop\n", encoding="utf-8")
    assert rent.read_public_key(private) == "ssh-ed25519 AAAAC3Nza+abc/def"
    Path(str(private) + ".pub").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n", encoding="utf-8")
    with pytest.raises(rent.RentError):
        rent.read_public_key(private)
    with pytest.raises(rent.RentError) as error:
        rent.read_public_key(tmp_path / "missing")
    assert "ssh-keygen" in str(error.value)


def test_watchdog_probe_reading(rent):
    armed = "WATCHDOG_ALIVE\n2026-11-17T08:00:00Z watchdog[41] ARMED: instance 7 will be destroyed in 90 min"
    assert rent.watchdog_armed(armed)
    assert not rent.watchdog_armed("2026-11-17T08:00:00Z watchdog[41] ARMED: instance 7 will be destroyed")
    assert not rent.watchdog_armed("WATCHDOG_ALIVE\n2026-11-17T08:00:00Z watchdog[41] FATAL: no key")
    assert not rent.watchdog_armed("")


# ---------------------------------------------------------------------------------------------------------
# The on-start command and the two shell scripts
# ---------------------------------------------------------------------------------------------------------
def test_onstart_fits_the_documented_limit_with_the_real_watchdog(rent):
    script = WATCHDOG_PATH.read_text(encoding="utf-8")
    command = rent.build_onstart(script, rent.MAX_MINUTES_LIMIT)
    assert len(command) <= rent.ONSTART_LIMIT < 4048
    assert command.startswith("mkdir -p /workspace; touch /root/.no_auto_tmux; echo ")
    assert f"bash /root/watchdog.sh {rent.MAX_MINUTES_LIMIT} " in command
    assert "WATCHDOG_SKIP_PREFLIGHT" not in command
    skipping = rent.build_onstart(script, 5, skip_preflight=True)
    assert "WATCHDOG_SKIP_PREFLIGHT=1 $(command -v setsid) nohup bash /root/watchdog.sh 5 " in skipping


def test_onstart_refuses_bad_minutes_and_a_script_that_is_too_long(rent):
    script = WATCHDOG_PATH.read_text(encoding="utf-8")
    for minutes in (0, -5, rent.MAX_MINUTES_LIMIT + 1):
        with pytest.raises(rent.RentError):
            rent.build_onstart(script, minutes)
    # lines that gzip cannot shrink much: 400 different 32-character words
    import hashlib
    long_script = "#!/bin/bash\n" + "\n".join("echo " + hashlib.sha256(str(i).encode()).hexdigest()[:32]
                                               for i in range(400)) + "\n"
    with pytest.raises(rent.RentError) as error:
        rent.build_onstart(long_script, 60)
    assert "limit" in str(error.value)


def test_strip_comments_keeps_the_code(rent):
    script = "#!/usr/bin/env bash\n# a comment\n\nset -u\n    # indented comment\necho hi  # trailing stays\n"
    assert rent.strip_comments(script) == "#!/usr/bin/env bash\nset -u\necho hi  # trailing stays\n"


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")


@needs_bash
def test_stripped_watchdog_is_still_valid_shell(rent, tmp_path):
    stripped = tmp_path / "watchdog_stripped.sh"
    stripped.write_text(rent.strip_comments(WATCHDOG_PATH.read_text(encoding="utf-8")), encoding="utf-8")
    for path in (stripped, WATCHDOG_PATH, SETUP_PATH):
        done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr
    text = stripped.read_text(encoding="utf-8")
    for needed in ("ARMED:", "FIRING:", "api_call DELETE", '{"state": "stopped"}', "Authorization: Bearer"):
        assert needed in text


def test_shell_scripts_never_trace_their_commands():
    for path in (WATCHDOG_PATH, SETUP_PATH):
        code_lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                      if not line.strip().startswith("#")]
        tracing = re.compile(r"\bset\s+-[a-zA-Z]*x|\bbash\s+-[a-zA-Z]*x|xtrace")
        assert not any(tracing.search(line) for line in code_lines)


def test_watchdog_uses_the_calls_of_the_vastai_tool():
    text = WATCHDOG_PATH.read_text(encoding="utf-8")
    assert 'API="${WATCHDOG_API_URL:-https://console.vast.ai/api/v0}"' in text
    assert 'URL="$API/instances/$ID/"' in text
    assert "api_call DELETE \"$URL\" '{}'" in text
    assert "api_call PUT \"$URL\" '{\"state\": \"stopped\"}'" in text
    assert '-H "Authorization: Bearer $KEY"' in text


def run_watchdog(tmp_path, minutes: str, **variables: str) -> tuple[int, str]:
    """Run watchdog.sh with a start-up error. Its API address points at a closed local port, so that even
    a watchdog that started by mistake could not destroy anything."""
    env = {"PATH": os.environ["PATH"], "WATCHDOG_DIR": str(tmp_path),
           "WATCHDOG_API_URL": "http://127.0.0.1:9/api/v0", **variables}
    done = subprocess.run(["bash", str(WATCHDOG_PATH), minutes], env=env, capture_output=True, text=True,
                          timeout=30, check=False)
    log_file = tmp_path / "watchdog.log"
    return done.returncode, log_file.read_text(encoding="utf-8") if log_file.exists() else ""


def on_a_vast_machine() -> bool:
    """True if this test runs inside a Vast.ai container, where the watchdog can find a real key."""
    if "CONTAINER_API_KEY" in os.environ or "CONTAINER_ID" in os.environ:
        return True
    for path in ("/proc/1/environ", "/etc/environment"):
        try:
            if b"CONTAINER_API_KEY" in Path(path).read_bytes():
                return True
        except OSError:
            continue
    return False


@needs_bash
@pytest.mark.parametrize("minutes", ["", "0", "abc", "12x", "-5", "1.5", "123456"])
def test_watchdog_refuses_bad_minutes_at_the_start(tmp_path, minutes):
    code, log = run_watchdog(tmp_path, minutes, CONTAINER_ID="123", CONTAINER_API_KEY=FAKE_KEY)
    assert code == 2
    assert "FATAL: usage: watchdog.sh <minutes>" in log and "ARMED" not in log
    assert FAKE_KEY not in log


@needs_bash
@pytest.mark.skipif(on_a_vast_machine(), reason="inside a Vast.ai container the watchdog finds real values")
@pytest.mark.parametrize("variables, expected", [
    ({"CONTAINER_ID": "123", "CONTAINER_API_KEY": ""}, "CONTAINER_API_KEY is missing"),
    ({"CONTAINER_ID": "123"}, "CONTAINER_API_KEY is missing"),
    ({"CONTAINER_ID": "123", "CONTAINER_API_KEY": "short"}, "CONTAINER_API_KEY does not look like a key"),
    ({"CONTAINER_ID": "123", "CONTAINER_API_KEY": "has a space in the key value"},
     "CONTAINER_API_KEY does not look like a key"),
    ({"CONTAINER_ID": "", "CONTAINER_API_KEY": FAKE_KEY}, "CONTAINER_ID is missing"),
    ({"CONTAINER_ID": "12a", "CONTAINER_API_KEY": FAKE_KEY}, "CONTAINER_ID is missing or is not a number"),
    ({"VAST_CONTAINERLABEL": "C.x", "CONTAINER_API_KEY": FAKE_KEY},
     "CONTAINER_ID is missing or is not a number"),
])
def test_watchdog_refuses_wrong_or_empty_variables_at_the_start(tmp_path, variables, expected):
    code, log = run_watchdog(tmp_path, "5", **variables)
    assert code == 2
    assert f"FATAL: {expected}" in log and "ARMED" not in log
    assert FAKE_KEY not in log and "has a space" not in log
