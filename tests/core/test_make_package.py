"""The package command (scripts/make_package.sh): the zip holds the committed files only, a dirty tree is
refused, the scan finds each kind of finding, a missing words file stops the build unless --no-words is given,
and the accepted findings are known by their matched text, not by their line number.

Every test works in a tiny made-up git repository in a temporary folder; the repository itself is never
archived here. The strings that the scan must find (key headers, e-mail addresses, paths under the home
folder) are put together at run time, so that this file itself holds none of them.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_package.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None,
                                reason="needs git and bash")

AT = "@"
HOME = "/" + "home" + "/"
DASHES = "-" * 5


def git_env() -> dict:
    env = dict(os.environ, PYTHON=sys.executable, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_AUTHOR_NAME="Test", GIT_COMMITTER_NAME="Test")
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test" + AT + "example.invalid"
    return env


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, env=git_env(), check=True, capture_output=True,
                          text=True).stdout


def make_repo(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    repo = tmp_path / "tiny"
    repo.mkdir()
    git(repo, "init", "-q")
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    git(repo, "add", "-A", "-f")
    git(repo, "commit", "-q", "-m", "first")
    return repo


def package(repo: Path, *args: str, words: bool = False) -> subprocess.CompletedProcess:
    """Run the script in `repo`. Without `words` it gets --no-words (the tiny repositories have no words file)."""
    options = list(args) if words else ["--no-words", *args]
    return subprocess.run(["bash", str(SCRIPT), *options], cwd=repo, env=git_env(), capture_output=True,
                          text=True, timeout=120)


def add_words(repo: Path, *words: str) -> None:
    (repo / "_dev").mkdir(exist_ok=True)
    (repo / "_dev" / "private_words.txt").write_text("\n" + "\n".join(words) + "\n", encoding="utf-8")  # untracked


def the_zip(repo: Path) -> Path:
    zips = sorted((repo / "dist").glob("*.zip"))
    assert len(zips) == 1, zips
    return zips[0]


CLEAN = {"README.md": "# tiny\n", "src/a.py": "x = 1\n", "data/b.bin": b"\x00\x01\x02 binary"}


def test_the_zip_holds_the_committed_files_only(tmp_path):
    repo = make_repo(tmp_path, CLEAN)
    (repo / "untracked.txt").write_text("not committed\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")     # untracked too
    (repo / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    result = package(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    short = git(repo, "rev-parse", "--short=7", "HEAD").strip()
    zip_path = the_zip(repo)
    assert zip_path.name == f"tiny_{short}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(name for name in zf.namelist() if not name.endswith("/"))
        assert names == sorted(f"tiny_{short}/{rel}" for rel in CLEAN)
        assert zf.read(f"tiny_{short}/src/a.py") == b"x = 1\n"
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert (repo / "dist" / (zip_path.name + ".sha256")).read_text() == f"{digest}  {zip_path.name}\n"
    summary = (repo / "dist" / (zip_path.name + ".txt")).read_text()
    assert "files: 3\n" in summary and f"unpacked bytes: {6 + 6 + 11}\n" in summary
    assert "the 20 largest files" in summary and "0 finding(s), 0 accepted" in result.stdout


def test_a_dirty_tree_is_refused_and_allow_dirty_still_packs_the_commit(tmp_path):
    repo = make_repo(tmp_path, CLEAN)
    (repo / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    result = package(repo)
    assert result.returncode == 2
    assert "refused" in result.stderr and "src/a.py" in result.stderr
    assert not (repo / "dist").exists() or not list((repo / "dist").glob("*.zip"))
    # A staged change is refused as well.
    git(repo, "add", "src/a.py")
    assert package(repo).returncode == 2
    result = package(repo, "--allow-dirty")
    assert result.returncode == 0, result.stdout + result.stderr
    zip_path = the_zip(repo)
    assert zip_path.name.endswith("_dirty.zip") and "src/a.py" in result.stdout
    with zipfile.ZipFile(zip_path) as zf:
        assert [zf.read(n) for n in zf.namelist() if n.endswith("src/a.py")] == [b"x = 1\n"]


FINDINGS = [
    ("_dev/notes.txt", "hello\n", "path in a folder that is never packaged: _dev/notes.txt"),
    ("_sealed/x.json", "{}\n", "path in a folder that is never packaged: _sealed/x.json"),
    (".venv/lib/x.py", "\n", "path in a folder that is never packaged: .venv/lib/x.py"),
    (".cache/hf/x.json", "{}\n", "path in a folder that is never packaged: .cache/hf/x.json"),
    ("dist/old.txt", "\n", "path in a folder that is never packaged: dist/old.txt"),
    ("conf/box.env", "A=1\n", "file named *.env: conf/box.env"),
    ("keys/id.txt", "a\n" + DASHES + "BEGIN OPENSSH PRIVATE KEY" + DASHES + "\nabc\n",
     "private key header: keys/id.txt line 2"),
    ("run.sh", "#!/bin/sh\nexport VAST_API_" + "KEY=abcd1234efgh5678\n", "VAST_API_KEY with a value: run.sh line 2"),
    ("doc.md", "line one\nwrite to some.one" + AT + "example.org please\n",
     "e-mail address some.one" + AT + "example.org: doc.md line 2"),
    ("log.txt", "a\nb\nsaved in " + HOME + "someone/project/x\n", "absolute path under /home/: log.txt line 3"),
    ("cfg.txt", "x\ntoken = hf" + "_" + "Ab3" * 8 + "\n", "Hugging Face token: cfg.txt line 2"),
]


@pytest.mark.parametrize("rel, content, expected", FINDINGS, ids=[f[0] for f in FINDINGS])
def test_each_kind_of_finding_is_found(tmp_path, rel, content, expected):
    repo = make_repo(tmp_path, {**CLEAN, rel: content})
    result = package(repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FINDING: " + expected in result.stdout.splitlines()
    assert "1 finding(s)" in result.stdout
    assert the_zip(repo).exists()                     # the zip is kept, to be looked at


def test_every_e_mail_address_is_reported(tmp_path):
    line = "a" + AT + "b.com and c" + AT + "d.org\n"
    repo = make_repo(tmp_path, {**CLEAN, "x.txt": "\n" + line})
    result = package(repo)
    assert result.returncode == 1
    found = [x for x in result.stdout.splitlines() if x.startswith("FINDING: e-mail")]
    assert found == ["FINDING: e-mail address a" + AT + "b.com: x.txt line 2",
                     "FINDING: e-mail address c" + AT + "d.org: x.txt line 2"]


def test_a_private_word_is_found_case_ignored_and_never_printed(tmp_path):
    word = "Zorblatt"
    repo = make_repo(tmp_path, {**CLEAN, "notes/a.md": "one\ntwo\nthree " + word.upper() + " four\n",
                                "notes/b.md": "zorblatty is another word\n"})
    add_words(repo, word.lower())
    result = package(repo, words=True)
    assert result.returncode == 1, result.stdout + result.stderr
    found = [x for x in result.stdout.splitlines() if x.startswith("FINDING")]
    assert found == ["FINDING: a word of _dev/private_words.txt: notes/a.md line 3"]
    assert word.lower() not in (result.stdout + result.stderr).lower()


def test_harmless_lines_are_not_findings(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN,
                                "a.sh": 'VAST_API_KEY="$KEY"\nVAST_API_' + 'KEY=<your key>\ncd ~/home/x\n'
                                        + "hf" + "_short and hf" + "_" + "x" * 19 + "\n",
                                "b.bin": b"\x00" + ("x" + AT + "y.com " + HOME).encode()})   # binary: not read
    result = package(repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_missing_words_file_stops_the_build_unless_no_words_is_given(tmp_path):
    repo = make_repo(tmp_path, CLEAN)
    result = package(repo, words=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "private_words.txt is missing" in result.stdout and "--no-words" in result.stdout
    assert "0 finding(s)" in result.stdout
    result = package(repo)                                 # with --no-words
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no words file, not scanned (--no-words)" in result.stdout
    add_words(repo, "zorblatt")
    result = package(repo, words=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "private words: 1 from _dev/private_words.txt" in result.stdout


def test_an_accepted_finding_is_known_by_its_text_not_by_its_line(tmp_path):
    # tests/core/test_rent_script.py with "someone at example.com" is in the ACCEPTED list of the script.
    # Here it stands on line 3, not on the line of the real file: it is still accepted.
    accepted = "someone" + AT + "example.com"
    repo = make_repo(tmp_path, {**CLEAN, "tests/core/test_rent_script.py": "a\nb\nmail = '" + accepted + "'\n"})
    result = package(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 finding(s), 1 accepted" in result.stdout
    assert "accepted: e-mail address " + accepted + ": tests/core/test_rent_script.py line 3" in result.stdout
    # Another address in the same file, and the same address in another file, are findings.
    other = "someone.else" + AT + "example.com"
    (tmp_path / "two").mkdir()
    repo2 = make_repo(tmp_path / "two", {**CLEAN, "tests/core/test_rent_script.py": accepted + " " + other + "\n",
                                         "notes.md": accepted + "\n"})
    result = package(repo2)
    assert result.returncode == 1
    found = [x for x in result.stdout.splitlines() if x.startswith("FINDING")]
    assert sorted(found) == sorted(["FINDING: e-mail address " + other + ": tests/core/test_rent_script.py line 1",
                                    "FINDING: e-mail address " + accepted + ": notes.md line 1"])


def test_a_private_word_is_never_accepted(tmp_path):
    # The line holds an accepted e-mail address; its first part is also a private word. The address stays
    # accepted, and the word is a finding.
    accepted = "someone" + AT + "example.com"
    repo = make_repo(tmp_path, {**CLEAN, "tests/core/test_rent_script.py": "x = '" + accepted + "'\n"})
    add_words(repo, "someone")
    result = package(repo, words=True)
    assert result.returncode == 1, result.stdout + result.stderr
    found = [x for x in result.stdout.splitlines() if x.startswith("FINDING")]
    assert found == ["FINDING: a word of _dev/private_words.txt: tests/core/test_rent_script.py line 1"]
    assert "1 finding(s), 1 accepted" in result.stdout
