#!/usr/bin/env bash
# Build the zip of the package from the committed files only, and scan it. Written on 2026-11-19, after the
# freeze; it makes no number of a result and is in no frozen group (tests/core/test_freeze.py, IN_NO_GROUP).
#
#     bash scripts/make_package.sh                  # from the top folder of the repository
#     bash scripts/make_package.sh --allow-dirty    # also when a tracked file has uncommitted changes
#     bash scripts/make_package.sh --no-words       # also when _dev/private_words.txt is missing (see 4c)
#     bash scripts/make_package.sh --name NAME      # the name of the zip and of its top folder (default: the
#                                                   # name of the top folder of the repository)
#
# What it does, in order:
#   1. It refuses to run if a tracked file has uncommitted changes (staged or not), and lists those files.
#      With --allow-dirty it goes on; the zip name then ends in _dirty, and the zip still holds the committed
#      state (HEAD), not the changes. Untracked files never enter the zip.
#   2. git archive of HEAD into dist/<name>_<short commit>.zip (or _dirty.zip), with one top folder
#      <name>_<short commit>/.
#   3. Beside the zip: <zip>.sha256 (sha256sum format) and <zip>.txt (the commit, the number of files, the
#      unpacked size in bytes, and the 20 largest files).
#   4. It scans every file of the archive (not the working tree):
#      (a) no path with a folder _dev, _sealed, .venv, .cache or dist, and no file whose name ends in .env;
#      (b) in each text file: no line with a private key header (-----BEGIN ... PRIVATE KEY-----), no line
#          VAST_API_KEY=<value> (a value of 8 or more letters, digits, - or _; VAST_API_KEY=$VAR does not
#          count), no Hugging Face token (hf_ and 20 or more letters or digits), no e-mail address (each one
#          is reported with its file and line number), no absolute path that starts with /home/ and a name;
#      (c) no word of _dev/private_words.txt (one word or phrase per line, case ignored, whole words) in any
#          text file of the archive. A match is reported by file and line number only: the word is never
#          printed. The words file is not in the package. If it is missing, the script stops with exit code 1
#          after the scan of (a) and (b), unless --no-words is given (then it says that the words were not
#          scanned).
#      A text file is a file with no NUL byte in its first 8,192 bytes. The scan matches whole words in text
#      files only: it does not read binary files, and it does not find other forms of a word (a plural, two
#      words with two spaces between them).
#   A few findings in the committed files are known and harmless: they are listed in ACCEPTED below (kind,
#   file, the matched text, reason), printed as "accepted", and not counted. The key holds the matched text,
#   not the line number, so an edit above such a line does not break the build. A private word is never
#   accepted. Any other finding makes the exit code 1 (the zip is kept, so that it can be looked at). Exit
#   code 2: the tree was refused or a command failed.
#
# The environment variable PYTHON names the interpreter of the scan (default .venv/bin/python, else python3);
# the scan uses the standard library only.
set -euo pipefail

allow_dirty=0
no_words=0
name=""
while [ $# -gt 0 ]; do
    case "$1" in
        --allow-dirty) allow_dirty=1; shift ;;
        --no-words) no_words=1; shift ;;
        --name) name="${2:?--name needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "make_package: unknown option: $1" >&2; exit 2 ;;
    esac
done

top="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "make_package: not inside a git repository" >&2; exit 2; }
cd "$top"
git rev-parse --verify -q HEAD >/dev/null || { echo "make_package: the repository has no commit" >&2; exit 2; }
[ -n "$name" ] || name="$(basename "$top")"
case "$name" in */*|"") echo "make_package: --name must be a plain name" >&2; exit 2 ;; esac

if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

# 1. Uncommitted changes of tracked files.
dirty="$(git status --porcelain --untracked-files=no)"
suffix=""
if [ -n "$dirty" ]; then
    if [ "$allow_dirty" -ne 1 ]; then
        echo "make_package: refused: tracked files have uncommitted changes (commit them, or give --allow-dirty):" >&2
        printf '%s\n' "$dirty" | sed 's/^/    /' >&2
        exit 2
    fi
    echo "make_package: --allow-dirty: these uncommitted changes are NOT in the zip (it holds HEAD):"
    printf '%s\n' "$dirty" | sed 's/^/    /'
    suffix="_dirty"
fi

# 2. The archive.
short="$(git rev-parse --short=7 HEAD)"
commit="$(git rev-parse HEAD)"
stem="${name}_${short}"
mkdir -p dist
zip="dist/${stem}${suffix}.zip"
rm -f "$zip" "$zip.sha256" "$zip.txt"
git archive --format=zip --prefix="${stem}/" -o "$zip" HEAD
(cd dist && sha256sum "$(basename "$zip")" > "$(basename "$zip").sha256")
tracked="$(git ls-tree -r --name-only HEAD | wc -l)"

# 3 and 4. The summary beside the zip, and the scan.
"$PYTHON" - "$zip" "$stem/" "$commit" "$tracked" "$no_words" <<'PY'
import os, re, sys, zipfile

zip_path, prefix, commit, tracked = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
no_words = sys.argv[5] == "1"
zf = zipfile.ZipFile(zip_path)
files = [i for i in zf.infolist() if not i.is_dir()]
size = sum(i.file_size for i in files)
largest = sorted(files, key=lambda i: (-i.file_size, i.filename))[:20]
with open(zip_path + ".txt", "w", encoding="utf-8") as fh:
    fh.write("zip: %s\ncommit: %s\nfiles: %d\nunpacked bytes: %d\nzip bytes: %d\n\n"
             "the 20 largest files (bytes, path):\n"
             % (os.path.basename(zip_path), commit, len(files), size, os.path.getsize(zip_path)))
    for i in largest:
        fh.write("%14d  %s\n" % (i.file_size, i.filename))
print("make_package: %s: %d files (git ls-tree of HEAD: %d), %d bytes unpacked, %d bytes zipped"
      % (zip_path, len(files), tracked, size, os.path.getsize(zip_path)))

# Known findings of the committed files, each read by hand on 2026-11-19 (kind, file, matched text): reason.
# The matched texts are put together from parts, so that this file itself holds none of them.
AT, DASHES, CALIB = "@", "-" * 5, "results/calibration/round4b/2b_plain_prompt/episodes.jsonl"
ACCEPTED = {
    ("e-mail", CALIB, "mariewoods" + AT + "example.com"):
        "an address that the untrained model made up in a calibration attempt (example.com)",
    ("e-mail", CALIB, "nmanaging_marketing_campaign_setup" + AT + "promenade.com"):
        "a role name that the untrained model wrote as an address in a calibration attempt",
    ("e-mail", CALIB, "managing_marketing_campaign_setup" + AT + "promenade.com"):
        "a role name that the untrained model wrote as an address in a calibration attempt",
    ("home", "tests/core/test_rebuild_results.py", "/" + "home/x/repo/_dev/report_code/fresh"):
        "a made-up path in a test of the path rule",
    ("vast", "tests/core/test_rent_script.py", "VAST_API_" + "KEY=abc123XYZ"):
        "a made-up key in a test that rent.py masks keys",
    ("vast", "tests/core/test_rent_script.py", "VAST_API_" + 'KEY="abcdefgh'):
        "a made-up key in a test that rent.py refuses bad key files",
    ("e-mail", "tests/core/test_rent_script.py", "someone" + AT + "example.com"):
        "a made-up account in a test (example.com)",
    ("key", "tests/core/test_rent_script.py", DASHES + "BEGIN OPENSSH PRIVATE KEY" + DASHES):
        "a made-up key file in a test of rent.py",
}
findings, accepted = [], []


def found(kind, rel, matched, text):
    """One finding: `matched` is the text that the rule matched. A private word is never accepted."""
    if kind != "word" and (kind, rel, matched) in ACCEPTED:
        accepted.append("%s (%s)" % (text, ACCEPTED[(kind, rel, matched)]))
    else:
        findings.append(text)


BAD_DIRS = {"_dev", "_sealed", ".venv", ".cache", "dist"}
key_header = re.compile(r"-{5}BEGIN [A-Z0-9 ]*PRIVATE KEY-{5}")
vast_key = re.compile(r"VAST_API_KEY\s*=\s*[\"']?[A-Za-z0-9_\-]{8,}")
hf_token = re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}")
email = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
home = re.compile(r"(?<![\w.~\-])/home/[A-Za-z0-9_][^\s\"'<>]*")
words = []
words_file = os.path.join("_dev", "private_words.txt")
has_words_file = os.path.isfile(words_file)
if has_words_file:
    with open(words_file, encoding="utf-8") as fh:
        words = [line.strip() for line in fh if line.strip()]
private = (re.compile(r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(re.escape(w) for w in words), re.IGNORECASE)
           if words else None)

n_text = 0
for info in files:
    rel = info.filename[len(prefix):] if info.filename.startswith(prefix) else info.filename
    parts = rel.split("/")
    if set(parts[:-1]) & BAD_DIRS:
        findings.append("path in a folder that is never packaged: %s" % rel)
    if parts[-1].endswith(".env"):
        findings.append("file named *.env: %s" % rel)
    with zf.open(info) as fh:
        head = fh.read(8192)
        if b"\x00" in head:
            continue
        n_text += 1
        data = head + fh.read()
    for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        for m in key_header.finditer(line):
            found("key", rel, m.group(0), "private key header: %s line %d" % (rel, number))
        for m in vast_key.finditer(line):
            found("vast", rel, m.group(0), "VAST_API_KEY with a value: %s line %d" % (rel, number))
        for m in hf_token.finditer(line):
            found("hf", rel, m.group(0), "Hugging Face token: %s line %d" % (rel, number))
        for m in email.finditer(line):
            found("e-mail", rel, m.group(0), "e-mail address %s: %s line %d" % (m.group(0), rel, number))
        for m in home.finditer(line):
            found("home", rel, m.group(0), "absolute path under /home/: %s line %d" % (rel, number))
        if private is not None and private.search(line):
            found("word", rel, None, "a word of _dev/private_words.txt: %s line %d" % (rel, number))

if words:
    words_note = "%d from %s" % (len(words), words_file)
elif has_words_file:
    words_note = "%s holds no word, not scanned" % words_file
elif no_words:
    words_note = "no words file, not scanned (--no-words)"
else:
    words_note = "no words file, not scanned"
print("make_package: scanned %d files (%d text files); private words: %s" % (len(files), n_text, words_note))
for item in accepted:
    print("accepted: " + item)
for finding in findings:
    print("FINDING: " + finding)
print("make_package: %d finding(s), %d accepted" % (len(findings), len(accepted)))
if not has_words_file and not no_words:
    print("make_package: %s is missing, so the private words were not scanned; give --no-words to build "
          "without that scan (a reader who builds the zip from the package has no words file)" % words_file)
    sys.exit(1)
sys.exit(1 if findings else 0)
PY
