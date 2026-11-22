"""Play one scenario by hand in the terminal: you write the assistant messages, the real harness answers.

    .venv/bin/python scripts/play.py --scenarios data/scenarios_placeholder --family ca-03 --style 2

The script prints the prompt exactly as the model gets it (the system turn with the tool list, the request,
the start of the assistant turn). Then, for each turn, you type the assistant message and end it with an empty
line. The message goes through the real parser and the real tools, and the tool result is printed. At the end
you see the final state, the expected state, pass or fail, and the reward.

Short form: one line `tool_name key=value key2="value two"` is expanded to the full tool call (and shown)
before it is sent. Anything else is sent exactly as typed, so you can also try out malformed messages.

Without --text the placeholder text is used; the text file must be the one that the scenarios were built
from. Families of the test split are refused unless --allow-test is given: looking at test scenarios before
the single test run is a choice that the write-up would have to report.
With --out the episode record is added to that file (run "hand_play").
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accessdesk.config import RunConfig  # noqa: E402
from accessdesk.env.generator import SPLITS  # noqa: E402
from accessdesk.env.rules import RULESETS  # noqa: E402
from accessdesk.env.scenarios import load_families  # noqa: E402
from accessdesk.harness.episode import PROMPTS, Episode  # noqa: E402
from accessdesk.harness.records import read_records, write_records  # noqa: E402
from accessdesk.harness.render import render_tool_call  # noqa: E402
from accessdesk.text import TOOLSETS, TextBundle, load_text, placeholder_path  # noqa: E402

DEFAULT_TOKENIZER = ROOT / ".cache" / "hf" / "Qwen3.5-4B"
STATE_PARTS = ("access", "approvals", "grants", "request")
LINE = "=" * 78


def choose_family(scenarios_dir: Path, family_id: str, text: TextBundle, allow_test: bool) -> dict:
    """The family to play, from whichever split file of the folder holds it.

    Raises ValueError with a plain message if the folder does not belong to this text file, if it has no such
    family, or if the family is in the test split and `allow_test` is False.
    """
    summary_path = scenarios_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"{summary_path} is missing: build the scenarios with scripts/build_scenarios.py")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["text_sha256"] != text.sha256:
        raise ValueError(f"the scenarios in {scenarios_dir} were built from another text file "
                         f"({summary['text_file']}), not from {text.path}")
    for split in SPLITS:
        path = scenarios_dir / f"{split}.jsonl"
        for family in load_families(path) if path.is_file() else []:
            if family["family_id"] != family_id:
                continue
            if split == "test" and not allow_test:
                raise ValueError(f"{family_id} is a test family. The test set is looked at once, at the "
                                 "end. Use a calibration or dev family, or pass --allow-test and say so in "
                                 "the notes.")
            return family
    raise ValueError(f"no family {family_id!r} in {scenarios_dir}")


def prompt_text(episode: Episode, tokenizer_dir: Path) -> tuple[str, bool]:
    """(the first prompt, True if it is the exact text of the chat template).

    With the tokenizer files the prompt is built by the harness's own token layout and decoded again: this is
    character for character what the model reads. Without them the parts are listed one after the other.
    """
    if tokenizer_dir.is_dir():
        from transformers import AutoTokenizer

        from accessdesk.harness.tokens import TokenLayout
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        layout = TokenLayout(tokenizer, episode.tool_specs)
        ids = layout.first_prompt(episode.system_prompt, episode.request_text)
        return tokenizer.decode(ids, skip_special_tokens=False), True
    parts = ["[system prompt]", episode.system_prompt, "", "[tools]"]
    parts += [json.dumps(spec.schema(), ensure_ascii=False) for spec in episode.tool_specs]
    parts += ["", "[request]", episode.request_text]
    return "\n".join(parts), False


def read_message(stdin: TextIO) -> str | None:
    """The lines up to the next empty line, joined. Empty lines before the message are skipped.

    None if the input ends before any line of a message was read.
    """
    lines: list[str] = []
    while True:
        line = stdin.readline()
        if line == "":                                   # end of input
            return "\n".join(lines) if lines else None
        line = line.rstrip("\r\n")
        if line.strip() == "":
            if lines:
                return "\n".join(lines)
            continue
        lines.append(line)


def expand_short_form(message: str, episode: Episode) -> tuple[str, str | None]:
    """(the message to send, a complaint or None).

    A one-line message that starts with the visible name of a tool is the short form
    `tool_name key=value key2="value two"`. It becomes the full tool call. The keys are written as typed, so
    a wrong key reaches the parser like any other mistake. Any other message is returned unchanged.
    """
    words = message.split(None, 1)
    names = [spec.name for spec in episode.tool_specs]
    if "\n" in message or not words or words[0] not in names:
        return message, None
    try:
        pairs = shlex.split(words[1]) if len(words) > 1 else []
    except ValueError as error:
        return message, f"the short form could not be read ({error})"
    if any("=" not in pair for pair in pairs):
        return message, "the short form is: tool_name key=value key2=\"value two\""
    args = dict(pair.split("=", 1) for pair in pairs)
    if len(args) != len(pairs):
        return message, "a key is written twice in the short form"
    return render_tool_call(words[0], args), None


def show_state(title: str, state: dict, access_at_start: list, out: TextIO) -> None:
    """Print a state dict. The access list is long, so only the pairs added since the start are shown."""
    added = [pair for pair in state["access"] if pair not in access_at_start]
    removed = [pair for pair in access_at_start if pair not in state["access"]]
    print(title, file=out)
    print(f"  {'access':9s} the {len(access_at_start)} pairs of the start, plus {json.dumps(added)}"
          + (f", minus {json.dumps(removed)}" if removed else ""), file=out)
    for part in STATE_PARTS[1:]:
        print(f"  {part:9s} {json.dumps(state[part], ensure_ascii=False)}", file=out)


def play(episode: Episode, stdin: TextIO, out: TextIO) -> bool:
    """Play the episode with messages read from `stdin`. False if the input ended before the episode did."""
    limit = episode.cfg.turn_limit
    while not episode.done:
        print(f"\n--- Turn {len(episode.turns) + 1} of {limit}: type the assistant message, "
              "then an empty line ---", file=out)
        message = read_message(stdin)
        if message is None:
            return False
        message, complaint = expand_short_form(message, episode)
        if complaint:
            print(f"Not sent: {complaint}. This did not use a turn.", file=out)
            continue
        turn = episode.step(message)
        print("Sent to the harness:", file=out)
        print(message, file=out)
        if turn["parse_ok"]:
            verdict = f"one valid call to the tool {turn['tool_key']!r} (internal name)"
            if turn["tool_error"]:
                verdict += f"; the tool answered with the error {turn['tool_error']!r}"
        else:
            verdict = f"NOT a valid tool call ({turn['parse_error']}); this turn counts as malformed"
        print(f"Parser: {verdict}", file=out)
        print("Tool result (what the model reads next):", file=out)
        print(turn["tool_response"], file=out)
    return True


def show_result(episode: Episode, record: dict, out: TextIO) -> None:
    expected = episode.family["expected"][episode.ruleset]
    ended = "the request was closed" if record["closed"] else "the turn limit was reached"
    print(f"\n{LINE}\nThe episode is over: {ended}.\n{LINE}", file=out)
    print(f"Turns: {record['n_turns']} (budget without a penalty: {record['turn_budget']}). "
          f"Malformed messages: {record['n_malformed']}. Tool errors: {record['n_tool_errors']}.", file=out)
    access_at_start = [list(pair) for pair in episode.family["world"]["access"]]
    show_state("Final state:", record["final_state"], access_at_start, out)
    show_state("Expected state:", expected, access_at_start, out)
    different = [part for part in STATE_PARTS if record["final_state"][part] != expected[part]]
    print(f"Parts that differ: {', '.join(different) if different else 'none'}", file=out)
    print(f"Result: {'PASS' if record['passed'] else 'FAIL'}, reward {record['reward']} of "
          f"{episode.cfg.pass_reward}   (verify: {record['impl']['verify']} version, "
          f"reward: {record['impl']['reward']} version)", file=out)


def main(argv: list[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin, out = stdin or sys.stdin, stdout or sys.stdout
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scenarios", type=Path, required=True, help="the scenario folder")
    parser.add_argument("--family", required=True, help="a family id, for example ca-03")
    parser.add_argument("--style", type=int, default=1, help="the wording style (1 to 4)")
    parser.add_argument("--text", type=Path, default=placeholder_path(), help="the text file (YAML)")
    parser.add_argument("--ruleset", choices=RULESETS, default=None,
                        help="default: the default rules; with --prompt policy_shift, that prompt's rules")
    parser.add_argument("--toolset", choices=TOOLSETS, default="five")
    parser.add_argument("--prompt", choices=PROMPTS, default="final")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="the tokenizer folder")
    parser.add_argument("--allow-test", action="store_true", help="allow a family of the test split")
    parser.add_argument("--out", type=Path, default=None, help="add the episode record to this file")
    args = parser.parse_args(argv)

    try:
        text = load_text(args.text)             # an incomplete text file is a ValueError with every problem
        ruleset = args.ruleset or (text.policy_shift_variant if args.prompt == "policy_shift" else "default")
        family = choose_family(args.scenarios, args.family, text, args.allow_test)
        episode = Episode(family, args.style, text, RunConfig(), ruleset=ruleset, toolset=args.toolset,
                          prompt=args.prompt)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    prompt, exact = prompt_text(episode, args.tokenizer)
    print(f"Family {family['family_id']} ({family['split']}), style {args.style}, rule set {ruleset}, "
          f"{len(episode.tool_specs)} tools, text file {Path(text.path).name}", file=out)
    note = "exactly as the model gets it" if exact else \
        f"NOT in the chat format: no tokenizer files in {args.tokenizer}"
    print(f"{LINE}\nThe prompt ({note})\n{LINE}", file=out)
    print(prompt, file=out)
    print(f"{LINE}\nEnd of the prompt\n{LINE}", file=out)
    example = episode.tool_specs[0]
    print(f"Short form, on one line: {example.name} {example.params[0].name}=\"...\"", file=out)

    if not play(episode, stdin, out):
        print("\nThe input ended before the episode was over. Nothing is saved.", file=out)
        return 1
    earlier = read_records(args.out) if args.out and args.out.is_file() else []
    sample = sum(1 for r in earlier if r["scenario_id"] == f"{family['family_id']}/s{args.style}")
    record = episode.record(run="hand_play", arm="human", sample=sample)
    show_result(episode, record, out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_records(args.out, earlier + [record])
        print(f"Added to {args.out} ({len(earlier) + 1} episodes in the file).", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
