"""Let an open-weight model write text/model_text.yaml, one field at a time. Runs on the rented GPU.

    python scripts/gpu/draft_text.py --model Qwen/Qwen3.5-9B \\
        --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --run-seed 1 \\
        --out text/model_text.yaml --log text/drafting_log.jsonl

Why: the brief of the exercise allows text from templates, from the candidate's own writing or from
open-weight models under permissive licenses. The text that the trained model reads is therefore written by an
open-weight model. The coding assistant (a closed API model) wrote only the instructions
(text/drafting_instructions.yaml) and this program. All the logic is in accessdesk/drafting.py; this script
adds the model and the files.

What happens: for every field of the text file, in the order of the instruction file, the model gets a system
message and a user message (the desk, the field's instruction, the earlier values that the instruction uses).
Up to --candidates replies are asked for, each with its own seed. A reply is cleaned and checked by fixed
rules, and the first valid one is taken. Nobody chooses between candidates. Every candidate goes into the log.
The text file is written only if every field has a value and `accessdesk.text.check_text` accepts the whole
file.

Three things are reported so that nothing about the writing stays hidden. They are printed at the end of a
run, and the first and the third also stand in the header of the text file:
    1. the longest run of words that a taken value shares with the text shown to its writer (our
       instructions). A candidate with 8 or more such words was refused (the copy rule of drafting.py);
    2. the longest run of words that a taken value shares with text/placeholder_text.yaml, which the writing
       model never sees. This is a report, not a rule. It is also in the last line of the log;
    3. for each field, in how many runs of the log a value for it was drawn. A field that was written again
       (--only) has more than 1.

    --dry-run     needs no model and no torch: prints every field with the exact messages, in order. Earlier
                  values are stand-ins of the form <<field>>.
    --only F ...  write only these fields and every field that depends on one of them; the other fields keep
                  the value that the log holds from the last run. The log must exist, and its values must be
                  from the same model and revision. The header of the text file then names the earlier runs
                  and the SHA-256 of their instruction files, lists the fields that this run wrote, and says
                  whether the kept values pass the checks as they are now (they are not checked again when
                  the file is written, so a kept value that fails is listed and stays; write it again
                  with --only if it should change).
    --resume      go on after a run that stopped: a candidate whose writer, seed and messages are already in
                  the log is not asked for again.

The log is only added to. Without --resume or --only, the script refuses to start if the log already exists.
Exit code 0: the text file was written (or the dry run was printed). 1: it was not written. 2: bad arguments.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accessdesk import drafting  # noqa: E402

# The model that writes the text: open weights, Apache-2.0 by its Hub listing. The revision is pinned.
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

# Sampling for WRITING the text. (It is not the sampling of the experiment, which is in accessdesk/config.py.)
# Source: the settings that the Qwen3.5 model card gives for non-thinking mode, as recorded in
# research_notes/Small GRPO take home build plan/tooling_and_model_mechanics.md, section 4, "Inferences":
# temperature 0.7, top_p 0.8, top_k 20. The card's fourth setting, presence_penalty 1.5, is an option of an
# inference server; `generate` of transformers has no such setting, so it is not used here.
SAMPLING = {"enable_thinking": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
SAMPLING_TEXT = ("thinking off, temperature 0.7, top_p 0.8, top_k 20 (the model card's settings for "
                 "non-thinking mode)")

STOP_TOKEN = "<|im_end|>"      # the end of a turn in the chat format; its id is read from the tokenizer


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id on the Hub, or a local folder")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="model revision")
    parser.add_argument("--run-seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "text" / "model_text.yaml")
    parser.add_argument("--log", type=Path, default=ROOT / "text" / "drafting_log.jsonl")
    parser.add_argument("--candidates", type=int, default=8, help="candidates per field, at most")
    parser.add_argument("--only", nargs="+", default=None, metavar="FIELD", help="write only these fields")
    parser.add_argument("--resume", action="store_true", help="reuse the replies that the log already holds")
    parser.add_argument("--dry-run", action="store_true", help="print every prompt; no model is loaded")
    parser.add_argument("--instructions", type=Path, default=drafting.INSTRUCTIONS_PATH)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build_writer(model_id: str, revision: str, device: str = "cuda") -> Any:
    """The real writer: `(messages, seed, max_new_tokens) -> text`, one `generate` call per candidate.

    The model is loaded as `accessdesk.harness.hf_backend.load_model` does: the text-only class, in bfloat16.
    The prompt is the chat template's text with `enable_thinking=False`. The random state is set from the seed
    just before `generate`. The reply ends at `<|im_end|>`, whose id is read from the tokenizer; a reply that
    reaches the token limit first raises `drafting.ReplyCutOff`. The reply is decoded with special tokens
    kept, so that a stray tag of the chat format is seen by the checks and the candidate is refused.

    torch is imported only here. The function has a `settings` attribute for the log.
    """
    import torch
    from transformers import GenerationConfig

    from accessdesk.harness import hf_backend

    model, tokenizer = hf_backend.load_model(model_id, revision, device=device)
    core = hf_backend.causal_lm(model)
    stop_id, pad_id = tokenizer.convert_tokens_to_ids(STOP_TOKEN), tokenizer.pad_token_id
    if not isinstance(stop_id, int) or tokenizer.decode([stop_id]) != STOP_TOKEN or pad_id in (None, stop_id):
        raise RuntimeError(f"the tokenizer gives the stop token id {stop_id!r} and the pad id {pad_id!r}")

    def generation_config(max_new_tokens: int) -> Any:
        # Every setting is explicit: a value given here wins over the model's own generation config and over
        # the library defaults (transformers 5.23.0 generation/utils.py lines 2072-2112). The same list as in
        # hf_backend.Sampler, with the three sampling values of SAMPLING switched on.
        return GenerationConfig(
            do_sample=True, num_beams=1, num_return_sequences=1, max_new_tokens=max_new_tokens, min_length=0,
            eos_token_id=stop_id, pad_token_id=pad_id,
            temperature=SAMPLING["temperature"], top_k=SAMPLING["top_k"], top_p=SAMPLING["top_p"],
            typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0, repetition_penalty=1.0,
            encoder_repetition_penalty=1.0, no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
            renormalize_logits=False, remove_invalid_values=False,
            use_cache=True, return_dict_in_generate=False, output_scores=False, output_logits=False,
        )

    # What generate() will really use: our values, then the model's own generation config, then the library
    # defaults (generation/utils.py lines 2072-2137). A setting of the model's own config that would add a
    # filter beside temperature, top-k and top-p (lines 1612-1628) stops the run: the header of the text file
    # states the sampling, and it must be true.
    effective, _ = core._prepare_generation_config(generation_config(drafting.MAX_NEW_TOKENS["line"]))
    mode = str(effective.get_generation_mode().value)
    extra = [name for name in hf_backend.MUST_BE_UNSET if getattr(effective, name, None) is not None]
    if mode != "sample" or extra:
        raise RuntimeError(f"generate() would not do plain sampling: mode {mode!r}, extra settings {extra}")

    def writer(messages: list[dict], seed: int, max_new_tokens: int) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=SAMPLING["enable_thinking"])
        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        input_ids, attention_mask = hf_backend.pad_batch([ids], pad_id, side="left")
        hf_backend.set_sampling_seed(seed)
        with torch.no_grad():
            sequences = core.generate(input_ids=input_ids.to(core.device),
                                      attention_mask=attention_mask.to(core.device),
                                      generation_config=generation_config(max_new_tokens))
        new_tokens = sequences[0, input_ids.shape[1]:].tolist()
        if stop_id not in new_tokens:
            raise drafting.ReplyCutOff(tokenizer.decode(new_tokens, skip_special_tokens=False))
        return tokenizer.decode(new_tokens[:new_tokens.index(stop_id)], skip_special_tokens=False)

    example = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=SAMPLING["enable_thinking"])
    writer.settings = {"sampling": SAMPLING, "stop_id": stop_id, "pad_id": pad_id, "dtype": "bfloat16",
                       "prompt_ends_with": example[-40:],
                       "generate_effective_config": effective.to_dict(),
                       "environment": hf_backend.environment_report()}
    # Peak GPU memory since the start of the process, in GB (for the end of the run; it changes no value).
    writer.peak_memory_gb = lambda: round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return writer


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def shown_path(path: Path) -> str:
    """A path as the header comment names it: relative to the repository if it lies inside it."""
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if ROOT in resolved.parents else str(path)


def summary_line(line: dict) -> str | None:
    """What is printed for one log line while the run goes on (None: nothing)."""
    kind, field = line["kind"], line.get("field")
    if kind == "failed":
        return f"  FAILED    {field}: {line['reason']}"
    if kind != "candidate":
        return None
    reused = " (from the log)" if line["reused"] else ""
    if "items" in line:
        taken = sum(entry["taken"] for entry in line["items"])
        return f"  candidate {line['index']} of {field}: {taken} of {len(line['items'])} items taken{reused}"
    verdict = "taken" if line["taken"] else f"refused: {line['reason']}"
    return f"  candidate {line['index']} of {field}: {verdict}{reused}"


def end_report(report: dict, placeholder: dict | None) -> str:
    """What is printed at the end of a run: the two shared-run reports and the table of draws."""
    draws = report["draws"]
    width = max(len(key) for key in draws)
    lines = [
        f"longest run of words shared with the shown text, among the taken values (limit "
        f"{drafting.COPY_LIMIT}): {drafting.shown_run(report['largest_run'])}",
        f"longest run of words shared with the placeholder text file (a report, not a rule): "
        f"{drafting.shown_run(placeholder)}",
        "runs in which a value was drawn, for each field:",
        *[f"  {key.ljust(width)}  {count}" for key, count in draws.items()],
    ]
    again = [key for key, count in draws.items() if count > 1]
    lines.append(f"fields drawn in more than one run: {again if again else 'none'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, writer: Any = None) -> int:
    """Run the script. `writer` replaces the real model (used by the tests); None loads the model."""
    args = parse_args(argv)
    try:
        ins = drafting.load_instructions(args.instructions)
    except (OSError, drafting.InstructionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(drafting.dry_run_text(ins, args.run_seed), end="")
        return 0
    if args.candidates < 1:
        print("error: --candidates must be at least 1", file=sys.stderr)
        return 2
    old_lines = drafting.read_log(args.log)
    if old_lines and not (args.resume or args.only):
        print(f"error: {args.log} exists. Use --resume to go on with it, or move it away.", file=sys.stderr)
        return 2
    previous = drafting.previous_values(old_lines) if args.only else None
    try:
        todo = drafting.fields_to_write(ins, args.only)
        drafting.kept_values(ins, args.only, previous)        # stops here if the log lacks a value
    except drafting.DraftError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    asked = [key for key in todo if ins.fields[key].asked]
    writer_id = f"{args.model}@{args.revision}"
    not_written = [key for key in ins.fields if key not in todo]
    kept = drafting.kept_sources(old_lines, not_written) if args.only else None
    if kept is not None and kept["writers"] not in ([], [writer_id]):
        print(f"error: the values that --only would keep were written by {kept['writers']}, not by "
              f"{writer_id}. One text file has one writer: run without --only.", file=sys.stderr)
        return 2
    print(f"{len(asked)} fields to write, with up to {args.candidates} candidates each")
    if writer is None:
        writer = build_writer(args.model, args.revision, args.device)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    new_lines: list[dict] = []
    with args.log.open("a", encoding="utf-8") as log:
        def emit(line: dict) -> None:
            # default=str: a value in the writer's settings that JSON does not know must not stop the run
            log.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            log.flush()
            new_lines.append(line)
            message = summary_line(line)
            if message is not None:
                print(message, flush=True)

        emit({"kind": "run", "started_at_utc": utc_now(), "model": args.model, "revision": args.revision,
              "run_seed": args.run_seed, "candidates": args.candidates, "only": args.only,
              "resume": args.resume, "instructions": shown_path(args.instructions),
              "instructions_sha256": ins.sha256, "settings": getattr(writer, "settings", None)})
        result = drafting.draft_all(
            ins, writer, writer_id=writer_id, run_seed=args.run_seed,
            n_candidates=args.candidates, only=args.only, previous=previous,
            cache=drafting.reply_cache(old_lines) if args.resume else None, emit=emit)
        # The reports of the run: read from the whole log (the earlier runs and this one), and measured on
        # the values that were taken. They go into the last line of the log whatever the end of the run is.
        report = drafting.log_report(ins, old_lines + new_lines)
        placeholder = drafting.placeholder_report(ins, result.chosen)
        measured = {"shared_run": report["largest_run"], "placeholder_run": placeholder}
        print(end_report(report, placeholder))
        if hasattr(writer, "peak_memory_gb"):
            print(f"peak GPU memory of this process: {writer.peak_memory_gb()} GB")
        if not result.complete:
            print(f"NOT WRITTEN: {len(result.failed)} field(s) have no value: {sorted(result.failed)}")
            emit({"kind": "not_written", "out": shown_path(args.out), "failed": result.failed, **measured})
            return 1
        written_at = utc_now()
        refused_now = drafting.kept_problems(ins, result.chosen, not_written) if args.only else None
        if refused_now:
            print(f"kept values that the checks of today would refuse ({len(refused_now)}; they stay as "
                  "they are and are listed in the header):")
            print("\n".join(f"  {key}: {reason}" for key, reason in refused_now.items()))
        header = drafting.header_comment(
            ins, model_id=args.model, revision=args.revision, written_at_utc=written_at,
            run_seed=args.run_seed, n_candidates=args.candidates, sampling=SAMPLING_TEXT,
            instructions_file=shown_path(args.instructions), log_file=shown_path(args.log), report=report,
            kept=kept, written=asked if args.only else None, refused_now=refused_now)
        written_by = f"{args.model} (open-weight model, revision {args.revision})"
        content = header + "\n" + drafting.render_yaml(drafting.assemble(result.chosen, written_by))
        problems = drafting.write_text_file(args.out, content)
        if problems:
            print(f"NOT WRITTEN: the finished text has {len(problems)} problem(s):")
            print("\n".join(f"  - {problem}" for problem in problems))
            emit({"kind": "not_written", "out": shown_path(args.out), "problems": problems, **measured})
            return 1
        digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
        emit({"kind": "written", "out": shown_path(args.out), "sha256": digest, "written_at_utc": written_at,
              **measured})
    print(f"written: {args.out} (SHA-256 {digest})")
    print("Next: read the file, change any line you want, and put your name after 'Read and approved by:'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
