"""The token layout of an episode: which tokens the model reads, and which of them it sampled itself.

An episode is one growing list of token ids, built piece by piece and never rebuilt from text:

    first prompt | sampled turn 0 | suffix 0 | sampled turn 1 | suffix 1 | ... | sampled last turn

    first prompt   the system turn with the tool schemas and the rules, the request, and the header of the
                   first assistant turn. Rendered by the model's own chat template.
    sampled turn   the token ids that the model sampled, as they are, ending with `<|im_end|>`. If the turn
                   reached the token cap it has no `<|im_end|>`; the buffer then adds one as "forced_eos".
    suffix         everything between that `<|im_end|>` and the next sampled token: the tool result inside a
                   user turn, and the header of the next assistant turn.

Every piece is recorded as a `Segment` (accessdesk/types.py), and the loss mask is built from the segments.
The tests check that for scripted episodes the pieces together are, token for token, what the chat template
renders for the whole conversation.

All of this depends on the chat template (`.cache/hf/Qwen3.5-4B/chat_template.jinja`):
- the template is always called with `enable_thinking=False`, so an assistant turn starts with an empty
  think block. (The 4B template has thinking on by default and the 2B template has it off; with the
  argument given, both render the same.)
- the template strips white space around every message content. `suffix` does the same with the tool result.
- no model-visible text is written here. The two constant parts of the suffix are read off the template.
"""
from __future__ import annotations

from typing import Any

from accessdesk.text import TextBundle
from accessdesk.types import Segment, ToolSpec

# Contents of the made-up conversation from which the constant parts of the suffix are read off:
# (request, assistant message, tool result). They are cut away again and never reach the model.
_DUMMY_REQUEST, _DUMMY_ASSISTANT, _DUMMY_RESULT = "1", "2", "3"

# record["prompt"] (BUILD_SPEC section 8) -> the field of the text bundle that holds the system prompt.
_PROMPT_FIELDS = {"final": "system_prompt", "plain": "plain_system_prompt",
                  "policy_shift": "policy_shift_system_prompt"}


class TokenLayout:
    """The token ids of the fixed parts of a conversation, for one tokenizer and one list of tools."""

    def __init__(self, tokenizer: Any, tool_specs: list[ToolSpec]) -> None:
        self.tokenizer = tokenizer
        self.tool_specs: tuple[ToolSpec, ...] = tuple(tool_specs)
        self.im_end_id: int = tokenizer.eos_token_id      # <|im_end|>: the token that ends a turn
        self.pad_id: int = tokenizer.pad_token_id         # <|endoftext|>
        if self.im_end_id is None or self.pad_id is None or self.im_end_id == self.pad_id:
            raise ValueError("the tokenizer must have an end-of-turn token and a different pad token, got "
                             f"{self.im_end_id} and {self.pad_id}")
        self.suffix_head, self.suffix_tail = self._suffix_parts()

    def encode(self, text: str) -> list[int]:
        """The token ids of a text. Nothing is added at the start or at the end."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def first_prompt(self, system_prompt: str, request_text: str) -> list[int]:
        """The ids of the system turn (tools, then the system prompt), the request and the first header."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": request_text}]
        schemas = [spec.schema() for spec in self.tool_specs]
        return self.encode(self._render(messages, add_generation_prompt=True, tools=schemas))

    def suffix(self, tool_response: str) -> list[int]:
        """The ids between a turn's `<|im_end|>` and the first token of the next assistant turn.

        The text is: line break, a user turn that holds the tool result between `<tool_response>` and
        `</tool_response>`, and the header of the next assistant turn with an empty think block.
        White space around the tool result is removed, as the chat template does.
        """
        return self.encode(self.suffix_head + tool_response.strip() + self.suffix_tail)

    def _render(self, messages: list[dict], *, add_generation_prompt: bool,
                tools: list[dict] | None = None) -> str:
        """The chat template's text for `messages`. Thinking is always switched off."""
        return self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt,
                                                  enable_thinking=False)

    def _suffix_parts(self) -> tuple[str, str]:
        """The text before and the text after the tool result in a suffix, read off the chat template.

        Two renders of a made-up conversation are compared. The first ends with an assistant turn. The second
        has a tool result after it and the header of the next assistant turn. What the second has more, after
        the `<|im_end|>` of the assistant turn, is the suffix; the tool result cuts it into its two parts.
        """
        end_of_turn = self.tokenizer.eos_token
        messages = [{"role": "user", "content": _DUMMY_REQUEST},
                    {"role": "assistant", "content": _DUMMY_ASSISTANT}]
        before = self._render(messages, add_generation_prompt=False)
        after = self._render(messages + [{"role": "tool", "content": _DUMMY_RESULT}],
                             add_generation_prompt=True)
        cut = before.rindex(end_of_turn) + len(end_of_turn)
        added = after[cut:]
        if after[:cut] != before[:cut] or added.count(_DUMMY_RESULT) != 1:
            raise ValueError("the chat template does not render a tool result as one piece after the turn")
        head, _, tail = added.partition(_DUMMY_RESULT)
        return head, tail


class TokenBuffer:
    """The token ids of one episode and their segments. It only grows; a piece is never changed."""

    def __init__(self, layout: TokenLayout, system_prompt: str, request_text: str) -> None:
        self.layout = layout
        self.ids: list[int] = []
        self.segments: list[Segment] = []
        self._append("prompt", layout.first_prompt(system_prompt, request_text), -1)

    def add_sampled(self, ids: list[int], turn: int) -> None:
        """Append the ids that the model sampled in assistant turn `turn` (0, 1, 2, ...), as they are.

        The ids end with `<|im_end|>`. If they do not (the turn reached the token cap), one `<|im_end|>` is
        appended as its own segment of kind "forced_eos", so that the conversation stays well-formed and the
        loss mask can leave out a token that the model did not choose.
        """
        last = self.segments[-1]
        if last.kind not in ("prompt", "suffix") or turn != last.turn + 1:
            raise ValueError(f"sampled turn {turn} cannot follow the {last.kind!r} segment of turn "
                             f"{last.turn}")
        if not ids:
            raise ValueError(f"sampled turn {turn} has no token")
        if self.layout.im_end_id in ids[:-1]:
            raise ValueError(f"sampled turn {turn} has an end-of-turn token before its last token")
        self._append("sampled", ids, turn)
        if ids[-1] != self.layout.im_end_id:
            self._append("forced_eos", [self.layout.im_end_id], turn)

    def add_suffix(self, tool_response: str, turn: int) -> None:
        """Append the tool result of turn `turn` and the header of the next assistant turn."""
        last = self.segments[-1]
        if last.kind not in ("sampled", "forced_eos") or turn != last.turn:
            raise ValueError(f"the suffix of turn {turn} cannot follow the {last.kind!r} segment of turn "
                             f"{last.turn}")
        self._append("suffix", self.layout.suffix(tool_response), turn)

    def _append(self, kind: str, ids: list[int], turn: int) -> None:
        start = len(self.ids)
        self.ids.extend(ids)
        self.segments.append(Segment(kind, start, len(self.ids), turn))


def buffer_from_record(layout: TokenLayout, record: dict, family: dict, text: TextBundle) -> TokenBuffer:
    """The token buffer of a finished episode of which only the texts are known (a scripted episode).

    Each assistant text is tokenised on its own and gets `<|im_end|>` as its last sampled token; a turn that
    is marked as truncated gets none, so the buffer adds the forced one. The tool result follows every turn
    but the last one.

    Use this for episodes that no model sampled. For a sampled episode the ids saved by the rollout are the
    truth: text that is tokenised again need not give back the ids that the model chose.
    The assistant text is tokenised as it is. The chat template would strip white space at its ends, so for
    such a text this buffer and the template's render differ; a call written by `render_tool_call` has none.
    A record that is marked as a harness error is refused: it must never become training data.
    """
    if record.get("harness_error") is not None:
        raise ValueError(f"the record is a harness error: {record['harness_error']}")
    if record["family_id"] != family["family_id"]:
        raise ValueError(f"the record is of family {record['family_id']}, not of {family['family_id']}")
    if tuple(text.tool_specs(record["toolset"])) != layout.tool_specs:
        raise ValueError(f"the layout was not built with the tools of toolset {record['toolset']!r}")
    wordings = [wording["text"] for wording in family["wordings"] if wording["style"] == record["style"]]
    if len(wordings) != 1:
        raise ValueError(f"family {family['family_id']} has no wording of style {record['style']}")
    buffer = TokenBuffer(layout, getattr(text, _PROMPT_FIELDS[record["prompt"]]), wordings[0])
    turns = record["turns"]
    for index, turn in enumerate(turns):
        ids = layout.encode(turn["assistant_text"])
        if not turn["truncated"]:
            ids.append(layout.im_end_id)
        buffer.add_sampled(ids, turn["turn"])
        if index + 1 < len(turns):
            buffer.add_suffix(turn["tool_response"], turn["turn"])
    return buffer
