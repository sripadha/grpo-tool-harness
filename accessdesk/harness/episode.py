"""One episode at message level: one request, the assistant's messages, the tool results, the final record.

The same class serves the scripted policies and the model. It knows nothing about tokens: the caller gives it
the text of each assistant message and gets back the tool result that the model reads next.

A turn is one assistant message, whatever it holds. The episode ends at a successful close_request or when
`cfg.turn_limit` turns are used, whichever comes first.

The verifier and the reward run once, in `record()`, on the final state only. Nothing in `step()` looks at
the expected state, so no tool result can depend on it.
"""
from __future__ import annotations

import json

from accessdesk import impl
from accessdesk.config import RunConfig
from accessdesk.env.rules import RULESETS
from accessdesk.env.tools import execute
from accessdesk.env.world import World
from accessdesk.harness.parser import parse_assistant_message
from accessdesk.text import TextBundle
from accessdesk.types import ToolSpec

SCHEMA = 1                                    # version of the episode record
PROMPTS = ("final", "plain", "policy_shift")  # which system prompt of the text file is used
META_KEYS = ("run", "arm", "sample", "seed")  # the record fields that the caller gives to record()
DATA_HASH_KEY = "data_sha256"                 # optional: the SHA-256 of the scenario file of the family


class Episode:
    """One rollout of one scenario (a family in one wording style).

    family   a family record (it is read, never changed)
    style    the wording style; the family must have a wording in this style
    ruleset  the rule set whose expected state the verifier uses
    toolset  "five", or "six" for the extra-tool set
    prompt   "final": text.system_prompt. "plain": text.plain_system_prompt (an error if it is empty).
             "policy_shift": text.policy_shift_system_prompt; the rule set must then be the one that this
             prompt describes (text.policy_shift_variant).
             With "final" and "plain" any rule set is accepted: the scripted policies need this to test the
             rule sets that have no prompt. The record holds both `prompt` and `ruleset`.

    Public state: system_prompt, request_text, tool_specs, messages, turns, done, world.
    `messages` is [system, user], then per turn {"role": "assistant", "content": the text as given} and
    {"role": "tool", "content": the JSON text of the tool result}.
    """

    def __init__(self, family: dict, style: int, text: TextBundle, cfg: RunConfig, *,
                 ruleset: str = "default", toolset: str = "five", prompt: str = "final") -> None:
        if ruleset not in RULESETS:
            raise ValueError(f"unknown rule set {ruleset!r}; known: {RULESETS}")
        wordings = [w["text"] for w in family["wordings"] if w["style"] == style]
        if len(wordings) != 1:
            raise ValueError(f"family {family['family_id']} has no wording in style {style!r}")
        self.family = family
        self.style = style
        self.text = text
        self.cfg = cfg
        self.ruleset = ruleset
        self.toolset = toolset
        self.prompt = prompt
        self.system_prompt: str = _system_prompt(text, prompt, ruleset)
        self.request_text: str = wordings[0]
        self.tool_specs: list[ToolSpec] = text.tool_specs(toolset)      # raises for an unknown toolset
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt},
                                     {"role": "user", "content": self.request_text}]
        self.turns: list[dict] = []
        self.done: bool = False
        self.world = World(family["world"])

    def step(self, assistant_text: str, *, truncated: bool = False, n_tokens: int | None = None) -> dict:
        """Take one assistant message, run its tool call if it is one, and return the turn record.

        `assistant_text` is the message without its end-of-turn token; `truncated` says that the token cap cut
        it off; `n_tokens` is only stored. The message is parsed with the strictness of cfg.parser_strict.

        - Not exactly one valid call (malformed): the turn counts, nothing is executed, and the tool result is
          the malformed-call error of the text file.
        - A valid call: it is executed. If the tool reports an error, the turn counts and nothing was written;
          this is a tool error, not a malformed call.
        The episode is done after a successful close_request, or when cfg.turn_limit turns are used.

        The returned dict is the turn record that is saved (do not change it):
            turn, assistant_text, truncated, n_tokens, parse_ok, parse_error, tool_key, args, tool_error,
            tool_response
        `args` are the arguments as they were executed: internal parameter keys, and for close_request the
        internal resolution key. `tool_response` is the JSON text that the model reads next.
        Raises RuntimeError if the episode is already done.
        """
        if self.done:
            raise RuntimeError("the episode is over: step() must not be called again")
        parsed = parse_assistant_message(assistant_text, self.tool_specs, strict=self.cfg.parser_strict,
                                         truncated=truncated)
        if parsed.ok:
            args = self._executable_args(parsed.tool_key, parsed.args)
            outcome = execute(self.world, self.text, parsed.tool_key, args)
            tool_error, response, closes = outcome.error, outcome.response, outcome.closes
        else:
            args, tool_error, closes = None, None, False
            error_body = {self.text.result_keys["error"]: self.text.error_text("malformed_call")}
            response = json.dumps(error_body, ensure_ascii=False)
        turn = {"turn": len(self.turns), "assistant_text": assistant_text, "truncated": truncated,
                "n_tokens": n_tokens, "parse_ok": parsed.ok, "parse_error": parsed.error,
                "tool_key": parsed.tool_key, "args": args, "tool_error": tool_error,
                "tool_response": response}
        self.turns.append(turn)
        self.messages.append({"role": "assistant", "content": assistant_text})
        self.messages.append({"role": "tool", "content": response})
        self.done = closes or len(self.turns) >= self.cfg.turn_limit
        return turn

    def _executable_args(self, tool_key: str, args: dict[str, str]) -> dict[str, str]:
        """The parsed arguments in the form that the tools take: a visible resolution becomes its key."""
        if tool_key != "close_request":
            return dict(args)
        resolution = self.text.resolution_key(args["resolution"])
        if resolution is None:      # the parser accepts only the values of the text file, so this is a bug
            raise RuntimeError(f"the parser let through an unknown resolution {args['resolution']!r}")
        return {**args, "resolution": resolution}

    @property
    def n_malformed(self) -> int:
        """Assistant messages that were not exactly one valid tool call."""
        return sum(1 for turn in self.turns if not turn["parse_ok"])

    @property
    def n_tool_errors(self) -> int:
        """Valid calls that the tool answered with an error (they wrote nothing)."""
        return sum(1 for turn in self.turns if turn["tool_error"] is not None)

    def record(self, **meta: object) -> dict:
        """The episode record: one JSON-serialisable dict, written as one line of a record file.

        `meta` may hold `run`, `arm`, `sample` and `seed`; they are stored as given (defaults: None, None, 0,
        None). It may also hold `data_sha256`: the SHA-256 (64 hexadecimal digits) of the scenario file that
        the family was loaded from. The record has the key `data_sha256` only if it is given; the results
        table uses it to refuse a comparison of runs on different scenario files.

        The verifier and the reward are called here, through accessdesk.impl:
            passed = verify(final_state, family["expected"][ruleset])
            reward = reward(passed, n_malformed, n_turns, family["turn_budget"][ruleset], cfg)
        The reward must be a whole number from 0 to cfg.pass_reward; if cfg.penalise_malformed_on_fail is
        on, from -cfg.malformed_cap (a failing episode can then be below 0).
        `config_sha256` is cfg.sha256(), computed for this very config object (two configs that compare
        equal, such as temperature 1 and 1.0, have different hashes, so the hash is never kept for another
        one). `settings` is cfg.compared_settings(): the values that every arm of a results table must share.
        Raises RuntimeError if the episode is not done, and TypeError for an unknown meta key, for a data
        hash of another form, for a verdict that is not a bool and for a reward outside that range.
        """
        if not self.done:
            raise RuntimeError("the episode is not done: there is no record yet")
        unknown = sorted(set(meta) - set(META_KEYS) - {DATA_HASH_KEY})
        if unknown:
            raise TypeError(f"record() takes only {META_KEYS} and {DATA_HASH_KEY!r}, not {unknown}")
        data_sha256 = meta.get(DATA_HASH_KEY)
        if data_sha256 is not None and not _is_sha256(data_sha256):
            raise TypeError(f"{DATA_HASH_KEY} must be 64 hexadecimal digits in lower case, "
                            f"not {data_sha256!r}")
        family, cfg = self.family, self.cfg
        final_state = self.world.state()
        turn_budget = family["turn_budget"][self.ruleset]
        passed = impl.get("verify")(final_state, family["expected"][self.ruleset])
        if not isinstance(passed, bool):
            raise TypeError(f"verify() must return True or False, not {passed!r}")
        reward = impl.get("reward")(passed, self.n_malformed, len(self.turns), turn_budget, cfg)
        lowest = -cfg.malformed_cap if cfg.penalise_malformed_on_fail else 0
        if isinstance(reward, bool) or not isinstance(reward, int) or not lowest <= reward <= cfg.pass_reward:
            raise TypeError(f"reward() must return an int from {lowest} to {cfg.pass_reward}, not {reward!r}")
        record = {
            "schema": SCHEMA, "run": meta.get("run"), "arm": meta.get("arm"),
            "split": family["split"], "family_id": family["family_id"], "style": self.style,
            "scenario_id": f"{family['family_id']}/s{self.style}", "outcome": family["outcome"],
            "ruleset": self.ruleset, "toolset": self.toolset, "prompt": self.prompt,
            "sample": meta.get("sample", 0), "seed": meta.get("seed"),
            "turns": self.turns, "n_turns": len(self.turns), "n_malformed": self.n_malformed,
            "n_tool_errors": self.n_tool_errors, "closed": self.world.closed,
            "ended_by": "close" if self.world.closed else "turn_limit",
            "final_state": final_state, "passed": passed, "reward": reward, "turn_budget": turn_budget,
            "impl": {"verify": impl.which("verify"), "reward": impl.which("reward")},
            "text_sha256": self.text.sha256, "text_is_placeholder": self.text.is_placeholder,
            "config_sha256": cfg.sha256(), "settings": cfg.compared_settings(),
            "harness_error": None,
        }
        if data_sha256 is not None:
            record[DATA_HASH_KEY] = data_sha256
        return record


def _is_sha256(value: object) -> bool:
    """True for a string of 64 hexadecimal digits in lower case, as hashlib's hexdigest() gives it."""
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _system_prompt(text: TextBundle, prompt: str, ruleset: str) -> str:
    if prompt == "final":
        return text.system_prompt
    if prompt == "plain":
        if not text.plain_system_prompt:
            raise ValueError("the text file has no plain system prompt")
        return text.plain_system_prompt
    if prompt == "policy_shift":
        if ruleset != text.policy_shift_variant:
            raise ValueError(f"the policy-shift prompt describes the rule set {text.policy_shift_variant!r}; "
                             f"it cannot be used with {ruleset!r}")
        return text.policy_shift_system_prompt
    raise ValueError(f"unknown prompt {prompt!r}; known: {PROMPTS}")
