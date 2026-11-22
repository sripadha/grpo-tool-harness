"""The model side of the harness: loading, the sampler, and the batched multi-turn rollout.

One model object samples (here) and trains (`accessdesk/train/`). There is no second copy and no server.

What "a sampled token" is, exactly. One `Sampler.sample` call is one `generate` call for a batch of prompts.
For each row the new tokens are cut at the first `<|im_end|>`:
  - ids      the new tokens up to and including the first `<|im_end|>`. Everything after it in the batch is
             the pad token that `generate` writes into finished rows; it is dropped.
  - logprobs one number per id: its log-probability under the distribution it was drawn from.
  - a row with no `<|im_end|>` is `truncated`; it has exactly `cfg.max_new_tokens` ids.

Why the log-probabilities are exact. `generate` applies a list of logits processors to the last-position
logits (cast to float32) and then draws from `softmax` of the result (transformers 5.23.0,
generation/utils.py lines 3247-3276). A processor that the caller passes is put BEFORE generate's own
temperature / top-k / top-p processors (line 1594 against lines 1597-1644), so it would not see the final
scores. Therefore the sampler builds the temperature / top-k / top-p processors itself, in generate's order,
puts `LogprobRecorder` last, and switches generate's own ones off. The list passed here is then the complete
list, and the recorder sees exactly the scores that are sampled from.

torch and the model code of transformers are imported inside the functions that need them. So this module, and
`rollout`, which needs neither, can be imported on a machine without torch.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
import warnings
from dataclasses import dataclass
from typing import Any

from accessdesk.config import RunConfig
from accessdesk.harness.tokens import TokenBuffer
from accessdesk.rng import derive_seed

# The text-only model class. `AutoModelForCausalLM` maps both config types of a Qwen3.5 repository to it
# (transformers models/auto/modeling_auto.py lines 917 and 920); it has no vision tower.
TEXT_ONLY_CLASS = "Qwen3_5ForCausalLM"

# Revisions read from the Hub on 2026-11-16 (GPU_SPEC section 0). The 9B's was added on 2026-11-17: it is the
# revision recorded in results/calibration/round1/9b/run_manifest.json, not read from the Hub again. Used when
# no revision is given.
PINNED_REVISIONS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}

PACKAGES = ("torch", "transformers", "peft", "accelerate", "flash-linear-attention", "fla-core",
            "causal-conv1d", "kernels", "safetensors", "tokenizers", "numpy")

# Texts of the warning in transformers integrations/hub_kernels.py lines 1044-1048.
FALLBACK_MARKERS = ("falling back", "much slower")

# Generation settings whose "off" value is None. If the model's own generation config sets one of them,
# generate() would add a logits processor or a stopping rule that this sampler does not know about
# (generation/utils.py lines 1431-1656 and 1666-1693), so the sampler refuses to start.
MUST_BE_UNSET = ("guidance_scale", "sequence_bias", "bad_words_ids", "min_new_tokens", "forced_bos_token_id",
                 "forced_eos_token_id", "exponential_decay_length_penalty", "suppress_tokens",
                 "begin_suppress_tokens", "top_h", "min_p", "watermarking_config", "stop_strings", "max_time",
                 "prefill_chunk_size", "cache_implementation")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def causal_lm(model: Any) -> Any:
    """The transformers model inside `model`: `model` itself, or the model that a peft wrapper holds.

    The sampler and the training forward both go through this function, so both use the same object.
    Do not use `peft_model.model` for this: peft forwards unknown attributes to the wrapped model, so that
    name gives the `Qwen3_5ForCausalLM` on a wrapper but its decoder on the bare model (peft 0.21.3
    peft_model.py lines 979-986 and 1079-1083).
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(core).__name__ != TEXT_ONLY_CLASS:
        raise TypeError(f"expected the text-only class {TEXT_ONLY_CLASS}, got {type(core).__name__}")
    return core


def resolve_revision(model_id: str, revision: str | None) -> str | None:
    """The revision to load: the given one, else the pinned one of a known model, else None (a local path)."""
    return revision if revision is not None else PINNED_REVISIONS.get(model_id)


def load_model(model_id: str, revision: str | None = None, *, adapter_dir: str | None = None,
               device: str = "cuda", adapter_trainable: bool = False) -> tuple[Any, Any]:
    """Load the text-only model in bfloat16 on one device, and its tokenizer with left padding.

    Returns (model, tokenizer); the model is in eval mode. With `adapter_dir` the LoRA adapter saved there is
    loaded on top and the peft wrapper is returned; its weights are frozen unless `adapter_trainable` is True.

    Raises if the class is not the text-only one, if a vision module is present, if a weight of the model was
    not found in the checkpoint (transformers would only warn and leave it at its random start value), or if
    peft reports adapter weights that it could not find in the saved file.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = resolve_revision(model_id, revision)
    source = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **source)
    tokenizer.padding_side = "left"
    model, loading = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                          output_loading_info=True, **source)
    core = causal_lm(model)
    not_loaded = {key: sorted(map(str, loading[key]))[:5]
                  for key in ("missing_keys", "mismatched_keys", "error_msgs") if loading[key]}
    if not_loaded:
        raise RuntimeError(f"weights of {model_id} were not loaded from the checkpoint: {not_loaded}")
    vision = [name for name, _ in core.named_modules() if "visual" in name or "vision" in name]
    if vision:
        raise RuntimeError(f"the loaded model has vision modules, for example {vision[0]!r}")
    model.to(device)
    if adapter_dir is not None:
        model = _load_adapter(model, adapter_dir, adapter_trainable)
    model.eval()
    return model, tokenizer


def _load_adapter(model: Any, adapter_dir: str, trainable: bool) -> Any:
    """`PeftModel.from_pretrained`, with its "missing adapter keys" warning turned into an error.

    peft only warns when saved weights do not match the modules (peft_model.py lines 628-644); the adapter
    would then be partly or fully without effect, and nothing else would show it.
    """
    from peft import PeftModel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
    missing = [str(w.message) for w in caught if "missing adapter keys" in str(w.message)]
    if missing:
        raise RuntimeError(f"the adapter in {adapter_dir} does not fit the model: {missing[0][:500]}")
    return wrapped


# ---------------------------------------------------------------------------
# Reports for the run manifest
# ---------------------------------------------------------------------------
def environment_report() -> dict:
    """Package versions, CUDA version and the GPU, for run_manifest.json."""
    import torch

    has_gpu = torch.cuda.is_available()
    memory_bytes = torch.cuda.get_device_properties(0).total_memory if has_gpu else None
    return {
        "python": sys.version.split()[0],
        "packages": {name: _package_version(name) for name in PACKAGES},
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if has_gpu else None,
        "gpu_memory_gb": round(memory_bytes / 2**30, 1) if has_gpu else None,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def kernel_report(model: Any) -> dict:
    """Which Gated DeltaNet kernels run: the fast ones of `fla` and `causal_conv1d`, or the slow torch code.

    transformers picks the implementation of four functions when its Qwen3.5 module is imported: the package
    function if the import works, else its own torch code (integrations/hub_kernels.py lines 990-1055; the hub
    `kernels` package is used only with `use_kernels=True`, which `load_model` does not set). A torch fallback
    is not an error there: it logs one warning, once per process, and goes on.

    This function (a) tries the same imports, and (b) runs the four functions once and collects that warning:
    a short prompt (the chunked kernel and `causal_conv1d_fn`) and one cached single-token step (the recurrent
    kernel and `causal_conv1d_update`; modeling_qwen3_5.py lines 572-643). The once-per-process memory of the
    warning is cleared first (utils/logging.py line 331), so the result does not depend on what ran before.

    Keys: `imports` (per module: ok and the error text), `use_kernels`, `fallback_warnings` (the messages),
    `torch_fallback_functions` (the names in them), `gated_delta_rule` ("fla" or "torch"), `causal_conv1d`
    ("causal_conv1d" or "torch"), and `fast_path` (True if nothing fell back).
    """
    core = causal_lm(model)
    modules = ("fla", "fla.ops", "fla.ops.gated_delta_rule", "causal_conv1d", "kernels")
    messages = _probe_messages(core)
    fallbacks = [m for m in messages if any(marker in m for marker in FALLBACK_MARKERS)]
    slow = sorted({m.split("`")[1] for m in fallbacks if m.count("`") >= 2})
    return {
        "imports": {name: _try_import(name) for name in modules},
        "use_kernels": bool(core.use_kernels),
        "fallback_warnings": fallbacks,
        "torch_fallback_functions": slow,
        "gated_delta_rule": "torch" if any("gated_delta_rule" in name for name in slow) else "fla",
        "causal_conv1d": "torch" if any("causal_conv1d" in name for name in slow) else "causal_conv1d",
        "fast_path": not fallbacks,
    }


def _try_import(name: str) -> dict:
    try:
        importlib.import_module(name)
    except Exception as error:  # an ImportError, or a broken compiled extension
        return {"ok": False, "error": f"{type(error).__name__}: {error}"[:300]}
    return {"ok": True, "error": None}


def _probe_messages(core: Any) -> list[str]:
    """Every message that transformers logs at warning level during a prompt pass and one cached step."""
    import torch
    from transformers.utils import logging as hf_logging

    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Collect(level=logging.WARNING)
    hf_logging.warning_once.cache_clear()
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_warning()
    hf_logging.add_handler(handler)
    try:
        with torch.no_grad():
            prompt = torch.arange(1, 9, device=core.device)[None]                 # eight arbitrary token ids
            first = core(input_ids=prompt, use_cache=True)
            core(input_ids=prompt[:, -1:], past_key_values=first.past_key_values, use_cache=True)
    finally:
        hf_logging.remove_handler(handler)
        hf_logging.set_verbosity(verbosity)
    return messages


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@dataclass
class SampledTurn:
    ids: list[int]          # the new tokens, up to and including <|im_end|> if it was produced
    logprobs: list[float]   # log-probability of each of those tokens under the sampling distribution
    truncated: bool         # True if the token cap was reached before <|im_end|>


def set_sampling_seed(seed: int) -> None:
    """Set torch's random state (CPU and every CUDA device) from `seed`. Used by the sampler only.

    Seeds from `derive_seed` use 64 bits; they are reduced to 63 bits here, so that every torch seeding
    function accepts them.
    """
    import torch

    seed = int(seed) % (2**63)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_batch(seqs: list[list[int]], pad_id: int, side: str = "left") -> tuple[Any, Any]:
    """Token-id lists of different lengths as one batch.

    Returns (input_ids, attention_mask): two int64 tensors [len(seqs), longest length] on the CPU. The mask is
    1 on real tokens and 0 on pad tokens. `side` is where the pad tokens go. The sampler and the training
    forward both use "left": a decoder-only model must have the real tokens at the end of the row.
    """
    import torch

    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    if not seqs or any(len(seq) == 0 for seq in seqs):
        raise ValueError("pad_batch needs at least one sequence, and no sequence may be empty")
    width = max(len(seq) for seq in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, seq in enumerate(seqs):
        start = width - len(seq) if side == "left" else 0
        input_ids[row, start:start + len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, start:start + len(seq)] = 1
    return input_ids, attention_mask


class LogprobRecorder:
    """Records the log-probability of every token that `generate` draws. It must be the LAST logits processor.

    It has the call signature of `transformers.LogitsProcessor`; `LogitsProcessorList` only calls its entries
    (generation/logits_process.py lines 86-97). It returns the scores unchanged.

    At decoding step k `generate` calls it with the final scores of that step, [rows, vocabulary], and then
    draws token k from `softmax(scores)`. The token is not known yet, so the scores are kept, and the
    log-probability `scores[token] - logsumexp(scores)` is taken one call later, when the drawn token is the
    last column of `input_ids`. `finish` does the same for the last step, with the tokens `generate` returned.
    """

    def __init__(self) -> None:
        self._scores: Any = None        # the scores of the step whose token is not known yet
        self._steps: list[Any] = []     # one [rows] tensor per finished step

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self._scores is not None:
            self._steps.append(_chosen_logprob(self._scores, input_ids[:, -1]))
        self._scores = scores
        return scores

    def finish(self, new_tokens: Any) -> Any:
        """`new_tokens`: [rows, steps], the tokens that `generate` appended. Returns float32 [rows, steps].

        In a row that ended early, the entries after its end token belong to pad tokens that were written, not
        drawn; the caller drops them together with those tokens.
        """
        import torch

        if self._scores is None:
            raise RuntimeError("the recorder was never called: it was not in the processor list")
        self._steps.append(_chosen_logprob(self._scores, new_tokens[:, -1]))
        self._scores = None
        if len(self._steps) != new_tokens.shape[1]:
            raise RuntimeError(f"{len(self._steps)} decoding steps were recorded, but generate returned "
                               f"{new_tokens.shape[1]} new tokens per row")
        return torch.stack(self._steps, dim=1)


def _chosen_logprob(scores: Any, tokens: Any) -> Any:
    """log_softmax(scores)[row, tokens[row]] for every row, without building the full log_softmax."""
    return scores.gather(1, tokens[:, None]).squeeze(1) - scores.logsumexp(dim=-1)


def cut_at_stop(ids: list[int], logprobs: list[float], stop_id: int, max_new_tokens: int) -> SampledTurn:
    """One row of new tokens as a SampledTurn: cut after the first `stop_id`, or truncated if it has none."""
    if stop_id in ids:
        n = ids.index(stop_id) + 1
        return SampledTurn(ids=ids[:n], logprobs=logprobs[:n], truncated=False)
    if len(ids) != max_new_tokens:
        raise RuntimeError(f"no stop token after {len(ids)} new tokens; the cap is {max_new_tokens}")
    return SampledTurn(ids=list(ids), logprobs=list(logprobs), truncated=True)


class Sampler:
    """Draws one assistant turn for each prompt of a batch, and records the log-probability of every token.

    Sampling settings come from the config: `temperature`, `top_k` (0 = off), `top_p` (1.0 = off) and
    `max_new_tokens`. There is no repetition penalty and no other filter. With temperature 1, top_k 0 and
    top_p 1.0 the recorded numbers are the model's own log-probabilities, the same quantity that
    `accessdesk.train.forward.token_logprobs` computes. With other settings they are log-probabilities under
    the changed distribution.

    The stop token is the tokenizer's end-of-turn token `<|im_end|>` (`tokenizer.eos_token_id`). The model
    config's own end token is `<|endoftext|>`, so the stop token is always passed to `generate` explicitly.
    """

    def __init__(self, model: Any, tokenizer: Any, cfg: RunConfig) -> None:
        from transformers import GenerationConfig

        self.core = causal_lm(model)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.stop_id: int = tokenizer.eos_token_id
        self.pad_id: int = tokenizer.pad_token_id
        if self.stop_id is None or self.pad_id is None or self.stop_id == self.pad_id:
            raise ValueError(f"the tokenizer must have different stop and pad tokens, got {self.stop_id} and "
                             f"{self.pad_id}")
        self.warpers = _sampling_warpers(cfg)
        # Every setting is explicit. A value given here wins over the model's own generation config and over
        # the library defaults (generation/utils.py lines 2107-2112; the library default top_k is 50).
        self.generation_config = GenerationConfig(
            do_sample=True, num_beams=1, num_return_sequences=1,
            max_new_tokens=cfg.max_new_tokens, min_length=0,
            eos_token_id=self.stop_id, pad_token_id=self.pad_id,
            # generate()'s own filters are all off: temperature, top-k and top-p are in `self.warpers`.
            temperature=1.0, top_k=0, top_p=1.0, typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0,
            repetition_penalty=1.0, encoder_repetition_penalty=1.0,
            no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
            renormalize_logits=False, remove_invalid_values=False,
            use_cache=True, return_dict_in_generate=False, output_scores=False, output_logits=False,
        )
        self.effective_config = _effective_generation_config(self.core, self.generation_config)

    def settings(self) -> dict:
        """What the sampler does, for the run manifest: its own settings and generate()'s effective config.

        In the effective config `temperature`, `top_k` and `top_p` are the switched-off values, because those
        filters run in the sampler's own processor list; and `max_length` is the library default that
        `max_new_tokens` replaces at each call (generation/utils.py lines 2027-2035).
        """
        return {
            "temperature": self.cfg.temperature, "top_k": self.cfg.top_k, "top_p": self.cfg.top_p,
            "max_new_tokens": self.cfg.max_new_tokens, "chunk_size": self.cfg.chunk_size,
            "stop_id": self.stop_id, "pad_id": self.pad_id, "padding_side": "left",
            "logits_processors": [type(p).__name__ for p in self.warpers] + [LogprobRecorder.__name__],
            "generate_effective_config": self.effective_config,
        }

    def sample(self, prompts: list[list[int]], seed: int) -> list[SampledTurn]:
        """One `generate` call for a list of token-id prompts of different lengths; one result per prompt.

        The prompts are padded on the left. The model is put in eval mode for the call (its mode is restored
        afterwards) and no gradient is built. Eval mode matters: in train mode with gradient checkpointing on,
        transformers switches the decoding cache off (utils/generic.py lines 1042-1046). The random state is
        set from `seed` just before the call, so the same prompts in the same order with the same seed give
        the same tokens on the same machine.
        """
        import torch
        from transformers import LogitsProcessorList

        input_ids, attention_mask = pad_batch(prompts, self.pad_id, side="left")
        recorder = LogprobRecorder()
        was_training = self.core.training
        self.core.eval()
        try:
            set_sampling_seed(seed)
            with torch.no_grad():
                sequences = self.core.generate(
                    input_ids=input_ids.to(self.core.device),
                    attention_mask=attention_mask.to(self.core.device),
                    generation_config=self.generation_config,
                    logits_processor=LogitsProcessorList([*self.warpers, recorder]),
                )
        finally:
            self.core.train(was_training)
        new_tokens = sequences[:, input_ids.shape[1]:]                # [rows, steps]: without the prompts
        logprobs = recorder.finish(new_tokens)                        # [rows, steps]
        return [cut_at_stop(row_ids, row_logprobs, self.stop_id, self.cfg.max_new_tokens)
                for row_ids, row_logprobs in zip(new_tokens.tolist(), logprobs.tolist())]


def _sampling_warpers(cfg: RunConfig) -> list[Any]:
    """Temperature, top-k and top-p as logits processors, in generate()'s own order and with its conditions
    (generation/utils.py lines 1612-1623). With temperature 1.0, top_k 0 and top_p 1.0 the list is empty."""
    from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    warpers: list[Any] = []
    if cfg.temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(float(cfg.temperature)))
    if cfg.top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=int(cfg.top_k)))
    if cfg.top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=float(cfg.top_p)))
    return warpers


def _effective_generation_config(core: Any, generation_config: Any) -> dict:
    """The config that generate() will really use: ours, then the model's own, then the library defaults.

    Computed with the function generate() itself calls (generation/utils.py lines 2072-2137). Raises if the
    result is not plain sampling, or if a setting of the model's own config would add a processor or a
    stopping rule that is not in the sampler's list.
    """
    effective, _ = core._prepare_generation_config(generation_config)
    mode = str(effective.get_generation_mode().value)
    leaks = [name for name in MUST_BE_UNSET if getattr(effective, name, None) is not None]
    if mode != "sample" or leaks:
        raise RuntimeError(f"generate() would not do plain sampling: mode {mode!r}, extra settings {leaks}")
    return effective.to_dict()


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------
def rollout(sampler: Any, layouts: dict, episodes: list, *, run_seed: int, pass_index: int,
            meta: dict) -> list[dict]:
    """Roll the model out on `episodes` (fresh `Episode` objects), in the given order; one record per episode.

    The episodes are cut into chunks of `sampler.cfg.chunk_size`. A chunk advances turn by turn: the token
    buffers of its unfinished episodes go into one `sampler.sample` call with the seed
    `derive_seed(run_seed, pass_index, chunk_index, turn_index)`, and each sampled turn is given to its
    episode. A token buffer is never rebuilt from text: the sampled ids are appended as they are, then the
    tool result and the next assistant header.

    `layouts` maps a tool set ("five", "six") to its `TokenLayout`. The episodes must have been built with
    `sampler.cfg`: one config holds the token cap, the turn limit, the parser mode and the sampling settings.

    `meta` may hold `run`, `arm` and `data_sha256` (the SHA-256 of the scenario file that the families were
    loaded from), the record fields that are the same for all episodes, and `first_sample`.
    The `sample` field is filled in here: `first_sample` (default 0) plus the number of earlier episodes of
    the same scenario in this call. So one call with eight episodes per scenario numbers them 0 to 7, and a
    call with one episode per scenario and `first_sample = s` numbers them all s.

    Each returned item is the episode record with `seed = {run_seed, pass, chunk, pos}` and one more key,
    `tokens = {"ids", "segments", "logprobs"}`: `segments` are [kind, start, end, turn], and `logprobs` has
    one entry per token of the "sampled" segments, in order.

    Harness errors. An exception inside one episode makes that episode a harness error and the others go on.
    If a `sample` call fails, every unfinished episode of that chunk becomes a harness error and the rollout
    goes on with the next chunk. Such a record has `harness_error` = the message and is left out of every
    metric; its `tokens` may be incomplete and must not be trained on.
    """
    unknown = sorted(set(meta) - {"run", "arm", "data_sha256", "first_sample"})
    if unknown:
        raise TypeError(f"meta may hold run, arm, data_sha256 and first_sample, not {unknown}")
    for layout in layouts.values():
        if (layout.im_end_id, layout.pad_id) != (sampler.stop_id, sampler.pad_id):
            raise ValueError("the sampler and the token layout disagree on the stop token or the pad token")
    if any(episode.cfg != sampler.cfg for episode in episodes):
        raise ValueError("an episode was built with another config than the sampler")
    common = {key: value for key, value in meta.items() if key != "first_sample"}
    seen: dict[str, int] = {}                        # scenario id -> episodes of it so far in this call
    records: list[dict] = []
    chunk_size = sampler.cfg.chunk_size
    for chunk_index, start in enumerate(range(0, len(episodes), chunk_size)):
        chunk = episodes[start:start + chunk_size]
        buffers, logprobs, errors = _run_chunk(sampler, layouts, chunk, (run_seed, pass_index, chunk_index))
        for pos, episode in enumerate(chunk):
            seed = {"run_seed": run_seed, "pass": pass_index, "chunk": chunk_index, "pos": pos}
            fields = {**common, "sample": 0, "seed": seed}           # `sample` gets its number just below
            record = _record(episode, buffers[pos], logprobs[pos], errors[pos], fields)
            record["sample"] = meta.get("first_sample", 0) + seen.get(record["scenario_id"], 0)
            seen[record["scenario_id"]] = seen.get(record["scenario_id"], 0) + 1
            records.append(record)
    return records


def _run_chunk(sampler: Any, layouts: dict, chunk: list,
               seed_parts: tuple[int, int, int]) -> tuple[list, list, list]:
    """Advance every episode of one chunk until it is done or has failed.

    Returns three lists with one entry per episode: its token buffer (None if it could not be built), the
    log-probabilities of its sampled tokens, and its error message (None if all went well).
    """
    buffers: list[Any] = [None] * len(chunk)
    logprobs: list[list[float]] = [[] for _ in chunk]
    errors: list[str | None] = [None] * len(chunk)
    for i, episode in enumerate(chunk):
        try:
            buffers[i] = _first_buffer(episode, layouts[episode.toolset])
        except Exception as error:
            errors[i] = f"first prompt: {_message(error)}"
    for turn_index in range(sampler.cfg.turn_limit):
        active = [i for i, episode in enumerate(chunk) if errors[i] is None and not episode.done]
        if not active:
            break
        seed = derive_seed(*seed_parts, turn_index)
        try:
            turns = sampler.sample([buffers[i].ids for i in active], seed)
        except Exception as error:
            for i in active:
                errors[i] = f"turn {turn_index}, sample call: {_message(error)}"
            break
        for i, turn in zip(active, turns):
            try:
                _apply_turn(chunk[i], buffers[i], logprobs[i], turn, turn_index, sampler.tokenizer)
            except Exception as error:
                errors[i] = f"turn {turn_index}: {_message(error)}"
    return buffers, logprobs, errors


def _first_buffer(episode: Any, layout: Any) -> TokenBuffer:
    """The token buffer of a fresh episode: its first prompt, with the tools that the episode accepts."""
    if layout.tool_specs != tuple(episode.tool_specs):
        raise ValueError("the token layout was built with other tools than the episode has")
    return TokenBuffer(layout, episode.system_prompt, episode.request_text)


def _apply_turn(episode: Any, buffer: Any, logprobs: list[float], turn: SampledTurn, turn_index: int,
                tokenizer: Any) -> None:
    """Give one sampled turn to its episode, and append the turn (and the tool result) to the token buffer."""
    body = turn.ids if turn.truncated else turn.ids[:-1]             # the text is parsed without <|im_end|>
    text = tokenizer.decode(body, skip_special_tokens=False)
    result = episode.step(text, truncated=turn.truncated, n_tokens=len(turn.ids))
    buffer.add_sampled(turn.ids, turn_index)
    logprobs.extend(turn.logprobs)
    if not episode.done:
        buffer.add_suffix(result["tool_response"], turn_index)


def _record(episode: Any, buffer: Any, logprobs: list[float], error: str | None, meta: dict) -> dict:
    """The record of one episode with its token data. `error`, or a failed check here, is a harness error."""
    segments = [] if buffer is None else [[s.kind, s.start, s.end, s.turn] for s in buffer.segments]
    n_sampled = sum(end - start for kind, start, end, _ in segments if kind == "sampled")
    if error is None and not episode.done:
        error = "the episode was not done when its chunk ended"
    if error is None and n_sampled != len(logprobs):
        error = f"{n_sampled} sampled tokens in the buffer but {len(logprobs)} log-probabilities"
    if error is not None:
        _close_out(episode)
    record = episode.record(**meta)
    record["harness_error"] = error
    record["tokens"] = {"ids": [] if buffer is None else list(buffer.ids), "segments": segments,
                        "logprobs": list(logprobs)}
    return record


def _close_out(episode: Any) -> None:
    """Bring a failed episode to its end, so that it can still write a record with its scenario and seed.

    An episode writes its record only when it is done. The turns added here are empty, cut-off messages that
    no model wrote; the record is marked as a harness error, and nothing in it is counted or trained on.
    """
    while not episode.done:
        episode.step("", truncated=True, n_tokens=0)


def _message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:500]
