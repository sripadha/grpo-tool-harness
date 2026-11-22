"""A second way to find the tokens that the assistant wrote: by reading the token ids alone.

The loss mask of the repository is built from the segments that the harness records while it builds an
episode's token buffer. The function here knows nothing about segments, and it does not use
accessdesk/harness/tokens.py. It reads a finished list of token ids the way a person reads the decoded text:

    an assistant turn starts with the header   <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    what the assistant wrote comes after the header and runs up to the next <|im_end|>, which is included

If both ways give the same weights, a mistake in the segment bookkeeping (a start that is off by one, a
header counted as sampled, a tool result counted as sampled) would have to be made twice in the same way.

What the scan cannot know: whether the model sampled an `<|im_end|>` or the harness added it after a turn
that reached the token cap. So it agrees with the segment mask only on episodes without a cut-off turn, and
on a cut-off turn it differs by exactly the one forced token. It would also be misled by a tool result that
holds the text of an assistant header; the segments are not.
"""
from __future__ import annotations

from typing import Any

# Written out here on purpose, and not taken from the harness: the check must not share its source.
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END_OF_TURN = "<|im_end|>"


def scan_mask(ids: list[int], tokenizer: Any) -> list[int]:
    """One weight per token id: 1 for every token after an assistant header up to and including the next
    `<|im_end|>` (or up to the end of the list, if there is none), 0 for every other token."""
    header = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
    (end_of_turn,) = tokenizer.encode(END_OF_TURN, add_special_tokens=False)
    weights = [0] * len(ids)
    position = 0
    while position < len(ids):
        if ids[position:position + len(header)] != header:
            position += 1
            continue
        position += len(header)                     # the header itself keeps weight 0
        while position < len(ids):
            weights[position] = 1
            position += 1
            if ids[position - 1] == end_of_turn:
                break
    return weights
