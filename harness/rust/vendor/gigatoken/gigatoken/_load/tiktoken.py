"""The .tiktoken vocabulary files OpenAI publishes, keyed by file name.

A .tiktoken file holds mergeable ranks and nothing else: the pretokenization
scheme (split regex) and the special tokens of an encoding live in the code
that defines it, not in the file. These are the definitions for the files
served from openaipublic.blob.core.windows.net/encodings, transcribed from
https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py.
Any other rank file has to name its scheme at load time. (p50k_base is
absent because its ranks are not dense — it leaves 50256 free for
<|endoftext|> — which the rank loader rejects.)
"""

from typing import NamedTuple

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"


class TiktokenEncoding(NamedTuple):
    """What a .tiktoken rank file does not carry: the name of the
    pretokenization scheme, and the special tokens with their ids."""

    pretokenizer: str
    special_tokens: dict[str, int]


ENCODINGS: dict[str, TiktokenEncoding] = {
    "r50k_base": TiktokenEncoding("gpt2", {ENDOFTEXT: 50256}),
    "cl100k_base": TiktokenEncoding(
        "gpt4",
        {ENDOFTEXT: 100257, FIM_PREFIX: 100258, FIM_MIDDLE: 100259, FIM_SUFFIX: 100260, ENDOFPROMPT: 100276},
    ),
    "o200k_base": TiktokenEncoding("o200k", {ENDOFTEXT: 199999, ENDOFPROMPT: 200018}),
}
