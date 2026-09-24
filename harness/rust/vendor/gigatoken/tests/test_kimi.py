"""End-to-end parity for the moonshotai Kimi tokenizer line.

Every repo in the line ships the same tiktoken.model (no tokenizer.json)
with the Kimi pretokenizer regex in its remote code; only the special
tokens of tokenizer_config.json differ. The reference is tiktoken.Encoding
built exactly as tokenization_kimi.py builds it.
"""

import json

import pytest
import tiktoken
from tiktoken.load import load_tiktoken_bpe

from hf_cache import hf_file

from gigatoken import Tokenizer

# The full Kimi/Moonlight line: one shared tiktoken.model, per-repo specials.
KIMI_REPOS = [
    "moonshotai/Kimi-K2-Base",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Instruct-0905",
    "moonshotai/Kimi-K2-Thinking",
    "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-K2.6",
    "moonshotai/Kimi-K2.7-Code",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "moonshotai/Kimi-VL-A3B-Instruct",
    "moonshotai/Moonlight-16B-A3B-Instruct",
]

# The pat_str of tokenization_kimi.py / tokenization_moonshot.py (identical
# across the line).
KIMI_PAT = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

TEXTS = [
    "Hello, world! The quick brown fox jumps over the lazy dog.",
    "I'll say we've done what they'd want, it's John's turn, WE'VE SHOUTED",
    "人工智能正在改变世界。月之暗面公司发布了Kimi大模型，支持超长上下文。",
    "使用Python调用API接口，每秒处理1000个请求。模型参数量达到1万亿。",
    "日本語のテキスト。漢字とひらがなカタカナが混在。한국어 텍스트입니다.",
    "1234567890 3.14159 v2.6 100,000,000 2026-07-18 一二三〇",
    "path/to/file http://example.com/a/b a/\nb /usr/bin\n/ etc",
    "}\n/// doc comment\n//! module doc\n*/\n/**\n.\n//x",
    "def foo(x: int) -> int:\n    return x + 1\n",
    "café résumé naïve Ñoño año Zürich İstanbul ﬁnancial",
    "emoji: 🚀🌍🎉 mixed hello 世界 🌎 done",
    "中English文 abc中文def A中B 中's 中'se ⼀⼁ 々中〇",
    "   \n\n\t\r\n   ",
]


@pytest.fixture(scope="session", params=KIMI_REPOS)
def repo_id(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def kimi_reference(repo_id) -> tiktoken.Encoding:
    """The reference encoder, built exactly as tokenization_kimi.py does."""
    ranks = load_tiktoken_bpe(str(hf_file(repo_id, "tiktoken.model")))
    config = json.loads(hf_file(repo_id, "tokenizer_config.json").read_bytes())
    specials = {
        t["content"]: int(i) for i, t in config["added_tokens_decoder"].items()
    }
    return tiktoken.Encoding(
        name=repo_id, pat_str=KIMI_PAT, mergeable_ranks=ranks, special_tokens=specials
    )


@pytest.fixture(scope="session")
def kimi_tok(repo_id) -> Tokenizer:
    return Tokenizer(repo_id)


def test_loads_with_specials(repo_id, kimi_tok, kimi_reference):
    """The repo loads from its id, with the config's special tokens."""
    assert kimi_tok._special_tokens() == kimi_reference._special_tokens


def test_parity_plain_text(kimi_tok, kimi_reference):
    for text in TEXTS:
        expected = kimi_reference.encode(text, allowed_special=set())
        actual = kimi_tok.encode(text).tolist()
        assert actual == expected, f"Mismatch on {text!r}"


def test_parity_special_tokens(kimi_tok, kimi_reference):
    """Special tokens in the input are matched atomically, like the
    reference with allowed_special='all'."""
    specials = sorted(kimi_reference._special_tokens)[:6]
    text = "".join(f"before {s} after中文" for s in specials)
    expected = kimi_reference.encode(text, allowed_special="all")
    assert kimi_tok.encode(text).tolist() == expected


def test_decode_roundtrip(kimi_tok, kimi_reference):
    for text in TEXTS:
        ids = kimi_reference.encode(text, allowed_special=set())
        assert kimi_tok.decode(ids) == text.encode()


def test_batch_matches_single(kimi_tok):
    docs = [t.encode() for t in TEXTS]
    batched = kimi_tok.encode_batch_list(docs)
    singles = [kimi_tok.encode(d).tolist() for d in docs]
    assert batched == singles


def test_load_from_model_path_and_dir(kimi_reference):
    """A tiktoken.model path or its directory load like the repo id."""
    model_path = hf_file("moonshotai/Kimi-K2.6", "tiktoken.model")
    hf_file("moonshotai/Kimi-K2.6", "tokenizer_config.json")
    text = TEXTS[2]
    expected = kimi_reference.encode(text, allowed_special=set())
    for source in (model_path, model_path.parent):
        tok = Tokenizer(source)
        assert tok.encode(text).tolist() == expected
