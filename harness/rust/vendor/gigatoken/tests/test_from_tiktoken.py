import pytest
import tiktoken
from conftest import tiktoken_vocab_path

import gigatoken
from gigatoken.gigatoken_rs import BPETokenizer


@pytest.fixture
def r50k(r50k_tiktoken_path) -> tuple[tiktoken.Encoding, BPETokenizer]:
    """Return (tiktoken_encoding, BPETokenizer) pair for r50k_base."""
    tt = tiktoken.get_encoding("r50k_base")
    bpe = BPETokenizer.from_tiktoken(r50k_tiktoken_path, "gpt2", {"<|endoftext|>": 50256})
    return tt, bpe


def _assert_same(tt_enc, bpe_tok, text: str):
    expected = tt_enc.encode(text)
    actual = bpe_tok.encode(text.encode("utf-8")).tolist()
    assert actual == expected, f"Mismatch for {text!r}:\n  tiktoken: {expected}\n  gigatoken:    {actual}"


SIMPLE_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "",
    " ",
    "   leading and trailing spaces   ",
]

UNICODE_STRINGS = [
    "café résumé naïve",
    "日本語テスト",
    "emoji: 🚀🌍🎉",
    "mixed: hello 世界 🌎",
    "Ñoño año",
]

CODE_STRINGS = [
    "def foo(x: int) -> int:\n    return x + 1\n",
    "import os\nos.path.join('a', 'b')",
    "if __name__ == '__main__':\n    print('hello')",
    "SELECT * FROM users WHERE id = 1;",
]

EDGE_CASE_STRINGS = [
    "\n",
    "\n\n\n",
    "\t\t",
    "a" * 1000,
    "hello " * 200,
    "a\x00b\x01c",
    "\r\n\r\n",
]

PARAGRAPHS = [
    (
        "The Rust programming language helps you write faster, more reliable software. "
        "High-level ergonomics and low-level control are often at odds in programming language design; "
        "Rust challenges that conflict. Through balancing powerful technical capacity and a great developer "
        "experience, Rust gives you the option to control low-level details (such as memory usage) without "
        "all the hassle traditionally associated with such control."
    ),
    ("```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```\n"),
]


@pytest.mark.parametrize("text", SIMPLE_STRINGS, ids=lambda t: repr(t)[:40])
def test_simple(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", UNICODE_STRINGS, ids=lambda t: repr(t)[:40])
def test_unicode(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", CODE_STRINGS, ids=lambda t: repr(t)[:40])
def test_code(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", EDGE_CASE_STRINGS, ids=lambda t: repr(t)[:40])
def test_edge_cases(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", PARAGRAPHS, ids=lambda t: repr(t)[:40])
def test_paragraphs(r50k, text):
    _assert_same(*r50k, text)


def test_roundtrip_token_count(r50k):
    """Token counts should match between tiktoken and gigatoken."""
    tt, bpe = r50k
    text = "Here is a moderately long sentence with some numbers 42 and symbols @#$%."
    assert len(tt.encode(text)) == len(bpe.encode(text.encode("utf-8")))


def test_single_characters(r50k):
    """Every printable ASCII character should tokenize identically."""
    tt, bpe = r50k
    for c in (chr(i) for i in range(32, 127)):
        _assert_same(tt, bpe, c)


def test_whitespace_variations(r50k):
    tt, bpe = r50k
    for text in ["a b", "a  b", "a   b", "a\tb", "a\nb", "a\r\nb", "a \n b"]:
        _assert_same(tt, bpe, text)


def test_repeated_tokens(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, "the " * 500)
    _assert_same(tt, bpe, "aaaa" * 250)


def test_mixed_scripts(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, "Hello мир 世界 مرحبا")
    _assert_same(tt, bpe, "price: €100 or ¥10000")


def test_json_like(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, '{"key": "value", "num": 123, "arr": [1, 2, 3]}')


def test_url_like(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, "https://example.com/path?query=value&other=123#fragment")


def test_endoftext_added_token(r50k):
    """<|endoftext|> encodes to id 50256 and round-trips, matching both
    tiktoken with specials allowed and the tokenizer.json-loaded GPT-2."""
    tt, bpe = r50k
    text = "Hello world.<|endoftext|>Next document."
    expected = tt.encode(text, allowed_special="all")
    actual = bpe.encode(text.encode("utf-8")).tolist()
    assert actual == expected
    assert 50256 in actual
    assert bpe.decode(actual) == text.encode("utf-8")


def test_multiline_code(r50k):
    tt, bpe = r50k
    code = """class Foo:
    def __init__(self, x):
        self.x = x

    def bar(self):
        return self.x * 2
"""
    _assert_same(tt, bpe, code)


# ---------------------------------------------------------------------------
# gigatoken.Tokenizer.from_tiktoken: which pretokenizer and special tokens a
# rank file gets. The file itself carries neither, so nothing may be guessed
# (issue #40: every vocabulary used to be pretokenized as r50k).
# ---------------------------------------------------------------------------

# Texts the schemes disagree on: r50k splits a leading punctuation character
# off a word and takes digits in twos, cl100k/o200k keep the punctuation and
# take digits in threes; o200k splits letter runs by case.
SCHEME_SENSITIVE = [".data", "(self", "_name", "1234", "123456789", "CamelCaseWord", "xé́y", "  \n  end"]


@pytest.mark.parametrize("encoding", ["r50k_base", "cl100k_base", "o200k_base"])
def test_published_encoding_matches_tiktoken(encoding):
    """Every .tiktoken vocabulary OpenAI publishes loads with its own
    pretokenization scheme and special tokens, identified by file name."""
    ref = tiktoken.get_encoding(encoding)
    tok = gigatoken.Tokenizer.from_tiktoken(tiktoken_vocab_path(encoding))
    for text in SCHEME_SENSITIVE + SIMPLE_STRINGS + UNICODE_STRINGS + CODE_STRINGS + PARAGRAPHS:
        assert tok.encode(text.encode("utf-8")).tolist() == ref.encode_ordinary(text), f"{encoding} mismatch for {text!r}"
    special = "Hello.<|endoftext|>Next."
    assert tok.encode(special.encode("utf-8")).tolist() == ref.encode(special, allowed_special="all")


def test_unnamed_vocab_needs_an_explicit_pretokenizer(tmp_path, r50k_tiktoken_path):
    """A rank file whose name is not a published encoding cannot be resolved,
    and says so instead of falling back to some default scheme."""
    unnamed = tmp_path / "custom.tiktoken"
    unnamed.symlink_to(r50k_tiktoken_path)
    with pytest.raises(ValueError, match="pretokenizer"):
        gigatoken.Tokenizer.from_tiktoken(unnamed)
    with pytest.raises(ValueError, match="unknown pretokenizer scheme"):
        gigatoken.Tokenizer.from_tiktoken(unnamed, "not-a-scheme")
    tok = gigatoken.Tokenizer.from_tiktoken(unnamed, "gpt2")
    ref = tiktoken.get_encoding("r50k_base")
    assert tok.encode(b".data").tolist() == ref.encode_ordinary(".data")
    # Specials are part of the encoding definition, not of the rank file:
    # none are registered unless asked for.
    assert tok._special_tokens() == {}
    tok = gigatoken.Tokenizer.from_tiktoken(unnamed, "gpt2", {"<|endoftext|>": 50256})
    assert tok._special_tokens() == {"<|endoftext|>": 50256}


def test_explicit_pretokenizer_overrides_the_file_name(r50k_tiktoken_path):
    """A named scheme takes precedence over the one the file name implies —
    the same ranks then split differently (r50k takes digits in twos, the
    cl100k scheme in threes)."""
    as_r50k = gigatoken.Tokenizer.from_tiktoken(r50k_tiktoken_path)
    as_cl100k = gigatoken.Tokenizer.from_tiktoken(r50k_tiktoken_path, "cl100k")
    assert as_r50k.encode(b"1234").tolist() != as_cl100k.encode(b"1234").tolist()
    assert as_r50k._special_tokens() == {"<|endoftext|>": 50256}
    assert as_cl100k._special_tokens() == {}
