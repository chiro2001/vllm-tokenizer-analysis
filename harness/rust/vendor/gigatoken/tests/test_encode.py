import numpy as np
import pytest
import tiktoken
from pytest import fixture

from gigatoken.gigatoken_rs import BPETokenizer


@fixture
def tiktoken_r50k():
    return tiktoken.get_encoding("r50k_base")


@fixture
def gigatoken_r50k(r50k_tiktoken_path):
    return BPETokenizer.from_tiktoken(r50k_tiktoken_path, "gpt2", {"<|endoftext|>": 50256})


def test_use_gigatoken_model(gigatoken_r50k):
    print(gigatoken_r50k)
    print(gigatoken_r50k.encode(b"Here's a test string"))


def test_decode_int64_array_matches_list(gigatoken_r50k):
    # int64 arrays (e.g. model outputs) take the one-pass checked-cast path;
    # the result must match plain list decoding.
    ids = gigatoken_r50k.encode(b"Here's a test string").tolist()
    assert gigatoken_r50k.decode(np.asarray(ids, dtype=np.int64)) == gigatoken_r50k.decode(ids)


def test_decode_negative_int64_raises(gigatoken_r50k):
    with pytest.raises(OverflowError):
        gigatoken_r50k.decode(np.asarray([1, -2, 3], dtype=np.int64))


def test_encode_batch_list_matches_encode(gigatoken_r50k):
    texts = ["Hello world, this is a test.", "The quick brown fox jumps.", ""]
    base = [gigatoken_r50k.encode(t).tolist() for t in texts]
    assert gigatoken_r50k.encode_batch_list(texts) == base


def test_encode_batch_list_prefix_suffix_truncation(gigatoken_r50k):
    from gigatoken.gigatoken_rs import _WrapTruncate

    def compat(texts, **options):
        return gigatoken_r50k._encode_batch_list_compat(texts, _WrapTruncate(**options))

    texts = ["Hello world, this is a test.", "The quick brown fox jumps."]
    base = [gigatoken_r50k.encode(t).tolist() for t in texts]
    assert all(len(row) > 3 for row in base)
    assert compat(texts, prefix=[7, 8], suffix=[9]) == [[7, 8] + row + [9] for row in base]
    assert compat(texts, max_tokens=3) == [row[:3] for row in base]
    assert compat(texts, max_tokens=3, truncate_left=True) == [row[-3:] for row in base]
    # max_tokens=0 keeps no content ids: rows are exactly prefix+suffix.
    assert compat(texts, prefix=[7], suffix=[9], max_tokens=0) == [[7, 9] for _ in texts]
