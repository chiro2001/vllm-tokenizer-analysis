"""TiktokenCompat: a gigatoken-backed drop-in for the `tiktoken.Encoding` API."""

from __future__ import annotations

import functools
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from gigatoken._tokenizer import Tokenizer
from gigatoken.gigatoken_rs import SpecialTokenFound, _SubstringMatcher, _WrapTruncate

if TYPE_CHECKING:
    from typing import NoReturn

    import numpy as np
    import numpy.typing as npt

_ENDOFTEXT = "<|endoftext|>"


def _as_list(text: Iterable[str]) -> list[str]:
    """Pass lists through as-is; materialize any other iterable of str."""
    return text if isinstance(text, list) else list(text)


class TiktokenCompat:
    """Adapt a `gigatoken.Tokenizer` to the `tiktoken.Encoding` API, so it can
    replace a tiktoken encoding in existing code: `encode`/`encode_ordinary`
    (and their batch variants), `decode`/`decode_bytes`/`decode_batch`,
    single-token conversions, and the vocab/special-token accessors.

    Obtain one with `gigatoken.Tokenizer.from_tiktoken(path).as_tiktoken()` (or
    `gigatoken.Tokenizer(source).as_tiktoken()` for any other source).

    One semantic difference is surfaced loudly instead of silently diverging:
    the gigatoken backend always recognizes special tokens while encoding, so
    text that contains a special token which tiktoken would encode as
    ordinary text (encode_ordinary, or encode with the special neither
    allowed nor disallowed) raises NotImplementedError. The common paths —
    encode() rejecting disallowed specials, and allowed_special="all" — match
    tiktoken exactly.
    """

    def __init__(self, tokenizer: Tokenizer, name: str = "gigatoken") -> None:
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(
                f"TiktokenCompat wraps a gigatoken.Tokenizer, not {type(tokenizer).__name__!r}; construct "
                "one first, e.g. gigatoken.Tokenizer.from_tiktoken(path).as_tiktoken()"
            )
        self.name = name
        self._tokenizer = tokenizer
        self._special_tokens: dict[str, int] = tokenizer._special_tokens()
        self._specials_set: frozenset[str] = frozenset(self._special_tokens)
        # Prebuilt encode options, each carrying an Aho-Corasick matcher over
        # the specials that a call must reject, keyed by that scan set; the
        # scan itself runs inside the backend's encode call, in parallel over
        # documents with the GIL released, instead of one Python-level
        # substring search per special token per document.
        self._matcher_cache: dict[frozenset[str], tuple[_WrapTruncate, list[str]]] = {}

    @property
    def tokenizer(self) -> Tokenizer:
        """The underlying gigatoken Tokenizer (numpy/awkward-native API)."""
        return self._tokenizer

    # -- encoding -----------------------------------------------------------

    def _forbid_matcher(
        self,
        allowed_special: Literal["all"] | Collection[str],
        disallowed_special: Literal["all"] | Collection[str],
    ) -> tuple[_WrapTruncate | None, list[str], set[str]]:
        """The encode options carrying a compiled matcher over the specials
        this call must reject — disallowed ones (ValueError, like tiktoken)
        and not-allowed ones (NotImplementedError, see the class docstring)
        — plus the matcher's pattern list and the disallowed set. The
        options are None when nothing can raise (e.g.
        allowed_special="all"), which skips the scan entirely. Cached per
        distinct scan set."""
        allowed = self._specials_set if allowed_special == "all" else set(allowed_special)
        disallowed = self._specials_set - allowed if disallowed_special == "all" else set(disallowed_special)
        scan = frozenset(s for s in self._specials_set if s in disallowed or s not in allowed)
        if not scan:
            return None, [], disallowed
        entry = self._matcher_cache.get(scan)
        if entry is None:
            # Callers use a handful of distinct scan sets in practice; if one
            # cycles through many, start over rather than grow forever.
            if len(self._matcher_cache) > 32:
                self._matcher_cache.clear()
            patterns = sorted(scan)
            entry = (_WrapTruncate(forbid=_SubstringMatcher(patterns)), patterns)
            self._matcher_cache[scan] = entry
        return entry[0], entry[1], disallowed

    def _raise_specials_found(self, present: set[str], disallowed: set[str]) -> NoReturn:
        # `from None`: the internal SpecialTokenFound is an implementation
        # detail; without it the traceback shows a confusing "During handling
        # of the above exception" chain.
        bad = sorted(present & disallowed)
        if bad:
            raise ValueError(
                f"Encountered text corresponding to disallowed special token {bad[0]!r}.\n"
                f"If you want this text to be encoded as a special token, pass it to `allowed_special`, "
                f"e.g. `allowed_special={{{bad[0]!r}, ...}}`.\n"
                f"If you want this text to be encoded as normal text, disable the check for this token "
                f"by passing `disallowed_special=(enc.special_tokens_set - {{{bad[0]!r}}})`."
            ) from None
        ordinary = sorted(present)
        raise NotImplementedError(
            f"gigatoken always recognizes special tokens while encoding, so {ordinary[0]!r} cannot be "
            "encoded as ordinary text; pass it in allowed_special (to encode it as a special token) "
            "or leave it disallowed (to reject it, tiktoken's default)"
        ) from None

    def _encode_list(
        self,
        texts: list[str],
        allowed_special: Literal["all"] | Collection[str],
        disallowed_special: Literal["all"] | Collection[str],
        parallel: bool | None,
    ) -> list[list[int]]:
        """One fused Rust call: scan every document for the specials this
        call must reject (in parallel, GIL released) and encode, returning
        plain lists built in Rust. Only the error path runs in Python."""
        options, patterns, disallowed = self._forbid_matcher(allowed_special, disallowed_special)
        if options is None:
            return self._tokenizer.encode_batch_list(texts, parallel=parallel)
        try:
            return self._tokenizer._encode_batch_list_compat(texts, options, parallel=parallel)
        except SpecialTokenFound as e:
            self._raise_specials_found({patterns[i] for i in e.args[0]}, disallowed)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ignoring special tokens; raises if the text contains one
        (see the class docstring)."""
        return self._encode_list([text], (), (), parallel=None)[0]

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | Collection[str] = (),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        return self._encode_list([text], allowed_special, disallowed_special, parallel=None)[0]

    def encode_ordinary_batch(self, text: list[str], num_threads: int = 8) -> list[list[int]]:
        """`num_threads` is accepted for signature compatibility; the Rust
        backend manages its own parallelism, except that `num_threads=1`
        forces single-threaded encoding (see encode_batch)."""
        return self._encode_list(_as_list(text), (), (), parallel=False if num_threads == 1 else None)

    def encode_batch(
        self,
        text: list[str],
        num_threads: int = 8,
        *,
        allowed_special: Literal["all"] | Collection[str] = (),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[list[int]]:
        """`num_threads` is accepted for signature compatibility; the Rust
        backend manages its own parallelism. `num_threads=1` forces
        single-threaded encoding; any other value leaves the choice to the
        backend (parallel, except inside a multiprocessing worker)."""
        return self._encode_list(_as_list(text), allowed_special, disallowed_special, parallel=False if num_threads == 1 else None)

    def encode_single_token(self, text_or_bytes: str | bytes) -> int:
        piece = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
        return self._id_by_bytes[piece]

    def encode_with_unstable(self, *args: object, **kwargs: object) -> tuple[list[int], list[list[int]]]:
        raise NotImplementedError("gigatoken.TiktokenCompat does not support encode_with_unstable")

    # -- decoding -----------------------------------------------------------

    def decode_bytes(self, tokens: Sequence[int] | npt.NDArray[np.uint32]) -> bytes:
        return self._tokenizer.decode(tokens)

    def decode(self, tokens: Sequence[int] | npt.NDArray[np.uint32], errors: str = "replace") -> str:
        return self._tokenizer.decode(tokens).decode("utf-8", errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return self._bytes_by_id[token]

    def decode_tokens_bytes(self, tokens: Sequence[int]) -> list[bytes]:
        return [self._bytes_by_id[int(t)] for t in tokens]

    def decode_batch(self, batch: Sequence[Sequence[int]], errors: str = "replace", num_threads: int = 8) -> list[str]:
        return [self.decode(tokens, errors=errors) for tokens in batch]

    def decode_bytes_batch(self, batch: Sequence[Sequence[int]], num_threads: int = 8) -> list[bytes]:
        return [self.decode_bytes(tokens) for tokens in batch]

    def decode_with_offsets(self, tokens: Sequence[int]) -> tuple[str, list[int]]:
        raise NotImplementedError("gigatoken.TiktokenCompat does not support decode_with_offsets")

    # -- vocab and special tokens ---------------------------------------------

    @functools.cached_property
    def _bytes_by_id(self) -> dict[int, bytes]:
        return self._tokenizer.vocab

    @functools.cached_property
    def _id_by_bytes(self) -> dict[bytes, int]:
        return {b: i for i, b in self._bytes_by_id.items()}

    def token_byte_values(self) -> list[bytes]:
        """Byte values of the mergeable ranks (excluding special tokens)."""
        special_ids = set(self._special_tokens.values())
        return [b for i, b in sorted(self._bytes_by_id.items()) if i not in special_ids]

    @property
    def special_tokens_set(self) -> set[str]:
        return set(self._specials_set)

    def is_special_token(self, token: int) -> bool:
        return token in set(self._special_tokens.values())

    @property
    def eot_token(self) -> int:
        return self._special_tokens[_ENDOFTEXT]

    @property
    def n_vocab(self) -> int:
        """For backwards compatibility. Prefer `max_token_value + 1`."""
        return self.max_token_value + 1

    @property
    def max_token_value(self) -> int:
        return self._tokenizer.vocab_size - 1

    def __repr__(self) -> str:
        return f"TiktokenCompat({self._tokenizer!r})"
