"""Tests for the OS-agnostic injection helpers that don't need native APIs."""

from wisperfree.inject.macos import _utf16_chunks


def _units(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def test_utf16_chunks_reassemble_exactly():
    text = "The quick brown fox jumps over the lazy dog, twice over again now."
    assert "".join(_utf16_chunks(text, 20)) == text


def test_utf16_chunks_respect_unit_cap():
    text = "a" * 105
    chunks = list(_utf16_chunks(text, 20))
    assert all(_units(c) <= 20 for c in chunks)
    assert "".join(chunks) == text


def test_utf16_chunks_never_split_surrogate_pairs():
    # Emoji are non-BMP: 2 UTF-16 units, 1 Python code point each.
    text = "ok " + "😀" * 15  # 15 emoji = 30 units
    chunks = list(_utf16_chunks(text, 20))
    assert "".join(chunks) == text
    for c in chunks:
        assert _units(c) <= 20
        # each chunk must be independently encodable (no lone surrogate)
        c.encode("utf-16")  # would raise on a split surrogate


def test_utf16_chunks_single_oversized_grapheme_still_emitted():
    # A cap smaller than one code point still yields that code point.
    assert list(_utf16_chunks("😀", 1)) == ["😀"]
