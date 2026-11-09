import itertools

import pytest

from app.services.chunking import count_tokens, split_text

CHUNK = 60
OVERLAP = 10


def _long_text(paragraphs: int = 30) -> str:
    return "\n\n".join(
        f"Paragraph {i}. The quick brown fox jumps over the lazy dog. "
        f"Sentence number {i} keeps the text going for a while."
        for i in range(paragraphs)
    )


def test_empty_text_returns_empty_list():
    assert split_text("", CHUNK, OVERLAP) == []
    assert split_text("   \n\n \t ", CHUNK, OVERLAP) == []


def test_short_text_is_single_chunk():
    assert split_text("Hello   world", CHUNK, OVERLAP) == ["Hello world"]


def test_long_text_is_split():
    chunks = split_text(_long_text(), CHUNK, OVERLAP)
    assert len(chunks) > 1


def test_no_chunk_exceeds_chunk_size():
    for chunk in split_text(_long_text(), CHUNK, OVERLAP):
        assert count_tokens(chunk) <= CHUNK


def test_oversized_single_word_is_cut_by_tokens():
    chunks = split_text("x" * 5000, CHUNK, OVERLAP)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= CHUNK for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(_long_text(), CHUNK, OVERLAP)
    overlapping = 0
    for prev, nxt in itertools.pairwise(chunks):
        # the start of the next chunk repeats the tail of the previous one
        first_sentence = nxt.split(".")[0].strip()
        if first_sentence and first_sentence in prev:
            overlapping += 1
    assert overlapping > 0


def test_zero_overlap_has_no_repeated_content():
    text = "\n\n".join(
        f"Unique paragraph number {i} with some filler words." for i in range(40)
    )
    chunks = split_text(text, CHUNK, 0)
    joined = "\n\n".join(chunks)
    for i in range(40):
        assert joined.count(f"Unique paragraph number {i} ") == 1


def test_all_content_is_preserved():
    text = _long_text()
    joined = " ".join(split_text(text, CHUNK, OVERLAP))
    for i in range(30):
        assert f"Paragraph {i}." in joined


def test_invalid_params():
    with pytest.raises(ValueError):
        split_text("text", 0, 0)
    with pytest.raises(ValueError):
        split_text("text", 10, 10)
