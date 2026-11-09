import re
from functools import lru_cache

import tiktoken

# Separators from coarse to fine; the last resort is a hard cut by tokens.
_SEPARATORS = (r"(\n\n)", r"(\n)", r"(?<=[.!?…])(\s+)")


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)  # collapse spaces/tabs, keep newlines
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_keep_sep(text: str, pattern: str) -> list[str]:
    """Split by pattern, keeping each separator glued to the end of the preceding part."""
    parts = re.split(pattern, text)
    out = [
        parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        for i in range(0, len(parts), 2)
    ]
    return [p for p in out if p]


def _split_by_tokens(text: str, limit: int) -> list[str]:
    enc = _encoding()
    tokens = enc.encode(text)
    pieces = (enc.decode(tokens[i : i + limit]) for i in range(0, len(tokens), limit))
    return [p for p in pieces if p.strip()]


def _atoms(text: str, limit: int, level: int = 0) -> list[str]:
    """Recursively split text into pieces of at most `limit` tokens."""
    if count_tokens(text) <= limit:
        return [text]
    if level >= len(_SEPARATORS):
        return _split_by_tokens(text, limit)
    parts = _split_keep_sep(text, _SEPARATORS[level])
    if len(parts) <= 1:  # this separator does not occur, go finer
        return _atoms(text, limit, level + 1)
    out: list[str] = []
    for part in parts:
        out.extend(_atoms(part, limit, level + 1))
    return out


def _token_tail(text: str, overlap: int) -> str:
    """Last ~`overlap` tokens of text, trimmed to start at a word boundary."""
    enc = _encoding()
    tail = enc.decode(enc.encode(text)[-overlap:]).lstrip("�")
    _head, sep, rest = tail.partition(" ")
    # drop a (probably) cut-off first word, unless that would leave nothing
    return rest if sep and rest.strip() else tail


def _overlap_tail(
    atoms: list[str], counts: list[int], overlap: int
) -> tuple[list[str], list[int]]:
    """Trailing context of the previous chunk worth at most `overlap` tokens.

    Prefers whole trailing atoms; if even the last atom is bigger than `overlap`
    (e.g. a long paragraph), falls back to a token-level tail of it.
    """
    if overlap <= 0:
        return [], []
    tail: list[str] = []
    tail_counts: list[int] = []
    total = 0
    for atom, n in zip(reversed(atoms), reversed(counts), strict=True):
        if total + n > overlap:
            break
        tail.insert(0, atom)
        tail_counts.insert(0, n)
        total += n
    if not tail:
        piece = _token_tail(atoms[-1], overlap)
        n = count_tokens(piece)
        if piece.strip() and n <= overlap:
            return [piece], [n]
    return tail, tail_counts


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` tokens (cl100k_base).

    Splits recursively by paragraphs, lines, sentences, then raw tokens; small pieces are
    packed together up to `chunk_size`, and consecutive chunks share ~`overlap` tokens.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = _normalize(text)
    if not text:
        return []

    # Atoms are capped so that an overlap prefix plus any atom still fits in a chunk.
    atoms = _atoms(text, chunk_size - overlap)
    counts = [count_tokens(a) for a in atoms]

    chunks: list[str] = []
    cur: list[str] = []
    cur_counts: list[int] = []
    fresh = 0  # tokens in `cur` that are not just carried-over overlap

    for atom, n in zip(atoms, counts, strict=True):
        if cur and sum(cur_counts) + n > chunk_size:
            chunks.append("".join(cur))
            cur, cur_counts = _overlap_tail(cur, cur_counts, overlap)
            fresh = 0
            # carried-over tail + new atom must still fit
            while cur and sum(cur_counts) + n > chunk_size:
                cur.pop(0)
                cur_counts.pop(0)
        cur.append(atom)
        cur_counts.append(n)
        fresh += n

    if cur and fresh > 0:
        chunks.append("".join(cur))

    return [c for c in (c.strip() for c in chunks) if c]
