"""Recursive character chunking with heading-aware metadata.

Walks the document paragraph by paragraph, starts a new chunk at each
Markdown heading, packs the rest into chunks of about ``chunk_size``
characters, carries ``overlap`` characters between consecutive chunks, and
hard-splits any oversized paragraph on finer separators.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

_FINE_SEPARATORS = ["\n", ". ", " "]
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class Chunk(BaseModel):
    text: str
    chunk_index: int
    section: str | None = None


def _hard_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break an oversized block down using progressively finer separators."""
    if len(text) <= chunk_size or not separators:
        return [text]
    sep, *rest = separators
    out: list[str] = []
    buf = ""
    for part in text.split(sep):
        candidate = f"{buf}{sep}{part}" if buf else part
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            out.extend(_hard_split(part, chunk_size, rest) if len(part) > chunk_size else [part])
            buf = ""
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Chunk ``text`` into overlapping, heading-tagged :class:`Chunk` records."""
    normalised = text.replace("\r\n", "\n").strip()
    if not normalised:
        return []

    # (body, section, is_heading) for every paragraph-sized block
    blocks: list[tuple[str, str | None, bool]] = []
    current_heading: str | None = None
    for para in re.split(r"\n\s*\n", normalised):
        para = para.strip()
        if not para:
            continue
        heading_match = _HEADING_RE.match(para.splitlines()[0].strip())
        if heading_match:
            current_heading = heading_match.group(2).strip()
            blocks.append((para, current_heading, True))
            continue
        for piece in _hard_split(para, chunk_size, _FINE_SEPARATORS):
            blocks.append((piece.strip(), current_heading, False))

    chunks: list[Chunk] = []
    buf = ""
    buf_section: str | None = None
    min_fill = max(1, int(chunk_size * 0.4))

    def flush() -> None:
        nonlocal buf
        if buf:
            chunks.append(Chunk(text=buf, chunk_index=len(chunks), section=buf_section))
            buf = ""

    for body, section, is_heading in blocks:
        if is_heading and buf:
            flush()
        candidate = f"{buf}\n\n{body}".strip() if buf else body
        if buf and len(buf) >= min_fill and len(candidate) > chunk_size:
            tail = buf[-overlap:] if overlap > 0 else ""
            flush()
            buf = f"{tail}\n\n{body}".strip() if tail else body
            buf_section = section
        else:
            if not buf:
                buf_section = section
            buf = candidate
    flush()
    return chunks
