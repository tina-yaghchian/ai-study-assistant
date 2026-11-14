"""Split documents into overlapping, page-aware chunks for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'$\\])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    index: int
    text: str

    @property
    def metadata(self) -> dict[str, str | int]:
        return {"doc_id": self.doc_id, "filename": self.filename, "page": self.page, "index": self.index}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _hard_wrap(sentence: str, size: int) -> list[str]:
    """Break a single over-long 'sentence' (tables, formula dumps) on word boundaries."""
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Group sentences into chunks of roughly ``chunk_size`` characters.

    Consecutive chunks share up to ``overlap`` characters of whole sentences so that an
    idea which straddles a boundary is still retrievable from either side.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    sentences: list[str] = []
    for sentence in split_sentences(text):
        sentences.extend(_hard_wrap(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and length + added > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) + 1 > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, max(carried_len - 1, 0)
            added = len(sentence) + (1 if current else 0)
        current.append(sentence)
        length += added
    if current:
        last = " ".join(current)
        # Avoid emitting a final chunk that is nothing but overlap.
        if not chunks or not chunks[-1].endswith(last):
            chunks.append(last)
    return chunks


def chunk_document(document: Document, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        for text in chunk_text(page.text, chunk_size, overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-{index:05d}",
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page=page.number,
                    index=index,
                    text=text,
                )
            )
    return chunks
