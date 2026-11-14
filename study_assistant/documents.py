"""PDF text extraction and cleanup."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentError(Exception):
    """Raised when a document cannot be read or contains no usable text."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based page number, as a student would see it in a PDF viewer
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    filename: str
    pages: list[Page]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


_REPLACEMENTS = {
    "\ufffd": "",  # replacement character from broken encodings
    "\u2212": "-",  # minus sign
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u00a0": " ",  # non-breaking space
    "\ufb01": "fi",  # ligatures that PDFs love
    "\ufb02": "fl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def clean_text(text: str) -> str:
    """Normalise messy PDF extraction output into readable plain text."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)

    # Re-join words hyphenated across a line break: "experi-\nment" -> "experiment".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Drop figure/image attribution lines that add noise to retrieval.
    text = re.sub(r"(?im)^\s*(image source|figure source|source):.*$", "", text)

    # Split words glued together by extraction ("lectureNotes" -> "lecture Notes").
    # Digits are left alone so that things like x2, H2O and CS101 survive.
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def file_id(data: bytes) -> str:
    """Stable id for a file's contents, so re-uploading the same PDF is a no-op."""
    return hashlib.sha256(data).hexdigest()[:16]


def load_pdf(data: bytes, filename: str) -> Document:
    """Extract and clean the text of every page of a PDF."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - pypdf raises several types here
                raise DocumentError(f"{filename} is password protected.") from exc
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise DocumentError(f"{filename} could not be read as a PDF ({exc}).") from exc

    pages = [
        Page(number=i + 1, text=cleaned)
        for i, raw in enumerate(raw_pages)
        if (cleaned := clean_text(raw))
    ]
    if not pages:
        raise DocumentError(
            f"No text found in {filename}. It may be a scanned document; "
            "try running it through OCR first."
        )
    return Document(doc_id=file_id(data), filename=filename, pages=pages)
