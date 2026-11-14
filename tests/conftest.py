"""Shared fixtures: a tiny PDF builder, an offline embedding function and a fake LLM."""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_assistant.store import StudyStore  # noqa: E402


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page (no extra deps)."""
    objects: list[bytes] = []
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 11 Tf 40 750 Td ({safe}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


class HashingEmbedding(EmbeddingFunction[Documents]):
    """Bag-of-words hashing embeddings: deterministic, offline, good enough for tests."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chroma's signature
        vectors = []
        for text in input:
            vec = np.zeros(256, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm else vec)
        return vectors

    @staticmethod
    def name() -> str:
        return "test-hashing"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> HashingEmbedding:
        return HashingEmbedding()


class FakeLLM:
    """Records the prompts it receives and replies with canned text."""

    def __init__(self, reply: str = "Answer [1]", json_reply: dict | None = None) -> None:
        self.reply = reply
        self.json_reply = json_reply or {"questions": []}
        self.calls: list[list[dict[str, str]]] = []

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.calls.append(messages)
        yield from self.reply.split(" ")

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply

    def complete_json(self, messages: list[dict[str, str]], schema: dict) -> dict:
        self.calls.append(messages)
        return self.json_reply


@pytest.fixture
def store(tmp_path: Path) -> StudyStore:
    client = chromadb.EphemeralClient()
    name = f"test_{tmp_path.name}"[:60]
    return StudyStore(tmp_path, collection_name=name, chunk_size=200, chunk_overlap=40,
                      embedding_function=HashingEmbedding(), client=client)
