"""Persistent document library: extracted pages on disk plus a ChromaDB vector index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking import Chunk, chunk_document
from .documents import Document, Page


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    filename: str
    pages: int
    chunks: int
    chars: int
    added_at: str


@dataclass(frozen=True)
class SearchResult:
    text: str
    doc_id: str
    filename: str
    page: int
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


class StudyStore:
    """Keeps uploaded lectures across app restarts so nothing is re-embedded needlessly."""

    def __init__(
        self,
        data_dir: Path,
        collection_name: str = "lecture_notes",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        embedding_function: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.docs_dir = Path(data_dir) / "documents"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._client = client or chromadb.PersistentClient(
            path=str(Path(data_dir) / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        kwargs: dict[str, Any] = {"name": collection_name, "configuration": {"hnsw": {"space": "cosine"}}}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(**kwargs)

    # ------------------------------------------------------------------ documents
    def _doc_path(self, doc_id: str) -> Path:
        return self.docs_dir / f"{doc_id}.json"

    def has_document(self, doc_id: str) -> bool:
        return self._doc_path(doc_id).exists()

    def add_document(self, document: Document) -> DocumentInfo:
        """Index a document. Adding the same file twice returns the existing entry."""
        if self.has_document(document.doc_id):
            return self._info(document.doc_id)

        chunks = chunk_document(document, self.chunk_size, self.chunk_overlap)
        self._index(chunks)
        record = {
            "doc_id": document.doc_id,
            "filename": document.filename,
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunks": len(chunks),
            "pages": [{"number": p.number, "text": p.text} for p in document.pages],
        }
        self._doc_path(document.doc_id).write_text(json.dumps(record), encoding="utf-8")
        return self._info(document.doc_id)

    def _index(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    def _load(self, doc_id: str) -> dict[str, Any]:
        return json.loads(self._doc_path(doc_id).read_text(encoding="utf-8"))

    def _info(self, doc_id: str) -> DocumentInfo:
        record = self._load(doc_id)
        return DocumentInfo(
            doc_id=record["doc_id"],
            filename=record["filename"],
            pages=len(record["pages"]),
            chunks=record["chunks"],
            chars=sum(len(p["text"]) for p in record["pages"]),
            added_at=record["added_at"],
        )

    def list_documents(self) -> list[DocumentInfo]:
        infos = [self._info(path.stem) for path in self.docs_dir.glob("*.json")]
        return sorted(infos, key=lambda info: info.added_at)

    def get_document(self, doc_id: str) -> Document:
        record = self._load(doc_id)
        pages = [Page(number=p["number"], text=p["text"]) for p in record["pages"]]
        return Document(doc_id=record["doc_id"], filename=record["filename"], pages=pages)

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
        self._doc_path(doc_id).unlink(missing_ok=True)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[SearchResult]:
        if not query.strip() or self._collection.count() == 0:
            return []
        where: dict[str, Any] | None = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}
        result = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchResult] = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            hits.append(
                SearchResult(
                    text=text,
                    doc_id=str(meta["doc_id"]),
                    filename=str(meta["filename"]),
                    page=int(meta["page"]),
                    score=round(1.0 - float(distance), 4),
                )
            )
        return hits
