"""Application settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional; settings still work from the real environment.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All tunable knobs in one place. Override any of them with environment variables."""

    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("STUDY_MODEL", "qwen3:8b"))
    temperature: float = field(default_factory=lambda: _float("STUDY_TEMPERATURE", 0.3))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("STUDY_DATA_DIR", "data")))
    collection_name: str = field(default_factory=lambda: os.getenv("STUDY_COLLECTION", "lecture_notes"))
    chunk_size: int = field(default_factory=lambda: _int("STUDY_CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _int("STUDY_CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _int("STUDY_TOP_K", 4))
    # Characters of lecture text sent to the model when generating notes or quizzes.
    max_context_chars: int = field(default_factory=lambda: _int("STUDY_MAX_CONTEXT_CHARS", 12000))
    max_upload_mb: int = field(default_factory=lambda: _int("STUDY_MAX_UPLOAD_MB", 50))

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("STUDY_CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("STUDY_CHUNK_OVERLAP must be >= 0 and smaller than STUDY_CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("STUDY_TOP_K must be positive")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


def get_settings() -> Settings:
    return Settings()
