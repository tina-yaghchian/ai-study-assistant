"""Thin wrapper around the local Ollama server, with streaming and friendly errors."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import ollama

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(Exception):
    """A user-facing problem talking to the model (server down, model missing, bad output)."""


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks that some models (e.g. Qwen3) emit before the answer."""
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:  # unterminated block: drop everything after it
        text = text.split("<think>", 1)[0]
    return text.strip()


class ThinkFilter:
    """Incrementally hides <think>...</think> sections from a token stream."""

    def __init__(self) -> None:
        self._buffer = ""
        self._thinking = False

    def feed(self, piece: str) -> str:
        self._buffer += piece
        out = []
        while self._buffer:
            tag = "</think>" if self._thinking else "<think>"
            idx = self._buffer.find(tag)
            if idx == -1:
                # Keep a tail that might be the start of a tag split across tokens.
                keep = next((n for n in range(len(tag) - 1, 0, -1) if self._buffer.endswith(tag[:n])), 0)
                emit, self._buffer = self._buffer[: len(self._buffer) - keep], self._buffer[len(self._buffer) - keep :]
                if not self._thinking:
                    out.append(emit)
                break
            if not self._thinking:
                out.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(tag) :]
            self._thinking = not self._thinking
        return "".join(out)

    def flush(self) -> str:
        rest, self._buffer = ("" if self._thinking else self._buffer), ""
        return rest


class OllamaLLM:
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen3:8b", temperature: float = 0.3):
        self.host = host
        self.model = model
        self.temperature = temperature
        self._client = ollama.Client(host=host)

    # ------------------------------------------------------------------ status
    def available_models(self) -> list[str]:
        try:
            return sorted(m.model for m in self._client.list().models if m.model)
        except Exception as exc:  # noqa: BLE001 - connection errors come in many shapes
            raise LLMError(self._connection_hint()) from exc

    def _connection_hint(self) -> str:
        return f"Can't reach Ollama at {self.host}. Start it with `ollama serve` (or open the Ollama app)."

    # ------------------------------------------------------------------ calls
    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        options = {"temperature": self.temperature}
        try:
            return self._client.chat(model=self.model, messages=messages, options=options, think=False, **kwargs)
        except ollama.ResponseError as exc:
            if "think" in str(exc).lower():  # model without a thinking mode
                return self._client.chat(model=self.model, messages=messages, options=options, **kwargs)
            if exc.status_code == 404:
                raise LLMError(
                    f"Model '{self.model}' isn't installed. Run `ollama pull {self.model}` and try again."
                ) from exc
            raise LLMError(f"Ollama returned an error: {exc.error}") from exc
        except (ConnectionError, OSError) as exc:
            raise LLMError(self._connection_hint()) from exc
        except Exception as exc:  # httpx.ConnectError and friends
            if "connect" in type(exc).__name__.lower() or "connect" in str(exc).lower():
                raise LLMError(self._connection_hint()) from exc
            raise

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the answer piece by piece, with any reasoning blocks filtered out."""
        filt = ThinkFilter()
        try:
            for part in self._chat(messages, stream=True):
                text = filt.feed(part.message.content or "")
                if text:
                    yield text
        except LLMError:
            raise
        except Exception as exc:  # stream dropped mid-way
            raise LLMError(f"The model stopped responding: {exc}") from exc
        tail = filt.flush()
        if tail:
            yield tail

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._chat(messages)
        return strip_thinking(response.message.content or "")

    def complete_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        """Ask for output matching a JSON schema (Ollama structured outputs)."""
        response = self._chat(messages, format=schema)
        raw = strip_thinking(response.message.content or "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned malformed JSON. Try again or pick a different model.") from exc
