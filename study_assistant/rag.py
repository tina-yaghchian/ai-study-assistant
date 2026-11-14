"""Retrieval-augmented generation: the study features built on top of the store and the LLM."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import prompts
from .documents import Document
from .formatting import normalize_math
from .store import SearchResult, StudyStore


class ChatModel(Protocol):
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...
    def complete(self, messages: list[dict[str, str]]) -> str: ...
    def complete_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class Answer:
    question: str
    sources: list[SearchResult]
    stream: Iterator[str]


@dataclass
class QuizQuestion:
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    page: int | None = None


@dataclass
class Quiz:
    questions: list[QuizQuestion] = field(default_factory=list)


def format_sources(results: list[SearchResult]) -> str:
    """Render retrieved chunks as numbered sources the model can cite."""
    return "\n\n".join(
        f"[{i}] ({r.filename}, page {r.page})\n{r.text}" for i, r in enumerate(results, start=1)
    )


def material_from_document(document: Document, budget: int) -> str:
    """Fit a whole lecture into ``budget`` characters, sampling every page evenly.

    Truncating at the front would only ever summarise the first few slides; giving each
    page a fair share keeps the notes representative of the entire lecture.
    """
    pages = document.pages
    if document.char_count <= budget:
        return "\n\n".join(f"[Page {p.number}] {p.text}" for p in pages)

    per_page = max(budget // len(pages), 200)
    parts, used = [], 0
    for p in pages:
        if used >= budget:
            break
        snippet = p.text[: min(per_page, budget - used)]
        if len(snippet) < len(p.text):
            snippet = snippet.rsplit(" ", 1)[0] + " ..."
        parts.append(f"[Page {p.number}] {snippet}")
        used += len(snippet)
    return "\n\n".join(parts)


def material_from_results(results: list[SearchResult]) -> str:
    ordered = sorted(results, key=lambda r: (r.filename, r.page))
    return "\n\n".join(f"[Page {r.page}] {r.text}" for r in ordered)


class StudyAssistant:
    def __init__(self, store: StudyStore, llm: ChatModel, top_k: int = 4, max_context_chars: int = 12000):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------ Q&A
    def retrieval_query(self, question: str, history: list[dict[str, str]]) -> str:
        """Short follow-ups ("why?", "give an example") borrow the previous question for context."""
        previous = [m["content"] for m in history if m["role"] == "user"]
        if previous and len(question.split()) < 6:
            return f"{previous[-1]} {question}"
        return question

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        doc_ids: list[str] | None = None,
        k: int | None = None,
    ) -> Answer:
        history = history or []
        sources = self.store.search(self.retrieval_query(question, history), k or self.top_k, doc_ids)
        context = format_sources(sources) if sources else "(no matching passages were found)"
        messages = [{"role": "system", "content": prompts.TUTOR_SYSTEM}]
        # Keep the last few turns so follow-up questions make sense, without blowing the context.
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
        messages.append({"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"})
        return Answer(question=question, sources=sources, stream=self.llm.stream(messages))

    # ------------------------------------------------------------------ material
    def _material(self, doc_id: str, topic: str | None) -> str:
        if topic and topic.strip():
            hits = self.store.search(topic, k=max(self.top_k * 2, 8), doc_ids=[doc_id])
            if hits:
                return material_from_results(hits)
        return material_from_document(self.store.get_document(doc_id), self.max_context_chars)

    # ------------------------------------------------------------------ notes
    def study_notes(self, doc_id: str, topic: str | None = None) -> Iterator[str]:
        document = self.store.get_document(doc_id)
        title = document.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
        user = prompts.NOTES_TEMPLATE.format(
            title=title, focus=prompts.focus_line(topic), material=self._material(doc_id, topic)
        )
        return self.llm.stream([
            {"role": "system", "content": prompts.NOTES_SYSTEM},
            {"role": "user", "content": user},
        ])

    # ------------------------------------------------------------------ quiz
    def quiz(
        self,
        doc_id: str,
        count: int = 5,
        difficulty: str = "medium",
        topic: str | None = None,
        rng: random.Random | None = None,
    ) -> Quiz:
        user = prompts.QUIZ_TEMPLATE.format(
            count=count,
            difficulty=difficulty,
            focus=prompts.focus_line(topic),
            material=self._material(doc_id, topic),
        )
        data = self.llm.complete_json(
            [{"role": "system", "content": prompts.QUIZ_SYSTEM}, {"role": "user", "content": user}],
            prompts.QUIZ_SCHEMA,
        )
        quiz = parse_quiz(data, limit=count)
        # Models love putting the right answer first; shuffle so "always pick A" doesn't work.
        rng = rng or random.Random()
        for q in quiz.questions:
            correct = q.options[q.answer_index]
            rng.shuffle(q.options)
            q.answer_index = q.options.index(correct)
        return quiz


def parse_quiz(data: dict[str, Any], limit: int | None = None) -> Quiz:
    """Validate model output defensively; drop malformed questions instead of crashing."""
    questions: list[QuizQuestion] = []
    for item in data.get("questions", []) if isinstance(data, dict) else []:
        try:
            # Inline \frac is tiny next to a radio button; \dfrac keeps fractions readable.
            options = [normalize_math(o).replace("\\frac", "\\dfrac") for o in item["options"]][:4]
            answer = int(item["answer_index"])
            question = normalize_math(item["question"])
            if len(options) != 4 or not all(options) or not 0 <= answer < 4 or not question:
                continue
            page = item.get("page")
            questions.append(
                QuizQuestion(
                    question=question,
                    options=options,
                    answer_index=answer,
                    explanation=normalize_math(item.get("explanation", "")),
                    page=int(page) if isinstance(page, (int, float)) and page > 0 else None,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return Quiz(questions=questions[:limit] if limit else questions)
