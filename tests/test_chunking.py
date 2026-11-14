import pytest

from study_assistant.chunking import chunk_document, chunk_text, split_sentences
from study_assistant.documents import Document, Page


def sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} talks about topic {i}." for i in range(n))


def test_split_sentences_keeps_decimals_together():
    assert split_sentences("Pi is 3.14 roughly. Next idea!") == ["Pi is 3.14 roughly.", "Next idea!"]


def test_chunks_respect_size_and_cover_everything():
    text = sentences(40)
    chunks = chunk_text(text, chunk_size=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == text


def test_consecutive_chunks_overlap():
    chunks = chunk_text(sentences(40), chunk_size=200, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(prev)[-1]
        assert nxt.startswith(last_sentence)


def test_overlong_sentence_is_wrapped():
    chunks = chunk_text("word " * 300, chunk_size=100, overlap=0)
    assert len(chunks) > 1 and all(len(c) <= 100 for c in chunks)


def test_short_text_is_single_chunk():
    assert chunk_text("Just one sentence.", 100, 20) == ["Just one sentence."]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


def test_chunk_document_tracks_pages_and_unique_ids():
    doc = Document("abc", "l.pdf", [Page(1, sentences(10)), Page(4, "Final remark.")])
    chunks = chunk_document(doc, chunk_size=200, overlap=0)
    assert {c.page for c in chunks} == {1, 4}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[-1].metadata == {"doc_id": "abc", "filename": "l.pdf", "page": 4, "index": len(chunks) - 1}
