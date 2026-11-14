import pytest

from study_assistant.documents import DocumentError, clean_text, file_id, load_pdf
from tests.conftest import make_pdf


def test_clean_text_normalises_symbols_and_whitespace():
    assert clean_text("a \u2212 b\u00a0=\u00a0c\n\n  next   line") == "a - b = c next line"


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("an experi-\nment") == "an experiment"


def test_clean_text_splits_glued_words_but_keeps_math_tokens():
    assert clean_text("lectureNotes on x2 and H2O in CS101") == "lecture Notes on x2 and H2O in CS101"


def test_clean_text_drops_image_credits():
    assert clean_text("Ellipse definition\nImage source: wikipedia.org\nFoci") == "Ellipse definition Foci"


def test_file_id_is_stable_and_content_based():
    assert file_id(b"abc") == file_id(b"abc")
    assert file_id(b"abc") != file_id(b"abd")


def test_load_pdf_extracts_pages_with_numbers():
    data = make_pdf(["Hello lecture one.", "", "Third page text."])
    doc = load_pdf(data, "week1.pdf")
    assert doc.filename == "week1.pdf"
    assert [p.number for p in doc.pages] == [1, 3]  # blank page skipped, numbering preserved
    assert "Hello lecture one." in doc.pages[0].text


def test_load_pdf_rejects_garbage():
    with pytest.raises(DocumentError):
        load_pdf(b"not a pdf at all", "bad.pdf")


def test_load_pdf_rejects_pdf_without_text():
    with pytest.raises(DocumentError, match="No text found"):
        load_pdf(make_pdf([""]), "scan.pdf")
