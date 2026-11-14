import pytest

from study_assistant.config import Settings
from study_assistant.llm import ThinkFilter, strip_thinking


def test_strip_thinking_removes_reasoning():
    assert strip_thinking("<think>hmm\nlet me see</think>\n\nFinal answer") == "Final answer"
    assert strip_thinking("Answer <think>unfinished") == "Answer"


@pytest.mark.parametrize(
    "pieces",
    [
        ["<think>secret</think>Hello", " world"],
        ["<th", "ink>sec", "ret</thi", "nk>Hel", "lo world"],
        ["Hello", " world"],
    ],
)
def test_think_filter_handles_tags_split_across_tokens(pieces):
    f = ThinkFilter()
    out = "".join(f.feed(p) for p in pieces) + f.flush()
    assert out == "Hello world"


def test_settings_validate(monkeypatch):
    monkeypatch.setenv("STUDY_CHUNK_OVERLAP", "5000")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("STUDY_CHUNK_OVERLAP", "abc")
    with pytest.raises(ValueError, match="integer"):
        Settings()
