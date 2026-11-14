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


def test_repair_latex_escapes_restores_commands_that_collide_with_json_escapes():
    import json

    from study_assistant.llm import repair_latex_escapes

    raw = r'{"q": "$\text{proj}_{\boldsymbol{u}} \boldsymbol{v} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|^2}$"}'
    # Without the repair, \t, \b and \f are silently decoded as tab, backspace and form feed.
    assert json.loads(r'{"q": "\boldsymbol{v} = \frac{a}{b}"}')["q"] == "\x08oldsymbol{v} = \x0crac{a}{b}"
    fixed = json.loads(repair_latex_escapes(raw))["q"]
    assert fixed == r"$\text{proj}_{\boldsymbol{u}} \boldsymbol{v} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|^2}$"


def test_repair_latex_escapes_leaves_valid_json_alone():
    import json

    from study_assistant.llm import repair_latex_escapes

    raw = r'{"a": "line one\nline two", "b": "caf\u00e9", "c": "$\\frac{1}{2}$", "d": "say \"hi\""}'
    assert json.loads(repair_latex_escapes(raw)) == json.loads(raw)
