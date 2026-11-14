from study_assistant.documents import Document, Page
from study_assistant.rag import StudyAssistant, material_from_document, parse_quiz
from tests.conftest import FakeLLM

ELLIPSES = Document(
    "doc-ellipse",
    "conics.pdf",
    [
        Page(1, "An ellipse is the set of points whose distances to two foci sum to a constant."),
        Page(2, "A parabola is the set of points equidistant from a focus and a directrix line."),
    ],
)
SORTING = Document("doc-sort", "algorithms.pdf", [Page(1, "Merge sort splits the array in half and merges sorted halves.")])


def test_add_list_get_delete(store):
    info = store.add_document(ELLIPSES)
    assert (info.filename, info.pages) == ("conics.pdf", 2)
    assert store.add_document(ELLIPSES) == info  # idempotent
    assert [d.doc_id for d in store.list_documents()] == ["doc-ellipse"]
    assert store.get_document("doc-ellipse").pages[1].number == 2

    store.delete_document("doc-ellipse")
    assert store.list_documents() == []
    assert store.search("ellipse foci") == []


def test_search_returns_page_citations_and_respects_scope(store):
    store.add_document(ELLIPSES)
    store.add_document(SORTING)

    top = store.search("what is a parabola directrix", k=1)[0]
    assert (top.filename, top.page) == ("conics.pdf", 2)

    scoped = store.search("parabola directrix", k=3, doc_ids=["doc-sort"])
    assert {r.filename for r in scoped} == {"algorithms.pdf"}


def test_ask_builds_numbered_sources_and_streams(store):
    store.add_document(ELLIPSES)
    llm = FakeLLM(reply="An ellipse has two foci [1]")
    answer = StudyAssistant(store, llm, top_k=2).ask("What is an ellipse?")

    assert "".join(answer.stream).replace(" ", "") == "Anellipsehastwofoci[1]"
    prompt = llm.calls[0][-1]["content"]
    assert "[1] (conics.pdf, page" in prompt and "Question: What is an ellipse?" in prompt
    assert answer.sources[0].filename == "conics.pdf"


def test_short_follow_up_reuses_previous_question(store):
    assistant = StudyAssistant(store, FakeLLM())
    history = [{"role": "user", "content": "Explain merge sort"}, {"role": "assistant", "content": "..."}]
    assert assistant.retrieval_query("why?", history) == "Explain merge sort why?"
    long_q = "How does a parabola relate to its directrix line?"
    assert assistant.retrieval_query(long_q, history) == long_q


def test_material_samples_every_page_within_budget():
    doc = Document("d", "big.pdf", [Page(i, ("word " * 400).strip()) for i in range(1, 11)])
    material = material_from_document(doc, budget=3000)
    assert all(f"[Page {i}]" in material for i in range(1, 11))
    assert len(material) < 3600


def test_parse_quiz_drops_malformed_items():
    data = {
        "questions": [
            {"question": "Q1", "options": ["a", "b", "c", "d"], "answer_index": 2, "explanation": "e", "page": 3},
            {"question": "bad", "options": ["a", "b"], "answer_index": 0, "explanation": "", "page": 1},
            {"question": "bad index", "options": ["a", "b", "c", "d"], "answer_index": 7},
            {"options": ["a", "b", "c", "d"], "answer_index": 1},
        ]
    }
    quiz = parse_quiz(data)
    assert len(quiz.questions) == 1 and quiz.questions[0].page == 3
    assert parse_quiz({"oops": 1}).questions == []


def test_quiz_uses_document_material(store):
    store.add_document(ELLIPSES)
    reply = {"questions": [{"question": "Q?", "options": list("abcd"), "answer_index": 0, "explanation": "", "page": 1}]}
    llm = FakeLLM(json_reply=reply)
    quiz = StudyAssistant(store, llm).quiz("doc-ellipse", count=1)
    assert len(quiz.questions) == 1
    q = quiz.questions[0]
    assert q.options[q.answer_index] == "a"  # correct answer tracked through the shuffle
    assert "[Page 2] A parabola" in llm.calls[0][-1]["content"]


def test_parse_quiz_renders_math_in_questions_options_and_explanations():
    data = {
        "questions": [
            {
                # As decoded from JSON where the model wrote single backslashes:
                "question": "What is \text{proj}_{\boldsymbol{u}} \boldsymbol{v}?",
                "options": ["\x0crac{u \\cdot v}{\\|u\\|^2} u", "`u \\times v`", "`x^2`", "The zero vector"],
                "answer_index": 0,
                "explanation": "Use \x0crac{u \\cdot v}{\\|u\\|^2}.",
                "page": 6,
            }
        ]
    }
    q = parse_quiz(data).questions[0]
    assert q.question == r"What is $\text{proj}_{\boldsymbol{u}} \boldsymbol{v}$?"
    assert q.options == [r"$\dfrac{u \cdot v}{\|u\|^2} u$", r"$u \times v$", "$x^2$", "The zero vector"]
    assert q.explanation == r"Use $\frac{u \cdot v}{\|u\|^2}$."


def test_markdown_math_converts_paren_delimiters():
    from study_assistant.formatting import markdown_math

    text = r"Ratio \(\frac{a}{b}\) and \[ e = \frac{c}{a} \] end, with a line break a \\[2pt] b"
    assert markdown_math(text) == "Ratio $\\frac{a}{b}$ and \n$$\ne = \\frac{c}{a}\n$$\n end, with a line break a \\\\[2pt] b"


def test_normalize_math_trims_padding_inside_inline_math():
    from study_assistant.formatting import normalize_math

    assert normalize_math("$ (3, 0) $") == "$(3, 0)$"
    assert normalize_math("Since $ u = (1, 0) $ and $$ x^2 $$ hold") == "Since $u = (1, 0)$ and $$ x^2 $$ hold"
