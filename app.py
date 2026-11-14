"""Streamlit front end for the AI Study Assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from study_assistant import __version__
from study_assistant.config import get_settings
from study_assistant.documents import DocumentError, file_id, load_pdf
from study_assistant.llm import LLMError, OllamaLLM
from study_assistant.rag import StudyAssistant
from study_assistant.store import StudyStore

st.set_page_config(page_title="AI Study Assistant", layout="wide")

settings = get_settings()

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1100px; }
.katex-display { overflow-x: auto; overflow-y: hidden; }
.hero h1 { font-size: 2.4rem; margin-bottom: .2rem; }
.hero p { font-size: 1.05rem; opacity: .75; margin-top: 0; }
.feature { border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1.1rem 1.2rem; min-height: 130px; }
.feature h4 { margin: 0 0 .4rem 0; }
.feature p { opacity: .75; font-size: .92rem; margin: 0; }
.step { opacity: .8; font-size: .95rem; }
.src { border-left: 3px solid var(--primary-color, #2563eb); padding: .35rem .8rem; margin: .5rem 0;
       background: rgba(37,99,235,.06); border-radius: 0 8px 8px 0; font-size: .9rem; }
.src .meta { font-weight: 600; font-size: .82rem; opacity: .85; }
.status-dot { display:inline-block; width:.55rem; height:.55rem; border-radius:50%; margin-right:.4rem; }
.muted { opacity: .65; font-size: .85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading your library...")
def get_store() -> StudyStore:
    return StudyStore(
        settings.data_dir,
        collection_name=settings.collection_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@st.cache_data(ttl=15, show_spinner=False)
def ollama_models(host: str) -> tuple[list[str], str | None]:
    try:
        return OllamaLLM(host=host).available_models(), None
    except LLMError as exc:
        return [], str(exc)


def init_state() -> None:
    defaults = {
        "messages": [],  # chat history: {"role", "content", "sources"}
        "notes": {},  # doc_id -> markdown
        "quiz": None,  # {"doc_id", "questions", "submitted"}
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()
store = get_store()


# ---------------------------------------------------------------------- sidebar
def ingest(files) -> None:
    added = 0
    for uploaded in files:
        data = uploaded.getvalue()
        if store.has_document(file_id(data)):
            st.toast(f"{uploaded.name} is already in your library.")
            continue
        with st.status(f"Processing {uploaded.name}...", expanded=False) as status:
            try:
                status.write("Extracting text from pages")
                document = load_pdf(data, uploaded.name)
                status.write(f"Embedding {len(document.pages)} pages")
                info = store.add_document(document)
                status.update(label=f"Added {info.filename} ({info.pages} pages, {info.chunks} chunks)", state="complete")
                added += 1
            except DocumentError as exc:
                status.update(label=str(exc), state="error")
    if added:
        st.session_state.uploader_key += 1  # reset the uploader so files aren't re-processed
        st.rerun()


with st.sidebar:
    st.markdown("## AI Study Assistant")

    models, model_error = ollama_models(settings.ollama_host)
    if model_error:
        st.markdown('<span class="status-dot" style="background:#ef4444"></span>**Ollama offline**', unsafe_allow_html=True)
        st.caption(model_error)
        model = settings.model
        if st.button("Retry connection", use_container_width=True):
            ollama_models.clear()
            st.rerun()
    else:
        st.markdown(
            f'<span class="status-dot" style="background:#22c55e"></span>**Ollama connected** '
            f'<span class="muted">· {len(models)} model{"s" if len(models) != 1 else ""}</span>',
            unsafe_allow_html=True,
        )
        if not models:
            st.warning(f"No models installed. Run `ollama pull {settings.model}`.")
            model = settings.model
        else:
            default = models.index(settings.model) if settings.model in models else 0
            model = st.selectbox("Model", models, index=default, help="Any model you've pulled with Ollama.")

    st.divider()
    st.markdown("#### Your lectures")
    uploads = st.file_uploader(
        "Upload lecture PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    if uploads:
        ingest(uploads)

    documents = store.list_documents()
    if not documents:
        st.caption("No lectures yet. Drop a PDF above to get started.")
    for doc in documents:
        with st.container(border=True):
            col_name, col_del = st.columns([5, 1], vertical_alignment="center")
            col_name.markdown(
                f"**{doc.filename}**  \n<span class='muted'>{doc.pages} pages · {doc.chunks} chunks</span>",
                unsafe_allow_html=True,
            )
            if col_del.button(":material/delete:", key=f"del_{doc.doc_id}", help=f"Remove {doc.filename}"):
                store.delete_document(doc.doc_id)
                st.session_state.notes.pop(doc.doc_id, None)
                st.rerun()

    with st.expander("Settings"):
        top_k = st.slider("Passages retrieved per question", 2, 10, settings.top_k)
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, settings.temperature, 0.05)
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.caption(f"v{__version__} · Runs locally with Ollama and ChromaDB")


llm = OllamaLLM(host=settings.ollama_host, model=model, temperature=temperature)
assistant = StudyAssistant(store, llm, top_k=top_k, max_context_chars=settings.max_context_chars)
doc_names = {d.doc_id: d.filename for d in documents}


# ---------------------------------------------------------------------- empty state
def render_welcome() -> None:
    st.markdown(
        '<div class="hero"><h1>AI Study Assistant</h1>'
        "<p>Upload lecture PDFs, then ask questions about them, generate study notes, or take a "
        "practice quiz. Answers cite the file and page they come from. The model runs locally "
        "through Ollama, so your documents stay on this computer.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    features = [
        ("Ask questions", "Answers are built from the most relevant passages and cite the file and page."),
        ("Study notes", "Structured notes with definitions, formulas, worked examples and exam tips."),
        ("Practice quiz", "Multiple-choice questions with scoring and an explanation for each answer."),
    ]
    for col, (title, text) in zip(st.columns(3), features):
        col.markdown(
            f'<div class="feature"><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown("##### Get started")
    st.markdown(
        '<div class="step">1. Make sure Ollama is running and a model is installed '
        f"(<code>ollama pull {settings.model}</code>).<br>"
        "2. Upload one or more lecture PDFs from the sidebar.<br>"
        "3. Ask a question, generate notes, or take a quiz.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------- chat
AVATARS = {"user": ":material/person:", "assistant": ":material/menu_book:"}

def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"{len(sources)} source{'s' if len(sources) != 1 else ''} from your notes"):
        for i, s in enumerate(sources, start=1):
            snippet = s["text"] if len(s["text"]) < 420 else s["text"][:420].rsplit(" ", 1)[0] + " ..."
            st.markdown(
                f'<div class="src"><div class="meta">[{i}] {s["filename"]} · page {s["page"]} · '
                f'{max(s["score"], 0):.0%} match</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"> {snippet}")


def render_chat() -> None:
    scope = st.multiselect(
        "Search in",
        options=list(doc_names),
        format_func=doc_names.get,
        placeholder="All lectures",
    )

    suggestions = st.empty()
    if not st.session_state.messages:
        with suggestions.container():
            st.markdown("##### Try asking")
            examples = [
                "Summarise the main idea of this lecture",
                "Explain the most important formula step by step",
                "What are the key definitions I need to know?",
            ]
            cols = st.columns(len(examples))
            for col, example in zip(cols, examples):
                if col.button(example, use_container_width=True):
                    st.session_state.pending_question = example

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    question = st.chat_input("Ask anything about your lectures...")
    question = question or st.session_state.pop("pending_question", None)
    if not question:
        return
    suggestions.empty()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Searching your notes..."):
                answer = assistant.ask(question, history=history, doc_ids=scope or None)
            text = st.write_stream(answer.stream)
        except LLMError as exc:
            st.error(str(exc))
            return
        sources = [s.__dict__ for s in answer.sources]
        render_sources(sources)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text if isinstance(text, str) else "".join(map(str, text)), "sources": sources},
    ]


# ---------------------------------------------------------------------- notes
def render_notes() -> None:
    col_doc, col_topic = st.columns([1, 1])
    doc_id = col_doc.selectbox("Lecture", list(doc_names), format_func=doc_names.get, key="notes_doc")
    topic = col_topic.text_input("Focus topic (optional)", placeholder="e.g. conic sections", key="notes_topic")

    generate = st.button("Generate study notes", type="primary")
    if generate:
        try:
            text = st.write_stream(assistant.study_notes(doc_id, topic))
            st.session_state.notes[doc_id] = text
            st.rerun()
        except LLMError as exc:
            st.error(str(exc))
            return

    notes = st.session_state.notes.get(doc_id)
    if not notes:
        st.info("Notes cover definitions, formulas, worked examples and exam tips for the whole lecture. "
                "Add a focus topic to concentrate on one part.")
        return
    with st.container(border=True):
        st.markdown(notes)
    stem = doc_names[doc_id].rsplit(".", 1)[0]
    st.download_button("Download as Markdown", notes, file_name=f"{stem}-notes.md", mime="text/markdown")


# ---------------------------------------------------------------------- quiz
def render_quiz() -> None:
    col_doc, col_n, col_level = st.columns([2, 1, 1])
    doc_id = col_doc.selectbox("Lecture", list(doc_names), format_func=doc_names.get, key="quiz_doc")
    count = col_n.number_input("Questions", 3, 15, 5)
    difficulty = col_level.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    topic = st.text_input("Focus topic (optional)", key="quiz_topic", placeholder="Leave blank to cover the whole lecture")

    if st.button("Generate quiz", type="primary"):
        try:
            with st.spinner("Writing questions..."):
                quiz = assistant.quiz(doc_id, count=int(count), difficulty=difficulty, topic=topic)
        except LLMError as exc:
            st.error(str(exc))
            return
        if not quiz.questions:
            st.error("The model didn't return any usable questions. Try again or switch models.")
            return
        st.session_state.quiz = {"doc_id": doc_id, "questions": quiz.questions, "submitted": False,
                                 "id": datetime.now().timestamp()}
        st.rerun()

    quiz = st.session_state.quiz
    if not quiz or quiz["doc_id"] != doc_id:
        st.info("Generate a quiz to test yourself. You'll get your score and an explanation for every answer.")
        return

    questions, submitted = quiz["questions"], quiz["submitted"]
    with st.form(f"quiz_{quiz['id']}"):
        choices = []
        for i, q in enumerate(questions):
            st.markdown(f"**{i + 1}. {q.question}**")
            choice = st.radio(
                f"Question {i + 1}", range(4), format_func=lambda j, q=q: q.options[j], index=None,
                key=f"quiz_{quiz['id']}_{i}", label_visibility="collapsed", disabled=submitted,
            )
            choices.append(choice)
            if submitted:
                where = f" (page {q.page})" if q.page else ""
                if choice == q.answer_index:
                    st.success(f"Correct. {q.explanation}{where}")
                else:
                    st.error(f"Answer: **{q.options[q.answer_index].rstrip('.')}**. {q.explanation}{where}")
            st.write("")
        done = st.form_submit_button("Check answers", type="primary", disabled=submitted)

    if done:
        quiz["submitted"] = True
        quiz["score"] = sum(c == q.answer_index for c, q in zip(choices, questions))
        st.rerun()
    if submitted:
        score, total = quiz["score"], len(questions)
        st.metric("Your score", f"{score} / {total}", f"{score / total:.0%}", delta_color="off")


# ---------------------------------------------------------------------- layout
if not documents:
    render_welcome()
else:
    view = st.segmented_control(
        "View",
        ["Ask", "Study notes", "Quiz"],
        default="Ask",
        key="view",
        label_visibility="collapsed",
    ) or "Ask"
    st.write("")
    if model_error:
        st.warning("Ollama isn't reachable, so answers can't be generated right now. See the sidebar for details.")
    {"Ask": render_chat, "Study notes": render_notes, "Quiz": render_quiz}[view]()
