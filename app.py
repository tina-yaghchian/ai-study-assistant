import re
import ollama
import streamlit as st
from pypdf import PdfReader
import chromadb

st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("AI Study Assistant")


st.markdown("""
<style>
.katex-display {
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)


MODEL_NAME = "qwen3:8b"   # or "llama3"


def clean_text(text):
    replacements = {
        "�": "",
        "−": "-",
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

    text = re.sub(r"Image source:.*?(?=[A-Z]|$)", "", text)
    text = re.sub(r"Figure:.*?(?=[A-Z]|$)", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text, chunk_size=800):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def ask_ollama(prompt):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response["message"]["content"]


pdf = st.file_uploader("Upload Lecture Notes", type="pdf")

if pdf:
    st.success("PDF uploaded successfully!")

    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    text = clean_text(text)

    with st.expander("Raw extracted text"):
        st.write(text[:4000])

    if st.button("Generate Clean Study Notes"):
        with st.spinner("Generating clean study notes with LaTeX..."):

            summary_prompt = f"""
You are an expert university tutor.

Create clean professional study notes from the lecture notes.

VERY IMPORTANT MATH RULES:

Every mathematical expression must be written in LaTeX.

Do NOT write formulas like this:
(x-h)^2/a^2 + (y-k)^2/b^2 = 1

Instead write:

$$
\\frac{{(x-h)^2}}{{a^2}}
+
\\frac{{(y-k)^2}}{{b^2}}
=
1
$$

Do NOT write:
Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0

Instead write:

$$
Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
$$

Do NOT write:
log_2(x)

Instead write:

$$
\\log_2(x)
$$

Use:
- superscripts with ^
- subscripts with _
- fractions with \\frac{{}}{{}}
- square roots with \\sqrt{{}}
- display equations using $$ $$

Format exactly like this:

# Lecture Summary

## Main Topic

## Key Definitions

## Important Formulas

## Worked Examples

## Exam Tips

Clean messy PDF extraction where possible.
Remove image references and unnecessary source text.

Lecture Notes:

{text[:7000]}
"""

            study_notes = ask_ollama(summary_prompt)

            st.markdown("## Clean Study Notes")
            st.markdown(study_notes, unsafe_allow_html=True)

    if st.button("Generate Practice Quiz"):
        with st.spinner("Generating quiz..."):

            quiz_prompt = f"""
Create a practice quiz from these lecture notes.

Requirements:
- Create 5 questions.
- Include answers.
- Use LaTeX for all math.
- Put formulas inside $$ $$.

Lecture Notes:

{text[:7000]}
"""

            quiz = ask_ollama(quiz_prompt)

            st.markdown("## Practice Quiz")
            st.markdown(quiz, unsafe_allow_html=True)

    chunks = chunk_text(text)

    client = chromadb.Client()

    try:
        client.delete_collection("lecture_notes")
    except:
        pass

    collection = client.create_collection("lecture_notes")

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    question = st.text_input("Ask a question about your notes")

    if question:
        with st.spinner("Thinking..."):

            results = collection.query(
                query_texts=[question],
                n_results=3
            )

            relevant_notes = "\n\n".join(results["documents"][0])

            qa_prompt = f"""
You are a friendly university tutor.

Answer the question using the lecture notes.

VERY IMPORTANT MATH RULES:

Every mathematical expression must be written in LaTeX.

Do NOT write:
x^2/a^2

Instead write:

$$
\\frac{{x^2}}{{a^2}}
$$

Do NOT write:
log_2(x)

Instead write:

$$
\\log_2(x)
$$

Use:
- Markdown formatting
- LaTeX for all math
- display equations with $$ $$
- beginner-friendly explanations
- examples when useful

Format:

## Short Answer

## Key Idea

## Example

Relevant Lecture Notes:

{relevant_notes}

Question:

{question}
"""

            answer = ask_ollama(qa_prompt)

            st.markdown("## Answer")
            st.markdown(answer, unsafe_allow_html=True)