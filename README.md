# 🤖 AI Study Assistant

An AI-powered study assistant designed to help university students turn lecture notes into useful study material.

Upload your lecture notes as a PDF, generate clean study notes, create practice quizzes, or ask questions directly about the material.

---

## ✨ Features

### 📄 Lecture Note Processing
- Upload lecture notes as PDF files
- Extract text from uploaded documents
- Clean and format messy PDF text
- Split lecture material into searchable chunks

### 📝 AI Study Notes
Generate structured study notes from your lecture material, including:

- Lecture summaries
- Main topics
- Key definitions
- Important formulas
- Worked examples
- Exam tips

Mathematical expressions are formatted using LaTeX for readability.

### 🧠 Practice Quiz Generation
Automatically generate practice questions based on uploaded lecture material, including answers to help with self-study.

### 💬 Ask Questions About Your Notes
Ask questions about your lecture material and receive answers based on the most relevant sections of your notes.

The application uses ChromaDB to retrieve relevant lecture-note chunks before generating an answer.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core application |
| **Streamlit** | Interactive web interface |
| **Ollama** | Local AI model inference |
| **ChromaDB** | Vector-based document retrieval |
| **PyPDF** | PDF text extraction |
| **LaTeX** | Mathematical formatting |

---

## 🔍 How It Works

```text
                ┌──────────────────┐
                │  Upload Lecture  │
                │       PDF        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Extract & Clean │
                │      Text        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Split Into Text  │
                │      Chunks      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    ChromaDB      │
                │     Retrieval    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      Ollama      │
                │    AI Model      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Study Notes /    │
                │ Quiz / Q&A       │
                └──────────────────┘