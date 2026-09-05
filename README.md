# AI Study Assistant

An AI-powered study assistant that helps university students turn lecture notes into clean study material, practice quizzes, and interactive question-and-answer sessions.

## Features

- Upload lecture notes as PDF files
- Extract and clean text from lecture notes
- Generate structured study notes
- Generate practice quizzes with answers
- Ask questions about uploaded lecture material
- Retrieve relevant sections of lecture notes using ChromaDB
- Generate beginner-friendly explanations
- Format mathematical expressions using LaTeX

## Tech Stack

- Python
- Streamlit
- Ollama
- ChromaDB
- PyPDF
- LaTeX / Markdown

## How It Works

1. The user uploads a PDF containing lecture notes.
2. The application extracts the text using PyPDF.
3. The extracted text is cleaned and divided into smaller chunks.
4. Ollama generates study notes or practice questions from the lecture material.
5. ChromaDB stores the lecture-note chunks and retrieves the most relevant sections when the user asks a question.
6. Ollama uses the retrieved material to generate a contextual answer.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-study-assistant.git
cd ai-study-assistant