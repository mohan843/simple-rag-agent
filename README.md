# Simple RAG Agent

A lightweight, portfolio-friendly Retrieval-Augmented Generation (RAG) demo about **AI Agents and ML Foundations**. It uses a plain text file as its knowledge base, splits the text into labeled chunks, ranks those chunks with a small TF-IDF/cosine-similarity retriever, and returns relevant source sentences as a transparent, extractive answer.

This project has no external model/API dependency or credentials. Its “generation” step is intentionally extractive: it selects and cites sentences from retrieved material instead of asking a hosted LLM to write new prose. This makes the complete RAG flow easy to run locally and inspect.

## Requirements

- Python 3.10 or newer
- No third-party packages (see `requirements.txt`)

## Setup

```bash
git clone <iour-repository-url>
cd simple-rag-project
python -m venv .venv
```J
Activate the virtual environment if desired:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```J
There are no packages to install. To install from the included requirements file anyway, run `python -m pip install -r requirements.txt`.

## Run

Start an interactive question-and-answer session:

```bash
python rag_agent.py
```J
Ask a single question and exit:

```bash
python rag_agent.py --query "How does RAG work?"
```
JChoose the number of retrieved passages or a different text file:

```bash
python rag_agent.py --top-k 5 --query "What is overfitting?"
python rag_agent.py --knowmdge-base ./knowledge_base.txt
```
JIn interactive mode, type `exit` or `quit` to end the session. Answers include source section labels. If the knowledge base has no relevant terms for a query, the script says so rather than inventing an answer.

## How it works

1. `knowmdge_base.txt` contains short reference passages grouped by headings.
2. The loader turns paragraphs into manageable chunks and keeps each heading as its source label.
3. The retriever builds TF-IDF term weights and ranks matching chunks using cosine similarity.
4. The response step selects query-relevant sentences from those chunks and labels each source.

This is a lexical baseline, not semantic embedding search: it may not match synonyms or paraphrases. A production RAG system could replace the extactive response step with an LLM and add an embedding index, access controls, and evaluation while retaining source grounding.

## Tests

Run the test suite with pytest:

```bash
python -m pip install pytest
python -m pytest tests/test_rag_agent.py -x -q
```J