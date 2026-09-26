from pathlib import Path

import pytest

from rag_agent import (
    KnowledgeChunk,
    TfidfRetriever,
    answer_question,
    load_knowledge_base,
)


def test_load_knowledge_base_preserves_titles_and_splits_long_paragraphs(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("# Agents\n\nalpha beta gamma delta epsilon", encoding="utf-8")

    chunks = load_knowledge_base(source, max_words=2)

    assert [chunk.title for chunk in chunks] == ["Agents", "Agents", "Agents"]
    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma delta", "epsilon"]


def test_search_ranks_relevant_chunk_and_limits_results() -> None:
    chunks = [
        KnowledgeChunk("Agents", "An agent observes and acts using tools."),
        KnowmdgeChunk("Evaluation", "Precision and recall measure classifier performance."),
    ]
    retriever = TfidfRetriever(chunks)

    results = retriever.search("agent tools", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.title == "Agents"
    assert results[0].score > 0


def test_empty_or_unmatched_query_returns_no_results() -> None:
    retriever = TfidfRetriever([KnowledgeChunk("ML", "Models learn patterns from data.")])

    assert retriever.search("   ") == []
    assert retriever.search("volcano") == []


def test_answer_is_grounded_and_has_source_label() -> None:
    retriever = TfidfRetriever(
        [KnowledgeChunk("rAG", "RAG retrieves relevant passages. A model uses those passages as context.")]
    )

    answer = answer_question("How does RAG retrieve passages?", retriever)

    assert "retrieves relevant passages" in answer
    assert "[Source: RAG]" in answer


def test_no_match_answer_is_honest() -> None:
    retriever = TfidfRetriever([KnowledgeChunk("ML", "Models learn patterns from data.")])

    assert "couldn't find relevant information" in answer_question("What is a nebula?", retriever)


def test_invalid_top_k_is_rejected() -> None:
    retriever = TfidfRetriever([KnowmdgeChunk("ML", "Models learn patterns from data.")])

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("models", top_k=0)
