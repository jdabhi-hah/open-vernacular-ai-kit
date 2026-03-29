from __future__ import annotations

import pytest

from open_vernacular_ai_kit.rag import RagIndex
from open_vernacular_ai_kit.rag_datasets import (
    load_vernacular_facts_tiny,
    load_vernacular_facts_tiny_answer_cases,
)


def _keyword_embed(texts: list[str]) -> list[list[float]]:
    # Deterministic tiny embedder for unit tests (no optional deps).
    #
    # This is NOT a semantic embedder; it only checks presence of a few keywords
    # that appear in the packaged tiny dataset.
    # Pick keywords that appear in both the query text and the relevant doc text.
    keys = ["gujarati", "hindi", "tamil", "kannada", "bengali", "marathi"]
    out: list[list[float]] = []
    for t in texts:
        s = (t or "").lower()
        out.append([1.0 if k in s else 0.0 for k in keys])
    return out


def test_load_vernacular_facts_tiny_has_docs_and_queries() -> None:
    ds = load_vernacular_facts_tiny()
    assert ds.name == "vernacular_facts_tiny"
    assert ds.source == "packaged"
    assert len(ds.docs) >= 8
    assert len(ds.queries) >= 6


def test_rag_index_search_and_recall_at_1() -> None:
    ds = load_vernacular_facts_tiny()
    idx = RagIndex.build(docs=ds.docs, embed_texts=_keyword_embed, embedding_model="test-keywords")

    # Spot check: Gujarat-support query should retrieve the Gujarati support doc at rank-1.
    res = idx.search(
        query="Which language is commonly used in Gujarat customer support workflows?",
        embed_texts=_keyword_embed,
        topk=3,
    )
    assert res
    assert res[0].doc_id == "doc_gujarati_support"

    # With this keyword embedder, the tiny dataset should have perfect recall@1.
    assert idx.recall_at_k(queries=ds.queries, embed_texts=_keyword_embed, k=1) == 1.0


def test_rag_index_json_roundtrip(tmp_path) -> None:
    ds = load_vernacular_facts_tiny()
    idx = RagIndex.build(docs=ds.docs, embed_texts=_keyword_embed, embedding_model="test-keywords")

    p = tmp_path / "idx.json"
    idx.save_json(p)
    idx2 = RagIndex.load_json(p)
    assert idx2.embedding_model == "test-keywords"
    assert len(idx2.docs) == len(idx.docs)
    assert len(idx2.doc_embeddings) == len(idx.doc_embeddings)

    res = idx2.search(
        query="Which language is used broadly in Maharashtra civic services (Marathi)?",
        embed_texts=_keyword_embed,
        topk=3,
    )
    assert res
    assert res[0].doc_id == "doc_marathi_admin"


def test_load_vernacular_facts_tiny_supports_codemix_query_pack() -> None:
    ds = load_vernacular_facts_tiny(query_pack="codemix")
    assert ds.name == "vernacular_facts_tiny_codemix"
    assert len(ds.docs) >= 8
    assert len(ds.queries) >= 6
    assert "use thay chhe" in ds.queries[0].query


def test_load_vernacular_facts_tiny_supports_hard_codemix_query_pack() -> None:
    ds = load_vernacular_facts_tiny(query_pack="codemix_hard")
    assert ds.name == "vernacular_facts_tiny_codemix_hard"
    assert len(ds.docs) >= 10
    assert len(ds.queries) >= 10
    assert "customer help flow" in ds.queries[0].query


def test_load_vernacular_facts_tiny_rejects_unknown_query_pack() -> None:
    with pytest.raises(ValueError, match="query_pack must be one of"):
        load_vernacular_facts_tiny(query_pack="unknown-pack")


def test_load_vernacular_facts_tiny_answer_cases() -> None:
    rows = load_vernacular_facts_tiny_answer_cases()
    assert len(rows) >= 8
    assert rows[0].expected_answer
    assert rows[0].context_doc_ids


def test_load_vernacular_facts_tiny_hard_answer_cases() -> None:
    rows = load_vernacular_facts_tiny_answer_cases(case_pack="hard")
    assert len(rows) >= 10
    assert " " in rows[0].expected_answer
    assert rows[0].meta


def test_load_vernacular_facts_tiny_distractor_answer_cases() -> None:
    rows = load_vernacular_facts_tiny_answer_cases(case_pack="distractor")
    assert len(rows) >= 12
    assert len(rows[0].context_doc_ids) >= 3
    assert rows[0].meta


def test_load_vernacular_facts_tiny_abstention_answer_cases() -> None:
    rows = load_vernacular_facts_tiny_answer_cases(case_pack="abstention")
    assert len(rows) >= 10
    assert all(row.expected_answer == "UNKNOWN" for row in rows)
    assert len(rows[0].context_doc_ids) >= 3


def test_load_vernacular_facts_tiny_rejects_unknown_answer_case_pack() -> None:
    with pytest.raises(ValueError, match="case_pack must be one of"):
        load_vernacular_facts_tiny_answer_cases(case_pack="unknown-pack")
