from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .dialect_datasets import packaged_data_path
from .errors import DownloadError, OptionalDependencyError
from .rag import RagDocument, RagQuery


@dataclass(frozen=True)
class RagAnswerCase:
    question: str
    expected_answer: str
    context_doc_ids: list[str]
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class RagDataset:
    name: str
    docs: list[RagDocument]
    queries: list[RagQuery]
    source: str = "packaged"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "open-vernacular-ai-kit"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            rec = json.loads(s)
            if isinstance(rec, dict):
                yield rec


def load_rag_docs_jsonl(path: str | Path) -> list[RagDocument]:
    p = Path(path)
    out: list[RagDocument] = []
    for rec in _iter_jsonl(p):
        out.append(
            RagDocument(
                doc_id=str(rec.get("doc_id", "") or ""),
                text=str(rec.get("text", "") or ""),
                meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else {}),
            )
        )
    return out


def load_rag_queries_jsonl(path: str | Path) -> list[RagQuery]:
    p = Path(path)
    out: list[RagQuery] = []
    for rec in _iter_jsonl(p):
        rel = rec.get("relevant_doc_ids", [])
        if not isinstance(rel, list):
            rel = []
        out.append(
            RagQuery(
                query=str(rec.get("query", "") or ""),
                relevant_doc_ids=[str(x) for x in rel if str(x or "").strip()],
            )
        )
    return out


def load_rag_answer_cases_jsonl(path: str | Path) -> list[RagAnswerCase]:
    p = Path(path)
    out: list[RagAnswerCase] = []
    for rec in _iter_jsonl(p):
        context_doc_ids = rec.get("context_doc_ids", [])
        if not isinstance(context_doc_ids, list):
            context_doc_ids = []
        out.append(
            RagAnswerCase(
                question=str(rec.get("question", "") or ""),
                expected_answer=str(rec.get("expected_answer", "") or ""),
                context_doc_ids=[str(x) for x in context_doc_ids if str(x or "").strip()],
                meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else {}),
            )
        )
    return out


def load_vernacular_facts_tiny(*, query_pack: str = "default") -> RagDataset:
    """
    Load a tiny curated India-focused vernacular snippets dataset (docs + queries).

    This dataset is shipped *inside* the package as a convenience for:
      - quick-start RAG demos
      - retrieval recall regression tests

    For larger / updated datasets, see `download_vernacular_facts_dataset(...)`.
    """

    docs_path = packaged_data_path("vernacular_facts_tiny_docs.jsonl")
    pack = str(query_pack or "default").strip().lower()
    query_pack_map = {
        "default": "vernacular_facts_tiny_queries.jsonl",
        "codemix": "vernacular_facts_tiny_codemix_queries.jsonl",
        "code-mix": "vernacular_facts_tiny_codemix_queries.jsonl",
        "mixed": "vernacular_facts_tiny_codemix_queries.jsonl",
        "codemix_hard": "vernacular_facts_tiny_codemix_hard_queries.jsonl",
        "codemix-hard": "vernacular_facts_tiny_codemix_hard_queries.jsonl",
        "hard": "vernacular_facts_tiny_codemix_hard_queries.jsonl",
    }
    query_filename = query_pack_map.get(pack)
    if query_filename is None:
        raise ValueError("query_pack must be one of: default, codemix, codemix_hard")
    if pack in {"code-mix", "mixed"}:
        pack = "codemix"
    if pack in {"codemix-hard", "hard"}:
        pack = "codemix_hard"

    queries_path = packaged_data_path(query_filename)
    docs = load_rag_docs_jsonl(docs_path)
    queries = load_rag_queries_jsonl(queries_path)
    if pack == "default":
        name = "vernacular_facts_tiny"
    elif pack == "codemix":
        name = "vernacular_facts_tiny_codemix"
    else:
        name = "vernacular_facts_tiny_codemix_hard"
    return RagDataset(name=name, docs=docs, queries=queries, source="packaged")


def load_vernacular_facts_tiny_answer_cases(*, case_pack: str = "default") -> list[RagAnswerCase]:
    pack = str(case_pack or "default").strip().lower()
    case_pack_map = {
        "default": "vernacular_facts_tiny_answer_cases.jsonl",
        "hard": "vernacular_facts_tiny_answer_cases_hard.jsonl",
        "distractor": "vernacular_facts_tiny_answer_cases_distractor.jsonl",
        "abstention": "vernacular_facts_tiny_answer_cases_abstention.jsonl",
    }
    filename = case_pack_map.get(pack)
    if filename is None:
        raise ValueError("case_pack must be one of: default, hard, distractor, abstention")
    return load_rag_answer_cases_jsonl(packaged_data_path(filename))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "Downloading RAG datasets requires requests. Install with: pip install -e '.[rag]'"
        ) from e

    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as e:  # pragma: no cover
        raise DownloadError(f"Failed to download: {url}") from e


def download_vernacular_facts_dataset(
    *,
    docs_url: str,
    queries_url: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> RagDataset:
    """
    Download a vernacular facts dataset JSONL pair to a local cache directory.

    Notes:
      - This is opt-in; the core SDK stays offline-first.
      - URLs are required because the SDK does not assume a canonical hosting location.
    """

    root = (cache_dir or _default_cache_dir()) / "rag-datasets" / "vernacular-facts"
    docs_path = root / "docs.jsonl"
    queries_path = root / "queries.jsonl"
    if force:
        try:
            docs_path.unlink(missing_ok=True)  # py3.11+
        except TypeError:  # pragma: no cover
            if docs_path.exists():
                docs_path.unlink()
        try:
            queries_path.unlink(missing_ok=True)  # py3.11+
        except TypeError:  # pragma: no cover
            if queries_path.exists():
                queries_path.unlink()

    _download(docs_url, docs_path)
    _download(queries_url, queries_path)
    return RagDataset(
        name="vernacular_facts_downloaded",
        docs=load_rag_docs_jsonl(docs_path),
        queries=load_rag_queries_jsonl(queries_path),
        source="downloaded",
    )


def load_gujarat_facts_tiny() -> RagDataset:
    """
    Backward-compatible alias for `load_vernacular_facts_tiny`.
    """

    return load_vernacular_facts_tiny()


def download_gujarat_facts_dataset(
    *,
    docs_url: str,
    queries_url: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> RagDataset:
    """
    Backward-compatible alias for `download_vernacular_facts_dataset`.
    """

    return download_vernacular_facts_dataset(
        docs_url=docs_url,
        queries_url=queries_url,
        cache_dir=cache_dir,
        force=force,
    )
