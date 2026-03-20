from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .codemix_render import render_codemix
from .eval_harness import _GOLDEN_TRANSLIT_CASES, _load_language_sentence_cases
from .normalize import normalize_text
from .transliterate import transliteration_backend


@dataclass(frozen=True)
class SarvamTeacherSeedRecord:
    text: str
    language_hint: str
    source: str
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language_hint": self.language_hint,
            "source": self.source,
            "meta": self.meta or {},
        }


def dump_teacher_seed_jsonl(path: str | Path, rows: Iterable[SarvamTeacherSeedRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def build_failure_seed_from_language_sentences(
    *,
    language: str = "all",
    topk: int = 1,
    translit_mode: str = "sentence",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
) -> list[SarvamTeacherSeedRecord]:
    requested = str(language or "all").strip().lower()
    rows: list[SarvamTeacherSeedRecord] = []
    for case in _load_language_sentence_cases():
        if requested != "all" and case.language != requested:
            continue
        got = render_codemix(
            case.raw,
            language=case.language,
            topk=topk,
            translit_mode=translit_mode,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
            aggressive_normalize=aggressive_normalize,
        )
        if normalize_text(got) == normalize_text(case.expected):
            continue
        rows.append(
            SarvamTeacherSeedRecord(
                text=case.raw,
                language_hint=case.language,
                source="language_sentence_failure",
                meta={
                    "dataset": "language_sentences",
                    "expected": case.expected,
                    "got": got,
                    "case_source": case.source,
                    "translit_mode": translit_mode,
                },
            )
        )
    return rows


def build_failure_seed_from_golden_translit(
    *,
    language: str = "all",
    topk: int = 1,
    translit_mode: str = "sentence",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    include_backend_skips: bool = False,
) -> list[SarvamTeacherSeedRecord]:
    requested = str(language or "all").strip().lower()
    rows: list[SarvamTeacherSeedRecord] = []
    for case in _GOLDEN_TRANSLIT_CASES:
        if requested != "all" and case.language != requested:
            continue
        backend = transliteration_backend(language=case.language)
        if case.requires_backend and backend == "none":
            if not include_backend_skips:
                continue
            rows.append(
                SarvamTeacherSeedRecord(
                    text=case.romanized,
                    language_hint=case.language,
                    source="golden_translit_backend_skip",
                    meta={
                        "dataset": "golden_translit",
                        "expected_any_of": list(case.expected_any_of),
                        "backend": backend,
                    },
                )
            )
            continue

        got = render_codemix(
            case.romanized,
            language=case.language,
            topk=topk,
            translit_mode=translit_mode,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
            aggressive_normalize=aggressive_normalize,
        )
        expected_norms = [normalize_text(x) for x in case.expected_any_of]
        if normalize_text(got) in expected_norms:
            continue
        rows.append(
            SarvamTeacherSeedRecord(
                text=case.romanized,
                language_hint=case.language,
                source="golden_translit_failure",
                meta={
                    "dataset": "golden_translit",
                    "expected_any_of": list(case.expected_any_of),
                    "got": got,
                    "translit_mode": translit_mode,
                    "requires_backend": bool(case.requires_backend),
                },
            )
        )
    return rows


def build_failure_seed(
    *,
    language: str = "all",
    include_language_sentences: bool = True,
    include_golden_translit: bool = True,
    include_backend_skips: bool = False,
    topk: int = 1,
    translit_mode: str = "sentence",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
) -> tuple[list[SarvamTeacherSeedRecord], dict[str, Any]]:
    rows: list[SarvamTeacherSeedRecord] = []
    counts = {
        "language_sentence_failures": 0,
        "golden_translit_failures": 0,
        "golden_translit_backend_skips": 0,
    }
    if include_language_sentences:
        sentence_rows = build_failure_seed_from_language_sentences(
            language=language,
            topk=topk,
            translit_mode=translit_mode,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
            aggressive_normalize=aggressive_normalize,
        )
        counts["language_sentence_failures"] = len(sentence_rows)
        rows.extend(sentence_rows)
    if include_golden_translit:
        golden_rows = build_failure_seed_from_golden_translit(
            language=language,
            topk=topk,
            translit_mode=translit_mode,
            preserve_case=preserve_case,
            preserve_numbers=preserve_numbers,
            aggressive_normalize=aggressive_normalize,
            include_backend_skips=include_backend_skips,
        )
        counts["golden_translit_failures"] = sum(1 for row in golden_rows if row.source == "golden_translit_failure")
        counts["golden_translit_backend_skips"] = sum(
            1 for row in golden_rows if row.source == "golden_translit_backend_skip"
        )
        rows.extend(golden_rows)

    deduped: list[SarvamTeacherSeedRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.language_hint, row.text, row.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    summary = {
        "language": str(language or "all").strip().lower(),
        "n_rows": len(deduped),
        **counts,
    }
    return deduped, summary
