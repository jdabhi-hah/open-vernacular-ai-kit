from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .sarvam_promote import load_language_profile_data
from .sarvam_teacher import SarvamTeacherCandidateRecord


def _contains_gujarati(text: str) -> bool:
    return any("\u0A80" <= ch <= "\u0AFF" for ch in text)


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _looks_latin_only(text: str) -> bool:
    s = str(text or "").strip()
    return bool(s) and all(ord(ch) < 128 for ch in s)


def _profile_state(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "common": set(str(x).strip().lower() for x in profile.get("common_roman_tokens", []) if str(x).strip()),
        "context": set(str(x).strip().lower() for x in profile.get("context_roman_tokens", []) if str(x).strip()),
        "exceptions": {
            str(k).strip().lower(): str(v).strip()
            for k, v in profile.get("default_exceptions", {}).items()
            if str(k).strip() and str(v).strip()
        },
    }


def _candidate_languages(record: SarvamTeacherCandidateRecord, native: str) -> list[str]:
    if _contains_gujarati(native) and not _contains_devanagari(native):
        return ["gu"]
    if _contains_devanagari(native) and not _contains_gujarati(native):
        return ["hi"]
    if record.language_hint in {"gu", "hi"}:
        return [record.language_hint]
    return []


def _aggregate(
    bucket: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    language: str,
    candidate_type: str,
    roman: str,
    native: str,
    record: SarvamTeacherCandidateRecord,
) -> None:
    key = (language, candidate_type, roman, native)
    entry = bucket.setdefault(
        key,
        {
            "language": language,
            "candidate_type": candidate_type,
            "roman": roman,
            "native": native,
            "count": 0,
            "sample_inputs": [],
            "domains": Counter(),
            "categories": Counter(),
            "sources": Counter(),
        },
    )
    entry["count"] += 1
    if len(entry["sample_inputs"]) < 3:
        entry["sample_inputs"].append(record.input)
    meta = record.meta or {}
    if meta.get("domain"):
        entry["domains"][str(meta["domain"])] += 1
    if meta.get("category"):
        entry["categories"][str(meta["category"])] += 1
    entry["sources"][record.source] += 1


def _finalize(entries: dict[tuple[str, str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for entry in entries.values():
        out.append(
            {
                "language": entry["language"],
                "candidate_type": entry["candidate_type"],
                "roman": entry["roman"],
                "native": entry["native"],
                "count": entry["count"],
                "sample_inputs": entry["sample_inputs"],
                "domains": dict(entry["domains"].most_common()),
                "categories": dict(entry["categories"].most_common()),
                "sources": dict(entry["sources"].most_common()),
            }
        )
    out.sort(key=lambda item: (-int(item["count"]), item["language"], item["roman"]))
    return out


def build_sarvam_candidate_report(
    candidate_rows: Iterable[SarvamTeacherCandidateRecord],
    *,
    profile_dir: str | Path,
) -> dict[str, Any]:
    profiles = {
        lang: _profile_state(load_language_profile_data(Path(profile_dir) / f"{lang}.json"))
        for lang in ("gu", "hi")
    }
    by_language_hint = Counter()
    by_candidate_type = Counter()
    rows_with_candidates = 0
    english_keep_rows = 0

    novel_single_token: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    already_known: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    mapping_conflicts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    phrase_candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    english_keep_candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    ambiguous_candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for record in candidate_rows:
        by_language_hint[record.language_hint] += 1
        if record.english_tokens_keep:
            english_keep_rows += 1
        if record.candidate_tokens:
            rows_with_candidates += 1

        for token in record.candidate_tokens:
            candidate_type = token.candidate_type
            roman = token.roman.strip().lower()
            native = token.native.strip()
            if not roman or not native:
                continue

            by_candidate_type[candidate_type] += 1

            if candidate_type == "english_keep":
                _aggregate(
                    english_keep_candidates,
                    language=record.language_hint,
                    candidate_type=candidate_type,
                    roman=roman,
                    native=native,
                    record=record,
                )
                continue

            if any(ch.isspace() for ch in roman):
                _aggregate(
                    phrase_candidates,
                    language=record.language_hint,
                    candidate_type=candidate_type,
                    roman=roman,
                    native=native,
                    record=record,
                )
                continue

            languages = _candidate_languages(record, native)
            if not languages:
                _aggregate(
                    ambiguous_candidates,
                    language=record.language_hint,
                    candidate_type=candidate_type,
                    roman=roman,
                    native=native,
                    record=record,
                )
                continue

            for language in languages:
                state = profiles[language]
                existing_native = state["exceptions"].get(roman)
                if existing_native and existing_native != native:
                    _aggregate(
                        mapping_conflicts,
                        language=language,
                        candidate_type=candidate_type,
                        roman=roman,
                        native=native,
                        record=record,
                    )
                    continue
                if existing_native or roman in state["common"] or roman in state["context"]:
                    _aggregate(
                        already_known,
                        language=language,
                        candidate_type=candidate_type,
                        roman=roman,
                        native=native,
                        record=record,
                    )
                    continue
                if _looks_latin_only(native):
                    _aggregate(
                        english_keep_candidates,
                        language=language,
                        candidate_type=candidate_type,
                        roman=roman,
                        native=native,
                        record=record,
                    )
                    continue
                _aggregate(
                    novel_single_token,
                    language=language,
                    candidate_type=candidate_type,
                    roman=roman,
                    native=native,
                    record=record,
                )

    return {
        "n_rows": sum(by_language_hint.values()),
        "n_rows_with_candidates": rows_with_candidates,
        "n_rows_with_english_keep": english_keep_rows,
        "language_hint_counts": dict(by_language_hint),
        "candidate_type_counts": dict(by_candidate_type),
        "novel_single_token_candidates": _finalize(novel_single_token),
        "already_known_candidates": _finalize(already_known),
        "mapping_conflict_candidates": _finalize(mapping_conflicts),
        "phrase_candidates": _finalize(phrase_candidates),
        "english_keep_candidates": _finalize(english_keep_candidates),
        "ambiguous_candidates": _finalize(ambiguous_candidates),
    }
