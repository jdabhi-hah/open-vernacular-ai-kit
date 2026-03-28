from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .codemix_render import render_codemix
from .errors import InvalidConfigError
from .normalize import normalize_text
from .sarvam_review import SarvamTeacherReviewedRecord


@dataclass(frozen=True)
class LanguageSentenceCaseRecord:
    language: str
    raw: str
    expected: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "language": self.language,
            "raw": self.raw,
            "expected": self.expected,
            "source": self.source,
        }


def _contains_gujarati(text: str) -> bool:
    return any("\u0A80" <= ch <= "\u0AFF" for ch in text)


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def infer_sentence_case_language(reviewed: SarvamTeacherReviewedRecord) -> str:
    if reviewed.candidate.language_hint in {"gu", "hi"}:
        return reviewed.candidate.language_hint

    expected = reviewed.reviewed_expected
    has_gu = _contains_gujarati(expected)
    has_hi = _contains_devanagari(expected)
    if has_gu and not has_hi:
        return "gu"
    if has_hi and not has_gu:
        return "hi"
    raise InvalidConfigError(
        f"Could not infer target language for sentence case: {reviewed.candidate.input!r}"
    )


def load_language_sentence_case_records(path: str | Path) -> list[LanguageSentenceCaseRecord]:
    out: list[LanguageSentenceCaseRecord] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            rec = json.loads(s)
            if not isinstance(rec, dict):
                continue
            out.append(
                LanguageSentenceCaseRecord(
                    language=str(rec.get("language", "gu") or "gu").strip().lower(),
                    raw=str(rec.get("raw", "") or ""),
                    expected=str(rec.get("expected", "") or ""),
                    source=str(rec.get("source", "unknown") or "unknown"),
                )
            )
    return out


def dump_language_sentence_case_records(
    path: str | Path, rows: Iterable[LanguageSentenceCaseRecord]
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def _normalized_token_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalized_exception_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, native in value.items():
        roman = str(key or "").strip().lower()
        target = str(native or "").strip()
        if roman and target:
            out[roman] = target
    return out


def infer_profile_candidate_language(
    reviewed: SarvamTeacherReviewedRecord, *, native_text: str
) -> str:
    has_gu = _contains_gujarati(native_text)
    has_hi = _contains_devanagari(native_text)
    if has_gu and not has_hi:
        return "gu"
    if has_hi and not has_gu:
        return "hi"

    if reviewed.candidate.language_hint in {"gu", "hi"}:
        return reviewed.candidate.language_hint

    expected = reviewed.reviewed_expected
    has_gu = _contains_gujarati(expected)
    has_hi = _contains_devanagari(expected)
    if has_gu and not has_hi:
        return "gu"
    if has_hi and not has_gu:
        return "hi"

    raise InvalidConfigError(
        f"Could not infer target language for profile candidate: {reviewed.candidate.input!r}"
    )


def load_language_profile_data(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_language_profile_data(path: str | Path, profile: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def promote_profile_candidates_from_review(
    reviewed_rows: Iterable[SarvamTeacherReviewedRecord],
    *,
    existing_profiles: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for language, profile in existing_profiles.items():
        data = deepcopy(profile)
        common = _normalized_token_list(data.get("common_roman_tokens"))
        context = _normalized_token_list(data.get("context_roman_tokens"))
        exceptions = _normalized_exception_map(data.get("default_exceptions"))
        data["common_roman_tokens"] = common
        data["context_roman_tokens"] = context
        data["default_exceptions"] = exceptions
        states[language] = {
            "data": data,
            "common_tokens": common,
            "common_set": set(common),
            "context_tokens": context,
            "context_set": set(context),
            "default_exceptions": exceptions,
        }

    promoted_examples: list[dict[str, str]] = []
    skipped_non_profile = 0
    skipped_missing_approved = 0
    invalid_candidates: list[dict[str, str]] = []
    mapping_conflicts: list[dict[str, str]] = []
    bucket_conflicts: list[dict[str, str]] = []
    duplicates_same = 0
    tokens_promoted = 0
    profile_rows_seen = 0
    tokens_considered = 0

    for reviewed in reviewed_rows:
        if reviewed.review_action not in {"accept_lexicon", "accept_context_rule"}:
            skipped_non_profile += 1
            continue

        profile_rows_seen += 1
        approved = reviewed.approved_candidate_tokens or []
        if not approved:
            skipped_missing_approved += 1
            continue

        target_bucket = (
            "common_roman_tokens"
            if reviewed.review_action == "accept_lexicon"
            else "context_roman_tokens"
        )
        opposite_bucket = (
            "context_roman_tokens"
            if target_bucket == "common_roman_tokens"
            else "common_roman_tokens"
        )

        for candidate in approved:
            tokens_considered += 1
            roman = str(candidate.roman or "").strip().lower()
            native = str(candidate.native or "").strip()
            if (
                not roman
                or not native
                or candidate.candidate_type == "english_keep"
                or any(ch.isspace() for ch in roman)
            ):
                invalid_candidates.append(
                    {
                        "input": reviewed.candidate.input,
                        "roman": roman,
                        "native": native,
                        "type": candidate.candidate_type,
                    }
                )
                continue

            language = infer_profile_candidate_language(reviewed, native_text=native)
            state = states.get(language)
            if state is None:
                raise InvalidConfigError(f"Unsupported profile language for promotion: {language}")

            target_tokens = state["common_tokens"] if target_bucket == "common_roman_tokens" else state["context_tokens"]
            target_set = state["common_set"] if target_bucket == "common_roman_tokens" else state["context_set"]
            opposite_set = state["context_set"] if target_bucket == "common_roman_tokens" else state["common_set"]
            exceptions = state["default_exceptions"]

            if roman in opposite_set and roman not in target_set:
                bucket_conflicts.append(
                    {
                        "language": language,
                        "input": reviewed.candidate.input,
                        "roman": roman,
                        "native": native,
                        "target_bucket": target_bucket,
                        "existing_bucket": opposite_bucket,
                    }
                )
                continue

            existing_native = exceptions.get(roman)
            if existing_native is not None and existing_native != native:
                mapping_conflicts.append(
                    {
                        "language": language,
                        "input": reviewed.candidate.input,
                        "roman": roman,
                        "expected_existing": existing_native,
                        "expected_reviewed": native,
                        "target_bucket": target_bucket,
                    }
                )
                continue

            if existing_native == native and roman in target_set:
                duplicates_same += 1
                continue

            exceptions[roman] = native
            if roman not in target_set:
                target_tokens.append(roman)
                target_set.add(roman)

            tokens_promoted += 1
            if len(promoted_examples) < 10:
                promoted_examples.append(
                    {
                        "language": language,
                        "roman": roman,
                        "native": native,
                        "target_bucket": target_bucket,
                        "source_input": reviewed.candidate.input,
                    }
                )

    merged = {language: state["data"] for language, state in states.items()}
    report = {
        "n_profile_rows_seen": profile_rows_seen,
        "n_tokens_considered": tokens_considered,
        "n_tokens_promoted": tokens_promoted,
        "n_duplicates_same": duplicates_same,
        "n_skipped_non_profile": skipped_non_profile,
        "n_skipped_missing_approved": skipped_missing_approved,
        "n_invalid_candidates": len(invalid_candidates),
        "n_mapping_conflicts": len(mapping_conflicts),
        "n_bucket_conflicts": len(bucket_conflicts),
        "invalid_candidates": invalid_candidates,
        "mapping_conflicts": mapping_conflicts,
        "bucket_conflicts": bucket_conflicts,
        "promoted_examples": promoted_examples,
    }
    return merged, report


def promote_sentence_cases_from_review(
    reviewed_rows: Iterable[SarvamTeacherReviewedRecord],
    *,
    existing_rows: Iterable[LanguageSentenceCaseRecord],
    source_suffix: str = "sarvam_review",
    require_pass: bool = True,
) -> tuple[list[LanguageSentenceCaseRecord], dict[str, Any]]:
    existing = list(existing_rows)
    index = {(row.language, row.raw): row for row in existing}
    additions: list[LanguageSentenceCaseRecord] = []
    duplicates_same = 0
    duplicates_conflict: list[dict[str, str]] = []
    skipped_non_sentence = 0
    validation_failures: list[dict[str, str]] = []

    for reviewed in reviewed_rows:
        if reviewed.review_action != "accept_sentence_case":
            skipped_non_sentence += 1
            continue

        lang = infer_sentence_case_language(reviewed)
        raw = reviewed.candidate.input
        expected = reviewed.reviewed_expected
        source = f"{reviewed.candidate.source}:{source_suffix}"
        candidate = LanguageSentenceCaseRecord(
            language=lang,
            raw=raw,
            expected=expected,
            source=source,
        )

        if require_pass:
            got = render_codemix(candidate.raw, language=candidate.language, translit_mode="sentence")
            if normalize_text(got) != normalize_text(candidate.expected):
                validation_failures.append(
                    {
                        "language": candidate.language,
                        "raw": candidate.raw,
                        "expected": candidate.expected,
                        "got": got,
                    }
                )
                continue

        key = (candidate.language, candidate.raw)
        existing_row = index.get(key)
        if existing_row is not None:
            if existing_row.expected == candidate.expected:
                duplicates_same += 1
                continue
            duplicates_conflict.append(
                {
                    "language": candidate.language,
                    "raw": candidate.raw,
                    "expected_existing": existing_row.expected,
                    "expected_reviewed": candidate.expected,
                }
            )
            continue

        additions.append(candidate)
        index[key] = candidate

    merged = existing + additions
    report = {
        "n_existing": len(existing),
        "n_added": len(additions),
        "n_duplicates_same": duplicates_same,
        "n_duplicates_conflict": len(duplicates_conflict),
        "n_skipped_non_sentence": skipped_non_sentence,
        "n_validation_failures": len(validation_failures),
        "conflicts": duplicates_conflict,
        "validation_failures": validation_failures,
        "added_examples": [row.to_dict() for row in additions[:10]],
    }
    return merged, report
