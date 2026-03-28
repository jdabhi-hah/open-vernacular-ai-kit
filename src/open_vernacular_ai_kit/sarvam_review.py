from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import InvalidConfigError
from .sarvam_teacher import (
    SarvamTeacherCandidateRecord,
    SarvamTeacherTokenCandidate,
    parse_sarvam_teacher_response,
)

_ALLOWED_REVIEW_ACTIONS = {
    "pending",
    "reject",
    "accept_sentence_case",
    "accept_lexicon",
    "accept_context_rule",
    "accept_dialect_case",
}


def _parse_review_action(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in _ALLOWED_REVIEW_ACTIONS:
        return s
    return "pending"


@dataclass(frozen=True)
class SarvamTeacherReviewedRecord:
    candidate: SarvamTeacherCandidateRecord
    review_action: str = "pending"
    reviewed_expected: str = ""
    approved_candidate_tokens: list[SarvamTeacherTokenCandidate] | None = None
    review_notes: str = ""

    def to_dict(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        out = self.candidate.to_dict(include_raw_response=include_raw_response)
        out["review_action"] = self.review_action
        out["reviewed_expected"] = self.reviewed_expected
        out["approved_candidate_tokens"] = [
            c.to_dict() for c in (self.approved_candidate_tokens or [])
        ]
        out["review_notes"] = self.review_notes
        return out


def init_review_record(
    candidate: SarvamTeacherCandidateRecord,
    *,
    review_action: str = "pending",
    review_notes: str = "",
    reviewed_expected: str | None = None,
    approved_candidate_tokens: list[SarvamTeacherTokenCandidate] | None = None,
    prefer_meta_expected: bool = True,
) -> SarvamTeacherReviewedRecord:
    action = _parse_review_action(review_action)
    meta_expected = ""
    if prefer_meta_expected and isinstance(candidate.meta, dict):
        meta_expected = str(candidate.meta.get("expected", "") or "").strip()

    expected = str(reviewed_expected or "").strip() or meta_expected or candidate.sarvam_canonical
    if not expected:
        raise InvalidConfigError("reviewed_expected could not be determined")

    return SarvamTeacherReviewedRecord(
        candidate=candidate,
        review_action=action,
        reviewed_expected=expected,
        approved_candidate_tokens=list(approved_candidate_tokens or []),
        review_notes=str(review_notes or "").strip(),
    )


def load_reviewed_records_jsonl(path: str | Path) -> list[SarvamTeacherReviewedRecord]:
    out: list[SarvamTeacherReviewedRecord] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            rec = json.loads(s)
            if not isinstance(rec, dict):
                continue

            candidate = parse_sarvam_teacher_response(
                json.dumps(
                    {
                        "language_hint": rec.get("language_hint"),
                        "sarvam_native": rec.get("sarvam_native"),
                        "sarvam_canonical": rec.get("sarvam_canonical"),
                        "english_tokens_keep": rec.get("english_tokens_keep", []),
                        "candidate_tokens": rec.get("candidate_tokens", []),
                        "notes": rec.get("notes", ""),
                    },
                    ensure_ascii=False,
                ),
                input_text=str(rec.get("input", "") or ""),
                source=str(rec.get("source", "unknown") or "unknown"),
                model=str(rec.get("model", "sarvam-m") or "sarvam-m"),
                ovak_baseline=str(rec.get("ovak_baseline", "") or ""),
                meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else None),
                fallback_language_hint=rec.get("language_hint"),
            )

            approved: list[SarvamTeacherTokenCandidate] = []
            approved_raw = rec.get("approved_candidate_tokens")
            if isinstance(approved_raw, list):
                for item in approved_raw:
                    if not isinstance(item, dict):
                        continue
                    roman = str(item.get("roman", "") or "").strip()
                    native = str(item.get("native", "") or "").strip()
                    if not roman or not native:
                        continue
                    approved.append(
                        SarvamTeacherTokenCandidate(
                            roman=roman,
                            native=native,
                            candidate_type=str(item.get("type", "lexicon") or "lexicon"),
                            confidence=(
                                None
                                if item.get("confidence") in (None, "")
                                else float(item.get("confidence"))
                            ),
                            notes=str(item.get("notes", "") or "").strip(),
                        )
                    )

            out.append(
                SarvamTeacherReviewedRecord(
                    candidate=candidate,
                    review_action=_parse_review_action(rec.get("review_action")),
                    reviewed_expected=str(rec.get("reviewed_expected", "") or "").strip()
                    or candidate.sarvam_canonical,
                    approved_candidate_tokens=approved,
                    review_notes=str(rec.get("review_notes", "") or "").strip(),
                )
            )
    return out


def dump_reviewed_records_jsonl(
    path: str | Path,
    records: Iterable[SarvamTeacherReviewedRecord],
    *,
    include_raw_response: bool = False,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(include_raw_response=include_raw_response), ensure_ascii=False) + "\n")


def init_review_records_from_candidates(
    candidates: Iterable[SarvamTeacherCandidateRecord],
    *,
    default_action: str = "pending",
    prefer_meta_expected: bool = True,
) -> list[SarvamTeacherReviewedRecord]:
    return [
        init_review_record(
            candidate,
            review_action=default_action,
            prefer_meta_expected=prefer_meta_expected,
        )
        for candidate in candidates
    ]
