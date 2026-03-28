from __future__ import annotations

import json

from open_vernacular_ai_kit.sarvam_report import build_sarvam_candidate_report
from open_vernacular_ai_kit.sarvam_teacher import (
    SarvamTeacherCandidateRecord,
    SarvamTeacherTokenCandidate,
)


def _record(
    *,
    text: str,
    language_hint: str,
    token: SarvamTeacherTokenCandidate,
) -> SarvamTeacherCandidateRecord:
    return SarvamTeacherCandidateRecord(
        input=text,
        language_hint=language_hint,
        source="unit-test",
        model="sarvam-m",
        ovak_baseline=text,
        sarvam_native=text,
        sarvam_canonical=text,
        english_tokens_keep=[],
        candidate_tokens=[token],
        meta={"domain": "support", "category": "unit"},
    )


def test_build_sarvam_candidate_report_buckets_candidates(tmp_path) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "gu.json").write_text(
        json.dumps(
            {
                "common_roman_tokens": ["tamne"],
                "context_roman_tokens": ["ma"],
                "default_exceptions": {"tamne": "તમને", "ma": "માં", "ready": "રેડી"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (profile_dir / "hi.json").write_text(
        json.dumps(
            {
                "common_roman_tokens": ["mujhe"],
                "context_roman_tokens": ["me"],
                "default_exceptions": {"mujhe": "मुझे", "ready": "रेडी"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = [
        _record(
            text="mari payment pending batave chhe",
            language_hint="gu",
            token=SarvamTeacherTokenCandidate("batave", "બતાવે", "verb_phrase"),
        ),
        _record(
            text="tamne otp malyo",
            language_hint="gu",
            token=SarvamTeacherTokenCandidate("tamne", "તમને", "lexicon"),
        ),
        _record(
            text="ready ho jao",
            language_hint="gu",
            token=SarvamTeacherTokenCandidate("ready", "તૈયાર", "lexicon"),
        ),
        _record(
            text="ghar par sab theek hai",
            language_hint="hi",
            token=SarvamTeacherTokenCandidate("ghar par", "घर पर", "context_token"),
        ),
        _record(
            text="please update",
            language_hint="hi",
            token=SarvamTeacherTokenCandidate("tracking", "tracking", "english_keep"),
        ),
    ]

    report = build_sarvam_candidate_report(rows, profile_dir=profile_dir)

    assert report["n_rows"] == 5
    assert report["n_rows_with_candidates"] == 5
    assert report["novel_single_token_candidates"][0]["roman"] == "batave"
    assert report["already_known_candidates"][0]["roman"] == "tamne"
    assert report["mapping_conflict_candidates"][0]["roman"] == "ready"
    assert report["phrase_candidates"][0]["roman"] == "ghar par"
    assert report["english_keep_candidates"][0]["roman"] == "tracking"
