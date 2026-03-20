from __future__ import annotations

import json

from open_vernacular_ai_kit.sarvam_promote import (
    LanguageSentenceCaseRecord,
    infer_profile_candidate_language,
    infer_sentence_case_language,
    promote_profile_candidates_from_review,
    promote_sentence_cases_from_review,
)
from open_vernacular_ai_kit.sarvam_review import init_review_record
from open_vernacular_ai_kit.sarvam_teacher import (
    SarvamTeacherTokenCandidate,
    mine_sarvam_teacher_candidate,
)


def _candidate(text: str, language_hint: str, canonical: str):
    def fake_call(_: str) -> str:
        return json.dumps(
            {
                "language_hint": language_hint,
                "sarvam_native": canonical,
                "sarvam_canonical": canonical,
                "english_tokens_keep": [],
                "candidate_tokens": [],
                "notes": "",
            },
            ensure_ascii=False,
        )

    return mine_sarvam_teacher_candidate(text, language_hint=language_hint, call_model=fake_call)


def test_infer_sentence_case_language_handles_mixed_via_script() -> None:
    reviewed = init_review_record(
        _candidate("tamne refund kyare malse", "mixed", "તમને refund ક્યારે મળશે"),
        review_action="accept_sentence_case",
        reviewed_expected="તમને refund ક્યારે મળશે",
        prefer_meta_expected=False,
    )
    assert infer_sentence_case_language(reviewed) == "gu"


def test_promote_sentence_cases_adds_new_and_skips_same() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("meri maa ka naam kya hai", "hi", "मेरी माँ का नाम क्या है"),
            review_action="accept_sentence_case",
            reviewed_expected="मेरी माँ का नाम क्या है",
            prefer_meta_expected=False,
        ),
        init_review_record(
            _candidate("tamne aaje office ma aavu chhe", "gu", "તમને આજે office માં આવવું છે"),
            review_action="accept_sentence_case",
            reviewed_expected="તમને આજે office માં આવવું છે",
            prefer_meta_expected=False,
        ),
        init_review_record(
            _candidate("ignore me", "hi", "ignore me"),
            review_action="reject",
            reviewed_expected="ignore me",
            prefer_meta_expected=False,
        ),
    ]
    existing = [
        LanguageSentenceCaseRecord(
            language="hi",
            raw="meri maa ka naam kya hai",
            expected="मेरी माँ का नाम क्या है",
            source="existing",
        )
    ]
    merged, report = promote_sentence_cases_from_review(
        reviewed_rows, existing_rows=existing, require_pass=False
    )
    assert len(merged) == 2
    assert report["n_added"] == 1
    assert report["n_duplicates_same"] == 1
    assert report["n_duplicates_conflict"] == 0
    assert report["n_skipped_non_sentence"] == 1


def test_promote_sentence_cases_reports_conflicts() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("mara paisa kyare avse", "mixed", "મારા પૈસા ક્યારે આવશે"),
            review_action="accept_sentence_case",
            reviewed_expected="મારા પૈસા ક્યારે આવશે",
            prefer_meta_expected=False,
        )
    ]
    existing = [
        LanguageSentenceCaseRecord(
            language="gu",
            raw="mara paisa kyare avse",
            expected="મારા પૈસા ક્યારે આવસે",
            source="existing",
        )
    ]
    merged, report = promote_sentence_cases_from_review(
        reviewed_rows, existing_rows=existing, require_pass=False
    )
    assert len(merged) == 1
    assert report["n_added"] == 0
    assert report["n_duplicates_conflict"] == 1


def test_promote_sentence_cases_skips_validation_failures_by_default() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("tamne aaje office ma aavu chhe", "gu", "તમને આજે office માં આવવું છે"),
            review_action="accept_sentence_case",
            reviewed_expected="તમને આજે office માં આવવું છે",
            prefer_meta_expected=False,
        )
    ]
    merged, report = promote_sentence_cases_from_review(reviewed_rows, existing_rows=[])
    assert len(merged) == 0
    assert report["n_added"] == 0
    assert report["n_validation_failures"] == 1


def test_infer_profile_candidate_language_prefers_native_script() -> None:
    reviewed = init_review_record(
        _candidate("mujhe madad chahiye", "mixed", "मुझे मदद चाहिए"),
        review_action="accept_lexicon",
        reviewed_expected="मुझे मदद चाहिए",
        approved_candidate_tokens=[
            SarvamTeacherTokenCandidate(
                roman="madad",
                native="मदद",
                candidate_type="lexicon",
            )
        ],
        prefer_meta_expected=False,
    )
    assert infer_profile_candidate_language(
        reviewed,
        native_text=reviewed.approved_candidate_tokens[0].native,
    ) == "hi"


def test_promote_profile_candidates_adds_lexicon_and_context_entries() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("tamne support ma jawab malse", "gu", "તમને support માં જવાબ મળશે"),
            review_action="accept_context_rule",
            reviewed_expected="તમને support માં જવાબ મળશે",
            approved_candidate_tokens=[
                SarvamTeacherTokenCandidate(
                    roman="jawab",
                    native="જવાબ",
                    candidate_type="context_token",
                )
            ],
            prefer_meta_expected=False,
        ),
        init_review_record(
            _candidate("mujhe voucher bhej dijiye", "hi", "मुझे voucher भेज दीजिए"),
            review_action="accept_lexicon",
            reviewed_expected="मुझे voucher भेज दीजिए",
            approved_candidate_tokens=[
                SarvamTeacherTokenCandidate(
                    roman="voucher",
                    native="वाउचर",
                    candidate_type="lexicon",
                )
            ],
            prefer_meta_expected=False,
        ),
    ]
    profiles = {
        "gu": {
            "code": "gu",
            "common_roman_tokens": ["tame"],
            "context_roman_tokens": ["ma"],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {"ma": "માં"},
        },
        "hi": {
            "code": "hi",
            "common_roman_tokens": ["mujhe"],
            "context_roman_tokens": ["me"],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {"me": "में"},
        },
    }

    merged, report = promote_profile_candidates_from_review(
        reviewed_rows,
        existing_profiles=profiles,
    )

    assert report["n_tokens_promoted"] == 2
    assert report["n_mapping_conflicts"] == 0
    assert merged["gu"]["default_exceptions"]["jawab"] == "જવાબ"
    assert "jawab" in merged["gu"]["context_roman_tokens"]
    assert merged["hi"]["default_exceptions"]["voucher"] == "वाउचर"
    assert "voucher" in merged["hi"]["common_roman_tokens"]


def test_promote_profile_candidates_reports_mapping_conflicts() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("mara paisa kyare avse", "gu", "મારા પૈસા ક્યારે આવશે"),
            review_action="accept_lexicon",
            reviewed_expected="મારા પૈસા ક્યારે આવશે",
            approved_candidate_tokens=[
                SarvamTeacherTokenCandidate(
                    roman="paisa",
                    native="પૈસા",
                    candidate_type="lexicon",
                )
            ],
            prefer_meta_expected=False,
        )
    ]
    profiles = {
        "gu": {
            "code": "gu",
            "common_roman_tokens": [],
            "context_roman_tokens": [],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {"paisa": "પૈસો"},
        },
        "hi": {
            "code": "hi",
            "common_roman_tokens": [],
            "context_roman_tokens": [],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {},
        },
    }

    merged, report = promote_profile_candidates_from_review(
        reviewed_rows,
        existing_profiles=profiles,
    )

    assert merged["gu"]["default_exceptions"]["paisa"] == "પૈસો"
    assert report["n_tokens_promoted"] == 0
    assert report["n_mapping_conflicts"] == 1


def test_promote_profile_candidates_reports_cross_bucket_conflicts() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("office ma aavo", "gu", "office માં આવો"),
            review_action="accept_lexicon",
            reviewed_expected="office માં આવો",
            approved_candidate_tokens=[
                SarvamTeacherTokenCandidate(
                    roman="ma",
                    native="માં",
                    candidate_type="lexicon",
                )
            ],
            prefer_meta_expected=False,
        )
    ]
    profiles = {
        "gu": {
            "code": "gu",
            "common_roman_tokens": [],
            "context_roman_tokens": ["ma"],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {"ma": "માં"},
        },
        "hi": {
            "code": "hi",
            "common_roman_tokens": [],
            "context_roman_tokens": [],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {},
        },
    }

    merged, report = promote_profile_candidates_from_review(
        reviewed_rows,
        existing_profiles=profiles,
    )

    assert merged["gu"]["common_roman_tokens"] == []
    assert report["n_tokens_promoted"] == 0
    assert report["n_bucket_conflicts"] == 1


def test_promote_profile_candidates_rejects_phrase_like_tokens() -> None:
    reviewed_rows = [
        init_review_record(
            _candidate("kya ghar par sab theek hai", "hi", "क्या घर पर सब ठीक है"),
            review_action="accept_context_rule",
            reviewed_expected="क्या घर पर सब ठीक है",
            approved_candidate_tokens=[
                SarvamTeacherTokenCandidate(
                    roman="ghar par",
                    native="घर पर",
                    candidate_type="context_token",
                )
            ],
            prefer_meta_expected=False,
        )
    ]
    profiles = {
        "gu": {
            "code": "gu",
            "common_roman_tokens": [],
            "context_roman_tokens": [],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {},
        },
        "hi": {
            "code": "hi",
            "common_roman_tokens": [],
            "context_roman_tokens": [],
            "roman_clusters": [],
            "roman_suffixes": [],
            "default_exceptions": {},
        },
    }

    merged, report = promote_profile_candidates_from_review(
        reviewed_rows,
        existing_profiles=profiles,
    )

    assert merged["hi"]["context_roman_tokens"] == []
    assert report["n_tokens_promoted"] == 0
    assert report["n_invalid_candidates"] == 1
