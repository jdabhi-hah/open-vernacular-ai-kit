from __future__ import annotations

import json

from open_vernacular_ai_kit.sarvam_review import (
    dump_reviewed_records_jsonl,
    init_review_record,
    init_review_records_from_candidates,
    load_reviewed_records_jsonl,
)
from open_vernacular_ai_kit.sarvam_teacher import mine_sarvam_teacher_candidate


def _fake_candidate(text: str, language_hint: str, canonical: str):
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


def test_init_review_record_prefers_meta_expected() -> None:
    candidate = _fake_candidate("tamne aaje office ma aavu chhe", "gu", "તમને આજે office માં આવવું છે")
    candidate = candidate.__class__(**{**candidate.__dict__, "meta": {"expected": "EXPECTED"}})

    reviewed = init_review_record(candidate, review_action="accept_sentence_case")

    assert reviewed.review_action == "accept_sentence_case"
    assert reviewed.reviewed_expected == "EXPECTED"


def test_review_round_trip_jsonl(tmp_path) -> None:
    candidate = _fake_candidate("meri maa ka naam kya hai", "hi", "मेरी माँ का नाम क्या है")
    reviewed = init_review_record(
        candidate,
        review_action="accept_sentence_case",
        reviewed_expected="मेरी माँ का नाम क्या है",
        review_notes="safe baseline sentence case",
    )

    path = tmp_path / "reviewed.jsonl"
    dump_reviewed_records_jsonl(path, [reviewed], include_raw_response=False)
    rows = load_reviewed_records_jsonl(path)
    assert len(rows) == 1
    assert rows[0].review_action == "accept_sentence_case"
    assert rows[0].reviewed_expected == "मेरी माँ का नाम क्या है"
    assert rows[0].review_notes == "safe baseline sentence case"


def test_init_review_records_from_candidates_sets_pending_by_default() -> None:
    rows = init_review_records_from_candidates(
        [
            _fake_candidate("aap hamare ghar aaiye", "hi", "आप हमारे घर आइए"),
            _fake_candidate("tame pachi ahi aavo", "gu", "તમે પછી અહીં આવો"),
        ]
    )

    assert len(rows) == 2
    assert rows[0].review_action == "pending"
    assert rows[1].review_action == "pending"
