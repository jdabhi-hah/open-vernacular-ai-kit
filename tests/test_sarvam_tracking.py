from __future__ import annotations

import json

from open_vernacular_ai_kit.sarvam_tracking import build_sarvam_failure_to_fix_report


def _write_reviewed(path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_sarvam_failure_to_fix_report_counts_fixed_sentence_and_profile(monkeypatch, tmp_path) -> None:
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_reviewed(
        reviewed_path,
        [
            {
                "input": "mara paisa kyare avse",
                "language_hint": "gu",
                "source": "teacher_seed",
                "model": "sarvam-m",
                "sarvam_native": "મારા પૈસા ક્યારે આવશે",
                "sarvam_canonical": "મારા પૈસા ક્યારે આવશે",
                "candidate_tokens": [],
                "review_action": "accept_sentence_case",
                "reviewed_expected": "મારા પૈસા ક્યારે આવશે",
                "approved_candidate_tokens": [],
            },
            {
                "input": "mujhe order status batayiye",
                "language_hint": "hi",
                "source": "teacher_seed",
                "model": "sarvam-m",
                "sarvam_native": "मुझे order status बताइए",
                "sarvam_canonical": "मुझे order status बताइए",
                "candidate_tokens": [],
                "review_action": "accept_lexicon",
                "reviewed_expected": "मुझे order status बताइए",
                "approved_candidate_tokens": [
                    {"roman": "batayiye", "native": "बताइए", "type": "lexicon"}
                ],
            },
        ],
    )

    monkeypatch.setattr(
        "open_vernacular_ai_kit.sarvam_tracking.render_codemix",
        lambda text, **kwargs: "મારા પૈસા ક્યારે આવશે",
    )
    monkeypatch.setattr(
        "open_vernacular_ai_kit.sarvam_tracking._load_current_profiles",
        lambda: {
            "gu": {
                "common_roman_tokens": [],
                "context_roman_tokens": [],
                "default_exceptions": {},
            },
            "hi": {
                "common_roman_tokens": ["batayiye"],
                "context_roman_tokens": [],
                "default_exceptions": {"batayiye": "बताइए"},
            },
        },
    )

    report = build_sarvam_failure_to_fix_report(reviewed_paths=[reviewed_path])

    assert report["n_reviewed_rows"] == 2
    assert report["accepted_summary"]["n_accepted_rows"] == 2
    assert report["accepted_summary"]["n_fixed_rows"] == 2
    assert report["by_action"]["accept_sentence_case"]["fix_rate"] == 1.0
    assert report["by_action"]["accept_lexicon"]["token_fix_rate"] == 1.0


def test_build_sarvam_failure_to_fix_report_collects_unfixed_examples(monkeypatch, tmp_path) -> None:
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_reviewed(
        reviewed_path,
        [
            {
                "input": "otp aj tk nthi aavyo yrr",
                "language_hint": "mixed",
                "source": "teacher_seed",
                "model": "sarvam-m",
                "sarvam_native": "OTP आज तक नहीं आया यार",
                "sarvam_canonical": "OTP आज तक नहीं आया यार",
                "candidate_tokens": [],
                "review_action": "accept_context_rule",
                "reviewed_expected": "otp आज तक nthi aavyo yrr",
                "approved_candidate_tokens": [
                    {"roman": "aj", "native": "आज", "type": "context_token"},
                    {"roman": "tk", "native": "तक", "type": "context_token"},
                ],
            }
        ],
    )

    monkeypatch.setattr(
        "open_vernacular_ai_kit.sarvam_tracking._load_current_profiles",
        lambda: {
            "gu": {
                "common_roman_tokens": [],
                "context_roman_tokens": [],
                "default_exceptions": {},
            },
            "hi": {
                "common_roman_tokens": [],
                "context_roman_tokens": ["aj"],
                "default_exceptions": {"aj": "आज"},
            },
        },
    )

    report = build_sarvam_failure_to_fix_report(reviewed_paths=[reviewed_path])

    assert report["accepted_summary"]["n_fixed_rows"] == 0
    assert report["profile_token_summary"]["n_token_checks"] == 2
    assert len(report["unfixed_examples"]) == 1
    assert report["unfixed_examples"][0]["missing_tokens"][0]["roman"] == "tk"
