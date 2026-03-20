from __future__ import annotations

import json
from pathlib import Path

from open_vernacular_ai_kit.eval_harness import GoldenTranslitCase, LanguageSentenceCase
from open_vernacular_ai_kit.sarvam_seed import (
    SarvamTeacherSeedRecord,
    build_failure_seed,
    build_failure_seed_from_golden_translit,
    build_failure_seed_from_language_sentences,
    dump_teacher_seed_jsonl,
)


def test_build_failure_seed_from_language_sentences_only_emits_mismatches(monkeypatch) -> None:
    monkeypatch.setattr(
        "open_vernacular_ai_kit.sarvam_seed._load_language_sentence_cases",
        lambda: [
            LanguageSentenceCase(language="gu", raw="foo", expected="ફૂ", source="case-a"),
            LanguageSentenceCase(language="hi", raw="bar", expected="बार", source="case-b"),
        ],
    )

    def fake_render(text: str, **_: object) -> str:
        return {"foo": "foo", "bar": "बार"}[text]

    monkeypatch.setattr("open_vernacular_ai_kit.sarvam_seed.render_codemix", fake_render)

    rows = build_failure_seed_from_language_sentences(language="all")
    assert len(rows) == 1
    assert rows[0].text == "foo"
    assert rows[0].language_hint == "gu"
    assert rows[0].source == "language_sentence_failure"
    assert rows[0].meta["expected"] == "ફૂ"
    assert rows[0].meta["got"] == "foo"


def test_build_failure_seed_from_golden_translit_handles_failures_and_backend_skips(monkeypatch) -> None:
    monkeypatch.setattr(
        "open_vernacular_ai_kit.sarvam_seed._GOLDEN_TRANSLIT_CASES",
        [
            GoldenTranslitCase("gu", "foo", ["ફૂ"]),
            GoldenTranslitCase("hi", "bar", ["बार"], requires_backend=True),
        ],
    )

    def fake_backend(*, language: str) -> str:
        return "none" if language == "hi" else "rule"

    def fake_render(text: str, **_: object) -> str:
        return {"foo": "foo"}[text]

    monkeypatch.setattr("open_vernacular_ai_kit.sarvam_seed.transliteration_backend", fake_backend)
    monkeypatch.setattr("open_vernacular_ai_kit.sarvam_seed.render_codemix", fake_render)

    rows = build_failure_seed_from_golden_translit(language="all", include_backend_skips=True)
    assert len(rows) == 2
    assert rows[0].source == "golden_translit_failure"
    assert rows[0].meta["got"] == "foo"
    assert rows[1].source == "golden_translit_backend_skip"
    assert rows[1].meta["backend"] == "none"


def test_build_failure_seed_dedupes_and_summarizes(monkeypatch) -> None:
    monkeypatch.setattr(
        "open_vernacular_ai_kit.sarvam_seed.build_failure_seed_from_language_sentences",
        lambda **_: [SarvamTeacherSeedRecord(text="foo", language_hint="gu", source="language_sentence_failure")],
    )
    monkeypatch.setattr(
        "open_vernacular_ai_kit.sarvam_seed.build_failure_seed_from_golden_translit",
        lambda **_: [
            SarvamTeacherSeedRecord(text="foo", language_hint="gu", source="language_sentence_failure"),
            SarvamTeacherSeedRecord(text="bar", language_hint="hi", source="golden_translit_backend_skip"),
        ],
    )

    rows, summary = build_failure_seed(language="all", include_backend_skips=True)
    assert len(rows) == 2
    assert summary["n_rows"] == 2
    assert summary["language_sentence_failures"] == 1
    assert summary["golden_translit_backend_skips"] == 1


def test_dump_teacher_seed_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "seed.jsonl"
    rows, _ = build_failure_seed(
        language="all",
        include_language_sentences=False,
        include_golden_translit=False,
    )
    dump_teacher_seed_jsonl(path, rows)
    assert path.read_text(encoding="utf-8") == ""

    sample = [
        {
            "text": "tamne aaje office ma aavu chhe",
            "language_hint": "gu",
            "source": "language_sentence_failure",
            "meta": {"expected": "તમને આજે office માં આવવું છે"},
        }
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample), encoding="utf-8")
    out = json.loads(path.read_text(encoding="utf-8").strip())
    assert out["language_hint"] == "gu"


def test_realworld_seed_pack_is_loadable_and_balanced() -> None:
    path = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "sarvam_teacher_realworld_seed.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 90
    assert {row["source"] for row in rows} == {"teacher_realworld_pack"}
    assert {row["language_hint"] for row in rows} == {"gu", "hi", "mixed"}
    assert sum(1 for row in rows if row["language_hint"] == "gu") == 30
    assert sum(1 for row in rows if row["language_hint"] == "hi") == 30
    assert sum(1 for row in rows if row["language_hint"] == "mixed") == 30
    assert all(isinstance(row.get("meta"), dict) for row in rows)
    assert all(row["meta"].get("category") for row in rows)
    assert all(row["meta"].get("domain") for row in rows)


def test_followup_realworld_seed_pack_is_loadable_and_balanced() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "datasets"
        / "sarvam_teacher_realworld_seed_followup.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 90
    assert {row["source"] for row in rows} == {"teacher_realworld_followup_pack"}
    assert {row["language_hint"] for row in rows} == {"gu", "hi", "mixed"}
    assert sum(1 for row in rows if row["language_hint"] == "gu") == 30
    assert sum(1 for row in rows if row["language_hint"] == "hi") == 30
    assert sum(1 for row in rows if row["language_hint"] == "mixed") == 30
    assert all(isinstance(row.get("meta"), dict) for row in rows)
    assert all(row["meta"].get("category") for row in rows)
    assert all(row["meta"].get("domain") for row in rows)


def test_noisy_chat_seed_pack_is_loadable_and_balanced() -> None:
    path = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "sarvam_teacher_noisy_chat_seed.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 90
    assert {row["source"] for row in rows} == {"teacher_noisy_chat_pack"}
    assert {row["language_hint"] for row in rows} == {"gu", "hi", "mixed"}
    assert sum(1 for row in rows if row["language_hint"] == "gu") == 30
    assert sum(1 for row in rows if row["language_hint"] == "hi") == 30
    assert sum(1 for row in rows if row["language_hint"] == "mixed") == 30
    assert all(isinstance(row.get("meta"), dict) for row in rows)
    assert all(row["meta"].get("category") for row in rows)
    assert all(row["meta"].get("domain") for row in rows)


def test_whatsapp_export_seed_pack_is_loadable_and_balanced() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "datasets"
        / "sarvam_teacher_whatsapp_export_seed.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 90
    assert {row["source"] for row in rows} == {"teacher_whatsapp_export_pack"}
    assert {row["language_hint"] for row in rows} == {"gu", "hi", "mixed"}
    assert sum(1 for row in rows if row["language_hint"] == "gu") == 30
    assert sum(1 for row in rows if row["language_hint"] == "hi") == 30
    assert sum(1 for row in rows if row["language_hint"] == "mixed") == 30
    assert all(isinstance(row.get("meta"), dict) for row in rows)
    assert all(row["meta"].get("category") for row in rows)
    assert all(row["meta"].get("domain") for row in rows)


def test_voice_note_seed_pack_is_loadable_and_balanced() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "datasets"
        / "sarvam_teacher_voice_note_seed.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 90
    assert {row["source"] for row in rows} == {"teacher_voice_note_pack"}
    assert {row["language_hint"] for row in rows} == {"gu", "hi", "mixed"}
    assert sum(1 for row in rows if row["language_hint"] == "gu") == 30
    assert sum(1 for row in rows if row["language_hint"] == "hi") == 30
    assert sum(1 for row in rows if row["language_hint"] == "mixed") == 30
    assert all(isinstance(row.get("meta"), dict) for row in rows)
    assert all(row["meta"].get("category") for row in rows)
    assert all(row["meta"].get("domain") for row in rows)


def test_ocr_screenshot_seed_pack_is_loadable_and_balanced() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "datasets"
        / "sarvam_teacher_ocr_screenshot_seed.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 90
    assert {row["source"] for row in rows} == {"teacher_ocr_screenshot_pack"}
    assert {row["language_hint"] for row in rows} == {"gu", "hi", "mixed"}
    assert sum(1 for row in rows if row["language_hint"] == "gu") == 30
    assert sum(1 for row in rows if row["language_hint"] == "hi") == 30
    assert sum(1 for row in rows if row["language_hint"] == "mixed") == 30
    assert all(isinstance(row.get("meta"), dict) for row in rows)
    assert all(row["meta"].get("category") for row in rows)
    assert all(row["meta"].get("domain") for row in rows)
