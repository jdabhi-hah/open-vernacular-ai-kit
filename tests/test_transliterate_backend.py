from __future__ import annotations

from open_vernacular_ai_kit import transliterate


def test_transliteration_backend_auto_does_not_fall_back_to_sanscript(monkeypatch) -> None:
    monkeypatch.setattr(transliterate, "_get_xlit_engine", lambda language: None)
    monkeypatch.setattr(transliterate, "_get_sanscript", lambda: object())

    assert transliterate.transliteration_backend_configured(preferred="auto", language="gu") == "none"


def test_transliteration_backend_explicit_sanscript_still_supported(monkeypatch) -> None:
    monkeypatch.setattr(transliterate, "_get_sanscript", lambda: object())

    assert (
        transliterate.transliteration_backend_configured(preferred="sanscript", language="gu")
        == "sanscript"
    )
