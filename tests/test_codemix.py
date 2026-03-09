from __future__ import annotations

from open_vernacular_ai_kit.codemix_render import analyze_codemix, render_codemix
from open_vernacular_ai_kit.token_lid import TokenLang, detect_token_lang


def test_detect_token_lang_common_gujlish() -> None:
    assert detect_token_lang("maru") == TokenLang.GU_ROMAN
    assert detect_token_lang("chhe") == TokenLang.GU_ROMAN
    assert detect_token_lang("plan") in {TokenLang.EN, TokenLang.GU_ROMAN}  # ambiguous; keep loose


def test_render_codemix_preserves_english_and_gujarati() -> None:
    s = "મારું business plan ready છે!!!"
    out = render_codemix(s)
    assert "business" in out
    assert "plan" in out
    assert "મારું" in out


def test_render_codemix_transliterates_chhe() -> None:
    s = "ready chhe!!!"
    out = render_codemix(s)
    assert "છે" in out


def test_render_codemix_sentence_mode_phrase_exceptions() -> None:
    s = "maru naam shu che?"
    out = render_codemix(s, translit_mode="sentence")
    assert "મારું" in out
    assert "નામ" in out
    assert "શું" in out
    assert "છે" in out

    a = analyze_codemix(s, translit_mode="sentence")
    assert a.n_gu_roman_tokens == 4
    assert a.n_gu_roman_transliterated == 4


def test_render_codemix_preserve_case_flag() -> None:
    s = "My PLAN is ready"
    assert render_codemix(s, preserve_case=True) == "My PLAN is ready"
    assert render_codemix(s, preserve_case=False) == "my plan is ready"


def test_render_codemix_preserve_numbers_flag() -> None:
    s = "મારો નંબર ૧૨૩ છે"
    out = render_codemix(s, preserve_numbers=False)
    assert "123" in out


def test_render_codemix_aggressive_normalize_variants() -> None:
    # Misspelled / over-extended Gujlish vowel runs.
    s = "ready chheee!!!"
    out = render_codemix(s, aggressive_normalize=True)
    assert "છે" in out


def test_render_codemix_hindi_beta_profile() -> None:
    out = render_codemix("mera naam Sudhir hai", language="hi", translit_mode="sentence")
    assert "मेरा" in out
    assert "नाम" in out
    assert "है" in out
