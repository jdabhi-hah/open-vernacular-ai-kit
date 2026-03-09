from __future__ import annotations

import pytest

from open_vernacular_ai_kit.codemix_render import render_codemix
from open_vernacular_ai_kit.token_lid import TokenLang, detect_token_lang


@pytest.mark.parametrize(
    ("token", "language"),
    [
        ("mane", "gu"),
        ("amne", "gu"),
        ("tamaro", "gu"),
        ("aapdu", "gu"),
        ("gayu", "gu"),
        ("chhiye", "gu"),
        ("aavo", "gu"),
        ("pachi", "gu"),
        ("mujhe", "hi"),
        ("tumhara", "hi"),
        ("jayenge", "hi"),
        ("bahut", "hi"),
        ("parivar", "hi"),
        ("dhanyavad", "hi"),
        ("dijiye", "hi"),
        ("chahiye", "hi"),
        ("madad", "hi"),
    ],
)
def test_detect_token_lang_expanded_language_hints(token: str, language: str) -> None:
    assert detect_token_lang(token, language=language) == TokenLang.TARGET_ROMAN


def test_short_context_tokens_do_not_become_global_target_words() -> None:
    assert detect_token_lang("me", language="hi") == TokenLang.EN
    assert detect_token_lang("ka", language="hi") == TokenLang.EN
    assert detect_token_lang("ma", language="gu") == TokenLang.EN
    assert detect_token_lang("ne", language="gu") == TokenLang.EN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mane tari vaat samajh nathi padti", "મને તારી વાત સમજ નથી પડતી"),
        ("tamne aaje office ma aavu chhe", "તમને આજે office માં આવું છે"),
        ("aapdu kaam saras rite thai gayu", "આપણું કામ સરસ રીતે થઈ ગયું"),
        ("shu tame mane madad kari shako?", "શું તમે મને મદદ કરી શકો?"),
        ("amne ahi badma aavu joie", "અમને અહીં બાદમાં આવું જોઈએ"),
        ("tamaro parivar kya chhe?", "તમારો પરિવાર ક્યાં છે?"),
        ("ame kale amdavad ma chhiye", "અમે કાલે અમદાવાદ માં છીએ"),
        ("tame savare ahi aavo", "તમે સવારે અહીં આવો"),
        ("tame sanje tya jao", "તમે સાંજે ત્યાં જાઓ"),
    ],
)
def test_render_codemix_gujarati_quality_cases(raw: str, expected: str) -> None:
    assert render_codemix(raw, language="gu", translit_mode="sentence") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mujhe tumse baat karni hai", "मुझे तुमसे बात करनी है"),
        ("aap kaise ho aur ghar kab aaoge", "आप कैसे हो और घर कब आओगे"),
        ("kal hum market jayenge", "कल हम market जाएंगे"),
        ("tumhara order aaj deliver hoga", "तुम्हारा order आज deliver होगा"),
        ("yeh bahut accha hai", "यह बहुत अच्छा है"),
        ("kahan ho tum", "कहाँ हो तुम"),
        ("meri maa ka naam kya hai", "मेरी माँ का नाम क्या है"),
        ("mera parivar kahan rehta hai", "मेरा परिवार कहाँ रहता है"),
        ("vah ghar me hai", "वह घर में है"),
        ("dhanyavad, main theek hun", "धन्यवाद, मैं ठीक हूँ"),
        ("mujhe paise dijiye", "मुझे पैसे दीजिए"),
        ("mujhe aap ki madad chahiye", "मुझे आप की मदद चाहिए"),
        ("aap hamare ghar aaiye", "आप हमारे घर आइए"),
    ],
)
def test_render_codemix_hindi_quality_cases(raw: str, expected: str) -> None:
    assert render_codemix(raw, language="hi", translit_mode="sentence") == expected
