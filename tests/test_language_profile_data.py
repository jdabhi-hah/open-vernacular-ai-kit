from __future__ import annotations

from open_vernacular_ai_kit.language_packs import get_language_pack


def test_language_profiles_load_from_packaged_data() -> None:
    gu = get_language_pack("gu")
    hi = get_language_pack("hi")

    assert gu.default_exceptions["tamaro"] == "તમારો"
    assert gu.default_exceptions["ma"] == "માં"
    assert "ma" in gu.context_roman_tokens

    assert hi.default_exceptions["mera"] == "मेरा"
    assert hi.default_exceptions["me"] == "में"
    assert "me" in hi.context_roman_tokens

    assert hi.default_exceptions["dijiye"] == "दीजिए"
    assert hi.default_exceptions["madad"] == "मदद"
    assert gu.default_exceptions["chhiye"] == "છીએ"
    assert gu.default_exceptions["aavo"] == "આવો"
