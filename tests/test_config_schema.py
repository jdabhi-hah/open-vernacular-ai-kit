from __future__ import annotations

from open_vernacular_ai_kit.config import CodeMixConfig


def test_config_to_dict_includes_schema_version() -> None:
    cfg = CodeMixConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert int(d.get("schema_version", 0) or 0) >= 1


def test_config_roundtrip_to_from_dict() -> None:
    cfg = CodeMixConfig(
        language="hi",
        numerals="ascii",
        preserve_case=False,
        preserve_numbers=False,
        topk=3,
        translit_mode="sentence",
        translit_backend="auto",
        aggressive_normalize=True,
        dialect_backend="heuristic",
        dialect_min_confidence=0.8,
        dialect_normalize=True,
        allow_remote_models=False,
    ).normalized()
    d = cfg.to_dict()
    cfg2 = CodeMixConfig.from_dict(d)
    assert cfg2.to_dict() == cfg.to_dict()


def test_config_from_dict_accepts_v0_without_schema_version() -> None:
    # Simulate an old config dict saved before schema_version existed.
    old = {
        "numerals": "keep",
        "topk": 2,
        "translit_mode": "token",
        "translit_backend": "auto",
        # Unknown key should be ignored by default for back-compat.
        "some_future_field": "ignored",
    }
    cfg = CodeMixConfig.from_dict(old)
    assert cfg.schema_version >= 1
    assert cfg.topk == 2


def test_config_from_dict_unknown_language_falls_back_to_default() -> None:
    cfg = CodeMixConfig.from_dict({"language": "unknown-lang"})
    assert cfg.language == "gu"


def test_config_from_dict_strict_rejects_unknown_keys() -> None:
    try:
        _ = CodeMixConfig.from_dict({"numerals": "keep", "unknown": 1}, strict=True)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
