from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Mapping, Optional

import regex as re


@dataclass(frozen=True)
class LanguagePack:
    code: str
    name: str
    native_script_re: re.Pattern
    common_roman_tokens: frozenset[str]
    context_roman_tokens: frozenset[str]
    roman_clusters: tuple[str, ...]
    roman_suffixes: tuple[str, ...]
    default_exceptions: Mapping[str, str]
    ai4bharat_lang_code: Optional[str]
    sanscript_target: Optional[str]
    translit_candidate_keys: tuple[str, ...]
    terminal_virama: Optional[str]
    dialect_enabled: bool = False


DEFAULT_LANGUAGE = "gu"
_SUPPORTED_LANGUAGE_CODES: tuple[str, ...] = ("gu", "hi")

_LANGUAGE_RUNTIME: dict[str, dict[str, object]] = {
    "gu": {
        "name": "Gujarati",
        "script_block": "Gujarati",
        "ai4bharat_lang_code": "gu",
        "sanscript_target": "GUJARATI",
        "translit_candidate_keys": ("gu", "guj", "gu-Gujr", "Gujarati"),
        "terminal_virama": "\u0acd",
        "dialect_enabled": True,
    },
    "hi": {
        "name": "Hindi",
        "script_block": "Devanagari",
        "ai4bharat_lang_code": "hi",
        "sanscript_target": "DEVANAGARI",
        "translit_candidate_keys": ("hi", "hin", "hi-Deva", "Hindi", "Devanagari"),
        "terminal_virama": "\u094d",
        "dialect_enabled": False,
    },
}

_LANGUAGE_ALIASES: dict[str, str] = {
    "gu": "gu",
    "gujarati": "gu",
    "g": "gu",
    "hi": "hi",
    "hindi": "hi",
    "h": "hi",
}


@lru_cache(maxsize=8)
def _load_language_profile_data(code: str) -> dict[str, object]:
    path = files("open_vernacular_ai_kit").joinpath("_data", "language_profiles", f"{code}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_strs(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(str(x).strip().lower() for x in value if str(x).strip())


def _tuple_strs(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(x).strip().lower() for x in value if str(x).strip())


def _mapping_strs(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k).strip().lower()
        val = str(v).strip()
        if key and val:
            out[key] = val
    return out


@lru_cache(maxsize=8)
def _build_language_pack(code: str) -> LanguagePack:
    runtime = _LANGUAGE_RUNTIME[code]
    data = _load_language_profile_data(code)
    script_block = str(runtime["script_block"])
    return LanguagePack(
        code=code,
        name=str(runtime["name"]),
        native_script_re=re.compile(rf"[\p{{{script_block}}}]", flags=re.VERSION1),
        common_roman_tokens=_frozen_strs(data.get("common_roman_tokens")),
        context_roman_tokens=_frozen_strs(data.get("context_roman_tokens")),
        roman_clusters=_tuple_strs(data.get("roman_clusters")),
        roman_suffixes=_tuple_strs(data.get("roman_suffixes")),
        default_exceptions=_mapping_strs(data.get("default_exceptions")),
        ai4bharat_lang_code=str(runtime["ai4bharat_lang_code"]),
        sanscript_target=str(runtime["sanscript_target"]),
        translit_candidate_keys=tuple(runtime["translit_candidate_keys"]),
        terminal_virama=str(runtime["terminal_virama"]),
        dialect_enabled=bool(runtime["dialect_enabled"]),
    )


def normalize_language_code(code: Optional[str]) -> str:
    raw = str(code or "").strip().lower()
    if not raw:
        return DEFAULT_LANGUAGE
    return _LANGUAGE_ALIASES.get(raw, raw)


def is_supported_language(code: Optional[str]) -> bool:
    return normalize_language_code(code) in _LANGUAGE_RUNTIME


def supported_language_codes() -> tuple[str, ...]:
    return _SUPPORTED_LANGUAGE_CODES


def get_language_pack(code: Optional[str]) -> LanguagePack:
    key = normalize_language_code(code)
    if key not in _LANGUAGE_RUNTIME:
        key = DEFAULT_LANGUAGE
    return _build_language_pack(key)
