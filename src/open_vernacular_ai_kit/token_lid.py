from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import regex as re

from .language_packs import get_language_pack


class TokenLang(str, Enum):
    EN = "en"
    # Canonical members used by language-pack aware code.
    TARGET_NATIVE = "gu_native"
    TARGET_ROMAN = "gu_roman"
    # Backward-compatible aliases retained for existing integrations.
    GU_NATIVE = "gu_native"
    GU_ROMAN = "gu_roman"
    OTHER = "other"


@dataclass(frozen=True)
class Token:
    text: str
    lang: TokenLang
    confidence: float = 0.0
    reason: str = ""


_LATIN_CHAR_RE = re.compile(r"[\p{Latin}]")
_LATIN_ONLY_RE = re.compile(r"[^\p{Latin}]+", flags=re.VERSION1)

# Tokenization that preserves punctuation as separate tokens.
_TOKEN_RE = re.compile(
    r"([\p{L}\p{M}]+|\p{N}+|[^\p{L}\p{M}\p{N}\s])",
    flags=re.VERSION1,
)

def tokenize(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _is_native_script(token: str, *, language: str) -> bool:
    pack = get_language_pack(language)
    return bool(pack.native_script_re.search(token))


def _is_latin(token: str) -> bool:
    return bool(_LATIN_CHAR_RE.search(token))


def _looks_like_english(token: str) -> bool:
    # Cheap heuristic: common English function words.
    t = token.lower()
    return t in {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "to",
        "for",
        "of",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "i",
        "you",
        "we",
        "they",
        "it",
    }


def _looks_like_target_roman(token: str, *, language: str) -> bool:
    """
    Fast heuristic for romanized target-language text.

    Not perfect; the ML classifier (if present) should override this.
    """
    pack = get_language_pack(language)
    t = token.lower()
    if len(t) <= 2:
        return t in pack.common_roman_tokens
    if t in pack.common_roman_tokens:
        return True
    if len(t) >= 4 and any(t.endswith(suffix) for suffix in pack.roman_suffixes):
        return True
    return any(c in t for c in pack.roman_clusters)


def _model_path() -> Path:
    return Path(__file__).with_name("_data") / "latin_lid.joblib"


@lru_cache(maxsize=1)
def _load_latin_classifier() -> Optional[object]:
    """
    Load a trained sklearn Pipeline, if present and joblib is installed.

    This is optional; the toolkit remains usable without it.
    """
    p = _model_path()
    if not p.exists():
        return None

    try:
        import joblib
    except Exception:
        return None

    try:
        return joblib.load(p)
    except Exception:
        return None


def _latin_predict_is_gu_roman(token: str) -> Optional[bool]:
    clf = _load_latin_classifier()
    if clf is None:
        return None
    try:
        pred = clf.predict([token])
        return bool(pred[0])
    except Exception:
        return None


def _latin_predict_proba_is_gu_roman(token: str) -> Optional[float]:
    """
    Return P(GU_ROMAN) for the Latin-token classifier if predict_proba is available.
    """
    clf = _load_latin_classifier()
    if clf is None:
        return None
    fn = getattr(clf, "predict_proba", None)
    if fn is None:
        return None
    try:
        proba = fn([token])
        # Shape: (1, n_classes)
        row = list(proba[0])
        classes = list(getattr(clf, "classes_", []))
        # Common cases: classes_ == [False, True] or [0, 1]
        if True in classes:
            idx = classes.index(True)
        elif 1 in classes:
            idx = classes.index(1)
        else:
            # Fallback: assume positive class is last.
            idx = len(row) - 1
        p = float(row[idx])
        if p != p:  # NaN guard
            return None
        return max(0.0, min(1.0, p))
    except Exception:
        return None


def _normalize_latin_key(token: str) -> str:
    # Match lexicon normalization: lower + latin-only.
    return _LATIN_ONLY_RE.sub("", (token or "").strip().lower())


def _resolve_fasttext_model_path(explicit_path: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env = os.getenv("GCK_FASTTEXT_MODEL_PATH")
    if env:
        candidates.append(Path(env).expanduser())
    # Common local filenames/locations users tend to keep it in.
    candidates.append(Path("lid.176.ftz"))
    candidates.append(Path.home() / ".cache" / "open_vernacular_ai_kit" / "lid.176.ftz")

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


@lru_cache(maxsize=2)
def _load_fasttext_model(path_str: str) -> Optional[object]:
    try:
        import fasttext  # type: ignore[reportMissingImports]
    except Exception:
        return None
    try:
        return fasttext.load_model(path_str)
    except Exception:
        return None


def _fasttext_predict_language(token: str, *, model_path: Optional[str]) -> Optional[tuple[str, float]]:
    p = _resolve_fasttext_model_path(model_path)
    if p is None:
        return None
    model = _load_fasttext_model(str(p))
    if model is None:
        return None
    try:
        labels, probs = model.predict(token, k=1)
        if not labels or not probs:
            return None
        lab = str(labels[0]).strip()
        prob = float(probs[0])
        if lab.startswith("__label__"):
            lab = lab[len("__label__") :]
        return lab, max(0.0, min(1.0, prob))
    except Exception:
        return None


def analyze_token(
    token: str,
    *,
    language: str = "gu",
    lexicon_keys: Optional[set[str]] = None,
    fasttext_model_path: Optional[str] = None,
) -> Token:
    """
    Token-level LID with lightweight confidence + reason codes.

    Reasons are intentionally stable strings since they can be surfaced in logs/analysis.
    """
    pack = get_language_pack(language)
    if not token:
        return Token(text=token, lang=TokenLang.OTHER, confidence=1.0, reason="empty")

    if _is_native_script(token, language=pack.code):
        return Token(
            text=token,
            lang=TokenLang.TARGET_NATIVE,
            confidence=1.0,
            reason=f"{pack.code}_script",
        )

    if not _is_latin(token):
        # Includes digits and punctuation and other scripts.
        if token.isdigit():
            return Token(text=token, lang=TokenLang.OTHER, confidence=1.0, reason="digits")
        return Token(text=token, lang=TokenLang.OTHER, confidence=1.0, reason="non_latin")

    t_lower = token.lower()
    norm = _normalize_latin_key(token)
    if lexicon_keys and norm in lexicon_keys:
        return Token(text=token, lang=TokenLang.TARGET_ROMAN, confidence=0.98, reason="user_lexicon")

    if t_lower in pack.common_roman_tokens:
        return Token(
            text=token,
            lang=TokenLang.TARGET_ROMAN,
            confidence=0.95,
            reason=f"common_{pack.code}_roman",
        )

    if pack.code == "gu":
        # Existing sklearn model is trained for Gujlish-vs-English only.
        p_gu = _latin_predict_proba_is_gu_roman(token)
        if p_gu is not None:
            if p_gu >= 0.5:
                return Token(text=token, lang=TokenLang.TARGET_ROMAN, confidence=p_gu, reason="ml_latin_lid")
            return Token(text=token, lang=TokenLang.EN, confidence=1.0 - p_gu, reason="ml_latin_lid")

        ml = _latin_predict_is_gu_roman(token)
        if ml is not None:
            return Token(
                text=token,
                lang=TokenLang.TARGET_ROMAN if ml else TokenLang.EN,
                confidence=0.7,
                reason="ml_latin_lid_no_proba",
            )

    # Optional fastText: use as a *fallback* signal (primarily to confidently identify English).
    ft = _fasttext_predict_language(token, model_path=fasttext_model_path)
    if ft is not None:
        lab, prob = ft
        if lab == "en" and prob >= 0.85 and len(norm) >= 3:
            return Token(text=token, lang=TokenLang.EN, confidence=prob, reason="fasttext_en")

    if _looks_like_english(token):
        return Token(text=token, lang=TokenLang.EN, confidence=0.8, reason="english_function_word")
    if _looks_like_target_roman(token, language=pack.code):
        return Token(
            text=token,
            lang=TokenLang.TARGET_ROMAN,
            confidence=0.6,
            reason=f"{pack.code}_roman_clusters",
        )
    return Token(text=token, lang=TokenLang.EN, confidence=0.5, reason="default_en")


def _is_target_neighbor(token: Token, *, pack_code: str) -> bool:
    if token.lang in {TokenLang.TARGET_NATIVE, TokenLang.TARGET_ROMAN}:
        return True
    if token.reason.startswith(f"common_{pack_code}_roman"):
        return True
    if token.reason.startswith(f"{pack_code}_roman"):
        return True
    return False


def detect_token_lang(token: str, *, language: str = "gu") -> TokenLang:
    return analyze_token(token, language=language).lang


def tag_tokens(
    tokens: Iterable[str],
    *,
    language: str = "gu",
    lexicon_keys: Optional[set[str]] = None,
    fasttext_model_path: Optional[str] = None,
) -> list[Token]:
    tagged = [
        analyze_token(
            t,
            language=language,
            lexicon_keys=lexicon_keys,
            fasttext_model_path=fasttext_model_path,
        )
        for t in tokens
    ]
    pack = get_language_pack(language)
    if not pack.context_roman_tokens:
        return tagged

    target_like_count = sum(
        1
        for tok in tagged
        if tok.lang in {TokenLang.TARGET_NATIVE, TokenLang.TARGET_ROMAN}
        or tok.reason.startswith(f"common_{pack.code}_roman")
        or tok.reason.startswith(f"{pack.code}_roman")
    )
    adjusted = list(tagged)
    for i, tok in enumerate(adjusted):
        norm = _normalize_latin_key(tok.text)
        if norm not in pack.context_roman_tokens:
            continue
        if tok.lang not in {TokenLang.EN, TokenLang.OTHER}:
            continue
        prev_is_target = i > 0 and _is_target_neighbor(adjusted[i - 1], pack_code=pack.code)
        next_is_target = i + 1 < len(adjusted) and _is_target_neighbor(adjusted[i + 1], pack_code=pack.code)
        if prev_is_target or next_is_target or target_like_count >= 2:
            adjusted[i] = Token(
                text=tok.text,
                lang=TokenLang.TARGET_ROMAN,
                confidence=0.72,
                reason=f"context_{pack.code}_roman",
            )
    return adjusted
 
