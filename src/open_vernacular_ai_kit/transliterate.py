from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Literal, Optional

import regex as re

from .language_packs import get_language_pack


def _normalize_roman_key(s: str) -> str:
    s = (s or "").strip().lower()
    # Keep ASCII-ish latin only; this makes dict lookups stable.
    s = re.sub(r"[^\p{Latin}]+", "", s, flags=re.VERSION1)
    return s


def _collapse_repeats(s: str, *, max_run: int = 2) -> str:
    """Collapse repeated letters: 'maaaru' -> 'maar u' style, but keep aa/ee/oo intact."""
    if not s:
        return s
    out: list[str] = []
    prev = ""
    run = 0
    for ch in s:
        if ch == prev:
            run += 1
        else:
            prev = ch
            run = 1
        if run <= max_run:
            out.append(ch)
    return "".join(out)


def _roman_variants(token: str, *, preserve_case: bool, aggressive_normalize: bool) -> list[str]:
    """
    Generate a small set of spelling variants to increase transliteration hit-rate.

    Keep this bounded (for speed) and deterministic.
    """
    t0 = (token or "").strip()
    if not t0:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def add(v: str) -> None:
        v = (v or "").strip()
        if not v:
            return
        if v in seen:
            return
        seen.add(v)
        variants.append(v)

    if preserve_case:
        add(t0)
    add(t0.lower())

    if aggressive_normalize:
        base = _collapse_repeats(t0.lower(), max_run=2)
        add(base)
        # Common romanization ambiguity pairs across Indic code-mix text.
        swaps: list[tuple[str, str]] = [
            ("chh", "ch"),
            ("oo", "u"),
            ("oo", "o"),
            ("ee", "i"),
            ("ee", "e"),
            ("aa", "a"),
            ("v", "w"),
            ("w", "v"),
        ]
        # Apply swaps in up to 2 rounds to allow simple chaining (e.g. "chhee" -> "chee" -> "che").
        frontier = [base]
        for _ in range(2):
            next_frontier: list[str] = []
            for cur in frontier:
                for a, b in swaps:
                    if a not in cur:
                        continue
                    v = cur.replace(a, b)
                    if v != cur:
                        add(v)
                        next_frontier.append(v)
            frontier = next_frontier
        # Drop trailing 'h' (e.g. "chhe" vs "che") but keep "sh".
        if base.endswith("h") and not base.endswith("sh") and len(base) >= 3:
            add(base[:-1])

    # Hard cap to avoid pathological blow-ups.
    return variants[:12]


@lru_cache(maxsize=4)
def _get_xlit_engine(language: str):
    # Best-effort: if the AI4Bharat Indic-Xlit python package is installed (it may pull heavy deps),
    # use it. We do not depend on it by default because its transitive deps can be brittle.
    pack = get_language_pack(language)
    if not pack.ai4bharat_lang_code:
        return None
    try:
        from ai4bharat.transliteration import XlitEngine  # pyright: ignore[reportMissingImports]
    except Exception:
        return None

    return XlitEngine(pack.ai4bharat_lang_code, beam_width=10, rescore=True)


@lru_cache(maxsize=1)
def _get_sanscript():
    try:
        from indic_transliteration import sanscript
    except Exception:
        return None
    return sanscript


def transliteration_backend(*, language: str = "gu") -> str:
    """
    Return which transliteration backend is selected/available (best-effort).

    This is used for product/demo reporting and should stay cheap and side-effect free.
    """

    return transliteration_backend_configured(preferred="auto", language=language)


TranslitBackend = Literal["auto", "ai4bharat", "sanscript", "none"]


def transliteration_backend_configured(
    *, preferred: TranslitBackend = "auto", language: str = "gu"
) -> str:
    """
    Resolve the actual backend used, considering user preference + availability.

    Returns one of: ai4bharat, sanscript, none
    """
    pack = get_language_pack(language)
    if preferred == "none":
        return "none"
    if preferred == "ai4bharat":
        return "ai4bharat" if _get_xlit_engine(pack.code) is not None else "none"
    if preferred == "sanscript":
        return "sanscript" if (_get_sanscript() is not None and pack.sanscript_target) else "none"

    # auto
    if _get_xlit_engine(pack.code) is not None:
        return "ai4bharat"
    if _get_sanscript() is not None and pack.sanscript_target:
        return "sanscript"
    return "none"


def transliteration_available(*, language: str = "gu") -> bool:
    """True if any transliteration backend is importable."""
    return transliteration_backend(language=language) != "none"


def translit_roman_to_native(text: str, *, topk: int = 1, language: str = "gu") -> Optional[list[str]]:
    """
    Transliterate a romanized token/phrase into the target native script candidates.

    Returns:
      - list[str] of candidates if available
      - None if transliteration engine not installed or fails
    """
    if not text:
        return None

    return translit_roman_to_native_configured(text, topk=topk, language=language)


def translit_gu_roman_to_native(token: str, *, topk: int = 1) -> Optional[list[str]]:
    return translit_roman_to_native(token, topk=topk, language="gu")


def translit_hi_roman_to_native(token: str, *, topk: int = 1) -> Optional[list[str]]:
    return translit_roman_to_native(token, topk=topk, language="hi")


def translit_roman_to_native_configured(
    text: str,
    *,
    topk: int = 1,
    preserve_case: bool = True,
    aggressive_normalize: bool = False,
    exceptions: Optional[dict[str, str]] = None,
    backend: TranslitBackend = "auto",
    language: str = "gu",
) -> Optional[list[str]]:
    """
    Transliterate romanized text into target native script candidates (token or phrase).

    Enhancements:
    - language-pack default exception dictionary (works even without optional backends installed)
    - user exception dictionary overrides
    - spelling variants (`aggressive_normalize=True`)
    - best-effort support for multi-word phrases
    """
    if not text:
        return None

    pack = get_language_pack(language)
    topk = max(1, int(topk))

    exc: dict[str, str] = {k: v for k, v in pack.default_exceptions.items()}
    if exceptions:
        # User dict should win.
        for k, v in exceptions.items():
            if k and v:
                exc[_normalize_roman_key(k)] = v

    s = text.strip()
    if not s:
        return None

    # Phrase mode: try exceptions-only join (offline), then backends.
    if re.search(r"\s", s, flags=re.VERSION1):
        parts = [p for p in re.split(r"\s+", s, flags=re.VERSION1) if p]
        if parts:
            mapped: list[str] = []
            for p in parts:
                k = _normalize_roman_key(p)
                if k in exc:
                    mapped.append(exc[k])
                else:
                    mapped = []
                    break
            if mapped:
                return [" ".join(mapped)]

        # Backends may or may not support phrases; we try best-effort.
        cands = _translit_backend(
            s,
            topk=topk,
            preserve_case=preserve_case,
            backend=backend,
            language=pack.code,
        )
        if cands:
            return cands[:topk]
        return None

    # Single-token: exceptions first (cheap + precise).
    for v in _roman_variants(s, preserve_case=True, aggressive_normalize=aggressive_normalize):
        k = _normalize_roman_key(v)
        if k in exc:
            return [exc[k]]

    # Try a few spelling variants and merge candidates.
    merged: list[str] = []
    seen: set[str] = set()
    for variant in _roman_variants(s, preserve_case=preserve_case, aggressive_normalize=aggressive_normalize):
        cands = _translit_backend(
            variant,
            topk=topk,
            preserve_case=preserve_case,
            backend=backend,
            language=pack.code,
        )
        if not cands:
            continue
        for c in cands:
            if c and c not in seen:
                seen.add(c)
                merged.append(c)
        if len(merged) >= topk:
            break

    return merged[:topk] if merged else None


def translit_gu_roman_to_native_configured(
    text: str,
    *,
    topk: int = 1,
    preserve_case: bool = True,
    aggressive_normalize: bool = False,
    exceptions: Optional[dict[str, str]] = None,
    backend: TranslitBackend = "auto",
) -> Optional[list[str]]:
    return translit_roman_to_native_configured(
        text,
        topk=topk,
        preserve_case=preserve_case,
        aggressive_normalize=aggressive_normalize,
        exceptions=exceptions,
        backend=backend,
        language="gu",
    )


def translit_hi_roman_to_native_configured(
    text: str,
    *,
    topk: int = 1,
    preserve_case: bool = True,
    aggressive_normalize: bool = False,
    exceptions: Optional[dict[str, str]] = None,
    backend: TranslitBackend = "auto",
) -> Optional[list[str]]:
    return translit_roman_to_native_configured(
        text,
        topk=topk,
        preserve_case=preserve_case,
        aggressive_normalize=aggressive_normalize,
        exceptions=exceptions,
        backend=backend,
        language="hi",
    )


def _extract_candidates(obj: object, *, candidate_keys: tuple[str, ...]) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        o = obj.strip()
        return [o] if o else []
    if isinstance(obj, dict):
        for key in candidate_keys:
            v = obj.get(key)
            if isinstance(v, str):
                v = v.strip()
                return [v] if v else []
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str) and x]
        return []
    if isinstance(obj, list):
        out: list[str] = []
        for it in obj:
            out.extend(_extract_candidates(it, candidate_keys=candidate_keys))
        return out
    return []


def _postprocess_candidate(s: str, *, terminal_virama: Optional[str]) -> str:
    """
    Normalize common transliterator artifacts.

    Some backends (notably ITRANS-ish fallbacks) may emit a terminal virama.
    In modern Indic orthography this is usually omitted, so we strip it.
    """

    out = (s or "").strip()
    if not terminal_virama:
        return out
    while out.endswith(terminal_virama):
        out = out[:-1]
    return out


def _translit_backend(
    text: str,
    *,
    topk: int,
    preserve_case: bool,
    backend: TranslitBackend,
    language: str,
) -> Optional[list[str]]:
    pack = get_language_pack(language)
    selected = transliteration_backend_configured(preferred=backend, language=pack.code)
    if selected == "none":
        return None

    engine = _get_xlit_engine(pack.code) if selected == "ai4bharat" else None
    if engine is None:
        sanscript = _get_sanscript() if selected == "sanscript" else None
        if sanscript is None or not pack.sanscript_target:
            return None

        target = getattr(sanscript, pack.sanscript_target, None)
        if target is None:
            return None

        # Fallback: treat input as ITRANS-ish ASCII and convert to the target native script.
        try:
            out = sanscript.transliterate(text.lower(), sanscript.ITRANS, target)
            out = _postprocess_candidate(out, terminal_virama=pack.terminal_virama)
            return [out] if out else None
        except Exception:
            return None

    # Indic-Xlit engine: prefer sentence-aware method when whitespace exists.
    try:
        if re.search(r"\s", text, flags=re.VERSION1):
            for meth in ("translit_sentence", "translit_line"):
                fn = getattr(engine, meth, None)
                if fn is None:
                    continue
                try:
                    out = fn(text, topk=topk)
                except TypeError:
                    out = fn(text)
                cands = _extract_candidates(out, candidate_keys=pack.translit_candidate_keys)
                if cands:
                    cleaned = [
                        _postprocess_candidate(c, terminal_virama=pack.terminal_virama) for c in cands
                    ]
                    cleaned = [c for c in cleaned if c]
                    return cleaned[:topk] if cleaned else None

            # Fallback: transliterate each word and join (still improves with variants/exceptions).
            words = [w for w in re.split(r"\s+", text.strip(), flags=re.VERSION1) if w]
            if not words:
                return None
            rendered: list[str] = []
            for w in words:
                out = engine.translit_word(w, topk=1)
                cands = _extract_candidates(out, candidate_keys=pack.translit_candidate_keys)
                best = (
                    _postprocess_candidate(cands[0], terminal_virama=pack.terminal_virama)
                    if cands
                    else ""
                )
                rendered.append(best if best else w)
            joined = " ".join(rendered).strip()
            return [joined] if joined and pack.native_script_re.search(joined) else None

        out = engine.translit_word(text if preserve_case else text.lower(), topk=topk)
        cands = _extract_candidates(out, candidate_keys=pack.translit_candidate_keys)
        cleaned = [_postprocess_candidate(c, terminal_virama=pack.terminal_virama) for c in cands]
        cleaned = [c for c in cleaned if c]
        return cleaned[:topk] if cleaned else None
    except Exception:
        return None


def translit_tokens_gu_roman(tokens: Iterable[str], *, topk: int = 1) -> list[Optional[str]]:
    """
    Transliterate tokens (romanized Gujarati) one by one.

    Returns list matching input length; each entry is the best candidate or None.
    """
    out: list[Optional[str]] = []
    for t in tokens:
        cands = translit_gu_roman_to_native(t, topk=topk)
        out.append(cands[0] if cands else None)
    return out
