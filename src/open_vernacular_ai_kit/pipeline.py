from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

from .codeswitch import CodeSwitchMetrics, compute_code_switch_metrics
from .config import CodeMixConfig
from .dialect_backends import get_dialect_backend
from .dialect_normalizers import get_dialect_normalizer
from .dialects import (
    DialectDetection,
    DialectNormalizationResult,
    detect_dialect_from_tagged_tokens,
)
from .errors import InvalidConfigError
from .language_packs import get_language_pack
from .lexicon import LexiconLoadResult, load_user_lexicon
from .normalize import normalize_text
from .rendering import render_tokens
from .token_lid import Token, TokenLang, tag_tokens, tokenize
from .transliterate import (
    translit_roman_to_native_configured,
    transliteration_backend_configured,
)

_JOINERS = {"-", "_", "/", "@"}

EventHook = Callable[[dict[str, Any]], None]


def _emit(hook: Optional[EventHook], event: dict[str, Any]) -> None:
    if hook is None:
        return
    try:
        hook(event)
    except Exception:
        # Logging must never break text processing.
        return


@dataclass(frozen=True)
class CodeMixPipelineResult:
    raw: str
    normalized: str
    tokens: list[str]
    tagged_tokens: list[Token]
    tagged_tokens_effective: list[Token]
    codeswitch: CodeSwitchMetrics
    dialect: DialectDetection
    dialect_normalization: DialectNormalizationResult
    rendered_tokens: list[str]
    codemix: str
    language: str

    n_tokens: int
    n_en_tokens: int
    n_gu_native_tokens: int
    n_gu_roman_tokens: int
    n_gu_roman_transliterated: int

    transliteration_backend: str


def normalize_stage(text: str, *, config: CodeMixConfig, on_event: Optional[EventHook] = None) -> str:
    cfg = config.normalized()
    numerals_eff = cfg.numerals_effective()

    norm = normalize_text(text or "", numerals=numerals_eff)
    if not cfg.preserve_case:
        norm = norm.lower()

    _emit(
        on_event,
        {
            "stage": "normalize",
            "numerals": numerals_eff,
            "preserve_case": cfg.preserve_case,
            "raw_len": len(text or ""),
            "normalized_len": len(norm),
        },
    )
    return norm


def tokenize_stage(text: str, *, on_event: Optional[EventHook] = None) -> list[str]:
    toks = tokenize(text or "")
    _emit(
        on_event,
        {
            "stage": "tokenize",
            "n_tokens": len(toks),
            "tokens_preview": toks[:12],
        },
    )
    return toks


def lid_stage(
    tokens: list[str],
    *,
    language: str = "gu",
    lexicon_keys: Optional[set[str]] = None,
    fasttext_model_path: Optional[str] = None,
    user_lexicon_source: str = "none",
    on_event: Optional[EventHook] = None,
) -> list[Token]:
    tagged = tag_tokens(
        tokens,
        language=language,
        lexicon_keys=lexicon_keys,
        fasttext_model_path=fasttext_model_path,
    )
    counts = {k.value: 0 for k in TokenLang}
    for t in tagged:
        counts[t.lang.value] = counts.get(t.lang.value, 0) + 1
    _emit(
        on_event,
        {
            "stage": "lid",
            "counts": counts,
            "preview": [
                {"text": t.text, "lang": t.lang.value, "confidence": t.confidence, "reason": t.reason}
                for t in tagged[:12]
            ],
            "user_lexicon": user_lexicon_source,
        },
    )
    return tagged


@lru_cache(maxsize=8)
def _load_user_lexicon_cached(path: str) -> LexiconLoadResult:
    # Cache by path string; this is a convenience for repeated pipeline runs.
    return load_user_lexicon(path)


def transliterate_stage(
    tagged: list[Token],
    *,
    config: CodeMixConfig,
    language: str = "gu",
    lexicon: Optional[dict[str, str]] = None,
    on_event: Optional[EventHook] = None,
) -> tuple[list[str], int]:
    cfg = config.normalized()
    pack = get_language_pack(language)

    if cfg.translit_mode not in ("token", "sentence"):
        raise InvalidConfigError("translit_mode must be one of: token, sentence")

    rendered: list[str] = []
    n_transliterated = 0

    if cfg.translit_mode == "token":
        for tok in tagged:
            if tok.lang != TokenLang.TARGET_ROMAN:
                rendered.append(tok.text if cfg.preserve_case else tok.text.lower())
                continue

            cands = translit_roman_to_native_configured(
                tok.text,
                topk=cfg.topk,
                preserve_case=cfg.preserve_case,
                aggressive_normalize=cfg.aggressive_normalize,
                exceptions=lexicon,
                backend=cfg.translit_backend,
                language=pack.code,
            )
            if not cands:
                rendered.append(tok.text if cfg.preserve_case else tok.text.lower())
                continue
            best = cands[0]
            rendered.append(best)
            if best != tok.text and pack.native_script_re.search(best):
                n_transliterated += 1

        _emit(
            on_event,
            {
                "stage": "transliterate",
                "mode": "token",
                "n_gu_roman_transliterated": n_transliterated,
            },
        )
        return rendered, n_transliterated

    # sentence mode
    i = 0
    while i < len(tagged):
        tok = tagged[i]
        if tok.lang != TokenLang.TARGET_ROMAN:
            rendered.append(tok.text if cfg.preserve_case else tok.text.lower())
            i += 1
            continue

        # Include common joiners (e.g. "hu-maru-naam") as part of a roman span so that
        # phrase-level transliteration can improve backend hit-rate.
        j = i
        span: list[Token] = []
        while j < len(tagged):
            cur = tagged[j]
            if cur.lang == TokenLang.TARGET_ROMAN:
                span.append(cur)
                j += 1
                continue
            if (
                cur.text in _JOINERS
                and j + 1 < len(tagged)
                and tagged[j + 1].lang == TokenLang.TARGET_ROMAN
            ):
                span.append(cur)  # keep joiner
                span.append(tagged[j + 1])
                j += 2
                continue
            break

        roman_words = [t for t in span if t.lang == TokenLang.TARGET_ROMAN]
        joiners = [t.text for t in span if t.text in _JOINERS and t.lang != TokenLang.TARGET_ROMAN]
        phrase = " ".join(t.text for t in roman_words)
        cands = translit_roman_to_native_configured(
            phrase,
            topk=cfg.topk,
            preserve_case=cfg.preserve_case,
            aggressive_normalize=cfg.aggressive_normalize,
            exceptions=lexicon,
            backend=cfg.translit_backend,
            language=pack.code,
        )
        if cands:
            best = cands[0]
            if pack.native_script_re.search(best):
                out_toks = tokenize(best)
                if joiners:
                    # Preserve joiners only when we can align word-to-word.
                    if len(out_toks) == len(roman_words):
                        for idx, w in enumerate(out_toks):
                            rendered.append(w)
                            if idx < len(joiners):
                                rendered.append(joiners[idx])
                        n_transliterated += len(roman_words)
                        i = j
                        continue
                else:
                    # Tokenize native-script output so spacing/punct render stays consistent.
                    rendered.extend(out_toks)
                    n_transliterated += len(roman_words)
                    i = j
                    continue

        # Fallback: token-by-token.
        for t in span:
            if t.lang != TokenLang.TARGET_ROMAN:
                rendered.append(t.text)
                continue
            cands = translit_roman_to_native_configured(
                t.text,
                topk=cfg.topk,
                preserve_case=cfg.preserve_case,
                aggressive_normalize=cfg.aggressive_normalize,
                exceptions=lexicon,
                backend=cfg.translit_backend,
                language=pack.code,
            )
            if not cands:
                rendered.append(t.text if cfg.preserve_case else t.text.lower())
                continue
            best = cands[0]
            rendered.append(best)
            if best != t.text and pack.native_script_re.search(best):
                n_transliterated += 1

        i = j

    _emit(
        on_event,
        {
            "stage": "transliterate",
            "mode": "sentence",
            "n_gu_roman_transliterated": n_transliterated,
        },
    )
    return rendered, n_transliterated


def render_stage(
    rendered_tokens: list[str], *, config: CodeMixConfig, on_event: Optional[EventHook] = None
) -> str:
    cfg = config.normalized()
    numerals_eff = cfg.numerals_effective()

    out = normalize_text(render_tokens(rendered_tokens), numerals=numerals_eff)
    _emit(on_event, {"stage": "render", "output_len": len(out)})
    return out


class CodeMixPipeline:
    """
    First-class pipeline API:
      normalize -> tokenize -> LID -> transliterate -> render
    """

    def __init__(self, *, config: Optional[CodeMixConfig] = None, on_event: Optional[EventHook] = None):
        self.config = (config or CodeMixConfig()).normalized()
        self.on_event = on_event
        # Cache per-pipeline expensive-ish lookups. Streamlit/batch flows often reuse a pipeline
        # instance across many inputs.
        self._lexicon_cached: Optional[tuple[LexiconLoadResult, dict[str, str], Optional[set[str]]]] = None
        self._dialect_backend_cached: Optional[object] = None
        self._dialect_backend_cached_set: bool = False
        self._dialect_normalizer_cached: Optional[object] = None
        self._dialect_normalizer_cached_set: bool = False

    def _lexicon_bundle(
        self,
    ) -> tuple[LexiconLoadResult, dict[str, str], Optional[set[str]]]:
        if self._lexicon_cached is not None:
            return self._lexicon_cached

        cfg = self.config
        lex_res = (
            _load_user_lexicon_cached(cfg.user_lexicon_path)  # type: ignore[arg-type]
            if cfg.user_lexicon_path
            else LexiconLoadResult(mappings={}, source="none")
        )
        lex = lex_res.mappings
        lex_keys = set(lex.keys()) if lex else None
        self._lexicon_cached = (lex_res, lex, lex_keys)
        return self._lexicon_cached

    def _dialect_backend(self) -> Optional[object]:
        if self._dialect_backend_cached_set:
            return self._dialect_backend_cached
        b = get_dialect_backend(self.config)
        self._dialect_backend_cached = b
        self._dialect_backend_cached_set = True
        return b

    def _dialect_normalizer(self) -> Optional[object]:
        if self._dialect_normalizer_cached_set:
            return self._dialect_normalizer_cached
        n = get_dialect_normalizer(self.config)
        self._dialect_normalizer_cached = n
        self._dialect_normalizer_cached_set = True
        return n

    def run(self, text: str) -> CodeMixPipelineResult:
        raw = text or ""
        cfg = self.config
        pack = get_language_pack(cfg.language)
        lex_res, lex, lex_keys = self._lexicon_bundle()

        norm = normalize_stage(raw, config=cfg, on_event=self.on_event)
        if not norm:
            backend = transliteration_backend_configured(
                preferred=cfg.translit_backend, language=pack.code
            )
            cs = compute_code_switch_metrics([])
            d = detect_dialect_from_tagged_tokens([])
            dn = DialectNormalizationResult(
                dialect=d.dialect, changed=False, tokens_in=[], tokens_out=[], backend="none"
            )
            _emit(
                self.on_event,
                {
                    "stage": "done",
                    "empty_input": True,
                    "backend": backend,
                    "language": pack.code,
                    "user_lexicon": lex_res.source,
                    "cmi": cs.cmi,
                    "switch_points": cs.n_switch_points,
                    "dialect": d.dialect.value,
                    "dialect_backend": d.backend,
                    "dialect_confidence": d.confidence,
                    "dialect_normalized": False,
                },
            )
            return CodeMixPipelineResult(
                raw=raw,
                normalized="",
                tokens=[],
                tagged_tokens=[],
                tagged_tokens_effective=[],
                codeswitch=cs,
                dialect=d,
                dialect_normalization=dn,
                rendered_tokens=[],
                codemix="",
                language=pack.code,
                n_tokens=0,
                n_en_tokens=0,
                n_gu_native_tokens=0,
                n_gu_roman_tokens=0,
                n_gu_roman_transliterated=0,
                transliteration_backend=backend,
            )

        toks = tokenize_stage(norm, on_event=self.on_event)
        tagged = lid_stage(
            toks,
            language=pack.code,
            lexicon_keys=lex_keys,
            fasttext_model_path=cfg.fasttext_model_path,
            user_lexicon_source=lex_res.source,
            on_event=self.on_event,
        )
        cs = compute_code_switch_metrics(tagged)

        if pack.dialect_enabled:
            # Dialect detection is pluggable; default stays offline heuristic.
            if cfg.dialect_force:
                forced = str(cfg.dialect_force).strip().lower().replace("-", "_").replace(" ", "_")
                # Avoid importing GujaratiDialect at module import time; keep it local.
                from .dialects import GujaratiDialect

                try:
                    forced_dialect = GujaratiDialect(forced)  # type: ignore[arg-type]
                except Exception:
                    forced_dialect = GujaratiDialect.UNKNOWN
                d = DialectDetection(
                    dialect=forced_dialect,
                    scores={"forced": 1},
                    markers_found={},
                    backend="forced",
                    confidence=1.0,
                )
            else:
                backend = self._dialect_backend()
                if backend is None:
                    d = DialectDetection(
                        dialect=detect_dialect_from_tagged_tokens(tagged).dialect,
                        scores={},
                        markers_found={},
                        backend="none",
                        confidence=0.0,
                    )
                else:
                    d = backend.detect(text=norm, tagged_tokens=tagged, config=cfg)  # type: ignore[attr-defined]
            normalizer = self._dialect_normalizer()
        else:
            # Fail-safe: dialect stack is Gujarati-only today; disable it for beta languages.
            from .dialects import GujaratiDialect

            d = DialectDetection(
                dialect=GujaratiDialect.UNKNOWN,
                scores={},
                markers_found={},
                backend="disabled_for_language",
                confidence=0.0,
            )
            normalizer = None

        dn = DialectNormalizationResult(
            dialect=d.dialect, changed=False, tokens_in=toks, tokens_out=toks, backend="none"
        )
        dialect_normalized = False
        tagged_eff = tagged
        if (
            pack.dialect_enabled
            and cfg.dialect_normalize
            and normalizer is not None
            and (d.confidence >= float(cfg.dialect_min_confidence))
        ):
            dn = normalizer.normalize(tagged_tokens=tagged, dialect=d.dialect, config=cfg)
            if dn.changed:
                # Re-tag after dialect normalization so transliteration + metrics see consistent TokenLang.
                tagged_eff = tag_tokens(
                    dn.tokens_out,
                    language=pack.code,
                    lexicon_keys=lex_keys,
                    fasttext_model_path=cfg.fasttext_model_path,
                )
                dialect_normalized = True

        n_en = 0
        n_gu_native = 0
        n_gu_roman = 0
        for tok in tagged_eff:
            if tok.lang == TokenLang.EN:
                n_en += 1
            elif tok.lang == TokenLang.TARGET_NATIVE:
                n_gu_native += 1
            elif tok.lang == TokenLang.TARGET_ROMAN:
                n_gu_roman += 1

        rendered, n_transliterated = transliterate_stage(
            tagged_eff,
            config=cfg,
            language=pack.code,
            lexicon=lex,
            on_event=self.on_event,
        )
        out = render_stage(rendered, config=cfg, on_event=self.on_event)

        backend_name = transliteration_backend_configured(
            preferred=cfg.translit_backend, language=pack.code
        )
        _emit(
            self.on_event,
            {
                "stage": "done",
                "backend": backend_name,
                "language": pack.code,
                "n_tokens": len(toks),
                "n_gu_roman_tokens": n_gu_roman,
                "n_gu_roman_transliterated": n_transliterated,
                "user_lexicon": lex_res.source,
                "cmi": cs.cmi,
                "switch_points": cs.n_switch_points,
                "dialect": d.dialect.value,
                "dialect_backend": d.backend,
                "dialect_confidence": d.confidence,
                "dialect_normalized": dialect_normalized,
            },
        )

        return CodeMixPipelineResult(
            raw=raw,
            normalized=norm,
            tokens=toks,
            tagged_tokens=tagged,
            tagged_tokens_effective=tagged_eff,
            codeswitch=cs,
            dialect=d,
            dialect_normalization=dn,
            rendered_tokens=rendered,
            codemix=out,
            language=pack.code,
            n_tokens=len(toks),
            n_en_tokens=n_en,
            n_gu_native_tokens=n_gu_native,
            n_gu_roman_tokens=n_gu_roman,
            n_gu_roman_transliterated=n_transliterated,
            transliteration_backend=backend_name,
        )

    def run_many(self, texts: Sequence[str]) -> list[CodeMixPipelineResult]:
        """
        Batch helper for higher throughput when processing many rows.

        This reuses per-pipeline caches (lexicon, backend objects) and avoids re-creating pipeline
        instances in tight loops.
        """

        out: list[CodeMixPipelineResult] = []
        for t in texts:
            out.append(self.run(t))
        return out
