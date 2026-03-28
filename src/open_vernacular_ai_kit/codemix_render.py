from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .codeswitch import CodeSwitchMetrics
from .config import CodeMixConfig
from .dialects import DialectDetection, DialectNormalizationResult
from .pipeline import CodeMixPipeline, CodeMixPipelineResult, EventHook


@dataclass(frozen=True)
class CodeMixAnalysis:
    """
    Product-facing analysis of CodeMix rendering.

    Canonical output format for this kit is the `codemix` string:
    native-script tokens stay native, English stays Latin, and romanized tokens are transliterated when possible.
    """

    raw: str
    normalized: str
    codemix: str
    language: str

    # v0.4 additions: code-switching + dialect heuristics (MVP).
    codeswitch: CodeSwitchMetrics
    dialect: DialectDetection
    dialect_normalization: DialectNormalizationResult

    n_tokens: int
    n_en_tokens: int
    n_gu_native_tokens: int
    n_gu_roman_tokens: int
    n_gu_roman_transliterated: int
    pct_gu_roman_transliterated: float

    transliteration_backend: str


def _result_to_analysis(result: CodeMixPipelineResult) -> CodeMixAnalysis:
    pct = (
        (result.n_gu_roman_transliterated / result.n_gu_roman_tokens)
        if result.n_gu_roman_tokens
        else 0.0
    )
    return CodeMixAnalysis(
        raw=result.raw,
        normalized=result.normalized,
        codemix=result.codemix,
        language=result.language,
        codeswitch=result.codeswitch,
        dialect=result.dialect,
        dialect_normalization=result.dialect_normalization,
        n_tokens=result.n_tokens,
        n_en_tokens=result.n_en_tokens,
        n_gu_native_tokens=result.n_gu_native_tokens,
        n_gu_roman_tokens=result.n_gu_roman_tokens,
        n_gu_roman_transliterated=result.n_gu_roman_transliterated,
        pct_gu_roman_transliterated=pct,
        transliteration_backend=result.transliteration_backend,
    )


def analyze_codemix_with_config(
    text: str,
    *,
    config: CodeMixConfig,
    on_event: Optional[EventHook] = None,
) -> CodeMixAnalysis:
    """
    Analyze + render CodeMix using a stable `CodeMixConfig`.
    """

    res = CodeMixPipeline(config=config, on_event=on_event).run(text)
    return _result_to_analysis(res)


def render_codemix_with_config(
    text: str,
    *,
    config: CodeMixConfig,
    on_event: Optional[EventHook] = None,
) -> str:
    """Render CodeMix using a stable `CodeMixConfig`."""

    return CodeMixPipeline(config=config, on_event=on_event).run(text).codemix


def analyze_codemix(
    text: str,
    *,
    language: str = "gu",
    topk: int = 1,
    numerals: str = "keep",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    translit_mode: str = "token",
    translit_backend: str = "auto",
    user_lexicon_path: Optional[str] = None,
    fasttext_model_path: Optional[str] = None,
    dialect_backend: str = "auto",
    dialect_model_id_or_path: Optional[str] = None,
    dialect_min_confidence: float = 0.70,
    dialect_normalize: bool = False,
    dialect_force: Optional[str] = None,
    dialect_normalizer_backend: str = "auto",
    dialect_normalizer_model_id_or_path: Optional[str] = None,
    allow_remote_models: bool = False,
) -> CodeMixAnalysis:
    """
    Analyze + render CodeMix in one pass.

    The primary "success metric" is `pct_gu_roman_transliterated`, i.e. the estimated fraction of
    Gujarati-roman tokens that were converted into native script.
    """
    cfg = CodeMixConfig(
        language=language,  # type: ignore[arg-type]
        topk=topk,
        numerals=numerals,  # type: ignore[arg-type]
        preserve_case=preserve_case,
        preserve_numbers=preserve_numbers,
        aggressive_normalize=aggressive_normalize,
        translit_mode=translit_mode,  # type: ignore[arg-type]
        translit_backend=translit_backend,  # type: ignore[arg-type]
        user_lexicon_path=user_lexicon_path,
        fasttext_model_path=fasttext_model_path,
        dialect_backend=dialect_backend,  # type: ignore[arg-type]
        dialect_model_id_or_path=dialect_model_id_or_path,
        dialect_min_confidence=float(dialect_min_confidence),
        dialect_normalize=bool(dialect_normalize),
        dialect_force=dialect_force,
        dialect_normalizer_backend=dialect_normalizer_backend,  # type: ignore[arg-type]
        dialect_normalizer_model_id_or_path=dialect_normalizer_model_id_or_path,
        allow_remote_models=bool(allow_remote_models),
    )
    return analyze_codemix_with_config(text, config=cfg)


def render_codemix(
    text: str,
    *,
    language: str = "gu",
    topk: int = 1,
    numerals: str = "keep",
    preserve_case: bool = True,
    preserve_numbers: bool = True,
    aggressive_normalize: bool = False,
    translit_mode: str = "token",
    translit_backend: str = "auto",
    user_lexicon_path: Optional[str] = None,
    fasttext_model_path: Optional[str] = None,
    dialect_backend: str = "auto",
    dialect_model_id_or_path: Optional[str] = None,
    dialect_min_confidence: float = 0.70,
    dialect_normalize: bool = False,
    dialect_force: Optional[str] = None,
    dialect_normalizer_backend: str = "auto",
    dialect_normalizer_model_id_or_path: Optional[str] = None,
    allow_remote_models: bool = False,
) -> str:
    """
    Convert mixed vernacular/English text into a stable code-mix representation:

    - Native-script tokens stay in native script
    - English stays in Latin
    - Romanized tokens are transliterated to native script when possible
    """
    cfg = CodeMixConfig(
        language=language,  # type: ignore[arg-type]
        topk=topk,
        numerals=numerals,  # type: ignore[arg-type]
        preserve_case=preserve_case,
        preserve_numbers=preserve_numbers,
        aggressive_normalize=aggressive_normalize,
        translit_mode=translit_mode,  # type: ignore[arg-type]
        translit_backend=translit_backend,  # type: ignore[arg-type]
        user_lexicon_path=user_lexicon_path,
        fasttext_model_path=fasttext_model_path,
        dialect_backend=dialect_backend,  # type: ignore[arg-type]
        dialect_model_id_or_path=dialect_model_id_or_path,
        dialect_min_confidence=float(dialect_min_confidence),
        dialect_normalize=bool(dialect_normalize),
        dialect_force=dialect_force,
        dialect_normalizer_backend=dialect_normalizer_backend,  # type: ignore[arg-type]
        dialect_normalizer_model_id_or_path=dialect_normalizer_model_id_or_path,
        allow_remote_models=bool(allow_remote_models),
    )
    return render_codemix_with_config(text, config=cfg)
 
