from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Optional

from .errors import InvalidConfigError
from .language_packs import DEFAULT_LANGUAGE, is_supported_language, normalize_language_code

NumeralsMode = Literal["keep", "ascii"]
TranslitMode = Literal["token", "sentence"]
TranslitBackend = Literal["auto", "ai4bharat", "sanscript", "none"]
DialectBackend = Literal["auto", "heuristic", "transformers", "llm", "none"]
DialectNormalizerBackend = Literal["auto", "heuristic", "seq2seq", "none"]
LanguageCode = Literal["gu", "hi"]


@dataclass(frozen=True)
class CodeMixConfig:
    """
    Stable, SDK-first configuration for CodeMix processing.

    The CLI should map flags -> this object; the SDK should accept this object directly.
    """

    # Target language profile for code-mix normalization/transliteration.
    language: LanguageCode = "gu"

    # Text normalization
    numerals: NumeralsMode = "keep"
    preserve_numbers: bool = True
    preserve_case: bool = True

    # Transliteration
    topk: int = 1
    aggressive_normalize: bool = False
    translit_mode: TranslitMode = "token"
    translit_backend: TranslitBackend = "auto"
    user_lexicon_path: Optional[str] = None

    # Determinism hook for future stochastic components.
    seed: Optional[int] = None

    # Optional LID signal for Latin tokens (fastText model path for lid.176.ftz).
    fasttext_model_path: Optional[str] = None

    # Dialect utilities (v0.4.x+): detection + optional normalization.
    dialect_backend: DialectBackend = "auto"
    dialect_model_id_or_path: Optional[str] = None
    dialect_min_confidence: float = 0.70
    dialect_normalize: bool = False
    dialect_force: Optional[str] = None
    dialect_normalizer_backend: DialectNormalizerBackend = "auto"
    dialect_normalizer_model_id_or_path: Optional[str] = None

    # Safety: offline-first behavior. When False, any HF model-id usage should be considered
    # an error unless the model is already present on disk.
    allow_remote_models: bool = False

    # Serialization schema version for backwards-compatible config dicts.
    # NOTE: keep this field last to avoid breaking positional construction.
    schema_version: int = 1

    def numerals_effective(self) -> NumeralsMode:
        """
        Effective numerals behavior after considering `preserve_numbers`.

        For backward compatibility, `preserve_numbers=False` forces ASCII numerals even if
        `numerals="keep"`.
        """

        if not self.preserve_numbers:
            return "ascii"
        return self.numerals

    def normalized(self) -> "CodeMixConfig":
        """Return a defensively normalized config (types/constraints)."""

        topk = max(1, int(self.topk))
        language = normalize_language_code(str(self.language))
        if not is_supported_language(language):
            # Fail-safe fallback for forwards compatibility with external config payloads.
            language = DEFAULT_LANGUAGE
        numerals: NumeralsMode = self.numerals
        if numerals not in ("keep", "ascii"):
            raise InvalidConfigError("numerals must be one of: keep, ascii")
        if self.translit_mode not in ("token", "sentence"):
            raise InvalidConfigError("translit_mode must be one of: token, sentence")
        if self.translit_backend not in ("auto", "ai4bharat", "sanscript", "none"):
            raise InvalidConfigError("translit_backend must be one of: auto, ai4bharat, sanscript, none")

        if self.dialect_backend not in ("auto", "heuristic", "transformers", "llm", "none"):
            raise InvalidConfigError(
                "dialect_backend must be one of: auto, heuristic, transformers, llm, none"
            )
        if self.dialect_normalizer_backend not in ("auto", "heuristic", "seq2seq", "none"):
            raise InvalidConfigError(
                "dialect_normalizer_backend must be one of: auto, heuristic, seq2seq, none"
            )
        if self.dialect_force is not None:
            v = str(self.dialect_force).strip().lower().replace("-", "_").replace(" ", "_")
            allowed = {
                "unknown",
                "standard",
                "kathiawadi",
                "surati",
                "charotar",
                "north_gujarat",
            }
            if v not in allowed:
                raise InvalidConfigError(
                    f"dialect_force must be one of: {', '.join(sorted(allowed))}"
                )
        dialect_min_conf = float(self.dialect_min_confidence)
        if not (0.0 <= dialect_min_conf <= 1.0):
            raise InvalidConfigError("dialect_min_confidence must be in [0.0, 1.0]")

        seed = None if self.seed is None else int(self.seed)
        schema_version = max(1, int(self.schema_version))
        return CodeMixConfig(
            language=language,  # type: ignore[arg-type]
            numerals=numerals,
            preserve_numbers=bool(self.preserve_numbers),
            preserve_case=bool(self.preserve_case),
            topk=topk,
            aggressive_normalize=bool(self.aggressive_normalize),
            translit_mode=self.translit_mode,
            translit_backend=self.translit_backend,
            user_lexicon_path=None if self.user_lexicon_path is None else str(self.user_lexicon_path),
            seed=seed,
            fasttext_model_path=(
                None if self.fasttext_model_path is None else str(self.fasttext_model_path)
            ),
            dialect_backend=self.dialect_backend,
            dialect_model_id_or_path=(
                None if self.dialect_model_id_or_path is None else str(self.dialect_model_id_or_path)
            ),
            dialect_min_confidence=dialect_min_conf,
            dialect_normalize=bool(self.dialect_normalize),
            dialect_force=(None if self.dialect_force is None else str(self.dialect_force)),
            dialect_normalizer_backend=self.dialect_normalizer_backend,
            dialect_normalizer_model_id_or_path=(
                None
                if self.dialect_normalizer_model_id_or_path is None
                else str(self.dialect_normalizer_model_id_or_path)
            ),
            allow_remote_models=bool(self.allow_remote_models),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        # Keep it JSON-friendly (no callables, regex, etc.)
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, object], *, strict: bool = False) -> "CodeMixConfig":
        """
        Load a config from a JSON-friendly dict.

        Backward compatibility policy:
          - Older dicts without `schema_version` are accepted.
          - Unknown keys are ignored by default (strict=False).
          - Values are coerced conservatively (e.g., "1" -> 1) where safe.
        """

        if not isinstance(data, dict):
            raise InvalidConfigError("config must be a dict")

        allowed = {f.name for f in fields(cls)}
        unknown = sorted([k for k in data.keys() if k not in allowed])
        if unknown and strict:
            raise InvalidConfigError(f"Unknown config keys: {', '.join(unknown)}")

        def as_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v or "").strip().lower()
            if s in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "f", "no", "n", "off"}:
                return False
            return bool(v)

        def as_int(v: Any, *, default: int) -> int:
            if v is None:
                return int(default)
            try:
                return int(v)
            except Exception:
                return int(default)

        def as_float(v: Any, *, default: float) -> float:
            if v is None:
                return float(default)
            try:
                return float(v)
            except Exception:
                return float(default)

        def as_opt_str(v: Any) -> Optional[str]:
            if v is None:
                return None
            s = str(v)
            s = s.strip()
            return s if s else None

        # NOTE: most string literals are validated in `.normalized()`; we just coerce to str here.
        kwargs: dict[str, Any] = {}
        if "language" in data:
            kwargs["language"] = str(data.get("language") or "").strip() or DEFAULT_LANGUAGE
        if "numerals" in data:
            kwargs["numerals"] = str(data.get("numerals") or "").strip() or "keep"
        if "preserve_numbers" in data:
            kwargs["preserve_numbers"] = as_bool(data.get("preserve_numbers"))
        if "preserve_case" in data:
            kwargs["preserve_case"] = as_bool(data.get("preserve_case"))

        if "topk" in data:
            kwargs["topk"] = as_int(data.get("topk"), default=1)
        if "aggressive_normalize" in data:
            kwargs["aggressive_normalize"] = as_bool(data.get("aggressive_normalize"))
        if "translit_mode" in data:
            kwargs["translit_mode"] = str(data.get("translit_mode") or "").strip() or "token"
        if "translit_backend" in data:
            kwargs["translit_backend"] = str(data.get("translit_backend") or "").strip() or "auto"
        if "user_lexicon_path" in data:
            kwargs["user_lexicon_path"] = as_opt_str(data.get("user_lexicon_path"))

        if "seed" in data:
            v = data.get("seed")
            kwargs["seed"] = None if v is None else as_int(v, default=0)
        if "fasttext_model_path" in data:
            kwargs["fasttext_model_path"] = as_opt_str(data.get("fasttext_model_path"))

        if "dialect_backend" in data:
            kwargs["dialect_backend"] = str(data.get("dialect_backend") or "").strip() or "auto"
        if "dialect_model_id_or_path" in data:
            kwargs["dialect_model_id_or_path"] = as_opt_str(data.get("dialect_model_id_or_path"))
        if "dialect_min_confidence" in data:
            kwargs["dialect_min_confidence"] = as_float(data.get("dialect_min_confidence"), default=0.70)
        if "dialect_normalize" in data:
            kwargs["dialect_normalize"] = as_bool(data.get("dialect_normalize"))
        if "dialect_force" in data:
            kwargs["dialect_force"] = as_opt_str(data.get("dialect_force"))
        if "dialect_normalizer_backend" in data:
            kwargs["dialect_normalizer_backend"] = (
                str(data.get("dialect_normalizer_backend") or "").strip() or "auto"
            )
        if "dialect_normalizer_model_id_or_path" in data:
            kwargs["dialect_normalizer_model_id_or_path"] = as_opt_str(
                data.get("dialect_normalizer_model_id_or_path")
            )
        if "allow_remote_models" in data:
            kwargs["allow_remote_models"] = as_bool(data.get("allow_remote_models"))

        # v1: schema_version introduced. Older dicts may not have it.
        kwargs["schema_version"] = as_int(data.get("schema_version"), default=1)

        return cls(**kwargs).normalized()
