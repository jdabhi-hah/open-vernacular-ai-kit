from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]

from .codemix_render import analyze_codemix_with_config
from .config import CodeMixConfig
from .errors import InvalidConfigError
from .normalize import normalize_text

API_SCHEMA_VERSION = 1


class ApiModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="ignore")
    else:  # pragma: no cover
        class Config:
            # Backward compatibility: ignore unknown request fields by default.
            extra = "ignore"


class NormalizeRequest(ApiModel):
    schema_version: int = Field(default=API_SCHEMA_VERSION, ge=1)
    text: str
    numerals: str = "keep"


class PipelineRequest(ApiModel):
    schema_version: int = Field(default=API_SCHEMA_VERSION, ge=1)
    text: str
    config: Optional[dict[str, Any]] = None


class NormalizeResponse(ApiModel):
    schema_version: int
    normalized: str


class CodemixResponse(ApiModel):
    schema_version: int
    language: str
    codemix: str
    transliteration_backend: str
    n_tokens: int
    n_gu_roman_tokens: int
    n_gu_roman_transliterated: int
    pct_gu_roman_transliterated: float


class AnalyzeResponse(ApiModel):
    schema_version: int
    analysis: dict[str, Any]


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, Enum):
        return x.value
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _ensure_supported_schema_version(v: int) -> None:
    # Backward-compat policy:
    # - missing schema_version -> defaults to current
    # - equal version -> accepted
    # - future version -> explicit client error
    if int(v) > API_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported schema_version={v}. "
                f"Server supports <= {API_SCHEMA_VERSION}."
            ),
        )


def _config_from_dict(data: Optional[dict[str, Any]]) -> CodeMixConfig:
    if not data:
        return CodeMixConfig()
    return CodeMixConfig.from_dict(data, strict=False)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Open Vernacular AI Kit API",
        version=str(API_SCHEMA_VERSION),
        description=(
            "FastAPI wrapper for open-vernacular-ai-kit normalization and codemix analysis."
        ),
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"ok": True, "schema_version": API_SCHEMA_VERSION}

    @app.post("/normalize", response_model=NormalizeResponse)
    def normalize_endpoint(req: NormalizeRequest) -> NormalizeResponse:
        _ensure_supported_schema_version(req.schema_version)
        try:
            normalized = normalize_text(req.text or "", numerals=req.numerals)
        except InvalidConfigError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return NormalizeResponse(schema_version=API_SCHEMA_VERSION, normalized=normalized)

    @app.post("/codemix", response_model=CodemixResponse)
    def codemix_endpoint(req: PipelineRequest) -> CodemixResponse:
        _ensure_supported_schema_version(req.schema_version)
        try:
            cfg = _config_from_dict(req.config)
            a = analyze_codemix_with_config(req.text or "", config=cfg)
        except InvalidConfigError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return CodemixResponse(
            schema_version=API_SCHEMA_VERSION,
            language=a.language,
            codemix=a.codemix,
            transliteration_backend=a.transliteration_backend,
            n_tokens=int(a.n_tokens),
            n_gu_roman_tokens=int(a.n_gu_roman_tokens),
            n_gu_roman_transliterated=int(a.n_gu_roman_transliterated),
            pct_gu_roman_transliterated=float(a.pct_gu_roman_transliterated),
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze_endpoint(req: PipelineRequest) -> AnalyzeResponse:
        _ensure_supported_schema_version(req.schema_version)
        try:
            cfg = _config_from_dict(req.config)
            a = analyze_codemix_with_config(req.text or "", config=cfg)
        except InvalidConfigError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        payload = _to_jsonable(asdict(a))
        return AnalyzeResponse(schema_version=API_SCHEMA_VERSION, analysis=payload)

    return app


app = create_app()
