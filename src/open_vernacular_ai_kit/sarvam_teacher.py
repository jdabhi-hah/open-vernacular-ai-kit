from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .codemix_render import render_codemix_with_config
from .config import CodeMixConfig
from .errors import IntegrationError, InvalidConfigError
from .normalize import normalize_text

TeacherCallFn = Callable[[str], str]

_ALLOWED_TOKEN_TYPES = {
    "context_token",
    "function_word",
    "lexicon",
    "phrase",
    "verb_phrase",
    "dialect_variant",
    "english_keep",
}

_ALLOWED_LANG_HINTS = {"gu", "hi", "mixed", "unknown"}


@dataclass(frozen=True)
class SarvamTeacherInput:
    text: str
    language_hint: str | None = None
    source: str = "unknown"
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class SarvamTeacherTokenCandidate:
    roman: str
    native: str
    candidate_type: str = "lexicon"
    confidence: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "roman": self.roman,
            "native": self.native,
            "type": self.candidate_type,
        }
        if self.confidence is not None:
            out["confidence"] = float(self.confidence)
        if self.notes:
            out["notes"] = self.notes
        return out


@dataclass(frozen=True)
class SarvamTeacherCandidateRecord:
    input: str
    language_hint: str
    source: str
    model: str
    ovak_baseline: str
    sarvam_native: str
    sarvam_canonical: str
    english_tokens_keep: list[str]
    candidate_tokens: list[SarvamTeacherTokenCandidate]
    notes: str = ""
    raw_response: str = ""
    meta: dict[str, Any] | None = None

    def to_dict(self, *, include_raw_response: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "input": self.input,
            "language_hint": self.language_hint,
            "source": self.source,
            "model": self.model,
            "ovak_baseline": self.ovak_baseline,
            "sarvam_native": self.sarvam_native,
            "sarvam_canonical": self.sarvam_canonical,
            "english_tokens_keep": list(self.english_tokens_keep),
            "candidate_tokens": [c.to_dict() for c in self.candidate_tokens],
            "notes": self.notes,
            "meta": self.meta or {},
        }
        if include_raw_response:
            out["raw_response"] = self.raw_response
        return out


def _normalize_language_hint(value: Any, *, default: str = "unknown") -> str:
    s = str(value or "").strip().lower()
    if s in {"gujarati", "gu"}:
        return "gu"
    if s in {"hindi", "hi"}:
        return "hi"
    if s in _ALLOWED_LANG_HINTS:
        return s
    return default


def _clip_notes(value: Any) -> str:
    return str(value or "").strip()[:400]


def _coerce_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        conf = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, conf))


def _parse_candidate_type(value: Any) -> str:
    s = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in _ALLOWED_TOKEN_TYPES:
        return s
    return "lexicon"


def _find_balanced_json_object(text: str) -> str | None:
    in_string = False
    escape = False
    depth = 0
    start = -1

    for idx, ch in enumerate(text):
        if start < 0:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escape = False
            continue

        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise IntegrationError("Sarvam teacher returned an empty response")

    for fenced_marker in ("```json", "```JSON", "```"):
        if fenced_marker not in text:
            continue
        start = text.find(fenced_marker)
        end = text.find("```", start + len(fenced_marker))
        if start >= 0 and end > start:
            candidate = text[start + len(fenced_marker) : end].strip()
            obj = _try_parse_json_object(candidate)
            if obj is not None:
                return obj
            balanced = _find_balanced_json_object(candidate)
            if balanced:
                obj = _try_parse_json_object(balanced)
                if obj is not None:
                    return obj

    balanced = _find_balanced_json_object(text)
    if balanced:
        try:
            obj = json.loads(balanced)
        except Exception as e:
            raise IntegrationError(f"Sarvam teacher returned invalid JSON: {e}") from e
        if isinstance(obj, dict):
            return obj

    raise IntegrationError("Sarvam teacher response did not contain a JSON object")


def build_sarvam_teacher_prompt(
    text: str,
    *,
    language_hint: str | None = None,
    ovak_baseline: str = "",
) -> str:
    lang = _normalize_language_hint(language_hint)
    example_output = {
        "language_hint": "gu",
        "sarvam_native": "તમને આજે office માં આવવું છે",
        "sarvam_canonical": "તમને આજે office માં આવવું છે",
        "english_tokens_keep": ["office"],
        "candidate_tokens": [
            {
                "roman": "ma",
                "native": "માં",
                "type": "context_token",
                "confidence": 0.98,
                "notes": "locative postposition in Gujarati context",
            }
        ],
        "notes": "Keep obvious English tokens in Latin script.",
    }
    instructions = [
        "You are helping improve a deterministic multilingual normalization toolkit.",
        "Return only one JSON object. Do not include markdown, prose, or code fences.",
        "Infer the intended vernacular rendering of the input.",
        "Keep obvious English tokens in Latin script.",
        "Only include high-confidence candidate tokens.",
        "If uncertain, leave candidate_tokens empty and explain in notes.",
    ]
    payload = {
        "task": "Analyze one messy Hindi/Gujarati/code-mixed input for normalization improvement.",
        "language_hint": lang,
        "input": text,
        "ovak_baseline": ovak_baseline,
        "required_output_schema": {
            "language_hint": "gu | hi | mixed | unknown",
            "sarvam_native": "native-script or mixed-script rendering",
            "sarvam_canonical": "preferred normalized rendering",
            "english_tokens_keep": ["list of tokens that should stay English"],
            "candidate_tokens": [
                {
                    "roman": "romanized token or phrase",
                    "native": "native-script rendering",
                    "type": "context_token | function_word | lexicon | phrase | verb_phrase | dialect_variant | english_keep",
                    "confidence": "0.0 to 1.0",
                    "notes": "short explanation",
                }
            ],
            "notes": "short ambiguity notes",
        },
        "example_output": example_output,
    }
    return "\n".join(instructions) + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_sarvam_teacher_response(
    raw_response: str,
    *,
    input_text: str,
    source: str,
    model: str,
    ovak_baseline: str,
    meta: dict[str, Any] | None = None,
    fallback_language_hint: str | None = None,
) -> SarvamTeacherCandidateRecord:
    obj = _extract_json_object(raw_response)

    lang = _normalize_language_hint(obj.get("language_hint"), default=_normalize_language_hint(fallback_language_hint))
    sarvam_native = str(obj.get("sarvam_native", "") or "").strip()
    sarvam_canonical = str(obj.get("sarvam_canonical", "") or "").strip()
    if not sarvam_native:
        sarvam_native = sarvam_canonical
    if not sarvam_canonical:
        sarvam_canonical = sarvam_native
    if not sarvam_canonical:
        raise IntegrationError("Sarvam teacher response did not include sarvam_native or sarvam_canonical")

    english_keep_raw = obj.get("english_tokens_keep")
    english_keep = []
    if isinstance(english_keep_raw, list):
        for token in english_keep_raw:
            s = str(token or "").strip()
            if s:
                english_keep.append(s)

    candidates_raw = obj.get("candidate_tokens")
    candidates: list[SarvamTeacherTokenCandidate] = []
    if isinstance(candidates_raw, list):
        for rec in candidates_raw:
            if not isinstance(rec, dict):
                continue
            roman = str(rec.get("roman", "") or "").strip()
            native = str(rec.get("native", "") or "").strip()
            if not roman or not native:
                continue
            candidates.append(
                SarvamTeacherTokenCandidate(
                    roman=roman,
                    native=native,
                    candidate_type=_parse_candidate_type(rec.get("type")),
                    confidence=_coerce_confidence(rec.get("confidence")),
                    notes=_clip_notes(rec.get("notes")),
                )
            )

    return SarvamTeacherCandidateRecord(
        input=input_text,
        language_hint=lang,
        source=source,
        model=model,
        ovak_baseline=ovak_baseline,
        sarvam_native=sarvam_native,
        sarvam_canonical=sarvam_canonical,
        english_tokens_keep=english_keep,
        candidate_tokens=candidates,
        notes=_clip_notes(obj.get("notes")),
        raw_response=str(raw_response or ""),
        meta=meta or {},
    )


def _default_ovak_baseline(text: str, *, language_hint: str | None = None) -> str:
    lang = _normalize_language_hint(language_hint)
    if lang in {"gu", "hi"}:
        cfg = CodeMixConfig(language=lang, translit_mode="sentence")
        return render_codemix_with_config(text, config=cfg)
    return normalize_text(text)


def mine_sarvam_teacher_candidate(
    text: str,
    *,
    model: str = "sarvam-m",
    api_key: Optional[str] = None,
    language_hint: str | None = None,
    source: str = "teacher_mining",
    meta: dict[str, Any] | None = None,
    ovak_baseline: str | None = None,
    call_model: TeacherCallFn | None = None,
) -> SarvamTeacherCandidateRecord:
    if not str(text or "").strip():
        raise InvalidConfigError("text is required")

    baseline = ovak_baseline if ovak_baseline is not None else _default_ovak_baseline(
        text, language_hint=language_hint
    )
    prompt = build_sarvam_teacher_prompt(
        text,
        language_hint=language_hint,
        ovak_baseline=baseline,
    )

    if call_model is None:
        from .sarvam_adapters import sarvam_chat

        def _call(p: str) -> str:
            return sarvam_chat(
                p,
                model=model,
                api_key=api_key,
                preprocess=False,
                temperature=0,
            )

        call_model = _call

    raw = call_model(prompt)
    return parse_sarvam_teacher_response(
        raw,
        input_text=text,
        source=source,
        model=model,
        ovak_baseline=baseline,
        meta=meta,
        fallback_language_hint=language_hint,
    )


def load_sarvam_teacher_inputs_jsonl(path: str | Path) -> list[SarvamTeacherInput]:
    out: list[SarvamTeacherInput] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            rec = json.loads(s)
            if not isinstance(rec, dict):
                continue
            text = str(rec.get("text", rec.get("input", "")) or "").strip()
            if not text:
                continue
            out.append(
                SarvamTeacherInput(
                    text=text,
                    language_hint=_normalize_language_hint(rec.get("language_hint"), default="unknown"),
                    source=str(rec.get("source", "unknown") or "unknown"),
                    meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else None),
                )
            )
    return out


def load_sarvam_teacher_records_jsonl(path: str | Path) -> list[SarvamTeacherCandidateRecord]:
    rows: list[SarvamTeacherCandidateRecord] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            rec = json.loads(s)
            if not isinstance(rec, dict):
                continue
            english_keep = []
            english_keep_raw = rec.get("english_tokens_keep")
            if isinstance(english_keep_raw, list):
                for token in english_keep_raw:
                    value = str(token or "").strip()
                    if value:
                        english_keep.append(value)

            candidates: list[SarvamTeacherTokenCandidate] = []
            candidates_raw = rec.get("candidate_tokens")
            if isinstance(candidates_raw, list):
                for candidate in candidates_raw:
                    if not isinstance(candidate, dict):
                        continue
                    roman = str(candidate.get("roman", "") or "").strip()
                    native = str(candidate.get("native", "") or "").strip()
                    if not roman or not native:
                        continue
                    candidates.append(
                        SarvamTeacherTokenCandidate(
                            roman=roman,
                            native=native,
                            candidate_type=_parse_candidate_type(candidate.get("type")),
                            confidence=_coerce_confidence(candidate.get("confidence")),
                            notes=_clip_notes(candidate.get("notes")),
                        )
                    )
            rows.append(
                SarvamTeacherCandidateRecord(
                    input=str(rec.get("input", "") or ""),
                    language_hint=_normalize_language_hint(rec.get("language_hint")),
                    source=str(rec.get("source", "unknown") or "unknown"),
                    model=str(rec.get("model", "sarvam-m") or "sarvam-m"),
                    ovak_baseline=str(rec.get("ovak_baseline", "") or ""),
                    sarvam_native=str(rec.get("sarvam_native", "") or "").strip(),
                    sarvam_canonical=str(rec.get("sarvam_canonical", "") or "").strip(),
                    english_tokens_keep=english_keep,
                    candidate_tokens=candidates,
                    notes=_clip_notes(rec.get("notes")),
                    raw_response=str(rec.get("raw_response", "") or ""),
                    meta=(rec.get("meta") if isinstance(rec.get("meta"), dict) else None),
                )
            )
    return rows


def dump_sarvam_teacher_records_jsonl(
    path: str | Path,
    records: Iterable[SarvamTeacherCandidateRecord],
    *,
    include_raw_response: bool = True,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(include_raw_response=include_raw_response), ensure_ascii=False) + "\n")
