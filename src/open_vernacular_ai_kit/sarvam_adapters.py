from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional

from .codemix_render import render_codemix
from .errors import IntegrationError, OptionalDependencyError
from .normalize import normalize_text


@dataclass(frozen=True)
class SarvamConfig:
    api_key: str


def _get_sarvam_client(api_key: Optional[str] = None):
    """Create a SarvamAI client if `sarvamai` is installed."""
    try:
        from sarvamai import SarvamAI
    except Exception as e:
        raise OptionalDependencyError(
            "Current hosted-provider integration is Sarvam (`sarvamai`). Install with: pip install -e '.[sarvam]'"
        ) from e

    key = api_key or os.environ.get("SARVAM_API_KEY")
    if not key:
        raise IntegrationError("Missing SARVAM_API_KEY environment variable (or pass api_key=...)")
    return SarvamAI(api_subscription_key=key)


def sarvam_translate_text(
    text: str,
    *,
    source_language_code: str = "auto",
    target_language_code: str = "en-IN",
    model: str = "mayura:v1",
    api_key: Optional[str] = None,
    preprocess: bool = True,
) -> Any:
    """
    Convenience wrapper around Sarvam translate API.

    For code-mixed Indian-vernacular inputs, `preprocess=True` will normalize + codemix-render
    to improve consistency before translation.
    """
    client = _get_sarvam_client(api_key=api_key)

    inp = text
    if preprocess:
        inp = render_codemix(normalize_text(text))

    return client.text.translate(
        input=inp,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        model=model,
    )


def sarvam_chat(
    user_text: str,
    *,
    model: str = "sarvam-m",
    api_key: Optional[str] = None,
    preprocess: bool = True,
    **kwargs: Any,
) -> str:
    """Call Sarvam chat completions with a single user message."""
    client = _get_sarvam_client(api_key=api_key)

    content = user_text
    if preprocess:
        content = render_codemix(normalize_text(user_text))

    completion_kwargs = {
        "messages": [{"role": "user", "content": content}],
        **kwargs,
    }
    try:
        params = inspect.signature(client.chat.completions).parameters
    except Exception:
        params = {}
    if "model" in params:
        completion_kwargs["model"] = model

    resp = client.chat.completions(**completion_kwargs)

    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)
 
