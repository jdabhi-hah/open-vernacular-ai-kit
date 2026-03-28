from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from open_vernacular_ai_kit.api_service import API_SCHEMA_VERSION, app
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]
    API_SCHEMA_VERSION = 1  # type: ignore[assignment]
    app = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    TestClient is None or app is None, reason="FastAPI service dependencies are not installed."
)


client = TestClient(app) if (TestClient is not None and app is not None) else None


def test_normalize_backward_compat_without_schema_version() -> None:
    assert client is not None
    r = client.post("/normalize", json={"text": "Hello   world!!!"})
    assert r.status_code == 200
    data = r.json()
    assert data["schema_version"] == API_SCHEMA_VERSION
    assert data["normalized"] == "Hello world!!"


def test_codemix_ignores_unknown_fields() -> None:
    assert client is not None
    r = client.post(
        "/codemix",
        json={
            "text": "maru plan ready chhe!!!",
            "unknown_top_level_field": "ignored",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["schema_version"] == API_SCHEMA_VERSION
    assert data["language"] == "gu"
    assert "codemix" in data
    assert "n_gu_roman_tokens" in data


def test_codemix_accepts_hindi_language_profile() -> None:
    assert client is not None
    r = client.post(
        "/codemix",
        json={
            "text": "mera naam Sudhir hai",
            "config": {"language": "hi", "translit_mode": "sentence"},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["language"] == "hi"
    assert "मेरा" in data["codemix"]


def test_analyze_with_config_dict() -> None:
    assert client is not None
    r = client.post(
        "/analyze",
        json={
            "schema_version": API_SCHEMA_VERSION,
            "text": "maru plan ready chhe!!!",
            "config": {"translit_mode": "sentence", "topk": 1},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["schema_version"] == API_SCHEMA_VERSION
    assert "analysis" in data
    assert "codemix" in data["analysis"]
    assert "n_tokens" in data["analysis"]


def test_rejects_future_schema_version() -> None:
    assert client is not None
    r = client.post("/normalize", json={"schema_version": API_SCHEMA_VERSION + 1, "text": "hi"})
    assert r.status_code == 400
    assert "Unsupported schema_version" in str(r.json().get("detail"))


def test_rejects_invalid_config() -> None:
    assert client is not None
    r = client.post(
        "/codemix",
        json={
            "text": "maru plan ready chhe!!!",
            "config": {"translit_mode": "invalid-mode"},
        },
    )
    assert r.status_code == 400
