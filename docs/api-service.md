# API Service

The project includes an optional FastAPI wrapper for serving normalization and codemix endpoints.

## Install

```bash
pip install -e ".[api,indic,ml,lexicon]"
```

## Run locally

```bash
uvicorn open_vernacular_ai_kit.api_service:app --host 0.0.0.0 --port 8000
```

OpenAPI docs:

- `http://localhost:8000/docs`

## Endpoints

- `GET /healthz`
- `POST /normalize`
- `POST /codemix`
- `POST /analyze`

## Request/Response schema versioning

Current API schema version: `1`.

Backward-compat behavior:

- If `schema_version` is missing, the server assumes `1`.
- Unknown request fields are ignored.
- If `schema_version` is higher than supported, request fails with `400`.

All endpoint responses include:

- `schema_version`

## Example requests

Normalize:

```bash
curl -s http://localhost:8000/normalize \
  -H 'content-type: application/json' \
  -d '{"text":"Hello   world!!!"}'
```

Codemix:

```bash
curl -s http://localhost:8000/codemix \
  -H 'content-type: application/json' \
  -d '{"text":"mera naam Sudhir hai","config":{"language":"hi","translit_mode":"sentence"}}'
```

`/codemix` responses include the effective `language` profile.

Analyze:

```bash
curl -s http://localhost:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"schema_version":1,"text":"maru plan ready chhe!!!"}'
```
