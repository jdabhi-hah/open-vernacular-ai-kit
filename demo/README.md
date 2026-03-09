# Web Demo (Streamlit)

This folder contains a web demo for Open Vernacular AI Kit (India-first release).

The demo is designed for non-technical users too:

- Paste a message (WhatsApp-style mixed Indian vernacular + English, including romanized text).
- Choose a `Gujarati` or `Hindi` language profile in the UI.
- See "Before vs After" (what a user wrote vs what we send to AI/search).
- Optionally compare Sarvam-M outputs (requires `SARVAM_API_KEY`).
- Choose runtime mode: in-process SDK (default) or API service mode.
- v0.5: try the RAG panel to retrieve from a tiny packaged India-focused mini-KB.

## Run locally

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e ".[demo,indic]"
streamlit run demo/streamlit_app.py
```

Optional (run demo against API service mode):

```bash
.venv/bin/pip install -e ".[api,indic,ml,lexicon,demo]"
uvicorn open_vernacular_ai_kit.api_service:app --host 0.0.0.0 --port 8000
# In another terminal:
OVAK_API_BASE_URL=http://localhost:8000 streamlit run demo/streamlit_app.py
```

Optional (to enable AI comparison):

```bash
.venv/bin/pip install -e ".[sarvam]"
export SARVAM_API_KEY="..."
streamlit run demo/streamlit_app.py
```

Optional (to enable Transformers/seq2seq dialect backends):

```bash
.venv/bin/pip install -e ".[dialect-ml]"
```

Optional (to enable semantic HF embeddings in the v0.5 RAG panel):

```bash
.venv/bin/pip install -e ".[rag-embeddings]"
```

## Host as a web page

Recommended: Streamlit Community Cloud or HuggingFace Spaces (Streamlit).

Key setup notes:

- Entry file: `demo/streamlit_app.py`
- No API key is required to demonstrate the core transformation.
- If you want live AI before/after, set `SARVAM_API_KEY` as a secret/environment variable.

Dialect backends:

- By default the demo runs offline-first (heuristic dialect detection + rules normalization).
- If you enable Transformers/seq2seq backends, the demo will require local model paths unless you
  explicitly toggle "Allow remote model downloads".

RAG embeddings:

- The demo ships with a keyword-based retrieval mode (no extra deps).
- If you enable HF embeddings in the UI, provide a local model path (recommended) or toggle
  "Allow remote model downloads" (requires outbound network access on your hosting platform).
