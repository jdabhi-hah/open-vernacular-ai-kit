# Before vs After LLM Output Notebook

A ready-to-run notebook is shipped at:

- `notebooks/before_after_llm_output.ipynb`

GitHub view:

- <https://github.com/SudhirGadhvi/open-vernacular-ai-kit/blob/develop/notebooks/before_after_llm_output.ipynb>

## What it measures

- Same labeled inputs (`docs/data/llm_uplift_examples.jsonl`)
- LLM intent output **before** preprocessing
- LLM intent output **after** `render_codemix(...)`
- Accuracy uplift: `after_accuracy - before_accuracy`

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[demo,api,indic]"
pip install jupyter openai
jupyter notebook notebooks/before_after_llm_output.ipynb
```

## Baseline snapshot (mock adapter)

From a local run using the included dataset and offline mock adapter:

- `before_accuracy`: `0.10`
- `after_accuracy`: `1.00`
- `uplift`: `+0.90`

You can switch to real OpenAI calls in the notebook by setting:

- `USE_OPENAI = True`
- `OPENAI_API_KEY`
