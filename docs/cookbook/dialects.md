# Dialects

Dialect utilities are offline-first and pluggable.

Heuristic dialect normalization (rules) gated by confidence:

```python
from open_vernacular_ai_kit import analyze_codemix

a = analyze_codemix(
    "kamaad thaalu rakhje",
    dialect_backend="heuristic",
    dialect_normalize=True,
    dialect_min_confidence=0.7,
)
print(a.codemix)
```

Packaged regression coverage now includes:

- `dialect_id`: `14` labeled examples across `kathiawadi`, `surati`, and `standard`
- `dialect_normalization`: `10` rule-backed examples across `kathiawadi` and `surati`

Run the packaged evals with:

```bash
gck eval --dataset dialect_id
gck eval --dataset dialect_normalization
```

Transformers backends require optional extras and usually a local model path:

```bash
pip install -e ".[dialect-ml]"
```
