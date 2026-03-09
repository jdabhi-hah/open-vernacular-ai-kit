# Benchmarks

This repo includes a tiny, dependency-light micro-benchmark:

```bash
python3 scripts/bench.py --mode run_many --n 50
python3 scripts/bench.py --mode run --n 50
python3 scripts/bench.py --mode render --n 50
```

Notes:

- This is meant for regression tracking, not absolute performance claims.
- Real-world throughput depends on optional backends (transliteration engines, transformers models, etc).

## North-Star Baseline Snapshot

For release tracking, generate the 3-metric baseline snapshot:

```bash
python3 scripts/snapshot_north_star_metrics.py --output docs/data/north_star_metrics_snapshot.json --iterations 200
```

Current snapshot (`2026-03-09T07:59:02Z`):

| Metric | Value | Notes |
| --- | --- | --- |
| `transliteration_success` | `1.000` | Golden transliteration accuracy across packaged Hindi/Gujarati cases (`90/90`; backend=`none`) |
| `dialect_accuracy` | `0.833` | Heuristic dialect-id accuracy (`5/6`) |
| `p95_latency_ms` | `0.205` | Pipeline p95 latency in ms (`iterations=200`, `n_calls=1200`) |

## Golden Transliteration Regression Guard

For offline regression tracking across the packaged Gujarati and Hindi language profiles, run:

```bash
gck eval --dataset golden_translit --language all --translit-mode sentence
```

This hand-validated suite is intended to catch regressions in common vernacular pronouns, question
words, support-style phrases, and inflected verb forms without requiring any hosted model access.

## Sentence-Level Language Regression Guard

For source-backed sentence regressions across the packaged Hindi and Gujarati language profiles, run:

```bash
gck eval --dataset language_sentences --language all --translit-mode sentence
```

This suite uses textbook/dialog-inspired examples plus Gujarati grammar-derived code-mix cases to
check that pronouns, possessives, case markers, adverbs, and common support phrases still render
correctly in full sentences.

The packaged dataset currently contains `118` exact-match sentence cases:

- `55` Hindi textbook/dialog-derived cases
- `63` Gujarati grammar and support-style cases

## Quality / Coverage (Gujarati Baseline Eval)

This project also includes a lightweight, reproducible "coverage-style" eval on public Gujarati
romanization data. It answers the question:

> After `codemix` rendering, how often does the output contain native Gujarati script (i.e., did we
> convert romanized Gujarati tokens into Gujarati)?

Run:

```bash
gck eval --dataset gujlish --report eval/out/report.json
```

Key fields in the JSON report:

- `pct_has_gujarati_codemix`: fraction of rows where output contains Gujarati script
- `pct_gu_roman_tokens_changed_est`: fraction of detected romanized Gujarati tokens that were transliterated

Important caveat:

- This is not a translation benchmark; it measures normalization / script conversion effects.

Example (from one local run with `topk=1`, `max_rows=2000`):

- Split `in22`: `pct_has_gujarati_codemix` ~= `0.987`
- Split `xnli`: `pct_has_gujarati_codemix` ~= `0.956`
