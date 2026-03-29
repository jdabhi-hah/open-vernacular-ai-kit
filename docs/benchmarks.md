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

Current snapshot (`2026-03-29T17:30:50Z`):

| Metric | Value | Notes |
| --- | --- | --- |
| `transliteration_success` | `1.000` | Golden transliteration accuracy across packaged Hindi/Gujarati cases (`90/90`; backend=`none`) |
| `dialect_accuracy` | `1.000` | Heuristic dialect-id accuracy (`14/14`) |
| `p95_latency_ms` | `0.213` | Pipeline p95 latency in ms (`iterations=200`, `n_calls=1200`) |

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

The packaged dataset currently contains `120` exact-match sentence cases:

- `56` Hindi textbook/dialog-derived cases
- `64` Gujarati grammar and support-style cases

## Dialect Regression Guard

For packaged Gujarati dialect detection and normalization regressions, run:

```bash
gck eval --dataset dialect_id
gck eval --dataset dialect_normalization
```

Current packaged dialect eval coverage:

- `dialect_id`: `14` labeled examples across `kathiawadi`, `surati`, and `standard`
- `dialect_normalization`: `10` rule-backed examples across `kathiawadi` and `surati`

## Downstream Uplift Benchmarks

To measure whether normalization improves downstream behavior rather than just token-level quality, run:

```bash
gck eval --dataset retrieval_uplift --k 5
gck eval --dataset retrieval_uplift --retrieval-query-pack codemix --k 5
gck eval --dataset retrieval_uplift --retrieval-query-pack codemix_hard --k 5
gck eval --dataset prompt_stability_uplift --n-variants 10
gck eval --dataset answer_quality_uplift
gck eval --dataset answer_quality_uplift --answer-case-pack hard
gck eval --dataset answer_quality_uplift --answer-case-pack distractor
gck eval --dataset answer_quality_uplift --answer-case-pack abstention
gck eval --dataset answer_quality_uplift --answer-case-pack suite
```

These compare raw vs OVAK-normalized inputs and report absolute uplift deltas:

- `retrieval_uplift`: top-k retrieval recall for raw queries vs normalized queries
- `prompt_stability_uplift`: pairwise similarity for Sarvam outputs under raw prompts vs normalized prompts
- `answer_quality_uplift`: short-answer quality for Sarvam under raw questions vs normalized questions using packaged gold contexts

The packaged retrieval uplift benchmark now supports three query packs:

- `default`: English-first retrieval prompts
- `codemix`: code-mixed/romanized retrieval prompts closer to OVAK’s target use case
- `codemix_hard`: noisier and less label-heavy code-mixed prompts across all packaged docs

For retrieval uplift, query preprocessing is intentionally conservative:

- English-first queries are left unchanged
- code-mixed / vernacular-bearing queries are normalized

This avoids inflating negative uplift by transliterating English retrieval prompts that should remain
in Latin script.

To snapshot these packaged downstream uplift baselines into a committed JSON artifact:

```bash
python3 scripts/snapshot_downstream_uplift_metrics.py --output docs/data/downstream_uplift_snapshot.json
python3 scripts/snapshot_downstream_uplift_metrics.py \
  --output docs/data/downstream_uplift_snapshot.json \
  --include-answer-quality \
  --include-prompt-stability
```

Current downstream snapshot (`docs/data/downstream_uplift_snapshot.json`):

| Query Pack | Raw recall@1 | Normalized recall@1 | Raw recall@3 | Normalized recall@3 | Notes |
| --- | --- | --- | --- | --- | --- |
| `default` | `1.0` | `1.0` | `1.0` | `1.0` | English-first baseline; no expected uplift |
| `codemix` | `1.0` | `1.0` | `1.0` | `1.0` | Light code-mix pack; structurally useful but still easy |
| `codemix_hard` | `0.8` | `1.0` | `0.9` | `1.0` | First packaged retrieval pack with non-trivial uplift (`+0.2` @1, `+0.1` @3) |

Prompt-stability snapshot details from the same artifact:

- model: `sarvam-m`
- variants: `10`
- raw `mean_offdiag`: `0.8718`
- normalized `mean_offdiag`: `0.8907`
- uplift `mean_offdiag`: `+0.0189`
- raw `ref_min`: `0.8024`
- normalized `ref_min`: `0.8826`
- uplift `ref_min`: `+0.0802`

The prompt-stability snapshot requires:

- Sarvam API access
- eval dependencies installed
- cached prompt-stability generations or a live Sarvam run

The answer-quality uplift benchmark now supports five packaged answer modes:

- `default`: easier label-answer cases such as language names
- `hard`: phrase-answer cases that require reading the gold context more precisely
- `distractor`: multi-doc distractor cases that force answer selection from semantically similar contexts
- `abstention`: unsupported-fact cases where the correct answer is `UNKNOWN`
- `suite`: combined release-facing benchmark over `distractor` + `abstention`

Use them differently:

- `distractor`: better for answer selection under semantically similar contexts
- `abstention`: better for breaking exact-match saturation and measuring whether normalization reduces unsupported guesses
- `suite`: better when you want one stable release-facing downstream answer-quality number

The suite is now the preferred default for the committed downstream snapshot because it combines:

- distractor-based answer selection
- abstention on unsupported facts

The answer-quality uplift benchmark uses a packaged QA set with:

- gold context doc ids
- short expected English answers
- code-mixed questions that are normalized before prompt construction

This keeps the benchmark focused on downstream prompt conditioning instead of retrieval noise.

Current answer-quality snapshot details from the same artifact:

- model: `sarvam-m`
- answer case pack: see `snapshot_config.answer_quality.answer_case_pack` in the committed snapshot
- raw `exact_match_rate`: `0.9545`
- normalized `exact_match_rate`: `1.0`
- uplift `exact_match_rate`: `+0.0455`
- raw `mean_answer_similarity`: `0.3582`
- normalized `mean_answer_similarity`: `0.3685`
- uplift `mean_answer_similarity`: `+0.0103`
- the snapshot now also includes per-pack details under `downstream_uplift_metrics.answer_quality_uplift.per_pack`

Interpretation rule:

- if exact-match is saturated, prefer `distractor`, `abstention`, or `suite` over the old default pack
- if you need a single release-facing answer-quality number, prefer `suite`
- treat small similarity changes as directional only
- this benchmark is still less mature than retrieval uplift

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
