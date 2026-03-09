# North-Star Metrics

This project tracks three north-star metrics for release-to-release progress:

1. `transliteration_success`
2. `dialect_accuracy`
3. `p95_latency_ms`

These are intentionally lightweight and reproducible from local code.

## Definitions

- `transliteration_success`:
  - Accuracy from `run_eval(dataset="golden_translit", language="all")`.
  - Measures whether expected packaged Hindi/Gujarati romanized outputs are matched on the golden set.

- `dialect_accuracy`:
  - Accuracy from `run_eval(dataset="dialect_id", dialect_backend="heuristic")`.
  - Measures correct dialect label prediction on packaged dialect-id examples.

- `p95_latency_ms`:
  - 95th percentile single-input pipeline latency in milliseconds.
  - Computed over a fixed representative sample set with `CodeMixPipeline`.

## Reproducible Snapshot Command

```bash
python3 scripts/snapshot_north_star_metrics.py --output docs/data/north_star_metrics_snapshot.json --iterations 200
```

## Current Baseline Snapshot

Source file:

- `docs/data/north_star_metrics_snapshot.json`

| Metric | Value | Notes |
| --- | --- | --- |
| `transliteration_success` | `1.000` | Golden transliteration accuracy across packaged Hindi/Gujarati cases (`90/90`; backend=`none`) |
| `dialect_accuracy` | `0.833` | Heuristic dialect-id accuracy (`5/6`) |
| `p95_latency_ms` | `0.205` | Pipeline p95 latency in ms (`iterations=200`, `n_calls=1200`) |

Snapshot timestamp: `2026-03-09T07:59:02Z`.

Update this table whenever you refresh the snapshot for a release.
