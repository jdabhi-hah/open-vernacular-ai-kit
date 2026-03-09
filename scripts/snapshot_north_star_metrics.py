from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_vernacular_ai_kit import CodeMixConfig, CodeMixPipeline  # noqa: E402
from open_vernacular_ai_kit.eval_harness import run_eval  # noqa: E402


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    # Nearest-rank definition.
    k = int(math.ceil(0.95 * len(xs))) - 1
    k = max(0, min(k, len(xs) - 1))
    return xs[k]


def _latency_snapshot(*, iterations: int) -> dict[str, object]:
    samples = [
        ("maru business plan ready chhe!!!", "gu"),
        ("hu aaje office jaish!!", "gu"),
        ("tame aaje ok chho?", "gu"),
        ("mujhe aap ki madad chahiye", "hi"),
        ("mera order aaj deliver hoga kya?", "hi"),
        ("meri maa ka naam kya hai?", "hi"),
    ]

    pipes = {
        language: CodeMixPipeline(config=CodeMixConfig(language=language, translit_mode="sentence"))
        for language in {lang for _, lang in samples}
    }

    # Warmup.
    for text, language in samples:
        _ = pipes[language].run(text).codemix

    latencies_ms: list[float] = []
    for _ in range(max(1, int(iterations))):
        for text, language in samples:
            pipe = pipes[language]
            t0 = time.perf_counter()
            _ = pipe.run(text).codemix
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    return {
        "sample_inputs": len(samples),
        "iterations": int(iterations),
        "n_calls": len(latencies_ms),
        "p50_ms": float(statistics.median(latencies_ms) if latencies_ms else 0.0),
        "p95_ms": float(_p95(latencies_ms)),
        "mean_ms": float(statistics.mean(latencies_ms) if latencies_ms else 0.0),
    }


def snapshot(*, iterations: int) -> dict[str, object]:
    golden = run_eval(dataset="golden_translit", language="all", topk=1, translit_mode="sentence")
    dialect = run_eval(dataset="dialect_id", dialect_backend="heuristic")
    latency = _latency_snapshot(iterations=iterations)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "metric_definitions": {
            "transliteration_success": (
                "Accuracy from run_eval(dataset='golden_translit', language='all'): expected "
                "Hindi/Gujarati romanized outputs matched on the packaged golden set."
            ),
            "dialect_accuracy": (
                "Accuracy from run_eval(dataset='dialect_id', dialect_backend='heuristic') "
                "on the packaged dialect-id set."
            ),
            "p95_latency_ms": (
                "95th percentile single-input pipeline latency (milliseconds) over a fixed "
                "representative sample set."
            ),
        },
        "north_star_metrics": {
            "transliteration_success": {
                "value": float(golden.get("accuracy", 0.0)),
                "dataset": str(golden.get("dataset", "golden_translit")),
                "language": str(golden.get("language", "all")),
                "n_cases": int(golden.get("n_cases", 0)),
                "n_ok": int(golden.get("n_ok", 0)),
                "language_slices": golden.get("language_slices", {}),
                "transliteration_backend": str(golden.get("transliteration_backend", "unknown")),
            },
            "dialect_accuracy": {
                "value": float(dialect.get("accuracy", 0.0)),
                "dataset": str(dialect.get("dataset", "dialect_id")),
                "n_rows": int(dialect.get("n_rows", 0)),
                "backend": str(dialect.get("dialect_backend", "heuristic")),
            },
            "p95_latency_ms": {
                "value": float(latency.get("p95_ms", 0.0)),
                "p50_ms": float(latency.get("p50_ms", 0.0)),
                "mean_ms": float(latency.get("mean_ms", 0.0)),
                "n_calls": int(latency.get("n_calls", 0)),
                "iterations": int(latency.get("iterations", 0)),
                "sample_inputs": int(latency.get("sample_inputs", 0)),
                "config": {"translit_mode": "sentence"},
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot north-star baseline metrics for this release.")
    ap.add_argument(
        "--output",
        default="docs/data/north_star_metrics_snapshot.json",
        help="Path to JSON output file.",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of iterations over the sample set for latency measurement.",
    )
    args = ap.parse_args()

    out = snapshot(iterations=max(1, int(args.iterations)))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote snapshot: {out_path}")
    print(json.dumps(out["north_star_metrics"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
