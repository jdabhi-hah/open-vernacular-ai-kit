from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_vernacular_ai_kit.downstream_snapshots import snapshot_downstream_uplift  # noqa: E402


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot downstream uplift benchmark metrics.")
    ap.add_argument(
        "--output",
        default="docs/data/downstream_uplift_snapshot.json",
        help="Path to JSON output file.",
    )
    ap.add_argument(
        "--retrieval-query-packs",
        default="default,codemix,codemix_hard",
        help="Comma-separated retrieval query packs to snapshot.",
    )
    ap.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated top-k values for retrieval uplift.",
    )
    ap.add_argument(
        "--embedding-model",
        default="ai4bharat/indic-bert",
        help="Embedding model requested for retrieval uplift.",
    )
    ap.add_argument(
        "--include-answer-quality",
        action="store_true",
        help="Also snapshot answer-quality uplift (requires Sarvam + eval dependencies).",
    )
    ap.add_argument(
        "--answer-model",
        default="sarvam-m",
        help="Sarvam model for answer-quality uplift.",
    )
    ap.add_argument(
        "--answer-case-pack",
        default="suite",
        help="Answer-quality case pack: default, hard, distractor, abstention, or suite.",
    )
    ap.add_argument(
        "--answer-cache-dir",
        default="",
        help="Optional cache directory override for answer-quality uplift.",
    )
    ap.add_argument(
        "--include-prompt-stability",
        action="store_true",
        help="Also snapshot prompt-stability uplift (requires Sarvam + eval dependencies).",
    )
    ap.add_argument(
        "--prompt-model",
        default="sarvam-m",
        help="Sarvam model for prompt-stability uplift.",
    )
    ap.add_argument(
        "--prompt-n-variants",
        type=int,
        default=10,
        help="Number of prompt variants for prompt-stability uplift.",
    )
    ap.add_argument(
        "--prompt-base-question-gu",
        default="અમદાવાદમાં શિયાળામાં કઈ ખાસ વાનગી લોકપ્રિય છે?",
        help="Gujarati base question for prompt-stability uplift.",
    )
    ap.add_argument(
        "--prompt-cache-dir",
        default="",
        help="Optional cache directory override for prompt-stability uplift.",
    )
    ap.add_argument(
        "--api-key",
        default="",
        help="Optional Sarvam API key override for prompt-stability uplift.",
    )
    args = ap.parse_args()

    payload = snapshot_downstream_uplift(
        retrieval_query_packs=_parse_csv(args.retrieval_query_packs),
        k_values=_parse_int_csv(args.k_values),
        embedding_model=str(args.embedding_model),
        include_answer_quality=bool(args.include_answer_quality),
        answer_model=str(args.answer_model),
        answer_case_pack=str(args.answer_case_pack),
        answer_cache_dir=Path(args.answer_cache_dir) if str(args.answer_cache_dir).strip() else None,
        include_prompt_stability=bool(args.include_prompt_stability),
        prompt_model=str(args.prompt_model),
        prompt_n_variants=int(args.prompt_n_variants),
        prompt_base_question_gu=str(args.prompt_base_question_gu),
        prompt_cache_dir=Path(args.prompt_cache_dir) if str(args.prompt_cache_dir).strip() else None,
        api_key=str(args.api_key).strip() or None,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote snapshot: {out_path}")
    print(json.dumps(payload["downstream_uplift_metrics"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
