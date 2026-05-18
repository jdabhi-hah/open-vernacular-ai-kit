from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .eval_harness import (
    _DEFAULT_EMBEDDING_MODEL,
    run_answer_quality_uplift_eval,
    run_prompt_stability_uplift_eval,
    run_retrieval_uplift_eval,
)


def _compact_retrieval_uplift(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_query_pack": str(result["retrieval_query_pack"]),
        "embedding_model_requested": str(result["embedding_model_requested"]),
        "embedding_model_used": str(result["embedding_model_used"]),
        "k_values": list(result["k_values"]),
        "n_queries": int(result["raw_eval"]["n_queries"]),
        "raw_recall_at_k": dict(result["raw_eval"]["recall_at_k"]),
        "normalized_recall_at_k": dict(result["normalized_eval"]["recall_at_k"]),
        "recall_uplift": dict(result["recall_uplift"]),
    }


def _compact_prompt_stability_uplift(result: dict[str, Any]) -> dict[str, Any]:
    raw_eval = result["raw_eval"]
    normalized_eval = result["normalized_eval"]
    return {
        "model": str(result["model"]),
        "n_variants": int(result["n_variants"]),
        "embedding_model_requested": str(result["embedding_model_requested"]),
        "embedding_model_used": str(result["embedding_model_used"]),
        "raw_pairwise_similarity": dict(raw_eval["pairwise_similarity"]),
        "normalized_pairwise_similarity": dict(normalized_eval["pairwise_similarity"]),
        "pairwise_similarity_uplift": dict(result["pairwise_similarity_uplift"]),
        "raw_used_cache_n": int(raw_eval["used_cache_n"]),
        "normalized_used_cache_n": int(normalized_eval["used_cache_n"]),
    }


def _compact_answer_quality_uplift(result: dict[str, Any]) -> dict[str, Any]:
    raw_eval = result["raw_eval"]
    normalized_eval = result["normalized_eval"]
    payload = {
        "model": str(result["model"]),
        "answer_case_pack": str(result["answer_case_pack"]),
        "embedding_model_requested": str(result["embedding_model_requested"]),
        "embedding_model_used": str(result["embedding_model_used"]),
        "raw_metrics": dict(raw_eval["metrics"]),
        "normalized_metrics": dict(normalized_eval["metrics"]),
        "answer_quality_uplift": dict(result["answer_quality_uplift"]),
        "raw_used_cache_n": int(raw_eval["used_cache_n"]),
        "normalized_used_cache_n": int(normalized_eval["used_cache_n"]),
    }
    case_packs = result.get("case_packs")
    case_pack_results = result.get("case_pack_results")
    if isinstance(case_packs, list) and isinstance(case_pack_results, dict):
        payload["case_packs"] = list(case_packs)
        payload["per_pack"] = {
            str(pack): {
                "raw_metrics": dict(case_pack_results[pack]["raw_eval"]["metrics"]),
                "normalized_metrics": dict(case_pack_results[pack]["normalized_eval"]["metrics"]),
                "answer_quality_uplift": dict(case_pack_results[pack]["answer_quality_uplift"]),
            }
            for pack in case_packs
            if pack in case_pack_results
        }
    return payload


def snapshot_downstream_uplift(
    *,
    retrieval_query_packs: Sequence[str] = ("default", "codemix", "codemix_hard"),
    k_values: Sequence[int] = (1, 3, 5),
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    include_answer_quality: bool = False,
    answer_model: str = "sarvam-m",
    answer_case_pack: str = "suite",
    answer_cache_dir: Optional[Path] = None,
    include_prompt_stability: bool = False,
    prompt_model: str = "sarvam-m",
    prompt_n_variants: int = 10,
    prompt_base_question_gu: str = "અમદાવાદમાં શિયાળામાં કઈ ખાસ વાનગી લોકપ્રિય છે?",
    prompt_cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    packs = [str(x).strip() for x in retrieval_query_packs if str(x).strip()]
    if not packs:
        raise ValueError("retrieval_query_packs must contain at least one pack")

    retrieval_snapshots: dict[str, Any] = {}
    for pack in packs:
        result = run_retrieval_uplift_eval(
            k_values=tuple(int(k) for k in k_values),
            embedding_model=embedding_model,
            retrieval_query_pack=pack,
        )
        retrieval_snapshots[pack] = _compact_retrieval_uplift(result)

    downstream_metrics: dict[str, Any] = {
        "retrieval_uplift": retrieval_snapshots,
    }
    metric_definitions = {
        "retrieval_uplift": (
            "Top-k retrieval recall delta from run_retrieval_uplift_eval: compares raw "
            "queries vs OVAK-normalized queries on packaged retrieval query packs."
        ),
    }

    snapshot_config: dict[str, Any] = {
        "retrieval_query_packs": packs,
        "k_values": [int(k) for k in k_values],
        "embedding_model_requested": embedding_model,
    }

    if include_answer_quality:
        answer_result = run_answer_quality_uplift_eval(
            model=answer_model,
            answer_case_pack=answer_case_pack,
            embedding_model=embedding_model,
            cache_dir=answer_cache_dir,
            api_key=api_key,
        )
        downstream_metrics["answer_quality_uplift"] = _compact_answer_quality_uplift(answer_result)
        metric_definitions["answer_quality_uplift"] = (
            "Short-answer quality delta from run_answer_quality_uplift_eval: compares Sarvam "
            "answers under raw questions vs OVAK-normalized questions using packaged gold contexts."
        )
        snapshot_config["answer_quality"] = {
            "included": True,
            "model": answer_model,
            "answer_case_pack": answer_case_pack,
            "cache_dir": str(answer_cache_dir) if answer_cache_dir else None,
        }
    else:
        snapshot_config["answer_quality"] = {"included": False}

    if include_prompt_stability:
        prompt_result = run_prompt_stability_uplift_eval(
            model=prompt_model,
            n_variants=int(prompt_n_variants),
            base_question_gu=prompt_base_question_gu,
            embedding_model=embedding_model,
            cache_dir=prompt_cache_dir,
            api_key=api_key,
        )
        downstream_metrics["prompt_stability_uplift"] = _compact_prompt_stability_uplift(prompt_result)
        metric_definitions["prompt_stability_uplift"] = (
            "Pairwise semantic similarity delta from run_prompt_stability_uplift_eval: compares "
            "raw prompts vs OVAK-normalized prompts for the configured Sarvam model."
        )
        snapshot_config["prompt_stability"] = {
            "included": True,
            "model": prompt_model,
            "n_variants": int(prompt_n_variants),
            "base_question_gu": prompt_base_question_gu,
            "cache_dir": str(prompt_cache_dir) if prompt_cache_dir else None,
        }
    else:
        snapshot_config["prompt_stability"] = {"included": False}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "metric_definitions": metric_definitions,
        "snapshot_config": snapshot_config,
        "downstream_uplift_metrics": downstream_metrics,
    }
