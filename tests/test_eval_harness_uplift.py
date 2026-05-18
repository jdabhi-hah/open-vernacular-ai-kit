from __future__ import annotations

import pytest

from open_vernacular_ai_kit import eval_harness


def test_run_retrieval_uplift_eval_compares_raw_vs_normalized(monkeypatch) -> None:
    calls: list[bool] = []

    def _fake_run_retrieval_eval(*, k_values, embedding_model, preprocess_query, retrieval_query_pack):
        calls.append(bool(preprocess_query))
        recall = {"1": 0.5, "3": 0.75, "5": 1.0}
        if preprocess_query:
            recall = {"1": 0.75, "3": 1.0, "5": 1.0}
        return {
            "dataset": "retrieval",
            "embedding_model_used": "test-model",
            "k_values": list(k_values),
            "retrieval_query_pack": retrieval_query_pack,
            "recall_at_k": recall,
        }

    monkeypatch.setattr(eval_harness, "run_retrieval_eval", _fake_run_retrieval_eval)

    res = eval_harness.run_retrieval_uplift_eval(k_values=(1, 3, 5), embedding_model="test-model")

    assert res["dataset"] == "retrieval_uplift"
    assert calls == [False, True]
    assert res["recall_uplift"]["1"]["raw"] == 0.5
    assert res["recall_uplift"]["1"]["normalized"] == 0.75
    assert res["recall_uplift"]["1"]["absolute_uplift"] == 0.25
    assert res["recall_uplift"]["3"]["absolute_uplift"] == 0.25
    assert res["recall_uplift"]["5"]["absolute_uplift"] == 0.0
    assert res["retrieval_query_pack"] == "default"


def test_run_prompt_stability_uplift_eval_compares_raw_vs_normalized(monkeypatch) -> None:
    calls: list[bool] = []

    def _fake_run_prompt_stability_eval(
        *,
        model,
        n_variants,
        base_question_gu,
        embedding_model,
        cache_dir,
        api_key,
        preprocess,
    ):
        calls.append(bool(preprocess))
        pairwise_similarity = {
            "mean_offdiag": 0.61,
            "min_offdiag": 0.4,
            "max_offdiag": 0.9,
            "ref_mean": 0.63,
            "ref_min": 0.45,
        }
        if preprocess:
            pairwise_similarity = {
                "mean_offdiag": 0.82,
                "min_offdiag": 0.67,
                "max_offdiag": 0.94,
                "ref_mean": 0.85,
                "ref_min": 0.7,
            }
        return {
            "dataset": "prompt_stability",
            "model": model,
            "embedding_model_used": embedding_model,
            "pairwise_similarity": pairwise_similarity,
        }

    monkeypatch.setattr(eval_harness, "run_prompt_stability_eval", _fake_run_prompt_stability_eval)

    res = eval_harness.run_prompt_stability_uplift_eval(
        model="sarvam-m",
        n_variants=8,
        embedding_model="test-model",
    )

    assert res["dataset"] == "prompt_stability_uplift"
    assert calls == [False, True]
    assert res["pairwise_similarity_uplift"]["mean_offdiag"]["raw"] == 0.61
    assert res["pairwise_similarity_uplift"]["mean_offdiag"]["normalized"] == 0.82
    assert res["pairwise_similarity_uplift"]["mean_offdiag"]["absolute_uplift"] == pytest.approx(0.21)
    assert res["pairwise_similarity_uplift"]["ref_min"]["absolute_uplift"] == pytest.approx(0.25)


def test_run_eval_dispatches_retrieval_uplift(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_harness,
        "run_retrieval_uplift_eval",
        lambda **kwargs: {"dataset": "retrieval_uplift", "kwargs": kwargs},
    )

    res = eval_harness.run_eval(
        dataset="retrieval_uplift",
        k=7,
        embedding_model="test-model",
        retrieval_query_pack="codemix",
    )

    assert res["dataset"] == "retrieval_uplift"
    assert res["kwargs"]["k_values"] == (1, 3, 7)
    assert res["kwargs"]["retrieval_query_pack"] == "codemix"


def test_run_eval_dispatches_prompt_stability_uplift(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_harness,
        "run_prompt_stability_uplift_eval",
        lambda **kwargs: {"dataset": "prompt_stability_uplift", "kwargs": kwargs},
    )

    res = eval_harness.run_eval(
        dataset="prompt_stability_uplift",
        sarvam_model="sarvam-m",
        n_variants=12,
        embedding_model="test-model",
        api_key="x",
    )

    assert res["dataset"] == "prompt_stability_uplift"
    assert res["kwargs"]["model"] == "sarvam-m"
    assert res["kwargs"]["n_variants"] == 12


def test_run_answer_quality_uplift_eval_compares_raw_vs_normalized(monkeypatch) -> None:
    calls: list[bool] = []

    def _fake_run_answer_quality_eval(
        *,
        model,
        embedding_model,
        cache_dir,
        api_key,
        preprocess_question,
        answer_case_pack,
    ):
        calls.append(bool(preprocess_question))
        metrics = {
            "exact_match_rate": 0.5,
            "mean_answer_similarity": 0.7,
            "min_answer_similarity": 0.55,
        }
        if preprocess_question:
            metrics = {
                "exact_match_rate": 0.75,
                "mean_answer_similarity": 0.88,
                "min_answer_similarity": 0.71,
            }
        return {
            "dataset": "answer_quality",
            "model": model,
            "answer_case_pack": answer_case_pack,
            "embedding_model_used": embedding_model,
            "metrics": metrics,
        }

    monkeypatch.setattr(eval_harness, "run_answer_quality_eval", _fake_run_answer_quality_eval)

    res = eval_harness.run_answer_quality_uplift_eval(
        model="sarvam-m",
        embedding_model="test-model",
        answer_case_pack="hard",
    )

    assert res["dataset"] == "answer_quality_uplift"
    assert res["answer_case_pack"] == "hard"
    assert calls == [False, True]
    assert res["answer_quality_uplift"]["exact_match_rate"]["absolute_uplift"] == pytest.approx(0.25)
    assert res["answer_quality_uplift"]["mean_answer_similarity"]["absolute_uplift"] == pytest.approx(0.18)


def test_run_answer_quality_suite_uplift_eval_aggregates_case_packs(monkeypatch) -> None:
    def _fake_run_answer_quality_uplift_eval(*, model, embedding_model, cache_dir, api_key, answer_case_pack):
        if answer_case_pack == "distractor":
            return {
                "dataset": "answer_quality_uplift",
                "model": model,
                "answer_case_pack": "distractor",
                "embedding_model_used": embedding_model,
                "raw_eval": {
                    "n_cases": 4,
                    "used_cache_n": 4,
                    "metrics": {
                        "exact_match_rate": 1.0,
                        "mean_answer_similarity": 0.6,
                        "min_answer_similarity": 0.2,
                    },
                },
                "normalized_eval": {
                    "n_cases": 4,
                    "used_cache_n": 4,
                    "metrics": {
                        "exact_match_rate": 1.0,
                        "mean_answer_similarity": 0.7,
                        "min_answer_similarity": 0.3,
                    },
                },
                "answer_quality_uplift": {
                    "exact_match_rate": {"raw": 1.0, "normalized": 1.0, "absolute_uplift": 0.0},
                    "mean_answer_similarity": {"raw": 0.6, "normalized": 0.7, "absolute_uplift": 0.1},
                    "min_answer_similarity": {"raw": 0.2, "normalized": 0.3, "absolute_uplift": 0.1},
                },
            }
        if answer_case_pack == "abstention":
            return {
                "dataset": "answer_quality_uplift",
                "model": model,
                "answer_case_pack": "abstention",
                "embedding_model_used": embedding_model,
                "raw_eval": {
                    "n_cases": 6,
                    "used_cache_n": 6,
                    "metrics": {
                        "exact_match_rate": 0.5,
                        "mean_answer_similarity": 0.2,
                        "min_answer_similarity": 0.1,
                    },
                },
                "normalized_eval": {
                    "n_cases": 6,
                    "used_cache_n": 6,
                    "metrics": {
                        "exact_match_rate": 1.0,
                        "mean_answer_similarity": 0.4,
                        "min_answer_similarity": 0.15,
                    },
                },
                "answer_quality_uplift": {
                    "exact_match_rate": {"raw": 0.5, "normalized": 1.0, "absolute_uplift": 0.5},
                    "mean_answer_similarity": {"raw": 0.2, "normalized": 0.4, "absolute_uplift": 0.2},
                    "min_answer_similarity": {"raw": 0.1, "normalized": 0.15, "absolute_uplift": 0.05},
                },
            }
        raise AssertionError(f"unexpected case pack: {answer_case_pack}")

    monkeypatch.setattr(eval_harness, "run_answer_quality_uplift_eval", _fake_run_answer_quality_uplift_eval)

    res = eval_harness.run_answer_quality_suite_uplift_eval(
        model="sarvam-m",
        embedding_model="test-model",
    )

    assert res["answer_case_pack"] == "suite"
    assert res["case_packs"] == ["distractor", "abstention"]
    assert res["raw_eval"]["n_cases"] == 10
    assert res["normalized_eval"]["used_cache_n"] == 10
    assert res["raw_eval"]["metrics"]["exact_match_rate"] == pytest.approx(0.7)
    assert res["normalized_eval"]["metrics"]["exact_match_rate"] == pytest.approx(1.0)
    assert res["answer_quality_uplift"]["exact_match_rate"]["absolute_uplift"] == pytest.approx(0.3)
    assert res["raw_eval"]["metrics"]["min_answer_similarity"] == pytest.approx(0.1)
    assert res["normalized_eval"]["metrics"]["min_answer_similarity"] == pytest.approx(0.15)


def test_answer_matches_expected_allows_short_label_inside_longer_output() -> None:
    assert eval_harness._answer_matches_expected("Gujarati", "The answer is Gujarati language.")
    assert not eval_harness._answer_matches_expected("Gujarati", "Hindi")


def test_run_eval_dispatches_answer_quality_uplift(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_harness,
        "run_answer_quality_uplift_eval",
        lambda **kwargs: {"dataset": "answer_quality_uplift", "kwargs": kwargs},
    )

    res = eval_harness.run_eval(
        dataset="answer_quality_uplift",
        sarvam_model="sarvam-m",
        embedding_model="test-model",
        answer_case_pack="hard",
        api_key="x",
    )

    assert res["dataset"] == "answer_quality_uplift"
    assert res["kwargs"]["model"] == "sarvam-m"
    assert res["kwargs"]["answer_case_pack"] == "hard"


def test_preprocess_retrieval_query_skips_english_first_queries(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_harness,
        "analyze_codemix",
        lambda text: type(
            "A",
            (),
            {
                "codemix": "વ્હિચ language ...",
                "n_gu_native_tokens": 0,
                "n_gu_roman_tokens": 1,
                "n_en_tokens": 10,
                "n_tokens": 12,
            },
        )(),
    )

    assert eval_harness._preprocess_retrieval_query("Which language ...") == "Which language ..."


def test_preprocess_retrieval_query_keeps_code_mixed_queries(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_harness,
        "analyze_codemix",
        lambda text: type(
            "A",
            (),
            {
                "codemix": "મારું order status શું છે",
                "n_gu_native_tokens": 0,
                "n_gu_roman_tokens": 4,
                "n_en_tokens": 2,
                "n_tokens": 6,
            },
        )(),
    )

    assert (
        eval_harness._preprocess_retrieval_query("maru order status shu chhe")
        == "મારું order status શું છે"
    )


def test_preprocess_retrieval_query_preserves_language_and_state_labels() -> None:
    out = eval_harness._preprocess_retrieval_query(
        "maharashtra civic services ma konsi language broadly use thay chhe Marathi?"
    )

    assert out.startswith("maharashtra civic services")
    assert out.endswith("Marathi?")
    assert "માં" in out
