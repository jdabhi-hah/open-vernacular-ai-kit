from __future__ import annotations

from open_vernacular_ai_kit.eval_harness import (
    run_eval,
    run_golden_translit_eval,
    run_language_sentence_eval,
)


def test_golden_translit_eval_supports_language_slices() -> None:
    res = run_golden_translit_eval(language="all", translit_mode="sentence")
    assert res["dataset"] == "golden_translit"
    assert res["language"] == "all"
    assert "gu" in res["language_slices"]
    assert "hi" in res["language_slices"]
    assert int(res["language_slices"]["gu"]["n_cases"]) > 0
    assert int(res["language_slices"]["hi"]["n_cases"]) > 0


def test_golden_eval_falls_back_to_default_for_unknown_language() -> None:
    res = run_eval(dataset="golden_translit", language="unknown-lang")
    assert res["dataset"] == "golden_translit"
    assert res["language"] == "gu"
    assert res["language_requested"] == "unknown-lang"


def test_language_sentence_eval_supports_language_slices() -> None:
    res = run_language_sentence_eval(language="all", translit_mode="sentence")
    assert res["dataset"] == "language_sentences"
    assert res["language"] == "all"
    assert int(res["language_slices"]["gu"]["n_cases"]) > 0
    assert int(res["language_slices"]["hi"]["n_cases"]) > 0
    assert int(res["n_cases"]) >= 100
    assert int(res["language_slices"]["gu"]["n_cases"]) >= 50
    assert int(res["language_slices"]["hi"]["n_cases"]) >= 50


def test_run_eval_dispatches_language_sentence_dataset() -> None:
    res = run_eval(dataset="language_sentences", language="hi", translit_mode="sentence")
    assert res["dataset"] == "language_sentences"
    assert res["language"] == "hi"
    assert int(res["n_cases"]) > 0
