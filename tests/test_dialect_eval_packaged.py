from __future__ import annotations

from open_vernacular_ai_kit.dialect_datasets import (
    load_dialect_id_jsonl,
    load_dialect_normalization_jsonl,
    packaged_data_path,
)
from open_vernacular_ai_kit.eval_harness import (
    run_dialect_id_eval,
    run_dialect_normalization_eval,
)


def test_packaged_dialect_id_dataset_is_expanded_and_balanced() -> None:
    rows = load_dialect_id_jsonl(packaged_data_path("dialect_id_samples.jsonl"))
    assert len(rows) == 14
    assert {row.dialect.value for row in rows} == {"kathiawadi", "surati", "standard"}


def test_packaged_dialect_normalization_dataset_is_expanded() -> None:
    rows = load_dialect_normalization_jsonl(packaged_data_path("dialect_norm_samples.jsonl"))
    assert len(rows) == 10
    assert {row.dialect.value for row in rows} == {"kathiawadi", "surati"}


def test_packaged_dialect_id_eval_is_perfect_for_supported_heuristics() -> None:
    result = run_dialect_id_eval(dialect_backend="heuristic")
    assert result["n_rows"] == 14
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_packaged_dialect_normalization_eval_is_perfect_for_supported_rules() -> None:
    result = run_dialect_normalization_eval(dialect_normalizer_backend="heuristic")
    assert result["n_rows"] == 10
    assert result["exact_match"] == 1.0
    assert result["chrf"] == 100.0
