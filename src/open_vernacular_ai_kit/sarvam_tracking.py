from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .codemix_render import render_codemix
from .dialect_datasets import packaged_data_path
from .sarvam_promote import infer_profile_candidate_language, infer_sentence_case_language
from .sarvam_review import SarvamTeacherReviewedRecord, load_reviewed_records_jsonl


@dataclass(frozen=True)
class SarvamTrackedReviewFile:
    path: Path
    records: list[SarvamTeacherReviewedRecord]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _display_path(path: Path) -> str:
    root = _repo_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def default_reviewed_dataset_paths() -> list[Path]:
    root = _repo_root() / "eval" / "datasets"
    return sorted(root.glob("sarvam_teacher*_reviewed*.jsonl"))


def load_reviewed_dataset_files(paths: Iterable[str | Path] | None = None) -> list[SarvamTrackedReviewFile]:
    use_paths = [Path(p) for p in (paths or default_reviewed_dataset_paths())]
    out: list[SarvamTrackedReviewFile] = []
    for path in use_paths:
        if not path.exists():
            continue
        out.append(SarvamTrackedReviewFile(path=path, records=load_reviewed_records_jsonl(path)))
    return out


def _load_current_profiles() -> dict[str, dict[str, Any]]:
    import json

    return {
        "gu": json.loads(packaged_data_path("language_profiles/gu.json").read_text(encoding="utf-8")),
        "hi": json.loads(packaged_data_path("language_profiles/hi.json").read_text(encoding="utf-8")),
    }


def _sentence_case_fixed(reviewed: SarvamTeacherReviewedRecord) -> tuple[bool, str]:
    try:
        language = infer_sentence_case_language(reviewed)
        actual = render_codemix(
            reviewed.candidate.input,
            language=language,
            translit_mode="sentence",
        )
    except Exception as exc:
        return False, f"render_error: {exc}"
    return actual == reviewed.reviewed_expected, actual


def _profile_token_fixed(
    reviewed: SarvamTeacherReviewedRecord,
    *,
    profiles: dict[str, dict[str, Any]],
) -> tuple[int, int, list[dict[str, str]]]:
    approved = reviewed.approved_candidate_tokens or []
    fixed = 0
    checked = 0
    missing: list[dict[str, str]] = []
    for candidate in approved:
        checked += 1
        language = infer_profile_candidate_language(reviewed, native_text=candidate.native)
        profile = profiles[language]
        roman = candidate.roman.strip().lower()
        native = candidate.native.strip()
        exceptions = profile.get("default_exceptions", {})
        common_tokens = {str(x).strip().lower() for x in profile.get("common_roman_tokens", [])}
        context_tokens = {str(x).strip().lower() for x in profile.get("context_roman_tokens", [])}
        expected_bucket = (
            "context_roman_tokens"
            if reviewed.review_action == "accept_context_rule"
            else "common_roman_tokens"
        )
        bucket_ok = roman in (context_tokens if expected_bucket == "context_roman_tokens" else common_tokens)
        mapping_ok = str(exceptions.get(roman, "")).strip() == native
        if bucket_ok and mapping_ok:
            fixed += 1
            continue
        missing.append(
            {
                "language": language,
                "roman": roman,
                "native": native,
                "expected_bucket": expected_bucket,
            }
        )
    return checked, fixed, missing


def build_sarvam_failure_to_fix_report(
    *,
    reviewed_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    files = load_reviewed_dataset_files(reviewed_paths)
    profiles = _load_current_profiles()

    action_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    accepted_rows = 0
    fixed_rows = 0
    token_checks = 0
    token_fixed = 0
    by_action: dict[str, dict[str, Any]] = {}
    unfixed_examples: list[dict[str, Any]] = []

    def _bucket(action: str) -> dict[str, Any]:
        if action not in by_action:
            by_action[action] = {"n_rows": 0, "n_fixed_rows": 0, "n_token_checks": 0, "n_token_fixed": 0}
        return by_action[action]

    for file in files:
        for reviewed in file.records:
            action = reviewed.review_action
            action_counts[action] = action_counts.get(action, 0) + 1
            source = reviewed.candidate.source
            source_counts[source] = source_counts.get(source, 0) + 1
            bucket = _bucket(action)
            bucket["n_rows"] += 1

            if action not in {
                "accept_sentence_case",
                "accept_lexicon",
                "accept_context_rule",
                "accept_dialect_case",
            }:
                continue

            accepted_rows += 1
            row_fixed = False
            if action in {"accept_sentence_case", "accept_dialect_case"}:
                row_fixed, actual = _sentence_case_fixed(reviewed)
                if not row_fixed and len(unfixed_examples) < 20:
                    unfixed_examples.append(
                        {
                            "action": action,
                            "input": reviewed.candidate.input,
                            "expected": reviewed.reviewed_expected,
                            "actual": actual,
                            "source": source,
                            "review_file": _display_path(file.path),
                        }
                    )
            else:
                checked, fixed, missing = _profile_token_fixed(reviewed, profiles=profiles)
                token_checks += checked
                token_fixed += fixed
                bucket["n_token_checks"] += checked
                bucket["n_token_fixed"] += fixed
                row_fixed = checked > 0 and checked == fixed
                if not row_fixed and len(unfixed_examples) < 20:
                    unfixed_examples.append(
                        {
                            "action": action,
                            "input": reviewed.candidate.input,
                            "source": source,
                            "review_file": _display_path(file.path),
                            "missing_tokens": missing,
                        }
                    )

            if row_fixed:
                fixed_rows += 1
                bucket["n_fixed_rows"] += 1

    for action, stats in by_action.items():
        n_rows = int(stats["n_rows"])
        n_fixed_rows = int(stats["n_fixed_rows"])
        stats["fix_rate"] = (n_fixed_rows / n_rows) if n_rows else 0.0
        token_n = int(stats["n_token_checks"])
        token_fixed_n = int(stats["n_token_fixed"])
        stats["token_fix_rate"] = (token_fixed_n / token_n) if token_n else 0.0

    reviewed_total = sum(action_counts.values())
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_review_files": len(files),
        "review_files": [_display_path(f.path) for f in files],
        "n_reviewed_rows": reviewed_total,
        "review_action_counts": action_counts,
        "review_source_counts": source_counts,
        "accepted_summary": {
            "n_accepted_rows": accepted_rows,
            "n_fixed_rows": fixed_rows,
            "acceptance_rate_vs_reviewed": (accepted_rows / reviewed_total) if reviewed_total else 0.0,
            "fix_conversion_rate_vs_reviewed": (fixed_rows / reviewed_total) if reviewed_total else 0.0,
            "fix_rate_vs_accepted": (fixed_rows / accepted_rows) if accepted_rows else 0.0,
        },
        "profile_token_summary": {
            "n_token_checks": token_checks,
            "n_token_fixed": token_fixed,
            "token_fix_rate": (token_fixed / token_checks) if token_checks else 0.0,
        },
        "by_action": by_action,
        "unfixed_examples": unfixed_examples,
    }
