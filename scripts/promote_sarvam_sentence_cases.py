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

from open_vernacular_ai_kit.dialect_datasets import packaged_data_path  # noqa: E402
from open_vernacular_ai_kit.errors import GckError  # noqa: E402
from open_vernacular_ai_kit.sarvam_promote import (  # noqa: E402
    dump_language_sentence_case_records,
    load_language_sentence_case_records,
    promote_sentence_cases_from_review,
)
from open_vernacular_ai_kit.sarvam_review import load_reviewed_records_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Promote reviewed Sarvam sentence cases into the packaged language sentence dataset."
    )
    ap.add_argument("--input", required=True, help="Reviewed JSONL input.")
    ap.add_argument(
        "--dataset",
        default=str(packaged_data_path("language_sentence_cases.jsonl")),
        help="Existing language sentence dataset path.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output dataset path. Defaults to overwriting --dataset.",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path with add/skip/conflict stats.",
    )
    ap.add_argument(
        "--allow-conflicts",
        action="store_true",
        help="Allow writing output even when reviewed rows conflict with existing dataset entries.",
    )
    ap.add_argument(
        "--allow-failing-cases",
        action="store_true",
        help="Allow promotion of accepted sentence cases even if current runtime output does not match reviewed expected output.",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else dataset_path

    try:
        reviewed = load_reviewed_records_jsonl(args.input)
        existing = load_language_sentence_case_records(dataset_path)
        merged, report = promote_sentence_cases_from_review(
            reviewed,
            existing_rows=existing,
            require_pass=not args.allow_failing_cases,
        )
        if report["n_duplicates_conflict"] and not args.allow_conflicts:
            if args.report:
                report_path = Path(args.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "input_path": str(args.input),
                        "dataset_path": str(dataset_path),
                        "output_path": str(output_path),
                        **report,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(3)

        dump_language_sentence_case_records(output_path, merged)
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "input_path": str(args.input),
                    "dataset_path": str(dataset_path),
                    "output_path": str(output_path),
                    **report,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except GckError as e:
        sys.stderr.write(f"promote_sarvam_sentence_cases: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
