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

from open_vernacular_ai_kit.errors import GckError  # noqa: E402
from open_vernacular_ai_kit.sarvam_report import build_sarvam_candidate_report  # noqa: E402
from open_vernacular_ai_kit.sarvam_teacher import load_sarvam_teacher_records_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a triage report from mined Sarvam candidate JSONL."
    )
    ap.add_argument("--input", required=True, help="Mined Sarvam candidate JSONL input.")
    ap.add_argument("--output", required=True, help="JSON report output path.")
    ap.add_argument(
        "--profile-dir",
        default=str(ROOT / "src" / "open_vernacular_ai_kit" / "_data" / "language_profiles"),
        help="Directory containing gu.json and hi.json.",
    )
    args = ap.parse_args()

    try:
        rows = load_sarvam_teacher_records_jsonl(args.input)
        report = build_sarvam_candidate_report(rows, profile_dir=args.profile_dir)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "input_path": str(args.input),
                    "output_path": str(args.output),
                    "n_rows": int(report["n_rows"]),
                    "n_novel_single_token_candidates": len(report["novel_single_token_candidates"]),
                    "n_mapping_conflict_candidates": len(report["mapping_conflict_candidates"]),
                    "n_phrase_candidates": len(report["phrase_candidates"]),
                    "n_english_keep_candidates": len(report["english_keep_candidates"]),
                    "n_ambiguous_candidates": len(report["ambiguous_candidates"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except GckError as e:
        sys.stderr.write(f"report_sarvam_candidates: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
