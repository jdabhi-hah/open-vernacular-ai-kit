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
from open_vernacular_ai_kit.sarvam_promote import (  # noqa: E402
    dump_language_profile_data,
    load_language_profile_data,
    promote_profile_candidates_from_review,
)
from open_vernacular_ai_kit.sarvam_review import load_reviewed_records_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Promote reviewed Sarvam lexicon/context candidates into packaged language profiles."
    )
    ap.add_argument("--input", required=True, help="Reviewed JSONL input.")
    ap.add_argument(
        "--profile-dir",
        default=str(ROOT / "src" / "open_vernacular_ai_kit" / "_data" / "language_profiles"),
        help="Existing profile directory containing gu.json and hi.json.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for promoted profile JSON files. Defaults to overwriting --profile-dir.",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path with add/skip/conflict stats.",
    )
    ap.add_argument(
        "--allow-conflicts",
        action="store_true",
        help="Allow writing output even when reviewed rows conflict with existing mappings or token buckets.",
    )
    args = ap.parse_args()

    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir) if args.output_dir else profile_dir

    try:
        reviewed = load_reviewed_records_jsonl(args.input)
        profiles = {
            language: load_language_profile_data(profile_dir / f"{language}.json")
            for language in ("gu", "hi")
        }
        merged, report = promote_profile_candidates_from_review(
            reviewed,
            existing_profiles=profiles,
        )
        if (report["n_mapping_conflicts"] or report["n_bucket_conflicts"]) and not args.allow_conflicts:
            if args.report:
                report_path = Path(args.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "input_path": str(args.input),
                        "profile_dir": str(profile_dir),
                        "output_dir": str(output_dir),
                        **report,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(3)

        output_dir.mkdir(parents=True, exist_ok=True)
        for language, profile in merged.items():
            dump_language_profile_data(output_dir / f"{language}.json", profile)
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "input_path": str(args.input),
                    "profile_dir": str(profile_dir),
                    "output_dir": str(output_dir),
                    **report,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except GckError as e:
        sys.stderr.write(f"promote_sarvam_profile_candidates: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
