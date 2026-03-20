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
from open_vernacular_ai_kit.sarvam_review import (  # noqa: E402
    dump_reviewed_records_jsonl,
    init_review_records_from_candidates,
)
from open_vernacular_ai_kit.sarvam_teacher import load_sarvam_teacher_records_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Initialize a reviewed JSONL scaffold from mined Sarvam candidate output."
    )
    ap.add_argument("--input", required=True, help="Mined candidate JSONL input.")
    ap.add_argument("--output", required=True, help="Reviewed JSONL output.")
    ap.add_argument(
        "--default-action",
        default="pending",
        help="Initial review action for all rows.",
    )
    args = ap.parse_args()

    try:
        rows = load_sarvam_teacher_records_jsonl(args.input)
        reviewed = init_review_records_from_candidates(rows, default_action=args.default_action)
        dump_reviewed_records_jsonl(args.output, reviewed, include_raw_response=False)
        print(
            json.dumps(
                {
                    "input_path": str(args.input),
                    "output_path": str(args.output),
                    "n_rows": len(reviewed),
                    "default_action": args.default_action,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except GckError as e:
        sys.stderr.write(f"init_sarvam_review: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
