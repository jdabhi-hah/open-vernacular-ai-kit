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

from open_vernacular_ai_kit.sarvam_tracking import build_sarvam_failure_to_fix_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Sarvam review-to-fix conversion report.")
    ap.add_argument(
        "--output",
        default="docs/data/sarvam_failure_to_fix_snapshot.json",
        help="Path to JSON output file.",
    )
    ap.add_argument(
        "--reviewed",
        nargs="*",
        default=None,
        help="Optional explicit reviewed JSONL paths. Defaults to all committed reviewed datasets.",
    )
    args = ap.parse_args()

    payload = build_sarvam_failure_to_fix_report(reviewed_paths=args.reviewed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")
    print(json.dumps(payload["accepted_summary"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
