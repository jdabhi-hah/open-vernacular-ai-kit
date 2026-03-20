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
from open_vernacular_ai_kit.sarvam_seed import (  # noqa: E402
    build_failure_seed,
    dump_teacher_seed_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a Sarvam teacher seed JSONL from current eval failures."
    )
    ap.add_argument("--output", required=True, help="Output teacher seed JSONL path.")
    ap.add_argument(
        "--language",
        default="all",
        help="Target language slice: gu, hi, or all.",
    )
    ap.add_argument(
        "--translit-mode",
        default="sentence",
        help="Transliteration mode for replayed evals.",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Top-K transliteration candidates to use when replaying evals.",
    )
    ap.add_argument(
        "--skip-language-sentences",
        action="store_true",
        help="Do not include failures from the packaged language sentence eval.",
    )
    ap.add_argument(
        "--skip-golden-translit",
        action="store_true",
        help="Do not include failures from the packaged golden transliteration eval.",
    )
    ap.add_argument(
        "--include-backend-skips",
        action="store_true",
        help="Include backend-only golden transliteration cases that were skipped.",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path for seed summary counts.",
    )
    args = ap.parse_args()

    try:
        rows, summary = build_failure_seed(
            language=args.language,
            include_language_sentences=not args.skip_language_sentences,
            include_golden_translit=not args.skip_golden_translit,
            include_backend_skips=args.include_backend_skips,
            topk=args.topk,
            translit_mode=args.translit_mode,
        )
        dump_teacher_seed_jsonl(args.output, rows)
        payload = {
            "output_path": str(args.output),
            **summary,
        }
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except GckError as e:
        sys.stderr.write(f"build_sarvam_failure_seed: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
