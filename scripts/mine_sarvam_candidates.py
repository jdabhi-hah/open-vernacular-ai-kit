from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_vernacular_ai_kit.errors import GckError  # noqa: E402
from open_vernacular_ai_kit.sarvam_teacher import (  # noqa: E402
    dump_sarvam_teacher_records_jsonl,
    load_sarvam_teacher_inputs_jsonl,
    mine_sarvam_teacher_candidate,
)


def _default_error_output_path(output_path: str | Path) -> Path:
    out = Path(output_path)
    if out.suffix:
        return out.with_name(f"{out.stem}.errors{out.suffix}")
    return out.with_name(f"{out.name}.errors.jsonl")


def _dump_error_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class _RowTimeoutError(RuntimeError):
    pass


def _mine_with_timeout(
    *,
    row: Any,
    model: str,
    api_key: str | None,
    language_hint: str | None,
    row_timeout_seconds: float,
):
    if row_timeout_seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return mine_sarvam_teacher_candidate(
            row.text,
            model=model,
            api_key=api_key,
            language_hint=language_hint or row.language_hint,
            source=row.source,
            meta=row.meta,
        )

    def _handle_timeout(_signum: int, _frame: object) -> None:
        raise _RowTimeoutError(f"row timed out after {row_timeout_seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, row_timeout_seconds)
    try:
        return mine_sarvam_teacher_candidate(
            row.text,
            model=model,
            api_key=api_key,
            language_hint=language_hint or row.language_hint,
            source=row.source,
            meta=row.meta,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mine reviewable Hindi/Gujarati normalization candidates using Sarvam."
    )
    ap.add_argument("--input", required=True, help="Input JSONL with `text` or `input` per row.")
    ap.add_argument("--output", required=True, help="Output JSONL for mined candidate records.")
    ap.add_argument("--model", default="sarvam-m", help="Sarvam model id.")
    ap.add_argument("--api-key", default=None, help="Sarvam API key override.")
    ap.add_argument(
        "--language-hint",
        default=None,
        help="Optional global language hint override (`gu`, `hi`, `mixed`, `unknown`).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional max input rows to process. Use 0 for all rows.",
    )
    ap.add_argument(
        "--exclude-raw-response",
        action="store_true",
        help="Do not write the raw model response into output JSONL.",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first row-level Sarvam mining error.",
    )
    ap.add_argument(
        "--error-output",
        default=None,
        help="Optional JSONL path for row-level mining failures. Defaults next to --output.",
    )
    ap.add_argument(
        "--row-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional per-row timeout for remote mining calls. Use 0 to disable.",
    )
    args = ap.parse_args()

    try:
        inputs = load_sarvam_teacher_inputs_jsonl(args.input)
        if args.max_rows and args.max_rows > 0:
            inputs = inputs[: args.max_rows]

        out = []
        errors: list[dict[str, Any]] = []
        for row in inputs:
            try:
                rec = _mine_with_timeout(
                    row=row,
                    model=args.model,
                    api_key=args.api_key,
                    language_hint=args.language_hint,
                    row_timeout_seconds=float(args.row_timeout_seconds or 0.0),
                )
            except GckError as e:
                if args.fail_fast:
                    raise
                errors.append(
                    {
                        "input": row.text,
                        "language_hint": row.language_hint,
                        "source": row.source,
                        "meta": row.meta or {},
                        "error": str(e),
                    }
                )
                continue
            except _RowTimeoutError as e:
                if args.fail_fast:
                    raise GckError(str(e)) from e
                errors.append(
                    {
                        "input": row.text,
                        "language_hint": row.language_hint,
                        "source": row.source,
                        "meta": row.meta or {},
                        "error": str(e),
                    }
                )
                continue
            out.append(rec)

        dump_sarvam_teacher_records_jsonl(
            args.output,
            out,
            include_raw_response=not args.exclude_raw_response,
        )
        error_output_path = None
        if errors:
            error_output_path = Path(args.error_output) if args.error_output else _default_error_output_path(args.output)
            _dump_error_rows(error_output_path, errors)

        print(
            json.dumps(
                {
                    "input_path": str(args.input),
                    "output_path": str(args.output),
                    "model": args.model,
                    "n_rows": len(out),
                    "n_errors": len(errors),
                    "error_output_path": str(error_output_path) if error_output_path else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not out and errors:
            raise SystemExit(2)
    except GckError as e:
        sys.stderr.write(f"mine_sarvam_candidates: {e}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
