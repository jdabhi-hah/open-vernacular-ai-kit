from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .codemix_render import (
    analyze_codemix_with_config,
    render_codemix_with_config,
)
from .config import CodeMixConfig
from .doctor import collect_doctor_info
from .normalize import normalize_text

app = typer.Typer(add_completion=False, no_args_is_help=True)
_console = Console()


@app.command()
def normalize(
    text: str = typer.Argument(..., help="Input text (vernacular / English / code-mixed)."),
    numerals: str = typer.Option("keep", help="Numerals: keep (native digits) or ascii."),
) -> None:
    """Normalize punctuation, whitespace, and Indic-script text."""
    _console.print(normalize_text(text, numerals=numerals))


@app.command()
def codemix(
    text: str = typer.Argument(..., help="Input text (may include romanized vernacular text)."),
    language: str = typer.Option(
        "gu", help="Target language profile: gu (stable), hi (beta)."
    ),
    topk: int = typer.Option(1, help="Top-K transliteration candidates to consider."),
    numerals: str = typer.Option("keep", help="Numerals: keep (native digits) or ascii."),
    translit_mode: str = typer.Option(
        "token", help="Transliteration mode for romanized vernacular text: token or sentence."
    ),
    translit_backend: str = typer.Option(
        "auto", help="Transliteration backend: auto, ai4bharat, sanscript, none."
    ),
    user_lexicon: Optional[Path] = typer.Option(
        None, "--user-lexicon", help="Path to JSON/YAML file of roman->native-script overrides."
    ),
    fasttext_model: Optional[Path] = typer.Option(
        None, "--fasttext-model", help="Optional path to lid.176.ftz (fastText LID model)."
    ),
    preserve_case: bool = typer.Option(
        True, help="Preserve original case for Latin tokens (English + romanized vernacular)."
    ),
    preserve_numbers: bool = typer.Option(
        True, help="Preserve native digits (disable to normalize digits to ASCII)."
    ),
    aggressive_normalize: bool = typer.Option(
        False, help="Try extra romanized spelling variants before transliteration."
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Write CodeMix conversion stats to stderr as JSON (stdout remains the rendered string).",
    ),
) -> None:
    """Render a clean vernacular-English code-mix string."""
    cfg = CodeMixConfig(
        language=language,  # type: ignore[arg-type]
        topk=topk,
        numerals=numerals,  # type: ignore[arg-type]
        translit_mode=translit_mode,  # type: ignore[arg-type]
        translit_backend=translit_backend,  # type: ignore[arg-type]
        user_lexicon_path=None if user_lexicon is None else str(user_lexicon),
        fasttext_model_path=None if fasttext_model is None else str(fasttext_model),
        preserve_case=preserve_case,
        preserve_numbers=preserve_numbers,
        aggressive_normalize=aggressive_normalize,
    )
    if stats:
        a = analyze_codemix_with_config(text, config=cfg)
        _console.print(a.codemix)
        sys.stderr.write(
            json.dumps(
                {
                    "language": a.language,
                    "transliteration_backend": a.transliteration_backend,
                    "n_tokens": a.n_tokens,
                    "n_gu_roman_tokens": a.n_gu_roman_tokens,
                    "n_gu_roman_transliterated": a.n_gu_roman_transliterated,
                    "pct_gu_roman_transliterated": a.pct_gu_roman_transliterated,
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    else:
        _console.print(render_codemix_with_config(text, config=cfg))


@app.command()
def doctor(
    as_json: bool = typer.Option(
        True, "--json/--no-json", help="Print as JSON (recommended for copying into issues)."
    )
) -> None:
    """Print installed optional backends + versions."""
    info = collect_doctor_info()
    # IMPORTANT: avoid Rich formatting/ANSI so output stays valid JSON when `--json`.
    # (Rich may inject escape codes for styling/wrapping in some terminals.)
    payload = json.dumps(info, ensure_ascii=True, indent=2) + "\n"
    if as_json:
        sys.stdout.write(payload)
    else:
        sys.stdout.write(payload)


@app.command()
def eval(
    dataset: str = typer.Option(
        "gujlish",
        help=(
            "Eval dataset/suite: gujlish, golden_translit, language_sentences, retrieval, prompt_stability, "
            "dialect_id, dialect_normalization."
        ),
    ),
    report: Optional[Path] = typer.Option(
        None, help="Write a JSON report to this path (directories auto-created)."
    ),
    topk: int = typer.Option(1, help="Top-K transliteration candidates (gujlish/golden_translit/language_sentences)."),
    language: str = typer.Option(
        "gu",
        help="Target language for language-aware evals: gu, hi, or all (golden_translit/language_sentences).",
    ),
    max_rows: Optional[int] = typer.Option(
        2000, help="Max rows per split (gujlish). Use 0 for no limit."
    ),
    translit_mode: str = typer.Option(
        "token", help="Transliteration mode: token or sentence (golden_translit/language_sentences)."
    ),
    k: int = typer.Option(5, help="Top-k for retrieval recall (retrieval)."),
    embedding_model: str = typer.Option(
        "ai4bharat/indic-bert",
        help=(
            "HF model for embeddings (retrieval/prompt_stability). "
            "Note: ai4bharat/indic-bert may be gated on HF (the eval will fall back automatically)."
        ),
    ),
    sarvam_model: str = typer.Option(
        "sarvam-m", help="Sarvam chat model (prompt_stability; current hosted provider in this release)."
    ),
    n_variants: int = typer.Option(10, help="Number of prompt variants (prompt_stability)."),
    api_key: Optional[str] = typer.Option(
        None, help="Sarvam API key override (prompt_stability; future providers planned via PRs)."
    ),
    preprocess: bool = typer.Option(
        True, help="Preprocess text with normalize+codemix before eval (retrieval/prompt_stability)."
    ),
    dialect_dataset: Optional[Path] = typer.Option(
        None, "--dialect-dataset", help="Path to dialect-id JSONL (dialect_id eval)."
    ),
    dialect_norm_dataset: Optional[Path] = typer.Option(
        None,
        "--dialect-norm-dataset",
        help="Path to dialect-normalization JSONL (dialect_normalization eval).",
    ),
    dialect_backend: str = typer.Option(
        "heuristic", help="Dialect backend for dialect_id eval: auto, heuristic, transformers, none."
    ),
    dialect_model: Optional[str] = typer.Option(
        None,
        "--dialect-model",
        help="Dialect model id/path for Transformers backend (dialect_id eval).",
    ),
    dialect_normalizer_backend: str = typer.Option(
        "heuristic",
        help="Dialect normalizer backend for dialect_normalization eval: auto, heuristic, seq2seq, none.",
    ),
    dialect_normalizer_model: Optional[str] = typer.Option(
        None,
        "--dialect-normalizer-model",
        help="Seq2seq model id/path for dialect_normalization eval (optional).",
    ),
    allow_remote_models: bool = typer.Option(
        False,
        help="Allow remote HF model downloads (otherwise only local paths work).",
    ),
) -> None:
    """Run a lightweight, reproducible eval harness (downloads data if needed)."""
    try:
        from .eval_harness import run_eval
    except Exception:
        # Keep core library lightweight; eval dependencies are optional.
        _console.print(
            "[red]Eval harness is not installed.[/red] Install with: "
            "`pip install -e \".[eval]\"`"
        )
        raise typer.Exit(code=2)

    if max_rows == 0:
        max_rows = None
    try:
        result = run_eval(
            dataset=dataset,
            language=language,
            topk=topk,
            max_rows=max_rows,
            translit_mode=translit_mode,
            k=k,
            embedding_model=embedding_model,
            sarvam_model=sarvam_model,
            n_variants=n_variants,
            api_key=api_key,
            preprocess=preprocess,
            dialect_dataset_path=None if dialect_dataset is None else str(dialect_dataset),
            dialect_norm_dataset_path=None if dialect_norm_dataset is None else str(dialect_norm_dataset),
            dialect_backend=dialect_backend,
            dialect_model_id_or_path=dialect_model,
            dialect_normalizer_backend=dialect_normalizer_backend,
            dialect_normalizer_model_id_or_path=dialect_normalizer_model,
            allow_remote_models=allow_remote_models,
        )
    except Exception as e:
        _console.print(f"[red]Eval failed:[/red] {e}")
        raise typer.Exit(code=1)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        _console.print(f"[green]Wrote report:[/green] {report}")
    else:
        _console.print(json.dumps(result, ensure_ascii=True, indent=2))
 
