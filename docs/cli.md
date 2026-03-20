# CLI

The CLI entrypoint is `gck` (installed via `pyproject.toml` scripts).

## Normalize text

```bash
gck normalize "maru business plan ready chhe!!!"
```

## Render CodeMix

```bash
gck codemix "maru business plan ready chhe!!!"
```

Language profile (Gujarati stable, Hindi beta):

```bash
gck codemix --language gu "maru order status shu chhe?"
gck codemix --language hi "mera naam Sudhir hai"
```

Use `--stats` to print conversion statistics to stderr (stdout remains the rendered string).

## Eval harness

```bash
gck eval --dataset gujlish --report eval/out/report.json
gck eval --dataset golden_translit --language all --translit-mode sentence
gck eval --dataset language_sentences --language all --translit-mode sentence
```

The packaged `language_sentences` dataset currently contains `120` exact-match Hindi and Gujarati
sentence regressions.

Note: eval dependencies are optional; install with `pip install -e ".[eval]"`.

## Batch recipes

See `cookbook/batch-cli-recipes.md` for support-ticket and ecommerce batch recipes built with CLI commands.
