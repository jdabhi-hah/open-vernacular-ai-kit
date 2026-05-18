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
gck eval --dataset retrieval_uplift --k 5
gck eval --dataset retrieval_uplift --retrieval-query-pack codemix --k 5
gck eval --dataset retrieval_uplift --retrieval-query-pack codemix_hard --k 5
gck eval --dataset prompt_stability_uplift --n-variants 10
gck eval --dataset answer_quality_uplift
gck eval --dataset answer_quality_uplift --answer-case-pack hard
gck eval --dataset answer_quality_uplift --answer-case-pack distractor
gck eval --dataset answer_quality_uplift --answer-case-pack abstention
gck eval --dataset answer_quality_uplift --answer-case-pack suite
```

The packaged `language_sentences` dataset currently contains `120` exact-match Hindi and Gujarati
sentence regressions.

For downstream-quality checks, use:

- `retrieval_uplift`: compares retrieval recall with raw queries vs OVAK-normalized queries
- `--retrieval-query-pack codemix`: uses the packaged code-mixed retrieval query pack
- `--retrieval-query-pack codemix_hard`: uses the harder noisy/code-mixed retrieval pack
- `prompt_stability_uplift`: compares Sarvam prompt-stability with raw prompts vs OVAK-normalized prompts
- `answer_quality_uplift`: compares short-answer quality with raw questions vs OVAK-normalized questions using packaged gold contexts
- `--answer-case-pack hard`: uses the harder phrase-answer benchmark pack instead of the easier label-answer pack
- `--answer-case-pack distractor`: uses the preferred multi-doc distractor answer-quality pack
- `--answer-case-pack abstention`: uses unsupported-fact cases where the correct answer is `UNKNOWN`
- `--answer-case-pack suite`: runs the combined release-facing answer benchmark over `distractor` + `abstention`

Note: eval dependencies are optional; install with `pip install -e ".[eval]"`.

## Batch recipes

See `cookbook/batch-cli-recipes.md` for support-ticket and ecommerce batch recipes built with CLI commands.
