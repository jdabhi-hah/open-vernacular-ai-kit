# Sarvam Teacher Mining

Use Sarvam as an offline teacher to discover better Hindi/Gujarati normalization candidates without
putting an LLM in the default runtime path.

## Why This Exists

The default OVAK normalization path should stay:

- deterministic
- offline-first
- easy to debug

This workflow is for mining candidate improvements that can later be reviewed and distilled into:

- language profile entries
- context-token rules
- sentence-level eval cases
- dialect assets

## Input Format

Create a JSONL file with one record per line:

```json
{"text":"tamne aaje office ma aavu chhe","language_hint":"gu","source":"support_chat"}
{"text":"meri maa ka naam kya hai","language_hint":"hi","source":"teacher_seed"}
```

Accepted fields:

- `text` or `input`
- `language_hint` (`gu`, `hi`, `mixed`, `unknown`)
- `source`
- `meta`

Bundled starter dataset:

- `eval/datasets/sarvam_teacher_seed.jsonl`
- `eval/datasets/sarvam_teacher_large_seed.jsonl`
- `eval/datasets/sarvam_teacher_realworld_seed.jsonl`
- `eval/datasets/sarvam_teacher_realworld_seed_followup.jsonl`
- `eval/datasets/sarvam_teacher_noisy_chat_seed.jsonl`
- `eval/datasets/sarvam_teacher_whatsapp_export_seed.jsonl`
- `eval/datasets/sarvam_teacher_voice_note_seed.jsonl`
- `eval/datasets/sarvam_teacher_ocr_screenshot_seed.jsonl`

## Build A Failure-Driven Seed

When a review cycle is exhausted, start the next one from actual eval regressions instead of another
manual seed list:

```bash
python3 scripts/build_sarvam_failure_seed.py \
  --output eval/out/sarvam_candidates/failure_seed.jsonl \
  --report eval/out/sarvam_candidates/failure_seed_report.json
```

This replays the packaged `language_sentences` and `golden_translit` evals and emits only rows that
currently fail. If the output is empty, the current shipped eval set is green and the next mining
cycle should use new real-world text instead of forcing more promotions from old seed data.

The bundled real-world pack exists for exactly that case:

- `eval/datasets/sarvam_teacher_realworld_seed.jsonl`

It contains `90` support, billing, logistics, account, and ecommerce-style prompts across:

- `30` Gujarati
- `30` Hindi
- `30` mixed Hindi/Gujarati + English

Use it when the failure-driven seed is empty:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_realworld_seed.jsonl \
  --output eval/out/sarvam_candidates/realworld_seed.jsonl \
  --model sarvam-m
```

After the first real-world review batches plateau, use the follow-up pack instead of repeatedly mining
the same `90` prompts:

- `eval/datasets/sarvam_teacher_realworld_seed_followup.jsonl`

It contains another `90` prompts across the same `30/30/30` Gujarati/Hindi/mixed split, but shifts
the domains toward harder operational flows such as:

- COD and duplicate charges
- exchange and replacement prechecks
- document submission and address-proof failures
- subscription and autorenew problems
- payout, EMI, and gift-card issues
- landmark, gate-number, and self-collect logistics prompts

Use it for the next Sarvam cycle after the first real-world pack has already produced its safe promotions:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_realworld_seed_followup.jsonl \
  --output eval/out/sarvam_candidates/realworld_seed_followup.jsonl \
  --model sarvam-m
```

After the follow-up pack also starts yielding mostly mixed-script or loanword-heavy leftovers, switch to
the noisy chat pack:

- `eval/datasets/sarvam_teacher_noisy_chat_seed.jsonl`

It keeps the same `30/30/30` Gujarati/Hindi/mixed split, but the prompts are intentionally noisier:

- abbreviations like `pls`, `stts`, `nthi`, `nhi`, `krdo`
- WhatsApp-style short messages and typo-heavy support text
- code-mixed customer-service phrasing closer to raw inbound chat than templated support copy

Use it to mine harder real-world vernacular behavior once the cleaner transactional packs plateau:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_noisy_chat_seed.jsonl \
  --output eval/out/sarvam_candidates/noisy_chat_seed.jsonl \
  --model sarvam-m
```

After the noisy chat pack starts yielding mostly single-message shorthand leftovers, switch to the
WhatsApp/export-style pack:

- `eval/datasets/sarvam_teacher_whatsapp_export_seed.jsonl`

It also keeps the `30/30/30` Gujarati/Hindi/mixed split, but the prompts are written to feel like
message exports and threaded support conversations rather than isolated support asks:

- last-message references like `last 3 msgs` or `kal bhi msg kiya tha`
- voice-note, screenshot, and chat-export references
- gate number, landmark, proof, and callback follow-ups phrased like ongoing chat threads
- WhatsApp-style code-mixed phrasing where context from earlier messages is assumed

Use it when the next Sarvam cycle needs conversational continuity instead of one-shot complaint text:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_whatsapp_export_seed.jsonl \
  --output eval/out/sarvam_candidates/whatsapp_export_seed.jsonl \
  --model sarvam-m
```

After the WhatsApp/export pack starts yielding mostly short, risky fragments, switch to the
voice-note / ASR-style pack:

- `eval/datasets/sarvam_teacher_voice_note_seed.jsonl`

It keeps the same `30/30/30` Gujarati/Hindi/mixed split, but the prompts are shaped like
spoken transcripts rather than typed chat:

- longer punctuation-light voice-note phrasing
- ASR-style merged clauses and lighter formatting
- repeated references to what was already said in audio
- support and logistics complaints phrased like someone speaking, not typing

Use it when the next Sarvam cycle should target spoken vernacular normalization instead of
message-thread fragments:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_voice_note_seed.jsonl \
  --output eval/out/sarvam_candidates/voice_note_seed.jsonl \
  --model sarvam-m
```

After the voice-note pack starts yielding mostly spoken fillers or malformed-response leftovers,
switch to the OCR / screenshot-style pack:

- `eval/datasets/sarvam_teacher_ocr_screenshot_seed.jsonl`

It keeps the same `30/30/30` Gujarati/Hindi/mixed split, but the prompts are shaped like text
lifted from screenshots, cropped invoices, OCR-ed labels, and broken UI captures:

- screenshot and OCR abbreviations like `stts`, `delvry`, and partially cut labels
- broken spacing and digit confusion in invoice, payout, and shipping text
- cropped field names, missing zeros, and line-break artifacts from screenshots
- image-derived support complaints where users refer to what the screenshot is showing

Use it when the next Sarvam cycle should target OCR-ish normalization and screenshot-derived
support text rather than more spoken or chatty inputs:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_ocr_screenshot_seed.jsonl \
  --output eval/out/sarvam_candidates/ocr_screenshot_seed.jsonl \
  --model sarvam-m
```

## Run Mining

Install the optional Sarvam dependency first:

```bash
pip install -e ".[sarvam]"
```

Then run:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_seed.jsonl \
  --output eval/out/sarvam_candidates/seed.jsonl \
  --model sarvam-m
```

Use `SARVAM_API_KEY` in your shell, or pass `--api-key`.

For a broader batch that mixes packaged sentence cases with extra support and ecommerce-style prompts:

```bash
python3 scripts/mine_sarvam_candidates.py \
  --input eval/datasets/sarvam_teacher_large_seed.jsonl \
  --output eval/out/sarvam_candidates/large_seed.jsonl \
  --model sarvam-m
```

The bundled large seed currently contains `165` rows and is intended for deeper review passes before
lexicon or context-rule promotion.

## Output Schema

Each output record contains:

- `input`
- `language_hint`
- `source`
- `model`
- `ovak_baseline`
- `sarvam_native`
- `sarvam_canonical`
- `english_tokens_keep`
- `candidate_tokens`
- `notes`
- `raw_response`

Example:

```json
{
  "input": "tamne aaje office ma aavu chhe",
  "language_hint": "gu",
  "source": "support_chat",
  "model": "sarvam-m",
  "ovak_baseline": "તમને આજે office માં આવું છે",
  "sarvam_native": "તમને આજે office માં આવવું છે",
  "sarvam_canonical": "તમને આજે office માં આવવું છે",
  "english_tokens_keep": ["office"],
  "candidate_tokens": [
    {
      "roman": "ma",
      "native": "માં",
      "type": "context_token",
      "confidence": 0.98,
      "notes": "locative postposition in Gujarati context"
    }
  ],
  "notes": "Keep obvious English tokens in Latin script."
}
```

## Important Constraint

Do not promote these records directly into shipped logic.

Use them as reviewed candidates only. The next step should be:

1. review mined records manually
2. accept or reject candidates
3. promote approved items into profile data or eval datasets
4. rerun tests and evals

## Initialize Review Scaffold

Create a reviewed JSONL scaffold from mined output:

```bash
python3 scripts/init_sarvam_review.py \
  --input eval/out/sarvam_candidates/seed.jsonl \
  --output eval/datasets/sarvam_teacher_seed_reviewed.jsonl
```

For larger mining batches, keep the initialized review scaffold under `eval/out/` until the review is
curated enough to become a committed dataset.

Reviewed records add:

- `review_action`
- `reviewed_expected`
- `approved_candidate_tokens`
- `review_notes`

Recommended actions:

- `accept_sentence_case`
- `accept_lexicon`
- `accept_context_rule`
- `accept_dialect_case`
- `reject`
- `pending`

## Build A Triage Report

Before manual review, generate a report that buckets mined candidates into:

- novel single-token candidates
- already-known mappings
- mapping conflicts
- phrase candidates
- English-keep candidates
- ambiguous candidates

Use it to focus review on low-risk profile additions and to avoid spending time on obvious loanwords
or phrase-like candidates that should not be auto-promoted.

```bash
python3 scripts/report_sarvam_candidates.py \
  --input eval/out/sarvam_candidates/realworld_seed.jsonl \
  --output eval/out/sarvam_candidates/realworld_seed_report.json
```

The report compares mined tokens against the current shipped `gu.json` and `hi.json` profiles and
surfaces the likely review buckets up front.

## Promote Sentence Cases

Only after review, promote accepted sentence cases:

```bash
python3 scripts/promote_sarvam_sentence_cases.py \
  --input eval/datasets/sarvam_teacher_seed_reviewed.jsonl \
  --report eval/out/sarvam_candidates/seed_promotion_report.json
```

This script:

- adds only `accept_sentence_case` rows
- skips exact duplicates already present in the packaged dataset
- reports conflicts instead of silently overwriting existing cases
- infers `gu` or `hi` for reviewed `mixed` rows from the expected script

## Promote Profile Candidates

Only after explicit token-level review, promote accepted lexicon and context-rule candidates:

```bash
python3 scripts/promote_sarvam_profile_candidates.py \
  --input eval/datasets/sarvam_teacher_seed_reviewed.jsonl \
  --report eval/out/sarvam_candidates/seed_profile_promotion_report.json
```

This script:

- uses only `approved_candidate_tokens`
- accepts only single-token roman candidates for profile promotion
- promotes `accept_lexicon` rows into `common_roman_tokens` + `default_exceptions`
- promotes `accept_context_rule` rows into `context_roman_tokens` + `default_exceptions`
- blocks mapping conflicts by default when an existing roman token maps to a different native form
- blocks cross-bucket moves by default when a context token is being promoted as a common token, or vice versa

This is intentionally stricter than sentence-case promotion. If a token is ambiguous enough to need a bucket move,
do that as a manual profile edit after review rather than through automatic promotion.
