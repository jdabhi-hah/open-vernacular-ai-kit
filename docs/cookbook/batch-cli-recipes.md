# Batch CLI Recipes (Support + Ecommerce)

These recipes use only the existing `gck codemix` CLI command and shell tools.

## Recipe 1: Support tickets JSONL

Input format (`tickets.jsonl`): one JSON object per line with a `text` field.

```json
{"ticket_id":"t1","text":"maru payment nathi thayu"}
{"ticket_id":"t2","text":"order status shu chhe?"}
```

Run batch normalization:

```bash
jq -rc '.' tickets.jsonl | while IFS= read -r row; do
  text="$(printf '%s' "$row" | jq -r '.text // ""')"
  codemix="$(gck codemix --language gu --translit-mode sentence "$text")"
  printf '%s\n' "$row" | jq --arg codemix "$codemix" '. + {codemix: $codemix}'
done > tickets.codemix.jsonl
```

Quick quality check:

```bash
jq -r '.codemix' tickets.codemix.jsonl | head -n 5
```

## Recipe 2: Ecommerce query feed (CSV)

Input format (`queries.csv`): has a `query` column.

```csv
query
maru order return karvu chhe
aaje delivery kyare aavse
```

Batch-run with a tiny Python wrapper around CLI:

```bash
python3 - <<'PY'
import csv
import subprocess

in_path = "queries.csv"
out_path = "queries.codemix.csv"

with open(in_path, "r", encoding="utf-8", newline="") as f_in, open(
    out_path, "w", encoding="utf-8", newline=""
) as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = list(reader.fieldnames or []) + ["codemix"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        query = (row.get("query") or "").strip()
        proc = subprocess.run(
            [
                "gck",
                "codemix",
                "--language",
                "gu",
                "--translit-mode",
                "sentence",
                query,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        row["codemix"] = proc.stdout.strip()
        writer.writerow(row)

print(f"wrote: {out_path}")
PY
```

## Recipe 3: Hindi beta stream preprocessing

```bash
while IFS= read -r line; do
  gck codemix --language hi --translit-mode sentence "$line"
done < incoming_hindi_queries.txt > incoming_hindi_queries.codemix.txt
```

## Notes

- Prefer `translit_mode="sentence"` for chat/support style text.
- Keep `--language gu` for production Gujarati flows.
- Use `--language hi` as beta profile while collecting feedback.
