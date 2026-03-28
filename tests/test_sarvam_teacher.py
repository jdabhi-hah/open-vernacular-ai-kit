from __future__ import annotations

import json

from open_vernacular_ai_kit.sarvam_teacher import (
    build_sarvam_teacher_prompt,
    dump_sarvam_teacher_records_jsonl,
    load_sarvam_teacher_inputs_jsonl,
    load_sarvam_teacher_records_jsonl,
    mine_sarvam_teacher_candidate,
    parse_sarvam_teacher_response,
)


def test_build_sarvam_teacher_prompt_includes_schema_and_baseline() -> None:
    prompt = build_sarvam_teacher_prompt(
        "tamne aaje office ma aavu chhe",
        language_hint="gu",
        ovak_baseline="તમને આજે office માં આવું છે",
    )

    assert "Return only one JSON object" in prompt
    assert '"language_hint": "gu"' in prompt
    assert '"ovak_baseline": "તમને આજે office માં આવું છે"' in prompt
    assert '"candidate_tokens"' in prompt


def test_parse_sarvam_teacher_response_handles_code_fence_and_filters_invalid_candidates() -> None:
    raw = """```json
{
  "language_hint": "Gujarati",
  "sarvam_native": "તમને આજે office માં આવવું છે",
  "sarvam_canonical": "તમને આજે office માં આવવું છે",
  "english_tokens_keep": ["office", ""],
  "candidate_tokens": [
    {"roman": "ma", "native": "માં", "type": "context-token", "confidence": 1.2, "notes": "locative"},
    {"roman": "", "native": "આવવું", "type": "verb_phrase"},
    {"roman": "aavu", "native": "આવવું", "type": "verb phrase", "confidence": "0.88"}
  ],
  "notes": "keep office in English"
}
```"""

    rec = parse_sarvam_teacher_response(
        raw,
        input_text="tamne aaje office ma aavu chhe",
        source="unit-test",
        model="sarvam-m",
        ovak_baseline="તમને આજે office માં આવું છે",
        fallback_language_hint="gu",
    )

    assert rec.language_hint == "gu"
    assert rec.sarvam_native == "તમને આજે office માં આવવું છે"
    assert rec.english_tokens_keep == ["office"]
    assert len(rec.candidate_tokens) == 2
    assert rec.candidate_tokens[0].candidate_type == "context_token"
    assert rec.candidate_tokens[0].confidence == 1.0
    assert rec.candidate_tokens[1].candidate_type == "verb_phrase"


def test_parse_sarvam_teacher_response_recovers_first_balanced_object_with_trailing_text() -> None:
    raw = """Here is the analysis:
{
  "language_hint": "gu",
  "sarvam_native": "invoice pic માં gst no half cut થયો છે તમે full copy મોકલો",
  "sarvam_canonical": "invoice pic માં GST નો half cut થયો છે તમે full copy મોકલો",
  "english_tokens_keep": ["invoice", "pic", "GST", "half", "copy"],
  "candidate_tokens": [
    {"roman": "ma", "native": "માં", "type": "context_token", "confidence": 0.98}
  ],
  "notes": "keep invoice terms in English"
}
Additional explanation that should be ignored.
"""

    rec = parse_sarvam_teacher_response(
        raw,
        input_text="invoice pic ma gst no half cut thay gyo chhe tame full copy moklo",
        source="unit-test",
        model="sarvam-m",
        ovak_baseline="invoice pic માં gst no half cut thay gyo છે તમે full copy મોકલો",
        fallback_language_hint="gu",
    )

    assert rec.language_hint == "gu"
    assert rec.sarvam_canonical == "invoice pic માં GST નો half cut થયો છે તમે full copy મોકલો"
    assert rec.candidate_tokens[0].roman == "ma"


def test_parse_sarvam_teacher_response_recovers_first_object_from_generic_code_fence() -> None:
    raw = """```
preface text
{
  "language_hint": "hi",
  "sarvam_native": "मेरे account से same amount दो बार debit hua",
  "sarvam_canonical": "मेरे account से same amount दो बार debit hua",
  "english_tokens_keep": ["account", "same", "debit"],
  "candidate_tokens": [
    {"roman": "do", "native": "दो", "type": "lexicon", "confidence": 0.91}
  ],
  "notes": "keep account terms in English"
}
```
"""

    rec = parse_sarvam_teacher_response(
        raw,
        input_text="mere account se same amount do baar debit hua",
        source="unit-test",
        model="sarvam-m",
        ovak_baseline="मेरे account से same amount do बार debit hua",
        fallback_language_hint="hi",
    )

    assert rec.language_hint == "hi"
    assert rec.candidate_tokens[0].roman == "do"
    assert rec.candidate_tokens[0].native == "दो"


def test_mine_sarvam_teacher_candidate_uses_injected_call_model() -> None:
    def fake_call(prompt: str) -> str:
        assert "meri maa ka naam kya hai" in prompt
        return json.dumps(
            {
                "language_hint": "hi",
                "sarvam_native": "मेरी माँ का नाम क्या है",
                "sarvam_canonical": "मेरी माँ का नाम क्या है",
                "english_tokens_keep": [],
                "candidate_tokens": [
                    {
                        "roman": "meri",
                        "native": "मेरी",
                        "type": "lexicon",
                        "confidence": 0.97,
                    }
                ],
                "notes": "possessive phrase in Hindi",
            },
            ensure_ascii=False,
        )

    rec = mine_sarvam_teacher_candidate(
        "meri maa ka naam kya hai",
        language_hint="hi",
        source="unit-test",
        call_model=fake_call,
    )

    assert rec.language_hint == "hi"
    assert rec.model == "sarvam-m"
    assert rec.sarvam_canonical == "मेरी माँ का नाम क्या है"
    assert rec.candidate_tokens[0].roman == "meri"


def test_teacher_jsonl_round_trip(tmp_path) -> None:
    input_path = tmp_path / "teacher_input.jsonl"
    input_path.write_text(
        json.dumps({"text": "shu tame mane madad kari shako?", "language_hint": "gu"}) + "\n",
        encoding="utf-8",
    )
    rows = load_sarvam_teacher_inputs_jsonl(input_path)
    assert rows[0].text == "shu tame mane madad kari shako?"
    assert rows[0].language_hint == "gu"

    def fake_call(_: str) -> str:
        return json.dumps(
            {
                "language_hint": "gu",
                "sarvam_native": "શું તમે મને મદદ કરી શકો?",
                "sarvam_canonical": "શું તમે મને મદદ કરી શકો?",
                "english_tokens_keep": [],
                "candidate_tokens": [],
                "notes": "",
            },
            ensure_ascii=False,
        )

    rec = mine_sarvam_teacher_candidate(rows[0].text, language_hint="gu", call_model=fake_call)
    out_path = tmp_path / "teacher_output.jsonl"
    dump_sarvam_teacher_records_jsonl(out_path, [rec], include_raw_response=False)
    out = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert out["sarvam_canonical"] == "શું તમે મને મદદ કરી શકો?"
    assert "raw_response" not in out


def test_load_teacher_records_jsonl_preserves_raw_response_and_normalizes_candidates(tmp_path) -> None:
    out_path = tmp_path / "teacher_output.jsonl"
    out_path.write_text(
        json.dumps(
            {
                "input": "mujhe order status batayiye",
                "language_hint": "Hindi",
                "source": "unit-test",
                "model": "sarvam-m",
                "ovak_baseline": "मुझे order status बतायीये",
                "sarvam_native": "मुझे order status बताइए",
                "sarvam_canonical": "मुझे order status बताइए",
                "english_tokens_keep": ["order", "", "status"],
                "candidate_tokens": [
                    {
                        "roman": "batayiye",
                        "native": "बताइए",
                        "type": "verb phrase",
                        "confidence": "0.91",
                        "notes": "imperative support request",
                    }
                ],
                "notes": "keep product terms in English",
                "raw_response": "```json {\"language_hint\":\"hi\"} ```",
                "meta": {"domain": "support"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_sarvam_teacher_records_jsonl(out_path)
    rec = rows[0]
    assert rec.language_hint == "hi"
    assert rec.english_tokens_keep == ["order", "status"]
    assert rec.candidate_tokens[0].candidate_type == "verb_phrase"
    assert rec.candidate_tokens[0].confidence == 0.91
    assert rec.raw_response == "```json {\"language_hint\":\"hi\"} ```"
