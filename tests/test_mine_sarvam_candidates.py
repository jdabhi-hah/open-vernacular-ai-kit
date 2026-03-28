from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from open_vernacular_ai_kit.errors import IntegrationError
from open_vernacular_ai_kit.sarvam_teacher import SarvamTeacherCandidateRecord, SarvamTeacherInput


def _load_script_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mine_sarvam_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("mine_sarvam_candidates_script", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_error_output_path_uses_neighboring_jsonl_name() -> None:
    mod = _load_script_module()
    out = mod._default_error_output_path("eval/out/sarvam_candidates/voice_note_seed.jsonl")
    assert out.as_posix().endswith("voice_note_seed.errors.jsonl")


def test_dump_error_rows_writes_jsonl(tmp_path) -> None:
    mod = _load_script_module()
    out = tmp_path / "errors.jsonl"
    mod._dump_error_rows(
        out,
        [
            {"input": "foo", "error": "bad response"},
            {"input": "bar", "error": "timeout"},
        ],
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["input"] == "foo"
    assert rows[1]["error"] == "timeout"


def test_main_continues_on_row_errors_and_writes_error_jsonl(tmp_path, monkeypatch, capsys) -> None:
    mod = _load_script_module()

    def fake_load(_: str):
        return [
            SarvamTeacherInput(text="ok row", language_hint="gu", source="seed-a"),
            SarvamTeacherInput(text="bad row", language_hint="gu", source="seed-b"),
        ]

    def fake_mine(text: str, **_: object):
        if text == "bad row":
            raise IntegrationError("invalid teacher response")
        return SarvamTeacherCandidateRecord(
            input=text,
            language_hint="gu",
            source="seed-a",
            model="sarvam-m",
            ovak_baseline="ઓકે",
            sarvam_native="ઓકે",
            sarvam_canonical="ઓકે",
            english_tokens_keep=[],
            candidate_tokens=[],
        )

    monkeypatch.setattr(mod, "load_sarvam_teacher_inputs_jsonl", fake_load)
    monkeypatch.setattr(mod, "mine_sarvam_teacher_candidate", fake_mine)

    output_path = tmp_path / "out.jsonl"
    argv = [
        "mine_sarvam_candidates.py",
        "--input",
        str(tmp_path / "input.jsonl"),
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr("sys.argv", argv)

    mod.main()

    out_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    err_path = tmp_path / "out.errors.jsonl"
    err_rows = [json.loads(line) for line in err_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    captured = json.loads(capsys.readouterr().out)

    assert len(out_rows) == 1
    assert out_rows[0]["input"] == "ok row"
    assert len(err_rows) == 1
    assert err_rows[0]["input"] == "bad row"
    assert captured["n_rows"] == 1
    assert captured["n_errors"] == 1


def test_main_times_out_slow_rows_and_writes_error_jsonl(tmp_path, monkeypatch, capsys) -> None:
    mod = _load_script_module()

    def fake_load(_: str):
        return [
            SarvamTeacherInput(text="slow row", language_hint="gu", source="seed-a"),
            SarvamTeacherInput(text="ok row", language_hint="hi", source="seed-b"),
        ]

    def fake_mine(text: str, **_: object):
        if text == "slow row":
            time.sleep(0.05)
        return SarvamTeacherCandidateRecord(
            input=text,
            language_hint="gu" if text == "slow row" else "hi",
            source="seed-a" if text == "slow row" else "seed-b",
            model="sarvam-m",
            ovak_baseline="ઓકે",
            sarvam_native="ઓકે",
            sarvam_canonical="ઓકે",
            english_tokens_keep=[],
            candidate_tokens=[],
        )

    monkeypatch.setattr(mod, "load_sarvam_teacher_inputs_jsonl", fake_load)
    monkeypatch.setattr(mod, "mine_sarvam_teacher_candidate", fake_mine)

    output_path = tmp_path / "out.jsonl"
    argv = [
        "mine_sarvam_candidates.py",
        "--input",
        str(tmp_path / "input.jsonl"),
        "--output",
        str(output_path),
        "--row-timeout-seconds",
        "0.01",
    ]
    monkeypatch.setattr("sys.argv", argv)

    mod.main()

    out_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    err_path = tmp_path / "out.errors.jsonl"
    err_rows = [json.loads(line) for line in err_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    captured = json.loads(capsys.readouterr().out)

    assert len(out_rows) == 1
    assert out_rows[0]["input"] == "ok row"
    assert len(err_rows) == 1
    assert err_rows[0]["input"] == "slow row"
    assert "timed out" in err_rows[0]["error"]
    assert captured["n_rows"] == 1
    assert captured["n_errors"] == 1
