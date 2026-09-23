"""CLI surface for the per-task action diagnostic (scripts/action_diagnostics.py).

Uses a tiny synthetic replay fixture so the tests are fast and deterministic
(the real-data validity checks live in tests/test_action_diagnostics.py).
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CLI = _REPO / "scripts" / "action_diagnostics.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("action_diagnostics_cli", _CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_replay_dir(tmp_path: Path, game: str, available) -> Path:
    """Minimal data/replays-style fixture: one recording with given actions."""
    root = tmp_path / "replays"
    gdir = root / game
    gdir.mkdir(parents=True)
    rec = gdir / "synthetic.recording.jsonl"
    rows = [
        {"data": {"available_actions": list(available), "win_levels": 7}},
        {"data": {"available_actions": list(available), "win_levels": 7}},
    ]
    rec.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return root


def test_cli_replays_writes_json_and_csv(tmp_path):
    cli = _load_cli()
    replays = _make_replay_dir(tmp_path, "ls20", [1, 2, 3, 4])
    out_json = tmp_path / "ls20.json"
    out_csv = tmp_path / "ls20.csv"
    rc = cli.main(
        [
            "--game-id", "ls20",
            "--source", "replays",
            "--replays-dir", str(replays),
            "--out-json", str(out_json),
            "--out-csv", str(out_csv),
        ]
    )
    assert rc == 0
    obj = json.loads(out_json.read_text())
    assert obj["task_id"] == "ls20"
    assert obj["n_valid"] == 4
    assert obj["dilution_ratio"] == pytest.approx(4102 / 4)
    # usage null without an action log
    assert obj["actions"][0]["usage_count"] is None
    rows = list(csv.DictReader(io.StringIO(out_csv.read_text())))
    assert len(rows) == 4102


def test_cli_summary_line(tmp_path, capsys):
    cli = _load_cli()
    replays = _make_replay_dir(tmp_path, "ls20", [1, 2, 3, 4])
    cli.main(["--game-id", "ls20", "--source", "replays", "--replays-dir", str(replays)])
    out = capsys.readouterr().out
    assert "ls20" in out and "n_valid=4" in out
    assert "usage=null" in out  # honest-null flagged in the summary


def test_cli_action_log_populates_usage(tmp_path):
    cli = _load_cli()
    replays = _make_replay_dir(tmp_path, "ls20", [1, 2, 3, 4])
    log = tmp_path / "actions.jsonl"
    log.write_text('{"actions": [0, 0, 1]}\n', encoding="utf-8")
    out_json = tmp_path / "ls20.json"
    cli.main(
        [
            "--game-id", "ls20",
            "--source", "replays",
            "--replays-dir", str(replays),
            "--action-log", str(log),
            "--out-json", str(out_json),
        ]
    )
    obj = json.loads(out_json.read_text())
    assert obj["actions"][0]["usage_count"] == 2
    assert obj["actions"][0]["usage_fraction"] == pytest.approx(2 / 3)
    assert obj["actions"][0]["sources"]["usage_count"] == "run-measured"


def test_cli_budget_model_override(tmp_path):
    cli = _load_cli()
    replays = _make_replay_dir(tmp_path, "ls20", [1, 2, 3, 4])
    out_json = tmp_path / "ls20.json"
    cli.main(
        [
            "--game-id", "ls20",
            "--source", "replays",
            "--replays-dir", str(replays),
            "--budget-model", "lf52",
            "--out-json", str(out_json),
        ]
    )
    obj = json.loads(out_json.read_text())
    # ACTION7 (idx 4101) carries the lf52 undo weight under the override.
    assert obj["actions"][4101]["budget_cost"] == 20
    assert obj["actions"][4101]["sources"]["budget_cost"] == "engine:lf52"
