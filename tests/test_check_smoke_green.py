"""Unit tests for scripts/check_smoke_green.py.

Pure-function tests against synthetic stdout strings and JSONL record
lists. No filesystem dependency beyond what tmp_path provides; no
JAX, no dreamerv3. Mirrors the structure of test_launcher_warmstart.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import scripts.check_smoke_green as G


# ----------------------------------------------------------------------
# parse_seed_line
# ----------------------------------------------------------------------


_GOOD_SEED_LINE = (
    "Replica: 0 / 1\n"
    "Logdir:  /root/logdir/p4-vc33-s0\n"
    "Init-ckpt resolved to: /root/cache/latest.pkl\n"
    "WM seeded: matched_keys=68 matched_params=9,898,179 "
    "counters_before_reset={'updates': 192000, 'batches': 192001, 'actions': 0} "
    "live_counters_after_load={'updates': 0, 'batches': 0, 'actions': 0}\n"
    "Start training loop\n"
)


def test_parse_seed_line_happy():
    seed = G.parse_seed_line(_GOOD_SEED_LINE)
    assert seed is not None
    assert seed["matched_keys"] == 68
    assert seed["matched_params"] == 9_898_179
    assert seed["live_counters_after_load"] == {"updates": 0, "batches": 0, "actions": 0}
    assert seed["counters_before_reset"] == {"updates": 192_000, "batches": 192_001, "actions": 0}


def test_parse_seed_line_absent_returns_none():
    assert G.parse_seed_line("nothing here") is None


# ----------------------------------------------------------------------
# Criterion a — WM regex matched
# ----------------------------------------------------------------------


def test_criterion_a_happy():
    seed = G.parse_seed_line(_GOOD_SEED_LINE)
    ok, msg = G.criterion_a(seed)
    assert ok is True
    assert "68 keys" in msg and "9,898,179 params" in msg


def test_criterion_a_seed_absent_fails():
    ok, msg = G.criterion_a(None)
    assert ok is False
    assert "absent" in msg.lower()


def test_criterion_a_key_count_drift_fails():
    seed = {
        "matched_keys": 67,
        "matched_params": 9_898_179,
        "live_counters_after_load": {"updates": 0, "batches": 0, "actions": 0},
        "counters_before_reset": {"updates": 192_000, "batches": 192_001, "actions": 0},
    }
    ok, _ = G.criterion_a(seed)
    assert ok is False


def test_criterion_a_param_count_drift_fails():
    seed = {
        "matched_keys": 68,
        "matched_params": 999,
        "live_counters_after_load": {"updates": 0, "batches": 0, "actions": 0},
        "counters_before_reset": {"updates": 192_000, "batches": 192_001, "actions": 0},
    }
    ok, _ = G.criterion_a(seed)
    assert ok is False


# ----------------------------------------------------------------------
# Criterion b — Counter reset
# ----------------------------------------------------------------------


def test_criterion_b_happy():
    seed = G.parse_seed_line(_GOOD_SEED_LINE)
    ok, msg = G.criterion_b(seed)
    assert ok is True
    assert "zero" in msg.lower()


def test_criterion_b_nonzero_counter_fails():
    seed = {
        "matched_keys": 68,
        "matched_params": 9_898_179,
        "live_counters_after_load": {"updates": 192_000, "batches": 0, "actions": 0},
        "counters_before_reset": {"updates": 192_000, "batches": 192_001, "actions": 0},
    }
    ok, _ = G.criterion_b(seed)
    assert ok is False


def test_criterion_b_seed_absent_fails():
    ok, _ = G.criterion_b(None)
    assert ok is False


# ----------------------------------------------------------------------
# Criterion c — WM losses moving
# ----------------------------------------------------------------------


def _records_with_moving_losses(n: int = 60) -> list[dict]:
    """Synthetic JSONL records: every loss has non-trivial std over N steps."""
    out = []
    for i in range(n):
        out.append({
            "step": i + 1,
            "loss/image": 100.0 - i * 0.5,    # monotonic decrease, big std
            "loss/dyn":   5.0 + 0.01 * i,
            "loss/rep":   2.0 - 0.005 * i,
            "loss/rew":   0.5 + 0.001 * (i % 5),
            "loss/con":   0.02 + 0.0001 * i,
        })
    return out


def test_criterion_c_happy_path():
    records = _records_with_moving_losses()
    all_ok, per = G.criterion_c(records)
    assert all_ok is True
    for k in G.WM_LOSS_KEYS:
        ok, _ = per[k]
        assert ok is True, f"{k}: expected PASS"


def test_criterion_c_flatlined_loss_fails():
    """If a loss is constant (std == 0), criterion fails for that key."""
    records = _records_with_moving_losses()
    for r in records:
        r["loss/dyn"] = 1.0  # flatline
    all_ok, per = G.criterion_c(records)
    assert all_ok is False
    assert per["loss/dyn"][0] is False
    # Other losses still moving.
    assert per["loss/image"][0] is True


def test_criterion_c_missing_key_fails():
    """Absent loss key → that criterion fails."""
    records = _records_with_moving_losses()
    for r in records:
        r.pop("loss/rep", None)
    all_ok, per = G.criterion_c(records)
    assert all_ok is False
    assert per["loss/rep"][0] is False
    assert "ABSENT" in per["loss/rep"][1]


def test_criterion_c_alias_resolution():
    """loss/reward (not loss/rew) and loss/cont (not loss/con) accepted."""
    records = _records_with_moving_losses()
    for r in records:
        r["loss/reward"] = r.pop("loss/rew")
        r["loss/cont"] = r.pop("loss/con")
    all_ok, per = G.criterion_c(records)
    assert all_ok is True
    # Reported under the alias actually present:
    assert "loss/reward" in per["loss/rew"][1]
    assert "loss/cont" in per["loss/con"][1]


# ----------------------------------------------------------------------
# Criterion d — No NaN
# ----------------------------------------------------------------------


def test_criterion_d_clean():
    records = _records_with_moving_losses()
    ok, _ = G.criterion_d(records)
    assert ok is True


def test_criterion_d_nan_in_loss_fails():
    records = _records_with_moving_losses()
    records[10]["loss/dyn"] = float("nan")
    ok, msg = G.criterion_d(records)
    assert ok is False
    assert "loss/dyn" in msg


def test_criterion_d_inf_in_loss_fails():
    records = _records_with_moving_losses()
    records[10]["loss/image"] = float("inf")
    ok, msg = G.criterion_d(records)
    assert ok is False
    assert "loss/image" in msg


def test_criterion_d_nan_in_raw_line_fails():
    """JSONL line that didn't parse but contains NaN flagged."""
    records = [{"_raw": '{"step": 1, "loss/image": NaN}', "_parse_error": True}]
    ok, msg = G.criterion_d(records)
    assert ok is False
    assert "NaN" in msg


# ----------------------------------------------------------------------
# Criterion e — No crash strings
# ----------------------------------------------------------------------


def test_criterion_e_clean():
    ok, _ = G.criterion_e(_GOOD_SEED_LINE)
    assert ok is True


def test_criterion_e_oom_fails():
    ok, msg = G.criterion_e("training ok\nRuntimeError: OOM out of memory\n")
    assert ok is False
    assert "OOM" in msg


def test_criterion_e_traceback_fails():
    ok, _ = G.criterion_e(
        "Start training loop\nTraceback (most recent call last):\n  File ...\n"
    )
    assert ok is False


def test_criterion_e_arcade_make_returned_none_fails():
    ok, msg = G.criterion_e(
        "Start training loop\n"
        "RuntimeError: arc_agi.Arcade.make('vc33') returned None; "
        "environment_files/vc33/ may not be cached.\n"
    )
    assert ok is False
    assert "Arcade" in msg or "returned None" in msg


def test_criterion_e_segfault_fails():
    ok, _ = G.criterion_e("Segmentation fault (core dumped)\n")
    assert ok is False


# ----------------------------------------------------------------------
# Criterion f — ≥1 env step
# ----------------------------------------------------------------------


def test_criterion_f_happy():
    records = _records_with_moving_losses()  # step 1..60
    ok, msg = G.criterion_f(records)
    assert ok is True
    assert "60" in msg


def test_criterion_f_zero_steps_fails():
    records = [{"step": 0, "loss/image": 100.0}]
    ok, _ = G.criterion_f(records)
    assert ok is False


def test_criterion_f_no_step_field_fails():
    records = [{"loss/image": 100.0}]
    ok, _ = G.criterion_f(records)
    assert ok is False


# ----------------------------------------------------------------------
# End-to-end verdict
# ----------------------------------------------------------------------


def test_verdict_all_green():
    records = _records_with_moving_losses()
    green, lines = G.verdict(_GOOD_SEED_LINE, records)
    assert green is True
    assert lines[0] == "GREEN"


def test_verdict_red_on_seed_absent():
    records = _records_with_moving_losses()
    green, lines = G.verdict("no seed line here", records)
    assert green is False
    assert lines[0] == "RED"
    # Criterion (a) and (b) both FAIL when seed line absent.
    assert any("(a)" in l and "FAIL" in l for l in lines)
    assert any("(b)" in l and "FAIL" in l for l in lines)


def test_verdict_red_on_nan_loss():
    records = _records_with_moving_losses()
    records[10]["loss/dyn"] = float("nan")
    green, lines = G.verdict(_GOOD_SEED_LINE, records)
    assert green is False
    assert any("(d)" in l and "FAIL" in l for l in lines)


def test_verdict_red_on_arcade_crash():
    records = _records_with_moving_losses()
    crash_stdout = _GOOD_SEED_LINE + "\nRuntimeError: arc_agi.Arcade.make('vc33') returned None\n"
    green, lines = G.verdict(crash_stdout, records)
    assert green is False
    assert any("(e)" in l and "FAIL" in l for l in lines)


# ----------------------------------------------------------------------
# CLI smoke
# ----------------------------------------------------------------------


def test_cli_main_green_exits_zero(tmp_path, capsys):
    stdout_log = tmp_path / "launcher.log"
    stdout_log.write_text(_GOOD_SEED_LINE, encoding="utf-8")
    jsonl = tmp_path / "metrics.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r) for r in _records_with_moving_losses()),
        encoding="utf-8",
    )

    rc = G.main(["--stdout", str(stdout_log), "--jsonl", str(jsonl)])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert captured.out.startswith("GREEN\n")


def test_cli_main_red_exits_nonzero(tmp_path, capsys):
    stdout_log = tmp_path / "launcher.log"
    stdout_log.write_text("nothing useful here", encoding="utf-8")
    jsonl = tmp_path / "metrics.jsonl"
    jsonl.write_text("", encoding="utf-8")

    rc = G.main(["--stdout", str(stdout_log), "--jsonl", str(jsonl)])
    captured = capsys.readouterr()
    assert rc != 0
    assert captured.out.startswith("RED\n")


def test_cli_main_missing_jsonl_red(tmp_path, capsys):
    stdout_log = tmp_path / "launcher.log"
    stdout_log.write_text(_GOOD_SEED_LINE, encoding="utf-8")
    rc = G.main(["--stdout", str(stdout_log), "--jsonl", str(tmp_path / "nope.jsonl")])
    captured = capsys.readouterr()
    assert rc != 0
    assert captured.out.startswith("RED\n")
