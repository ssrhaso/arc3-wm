"""scripts/check_smoke_green.py — Phase-4 smoke verdict analyzer.

Reads launcher stdout + ``metrics.jsonl`` and prints a GREEN/RED
verdict followed by a per-criterion breakdown. Exit code 0 iff all
criteria PASS, non-zero otherwise.

Criteria (from the Phase-4 smoke spec):

  (a) WM regex matched: ``<N> keys / <P> params`` (expected 68 /
      9,898,179)
  (b) Counter reset confirmed: all three of {updates, batches, actions}
      are zero post-load (from launcher's ``live_counters_after_load=``
      line)
  (c) WM losses moving: std across the last 50 logged values is > 0
      for each of {loss/image, loss/dyn, loss/rep, loss/rew, loss/con}.
      ``rew``/``reward`` and ``con``/``cont`` aliases accepted.
  (d) No NaN in any loss column.
  (e) No OOM / Arcade crash strings in stdout.
  (f) ≥1 env step recorded in JSONL (any record with ``step`` > 0).

Usage:

    python scripts/check_smoke_green.py \\
        --stdout path/to/launcher.log \\
        --jsonl  path/to/logdir/metrics.jsonl

``--stdout -`` reads from stdin. The analyzer is laptop-runnable; no
JAX, no dreamerv3.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# Mirror the constants in scripts/launch_pergame.py to avoid a deferred
# import (the launcher module touches argparse + sys.path at import).
EXPECTED_KEYS = 68
EXPECTED_PARAMS = 9_898_179
WM_LOSS_KEYS = ("loss/image", "loss/dyn", "loss/rep", "loss/rew", "loss/con")
LOSS_KEY_ALIASES = {
    "loss/rew": ("loss/rew", "loss/reward"),
    "loss/con": ("loss/con", "loss/cont"),
}

CRASH_PATTERNS = [
    re.compile(r"\bOOM\b"),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"RESOURCE_EXHAUSTED"),
    re.compile(r"^Traceback \(most recent call last\)", re.MULTILINE),
    re.compile(r"\bSegmentation fault\b"),
    re.compile(r"\bKilled\b"),
    # Arcade-specific crashes (env_files missing, etc.)
    re.compile(r"arc_agi\.Arcade\.make.*returned None"),
    re.compile(r"environment_files.*not.*cached"),
    re.compile(r"AssertionError", re.MULTILINE),
]
"""Lines/substrings that indicate a fatal crash. The list is broad on
purpose — false positives are recoverable (re-run the analyzer), false
negatives mean missing a real bug."""


# ----------------------------------------------------------------------
# Parsers (pure functions over text / records)
# ----------------------------------------------------------------------


def parse_seed_line(stdout: str) -> dict[str, Any] | None:
    """Extract the launcher's ``WM seeded: ...`` line, return parsed fields.

    The launcher prints exactly one of these (only when --init-from-ckpt
    is set). Returns ``None`` if the line is absent (treated as criterion-a
    + criterion-b FAIL by the caller).

    Fields parsed: ``matched_keys`` (int), ``matched_params`` (int),
    ``live_counters_after_load`` (dict).
    """
    m = re.search(
        r"WM seeded:\s+matched_keys=(?P<keys>\d+)\s+"
        r"matched_params=(?P<params>[\d,]+)\s+"
        r"counters_before_reset=(?P<before>\{[^}]*\})\s+"
        r"live_counters_after_load=(?P<after>\{[^}]*\})",
        stdout,
    )
    if not m:
        return None
    # Live counters dict uses Python repr syntax; safe to eval as a literal.
    import ast
    return {
        "matched_keys": int(m.group("keys")),
        "matched_params": int(m.group("params").replace(",", "")),
        "live_counters_after_load": ast.literal_eval(m.group("after")),
        "counters_before_reset": ast.literal_eval(m.group("before")),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a metrics.jsonl into a list of records. NaN/Infinity tolerated."""
    records = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # DV3 occasionally emits NaN/Infinity in JSONL; the stdlib
            # parser tolerates these with allow_nan=True (the default).
            # A malformed line is a separate issue; record verbatim for
            # criterion-d to flag.
            records.append({"_raw": line, "_parse_error": True})
    return records


def std(values: list[float]) -> float:
    """Population std; 0.0 for a single sample or empty list."""
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def last_n_for_key(records: Iterable[dict], key: str, n: int = 50) -> list[float]:
    """Last N numeric values of ``key`` across records. NaN entries kept
    (criterion-d will catch them); records lacking the key are skipped."""
    vals: list[float] = []
    for r in records:
        if key in r:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                continue
    return vals[-n:]


def resolve_alias(records: Iterable[dict], key: str) -> str | None:
    """Pick whichever of the key's aliases appears in records."""
    records = list(records)
    for alias in LOSS_KEY_ALIASES.get(key, (key,)):
        for r in records:
            if alias in r:
                return alias
    return None


# ----------------------------------------------------------------------
# Criteria
# ----------------------------------------------------------------------


def criterion_a(seed: dict | None) -> tuple[bool, str]:
    """WM regex matched: <N> keys / <P> params (expected 68 / 9,898,179)."""
    if seed is None:
        return False, f"WM seeded line absent in stdout (expected {EXPECTED_KEYS} keys / {EXPECTED_PARAMS:,} params)"
    keys_ok = seed["matched_keys"] == EXPECTED_KEYS
    params_ok = seed["matched_params"] == EXPECTED_PARAMS
    detail = (
        f"WM regex matched: {seed['matched_keys']} keys / "
        f"{seed['matched_params']:,} params "
        f"(expected {EXPECTED_KEYS} / {EXPECTED_PARAMS:,})"
    )
    return (keys_ok and params_ok), detail


def criterion_b(seed: dict | None) -> tuple[bool, str]:
    """Counter reset confirmed (all three zero post-load)."""
    if seed is None:
        return False, "Counter reset unverified — WM seeded line absent"
    live = seed["live_counters_after_load"]
    all_zero = all(int(v) == 0 for v in live.values())
    return all_zero, f"Counter reset confirmed (all three zero post-load): live={live}"


def criterion_c(records: list[dict]) -> tuple[bool, dict[str, tuple[bool, str]]]:
    """WM losses moving: std > 0 across last 50 logged values, per loss."""
    per_loss: dict[str, tuple[bool, str]] = {}
    for key in WM_LOSS_KEYS:
        actual_key = resolve_alias(records, key)
        if actual_key is None:
            per_loss[key] = (False, f"{key}: KEY ABSENT in JSONL")
            continue
        vals = last_n_for_key(records, actual_key, n=50)
        if not vals:
            per_loss[key] = (False, f"{key}: no numeric values in JSONL")
            continue
        s = std(vals)
        ok = s > 1e-9
        per_loss[key] = (ok, f"{actual_key}: n={len(vals)} std={s:.4g}")
    all_pass = all(ok for ok, _ in per_loss.values())
    return all_pass, per_loss


def criterion_d(records: list[dict]) -> tuple[bool, str]:
    """No NaN in any loss column."""
    nan_hits: list[str] = []
    for r in records:
        if r.get("_parse_error"):
            # NaN/Infinity in a malformed line — flag with line preview.
            raw = r.get("_raw", "")[:100]
            if "NaN" in raw or "Infinity" in raw:
                nan_hits.append(f"raw line contains NaN/Infinity: {raw!r}")
            continue
        for k, v in r.items():
            if not k.startswith("loss/"):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                nan_hits.append(f"{k}={v!r}")
    if nan_hits:
        return False, f"NaN/Inf found in losses: {nan_hits[:5]}"
    return True, "No NaN/Inf in any loss column"


def criterion_e(stdout: str) -> tuple[bool, str]:
    """No OOM / Arcade crash strings in stdout."""
    hits: list[str] = []
    for pat in CRASH_PATTERNS:
        m = pat.search(stdout)
        if m:
            hits.append(f"{pat.pattern!r}: {m.group(0)!r}")
    if hits:
        return False, f"Crash signatures in stdout: {hits[:5]}"
    return True, "No OOM / Arcade crash strings in stdout"


def criterion_f(records: list[dict]) -> tuple[bool, str]:
    """≥1 env step recorded in JSONL."""
    max_step = 0
    for r in records:
        s = r.get("step")
        if s is None:
            continue
        try:
            max_step = max(max_step, int(s))
        except (TypeError, ValueError):
            continue
    ok = max_step > 0
    return ok, f"max(step) in JSONL = {max_step} (need ≥ 1)"


# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------


def verdict(stdout: str, records: list[dict]) -> tuple[bool, list[str]]:
    """Run all criteria, return ``(green, lines)``."""
    seed = parse_seed_line(stdout)

    a_ok, a_msg = criterion_a(seed)
    b_ok, b_msg = criterion_b(seed)
    c_ok, c_per = criterion_c(records)
    d_ok, d_msg = criterion_d(records)
    e_ok, e_msg = criterion_e(stdout)
    f_ok, f_msg = criterion_f(records)

    green = a_ok and b_ok and c_ok and d_ok and e_ok and f_ok

    lines = [
        "GREEN" if green else "RED",
        f"  (a) {a_msg} → {'PASS' if a_ok else 'FAIL'}",
        f"  (b) {b_msg} → {'PASS' if b_ok else 'FAIL'}",
        "  (c) WM losses moving (last 50 logged values per loss):",
    ]
    for key in WM_LOSS_KEYS:
        ok, msg = c_per[key]
        lines.append(f"        {msg} → {'PASS' if ok else 'FAIL'}")
    lines.append(f"  (d) {d_msg} → {'PASS' if d_ok else 'FAIL'}")
    lines.append(f"  (e) {e_msg} → {'PASS' if e_ok else 'FAIL'}")
    lines.append(f"  (f) {f_msg} → {'PASS' if f_ok else 'FAIL'}")
    return green, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_smoke_green.py", description=__doc__)
    parser.add_argument(
        "--stdout",
        required=True,
        help="Path to launcher stdout log, or '-' for stdin.",
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to metrics.jsonl produced by the launcher.",
    )
    args = parser.parse_args(argv)

    if args.stdout == "-":
        stdout_text = sys.stdin.read()
    else:
        stdout_text = Path(args.stdout).read_text(encoding="utf-8", errors="replace")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print("RED")
        print(f"  pre-flight: jsonl path does not exist: {jsonl_path} → FAIL")
        return 1
    records = read_jsonl(jsonl_path)

    green, lines = verdict(stdout_text, records)
    for line in lines:
        print(line)
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
