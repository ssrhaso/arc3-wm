"""Tests for scripts/build_ladder_table.py: run discovery, Wilson intervals, LaTeX."""
from __future__ import annotations

import json

import pytest

from scripts.build_ladder_table import (
    ARMS,
    fmt_ladder_tex,
    fmt_pergame_tex,
    ladder_rows,
    load_runs,
    wilson,
)


def _run(root, batch, arm, game, seed, *, clears, n, official, done=True, score=True):
    d = root / batch / f"{arm}-{game}-s{seed}"
    d.mkdir(parents=True)
    if done:
        (d / "DONE").touch()
    if score:
        (d / "score_eval100.json").write_text(json.dumps({
            "game": game, "n": n, "clears": clears, "clear_rate": clears / n if n else 0.0,
            "official_max": official,
        }), encoding="utf-8")
    return d


# --- Wilson ---------------------------------------------------------------

def test_wilson_brackets_the_point_estimate():
    """50/100 gives the textbook Wilson interval [0.4038, 0.5962]."""
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_at_zero_and_one_stays_in_range():
    """The interval never leaves [0, 1] and is one-sided at the extremes."""
    lo, hi = wilson(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = wilson(100, 100)
    assert 0.95 < lo < 1.0 and hi == pytest.approx(1.0, abs=1e-3) and hi <= 1.0


def test_wilson_empty_sample():
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_narrows_with_more_data():
    narrow = wilson(500, 1000)
    wide = wilson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --- discovery ------------------------------------------------------------

def test_load_runs_groups_by_batch_and_game(tmp_path):
    _run(tmp_path, "paired", "cold", "cd82", 0, clears=0, n=100, official=0.0)
    _run(tmp_path, "paired", "cold", "cd82", 1, clears=0, n=100, official=0.0)
    _run(tmp_path, "shape01", "cold", "cd82", 0, clears=99, n=100, official=0.0476)
    runs = load_runs(tmp_path)
    assert set(runs) == {"paired", "shape01"}
    assert len(runs["paired"]["cd82"]) == 2
    assert runs["shape01"]["cd82"][0]["clear_rate"] == pytest.approx(0.99)


def test_load_runs_skips_unfinished_unscored_and_wrong_arm(tmp_path):
    _run(tmp_path, "paired", "cold", "vc33", 0, clears=8, n=100, official=0.017)
    _run(tmp_path, "paired", "cold", "vc33", 1, clears=0, n=100, official=0.0, done=False)
    _run(tmp_path, "paired", "cold", "vc33", 2, clears=0, n=100, official=0.0, score=False)
    _run(tmp_path, "paired", "warm", "vc33", 3, clears=8, n=100, official=0.02)
    runs = load_runs(tmp_path, arm="cold")
    assert [c["seed"] for c in runs["paired"]["vc33"]] == [0]


def test_load_runs_ignores_tools_and_logs_and_malformed_names(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "paired" / "not-a-run").mkdir(parents=True)
    _run(tmp_path, "paired", "cold", "cd82", 0, clears=0, n=100, official=0.0)
    assert set(load_runs(tmp_path)) == {"paired"}


def test_load_runs_on_missing_root(tmp_path):
    assert load_runs(tmp_path / "nope") == {}


# --- rows -----------------------------------------------------------------

def test_ladder_rows_pool_episodes_across_seeds(tmp_path):
    for seed, clears in enumerate((99, 89, 100)):
        _run(tmp_path, "shape01", "cold", "cd82", seed, clears=clears, n=100, official=0.0476)
    (row,) = ladder_rows(load_runs(tmp_path))
    assert row["n_seeds"] == 3 and row["seeds_clearing"] == 3
    assert row["clear_rate"] == pytest.approx(288 / 300)
    assert row["wilson_lo"] > 0.93 and row["wilson_hi"] < 1.0
    assert row["official"] == pytest.approx(0.0476)
    assert row["ingredient"] == ARMS["shape01"][1]


def test_ladder_rows_count_seeds_that_never_clear(tmp_path):
    for seed in range(3):
        _run(tmp_path, "shape01", "cold", "sb26", seed, clears=0, n=100, official=0.0)
    (row,) = ladder_rows(load_runs(tmp_path))
    assert row["seeds_clearing"] == 0 and row["clear_rate"] == 0.0
    assert row["wilson_lo"] == 0.0 and row["wilson_hi"] < 0.02


def test_ladder_rows_filter_by_game(tmp_path):
    _run(tmp_path, "shape01", "cold", "cd82", 0, clears=99, n=100, official=0.05)
    _run(tmp_path, "shape01", "cold", "sb26", 0, clears=0, n=100, official=0.0)
    rows = ladder_rows(load_runs(tmp_path), games=["cd82"])
    assert [r["game"] for r in rows] == ["cd82"]


def test_ladder_rows_skip_unknown_batches(tmp_path):
    _run(tmp_path, "experiment-x", "cold", "cd82", 0, clears=1, n=100, official=0.01)
    assert ladder_rows(load_runs(tmp_path)) == []


# --- LaTeX ----------------------------------------------------------------

def test_ladder_tex_is_wellformed_and_labels_once_per_arm(tmp_path):
    for seed in range(2):
        _run(tmp_path, "shape01", "cold", "cd82", seed, clears=99, n=100, official=0.0476)
        _run(tmp_path, "shape01", "cold", "sb26", seed, clears=0, n=100, official=0.0)
    tex = fmt_ladder_tex(ladder_rows(load_runs(tmp_path)))
    assert tex.startswith(r"\begin{tabular}") and tex.rstrip().endswith(r"\end{tabular}")
    assert tex.count(r"\toprule") == 1 and tex.count(r"\bottomrule") == 1
    assert tex.count("State-change bonus") == 1, "arm label prints once, not per row"
    assert r"\texttt{cd82}" in tex and r"\texttt{sb26}" in tex
    body = [l for l in tex.splitlines() if l.endswith(r"\\") and "Intervention" not in l]
    assert all(l.count("&") == 5 for l in body), "every row needs six columns"


def test_pergame_tex_marks_missing_cells(tmp_path):
    _run(tmp_path, "paired", "cold", "cd82", 0, clears=0, n=100, official=0.0)
    _run(tmp_path, "shape01", "cold", "cd82", 0, clears=99, n=100, official=0.0476)
    tex = fmt_pergame_tex(load_runs(tmp_path), ["cd82", "sb26"], ["paired", "shape01", "mixture"])
    assert "0.0476" in tex
    assert tex.count("--") >= 3, "sb26 row and the empty mixture column are dashes"
    body = [l for l in tex.splitlines() if l.endswith(r"\\") and "Game" not in l]
    assert all(l.count("&") == 3 for l in body)


def test_tables_are_deterministic(tmp_path):
    for seed in range(2):
        _run(tmp_path, "shape01", "cold", "cd82", seed, clears=50, n=100, official=0.02)
    runs = load_runs(tmp_path)
    assert fmt_ladder_tex(ladder_rows(runs)) == fmt_ladder_tex(ladder_rows(load_runs(tmp_path)))
