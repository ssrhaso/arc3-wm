"""Generate the paper's result tables directly from the run tree.

Several numbers in the workshop draft exist in no committed artifact, which is
how the reward-proximity row came to disagree with the file it was supposed to
summarize. This script closes that gap: every table it emits is a pure function
of ``<runs>/<batch>/<arm>-<game>-s<seed>/score_eval100.json``, so a table can be
regenerated and diffed rather than retyped.

Two tables:

* ``ladder``  the give-back ladder. One row per intervention, columns for the
  ingredient it restores, the games it was run on, how many seeds cleared a
  level at all, and the frozen clear rate with a Wilson interval.
* ``perame``  per-game official RHAE by arm, the headline results table.

Scores come from the frozen 100-episode evaluation on the native environment,
which is comparable across arms because no arm shapes the evaluation env. The
training-time episode score is deliberately not used: under a shaped reward it
counts bonus rather than level clears.

Usage::

    python scripts/build_ladder_table.py --runs ~/arc3-runs --table ladder
    python scripts/build_ladder_table.py --runs ~/arc3-runs --table pergame --tex out.tex
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Optional

# Batch name -> (row label, ingredient the intervention gives back).
ARMS: dict[str, tuple[str, str]] = {
    "paired": ("Stock DreamerV3", "nothing (baseline)"),
    "sweep": ("Stock DreamerV3", "nothing (baseline)"),
    "shape01": ("State-change bonus ($\\beta=0.1$)", "reward contact"),
    "shape001": ("State-change bonus ($\\beta=0.01$)", "reward contact"),
    "mixture": ("Type-balanced prior", "a prior over action types"),
    "ratio512": ("Update ratio 512", "more gradient steps"),
    "long5m": ("5M env steps", "more experience"),
    "ppo": ("PPO", "a different learner"),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 0) when n is 0."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def load_runs(root: Path, arm: str = "cold") -> dict[str, dict[str, list[dict]]]:
    """``{batch: {game: [cell, ...]}}`` for every finished run under ``root``."""
    out: dict[str, dict[str, list[dict]]] = {}
    if not root.is_dir():
        return out
    for batch_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("logs", "tools")):
        for run in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
            parts = run.name.split("-")
            if len(parts) != 3 or not parts[2].startswith("s"):
                continue
            run_arm, game, seed = parts[0], parts[1], int(parts[2][1:])
            if arm not in (None, "any") and run_arm != arm:
                continue
            score = run / "score_eval100.json"
            if not (run / "DONE").exists() or not score.exists():
                continue
            try:
                s = json.loads(score.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "official_max" not in s:
                continue
            out.setdefault(batch_dir.name, {}).setdefault(game, []).append(
                {"seed": seed, "clear_rate": s.get("clear_rate", 0.0),
                 "official": s["official_max"], "n": s.get("n", 0),
                 "clears": s.get("clears", 0)}
            )
    return out


def ladder_rows(runs: Mapping[str, Mapping[str, list[dict]]],
                games: Optional[Iterable[str]] = None) -> list[dict]:
    """One row per (batch, game): seeds that cleared, pooled clear rate, mean RHAE."""
    rows = []
    for batch, per_game in runs.items():
        if batch not in ARMS:
            continue
        label, ingredient = ARMS[batch]
        for game, cells in sorted(per_game.items()):
            if games is not None and game not in games:
                continue
            n_seeds = len(cells)
            seeds_clearing = sum(1 for c in cells if (c["clear_rate"] or 0) > 0)
            total_clears = sum(int(c["clears"] or 0) for c in cells)
            total_eps = sum(int(c["n"] or 0) for c in cells)
            lo, hi = wilson(total_clears, total_eps)
            rows.append({
                "batch": batch, "label": label, "ingredient": ingredient, "game": game,
                "n_seeds": n_seeds, "seeds_clearing": seeds_clearing,
                "clear_rate": (total_clears / total_eps) if total_eps else 0.0,
                "wilson_lo": lo, "wilson_hi": hi,
                "official": statistics.fmean(c["official"] for c in cells),
            })
    return rows


def fmt_ladder_tex(rows: list[dict]) -> str:
    """LaTeX booktabs table: the give-back ladder."""
    out = [
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Intervention & Restores & Game & Seeds clearing & Clear rate (95\% CI) & Official RHAE \\",
        r"\midrule",
    ]
    last = None
    for r in sorted(rows, key=lambda r: (r["label"], r["game"])):
        label = "" if r["label"] == last else r["label"]
        ingredient = "" if r["label"] == last else r["ingredient"]
        last = r["label"]
        if label and out[-1] != r"\midrule":
            out.append(r"\addlinespace[2pt]")
        out.append(
            f"{label} & {ingredient} & \\texttt{{{r['game']}}} & "
            f"{r['seeds_clearing']}/{r['n_seeds']} & "
            f"{r['clear_rate']:.2f} [{r['wilson_lo']:.2f}, {r['wilson_hi']:.2f}] & "
            f"{r['official']:.4f} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def fmt_pergame_tex(runs: Mapping[str, Mapping[str, list[dict]]], games: list[str],
                    batches: list[str]) -> str:
    """LaTeX table: per-game official RHAE, one column per arm."""
    heads = " & ".join(ARMS.get(b, (b, ""))[0] for b in batches)
    out = [
        r"\begin{tabular}{l" + "c" * len(batches) + "}", r"\toprule",
        f"Game & {heads} \\\\", r"\midrule",
    ]
    for g in games:
        cells = []
        for b in batches:
            v = runs.get(b, {}).get(g)
            cells.append(f"{statistics.fmean(c['official'] for c in v):.4f}" if v else "--")
        out.append(f"\\texttt{{{g}}} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path.home() / "arc3-runs")
    ap.add_argument("--table", choices=["ladder", "pergame"], default="ladder")
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--batches", nargs="*", default=["paired", "shape01", "mixture", "ratio512"])
    ap.add_argument("--tex", type=Path, default=None, help="write LaTeX here instead of stdout")
    args = ap.parse_args(argv)

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no finished runs with scores under {args.runs}")
    if args.table == "ladder":
        text = fmt_ladder_tex(ladder_rows(runs, args.games))
    else:
        games = args.games or sorted({g for per in runs.values() for g in per})
        text = fmt_pergame_tex(runs, games, args.batches)
    if args.tex:
        args.tex.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.tex}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
