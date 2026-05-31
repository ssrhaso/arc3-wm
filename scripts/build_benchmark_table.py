"""Build the 6-game RHAE benchmark table (the paper's headline result).

Scans a logdir for Phase-4 run directories, computes post-hoc RHAE per
run via the canonical ``scripts/compute_rhae.py`` helpers, and emits a
paired warm-vs-cold table over the 6 benchmark games.

RHAE is **independently level-weighted per game** (each game's score is
a 1-indexed-level-weighted mean - see ``arc3_wm.rhae.game_score``) and
then combined across games by an unweighted mean
(``arc3_wm.rhae.total_score``). Global human-baseline coverage
(``arc3_wm.rhae.coverage``) is reported alongside, per the D-A/D-B
methodology.

This is **staged plumbing**: it degrades gracefully before results
land. Run dirs without ``eval_episodes.jsonl`` are reported as PENDING;
with zero runs found it still emits the template (6 games + baseline
coverage) so the artifact is meaningful pre-results.

Run-name grammar (matches scripts/launch_phase4_*.sh):
    warm : p4-<game>-s<seed>-warm-<sha>
    cold : p4-fromscratch-<game>-s<seed>-<sha>

Usage:
    python scripts/build_benchmark_table.py
    python scripts/build_benchmark_table.py --logdir _p4_analysis/logdir --step 500000
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc3_wm.rhae import coverage, total_score  # noqa: E402

# The 6 benchmark games: pilot trio + Phase-4 expansion trio. Fixed so
# the template is populated even with zero discovered runs.
BENCHMARK_GAMES = ["vc33", "sb26", "cd82", "tn36", "ls20", "lf52"]

DEFAULT_LOGDIR = REPO_ROOT / "_p4_analysis" / "logdir"
DEFAULT_BASELINES = REPO_ROOT / "data" / "human_baselines.json"
DEFAULT_OUT_JSON = REPO_ROOT / "analysis" / "benchmark_table.json"
DEFAULT_OUT_MD = REPO_ROOT / "analysis" / "benchmark_table.md"

WARM_RE = re.compile(r"^p4-(?P<game>[a-z0-9]+)-s(?P<seed>\d+)-warm-[0-9a-f]+$")
COLD_RE = re.compile(r"^p4-fromscratch-(?P<game>[a-z0-9]+)-s(?P<seed>\d+)-[0-9a-f]+$")


def discover_runs(logdir: Path) -> list[dict]:
    """Return [{arm, game, seed, dir, episodes_file|None}] for Phase-4 runs."""
    runs: list[dict] = []
    if not logdir.is_dir():
        return runs
    for child in sorted(logdir.iterdir()):
        if not child.is_dir():
            continue
        for arm, rx in (("warm", WARM_RE), ("cold", COLD_RE)):
            m = rx.match(child.name)
            if not m:
                continue
            ep = child / "eval_episodes.jsonl"
            runs.append(
                {
                    "arm": arm,
                    "game": m.group("game"),
                    "seed": int(m.group("seed")),
                    "dir": child.name,
                    "episodes_file": ep if ep.is_file() else None,
                }
            )
            break
    return runs


def score_runs(runs: list[dict], baselines: dict) -> list[dict]:
    """Attach per-run RHAE. Lazy-imports compute_rhae (no JAX needed)."""
    from compute_rhae import compute_rhae, load_episodes_from_jsonl

    for r in runs:
        if r["episodes_file"] is None:
            r["status"], r["rhae"], r["levels"] = "PENDING", None, None
            continue
        try:
            episodes = load_episodes_from_jsonl(r["episodes_file"])
            metrics = compute_rhae(
                episodes_rewards=episodes,
                game_id=r["game"],
                baselines=baselines,
            )
            r["rhae"] = float(metrics[f"eval/rhae/per_game/{r['game']}"])
            r["levels"] = int(
                metrics[f"eval/rhae/levels_completed/{r['game']}"]
            )
            r["status"] = "OK"
        except Exception as e:  # surface, don't crash the whole table
            r["status"], r["rhae"], r["levels"] = f"ERROR: {e}", None, None
    return runs


def _mean(xs: list[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def build_table(runs: list[dict]) -> dict:
    """Group runs into a {arm: {game: {seeds, mean_rhae}}} structure +
    per-arm combined total_score over games that have >=1 scored seed."""
    out: dict = {"per_arm": {}, "combined": {}}
    for arm in ("cold", "warm"):
        per_game: dict = {}
        for game in BENCHMARK_GAMES:
            seeds = sorted(
                (r for r in runs if r["arm"] == arm and r["game"] == game),
                key=lambda r: r["seed"],
            )
            scored = [r["rhae"] for r in seeds if r.get("rhae") is not None]
            per_game[game] = {
                "seeds": {
                    r["seed"]: {
                        "rhae": r["rhae"],
                        "levels": r["levels"],
                        "status": r["status"],
                    }
                    for r in seeds
                },
                "mean_rhae": _mean(scored),
                "n_scored_seeds": len(scored),
            }
        scored_games = [
            v["mean_rhae"] for v in per_game.values() if v["mean_rhae"] is not None
        ]
        out["per_arm"][arm] = per_game
        out["combined"][arm] = {
            "total_score": total_score(scored_games) if scored_games else None,
            "n_scored_games": len(scored_games),
            "n_games": len(BENCHMARK_GAMES),
        }
    return out


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.4f}"


def render_md(table: dict, cov: float, step: Optional[int]) -> str:
    step_s = f" @ {step:,} env steps" if step else ""
    lines = [
        f"# ARC-AGI-3 world-model benchmark - RHAE{step_s}",
        "",
        "Stock DreamerV3 (`size12m`), 6 public games x 2 seeds, paired "
        "cold (from-scratch) vs warm (cross-game WM-pretrained).",
        "",
        f"Human-baseline coverage (D-A/D-B): **{cov:.2%}**. RHAE is "
        "per-game level-index-weighted, then averaged across games.",
        "",
        "| Game | Cold (mean) | Warm (mean) | Cold seeds | Warm seeds |",
        "|---|---|---|---|---|",
    ]
    for game in BENCHMARK_GAMES:
        c = table["per_arm"]["cold"][game]
        w = table["per_arm"]["warm"][game]

        def seedcell(g: dict) -> str:
            if not g["seeds"]:
                return "PENDING"
            return ", ".join(
                f"s{s}={_fmt(v['rhae']) if v['rhae'] is not None else v['status']}"
                for s, v in sorted(g["seeds"].items())
            )

        lines.append(
            f"| {game} | {_fmt(c['mean_rhae'])} | {_fmt(w['mean_rhae'])} "
            f"| {seedcell(c)} | {seedcell(w)} |"
        )
    cc = table["combined"]["cold"]
    cw = table["combined"]["warm"]
    lines += [
        f"| **combined (mean of {cc['n_games']})** "
        f"| **{_fmt(cc['total_score'])}** | **{_fmt(cw['total_score'])}** "
        f"| {cc['n_scored_games']}/{cc['n_games']} games scored "
        f"| {cw['n_scored_games']}/{cw['n_games']} games scored |",
        "",
        "_- = no scored seed yet (run pending or no level cleared). "
        "Paired cold-vs-warm delta is the cross-game-pretraining test._",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    p.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    p.add_argument("--step", type=int, default=500_000)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = p.parse_args(argv)

    if not args.baselines.is_file():
        print(f"baselines fixture not found: {args.baselines}", file=sys.stderr)
        return 2
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    cov = coverage(baselines)

    runs = score_runs(discover_runs(args.logdir), baselines)
    table = build_table(runs)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {"step": args.step, "coverage": cov, "runs": runs, "table": table},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    md = render_md(table, cov, args.step)
    args.out_md.write_text(md, encoding="utf-8")

    n_ok = sum(1 for r in runs if r.get("status") == "OK")
    print(f"discovered {len(runs)} runs, {n_ok} scored; coverage={cov:.2%}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print()
    # Echo the table. The .md file is UTF-8; a non-UTF-8 console (e.g.
    # Windows cp1252) can't encode delta/- - degrade the echo, never crash.
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(md.encode(enc, errors="replace").decode(enc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
