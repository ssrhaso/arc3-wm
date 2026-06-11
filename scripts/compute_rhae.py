"""Phase-4 post-hoc RHAE CLI.

Reads a JSONL of per-eval-episode reward streams, segments each episode
by level via the cumulative-reward signal from ``ARC3GymEnv.step``
(``r = delta levels_completed``), takes MIN action count per level across
eval episodes (matching ``extract_human_baselines.extract_per_session_baselines``
so agent and human are scored under the same "best attempt" framing),
and feeds the result to ``arc3_wm.rhae.RHAEAggregator`` against
``data/human_baselines.json``.

Emits the three-key family from Notion "Logging & analysis plan":

  eval/rhae/per_game/{game_id}            float
  eval/rhae/level_scores/{game_id}/{i}    float (1-indexed level i)
  eval/rhae/levels_completed/{game_id}    int

Plus a one-line stdout summary per the task brief:

  "vc33 @ 500k env steps: levels_completed=2, per_game_rhae=0.42"

Usage::

    python scripts/compute_rhae.py \\
        --episodes-file path/to/eval_episodes.jsonl \\
        --game-id vc33 \\
        --baselines data/human_baselines.json \\
        --step 500000

Reward-stream source: DV3's eval logfn pops the per-step rewards stack
before adding to epstats, so the stack is NOT in ``eval/episode/rewards``
on wandb. The episodes JSONL this CLI consumes is produced by
``arc3_wm.eval_reward_sink.EvalRewardSink``, the eval-env wrapper wired
into ``scripts/launch_pergame.py`` (D-C): it buffers per-step rewards and
flushes one ``{"rewards": [...]}`` line per episode. The CLI's primary
testable interface is that file mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

# Add repo root so ``from arc3_wm.rhae import RHAEAggregator`` resolves
# when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc3_wm.rhae import RHAEAggregator  # noqa: E402


def segment_episode_actions_per_level(
    rewards: List[float],
) -> dict[int, int]:
    """Single eval episode's reward stream -> ``{1-indexed_level: action_count}``
    for CLEARED levels only.

    DV3 convention (see ``embodied/run/train_eval.py:48-51``):
    ``rewards[0]`` is the reward at the initial-obs step (always 0; no
    action has been taken yet). ``rewards[1..N-1]`` are post-action
    rewards. ``r in {0, +1}`` per ``arc3_wm/env.py:113``; +1 fires at the
    post-level-up obs.

    Algorithm: walk ``rewards[1:]`` with a running ``cum_reward``
    counter. For each step, the action was taken on level
    ``cum_reward + 1`` BEFORE this step's reward fires; count that
    level, then update ``cum_reward``. After the walk, ``cum_reward``
    equals total levels cleared; filter counts to keep only completed
    levels (drops partial counts for the level the agent died on).
    """
    if len(rewards) <= 1:
        return {}
    counts: dict[int, int] = {}
    cum_reward = 0
    for r in rewards[1:]:
        r_int = int(r)
        if r_int < 0:
            raise ValueError(
                f"non-binary or negative reward in stream: {r!r} "
                f"(expected r in {{0, +1}} per arc3_wm/env.py:113)"
            )
        level = cum_reward + 1
        counts[level] = counts.get(level, 0) + 1
        cum_reward += r_int
    completed_max = cum_reward
    return {k: v for k, v in counts.items() if k <= completed_max}


def aggregate_eval_episodes(
    episodes_rewards: Iterable[List[float]],
) -> dict[int, int]:
    """Multiple eval episodes for one game -> MIN action count per cleared
    level. A level cleared by at least one episode contributes its
    minimum action count; levels no episode cleared are absent.
    """
    per_level: dict[int, int] = {}
    for ep in episodes_rewards:
        ep_counts = segment_episode_actions_per_level(list(ep))
        for level, count in ep_counts.items():
            existing = per_level.get(level)
            if existing is None or count < existing:
                per_level[level] = count
    return per_level


def compute_rhae(
    *,
    episodes_rewards: Iterable[List[float]],
    game_id: str,
    baselines: Mapping[str, Mapping],
) -> dict:
    """End-to-end: episodes + game_id + baselines -> three-key metrics dict.

    ``baselines`` is the D-A/D-B fixture shape: per-game
    ``{"total_levels": int, "baselines": {level_idx: int}}``. Level keys
    may be int or string (the fixture file uses strings after
    ``json.loads``); RHAEAggregator's constructor coerces internally.
    """
    aggregator = RHAEAggregator(human_baselines=baselines)
    ai_actions_per_level = aggregate_eval_episodes(episodes_rewards)
    return aggregator(
        game_id=game_id, ai_actions_per_level=ai_actions_per_level
    )


def load_episodes_from_jsonl(path: Path) -> List[List[float]]:
    """Parse a JSONL where each non-blank line is ``{"rewards": [...]}``.

    Missing file raises ``FileNotFoundError`` (the CLI surfaces verbatim).
    A row without a ``rewards`` key raises ``ValueError`` - malformed
    eval log, surface rather than silently emit an empty episode.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"episodes file not found: {path}")
    episodes: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if "rewards" not in obj:
                raise ValueError(
                    f"{path}:line {line_no}: row missing 'rewards' key "
                    f"(got keys {sorted(obj)})"
                )
            episodes.append(list(obj["rewards"]))
    return episodes


def _format_step(step: Optional[int]) -> str:
    if step is None:
        return ""
    if step >= 1_000_000 and step % 1_000_000 == 0:
        return f"{step // 1_000_000}M"
    if step >= 1_000 and step % 1_000 == 0:
        return f"{step // 1_000}k"
    return str(step)


def format_summary(
    *, game_id: str, step: Optional[int], metrics: Mapping
) -> str:
    """One-liner per task brief: ``vc33 @ 500k env steps: levels_completed=2,
    per_game_rhae=0.42``. When ``step`` is None, the ``@ ...`` clause is
    omitted."""
    per_game = float(metrics.get(f"eval/rhae/per_game/{game_id}", 0.0))
    levels = int(metrics.get(f"eval/rhae/levels_completed/{game_id}", 0))
    step_fragment = _format_step(step)
    prefix = f"{game_id} @ {step_fragment} env steps" if step_fragment else game_id
    return (
        f"{prefix}: levels_completed={levels}, "
        f"per_game_rhae={per_game:.2f}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-4 post-hoc RHAE: episodes JSONL + per-game baselines "
            "-> eval/rhae/* metrics + one-line summary."
        )
    )
    parser.add_argument(
        "--episodes-file",
        type=Path,
        required=True,
        help="JSONL of eval-episode reward streams ({\"rewards\": [...]}).",
    )
    parser.add_argument(
        "--game-id", required=True, help="ARC-AGI-3 game id (e.g. vc33)."
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("data/human_baselines.json"),
        help="Per-game human baselines JSON (from extract_human_baselines).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step count for the summary line (e.g. 500000).",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Also print the full three-key family to stdout.",
    )
    args = parser.parse_args(argv)

    episodes = load_episodes_from_jsonl(args.episodes_file)
    if not args.baselines.exists():
        raise FileNotFoundError(f"baselines file not found: {args.baselines}")
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))

    metrics = compute_rhae(
        episodes_rewards=episodes,
        game_id=args.game_id,
        baselines=baselines,
    )
    if args.print_metrics:
        for k in sorted(metrics):
            print(f"{k}: {metrics[k]}")
    print(format_summary(game_id=args.game_id, step=args.step, metrics=metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
