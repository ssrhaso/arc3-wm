"""Official ARC-AGI-3 scoring, reproduced from the toolkit's own calculator.

``arc3_wm.rhae`` implements the corpus variant used through Phase 4: human
baselines derived from the replay corpus, levels without two completer
sessions dropped from both numerator and denominator (D-B), and no
game-level cap. That variant is defensible on its own terms but its numbers
cannot be placed beside any published ARC-AGI-3 result, because the toolkit
scores three things differently:

1. **Baselines** come from the environment's shipped
   ``environment_files/<game>/<version>/metadata.json`` ``baseline_actions``
   vector, not from the replay corpus. The two disagree by up to 2.75x per
   level (vc33 level 1 is 7 officially and 13 in the corpus fixture).
2. **Every level counts** in the denominator, covered or not.
3. **A game-level cap** applies: the score cannot exceed the share of total
   level weight that was actually cleared. A level-1-only clear on a
   seven-level game is capped at 1/28, whatever the efficiency.

This module mirrors ``arc_agi.scorecard.EnvironmentScoreCalculator`` and
``EnvironmentScoreList`` so the paper can report the benchmark's own metric.
It is a pure function of a reward stream and a baseline vector: no engine
state, no ``Arcade``. ``tests/test_official_rhae.py`` pins it against the
toolkit class itself, so a change upstream trips a test rather than silently
moving the numbers.

Units: the toolkit works in 0..100. Every function here returns toolkit
units; ``compute_official_rhae`` also reports the 0..1 value the paper uses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "LEVEL_SCORE_CAP_PCT",
    "EpisodeScore",
    "official_level_score",
    "official_environment_score",
    "score_episode",
    "score_episodes",
    "load_official_baselines",
    "compute_official_rhae",
]

LEVEL_SCORE_CAP_PCT = 115.0
"""Per-level cap in toolkit units (1.15x baseline), from scorecard.py."""


@dataclass(frozen=True)
class EpisodeScore:
    """One episode ("run") scored the toolkit's way."""

    score: float
    """Game score in toolkit units, 0..100, already game-capped."""
    levels_completed: int
    actions: int
    level_scores: tuple[float, ...]
    """Per-level scores in toolkit units, one entry per baseline level."""

    @property
    def score_unit(self) -> float:
        """Score on the paper's 0..1 scale."""
        return self.score / 100.0


def official_level_score(
    baseline_actions: int, actions_taken: int, completed: bool
) -> float:
    """Per-level score in toolkit units.

    Mirrors ``EnvironmentScoreCalculator.add_level``: an uncompleted level
    scores 0 whatever it cost, and a completed level scores
    ``(baseline / actions)^2 * 100`` capped at 115.
    """
    if not completed:
        return 0.0
    if actions_taken <= 0:
        return 0.0
    return min((baseline_actions / actions_taken) ** 2 * 100.0, LEVEL_SCORE_CAP_PCT)


def official_environment_score(level_scores: Sequence[float]) -> float:
    """Level-index-weighted mean with the toolkit's game cap, in toolkit units.

    Mirrors ``EnvironmentScoreCalculator.to_score``. ``level_scores`` is
    ordered by level, index 0 being level 1. The cap is the fraction of total
    level weight that scored above zero, so a run that clears only early
    levels cannot approach 100 however efficient it was.
    """
    if not level_scores:
        return 0.0
    total = 0.0
    total_weights = 0
    max_weights = 0
    for i, s in enumerate(level_scores):
        weight = i + 1
        total += s * weight
        total_weights += weight
        if s > 0:
            max_weights += weight
    if total_weights == 0:
        return 0.0
    return min(total / total_weights, max_weights / total_weights * 100.0)


def _levels_cleared(rewards: Sequence[float]) -> dict[int, int]:
    """``{1-indexed level: actions spent on it}`` for cleared levels only.

    Same convention as ``scripts.compute_rhae.segment_episode_actions_per_level``:
    ``rewards[0]`` is the reset step and ``rewards[1:]`` are post-action
    rewards, with ``+1`` at a level clear.
    """
    if len(rewards) <= 1:
        return {}
    counts: dict[int, int] = {}
    cum = 0
    for r in rewards[1:]:
        r_int = int(r)
        if r_int < 0:
            raise ValueError(f"negative reward in stream: {r!r}")
        counts[cum + 1] = counts.get(cum + 1, 0) + 1
        cum += r_int
    return {k: v for k, v in counts.items() if k <= cum}


def score_episode(rewards: Sequence[float], baselines: Sequence[int]) -> EpisodeScore:
    """Score one episode's reward stream against a baseline vector.

    Uncleared levels are charged the episode's remaining actions, exactly as
    the toolkit charges them from the scorecard, which does not change the
    score (they contribute 0) but does make ``actions`` match the toolkit.
    """
    if not baselines:
        raise ValueError("baselines must be non-empty")
    cleared = _levels_cleared(rewards)
    total_actions = max(len(rewards) - 1, 0)
    spent = sum(cleared.values())
    scores: list[float] = []
    for idx, baseline in enumerate(baselines, start=1):
        if idx in cleared:
            scores.append(official_level_score(baseline, cleared[idx], True))
        else:
            scores.append(0.0)
            spent = total_actions
    return EpisodeScore(
        score=official_environment_score(scores),
        levels_completed=len(cleared),
        actions=total_actions,
        level_scores=tuple(scores),
    )


def score_episodes(
    episodes_rewards: Iterable[Sequence[float]], baselines: Sequence[int]
) -> dict:
    """Score a set of episodes for one game the toolkit's way.

    The toolkit takes the **maximum** over plays of the same environment
    (``EnvironmentScoreList.score``), so the reported number is best-of-N and
    grows with N. ``n_episodes`` is returned so a table can state it.
    """
    runs = [score_episode(r, baselines) for r in episodes_rewards]
    if not runs:
        return {
            "score": 0.0,
            "score_unit": 0.0,
            "mean_score_unit": 0.0,
            "levels_completed": 0,
            "n_episodes": 0,
            "n_cleared_episodes": 0,
        }
    best = max(r.score for r in runs)
    return {
        "score": best,
        "score_unit": best / 100.0,
        "mean_score_unit": sum(r.score for r in runs) / len(runs) / 100.0,
        "levels_completed": max(r.levels_completed for r in runs),
        "n_episodes": len(runs),
        "n_cleared_episodes": sum(1 for r in runs if r.levels_completed > 0),
    }


def load_official_baselines(
    game_id: str, environment_files: Path | str = "environment_files"
) -> list[int]:
    """Read ``baseline_actions`` from a cached game's ``metadata.json``.

    Raises ``FileNotFoundError`` when the game is not cached, so a missing
    cache surfaces as a clear error instead of a silently wrong metric.
    """
    root = Path(environment_files) / game_id
    metas = sorted(root.glob("*/metadata.json"))
    if not metas:
        raise FileNotFoundError(
            f"no metadata.json under {root}; run scripts/cache_env_files.py {game_id}"
        )
    data = json.loads(metas[-1].read_text(encoding="utf-8"))
    baselines = data.get("baseline_actions")
    if not baselines:
        raise ValueError(f"{metas[-1]} has no baseline_actions")
    return [int(b) for b in baselines]


def compute_official_rhae(
    *,
    episodes_rewards: Iterable[Sequence[float]],
    game_id: str,
    baselines: Optional[Sequence[int]] = None,
    environment_files: Path | str = "environment_files",
) -> Mapping[str, float]:
    """End-to-end official score for one game, in the metric-key family.

    Mirrors ``RHAEAggregator``'s key shape so both metrics can be logged side
    by side: ``eval/rhae_official/per_game/<game>`` is the 0..1 value.
    """
    if baselines is None:
        baselines = load_official_baselines(game_id, environment_files)
    res = score_episodes(episodes_rewards, baselines)
    return {
        f"eval/rhae_official/per_game/{game_id}": res["score_unit"],
        f"eval/rhae_official/per_game_mean/{game_id}": res["mean_score_unit"],
        f"eval/rhae_official/levels_completed/{game_id}": float(res["levels_completed"]),
        f"eval/rhae_official/n_episodes/{game_id}": float(res["n_episodes"]),
        f"eval/rhae_official/n_cleared_episodes/{game_id}": float(res["n_cleared_episodes"]),
        f"eval/rhae_official/total_levels/{game_id}": float(len(baselines)),
    }
