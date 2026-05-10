"""Reference implementation of Relative Human Action Efficiency (RHAE).

Source of truth at training time is ``arc.get_scorecard().score`` — the
toolkit computes RHAE from the scorecard the engine maintains internally
(per Phase-0 verification, ``docs/phase-checklists.md``). This module
exists per D6 for two reasons:

1. **Per-checkpoint logging** during Phase 4–5 without re-instantiating
   ``Arcade`` (cheap pure functions, no engine state).
2. **Sanity fixture** so any silent change to the toolkit's scoring
   trips a test.

Spec: ``docs/arc-agi-3/methodology.md``. Formula:

    level_score = min((human / ai)^2, 1.15)
    game_score  = sum(level_score_i * i, completed) / sum(i, all levels)
    total_score = mean(game_scores)

The denominator includes uncompleted levels, so failing the final level
mechanically caps the game score (the largest-weight term drops from the
numerator). Per-game weights underweight tutorial levels and overweight
late-game mastery.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

LEVEL_SCORE_CAP = 1.15


def level_score(human_baseline_actions: int, ai_actions: int) -> float:
    """Per-level efficiency, capped at ``LEVEL_SCORE_CAP`` (1.15× baseline)."""
    if human_baseline_actions <= 0:
        raise ValueError(
            f"human_baseline_actions must be positive; got {human_baseline_actions}"
        )
    if ai_actions <= 0:
        raise ValueError(f"ai_actions must be positive; got {ai_actions}")
    raw = (human_baseline_actions / ai_actions) ** 2
    return min(raw, LEVEL_SCORE_CAP)


def game_score(
    level_scores: Optional[Mapping[int, float]],
    total_levels: int,
) -> float:
    """Weighted-mean per-game score; weights are 1-indexed level numbers.

    ``level_scores`` keys are 1-indexed level numbers (1..total_levels).
    Only completed levels appear; uncompleted levels are absent. The
    denominator includes ALL levels — missing level indices effectively
    score zero, which is what makes failing the final level so costly.
    """
    if total_levels < 1:
        raise ValueError(f"total_levels must be >= 1; got {total_levels}")
    scores = dict(level_scores) if level_scores else {}
    for k in scores:
        if not 1 <= k <= total_levels:
            raise ValueError(
                f"level {k} out of range [1, {total_levels}]"
            )
    if not scores:
        return 0.0
    numerator = sum(score * level_idx for level_idx, score in scores.items())
    denominator = total_levels * (total_levels + 1) // 2  # = sum(1..total_levels)
    return numerator / denominator


def total_score(game_scores: Iterable[float]) -> float:
    """Arithmetic mean across games. Empty iterable → 0.0."""
    scores = list(game_scores)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
