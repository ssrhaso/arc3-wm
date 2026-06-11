"""Reference implementation of Relative Human Action Efficiency (RHAE).

Source of truth at training time is ``arc.get_scorecard().score`` - the
toolkit computes RHAE from the scorecard the engine maintains internally
(per Phase-0 verification, ``docs/phase-checklists.md``). This module
exists per D6 for two reasons:

1. **Per-checkpoint logging** during Phase 4-5 without re-instantiating
   ``Arcade`` (cheap pure functions, no engine state).
2. **Sanity fixture** so any silent change to the toolkit's scoring
   trips a test.

Spec: ``docs/arc-agi-3/methodology.md``. Formula:

    level_score = min((human / ai)^2, 1.15)
    game_score  = sum(level_score_i * i, completed) / sum(i, scoreable levels)
    total_score = mean(game_scores)

Per D-B (n>=2 coverage threshold), only levels with a defensible
upper-median baseline (>= 2 completer sessions) are scoreable
- uncovered levels are excluded from BOTH the per-game numerator and
denominator. The "failing the final scored level caps the game score"
property is preserved for covered levels: their weight stays in the
denominator while their numerator term goes to zero.

Per D-A, per-game ``total_levels`` is the engine's ``win_levels`` field
(the full level count, INCLUDING uncovered ones).
It bounds the valid level-index range; the per-game denominator uses
only the covered subset.

Fixture shape (per ``scripts.extract_human_baselines``)::

    {
      "vc33": {
        "total_levels": 7,                       # D-A: from JSONL win_levels
        "baselines": {"1": 13, "2": 47, ...}    # D-B: only n>=2 levels
      },
      ...
    }
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

__all__ = [
    "LEVEL_SCORE_CAP",
    "level_score",
    "game_score",
    "total_score",
    "coverage",
    "RHAEAggregator",
]

LEVEL_SCORE_CAP = 1.15


def level_score(human_baseline_actions: int, ai_actions: int) -> float:
    """Per-level efficiency, capped at ``LEVEL_SCORE_CAP`` (1.15x baseline)."""
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
    covered_levels: Iterable[int],
) -> float:
    """Weighted-mean per-game score; weights are 1-indexed level numbers.

    Per D-B, ``covered_levels`` is the set of level indices that
    contribute to BOTH numerator and denominator. ``level_scores`` keys
    must be a subset of ``covered_levels`` - uncovered AI completions
    must be skipped by the caller (RHAEAggregator does this). Uncompleted
    covered levels are absent from ``level_scores`` and contribute 0 to
    the numerator; their weight stays in the denominator, which is what
    preserves the "failing the final scored level caps the game score"
    property under D-B.

    Returns 0.0 when ``covered_levels`` is empty (all-uncovered game -
    no real fixture data triggers this but pinned for safety).
    """
    covered = set(covered_levels)
    if not covered:
        return 0.0
    for k in covered:
        if k < 1:
            raise ValueError(
                f"covered_levels must be 1-indexed positive ints; got {k}"
            )
    scores = dict(level_scores) if level_scores else {}
    for k in scores:
        if k not in covered:
            raise ValueError(
                f"level {k} not in covered_levels {sorted(covered)}"
            )
    if not scores:
        return 0.0
    numerator = sum(score * level_idx for level_idx, score in scores.items())
    denominator = sum(covered)
    return numerator / denominator


def total_score(game_scores: Iterable[float]) -> float:
    """Arithmetic mean across games. Empty iterable -> 0.0."""
    scores = list(game_scores)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def coverage(human_baselines: Mapping[str, Mapping]) -> float:
    """Global RHAE coverage = covered levels / total levels across all games.

    Per D-B, "covered" = has a defensible upper-median baseline (n>=2
    completers). "Total" = engine ``win_levels`` per D-A. This is the
    number reported in the paper alongside headline RHAE (currently
    129/183 = 0.70 on the 340-replay public-demo dataset).

    Empty input or zero total returns 0.0 - avoids ZeroDivisionError on
    a sentinel-empty fixture.
    """
    n_covered = 0
    n_total = 0
    for entry in human_baselines.values():
        n_covered += len(entry["baselines"])
        n_total += int(entry["total_levels"])
    if n_total == 0:
        return 0.0
    return n_covered / n_total


class RHAEAggregator:
    """Post-hoc RHAE aggregator: per-level AI action counts -> wandb-key metrics.

    Per D2 the Phase-4 pipeline computes RHAE post-hoc from the
    ``eval/episode/*`` series ``embodied.run.train_eval`` already emits;
    this class is the pure-math step the post-hoc CLI calls once it has
    segmented an eval rollout into per-level AI action counts. No
    scheduling, no in-loop hook semantics - the caller decides when to
    call.

    Contract (post D-A/D-B migration):

    - ``human_baselines``: per-game entry ``{"total_levels": int,
      "baselines": {level_idx (int or str): action_count}}``. Level keys
      are coerced to int at construction time so the fixture file
      (string keys after ``json.loads``) and synthetic test fixtures
      (int keys) are both accepted. Baselines reflect the D-B coverage
      threshold (only levels with n>=2 completer sessions); D-A
      ``total_levels`` is the engine ``win_levels`` per
      ``docs/replay-format.md``.

    - ``__call__(game_id, ai_actions_per_level)``: returns the three-key
      family ``eval/rhae/{per_game,level_scores,levels_completed}/...``
      spec'd in Notion "Logging & analysis plan".

    Semantics:

    - AI completion of a level in [1, total_levels] but NOT in
      ``baselines`` (uncovered under D-B): silently skipped - no
      ``level_scores`` atom emitted, no contribution to the per-game
      numerator/denominator. The agent gets neither credit nor penalty.
    - AI completion of a level OUTSIDE [1, total_levels]: raises
      ``ValueError`` (level-indexing bug upstream - surface).
    - Unknown ``game_id``: raises ``KeyError`` (typo or missing-data
      drift).
    - Zero-completion run on a game with covered levels: per_game=0.0,
      levels_completed=0, no level_scores atoms.
    - All-uncovered game (``baselines={}``): per_game=0.0,
      levels_completed=0, no level_scores atoms regardless of AI
      completions in range. No real fixture data triggers this but
      pinned for safety.
    """

    def __init__(
        self,
        *,
        human_baselines: Mapping[str, Mapping],
    ) -> None:
        normalized: dict[str, dict] = {}
        for game_id, entry in human_baselines.items():
            if "total_levels" not in entry or "baselines" not in entry:
                raise ValueError(
                    f"human_baselines[{game_id!r}] must have keys "
                    f"'total_levels' and 'baselines'; got {sorted(entry)}"
                )
            total_levels = int(entry["total_levels"])
            if total_levels < 1:
                raise ValueError(
                    f"human_baselines[{game_id!r}].total_levels must be >= 1; "
                    f"got {total_levels}"
                )
            baselines_int = {int(k): int(v) for k, v in entry["baselines"].items()}
            for level_idx in baselines_int:
                if not 1 <= level_idx <= total_levels:
                    raise ValueError(
                        f"human_baselines[{game_id!r}] level {level_idx} "
                        f"out of [1, {total_levels}]"
                    )
            normalized[game_id] = {
                "total_levels": total_levels,
                "baselines": baselines_int,
            }
        self.human_baselines = normalized

    def __call__(
        self,
        *,
        game_id: str,
        ai_actions_per_level: Mapping[int, int],
    ) -> dict:
        if game_id not in self.human_baselines:
            raise KeyError(
                f"no human baseline for game_id {game_id!r}; "
                f"known games: {sorted(self.human_baselines)}"
            )
        entry = self.human_baselines[game_id]
        total_levels: int = entry["total_levels"]
        game_baselines: dict[int, int] = entry["baselines"]
        covered = set(game_baselines)

        metrics: dict = {}
        level_scores: dict[int, float] = {}
        for level_idx, ai_actions in ai_actions_per_level.items():
            level_idx = int(level_idx)
            if not 1 <= level_idx <= total_levels:
                raise ValueError(
                    f"level {level_idx} out of [1, {total_levels}] for "
                    f"game_id {game_id!r}"
                )
            if level_idx not in covered:
                # D-B: AI cleared an uncovered level - no baseline,
                # silently skip (no credit, no penalty).
                continue
            ls = level_score(
                human_baseline_actions=game_baselines[level_idx],
                ai_actions=ai_actions,
            )
            level_scores[level_idx] = ls
            metrics[f"eval/rhae/level_scores/{game_id}/{level_idx}"] = ls

        metrics[f"eval/rhae/per_game/{game_id}"] = game_score(
            level_scores, covered_levels=covered
        )
        metrics[f"eval/rhae/levels_completed/{game_id}"] = len(level_scores)
        return metrics
