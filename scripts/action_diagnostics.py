"""Per-task action-diagnostic CLI.

Emits the structural / validity profile for one ARC-AGI-3 task (the 4102-way
flat action space) as JSON and/or CSV, plus a one-line rollup summary. The
empirical-usage (b) columns stay null unless a real per-step action log is
supplied via ``--action-log`` -- they are never synthesised.

Examples
--------
ls20 (directional-only; not cached as environment_files -> derive validity
from human replays) to JSON + CSV::

    python scripts/action_diagnostics.py --game-id ls20 --source replays \\
        --out-json results/ls20_actions.json --out-csv results/ls20_actions.csv

A cached game from the live engine, with measured usage from an instrumented
rollout's action log::

    python scripts/action_diagnostics.py --game-id vc33 --source env \\
        --action-log path/to/actions.jsonl --out-json results/vc33_actions.json

The action log is JSONL, one ``{"actions": [flat_idx, ...]}`` object per
episode (same shape family as ``arc3_wm.eval_reward_sink``'s reward log).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc3_wm.action_diagnostics import (  # noqa: E402
    LF52_BUDGET,
    UNIFORM_BUDGET,
    TaskActionProfile,
    from_env,
    from_replays,
    load_action_log_jsonl,
    usage_counts_from_action_indices,
)

_BUDGET_MODELS = {"auto": None, "uniform": UNIFORM_BUDGET, "lf52": LF52_BUDGET}


def build_profile(
    *,
    game_id: str,
    source: str,
    replays_dir: Path,
    action_log: Optional[Path],
    budget_model_name: str,
) -> TaskActionProfile:
    """Resolve the validity source, optional usage log, and budget model."""
    usage_counts = None
    if action_log is not None:
        usage_counts = usage_counts_from_action_indices(
            load_action_log_jsonl(action_log)
        )

    budget_model = _BUDGET_MODELS[budget_model_name]  # None -> per-game default
    kwargs = dict(budget_model=budget_model, usage_counts=usage_counts)

    if source == "env":
        return from_env(game_id, **kwargs)
    return from_replays(game_id, replays_dir=replays_dir, **kwargs)


def format_summary(profile: TaskActionProfile, *, usage_measured: bool) -> str:
    dil = profile.dilution_ratio
    dil_str = "n/a" if dil is None else f"{dil:.1f}:1"
    usage_str = "measured" if usage_measured else "null (no action log)"
    return (
        f"{profile.task_id}: n_valid={profile.n_valid} "
        f"n_state_changing={profile.n_state_changing} "
        f"dilution={dil_str} usage={usage_str}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per-task action diagnostic over the flat 4102-way action space: "
            "engine-derived validity/state-change/budget + honest-null usage."
        )
    )
    parser.add_argument("--game-id", required=True, help="ARC-AGI-3 game id (e.g. ls20).")
    parser.add_argument(
        "--source",
        choices=("env", "replays"),
        default="replays",
        help="Validity source: live engine (needs cached environment_files) "
        "or human replays (default; works for all 25 games).",
    )
    parser.add_argument(
        "--replays-dir",
        type=Path,
        default=Path("data/replays"),
        help="Replay root for --source replays (default: data/replays).",
    )
    parser.add_argument(
        "--action-log",
        type=Path,
        default=None,
        help='Optional per-step action JSONL ({"actions": [...]} per episode) '
        "to populate empirical usage. Omitted -> usage stays null.",
    )
    parser.add_argument(
        "--budget-model",
        choices=tuple(_BUDGET_MODELS),
        default="auto",
        help="Budget-cost model (default: auto -> per-game; lf52 -> its "
        "survival budget, else uniform +1).",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Write JSON here.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Write CSV here.")
    args = parser.parse_args(argv)

    profile = build_profile(
        game_id=args.game_id,
        source=args.source,
        replays_dir=args.replays_dir,
        action_log=args.action_log,
        budget_model_name=args.budget_model,
    )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        profile.to_json(args.out_json)
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(args.out_csv)

    print(format_summary(profile, usage_measured=args.action_log is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
