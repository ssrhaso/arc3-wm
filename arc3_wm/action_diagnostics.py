"""Per-task action-diagnostic profile for the flat 4102-way action space.

A task id maps to a per-action diagnostic: for every flat index in
``[0, 4102)`` it reports whether the action is valid on the task, whether it
changes state, its budget cost, and -- when (and only when) an instrumented
rollout supplies a per-step action log -- how often a policy actually fired it.

Two kinds of per-action data live here and are **never conflated**:

(a) **Structural / validity profile** -- static per task, ground-truth from the
    engine: ``valid_on_task``, ``is_state_changing``, ``budget_cost`` and the
    task-level rollups (``n_valid``, ``n_state_changing``, ``dilution_ratio``).
    The rollups are derived *from the rows*, never hardcoded.

(b) **Empirical usage counts** -- how often a trained policy fired each action.
    This data is not produced anywhere in the current pipeline (checkpoint
    replay buffers pickle to ``None``; only ``train/ent/action`` and
    ``train/rand/action`` aggregates are logged -- see the module-level
    findings in ``docs`` / the action-space audit). So ``usage_count`` and
    ``usage_fraction`` are **null** unless a real per-step action log is
    supplied via :func:`usage_counts_from_action_indices`. They are *never*
    synthesised or backed out of entropy / random-action scalars.

Each field carries a per-field source tag (see the ``SOURCE_*`` constants) so a
consumer can tell engine-derived ground truth from run-measured data from an
honest null.

The action-TYPE validity of a task comes from the engine's
``available_actions`` set (the integer action ids ``1..7`` the engine exposes
at every frame). ``build_task_action_profile`` takes that set directly;
:mod:`arc3_wm.action_diagnostics` also offers :func:`from_env` /
:func:`from_replays` sources that obtain it from a live game or the human
replays.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

from . import action_space as A

__all__ = [
    "SOURCE_ENGINE",
    "SOURCE_ENGINE_PROBE",
    "SOURCE_RUN_MEASURED",
    "SOURCE_ABSENT",
    "BudgetModel",
    "UNIFORM_BUDGET",
    "LF52_BUDGET",
    "GAME_BUDGET_MODELS",
    "resolve_budget_model",
    "ActionRow",
    "TaskActionProfile",
    "build_task_action_profile",
    "usage_counts_from_action_indices",
    "load_action_log_jsonl",
    "available_actions_from_replays",
    "from_env",
    "from_replays",
]

#: The seven flat-space action TYPEs (RESET is not in the flat action space).
_ACTION_TYPE_NAMES: tuple[str, ...] = tuple(f"ACTION{i}" for i in range(1, 8))

# --- per-field provenance tags -------------------------------------------

#: Static ground truth read from the ARC-AGI-3 engine (action-TYPE validity,
#: structural state-change, budget weights). Never a function of any run.
SOURCE_ENGINE = "engine"

#: A valid index refined by a live state-change probe -- the only honest way to
#: learn that a *valid* action is *inert* (e.g. a sparse click cell, or undo
#: with nothing to undo). Carried so the valid-but-inert distinction is not
#: flattened. See ``analysis/action_space_audit.md`` for the probe method.
SOURCE_ENGINE_PROBE = "engine-probe"

#: Measured from an instrumented rollout's per-step action-index log.
SOURCE_RUN_MEASURED = "run-measured"

#: Honest null: no measurement was supplied for this field.
SOURCE_ABSENT = "absent"


# --- budget-cost models --------------------------------------------------


@dataclass(frozen=True)
class BudgetModel:
    """Per-action-TYPE budget cost, with a name for its provenance tag.

    A model maps each of the seven flat action types (``ACTION1..ACTION7``) to
    the integer it adds to a task's action budget when taken. ``budget_cost``
    is therefore a property of the action *type*, independent of whether that
    action is currently valid.
    """

    name: str
    costs: Mapping[str, int]

    def cost(self, action_type: str) -> int:
        """Budget cost of ``action_type``. Raises ``KeyError`` if unknown."""
        return self.costs[action_type]


#: The global engine truth: the scorecard counts every action id ``1..7`` as
#: exactly ``+1`` (``arc_agi/scorecard.py`` ``Card.inc_action_count`` /
#: ``Scorecard.take_action``). This is the RHAE action accounting and the right
#: default for any game without a distinct internal budget.
UNIFORM_BUDGET = BudgetModel("uniform", {name: 1 for name in _ACTION_TYPE_NAMES})

#: lf52's internal survival-budget counter ``asqvqzpfdi``, confirmed from
#: ``environment_files/lf52/271a04aa/lf52.py``:
#:   ACTION1-4 directional move ``tmhxwcojkh`` -> +1 (lf52.py:5277)
#:   ACTION6 grid-click ``dghsidbuet``         -> +1 (lf52.py:5335)
#:   ACTION7 undo (on commit)                  -> +20 (lf52.py:5805)
#:   ACTION5 + special-region ACTION6 click    -> +0 (lf52.py:5327, :5832)
#: These are the paper's move=1 / undo=20 / no-op=0 weights -- real, but a
#: per-game survival mechanic, NOT a universal action cost. The +0 for the
#: special-region ACTION6 click is a *cell-level* exemption that the type-level
#: ACTION6 cost (1, the dominant grid-click) cannot express.
LF52_BUDGET = BudgetModel(
    "lf52",
    {
        "ACTION1": 1,
        "ACTION2": 1,
        "ACTION3": 1,
        "ACTION4": 1,
        "ACTION5": 0,
        "ACTION6": 1,
        "ACTION7": 20,
    },
)

#: Games whose engine defines an internal action budget distinct from the
#: uniform scorecard accounting. Everything else resolves to UNIFORM_BUDGET.
GAME_BUDGET_MODELS: Mapping[str, BudgetModel] = {"lf52": LF52_BUDGET}


def resolve_budget_model(
    task_id: str, override: Optional[BudgetModel] = None
) -> BudgetModel:
    """Pick the budget model for ``task_id`` (``override`` wins if given)."""
    if override is not None:
        return override
    return GAME_BUDGET_MODELS.get(task_id, UNIFORM_BUDGET)


# --- schema --------------------------------------------------------------


@dataclass(frozen=True)
class ActionRow:
    """One flat action index's diagnostic row.

    The field order here *is* the pinned output schema; ``to_dict`` preserves
    it. ``usage_count``/``usage_fraction`` are ``None`` (honest null) unless a
    per-step action log was supplied to the builder. ``sources`` maps each
    provenance-bearing field name to one of the ``SOURCE_*`` tags.
    """

    action_index: int
    action_type: str
    valid_on_task: bool
    is_state_changing: bool
    budget_cost: int
    usage_count: Optional[int]
    usage_fraction: Optional[float]
    sources: Mapping[str, str]

    def to_dict(self) -> dict:
        """Schema-ordered plain dict (JSON-ready; ``sources`` nested)."""
        return {
            "action_index": self.action_index,
            "action_type": self.action_type,
            "valid_on_task": self.valid_on_task,
            "is_state_changing": self.is_state_changing,
            "budget_cost": self.budget_cost,
            "usage_count": self.usage_count,
            "usage_fraction": self.usage_fraction,
            "sources": dict(self.sources),
        }

    def to_flat_dict(self) -> dict:
        """Flat dict for CSV: ``sources`` exploded to ``<field>_source`` cols.

        ``None`` usage stays ``None`` so the CSV writer renders an empty cell
        (not the string ``"None"``).
        """
        out: dict = {
            "action_index": self.action_index,
            "action_type": self.action_type,
            "valid_on_task": self.valid_on_task,
            "is_state_changing": self.is_state_changing,
            "budget_cost": self.budget_cost,
            "usage_count": self.usage_count,
            "usage_fraction": self.usage_fraction,
        }
        for field_name, tag in self.sources.items():
            out[f"{field_name}_source"] = tag
        return out


@dataclass
class TaskActionProfile:
    """A task id plus its 4102 :class:`ActionRow` rows and derived rollups.

    All rollups are computed from ``rows`` on access -- there is no place to
    hardcode them, so they cannot drift from the per-action data.
    """

    task_id: str
    rows: list[ActionRow] = field(default_factory=list)

    # --- rollups (derived) ------------------------------------------------

    @property
    def n_valid(self) -> int:
        return sum(1 for r in self.rows if r.valid_on_task)

    @property
    def n_state_changing(self) -> int:
        return sum(1 for r in self.rows if r.is_state_changing)

    @property
    def dilution_ratio(self) -> Optional[float]:
        """Total-to-valid ratio (``N_ACTIONS / n_valid``): "1 useful per N".

        ``None`` when no action is valid (the ratio is undefined rather than
        infinite). For ls20 (``n_valid=4``) this is ``4102/4 = 1025.5``,
        reproducing the Fig-3 panel-A ~1025:1 figure.
        """
        n = self.n_valid
        if n == 0:
            return None
        return A.N_ACTIONS / n

    # --- serialisation (single code path) ---------------------------------

    def to_json_obj(self) -> dict:
        """Top-level dict: task id, derived rollups, and the per-action rows."""
        return {
            "task_id": self.task_id,
            "n_valid": self.n_valid,
            "n_state_changing": self.n_state_changing,
            "dilution_ratio": self.dilution_ratio,
            "actions": [r.to_dict() for r in self.rows],
        }

    def to_json(self, path: Optional[Union[str, Path]] = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_json_obj(), indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        """One row per action; ``sources`` flattened to ``<field>_source`` cols.

        Built from the same :class:`ActionRow` objects as :meth:`to_json`, so
        the two emitters can never disagree.
        """
        flat = [r.to_flat_dict() for r in self.rows]
        fieldnames = list(flat[0].keys()) if flat else []
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8", newline="")
        return text


# --- empirical usage (b): from a real per-step action log ----------------


def usage_counts_from_action_indices(
    action_indices: Iterable[int],
) -> dict[int, int]:
    """Tally a per-step action-index stream into ``{flat_index: times_fired}``.

    This is the ONLY supported way to obtain empirical usage: an explicit
    stream of the flat action indices a policy actually emitted, step by step,
    from an instrumented rollout. There is no path that derives usage from
    entropy, random-action fraction, or any other aggregate -- doing so would
    re-introduce a hedge the paper deliberately removed.

    Every index must be a valid flat index in ``[0, N_ACTIONS)``; an
    out-of-range value raises ``ValueError`` rather than being silently dropped.
    """
    counts: dict[int, int] = {}
    for raw in action_indices:
        idx = int(raw)
        if not (0 <= idx < A.N_ACTIONS):
            raise ValueError(
                f"action index {idx} out of range [0, {A.N_ACTIONS})"
            )
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def load_action_log_jsonl(path: Union[str, Path]) -> list[int]:
    """Read an instrumented-rollout action log into a flat per-step stream.

    The pinned on-disk format mirrors :class:`arc3_wm.eval_reward_sink`'s
    reward log: one JSON object per line, each ``{"actions": [idx, ...]}`` for
    one episode (flat action indices in emission order). Episodes are
    concatenated into a single stream ready for
    :func:`usage_counts_from_action_indices`.

    A missing file raises ``FileNotFoundError``; a row without an ``"actions"``
    key raises ``ValueError`` (a malformed log is surfaced, not silently
    treated as an empty episode). Blank lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"action log not found: {path}")
    stream: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if "actions" not in obj:
                raise ValueError(
                    f"{path}:line {line_no}: row missing 'actions' key "
                    f"(got keys {sorted(obj)})"
                )
            stream.extend(int(a) for a in obj["actions"])
    return stream


# --- builder -------------------------------------------------------------


def build_task_action_profile(
    task_id: str,
    available_actions: Iterable[int],
    *,
    budget_model: Optional[BudgetModel] = None,
    inert_indices: Optional[Iterable[int]] = None,
    usage_counts: Optional[Mapping[int, int]] = None,
) -> TaskActionProfile:
    """Build a :class:`TaskActionProfile` from a task's available action TYPES.

    Parameters
    ----------
    task_id:
        ARC-AGI-3 game id (e.g. ``"ls20"``). Stored verbatim; not validated
        against the public set so single-game diagnostics work in any clone.
    available_actions:
        The engine's action-TYPE set for this task -- integer ids in ``1..7``
        from ``FrameData.available_actions``. An index is ``valid_on_task``
        iff its action type is present here (ACTION6's 4096 cells all share the
        type-level validity of id ``6``; the engine does not expose cell-level
        ACTION6 validity at reset).
    budget_model:
        Optional :class:`BudgetModel` for ``budget_cost``. Defaults to the
        per-game model from :func:`resolve_budget_model` (lf52 -> its survival
        budget; everything else -> :data:`UNIFORM_BUDGET`).
    inert_indices:
        Optional set of *valid* flat indices known (from a live state-change
        probe) to be inert -- accepted by the engine but changing nothing.
        Those rows get ``is_state_changing=False`` with source
        :data:`SOURCE_ENGINE_PROBE`, so the valid-but-inert case is
        representable rather than flattened. Indices that are not valid are
        ignored (already non-state-changing).
    usage_counts:
        Optional mapping ``flat_index -> times_fired`` from an instrumented
        rollout's per-step action log. When ``None`` (the default), every row's
        usage is the honest null (``None`` / :data:`SOURCE_ABSENT`). When
        supplied, ``usage_fraction`` is ``count / sum(all counts)``. Usage is
        never derived from anything else.
    """
    model = resolve_budget_model(task_id, budget_model)
    budget_source = f"{SOURCE_ENGINE}:{model.name}"
    avail = {int(a) for a in available_actions}
    inert = {int(i) for i in inert_indices} if inert_indices is not None else set()

    measured = usage_counts is not None
    total_usage = sum(int(v) for v in usage_counts.values()) if measured else 0

    rows: list[ActionRow] = []
    for idx in range(A.N_ACTIONS):
        arc_action, _ = A.flat_to_arc(idx)
        action_type = arc_action.name
        type_id = arc_action.value  # 1..7

        valid = type_id in avail
        state_changing = valid and idx not in inert
        sc_source = (
            SOURCE_ENGINE_PROBE if (valid and idx in inert) else SOURCE_ENGINE
        )

        if measured:
            count = int(usage_counts.get(idx, 0))
            frac = (count / total_usage) if total_usage > 0 else 0.0
            usage_count: Optional[int] = count
            usage_fraction: Optional[float] = frac
            usage_source = SOURCE_RUN_MEASURED
        else:
            usage_count = None
            usage_fraction = None
            usage_source = SOURCE_ABSENT

        rows.append(
            ActionRow(
                action_index=idx,
                action_type=action_type,
                valid_on_task=valid,
                is_state_changing=state_changing,
                budget_cost=model.cost(action_type),
                usage_count=usage_count,
                usage_fraction=usage_fraction,
                sources={
                    "valid_on_task": SOURCE_ENGINE,
                    "is_state_changing": sc_source,
                    "budget_cost": budget_source,
                    "usage_count": usage_source,
                    "usage_fraction": usage_source,
                },
            )
        )

    return TaskActionProfile(task_id=task_id, rows=rows)


# --- task validity sources -----------------------------------------------


def from_env(
    game_id: str,
    *,
    seed: int = 0,
    arcade=None,
    **profile_kwargs,
) -> TaskActionProfile:
    """Build a profile from the live engine (authoritative, OFFLINE mode).

    Constructs :class:`arc3_wm.env.ARC3GymEnv`, resets it, and reads
    ``info["available_actions"]``. Requires the game's ``environment_files/``
    to be cached locally (the engine is the source of truth, identical to the
    replay-derived set where both exist). Extra keyword arguments
    (``budget_model``, ``inert_indices``, ``usage_counts``) pass through to
    :func:`build_task_action_profile`.
    """
    from .env import ARC3GymEnv  # local import: keeps module JAX/Gym-free to import

    env = ARC3GymEnv(game_id=game_id, seed=seed, arcade=arcade)
    try:
        _, info = env.reset()
        available = list(info["available_actions"])
    finally:
        env.close()
    return build_task_action_profile(game_id, available, **profile_kwargs)


def available_actions_from_replays(
    game_id: str,
    replays_dir: Union[str, Path] = "data/replays",
) -> list[int]:
    """Union of the engine's ``available_actions`` across a game's replays.

    Reads every ``data/replays/<game_id>/*.recording.jsonl`` frame and unions
    the per-frame ``available_actions`` arrays. This covers games whose
    ``environment_files/`` are not cached (the only source for, e.g., ls20 /
    tn36). The set is a game-constant in the logged data (constant across every
    frame for all audited games), so the union is the task's action-TYPE set.

    Raises ``FileNotFoundError`` if the game's replay directory has no
    recordings.
    """
    game_dir = Path(replays_dir) / game_id
    files = sorted(game_dir.glob("*.recording.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no *.recording.jsonl under {game_dir} - cannot derive "
            f"available_actions for {game_id!r} from replays"
        )
    avail: set[int] = set()
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                stripped = raw.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                data = obj.get("data", obj)  # rows are {"data": {...}, "timestamp": ...}
                actions = data.get("available_actions")
                if actions:
                    avail.update(int(a) for a in actions)
    return sorted(avail)


def from_replays(
    game_id: str,
    replays_dir: Union[str, Path] = "data/replays",
    **profile_kwargs,
) -> TaskActionProfile:
    """Build a profile from the human replays' ``available_actions`` union.

    The provider-agnostic source for any of the 25 public games, including
    those whose engine is not cached. Extra keyword arguments pass through to
    :func:`build_task_action_profile`.
    """
    available = available_actions_from_replays(game_id, replays_dir)
    return build_task_action_profile(game_id, available, **profile_kwargs)
