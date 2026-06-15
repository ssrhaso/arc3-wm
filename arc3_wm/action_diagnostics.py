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
    "ActionRow",
    "TaskActionProfile",
    "build_task_action_profile",
]

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


# --- builder -------------------------------------------------------------


def _budget_cost(action_type: str) -> int:
    """Uniform budget cost (every action costs 1).

    This is the global engine truth: the scorecard counts every action id in
    ``1..7`` as exactly ``+1`` (``arc_agi/scorecard.py`` ``inc_action_count``).
    Game-specific internal budgets that weight actions differently (e.g. lf52's
    survival counter: move=1/undo=20/no-op=0) are layered in separately.
    """
    return 1


def build_task_action_profile(
    task_id: str,
    available_actions: Iterable[int],
    *,
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
                budget_cost=_budget_cost(action_type),
                usage_count=usage_count,
                usage_fraction=usage_fraction,
                sources={
                    "valid_on_task": SOURCE_ENGINE,
                    "is_state_changing": sc_source,
                    "budget_cost": SOURCE_ENGINE,
                    "usage_count": usage_source,
                    "usage_fraction": usage_source,
                },
            )
        )

    return TaskActionProfile(task_id=task_id, rows=rows)
