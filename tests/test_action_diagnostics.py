"""Per-task action-diagnostic schema, builder, rollups, and JSON/CSV emit.

Covers the *structural / validity profile* (the (a) half of the task brief):
which of the 4102 flat indices are valid, which are state-changing, the
per-action-type budget cost, and the task-level rollups derived FROM the rows
(never hardcoded). The empirical-usage (b) half is honest-null by default:
``usage_count``/``usage_fraction`` are ``None`` with a source tag that says so,
and are never synthesised from anything.

Ground-truth action-TYPE sets used as fixtures come from the read-only engine
audit (``analysis/action_space_audit.md``), cross-checked live in this session:

    vc33 -> {6}            n_valid 4096
    sb26 -> {5,6,7}        n_valid 4098
    cd82 -> {1,2,3,4,5,6}  n_valid 4101
    ls20 -> {1,2,3,4}      n_valid 4    (directional-only; Fig-3 panel A "4/4102")
    lf52 -> {1,2,3,4,6,7}  n_valid 4101
"""
from __future__ import annotations

import csv
import io
import json

import pytest

from arc3_wm.action_diagnostics import (
    SOURCE_ABSENT,
    SOURCE_ENGINE,
    SOURCE_ENGINE_PROBE,
    ActionRow,
    TaskActionProfile,
    build_task_action_profile,
)
from arc3_wm.action_space import ACTION6_BASE, ACTION7_INDEX, N_ACTIONS

# --- fixtures -------------------------------------------------------------

LS20 = [1, 2, 3, 4]
VC33 = [6]
CD82 = [1, 2, 3, 4, 5, 6]
LF52 = [1, 2, 3, 4, 6, 7]


# --- row count & schema ---------------------------------------------------


def test_profile_has_one_row_per_flat_index():
    prof = build_task_action_profile("ls20", LS20)
    assert len(prof.rows) == N_ACTIONS
    assert [r.action_index for r in prof.rows] == list(range(N_ACTIONS))


def test_row_schema_exact_field_set():
    prof = build_task_action_profile("vc33", VC33)
    row = prof.rows[ACTION6_BASE].to_dict()
    # Pin the schema exactly - a future instrumented rollout and the
    # collaborator must consume an identical shape.
    assert list(row.keys()) == [
        "action_index",
        "action_type",
        "valid_on_task",
        "is_state_changing",
        "budget_cost",
        "usage_count",
        "usage_fraction",
        "sources",
    ]


def test_action_type_labels():
    prof = build_task_action_profile("cd82", CD82)
    assert prof.rows[0].action_type == "ACTION1"
    assert prof.rows[4].action_type == "ACTION5"
    assert prof.rows[ACTION6_BASE].action_type == "ACTION6"
    assert prof.rows[ACTION7_INDEX].action_type == "ACTION7"


# --- valid_on_task (engine-derived from available action TYPES) -----------


def test_ls20_directional_only_valid_set():
    prof = build_task_action_profile("ls20", LS20)
    valid = {r.action_index for r in prof.rows if r.valid_on_task}
    assert valid == {0, 1, 2, 3}  # ACTION1..ACTION4 only
    # ACTION6 grid and ACTION7 are NOT valid for ls20.
    assert not prof.rows[ACTION6_BASE].valid_on_task
    assert not prof.rows[ACTION7_INDEX].valid_on_task


def test_vc33_click_grid_valid():
    prof = build_task_action_profile("vc33", VC33)
    assert all(
        prof.rows[i].valid_on_task
        for i in range(ACTION6_BASE, ACTION6_BASE + 4096)
    )
    assert not prof.rows[0].valid_on_task  # ACTION1 not available
    assert not prof.rows[ACTION7_INDEX].valid_on_task


def test_valid_source_is_engine():
    prof = build_task_action_profile("ls20", LS20)
    assert prof.rows[0].sources["valid_on_task"] == SOURCE_ENGINE


# --- rollups derived FROM rows, not hardcoded -----------------------------


@pytest.mark.parametrize(
    "task,avail,n_valid",
    [
        ("vc33", VC33, 4096),
        ("cd82", CD82, 4101),
        ("ls20", LS20, 4),
        ("lf52", LF52, 4101),
    ],
)
def test_n_valid_rollup(task, avail, n_valid):
    prof = build_task_action_profile(task, avail)
    assert prof.n_valid == n_valid
    # Derived, not stored: equals the count of valid rows.
    assert prof.n_valid == sum(r.valid_on_task for r in prof.rows)


def test_dilution_ratio_matches_fig3_for_ls20():
    # Fig-3 panel A reports ls20 ~1025:1 (4102/4 = 1025.5).
    prof = build_task_action_profile("ls20", LS20)
    assert prof.dilution_ratio == pytest.approx(N_ACTIONS / 4)
    assert round(prof.dilution_ratio) == 1026  # rounds to the paper's ~1024-1026


def test_dilution_ratio_near_one_for_click_game():
    prof = build_task_action_profile("vc33", VC33)
    assert prof.dilution_ratio == pytest.approx(N_ACTIONS / 4096, rel=1e-6)
    assert prof.dilution_ratio < 1.01


def test_dilution_ratio_none_when_no_valid_actions():
    prof = build_task_action_profile("empty", [])
    assert prof.n_valid == 0
    assert prof.dilution_ratio is None  # undefined, not inf/crash


def test_n_state_changing_rollup_equals_valid_by_default():
    # Without a state-change probe, every valid action is presumed
    # state-changing (structural default); the rollup is still derived.
    prof = build_task_action_profile("cd82", CD82)
    assert prof.n_state_changing == prof.n_valid
    assert prof.n_state_changing == sum(r.is_state_changing for r in prof.rows)


# --- honest-null (b): usage is null unless measured -----------------------


def test_usage_is_null_by_default():
    prof = build_task_action_profile("vc33", VC33)
    for r in prof.rows:
        assert r.usage_count is None
        assert r.usage_fraction is None
        assert r.sources["usage_count"] == SOURCE_ABSENT
        assert r.sources["usage_fraction"] == SOURCE_ABSENT


def test_no_back_door_to_synthesize_usage():
    # The builder has no parameter that could carry entropy / rand-fraction
    # aggregates - usage can ONLY arrive as explicit per-index counts.
    import inspect

    sig = inspect.signature(build_task_action_profile)
    banned = {"entropy", "rand_fraction", "rand_action", "ent"}
    assert banned.isdisjoint(sig.parameters)


# --- is_state_changing independent of valid (Tier B representable) --------


def test_inert_indices_flip_state_changing_not_valid():
    # A valid-but-inert action (e.g. sb26 sparse click cell, or undo with
    # nothing to undo) must be representable: valid=True, state_changing=False.
    inert = {ACTION6_BASE + 10, ACTION6_BASE + 20}
    prof = build_task_action_profile("vc33", VC33, inert_indices=inert)
    for idx in inert:
        assert prof.rows[idx].valid_on_task is True
        assert prof.rows[idx].is_state_changing is False
        assert prof.rows[idx].sources["is_state_changing"] == SOURCE_ENGINE_PROBE
    # n_state_changing drops below n_valid by exactly the inert count.
    assert prof.n_state_changing == prof.n_valid - len(inert)


def test_inert_index_on_invalid_action_is_noop():
    # Marking an already-invalid index inert cannot make it "more false".
    prof = build_task_action_profile("ls20", LS20, inert_indices={ACTION6_BASE})
    assert prof.rows[ACTION6_BASE].valid_on_task is False
    assert prof.rows[ACTION6_BASE].is_state_changing is False


def test_invalid_action_never_state_changing():
    prof = build_task_action_profile("ls20", LS20)
    for r in prof.rows:
        if not r.valid_on_task:
            assert r.is_state_changing is False


# --- JSON / CSV emitted from one code path --------------------------------


def test_to_json_top_level_shape():
    prof = build_task_action_profile("ls20", LS20)
    obj = json.loads(prof.to_json())
    assert obj["task_id"] == "ls20"
    assert obj["n_valid"] == 4
    assert obj["n_state_changing"] == 4
    assert obj["dilution_ratio"] == pytest.approx(N_ACTIONS / 4)
    assert len(obj["actions"]) == N_ACTIONS
    assert obj["actions"][0]["action_type"] == "ACTION1"


def test_to_csv_header_and_rowcount():
    prof = build_task_action_profile("ls20", LS20)
    reader = csv.DictReader(io.StringIO(prof.to_csv()))
    rows = list(reader)
    assert len(rows) == N_ACTIONS
    # Per-field source tags flatten to <field>_source columns; nested
    # "sources" dict must NOT leak into the CSV as a raw column.
    assert "sources" not in reader.fieldnames
    for col in (
        "action_index",
        "action_type",
        "valid_on_task",
        "is_state_changing",
        "budget_cost",
        "usage_count",
        "usage_fraction",
        "valid_on_task_source",
        "usage_count_source",
    ):
        assert col in reader.fieldnames


def test_csv_and_json_agree_on_core_fields():
    prof = build_task_action_profile("cd82", CD82)
    obj = json.loads(prof.to_json())
    csv_rows = list(csv.DictReader(io.StringIO(prof.to_csv())))
    # Same code path -> identical values for a sampled index.
    i = ACTION6_BASE
    assert csv_rows[i]["action_type"] == obj["actions"][i]["action_type"]
    assert (csv_rows[i]["valid_on_task"] == "True") == obj["actions"][i][
        "valid_on_task"
    ]


def test_csv_null_usage_renders_empty():
    prof = build_task_action_profile("vc33", VC33)
    csv_rows = list(csv.DictReader(io.StringIO(prof.to_csv())))
    assert csv_rows[0]["usage_count"] == ""  # None -> empty cell, not "None"
    assert csv_rows[0]["usage_fraction"] == ""


def test_to_json_and_to_csv_write_files(tmp_path):
    prof = build_task_action_profile("ls20", LS20)
    jp = tmp_path / "ls20.json"
    cp = tmp_path / "ls20.csv"
    prof.to_json(jp)
    prof.to_csv(cp)
    assert json.loads(jp.read_text())["task_id"] == "ls20"
    assert len(cp.read_text().splitlines()) == N_ACTIONS + 1  # header + rows


# --- ActionRow direct construction is well-formed -------------------------


def test_actionrow_is_frozen():
    prof = build_task_action_profile("ls20", LS20)
    with pytest.raises((AttributeError, TypeError)):
        prof.rows[0].valid_on_task = False  # type: ignore[misc]


def test_profile_task_id_preserved():
    assert build_task_action_profile("tn36", [6]).task_id == "tn36"
    assert isinstance(build_task_action_profile("ls20", LS20).rows[0], ActionRow)
