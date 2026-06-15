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

from pathlib import Path

from arc3_wm.action_diagnostics import (
    LF52_BUDGET,
    SOURCE_ABSENT,
    SOURCE_ENGINE,
    SOURCE_ENGINE_PROBE,
    SOURCE_RUN_MEASURED,
    UNIFORM_BUDGET,
    ActionRow,
    BudgetModel,
    TaskActionProfile,
    available_actions_from_replays,
    build_task_action_profile,
    from_env,
    from_replays,
    load_action_log_jsonl,
    resolve_budget_model,
    usage_counts_from_action_indices,
)
from arc3_wm.action_space import ACTION6_BASE, ACTION7_INDEX, N_ACTIONS

_REPO = Path(__file__).resolve().parents[1]


def _game_cached(game: str) -> bool:
    return (_REPO / "environment_files" / game).is_dir()


def _has_replays(game: str) -> bool:
    d = _REPO / "data" / "replays" / game
    return d.is_dir() and any(d.glob("*.recording.jsonl"))

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


# --- (b) usage population from a real per-step action log -----------------


def test_usage_counts_from_action_indices_counts():
    counts = usage_counts_from_action_indices([5, 5, 5, 0, 4101, 0])
    assert counts == {5: 3, 0: 2, 4101: 1}


def test_usage_counts_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        usage_counts_from_action_indices([0, N_ACTIONS])
    with pytest.raises(ValueError, match="out of range"):
        usage_counts_from_action_indices([-1])


def test_usage_counts_empty_stream():
    assert usage_counts_from_action_indices([]) == {}


def test_measured_usage_populates_rows_and_fraction():
    # A per-step log fired ACTION6 cell idx 5 three times and ACTION1 once.
    log = [5, 5, 5, 0]
    prof = build_task_action_profile(
        "vc33", VC33, usage_counts=usage_counts_from_action_indices(log)
    )
    assert prof.rows[5].usage_count == 3
    assert prof.rows[5].usage_fraction == pytest.approx(0.75)
    assert prof.rows[5].sources["usage_count"] == SOURCE_RUN_MEASURED
    assert prof.rows[0].usage_count == 1
    assert prof.rows[0].usage_fraction == pytest.approx(0.25)
    # Fractions over fired indices sum to 1.
    total = sum(r.usage_fraction for r in prof.rows if r.usage_fraction)
    assert total == pytest.approx(1.0)


def test_measured_zero_is_not_null():
    # An action present in the log's denominator but never fired reads as a
    # measured 0 (count 0, fraction 0.0, run-measured) -- NOT null. This is the
    # honest distinction between "fired zero times" and "never measured".
    prof = build_task_action_profile(
        "vc33", VC33, usage_counts=usage_counts_from_action_indices([5])
    )
    # idx 5 fired once; a different valid cell (idx 6) fired zero times.
    fired = prof.rows[5]
    unfired = prof.rows[6]
    assert fired.usage_count == 1 and fired.usage_fraction == pytest.approx(1.0)
    assert unfired.usage_count == 0 and unfired.usage_fraction == 0.0
    assert unfired.sources["usage_count"] == SOURCE_RUN_MEASURED
    assert unfired.usage_count is not None  # measured, not null


def test_load_action_log_jsonl_roundtrip(tmp_path):
    p = tmp_path / "actions.jsonl"
    p.write_text(
        '{"actions": [5, 5, 0]}\n'
        "\n"  # blank line tolerated
        '{"actions": [5, 4101]}\n',
        encoding="utf-8",
    )
    stream = load_action_log_jsonl(p)
    assert stream == [5, 5, 0, 5, 4101]
    counts = usage_counts_from_action_indices(stream)
    assert counts == {5: 3, 0: 1, 4101: 1}


def test_load_action_log_jsonl_missing_key_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"rewards": [0, 1]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="actions"):
        load_action_log_jsonl(p)


def test_load_action_log_jsonl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_action_log_jsonl(tmp_path / "nope.jsonl")


def test_end_to_end_log_to_csv_marks_measured(tmp_path):
    p = tmp_path / "actions.jsonl"
    p.write_text('{"actions": [5, 5, 0]}\n', encoding="utf-8")
    counts = usage_counts_from_action_indices(load_action_log_jsonl(p))
    prof = build_task_action_profile("vc33", VC33, usage_counts=counts)
    obj = json.loads(prof.to_json())
    assert obj["actions"][5]["usage_count"] == 2
    assert obj["actions"][5]["sources"]["usage_count"] == SOURCE_RUN_MEASURED
    # An unfired valid cell is a measured 0 in JSON (not null).
    assert obj["actions"][7]["usage_count"] == 0


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


# --- budget-cost model (engine-confirmed weights) -------------------------


def test_uniform_budget_is_default_and_costs_one():
    # The global engine truth: every action id 1..7 counts as +1
    # (arc_agi/scorecard.py inc_action_count). So absent a game-specific
    # internal budget, every row costs 1 regardless of validity.
    prof = build_task_action_profile("cd82", CD82)
    assert all(r.budget_cost == 1 for r in prof.rows)
    assert all(r.sources["budget_cost"] == "engine:uniform" for r in prof.rows)


def test_lf52_budget_weights_match_engine_source():
    # Confirmed from environment_files/lf52/271a04aa/lf52.py:
    #   directional moves tmhxwcojkh -> +1 (lf52.py:5277)
    #   ACTION6 grid-click dghsidbuet -> +1 (lf52.py:5335)
    #   undo commit -> +20 (lf52.py:5805)
    #   ACTION5 + special-region click -> +0 (lf52.py:5327, :5832)
    # i.e. the paper's move=1 / undo=20 / no-op=0 weights are lf52's internal
    # survival-budget counter, resolved automatically by task id.
    prof = build_task_action_profile("lf52", LF52)
    by_type = {r.action_type: r.budget_cost for r in prof.rows}
    assert by_type["ACTION1"] == 1
    assert by_type["ACTION2"] == 1
    assert by_type["ACTION3"] == 1
    assert by_type["ACTION4"] == 1
    assert by_type["ACTION5"] == 0  # budget-exempt (no-op)
    assert by_type["ACTION6"] == 1  # grid-click
    assert by_type["ACTION7"] == 20  # undo
    assert all(r.sources["budget_cost"] == "engine:lf52" for r in prof.rows)


def test_resolve_budget_model_registry():
    assert resolve_budget_model("lf52") is LF52_BUDGET
    assert resolve_budget_model("vc33") is UNIFORM_BUDGET  # default
    assert resolve_budget_model("anything") is UNIFORM_BUDGET


def test_explicit_budget_model_override_wins():
    # A caller can force a model regardless of task id (e.g. score lf52 under
    # the uniform RHAE accounting instead of its survival budget).
    prof = build_task_action_profile("lf52", LF52, budget_model=UNIFORM_BUDGET)
    assert all(r.budget_cost == 1 for r in prof.rows)
    assert all(r.sources["budget_cost"] == "engine:uniform" for r in prof.rows)


def test_budget_model_covers_all_seven_action_types():
    for model in (UNIFORM_BUDGET, LF52_BUDGET):
        for name in (f"ACTION{i}" for i in range(1, 8)):
            assert isinstance(model.cost(name), int)


def test_budget_model_rejects_unknown_action_type():
    with pytest.raises(KeyError):
        UNIFORM_BUDGET.cost("RESET")  # RESET is not in the flat action space


def test_budget_cost_is_type_property_not_validity():
    # An invalid action still reports its action-TYPE budget cost (what it
    # WOULD cost if taken), not 0-because-invalid.
    prof = build_task_action_profile("lf52", LF52)
    a7 = prof.rows[ACTION7_INDEX]
    assert a7.valid_on_task is True and a7.budget_cost == 20
    # ls20 has no ACTION7, but the cost of the ACTION7 type is still defined.
    ls20 = build_task_action_profile("ls20", LS20, budget_model=LF52_BUDGET)
    assert ls20.rows[ACTION7_INDEX].valid_on_task is False
    assert ls20.rows[ACTION7_INDEX].budget_cost == 20


# --- validity sources: live engine + replays ------------------------------


@pytest.mark.parametrize(
    "game,n_valid",
    [("vc33", 4096), ("sb26", 4098), ("cd82", 4101), ("lf52", 4101)],
)
def test_from_env_matches_audited_n_valid(game, n_valid):
    if not _game_cached(game):
        pytest.skip(f"environment_files/{game} not cached")
    prof = from_env(game)
    assert prof.task_id == game
    assert prof.n_valid == n_valid


def test_from_env_lf52_uses_lf52_budget_automatically():
    if not _game_cached("lf52"):
        pytest.skip("environment_files/lf52 not cached")
    prof = from_env("lf52")
    by_type = {r.action_type: r.budget_cost for r in prof.rows}
    assert by_type["ACTION7"] == 20 and by_type["ACTION5"] == 0


def test_available_actions_from_replays_ls20():
    if not _has_replays("ls20"):
        pytest.skip("ls20 replays absent")
    # ls20 is directional-only and NOT cached as environment_files -> replays
    # are the only validity source. Ground truth: {1,2,3,4}.
    assert available_actions_from_replays("ls20") == [1, 2, 3, 4]


def test_from_replays_ls20_fig3_numbers():
    if not _has_replays("ls20"):
        pytest.skip("ls20 replays absent")
    prof = from_replays("ls20")
    assert prof.n_valid == 4  # Fig-3 panel A: 4/4102
    assert round(prof.dilution_ratio) == 1026  # ~1025:1


def test_from_replays_lf52_matches_audit():
    if not _has_replays("lf52"):
        pytest.skip("lf52 replays absent")
    # lf52 -> {1,2,3,4,6,7}: 4 directional + 4096 click + 1 undo = 4101.
    assert available_actions_from_replays("lf52") == [1, 2, 3, 4, 6, 7]
    assert from_replays("lf52").n_valid == 4101


def test_from_replays_missing_game_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        available_actions_from_replays("zz99", replays_dir=tmp_path)


def test_env_and_replays_agree_where_both_exist():
    # Cross-check: the live engine and the replay union must be byte-identical
    # for a cached game (vc33), validating replays as a source for uncached ones.
    if not (_game_cached("vc33") and _has_replays("vc33")):
        pytest.skip("vc33 env files or replays absent")
    live = {r.action_index for r in from_env("vc33").rows if r.valid_on_task}
    repl = {r.action_index for r in from_replays("vc33").rows if r.valid_on_task}
    assert live == repl


def test_package_reexports_action_profile_helpers():
    import arc3_wm

    assert arc3_wm.action_profile_from_env is from_env
    assert arc3_wm.action_profile_from_replays is from_replays
    assert arc3_wm.build_task_action_profile is build_task_action_profile


# --- ActionRow direct construction is well-formed -------------------------


def test_actionrow_is_frozen():
    prof = build_task_action_profile("ls20", LS20)
    with pytest.raises((AttributeError, TypeError)):
        prof.rows[0].valid_on_task = False  # type: ignore[misc]


def test_profile_task_id_preserved():
    assert build_task_action_profile("tn36", [6]).task_id == "tn36"
    assert isinstance(build_task_action_profile("ls20", LS20).rows[0], ActionRow)
