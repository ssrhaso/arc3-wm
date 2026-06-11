"""Round-trip and mask invariants for the flat 4102-way action space.

Property: every flat idx in [0, 4102) decodes to a (GameAction, data) pair
that re-encodes to the same idx. The mask correctly enumerates ACTION6's
4096-cell grid only when 6 in available_actions.
"""
from __future__ import annotations

import numpy as np
import pytest
from arcengine import GameAction

from arc3_wm.action_space import (
    ACTION6_BASE,
    ACTION6_COUNT,
    ACTION7_INDEX,
    GRID,
    N_ACTIONS,
    arc_to_flat,
    build_mask,
    describe_action,
    flat_to_arc,
    logit_bias,
)


def test_layout_constants():
    assert GRID == 64
    assert ACTION6_BASE == 5
    assert ACTION6_COUNT == 64 * 64 == 4096
    assert ACTION7_INDEX == 4101
    assert N_ACTIONS == 4102


def test_simple_actions_endpoints():
    assert flat_to_arc(0) == (GameAction.ACTION1, None)
    assert flat_to_arc(1) == (GameAction.ACTION2, None)
    assert flat_to_arc(2) == (GameAction.ACTION3, None)
    assert flat_to_arc(3) == (GameAction.ACTION4, None)
    assert flat_to_arc(4) == (GameAction.ACTION5, None)
    assert flat_to_arc(4101) == (GameAction.ACTION7, None)


def test_action6_corners():
    a, d = flat_to_arc(5)
    assert a == GameAction.ACTION6 and d == {"x": 0, "y": 0}
    # last ACTION6 cell: idx 4100 -> y=63, x=63
    a, d = flat_to_arc(4100)
    assert a == GameAction.ACTION6 and d == {"x": 63, "y": 63}
    # row-major: idx = 5 + y*64 + x
    a, d = flat_to_arc(5 + 7 * 64 + 12)
    assert d == {"x": 12, "y": 7}


def test_round_trip_all_indices():
    """Property test: every valid flat idx round-trips through encode/decode."""
    for idx in range(N_ACTIONS):
        arc, data = flat_to_arc(idx)
        if data is None:
            assert arc_to_flat(arc) == idx
        else:
            assert arc_to_flat(arc, **data) == idx


def test_out_of_range_decode():
    with pytest.raises(ValueError, match="out of range"):
        flat_to_arc(-1)
    with pytest.raises(ValueError, match="out of range"):
        flat_to_arc(N_ACTIONS)


def test_encode_reset_rejected():
    with pytest.raises(ValueError, match="RESET"):
        arc_to_flat(GameAction.RESET)


def test_encode_action6_requires_xy():
    with pytest.raises(ValueError, match="ACTION6 requires"):
        arc_to_flat(GameAction.ACTION6)
    with pytest.raises(ValueError, match=r"\[0, 64\)"):
        arc_to_flat(GameAction.ACTION6, x=64, y=0)
    with pytest.raises(ValueError, match=r"\[0, 64\)"):
        arc_to_flat(GameAction.ACTION6, x=0, y=-1)


def test_mask_no_actions():
    m = build_mask([])
    assert m.dtype == bool
    assert m.shape == (N_ACTIONS,)
    assert not m.any()


def test_mask_all_simple():
    # vc33: only ACTION6 is available initially -> all 4096 grid cells live.
    m = build_mask([6])
    assert m.sum() == ACTION6_COUNT
    assert m[ACTION6_BASE : ACTION6_BASE + ACTION6_COUNT].all()
    assert not m[:ACTION6_BASE].any()
    assert not m[ACTION7_INDEX]


def test_mask_keyboard_only():
    # tu93: ACTION1..ACTION4 -> first 4 indices live, nothing else.
    m = build_mask([1, 2, 3, 4])
    expected = np.zeros(N_ACTIONS, dtype=bool)
    expected[:4] = True
    np.testing.assert_array_equal(m, expected)


def test_mask_combined():
    # cd82: 1..6 -> first 5 simple + 4096 grid live, ACTION7 still dead.
    m = build_mask([1, 2, 3, 4, 5, 6])
    assert m.sum() == 5 + ACTION6_COUNT
    assert m[:5].all()
    assert m[ACTION6_BASE : ACTION6_BASE + ACTION6_COUNT].all()
    assert not m[ACTION7_INDEX]


def test_mask_includes_action7():
    m = build_mask([7])
    assert m.sum() == 1
    assert m[ACTION7_INDEX]


def test_mask_ignores_unknown_ids():
    # Future-proof: unknown ids in fd.available_actions must not crash.
    m = build_mask([1, 99, 6])
    assert m[0] and m[ACTION6_BASE]  # 1 and 6 honoured
    assert m.sum() == 1 + ACTION6_COUNT


def test_logit_bias_zero_on_allowed_neg_inf_on_masked():
    mask = build_mask([1, 2, 3, 4])  # tu93-style keyboard-only
    bias = logit_bias(mask)
    assert bias.shape == (N_ACTIONS,)
    assert bias.dtype == np.float32
    # Allowed indices contribute nothing additive; masked ones are -inf.
    assert np.array_equal(bias == 0.0, mask)
    assert np.isneginf(bias[~mask]).all()
    assert not np.isneginf(bias[mask]).any()


def test_logit_bias_drives_softmax_to_zero_on_masked():
    mask = build_mask([1, 2, 3, 4])
    logits = np.ones(N_ACTIONS, dtype=np.float32)
    probs = np.exp(logits + logit_bias(mask))
    probs /= probs.sum()
    # Masked actions get exactly zero probability; mass is on the 4 allowed.
    assert probs[~mask].sum() == 0.0
    np.testing.assert_allclose(probs[mask], 0.25)


def test_logit_bias_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        logit_bias(np.zeros(10, dtype=bool))


def test_logit_bias_honours_requested_dtype():
    # The dtype argument controls the output precision (a float64 actor
    # head wants a float64 bias). -inf is representable in either width.
    mask = build_mask([1, 2, 3, 4])
    bias = logit_bias(mask, dtype=np.float64)
    assert bias.dtype == np.float64
    assert np.array_equal(bias == 0.0, mask)
    assert np.isneginf(bias[~mask]).all()


def test_describe_action_parameterless():
    assert describe_action(0) == "ACTION1"
    assert describe_action(4) == "ACTION5"
    assert describe_action(ACTION7_INDEX) == "ACTION7"


def test_describe_action_action6():
    assert describe_action(5) == "ACTION6(x=0, y=0)"
    assert describe_action(4100) == "ACTION6(x=63, y=63)"
    assert describe_action(5 + 7 * 64 + 12) == "ACTION6(x=12, y=7)"


def test_describe_action_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        describe_action(N_ACTIONS)


def test_describe_action_matches_decode_for_all_indices():
    # describe_action is a thin label over flat_to_arc; it must agree on every idx.
    for idx in range(N_ACTIONS):
        arc, data = flat_to_arc(idx)
        label = describe_action(idx)
        assert label.startswith(arc.name)
        if data is not None:
            assert f"x={data['x']}" in label and f"y={data['y']}" in label
