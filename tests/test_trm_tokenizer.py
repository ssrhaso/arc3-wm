"""Spec for arc3_wm.trm.tokenizer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from arc3_wm.action_space import arc_to_flat, flat_to_arc  # noqa: E402
from arc3_wm.trm.config import TokenizerConfig  # noqa: E402
from arc3_wm.trm.tokenizer import (  # noqa: E402
    ActionEncoder,
    CellHead,
    GridHead,
    GridTokenizer,
    assemble_flat_logits,
    click_logits_from_cells,
    flat_action_components,
    rgb_free_grid_check,
)

CFG = TokenizerConfig(d_model=64, patch_size=8, cell_embed_dim=4)


def test_grid_tokenizer_shapes():
    tok = GridTokenizer(CFG)
    grid = torch.randint(0, 16, (2, 64, 64))
    tokens = tok(grid)
    assert tokens.shape == (2, CFG.n_tokens, 64)
    with pytest.raises(ValueError):
        tok(torch.randint(0, 16, (2, 32, 32)))


def test_grid_tokenizer_patch_locality():
    # Changing one cell must change exactly one token (its patch).
    cfg = TokenizerConfig(d_model=32, patch_size=8, cell_embed_dim=4, learned_pos=False)
    tok = GridTokenizer(cfg)
    grid = torch.zeros(1, 64, 64, dtype=torch.long)
    base = tok(grid)
    grid2 = grid.clone()
    grid2[0, 10, 20] = 5  # patch row 1, col 2 -> token index 1*8+2 = 10
    changed = (tok(grid2) - base).abs().sum(-1).nonzero()
    assert changed.tolist() == [[0, 10]]


def test_grid_head_round_trip_positions():
    # GridHead must place patch logits back at the right cells: train a
    # linear-only path sanity by checking locality, mirroring the tokenizer.
    head = GridHead(CFG)
    tokens = torch.zeros(1, CFG.n_tokens, CFG.d_model)
    base = head(tokens)
    assert base.shape == (1, 64, 64, 16)
    tokens2 = tokens.clone()
    tokens2[0, 10] = 1.0  # token 10 -> patch (row 1, col 2) -> cells rows 8..15, cols 16..23
    delta = (head(tokens2) - base).abs().sum(-1)[0]
    rows, cols = delta.nonzero(as_tuple=True)
    assert rows.min() == 8 and rows.max() == 15
    assert cols.min() == 16 and cols.max() == 23


def test_cell_head_locality_matches_grid_head():
    head = CellHead(CFG)
    tokens = torch.zeros(1, CFG.n_tokens, CFG.d_model)
    base = head(tokens)
    tokens[0, 10] = 1.0
    delta = (head(tokens) - base)[0].abs()
    rows, cols = delta.nonzero(as_tuple=True)
    assert rows.min() == 8 and rows.max() == 15
    assert cols.min() == 16 and cols.max() == 23


def test_flat_action_components_match_action_space():
    # Cross-check against the canonical arc3_wm.action_space bijection.
    idxs = torch.tensor([0, 4, 5, 5 + 63, 5 + 64 * 3 + 7, 4100, 4101])
    a_type, x, y = flat_action_components(idxs)
    for i, idx in enumerate(idxs.tolist()):
        action, data = flat_to_arc(idx)
        if data is None:
            assert x[i] == -1 and y[i] == -1
        else:
            assert (x[i].item(), y[i].item()) == (data["x"], data["y"])
            assert a_type[i] == 5
    assert a_type[0] == 0 and a_type[1] == 4 and a_type[-1] == 6


def test_flat_action_components_rejects_out_of_range():
    with pytest.raises(ValueError):
        flat_action_components(torch.tensor([4102]))


def test_action_encoder_distinguishes_actions():
    enc = ActionEncoder(CFG)
    tokens = enc(torch.tensor([0, 1, 5, 5 + 64, 4101]))
    assert tokens.shape == (5, CFG.d_model)
    # All pairwise distinct.
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.allclose(tokens[i], tokens[j])


def test_assemble_flat_logits_is_joint_distribution():
    type_logits = torch.randn(3, 7)
    click_logits = torch.randn(3, 4096)
    flat = assemble_flat_logits(type_logits, click_logits)
    assert flat.shape == (3, 4102)
    # Sum over the ACTION6 block of softmax(flat) equals softmax(type)[5]
    # when type logits are compared within the same normalisation:
    p_flat = torch.softmax(flat, dim=-1)
    p_type = torch.softmax(type_logits.float(), dim=-1)
    a6_mass = p_flat[:, 5:4101].sum(-1)
    # Normalisers differ (7-way vs 4102-way), so compare ratios instead:
    ratio = p_flat[:, 0] / a6_mass  # p(type0) / p(type5)
    expected = p_type[:, 0] / p_type[:, 5]
    assert torch.allclose(ratio, expected, rtol=1e-4)


def test_click_logits_row_major_matches_flat_index():
    cells = torch.zeros(1, 64, 64)
    cells[0, 3, 7] = 9.0  # y=3, x=7
    flat_click = click_logits_from_cells(cells)
    hot = flat_click[0].argmax().item()
    assert hot == 3 * 64 + 7
    assert arc_to_flat(flat_to_arc(5 + hot)[0], x=7, y=3) == 5 + hot


def test_rgb_tripwire():
    with pytest.raises(ValueError, match="palette"):
        rgb_free_grid_check(torch.full((1, 64, 64), 255))
    rgb_free_grid_check(torch.full((1, 64, 64), 15))


def test_quantize_to_palette_round_trip_with_tokenizer():
    # End-to-end: palette indices -> RGB (env obs) -> quantize -> identical.
    from arc3_wm.dynamics_probe import quantize_to_palette
    from arc3_wm.palette import decode_frame

    rng = np.random.default_rng(0)
    grid = rng.integers(0, 16, size=(64, 64), dtype=np.int16)
    rgb = decode_frame(grid)
    back = quantize_to_palette(rgb)
    assert (back == grid).all()
