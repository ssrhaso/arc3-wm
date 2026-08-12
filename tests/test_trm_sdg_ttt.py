"""Spec for synthetic game generation and online test-time training."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arc3_wm.trm import config as C  # noqa: E402
from arc3_wm.trm.data import WMTransitionDataset  # noqa: E402
from arc3_wm.trm.synthetic import (  # noqa: E402
    FAMILIES,
    generate_corpus,
    make_game,
    rollout_game,
)
from arc3_wm.trm.ttt import OnlineFineTuner  # noqa: E402
from arc3_wm.trm.world_model import TRMWorldModel  # noqa: E402

TINY = C.WorldModelConfig(
    core=C.TRMCoreConfig(
        d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
        n_supervision=1, halt_max_steps=1,
    ),
    tokenizer=C.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4),
)


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_rolls_valid_transitions(family):
    rng = np.random.default_rng(0)
    game = make_game(family, rng)
    ep = rollout_game(game, rng, max_steps=80, scripted_prob=1.0)
    assert ep["grids"].dtype == np.uint8
    assert ep["grids"].min() >= 0 and ep["grids"].max() < 16
    assert len(ep["grids"]) == len(ep["actions"]) == len(ep["levels"])
    assert ep["actions"][0] == -1
    # Scripted play must reach at least one reward in most families.
    if family != "pusher":  # pusher's script is random; win not guaranteed
        assert ep["levels"][-1] >= 1 or ep["states"][-1] == 2


def test_generate_corpus_is_pipeline_compatible(tmp_path):
    paths = generate_corpus(tmp_path, n_games=5, episodes_per_game=2, seed=0)
    assert len(paths) == 5
    ds = WMTransitionDataset(paths, dedup=True)
    assert len(ds) > 50
    sample = ds[0]
    assert sample["grid"].shape == (64, 64)
    # Reward transitions exist somewhere in the corpus.
    rewards = sum(float(ds[i]["reward"]) for i in range(len(ds)))
    assert rewards > 0


def test_corpus_determinism(tmp_path):
    a = generate_corpus(tmp_path / "a", n_games=2, episodes_per_game=1, seed=7)
    b = generate_corpus(tmp_path / "b", n_games=2, episodes_per_game=1, seed=7)
    for pa, pb in zip(a, b):
        da, db = np.load(pa), np.load(pb)
        assert np.array_equal(da["grids"], db["grids"])
        assert np.array_equal(da["actions"], db["actions"])


def test_online_fine_tuner_reduces_loss_on_repeated_dynamics():
    torch.manual_seed(0)
    model = TRMWorldModel(TINY)
    tuner = OnlineFineTuner(model, lr=1e-3, warmup_steps=1, batch_size=8)
    rng = np.random.default_rng(0)
    g1 = rng.integers(0, 4, size=(64, 64)).astype(np.uint8)
    g2 = np.roll(g1, 2, axis=1)
    frames = {g1.tobytes(): g1, g2.tobytes(): g2}
    log = [(0, g1.tobytes(), 1, 0, g2.tobytes(), 0.0, None)] * 40
    first = tuner.update(log[:20], frames, max_steps=10)
    assert first["steps"] > 0
    loss_a = first.get("grid")
    second = tuner.update(log, frames, max_steps=10)  # 20 fresh rows
    assert second["fresh"] == 20
    assert second.get("grid") is not None and loss_a is not None
    assert second["grid"] < loss_a  # same dynamics repeated -> loss drops


def test_fine_tuner_skips_without_fresh_data():
    model = TRMWorldModel(TINY)
    tuner = OnlineFineTuner(model, batch_size=8)
    g = np.zeros((64, 64), dtype=np.uint8)
    frames = {g.tobytes(): g}
    log = [(0, g.tobytes(), 0, 0, g.tobytes(), 0.0, None)] * 10
    tuner.update(log, frames)
    again = tuner.update(log, frames)  # no new rows
    assert again["steps"] == tuner.global_step and again["fresh"] == 0


def test_policy_self_imitation_learns_success_segment():
    from arc3_wm.trm.policy import TRMPolicy
    from arc3_wm.trm.ttt import PolicySelfImitation

    torch.manual_seed(0)
    pol = TRMPolicy(C.PolicyConfig(
        core=TINY.core, tokenizer=TINY.tokenizer, plan_length=4))
    tuner = PolicySelfImitation(pol, lr=1e-3, warmup_steps=1, batch_size=8)
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 4, size=(64, 64)).astype(np.uint8)
    mask = np.ones(4102, dtype=bool)
    record = {
        "rewards": [0.0, 0.0, 1.0, 0.0],
        "grids": [grid, grid, grid, grid],
        "actions": [7, 8, 9, 10],
        "masks": [mask] * 4,
    }
    assert tuner.ingest(record) == 1  # one cleared segment (steps 0-2)
    assert len(tuner.segments) == 1
    assert tuner.segments[0][1].tolist() == [7, 8, 9]  # failed tail excluded
    first = tuner.update(max_steps=6)
    assert first["steps"] > 0
    tuner._pending_new = 1  # force a second burst on the same data
    second = tuner.update(max_steps=6)
    assert second["bc"] < first["bc"]  # same demo repeated -> loss drops
    noop = tuner.update()
    assert noop["steps"] == tuner.global_step  # no new segments -> skip
