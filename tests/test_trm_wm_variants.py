"""Spec for the WM-variant components: ensemble disagreement, unroll loss,
rollout probe windows."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arc3_wm.action_space import build_mask  # noqa: E402
from arc3_wm.trm import config as C  # noqa: E402
from arc3_wm.trm.agents import TRMAgent  # noqa: E402
from arc3_wm.trm.config import AgentConfig  # noqa: E402
from arc3_wm.trm.core import EMAHelper  # noqa: E402
from arc3_wm.trm.data import WMTransitionDataset  # noqa: E402
from arc3_wm.trm.training import (  # noqa: E402
    TrainConfig,
    deep_supervision_batch,
    make_optimizer,
)
from arc3_wm.trm.world_model import TRMWorldModel  # noqa: E402

TINY = C.WorldModelConfig(
    core=C.TRMCoreConfig(
        d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
        n_supervision=1, halt_max_steps=1,
    ),
    tokenizer=C.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4),
)


class ConstWM:
    """Stub WM predicting a fixed fill everywhere."""

    def __init__(self, fill):
        self.fill = fill

    def predict(self, grids, actions, max_steps=None):
        import torch as t

        logits = t.zeros(grids.shape[0], 64, 64, 16)
        logits[..., self.fill] = 10.0

        class Out:
            next_logits = logits
            reward_logit = None
            state_logits = None
            change_logits = None

        return Out()


def test_ensemble_disagreement_bonus_prefers_uncertain():
    # Members agree everywhere -> no bonus; a second agent whose members
    # disagree fully gets w_disagree added uniformly. To make the bonus
    # decide an argmax, disable everything else and compare score paths.
    grid = np.zeros((64, 64), dtype=np.uint8)
    mask = build_mask([1, 2])

    class SplitWM(ConstWM):
        """Disagrees with member 0 only for action 1."""

        def predict(self, grids, actions, max_steps=None):
            import torch as t

            logits = t.zeros(grids.shape[0], 64, 64, 16)
            for i, a in enumerate(actions.tolist()):
                logits[i, :, :, 0 if a == 0 else 7] = 10.0

            class Out:
                next_logits = logits
                reward_logit = None
                state_logits = None
                change_logits = None

            return Out()

    cfg = AgentConfig(
        use_bc=False, use_wm=True, epsilon=0.0, w_change=0.0,
        w_novelty=0.0, w_disagree=1.0, seed=0,
    )
    agent = TRMAgent(cfg, world_model=[ConstWM(0), SplitWM(0)])
    # Action 0: both predict fill 0 (agree). Action 1: member0 fill 0,
    # member1 fill 7 (full disagreement) -> action 1 wins.
    assert agent.act(grid, mask) == 1


def test_single_model_ignores_disagree_weight():
    cfg = AgentConfig(use_bc=False, use_wm=True, epsilon=0.0,
                      w_change=0.0, w_disagree=5.0, seed=0)
    agent = TRMAgent(cfg, world_model=ConstWM(0))
    assert agent.act(np.zeros((64, 64), dtype=np.uint8), build_mask([1, 2])) in (0, 1)


@pytest.fixture()
def seq_cache(tmp_path):
    rng = np.random.default_rng(0)
    grids = rng.integers(0, 4, size=(6, 64, 64)).astype(np.uint8)
    path = tmp_path / "toy.npz"
    np.savez_compressed(
        path,
        grids=grids,
        actions=np.array([-1, 0, 1, 2, 3, 4], dtype=np.int16),
        levels=np.zeros(6, dtype=np.int16),
        states=np.zeros(6, dtype=np.int8),
        avail=np.ones((6, 7), dtype=np.uint8),
        episode_starts=np.array([0], dtype=np.int64),
    )
    return path


def test_two_step_dataset_shapes_and_boundaries(seq_cache):
    ds1 = WMTransitionDataset([seq_cache], dedup=False, n_steps=1)
    ds2 = WMTransitionDataset([seq_cache], dedup=False, n_steps=2)
    assert len(ds1) == 5
    assert len(ds2) == 4  # last transition has no successor
    sample = ds2[0]
    assert "action_2" in sample and sample["next_grid_2"].shape == (64, 64)
    with pytest.raises(ValueError):
        WMTransitionDataset([seq_cache], n_steps=3)


def test_unroll_loss_included_in_training(seq_cache):
    torch.manual_seed(0)
    model = TRMWorldModel(TINY)
    ds = WMTransitionDataset([seq_cache], dedup=False, n_steps=2)
    batch = torch.utils.data.default_collate([ds[i] for i in range(4)])
    cfg = TrainConfig(lr=1e-3, warmup_steps=1, unroll_weight=0.5)
    opt = make_optimizer(model, cfg)
    ema = EMAHelper()
    ema.register(model)
    parts, _ = deep_supervision_batch(model, batch, opt, ema, cfg, 0, mode="wm")
    assert "unroll" in parts and parts["unroll"] > 0


def test_rollout_probe_windows(seq_cache):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import trm_rollout_probe as R

    from arc3_wm.trm.data import _GameCache

    cache = _GameCache(seq_cache)
    rng = np.random.default_rng(0)
    assert len(R.val_windows(cache, 2, 100, rng)) == 4
    assert len(R.val_windows(cache, 4, 100, rng)) == 2
    assert len(R.val_windows(cache, 8, 100, rng)) == 0


def test_sample_mask_excludes_halted_from_loss():
    torch.manual_seed(0)
    model = TRMWorldModel(TINY)
    g = torch.randint(0, 4, (4, 64, 64))
    nxt = torch.randint(0, 4, (4, 64, 64))
    out = model(g, torch.tensor([0, 1, 2, 3]))
    full = model.loss(out, nxt, prev_grid=g)
    half = model.loss(out, nxt, prev_grid=g,
                      sample_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert not torch.isclose(full["grid"], half["grid"])
    zero = model.loss(out, nxt, prev_grid=g,
                      sample_mask=torch.tensor([0.0, 0.0, 0.0, 0.0]))
    assert float(zero["grid"]) == 0.0


def test_training_exits_when_all_halt(tmp_path):
    import numpy as np

    from arc3_wm.trm import config as C2
    from arc3_wm.trm.data import WMTransitionDataset

    cfg_model = C2.WorldModelConfig(
        core=C2.TRMCoreConfig(
            d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
            n_supervision=4, halt_max_steps=4, halt_exploration_prob=0.0,
            halt_bias_init=5.0,  # force immediate halting
        ),
        tokenizer=C2.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4),
    )
    rng = np.random.default_rng(0)
    path = tmp_path / "toy.npz"
    np.savez_compressed(
        path,
        grids=rng.integers(0, 4, size=(5, 64, 64)).astype(np.uint8),
        actions=np.array([-1, 0, 1, 2, 3], dtype=np.int16),
        levels=np.zeros(5, dtype=np.int16),
        states=np.zeros(5, dtype=np.int8),
        avail=np.ones((5, 7), dtype=np.uint8),
        episode_starts=np.array([0], dtype=np.int64),
    )
    ds = WMTransitionDataset([path], dedup=False)
    batch = torch.utils.data.default_collate([ds[i] for i in range(4)])
    torch.manual_seed(0)
    model = TRMWorldModel(cfg_model)
    cfg = TrainConfig(lr=1e-3, warmup_steps=1)
    opt = make_optimizer(model, cfg)
    ema = EMAHelper()
    ema.register(model)
    _, consumed = deep_supervision_batch(model, batch, opt, ema, cfg, 0, mode="wm")
    assert consumed == 1  # every sample halts at step 1 -> loop exits


def test_predict_freezes_per_sample_outputs():
    torch.manual_seed(0)
    cfg_model = C.WorldModelConfig(
        core=C.TRMCoreConfig(
            d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
            halt_max_steps=4, halt_bias_init=5.0,
        ),
        tokenizer=C.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4),
    )
    model = TRMWorldModel(cfg_model)
    g = torch.randint(0, 4, (2, 64, 64))
    out = model.predict(g, torch.tensor([0, 1]))
    # bias +5 -> all halt at step 1; frozen output equals a single step.
    single = model(g, torch.tensor([0, 1]))
    assert torch.allclose(out.next_logits, single.next_logits)
