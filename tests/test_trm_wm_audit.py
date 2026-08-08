"""Spec for scripts/trm_wm_audit.py - novel-transition overfit audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import trm_wm_audit as A  # noqa: E402

from arc3_wm.trm import config as C  # noqa: E402
from arc3_wm.trm.core import EMAHelper  # noqa: E402
from arc3_wm.trm.data import _GameCache, train_val_split_episodes  # noqa: E402
from arc3_wm.trm.training import TrainConfig, make_optimizer, save_checkpoint  # noqa: E402
from arc3_wm.trm.world_model import TRMWorldModel  # noqa: E402

TINY = C.WorldModelConfig(
    core=C.TRMCoreConfig(
        d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
        n_supervision=1, halt_max_steps=1,
    ),
    tokenizer=C.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4),
)


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    grids = []
    # Episode 0 and 1 share their first transition input exactly (dup);
    # remaining frames are unique.
    shared = rng.integers(0, 4, size=(64, 64)).astype(np.uint8)
    for ep in range(2):
        g = shared.copy()
        grids.append(g)
        for _ in range(3):
            g = np.roll(g, 1 + ep, axis=1)  # ep1 diverges after frame 0
            grids.append(g.copy())
    actions = np.array([-1, 0, 1, 2, -1, 0, 3, 4], dtype=np.int16)
    path = tmp_path / "toyg.npz"
    np.savez_compressed(
        path,
        grids=np.stack(grids),
        actions=actions,
        levels=np.zeros(8, dtype=np.int16),
        states=np.zeros(8, dtype=np.int8),
        avail=np.ones((8, 7), dtype=np.uint8),
        episode_starts=np.array([0, 4], dtype=np.int64),
    )
    return tmp_path


def test_novel_indices_exclude_seen_inputs(data_dir):
    npz = data_dir / "toyg.npz"
    tr, va = train_val_split_episodes(npz, 0.5, seed=0)
    ct, cv = _GameCache(npz, episodes=tr), _GameCache(npz, episodes=va)
    novel, overlap = A.novel_val_indices(ct, cv)
    # Both episodes start from the same frame with action 0 (rows 1 and 5),
    # so exactly one val transition input is duplicated in train.
    assert 0.0 < overlap < 1.0
    assert len(novel) == len(cv.transition_idx) - 1


def test_audit_main_end_to_end(data_dir, tmp_path):
    torch.manual_seed(0)
    model = TRMWorldModel(TINY)
    ema = EMAHelper(model)
    opt = make_optimizer(model, TrainConfig())
    ckpt_dir = tmp_path / "run"
    save_checkpoint(ckpt_dir / "best.pt", model, ema, opt, C.to_dict(TINY), 1)
    (ckpt_dir / "metrics.jsonl").write_text(
        json.dumps({"epoch": 4, "step": 1, "val": {"exact_match": 0.5}}) + "\n"
        + json.dumps({"epoch": 9, "step": 2, "val": {"exact_match": 0.1}}) + "\n"
    )
    rc = A.main([
        "--data", str(data_dir), "--game", "toyg",
        "--ckpt", str(ckpt_dir / "best.pt"),
        "--val-fraction", "0.5", "--device", "cpu",
    ])
    assert rc == 0


def test_best_epoch_report_flags_late_decline(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(
        json.dumps({"epoch": 4, "step": 1, "val": {"exact_match": 0.5}}) + "\n"
        + json.dumps({"epoch": 9, "step": 2, "val": {"exact_match": 0.1}}) + "\n"
    )
    rep = A.best_epoch_report(p)
    assert rep["best_epoch"] == 4 and rep["late_decline"] is True
    p2 = tmp_path / "m2.jsonl"
    p2.write_text(
        json.dumps({"epoch": 4, "step": 1, "val": {"exact_match": 0.2}}) + "\n"
        + json.dumps({"epoch": 9, "step": 2, "val": {"exact_match": 0.5}}) + "\n"
    )
    assert A.best_epoch_report(p2)["late_decline"] is False
