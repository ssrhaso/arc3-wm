"""Spec for train_loop resume (requeued-job robustness)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arc3_wm.trm import config as C  # noqa: E402
from arc3_wm.trm.data import WMTransitionDataset  # noqa: E402
from arc3_wm.trm.training import TrainConfig, train_loop  # noqa: E402
from arc3_wm.trm.world_model import TRMWorldModel  # noqa: E402

TINY = C.WorldModelConfig(
    core=C.TRMCoreConfig(
        d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
        n_supervision=1, halt_max_steps=1,
    ),
    tokenizer=C.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4),
)


@pytest.fixture()
def cache(tmp_path):
    rng = np.random.default_rng(0)
    grids = rng.integers(0, 4, size=(8, 64, 64)).astype(np.uint8)
    path = tmp_path / "toy.npz"
    np.savez_compressed(
        path,
        grids=grids,
        actions=np.array([-1, 0, 1, 2, -1, 3, 4, 0], dtype=np.int16),
        levels=np.zeros(8, dtype=np.int16),
        states=np.zeros(8, dtype=np.int8),
        avail=np.ones((8, 7), dtype=np.uint8),
        episode_starts=np.array([0, 4], dtype=np.int64),
    )
    return path


def _cfg(epochs):
    return TrainConfig(
        lr=1e-3, warmup_steps=1, batch_size=4, epochs=epochs,
        num_workers=0, device="cpu", bf16=False,
    )


def test_resume_continues_from_latest(cache, tmp_path):
    out = tmp_path / "run"
    ds = WMTransitionDataset([cache], dedup=False)

    torch.manual_seed(0)
    r1 = train_loop(TRMWorldModel(TINY), ds, _cfg(1), C.to_dict(TINY), out, mode="wm")
    steps_1 = r1["steps"]
    assert steps_1 > 0

    # Resume with a higher epoch target: continues, does not restart.
    torch.manual_seed(0)
    r2 = train_loop(
        TRMWorldModel(TINY), ds, _cfg(2), C.to_dict(TINY), out, mode="wm", resume=True
    )
    assert r2["steps"] > steps_1
    lines = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    resumed = [l for l in lines if l.get("resumed")]
    assert resumed and resumed[0]["epoch"] == 1

    # Resuming at the same epoch target is a no-op (already complete).
    r3 = train_loop(
        TRMWorldModel(TINY), ds, _cfg(2), C.to_dict(TINY), out, mode="wm", resume=True
    )
    assert r3["steps"] == r2["steps"]
