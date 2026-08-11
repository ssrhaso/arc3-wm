"""Spec for TrainConfig.eval_every."""

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


def test_eval_every_skips_intermediate_epochs(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "toy.npz"
    np.savez_compressed(
        path,
        grids=rng.integers(0, 4, size=(6, 64, 64)).astype(np.uint8),
        actions=np.array([-1, 0, 1, 2, 3, 4], dtype=np.int16),
        levels=np.zeros(6, dtype=np.int16),
        states=np.zeros(6, dtype=np.int8),
        avail=np.ones((6, 7), dtype=np.uint8),
        episode_starts=np.array([0], dtype=np.int64),
    )
    ds = WMTransitionDataset([path], dedup=False)
    calls = []

    def fake_eval(model):
        calls.append(1)
        return {"exact_match": 0.0}

    cfg = TrainConfig(
        lr=1e-3, warmup_steps=1, batch_size=8, epochs=5, eval_every=2,
        num_workers=0, device="cpu", bf16=False,
    )
    train_loop(TRMWorldModel(TINY), ds, cfg, C.to_dict(TINY), tmp_path / "run", mode="wm",
               evaluate=fake_eval)
    # Epochs 2, 4 (every 2nd) and the final epoch 5 -> 3 calls.
    assert len(calls) == 3
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert sum(1 for l in lines if "val" in json.loads(l)) == 3


def test_resume_episode_count_drops_truncated_tail(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from trm_eval_agent import resume_episode_count

    sink = tmp_path / "eval_episodes.jsonl"
    good = '{"rewards": [0.0, 1.0], "terminal_state": "GAME_OVER"}'
    sink.write_text(good + "\n" + good + "\n" + '{"rewards": [0.0, 0.')
    assert resume_episode_count(sink) == 2
    assert sink.read_text() == good + "\n" + good + "\n"
    assert resume_episode_count(sink) == 2  # idempotent on a clean file
