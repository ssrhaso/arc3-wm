"""Spec for arc3_wm.trm.training and the trm_* script entry points."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arc3_wm.trm import config as C  # noqa: E402
from arc3_wm.trm.core import EMAHelper  # noqa: E402
from arc3_wm.trm.data import BCDataset, WMTransitionDataset  # noqa: E402
from arc3_wm.trm.policy import TRMPolicy  # noqa: E402
from arc3_wm.trm.training import (  # noqa: E402
    TrainConfig,
    deep_supervision_batch,
    evaluate_bc,
    evaluate_wm,
    load_policy,
    load_wm,
    make_optimizer,
    save_checkpoint,
    train_loop,
)
from arc3_wm.trm.world_model import TRMWorldModel  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

TINY_CORE = C.TRMCoreConfig(
    d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
    n_supervision=2, halt_max_steps=2,
)
TINY_TOK = C.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4)
WM_CFG = C.WorldModelConfig(core=TINY_CORE, tokenizer=TINY_TOK)
POL_CFG = C.PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK)


@pytest.fixture(scope="module")
def cache_npz(tmp_path_factory):
    """A tiny synthetic npz cache: 2 episodes, deterministic dynamics."""
    tmp = tmp_path_factory.mktemp("cache")
    rng = np.random.default_rng(0)
    grids, actions, levels, states, avail = [], [], [], [], []
    starts = []
    offset = 0
    for _ep in range(2):
        starts.append(offset)
        t_len = 6
        g = rng.integers(0, 4, size=(64, 64), dtype=np.uint8)
        for t in range(t_len):
            grids.append(g.copy())
            actions.append(-1 if t == 0 else int(rng.integers(0, 5)))
            levels.append(0 if t < t_len - 1 else 1)
            states.append(0 if t < t_len - 1 else 1)
            avail.append(np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.uint8))
            g = np.roll(g, 1, axis=1)  # deterministic drift
        offset += t_len
    path = tmp / "toy.npz"
    np.savez_compressed(
        path,
        grids=np.stack(grids),
        actions=np.asarray(actions, dtype=np.int16),
        levels=np.asarray(levels, dtype=np.int16),
        states=np.asarray(states, dtype=np.int8),
        avail=np.stack(avail),
        episode_starts=np.asarray(starts, dtype=np.int64),
    )
    return path


def test_deep_supervision_batch_updates_weights(cache_npz):
    torch.manual_seed(0)
    model = TRMWorldModel(WM_CFG)
    ds = WMTransitionDataset([cache_npz], dedup=False)
    batch = torch.utils.data.default_collate([ds[i] for i in range(4)])
    cfg = TrainConfig(lr=1e-3, warmup_steps=1)
    opt = make_optimizer(model, cfg)
    ema = EMAHelper(mu=0.9)
    ema.register(model)
    before = model.grid_head.decode.weight.detach().clone()
    parts, steps = deep_supervision_batch(
        model, batch, opt, ema, cfg, 0, mode="wm",
    )
    assert 1 <= steps <= TINY_CORE.n_supervision
    assert "grid" in parts and "halt" in parts and "loss" in parts
    # ACT observability: supervision steps consumed, halt rate, halt-decision
    # accuracy (the official q_halt_accuracy analogue).
    assert parts["sup_steps"] == float(steps)
    assert 0.0 <= parts["halt_rate"] <= 1.0
    assert 0.0 <= parts["q_halt_accuracy"] <= 1.0
    assert not torch.allclose(model.grid_head.decode.weight, before)


def test_train_loop_wm_end_to_end(cache_npz, tmp_path):
    torch.manual_seed(0)
    model = TRMWorldModel(WM_CFG)
    ds = WMTransitionDataset([cache_npz], dedup=False)
    cfg = TrainConfig(
        lr=1e-3, warmup_steps=1, batch_size=4, epochs=1,
        num_workers=0, device="cpu", bf16=False,
    )
    result = train_loop(
        model, ds, cfg, C.to_dict(WM_CFG), tmp_path, mode="wm",
        evaluate=lambda m: evaluate_wm(m, ds, max_batches=2),
    )
    assert result["steps"] > 0
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    loaded = load_wm(tmp_path / "latest.pt")
    assert isinstance(loaded, TRMWorldModel)


def test_train_loop_bc_end_to_end(cache_npz, tmp_path):
    torch.manual_seed(0)
    model = TRMPolicy(POL_CFG)
    ds = BCDataset([cache_npz])
    cfg = TrainConfig(
        lr=1e-3, warmup_steps=1, batch_size=4, epochs=1,
        num_workers=0, device="cpu", bf16=False,
    )
    result = train_loop(
        model, ds, cfg, C.to_dict(POL_CFG), tmp_path, mode="bc",
        evaluate=lambda m: evaluate_bc(m, ds, max_batches=2),
    )
    assert result["steps"] > 0
    loaded = load_policy(tmp_path / "latest.pt")
    assert isinstance(loaded, TRMPolicy)


def test_evaluate_wm_reports_copy_baseline(cache_npz):
    model = TRMWorldModel(WM_CFG)
    ds = WMTransitionDataset([cache_npz], dedup=False)
    metrics = evaluate_wm(model, ds, max_batches=2)
    for key in ("exact_match", "cell_acc", "copy_cell_acc", "changed_cell_acc", "changed_cell_frac"):
        assert key in metrics
    # The toy dynamics shift every column: copy should be far from perfect.
    assert metrics["copy_cell_acc"] < 0.9


def test_checkpoint_round_trip_preserves_ema(tmp_path):
    model = TRMWorldModel(WM_CFG)
    ema = EMAHelper(mu=0.5)
    ema.register(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    # EMA still near init; loading with use_ema=True must differ from raw.
    opt = make_optimizer(model, TrainConfig())
    save_checkpoint(tmp_path / "ck.pt", model, ema, opt, C.to_dict(WM_CFG), 1)
    raw = load_wm(tmp_path / "ck.pt", use_ema=False)
    smoothed = load_wm(tmp_path / "ck.pt", use_ema=True)
    p_raw = next(iter(raw.parameters()))
    p_ema = next(iter(smoothed.parameters()))
    assert not torch.allclose(p_raw, p_ema)


def test_wm_script_cli_smoke(cache_npz, tmp_path, monkeypatch):
    import importlib

    trm_train_wm = importlib.import_module("trm_train_wm")
    data_dir = cache_npz.parent
    (data_dir / "toygame.npz").write_bytes(cache_npz.read_bytes())
    rc = trm_train_wm.main(
        [
            "--data", str(data_dir), "--games", "toygame",
            "--out", str(tmp_path / "run"),
            "--epochs", "1", "--batch-size", "4", "--num-workers", "0",
            "--device", "cpu", "--no-bf16", "--warmup-steps", "1",
            "--d-model", "32", "--n-layers", "1", "--patch-size", "16",
            "--h-cycles", "1", "--l-cycles", "1", "--n-supervision", "1",
            "--halt-max-steps", "1", "--val-fraction", "0.5",
        ]
    )
    assert rc == 0
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["games"] == ["toygame"]
    assert (tmp_path / "run" / "latest.pt").exists()


def test_bc_script_cli_smoke(cache_npz, tmp_path):
    import importlib

    trm_train_bc = importlib.import_module("trm_train_bc")
    data_dir = cache_npz.parent
    (data_dir / "toygame2.npz").write_bytes(cache_npz.read_bytes())
    rc = trm_train_bc.main(
        [
            "--data", str(data_dir), "--games", "toygame2",
            "--out", str(tmp_path / "run"),
            "--epochs", "1", "--batch-size", "4", "--num-workers", "0",
            "--device", "cpu", "--no-bf16", "--warmup-steps", "1",
            "--d-model", "32", "--n-layers", "1", "--patch-size", "16",
            "--h-cycles", "1", "--l-cycles", "1", "--n-supervision", "1",
            "--halt-max-steps", "1", "--val-fraction", "0.5",
        ]
    )
    assert rc == 0
    assert (tmp_path / "run" / "best.pt").exists()
