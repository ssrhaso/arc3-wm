"""Tests for the numpy linear-probe fitter (scripts/probe_fit_probes.py).

CPU-only, no sklearn. Validates that the probe separates a genuinely-decodable
latent (signal > permutation control) from noise (signal ~ control), and that
the group split + degenerate cases behave.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "probe_fit_probes",
    Path(__file__).resolve().parent.parent / "scripts" / "probe_fit_probes.py",
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def _grouped(n_per_group, n_groups, dim, rng, signal):
    """Build (X, y, groups): label is per-group; X linearly encodes it iff signal."""
    ys, Xs, gs = [], [], []
    for g in range(n_groups):
        label = g % 3
        for _ in range(n_per_group):
            ys.append(label)
            gs.append(g)
            base = np.zeros(dim)
            if signal:
                base[label] = 5.0  # class-separating direction
            Xs.append(base + rng.normal(0, 0.3, dim))
    return np.array(Xs), np.array(ys), np.array(gs)


def test_probe_detects_decodable_signal():
    rng = np.random.default_rng(0)
    X, y, g = _grouped(20, 9, 8, rng, signal=True)
    r = pf.fit_linear_probe(X, y, g, seed=1, n_perm=5)
    assert r["status"] == "ok"
    assert r["balanced_acc"] > 0.8
    assert r["balanced_acc"] > r["perm_balanced_acc_mean"] + 0.2
    assert r["above_control"] is True


def test_probe_noise_is_at_control():
    rng = np.random.default_rng(1)
    X, y, g = _grouped(20, 9, 8, rng, signal=False)  # label not in X
    r = pf.fit_linear_probe(X, y, g, seed=2, n_perm=5)
    assert r["status"] == "ok"
    # No real signal: balanced acc should not clear the permutation control by much.
    assert r["above_control"] is False


def test_group_split_holds_out_whole_episodes():
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    tr, te = pf.group_split(groups, test_frac=0.5, seed=0)
    # every group is entirely in train or entirely in test (no leakage)
    for gid in np.unique(groups):
        m = groups == gid
        assert tr[m].all() or te[m].all()
    assert tr.sum() > 0 and te.sum() > 0


def test_balanced_accuracy_imbalance():
    # 90% class 0, 10% class 1; predicting all-0 => acc 0.9 but balanced 0.5
    y_true = np.array([0] * 9 + [1])
    y_pred = np.zeros(10, dtype=int)
    assert pf.balanced_accuracy(y_true, y_pred, np.array([0, 1])) == 0.5


def test_single_class_is_degenerate():
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.zeros(20, dtype=int)
    g = np.arange(20) // 5
    r = pf.fit_linear_probe(X, y, g)
    assert r["status"] == "degenerate_single_class"
